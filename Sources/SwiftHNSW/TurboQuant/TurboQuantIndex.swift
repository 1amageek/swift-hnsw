#if canImport(FoundationEssentials)
import FoundationEssentials
#elseif canImport(Foundation)
import Foundation
#endif
import Synchronization

/// HNSW search over inner-product optimized TurboQuant packed vectors.
///
/// With the mean-squared-error objective, each normalized vector stores a `bitWidth`-bit
/// MSE code. With the inner-product objective, it stores a `(bitWidth - 1)`-bit MSE code,
/// one QJL sign bit per original coordinate, and the residual norm required by
/// `TurboQuant_prod`.
/// The temporary Float32 representation is used only while constructing the HNSW topology.
/// `finalize()` captures that topology and releases the Float32 graph vectors; all subsequent
/// search uses the selected packed estimator throughout graph traversal and final ordering
/// without reconstructing stored vectors.
public final class TurboQuantIndex: Sendable {

    private struct Graph: Sendable {
        var labelOrder: [UInt64]
        var deletedFlags: [UInt8]
        var levels: [Int]
        var connections: HNSWConnectionStore
        var entryPoint: HNSWInternalID?
        var maxLevel: Int
    }

    private struct State: Sendable {
        var finalized: Bool
        var efSearch: Int
        var builder: HNSWIndex<Float>?
        var codeOffsetsByLabel: [UInt64: Int]
        var stagingPackedStorage: [UInt8]
        var recordWorkspace: [UInt8]
        var packedStorage: [UInt8]
        var graph: Graph?
        var normalizedWorkspace: [Float]
        var rotationWorkspace: [Float]
        var rotatedWorkspace: [Float]
        var reconstructedWorkspace: [Float]
        var projectedWorkspace: [Float]
        var queryWorkspace: [Float]
        var queryDistanceTable: [Float]
        var queryPackedDistanceTable: [Float]
        var queryQJLDistanceTable: [Float]
        var visited: [UInt16]
        var visitedTag: UInt16
        var searchWorkspace: HNSWSearchWorkspace
    }

    public let dimensions: Int
    public let bitWidth: Int
    public let mseBitWidth: Int
    public let objective: TurboQuantObjective
    public let rotationStrategy: TurboQuantRotationStrategy
    public let configuration: HNSWConfiguration
    public let seed: UInt64
    public let paddedDimensions: Int

    private let quantizer: ScalarQuantizer
    private let rotation: TurboQuantRotation
    private let qjlProjection: QJLProjection?
    private let msePackedSize: Int
    private let qjlPackedSize: Int
    private let _packedSize: Int
    private let usesAcceleratedFourBitKernel: Bool
    private let usesQJLDistancePruning: Bool
    private let qjlScale: Float
    private let maximumResidualNorm: Float
    private let maximumElementCount: Int
    private let state: Mutex<State>
    private static let maximumSerializedLevelCount = 64
    private static let maximumRetainedWorkspaceBytes = 256 * 1_024 * 1_024
    private static let maximumPackedLookupBytes = 64 * 1_024 * 1_024

    public convenience init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int = 4,
        objective: TurboQuantObjective = .meanSquaredError,
        rotationStrategy: TurboQuantRotationStrategy = .structuredHadamard,
        configuration: HNSWConfiguration = .balanced,
        seed: UInt64 = 42
    ) throws {
        try self.init(
            dimensions: dimensions,
            maxElements: maxElements,
            bitWidth: bitWidth,
            objective: objective,
            rotationStrategy: rotationStrategy,
            configuration: configuration,
            seed: seed,
            acceleratedFourBitKernelAvailable:
                ScalarQuantizer.hasAcceleratedFourBitKernel,
            qjlDistancePruningEnabled: true,
            createsBuilder: true
        )
    }

    convenience init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int,
        objective: TurboQuantObjective,
        rotationStrategy: TurboQuantRotationStrategy,
        configuration: HNSWConfiguration,
        seed: UInt64,
        acceleratedFourBitKernelAvailable: Bool,
        qjlDistancePruningEnabled: Bool = true
    ) throws {
        try self.init(
            dimensions: dimensions,
            maxElements: maxElements,
            bitWidth: bitWidth,
            objective: objective,
            rotationStrategy: rotationStrategy,
            configuration: configuration,
            seed: seed,
            acceleratedFourBitKernelAvailable:
                acceleratedFourBitKernelAvailable,
            qjlDistancePruningEnabled: qjlDistancePruningEnabled,
            createsBuilder: true
        )
    }

    private init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int,
        objective: TurboQuantObjective,
        rotationStrategy: TurboQuantRotationStrategy,
        configuration: HNSWConfiguration,
        seed: UInt64,
        acceleratedFourBitKernelAvailable: Bool,
        qjlDistancePruningEnabled: Bool,
        createsBuilder: Bool
    ) throws {
        guard dimensions > 0 else {
            throw HNSWError.initializationFailed("dimensions must be positive")
        }
        guard dimensions < Int.max,
              UInt64(dimensions) <= UInt64(UInt32.max) else {
            throw HNSWError.initializationFailed(
                "dimensions must fit the TurboQuant archive format"
            )
        }
        guard maxElements > 0,
              UInt64(maxElements) <= UInt64(UInt32.max) else {
            throw HNSWError.initializationFailed("maxElements must be positive and fit the HNSW internal id")
        }
        let mseBitWidth = objective == .innerProduct ? bitWidth - 1 : bitWidth
        guard (0...4).contains(mseBitWidth), bitWidth > 0 else {
            let supported = objective == .innerProduct ? "1 through 5" : "1 through 4"
            throw HNSWError.initializationFailed(
                "bitWidth must be \(supported) for the selected TurboQuant objective"
            )
        }
        guard (2...HNSWConfiguration.maximumM).contains(configuration.m) else {
            throw HNSWError.invalidArgument("m must be between 2 and 10000")
        }
        guard configuration.efConstruction >= configuration.m else {
            throw HNSWError.invalidArgument("efConstruction must be at least m")
        }
        guard configuration.efSearch > 0 else {
            throw HNSWError.invalidArgument("efSearch must be positive")
        }
        guard UInt64(configuration.m) <= UInt64(UInt32.max),
              UInt64(configuration.efConstruction) <= UInt64(UInt32.max),
              UInt64(configuration.efSearch) <= UInt64(UInt32.max) else {
            throw HNSWError.invalidArgument("HNSW configuration values must fit the archive format")
        }

        let padded: Int
        if rotationStrategy == .haar {
            padded = dimensions
            let maximumHaarFloatCount = 256 * 1_024 * 1_024
                / MemoryLayout<Float>.stride
            let firstFactor = dimensions.isMultiple(of: 2)
                ? dimensions / 2
                : dimensions
            let secondFactor = dimensions.isMultiple(of: 2)
                ? dimensions + 1
                : (dimensions + 1) / 2
            guard dimensions <= maximumHaarFloatCount,
                  firstFactor <= (
                    maximumHaarFloatCount - dimensions + 1
                  ) / secondFactor else {
                throw HNSWError.initializationFailed(
                    "Haar rotation storage exceeds the 256 MiB limit"
                )
            }
        } else {
            var nextPowerOfTwo = 1
            while nextPowerOfTwo < dimensions {
                guard nextPowerOfTwo <= Int.max / 2 else {
                    throw HNSWError.initializationFailed(
                        "padded dimension overflows Int"
                    )
                }
                nextPowerOfTwo *= 2
            }
            padded = nextPowerOfTwo
        }
        guard UInt64(padded) <= UInt64(UInt32.max) else {
            throw HNSWError.initializationFailed("dimensions must fit the TurboQuant archive format")
        }

        self.dimensions = dimensions
        self.bitWidth = bitWidth
        self.mseBitWidth = mseBitWidth
        self.objective = objective
        self.rotationStrategy = rotationStrategy
        self.configuration = configuration
        self.seed = seed
        self.paddedDimensions = padded
        guard padded <= Int.max / (1 << mseBitWidth) else {
            throw HNSWError.initializationFailed("distance table size overflows Int")
        }
        let msePackedSize = BitPacking.packedSize(count: padded, bitWidth: mseBitWidth)
        let qjlProjection: QJLProjection?
        if objective == .innerProduct {
            qjlProjection = try QJLProjection(
                dimension: dimensions,
                seed: seed ^ 0xD1B5_4A32_D192_ED03
            )
        } else {
            qjlProjection = nil
        }
        let qjlPackedSize = qjlProjection?.packedSize ?? 0
        let residualNormSize = objective == .innerProduct ? MemoryLayout<Float>.size : 0
        guard msePackedSize <= Int.max - qjlPackedSize - residualNormSize else {
            throw HNSWError.initializationFailed("TurboQuant record size overflows Int")
        }
        let packedSize = msePackedSize + qjlPackedSize + residualNormSize
        let usesAcceleratedFourBitKernel = mseBitWidth == 4
            && acceleratedFourBitKernelAvailable
        let supportsPackedLookup = mseBitWidth == 1
            || mseBitWidth == 2
            || (
                mseBitWidth == 4
                    && !usesAcceleratedFourBitKernel
                    && padded >= 256
            )
        guard UInt64(packedSize) <= UInt64(UInt32.max) else {
            throw HNSWError.initializationFailed("packed code size must fit the TurboQuant archive format")
        }
        guard !supportsPackedLookup || msePackedSize <= Int.max / 256 else {
            throw HNSWError.initializationFailed("packed distance table size overflows Int")
        }
        guard qjlPackedSize == 0
                || qjlPackedSize <= Int.max / QJLProjection.lookupEntriesPerByte else {
            throw HNSWError.initializationFailed("QJL distance table size overflows Int")
        }
        let distanceTableCount = mseBitWidth == 0 || usesAcceleratedFourBitKernel
            ? 0
            : padded * (1 << mseBitWidth)
        let qjlTableCount = qjlProjection?.lookupTableCount ?? 0
        let rotationWorkspaceCount = mseBitWidth == 0 ? 0 : padded
        let queryWorkspaceCount = mseBitWidth == 0 ? 0 : padded
        let workspaceCounts = [
            dimensions,
            rotationWorkspaceCount,
            padded,
            objective == .innerProduct ? dimensions : 0,
            objective == .innerProduct ? dimensions : 0,
            queryWorkspaceCount,
            distanceTableCount,
            qjlTableCount,
        ]
        var retainedWorkspaceFloatCount = 0
        for count in workspaceCounts {
            guard count <= Int.max - retainedWorkspaceFloatCount else {
                throw HNSWError.initializationFailed(
                    "TurboQuant retained workspace size overflows Int"
                )
            }
            retainedWorkspaceFloatCount += count
        }
        guard retainedWorkspaceFloatCount
                <= Self.maximumRetainedWorkspaceBytes / MemoryLayout<Float>.stride else {
            throw HNSWError.initializationFailed(
                "TurboQuant retained workspace exceeds the 256 MiB limit"
            )
        }
        let availableLookupBytes = min(
            Self.maximumPackedLookupBytes,
            Self.maximumRetainedWorkspaceBytes
                - retainedWorkspaceFloatCount * MemoryLayout<Float>.stride
        )
        let packedLookupCount = supportsPackedLookup
            && msePackedSize <= availableLookupBytes
                / (256 * MemoryLayout<Float>.stride)
            ? msePackedSize * 256
            : 0
        self.quantizer = ScalarQuantizer(
            bitWidth: mseBitWidth,
            dimension: padded,
            usesExactSphericalDistribution: rotationStrategy == .haar
        )
        self.rotation = TurboQuantRotation(
            dimensions: dimensions,
            paddedDimensions: padded,
            strategy: rotationStrategy,
            seed: seed
        )
        self.qjlProjection = qjlProjection
        self.msePackedSize = msePackedSize
        self.qjlPackedSize = qjlPackedSize
        self._packedSize = packedSize
        self.usesAcceleratedFourBitKernel = usesAcceleratedFourBitKernel
        self.usesQJLDistancePruning = qjlDistancePruningEnabled
        self.qjlScale = objective == .innerProduct
            ? Float(1.253_314_137_315_500_3) / Float(dimensions)
            : 0
        let maximumCentroidMagnitude = quantizer.centroids.reduce(Float.zero) {
            max($0, abs($1))
        }
        self.maximumResidualNorm = (
            1 + Float(padded).squareRoot() * maximumCentroidMagnitude
        ) * (1 + 64 * Float.ulpOfOne)
        self.maximumElementCount = maxElements

        let builder: HNSWIndex<Float>?
        if createsBuilder {
            builder = try HNSWIndex<Float>(
                dimensions: padded,
                maxElements: maxElements,
                metric: .l2,
                configuration: configuration
            )
        } else {
            builder = nil
        }
        self.state = Mutex(State(
            finalized: !createsBuilder,
            efSearch: configuration.efSearch,
            builder: builder,
            codeOffsetsByLabel: [:],
            stagingPackedStorage: [],
            recordWorkspace: [UInt8](repeating: 0, count: packedSize),
            packedStorage: [],
            graph: nil,
            normalizedWorkspace: [Float](repeating: 0, count: dimensions),
            rotationWorkspace: [Float](repeating: 0, count: rotationWorkspaceCount),
            rotatedWorkspace: [Float](repeating: 0, count: padded),
            reconstructedWorkspace: objective == .innerProduct
                ? [Float](repeating: 0, count: dimensions)
                : [],
            projectedWorkspace: objective == .innerProduct
                ? [Float](repeating: 0, count: dimensions)
                : [],
            queryWorkspace: [Float](repeating: 0, count: queryWorkspaceCount),
            queryDistanceTable: [Float](repeating: 0, count: distanceTableCount),
            queryPackedDistanceTable: [Float](repeating: 0, count: packedLookupCount),
            queryQJLDistanceTable: qjlPackedSize == 0
                ? []
                : [Float](repeating: 0, count: qjlTableCount),
            visited: [],
            visitedTag: 0,
            searchWorkspace: HNSWSearchWorkspace()
        ))
    }

    public var count: Int {
        state.withLock { state in
            state.graph?.labelOrder.count ?? state.codeOffsetsByLabel.count
        }
    }

    public var capacity: Int { maximumElementCount }
    public var isEmpty: Bool { count == 0 }
    public var isFinalized: Bool { state.withLock { $0.finalized } }
    public var bytesPerVector: Int { _packedSize }
    public var projectionBytes: Int { qjlProjection?.retainedBytes ?? 0 }
    public var compressionRatio: Float {
        Float(dimensions * MemoryLayout<Float>.size) / Float(_packedSize)
    }

    public func setEfSearch(_ ef: Int) throws {
        guard ef > 0 else {
            throw HNSWError.invalidArgument("efSearch must be positive")
        }
        guard UInt64(ef) <= UInt64(UInt32.max) else {
            throw HNSWError.invalidArgument("efSearch must fit the archive format")
        }
        state.withLock {
            $0.efSearch = ef
        }
    }

    /// Finalizes graph construction and releases the temporary Float32 graph vectors.
    public func finalize() throws {
        try state.withLock { state in
            try finalizeLocked(state: &state)
        }
    }

    public func add(_ vector: [Float], label: UInt64) throws {
        try vector.withUnsafeBufferPointer { buffer in
            try add(buffer, label: label)
        }
    }

    /// Adds a borrowed vector without materializing an intermediate input array.
    public func add(_ vector: UnsafeBufferPointer<Float>, label: UInt64) throws {
        guard vector.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vector.count)
        }
        try Self.validateVector(vector, name: "vector")

        try state.withLock { state in
            guard !state.finalized else {
                throw HNSWError.addPointFailed("Cannot add after finalize()")
            }
            guard state.codeOffsetsByLabel[label] != nil
                    || state.codeOffsetsByLabel.count < maximumElementCount else {
                throw HNSWError.capacityExceeded(
                    current: state.codeOffsetsByLabel.count,
                    maximum: maximumElementCount
                )
            }
            guard let builder = state.builder else {
                throw HNSWError.indexNotInitialized
            }

            try state.normalizedWorkspace.withUnsafeMutableBufferPointer { normalized in
                try state.rotationWorkspace.withUnsafeMutableBufferPointer { rotationWork in
                    try state.rotatedWorkspace.withUnsafeMutableBufferPointer { rotated in
                        Self.normalizeValidatedVector(vector, into: normalized)
                        if mseBitWidth == 0 {
                            for index in 0..<dimensions {
                                rotated[index] = normalized[index]
                            }
                            for index in dimensions..<paddedDimensions {
                                rotated[index] = 0
                            }
                        } else {
                            rotation.rotateNormalized(
                                UnsafeBufferPointer(start: normalized.baseAddress, count: normalized.count),
                                into: rotated,
                                work: rotationWork
                            )
                        }
                        try builder.add(
                            UnsafeBufferPointer(start: rotated.baseAddress, count: rotated.count),
                            label: label
                        )
                        state.recordWorkspace.withUnsafeMutableBufferPointer { recordBuffer in
                            for index in recordBuffer.indices {
                                recordBuffer[index] = 0
                            }
                            let mseCode = UnsafeMutableBufferPointer(
                                start: recordBuffer.baseAddress,
                                count: msePackedSize
                            )
                            quantizer.quantizeAndPack(
                                UnsafeBufferPointer(
                                    start: rotated.baseAddress,
                                    count: rotated.count
                                ),
                                into: mseCode
                            )
                            if let qjlProjection {
                                state.reconstructedWorkspace.withUnsafeMutableBufferPointer { residual in
                                    var residualSquaredNorm: Float = 0
                                    if mseBitWidth == 0 {
                                        for index in 0..<dimensions {
                                            let value = normalized[index]
                                            residual[index] = value
                                            residualSquaredNorm += value * value
                                        }
                                    } else {
                                        quantizer.dequantize(
                                            UnsafeBufferPointer(
                                                start: mseCode.baseAddress,
                                                count: mseCode.count
                                            ),
                                            into: rotated
                                        )
                                        rotation.inverseRotate(
                                            UnsafeBufferPointer(
                                                start: rotated.baseAddress,
                                                count: rotated.count
                                            ),
                                            into: residual,
                                            work: rotationWork
                                        )
                                        for index in 0..<dimensions {
                                            let value = normalized[index] - residual[index]
                                            residual[index] = value
                                            residualSquaredNorm += value * value
                                        }
                                    }
                                    state.projectedWorkspace.withUnsafeMutableBufferPointer { projected in
                                        qjlProjection.project(
                                            UnsafeBufferPointer(
                                                start: residual.baseAddress,
                                                count: residual.count
                                            ),
                                            into: projected
                                        )
                                        let signs = UnsafeMutableBufferPointer(
                                            start: recordBuffer.baseAddress! + msePackedSize,
                                            count: qjlPackedSize
                                        )
                                        qjlProjection.packSigns(
                                            UnsafeBufferPointer(
                                                start: projected.baseAddress,
                                                count: projected.count
                                            ),
                                            into: signs
                                        )
                                    }
                                    Self.storeResidualNorm(
                                        residualSquaredNorm.squareRoot(),
                                        in: recordBuffer,
                                        at: msePackedSize + qjlPackedSize
                                    )
                                }
                            }
                        }
                        if let offset = state.codeOffsetsByLabel[label] {
                            state.stagingPackedStorage.replaceSubrange(
                                offset..<(offset + _packedSize),
                                with: state.recordWorkspace
                            )
                        } else {
                            let offset = state.stagingPackedStorage.count
                            state.stagingPackedStorage.append(
                                contentsOf: state.recordWorkspace
                            )
                            state.codeOffsetsByLabel[label] = offset
                        }
                    }
                }
            }
        }
    }

    public func search(_ query: [Float], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Searches the finalized graph with the selected distortion objective.
    public func search(_ query: UnsafeBufferPointer<Float>, k: Int) throws -> [SearchResult] {
        guard query.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: query.count)
        }
        guard k > 0 else {
            throw HNSWError.invalidArgument("k must be positive")
        }
        try Self.validateVector(query, name: "query")

        return try state.withLock { state in
            try finalizeLocked(state: &state)
            guard let graph = state.graph, let entryPoint = graph.entryPoint else {
                return []
            }
            let nodeCount = graph.labelOrder.count
            guard nodeCount > 0 else { return [] }
            try HNSWSearchWorkspace.validateRetainedCapacity(
                candidateCapacity: nodeCount,
                nearestCapacity: min(max(state.efSearch, k), nodeCount),
                resultCapacity: min(k, nodeCount),
                visitedCapacity: nodeCount
            )

            return state.normalizedWorkspace.withUnsafeMutableBufferPointer { normalized in
                state.rotationWorkspace.withUnsafeMutableBufferPointer { work in
                    state.queryWorkspace.withUnsafeMutableBufferPointer { rotatedQuery in
                        Self.normalizeValidatedVector(query, into: normalized)
                        if mseBitWidth > 0 {
                            rotation.rotateNormalized(
                                UnsafeBufferPointer(start: normalized.baseAddress, count: normalized.count),
                                into: rotatedQuery,
                                work: work
                            )
                        }
                        return state.queryDistanceTable.withUnsafeMutableBufferPointer { distanceTable in
                            if mseBitWidth > 0, !usesAcceleratedFourBitKernel {
                                quantizer.fillInnerProductTable(
                                    for: UnsafeBufferPointer(start: rotatedQuery.baseAddress, count: rotatedQuery.count),
                                    into: distanceTable
                                )
                            }
                            return state.queryPackedDistanceTable.withUnsafeMutableBufferPointer { packedDistanceTable in
                                let usesPackedLookup = !packedDistanceTable.isEmpty
                                    && (mseBitWidth != 4 || state.efSearch >= 32)
                                if usesPackedLookup {
                                    quantizer.fillPackedInnerProductTable(
                                        from: UnsafeBufferPointer(start: distanceTable.baseAddress, count: distanceTable.count),
                                        into: packedDistanceTable
                                    )
                                }
                                let packedLookup = usesPackedLookup
                                    ? UnsafeBufferPointer(
                                        start: packedDistanceTable.baseAddress,
                                        count: packedDistanceTable.count
                                    )
                                    : UnsafeBufferPointer<Float>(
                                        start: nil,
                                        count: 0
                                    )
                                return state.packedStorage.withUnsafeBufferPointer { packedStorage in
                                    guard let qjlProjection else {
                                        return self.searchGraph(
                                            k: k,
                                            entryPoint: entryPoint,
                                            graph: graph,
                                            packedStorage: packedStorage,
                                            mseQuery: UnsafeBufferPointer(start: rotatedQuery.baseAddress, count: rotatedQuery.count),
                                            distanceTable: UnsafeBufferPointer(start: distanceTable.baseAddress, count: distanceTable.count),
                                            packedDistanceTable: packedLookup,
                                            qjlDistanceTable: UnsafeBufferPointer(start: nil, count: 0),
                                            qjlAbsoluteSum: 0,
                                            efSearch: state.efSearch,
                                            visited: &state.visited,
                                            visitedTag: &state.visitedTag,
                                            searchWorkspace: &state.searchWorkspace
                                        )
                                    }
                                    return state.projectedWorkspace.withUnsafeMutableBufferPointer { projectedQuery in
                                        qjlProjection.project(
                                            UnsafeBufferPointer(
                                                start: normalized.baseAddress,
                                                count: normalized.count
                                            ),
                                            into: projectedQuery
                                        )
                                        return state.queryQJLDistanceTable.withUnsafeMutableBufferPointer { qjlTable in
                                            let qjlAbsoluteSum = qjlProjection.fillInnerProductTable(
                                                for: UnsafeBufferPointer(
                                                    start: projectedQuery.baseAddress,
                                                    count: projectedQuery.count
                                                ),
                                                into: qjlTable
                                            )
                                            return self.searchGraph(
                                                k: k,
                                                entryPoint: entryPoint,
                                                graph: graph,
                                                packedStorage: packedStorage,
                                                mseQuery: UnsafeBufferPointer(start: rotatedQuery.baseAddress, count: rotatedQuery.count),
                                                distanceTable: UnsafeBufferPointer(start: distanceTable.baseAddress, count: distanceTable.count),
                                                packedDistanceTable: packedLookup,
                                                qjlDistanceTable: UnsafeBufferPointer(start: qjlTable.baseAddress, count: qjlTable.count),
                                                qjlAbsoluteSum: qjlAbsoluteSum,
                                                efSearch: state.efSearch,
                                                visited: &state.visited,
                                                visitedTag: &state.visitedTag,
                                                searchWorkspace: &state.searchWorkspace
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @discardableResult
    public func addBatch(_ vectors: [Float], labels: [UInt64]) throws -> Int {
        guard labels.count <= Int.max / dimensions else {
            throw HNSWError.invalidArgument("batch dimensions exceed the supported integer range")
        }
        let expectedVectorCount = labels.count * dimensions
        guard vectors.count == expectedVectorCount else {
            throw HNSWError.dimensionMismatch(expected: expectedVectorCount, got: vectors.count)
        }
        var added = 0
        try vectors.withUnsafeBufferPointer { vectorBuffer in
            try labels.withUnsafeBufferPointer { labelBuffer in
                for index in 0..<labelBuffer.count {
                    let vector = UnsafeBufferPointer(
                        start: vectorBuffer.baseAddress! + index * dimensions,
                        count: dimensions
                    )
                    try add(vector, label: labelBuffer[index])
                    added += 1
                }
            }
        }
        return added
    }

    public func searchBatch(_ queries: [Float], numQueries: Int, k: Int) throws -> [[SearchResult]] {
        guard numQueries >= 0 else {
            throw HNSWError.invalidArgument("numQueries must not be negative")
        }
        guard numQueries <= Int.max / dimensions else {
            throw HNSWError.invalidArgument("batch dimensions exceed the supported integer range")
        }
        let expectedQueryCount = numQueries * dimensions
        guard queries.count == expectedQueryCount else {
            throw HNSWError.dimensionMismatch(expected: expectedQueryCount, got: queries.count)
        }
        guard k > 0 else {
            throw HNSWError.invalidArgument("k must be positive")
        }
        if numQueries == 0 { return [] }
        var output: [[SearchResult]] = []
        output.reserveCapacity(numQueries)
        try queries.withUnsafeBufferPointer { queryBuffer in
            for index in 0..<numQueries {
                let query = UnsafeBufferPointer(
                    start: queryBuffer.baseAddress! + index * dimensions,
                    count: dimensions
                )
                output.append(try search(query, k: k))
            }
        }
        return output
    }

    private func finalizeLocked(state: inout State) throws {
        guard !state.finalized else { return }
        guard let builder = state.builder else {
            throw HNSWError.indexNotInitialized
        }
        let snapshot = builder.graphSnapshot()
        guard snapshot.labelOrder.count == state.codeOffsetsByLabel.count else {
            throw HNSWError.serializationFailed("HNSW graph and TurboQuant code counts differ")
        }
        guard snapshot.labelOrder.count <= Int.max / _packedSize else {
            throw HNSWError.serializationFailed(
                "TurboQuant packed storage exceeds the addressable range"
            )
        }

        var canReuseStagingStorage = true
        for (index, label) in snapshot.labelOrder.enumerated() {
            guard let offset = state.codeOffsetsByLabel[label],
                  state.stagingPackedStorage.count >= _packedSize,
                  offset <= state.stagingPackedStorage.count - _packedSize else {
                throw HNSWError.serializationFailed(
                    "Missing TurboQuant code for label \(label)"
                )
            }
            canReuseStagingStorage = canReuseStagingStorage
                && offset == index * _packedSize
        }

        let packed: [UInt8]
        if canReuseStagingStorage {
            packed = state.stagingPackedStorage
        } else {
            var reordered = [UInt8]()
            reordered.reserveCapacity(snapshot.labelOrder.count * _packedSize)
            for label in snapshot.labelOrder {
                guard let offset = state.codeOffsetsByLabel[label] else {
                    throw HNSWError.serializationFailed(
                        "Missing TurboQuant code for label \(label)"
                    )
                }
                reordered.append(
                    contentsOf: state.stagingPackedStorage[
                        offset..<(offset + _packedSize)
                    ]
                )
            }
            packed = reordered
        }

        state.packedStorage = packed
        state.graph = Graph(
            labelOrder: snapshot.labelOrder,
            deletedFlags: snapshot.deletedFlags,
            levels: snapshot.levels,
            connections: snapshot.connections,
            entryPoint: snapshot.entryPoint,
            maxLevel: snapshot.maxLevel
        )
        state.visited = [UInt16](repeating: 0, count: snapshot.labelOrder.count)
        state.visitedTag = 0
        try HNSWSearchWorkspace.validateRetainedCapacity(
            candidateCapacity: snapshot.labelOrder.count,
            nearestCapacity: min(state.efSearch, snapshot.labelOrder.count),
            resultCapacity: 1,
            visitedCapacity: snapshot.labelOrder.count
        )
        state.searchWorkspace.prepare(
            candidateCapacity: snapshot.labelOrder.count,
            nearestCapacity: min(state.efSearch, snapshot.labelOrder.count),
            resultCapacity: 1,
            visitedCapacity: snapshot.labelOrder.count
        )
        state.builder = nil
        state.codeOffsetsByLabel.removeAll(keepingCapacity: false)
        state.stagingPackedStorage.removeAll(keepingCapacity: false)
        state.recordWorkspace.removeAll(keepingCapacity: false)
        state.finalized = true
    }

    private func searchGraph(
        k: Int,
        entryPoint: HNSWInternalID,
        graph: Graph,
        packedStorage: UnsafeBufferPointer<UInt8>,
        mseQuery: UnsafeBufferPointer<Float>,
        distanceTable: UnsafeBufferPointer<Float>,
        packedDistanceTable: UnsafeBufferPointer<Float>,
        qjlDistanceTable: UnsafeBufferPointer<Float>,
        qjlAbsoluteSum: Float,
        efSearch: Int,
        visited: inout [UInt16],
        visitedTag: inout UInt16,
        searchWorkspace: inout HNSWSearchWorkspace
    ) -> [SearchResult] {
        var current = entryPoint
        guard let initialDistance = searchDistance(
            to: current,
            packedStorage: packedStorage,
            mseQuery: mseQuery,
            distanceTable: distanceTable,
            packedDistanceTable: packedDistanceTable,
            qjlDistanceTable: qjlDistanceTable,
            qjlAbsoluteSum: qjlAbsoluteSum,
            lowerBound: nil
        ) else {
            return []
        }
        var currentDistance = initialDistance

        if graph.maxLevel > 0 {
            for level in stride(from: graph.maxLevel, through: 1, by: -1) {
                var changed = true
                while changed {
                    changed = false
                    let neighborRange = graph.connections.neighborStorageRange(
                        for: current,
                        at: level
                    )
                    guard !neighborRange.isEmpty else { break }
                    for neighborStorageIndex in neighborRange {
                        let neighbor = graph.connections.neighborInStorage(
                            at: neighborStorageIndex,
                            level: level
                        )
                        guard let candidateDistance = searchDistance(
                            to: neighbor,
                            packedStorage: packedStorage,
                            mseQuery: mseQuery,
                            distanceTable: distanceTable,
                            packedDistanceTable: packedDistanceTable,
                            qjlDistanceTable: qjlDistanceTable,
                            qjlAbsoluteSum: qjlAbsoluteSum,
                            lowerBound: currentDistance
                        ) else {
                            continue
                        }
                        let candidate = HNSWNeighborCandidate(
                            internalID: neighbor,
                            distance: candidateDistance
                        )
                        if isCloserHNSWCandidate(candidate, than: HNSWNeighborCandidate(
                            internalID: current,
                            distance: currentDistance
                        )) {
                            current = neighbor
                            currentDistance = candidate.distance
                            changed = true
                        }
                    }
                }
            }
        }

        let nodeCount = graph.labelOrder.count
        let effectiveEF = max(efSearch, k)
        let candidateCapacity = max(1, nodeCount)
        let nearestCapacity = max(1, min(effectiveEF, nodeCount))
        let resultCapacity = max(1, min(k, nodeCount))
        return searchWorkspace.withCandidateBuffers(
            candidateCapacity: candidateCapacity,
            nearestCapacity: nearestCapacity,
            resultCapacity: resultCapacity,
            visitedCapacity: nodeCount
        ) { candidateStorage, nearestStorage, resultStorage in
            visited.withUnsafeMutableBufferPointer { visitedBuffer in
                let tag = nextVisitedTag(visited: visitedBuffer, visitedTag: &visitedTag)
                var candidates = HNSWBoundedNearestCandidateHeap(storage: candidateStorage)
                var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestStorage)
                let entry = HNSWNeighborCandidate(internalID: current, distance: currentDistance)
                candidates.pushUnchecked(entry)
                nearest.pushUnchecked(entry)
                var lowerBound = entry.distance
                markVisited(current, tag: tag, visited: visitedBuffer)

                while !candidates.isEmpty {
                    let candidate = candidates.popUnchecked()
                    if candidate.distance > lowerBound, nearest.count >= effectiveEF {
                        break
                    }
                    let neighborRange = graph.connections.level0NeighborStorageRange(
                        for: candidate.internalID
                    )
                    guard !neighborRange.isEmpty else { continue }
                    for neighborStorageIndex in neighborRange {
                        let neighbor = graph.connections.level0NeighborInStorage(
                            at: neighborStorageIndex
                        )
                        guard markVisited(neighbor, tag: tag, visited: visitedBuffer) else { continue }
                        let lowerBoundForScoring = nearest.count >= effectiveEF
                            ? lowerBound
                            : nil
                        guard let nextDistance = searchDistance(
                            to: neighbor,
                            packedStorage: packedStorage,
                            mseQuery: mseQuery,
                            distanceTable: distanceTable,
                            packedDistanceTable: packedDistanceTable,
                            qjlDistanceTable: qjlDistanceTable,
                            qjlAbsoluteSum: qjlAbsoluteSum,
                            lowerBound: lowerBoundForScoring
                        ) else {
                            continue
                        }
                        let next = HNSWNeighborCandidate(
                            internalID: neighbor,
                            distance: nextDistance
                        )
                        if nearest.count < effectiveEF || next.distance < lowerBound {
                            candidates.pushUnchecked(next)
                            if nearest.count < effectiveEF {
                                nearest.pushUnchecked(next)
                            } else {
                                nearest.replaceTopUnchecked(with: next)
                            }
                            lowerBound = nearest.topDistanceUnchecked
                        }
                    }
                }
                var output = [SearchResult](repeating: SearchResult(label: 0, distance: 0), count: resultCapacity)
                let count = output.withUnsafeMutableBufferPointer { outputBuffer in
                    return nearest.writeClosestReranked(
                        limit: k,
                        using: resultStorage,
                        to: outputBuffer,
                        labelOrder: graph.labelOrder,
                        metric: .innerProduct,
                        isEligible: { internalID in
                            let index = Int(internalID)
                            return index < graph.deletedFlags.count
                                && graph.deletedFlags[index] == 0
                        },
                        comparisonDistance: { $0.distance }
                    )
                }
                if count < output.count {
                    output.removeLast(output.count - count)
                }
                return output
            }
        }
    }

    @inline(__always)
    private func searchDistance(
        to internalID: HNSWInternalID,
        packedStorage: UnsafeBufferPointer<UInt8>,
        mseQuery: UnsafeBufferPointer<Float>,
        distanceTable: UnsafeBufferPointer<Float>,
        packedDistanceTable: UnsafeBufferPointer<Float>,
        qjlDistanceTable: UnsafeBufferPointer<Float>,
        qjlAbsoluteSum: Float,
        lowerBound: Float?
    ) -> Float? {
        let mse = mseDistance(
            to: internalID,
            packedStorage: packedStorage,
            mseQuery: mseQuery,
            distanceTable: distanceTable,
            packedDistanceTable: packedDistanceTable
        )
        guard !qjlDistanceTable.isEmpty else {
            return mse
        }
        let offset = Int(internalID) * _packedSize
        let residualNorm = Self.loadResidualNorm(
            from: packedStorage,
            at: offset + msePackedSize + qjlPackedSize
        )
        if usesQJLDistancePruning,
           let lowerBound,
           mse - residualNorm * qjlScale * qjlAbsoluteSum > lowerBound {
            return nil
        }
        return productDistance(
            fromMSEDistance: mse,
            to: internalID,
            packedStorage: packedStorage,
            qjlDistanceTable: qjlDistanceTable,
            residualNorm: residualNorm
        )
    }

    @inline(__always)
    private func mseDistance(
        to internalID: HNSWInternalID,
        packedStorage: UnsafeBufferPointer<UInt8>,
        mseQuery: UnsafeBufferPointer<Float>,
        distanceTable: UnsafeBufferPointer<Float>,
        packedDistanceTable: UnsafeBufferPointer<Float>
    ) -> Float {
        let offset = Int(internalID) * _packedSize
        let mseCode = UnsafeBufferPointer(
            start: packedStorage.baseAddress! + offset,
            count: msePackedSize
        )
        let mseInnerProduct: Float
        if mseBitWidth == 0 {
            mseInnerProduct = 0
        } else if usesAcceleratedFourBitKernel {
            mseInnerProduct = quantizer.fourBitInnerProduct(
                from: mseCode,
                query: mseQuery
            )
        } else if !packedDistanceTable.isEmpty {
            mseInnerProduct = quantizer.packedInnerProduct(
                from: mseCode,
                using: packedDistanceTable
            )
        } else {
            mseInnerProduct = quantizer.innerProduct(
                from: mseCode,
                using: distanceTable
            )
        }
        return 1 - mseInnerProduct
    }

    @inline(__always)
    private func productDistance(
        fromMSEDistance mseDistance: Float,
        to internalID: HNSWInternalID,
        packedStorage: UnsafeBufferPointer<UInt8>,
        qjlDistanceTable: UnsafeBufferPointer<Float>,
        residualNorm: Float
    ) -> Float {
        let offset = Int(internalID) * _packedSize
        let qjlSigns = UnsafeBufferPointer(
            start: packedStorage.baseAddress! + offset + msePackedSize,
            count: qjlPackedSize
        )
        let qjlInnerProduct = QJLProjection.innerProduct(
            signs: qjlSigns,
            using: qjlDistanceTable
        )
        return mseDistance - residualNorm * qjlScale * qjlInnerProduct
    }

    private static func storeResidualNorm(
        _ value: Float,
        in record: UnsafeMutableBufferPointer<UInt8>,
        at offset: Int
    ) {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { bytes in
            for index in bytes.indices {
                record[offset + index] = bytes[index]
            }
        }
    }

    private static func loadResidualNorm(
        from storage: UnsafeBufferPointer<UInt8>,
        at offset: Int
    ) -> Float {
        // The packed-storage Array owns the initialized record for this synchronous borrow.
        // Archive and constructor validation guarantee four readable bytes at `offset`;
        // unaligned loading avoids an alignment assumption and the pointer does not escape.
        let raw = UnsafeRawPointer(storage.baseAddress! + offset)
            .loadUnaligned(as: UInt32.self)
        return Float(bitPattern: UInt32(littleEndian: raw))
    }

    private static func validateVector(
        _ vector: UnsafeBufferPointer<Float>,
        name: String
    ) throws {
        var maximumMagnitude: Float = 0
        for value in vector {
            guard value.isFinite else {
                throw HNSWError.invalidArgument(
                    "\(name) must contain only finite values"
                )
            }
            maximumMagnitude = max(maximumMagnitude, abs(value))
        }
        guard maximumMagnitude > 0 else {
            throw HNSWError.invalidArgument("\(name) must have nonzero norm")
        }
    }

    private static func normalizeValidatedVector(
        _ input: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        var maximumMagnitude: Float = 0
        for value in input {
            maximumMagnitude = max(maximumMagnitude, abs(value))
        }
        var scaledSquaredNorm: Float = 0
        for value in input {
            let scaled = value / maximumMagnitude
            scaledSquaredNorm += scaled * scaled
        }
        let inverseScaledNorm = 1 / scaledSquaredNorm.squareRoot()
        for index in input.indices {
            output[index] = (input[index] / maximumMagnitude)
                * inverseScaledNorm
        }
    }

    @inline(__always)
    private func nextVisitedTag(
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16
    ) -> UInt16 {
        if visitedTag == UInt16.max {
            for index in visited.indices { visited[index] = 0 }
            visitedTag = 0
        }
        visitedTag += 1
        return visitedTag
    }

    @discardableResult
    @inline(__always)
    private func markVisited(
        _ internalID: HNSWInternalID,
        tag: UInt16,
        visited: UnsafeMutableBufferPointer<UInt16>
    ) -> Bool {
        let index = Int(internalID)
        guard index >= 0, index < visited.count else { return false }
        guard visited[index] != tag else { return false }
        visited[index] = tag
        return true
    }

}

#if canImport(FoundationEssentials) || canImport(Foundation)
extension TurboQuantIndex {
    private static let headerMagic: UInt32 = 0x5451_5746 // "TQWF"
    private static let headerVersion: UInt32 = 5

    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        let data = try state.withLock { state -> Data in
            try finalizeLocked(state: &state)
            guard let graph = state.graph else {
                throw HNSWError.serializationFailed("TurboQuant graph is unavailable")
            }

            var output = Data()
            output.appendLittleEndian(Self.headerMagic)
            output.appendLittleEndian(Self.headerVersion)
            output.appendLittleEndian(UInt32(dimensions))
            output.appendLittleEndian(UInt32(bitWidth))
            output.appendLittleEndian(objective.rawValue)
            output.appendLittleEndian(rotationStrategy.rawValue)
            output.appendLittleEndian(seed)
            output.appendLittleEndian(UInt32(paddedDimensions))
            output.appendLittleEndian(UInt32(maximumElementCount))
            output.appendLittleEndian(UInt32(graph.labelOrder.count))
            output.appendLittleEndian(UInt32(configuration.m))
            output.appendLittleEndian(UInt32(configuration.efConstruction))
            output.appendLittleEndian(UInt32(state.efSearch))
            output.appendLittleEndian(UInt64(bitPattern: Int64(configuration.randomSeed)))
            output.append(configuration.allowReplaceDeleted ? 1 : 0)
            output.append(contentsOf: [0, 0, 0])
            output.appendLittleEndian(graph.entryPoint ?? UInt32.max)
            output.appendLittleEndian(UInt32(bitPattern: Int32(graph.maxLevel)))
            output.appendLittleEndian(UInt32(_packedSize))

            for index in graph.labelOrder.indices {
                output.appendLittleEndian(graph.labelOrder[index])
                output.append(graph.deletedFlags[index])
                output.append(contentsOf: [0, 0, 0])
                output.appendLittleEndian(UInt32(graph.levels[index]))
                let levelCount = graph.connections.levelCount(
                    for: HNSWInternalID(index)
                )
                output.appendLittleEndian(UInt32(levelCount))
                for level in 0..<levelCount {
                    let neighborRange = graph.connections.neighborStorageRange(
                        for: HNSWInternalID(index),
                        at: level
                    )
                    output.appendLittleEndian(UInt32(neighborRange.count))
                    for neighborStorageIndex in neighborRange {
                        let neighbor = graph.connections.neighborInStorage(
                            at: neighborStorageIndex,
                            level: level
                        )
                        output.appendLittleEndian(neighbor)
                    }
                }
                let start = index * _packedSize
                output.append(contentsOf: state.packedStorage[start..<(start + _packedSize)])
            }
            return output
        }

        do {
#if os(WASI)
            try data.write(to: URL(fileURLWithPath: path))
#else
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
#endif
        } catch {
            throw HNSWError.saveFailed("Failed to save index to \(path)")
        }
    }

    public static func load(from url: URL) throws -> TurboQuantIndex {
        try load(from: url.path)
    }

    public static func load(from path: String) throws -> TurboQuantIndex {
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            throw HNSWError.loadFailed("Failed to load index from \(path)")
        }

        var reader = TurboQuantArchiveReader(data: data)
        let magic = try reader.readUInt32()
        guard magic == headerMagic else {
            throw HNSWError.loadFailed("Invalid TurboQuant archive magic")
        }
        let version = try reader.readUInt32()
        guard version == headerVersion else {
            throw HNSWError.loadFailed("Unsupported TurboQuant archive version \(version)")
        }
        let dimensions = try reader.readPositiveInt32(name: "dimensions")
        let bitWidth = try reader.readPositiveInt32(name: "bitWidth")
        let objectiveRawValue = try reader.readUInt32()
        guard let objective = TurboQuantObjective(rawValue: objectiveRawValue) else {
            throw HNSWError.loadFailed("TurboQuant archive contains an invalid objective")
        }
        let rotationStrategyRawValue = try reader.readUInt32()
        guard let rotationStrategy = TurboQuantRotationStrategy(
            rawValue: rotationStrategyRawValue
        ) else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains an invalid rotation strategy"
            )
        }
        let seed = try reader.readUInt64()
        let paddedDimensions = try reader.readPositiveInt32(name: "paddedDimensions")
        let maximumElementCount = try reader.readPositiveInt32(name: "maximumElementCount")
        let labelCount = try reader.readNonNegativeInt32(name: "labelCount")
        let m = try reader.readPositiveInt32(name: "m")
        let efConstruction = try reader.readPositiveInt32(name: "efConstruction")
        let efSearch = try reader.readPositiveInt32(name: "efSearch")
        guard (2...HNSWConfiguration.maximumM).contains(m) else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains an invalid m"
            )
        }
        guard efConstruction >= m else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains an invalid efConstruction"
            )
        }
        let randomSeedValue = Int64(bitPattern: try reader.readUInt64())
        guard randomSeedValue >= Int64(Int.min), randomSeedValue <= Int64(Int.max) else {
            throw HNSWError.loadFailed("TurboQuant archive contains an invalid random seed")
        }
        let randomSeed = Int(randomSeedValue)
        let allowReplaceDeletedRaw = try reader.readUInt8()
        guard allowReplaceDeletedRaw <= 1 else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains an invalid allowReplaceDeleted value"
            )
        }
        let allowReplaceDeleted = allowReplaceDeletedRaw == 1
        guard try reader.readUInt8() == 0,
              try reader.readUInt8() == 0,
              try reader.readUInt8() == 0 else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains nonzero reserved header bytes"
            )
        }
        let entryPointRaw = try reader.readUInt32()
        let maxLevel = Int(Int32(bitPattern: try reader.readUInt32()))
        let storedPackedSize = try reader.readPositiveInt32(name: "packedSize")

        guard labelCount <= maximumElementCount else {
            throw HNSWError.loadFailed("TurboQuant archive contains more labels than its capacity")
        }
        let minimumNodeMetadataBytes = 24
        guard storedPackedSize <= Int.max - minimumNodeMetadataBytes else {
            throw HNSWError.loadFailed(
                "TurboQuant archive record size exceeds the addressable range"
            )
        }
        let minimumRecordBytes = minimumNodeMetadataBytes + storedPackedSize
        guard labelCount == 0
                || labelCount <= reader.remainingByteCount / minimumRecordBytes else {
            throw HNSWError.loadFailed(
                "TurboQuant archive payload is too small for its declared label count"
            )
        }
        try validateArchiveRetainedCapacity(
            labelCount: labelCount,
            totalUpperLevelSlotCount: 0,
            m: m,
            efSearch: efSearch
        )

        let configuration = HNSWConfiguration(
            m: m,
            efConstruction: efConstruction,
            efSearch: efSearch,
            randomSeed: randomSeed,
            allowReplaceDeleted: allowReplaceDeleted
        )
        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: max(maximumElementCount, labelCount),
            bitWidth: bitWidth,
            objective: objective,
            rotationStrategy: rotationStrategy,
            configuration: configuration,
            seed: seed,
            acceleratedFourBitKernelAvailable:
                ScalarQuantizer.hasAcceleratedFourBitKernel,
            qjlDistancePruningEnabled: true,
            createsBuilder: false
        )
        guard index.paddedDimensions == paddedDimensions,
              index.bytesPerVector == storedPackedSize else {
            throw HNSWError.loadFailed("TurboQuant archive metadata does not match the quantizer")
        }

        var labels: [UInt64] = []
        var deletedFlags: [UInt8] = []
        var levels: [Int] = []
        var loadedConnections: [[[HNSWInternalID]]] = []
        var packedStorage: [UInt8] = []
        guard labelCount == 0 || labelCount <= Int.max / storedPackedSize else {
            throw HNSWError.loadFailed("TurboQuant archive exceeds addressable packed storage")
        }
        labels.reserveCapacity(labelCount)
        deletedFlags.reserveCapacity(labelCount)
        levels.reserveCapacity(labelCount)
        loadedConnections.reserveCapacity(labelCount)
        packedStorage.reserveCapacity(labelCount * storedPackedSize)
        var seenLabels = Set<UInt64>()
        seenLabels.reserveCapacity(labelCount)
        var totalUpperLevelSlotCount = 0

        for internalID in 0..<labelCount {
            let label = try reader.readUInt64()
            let deleted = try reader.readUInt8()
            guard deleted <= 1, seenLabels.insert(label).inserted else {
                throw HNSWError.loadFailed("TurboQuant archive contains invalid or duplicate labels")
            }
            guard try reader.readUInt8() == 0,
                  try reader.readUInt8() == 0,
                  try reader.readUInt8() == 0 else {
                throw HNSWError.loadFailed(
                    "TurboQuant archive contains nonzero reserved node bytes"
                )
            }
            let level = try reader.readNonNegativeInt32(name: "level")
            guard level < maximumSerializedLevelCount else {
                throw HNSWError.loadFailed("TurboQuant archive contains an invalid level")
            }
            let (nextUpperLevelSlotCount, upperLevelCountOverflowed) =
                totalUpperLevelSlotCount.addingReportingOverflow(level)
            guard !upperLevelCountOverflowed else {
                throw HNSWError.loadFailed(
                    "TurboQuant archive retained graph storage exceeds the addressable range"
                )
            }
            try validateArchiveRetainedCapacity(
                labelCount: labelCount,
                totalUpperLevelSlotCount: nextUpperLevelSlotCount,
                m: m,
                efSearch: efSearch
            )
            totalUpperLevelSlotCount = nextUpperLevelSlotCount
            let levelCount = try reader.readNonNegativeInt32(name: "levelCount")
            guard levelCount == level + 1 else {
                throw HNSWError.loadFailed("TurboQuant archive level count does not match node level")
            }
            var nodeConnections: [[HNSWInternalID]] = []
            nodeConnections.reserveCapacity(levelCount)
            for levelIndex in 0..<levelCount {
                let neighborCount = try reader.readNonNegativeInt32(name: "neighborCount")
                guard neighborCount <= max(0, labelCount - 1) else {
                    throw HNSWError.loadFailed(
                        "TurboQuant archive contains too many neighbors"
                    )
                }
                guard neighborCount <= maximumConnections(
                    at: levelIndex,
                    m: m
                ) else {
                    throw HNSWError.loadFailed(
                        "TurboQuant archive exceeds the configured connection limit"
                    )
                }
                var neighbors: [HNSWInternalID] = []
                neighbors.reserveCapacity(neighborCount)
                for _ in 0..<neighborCount {
                    let neighbor = try reader.readUInt32()
                    guard UInt64(neighbor) < UInt64(labelCount) else {
                        throw HNSWError.loadFailed("TurboQuant archive contains an invalid neighbor id")
                    }
                    guard UInt64(neighbor) != UInt64(internalID) else {
                        throw HNSWError.loadFailed(
                            "TurboQuant archive contains a self edge"
                        )
                    }
                    neighbors.append(neighbor)
                }
                nodeConnections.append(neighbors)
            }
            loadedConnections.append(nodeConnections)
            labels.append(label)
            deletedFlags.append(deleted)
            levels.append(level)
            let recordStart = packedStorage.count
            try reader.appendBytes(
                count: storedPackedSize,
                to: &packedStorage
            )
            try validatePackedPadding(
                storage: packedStorage,
                byteOffset: recordStart,
                byteCount: index.msePackedSize,
                meaningfulBitCount: index.paddedDimensions * index.mseBitWidth,
                bitStreamOrder: index.mseBitWidth == 3 ? .leastSignificantFirst : .mostSignificantFirst,
                name: "MSE code"
            )
            try validatePackedPadding(
                storage: packedStorage,
                byteOffset: recordStart + index.msePackedSize,
                byteCount: index.qjlPackedSize,
                meaningfulBitCount: objective == .innerProduct ? dimensions : 0,
                bitStreamOrder: .mostSignificantFirst,
                name: "QJL code"
            )
            if objective == .innerProduct {
                let residualNorm = packedStorage.withUnsafeBufferPointer { storage in
                    Self.loadResidualNorm(
                        from: storage,
                        at: recordStart + index.msePackedSize + index.qjlPackedSize
                    )
                }
                guard residualNorm.isFinite,
                      residualNorm >= 0,
                      residualNorm <= index.maximumResidualNorm else {
                    throw HNSWError.loadFailed(
                        "TurboQuant archive contains an invalid residual norm"
                    )
                }
            }
        }
        try reader.ensureAtEnd()
        var connectionStore = HNSWConnectionStore(m: m)
        connectionStore.replaceAll(withInternalIDs: loadedConnections)
        let entryPoint = entryPointRaw == UInt32.max ? nil : entryPointRaw
        try validateLoadedGraph(
            labels: labels,
            levels: levels,
            connections: connectionStore,
            entryPoint: entryPoint,
            maxLevel: maxLevel,
            m: m
        )

        index.state.withLock { state in
            state.builder = nil
            state.codeOffsetsByLabel.removeAll(keepingCapacity: false)
            state.stagingPackedStorage.removeAll(keepingCapacity: false)
            state.recordWorkspace.removeAll(keepingCapacity: false)
            state.packedStorage = packedStorage
            state.graph = Graph(
                labelOrder: labels,
                deletedFlags: deletedFlags,
                levels: levels,
                connections: connectionStore,
                entryPoint: entryPoint,
                maxLevel: maxLevel
            )
            state.visited = [UInt16](repeating: 0, count: labelCount)
            state.visitedTag = 0
            state.searchWorkspace.prepare(
                candidateCapacity: labelCount,
                nearestCapacity: min(efSearch, labelCount),
                resultCapacity: 1,
                visitedCapacity: labelCount
            )
            state.finalized = true
        }
        return index
    }

    private static func validatePackedPadding(
        storage: [UInt8],
        byteOffset: Int,
        byteCount: Int,
        meaningfulBitCount: Int,
        bitStreamOrder: PackedBitStreamOrder,
        name: String
    ) throws {
        guard byteCount > 0 else {
            return
        }
        let usedBitsInFinalByte = meaningfulBitCount % 8
        guard usedBitsInFinalByte > 0 else {
            return
        }
        let unusedBitCount = 8 - usedBitsInFinalByte
        let unusedMask: UInt8
        switch bitStreamOrder {
        case .mostSignificantFirst:
            unusedMask = UInt8((1 << unusedBitCount) - 1)
        case .leastSignificantFirst:
            unusedMask = UInt8.max << usedBitsInFinalByte
        }
        guard storage[byteOffset + byteCount - 1] & unusedMask == 0 else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains nonzero \(name) padding bits"
            )
        }
    }

    private enum PackedBitStreamOrder {
        case mostSignificantFirst
        case leastSignificantFirst
    }

    private static func validateArchiveRetainedCapacity(
        labelCount: Int,
        totalUpperLevelSlotCount: Int,
        m: Int,
        efSearch: Int
    ) throws {
        guard let graphBytes = HNSWConnectionStore.retainedByteCount(
            nodeCount: labelCount,
            totalUpperLevelSlotCount: totalUpperLevelSlotCount,
            m: m
        ) else {
            throw HNSWError.loadFailed(
                "TurboQuant archive retained graph storage exceeds the addressable range"
            )
        }
        guard graphBytes <= HNSWConnectionStore.maximumRetainedByteCount else {
            throw HNSWError.loadFailed(
                "TurboQuant archive retained graph storage exceeds the 256 MiB limit"
            )
        }
        guard let searchBytes = HNSWSearchWorkspace.retainedByteCount(
            candidateCapacity: labelCount,
            nearestCapacity: min(efSearch, labelCount),
            resultCapacity: 1,
            visitedCapacity: labelCount
        ) else {
            throw HNSWError.loadFailed(
                "TurboQuant archive search workspace exceeds the addressable range"
            )
        }
        guard searchBytes <= HNSWSearchWorkspace.maximumRetainedByteCount else {
            throw HNSWError.loadFailed(
                "TurboQuant archive search workspace exceeds the 256 MiB limit"
            )
        }
    }

    private static func validateLoadedGraph(
        labels: [UInt64],
        levels: [Int],
        connections: HNSWConnectionStore,
        entryPoint: HNSWInternalID?,
        maxLevel: Int,
        m: Int
    ) throws {
        guard labels.count == levels.count,
              labels.count == connections.nodeCount else {
            throw HNSWError.loadFailed(
                "TurboQuant archive graph metadata is inconsistent"
            )
        }
        guard !labels.isEmpty else {
            guard entryPoint == nil, maxLevel == -1 else {
                throw HNSWError.loadFailed(
                    "Empty TurboQuant archive contains graph metadata"
                )
            }
            return
        }
        guard let entryPoint,
              UInt64(entryPoint) < UInt64(labels.count) else {
            throw HNSWError.loadFailed(
                "TurboQuant archive contains an invalid entry point"
            )
        }
        let entryPointIndex = Int(entryPoint)

        var observedMaxLevel = -1
        for internalID in labels.indices {
            let level = levels[internalID]
            guard level >= 0, level < maximumSerializedLevelCount else {
                throw HNSWError.loadFailed(
                    "TurboQuant archive contains an invalid level"
                )
            }
            let nodeID = HNSWInternalID(internalID)
            guard connections.levelCount(for: nodeID) == level + 1 else {
                throw HNSWError.loadFailed(
                    "TurboQuant archive connection level count is inconsistent"
                )
            }
            observedMaxLevel = max(observedMaxLevel, level)

            for levelIndex in 0...level {
                let neighborRange = connections.neighborStorageRange(
                    for: nodeID,
                    at: levelIndex
                )
                guard neighborRange.count <= maximumConnections(
                    at: levelIndex,
                    m: m
                ) else {
                    throw HNSWError.loadFailed(
                        "TurboQuant archive exceeds the configured connection limit"
                    )
                }
                var seenNeighbors = Set<HNSWInternalID>()
                seenNeighbors.reserveCapacity(neighborRange.count)
                for neighborStorageIndex in neighborRange {
                    let neighbor = connections.neighborInStorage(
                        at: neighborStorageIndex,
                        level: levelIndex
                    )
                    guard UInt64(neighbor) < UInt64(labels.count) else {
                        throw HNSWError.loadFailed(
                            "TurboQuant archive contains an invalid neighbor id"
                        )
                    }
                    let neighborIndex = Int(neighbor)
                    guard neighborIndex != internalID else {
                        throw HNSWError.loadFailed(
                            "TurboQuant archive contains a self edge"
                        )
                    }
                    guard connections.levelCount(for: neighbor) > levelIndex else {
                        throw HNSWError.loadFailed(
                            "TurboQuant archive references a missing neighbor level"
                        )
                    }
                    guard seenNeighbors.insert(neighbor).inserted else {
                        throw HNSWError.loadFailed(
                            "TurboQuant archive contains a duplicate neighbor"
                        )
                    }
                }
            }
        }

        guard observedMaxLevel == maxLevel else {
            throw HNSWError.loadFailed(
                "TurboQuant archive maximum level is inconsistent"
            )
        }
        guard levels[entryPointIndex] == maxLevel else {
            throw HNSWError.loadFailed(
                "TurboQuant archive entry point level is inconsistent"
            )
        }
    }

    private static func maximumConnections(at level: Int, m: Int) -> Int {
        level == 0 ? max(1, m * 2) : max(1, m)
    }
}

private struct TurboQuantArchiveReader {
    let data: Data
    var offset: Int = 0

    var remainingByteCount: Int {
        data.count - offset
    }

    mutating func readUInt8() throws -> UInt8 {
        guard offset < data.count else { throw HNSWError.loadFailed("TurboQuant archive is truncated") }
        defer { offset += 1 }
        return data[offset]
    }

    mutating func readUInt32() throws -> UInt32 {
        let value: UInt32 = try readInteger()
        return UInt32(littleEndian: value)
    }

    mutating func readUInt64() throws -> UInt64 {
        let value: UInt64 = try readInteger()
        return UInt64(littleEndian: value)
    }

    private mutating func readInteger<Value>() throws -> Value {
        let byteCount = MemoryLayout<Value>.size
        guard offset <= data.count, byteCount <= data.count - offset else {
            throw HNSWError.loadFailed("TurboQuant archive is truncated")
        }
        let value = data.withUnsafeBytes { bytes in
            bytes.loadUnaligned(fromByteOffset: offset, as: Value.self)
        }
        offset += byteCount
        return value
    }

    mutating func appendBytes(
        count: Int,
        to output: inout [UInt8]
    ) throws {
        guard count >= 0,
              offset <= data.count,
              count <= data.count - offset,
              output.count <= Int.max - count else {
            throw HNSWError.loadFailed("TurboQuant archive is truncated")
        }
        output.append(contentsOf: data[offset..<(offset + count)])
        offset += count
    }

    mutating func readPositiveInt32(name: String) throws -> Int {
        let raw = try readUInt32()
        guard raw > 0, UInt64(raw) <= UInt64(Int.max) else {
            throw HNSWError.loadFailed("TurboQuant archive contains invalid \(name)")
        }
        return Int(raw)
    }

    mutating func readNonNegativeInt32(name: String) throws -> Int {
        let raw = try readUInt32()
        guard UInt64(raw) <= UInt64(Int.max) else {
            throw HNSWError.loadFailed("TurboQuant archive contains invalid \(name)")
        }
        return Int(raw)
    }

    mutating func ensureAtEnd() throws {
        guard offset == data.count else {
            throw HNSWError.loadFailed("TurboQuant archive has trailing bytes")
        }
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }
}
#endif
