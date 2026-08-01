#if canImport(FoundationEssentials)
import FoundationEssentials
#elseif canImport(Foundation)
import Foundation
#endif
import Synchronization

/// In-memory HNSW graph index.
public final class HNSWIndex<Scalar: HNSWScalar>: Sendable {

    private struct Entry: Sendable {
        var internalID: HNSWInternalID
        var offset: Int
        var deleted: Bool
        var level: Int
    }

    private struct State: Sendable {
        var maximumElementCount: Int
        var efSearch: Int
        var entries: [UInt64: Entry]
        var labelOrder: [UInt64]
        var deletedFlags: [UInt8]
        var levels: [Int]
        var liveCount: Int
        var comparisonStorage: [Float]
        var halfComparisonStorage: [Float16]
        var connections: HNSWConnectionStore
        var entryPoint: HNSWInternalID?
        var maxLevel: Int
        var levelGenerator: HNSWLevelGenerator
        var visited: [UInt16]
        var visitedTag: UInt16
        var queryWorkspace: [Float]
        var halfPrecisionQueryWorkspace: [Float16]
        var searchWorkspace: HNSWSearchWorkspace
    }

    public let dimensions: Int
    public let metric: DistanceMetric
    public let configuration: HNSWConfiguration

    private let state: Mutex<State>

    public init(
        dimensions: Int,
        maxElements: Int,
        metric: DistanceMetric = .l2,
        configuration: HNSWConfiguration = .balanced
    ) throws {
        guard dimensions > 0 else {
            throw HNSWError.initializationFailed("Dimensions must be positive")
        }
        guard UInt64(dimensions) <= UInt64(UInt32.max) else {
            throw HNSWError.initializationFailed(
                "Dimensions must fit the archive format"
            )
        }
        let queryScalarSize = Scalar.self == Float16.self
            ? MemoryLayout<Float16>.stride
            : MemoryLayout<Float>.stride
        guard dimensions <= HNSWSearchWorkspace.maximumRetainedByteCount
                / queryScalarSize else {
            throw HNSWError.initializationFailed(
                "Query workspace exceeds the 256 MiB limit"
            )
        }
        guard maxElements > 0 else {
            throw HNSWError.initializationFailed("Maximum element count must be positive")
        }
        guard UInt64(maxElements) <= UInt64(UInt32.max) else {
            throw HNSWError.initializationFailed("Maximum element count exceeds UInt32 internal id capacity")
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
#if _pointerBitWidth(_64)
        guard configuration.randomSeed >= Int(Int32.min),
              configuration.randomSeed <= Int(Int32.max) else {
            throw HNSWError.invalidArgument(
                "randomSeed must fit the archive format"
            )
        }
#endif
        self.dimensions = dimensions
        self.metric = metric
        self.configuration = configuration
        let usesHalfStorage = Scalar.self == Float16.self
        var comparisonStorage: [Float] = []
        var halfComparisonStorage: [Float16] = []
        let maximumEagerComparisonBytes = 64 * 1_024 * 1_024
        if maxElements <= Int.max / dimensions {
            let elementCount = maxElements * dimensions
            let scalarSize = usesHalfStorage
                ? MemoryLayout<Float16>.stride
                : MemoryLayout<Float>.stride
            let canReserve = elementCount <= maximumEagerComparisonBytes / scalarSize
            if canReserve {
                if usesHalfStorage {
                    halfComparisonStorage.reserveCapacity(elementCount)
                } else {
                    comparisonStorage.reserveCapacity(elementCount)
                }
            }
        }
        self.state = Mutex(State(
            maximumElementCount: maxElements,
            efSearch: configuration.efSearch,
            entries: [:],
            labelOrder: [],
            deletedFlags: [],
            levels: [],
            liveCount: 0,
            comparisonStorage: comparisonStorage,
            halfComparisonStorage: halfComparisonStorage,
            connections: HNSWConnectionStore(m: configuration.m),
            entryPoint: nil,
            maxLevel: -1,
            levelGenerator: HNSWLevelGenerator(seed: configuration.randomSeed),
            visited: [],
            visitedTag: 0,
            queryWorkspace: usesHalfStorage ? [] : [Float](repeating: 0, count: dimensions),
            halfPrecisionQueryWorkspace: usesHalfStorage ? [Float16](repeating: 0, count: dimensions) : [],
            searchWorkspace: HNSWSearchWorkspace()
        ))
    }

    public var count: Int {
        state.withLock {
            $0.entries.count
        }
    }

    public var capacity: Int {
        state.withLock {
            $0.maximumElementCount
        }
    }

    public var isEmpty: Bool { count == 0 }

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

    public func resize(to newCapacity: Int) throws {
        guard newCapacity > 0 else {
            throw HNSWError.invalidArgument("newCapacity must be positive")
        }
        guard UInt64(newCapacity) <= UInt64(UInt32.max) else {
            throw HNSWError.invalidArgument("newCapacity must fit the archive format")
        }
        try state.withLock {
            guard newCapacity >= $0.entries.count else {
                throw HNSWError.addPointFailed("Cannot resize index below current element count")
            }
            $0.maximumElementCount = newCapacity
        }
    }
}

extension HNSWIndex {

    public func add(_ vector: [Scalar], label: UInt64) throws {
        try vector.withUnsafeBufferPointer { buffer in
            try add(buffer, label: label)
        }
    }

    /// Add a borrowed vector to the index without materializing an intermediate array.
    ///
    /// The index stores vectors in an owned contiguous region. This call performs
    /// the required ownership copy exactly once and avoids temporary slice arrays.
    public func add(_ vector: UnsafeBufferPointer<Scalar>, label: UInt64) throws {
        try validateDimensions(vector.count)
        let normalizationScale = try validateInputVector(vector, name: "vector")

        try state.withLock {
            try upsertVector(
                vector,
                label: label,
                normalizationScale: normalizationScale,
                state: &$0
            )
        }
    }

    public func search(_ query: [Scalar], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Search with a borrowed query vector without materializing an intermediate array.
    ///
    /// `k` must be positive.
    public func search(_ query: UnsafeBufferPointer<Scalar>, k: Int) throws -> [SearchResult] {
        try validateDimensions(query.count)
        guard k > 0 else {
            throw HNSWError.invalidArgument("k must be positive")
        }
        let normalizationScale = try validateInputVector(query, name: "query")

        return try state.withLock { state in
            try validateSearchWorkspace(k: k, state: state)
            if Scalar.self == Float.self, !metric.requiresNormalization {
                let floatQuery = UnsafeBufferPointer<Float>(
                    start: UnsafeRawPointer(query.baseAddress!).assumingMemoryBound(to: Float.self),
                    count: query.count
                )
                return searchNormalized(floatQuery, k: k, state: &state)
            }

            if Scalar.self == Float16.self {
                if !metric.requiresNormalization {
                    let halfQuery = UnsafeBufferPointer<Float16>(
                        start: UnsafeRawPointer(query.baseAddress!).assumingMemoryBound(to: Float16.self),
                        count: query.count
                    )
                    return searchNormalizedHalf(halfQuery, k: k, state: &state)
                }

                ensureCapacity(&state.halfPrecisionQueryWorkspace, count: dimensions)
                state.halfPrecisionQueryWorkspace.withUnsafeMutableBufferPointer { preparedQuery in
                    storeNormalizedHalfVector(
                        query,
                        normalizationScale: normalizationScale,
                        into: preparedQuery
                    )
                }
                return state.halfPrecisionQueryWorkspace.withUnsafeBufferPointer { preparedQuery in
                    searchNormalizedHalf(preparedQuery, k: k, state: &state)
                }
            }

            ensureCapacity(&state.queryWorkspace, count: dimensions)
            state.queryWorkspace.withUnsafeMutableBufferPointer { preparedQuery in
                if metric.requiresNormalization {
                    storeNormalizedVector(
                        query,
                        normalizationScale: normalizationScale,
                        into: preparedQuery
                    )
                } else {
                    storeVector(query, into: preparedQuery)
                }
            }
            return state.queryWorkspace.withUnsafeBufferPointer { preparedQuery in
                searchNormalized(preparedQuery, k: k, state: &state)
            }
        }
    }

    /// Search into a caller-provided output buffer.
    ///
    /// Returns the number of results written. `k` must be positive and `results` must have
    /// room for at least `k` values.
    @discardableResult
    public func search(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>
    ) throws -> Int {
        try validateDimensions(query.count)
        guard k > 0 else {
            throw HNSWError.invalidArgument("k must be positive")
        }
        guard results.count >= k else {
            throw HNSWError.invalidArgument("Result buffer must contain at least k elements")
        }
        let normalizationScale = try validateInputVector(query, name: "query")

        return try state.withLock { state in
            try validateSearchWorkspace(k: k, state: state)
            if Scalar.self == Float.self, !metric.requiresNormalization {
                let floatQuery = UnsafeBufferPointer<Float>(
                    start: UnsafeRawPointer(query.baseAddress!).assumingMemoryBound(to: Float.self),
                    count: query.count
                )
                return searchNormalized(floatQuery, k: k, into: results, state: &state)
            }
            if Scalar.self == Float16.self {
                if !metric.requiresNormalization {
                    let halfQuery = UnsafeBufferPointer<Float16>(
                        start: UnsafeRawPointer(query.baseAddress!).assumingMemoryBound(to: Float16.self),
                        count: query.count
                    )
                    return searchNormalizedHalf(halfQuery, k: k, into: results, state: &state)
                }

                ensureCapacity(&state.halfPrecisionQueryWorkspace, count: dimensions)
                state.halfPrecisionQueryWorkspace.withUnsafeMutableBufferPointer { preparedQuery in
                    storeNormalizedHalfVector(
                        query,
                        normalizationScale: normalizationScale,
                        into: preparedQuery
                    )
                }
                return state.halfPrecisionQueryWorkspace.withUnsafeBufferPointer { preparedQuery in
                    searchNormalizedHalf(preparedQuery, k: k, into: results, state: &state)
                }
            }

            ensureCapacity(&state.queryWorkspace, count: dimensions)
            state.queryWorkspace.withUnsafeMutableBufferPointer { preparedQuery in
                if metric.requiresNormalization {
                    storeNormalizedVector(
                        query,
                        normalizationScale: normalizationScale,
                        into: preparedQuery
                    )
                } else {
                    storeVector(query, into: preparedQuery)
                }
            }
            return state.queryWorkspace.withUnsafeBufferPointer { preparedQuery in
                searchNormalized(preparedQuery, k: k, into: results, state: &state)
            }
        }
    }

    @discardableResult
    public func search(
        _ query: [Scalar],
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>
    ) throws -> Int {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k, into: results)
        }
    }

    public func markDeleted(label: UInt64) throws {
        try state.withLock {
            guard var entry = $0.entries[label], !entry.deleted else {
                throw HNSWError.deleteFailed("Failed to delete element with label \(label)")
            }
            entry.deleted = true
            $0.entries[label] = entry
            let internalIndex = Int(entry.internalID)
            if internalIndex < $0.deletedFlags.count {
                $0.deletedFlags[internalIndex] = 1
            }
            $0.liveCount -= 1
        }
    }

    public func unmarkDeleted(label: UInt64) throws {
        try state.withLock {
            guard var entry = $0.entries[label], entry.deleted else {
                throw HNSWError.deleteFailed("Failed to undelete element with label \(label)")
            }
            entry.deleted = false
            $0.entries[label] = entry
            let internalIndex = Int(entry.internalID)
            if internalIndex < $0.deletedFlags.count {
                $0.deletedFlags[internalIndex] = 0
            }
            $0.liveCount += 1
        }
    }
}
extension HNSWIndex {

    @discardableResult
    public func addBatch(_ vectors: [Scalar], labels: [UInt64]) throws -> Int {
        try vectors.withUnsafeBufferPointer { vectorBuffer in
            try labels.withUnsafeBufferPointer { labelBuffer in
                try addBatch(vectorBuffer, labels: labelBuffer)
            }
        }
    }

    /// Add borrowed contiguous vectors to the index.
    ///
    /// `vectors` must contain `labels.count * dimensions` scalars in row-major order.
    @discardableResult
    public func addBatch(
        _ vectors: UnsafeBufferPointer<Scalar>,
        labels: UnsafeBufferPointer<UInt64>
    ) throws -> Int {
        let numVectors = labels.count
        guard numVectors <= Int.max / dimensions else {
            throw HNSWError.invalidArgument(
                "batch dimensions exceed the supported integer range"
            )
        }
        let expectedVectorCount = numVectors * dimensions
        try validateDimensions(vectors.count, expectedTotal: expectedVectorCount)
        guard numVectors > 0 else { return 0 }

        for index in 0..<numVectors {
            let offset = index * dimensions
            let vector = UnsafeBufferPointer(
                start: vectors.baseAddress! + offset,
                count: dimensions
            )
            _ = try validateInputVector(vector, name: "vector")
        }

        return try state.withLock { state in
            var addedCount = 0
            for index in 0..<numVectors {
                let offset = index * dimensions
                let vector = UnsafeBufferPointer(start: vectors.baseAddress! + offset, count: dimensions)
                try upsertVector(vector, label: labels[index], state: &state)
                addedCount += 1
            }
            return addedCount
        }
    }

    @discardableResult
    public func addBatch(_ vectors: [[Scalar]], startingLabel: UInt64? = nil) throws -> Int {
        guard !vectors.isEmpty else { return 0 }
        guard vectors.allSatisfy({ $0.count == dimensions }) else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vectors.first?.count ?? 0)
        }
        for vector in vectors {
            try vector.withUnsafeBufferPointer {
                _ = try validateInputVector($0, name: "vector")
            }
        }

        let start = startingLabel ?? UInt64(count)
        guard UInt64(vectors.count - 1) <= UInt64.max - start else {
            throw HNSWError.invalidArgument(
                "startingLabel and batch size exceed the UInt64 label range"
            )
        }
        var addedCount = 0
        for index in vectors.indices {
            try add(vectors[index], label: start + UInt64(index))
            addedCount += 1
        }
        return addedCount
    }

    public func searchBatch(_ queries: [Scalar], numQueries: Int, k: Int) throws -> [[SearchResult]] {
        try queries.withUnsafeBufferPointer { buffer in
            try searchBatch(buffer, numQueries: numQueries, k: k)
        }
    }

    /// Search borrowed contiguous query vectors. `numQueries` must not be negative and
    /// `k` must be positive.
    public func searchBatch(
        _ queries: UnsafeBufferPointer<Scalar>,
        numQueries: Int,
        k: Int
    ) throws -> [[SearchResult]] {
        guard numQueries >= 0 else {
            throw HNSWError.invalidArgument("numQueries must not be negative")
        }
        guard k > 0 else {
            throw HNSWError.invalidArgument("k must be positive")
        }
        guard numQueries <= Int.max / dimensions else {
            throw HNSWError.invalidArgument(
                "batch dimensions exceed the supported integer range"
            )
        }
        let expectedQueryCount = numQueries * dimensions
        try validateDimensions(queries.count, expectedTotal: expectedQueryCount)
        guard numQueries > 0 else { return [] }

        for index in 0..<numQueries {
            let offset = index * dimensions
            let query = UnsafeBufferPointer(
                start: queries.baseAddress! + offset,
                count: dimensions
            )
            _ = try validateInputVector(query, name: "query")
        }

        return try state.withLock { state in
            try validateSearchWorkspace(k: k, state: state)
            var results: [[SearchResult]] = []
            results.reserveCapacity(numQueries)
            for index in 0..<numQueries {
                let offset = index * dimensions
                let query = UnsafeBufferPointer(start: queries.baseAddress! + offset, count: dimensions)
                if Scalar.self == Float.self, !metric.requiresNormalization {
                    let floatQuery = UnsafeBufferPointer<Float>(
                        start: UnsafeRawPointer(query.baseAddress!).assumingMemoryBound(to: Float.self),
                        count: dimensions
                    )
                    results.append(searchNormalized(floatQuery, k: k, state: &state))
                } else if Scalar.self == Float16.self {
                    if !metric.requiresNormalization {
                        let halfQuery = UnsafeBufferPointer<Float16>(
                            start: UnsafeRawPointer(query.baseAddress!).assumingMemoryBound(to: Float16.self),
                            count: dimensions
                        )
                        results.append(searchNormalizedHalf(halfQuery, k: k, state: &state))
                    } else {
                        ensureCapacity(&state.halfPrecisionQueryWorkspace, count: dimensions)
                        state.halfPrecisionQueryWorkspace.withUnsafeMutableBufferPointer { preparedQuery in
                            storeNormalizedHalfVector(query, into: preparedQuery)
                        }
                        let searchResults = state.halfPrecisionQueryWorkspace.withUnsafeBufferPointer { preparedQuery in
                            searchNormalizedHalf(preparedQuery, k: k, state: &state)
                        }
                        results.append(searchResults)
                    }
                } else {
                    ensureCapacity(&state.queryWorkspace, count: dimensions)
                    state.queryWorkspace.withUnsafeMutableBufferPointer { preparedQuery in
                        if metric.requiresNormalization {
                            storeNormalizedVector(query, into: preparedQuery)
                        } else {
                            storeVector(query, into: preparedQuery)
                        }
                    }
                    let searchResults = state.queryWorkspace.withUnsafeBufferPointer { preparedQuery in
                        searchNormalized(preparedQuery, k: k, state: &state)
                    }
                    results.append(searchResults)
                }
            }
            return results
        }
    }

    public func searchBatch(_ queries: [[Scalar]], k: Int) throws -> [[SearchResult]] {
        guard k > 0 else {
            throw HNSWError.invalidArgument("k must be positive")
        }
        guard !queries.isEmpty else { return [] }
        guard queries.allSatisfy({ $0.count == dimensions }) else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: queries.first?.count ?? 0)
        }
        var results: [[SearchResult]] = []
        results.reserveCapacity(queries.count)
        for query in queries {
            results.append(try search(query, k: k))
        }
        return results
    }
}

#if canImport(FoundationEssentials) || canImport(Foundation)
extension HNSWIndex {

    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        do {
#if os(WASI)
            try Data(serializedArchive().bytes).write(
                to: URL(fileURLWithPath: path)
            )
#else
            try Data(serializedArchive().bytes).write(
                to: URL(fileURLWithPath: path),
                options: .atomic
            )
#endif
        } catch {
            throw HNSWError.saveFailed("Failed to save index to \(path)")
        }
    }

    public static func load(
        from url: URL,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        try load(from: url.path, dimensions: dimensions, metric: metric, maxElements: maxElements)
    }

    public static func load(
        from path: String,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            return try restore(
                from: data,
                dimensions: dimensions,
                metric: metric,
                maxElements: maxElements
            )
        } catch let error as HNSWError {
            throw error
        } catch {
            throw HNSWError.loadFailed("Failed to load index from \(path)")
        }
    }
}

extension Data: HNSWArchiveBytes {}
#endif

extension HNSWIndex {

    public func contains(label: UInt64) -> Bool {
        state.withLock {
            guard let entry = $0.entries[label] else {
                return false
            }
            return !entry.deleted
        }
    }

    public func getVector(label: UInt64) -> [Scalar]? {
        state.withLock {
            guard let entry = $0.entries[label], !entry.deleted else {
                return nil
            }
            let start = entry.offset
            let end = start + dimensions
            if Scalar.self == Float16.self {
                return $0.halfComparisonStorage[start..<end].map(Scalar.init)
            }
            return $0.comparisonStorage[start..<end].map(Scalar.init)
        }
    }

    /// Borrow a stored vector for the duration of the closure.
    ///
    /// Float and Float16 indexes expose the internal contiguous arena without creating an
    /// intermediate array. Other scalar conformances fall back to materialization because
    /// comparison values are stored as Float.
    ///
    /// Safety invariants:
    /// - The local Array snapshot retains the initialized storage for the entire callback.
    /// - Entry offsets are created from complete `dimensions`-sized vectors and remain in bounds.
    /// - Runtime type checks guarantee that raw storage is rebound only to its exact scalar type.
    /// - The buffer is valid only for the synchronous callback and must not escape it.
    /// - All index mutations use the mutex and Array copy-on-write preserves an active snapshot.
    public func withVector<R: Sendable>(
        label: UInt64,
        _ body: @Sendable (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R? {
        if Scalar.self == Float.self {
            let borrowedStorage: (storage: [Float], start: Int)? = state.withLock { state in
                guard let entry = state.entries[label], !entry.deleted else {
                    return nil
                }
                // Array's copy-on-write storage acts as the retained borrow owner. Writers
                // preserve this snapshot by copying before mutation while the body executes.
                return (state.comparisonStorage, entry.offset)
            }
            guard let borrowedStorage else {
                return nil
            }
            return try borrowedStorage.storage.withUnsafeBufferPointer { storage in
                let pointer = UnsafeRawPointer(storage.baseAddress! + borrowedStorage.start)
                    .assumingMemoryBound(to: Scalar.self)
                return try body(UnsafeBufferPointer(start: pointer, count: dimensions))
            }
        }

        if Scalar.self == Float16.self {
            let borrowedStorage: (storage: [Float16], start: Int)? = state.withLock { state in
                guard let entry = state.entries[label], !entry.deleted else {
                    return nil
                }
                // Array's copy-on-write storage acts as the retained borrow owner. Writers
                // preserve this snapshot by copying before mutation while the body executes.
                return (state.halfComparisonStorage, entry.offset)
            }
            guard let borrowedStorage else {
                return nil
            }
            return try borrowedStorage.storage.withUnsafeBufferPointer { storage in
                let pointer = UnsafeRawPointer(storage.baseAddress! + borrowedStorage.start)
                    .assumingMemoryBound(to: Scalar.self)
                return try body(UnsafeBufferPointer(start: pointer, count: dimensions))
            }
        }

        let output: [Scalar]? = state.withLock { state in
            guard let entry = state.entries[label], !entry.deleted else {
                return nil
            }
            let start = entry.offset
            return state.comparisonStorage[start..<(start + dimensions)].map(Scalar.init)
        }
        guard let output else {
            return nil
        }
        return try output.withUnsafeBufferPointer(body)
    }

    public var allLabels: [UInt64] {
        state.withLock { state in
            state.labelOrder.filter { label in
                guard let entry = state.entries[label] else {
                    return false
                }
                return !entry.deleted
            }
        }
    }
}

extension HNSWIndex {

    /// Inspects a graph archive and calculates conservative payload estimates
    /// before restore allocates vector, graph, and workspace storage.
    ///
    /// The archive remains borrowed for the duration of this call. The method
    /// supports the graph archive emitted by `serializedArchive()`; legacy flat
    /// archives remain accepted by `restore` but are not valid persisted graph
    /// snapshots for resource admission.
    public static func inspectArchiveResourceProfile<Archive: HNSWArchiveBytes>(
        from archive: borrowing Archive,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWArchiveResourceProfile {
        do {
            return try archive.withUnsafeBytes { bytes in
                var reader = HNSWArchiveReader(bytes: bytes)
                guard try reader.readMagic() == HNSWArchiveReader.graphMagic,
                      try reader.readUInt32() == 2 else {
                    throw HNSWError.loadFailed(
                        "Unsupported HNSW graph archive"
                    )
                }
                let storedDimensions = try reader.readIntFromUInt32(
                    name: "dimensions"
                )
                guard storedDimensions == dimensions else {
                    throw HNSWError.dimensionMismatch(
                        expected: dimensions,
                        got: storedDimensions
                    )
                }
                let storedCapacity = try reader.readIntFromUInt32(
                    name: "capacity"
                )
                let storedMetric = try reader.readString()
                guard storedMetric == metric.rawValue else {
                    throw HNSWError.loadFailed(
                        "Stored metric \(storedMetric) does not match \(metric.rawValue)"
                    )
                }
                let storedM = try reader.readIntFromUInt32(name: "m")
                let storedEfConstruction = try reader.readIntFromUInt32(
                    name: "efConstruction"
                )
                let storedEfSearch = try reader.readIntFromUInt32(
                    name: "efSearch"
                )
                guard (2...HNSWConfiguration.maximumM).contains(storedM) else {
                    throw HNSWError.loadFailed(
                        "Graph index contains invalid M"
                    )
                }
                guard storedEfConstruction >= storedM,
                      storedEfSearch > 0 else {
                    throw HNSWError.loadFailed(
                        "Graph index contains invalid ef value"
                    )
                }
                _ = try reader.readUInt32() // randomSeed
                _ = try reader.readBool() // allowReplaceDeleted
                _ = try reader.readUInt32() // maxLevel
                _ = try reader.readUInt32() // entryPoint
                _ = try reader.readUInt64() // generatorState
                let labelCount = try reader.readIntFromUInt32(
                    name: "labelCount"
                )
                _ = try validateArchivePayloadCapacity(
                    labelCount: labelCount,
                    dimensions: dimensions,
                    fixedRecordByteCount: 21,
                    remainingByteCount: reader.remainingByteCount,
                    archiveName: "Graph index"
                )

                let vectorByteCount = try resourceProduct(
                    dimensions,
                    MemoryLayout<Float>.stride,
                    message: "Graph archive vector size exceeds the addressable range"
                )
                var activeLabelCount = 0
                var upperLevelSlotCount = 0
                var levelStorageCount = 0
                var neighborCount = 0
                for _ in 0..<labelCount {
                    _ = try reader.readUInt64()
                    if try !reader.readBool() {
                        activeLabelCount = try resourceSum(
                            activeLabelCount,
                            1,
                            message: "Graph archive active label count exceeds the addressable range"
                        )
                    }
                    let level = try reader.readIntFromUInt32(name: "level")
                    guard level >= 0,
                          level < maximumSerializedLevelCount else {
                        throw HNSWError.loadFailed(
                            "Graph index contains invalid level"
                        )
                    }
                    upperLevelSlotCount = try resourceSum(
                        upperLevelSlotCount,
                        level,
                        message: "Graph archive level count exceeds the addressable range"
                    )
                    try reader.skipBytes(count: vectorByteCount)
                    let levelCount = try reader.readIntFromUInt32(
                        name: "levelCount"
                    )
                    guard levelCount == level + 1 else {
                        throw HNSWError.loadFailed(
                            "Graph index level count does not match node level"
                        )
                    }
                    levelStorageCount = try resourceSum(
                        levelStorageCount,
                        levelCount,
                        message: "Graph archive level storage exceeds the addressable range"
                    )
                    for levelIndex in 0..<levelCount {
                        let count = try reader.readIntFromUInt32(
                            name: "neighborCount"
                        )
                        guard count <= max(0, labelCount - 1),
                              count <= maximumConnections(
                                at: levelIndex,
                                m: storedM
                              ) else {
                            throw HNSWError.loadFailed(
                                "Graph index contains too many neighbors"
                            )
                        }
                        neighborCount = try resourceSum(
                            neighborCount,
                            count,
                            message: "Graph archive neighbor count exceeds the addressable range"
                        )
                        try reader.skipBytes(
                            count: try resourceProduct(
                                count,
                                MemoryLayout<UInt32>.stride,
                                message: "Graph archive neighbor storage exceeds the addressable range"
                            )
                        )
                    }
                }
                try reader.ensureFullyRead()

                return try archiveResourceProfile(
                    archiveByteCount: bytes.count,
                    storedCapacity: storedCapacity,
                    requestedCapacity: maxElements,
                    labelCount: labelCount,
                    activeLabelCount: activeLabelCount,
                    upperLevelSlotCount: upperLevelSlotCount,
                    levelStorageCount: levelStorageCount,
                    neighborCount: neighborCount,
                    m: storedM,
                    efSearch: storedEfSearch,
                    dimensions: dimensions
                )
            }
        } catch let error as HNSWError {
            throw error
        } catch {
            throw HNSWError.loadFailed(
                "Failed to inspect HNSW graph archive"
            )
        }
    }

    private static func archiveResourceProfile(
        archiveByteCount: Int,
        storedCapacity: Int,
        requestedCapacity: Int,
        labelCount: Int,
        activeLabelCount: Int,
        upperLevelSlotCount: Int,
        levelStorageCount: Int,
        neighborCount: Int,
        m: Int,
        efSearch: Int,
        dimensions: Int
    ) throws -> HNSWArchiveResourceProfile {
        let scalarByteCount = Scalar.self == Float16.self
            ? MemoryLayout<Float16>.stride
            : MemoryLayout<Float>.stride
        let vectorBytes = try resourceProduct(
            try resourceProduct(
                labelCount,
                dimensions,
                message: "Restored vector count exceeds the addressable range"
            ),
            scalarByteCount,
            message: "Restored vector storage exceeds the addressable range"
        )
        guard let graphBytes = HNSWConnectionStore.retainedByteCount(
            nodeCount: labelCount,
            totalUpperLevelSlotCount: upperLevelSlotCount,
            m: m
        ), let searchBytes = HNSWSearchWorkspace.retainedByteCount(
            candidateCapacity: labelCount,
            nearestCapacity: min(efSearch, labelCount),
            resultCapacity: 1,
            visitedCapacity: labelCount
        ) else {
            throw HNSWError.loadFailed(
                "Restored HNSW payload exceeds the addressable range"
            )
        }
        let queryBytes = try resourceProduct(
            dimensions,
            scalarByteCount,
            message: "Restored query workspace exceeds the addressable range"
        )
        // This allowance covers dictionary buckets, labels, deletion flags,
        // levels, and collection headers retained for every graph node.
        let retainedBookkeepingBytes = try resourceProduct(
            labelCount,
            256,
            message: "Restored node bookkeeping exceeds the addressable range"
        )
        var retainedBytes = 0
        for component in [
            vectorBytes,
            graphBytes,
            searchBytes,
            queryBytes,
            retainedBookkeepingBytes,
        ] {
            retainedBytes = try resourceSum(
                retainedBytes,
                component,
                message: "Restored HNSW payload exceeds the addressable range"
            )
        }

        let capacity = max(1, storedCapacity, requestedCapacity, labelCount)
        let eagerVectorBytes = try resourceProduct(
            try resourceProduct(
                capacity,
                dimensions,
                message: "Eager vector capacity exceeds the addressable range"
            ),
            scalarByteCount,
            message: "Eager vector storage exceeds the addressable range"
        )
        let maximumEagerComparisonBytes = 64 * 1_024 * 1_024
        let eagerAllocationUpperBound = eagerVectorBytes
                <= maximumEagerComparisonBytes
            ? try resourceProduct(
                eagerVectorBytes,
                2,
                message: "Eager vector allocation exceeds the addressable range"
            )
            : 0
        let temporaryLabelBytes = try resourceProduct(
            labelCount,
            256,
            message: "Temporary label storage exceeds the addressable range"
        )
        let temporaryLevelBytes = try resourceProduct(
            levelStorageCount,
            128,
            message: "Temporary level storage exceeds the addressable range"
        )
        let temporaryNeighborBytes = try resourceProduct(
            neighborCount,
            MemoryLayout<Int>.stride,
            message: "Temporary neighbor storage exceeds the addressable range"
        )
        var restoreBytes = retainedBytes
        for component in [
            archiveByteCount,
            eagerAllocationUpperBound,
            temporaryLabelBytes,
            temporaryLevelBytes,
            temporaryNeighborBytes,
        ] {
            restoreBytes = try resourceSum(
                restoreBytes,
                component,
                message: "HNSW restore payload exceeds the addressable range"
            )
        }
        return HNSWArchiveResourceProfile(
            archiveByteCount: archiveByteCount,
            storedCapacity: storedCapacity,
            labelCount: labelCount,
            activeLabelCount: activeLabelCount,
            upperLevelSlotCount: upperLevelSlotCount,
            estimatedRetainedPayloadByteCount: retainedBytes,
            estimatedRestoreWorkingPayloadByteCount: restoreBytes
        )
    }

    private static func resourceSum(
        _ lhs: Int,
        _ rhs: Int,
        message: String
    ) throws -> Int {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw HNSWError.loadFailed(message)
        }
        return result
    }

    private static func resourceProduct(
        _ lhs: Int,
        _ rhs: Int,
        message: String
    ) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw HNSWError.loadFailed(message)
        }
        return result
    }

    public func serializedArchive() throws -> HNSWArchive {
        state.withLock {
            var writer = HNSWArchiveWriter()
            writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
            writer.writeUInt32(2)
            writer.writeUInt32(UInt32(dimensions))
            writer.writeUInt32(UInt32($0.maximumElementCount))
            writer.writeString(metric.rawValue)
            writer.writeUInt32(UInt32(configuration.m))
            writer.writeUInt32(UInt32(configuration.efConstruction))
            writer.writeUInt32(UInt32($0.efSearch))
            writer.writeUInt32(UInt32(bitPattern: Int32(configuration.randomSeed)))
            writer.writeBool(configuration.allowReplaceDeleted)
            writer.writeUInt32(UInt32(bitPattern: Int32($0.maxLevel)))
            writer.writeUInt32($0.entryPoint ?? UInt32.max)
            writer.writeUInt64($0.levelGenerator.currentState)
            writer.writeUInt32(UInt32($0.labelOrder.count))
            for internalID in $0.labelOrder.indices {
                let label = $0.labelOrder[internalID]
                guard let entry = $0.entries[label] else {
                    continue
                }
                writer.writeUInt64(label)
                writer.writeBool(entry.deleted)
                writer.writeUInt32(UInt32(entry.level))
                let start = entry.offset
                let end = start + dimensions
                if Scalar.self == Float16.self {
                    for value in $0.halfComparisonStorage[start..<end] {
                        writer.writeFloat(Float(value))
                    }
                } else {
                    for value in $0.comparisonStorage[start..<end] {
                        writer.writeFloat(value)
                    }
                }
                let typedInternalID = HNSWInternalID(internalID)
                let levelCount = $0.connections.levelCount(for: typedInternalID)
                writer.writeUInt32(UInt32(levelCount))
                for level in 0..<levelCount {
                    let neighborRange = $0.connections.neighborStorageRange(for: typedInternalID, at: level)
                    writer.writeUInt32(UInt32(neighborRange.count))
                    for neighborStorageIndex in neighborRange {
                        let neighbor = $0.connections.neighborInStorage(at: neighborStorageIndex, level: level)
                        writer.writeUInt32(neighbor)
                    }
                }
            }
            return HNSWArchive(bytes: writer.bytes)
        }
    }

    public static func restore<Archive: HNSWArchiveBytes>(
        from archive: borrowing Archive,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        do {
            return try archive.withUnsafeBytes { bytes in
                var reader = HNSWArchiveReader(bytes: bytes)
                let magic = try reader.readMagic()
                let version = try reader.readUInt32()
                switch magic {
                case HNSWArchiveReader.graphMagic:
                    return try loadGraph(
                        reader: &reader,
                        version: version,
                        dimensions: dimensions,
                        metric: metric,
                        maxElements: maxElements
                    )
                case HNSWArchiveReader.flatMagic:
                    return try loadFlat(
                        reader: &reader,
                        version: version,
                        dimensions: dimensions,
                        metric: metric,
                        maxElements: maxElements
                    )
                default:
                    throw HNSWError.loadFailed("Invalid HNSW index archive magic")
                }
            }
        } catch let error as HNSWError {
            throw error
        } catch {
            throw HNSWError.loadFailed("Failed to load index from data")
        }
    }

    private static func loadGraph(
        reader: inout HNSWArchiveReader,
        version: UInt32,
        dimensions: Int,
        metric: DistanceMetric,
        maxElements: Int
    ) throws -> HNSWIndex {
        guard version == 2 else {
            throw HNSWError.loadFailed("Unsupported graph index version")
        }
        let storedDimensions = try reader.readIntFromUInt32(name: "dimensions")
        guard storedDimensions == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: storedDimensions)
        }
        let storedCapacity = try reader.readIntFromUInt32(name: "capacity")
        let storedMetric = try reader.readString()
        guard storedMetric == metric.rawValue else {
            throw HNSWError.loadFailed("Stored metric \(storedMetric) does not match \(metric.rawValue)")
        }

        let storedM = try reader.readIntFromUInt32(name: "m")
        let storedEfConstruction = try reader.readIntFromUInt32(name: "efConstruction")
        let storedEfSearch = try reader.readIntFromUInt32(name: "efSearch")
        guard (2...HNSWConfiguration.maximumM).contains(storedM) else {
            throw HNSWError.loadFailed("Graph index contains invalid M")
        }
        guard storedEfConstruction >= storedM, storedEfSearch > 0 else {
            throw HNSWError.loadFailed("Graph index contains invalid ef value")
        }
        let storedSeed = Int(Int32(bitPattern: try reader.readUInt32()))
        let storedAllowReplaceDeleted = try reader.readBool()
        let storedMaxLevel = Int(Int32(bitPattern: try reader.readUInt32()))
        let storedEntryPoint = try reader.readUInt32()
        let generatorState = try reader.readUInt64()
        let labelCount = try reader.readIntFromUInt32(name: "labelCount")
        let vectorValueCount = try validateArchivePayloadCapacity(
            labelCount: labelCount,
            dimensions: dimensions,
            fixedRecordByteCount: 21,
            remainingByteCount: reader.remainingByteCount,
            archiveName: "Graph index"
        )
        try validateArchiveRetainedCapacity(
            labelCount: labelCount,
            totalUpperLevelSlotCount: 0,
            m: storedM,
            efSearch: storedEfSearch,
            archiveName: "Graph index"
        )

        let capacity = max(maxElements, storedCapacity, labelCount)
        let index = try HNSWIndex(
            dimensions: dimensions,
            maxElements: max(1, capacity),
            metric: metric,
            configuration: HNSWConfiguration(
                m: storedM,
                efConstruction: storedEfConstruction,
                efSearch: storedEfSearch,
                randomSeed: storedSeed,
                allowReplaceDeleted: storedAllowReplaceDeleted
            )
        )

        var loadedEntries: [UInt64: Entry] = [:]
        var loadedLabelOrder: [UInt64] = []
        var loadedDeletedFlags: [UInt8] = []
        var loadedLevels: [Int] = []
        var loadedComparisonStorage: [Float] = []
        var loadedHalfComparisonStorage: [Float16] = []
        var loadedConnections: [[[Int]]] = []
        loadedEntries.reserveCapacity(labelCount)
        loadedLabelOrder.reserveCapacity(labelCount)
        loadedDeletedFlags.reserveCapacity(labelCount)
        loadedLevels.reserveCapacity(labelCount)
        if Scalar.self == Float16.self {
            loadedHalfComparisonStorage.reserveCapacity(vectorValueCount)
        } else {
            loadedComparisonStorage.reserveCapacity(vectorValueCount)
        }
        loadedConnections.reserveCapacity(labelCount)
        var totalUpperLevelSlotCount = 0

        for internalID in 0..<labelCount {
            let label = try reader.readUInt64()
            guard loadedEntries[label] == nil else {
                throw HNSWError.loadFailed("Graph index contains duplicate labels")
            }
            let deleted = try reader.readBool()
            let level = try reader.readIntFromUInt32(name: "level")
            guard level >= 0, level < Self.maximumSerializedLevelCount else {
                throw HNSWError.loadFailed("Graph index contains invalid level")
            }
            let (nextUpperLevelSlotCount, upperLevelCountOverflowed) =
                totalUpperLevelSlotCount.addingReportingOverflow(level)
            guard !upperLevelCountOverflowed else {
                throw HNSWError.loadFailed(
                    "Graph index retained graph storage exceeds the addressable range"
                )
            }
            try validateArchiveRetainedCapacity(
                labelCount: labelCount,
                totalUpperLevelSlotCount: nextUpperLevelSlotCount,
                m: storedM,
                efSearch: storedEfSearch,
                archiveName: "Graph index"
            )
            totalUpperLevelSlotCount = nextUpperLevelSlotCount
            let offset = Scalar.self == Float16.self ? loadedHalfComparisonStorage.count : loadedComparisonStorage.count
            try appendValidatedArchiveVector(
                reader: &reader,
                dimensions: dimensions,
                metric: metric,
                archiveName: "Graph index",
                floatStorage: &loadedComparisonStorage,
                halfStorage: &loadedHalfComparisonStorage
            )

            let levelCount = try reader.readIntFromUInt32(name: "levelCount")
            guard levelCount == level + 1 else {
                throw HNSWError.loadFailed("Graph index level count does not match node level")
            }
            var nodeConnections: [[Int]] = []
            nodeConnections.reserveCapacity(levelCount)
            for levelIndex in 0..<levelCount {
                let neighborCount = try reader.readIntFromUInt32(name: "neighborCount")
                guard neighborCount <= max(0, labelCount - 1) else {
                    throw HNSWError.loadFailed("Graph index contains too many neighbors")
                }
                guard neighborCount <= Self.maximumConnections(at: levelIndex, m: storedM) else {
                    throw HNSWError.loadFailed("Graph index exceeds configured connection limit")
                }
                var neighbors: [Int] = []
                neighbors.reserveCapacity(neighborCount)
                for _ in 0..<neighborCount {
                    let neighbor = try reader.readIntFromUInt32(name: "neighbor")
                    guard neighbor >= 0, neighbor < labelCount else {
                        throw HNSWError.loadFailed("Graph index contains invalid neighbor id")
                    }
                    neighbors.append(neighbor)
                }
                nodeConnections.append(neighbors)
            }

            loadedEntries[label] = Entry(
                internalID: HNSWInternalID(internalID),
                offset: offset,
                deleted: deleted,
                level: level
            )
            loadedLabelOrder.append(label)
            loadedDeletedFlags.append(deleted ? 1 : 0)
            loadedLevels.append(level)
            loadedConnections.append(nodeConnections)
        }
        try reader.ensureFullyRead()

        let entryPoint: HNSWInternalID?
        if storedEntryPoint == UInt32.max {
            entryPoint = nil
        } else {
            let value = Int(storedEntryPoint)
            guard value >= 0, value < labelCount else {
                throw HNSWError.loadFailed("Graph index contains invalid entry point")
            }
            entryPoint = storedEntryPoint
        }
        try validateLoadedGraph(
            entries: loadedEntries,
            labelOrder: loadedLabelOrder,
            connections: loadedConnections,
            entryPoint: entryPoint.map(Int.init),
            maxLevel: storedMaxLevel,
            m: storedM
        )

        index.state.withLock {
            $0.maximumElementCount = max(1, capacity)
            $0.efSearch = storedEfSearch
            $0.entries = loadedEntries
            $0.labelOrder = loadedLabelOrder
            $0.deletedFlags = loadedDeletedFlags
            $0.levels = loadedLevels
            $0.liveCount = loadedEntries.values.reduce(0) { count, entry in
                entry.deleted ? count : count + 1
            }
            $0.comparisonStorage = loadedComparisonStorage
            $0.halfComparisonStorage = loadedHalfComparisonStorage
            $0.connections.replaceAll(with: loadedConnections)
            $0.entryPoint = entryPoint
            $0.maxLevel = storedMaxLevel
            $0.levelGenerator = HNSWLevelGenerator(state: generatorState)
            $0.visited = [UInt16](repeating: 0, count: labelCount)
            $0.visitedTag = 0
            $0.searchWorkspace.prepare(
                candidateCapacity: labelCount,
                nearestCapacity: min(storedEfSearch, labelCount),
                resultCapacity: 1,
                visitedCapacity: labelCount
            )
            $0.queryWorkspace = Scalar.self == Float16.self ? [] : [Float](repeating: 0, count: dimensions)
            $0.halfPrecisionQueryWorkspace = Scalar.self == Float16.self ? [Float16](repeating: 0, count: dimensions) : []
        }
        return index
    }

    private static var maximumSerializedLevelCount: Int { 64 }

    private static func validateLoadedGraph(
        entries: [UInt64: Entry],
        labelOrder: [UInt64],
        connections: [[[Int]]],
        entryPoint: Int?,
        maxLevel: Int,
        m: Int
    ) throws {
        guard entries.count == labelOrder.count, connections.count == labelOrder.count else {
            throw HNSWError.loadFailed("Graph index metadata is inconsistent")
        }

        guard !labelOrder.isEmpty else {
            guard entryPoint == nil, maxLevel == -1 else {
                throw HNSWError.loadFailed("Empty graph index has invalid entry point")
            }
            return
        }

        guard let entryPoint else {
            throw HNSWError.loadFailed("Graph index is missing entry point")
        }
        guard entryPoint >= 0, entryPoint < labelOrder.count else {
            throw HNSWError.loadFailed("Graph index contains invalid entry point")
        }

        var observedMaxLevel = -1
        for internalID in labelOrder.indices {
            let label = labelOrder[internalID]
            guard let entry = entries[label], entry.internalID == internalID else {
                throw HNSWError.loadFailed("Graph index entry id is inconsistent")
            }
            guard entry.level >= 0, entry.level < maximumSerializedLevelCount else {
                throw HNSWError.loadFailed("Graph index contains invalid level")
            }
            guard connections[internalID].count == entry.level + 1 else {
                throw HNSWError.loadFailed("Graph index connection level count is inconsistent")
            }
            observedMaxLevel = max(observedMaxLevel, entry.level)

            for level in connections[internalID].indices {
                let neighbors = connections[internalID][level]
                guard neighbors.count <= maximumConnections(at: level, m: m) else {
                    throw HNSWError.loadFailed("Graph index exceeds configured connection limit")
                }

                var seenNeighbors = Set<Int>()
                seenNeighbors.reserveCapacity(neighbors.count)
                for neighborID in neighbors {
                    guard neighborID >= 0, neighborID < labelOrder.count else {
                        throw HNSWError.loadFailed("Graph index contains invalid neighbor id")
                    }
                    guard neighborID != internalID else {
                        throw HNSWError.loadFailed("Graph index contains self edge")
                    }
                    guard connections[neighborID].indices.contains(level) else {
                        throw HNSWError.loadFailed("Graph index references missing neighbor level")
                    }
                    guard seenNeighbors.insert(neighborID).inserted else {
                        throw HNSWError.loadFailed("Graph index contains duplicate neighbor")
                    }
                }
            }
        }

        guard observedMaxLevel == maxLevel else {
            throw HNSWError.loadFailed("Graph index max level is inconsistent")
        }
        guard let entry = entries[labelOrder[entryPoint]], entry.level == maxLevel else {
            throw HNSWError.loadFailed("Graph index entry point level is inconsistent")
        }
    }

    private static func maximumConnections(at level: Int, m: Int) -> Int {
        level == 0 ? max(1, m * 2) : max(1, m)
    }

    private static func appendValidatedArchiveVector(
        reader: inout HNSWArchiveReader,
        dimensions: Int,
        metric: DistanceMetric,
        archiveName: String,
        floatStorage: inout [Float],
        halfStorage: inout [Float16]
    ) throws {
        var squaredNorm: Float = 0
        for _ in 0..<dimensions {
            let archiveValue = try reader.readFloat()
            guard archiveValue.isFinite else {
                throw HNSWError.loadFailed(
                    "\(archiveName) contains non-finite vector values"
                )
            }

            let storedValue: Float
            if Scalar.self == Float16.self {
                let halfValue = Float16(archiveValue)
                guard halfValue.isFinite else {
                    throw HNSWError.loadFailed(
                        "\(archiveName) contains values outside the Float16 range"
                    )
                }
                halfStorage.append(halfValue)
                storedValue = Float(halfValue)
            } else {
                floatStorage.append(archiveValue)
                storedValue = archiveValue
            }

            if metric.requiresNormalization {
                squaredNorm += storedValue * storedValue
            }
        }

        guard !metric.requiresNormalization
                || (squaredNorm > 0 && squaredNorm.isFinite) else {
            throw HNSWError.loadFailed(
                "\(archiveName) contains an invalid cosine vector norm"
            )
        }
    }

    private static func loadFlat(
        reader: inout HNSWArchiveReader,
        version: UInt32,
        dimensions: Int,
        metric: DistanceMetric,
        maxElements: Int
    ) throws -> HNSWIndex {
        guard version == 1 else {
            throw HNSWError.loadFailed("Unsupported flat index version")
        }
        let storedDimensions = try reader.readIntFromUInt32(name: "dimensions")
        guard storedDimensions == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: storedDimensions)
        }
        let storedCapacity = try reader.readIntFromUInt32(name: "capacity")
        let storedMetric = try reader.readString()
        guard storedMetric == metric.rawValue else {
            throw HNSWError.loadFailed("Stored metric \(storedMetric) does not match \(metric.rawValue)")
        }

        let labelCount = try reader.readIntFromUInt32(name: "labelCount")
        let vectorValueCount = try validateArchivePayloadCapacity(
            labelCount: labelCount,
            dimensions: dimensions,
            fixedRecordByteCount: 9,
            remainingByteCount: reader.remainingByteCount,
            archiveName: "Flat index"
        )
        try validateArchiveRetainedCapacity(
            labelCount: labelCount,
            totalUpperLevelSlotCount: 0,
            m: HNSWConfiguration.balanced.m,
            efSearch: HNSWConfiguration.balanced.efSearch,
            archiveName: "Flat index"
        )
        let capacity = max(maxElements, storedCapacity, labelCount)
        let index = try HNSWIndex(
            dimensions: dimensions,
            maxElements: max(1, capacity),
            metric: metric,
            configuration: .balanced
        )
        var loadedEntries: [UInt64: Entry] = [:]
        var loadedLabelOrder: [UInt64] = []
        var loadedDeletedFlags: [UInt8] = []
        var loadedLevels: [Int] = []
        var loadedComparisonStorage: [Float] = []
        var loadedHalfComparisonStorage: [Float16] = []
        var loadedConnections: [[[Int]]] = []
        loadedEntries.reserveCapacity(labelCount)
        loadedLabelOrder.reserveCapacity(labelCount)
        loadedDeletedFlags.reserveCapacity(labelCount)
        loadedLevels.reserveCapacity(labelCount)
        if Scalar.self == Float16.self {
            loadedHalfComparisonStorage.reserveCapacity(vectorValueCount)
        } else {
            loadedComparisonStorage.reserveCapacity(vectorValueCount)
        }
        loadedConnections.reserveCapacity(labelCount)

        var generator = HNSWLevelGenerator(seed: HNSWConfiguration.balanced.randomSeed)
        let multiplier = 1.0 / hnswNaturalLog(
            Double(max(2, HNSWConfiguration.balanced.m))
        )
        var totalUpperLevelSlotCount = 0
        for internalID in 0..<labelCount {
            let label = try reader.readUInt64()
            guard loadedEntries[label] == nil else {
                throw HNSWError.loadFailed("Flat index contains duplicate labels")
            }
            let deleted = try reader.readBool()
            let offset = Scalar.self == Float16.self ? loadedHalfComparisonStorage.count : loadedComparisonStorage.count
            try appendValidatedArchiveVector(
                reader: &reader,
                dimensions: dimensions,
                metric: metric,
                archiveName: "Flat index",
                floatStorage: &loadedComparisonStorage,
                halfStorage: &loadedHalfComparisonStorage
            )
            let level = generator.randomLevel(multiplier: multiplier)
            let (nextUpperLevelSlotCount, upperLevelCountOverflowed) =
                totalUpperLevelSlotCount.addingReportingOverflow(level)
            guard !upperLevelCountOverflowed else {
                throw HNSWError.loadFailed(
                    "Flat index retained graph storage exceeds the addressable range"
                )
            }
            try validateArchiveRetainedCapacity(
                labelCount: labelCount,
                totalUpperLevelSlotCount: nextUpperLevelSlotCount,
                m: HNSWConfiguration.balanced.m,
                efSearch: HNSWConfiguration.balanced.efSearch,
                archiveName: "Flat index"
            )
            totalUpperLevelSlotCount = nextUpperLevelSlotCount
            loadedEntries[label] = Entry(
                internalID: HNSWInternalID(internalID),
                offset: offset,
                deleted: deleted,
                level: level
            )
            loadedLabelOrder.append(label)
            loadedDeletedFlags.append(deleted ? 1 : 0)
            loadedLevels.append(level)
            loadedConnections.append(Array(repeating: [], count: level + 1))
        }
        try reader.ensureFullyRead()

        index.state.withLock {
            $0.entries = loadedEntries
            $0.labelOrder = loadedLabelOrder
            $0.deletedFlags = loadedDeletedFlags
            $0.levels = loadedLevels
            $0.liveCount = loadedEntries.values.reduce(0) { count, entry in
                entry.deleted ? count : count + 1
            }
            $0.comparisonStorage = loadedComparisonStorage
            $0.halfComparisonStorage = loadedHalfComparisonStorage
            $0.connections.replaceAll(with: loadedConnections)
            $0.levelGenerator = generator
            $0.visited = [UInt16](repeating: 0, count: labelCount)
            $0.visitedTag = 0
            $0.searchWorkspace.prepare(
                candidateCapacity: labelCount,
                nearestCapacity: min(HNSWConfiguration.balanced.efSearch, labelCount),
                resultCapacity: 1,
                visitedCapacity: labelCount
            )
            $0.queryWorkspace = Scalar.self == Float16.self ? [] : [Float](repeating: 0, count: dimensions)
            $0.halfPrecisionQueryWorkspace = Scalar.self == Float16.self ? [Float16](repeating: 0, count: dimensions) : []
            index.rebuildGraph(state: &$0)
        }
        return index
    }

    private static func validateArchivePayloadCapacity(
        labelCount: Int,
        dimensions: Int,
        fixedRecordByteCount: Int,
        remainingByteCount: Int,
        archiveName: String
    ) throws -> Int {
        guard labelCount == 0 || labelCount <= Int.max / dimensions else {
            throw HNSWError.loadFailed(
                "\(archiveName) vector storage exceeds the addressable range"
            )
        }
        guard dimensions <= (Int.max - fixedRecordByteCount)
                / MemoryLayout<Float>.stride else {
            throw HNSWError.loadFailed(
                "\(archiveName) record size exceeds the addressable range"
            )
        }
        let minimumRecordByteCount = fixedRecordByteCount
            + dimensions * MemoryLayout<Float>.stride
        guard labelCount == 0
                || labelCount <= remainingByteCount / minimumRecordByteCount else {
            throw HNSWError.loadFailed(
                "\(archiveName) payload is too small for its declared label count"
            )
        }
        return labelCount * dimensions
    }

    private static func validateArchiveRetainedCapacity(
        labelCount: Int,
        totalUpperLevelSlotCount: Int,
        m: Int,
        efSearch: Int,
        archiveName: String
    ) throws {
        guard let graphBytes = HNSWConnectionStore.retainedByteCount(
            nodeCount: labelCount,
            totalUpperLevelSlotCount: totalUpperLevelSlotCount,
            m: m
        ) else {
            throw HNSWError.loadFailed(
                "\(archiveName) retained graph storage exceeds the addressable range"
            )
        }
        guard graphBytes <= HNSWConnectionStore.maximumRetainedByteCount else {
            throw HNSWError.loadFailed(
                "\(archiveName) retained graph storage exceeds the 256 MiB limit"
            )
        }
        guard let searchBytes = HNSWSearchWorkspace.retainedByteCount(
            candidateCapacity: labelCount,
            nearestCapacity: min(efSearch, labelCount),
            resultCapacity: 1,
            visitedCapacity: labelCount
        ) else {
            throw HNSWError.loadFailed(
                "\(archiveName) search workspace exceeds the addressable range"
            )
        }
        guard searchBytes <= HNSWSearchWorkspace.maximumRetainedByteCount else {
            throw HNSWError.loadFailed(
                "\(archiveName) search workspace exceeds the 256 MiB limit"
            )
        }
    }
}

extension HNSWIndex {

    @inline(__always)
    private func validateDimensions(_ got: Int) throws {
        guard got == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: got)
        }
    }

    @inline(__always)
    private func validateDimensions(_ got: Int, expectedTotal: Int) throws {
        guard got == expectedTotal else {
            throw HNSWError.dimensionMismatch(expected: expectedTotal, got: got)
        }
    }

    @inline(__always)
    private func validateInputVector(
        _ vector: UnsafeBufferPointer<Scalar>,
        name: String
    ) throws -> Float? {
        let containsOnlyFiniteValues: Bool
        if Scalar.self == Float.self {
            // Runtime type identity preserves the binding, and the typed view remains
            // inside the caller-owned buffer's synchronous borrow.
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(vector.baseAddress!)
                    .assumingMemoryBound(to: Float.self),
                count: vector.count
            )
            containsOnlyFiniteValues = VectorOperations
                .containsOnlyFiniteValues(inputFloats)
        } else if Scalar.self == Float16.self {
            // Runtime type identity preserves the binding, and the typed view remains
            // inside the caller-owned buffer's synchronous borrow.
            let inputHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(vector.baseAddress!)
                    .assumingMemoryBound(to: Float16.self),
                count: vector.count
            )
            containsOnlyFiniteValues = VectorOperations
                .containsOnlyFiniteValues(inputHalves)
        } else {
            containsOnlyFiniteValues = vector.allSatisfy {
                $0.hnswFloatValue.isFinite
            }
        }
        guard containsOnlyFiniteValues else {
            throw HNSWError.invalidArgument(
                "\(name) must contain only finite values"
            )
        }

        guard metric.requiresNormalization else {
            return nil
        }

        let squaredNorm: Float
        if Scalar.self == Float.self {
            // Runtime type identity preserves the binding, and the typed view remains
            // inside the caller-owned buffer's synchronous borrow.
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(vector.baseAddress!)
                    .assumingMemoryBound(to: Float.self),
                count: vector.count
            )
            squaredNorm = VectorOperations.squaredMagnitude(inputFloats)
        } else if Scalar.self == Float16.self {
            // Runtime type identity preserves the binding, and the typed view remains
            // inside the caller-owned buffer's synchronous borrow.
            let inputHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(vector.baseAddress!)
                    .assumingMemoryBound(to: Float16.self),
                count: vector.count
            )
            squaredNorm = VectorOperations.squaredMagnitude(inputHalves)
        } else {
            var genericSquaredNorm: Float = 0
            for value in vector {
                let floatValue = value.hnswFloatValue
                genericSquaredNorm += floatValue * floatValue
            }
            squaredNorm = genericSquaredNorm
        }
        guard squaredNorm > 0 else {
            throw HNSWError.invalidArgument("\(name) must have nonzero norm")
        }
        guard squaredNorm.isFinite else {
            throw HNSWError.invalidArgument("\(name) norm must be finite")
        }
        return 1 / squaredNorm.squareRoot()
    }

    @inline(__always)
    private func ensureCapacity(_ buffer: inout [Float], count: Int) {
        guard buffer.count != count else { return }
        buffer = [Float](repeating: 0, count: count)
    }

    @inline(__always)
    private func ensureCapacity(_ buffer: inout [Float16], count: Int) {
        guard buffer.count != count else { return }
        buffer = [Float16](repeating: 0, count: count)
    }

    private func storeVector(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            VectorOperations.copy(inputFloats, into: output)
        } else {
            for index in 0..<input.count {
                output[index] = input[index].hnswFloatValue
            }
        }
    }

    private func storeNormalizedVector(
        _ input: UnsafeBufferPointer<Scalar>,
        normalizationScale: Float? = nil,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            if let normalizationScale {
                VectorOperations.scale(
                    inputFloats,
                    by: normalizationScale,
                    into: output
                )
            } else {
                VectorOperations.normalize(inputFloats, into: output)
            }
        } else {
            storeVector(input, into: output)
            let outputSource = UnsafeBufferPointer(start: output.baseAddress!, count: output.count)
            VectorOperations.normalize(outputSource, into: output)
        }
    }

    private func storeHalfVector(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Float16>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        if Scalar.self == Float16.self {
            let inputHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: input.count
            )
            VectorOperations.copy(inputHalves, into: output)
        } else {
            for index in 0..<input.count {
                output[index] = Float16(input[index].hnswFloatValue)
            }
        }
    }

    private func storeNormalizedHalfVector(
        _ input: UnsafeBufferPointer<Scalar>,
        normalizationScale: Float? = nil,
        into output: UnsafeMutableBufferPointer<Float16>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        if Scalar.self == Float16.self {
            let inputHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: input.count
            )
            if let normalizationScale {
                VectorOperations.scale(
                    inputHalves,
                    by: normalizationScale,
                    into: output
                )
            } else {
                VectorOperations.normalize(inputHalves, into: output)
            }
        } else {
            storeHalfVector(input, into: output)
            let outputSource = UnsafeBufferPointer(start: output.baseAddress!, count: output.count)
            VectorOperations.normalize(outputSource, into: output)
        }
    }

    private func upsertVector(
        _ vector: UnsafeBufferPointer<Scalar>,
        label: UInt64,
        normalizationScale: Float? = nil,
        state: inout State
    ) throws {
        let entry: Entry
        let shouldRebuildGraph: Bool
        if let existing = state.entries[label] {
            entry = Entry(
                internalID: existing.internalID,
                offset: existing.offset,
                deleted: false,
                level: existing.level
            )
            if existing.deleted {
                state.liveCount += 1
            }
            let existingIndex = Int(existing.internalID)
            if existingIndex < state.deletedFlags.count {
                state.deletedFlags[existingIndex] = 0
            }
            shouldRebuildGraph = state.connections.hasAnyConnection(for: existing.internalID)
        } else {
            if state.entries.count < state.maximumElementCount {
                let nextNodeCount = state.labelOrder.count + 1
                let constructionCapacity = min(
                    configuration.efConstruction,
                    nextNodeCount
                )
                try HNSWSearchWorkspace.validateRetainedCapacity(
                    candidateCapacity: nextNodeCount,
                    nearestCapacity: constructionCapacity,
                    resultCapacity: 1,
                    visitedCapacity: nextNodeCount
                )
                let internalID = HNSWInternalID(state.labelOrder.count)
                let offset = Scalar.self == Float16.self
                    ? state.halfComparisonStorage.count
                    : state.comparisonStorage.count
                guard offset <= Int.max - dimensions else {
                    throw HNSWError.addPointFailed(
                        "Vector storage exceeds the addressable range"
                    )
                }
                var levelGenerator = state.levelGenerator
                let level = levelGenerator.randomLevel(multiplier: levelMultiplier)
                try validateGraphRetainedCapacity(
                    nodeCount: nextNodeCount,
                    additionalUpperLevelSlotCount: level,
                    state: state
                )
                state.levelGenerator = levelGenerator
                entry = Entry(internalID: internalID, offset: offset, deleted: false, level: level)
                if Scalar.self == Float16.self {
                    for _ in 0..<dimensions {
                        state.halfComparisonStorage.append(.zero)
                    }
                } else {
                    for _ in 0..<dimensions {
                        state.comparisonStorage.append(.zero)
                    }
                }
                state.labelOrder.append(label)
                state.deletedFlags.append(0)
                state.levels.append(level)
                state.connections.appendNode(level: level)
                state.visited.append(0)
                state.liveCount += 1
                shouldRebuildGraph = false
            } else if configuration.allowReplaceDeleted,
                      let reusable = reusableDeletedEntry(state: state) {
                var levelGenerator = state.levelGenerator
                let level = levelGenerator.randomLevel(multiplier: levelMultiplier)
                let oldLevelCount = state.connections.levelCount(
                    for: reusable.entry.internalID
                )
                let additionalUpperLevelSlotCount = level + 1 > oldLevelCount
                    ? level
                    : 0
                try validateGraphRetainedCapacity(
                    nodeCount: state.labelOrder.count,
                    additionalUpperLevelSlotCount: additionalUpperLevelSlotCount,
                    state: state
                )
                state.levelGenerator = levelGenerator
                entry = Entry(
                    internalID: reusable.entry.internalID,
                    offset: reusable.entry.offset,
                    deleted: false,
                    level: level
                )
                let reusableIndex = Int(reusable.entry.internalID)
                state.entries.removeValue(forKey: reusable.label)
                state.labelOrder[reusableIndex] = label
                state.deletedFlags[reusableIndex] = 0
                state.levels[reusableIndex] = level
                state.connections.resetNode(reusable.entry.internalID, level: level)
                state.liveCount += 1
                shouldRebuildGraph = true
            } else {
                throw HNSWError.capacityExceeded(
                    current: state.entries.count,
                    maximum: state.maximumElementCount
                )
            }
        }

        if Scalar.self == Float16.self {
            state.halfComparisonStorage.withUnsafeMutableBufferPointer { storage in
                let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                if metric.requiresNormalization {
                    storeNormalizedHalfVector(
                        vector,
                        normalizationScale: normalizationScale,
                        into: destination
                    )
                } else {
                    storeHalfVector(vector, into: destination)
                }
            }
        } else {
            state.comparisonStorage.withUnsafeMutableBufferPointer { storage in
                let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                if metric.requiresNormalization {
                    storeNormalizedVector(
                        vector,
                        normalizationScale: normalizationScale,
                        into: destination
                    )
                } else {
                    storeVector(vector, into: destination)
                }
            }
        }
        state.entries[label] = entry

        if shouldRebuildGraph {
            rebuildGraph(state: &state)
        } else {
            connectNewElement(entry.internalID, state: &state)
        }
    }

    private func validateGraphRetainedCapacity(
        nodeCount: Int,
        additionalUpperLevelSlotCount: Int,
        state: borrowing State
    ) throws {
        let (upperLevelSlotCount, upperLevelCountOverflowed) =
            state.connections.totalUpperLevelSlotCount.addingReportingOverflow(
                additionalUpperLevelSlotCount
            )
        guard !upperLevelCountOverflowed,
              let retainedByteCount = HNSWConnectionStore.retainedByteCount(
                  nodeCount: nodeCount,
                  totalUpperLevelSlotCount: upperLevelSlotCount,
                  m: configuration.m
              ) else {
            throw HNSWError.addPointFailed(
                "Graph storage exceeds the addressable range"
            )
        }
        guard retainedByteCount <= HNSWConnectionStore.maximumRetainedByteCount else {
            throw HNSWError.addPointFailed(
                "Graph storage exceeds the 256 MiB limit"
            )
        }
    }

    private func validateSearchWorkspace(k: Int, state: borrowing State) throws {
        let nodeCount = state.labelOrder.count
        try HNSWSearchWorkspace.validateRetainedCapacity(
            candidateCapacity: nodeCount,
            nearestCapacity: min(max(state.efSearch, k), nodeCount),
            resultCapacity: min(k, nodeCount),
            visitedCapacity: nodeCount
        )
    }

    private func searchNormalized(
        _ query: UnsafeBufferPointer<Float>,
        k: Int,
        state: inout State
    ) -> [SearchResult] {
        guard let entryPoint = state.entryPoint else { return [] }
        var visitedTag = state.visitedTag
        let nodeCount = state.labelOrder.count
        let effectiveEF = max(state.efSearch, k)
        let candidateCapacity = max(1, nodeCount)
        let nearestCapacity = max(1, min(effectiveEF, nodeCount))
        let resultCapacity = max(1, min(k, nodeCount))
        let results = Array<SearchResult>(unsafeUninitializedCapacity: resultCapacity) { output, initializedCount in
            initializedCount = state.searchWorkspace.withCandidateBuffers(
                candidateCapacity: candidateCapacity,
                nearestCapacity: nearestCapacity,
                resultCapacity: resultCapacity,
                visitedCapacity: nodeCount
            ) { candidateQueueStorage, nearestCandidateStorage, resultCandidateStorage in
                state.visited.withUnsafeMutableBufferPointer { visited in
                    state.comparisonStorage.withUnsafeBufferPointer { storage in
                        switch metric {
                        case .l2:
                            searchPrepared(
                                query,
                                k: k,
                                into: output,
                                entryPoint: entryPoint,
                                storage: storage,
                                connections: state.connections,
                                labelOrder: state.labelOrder,
                                deletedFlags: state.deletedFlags,
                                liveCount: state.liveCount,
                                maxLevel: state.maxLevel,
                                efSearch: state.efSearch,
                                visited: visited,
                                visitedTag: &visitedTag,
                                candidateQueueStorage: candidateQueueStorage,
                                nearestCandidateStorage: nearestCandidateStorage,
                                resultCandidateStorage: resultCandidateStorage,
                                distanceComputer: HNSWDistanceComputers.L2.self
                            )
                        case .innerProduct, .cosine:
                            searchPrepared(
                                query,
                                k: k,
                                into: output,
                                entryPoint: entryPoint,
                                storage: storage,
                                connections: state.connections,
                                labelOrder: state.labelOrder,
                                deletedFlags: state.deletedFlags,
                                liveCount: state.liveCount,
                                maxLevel: state.maxLevel,
                                efSearch: state.efSearch,
                                visited: visited,
                                visitedTag: &visitedTag,
                                candidateQueueStorage: candidateQueueStorage,
                                nearestCandidateStorage: nearestCandidateStorage,
                                resultCandidateStorage: resultCandidateStorage,
                                distanceComputer: HNSWDistanceComputers.InnerProduct.self
                            )
                        }
                    }
                }
            }
        }
        state.visitedTag = visitedTag
        return results
    }

    private func searchNormalized(
        _ query: UnsafeBufferPointer<Float>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>,
        state: inout State
    ) -> Int {
        guard let entryPoint = state.entryPoint else { return 0 }
        var visitedTag = state.visitedTag
        let nodeCount = state.labelOrder.count
        let effectiveEF = max(state.efSearch, k)
        let candidateCapacity = max(1, nodeCount)
        let nearestCapacity = max(1, min(effectiveEF, nodeCount))
        let resultCapacity = max(1, min(k, nodeCount))
        let resultCount = state.searchWorkspace.withCandidateBuffers(
            candidateCapacity: candidateCapacity,
            nearestCapacity: nearestCapacity,
            resultCapacity: resultCapacity,
            visitedCapacity: nodeCount
        ) { candidateQueueStorage, nearestCandidateStorage, resultCandidateStorage in
            state.visited.withUnsafeMutableBufferPointer { visited in
                state.comparisonStorage.withUnsafeBufferPointer { storage in
                    switch metric {
                    case .l2:
                        searchPrepared(
                            query,
                            k: k,
                            into: results,
                            entryPoint: entryPoint,
                            storage: storage,
                            connections: state.connections,
                            labelOrder: state.labelOrder,
                            deletedFlags: state.deletedFlags,
                            liveCount: state.liveCount,
                            maxLevel: state.maxLevel,
                            efSearch: state.efSearch,
                            visited: visited,
                            visitedTag: &visitedTag,
                            candidateQueueStorage: candidateQueueStorage,
                            nearestCandidateStorage: nearestCandidateStorage,
                            resultCandidateStorage: resultCandidateStorage,
                            distanceComputer: HNSWDistanceComputers.L2.self
                        )
                    case .innerProduct, .cosine:
                        searchPrepared(
                            query,
                            k: k,
                            into: results,
                            entryPoint: entryPoint,
                            storage: storage,
                            connections: state.connections,
                            labelOrder: state.labelOrder,
                            deletedFlags: state.deletedFlags,
                            liveCount: state.liveCount,
                            maxLevel: state.maxLevel,
                            efSearch: state.efSearch,
                            visited: visited,
                            visitedTag: &visitedTag,
                            candidateQueueStorage: candidateQueueStorage,
                            nearestCandidateStorage: nearestCandidateStorage,
                            resultCandidateStorage: resultCandidateStorage,
                            distanceComputer: HNSWDistanceComputers.InnerProduct.self
                        )
                    }
                }
            }
        }
        state.visitedTag = visitedTag
        return resultCount
    }

    private func searchNormalizedHalf(
        _ query: UnsafeBufferPointer<Float16>,
        k: Int,
        state: inout State
    ) -> [SearchResult] {
        guard let entryPoint = state.entryPoint else { return [] }
        var visitedTag = state.visitedTag
        let nodeCount = state.labelOrder.count
        let effectiveEF = max(state.efSearch, k)
        let candidateCapacity = max(1, nodeCount)
        let nearestCapacity = max(1, min(effectiveEF, nodeCount))
        let resultCapacity = max(1, min(k, nodeCount))
        let results = Array<SearchResult>(unsafeUninitializedCapacity: resultCapacity) { output, initializedCount in
            initializedCount = state.searchWorkspace.withCandidateBuffers(
                candidateCapacity: candidateCapacity,
                nearestCapacity: nearestCapacity,
                resultCapacity: resultCapacity,
                visitedCapacity: nodeCount
            ) { candidateQueueStorage, nearestCandidateStorage, resultCandidateStorage in
                state.visited.withUnsafeMutableBufferPointer { visited in
                    state.halfComparisonStorage.withUnsafeBufferPointer { storage in
                        switch metric {
                        case .l2:
                            searchPrepared(
                                query,
                                k: k,
                                into: output,
                                entryPoint: entryPoint,
                                storage: storage,
                                connections: state.connections,
                                labelOrder: state.labelOrder,
                                deletedFlags: state.deletedFlags,
                                liveCount: state.liveCount,
                                maxLevel: state.maxLevel,
                                efSearch: state.efSearch,
                                visited: visited,
                                visitedTag: &visitedTag,
                                candidateQueueStorage: candidateQueueStorage,
                                nearestCandidateStorage: nearestCandidateStorage,
                                resultCandidateStorage: resultCandidateStorage,
                                distanceComputer: HNSWDistanceComputers.L2.self
                            )
                        case .innerProduct, .cosine:
                            searchPrepared(
                                query,
                                k: k,
                                into: output,
                                entryPoint: entryPoint,
                                storage: storage,
                                connections: state.connections,
                                labelOrder: state.labelOrder,
                                deletedFlags: state.deletedFlags,
                                liveCount: state.liveCount,
                                maxLevel: state.maxLevel,
                                efSearch: state.efSearch,
                                visited: visited,
                                visitedTag: &visitedTag,
                                candidateQueueStorage: candidateQueueStorage,
                                nearestCandidateStorage: nearestCandidateStorage,
                                resultCandidateStorage: resultCandidateStorage,
                                distanceComputer: HNSWDistanceComputers.InnerProduct.self
                            )
                        }
                    }
                }
            }
        }
        state.visitedTag = visitedTag
        return results
    }

    private func searchNormalizedHalf(
        _ query: UnsafeBufferPointer<Float16>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>,
        state: inout State
    ) -> Int {
        guard let entryPoint = state.entryPoint else { return 0 }
        var visitedTag = state.visitedTag
        let nodeCount = state.labelOrder.count
        let effectiveEF = max(state.efSearch, k)
        let candidateCapacity = max(1, nodeCount)
        let nearestCapacity = max(1, min(effectiveEF, nodeCount))
        let resultCapacity = max(1, min(k, nodeCount))
        let resultCount = state.searchWorkspace.withCandidateBuffers(
            candidateCapacity: candidateCapacity,
            nearestCapacity: nearestCapacity,
            resultCapacity: resultCapacity,
            visitedCapacity: nodeCount
        ) { candidateQueueStorage, nearestCandidateStorage, resultCandidateStorage in
            state.visited.withUnsafeMutableBufferPointer { visited in
                state.halfComparisonStorage.withUnsafeBufferPointer { storage in
                    switch metric {
                    case .l2:
                        searchPrepared(
                            query,
                            k: k,
                            into: results,
                            entryPoint: entryPoint,
                            storage: storage,
                            connections: state.connections,
                            labelOrder: state.labelOrder,
                            deletedFlags: state.deletedFlags,
                            liveCount: state.liveCount,
                            maxLevel: state.maxLevel,
                            efSearch: state.efSearch,
                            visited: visited,
                            visitedTag: &visitedTag,
                            candidateQueueStorage: candidateQueueStorage,
                            nearestCandidateStorage: nearestCandidateStorage,
                            resultCandidateStorage: resultCandidateStorage,
                            distanceComputer: HNSWDistanceComputers.L2.self
                        )
                    case .innerProduct, .cosine:
                        searchPrepared(
                            query,
                            k: k,
                            into: results,
                            entryPoint: entryPoint,
                            storage: storage,
                            connections: state.connections,
                            labelOrder: state.labelOrder,
                            deletedFlags: state.deletedFlags,
                            liveCount: state.liveCount,
                            maxLevel: state.maxLevel,
                            efSearch: state.efSearch,
                            visited: visited,
                            visitedTag: &visitedTag,
                            candidateQueueStorage: candidateQueueStorage,
                            nearestCandidateStorage: nearestCandidateStorage,
                            resultCandidateStorage: resultCandidateStorage,
                            distanceComputer: HNSWDistanceComputers.InnerProduct.self
                        )
                    }
                }
            }
        }
        state.visitedTag = visitedTag
        return resultCount
    }

    private func searchPrepared<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        k: Int,
        entryPoint: HNSWInternalID,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        deletedFlags: borrowing [UInt8],
        liveCount: Int,
        maxLevel: Int,
        efSearch: Int,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type
    ) -> [SearchResult] {
        var current = entryPoint
        var currentDistance = Distance.distanceFromQuery(
            query,
            to: current,
            storage: storage,
            dimensions: dimensions
        )

        if maxLevel > 0 {
            for level in stride(from: maxLevel, through: 1, by: -1) {
                let closest = greedySearchLayer(
                    query,
                    entryPoint: current,
                    entryDistance: currentDistance,
                    level: level,
                    storage: storage,
                    connections: connections,
                    distanceComputer: Distance.self
                )
                current = closest.internalID
                currentDistance = closest.distance
            }
        }

        let candidates = searchLayer(
            query,
            entryPoint: current,
            ef: max(efSearch, k),
            level: 0,
            includeDeleted: false,
            storage: storage,
            connections: connections,
            labelOrder: labelOrder,
            deletedFlags: deletedFlags,
            liveCount: liveCount,
            visited: visited,
            visitedTag: &visitedTag,
            candidateQueueStorage: candidateQueueStorage,
            nearestCandidateStorage: nearestCandidateStorage,
            resultCandidateStorage: resultCandidateStorage,
            distanceComputer: Distance.self,
            resultLimit: k
        )
        var results: [SearchResult] = []
        results.reserveCapacity(candidates.count)
        var needsTieBreakSort = false
        var previousResult: SearchResult?
        for candidate in candidates {
            let result = SearchResult(
                label: label(for: candidate.internalID, labelOrder: labelOrder),
                distance: VectorOperations.publicDistance(fromComparisonDistance: candidate.distance, metric: metric)
            )
            if let previousResult,
               previousResult.distance == result.distance,
               previousResult.label > result.label {
                needsTieBreakSort = true
            }
            results.append(result)
            previousResult = result
        }
        if needsTieBreakSort {
            results.sort {
                if $0.distance == $1.distance {
                    return $0.label < $1.label
                }
                return $0.distance < $1.distance
            }
        }
        return results
    }

    private func searchPrepared<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>,
        entryPoint: HNSWInternalID,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        deletedFlags: borrowing [UInt8],
        liveCount: Int,
        maxLevel: Int,
        efSearch: Int,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type
    ) -> Int {
        var current = entryPoint
        var currentDistance = Distance.distanceFromQuery(
            query,
            to: current,
            storage: storage,
            dimensions: dimensions
        )

        if maxLevel > 0 {
            for level in stride(from: maxLevel, through: 1, by: -1) {
                let closest = greedySearchLayer(
                    query,
                    entryPoint: current,
                    entryDistance: currentDistance,
                    level: level,
                    storage: storage,
                    connections: connections,
                    distanceComputer: Distance.self
                )
                current = closest.internalID
                currentDistance = closest.distance
            }
        }

        if liveCount == labelOrder.count {
            return searchLayerForQueryBareLevel0(
                query,
                entryPoint: current,
                entryDistance: currentDistance,
                ef: max(efSearch, k),
                storage: storage,
                connections: connections,
                labelOrder: labelOrder,
                visited: visited,
                visitedTag: &visitedTag,
                candidateQueueStorage: candidateQueueStorage,
                nearestCandidateStorage: nearestCandidateStorage,
                resultCandidateStorage: resultCandidateStorage,
                distanceComputer: Distance.self,
                resultLimit: k,
                into: results
            )
        }

        let candidates = searchLayer(
            query,
            entryPoint: current,
            ef: max(efSearch, k),
            level: 0,
            includeDeleted: false,
            storage: storage,
            connections: connections,
            labelOrder: labelOrder,
            deletedFlags: deletedFlags,
            liveCount: liveCount,
            visited: visited,
            visitedTag: &visitedTag,
            candidateQueueStorage: candidateQueueStorage,
            nearestCandidateStorage: nearestCandidateStorage,
            resultCandidateStorage: resultCandidateStorage,
            distanceComputer: Distance.self,
            resultLimit: k
        )
        let resultCount = min(candidates.count, results.count)
        for index in 0..<resultCount {
            let candidate = candidates[index]
            results[index] = SearchResult(
                label: label(for: candidate.internalID, labelOrder: labelOrder),
                distance: VectorOperations.publicDistance(fromComparisonDistance: candidate.distance, metric: metric)
            )
        }
        return resultCount
    }

    private var levelMultiplier: Double {
        1.0 / hnswNaturalLog(Double(configuration.m))
    }

    @inline(__always)
    private func maxConnections(at level: Int) -> Int {
        level == 0 ? configuration.m * 2 : configuration.m
    }

    @inline(__always)
    private func newElementConnections(at level: Int) -> Int {
        configuration.m
    }

    private func connectNewElement(_ internalID: HNSWInternalID, state: inout State) {
        if Scalar.self == Float16.self {
            state.halfComparisonStorage.withUnsafeBufferPointer { storage in
                connectNewElement(internalID, storage: storage, state: &state)
            }
        } else {
            state.comparisonStorage.withUnsafeBufferPointer { storage in
                connectNewElement(internalID, storage: storage, state: &state)
            }
        }
    }

    private func connectNewElement<Stored: HNSWScalar>(
        _ internalID: HNSWInternalID,
        storage: UnsafeBufferPointer<Stored>,
        state: inout State
    ) {
        switch metric {
        case .l2:
            connectNewElement(
                internalID,
                storage: storage,
                state: &state,
                distanceComputer: HNSWDistanceComputers.L2.self
            )
        case .innerProduct, .cosine:
            connectNewElement(
                internalID,
                storage: storage,
                state: &state,
                distanceComputer: HNSWDistanceComputers.InnerProduct.self
            )
        }
    }

    private func connectNewElement<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ internalID: HNSWInternalID,
        storage: UnsafeBufferPointer<Stored>,
        state: inout State,
        distanceComputer: Distance.Type
    ) {
        guard let entry = entry(for: internalID, state: state) else { return }
        guard let entryPoint = state.entryPoint else {
            state.entryPoint = internalID
            state.maxLevel = entry.level
            return
        }
        guard hasLiveEntry(excluding: internalID, state: state) else {
            state.entryPoint = internalID
            state.maxLevel = entry.level
            return
        }

        var current = entryPoint
        var currentDistance = Distance.distanceBetween(
            internalID,
            current,
            storage: storage,
            dimensions: dimensions
        )
        let previousMaxLevel = state.maxLevel

        if entry.level < previousMaxLevel {
            for level in stride(from: previousMaxLevel, through: entry.level + 1, by: -1) {
                let closest = greedySearchNode(
                    internalID,
                    entryPoint: current,
                    entryDistance: currentDistance,
                    level: level,
                    storage: storage,
                    connections: state.connections,
                    distanceComputer: Distance.self
                )
                current = closest.internalID
                currentDistance = closest.distance
            }
        }

        let topLevel = min(entry.level, previousMaxLevel)
        if topLevel >= 0 {
            var visitedTag = state.visitedTag
            for level in stride(from: topLevel, through: 0, by: -1) {
                let efConstruction = configuration.efConstruction
                let nodeCount = state.labelOrder.count
                let candidateCapacity = max(1, nodeCount)
                let nearestCapacity = max(1, min(efConstruction, nodeCount))
                var candidates = state.searchWorkspace.withCandidateBuffers(
                    candidateCapacity: candidateCapacity,
                    nearestCapacity: nearestCapacity,
                    visitedCapacity: nodeCount
                ) { candidateQueueStorage, nearestCandidateStorage, _ in
                    state.visited.withUnsafeMutableBufferPointer { visited in
                        searchLayerForNode(
                            internalID,
                            entryPoint: current,
                            ef: efConstruction,
                            level: level,
                            includeDeleted: false,
                            storage: storage,
                            connections: state.connections,
                            labelOrder: state.labelOrder,
                            deletedFlags: state.deletedFlags,
                            liveCount: state.liveCount,
                            visited: visited,
                            visitedTag: &visitedTag,
                            candidateQueueStorage: candidateQueueStorage,
                            nearestCandidateStorage: nearestCandidateStorage,
                            distanceComputer: Distance.self
                        )
                    }
                }
                let selected = selectNeighbors(
                    candidates: &candidates,
                    limit: newElementConnections(at: level),
                    candidatesAreSorted: true,
                    storage: storage,
                    state: state,
                    distanceComputer: Distance.self
                )
                setConnections(selected, for: internalID, at: level, state: &state)
                for neighbor in selected {
                    connectBidirectional(
                        internalID,
                        neighborID: neighbor.internalID,
                        level: level,
                        storage: storage,
                        state: &state,
                        distanceComputer: Distance.self
                    )
                }
                if let first = selected.first {
                    current = first.internalID
                    currentDistance = first.distance
                }
            }
            state.visitedTag = visitedTag
        }

        if entry.level > previousMaxLevel {
            state.entryPoint = internalID
            state.maxLevel = entry.level
        }
    }

    private func rebuildGraph(state: inout State) {
        var rebuiltConnections = HNSWConnectionStore(m: configuration.m)
        rebuiltConnections.reserveCapacity(state.labelOrder.count)
        for (internalID, label) in state.labelOrder.enumerated() {
            let level = internalID < state.levels.count ? state.levels[internalID] : state.entries[label]?.level ?? 0
            rebuiltConnections.appendNode(level: level)
        }
        state.connections = rebuiltConnections
        state.entryPoint = nil
        state.maxLevel = -1

        for internalID in state.labelOrder.indices {
            let typedID = HNSWInternalID(internalID)
            guard entry(for: typedID, state: state) != nil else { continue }
            connectNewElement(typedID, state: &state)
        }
    }

    private func greedySearchNode<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ queryID: HNSWInternalID,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        level: Int,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        distanceComputer: Distance.Type
    ) -> HNSWNeighborCandidate {
        var current = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        var changed = true
        while changed {
            changed = false
            let neighborRange = connections.neighborStorageRange(for: current.internalID, at: level)
            guard !neighborRange.isEmpty else {
                return current
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.neighborInStorage(at: neighborStorageIndex, level: level)
                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceBetween(
                        queryID,
                        neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if isCloserHNSWCandidate(candidate, than: current) {
                    current = candidate
                    changed = true
                }
            }
        }
        return current
    }

    private func greedySearchLayer<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        level: Int,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        distanceComputer: Distance.Type
    ) -> HNSWNeighborCandidate {
        var current = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        var changed = true
        while changed {
            changed = false
            let neighborRange = connections.neighborStorageRange(for: current.internalID, at: level)
            guard !neighborRange.isEmpty else {
                return current
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.neighborInStorage(at: neighborStorageIndex, level: level)
                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceFromQuery(
                        query,
                        to: neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if isCloserHNSWCandidate(candidate, than: current) {
                    current = candidate
                    changed = true
                }
            }
        }
        return current
    }

    private func searchLayerForNode<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ queryID: HNSWInternalID,
        entryPoint: HNSWInternalID,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        deletedFlags: borrowing [UInt8],
        liveCount: Int,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type
    ) -> [HNSWNeighborCandidate] {
        searchLayerForNodeWithVisitedArray(
            queryID,
            entryPoint: entryPoint,
            entryDistance: Distance.distanceBetween(
                queryID,
                entryPoint,
                storage: storage,
                dimensions: dimensions
            ),
            ef: ef,
            level: level,
            includeDeleted: includeDeleted,
            storage: storage,
            connections: connections,
            labelOrder: labelOrder,
            deletedFlags: deletedFlags,
            liveCount: liveCount,
            visited: visited,
            visitedTag: &visitedTag,
            candidateQueueStorage: candidateQueueStorage,
            nearestCandidateStorage: nearestCandidateStorage,
            distanceComputer: Distance.self
        )
    }

    private func searchLayerForNodeWithVisitedArray<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ queryID: HNSWInternalID,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        deletedFlags: borrowing [UInt8],
        liveCount: Int,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type
    ) -> [HNSWNeighborCandidate] {
        if !includeDeleted, liveCount == labelOrder.count {
            return searchLayerForNodeBare(
                queryID,
                entryPoint: entryPoint,
                entryDistance: entryDistance,
                ef: ef,
                level: level,
                storage: storage,
                connections: connections,
                visited: visited,
                visitedTag: &visitedTag,
                candidateQueueStorage: candidateQueueStorage,
                nearestCandidateStorage: nearestCandidateStorage,
                distanceComputer: Distance.self
            )
        }

        let effectiveEF = ef
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWBoundedNearestCandidateHeap(storage: candidateQueueStorage)
        var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestCandidateStorage)
        var lowerBound = Float.greatestFiniteMagnitude

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        markVisited(entryPoint, tag: tag, visited: visited)
        candidateQueue.push(entry)
        if includeDeleted || !isDeleted(entryPoint, deletedFlags: deletedFlags) {
            nearest.push(entry)
            lowerBound = entry.distance
        }

        while !candidateQueue.isEmpty {
            let current = candidateQueue.popUnchecked()
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            let neighborRange = connections.neighborStorageRange(for: current.internalID, at: level)
            guard !neighborRange.isEmpty else {
                continue
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.neighborInStorage(at: neighborStorageIndex, level: level)
                guard markVisited(neighborID, tag: tag, visited: visited) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceBetween(
                        queryID,
                        neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if nearest.count < effectiveEF || candidate.distance < lowerBound {
                    candidateQueue.push(candidate)
                    if includeDeleted || !isDeleted(neighborID, deletedFlags: deletedFlags) {
                        if nearest.count < effectiveEF {
                            nearest.push(candidate)
                        } else {
                            nearest.replaceTopUnchecked(with: candidate)
                        }
                        lowerBound = nearest.topDistanceUnchecked
                    }
                }
            }
        }

        return nearest.sortedElements()
    }

    private func searchLayerForNodeBare<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ queryID: HNSWInternalID,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        ef: Int,
        level: Int,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = ef
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWBoundedNearestCandidateHeap(storage: candidateQueueStorage)
        var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestCandidateStorage)

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        var lowerBound = entry.distance
        markVisited(entryPoint, tag: tag, visited: visited)
        candidateQueue.push(entry)
        nearest.push(entry)

        while !candidateQueue.isEmpty {
            let current = candidateQueue.popUnchecked()
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            let neighborRange = connections.neighborStorageRange(for: current.internalID, at: level)
            guard !neighborRange.isEmpty else {
                continue
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.neighborInStorage(at: neighborStorageIndex, level: level)
                guard markVisited(neighborID, tag: tag, visited: visited) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceBetween(
                        queryID,
                        neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if nearest.count < effectiveEF || candidate.distance < lowerBound {
                    candidateQueue.push(candidate)
                    if nearest.count < effectiveEF {
                        nearest.push(candidate)
                    } else {
                        nearest.replaceTopUnchecked(with: candidate)
                    }
                    lowerBound = nearest.topDistanceUnchecked
                }
            }
        }

        return nearest.sortedElements()
    }

    private func searchLayer<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        entryPoint: HNSWInternalID,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        deletedFlags: borrowing [UInt8],
        liveCount: Int,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int
    ) -> [HNSWNeighborCandidate] {
        searchLayerForQueryWithVisitedArray(
            query,
            entryPoint: entryPoint,
            entryDistance: Distance.distanceFromQuery(
                query,
                to: entryPoint,
                storage: storage,
                dimensions: dimensions
            ),
            ef: ef,
            level: level,
            includeDeleted: includeDeleted,
            storage: storage,
            connections: connections,
            labelOrder: labelOrder,
            deletedFlags: deletedFlags,
            liveCount: liveCount,
            visited: visited,
            visitedTag: &visitedTag,
            candidateQueueStorage: candidateQueueStorage,
            nearestCandidateStorage: nearestCandidateStorage,
            resultCandidateStorage: resultCandidateStorage,
            distanceComputer: Distance.self,
            resultLimit: resultLimit
        )
    }

    private func searchLayerForQueryWithVisitedArray<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        deletedFlags: borrowing [UInt8],
        liveCount: Int,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int
    ) -> [HNSWNeighborCandidate] {
        if !includeDeleted, liveCount == labelOrder.count {
            if level == 0 {
                return searchLayerForQueryBareLevel0(
                    query,
                    entryPoint: entryPoint,
                    entryDistance: entryDistance,
                    ef: ef,
                    storage: storage,
                    connections: connections,
                    visited: visited,
                    visitedTag: &visitedTag,
                    candidateQueueStorage: candidateQueueStorage,
                    nearestCandidateStorage: nearestCandidateStorage,
                    resultCandidateStorage: resultCandidateStorage,
                    distanceComputer: Distance.self,
                    resultLimit: resultLimit
                )
            }
            return searchLayerForQueryBare(
                query,
                entryPoint: entryPoint,
                entryDistance: entryDistance,
                ef: ef,
                level: level,
                storage: storage,
                connections: connections,
                visited: visited,
                visitedTag: &visitedTag,
                candidateQueueStorage: candidateQueueStorage,
                nearestCandidateStorage: nearestCandidateStorage,
                resultCandidateStorage: resultCandidateStorage,
                distanceComputer: Distance.self,
                resultLimit: resultLimit
            )
        }

        let effectiveEF = ef
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWBoundedNearestCandidateHeap(storage: candidateQueueStorage)
        var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestCandidateStorage)
        var lowerBound = Float.greatestFiniteMagnitude

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        markVisited(entryPoint, tag: tag, visited: visited)
        candidateQueue.push(entry)
        if includeDeleted || !isDeleted(entryPoint, deletedFlags: deletedFlags) {
            nearest.push(entry)
            lowerBound = entry.distance
        }

        while !candidateQueue.isEmpty {
            let current = candidateQueue.popUnchecked()
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            let neighborRange = connections.neighborStorageRange(for: current.internalID, at: level)
            guard !neighborRange.isEmpty else {
                continue
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.neighborInStorage(at: neighborStorageIndex, level: level)
                guard markVisited(neighborID, tag: tag, visited: visited) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceFromQuery(
                        query,
                        to: neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if nearest.count < effectiveEF || candidate.distance < lowerBound {
                    candidateQueue.push(candidate)
                    if includeDeleted || !isDeleted(neighborID, deletedFlags: deletedFlags) {
                        if nearest.count < effectiveEF {
                            nearest.push(candidate)
                        } else {
                            nearest.replaceTopUnchecked(with: candidate)
                        }
                        lowerBound = nearest.topDistanceUnchecked
                    }
                }
            }
        }

        return nearest.closestSorted(limit: resultLimit, using: resultCandidateStorage)
    }

    private func searchLayerForQueryBareLevel0<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        ef: Int,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = ef
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWBoundedNearestCandidateHeap(storage: candidateQueueStorage)
        var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestCandidateStorage)

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        var lowerBound = entry.distance
        markVisited(entryPoint, tag: tag, visited: visited)
        candidateQueue.push(entry)
        nearest.push(entry)

        while !candidateQueue.isEmpty {
            let current = candidateQueue.popUnchecked()
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            let neighborRange = connections.level0NeighborStorageRange(for: current.internalID)
            guard !neighborRange.isEmpty else {
                continue
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.level0NeighborInStorage(at: neighborStorageIndex)
                guard markVisited(neighborID, tag: tag, visited: visited) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceFromQuery(
                        query,
                        to: neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if nearest.count < effectiveEF || candidate.distance < lowerBound {
                    candidateQueue.push(candidate)
                    if nearest.count < effectiveEF {
                        nearest.push(candidate)
                    } else {
                        nearest.replaceTopUnchecked(with: candidate)
                    }
                    lowerBound = nearest.topDistanceUnchecked
                }
            }
        }

        return nearest.closestSorted(limit: resultLimit, using: resultCandidateStorage)
    }

    private func searchLayerForQueryBareLevel0<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        ef: Int,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        labelOrder: borrowing [UInt64],
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>
    ) -> Int {
        let effectiveEF = ef
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWBoundedNearestCandidateHeap(storage: candidateQueueStorage)
        var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestCandidateStorage)

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        var lowerBound = entry.distance
        markVisited(entryPoint, tag: tag, visited: visited)
        candidateQueue.push(entry)
        nearest.push(entry)

        while !candidateQueue.isEmpty {
            let current = candidateQueue.popUnchecked()
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            let neighborRange = connections.level0NeighborStorageRange(for: current.internalID)
            guard !neighborRange.isEmpty else {
                continue
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.level0NeighborInStorage(at: neighborStorageIndex)
                guard markVisited(neighborID, tag: tag, visited: visited) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceFromQuery(
                        query,
                        to: neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if nearest.count < effectiveEF || candidate.distance < lowerBound {
                    candidateQueue.push(candidate)
                    if nearest.count < effectiveEF {
                        nearest.push(candidate)
                    } else {
                        nearest.replaceTopUnchecked(with: candidate)
                    }
                    lowerBound = nearest.topDistanceUnchecked
                }
            }
        }

        return nearest.writeClosestSorted(
            limit: resultLimit,
            using: resultCandidateStorage,
            to: results,
            labelOrder: labelOrder,
            metric: metric
        )
    }

    private func searchLayerForQueryBare<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ query: UnsafeBufferPointer<Stored>,
        entryPoint: HNSWInternalID,
        entryDistance: Float,
        ef: Int,
        level: Int,
        storage: UnsafeBufferPointer<Stored>,
        connections: borrowing HNSWConnectionStore,
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16,
        candidateQueueStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultCandidateStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = ef
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWBoundedNearestCandidateHeap(storage: candidateQueueStorage)
        var nearest = HNSWBoundedFarthestCandidateHeap(storage: nearestCandidateStorage)

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            distance: entryDistance
        )
        var lowerBound = entry.distance
        markVisited(entryPoint, tag: tag, visited: visited)
        candidateQueue.push(entry)
        nearest.push(entry)

        while !candidateQueue.isEmpty {
            let current = candidateQueue.popUnchecked()
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            let neighborRange = connections.neighborStorageRange(for: current.internalID, at: level)
            guard !neighborRange.isEmpty else {
                continue
            }
            for neighborStorageIndex in neighborRange {
                let neighborID = connections.neighborInStorage(at: neighborStorageIndex, level: level)
                guard markVisited(neighborID, tag: tag, visited: visited) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    distance: Distance.distanceFromQuery(
                        query,
                        to: neighborID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
                if nearest.count < effectiveEF || candidate.distance < lowerBound {
                    candidateQueue.push(candidate)
                    if nearest.count < effectiveEF {
                        nearest.push(candidate)
                    } else {
                        nearest.replaceTopUnchecked(with: candidate)
                    }
                    lowerBound = nearest.topDistanceUnchecked
                }
            }
        }

        return nearest.closestSorted(limit: resultLimit, using: resultCandidateStorage)
    }

    private func selectNeighbors<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        candidates: inout [HNSWNeighborCandidate],
        limit: Int,
        candidatesAreSorted: Bool,
        storage: UnsafeBufferPointer<Stored>,
        state: borrowing State,
        distanceComputer: Distance.Type
    ) -> [HNSWNeighborCandidate] {
        guard limit > 0 else { return [] }

        var selected: [HNSWNeighborCandidate] = []
        selected.reserveCapacity(min(limit, candidates.count))
        if !candidatesAreSorted {
            candidates.sort(by: isCloserHNSWCandidate)
        }
        for candidate in candidates {
            guard selected.count < limit else { break }
            var isDiverse = true
            for existing in selected {
                let neighborDistance = Distance.distanceBetween(
                    candidate.internalID,
                    existing.internalID,
                    storage: storage,
                    dimensions: dimensions
                )
                if neighborDistance < candidate.distance {
                    isDiverse = false
                    break
                }
            }
            if isDiverse {
                selected.append(candidate)
            }
        }
        return selected
    }

    private func hasLiveEntry(excluding excludedID: HNSWInternalID, state: borrowing State) -> Bool {
        let excludedIndex = Int(excludedID)
        guard excludedIndex < state.deletedFlags.count else {
            return state.liveCount > 0
        }
        return state.liveCount > (state.deletedFlags[excludedIndex] != 0 ? 0 : 1)
    }

    private func reusableDeletedEntry(state: borrowing State) -> (label: UInt64, entry: Entry)? {
        for label in state.labelOrder {
            guard let entry = state.entries[label], entry.deleted else { continue }
            return (label, entry)
        }
        return nil
    }

    private func connectBidirectional<Stored: HNSWScalar, Distance: HNSWDistanceComputers.Computer>(
        _ internalID: HNSWInternalID,
        neighborID: HNSWInternalID,
        level: Int,
        storage: UnsafeBufferPointer<Stored>,
        state: inout State,
        distanceComputer: Distance.Type
    ) {
        guard internalID != neighborID else { return }
        state.connections.ensureNode(neighborID, through: level)
        if !state.connections.contains(internalID, for: neighborID, at: level) {
            state.connections.append(internalID, for: neighborID, at: level)
        }

        let maxConnections = maxConnections(at: level)
        let neighborRange = state.connections.neighborStorageRange(for: neighborID, at: level)
        let neighborCount = neighborRange.count
        guard neighborCount > maxConnections else { return }

        var candidates: [HNSWNeighborCandidate] = []
        candidates.reserveCapacity(neighborCount)
        for neighborStorageIndex in neighborRange {
            let candidateID = state.connections.neighborInStorage(at: neighborStorageIndex, level: level)
            candidates.append(
                HNSWNeighborCandidate(
                    internalID: candidateID,
                    distance: Distance.distanceBetween(
                        neighborID,
                        candidateID,
                        storage: storage,
                        dimensions: dimensions
                    )
                )
            )
        }
        let selected = selectNeighbors(
            candidates: &candidates,
            limit: maxConnections,
            candidatesAreSorted: false,
            storage: storage,
            state: state,
            distanceComputer: Distance.self
        )
        state.connections.replaceNeighbors(from: selected, excluding: nil, for: neighborID, at: level)
    }

    private func setConnections(
        _ neighbors: [HNSWNeighborCandidate],
        for internalID: HNSWInternalID,
        at level: Int,
        state: inout State
    ) {
        state.connections.replaceNeighbors(from: neighbors, excluding: internalID, for: internalID, at: level)
    }

    @inline(__always)
    private func nextVisitedTag(
        visited: UnsafeMutableBufferPointer<UInt16>,
        visitedTag: inout UInt16
    ) -> UInt16 {
        if visitedTag == UInt16.max {
            for index in visited.indices {
                visited[index] = 0
            }
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
        let entry = visited.baseAddress! + Int(internalID)
        guard entry.pointee != tag else { return false }
        entry.pointee = tag
        return true
    }

    private func entry(for internalID: HNSWInternalID, state: borrowing State) -> Entry? {
        let internalIndex = Int(internalID)
        guard internalIndex < state.labelOrder.count else { return nil }
        return state.entries[state.labelOrder[internalIndex]]
    }

    @inline(__always)
    private func label(for internalID: HNSWInternalID, state: borrowing State) -> UInt64 {
        state.labelOrder[Int(internalID)]
    }

    @inline(__always)
    private func label(for internalID: HNSWInternalID, labelOrder: borrowing [UInt64]) -> UInt64 {
        labelOrder[Int(internalID)]
    }

    @inline(__always)
    private func isDeleted(_ internalID: HNSWInternalID, state: borrowing State) -> Bool {
        let internalIndex = Int(internalID)
        guard internalIndex < state.deletedFlags.count else { return true }
        return state.deletedFlags[internalIndex] != 0
    }

    @inline(__always)
    private func isDeleted(_ internalID: HNSWInternalID, deletedFlags: borrowing [UInt8]) -> Bool {
        let internalIndex = Int(internalID)
        guard internalIndex < deletedFlags.count else { return true }
        return deletedFlags[internalIndex] != 0
    }
}

extension HNSWIndex {
    var comparisonValueCounts: (singlePrecision: Int, halfPrecision: Int) {
        state.withLock {
            ($0.comparisonStorage.count, $0.halfComparisonStorage.count)
        }
    }

    /// Captures the graph topology for an internal distance backend.
    ///
    /// The returned snapshot may outlive this index and can be paired with a different
    /// immutable vector representation, such as packed TurboQuant codes. Array storage in
    /// the flat connection store remains copy-on-write. The snapshot is intentionally
    /// internal; public callers continue to use `search` and `serializedArchive` instead of
    /// depending on graph layout.
    func graphSnapshot() -> HNSWGraphSnapshot {
        state.withLock { state in
            HNSWGraphSnapshot(
                labelOrder: state.labelOrder,
                deletedFlags: state.deletedFlags,
                levels: state.levels,
                connections: state.connections,
                entryPoint: state.entryPoint,
                maxLevel: state.maxLevel
            )
        }
    }
}

private struct HNSWArchiveWriter {
    var bytes: [UInt8] = []

    mutating func writeBytes(_ bytes: [UInt8]) {
        self.bytes.append(contentsOf: bytes)
    }

    mutating func writeBool(_ value: Bool) {
        writeBytes([value ? 1 : 0])
    }

    mutating func writeUInt32(_ value: UInt32) {
        writeBytes([
            UInt8(truncatingIfNeeded: value),
            UInt8(truncatingIfNeeded: value >> 8),
            UInt8(truncatingIfNeeded: value >> 16),
            UInt8(truncatingIfNeeded: value >> 24),
        ])
    }

    mutating func writeUInt64(_ value: UInt64) {
        writeBytes([
            UInt8(truncatingIfNeeded: value),
            UInt8(truncatingIfNeeded: value >> 8),
            UInt8(truncatingIfNeeded: value >> 16),
            UInt8(truncatingIfNeeded: value >> 24),
            UInt8(truncatingIfNeeded: value >> 32),
            UInt8(truncatingIfNeeded: value >> 40),
            UInt8(truncatingIfNeeded: value >> 48),
            UInt8(truncatingIfNeeded: value >> 56),
        ])
    }

    mutating func writeFloat(_ value: Float) {
        writeUInt32(value.bitPattern)
    }

    mutating func writeString(_ value: String) {
        writeUInt32(UInt32(value.utf8.count))
        bytes.append(contentsOf: value.utf8)
    }
}

private struct HNSWArchiveReader {
    static let flatMagic: UInt64 = 0x414C_4657_534E_4853
    static let graphMagic: UInt64 = 0x4652_4757_534E_4853

    let bytes: UnsafeRawBufferPointer
    var offset = 0

    var remainingByteCount: Int {
        bytes.count - offset
    }

    mutating func readMagic() throws -> UInt64 {
        let magic = try readUInt64()
        return magic
    }

    mutating func readByte() throws -> UInt8 {
        guard offset < bytes.count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += 1 }
        return bytes.loadUnaligned(fromByteOffset: offset, as: UInt8.self)
    }

    mutating func skipBytes(count: Int) throws {
        guard count >= 0,
              bytes.count >= count,
              offset <= bytes.count - count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        offset += count
    }

    mutating func readBool() throws -> Bool {
        let value = try readByte()
        guard value == 0 || value == 1 else {
            throw HNSWError.loadFailed("Invalid boolean in flat index data")
        }
        return value == 1
    }

    mutating func readUInt32() throws -> UInt32 {
        guard bytes.count >= 4, offset <= bytes.count - 4 else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += 4 }
        return UInt32(
            littleEndian: bytes.loadUnaligned(
                fromByteOffset: offset,
                as: UInt32.self
            )
        )
    }

    mutating func readUInt64() throws -> UInt64 {
        guard bytes.count >= 8, offset <= bytes.count - 8 else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += 8 }
        return UInt64(
            littleEndian: bytes.loadUnaligned(
                fromByteOffset: offset,
                as: UInt64.self
            )
        )
    }

    mutating func readIntFromUInt32(name: String) throws -> Int {
        let value = try readUInt32()
        guard UInt64(value) <= UInt64(Int.max) else {
            throw HNSWError.loadFailed(
                "Flat index contains an out-of-range \(name)"
            )
        }
        return Int(value)
    }

    mutating func readFloat() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readString() throws -> String {
        let count = try readIntFromUInt32(name: "string length")
        guard bytes.count >= count,
              offset <= bytes.count - count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += count }
        let start = bytes.baseAddress!.advanced(by: offset)
            .assumingMemoryBound(to: UInt8.self)
        return String(
            decoding: UnsafeBufferPointer(start: start, count: count),
            as: UTF8.self
        )
    }

    func ensureFullyRead() throws {
        guard offset == bytes.count else {
            throw HNSWError.loadFailed("Flat index data has trailing bytes")
        }
    }
}


// MARK: - Type Aliases for Convenience

/// HNSW Index using Float32 vectors (standard precision)
public typealias HNSWIndexF32 = HNSWIndex<Float>

/// HNSW Index using Float16 vectors (half precision, 50% memory savings)
public typealias HNSWIndexF16 = HNSWIndex<Float16>
