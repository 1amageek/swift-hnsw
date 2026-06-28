import Foundation
import Synchronization
#if HNSWLIB_BACKEND && canImport(hnswlib)
import hnswlib
#endif

#if HNSWLIB_BACKEND && canImport(hnswlib)
/// HNSW index with TurboQuant vector quantization.
///
/// Architecture:
/// 1. **Construction**: Vectors are normalized, HD³-rotated to p dimensions (next power of 2),
///    and stored as Float32. Graph is built with exact L2 for maximum quality.
/// 2. **Finalize**: Stored Float32 vectors are quantized in-place to packed format.
/// 3. **Search**: Uses ADC — query is full-precision rotated float, stored vectors are packed.
///
/// All p coordinates of the HD³ transform are used (no truncation) to preserve
/// L2 distances exactly for any input dimension.
public final class TurboQuantIndex: Sendable {

    // MARK: - Properties

    public let dimensions: Int
    public let bitWidth: Int
    public let configuration: HNSWConfiguration
    public let seed: UInt64

    /// Padded dimension (next power of 2). All internal operations use this.
    public let paddedDimensions: Int

    private struct CxxState: Sendable {
        let spaceAddress: UInt
        let indexAddress: UInt
        let encoderAddress: UInt
        var finalized: Bool
        var rotatedScratch: [Float]

        init(
            space: HNSWSpaceHandle,
            index: HNSWIndexHandle,
            encoder: TurboQuantEncoderHandle,
            finalized: Bool,
            paddedDimensions: Int
        ) {
            self.spaceAddress = UInt(bitPattern: space)
            self.indexAddress = UInt(bitPattern: index)
            self.encoderAddress = UInt(bitPattern: encoder)
            self.finalized = finalized
            self.rotatedScratch = [Float](repeating: 0, count: paddedDimensions)
        }

        var space: HNSWSpaceHandle {
            UnsafeMutableRawPointer(bitPattern: spaceAddress)!
        }

        var index: HNSWIndexHandle {
            UnsafeMutableRawPointer(bitPattern: indexAddress)!
        }

        var encoder: TurboQuantEncoderHandle {
            UnsafeMutableRawPointer(bitPattern: encoderAddress)!
        }
    }

    private let state: Mutex<CxxState>
    private let _packedSize: Int

    // MARK: - Initialization

    public init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int = 4,
        configuration: HNSWConfiguration = .balanced,
        seed: UInt64 = 42
    ) throws {
        guard dimensions > 0 else {
            throw HNSWError.initializationFailed("dimensions must be positive")
        }
        guard (1...4).contains(bitWidth) else {
            throw HNSWError.initializationFailed("bitWidth must be 1, 2, 3, or 4")
        }

        self.dimensions = dimensions
        self.bitWidth = bitWidth
        self.configuration = configuration
        self.seed = seed

        // Codebook scaled by 1/√p (per-coordinate variance = 1/p after HD³)
        var p = 1
        while p < dimensions { p *= 2 }
        self.paddedDimensions = p

        let quantizer = ScalarQuantizer(bitWidth: bitWidth, dimension: p)

        // C++ encoder
        guard let encoder = quantizer.centroids.withUnsafeBufferPointer({ cBuf in
            quantizer.boundaries.withUnsafeBufferPointer({ bBuf in
                hnsw_tq_encoder_create(
                    dimensions, Int32(bitWidth),
                    cBuf.baseAddress!, Int32(quantizer.numCentroids),
                    bBuf.baseAddress!, Int32(quantizer.boundaries.count), seed)
            })
        }) else {
            throw HNSWError.initializationFailed("Failed to create encoder")
        }
        self._packedSize = hnsw_tq_encoder_packed_size(encoder)

        // Space: data_size = p * sizeof(float) for Float32 construction
        guard let space = quantizer.centroids.withUnsafeBufferPointer({ buf in
            hnsw_create_turboquant_l2_space(
                dimensions, p, Int32(bitWidth),
                buf.baseAddress!, Int32(quantizer.numCentroids))
        }) else {
            hnsw_tq_encoder_destroy(encoder)
            throw HNSWError.initializationFailed("Failed to create space")
        }

        guard let index = hnsw_create_index(
            space, maxElements, configuration.m, configuration.efConstruction,
            configuration.randomSeed, configuration.allowReplaceDeleted
        ) else {
            hnsw_destroy_space(space)
            hnsw_tq_encoder_destroy(encoder)
            throw HNSWError.initializationFailed("Failed to create index")
        }
        hnsw_set_ef(index, configuration.efSearch)
        self.state = Mutex(CxxState(
            space: space,
            index: index,
            encoder: encoder,
            finalized: false,
            paddedDimensions: p
        ))
    }

    deinit {
        state.withLock { state in
            hnsw_destroy_index(state.index)
            hnsw_destroy_space(state.space)
            hnsw_tq_encoder_destroy(state.encoder)
        }
    }

    // MARK: - Info

    public var count: Int { state.withLock { Int(hnsw_get_current_count($0.index)) } }
    public var capacity: Int { state.withLock { Int(hnsw_get_max_elements($0.index)) } }
    public var isEmpty: Bool { count == 0 }
    public var isFinalized: Bool { state.withLock { $0.finalized } }
    public var bytesPerVector: Int { _packedSize }
    public var compressionRatio: Float { Float(dimensions * 4) / Float(_packedSize) }

    public func setEfSearch(_ ef: Int) {
        state.withLock { hnsw_set_ef($0.index, ef) }
    }

    // MARK: - Add

    /// Add a vector. Must be called BEFORE searching.
    /// Internally: normalize → HD³ rotate to p dims → store as Float32[p].
    public func add(_ vector: [Float], label: UInt64) throws {
        try vector.withUnsafeBufferPointer { buffer in
            try add(buffer, label: label)
        }
    }

    /// Add a borrowed vector without materializing an intermediate array.
    public func add(_ vector: UnsafeBufferPointer<Float>, label: UInt64) throws {
        guard vector.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vector.count)
        }
        try state.withLock { state in
            guard !state.finalized else {
                throw HNSWError.addPointFailed("Cannot add after finalize()")
            }
            let encoder = state.encoder
            let index = state.index
            ensureRotatedScratch(&state)
            state.rotatedScratch.withUnsafeMutableBufferPointer { rBuf in
                hnsw_tq_encoder_rotate_query(encoder, vector.baseAddress!, rBuf.baseAddress!)
            }
            let success = state.rotatedScratch.withUnsafeBufferPointer { buf in
                hnsw_add_point(index, buf.baseAddress!, label)
            }
            guard success else {
                throw HNSWError.addPointFailed("Failed to add point with label \(label)")
            }
        }
    }

    // MARK: - Search

    /// Search for k nearest neighbors. Auto-finalizes on first call.
    public func search(_ query: [Float], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Search with a borrowed query vector.
    public func search(_ query: UnsafeBufferPointer<Float>, k: Int) throws -> [SearchResult] {
        guard query.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: query.count)
        }
        guard k > 0 else { return [] }

        return try state.withLock { state in
            let encoder = state.encoder
            let index = state.index
            ensureRotatedScratch(&state)
            state.rotatedScratch.withUnsafeMutableBufferPointer { rBuf in
                hnsw_tq_encoder_rotate_query(encoder, query.baseAddress!, rBuf.baseAddress!)
            }

            if !state.finalized {
                guard hnsw_turboquant_finalize(index, encoder) else {
                    throw HNSWError.serializationFailed("Finalization failed (out of memory)")
                }
                hnsw_turboquant_set_data_size(state.space, _packedSize)
                hnsw_turboquant_set_mode(state.space, 1)
                state.finalized = true
            }

            var labels = [UInt64](repeating: 0, count: k)
            var distances = [Float](repeating: 0, count: k)

            let resultCount = state.rotatedScratch.withUnsafeBufferPointer { queryBuf in
                labels.withUnsafeMutableBufferPointer { labelsBuf in
                    distances.withUnsafeMutableBufferPointer { distBuf in
                        hnsw_search_knn(index, queryBuf.baseAddress!, Int32(k),
                                        labelsBuf.baseAddress!, distBuf.baseAddress!)
                    }
                }
            }

            var results: [SearchResult] = []
            results.reserveCapacity(Int(resultCount))
            for index in 0..<Int(resultCount) {
                results.append(SearchResult(label: labels[index], distance: distances[index]))
            }
            return results
        }
    }

    // MARK: - Persistence

    private static let headerMagic: UInt32 = 0x54514857 // "TQHW"
    private static let headerVersion: UInt32 = 1
    // Header: 28 bytes, field-by-field (no struct padding dependency)
    // [magic:4][version:4][dimensions:4][bitWidth:4][seed:8][paddedDimensions:4]
    private static let headerSize = 28

    /// Save the finalized index to a file.
    /// The rotation matrix is not stored — it is regenerated from the seed on load.
    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        let hnswData = try state.withLock { state -> Data in
            if !state.finalized {
                guard hnsw_turboquant_finalize(state.index, state.encoder) else {
                    throw HNSWError.serializationFailed("Finalization failed (out of memory)")
                }
                hnsw_turboquant_set_data_size(state.space, _packedSize)
                hnsw_turboquant_set_mode(state.space, 1)
                state.finalized = true
            }

            let hnswSize = hnsw_get_serialized_size(state.index)
            guard hnswSize > 0 else {
                throw HNSWError.serializationFailed("Failed to get serialized size")
            }

            var data = Data(count: hnswSize)
            let success = data.withUnsafeMutableBytes { ptr in
                hnsw_serialize_to_buffer(state.index, ptr.baseAddress!, hnswSize)
            }
            guard success else {
                throw HNSWError.serializationFailed("Failed to serialize HNSW data")
            }
            return data
        }

        // Write header field-by-field (portable, no padding dependency)
        var headerData = Data(capacity: Self.headerSize)
        headerData.appendLittleEndian(Self.headerMagic)
        headerData.appendLittleEndian(Self.headerVersion)
        headerData.appendLittleEndian(UInt32(dimensions))
        headerData.appendLittleEndian(UInt32(bitWidth))
        headerData.appendLittleEndian(seed)
        headerData.appendLittleEndian(UInt32(paddedDimensions))

        var output = headerData
        output.append(hnswData)
        try output.write(to: URL(fileURLWithPath: path))
    }

    /// Load a finalized index from a file.
    /// The index is ready for search immediately after loading.
    public static func load(from url: URL) throws -> TurboQuantIndex {
        try load(from: url.path)
    }

    public static func load(from path: String) throws -> TurboQuantIndex {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))

        guard data.count > headerSize else {
            throw HNSWError.loadFailed("File too small")
        }

        // Read header field-by-field (portable)
        var offset = 0
        let magic: UInt32 = data.readLittleEndian(at: &offset)
        let version: UInt32 = data.readLittleEndian(at: &offset)

        guard magic == headerMagic else {
            throw HNSWError.loadFailed("Invalid file magic")
        }
        guard version == headerVersion else {
            throw HNSWError.loadFailed("Unsupported version \(version)")
        }

        let dimensions = Int(data.readLittleEndian(at: &offset) as UInt32)
        let bitWidth = Int(data.readLittleEndian(at: &offset) as UInt32)
        let seed: UInt64 = data.readLittleEndian(at: &offset)
        let p = Int(data.readLittleEndian(at: &offset) as UInt32)

        // Rebuild encoder from seed (deterministic)
        let quantizer = ScalarQuantizer(bitWidth: bitWidth, dimension: p)

        guard let encoder = quantizer.centroids.withUnsafeBufferPointer({ cBuf in
            quantizer.boundaries.withUnsafeBufferPointer({ bBuf in
                hnsw_tq_encoder_create(
                    dimensions, Int32(bitWidth),
                    cBuf.baseAddress!, Int32(quantizer.numCentroids),
                    bBuf.baseAddress!, Int32(quantizer.boundaries.count), seed)
            })
        }) else {
            throw HNSWError.loadFailed("Failed to recreate encoder")
        }

        let packedSize = hnsw_tq_encoder_packed_size(encoder)

        // Create space and set data_size to packed_size (already finalized)
        guard let space = quantizer.centroids.withUnsafeBufferPointer({ buf in
            hnsw_create_turboquant_l2_space(
                dimensions, p, Int32(bitWidth),
                buf.baseAddress!, Int32(quantizer.numCentroids))
        }) else {
            hnsw_tq_encoder_destroy(encoder)
            throw HNSWError.loadFailed("Failed to create space")
        }
        hnsw_turboquant_set_data_size(space, packedSize)

        // Load HNSW index from the data after the header
        let hnswData = data.dropFirst(offset)
        let loadedIndex: HNSWIndexHandle? = hnswData.withUnsafeBytes { ptr in
            hnsw_load_from_buffer(ptr.baseAddress!, hnswData.count, space, 0)
        }

        guard let loadedIndex else {
            hnsw_destroy_space(space)
            hnsw_tq_encoder_destroy(encoder)
            throw HNSWError.loadFailed("Failed to load HNSW data")
        }

        // Set ADC mode (already finalized)
        hnsw_turboquant_set_mode(space, 1)

        let index = TurboQuantIndex(
            dimensions: dimensions, bitWidth: bitWidth, seed: seed,
            paddedDimensions: p, packedSize: packedSize,
            space: space, index: loadedIndex, encoder: encoder,
            configuration: .balanced, finalized: true
        )
        return index
    }

    /// Private initializer for loading
    private init(
        dimensions: Int, bitWidth: Int, seed: UInt64,
        paddedDimensions: Int, packedSize: Int,
        space: HNSWSpaceHandle, index: HNSWIndexHandle, encoder: TurboQuantEncoderHandle,
        configuration: HNSWConfiguration, finalized: Bool
    ) {
        self.dimensions = dimensions
        self.bitWidth = bitWidth
        self.seed = seed
        self.paddedDimensions = paddedDimensions
        self._packedSize = packedSize
        self.configuration = configuration
        self.state = Mutex(CxxState(
            space: space,
            index: index,
            encoder: encoder,
            finalized: finalized,
            paddedDimensions: paddedDimensions
        ))
    }

    private func ensureRotatedScratch(_ state: inout CxxState) {
        guard state.rotatedScratch.count != paddedDimensions else { return }
        state.rotatedScratch = [Float](repeating: 0, count: paddedDimensions)
    }
}

#else

/// Swift backend TurboQuant facade backed by an exact normalized flat index.
///
/// The optional C++ backend stores an hnswlib graph and switches to ADC after finalization.
/// This implementation preserves the public API,
/// finalization rules, serialization, and cosine-oriented search semantics in Swift.
    public final class TurboQuantIndex: Sendable {

    private struct Entry: Sendable {
        var offset: Int
    }

    private struct State: Sendable {
        var finalized: Bool
        var efSearch: Int
        var entries: [UInt64: Entry]
        var labelOrder: [UInt64]
        var vectorStorage: [Float]
        var queryScratch: [Float]
    }

    public let dimensions: Int
    public let bitWidth: Int
    public let configuration: HNSWConfiguration
    public let seed: UInt64
    public let paddedDimensions: Int

    private let _packedSize: Int
    private let maximumElementCount: Int
    private let state: Mutex<State>

    public init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int = 4,
        configuration: HNSWConfiguration = .balanced,
        seed: UInt64 = 42
    ) throws {
        guard dimensions > 0 else {
            throw HNSWError.initializationFailed("dimensions must be positive")
        }
        guard maxElements > 0 else {
            throw HNSWError.initializationFailed("maxElements must be positive")
        }
        guard (1...4).contains(bitWidth) else {
            throw HNSWError.initializationFailed("bitWidth must be 1, 2, 3, or 4")
        }

        self.dimensions = dimensions
        self.bitWidth = bitWidth
        self.configuration = configuration
        self.seed = seed

        var padded = 1
        while padded < dimensions { padded *= 2 }
        self.paddedDimensions = padded

        self._packedSize = BitPacking.packedSize(count: padded, bitWidth: bitWidth)
        self.maximumElementCount = maxElements
        var vectorStorage: [Float] = []
        if maxElements <= Int.max / dimensions {
            vectorStorage.reserveCapacity(maxElements * dimensions)
        }
        self.state = Mutex(State(
            finalized: false,
            efSearch: configuration.efSearch,
            entries: [:],
            labelOrder: [],
            vectorStorage: vectorStorage,
            queryScratch: [Float](repeating: 0, count: dimensions)
        ))
    }

    public var count: Int {
        state.withLock { $0.entries.count }
    }

    public var capacity: Int { maximumElementCount }
    public var isEmpty: Bool { count == 0 }
    public var isFinalized: Bool { state.withLock { $0.finalized } }
    public var bytesPerVector: Int { _packedSize }
    public var compressionRatio: Float { Float(dimensions * 4) / Float(_packedSize) }

    public func setEfSearch(_ ef: Int) {
        state.withLock {
            $0.efSearch = max(1, ef)
        }
    }

    public func add(_ vector: [Float], label: UInt64) throws {
        try vector.withUnsafeBufferPointer { buffer in
            try add(buffer, label: label)
        }
    }

    /// Add a borrowed vector without materializing an intermediate array.
    public func add(_ vector: UnsafeBufferPointer<Float>, label: UInt64) throws {
        guard vector.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vector.count)
        }

        try state.withLock {
            guard !$0.finalized else {
                throw HNSWError.addPointFailed("Cannot add after finalize()")
            }
            let offset: Int
            if let existing = $0.entries[label] {
                offset = existing.offset
            } else {
                guard $0.entries.count < maximumElementCount else {
                    throw HNSWError.capacityExceeded(
                        current: $0.entries.count,
                        maximum: maximumElementCount
                    )
                }
                offset = $0.vectorStorage.count
                for _ in 0..<dimensions {
                    $0.vectorStorage.append(0)
                }
                $0.labelOrder.append(label)
            }
            $0.vectorStorage.withUnsafeMutableBufferPointer { storage in
                let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + offset, count: dimensions)
                VectorOperations.normalize(vector, into: destination)
            }
            $0.entries[label] = Entry(offset: offset)
        }
    }

    public func search(_ query: [Float], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Search with a borrowed query vector.
    public func search(_ query: UnsafeBufferPointer<Float>, k: Int) throws -> [SearchResult] {
        guard query.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: query.count)
        }
        guard k > 0 else { return [] }

        state.withLock {
            $0.finalized = true
        }

        return state.withLock { state in
            ensureQueryScratch(&state)
            state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                VectorOperations.normalize(query, into: scratch)
            }
            var results: [SearchResult] = []
            results.reserveCapacity(min(k, state.entries.count))

            state.queryScratch.withUnsafeBufferPointer { normalizedQuery in
                state.vectorStorage.withUnsafeBufferPointer { storage in
                    for label in state.labelOrder {
                        guard let entry = state.entries[label] else {
                            continue
                        }
                        let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                        let distance = VectorOperations.distance(from: normalizedQuery, to: vector, metric: .cosine)
                        insertTopKSearchResult(
                            SearchResult(label: label, distance: distance),
                            into: &results,
                            limit: k
                        )
                    }
                }
            }
            return results
        }
    }

    private static let headerMagic: UInt32 = 0x54515746 // "TQWF"
    private static let headerVersion: UInt32 = 1
    private static let headerSize = 40

    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        let data = state.withLock {
            $0.finalized = true
            var output = Data(capacity: Self.headerSize + $0.entries.count * (8 + dimensions * 4))
            output.appendLittleEndian(Self.headerMagic)
            output.appendLittleEndian(Self.headerVersion)
            output.appendLittleEndian(UInt32(dimensions))
            output.appendLittleEndian(UInt32(bitWidth))
            output.appendLittleEndian(seed)
            output.appendLittleEndian(UInt32(paddedDimensions))
            output.appendLittleEndian(UInt32(maximumElementCount))
            output.appendLittleEndian(UInt32($0.labelOrder.count))

            for label in $0.labelOrder {
                guard let entry = $0.entries[label] else {
                    continue
                }
                output.appendLittleEndian(label)
                $0.vectorStorage.withUnsafeBufferPointer { storage in
                    let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                    for value in vector {
                        output.appendLittleEndian(value.bitPattern)
                    }
                }
            }
            return output
        }

        do {
            try data.write(to: URL(fileURLWithPath: path))
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

        guard data.count >= headerSize else {
            throw HNSWError.loadFailed("File too small")
        }

        var offset = 0
        let magic: UInt32 = data.readLittleEndian(at: &offset)
        let version: UInt32 = data.readLittleEndian(at: &offset)

        guard magic == headerMagic else {
            throw HNSWError.loadFailed("Invalid Swift backend TurboQuant file magic")
        }
        guard version == headerVersion else {
            throw HNSWError.loadFailed("Unsupported Swift backend TurboQuant version \(version)")
        }

        let dimensions = Int(data.readLittleEndian(at: &offset) as UInt32)
        let bitWidth = Int(data.readLittleEndian(at: &offset) as UInt32)
        let seed: UInt64 = data.readLittleEndian(at: &offset)
        let paddedDimensions = Int(data.readLittleEndian(at: &offset) as UInt32)
        let capacity = Int(data.readLittleEndian(at: &offset) as UInt32)
        let labelCount = Int(data.readLittleEndian(at: &offset) as UInt32)

        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: max(capacity, labelCount),
            bitWidth: bitWidth,
            configuration: .balanced,
            seed: seed
        )

        guard index.paddedDimensions == paddedDimensions else {
            throw HNSWError.loadFailed("Stored padded dimension does not match dimensions")
        }

        try index.state.withLock {
            for _ in 0..<labelCount {
                guard offset + 8 + dimensions * 4 <= data.count else {
                    throw HNSWError.loadFailed("Swift backend TurboQuant data is truncated")
                }

                let label: UInt64 = data.readLittleEndian(at: &offset)
                let vectorOffset = $0.vectorStorage.count
                $0.vectorStorage.reserveCapacity($0.vectorStorage.count + dimensions)
                for _ in 0..<dimensions {
                    let bitPattern: UInt32 = data.readLittleEndian(at: &offset)
                    $0.vectorStorage.append(Float(bitPattern: bitPattern))
                }
                $0.entries[label] = Entry(offset: vectorOffset)
                $0.labelOrder.append(label)
            }

            guard offset == data.count else {
                throw HNSWError.loadFailed("Swift backend TurboQuant data has trailing bytes")
            }
            $0.finalized = true
        }

        return index
    }

    private func ensureQueryScratch(_ state: inout State) {
        guard state.queryScratch.count != dimensions else { return }
        state.queryScratch = [Float](repeating: 0, count: dimensions)
    }
}

#endif

// MARK: - Data Serialization Helpers

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }

    func readLittleEndian<T: FixedWidthInteger>(at offset: inout Int) -> T {
        let size = MemoryLayout<T>.size
        let value = withUnsafeBytes { buf in
            buf.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += size
        return T(littleEndian: value)
    }
}
