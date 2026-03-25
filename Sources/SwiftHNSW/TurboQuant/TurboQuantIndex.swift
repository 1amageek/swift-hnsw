import Foundation
import hnswlib

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
public final class TurboQuantIndex: @unchecked Sendable {

    // MARK: - Properties

    public let dimensions: Int
    public let bitWidth: Int
    public let configuration: HNSWConfiguration
    public let seed: UInt64

    /// Padded dimension (next power of 2). All internal operations use this.
    public let paddedDimensions: Int

    private let space: HNSWSpaceHandle
    private let index: HNSWIndexHandle
    private let encoder: TurboQuantEncoderHandle
    private let lock: RWLock
    private let _packedSize: Int
    private var _finalized: Bool

    // MARK: - Initialization

    public init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int = 4,
        configuration: HNSWConfiguration = .balanced,
        seed: UInt64 = 42
    ) throws {
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
        self.encoder = encoder
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
        self.space = space

        guard let index = hnsw_create_index(
            space, maxElements, configuration.m, configuration.efConstruction,
            configuration.randomSeed, configuration.allowReplaceDeleted
        ) else {
            hnsw_destroy_space(space)
            hnsw_tq_encoder_destroy(encoder)
            throw HNSWError.initializationFailed("Failed to create index")
        }
        self.index = index
        self.lock = RWLock()
        self._finalized = false
        hnsw_set_ef(index, configuration.efSearch)
    }

    deinit {
        hnsw_destroy_index(index)
        hnsw_destroy_space(space)
        hnsw_tq_encoder_destroy(encoder)
    }

    // MARK: - Info

    public var count: Int { lock.withReadLock { Int(hnsw_get_current_count(index)) } }
    public var capacity: Int { lock.withReadLock { Int(hnsw_get_max_elements(index)) } }
    public var isEmpty: Bool { count == 0 }
    public var isFinalized: Bool { _finalized }
    public var bytesPerVector: Int { _packedSize }
    public var compressionRatio: Float { Float(dimensions * 4) / Float(_packedSize) }

    public func setEfSearch(_ ef: Int) {
        lock.withWriteLock { hnsw_set_ef(index, ef) }
    }

    // MARK: - Add

    /// Add a vector. Must be called BEFORE searching.
    /// Internally: normalize → HD³ rotate to p dims → store as Float32[p].
    public func add(_ vector: [Float], label: UInt64) throws {
        guard !_finalized else {
            throw HNSWError.addPointFailed("Cannot add after finalize()")
        }
        guard vector.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vector.count)
        }

        // Normalize + rotate → p floats (C++)
        var rotated = [Float](repeating: 0, count: paddedDimensions)
        vector.withUnsafeBufferPointer { vBuf in
            rotated.withUnsafeMutableBufferPointer { rBuf in
                hnsw_tq_encoder_rotate_query(encoder, vBuf.baseAddress!, rBuf.baseAddress!)
            }
        }

        try lock.withWriteLock {
            hnsw_turboquant_set_mode(space, 0)
            let success = rotated.withUnsafeBufferPointer { buf in
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
        guard query.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: query.count)
        }

        lock.withWriteLock {
            if !_finalized {
                hnsw_turboquant_finalize(index, encoder)
                hnsw_turboquant_set_mode(space, 1)
                _finalized = true
            }
        }

        // Normalize + rotate query → p floats (full precision for ADC)
        var rotated = [Float](repeating: 0, count: paddedDimensions)
        query.withUnsafeBufferPointer { qBuf in
            rotated.withUnsafeMutableBufferPointer { rBuf in
                hnsw_tq_encoder_rotate_query(encoder, qBuf.baseAddress!, rBuf.baseAddress!)
            }
        }

        return lock.withReadLock {
            var labels = [UInt64](repeating: 0, count: k)
            var distances = [Float](repeating: 0, count: k)

            let resultCount = rotated.withUnsafeBufferPointer { queryBuf in
                labels.withUnsafeMutableBufferPointer { labelsBuf in
                    distances.withUnsafeMutableBufferPointer { distBuf in
                        hnsw_search_knn(index, queryBuf.baseAddress!, Int32(k),
                                        labelsBuf.baseAddress!, distBuf.baseAddress!)
                    }
                }
            }

            return (0..<Int(resultCount)).map { i in
                SearchResult(label: labels[i], distance: distances[i])
            }
        }
    }

    // MARK: - Persistence

    /// File header for TurboQuant serialization
    private struct Header {
        static let magic: UInt32 = 0x54514857 // "TQHW"
        static let version: UInt32 = 1

        var magic: UInt32
        var version: UInt32
        var dimensions: UInt32
        var bitWidth: UInt32
        var seed: UInt64
        var paddedDimensions: UInt32
        var reserved: UInt32
    }

    /// Save the finalized index to a file.
    /// The rotation matrix is not stored — it is regenerated from the seed on load.
    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        // Auto-finalize if needed
        lock.withWriteLock {
            if !_finalized {
                hnsw_turboquant_finalize(index, encoder)
                hnsw_turboquant_set_mode(space, 1)
                _finalized = true
            }
        }

        // Write header + HNSW data
        var header = Header(
            magic: Header.magic,
            version: Header.version,
            dimensions: UInt32(dimensions),
            bitWidth: UInt32(bitWidth),
            seed: seed,
            paddedDimensions: UInt32(paddedDimensions),
            reserved: 0
        )

        let headerData = Data(bytes: &header, count: MemoryLayout<Header>.size)

        // Get HNSW serialized data
        let hnswSize = lock.withReadLock { hnsw_get_serialized_size(index) }
        guard hnswSize > 0 else {
            throw HNSWError.serializationFailed("Failed to get serialized size")
        }

        var hnswData = Data(count: hnswSize)
        let success = lock.withReadLock {
            hnswData.withUnsafeMutableBytes { ptr in
                hnsw_serialize_to_buffer(index, ptr.baseAddress!, hnswSize)
            }
        }
        guard success else {
            throw HNSWError.serializationFailed("Failed to serialize HNSW data")
        }

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
        let headerSize = MemoryLayout<Header>.size

        guard data.count > headerSize else {
            throw HNSWError.loadFailed("File too small")
        }

        // Read header
        let header: Header = data.withUnsafeBytes { buf in
            buf.load(as: Header.self)
        }

        guard header.magic == Header.magic else {
            throw HNSWError.loadFailed("Invalid file magic")
        }
        guard header.version == Header.version else {
            throw HNSWError.loadFailed("Unsupported version \(header.version)")
        }

        let dimensions = Int(header.dimensions)
        let bitWidth = Int(header.bitWidth)
        let seed = header.seed
        let p = Int(header.paddedDimensions)

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

        // Create space with packed_size as data_size (already finalized)
        guard let space = quantizer.centroids.withUnsafeBufferPointer({ buf in
            hnsw_create_turboquant_l2_space(
                dimensions, p, Int32(bitWidth),
                buf.baseAddress!, Int32(quantizer.numCentroids))
        }) else {
            hnsw_tq_encoder_destroy(encoder)
            throw HNSWError.loadFailed("Failed to create space")
        }

        // Load HNSW index from the data after the header
        let hnswData = data.dropFirst(headerSize)
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
        self.space = space
        self.index = index
        self.encoder = encoder
        self.configuration = configuration
        self._finalized = finalized
        self.lock = RWLock()
    }
}
