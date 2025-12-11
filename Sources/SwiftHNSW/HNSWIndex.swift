import Foundation
import hnswlib

/// HNSW Index for approximate nearest neighbor search
public final class HNSWIndex: @unchecked Sendable {

    // MARK: - Properties

    /// Vector dimensions
    public let dimensions: Int

    /// Distance metric used by this index
    public let metric: DistanceMetric

    /// Index configuration
    public let configuration: HNSWConfiguration

    private let space: HNSWSpaceHandle
    private let index: HNSWIndexHandle
    private let lock = RWLock()

    // MARK: - Initialization

    /// Create a new HNSW index
    /// - Parameters:
    ///   - dimensions: Vector dimensions
    ///   - maxElements: Maximum number of elements in the index
    ///   - metric: Distance metric to use
    ///   - configuration: Index configuration
    public init(
        dimensions: Int,
        maxElements: Int,
        metric: DistanceMetric = .l2,
        configuration: HNSWConfiguration = .balanced
    ) throws {
        self.dimensions = dimensions
        self.metric = metric
        self.configuration = configuration

        guard let space = metric.createSpace(dimensions: dimensions) else {
            throw HNSWError.initializationFailed("Failed to create distance space")
        }
        self.space = space

        guard let index = hnsw_create_index(
            space,
            maxElements,
            configuration.m,
            configuration.efConstruction,
            configuration.randomSeed,
            configuration.allowReplaceDeleted
        ) else {
            hnsw_destroy_space(space)
            throw HNSWError.initializationFailed("Failed to create HNSW index")
        }
        self.index = index
        hnsw_set_ef(index, configuration.efSearch)
    }

    /// Private initializer for loading from file
    private init(
        dimensions: Int,
        metric: DistanceMetric,
        configuration: HNSWConfiguration,
        space: HNSWSpaceHandle,
        index: HNSWIndexHandle
    ) {
        self.dimensions = dimensions
        self.metric = metric
        self.configuration = configuration
        self.space = space
        self.index = index
    }

    deinit {
        hnsw_destroy_index(index)
        hnsw_destroy_space(space)
    }

    // MARK: - Count & Capacity

    /// Current number of elements in the index
    public var count: Int {
        lock.withReadLock {
            Int(hnsw_get_current_count(index))
        }
    }

    /// Maximum number of elements the index can hold
    public var capacity: Int {
        lock.withReadLock {
            Int(hnsw_get_max_elements(index))
        }
    }

    /// Whether the index is empty
    public var isEmpty: Bool { count == 0 }

    // MARK: - Configuration

    /// Set the ef parameter for search
    /// - Parameter ef: Higher values improve recall but increase search time
    public func setEfSearch(_ ef: Int) {
        lock.withWriteLock {
            hnsw_set_ef(index, ef)
        }
    }

    /// Resize the index to accommodate more elements
    /// - Parameter newCapacity: New maximum capacity
    public func resize(to newCapacity: Int) throws {
        try lock.withWriteLock {
            guard hnsw_resize_index(index, newCapacity) else {
                throw HNSWError.addPointFailed("Failed to resize index to \(newCapacity)")
            }
        }
    }
}

// MARK: - Single Operations

extension HNSWIndex {

    /// Add a vector to the index
    /// - Parameters:
    ///   - vector: The vector to add
    ///   - label: Unique identifier for this vector
    public func add(_ vector: [Float], label: UInt64) throws {
        try validateDimensions(vector.count)

        let processedVector = metric.requiresNormalization
            ? VectorOperations.normalize(vector)
            : vector

        try lock.withWriteLock {
            let success = processedVector.withUnsafeBufferPointer { buffer in
                hnsw_add_point(index, buffer.baseAddress!, label)
            }
            guard success else {
                throw HNSWError.addPointFailed("Failed to add point with label \(label)")
            }
        }
    }

    /// Search for k nearest neighbors
    /// - Parameters:
    ///   - query: The query vector
    ///   - k: Number of nearest neighbors to find
    /// - Returns: Array of search results sorted by distance (closest first)
    public func search(_ query: [Float], k: Int) throws -> [SearchResult] {
        try validateDimensions(query.count)

        let processedQuery = metric.requiresNormalization
            ? VectorOperations.normalize(query)
            : query

        return lock.withReadLock {
            var labels = [UInt64](repeating: 0, count: k)
            var distances = [Float](repeating: 0, count: k)

            let resultCount = processedQuery.withUnsafeBufferPointer { queryBuffer in
                labels.withUnsafeMutableBufferPointer { labelsBuffer in
                    distances.withUnsafeMutableBufferPointer { distancesBuffer in
                        hnsw_search_knn(
                            index,
                            queryBuffer.baseAddress!,
                            Int32(k),
                            labelsBuffer.baseAddress!,
                            distancesBuffer.baseAddress!
                        )
                    }
                }
            }

            return (0..<Int(resultCount)).map { i in
                SearchResult(label: labels[i], distance: distances[i])
            }
        }
    }

    /// Mark an element as deleted
    /// - Parameter label: The label of the element to delete
    public func markDeleted(label: UInt64) throws {
        try lock.withWriteLock {
            guard hnsw_mark_deleted(index, label) else {
                throw HNSWError.deleteFailed("Failed to delete element with label \(label)")
            }
        }
    }

    /// Unmark a deleted element
    /// - Parameter label: The label of the element to restore
    public func unmarkDeleted(label: UInt64) throws {
        try lock.withWriteLock {
            guard hnsw_unmark_deleted(index, label) else {
                throw HNSWError.deleteFailed("Failed to undelete element with label \(label)")
            }
        }
    }
}

// MARK: - Batch Operations

extension HNSWIndex {

    /// Add multiple vectors to the index in batch
    /// - Parameters:
    ///   - vectors: Flattened array of vectors
    ///   - labels: Labels for each vector
    /// - Returns: Number of successfully added points
    @discardableResult
    public func addBatch(_ vectors: [Float], labels: [UInt64]) throws -> Int {
        let numVectors = labels.count
        try validateDimensions(vectors.count, expectedTotal: numVectors * dimensions)

        let processedVectors = metric.requiresNormalization
            ? VectorOperations.normalizeBatch(vectors, count: numVectors, dimensions: dimensions)
            : vectors

        return lock.withWriteLock {
            Int(processedVectors.withUnsafeBufferPointer { vectorsBuffer in
                labels.withUnsafeBufferPointer { labelsBuffer in
                    hnsw_add_points_batch(
                        index,
                        vectorsBuffer.baseAddress!,
                        labelsBuffer.baseAddress!,
                        numVectors,
                        dimensions
                    )
                }
            })
        }
    }

    /// Add multiple vectors with auto-generated labels
    /// - Parameters:
    ///   - vectors: Array of vectors
    ///   - startingLabel: Starting label (default: current count)
    /// - Returns: Number of successfully added points
    @discardableResult
    public func addBatch(_ vectors: [[Float]], startingLabel: UInt64? = nil) throws -> Int {
        guard !vectors.isEmpty else { return 0 }
        guard vectors.allSatisfy({ $0.count == dimensions }) else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vectors.first?.count ?? 0)
        }

        let start = startingLabel ?? UInt64(count)
        let labels = (0..<vectors.count).map { start + UInt64($0) }
        let flattened = vectors.flatMap { $0 }

        return try addBatch(flattened, labels: labels)
    }

    /// Search for k nearest neighbors for multiple queries
    /// - Parameters:
    ///   - queries: Flattened array of query vectors
    ///   - numQueries: Number of queries
    ///   - k: Number of nearest neighbors per query
    /// - Returns: Array of search results for each query
    public func searchBatch(_ queries: [Float], numQueries: Int, k: Int) throws -> [[SearchResult]] {
        try validateDimensions(queries.count, expectedTotal: numQueries * dimensions)

        let processedQueries = metric.requiresNormalization
            ? VectorOperations.normalizeBatch(queries, count: numQueries, dimensions: dimensions)
            : queries

        return lock.withReadLock {
            var labels = [UInt64](repeating: 0, count: numQueries * k)
            var distances = [Float](repeating: 0, count: numQueries * k)

            processedQueries.withUnsafeBufferPointer { queriesBuffer in
                labels.withUnsafeMutableBufferPointer { labelsBuffer in
                    distances.withUnsafeMutableBufferPointer { distancesBuffer in
                        _ = hnsw_search_knn_batch(
                            index,
                            queriesBuffer.baseAddress!,
                            numQueries,
                            dimensions,
                            Int32(k),
                            labelsBuffer.baseAddress!,
                            distancesBuffer.baseAddress!
                        )
                    }
                }
            }

            return (0..<numQueries).map { q in
                (0..<k).compactMap { i in
                    let idx = q * k + i
                    let label = labels[idx]
                    let distance = distances[idx]
                    // Filter out empty results (except first which might be valid)
                    guard i == 0 || label != 0 || distance != 0 else { return nil }
                    return SearchResult(label: label, distance: distance)
                }
            }
        }
    }

    /// Search for k nearest neighbors for multiple queries
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - k: Number of nearest neighbors per query
    /// - Returns: Array of search results for each query
    public func searchBatch(_ queries: [[Float]], k: Int) throws -> [[SearchResult]] {
        guard !queries.isEmpty else { return [] }
        guard queries.allSatisfy({ $0.count == dimensions }) else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: queries.first?.count ?? 0)
        }

        let flattened = queries.flatMap { $0 }
        return try searchBatch(flattened, numQueries: queries.count, k: k)
    }
}

// MARK: - Persistence

extension HNSWIndex {

    /// Save the index to a file
    /// - Parameter url: File URL to save to
    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    /// Save the index to a file
    /// - Parameter path: File path to save to
    public func save(to path: String) throws {
        try lock.withReadLock {
            guard hnsw_save_index(index, path) else {
                throw HNSWError.saveFailed("Failed to save index to \(path)")
            }
        }
    }

    /// Load an index from a file
    /// - Parameters:
    ///   - url: File URL to load from
    ///   - dimensions: Vector dimensions
    ///   - metric: Distance metric
    ///   - maxElements: Maximum elements (0 to use saved value)
    /// - Returns: Loaded HNSW index
    public static func load(
        from url: URL,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        try load(from: url.path, dimensions: dimensions, metric: metric, maxElements: maxElements)
    }

    /// Load an index from a file
    /// - Parameters:
    ///   - path: File path to load from
    ///   - dimensions: Vector dimensions
    ///   - metric: Distance metric
    ///   - maxElements: Maximum elements (0 to use saved value)
    /// - Returns: Loaded HNSW index
    public static func load(
        from path: String,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        guard let space = metric.createSpace(dimensions: dimensions) else {
            throw HNSWError.loadFailed("Failed to create distance space")
        }

        guard let loadedIndex = hnsw_load_index(path, space, maxElements) else {
            hnsw_destroy_space(space)
            throw HNSWError.loadFailed("Failed to load index from \(path)")
        }

        return HNSWIndex(
            dimensions: dimensions,
            metric: metric,
            configuration: .balanced,
            space: space,
            index: loadedIndex
        )
    }
}

// MARK: - Validation

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
}

