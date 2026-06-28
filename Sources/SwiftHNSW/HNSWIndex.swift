import Foundation
import Synchronization
#if HNSWLIB_BACKEND && canImport(hnswlib)
import hnswlib

/// HNSW Index for approximate nearest neighbor search
/// Generic over scalar type (Float or Float16)
public final class HNSWIndex<Scalar: HNSWNativeScalar>: Sendable {

    // MARK: - Properties

    /// Vector dimensions
    public let dimensions: Int

    /// Distance metric used by this index
    public let metric: DistanceMetric

    /// Index configuration
    public let configuration: HNSWConfiguration

    private struct CxxState: Sendable {
        let spaceAddress: UInt
        let indexAddress: UInt
        var vectorScratch: [Scalar]
        var batchScratch: [Scalar]

        init(space: HNSWSpaceHandle, index: HNSWIndexHandle) {
            self.spaceAddress = UInt(bitPattern: space)
            self.indexAddress = UInt(bitPattern: index)
            self.vectorScratch = []
            self.batchScratch = []
        }

        var space: HNSWSpaceHandle {
            UnsafeMutableRawPointer(bitPattern: spaceAddress)!
        }

        var index: HNSWIndexHandle {
            UnsafeMutableRawPointer(bitPattern: indexAddress)!
        }
    }

    private let state: Mutex<CxxState>

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

        guard let space = metric.createSpace(dimensions: dimensions, scalar: Scalar.self) else {
            throw HNSWError.initializationFailed("Failed to create distance space")
        }

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
        hnsw_set_ef(index, configuration.efSearch)
        self.state = Mutex(CxxState(space: space, index: index))
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
        self.state = Mutex(CxxState(space: space, index: index))
    }

    deinit {
        state.withLock { state in
            hnsw_destroy_index(state.index)
            hnsw_destroy_space(state.space)
        }
    }

    // MARK: - Count & Capacity

    /// Current number of elements in the index
    public var count: Int {
        state.withLock { state in
            Int(hnsw_get_current_count(state.index))
        }
    }

    /// Maximum number of elements the index can hold
    public var capacity: Int {
        state.withLock { state in
            Int(hnsw_get_max_elements(state.index))
        }
    }

    /// Whether the index is empty
    public var isEmpty: Bool { count == 0 }

    // MARK: - Configuration

    /// Set the ef parameter for search
    /// - Parameter ef: Higher values improve recall but increase search time
    public func setEfSearch(_ ef: Int) {
        state.withLock { state in
            hnsw_set_ef(state.index, ef)
        }
    }

    /// Resize the index to accommodate more elements
    /// - Parameter newCapacity: New maximum capacity
    public func resize(to newCapacity: Int) throws {
        try state.withLock { state in
            guard hnsw_resize_index(state.index, newCapacity) else {
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
    public func add(_ vector: [Scalar], label: UInt64) throws {
        try vector.withUnsafeBufferPointer { buffer in
            try add(buffer, label: label)
        }
    }

    /// Add a borrowed vector to the index without materializing an intermediate array.
    ///
    /// The index copies the vector into its owned storage or native backend during this call.
    /// The caller retains ownership of the provided memory after the method returns.
    public func add(_ vector: UnsafeBufferPointer<Scalar>, label: UInt64) throws {
        try validateDimensions(vector.count)

        try state.withLock { state in
            let success: Bool
            if metric.requiresNormalization {
                let index = state.index
                ensureCapacity(&state.vectorScratch, count: dimensions)
                success = state.vectorScratch.withUnsafeMutableBufferPointer { scratch in
                    normalizeVector(vector, into: scratch)
                    return Scalar.addPoint(index, data: scratch.baseAddress!, label: label)
                }
            } else {
                success = Scalar.addPoint(state.index, data: vector.baseAddress!, label: label)
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
    public func search(_ query: [Scalar], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Search with a borrowed query vector without materializing an intermediate array.
    public func search(_ query: UnsafeBufferPointer<Scalar>, k: Int) throws -> [SearchResult] {
        try validateDimensions(query.count)
        guard k > 0 else { return [] }

        return state.withLock { state in
            if metric.requiresNormalization {
                let index = state.index
                ensureCapacity(&state.vectorScratch, count: dimensions)
                return state.vectorScratch.withUnsafeMutableBufferPointer { scratch in
                    normalizeVector(query, into: scratch)
                    let normalized = UnsafeBufferPointer(start: scratch.baseAddress, count: scratch.count)
                    return searchNormalized(normalized, k: k, index: index)
                }
            }
            return searchNormalized(query, k: k, index: state.index)
        }
    }

    /// Mark an element as deleted
    /// - Parameter label: The label of the element to delete
    public func markDeleted(label: UInt64) throws {
        try state.withLock { state in
            guard hnsw_mark_deleted(state.index, label) else {
                throw HNSWError.deleteFailed("Failed to delete element with label \(label)")
            }
        }
    }

    /// Unmark a deleted element
    /// - Parameter label: The label of the element to restore
    public func unmarkDeleted(label: UInt64) throws {
        try state.withLock { state in
            guard hnsw_unmark_deleted(state.index, label) else {
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
        guard numVectors > 0 else { return 0 }
        try validateDimensions(vectors.count, expectedTotal: numVectors * dimensions)

        return state.withLock { state in
            let added: Int32
            if metric.requiresNormalization {
                let index = state.index
                ensureCapacity(&state.batchScratch, count: vectors.count)
                added = state.batchScratch.withUnsafeMutableBufferPointer { scratch in
                    normalizeVectorsBatch(vectors, count: numVectors, dimensions: dimensions, into: scratch)
                    return Scalar.addPointsBatch(
                        index,
                        data: scratch.baseAddress!,
                        labels: labels.baseAddress!,
                        numPoints: numVectors,
                        dimension: dimensions
                    )
                }
            } else {
                added = Scalar.addPointsBatch(
                    state.index,
                    data: vectors.baseAddress!,
                    labels: labels.baseAddress!,
                    numPoints: numVectors,
                    dimension: dimensions
                )
            }
            return Int(added)
        }
    }

    /// Add multiple vectors with auto-generated labels
    /// - Parameters:
    ///   - vectors: Array of vectors
    ///   - startingLabel: Starting label (default: current count)
    /// - Returns: Number of successfully added points
    @discardableResult
    public func addBatch(_ vectors: [[Scalar]], startingLabel: UInt64? = nil) throws -> Int {
        guard !vectors.isEmpty else { return 0 }
        guard vectors.allSatisfy({ $0.count == dimensions }) else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vectors.first?.count ?? 0)
        }

        let start = startingLabel ?? UInt64(count)
        var addedCount = 0
        for index in vectors.indices {
            try add(vectors[index], label: start + UInt64(index))
            addedCount += 1
        }
        return addedCount
    }

    /// Search for k nearest neighbors for multiple queries
    /// - Parameters:
    ///   - queries: Flattened array of query vectors
    ///   - numQueries: Number of queries
    ///   - k: Number of nearest neighbors per query
    /// - Returns: Array of search results for each query
    public func searchBatch(_ queries: [Scalar], numQueries: Int, k: Int) throws -> [[SearchResult]] {
        try queries.withUnsafeBufferPointer { buffer in
            try searchBatch(buffer, numQueries: numQueries, k: k)
        }
    }

    /// Search borrowed contiguous query vectors.
    ///
    /// `queries` must contain `numQueries * dimensions` scalars in row-major order.
    public func searchBatch(
        _ queries: UnsafeBufferPointer<Scalar>,
        numQueries: Int,
        k: Int
    ) throws -> [[SearchResult]] {
        try validateDimensions(queries.count, expectedTotal: numQueries * dimensions)
        guard numQueries > 0 else { return [] }
        guard k > 0 else { return Array(repeating: [], count: numQueries) }

        return state.withLock { state in
            var labels = [UInt64](repeating: 0, count: numQueries * k)
            var distances = [Float](repeating: 0, count: numQueries * k)

            let resultCounts: Int32
            if metric.requiresNormalization {
                let index = state.index
                ensureCapacity(&state.batchScratch, count: queries.count)
                resultCounts = state.batchScratch.withUnsafeMutableBufferPointer { scratch in
                    normalizeVectorsBatch(queries, count: numQueries, dimensions: dimensions, into: scratch)
                    let normalized = UnsafeBufferPointer(start: scratch.baseAddress, count: scratch.count)
                    return searchBatchNormalized(
                        normalized,
                        numQueries: numQueries,
                        k: k,
                        labels: &labels,
                        distances: &distances,
                        index: index
                    )
                }
            } else {
                resultCounts = searchBatchNormalized(
                    queries,
                    numQueries: numQueries,
                    k: k,
                    labels: &labels,
                    distances: &distances,
                    index: state.index
                )
            }

            return buildBatchResults(
                labels: labels,
                distances: distances,
                numQueries: numQueries,
                k: k,
                resultCounts: Int(resultCounts)
            )
        }
    }

    /// Search for k nearest neighbors for multiple queries
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - k: Number of nearest neighbors per query
    /// - Returns: Array of search results for each query
    public func searchBatch(_ queries: [[Scalar]], k: Int) throws -> [[SearchResult]] {
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
        try state.withLock { state in
            guard hnsw_save_index(state.index, path) else {
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
        guard let space = metric.createSpace(dimensions: dimensions, scalar: Scalar.self) else {
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

    @inline(__always)
    private func ensureCapacity(_ buffer: inout [Scalar], count: Int) {
        guard buffer.count != count else { return }
        buffer = [Scalar](repeating: .zero, count: count)
    }

    private func normalizeVector(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            let outputFloats = UnsafeMutableBufferPointer<Float>(
                start: UnsafeMutableRawPointer(output.baseAddress!).assumingMemoryBound(to: Float.self),
                count: output.count
            )
            VectorOperations.normalize(inputFloats, into: outputFloats)
        } else {
            VectorOperations.normalize(input, into: output)
        }
    }

    private func normalizeVectorsBatch(
        _ input: UnsafeBufferPointer<Scalar>,
        count: Int,
        dimensions: Int,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            let outputFloats = UnsafeMutableBufferPointer<Float>(
                start: UnsafeMutableRawPointer(output.baseAddress!).assumingMemoryBound(to: Float.self),
                count: output.count
            )
            VectorOperations.normalizeBatch(inputFloats, count: count, dimensions: dimensions, into: outputFloats)
        } else {
            VectorOperations.normalizeBatch(input, count: count, dimensions: dimensions, into: output)
        }
    }

    private func searchNormalized(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        index: HNSWIndexHandle
    ) -> [SearchResult] {
        var labels = [UInt64](repeating: 0, count: k)
        var distances = [Float](repeating: 0, count: k)

        let resultCount = labels.withUnsafeMutableBufferPointer { labelsBuffer in
            distances.withUnsafeMutableBufferPointer { distancesBuffer in
                Scalar.searchKnn(
                    index,
                    query: query.baseAddress!,
                    k: Int32(k),
                    labels: labelsBuffer.baseAddress!,
                    distances: distancesBuffer.baseAddress!
                )
            }
        }

        var results: [SearchResult] = []
        results.reserveCapacity(Int(resultCount))
        for index in 0..<Int(resultCount) {
            results.append(SearchResult(label: labels[index], distance: distances[index]))
        }
        return results
    }

    private func searchBatchNormalized(
        _ queries: UnsafeBufferPointer<Scalar>,
        numQueries: Int,
        k: Int,
        labels: inout [UInt64],
        distances: inout [Float],
        index: HNSWIndexHandle
    ) -> Int32 {
        labels.withUnsafeMutableBufferPointer { labelsBuffer in
            distances.withUnsafeMutableBufferPointer { distancesBuffer in
                Scalar.searchKnnBatch(
                    index,
                    queries: queries.baseAddress!,
                    numQueries: numQueries,
                    dimension: dimensions,
                    k: Int32(k),
                    labels: labelsBuffer.baseAddress!,
                    distances: distancesBuffer.baseAddress!
                )
            }
        }
    }

    private func buildBatchResults(
        labels: [UInt64],
        distances: [Float],
        numQueries: Int,
        k: Int,
        resultCounts: Int
    ) -> [[SearchResult]] {
        var output: [[SearchResult]] = []
        output.reserveCapacity(numQueries)
        for queryIndex in 0..<numQueries {
            var queryResults: [SearchResult] = []
            queryResults.reserveCapacity(k)
            for resultIndex in 0..<k {
                let index = queryIndex * k + resultIndex
                let label = labels[index]
                let distance = distances[index]
                guard resultIndex == 0 || label != 0 || distance != 0 else { continue }
                queryResults.append(SearchResult(label: label, distance: distance))
            }
            output.append(queryResults)
        }
        _ = resultCounts
        return output
    }
}

// MARK: - Label Operations

extension HNSWIndex {

    /// Check if a label exists in the index
    /// - Parameter label: The label to check
    /// - Returns: True if the label exists and is not marked as deleted
    public func contains(label: UInt64) -> Bool {
        state.withLock { state in
            hnsw_contains_label(state.index, label)
        }
    }

    /// Get the vector associated with a label
    /// - Parameter label: The label to look up
    /// - Returns: The vector if found, nil otherwise
    public func getVector(label: UInt64) -> [Scalar]? {
        state.withLock { state in
            var output = [Scalar](repeating: .zero, count: dimensions)
            let success = output.withUnsafeMutableBufferPointer { buffer in
                Scalar.getVector(state.index, label: label, output: buffer.baseAddress!, dimension: dimensions)
            }
            return success ? output : nil
        }
    }

    /// Get all labels currently in the index (excluding deleted elements)
    public var allLabels: [UInt64] {
        state.withLock { state in
            let totalCount = hnsw_get_all_labels(state.index, nil, 0)
            guard totalCount > 0 else { return [] }

            var labels = [UInt64](repeating: 0, count: totalCount)
            let actualCount = labels.withUnsafeMutableBufferPointer { buffer in
                hnsw_get_all_labels(state.index, buffer.baseAddress!, totalCount)
            }
            if actualCount < labels.count {
                labels.removeSubrange(actualCount..<labels.count)
            }
            return labels
        }
    }
}

// MARK: - Data Serialization

extension HNSWIndex {

    /// Serialize the index to Data
    /// - Returns: Serialized index data
    public func serialize() throws -> Data {
        try state.withLock { state in
            let size = hnsw_get_serialized_size(state.index)
            guard size > 0 else {
                throw HNSWError.serializationFailed("Failed to get serialized size")
            }

            var buffer = Data(count: size)
            let success = buffer.withUnsafeMutableBytes { ptr in
                hnsw_serialize_to_buffer(state.index, ptr.baseAddress!, size)
            }

            guard success else {
                throw HNSWError.serializationFailed("Failed to serialize index to buffer")
            }

            return buffer
        }
    }

    /// Load an index from Data
    /// - Parameters:
    ///   - data: Serialized index data
    ///   - dimensions: Vector dimensions
    ///   - metric: Distance metric
    ///   - maxElements: Maximum elements (0 to use saved value)
    /// - Returns: Loaded HNSW index
    public static func load(
        from data: Data,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        guard let space = metric.createSpace(dimensions: dimensions, scalar: Scalar.self) else {
            throw HNSWError.loadFailed("Failed to create distance space")
        }

        let loadedIndex: HNSWIndexHandle? = data.withUnsafeBytes { ptr in
            hnsw_load_from_buffer(ptr.baseAddress!, data.count, space, maxElements)
        }

        guard let loadedIndex else {
            hnsw_destroy_space(space)
            throw HNSWError.loadFailed("Failed to load index from data")
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

#else

/// Swift backend HNSW graph index with the same public API as the optional hnswlib-backed index.
public final class HNSWIndex<Scalar: HNSWScalar>: Sendable {

    private struct Entry: Sendable {
        var internalID: Int
        var offset: Int
        var deleted: Bool
        var level: Int
    }

    private struct State: Sendable {
        var maximumElementCount: Int
        var efSearch: Int
        var entries: [UInt64: Entry]
        var labelOrder: [UInt64]
        var liveCount: Int
        var vectorStorage: [Scalar]
        var connections: [[[Int]]]
        var entryPoint: Int?
        var maxLevel: Int
        var levelGenerator: HNSWLevelGenerator
        var visited: [UInt32]
        var visitedTag: UInt32
        var queryScratch: [Scalar]
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
        guard maxElements > 0 else {
            throw HNSWError.initializationFailed("Maximum element count must be positive")
        }
        self.dimensions = dimensions
        self.metric = metric
        self.configuration = configuration
        var vectorStorage: [Scalar] = []
        if maxElements <= Int.max / dimensions {
            vectorStorage.reserveCapacity(maxElements * dimensions)
        }
        self.state = Mutex(State(
            maximumElementCount: maxElements,
            efSearch: configuration.efSearch,
            entries: [:],
            labelOrder: [],
            liveCount: 0,
            vectorStorage: vectorStorage,
            connections: [],
            entryPoint: nil,
            maxLevel: -1,
            levelGenerator: HNSWLevelGenerator(seed: configuration.randomSeed),
            visited: [],
            visitedTag: 0,
            queryScratch: [Scalar](repeating: .zero, count: dimensions)
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

    public func setEfSearch(_ ef: Int) {
        state.withLock {
            $0.efSearch = max(1, ef)
        }
    }

    public func resize(to newCapacity: Int) throws {
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
    /// The Swift backend stores vectors in an internal contiguous arena. This call performs
    /// the required ownership copy exactly once and avoids temporary slice arrays.
    public func add(_ vector: UnsafeBufferPointer<Scalar>, label: UInt64) throws {
        try validateDimensions(vector.count)

        try state.withLock {
            try upsertVector(vector, label: label, state: &$0)
        }
    }

    public func search(_ query: [Scalar], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Search with a borrowed query vector without materializing an intermediate array.
    public func search(_ query: UnsafeBufferPointer<Scalar>, k: Int) throws -> [SearchResult] {
        try validateDimensions(query.count)
        guard k > 0 else { return [] }

        return state.withLock { state in
            if metric.requiresNormalization {
                var normalizedQuery = [Scalar](repeating: .zero, count: dimensions)
                normalizedQuery.withUnsafeMutableBufferPointer { scratch in
                    normalizeVector(query, into: scratch)
                }
                return normalizedQuery.withUnsafeBufferPointer { normalized in
                    searchNormalized(normalized, k: k, state: &state)
                }
            }
            return searchNormalized(query, k: k, state: &state)
        }
    }

    public func markDeleted(label: UInt64) throws {
        try state.withLock {
            guard var entry = $0.entries[label], !entry.deleted else {
                throw HNSWError.deleteFailed("Failed to delete element with label \(label)")
            }
            entry.deleted = true
            $0.entries[label] = entry
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
        guard numVectors > 0 else { return 0 }
        try validateDimensions(vectors.count, expectedTotal: numVectors * dimensions)

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

        let start = startingLabel ?? UInt64(count)
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

    /// Search borrowed contiguous query vectors.
    public func searchBatch(
        _ queries: UnsafeBufferPointer<Scalar>,
        numQueries: Int,
        k: Int
    ) throws -> [[SearchResult]] {
        try validateDimensions(queries.count, expectedTotal: numQueries * dimensions)
        guard numQueries > 0 else { return [] }
        guard k > 0 else { return Array(repeating: [], count: numQueries) }

        return state.withLock { state in
            var results: [[SearchResult]] = []
            results.reserveCapacity(numQueries)
            for index in 0..<numQueries {
                let offset = index * dimensions
                let query = UnsafeBufferPointer(start: queries.baseAddress! + offset, count: dimensions)
                if metric.requiresNormalization {
                    var normalizedQuery = [Scalar](repeating: .zero, count: dimensions)
                    normalizedQuery.withUnsafeMutableBufferPointer { scratch in
                        normalizeVector(query, into: scratch)
                    }
                    let searchResults = normalizedQuery.withUnsafeBufferPointer { normalized in
                        searchNormalized(normalized, k: k, state: &state)
                    }
                    results.append(searchResults)
                } else {
                    results.append(searchNormalized(query, k: k, state: &state))
                }
            }
            return results
        }
    }

    public func searchBatch(_ queries: [[Scalar]], k: Int) throws -> [[SearchResult]] {
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

extension HNSWIndex {

    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        do {
#if os(WASI)
            try serialize().write(to: URL(fileURLWithPath: path))
#else
            try serialize().write(to: URL(fileURLWithPath: path), options: .atomic)
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
            return try load(from: data, dimensions: dimensions, metric: metric, maxElements: maxElements)
        } catch let error as HNSWError {
            throw error
        } catch {
            throw HNSWError.loadFailed("Failed to load index from \(path)")
        }
    }
}

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
            return $0.vectorStorage.withUnsafeBufferPointer { storage in
                let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                var output = [Scalar](repeating: .zero, count: dimensions)
                output.withUnsafeMutableBufferPointer { destination in
                    VectorOperations.copy(vector, into: destination)
                }
                return output
            }
        }
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

    public func serialize() throws -> Data {
        state.withLock {
            var writer = FlatIndexWriter()
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
            writer.writeUInt32($0.entryPoint.map(UInt32.init) ?? UInt32.max)
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
                $0.vectorStorage.withUnsafeBufferPointer { storage in
                    let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                    for value in vector {
                        writer.writeFloat(value.hnswFloatValue)
                    }
                }
                let levels = internalID < $0.connections.count ? $0.connections[internalID] : []
                writer.writeUInt32(UInt32(levels.count))
                for neighbors in levels {
                    writer.writeUInt32(UInt32(neighbors.count))
                    for neighbor in neighbors {
                        writer.writeUInt32(UInt32(neighbor))
                    }
                }
            }
            return writer.data
        }
    }

    public static func load(
        from data: Data,
        dimensions: Int,
        metric: DistanceMetric = .l2,
        maxElements: Int = 0
    ) throws -> HNSWIndex {
        do {
            var reader = FlatIndexReader(data: data)
            let magic = try reader.readMagic()
            let version = try reader.readUInt32()
            switch magic {
            case FlatIndexReader.graphMagic:
                return try loadGraph(
                    reader: &reader,
                    version: version,
                    dimensions: dimensions,
                    metric: metric,
                    maxElements: maxElements
                )
            case FlatIndexReader.flatMagic:
                return try loadFlat(
                    reader: &reader,
                    version: version,
                    dimensions: dimensions,
                    metric: metric,
                    maxElements: maxElements
                )
            default:
                throw HNSWError.loadFailed("Invalid Swift HNSW index magic")
            }
        } catch let error as HNSWError {
            throw error
        } catch {
            throw HNSWError.loadFailed("Failed to load index from data")
        }
    }

    private static func loadGraph(
        reader: inout FlatIndexReader,
        version: UInt32,
        dimensions: Int,
        metric: DistanceMetric,
        maxElements: Int
    ) throws -> HNSWIndex {
        guard version == 2 else {
            throw HNSWError.loadFailed("Unsupported graph index version")
        }
        let storedDimensions = Int(try reader.readUInt32())
        guard storedDimensions == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: storedDimensions)
        }
        let storedCapacity = Int(try reader.readUInt32())
        let storedMetric = try reader.readString()
        guard storedMetric == metric.rawValue else {
            throw HNSWError.loadFailed("Stored metric \(storedMetric) does not match \(metric.rawValue)")
        }

        let storedM = Int(try reader.readUInt32())
        let storedEfConstruction = Int(try reader.readUInt32())
        let storedEfSearch = Int(try reader.readUInt32())
        guard storedM > 0 else {
            throw HNSWError.loadFailed("Graph index contains invalid M")
        }
        guard storedEfConstruction > 0, storedEfSearch > 0 else {
            throw HNSWError.loadFailed("Graph index contains invalid ef value")
        }
        let storedSeed = Int(Int32(bitPattern: try reader.readUInt32()))
        let storedAllowReplaceDeleted = try reader.readBool()
        let storedMaxLevel = Int(Int32(bitPattern: try reader.readUInt32()))
        let storedEntryPoint = try reader.readUInt32()
        let generatorState = try reader.readUInt64()
        let labelCount = Int(try reader.readUInt32())

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
        var loadedVectorStorage: [Scalar] = []
        var loadedConnections: [[[Int]]] = []
        loadedEntries.reserveCapacity(labelCount)
        loadedLabelOrder.reserveCapacity(labelCount)
        loadedVectorStorage.reserveCapacity(labelCount * dimensions)
        loadedConnections.reserveCapacity(labelCount)

        for internalID in 0..<labelCount {
            let label = try reader.readUInt64()
            guard loadedEntries[label] == nil else {
                throw HNSWError.loadFailed("Graph index contains duplicate labels")
            }
            let deleted = try reader.readBool()
            let level = Int(try reader.readUInt32())
            guard level >= 0, level < Self.maximumSerializedLevelCount else {
                throw HNSWError.loadFailed("Graph index contains invalid level")
            }
            let offset = loadedVectorStorage.count
            for _ in 0..<dimensions {
                loadedVectorStorage.append(Scalar(try reader.readFloat()))
            }

            let levelCount = Int(try reader.readUInt32())
            guard levelCount == level + 1 else {
                throw HNSWError.loadFailed("Graph index level count does not match node level")
            }
            var nodeConnections: [[Int]] = []
            nodeConnections.reserveCapacity(levelCount)
            for levelIndex in 0..<levelCount {
                let neighborCount = Int(try reader.readUInt32())
                guard neighborCount <= max(0, labelCount - 1) else {
                    throw HNSWError.loadFailed("Graph index contains too many neighbors")
                }
                guard neighborCount <= Self.maximumConnections(at: levelIndex, m: storedM) else {
                    throw HNSWError.loadFailed("Graph index exceeds configured connection limit")
                }
                var neighbors: [Int] = []
                neighbors.reserveCapacity(neighborCount)
                for _ in 0..<neighborCount {
                    let neighbor = Int(try reader.readUInt32())
                    guard neighbor >= 0, neighbor < labelCount else {
                        throw HNSWError.loadFailed("Graph index contains invalid neighbor id")
                    }
                    neighbors.append(neighbor)
                }
                nodeConnections.append(neighbors)
            }

            loadedEntries[label] = Entry(
                internalID: internalID,
                offset: offset,
                deleted: deleted,
                level: level
            )
            loadedLabelOrder.append(label)
            loadedConnections.append(nodeConnections)
        }
        try reader.ensureFullyRead()

        let entryPoint: Int?
        if storedEntryPoint == UInt32.max {
            entryPoint = nil
        } else {
            let value = Int(storedEntryPoint)
            guard value >= 0, value < labelCount else {
                throw HNSWError.loadFailed("Graph index contains invalid entry point")
            }
            entryPoint = value
        }
        try validateLoadedGraph(
            entries: loadedEntries,
            labelOrder: loadedLabelOrder,
            connections: loadedConnections,
            entryPoint: entryPoint,
            maxLevel: storedMaxLevel,
            m: storedM
        )

        index.state.withLock {
            $0.maximumElementCount = max(1, capacity)
            $0.efSearch = storedEfSearch
            $0.entries = loadedEntries
            $0.labelOrder = loadedLabelOrder
            $0.liveCount = loadedEntries.values.reduce(0) { count, entry in
                entry.deleted ? count : count + 1
            }
            $0.vectorStorage = loadedVectorStorage
            $0.connections = loadedConnections
            $0.entryPoint = entryPoint
            $0.maxLevel = storedMaxLevel
            $0.levelGenerator = HNSWLevelGenerator(state: generatorState)
            $0.visited = [UInt32](repeating: 0, count: labelCount)
            $0.visitedTag = 0
            $0.queryScratch = [Scalar](repeating: .zero, count: dimensions)
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

    private static func loadFlat(
        reader: inout FlatIndexReader,
        version: UInt32,
        dimensions: Int,
        metric: DistanceMetric,
        maxElements: Int
    ) throws -> HNSWIndex {
        guard version == 1 else {
            throw HNSWError.loadFailed("Unsupported flat index version")
        }
        let storedDimensions = Int(try reader.readUInt32())
        guard storedDimensions == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: storedDimensions)
        }
        let storedCapacity = Int(try reader.readUInt32())
        let storedMetric = try reader.readString()
        guard storedMetric == metric.rawValue else {
            throw HNSWError.loadFailed("Stored metric \(storedMetric) does not match \(metric.rawValue)")
        }

        let capacity = max(maxElements, storedCapacity)
        let index = try HNSWIndex(
            dimensions: dimensions,
            maxElements: max(1, capacity),
            metric: metric,
            configuration: .balanced
        )
        let labelCount = Int(try reader.readUInt32())
        var loadedEntries: [UInt64: Entry] = [:]
        var loadedLabelOrder: [UInt64] = []
        var loadedVectorStorage: [Scalar] = []
        var loadedConnections: [[[Int]]] = []
        loadedEntries.reserveCapacity(labelCount)
        loadedLabelOrder.reserveCapacity(labelCount)
        loadedVectorStorage.reserveCapacity(labelCount * dimensions)
        loadedConnections.reserveCapacity(labelCount)

        var generator = HNSWLevelGenerator(seed: HNSWConfiguration.balanced.randomSeed)
        let multiplier = 1.0 / Foundation.log(Double(max(2, HNSWConfiguration.balanced.m)))
        for internalID in 0..<labelCount {
            let label = try reader.readUInt64()
            let deleted = try reader.readBool()
            let offset = loadedVectorStorage.count
            for _ in 0..<dimensions {
                loadedVectorStorage.append(Scalar(try reader.readFloat()))
            }
            let level = generator.randomLevel(multiplier: multiplier)
            loadedEntries[label] = Entry(
                internalID: internalID,
                offset: offset,
                deleted: deleted,
                level: level
            )
            loadedLabelOrder.append(label)
            loadedConnections.append(Array(repeating: [], count: level + 1))
        }
        try reader.ensureFullyRead()

        index.state.withLock {
            $0.entries = loadedEntries
            $0.labelOrder = loadedLabelOrder
            $0.liveCount = loadedEntries.values.reduce(0) { count, entry in
                entry.deleted ? count : count + 1
            }
            $0.vectorStorage = loadedVectorStorage
            $0.connections = loadedConnections
            $0.levelGenerator = generator
            $0.visited = [UInt32](repeating: 0, count: labelCount)
            $0.visitedTag = 0
            $0.queryScratch = [Scalar](repeating: .zero, count: dimensions)
            index.rebuildGraph(state: &$0)
        }
        return index
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
    private func ensureCapacity(_ buffer: inout [Scalar], count: Int) {
        guard buffer.count != count else { return }
        buffer = [Scalar](repeating: .zero, count: count)
    }

    private func normalizeVector(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            let outputFloats = UnsafeMutableBufferPointer<Float>(
                start: UnsafeMutableRawPointer(output.baseAddress!).assumingMemoryBound(to: Float.self),
                count: output.count
            )
            VectorOperations.normalize(inputFloats, into: outputFloats)
        } else {
            VectorOperations.normalize(input, into: output)
        }
    }

    private func upsertVector(
        _ vector: UnsafeBufferPointer<Scalar>,
        label: UInt64,
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
            shouldRebuildGraph = existing.internalID < state.connections.count &&
                state.connections[existing.internalID].contains(where: { !$0.isEmpty })
        } else {
            if state.entries.count < state.maximumElementCount {
                let internalID = state.labelOrder.count
                let offset = state.vectorStorage.count
                let level = state.levelGenerator.randomLevel(multiplier: levelMultiplier)
                entry = Entry(internalID: internalID, offset: offset, deleted: false, level: level)
                for _ in 0..<dimensions {
                    state.vectorStorage.append(.zero)
                }
                state.labelOrder.append(label)
                state.connections.append(Array(repeating: [], count: level + 1))
                state.visited.append(0)
                state.liveCount += 1
                shouldRebuildGraph = false
            } else if configuration.allowReplaceDeleted,
                      let reusable = reusableDeletedEntry(state: state) {
                let level = state.levelGenerator.randomLevel(multiplier: levelMultiplier)
                entry = Entry(
                    internalID: reusable.entry.internalID,
                    offset: reusable.entry.offset,
                    deleted: false,
                    level: level
                )
                state.entries.removeValue(forKey: reusable.label)
                state.labelOrder[reusable.entry.internalID] = label
                state.connections[reusable.entry.internalID] = Array(repeating: [], count: level + 1)
                state.liveCount += 1
                shouldRebuildGraph = true
            } else {
                throw HNSWError.capacityExceeded(
                    current: state.entries.count,
                    maximum: state.maximumElementCount
                )
            }
        }

        state.vectorStorage.withUnsafeMutableBufferPointer { storage in
            let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
            if metric.requiresNormalization {
                normalizeVector(vector, into: destination)
            } else {
                VectorOperations.copy(vector, into: destination)
            }
        }
        state.entries[label] = entry

        if shouldRebuildGraph {
            rebuildGraph(state: &state)
        } else {
            connectNewElement(entry.internalID, state: &state)
        }
    }

    private func searchNormalized(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        state: inout State
    ) -> [SearchResult] {
        guard let entryPoint = state.entryPoint else { return [] }
        var current = entryPoint
        var currentDistance = distance(from: query, to: current, state: state)

        if state.maxLevel > 0 {
            for level in stride(from: state.maxLevel, through: 1, by: -1) {
                let closest = greedySearchLayer(
                    query,
                    entryPoint: current,
                    entryDistance: currentDistance,
                    level: level,
                    state: state
                )
                current = closest.internalID
                currentDistance = closest.distance
            }
        }

        return searchLayer(
            query,
            entryPoint: current,
            ef: max(state.efSearch, k),
            level: 0,
            includeDeleted: false,
            state: &state
        )
        .prefix(k)
        .map {
            SearchResult(
                label: $0.label,
                distance: VectorOperations.publicDistance(fromComparisonDistance: $0.distance, metric: metric)
            )
        }
    }

    private var levelMultiplier: Double {
        1.0 / Foundation.log(Double(max(2, configuration.m)))
    }

    private func maxConnections(at level: Int) -> Int {
        level == 0 ? max(1, configuration.m * 2) : max(1, configuration.m)
    }

    private func newElementConnections(at level: Int) -> Int {
        max(1, configuration.m)
    }

    private func connectNewElement(_ internalID: Int, state: inout State) {
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
        var currentDistance = distanceBetween(internalID, current, state: state)
        let previousMaxLevel = state.maxLevel

        if entry.level < previousMaxLevel {
            for level in stride(from: previousMaxLevel, through: entry.level + 1, by: -1) {
                let closest = greedySearchNode(
                    internalID,
                    entryPoint: current,
                    entryDistance: currentDistance,
                    level: level,
                    state: state
                )
                current = closest.internalID
                currentDistance = closest.distance
            }
        }

        let topLevel = min(entry.level, previousMaxLevel)
        if topLevel >= 0 {
            for level in stride(from: topLevel, through: 0, by: -1) {
                let candidates = searchLayerForNode(
                    internalID,
                    entryPoint: current,
                    ef: max(configuration.efConstruction, configuration.m),
                    level: level,
                    includeDeleted: false,
                    state: &state
                )
                let selected = selectNeighborsForNode(
                    internalID,
                    candidates: candidates,
                    limit: newElementConnections(at: level),
                    state: state
                )
                setConnections(selected.map(\.internalID), for: internalID, at: level, state: &state)
                for neighbor in selected {
                    connectBidirectional(internalID, neighborID: neighbor.internalID, level: level, state: &state)
                }
                if let first = selected.first {
                    current = first.internalID
                    currentDistance = first.distance
                }
            }
        }

        if entry.level > previousMaxLevel {
            state.entryPoint = internalID
            state.maxLevel = entry.level
        }
    }

    private func rebuildGraph(state: inout State) {
        state.connections = state.labelOrder.enumerated().map { internalID, label in
            let level = state.entries[label]?.level ?? 0
            return Array(repeating: [], count: max(0, level) + 1)
        }
        state.entryPoint = nil
        state.maxLevel = -1

        for internalID in state.labelOrder.indices {
            guard entry(for: internalID, state: state) != nil else { continue }
            connectNewElement(internalID, state: &state)
        }
    }

    private func greedySearchNode(
        _ queryID: Int,
        entryPoint: Int,
        entryDistance: Float,
        level: Int,
        state: State
    ) -> HNSWNeighborCandidate {
        greedySearch(
            entryPoint: entryPoint,
            entryDistance: entryDistance,
            level: level,
            state: state
        ) { candidateID in
            distanceBetween(queryID, candidateID, state: state)
        }
    }

    private func greedySearchLayer(
        _ query: UnsafeBufferPointer<Scalar>,
        entryPoint: Int,
        entryDistance: Float,
        level: Int,
        state: State
    ) -> HNSWNeighborCandidate {
        greedySearch(
            entryPoint: entryPoint,
            entryDistance: entryDistance,
            level: level,
            state: state
        ) { candidateID in
            distance(from: query, to: candidateID, state: state)
        }
    }

    private func greedySearch(
        entryPoint: Int,
        entryDistance: Float,
        level: Int,
        state: State,
        distanceToCandidate: (Int) -> Float
    ) -> HNSWNeighborCandidate {
        var current = HNSWNeighborCandidate(
            internalID: entryPoint,
            label: label(for: entryPoint, state: state),
            distance: entryDistance
        )
        var changed = true
        while changed {
            changed = false
            for neighborID in connections(for: current.internalID, at: level, state: state) {
                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    label: label(for: neighborID, state: state),
                    distance: distanceToCandidate(neighborID)
                )
                if isCloserHNSWCandidate(candidate, than: current) {
                    current = candidate
                    changed = true
                }
            }
        }
        return current
    }

    private func searchLayerForNode(
        _ queryID: Int,
        entryPoint: Int,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        state: inout State
    ) -> [HNSWNeighborCandidate] {
        searchLayerForNodeWithVisitedArray(
            queryID,
            entryPoint: entryPoint,
            entryDistance: distanceBetween(queryID, entryPoint, state: state),
            ef: ef,
            level: level,
            includeDeleted: includeDeleted,
            state: &state
        )
    }

    private func searchLayerForNodeWithVisitedArray(
        _ queryID: Int,
        entryPoint: Int,
        entryDistance: Float,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        state: inout State
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(state: &state)

        var candidateQueue = BinaryHeap<HNSWNeighborCandidate>(
            hasHigherPriority: isCloserHNSWCandidate
        )
        var nearest = BinaryHeap<HNSWNeighborCandidate>(
            hasHigherPriority: isFartherHNSWCandidate
        )
        candidateQueue.reserveCapacity(effectiveEF)
        nearest.reserveCapacity(effectiveEF)

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            label: label(for: entryPoint, state: state),
            distance: entryDistance
        )
        markVisited(entryPoint, tag: tag, state: &state)
        candidateQueue.push(entry)
        if includeDeleted || !isDeleted(entryPoint, state: state) {
            nearest.push(entry)
        }

        while let current = candidateQueue.pop() {
            let lowerBound = nearest.peek?.distance ?? Float.greatestFiniteMagnitude
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            for neighborID in connections(for: current.internalID, at: level, state: state) {
                guard markVisited(neighborID, tag: tag, state: &state) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    label: label(for: neighborID, state: state),
                    distance: distanceBetween(queryID, neighborID, state: state)
                )
                let candidateLowerBound = nearest.peek?.distance ?? Float.greatestFiniteMagnitude
                if nearest.count < effectiveEF || candidate.distance < candidateLowerBound {
                    candidateQueue.push(candidate)
                    if includeDeleted || !isDeleted(neighborID, state: state) {
                        nearest.push(candidate)
                        if nearest.count > effectiveEF {
                            nearest.pop()
                        }
                    }
                }
            }
        }

        return nearest.unorderedElements().sorted(by: isCloserHNSWCandidate)
    }

    private func searchLayer(
        _ query: UnsafeBufferPointer<Scalar>,
        entryPoint: Int,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        state: inout State
    ) -> [HNSWNeighborCandidate] {
        searchLayerForQueryWithVisitedArray(
            query,
            entryPoint: entryPoint,
            entryDistance: distance(from: query, to: entryPoint, state: state),
            ef: ef,
            level: level,
            includeDeleted: includeDeleted,
            state: &state
        )
    }

    private func searchLayerForQueryWithVisitedArray(
        _ query: UnsafeBufferPointer<Scalar>,
        entryPoint: Int,
        entryDistance: Float,
        ef: Int,
        level: Int,
        includeDeleted: Bool,
        state: inout State
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(state: &state)

        var candidateQueue = BinaryHeap<HNSWNeighborCandidate>(
            hasHigherPriority: isCloserHNSWCandidate
        )
        var nearest = BinaryHeap<HNSWNeighborCandidate>(
            hasHigherPriority: isFartherHNSWCandidate
        )

        let entry = HNSWNeighborCandidate(
            internalID: entryPoint,
            label: label(for: entryPoint, state: state),
            distance: entryDistance
        )
        markVisited(entryPoint, tag: tag, state: &state)
        candidateQueue.push(entry)
        if includeDeleted || !isDeleted(entryPoint, state: state) {
            nearest.push(entry)
        }

        while let current = candidateQueue.pop() {
            let lowerBound = nearest.peek?.distance ?? Float.greatestFiniteMagnitude
            if current.distance > lowerBound, nearest.count >= effectiveEF {
                break
            }

            for neighborID in connections(for: current.internalID, at: level, state: state) {
                guard markVisited(neighborID, tag: tag, state: &state) else { continue }

                let candidate = HNSWNeighborCandidate(
                    internalID: neighborID,
                    label: label(for: neighborID, state: state),
                    distance: distance(from: query, to: neighborID, state: state)
                )
                let candidateLowerBound = nearest.peek?.distance ?? Float.greatestFiniteMagnitude
                if nearest.count < effectiveEF || candidate.distance < candidateLowerBound {
                    candidateQueue.push(candidate)
                    if includeDeleted || !isDeleted(neighborID, state: state) {
                        nearest.push(candidate)
                        if nearest.count > effectiveEF {
                            nearest.pop()
                        }
                    }
                }
            }
        }

        return nearest.unorderedElements().sorted(by: isCloserHNSWCandidate)
    }

    private func selectNeighborsForNode(
        _ queryID: Int,
        candidates: [HNSWNeighborCandidate],
        limit: Int,
        state: State
    ) -> [HNSWNeighborCandidate] {
        selectNeighbors(candidates: candidates, limit: limit, state: state)
    }

    private func selectNeighbors(
        candidates: [HNSWNeighborCandidate],
        limit: Int,
        state: State
    ) -> [HNSWNeighborCandidate] {
        guard limit > 0 else { return [] }

        var selected: [HNSWNeighborCandidate] = []
        selected.reserveCapacity(min(limit, candidates.count))
        for candidate in candidates.sorted(by: isCloserHNSWCandidate) {
            guard selected.count < limit else { break }
            var isDiverse = true
            for existing in selected {
                let neighborDistance = distanceBetween(candidate.internalID, existing.internalID, state: state)
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

    private func hasLiveEntry(excluding excludedID: Int, state: State) -> Bool {
        guard let excluded = entry(for: excludedID, state: state) else {
            return state.liveCount > 0
        }
        return state.liveCount > (excluded.deleted ? 0 : 1)
    }

    private func reusableDeletedEntry(state: State) -> (label: UInt64, entry: Entry)? {
        for label in state.labelOrder {
            guard let entry = state.entries[label], entry.deleted else { continue }
            return (label, entry)
        }
        return nil
    }

    private func connectBidirectional(
        _ internalID: Int,
        neighborID: Int,
        level: Int,
        state: inout State
    ) {
        guard internalID != neighborID else { return }
        ensureConnectionStorage(for: neighborID, through: level, state: &state)
        if !state.connections[neighborID][level].contains(internalID) {
            state.connections[neighborID][level].append(internalID)
        }

        let maxConnections = maxConnections(at: level)
        guard state.connections[neighborID][level].count > maxConnections else { return }

        let candidates = state.connections[neighborID][level].map { candidateID in
            HNSWNeighborCandidate(
                internalID: candidateID,
                label: label(for: candidateID, state: state),
                distance: distanceBetween(neighborID, candidateID, state: state)
            )
        }
        let selected = selectNeighborsForNode(
            neighborID,
            candidates: candidates,
            limit: maxConnections,
            state: state
        )
        state.connections[neighborID][level] = selected.map(\.internalID)
    }

    private func setConnections(
        _ neighbors: [Int],
        for internalID: Int,
        at level: Int,
        state: inout State
    ) {
        ensureConnectionStorage(for: internalID, through: level, state: &state)
        state.connections[internalID][level] = neighbors.filter { $0 != internalID }
    }

    private func ensureConnectionStorage(for internalID: Int, through level: Int, state: inout State) {
        while state.connections.count <= internalID {
            state.connections.append([])
        }
        while state.connections[internalID].count <= level {
            state.connections[internalID].append([])
        }
    }

    private func connections(for internalID: Int, at level: Int, state: State) -> [Int] {
        guard internalID >= 0,
              internalID < state.connections.count,
              level >= 0,
              level < state.connections[internalID].count else {
            return []
        }
        return state.connections[internalID][level]
    }

    private func nextVisitedTag(state: inout State) -> UInt32 {
        if state.visitedTag == UInt32.max {
            state.visited = [UInt32](repeating: 0, count: state.labelOrder.count)
            state.visitedTag = 0
        }
        state.visitedTag += 1
        return state.visitedTag
    }

    @discardableResult
    private func markVisited(_ internalID: Int, tag: UInt32, state: inout State) -> Bool {
        guard internalID >= 0, internalID < state.visited.count else { return false }
        guard state.visited[internalID] != tag else { return false }
        state.visited[internalID] = tag
        return true
    }

    private func distanceBetween(_ lhsID: Int, _ rhsID: Int, state: State) -> Float {
        state.vectorStorage.withUnsafeBufferPointer { storage in
            let lhs = vector(for: lhsID, storage: storage)
            let rhs = vector(for: rhsID, storage: storage)
            return VectorOperations.comparisonDistance(from: lhs, to: rhs, metric: metric)
        }
    }

    private func distance(
        from query: UnsafeBufferPointer<Scalar>,
        to internalID: Int,
        state: State
    ) -> Float {
        state.vectorStorage.withUnsafeBufferPointer { storage in
            let candidate = vector(for: internalID, storage: storage)
            return VectorOperations.comparisonDistance(from: query, to: candidate, metric: metric)
        }
    }

    private func vector(
        for internalID: Int,
        storage: UnsafeBufferPointer<Scalar>
    ) -> UnsafeBufferPointer<Scalar> {
        UnsafeBufferPointer(start: storage.baseAddress! + internalID * dimensions, count: dimensions)
    }

    private func entry(for internalID: Int, state: State) -> Entry? {
        guard internalID >= 0, internalID < state.labelOrder.count else { return nil }
        return state.entries[state.labelOrder[internalID]]
    }

    private func label(for internalID: Int, state: State) -> UInt64 {
        state.labelOrder[internalID]
    }

    private func isDeleted(_ internalID: Int, state: State) -> Bool {
        entry(for: internalID, state: state)?.deleted ?? true
    }
}

private struct FlatIndexWriter {
    var data = Data()

    mutating func writeBytes(_ bytes: [UInt8]) {
        data.append(contentsOf: bytes)
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
        data.append(contentsOf: value.utf8)
    }
}

private struct FlatIndexReader {
    static let flatMagic: UInt64 = 0x414C_4657_534E_4853
    static let graphMagic: UInt64 = 0x4652_4757_534E_4853

    let data: Data
    var offset = 0

    mutating func readMagic() throws -> UInt64 {
        let magic = try readUInt64()
        return magic
    }

    mutating func readByte() throws -> UInt8 {
        guard offset < data.count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += 1 }
        return data.withUnsafeBytes { buffer in
            buffer.loadUnaligned(fromByteOffset: offset, as: UInt8.self)
        }
    }

    mutating func skipBytes(count: Int) throws {
        guard count >= 0, offset + count <= data.count else {
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
        guard offset + 4 <= data.count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += 4 }
        return data.withUnsafeBytes { buffer in
            UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }

    mutating func readUInt64() throws -> UInt64 {
        guard offset + 8 <= data.count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += 8 }
        return data.withUnsafeBytes { buffer in
            UInt64(littleEndian: buffer.loadUnaligned(fromByteOffset: offset, as: UInt64.self))
        }
    }

    mutating func readFloat() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readString() throws -> String {
        let count = Int(try readUInt32())
        guard count >= 0, offset + count <= data.count else {
            throw HNSWError.loadFailed("Flat index data is truncated")
        }
        defer { offset += count }
        return data.withUnsafeBytes { buffer in
            let start = buffer.baseAddress!.advanced(by: offset).assumingMemoryBound(to: UInt8.self)
            let bytes = UnsafeBufferPointer(start: start, count: count)
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    func ensureFullyRead() throws {
        guard offset == data.count else {
            throw HNSWError.loadFailed("Flat index data has trailing bytes")
        }
    }
}

#endif

// MARK: - Type Aliases for Convenience

/// HNSW Index using Float32 vectors (standard precision)
public typealias HNSWIndexF32 = HNSWIndex<Float>

/// HNSW Index using Float16 vectors (half precision, 50% memory savings)
public typealias HNSWIndexF16 = HNSWIndex<Float16>
