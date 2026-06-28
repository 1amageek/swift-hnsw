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

/// Swift backend exact vector index with the same public API as the optional hnswlib-backed index.
public final class HNSWIndex<Scalar: HNSWScalar>: Sendable {

    private struct Entry: Sendable {
        var offset: Int
        var deleted: Bool
    }

    private struct State: Sendable {
        var maximumElementCount: Int
        var efSearch: Int
        var entries: [UInt64: Entry]
        var labelOrder: [UInt64]
        var vectorStorage: [Scalar]
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
            vectorStorage: vectorStorage,
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
                ensureCapacity(&state.queryScratch, count: dimensions)
                state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                    normalizeVector(query, into: scratch)
                }
                return state.queryScratch.withUnsafeBufferPointer { normalized in
                    searchNormalized(normalized, k: k, state: state)
                }
            }
            return searchNormalized(query, k: k, state: state)
        }
    }

    public func markDeleted(label: UInt64) throws {
        try state.withLock {
            guard var entry = $0.entries[label], !entry.deleted else {
                throw HNSWError.deleteFailed("Failed to delete element with label \(label)")
            }
            entry.deleted = true
            $0.entries[label] = entry
        }
    }

    public func unmarkDeleted(label: UInt64) throws {
        try state.withLock {
            guard var entry = $0.entries[label], entry.deleted else {
                throw HNSWError.deleteFailed("Failed to undelete element with label \(label)")
            }
            entry.deleted = false
            $0.entries[label] = entry
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
                    ensureCapacity(&state.queryScratch, count: dimensions)
                    state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                        normalizeVector(query, into: scratch)
                    }
                    let searchResults = state.queryScratch.withUnsafeBufferPointer { normalized in
                        searchNormalized(normalized, k: k, state: state)
                    }
                    results.append(searchResults)
                } else {
                    results.append(searchNormalized(query, k: k, state: state))
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
            writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x46, 0x4C, 0x41])
            writer.writeUInt32(1)
            writer.writeUInt32(UInt32(dimensions))
            writer.writeUInt32(UInt32($0.maximumElementCount))
            writer.writeString(metric.rawValue)
            writer.writeUInt32(UInt32($0.labelOrder.count))
            for label in $0.labelOrder {
                guard let entry = $0.entries[label] else {
                    continue
                }
                writer.writeUInt64(label)
                writer.writeBool(entry.deleted)
                $0.vectorStorage.withUnsafeBufferPointer { storage in
                    let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                    for value in vector {
                        writer.writeFloat(value.hnswFloatValue)
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
            try reader.readMagic()
            let version = try reader.readUInt32()
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
            loadedEntries.reserveCapacity(labelCount)
            loadedLabelOrder.reserveCapacity(labelCount)
            loadedVectorStorage.reserveCapacity(labelCount * dimensions)
            for _ in 0..<labelCount {
                let label = try reader.readUInt64()
                let deleted = try reader.readBool()
                let offset = loadedVectorStorage.count
                for _ in 0..<dimensions {
                    loadedVectorStorage.append(Scalar(try reader.readFloat()))
                }
                loadedEntries[label] = Entry(offset: offset, deleted: deleted)
                loadedLabelOrder.append(label)
            }
            try reader.ensureFullyRead()
            index.state.withLock {
                $0.entries = loadedEntries
                $0.labelOrder = loadedLabelOrder
                $0.vectorStorage = loadedVectorStorage
                $0.queryScratch = [Scalar](repeating: .zero, count: dimensions)
            }
            return index
        } catch let error as HNSWError {
            throw error
        } catch {
            throw HNSWError.loadFailed("Failed to load index from data")
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
        let offset: Int
        if let existing = state.entries[label] {
            offset = existing.offset
        } else {
            guard state.entries.count < state.maximumElementCount else {
                throw HNSWError.capacityExceeded(
                    current: state.entries.count,
                    maximum: state.maximumElementCount
                )
            }
            offset = state.vectorStorage.count
            for _ in 0..<dimensions {
                state.vectorStorage.append(.zero)
            }
            state.labelOrder.append(label)
        }

        state.vectorStorage.withUnsafeMutableBufferPointer { storage in
            let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + offset, count: dimensions)
            if metric.requiresNormalization {
                normalizeVector(vector, into: destination)
            } else {
                VectorOperations.copy(vector, into: destination)
            }
        }
        state.entries[label] = Entry(offset: offset, deleted: false)
    }

    private func searchNormalized(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        state: State
    ) -> [SearchResult] {
        var results: [SearchResult] = []
        results.reserveCapacity(min(k, state.entries.count))

        state.vectorStorage.withUnsafeBufferPointer { storage in
            for label in state.labelOrder {
                guard let entry = state.entries[label], !entry.deleted else {
                    continue
                }
                let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                insertTopKSearchResult(
                    SearchResult(
                        label: label,
                        distance: VectorOperations.distance(from: query, to: vector, metric: metric)
                    ),
                    into: &results,
                    limit: k
                )
            }
        }
        return results
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
    let data: Data
    var offset = 0

    mutating func readMagic() throws {
        let magic = try readUInt64()
        guard magic == 0x414C_4657_534E_4853 else {
            throw HNSWError.loadFailed("Invalid flat index magic")
        }
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
