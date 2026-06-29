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
        var labelScratch: [UInt64]
        var distanceScratch: [Float]

        init(space: HNSWSpaceHandle, index: HNSWIndexHandle) {
            self.spaceAddress = UInt(bitPattern: space)
            self.indexAddress = UInt(bitPattern: index)
            self.vectorScratch = []
            self.batchScratch = []
            self.labelScratch = []
            self.distanceScratch = []
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

    /// Search into a caller-provided output buffer.
    ///
    /// Returns the number of results written. `results` must have room for at least `k` values.
    @discardableResult
    public func search(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>
    ) throws -> Int {
        try validateDimensions(query.count)
        guard k > 0 else { return 0 }
        guard results.count >= k else {
            throw HNSWError.invalidArgument("Result buffer must contain at least k elements")
        }

        return state.withLock { state in
            if metric.requiresNormalization {
                let index = state.index
                ensureCapacity(&state.vectorScratch, count: dimensions)
                return state.vectorScratch.withUnsafeMutableBufferPointer { scratch in
                    normalizeVector(query, into: scratch)
                    let normalized = UnsafeBufferPointer(start: scratch.baseAddress, count: scratch.count)
                    let searchResults = searchNormalized(normalized, k: k, index: index)
                    let resultCount = min(searchResults.count, results.count)
                    for index in 0..<resultCount {
                        results[index] = searchResults[index]
                    }
                    return resultCount
                }
            }
            return searchNormalized(query, k: k, into: results, state: &state, index: state.index)
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

    @inline(__always)
    private func ensureLabelCapacity(_ buffer: inout [UInt64], count: Int) {
        guard buffer.count != count else { return }
        buffer = [UInt64](repeating: 0, count: count)
    }

    @inline(__always)
    private func ensureDistanceCapacity(_ buffer: inout [Float], count: Int) {
        guard buffer.count != count else { return }
        buffer = [Float](repeating: 0, count: count)
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

    private func searchNormalized(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>,
        state: inout CxxState,
        index: HNSWIndexHandle
    ) -> Int {
        ensureLabelCapacity(&state.labelScratch, count: k)
        ensureDistanceCapacity(&state.distanceScratch, count: k)

        let resultCount = state.labelScratch.withUnsafeMutableBufferPointer { labelsBuffer in
            state.distanceScratch.withUnsafeMutableBufferPointer { distancesBuffer in
                Scalar.searchKnn(
                    index,
                    query: query.baseAddress!,
                    k: Int32(k),
                    labels: labelsBuffer.baseAddress!,
                    distances: distancesBuffer.baseAddress!
                )
            }
        }

        let count = min(Int(resultCount), results.count)
        for index in 0..<count {
            results[index] = SearchResult(
                label: state.labelScratch[index],
                distance: state.distanceScratch[index]
            )
        }
        return count
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

    /// Access a vector for the duration of the closure.
    ///
    /// The C++ backend does not expose stable borrowed vector storage, so this method uses
    /// the bridge copy-out API before invoking the closure.
    public func withVector<R: Sendable>(
        label: UInt64,
        _ body: @Sendable (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R? {
        guard let vector = getVector(label: label) else {
            return nil
        }
        return try vector.withUnsafeBufferPointer(body)
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
        var queryScratch: [Float]
        var halfQueryScratch: [Float16]
        var searchScratch: HNSWSearchScratch
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
        guard maxElements <= Int(UInt32.max) else {
            throw HNSWError.initializationFailed("Maximum element count exceeds UInt32 internal id capacity")
        }
        self.dimensions = dimensions
        self.metric = metric
        self.configuration = configuration
        let usesHalfStorage = Scalar.self == Float16.self
        var comparisonStorage: [Float] = []
        var halfComparisonStorage: [Float16] = []
        if maxElements <= Int.max / dimensions {
            if usesHalfStorage {
                halfComparisonStorage.reserveCapacity(maxElements * dimensions)
            } else {
                comparisonStorage.reserveCapacity(maxElements * dimensions)
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
            queryScratch: usesHalfStorage ? [] : [Float](repeating: 0, count: dimensions),
            halfQueryScratch: usesHalfStorage ? [Float16](repeating: 0, count: dimensions) : [],
            searchScratch: HNSWSearchScratch()
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

                ensureCapacity(&state.halfQueryScratch, count: dimensions)
                state.halfQueryScratch.withUnsafeMutableBufferPointer { scratch in
                    storeNormalizedHalfVector(query, into: scratch)
                }
                return state.halfQueryScratch.withUnsafeBufferPointer { preparedQuery in
                    searchNormalizedHalf(preparedQuery, k: k, state: &state)
                }
            }

            ensureCapacity(&state.queryScratch, count: dimensions)
            state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                if metric.requiresNormalization {
                    storeNormalizedVector(query, into: scratch)
                } else {
                    storeVector(query, into: scratch)
                }
            }
            return state.queryScratch.withUnsafeBufferPointer { preparedQuery in
                searchNormalized(preparedQuery, k: k, state: &state)
            }
        }
    }

    /// Search into a caller-provided output buffer.
    ///
    /// Returns the number of results written. `results` must have room for at least `k` values.
    @discardableResult
    public func search(
        _ query: UnsafeBufferPointer<Scalar>,
        k: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>
    ) throws -> Int {
        try validateDimensions(query.count)
        guard k > 0 else { return 0 }
        guard results.count >= k else {
            throw HNSWError.invalidArgument("Result buffer must contain at least k elements")
        }

        return state.withLock { state in
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

                ensureCapacity(&state.halfQueryScratch, count: dimensions)
                state.halfQueryScratch.withUnsafeMutableBufferPointer { scratch in
                    storeNormalizedHalfVector(query, into: scratch)
                }
                return state.halfQueryScratch.withUnsafeBufferPointer { preparedQuery in
                    searchNormalizedHalf(preparedQuery, k: k, into: results, state: &state)
                }
            }

            ensureCapacity(&state.queryScratch, count: dimensions)
            state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                if metric.requiresNormalization {
                    storeNormalizedVector(query, into: scratch)
                } else {
                    storeVector(query, into: scratch)
                }
            }
            return state.queryScratch.withUnsafeBufferPointer { preparedQuery in
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
                        ensureCapacity(&state.halfQueryScratch, count: dimensions)
                        state.halfQueryScratch.withUnsafeMutableBufferPointer { scratch in
                            storeNormalizedHalfVector(query, into: scratch)
                        }
                        let searchResults = state.halfQueryScratch.withUnsafeBufferPointer { preparedQuery in
                            searchNormalizedHalf(preparedQuery, k: k, state: &state)
                        }
                        results.append(searchResults)
                    }
                } else {
                    ensureCapacity(&state.queryScratch, count: dimensions)
                    state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                        if metric.requiresNormalization {
                            storeNormalizedVector(query, into: scratch)
                        } else {
                            storeVector(query, into: scratch)
                        }
                    }
                    let searchResults = state.queryScratch.withUnsafeBufferPointer { preparedQuery in
                        searchNormalized(preparedQuery, k: k, state: &state)
                    }
                    results.append(searchResults)
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
    /// the Swift backend stores comparison values as Float.
    public func withVector<R: Sendable>(
        label: UInt64,
        _ body: @Sendable (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R? {
        try state.withLock { state in
            guard let entry = state.entries[label], !entry.deleted else {
                return nil
            }
            let start = entry.offset

            if Scalar.self == Float.self {
                return try state.comparisonStorage.withUnsafeBufferPointer { storage in
                    let pointer = UnsafeRawPointer(storage.baseAddress! + start)
                        .assumingMemoryBound(to: Scalar.self)
                    return try body(UnsafeBufferPointer(start: pointer, count: dimensions))
                }
            }

            if Scalar.self == Float16.self {
                return try state.halfComparisonStorage.withUnsafeBufferPointer { storage in
                    let pointer = UnsafeRawPointer(storage.baseAddress! + start)
                        .assumingMemoryBound(to: Scalar.self)
                    return try body(UnsafeBufferPointer(start: pointer, count: dimensions))
                }
            }

            var output: [Scalar] = []
            output.reserveCapacity(dimensions)
            for value in state.comparisonStorage[start..<(start + dimensions)] {
                output.append(Scalar(value))
            }
            return try output.withUnsafeBufferPointer(body)
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
            loadedHalfComparisonStorage.reserveCapacity(labelCount * dimensions)
        } else {
            loadedComparisonStorage.reserveCapacity(labelCount * dimensions)
        }
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
            let offset = Scalar.self == Float16.self ? loadedHalfComparisonStorage.count : loadedComparisonStorage.count
            for _ in 0..<dimensions {
                let value = try reader.readFloat()
                if Scalar.self == Float16.self {
                    loadedHalfComparisonStorage.append(Float16(value))
                } else {
                    loadedComparisonStorage.append(value)
                }
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
            $0.queryScratch = Scalar.self == Float16.self ? [] : [Float](repeating: 0, count: dimensions)
            $0.halfQueryScratch = Scalar.self == Float16.self ? [Float16](repeating: 0, count: dimensions) : []
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
            loadedHalfComparisonStorage.reserveCapacity(labelCount * dimensions)
        } else {
            loadedComparisonStorage.reserveCapacity(labelCount * dimensions)
        }
        loadedConnections.reserveCapacity(labelCount)

        var generator = HNSWLevelGenerator(seed: HNSWConfiguration.balanced.randomSeed)
        let multiplier = 1.0 / Foundation.log(Double(max(2, HNSWConfiguration.balanced.m)))
        for internalID in 0..<labelCount {
            let label = try reader.readUInt64()
            let deleted = try reader.readBool()
            let offset = Scalar.self == Float16.self ? loadedHalfComparisonStorage.count : loadedComparisonStorage.count
            for _ in 0..<dimensions {
                let value = try reader.readFloat()
                if Scalar.self == Float16.self {
                    loadedHalfComparisonStorage.append(Float16(value))
                } else {
                    loadedComparisonStorage.append(value)
                }
            }
            let level = generator.randomLevel(multiplier: multiplier)
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
            $0.queryScratch = Scalar.self == Float16.self ? [] : [Float](repeating: 0, count: dimensions)
            $0.halfQueryScratch = Scalar.self == Float16.self ? [Float16](repeating: 0, count: dimensions) : []
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
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            VectorOperations.normalize(inputFloats, into: output)
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
        into output: UnsafeMutableBufferPointer<Float16>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        if Scalar.self == Float16.self {
            let inputHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: input.count
            )
            VectorOperations.normalize(inputHalves, into: output)
        } else {
            storeHalfVector(input, into: output)
            let outputSource = UnsafeBufferPointer(start: output.baseAddress!, count: output.count)
            VectorOperations.normalize(outputSource, into: output)
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
            let existingIndex = Int(existing.internalID)
            if existingIndex < state.deletedFlags.count {
                state.deletedFlags[existingIndex] = 0
            }
            shouldRebuildGraph = state.connections.hasAnyConnection(for: existing.internalID)
        } else {
            if state.entries.count < state.maximumElementCount {
                let internalID = HNSWInternalID(state.labelOrder.count)
                let offset = Scalar.self == Float16.self
                    ? state.halfComparisonStorage.count
                    : state.comparisonStorage.count
                let level = state.levelGenerator.randomLevel(multiplier: levelMultiplier)
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
                let level = state.levelGenerator.randomLevel(multiplier: levelMultiplier)
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
                    storeNormalizedHalfVector(vector, into: destination)
                } else {
                    storeHalfVector(vector, into: destination)
                }
            }
        } else {
            state.comparisonStorage.withUnsafeMutableBufferPointer { storage in
                let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                if metric.requiresNormalization {
                    storeNormalizedVector(vector, into: destination)
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
            initializedCount = state.searchScratch.withCandidateBuffers(
                candidateCapacity: candidateCapacity,
                nearestCapacity: nearestCapacity,
                resultCapacity: resultCapacity
            ) { candidateScratch, nearestScratch, resultScratch in
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
                                candidateScratch: candidateScratch,
                                nearestScratch: nearestScratch,
                                resultScratch: resultScratch,
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
                                candidateScratch: candidateScratch,
                                nearestScratch: nearestScratch,
                                resultScratch: resultScratch,
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
        let resultCount = state.searchScratch.withCandidateBuffers(
            candidateCapacity: candidateCapacity,
            nearestCapacity: nearestCapacity,
            resultCapacity: resultCapacity
        ) { candidateScratch, nearestScratch, resultScratch in
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
                            candidateScratch: candidateScratch,
                            nearestScratch: nearestScratch,
                            resultScratch: resultScratch,
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
                            candidateScratch: candidateScratch,
                            nearestScratch: nearestScratch,
                            resultScratch: resultScratch,
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
            initializedCount = state.searchScratch.withCandidateBuffers(
                candidateCapacity: candidateCapacity,
                nearestCapacity: nearestCapacity,
                resultCapacity: resultCapacity
            ) { candidateScratch, nearestScratch, resultScratch in
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
                                candidateScratch: candidateScratch,
                                nearestScratch: nearestScratch,
                                resultScratch: resultScratch,
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
                                candidateScratch: candidateScratch,
                                nearestScratch: nearestScratch,
                                resultScratch: resultScratch,
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
        let resultCount = state.searchScratch.withCandidateBuffers(
            candidateCapacity: candidateCapacity,
            nearestCapacity: nearestCapacity,
            resultCapacity: resultCapacity
        ) { candidateScratch, nearestScratch, resultScratch in
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
                            candidateScratch: candidateScratch,
                            nearestScratch: nearestScratch,
                            resultScratch: resultScratch,
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
                            candidateScratch: candidateScratch,
                            nearestScratch: nearestScratch,
                            resultScratch: resultScratch,
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
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
            candidateScratch: candidateScratch,
            nearestScratch: nearestScratch,
            resultScratch: resultScratch,
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
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
                candidateScratch: candidateScratch,
                nearestScratch: nearestScratch,
                resultScratch: resultScratch,
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
            candidateScratch: candidateScratch,
            nearestScratch: nearestScratch,
            resultScratch: resultScratch,
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
        1.0 / Foundation.log(Double(max(2, configuration.m)))
    }

    @inline(__always)
    private func maxConnections(at level: Int) -> Int {
        level == 0 ? max(1, configuration.m * 2) : max(1, configuration.m)
    }

    @inline(__always)
    private func newElementConnections(at level: Int) -> Int {
        max(1, configuration.m)
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
                let efConstruction = max(configuration.efConstruction, configuration.m)
                let nodeCount = state.labelOrder.count
                let candidateCapacity = max(1, nodeCount)
                let nearestCapacity = max(1, min(efConstruction, nodeCount))
                var candidates = state.searchScratch.withCandidateBuffers(
                    candidateCapacity: candidateCapacity,
                    nearestCapacity: nearestCapacity
                ) { candidateScratch, nearestScratch, _ in
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
                            candidateScratch: candidateScratch,
                            nearestScratch: nearestScratch,
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
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
            candidateScratch: candidateScratch,
            nearestScratch: nearestScratch,
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
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
                candidateScratch: candidateScratch,
                nearestScratch: nearestScratch,
                distanceComputer: Distance.self
            )
        }

        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWFixedNearestCandidateHeap(storage: candidateScratch)
        var nearest = HNSWFixedFarthestCandidateHeap(storage: nearestScratch)
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWFixedNearestCandidateHeap(storage: candidateScratch)
        var nearest = HNSWFixedFarthestCandidateHeap(storage: nearestScratch)

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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
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
            candidateScratch: candidateScratch,
            nearestScratch: nearestScratch,
            resultScratch: resultScratch,
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
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
                    candidateScratch: candidateScratch,
                    nearestScratch: nearestScratch,
                    resultScratch: resultScratch,
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
                candidateScratch: candidateScratch,
                nearestScratch: nearestScratch,
                resultScratch: resultScratch,
                distanceComputer: Distance.self,
                resultLimit: resultLimit
            )
        }

        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWFixedNearestCandidateHeap(storage: candidateScratch)
        var nearest = HNSWFixedFarthestCandidateHeap(storage: nearestScratch)
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

        return nearest.closestSorted(limit: resultLimit, using: resultScratch)
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWFixedNearestCandidateHeap(storage: candidateScratch)
        var nearest = HNSWFixedFarthestCandidateHeap(storage: nearestScratch)

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

        return nearest.closestSorted(limit: resultLimit, using: resultScratch)
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int,
        into results: UnsafeMutableBufferPointer<SearchResult>
    ) -> Int {
        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWFixedNearestCandidateHeap(storage: candidateScratch)
        var nearest = HNSWFixedFarthestCandidateHeap(storage: nearestScratch)

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
            using: resultScratch,
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
        candidateScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        nearestScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        resultScratch: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        distanceComputer: Distance.Type,
        resultLimit: Int
    ) -> [HNSWNeighborCandidate] {
        let effectiveEF = max(1, ef)
        let tag = nextVisitedTag(visited: visited, visitedTag: &visitedTag)

        var candidateQueue = HNSWFixedNearestCandidateHeap(storage: candidateScratch)
        var nearest = HNSWFixedFarthestCandidateHeap(storage: nearestScratch)

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

        return nearest.closestSorted(limit: resultLimit, using: resultScratch)
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
    var swiftBackendStorageCounts: (float: Int, half: Int) {
        state.withLock {
            ($0.comparisonStorage.count, $0.halfComparisonStorage.count)
        }
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
