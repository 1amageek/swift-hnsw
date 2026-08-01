import Foundation
import Testing
@testable import SwiftHNSW

@Suite("HNSW index contract", .serialized)
struct HNSWIndexContractTests {

    @Test("Rejects capacities outside the internal id range")
    func rejectsCapacitiesOutsideInternalIDRange() throws {
        #expect(throws: HNSWError.self) {
            _ = try HNSWIndex<Float>(
                dimensions: 1,
                maxElements: Int(UInt32.max) + 1,
                metric: .l2
            )
        }
        #expect(
            throws: HNSWError.initializationFailed(
                "Query workspace exceeds the 256 MiB limit"
            )
        ) {
            _ = try HNSWIndex<Float>(
                dimensions: HNSWSearchWorkspace.maximumRetainedByteCount
                    / MemoryLayout<Float>.stride + 1,
                maxElements: 1,
                metric: .l2
            )
        }
    }

    @Test("Exact L2 ordering and deterministic tie break")
    func exactL2OrderingAndTieBreak() throws {
        let index = try HNSWIndex<Float>(dimensions: 1, maxElements: 4, metric: .l2)
        try index.add([1], label: 20)
        try index.add([-1], label: 30)
        try index.add([0], label: 10)
        try index.add([0], label: 5)

        let results = try index.search([0], k: 4)

        #expect(results.map(\.label) == [5, 10, 20, 30])
        #expect(results.map(\.distance) == [0, 0, 1, 1])
        #expect(results.reversed().sorted().map(\.label) == [5, 10, 20, 30])
    }

    @Test("Deletion, restore, replacement, and capacity semantics")
    func mutationSemantics() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 2, metric: .l2)
        try index.add([0, 0], label: 1)
        try index.add([10, 0], label: 2)

        #expect(throws: HNSWError.self) {
            try index.add([20, 0], label: 3)
        }

        try index.markDeleted(label: 1)
        #expect(!index.contains(label: 1))
        #expect(index.getVector(label: 1) == nil)
        #expect(try index.search([0, 0], k: 2).map(\.label) == [2])

        try index.unmarkDeleted(label: 1)
        #expect(index.contains(label: 1))
        #expect(try index.search([0, 0], k: 1).first?.label == 1)

        try index.add([0, 10], label: 2)
        #expect(try index.search([0, 10], k: 1).first?.label == 2)
    }

    @Test("Deleted entry point remains navigable for later inserts")
    func deletedEntryPointRemainsNavigableForLaterInserts() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 2,
            maxElements: 2,
            metric: .l2,
            configuration: HNSWConfiguration(m: 2, efConstruction: 8, efSearch: 8)
        )
        try index.add([0, 0], label: 1)
        try index.markDeleted(label: 1)
        try index.add([1, 0], label: 2)

        let results = try index.search([1, 0], k: 2)

        #expect(results.map(\.label) == [2])
        #expect(!index.contains(label: 1))
    }

    @Test("Deleted capacity is reusable only when enabled")
    func deletedCapacityReuseRequiresConfiguration() throws {
        let disabled = try HNSWIndex<Float>(
            dimensions: 2,
            maxElements: 1,
            metric: .l2,
            configuration: HNSWConfiguration(allowReplaceDeleted: false)
        )
        try disabled.add([0, 0], label: 1)
        try disabled.markDeleted(label: 1)
        #expect(throws: HNSWError.self) {
            try disabled.add([1, 0], label: 2)
        }

        let enabled = try HNSWIndex<Float>(
            dimensions: 2,
            maxElements: 1,
            metric: .l2,
            configuration: HNSWConfiguration(allowReplaceDeleted: true)
        )
        try enabled.add([0, 0], label: 1)
        try enabled.markDeleted(label: 1)
        try enabled.add([1, 0], label: 2)

        #expect(enabled.count == 1)
        #expect(enabled.allLabels == [2])
        #expect(!enabled.contains(label: 1))
        #expect(try enabled.search([1, 0], k: 1).map(\.label) == [2])
    }

    @Test("Batch search and serialization roundtrip preserve labels and distances")
    func batchAndSerializationRoundtrip() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 8, metric: .l2)
        let vectors: [[Float]] = [
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 2],
        ]
        try index.addBatch(vectors, startingLabel: 100)

        let batchResults = try index.searchBatch([[0, 0], [2, 2]], k: 2)
        #expect(batchResults.map { $0.first?.label } == [100, 103])

        let archive = try index.serializedArchive()
        let loaded = try HNSWIndex<Float>.restore(
            from: archive,
            dimensions: 2,
            metric: .l2
        )

        #expect(loaded.count == 4)
        #expect(loaded.allLabels == [100, 101, 102, 103])
        #expect(try loaded.search([0, 0], k: 2).map(\.label) == [100, 101])
    }

    @Test("Borrowed flat buffer APIs preserve batch semantics")
    func borrowedFlatBufferAPIs() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 4, metric: .l2)
        let vectors: [Float] = [
            0, 0,
            2, 0,
            0, 3,
        ]
        let labels: [UInt64] = [10, 20, 30]

        let added = try vectors.withUnsafeBufferPointer { vectorBuffer in
            try labels.withUnsafeBufferPointer { labelBuffer in
                try index.addBatch(vectorBuffer, labels: labelBuffer)
            }
        }
        #expect(added == 3)

        let queries: [Float] = [
            0, 0,
            3, 0,
        ]
        let batchResults = try queries.withUnsafeBufferPointer { queryBuffer in
            try index.searchBatch(queryBuffer, numQueries: 2, k: 2)
        }

        #expect(batchResults.map { $0.first?.label } == [10, 20])

        let query: [Float] = [0, 2]
        let single = try query.withUnsafeBufferPointer { queryBuffer in
            try index.search(queryBuffer, k: 1)
        }
        #expect(single.first?.label == 30)
    }

    @Test("Float16 roundtrip preserves index semantics")
    func float16Roundtrip() throws {
        let index = try HNSWIndex<Float16>(dimensions: 2, maxElements: 4, metric: .cosine)
        try index.add([1, 0], label: 1)
        try index.add([0, 1], label: 2)

        let storageCounts = index.comparisonValueCounts
        #expect(storageCounts.singlePrecision == 0)
        #expect(storageCounts.halfPrecision == 4)

        let archive = try index.serializedArchive()
        let loaded = try HNSWIndex<Float16>.restore(
            from: archive,
            dimensions: 2,
            metric: .cosine
        )
        let loadedStorageCounts = loaded.comparisonValueCounts

        #expect(loadedStorageCounts.singlePrecision == 0)
        #expect(loadedStorageCounts.halfPrecision == 4)

        #expect(try loaded.search([2, 0], k: 2).map(\.label) == [1, 2])
    }

    @Test("Borrowed vector access matches materialized vector semantics")
    func borrowedVectorAccessMatchesMaterializedVectorSemantics() throws {
        let floatIndex = try HNSWIndex<Float>(dimensions: 3, maxElements: 2, metric: .l2)
        try floatIndex.add([1, 2, 3], label: 10)

        let floatSum = try #require(floatIndex.withVector(label: 10) { vector in
            vector.reduce(0, +)
        })
        #expect(floatSum == 6)
        #expect(floatIndex.getVector(label: 10) == [1, 2, 3])

        let halfIndex = try HNSWIndex<Float16>(dimensions: 3, maxElements: 2, metric: .l2)
        try halfIndex.add([1, 2, 3], label: 20)
        try halfIndex.markDeleted(label: 20)

        let missing = halfIndex.withVector(label: 20) { vector in
            vector.count
        }
        #expect(missing == nil)
    }

    @Test("Borrowed vector callback supports reentrant mutation", .timeLimit(.minutes(1)))
    func borrowedVectorCallbackSupportsReentrantMutation() throws {
        let floatIndex = try HNSWIndex<Float>(dimensions: 3, maxElements: 2, metric: .l2)
        try floatIndex.add([1, 2, 3], label: 10)

        let borrowedFloat = try floatIndex.withVector(label: 10) { vector in
            #expect(floatIndex.contains(label: 10))
            try floatIndex.add([4, 5, 6], label: 10)
            return Array(vector)
        }

        #expect(borrowedFloat == [1, 2, 3])
        #expect(floatIndex.getVector(label: 10) == [4, 5, 6])

        let halfIndex = try HNSWIndex<Float16>(dimensions: 3, maxElements: 2, metric: .l2)
        try halfIndex.add([1, 2, 3], label: 20)

        let borrowedHalf = try halfIndex.withVector(label: 20) { vector in
            #expect(halfIndex.contains(label: 20))
            try halfIndex.add([4, 5, 6], label: 20)
            return Array(vector)
        }

        #expect(borrowedHalf == [1, 2, 3])
        #expect(halfIndex.getVector(label: 20) == [4, 5, 6])
    }

    @Test("Invalid efSearch throws a typed error")
    func invalidEfSearchThrowsTypedError() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 2, metric: .l2)

        #expect(throws: HNSWError.invalidArgument("efSearch must be positive")) {
            try index.setEfSearch(0)
        }
        #expect(throws: HNSWError.invalidArgument("efSearch must be positive")) {
            try index.setEfSearch(-1)
        }
        try index.setEfSearch(1)
    }

    @Test("Invalid construction and search arguments throw typed errors")
    func invalidConstructionAndSearchArgumentsThrowTypedErrors() throws {
        #expect(throws: HNSWError.invalidArgument("m must be between 2 and 10000")) {
            _ = try HNSWIndex<Float>(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(m: 1)
            )
        }
        #expect(throws: HNSWError.invalidArgument("m must be between 2 and 10000")) {
            _ = try HNSWIndex<Float>(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(
                    m: HNSWConfiguration.maximumM + 1,
                    efConstruction: HNSWConfiguration.maximumM + 1
                )
            )
        }
        #expect(throws: HNSWError.invalidArgument("efConstruction must be at least m")) {
            _ = try HNSWIndex<Float>(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(m: 16, efConstruction: 15)
            )
        }
        #expect(throws: HNSWError.invalidArgument("efSearch must be positive")) {
            _ = try HNSWIndex<Float>(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(efSearch: 0)
            )
        }

        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 2)
        #expect(throws: HNSWError.invalidArgument("k must be positive")) {
            _ = try index.search([0, 0], k: 0)
        }
        #expect(throws: HNSWError.invalidArgument("k must be positive")) {
            _ = try index.searchBatch([[0, 0]], k: 0)
        }
        #expect(throws: HNSWError.invalidArgument("numQueries must not be negative")) {
            _ = try index.searchBatch([], numQueries: -1, k: 1)
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "batch dimensions exceed the supported integer range"
            )
        ) {
            _ = try index.searchBatch([], numQueries: Int.max, k: 1)
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "startingLabel and batch size exceed the UInt64 label range"
            )
        ) {
            _ = try index.addBatch([[0, 0], [1, 0]], startingLabel: UInt64.max)
        }
        #expect(throws: HNSWError.dimensionMismatch(expected: 0, got: 2)) {
            _ = try index.addBatch([0, 0], labels: [])
        }
    }

    @Test("Invalid numeric vectors fail before mutating the index")
    func invalidNumericVectorsFailBeforeMutation() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 4, metric: .l2)

        #expect(
            throws: HNSWError.invalidArgument(
                "vector must contain only finite values"
            )
        ) {
            try index.add([1, .nan], label: 1)
        }
        #expect(index.count == 0)

        #expect(
            throws: HNSWError.invalidArgument(
                "vector must contain only finite values"
            )
        ) {
            _ = try index.addBatch(
                [1, 0, .infinity, 0],
                labels: [1, 2]
            )
        }
        #expect(index.count == 0)

        #expect(
            throws: HNSWError.invalidArgument(
                "vector must contain only finite values"
            )
        ) {
            _ = try index.addBatch(
                [[1, 0], [.nan, 0]],
                startingLabel: 1
            )
        }
        #expect(index.count == 0)

        try index.add([1, 0], label: 1)
        #expect(
            throws: HNSWError.invalidArgument(
                "query must contain only finite values"
            )
        ) {
            _ = try index.search([1, -.infinity], k: 1)
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "query must contain only finite values"
            )
        ) {
            _ = try index.searchBatch(
                [1, 0, .nan, 0],
                numQueries: 2,
                k: 1
            )
        }

        var output = [SearchResult](
            repeating: SearchResult(label: 0, distance: 0),
            count: 1
        )
        #expect(
            throws: HNSWError.invalidArgument(
                "query must contain only finite values"
            )
        ) {
            try output.withUnsafeMutableBufferPointer { outputBuffer in
                _ = try index.search(
                    [.nan, 0],
                    k: 1,
                    into: outputBuffer
                )
            }
        }
    }

    @Test("Cosine vectors require finite nonzero norms")
    func cosineVectorsRequireFiniteNonzeroNorms() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 2,
            maxElements: 2,
            metric: .cosine
        )

        #expect(
            throws: HNSWError.invalidArgument("vector must have nonzero norm")
        ) {
            try index.add([0, 0], label: 1)
        }
        #expect(
            throws: HNSWError.invalidArgument("vector norm must be finite")
        ) {
            try index.add(
                [.greatestFiniteMagnitude, .greatestFiniteMagnitude],
                label: 1
            )
        }

        try index.add([1, 0], label: 1)
        #expect(
            throws: HNSWError.invalidArgument("query must have nonzero norm")
        ) {
            _ = try index.search([0, 0], k: 1)
        }
    }

    @Test("Resize preserves the archive capacity contract")
    func resizePreservesArchiveCapacityContract() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 2)

        #expect(
            throws: HNSWError.invalidArgument("newCapacity must be positive")
        ) {
            try index.resize(to: 0)
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "newCapacity must fit the archive format"
            )
        ) {
            try index.resize(to: Int(UInt32.max) + 1)
        }
        try index.resize(to: 4)
        #expect(index.capacity == 4)
        _ = try index.serializedArchive()
    }

    @Test("Search workspace rejects unaddressable retained storage")
    func searchWorkspaceRejectsUnaddressableRetainedStorage() {
        #expect(
            throws: HNSWError.invalidArgument(
                "search workspace size exceeds the supported integer range"
            )
        ) {
            try HNSWSearchWorkspace.validateRetainedCapacity(
                candidateCapacity: Int.max,
                nearestCapacity: Int.max,
                resultCapacity: Int.max,
                visitedCapacity: Int.max
            )
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "search workspace exceeds the 256 MiB limit"
            )
        ) {
            try HNSWSearchWorkspace.validateRetainedCapacity(
                candidateCapacity: 30_000_000,
                nearestCapacity: 1,
                resultCapacity: 1,
                visitedCapacity: 30_000_000
            )
        }
    }

    @Test("Search workspace grows geometrically without changing requested bounds")
    func searchWorkspaceGrowsGeometrically() {
        var workspace = HNSWSearchWorkspace()
        let firstCounts = workspace.withCandidateBuffers(
            candidateCapacity: 2,
            nearestCapacity: 1,
            visitedCapacity: 2
        ) { candidates, nearest, results in
            [candidates.count, nearest.count, results.count]
        }
        let secondCounts = workspace.withCandidateBuffers(
            candidateCapacity: 3,
            nearestCapacity: 1,
            visitedCapacity: 3
        ) { candidates, nearest, results in
            [candidates.count, nearest.count, results.count]
        }

        #expect(firstCounts == [2, 1, 1])
        #expect(secondCounts == [3, 1, 1])
        #expect(workspace.retainedCandidateCapacities.candidate == 4)
    }

    @Test("Concurrent reads and writes preserve index state", .timeLimit(.minutes(1)))
    func concurrentReadsAndWritesPreserveIndexState() async throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 50, metric: .l2)
        for label in 0..<10 {
            try index.add([Float(label), 0], label: UInt64(label))
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    for _ in 0..<50 {
                        _ = try index.search([0, 0], k: 5)
                    }
                }
            }
            group.addTask {
                for label in 10..<50 {
                    try index.add([Float(label), 0], label: UInt64(label))
                }
            }
            try await group.waitForAll()
        }

        #expect(index.count == 50)
        #expect(index.allLabels == (0..<50).map(UInt64.init))
        #expect(try index.search([49, 0], k: 1).first?.label == 49)
    }

    @Test("Search into caller buffer preserves ordering and count semantics")
    func searchIntoCallerBufferPreservesOrderingAndCountSemantics() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 4, metric: .l2)
        try index.add([0, 0], label: 10)
        try index.add([2, 0], label: 20)
        try index.add([4, 0], label: 30)

        var exactBuffer = [SearchResult](repeating: SearchResult(label: 0, distance: 0), count: 2)
        let exactCount = try exactBuffer.withUnsafeMutableBufferPointer { output in
            try index.search([1, 0], k: 2, into: output)
        }

        #expect(exactCount == 2)
        #expect(exactBuffer.map(\.label) == [10, 20])

        var oversizedBuffer = [SearchResult](repeating: SearchResult(label: 0, distance: 0), count: 5)
        let oversizedCount = try oversizedBuffer.withUnsafeMutableBufferPointer { output in
            try index.search([1, 0], k: 5, into: output)
        }

        #expect(oversizedCount == 3)
        #expect(oversizedBuffer.prefix(oversizedCount).map(\.label) == [10, 20, 30])

        var undersizedBuffer = [SearchResult](repeating: SearchResult(label: 0, distance: 0), count: 1)
        #expect(throws: HNSWError.self) {
            try undersizedBuffer.withUnsafeMutableBufferPointer { output in
                try index.search([1, 0], k: 2, into: output)
            }
        }
    }

    @Test("Visited tag wraparound keeps search correct", .timeLimit(.minutes(1)))
    func visitedTagWraparoundKeepsSearchCorrect() throws {
        let index = try HNSWIndex<Float>(dimensions: 2, maxElements: 3, metric: .l2)
        try index.add([0, 0], label: 10)
        try index.add([1, 0], label: 20)
        try index.add([2, 0], label: 30)

        for _ in 0..<(Int(UInt16.max) + 4) {
            let results = try index.search([0, 0], k: 2)
            #expect(results.map(\.label) == [10, 20])
        }
    }

    @Test("Graph loader rejects invalid self edge")
    func graphLoaderRejectsInvalidSelfEdge() throws {
        let payload = invalidGraphPayloadWithSelfEdge()

        #expect(throws: HNSWError.self) {
            _ = try HNSWIndex<Float>.restore(
                from: payload,
                dimensions: 1,
                metric: .l2
            )
        }
    }

    @Test("Graph loader rejects invalid neighbor id")
    func graphLoaderRejectsInvalidNeighborID() throws {
        let payload = invalidGraphPayloadWithInvalidNeighborID()

        #expect(throws: HNSWError.self) {
            _ = try HNSWIndex<Float>.restore(
                from: payload,
                dimensions: 1,
                metric: .l2
            )
        }
    }

    @Test("Graph loader rejects invalid vector values")
    func graphLoaderRejectsInvalidVectorValues() throws {
        #expect(
            throws: HNSWError.loadFailed(
                "Graph index contains non-finite vector values"
            )
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: invalidGraphPayloadWithVector(.nan, metric: .l2),
                dimensions: 1,
                metric: .l2
            )
        }

        #expect(
            throws: HNSWError.loadFailed(
                "Graph index contains values outside the Float16 range"
            )
        ) {
            _ = try HNSWIndex<Float16>.restore(
                from: invalidGraphPayloadWithVector(
                    .greatestFiniteMagnitude,
                    metric: .l2
                ),
                dimensions: 1,
                metric: .l2
            )
        }

        #expect(
            throws: HNSWError.loadFailed(
                "Graph index contains an invalid cosine vector norm"
            )
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: invalidGraphPayloadWithVector(0, metric: .cosine),
                dimensions: 1,
                metric: .cosine
            )
        }
    }

    @Test("Legacy flat loader rejects invalid vectors and duplicate labels")
    func flatLoaderRejectsInvalidVectorsAndDuplicateLabels() throws {
        #expect(
            throws: HNSWError.loadFailed(
                "Flat index contains non-finite vector values"
            )
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: invalidFlatPayload(vectorValues: [.nan]),
                dimensions: 1,
                metric: .l2
            )
        }

        #expect(
            throws: HNSWError.loadFailed("Flat index contains duplicate labels")
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: invalidFlatPayload(
                    vectorValues: [0, 1],
                    labels: [1, 1]
                ),
                dimensions: 1,
                metric: .l2
            )
        }
    }

    @Test("Graph loader rejects vector storage overflow before allocation")
    func graphLoaderRejectsVectorStorageOverflowBeforeAllocation() throws {
        let payload = invalidGraphPayloadWithOversizedVectorStorage()

        #expect(
            throws: HNSWError.loadFailed(
                "Graph index vector storage exceeds the addressable range"
            )
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: payload,
                dimensions: Int(UInt32.max),
                metric: .l2
            )
        }
    }

    @Test("Graph loader rejects retained graph storage before allocation")
    func graphLoaderRejectsOversizedRetainedGraph() throws {
        let payload = invalidGraphPayloadWithOversizedRetainedGraph()

        #expect(
            throws: HNSWError.loadFailed(
                "Graph index retained graph storage exceeds the 256 MiB limit"
            )
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: payload,
                dimensions: 1,
                metric: .l2
            )
        }
    }

    @Test("Graph loader rejects cumulative upper-level storage before allocation")
    func graphLoaderRejectsOversizedUpperLevelStorage() throws {
        let payload = invalidGraphPayloadWithOversizedUpperLevelStorage()

        #expect(
            throws: HNSWError.loadFailed(
                "Graph index retained graph storage exceeds the 256 MiB limit"
            )
        ) {
            _ = try HNSWIndex<Float>.restore(
                from: payload,
                dimensions: 1,
                metric: .l2
            )
        }
    }
}

@Suite("TurboQuant index contract", .serialized)
struct TurboQuantIndexContractTests {

    @Test("Invalid efSearch throws a typed error")
    func invalidEfSearchThrowsTypedError() throws {
        let index = try TurboQuantIndex(dimensions: 2, maxElements: 2, bitWidth: 4)

        #expect(throws: HNSWError.invalidArgument("efSearch must be positive")) {
            try index.setEfSearch(0)
        }
        #expect(throws: HNSWError.invalidArgument("efSearch must be positive")) {
            try index.setEfSearch(-1)
        }
        #expect(throws: HNSWError.invalidArgument("efSearch must fit the archive format")) {
            try index.setEfSearch(Int(UInt32.max) + 1)
        }
        try index.setEfSearch(1)
    }

    @Test("Invalid construction and search arguments throw typed errors")
    func invalidConstructionAndSearchArgumentsThrowTypedErrors() throws {
        #expect(throws: HNSWError.invalidArgument("m must be between 2 and 10000")) {
            _ = try TurboQuantIndex(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(
                    m: HNSWConfiguration.maximumM + 1,
                    efConstruction: HNSWConfiguration.maximumM + 1
                )
            )
        }
        #expect(throws: HNSWError.invalidArgument("efConstruction must be at least m")) {
            _ = try TurboQuantIndex(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(m: 16, efConstruction: 15)
            )
        }
        #expect(throws: HNSWError.invalidArgument("efSearch must be positive")) {
            _ = try TurboQuantIndex(
                dimensions: 2,
                maxElements: 2,
                configuration: HNSWConfiguration(efSearch: 0)
            )
        }

        let index = try TurboQuantIndex(dimensions: 2, maxElements: 2)
        #expect(throws: HNSWError.invalidArgument("k must be positive")) {
            _ = try index.search([0, 0], k: 0)
        }
    }

    @Test("TurboQuant cosine ordering and deterministic tie break")
    func turboQuantCosineOrderingAndTieBreak() throws {
        let index = try TurboQuantIndex(dimensions: 2, maxElements: 4, bitWidth: 4)
        try index.add([1, 0], label: 20)
        try index.add([0, 1], label: 30)
        try index.add([2, 0], label: 10)

        let results = try index.search([4, 0], k: 3)

        #expect(results.map(\.label) == [10, 20, 30])
        // TurboQuant reports ADC distance, so quantization error is expected even for
        // identical normalized vectors. Ordering and deterministic label tie-breaking
        // remain part of the public contract.
        #expect(results[0].distance < 0.1)
        #expect(results[1].distance < 0.1)
    }

    @Test("Finalize, capacity, save, and load semantics")
    func finalizeCapacityAndLoadSemantics() throws {
        let index = try TurboQuantIndex(dimensions: 3, maxElements: 2, bitWidth: 3, seed: 99)
        try index.add([1, 0, 0], label: 1)
        try index.add([0, 1, 0], label: 2)

        #expect(throws: HNSWError.self) {
            try index.add([0, 0, 1], label: 3)
        }

        let before = try index.search([1, 0, 0], k: 2)
        #expect(index.isFinalized)
        #expect(throws: HNSWError.self) {
            try index.add([0, 0, 1], label: 3)
        }

        let path = temporaryPath(name: "turboquant_roundtrip")
        defer { removeFileIfPresent(path) }
        try index.save(to: path)

        let loaded = try TurboQuantIndex.load(from: path)
        let after = try loaded.search([1, 0, 0], k: 2)

        #expect(loaded.dimensions == 3)
        #expect(loaded.bitWidth == 3)
        #expect(loaded.seed == 99)
        #expect(loaded.isFinalized)
        #expect(before.map(\.label) == after.map(\.label))
    }

    @Test("Borrowed TurboQuant APIs preserve cosine semantics")
    func borrowedTurboQuantAPIs() throws {
        let index = try TurboQuantIndex(dimensions: 2, maxElements: 2, bitWidth: 4)
        let first: [Float] = [1, 0]
        let second: [Float] = [0, 1]

        try first.withUnsafeBufferPointer { buffer in
            try index.add(buffer, label: 1)
        }
        try second.withUnsafeBufferPointer { buffer in
            try index.add(buffer, label: 2)
        }

        let query: [Float] = [3, 0]
        let results = try query.withUnsafeBufferPointer { buffer in
            try index.search(buffer, k: 2)
        }

        #expect(results.map(\.label) == [1, 2])
    }

    @Test("Concurrent TurboQuant searches preserve deterministic results")
    func concurrentTurboQuantSearches() async throws {
        let dimensions = 32
        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: 64,
            bitWidth: 3,
            objective: .innerProduct
        )
        for label in 0..<64 {
            try index.add(
                (0..<dimensions).map {
                    Float((label * 13 + $0 * 7) % 31 - 15)
                },
                label: UInt64(label)
            )
        }
        let query = (0..<dimensions).map { Float(($0 * 11) % 29 - 14) }
        let expected = try index.search(query, k: 10)

        try await withThrowingTaskGroup(of: [SearchResult].self) { group in
            for _ in 0..<16 {
                group.addTask {
                    try index.search(query, k: 10)
                }
            }
            for try await results in group {
                #expect(results == expected)
            }
        }
    }

    @Test("Invalid serialized payload throws typed load error")
    func invalidPayloadThrows() throws {
        let path = temporaryPath(name: "turboquant_invalid_archive")
        defer { removeFileIfPresent(path) }
        try Data([0, 1, 2, 3]).write(to: URL(fileURLWithPath: path))

        #expect(throws: HNSWError.self) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }

    @Test("Invalid TurboQuant objective and rotation metadata fail explicitly")
    func invalidTurboQuantMetadataThrows() throws {
        let index = try TurboQuantIndex(
            dimensions: 4,
            maxElements: 1,
            bitWidth: 4,
            objective: .innerProduct,
            rotationStrategy: .haar
        )
        try index.add([1, 0, 0, 0], label: 1)

        let path = temporaryPath(name: "turboquant_invalid_metadata")
        defer { removeFileIfPresent(path) }
        try index.save(to: path)
        let validArchive = try Data(contentsOf: URL(fileURLWithPath: path))

        var invalidObjective = validArchive
        invalidObjective.replaceSubrange(16..<20, with: [0xFF, 0xFF, 0xFF, 0xFF])
        try invalidObjective.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains an invalid objective"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var invalidRotation = validArchive
        invalidRotation.replaceSubrange(20..<24, with: [0xFF, 0xFF, 0xFF, 0xFF])
        try invalidRotation.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains an invalid rotation strategy"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var invalidM = validArchive
        invalidM.replaceSubrange(44..<48, with: [0x11, 0x27, 0, 0])
        try invalidM.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains an invalid m"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }

    @Test("TurboQuant loader rejects retained graph storage before allocation")
    func turboQuantLoaderRejectsOversizedRetainedGraph() throws {
        let path = temporaryPath(name: "turboquant_oversized_retained_graph")
        defer { removeFileIfPresent(path) }
        try invalidTurboQuantPayloadWithOversizedRetainedGraph().write(
            to: URL(fileURLWithPath: path)
        )

        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive retained graph storage exceeds the 256 MiB limit"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }

    @Test("TurboQuant loader rejects cumulative upper-level storage before allocation")
    func turboQuantLoaderRejectsOversizedUpperLevelStorage() throws {
        let path = temporaryPath(name: "turboquant_oversized_upper_level_storage")
        defer { removeFileIfPresent(path) }
        try invalidTurboQuantPayloadWithOversizedUpperLevelStorage().write(
            to: URL(fileURLWithPath: path)
        )

        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive retained graph storage exceeds the 256 MiB limit"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }

    @Test("TurboQuant archive rejects noncanonical values and truncated records")
    func invalidTurboQuantRecordThrows() throws {
        let index = try TurboQuantIndex(
            dimensions: 4,
            maxElements: 1,
            bitWidth: 4,
            objective: .innerProduct,
            rotationStrategy: .haar
        )
        try index.add([1, 0, 0, 0], label: 1)

        let path = temporaryPath(name: "turboquant_invalid_record")
        defer { removeFileIfPresent(path) }
        try index.save(to: path)
        let validArchive = try Data(contentsOf: URL(fileURLWithPath: path))

        var invalidBoolean = validArchive
        invalidBoolean[64] = 2
        try invalidBoolean.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains an invalid allowReplaceDeleted value"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var invalidReservedByte = validArchive
        invalidReservedByte[65] = 1
        try invalidReservedByte.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains nonzero reserved header bytes"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var invalidPadding = validArchive
        invalidPadding[105] |= 0x80
        try invalidPadding.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains nonzero MSE code padding bits"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var invalidResidualNorm = validArchive
        let residualOffset = 107
        var nanBits = Float.nan.bitPattern.littleEndian
        withUnsafeBytes(of: &nanBits) { bytes in
            invalidResidualNorm.replaceSubrange(
                residualOffset..<(residualOffset + bytes.count),
                with: bytes
            )
        }
        try invalidResidualNorm.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains an invalid residual norm"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var oversizedResidualNorm = validArchive
        var maximumFiniteBits = Float.greatestFiniteMagnitude.bitPattern.littleEndian
        withUnsafeBytes(of: &maximumFiniteBits) { bytes in
            oversizedResidualNorm.replaceSubrange(
                residualOffset..<(residualOffset + bytes.count),
                with: bytes
            )
        }
        try oversizedResidualNorm.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains an invalid residual norm"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }

        var truncatedRecord = validArchive
        truncatedRecord.removeLast()
        try truncatedRecord.write(to: URL(fileURLWithPath: path))
        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive payload is too small for its declared label count"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }

    @Test("TurboQuant archive rejects invalid graph topology")
    func invalidTurboQuantGraphThrows() throws {
        let path = temporaryPath(name: "turboquant_invalid_graph")
        defer { removeFileIfPresent(path) }
        try invalidTurboQuantGraphPayloadWithSelfEdge().write(
            to: URL(fileURLWithPath: path)
        )

        #expect(
            throws: HNSWError.loadFailed(
                "TurboQuant archive contains a self edge"
            )
        ) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }
}

@Suite(
    "Index performance",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["HNSW_PERFORMANCE_TESTS"] != nil)
)
struct IndexPerformanceTests {

    @Test("Flat HNSW index performance smoke")
    func hnswFlatSearchPerformanceSmoke() throws {
        let dimensions = 64
        let count = 2_000
        let queryCount = 20
        let index = try HNSWIndex<Float>(dimensions: dimensions, maxElements: count, metric: .l2)
        let vectors = deterministicVectors(count: count, dimensions: dimensions)
        let queries = deterministicVectors(count: queryCount, dimensions: dimensions)

        let buildSeconds = try measureSeconds {
            for (offset, vector) in vectors.enumerated() {
                try index.add(vector, label: UInt64(offset))
            }
        }

        let searchSeconds = try measureSeconds {
            for query in queries {
                let results = try index.search(query, k: 10)
                #expect(results.count == 10)
            }
        }

        let vectorsPerSecond = Double(count) / max(buildSeconds, .leastNonzeroMagnitude)
        let queriesPerSecond = Double(queryCount) / max(searchSeconds, .leastNonzeroMagnitude)
        print("HNSW index: build=\(vectorsPerSecond) vectors/s search=\(queriesPerSecond) qps")

        #expect(vectorsPerSecond > 10_000)
        #expect(queriesPerSecond > 100)
    }

    @Test("TurboQuant index performance smoke")
    func turboQuantSearchPerformanceSmoke() throws {
        let dimensions = 64
        let count = 2_000
        let queryCount = 20
        let index = try TurboQuantIndex(dimensions: dimensions, maxElements: count, bitWidth: 4)
        let vectors = deterministicVectors(count: count, dimensions: dimensions)
        let queries = deterministicVectors(count: queryCount, dimensions: dimensions)

        let buildSeconds = try measureSeconds {
            for (offset, vector) in vectors.enumerated() {
                try index.add(vector, label: UInt64(offset))
            }
        }

        let searchSeconds = try measureSeconds {
            for query in queries {
                let results = try index.search(query, k: 10)
                #expect(results.count == 10)
            }
        }

        let vectorsPerSecond = Double(count) / max(buildSeconds, .leastNonzeroMagnitude)
        let queriesPerSecond = Double(queryCount) / max(searchSeconds, .leastNonzeroMagnitude)
        print("TurboQuant index: build=\(vectorsPerSecond) vectors/s search=\(queriesPerSecond) qps")

        #expect(vectorsPerSecond > 10_000)
        #expect(queriesPerSecond > 100)
    }
}

private func deterministicVectors(count: Int, dimensions: Int) -> [[Float]] {
    (0..<count).map { row in
        (0..<dimensions).map { column in
            let value = ((row * 31 + column * 17) % 97) - 48
            return Float(value) / 48
        }
    }
}

private func measureSeconds(_ body: () throws -> Void) rethrows -> Double {
    let clock = ContinuousClock()
    let start = clock.now
    try body()
    let duration = start.duration(to: clock.now)
    let components = duration.components
    let seconds = Double(components.seconds)
    let attoseconds = Double(components.attoseconds) / 1_000_000_000_000_000_000
    return seconds + attoseconds
}

private func temporaryPath(name: String) -> String {
    NSTemporaryDirectory() + "\(name)_\(UUID().uuidString).bin"
}

private func removeFileIfPresent(_ path: String) {
    do {
        try FileManager.default.removeItem(atPath: path)
    } catch CocoaError.fileNoSuchFile {
        return
    } catch {
        return
    }
}

private func invalidGraphPayloadWithVector(
    _ value: Float,
    metric: DistanceMetric
) -> Data {
    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
    writer.writeUInt32(2)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeString(metric.rawValue)
    writer.writeUInt32(16)
    writer.writeUInt32(200)
    writer.writeUInt32(50)
    writer.writeUInt32(100)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeUInt32(0)
    writer.writeUInt64(1)
    writer.writeUInt32(1)
    writer.writeUInt64(1)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeFloat(value)
    writer.writeUInt32(1)
    writer.writeUInt32(0)
    return writer.data
}

private func invalidFlatPayload(
    vectorValues: [Float],
    labels: [UInt64]? = nil
) -> Data {
    let archiveLabels = labels ?? Array(1...UInt64(vectorValues.count))
    precondition(archiveLabels.count == vectorValues.count)

    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x46, 0x4C, 0x41])
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(UInt32(vectorValues.count))
    writer.writeString("l2")
    writer.writeUInt32(UInt32(vectorValues.count))
    for (label, value) in zip(archiveLabels, vectorValues) {
        writer.writeUInt64(label)
        writer.writeBool(false)
        writer.writeFloat(value)
    }
    return writer.data
}

private func invalidGraphPayloadWithSelfEdge() -> Data {
    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
    writer.writeUInt32(2)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeString("l2")
    writer.writeUInt32(16)
    writer.writeUInt32(200)
    writer.writeUInt32(50)
    writer.writeUInt32(100)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeUInt32(0)
    writer.writeUInt64(1)
    writer.writeUInt32(1)
    writer.writeUInt64(1)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeFloat(0)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(0)
    return writer.data
}

private func invalidGraphPayloadWithInvalidNeighborID() -> Data {
    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
    writer.writeUInt32(2)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeString("l2")
    writer.writeUInt32(16)
    writer.writeUInt32(200)
    writer.writeUInt32(50)
    writer.writeUInt32(100)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeUInt32(0)
    writer.writeUInt64(1)
    writer.writeUInt32(1)
    writer.writeUInt64(1)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeFloat(0)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    return writer.data
}

private func invalidGraphPayloadWithOversizedVectorStorage() -> Data {
    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
    writer.writeUInt32(2)
    writer.writeUInt32(UInt32.max)
    writer.writeUInt32(1)
    writer.writeString("l2")
    writer.writeUInt32(16)
    writer.writeUInt32(200)
    writer.writeUInt32(50)
    writer.writeUInt32(100)
    writer.writeBool(false)
    writer.writeUInt32(UInt32.max)
    writer.writeUInt32(UInt32.max)
    writer.writeUInt64(1)
    writer.writeUInt32(UInt32.max)
    return writer.data
}

private func invalidGraphPayloadWithOversizedRetainedGraph() -> Data {
    let labelCount = 4_000
    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
    writer.writeUInt32(2)
    writer.writeUInt32(1)
    writer.writeUInt32(UInt32(labelCount))
    writer.writeString("l2")
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(50)
    writer.writeUInt32(100)
    writer.writeBool(false)
    writer.writeUInt32(0)
    writer.writeUInt32(0)
    writer.writeUInt64(1)
    writer.writeUInt32(UInt32(labelCount))
    for label in 0..<labelCount {
        writer.writeUInt64(UInt64(label))
        writer.writeBool(false)
        writer.writeUInt32(0)
        writer.writeFloat(0)
        writer.writeUInt32(1)
        writer.writeUInt32(0)
    }
    return writer.data
}

private func invalidGraphPayloadWithOversizedUpperLevelStorage() -> Data {
    let labelCount = 107
    let level = 63
    var writer = InvalidGraphPayloadWriter()
    writer.writeBytes([0x53, 0x48, 0x4E, 0x53, 0x57, 0x47, 0x52, 0x46])
    writer.writeUInt32(2)
    writer.writeUInt32(1)
    writer.writeUInt32(UInt32(labelCount))
    writer.writeString("l2")
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(50)
    writer.writeUInt32(100)
    writer.writeBool(false)
    writer.writeUInt32(UInt32(level))
    writer.writeUInt32(0)
    writer.writeUInt64(1)
    writer.writeUInt32(UInt32(labelCount))
    for label in 0..<labelCount {
        writer.writeUInt64(UInt64(label))
        writer.writeBool(false)
        writer.writeUInt32(UInt32(level))
        writer.writeFloat(0)
        writer.writeUInt32(UInt32(level + 1))
        for _ in 0...level {
            writer.writeUInt32(0)
        }
    }
    return writer.data
}

private func invalidTurboQuantGraphPayloadWithSelfEdge() -> Data {
    var writer = InvalidGraphPayloadWriter()
    writer.writeUInt32(0x5451_5746)
    writer.writeUInt32(5)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(TurboQuantObjective.meanSquaredError.rawValue)
    writer.writeUInt32(TurboQuantRotationStrategy.structuredHadamard.rawValue)
    writer.writeUInt64(42)
    writer.writeUInt32(1)
    writer.writeUInt32(2)
    writer.writeUInt32(2)
    writer.writeUInt32(2)
    writer.writeUInt32(2)
    writer.writeUInt32(10)
    writer.writeUInt64(100)
    writer.writeBool(false)
    writer.writeBytes([0, 0, 0])
    writer.writeUInt32(0)
    writer.writeUInt32(0)
    writer.writeUInt32(1)

    writer.writeUInt64(10)
    writer.writeBool(false)
    writer.writeBytes([0, 0, 0])
    writer.writeUInt32(0)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(0)
    writer.writeBytes([0])

    writer.writeUInt64(20)
    writer.writeBool(false)
    writer.writeBytes([0, 0, 0])
    writer.writeUInt32(0)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(0)
    writer.writeBytes([0])
    return writer.data
}

private func invalidTurboQuantPayloadWithOversizedRetainedGraph() -> Data {
    let labelCount = 4_000
    var writer = InvalidGraphPayloadWriter()
    writer.writeUInt32(0x5451_5746)
    writer.writeUInt32(5)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(TurboQuantObjective.meanSquaredError.rawValue)
    writer.writeUInt32(TurboQuantRotationStrategy.structuredHadamard.rawValue)
    writer.writeUInt64(42)
    writer.writeUInt32(1)
    writer.writeUInt32(UInt32(labelCount))
    writer.writeUInt32(UInt32(labelCount))
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(10)
    writer.writeUInt64(100)
    writer.writeBool(false)
    writer.writeBytes([0, 0, 0])
    writer.writeUInt32(0)
    writer.writeUInt32(0)
    writer.writeUInt32(1)

    for label in 0..<labelCount {
        writer.writeUInt64(UInt64(label))
        writer.writeBool(false)
        writer.writeBytes([0, 0, 0])
        writer.writeUInt32(0)
        writer.writeUInt32(1)
        writer.writeUInt32(0)
        writer.writeBytes([0])
    }
    return writer.data
}

private func invalidTurboQuantPayloadWithOversizedUpperLevelStorage() -> Data {
    let labelCount = 107
    let level = 63
    var writer = InvalidGraphPayloadWriter()
    writer.writeUInt32(0x5451_5746)
    writer.writeUInt32(5)
    writer.writeUInt32(1)
    writer.writeUInt32(1)
    writer.writeUInt32(TurboQuantObjective.meanSquaredError.rawValue)
    writer.writeUInt32(TurboQuantRotationStrategy.structuredHadamard.rawValue)
    writer.writeUInt64(42)
    writer.writeUInt32(1)
    writer.writeUInt32(UInt32(labelCount))
    writer.writeUInt32(UInt32(labelCount))
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(UInt32(HNSWConfiguration.maximumM))
    writer.writeUInt32(10)
    writer.writeUInt64(100)
    writer.writeBool(false)
    writer.writeBytes([0, 0, 0])
    writer.writeUInt32(0)
    writer.writeUInt32(UInt32(level))
    writer.writeUInt32(1)

    for label in 0..<labelCount {
        writer.writeUInt64(UInt64(label))
        writer.writeBool(false)
        writer.writeBytes([0, 0, 0])
        writer.writeUInt32(UInt32(level))
        writer.writeUInt32(UInt32(level + 1))
        for _ in 0...level {
            writer.writeUInt32(0)
        }
        writer.writeBytes([0])
    }
    return writer.data
}

private struct InvalidGraphPayloadWriter {
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
