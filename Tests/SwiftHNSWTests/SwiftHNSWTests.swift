import Testing
@testable import SwiftHNSW

@Suite("HNSW Index Tests")
struct SwiftHNSWTests {

    @Test("Create index and add points")
    func testCreateIndexAndAddPoints() throws {
        let index = try HNSWIndex(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        // Add some test vectors
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 0.0, 1.0, 0.0], label: 2)
        try index.add([0.0, 0.0, 0.0, 1.0], label: 3)

        #expect(index.count == 4)
    }

    @Test("Search for nearest neighbors")
    func testSearch() throws {
        let index = try HNSWIndex(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        // Add test vectors
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.9, 0.1, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)
        try index.add([0.0, 0.0, 1.0, 0.0], label: 3)

        // Search for nearest to [1, 0, 0, 0]
        let results = try index.search([1.0, 0.0, 0.0, 0.0], k: 2)

        #expect(results.count == 2)
        #expect(results[0].label == 0) // Exact match should be first
        #expect(results[0].distance == 0.0) // Distance should be 0
    }

    @Test("Dimension mismatch throws error")
    func testDimensionMismatch() throws {
        let index = try HNSWIndex(
            dimensions: 4,
            maxElements: 100
        )

        #expect(throws: HNSWError.self) {
            try index.add([1.0, 0.0, 0.0], label: 0) // Only 3 dimensions
        }
    }

    @Test("Inner product distance")
    func testInnerProductDistance() throws {
        let index = try HNSWIndex(
            dimensions: 4,
            maxElements: 100,
            metric: .innerProduct
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.5, 0.5, 0.0, 0.0], label: 1)

        let results = try index.search([1.0, 0.0, 0.0, 0.0], k: 2)
        #expect(results.count == 2)
    }

    @Test("Cosine similarity")
    func testCosineSimilarity() throws {
        let index = try HNSWIndex(
            dimensions: 4,
            maxElements: 100,
            metric: .cosine
        )

        // Different magnitudes but same direction
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([2.0, 0.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)

        let results = try index.search([3.0, 0.0, 0.0, 0.0], k: 2)
        #expect(results.count == 2)
        // Labels 0 and 1 should be closest (same direction)
        let topLabels = Set(results.map { $0.label })
        #expect(topLabels.contains(0) || topLabels.contains(1))
    }

    @Test("Set ef search parameter")
    func testSetEfSearch() throws {
        let index = try HNSWIndex(
            dimensions: 4,
            maxElements: 100,
            configuration: HNSWConfiguration(efSearch: 10)
        )

        index.setEfSearch(50)
        // Just verify it doesn't crash
        #expect(true)
    }

    @Test("Large scale test")
    func testLargeScale() throws {
        let dimensions = 128
        let numElements = 1000
        let index = try HNSWIndex(
            dimensions: dimensions,
            maxElements: numElements,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        )

        // Generate random vectors
        for i in 0..<numElements {
            var vector = [Float](repeating: 0, count: dimensions)
            for j in 0..<dimensions {
                vector[j] = Float.random(in: -1...1)
            }
            try index.add(vector, label: UInt64(i))
        }

        #expect(index.count == numElements)

        // Search
        var query = [Float](repeating: 0, count: dimensions)
        for i in 0..<dimensions {
            query[i] = Float.random(in: -1...1)
        }

        let results = try index.search(query, k: 10)
        #expect(results.count == 10)
    }
}
