import Testing
@testable import SwiftHNSW

@Suite("HNSW Index Tests")
struct HNSWIndexTests {

    @Test("Create index and add points")
    func testCreateIndexAndAddPoints() throws {
        let index = try HNSWIndex<Float>(
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
        let index = try HNSWIndex<Float>(
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
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100
        )

        #expect(throws: HNSWError.self) {
            try index.add([1.0, 0.0, 0.0], label: 0) // Only 3 dimensions
        }
    }

    @Test("Inner product distance")
    func testInnerProductDistance() throws {
        let index = try HNSWIndex<Float>(
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
        let index = try HNSWIndex<Float>(
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
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100,
            configuration: HNSWConfiguration(efSearch: 10)
        )

        try index.setEfSearch(50)
        // Just verify it doesn't crash
        #expect(Bool(true))
    }

}

// MARK: - Float16 Tests

@Suite("Float16 Index Tests")
struct Float16Tests {

    @Test("Create Float16 index and add points")
    func testCreateFloat16IndexAndAddPoints() throws {
        let index = try HNSWIndex<Float16>(
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

    @Test("Float16 search for nearest neighbors")
    func testFloat16Search() throws {
        let index = try HNSWIndex<Float16>(
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
        let query: [Float16] = [1.0, 0.0, 0.0, 0.0]
        let results = try index.search(query, k: 2)

        #expect(results.count == 2)
        #expect(results[0].label == 0) // Exact match should be first
    }

    @Test("Float16 inner product")
    func testFloat16InnerProduct() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .innerProduct
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.5, 0.5, 0.0, 0.0], label: 1)

        let query: [Float16] = [1.0, 0.0, 0.0, 0.0]
        let results = try index.search(query, k: 2)
        #expect(results.count == 2)
    }

    @Test("Float16 cosine similarity")
    func testFloat16Cosine() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .cosine
        )

        // Different magnitudes but same direction
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([2.0, 0.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)

        let query: [Float16] = [3.0, 0.0, 0.0, 0.0]
        let results = try index.search(query, k: 2)
        #expect(results.count == 2)
    }

    @Test("Float16 getVector roundtrip")
    func testFloat16GetVector() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        let vector: [Float16] = [1.0, 0.5, 0.25, 0.125]
        try index.add(vector, label: 42)

        let retrieved = index.getVector(label: 42)
        #expect(retrieved != nil)

        // Check values are approximately equal (Float16 has limited precision)
        if let retrieved = retrieved {
            for i in 0..<4 {
                let diff = abs(Float(retrieved[i]) - Float(vector[i]))
                #expect(diff < 0.01)
            }
        }
    }

    @Test("Float16 serialization roundtrip")
    func testFloat16Serialization() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 64,
            maxElements: 100,
            metric: .l2
        )

        // Add some vectors
        for i in 0..<50 {
            let vector: [Float16] = (0..<64).map { _ in Float16.random(in: -1...1) }
            try index.add(vector, label: UInt64(i))
        }

        // Serialize
        let archive = try index.serializedArchive()
        #expect(archive.bytes.count > 0)

        // Load
        let loaded = try HNSWIndex<Float16>.restore(
            from: archive,
            dimensions: 64,
            metric: .l2
        )

        #expect(loaded.count == 50)
    }

    @Test("Type alias HNSWIndexF16 works")
    func testTypeAlias() throws {
        let index = try HNSWIndexF16(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        #expect(index.count == 1)
    }
}
