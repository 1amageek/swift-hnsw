import Foundation
import SwiftHNSW
import Testing

@Suite("Index performance benchmarks", .serialized)
struct IndexPerformanceBenchmarks {
    @Test("Flat HNSW index performance smoke")
    func hnswFlatSearchPerformanceSmoke() throws {
        let dimensions = 64
        let count = 2_000
        let queryCount = 20
        let index = try HNSWIndex<Float>(
            dimensions: dimensions,
            maxElements: count,
            metric: .l2
        )
        let vectors = deterministicVectors(
            count: count,
            dimensions: dimensions
        )
        let queries = deterministicVectors(
            count: queryCount,
            dimensions: dimensions
        )

        let buildSeconds = try measureSeconds {
            for (offset, vector) in vectors.enumerated() {
                try index.add(vector, label: UInt64(offset))
            }
        }
        let searchSeconds = try measureSeconds {
            for query in queries {
                #expect(try index.search(query, k: 10).count == 10)
            }
        }

        let vectorsPerSecond = Double(count)
            / max(buildSeconds, .leastNonzeroMagnitude)
        let queriesPerSecond = Double(queryCount)
            / max(searchSeconds, .leastNonzeroMagnitude)
        print(
            "HNSW index: build=\(vectorsPerSecond) vectors/s "
                + "search=\(queriesPerSecond) qps"
        )
        #expect(vectorsPerSecond > 10_000)
        #expect(queriesPerSecond > 100)
    }

    @Test("TurboQuant index performance smoke")
    func turboQuantSearchPerformanceSmoke() throws {
        let dimensions = 64
        let count = 2_000
        let queryCount = 20
        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: count,
            bitWidth: 4
        )
        let vectors = deterministicVectors(
            count: count,
            dimensions: dimensions
        )
        let queries = deterministicVectors(
            count: queryCount,
            dimensions: dimensions
        )

        let buildSeconds = try measureSeconds {
            for (offset, vector) in vectors.enumerated() {
                try index.add(vector, label: UInt64(offset))
            }
        }
        let searchSeconds = try measureSeconds {
            for query in queries {
                #expect(try index.search(query, k: 10).count == 10)
            }
        }

        let vectorsPerSecond = Double(count)
            / max(buildSeconds, .leastNonzeroMagnitude)
        let queriesPerSecond = Double(queryCount)
            / max(searchSeconds, .leastNonzeroMagnitude)
        print(
            "TurboQuant index: build=\(vectorsPerSecond) vectors/s "
                + "search=\(queriesPerSecond) qps"
        )
        #expect(vectorsPerSecond > 10_000)
        #expect(queriesPerSecond > 100)
    }

    @Test("Float16 HNSW index performance smoke")
    func float16SearchPerformanceSmoke() throws {
        let dimensions = 128
        let count = 1_000
        let index = try HNSWIndex<Float16>(
            dimensions: dimensions,
            maxElements: count,
            metric: .l2,
            configuration: HNSWConfiguration(
                m: 16,
                efConstruction: 200,
                efSearch: 50
            )
        )
        let vectors: [[Float16]] = deterministicVectors(
            count: count,
            dimensions: dimensions
        ).map { $0.map(Float16.init) }
        let query: [Float16] = deterministicVectors(
            count: 1,
            dimensions: dimensions
        )[0].map(Float16.init)

        let buildSeconds = try measureSeconds {
            for (offset, vector) in vectors.enumerated() {
                try index.add(vector, label: UInt64(offset))
            }
        }
        let searchSeconds = try measureSeconds {
            #expect(try index.search(query, k: 10).count == 10)
        }

        print(
            "Float16 HNSW index: build=\(buildSeconds)s "
                + "search=\(searchSeconds)s"
        )
    }

    private func deterministicVectors(
        count: Int,
        dimensions: Int
    ) -> [[Float]] {
        (0..<count).map { row in
            (0..<dimensions).map { column in
                let value = ((row * 31 + column * 17) % 97) - 48
                return Float(value) / 48
            }
        }
    }

    private func measureSeconds(
        _ body: () throws -> Void
    ) rethrows -> Double {
        let clock = ContinuousClock()
        let start = clock.now
        try body()
        let components = start.duration(to: clock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds)
                / 1_000_000_000_000_000_000
    }
}
