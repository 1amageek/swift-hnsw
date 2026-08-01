import SwiftHNSW

@main
enum TurboQuantConcurrencyCheck {
    static func main() async throws {
        let dimensions = 64
        let vectorCount = 128
        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: vectorCount,
            bitWidth: 4,
            configuration: HNSWConfiguration(
                m: 8,
                efConstruction: 48,
                efSearch: 40
            ),
            seed: 42
        )
        for label in 0..<vectorCount {
            try index.add(
                makeVector(index: label, dimensions: dimensions),
                label: UInt64(label)
            )
        }
        try index.finalize()
        let query = makeVector(index: 10_000, dimensions: dimensions)
        let expected = try index.search(query, k: 10)

        let checksums = try await withThrowingTaskGroup(
            of: UInt64.self,
            returning: [UInt64].self
        ) { group in
            for _ in 0..<8 {
                group.addTask {
                    var checksum: UInt64 = 0
                    for _ in 0..<100 {
                        let results = try index.search(query, k: 10)
                        precondition(
                            results.map { $0.label } == expected.map { $0.label },
                            "Concurrent search returned different labels"
                        )
                        for result in results {
                            checksum &+= result.label
                            checksum &+= UInt64(result.distance.bitPattern)
                        }
                    }
                    return checksum
                }
            }
            var values = [UInt64]()
            values.reserveCapacity(8)
            for try await checksum in group {
                values.append(checksum)
            }
            return values
        }
        precondition(
            checksums.allSatisfy { $0 == checksums[0] },
            "Concurrent search checksums differ"
        )
        print("turboquant_concurrency_check=passed checksum=\(checksums[0])")
    }

    private static func makeVector(index: Int, dimensions: Int) -> [Float] {
        (0..<dimensions).map { coordinate in
            Float(((index + 3) * (coordinate + 5)) % 97 - 48) / 49
        }
    }
}
