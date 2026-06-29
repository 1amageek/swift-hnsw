import Foundation
import Testing
@testable import SwiftHNSW

@Suite("Swift HNSW Backend", .serialized)
struct SwiftBackendHNSWTests {

    @Test("Rejects capacities outside the internal id range")
    func rejectsCapacitiesOutsideInternalIDRange() throws {
        #expect(throws: HNSWError.self) {
            _ = try HNSWIndex<Float>(
                dimensions: 1,
                maxElements: Int(UInt32.max) + 1,
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

        let data = try index.serialize()
        let loaded = try HNSWIndex<Float>.load(from: data, dimensions: 2, metric: .l2)

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

    @Test("Float16 roundtrip uses the same backend semantics")
    func float16Roundtrip() throws {
        let index = try HNSWIndex<Float16>(dimensions: 2, maxElements: 4, metric: .cosine)
        try index.add([1, 0], label: 1)
        try index.add([0, 1], label: 2)

        let storageCounts = index.swiftBackendStorageCounts
        #expect(storageCounts.float == 0)
        #expect(storageCounts.half == 4)

        let data = try index.serialize()
        let loaded = try HNSWIndex<Float16>.load(from: data, dimensions: 2, metric: .cosine)
        let loadedStorageCounts = loaded.swiftBackendStorageCounts

        #expect(loadedStorageCounts.float == 0)
        #expect(loadedStorageCounts.half == 4)

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
            _ = try HNSWIndex<Float>.load(from: payload, dimensions: 1, metric: .l2)
        }
    }

    @Test("Graph loader rejects invalid neighbor id")
    func graphLoaderRejectsInvalidNeighborID() throws {
        let payload = invalidGraphPayloadWithInvalidNeighborID()

        #expect(throws: HNSWError.self) {
            _ = try HNSWIndex<Float>.load(from: payload, dimensions: 1, metric: .l2)
        }
    }
}

@Suite("Swift TurboQuant Backend", .serialized)
struct SwiftBackendTurboQuantTests {

    @Test("Exact cosine ordering and deterministic tie break")
    func exactCosineOrderingAndTieBreak() throws {
        let index = try TurboQuantIndex(dimensions: 2, maxElements: 4, bitWidth: 4)
        try index.add([1, 0], label: 20)
        try index.add([0, 1], label: 30)
        try index.add([2, 0], label: 10)

        let results = try index.search([4, 0], k: 3)

        #expect(results.map(\.label) == [10, 20, 30])
        #expect(results[0].distance == 0)
        #expect(results[1].distance == 0)
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

        let path = temporaryPath(name: "swift_backend_tq")
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

    @Test("Invalid serialized payload throws typed load error")
    func invalidPayloadThrows() throws {
        let path = temporaryPath(name: "swift_backend_tq_invalid")
        defer { removeFileIfPresent(path) }
        try Data([0, 1, 2, 3]).write(to: URL(fileURLWithPath: path))

        #expect(throws: HNSWError.self) {
            _ = try TurboQuantIndex.load(from: path)
        }
    }
}

@Suite(
    "Swift Backend Performance",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWIFT_BACKEND_PERF"] != nil)
)
struct SwiftBackendPerformanceTests {

    @Test("Flat HNSW backend performance smoke")
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
        print("Swift HNSW backend: build=\(vectorsPerSecond) vectors/s search=\(queriesPerSecond) qps")

        #expect(vectorsPerSecond > 10_000)
        #expect(queriesPerSecond > 100)
    }

    @Test("TurboQuant backend performance smoke")
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
        print("Swift TurboQuant backend: build=\(vectorsPerSecond) vectors/s search=\(queriesPerSecond) qps")

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

private func invalidGraphPayloadWithSelfEdge() -> Data {
    var writer = TestGraphPayloadWriter()
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
    var writer = TestGraphPayloadWriter()
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

private struct TestGraphPayloadWriter {
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
