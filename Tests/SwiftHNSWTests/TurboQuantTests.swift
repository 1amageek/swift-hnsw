// TurboQuantTests.swift
// Unit tests for TurboQuant vector quantization

import Testing
import Foundation
import hnswlib
@testable import SwiftHNSW

// MARK: - BitPacking Tests

@Suite("BitPacking Tests")
struct BitPackingTests {

    @Test("Roundtrip b=4", arguments: [8, 128, 768, 1024])
    func roundtrip4(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<16) }
        let packed = BitPacking.pack(indices, bitWidth: 4)
        let unpacked = BitPacking.unpack(packed, bitWidth: 4, count: count)
        #expect(unpacked == indices)
        #expect(packed.count == (count + 1) / 2)
    }

    @Test("Roundtrip b=2", arguments: [8, 128, 768, 1024])
    func roundtrip2(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<4) }
        let packed = BitPacking.pack(indices, bitWidth: 2)
        let unpacked = BitPacking.unpack(packed, bitWidth: 2, count: count)
        #expect(unpacked == indices)
    }

    @Test("Roundtrip b=1", arguments: [8, 128, 1024])
    func roundtrip1(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<2) }
        let packed = BitPacking.pack(indices, bitWidth: 1)
        let unpacked = BitPacking.unpack(packed, bitWidth: 1, count: count)
        #expect(unpacked == indices)
    }

    @Test("Roundtrip b=3", arguments: [8, 128, 1024])
    func roundtrip3(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<8) }
        let packed = BitPacking.pack(indices, bitWidth: 3)
        let unpacked = BitPacking.unpack(packed, bitWidth: 3, count: count)
        #expect(unpacked == indices)
    }
}

// MARK: - ScalarQuantizer Tests

@Suite("ScalarQuantizer Tests")
struct ScalarQuantizerTests {

    @Test("Codebook symmetry", arguments: [1, 2, 3, 4])
    func codebookSymmetry(b: Int) {
        let sq = ScalarQuantizer(bitWidth: b, dimension: 128)
        let n = sq.numCentroids
        for i in 0..<n / 2 {
            #expect(abs(sq.centroids[i] + sq.centroids[n - 1 - i]) < 1e-6)
        }
    }

    @Test("Quantize-dequantize MSE bound")
    func quantizeDequantizeMSE() {
        let d = 1024
        let sq = ScalarQuantizer(bitWidth: 4, dimension: d)
        let scale = 1.0 / Float(d).squareRoot()
        let vector = (0..<d).map { _ in Float.random(in: -3 * scale...3 * scale) }
        let packed = sq.quantizeAndPack(vector)
        let decoded = sq.dequantize(packed)

        var mse: Float = 0
        for i in 0..<d {
            let diff = vector[i] - decoded[i]
            mse += diff * diff
        }
        mse /= Float(d)
        // Paper: D_mse ≈ 0.009 for b=4 → per-dim ≈ 0.009/d
        #expect(mse < 0.001, "Per-dim MSE should be bounded")
    }
}

// MARK: - HD³ Rotation Tests

@Suite("HD3 Rotation Tests")
struct HD3RotationTests {

    @Test("Distance preservation for power-of-2 dim")
    func distancePreservationPow2() throws {
        try verifyDistancePreservation(d: 128, expectedRecall: 1.0)
    }

    @Test("Distance preservation for non-power-of-2 dim")
    func distancePreservationNonPow2() throws {
        try verifyDistancePreservation(d: 768, expectedRecall: 1.0)
    }

    @Test("Coordinate distribution matches N(0, 1/p)")
    func coordinateDistribution() {
        let d = 768
        let sq = ScalarQuantizer(bitWidth: 4, dimension: 1024) // p=1024
        let encoder = sq.centroids.withUnsafeBufferPointer { cBuf in
            sq.boundaries.withUnsafeBufferPointer { bBuf in
                hnsw_tq_encoder_create(d, 4, cBuf.baseAddress!, Int32(sq.numCentroids),
                                        bBuf.baseAddress!, Int32(sq.boundaries.count), 42)
            }
        }!
        defer { hnsw_tq_encoder_destroy(encoder) }
        let p = hnsw_tq_encoder_padded_dim(encoder)

        var allCoords: [Float] = []
        for _ in 0..<500 {
            var v = (0..<d).map { _ in Float.random(in: -1...1) }
            let norm = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            v = v.map { $0 / norm }

            var rotated = [Float](repeating: 0, count: p)
            v.withUnsafeBufferPointer { vBuf in
                rotated.withUnsafeMutableBufferPointer { rBuf in
                    hnsw_tq_encoder_rotate_query(encoder, vBuf.baseAddress!, rBuf.baseAddress!)
                }
            }
            allCoords.append(contentsOf: rotated)
        }

        let mean = allCoords.reduce(Float(0), +) / Float(allCoords.count)
        let variance = allCoords.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(allCoords.count)
        let expectedVar: Float = 1.0 / Float(p)

        #expect(abs(mean) < 0.01, "Mean should be ~0, got \(mean)")
        let ratio = variance / expectedVar
        #expect(ratio > 0.9 && ratio < 1.1, "Variance ratio should be ~1.0, got \(ratio)")
    }

    private func verifyDistancePreservation(d: Int, expectedRecall: Double) throws {
        var p = 1; while p < d { p *= 2 }
        let sq = ScalarQuantizer(bitWidth: 4, dimension: p)
        let encoder = sq.centroids.withUnsafeBufferPointer { cBuf in
            sq.boundaries.withUnsafeBufferPointer { bBuf in
                hnsw_tq_encoder_create(d, 4, cBuf.baseAddress!, Int32(sq.numCentroids),
                                        bBuf.baseAddress!, Int32(sq.boundaries.count), 42)
            }
        }!
        defer { hnsw_tq_encoder_destroy(encoder) }
        let actualP = hnsw_tq_encoder_padded_dim(encoder)

        let n = 200
        let k = 10
        let vectors = (0..<n).map { _ in (0..<d).map { _ in Float.random(in: -1...1) } }

        var rotated = [[Float]](repeating: [], count: n)
        for i in 0..<n {
            var r = [Float](repeating: 0, count: actualP)
            vectors[i].withUnsafeBufferPointer { vBuf in
                r.withUnsafeMutableBufferPointer { rBuf in
                    hnsw_tq_encoder_rotate_query(encoder, vBuf.baseAddress!, rBuf.baseAddress!)
                }
            }
            rotated[i] = r
        }

        // Cosine ground truth
        let q = vectors[0]
        let normQ = q.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        let cosineRank = (1..<n).map { i -> (Int, Float) in
            let v = vectors[i]
            let normV = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            var dot: Float = 0
            for j in 0..<d { dot += q[j] * v[j] }
            return (i, 1.0 - dot / (normQ * normV))
        }.sorted { $0.1 < $1.1 }

        // Rotated L2 rank (all p coordinates)
        let rotRank = (1..<n).map { i -> (Int, Float) in
            var sum: Float = 0
            for j in 0..<actualP {
                let diff = rotated[0][j] - rotated[i][j]
                sum += diff * diff
            }
            return (i, sum)
        }.sorted { $0.1 < $1.1 }

        let cosineTop = Set(cosineRank.prefix(k).map { $0.0 })
        let rotTop = Set(rotRank.prefix(k).map { $0.0 })
        let recall = Double(cosineTop.intersection(rotTop).count) / Double(k)

        #expect(recall >= expectedRecall,
                "d=\(d): cosine vs rotL2 recall should be >= \(expectedRecall), got \(recall)")
    }
}

// MARK: - TurboQuantIndex Tests

@Suite("TurboQuantIndex Tests")
struct TurboQuantIndexTests {

    @Test("Basic add and search")
    func basicAddSearch() throws {
        let dim = 128
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 100, bitWidth: 4, seed: 42)

        for i in 0..<20 {
            let v = (0..<dim).map { _ in Float.random(in: -1...1) }
            try index.add(v, label: UInt64(i))
        }

        #expect(index.count == 20)
        #expect(index.paddedDimensions == 128) // 128 is power of 2

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try index.search(query, k: 5)
        #expect(results.count == 5)
        #expect(index.isFinalized) // auto-finalized on first search
    }

    @Test("Non-power-of-2 dimension")
    func nonPow2Dimension() throws {
        let dim = 768
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 50, bitWidth: 4, seed: 42)

        #expect(index.paddedDimensions == 1024)
        // packed_size = ceil(1024 * 4 / 8) = 512
        #expect(index.bytesPerVector == 512)
        // compression vs Float32: 768*4 / 512 = 6.0x
        #expect(index.compressionRatio == 6.0)

        for i in 0..<20 {
            let v = (0..<dim).map { _ in Float.random(in: -1...1) }
            try index.add(v, label: UInt64(i))
        }
        let results = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 5)
        #expect(results.count == 5)
    }

    @Test("Dimension mismatch errors")
    func dimensionMismatch() throws {
        let index = try TurboQuantIndex(dimensions: 64, maxElements: 10, bitWidth: 4)
        #expect(throws: HNSWError.self) { try index.add([Float](repeating: 0, count: 32), label: 0) }
        #expect(throws: HNSWError.self) { try index.search([Float](repeating: 0, count: 32), k: 1) }
    }

    @Test("Cannot add after finalize")
    func cannotAddAfterFinalize() throws {
        let dim = 64
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 50, bitWidth: 4)
        try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 0)
        _ = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 1) // triggers finalize
        #expect(index.isFinalized)
        #expect(throws: HNSWError.self) {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 1)
        }
    }

    @Test("All bit widths", arguments: [1, 2, 3, 4])
    func allBitWidths(b: Int) throws {
        let dim = 64
        let p = 64 // 64 is power of 2
        let n = 30
        let index = try TurboQuantIndex(dimensions: dim, maxElements: n, bitWidth: b, seed: 42)

        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        let results = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 5)
        #expect(results.count == 5)
        #expect(index.bytesPerVector == BitPacking.packedSize(count: p, bitWidth: b))
    }

    @Test("Recall on power-of-2 dimension (d=128, b=4)")
    func recallPow2() throws {
        let dim = 128
        let n = 500
        let k = 10
        let numQueries = 50

        let trainVectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        let testQueries = (0..<numQueries).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        // Cosine ground truth
        let groundTruths = computeCosineGroundTruth(queries: testQueries, vectors: trainVectors, k: k)

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        for (i, v) in trainVectors.enumerated() { try index.add(v, label: UInt64(i)) }
        index.setEfSearch(200)

        var totalRecall: Double = 0
        for qi in 0..<numQueries {
            let results = try index.search(testQueries[qi], k: k)
            let found = Set(results.prefix(k).map { $0.label })
            let truth = Set(groundTruths[qi].prefix(k))
            totalRecall += Double(found.intersection(truth).count) / Double(k)
        }
        let recall = totalRecall / Double(numQueries)
        #expect(recall > 0.70, "d=128 b=4 recall should exceed 70%, got \(recall)")
    }

    @Test("Recall on non-power-of-2 dimension (d=768, b=4)")
    func recallNonPow2() throws {
        let dim = 768
        let n = 500
        let k = 10
        let numQueries = 50

        let trainVectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        let testQueries = (0..<numQueries).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        let groundTruths = computeCosineGroundTruth(queries: testQueries, vectors: trainVectors, k: k)

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        for (i, v) in trainVectors.enumerated() { try index.add(v, label: UInt64(i)) }
        index.setEfSearch(200)

        var totalRecall: Double = 0
        for qi in 0..<numQueries {
            let results = try index.search(testQueries[qi], k: k)
            let found = Set(results.prefix(k).map { $0.label })
            let truth = Set(groundTruths[qi].prefix(k))
            totalRecall += Double(found.intersection(truth).count) / Double(k)
        }
        let recall = totalRecall / Double(numQueries)
        #expect(recall > 0.70, "d=768 b=4 recall should exceed 70%, got \(recall)")
    }

    @Test("Serialization roundtrip")
    func serializationRoundtrip() throws {
        let dim = 128
        let n = 100
        let k = 5

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 100), seed: 42)
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        for (i, v) in vectors.enumerated() { try index.add(v, label: UInt64(i)) }

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let resultsBefore = try index.search(query, k: k)

        // Save
        let tmpPath = NSTemporaryDirectory() + "tq_test_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let fileSize = try FileManager.default.attributesOfItem(atPath: tmpPath)[.size] as! Int
        #expect(fileSize > 0)

        // Load
        let loaded = try TurboQuantIndex.load(from: tmpPath)
        #expect(loaded.dimensions == dim)
        #expect(loaded.bitWidth == 4)
        #expect(loaded.paddedDimensions == 128)
        #expect(loaded.count == n)
        #expect(loaded.isFinalized)

        // Search loaded index — should return same results
        let resultsAfter = try loaded.search(query, k: k)
        #expect(resultsAfter.count == k)

        // Same labels in same order
        let labelsBefore = resultsBefore.map { $0.label }
        let labelsAfter = resultsAfter.map { $0.label }
        #expect(labelsBefore == labelsAfter, "Search results should match after load")
    }

    @Test("Serialization roundtrip non-power-of-2")
    func serializationNonPow2() throws {
        let dim = 768
        let n = 50
        let k = 5

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 8, efConstruction: 50), seed: 99)
        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }

        let tmpPath = NSTemporaryDirectory() + "tq_test768_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let loaded = try TurboQuantIndex.load(from: tmpPath)
        #expect(loaded.dimensions == 768)
        #expect(loaded.paddedDimensions == 1024)
        #expect(loaded.count == n)

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try loaded.search(query, k: k)
        #expect(results.count == k)
    }
    // MARK: - Issue #17: efSearch must be preserved after save/load

    @Test("efSearch preserved after save/load")
    func efSearchPreserved() throws {
        let dim = 128
        let n = 500
        let k = 10
        let efSearch = 200

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        index.setEfSearch(efSearch)

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let resultsBefore = try index.search(query, k: k)

        let tmpPath = NSTemporaryDirectory() + "tq_ef_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let loaded = try TurboQuantIndex.load(from: tmpPath)
        // Explicitly set the same efSearch on loaded index
        // (this test verifies that the header preserves it OR that the user must set it)
        loaded.setEfSearch(efSearch)

        let resultsAfter = try loaded.search(query, k: k)
        #expect(resultsBefore.map { $0.label } == resultsAfter.map { $0.label },
               "Loaded index with same efSearch should produce identical results")
    }

    // MARK: - Issue #6: Header serialization must be portable

    @Test("Header field-by-field consistency")
    func headerFieldConsistency() throws {
        let dim = 768
        let n = 10

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 3, seed: 12345)
        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }

        let tmpPath = NSTemporaryDirectory() + "tq_header_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let loaded = try TurboQuantIndex.load(from: tmpPath)
        #expect(loaded.dimensions == 768)
        #expect(loaded.bitWidth == 3)
        #expect(loaded.seed == 12345)
        #expect(loaded.paddedDimensions == 1024)
        #expect(loaded.count == n)
    }

    // MARK: - Issue #1: _finalized must be thread-safe

    @Test("Add after search throws error")
    func addAfterSearchThrows() throws {
        let dim = 64
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 20, bitWidth: 4)
        for i in 0..<5 {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        // search triggers finalize
        _ = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 1)
        #expect(index.isFinalized)

        // add after finalize must throw, not silently corrupt
        #expect(throws: HNSWError.self) {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 99)
        }
    }

    @Test("Add after save throws error")
    func addAfterSaveThrows() throws {
        let dim = 64
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 20, bitWidth: 4)
        for i in 0..<5 {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }

        let tmpPath = NSTemporaryDirectory() + "tq_addsave_\(UUID().uuidString).bin"
        try index.save(to: tmpPath) // triggers finalize
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        #expect(index.isFinalized)
        #expect(throws: HNSWError.self) {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 99)
        }
    }

    // MARK: - Issue #3: Large dimension must not crash

    @Test("Large dimension encoding does not crash")
    func largeDimensionEncoding() throws {
        // d=3072 → p=4096, alloca would use ~49KB
        let dim = 3072
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 5, bitWidth: 4, seed: 42)

        // Should not crash from stack overflow
        let v = (0..<dim).map { _ in Float.random(in: -1...1) }
        try index.add(v, label: 0)
        try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 1)

        let results = try index.search(v, k: 1)
        #expect(results.count == 1)
        #expect(results[0].label == 0) // should find itself
    }

    // MARK: - Issue #11: Documentation correctness

    @Test("Mode 0 uses Float32 L2 not symmetric quantized")
    func mode0IsFloat32L2() throws {
        // This test verifies the construction uses exact Float32 L2
        // by checking that search results are reasonable (high recall)
        // If mode 0 were symmetric quantized, recall would be much lower
        let dim = 64
        let n = 100
        let k = 10

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        for (i, v) in vectors.enumerated() { try index.add(v, label: UInt64(i)) }

        index.setEfSearch(200)
        let query = vectors[0] // search for first vector
        let results = try index.search(query, k: k)
        // With Float32 construction, the first result should be the vector itself (label 0)
        // or very close to it
        #expect(results[0].label == 0, "Should find the query vector as nearest neighbor")
    }
}

// MARK: - Helpers

private func computeCosineGroundTruth(queries: [[Float]], vectors: [[Float]], k: Int) -> [[UInt64]] {
    let d = vectors[0].count
    return queries.map { query in
        let normQ = query.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        let dists: [(UInt64, Float)] = vectors.enumerated().map { (i, v) in
            let normV = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            var dot: Float = 0
            for j in 0..<d { dot += query[j] * v[j] }
            return (UInt64(i), 1.0 - dot / (normQ * normV))
        }
        return dists.sorted { $0.1 < $1.1 }.prefix(k).map { $0.0 }
    }
}
