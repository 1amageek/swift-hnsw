import Testing
import Foundation
@testable import SwiftHNSW

@Suite("Benchmarks", .serialized, .enabled(if: ProcessInfo.processInfo.environment["BENCHMARK"] != nil))
struct Benchmarks {

    @Test("Float32 / Float16 / TurboQuant comparison")
    func fullComparison() throws {
        let n = 500
        let k = 10
        let numQueries = 50
        let efSearch = 100

        print("\n=== Full Benchmark (n=\(n), k=\(k), queries=\(numQueries), efSearch=\(efSearch)) ===\n")
        print("| Dim | Type | Build (s) | QPS | Recall@10 | Vec Mem | Compression |")
        print("| --- | --- | --- | --- | --- | --- | --- |")

        for d in [128, 768] {
            let vectors = (0..<n).map { _ in (0..<d).map { _ in Float.random(in: -1...1) } }
            let queries = (0..<numQueries).map { _ in (0..<d).map { _ in Float.random(in: -1...1) } }

            // Cosine ground truth (TurboQuant normalizes internally)
            let gtCosine = queries.map { q -> [UInt64] in
                let nq = q.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
                return vectors.enumerated().map { (i, v) in
                    let nv = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
                    var dot: Float = 0
                    for j in 0..<d { dot += q[j] * v[j] }
                    return (UInt64(i), 1.0 - dot / (nq * nv))
                }.sorted { $0.1 < $1.1 }.prefix(k).map { $0.0 }
            }

            // L2 ground truth (Float32/Float16)
            let gtL2 = queries.map { q -> [UInt64] in
                return vectors.enumerated().map { (i, v) in
                    var s: Float = 0
                    for j in 0..<d { let diff = q[j] - v[j]; s += diff * diff }
                    return (UInt64(i), s)
                }.sorted { $0.1 < $1.1 }.prefix(k).map { $0.0 }
            }

            // --- Float32 ---
            let f32 = try benchFloat32(vectors: vectors, queries: queries, gt: gtL2,
                                        d: d, n: n, k: k, efSearch: efSearch)
            print(f32)

            // --- Float16 ---
            let f16 = try benchFloat16(vectors: vectors, queries: queries, gt: gtL2,
                                        d: d, n: n, k: k, efSearch: efSearch)
            print(f16)

            // --- TurboQuant 4-bit ---
            let tq4 = try benchTurboQuant(vectors: vectors, queries: queries, gt: gtCosine,
                                           d: d, n: n, k: k, b: 4, efSearch: efSearch)
            print(tq4)

            // --- TurboQuant 2-bit ---
            let tq2 = try benchTurboQuant(vectors: vectors, queries: queries, gt: gtCosine,
                                           d: d, n: n, k: k, b: 2, efSearch: efSearch)
            print(tq2)
        }
    }

    // MARK: - Helpers

    private func benchFloat32(
        vectors: [[Float]], queries: [[Float]], gt: [[UInt64]],
        d: Int, n: Int, k: Int, efSearch: Int
    ) throws -> String {
        let t0 = CFAbsoluteTimeGetCurrent()
        let idx = try HNSWIndex<Float>(dimensions: d, maxElements: n, metric: .l2,
                                        configuration: HNSWConfiguration(m: 16, efConstruction: 100))
        for (i, v) in vectors.enumerated() { try idx.add(v, label: UInt64(i)) }
        let buildTime = CFAbsoluteTimeGetCurrent() - t0

        idx.setEfSearch(efSearch)
        _ = try idx.search(queries[0], k: k)
        let t1 = CFAbsoluteTimeGetCurrent()
        var results: [[SearchResult]] = []
        for q in queries { results.append(try idx.search(q, k: k)) }
        let searchTime = CFAbsoluteTimeGetCurrent() - t1

        let recall = avgRecall(results: results, gt: gt, k: k)
        let qps = Double(queries.count) / searchTime
        let mem = n * d * 4

        return "| \(d) | Float32 | \(f(buildTime)) | \(f0(qps)) | \(f1(recall*100))% | \(fmem(mem)) | 1.0x |"
    }

    private func benchFloat16(
        vectors: [[Float]], queries: [[Float]], gt: [[UInt64]],
        d: Int, n: Int, k: Int, efSearch: Int
    ) throws -> String {
        let vf16: [[Float16]] = vectors.map { $0.map { Float16($0) } }
        let qf16: [[Float16]] = queries.map { $0.map { Float16($0) } }

        let t0 = CFAbsoluteTimeGetCurrent()
        let idx = try HNSWIndex<Float16>(dimensions: d, maxElements: n, metric: .l2,
                                          configuration: HNSWConfiguration(m: 16, efConstruction: 100))
        for (i, v) in vf16.enumerated() { try idx.add(v, label: UInt64(i)) }
        let buildTime = CFAbsoluteTimeGetCurrent() - t0

        idx.setEfSearch(efSearch)
        _ = try idx.search(qf16[0], k: k)
        let t1 = CFAbsoluteTimeGetCurrent()
        var results: [[SearchResult]] = []
        for q in qf16 { results.append(try idx.search(q, k: k)) }
        let searchTime = CFAbsoluteTimeGetCurrent() - t1

        let recall = avgRecall(results: results, gt: gt, k: k)
        let qps = Double(queries.count) / searchTime
        let mem = n * d * 2

        return "| \(d) | Float16 | \(f(buildTime)) | \(f0(qps)) | \(f1(recall*100))% | \(fmem(mem)) | 2.0x |"
    }

    private func benchTurboQuant(
        vectors: [[Float]], queries: [[Float]], gt: [[UInt64]],
        d: Int, n: Int, k: Int, b: Int, efSearch: Int
    ) throws -> String {
        let t0 = CFAbsoluteTimeGetCurrent()
        let idx = try TurboQuantIndex(dimensions: d, maxElements: n, bitWidth: b,
                                       configuration: HNSWConfiguration(m: 16, efConstruction: 100), seed: 42)
        for (i, v) in vectors.enumerated() { try idx.add(v, label: UInt64(i)) }
        let buildTime = CFAbsoluteTimeGetCurrent() - t0

        idx.setEfSearch(efSearch)
        _ = try idx.search(queries[0], k: k)
        let t1 = CFAbsoluteTimeGetCurrent()
        var results: [[SearchResult]] = []
        for q in queries { results.append(try idx.search(q, k: k)) }
        let searchTime = CFAbsoluteTimeGetCurrent() - t1

        let recall = avgRecall(results: results, gt: gt, k: k)
        let qps = Double(queries.count) / searchTime
        let mem = n * idx.bytesPerVector
        let comp = idx.compressionRatio

        return "| \(d) | TQ-\(b)b | \(f(buildTime)) | \(f0(qps)) | \(f1(recall*100))% | \(fmem(mem)) | \(f1(comp))x |"
    }

    private func avgRecall(results: [[SearchResult]], gt: [[UInt64]], k: Int) -> Double {
        zip(results, gt).map { (res, truth) in
            Double(Set(res.prefix(k).map { $0.label }).intersection(Set(truth.prefix(k))).count) / Double(k)
        }.reduce(0, +) / Double(gt.count)
    }

    private func f(_ v: Double) -> String { String(format: "%.2f", v) }
    private func f0(_ v: Double) -> String { String(format: "%.0f", v) }
    private func f1(_ v: Double) -> String { String(format: "%.1f", v) }
    private func f1(_ v: Float) -> String { String(format: "%.1f", v) }
    private func fmem(_ bytes: Int) -> String {
        bytes >= 1024*1024 ? String(format: "%.1f MB", Double(bytes)/(1024*1024))
                           : String(format: "%.0f KB", Double(bytes)/1024)
    }
}
