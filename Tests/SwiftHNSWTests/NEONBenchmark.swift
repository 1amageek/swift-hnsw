import Testing
import Foundation
@testable import SwiftHNSW

@Suite("NEON Benchmark", .serialized)
struct NEONBenchmark {

    @Test("Float32 L2 + IP + Cosine across dimensions")
    func float32Benchmark() throws {
        let n = 500
        let k = 10
        let numQueries = 50
        let efSearch = 100

        print("\n=== NEON Benchmark (n=\(n), k=\(k), queries=\(numQueries), efSearch=\(efSearch)) ===\n")
        print("| Dim | Metric | Build (s) | Vec/s | QPS | Latency (ms) |")
        print("| --- | --- | --- | --- | --- | --- |")

        for d in [128, 768] {
            for metric in [DistanceMetric.l2, .innerProduct, .cosine] {
                let metricName: String
                switch metric {
                case .l2: metricName = "L2"
                case .innerProduct: metricName = "IP"
                case .cosine: metricName = "Cosine"
                }

                let vectors = (0..<n).map { _ in (0..<d).map { _ in Float.random(in: -1...1) } }
                let queries = (0..<numQueries).map { _ in (0..<d).map { _ in Float.random(in: -1...1) } }

                let buildStart = CFAbsoluteTimeGetCurrent()
                let index = try HNSWIndex<Float>(
                    dimensions: d, maxElements: n, metric: metric,
                    configuration: HNSWConfiguration(m: 16, efConstruction: 100)
                )
                for (i, v) in vectors.enumerated() { try index.add(v, label: UInt64(i)) }
                let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

                index.setEfSearch(efSearch)
                _ = try index.search(queries[0], k: k)

                let searchStart = CFAbsoluteTimeGetCurrent()
                for q in queries { _ = try index.search(q, k: k) }
                let searchTime = CFAbsoluteTimeGetCurrent() - searchStart

                let qps = Double(numQueries) / searchTime
                let latency = (searchTime / Double(numQueries)) * 1000
                let vecPerSec = Double(n) / buildTime

                print("| \(d) | \(metricName) | \(String(format: "%.2f", buildTime)) | \(String(format: "%.0f", vecPerSec)) | \(String(format: "%.0f", qps)) | \(String(format: "%.3f", latency)) |")
            }
        }
    }
}
