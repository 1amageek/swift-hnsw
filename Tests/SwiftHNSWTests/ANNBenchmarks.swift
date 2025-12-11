// ANNBenchmarks.swift
// ANN-Benchmarks standard benchmarks for SwiftHNSW
// Based on https://ann-benchmarks.com methodology
//
// Run all:     swift test --filter ANNBenchmarks
// Run HNSW:    swift test --filter ANNBenchmarks/benchmark

import Testing
import Foundation
@testable import SwiftHNSW

// MARK: - Shared Utilities

/// Generate random vectors
func randomVectors(count: Int, dimension: Int) -> [[Float]] {
    (0..<count).map { _ in
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }
}

/// Generate random vector
func randomVector(dimension: Int) -> [Float] {
    (0..<dimension).map { _ in Float.random(in: -1...1) }
}

/// Brute force search for ground truth
func bruteForceSearch(query: [Float], vectors: [[Float]], k: Int) -> [UInt64] {
    let distances: [(UInt64, Float)] = vectors.enumerated().map { (index, vector) in
        let dist = l2Distance(query, vector)
        return (UInt64(index), dist)
    }
    return distances.sorted { $0.1 < $1.1 }.prefix(k).map { $0.0 }
}

/// L2 distance calculation
func l2Distance(_ a: [Float], _ b: [Float]) -> Float {
    var sum: Float = 0
    for i in 0..<a.count {
        let diff = a[i] - b[i]
        sum += diff * diff
    }
    return sum
}

/// Calculate Recall@k: fraction of true nearest neighbors found
func calculateRecall(result: [SearchResult], groundTruth: [UInt64], k: Int) -> Double {
    let resultSet = Set(result.prefix(k).map { $0.label })
    let truthSet = Set(groundTruth.prefix(k))
    let intersection = resultSet.intersection(truthSet)
    return Double(intersection.count) / Double(k)
}

/// Calculate average Recall@k over multiple queries
func calculateAverageRecall(results: [[SearchResult]], groundTruths: [[UInt64]], k: Int) -> Double {
    guard results.count == groundTruths.count else { return 0.0 }
    let recalls = zip(results, groundTruths).map { calculateRecall(result: $0, groundTruth: $1, k: k) }
    return recalls.reduce(0.0, +) / Double(recalls.count)
}

/// Benchmark result for a single configuration
struct BenchmarkPoint {
    let algorithm: String
    let parameters: String
    let recall: Double
    let qps: Double
    let latencyMs: Double
    let buildTime: Double
    let indexSizeKB: Double
}

// MARK: - String Formatting Helpers

func pad(_ s: String, _ width: Int, left: Bool = false) -> String {
    if s.count >= width { return String(s.prefix(width)) }
    let padding = String(repeating: " ", count: width - s.count)
    return left ? padding + s : s + padding
}

func formatDouble(_ value: Double, decimals: Int) -> String {
    String(format: "%.\(decimals)f", value)
}

func formatSize(_ bytes: Int) -> String {
    if bytes >= 1024 * 1024 {
        return formatDouble(Double(bytes) / (1024 * 1024), decimals: 1) + " MB"
    } else {
        return formatDouble(Double(bytes) / 1024, decimals: 0) + " KB"
    }
}

/// Compute ground truth using brute force
func computeGroundTruth(
    trainVectors: [[Float]],
    testQueries: [[Float]],
    k: Int
) -> [[UInt64]] {
    testQueries.map { query in
        bruteForceSearch(query: query, vectors: trainVectors, k: k)
    }
}

/// Find Pareto frontier (best recall for given QPS)
func findParetoFrontier(_ points: [BenchmarkPoint]) -> [BenchmarkPoint] {
    var pareto: [BenchmarkPoint] = []
    let sorted = points.sorted { $0.qps > $1.qps }
    var maxRecall = -1.0

    for point in sorted {
        if point.recall > maxRecall {
            pareto.append(point)
            maxRecall = point.recall
        }
    }

    return pareto
}

// MARK: - ANN Benchmarks Suite

@Suite("ANN Benchmarks", .serialized)
struct ANNBenchmarks {

    // MARK: - HNSW Recall vs QPS Benchmark

    @Test("HNSW Recall vs QPS")
    func benchmark() throws {
        let n = 10_000
        let d = 128
        let k = 10
        let queries = 100

        print("\n")
        print(String(repeating: "=", count: 85))
        print("HNSW Benchmark: Recall vs QPS")
        print("n=\(n), d=\(d), k=\(k), queries=\(queries)")
        print(String(repeating: "=", count: 85))

        print("\n[1/3] Generating data...")
        let trainVectors = randomVectors(count: n, dimension: d)
        let testQueries = randomVectors(count: queries, dimension: d)

        print("[2/3] Computing ground truth (brute force)...")
        let gtStart = CFAbsoluteTimeGetCurrent()
        let groundTruths = computeGroundTruth(
            trainVectors: trainVectors,
            testQueries: testQueries,
            k: k
        )
        print("   Completed in \(formatDouble(CFAbsoluteTimeGetCurrent() - gtStart, decimals: 2))s")

        print("[3/3] Running benchmarks...\n")

        let efSearchValues = [10, 20, 40, 80, 160, 320]
        var results: [BenchmarkPoint] = []

        // Build index
        print("Building index (M=16, efConstruction=200)...")
        let buildStart = CFAbsoluteTimeGetCurrent()
        let index = try HNSWIndex(
            dimensions: d,
            maxElements: n,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )
        for (i, vector) in trainVectors.enumerated() {
            try index.add(vector, label: UInt64(i))
        }
        let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

        // Calculate index size: vectors + graph
        let vectorsSize = n * d * MemoryLayout<Float>.size
        let graphSize = n * 16 * 2 * MemoryLayout<Int>.size  // M=16, bidirectional
        let indexSize = vectorsSize + graphSize

        print("Build time: \(formatDouble(buildTime, decimals: 2))s (\(formatDouble(Double(n) / buildTime, decimals: 0)) vec/s)\n")

        print(String(repeating: "-", count: 85))
        print("| \(pad("efSearch", 10)) | \(pad("Recall@10", 10)) | \(pad("QPS", 10)) | \(pad("Latency", 12)) | \(pad("Build(s)", 10)) | \(pad("Index Size", 12)) |")
        print(String(repeating: "-", count: 85))

        for efSearch in efSearchValues {
            index.setEfSearch(efSearch)

            // Warmup
            _ = try index.search(testQueries[0], k: k)

            // Benchmark
            let searchStart = CFAbsoluteTimeGetCurrent()
            var searchResults: [[SearchResult]] = []
            for query in testQueries {
                searchResults.append(try index.search(query, k: k))
            }
            let searchTime = CFAbsoluteTimeGetCurrent() - searchStart

            let qps = Double(queries) / searchTime
            let latencyMs = (searchTime / Double(queries)) * 1000
            let recall = calculateAverageRecall(results: searchResults, groundTruths: groundTruths, k: k)

            print("| \(pad(String(efSearch), 10)) | \(pad(formatDouble(recall * 100, decimals: 1) + "%", 10)) | \(pad(formatDouble(qps, decimals: 0), 10)) | \(pad(formatDouble(latencyMs, decimals: 3) + "ms", 12)) | \(pad(formatDouble(buildTime, decimals: 2), 10)) | \(pad(formatSize(indexSize), 12)) |")

            results.append(BenchmarkPoint(
                algorithm: "HNSW",
                parameters: "ef=\(efSearch)",
                recall: recall,
                qps: qps,
                latencyMs: latencyMs,
                buildTime: buildTime,
                indexSizeKB: Double(indexSize) / 1024
            ))
        }

        print(String(repeating: "-", count: 85))

        // Summary
        print("\n[Summary]")
        print("  Index Size: \(formatSize(indexSize)) (vectors: \(formatSize(vectorsSize)), graph: \(formatSize(graphSize)))")
        print("  Build Time: \(formatDouble(buildTime, decimals: 2))s (\(formatDouble(Double(n) / buildTime, decimals: 0)) vec/s)")

        print("\n[Pareto Frontier]")
        let pareto = findParetoFrontier(results)
        for point in pareto.sorted(by: { $0.recall > $1.recall }) {
            print("  \(point.parameters): \(formatDouble(point.recall * 100, decimals: 1))% @ \(formatDouble(point.qps, decimals: 0)) QPS")
        }

        // Verify minimum recall
        let bestRecall = results.map { $0.recall }.max() ?? 0
        #expect(bestRecall > 0.9, "Best recall should be > 90%")
    }

    // MARK: - HNSW Parameter Sweep

    @Test("HNSW Parameter Sweep")
    func parameterSweep() throws {
        let n = 10_000
        let d = 128
        let k = 10
        let queries = 50

        print("\n")
        print(String(repeating: "=", count: 90))
        print("HNSW Parameter Sweep: M and efConstruction impact")
        print("n=\(n), d=\(d), k=\(k), queries=\(queries)")
        print(String(repeating: "=", count: 90))

        print("\n[1/2] Generating data...")
        let trainVectors = randomVectors(count: n, dimension: d)
        let testQueries = randomVectors(count: queries, dimension: d)

        print("[2/2] Computing ground truth...")
        let groundTruths = computeGroundTruth(
            trainVectors: trainVectors,
            testQueries: testQueries,
            k: k
        )

        let configs: [(m: Int, efC: Int)] = [
            (8, 100),
            (8, 200),
            (16, 100),
            (16, 200),
            (16, 400),
            (32, 200),
        ]

        print("\n" + String(repeating: "-", count: 90))
        print("| \(pad("M", 4)) | \(pad("efConstr", 10)) | \(pad("Build(s)", 10)) | \(pad("Recall@10", 10)) | \(pad("QPS", 10)) | \(pad("Latency", 12)) |")
        print(String(repeating: "-", count: 90))

        for config in configs {
            let buildStart = CFAbsoluteTimeGetCurrent()
            let index = try HNSWIndex(
                dimensions: d,
                maxElements: n,
                metric: .l2,
                configuration: HNSWConfiguration(m: config.m, efConstruction: config.efC)
            )
            for (i, vector) in trainVectors.enumerated() {
                try index.add(vector, label: UInt64(i))
            }
            let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

            index.setEfSearch(100)

            // Warmup
            _ = try index.search(testQueries[0], k: k)

            // Benchmark
            let searchStart = CFAbsoluteTimeGetCurrent()
            var searchResults: [[SearchResult]] = []
            for query in testQueries {
                searchResults.append(try index.search(query, k: k))
            }
            let searchTime = CFAbsoluteTimeGetCurrent() - searchStart

            let qps = Double(queries) / searchTime
            let latencyMs = (searchTime / Double(queries)) * 1000
            let recall = calculateAverageRecall(results: searchResults, groundTruths: groundTruths, k: k)

            print("| \(pad(String(config.m), 4)) | \(pad(String(config.efC), 10)) | \(pad(formatDouble(buildTime, decimals: 2), 10)) | \(pad(formatDouble(recall * 100, decimals: 1) + "%", 10)) | \(pad(formatDouble(qps, decimals: 0), 10)) | \(pad(formatDouble(latencyMs, decimals: 3) + "ms", 12)) |")
        }

        print(String(repeating: "-", count: 90))
    }

    // MARK: - Scale Test

    @Test("HNSW Scale Test")
    func scaleTest() throws {
        let d = 128
        let k = 10
        let queries = 50

        print("\n")
        print(String(repeating: "=", count: 80))
        print("HNSW Scale Test: Performance vs Dataset Size")
        print("d=\(d), k=\(k), queries=\(queries), M=16, efConstruction=200")
        print(String(repeating: "=", count: 80))

        print("\n" + String(repeating: "-", count: 80))
        print("| \(pad("n", 10)) | \(pad("Build(s)", 10)) | \(pad("Vec/s", 10)) | \(pad("QPS", 10)) | \(pad("Latency", 12)) | \(pad("Index", 12)) |")
        print(String(repeating: "-", count: 80))

        for n in [1_000, 5_000, 10_000, 20_000, 50_000] {
            let trainVectors = randomVectors(count: n, dimension: d)
            let testQueries = randomVectors(count: queries, dimension: d)

            let buildStart = CFAbsoluteTimeGetCurrent()
            let index = try HNSWIndex(
                dimensions: d,
                maxElements: n,
                metric: .l2,
                configuration: HNSWConfiguration(m: 16, efConstruction: 200)
            )
            for (i, vector) in trainVectors.enumerated() {
                try index.add(vector, label: UInt64(i))
            }
            let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

            index.setEfSearch(100)

            // Warmup
            _ = try index.search(testQueries[0], k: k)

            // Benchmark
            let searchStart = CFAbsoluteTimeGetCurrent()
            for query in testQueries {
                _ = try index.search(query, k: k)
            }
            let searchTime = CFAbsoluteTimeGetCurrent() - searchStart

            let qps = Double(queries) / searchTime
            let latencyMs = (searchTime / Double(queries)) * 1000
            let vecPerSec = Double(n) / buildTime

            let vectorsSize = n * d * MemoryLayout<Float>.size
            let graphSize = n * 16 * 2 * MemoryLayout<Int>.size
            let indexSize = vectorsSize + graphSize

            print("| \(pad(String(n), 10)) | \(pad(formatDouble(buildTime, decimals: 2), 10)) | \(pad(formatDouble(vecPerSec, decimals: 0), 10)) | \(pad(formatDouble(qps, decimals: 0), 10)) | \(pad(formatDouble(latencyMs, decimals: 3) + "ms", 12)) | \(pad(formatSize(indexSize), 12)) |")
        }

        print(String(repeating: "-", count: 80))
    }

    // MARK: - Distance Metric Comparison

    @Test("Distance Metric Comparison")
    func metricComparison() throws {
        let n = 5_000
        let d = 128
        let k = 10
        let queries = 50

        print("\n")
        print(String(repeating: "=", count: 80))
        print("Distance Metric Comparison")
        print("n=\(n), d=\(d), k=\(k), queries=\(queries)")
        print(String(repeating: "=", count: 80))

        let trainVectors = randomVectors(count: n, dimension: d)
        let testQueries = randomVectors(count: queries, dimension: d)

        print("\n" + String(repeating: "-", count: 80))
        print("| \(pad("Metric", 15)) | \(pad("Build(s)", 10)) | \(pad("QPS", 10)) | \(pad("Latency", 12)) |")
        print(String(repeating: "-", count: 80))

        for metric in [DistanceMetric.l2, DistanceMetric.innerProduct, DistanceMetric.cosine] {
            let metricName: String
            switch metric {
            case .l2: metricName = "L2 (Euclidean)"
            case .innerProduct: metricName = "Inner Product"
            case .cosine: metricName = "Cosine"
            }

            let buildStart = CFAbsoluteTimeGetCurrent()
            let index = try HNSWIndex(
                dimensions: d,
                maxElements: n,
                metric: metric,
                configuration: HNSWConfiguration(m: 16, efConstruction: 200)
            )
            for (i, vector) in trainVectors.enumerated() {
                try index.add(vector, label: UInt64(i))
            }
            let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

            index.setEfSearch(100)

            // Warmup
            _ = try index.search(testQueries[0], k: k)

            // Benchmark
            let searchStart = CFAbsoluteTimeGetCurrent()
            for query in testQueries {
                _ = try index.search(query, k: k)
            }
            let searchTime = CFAbsoluteTimeGetCurrent() - searchStart

            let qps = Double(queries) / searchTime
            let latencyMs = (searchTime / Double(queries)) * 1000

            print("| \(pad(metricName, 15)) | \(pad(formatDouble(buildTime, decimals: 2), 10)) | \(pad(formatDouble(qps, decimals: 0), 10)) | \(pad(formatDouble(latencyMs, decimals: 3) + "ms", 12)) |")
        }

        print(String(repeating: "-", count: 80))
    }

    // MARK: - Batch Operations Benchmark

    @Test("Batch vs Single Operations")
    func batchOperationsBenchmark() throws {
        let n = 10_000
        let d = 128
        let k = 10
        let queries = 100

        print("\n")
        print(String(repeating: "=", count: 80))
        print("Batch vs Single Operations Benchmark")
        print("n=\(n), d=\(d), k=\(k), queries=\(queries)")
        print(String(repeating: "=", count: 80))

        let trainVectors = randomVectors(count: n, dimension: d)
        let testQueries = randomVectors(count: queries, dimension: d)

        // Single operations
        print("\n[Single Operations]")
        let singleIndex = try HNSWIndex(
            dimensions: d,
            maxElements: n,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )

        let singleAddStart = CFAbsoluteTimeGetCurrent()
        for (i, vector) in trainVectors.enumerated() {
            try singleIndex.add(vector, label: UInt64(i))
        }
        let singleAddTime = CFAbsoluteTimeGetCurrent() - singleAddStart

        singleIndex.setEfSearch(100)
        let singleSearchStart = CFAbsoluteTimeGetCurrent()
        for query in testQueries {
            _ = try singleIndex.search(query, k: k)
        }
        let singleSearchTime = CFAbsoluteTimeGetCurrent() - singleSearchStart

        // Batch operations
        print("[Batch Operations]")
        let batchIndex = try HNSWIndex(
            dimensions: d,
            maxElements: n,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )

        let batchAddStart = CFAbsoluteTimeGetCurrent()
        try batchIndex.addBatch(trainVectors)
        let batchAddTime = CFAbsoluteTimeGetCurrent() - batchAddStart

        batchIndex.setEfSearch(100)
        let batchSearchStart = CFAbsoluteTimeGetCurrent()
        _ = try batchIndex.searchBatch(testQueries, k: k)
        let batchSearchTime = CFAbsoluteTimeGetCurrent() - batchSearchStart

        // Results
        print("\n" + String(repeating: "-", count: 80))
        print("| \(pad("Operation", 20)) | \(pad("Single", 12)) | \(pad("Batch", 12)) | \(pad("Speedup", 10)) |")
        print(String(repeating: "-", count: 80))
        print("| \(pad("Add " + String(n) + " vectors", 20)) | \(pad(formatDouble(singleAddTime, decimals: 2) + "s", 12)) | \(pad(formatDouble(batchAddTime, decimals: 2) + "s", 12)) | \(pad(formatDouble(singleAddTime / batchAddTime, decimals: 2) + "x", 10)) |")
        print("| \(pad("Search " + String(queries) + " queries", 20)) | \(pad(formatDouble(singleSearchTime * 1000, decimals: 1) + "ms", 12)) | \(pad(formatDouble(batchSearchTime * 1000, decimals: 1) + "ms", 12)) | \(pad(formatDouble(singleSearchTime / batchSearchTime, decimals: 2) + "x", 10)) |")
        print(String(repeating: "-", count: 80))

        let singleQPS = Double(queries) / singleSearchTime
        let batchQPS = Double(queries) / batchSearchTime
        print("\n[Search QPS]")
        print("  Single: \(formatDouble(singleQPS, decimals: 0)) QPS")
        print("  Batch:  \(formatDouble(batchQPS, decimals: 0)) QPS")
    }

    // MARK: - Concurrent Read Test

    @Test("Concurrent Read Performance")
    func concurrentReadTest() async throws {
        let n = 10_000
        let d = 128
        let k = 10
        let queriesPerThread = 100
        let threadCounts = [1, 2, 4, 8]

        print("\n")
        print(String(repeating: "=", count: 70))
        print("Concurrent Read Performance (RWLock)")
        print("n=\(n), d=\(d), k=\(k), queries/thread=\(queriesPerThread)")
        print(String(repeating: "=", count: 70))

        let trainVectors = randomVectors(count: n, dimension: d)
        let testQueries = randomVectors(count: queriesPerThread, dimension: d)

        let index = try HNSWIndex(
            dimensions: d,
            maxElements: n,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )
        for (i, vector) in trainVectors.enumerated() {
            try index.add(vector, label: UInt64(i))
        }
        index.setEfSearch(100)

        print("\n" + String(repeating: "-", count: 70))
        print("| \(pad("Threads", 10)) | \(pad("Total QPS", 12)) | \(pad("QPS/Thread", 12)) | \(pad("Speedup", 10)) |")
        print(String(repeating: "-", count: 70))

        var baselineQps: Double = 0

        for threadCount in threadCounts {
            let start = CFAbsoluteTimeGetCurrent()

            await withTaskGroup(of: Void.self) { group in
                for _ in 0..<threadCount {
                    group.addTask {
                        for query in testQueries {
                            _ = try? index.search(query, k: k)
                        }
                    }
                }
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let totalQueries = threadCount * queriesPerThread
            let totalQps = Double(totalQueries) / elapsed
            let qpsPerThread = totalQps / Double(threadCount)

            if threadCount == 1 {
                baselineQps = totalQps
            }
            let speedup = totalQps / baselineQps

            print("| \(pad(String(threadCount), 10)) | \(pad(formatDouble(totalQps, decimals: 0), 12)) | \(pad(formatDouble(qpsPerThread, decimals: 0), 12)) | \(pad(formatDouble(speedup, decimals: 2) + "x", 10)) |")
        }

        print(String(repeating: "-", count: 70))
    }
}
