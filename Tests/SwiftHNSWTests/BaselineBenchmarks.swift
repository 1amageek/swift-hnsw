// BaselineBenchmarks.swift
// Baseline performance measurements for Float32 and Float16
// across multiple embedding dimensions.
//
// Run: xcodebuild test -scheme swift-hnsw -only-testing SwiftHNSWTests/BaselineBenchmarks

import Testing
import Foundation
@testable import SwiftHNSW

// MARK: - Report Builder

/// Builds a Markdown report from benchmark results
final class ReportBuilder {
    private var sections: [(title: String, content: String)] = []

    func addSection(title: String, content: String) {
        sections.append((title, content))
    }

    func addTable(title: String, headers: [String], rows: [[String]]) {
        var lines: [String] = []
        lines.append("### \(title)")
        lines.append("")

        let separator = headers.map { _ in "---" }
        lines.append("| " + headers.joined(separator: " | ") + " |")
        lines.append("| " + separator.joined(separator: " | ") + " |")
        for row in rows {
            lines.append("| " + row.joined(separator: " | ") + " |")
        }
        lines.append("")

        sections.append((title, lines.joined(separator: "\n")))
    }

    func build(header: String) -> String {
        var output = header + "\n\n"
        for section in sections {
            output += section.content + "\n"
        }
        return output
    }

    func write(to path: String) throws {
        let header = """
        # Baseline Benchmark Report

        **Date**: \(ISO8601DateFormatter().string(from: Date()))
        **Platform**: \(platformInfo())
        """
        let content = build(header: header)
        try content.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

private func platformInfo() -> String {
    var sysinfo = utsname()
    uname(&sysinfo)
    let machine = withUnsafePointer(to: &sysinfo.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: 1) {
            String(cString: $0)
        }
    }
    let release = withUnsafePointer(to: &sysinfo.release) {
        $0.withMemoryRebound(to: CChar.self, capacity: 1) {
            String(cString: $0)
        }
    }
    return "\(machine), Darwin \(release)"
}

// MARK: - Benchmark Helpers

private func generateRandomVectors(count: Int, dimension: Int) -> [[Float]] {
    (0..<count).map { _ in
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }
}

private func bruteForceKNN(query: [Float], vectors: [[Float]], k: Int) -> [UInt64] {
    let distances: [(UInt64, Float)] = vectors.enumerated().map { (i, v) in
        var sum: Float = 0
        for j in 0..<query.count {
            let diff = query[j] - v[j]
            sum += diff * diff
        }
        return (UInt64(i), sum)
    }
    return distances.sorted { $0.1 < $1.1 }.prefix(k).map { $0.0 }
}

private func recallAtK(result: [SearchResult], truth: [UInt64], k: Int) -> Double {
    let resultSet = Set(result.prefix(k).map { $0.label })
    let truthSet = Set(truth.prefix(k))
    return Double(resultSet.intersection(truthSet).count) / Double(k)
}

private func averageRecall(results: [[SearchResult]], truths: [[UInt64]], k: Int) -> Double {
    let recalls = zip(results, truths).map { recallAtK(result: $0, truth: $1, k: k) }
    return recalls.reduce(0.0, +) / Double(recalls.count)
}

private func formatBytes(_ bytes: Int) -> String {
    if bytes >= 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    }
    return String(format: "%.0f KB", Double(bytes) / 1024)
}

// MARK: - Benchmark Result Types

struct DimensionResult {
    let dimension: Int
    let scalarType: String
    let buildTime: Double
    let buildThroughput: Double
    let searchQPS: Double
    let searchLatencyMs: Double
    let recall: Double
    let vectorMemory: Int
    let indexSize: Int
}

struct EfSearchResult {
    let efSearch: Int
    let scalarType: String
    let recall: Double
    let qps: Double
    let latencyMs: Double
}

// MARK: - Benchmark Suite

@Suite("Baseline Benchmarks", .serialized)
struct BaselineBenchmarks {

    static let n = 10_000
    static let k = 10
    static let numQueries = 100
    static let m = 16
    static let efConstruction = 200

    // MARK: - Multi-Dimension Baseline

    @Test("Multi-Dimension Baseline")
    func multiDimensionBaseline() throws {
        let dimensions = [128, 384, 768, 1536]
        let efSearch = 100

        let report = ReportBuilder()
        var allResults: [DimensionResult] = []

        print("\n" + String(repeating: "=", count: 100))
        print("Baseline Benchmark: Float32 vs Float16 across dimensions")
        print("n=\(Self.n), k=\(Self.k), queries=\(Self.numQueries), M=\(Self.m), efConstruction=\(Self.efConstruction), efSearch=\(efSearch)")
        print(String(repeating: "=", count: 100))

        for d in dimensions {
            print("\n--- Dimension \(d) ---")

            // Generate data
            print("  Generating \(Self.n) vectors (d=\(d))...")
            let trainVectors = generateRandomVectors(count: Self.n, dimension: d)
            let testQueries = generateRandomVectors(count: Self.numQueries, dimension: d)

            // Ground truth
            print("  Computing ground truth...")
            let groundTruths = testQueries.map { bruteForceKNN(query: $0, vectors: trainVectors, k: Self.k) }

            // Float32 benchmark
            print("  Benchmarking Float32...")
            let f32Result = try benchmarkFloat32(
                trainVectors: trainVectors,
                testQueries: testQueries,
                groundTruths: groundTruths,
                dimension: d,
                efSearch: efSearch
            )
            allResults.append(f32Result)

            // Float16 benchmark
            print("  Benchmarking Float16...")
            let f16Result = try benchmarkFloat16(
                trainVectors: trainVectors,
                testQueries: testQueries,
                groundTruths: groundTruths,
                dimension: d,
                efSearch: efSearch
            )
            allResults.append(f16Result)

            print("  Float32: Recall=\(String(format: "%.1f%%", f32Result.recall * 100)), QPS=\(String(format: "%.0f", f32Result.searchQPS))")
            print("  Float16: Recall=\(String(format: "%.1f%%", f16Result.recall * 100)), QPS=\(String(format: "%.0f", f16Result.searchQPS))")
        }

        // Build report table
        let headers = ["Dimension", "Type", "Build (s)", "Vec/s", "QPS", "Latency (ms)", "Recall@10", "Vector Mem", "Index Size"]
        let rows: [[String]] = allResults.map { r in
            [
                String(r.dimension),
                r.scalarType,
                String(format: "%.2f", r.buildTime),
                String(format: "%.0f", r.buildThroughput),
                String(format: "%.0f", r.searchQPS),
                String(format: "%.3f", r.searchLatencyMs),
                String(format: "%.1f%%", r.recall * 100),
                formatBytes(r.vectorMemory),
                formatBytes(r.indexSize),
            ]
        }
        report.addTable(title: "Multi-Dimension Results (efSearch=\(efSearch))", headers: headers, rows: rows)

        // Compression comparison table
        let compHeaders = ["Dimension", "F32 Memory", "F16 Memory", "Savings", "F32 QPS", "F16 QPS", "Speedup", "Recall Diff"]
        var compRows: [[String]] = []
        for d in dimensions {
            let f32 = allResults.first { $0.dimension == d && $0.scalarType == "Float32" }!
            let f16 = allResults.first { $0.dimension == d && $0.scalarType == "Float16" }!
            compRows.append([
                String(d),
                formatBytes(f32.vectorMemory),
                formatBytes(f16.vectorMemory),
                String(format: "%.1fx", Double(f32.vectorMemory) / Double(f16.vectorMemory)),
                String(format: "%.0f", f32.searchQPS),
                String(format: "%.0f", f16.searchQPS),
                String(format: "%.2fx", f16.searchQPS / f32.searchQPS),
                String(format: "%.2f%%", abs(f16.recall - f32.recall) * 100),
            ])
        }
        report.addTable(title: "Float32 vs Float16 Comparison", headers: compHeaders, rows: compRows)

        // Write report
        let reportPath = findProjectRoot() + "/reports/baseline_benchmark.md"
        try report.write(to: reportPath)
        print("\nReport written to: \(reportPath)")
    }

    // MARK: - efSearch Sweep

    @Test("efSearch Sweep d=128")
    func efSearchSweep() throws {
        let d = 128
        let efSearchValues = [10, 20, 50, 100, 200, 320]

        print("\n" + String(repeating: "=", count: 90))
        print("efSearch Sweep: Recall vs QPS Tradeoff (d=\(d))")
        print("n=\(Self.n), k=\(Self.k), queries=\(Self.numQueries)")
        print(String(repeating: "=", count: 90))

        // Generate data
        let trainVectors = generateRandomVectors(count: Self.n, dimension: d)
        let testQueries = generateRandomVectors(count: Self.numQueries, dimension: d)
        let groundTruths = testQueries.map { bruteForceKNN(query: $0, vectors: trainVectors, k: Self.k) }

        var sweepResults: [EfSearchResult] = []

        // Float32 index
        let indexF32 = try HNSWIndex<Float>(
            dimensions: d,
            maxElements: Self.n,
            metric: .l2,
            configuration: HNSWConfiguration(m: Self.m, efConstruction: Self.efConstruction)
        )
        for (i, v) in trainVectors.enumerated() {
            try indexF32.add(v, label: UInt64(i))
        }

        // Float16 index
        let trainF16: [[Float16]] = trainVectors.map { $0.map { Float16($0) } }
        let testF16: [[Float16]] = testQueries.map { $0.map { Float16($0) } }
        let indexF16 = try HNSWIndex<Float16>(
            dimensions: d,
            maxElements: Self.n,
            metric: .l2,
            configuration: HNSWConfiguration(m: Self.m, efConstruction: Self.efConstruction)
        )
        for (i, v) in trainF16.enumerated() {
            try indexF16.add(v, label: UInt64(i))
        }

        for ef in efSearchValues {
            // Float32
            indexF32.setEfSearch(ef)
            _ = try indexF32.search(testQueries[0], k: Self.k) // warmup

            let startF32 = CFAbsoluteTimeGetCurrent()
            var resultsF32: [[SearchResult]] = []
            for q in testQueries {
                resultsF32.append(try indexF32.search(q, k: Self.k))
            }
            let timeF32 = CFAbsoluteTimeGetCurrent() - startF32
            let recallF32 = averageRecall(results: resultsF32, truths: groundTruths, k: Self.k)

            sweepResults.append(EfSearchResult(
                efSearch: ef,
                scalarType: "Float32",
                recall: recallF32,
                qps: Double(Self.numQueries) / timeF32,
                latencyMs: (timeF32 / Double(Self.numQueries)) * 1000
            ))

            // Float16
            indexF16.setEfSearch(ef)
            _ = try indexF16.search(testF16[0], k: Self.k) // warmup

            let startF16 = CFAbsoluteTimeGetCurrent()
            var resultsF16: [[SearchResult]] = []
            for q in testF16 {
                resultsF16.append(try indexF16.search(q, k: Self.k))
            }
            let timeF16 = CFAbsoluteTimeGetCurrent() - startF16
            let recallF16 = averageRecall(results: resultsF16, truths: groundTruths, k: Self.k)

            sweepResults.append(EfSearchResult(
                efSearch: ef,
                scalarType: "Float16",
                recall: recallF16,
                qps: Double(Self.numQueries) / timeF16,
                latencyMs: (timeF16 / Double(Self.numQueries)) * 1000
            ))
        }

        // Print results
        print("\n| efSearch | Type    | Recall@10 | QPS      | Latency (ms) |")
        print("|----------|---------|-----------|----------|--------------|")
        for r in sweepResults {
            let line = "| \(r.efSearch) | \(r.scalarType) | \(String(format: "%.1f", r.recall * 100))% | \(String(format: "%.0f", r.qps)) | \(String(format: "%.3f", r.latencyMs)) |"
            print(line)
        }

        // Append to report
        let reportPath = findProjectRoot() + "/reports/baseline_benchmark.md"
        let headers = ["efSearch", "Type", "Recall@10", "QPS", "Latency (ms)"]
        let rows: [[String]] = sweepResults.map { r in
            [
                String(r.efSearch),
                r.scalarType,
                String(format: "%.1f%%", r.recall * 100),
                String(format: "%.0f", r.qps),
                String(format: "%.3f", r.latencyMs),
            ]
        }

        var tableLines = "\n### efSearch Sweep (d=\(d), n=\(Self.n))\n\n"
        tableLines += "| " + headers.joined(separator: " | ") + " |\n"
        tableLines += "| " + headers.map { _ in "---" }.joined(separator: " | ") + " |\n"
        for row in rows {
            tableLines += "| " + row.joined(separator: " | ") + " |\n"
        }

        if FileManager.default.fileExists(atPath: reportPath) {
            let existing = try String(contentsOfFile: reportPath, encoding: .utf8)
            try (existing + tableLines).write(toFile: reportPath, atomically: true, encoding: .utf8)
        } else {
            try tableLines.write(toFile: reportPath, atomically: true, encoding: .utf8)
        }

        print("\nefSearch sweep appended to: \(reportPath)")

        // Verify high efSearch achieves good recall
        let bestRecall = sweepResults.map { $0.recall }.max() ?? 0
        #expect(bestRecall > 0.9, "Best recall should exceed 90%")
    }

    // MARK: - Private Helpers

    private func benchmarkFloat32(
        trainVectors: [[Float]],
        testQueries: [[Float]],
        groundTruths: [[UInt64]],
        dimension: Int,
        efSearch: Int
    ) throws -> DimensionResult {
        let buildStart = CFAbsoluteTimeGetCurrent()
        let index = try HNSWIndex<Float>(
            dimensions: dimension,
            maxElements: Self.n,
            metric: .l2,
            configuration: HNSWConfiguration(m: Self.m, efConstruction: Self.efConstruction)
        )
        for (i, v) in trainVectors.enumerated() {
            try index.add(v, label: UInt64(i))
        }
        let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

        index.setEfSearch(efSearch)
        _ = try index.search(testQueries[0], k: Self.k) // warmup

        let searchStart = CFAbsoluteTimeGetCurrent()
        var results: [[SearchResult]] = []
        for q in testQueries {
            results.append(try index.search(q, k: Self.k))
        }
        let searchTime = CFAbsoluteTimeGetCurrent() - searchStart
        let recall = averageRecall(results: results, truths: groundTruths, k: Self.k)

        let vectorMemory = Self.n * dimension * MemoryLayout<Float>.size
        let graphMemory = Self.n * Self.m * 2 * MemoryLayout<Int32>.size

        return DimensionResult(
            dimension: dimension,
            scalarType: "Float32",
            buildTime: buildTime,
            buildThroughput: Double(Self.n) / buildTime,
            searchQPS: Double(Self.numQueries) / searchTime,
            searchLatencyMs: (searchTime / Double(Self.numQueries)) * 1000,
            recall: recall,
            vectorMemory: vectorMemory,
            indexSize: vectorMemory + graphMemory
        )
    }

    private func benchmarkFloat16(
        trainVectors: [[Float]],
        testQueries: [[Float]],
        groundTruths: [[UInt64]],
        dimension: Int,
        efSearch: Int
    ) throws -> DimensionResult {
        let trainF16: [[Float16]] = trainVectors.map { $0.map { Float16($0) } }
        let testF16: [[Float16]] = testQueries.map { $0.map { Float16($0) } }

        let buildStart = CFAbsoluteTimeGetCurrent()
        let index = try HNSWIndex<Float16>(
            dimensions: dimension,
            maxElements: Self.n,
            metric: .l2,
            configuration: HNSWConfiguration(m: Self.m, efConstruction: Self.efConstruction)
        )
        for (i, v) in trainF16.enumerated() {
            try index.add(v, label: UInt64(i))
        }
        let buildTime = CFAbsoluteTimeGetCurrent() - buildStart

        index.setEfSearch(efSearch)
        _ = try index.search(testF16[0], k: Self.k) // warmup

        let searchStart = CFAbsoluteTimeGetCurrent()
        var results: [[SearchResult]] = []
        for q in testF16 {
            results.append(try index.search(q, k: Self.k))
        }
        let searchTime = CFAbsoluteTimeGetCurrent() - searchStart
        let recall = averageRecall(results: results, truths: groundTruths, k: Self.k)

        let vectorMemory = Self.n * dimension * MemoryLayout<Float16>.size
        let graphMemory = Self.n * Self.m * 2 * MemoryLayout<Int32>.size

        return DimensionResult(
            dimension: dimension,
            scalarType: "Float16",
            buildTime: buildTime,
            buildThroughput: Double(Self.n) / buildTime,
            searchQPS: Double(Self.numQueries) / searchTime,
            searchLatencyMs: (searchTime / Double(Self.numQueries)) * 1000,
            recall: recall,
            vectorMemory: vectorMemory,
            indexSize: vectorMemory + graphMemory
        )
    }
}

// MARK: - Project Root Detection

func findProjectRoot() -> String {
    // Walk up from the test binary location to find Package.swift
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<10 {
        url = url.deletingLastPathComponent()
        let packageSwift = url.appendingPathComponent("Package.swift")
        if FileManager.default.fileExists(atPath: packageSwift.path) {
            return url.path
        }
    }
    // Fallback
    return FileManager.default.currentDirectoryPath
}
