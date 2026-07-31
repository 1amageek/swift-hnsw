import Foundation
import Darwin
@_spi(Benchmarking)
import SwiftHNSW

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func nextUnit() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        let unit = Double(value >> 11) * (1.0 / 9_007_199_254_740_992.0)
        return Float(unit * 2.0 - 1.0)
    }
}

private struct BenchmarkCase {
    let name: String
    let count: Int
    let dimensions: Int
    let queryCount: Int
    let k: Int
    let iterations: Int
}

private struct Measurement {
    let index: String
    let bitWidth: Int?
    let efSearch: Int
    let medianMilliseconds: Double
    let p95Milliseconds: Double
    let queriesPerSecond: Double
    let cpuMilliseconds: Double
    let cpuQueriesPerSecond: Double
    let recall: Double
    let checksum: UInt64
    let qjlScoredCandidates: Int?
    let qjlPrunedCandidates: Int?
}

private struct SearchTarget {
    let name: String
    let bitWidth: Int?
    let setEfSearch: (Int) throws -> Void
    let search: ([Float]) throws -> [SearchResult]
    let resetQJLPruningStatistics: (() -> Void)?
    let qjlPruningStatistics: (() -> TurboQuantQJLPruningStatistics)?
}

private struct MeasurementSamples {
    var wallMilliseconds: [Double] = []
    var cpuMilliseconds: [Double] = []
    var individualMilliseconds: [Double] = []
    var lastResults: [[SearchResult]] = []
    var checksum: UInt64 = 0
    var qjlScoredCandidates: [Int] = []
    var qjlPrunedCandidates: [Int] = []
}

private enum BenchmarkRunnerError: Error {
    case unsupportedTarget(String)
}

@main
struct BenchmarkRunner {
    private static let efSearchValues = [10, 20, 40, 60, 100, 200, 400]
    private static let supportedTargets = [
        "hnsw-cosine",
        "turboquant-mse-2",
        "turboquant-mse-4",
        "turboquant-product-5",
    ]

    static func main() throws {
        let environment = ProcessInfo.processInfo.environment
        let toolchainIdentifier = environment[
            "TOOLCHAINS"
        ] ?? "default"
        let requestedTarget = environment[
            "TURBOQUANT_BENCHMARK_TARGET"
        ]
        let qjlPruningModeName = environment[
            "TURBOQUANT_QJL_PRUNING"
        ] ?? "automatic"
        let qjlPruningMode: TurboQuantQJLPruningMode
        let collectQJLPruningStatistics: Bool
        switch qjlPruningModeName {
        case "automatic":
            qjlPruningMode = .automatic
            collectQJLPruningStatistics = false
        case "automatic-stats":
            qjlPruningMode = .automatic
            collectQJLPruningStatistics = true
        case "always":
            qjlPruningMode = .always
            collectQJLPruningStatistics = false
        case "always-stats":
            qjlPruningMode = .always
            collectQJLPruningStatistics = true
        case "never":
            qjlPruningMode = .never
            collectQJLPruningStatistics = false
        case "never-stats":
            qjlPruningMode = .never
            collectQJLPruningStatistics = true
        default:
            throw BenchmarkRunnerError.unsupportedTarget(
                "TURBOQUANT_QJL_PRUNING=\(qjlPruningModeName)"
            )
        }
        if let requestedTarget,
           !supportedTargets.contains(requestedTarget) {
            throw BenchmarkRunnerError.unsupportedTarget(requestedTarget)
        }
        print(
            "turboquant_benchmark " +
            "toolchain_identifier=\(toolchainIdentifier)"
        )
        print(
            "turboquant_benchmark compiler=" +
            commandLineValue(swiftVersion())
        )
        print(
            "turboquant_benchmark " +
            "comparison=HNSW cosine vs TurboQuant MSE and product objectives"
        )
        print(
            "turboquant_benchmark run_id=" +
            commandLineValue(
                environment["TURBOQUANT_BENCHMARK_RUN_ID"] ?? "unspecified"
            )
        )
        print(
            "turboquant_benchmark source_revision=" +
            commandLineValue(
                commandOutput(
                    executable: "/usr/bin/env",
                    arguments: ["git", "rev-parse", "HEAD"]
                ) ?? "unavailable"
            )
        )
        let workingTreeStatus = commandOutput(
            executable: "/usr/bin/env",
            arguments: ["git", "status", "--porcelain"]
        )
        print(
            "turboquant_benchmark source_state=" +
            (workingTreeStatus?.isEmpty == true ? "clean" : "dirty")
        )
        print(
            "turboquant_benchmark measurement_mode=" +
            (requestedTarget == nil ? "interleaved" : "isolated") +
            " target=\(requestedTarget ?? "all")"
        )
        print(
            "turboquant_benchmark qjl_pruning=\(qjlPruningModeName)"
        )
        let allCases = [
            BenchmarkCase(
                name: "n10k-d128",
                count: 10_000,
                dimensions: 128,
                queryCount: 50,
                k: 10,
                iterations: 5
            ),
            BenchmarkCase(
                name: "n10k-d768",
                count: 10_000,
                dimensions: 768,
                queryCount: 50,
                k: 10,
                iterations: 5
            ),
        ]
        let selectedCases: [BenchmarkCase]
        let requestedCase = ProcessInfo.processInfo.environment[
            "TURBOQUANT_BENCHMARK_CASE"
        ]
        if requestedCase == "smoke" {
            selectedCases = [
                BenchmarkCase(
                    name: "smoke",
                    count: 500,
                    dimensions: 64,
                    queryCount: 10,
                    k: 10,
                    iterations: 1
                )
            ]
        } else if requestedCase == "large-d768" {
            selectedCases = [
                BenchmarkCase(
                    name: "n50k-d768",
                    count: 50_000,
                    dimensions: 768,
                    queryCount: 30,
                    k: 10,
                    iterations: 3
                )
            ]
        } else if requestedCase == "d768" {
            selectedCases = allCases.filter { $0.dimensions == 768 }
        } else if requestedCase == "d128" {
            selectedCases = allCases.filter { $0.dimensions == 128 }
        } else {
            selectedCases = allCases
        }
        for benchmarkCase in selectedCases {
            try run(
                benchmarkCase,
                requestedTarget: requestedTarget,
                qjlPruningMode: qjlPruningMode,
                collectQJLPruningStatistics: collectQJLPruningStatistics
            )
        }
    }

    private static func run(
        _ benchmarkCase: BenchmarkCase,
        requestedTarget: String?,
        qjlPruningMode: TurboQuantQJLPruningMode,
        collectQJLPruningStatistics: Bool
    ) throws {
        print("")
        print(
            "case=\(benchmarkCase.name) n=\(benchmarkCase.count) " +
            "d=\(benchmarkCase.dimensions) q=\(benchmarkCase.queryCount) " +
            "k=\(benchmarkCase.k)"
        )
        let vectors = makeVectors(
            count: benchmarkCase.count,
            dimensions: benchmarkCase.dimensions,
            seed: 0x1234_5678
        )
        let queries = makeVectors(
            count: benchmarkCase.queryCount,
            dimensions: benchmarkCase.dimensions,
            seed: 0x8765_4321
        )
        let groundTruth = cosineGroundTruth(
            vectors: vectors,
            queries: queries,
            k: benchmarkCase.k
        )

        var targets: [SearchTarget] = []
        if includesTarget(
            name: "hnsw-cosine",
            bitWidth: nil,
            requestedTarget: requestedTarget
        ) {
            let hnswStart = ContinuousClock.now
            let hnsw = try HNSWIndex<Float>(
                dimensions: benchmarkCase.dimensions,
                maxElements: benchmarkCase.count,
                metric: .cosine,
                configuration: HNSWConfiguration(m: 16, efConstruction: 200)
            )
            for (label, vector) in vectors.enumerated() {
                try hnsw.add(vector, label: UInt64(label))
            }
            let hnswBuildMilliseconds = milliseconds(
                from: hnswStart.duration(to: .now)
            )
            print(
                "build index=hnsw-cosine ms=\(format(hnswBuildMilliseconds)) " +
                "retained_vector_bytes=" +
                "\(benchmarkCase.count * benchmarkCase.dimensions * 4)"
            )
            targets.append(
                SearchTarget(
                    name: "hnsw-cosine",
                    bitWidth: nil,
                    setEfSearch: { try hnsw.setEfSearch($0) },
                    search: { try hnsw.search($0, k: benchmarkCase.k) },
                    resetQJLPruningStatistics: nil,
                    qjlPruningStatistics: nil
                )
            )
        }

        let allTurboCases: [(String, TurboQuantObjective, Int)] = [
            ("turboquant-mse", .meanSquaredError, 4),
            ("turboquant-mse", .meanSquaredError, 2),
            ("turboquant-product", .innerProduct, 5),
        ]
        let turboCases: [(String, TurboQuantObjective, Int)]
        if ProcessInfo.processInfo.environment["TURBOQUANT_MSE_ONLY"] == "1" {
            turboCases = allTurboCases.filter {
                $0.1 == .meanSquaredError && $0.2 == 4
            }
        } else if ProcessInfo.processInfo.environment[
            "TURBOQUANT_PRIMARY_ONLY"
        ] == "1" {
            turboCases = allTurboCases.filter { $0.2 >= 4 }
        } else {
            turboCases = allTurboCases
        }
        for (name, objective, bitWidth) in turboCases where includesTarget(
            name: name,
            bitWidth: bitWidth,
            requestedTarget: requestedTarget
        ) {
            let turboStart = ContinuousClock.now
            let turbo: TurboQuantIndex
            if objective == .innerProduct {
                turbo = try TurboQuantIndex(
                    dimensions: benchmarkCase.dimensions,
                    maxElements: benchmarkCase.count,
                    bitWidth: bitWidth,
                    objective: objective,
                    configuration: HNSWConfiguration(
                        m: 16,
                        efConstruction: 200
                    ),
                    seed: 42,
                    qjlPruningMode: qjlPruningMode,
                    collectQJLPruningStatistics:
                        collectQJLPruningStatistics
                )
            } else {
                turbo = try TurboQuantIndex(
                    dimensions: benchmarkCase.dimensions,
                    maxElements: benchmarkCase.count,
                    bitWidth: bitWidth,
                    objective: objective,
                    configuration: HNSWConfiguration(
                        m: 16,
                        efConstruction: 200
                    ),
                    seed: 42
                )
            }
            for (label, vector) in vectors.enumerated() {
                try turbo.add(vector, label: UInt64(label))
            }
            try turbo.finalize()
            let turboBuildMilliseconds = milliseconds(
                from: turboStart.duration(to: .now)
            )
            print(
                "build index=\(name) bits=\(bitWidth) " +
                "ms=\(format(turboBuildMilliseconds)) " +
                "input_vector_bytes=" +
                "\(benchmarkCase.count * benchmarkCase.dimensions * 4) " +
                "retained_packed_record_bytes=" +
                "\(benchmarkCase.count * turbo.bytesPerVector) " +
                "shared_projection_bytes=\(turbo.projectionBytes)"
            )
            targets.append(
                SearchTarget(
                    name: name,
                    bitWidth: bitWidth,
                    setEfSearch: { try turbo.setEfSearch($0) },
                    search: { try turbo.search($0, k: benchmarkCase.k) },
                    resetQJLPruningStatistics:
                        objective == .innerProduct && collectQJLPruningStatistics
                        ? { turbo.resetQJLPruningStatistics() }
                        : nil,
                    qjlPruningStatistics:
                        objective == .innerProduct && collectQJLPruningStatistics
                        ? { turbo.qjlPruningStatistics() }
                        : nil
                )
            )
        }
        guard !targets.isEmpty else {
            throw BenchmarkRunnerError.unsupportedTarget(
                requestedTarget ?? "all"
            )
        }

        for (efIndex, efSearch) in efSearchValues.enumerated() {
            let measurements = try measure(
                targets: targets,
                efSearch: efSearch,
                scheduleOffset: efIndex,
                queries: queries,
                groundTruth: groundTruth,
                k: benchmarkCase.k,
                iterations: benchmarkCase.iterations
            )
            for measurement in measurements {
                print(measurementLine(measurement))
            }
        }
    }

    private static func measure(
        targets: [SearchTarget],
        efSearch: Int,
        scheduleOffset: Int,
        queries: [[Float]],
        groundTruth: [[UInt64]],
        k: Int,
        iterations: Int
    ) throws -> [Measurement] {
        for target in targets {
            try target.setEfSearch(efSearch)
        }

        for targetIndex in scheduledIndices(
            count: targets.count,
            offset: scheduleOffset
        ) {
            for query in queries {
                _ = try targets[targetIndex].search(query)
            }
        }
        for target in targets {
            target.resetQJLPruningStatistics?()
        }

        var samples = targets.map { _ in MeasurementSamples() }
        for index in samples.indices {
            samples[index].wallMilliseconds.reserveCapacity(iterations)
            samples[index].cpuMilliseconds.reserveCapacity(iterations)
            samples[index].individualMilliseconds.reserveCapacity(queries.count)
        }

        for iteration in 0..<iterations {
            for targetIndex in scheduledIndices(
                count: targets.count,
                offset: scheduleOffset + iteration
            ) {
                let target = targets[targetIndex]
                target.resetQJLPruningStatistics?()
                let start = ContinuousClock.now
                let cpuStart = processCPUSeconds()
                var results: [[SearchResult]] = []
                results.reserveCapacity(queries.count)
                var trialChecksum = UInt64(iteration)
                for query in queries {
                    let result = try target.search(query)
                    trialChecksum &+= result.reduce(UInt64.zero) {
                        $0 &+ $1.label
                    }
                    results.append(result)
                }
                let wallMilliseconds = milliseconds(
                    from: start.duration(to: .now)
                )
                let cpuMilliseconds = (
                    processCPUSeconds() - cpuStart
                ) * 1_000
                samples[targetIndex].wallMilliseconds.append(wallMilliseconds)
                samples[targetIndex].cpuMilliseconds.append(cpuMilliseconds)
                samples[targetIndex].lastResults = results
                samples[targetIndex].checksum &+= trialChecksum
                if let qjlStatistics = target.qjlPruningStatistics?() {
                    samples[targetIndex].qjlScoredCandidates.append(
                        qjlStatistics.scoredCandidates
                    )
                    samples[targetIndex].qjlPrunedCandidates.append(
                        qjlStatistics.prunedCandidates
                    )
                }
                print(
                    "sample phase=batch index=\(target.name) " +
                    "bits=\(target.bitWidth.map(String.init) ?? "-") " +
                    "ef=\(efSearch) trial=\(iteration) " +
                    "wall_ms=\(format(wallMilliseconds)) " +
                    "cpu_ms=\(format(cpuMilliseconds)) " +
                    "checksum=\(trialChecksum)" +
                    qjlStatisticsSuffix(
                        target.qjlPruningStatistics?()
                    )
                )
            }
        }

        for queryIndex in queries.indices {
            for targetIndex in scheduledIndices(
                count: targets.count,
                offset: scheduleOffset + queryIndex
            ) {
                let target = targets[targetIndex]
                let start = ContinuousClock.now
                let result = try target.search(queries[queryIndex])
                let duration = milliseconds(from: start.duration(to: .now))
                let queryChecksum = result.reduce(UInt64.zero) {
                    $0 &+ $1.label
                }
                samples[targetIndex].individualMilliseconds.append(duration)
                samples[targetIndex].checksum &+= queryChecksum
                print(
                    "sample phase=query index=\(target.name) " +
                    "bits=\(target.bitWidth.map(String.init) ?? "-") " +
                    "ef=\(efSearch) query=\(queryIndex) " +
                    "wall_ms=\(format(duration)) " +
                    "checksum=\(queryChecksum)"
                )
            }
        }

        return targets.indices.map { targetIndex in
            let target = targets[targetIndex]
            let targetSamples = samples[targetIndex]
            let sortedBatches = targetSamples.wallMilliseconds.sorted()
            let median = sortedBatches[sortedBatches.count / 2]
            let sortedCPUBatches = targetSamples.cpuMilliseconds.sorted()
            let cpuMedian = sortedCPUBatches[sortedCPUBatches.count / 2]
            let sortedIndividuals = targetSamples.individualMilliseconds.sorted()
            let p95Index = min(
                sortedIndividuals.count - 1,
                max(
                    0,
                    Int(
                        (Double(sortedIndividuals.count) * 0.95).rounded(.up)
                    ) - 1
                )
            )
            let p95 = sortedIndividuals[p95Index]
            let qps = Double(queries.count) / (median / 1_000.0)
            let recall = averageRecall(
                results: targetSamples.lastResults,
                groundTruth: groundTruth,
                k: k
            )
            return Measurement(
                index: target.name,
                bitWidth: target.bitWidth,
                efSearch: efSearch,
                medianMilliseconds: median / Double(queries.count),
                p95Milliseconds: p95,
                queriesPerSecond: qps,
                cpuMilliseconds: cpuMedian / Double(queries.count),
                cpuQueriesPerSecond: Double(queries.count) /
                (cpuMedian / 1_000),
                recall: recall,
                checksum: targetSamples.checksum,
                qjlScoredCandidates: targetSamples.qjlScoredCandidates.isEmpty
                    ? nil
                    : Self.median(targetSamples.qjlScoredCandidates),
                qjlPrunedCandidates: targetSamples.qjlPrunedCandidates.isEmpty
                    ? nil
                    : Self.median(targetSamples.qjlPrunedCandidates)
            )
        }
    }

    private static func scheduledIndices(
        count: Int,
        offset: Int
    ) -> [Int] {
        guard count > 0 else { return [] }
        let start = offset % count
        return (0..<count).map { (start + $0) % count }
    }

    private static func includesTarget(
        name: String,
        bitWidth: Int?,
        requestedTarget: String?
    ) -> Bool {
        guard let requestedTarget else { return true }
        if name == "hnsw-cosine" {
            return requestedTarget == name
        }
        guard let bitWidth else { return false }
        return requestedTarget == "\(name)-\(bitWidth)"
    }

    private static func measurementLine(
        _ measurement: Measurement
    ) -> String {
        let bits = measurement.bitWidth.map(String.init) ?? "-"
        return "search index=\(measurement.index) bits=\(bits) " +
            "ef=\(measurement.efSearch) " +
            "median_ms_per_query=" +
            "\(format(measurement.medianMilliseconds)) " +
            "p95_ms_per_query=\(format(measurement.p95Milliseconds)) " +
            "qps=\(format(measurement.queriesPerSecond)) " +
            "cpu_ms_per_query=\(format(measurement.cpuMilliseconds)) " +
            "cpu_qps=\(format(measurement.cpuQueriesPerSecond)) " +
            "recall_at_10=\(format(measurement.recall)) " +
            "checksum=\(measurement.checksum)" +
            qjlStatisticsSuffix(
                scoredCandidates: measurement.qjlScoredCandidates,
                prunedCandidates: measurement.qjlPrunedCandidates
            )
    }

    private static func qjlStatisticsSuffix(
        _ statistics: TurboQuantQJLPruningStatistics?
    ) -> String {
        guard let statistics else { return "" }
        return qjlStatisticsSuffix(
            scoredCandidates: statistics.scoredCandidates,
            prunedCandidates: statistics.prunedCandidates
        )
    }

    private static func qjlStatisticsSuffix(
        scoredCandidates: Int?,
        prunedCandidates: Int?
    ) -> String {
        guard let scoredCandidates, let prunedCandidates else { return "" }
        let rate = scoredCandidates > 0
            ? Double(prunedCandidates) / Double(scoredCandidates)
            : 0
        return " qjl_scored=\(scoredCandidates)" +
            " qjl_pruned=\(prunedCandidates)" +
            " qjl_pruning_rate=\(format(rate))"
    }

    private static func median(_ values: [Int]) -> Int {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private static func makeVectors(
        count: Int,
        dimensions: Int,
        seed: UInt64
    ) -> [[Float]] {
        var generator = SplitMix64(seed: seed)
        return (0..<count).map { _ in
            (0..<dimensions).map { _ in generator.nextUnit() }
        }
    }

    private static func cosineGroundTruth(
        vectors: [[Float]],
        queries: [[Float]],
        k: Int
    ) -> [[UInt64]] {
        queries.map { query in
            let queryNorm = query.reduce(Float.zero) {
                $0 + $1 * $1
            }.squareRoot()
            return vectors.enumerated()
                .map { label, vector in
                    let vectorNorm = vector.reduce(Float.zero) {
                        $0 + $1 * $1
                    }.squareRoot()
                    var dot = Float.zero
                    for dimension in 0..<query.count {
                        dot += query[dimension] * vector[dimension]
                    }
                    return (
                        UInt64(label),
                        1.0 - dot / (queryNorm * vectorNorm)
                    )
                }
                .sorted { $0.1 < $1.1 }
                .prefix(k)
                .map(\.0)
        }
    }

    private static func averageRecall(
        results: [[SearchResult]],
        groundTruth: [[UInt64]],
        k: Int
    ) -> Double {
        zip(results, groundTruth).map { result, truth in
            let resultLabels = Set(result.prefix(k).map(\.label))
            let truthLabels = Set(truth.prefix(k))
            return Double(
                resultLabels.intersection(truthLabels).count
            ) / Double(k)
        }.reduce(0, +) / Double(groundTruth.count)
    }

    private static func milliseconds(
        from duration: ContinuousClock.Duration
    ) -> Double {
        let components = duration.components
        return Double(components.seconds) * 1_000 +
            Double(components.attoseconds) / 1_000_000_000_000_000
    }

    private static func processCPUSeconds() -> Double {
        var value = timespec()
        precondition(
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) == 0
        )
        return Double(value.tv_sec) +
            Double(value.tv_nsec) / 1_000_000_000
    }

    private static func swiftVersion() -> String {
        commandOutput(
            executable: "/usr/bin/env",
            arguments: ["swift", "--version"]
        ) ?? "unavailable"
    }

    private static func commandOutput(
        executable: String,
        arguments: [String]
    ) -> String? {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = output
        do {
            try process.run()
            process.waitUntilExit()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            guard process.terminationStatus == 0,
                  let value = String(data: data, encoding: .utf8) else {
                return nil
            }
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }

    private static func commandLineValue(_ value: String) -> String {
        value.split(whereSeparator: \.isWhitespace).joined(separator: "_")
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}
