import Foundation
import Testing
@testable import SwiftHNSW

@Suite(
    "Backend Comparison Benchmarks",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["BACKEND_COMPARISON_BENCHMARK"] != nil)
)
struct BackendComparisonBenchmarkTests {
    private struct BenchmarkCase {
        let name: String
        let dimensions: Int
        let vectorCount: Int
        let queryCount: Int
        let k: Int
        let efSearchValues: [Int]
    }

    private struct SearchMeasurement {
        let efSearch: Int
        let medianSeconds: Double
        let p95Seconds: Double
        let medianQPS: Double
        let medianLatencyMilliseconds: Double
        let p95LatencyMilliseconds: Double
        let recall: Double
    }

    #if HNSWLIB_BACKEND
    private static let backendName = "cxx-hnswlib"
    #else
    private static let backendName = "swift-hnsw-simd"
    #endif

    private var searchIterations: Int {
        let rawValue = ProcessInfo.processInfo.environment["BACKEND_COMPARISON_ITERATIONS"]
        guard let rawValue, let parsed = Int(rawValue) else { return 5 }
        return max(1, parsed)
    }

    @Test("Float32 backend comparison")
    func float32BackendComparison() throws {
        try runFloat32(
            BenchmarkCase(
                name: "float32-l2-10k",
                dimensions: 128,
                vectorCount: 10_000,
                queryCount: 500,
                k: 10,
                efSearchValues: [100, 320]
            )
        )
    }

    @Test("Float32 scale backend comparison")
    func float32ScaleBackendComparison() throws {
        try runFloat32(
            BenchmarkCase(
                name: "float32-l2-50k",
                dimensions: 128,
                vectorCount: 50_000,
                queryCount: 200,
                k: 10,
                efSearchValues: [100, 320, 1_000]
            )
        )
    }

    @Test("Float16 backend comparison")
    func float16BackendComparison() throws {
        try runFloat16(
            BenchmarkCase(
                name: "float16-l2-10k",
                dimensions: 128,
                vectorCount: 10_000,
                queryCount: 500,
                k: 10,
                efSearchValues: [100, 320]
            )
        )
    }

    private func runFloat32(_ benchmark: BenchmarkCase) throws {
        let vectors = deterministicFloatVectors(
            count: benchmark.vectorCount,
            dimensions: benchmark.dimensions,
            seed: 0x1234_5678
        )
        let queries = deterministicFloatVectors(
            count: benchmark.queryCount,
            dimensions: benchmark.dimensions,
            seed: 0x8765_4321
        )
        let labels = labels(count: benchmark.vectorCount)
        let groundTruth = groundTruthFloat32(
            vectors: vectors,
            queries: queries,
            vectorCount: benchmark.vectorCount,
            queryCount: benchmark.queryCount,
            dimensions: benchmark.dimensions,
            k: benchmark.k
        )

        let index = try HNSWIndex<Float>(
            dimensions: benchmark.dimensions,
            maxElements: benchmark.vectorCount,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )

        let buildSeconds = try measureSeconds {
            try vectors.withUnsafeBufferPointer { vectorBuffer in
                try labels.withUnsafeBufferPointer { labelBuffer in
                    _ = try index.addBatch(vectorBuffer, labels: labelBuffer)
                }
            }
        }

        let measurements = try benchmark.efSearchValues.map { efSearch in
            try measureFloat32Search(
                index: index,
                queries: queries,
                groundTruth: groundTruth,
                benchmark: benchmark,
                efSearch: efSearch
            )
        }

        printFloat32Results(
            benchmark: benchmark,
            buildSeconds: buildSeconds,
            measurements: measurements
        )
    }

    private func runFloat16(_ benchmark: BenchmarkCase) throws {
        let vectors = deterministicFloatVectors(
            count: benchmark.vectorCount,
            dimensions: benchmark.dimensions,
            seed: 0x2345_6789
        ).map(Float16.init)
        let queries = deterministicFloatVectors(
            count: benchmark.queryCount,
            dimensions: benchmark.dimensions,
            seed: 0x9876_5432
        ).map(Float16.init)
        let labels = labels(count: benchmark.vectorCount)
        let groundTruth = groundTruthFloat16(
            vectors: vectors,
            queries: queries,
            vectorCount: benchmark.vectorCount,
            queryCount: benchmark.queryCount,
            dimensions: benchmark.dimensions,
            k: benchmark.k
        )

        let index = try HNSWIndex<Float16>(
            dimensions: benchmark.dimensions,
            maxElements: benchmark.vectorCount,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )

        let buildSeconds = try measureSeconds {
            try vectors.withUnsafeBufferPointer { vectorBuffer in
                try labels.withUnsafeBufferPointer { labelBuffer in
                    _ = try index.addBatch(vectorBuffer, labels: labelBuffer)
                }
            }
        }

        let measurements = try benchmark.efSearchValues.map { efSearch in
            try measureFloat16Search(
                index: index,
                queries: queries,
                groundTruth: groundTruth,
                benchmark: benchmark,
                efSearch: efSearch
            )
        }

        printFloat16Results(
            benchmark: benchmark,
            buildSeconds: buildSeconds,
            measurements: measurements
        )
    }

    private func measureFloat32Search(
        index: HNSWIndex<Float>,
        queries: [Float],
        groundTruth: [[UInt64]],
        benchmark: BenchmarkCase,
        efSearch: Int
    ) throws -> SearchMeasurement {
        index.setEfSearch(efSearch)
        _ = try queries.withUnsafeBufferPointer { queryBuffer in
            let firstQuery = UnsafeBufferPointer(start: queryBuffer.baseAddress!, count: benchmark.dimensions)
            return try index.search(firstQuery, k: benchmark.k)
        }

        var lastResults: [[SearchResult]] = []
        var durations: [Double] = []
        durations.reserveCapacity(searchIterations)
        for _ in 0..<searchIterations {
            let seconds = try measureSeconds {
                lastResults = try queries.withUnsafeBufferPointer { queryBuffer in
                    try index.searchBatch(queryBuffer, numQueries: benchmark.queryCount, k: benchmark.k)
                }
            }
            durations.append(seconds)
        }

        return searchMeasurement(
            efSearch: efSearch,
            durations: durations,
            queryCount: benchmark.queryCount,
            results: lastResults,
            groundTruth: groundTruth,
            k: benchmark.k
        )
    }

    private func measureFloat16Search(
        index: HNSWIndex<Float16>,
        queries: [Float16],
        groundTruth: [[UInt64]],
        benchmark: BenchmarkCase,
        efSearch: Int
    ) throws -> SearchMeasurement {
        index.setEfSearch(efSearch)
        _ = try queries.withUnsafeBufferPointer { queryBuffer in
            let firstQuery = UnsafeBufferPointer(start: queryBuffer.baseAddress!, count: benchmark.dimensions)
            return try index.search(firstQuery, k: benchmark.k)
        }

        var lastResults: [[SearchResult]] = []
        var durations: [Double] = []
        durations.reserveCapacity(searchIterations)
        for _ in 0..<searchIterations {
            let seconds = try measureSeconds {
                lastResults = try queries.withUnsafeBufferPointer { queryBuffer in
                    try index.searchBatch(queryBuffer, numQueries: benchmark.queryCount, k: benchmark.k)
                }
            }
            durations.append(seconds)
        }

        return searchMeasurement(
            efSearch: efSearch,
            durations: durations,
            queryCount: benchmark.queryCount,
            results: lastResults,
            groundTruth: groundTruth,
            k: benchmark.k
        )
    }

    private func searchMeasurement(
        efSearch: Int,
        durations: [Double],
        queryCount: Int,
        results: [[SearchResult]],
        groundTruth: [[UInt64]],
        k: Int
    ) -> SearchMeasurement {
        let medianSeconds = percentile(durations, percentile: 0.5)
        let p95Seconds = percentile(durations, percentile: 0.95)
        let medianLatency = medianSeconds * 1_000 / Double(queryCount)
        let p95Latency = p95Seconds * 1_000 / Double(queryCount)
        return SearchMeasurement(
            efSearch: efSearch,
            medianSeconds: medianSeconds,
            p95Seconds: p95Seconds,
            medianQPS: Double(queryCount) / max(medianSeconds, .leastNonzeroMagnitude),
            medianLatencyMilliseconds: medianLatency,
            p95LatencyMilliseconds: p95Latency,
            recall: averageRecall(results: results, groundTruth: groundTruth, k: k)
        )
    }

    private func printFloat32Results(
        benchmark: BenchmarkCase,
        buildSeconds: Double,
        measurements: [SearchMeasurement]
    ) {
        printResults(
            scalar: "Float32",
            benchmark: benchmark,
            buildSeconds: buildSeconds,
            measurements: measurements
        )
    }

    private func printFloat16Results(
        benchmark: BenchmarkCase,
        buildSeconds: Double,
        measurements: [SearchMeasurement]
    ) {
        printResults(
            scalar: "Float16",
            benchmark: benchmark,
            buildSeconds: buildSeconds,
            measurements: measurements
        )
    }

    private func printResults(
        scalar: String,
        benchmark: BenchmarkCase,
        buildSeconds: Double,
        measurements: [SearchMeasurement]
    ) {
        let buildVectorsPerSecond = Double(benchmark.vectorCount) / max(buildSeconds, .leastNonzeroMagnitude)
        print("")
        print("backend_comparison backend=\(Self.backendName) scalar=\(scalar) case=\(benchmark.name) n=\(benchmark.vectorCount) d=\(benchmark.dimensions) q=\(benchmark.queryCount) k=\(benchmark.k) iterations=\(searchIterations) build_seconds=\(format(buildSeconds)) build_vectors_per_second=\(format(buildVectorsPerSecond))")
        for measurement in measurements {
            print("backend_comparison backend=\(Self.backendName) scalar=\(scalar) case=\(benchmark.name) ef_search=\(measurement.efSearch) search_seconds_median=\(format(measurement.medianSeconds)) search_seconds_p95=\(format(measurement.p95Seconds)) qps_median=\(format(measurement.medianQPS)) latency_ms_median=\(format(measurement.medianLatencyMilliseconds)) latency_ms_p95=\(format(measurement.p95LatencyMilliseconds)) recall_at_\(benchmark.k)=\(format(measurement.recall))")
        }
    }

    private func deterministicFloatVectors(count: Int, dimensions: Int, seed: UInt64) -> [Float] {
        var state = seed
        var values: [Float] = []
        values.reserveCapacity(count * dimensions)
        for row in 0..<count {
            for column in 0..<dimensions {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let mixed = state ^ UInt64(row &* 0x9E37) ^ UInt64(column &* 0x85EB)
                let bucket = Int((mixed >> 33) % 2_001) - 1_000
                values.append(Float(bucket) / 1_000)
            }
        }
        return values
    }

    private func labels(count: Int) -> [UInt64] {
        (0..<count).map(UInt64.init)
    }

    private func groundTruthFloat32(
        vectors: [Float],
        queries: [Float],
        vectorCount: Int,
        queryCount: Int,
        dimensions: Int,
        k: Int
    ) -> [[UInt64]] {
        (0..<queryCount).map { queryIndex in
            topK(
                vectorCount: vectorCount,
                k: k
            ) { vectorIndex in
                squaredL2Float32(
                    vectors: vectors,
                    queries: queries,
                    vectorIndex: vectorIndex,
                    queryIndex: queryIndex,
                    dimensions: dimensions
                )
            }
        }
    }

    private func groundTruthFloat16(
        vectors: [Float16],
        queries: [Float16],
        vectorCount: Int,
        queryCount: Int,
        dimensions: Int,
        k: Int
    ) -> [[UInt64]] {
        (0..<queryCount).map { queryIndex in
            topK(
                vectorCount: vectorCount,
                k: k
            ) { vectorIndex in
                squaredL2Float16(
                    vectors: vectors,
                    queries: queries,
                    vectorIndex: vectorIndex,
                    queryIndex: queryIndex,
                    dimensions: dimensions
                )
            }
        }
    }

    private func topK(
        vectorCount: Int,
        k: Int,
        distance: (Int) -> Float
    ) -> [UInt64] {
        var scored: [(label: UInt64, distance: Float)] = []
        scored.reserveCapacity(vectorCount)
        for vectorIndex in 0..<vectorCount {
            scored.append((UInt64(vectorIndex), distance(vectorIndex)))
        }
        scored.sort {
            if $0.distance == $1.distance {
                return $0.label < $1.label
            }
            return $0.distance < $1.distance
        }
        return scored.prefix(k).map(\.label)
    }

    private func squaredL2Float32(
        vectors: [Float],
        queries: [Float],
        vectorIndex: Int,
        queryIndex: Int,
        dimensions: Int
    ) -> Float {
        var sum: Float = 0
        let vectorOffset = vectorIndex * dimensions
        let queryOffset = queryIndex * dimensions
        for dimension in 0..<dimensions {
            let diff = vectors[vectorOffset + dimension] - queries[queryOffset + dimension]
            sum += diff * diff
        }
        return sum
    }

    private func squaredL2Float16(
        vectors: [Float16],
        queries: [Float16],
        vectorIndex: Int,
        queryIndex: Int,
        dimensions: Int
    ) -> Float {
        var sum: Float = 0
        let vectorOffset = vectorIndex * dimensions
        let queryOffset = queryIndex * dimensions
        for dimension in 0..<dimensions {
            let diff = Float(vectors[vectorOffset + dimension]) - Float(queries[queryOffset + dimension])
            sum += diff * diff
        }
        return sum
    }

    private func averageRecall(results: [[SearchResult]], groundTruth: [[UInt64]], k: Int) -> Double {
        zip(results, groundTruth).map { result, truth in
            let found = Set(result.prefix(k).map(\.label))
            let expected = Set(truth.prefix(k))
            return Double(found.intersection(expected).count) / Double(k)
        }.reduce(0, +) / Double(groundTruth.count)
    }

    private func measureSeconds(_ body: () throws -> Void) rethrows -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        try body()
        return CFAbsoluteTimeGetCurrent() - start
    }

    private func percentile(_ values: [Double], percentile: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let boundedPercentile = min(max(percentile, 0), 1)
        let index = Int((Double(sorted.count - 1) * boundedPercentile).rounded(.up))
        return sorted[min(index, sorted.count - 1)]
    }

    private func format(_ value: Double) -> String {
        String(format: "%.6f", value)
    }
}
