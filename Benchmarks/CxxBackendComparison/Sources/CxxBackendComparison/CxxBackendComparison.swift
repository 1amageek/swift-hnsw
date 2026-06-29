import Foundation
import SwiftHNSW
import hnswlib

@main
struct CxxBackendComparison {
    fileprivate struct BenchmarkCase {
        let name: String
        let dimensions: Int
        let vectorCount: Int
        let queryCount: Int
        let k: Int
        let efSearchValues: [Int]
    }

    private struct Measurement {
        let backend: String
        let scalar: String
        let benchmark: BenchmarkCase
        let buildSeconds: Double
        let searchMeasurements: [SearchMeasurement]
    }

    private struct SearchMeasurement {
        let efSearch: Int
        let medianSeconds: Double
        let p95Seconds: Double
        let recall: Double
    }

    static func main() throws {
        let iterations = max(
            1,
            Int(ProcessInfo.processInfo.environment["BACKEND_COMPARISON_ITERATIONS"] ?? "") ?? 3
        )
        let cases = [
            BenchmarkCase(
                name: "float32-l2-10k",
                dimensions: 128,
                vectorCount: 10_000,
                queryCount: 500,
                k: 10,
                efSearchValues: [100, 320]
            ),
            BenchmarkCase(
                name: "float32-l2-50k",
                dimensions: 128,
                vectorCount: 50_000,
                queryCount: 200,
                k: 10,
                efSearchValues: [100, 320, 1_000]
            ),
        ]

        for benchmark in cases {
            try runFloat32(benchmark, iterations: iterations)
        }

        try runFloat16(
            BenchmarkCase(
                name: "float16-l2-10k",
                dimensions: 128,
                vectorCount: 10_000,
                queryCount: 500,
                k: 10,
                efSearchValues: [100, 320]
            ),
            iterations: iterations
        )
    }

    private static func runFloat32(_ benchmark: BenchmarkCase, iterations: Int) throws {
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
            labels: labels,
            benchmark: benchmark
        )

        let swiftIndex = try HNSWIndex<Float>(
            dimensions: benchmark.dimensions,
            maxElements: benchmark.vectorCount,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )
        let swiftBuildSeconds = try measureSeconds {
            try vectors.withUnsafeBufferPointer { vectorBuffer in
                try labels.withUnsafeBufferPointer { labelBuffer in
                    _ = try swiftIndex.addBatch(vectorBuffer, labels: labelBuffer)
                }
            }
        }
        let swiftMeasurement = try measureSwiftFloat32(
            index: swiftIndex,
            queries: queries,
            groundTruth: groundTruth,
            benchmark: benchmark,
            iterations: iterations
        )
        printMeasurement(
            Measurement(
                backend: "swift-hnsw",
                scalar: "Float32",
                benchmark: benchmark,
                buildSeconds: swiftBuildSeconds,
                searchMeasurements: swiftMeasurement
            )
        )

        let cxxIndex = try CxxFloat32Index(
            dimensions: benchmark.dimensions,
            maxElements: benchmark.vectorCount
        )
        let cxxBuildSeconds = try measureSeconds {
            try vectors.withUnsafeBufferPointer { vectorBuffer in
                try labels.withUnsafeBufferPointer { labelBuffer in
                    let added = cxxIndex.addBatch(vectors: vectorBuffer, labels: labelBuffer)
                    guard added == benchmark.vectorCount else {
                        throw BenchmarkError.backendFailure("C++ added \(added) vectors")
                    }
                }
            }
        }
        let cxxMeasurement = measureCxxFloat32(
            index: cxxIndex,
            queries: queries,
            groundTruth: groundTruth,
            benchmark: benchmark,
            iterations: iterations
        )
        printMeasurement(
            Measurement(
                backend: "cxx-hnswlib-reference",
                scalar: "Float32",
                benchmark: benchmark,
                buildSeconds: cxxBuildSeconds,
                searchMeasurements: cxxMeasurement
            )
        )
    }

    private static func runFloat16(_ benchmark: BenchmarkCase, iterations: Int) throws {
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
            labels: labels,
            benchmark: benchmark
        )

        let swiftIndex = try HNSWIndex<Float16>(
            dimensions: benchmark.dimensions,
            maxElements: benchmark.vectorCount,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200)
        )
        let swiftBuildSeconds = try measureSeconds {
            try vectors.withUnsafeBufferPointer { vectorBuffer in
                try labels.withUnsafeBufferPointer { labelBuffer in
                    _ = try swiftIndex.addBatch(vectorBuffer, labels: labelBuffer)
                }
            }
        }
        let swiftMeasurement = try measureSwiftFloat16(
            index: swiftIndex,
            queries: queries,
            groundTruth: groundTruth,
            benchmark: benchmark,
            iterations: iterations
        )
        printMeasurement(
            Measurement(
                backend: "swift-hnsw",
                scalar: "Float16",
                benchmark: benchmark,
                buildSeconds: swiftBuildSeconds,
                searchMeasurements: swiftMeasurement
            )
        )

        let cxxIndex = try CxxFloat16Index(
            dimensions: benchmark.dimensions,
            maxElements: benchmark.vectorCount
        )
        let cxxBuildSeconds = try measureSeconds {
            try vectors.withUnsafeBufferPointer { vectorBuffer in
                try labels.withUnsafeBufferPointer { labelBuffer in
                    let added = cxxIndex.addBatch(vectors: vectorBuffer, labels: labelBuffer)
                    guard added == benchmark.vectorCount else {
                        throw BenchmarkError.backendFailure("C++ added \(added) vectors")
                    }
                }
            }
        }
        let cxxMeasurement = measureCxxFloat16(
            index: cxxIndex,
            queries: queries,
            groundTruth: groundTruth,
            benchmark: benchmark,
            iterations: iterations
        )
        printMeasurement(
            Measurement(
                backend: "cxx-hnswlib-reference",
                scalar: "Float16",
                benchmark: benchmark,
                buildSeconds: cxxBuildSeconds,
                searchMeasurements: cxxMeasurement
            )
        )
    }

    private static func measureSwiftFloat32(
        index: HNSWIndex<Float>,
        queries: [Float],
        groundTruth: [[UInt64]],
        benchmark: BenchmarkCase,
        iterations: Int
    ) throws -> [SearchMeasurement] {
        try benchmark.efSearchValues.map { efSearch in
            index.setEfSearch(efSearch)
            var lastResults: [[SearchResult]] = []
            var durations: [Double] = []
            durations.reserveCapacity(iterations)
            for _ in 0..<iterations {
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
                results: lastResults,
                groundTruth: groundTruth,
                k: benchmark.k
            )
        }
    }

    private static func measureSwiftFloat16(
        index: HNSWIndex<Float16>,
        queries: [Float16],
        groundTruth: [[UInt64]],
        benchmark: BenchmarkCase,
        iterations: Int
    ) throws -> [SearchMeasurement] {
        try benchmark.efSearchValues.map { efSearch in
            index.setEfSearch(efSearch)
            var lastResults: [[SearchResult]] = []
            var durations: [Double] = []
            durations.reserveCapacity(iterations)
            for _ in 0..<iterations {
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
                results: lastResults,
                groundTruth: groundTruth,
                k: benchmark.k
            )
        }
    }

    private static func measureCxxFloat32(
        index: CxxFloat32Index,
        queries: [Float],
        groundTruth: [[UInt64]],
        benchmark: BenchmarkCase,
        iterations: Int
    ) -> [SearchMeasurement] {
        benchmark.efSearchValues.map { efSearch in
            index.setEfSearch(efSearch)
            var lastResults: [[SearchResult]] = []
            var durations: [Double] = []
            durations.reserveCapacity(iterations)
            for _ in 0..<iterations {
                let seconds = measureSeconds {
                    lastResults = queries.withUnsafeBufferPointer { queryBuffer in
                        index.searchBatch(queries: queryBuffer, numQueries: benchmark.queryCount, k: benchmark.k)
                    }
                }
                durations.append(seconds)
            }
            return searchMeasurement(
                efSearch: efSearch,
                durations: durations,
                results: lastResults,
                groundTruth: groundTruth,
                k: benchmark.k
            )
        }
    }

    private static func measureCxxFloat16(
        index: CxxFloat16Index,
        queries: [Float16],
        groundTruth: [[UInt64]],
        benchmark: BenchmarkCase,
        iterations: Int
    ) -> [SearchMeasurement] {
        benchmark.efSearchValues.map { efSearch in
            index.setEfSearch(efSearch)
            var lastResults: [[SearchResult]] = []
            var durations: [Double] = []
            durations.reserveCapacity(iterations)
            for _ in 0..<iterations {
                let seconds = measureSeconds {
                    lastResults = queries.withUnsafeBufferPointer { queryBuffer in
                        index.searchBatch(queries: queryBuffer, numQueries: benchmark.queryCount, k: benchmark.k)
                    }
                }
                durations.append(seconds)
            }
            return searchMeasurement(
                efSearch: efSearch,
                durations: durations,
                results: lastResults,
                groundTruth: groundTruth,
                k: benchmark.k
            )
        }
    }

    private static func searchMeasurement(
        efSearch: Int,
        durations: [Double],
        results: [[SearchResult]],
        groundTruth: [[UInt64]],
        k: Int
    ) -> SearchMeasurement {
        let sortedDurations = durations.sorted()
        let median = sortedDurations[sortedDurations.count / 2]
        let p95Index = min(sortedDurations.count - 1, Int((Double(sortedDurations.count) * 0.95).rounded(.up)) - 1)
        return SearchMeasurement(
            efSearch: efSearch,
            medianSeconds: median,
            p95Seconds: sortedDurations[p95Index],
            recall: recall(results: results, groundTruth: groundTruth, k: k)
        )
    }

    private static func printMeasurement(_ measurement: Measurement) {
        let buildRate = Double(measurement.benchmark.vectorCount) / max(measurement.buildSeconds, .leastNonzeroMagnitude)
        print(
            "backend_comparison backend=\(measurement.backend) scalar=\(measurement.scalar) " +
            "case=\(measurement.benchmark.name) n=\(measurement.benchmark.vectorCount) " +
            "d=\(measurement.benchmark.dimensions) q=\(measurement.benchmark.queryCount) " +
            "k=\(measurement.benchmark.k) build_seconds=\(format(measurement.buildSeconds)) " +
            "build_vectors_per_second=\(format(buildRate))"
        )

        for search in measurement.searchMeasurements {
            let medianLatency = search.medianSeconds / Double(measurement.benchmark.queryCount) * 1_000
            let p95Latency = search.p95Seconds / Double(measurement.benchmark.queryCount) * 1_000
            let qps = Double(measurement.benchmark.queryCount) / max(search.medianSeconds, .leastNonzeroMagnitude)
            print(
                "backend_comparison backend=\(measurement.backend) scalar=\(measurement.scalar) " +
                "case=\(measurement.benchmark.name) ef_search=\(search.efSearch) " +
                "search_seconds_median=\(format(search.medianSeconds)) " +
                "search_seconds_p95=\(format(search.p95Seconds)) " +
                "qps_median=\(format(qps)) latency_ms_median=\(format(medianLatency)) " +
                "latency_ms_p95=\(format(p95Latency)) recall_at_\(measurement.benchmark.k)=\(format(search.recall))"
            )
        }
    }
}

private final class CxxFloat32Index {
    private let dimensions: Int
    private let space: HNSWSpaceHandle
    private let index: HNSWIndexHandle

    init(dimensions: Int, maxElements: Int) throws {
        self.dimensions = dimensions
        guard let space = hnsw_create_l2_space(dimensions) else {
            throw BenchmarkError.backendFailure("Failed to create C++ Float32 L2 space")
        }
        guard let index = hnsw_create_index(space, maxElements, 16, 200, 100, false) else {
            hnsw_destroy_space(space)
            throw BenchmarkError.backendFailure("Failed to create C++ Float32 index")
        }
        self.space = space
        self.index = index
    }

    deinit {
        hnsw_destroy_index(index)
        hnsw_destroy_space(space)
    }

    func setEfSearch(_ efSearch: Int) {
        hnsw_set_ef(index, efSearch)
    }

    func addBatch(
        vectors: UnsafeBufferPointer<Float>,
        labels: UnsafeBufferPointer<UInt64>
    ) -> Int {
        Int(hnsw_add_points_batch(index, vectors.baseAddress!, labels.baseAddress!, labels.count, dimensions))
    }

    func searchBatch(
        queries: UnsafeBufferPointer<Float>,
        numQueries: Int,
        k: Int
    ) -> [[SearchResult]] {
        var labels = [UInt64](repeating: 0, count: numQueries * k)
        var distances = [Float](repeating: 0, count: numQueries * k)
        _ = labels.withUnsafeMutableBufferPointer { labelBuffer in
            distances.withUnsafeMutableBufferPointer { distanceBuffer in
                hnsw_search_knn_batch(
                    index,
                    queries.baseAddress!,
                    numQueries,
                    dimensions,
                    Int32(k),
                    labelBuffer.baseAddress!,
                    distanceBuffer.baseAddress!
                )
            }
        }
        return buildBatchResults(labels: labels, distances: distances, numQueries: numQueries, k: k)
    }
}

private final class CxxFloat16Index {
    private let dimensions: Int
    private let space: HNSWSpaceHandle
    private let index: HNSWIndexHandle

    init(dimensions: Int, maxElements: Int) throws {
        self.dimensions = dimensions
        guard let space = hnsw_create_l2_space_f16(dimensions) else {
            throw BenchmarkError.backendFailure("Failed to create C++ Float16 L2 space")
        }
        guard let index = hnsw_create_index(space, maxElements, 16, 200, 100, false) else {
            hnsw_destroy_space(space)
            throw BenchmarkError.backendFailure("Failed to create C++ Float16 index")
        }
        self.space = space
        self.index = index
    }

    deinit {
        hnsw_destroy_index(index)
        hnsw_destroy_space(space)
    }

    func setEfSearch(_ efSearch: Int) {
        hnsw_set_ef(index, efSearch)
    }

    func addBatch(
        vectors: UnsafeBufferPointer<Float16>,
        labels: UnsafeBufferPointer<UInt64>
    ) -> Int {
        vectors.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: vectors.count) { vectorBits in
            Int(hnsw_add_points_batch_f16(index, vectorBits, labels.baseAddress!, labels.count, dimensions))
        }
    }

    func searchBatch(
        queries: UnsafeBufferPointer<Float16>,
        numQueries: Int,
        k: Int
    ) -> [[SearchResult]] {
        var labels = [UInt64](repeating: 0, count: numQueries * k)
        var distances = [Float](repeating: 0, count: numQueries * k)
        queries.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: queries.count) { queryBits in
            _ = labels.withUnsafeMutableBufferPointer { labelBuffer in
                distances.withUnsafeMutableBufferPointer { distanceBuffer in
                    hnsw_search_knn_batch_f16(
                        index,
                        queryBits,
                        numQueries,
                        dimensions,
                        Int32(k),
                        labelBuffer.baseAddress!,
                        distanceBuffer.baseAddress!
                    )
                }
            }
        }
        return buildBatchResults(labels: labels, distances: distances, numQueries: numQueries, k: k)
    }
}

private enum BenchmarkError: Error {
    case backendFailure(String)
}

private func deterministicFloatVectors(count: Int, dimensions: Int, seed: UInt64) -> [Float] {
    var state = seed
    var output: [Float] = []
    output.reserveCapacity(count * dimensions)
    for _ in 0..<(count * dimensions) {
        state = state &* 6_364_136_223_846_793_005 &+ 1
        let value = UInt32(truncatingIfNeeded: state >> 32)
        let normalized = Float(value) / Float(UInt32.max)
        output.append(normalized * 2 - 1)
    }
    return output
}

private func labels(count: Int) -> [UInt64] {
    (0..<count).map(UInt64.init)
}

private func groundTruthFloat32(
    vectors: [Float],
    queries: [Float],
    labels: [UInt64],
    benchmark: CxxBackendComparison.BenchmarkCase
) -> [[UInt64]] {
    var truth: [[UInt64]] = []
    truth.reserveCapacity(benchmark.queryCount)
    vectors.withUnsafeBufferPointer { vectorBuffer in
        queries.withUnsafeBufferPointer { queryBuffer in
            for queryIndex in 0..<benchmark.queryCount {
                let query = UnsafeBufferPointer(
                    start: queryBuffer.baseAddress! + queryIndex * benchmark.dimensions,
                    count: benchmark.dimensions
                )
                var topK: [SearchResult] = []
                topK.reserveCapacity(benchmark.k)
                for vectorIndex in 0..<benchmark.vectorCount {
                    let vector = UnsafeBufferPointer(
                        start: vectorBuffer.baseAddress! + vectorIndex * benchmark.dimensions,
                        count: benchmark.dimensions
                    )
                    insertTopKSearchResult(
                        SearchResult(
                            label: labels[vectorIndex],
                            distance: squaredL2Distance(query, vector)
                        ),
                        into: &topK,
                        limit: benchmark.k
                    )
                }
                truth.append(topK.map(\.label))
            }
        }
    }
    return truth
}

private func groundTruthFloat16(
    vectors: [Float16],
    queries: [Float16],
    labels: [UInt64],
    benchmark: CxxBackendComparison.BenchmarkCase
) -> [[UInt64]] {
    var truth: [[UInt64]] = []
    truth.reserveCapacity(benchmark.queryCount)
    vectors.withUnsafeBufferPointer { vectorBuffer in
        queries.withUnsafeBufferPointer { queryBuffer in
            for queryIndex in 0..<benchmark.queryCount {
                let query = UnsafeBufferPointer(
                    start: queryBuffer.baseAddress! + queryIndex * benchmark.dimensions,
                    count: benchmark.dimensions
                )
                var topK: [SearchResult] = []
                topK.reserveCapacity(benchmark.k)
                for vectorIndex in 0..<benchmark.vectorCount {
                    let vector = UnsafeBufferPointer(
                        start: vectorBuffer.baseAddress! + vectorIndex * benchmark.dimensions,
                        count: benchmark.dimensions
                    )
                    insertTopKSearchResult(
                        SearchResult(
                            label: labels[vectorIndex],
                            distance: squaredL2Distance(query, vector)
                        ),
                        into: &topK,
                        limit: benchmark.k
                    )
                }
                truth.append(topK.map(\.label))
            }
        }
    }
    return truth
}

private func insertTopKSearchResult(
    _ result: SearchResult,
    into results: inout [SearchResult],
    limit: Int
) {
    guard limit > 0 else { return }
    if results.count < limit {
        results.append(result)
        results.sort()
        return
    }
    guard let last = results.last, result.distance < last.distance else { return }
    results[results.count - 1] = result
    results.sort()
}

private func recall(results: [[SearchResult]], groundTruth: [[UInt64]], k: Int) -> Double {
    guard !results.isEmpty else { return 0 }
    var matches = 0
    for queryIndex in results.indices {
        let truth = Set(groundTruth[queryIndex])
        for result in results[queryIndex].prefix(k) where truth.contains(result.label) {
            matches += 1
        }
    }
    return Double(matches) / Double(results.count * k)
}

private func buildBatchResults(
    labels: [UInt64],
    distances: [Float],
    numQueries: Int,
    k: Int
) -> [[SearchResult]] {
    var output: [[SearchResult]] = []
    output.reserveCapacity(numQueries)
    for queryIndex in 0..<numQueries {
        var queryResults: [SearchResult] = []
        queryResults.reserveCapacity(k)
        for resultIndex in 0..<k {
            let offset = queryIndex * k + resultIndex
            let label = labels[offset]
            let distance = distances[offset]
            guard resultIndex == 0 || label != 0 || distance != 0 else { continue }
            queryResults.append(SearchResult(label: label, distance: distance))
        }
        output.append(queryResults)
    }
    return output
}

private func measureSeconds(_ body: () throws -> Void) rethrows -> Double {
    let clock = ContinuousClock()
    let start = clock.now
    try body()
    let duration = start.duration(to: clock.now)
    let components = duration.components
    return Double(components.seconds) + Double(components.attoseconds) / 1_000_000_000_000_000_000
}

private func format(_ value: Double) -> String {
    String(format: "%.6f", value)
}

private func squaredL2Distance(
    _ lhs: UnsafeBufferPointer<Float>,
    _ rhs: UnsafeBufferPointer<Float>
) -> Float {
    var sum: Float = 0
    for index in 0..<lhs.count {
        let diff = lhs[index] - rhs[index]
        sum += diff * diff
    }
    return sum
}

private func squaredL2Distance(
    _ lhs: UnsafeBufferPointer<Float16>,
    _ rhs: UnsafeBufferPointer<Float16>
) -> Float {
    var sum: Float = 0
    for index in 0..<lhs.count {
        let diff = Float(lhs[index]) - Float(rhs[index])
        sum += diff * diff
    }
    return sum
}
