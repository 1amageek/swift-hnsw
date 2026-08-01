@_spi(Benchmarking)
import SwiftHNSW
import CTurboQuantBenchmarkReference
import Synchronization
#if hasFeature(Embedded)
import WASILibc
#endif

@main
enum TurboQuantKernelComparison {
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

    private enum Backend {
        case library(TurboQuantFourBitBackend)
        case cScalarReference
    }

    private struct BackendSpec {
        let name: String
        let backend: Backend
    }

    private struct Sample {
        let elapsedNanoseconds: UInt64
        let checksum: UInt64
    }

    private static let dimensions = 768
    private static let maximumCandidateCount = 64
    private static let queryCount = 16
    private static let iterations = 11
#if os(WASI)
    private static let measurementRepetitions = 4
#else
    private static let measurementRepetitions = 128
#endif
    private static let candidateCounts = [16, 24, 32, 48, 64]
    private static let backends = [
        BackendSpec(name: "platform", backend: .library(.platform)),
        BackendSpec(name: "swift-direct", backend: .library(.swiftDirect)),
        BackendSpec(name: "c-scalar-reference", backend: .cScalarReference),
    ]

    static func main() {
        print(
            "turboquant_kernel_comparison target=\(targetName()) " +
            "vectors=\(maximumCandidateCount) dimensions=\(dimensions) " +
            "queries=\(queryCount) iterations=\(iterations) " +
            "measurement_repetitions=\(measurementRepetitions)"
        )
        let harness = TurboQuantFourBitBenchmarkHarness(dimensions: dimensions)
        let vectors = makeNormalizedVectors(
            count: maximumCandidateCount,
            dimensions: dimensions,
            seed: 0x1234_5678
        )
        let queries = makeNormalizedVectors(
            count: queryCount,
            dimensions: dimensions,
            seed: 0x8765_4321
        )
        let packedVectors = vectors.map(harness.quantize)
        let cReference = CScalarReferenceHarness(
            dimensions: dimensions,
            packedSize: harness.packedSize,
            centroids: harness.centroidTable()
        )

        for candidateCount in candidateCounts {
            let activeVectors = Array(packedVectors.prefix(candidateCount))
            let usesPackedLookup = candidateCount >= 32
            verifyEquivalentResults(
                harness: harness,
                cReference: cReference,
                packedVectors: activeVectors,
                queries: queries,
                usesPackedLookup: usesPackedLookup
            )

            var samples = [[Sample]](
                repeating: [],
                count: backends.count
            )
            for trial in 0..<iterations {
                for backendIndex in scheduledIndices(
                    count: backends.count,
                    offset: trial
                ) {
                    samples[backendIndex].append(
                        measure(
                            harness: harness,
                            cReference: cReference,
                            packedVectors: activeVectors,
                            queries: queries,
                            backend: backends[backendIndex].backend,
                            usesPackedLookup: usesPackedLookup
                        )
                    )
                }
            }

            for backendIndex in backends.indices {
                for (trial, sample) in samples[backendIndex].enumerated() {
                    print(
                        "sample backend=\(backends[backendIndex].name) " +
                        "candidates=\(candidateCount) trial=\(trial) " +
                        "elapsed_ns=\(sample.elapsedNanoseconds) " +
                        "checksum=\(sample.checksum)"
                    )
                }
                var sorted = [UInt64]()
                sorted.reserveCapacity(samples[backendIndex].count)
                for sample in samples[backendIndex] {
                    sorted.append(sample.elapsedNanoseconds)
                }
                sorted.sort()
                let median = sorted[sorted.count / 2]
                let scoresPerSecond = Double(
                    candidateCount * queryCount * measurementRepetitions
                )
                    * 1_000_000_000.0 / Double(median)
                print(
                    "summary backend=\(backends[backendIndex].name) " +
                    "candidates=\(candidateCount) median_ns=\(median) " +
                    "scores_per_second=\(scoresPerSecond)"
                )
            }
        }
    }

    private static func verifyEquivalentResults(
        harness: TurboQuantFourBitBenchmarkHarness,
        cReference: CScalarReferenceHarness,
        packedVectors: [[UInt8]],
        queries: [[Float]],
        usesPackedLookup: Bool
    ) {
        var platform = [Float]()
        var direct = [Float]()
        var cScalar = [Float]()
        var lookup = [Float]()
        for query in queries {
            harness.score(
                packedVectors,
                query: query,
                backend: .platform,
                usesPackedLookup: usesPackedLookup,
                into: &platform
            )
            harness.score(
                packedVectors,
                query: query,
                backend: .swiftDirect,
                usesPackedLookup: usesPackedLookup,
                into: &direct
            )
            cReference.score(
                packedVectors,
                query: query,
                into: &cScalar
            )
            harness.score(
                packedVectors,
                query: query,
                backend: .swiftLookup,
                usesPackedLookup: usesPackedLookup,
                into: &lookup
            )
            for index in platform.indices {
                precondition(
                    abs(platform[index] - direct[index]) < 1e-4,
                    "Platform and direct Swift returned different scores"
                )
                precondition(
                    abs(direct[index] - cScalar[index]) < 1e-4,
                    "Swift and C scalar implementations returned different scores"
                )
                precondition(
                    abs(platform[index] - lookup[index]) < 1e-4,
                    "Four-bit backends returned different scores"
                )
            }
        }
    }

    private static func measure(
        harness: TurboQuantFourBitBenchmarkHarness,
        cReference: CScalarReferenceHarness,
        packedVectors: [[UInt8]],
        queries: [[Float]],
        backend: Backend,
        usesPackedLookup: Bool
    ) -> Sample {
        var scores = [Float]()
        var checksum: UInt64 = 0
#if hasFeature(Embedded)
        let start = monotonicNanoseconds()
#else
        let start = ContinuousClock.now
#endif
        for _ in 0..<measurementRepetitions {
            for query in queries {
                switch backend {
                case .library(let libraryBackend):
                    harness.score(
                        packedVectors,
                        query: query,
                        backend: libraryBackend,
                        usesPackedLookup: usesPackedLookup,
                        into: &scores
                    )
                case .cScalarReference:
                    cReference.score(
                        packedVectors,
                        query: query,
                        into: &scores
                    )
                }
                for score in scores {
                    checksum &+= UInt64(score.bitPattern)
                }
            }
        }
#if hasFeature(Embedded)
        let elapsedNanoseconds = monotonicNanoseconds() - start
#else
        let elapsedNanoseconds = nanoseconds(start.duration(to: .now))
#endif
        return Sample(elapsedNanoseconds: elapsedNanoseconds, checksum: checksum)
    }

    private final class CScalarReferenceHarness: Sendable {
        private let dimensions: Int
        private let packedSize: Int
        private let centroids: [Float]
        private let access: Mutex<Void>

        init(dimensions: Int, packedSize: Int, centroids: [Float]) {
            precondition(dimensions > 0, "Dimensions must be positive")
            precondition(packedSize == (dimensions + 1) / 2)
            precondition(centroids.count == 16)
            self.dimensions = dimensions
            self.packedSize = packedSize
            self.centroids = centroids
            self.access = Mutex(())
        }

        /// Scores through the benchmark-only C implementation with the same lock granularity.
        ///
        /// All owners remain retained throughout each synchronous borrow. Validated counts
        /// bound every access, the buffers do not overlap, and no pointer escapes the C call.
        func score(
            _ packedVectors: [[UInt8]],
            query: [Float],
            into scores: inout [Float]
        ) {
            precondition(query.count == dimensions)
            precondition(
                packedVectors.allSatisfy { $0.count == packedSize }
            )
            if scores.count != packedVectors.count {
                scores = [Float](repeating: 0, count: packedVectors.count)
            }
            access.withLock { _ in
                query.withUnsafeBufferPointer { queryBuffer in
                    centroids.withUnsafeBufferPointer { centroidBuffer in
                        for index in packedVectors.indices {
                            scores[index] = packedVectors[index]
                                .withUnsafeBufferPointer { packedBuffer in
                                    swift_hnsw_benchmark_four_bit_inner_product(
                                        packedBuffer.baseAddress!,
                                        queryBuffer.baseAddress!,
                                        dimensions,
                                        centroidBuffer.baseAddress!
                                    )
                                }
                        }
                    }
                }
            }
        }
    }

    private static func makeNormalizedVectors(
        count: Int,
        dimensions: Int,
        seed: UInt64
    ) -> [[Float]] {
        var generator = SplitMix64(seed: seed)
        return (0..<count).map { _ in
            var vector = (0..<dimensions).map { _ in generator.nextUnit() }
            var magnitudeSquared: Float = 0
            for value in vector {
                magnitudeSquared += value * value
            }
            let inverseMagnitude = 1 / magnitudeSquared.squareRoot()
            for index in vector.indices {
                vector[index] *= inverseMagnitude
            }
            return vector
        }
    }

    private static func scheduledIndices(count: Int, offset: Int) -> [Int] {
        let start = offset % count
        return (0..<count).map { (start + $0) % count }
    }

#if hasFeature(Embedded)
    private static func monotonicNanoseconds() -> UInt64 {
        var value: __wasi_timestamp_t = 0
        precondition(
            __wasi_clock_time_get(1, 1, &value) == 0
        )
        return UInt64(value)
    }
#else
    private static func nanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        precondition(components.seconds >= 0 && components.attoseconds >= 0)
        return UInt64(components.seconds) * 1_000_000_000
            + UInt64(components.attoseconds / 1_000_000_000)
    }
#endif

    private static func targetName() -> String {
#if hasFeature(Embedded)
        "embedded-wasm"
#elseif os(WASI)
        "wasm"
#else
        "native"
#endif
    }
}
