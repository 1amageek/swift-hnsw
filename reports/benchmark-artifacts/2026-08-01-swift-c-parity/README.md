# TurboQuant Swift/C Backend Decision

This artifact measures the production four-bit scoring path against the direct
Swift implementation and an independently linked scalar C reference. The
decision is performance-based: ARM64 uses the NEON kernel, while regular WASM,
Embedded WASM, and other linked targets use the scalar C kernel.

```text
Four-bit production scoring
    -> ARM64 Apple: C NEON
    -> other linked targets: C scalar
    -> Swift direct: differential and benchmark reference
```

## Fixed environment

- Base commit: `238aeb6` (`1.1.2`), with the uncommitted backend follow-up applied
- Toolchain: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a`
- Toolchain identifier: `org.swift.64202607231a`
- Compiler commit: `ef761e567dc94ee`
- Regular SDK: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a_wasm`
- Embedded SDK: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a_wasm-embedded`

## Kernel methodology

- Dimension: 768; vectors: 64; queries: 16
- Candidate counts: 16, 24, 32, 48, 64
- Eleven interleaved samples per backend and candidate count
- 128 measurement repetitions per Native sample and four per WASM sample
- Three independent processes per target
- Reported throughput is the median of the three process medians
- Every backend score is compared within `1e-4` before timing
- The scalar C reference and the Swift direct implementation use the same four
  accumulators, packed-byte order, pair tail, and odd-dimension tail
- The production scalar C function consumes the existing Float centroid table;
  it does not reconstruct centroids inside the scored call

## Kernel result

| Target | Candidates | Production scores/s | Swift direct scores/s | Production speedup |
| --- | ---: | ---: | ---: | ---: |
| Native ARM64 NEON | 16 | 3,348,671 | 1,796,848 | 1.864x |
| Native ARM64 NEON | 24 | 2,684,899 | 1,229,039 | 2.185x |
| Native ARM64 NEON | 32 | 3,411,889 | 1,722,495 | 1.981x |
| Native ARM64 NEON | 48 | 5,719,464 | 2,296,503 | 2.491x |
| Native ARM64 NEON | 64 | 7,411,950 | 3,252,148 | 2.279x |
| regular WASM C scalar | 16 | 180,864 | 169,686 | 1.066x |
| regular WASM C scalar | 24 | 173,002 | 163,973 | 1.055x |
| regular WASM C scalar | 32 | 165,335 | 154,968 | 1.067x |
| regular WASM C scalar | 48 | 146,455 | 140,312 | 1.044x |
| regular WASM C scalar | 64 | 155,312 | 147,837 | 1.051x |
| Embedded WASM C scalar | 16 | 181,791 | 171,332 | 1.061x |
| Embedded WASM C scalar | 24 | 179,328 | 171,711 | 1.044x |
| Embedded WASM C scalar | 32 | 177,228 | 169,782 | 1.044x |
| Embedded WASM C scalar | 48 | 179,032 | 170,355 | 1.051x |
| Embedded WASM C scalar | 64 | 178,832 | 170,249 | 1.050x |

The independently linked scalar C reference was within 0.960x–0.995x of the
regular/Embedded WASM production C result across these cases. This confirms
that the selected production function follows the measured scalar C behavior;
the remaining difference is process and code-layout variance.

## Native end-to-end search

Two independent processes built separate but deterministic 10,000-vector,
768-dimensional four-bit indexes for the production and Swift-direct backends.
Each `efSearch` value used fifty queries and five interleaved batch samples.
Recall, result checksum, packed representation, graph parameters, and query
inputs were identical.

| efSearch | Production QPS | Swift direct QPS | Production speedup | Recall@10 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 22,542 | 11,288 | 1.997x | 0.070 |
| 20 | 13,411 | 6,394 | 2.097x | 0.142 |
| 40 | 8,164 | 3,878 | 2.105x | 0.208 |
| 60 | 6,046 | 2,832 | 2.135x | 0.288 |
| 100 | 4,076 | 1,929 | 2.114x | 0.414 |
| 200 | 2,444 | 1,177 | 2.077x | 0.594 |
| 400 | 1,521 | 746 | 2.039x | 0.766 |

The fixed snapshot still traps during generic HNSW construction on both WASM
targets, so the WASM evidence stops at the production scoring boundary. Native
end-to-end search establishes the whole-index impact where the lifecycle runs.

## Reproduction

```bash
TOOLCHAINS=org.swift.64202607231a swift run --disable-sandbox -c release \
  --package-path Benchmarks/HNSWReferenceComparison \
  TurboQuantKernelComparison

TOOLCHAINS=org.swift.64202607231a swift run --disable-sandbox -c release \
  --swift-sdk swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a_wasm \
  --package-path Benchmarks/HNSWReferenceComparison \
  TurboQuantKernelComparison

TOOLCHAINS=org.swift.64202607231a swift run --disable-sandbox -c release \
  --swift-sdk swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a_wasm-embedded \
  --package-path Benchmarks/HNSWReferenceComparison \
  TurboQuantKernelComparison

TURBOQUANT_BENCHMARK_CASE=d768 \
TURBOQUANT_MSE_ONLY=1 \
TURBOQUANT_BENCHMARK_TARGET=turboquant-mse-4 \
TURBOQUANT_MSE_BACKEND_COMPARISON=1 \
TOOLCHAINS=org.swift.64202607231a \
swift run --disable-sandbox -c release \
  --package-path Benchmarks/HNSWReferenceComparison \
  TurboQuantComparison
```

Raw program output:

- `final-native-run-1.log` through `final-native-run-3.log`
- `final-wasm-run-1.log` through `final-wasm-run-3.log`
- `final-embedded-wasm-run-1.log` through `final-embedded-wasm-run-3.log`
- `search-native-run-1.log` and `search-native-run-2.log`
