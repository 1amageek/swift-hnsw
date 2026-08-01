# TurboQuant Four-Bit Backend Selection

> Historical intermediate artifact. The final matched comparison retains C on
> every verified production target and supersedes the measurements below. See
> [`../2026-08-01-swift-c-parity`](../2026-08-01-swift-c-parity/README.md).

This artifact compares the explicit portable C implementation with the pure
Swift coordinate/packed lookup reference. Every d=768 score is checked within
`1e-4` before timing. Each reported value is the median of eleven interleaved
samples over sixteen queries.

## Fixed environment

- Base commit: `238aeb6` (`1.1.2`), with the uncommitted review fix applied.
- Toolchain: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a`
- Toolchain identifier: `org.swift.64202607231a`
- Compiler commit: `ef761e567dc94ee`
- Regular SDK: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a_wasm`
- Embedded SDK: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a_wasm-embedded`
- Dimension: 768; packed vectors: 64; queries: 16; iterations: 11
- Pure Swift policy: coordinate table below 32 candidates, packed table at 32+

## Median scores per second

| Target | Candidates | portable C | pure Swift | C speedup |
| --- | ---: | ---: | ---: | ---: |
| Native ARM64 | 16 | 2,968,116 | 1,914,620 | 1.55x |
| Native ARM64 | 24 | 2,995,133 | 2,952,892 | 1.01x |
| Native ARM64 | 32 | 2,989,065 | 176,666 | 16.92x |
| Native ARM64 | 48 | 2,983,498 | 255,539 | 11.68x |
| Native ARM64 | 64 | 2,977,826 | 302,515 | 9.84x |
| regular WASM | 16 | 83,316 | 53,736 | 1.55x |
| regular WASM | 24 | 82,776 | 62,033 | 1.33x |
| regular WASM | 32 | 108,851 | 9,190 | 11.84x |
| regular WASM | 48 | 121,397 | 15,280 | 7.94x |
| regular WASM | 64 | 97,886 | 15,998 | 6.12x |
| Embedded WASM | 16 | 97,126 | 58,293 | 1.67x |
| Embedded WASM | 24 | 97,501 | 74,585 | 1.31x |
| Embedded WASM | 32 | 84,403 | 7,344 | 11.49x |
| Embedded WASM | 48 | 84,114 | 10,833 | 7.76x |
| Embedded WASM | 64 | 98,213 | 16,453 | 5.97x |

The measurements support keeping the platform C entry point as the production
four-bit backend on all verified targets. The pure Swift implementation remains
an explicit differential and benchmarking reference; there is no silent runtime
fallback.

## Repeated-process variance

A final-source regular-WASM rerun showed that the 24-candidate case is close
enough to be sensitive to transient host load. Four independent process runs
produced the following portable-C speedup over pure Swift:

| Run | C speedup at 24 candidates |
| --- | ---: |
| Recorded raw artifact | 1.33x |
| Final-source rerun 1 | 0.91x |
| Final-source rerun 2 | 1.31x |
| Final-source rerun 3 | 1.33x |

The median process-level result favors portable C, while the isolated reversal
prevents a claim that C wins every small-candidate run. At 32 through 64
candidates, avoiding the Swift packed-table construction remains a clear and
material C advantage. This evidence supports one deterministic production C
backend instead of an unstable runtime threshold.

## Reproduction

The commands below describe the intermediate source recorded by this artifact;
the current Unreleased source no longer exposes the portable C benchmarking
backend. The raw samples remain preserved for auditability.

```bash
swift run --disable-sandbox -c release \
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
```

Raw program output:

- [`native.log`](native.log)
- [`wasm.log`](wasm.log)
- [`embedded-wasm.log`](embedded-wasm.log)
