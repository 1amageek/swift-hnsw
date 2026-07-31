# Pure Swift Unsafe QJL Eight-Byte Reduction Benchmark

This follow-up measures the QJL lookup loop after the 1.1.0 release. The
production path remains pure Swift: it borrows packed signs and loads eight
initialized bytes with `UnsafeRawPointer.loadUnaligned(as: UInt64.self)`, then
uses the existing four Float32 accumulators in the same reduction order. A
bounded four-byte `UInt32` loop handles the remaining groups before the tail.

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Host | Apple M4 Max, arm64, 14 CPU cores, 36 GB |
| Toolchain | `org.swift.64202607231a` |
| Swift snapshot | `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a` |
| Workload | 10,000 vectors, 50 queries, `k=10`, seed `42` |
| Search target | TurboQuant Product 5-bit, `TURBOQUANT_QJL_PRUNING=never` |
| Baseline | 1.1.0 pure Swift four-byte `UInt32` loop |
| Candidate | Eight-byte `UInt64` loop plus four-byte fallback |

## Paired d=768 comparison

Each cycle ran the baseline and candidate concurrently, exchanging process
order across cycles. CPU-QPS ratios are `candidate / baseline`. Recall@10 and
checksums were identical in every row.

| `efSearch` | cycle 1 | cycle 2 | cycle 3 | median |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.992x | 0.995x | 1.099x | 0.995x |
| 20 | 0.943x | 1.110x | 1.071x | **1.071x** |
| 40 | 1.034x | 1.070x | 1.050x | **1.050x** |
| 60 | 1.052x | 1.054x | 1.005x | **1.052x** |
| 100 | 1.058x | 1.034x | 1.019x | **1.034x** |
| 200 | 1.045x | 1.193x | 1.059x | **1.059x** |
| 400 | 1.020x | 1.027x | 1.069x | **1.027x** |

The low-`efSearch` result is noisy because graph and heap work dominate the
short searches. From `efSearch=20` through `400`, every median is positive;
this is evidence for a modest end-to-end improvement, not a universal fixed
multiplier across hardware and datasets.

## Paired d=128 control

The same three cycles preserved recall and checksums. Median CPU-QPS ratios
were 1.056x, 1.075x, 1.056x, 1.014x, 1.023x, 1.052x, and 1.045x for
`efSearch` 10, 20, 40, 60, 100, 200, and 400 respectively. No target-specific
fallback or accuracy change was observed.

## Safety and compatibility

The differential test now covers the eight-byte path, the four-byte fallback,
and tails after an eight-byte group (`8`, `9`, and `15` packed bytes). The
existing ASan/TSan coverage applies to the same synchronous pointer borrow.
The algorithm and archive representation are unchanged, so this is a patch
update after 1.1.0 rather than a new archive format.
