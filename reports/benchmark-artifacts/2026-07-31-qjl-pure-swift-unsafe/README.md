# Pure Swift Unsafe QJL Reduction Benchmark

This artifact records the final QJL implementation decision for the release
candidate. The QJL lookup reduction stays in Swift and uses a bounded
`UnsafeRawPointer.loadUnaligned(as: UInt32.self)` to read four packed sign
bytes per loop. No QJL C symbol or C module boundary is involved.

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-07-31 |
| Host | Apple M4 Max, arm64, 14 CPU cores, 36 GB |
| Toolchain | `org.swift.64202607231a` |
| Swift snapshot | `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a` |
| Workload | 10,000 vectors, 768 dimensions, 50 queries, `k=10`, seed `42` |
| Search target | TurboQuant Product 5-bit, `TURBOQUANT_QJL_PRUNING=never` |
| Source state | Dirty release candidate; source and test changes were uncommitted during measurement |

## Reduction contract

```text
Array owners
    -> UnsafeBufferPointer borrow
        -> count/overflow validation
            -> unaligned UInt32 sign load
                -> four Float32 accumulators
                    -> Float32 result
```

The loop reads only when four initialized bytes remain. `UInt32(littleEndian:)`
preserves byte order on every target, while the tail loop handles 1–3 bytes.
The pointer is borrowed synchronously, does not escape, and is not retained or
mutated. ASan and TSan checks passed for this path.

## Paired end-to-end comparison

Each row compares a release binary containing the pointer implementation with
the previous pure-Swift byte-load binary. The two processes ran concurrently;
process order was exchanged across cycles. CPU QPS ratios are `current / old`.
Checksums and Recall@10 were identical in every row.

| `efSearch` | d=768 cycle 1 | cycle 2 | cycle 3 | median |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1.240x | 1.140x | 1.153x | **1.153x** |
| 20 | 1.288x | 1.007x | 0.978x | 1.007x |
| 40 | 1.146x | 0.980x | 1.002x | 1.002x |
| 60 | 0.948x | 1.022x | 0.994x | 0.994x |
| 100 | 1.030x | 0.990x | 1.002x | 1.002x |
| 200 | 1.028x | 0.981x | 1.000x | 1.000x |
| 400 | 0.999x | 0.997x | 1.022x | 0.999x |

The result is a hot-path improvement signal at low `efSearch`, not a claim
that the whole HNSW search is universally faster. At larger `efSearch`, graph
traversal and heap work dominate, so the observed difference is within host
variance. Two complete d=128 cycles showed the same recall and checksums but
did not establish a stable speedup; no d=128 speed claim is made.

## Rejected alternatives

- A QJL C function was rejected at the user's direction; the production path
  has no QJL C ABI or C target dependency.
- A static-inline C header experiment was rejected because its measurements
  were unstable under host contention and it violated the pure-Swift boundary.
- A speculative SIMD/gather rewrite was not adopted without target-specific
  assembly and stable allocation/latency evidence.

The durable correctness coverage is the bitwise differential test in
`Tests/SwiftHNSWTests/TurboQuantTests.swift`, covering empty, tail, threshold,
and 1,024-byte packed sizes.
