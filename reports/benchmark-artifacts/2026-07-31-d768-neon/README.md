# ARM64 NEON TurboQuant Benchmark Summary

This directory records the process-level summaries from the optimized
four-bit TurboQuant rerun on 2026-07-31. The benchmark used the optimized
production search implementation based on commit
`989fe137d5e8ab5aabe8a308eae52143a297edf8`. A subsequent verification-only
change made the platform capability internally injectable and added a forced
fallback differential test. The public initializer still supplies the same
compiled platform capability, and the measured search kernel and search path
did not change.

## Environment

- Apple M4 Max, 14 CPU cores, 36 GB memory
- arm64 macOS
- Swift `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a`
- deterministic synthetic unit vectors
- 10,000 indexed vectors, 768 dimensions, 50 queries, Recall@10
- `M=16`, `efConstruction=200`, seed `42`

## Schedule

Each representation ran in a separate process. The order was rotated to
reduce systematic thermal and system-load bias:

1. Float32, MSE 4-bit, Product 5-bit
2. MSE 4-bit, Product 5-bit, Float32
3. Product 5-bit, Float32, MSE 4-bit

Each process warmed every search point and measured five batches of 50
queries. `summary.csv` retains every process-level CPU-QPS summary and build
time observed in the three cycles. The report uses the median of the three
process summaries. Per-query raw samples were not retained for this optimized
rerun, so no p95 latency claim is made from these records.

## Matched-Recall Result

All three representations reached `0.414` Recall@10 at `efSearch=100`.

| Representation | Median CPU QPS | Ratio of independent medians |
| --- | ---: | ---: |
| Float32 | 1,873.150 | 1.000x |
| TurboQuant MSE 4-bit | 3,326.901 | 1.776x |
| TurboQuant Product 5-bit | 1,789.293 | 0.955x |

The primary comparison pairs representations within each cycle:

| Representation | Cycle 1 | Cycle 2 | Cycle 3 | Median paired ratio |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant MSE 4-bit / Float32 | 1.515x | 2.144x | 0.708x | 1.515x |
| TurboQuant Product 5-bit / Float32 | 1.034x | 0.837x | 0.955x | 0.955x |

System load varied materially between cycles. Process CPU time and rotated
order reduce descheduling and order bias, but they do not remove frequency or
system-wide interference. MSE won two of three paired cycles, but lost the
third; the median paired result is positive evidence rather than a stable
capacity guarantee. Build-time differences are not used as evidence of the
search-kernel speedup.
