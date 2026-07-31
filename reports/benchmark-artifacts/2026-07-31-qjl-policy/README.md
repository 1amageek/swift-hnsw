# Adaptive QJL pruning policy artifact

This artifact records the final benchmark-only policy comparison for commit
`d8b1bc1`. It validates policy activation and result identity on both ends of
the production workload dimensions; it is not a pooled latency claim because
the policy processes were run sequentially while system load varied.

## Environment

| Field | Value |
| --- | --- |
| Host | Apple M4 Max, 14 CPU cores, 36 GB memory |
| Toolchain | `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a` |
| Dataset | 10,000 deterministic vectors; 50 queries; `k=10`; `M=16`; seed `42` |
| Dimensions | 128 and 768 |
| Policies | `automatic` (`efSearch >= 100`), `always`, `never` |

The benchmark uses `automatic-stats`, `always-stats`, and `never-stats` so the
QJL candidate counters are observable. All three policies produced identical
labels, distances, Recall@10, and checksums at every search point.

## Observed pruning

| Dimensions | `efSearch` | QJL scored | Automatic pruned | Automatic rate | Always rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10 | 20,438 | 0 | 0.000 | 0.625 |
| 128 | 20 | 33,503 | 0 | 0.000 | 0.589 |
| 128 | 40 | 57,365 | 0 | 0.000 | 0.563 |
| 128 | 60 | 77,124 | 0 | 0.000 | 0.547 |
| 128 | 100 | 112,892 | 59,500 | 0.527 | 0.527 |
| 128 | 200 | 189,063 | 95,929 | 0.507 | 0.507 |
| 128 | 400 | 294,187 | 141,318 | 0.480 | 0.480 |
| 768 | 10 | 21,884 | 0 | 0.000 | 0.242 |
| 768 | 20 | 36,352 | 0 | 0.000 | 0.211 |
| 768 | 40 | 62,035 | 0 | 0.000 | 0.190 |
| 768 | 60 | 84,233 | 0 | 0.000 | 0.176 |
| 768 | 100 | 125,330 | 20,441 | 0.163 | 0.163 |
| 768 | 200 | 203,195 | 29,261 | 0.144 | 0.144 |
| 768 | 400 | 315,463 | 38,070 | 0.121 | 0.121 |

`never` pruned zero candidates in all rows. The full process summaries remain
in `/tmp` from the measurement session; this repository artifact records the
durable counters and policy contract rather than host-noisy raw timings.
