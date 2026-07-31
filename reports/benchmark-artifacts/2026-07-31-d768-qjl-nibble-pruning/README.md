# QJL nibble-table and pruning verification

This artifact records the clean-source Release verification run for commit
`0b12406c36c8915de33dc2fcc2dcfd3e8c9483ec`.

## Configuration

- Date: 2026-07-31
- Host: Apple M4 Max, 14 CPU cores, 36 GB memory
- Toolchain: `org.swift.64202607231a`
- Swift snapshot: `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-23-a`
- Workload: 10,000 deterministic unit vectors, 768 dimensions, 50 queries,
  `k=10`, `M=16`, `efConstruction=200`, seed `42`
- Measurement: isolated target, five batches after warmup
- Source state: clean

The Product QJL table uses 32 entries per packed byte (two 16-entry nibble
rows). Its d=768 query table is 12 KiB, versus 96 KiB for the previous
256-entry byte table. The Product path also applies a conservatively rounded
absolute-sum lower bound before evaluating the QJL signed lookup.

## Results

| efSearch | Product recall | Product QPS | Product p95 ms/query | MSE4 recall | MSE4 QPS | MSE4 p95 ms/query |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.082 | 11,144.960 | 0.125 | 0.070 | 22,911.261 | 0.089 |
| 20 | 0.150 | 7,795.700 | 0.185 | 0.142 | 13,479.211 | 0.158 |
| 40 | 0.230 | 5,044.433 | 0.277 | 0.208 | 6,950.719 | 0.203 |
| 60 | 0.296 | 3,582.229 | 0.425 | 0.288 | 5,235.899 | 0.240 |
| 100 | 0.414 | 2,403.808 | 0.505 | 0.414 | 3,615.612 | 0.371 |
| 200 | 0.604 | 1,420.745 | 0.996 | 0.594 | 2,175.056 | 0.545 |
| 400 | 0.758 | 976.070 | 1.307 | 0.766 | 1,370.944 | 1.091 |

The full raw logs were kept outside the repository during execution:

- `/tmp/turboquant-2026-07-31-qjl-nibble-pruning-committed.log`
- `/tmp/turboquant-2026-07-31-qjl-nibble-pruning-mse4-committed.log`

These are isolated post-change measurements, not paired old/new samples. No
causal speedup claim is made without a matched old-QJL control run.
