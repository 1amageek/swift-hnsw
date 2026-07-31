# Isolated 10K × 768 TurboQuant Benchmark Artifact

This directory contains every raw sample from the isolated-process control
described in `reports/turboquant_benchmark.md`.

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-07-31 |
| Hardware | Apple M4 Max (`Mac16,6`), 36 GiB |
| Toolchain | `org.swift.64202607231a` |
| Compiler commit | `ef761e567dc94ee` |
| Base revision recorded by the runner | `a1f7ec0a847fd871a32af13b7c89a2e5b7ee67bd` |
| Source state recorded by the runner | `dirty` |
| Dataset | 10,000 vectors, 768 dimensions, 50 queries, `k = 10` |

The benchmark was executed before the enclosing source and report changes were
committed. The runner therefore records the previous revision and a dirty
working tree. The enclosing commit contains the exact production and benchmark
source used by these runs; only this artifact and its report were added after
measurement.

## Process Schedule

Each process retained exactly one index representation. Process order rotated
across three cycles:

```text
cycle 1: HNSW -> MSE 4-bit -> Product 5-bit
cycle 2: MSE 4-bit -> Product 5-bit -> HNSW
cycle 3: Product 5-bit -> HNSW -> MSE 4-bit
```

Each log contains five batch samples and fifty individual-query samples for
every `efSearch` value. No sample was rejected. Reported CPU QPS pools the
fifteen batch CPU-time samples per representation and `efSearch`, then uses the
median. Reported p95 latency pools the 150 individual-query wall-time samples.

## Raw Logs

| File | SHA-256 |
| --- | --- |
| `cycle1-hnsw.log` | `45e4238958348838817973c6d83c53ac9ddc71cddcae1904053400d8f341036b` |
| `cycle1-mse4.log` | `0e43cebccc8220aac0a5be4018639f397f751739b09bfb23a9f9a083a7c13a45` |
| `cycle1-product5.log` | `e87c436f6de81e002a58b0574cfa5e4b4a342e00d95d518febea1280ebcdb29e` |
| `cycle2-hnsw.log` | `0729107cbe27a40504d4cfa27d0d9a633bbe42836fcdbc1311308ac3891a6455` |
| `cycle2-mse4.log` | `685a3e0b6ee76c0319b16dd82f81c781272a70dc6d97f9cd329842a22d994554` |
| `cycle2-product5.log` | `d8e2e23c3bc7359af49f7b19ff3a1529513cd02d3e2a41ed6cd005ab1f53582d` |
| `cycle3-hnsw.log` | `16f858fc30438c92357d8b9b13e85fca5263eac2faf5197a2e94b23156833ade` |
| `cycle3-mse4.log` | `ddcab03a266c36f718b33424017b8036fde49462fb92c38446fab18dcba02136` |
| `cycle3-product5.log` | `ae5da2acc0748efe635b501cf903edb3496fe6adfed204132c34f2319134e53c` |
