# TurboQuant Benchmark Report

**Date**: 2026-03-25T08:15:16Z
**Parameters**: n=1000, k=10, queries=100, M=16, efConstruction=200, efSearch=200



### TurboQuant vs Float32 vs Float16 (efSearch=200)

| Dim | Type | Build (s) | QPS | Latency (ms) | Recall@10 | Vec Mem | Compression |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 768 | Float32 | 1.94 | 634 | 1.578 | 99.7% | 2.9 MB | 1.0x |
| 768 | TQ-4b | 2.19 | 509 | 1.963 | 83.2% | 500 KB | 6.0x |
| 1536 | Float32 | 3.17 | 379 | 2.638 | 100.0% | 5.9 MB | 1.0x |
| 1536 | TQ-4b | 4.07 | 293 | 3.413 | 80.8% | 1000 KB | 6.0x |

