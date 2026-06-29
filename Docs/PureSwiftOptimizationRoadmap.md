# Pure Swift Optimization Roadmap

## Goal

SwiftHNSW should prove whether the Pure Swift backend can replace the C++ backend for production workloads. The C++ backend remains an explicit native accelerator until the Pure Swift backend reaches measured parity on the target workloads.

## Design Contract

| Area | Contract |
| --- | --- |
| Public API | Keep `HNSWIndex<Float>` and `HNSWIndex<Float16>` as the stable user-facing types. |
| WASM | The Pure Swift backend must build with the Swift WebAssembly SDK. |
| Runtime vectors | Store comparison vectors in type-specific contiguous arenas: `[Float]` for Float32 and `[Float16]` for Float16. |
| Graph connections | Store graph connections in a dedicated fixed-slot connection store, with `UInt32` internal IDs and a direct level-0 layout. |
| Persistence | Preserve the versioned serialized graph format unless a migration is explicitly introduced. |
| C++ backend | Keep it optional until benchmark evidence shows the Pure Swift backend is consistently sufficient. |

```mermaid
flowchart LR
    A["User API: [Float] / [Float16]"] --> B["Boundary conversion"]
    B --> C["Typed contiguous comparison storage"]
    C --> D["HNSW graph search"]
    D --> E["UInt32 fixed-slot connection store"]
    E --> F["Direct level-0 layout"]
    D --> G["Versioned binary snapshot"]
```

## Milestones

| Milestone | Scope | Exit Criteria | Status |
| --- | --- | --- | --- |
| M0 Baseline | Keep a repeatable Pure Swift vs C++ benchmark suite. | Build/search latency, median search latency, p95 search latency, and recall are recorded for Float32 and Float16. | Done |
| M1 Hot Path Cleanup | Remove closure heaps, use pointer distance kernels, and keep lower-bound state in the search loop. | Pure Swift tests, C++ trait tests, and WASM build pass. | Done |
| M2 Graph Storage | Replace nested graph arrays with a dedicated connection store. | No runtime `[[[Int]]]` graph storage remains in the Pure Swift backend. | Done |
| M2.5 Internal ID and Level-0 Layout | Use `UInt32` internal IDs, direct level-0 neighbor offsets, and separate upper-level storage. | Level-0 reads do not require upper-slot lookup; overflow is rejected at initialization. | Done |
| M2.6 Float16 Arena | Store Float16 indexes in `[Float16]` and use Float16 SIMD distance kernels. | Float16 debug diagnostics show zero Float arena usage; serialization roundtrip still passes. | Done |
| M2.7 Metric Dispatch | Select L2 vs inner-product distance strategy once per search/build path. | Distance hot loops do not switch on `DistanceMetric` per comparison. | Done |
| M2.8 Search Scratch Arena | Reuse caller-scoped scratch buffers for candidate and nearest heaps without truncating graph exploration. | Candidate queue capacity is based on node count, nearest queue capacity is based on `ef`, and tests pass. | Done |
| M2.9 Caller-Owned Output | Provide a search API that writes into caller-owned result storage. | `search(_:k:into:)` is available on Swift and C++ backends and preserves ordering/count semantics. | Done |
| M3 Snapshot Layout | Align persisted snapshots with runtime storage without breaking existing loads. | Existing graph snapshots still load; new snapshots remain versioned. | In progress |
| M4 Database Integration | Ensure database-framework stores vector payloads as binary and feeds SwiftHNSW without tuple expansion. | Flat, HNSW, IVF, and PQ vector payloads use Float32 little-endian bytes. | In progress |
| M5 Parity Decision | Compare Pure Swift and C++ across target workloads. | Pure Swift is within the accepted performance envelope, or C++ remains optional. | Pending |
| M6 Release Gate | Run full tests and publish benchmark report. | Swift tests, C++ tests, WASM build, and dependent package tests pass. | Pending |

## Performance Decision Rule

Pure Swift can make the C++ backend unnecessary only if it is consistently close to the C++ backend on production workloads:

| Metric | Target |
| --- | --- |
| Search p50 / p95 | Within 1.2x of C++ for target dimensions and corpus sizes. |
| Build time | Within 1.2x of C++ or justified by better portability. |
| Recall | No regression against the current graph algorithm at the same parameters. |
| WASM | Pure Swift remains the default portable backend. |

If these conditions are not met, the C++ backend should stay as an optional native acceleration path.

## Current Implementation Notes

- The Pure Swift backend stores Float32 comparison vectors in `comparisonStorage: [Float]`.
- The Pure Swift backend stores Float16 comparison vectors in `halfComparisonStorage: [Float16]`.
- Float16 distance kernels convert SIMD lanes to Float for accumulation without expanding the stored vector arena.
- Candidate queues use fixed heap views over reusable scratch buffers instead of per-search heap arrays.
- Search candidate queues are sized by node count rather than `ef`; only the nearest-result heap is bounded by `ef`.
- HNSW graph traversal uses metric-specific distance computers so the hot loop does not branch on `DistanceMetric` for every comparison.
- Graph connections are stored in `HNSWConnectionStore`, which uses `UInt32` internal IDs, direct level-0 slots, and compact upper-level slots.
- Search traversal reads neighbor storage ranges directly to avoid per-neighbor slot lookup and optional branching.
- Search uses a bare path when there are no deleted entries and caches the layer lower bound locally instead of repeatedly reading heap top state.
- Query search returns the requested top-k without sorting the full `efSearch` candidate set.
- Latency-sensitive callers can use `search(_:k:into:)` to reuse result buffers.
- Backend comparison benchmarks record median and p95 search latency across repeated search iterations.
- Serialization still writes the existing graph format for compatibility.

## Latest Local Benchmark Snapshot

Environment: local Apple Silicon debug workspace, release test build, Swift 6.3.1. These historical rows are single-run measurements; new backend comparison output records median and p95 search latency before making a backend removal decision.

| Case | Backend | Build Seconds | efSearch | Latency ms/query | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Float32 10k x 128 | Pure Swift | 1.431549 | 100 | 0.078152 | 0.829600 |
| Float32 10k x 128 | C++ hnswlib | 1.199120 | 100 | 0.059270 | 0.831600 |
| Float32 10k x 128 | Pure Swift | 1.431549 | 320 | 0.221124 | 0.980800 |
| Float32 10k x 128 | C++ hnswlib | 1.199120 | 320 | 0.159512 | 0.981000 |
| Float32 50k x 128 | Pure Swift | 12.695528 | 100 | 0.143470 | 0.545000 |
| Float32 50k x 128 | C++ hnswlib | 10.822952 | 100 | 0.115450 | 0.547000 |
| Float32 50k x 128 | Pure Swift | 12.695528 | 320 | 0.434075 | 0.840000 |
| Float32 50k x 128 | C++ hnswlib | 10.822952 | 320 | 0.340625 | 0.839000 |
| Float32 50k x 128 | Pure Swift | 12.695528 | 1000 | 1.214405 | 0.963500 |
| Float32 50k x 128 | C++ hnswlib | 10.822952 | 1000 | 1.001830 | 0.968000 |
| Float16 10k x 128 | Pure Swift | 1.381076 | 100 | 0.071308 | 0.838600 |
| Float16 10k x 128 | C++ hnswlib | 1.248810 | 100 | 0.059938 | 0.836400 |
| Float16 10k x 128 | Pure Swift | 1.381076 | 320 | 0.202762 | 0.981200 |
| Float16 10k x 128 | C++ hnswlib | 1.248810 | 320 | 0.157752 | 0.980400 |

Interpretation:

- Pure Swift is close but not consistently within the parity target across all measured cases.
- C++ remains faster in this single run.
- Pure Swift remains close enough to continue optimization, but this run does not justify removing the C++ backend.
- Float16 now uses half storage in the Swift backend; repeated measurements are still required before a final parity decision.
- Metric-specific dispatch is implemented; the latest single-run comparison shows only small and mixed movement, so repeated benchmark medians remain required.
- The C++ backend should remain optional until repeated benchmark runs and larger corpus sizes confirm the decision.
