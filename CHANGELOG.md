# Changelog

## 1.1.4 — 2026-08-02

- Reject non-finite HNSW vectors and invalid cosine norms before mutation or
  search, and reject the same values when restoring archives.
- Reject duplicate labels in legacy flat archives instead of constructing an
  inconsistent graph.
- Make `SearchResult` ordering deterministic for equal distances by comparing
  labels.
- Add a continuous-integration gate for optimized macOS and iOS builds, the
  full macOS test suite, and Address/Thread Sanitizer runs.
- Correct the documented minimum deployment targets to macOS 26 and iOS 26.
- Add borrowed graph-archive resource inspection with conservative retained and
  restore-working payload estimates so constrained hosts can reject restoration
  before allocation while retaining allocator headroom.

## 1.1.3 — 2026-08-01

- Added an allocation-free direct Swift four-bit reference with differential
  coverage for odd, even, vector-boundary, and tail dimensions.
- Added a scalar C kernel that consumes the existing Float centroid table and
  retained it as the production WASM and Embedded WASM backend after it measured
  1.044x to 1.067x faster than the direct Swift reference.
- Retained ARM64 NEON after it measured 1.864x to 2.491x faster at the kernel
  boundary and 1.997x to 2.135x faster in repeated 10,000-vector d=768 searches
  with identical Recall@10 and result checksums.
- Added reproducible three-process Native ARM64, regular WASM, and Embedded WASM
  kernel comparisons plus two Native end-to-end search runs.
- Corrected the evidence scope: the fixed toolchain verifies the production
  kernel boundary on Native ARM64, regular WASM, and Embedded WASM. Full WASM
  index lifecycle support is not claimed because the fixed snapshot traps in
  generic `HNSWIndex<Float>` construction before the kernel is reached. The
  same toolchain also lacks the package's Float16 APIs on macOS x86_64.

## 1.1.2 — 2026-08-01

- Enabled the portable `CTurboQuantKernels` four-bit MSE path on regular WASM
  and Embedded WASM while retaining ARM64 NEON.
- Verified C compilation and target linking with the fixed Swift 6.4 snapshot;
  the archive representation is unchanged.

## 1.1.1 — 2026-08-01

- Reduced QJL lookup-loop load overhead with a bounded pure Swift unaligned
  `UInt64` read and a `UInt32` fallback while preserving Float32 reduction
  order and result compatibility.
- Added paired d=768 and d=128 benchmark evidence for the follow-up.
- Kept the archive representation unchanged; TurboQuant 1.1.0 archives remain
  loadable by 1.1.1.

## 1.1.0 — 2026-07-31

- Added the production `TurboQuantIndex` MSE and QJL inner-product objectives,
  structured Hadamard and Haar rotation strategies, bounded workspaces, and
  typed archive validation.
- Added adaptive QJL pruning statistics to the benchmarking SPI.
- Kept the QJL reduction in pure Swift with an unsafe, bounded unaligned word
  load; no QJL C ABI is part of the production path.
- Preserved the existing Swift concurrency contract across Native, WASM, and
  Embedded targets.

### Migration

TurboQuant archives written by 1.0.x use an older representation and are not
loadable by 1.1.0. Rebuild the index from source vectors before upgrading.
Regular `HNSWIndex` archives remain on their existing format.
