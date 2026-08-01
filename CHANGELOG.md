# Changelog

## 1.1.2 — 2026-08-01

- Enabled the portable `CTurboQuantKernels` four-bit MSE path on Native
  non-ARM, regular WASM, and Embedded WASM while retaining ARM64 NEON.
- Verified C compilation and target linking with the fixed Swift 6.4 snapshot;
  the archive representation is unchanged.

## 1.1.1 — 2026-08-01

- Reduced QJL lookup-loop load overhead with a bounded pure Swift unaligned
  `UInt64` read and a `UInt32` fallback while preserving Float32 reduction
  order and result compatibility.
- Activated the portable `CTurboQuantKernels` four-bit MSE path for Native
  non-ARM, regular WASM, and Embedded WASM while retaining ARM64 NEON
  specialization.
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
