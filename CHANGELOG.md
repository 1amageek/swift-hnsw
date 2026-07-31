# Changelog

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
