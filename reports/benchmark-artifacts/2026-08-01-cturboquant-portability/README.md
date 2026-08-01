# CTurboQuantKernels Portability Verification

> Historical 1.1.2 artifact. The latest follow-up remeasures C and Swift under
> matched conditions and retains C on every verified production target. See the
> [Swift/C backend decision](../2026-08-01-swift-c-parity/README.md).

`CTurboQuantKernels` is an active production dependency for the four-bit MSE
inner-product path. The platform entry point selects ARM64 NEON only when the
compiler target provides it and otherwise executes the portable scalar C path.

```text
SwiftHNSW
    -> CTurboQuantKernels
        -> ARM64 NEON kernel (Native ARM64)
        -> portable C kernel (regular WASM, Embedded WASM)
```

## Target evidence

| Target | C compile | C target link | Swift selection |
| --- | --- | --- | --- |
| Native arm64 | Passed | Passed | Platform differential test passed; NEON specialization compiled |
| regular WASM | Passed | Passed | Portable C differential check and runtime benchmark passed |
| Embedded WASM | Passed | Passed | Portable C differential check and runtime benchmark passed |
| macOS x86_64 | C target only | Not a supported package build | Float16 package APIs are unavailable for this destination in the fixed toolchain |

The WASM rows verify the production kernel boundary, not a complete index
lifecycle. A minimal public `HNSWIndex<Float>.add` smoke run on the fixed regular
WASM snapshot traps in generic buffer metadata before the C kernel is reached.
Concrete construction experiments progressed through insertion but later
trapped while releasing the generic temporary HNSW builder. Retaining that
builder indefinitely would violate `TurboQuantIndex.finalize()`'s memory-release
contract, so no leak-based workaround is included and end-to-end WASM index
support is not claimed.

The portable path decodes the sixteen Float32 centroids once per call, then
uses four accumulators over packed bytes. It does not depend on Objective-C,
Foundation, libc-specific extensions, threads, or platform SIMD intrinsics.
The ARM64 path retains its existing NEON table-lookup implementation; its
scalar tail still uses the original byte-plane decode.

## Correctness and performance check

The platform four-bit test covers dimensions 17, 128, and 768. The explicit
portable C differential test covers dimensions 1 through the critical odd,
even, vector-width, and tail boundaries up to 769. A d=768 direct scoring
benchmark verifies every score before timing it. At 16 candidates, portable C
was 1.55x faster than pure Swift on the Native ARM64 host and regular WASM, and
1.67x on Embedded WASM. The advantage grows when the Swift path
constructs its packed lookup table at 32 or more candidates.

Exact commands, toolchain metadata, and raw samples are stored in
[`../2026-08-01-cturboquant-backend-selection`](../2026-08-01-cturboquant-backend-selection/README.md).
