# CTurboQuantKernels Portability Verification

`CTurboQuantKernels` is now an active production dependency for the four-bit
MSE inner-product path. The target exposes a portable C kernel on every
supported build and reports ARM64 NEON as an additional specialization only
when the compiler target provides it.

```text
SwiftHNSW
    -> CTurboQuantKernels
        -> ARM64 NEON kernel (Native ARM64)
        -> portable C kernel (Native non-ARM, regular WASM, Embedded WASM)
```

## Target evidence

| Target | C compile | C target link | Swift selection |
| --- | --- | --- | --- |
| Native arm64 | Passed | Passed | `has_four_bit_inner_product_kernel == 1`, NEON specialization enabled |
| regular WASM | `clang -target wasm32-unknown-wasip1` passed | `CTurboQuantKernels.o` linked into `SwiftHNSW.o` | portable C kernel selected |
| Embedded WASM | `clang -target wasm32-unknown-wasip1 -D__EMBEDDED_SWIFT__` passed | `CTurboQuantKernels.o` linked into `SwiftHNSW.o` | portable C kernel selected |

The portable path decodes the sixteen Float32 centroids once per call, then
uses four accumulators over packed bytes. It does not depend on Objective-C,
Foundation, libc-specific extensions, threads, or platform SIMD intrinsics.
The ARM64 path retains its existing NEON table-lookup implementation; its
scalar tail still uses the original byte-plane decode.

## Native correctness and performance check

The direct four-bit kernel test passed for dimensions 17, 128, and 768. The
full Native suite passed with 116 tests in 13 suites; Address Sanitizer passed
11 ScalarQuantizer tests and Thread Sanitizer passed the concurrent TurboQuant
search test. A paired d=768 MSE 4-bit run preserved Recall@10 and checksums at
every `efSearch`; CPU-QPS differences were within host noise, so no standalone
MSE speedup claim is made for the unchanged NEON path.
