#ifndef SWIFT_HNSW_TURBO_QUANT_KERNELS_H
#define SWIFT_HNSW_TURBO_QUANT_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes a scalar four-bit centroid dot product without retaining or modifying input memory.
//
// The caller owns and synchronizes every buffer for the duration of this synchronous call.
// `packed_codes` contains at least ceil(dimension / 2) initialized bytes, `query` contains
// `dimension` initialized Float32 values, and `centroids` contains sixteen initialized
// Float32 values. The validated buffers do not overlap, this function does not bind, rebind,
// initialize, mutate, or deallocate their memory, and no pointer escapes the call.
float swift_hnsw_turboquant_four_bit_inner_product_scalar(
    const uint8_t *packed_codes,
    const float *query,
    size_t dimension,
    const float *centroids
);

#if defined(__aarch64__) && defined(__ARM_NEON)

// Computes a four-bit centroid dot product with ARM64 NEON without retaining or modifying input memory.
//
// The caller owns and synchronizes every buffer for the duration of this synchronous call
// and remains solely responsible for deallocation. `packed_codes` contains at least
// ceil(dimension / 2) initialized bytes, `query` contains `dimension` initialized,
// naturally aligned Float32 values, and `centroid_byte_planes` contains four initialized
// 16-byte little-endian planes. The caller validates those counts before pointer creation,
// so every offset and byte count is representable. The buffers do not overlap; this
// function does not bind, rebind, initialize, mutate, or deallocate their memory, and no
// pointer escapes the call.
float swift_hnsw_turboquant_four_bit_inner_product(
    const uint8_t *packed_codes,
    const float *query,
    size_t dimension,
    const uint8_t *centroid_byte_planes
);

#endif

#ifdef __cplusplus
}
#endif

#endif
