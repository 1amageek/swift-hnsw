#ifndef SWIFT_HNSW_TURBO_QUANT_BENCHMARK_REFERENCE_H
#define SWIFT_HNSW_TURBO_QUANT_BENCHMARK_REFERENCE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes the scalar four-bit centroid dot product used only as a benchmark reference.
//
// The caller retains all initialized buffers for this synchronous call. `packed_codes`
// contains at least ceil(dimension / 2) bytes, `query` contains `dimension` Float32
// values, and `centroids` contains sixteen Float32 values. The buffers do not overlap,
// this function does not mutate or deallocate them, and no pointer escapes the call.
float swift_hnsw_benchmark_four_bit_inner_product(
    const uint8_t *packed_codes,
    const float *query,
    size_t dimension,
    const float *centroids
);

#ifdef __cplusplus
}
#endif

#endif
