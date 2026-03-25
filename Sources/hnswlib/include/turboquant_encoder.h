#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle for TurboQuant encoder (holds rotation signs, codebook, etc.)
typedef void* TurboQuantEncoderHandle;

/// Create a TurboQuant encoder.
/// Internally generates HD³ random signs from seed, stores codebook/boundaries.
TurboQuantEncoderHandle hnsw_tq_encoder_create(
    size_t dimension,
    int bits,
    const float* codebook,
    int num_centroids,
    const float* boundaries,
    int num_boundaries,
    uint64_t seed
);

void hnsw_tq_encoder_destroy(TurboQuantEncoderHandle encoder);

/// Encode a single vector: normalize → HD³ rotate → quantize → pack
/// output must have at least hnsw_tq_encoder_packed_size() bytes.
void hnsw_tq_encoder_encode(
    TurboQuantEncoderHandle encoder,
    const float* input,
    uint8_t* output
);

/// Encode a batch of vectors.
void hnsw_tq_encoder_encode_batch(
    TurboQuantEncoderHandle encoder,
    const float* inputs,
    uint8_t* outputs,
    size_t count
);

/// Rotate a query vector: normalize → HD³ rotate (keep full precision for ADC).
/// output must have at least `dimension` floats.
void hnsw_tq_encoder_rotate_query(
    TurboQuantEncoderHandle encoder,
    const float* input,
    float* output
);

/// Rotate a batch of query vectors.
void hnsw_tq_encoder_rotate_query_batch(
    TurboQuantEncoderHandle encoder,
    const float* inputs,
    float* outputs,
    size_t count
);

/// Get the packed size in bytes per vector.
size_t hnsw_tq_encoder_packed_size(TurboQuantEncoderHandle encoder);

/// Get the padded dimension (next power of 2 of original dimension).
size_t hnsw_tq_encoder_padded_dim(TurboQuantEncoderHandle encoder);

/// Quantize an already-rotated vector (skip normalization and rotation).
/// Used during finalize to convert stored float vectors to packed format.
void hnsw_tq_encoder_quantize_rotated(
    TurboQuantEncoderHandle encoder,
    const float* rotated_input,
    uint8_t* output
);

#ifdef __cplusplus
}
#endif
