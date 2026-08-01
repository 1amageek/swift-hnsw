#include "TurboQuantKernels.h"

#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static inline float32x4_t decode_centroids(
    uint16x8_t lower_words,
    uint16x8_t upper_words,
    int upper_half
) {
    uint16x8x2_t words = vzipq_u16(lower_words, upper_words);
    return vreinterpretq_f32_u16(words.val[upper_half]);
}
#endif

int swift_hnsw_turboquant_has_four_bit_inner_product_kernel(void) {
    return 1;
}

#if !defined(__aarch64__) || !defined(__ARM_NEON)
static inline void decode_centroid_table(
    float *centroids,
    const uint8_t *centroid_byte_planes
) {
    for (uint8_t index = 0; index < 16; ++index) {
        uint32_t bits =
            ((uint32_t)centroid_byte_planes[index]) |
            ((uint32_t)centroid_byte_planes[16 + index] << 8) |
            ((uint32_t)centroid_byte_planes[32 + index] << 16) |
            ((uint32_t)centroid_byte_planes[48 + index] << 24);
        memcpy(&centroids[index], &bits, sizeof(bits));
    }
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline float load_centroid(
    const uint8_t *centroid_byte_planes,
    uint8_t index
) {
    uint32_t bits =
        ((uint32_t)centroid_byte_planes[index]) |
        ((uint32_t)centroid_byte_planes[16 + index] << 8) |
        ((uint32_t)centroid_byte_planes[32 + index] << 16) |
        ((uint32_t)centroid_byte_planes[48 + index] << 24);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
#endif

float swift_hnsw_turboquant_four_bit_inner_product(
    const uint8_t *restrict packed_codes,
    const float *restrict query,
    size_t dimension,
    const uint8_t *restrict centroid_byte_planes
) {
    size_t coordinate = 0;
    float total = 0.0f;

#if defined(__aarch64__) && defined(__ARM_NEON)
    const uint8x16_t centroid_byte0 =
        vld1q_u8(centroid_byte_planes);
    const uint8x16_t centroid_byte1 =
        vld1q_u8(centroid_byte_planes + 16);
    const uint8x16_t centroid_byte2 =
        vld1q_u8(centroid_byte_planes + 32);
    const uint8x16_t centroid_byte3 =
        vld1q_u8(centroid_byte_planes + 48);
    const uint8x8_t low_nibble_mask = vdup_n_u8(0x0f);
    float32x4_t accumulator0 = vdupq_n_f32(0.0f);
    float32x4_t accumulator1 = vdupq_n_f32(0.0f);
    float32x4_t accumulator2 = vdupq_n_f32(0.0f);
    float32x4_t accumulator3 = vdupq_n_f32(0.0f);

    for (; coordinate + 16 <= dimension; coordinate += 16) {
        const uint8x8_t packed = vld1_u8(
            packed_codes + coordinate / 2
        );
        const uint8x8_t high = vshr_n_u8(packed, 4);
        const uint8x8_t low = vand_u8(packed, low_nibble_mask);
        const uint8x8x2_t interleaved_indices = vzip_u8(high, low);
        const uint8x16_t indices = vcombine_u8(
            interleaved_indices.val[0],
            interleaved_indices.val[1]
        );

        const uint8x16_t byte0 = vqtbl1q_u8(
            centroid_byte0,
            indices
        );
        const uint8x16_t byte1 = vqtbl1q_u8(
            centroid_byte1,
            indices
        );
        const uint8x16_t byte2 = vqtbl1q_u8(
            centroid_byte2,
            indices
        );
        const uint8x16_t byte3 = vqtbl1q_u8(
            centroid_byte3,
            indices
        );
        const uint8x16x2_t lower_bytes = vzipq_u8(byte0, byte1);
        const uint8x16x2_t upper_bytes = vzipq_u8(byte2, byte3);

        const float32x4_t centroids0 = decode_centroids(
            vreinterpretq_u16_u8(lower_bytes.val[0]),
            vreinterpretq_u16_u8(upper_bytes.val[0]),
            0
        );
        const float32x4_t centroids1 = decode_centroids(
            vreinterpretq_u16_u8(lower_bytes.val[0]),
            vreinterpretq_u16_u8(upper_bytes.val[0]),
            1
        );
        const float32x4_t centroids2 = decode_centroids(
            vreinterpretq_u16_u8(lower_bytes.val[1]),
            vreinterpretq_u16_u8(upper_bytes.val[1]),
            0
        );
        const float32x4_t centroids3 = decode_centroids(
            vreinterpretq_u16_u8(lower_bytes.val[1]),
            vreinterpretq_u16_u8(upper_bytes.val[1]),
            1
        );

        accumulator0 = vfmaq_f32(
            accumulator0,
            vld1q_f32(query + coordinate),
            centroids0
        );
        accumulator1 = vfmaq_f32(
            accumulator1,
            vld1q_f32(query + coordinate + 4),
            centroids1
        );
        accumulator2 = vfmaq_f32(
            accumulator2,
            vld1q_f32(query + coordinate + 8),
            centroids2
        );
        accumulator3 = vfmaq_f32(
            accumulator3,
            vld1q_f32(query + coordinate + 12),
            centroids3
        );
    }
    total = vaddvq_f32(
        vaddq_f32(
            vaddq_f32(accumulator0, accumulator1),
            vaddq_f32(accumulator2, accumulator3)
        )
    );
#else
    float centroids[16];
    decode_centroid_table(centroids, centroid_byte_planes);
    const size_t pair_count = dimension / 2;
    size_t byte_index = 0;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    float accumulator2 = 0.0f;
    float accumulator3 = 0.0f;

    // The portable path is used by WASM, Embedded, and non-ARM targets. It
    // keeps the same bounded packed-byte contract while avoiding a centroid
    // bit reconstruction for every coordinate.
    for (; byte_index + 4 <= pair_count; byte_index += 4) {
        const size_t coordinate0 = byte_index * 2;
        const uint8_t packed0 = packed_codes[byte_index];
        const uint8_t packed1 = packed_codes[byte_index + 1];
        const uint8_t packed2 = packed_codes[byte_index + 2];
        const uint8_t packed3 = packed_codes[byte_index + 3];
        accumulator0 += query[coordinate0] * centroids[packed0 >> 4];
        accumulator0 += query[coordinate0 + 1] * centroids[packed0 & 0x0f];
        accumulator1 += query[coordinate0 + 2] * centroids[packed1 >> 4];
        accumulator1 += query[coordinate0 + 3] * centroids[packed1 & 0x0f];
        accumulator2 += query[coordinate0 + 4] * centroids[packed2 >> 4];
        accumulator2 += query[coordinate0 + 5] * centroids[packed2 & 0x0f];
        accumulator3 += query[coordinate0 + 6] * centroids[packed3 >> 4];
        accumulator3 += query[coordinate0 + 7] * centroids[packed3 & 0x0f];
    }
    total = accumulator0 + accumulator1 + accumulator2 + accumulator3;

    for (; byte_index < pair_count; ++byte_index) {
        const size_t coordinate0 = byte_index * 2;
        const uint8_t packed = packed_codes[byte_index];
        total += query[coordinate0] * centroids[packed >> 4];
        total += query[coordinate0 + 1] * centroids[packed & 0x0f];
    }
    coordinate = pair_count * 2;
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; coordinate < dimension; ++coordinate) {
        const uint8_t packed = packed_codes[coordinate / 2];
        const uint8_t index = (coordinate & 1) == 0
            ? packed >> 4
            : packed & 0x0f;
        total += query[coordinate] * load_centroid(
            centroid_byte_planes,
            index
        );
    }
#else
    for (; coordinate < dimension; ++coordinate) {
        const uint8_t packed = packed_codes[coordinate / 2];
        const uint8_t index = (coordinate & 1) == 0
            ? packed >> 4
            : packed & 0x0f;
        total += query[coordinate] * centroids[index];
    }
#endif
    return total;
}
