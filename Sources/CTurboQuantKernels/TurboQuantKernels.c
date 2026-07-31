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

int swift_hnsw_turboquant_has_accelerated_four_bit_inner_product(void) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

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
#endif

    for (; coordinate < dimension; ++coordinate) {
        const uint8_t packed = packed_codes[coordinate / 2];
        const uint8_t index = (coordinate & 1) == 0
            ? packed >> 4
            : packed & 0x0f;
        total += query[coordinate]
            * load_centroid(centroid_byte_planes, index);
    }
    return total;
}
