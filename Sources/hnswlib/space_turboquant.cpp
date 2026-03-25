#include "include/space_turboquant.h"
#include <cstring>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hnswlib {

// ============================================================
// Index extraction helpers
// ============================================================

static inline uint8_t extract_index_4bit(const uint8_t* packed, size_t idx) {
    uint8_t byte = packed[idx / 2];
    return (idx % 2 == 0) ? (byte >> 4) : (byte & 0x0F);
}

static inline uint8_t extract_index_2bit(const uint8_t* packed, size_t idx) {
    uint8_t byte = packed[idx / 4];
    int shift = 6 - (int)(idx % 4) * 2;
    return (byte >> shift) & 0x03;
}

static inline uint8_t extract_index_1bit(const uint8_t* packed, size_t idx) {
    uint8_t byte = packed[idx / 8];
    int shift = 7 - (int)(idx % 8);
    return (byte >> shift) & 0x01;
}

static inline uint8_t extract_index_3bit(const uint8_t* packed, size_t idx) {
    size_t bit_offset = idx * 3;
    size_t byte_idx = bit_offset / 8;
    int bit_in_byte = (int)(bit_offset % 8);
    uint8_t val = (packed[byte_idx] >> bit_in_byte);
    if (bit_in_byte > 5) {
        val |= (packed[byte_idx + 1] << (8 - bit_in_byte));
    }
    return val & 0x07;
}

// ============================================================
// Asymmetric Distance Computation (ADC)
// ============================================================

// ADC for 4-bit quantization (most common path)
static float TurboQuantL2_ADC_4bit(const float* query, const uint8_t* packed,
                                    const float* codebook, size_t dim) {
    float sum = 0.0f;
    size_t pairs = dim / 2;

    for (size_t i = 0; i < pairs; i++) {
        uint8_t byte = packed[i];
        float c0 = codebook[byte >> 4];
        float c1 = codebook[byte & 0x0F];
        float d0 = query[2 * i] - c0;
        float d1 = query[2 * i + 1] - c1;
        sum += d0 * d0 + d1 * d1;
    }
    // Handle odd dimension
    if (dim % 2 != 0) {
        float c = codebook[packed[pairs] >> 4];
        float d = query[dim - 1] - c;
        sum += d * d;
    }
    return sum;
}

// ADC for 2-bit quantization
static float TurboQuantL2_ADC_2bit(const float* query, const uint8_t* packed,
                                    const float* codebook, size_t dim) {
    float sum = 0.0f;
    size_t quads = dim / 4;

    for (size_t i = 0; i < quads; i++) {
        uint8_t byte = packed[i];
        float c0 = codebook[(byte >> 6) & 0x03];
        float c1 = codebook[(byte >> 4) & 0x03];
        float c2 = codebook[(byte >> 2) & 0x03];
        float c3 = codebook[byte & 0x03];
        float d0 = query[4 * i] - c0;
        float d1 = query[4 * i + 1] - c1;
        float d2 = query[4 * i + 2] - c2;
        float d3 = query[4 * i + 3] - c3;
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    size_t remaining = dim % 4;
    for (size_t j = 0; j < remaining; j++) {
        size_t idx = quads * 4 + j;
        float c = codebook[extract_index_2bit(packed, idx)];
        float d = query[idx] - c;
        sum += d * d;
    }
    return sum;
}

// ADC for 1-bit quantization
static float TurboQuantL2_ADC_1bit(const float* query, const uint8_t* packed,
                                    const float* codebook, size_t dim) {
    float sum = 0.0f;
    for (size_t j = 0; j < dim; j++) {
        float c = codebook[extract_index_1bit(packed, j)];
        float d = query[j] - c;
        sum += d * d;
    }
    return sum;
}

// ADC for 3-bit quantization
static float TurboQuantL2_ADC_3bit(const float* query, const uint8_t* packed,
                                    const float* codebook, size_t dim) {
    float sum = 0.0f;
    for (size_t j = 0; j < dim; j++) {
        float c = codebook[extract_index_3bit(packed, j)];
        float d = query[j] - c;
        sum += d * d;
    }
    return sum;
}

// ============================================================
// Symmetric Distance Computation (construction)
// ============================================================

static float TurboQuantL2_Symmetric(const uint8_t* packed1, const uint8_t* packed2,
                                     const float* codebook, size_t dim, int bits) {
    float sum = 0.0f;

    if (bits == 4) {
        size_t pairs = dim / 2;
        for (size_t i = 0; i < pairs; i++) {
            uint8_t b1 = packed1[i], b2 = packed2[i];
            float c1_0 = codebook[b1 >> 4], c2_0 = codebook[b2 >> 4];
            float c1_1 = codebook[b1 & 0x0F], c2_1 = codebook[b2 & 0x0F];
            float d0 = c1_0 - c2_0, d1 = c1_1 - c2_1;
            sum += d0 * d0 + d1 * d1;
        }
        if (dim % 2 != 0) {
            float c1 = codebook[packed1[pairs] >> 4];
            float c2 = codebook[packed2[pairs] >> 4];
            float d = c1 - c2;
            sum += d * d;
        }
    } else if (bits == 2) {
        size_t quads = dim / 4;
        for (size_t i = 0; i < quads; i++) {
            uint8_t b1 = packed1[i], b2 = packed2[i];
            for (int s = 6; s >= 0; s -= 2) {
                float c1 = codebook[(b1 >> s) & 0x03];
                float c2 = codebook[(b2 >> s) & 0x03];
                float d = c1 - c2;
                sum += d * d;
            }
        }
    } else {
        // Generic fallback for 1-bit and 3-bit
        for (size_t j = 0; j < dim; j++) {
            uint8_t idx1, idx2;
            if (bits == 1) {
                idx1 = extract_index_1bit(packed1, j);
                idx2 = extract_index_1bit(packed2, j);
            } else {
                idx1 = extract_index_3bit(packed1, j);
                idx2 = extract_index_3bit(packed2, j);
            }
            float d = codebook[idx1] - codebook[idx2];
            sum += d * d;
        }
    }

    return sum;
}

// ============================================================
// Unified Distance Function
// ============================================================

static float TurboQuantL2_ADC_dispatch(const float* query, const uint8_t* packed,
                                        const TurboQuantParams* params) {
    switch (params->bits) {
    case 4: return TurboQuantL2_ADC_4bit(query, packed, params->codebook, params->padded_dim);
    case 2: return TurboQuantL2_ADC_2bit(query, packed, params->codebook, params->padded_dim);
    case 1: return TurboQuantL2_ADC_1bit(query, packed, params->codebook, params->padded_dim);
    case 3: return TurboQuantL2_ADC_3bit(query, packed, params->codebook, params->padded_dim);
    default: return 0.0f;
    }
}

static float TurboQuantL2_Distance(const void* pVect1v, const void* pVect2v, const void* paramv) {
    const TurboQuantParams* params = static_cast<const TurboQuantParams*>(paramv);

    if (params->mode == 1) {
            // Search ADC: pVect1 = rotated float query, pVect2 = packed quantized (after finalize)
            return TurboQuantL2_ADC_dispatch(
                static_cast<const float*>(pVect1v),
                static_cast<const uint8_t*>(pVect2v), params);
        } else {
            // Mode 0: Construction — both args are Float32 rotated vectors → exact L2
            const float* a = static_cast<const float*>(pVect1v);
            const float* b = static_cast<const float*>(pVect2v);
            size_t dim = params->padded_dim;
            float sum = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
            float32x4_t sum4 = vdupq_n_f32(0);
            size_t i = 0;
            for (; i + 4 <= dim; i += 4) {
                float32x4_t va = vld1q_f32(a + i);
                float32x4_t vb = vld1q_f32(b + i);
                float32x4_t d = vsubq_f32(va, vb);
                sum4 = vfmaq_f32(sum4, d, d);
            }
            sum = vaddvq_f32(sum4);
            for (; i < dim; i++) { float d = a[i] - b[i]; sum += d * d; }
#else
            for (size_t i = 0; i < dim; i++) { float d = a[i] - b[i]; sum += d * d; }
#endif
            return sum;
        }
}

// ============================================================
// NEON-optimized ADC for 4-bit (ARM64)
// ============================================================

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

static float TurboQuantL2_ADC_4bit_NEON(const float* query, const uint8_t* packed,
                                         const float* codebook, size_t dim) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    // Process 8 dimensions at a time (4 packed bytes = 8 nibbles)
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        // Load 4 packed bytes containing 8 nibbles
        uint8_t b0 = packed[i / 2];
        uint8_t b1 = packed[i / 2 + 1];
        uint8_t b2 = packed[i / 2 + 2];
        uint8_t b3 = packed[i / 2 + 3];

        // Extract indices and lookup centroids
        float32x4_t c_lo = {codebook[b0 >> 4], codebook[b0 & 0x0F],
                            codebook[b1 >> 4], codebook[b1 & 0x0F]};
        float32x4_t c_hi = {codebook[b2 >> 4], codebook[b2 & 0x0F],
                            codebook[b3 >> 4], codebook[b3 & 0x0F]};

        float32x4_t q_lo = vld1q_f32(query + i);
        float32x4_t q_hi = vld1q_f32(query + i + 4);

        float32x4_t d_lo = vsubq_f32(q_lo, c_lo);
        float32x4_t d_hi = vsubq_f32(q_hi, c_hi);

        sum0 = vfmaq_f32(sum0, d_lo, d_lo);
        sum1 = vfmaq_f32(sum1, d_hi, d_hi);
    }

    sum0 = vaddq_f32(sum0, sum1);
    float result = vaddvq_f32(sum0);

    // Handle remaining dimensions
    for (; i < dim; i++) {
        float c = codebook[extract_index_4bit(packed, i)];
        float d = query[i] - c;
        result += d * d;
    }

    return result;
}

#endif // __ARM_NEON

// ============================================================
// Constructor
// ============================================================

TurboQuantL2Space::TurboQuantL2Space(size_t dim, size_t padded_dim, int bits,
                                     const float* codebook, int num_centroids) {
    params_.dim = dim;
    params_.padded_dim = padded_dim;
    params_.bits = bits;
    params_.num_centroids = num_centroids;
    params_.mode = 0;

    for (int i = 0; i < num_centroids && i < 16; i++) {
        params_.codebook[i] = codebook[i];
    }

    // data_size = p * sizeof(float): store full float vectors (p dimensions) during construction.
    // After finalize(), the packed quantized data overwrites the float data in-place.
    data_size_ = padded_dim * sizeof(float);

    fstdistfunc_ = TurboQuantL2_Distance;
}

} // namespace hnswlib
