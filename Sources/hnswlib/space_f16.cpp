#include "include/space_f16.h"
#include <cstring>

namespace hnswlib {

// ============================================================
// ARM NEON Implementation (Apple Silicon, ARM64)
// ============================================================
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

// Optimized for dim % 16 == 0 (processes 16 elements per iteration with 2x unroll)
static float L2SqrF16_NEON16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float16_t *pVect1 = (const float16_t *)pVect1v;
    const float16_t *pVect2 = (const float16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    size_t qty16 = qty >> 4;
    const float16_t *pEnd = pVect1 + (qty16 << 4);

    while (pVect1 < pEnd) {
        // First 8 elements
        float16x8_t v1_0 = vld1q_f16(pVect1);
        float16x8_t v2_0 = vld1q_f16(pVect2);
        float16x8_t diff0 = vsubq_f16(v1_0, v2_0);

        float32x4_t diff0_lo = vcvt_f32_f16(vget_low_f16(diff0));
        float32x4_t diff0_hi = vcvt_f32_f16(vget_high_f16(diff0));
        sum0 = vfmaq_f32(sum0, diff0_lo, diff0_lo);
        sum1 = vfmaq_f32(sum1, diff0_hi, diff0_hi);

        // Second 8 elements (2x unroll)
        float16x8_t v1_1 = vld1q_f16(pVect1 + 8);
        float16x8_t v2_1 = vld1q_f16(pVect2 + 8);
        float16x8_t diff1 = vsubq_f16(v1_1, v2_1);

        float32x4_t diff1_lo = vcvt_f32_f16(vget_low_f16(diff1));
        float32x4_t diff1_hi = vcvt_f32_f16(vget_high_f16(diff1));
        sum0 = vfmaq_f32(sum0, diff1_lo, diff1_lo);
        sum1 = vfmaq_f32(sum1, diff1_hi, diff1_hi);

        pVect1 += 16;
        pVect2 += 16;
    }

    // Combine accumulators and horizontal sum
    sum0 = vaddq_f32(sum0, sum1);
    return vaddvq_f32(sum0);
}

// Optimized for dim % 8 == 0
static float L2SqrF16_NEON8(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float16_t *pVect1 = (const float16_t *)pVect1v;
    const float16_t *pVect2 = (const float16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t qty8 = qty >> 3;
    const float16_t *pEnd = pVect1 + (qty8 << 3);

    while (pVect1 < pEnd) {
        float16x8_t v1 = vld1q_f16(pVect1);
        float16x8_t v2 = vld1q_f16(pVect2);
        float16x8_t diff = vsubq_f16(v1, v2);

        float32x4_t diff_lo = vcvt_f32_f16(vget_low_f16(diff));
        float32x4_t diff_hi = vcvt_f32_f16(vget_high_f16(diff));

        sum = vfmaq_f32(sum, diff_lo, diff_lo);
        sum = vfmaq_f32(sum, diff_hi, diff_hi);

        pVect1 += 8;
        pVect2 += 8;
    }

    return vaddvq_f32(sum);
}

// General case with residual handling
static float L2SqrF16_NEON_Residual(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float16_t *pVect1 = (const float16_t *)pVect1v;
    const float16_t *pVect2 = (const float16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 8 <= qty; i += 8) {
        float16x8_t v1 = vld1q_f16(pVect1 + i);
        float16x8_t v2 = vld1q_f16(pVect2 + i);
        float16x8_t diff = vsubq_f16(v1, v2);

        float32x4_t diff_lo = vcvt_f32_f16(vget_low_f16(diff));
        float32x4_t diff_hi = vcvt_f32_f16(vget_high_f16(diff));

        sum = vfmaq_f32(sum, diff_lo, diff_lo);
        sum = vfmaq_f32(sum, diff_hi, diff_hi);
    }

    float result = vaddvq_f32(sum);

    // Process remaining elements
    for (; i < qty; i++) {
        float diff = (float)pVect1[i] - (float)pVect2[i];
        result += diff * diff;
    }

    return result;
}

// Inner Product - dim % 16 == 0
static float InnerProductF16_NEON16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float16_t *pVect1 = (const float16_t *)pVect1v;
    const float16_t *pVect2 = (const float16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    size_t qty16 = qty >> 4;
    const float16_t *pEnd = pVect1 + (qty16 << 4);

    while (pVect1 < pEnd) {
        // First 8 elements
        float16x8_t v1_0 = vld1q_f16(pVect1);
        float16x8_t v2_0 = vld1q_f16(pVect2);

        float32x4_t v1_0_lo = vcvt_f32_f16(vget_low_f16(v1_0));
        float32x4_t v1_0_hi = vcvt_f32_f16(vget_high_f16(v1_0));
        float32x4_t v2_0_lo = vcvt_f32_f16(vget_low_f16(v2_0));
        float32x4_t v2_0_hi = vcvt_f32_f16(vget_high_f16(v2_0));

        sum0 = vfmaq_f32(sum0, v1_0_lo, v2_0_lo);
        sum1 = vfmaq_f32(sum1, v1_0_hi, v2_0_hi);

        // Second 8 elements (2x unroll)
        float16x8_t v1_1 = vld1q_f16(pVect1 + 8);
        float16x8_t v2_1 = vld1q_f16(pVect2 + 8);

        float32x4_t v1_1_lo = vcvt_f32_f16(vget_low_f16(v1_1));
        float32x4_t v1_1_hi = vcvt_f32_f16(vget_high_f16(v1_1));
        float32x4_t v2_1_lo = vcvt_f32_f16(vget_low_f16(v2_1));
        float32x4_t v2_1_hi = vcvt_f32_f16(vget_high_f16(v2_1));

        sum0 = vfmaq_f32(sum0, v1_1_lo, v2_1_lo);
        sum1 = vfmaq_f32(sum1, v1_1_hi, v2_1_hi);

        pVect1 += 16;
        pVect2 += 16;
    }

    sum0 = vaddq_f32(sum0, sum1);
    return vaddvq_f32(sum0);
}

// Inner Product - dim % 8 == 0
static float InnerProductF16_NEON8(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float16_t *pVect1 = (const float16_t *)pVect1v;
    const float16_t *pVect2 = (const float16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t qty8 = qty >> 3;
    const float16_t *pEnd = pVect1 + (qty8 << 3);

    while (pVect1 < pEnd) {
        float16x8_t v1 = vld1q_f16(pVect1);
        float16x8_t v2 = vld1q_f16(pVect2);

        float32x4_t v1_lo = vcvt_f32_f16(vget_low_f16(v1));
        float32x4_t v1_hi = vcvt_f32_f16(vget_high_f16(v1));
        float32x4_t v2_lo = vcvt_f32_f16(vget_low_f16(v2));
        float32x4_t v2_hi = vcvt_f32_f16(vget_high_f16(v2));

        sum = vfmaq_f32(sum, v1_lo, v2_lo);
        sum = vfmaq_f32(sum, v1_hi, v2_hi);

        pVect1 += 8;
        pVect2 += 8;
    }

    return vaddvq_f32(sum);
}

// Inner Product - general case with residual
static float InnerProductF16_NEON_Residual(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float16_t *pVect1 = (const float16_t *)pVect1v;
    const float16_t *pVect2 = (const float16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 8 <= qty; i += 8) {
        float16x8_t v1 = vld1q_f16(pVect1 + i);
        float16x8_t v2 = vld1q_f16(pVect2 + i);

        float32x4_t v1_lo = vcvt_f32_f16(vget_low_f16(v1));
        float32x4_t v1_hi = vcvt_f32_f16(vget_high_f16(v1));
        float32x4_t v2_lo = vcvt_f32_f16(vget_low_f16(v2));
        float32x4_t v2_hi = vcvt_f32_f16(vget_high_f16(v2));

        sum = vfmaq_f32(sum, v1_lo, v2_lo);
        sum = vfmaq_f32(sum, v1_hi, v2_hi);
    }

    float result = vaddvq_f32(sum);

    for (; i < qty; i++) {
        result += (float)pVect1[i] * (float)pVect2[i];
    }

    return result;
}

// Function pointers for dynamic dispatch
static DISTFUNC<float> L2SqrF16_NEON = L2SqrF16_NEON_Residual;
static DISTFUNC<float> InnerProductF16_NEON = InnerProductF16_NEON_Residual;

#define L2SqrF16 L2SqrF16_NEON
#define InnerProductF16 InnerProductF16_NEON

// ============================================================
// x86_64 Implementation (F16C + AVX)
// ============================================================
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__F16C__)
#include <immintrin.h>

// Optimized for dim % 16 == 0 (2x unroll)
static float L2SqrF16_AVX16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    size_t qty16 = qty >> 4;
    const uint16_t *pEnd = pVect1 + (qty16 << 4);

    while (pVect1 < pEnd) {
        // First 8 elements
        __m128i v1_f16_0 = _mm_loadu_si128((const __m128i*)pVect1);
        __m128i v2_f16_0 = _mm_loadu_si128((const __m128i*)pVect2);
        __m256 v1_f32_0 = _mm256_cvtph_ps(v1_f16_0);
        __m256 v2_f32_0 = _mm256_cvtph_ps(v2_f16_0);
        __m256 diff0 = _mm256_sub_ps(v1_f32_0, v2_f32_0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        // Second 8 elements (2x unroll)
        __m128i v1_f16_1 = _mm_loadu_si128((const __m128i*)(pVect1 + 8));
        __m128i v2_f16_1 = _mm_loadu_si128((const __m128i*)(pVect2 + 8));
        __m256 v1_f32_1 = _mm256_cvtph_ps(v1_f16_1);
        __m256 v2_f32_1 = _mm256_cvtph_ps(v2_f16_1);
        __m256 diff1 = _mm256_sub_ps(v1_f32_1, v2_f32_1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        pVect1 += 16;
        pVect2 += 16;
    }

    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);

    // Horizontal sum
    __m128 sum_lo = _mm256_castps256_ps128(sum0);
    __m128 sum_hi = _mm256_extractf128_ps(sum0, 1);
    __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// Optimized for dim % 8 == 0
static float L2SqrF16_AVX8(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    __m256 sum = _mm256_setzero_ps();

    size_t qty8 = qty >> 3;
    const uint16_t *pEnd = pVect1 + (qty8 << 3);

    while (pVect1 < pEnd) {
        __m128i v1_f16 = _mm_loadu_si128((const __m128i*)pVect1);
        __m128i v2_f16 = _mm_loadu_si128((const __m128i*)pVect2);
        __m256 v1_f32 = _mm256_cvtph_ps(v1_f16);
        __m256 v2_f32 = _mm256_cvtph_ps(v2_f16);
        __m256 diff = _mm256_sub_ps(v1_f32, v2_f32);
        sum = _mm256_fmadd_ps(diff, diff, sum);

        pVect1 += 8;
        pVect2 += 8;
    }

    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// General case with residual
static float L2SqrF16_AVX_Residual(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= qty; i += 8) {
        __m128i v1_f16 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
        __m128i v2_f16 = _mm_loadu_si128((const __m128i*)(pVect2 + i));
        __m256 v1_f32 = _mm256_cvtph_ps(v1_f16);
        __m256 v2_f32 = _mm256_cvtph_ps(v2_f16);
        __m256 diff = _mm256_sub_ps(v1_f32, v2_f32);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Scalar remainder
    for (; i < qty; i++) {
        float v1 = _cvtsh_ss(pVect1[i]);
        float v2 = _cvtsh_ss(pVect2[i]);
        float diff = v1 - v2;
        result += diff * diff;
    }

    return result;
}

// Inner Product functions (similar optimization pattern)
static float InnerProductF16_AVX16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    size_t qty16 = qty >> 4;
    const uint16_t *pEnd = pVect1 + (qty16 << 4);

    while (pVect1 < pEnd) {
        __m128i v1_f16_0 = _mm_loadu_si128((const __m128i*)pVect1);
        __m128i v2_f16_0 = _mm_loadu_si128((const __m128i*)pVect2);
        __m256 v1_f32_0 = _mm256_cvtph_ps(v1_f16_0);
        __m256 v2_f32_0 = _mm256_cvtph_ps(v2_f16_0);
        sum0 = _mm256_fmadd_ps(v1_f32_0, v2_f32_0, sum0);

        __m128i v1_f16_1 = _mm_loadu_si128((const __m128i*)(pVect1 + 8));
        __m128i v2_f16_1 = _mm_loadu_si128((const __m128i*)(pVect2 + 8));
        __m256 v1_f32_1 = _mm256_cvtph_ps(v1_f16_1);
        __m256 v2_f32_1 = _mm256_cvtph_ps(v2_f16_1);
        sum1 = _mm256_fmadd_ps(v1_f32_1, v2_f32_1, sum1);

        pVect1 += 16;
        pVect2 += 16;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    __m128 sum_lo = _mm256_castps256_ps128(sum0);
    __m128 sum_hi = _mm256_extractf128_ps(sum0, 1);
    __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static float InnerProductF16_AVX8(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    __m256 sum = _mm256_setzero_ps();

    size_t qty8 = qty >> 3;
    const uint16_t *pEnd = pVect1 + (qty8 << 3);

    while (pVect1 < pEnd) {
        __m128i v1_f16 = _mm_loadu_si128((const __m128i*)pVect1);
        __m128i v2_f16 = _mm_loadu_si128((const __m128i*)pVect2);
        __m256 v1_f32 = _mm256_cvtph_ps(v1_f16);
        __m256 v2_f32 = _mm256_cvtph_ps(v2_f16);
        sum = _mm256_fmadd_ps(v1_f32, v2_f32, sum);

        pVect1 += 8;
        pVect2 += 8;
    }

    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static float InnerProductF16_AVX_Residual(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= qty; i += 8) {
        __m128i v1_f16 = _mm_loadu_si128((const __m128i*)(pVect1 + i));
        __m128i v2_f16 = _mm_loadu_si128((const __m128i*)(pVect2 + i));
        __m256 v1_f32 = _mm256_cvtph_ps(v1_f16);
        __m256 v2_f32 = _mm256_cvtph_ps(v2_f16);
        sum = _mm256_fmadd_ps(v1_f32, v2_f32, sum);
    }

    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    for (; i < qty; i++) {
        float v1 = _cvtsh_ss(pVect1[i]);
        float v2 = _cvtsh_ss(pVect2[i]);
        result += v1 * v2;
    }

    return result;
}

// Function pointers for dynamic dispatch
static DISTFUNC<float> L2SqrF16_AVX = L2SqrF16_AVX_Residual;
static DISTFUNC<float> InnerProductF16_AVX = InnerProductF16_AVX_Residual;

#define L2SqrF16 L2SqrF16_AVX
#define InnerProductF16 InnerProductF16_AVX

// ============================================================
// Fallback: Scalar Implementation
// ============================================================
#else

// Software Float16 to Float32 conversion (IEEE 754 half-precision)
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        // Denormalized number
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float L2SqrF16_Scalar(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float result = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        float diff = f16_to_f32(pVect1[i]) - f16_to_f32(pVect2[i]);
        result += diff * diff;
    }
    return result;
}

static float InnerProductF16_Scalar(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const uint16_t *pVect1 = (const uint16_t *)pVect1v;
    const uint16_t *pVect2 = (const uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float result = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        result += f16_to_f32(pVect1[i]) * f16_to_f32(pVect2[i]);
    }
    return result;
}

#define L2SqrF16 L2SqrF16_Scalar
#define InnerProductF16 InnerProductF16_Scalar

#endif

// ============================================================
// Common: Distance wrapper and constructors
// ============================================================

static float InnerProductDistanceF16(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProductF16(pVect1, pVect2, qty_ptr);
}

L2SpaceF16::L2SpaceF16(size_t dim) : dim_(dim) {
    data_size_ = dim * sizeof(uint16_t);  // 2 bytes per Float16 element

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    // Select optimal function based on dimension
    if (dim % 16 == 0) {
        L2SqrF16_NEON = L2SqrF16_NEON16;
    } else if (dim % 8 == 0) {
        L2SqrF16_NEON = L2SqrF16_NEON8;
    } else {
        L2SqrF16_NEON = L2SqrF16_NEON_Residual;
    }
    fstdistfunc_ = L2SqrF16_NEON;
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__F16C__)
    if (dim % 16 == 0) {
        L2SqrF16_AVX = L2SqrF16_AVX16;
    } else if (dim % 8 == 0) {
        L2SqrF16_AVX = L2SqrF16_AVX8;
    } else {
        L2SqrF16_AVX = L2SqrF16_AVX_Residual;
    }
    fstdistfunc_ = L2SqrF16_AVX;
#else
    fstdistfunc_ = L2SqrF16;
#endif
}

InnerProductSpaceF16::InnerProductSpaceF16(size_t dim) : dim_(dim) {
    data_size_ = dim * sizeof(uint16_t);  // 2 bytes per Float16 element

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    if (dim % 16 == 0) {
        InnerProductF16_NEON = InnerProductF16_NEON16;
    } else if (dim % 8 == 0) {
        InnerProductF16_NEON = InnerProductF16_NEON8;
    } else {
        InnerProductF16_NEON = InnerProductF16_NEON_Residual;
    }
    fstdistfunc_ = InnerProductDistanceF16;
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__F16C__)
    if (dim % 16 == 0) {
        InnerProductF16_AVX = InnerProductF16_AVX16;
    } else if (dim % 8 == 0) {
        InnerProductF16_AVX = InnerProductF16_AVX8;
    } else {
        InnerProductF16_AVX = InnerProductF16_AVX_Residual;
    }
    fstdistfunc_ = InnerProductDistanceF16;
#else
    fstdistfunc_ = InnerProductDistanceF16;
#endif
}

} // namespace hnswlib
