#include "include/turboquant_encoder.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

// ============================================================
// SplitMix64 PRNG (matches Swift implementation)
// ============================================================

struct SplitMix64 {
    uint64_t state;

    SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        return z;
    }
};

// ============================================================
// TurboQuantEncoder
// ============================================================

struct TurboQuantEncoder {
    size_t dimension;
    size_t padded_dim;
    int bits;
    int num_centroids;
    size_t packed_size;
    float scale;

    std::vector<float> signs1, signs2, signs3;
    std::vector<float> codebook;
    std::vector<float> boundaries;
};

static size_t next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p *= 2;
    return p;
}

// ============================================================
// FWHT with NEON acceleration
// ============================================================

static void fwht_inplace(float* data, size_t n) {
    size_t h = 1;
    while (h < n) {
#if HAS_NEON
        // NEON-accelerated butterfly for blocks >= 4
        if (h >= 4) {
            for (size_t i = 0; i < n; i += h * 2) {
                float* top = data + i;
                float* bot = data + i + h;
                size_t j = 0;
                for (; j + 4 <= h; j += 4) {
                    float32x4_t a = vld1q_f32(top + j);
                    float32x4_t b = vld1q_f32(bot + j);
                    vst1q_f32(top + j, vaddq_f32(a, b));
                    vst1q_f32(bot + j, vsubq_f32(a, b));
                }
                for (; j < h; j++) {
                    float a = top[j], b = bot[j];
                    top[j] = a + b;
                    bot[j] = a - b;
                }
            }
        } else
#endif
        {
            // Scalar fallback
            for (size_t i = 0; i < n; i += h * 2) {
                float* top = data + i;
                float* bot = data + i + h;
                for (size_t j = 0; j < h; j++) {
                    float a = top[j], b = bot[j];
                    top[j] = a + b;
                    bot[j] = a - b;
                }
            }
        }
        h *= 2;
    }
}

// ============================================================
// Apply random signs (element-wise multiply by ±1)
// ============================================================

static void apply_signs(float* data, const float* signs, size_t n) {
#if HAS_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(data + i);
        float32x4_t s = vld1q_f32(signs + i);
        vst1q_f32(data + i, vmulq_f32(d, s));
    }
    for (; i < n; i++) {
        data[i] *= signs[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        data[i] *= signs[i];
    }
#endif
}

// ============================================================
// Normalize vector to unit length
// ============================================================

static float normalize_inplace(float* data, size_t n) {
    float norm_sq = 0;
#if HAS_NEON
    float32x4_t sum4 = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        sum4 = vfmaq_f32(sum4, v, v);
    }
    norm_sq = vaddvq_f32(sum4);
    for (; i < n; i++) norm_sq += data[i] * data[i];
#else
    for (size_t i = 0; i < n; i++) norm_sq += data[i] * data[i];
#endif

    if (norm_sq <= 0) return 0;
    float norm = sqrtf(norm_sq);
    float inv_norm = 1.0f / norm;

#if HAS_NEON
    float32x4_t scale4 = vdupq_n_f32(inv_norm);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmulq_f32(v, scale4));
    }
    for (; i < n; i++) data[i] *= inv_norm;
#else
    for (size_t i = 0; i < n; i++) data[i] *= inv_norm;
#endif

    return norm;
}

// ============================================================
// HD³ rotation: scale · H · D₃ · H · D₂ · H · D₁ · [x; 0]
// ============================================================

static void hd3_rotate(const TurboQuantEncoder* enc, const float* input, float* output,
                        float* work_buf) {
    size_t d = enc->dimension;
    size_t p = enc->padded_dim;

    // Zero-pad and apply D₁
    memset(work_buf, 0, p * sizeof(float));
    memcpy(work_buf, input, d * sizeof(float));
    apply_signs(work_buf, enc->signs1.data(), p);

    // Round 1: H
    fwht_inplace(work_buf, p);

    // Round 2: D₂ then H
    apply_signs(work_buf, enc->signs2.data(), p);
    fwht_inplace(work_buf, p);

    // Round 3: D₃ then H
    apply_signs(work_buf, enc->signs3.data(), p);
    fwht_inplace(work_buf, p);

    // Scale and output ALL p coordinates (not truncated to d).
    // Truncation to d breaks distance preservation for non-power-of-2 dims.
    float s = enc->scale;
#if HAS_NEON
    float32x4_t sv = vdupq_n_f32(s);
    size_t i = 0;
    for (; i + 4 <= p; i += 4) {
        float32x4_t v = vld1q_f32(work_buf + i);
        vst1q_f32(output + i, vmulq_f32(v, sv));
    }
    for (; i < p; i++) output[i] = work_buf[i] * s;
#else
    for (size_t i = 0; i < p; i++) output[i] = work_buf[i] * s;
#endif
}

// ============================================================
// Scalar quantization: find nearest centroid index
// ============================================================

static inline uint8_t quantize_scalar(float value, const float* boundaries, int num_boundaries) {
    // Binary search through boundaries
    int lo = 0, hi = num_boundaries;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (value <= boundaries[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return (uint8_t)lo;
}

// ============================================================
// Bit packing
// ============================================================

static void pack_4bit(const uint8_t* indices, uint8_t* output, size_t count) {
    size_t pairs = count / 2;
    for (size_t i = 0; i < pairs; i++) {
        output[i] = (indices[2 * i] << 4) | (indices[2 * i + 1] & 0x0F);
    }
    if (count % 2) {
        output[pairs] = indices[count - 1] << 4;
    }
}

static void pack_2bit(const uint8_t* indices, uint8_t* output, size_t count) {
    size_t quads = count / 4;
    for (size_t i = 0; i < quads; i++) {
        size_t b = 4 * i;
        output[i] = (indices[b] << 6) | ((indices[b+1] & 3) << 4)
                   | ((indices[b+2] & 3) << 2) | (indices[b+3] & 3);
    }
    size_t rem = count % 4;
    if (rem > 0) {
        uint8_t byte = 0;
        for (size_t j = 0; j < rem; j++) {
            byte |= (indices[quads * 4 + j] & 3) << (6 - j * 2);
        }
        output[quads] = byte;
    }
}

static void pack_1bit(const uint8_t* indices, uint8_t* output, size_t count) {
    size_t octets = count / 8;
    for (size_t i = 0; i < octets; i++) {
        uint8_t byte = 0;
        for (int j = 0; j < 8; j++) {
            byte |= (indices[8 * i + j] & 1) << (7 - j);
        }
        output[i] = byte;
    }
    size_t rem = count % 8;
    if (rem > 0) {
        uint8_t byte = 0;
        for (size_t j = 0; j < rem; j++) {
            byte |= (indices[octets * 8 + j] & 1) << (7 - (int)j);
        }
        output[octets] = byte;
    }
}

static void pack_3bit(const uint8_t* indices, uint8_t* output, size_t count, size_t out_size) {
    memset(output, 0, out_size);
    int bit_offset = 0;
    for (size_t i = 0; i < count; i++) {
        int byte_idx = bit_offset / 8;
        int bit_in = bit_offset % 8;
        uint8_t val = indices[i] & 0x07;
        output[byte_idx] |= val << bit_in;
        if (bit_in > 5) {
            output[byte_idx + 1] |= val >> (8 - bit_in);
        }
        bit_offset += 3;
    }
}

static void pack_indices(const uint8_t* indices, uint8_t* output, size_t count,
                          int bits, size_t packed_size) {
    switch (bits) {
        case 4: pack_4bit(indices, output, count); break;
        case 2: pack_2bit(indices, output, count); break;
        case 1: pack_1bit(indices, output, count); break;
        case 3: pack_3bit(indices, output, count, packed_size); break;
    }
}

// ============================================================
// Full encode pipeline
// ============================================================

// Max stack allocation threshold: 16KB. Above this, use heap.
static const size_t ALLOCA_THRESHOLD = 16 * 1024;

static void encode_single(const TurboQuantEncoder* enc, const float* input, uint8_t* output) {
    size_t d = enc->dimension;
    size_t p = enc->padded_dim;
    size_t total_bytes = d * sizeof(float) + p * sizeof(float) * 2 + p;
    bool use_heap = total_bytes > ALLOCA_THRESHOLD;

    std::vector<uint8_t> heap_buf;
    uint8_t* buf;
    if (use_heap) {
        heap_buf.resize(total_bytes);
        buf = heap_buf.data();
    } else {
        buf = (uint8_t*)alloca(total_bytes);
    }

    float* normalized = (float*)buf;
    float* work = (float*)(buf + d * sizeof(float));
    float* rotated = (float*)(buf + d * sizeof(float) + p * sizeof(float));
    uint8_t* indices = buf + d * sizeof(float) + p * sizeof(float) * 2;

    memcpy(normalized, input, d * sizeof(float));
    normalize_inplace(normalized, d);
    hd3_rotate(enc, normalized, rotated, work);

    const float* bounds = enc->boundaries.data();
    int n_bounds = (int)enc->boundaries.size();
    for (size_t j = 0; j < p; j++) {
        indices[j] = quantize_scalar(rotated[j], bounds, n_bounds);
    }
    pack_indices(indices, output, p, enc->bits, enc->packed_size);
}

static void rotate_query_single(const TurboQuantEncoder* enc, const float* input, float* output) {
    size_t d = enc->dimension;
    size_t p = enc->padded_dim;
    size_t total_bytes = d * sizeof(float) + p * sizeof(float);
    bool use_heap = total_bytes > ALLOCA_THRESHOLD;

    std::vector<uint8_t> heap_buf;
    uint8_t* buf;
    if (use_heap) {
        heap_buf.resize(total_bytes);
        buf = heap_buf.data();
    } else {
        buf = (uint8_t*)alloca(total_bytes);
    }

    float* normalized = (float*)buf;
    float* work = (float*)(buf + d * sizeof(float));

    memcpy(normalized, input, d * sizeof(float));
    normalize_inplace(normalized, d);
    hd3_rotate(enc, normalized, output, work);
}

// ============================================================
// C API Implementation
// ============================================================

extern "C" {

TurboQuantEncoderHandle hnsw_tq_encoder_create(
    size_t dimension, int bits,
    const float* codebook, int num_centroids,
    const float* boundaries, int num_boundaries,
    uint64_t seed
) {
    try {
        auto* enc = new TurboQuantEncoder();
        enc->dimension = dimension;
        enc->padded_dim = next_power_of_2(dimension);
        enc->bits = bits;
        enc->num_centroids = num_centroids;
        // Use ALL p coordinates (not truncated to d) to preserve distances exactly.
        enc->packed_size = (enc->padded_dim * bits + 7) / 8;
        // Scale = p^(-3/2): preserves norms over all p coordinates.
        float pf = (float)enc->padded_dim;
        enc->scale = 1.0f / (pf * sqrtf(pf));

        enc->codebook.assign(codebook, codebook + num_centroids);
        enc->boundaries.assign(boundaries, boundaries + num_boundaries);

        // Generate random signs from seed (must match Swift SplitMix64)
        SplitMix64 rng(seed);
        size_t p = enc->padded_dim;
        enc->signs1.resize(p);
        enc->signs2.resize(p);
        enc->signs3.resize(p);
        for (size_t i = 0; i < p; i++) enc->signs1[i] = (rng.next() & 1) ? -1.0f : 1.0f;
        for (size_t i = 0; i < p; i++) enc->signs2[i] = (rng.next() & 1) ? -1.0f : 1.0f;
        for (size_t i = 0; i < p; i++) enc->signs3[i] = (rng.next() & 1) ? -1.0f : 1.0f;

        return enc;
    } catch (...) {
        return nullptr;
    }
}

void hnsw_tq_encoder_destroy(TurboQuantEncoderHandle encoder) {
    delete static_cast<TurboQuantEncoder*>(encoder);
}

void hnsw_tq_encoder_encode(TurboQuantEncoderHandle encoder, const float* input, uint8_t* output) {
    auto* enc = static_cast<TurboQuantEncoder*>(encoder);
    encode_single(enc, input, output);
}

void hnsw_tq_encoder_encode_batch(
    TurboQuantEncoderHandle encoder, const float* inputs, uint8_t* outputs, size_t count
) {
    auto* enc = static_cast<TurboQuantEncoder*>(encoder);
    size_t d = enc->dimension;
    size_t ps = enc->packed_size;

    // Process sequentially (each encode uses alloca so can't easily parallelize here)
    for (size_t i = 0; i < count; i++) {
        encode_single(enc, inputs + i * d, outputs + i * ps);
    }
}

void hnsw_tq_encoder_rotate_query(TurboQuantEncoderHandle encoder, const float* input, float* output) {
    auto* enc = static_cast<TurboQuantEncoder*>(encoder);
    rotate_query_single(enc, input, output);
}

void hnsw_tq_encoder_rotate_query_batch(
    TurboQuantEncoderHandle encoder, const float* inputs, float* outputs, size_t count
) {
    auto* enc = static_cast<TurboQuantEncoder*>(encoder);
    size_t d = enc->dimension;
    for (size_t i = 0; i < count; i++) {
        rotate_query_single(enc, inputs + i * d, outputs + i * d);
    }
}

size_t hnsw_tq_encoder_packed_size(TurboQuantEncoderHandle encoder) {
    return static_cast<TurboQuantEncoder*>(encoder)->packed_size;
}

size_t hnsw_tq_encoder_padded_dim(TurboQuantEncoderHandle encoder) {
    return static_cast<TurboQuantEncoder*>(encoder)->padded_dim;
}

void hnsw_tq_encoder_quantize_rotated(TurboQuantEncoderHandle encoder,
                                       const float* rotated_input, uint8_t* output) {
    auto* enc = static_cast<TurboQuantEncoder*>(encoder);
    size_t p = enc->padded_dim;

    std::vector<uint8_t> heap_buf;
    uint8_t* indices;
    if (p > ALLOCA_THRESHOLD) {
        heap_buf.resize(p);
        indices = heap_buf.data();
    } else {
        indices = (uint8_t*)alloca(p);
    }

    const float* bounds = enc->boundaries.data();
    int n_bounds = (int)enc->boundaries.size();
    for (size_t j = 0; j < p; j++) {
        indices[j] = quantize_scalar(rotated_input[j], bounds, n_bounds);
    }
    pack_indices(indices, output, p, enc->bits, enc->packed_size);
}

} // extern "C"
