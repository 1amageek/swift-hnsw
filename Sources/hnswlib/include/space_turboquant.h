#pragma once

#include "hnswlib.h"
#include <cstdint>

namespace hnswlib {

/// Parameters shared between space and distance function
struct TurboQuantParams {
    size_t dim;             // original vector dimension
    size_t padded_dim;      // next power of 2 (actual working dimension)
    int bits;               // total quantization bits (1, 2, 3, or 4)
    int num_centroids;      // 2^bits (mse) or 2^(bits-1) (prod)
    float codebook[16];     // scaled centroids
    int mode;               // 0 = float L2 (construction), 1 = ADC (search after finalize)
};

/// L2 distance space for TurboQuant quantized vectors.
///
/// Supports two modes:
/// - Symmetric (mode=0): both arguments are packed quantized bytes.
///   Used during index construction.
/// - Asymmetric/ADC (mode=1): first argument is a rotated float query (d floats),
///   second argument is packed quantized bytes. Used during search.
class TurboQuantL2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    TurboQuantParams params_;

public:
    TurboQuantL2Space(size_t dim, size_t padded_dim, int bits, const float* codebook,
                      int num_centroids);
    ~TurboQuantL2Space() = default;

    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void* get_dist_func_param() override { return &params_; }

    void setMode(int mode) { params_.mode = mode; }

    /// Update data_size after finalize/repack to match packed format.
    /// This ensures load_from_buffer reads the correct data_size.
    void setDataSize(size_t new_data_size) { data_size_ = new_data_size; }
};

} // namespace hnswlib
