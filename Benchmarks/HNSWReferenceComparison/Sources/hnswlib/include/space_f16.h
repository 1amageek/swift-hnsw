#pragma once

#include "hnswlib.h"

namespace hnswlib {

/// Float16 L2 (Euclidean) Distance Space
/// Uses NEON on ARM64, F16C+AVX on x86_64, scalar fallback elsewhere
class L2SpaceF16 : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    L2SpaceF16(size_t dim);
    ~L2SpaceF16() = default;

    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void *get_dist_func_param() override { return &dim_; }
};

/// Float16 Inner Product Distance Space
/// Uses NEON on ARM64, F16C+AVX on x86_64, scalar fallback elsewhere
class InnerProductSpaceF16 : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    InnerProductSpaceF16(size_t dim);
    ~InnerProductSpaceF16() = default;

    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void *get_dist_func_param() override { return &dim_; }
};

} // namespace hnswlib
