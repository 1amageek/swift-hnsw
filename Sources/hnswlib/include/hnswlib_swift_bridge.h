#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types
typedef void* HNSWIndexHandle;
typedef void* HNSWSpaceHandle;

// Space functions
HNSWSpaceHandle hnsw_create_l2_space(size_t dim);
HNSWSpaceHandle hnsw_create_ip_space(size_t dim);
void hnsw_destroy_space(HNSWSpaceHandle space);

// Index functions
HNSWIndexHandle hnsw_create_index(
    HNSWSpaceHandle space,
    size_t max_elements,
    size_t M,
    size_t ef_construction,
    size_t random_seed,
    bool allow_replace_deleted
);

void hnsw_destroy_index(HNSWIndexHandle index);

// Index operations
bool hnsw_add_point(HNSWIndexHandle index, const float* data, uint64_t label);
int32_t hnsw_search_knn(
    HNSWIndexHandle index,
    const float* query,
    int32_t k,
    uint64_t* labels,
    float* distances
);

// Batch operations for high performance
int32_t hnsw_add_points_batch(
    HNSWIndexHandle index,
    const float* data,
    const uint64_t* labels,
    size_t num_points,
    size_t dimension
);

int32_t hnsw_search_knn_batch(
    HNSWIndexHandle index,
    const float* queries,
    size_t num_queries,
    size_t dimension,
    int32_t k,
    uint64_t* labels,
    float* distances
);

void hnsw_set_ef(HNSWIndexHandle index, size_t ef);
bool hnsw_mark_deleted(HNSWIndexHandle index, uint64_t label);
bool hnsw_unmark_deleted(HNSWIndexHandle index, uint64_t label);

// Resize index
bool hnsw_resize_index(HNSWIndexHandle index, size_t new_max_elements);

// Index info
size_t hnsw_get_current_count(HNSWIndexHandle index);
size_t hnsw_get_max_elements(HNSWIndexHandle index);

// Serialization
bool hnsw_save_index(HNSWIndexHandle index, const char* path);
HNSWIndexHandle hnsw_load_index(
    const char* path,
    HNSWSpaceHandle space,
    size_t max_elements
);

#ifdef __cplusplus
}
#endif
