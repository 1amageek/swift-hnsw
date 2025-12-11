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

// Serialization - File
bool hnsw_save_index(HNSWIndexHandle index, const char* path);
HNSWIndexHandle hnsw_load_index(
    const char* path,
    HNSWSpaceHandle space,
    size_t max_elements
);

// Serialization - Memory buffer
size_t hnsw_get_serialized_size(HNSWIndexHandle index);
bool hnsw_serialize_to_buffer(HNSWIndexHandle index, void* buffer, size_t buffer_size);
HNSWIndexHandle hnsw_load_from_buffer(
    const void* buffer,
    size_t buffer_size,
    HNSWSpaceHandle space,
    size_t max_elements
);

// Label operations
bool hnsw_contains_label(HNSWIndexHandle index, uint64_t label);
bool hnsw_get_vector(
    HNSWIndexHandle index,
    uint64_t label,
    float* output,
    size_t dimension
);
size_t hnsw_get_all_labels(
    HNSWIndexHandle index,
    uint64_t* output,
    size_t max_count
);

// ============================================================
// Float16 Support
// ============================================================

// Float16 Space functions
HNSWSpaceHandle hnsw_create_l2_space_f16(size_t dim);
HNSWSpaceHandle hnsw_create_ip_space_f16(size_t dim);

// Float16 Index operations (data as uint16_t* for Float16 binary representation)
bool hnsw_add_point_f16(HNSWIndexHandle index, const uint16_t* data, uint64_t label);
int32_t hnsw_search_knn_f16(
    HNSWIndexHandle index,
    const uint16_t* query,
    int32_t k,
    uint64_t* labels,
    float* distances
);

// Float16 Batch operations
int32_t hnsw_add_points_batch_f16(
    HNSWIndexHandle index,
    const uint16_t* data,
    const uint64_t* labels,
    size_t num_points,
    size_t dimension
);

int32_t hnsw_search_knn_batch_f16(
    HNSWIndexHandle index,
    const uint16_t* queries,
    size_t num_queries,
    size_t dimension,
    int32_t k,
    uint64_t* labels,
    float* distances
);

// Float16 Label operations
bool hnsw_get_vector_f16(
    HNSWIndexHandle index,
    uint64_t label,
    uint16_t* output,
    size_t dimension
);

#ifdef __cplusplus
}
#endif
