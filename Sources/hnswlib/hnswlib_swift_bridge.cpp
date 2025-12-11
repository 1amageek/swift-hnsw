#include "include/hnswlib_swift_bridge.h"
#include "include/hnswlib.h"
#include <string>

using namespace hnswlib;

extern "C" {

// Space functions
HNSWSpaceHandle hnsw_create_l2_space(size_t dim) {
    try {
        return new L2Space(dim);
    } catch (...) {
        return nullptr;
    }
}

HNSWSpaceHandle hnsw_create_ip_space(size_t dim) {
    try {
        return new InnerProductSpace(dim);
    } catch (...) {
        return nullptr;
    }
}

void hnsw_destroy_space(HNSWSpaceHandle space) {
    if (space) {
        // We need to check the type, but for simplicity we just delete as SpaceInterface
        delete static_cast<SpaceInterface<float>*>(space);
    }
}

// Index functions
HNSWIndexHandle hnsw_create_index(
    HNSWSpaceHandle space,
    size_t max_elements,
    size_t M,
    size_t ef_construction,
    size_t random_seed,
    bool allow_replace_deleted
) {
    try {
        auto* spacePtr = static_cast<SpaceInterface<float>*>(space);
        return new HierarchicalNSW<float>(
            spacePtr,
            max_elements,
            M,
            ef_construction,
            random_seed,
            allow_replace_deleted
        );
    } catch (...) {
        return nullptr;
    }
}

void hnsw_destroy_index(HNSWIndexHandle index) {
    if (index) {
        delete static_cast<HierarchicalNSW<float>*>(index);
    }
}

// Index operations
bool hnsw_add_point(HNSWIndexHandle index, const float* data, uint64_t label) {
    if (!index || !data) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        idx->addPoint(data, static_cast<labeltype>(label), false);
        return true;
    } catch (...) {
        return false;
    }
}

int32_t hnsw_search_knn(
    HNSWIndexHandle index,
    const float* query,
    int32_t k,
    uint64_t* labels,
    float* distances
) {
    if (!index || !query || !labels || !distances) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        auto result = idx->searchKnn(query, static_cast<size_t>(k), nullptr);

        int32_t count = 0;
        while (!result.empty()) {
            auto& top = result.top();
            labels[count] = static_cast<uint64_t>(top.second);
            distances[count] = top.first;
            result.pop();
            count++;
        }

        // Results are in reverse order (furthest first), so reverse them
        for (int32_t i = 0; i < count / 2; i++) {
            std::swap(labels[i], labels[count - 1 - i]);
            std::swap(distances[i], distances[count - 1 - i]);
        }

        return count;
    } catch (...) {
        return 0;
    }
}

void hnsw_set_ef(HNSWIndexHandle index, size_t ef) {
    if (!index) return;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        idx->setEf(ef);
    } catch (...) {
    }
}

bool hnsw_mark_deleted(HNSWIndexHandle index, uint64_t label) {
    if (!index) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        idx->markDelete(static_cast<labeltype>(label));
        return true;
    } catch (...) {
        return false;
    }
}

bool hnsw_unmark_deleted(HNSWIndexHandle index, uint64_t label) {
    if (!index) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        idx->unmarkDelete(static_cast<labeltype>(label));
        return true;
    } catch (...) {
        return false;
    }
}

// Batch operations for high performance
int32_t hnsw_add_points_batch(
    HNSWIndexHandle index,
    const float* data,
    const uint64_t* labels,
    size_t num_points,
    size_t dimension
) {
    if (!index || !data || !labels || num_points == 0) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        int32_t added = 0;
        for (size_t i = 0; i < num_points; i++) {
            try {
                idx->addPoint(data + i * dimension, static_cast<labeltype>(labels[i]), false);
                added++;
            } catch (...) {
                // Skip failed points but continue
            }
        }
        return added;
    } catch (...) {
        return 0;
    }
}

int32_t hnsw_search_knn_batch(
    HNSWIndexHandle index,
    const float* queries,
    size_t num_queries,
    size_t dimension,
    int32_t k,
    uint64_t* labels,
    float* distances
) {
    if (!index || !queries || !labels || !distances || num_queries == 0) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        int32_t total_results = 0;

        for (size_t q = 0; q < num_queries; q++) {
            const float* query = queries + q * dimension;
            uint64_t* result_labels = labels + q * k;
            float* result_distances = distances + q * k;

            auto result = idx->searchKnn(query, static_cast<size_t>(k), nullptr);

            int32_t count = 0;
            while (!result.empty() && count < k) {
                auto& top = result.top();
                result_labels[count] = static_cast<uint64_t>(top.second);
                result_distances[count] = top.first;
                result.pop();
                count++;
            }

            // Results are in reverse order (furthest first), so reverse them
            for (int32_t i = 0; i < count / 2; i++) {
                std::swap(result_labels[i], result_labels[count - 1 - i]);
                std::swap(result_distances[i], result_distances[count - 1 - i]);
            }

            total_results += count;
        }

        return total_results;
    } catch (...) {
        return 0;
    }
}

bool hnsw_resize_index(HNSWIndexHandle index, size_t new_max_elements) {
    if (!index) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        idx->resizeIndex(new_max_elements);
        return true;
    } catch (...) {
        return false;
    }
}

// Index info
size_t hnsw_get_current_count(HNSWIndexHandle index) {
    if (!index) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        return idx->getCurrentElementCount();
    } catch (...) {
        return 0;
    }
}

size_t hnsw_get_max_elements(HNSWIndexHandle index) {
    if (!index) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        return idx->getMaxElements();
    } catch (...) {
        return 0;
    }
}

// Serialization
bool hnsw_save_index(HNSWIndexHandle index, const char* path) {
    if (!index || !path) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        idx->saveIndex(std::string(path));
        return true;
    } catch (...) {
        return false;
    }
}

HNSWIndexHandle hnsw_load_index(
    const char* path,
    HNSWSpaceHandle space,
    size_t max_elements
) {
    if (!path || !space) return nullptr;
    try {
        auto* spacePtr = static_cast<SpaceInterface<float>*>(space);
        return new HierarchicalNSW<float>(spacePtr, std::string(path), false, max_elements, false);
    } catch (...) {
        return nullptr;
    }
}

// Serialization - Memory buffer
size_t hnsw_get_serialized_size(HNSWIndexHandle index) {
    if (!index) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        return idx->indexFileSize();
    } catch (...) {
        return 0;
    }
}

bool hnsw_serialize_to_buffer(HNSWIndexHandle index, void* buffer, size_t buffer_size) {
    if (!index || !buffer) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);

        // Check buffer size
        size_t required_size = idx->indexFileSize();
        if (buffer_size < required_size) return false;

        char* ptr = static_cast<char*>(buffer);
        size_t offset = 0;

        // Write header fields (same order as saveIndex)
        auto write_pod = [&](const auto& value) {
            memcpy(ptr + offset, &value, sizeof(value));
            offset += sizeof(value);
        };

        write_pod(idx->offsetLevel0_);
        write_pod(idx->max_elements_);
        write_pod(idx->cur_element_count.load());
        write_pod(idx->size_data_per_element_);
        write_pod(idx->label_offset_);
        write_pod(idx->offsetData_);
        write_pod(idx->maxlevel_);
        write_pod(idx->enterpoint_node_);
        write_pod(idx->maxM_);
        write_pod(idx->maxM0_);
        write_pod(idx->M_);
        write_pod(idx->mult_);
        write_pod(idx->ef_construction_);

        // Write level0 data
        size_t cur_count = idx->cur_element_count.load();
        memcpy(ptr + offset, idx->data_level0_memory_, cur_count * idx->size_data_per_element_);
        offset += cur_count * idx->size_data_per_element_;

        // Write link lists for each element
        for (size_t i = 0; i < cur_count; i++) {
            unsigned int linkListSize = idx->element_levels_[i] > 0
                ? static_cast<unsigned int>(idx->size_links_per_element_ * idx->element_levels_[i])
                : 0;
            memcpy(ptr + offset, &linkListSize, sizeof(linkListSize));
            offset += sizeof(linkListSize);
            if (linkListSize) {
                memcpy(ptr + offset, idx->linkLists_[i], linkListSize);
                offset += linkListSize;
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

HNSWIndexHandle hnsw_load_from_buffer(
    const void* buffer,
    size_t buffer_size,
    HNSWSpaceHandle space,
    size_t max_elements_override
) {
    if (!buffer || !space || buffer_size == 0) return nullptr;
    try {
        auto* spacePtr = static_cast<SpaceInterface<float>*>(space);
        auto* idx = new HierarchicalNSW<float>(spacePtr);

        const char* ptr = static_cast<const char*>(buffer);
        size_t offset = 0;

        auto read_pod = [&](auto& value) {
            memcpy(&value, ptr + offset, sizeof(value));
            offset += sizeof(value);
        };

        // Read header
        read_pod(idx->offsetLevel0_);
        read_pod(idx->max_elements_);

        size_t cur_element_count_val;
        read_pod(cur_element_count_val);
        idx->cur_element_count = cur_element_count_val;

        read_pod(idx->size_data_per_element_);
        read_pod(idx->label_offset_);
        read_pod(idx->offsetData_);
        read_pod(idx->maxlevel_);
        read_pod(idx->enterpoint_node_);
        read_pod(idx->maxM_);
        read_pod(idx->maxM0_);
        read_pod(idx->M_);
        read_pod(idx->mult_);
        read_pod(idx->ef_construction_);

        // Determine max elements
        size_t max_elements = max_elements_override;
        if (max_elements < idx->cur_element_count.load()) {
            max_elements = idx->max_elements_;
        }
        idx->max_elements_ = max_elements;

        // Setup space
        idx->data_size_ = spacePtr->get_data_size();
        idx->fstdistfunc_ = spacePtr->get_dist_func();
        idx->dist_func_param_ = spacePtr->get_dist_func_param();

        // Allocate and read level0 data
        idx->data_level0_memory_ = (char*)malloc(max_elements * idx->size_data_per_element_);
        if (!idx->data_level0_memory_) {
            delete idx;
            return nullptr;
        }
        memcpy(idx->data_level0_memory_, ptr + offset, cur_element_count_val * idx->size_data_per_element_);
        offset += cur_element_count_val * idx->size_data_per_element_;

        // Initialize structures
        idx->size_links_per_element_ = idx->maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        idx->size_links_level0_ = idx->maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        std::vector<std::mutex>(max_elements).swap(idx->link_list_locks_);
        std::vector<std::mutex>(HierarchicalNSW<float>::MAX_LABEL_OPERATION_LOCKS).swap(idx->label_op_locks_);

        idx->visited_list_pool_.reset(new VisitedListPool(1, static_cast<int>(max_elements)));

        idx->linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
        if (!idx->linkLists_) {
            free(idx->data_level0_memory_);
            delete idx;
            return nullptr;
        }

        idx->element_levels_ = std::vector<int>(max_elements);
        idx->revSize_ = 1.0 / idx->mult_;
        idx->ef_ = 10;

        // Read link lists
        for (size_t i = 0; i < cur_element_count_val; i++) {
            labeltype label = idx->getExternalLabel(static_cast<tableint>(i));
            idx->label_lookup_[label] = static_cast<tableint>(i);

            unsigned int linkListSize;
            memcpy(&linkListSize, ptr + offset, sizeof(linkListSize));
            offset += sizeof(linkListSize);

            if (linkListSize == 0) {
                idx->element_levels_[i] = 0;
                idx->linkLists_[i] = nullptr;
            } else {
                idx->element_levels_[i] = linkListSize / idx->size_links_per_element_;
                idx->linkLists_[i] = (char*)malloc(linkListSize);
                if (!idx->linkLists_[i]) {
                    // Cleanup on failure
                    for (size_t j = 0; j < i; j++) {
                        if (idx->linkLists_[j]) free(idx->linkLists_[j]);
                    }
                    free(idx->linkLists_);
                    free(idx->data_level0_memory_);
                    delete idx;
                    return nullptr;
                }
                memcpy(idx->linkLists_[i], ptr + offset, linkListSize);
                offset += linkListSize;
            }
        }

        // Count deleted elements
        for (size_t i = 0; i < cur_element_count_val; i++) {
            if (idx->isMarkedDeleted(static_cast<tableint>(i))) {
                idx->num_deleted_ += 1;
            }
        }

        return idx;
    } catch (...) {
        return nullptr;
    }
}

// Label operations
bool hnsw_contains_label(HNSWIndexHandle index, uint64_t label) {
    if (!index) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        std::unique_lock<std::mutex> lock(idx->label_lookup_lock);
        auto search = idx->label_lookup_.find(static_cast<labeltype>(label));
        if (search == idx->label_lookup_.end()) {
            return false;
        }
        // Check if marked as deleted
        return !idx->isMarkedDeleted(search->second);
    } catch (...) {
        return false;
    }
}

bool hnsw_get_vector(
    HNSWIndexHandle index,
    uint64_t label,
    float* output,
    size_t dimension
) {
    if (!index || !output) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        std::vector<float> data = idx->template getDataByLabel<float>(static_cast<labeltype>(label));
        if (data.size() != dimension) return false;
        memcpy(output, data.data(), dimension * sizeof(float));
        return true;
    } catch (...) {
        return false;
    }
}

size_t hnsw_get_all_labels(
    HNSWIndexHandle index,
    uint64_t* output,
    size_t max_count
) {
    if (!index) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        std::unique_lock<std::mutex> lock(idx->label_lookup_lock);

        size_t count = 0;
        for (const auto& pair : idx->label_lookup_) {
            // Skip deleted elements
            if (idx->isMarkedDeleted(pair.second)) continue;

            if (output && count < max_count) {
                output[count] = static_cast<uint64_t>(pair.first);
            }
            count++;
        }
        return count;
    } catch (...) {
        return 0;
    }
}

// ============================================================
// Float16 Support
// ============================================================

HNSWSpaceHandle hnsw_create_l2_space_f16(size_t dim) {
    try {
        return new L2SpaceF16(dim);
    } catch (...) {
        return nullptr;
    }
}

HNSWSpaceHandle hnsw_create_ip_space_f16(size_t dim) {
    try {
        return new InnerProductSpaceF16(dim);
    } catch (...) {
        return nullptr;
    }
}

bool hnsw_add_point_f16(HNSWIndexHandle index, const uint16_t* data, uint64_t label) {
    if (!index || !data) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        // Data is already in Float16 format (uint16_t binary representation)
        // The distance function will handle it correctly
        idx->addPoint(data, static_cast<labeltype>(label), false);
        return true;
    } catch (...) {
        return false;
    }
}

int32_t hnsw_search_knn_f16(
    HNSWIndexHandle index,
    const uint16_t* query,
    int32_t k,
    uint64_t* labels,
    float* distances
) {
    if (!index || !query || !labels || !distances) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        auto result = idx->searchKnn(query, static_cast<size_t>(k), nullptr);

        int32_t count = 0;
        while (!result.empty()) {
            auto& top = result.top();
            labels[count] = static_cast<uint64_t>(top.second);
            distances[count] = top.first;
            result.pop();
            count++;
        }

        // Results are in reverse order (furthest first), so reverse them
        for (int32_t i = 0; i < count / 2; i++) {
            std::swap(labels[i], labels[count - 1 - i]);
            std::swap(distances[i], distances[count - 1 - i]);
        }

        return count;
    } catch (...) {
        return 0;
    }
}

int32_t hnsw_add_points_batch_f16(
    HNSWIndexHandle index,
    const uint16_t* data,
    const uint64_t* labels,
    size_t num_points,
    size_t dimension
) {
    if (!index || !data || !labels || num_points == 0) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        int32_t added = 0;
        for (size_t i = 0; i < num_points; i++) {
            try {
                idx->addPoint(data + i * dimension, static_cast<labeltype>(labels[i]), false);
                added++;
            } catch (...) {
                // Skip failed points but continue
            }
        }
        return added;
    } catch (...) {
        return 0;
    }
}

int32_t hnsw_search_knn_batch_f16(
    HNSWIndexHandle index,
    const uint16_t* queries,
    size_t num_queries,
    size_t dimension,
    int32_t k,
    uint64_t* labels,
    float* distances
) {
    if (!index || !queries || !labels || !distances || num_queries == 0) return 0;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        int32_t total_results = 0;

        for (size_t q = 0; q < num_queries; q++) {
            const uint16_t* query = queries + q * dimension;
            uint64_t* result_labels = labels + q * k;
            float* result_distances = distances + q * k;

            auto result = idx->searchKnn(query, static_cast<size_t>(k), nullptr);

            int32_t count = 0;
            while (!result.empty() && count < k) {
                auto& top = result.top();
                result_labels[count] = static_cast<uint64_t>(top.second);
                result_distances[count] = top.first;
                result.pop();
                count++;
            }

            // Results are in reverse order (furthest first), so reverse them
            for (int32_t i = 0; i < count / 2; i++) {
                std::swap(result_labels[i], result_labels[count - 1 - i]);
                std::swap(result_distances[i], result_distances[count - 1 - i]);
            }

            total_results += count;
        }

        return total_results;
    } catch (...) {
        return 0;
    }
}

bool hnsw_get_vector_f16(
    HNSWIndexHandle index,
    uint64_t label,
    uint16_t* output,
    size_t dimension
) {
    if (!index || !output) return false;
    try {
        auto* idx = static_cast<HierarchicalNSW<float>*>(index);
        // Get the raw data pointer for this label
        std::unique_lock<std::mutex> lock(idx->label_lookup_lock);
        auto search = idx->label_lookup_.find(static_cast<labeltype>(label));
        if (search == idx->label_lookup_.end()) {
            return false;
        }
        tableint internal_id = search->second;
        lock.unlock();

        // Get pointer to the stored data (which is in Float16 format)
        char* data_ptr = idx->getDataByInternalId(internal_id);
        memcpy(output, data_ptr, dimension * sizeof(uint16_t));
        return true;
    } catch (...) {
        return false;
    }
}

} // extern "C"
