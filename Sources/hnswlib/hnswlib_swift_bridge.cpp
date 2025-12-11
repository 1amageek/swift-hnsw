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

} // extern "C"
