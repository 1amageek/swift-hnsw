#include "TurboQuantBenchmarkReference.h"

float swift_hnsw_benchmark_four_bit_inner_product(
    const uint8_t *restrict packed_codes,
    const float *restrict query,
    size_t dimension,
    const float *restrict centroids
) {
    const size_t pair_count = dimension / 2;
    size_t byte_index = 0;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    float accumulator2 = 0.0f;
    float accumulator3 = 0.0f;

    for (; byte_index + 4 <= pair_count; byte_index += 4) {
        const size_t coordinate = byte_index * 2;
        const uint8_t packed0 = packed_codes[byte_index];
        const uint8_t packed1 = packed_codes[byte_index + 1];
        const uint8_t packed2 = packed_codes[byte_index + 2];
        const uint8_t packed3 = packed_codes[byte_index + 3];
        accumulator0 += query[coordinate] * centroids[packed0 >> 4];
        accumulator0 += query[coordinate + 1] * centroids[packed0 & 0x0f];
        accumulator1 += query[coordinate + 2] * centroids[packed1 >> 4];
        accumulator1 += query[coordinate + 3] * centroids[packed1 & 0x0f];
        accumulator2 += query[coordinate + 4] * centroids[packed2 >> 4];
        accumulator2 += query[coordinate + 5] * centroids[packed2 & 0x0f];
        accumulator3 += query[coordinate + 6] * centroids[packed3 >> 4];
        accumulator3 += query[coordinate + 7] * centroids[packed3 & 0x0f];
    }

    float total = accumulator0 + accumulator1 + accumulator2 + accumulator3;
    for (; byte_index < pair_count; ++byte_index) {
        const size_t coordinate = byte_index * 2;
        const uint8_t packed = packed_codes[byte_index];
        total += query[coordinate] * centroids[packed >> 4];
        total += query[coordinate + 1] * centroids[packed & 0x0f];
    }
    if ((dimension & 1) != 0) {
        total += query[dimension - 1] * centroids[packed_codes[pair_count] >> 4];
    }
    return total;
}
