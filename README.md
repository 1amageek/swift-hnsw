# SwiftHNSW

A Swift vector search library for Hierarchical Navigable Small World style nearest-neighbor APIs.
The production backend is implemented in Swift and supports WebAssembly builds. A separate C++ reference benchmark package uses [hnswlib](https://github.com/nmslib/hnswlib) only for performance comparison.

## Features

- **Swift Backend Only in Production**: Default builds do not include or compile C++ targets
- **Reference C++ Benchmarks**: hnswlib is kept in a separate benchmark package for comparison only
- **Fast Vector Search**: Nearest-neighbor queries through a stable Swift API
- **Contiguous Runtime Storage**: Pure Swift stores Float32 and Float16 vectors in type-specific contiguous arenas and graph edges in a fixed-slot connection store
- **Float16 Support**: Native half-precision for 50% memory reduction
- **TurboQuant**: 4-bit vector quantization with HD³ rotation for 6-8x memory compression
- **Multiple Distance Metrics**: L2 (Euclidean), Inner Product, and Cosine similarity
- **Thread-Safe**: Sendable API with backend-specific synchronization
- **Batch Operations**: Efficient bulk add and search operations
- **Persistence**: Save and load indexes to/from disk
- **Swift 6 Ready**: Full Sendable conformance for modern concurrency

## Requirements

- Swift 6.2+
- macOS 13+ / iOS 16+
- Swift WebAssembly SDK for WASM builds
- C++17 toolchain only for the separate reference benchmark package

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-hnsw.git", from: "1.0.0")
]
```

Then add `SwiftHNSW` to your target dependencies:

```swift
.target(
    name: "YourTarget",
    dependencies: ["SwiftHNSW"]
)
```

### Backend Selection

SwiftHNSW uses the Swift backend in production. The root package has no C++ target:

```bash
swift build
swift test
swift build --swift-sdk swift-6.3.1-RELEASE_wasm
```

Run the optional hnswlib reference comparison from its separate benchmark package:

```bash
BACKEND_COMPARISON_ITERATIONS=5 swift run -c release --package-path Benchmarks/CxxBackendComparison CxxBackendComparison
```

The Pure Swift backend is the portability and production baseline. The C++ code is retained only as a benchmark reference and is not part of default library builds.

## Quick Start

### Float32 (Standard Precision)

```swift
import SwiftHNSW

// Create an index with Float32 precision
let index = try HNSWIndex<Float>(
    dimensions: 128,
    maxElements: 10000,
    metric: .l2
)

// Add vectors
try index.add([1.0, 0.5, 0.3, ...], label: 0)
try index.add([0.2, 0.8, 0.1, ...], label: 1)

// Search for nearest neighbors
let results = try index.search([1.0, 0.5, 0.3, ...], k: 10)

for result in results {
    print("Label: \(result.label), Distance: \(result.distance)")
}
```

### Float16 (Half Precision)

```swift
import SwiftHNSW

// Create an index with Float16 precision (50% memory savings)
let index = try HNSWIndex<Float16>(
    dimensions: 384,
    maxElements: 100000,
    metric: .l2
)

// Add vectors
let embedding: [Float16] = [0.5, 0.3, 0.1, ...]
try index.add(embedding, label: 0)

// Search
let query: [Float16] = [0.5, 0.3, 0.1, ...]
let results = try index.search(query, k: 10)
```

### Type Aliases

For convenience, type aliases are provided:

```swift
typealias HNSWIndexF32 = HNSWIndex<Float>    // Standard precision
typealias HNSWIndexF16 = HNSWIndex<Float16>  // Half precision
```

## Float16 vs Float32

| Aspect | Float32 | Float16 |
|--------|---------|---------|
| **Memory** | 4 bytes/element | 2 bytes/element |
| **Precision** | ~7 digits | ~3 digits |
| **Recall** | Baseline | ~Same (within 1%) |
| **Speed** | Faster for Float32-heavy workloads | Competitive when memory bandwidth matters* |
| **Use Case** | General purpose | Large indexes, memory-constrained |

\* Float16 is stored as half precision and accumulated in Float precision during SIMD distance kernels.

### When to Use Float16

- **Large indexes** (millions of vectors) where memory is the bottleneck
- **High-dimensional embeddings** (384, 768, 1536 dimensions)
- **Mobile/embedded** devices with limited RAM
- **Cloud deployments** where memory = cost

### When to Use Float32

- **Maximum search speed** is critical
- **Small to medium indexes** that fit comfortably in RAM
- **High precision** requirements

## API Reference

### Creating an Index

```swift
// Float32 index
let index = try HNSWIndex<Float>(
    dimensions: 128,           // Vector dimensionality
    maxElements: 10000,        // Maximum capacity
    metric: .l2,               // Distance metric
    configuration: .balanced   // HNSW parameters
)

// Float16 index
let indexF16 = try HNSWIndex<Float16>(
    dimensions: 384,
    maxElements: 100000,
    metric: .cosine,
    configuration: .highAccuracy
)
```

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `.l2` | Euclidean distance | General purpose |
| `.innerProduct` | Inner product (1 - dot product) | Normalized embeddings |
| `.cosine` | Cosine similarity | Text embeddings |

### Configuration Presets

| Preset | M | efConstruction | efSearch | Description |
|--------|---|----------------|----------|-------------|
| `.fast` | 8 | 100 | 10 | Speed optimized |
| `.balanced` | 16 | 200 | 50 | Default, good trade-off |
| `.highAccuracy` | 32 | 400 | 200 | Maximum recall |

Custom configuration:

```swift
let config = HNSWConfiguration(
    m: 24,
    efConstruction: 300,
    efSearch: 100
)
```

### Adding Vectors

**Single vector:**
```swift
try index.add([1.0, 0.5, 0.3, ...], label: 0)
```

**Batch (flattened array):**
```swift
let vectors: [Float] = [...] // n * dimensions elements
let labels: [UInt64] = [0, 1, 2, ..., n-1]
try index.addBatch(vectors, labels: labels)
```

**Batch (2D array with auto-labels):**
```swift
let vectors: [[Float]] = [[1.0, 0.5, ...], [0.3, 0.2, ...], ...]
try index.addBatch(vectors, startingLabel: 0)
```

### Searching

**Single query:**
```swift
let results = try index.search(queryVector, k: 10)
// Returns [SearchResult] sorted by distance (closest first)
```

**Batch queries:**
```swift
let queries: [[Float]] = [...]
let allResults = try index.searchBatch(queries, k: 10)
// Returns [[SearchResult]] for each query
```

**Tuning search accuracy:**
```swift
index.setEfSearch(100)  // Higher = better recall, slower search
```

**Searching into caller-owned storage:**
```swift
var output = [SearchResult](repeating: SearchResult(label: 0, distance: 0), count: 10)
let count = try output.withUnsafeMutableBufferPointer { results in
    try index.search(queryVector, k: 10, into: results)
}
print(output.prefix(count))
```

Use `search(_:k:into:)` on latency-sensitive paths that already own a reusable
result buffer. The standard `search(_:k:)` API remains the ergonomic option when
an owned `[SearchResult]` is desired.

### Retrieving Vectors

```swift
// Materialize the stored vector as an Array for display or ownership transfer.
if let vector: [Float] = index.getVector(label: 0) {
    print("Vector: \(vector)")
}

// For Float16 index
if let vector: [Float16] = indexF16.getVector(label: 0) {
    print("Vector: \(vector)")
}
```

For performance-sensitive reads, borrow the stored vector instead of materializing
an intermediate array:

```swift
let magnitude = index.withVector(label: 0) { vector in
    vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
}
```

### Deletion

```swift
// Soft delete (mark as deleted)
try index.markDeleted(label: 0)

// Restore deleted element
try index.unmarkDeleted(label: 0)
```

### Persistence

```swift
// Save index
try index.save(to: URL(fileURLWithPath: "/path/to/index.dat"))

// Load Float32 index
let loadedIndex = try HNSWIndex<Float>.load(
    from: "/path/to/index.dat",
    dimensions: 128,
    metric: .l2
)

// Load Float16 index
let loadedIndexF16 = try HNSWIndex<Float16>.load(
    from: "/path/to/index_f16.dat",
    dimensions: 384,
    metric: .l2
)
```

### Index Info

```swift
index.count      // Current number of elements
index.capacity   // Maximum capacity
index.isEmpty    // Boolean check
```

## Backend Semantics

| Backend | Selection | Search behavior | Synchronization |
|---------|-----------|-----------------|-----------------|
| Swift backend | Root package default | Approximate HNSW graph search | Serialized state access with `Mutex` |
| C++ reference | `Benchmarks/CxxBackendComparison` only | hnswlib reference search | Benchmark-local bridge lock / native hnswlib state |

The root package exposes the Swift backend API and typed errors. The C++ reference is intentionally outside the production package graph.

## HNSW Algorithm

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor search algorithm. It builds a multi-layer graph structure where:

- **Layer 0** contains all elements with maximum `M * 2` connections
- **Higher layers** contain progressively fewer elements with maximum `M` connections
- **Search** starts from the top layer and greedily descends to find nearest neighbors

### Key Parameters

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `M` | Connections per element | Memory vs Recall |
| `efConstruction` | Build-time search depth | Build time vs Recall |
| `efSearch` | Query-time search depth | Query time vs Recall |

### Performance Characteristics

- **Build time**: O(n * log(n)) average
- **Search time**: O(log(n)) average
- **Memory**: O(n * M) for connections + O(n * d * sizeof(Scalar)) for vectors

The Swift backend follows the same graph model with Swift-native storage: level 0 uses direct fixed-width neighbor slots keyed by internal ID, upper levels use compact fixed slots, and internal graph IDs are stored as `UInt32`.

## C++ Reference Benchmark

hnswlib is retained under `Benchmarks/CxxBackendComparison` to compare the Swift production backend against a native C++ reference. It is not a product dependency and is not compiled by `swift build` or `swift test` at the package root.

## Thread Safety

`HNSWIndex` is thread-safe with the following guarantees:

- **Swift backend**: State access is serialized with `Mutex`
- **C++ reference**: Used only by the benchmark package
- **Sendable**: Safe to use across actor boundaries

```swift
// Concurrent search example
await withTaskGroup(of: [SearchResult].self) { group in
    for query in queries {
        group.addTask {
            try! index.search(query, k: 10)
        }
    }
}
```

## Reference Backend Benchmarks

**Measurement methodology:**

- **Platform**: Apple Silicon (arm64), macOS 26.5.1
- **Swift**: 6.3.1 (`swift-6.3.1-RELEASE`)
- **Build**: Release (`swift run -c release --package-path Benchmarks/CxxBackendComparison CxxBackendComparison`)
- **Backends**: Swift production backend and hnswlib reference backend
- **Data**: Random vectors sampled uniformly from [-1, 1]
- **Ground truth**: Brute-force exact L2 search
- **Recall@k**: Fraction of true k-nearest neighbors found in the approximate result
- **QPS**: Queries per second (single-threaded, excluding index construction)
- **Timing**: `ContinuousClock`, median and p95 over repeated full-query batches

Run backend parity measurements with median and p95 search latency output:

```bash
BACKEND_COMPARISON_ITERATIONS=3 swift run -c release --package-path Benchmarks/CxxBackendComparison CxxBackendComparison
```

### Swift vs C++ Reference Results

The production Swift backend is within the target parity threshold and was faster than the hnswlib reference in these search measurements. Positive values in `Swift vs C++` mean Swift is faster.

| Scalar | Case | efSearch | Swift latency | C++ latency | Swift vs C++ | Swift recall@10 | C++ recall@10 |
|--------|------|---------:|--------------:|------------:|-------------:|----------------:|--------------:|
| Float32 | 10k x 128 | 100 | 0.058 ms | 0.062 ms | 7.2% faster | 0.8186 | 0.8206 |
| Float32 | 10k x 128 | 320 | 0.160 ms | 0.167 ms | 4.2% faster | 0.9838 | 0.9836 |
| Float32 | 50k x 128 | 100 | 0.112 ms | 0.118 ms | 5.0% faster | 0.5515 | 0.5580 |
| Float32 | 50k x 128 | 320 | 0.334 ms | 0.370 ms | 9.7% faster | 0.8355 | 0.8340 |
| Float32 | 50k x 128 | 1000 | 0.946 ms | 1.009 ms | 6.3% faster | 0.9730 | 0.9735 |
| Float16 | 10k x 128 | 100 | 0.056 ms | 0.058 ms | 3.5% faster | 0.8172 | 0.8188 |
| Float16 | 10k x 128 | 320 | 0.148 ms | 0.159 ms | 6.7% faster | 0.9796 | 0.9802 |

Build throughput:

| Scalar | Case | Swift build | C++ build | Swift vs C++ |
|--------|------|------------:|----------:|-------------:|
| Float32 | 10k x 128 | 8,356 vectors/s | 8,052 vectors/s | 3.8% faster |
| Float32 | 50k x 128 | 4,695 vectors/s | 4,519 vectors/s | 3.9% faster |
| Float16 | 10k x 128 | 8,310 vectors/s | 7,945 vectors/s | 4.6% faster |

The C++ reference remains useful as a regression comparison target, but it is not part of the production package graph.

Run the Swift backend performance smoke tests with:

```bash
SWIFT_BACKEND_PERF=1 swift test
```

### Swift Backend Recall vs QPS Trade-off (128-dim, 10K vectors)

| efSearch | Recall@10 | QPS | Latency |
|----------|-----------|-----|---------|
| 10 | 28.2% | 112,493 | 0.009ms |
| 20 | 44.8% | 48,828 | 0.020ms |
| 40 | 63.1% | 31,980 | 0.031ms |
| 80 | 80.7% | 18,215 | 0.055ms |
| 160 | 92.7% | 9,839 | 0.102ms |
| 320 | 98.1% | 5,544 | 0.180ms |

### Swift Backend Scale Performance (128-dim, M=16, efConstruction=200, efSearch=100)

| Vectors | Build Time | Build Rate | Search QPS | Latency | Index Size |
|---------|------------|------------|------------|---------|------------|
| 1,000 | 0.05s | 21,359/s | 27,840 | 0.036ms | 750 KB |
| 5,000 | 0.48s | 10,430/s | 16,978 | 0.059ms | 3.7 MB |
| 10,000 | 1.23s | 8,145/s | 16,239 | 0.062ms | 7.3 MB |
| 20,000 | 3.31s | 6,041/s | 14,384 | 0.070ms | 14.6 MB |
| 50,000 | 10.77s | 4,642/s | 8,087 | 0.124ms | 36.6 MB |

### Float16 vs Float32 (384-dim, 10K vectors)

| Metric | Float32 | Float16 | Comparison |
|--------|---------|---------|------------|
| Vector Memory | 14.6 MB | 7.3 MB | **50% smaller** |
| Build Time | 2.85s | 2.79s | 1.02x |
| Search QPS | 7,222 | 8,641 | **1.20x faster** |
| Search Latency | 0.138ms | 0.116ms | 1.20x |
| Recall@10 | 69.7% | 69.9% | **Same** |

### TurboQuant Recall vs QPS (128-dim, 5K vectors)

| Algorithm | efSearch | Recall@10 | QPS | Memory | Compression |
|-----------|----------|-----------|-----|--------|-------------|
| Float32 | 100 | 89.5% | 18,041 | 2.4 MB | 1.0x |
| Float32 | 320 | 99.7% | 7,121 | 2.4 MB | 1.0x |
| TQ-4bit | 100 | 75.3% | 9,501 | 312 KB | **8.0x** |
| TQ-4bit | 320 | 80.9% | 4,184 | 312 KB | **8.0x** |
| TQ-2bit | 100 | 44.9% | 11,893 | 156 KB | **16.0x** |
| TQ-2bit | 320 | 47.5% | 5,103 | 156 KB | **16.0x** |

### Distance Metrics (128-dim, 5K vectors)

| Metric | Build Time | Search QPS | Latency |
|--------|------------|------------|---------|
| L2 (Euclidean) | 0.48s | 18,004 | 0.056ms |
| Inner Product | 0.44s | 17,788 | 0.056ms |
| Cosine | 0.44s | 17,024 | 0.059ms |

### Concurrent Read Scaling (128-dim, 10K vectors)

| Threads | Total QPS | Per-Thread QPS | Speedup |
|---------|-----------|----------------|---------|
| 1 | 16,059 | 16,059 | 1.0x |
| 2 | 24,325 | 12,163 | 1.51x |
| 4 | 29,227 | 7,307 | 1.82x |
| 8 | 32,674 | 4,084 | 2.03x |

## Error Handling

```swift
do {
    try index.add(vector, label: 0)
} catch HNSWError.dimensionMismatch(let expected, let got) {
    print("Expected \(expected) dimensions, got \(got)")
} catch HNSWError.capacityExceeded(let current, let maximum) {
    print("Index full: \(current)/\(maximum)")
}
```

## License

The C++ reference benchmark wraps [hnswlib](https://github.com/nmslib/hnswlib), which is licensed under Apache 2.0.

## Acknowledgments

- [hnswlib](https://github.com/nmslib/hnswlib) - C++ reference benchmark backend
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) - Original HNSW paper
