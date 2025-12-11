# SwiftHNSW

A high-performance Swift wrapper for [hnswlib](https://github.com/nmslib/hnswlib) - Hierarchical Navigable Small World graphs for approximate nearest neighbor search.

## Features

- **Fast ANN Search**: Sub-millisecond approximate nearest neighbor queries
- **Multiple Distance Metrics**: L2 (Euclidean), Inner Product, and Cosine similarity
- **Thread-Safe**: Concurrent read access with RWLock synchronization
- **SIMD Optimized**: Leverages Apple Accelerate framework for vector operations
- **Batch Operations**: Efficient bulk add and search operations
- **Persistence**: Save and load indexes to/from disk
- **Swift 6 Ready**: Full Sendable conformance for modern concurrency

## Requirements

- Swift 6.2+
- macOS 13+ / iOS 16+

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

## Quick Start

```swift
import SwiftHNSW

// Create an index
let index = try HNSWIndex(
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

## API Reference

### Creating an Index

```swift
let index = try HNSWIndex(
    dimensions: 128,           // Vector dimensionality
    maxElements: 10000,        // Maximum capacity
    metric: .l2,               // Distance metric
    configuration: .balanced   // HNSW parameters
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

// Load index
let loadedIndex = try HNSWIndex.load(
    from: "/path/to/index.dat",
    dimensions: 128,
    metric: .l2
)
```

### Index Info

```swift
index.count      // Current number of elements
index.capacity   // Maximum capacity
index.isEmpty    // Boolean check
```

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
- **Memory**: O(n * M) for connections + O(n * d) for vectors

## Thread Safety

`HNSWIndex` is thread-safe with the following guarantees:

- **Multiple concurrent reads**: Search operations can run in parallel
- **Exclusive writes**: Add/delete operations have exclusive access
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

## Benchmarks

Run the included benchmarks:

```bash
swift test --filter ANNBenchmarks
```

Typical performance (M1 MacBook Pro, 128-dim vectors):

| Operation | 10K vectors | 100K vectors |
|-----------|-------------|--------------|
| Build | ~50ms | ~600ms |
| Search (k=10) | ~0.1ms | ~0.2ms |
| Recall@10 | >95% | >95% |

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

This project wraps [hnswlib](https://github.com/nmslib/hnswlib) which is licensed under Apache 2.0.

## Acknowledgments

- [hnswlib](https://github.com/nmslib/hnswlib) - The underlying C++ library
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) - Original HNSW paper
