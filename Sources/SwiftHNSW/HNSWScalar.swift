import Foundation
import hnswlib

/// Protocol for scalar types supported by HNSW index
public protocol HNSWScalar: BinaryFloatingPoint, Sendable {
    /// Create L2 space handle for this scalar type
    static func createL2Space(dimensions: Int) -> HNSWSpaceHandle?

    /// Create Inner Product space handle for this scalar type
    static func createIPSpace(dimensions: Int) -> HNSWSpaceHandle?

    /// Add a point to the index
    static func addPoint(
        _ index: HNSWIndexHandle,
        data: UnsafePointer<Self>,
        label: UInt64
    ) -> Bool

    /// Search k nearest neighbors
    static func searchKnn(
        _ index: HNSWIndexHandle,
        query: UnsafePointer<Self>,
        k: Int32,
        labels: UnsafeMutablePointer<UInt64>,
        distances: UnsafeMutablePointer<Float>
    ) -> Int32

    /// Add points in batch
    static func addPointsBatch(
        _ index: HNSWIndexHandle,
        data: UnsafePointer<Self>,
        labels: UnsafePointer<UInt64>,
        numPoints: Int,
        dimension: Int
    ) -> Int32

    /// Search k nearest neighbors in batch
    static func searchKnnBatch(
        _ index: HNSWIndexHandle,
        queries: UnsafePointer<Self>,
        numQueries: Int,
        dimension: Int,
        k: Int32,
        labels: UnsafeMutablePointer<UInt64>,
        distances: UnsafeMutablePointer<Float>
    ) -> Int32

    /// Get vector by label
    static func getVector(
        _ index: HNSWIndexHandle,
        label: UInt64,
        output: UnsafeMutablePointer<Self>,
        dimension: Int
    ) -> Bool
}

// MARK: - Float Conformance

extension Float: HNSWScalar {
    public static func createL2Space(dimensions: Int) -> HNSWSpaceHandle? {
        hnsw_create_l2_space(dimensions)
    }

    public static func createIPSpace(dimensions: Int) -> HNSWSpaceHandle? {
        hnsw_create_ip_space(dimensions)
    }

    public static func addPoint(
        _ index: HNSWIndexHandle,
        data: UnsafePointer<Float>,
        label: UInt64
    ) -> Bool {
        hnsw_add_point(index, data, label)
    }

    public static func searchKnn(
        _ index: HNSWIndexHandle,
        query: UnsafePointer<Float>,
        k: Int32,
        labels: UnsafeMutablePointer<UInt64>,
        distances: UnsafeMutablePointer<Float>
    ) -> Int32 {
        hnsw_search_knn(index, query, k, labels, distances)
    }

    public static func addPointsBatch(
        _ index: HNSWIndexHandle,
        data: UnsafePointer<Float>,
        labels: UnsafePointer<UInt64>,
        numPoints: Int,
        dimension: Int
    ) -> Int32 {
        hnsw_add_points_batch(index, data, labels, numPoints, dimension)
    }

    public static func searchKnnBatch(
        _ index: HNSWIndexHandle,
        queries: UnsafePointer<Float>,
        numQueries: Int,
        dimension: Int,
        k: Int32,
        labels: UnsafeMutablePointer<UInt64>,
        distances: UnsafeMutablePointer<Float>
    ) -> Int32 {
        hnsw_search_knn_batch(index, queries, numQueries, dimension, k, labels, distances)
    }

    public static func getVector(
        _ index: HNSWIndexHandle,
        label: UInt64,
        output: UnsafeMutablePointer<Float>,
        dimension: Int
    ) -> Bool {
        hnsw_get_vector(index, label, output, dimension)
    }
}

// MARK: - Float16 Conformance

extension Float16: HNSWScalar {
    public static func createL2Space(dimensions: Int) -> HNSWSpaceHandle? {
        hnsw_create_l2_space_f16(dimensions)
    }

    public static func createIPSpace(dimensions: Int) -> HNSWSpaceHandle? {
        hnsw_create_ip_space_f16(dimensions)
    }

    public static func addPoint(
        _ index: HNSWIndexHandle,
        data: UnsafePointer<Float16>,
        label: UInt64
    ) -> Bool {
        data.withMemoryRebound(to: UInt16.self, capacity: 1) { ptr in
            hnsw_add_point_f16(index, ptr, label)
        }
    }

    public static func searchKnn(
        _ index: HNSWIndexHandle,
        query: UnsafePointer<Float16>,
        k: Int32,
        labels: UnsafeMutablePointer<UInt64>,
        distances: UnsafeMutablePointer<Float>
    ) -> Int32 {
        query.withMemoryRebound(to: UInt16.self, capacity: 1) { ptr in
            hnsw_search_knn_f16(index, ptr, k, labels, distances)
        }
    }

    public static func addPointsBatch(
        _ index: HNSWIndexHandle,
        data: UnsafePointer<Float16>,
        labels: UnsafePointer<UInt64>,
        numPoints: Int,
        dimension: Int
    ) -> Int32 {
        data.withMemoryRebound(to: UInt16.self, capacity: numPoints * dimension) { ptr in
            hnsw_add_points_batch_f16(index, ptr, labels, numPoints, dimension)
        }
    }

    public static func searchKnnBatch(
        _ index: HNSWIndexHandle,
        queries: UnsafePointer<Float16>,
        numQueries: Int,
        dimension: Int,
        k: Int32,
        labels: UnsafeMutablePointer<UInt64>,
        distances: UnsafeMutablePointer<Float>
    ) -> Int32 {
        queries.withMemoryRebound(to: UInt16.self, capacity: numQueries * dimension) { ptr in
            hnsw_search_knn_batch_f16(index, ptr, numQueries, dimension, k, labels, distances)
        }
    }

    public static func getVector(
        _ index: HNSWIndexHandle,
        label: UInt64,
        output: UnsafeMutablePointer<Float16>,
        dimension: Int
    ) -> Bool {
        output.withMemoryRebound(to: UInt16.self, capacity: dimension) { ptr in
            hnsw_get_vector_f16(index, label, ptr, dimension)
        }
    }
}
