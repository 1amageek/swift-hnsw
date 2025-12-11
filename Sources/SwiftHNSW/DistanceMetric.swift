import hnswlib

/// Distance metric types for HNSW index
public enum DistanceMetric: String, Sendable, CaseIterable {
    /// L2 (Euclidean) distance - d = sqrt(sum((a-b)^2))
    case l2
    /// Inner product distance - d = 1 - sum(a*b)
    case innerProduct
    /// Cosine similarity - d = 1 - cos(a, b)
    case cosine

    /// Whether this metric requires vector normalization
    var requiresNormalization: Bool {
        self == .cosine
    }

    /// Create the appropriate space handle for this metric (Float32)
    internal func createSpace(dimensions: Int) -> HNSWSpaceHandle? {
        createSpace(dimensions: dimensions, scalar: Float.self)
    }

    /// Create the appropriate space handle for this metric with generic scalar type
    internal func createSpace<S: HNSWScalar>(dimensions: Int, scalar: S.Type) -> HNSWSpaceHandle? {
        switch self {
        case .l2:
            return S.createL2Space(dimensions: dimensions)
        case .innerProduct, .cosine:
            return S.createIPSpace(dimensions: dimensions)
        }
    }
}
