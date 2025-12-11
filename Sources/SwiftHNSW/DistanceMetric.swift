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

    /// Create the appropriate space handle for this metric
    internal func createSpace(dimensions: Int) -> HNSWSpaceHandle? {
        switch self {
        case .l2:
            return hnsw_create_l2_space(dimensions)
        case .innerProduct, .cosine:
            return hnsw_create_ip_space(dimensions)
        }
    }
}
