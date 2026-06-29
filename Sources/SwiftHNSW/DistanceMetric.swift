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
}
