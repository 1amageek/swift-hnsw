/// Search result containing label and distance
public struct SearchResult: Sendable, Hashable {
    /// Unique identifier of the vector
    public let label: UInt64
    /// Distance from query vector
    public let distance: Float

    public init(label: UInt64, distance: Float) {
        self.label = label
        self.distance = distance
    }
}

extension SearchResult: Comparable {
    public static func < (lhs: SearchResult, rhs: SearchResult) -> Bool {
        lhs.distance < rhs.distance
    }
}

extension SearchResult: CustomStringConvertible {
    public var description: String {
        "SearchResult(label: \(label), distance: \(String(format: "%.4f", distance)))"
    }
}
