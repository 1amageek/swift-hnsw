import Foundation

/// Errors that can occur when using HNSW index
public enum HNSWError: Error, Sendable {
    case initializationFailed(String)
    case dimensionMismatch(expected: Int, got: Int)
    case indexNotInitialized
    case addPointFailed(String)
    case deleteFailed(String)
    case saveFailed(String)
    case loadFailed(String)
    case capacityExceeded(current: Int, maximum: Int)
}

extension HNSWError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .initializationFailed(let message):
            return "Initialization failed: \(message)"
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected), got \(got)"
        case .indexNotInitialized:
            return "Index not initialized"
        case .addPointFailed(let message):
            return "Add point failed: \(message)"
        case .deleteFailed(let message):
            return "Delete failed: \(message)"
        case .saveFailed(let message):
            return "Save failed: \(message)"
        case .loadFailed(let message):
            return "Load failed: \(message)"
        case .capacityExceeded(let current, let maximum):
            return "Capacity exceeded: \(current) elements, maximum \(maximum)"
        }
    }
}

extension HNSWError: Equatable {
    public static func == (lhs: HNSWError, rhs: HNSWError) -> Bool {
        switch (lhs, rhs) {
        case (.initializationFailed(let a), .initializationFailed(let b)):
            return a == b
        case (.dimensionMismatch(let e1, let g1), .dimensionMismatch(let e2, let g2)):
            return e1 == e2 && g1 == g2
        case (.indexNotInitialized, .indexNotInitialized):
            return true
        case (.addPointFailed(let a), .addPointFailed(let b)):
            return a == b
        case (.deleteFailed(let a), .deleteFailed(let b)):
            return a == b
        case (.saveFailed(let a), .saveFailed(let b)):
            return a == b
        case (.loadFailed(let a), .loadFailed(let b)):
            return a == b
        case (.capacityExceeded(let c1, let m1), .capacityExceeded(let c2, let m2)):
            return c1 == c2 && m1 == m2
        default:
            return false
        }
    }
}
