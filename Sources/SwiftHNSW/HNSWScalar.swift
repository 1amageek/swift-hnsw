import Foundation

/// Protocol for scalar types supported by HNSW index
public protocol HNSWScalar: BinaryFloatingPoint, Sendable {
    var hnswFloatValue: Float { get }
}

extension Float: HNSWScalar {
    public var hnswFloatValue: Float { self }
}

extension Float16: HNSWScalar {
    public var hnswFloatValue: Float { Float(self) }
}
