import Testing
@testable import SwiftHNSW

@Suite("Vector Operations")
struct VectorOperationsTests {
    @Test("Float SIMD distance handles full lanes and tail")
    func floatSIMDDistanceHandlesFullLanesAndTail() {
        let lhs: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let rhs: [Float] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        let l2 = lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                VectorOperations.distance(from: lhsBuffer, to: rhsBuffer, metric: .l2)
            }
        }

        let innerProduct = lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                VectorOperations.distance(from: lhsBuffer, to: rhsBuffer, metric: .innerProduct)
            }
        }

        #expect(abs(l2 - Float(330).squareRoot()) < 0.0001)
        #expect(innerProduct == -219)
    }

    @Test("Float SIMD normalization keeps unit length across tail")
    func floatSIMDNormalizeHandlesTail() {
        let vector: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        var normalized = [Float](repeating: 0, count: vector.count)

        vector.withUnsafeBufferPointer { input in
            normalized.withUnsafeMutableBufferPointer { output in
                VectorOperations.normalize(input, into: output)
            }
        }

        let squaredMagnitude = normalized.reduce(Float(0)) { partial, value in
            partial + value * value
        }

        #expect(abs(squaredMagnitude - 1) < 0.0001)
    }

    @Test("Float16 SIMD distance handles full lanes and tail")
    func float16SIMDDistanceHandlesFullLanesAndTail() {
        let lhs: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let rhs: [Float16] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        let l2 = lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                VectorOperations.distance(from: lhsBuffer, to: rhsBuffer, metric: .l2)
            }
        }

        let innerProduct = lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                VectorOperations.distance(from: lhsBuffer, to: rhsBuffer, metric: .innerProduct)
            }
        }

        #expect(abs(l2 - Float(330).squareRoot()) < 0.0001)
        #expect(innerProduct == -219)
    }

    @Test("Float16 distance accumulates in Float precision")
    func float16DistanceAccumulatesInFloatPrecision() {
        let lhs = [Float16](repeating: Float16(0.1), count: 1024)
        let rhs = lhs

        let innerProductDistance = lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                VectorOperations.distance(from: lhsBuffer, to: rhsBuffer, metric: .innerProduct)
            }
        }
        let expectedDot = lhs.reduce(Float(0)) { partial, value in
            partial + Float(value) * Float(value)
        }

        #expect(abs(innerProductDistance - (1 - expectedDot)) < 0.0001)
    }

    @Test("Float16 SIMD normalization keeps unit length across tail")
    func float16SIMDNormalizeHandlesTail() {
        let vector: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        var normalized = [Float16](repeating: 0, count: vector.count)

        vector.withUnsafeBufferPointer { input in
            normalized.withUnsafeMutableBufferPointer { output in
                VectorOperations.normalize(input, into: output)
            }
        }

        let squaredMagnitude = normalized.reduce(Float(0)) { partial, value in
            let floatValue = Float(value)
            return partial + floatValue * floatValue
        }

        #expect(abs(squaredMagnitude - 1) < 0.001)
    }
}
