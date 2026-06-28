#if canImport(Accelerate)
import Accelerate
#endif

/// High-performance vector operations using Accelerate framework
enum VectorOperations {

    // MARK: - Float32 Operations

    /// Normalize a single vector to unit length using SIMD
    static func normalize(_ vector: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: vector.count)
        vector.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                normalize(input, into: output)
            }
        }
        return result
    }

    static func normalize(
        _ input: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        guard input.count > 0 else { return }

#if canImport(Accelerate)
        var magnitude: Float = 0
        vDSP_dotpr(input.baseAddress!, 1, input.baseAddress!, 1, &magnitude, vDSP_Length(input.count))
        magnitude = sqrt(magnitude)
        guard magnitude > 0 else {
            copy(input, into: output)
            return
        }
        var scale = 1.0 / magnitude
        vDSP_vsmul(input.baseAddress!, 1, &scale, output.baseAddress!, 1, vDSP_Length(input.count))
#else
        var magnitude: Float = 0
        for value in input {
            magnitude += value * value
        }
        magnitude = magnitude.squareRoot()
        guard magnitude > 0 else {
            copy(input, into: output)
            return
        }
        for index in 0..<input.count {
            output[index] = input[index] / magnitude
        }
#endif
    }

    /// Normalize a batch of vectors to unit length
    static func normalizeBatch(_ vectors: [Float], count: Int, dimensions: Int) -> [Float] {
        var result = [Float](repeating: 0, count: vectors.count)
        vectors.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                normalizeBatch(input, count: count, dimensions: dimensions, into: output)
            }
        }
        return result
    }

    static func normalizeBatch(
        _ input: UnsafeBufferPointer<Float>,
        count: Int,
        dimensions: Int,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == count * dimensions, "Input size must match count * dimensions")
        precondition(output.count == input.count, "Input and output dimensions must match")
        guard count > 0 else { return }

#if canImport(Accelerate)
        for i in 0..<count {
            let offset = i * dimensions
            var magnitude: Float = 0
            vDSP_dotpr(
                input.baseAddress! + offset, 1,
                input.baseAddress! + offset, 1,
                &magnitude, vDSP_Length(dimensions)
            )
            magnitude = sqrt(magnitude)
            guard magnitude > 0 else {
                for index in 0..<dimensions {
                    output[offset + index] = input[offset + index]
                }
                continue
            }
            var scale = 1.0 / magnitude
            vDSP_vsmul(
                input.baseAddress! + offset, 1,
                &scale,
                output.baseAddress! + offset, 1,
                vDSP_Length(dimensions)
            )
        }
#else
        for i in 0..<count {
            let offset = i * dimensions
            let inputSlice = UnsafeBufferPointer(start: input.baseAddress! + offset, count: dimensions)
            let outputSlice = UnsafeMutableBufferPointer(start: output.baseAddress! + offset, count: dimensions)
            normalize(inputSlice, into: outputSlice)
        }
#endif
    }

    static func copy<Scalar>(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        for index in 0..<input.count {
            output[index] = input[index]
        }
    }

    static func normalize<Scalar: HNSWScalar>(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        guard input.count > 0 else { return }

        var magnitude: Float = 0
        for value in input {
            let floatValue = value.hnswFloatValue
            magnitude += floatValue * floatValue
        }
        magnitude = magnitude.squareRoot()
        guard magnitude > 0 else {
            copy(input, into: output)
            return
        }
        for index in 0..<input.count {
            output[index] = Scalar(input[index].hnswFloatValue / magnitude)
        }
    }

    static func normalizeBatch<Scalar: HNSWScalar>(
        _ input: UnsafeBufferPointer<Scalar>,
        count: Int,
        dimensions: Int,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        precondition(input.count == count * dimensions, "Input size must match count * dimensions")
        precondition(output.count == input.count, "Input and output dimensions must match")
        guard count > 0 else { return }

        for index in 0..<count {
            let offset = index * dimensions
            let inputSlice = UnsafeBufferPointer(start: input.baseAddress! + offset, count: dimensions)
            let outputSlice = UnsafeMutableBufferPointer(start: output.baseAddress! + offset, count: dimensions)
            normalize(inputSlice, into: outputSlice)
        }
    }

    static func distance<Scalar: HNSWScalar>(
        from lhs: UnsafeBufferPointer<Scalar>,
        to rhs: UnsafeBufferPointer<Scalar>,
        metric: DistanceMetric
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        switch metric {
        case .l2:
            var sum: Float = 0
            for index in 0..<lhs.count {
                let diff = lhs[index].hnswFloatValue - rhs[index].hnswFloatValue
                sum += diff * diff
            }
            return sum.squareRoot()
        case .innerProduct, .cosine:
            var dotProduct: Float = 0
            for index in 0..<lhs.count {
                dotProduct += lhs[index].hnswFloatValue * rhs[index].hnswFloatValue
            }
            return 1 - dotProduct
        }
    }

    // MARK: - Float16 Operations

    /// Normalize a single Float16 vector to unit length
    static func normalize(_ vector: [Float16]) -> [Float16] {
        var result = [Float16](repeating: 0, count: vector.count)
        vector.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                normalize(input, into: output)
            }
        }
        return result
    }

    /// Normalize a batch of Float16 vectors to unit length
    static func normalizeBatch(_ vectors: [Float16], count: Int, dimensions: Int) -> [Float16] {
        var result = [Float16](repeating: 0, count: vectors.count)
        vectors.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                normalizeBatch(input, count: count, dimensions: dimensions, into: output)
            }
        }
        return result
    }
}
