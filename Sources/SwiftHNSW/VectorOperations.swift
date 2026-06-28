#if canImport(Accelerate)
import Accelerate
#endif

/// High-performance vector operations for native and Wasm builds.
enum VectorOperations {
    private typealias FloatSIMD = SIMD8<Float>
    private typealias Float16SIMD = SIMD8<Float16>

    // MARK: - Typed Operations

    /// Normalize a single vector to unit length.
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
        var magnitude = dotProduct(input, input)
        magnitude = magnitude.squareRoot()
        guard magnitude > 0 else {
            copy(input, into: output)
            return
        }
        scale(input, by: 1.0 / magnitude, into: output)
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

    static func distance(
        from lhs: UnsafeBufferPointer<Float>,
        to rhs: UnsafeBufferPointer<Float>,
        metric: DistanceMetric
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        return publicDistance(fromComparisonDistance: comparisonDistance(from: lhs, to: rhs, metric: metric), metric: metric)
    }

    static func comparisonDistance(
        from lhs: UnsafeBufferPointer<Float>,
        to rhs: UnsafeBufferPointer<Float>,
        metric: DistanceMetric
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        switch metric {
        case .l2:
            return squaredL2Distance(lhs, rhs)
        case .innerProduct, .cosine:
            return 1 - dotProduct(lhs, rhs)
        }
    }

    static func normalize(
        _ input: UnsafeBufferPointer<Float16>,
        into output: UnsafeMutableBufferPointer<Float16>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        guard input.count > 0 else { return }

        var magnitude = dotProduct(input, input)
        magnitude = magnitude.squareRoot()
        guard magnitude > 0 else {
            copy(input, into: output)
            return
        }
        scale(input, by: 1.0 / magnitude, into: output)
    }

    static func distance(
        from lhs: UnsafeBufferPointer<Float16>,
        to rhs: UnsafeBufferPointer<Float16>,
        metric: DistanceMetric
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        return publicDistance(fromComparisonDistance: comparisonDistance(from: lhs, to: rhs, metric: metric), metric: metric)
    }

    static func comparisonDistance(
        from lhs: UnsafeBufferPointer<Float16>,
        to rhs: UnsafeBufferPointer<Float16>,
        metric: DistanceMetric
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        switch metric {
        case .l2:
            return squaredL2Distance(lhs, rhs)
        case .innerProduct, .cosine:
            return 1 - dotProduct(lhs, rhs)
        }
    }

    @inline(__always)
    private static func loadFloatSIMD(
        _ buffer: UnsafeBufferPointer<Float>,
        at index: Int
    ) -> FloatSIMD {
        UnsafeRawPointer(buffer.baseAddress! + index)
            .loadUnaligned(as: FloatSIMD.self)
    }

    @inline(__always)
    private static func loadFloat16SIMD(
        _ buffer: UnsafeBufferPointer<Float16>,
        at index: Int
    ) -> Float16SIMD {
        UnsafeRawPointer(buffer.baseAddress! + index)
            .loadUnaligned(as: Float16SIMD.self)
    }

    @inline(__always)
    private static func reduceSum(_ vector: FloatSIMD) -> Float {
        var sum: Float = 0
        for lane in 0..<FloatSIMD.scalarCount {
            sum += vector[lane]
        }
        return sum
    }

    @inline(__always)
    private static func dotProduct(
        _ lhs: UnsafeBufferPointer<Float>,
        _ rhs: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")

        var index = 0
        var accumulator = FloatSIMD(repeating: 0)
        let simdEnd = lhs.count - (lhs.count % FloatSIMD.scalarCount)

        while index < simdEnd {
            accumulator += loadFloatSIMD(lhs, at: index) * loadFloatSIMD(rhs, at: index)
            index += FloatSIMD.scalarCount
        }

        var sum = reduceSum(accumulator)
        while index < lhs.count {
            sum += lhs[index] * rhs[index]
            index += 1
        }
        return sum
    }

    @inline(__always)
    private static func dotProduct(
        _ lhs: UnsafeBufferPointer<Float16>,
        _ rhs: UnsafeBufferPointer<Float16>
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")

        var index = 0
        var accumulator = FloatSIMD(repeating: 0)
        let simdEnd = lhs.count - (lhs.count % Float16SIMD.scalarCount)

        while index < simdEnd {
            let lhsVector = FloatSIMD(loadFloat16SIMD(lhs, at: index))
            let rhsVector = FloatSIMD(loadFloat16SIMD(rhs, at: index))
            accumulator += lhsVector * rhsVector
            index += Float16SIMD.scalarCount
        }

        var sum = reduceSum(accumulator)
        while index < lhs.count {
            sum += Float(lhs[index]) * Float(rhs[index])
            index += 1
        }
        return sum
    }

    @inline(__always)
    private static func squaredL2Distance(
        _ lhs: UnsafeBufferPointer<Float>,
        _ rhs: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")

        var index = 0
        var accumulator = FloatSIMD(repeating: 0)
        let simdEnd = lhs.count - (lhs.count % FloatSIMD.scalarCount)

        while index < simdEnd {
            let diff = loadFloatSIMD(lhs, at: index) - loadFloatSIMD(rhs, at: index)
            accumulator += diff * diff
            index += FloatSIMD.scalarCount
        }

        var sum = reduceSum(accumulator)
        while index < lhs.count {
            let diff = lhs[index] - rhs[index]
            sum += diff * diff
            index += 1
        }
        return sum
    }

    @inline(__always)
    private static func squaredL2Distance(
        _ lhs: UnsafeBufferPointer<Float16>,
        _ rhs: UnsafeBufferPointer<Float16>
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")

        var index = 0
        var accumulator = FloatSIMD(repeating: 0)
        let simdEnd = lhs.count - (lhs.count % Float16SIMD.scalarCount)

        while index < simdEnd {
            let lhsVector = FloatSIMD(loadFloat16SIMD(lhs, at: index))
            let rhsVector = FloatSIMD(loadFloat16SIMD(rhs, at: index))
            let diff = lhsVector - rhsVector
            accumulator += diff * diff
            index += Float16SIMD.scalarCount
        }

        var sum = reduceSum(accumulator)
        while index < lhs.count {
            let diff = Float(lhs[index]) - Float(rhs[index])
            sum += diff * diff
            index += 1
        }
        return sum
    }

    @inline(__always)
    private static func scale(
        _ input: UnsafeBufferPointer<Float>,
        by scale: Float,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")

        var index = 0
        let scaleVector = FloatSIMD(repeating: scale)
        let simdEnd = input.count - (input.count % FloatSIMD.scalarCount)

        while index < simdEnd {
            let scaled = loadFloatSIMD(input, at: index) * scaleVector
            UnsafeMutableRawPointer(output.baseAddress! + index)
                .storeBytes(of: scaled, as: FloatSIMD.self)
            index += FloatSIMD.scalarCount
        }

        while index < input.count {
            output[index] = input[index] * scale
            index += 1
        }
    }

    @inline(__always)
    private static func scale(
        _ input: UnsafeBufferPointer<Float16>,
        by scale: Float,
        into output: UnsafeMutableBufferPointer<Float16>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")

        var index = 0
        let scaleVector = FloatSIMD(repeating: scale)
        let simdEnd = input.count - (input.count % Float16SIMD.scalarCount)

        while index < simdEnd {
            let scaled = Float16SIMD(FloatSIMD(loadFloat16SIMD(input, at: index)) * scaleVector)
            UnsafeMutableRawPointer(output.baseAddress! + index)
                .storeBytes(of: scaled, as: Float16SIMD.self)
            index += Float16SIMD.scalarCount
        }

        while index < input.count {
            output[index] = Float16(Float(input[index]) * scale)
            index += 1
        }
    }

    static func normalize<Scalar: HNSWScalar>(
        _ input: UnsafeBufferPointer<Scalar>,
        into output: UnsafeMutableBufferPointer<Scalar>
    ) {
        precondition(input.count == output.count, "Input and output dimensions must match")
        guard input.count > 0 else { return }

        if Scalar.self == Float.self {
            let inputFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float.self),
                count: input.count
            )
            let outputFloats = UnsafeMutableBufferPointer<Float>(
                start: UnsafeMutableRawPointer(output.baseAddress!).assumingMemoryBound(to: Float.self),
                count: output.count
            )
            normalize(inputFloats, into: outputFloats)
            return
        }

        if Scalar.self == Float16.self {
            let inputHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(input.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: input.count
            )
            let outputHalves = UnsafeMutableBufferPointer<Float16>(
                start: UnsafeMutableRawPointer(output.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: output.count
            )
            normalize(inputHalves, into: outputHalves)
            return
        }

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
        publicDistance(fromComparisonDistance: comparisonDistance(from: lhs, to: rhs, metric: metric), metric: metric)
    }

    static func comparisonDistance<Scalar: HNSWScalar>(
        from lhs: UnsafeBufferPointer<Scalar>,
        to rhs: UnsafeBufferPointer<Scalar>,
        metric: DistanceMetric
    ) -> Float {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")

        if Scalar.self == Float.self {
            guard lhs.count > 0 else {
                switch metric {
                case .l2:
                    return 0
                case .innerProduct, .cosine:
                    return 1
                }
            }
            let lhsFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(lhs.baseAddress!).assumingMemoryBound(to: Float.self),
                count: lhs.count
            )
            let rhsFloats = UnsafeBufferPointer<Float>(
                start: UnsafeRawPointer(rhs.baseAddress!).assumingMemoryBound(to: Float.self),
                count: rhs.count
            )
            return comparisonDistance(from: lhsFloats, to: rhsFloats, metric: metric)
        }

        if Scalar.self == Float16.self {
            guard lhs.count > 0 else {
                switch metric {
                case .l2:
                    return 0
                case .innerProduct, .cosine:
                    return 1
                }
            }
            let lhsHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(lhs.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: lhs.count
            )
            let rhsHalves = UnsafeBufferPointer<Float16>(
                start: UnsafeRawPointer(rhs.baseAddress!).assumingMemoryBound(to: Float16.self),
                count: rhs.count
            )
            return comparisonDistance(from: lhsHalves, to: rhsHalves, metric: metric)
        }

        switch metric {
        case .l2:
            var sum: Float = 0
            for index in 0..<lhs.count {
                let diff = lhs[index].hnswFloatValue - rhs[index].hnswFloatValue
                sum += diff * diff
            }
            return sum
        case .innerProduct, .cosine:
            var dotProduct: Float = 0
            for index in 0..<lhs.count {
                dotProduct += lhs[index].hnswFloatValue * rhs[index].hnswFloatValue
            }
            return 1 - dotProduct
        }
    }

    static func publicDistance(fromComparisonDistance distance: Float, metric: DistanceMetric) -> Float {
        switch metric {
        case .l2:
            return max(0, distance).squareRoot()
        case .innerProduct, .cosine:
            return distance
        }
    }

    // MARK: - Float16 Array Convenience

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
