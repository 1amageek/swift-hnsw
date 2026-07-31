#if canImport(Accelerate)
import Accelerate

@_silgen_name("cblas_sgemv$NEWLAPACK")
private func hnswAccelerateMatrixVectorMultiply(
    _ order: CBLAS_ORDER,
    _ transpose: CBLAS_TRANSPOSE,
    _ rowCount: Int32,
    _ columnCount: Int32,
    _ alpha: Float,
    _ matrix: UnsafePointer<Float>?,
    _ leadingDimension: Int32,
    _ input: UnsafePointer<Float>?,
    _ inputStride: Int32,
    _ beta: Float,
    _ output: UnsafeMutablePointer<Float>?,
    _ outputStride: Int32
)
#endif

/// Deterministic Gaussian projection used by Quantized Johnson-Lindenstrauss.
///
/// The row-major matrix contains independent pseudorandom standard-normal samples. It is
/// immutable after initialization, so concurrent indexes may read it without synchronization.
struct QJLProjection: Sendable {
    private typealias FloatSIMD = SIMD8<Float>

    let dimension: Int
    let packedSize: Int
    let retainedBytes: Int
    private let matrix: [Float]

    init(dimension: Int, seed: UInt64) throws {
        precondition(dimension > 0, "dimension must be positive")
        guard dimension <= Int.max / dimension else {
            throw HNSWError.initializationFailed("QJL projection dimensions overflow Int")
        }
        let elementCount = dimension * dimension
        let maximumElementCount = 64 * 1_024 * 1_024
        guard elementCount <= maximumElementCount else {
            throw HNSWError.initializationFailed(
                "QJL projection exceeds the 256 MiB retained-matrix limit"
            )
        }

        self.dimension = dimension
        self.packedSize = BitPacking.packedSize(count: dimension, bitWidth: 1)
        self.retainedBytes = elementCount * MemoryLayout<Float>.size

        var values = [Float]()
        values.reserveCapacity(elementCount)
        var generator = TurboQuantSplitMix64(seed: seed)
        while values.count < elementCount {
            let pair = generator.nextGaussianPair()
            values.append(Float(pair.0))
            if values.count < elementCount {
                values.append(Float(pair.1))
            }
        }
        self.matrix = values
    }

    /// Projects a borrowed vector into a caller-owned buffer.
    func project(
        _ input: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == dimension, "Input dimension must match")
        precondition(output.count == dimension, "Output dimension must match")

        matrix.withUnsafeBufferPointer { matrixBuffer in
#if canImport(Accelerate)
            if dimension >= 128 {
                // The matrix owner remains retained by this synchronous borrow. All
                // dimensions and strides describe initialized contiguous Float storage,
                // and no pointer escapes the BLAS call.
                hnswAccelerateMatrixVectorMultiply(
                    CblasRowMajor,
                    CblasNoTrans,
                    Int32(dimension),
                    Int32(dimension),
                    1,
                    matrixBuffer.baseAddress!,
                    Int32(dimension),
                    input.baseAddress!,
                    1,
                    0,
                    output.baseAddress!,
                    1
                )
                return
            }
#endif
            for row in 0..<dimension {
                output[row] = Self.dot(
                    matrixBuffer.baseAddress! + row * dimension,
                    input.baseAddress!,
                    count: dimension
                )
            }
        }
    }

    /// Packs `sign(projected)` with one positive-sign bit per coordinate.
    func packSigns(
        _ projected: UnsafeBufferPointer<Float>,
        into packed: UnsafeMutableBufferPointer<UInt8>
    ) {
        precondition(projected.count == dimension, "Projected dimension must match")
        precondition(packed.count == packedSize, "Packed sign dimension must match")
        for index in packed.indices {
            packed[index] = 0
        }
        for coordinate in 0..<dimension where projected[coordinate] >= 0 {
            BitPacking.store(1, at: coordinate, in: packed, bitWidth: 1)
        }
    }

    /// Builds one 256-entry signed-sum row for every packed sign byte.
    func fillInnerProductTable(
        for projectedQuery: UnsafeBufferPointer<Float>,
        into table: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(projectedQuery.count == dimension, "Projected query dimension must match")
        precondition(table.count == packedSize * 256, "QJL lookup table dimension must match")

        for byteIndex in 0..<packedSize {
            let firstCoordinate = byteIndex * 8
            let row = table.baseAddress! + byteIndex * 256
            var negativeSum: Float = 0
            for lane in 0..<8 {
                let coordinate = firstCoordinate + lane
                guard coordinate < dimension else { break }
                negativeSum -= projectedQuery[coordinate]
            }
            row[0] = negativeSum
            for packedValue in 1..<256 {
                let value = UInt8(packedValue)
                let changedBit = value & (~value &+ 1)
                let lane = 7 - changedBit.trailingZeroBitCount
                let coordinate = firstCoordinate + lane
                let delta: Float = coordinate < dimension
                    ? 2 * projectedQuery[coordinate]
                    : 0
                row[packedValue] = row[packedValue ^ Int(changedBit)] + delta
            }
        }
    }

    /// Computes `dot(projectedQuery, signs)` directly from packed sign bytes.
    @inline(__always)
    func innerProduct(
        signs: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(signs.count >= packedSize, "Packed QJL signs are truncated")
        precondition(table.count == packedSize * 256, "QJL lookup table dimension must match")
        return Self.innerProduct(signs: signs, using: table)
    }

    @inline(__always)
    static func innerProduct(
        signs: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(table.count == signs.count * 256, "QJL lookup table dimension must match")
        var byteIndex = 0
        var accumulator0: Float = 0
        var accumulator1: Float = 0
        var accumulator2: Float = 0
        var accumulator3: Float = 0
        while byteIndex + 4 <= signs.count {
            let row = table.baseAddress! + byteIndex * 256
            accumulator0 += row[Int(signs[byteIndex])]
            accumulator1 += row[256 + Int(signs[byteIndex + 1])]
            accumulator2 += row[512 + Int(signs[byteIndex + 2])]
            accumulator3 += row[768 + Int(signs[byteIndex + 3])]
            byteIndex += 4
        }
        var total = accumulator0 + accumulator1 + accumulator2 + accumulator3
        while byteIndex < signs.count {
            total += table.baseAddress![
                byteIndex * 256 + Int(signs[byteIndex])
            ]
            byteIndex += 1
        }
        return total
    }

    @inline(__always)
    private static func dot(
        _ lhs: UnsafePointer<Float>,
        _ rhs: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        // The Array or workspace owners retain both initialized Float ranges for this
        // synchronous call. `count` comes from the validated projection dimension, each
        // SIMD load remains inside those ranges, and `loadUnaligned` removes any stronger
        // alignment assumption. Both borrows are read-only, never escape, and immutable
        // projection storage may therefore cross Sendable boundaries without synchronization.
        var index = 0
        var accumulator0 = FloatSIMD(repeating: 0)
        var accumulator1 = FloatSIMD(repeating: 0)
        let unrolledWidth = FloatSIMD.scalarCount * 2
        let unrolledEnd = count - count % unrolledWidth

        while index < unrolledEnd {
            let lhs0 = UnsafeRawPointer(lhs + index).loadUnaligned(as: FloatSIMD.self)
            let rhs0 = UnsafeRawPointer(rhs + index).loadUnaligned(as: FloatSIMD.self)
            let lhs1 = UnsafeRawPointer(lhs + index + FloatSIMD.scalarCount)
                .loadUnaligned(as: FloatSIMD.self)
            let rhs1 = UnsafeRawPointer(rhs + index + FloatSIMD.scalarCount)
                .loadUnaligned(as: FloatSIMD.self)
            accumulator0 += lhs0 * rhs0
            accumulator1 += lhs1 * rhs1
            index += unrolledWidth
        }

        var accumulator = accumulator0 + accumulator1
        let simdEnd = count - count % FloatSIMD.scalarCount
        while index < simdEnd {
            let lhsValue = UnsafeRawPointer(lhs + index).loadUnaligned(as: FloatSIMD.self)
            let rhsValue = UnsafeRawPointer(rhs + index).loadUnaligned(as: FloatSIMD.self)
            accumulator += lhsValue * rhsValue
            index += FloatSIMD.scalarCount
        }

        var total: Float = 0
        for lane in 0..<FloatSIMD.scalarCount {
            total += accumulator[lane]
        }
        while index < count {
            total += lhs[index] * rhs[index]
            index += 1
        }
        return total
    }
}
