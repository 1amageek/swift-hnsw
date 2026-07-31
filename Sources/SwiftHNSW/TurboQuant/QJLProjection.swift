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
    /// Each packed sign byte is evaluated as two independent four-bit nibbles.
    /// Keeping 32 Float32 values per byte keeps the query table in cache while
    /// preserving Float32 signed-sum precision.
    static let lookupEntriesPerByte = 32

    let dimension: Int
    let packedSize: Int
    let retainedBytes: Int
    private let matrix: [Float]

    var lookupTableCount: Int {
        packedSize * Self.lookupEntriesPerByte
    }

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

    /// Builds two 16-entry signed-sum rows for every packed sign byte.
    @discardableResult
    func fillInnerProductTable(
        for projectedQuery: UnsafeBufferPointer<Float>,
        into table: UnsafeMutableBufferPointer<Float>
    ) -> Float {
        precondition(projectedQuery.count == dimension, "Projected query dimension must match")
        precondition(table.count == lookupTableCount, "QJL lookup table dimension must match")

        // The pruning bound must dominate the Float32 reduction used by `innerProduct`.
        // The Float32 operation margin also dominates Double accumulation error for the
        // bounded dimension and finite query values accepted by the index.
        var absoluteSum: Double = 0
        for value in projectedQuery {
            absoluteSum += Double(abs(value))
        }
        for byteIndex in 0..<packedSize {
            let firstCoordinate = byteIndex * 8
            let row = table.baseAddress! + byteIndex * Self.lookupEntriesPerByte
            for nibble in 0..<16 {
                var sum: Float = 0
                for lane in 0..<4 {
                    let coordinate = firstCoordinate + lane
                    guard coordinate < dimension else { break }
                    let value = projectedQuery[coordinate]
                    let sign = (nibble >> (3 - lane)) & 1
                    sum += sign == 0 ? -value : value
                }
                row[nibble] = sum
            }
            for nibble in 0..<16 {
                var sum: Float = 0
                for lane in 0..<4 {
                    let coordinate = firstCoordinate + 4 + lane
                    guard coordinate < dimension else { break }
                    let value = projectedQuery[coordinate]
                    let sign = (nibble >> (3 - lane)) & 1
                    sum += sign == 0 ? -value : value
                }
                row[16 + nibble] = sum
            }
        }
        let operationCount = Double(packedSize) * 8 + 8
        let unitRoundoff = Double(Float.ulpOfOne)
        let errorRatio = operationCount * unitRoundoff
        guard errorRatio < 1 else { return .infinity }
        let conservativeSum = absoluteSum / (1 - errorRatio)
        let roundedSum = Float(conservativeSum)
        guard roundedSum.isFinite else { return .infinity }
        return Double(roundedSum) < conservativeSum ? roundedSum.nextUp : roundedSum
    }

    /// Computes `dot(projectedQuery, signs)` directly from packed sign bytes.
    @inline(__always)
    func innerProduct(
        signs: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(signs.count >= packedSize, "Packed QJL signs are truncated")
        precondition(table.count == lookupTableCount, "QJL lookup table dimension must match")
        return Self.innerProduct(signs: signs, using: table)
    }

    @inline(__always)
    static func innerProduct(
        signs: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(
            signs.count <= Int.max / lookupEntriesPerByte,
            "QJL sign count overflows the lookup-table size"
        )
        precondition(
            table.count == signs.count * lookupEntriesPerByte,
            "QJL lookup table dimension must match"
        )
        guard !signs.isEmpty else { return 0 }
        // The owners retain both initialized contiguous buffers for this synchronous
        // borrow. Count validation bounds every offset, Float storage is naturally
        // aligned, and these pointers never escape or cross a concurrency boundary.
        let signsBase = signs.baseAddress!
        let tableBase = table.baseAddress!
        var byteIndex = 0
        var accumulator0: Float = 0
        var accumulator1: Float = 0
        var accumulator2: Float = 0
        var accumulator3: Float = 0
        while byteIndex + 8 <= signs.count {
            let row = tableBase.advanced(by: byteIndex * lookupEntriesPerByte)
            // The loop bound proves eight initialized sign bytes are available. The
            // unaligned load avoids imposing an alignment requirement on `[UInt8]`.
            let packedBytes = UInt64(littleEndian: UnsafeRawPointer(
                signsBase.advanced(by: byteIndex)
            ).loadUnaligned(as: UInt64.self))
            let byte0 = UInt8(truncatingIfNeeded: packedBytes)
            let byte1 = UInt8(truncatingIfNeeded: packedBytes >> 8)
            let byte2 = UInt8(truncatingIfNeeded: packedBytes >> 16)
            let byte3 = UInt8(truncatingIfNeeded: packedBytes >> 24)
            let byte4 = UInt8(truncatingIfNeeded: packedBytes >> 32)
            let byte5 = UInt8(truncatingIfNeeded: packedBytes >> 40)
            let byte6 = UInt8(truncatingIfNeeded: packedBytes >> 48)
            let byte7 = UInt8(truncatingIfNeeded: packedBytes >> 56)
            accumulator0 += row[Int(byte0 >> 4)] + row[16 + Int(byte0 & 0x0F)]
            accumulator1 += row[32 + Int(byte1 >> 4)] + row[48 + Int(byte1 & 0x0F)]
            accumulator2 += row[64 + Int(byte2 >> 4)] + row[80 + Int(byte2 & 0x0F)]
            accumulator3 += row[96 + Int(byte3 >> 4)] + row[112 + Int(byte3 & 0x0F)]
            let nextRow = row.advanced(by: 4 * lookupEntriesPerByte)
            accumulator0 += nextRow[Int(byte4 >> 4)] + nextRow[16 + Int(byte4 & 0x0F)]
            accumulator1 += nextRow[32 + Int(byte5 >> 4)] + nextRow[48 + Int(byte5 & 0x0F)]
            accumulator2 += nextRow[64 + Int(byte6 >> 4)] + nextRow[80 + Int(byte6 & 0x0F)]
            accumulator3 += nextRow[96 + Int(byte7 >> 4)] + nextRow[112 + Int(byte7 & 0x0F)]
            byteIndex += 8
        }
        while byteIndex + 4 <= signs.count {
            let row = tableBase.advanced(by: byteIndex * lookupEntriesPerByte)
            let packedBytes = UInt32(littleEndian: UnsafeRawPointer(
                signsBase.advanced(by: byteIndex)
            ).loadUnaligned(as: UInt32.self))
            let byte0 = UInt8(truncatingIfNeeded: packedBytes)
            let byte1 = UInt8(truncatingIfNeeded: packedBytes >> 8)
            let byte2 = UInt8(truncatingIfNeeded: packedBytes >> 16)
            let byte3 = UInt8(truncatingIfNeeded: packedBytes >> 24)
            accumulator0 += row[Int(byte0 >> 4)] + row[16 + Int(byte0 & 0x0F)]
            accumulator1 += row[32 + Int(byte1 >> 4)] + row[48 + Int(byte1 & 0x0F)]
            accumulator2 += row[64 + Int(byte2 >> 4)] + row[80 + Int(byte2 & 0x0F)]
            accumulator3 += row[96 + Int(byte3 >> 4)] + row[112 + Int(byte3 & 0x0F)]
            byteIndex += 4
        }
        var total = accumulator0 + accumulator1 + accumulator2 + accumulator3
        while byteIndex < signs.count {
            let row = tableBase.advanced(by: byteIndex * lookupEntriesPerByte)
            let byte = signsBase[byteIndex]
            total += row[Int(byte >> 4)] + row[16 + Int(byte & 0x0F)]
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
