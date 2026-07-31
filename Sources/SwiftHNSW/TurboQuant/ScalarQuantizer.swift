import CTurboQuantKernels

/// Scalar quantizer using Lloyd-Max centroids for a rotated sphere coordinate.
///
/// The structured rotation uses the Gaussian-limit centroids scaled by `1 / sqrt(d)`.
/// The exact Haar strategy numerically integrates the finite-dimensional coordinate density
/// proportional to `(1 - x^2)^((d - 3) / 2)` on `[-1, 1]`.
struct ScalarQuantizer: Sendable {

    static let hasAcceleratedFourBitKernel =
        swift_hnsw_turboquant_has_accelerated_four_bit_inner_product() != 0

    /// Quantization bit-width (0, 1, 2, 3, or 4)
    let bitWidth: Int

    /// Vector dimension
    let dimension: Int

    /// Number of centroids = 2^bitWidth
    let numCentroids: Int

    /// Centroids scaled by 1/sqrt(d), sorted ascending
    let centroids: [Float]

    /// Decision boundaries (midpoints between adjacent centroids)
    let boundaries: [Float]

    /// Bytes per packed quantized vector
    let packedSize: Int

    /// Four little-endian byte planes for the exact Float32 centroid table.
    private let fourBitCentroidBytePlanes: [UInt8]

    init(
        bitWidth: Int,
        dimension: Int,
        usesExactSphericalDistribution: Bool = false
    ) {
        precondition((0...4).contains(bitWidth), "bitWidth must be between 0 and 4")
        precondition(dimension > 0, "dimension must be positive")

        self.bitWidth = bitWidth
        self.dimension = dimension
        self.numCentroids = 1 << bitWidth

        if usesExactSphericalDistribution {
            self.centroids = Self.sphericalLloydMaxCentroids(
                bitWidth: bitWidth,
                dimension: dimension
            )
        } else {
            let scale = 1.0 / Float(dimension).squareRoot()
            let unscaled = Self.gaussianLloydMaxCentroids(bitWidth: bitWidth)
            self.centroids = unscaled.map { $0 * scale }
        }

        // Decision boundaries are midpoints between consecutive centroids
        var bounds = [Float](repeating: 0, count: numCentroids - 1)
        for i in 0..<bounds.count {
            bounds[i] = (self.centroids[i] + self.centroids[i + 1]) * 0.5
        }
        self.boundaries = bounds

        self.packedSize = BitPacking.packedSize(count: dimension, bitWidth: bitWidth)
        if bitWidth == 4 {
            var bytePlanes = [UInt8](repeating: 0, count: 64)
            for centroidIndex in 0..<16 {
                let bits = centroids[centroidIndex].bitPattern
                for byteIndex in 0..<4 {
                    bytePlanes[byteIndex * 16 + centroidIndex] = UInt8(
                        truncatingIfNeeded: bits >> UInt32(byteIndex * 8)
                    )
                }
            }
            self.fourBitCentroidBytePlanes = bytePlanes
        } else {
            self.fourBitCentroidBytePlanes = []
        }
    }

    /// Quantize and pack a rotated vector into compact bytes.
    func quantizeAndPack(_ rotatedVector: [Float]) -> [UInt8] {
        rotatedVector.withUnsafeBufferPointer { quantizeAndPack($0) }
    }

    /// Quantize and pack a borrowed rotated vector without materializing a second float array.
    func quantizeAndPack(_ rotatedVector: UnsafeBufferPointer<Float>) -> [UInt8] {
        precondition(rotatedVector.count == dimension, "Rotated vector dimension must match")
        var packed = [UInt8](repeating: 0, count: packedSize)
        packed.withUnsafeMutableBufferPointer { output in
            quantizeAndPack(rotatedVector, into: output)
        }
        return packed
    }

    /// Quantizes directly into an already allocated packed record.
    func quantizeAndPack(
        _ rotatedVector: UnsafeBufferPointer<Float>,
        into packed: UnsafeMutableBufferPointer<UInt8>
    ) {
        precondition(rotatedVector.count == dimension, "Rotated vector dimension must match")
        precondition(packed.count == packedSize, "Packed vector dimension must match")
        for index in packed.indices {
            packed[index] = 0
        }
        guard bitWidth > 0 else { return }
        for index in 0..<dimension {
            BitPacking.store(
                findNearest(rotatedVector[index]),
                at: index,
                in: packed,
                bitWidth: bitWidth
            )
        }
    }

    /// Fills a query/centroid squared-distance table. The table is laid out as
    /// `coordinate * numCentroids + centroidIndex` for cache-friendly packed lookup.
    func fillDistanceTable(
        for query: UnsafeBufferPointer<Float>,
        into table: inout [Float]
    ) {
        precondition(query.count == dimension, "Query dimension must match")
        precondition(table.count == dimension * numCentroids, "Distance table dimension must match")
        table.withUnsafeMutableBufferPointer { output in
            fillDistanceTable(for: query, into: output)
        }
    }

    func fillDistanceTable(
        for query: UnsafeBufferPointer<Float>,
        into table: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(query.count == dimension, "Query dimension must match")
        precondition(table.count == dimension * numCentroids, "Distance table dimension must match")
        for coordinate in 0..<dimension {
            let value = query[coordinate]
            let row = table.baseAddress! + coordinate * numCentroids
            for centroidIndex in 0..<numCentroids {
                let difference = value - centroids[centroidIndex]
                row[centroidIndex] = difference * difference
            }
        }
    }

    /// Fills a query/centroid inner-product table for cosine/inner-product search.
    /// The table is laid out as `coordinate * numCentroids + centroidIndex`.
    func fillInnerProductTable(
        for query: UnsafeBufferPointer<Float>,
        into table: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(query.count == dimension, "Query dimension must match")
        precondition(table.count == dimension * numCentroids, "Inner-product table dimension must match")
        for coordinate in 0..<dimension {
            let value = query[coordinate]
            let row = table.baseAddress! + coordinate * numCentroids
            for centroidIndex in 0..<numCentroids {
                row[centroidIndex] = value * centroids[centroidIndex]
            }
        }
    }

    /// Builds a byte lookup table for packed inner-product accumulation.
    /// One 256-entry row is produced for every packed byte. The table is used by the
    /// production search path for 1-, 2-, and 4-bit codes; 3-bit codes cross byte
    /// boundaries and use the coordinate table instead.
    func fillPackedInnerProductTable(
        from coordinateTable: UnsafeBufferPointer<Float>,
        into packedTable: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(coordinateTable.count == dimension * numCentroids, "Inner-product table dimension must match")
        precondition(packedTable.count == packedSize * 256, "Packed inner-product table dimension must match")
        precondition(
            bitWidth == 1 || bitWidth == 2 || bitWidth == 4,
            "Packed lookup supports only one-, two-, and four-bit codes"
        )

        var twoBitUpper = SIMD16<Float>(repeating: 0)
        var twoBitLower = SIMD16<Float>(repeating: 0)
        for byteIndex in 0..<packedSize {
            let row = packedTable.baseAddress! + byteIndex * 256
            switch bitWidth {
            case 4:
                let firstCoordinate = byteIndex * 2
                for packedValue in 0..<256 {
                    let first = packedValue >> 4
                    let second = packedValue & 0x0F
                    var total: Float = 0
                    if firstCoordinate < dimension {
                        total += coordinateTable[firstCoordinate * numCentroids + first]
                    }
                    if firstCoordinate + 1 < dimension {
                        total += coordinateTable[
                            (firstCoordinate + 1) * numCentroids + second
                        ]
                    }
                    row[packedValue] = total
                }
            case 2:
                let firstCoordinate = byteIndex * 4
                for value in 0..<16 {
                    twoBitUpper[value] = 0
                    twoBitLower[value] = 0
                    if firstCoordinate < dimension {
                        twoBitUpper[value] += coordinateTable[
                            firstCoordinate * numCentroids + (value >> 2)
                        ]
                    }
                    if firstCoordinate + 1 < dimension {
                        twoBitUpper[value] += coordinateTable[
                            (firstCoordinate + 1) * numCentroids + (value & 0x03)
                        ]
                    }
                    if firstCoordinate + 2 < dimension {
                        twoBitLower[value] += coordinateTable[
                            (firstCoordinate + 2) * numCentroids + (value >> 2)
                        ]
                    }
                    if firstCoordinate + 3 < dimension {
                        twoBitLower[value] += coordinateTable[
                            (firstCoordinate + 3) * numCentroids + (value & 0x03)
                        ]
                    }
                }
                for packedValue in 0..<256 {
                    row[packedValue] = twoBitUpper[packedValue >> 4]
                        + twoBitLower[packedValue & 0x0F]
                }
            case 1:
                let firstCoordinate = byteIndex * 8
                var baseline: Float = 0
                for lane in 0..<8 {
                    let coordinate = firstCoordinate + lane
                    guard coordinate < dimension else { break }
                    baseline += coordinateTable[coordinate * numCentroids]
                }
                row[0] = baseline
                for packedValue in 1..<256 {
                    let value = UInt8(packedValue)
                    let changedBit = value & (~value &+ 1)
                    let lane = 7 - changedBit.trailingZeroBitCount
                    let coordinate = firstCoordinate + lane
                    let delta: Float
                    if coordinate < dimension {
                        let coordinateOffset = coordinate * numCentroids
                        delta = coordinateTable[coordinateOffset + 1]
                            - coordinateTable[coordinateOffset]
                    } else {
                        delta = 0
                    }
                    row[packedValue] = row[packedValue ^ Int(changedBit)] + delta
                }
            default:
                break
            }
        }
    }

    /// Computes ADC distance using a precomputed query table and packed indices.
    @inline(__always)
    func distance(
        from packed: UnsafeBufferPointer<UInt8>,
        using table: [Float]
    ) -> Float {
        precondition(packed.count >= packedSize, "Packed vector is truncated")
        precondition(table.count == dimension * numCentroids, "Distance table dimension must match")
        return table.withUnsafeBufferPointer { distance(from: packed, using: $0) }
    }

    @inline(__always)
    func distance(
        from packed: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(packed.count >= packedSize, "Packed vector is truncated")
        precondition(table.count == dimension * numCentroids, "Distance table dimension must match")
        var total: Float = 0
        switch bitWidth {
        case 4:
            for byteIndex in 0..<packedSize {
                let byte = packed[byteIndex]
                let first = byteIndex * 2
                if first < dimension {
                    total += table.baseAddress![first * numCentroids + Int(byte >> 4)]
                }
                if first + 1 < dimension {
                    total += table.baseAddress![(first + 1) * numCentroids + Int(byte & 0x0F)]
                }
            }
        case 2:
            for byteIndex in 0..<packedSize {
                let byte = packed[byteIndex]
                let first = byteIndex * 4
                for lane in 0..<4 where first + lane < dimension {
                    let index = (byte >> (6 - lane * 2)) & 0x03
                    total += table.baseAddress![(first + lane) * numCentroids + Int(index)]
                }
            }
        case 1:
            for byteIndex in 0..<packedSize {
                let byte = packed[byteIndex]
                let first = byteIndex * 8
                for lane in 0..<8 where first + lane < dimension {
                    let index = (byte >> (7 - lane)) & 0x01
                    total += table.baseAddress![(first + lane) * numCentroids + Int(index)]
                }
            }
        default:
            for coordinate in 0..<dimension {
                let index = BitPacking.value(at: coordinate, from: packed, bitWidth: bitWidth)
                total += table.baseAddress![coordinate * numCentroids + Int(index)]
            }
        }
        return total
    }

    /// Computes the approximate inner-product distance `1 - dot(query, dequantized)`.
    @inline(__always)
    func innerProductDistance(
        from packed: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        1 - innerProduct(from: packed, using: table)
    }

    /// Computes the approximate inner product from coordinate-level lookup values.
    @inline(__always)
    func innerProduct(
        from packed: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(packed.count >= packedSize, "Packed vector is truncated")
        precondition(table.count == dimension * numCentroids, "Inner-product table dimension must match")
        guard bitWidth > 0 else { return 0 }
        if bitWidth == 4 {
            return innerProductFourBit(from: packed, using: table)
        } else if bitWidth == 3 {
            return innerProductThreeBit(from: packed, using: table)
        }
        var dot: Float = 0
        for coordinate in 0..<dimension {
            let index = BitPacking.value(at: coordinate, from: packed, bitWidth: bitWidth)
            dot += table.baseAddress![coordinate * numCentroids + Int(index)]
        }
        return dot
    }

    @inline(__always)
    private func innerProductFourBit(
        from packed: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        let pairCount = dimension / 2
        var byteIndex = 0
        var accumulator0: Float = 0
        var accumulator1: Float = 0
        var accumulator2: Float = 0
        var accumulator3: Float = 0
        while byteIndex + 4 <= pairCount {
            let byte0 = packed[byteIndex]
            let byte1 = packed[byteIndex + 1]
            let byte2 = packed[byteIndex + 2]
            let byte3 = packed[byteIndex + 3]
            let row0 = table.baseAddress! + byteIndex * 32
            let row1 = row0 + 32
            let row2 = row1 + 32
            let row3 = row2 + 32
            accumulator0 += row0[Int(byte0 >> 4)]
                + row0[16 + Int(byte0 & 0x0F)]
            accumulator1 += row1[Int(byte1 >> 4)]
                + row1[16 + Int(byte1 & 0x0F)]
            accumulator2 += row2[Int(byte2 >> 4)]
                + row2[16 + Int(byte2 & 0x0F)]
            accumulator3 += row3[Int(byte3 >> 4)]
                + row3[16 + Int(byte3 & 0x0F)]
            byteIndex += 4
        }
        var total = accumulator0 + accumulator1 + accumulator2 + accumulator3
        while byteIndex < pairCount {
            let byte = packed[byteIndex]
            let row = table.baseAddress! + byteIndex * 32
            total += row[Int(byte >> 4)]
            total += row[16 + Int(byte & 0x0F)]
            byteIndex += 1
        }
        if dimension & 1 != 0 {
            total += table[(dimension - 1) * 16 + Int(packed[pairCount] >> 4)]
        }
        return total
    }

    /// Computes the exact Float32 four-bit centroid dot product through the
    /// platform kernel without constructing a per-query packed lookup table.
    @inline(__always)
    func fourBitInnerProduct(
        from packed: UnsafeBufferPointer<UInt8>,
        query: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(bitWidth == 4, "Native four-bit lookup requires four-bit codes")
        precondition(packed.count >= packedSize, "Packed vector is truncated")
        precondition(query.count == dimension, "Query dimension must match")
        // The Arrays that produced these buffers retain every initialized element for this
        // synchronous borrow. The validated counts bound all byte and Float32 loads, the
        // buffers are independently owned, and the C kernel neither stores nor returns a pointer.
        return fourBitCentroidBytePlanes.withUnsafeBufferPointer { bytePlanes in
            swift_hnsw_turboquant_four_bit_inner_product(
                packed.baseAddress!,
                query.baseAddress!,
                dimension,
                bytePlanes.baseAddress!
            )
        }
    }

    @inline(__always)
    private func innerProductThreeBit(
        from packed: UnsafeBufferPointer<UInt8>,
        using table: UnsafeBufferPointer<Float>
    ) -> Float {
        var total: Float = 0
        var coordinate = 0
        var byteIndex = 0
        while coordinate + 8 <= dimension {
            let first = packed[byteIndex]
            let second = packed[byteIndex + 1]
            let third = packed[byteIndex + 2]
            let row = table.baseAddress! + coordinate * 8
            total += row[Int(first & 0x07)]
            total += row[8 + Int((first >> 3) & 0x07)]
            total += row[16 + Int((first >> 6) | ((second & 0x01) << 2))]
            total += row[24 + Int((second >> 1) & 0x07)]
            total += row[32 + Int((second >> 4) & 0x07)]
            total += row[40 + Int((second >> 7) | ((third & 0x03) << 1))]
            total += row[48 + Int((third >> 2) & 0x07)]
            total += row[56 + Int((third >> 5) & 0x07)]
            coordinate += 8
            byteIndex += 3
        }
        while coordinate < dimension {
            let index = BitPacking.value(at: coordinate, from: packed, bitWidth: 3)
            total += table[coordinate * 8 + Int(index)]
            coordinate += 1
        }
        return total
    }

    /// Computes the approximate inner-product distance using byte-level lookup tables.
    @inline(__always)
    func packedInnerProductDistance(
        from packed: UnsafeBufferPointer<UInt8>,
        using packedTable: UnsafeBufferPointer<Float>
    ) -> Float {
        1 - packedInnerProduct(from: packed, using: packedTable)
    }

    /// Computes the approximate inner product with byte-level lookup values.
    @inline(__always)
    func packedInnerProduct(
        from packed: UnsafeBufferPointer<UInt8>,
        using packedTable: UnsafeBufferPointer<Float>
    ) -> Float {
        precondition(packed.count >= packedSize, "Packed vector is truncated")
        precondition(packedTable.count == packedSize * 256, "Packed inner-product table dimension must match")
        precondition(
            bitWidth == 1 || bitWidth == 2 || bitWidth == 4,
            "Packed lookup supports only one-, two-, and four-bit codes"
        )
        var byteIndex = 0
        var accumulator0: Float = 0
        var accumulator1: Float = 0
        var accumulator2: Float = 0
        var accumulator3: Float = 0
        var accumulator4: Float = 0
        var accumulator5: Float = 0
        var accumulator6: Float = 0
        var accumulator7: Float = 0
        while byteIndex + 8 <= packedSize {
            let row = packedTable.baseAddress! + byteIndex * 256
            accumulator0 += row[Int(packed[byteIndex])]
            accumulator1 += row[256 + Int(packed[byteIndex + 1])]
            accumulator2 += row[512 + Int(packed[byteIndex + 2])]
            accumulator3 += row[768 + Int(packed[byteIndex + 3])]
            accumulator4 += row[1_024 + Int(packed[byteIndex + 4])]
            accumulator5 += row[1_280 + Int(packed[byteIndex + 5])]
            accumulator6 += row[1_536 + Int(packed[byteIndex + 6])]
            accumulator7 += row[1_792 + Int(packed[byteIndex + 7])]
            byteIndex += 8
        }
        var dot = accumulator0 + accumulator1 + accumulator2 + accumulator3
            + accumulator4 + accumulator5 + accumulator6 + accumulator7
        while byteIndex < packedSize {
            dot += packedTable.baseAddress![
                byteIndex * 256 + Int(packed[byteIndex])
            ]
            byteIndex += 1
        }
        return dot
    }

    /// Dequantize packed bytes back to a float vector (centroid lookup).
    func dequantize(_ packed: [UInt8]) -> [Float] {
        var output = [Float](repeating: 0, count: dimension)
        packed.withUnsafeBufferPointer { packedBuffer in
            output.withUnsafeMutableBufferPointer { outputBuffer in
                dequantize(packedBuffer, into: outputBuffer)
            }
        }
        return output
    }

    /// Reconstructs centroid values into a caller-owned workspace.
    func dequantize(
        _ packed: UnsafeBufferPointer<UInt8>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(packed.count >= packedSize, "Packed vector is truncated")
        precondition(output.count == dimension, "Output dimension must match")
        guard bitWidth > 0 else {
            for index in output.indices {
                output[index] = 0
            }
            return
        }
        for coordinate in 0..<dimension {
            let index = BitPacking.value(at: coordinate, from: packed, bitWidth: bitWidth)
            output[coordinate] = centroids[Int(index)]
        }
    }

    /// Find the nearest centroid index for a single value using binary search on boundaries.
    @inline(__always)
    private func findNearest(_ value: Float) -> UInt8 {
        // Binary search through boundaries
        var lo = 0
        var hi = boundaries.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if value <= boundaries[mid] {
                hi = mid
            } else {
                lo = mid + 1
            }
        }
        return UInt8(lo)
    }

    // MARK: - Lloyd-Max Centroids

    /// Returns the unscaled Lloyd-Max optimal centroids for N(0,1) distribution.
    /// These are well-known values from quantization theory.
    private static func gaussianLloydMaxCentroids(bitWidth: Int) -> [Float] {
        switch bitWidth {
        case 0:
            return [0]
        case 1:
            // 2-level: E[|X|] for half-normal = sqrt(2/pi)
            return [-0.7978845608, 0.7978845608]
        case 2:
            // 4-level Lloyd-Max for N(0,1)
            return [-1.5104176099, -0.4527800398, 0.4527800398, 1.5104176099]
        case 3:
            // 8-level Lloyd-Max for N(0,1)
            return [
                -2.1519757910, -1.3439092514, -0.7560052486, -0.2451209526,
                 0.2451209526,  0.7560052486,  1.3439092514,  2.1519757910,
            ]
        case 4:
            // 16-level Lloyd-Max for N(0,1)
            return [
                -2.7326368678, -2.0690724890, -1.6180447530, -1.2562067015,
                -0.9423402190, -0.6568065643, -0.3880170386, -0.1284185495,
                 0.1284185495,  0.3880170386,  0.6568065643,  0.9423402190,
                 1.2562067015,  1.6180447530,  2.0690724890,  2.7326368678,
            ]
        default:
            preconditionFailure("Unsupported bitWidth: \(bitWidth)")
        }
    }

    /// Computes finite-dimensional Lloyd-Max centroids for one coordinate of a uniformly
    /// distributed unit-sphere vector. Midpoint quadrature avoids evaluating the integrable
    /// endpoint singularity present when `dimension == 2`.
    private static func sphericalLloydMaxCentroids(
        bitWidth: Int,
        dimension: Int
    ) -> [Float] {
        let levelCount = 1 << bitWidth
        guard bitWidth > 0 else { return [0] }
        if dimension == 1 {
            let negativeCount = levelCount / 2
            return (0..<levelCount).map { $0 < negativeCount ? -1 : 1 }
        }
        if dimension == 3 {
            let step = 2.0 / Float(levelCount)
            return (0..<levelCount).map { -1 + (Float($0) + 0.5) * step }
        }

        let sampleCount = 32_768
        let step = 2.0 / Double(sampleCount)
        let exponent = Double(dimension - 3) / 2
        var samples = [Double](repeating: 0, count: sampleCount)
        var logarithmicWeights = [Double](repeating: 0, count: sampleCount)
        var maximumLogarithmicWeight = -Double.infinity
        for index in 0..<sampleCount {
            let value = -1 + (Double(index) + 0.5) * step
            let logarithmicWeight = exponent * hnswNaturalLog(1 - value * value)
            samples[index] = value
            logarithmicWeights[index] = logarithmicWeight
            maximumLogarithmicWeight = max(
                maximumLogarithmicWeight,
                logarithmicWeight
            )
        }
        var weights = [Double](repeating: 0, count: sampleCount)
        for index in 0..<sampleCount {
            weights[index] = hnswExponential(
                logarithmicWeights[index] - maximumLogarithmicWeight
            )
        }

        let gaussian = gaussianLloydMaxCentroids(bitWidth: bitWidth)
        let scale = 1.0 / Double(dimension).squareRoot()
        var centroids = gaussian.map {
            max(-1, min(1, Double($0) * scale))
        }
        var boundaries = [Double](repeating: 0, count: levelCount - 1)
        var weightedSums = [Double](repeating: 0, count: levelCount)
        var weightSums = [Double](repeating: 0, count: levelCount)

        for _ in 0..<64 {
            for index in boundaries.indices {
                boundaries[index] = (centroids[index] + centroids[index + 1]) / 2
            }
            for index in weightedSums.indices {
                weightedSums[index] = 0
                weightSums[index] = 0
            }

            var region = 0
            for sampleIndex in 0..<sampleCount {
                let value = samples[sampleIndex]
                while region < boundaries.count, value > boundaries[region] {
                    region += 1
                }
                let weight = weights[sampleIndex]
                weightedSums[region] += value * weight
                weightSums[region] += weight
            }

            var maximumChange: Double = 0
            for index in centroids.indices where weightSums[index] > 0 {
                let updated = weightedSums[index] / weightSums[index]
                maximumChange = max(maximumChange, abs(updated - centroids[index]))
                centroids[index] = updated
            }
            for index in 0..<(levelCount / 2) {
                let opposite = levelCount - 1 - index
                let magnitude = (centroids[opposite] - centroids[index]) / 2
                centroids[index] = -magnitude
                centroids[opposite] = magnitude
            }
            if maximumChange < 1e-10 {
                break
            }
        }
        return centroids.map(Float.init)
    }
}
