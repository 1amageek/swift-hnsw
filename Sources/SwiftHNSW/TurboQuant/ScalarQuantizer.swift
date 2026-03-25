import Foundation

/// Scalar quantizer using Lloyd-Max centroids for the standard normal distribution.
///
/// After random rotation, each coordinate of a unit-sphere vector approximately follows
/// N(0, 1/d). The optimal scalar quantizer centroids for N(0,1) are well-known and
/// hardcoded here, then scaled by 1/sqrt(d).
struct ScalarQuantizer: Sendable {

    /// Quantization bit-width (1, 2, 3, or 4)
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

    init(bitWidth: Int, dimension: Int) {
        precondition((1...4).contains(bitWidth), "bitWidth must be 1, 2, 3, or 4")
        precondition(dimension > 0, "dimension must be positive")

        self.bitWidth = bitWidth
        self.dimension = dimension
        self.numCentroids = 1 << bitWidth

        let scale = 1.0 / Float(dimension).squareRoot()
        let unscaled = Self.lloydMaxCentroids(bitWidth: bitWidth)
        self.centroids = unscaled.map { $0 * scale }

        // Decision boundaries are midpoints between consecutive centroids
        var bounds = [Float](repeating: 0, count: numCentroids - 1)
        for i in 0..<bounds.count {
            bounds[i] = (self.centroids[i] + self.centroids[i + 1]) * 0.5
        }
        self.boundaries = bounds

        self.packedSize = BitPacking.packedSize(count: dimension, bitWidth: bitWidth)
    }

    /// Quantize a rotated vector to an array of centroid indices.
    func quantize(_ rotatedVector: [Float]) -> [UInt8] {
        var indices = [UInt8](repeating: 0, count: dimension)
        for j in 0..<dimension {
            indices[j] = findNearest(rotatedVector[j])
        }
        return indices
    }

    /// Quantize and pack a rotated vector into compact bytes.
    func quantizeAndPack(_ rotatedVector: [Float]) -> [UInt8] {
        let indices = quantize(rotatedVector)
        return BitPacking.pack(indices, bitWidth: bitWidth)
    }

    /// Dequantize packed bytes back to a float vector (centroid lookup).
    func dequantize(_ packed: [UInt8]) -> [Float] {
        let indices = BitPacking.unpack(packed, bitWidth: bitWidth, count: dimension)
        return indices.map { centroids[Int($0)] }
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

    // MARK: - Lloyd-Max Centroids for N(0,1)

    /// Returns the unscaled Lloyd-Max optimal centroids for N(0,1) distribution.
    /// These are well-known values from quantization theory.
    private static func lloydMaxCentroids(bitWidth: Int) -> [Float] {
        switch bitWidth {
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
            fatalError("Unsupported bitWidth: \(bitWidth)")
        }
    }
}
