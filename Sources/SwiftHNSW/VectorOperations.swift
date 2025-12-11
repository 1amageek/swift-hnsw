import Accelerate

/// High-performance vector operations using Accelerate framework
enum VectorOperations {

    /// Normalize a single vector to unit length using SIMD
    static func normalize(_ vector: [Float]) -> [Float] {
        var result = vector
        var magnitude: Float = 0
        vDSP_dotpr(vector, 1, vector, 1, &magnitude, vDSP_Length(vector.count))
        magnitude = sqrt(magnitude)
        guard magnitude > 0 else { return vector }
        var scale = 1.0 / magnitude
        vDSP_vsmul(vector, 1, &scale, &result, 1, vDSP_Length(vector.count))
        return result
    }

    /// Normalize a batch of vectors to unit length
    static func normalizeBatch(_ vectors: [Float], count: Int, dimensions: Int) -> [Float] {
        var result = vectors
        for i in 0..<count {
            let offset = i * dimensions
            var magnitude: Float = 0
            vectors.withUnsafeBufferPointer { buffer in
                vDSP_dotpr(
                    buffer.baseAddress! + offset, 1,
                    buffer.baseAddress! + offset, 1,
                    &magnitude, vDSP_Length(dimensions)
                )
            }
            magnitude = sqrt(magnitude)
            if magnitude > 0 {
                var scale = 1.0 / magnitude
                result.withUnsafeMutableBufferPointer { buffer in
                    vDSP_vsmul(
                        buffer.baseAddress! + offset, 1,
                        &scale,
                        buffer.baseAddress! + offset, 1,
                        vDSP_Length(dimensions)
                    )
                }
            }
        }
        return result
    }
}
