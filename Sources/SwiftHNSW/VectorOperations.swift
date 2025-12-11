import Accelerate

/// High-performance vector operations using Accelerate framework
enum VectorOperations {

    // MARK: - Float32 Operations

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

    // MARK: - Float16 Operations

    /// Normalize a single Float16 vector to unit length
    /// Converts to Float32 for Accelerate operations, then back to Float16
    static func normalize(_ vector: [Float16]) -> [Float16] {
        // Convert to Float32 for Accelerate
        let float32 = vector.map { Float($0) }
        let normalized = normalize(float32)
        return normalized.map { Float16($0) }
    }

    /// Normalize a batch of Float16 vectors to unit length
    static func normalizeBatch(_ vectors: [Float16], count: Int, dimensions: Int) -> [Float16] {
        // Convert to Float32 for Accelerate
        let float32 = vectors.map { Float($0) }
        let normalized = normalizeBatch(float32, count: count, dimensions: dimensions)
        return normalized.map { Float16($0) }
    }
}
