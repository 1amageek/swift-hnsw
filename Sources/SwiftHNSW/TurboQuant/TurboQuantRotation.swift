/// Deterministic three-stage randomized Hadamard rotation used by TurboQuant.
///
/// The transform is `p^(-3/2) H D3 H D2 H D1 [x; 0]`, where `p` is the next power of
/// two above the input dimension. The scale preserves the Euclidean norm because each
/// unnormalised Hadamard stage multiplies the norm by `sqrt(p)`.
struct TurboQuantRotation: Sendable {
    let dimensions: Int
    let paddedDimensions: Int
    let strategy: TurboQuantRotationStrategy
    private let signs1: [Float]
    private let signs2: [Float]
    private let signs3: [Float]
    private let haarReflections: [Float]
    private let haarOffsets: [Int]
    private let haarDiagonal: [Float]
    private let scale: Float

    init(
        dimensions: Int,
        paddedDimensions: Int,
        strategy: TurboQuantRotationStrategy = .structuredHadamard,
        seed: UInt64
    ) {
        precondition(
            strategy == .structuredHadamard || paddedDimensions == dimensions,
            "Haar rotation must not pad the input dimension"
        )
        self.dimensions = dimensions
        self.paddedDimensions = paddedDimensions
        self.strategy = strategy
        let scaleDimension = Float(paddedDimensions)
        self.scale = 1 / (scaleDimension * scaleDimension.squareRoot())

        var generator = TurboQuantSplitMix64(seed: seed)
        if strategy == .structuredHadamard {
            self.signs1 = Self.makeSigns(count: paddedDimensions, generator: &generator)
            self.signs2 = Self.makeSigns(count: paddedDimensions, generator: &generator)
            self.signs3 = Self.makeSigns(count: paddedDimensions, generator: &generator)
            self.haarReflections = []
            self.haarOffsets = []
            self.haarDiagonal = []
        } else {
            self.signs1 = []
            self.signs2 = []
            self.signs3 = []
            let haar = Self.makeHaarRepresentation(
                dimension: dimensions,
                generator: &generator
            )
            self.haarReflections = haar.reflections
            self.haarOffsets = haar.offsets
            self.haarDiagonal = haar.diagonal
        }
    }

    /// Rotates an already normalized vector. This is used by callers that retain the
    /// normalized input and avoids changing the rotation's normalization contract.
    func rotateNormalized(
        _ input: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>,
        work: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == dimensions, "Input dimension must match")
        precondition(output.count == paddedDimensions, "Output dimension must match")
        precondition(work.count == paddedDimensions, "Work dimension must match")

        if strategy == .haar {
            for index in 0..<dimensions {
                work[index] = input[index] * haarDiagonal[index]
            }
            for reflectionIndex in haarOffsets.indices.reversed() {
                applyHaarReflection(
                    reflectionIndex,
                    to: work
                )
            }
            for index in 0..<dimensions {
                output[index] = work[index]
            }
            return
        }

        for index in 0..<paddedDimensions {
            work[index] = index < dimensions ? input[index] : 0
            work[index] *= signs1[index]
        }
        hadamardTransform(work)
        apply(signs2, to: work)
        hadamardTransform(work)
        apply(signs3, to: work)
        hadamardTransform(work)
        for index in 0..<paddedDimensions {
            output[index] = work[index] * scale
        }
    }

    /// Applies the transpose of the rotation and returns the original, unpadded coordinates.
    ///
    /// The transform is orthogonal over the padded space. The three Hadamard stages are
    /// reversed and the same scale is applied once after all stages.
    func inverseRotate(
        _ input: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>,
        work: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count == paddedDimensions, "Input dimension must match")
        precondition(output.count == dimensions, "Output dimension must match")
        precondition(work.count == paddedDimensions, "Work dimension must match")

        if strategy == .haar {
            for index in 0..<dimensions {
                work[index] = input[index]
            }
            for reflectionIndex in haarOffsets.indices {
                applyHaarReflection(
                    reflectionIndex,
                    to: work
                )
            }
            for index in 0..<dimensions {
                output[index] = work[index] * haarDiagonal[index]
            }
            return
        }

        for index in 0..<paddedDimensions {
            work[index] = input[index]
        }
        hadamardTransform(work)
        apply(signs3, to: work)
        hadamardTransform(work)
        apply(signs2, to: work)
        hadamardTransform(work)
        apply(signs1, to: work)
        for index in 0..<dimensions {
            output[index] = work[index] * scale
        }
    }

    private func hadamardTransform(_ buffer: UnsafeMutableBufferPointer<Float>) {
        var blockSize = 1
        while blockSize < paddedDimensions {
            var blockStart = 0
            while blockStart < paddedDimensions {
                let lowerStart = blockStart + blockSize
                for offset in 0..<blockSize {
                    let upperIndex = blockStart + offset
                    let lowerIndex = lowerStart + offset
                    let upper = buffer[upperIndex]
                    let lower = buffer[lowerIndex]
                    buffer[upperIndex] = upper + lower
                    buffer[lowerIndex] = upper - lower
                }
                blockStart += blockSize * 2
            }
            blockSize *= 2
        }
    }

    private func applyHaarReflection(
        _ reflectionIndex: Int,
        to buffer: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorOffset = haarOffsets[reflectionIndex]
        let coordinateOffset = reflectionIndex
        let count = dimensions - coordinateOffset
        var dot: Float = 0
        for index in 0..<count {
            dot += haarReflections[vectorOffset + index]
                * buffer[coordinateOffset + index]
        }
        let scale = 2 * dot
        for index in 0..<count {
            buffer[coordinateOffset + index] -= scale
                * haarReflections[vectorOffset + index]
        }
    }

    @inline(__always)
    private func apply(_ signs: [Float], to buffer: UnsafeMutableBufferPointer<Float>) {
        for index in 0..<paddedDimensions {
            buffer[index] *= signs[index]
        }
    }

    private static func makeSigns(
        count: Int,
        generator: inout TurboQuantSplitMix64
    ) -> [Float] {
        (0..<count).map { _ in generator.next() & 1 == 0 ? 1 : -1 }
    }

    private static func makeHaarRepresentation(
        dimension: Int,
        generator: inout TurboQuantSplitMix64
    ) -> (reflections: [Float], offsets: [Int], diagonal: [Float]) {
        var reflections: [Float] = []
        var offsets: [Int] = []
        if dimension > 1 {
            reflections.reserveCapacity(dimension * (dimension + 1) / 2 - 1)
            offsets.reserveCapacity(dimension - 1)
        }
        var vector = [Double](repeating: 0, count: dimension)

        for coordinateOffset in 0..<max(0, dimension - 1) {
            let count = dimension - coordinateOffset
            var squaredNorm: Double = 0
            var index = 0
            while index < count {
                let pair = generator.nextGaussianPair()
                vector[index] = pair.0
                squaredNorm += pair.0 * pair.0
                index += 1
                if index < count {
                    vector[index] = pair.1
                    squaredNorm += pair.1 * pair.1
                    index += 1
                }
            }

            let norm = squaredNorm.squareRoot()
            let direction: Double = vector[0] >= 0 ? 1 : -1
            vector[0] += direction * norm
            var reflectorSquaredNorm: Double = 0
            for reflectorIndex in 0..<count {
                reflectorSquaredNorm += vector[reflectorIndex]
                    * vector[reflectorIndex]
            }
            let inverseNorm = 1 / reflectorSquaredNorm.squareRoot()
            offsets.append(reflections.count)
            for reflectorIndex in 0..<count {
                reflections.append(Float(vector[reflectorIndex] * inverseNorm))
            }
        }

        return (
            reflections,
            offsets,
            makeSigns(count: dimension, generator: &generator)
        )
    }
}
