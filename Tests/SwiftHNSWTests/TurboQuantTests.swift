// TurboQuantTests.swift
// Unit tests for TurboQuant vector quantization

import Testing
import Foundation
@testable import SwiftHNSW

private func removeTemporaryFile(atPath path: String) {
    guard FileManager.default.fileExists(atPath: path) else {
        return
    }
    do {
        try FileManager.default.removeItem(atPath: path)
    } catch {
        #expect(Bool(false), "Failed to remove temporary file: \(error)")
    }
}

// MARK: - BitPacking Tests

@Suite("BitPacking Tests")
struct BitPackingTests {

    @Test("Roundtrip b=4", arguments: [8, 128, 768, 1024])
    func roundtrip4(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<16) }
        let packed = BitPacking.pack(indices, bitWidth: 4)
        let unpacked = BitPacking.unpack(packed, bitWidth: 4, count: count)
        #expect(unpacked == indices)
        #expect(packed.count == (count + 1) / 2)
    }

    @Test("Roundtrip b=2", arguments: [8, 128, 768, 1024])
    func roundtrip2(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<4) }
        let packed = BitPacking.pack(indices, bitWidth: 2)
        let unpacked = BitPacking.unpack(packed, bitWidth: 2, count: count)
        #expect(unpacked == indices)
    }

    @Test("Roundtrip b=1", arguments: [8, 128, 1024])
    func roundtrip1(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<2) }
        let packed = BitPacking.pack(indices, bitWidth: 1)
        let unpacked = BitPacking.unpack(packed, bitWidth: 1, count: count)
        #expect(unpacked == indices)
    }

    @Test("Roundtrip b=3", arguments: [8, 128, 1024])
    func roundtrip3(count: Int) {
        let indices: [UInt8] = (0..<count).map { _ in UInt8.random(in: 0..<8) }
        let packed = BitPacking.pack(indices, bitWidth: 3)
        let unpacked = BitPacking.unpack(packed, bitWidth: 3, count: count)
        #expect(unpacked == indices)
    }
}

// MARK: - ScalarQuantizer Tests

@Suite("ScalarQuantizer Tests")
struct ScalarQuantizerTests {

    @Test("Codebook symmetry", arguments: [0, 1, 2, 3, 4])
    func codebookSymmetry(b: Int) {
        let sq = ScalarQuantizer(bitWidth: b, dimension: 128)
        let n = sq.numCentroids
        for i in 0..<n / 2 {
            #expect(abs(sq.centroids[i] + sq.centroids[n - 1 - i]) < 1e-6)
        }
    }

    @Test("Finite spherical Lloyd-Max matches the uniform three-dimensional case")
    func sphericalLloydMaxUniformCase() {
        let quantizer = ScalarQuantizer(
            bitWidth: 2,
            dimension: 3,
            usesExactSphericalDistribution: true
        )
        let expected: [Float] = [-0.75, -0.25, 0.25, 0.75]
        for index in expected.indices {
            #expect(abs(quantizer.centroids[index] - expected[index]) < 1e-7)
        }
    }

    @Test("Finite spherical Lloyd-Max matches the two-dimensional one-bit moment")
    func sphericalLloydMaxTwoDimensionalMoment() {
        let quantizer = ScalarQuantizer(
            bitWidth: 1,
            dimension: 2,
            usesExactSphericalDistribution: true
        )
        let expected = Float(2 / Double.pi)
        #expect(abs(quantizer.centroids[0] + expected) < 0.005)
        #expect(abs(quantizer.centroids[1] - expected) < 0.005)
    }

    @Test("Finite spherical Lloyd-Max is symmetric", arguments: [2, 17, 128])
    func sphericalLloydMaxSymmetry(dimension: Int) {
        let quantizer = ScalarQuantizer(
            bitWidth: 4,
            dimension: dimension,
            usesExactSphericalDistribution: true
        )
        for index in 0..<(quantizer.numCentroids / 2) {
            let opposite = quantizer.numCentroids - 1 - index
            #expect(
                abs(quantizer.centroids[index] + quantizer.centroids[opposite])
                    < 1e-6
            )
        }
    }

    @Test("Zero-bit quantizer reconstructs zero")
    func zeroBitQuantizer() {
        let dimension = 17
        let quantizer = ScalarQuantizer(bitWidth: 0, dimension: dimension)
        let vector = (0..<dimension).map { Float($0) / 17 }
        let packed = quantizer.quantizeAndPack(vector)
        #expect(packed.isEmpty)
        #expect(quantizer.dequantize(packed) == [Float](repeating: 0, count: dimension))
    }

    @Test("Quantize-dequantize MSE bound")
    func quantizeDequantizeMSE() {
        let d = 1024
        let sq = ScalarQuantizer(bitWidth: 4, dimension: d)
        let scale = 1.0 / Float(d).squareRoot()
        let vector = (0..<d).map { _ in Float.random(in: -3 * scale...3 * scale) }
        let packed = sq.quantizeAndPack(vector)
        let decoded = sq.dequantize(packed)

        var mse: Float = 0
        for i in 0..<d {
            let diff = vector[i] - decoded[i]
            mse += diff * diff
        }
        mse /= Float(d)
        // Paper: D_mse ≈ 0.009 for b=4 → per-dim ≈ 0.009/d
        #expect(mse < 0.001, "Per-dim MSE should be bounded")
    }

    @Test("Precomputed ADC table matches decoded distance", arguments: [1, 2, 3, 4])
    func precomputedADCMatchesDecodedDistance(bitWidth: Int) {
        let dimension = 17
        let quantizer = ScalarQuantizer(bitWidth: bitWidth, dimension: dimension)
        let vector = (0..<dimension).map { index in
            Float((index * 13 % 19) - 9) / 12
        }
        let packed = quantizer.quantizeAndPack(vector)
        var table = [Float](repeating: 0, count: dimension * quantizer.numCentroids)
        vector.withUnsafeBufferPointer { buffer in
            quantizer.fillDistanceTable(for: buffer, into: &table)
        }
        let packedDistance = packed.withUnsafeBufferPointer { buffer in
            quantizer.distance(from: buffer, using: table)
        }
        let decoded = quantizer.dequantize(packed)
        let decodedDistance = zip(vector, decoded).reduce(Float.zero) { partial, pair in
            let difference = pair.0 - pair.1
            return partial + difference * difference
        }
        #expect(abs(packedDistance - decodedDistance) < 1e-5)
    }

    @Test("Packed inner-product lookup matches decoded score", arguments: [1, 2, 4])
    func packedInnerProductMatchesDecodedScore(bitWidth: Int) {
        let dimension = 17
        let quantizer = ScalarQuantizer(bitWidth: bitWidth, dimension: dimension)
        let query = (0..<dimension).map { index in
            Float((index * 7 % 23) - 11) / 13
        }
        let vector = (0..<dimension).map { index in
            Float((index * 13 % 19) - 9) / 12
        }
        let packed = quantizer.quantizeAndPack(vector)
        var coordinateTable = [Float](repeating: 0, count: dimension * quantizer.numCentroids)
        query.withUnsafeBufferPointer { buffer in
            coordinateTable.withUnsafeMutableBufferPointer { table in
                quantizer.fillInnerProductTable(for: buffer, into: table)
            }
        }
        let coordinateDistance = packed.withUnsafeBufferPointer { packedBuffer in
            coordinateTable.withUnsafeBufferPointer { table in
                quantizer.innerProductDistance(from: packedBuffer, using: table)
            }
        }
        var packedTable = [Float](repeating: 0, count: quantizer.packedSize * 256)
        coordinateTable.withUnsafeBufferPointer { table in
            packedTable.withUnsafeMutableBufferPointer { packedBuffer in
                quantizer.fillPackedInnerProductTable(from: table, into: packedBuffer)
            }
        }
        let packedDistance = packed.withUnsafeBufferPointer { packedBuffer in
            packedTable.withUnsafeBufferPointer { packedBufferTable in
                quantizer.packedInnerProductDistance(
                    from: packedBuffer,
                    using: packedBufferTable
                )
            }
        }
        let decoded = quantizer.dequantize(packed)
        let decodedDistance = 1 - zip(query, decoded).reduce(Float.zero) { partial, pair in
            partial + pair.0 * pair.1
        }
        #expect(abs(coordinateDistance - decodedDistance) < 1e-5)
        #expect(abs(packedDistance - decodedDistance) < 1e-5)
    }

    @Test(
        "Four-bit kernels match decoded score",
        arguments: [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 767, 768, 769]
    )
    func fourBitKernelsMatchDecodedScore(dimension: Int) {
        let quantizer = ScalarQuantizer(bitWidth: 4, dimension: dimension)
        let query = (0..<dimension).map { index in
            Float((index * 7 % 23) - 11) / 13
        }
        let vector = (0..<dimension).map { index in
            Float((index * 13 % 19) - 9) / 12
        }
        let packed = quantizer.quantizeAndPack(vector)
        let platformScore = packed.withUnsafeBufferPointer { packedBuffer in
            query.withUnsafeBufferPointer { queryBuffer in
                quantizer.fourBitInnerProduct(
                    from: packedBuffer,
                    query: queryBuffer
                )
            }
        }
        let directScore = packed.withUnsafeBufferPointer { packedBuffer in
            query.withUnsafeBufferPointer { queryBuffer in
                quantizer.directFourBitInnerProduct(
                    from: packedBuffer,
                    query: queryBuffer
                )
            }
        }
        let scalarCScore = packed.withUnsafeBufferPointer { packedBuffer in
            query.withUnsafeBufferPointer { queryBuffer in
                quantizer.scalarCFourBitInnerProduct(
                    from: packedBuffer,
                    query: queryBuffer
                )
            }
        }
        let decoded = quantizer.dequantize(packed)
        let decodedScore = zip(query, decoded).reduce(Float.zero) { partial, pair in
            partial + pair.0 * pair.1
        }
        #expect(abs(platformScore - decodedScore) < 1e-4)
        #expect(abs(scalarCScore - decodedScore) < 1e-4)
        #expect(abs(directScore - decodedScore) < 1e-4)
    }

    @Test("Four-bit fallback coordinate and packed paths agree")
    func fourBitFallbackPathsAgree() throws {
        let dimension = 257
        let vectorCount = 16
        let index = try TurboQuantIndex(
            dimensions: dimension,
            maxElements: vectorCount,
            bitWidth: 4,
            objective: .meanSquaredError,
            rotationStrategy: .structuredHadamard,
            configuration: HNSWConfiguration(
                m: 8,
                efConstruction: 32,
                efSearch: 31
            ),
            seed: 42,
            fourBitBackend: .swiftLookup
        )
        for vectorIndex in 0..<vectorCount {
            let vector = (0..<dimension).map { coordinate in
                Float(
                    ((vectorIndex + 3) * (coordinate + 5)) % 37 - 18
                ) / 19
            }
            try index.add(vector, label: UInt64(vectorIndex))
        }
        try index.finalize()
        let query = (0..<dimension).map { coordinate in
            Float((coordinate * 11) % 31 - 15) / 17
        }

        let coordinateResults = try index.search(query, k: vectorCount)
        try index.setEfSearch(32)
        let packedResults = try index.search(query, k: vectorCount)

        #expect(coordinateResults.map(\.label) == packedResults.map(\.label))
        for (coordinate, packed) in zip(coordinateResults, packedResults) {
            #expect(abs(coordinate.distance - packed.distance) < 1e-4)
        }
    }
}

// MARK: - TurboQuant Product Estimator Tests

@Suite("TurboQuant Product Estimator Tests")
struct TurboQuantProductEstimatorTests {

    @Test("Structured rotation roundtrip", arguments: [17, 64, 129])
    func rotationRoundtrip(dimension: Int) {
        var padded = 1
        while padded < dimension {
            padded *= 2
        }
        let rotation = TurboQuantRotation(
            dimensions: dimension,
            paddedDimensions: padded,
            seed: 42
        )
        let input = (0..<dimension).map { index in
            Float((index * 17 % 31) - 15) / 19
        }
        var rotated = [Float](repeating: 0, count: padded)
        var work = [Float](repeating: 0, count: padded)
        input.withUnsafeBufferPointer { inputBuffer in
            rotated.withUnsafeMutableBufferPointer { rotatedBuffer in
                work.withUnsafeMutableBufferPointer { workBuffer in
                    rotation.rotateNormalized(
                        inputBuffer,
                        into: rotatedBuffer,
                        work: workBuffer
                    )
                }
            }
        }

        var reconstructed = [Float](repeating: 0, count: dimension)
        rotated.withUnsafeBufferPointer { rotatedBuffer in
            reconstructed.withUnsafeMutableBufferPointer { reconstructedBuffer in
                work.withUnsafeMutableBufferPointer { workBuffer in
                    rotation.inverseRotate(
                        rotatedBuffer,
                        into: reconstructedBuffer,
                        work: workBuffer
                    )
                }
            }
        }
        for index in 0..<dimension {
            #expect(abs(input[index] - reconstructed[index]) < 2e-5)
        }
    }

    @Test("Haar rotation roundtrip and norm preservation", arguments: [17, 64])
    func haarRotationRoundtrip(dimension: Int) {
        let rotation = TurboQuantRotation(
            dimensions: dimension,
            paddedDimensions: dimension,
            strategy: .haar,
            seed: 42
        )
        let input = VectorOperations.normalize(
            (0..<dimension).map { index in
                Float((index * 17 % 31) - 15) / 19
            }
        )
        var rotated = [Float](repeating: 0, count: dimension)
        var work = [Float](repeating: 0, count: dimension)
        input.withUnsafeBufferPointer { inputBuffer in
            rotated.withUnsafeMutableBufferPointer { rotatedBuffer in
                work.withUnsafeMutableBufferPointer { workBuffer in
                    rotation.rotateNormalized(
                        inputBuffer,
                        into: rotatedBuffer,
                        work: workBuffer
                    )
                }
            }
        }
        let rotatedNorm = rotated.reduce(Float.zero) { $0 + $1 * $1 }
        #expect(abs(rotatedNorm - 1) < 2e-5)

        var reconstructed = [Float](repeating: 0, count: dimension)
        rotated.withUnsafeBufferPointer { rotatedBuffer in
            reconstructed.withUnsafeMutableBufferPointer { reconstructedBuffer in
                work.withUnsafeMutableBufferPointer { workBuffer in
                    rotation.inverseRotate(
                        rotatedBuffer,
                        into: reconstructedBuffer,
                        work: workBuffer
                    )
                }
            }
        }
        for index in input.indices {
            #expect(abs(input[index] - reconstructed[index]) < 2e-5)
        }
    }

    @Test("Haar rotation has the expected coordinate moments")
    func haarCoordinateMoments() {
        let dimension = 16
        let sampleCount = 512
        let input = [Float](repeating: 0, count: dimension).enumerated().map {
            $0.offset == 0 ? 1 : $0.element
        }
        var coordinateSum: Float = 0
        var coordinateSquaredSum: Float = 0

        for seed in 0..<sampleCount {
            let rotation = TurboQuantRotation(
                dimensions: dimension,
                paddedDimensions: dimension,
                strategy: .haar,
                seed: UInt64(seed)
            )
            var rotated = [Float](repeating: 0, count: dimension)
            var work = [Float](repeating: 0, count: dimension)
            input.withUnsafeBufferPointer { inputBuffer in
                rotated.withUnsafeMutableBufferPointer { rotatedBuffer in
                    work.withUnsafeMutableBufferPointer { workBuffer in
                        rotation.rotateNormalized(
                            inputBuffer,
                            into: rotatedBuffer,
                            work: workBuffer
                        )
                    }
                }
            }
            coordinateSum += rotated[0]
            coordinateSquaredSum += rotated[0] * rotated[0]
        }

        let mean = coordinateSum / Float(sampleCount)
        let secondMoment = coordinateSquaredSum / Float(sampleCount)
        #expect(abs(mean) < 0.04)
        #expect(abs(secondMoment - 1 / Float(dimension)) < 0.015)
    }

    @Test("Packed QJL lookup matches direct signed sum")
    func packedQJLLookup() throws {
        let dimension = 17
        let projection = try QJLProjection(dimension: dimension, seed: 123)
        let query = (0..<dimension).map { index in
            Float((index * 7 % 23) - 11) / 13
        }
        let vector = (0..<dimension).map { index in
            Float((index * 11 % 29) - 14) / 17
        }
        var projectedQuery = [Float](repeating: 0, count: dimension)
        var projectedVector = [Float](repeating: 0, count: dimension)
        query.withUnsafeBufferPointer { input in
            projectedQuery.withUnsafeMutableBufferPointer { output in
                projection.project(input, into: output)
            }
        }
        vector.withUnsafeBufferPointer { input in
            projectedVector.withUnsafeMutableBufferPointer { output in
                projection.project(input, into: output)
            }
        }

        var packedSigns = [UInt8](repeating: 0, count: projection.packedSize)
        projectedVector.withUnsafeBufferPointer { projected in
            packedSigns.withUnsafeMutableBufferPointer { packed in
                projection.packSigns(projected, into: packed)
            }
        }
        var table = [Float](repeating: 0, count: projection.lookupTableCount)
        let absoluteSum = projectedQuery.withUnsafeBufferPointer { projected in
            table.withUnsafeMutableBufferPointer { output in
                projection.fillInnerProductTable(for: projected, into: output)
            }
        }
        let packedValue = packedSigns.withUnsafeBufferPointer { signs in
            table.withUnsafeBufferPointer { lookup in
                projection.innerProduct(signs: signs, using: lookup)
            }
        }
        let directValue = zip(projectedQuery, projectedVector).reduce(Float.zero) {
            $0 + ($1.1 >= 0 ? $1.0 : -$1.0)
        }
        let directAbsoluteSum = projectedQuery.reduce(Float.zero) { $0 + abs($1) }
        #expect(abs(packedValue - directValue) < 1e-5)
        #expect(absoluteSum >= directAbsoluteSum)
    }

    @Test(
        "QJL platform lookup preserves Swift reduction order",
        arguments: [0, 1, 2, 3, 4, 5, 8, 9, 15, 16, 63, 64, 65, 96, 97, 128, 1_024]
    )
    func qjlPlatformLookupPreservesSwiftReductionOrder(packedSize: Int) {
        let signs = (0..<packedSize).map { byte in
            UInt8(truncatingIfNeeded: byte &* 73 &+ 0x5A)
        }
        let table = (0..<(packedSize * QJLProjection.lookupEntriesPerByte)).map { index in
            Float((index &* 29 % 113) - 56) / 37
        }
        var byteIndex = 0
        var accumulator0: Float = 0
        var accumulator1: Float = 0
        var accumulator2: Float = 0
        var accumulator3: Float = 0
        while byteIndex + 4 <= packedSize {
            let row = byteIndex * QJLProjection.lookupEntriesPerByte
            let byte0 = signs[byteIndex]
            let byte1 = signs[byteIndex + 1]
            let byte2 = signs[byteIndex + 2]
            let byte3 = signs[byteIndex + 3]
            accumulator0 += table[row + Int(byte0 >> 4)]
                + table[row + 16 + Int(byte0 & 0x0F)]
            accumulator1 += table[row + 32 + Int(byte1 >> 4)]
                + table[row + 48 + Int(byte1 & 0x0F)]
            accumulator2 += table[row + 64 + Int(byte2 >> 4)]
                + table[row + 80 + Int(byte2 & 0x0F)]
            accumulator3 += table[row + 96 + Int(byte3 >> 4)]
                + table[row + 112 + Int(byte3 & 0x0F)]
            byteIndex += 4
        }
        var expected = accumulator0 + accumulator1 + accumulator2 + accumulator3
        while byteIndex < packedSize {
            let row = byteIndex * QJLProjection.lookupEntriesPerByte
            let byte = signs[byteIndex]
            expected += table[row + Int(byte >> 4)]
                + table[row + 16 + Int(byte & 0x0F)]
            byteIndex += 1
        }
        let actual = signs.withUnsafeBufferPointer { packedSigns in
            table.withUnsafeBufferPointer { lookupTable in
                QJLProjection.innerProduct(signs: packedSigns, using: lookupTable)
            }
        }
        #expect(actual.bitPattern == expected.bitPattern)
    }

    @Test("QJL pruning bound dominates every packed sign reduction", arguments: [17, 128, 768])
    func qjlPruningBoundDominatesPackedReduction(dimension: Int) throws {
        let projection = try QJLProjection(dimension: dimension, seed: 0xA11C_E55)
        var query: [Float] = []
        query.reserveCapacity(dimension)
        for index in 0..<dimension {
            let numerator = ((index * 37 + 11) % 101) - 50
            query.append(Float(numerator) / 53)
        }
        var projectedQuery = [Float](repeating: 0, count: dimension)
        query.withUnsafeBufferPointer { input in
            projectedQuery.withUnsafeMutableBufferPointer { output in
                projection.project(input, into: output)
            }
        }

        var table = [Float](repeating: 0, count: projection.lookupTableCount)
        let absoluteSum = projectedQuery.withUnsafeBufferPointer { projected in
            table.withUnsafeMutableBufferPointer { output in
                projection.fillInnerProductTable(for: projected, into: output)
            }
        }

        for sample in 0..<128 {
            var signs = [UInt8](repeating: 0, count: projection.packedSize)
            for byte in signs.indices {
                signs[byte] = UInt8(truncatingIfNeeded: sample &* 29 &+ byte &* 71 &+ 0x5A)
            }
            let value = signs.withUnsafeBufferPointer { packedSigns in
                table.withUnsafeBufferPointer { lookup in
                    projection.innerProduct(signs: packedSigns, using: lookup)
                }
            }
            #expect(
                abs(value) <= absoluteSum,
                "Packed QJL reduction exceeded its pruning bound at sample \(sample)"
            )
        }
    }

    @Test("QJL distance pruning preserves product search results")
    func qjlDistancePruningPreservesResults() throws {
        let dimension = 32
        let vectorCount = 256
        let configuration = HNSWConfiguration(
            m: 8,
            efConstruction: 32,
            efSearch: 100
        )

        let makeIndex: (TurboQuantQJLPruningMode) throws -> TurboQuantIndex = {
            pruningMode in
            let index = try TurboQuantIndex(
                dimensions: dimension,
                maxElements: vectorCount,
                bitWidth: 5,
                objective: .innerProduct,
                rotationStrategy: .structuredHadamard,
                configuration: configuration,
                seed: 19,
                qjlPruningMode: pruningMode,
                collectQJLPruningStatistics: pruningMode != .never
            )
            for vectorIndex in 0..<vectorCount {
                let vector = (0..<dimension).map { coordinate in
                    Float(((vectorIndex + 5) * (coordinate + 3)) % 47 - 23) / 29
                }
                try index.add(vector, label: UInt64(vectorIndex))
            }
            try index.finalize()
            return index
        }

        let query = (0..<dimension).map { coordinate in
            Float((coordinate * 13) % 31 - 15) / 17
        }
        let reference = try makeIndex(.never)
        let optimized = try makeIndex(.automatic)
        let referenceResults = try reference.search(query, k: 10)
        optimized.resetQJLPruningStatistics()
        let optimizedResults = try optimized.search(query, k: 10)

        #expect(referenceResults.map(\.label) == optimizedResults.map(\.label))
        for (reference, optimized) in zip(referenceResults, optimizedResults) {
            #expect(abs(reference.distance - optimized.distance) < 1e-5)
        }
        let statistics = optimized.qjlPruningStatistics()
        #expect(statistics.scoredCandidates > 0)
        #expect(statistics.prunedCandidates > 0)
        #expect(statistics.prunedCandidates < statistics.scoredCandidates)

        try optimized.setEfSearch(99)
        optimized.resetQJLPruningStatistics()
        _ = try optimized.search(query, k: 10)
        #expect(optimized.qjlPruningStatistics().prunedCandidates == 0)

        try optimized.setEfSearch(100)
        optimized.resetQJLPruningStatistics()
        _ = try optimized.search(query, k: 10)
        #expect(optimized.qjlPruningStatistics().prunedCandidates > 0)
    }

    @Test("TurboQuant product estimate is unbiased across projection seeds")
    func productEstimateIsUnbiased() throws {
        let dimension = 32
        let totalBitWidth = 3
        let sampleCount = 256
        let input = VectorOperations.normalize(
            (0..<dimension).map { index in
                Float((index * 17 % 31) - 15) / 19
            }
        )
        let query = VectorOperations.normalize(
            (0..<dimension).map { index in
                Float((index * 11 % 29) - 14) / 17
            }
        )
        let exact = zip(input, query).reduce(Float.zero) { $0 + $1.0 * $1.1 }
        var estimateSum: Float = 0

        for seed in 0..<sampleCount {
            let typedSeed = UInt64(seed)
            let rotation = TurboQuantRotation(
                dimensions: dimension,
                paddedDimensions: dimension,
                seed: typedSeed
            )
            let quantizer = ScalarQuantizer(
                bitWidth: totalBitWidth - 1,
                dimension: dimension
            )
            let projection = try QJLProjection(
                dimension: dimension,
                seed: typedSeed ^ 0xD1B5_4A32_D192_ED03
            )

            var rotationWork = [Float](repeating: 0, count: dimension)
            var rotatedInput = [Float](repeating: 0, count: dimension)
            input.withUnsafeBufferPointer { inputBuffer in
                rotatedInput.withUnsafeMutableBufferPointer { rotatedBuffer in
                    rotationWork.withUnsafeMutableBufferPointer { workBuffer in
                        rotation.rotateNormalized(
                            inputBuffer,
                            into: rotatedBuffer,
                            work: workBuffer
                        )
                    }
                }
            }
            let code = quantizer.quantizeAndPack(rotatedInput)
            let reconstructedRotated = quantizer.dequantize(code)
            var residual = [Float](repeating: 0, count: dimension)
            reconstructedRotated.withUnsafeBufferPointer { reconstructedBuffer in
                residual.withUnsafeMutableBufferPointer { residualBuffer in
                    rotationWork.withUnsafeMutableBufferPointer { workBuffer in
                        rotation.inverseRotate(
                            reconstructedBuffer,
                            into: residualBuffer,
                            work: workBuffer
                        )
                    }
                }
            }
            var residualSquaredNorm: Float = 0
            for index in 0..<dimension {
                residual[index] = input[index] - residual[index]
                residualSquaredNorm += residual[index] * residual[index]
            }

            var projectedResidual = [Float](repeating: 0, count: dimension)
            var projectedQuery = [Float](repeating: 0, count: dimension)
            residual.withUnsafeBufferPointer { residualBuffer in
                projectedResidual.withUnsafeMutableBufferPointer { projectedBuffer in
                    projection.project(residualBuffer, into: projectedBuffer)
                }
            }
            query.withUnsafeBufferPointer { queryBuffer in
                projectedQuery.withUnsafeMutableBufferPointer { projectedBuffer in
                    projection.project(queryBuffer, into: projectedBuffer)
                }
            }
            let qjlSignedSum = zip(projectedQuery, projectedResidual).reduce(Float.zero) {
                $0 + ($1.1 >= 0 ? $1.0 : -$1.0)
            }

            var rotatedQuery = [Float](repeating: 0, count: dimension)
            query.withUnsafeBufferPointer { queryBuffer in
                rotatedQuery.withUnsafeMutableBufferPointer { rotatedBuffer in
                    rotationWork.withUnsafeMutableBufferPointer { workBuffer in
                        rotation.rotateNormalized(
                            queryBuffer,
                            into: rotatedBuffer,
                            work: workBuffer
                        )
                    }
                }
            }
            let mseInnerProduct = zip(rotatedQuery, reconstructedRotated)
                .reduce(Float.zero) { $0 + $1.0 * $1.1 }
            let qjlScale = Float(1.253_314_137_315_500_3) / Float(dimension)
            estimateSum += mseInnerProduct
                + residualSquaredNorm.squareRoot() * qjlScale * qjlSignedSum
        }

        let meanEstimate = estimateSum / Float(sampleCount)
        #expect(abs(meanEstimate - exact) < 0.04)
    }
}

// MARK: - TurboQuantIndex Tests

@Suite("TurboQuantIndex Tests")
struct TurboQuantIndexTests {

    @Test("Basic add and search")
    func basicAddSearch() throws {
        let dim = 128
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 100, bitWidth: 4, seed: 42)

        for i in 0..<20 {
            let v = (0..<dim).map { _ in Float.random(in: -1...1) }
            try index.add(v, label: UInt64(i))
        }

        #expect(index.count == 20)
        #expect(index.paddedDimensions == 128) // 128 is power of 2

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try index.search(query, k: 5)
        #expect(results.count == 5)
        #expect(index.isFinalized) // auto-finalized on first search
    }

    @Test("Non-power-of-2 dimension")
    func nonPow2Dimension() throws {
        let dim = 768
        let index = try TurboQuantIndex(
            dimensions: dim,
            maxElements: 50,
            bitWidth: 4,
            objective: .innerProduct,
            seed: 42
        )

        #expect(index.paddedDimensions == 1024)
        #expect(index.objective == .innerProduct)
        // MSE: 1024*3/8, QJL: 768/8, residual norm: 4 bytes.
        #expect(index.bytesPerVector == 484)
        #expect(abs(index.compressionRatio - Float(3072) / 484) < 1e-6)

        for i in 0..<20 {
            let v = (0..<dim).map { _ in Float.random(in: -1...1) }
            try index.add(v, label: UInt64(i))
        }
        let results = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 5)
        #expect(results.count == 5)
    }

    @Test("Exact Haar strategy uses the original dimension")
    func haarNonPowerOfTwoDimension() throws {
        let dimensions = 33
        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: 8,
            bitWidth: 4,
            rotationStrategy: .haar,
            seed: 42
        )
        #expect(index.paddedDimensions == dimensions)
        #expect(index.rotationStrategy == .haar)
        #expect(index.bytesPerVector == 17)

        for label in 0..<8 {
            try index.add(
                (0..<dimensions).map {
                    Float((label * 13 + $0 * 7) % 29 - 14)
                },
                label: UInt64(label)
            )
        }
        #expect(try index.search([Float](repeating: 1, count: dimensions), k: 3).count == 3)
    }

    @Test("Dimension mismatch errors")
    func dimensionMismatch() throws {
        let index = try TurboQuantIndex(dimensions: 64, maxElements: 10, bitWidth: 4)
        #expect(throws: HNSWError.self) { try index.add([Float](repeating: 0, count: 32), label: 0) }
        #expect(throws: HNSWError.self) { try index.search([Float](repeating: 0, count: 32), k: 1) }
        #expect(
            throws: HNSWError.invalidArgument(
                "batch dimensions exceed the supported integer range"
            )
        ) {
            _ = try index.searchBatch([], numQueries: Int.max, k: 1)
        }
    }

    @Test("Invalid vector values fail explicitly")
    func invalidVectorValues() throws {
        let index = try TurboQuantIndex(dimensions: 3, maxElements: 4)
        #expect(throws: HNSWError.invalidArgument("vector must have nonzero norm")) {
            try index.add([0, 0, 0], label: 0)
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "vector must contain only finite values"
            )
        ) {
            try index.add([1, .nan, 0], label: 0)
        }
        try index.add([1, 0, 0], label: 0)
        #expect(throws: HNSWError.invalidArgument("query must have nonzero norm")) {
            _ = try index.search([0, 0, 0], k: 1)
        }
        #expect(
            throws: HNSWError.invalidArgument(
                "query must contain only finite values"
            )
        ) {
            _ = try index.search([1, .infinity, 0], k: 1)
        }
    }

    @Test("Haar storage limit fails before allocation")
    func haarStorageLimit() {
        #expect(
            throws: HNSWError.initializationFailed(
                "Haar rotation storage exceeds the 256 MiB limit"
            )
        ) {
            _ = try TurboQuantIndex(
                dimensions: 12_000,
                maxElements: 1,
                rotationStrategy: .haar
            )
        }
    }

    @Test("Cannot add after finalize")
    func cannotAddAfterFinalize() throws {
        let dim = 64
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 50, bitWidth: 4)
        try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 0)
        _ = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 1) // triggers finalize
        #expect(index.isFinalized)
        #expect(throws: HNSWError.self) {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 1)
        }
    }

    @Test("All MSE bit widths", arguments: [1, 2, 3, 4])
    func allMSEBitWidths(b: Int) throws {
        let dim = 64
        let p = 64 // 64 is power of 2
        let n = 30
        let index = try TurboQuantIndex(dimensions: dim, maxElements: n, bitWidth: b, seed: 42)

        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        let results = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 5)
        #expect(results.count == 5)
        #expect(index.objective == .meanSquaredError)
        #expect(index.bytesPerVector == BitPacking.packedSize(count: p, bitWidth: b))
    }

    @Test("All product bit widths", arguments: [1, 2, 3, 4, 5])
    func allProductBitWidths(b: Int) throws {
        let dim = 64
        let p = 64
        let n = 30
        let index = try TurboQuantIndex(
            dimensions: dim,
            maxElements: n,
            bitWidth: b,
            objective: .innerProduct,
            seed: 42
        )

        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        let results = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 5)
        #expect(results.count == 5)
        #expect(index.objective == .innerProduct)
        let expectedBytes = BitPacking.packedSize(count: p, bitWidth: b - 1)
            + BitPacking.packedSize(count: dim, bitWidth: 1)
            + MemoryLayout<Float>.size
        #expect(index.bytesPerVector == expectedBytes)
    }

    @Test("One-bit product objective drives graph traversal with QJL")
    func oneBitProductTraversal() throws {
        let dimension = 16
        var positive = [Float](repeating: 0, count: dimension)
        positive[0] = 1
        let negative = positive.map { -$0 }
        let index = try TurboQuantIndex(
            dimensions: dimension,
            maxElements: 2,
            bitWidth: 1,
            objective: .innerProduct,
            configuration: HNSWConfiguration(m: 2, efConstruction: 2),
            seed: 42
        )
        try index.add(positive, label: 10)
        try index.add(negative, label: 20)
        try index.setEfSearch(1)

        #expect(try index.search(positive, k: 1).first?.label == 10)
        #expect(try index.search(negative, k: 1).first?.label == 20)
    }

    @Test("Recall on power-of-2 dimension (d=128, b=4)")
    func recallPow2() throws {
        let dim = 128
        let n = 500
        let k = 10
        let numQueries = 50

        let trainVectors = seededVectors(count: n, dimension: dim, seed: 0x1280_0001)
        let testQueries = seededVectors(count: numQueries, dimension: dim, seed: 0x1280_0002)

        // Cosine ground truth
        let groundTruths = computeCosineGroundTruth(queries: testQueries, vectors: trainVectors, k: k)

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        for (i, v) in trainVectors.enumerated() { try index.add(v, label: UInt64(i)) }
        try index.setEfSearch(200)

        var totalRecall: Double = 0
        for qi in 0..<numQueries {
            let results = try index.search(testQueries[qi], k: k)
            let found = Set(results.prefix(k).map { $0.label })
            let truth = Set(groundTruths[qi].prefix(k))
            totalRecall += Double(found.intersection(truth).count) / Double(k)
        }
        let recall = totalRecall / Double(numQueries)
        #expect(recall > 0.70, "d=128 b=4 recall should exceed 70%, got \(recall)")
    }

    @Test("Recall on non-power-of-2 dimension (d=768, b=4)")
    func recallNonPow2() throws {
        let dim = 768
        let n = 500
        let k = 10
        let numQueries = 50

        let trainVectors = seededVectors(count: n, dimension: dim, seed: 0x7680_0001)
        let testQueries = seededVectors(count: numQueries, dimension: dim, seed: 0x7680_0002)

        let groundTruths = computeCosineGroundTruth(queries: testQueries, vectors: trainVectors, k: k)

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        for (i, v) in trainVectors.enumerated() { try index.add(v, label: UInt64(i)) }
        try index.setEfSearch(200)

        var totalRecall: Double = 0
        for qi in 0..<numQueries {
            let results = try index.search(testQueries[qi], k: k)
            let found = Set(results.prefix(k).map { $0.label })
            let truth = Set(groundTruths[qi].prefix(k))
            totalRecall += Double(found.intersection(truth).count) / Double(k)
        }
        let recall = totalRecall / Double(numQueries)
        #expect(recall > 0.70, "d=768 b=4 recall should exceed 70%, got \(recall)")
    }

    @Test("Serialization roundtrip")
    func serializationRoundtrip() throws {
        let dim = 128
        let n = 100
        let k = 5

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            objective: .innerProduct,
            rotationStrategy: .haar,
            configuration: HNSWConfiguration(m: 16, efConstruction: 100), seed: 42)
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        for (i, v) in vectors.enumerated() { try index.add(v, label: UInt64(i)) }

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let resultsBefore = try index.search(query, k: k)

        // Save
        let tmpPath = NSTemporaryDirectory() + "tq_test_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { removeTemporaryFile(atPath: tmpPath) }

        let fileSize = try FileManager.default.attributesOfItem(atPath: tmpPath)[.size] as! Int
        #expect(fileSize > 0)

        // Load
        let loaded = try TurboQuantIndex.load(from: tmpPath)
        #expect(loaded.dimensions == dim)
        #expect(loaded.bitWidth == 4)
        #expect(loaded.objective == .innerProduct)
        #expect(loaded.rotationStrategy == .haar)
        #expect(loaded.paddedDimensions == 128)
        #expect(loaded.count == n)
        #expect(loaded.isFinalized)

        // Search loaded index — should return same results
        let resultsAfter = try loaded.search(query, k: k)
        #expect(resultsAfter.count == k)

        // Same labels in same order
        let labelsBefore = resultsBefore.map { $0.label }
        let labelsAfter = resultsAfter.map { $0.label }
        #expect(labelsBefore == labelsAfter, "Search results should match after load")
        #expect(resultsBefore == resultsAfter)
    }

    @Test("Archive stores packed vectors and graph metadata")
    func archiveUsesPackedRepresentation() throws {
        let dimensions = 128
        let count = 32
        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: count,
            bitWidth: 4,
            configuration: HNSWConfiguration(m: 8, efConstruction: 32),
            seed: 42
        )
        for label in 0..<count {
            try index.add(
                (0..<dimensions).map { Float((label * 17 + $0 * 5) % 31) / 31 },
                label: UInt64(label)
            )
        }

        let path = NSTemporaryDirectory() + "tq_packed_\(UUID().uuidString).bin"
        defer { removeTemporaryFile(atPath: path) }
        try index.save(to: path)
        let fileSize = try FileManager.default.attributesOfItem(atPath: path)[.size] as! Int
        let rawFloatBytes = dimensions * MemoryLayout<Float>.size * count
        #expect(fileSize < rawFloatBytes)
    }

    @Test("Serialization roundtrip non-power-of-2")
    func serializationNonPow2() throws {
        let dim = 768
        let n = 50
        let k = 5

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 8, efConstruction: 50), seed: 99)
        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }

        let tmpPath = NSTemporaryDirectory() + "tq_test768_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { removeTemporaryFile(atPath: tmpPath) }

        let loaded = try TurboQuantIndex.load(from: tmpPath)
        #expect(loaded.dimensions == 768)
        #expect(loaded.paddedDimensions == 1024)
        #expect(loaded.count == n)

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let results = try loaded.search(query, k: k)
        #expect(results.count == k)
    }
    // MARK: - Issue #17: efSearch must be preserved after save/load

    @Test("efSearch preserved after save/load")
    func efSearchPreserved() throws {
        let dim = 128
        let n = 500
        let k = 10
        let efSearch = 200

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        try index.setEfSearch(efSearch)

        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let resultsBefore = try index.search(query, k: k)

        let tmpPath = NSTemporaryDirectory() + "tq_ef_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { removeTemporaryFile(atPath: tmpPath) }

        let loaded = try TurboQuantIndex.load(from: tmpPath)
        let resultsAfter = try loaded.search(query, k: k)
        #expect(
            resultsBefore == resultsAfter,
            "Loaded index should preserve efSearch and produce identical results"
        )
    }

    // MARK: - Issue #6: Header serialization must be portable

    @Test("Header field-by-field consistency")
    func headerFieldConsistency() throws {
        let dim = 768
        let n = 10

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 3, seed: 12345)
        for i in 0..<n {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }

        let tmpPath = NSTemporaryDirectory() + "tq_header_\(UUID().uuidString).bin"
        try index.save(to: tmpPath)
        defer { removeTemporaryFile(atPath: tmpPath) }

        let loaded = try TurboQuantIndex.load(from: tmpPath)
        #expect(loaded.dimensions == 768)
        #expect(loaded.bitWidth == 3)
        #expect(loaded.seed == 12345)
        #expect(loaded.paddedDimensions == 1024)
        #expect(loaded.count == n)
    }

    // MARK: - Issue #1: _finalized must be thread-safe

    @Test("Add after search throws error")
    func addAfterSearchThrows() throws {
        let dim = 64
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 20, bitWidth: 4)
        for i in 0..<5 {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }
        // search triggers finalize
        _ = try index.search((0..<dim).map { _ in Float.random(in: -1...1) }, k: 1)
        #expect(index.isFinalized)

        // add after finalize must throw, not silently corrupt
        #expect(throws: HNSWError.self) {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 99)
        }
    }

    @Test("Add after save throws error")
    func addAfterSaveThrows() throws {
        let dim = 64
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 20, bitWidth: 4)
        for i in 0..<5 {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: UInt64(i))
        }

        let tmpPath = NSTemporaryDirectory() + "tq_addsave_\(UUID().uuidString).bin"
        try index.save(to: tmpPath) // triggers finalize
        defer { removeTemporaryFile(atPath: tmpPath) }

        #expect(index.isFinalized)
        #expect(throws: HNSWError.self) {
            try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 99)
        }
    }

    // MARK: - Issue #3: Large dimension must not crash

    @Test("Large dimension encoding does not crash")
    func largeDimensionEncoding() throws {
        // d=3072 → p=4096, alloca would use ~49KB
        let dim = 3072
        let index = try TurboQuantIndex(dimensions: dim, maxElements: 5, bitWidth: 4, seed: 42)

        // Should not crash from stack overflow
        let v = (0..<dim).map { _ in Float.random(in: -1...1) }
        try index.add(v, label: 0)
        try index.add((0..<dim).map { _ in Float.random(in: -1...1) }, label: 1)

        let results = try index.search(v, k: 1)
        #expect(results.count == 1)
        #expect(results[0].label == 0) // should find itself
    }

    // MARK: - Quantized graph contract

    @Test("Graph construction uses full precision and search uses ADC")
    func graphConstructionAndADCSearch() throws {
        // The graph is built from rotated Float32 vectors, then searched with packed
        // asymmetric distances after finalization. A self-query must still rank first.
        let dim = 64
        let n = 100
        let k = 10

        let index = try TurboQuantIndex(
            dimensions: dim, maxElements: n, bitWidth: 4,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200), seed: 42)
        let vectors = (0..<n).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }
        for (i, v) in vectors.enumerated() { try index.add(v, label: UInt64(i)) }

        try index.setEfSearch(200)
        let query = vectors[0] // search for first vector
        let results = try index.search(query, k: k)
        // With Float32 construction, the first result should be the vector itself (label 0)
        // or very close to it
        #expect(results[0].label == 0, "Should find the query vector as nearest neighbor")
    }
}

// MARK: - Temporary Archive Lifecycle

private func computeCosineGroundTruth(queries: [[Float]], vectors: [[Float]], k: Int) -> [[UInt64]] {
    let d = vectors[0].count
    return queries.map { query in
        let normQ = query.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        let dists: [(UInt64, Float)] = vectors.enumerated().map { (i, v) in
            let normV = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            var dot: Float = 0
            for j in 0..<d { dot += query[j] * v[j] }
            return (UInt64(i), 1.0 - dot / (normQ * normV))
        }
        return dists.sorted { $0.1 < $1.1 }.prefix(k).map { $0.0 }
    }
}
