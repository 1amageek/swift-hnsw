import Synchronization

/// Benchmark-only access to the production four-bit scoring implementations.
@_spi(Benchmarking)
public final class TurboQuantFourBitBenchmarkHarness: Sendable {
    private struct State: Sendable {
        var coordinateTable: [Float]
        var packedTable: [Float]
    }

    public let dimensions: Int
    public let packedSize: Int

    private let quantizer: ScalarQuantizer
    private let state: Mutex<State>

    public init(dimensions: Int) {
        precondition(dimensions > 0, "Dimensions must be positive")
        let quantizer = ScalarQuantizer(bitWidth: 4, dimension: dimensions)
        self.dimensions = dimensions
        self.packedSize = quantizer.packedSize
        self.quantizer = quantizer
        self.state = Mutex(
            State(
                coordinateTable: [Float](repeating: 0, count: dimensions * 16),
                packedTable: [Float](repeating: 0, count: quantizer.packedSize * 256)
            )
        )
    }

    public func quantize(_ vector: [Float]) -> [UInt8] {
        precondition(vector.count == dimensions, "Vector dimension must match")
        return quantizer.quantizeAndPack(vector)
    }

    /// Returns the sixteen immutable centroids for an external benchmark reference.
    public func centroidTable() -> [Float] {
        quantizer.centroids
    }

    /// Scores borrowed packed codes while reusing the same lookup storage as an index query.
    ///
    /// Every pointer is borrowed only by the synchronous quantizer call. The owning arrays
    /// remain retained for the complete borrow, all counts are validated before pointer
    /// creation, no buffers overlap, and no pointer escapes or is deallocated by this method.
    public func score(
        _ packedVectors: [[UInt8]],
        query: [Float],
        backend: TurboQuantFourBitBackend,
        usesPackedLookup: Bool,
        into scores: inout [Float]
    ) {
        precondition(query.count == dimensions, "Query dimension must match")
        precondition(
            packedVectors.allSatisfy { $0.count == packedSize },
            "Packed vector dimension must match"
        )
        if scores.count != packedVectors.count {
            scores = [Float](repeating: 0, count: packedVectors.count)
        }

        state.withLock { state in
            query.withUnsafeBufferPointer { queryBuffer in
                switch backend {
                case .platform, .swiftDirect:
                    for index in packedVectors.indices {
                        scores[index] = packedVectors[index].withUnsafeBufferPointer { packed in
                            if backend == .platform {
                                quantizer.fourBitInnerProduct(
                                    from: packed,
                                    query: queryBuffer
                                )
                            } else {
                                quantizer.directFourBitInnerProduct(
                                    from: packed,
                                    query: queryBuffer
                                )
                            }
                        }
                    }
                case .swiftLookup:
                    state.coordinateTable.withUnsafeMutableBufferPointer { coordinateTable in
                        quantizer.fillInnerProductTable(
                            for: queryBuffer,
                            into: coordinateTable
                        )
                        if usesPackedLookup {
                            state.packedTable.withUnsafeMutableBufferPointer { packedTable in
                                quantizer.fillPackedInnerProductTable(
                                    from: UnsafeBufferPointer(coordinateTable),
                                    into: packedTable
                                )
                                let lookup = UnsafeBufferPointer(packedTable)
                                for index in packedVectors.indices {
                                    scores[index] = packedVectors[index]
                                        .withUnsafeBufferPointer { packed in
                                            quantizer.packedInnerProduct(
                                                from: packed,
                                                using: lookup
                                            )
                                        }
                                }
                            }
                        } else {
                            let lookup = UnsafeBufferPointer(coordinateTable)
                            for index in packedVectors.indices {
                                scores[index] = packedVectors[index]
                                    .withUnsafeBufferPointer { packed in
                                        quantizer.innerProduct(
                                            from: packed,
                                            using: lookup
                                        )
                                    }
                            }
                        }
                    }
                }
            }
        }
    }
}
