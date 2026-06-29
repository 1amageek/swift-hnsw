import Foundation
import Synchronization

/// Swift backend TurboQuant facade backed by an exact normalized flat index.
public final class TurboQuantIndex: Sendable {

    private struct Entry: Sendable {
        var offset: Int
    }

    private struct State: Sendable {
        var finalized: Bool
        var efSearch: Int
        var entries: [UInt64: Entry]
        var labelOrder: [UInt64]
        var vectorStorage: [Float]
        var queryScratch: [Float]
    }

    public let dimensions: Int
    public let bitWidth: Int
    public let configuration: HNSWConfiguration
    public let seed: UInt64
    public let paddedDimensions: Int

    private let _packedSize: Int
    private let maximumElementCount: Int
    private let state: Mutex<State>

    public init(
        dimensions: Int,
        maxElements: Int,
        bitWidth: Int = 4,
        configuration: HNSWConfiguration = .balanced,
        seed: UInt64 = 42
    ) throws {
        guard dimensions > 0 else {
            throw HNSWError.initializationFailed("dimensions must be positive")
        }
        guard maxElements > 0 else {
            throw HNSWError.initializationFailed("maxElements must be positive")
        }
        guard (1...4).contains(bitWidth) else {
            throw HNSWError.initializationFailed("bitWidth must be 1, 2, 3, or 4")
        }

        self.dimensions = dimensions
        self.bitWidth = bitWidth
        self.configuration = configuration
        self.seed = seed

        var padded = 1
        while padded < dimensions { padded *= 2 }
        self.paddedDimensions = padded

        self._packedSize = BitPacking.packedSize(count: padded, bitWidth: bitWidth)
        self.maximumElementCount = maxElements
        var vectorStorage: [Float] = []
        if maxElements <= Int.max / dimensions {
            vectorStorage.reserveCapacity(maxElements * dimensions)
        }
        self.state = Mutex(State(
            finalized: false,
            efSearch: configuration.efSearch,
            entries: [:],
            labelOrder: [],
            vectorStorage: vectorStorage,
            queryScratch: [Float](repeating: 0, count: dimensions)
        ))
    }

    public var count: Int {
        state.withLock { $0.entries.count }
    }

    public var capacity: Int { maximumElementCount }
    public var isEmpty: Bool { count == 0 }
    public var isFinalized: Bool { state.withLock { $0.finalized } }
    public var bytesPerVector: Int { _packedSize }
    public var compressionRatio: Float { Float(dimensions * 4) / Float(_packedSize) }

    public func setEfSearch(_ ef: Int) {
        state.withLock {
            $0.efSearch = max(1, ef)
        }
    }

    public func add(_ vector: [Float], label: UInt64) throws {
        try vector.withUnsafeBufferPointer { buffer in
            try add(buffer, label: label)
        }
    }

    /// Add a borrowed vector without materializing an intermediate array.
    public func add(_ vector: UnsafeBufferPointer<Float>, label: UInt64) throws {
        guard vector.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: vector.count)
        }

        try state.withLock {
            guard !$0.finalized else {
                throw HNSWError.addPointFailed("Cannot add after finalize()")
            }
            let offset: Int
            if let existing = $0.entries[label] {
                offset = existing.offset
            } else {
                guard $0.entries.count < maximumElementCount else {
                    throw HNSWError.capacityExceeded(
                        current: $0.entries.count,
                        maximum: maximumElementCount
                    )
                }
                offset = $0.vectorStorage.count
                for _ in 0..<dimensions {
                    $0.vectorStorage.append(0)
                }
                $0.labelOrder.append(label)
            }
            $0.vectorStorage.withUnsafeMutableBufferPointer { storage in
                let destination = UnsafeMutableBufferPointer(start: storage.baseAddress! + offset, count: dimensions)
                VectorOperations.normalize(vector, into: destination)
            }
            $0.entries[label] = Entry(offset: offset)
        }
    }

    public func search(_ query: [Float], k: Int) throws -> [SearchResult] {
        try query.withUnsafeBufferPointer { buffer in
            try search(buffer, k: k)
        }
    }

    /// Search with a borrowed query vector.
    public func search(_ query: UnsafeBufferPointer<Float>, k: Int) throws -> [SearchResult] {
        guard query.count == dimensions else {
            throw HNSWError.dimensionMismatch(expected: dimensions, got: query.count)
        }
        guard k > 0 else { return [] }

        state.withLock {
            $0.finalized = true
        }

        return state.withLock { state in
            ensureQueryScratch(&state)
            state.queryScratch.withUnsafeMutableBufferPointer { scratch in
                VectorOperations.normalize(query, into: scratch)
            }
            var results: [SearchResult] = []
            results.reserveCapacity(min(k, state.entries.count))

            state.queryScratch.withUnsafeBufferPointer { normalizedQuery in
                state.vectorStorage.withUnsafeBufferPointer { storage in
                    for label in state.labelOrder {
                        guard let entry = state.entries[label] else {
                            continue
                        }
                        let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                        let distance = VectorOperations.distance(from: normalizedQuery, to: vector, metric: .cosine)
                        insertTopKSearchResult(
                            SearchResult(label: label, distance: distance),
                            into: &results,
                            limit: k
                        )
                    }
                }
            }
            return results
        }
    }

    private static let headerMagic: UInt32 = 0x54515746 // "TQWF"
    private static let headerVersion: UInt32 = 1
    private static let headerSize = 40

    public func save(to url: URL) throws {
        try save(to: url.path)
    }

    public func save(to path: String) throws {
        let data = state.withLock {
            $0.finalized = true
            var output = Data(capacity: Self.headerSize + $0.entries.count * (8 + dimensions * 4))
            output.appendLittleEndian(Self.headerMagic)
            output.appendLittleEndian(Self.headerVersion)
            output.appendLittleEndian(UInt32(dimensions))
            output.appendLittleEndian(UInt32(bitWidth))
            output.appendLittleEndian(seed)
            output.appendLittleEndian(UInt32(paddedDimensions))
            output.appendLittleEndian(UInt32(maximumElementCount))
            output.appendLittleEndian(UInt32($0.labelOrder.count))

            for label in $0.labelOrder {
                guard let entry = $0.entries[label] else {
                    continue
                }
                output.appendLittleEndian(label)
                $0.vectorStorage.withUnsafeBufferPointer { storage in
                    let vector = UnsafeBufferPointer(start: storage.baseAddress! + entry.offset, count: dimensions)
                    for value in vector {
                        output.appendLittleEndian(value.bitPattern)
                    }
                }
            }
            return output
        }

        do {
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            throw HNSWError.saveFailed("Failed to save index to \(path)")
        }
    }

    public static func load(from url: URL) throws -> TurboQuantIndex {
        try load(from: url.path)
    }

    public static func load(from path: String) throws -> TurboQuantIndex {
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            throw HNSWError.loadFailed("Failed to load index from \(path)")
        }

        guard data.count >= headerSize else {
            throw HNSWError.loadFailed("File too small")
        }

        var offset = 0
        let magic: UInt32 = data.readLittleEndian(at: &offset)
        let version: UInt32 = data.readLittleEndian(at: &offset)

        guard magic == headerMagic else {
            throw HNSWError.loadFailed("Invalid Swift backend TurboQuant file magic")
        }
        guard version == headerVersion else {
            throw HNSWError.loadFailed("Unsupported Swift backend TurboQuant version \(version)")
        }

        let dimensions = Int(data.readLittleEndian(at: &offset) as UInt32)
        let bitWidth = Int(data.readLittleEndian(at: &offset) as UInt32)
        let seed: UInt64 = data.readLittleEndian(at: &offset)
        let paddedDimensions = Int(data.readLittleEndian(at: &offset) as UInt32)
        let capacity = Int(data.readLittleEndian(at: &offset) as UInt32)
        let labelCount = Int(data.readLittleEndian(at: &offset) as UInt32)

        let index = try TurboQuantIndex(
            dimensions: dimensions,
            maxElements: max(capacity, labelCount),
            bitWidth: bitWidth,
            configuration: .balanced,
            seed: seed
        )

        guard index.paddedDimensions == paddedDimensions else {
            throw HNSWError.loadFailed("Stored padded dimension does not match dimensions")
        }

        try index.state.withLock {
            for _ in 0..<labelCount {
                guard offset + 8 + dimensions * 4 <= data.count else {
                    throw HNSWError.loadFailed("Swift backend TurboQuant data is truncated")
                }

                let label: UInt64 = data.readLittleEndian(at: &offset)
                let vectorOffset = $0.vectorStorage.count
                $0.vectorStorage.reserveCapacity($0.vectorStorage.count + dimensions)
                for _ in 0..<dimensions {
                    let bitPattern: UInt32 = data.readLittleEndian(at: &offset)
                    $0.vectorStorage.append(Float(bitPattern: bitPattern))
                }
                $0.entries[label] = Entry(offset: vectorOffset)
                $0.labelOrder.append(label)
            }

            guard offset == data.count else {
                throw HNSWError.loadFailed("Swift backend TurboQuant data has trailing bytes")
            }
            $0.finalized = true
        }

        return index
    }

    private func ensureQueryScratch(_ state: inout State) {
        guard state.queryScratch.count != dimensions else { return }
        state.queryScratch = [Float](repeating: 0, count: dimensions)
    }
}


// MARK: - Data Serialization Helpers

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }

    func readLittleEndian<T: FixedWidthInteger>(at offset: inout Int) -> T {
        let size = MemoryLayout<T>.size
        let value = withUnsafeBytes { buf in
            buf.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += size
        return T(littleEndian: value)
    }
}
