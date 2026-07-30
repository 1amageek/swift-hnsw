/// An immutable serialized HNSW index with owned contiguous storage.
public struct HNSWArchive: HNSWArchiveBytes, Sendable {
    /// The archive's canonical bytes.
    ///
    /// Array copy-on-write lets downstream immutable owners retain this storage
    /// without copying the payload.
    public let bytes: [UInt8]

    public init(bytes: [UInt8]) {
        self.bytes = bytes
    }

    public func withUnsafeBytes<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        try bytes.withUnsafeBytes(body)
    }
}
