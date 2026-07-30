/// A stable contiguous byte source containing one HNSW archive.
///
/// Implementations retain the underlying storage for the complete synchronous
/// borrow. The pointer passed to `body` must not escape the closure.
public protocol HNSWArchiveBytes: Sendable {
    func withUnsafeBytes<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result
}
