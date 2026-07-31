/// Immutable graph topology captured from an in-memory HNSW index.
///
/// The snapshot intentionally contains topology and labels only. Vector storage remains
/// owned by the consumer so a specialized distance backend can replace it without weakening
/// the HNSW graph invariants. The flat connection store preserves the production search
/// layout and is shared copy-on-write until either owner mutates it.
struct HNSWGraphSnapshot: Sendable {
    let labelOrder: [UInt64]
    let deletedFlags: [UInt8]
    let levels: [Int]
    let connections: HNSWConnectionStore
    let entryPoint: HNSWInternalID?
    let maxLevel: Int
}
