/// Resource facts derived from a structurally inspected HNSW graph archive without
/// materializing its vectors or connection lists.
///
/// The byte estimates cover payload storage requested by SwiftHNSW's known
/// arrays, dictionaries, graph storage, and search workspaces. They include
/// conservative collection allowances but do not measure allocator metadata,
/// allocator capacity rounding, or unrelated process memory.
public struct HNSWArchiveResourceProfile: Sendable, Hashable {
    public let archiveByteCount: Int
    public let storedCapacity: Int
    public let labelCount: Int
    public let activeLabelCount: Int
    public let upperLevelSlotCount: Int
    public let estimatedRetainedPayloadByteCount: Int
    public let estimatedRestoreWorkingPayloadByteCount: Int

    init(
        archiveByteCount: Int,
        storedCapacity: Int,
        labelCount: Int,
        activeLabelCount: Int,
        upperLevelSlotCount: Int,
        estimatedRetainedPayloadByteCount: Int,
        estimatedRestoreWorkingPayloadByteCount: Int
    ) {
        self.archiveByteCount = archiveByteCount
        self.storedCapacity = storedCapacity
        self.labelCount = labelCount
        self.activeLabelCount = activeLabelCount
        self.upperLevelSlotCount = upperLevelSlotCount
        self.estimatedRetainedPayloadByteCount =
            estimatedRetainedPayloadByteCount
        self.estimatedRestoreWorkingPayloadByteCount =
            estimatedRestoreWorkingPayloadByteCount
    }
}
