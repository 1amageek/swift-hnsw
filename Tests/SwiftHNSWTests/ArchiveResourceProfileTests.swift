import Testing
@testable import SwiftHNSW

@Suite("HNSW Archive Resource Profile")
struct ArchiveResourceProfileTests {
    @Test("Profile inspects an owned archive without restoring it")
    func profileInspectsArchive() throws {
        let index = try HNSWIndexF32(
            dimensions: 2,
            maxElements: 4,
            metric: .l2,
            configuration: HNSWConfiguration(
                m: 2,
                efConstruction: 4,
                efSearch: 4,
                randomSeed: 7,
                allowReplaceDeleted: true
            )
        )
        try index.add([1, 0], label: 11)
        try index.add([0, 1], label: 12)
        try index.markDeleted(label: 12)
        let archive = try index.serializedArchive()

        let profile = try HNSWIndexF32.inspectArchiveResourceProfile(
            from: archive,
            dimensions: 2,
            metric: .l2
        )

        #expect(profile.archiveByteCount == archive.bytes.count)
        #expect(profile.storedCapacity == 4)
        #expect(profile.labelCount == 2)
        #expect(profile.activeLabelCount == 1)
        #expect(profile.estimatedRetainedPayloadByteCount > 0)
        #expect(
            profile.estimatedRestoreWorkingPayloadByteCount
                >= profile.estimatedRetainedPayloadByteCount
                    + profile.archiveByteCount
        )
    }

    @Test("Profile rejects truncated and mismatched archives")
    func profileRejectsInvalidArchive() throws {
        let index = try HNSWIndexF32(
            dimensions: 2,
            maxElements: 2
        )
        try index.add([1, 0], label: 1)
        let archive = try index.serializedArchive()
        let truncated = HNSWArchive(bytes: Array(archive.bytes.dropLast()))

        #expect(throws: HNSWError.self) {
            _ = try HNSWIndexF32.inspectArchiveResourceProfile(
                from: truncated,
                dimensions: 2
            )
        }
        #expect(throws: HNSWError.self) {
            _ = try HNSWIndexF32.inspectArchiveResourceProfile(
                from: archive,
                dimensions: 3
            )
        }
    }
}
