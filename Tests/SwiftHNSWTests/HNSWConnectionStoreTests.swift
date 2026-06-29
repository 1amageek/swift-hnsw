#if !HNSWLIB_BACKEND
import Testing
@testable import SwiftHNSW

@Suite("HNSW Connection Store")
struct HNSWConnectionStoreTests {

    @Test("Stores fixed-slot neighbors by node and level")
    func storesNeighborsByNodeAndLevel() {
        var store = HNSWConnectionStore(m: 2)
        store.appendNode(level: 1)
        store.append(10, for: 0, at: 0)
        store.append(11, for: 0, at: 0)
        store.append(20, for: 0, at: 1)

        #expect(store.nodeCount == 1)
        #expect(store.levelCount(for: 0) == 2)
        #expect(store.neighborCount(for: 0, at: 0) == 2)
        #expect(store.neighbor(at: 0, for: 0, at: 0) == 10)
        #expect(store.neighbor(at: 1, for: 0, at: 0) == 11)
        #expect(store.neighbor(at: 0, for: 0, at: 1) == 20)
        #expect(store.levels(for: 0) == [[10, 11], [20]])
    }

    @Test("Replaces and resets node connections")
    func replacesAndResetsNodeConnections() {
        var store = HNSWConnectionStore(m: 2)
        store.appendNode(level: 1)
        store.replaceNeighbors([1, 2, 3], for: 0, at: 0)
        store.replaceNeighbors([4], for: 0, at: 1)

        #expect(store.levels(for: 0) == [[1, 2, 3], [4]])

        store.resetNode(0, level: 0)

        #expect(store.levelCount(for: 0) == 1)
        #expect(store.neighborCount(for: 0, at: 0) == 0)
        #expect(store.levels(for: 0) == [[]])
    }

    @Test("Extending a node preserves existing lower levels")
    func extendingNodePreservesExistingLowerLevels() {
        var store = HNSWConnectionStore(m: 2)
        store.appendNode(level: 0)
        store.replaceNeighbors([1, 2], for: 0, at: 0)

        store.ensureNode(0, through: 2)

        #expect(store.levelCount(for: 0) == 3)
        #expect(store.levels(for: 0) == [[1, 2], [], []])
    }

    @Test("Replaces neighbors directly from candidates with exclusion and capacity clamp")
    func replacesNeighborsFromCandidates() {
        var store = HNSWConnectionStore(m: 1)
        store.appendNode(level: 1)
        let candidates = [
            HNSWNeighborCandidate(internalID: 1, distance: 0.1),
            HNSWNeighborCandidate(internalID: 2, distance: 0.2),
            HNSWNeighborCandidate(internalID: 3, distance: 0.3),
            HNSWNeighborCandidate(internalID: 4, distance: 0.4),
        ]

        store.replaceNeighbors(from: candidates, excluding: nil, for: 0, at: 0)
        store.replaceNeighbors(from: candidates, excluding: 2, for: 0, at: 1)

        #expect(store.levels(for: 0) == [[1, 2, 3], [1, 3]])
    }
}
#endif
