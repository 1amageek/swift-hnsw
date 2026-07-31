struct HNSWSearchWorkspace: Sendable {
    private static let emptyCandidateSlot = HNSWNeighborCandidate(internalID: 0, distance: 0)
    static let maximumRetainedByteCount = 256 * 1_024 * 1_024

    private var candidateQueueStorage: [HNSWNeighborCandidate] = []
    private var nearestCandidateStorage: [HNSWNeighborCandidate] = []
    private var resultCandidateStorage: [HNSWNeighborCandidate] = []

    var retainedCandidateCapacities: (
        candidate: Int,
        nearest: Int,
        result: Int
    ) {
        (
            candidateQueueStorage.count,
            nearestCandidateStorage.count,
            resultCandidateStorage.count
        )
    }

    static func validateRetainedCapacity(
        candidateCapacity: Int,
        nearestCapacity: Int,
        resultCapacity: Int,
        visitedCapacity: Int
    ) throws {
        guard let retainedByteCount = retainedByteCount(
            candidateCapacity: candidateCapacity,
            nearestCapacity: nearestCapacity,
            resultCapacity: resultCapacity,
            visitedCapacity: visitedCapacity
        ) else {
            throw HNSWError.invalidArgument(
                "search workspace size exceeds the supported integer range"
            )
        }
        guard retainedByteCount <= maximumRetainedByteCount else {
            throw HNSWError.invalidArgument(
                "search workspace exceeds the 256 MiB limit"
            )
        }
    }

    static func retainedByteCount(
        candidateCapacity: Int,
        nearestCapacity: Int,
        resultCapacity: Int,
        visitedCapacity: Int
    ) -> Int? {
        guard visitedCapacity >= 0 else {
            return nil
        }
        let capacities = [
            max(1, candidateCapacity),
            max(1, nearestCapacity),
            max(1, resultCapacity),
        ]
        var candidateCount = 0
        for capacity in capacities {
            guard capacity <= Int.max - candidateCount else {
                return nil
            }
            candidateCount += capacity
        }
        guard candidateCount <= Int.max / MemoryLayout<HNSWNeighborCandidate>.stride,
              visitedCapacity <= Int.max / MemoryLayout<UInt16>.stride else {
            return nil
        }
        let candidateBytes = candidateCount
            * MemoryLayout<HNSWNeighborCandidate>.stride
        let visitedBytes = visitedCapacity * MemoryLayout<UInt16>.stride
        guard candidateBytes <= Int.max - visitedBytes else {
            return nil
        }
        return candidateBytes + visitedBytes
    }

    mutating func withCandidateBuffers<R>(
        candidateCapacity: Int,
        nearestCapacity: Int,
        resultCapacity: Int = 1,
        visitedCapacity: Int,
        _ body: (
            UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
            UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
            UnsafeMutableBufferPointer<HNSWNeighborCandidate>
        ) -> R
    ) -> R {
        let requiredCounts = [
            max(1, candidateCapacity),
            max(1, nearestCapacity),
            max(1, resultCapacity),
        ]
        Self.ensureCounts(
            candidateQueueStorage: &candidateQueueStorage,
            nearestCandidateStorage: &nearestCandidateStorage,
            resultCandidateStorage: &resultCandidateStorage,
            requiredCounts: requiredCounts,
            visitedCapacity: visitedCapacity
        )

        return candidateQueueStorage.withUnsafeMutableBufferPointer { candidateQueue in
            nearestCandidateStorage.withUnsafeMutableBufferPointer { nearestCandidates in
                resultCandidateStorage.withUnsafeMutableBufferPointer { resultCandidates in
                    body(
                        UnsafeMutableBufferPointer(
                            rebasing: candidateQueue[..<requiredCounts[0]]
                        ),
                        UnsafeMutableBufferPointer(
                            rebasing: nearestCandidates[..<requiredCounts[1]]
                        ),
                        UnsafeMutableBufferPointer(
                            rebasing: resultCandidates[..<requiredCounts[2]]
                        )
                    )
                }
            }
        }
    }

    mutating func prepare(
        candidateCapacity: Int,
        nearestCapacity: Int,
        resultCapacity: Int,
        visitedCapacity: Int
    ) {
        Self.ensureCounts(
            candidateQueueStorage: &candidateQueueStorage,
            nearestCandidateStorage: &nearestCandidateStorage,
            resultCandidateStorage: &resultCandidateStorage,
            requiredCounts: [
                max(1, candidateCapacity),
                max(1, nearestCapacity),
                max(1, resultCapacity),
            ],
            visitedCapacity: visitedCapacity
        )
    }

    private static func ensureCounts(
        candidateQueueStorage: inout [HNSWNeighborCandidate],
        nearestCandidateStorage: inout [HNSWNeighborCandidate],
        resultCandidateStorage: inout [HNSWNeighborCandidate],
        requiredCounts: [Int],
        visitedCapacity: Int
    ) {
        var targetCounts = [
            max(candidateQueueStorage.count, requiredCounts[0]),
            max(nearestCandidateStorage.count, requiredCounts[1]),
            max(resultCandidateStorage.count, requiredCounts[2]),
        ]
        let retainedCounts = [
            candidateQueueStorage.count,
            nearestCandidateStorage.count,
            resultCandidateStorage.count,
        ]
        let visitedBytes = visitedCapacity * MemoryLayout<UInt16>.stride
        let candidateByteBudget = maximumRetainedByteCount - visitedBytes

        if totalCandidateBytes(targetCounts) > candidateByteBudget {
            targetCounts = requiredCounts
        }
        for index in targetCounts.indices {
            guard retainedCounts[index] < requiredCounts[index] else {
                continue
            }
            let retainedCount = retainedCounts[index]
            let growthCount: Int
            if retainedCount == 0 {
                growthCount = requiredCounts[index]
            } else if retainedCount <= Int.max / 2 {
                growthCount = max(requiredCounts[index], retainedCount * 2)
            } else {
                growthCount = requiredCounts[index]
            }
            var proposedCounts = targetCounts
            proposedCounts[index] = growthCount
            if totalCandidateBytes(proposedCounts) <= candidateByteBudget {
                targetCounts = proposedCounts
            }
        }

        ensureCount(&candidateQueueStorage, count: targetCounts[0])
        ensureCount(&nearestCandidateStorage, count: targetCounts[1])
        ensureCount(&resultCandidateStorage, count: targetCounts[2])
    }

    private static func totalCandidateBytes(_ counts: [Int]) -> Int {
        counts.reduce(0, +) * MemoryLayout<HNSWNeighborCandidate>.stride
    }

    private static func ensureCount(
        _ storage: inout [HNSWNeighborCandidate],
        count: Int
    ) {
        guard storage.count != count else { return }
        storage = [HNSWNeighborCandidate](
            repeating: Self.emptyCandidateSlot,
            count: count
        )
    }
}
