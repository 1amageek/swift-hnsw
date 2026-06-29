struct HNSWSearchScratch: Sendable {
    private static let placeholder = HNSWNeighborCandidate(internalID: 0, distance: 0)

    private var candidateQueueStorage: [HNSWNeighborCandidate] = []
    private var nearestCandidateStorage: [HNSWNeighborCandidate] = []
    private var resultCandidateStorage: [HNSWNeighborCandidate] = []

    mutating func withCandidateBuffers<R>(
        candidateCapacity: Int,
        nearestCapacity: Int,
        resultCapacity: Int = 1,
        _ body: (
            UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
            UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
            UnsafeMutableBufferPointer<HNSWNeighborCandidate>
        ) -> R
    ) -> R {
        Self.ensureCount(&candidateQueueStorage, count: max(1, candidateCapacity))
        Self.ensureCount(&nearestCandidateStorage, count: max(1, nearestCapacity))
        Self.ensureCount(&resultCandidateStorage, count: max(1, resultCapacity))

        return candidateQueueStorage.withUnsafeMutableBufferPointer { candidateQueue in
            nearestCandidateStorage.withUnsafeMutableBufferPointer { nearestCandidates in
                resultCandidateStorage.withUnsafeMutableBufferPointer { resultCandidates in
                    body(candidateQueue, nearestCandidates, resultCandidates)
                }
            }
        }
    }

    private static func ensureCount(
        _ storage: inout [HNSWNeighborCandidate],
        count: Int
    ) {
        guard storage.count < count else { return }
        storage = [HNSWNeighborCandidate](repeating: Self.placeholder, count: count)
    }
}
