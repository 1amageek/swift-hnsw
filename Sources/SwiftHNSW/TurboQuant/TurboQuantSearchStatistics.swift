/// Benchmark-only counters for Product QJL candidate scoring.
@_spi(Benchmarking)
public struct TurboQuantQJLPruningStatistics: Sendable, Equatable {
    public let scoredCandidates: Int
    public let prunedCandidates: Int

    public init(scoredCandidates: Int, prunedCandidates: Int) {
        self.scoredCandidates = scoredCandidates
        self.prunedCandidates = prunedCandidates
    }

    public var pruningRate: Double {
        guard scoredCandidates > 0 else { return 0 }
        return Double(prunedCandidates) / Double(scoredCandidates)
    }
}
