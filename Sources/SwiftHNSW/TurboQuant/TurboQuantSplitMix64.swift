/// SplitMix64 is part of the on-disk TurboQuant contract.
///
/// Changing this generator changes every rotation and QJL projection produced from a seed.
struct TurboQuantSplitMix64: Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func nextGaussianPair() -> (Double, Double) {
        while true {
            let first = 2 * nextUnitInterval() - 1
            let second = 2 * nextUnitInterval() - 1
            let radiusSquared = first * first + second * second
            guard radiusSquared > 0, radiusSquared < 1 else { continue }
            let multiplier = (-2 * hnswNaturalLog(radiusSquared) / radiusSquared)
                .squareRoot()
            return (first * multiplier, second * multiplier)
        }
    }

    @inline(__always)
    private mutating func nextUnitInterval() -> Double {
        let value = next() >> 11
        return Double(value) * (1.0 / 9_007_199_254_740_992.0)
    }
}
