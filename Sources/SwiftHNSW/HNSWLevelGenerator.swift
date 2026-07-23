#if os(WASI)
import WASILibc
#else
import Foundation
#endif

struct HNSWLevelGenerator: Sendable {
    private var state: UInt64

    init(seed: Int) {
        let normalizedSeed = UInt64(bitPattern: Int64(seed))
        self.state = normalizedSeed == 0 ? 0x9E37_79B9_7F4A_7C15 : normalizedSeed
    }

    init(state: UInt64) {
        self.state = state == 0 ? 0x9E37_79B9_7F4A_7C15 : state
    }

    var currentState: UInt64 {
        state
    }

    mutating func randomLevel(multiplier: Double) -> Int {
        let unit = max(nextUnitInterval(), Double.leastNonzeroMagnitude)
        return Int(-hnswNaturalLog(unit) * multiplier)
    }

    private mutating func nextUnitInterval() -> Double {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let value = state >> 11
        return Double(value) / Double(UInt64(1) << 53)
    }
}

@inline(__always)
func hnswNaturalLog(_ value: Double) -> Double {
    #if os(WASI)
    WASILibc.log(value)
    #else
    Foundation.log(value)
    #endif
}
