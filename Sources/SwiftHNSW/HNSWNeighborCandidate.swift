struct HNSWNeighborCandidate: Sendable {
    let internalID: Int
    let label: UInt64
    let distance: Float
}

@inline(__always)
func isCloserHNSWCandidate(_ lhs: HNSWNeighborCandidate, than rhs: HNSWNeighborCandidate) -> Bool {
    if lhs.distance == rhs.distance {
        return lhs.label < rhs.label
    }
    return lhs.distance < rhs.distance
}

@inline(__always)
func isFartherHNSWCandidate(_ lhs: HNSWNeighborCandidate, than rhs: HNSWNeighborCandidate) -> Bool {
    if lhs.distance == rhs.distance {
        return lhs.label > rhs.label
    }
    return lhs.distance > rhs.distance
}
