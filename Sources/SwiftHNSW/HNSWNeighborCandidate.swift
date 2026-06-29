struct HNSWNeighborCandidate: Sendable {
    let internalID: Int
    let distance: Float
}

@inline(__always)
func isCloserHNSWCandidate(_ lhs: HNSWNeighborCandidate, than rhs: HNSWNeighborCandidate) -> Bool {
    if lhs.distance == rhs.distance {
        return lhs.internalID < rhs.internalID
    }
    return lhs.distance < rhs.distance
}

@inline(__always)
func isFartherHNSWCandidate(_ lhs: HNSWNeighborCandidate, than rhs: HNSWNeighborCandidate) -> Bool {
    if lhs.distance == rhs.distance {
        return lhs.internalID > rhs.internalID
    }
    return lhs.distance > rhs.distance
}
