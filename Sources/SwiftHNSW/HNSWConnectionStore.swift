struct HNSWConnectionStore: Sendable {
    private var nodeLevelCounts: [Int]
    private var upperLevelStarts: [Int]
    private var upperLevelCounts: [Int]
    private var upperLevelOffsets: [Int]
    private var upperNeighborCounts: [Int]
    private var upperNeighbors: [HNSWInternalID]
    private var level0NeighborCounts: [Int]
    private var level0Neighbors: [HNSWInternalID]
    private let m: Int
    private let level0Capacity: Int
    private let upperLevelCapacity: Int

    init(m: Int) {
        self.nodeLevelCounts = []
        self.upperLevelStarts = []
        self.upperLevelCounts = []
        self.upperLevelOffsets = []
        self.upperNeighborCounts = []
        self.upperNeighbors = []
        self.level0NeighborCounts = []
        self.level0Neighbors = []
        self.m = m
        self.level0Capacity = max(1, m * 2 + 1)
        self.upperLevelCapacity = max(1, m + 1)
    }

    var nodeCount: Int {
        nodeLevelCounts.count
    }

    var isEmpty: Bool {
        nodeLevelCounts.isEmpty
    }

    mutating func reserveCapacity(_ nodeCount: Int) {
        nodeLevelCounts.reserveCapacity(nodeCount)
        upperLevelStarts.reserveCapacity(nodeCount)
        upperLevelCounts.reserveCapacity(nodeCount)
        level0NeighborCounts.reserveCapacity(nodeCount)
        if nodeCount <= Int.max / level0Capacity {
            level0Neighbors.reserveCapacity(nodeCount * level0Capacity)
        }
    }

    func levelCount(for internalID: HNSWInternalID) -> Int {
        let nodeIndex = Int(internalID)
        guard isValidNode(nodeIndex) else { return 0 }
        return nodeLevelCounts[nodeIndex]
    }

    func hasAnyConnection(for internalID: HNSWInternalID) -> Bool {
        let nodeIndex = Int(internalID)
        guard isValidNode(nodeIndex) else { return false }
        if level0NeighborCounts[nodeIndex] > 0 {
            return true
        }
        let start = upperLevelStarts[nodeIndex]
        let upperCount = upperLevelCounts[nodeIndex]
        guard upperCount > 0 else { return false }
        for slot in start..<(start + upperCount) where upperNeighborCounts[slot] > 0 {
            return true
        }
        return false
    }

    func neighborCount(for internalID: HNSWInternalID, at level: Int) -> Int {
        let nodeIndex = Int(internalID)
        guard isValidNode(nodeIndex), level >= 0, level < nodeLevelCounts[nodeIndex] else { return 0 }
        if level == 0 {
            return level0NeighborCounts[nodeIndex]
        }
        guard let slot = upperSlot(forNodeIndex: nodeIndex, at: level) else { return 0 }
        return upperNeighborCounts[slot]
    }

    @inline(__always)
    func neighborStorageRange(for internalID: HNSWInternalID, at level: Int) -> Range<Int> {
        let nodeIndex = Int(internalID)
        guard isValidNode(nodeIndex), level >= 0, level < nodeLevelCounts[nodeIndex] else { return 0..<0 }
        if level == 0 {
            let offset = nodeIndex * level0Capacity
            return offset..<(offset + level0NeighborCounts[nodeIndex])
        }
        guard let slot = upperSlot(forNodeIndex: nodeIndex, at: level) else { return 0..<0 }
        let offset = upperLevelOffsets[slot]
        return offset..<(offset + upperNeighborCounts[slot])
    }

    @inline(__always)
    func neighborInStorage(at storageIndex: Int, level: Int) -> HNSWInternalID {
        if level == 0 {
            return level0Neighbors[storageIndex]
        }
        return upperNeighbors[storageIndex]
    }

    func neighbor(at neighborIndex: Int, for internalID: HNSWInternalID, at level: Int) -> HNSWInternalID? {
        let nodeIndex = Int(internalID)
        guard isValidNode(nodeIndex),
              level >= 0,
              level < nodeLevelCounts[nodeIndex],
              neighborIndex >= 0 else {
            return nil
        }
        if level == 0 {
            guard neighborIndex < level0NeighborCounts[nodeIndex] else { return nil }
            return level0Neighbors[nodeIndex * level0Capacity + neighborIndex]
        }
        guard let slot = upperSlot(forNodeIndex: nodeIndex, at: level),
              neighborIndex < upperNeighborCounts[slot] else {
            return nil
        }
        return upperNeighbors[upperLevelOffsets[slot] + neighborIndex]
    }

    func contains(_ neighborID: HNSWInternalID, for internalID: HNSWInternalID, at level: Int) -> Bool {
        let range = neighborStorageRange(for: internalID, at: level)
        guard !range.isEmpty else { return false }
        if level == 0 {
            for index in range where level0Neighbors[index] == neighborID {
                return true
            }
            return false
        }
        for index in range where upperNeighbors[index] == neighborID {
            return true
        }
        return false
    }

    mutating func appendNode(level: Int) {
        appendNode(levelCount: max(0, level) + 1)
    }

    mutating func resetNode(_ internalID: HNSWInternalID, level: Int) {
        let nodeIndex = Int(internalID)
        ensureNode(internalID, through: level)
        let newLevelCount = max(0, level) + 1
        let oldLevelCount = nodeLevelCounts[nodeIndex]
        nodeLevelCounts[nodeIndex] = newLevelCount
        level0NeighborCounts[nodeIndex] = 0
        if newLevelCount <= oldLevelCount {
            let start = upperLevelStarts[nodeIndex]
            let oldUpperCount = upperLevelCounts[nodeIndex]
            let newUpperCount = max(0, newLevelCount - 1)
            upperLevelCounts[nodeIndex] = newUpperCount
            for slot in start..<(start + oldUpperCount) {
                upperNeighborCounts[slot] = 0
            }
            return
        }
        rebuildNodeUpperSlots(nodeIndex: nodeIndex, levelCount: newLevelCount, preserving: [])
    }

    mutating func ensureNode(_ internalID: HNSWInternalID, through level: Int) {
        let nodeIndex = Int(internalID)
        while nodeLevelCounts.count <= nodeIndex {
            appendNode(levelCount: 0)
        }
        ensureLevel(nodeIndex: nodeIndex, level: level)
    }

    mutating func append(_ neighborID: HNSWInternalID, for internalID: HNSWInternalID, at level: Int) {
        ensureNode(internalID, through: level)
        let nodeIndex = Int(internalID)
        if level == 0 {
            let count = level0NeighborCounts[nodeIndex]
            guard count < level0Capacity else { return }
            level0Neighbors[nodeIndex * level0Capacity + count] = neighborID
            level0NeighborCounts[nodeIndex] = count + 1
            return
        }
        guard let slot = upperSlot(forNodeIndex: nodeIndex, at: level) else { return }
        let count = upperNeighborCounts[slot]
        guard count < upperLevelCapacity else { return }
        upperNeighbors[upperLevelOffsets[slot] + count] = neighborID
        upperNeighborCounts[slot] = count + 1
    }

    mutating func replaceNeighbors(
        _ newNeighbors: [HNSWInternalID],
        for internalID: HNSWInternalID,
        at level: Int
    ) {
        ensureNode(internalID, through: level)
        let nodeIndex = Int(internalID)
        if level == 0 {
            let count = min(newNeighbors.count, level0Capacity)
            let offset = nodeIndex * level0Capacity
            level0NeighborCounts[nodeIndex] = count
            for index in 0..<count {
                level0Neighbors[offset + index] = newNeighbors[index]
            }
            return
        }
        guard let slot = upperSlot(forNodeIndex: nodeIndex, at: level) else { return }
        let count = min(newNeighbors.count, upperLevelCapacity)
        let offset = upperLevelOffsets[slot]
        upperNeighborCounts[slot] = count
        for index in 0..<count {
            upperNeighbors[offset + index] = newNeighbors[index]
        }
    }

    mutating func replaceNeighbors(
        from candidates: [HNSWNeighborCandidate],
        excluding excludedID: HNSWInternalID?,
        for internalID: HNSWInternalID,
        at level: Int
    ) {
        ensureNode(internalID, through: level)
        let nodeIndex = Int(internalID)
        let capacity = capacityForLevel(level)
        let offset: Int
        if level == 0 {
            offset = nodeIndex * level0Capacity
        } else {
            guard let slot = upperSlot(forNodeIndex: nodeIndex, at: level) else { return }
            offset = upperLevelOffsets[slot]
        }
        var count = 0
        for candidate in candidates {
            let neighborID = candidate.internalID
            if let excludedID, neighborID == excludedID {
                continue
            }
            guard count < capacity else {
                break
            }
            if level == 0 {
                level0Neighbors[offset + count] = neighborID
            } else {
                upperNeighbors[offset + count] = neighborID
            }
            count += 1
        }
        if level == 0 {
            level0NeighborCounts[nodeIndex] = count
        } else if let slot = upperSlot(forNodeIndex: nodeIndex, at: level) {
            upperNeighborCounts[slot] = count
        }
    }

    func levels(for internalID: HNSWInternalID) -> [[HNSWInternalID]] {
        let nodeIndex = Int(internalID)
        guard isValidNode(nodeIndex) else { return [] }
        let levelCount = nodeLevelCounts[nodeIndex]
        var result: [[HNSWInternalID]] = []
        result.reserveCapacity(levelCount)
        if levelCount > 0 {
            let offset = nodeIndex * level0Capacity
            let count = level0NeighborCounts[nodeIndex]
            result.append(Array(level0Neighbors[offset..<(offset + count)]))
        }
        guard levelCount > 1 else { return result }
        let start = upperLevelStarts[nodeIndex]
        for upperLevel in 0..<upperLevelCounts[nodeIndex] {
            let slot = start + upperLevel
            let offset = upperLevelOffsets[slot]
            let count = upperNeighborCounts[slot]
            result.append(Array(upperNeighbors[offset..<(offset + count)]))
        }
        return result
    }

    mutating func replaceAll(with nestedConnections: [[[Int]]]) {
        nodeLevelCounts.removeAll(keepingCapacity: true)
        upperLevelStarts.removeAll(keepingCapacity: true)
        upperLevelCounts.removeAll(keepingCapacity: true)
        upperLevelOffsets.removeAll(keepingCapacity: true)
        upperNeighborCounts.removeAll(keepingCapacity: true)
        upperNeighbors.removeAll(keepingCapacity: true)
        level0NeighborCounts.removeAll(keepingCapacity: true)
        level0Neighbors.removeAll(keepingCapacity: true)
        reserveCapacity(nestedConnections.count)

        for levels in nestedConnections {
            appendNode(levelCount: levels.count)
            let internalID = HNSWInternalID(nodeLevelCounts.count - 1)
            for (level, levelNeighbors) in levels.enumerated() {
                replaceNeighbors(
                    levelNeighbors.prefix(capacityForLevel(level)).map(HNSWInternalID.init),
                    for: internalID,
                    at: level
                )
            }
        }
    }

    private mutating func appendNode(levelCount: Int) {
        nodeLevelCounts.append(levelCount)
        level0NeighborCounts.append(0)
        level0Neighbors.append(contentsOf: repeatElement(0, count: level0Capacity))
        let upperCount = max(0, levelCount - 1)
        upperLevelStarts.append(upperLevelOffsets.count)
        upperLevelCounts.append(upperCount)
        appendUpperLevelSlots(count: upperCount)
    }

    private mutating func ensureLevel(nodeIndex: Int, level: Int) {
        guard level >= 0 else { return }
        let currentCount = nodeLevelCounts[nodeIndex]
        guard currentCount <= level else { return }
        let existingLevels = levels(for: HNSWInternalID(nodeIndex))
        rebuildNodeUpperSlots(nodeIndex: nodeIndex, levelCount: level + 1, preserving: existingLevels)
    }

    private mutating func rebuildNodeUpperSlots(
        nodeIndex: Int,
        levelCount: Int,
        preserving existingLevels: [[HNSWInternalID]]
    ) {
        nodeLevelCounts[nodeIndex] = levelCount
        let upperCount = max(0, levelCount - 1)
        upperLevelStarts[nodeIndex] = upperLevelOffsets.count
        upperLevelCounts[nodeIndex] = upperCount
        appendUpperLevelSlots(count: upperCount)
        let internalID = HNSWInternalID(nodeIndex)
        for (level, neighbors) in existingLevels.enumerated() where level < levelCount {
            replaceNeighbors(neighbors, for: internalID, at: level)
        }
    }

    private mutating func appendUpperLevelSlots(count: Int) {
        guard count > 0 else { return }
        for _ in 0..<count {
            let offset = upperNeighbors.count
            upperLevelOffsets.append(offset)
            upperNeighborCounts.append(0)
            upperNeighbors.append(contentsOf: repeatElement(0, count: upperLevelCapacity))
        }
    }

    private func upperSlot(forNodeIndex nodeIndex: Int, at level: Int) -> Int? {
        guard isValidNode(nodeIndex), level > 0, level < nodeLevelCounts[nodeIndex] else {
            return nil
        }
        return upperLevelStarts[nodeIndex] + level - 1
    }

    private func isValidNode(_ nodeIndex: Int) -> Bool {
        nodeIndex >= 0 && nodeIndex < nodeLevelCounts.count
    }

    private func capacityForLevel(_ level: Int) -> Int {
        if level == 0 {
            return level0Capacity
        }
        return upperLevelCapacity
    }
}
