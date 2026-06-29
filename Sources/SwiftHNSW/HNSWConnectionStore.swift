struct HNSWConnectionStore: Sendable {
    private var nodeLevelStarts: [Int]
    private var nodeLevelCounts: [Int]
    private var levelOffsets: [Int]
    private var levelNeighborCounts: [Int]
    private var neighbors: [Int]
    private let m: Int

    init(m: Int) {
        self.nodeLevelStarts = []
        self.nodeLevelCounts = []
        self.levelOffsets = []
        self.levelNeighborCounts = []
        self.neighbors = []
        self.m = m
    }

    var nodeCount: Int {
        nodeLevelCounts.count
    }

    var isEmpty: Bool {
        nodeLevelCounts.isEmpty
    }

    mutating func reserveCapacity(_ nodeCount: Int) {
        nodeLevelStarts.reserveCapacity(nodeCount)
        nodeLevelCounts.reserveCapacity(nodeCount)
    }

    func levelCount(for internalID: Int) -> Int {
        guard isValidNode(internalID) else { return 0 }
        return nodeLevelCounts[internalID]
    }

    func hasAnyConnection(for internalID: Int) -> Bool {
        guard isValidNode(internalID) else { return false }
        let start = nodeLevelStarts[internalID]
        let levelCount = nodeLevelCounts[internalID]
        guard levelCount > 0 else { return false }
        for slot in start..<(start + levelCount) where levelNeighborCounts[slot] > 0 {
            return true
        }
        return false
    }

    func neighborCount(for internalID: Int, at level: Int) -> Int {
        guard let slot = slot(for: internalID, at: level) else { return 0 }
        return levelNeighborCounts[slot]
    }

    @inline(__always)
    func neighborStorageRange(for internalID: Int, at level: Int) -> Range<Int> {
        guard let slot = slot(for: internalID, at: level) else { return 0..<0 }
        let offset = levelOffsets[slot]
        return offset..<(offset + levelNeighborCounts[slot])
    }

    @inline(__always)
    func neighborInStorage(at storageIndex: Int) -> Int {
        neighbors[storageIndex]
    }

    func neighbor(at neighborIndex: Int, for internalID: Int, at level: Int) -> Int? {
        guard let slot = slot(for: internalID, at: level),
              neighborIndex >= 0,
              neighborIndex < levelNeighborCounts[slot] else {
            return nil
        }
        return neighbors[levelOffsets[slot] + neighborIndex]
    }

    func contains(_ neighborID: Int, for internalID: Int, at level: Int) -> Bool {
        guard let slot = slot(for: internalID, at: level) else { return false }
        let offset = levelOffsets[slot]
        let count = levelNeighborCounts[slot]
        guard count > 0 else { return false }
        for index in offset..<(offset + count) where neighbors[index] == neighborID {
            return true
        }
        return false
    }

    mutating func appendNode(level: Int) {
        appendNode(levelCount: max(0, level) + 1)
    }

    mutating func resetNode(_ internalID: Int, level: Int) {
        ensureNode(internalID, through: level)
        let newLevelCount = max(0, level) + 1
        let oldLevelCount = nodeLevelCounts[internalID]
        if newLevelCount <= oldLevelCount {
            nodeLevelCounts[internalID] = newLevelCount
            let start = nodeLevelStarts[internalID]
            for slot in start..<(start + oldLevelCount) {
                levelNeighborCounts[slot] = 0
            }
            return
        }

        let start = levelOffsets.count
        nodeLevelStarts[internalID] = start
        nodeLevelCounts[internalID] = newLevelCount
        appendLevelSlots(levels: 0..<newLevelCount)
    }

    mutating func ensureNode(_ internalID: Int, through level: Int) {
        while nodeLevelCounts.count <= internalID {
            appendNode(levelCount: 0)
        }
        ensureLevel(internalID: internalID, level: level)
    }

    mutating func append(_ neighborID: Int, for internalID: Int, at level: Int) {
        ensureNode(internalID, through: level)
        guard let slot = slot(for: internalID, at: level) else { return }
        let count = levelNeighborCounts[slot]
        let offset = levelOffsets[slot]
        let capacity = capacityForLevel(level)
        guard count < capacity else { return }
        neighbors[offset + count] = neighborID
        levelNeighborCounts[slot] = count + 1
    }

    mutating func replaceNeighbors(_ newNeighbors: [Int], for internalID: Int, at level: Int) {
        ensureNode(internalID, through: level)
        guard let slot = slot(for: internalID, at: level) else { return }
        let capacity = capacityForLevel(level)
        let count = min(newNeighbors.count, capacity)
        let offset = levelOffsets[slot]
        levelNeighborCounts[slot] = count
        for index in 0..<count {
            neighbors[offset + index] = newNeighbors[index]
        }
    }

    mutating func replaceNeighbors(
        from candidates: [HNSWNeighborCandidate],
        excluding excludedID: Int?,
        for internalID: Int,
        at level: Int
    ) {
        ensureNode(internalID, through: level)
        guard let slot = slot(for: internalID, at: level) else { return }
        let capacity = capacityForLevel(level)
        let offset = levelOffsets[slot]
        var count = 0
        for candidate in candidates {
            let neighborID = candidate.internalID
            if let excludedID, neighborID == excludedID {
                continue
            }
            guard count < capacity else {
                break
            }
            neighbors[offset + count] = neighborID
            count += 1
        }
        levelNeighborCounts[slot] = count
    }

    func levels(for internalID: Int) -> [[Int]] {
        guard isValidNode(internalID) else { return [] }
        let start = nodeLevelStarts[internalID]
        let levelCount = nodeLevelCounts[internalID]
        var result: [[Int]] = []
        result.reserveCapacity(levelCount)
        for level in 0..<levelCount {
            let slot = start + level
            let offset = levelOffsets[slot]
            let count = levelNeighborCounts[slot]
            result.append(Array(neighbors[offset..<(offset + count)]))
        }
        return result
    }

    mutating func replaceAll(with nestedConnections: [[[Int]]]) {
        nodeLevelStarts.removeAll(keepingCapacity: true)
        nodeLevelCounts.removeAll(keepingCapacity: true)
        levelOffsets.removeAll(keepingCapacity: true)
        levelNeighborCounts.removeAll(keepingCapacity: true)
        neighbors.removeAll(keepingCapacity: true)
        reserveCapacity(nestedConnections.count)

        for levels in nestedConnections {
            let start = levelOffsets.count
            nodeLevelStarts.append(start)
            nodeLevelCounts.append(levels.count)
            for (level, levelNeighbors) in levels.enumerated() {
                let offset = neighbors.count
                let capacity = capacityForLevel(level)
                levelOffsets.append(offset)
                levelNeighborCounts.append(min(levelNeighbors.count, capacity))
                neighbors.append(contentsOf: levelNeighbors.prefix(capacity))
                if levelNeighbors.count < capacity {
                    neighbors.append(contentsOf: repeatElement(0, count: capacity - levelNeighbors.count))
                }
            }
        }
    }

    private mutating func appendNode(levelCount: Int) {
        let start = levelOffsets.count
        nodeLevelStarts.append(start)
        nodeLevelCounts.append(levelCount)
        appendLevelSlots(levels: 0..<levelCount)
    }

    private mutating func ensureLevel(internalID: Int, level: Int) {
        guard level >= 0 else { return }
        let currentCount = nodeLevelCounts[internalID]
        guard currentCount <= level else { return }
        let existingLevels = levels(for: internalID)
        let newLevelCount = level + 1
        let start = levelOffsets.count
        nodeLevelStarts[internalID] = start
        nodeLevelCounts[internalID] = newLevelCount
        appendLevelSlots(levels: 0..<newLevelCount)
        for (existingLevel, existingNeighbors) in existingLevels.enumerated() {
            replaceNeighbors(existingNeighbors, for: internalID, at: existingLevel)
        }
    }

    private mutating func appendLevelSlots(levels: Range<Int>) {
        for level in levels {
            let offset = neighbors.count
            let capacity = capacityForLevel(level)
            levelOffsets.append(offset)
            levelNeighborCounts.append(0)
            neighbors.append(contentsOf: repeatElement(0, count: capacity))
        }
    }

    private func slot(for internalID: Int, at level: Int) -> Int? {
        guard isValidNode(internalID), level >= 0, level < nodeLevelCounts[internalID] else {
            return nil
        }
        return nodeLevelStarts[internalID] + level
    }

    private func isValidNode(_ internalID: Int) -> Bool {
        internalID >= 0 && internalID < nodeLevelCounts.count
    }

    private func capacityForLevel(_ level: Int) -> Int {
        if level == 0 {
            return max(1, m * 2 + 1)
        }
        return max(1, m + 1)
    }
}
