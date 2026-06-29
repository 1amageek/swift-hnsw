struct HNSWFarthestCandidateHeap: Sendable {
    private var elements: [HNSWNeighborCandidate] = []

    var count: Int {
        elements.count
    }

    var isEmpty: Bool {
        elements.isEmpty
    }

    var peek: HNSWNeighborCandidate? {
        elements.first
    }

    mutating func reserveCapacity(_ minimumCapacity: Int) {
        elements.reserveCapacity(minimumCapacity)
    }

    @inline(__always)
    mutating func push(_ element: HNSWNeighborCandidate) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    @inline(__always)
    mutating func replaceTop(with element: HNSWNeighborCandidate) {
        guard !elements.isEmpty else {
            push(element)
            return
        }
        elements[0] = element
        siftDown(from: 0)
    }

    @discardableResult
    @inline(__always)
    mutating func pop() -> HNSWNeighborCandidate? {
        guard !elements.isEmpty else { return nil }
        return popUnchecked()
    }

    @inline(__always)
    mutating func popUnchecked() -> HNSWNeighborCandidate {
        precondition(!elements.isEmpty, "Heap must not be empty")
        guard elements.count > 1 else {
            return elements.removeLast()
        }

        let value = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        return value
    }

    func unorderedElements() -> [HNSWNeighborCandidate] {
        elements
    }

    func closestSorted(limit: Int) -> [HNSWNeighborCandidate] {
        guard limit > 0 else { return [] }
        guard elements.count > limit else {
            return elements.sorted(by: isCloserHNSWCandidate)
        }

        var topK = HNSWFarthestCandidateHeap()
        topK.reserveCapacity(limit)
        for element in elements {
            if topK.count < limit {
                topK.push(element)
            } else if let farthest = topK.peek, isCloserHNSWCandidate(element, than: farthest) {
                topK.replaceTop(with: element)
            }
        }
        return topK.unorderedElements().sorted(by: isCloserHNSWCandidate)
    }

    @inline(__always)
    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = parentIndex(of: child)
        while child > 0, isFartherHNSWCandidate(elements[child], than: elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = parentIndex(of: child)
        }
    }

    @inline(__always)
    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left = leftChildIndex(of: parent)
            let right = rightChildIndex(of: parent)
            var candidate = parent

            if left < elements.count, isFartherHNSWCandidate(elements[left], than: elements[candidate]) {
                candidate = left
            }
            if right < elements.count, isFartherHNSWCandidate(elements[right], than: elements[candidate]) {
                candidate = right
            }
            guard candidate != parent else { return }

            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }

    @inline(__always)
    private func parentIndex(of index: Int) -> Int {
        (index - 1) / 2
    }

    @inline(__always)
    private func leftChildIndex(of index: Int) -> Int {
        index * 2 + 1
    }

    @inline(__always)
    private func rightChildIndex(of index: Int) -> Int {
        index * 2 + 2
    }
}
