struct HNSWFixedNearestCandidateHeap {
    private var storage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>
    private(set) var count: Int = 0

    init(storage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>) {
        self.storage = storage
    }

    var isEmpty: Bool {
        count == 0
    }

    var peek: HNSWNeighborCandidate? {
        guard count > 0 else { return nil }
        return storage[0]
    }

    @inline(__always)
    mutating func push(_ element: HNSWNeighborCandidate) {
        precondition(count < storage.count, "Candidate queue capacity is exhausted")
        pushUnchecked(element)
    }

    @inline(__always)
    mutating func pushUnchecked(_ element: HNSWNeighborCandidate) {
        storage[count] = element
        count += 1
        siftUp(from: count - 1)
    }

    @inline(__always)
    mutating func popUnchecked() -> HNSWNeighborCandidate {
        if count == 1 {
            count = 0
            return storage[0]
        }

        let value = storage[0]
        count -= 1
        storage[0] = storage[count]
        siftDown(from: 0)
        return value
    }

    @inline(__always)
    private mutating func siftUp(from index: Int) {
        var child = index
        let value = storage[child]
        while child > 0 {
            let parent = parentIndex(of: child)
            let parentValue = storage[parent]
            guard isCloserHNSWCandidate(value, than: parentValue) else { break }
            storage[child] = parentValue
            child = parent
        }
        storage[child] = value
    }

    @inline(__always)
    private mutating func siftDown(from index: Int) {
        var parent = index
        let value = storage[parent]
        while true {
            let left = leftChildIndex(of: parent)
            let right = rightChildIndex(of: parent)
            guard left < count else { break }

            var child = left
            var childValue = storage[left]
            if right < count {
                let rightValue = storage[right]
                if isCloserHNSWCandidate(rightValue, than: childValue) {
                    child = right
                    childValue = rightValue
                }
            }

            guard isCloserHNSWCandidate(childValue, than: value) else { break }
            storage[parent] = childValue
            parent = child
        }
        storage[parent] = value
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
