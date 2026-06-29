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
        storage[count] = element
        count += 1
        siftUp(from: count - 1)
    }

    @inline(__always)
    mutating func popUnchecked() -> HNSWNeighborCandidate {
        precondition(count > 0, "Heap must not be empty")
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
        var parent = parentIndex(of: child)
        while child > 0, isCloserHNSWCandidate(storage[child], than: storage[parent]) {
            storage.swapAt(child, parent)
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

            if left < count, isCloserHNSWCandidate(storage[left], than: storage[candidate]) {
                candidate = left
            }
            if right < count, isCloserHNSWCandidate(storage[right], than: storage[candidate]) {
                candidate = right
            }
            guard candidate != parent else { return }

            storage.swapAt(parent, candidate)
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
