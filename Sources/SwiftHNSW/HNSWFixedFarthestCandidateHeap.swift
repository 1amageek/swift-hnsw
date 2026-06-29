struct HNSWFixedFarthestCandidateHeap {
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
        precondition(count < storage.count, "Nearest candidate capacity is exhausted")
        storage[count] = element
        count += 1
        siftUp(from: count - 1)
    }

    @inline(__always)
    mutating func replaceTop(with element: HNSWNeighborCandidate) {
        guard count > 0 else {
            push(element)
            return
        }
        storage[0] = element
        siftDown(from: 0)
    }

    mutating func sortedElements() -> [HNSWNeighborCandidate] {
        sortStoredElements()
        return Array(UnsafeBufferPointer(start: storage.baseAddress!, count: count))
    }

    mutating func closestSorted(
        limit: Int,
        using resultStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>
    ) -> [HNSWNeighborCandidate] {
        guard limit > 0, count > 0 else { return [] }
        let boundedLimit = min(limit, count)
        guard count > boundedLimit else {
            return sortedElements()
        }

        var topK = HNSWFixedFarthestCandidateHeap(storage: resultStorage)
        for index in 0..<count {
            let element = storage[index]
            if topK.count < boundedLimit {
                topK.push(element)
            } else if let farthest = topK.peek, isCloserHNSWCandidate(element, than: farthest) {
                topK.replaceTop(with: element)
            }
        }
        return topK.sortedElements()
    }

    mutating func writeClosestSorted(
        limit: Int,
        using resultStorage: UnsafeMutableBufferPointer<HNSWNeighborCandidate>,
        to results: UnsafeMutableBufferPointer<SearchResult>,
        labelOrder: borrowing [UInt64],
        metric: DistanceMetric
    ) -> Int {
        guard limit > 0, count > 0, results.count > 0 else { return 0 }
        let boundedLimit = min(limit, count, results.count)
        guard count > boundedLimit else {
            sortStoredElements()
            return writeStoredElements(
                limit: boundedLimit,
                to: results,
                labelOrder: labelOrder,
                metric: metric
            )
        }

        var topK = HNSWFixedFarthestCandidateHeap(storage: resultStorage)
        for index in 0..<count {
            let element = storage[index]
            if topK.count < boundedLimit {
                topK.push(element)
            } else if let farthest = topK.peek, isCloserHNSWCandidate(element, than: farthest) {
                topK.replaceTop(with: element)
            }
        }
        topK.sortStoredElements()
        return topK.writeStoredElements(
            limit: boundedLimit,
            to: results,
            labelOrder: labelOrder,
            metric: metric
        )
    }

    private mutating func sortStoredElements() {
        guard count > 1 else { return }
        var sortedStorage = UnsafeMutableBufferPointer(start: storage.baseAddress!, count: count)
        sortedStorage.sort(by: isCloserHNSWCandidate)
    }

    private mutating func writeStoredElements(
        limit: Int,
        to results: UnsafeMutableBufferPointer<SearchResult>,
        labelOrder: borrowing [UInt64],
        metric: DistanceMetric
    ) -> Int {
        let resultCount = min(limit, count, results.count)
        guard resultCount > 0 else { return 0 }

        var needsTieBreakSort = false
        var previousResult: SearchResult?
        for index in 0..<resultCount {
            let candidate = storage[index]
            let result = SearchResult(
                label: labelOrder[Int(candidate.internalID)],
                distance: VectorOperations.publicDistance(fromComparisonDistance: candidate.distance, metric: metric)
            )
            if let previousResult,
               previousResult.distance == result.distance,
               previousResult.label > result.label {
                needsTieBreakSort = true
            }
            results[index] = result
            previousResult = result
        }

        if needsTieBreakSort {
            var sortedResults = UnsafeMutableBufferPointer(start: results.baseAddress!, count: resultCount)
            sortedResults.sort {
                if $0.distance == $1.distance {
                    return $0.label < $1.label
                }
                return $0.distance < $1.distance
            }
        }
        return resultCount
    }

    @inline(__always)
    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = parentIndex(of: child)
        while child > 0, isFartherHNSWCandidate(storage[child], than: storage[parent]) {
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

            if left < count, isFartherHNSWCandidate(storage[left], than: storage[candidate]) {
                candidate = left
            }
            if right < count, isFartherHNSWCandidate(storage[right], than: storage[candidate]) {
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
