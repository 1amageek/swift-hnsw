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
    var topDistanceUnchecked: Float {
        storage[0].distance
    }

    @inline(__always)
    var topUnchecked: HNSWNeighborCandidate {
        storage[0]
    }

    @inline(__always)
    mutating func push(_ element: HNSWNeighborCandidate) {
        precondition(count < storage.count, "Nearest candidate capacity is exhausted")
        pushUnchecked(element)
    }

    @inline(__always)
    mutating func pushUnchecked(_ element: HNSWNeighborCandidate) {
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

    @inline(__always)
    mutating func replaceTopUnchecked(with element: HNSWNeighborCandidate) {
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
                topK.pushUnchecked(element)
            } else if isCloserHNSWCandidate(element, than: topK.topUnchecked) {
                topK.replaceTopUnchecked(with: element)
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
                topK.pushUnchecked(element)
            } else if isCloserHNSWCandidate(element, than: topK.topUnchecked) {
                topK.replaceTopUnchecked(with: element)
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
        let value = storage[child]
        while child > 0 {
            let parent = parentIndex(of: child)
            let parentValue = storage[parent]
            guard isFartherHNSWCandidate(value, than: parentValue) else { break }
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
                if isFartherHNSWCandidate(rightValue, than: childValue) {
                    child = right
                    childValue = rightValue
                }
            }

            guard isFartherHNSWCandidate(childValue, than: value) else { break }
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
