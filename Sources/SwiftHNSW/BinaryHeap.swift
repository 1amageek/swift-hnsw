struct BinaryHeap<Element> {
    private var elements: [Element] = []
    private let hasHigherPriority: (Element, Element) -> Bool

    init(hasHigherPriority: @escaping (Element, Element) -> Bool) {
        self.hasHigherPriority = hasHigherPriority
    }

    var isEmpty: Bool {
        elements.isEmpty
    }

    var count: Int {
        elements.count
    }

    var peek: Element? {
        elements.first
    }

    mutating func reserveCapacity(_ minimumCapacity: Int) {
        elements.reserveCapacity(minimumCapacity)
    }

    mutating func push(_ element: Element) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    @discardableResult
    mutating func pop() -> Element? {
        guard !elements.isEmpty else { return nil }
        guard elements.count > 1 else {
            return elements.removeLast()
        }

        let value = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        return value
    }

    func unorderedElements() -> [Element] {
        elements
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = parentIndex(of: child)
        while child > 0, hasHigherPriority(elements[child], elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = parentIndex(of: child)
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left = leftChildIndex(of: parent)
            let right = rightChildIndex(of: parent)
            var candidate = parent

            if left < elements.count, hasHigherPriority(elements[left], elements[candidate]) {
                candidate = left
            }
            if right < elements.count, hasHigherPriority(elements[right], elements[candidate]) {
                candidate = right
            }
            guard candidate != parent else { return }

            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }

    private func parentIndex(of index: Int) -> Int {
        (index - 1) / 2
    }

    private func leftChildIndex(of index: Int) -> Int {
        index * 2 + 1
    }

    private func rightChildIndex(of index: Int) -> Int {
        index * 2 + 2
    }
}
