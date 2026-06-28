@inline(__always)
func insertTopKSearchResult(
    _ candidate: SearchResult,
    into results: inout [SearchResult],
    limit: Int
) {
    guard limit > 0 else { return }

    if results.count == limit,
       let worst = results.last,
       !isSearchResultOrderedBefore(candidate, worst) {
        return
    }

    var insertionIndex = results.count
    while insertionIndex > 0,
          isSearchResultOrderedBefore(candidate, results[insertionIndex - 1]) {
        insertionIndex -= 1
    }

    results.insert(candidate, at: insertionIndex)
    if results.count > limit {
        results.removeLast()
    }
}

@inline(__always)
private func isSearchResultOrderedBefore(_ lhs: SearchResult, _ rhs: SearchResult) -> Bool {
    if lhs.distance == rhs.distance {
        return lhs.label < rhs.label
    }
    return lhs.distance < rhs.distance
}
