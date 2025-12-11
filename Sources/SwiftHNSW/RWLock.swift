import Foundation

/// Read-Write Lock for efficient concurrent access
/// Allows multiple concurrent readers, but exclusive access for writers
final class RWLock: @unchecked Sendable {
    private var lock = pthread_rwlock_t()

    init() {
        pthread_rwlock_init(&lock, nil)
    }

    deinit {
        pthread_rwlock_destroy(&lock)
    }

    func withReadLock<T>(_ body: () throws -> T) rethrows -> T {
        pthread_rwlock_rdlock(&lock)
        defer { pthread_rwlock_unlock(&lock) }
        return try body()
    }

    func withWriteLock<T>(_ body: () throws -> T) rethrows -> T {
        pthread_rwlock_wrlock(&lock)
        defer { pthread_rwlock_unlock(&lock) }
        return try body()
    }
}
