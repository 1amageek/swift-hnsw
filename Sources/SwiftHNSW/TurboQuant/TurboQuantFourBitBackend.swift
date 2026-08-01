/// Selects the four-bit MSE scoring implementation for controlled benchmarking.
@_spi(Benchmarking)
public enum TurboQuantFourBitBackend: Sendable, Equatable {
    /// Uses ARM64 NEON when available and the scalar C kernel elsewhere.
    case platform

    /// Uses the allocation-free direct Swift implementation.
    case swiftDirect

    /// Uses the coordinate or packed lookup-table Swift implementation.
    case swiftLookup
}
