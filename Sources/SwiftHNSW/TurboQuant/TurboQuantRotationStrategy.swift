/// The random rotation contract used by the TurboQuant MSE stage.
public enum TurboQuantRotationStrategy: UInt32, Sendable {
    /// Three randomized Hadamard stages with `O(d log d)` application cost.
    case structuredHadamard = 0

    /// A Haar-distributed orthogonal transform represented by Householder reflectors.
    case haar = 1
}
