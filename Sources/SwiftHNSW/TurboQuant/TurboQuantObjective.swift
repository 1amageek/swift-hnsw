/// The distortion objective implemented by a TurboQuant index.
public enum TurboQuantObjective: UInt32, Sendable {
    /// Algorithm 1: stores `bitWidth` Lloyd-Max indices per coordinate.
    case meanSquaredError = 0

    /// Algorithm 2: stores `(bitWidth - 1)` Lloyd-Max indices, one QJL sign bit
    /// per coordinate, and the residual norm used by the unbiased estimator.
    case innerProduct = 1
}
