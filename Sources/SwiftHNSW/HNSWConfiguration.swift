/// Configuration for HNSW index
public struct HNSWConfiguration: Sendable, Hashable {
    /// Maximum number of connections per element (default: 16)
    /// Higher values improve recall but increase memory and build time
    public var m: Int

    /// Size of the dynamic candidate list during construction (default: 200)
    /// Higher values improve index quality but increase build time
    public var efConstruction: Int

    /// Size of the dynamic candidate list during search (default: 10)
    /// Higher values improve recall but increase search time
    public var efSearch: Int

    /// Random seed for reproducibility (default: 100)
    public var randomSeed: Int

    /// Allow replacing deleted elements (default: false)
    public var allowReplaceDeleted: Bool

    public init(
        m: Int = 16,
        efConstruction: Int = 200,
        efSearch: Int = 10,
        randomSeed: Int = 100,
        allowReplaceDeleted: Bool = false
    ) {
        self.m = m
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.randomSeed = randomSeed
        self.allowReplaceDeleted = allowReplaceDeleted
    }
}

// MARK: - Presets

extension HNSWConfiguration {
    /// Fast configuration prioritizing speed over accuracy
    public static var fast: HNSWConfiguration {
        HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 10)
    }

    /// Balanced configuration for general use
    public static var balanced: HNSWConfiguration {
        HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
    }

    /// High accuracy configuration prioritizing recall
    public static var highAccuracy: HNSWConfiguration {
        HNSWConfiguration(m: 32, efConstruction: 400, efSearch: 200)
    }
}
