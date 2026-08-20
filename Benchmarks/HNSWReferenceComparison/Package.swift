// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "hnsw-reference-comparison",
    platforms: [
        .macOS(.v26)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .target(
            name: "hnswlib",
            path: "Sources/hnswlib",
            sources: [
                "hnswlib_reference.cpp",
                "space_f16.cpp"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include")
            ]
        ),
        .target(
            name: "CTurboQuantBenchmarkReference",
            path: "Sources/CTurboQuantBenchmarkReference",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "HNSWReferenceComparison",
            dependencies: [
                .product(name: "SwiftHNSW", package: "swift-hnsw"),
                "hnswlib"
            ]
        ),
        .executableTarget(
            name: "TurboQuantComparison",
            dependencies: [
                .product(name: "SwiftHNSW", package: "swift-hnsw")
            ]
        ),
        .executableTarget(
            name: "TurboQuantKernelComparison",
            dependencies: [
                .product(name: "SwiftHNSW", package: "swift-hnsw"),
                "CTurboQuantBenchmarkReference"
            ]
        ),
        .executableTarget(
            name: "TurboQuantConcurrencyCheck",
            dependencies: [
                .product(name: "SwiftHNSW", package: "swift-hnsw")
            ]
        ),
        .testTarget(
            name: "SwiftHNSWBenchmarks",
            dependencies: [
                .product(name: "SwiftHNSW", package: "swift-hnsw")
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
