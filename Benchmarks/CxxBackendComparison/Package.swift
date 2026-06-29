// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "swift-hnsw-cxx-backend-comparison",
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
                "hnswlib_swift_bridge.cpp",
                "space_f16.cpp",
                "space_turboquant.cpp",
                "turboquant_encoder.cpp"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include")
            ]
        ),
        .executableTarget(
            name: "CxxBackendComparison",
            dependencies: [
                .product(name: "SwiftHNSW", package: "swift-hnsw"),
                "hnswlib"
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
