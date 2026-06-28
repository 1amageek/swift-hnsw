// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let nativePlatforms: [Platform] = [
    .macOS,
    .iOS,
    .linux,
]

let package = Package(
    name: "swift-hnsw",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        .library(
            name: "SwiftHNSW",
            targets: ["SwiftHNSW"]
        ),
    ],
    traits: [
        .trait(name: "CxxBackend"),
    ],
    targets: [
        // C++ hnswlib target with C bridge
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
        // Public Swift target
        .target(
            name: "SwiftHNSW",
            dependencies: [
                .target(
                    name: "hnswlib",
                    condition: .when(platforms: nativePlatforms, traits: ["CxxBackend"])
                ),
            ],
            path: "Sources/SwiftHNSW",
            swiftSettings: [
                .define("HNSWLIB_BACKEND", .when(platforms: nativePlatforms, traits: ["CxxBackend"])),
            ]
        ),
        .testTarget(
            name: "SwiftHNSWTests",
            dependencies: ["SwiftHNSW"],
            swiftSettings: [
                .define("HNSWLIB_BACKEND", .when(platforms: nativePlatforms, traits: ["CxxBackend"])),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
