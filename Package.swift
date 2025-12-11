// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-hnsw",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .library(
            name: "SwiftHNSW",
            targets: ["SwiftHNSW"]
        ),
    ],
    targets: [
        // C++ hnswlib target with C bridge
        .target(
            name: "hnswlib",
            path: "Sources/hnswlib",
            sources: [
                "hnswlib_swift_bridge.cpp",
                "space_f16.cpp"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                // Enable SIMD optimizations
                .unsafeFlags(["-std=c++17", "-O3", "-ffast-math", "-march=native"], .when(configuration: .release)),
                .unsafeFlags(["-std=c++17"], .when(configuration: .debug))
            ]
        ),
        // Swift wrapper target
        .target(
            name: "SwiftHNSW",
            dependencies: ["hnswlib"],
            path: "Sources/SwiftHNSW",
            swiftSettings: [
                .unsafeFlags(["-O", "-whole-module-optimization"], .when(configuration: .release))
            ]
        ),
        .testTarget(
            name: "SwiftHNSWTests",
            dependencies: ["SwiftHNSW"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
