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
                .headerSearchPath("include")
            ]
        ),
        // Swift wrapper target
        .target(
            name: "SwiftHNSW",
            dependencies: ["hnswlib"],
            path: "Sources/SwiftHNSW"
        ),
        .testTarget(
            name: "SwiftHNSWTests",
            dependencies: ["SwiftHNSW"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
