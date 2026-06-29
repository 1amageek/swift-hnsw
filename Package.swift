// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

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
    targets: [
        .target(
            name: "SwiftHNSW",
            path: "Sources/SwiftHNSW"
        ),
        .testTarget(
            name: "SwiftHNSWTests",
            dependencies: ["SwiftHNSW"]
        ),
    ]
)
