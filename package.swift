// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SESIInference",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "SESIInference", targets: ["SESIInference"])
    ],
    targets: [
        .target(
            name: "SESIInference",
            path: "Sources/SESIInference",
            resources: [
                .copy("Resources/sesi_cfd.mlpackage")
            ]
        )
    ]
)
