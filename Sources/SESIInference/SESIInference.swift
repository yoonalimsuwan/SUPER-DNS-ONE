import Foundation
import CoreML
import Accelerate

/// Thread-safe, actor-isolated CoreML inference for the SESI CFD surrogate.
public actor SESIInference {

    public enum ComputeUnit: String, Sendable {
        case all, cpuAndGPU, cpuOnly
    }

    public struct Prediction: Sendable {
        public let outputs: [String: [Float]]
        public let shape: [Int]
        public let elapsedMilliseconds: Double
    }

    public struct Configuration: Sendable {
        public var channels: Int
        public var height: Int
        public var width: Int
        public var inputName: String
        public var outputName: String
        public var computeUnit: ComputeUnit

        public init(channels: Int = 3, height: Int = 64, width: Int = 64,
                    inputName: String = "input", outputName: String = "output",
                    computeUnit: ComputeUnit = .all) {
            self.channels = channels
            self.height = height
            self.width = width
            self.inputName = inputName
            self.outputName = outputName
            self.computeUnit = computeUnit
        }
    }

    private let model: MLModel
    private let config: Configuration

    public init(configuration: Configuration = Configuration(),
                modelURL: URL? = nil) throws {
        self.config = configuration

        let url = try modelURL ?? Self.bundledModelURL()

        let mlConfig = MLModelConfiguration()
        switch configuration.computeUnit {
        case .all:        mlConfig.computeUnits = .all
        case .cpuAndGPU:  mlConfig.computeUnits = .cpuAndGPU
        case .cpuOnly:    mlConfig.computeUnits = .cpuOnly
        }

        self.model = try MLModel(contentsOf: url, configuration: mlConfig)
    }

    private static func bundledModelURL() throws -> URL {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle(for: BundleToken.self)
        #endif

        guard let url = bundle.url(forResource: "sesi_cfd",
                                   withExtension: "mlpackage",
                                   subdirectory: "Resources")
                ?? bundle.url(forResource: "sesi_cfd", withExtension: "mlpackage")
        else {
            throw NSError(domain: "SESIInference", code: 404,
                          userInfo: [NSLocalizedDescriptionKey:
                            "sesi_cfd.mlpackage not found in bundle Resources."])
        }
        return url
    }

    /// Predict a flow field.
    /// - Parameter input: flat row-major array, shape [C, H, W], batch = 1.
    public func predict(_ input: [Float]) async throws -> Prediction {
        let c = config.channels, h = config.height, w = config.width
        let expected = c * h * w
        guard input.count == expected else {
            throw NSError(domain: "SESIInference", code: 400,
                userInfo: [NSLocalizedDescriptionKey:
                    "Expected \(expected) floats, got \(input.count)"])
        }

        let shape = [1, c, h, w].map { NSNumber(value: $0) }
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        let dst = array.dataPointer.bindMemory(to: Float.self, capacity: expected)
        input.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: expected)
        }

        let provider = try MLDictionaryFeatureProvider(
            dictionary: [config.inputName: MLFeatureValue(multiArray: array)])

        let start = ContinuousClock.now
        let result = try model.prediction(from: provider)
        let elapsed = start.duration(to: .now).components.seconds * 1000
            + Double(start.duration(to: .now).components.attoseconds) / 1e15

        guard let out = result.featureValue(for: config.outputName)?.multiArray
        else {
            throw NSError(domain: "SESIInference", code: 500,
                userInfo: [NSLocalizedDescriptionKey:
                    "Output '\(config.outputName)' is not an MLMultiArray."])
        }

        let count = out.count
        let ptr = out.dataPointer.bindMemory(to: Float.self, capacity: count)
        let values = UnsafeBufferPointer(start: ptr, count: count)

        return Prediction(
            outputs: [config.outputName: Array(values)],
            shape: out.shape.map { $0.intValue },
            elapsedMilliseconds: elapsed
        )
    }
}

private final class BundleToken {}
