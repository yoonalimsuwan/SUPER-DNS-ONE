"""
=============================================================================
SESI CoreML iOS DEPLOYMENT
PyTorch → CoreML (.mlpackage) → Swift Wrapper → Xcode Project
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

COREML CONVERSION PIPELINE
──────────────────────────
1. torch.jit.trace model with example input
2. coremltools.convert() with:
   - compute_units: ALL (ANE + GPU + CPU)
   - minimum_deployment_target: iOS16
   - dynamic shapes via ct.Shape(ct.RangeDim)
   - mlprogram format (MLX backend)
3. Optional quantization (int8 weight-only)
4. Save as .mlpackage

SWIFT WRAPPER GENERATION
────────────────────────
Generates a complete Swift package:
- SESIInference.swift (actor-based, thread-safe)
- Package.swift (SPM manifest)
- Info.plist additions
- Usage example

HONEST SCOPE
────────────
• coremltools must run on macOS (not Linux/Windows)
• ANE placement is op-dependent; verify with Xcode Instruments
• FFT-based models (FNO) may not convert; use conv-safe variant
• Dynamic shapes have limited ANE support
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("sesi.coreml_ios")

__all__ = [
    "CoreMLIOSConfig",
    "CoreMLIOSConverter",
    "SwiftWrapperGenerator",
]


# ── Optional imports ──────────────────────────────────────────────────────
try:
    import coremltools as ct
    HAS_COREMLTOOLS = True
except ImportError:
    HAS_COREMLTOOLS = False
    ct = None


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoreMLIOSConfig:
    # ── Model identity ──────────────────────────────────────────────────
    model_name: str = "SESICFDSurrogate"
    model_version: str = "1.0.0"
    author: str = "PAI & Yoon A Limsuwan / MSPS NETWORK"
    description: str = "SESI differentiable CFD surrogate for iOS"
    license: str = "MIT"

    # ── Input spec ──────────────────────────────────────────────────────
    input_name: str = "input"
    output_name: str = "output"
    input_shape: List[int] = field(default_factory=lambda: [1, 3, 64, 64])
    
    # ── Dynamic shapes (for variable-length proteins) ───────────────────
    use_dynamic_shapes: bool = True
    batch_range: Tuple[int, int] = (1, 4)
    seq_len_range: Tuple[int, int] = (32, 512)
    spatial_range: Tuple[int, int] = (32, 128)  # H, W

    # ── Conversion options ──────────────────────────────────────────────
    compute_units: str = "all"  # all | cpu_and_gpu | cpu_only
    minimum_deployment_target: str = "iOS16"
    convert_to: str = "mlprogram"  # mlprogram | neuralnetwork
    quantize: bool = False
    quantize_bits: int = 8  # 8 | 4

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "./coreml_ios"
    generate_swift: bool = True
    generate_xcode_project: bool = True


# ═══════════════════════════════════════════════════════════════════════
# COREML CONVERTER
# ═══════════════════════════════════════════════════════════════════════

class CoreMLIOSConverter:
    """
    Converts PyTorch models to CoreML .mlpackage for iOS deployment.
    
    Features:
    - Dynamic shape support via ct.RangeDim
    - Compute unit selection (ANE/GPU/CPU)
    - Optional int8 quantization
    - Metadata embedding
    """

    def __init__(self, cfg: CoreMLIOSConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        if not HAS_COREMLTOOLS:
            logger.warning(
                "coremltools not installed. Install with: pip install coremltools. "
                "Must run on macOS for actual conversion."
            )

    def _build_input_spec(self) -> Any:
        """Build CoreML input specification with optional dynamic shapes."""
        if not HAS_COREMLTOOLS:
            return None

        cfg = self.cfg
        shape = list(cfg.input_shape)

        if cfg.use_dynamic_shapes:
            # Dynamic batch
            batch_dim = ct.RangeDim(
                lower_bound=cfg.batch_range[0],
                upper_bound=cfg.batch_range[1],
                default=shape[0],
            )
            
            # Dynamic spatial dims (for CFD grids or sequence lengths)
            if len(shape) == 4:  # (B, C, H, W)
                h_dim = ct.RangeDim(
                    lower_bound=cfg.spatial_range[0],
                    upper_bound=cfg.spatial_range[1],
                    default=shape[2],
                )
                w_dim = ct.RangeDim(
                    lower_bound=cfg.spatial_range[0],
                    upper_bound=cfg.spatial_range[1],
                    default=shape[3],
                )
                dynamic_shape = (batch_dim, shape[1], h_dim, w_dim)
            elif len(shape) == 3:  # (B, C, L) for sequences
                l_dim = ct.RangeDim(
                    lower_bound=cfg.seq_len_range[0],
                    upper_bound=cfg.seq_len_range[1],
                    default=shape[2],
                )
                dynamic_shape = (batch_dim, shape[1], l_dim)
            else:
                dynamic_shape = shape

            return ct.TensorType(
                name=cfg.input_name,
                shape=ct.Shape(dynamic_shape),
            )
        else:
            return ct.TensorType(
                name=cfg.input_name,
                shape=ct.Shape(shape),
            )

    def _get_compute_units(self) -> Any:
        """Map config string to coremltools ComputeUnit enum."""
        if not HAS_COREMLTOOLS:
            return None
        
        mapping = {
            "all": ct.ComputeUnit.ALL,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }
        return mapping.get(self.cfg.compute_units, ct.ComputeUnit.ALL)

    def convert(self, model: nn.Module) -> Dict[str, Any]:
        """
        Convert PyTorch model to CoreML.
        
        Args:
            model: PyTorch model (will be set to eval mode)
            
        Returns:
            dict with success status, path, and warnings
        """
        result = {
            "success": False,
            "path": None,
            "warnings": [],
            "notes": [],
        }

        if not HAS_COREMLTOOLS:
            result["warnings"].append(
                "coremltools not installed. Cannot convert. "
                "Install with: pip install coremltools (macOS only for actual conversion)"
            )
            return result

        cfg = self.cfg
        model.eval()
        model = model.cpu()

        # Create example input
        example_input = torch.randn(*cfg.input_shape)

        # Trace model
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input, strict=False)
        except Exception as e:
            result["warnings"].append(f"torch.jit.trace failed: {e}")
            return result

        # Build input spec
        input_spec = self._build_input_spec()

        # Convert
        try:
            mlmodel = ct.convert(
                traced,
                inputs=[input_spec],
                compute_units=self._get_compute_units(),
                minimum_deployment_target=getattr(ct.target, cfg.minimum_deployment_target),
                convert_to=cfg.convert_to,
            )
        except Exception as e:
            result["warnings"].append(f"coremltools.convert failed: {e}")
            result["notes"].append(
                "If model uses FFT (torch.fft), consider using CoreMLCompatibleCFDSurrogate "
                "(conv-based) instead."
            )
            return result

        # Add metadata
        mlmodel.author = cfg.author
        mlmodel.license = cfg.license
        mlmodel.short_description = cfg.description
        mlmodel.version = cfg.model_version
        mlmodel.user_defined_metadata["sesi.model_name"] = cfg.model_name
        mlmodel.user_defined_metadata["sesi.compute_units"] = cfg.compute_units
        mlmodel.user_defined_metadata["sesi.quantized"] = str(cfg.quantize)

        # Quantize if requested
        if cfg.quantize:
            try:
                op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype=f"int{cfg.quantize_bits}",
                    granularity="per_tensor",
                )
                opt_config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
                mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=opt_config)
                result["notes"].append(f"Quantized to int{cfg.quantize_bits}")
            except Exception as e:
                result["warnings"].append(f"Quantization failed: {e}")

        # Save
        output_path = Path(cfg.output_dir) / f"{cfg.model_name}.mlpackage"
        mlmodel.save(str(output_path))
        
        result["success"] = True
        result["path"] = str(output_path)
        result["notes"].append(f"Saved to {output_path}")

        logger.info(f"CoreML conversion complete: {output_path}")
        return result


# ═══════════════════════════════════════════════════════════════════════
# SWIFT WRAPPER GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class SwiftWrapperGenerator:
    """
    Generates a complete Swift package for iOS integration.
    
    Generates:
    - Package.swift (SPM manifest)
    - Sources/SESIInference/SESIInference.swift (main wrapper)
    - Sources/SESIInference/Resources/ (model location)
    - README.md with usage instructions
    """

    SWIFT_PACKAGE_TEMPLATE = '''// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SESIInference",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "SESIInference", targets: ["SESIInference"]),
    ],
    targets: [
        .target(
            name: "SESIInference",
            path: "Sources/SESIInference",
            resources: [
                .copy("Resources/{model_name}.mlpackage"),
            ]
        ),
    ]
)
'''

    SWIFT_WRAPPER_TEMPLATE = '''import Foundation
import CoreML
import Accelerate

/// Thread-safe SESI inference actor for iOS/macOS.
///
/// Usage:
/// ```swift
/// let inference = try await SESIInference()
/// let input: [Float] = ... // your input data
/// let output = try await inference.predict(input: input, height: 64, width: 64, channels: 3)
/// ```
public actor SESIInference {{
    
    public struct Prediction {{
        public let output: [Float]
        public let shape: [Int]
        public let elapsedMs: Double
    }}
    
    public enum ComputeUnit: String {{
        case all = "all"
        case cpuAndGPU = "cpu_and_gpu"
        case cpuOnly = "cpu_only"
    }}
    
    private let model: MLModel
    private let config: MLModelConfiguration
    
    /// Initialize with bundled model.
    public init(computeUnit: ComputeUnit = .all) throws {{
        let mlConfig = MLModelConfiguration()
        switch computeUnit {{
        case .all:
            mlConfig.computeUnits = .all
        case .cpuAndGPU:
            mlConfig.computeUnits = .cpuAndGPU
        case .cpuOnly:
            mlConfig.computeUnits = .cpuOnly
        }}
        self.config = mlConfig
        
        guard let modelURL = Bundle.module.url(
            forResource: "{model_name}",
            withExtension: "mlpackage",
            subdirectory: "Resources"
        ) else {{
            throw SESIInferenceError.modelNotFound
        }}
        
        self.model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
    }}
    
    /// Initialize with custom model path.
    public init(modelURL: URL, computeUnit: ComputeUnit = .all) throws {{
        let mlConfig = MLModelConfiguration()
        switch computeUnit {{
        case .all:
            mlConfig.computeUnits = .all
        case .cpuAndGPU:
            mlConfig.computeUnits = .cpuAndGPU
        case .cpuOnly:
            mlConfig.computeUnits = .cpuOnly
        }}
        self.config = mlConfig
        self.model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
    }}
    
    /// Run inference on input data.
    ///
    /// - Parameters:
    ///   - input: Flat array of Float values in NCHW order
    ///   - height: Spatial height (for 4D input)
    ///   - width: Spatial width (for 4D input)
    ///   - channels: Number of channels
    /// - Returns: Prediction with output data and timing
    public func predict(
        input: [Float],
        height: Int,
        width: Int,
        channels: Int
    ) async throws -> Prediction {{
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create MLMultiArray
        let shape = [1, channels, height, width].map {{ NSNumber(value: $0) }}
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Copy input data
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        input.withUnsafeBufferPointer {{ buffer in
            ptr.update(from: buffer.baseAddress!, count: input.count)
        }}
        
        // Create feature provider
        let inputFeature = MLFeatureValue(multiArray: multiArray)
        let inputProvider = try MLDictionaryFeatureProvider(
            dictionary: ["{input_name}": inputFeature]
        )
        
        // Run prediction
        let outputProvider = try model.prediction(from: inputProvider)
        
        guard let outputFeature = outputProvider.featureValue(for: "{output_name}"),
              let outputArray = outputFeature.multiArrayValue else {{
            throw SESIInferenceError.outputNotFound
        }}
        
        // Extract output
        let outputCount = outputArray.count
        let outputPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputCount)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: outputCount))
        
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        
        return Prediction(
            output: output,
            shape: outputArray.shape.map {{ $0.intValue }},
            elapsedMs: elapsed
        )
    }}
    
    /// Get model metadata.
    public var metadata: [String: String] {{
        var meta: [String: String] = [:]
        if let author = model.modelDescription.metadata[.author] as? String {{
            meta["author"] = author
        }}
        if let description = model.modelDescription.metadata[.description] as? String {{
            meta["description"] = description
        }}
        if let version = model.modelDescription.metadata[.version] as? String {{
            meta["version"] = version
        }}
        return meta
    }}
}}

public enum SESIInferenceError: Error, LocalizedError {{
    case modelNotFound
    case outputNotFound
    
    public var errorDescription: String? {{
        switch self {{
        case .modelNotFound:
            return "SESI model not found in bundle resources"
        case .outputNotFound:
            return "Model output not found"
        }}
    }}
}}
'''

    README_TEMPLATE = '''# SESI Inference iOS Package

CoreML inference wrapper for SESI models on iOS/macOS.

## Installation

1. Copy the `.mlpackage` file to `Sources/SESIInference/Resources/`
2. Add this package to your Xcode project via Swift Package Manager
3. Import and use:

```swift
import SESIInference

let inference = try await SESIInference()
let input: [Float] = // your input data
let prediction = try await inference.predict(
    input: input,
    height: 64,
    width: 64,
    channels: 3
)
print("Output shape: \\(prediction.shape)")
print("Elapsed: \\(prediction.elapsedMs) ms")
```

## Compute Units

The model can run on different hardware:

```swift
// Use Neural Engine + GPU + CPU (fastest, op-dependent)
let inference = try await SESIInference(computeUnit: .all)

// Use GPU + CPU only (more predictable)
let inference = try await SESIInference(computeUnit: .cpuAndGPU)

// Use CPU only (for debugging)
let inference = try await SESIInference(computeUnit: .cpuOnly)
```

## Notes

- ANE (Neural Engine) placement is op-dependent; verify with Xcode Instruments
- Dynamic shapes have limited ANE support
- First inference includes model compilation time (~100ms-2s)
- Subsequent inferences are much faster

## Requirements

- iOS 16.0+ / macOS 13.0+
- Xcode 14.0+
'''

    def __init__(self, cfg: CoreMLIOSConfig):
        self.cfg = cfg

    def generate(self, mlpackage_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate complete Swift package structure.
        
        Args:
            mlpackage_path: Path to .mlpackage file (will be copied)
            
        Returns:
            dict with generated file paths
        """
        cfg = self.cfg
        output_dir = Path(cfg.output_dir) / "swift_package"
        output_dir.mkdir(parents=True, exist_ok=True)

        generated = {}

        # Package.swift
        package_swift = self.SWIFT_PACKAGE_TEMPLATE.format(model_name=cfg.model_name)
        package_path = output_dir / "Package.swift"
        package_path.write_text(package_swift)
        generated["Package.swift"] = str(package_path)

        # Sources directory
        sources_dir = output_dir / "Sources" / "SESIInference"
        sources_dir.mkdir(parents=True, exist_ok=True)

        # Resources directory
        resources_dir = sources_dir / "Resources"
        resources_dir.mkdir(parents=True, exist_ok=True)

        # Copy mlpackage if provided
        if mlpackage_path and Path(mlpackage_path).exists():
            import shutil
            dest = resources_dir / f"{cfg.model_name}.mlpackage"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(mlpackage_path, dest)
            generated["mlpackage"] = str(dest)

        # Swift wrapper
        swift_code = self.SWIFT_WRAPPER_TEMPLATE.format(
            model_name=cfg.model_name,
            input_name=cfg.input_name,
            output_name=cfg.output_name,
        )
        swift_path = sources_dir / "SESIInference.swift"
        swift_path.write_text(swift_code)
        generated["SESIInference.swift"] = str(swift_path)

        # README
        readme_path = output_dir / "README.md"
        readme_path.write_text(self.README_TEMPLATE)
        generated["README.md"] = str(readme_path)

        logger.info(f"Swift package generated at {output_dir}")
        return generated


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def convert_for_ios(
    model: nn.Module,
    cfg: Optional[CoreMLIOSConfig] = None,
) -> Dict[str, Any]:
    """
    One-stop function to convert model and generate Swift package.
    
    Args:
        model: PyTorch model
        cfg: Configuration (uses defaults if None)
        
    Returns:
        dict with conversion result and generated files
    """
    cfg = cfg or CoreMLIOSConfig()
    
    # Convert
    converter = CoreMLIOSConverter(cfg)
    convert_result = converter.convert(model)
    
    result = {"conversion": convert_result, "swift_package": None}
    
    # Generate Swift package if conversion succeeded
    if convert_result["success"] and cfg.generate_swift:
        generator = SwiftWrapperGenerator(cfg)
        swift_files = generator.generate(convert_result["path"])
        result["swift_package"] = swift_files
    
    return result
