"""
=============================================================================
SESI ONNX RUNTIME MOBILE PIPELINE
ONNX → simplify → QDQ int8 → mobile EP profile → latency benchmark
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT | Year: 2026 | ORCID: 0009-0008-2374-0788

IMPORTANT
─────────
"ONNX Runtime Mobile" = a MINIMAL build of ONNX Runtime + Mobile EPs.
There is no separate mobile model file. The artifact is still .onnx.

Mobile Execution Providers
──────────────────────────
  XNNPACK  : CPU, ARM NEON / FP16, cross-platform
  CoreML   : iOS/macOS → ANE + GPU + CPU (op-dependent)
  NNAPI    : Android → DSP/NPU (vendor-dependent, often falls back to CPU)
  WebGPU   : browser / wasm

HONEST SCOPE
────────────
• FFT-based FNO will usually FAIL or fall back on CoreML EP.
  Use CoreMLCompatibleCFDSurrogate (conv-only) for mobile deployment.
• INT8 QDQ quantization REQUIRES representative calibration data.
  Quantizing without calibration is not a valid accuracy claim.
• ANE placement must be verified on device; it is not guaranteed.
• This script does not build ORT; it emits the build commands + profiles.
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.ort_mobile")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        QuantType, QuantFormat, CalibrationMethod, quantize_static,
    )
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = ["ORTMobileConfig", "NumpyCalibrationReader", "ORTMobilePipeline"]


@dataclass
class ORTMobileConfig:
    model_path: str = "./exported/sesi_cfd.onnx"
    output_dir: str = "./mobile_models"

    # Input contract (must match exported ONNX)
    in_channels: int = 3
    grid_h: int = 64
    grid_w: int = 64
    batch: int = 1

    # Quantization
    quantize: bool = True
    quant_format: str = "QDQ"            # QDQ is required for most mobile EPs
    quant_dtype: str = "QInt8"           # QInt8 | QUInt8
    calibration_path: Optional[str] = None  # .npz with key "input"
    calibration_samples: int = 64
    per_channel: bool = True

    # Providers (tried in order, with fallback)
    preferred_providers: List[str] = field(default_factory=lambda: [
        "XNNPACKExecutionProvider", "CPUExecutionProvider"
    ])

    # CoreML EP options (iOS/macOS)
    ml_compute_units: str = "ALL"        # ALL | CPUAndGPU | CPUOnly
    ml_model_format: str = "MLProgram"   # MLProgram | NeuralNetwork

    # Threading
    intra_op_threads: int = 2
    inter_op_threads: int = 1

    # Benchmark
    warmup_runs: int = 10
    benchmark_runs: int = 50


class NumpyCalibrationReader:
    """
    CalibrationDataReader for onnxruntime.quantization.quantize_static.

    Loads a .npz containing key "input" with shape (N, C, H, W),
    or generates deterministic synthetic calibration if none supplied.

    HONEST: synthetic calibration only validates the plumbing.
    For real INT8 accuracy, use in-distribution OpenFOAM snapshots.
    """

    def __init__(self, cfg: ORTMobileConfig):
        self.cfg = cfg
        self._i = 0

        if cfg.calibration_path and Path(cfg.calibration_path).exists():
            data = np.load(cfg.calibration_path)
            self.samples = data["input"][:cfg.calibration_samples].astype(np.float32)
            self.is_real = True
        else:
            rng = np.random.RandomState(42)
            self.samples = rng.randn(
                cfg.calibration_samples, cfg.in_channels, cfg.grid_h, cfg.grid_w
            ).astype(np.float32)
            self.is_real = False
            logger.warning(
                "Using SYNTHETIC calibration. INT8 accuracy is NOT validated. "
                "Supply calibration_path with real OpenFOAM snapshots."
            )

        self._n = len(self.samples)

    def get_next(self) -> Optional[np.ndarray]:
        if self._i >= self._n:
            return None
        x = self.samples[self._i]
        self._i += 1
        return x

    def rewind(self):
        self._i = 0


class ORTMobilePipeline:
    """Full ONNX → mobile-ready artifact + EP profile + benchmark."""

    def __init__(self, cfg: ORTMobileConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Simplify ONNX ────────────────────────────────────────────────
    def simplify(self) -> Path:
        src = Path(self.cfg.model_path)
        dst = Path(self.cfg.output_dir) / "simplified.onnx"

        if not HAS_ORT:
            raise RuntimeError("onnx + onnxruntime required. pip install onnx onnxruntime")

        try:
            import onnxsim
            model, ok = onnxsim.simplify(str(src))
            if not ok:
                raise RuntimeError("onnxsim reported an invalid simplified graph")
            onnx.save(model, str(dst))
            logger.info("ONNX simplified → %s", dst)
        except ImportError:
            # Still valid: copy and let ORT graph optimizer handle it
            import shutil
            shutil.copyfile(src, dst)
            logger.info("onnxsim unavailable; using original graph → %s", dst)

        return dst

    # ── 2. INT8 QDQ quantization ────────────────────────────────────────
    def quantize(self, onnx_path: Path) -> Path:
        dst = Path(self.cfg.output_dir) / "sesi_cfd_int8.onnx"

        if not self.cfg.quantize:
            return onnx_path

        reader = NumpyCalibrationReader(self.cfg)

        qtype = QuantType.QInt8 if self.cfg.quant_dtype == "QInt8" else QuantType.QUInt8
        qformat = (QuantFormat.QDQ if self.cfg.quant_format == "QDQ"
                   else QuantFormat.QOperator)

        quantize_static(
            model_input=str(onnx_path),
            model_output=str(dst),
            calibration_data_reader=reader,
            quant_format=qformat,
            per_channel=self.cfg.per_channel,
            weight_type=qtype,
            activation_type=qtype,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ReduceRange": True},
        )

        logger.info("INT8 QDQ model → %s (calibration_real=%s)", dst, reader.is_real)
        return dst

    # ── 3. Session + provider profile ───────────────────────────────────
    def make_session(self, model_path: Path) -> Tuple[Any, Dict[str, Any]]:
        available = set(ort.get_available_providers())
        providers: List[Any] = []

        for name in self.cfg.preferred_providers:
            if name not in available:
                logger.warning("Provider unavailable: %s", name)
                continue

            if name == "CoreMLExecutionProvider":
                providers.append((name, {
                    "MLComputeUnits": self.cfg.ml_compute_units,
                    "ModelFormat": self.cfg.ml_model_format,
                    "RequireStaticInputShapes": "0",
                    "EnableOnSubgraphs": "1",
                }))
            else:
                providers.append(name)

        if not providers:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.cfg.intra_op_threads
        opts.inter_op_num_threads = self.cfg.inter_op_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True

        session = ort.InferenceSession(str(model_path), sess_options=opts,
                                       providers=providers)

        profile = {
            "requested": self.cfg.preferred_providers,
            "available": sorted(available),
            "active": session.get_providers(),
            "ml_compute_units": self.cfg.ml_compute_units,
            "ml_model_format": self.cfg.ml_model_format,
            "intra_op_threads": self.cfg.intra_op_threads,
            "inputs": [(i.name, i.shape, i.type) for i in session.get_inputs()],
            "outputs": [(o.name, o.shape, o.type) for o in session.get_outputs()],
            "honest_note": (
                "Active provider list does not prove ANE usage. "
                "Inspect CoreML EP subgraph placement / Xcode instrumentation on device."
            ),
        }
        return session, profile

    # ── 4. Latency benchmark ────────────────────────────────────────────
    def benchmark(self, session: Any) -> Dict[str, float]:
        cfg = self.cfg
        input_name = session.get_inputs()[0].name

        # Respect dynamic batch if present
        shape = [cfg.batch, cfg.in_channels, cfg.grid_h, cfg.grid_w]
        rng = np.random.RandomState(0)
        x = rng.randn(*shape).astype(np.float32)

        for _ in range(cfg.warmup_runs):
            session.run(None, {input_name: x})

        times = []
        for _ in range(cfg.benchmark_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: x})
            times.append((time.perf_counter() - t0) * 1000.0)

        a = np.array(times)
        return {
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()),
            "std_ms": float(a.std()),
            "runs": cfg.benchmark_runs,
            "note": "Host-side latency only; excludes model compilation on first load.",
        }

    # ── 5. Emit mobile build commands (not executed) ────────────────────
    def build_commands(self) -> Dict[str, List[str]]:
        return {
            "android_arm64": [
                "./build.sh", "--config", "Release", "--minimal",
                "--build_shared_lib", "--skip_submodule_sync",
                "--android", "--android_abi", "arm64-v8a", "--android_api", "28",
                "--cmake_extra_defines", "onnxruntime_USE_XNNPACK=ON",
            ],
            "ios_arm64": [
                "./build.py", "--config", "Release", "--ios",
                "--ios_arch", "arm64", "--osx_arch", "arm64",
                "--cmake_generator", "Xcode", "--build_framework",
                "--minimal", "--skip_submodule_sync",
            ],
            "note": [
                "Flags are version-dependent; check the ORT release you pin.",
                "Enable onnxruntime_USE_COREML=ON for iOS CoreML EP.",
                "Enable onnxruntime_USE_NNAPI_BUILTIN=ON for Android NNAPI.",
            ],
        }

    # ── Run ────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        if not HAS_ORT:
            return {"success": False, "reason": "onnxruntime not installed",
                    "build_commands": self.build_commands()}

        simplified = self.simplify()
        quantized = self.quantize(simplified)

        results = {}
        for label, path in [("fp32", simplified), ("int8", quantized)]:
            if not path.exists():
                continue
            session, profile = self.make_session(path)
            results[label] = {
                "model": str(path),
                "provider_profile": profile,
                "latency": self.benchmark(session),
            }

        out = Path(self.cfg.output_dir) / "mobile_report.json"
        out.write_text(json.dumps({
            "config": asdict(self.cfg),
            "results": results,
            "build_commands": self.build_commands(),
        }, indent=2))

        return {"success": True, "report": str(out), "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(ORTMobilePipeline(ORTMobileConfig()).run(), indent=2, default=str))
