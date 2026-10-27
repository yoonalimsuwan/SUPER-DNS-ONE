"""
=============================================================================
SESI TFLITE CONVERTER — Android / Edge Deployment
PyTorch → (ai_edge_torch | ONNX) → TFLite with Quantization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

CONVERSION PATHS
────────────────
Path A: ai_edge_torch (Google, recommended for PyTorch)
    pip install ai-edge-torch
    → torch.export() → ai_edge_torch.export() → .tflite

Path B: ONNX bridge (fallback)
    pip install onnx onnx-tf tensorflow
    → torch.onnx.export() → onnx_tf → TFLiteConverter

FFT LIMITATION
──────────────
torch.fft.rfft2 / irfftn are NOT supported by TFLite.
Use CoreMLCompatibleCFDSurrogate (conv-based, no FFT) for FNO models.
This module detects FFT ops and warns/fails gracefully.

QUANTIZATION MODES
──────────────────
  fp32        — full precision (largest, most accurate)
  fp16        — half precision (~2x smaller)
  dynamic     — dynamic range int8 for weights
  int8        — full integer, needs representative data
  int8_hybrid — int8 weights, fp32 activations
"""

from __future__ import annotations

import io
import os
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("sesi.tflite")

__all__ = [
    "TFLiteConvertConfig",
    "TFLiteConverter",
    "detect_fft_ops",
    "create_android_wrapper",
]


# ─── Optional dependencies ───────────────────────────────────────────
try:
    import ai_edge_torch
    HAS_AI_EDGE = True
except ImportError:
    HAS_AI_EDGE = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


@dataclass
class TFLiteConvertConfig:
    model_path: str = "./checkpoints/best.pt"
    output_path: str = "./tflite/model.tflite"

    # Input spec (must match model)
    input_shape: List[int] = field(default_factory=lambda: [1, 3, 64, 64])
    input_name: str = "input"

    # Quantization
    quantize: str = "fp32"          # fp32 | fp16 | dynamic | int8 | int8_hybrid
    calibration_samples: int = 100

    # Conversion backend
    backend: str = "auto"           # auto | ai_edge | onnx

    # FFT safety
    allow_fft_fallback: bool = False  # fail if FFT detected unless True

    # Android wrapper
    generate_android_wrapper: bool = True


# ═══════════════════════════════════════════════════════════════════════
# FFT DETECTION — critical safety check
# ═══════════════════════════════════════════════════════════════════════

def detect_fft_ops(model: nn.Module, sample_input: torch.Tensor) -> List[str]:
    """
    Trace the model and detect FFT operations that TFLite cannot handle.

    Returns list of FFT op names found (empty = safe).
    """
    found_fft = []

    try:
        model.eval()
        traced = torch.jit.trace(model, sample_input, strict=False)
        graph_str = str(traced.graph)

        fft_ops = ["fft_fft", "fft_rfft", "fft_irfft", "fft_fft2",
                   "fft_rfft2", "fft_irfft2", "fft_fftn", "fft_ifftn"]
        for op in fft_ops:
            if op in graph_str:
                found_fft.append(op)
    except Exception as e:
        logger.warning(f"FFT detection trace failed: {e}")

    return found_fft


# ═══════════════════════════════════════════════════════════════════════
# CALIBRATION DATA GENERATOR (for int8 quantization)
# ═══════════════════════════════════════════════════════════════════════

def _make_representative_dataset(
    input_shape: List[int],
    n_samples: int = 100,
    data_source: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Generate representative dataset for int8 calibration.

    HONEST NOTE: synthetic random data gives WRONG quantization ranges.
    For production, pass data_source that yields real input samples
    matching your inference distribution.
    """
    if data_source is not None:
        return [data_source(i) for i in range(n_samples)]

    if not HAS_NP:
        return []

    logger.warning(
        "Using SYNTHETIC calibration data. int8 accuracy will be unreliable. "
        "Provide real data via data_source for production."
    )
    dataset = []
    for i in range(n_samples):
        x = np.random.randn(*input_shape).astype(np.float32)
        dataset.append({"input": x})
    return dataset


# ═══════════════════════════════════════════════════════════════════════
# MAIN CONVERTER
# ═══════════════════════════════════════════════════════════════════════

class TFLiteConverter:
    """
    Converts a PyTorch SESI model to TFLite.

    Automatically selects the best backend:
      • ai_edge_torch if available (best for PyTorch)
      • ONNX bridge otherwise
      • Falls back to error message if neither available
    """

    def __init__(self, cfg: TFLiteConvertConfig):
        self.cfg = cfg
        Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────
    def load_model(self, model_class: nn.Module, device: str = "cpu") -> nn.Module:
        """Load a PyTorch model from checkpoint."""
        checkpoint = torch.load(self.cfg.model_path, map_location=device)

        # Support both raw state_dict and wrapped checkpoint
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_class.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_class.load_state_dict(checkpoint["state_dict"])
        else:
            model_class.load_state_dict(checkpoint)

        model_class.eval()
        logger.info(f"Loaded model from {self.cfg.model_path}")
        return model_class

    # ── Path A: ai_edge_torch ──────────────────────────────────────────
    def _convert_ai_edge(self, model: nn.Module) -> bytes:
        """Convert using Google's ai_edge_torch (native PyTorch → TFLite)."""
        if not HAS_AI_EDGE:
            raise RuntimeError("ai_edge_torch not installed. pip install ai-edge-torch")

        sample = torch.randn(*self.cfg.input_shape)

        with torch.no_grad():
            exported = ai_edge_torch.export(model, (sample,))

        tflite_bytes = exported.buffer()
        logger.info(f"ai_edge_torch conversion OK ({len(tflite_bytes)/1e6:.2f} MB)")
        return tflite_bytes

    # ── Path B: ONNX bridge ────────────────────────────────────────────
    def _convert_onnx(self, model: nn.Module) -> bytes:
        """Convert via ONNX → TensorFlow → TFLite."""
        if not HAS_TF:
            raise RuntimeError("tensorflow not installed for ONNX path")

        sample = torch.randn(*self.cfg.input_shape)

        # Step 1: PyTorch → ONNX
        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            model,
            sample,
            onnx_buffer,
            input_names=[self.cfg.input_name],
            output_names=["output"],
            opset_version=13,
            dynamic_axes={
                self.cfg.input_name: {0: "batch"},
                "output": {0: "batch"},
            },
        )
        onnx_buffer.seek(0)
        logger.info("ONNX export OK")

        # Step 2: ONNX → TF SavedModel → TFLite
        try:
            from onnx_tf.backend import prepare
            import onnx
            onnx_model = onnx.load_model_from_string(onnx_buffer.read())
            tf_rep = prepare(onnx_model)

            saved_model_dir = Path(self.cfg.output_path).parent / "_tmp_savedmodel"
            tf_rep.export_graph(str(saved_model_dir))

            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            self._apply_quantization(converter)
            tflite_bytes = converter.convert()

            # Cleanup
            import shutil
            shutil.rmtree(saved_model_dir, ignore_errors=True)

            logger.info(f"ONNX→TFLite OK ({len(tflite_bytes)/1e6:.2f} MB)")
            return tflite_bytes

        except ImportError as e:
            raise RuntimeError(f"ONNX path needs onnx + onnx-tf: {e}")

    # ── Quantization config ────────────────────────────────────────────
    def _apply_quantization(self, converter) -> None:
        """Apply quantization settings to a TF TFLiteConverter."""
        mode = self.cfg.quantize

        if mode == "fp32":
            pass  # default

        elif mode == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif mode == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif mode in ("int8", "int8_hybrid"):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            dataset = _make_representative_dataset(
                self.cfg.input_shape, self.cfg.calibration_samples
            )

            def representative_gen():
                for sample in dataset:
                    yield [sample[self.cfg.input_name]]

            converter.representative_dataset = representative_gen

            if mode == "int8":
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            # int8_hybrid keeps fp32 activations (default)

        else:
            raise ValueError(f"Unknown quantize mode: {mode}")

        logger.info(f"Quantization mode: {mode}")

    # ── Main convert entry ─────────────────────────────────────────────
    def convert(
        self,
        model: nn.Module,
        data_source: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Convert model to TFLite. Returns report dict.
        """
        result = {
            "success": False,
            "output_path": None,
            "size_mb": None,
            "backend_used": None,
            "fft_warnings": [],
            "notes": [],
        }

        # 1. FFT safety check
        sample = torch.randn(*self.cfg.input_shape)
        fft_ops = detect_fft_ops(model, sample)
        if fft_ops:
            msg = f"FFT ops detected: {fft_ops}. TFLite cannot run these."
            if not self.cfg.allow_fft_fallback:
                result["fft_warnings"].append(msg + " ABORTED. Use conv-safe surrogate.")
                logger.error(msg)
                return result
            else:
                result["fft_warnings"].append(msg + " Proceeding (may fail at convert).")
                logger.warning(msg)

        # 2. Select backend
        backend = self.cfg.backend
        if backend == "auto":
            backend = "ai_edge" if HAS_AI_EDGE else "onnx"
        result["backend_used"] = backend

        # 3. Convert
        try:
            if backend == "ai_edge":
                tflite_bytes = self._convert_ai_edge(model)
            elif backend == "onnx":
                tflite_bytes = self._convert_onnx(model)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as e:
            result["notes"].append(f"Conversion failed: {e}")
            logger.error(f"Conversion failed: {e}")
            return result

        # 4. Save
        out_path = Path(self.cfg.output_path)
        out_path.write_bytes(tflite_bytes)
        result["success"] = True
        result["output_path"] = str(out_path)
        result["size_mb"] = len(tflite_bytes) / 1e6

        # 5. Android wrapper
        if self.cfg.generate_android_wrapper:
            wrapper_path = create_android_wrapper(out_path, self.cfg)
            result["android_wrapper"] = wrapper_path

        logger.info(f"TFLite saved → {out_path} ({result['size_mb']:.2f} MB)")
        return result

    # ── Validate converted model ───────────────────────────────────────
    def validate(self, model: nn.Module, n_tests: int = 5) -> Dict[str, float]:
        """Compare PyTorch vs TFLite outputs numerically."""
        if not HAS_TF or not HAS_NP:
            return {"error": -1.0, "note": "tensorflow/numpy required"}

        interpreter = tf.lite.Interpreter(model_path=self.cfg.output_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        max_diffs = []
        for _ in range(n_tests):
            x_np = np.random.randn(*self.cfg.input_shape).astype(np.float32)
            x_torch = torch.from_numpy(x_np)

            with torch.no_grad():
                model.eval()
                y_torch = model(x_torch).numpy()

            # Handle int8 input type
            if input_details[0]["dtype"] == np.int8:
                scale, zp = input_details[0]["quantization"]
                x_input = (x_np / scale + zp).astype(np.int8)
            else:
                x_input = x_np

            interpreter.set_tensor(input_details[0]["index"], x_input)
            interpreter.invoke()
            y_tflite = interpreter.get_tensor(output_details[0]["index"])

            if output_details[0]["dtype"] == np.int8:
                scale, zp = output_details[0]["quantization"]
                y_tflite = (y_tflite.astype(np.float32) - zp) * scale

            # Flatten both
            diff = np.abs(y_torch.flatten() - y_tflite.flatten())
            max_diffs.append(diff.max() if diff.size > 0 else 0.0)

        return {
            "max_abs_diff": float(np.max(max_diffs)),
            "mean_max_diff": float(np.mean(max_diffs)),
            "n_tests": n_tests,
        }


# ═══════════════════════════════════════════════════════════════════════
# ANDROID WRAPPER GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def create_android_wrapper(tflite_path: Path, cfg: TFLiteConvertConfig) -> str:
    """Generate a Kotlin inference wrapper for Android."""
    wrapper_dir = Path(cfg.output_path).parent / "android"
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    kotlin_code = f'''package org.mspsnetwork.sesi

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * SESI TFLite Inference Wrapper for Android
 * Model: {tflite_path.name}
 * Input shape: {cfg.input_shape}
 * Quantization: {cfg.quantize}
 *
 * Usage:
 *   val model = SesiTFLiteModel(context)
 *   val output = model.predict(inputArray)
 *   model.close()
 */
class SesiTFLiteModel(context: Context) {{

    private var interpreter: Interpreter? = null
    private val inputShape = intArrayOf({", ".join(map(str, cfg.input_shape))})

    init {{
        val modelBuffer = loadModelFile(context, "{tflite_path.name}")
        val options = Interpreter.Options().apply {{
            setNumThreads(4)
            setUseNNAPI(true)  // Enable Android NNAPI acceleration
        }}
        interpreter = Interpreter(modelBuffer, options)
    }}

    private fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {{
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }}

    fun predict(input: Array<FloatArray>): FloatArray {{
        val output = FloatArray(/* output size */) {{ 0f }}
        interpreter?.run(input, output)
        return output
    }}

    fun close() {{
        interpreter?.close()
        interpreter = null
    }}
}}
'''
    wrapper_path = wrapper_dir / "SesiTFLiteModel.kt"
    wrapper_path.write_text(kotlin_code)

    # Also copy model into assets structure
    gradle_snippet = '''
// Add to app/build.gradle:
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'  // optional GPU
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
'''
    (wrapper_dir / "gradle_snippet.txt").write_text(gradle_snippet)

    logger.info(f"Android wrapper → {wrapper_path}")
    return str(wrapper_path)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI TFLite Converter — Smoke Test")
    print("=" * 72)
    print(f"  ai_edge_torch available: {HAS_AI_EDGE}")
    print(f"  tensorflow available:    {HAS_TF}")

    # Build a small conv model (TFLite-safe, no FFT)
    class SafeConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 3, 3, padding=1),
            )
        def forward(self, x):
            return self.net(x)

    model = SafeConvModel()
    cfg = TFLiteConvertConfig(
        output_path="/tmp/sesi_test/model.tflite",
        input_shape=[1, 3, 32, 32],
        quantize="fp32",
    )

    converter = TFLiteConverter(cfg)

    # FFT detection test
    fft_ops = detect_fft_ops(model, torch.randn(*cfg.input_shape))
    print(f"\n  [1] FFT detection on safe model: {fft_ops or 'none found (GOOD)'}")

    # Conversion test (only if backend available)
    if HAS_AI_EDGE or HAS_TF:
        result = converter.convert(model)
        print(f"  [2] Conversion success: {result['success']}")
        print(f"      Backend: {result['backend_used']}")
        if result['size_mb']:
            print(f"      Size: {result['size_mb']:.2f} MB")
    else:
        print("\n  [2] No backend available — install ai-edge-torch or tensorflow")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • FFT models need conv-safe surrogate before TFLite conversion")
    print("  • int8 quantization needs REAL calibration data")
    print("  • NNAPI acceleration is device-dependent")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
