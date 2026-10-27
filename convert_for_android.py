"""
ONNX → ORT Mobile format converter for Android deployment.

Produces:
  • model.ort   — mobile-optimized format (NNAPI-compatible)
  • nnapi_config.json — NNAPI flags reference for Kotlin side

Usage:
    python convert_for_android.py --onnx exported/sesi_cfd.onnx \
           --out app/src/main/assets/model.ort
"""

import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--optimization-level", type=int, default=99)  # all
    args = p.parse_args()

    try:
        from onnxruntime.tools.convert_onnx_to_ort import convert_onnx_to_ort
    except ImportError:
        print("ERROR: onnxruntime >= 1.14 required.")
        print("  pip install onnxruntime")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    convert_onnx_to_ort(
        args.onnx,
        str(out_path),
        optimization_level=args.optimization_level,
    )
    print(f"✓ Converted → {out_path}")

    # Write NNAPI reference config
    nnapi_ref = {
        "flags": {
            "NNAPI_FLAG_USE_FP16": "0x001 — allow FP16 execution",
            "NNAPI_FLAG_USE_NCHW": "0x002 — use NCHW layout (some accelerators prefer)",
            "NNAPI_FLAG_CPU_DISABLED": "0x004 — no CPU fallback (fail if NNAPI unavailable)",
            "NNAPI_FLAG_CPU_ONLY": "0x008 — force CPU via NNAPI (testing)",
        },
        "recommended": {
            "production": "NNAPI_FLAG_USE_FP16",
            "debug": "NNAPI_FLAG_CPU_ONLY",
            "latency_critical": "NNAPI_FLAG_USE_FP16 | NNAPI_FLAG_CPU_DISABLED",
        },
        "notes": [
            "Dynamic shapes reduce NNAPI support; prefer static batch=1",
            "First NNAPI run compiles the model — cache via OrtSession cache dir",
            "Not all ops are NNAPI-supported; ORT falls back to CPU automatically",
        ],
    }
    ref_path = out_path.parent / "nnapi_config.json"
    with open(ref_path, "w") as f:
        json.dump(nnapi_ref, f, indent=2)
    print(f"✓ NNAPI reference → {ref_path}")


if __name__ == "__main__":
    main()
