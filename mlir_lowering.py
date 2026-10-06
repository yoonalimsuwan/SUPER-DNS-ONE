"""
nsem/compile/mlir_lowering.py

Drives the ONNX -> MLIR -> backend-specific IR pipeline.

We use three MLIR dialect families:
  * torch / onnx dialects     -> front-end
  * linalg / tensor / arith   -> hardware-agnostic middle
  * quake (MQT) / catalyst    -> quantum branch
  * photonic custom dialect   -> photonic branch (Xanadu / PsiQuantum stack)
  * iree_hal                  -> HAL dispatch to concrete drivers

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026

"""
from __future__ import annotations

import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Literal


Backend = Literal["photonic", "superconducting", "fpga", "asic_dac", "chemical"]


@dataclass
class LoweringConfig:
    onnx_path: Path
    out_dir: Path
    targets: List[Backend] = field(default_factory=lambda: ["photonic", "fpga"])
    # Pass pipeline mirrors CoVA / IREE / MQT conventions
    mlir_passes: List[str] = field(default_factory=lambda: [
        # 1. Front-end -> linalg on tensors
        "--torch-onnx-to-torch",
        "--torch-to-linalg-on-tensors",
        # 2. Hardware-agnostic graph optimization
        "--linalg-fuse-elementwise-ops",
        "--linalg-generalize-named-ops",
        "--canonicalize",
        "--cse",
        # 3. Partition by backend annotation
        "--nsem-branch-partition",
        # 4. Target-specific lowering
        "--nsem-photonic-lower",
        "--nsem-quake-lower",
        "--nsem-dac-lower",
        # 5. Bufferize + lower to HAL
        "--one-shot-bufferize=bufferize-function-boundaries",
        "--iree-hal-target-backends=llvm-cpu,rocm,cuda,photonic-quake",
        "--iree-flow-dispatch-formation",
    ])
    emit_vmfb: bool = True   # IREE VMFB artifact for runtime dispatch


def lower(
    cfg: LoweringConfig,
    mlir_opt: str = "iree-opt",
    iree_compile: str = "iree-compile",
) -> dict:
    """
    Runs the full lowering.  Returns a dict of {backend: artifact_path}.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = cfg.out_dir / "nsem.mlir"

    # ---- Stage A: ONNX -> canonical MLIR --------------------------------
    subprocess.run(
        [mlir_opt, str(cfg.onnx_path), *cfg.mlir_passes, "-o", str(mlir_path)],
        check=True,
    )

    # ---- Stage B: split into per-backend MLIR modules -------------------
    artifacts: dict = {}
    for tgt in cfg.targets:
        tgt_mlir = cfg.out_dir / f"nsem_{tgt}.mlir"
        subprocess.run(
            [mlir_opt, str(mlir_path),
             f"--nsem-extract-backend={tgt}",
             "--canonicalize", "--cse",
             "-o", str(tgt_mlir)],
            check=True,
        )
        # ---- Stage C: compile to VMFB / binary --------------------------
        if cfg.emit_vmfb:
            vmfb = cfg.out_dir / f"nsem_{tgt}.vmfb"
            subprocess.run(
                [iree_compile, str(tgt_mlir),
                 f"--iree-hal-target-backends={_hal_target(tgt)}",
                 "-o", str(vmfb)],
                check=True,
            )
            artifacts[tgt] = vmfb
        else:
            artifacts[tgt] = tgt_mlir
    return artifacts


def _hal_target(backend: Backend) -> str:
    return {
        "photonic":       "photonic-quake",
        "superconducting": "quake",
        "fpga":           "llvm-cpu",
        "asic_dac":       "llvm-cpu",
        "chemical":       "llvm-cpu",
    }[backend]
