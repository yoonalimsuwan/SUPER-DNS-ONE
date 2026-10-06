"""
nsem/compile/export.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
from typing import Dict, Optional, Tuple


def export_onnx(
    model: nn.Module,
    sample_spikes: torch.Tensor,
    sample_memory: Optional[torch.Tensor],
    out_path: Path,
    opset: int = 18,
    dynamic_axes: Optional[Dict] = None,
) -> Path:
    """
    Export NSEM to ONNX with dynamic batch and memory axes.

    opset 18 is required for:
      * LayerNormalization
      * GELU decomposition
      * ScatterND (used by the microfluidic routing mask)
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dynamic_axes is None:
        dynamic_axes = {
            "spikes": {0: "batch"},
            "memory_state": {0: "batch"},
            "vision_latent": {0: "batch"},
            "auditory": {0: "batch"},
            "cognitive_feedback": {0: "batch"},
            "motor_control": {0: "batch"},
            "olfactory_gustatory": {0: "batch"},
            "memory_readout": {0: "batch"},
        }
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample_spikes, sample_memory),
            str(out_path),
            opset_version=opset,
            input_names=["spikes", "memory_state"],
            output_names=[
                "vision_latent", "auditory", "cognitive_feedback",
                "motor_control", "olfactory_gustatory", "memory_readout",
            ],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
    return out_path
