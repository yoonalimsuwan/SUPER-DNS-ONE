"""
nsem/runtime/dispatch.py

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026

"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import iree.runtime as ireert
from typing import Dict


class NSEMRuntime:
    """
    Dispatches each NSEM branch to its compiled backend.

    Branch -> backend contract:
      vision_latent        -> photonic   (Xanadu / PsiQuantum stack)
      cognitive_feedback   -> superconducting (Quake VMFB)
      auditory             -> fpga
      motor_control        -> asic_dac   (current-steering DAC)
      olfactory_gustatory  -> chemical   (microfluidic controller)
    """

    def __init__(self, vmfb_paths: Dict[str, str], onnx_fallback: str):
        self.sessions = {
            name: ireert.VmModule.mmap(ireert.Config(driver_name="local-task"), path)
            for name, path in vmfb_paths.items()
        }
        self.fallback = ort.InferenceSession(
            onnx_fallback, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def __call__(self, spikes: np.ndarray, memory: np.ndarray) -> dict:
        # ---- 1. Photonic branch -----------------------------------------
        vision = self.sessions["photonic"].vision_latent(spikes, memory)

        # ---- 2. Quantum cognition ---------------------------------------
        cognition = self.sessions["superconducting"].cognitive_feedback(spikes)

        # ---- 3. Everything else via the ONNX fallback -------------------
        rest = self.fallback.run(None, {"spikes": spikes, "memory_state": memory})
        return {
            "vision_latent": vision,
            "cognitive_feedback": cognition,
            "auditory": rest[1],
            "motor_control": rest[3],
            "olfactory_gustatory": rest[4],
            "memory_readout": rest[5],
        }
