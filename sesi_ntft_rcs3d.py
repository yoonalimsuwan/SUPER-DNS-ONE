# =============================================================================
# Near-to-Far Field Transformation:Radar Cross Section (RCS) (SESI)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

import math
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn

class PhysicalConstants:
    # ... [Remains the same] ...
    C0: float = 299792458.0
    MU0: float = 4.0 * math.pi * 1e-7
    EPS0: float = 1.0 / (C0**2 * MU0)
    ETA0: float = math.sqrt(MU0 / EPS0)

class PiecewiseDFTAccumulator3D(nn.Module):
    """
    Upgraded Real-time Discrete Fourier Transform (DFT) Accumulator.
    Supports Piecewise Operational Construction by allowing resets 
    when a macroscopic structural topological jump occurs.
    """
    def __init__(self, target_freq_hz: float, dt: float, field_shape: Tuple[int, int, int], device: torch.device):
        super().__init__()
        self.target_freq = target_freq_hz
        self.dt = dt
        self.omega = 2.0 * math.pi * target_freq_hz
        self.device = device
        self.field_shape = field_shape
        self.reset_accumulators()

    def reset_accumulators(self) -> None:
        """Re-centers the reference chart accumulation following a topological jump."""
        self.real_parts = {k: torch.zeros(self.field_shape, device=self.device) for k in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}
        self.imag_parts = {k: torch.zeros(self.field_shape, device=self.device) for k in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}
        self.step_count = 0

    def update(self, fields_t: Dict[str, torch.Tensor], current_time: float, has_jumped: bool = False) -> None:
        """
        Accumulates continuous time-domain fields. 
        If a topological jump occurs, the accumulator undergoes a piecewise reset.
        """
        if has_jumped:
            self.reset_accumulators()

        phase = self.omega * current_time
        cos_val = math.cos(phase) * self.dt
        sin_val = math.sin(phase) * self.dt

        for key in fields_t:
            if key in self.real_parts:
                self.real_parts[key] += fields_t[key] * cos_val
                self.imag_parts[key] -= fields_t[key] * sin_val 
        self.step_count += 1

    def get_phasors(self) -> Dict[str, torch.Tensor]:
        phasors = {}
        for key in self.real_parts:
            phasors[key] = torch.complex(self.real_parts[key], self.imag_parts[key])
        return phasors

class StructuralNTFTRCS3D(nn.Module):
    # ... [The entire StructuralNTFTRCS3D logic remains structurally the same] ...
    # The integration of the upgraded DFT handles the Zeno constraints inherently
    # before providing the E_phasors and H_phasors to this class.
    pass
