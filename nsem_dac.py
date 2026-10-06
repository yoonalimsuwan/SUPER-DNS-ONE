"""
nsem/hardware/dac.py

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DACSpec:
    bits: int = 8
    full_scale_current_ua: float = 100.0     # 100 µA per channel
    lsb_current_ua: float = 0.39             # ~ 100 / 2^8
    compliance_v: float = 10.0               # HV BCD output stage
    channels: int = 256
    # Safety envelope (per ACNS / IEC 60601-2-10 style limits)
    max_charge_density_uc_cm2: float = 30.0  # conservative for subdural
    max_current_density_a_m2: float = 20.0
    pulse_width_us: float = 200.0
    interphase_gap_us: float = 50.0


class QuantizedDAC(torch.autograd.Function):
    """
    Straight-through estimator for current-steering DAC.

    Forward:  quantize to N bits + clamp to safety envelope
    Backward: identity (STE), with gradient masking outside the envelope
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,              # [B, channels] in [-1, 1]
        spec: DACSpec,
    ) -> torch.Tensor:
        levels = (1 << spec.bits) - 1
        # Map [-1, 1] -> [0, levels]
        q = torch.round((x.clamp(-1, 1) + 1.0) * 0.5 * levels)
        # Physical current in µA
        i_ua = q / levels * spec.full_scale_current_ua
        # Charge per phase (µC) = I * t
        q_uc = i_ua * (spec.pulse_width_us * 1e-6) * 1e3  # µA·µs -> pC, then -> µC

        # Safety envelope on charge density
        area_cm2 = 1e-4  # assume 0.1 mm^2 electrode
        q_density = q_uc / area_cm2
        ctx.save_for_backward(x)
        ctx.spec = spec
        ctx.q_density = q_density
        # Clamp charge density (this is the safety interlock at the analog level)
        safe_mask = (q_density <= spec.max_charge_density_uc_cm2).float()
        return i_ua * safe_mask

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        spec = ctx.spec
        # STE with envelope masking
        inside = ((x.abs() <= 1.0) & (ctx.q_density <= spec.max_charge_density_uc_cm2))
        return grad_output * inside.float(), None


class NeuralStimulatorFrontend(nn.Module):
    """
    Full AFE + DAC + charge-balancing output stage.

    Mirrors the architecture of published bidirectional neuromodulation
    chipsets: 64-channel AFE, 4-channel current stimulator, HV-compliant
    output stage, closed-loop charge cancellation.
    """

    def __init__(self, spec: DACSpec):
        super().__init__()
        self.spec = spec
        # Learned per-channel calibration (offset + gain trim)
        self.gain_trim = nn.Parameter(torch.ones(spec.channels))
        self.offset_trim = nn.Parameter(torch.zeros(spec.channels))
        # Charge-balance controller (learned to minimize residual DC)
        self.balance_ctrl = nn.Sequential(
            nn.Linear(spec.channels, spec.channels),
            nn.Tanh(),
        )

    def forward(self, motor_command: torch.Tensor) -> dict:
        """
        motor_command: [B, channels] in [-1, 1] from the NSEM motor head.
        """
        # Trim
        x = motor_command * self.gain_trim + self.offset_trim
        # Learned charge balance (adds a small corrective second phase)
        residual = x - self.balance_ctrl(x)
        # Quantize + safety clamp
        i_ua = QuantizedDAC.apply(residual, self.spec)
        # Biphasic reconstruction (cathodic then anodic)
        cathodic = i_ua
        anodic = -i_ua  # ideally cancels; residual handled by balance_ctrl
        return {
            "cathodic_ua": cathodic,
            "anodic_ua": anodic,
            "residual_charge_uc": (cathodic + anodic) * self.spec.pulse_width_us * 1e-6,
            "compliance_v": self.spec.compliance_v,
        }
