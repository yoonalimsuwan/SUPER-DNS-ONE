"""
nsem/hardware/chemical.py

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


class MicrofluidicPulseModel(nn.Module):
    """
    Differentiable model of a microfluidic neurotransmitter delivery channel.

    Maps the NSEM chemical head output to:
      * volumetric flow rate (nL/s)
      * pulse duration (ms)
      * bolus concentration (µM)
      * estimated receptor occupancy (fraction of max)

    The forward pass is a differentiable approximation of the
    convection-diffusion equation in the channel, sufficient for
    gradient-based optimization of delivery policies.
    """

    def __init__(
        self,
        n_receptors: int = 128,
        max_flow_nl_s: float = 50.0,
        max_bolus_um: float = 200.0,
    ):
        super().__init__()
        self.n = n_receptors
        self.max_flow = max_flow_nl_s
        self.max_bolus = max_bolus_um
        # Learned channel resistance / diffusion time constants
        self.log_tau = nn.Parameter(torch.zeros(n_receptors))
        # Receptor affinity (Kd in µM), learned per receptor
        self.log_kd = nn.Parameter(torch.zeros(n_receptors))

    def forward(self, chemical_command: torch.Tensor) -> dict:
        """
        chemical_command: [B, n_receptors] in [0, 1] (sigmoid output).
        """
        # Flow rate and bolus from the command
        flow = chemical_command * self.max_flow               # nL/s
        bolus = chemical_command * self.max_bolus             # µM

        # First-order transport lag (differentiable)
        tau = F.softplus(self.log_tau).clamp(min=1e-3)        # s
        # Discrete lag update (Euler step, one step for simplicity)
        delivered = bolus / (1.0 + tau)

        # Hill-Langmuir receptor occupancy
        kd = F.softplus(self.log_kd).clamp(min=1e-3)
        occupancy = delivered / (delivered + kd)

        return {
            "flow_nl_s": flow,
            "bolus_um": bolus,
            "delivered_um": delivered,
            "receptor_occupancy": occupancy,
            # Safety: total volume delivered per pulse
            "volume_nl": flow * 0.050,  # 50 ms pulse
        }
