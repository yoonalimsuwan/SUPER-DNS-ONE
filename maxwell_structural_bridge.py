# =============================================================================
# MAXWELL STRUCTURAL BRIDGE — Differentiable Electromagnetic Surrogate Module
# EVOLUTION ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

MAXWELL_BRIDGE_VERSION: str = "1.0.0"

__all__ = [
    "MAXWELL_BRIDGE_VERSION",
    "MaxwellSurrogateOperator",
    "StructuralMaxwellBridge",
]


class MaxwellSurrogateOperator(nn.Module):
    """
    Differentiable Neural Operator surrogate for Maxwell's Equations.
    
    Approximates electric field (E) and magnetic field (B) evolution
    coupled with structural tensor fields (Delta_S) without runtime latency
    from external solvers. Fully compatible with PyTorch autograd.
    
    Args:
        in_channels  : number of input field channels (e.g., density, velocity, charge).
        hidden_dim   : internal feature dimension.
        device       : compute device.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dev = device or torch.device("cpu")
        
        # Spatiotemporal convolution layers acting as surrogate Maxwell solver
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(hidden_dim),
        )
        
        self.processor = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Predicts 6 components: E_x, E_y, E_z (Electric) and B_x, B_y, B_z (Magnetic)
        self.decoder = nn.Conv3d(hidden_dim, 6, kernel_size=3, padding=1)
        
        self.to(self.dev)

    def forward(self, field_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for electromagnetic field evolution.
        
        Args:
            field_tensor : Tensor of shape (Batch, in_channels, D, H, W)
        Returns:
            EM tensor of shape (Batch, 6, D, H, W) -> [E_xyz, B_xyz]
        """
        x = self.encoder(field_tensor)
        x = x + self.processor(x)  # Residual connection
        out = self.decoder(x)
        return out


class StructuralMaxwellBridge(nn.Module):
    """
    Bridge connecting Structural Calculus (Delta_S) and Electromagnetic Fields.
    
    Allows simultaneous computation of fluid/phase dynamics (CH3D, PFC 3D, Langevin)
    and radar/electromagnetic wave propagation within the same training loop.
    
    Args:
        kappa     : gradient coupling parameter.
        device    : compute device.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.dev = device or torch.device("cpu")
        self.maxwell_op = MaxwellSurrogateOperator(in_channels=4, hidden_dim=64, device=self.dev)

    def compute_coupled_dynamics(
        self,
        structural_state: torch.Tensor,
        velocity_field: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the interaction between structural state, fluid velocity,
        and induced electromagnetic fields.
        
        Args:
            structural_state : Tensor representing phase or density field (B, 1, D, H, W).
            velocity_field   : Tensor representing velocity components (B, 3, D, H, W).
        Returns:
            Dictionary containing EM fields and coupled energy metrics.
        """
        # Concatenate structural state and velocity to form 4-channel input
        if structural_state.shape[1] == 1:
            inputs = torch.cat([structural_state, velocity_field], dim=1)
        else:
            inputs = velocity_field

        # Predict electromagnetic fields via surrogate operator
        em_fields = self.maxwell_op(inputs)
        
        e_field = em_fields[:, :3, ...]  # E_x, E_y, E_z
        b_field = em_fields[:, 3:, ...]  # B_x, B_y, B_z
        
        # Energy density calculation (Poynting-like coupling proxy)
        magnetic_energy = 0.5 * (b_field ** 2).sum(dim=1, keepdim=True)
        electric_energy = 0.5 * (e_field ** 2).sum(dim=1, keepdim=True)
        
        total_em_energy = electric_energy + magnetic_energy
        
        return {
            "electric_field": e_field,
            "magnetic_field": b_field,
            "em_energy_density": total_em_energy,
            "coupled_score": F.softplus(total_em_energy.mean() * self.kappa),
        }

    def verify_bridge(self) -> bool:
        """Sanity check for tensor shapes and gradient flow."""
        dummy_struct = torch.randn(1, 1, 16, 16, 16, device=self.dev, requires_grad=True)
        dummy_vel = torch.randn(1, 3, 16, 16, 16, device=self.dev, requires_grad=True)
        
        result = self.compute_coupled_dynamics(dummy_struct, dummy_vel)
        loss = result["coupled_score"]
        loss.backward()
        
        return dummy_struct.grad is not None and not torch.isnan(loss)
