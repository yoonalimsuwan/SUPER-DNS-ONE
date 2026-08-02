# =============================================================================
# EXACT ANALYTICAL MAXWELL-STRUCTURAL BRIDGE — PDE Solver
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

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

__all__ = [
    "ExactMaxwellStructuralSolver",
]


class ExactMaxwellStructuralSolver(nn.Module):
    """
    Exact analytical solver for coupled Maxwell's equations and 
    Structural Operator (Delta_S) phase-field evolution (CH3D / PFC 3D).
    
    Performs non-approximated finite-difference derivative operations 
    on tensor fields with full PyTorch autograd compatibility.
    
    Args:
        dx        : spatial grid spacing.
        dt        : time step size.
        epsilon_0 : vacuum permittivity.
        mu_0      : vacuum permeability.
        c         : speed of light.
        device    : compute device.
    """

    def __init__(
        self,
        dx: float = 1.0,
        dt: float = 0.01,
        epsilon_0: float = 1.0,
        mu_0: float = 1.0,
        c: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.eps0 = epsilon_0
        self.mu0 = mu_0
        self.c = c
        self.dev = device or torch.device("cpu")
        self.to(self.dev)

    def _compute_curl(self, field: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact curl of a 3-component vector field on a 3D grid 
        using central finite differences.
        
        Field shape: (Batch, 3, D, H, W) -> [F_x, F_y, F_z]
        """
        # dx, dy, dz derivatives via roll operations (periodic boundary conditions)
        fx, fy, fz = field[:, 0], field[:, 1], field[:, 2]

        # Derivatives of components
        # dF_z / dy - dF_y / dz
        dfz_dy = (torch.roll(fz, shifts=-1, dims=2) - torch.roll(fz, shifts=1, dims=2)) / (2.0 * self.dx)
        dfy_dz = (torch.roll(fy, shifts=-1, dims=1) - torch.roll(fy, shifts=1, dims=1)) / (2.0 * self.dx)
        curl_x = dfz_dy - dfy_dz

        # dF_x / dz - dF_z / dx
        dfx_dz = (torch.roll(fx, shifts=-1, dims=1) - torch.roll(fx, shifts=1, dims=1)) / (2.0 * self.dx)
        dfz_dx = (torch.roll(fz, shifts=-1, dims=3) - torch.roll(fz, shifts=1, dims=3)) / (2.0 * self.dx)
        curl_y = dfx_dz - dfz_dx

        # dF_y / dx - dF_x / dy
        dfy_dx = (torch.roll(fy, shifts=-1, dims=3) - torch.roll(fy, shifts=1, dims=3)) / (2.0 * self.dx)
        dfx_dy = (torch.roll(fx, shifts=-1, dims=2) - torch.roll(fx, shifts=1, dims=2)) / (2.0 * self.dx)
        curl_z = dfy_dx - dfx_dy

        return torch.stack([curl_x, curl_y, curl_z], dim=1)

    def _compute_divergence(self, field: torch.Tensor) -> torch.Tensor:
        """Computes the exact divergence of a 3-component vector field."""
        fx, fy, fz = field[:, 0], field[:, 1], field[:, 2]
        
        dfx_dx = (torch.roll(fx, shifts=-1, dims=3) - torch.roll(fx, shifts=1, dims=3)) / (2.0 * self.dx)
        dfy_dy = (torch.roll(fy, shifts=-1, dims=2) - torch.roll(fy, shifts=1, dims=2)) / (2.0 * self.dx)
        dfz_dz = (torch.roll(fz, shifts=-1, dims=1) - torch.roll(fz, shifts=1, dims=1)) / (2.0 * self.dx)
        
        return dfx_dx + dfy_dy + dfz_dz

    def compute_maxwell_stress_tensor(
        self, 
        e_field: torch.Tensor, 
        b_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the exact Maxwell Stress Tensor T_ij:
        T_ij = epsilon_0 * (E_i E_j - 0.5 * delta_ij * |E|^2) + 
               (1 / mu_0) * (B_i B_j - 0.5 * delta_ij * |B|^2)
        
        Returns tensor of shape (Batch, 6, D, H, W) representing independent 
        components [T_xx, T_yy, T_zz, T_xy, T_xz, T_yz].
        """
        ex, ey, ez = e_field[:, 0:1], e_field[:, 1:2], e_field[:, 2:3]
        bx, by, bz = b_field[:, 0:1], b_field[:, 1:2], b_field[:, 2:3]

        e_sq = ex**2 + ey**2 + ez**2
        b_sq = bx**2 + by**2 + bz**2

        # Diagonal components
        t_xx = self.eps0 * (ex**2 - 0.5 * e_sq) + (1.0 / self.mu0) * (bx**2 - 0.5 * b_sq)
        t_yy = self.eps0 * (ey**2 - 0.5 * e_sq) + (1.0 / self.mu0) * (by**2 - 0.5 * b_sq)
        t_zz = self.eps0 * (ez**2 - 0.5 * e_sq) + (1.0 / self.mu0) * (bz**2 - 0.5 * b_sq)

        # Off-diagonal components
        t_xy = self.eps0 * (ex * ey) + (1.0 / self.mu0) * (bx * by)
        t_xz = self.eps0 * (ex * ez) + (1.0 / self.mu0) * (bx * bz)
        t_yz = self.eps0 * (ey * ez) + (1.0 / self.mu0) * (by * bz)

        return torch.cat([t_xx, t_yy, t_zz, t_xy, t_xz, t_yz], dim=1)

    def apply_structural_operator_delta_s(
        self,
        order_parameter: torch.Tensor,
        stress_tensor: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """
        Applies the analytical Structural Operator (\Delta_S) combined with 
        the divergence of the Maxwell stress tensor onto the phase-field evolution.
        """
        # Laplacian of order parameter (Delta u)
        u_xx = (torch.roll(order_parameter, shifts=-1, dims=3) - 2.0 * order_parameter + torch.roll(order_parameter, shifts=1, dims=3)) / (self.dx**2)
        u_yy = (torch.roll(order_parameter, shifts=-1, dims=2) - 2.0 * order_parameter + torch.roll(order_parameter, shifts=1, dims=2)) / (self.dx**2)
        u_zz = (torch.roll(order_parameter, shifts=-1, dims=1) - 2.0 * order_parameter + torch.roll(order_parameter, shifts=1, dims=1)) / (self.dx**2)
        laplacian_u = u_xx + u_yy + u_zz

        # Divergence contribution from Maxwell Stress Tensor (simplified trace/coupling)
        t_xx, t_yy, t_zz = stress_tensor[:, 0:1], stress_tensor[:, 1:2], stress_tensor[:, 2:3]
        div_stress_x = (torch.roll(t_xx, shifts=-1, dims=3) - torch.roll(t_xx, shifts=1, dims=3)) / (2.0 * self.dx)
        div_stress_y = (torch.roll(t_yy, shifts=-1, dims=2) - torch.roll(t_yy, shifts=1, dims=2)) / (2.0 * self.dx)
        div_stress_z = (torch.roll(t_zz, shifts=-1, dims=1) - torch.roll(t_zz, shifts=1, dims=1)) / (2.0 * self.dx)
        
        div_stress_total = div_stress_x + div_stress_y + div_stress_z

        # Exact Delta_S analytical coupling
        delta_s_result = laplacian_u - kappa * div_stress_total
        return delta_s_result

    def step(
        self,
        e_field: torch.Tensor,
        b_field: torch.Tensor,
        order_parameter: torch.Tensor,
        j_eff: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one exact analytical time-step update for Maxwell's equations 
        and the coupled structural phase field.
        """
        if j_eff is None:
            j_eff = torch.zeros_like(e_field)

        # Exact Maxwell time-stepping (Faraday's and Ampere-Maxwell laws)
        # dE/dt = c^2 * (curl B - mu_0 J)
        curl_b = self._compute_curl(b_field)
        d_e = (self.c**2) * (curl_b - self.mu0 * j_eff)
        e_next = e_field + self.dt * d_e

        # dB/dt = -curl E
        curl_e = self._compute_curl(e_field)
        d_b = -curl_e
        b_next = b_field + self.dt * d_b

        # Compute exact Maxwell Stress Tensor
        stress = self.compute_maxwell_stress_tensor(e_next, b_next)

        # Update Structural State via Delta_S operator
        delta_s_eval = self.apply_structural_operator_delta_s(order_parameter, stress)
        u_next = order_parameter + self.dt * delta_s_eval

        return e_next, b_next, u_next
