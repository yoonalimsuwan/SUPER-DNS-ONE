# =============================================================================
# COVARIANT 4-VECTOR POTENTIAL MAXWELL-STRUCTURAL BRIDGE
# SUPER DNS ONE Cluster / ONE Ecosystem
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
    "CovariantMaxwellStructuralBridge",
]


class CovariantMaxwellStructuralBridge(nn.Module):
    """
    Covariant Formulated Maxwell-Structural Bridge using 4-Vector Potential ($A_\mu$).
    
    Implements Nakanishi-Lautrup auxiliary field formalism for classical covariant 
    gauge fixing (Lorenz gauge constraint damping) instead of quantum BV ghosts,
    coupling electromagnetic stress tensors directly with structural phase-field 
    evolution (\Delta_S).
    
    Args:
        dx        : spatial grid spacing.
        dt        : time step size.
        alpha     : gauge fixing parameter (damping coefficient).
        epsilon_0 : vacuum permittivity.
        mu_0      : vacuum permeability.
        c         : speed of light.
        device    : compute device.
    """

    def __init__(
        self,
        dx: float = 1.0,
        dt: float = 0.01,
        alpha: float = 1.0,
        epsilon_0: float = 1.0,
        mu_0: float = 1.0,
        c: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.alpha = alpha
        self.eps0 = epsilon_0
        self.mu0 = mu_0
        self.c = c
        self.dev = device or torch.device("cpu")
        self.to(self.dev)

    def _compute_4d_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """
        Computes spatial gradients (dx, dy, dz) for 4-vector components.
        Field shape: (Batch, 4, D, H, W) -> [A_0 (scalar), A_x, A_y, A_z]
        """
        df_dx = (torch.roll(field, shifts=-1, dims=4) - torch.roll(field, shifts=1, dims=4)) / (2.0 * self.dx)
        df_dy = (torch.roll(field, shifts=-1, dims=3) - torch.roll(field, shifts=1, dims=3)) / (2.0 * self.dx)
        df_dz = (torch.roll(field, shifts=-1, dims=2) - torch.roll(field, shifts=1, dims=2)) / (2.0 * self.dx)
        return df_dx, df_dy, df_dz

    def compute_field_tensor_and_gauge(
        self, 
        a_mu: torch.Tensor, 
        aux_field: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the Magnetic Field Tensor and applies Covariant Gauge Fixing
        via the Nakanishi-Lautrup formalism.
        """
        a0 = a_mu[:, 0:1]  
        ax = a_mu[:, 1:2]
        ay = a_mu[:, 2:3]
        az = a_mu[:, 3:4]

        # Spatial gradients for Vector potential
        dx_ax, dy_ax, dz_ax = self._compute_4d_gradient(ax)
        dx_ay, dy_ay, dz_ay = self._compute_4d_gradient(ay)
        dx_az, dy_az, dz_az = self._compute_4d_gradient(az)

        # Magnetic field components: B = \nabla \times A
        bx = dy_az - dz_ay
        by = dz_ax - dx_az
        bz = dx_ay - dy_ax
        b_field = torch.cat([bx, by, bz], dim=1)

        # Lorenz Gauge condition: \partial^\mu A_\mu
        div_a = (
            (torch.roll(a0, shifts=-1, dims=4) - torch.roll(a0, shifts=1, dims=4)) / (2.0 * self.dx) +
            dx_ax + dy_ay + dz_az
        )
        
        # Nakanishi-Lautrup constraint damping
        nl_gauge_constraint = div_a + self.alpha * aux_field

        return b_field, div_a, nl_gauge_constraint

    def step_system(
        self,
        a_mu: torch.Tensor,
        p_mu: torch.Tensor,
        aux_field: torch.Tensor,
        conjugate_aux: torch.Tensor,
        order_parameter: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one time-step update coupling classical covariant electromagnetism
        with the Structural Operator \Delta_S.
        """
        # 1. Gauge constraints and B-field
        b_field, div_a, nl_constraint = self.compute_field_tensor_and_gauge(a_mu, aux_field)

        # 2. Momentum and Potential Evolution
        d_p = -(self.c**2) * div_a - self.alpha * nl_constraint
        p_next = p_mu + self.dt * d_p
        a_next = a_mu + self.dt * (p_next / (self.eps0 * self.c**2))

        # 3. Auxiliary Field Evolution (Constraint Damping Dynamics)
        d_aux = conjugate_aux - self.alpha * aux_field
        aux_next = aux_field + self.dt * d_aux
        conjugate_aux_next = conjugate_aux + self.dt * nl_constraint

        # 4. Corrected E-field Calculation: E = -\nabla\phi - \partial A / \partial t
        dx_a0, dy_a0, dz_a0 = self._compute_4d_gradient(a_next[:, 0:1])
        
        # Finite difference for the time derivative of Vector Potential
        dt_ax = (a_next[:, 1:2] - a_mu[:, 1:2]) / self.dt
        dt_ay = (a_next[:, 2:3] - a_mu[:, 2:3]) / self.dt
        dt_az = (a_next[:, 3:4] - a_mu[:, 3:4]) / self.dt

        ex = -dx_a0 - dt_ax
        ey = -dy_a0 - dt_ay
        ez = -dz_a0 - dt_az
        e_field = torch.cat([ex, ey, ez], dim=1)

        # 5. Maxwell Stress Trace
        e_sq = (e_field**2).sum(dim=1, keepdim=True)
        b_sq = (b_field**2).sum(dim=1, keepdim=True)
        stress_trace = self.eps0 * e_sq + (1.0 / self.mu0) * b_sq

        # 6. Apply Structural Operator (\Delta_S)
        u_xx = (torch.roll(order_parameter, shifts=-1, dims=4) - 2.0 * order_parameter + torch.roll(order_parameter, shifts=1, dims=4)) / (self.dx**2)
        u_yy = (torch.roll(order_parameter, shifts=-1, dims=3) - 2.0 * order_parameter + torch.roll(order_parameter, shifts=1, dims=3)) / (self.dx**2)
        u_zz = (torch.roll(order_parameter, shifts=-1, dims=2) - 2.0 * order_parameter + torch.roll(order_parameter, shifts=1, dims=2)) / (self.dx**2)
        laplacian_u = u_xx + u_yy + u_zz

        delta_s_eval = laplacian_u - 0.5 * torch.gradient(stress_trace, dim=4)[0]
        u_next = order_parameter + self.dt * delta_s_eval

        return a_next, p_next, aux_next, conjugate_aux_next, u_next
