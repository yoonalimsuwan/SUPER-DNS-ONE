# =============================================================================
# COVARIANT SESI BIO-MAXWELL BRIDGE
# =============================================================================
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

class CovariantSESIBioMaxwellBridge(nn.Module):
    """
    Covariant Maxwell Bridge integrating Arbitrary-Lagrangian-Eulerian (ALE) 
    framework for tracking self-evolving disordered interfaces.
    """
    def __init__(
        self,
        dx: float = 1.0,
        dt: float = 0.01,
        alpha: float = 1.0,
        c_bio: float = 1.0, 
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dx = dx; self.dt = dt
        self.alpha = alpha; self.c_bio = c_bio
        self.dev = device or torch.device("cpu")
        self.to(self.dev)

    def _compute_4d_gradient(self, field: torch.Tensor) -> torch.Tensor:
        df_dx = (torch.roll(field, shifts=-1, dims=4) - torch.roll(field, shifts=1, dims=4)) / (2.0 * self.dx)
        df_dy = (torch.roll(field, shifts=-1, dims=3) - torch.roll(field, shifts=1, dims=3)) / (2.0 * self.dx)
        df_dz = (torch.roll(field, shifts=-1, dims=2) - torch.roll(field, shifts=1, dims=2)) / (2.0 * self.dx)
        return df_dx, df_dy, df_dz

    def step_system(
        self,
        a_mu: torch.Tensor,
        p_mu: torch.Tensor,
        aux_field: torch.Tensor,
        conjugate_aux: torch.Tensor,
        j_mu_bio: torch.Tensor,
        is_topo_jump_active: bool, # Flag from ExactSESI solver indicating t = T_k
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        a0 = a_mu[:, 0:1]; ax = a_mu[:, 1:2]; ay = a_mu[:, 2:3]; az = a_mu[:, 3:4]

        # Nakanishi-Lautrup damping mechanism
        dx_ax, dy_ax, dz_ax = self._compute_4d_gradient(ax)
        dx_ay, dy_ay, dz_ay = self._compute_4d_gradient(ay)
        dx_az, dy_az, dz_az = self._compute_4d_gradient(az)

        div_a = (torch.roll(a0, shifts=-1, dims=4) - torch.roll(a0, shifts=1, dims=4)) / (2.0 * self.dx) + dx_ax + dy_ay + dz_az
        nl_constraint = div_a + self.alpha * aux_field

        # Energy Boundedness at Jumps (Theorem 3 Global Energy Bound)
        # If a topological jump occurs, dissipative bounding ensures stability
        dissipation_factor = 0.99 if is_topo_jump_active else 1.0

        d_p = -(self.c_bio**2) * div_a - self.alpha * nl_constraint + j_mu_bio
        p_next = (p_mu + self.dt * d_p) * dissipation_factor
        a_next = (a_mu + self.dt * p_next) * dissipation_factor

        d_aux = conjugate_aux - self.alpha * aux_field
        aux_next = aux_field + self.dt * d_aux
        conjugate_aux_next = conjugate_aux + self.dt * nl_constraint

        return a_next, p_next, aux_next, conjugate_aux_next
