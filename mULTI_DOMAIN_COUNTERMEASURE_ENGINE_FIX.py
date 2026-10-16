# =============================================================================
# PRODUCTION MULTI-DOMAIN COUNTERMEASURE ENGINE v2 — REAL PHYSICS
# v2.0.1 (FIXED: vectorized gravimetry, trajectory shape mismatch)
# =============================================================================
# Developer  : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# =============================================================================

from __future__ import annotations
import math
from typing import Tuple, Dict, NamedTuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import (
    SI,
    newton_gravity, j2_perturbation_potential,
    exponential_atmosphere_rho, aerodynamic_drag,
    cosmic_muon_flux, muon_energy_loss_dEdx,
    gaussian_beam_intensity, thermal_damage_time,
    beer_lambert_transmittance,
    soft_floor,
)

__all__ = [
    "TargetState",
    "DefenseParameters",
    "DifferentiableCountermeasureEngineV2",
]


class TargetState(NamedTuple):
    position: torch.Tensor
    velocity: torch.Tensor
    mass: torch.Tensor


class DefenseParameters(NamedTuple):
    gravimeter_weights: torch.Tensor
    muon_sensor_grid: torch.Tensor
    dew_focal_controls: torch.Tensor


class DifferentiableCountermeasureEngineV2(nn.Module):

    def __init__(
        self,
        grid_resolution: Tuple[int, int, int] = (16, 16, 16),
        *,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self.nx, self.ny, self.nz = grid_resolution
        self.validate_inputs = bool(validate_inputs)

        x = torch.linspace(-SI.R_earth * 0.01, SI.R_earth * 0.01, self.nx)
        y = torch.linspace(-SI.R_earth * 0.01, SI.R_earth * 0.01, self.ny)
        z = torch.linspace(0.0, 5e4, self.nz)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        grid_coords = torch.stack([xx, yy, zz], dim=-1)
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        self.register_buffer("eps", torch.tensor(1e-8), persistent=False)

    def _grid_coords_like(self, device, dtype):
        return self.grid_coords.to(device=device, dtype=dtype)

    # === FIXED: fully vectorized (no Python loop, no .item()) ===
    def compute_quantum_gravimetry_loss(self, target, params):
        dtype = target.position.dtype
        device = target.position.device
        grid = self._grid_coords_like(device, dtype)                    # (nx,ny,nz,3)

        B = target.position.shape[0]

        # Broadcast: grid [1,nx,ny,nz,3] − target.pos [B,1,1,1,3]
        r_vec = grid.unsqueeze(0) - target.position.view(B, 1, 1, 1, 3)

        F_earth = newton_gravity(SI.M_earth, 1.0, r_vec)
        m_view  = target.mass.view(B, 1, 1, 1, 1)
        F_target = newton_gravity(m_view, 1.0, r_vec)

        lat = torch.asin((grid[..., 2] / SI.R_earth).clamp(-1.0, 1.0))
        r_norm = (grid[..., 2].pow(2) + SI.R_earth ** 2).sqrt()
        U2 = j2_perturbation_potential(r_norm, lat)                      # [nx,ny,nz]
        U2_b = U2.unsqueeze(0).expand(B, -1, -1, -1)

        sensor = F_earth.norm(dim=-1) + F_target.norm(dim=-1) + U2_b.abs()
        w_scalar = params.gravimeter_weights.mean()
        return -(sensor * w_scalar).mean()

    def compute_muon_tomography_loss(self, target, params):
        dtype = target.position.dtype
        device = target.position.device
        alt = target.position[:, 2].clamp(min=0.0, max=2e4)
        E_mu = torch.tensor(10.0, dtype=dtype, device=device)
        rho_air = exponential_atmosphere_rho(alt)

        L_col_gcm2 = (target.mass / 1e3) * 1.0
        dEdx = muon_energy_loss_dEdx(E_mu, rho=1.0)
        E_loss = (dEdx * L_col_gcm2).clamp(min=0.0)
        E_out = (E_mu - E_loss).clamp(min=0.1)
        shadow = 1.0 - E_out / E_mu

        flux = cosmic_muon_flux(alt, E_out)
        signature = flux * shadow
        muon_gain = params.muon_sensor_grid.mean()
        return F.mse_loss(signature * muon_gain, torch.full_like(signature, 0.85))

    # === FIXED: shape [B,H,3] vs [B] ===
    def compute_predictive_trajectory_loss(self, target, predicted):
        dtype = target.position.dtype
        device = target.position.device
        r = target.position.norm(dim=-1).clamp(min=SI.R_earth)         # [B]
        v = target.velocity.norm(dim=-1)                                # [B]
        mu = SI.G * SI.M_earth
        inv_a = (2.0 / r - v * v / mu).clamp(min=1e-12)
        a = 1.0 / inv_a                                                 # [B]

        alt = target.position[:, 2].clamp(min=0.0)
        rho = exponential_atmosphere_rho(alt)
        D = aerodynamic_drag(rho, target.velocity, Cd=0.3, A=1.0).squeeze(-1)  # [B]

        # FIXED: เปรียบ predicted radius (mean over horizon) กับ a
        pred_r = predicted.norm(dim=-1).mean(dim=-1)                    # [B]
        a_residual = (pred_r - a).abs() + 0.001 * D
        return F.smooth_l1_loss(a_residual, torch.zeros_like(a_residual))

    def compute_directed_energy_grid_loss(self, target, params):
        dtype = target.position.dtype
        device = target.position.device
        lam = 1.06e-6
        w0 = 0.3
        P_laser = 1e6
        r = (params.dew_focal_controls - target.position).norm(dim=-1)
        z = target.position[:, 2].clamp(min=1.0)
        I_beam = gaussian_beam_intensity(P_laser, w0, r, z, lam)
        alpha = torch.as_tensor(1e-5, dtype=dtype, device=device)
        I_beam = I_beam * beer_lambert_transmittance(alpha, z)
        thickness = torch.as_tensor(0.02, dtype=dtype, device=device)
        t_dmg = thermal_damage_time(
            I_beam,
            T_melt=1800.0, T_0=300.0,
            rho=2700.0, c_p=900.0, k_cond=200.0,
            thickness=thickness,
        )
        return F.softplus(t_dmg - 2.0, beta=1.0).mean()

    def forward(
        self,
        target: TargetState,
        params: DefenseParameters,
        predicted_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_grav = self.compute_quantum_gravimetry_loss(target, params)
        loss_muon = self.compute_muon_tomography_loss(target, params)
        loss_traj = self.compute_predictive_trajectory_loss(target, predicted_trajectory)
        loss_dew  = self.compute_directed_energy_grid_loss(target, params)

        total = 1.0 * loss_grav + 0.8 * loss_muon + 1.5 * loss_traj + 1.2 * loss_dew

        metrics = {
            "loss_gravimetry":     loss_grav.detach(),
            "loss_muon":           loss_muon.detach(),
            "loss_trajectory":     loss_traj.detach(),
            "loss_dew":            loss_dew.detach(),
            "total_loss":          total.detach(),
        }
        return total, metrics


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-Countermeasure v2] Running on: {device}")

    engine = DifferentiableCountermeasureEngineV2((8, 8, 8)).to(device)
    engine.train()

    B = 2
    target = TargetState(
        position=torch.tensor([[1e5, 2e5, 3e4],
                               [1.2e5, 2.2e5, 2.8e4]], device=device),
        velocity=torch.tensor([[2000.0, 500.0, -100.0],
                               [1800.0, 400.0, -80.0]], device=device),
        mass=torch.tensor([1500.0, 1800.0], device=device),
    )
    params = DefenseParameters(
        gravimeter_weights=torch.randn(4, device=device),
        muon_sensor_grid=torch.randn(4, device=device),
        dew_focal_controls=(target.position + 100.0).detach().clone().requires_grad_(True),
    )
    predicted = target.position.unsqueeze(1).expand(B, 20, 3).clone() \
                + torch.randn(B, 20, 3, device=device) * 100.0

    total, metrics = engine(target, params, predicted)
    total.backward()

    print(f"  total loss         : {metrics['total_loss'].item():.6f}")
    print(f"  gravimetry         : {metrics['loss_gravimetry'].item():.6e}")
    print(f"  muon tomography    : {metrics['loss_muon'].item():.6e}")
    print(f"  predictive AI      : {metrics['loss_trajectory'].item():.6e}")
    print(f"  directed energy    : {metrics['loss_dew'].item():.6e}")
    print(f"  |∂L/∂dew_focal|    : {params.dew_focal_controls.grad.norm().item():.6e}")
    print("Status: real G + J2 + Bethe-Bloch + Gaussian beam; gradients flow.")
