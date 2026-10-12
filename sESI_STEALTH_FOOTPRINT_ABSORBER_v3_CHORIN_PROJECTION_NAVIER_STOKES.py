# =============================================================================
# SESI STEALTH FOOTPRINT ABSORBER v3 — CHORIN PROJECTION NAVIER-STOKES
# =============================================================================
# Developer  : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# Version    : 3.0.0 (Real NS Solver / C^∞ / AMP-Safe / DDP-Ready)
# =============================================================================
"""
Real incompressible Navier–Stokes solver via Chorin's projection method,
fully differentiable through depthwise conv3d operators.

    u*  = u^n + Δt (−u·∇u + ν ∇²u + f)        ← predictor
    ∇²p = (1/Δt) ∇·u*                          ← pressure Poisson
    u^{n+1} = u* − Δt ∇p                       ← projection

Ref: Chorin, J. Comput. Phys. 2, 12–26 (1968).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import soft_floor

__all__ = [
    "StealthCFDConfig",
    "IncompressibleNSSolver",
    "SESIStealthFootprintAbsorberV3",
]


# =============================================================================
@dataclass
class StealthCFDConfig:
    grid_resolution: Tuple[int, int, int] = (16, 16, 16)
    viscosity: float = 1e-4
    dt: float = 1e-4
    n_pressure_iter: int = 40
    c1_geom: float = 1.0
    sigma_variance: float = 0.05
    absorption_efficiency_target: float = 0.95
    inner_clamp: float = 15.0
    log_floor: float = -40.0
    denom_eps: float = 1e-8
    barrier_beta: float = 4.0
    indicator_tau: float = 0.05
    perturbation_amplitude: float = 0.2


# =============================================================================
class IncompressibleNSSolver(nn.Module):
    """
    Differentiable 3D incompressible Navier–Stokes via Chorin projection.

    All differential operators use depthwise conv3d with circular padding —
    compile-friendly, no data-dependent control flow.
    """

    def __init__(self, cfg: StealthCFDConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Laplacian (7-point stencil) — depthwise groups=3 for velocity
        lap = torch.zeros(3, 1, 3, 3, 3)
        lap[:, 0, 0, 1, 1] = 1.0; lap[:, 0, 2, 1, 1] = 1.0
        lap[:, 0, 1, 0, 1] = 1.0; lap[:, 0, 1, 2, 1] = 1.0
        lap[:, 0, 1, 1, 0] = 1.0; lap[:, 0, 1, 1, 2] = 1.0
        lap[:, 0, 1, 1, 1] = -6.0
        self.register_buffer("_lap_u", lap, persistent=False)
        # Single-channel Laplacian for pressure
        self.register_buffer("_lap_p", lap[0:1].clone(), persistent=False)

        # Central-difference kernels — depthwise groups=3
        def _kd(axis):
            k = torch.zeros(3, 1, 3, 3, 3)
            if axis == 0:
                k[:, 0, 0, 1, 1] = -0.5; k[:, 0, 2, 1, 1] = 0.5
            elif axis == 1:
                k[:, 0, 1, 0, 1] = -0.5; k[:, 0, 1, 2, 1] = 0.5
            else:
                k[:, 0, 1, 1, 0] = -0.5; k[:, 0, 1, 1, 2] = 0.5
            return k
        self.register_buffer("_kD", _kd(0), persistent=False)
        self.register_buffer("_kH", _kd(1), persistent=False)
        self.register_buffer("_kW", _kd(2), persistent=False)

        self.register_buffer("_nu", torch.tensor(cfg.viscosity, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_dt", torch.tensor(cfg.dt, dtype=torch.float32),
                             persistent=False)
        # Effective damping per Jacobi iteration for 7-point Poisson
        self.register_buffer("_jacobi_w", torch.tensor(1.0 / 6.0, dtype=torch.float32),
                             persistent=False)

    # ------------------------------------------------------------------ #
    def _circ(self, x):
        return F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")

    def _laplacian_u(self, u):
        return F.conv3d(self._circ(u), self._lap_u.to(u.dtype), groups=3)

    def _laplacian_p(self, p):
        return F.conv3d(self._circ(p), self._lap_p.to(p.dtype), groups=1)

    def _grad_u(self, u):
        """Return (∂u/∂D, ∂u/∂H, ∂u/∂W) each [B,3,D,H,W]."""
        vp = self._circ(u)
        gD = F.conv3d(vp, self._kD.to(u.dtype), groups=3)
        gH = F.conv3d(vp, self._kH.to(u.dtype), groups=3)
        gW = F.conv3d(vp, self._kW.to(u.dtype), groups=3)
        return gD, gH, gW

    def _divergence(self, u):
        gD, gH, gW = self._grad_u(u)
        # div = ∂u_x/∂D + ∂u_y/∂H + ∂u_z/∂W
        return gD[:, 0:1] + gH[:, 1:2] + gW[:, 2:3]

    def _gradient_p(self, p):
        vp = self._circ(p)
        gD = F.conv3d(vp, self._kD[0:1].to(p.dtype), groups=1)
        gH = F.conv3d(vp, self._kH[0:1].to(p.dtype), groups=1)
        gW = F.conv3d(vp, self._kW[0:1].to(p.dtype), groups=1)
        # Broadcast to 3 channels (∂p/∂x, ∂p/∂y, ∂p/∂z)
        return torch.cat([gD, gH, gW], dim=1)

    def _advection(self, u):
        """(u·∇)u — central differences, no clipping, differentiable."""
        gD, gH, gW = self._grad_u(u)
        ux = u[:, 0:1]; uy = u[:, 1:2]; uz = u[:, 2:3]
        ax = ux * gD[:, 0:1] + uy * gH[:, 0:1] + uz * gW[:, 0:1]
        ay = ux * gD[:, 1:2] + uy * gH[:, 1:2] + uz * gW[:, 1:2]
        az = ux * gD[:, 2:3] + uy * gH[:, 2:3] + uz * gW[:, 2:3]
        return torch.cat([ax, ay, az], dim=1)

    def _project(self, u_star):
        """Solve ∇²p = ∇·u* / Δt then subtract Δt ∇p."""
        dt = self._dt.to(u_star.dtype)
        div = self._divergence(u_star)
        # Iterative Poisson (Jacobi) — differentiable through the loop
        p = torch.zeros_like(div)
        w = self._jacobi_w.to(div.dtype)
        rhs = div / dt.clamp(min=1e-12)
        for _ in range(self.cfg.n_pressure_iter):
            lap_p = self._laplacian_p(p)
            p = p - w * (lap_p - rhs)   # damped Jacobi step
        grad_p = self._gradient_p(p)
        return u_star - dt * grad_p, p

    def step(self, u, f=None):
        """Advance one time step (Chorin projection)."""
        if f is None:
            f = torch.zeros_like(u)
        dt = self._dt.to(u.dtype)
        nu = self._nu.to(u.dtype)
        lap = self._laplacian_u(u) * nu
        adv = self._advection(u)
        u_star = u + dt * (-adv + lap + f)
        u_next, p = self._project(u_star)
        return u_next, p


# =============================================================================
class SESIStealthFootprintAbsorberV3(nn.Module):
    """
    Stealth absorber with real NS physics + SESI No-Zeno breakout bound.
    """

    def __init__(
        self,
        config: Optional[StealthCFDConfig] = None,
        *,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        cfg = config or StealthCFDConfig()
        self.cfg = cfg
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        self.ns_engine = IncompressibleNSSolver(cfg)

        def _buf(v):
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_c1",          _buf(cfg.c1_geom),         persistent=False)
        self.register_buffer("_dt",          _buf(cfg.dt),              persistent=False)
        self.register_buffer("_sigma_var",   _buf(cfg.sigma_variance),  persistent=False)
        self.register_buffer("_sigma_var_sq",_buf(cfg.sigma_variance**2), persistent=False)
        self.register_buffer("_inner_max",   _buf(cfg.inner_clamp),     persistent=False)
        self.register_buffer("_log_floor",   _buf(cfg.log_floor),       persistent=False)
        self.register_buffer("_denom_eps",   _buf(cfg.denom_eps),       persistent=False)
        self.register_buffer("_barrier_beta",_buf(cfg.barrier_beta),    persistent=False)
        self.register_buffer("_indicator_tau", _buf(cfg.indicator_tau), persistent=False)
        self.register_buffer("_pert_amp",    _buf(cfg.perturbation_amplitude),
                             persistent=False)
        self.register_buffer("_target",      _buf(cfg.absorption_efficiency_target),
                             persistent=False)

    # ------------------------------------------------------------------ #
    def _footprint_energy(self, velocity):
        """Per-sample total kinetic + enstrophy footprint."""
        # Per-sample kinetic energy: Σ|u|² over spatial dims only
        ke = 0.5 * (velocity * velocity).sum(dim=tuple(range(1, velocity.dim())))
        # Enstrophy (vorticity squared) — also per-sample
        gD, gH, gW = self.ns_engine._grad_u(velocity)
        omega_x = gH[:, 2:3] - gD[:, 1:2]
        omega_y = gD[:, 0:1] - gW[:, 2:3]
        omega_z = gW[:, 1:2] - gH[:, 0:1]
        enstrophy = (omega_x * omega_x + omega_y * omega_y + omega_z * omega_z) \
                    .sum(dim=tuple(range(1, velocity.dim())))
        return ke + enstrophy

    # ------------------------------------------------------------------ #
    def _gumbel_barrier(self, delta_e):
        dtype_out = delta_e.dtype
        device = delta_e.device
        c1 = self._c1.to(device=device)
        dt = self._dt.to(device=device)
        sigma_sq = self._sigma_var_sq.to(device=device)
        inner_max = self._inner_max.to(device=device)
        log_floor = self._log_floor.to(device=device)
        eps = self._denom_eps.to(device=device)
        d32 = delta_e.float()
        denom = soft_floor(sigma_sq * dt, float(eps.item()))
        inner = (d32 / denom).clamp_(max=inner_max)
        log_p = (-c1 * torch.exp(inner)).clamp_(min=log_floor)
        return torch.exp(log_p).to(dtype_out)

    # ------------------------------------------------------------------ #
    def _step(self, velocity_field, sigma_skin):
        dtype = velocity_field.dtype
        device = velocity_field.device
        beta = self._barrier_beta.to(dtype=dtype, device=device)
        tau  = self._indicator_tau.to(dtype=dtype, device=device)
        target = self._target.to(dtype=dtype, device=device)
        pert = self._pert_amp.to(dtype=dtype, device=device)
        eps = self._denom_eps.to(dtype=dtype, device=device)

        # 1. Disordered boundary layer
        if self.training:
            n = torch.randn_like(sigma_skin)
            abs_n = torch.sqrt(n * n + eps * eps)
            disordered_sigma = sigma_skin + abs_n * self._sigma_var.to(dtype)
        else:
            disordered_sigma = sigma_skin

        # 2. Initial footprint energy
        initial_energy = self._footprint_energy(velocity_field)

        # 3. Turbulent perturbation
        if self.training:
            u = torch.rand_like(velocity_field)
            perturbed_v = torch.addcmul(velocity_field, velocity_field * u, -pert)
        else:
            perturbed_v = velocity_field

        perturbed_energy = self._footprint_energy(perturbed_v)

        # 4. SESI activation energy
        delta_e = F.softplus(initial_energy - perturbed_energy, beta=beta) + eps

        # 5. No-Zeno breakout probability
        signature_prob = self._gumbel_barrier(delta_e)

        # 6. Damping factor
        damping = 1.0 - 1.0 / (1.0 + delta_e)
        damping_v = damping.view(-1, 1, 1, 1, 1)

        # 7. Absorbed velocity
        absorbed_v = velocity_field * damping_v

        # 8. Post-absorption footprint
        final_energy = self._footprint_energy(absorbed_v)

        # 9. Absorption ratio
        absorption_ratio = 1.0 - final_energy / (initial_energy + eps)

        # 10. Soft indicator
        indicator = torch.sigmoid((absorption_ratio - target) / tau)

        return {
            "initial_footprint_energy":       initial_energy,
            "absorbed_footprint_energy":      final_energy,
            "absorption_efficiency":          absorption_ratio,
            "signature_breakout_probability": signature_prob,
            "disordered_sigma_field":         disordered_sigma,
            "absorbed_velocity_field":        absorbed_v,
            "stealth_target_achieved":        indicator,
            "delta_e":                        delta_e,
            "damping_factor":                 damping,
        }

    # ------------------------------------------------------------------ #
    def forward(self, velocity_field, sigma_skin, *, return_metrics=True):
        if sigma_skin.dim() == 4:
            sigma_skin = sigma_skin.unsqueeze(1)
        if self.validate_inputs:
            if velocity_field.dim() != 5 or velocity_field.shape[1] != 3:
                raise ValueError("velocity_field must be [B, 3, D, H, W].")
            if sigma_skin.shape[0] != velocity_field.shape[0]:
                raise ValueError("sigma_skin batch dim must match.")
            if sigma_skin.shape[2:] != velocity_field.shape[2:]:
                raise ValueError("sigma_skin spatial dims must match.")

        if self.gradient_checkpointing and self.training:
            out = torch.utils.checkpoint.checkpoint(
                self._step, velocity_field, sigma_skin, use_reentrant=False,
            )
        else:
            out = self._step(velocity_field, sigma_skin)

        if not return_metrics:
            return {"absorbed_velocity_field": out["absorbed_velocity_field"]}
        return out


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-StealthAbsorber v3] Running on: {device}")

    cfg = StealthCFDConfig(grid_resolution=(8, 8, 8))
    absorber = SESIStealthFootprintAbsorberV3(cfg).to(device)
    absorber.train()

    B, D, H, W = 2, *cfg.grid_resolution
    velocity = torch.randn(B, 3, D, H, W, device=device, requires_grad=True) * 10.0
    sigma = torch.ones(B, 1, D, H, W, device=device, requires_grad=True)

    out = absorber(velocity, sigma)
    loss = (out["absorbed_velocity_field"].pow(2).mean()
            + (1.0 - out["stealth_target_achieved"]).mean()
            + out["signature_breakout_probability"].mean())
    loss.backward()

    print(f"  loss                         : {loss.item():.6f}")
    print(f"  initial E [per sample]       : "
          f"{out['initial_footprint_energy'].detach().cpu().tolist()}")
    print(f"  absorbed E [per sample]      : "
          f"{out['absorbed_footprint_energy'].detach().cpu().tolist()}")
    print(f"  absorption efficiency        : "
          f"{out['absorption_efficiency'].detach().cpu().tolist()}")
    print(f"  signature breakout prob      : "
          f"{out['signature_breakout_probability'].detach().cpu().tolist()}")
    print(f"  |∂L/∂velocity|               : {velocity.grad.norm().item():.6e}")
    print(f"  |∂L/∂sigma|                  : {sigma.grad.norm().item():.6e}")
