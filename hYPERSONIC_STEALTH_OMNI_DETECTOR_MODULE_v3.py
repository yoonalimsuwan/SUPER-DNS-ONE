# =============================================================================
# HYPERSONIC STEALTH OMNI-DETECTOR MODULE v3 (SESI FRAMEWORK)
# Physics-grounded — Saha + Fay-Riddell + collisional ε_r + |Γ|²
# =============================================================================
# Developer  : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# Version    : 3.0.0 (Physics-Grounded / C^∞ / AMP-Safe / DDP-Ready)
# =============================================================================
"""
Physics chain
-------------
V∞, T∞, p∞  →  shock jump (Rankine-Hugoniot)
            →  post-shock (n_n, T2)
            →  Saha ionization → n_e, T_e
            →  ω_p, ν_en
            →  ε_r(ω) = ε' − i ε''
            →  power reflection |Γ|²  ⇒  detectable radar return
            →  Fay-Riddell stagnation heat flux (aero-thermal signature)
            →  Gumbel-Sigmoid classifier on per-sample pooled features
"""

from __future__ import annotations
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import (
    SI,
    rankine_hugoniot,
    saha_electron_density,
    plasma_frequency,
    electron_neutral_collision_freq,
    plasma_complex_permittivity,
    plasma_reflection_coefficient,
    fay_riddell_stagnation_heat_flux,
    soft_floor,
)

__all__ = ["HypersonicStealthOmniDetectorV3"]


class HypersonicStealthOmniDetectorV3(nn.Module):
    """
    Physics-grounded hypersonic/stealth detector.

    Parameters
    ----------
    radar_freq_hz : float
        Radar illumination frequency (default X-band, 10 GHz).
    nose_radius : float
        Vehicle nose radius in metres (for Fay-Riddell heat flux).
    body_diameter : float
        Reference length scale (for surface density estimate).
    c1, ste_tau, barrier_beta, zeno_clamp, log_floor : float
        SESI barrier + Gumbel-STE hyperparameters (as in v2).
    """

    def __init__(
        self,
        radar_freq_hz: float = 10e9,
        nose_radius: float = 0.15,
        body_diameter: float = 1.0,
        *,
        c1: float = 1.0,
        ste_tau: float = 0.05,
        barrier_beta: float = 50.0,
        zeno_clamp: float = 15.0,
        log_floor: float = -25.0,
        denom_eps: float = 1e-6,
        hidden_dim: int = 16,
        use_hard_ste: bool = True,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not (math.isfinite(c1) and c1 > 0.0):
            raise ValueError("c1 must be > 0.")
        if not (math.isfinite(ste_tau) and ste_tau > 0.0):
            raise ValueError("ste_tau must be > 0.")
        if not (math.isfinite(radar_freq_hz) and radar_freq_hz > 0.0):
            raise ValueError("radar_freq_hz must be > 0.")
        if not (math.isfinite(nose_radius) and nose_radius > 0.0):
            raise ValueError("nose_radius must be > 0.")

        self.c1              = float(c1)
        self.ste_tau         = float(ste_tau)
        self.barrier_beta    = float(barrier_beta)
        self.zeno_clamp      = float(zeno_clamp)
        self.log_floor       = float(log_floor)
        self.denom_eps       = float(denom_eps)
        self.hidden_dim      = int(hidden_dim)
        self.use_hard_ste    = bool(use_hard_ste)
        self.validate_inputs = bool(validate_inputs)

        self.register_buffer("_omega_radar",
                             torch.tensor(2.0 * math.pi * radar_freq_hz, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_R_nose", torch.tensor(nose_radius, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_L_body", torch.tensor(body_diameter, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_c1", torch.tensor(c1, dtype=torch.float32), persistent=False)
        self.register_buffer("_tau", torch.tensor(ste_tau, dtype=torch.float32), persistent=False)
        self.register_buffer("_beta", torch.tensor(barrier_beta, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_zeno_max", torch.tensor(zeno_clamp, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_log_floor", torch.tensor(log_floor, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_eps", torch.tensor(denom_eps, dtype=torch.float32),
                             persistent=False)

        # Multi-physics feature weights (learnable)
        self.w_aero    = nn.Parameter(torch.tensor(1.2))
        self.w_plasma  = nn.Parameter(torch.tensor(1.5))
        self.w_heat    = nn.Parameter(torch.tensor(1.0))

        self.classifier_head = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _gumbel_sigmoid_ste(logits, tau, hard, training):
        if not training:
            return torch.sigmoid(logits / tau)
        u = torch.rand_like(logits).clamp_(1e-7, 1.0 - 1e-7)
        g = -torch.log(-torch.log(u))
        y_soft = torch.sigmoid((logits + g) / tau)
        if not hard:
            return y_soft
        y_hard = (y_soft > 0.5).to(y_soft.dtype)
        return y_hard - y_soft.detach() + y_soft

    def _sesi_barrier(self, delta_e, sigma_sq, dt):
        dtype_out = delta_e.dtype
        device    = delta_e.device
        x32 = delta_e.float()
        s32 = sigma_sq.float()
        c1        = self._c1.to(device=device)
        zeno_max  = self._zeno_max.to(device=device)
        log_floor = self._log_floor.to(device=device)
        eps       = self._eps.to(device=device)
        if not isinstance(dt, torch.Tensor):
            dt_t = torch.tensor(float(dt), dtype=torch.float32, device=device)
        else:
            dt_t = dt.float().to(device)
        denom = s32 * dt_t + eps
        arg   = (x32 / denom).clamp_(max=zeno_max)
        inner = torch.exp(arg)
        log_p = (-c1 * inner).clamp_(min=log_floor)
        return torch.exp(log_p).to(dtype_out)

    # ------------------------------------------------------------------ #
    def _plasma_signature(self, V_mag, T_inf, p_inf, n_total):
        """
        Full physics chain from freestream (V, T, p) to radar-visible
        plasma sheath reflection and stagnation heat flux.

        All quantities per spatial cell: [B, 1, D, H, W].
        """
        # Speed of sound and Mach number
        a_inf = torch.sqrt(SI.gamma_air * SI.R_air * T_inf.clamp(min=1.0))
        M     = V_mag / a_inf.clamp(min=1.0)

        # Rankine-Hugoniot jump
        _, p2_p1, rho2_rho1, T2_T1 = rankine_hugoniot(M)

        # Post-shock state
        T2 = T_inf * T2_T1
        rho_inf = p_inf / (SI.R_air * T_inf.clamp(min=1.0))
        rho2 = rho_inf * rho2_rho1

        # Neutral number density behind shock (mean air molecule)
        n_n = rho2 / SI.m_air

        # Saha ionization
        n_e = saha_electron_density(T2, n_n)

        # Frequencies
        omega_p = plasma_frequency(n_e)
        nu_en   = electron_neutral_collision_freq(n_e, T2, n_n)

        # Complex permittivity at radar ω
        omega = self._omega_radar.to(n_e.dtype)
        eps_r, eps_i = plasma_complex_permittivity(omega, omega_p, nu_en)

        # Power reflection from plasma sheath
        R_sheath = plasma_reflection_coefficient(eps_r, eps_i)

        # Stagnation heat flux (per-cell, using local density behind shock)
        R_nose = self._R_nose.to(n_e.dtype)
        q_s = fay_riddell_stagnation_heat_flux(V_mag, rho2, R_nose)

        return R_sheath, q_s, n_e, T2, M

    # ------------------------------------------------------------------ #
    def forward(
        self,
        velocity_field: torch.Tensor,          # [B, 3, D, H, W]  m/s
        thermal_field: torch.Tensor,           # [B, 1, D, H, W]  K
        em_field: torch.Tensor,                # [B, Ce, D, H, W] (unused here)
        plasma_density: torch.Tensor,          # [B, 1, D, H, W]  kg/m³
        propulsion_em_coupling: torch.Tensor,  # [B, 1, D, H, W]  arbitrary
        pressure_field: torch.Tensor = None,   # [B, 1, D, H, W]  Pa  (optional)
        dt: float | torch.Tensor = 1e-3,
        *,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.validate_inputs:
            if velocity_field.dim() != 5 or velocity_field.shape[1] != 3:
                raise ValueError("velocity_field must be [B, 3, D, H, W].")
            for name, t in (
                ("thermal_field", thermal_field),
                ("plasma_density", plasma_density),
                ("propulsion_em_coupling", propulsion_em_coupling),
            ):
                if t.dim() != 5 or t.shape[0] != velocity_field.shape[0]:
                    raise ValueError(f"{name} must be 5D and match batch dim.")
                if t.shape[2:] != velocity_field.shape[2:]:
                    raise ValueError(f"{name} spatial dims must match velocity_field.")

        dtype  = velocity_field.dtype
        device = velocity_field.device
        beta   = self._beta.to(device=device)
        tau    = self._tau.to(device=device)
        eps    = self._eps.to(device=device)

        # Freestream primitive fields
        V_mag = torch.sqrt((velocity_field * velocity_field).sum(1, keepdim=True) + 1e-6)
        T_inf = thermal_field
        p_inf = pressure_field if pressure_field is not None else \
                plasma_density * SI.R_air * T_inf

        # --- Physics chain ---
        R_sheath, q_s, n_e, T2, M = self._plasma_signature(
            V_mag, T_inf, p_inf, plasma_density,
        )

        # Total activation energy for the SESI barrier (post-shock enthalpy +
        # EMF-induced excess) — physically an energy per unit volume.
        delta_e_field = (rho := plasma_density) * SI.c_p_air * (T2 - T_inf) \
                        + propulsion_em_coupling * propulsion_em_coupling

        # Total variance: thermal + kinetic + EMF  (energy per unit volume scale)
        sigma_sq = (SI.k_B * T_inf
                    + 0.5 * plasma_density * V_mag * V_mag
                    + propulsion_em_coupling * propulsion_em_coupling)

        prob_bound = self._sesi_barrier(delta_e_field, sigma_sq, dt)

        # Per-sample spatial pooling
        spatial_dims = (2, 3, 4)
        pooled_aero   = (R_sheath * prob_bound).mean(dim=spatial_dims) * self.w_aero
        pooled_plasma = (n_e       * prob_bound).mean(dim=spatial_dims) * self.w_plasma
        pooled_heat   = (q_s       * prob_bound).mean(dim=spatial_dims) * self.w_heat

        features = torch.stack([pooled_aero, pooled_plasma, pooled_heat], dim=-1)
        logits   = self.classifier_head(features)
        soft_conf = torch.sigmoid(logits / tau)
        pred = self._gumbel_sigmoid_ste(logits, tau, self.use_hard_ste, self.training)

        if not return_metrics:
            return {"is_hypersonic_drone": pred}
        return {
            "is_hypersonic_drone":       pred,
            "classification_confidence": soft_conf,
            "extracted_features":        features,
            "prob_bound":                prob_bound,
            "logits":                    logits,
            "plasma_reflection":         R_sheath,
            "stagnation_heat_flux":      q_s,
            "electron_density":          n_e,
            "mach_number":               M,
        }


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-HypersonicDetector v3] Running on: {device}")

    model = HypersonicStealthOmniDetectorV3().to(device)
    model.train()

    B, D, H, W = 2, 12, 12, 12
    velocity    = torch.randn(B, 3, D, H, W, device=device) * 1500.0   # ~Mach 5-8
    thermal     = torch.randn(B, 1, D, H, W, device=device) * 20.0 + 250.0
    em          = torch.randn(B, 3, D, H, W, device=device)
    density     = torch.rand(B, 1, D, H, W, device=device) * 0.1 + 0.01   # 10–100 g/m³
    emf_coupling = torch.randn(B, 1, D, H, W, device=device) * 0.05

    out = model(velocity, thermal, em, density, emf_coupling, dt=1e-3)
    loss = out["is_hypersonic_drone"].float().mean() \
           + out["logits"].float().pow(2).mean()
    loss.backward()

    print(f"  loss                     : {loss.item():.6f}")
    print(f"  plasma reflection range  : "
          f"{out['plasma_reflection'].detach().min().item():.3e} – "
          f"{out['plasma_reflection'].detach().max().item():.3e}")
    print(f"  stagnation q_s [W/m²]    : "
          f"{out['stagnation_heat_flux'].detach().min().item():.3e} – "
          f"{out['stagnation_heat_flux'].detach().max().item():.3e}")
    print(f"  n_e [m⁻³]                : "
          f"{out['electron_density'].detach().min().item():.3e} – "
          f"{out['electron_density'].detach().max().item():.3e}")
    print(f"  Mach                     : "
          f"{out['mach_number'].detach().min().item():.3f} – "
          f"{out['mach_number'].detach().max().item():.3f}")
    print("Status: full physics chain executed; gradients flow.")
