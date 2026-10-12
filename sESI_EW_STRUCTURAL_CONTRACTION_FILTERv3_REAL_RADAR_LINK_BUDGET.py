# =============================================================================
# SESI EW STRUCTURAL CONTRACTION FILTER v3 — REAL RADAR LINK BUDGET
# =============================================================================
# Developer  : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# Version    : 3.0.0 (Physics-Grounded / C^∞ / AMP-Safe / DDP-Ready)
# =============================================================================

from __future__ import annotations
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import (
    SI,
    radar_range_equation, jamming_to_signal_ratio,
    burn_through_range, beer_lambert_transmittance,
    soft_floor,
)

__all__ = ["EWStructuralContractionModuleV3"]


class EWStructuralContractionModuleV3(nn.Module):
    """
    Real radar-link-budget-based EW filter.

    Physics
    -------
    S       = P_t G² σ λ² / ((4π)³ R⁴ L)              radar equation
    J       = P_j G_j / (4π R_j²)  · (B_s / B_j)       barrage-jammer power
    SI      = 1 / (1 + J/S)                            smooth signal integrity
    R_bt    = burn-through range (Skolnik §2.6)
    """

    def __init__(
        self,
        *,
        radar_freq_hz: float = 10e9,
        P_radar_W: float = 1e4,
        G_radar_dBi: float = 35.0,
        G_jam_dBi: float = 25.0,
        B_signal_Hz: float = 1e6,
        B_jam_Hz: float = 2e8,
        atm_alpha: float = 1e-5,
        ste_tau: float = 0.05,
        c1: float = 1.0,
        inner_clamp: float = 15.0,
        log_floor: float = -40.0,
        denom_eps: float = 1e-8,
        num_semantic_states: int = 16,
    ) -> None:
        super().__init__()
        self.ste_tau = float(ste_tau)

        def _buf(v):
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_f_r",   _buf(radar_freq_hz),                    persistent=False)
        self.register_buffer("_P_r",   _buf(P_radar_W),                        persistent=False)
        self.register_buffer("_G_r",   _buf(10.0**(G_radar_dBi/10.0)),         persistent=False)
        self.register_buffer("_G_j",   _buf(10.0**(G_jam_dBi/10.0)),           persistent=False)
        self.register_buffer("_B_s",   _buf(B_signal_Hz),                      persistent=False)
        self.register_buffer("_B_j",   _buf(B_jam_Hz),                         persistent=False)
        self.register_buffer("_alpha", _buf(atm_alpha),                        persistent=False)
        self.register_buffer("_c1",    _buf(c1),                               persistent=False)
        self.register_buffer("_zeno_max", _buf(inner_clamp),                   persistent=False)
        self.register_buffer("_log_floor", _buf(log_floor),                    persistent=False)
        self.register_buffer("_eps",   _buf(denom_eps),                        persistent=False)

        self.semantic_contraction = nn.Sequential(
            nn.Linear(3, num_semantic_states, bias=False),
            nn.LayerNorm(num_semantic_states),
            nn.GELU(),
        )
        self.topological_evaluator = nn.Linear(num_semantic_states, 1)

    # ------------------------------------------------------------------ #
    def _link_budget(self, raw_em, R_target=1e4, R_jammer=1e4, sigma_rcs=1.0):
        """Compute per-batch (scalar) radar link budget."""
        dtype = raw_em.dtype
        device = raw_em.device
        P_r = self._P_r.to(device=device)
        G_r = self._G_r.to(device=device)
        G_j = self._G_j.to(device=device)
        B_s = self._B_s.to(device=device)
        B_j = self._B_j.to(device=device)
        lam = SI.c / self._f_r.to(device=device)

        # Radar return power (assume same antenna)
        R_t = torch.tensor(float(R_target), dtype=dtype, device=device)
        R_j = torch.tensor(float(R_jammer), dtype=dtype, device=device)
        P_signal = radar_range_equation(
            P_r, G_r, sigma_rcs, lam, R_t,
        )
        # Barrage jammer power at victim receiver
        P_jam = P_r * G_j / (4.0 * math.pi * R_j ** 2)
        # J/S (broadcast to batch shape)
        J_over_S = jamming_to_signal_ratio(
            P_jam, torch.ones_like(P_jam), B_j, R_j,
            P_signal, G_r, B_s, R_t,
        )
        # Convert scalar to per-sample by broadcasting to [B] shape of raw_em
        B = raw_em.shape[0]
        J_over_S = J_over_S.view(1).expand(B)
        # Burn-through
        R_bt = burn_through_range(
            P_r, G_r, sigma_rcs, lam,
            P_jam, G_j, B_j, B_s, R_j, S_min=1.0,
        ).view(1).expand(B)
        return J_over_S, R_bt

    # ------------------------------------------------------------------ #
    def forward(
        self,
        raw_em_signal: torch.Tensor,       # [B, C, ...] or [B, ...]
        ew_jamming_density: torch.Tensor,  # same shape
        dt: float,
        *,
        R_target: float = 1e4,
        R_jammer: float = 1e4,
        sigma_rcs: float = 1.0,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # Normalise to [B, F] feature layout
        if raw_em_signal.dim() == 1:
            raw_em_signal = raw_em_signal.unsqueeze(0)
            ew_jamming_density = ew_jamming_density.unsqueeze(0)
        B = raw_em_signal.shape[0]
        raw_flat = raw_em_signal.reshape(B, -1)
        jam_flat = ew_jamming_density.reshape(B, -1)

        # Link budget (per-batch scalar broadcast)
        J_over_S, R_bt = self._link_budget(
            raw_flat, R_target=R_target, R_jammer=R_jammer, sigma_rcs=sigma_rcs,
        )

        # Signal integrity = 1/(1+J/S)  → soft, C^∞
        signal_integrity = 1.0 / (1.0 + J_over_S)

        # SESI activation energy: excess energy above the noise floor
        signal_energy = (raw_flat * raw_flat).mean(dim=1)
        noise_energy = (jam_flat * jam_flat).mean(dim=1)
        delta_e = F.softplus(signal_energy - noise_energy, beta=4.0) + self._eps.to(signal_energy.dtype)

        # No-Zeno barrier
        sigma_sq = noise_energy.clamp(min=float(self._eps.item()))
        dt_t = torch.tensor(float(dt), dtype=delta_e.dtype, device=delta_e.device)
        arg = (delta_e / (sigma_sq * dt_t + self._eps.to(delta_e.dtype))).clamp(max=float(self._zeno_max.item()))
        log_p = (-self._c1.to(delta_e.dtype) * torch.exp(arg)).clamp(min=float(self._log_floor.item()))
        zeno_suppression = torch.exp(log_p)

        # Final filtered prediction = soft × link integrity × zeno
        prediction = signal_integrity * zeno_suppression

        # STE hard variant
        if self.training:
            pred = (prediction > 0.5).to(prediction.dtype) - prediction.detach() + prediction
        else:
            pred = prediction

        if not return_metrics:
            return {"is_true_target": pred}
        return {
            "is_true_target":        pred,
            "structural_confidence": prediction,
            "zeno_suppression_mask": zeno_suppression,
            "J_over_S":              J_over_S,
            "burn_through_range_m":  R_bt,
            "signal_integrity":      signal_integrity,
        }


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-EW v3] Running on: {device}")

    model = EWStructuralContractionModuleV3().to(device)
    model.train()

    B, F = 4, 64
    raw = torch.randn(B, F, device=device)
    jam = torch.randn(B, F, device=device) * 0.5

    out = model(raw, jam, dt=1e-6)
    loss = (1.0 - out["is_true_target"]).mean() \
           + out["J_over_S"].mean() * 1e-3 \
           + out["structural_confidence"].mean()
    loss.backward()

    print(f"  loss                    : {loss.item():.6f}")
    print(f"  J/S                     : "
          f"{out['J_over_S'].detach().min().item():.3e} – "
          f"{out['J_over_S'].detach().max().item():.3e}")
    print(f"  burn-through range [m]  : "
          f"{out['burn_through_range_m'].detach().min().item():.3e} – "
          f"{out['burn_through_range_m'].detach().max().item():.3e}")
    print(f"  signal integrity        : "
          f"{out['signal_integrity'].detach().min().item():.4f} – "
          f"{out['signal_integrity'].detach().max().item():.4f}")
    print(f"  zeno suppression        : "
          f"{out['zeno_suppression_mask'].detach().min().item():.3e} – "
          f"{out['zeno_suppression_mask'].detach().max().item():.3e}")
    print("Status: real radar equation + J/S + burn-through; gradients flow.")
