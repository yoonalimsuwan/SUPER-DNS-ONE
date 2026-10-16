# =============================================================================
# OMNI-SPECTRAL DRONE vs BIOLOGICAL DISCRIMINATOR v3
# v3.0.1 (FIXED: dead methods removed, acoustic shape [B])
# =============================================================================
# Developer  : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# =============================================================================

from __future__ import annotations
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import (
    SI,
    radar_range_equation,
    rotor_blade_passing_freq,
    doppler_shift,
    soft_floor,
)

__all__ = ["OmniTargetDetectorV3"]


class OmniTargetDetectorV3(nn.Module):

    def __init__(
        self,
        *,
        c1: float = 1.0,
        ste_tau: float = 0.05,
        k_b: float = SI.k_B,
        sigma_drone: float = SI.sigma_drone,
        sigma_bird: float = SI.sigma_bird,
        rotor_rpm: float = 6000.0,
        n_blades: int = 2,
        carrier_hz: float = 40000.0,
        barrier_beta: float = 50.0,
        zeno_clamp: float = 15.0,
        log_floor: float = -25.0,
        denom_eps: float = 1e-6,
        hidden_dim: int = 16,
        use_hard_ste: bool = True,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.ste_tau = float(ste_tau)
        self.use_hard_ste = bool(use_hard_ste)
        self.validate_inputs = bool(validate_inputs)

        bpf = rotor_blade_passing_freq(
            torch.tensor(rotor_rpm, dtype=torch.float32), n_blades,
        )

        def _buf(v):
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_c1",         _buf(c1),             persistent=False)
        self.register_buffer("_tau",        _buf(ste_tau),        persistent=False)
        self.register_buffer("_k_b",        _buf(k_b),            persistent=False)
        self.register_buffer("_sig_drone",  _buf(sigma_drone),    persistent=False)
        self.register_buffer("_sig_bird",   _buf(sigma_bird),     persistent=False)
        self.register_buffer("_bpf",        _buf(bpf.item()),     persistent=False)
        self.register_buffer("_carrier",    _buf(carrier_hz),     persistent=False)
        self.register_buffer("_beta",       _buf(barrier_beta),   persistent=False)
        self.register_buffer("_zeno_max",   _buf(zeno_clamp),     persistent=False)
        self.register_buffer("_log_floor",  _buf(log_floor),      persistent=False)
        self.register_buffer("_eps",        _buf(denom_eps),      persistent=False)

        self.w_rcs      = nn.Parameter(torch.tensor(1.0))
        self.w_acoustic = nn.Parameter(torch.tensor(1.0))
        self.w_doppler  = nn.Parameter(torch.tensor(1.0))

        self.classifier_head = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    # === FIXED: ลบ _acoustic_feature และ _rcs_feature ที่เป็น dead code ===

    def _doppler_feature(self, velocity_field):
        V = velocity_field.norm(dim=1, keepdim=True)
        f0 = self._bpf.to(V.dtype)
        c_wave = 343.0
        f_obs = f0 * c_wave / (c_wave - V).clamp(min=1.0)
        return (f_obs - f0).abs()

    def forward(
        self,
        velocity_field: torch.Tensor,
        thermal_field: torch.Tensor,
        em_field: torch.Tensor,
        plasma_density: torch.Tensor,
        propulsion_em_coupling: torch.Tensor,
        dt: float | torch.Tensor = 1e-3,
        *,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.validate_inputs:
            if velocity_field.dim() != 5 or velocity_field.shape[1] != 3:
                raise ValueError("velocity_field must be [B, 3, D, H, W].")
            B0 = velocity_field.shape[0]
            for name, t in (("thermal", thermal_field),
                            ("em", em_field),
                            ("density", plasma_density),
                            ("emf", propulsion_em_coupling)):
                if t.dim() != 5 or t.shape[0] != B0:
                    raise ValueError(f"{name} must be [B, ...] and match batch.")

        dtype = velocity_field.dtype
        device = velocity_field.device
        tau = self._tau.to(device=device)
        beta = self._beta.to(device=device)
        eps = self._eps.to(device=device)

        B = velocity_field.shape[0]

        # Spectrum along last spatial axis
        em_flat = em_field.reshape(B, em_field.shape[1], -1)
        spec = torch.fft.rfft(em_flat, dim=-1).abs()
        F_axis = spec.shape[-1]

        f = torch.linspace(0.0, 20000.0, F_axis, dtype=dtype, device=device)
        bpf = self._bpf.to(dtype=dtype, device=device)
        bw = 5.0
        resonance = torch.exp(-((f - bpf) / bw) ** 2)

        # === FIXED: acoustic → [B] ตั้งแต่ต้น (mean over C_e AND F_axis) ===
        acoustic = (spec * resonance.view(1, 1, -1)).mean(dim=(1, 2))   # [B]

        # RCS feature
        em_power = (em_field * em_field).mean(dim=1, keepdim=True)      # [B,1,D,H,W]
        rcs_norm = em_power.mean(dim=(2, 3, 4))                          # [B,1]
        rcs_feat = rcs_norm / (self._sig_bird.to(dtype) + 1e-12)         # [B,1]

        # Doppler feature
        V = velocity_field.norm(dim=1, keepdim=True)                     # [B,1,D,H,W]
        c_snd = 343.0
        f0 = self._bpf.to(dtype)
        f_obs = f0 * c_snd / (c_snd - V).clamp(min=1.0)
        doppler = (f_obs - f0).abs().mean(dim=(2, 3, 4))                 # [B,1]

        # SESI barrier
        thermal_energy = self._k_b.to(dtype) * thermal_field
        kinetic_energy = 0.5 * plasma_density * (
            (velocity_field * velocity_field).sum(dim=1, keepdim=True)
        )
        sigma_sq = thermal_energy + kinetic_energy \
                   + propulsion_em_coupling * propulsion_em_coupling
        rho_mean = plasma_density.mean(dim=(2, 3, 4), keepdim=True)
        rho_mean_safe = soft_floor(rho_mean, float(eps.item()), beta=beta)
        compression_ratio = plasma_density / rho_mean_safe
        delta_e_field = torch.log1p(compression_ratio) + sigma_sq

        x32 = delta_e_field.float()
        s32 = sigma_sq.float()
        c1 = self._c1.to(device=device)
        zmax = self._zeno_max.to(device=device)
        lfloor = self._log_floor.to(device=device)
        eps_t = self._eps.to(device=device)
        dt_t = (torch.as_tensor(float(dt), dtype=torch.float32, device=device)
                if not isinstance(dt, torch.Tensor)
                else dt.float().to(device))
        denom = s32 * dt_t + eps_t
        arg = (x32 / denom).clamp_(max=float(zmax.item()))
        inner = torch.exp(arg)
        log_p = (-c1 * inner).clamp_(min=float(lfloor.item()))
        prob_bound = torch.exp(log_p).to(dtype)

        pooled_anomaly = prob_bound.mean(dim=(2, 3, 4))                  # [B]

        # === FIXED: ทุก feature เป็น [B] → stack [B,3] ===
        features = torch.stack([
            self.w_rcs      * rcs_feat.squeeze(-1),          # [B]
            self.w_acoustic * acoustic * pooled_anomaly,     # [B]
            self.w_doppler  * doppler.squeeze(-1),           # [B]
        ], dim=-1)                                            # [B,3]

        logits = self.classifier_head(features)
        soft_conf = torch.sigmoid(logits / tau)
        pred = self._gumbel_sigmoid_ste(logits, tau, self.use_hard_ste, self.training)

        if not return_metrics:
            return {"is_drone": pred}
        return {
            "is_drone":                  pred,
            "classification_confidence": soft_conf,
            "extracted_features":        features,
            "prob_bound":                prob_bound,
            "logits":                    logits,
        }


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-Discriminator v3] Running on: {device}")

    model = OmniTargetDetectorV3().to(device)
    model.train()

    B, D, H, W = 4, 16, 16, 16
    velocity    = torch.randn(B, 3, D, H, W, device=device)
    thermal     = torch.randn(B, 1, D, H, W, device=device) * 5.0 + 300.0
    em          = torch.randn(B, 3, D, H, W, device=device)
    density     = torch.rand(B, 1, D, H, W, device=device) + 0.5
    emf_coupling = torch.randn(B, 1, D, H, W, device=device) * 0.1

    out = model(velocity, thermal, em, density, emf_coupling, dt=1e-3)
    loss = out["is_drone"].float().mean() + out["logits"].float().pow(2).mean()
    loss.backward()

    print(f"  loss               : {loss.item():.6f}")
    print(f"  |∂L/∂w_rcs|        : {model.w_rcs.grad.abs().item():.6e}")
    print(f"  |∂L/∂w_acoustic|   : {model.w_acoustic.grad.abs().item():.6e}")
    print(f"  |∂L/∂w_doppler|    : {model.w_doppler.grad.abs().item():.6e}")
    print(f"  features           : {tuple(out['extracted_features'].shape)}")
    print(f"  prob_bound         : {tuple(out['prob_bound'].shape)}")
    print("Status: real RCS + BPF + Doppler; gradients flow.")
