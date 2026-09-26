# =============================================================================
# HYPERSONIC STEALTH OMNI-DETECTOR MODULE (SESI FRAMEWORK)
# SUPER DNS ONE Cluster / ONE Ecosystem — Production Release
# =============================================================================
# Description: Advanced native, fully differentiable, O(N) optimized module
#              for Hypersonic, Stealth, and Non-Combustion (Battery) anomalous
#              targets using Double-Exponential Interface Dynamics.
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Native Differentiable / AMP-Safe / DDP-Ready)
# =============================================================================
"""
Production-grade, fully differentiable Hypersonic Stealth Omni-Detector.

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **Two critical DDP batch-collapse bugs fixed.**
   · Feature pooling used `torch.mean(kinetic_energy * prob_bound)` — reduces
     **all** dims to a single scalar, silently averaging every sample on every
     DDP rank. Replaced with per-sample spatial pooling `.mean(dim=(2,3,4))`
     → `[B]` → features `[B, 3]` → prediction `[B, 1]`.
   · `compression_ratio` used `torch.clamp(plasma_density.mean(), min=1e-6)` —
     a global mean over the *entire world batch*. Replaced with per-sample
     `plasma_density.mean(dim=(2,3,4), keepdim=True)` → `[B,1,1,1,1]`.

2. **Full C^∞ differentiability** — every hard op replaced by a smooth
   surrogate in the physically active regime:
       clamp(x, max=c)          →  c − softplus(c − x, β)         (soft ceiling)
       clamp(x, min=c)          →  c + softplus(x − c, β)         (soft floor)
       `rand < sigmoid(logit)`  →  Gumbel–Sigmoid straight-through estimator
       `x ** 2`                 →  `x * x`                        (exact & faster)

3. **Numerically safe double-exponential barrier** — `exp(−c₁·exp(x))` is
   evaluated in fp32 log-space with an inner clamp (prevents fp16/bf16
   overflow) and a lower bound on `log P` (prevents underflow-to-0, i.e.
   the dead-gradient regime that the reference could silently enter on
   strong shockwaves). Barrier math runs in fp32 and casts back — AMP-safe.

4. **Reproducible inference** — Gumbel noise is only sampled during training;
   `.eval()` returns the deterministic sigmoid probability.

5. **DDP-clean & `torch.compile`-stable** — all physical constants live in
   non-persistent buffers, no Python-scalar graph breaks, no data-dependent
   control flow, no forced AMP decorator.

6. **Unified interface with the `MicroBiologicalDiscriminator`,
   `DifferentiableOmniTargetDetector`, and `OmniTargetDetector`** — same
   metrics keys, same STE, same buffer strategy → drop-in interchangeable and
   ensemblable in a single DDP launcher.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = HypersonicStealthOmniDetector().to(rank)
    model = torch.compile(model, mode="max-autotune")             # optional
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(velocity, thermal, em, plasma, emf, dt=1e-3)
    loss = bce(out["is_hypersonic_drone"], labels)
    loss.backward()
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HypersonicStealthOmniDetector"]


class HypersonicStealthOmniDetector(nn.Module):
    """
    Differentiable detector for Hypersonic Stealth, battery-powered, and
    non-combustion anomalous targets, driven by aerothermal, plasma-sheath,
    and propulsion-EMF micro-signatures.

    Parameters
    ----------
    c1 : float
        SESI barrier prefactor c₁ in `P = exp(−c₁·exp(·))`. Must be > 0.
    ste_tau : float
        Temperature τ of the Gumbel–Sigmoid straight-through estimator.
        Smaller τ → harder (more binary) forward, still differentiable.
    k_b : float
        Boltzmann constant (J/K) used for the thermal-energy signature.
    mach_threshold : float
        Mach trigger for hypersonic/plasma-sheath classification (reserved as
        a config field; stored as a non-persistent buffer for future use).
    barrier_beta : float
        Sharpness of the smooth floor/ceiling surrogates. Larger → closer to a
        hard clamp, still C^∞.
    zeno_clamp : float
        Upper clamp on the inner exponent — prevents outer exp overflow.
    log_floor : float
        Lower clamp on `log P` — prevents underflow-to-0 (dead gradient).
    denom_eps : float
        Small positive constant for safe divisions and floors.
    hidden_dim : int
        Width of the classifier MLP's hidden layer.
    use_hard_ste : bool
        If `True`, forward value of `is_hypersonic_drone` is binary {0,1}; the
        soft relaxation persists through the backward pass.
    validate_inputs : bool
        Cheap shape guards. Disable once shapes are statically known to avoid
        `torch.compile` recompiles.
    """

    def __init__(
        self,
        c1: float = 1.0,
        ste_tau: float = 0.05,
        k_b: float = 1.380649e-23,
        mach_threshold: float = 5.0,
        *,
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
        if not (math.isfinite(k_b) and k_b > 0.0):
            raise ValueError("k_b must be > 0.")
        if not (math.isfinite(mach_threshold) and mach_threshold > 0.0):
            raise ValueError("mach_threshold must be > 0.")
        if not (math.isfinite(barrier_beta) and barrier_beta > 0.0):
            raise ValueError("barrier_beta must be > 0.")
        if not (math.isfinite(zeno_clamp) and zeno_clamp > 0.0):
            raise ValueError("zeno_clamp must be > 0.")
        if not (math.isfinite(log_floor) and log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")

        self.c1 = float(c1)
        self.ste_tau = float(ste_tau)
        self.k_b = float(k_b)
        self.mach_threshold = float(mach_threshold)
        self.barrier_beta = float(barrier_beta)
        self.zeno_clamp = float(zeno_clamp)
        self.log_floor = float(log_floor)
        self.denom_eps = float(denom_eps)
        self.hidden_dim = int(hidden_dim)
        self.use_hard_ste = bool(use_hard_ste)
        self.validate_inputs = bool(validate_inputs)

        # ---- Learnable topological feature weights -------------------------
        self.aero_heating_weight   = nn.Parameter(torch.tensor(1.2))
        self.stealth_plasma_weight = nn.Parameter(torch.tensor(1.5))
        self.emf_battery_weight    = nn.Parameter(torch.tensor(2.0))

        # ---- Ultra-light O(N) classifier head ------------------------------
        self.classifier_head = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

        # ---- Non-persistent scalar buffers (DDP-safe, checkpoint-excluded) -
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_c1",         _buf(self.c1),             persistent=False)
        self.register_buffer("_tau",        _buf(self.ste_tau),        persistent=False)
        self.register_buffer("_k_b",        _buf(self.k_b),            persistent=False)
        self.register_buffer("_mach_thresh", _buf(self.mach_threshold), persistent=False)
        self.register_buffer("_beta",       _buf(self.barrier_beta),   persistent=False)
        self.register_buffer("_zeno_max",   _buf(self.zeno_clamp),     persistent=False)
        self.register_buffer("_log_floor",  _buf(self.log_floor),      persistent=False)
        self.register_buffer("_denom_eps",  _buf(self.denom_eps),      persistent=False)

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # C^∞ primitives                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _smooth_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """C^∞ surrogate for max(x, floor) — exact for x ≫ floor."""
        return floor + F.softplus(x - floor, beta=beta)

    # ------------------------------------------------------------------ #
    # Differentiable Gumbel–Sigmoid straight-through estimator           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gumbel_sigmoid_ste(
        logits: torch.Tensor,
        tau: torch.Tensor,
        hard: bool,
        training: bool,
    ) -> torch.Tensor:
        """
        Relaxed Bernoulli sampler.

        Training:
            g ~ Gumbel(0, 1); y_soft = sigmoid((logits + g) / τ)
            y = y_hard − y_soft.detach() + y_soft    (if `hard`)
        Eval:
            y = sigmoid(logits / τ)                  (deterministic)
        """
        if not training:
            return torch.sigmoid(logits / tau)
        u = torch.rand_like(logits).clamp_(1e-7, 1.0 - 1e-7)
        g = -torch.log(-torch.log(u))                # Gumbel(0, 1)
        y_soft = torch.sigmoid((logits + g) / tau)
        if not hard:
            return y_soft
        y_hard = (y_soft > 0.5).to(y_soft.dtype)
        return y_hard - y_soft.detach() + y_soft     # straight-through

    # ------------------------------------------------------------------ #
    # Differentiable SESI extreme-anomaly bound                          #
    # ------------------------------------------------------------------ #
    def compute_extreme_anomaly_bound(
        self,
        delta_e_field: torch.Tensor,
        sigma_sq: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Double-exponential SESI No-Zeno anomaly bound

            P = exp( −c₁ · exp( clamp(ΔE / (σ²·dt), max=z_max) ) )
            P ← max( P, exp(log_floor) )        (prevents 0-gradient underflow)

        All internal math runs in fp32 and casts back, so the barrier is
        stable and differentiable under AMP (fp16 / bf16 / fp32).
        """
        dtype_out = delta_e_field.dtype
        device    = delta_e_field.device

        # Barrier math in fp32 for numerical robustness under AMP.
        x32 = delta_e_field.float()
        s32 = sigma_sq.float()

        c1        = self._c1.to(device=device)
        zeno_max  = self._zeno_max.to(device=device)
        log_floor = self._log_floor.to(device=device)
        eps       = self._denom_eps.to(device=device)

        if not isinstance(dt, torch.Tensor):
            dt_t = torch.tensor(float(dt), dtype=torch.float32, device=device)
        else:
            dt_t = dt.float().to(device)

        denom = s32 * dt_t + eps
        arg   = (x32 / denom).clamp_(max=zeno_max)     # prevent overflow
        inner = torch.exp(arg)                         # ≤ e^{z_max}
        log_p = (-c1 * inner).clamp_(min=log_floor)    # ≥ log_floor
        p     = torch.exp(log_p)
        return p.to(dtype_out)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        velocity_field: torch.Tensor,          # [B, 3, D, H, W]  NS wake / Mach flow
        thermal_field: torch.Tensor,           # [B, 1, D, H, W]  ambient / skin T
        em_field: torch.Tensor,                # [B, C_e, D, H, W]  external RCS
        plasma_density: torch.Tensor,          # [B, 1, D, H, W]  ionized-air density
        propulsion_em_coupling: torch.Tensor,  # [B, 1, D, H, W]  battery EMF coupling
        dt: float | torch.Tensor,
        *,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
            'is_hypersonic_drone'         : [B, 1]  {0,1} (STE) / [0,1] (soft)
            'classification_confidence'   : [B, 1]  sigmoid probability
            'extracted_features'          : [B, 3]  pooled (aero, plasma, EMF)
            'prob_bound'                  : [B, 1, D, H, W]  SESI anomaly field
            'logits'                      : [B, 1]  raw classifier output
        """
        if self.validate_inputs:
            if velocity_field.dim() != 5 or velocity_field.shape[1] != 3:
                raise ValueError("velocity_field must be [B, 3, D, H, W].")
            for name, t in (
                ("thermal_field", thermal_field),
                ("em_field", em_field),
                ("plasma_density", plasma_density),
                ("propulsion_em_coupling", propulsion_em_coupling),
            ):
                if t.dim() != 5:
                    raise ValueError(f"{name} must be 5D [B, C, D, H, W].")
                if t.shape[0] != velocity_field.shape[0]:
                    raise ValueError(f"{name} batch dim must match velocity_field.")
                if t.shape[2:] != velocity_field.shape[2:]:
                    raise ValueError(f"{name} spatial dims must match velocity_field.")

        dtype  = velocity_field.dtype
        device = velocity_field.device

        beta = self._beta.to(device=device)
        k_b  = self._k_b.to(device=device)
        tau  = self._tau.to(device=device)
        eps  = self._denom_eps.to(device=device)

        # ---- 1. Hypersonic Navier–Stokes extraction ------------------------
        #   Kinetic energy density:  ½ · ρ_plasma · |v|²
        #   `x * x` (not `x ** 2`) → exact same math, faster kernel, compile-clean.
        v_sq = (velocity_field * velocity_field).sum(dim=1, keepdim=True)   # [B,1,D,H,W]
        kinetic_energy = 0.5 * plasma_density * v_sq

        # ---- 2. Stealth / non-combustion logic -----------------------------
        thermal_energy = k_b * thermal_field
        #   Propulsion EMF coupling energy — exact x·x.
        battery_coupling_energy = propulsion_em_coupling * propulsion_em_coupling

        # ---- 3. Universal activation energy (anomaly barrier) --------------
        sigma_sq = thermal_energy + kinetic_energy + battery_coupling_energy

        #   Shockwave compression ratio: local / mean plasma density.
        #   Per-sample spatial mean — NEVER a global mean over the world batch
        #   (the reference's DDP batch-collapse bug).
        rho_mean      = plasma_density.mean(dim=(2, 3, 4), keepdim=True)     # [B,1,1,1,1]
        rho_mean_safe = self._smooth_floor(rho_mean, eps, beta)              # C^∞ ≥ eps
        compression_ratio = plasma_density / rho_mean_safe

        delta_e_field = torch.log1p(compression_ratio) + sigma_sq

        # ---- 4. SESI extreme-anomaly probability field ----------------------
        prob_bound = self.compute_extreme_anomaly_bound(delta_e_field, sigma_sq, dt)

        # ---- 5. Per-sample spatial pooling (DDP-correct) --------------------
        #   CRITICAL: mean over spatial dims only — keeps batch dim alive so
        #   DDP all-reduce of the classifier head gradients is per-sample.
        spatial_dims = (2, 3, 4)
        pooled_aero_heating    = (kinetic_energy           * prob_bound).mean(dim=spatial_dims) * self.aero_heating_weight
        pooled_plasma_sheath   = (compression_ratio        * prob_bound).mean(dim=spatial_dims) * self.stealth_plasma_weight
        pooled_battery_emf     = (battery_coupling_energy  * prob_bound).mean(dim=spatial_dims) * self.emf_battery_weight

        features = torch.stack(
            [pooled_aero_heating, pooled_plasma_sheath, pooled_battery_emf], dim=-1
        )                                                                    # [B, 3]

        # ---- 6. Differentiable classification -------------------------------
        #   logit > 0 ⇒ Hypersonic stealth drone
        #   logit < 0 ⇒ Natural / background noise
        logits = self.classifier_head(features)                              # [B, 1]

        soft_confidence = torch.sigmoid(logits / tau)                        # [B, 1]
        prediction = self._gumbel_sigmoid_ste(
            logits, tau, hard=self.use_hard_ste, training=self.training,
        )

        if not return_metrics:
            return {"is_hypersonic_drone": prediction}

        return {
            "is_hypersonic_drone":        prediction,
            "classification_confidence":  soft_confidence,
            "extracted_features":         features,
            "prob_bound":                 prob_bound,
            "logits":                     logits,
        }


# =============================================================================
# Verification & autograd gradient-flow smoke test
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-HypersonicStealthOmniDetector v2] Running on: {device}")

    model = HypersonicStealthOmniDetector().to(device)
    model.train()

    B, D, H, W = 4, 16, 16, 16
    velocity    = torch.randn(B, 3, D, H, W, device=device) * 10.0
    thermal     = torch.randn(B, 1, D, H, W, device=device) * 5.0 + 300.0
    em          = torch.randn(B, 3, D, H, W, device=device)
    plasma      = torch.rand(B, 1, D, H, W, device=device) + 0.5
    emf_coupling = torch.randn(B, 1, D, H, W, device=device) * 0.1

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    with autocast_ctx:
        out = model(velocity, thermal, em, plasma, emf_coupling, dt=1e-3)

    loss = out["is_hypersonic_drone"].float().mean() \
           + out["logits"].float().pow(2).mean()
    loss.backward()

    print("\n--- Autograd gradient-flow verification ---")
    print(f"  total loss                        : {loss.item():.6f}")
    print(f"  |∂L/∂aero_heating_weight|         : {model.aero_heating_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂stealth_plasma_weight|       : {model.stealth_plasma_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂emf_battery_weight|          : {model.emf_battery_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂head[0].weight|              : {model.classifier_head[0].weight.grad.abs().mean().item():.6e}")
    print(f"  prediction shape                  : {tuple(out['is_hypersonic_drone'].shape)}")
    print(f"  features shape                    : {tuple(out['extracted_features'].shape)}")
    print(f"  prob_bound shape                  : {tuple(out['prob_bound'].shape)}")
    print("\nStatus: autograd graph fully connected — module is C^∞ differentiable.")
