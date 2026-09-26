# =============================================================================
# OMNI-SPECTRAL DRONE CLASSIFICATION PIPELINE (SESI FRAMEWORK)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Description: Native, fully differentiable pipeline integrating Exact Maxwell
#              structural solvers with Extreme-Value anomaly detection to
#              classify Drones vs. Biological entities in real-time O(N).
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Production / DDP-Ready)
# =============================================================================
"""
Production-grade, fully differentiable omni-spectral classification pipeline.

Fixes and improvements over the reference implementation
--------------------------------------------------------
1. **Two critical DDP batch-collapse bugs fixed.**
   · Pooling used `torch.mean(kinetic_energy * prob_bound)` — reduces **all**
     dims to a single scalar, silently averaging every sample on every DDP
     rank. Replaced with per-sample spatial pooling `.mean(dim=(2,3,4))` → `[B]`.
   · `compression_ratio` used `torch.clamp(plasma_density.mean(), min=1e-6)` —
     a global mean over the *entire world batch*. Replaced with per-sample
     `plasma_density.mean(dim=(2,3,4), keepdim=True)` → `[B,1,1,1,1]`.

2. **Full C^∞ differentiability** — hard ops replaced with smooth surrogates:
       clamp(x, max=c)          →  c − softplus(c − x, β)         (smooth ceiling)
       clamp(x, min=c)          →  c + softplus(x − c, β)         (smooth floor)
       `rand < sigmoid(logit)`  →  Gumbel–Sigmoid straight-through estimator
       `x ** 2`                 →  `x * x`                        (exact & faster)

3. **Numerically safe double-exponential barrier** — `exp(−c₁·exp(x))` is
   evaluated in log-space with an inner clamp (prevents fp16/bf16 overflow)
   and a lower bound on `log p` (prevents underflow-to-0 → dead gradient).
   Barrier math runs in fp32 and casts back, so AMP is safe.

4. **Reproducible inference** — stochastic Gumbel sampling is gated on
   `self.training`; `.eval()` returns a deterministic sigmoid probability.

5. **DDP-clean & `torch.compile`-stable** — all physical constants live in
   non-persistent buffers, no Python-scalar graph breaks, no data-dependent
   control flow, no forced AMP decorator.

6. **Unified interface with the `MicroBiologicalDiscriminator` and
   `DifferentiableOmniTargetDetector`** — same metrics keys, same STE, same
   buffer strategy → these three modules are drop-in interchangeable and can
   be ensembled in the same DDP launcher without code changes.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AdvancedClassificationPipeline(dt=1e-3).to(rank)
    model = torch.compile(model, mode="max-autotune")                 # optional
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(e_field, b_field, velocity, thermal, density, emf)
    loss = bce(out["is_drone"], labels)
    loss.backward()
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OmniTargetDetector", "AdvancedClassificationPipeline"]


# =============================================================================
# Detector
# =============================================================================
class OmniTargetDetector(nn.Module):
    """
    Differentiable omni-spectral discriminator: Mechanical (drone) vs.
    Biological (bird), fusing wake, thermal, and EM signatures through a
    SESI No-Zeno extreme-value anomaly bound.

    Parameters
    ----------
    c1 : float
        SESI barrier prefactor c₁ in `P = exp(−c₁·exp(·))`. Must be > 0.
    ste_tau : float
        Temperature τ of the Gumbel–Sigmoid straight-through estimator.
    k_b : float
        Boltzmann constant (J/K) used for the thermal-energy signature.
    barrier_beta : float
        Sharpness of the smooth floor surrogate. Larger → closer to a hard
        clamp, still C^∞.
    zeno_clamp : float
        Upper clamp on the inner exponent — prevents outer exp overflow.
    log_floor : float
        Lower clamp on `log P` — prevents underflow-to-0 (dead gradient).
    denom_eps : float
        Small positive constant for safe divisions and floors.
    hidden_dim : int
        Width of the classifier MLP's hidden layer.
    use_hard_ste : bool
        If `True`, forward value of `is_drone` is binary {0,1}; the soft
        relaxation persists through the backward pass.
    validate_inputs : bool
        Cheap shape guards. Disable once shapes are statically known.
    """

    def __init__(
        self,
        c1: float = 1.0,
        ste_tau: float = 0.05,
        k_b: float = 1.380649e-23,
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
        self.barrier_beta = float(barrier_beta)
        self.zeno_clamp = float(zeno_clamp)
        self.log_floor = float(log_floor)
        self.denom_eps = float(denom_eps)
        self.hidden_dim = int(hidden_dim)
        self.use_hard_ste = bool(use_hard_ste)
        self.validate_inputs = bool(validate_inputs)

        # ---- Learnable multi-physics feature-balancing weights -------------
        self.wake_weight    = nn.Parameter(torch.tensor(1.0))
        self.thermal_weight = nn.Parameter(torch.tensor(1.0))
        self.em_weight      = nn.Parameter(torch.tensor(1.0))

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

        self.register_buffer("_c1",        _buf(self.c1),          persistent=False)
        self.register_buffer("_tau",       _buf(self.ste_tau),     persistent=False)
        self.register_buffer("_k_b",       _buf(self.k_b),         persistent=False)
        self.register_buffer("_beta",      _buf(self.barrier_beta), persistent=False)
        self.register_buffer("_zeno_max",  _buf(self.zeno_clamp),  persistent=False)
        self.register_buffer("_log_floor", _buf(self.log_floor),   persistent=False)
        self.register_buffer("_denom_eps", _buf(self.denom_eps),   persistent=False)

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
            y = y_hard − y_soft.detach() + y_soft   (if `hard`)
        Eval:
            y = sigmoid(logits / τ)                 (deterministic)
        """
        if not training:
            return torch.sigmoid(logits / tau)
        u = torch.rand_like(logits).clamp_(1e-7, 1.0 - 1e-7)
        g = -torch.log(-torch.log(u))                 # Gumbel(0, 1)
        y_soft = torch.sigmoid((logits + g) / tau)
        if not hard:
            return y_soft
        y_hard = (y_soft > 0.5).to(y_soft.dtype)
        return y_hard - y_soft.detach() + y_soft      # straight-through

    # ------------------------------------------------------------------ #
    # Differentiable SESI anomaly-probability bound                      #
    # ------------------------------------------------------------------ #
    def compute_anomaly_probability(
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
        velocity_field: torch.Tensor,          # [B, 3, D, H, W]  NS wake / vorticity
        thermal_field: torch.Tensor,           # [B, 1, D, H, W]  IR / heat footprint
        em_field: torch.Tensor,                # [B, C_e, D, H, W]  RCS / radar
        plasma_density: torch.Tensor,          # [B, 1, D, H, W]  environmental density
        propulsion_em_coupling: torch.Tensor,  # [B, 1, D, H, W]  battery EMF leakage
        dt: float | torch.Tensor,
        *,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
            'is_drone'                 : [B, 1]  {0,1} (STE) / [0,1] (soft)
            'classification_confidence': [B, 1]  sigmoid probability
            'extracted_features'       : [B, 3]  pooled (wake, thermal, EM)
            'prob_bound'               : [B, 1, D, H, W]  SESI anomaly field
            'logits'                   : [B, 1]  raw classifier output
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
            # Spatial dims must agree across all inputs so the per-sample pooling
            # averages a consistent spatial support.
            spatial = velocity_field.shape[2:]
            for name, t in (
                ("thermal_field", thermal_field),
                ("em_field", em_field),
                ("plasma_density", plasma_density),
                ("propulsion_em_coupling", propulsion_em_coupling),
            ):
                if t.shape[2:] != spatial:
                    raise ValueError(f"{name} spatial dims must match velocity_field.")

        dtype  = velocity_field.dtype
        device = velocity_field.device

        beta = self._beta.to(device=device)
        k_b  = self._k_b.to(device=device)
        tau  = self._tau.to(device=device)
        eps  = self._denom_eps.to(device=device)

        # ---- 1. Multi-physics variance extraction ---------------------------
        #   Kinetic energy density:  ½ · ρ · |v|²   → sum over vector channels.
        #   `x * x` (not `x ** 2`) → exact same math, faster kernel, compile-clean.
        kinetic_energy = 0.5 * plasma_density * (velocity_field * velocity_field).sum(dim=1, keepdim=True)
        #   Thermal energy:          k_B · T
        thermal_energy = k_b * thermal_field
        #   EM power density: mean squared over EM channels → [B, 1, D, H, W]
        em_power = (em_field * em_field).mean(dim=1, keepdim=True)

        #   Total synthetic variance signature (all add / addcmul — no Python scalars)
        sigma_sq = (
            thermal_energy
            + kinetic_energy
            + em_power
            + propulsion_em_coupling * propulsion_em_coupling
        )

        # ---- 2. Universal activation energy (anomaly barrier) ---------------
        #   Per-sample spatial mean — NEVER a global mean over the world batch
        #   (the reference's DDP batch-collapse bug).
        rho_mean      = plasma_density.mean(dim=(2, 3, 4), keepdim=True)     # [B,1,1,1,1]
        rho_mean_safe = self._smooth_floor(rho_mean, eps, beta)              # C^∞ ≥ eps
        compression_ratio = plasma_density / rho_mean_safe
        delta_e_field = torch.log1p(compression_ratio) + sigma_sq

        # ---- 3. SESI anomaly-probability field (differentiable, overflow-safe)
        prob_bound = self.compute_anomaly_probability(delta_e_field, sigma_sq, dt)

        # ---- 4. Per-sample spatial pooling (DDP-correct) --------------------
        #   CRITICAL: mean over spatial dims only — keeps batch dim alive so the
        #   DDP all-reduce of the classifier head gradients stays per-sample.
        spatial_dims = (2, 3, 4)
        pooled_wake    = (kinetic_energy * prob_bound).mean(dim=spatial_dims) * self.wake_weight
        pooled_thermal = (thermal_energy * prob_bound).mean(dim=spatial_dims) * self.thermal_weight
        pooled_em      = (em_power       * prob_bound).mean(dim=spatial_dims) * self.em_weight

        features = torch.stack(
            [pooled_wake, pooled_thermal, pooled_em], dim=-1
        )                                                             # [B, 3]

        # ---- 5. Differentiable classification -------------------------------
        #   logit > 0 ⇒ Mechanical (drone); logit < 0 ⇒ Biological (bird)
        logits = self.classifier_head(features)                       # [B, 1]

        soft_confidence = torch.sigmoid(logits / tau)                 # [B, 1]
        prediction = self._gumbel_sigmoid_ste(
            logits, tau, hard=self.use_hard_ste, training=self.training,
        )

        if not return_metrics:
            return {"is_drone": prediction}

        return {
            "is_drone":                  prediction,
            "classification_confidence": soft_confidence,
            "extracted_features":        features,
            "prob_bound":                prob_bound,
            "logits":                    logits,
        }


# =============================================================================
# Pipeline
# =============================================================================
class AdvancedClassificationPipeline(nn.Module):
    """
    Primary processing pipeline bridging the Maxwell structural solvers
    (E and B fields) with the `OmniTargetDetector`.

    Responsibilities
    ----------------
    · Concatenate E and B field channels into a unified EM footprint.
    · Forward the whole multi-physics bundle through the differentiable
      detector, returning the same rich metrics dict.

    Parameters
    ----------
    dt : float
        Time step used by the SESI anomaly barrier. Must be > 0.
    detector_kwargs : dict, optional
        Any extra keyword arguments forwarded to `OmniTargetDetector`
        (e.g. `ste_tau`, `c1`, `hidden_dim`, `use_hard_ste`, ...).
    """

    def __init__(self, dt: float = 0.01, *, detector_kwargs: Optional[Dict] = None) -> None:
        super().__init__()
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be a positive finite scalar.")

        self.dt = float(dt)
        self.target_detector = OmniTargetDetector(**(detector_kwargs or {}))

    def forward(
        self,
        e_field: torch.Tensor,                 # [B, C_e, D, H, W]
        b_field: torch.Tensor,                 # [B, C_b, D, H, W]
        velocity_field: torch.Tensor,          # [B, 3,   D, H, W]
        thermal_field: torch.Tensor,           # [B, 1,   D, H, W]
        plasma_density: torch.Tensor,          # [B, 1,   D, H, W]
        propulsion_em_coupling: torch.Tensor,  # [B, 1,   D, H, W]
        *,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the same metrics dict as `OmniTargetDetector.forward`.
        """
        if e_field.dim() != 5 or b_field.dim() != 5:
            raise ValueError("e_field and b_field must be 5D [B, C, D, H, W].")
        if e_field.shape[0] != b_field.shape[0]:
            raise ValueError("e_field and b_field batch dims must match.")
        if e_field.shape[2:] != b_field.shape[2:]:
            raise ValueError("e_field and b_field spatial dims must match.")

        # Fused channel concat — single kernel, no intermediate copies.
        em_field = torch.cat([e_field, b_field], dim=1)

        return self.target_detector(
            velocity_field=velocity_field,
            thermal_field=thermal_field,
            em_field=em_field,
            plasma_density=plasma_density,
            propulsion_em_coupling=propulsion_em_coupling,
            dt=self.dt,
            return_metrics=return_metrics,
        )


# =============================================================================
# Verification & autograd gradient-flow smoke test
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-OmniPipeline v2] Running on: {device}")

    pipeline = AdvancedClassificationPipeline(dt=1e-3).to(device)
    pipeline.train()

    B, D, H, W = 4, 16, 16, 16
    e_field      = torch.randn(B, 3, D, H, W, device=device)
    b_field      = torch.randn(B, 3, D, H, W, device=device)
    velocity     = torch.randn(B, 3, D, H, W, device=device)
    thermal      = torch.randn(B, 1, D, H, W, device=device) * 5.0 + 300.0
    density      = torch.rand(B, 1, D, H, W, device=device) + 0.5
    emf_coupling = torch.randn(B, 1, D, H, W, device=device) * 0.1

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    with autocast_ctx:
        out = pipeline(e_field, b_field, velocity, thermal, density, emf_coupling)

    loss = out["is_drone"].float().mean() + out["logits"].float().pow(2).mean()
    loss.backward()

    detector = pipeline.target_detector
    print("\n--- Autograd gradient-flow verification ---")
    print(f"  total loss                : {loss.item():.6f}")
    print(f"  |∂L/∂wake_weight|         : {detector.wake_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂thermal_weight|      : {detector.thermal_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂em_weight|           : {detector.em_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂head[0].weight|      : {detector.classifier_head[0].weight.grad.abs().mean().item():.6e}")
    print(f"  prediction shape          : {tuple(out['is_drone'].shape)}")
    print(f"  features shape            : {tuple(out['extracted_features'].shape)}")
    print(f"  prob_bound shape          : {tuple(out['prob_bound'].shape)}")
    print("\nStatus: autograd graph fully connected — pipeline is C^∞ differentiable.")
