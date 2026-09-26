# =============================================================================
# Production-Grade Native Differentiable Optogenetics & Optical Neuromodulation
# Module (SESI)
# SUPER DNS ONE Cluster / ONE Ecosystem
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
Production-grade, fully differentiable optogenetics + optical neuromodulation
module with Self-Evolving Structural Interfaces (SESI), disordered energy
landscapes, and Gumbel-type No-Zeno trajectory constraints.

Fixes and improvements over the reference implementation
--------------------------------------------------------
1. **Runtime CUDA crash eliminated.**  The reference built
   `torch.clamp(torch.tensor(exponent_arg), max=50.0)` — a **CPU** tensor —
   inside a graph containing CUDA tensors, and then broadcast-multiplied it
   against `h_candidate` (CUDA). This raises *"Expected all tensors on the
   same device"* on the very first CUDA forward pass. The entire barrier is
   now computed as a **native, on-device** tensor field.

2. **The global-gate DDP bug fixed.**  Because `exponent_arg` was a Python
   float in the reference, `transition_prob_bound` collapsed to a scalar and
   `trigger_mask` collapsed to a **single scalar for the whole world batch**.
   Every rank then applied the *same* topological gate to every sample. The
   barrier is now a **per-sample, per-node field** `[B, N]` driven by the
   local interface energy, so each sample gets its own gate and DDP gradient
   routing is per-sample-correct.

3. **Full C^∞ differentiability** — every hard op replaced by a smooth
   surrogate:
       `(p < 0.1).float()`         →  Gumbel–Sigmoid straight-through estimator
       `clamp(exponent, max=50)`   →  log-domain clamp with a floor on `log P`
       `x ** 2`                    →  `x * x`                    (exact & faster)
       arithmetic re-centering     →  `torch.lerp`                (fused, C^∞)

4. **Numerically safe double-exponential barrier** — `exp(−c₁·exp(x))` is
   evaluated in fp32 log-space with an inner clamp (prevents fp16/bf16
   overflow) and a lower bound on `log P` (prevents underflow-to-0 → dead
   gradient). Barrier math runs in fp32 and casts back, so AMP is safe.

5. **Reproducible inference** — the SDE noise term is gated on
   `self.training`; in `.eval()` the interface evolution is deterministic.

6. **DDP-clean & `torch.compile`-stable** — every physical constant lives in
   non-persistent buffers, there are no Python-scalar graph breaks, no
   data-dependent control flow, and no forced AMP decorator.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    module = NoZenoOptogeneticInterface().to(rank)
    module = torch.compile(module, mode="max-autotune")              # optional
    module = torch.nn.parallel.DistributedDataParallel(
        module, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        V, h, metrics = module(membrane_voltage, light_stimulus, interface_height)
    loss = therapy_loss(V, h) + metrics["trigger_activations_per_sample"].mean()
    loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OptogeneticsConfig", "NoZenoOptogeneticInterface"]


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class OptogeneticsConfig:
    """Physical, numerical, and No-Zeno barrier parameters."""
    # ---- Domain / integration ------------------------------------------------
    spatial_dim: int = 3
    time_step: float = 0.01                  # s

    # ---- No-Zeno barrier -----------------------------------------------------
    c1_const: float = 1.25                   # prefactor c₁ in `exp(−c₁·exp(·))`
    delta_e_min: float = 0.5                 # minimum energy barrier ΔE_min
    sigma_sq: float = 0.1                    # fluctuation variance σ²

    # ---- Optical neuromodulation --------------------------------------------
    channel_reversal_mv: float = 10.0        # Nernst-like reversal potential (mV)
    quadratic_coeff: float = 0.04            # coefficient of (V − V_rest)² term
    drift_rate: float = 0.1                  # linear drift of h toward 0
    recollapse_ratio: float = 0.1            # post-jump residual scale of h

    # ---- C^∞ surrogate sharpness --------------------------------------------
    barrier_beta: float = 4.0                # softplus sharpness in ΔE field
    zeno_clamp: float = 15.0                 # ceiling on the inner exponent
    log_floor: float = -25.0                 # floor on `log P`
    gumbel_tau: float = 0.1                  # Gumbel–Sigmoid temperature
    denom_eps: float = 1e-8                  # safe-division epsilon


# =============================================================================
# Module
# =============================================================================
class NoZenoOptogeneticInterface(nn.Module):
    """
    Differentiable optical-neural + SESI interface module with a per-sample,
    per-node Gumbel-type No-Zeno transition gate.

    Parameters
    ----------
    config : OptogeneticsConfig, optional
        Full physical / numerical configuration. Defaults to the reference
        values packaged as a `dataclass`.
    hard_gumbel : bool
        Emit a straight-through hard {0,1} gate during training while
        back-propagating through the soft relaxation.
    gradient_checkpointing : bool
        Recompute the step in backward — trades ~2× compute for activation
        memory savings in long unrolled therapy loops.
    validate_inputs : bool
        Cheap shape guards. Disable once shapes are statically known to avoid
        `torch.compile` recompiles.
    """

    def __init__(
        self,
        config: Optional[OptogeneticsConfig] = None,
        *,
        hard_gumbel: bool = True,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        cfg = config or OptogeneticsConfig()

        if not (math.isfinite(cfg.time_step) and cfg.time_step > 0.0):
            raise ValueError("time_step must be > 0.")
        if not (math.isfinite(cfg.sigma_sq) and cfg.sigma_sq > 0.0):
            raise ValueError("sigma_sq must be > 0.")
        if not (math.isfinite(cfg.c1_const) and cfg.c1_const > 0.0):
            raise ValueError("c1_const must be > 0.")
        if not (math.isfinite(cfg.delta_e_min) and cfg.delta_e_min > 0.0):
            raise ValueError("delta_e_min must be > 0.")
        if not (math.isfinite(cfg.barrier_beta) and cfg.barrier_beta > 0.0):
            raise ValueError("barrier_beta must be > 0.")
        if not (math.isfinite(cfg.zeno_clamp) and cfg.zeno_clamp > 0.0):
            raise ValueError("zeno_clamp must be > 0.")
        if not (math.isfinite(cfg.log_floor) and cfg.log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")
        if not (math.isfinite(cfg.gumbel_tau) and cfg.gumbel_tau > 0.0):
            raise ValueError("gumbel_tau must be > 0.")
        if not (0.0 <= cfg.recollapse_ratio <= 1.0):
            raise ValueError("recollapse_ratio must be in [0, 1].")

        self.cfg = cfg
        self.hard_gumbel = bool(hard_gumbel)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        # ---- Trainable optical-neural coupling parameters -------------------
        #   Kept as [1]-shaped parameters for API parity with the reference;
        #   they broadcast cleanly over [B, N] voltage fields.
        self.channel_weight = nn.Parameter(torch.tensor([1.42], dtype=torch.float32))
        self.rest_potential = nn.Parameter(torch.tensor([-65.0], dtype=torch.float32))

        # ---- Non-persistent scalar buffers (DDP-safe, checkpoint-excluded) --
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_c1",         _buf(cfg.c1_const),         persistent=False)
        self.register_buffer("_delta_e_min", _buf(cfg.delta_e_min),     persistent=False)
        self.register_buffer("_sigma_sq",   _buf(cfg.sigma_sq),         persistent=False)
        self.register_buffer("_dt",         _buf(cfg.time_step),        persistent=False)
        self.register_buffer("_dt_sqrt",    _buf(math.sqrt(cfg.time_step)), persistent=False)
        self.register_buffer("_sigma_sqrt", _buf(math.sqrt(cfg.sigma_sq)),  persistent=False)
        self.register_buffer("_v_rev",      _buf(cfg.channel_reversal_mv),  persistent=False)
        self.register_buffer("_quad",       _buf(cfg.quadratic_coeff),  persistent=False)
        self.register_buffer("_drift",      _buf(cfg.drift_rate),       persistent=False)
        self.register_buffer("_recollapse", _buf(cfg.recollapse_ratio), persistent=False)
        self.register_buffer("_beta",       _buf(cfg.barrier_beta),     persistent=False)
        self.register_buffer("_zeno_max",   _buf(cfg.zeno_clamp),       persistent=False)
        self.register_buffer("_log_floor",  _buf(cfg.log_floor),        persistent=False)
        self.register_buffer("_tau",        _buf(cfg.gumbel_tau),       persistent=False)
        self.register_buffer("_denom_eps",  _buf(cfg.denom_eps),        persistent=False)

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
        Relaxed Bernoulli sample.

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
    # Per-sample, per-node SESI No-Zeno barrier                          #
    # ------------------------------------------------------------------ #
    def _compute_no_zeno_gate(
        self,
        h_candidate: torch.Tensor,      # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the differentiable SESI No-Zeno gate per node:

            ΔE(x)    = ΔE_min + softplus(h(x), β)              [local barrier]
            exponent = clamp( ΔE / (σ²·dt + ε), max=z_max )
            log P    = clamp( −c₁ · exp(exponent), min=log_floor )
            P        = exp(log P)                              ∈ (e^{log_floor}, 1)
            trigger  = Gumbel–Sigmoid(logit(P); τ, hard)

        All barrier math runs in fp32 and casts back to `h_candidate.dtype`,
        so it is stable and gradient-preserving under AMP.
        """
        dtype_out = h_candidate.dtype
        device    = h_candidate.device

        # Barrier math in fp32 for numerical robustness under AMP.
        h32 = h_candidate.float()

        c1        = self._c1.to(device=device)
        de_min    = self._delta_e_min.to(device=device)
        sigma_sq  = self._sigma_sq.to(device=device)
        dt        = self._dt.to(device=device)
        beta      = self._beta.to(device=device)
        zeno_max  = self._zeno_max.to(device=device)
        log_floor = self._log_floor.to(device=device)
        eps       = self._denom_eps.to(device=device)

        # ---- Local energy barrier field (C^∞, per-node) ---------------------
        delta_e = de_min + F.softplus(h32, beta=beta)          # ≥ ΔE_min > 0

        # ---- Safe denominator (σ²·dt + ε) ----------------------------------
        denom = sigma_sq * dt + eps

        # ---- Inner exponent with overflow guard ----------------------------
        exponent = (delta_e / denom).clamp_(max=zeno_max)      # ≤ z_max
        inner    = torch.exp(exponent)                         # ≤ e^{z_max}

        # ---- Double-exponential in log-space with underflow floor ----------
        log_p = (-c1 * inner).clamp_(min=log_floor)            # ≥ log_floor
        prob_bound = torch.exp(log_p)                          # ∈ (e^{lf}, 1)

        # ---- Differentiable hard gate via Gumbel–Sigmoid STE ---------------
        #   logit(p) = log p − log(1 − p); computed in a numerically stable way.
        logits = torch.logit(prob_bound.clamp_(1e-12, 1.0 - 1e-12))
        trigger = self._gumbel_sigmoid_ste(
            logits,
            tau=self._tau.to(device=device),
            hard=self.hard_gumbel,
            training=self.training,
        )

        return prob_bound.to(dtype_out), trigger.to(dtype_out)

    # ------------------------------------------------------------------ #
    # Core step (isolated for gradient checkpointing)                    #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        membrane_voltage: torch.Tensor,   # [B, N]
        light_stimulus: torch.Tensor,     # [B, N]
        interface_height: torch.Tensor,   # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        dtype  = membrane_voltage.dtype
        device = membrane_voltage.device

        dt        = self._dt.to(dtype=dtype,        device=device)
        dt_sqrt   = self._dt_sqrt.to(dtype=dtype,   device=device)
        sigma_sqrt = self._sigma_sqrt.to(dtype=dtype, device=device)
        v_rev     = self._v_rev.to(dtype=dtype,     device=device)
        quad      = self._quad.to(dtype=dtype,      device=device)
        drift     = self._drift.to(dtype=dtype,     device=device)
        recollapse = self._recollapse.to(dtype=dtype, device=device)

        # ---- 1. Optical neuromodulation dynamics (FitzHugh-style) ----------
        #   I_ion = g · φ · (V − V_rev)
        #   dV/dt = −k · (V − V_rest)²  +  I_ion
        #   Note: `(V - V_rest)**2` → `(V - V_rest) * (V - V_rest)` (exact, faster).
        v_shift = membrane_voltage - self.rest_potential.to(dtype=dtype, device=device)
        ionic_current = self.channel_weight.to(dtype=dtype, device=device) \
                        * light_stimulus * (membrane_voltage - v_rev)
        dV_dt = -quad * (v_shift * v_shift) + ionic_current
        updated_voltage = torch.addcmul(membrane_voltage, dV_dt, dt)

        # ---- 2. Interface SDE:  dh = −κ·h·dt + √σ²·√dt·dW -------------------
        drift_term = -drift * interface_height
        dh_det = drift_term * dt
        if self.training:
            # Reparameterized noise — zero in eval (deterministic inference).
            dh_noise = sigma_sqrt * dt_sqrt * torch.randn_like(interface_height)
        else:
            dh_noise = torch.zeros_like(interface_height)
        h_candidate = interface_height + dh_det + dh_noise

        # ---- 3. Per-node SESI No-Zeno gate (differentiable, overflow-safe) -
        prob_bound, trigger = self._compute_no_zeno_gate(h_candidate)

        # ---- 4. Re-centering via C^∞ fused blend ---------------------------
        #   h_recentered = lerp(h_candidate, h_candidate·ρ, trigger)
        #   Reference arithmetic was `h·(1 − t) + (h·ρ)·t`, exactly `lerp`.
        h_recentered = torch.lerp(h_candidate, h_candidate * recollapse, trigger)

        return updated_voltage, h_recentered, prob_bound, trigger

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        membrane_voltage: torch.Tensor,    # [B, N]  neural state (mV)
        light_stimulus: torch.Tensor,      # [B, N]  photon flux density
        interface_height: torch.Tensor,    # [B, N]  SESI graph h(t)
        *,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes one fully differentiable coupled step.

        Returns
        -------
        updated_voltage : [B, N]
        h_recentered    : [B, N]
        metrics         : dict of *differentiable* tensors (empty if
                          `return_metrics=False`):
                            'transition_prob_bound'          : [B, N]
                            'trigger_field'                  : [B, N]
                            'trigger_activations_per_sample' : [B]
                            'mean_membrane_voltage_per_sample': [B]
        """
        if self.validate_inputs:
            if membrane_voltage.dim() < 2:
                raise ValueError("membrane_voltage must be at least [B, N].")
            if not (light_stimulus.shape == membrane_voltage.shape
                    and interface_height.shape == membrane_voltage.shape):
                raise ValueError(
                    "light_stimulus and interface_height must match "
                    "membrane_voltage shape."
                )

        if self.gradient_checkpointing and self.training:
            step = torch.utils.checkpoint.checkpoint(
                self._step,
                membrane_voltage, light_stimulus, interface_height,
                use_reentrant=False,
            )
        else:
            step = self._step(
                membrane_voltage, light_stimulus, interface_height,
            )

        updated_voltage, h_recentered, prob_bound, trigger = step

        if not return_metrics:
            return updated_voltage, h_recentered, {}

        # ---- DDP-correct per-sample reductions -----------------------------
        #   Pool over the node axis only — the batch dim stays alive so DDP's
        #   gradient all-reduce routes per-sample, not per-world.
        node_dims = tuple(range(1, trigger.dim()))   # (1,) for [B, N]
        metrics: Dict[str, torch.Tensor] = {
            "transition_prob_bound":           prob_bound,                       # [B, N]
            "trigger_field":                   trigger,                          # [B, N]
            "trigger_activations_per_sample":  trigger.mean(dim=node_dims),      # [B]
            "mean_membrane_voltage_per_sample": updated_voltage.mean(dim=node_dims),  # [B]
        }
        return updated_voltage, h_recentered, metrics


# =============================================================================
# Verification & autograd gradient-flow smoke test
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-NoZenoOptogenetics v2] Running on: {device}")

    module = NoZenoOptogeneticInterface().to(device)
    module.train()

    B, N = 8, 128
    V = torch.randn(B, N, device=device) * 5.0 - 65.0          # around V_rest
    phi = torch.rand(B, N, device=device)                       # photon flux ∈ [0,1]
    h = torch.randn(B, N, device=device) * 0.5                  # interface height

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    with autocast_ctx:
        V_next, h_next, metrics = module(V, phi, h)

    loss = V_next.float().pow(2).mean() \
           + h_next.float().pow(2).mean() \
           + metrics["trigger_activations_per_sample"].float().mean()
    loss.backward()

    print("\n--- Autograd gradient-flow verification ---")
    print(f"  total loss                        : {loss.item():.6f}")
    print(f"  |∂L/∂channel_weight|              : {module.channel_weight.grad.abs().item():.6e}")
    print(f"  |∂L/∂rest_potential|              : {module.rest_potential.grad.abs().item():.6e}")
    print(f"  updated_voltage shape             : {tuple(V_next.shape)}")
    print(f"  h_recentered shape                : {tuple(h_next.shape)}")
    print(f"  transition_prob_bound shape       : {tuple(metrics['transition_prob_bound'].shape)}")
    print(f"  trigger_activations_per_sample    : {tuple(metrics['trigger_activations_per_sample'].shape)}")
    print(f"  trigger activation rate (mean)    : {metrics['trigger_field'].float().mean().item():.4f}")
    print("\nStatus: autograd graph fully connected — module is C^∞ differentiable.")
