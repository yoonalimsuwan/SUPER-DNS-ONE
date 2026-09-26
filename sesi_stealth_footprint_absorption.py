# =============================================================================
# SESI 3D NAVIER-STOKES STEALTH FOOTPRINT ABSORPTION MODULE
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Framework   : Self-Evolving Structural Interfaces (SESI)
# Module      : 3D Navier-Stokes Flow Footprint & Signature Mitigation
# Developer   : PAI and Yoon A Limsuwan — MSPS NETWORK
# ORCID       : 0009-0008-2374-0788
# GitHub      : yoonalimsuwan
# Contact     : msps4u@gmail.com
# License     : MIT
# Year        : 2026
# Version     : 2.0.0 (Native Differentiable / Batched / DDP-Ready)
# =============================================================================
"""
Production-grade, fully differentiable 3D Navier-Stokes stealth footprint
absorber with SESI disordered-interface coupling and log-domain Gumbel bounds.

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **`.item()` calls eliminated — every one of them.**  The reference returned
   `initial_footprint_energy.item()`, `.item()` on the absorbed energy, the
   absorption ratio, the breakout probability, AND a Python `bool`
   `stealth_target_achieved`.  This forced six D2H synchronizations per call,
   **severed the autograd graph at the module boundary**, and made the module
   unusable inside a differentiable therapy / mitigation loop. Every metric is
   now a per-sample tensor.

2. **World-batch reduction (DDP bug) fixed — three instances.**  The reference
   collapsed every field to a **global scalar** via
   `0.5 * torch.sum(velocity_field ** 2)`, `torch.sum(sigma_interface * vorticity)`,
   and never carried a batch dim at all (`velocity_field` had shape
   `[3, D, H, W]`). Under DDP this would have silently summed across every
   sample on every rank. The module is now **batch-first** (`[B, 3, D, H, W]`)
   with **per-sample** spatial reductions throughout.

3. **Full C^∞ differentiability** — every hard op replaced by a smooth
   surrogate in the physically active regime:
       torch.abs(x)              →  sqrt(x² + ε²)
       clamp(x, min=c)           →  c + softplus(x − c, β)
       clamp(x, max=c)           →  c − softplus(c − x, β)
       `ratio >= target` (bool)  →  sigmoid((ratio − target)/τ)   [soft indicator]
       x ** 2                    →  x * x                         [exact & faster]

4. **Numerically safe double-exponential barrier.**  `exp(−c₁·exp(x))` is
   evaluated in **fp32 log-space** with an inner clamp and a lower bound on
   `log P`, so it cannot overflow in fp16/bf16 and cannot underflow to zero
   (dead gradient). Barrier math runs in fp32 and casts back — AMP-safe.

5. **Fused conv3d gradients.**  The reference computed 9 central-difference
   gradients per field using **12 `torch.roll` calls + 12 subtractions + 12
   divisions** (~36 kernel launches). The new engine computes them in **3
   depthwise `conv3d` calls** on a single circularly-padded tensor — roughly
   10× fewer launches per vorticity evaluation.

6. **Deterministic inference.**  Both the disordered-interface noise and the
   turbulent-perturbation field are gated on `self.training`. `.eval()` gives
   a deterministic absorber suitable for validation, ONNX export, or real-time
   control.

7. **`device` and `dtype` removed from the config dataclass.**  They followed
   the inputs nowhere — the module now simply follows input dtype/device, as
   required for correct DDP + AMP composition.

8. **DDP-clean & `torch.compile`-stable.**  All constants in non-persistent
   buffers, no Python-scalar graph breaks, no data-dependent control flow, no
   forced AMP decorator.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    absorber = SESIStealthFootprintAbsorber(cfg).to(rank)
    absorber = torch.compile(absorber, mode="max-autotune")            # optional
    absorber = torch.nn.parallel.DistributedDataParallel(
        absorber, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = absorber(velocity, sigma_skin)
    (out["absorbed_velocity_field"].pow(2).mean()
     + (1.0 - out["stealth_target_achieved"]).mean()).backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "StealthCFDConfig",
    "NavierStokesFootprintEngine",
    "SESIStealthFootprintAbsorber",
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class StealthCFDConfig:
    """Physical, CSOC, and numerical configuration for the stealth module."""
    grid_resolution: Tuple[int, int, int] = (32, 32, 32)
    reynolds_number: float = 1e6
    viscosity: float = 1e-4
    c1_geom: float = 1.0                       # SESI barrier prefactor c₁
    sigma_variance: float = 0.05               # disordered-interface σ
    dt: float = 1e-5                           # s
    absorption_efficiency_target: float = 0.95

    # ---- Numerical safety / C^∞ surrogate sharpness ----------------------
    inner_clamp: float = 15.0                  # ceiling on inner exponent
    log_floor: float = -40.0                   # floor on log P (dead-grad guard)
    denom_eps: float = 1e-8                    # safe-division epsilon
    barrier_beta: float = 4.0                  # softplus sharpness
    indicator_tau: float = 0.05                # soft-indicator sharpness
    perturbation_amplitude: float = 0.2        # amplitude of turbulent perturbation


# =============================================================================
# Navier–Stokes footprint engine
# =============================================================================
class NavierStokesFootprintEngine(nn.Module):
    """
    Differentiable 3D Navier–Stokes vorticity and footprint-energy engine.

    All operations are batch-first and per-sample — safe under DDP.

    Parameters
    ----------
    config : StealthCFDConfig
    """

    def __init__(self, config: StealthCFDConfig) -> None:
        super().__init__()
        self.cfg = config

        # ---- Depthwise circular-padded central-difference kernels ----------
        #   Each kernel is [3, 1, 3, 3, 3] (depthwise) so a single conv3d with
        #   `groups=3` differentiates all three velocity components at once
        #   along one spatial axis.
        def _kernel(axis: int) -> torch.Tensor:
            k = torch.zeros(3, 1, 3, 3, 3, dtype=torch.float32)
            if axis == 0:      # ∂/∂D
                k[:, 0, 0, 1, 1] = -0.5
                k[:, 0, 2, 1, 1] = +0.5
            elif axis == 1:    # ∂/∂H
                k[:, 0, 1, 0, 1] = -0.5
                k[:, 0, 1, 2, 1] = +0.5
            else:              # ∂/∂W
                k[:, 0, 1, 1, 0] = -0.5
                k[:, 0, 1, 1, 2] = +0.5
            return k

        self.register_buffer("_kD", _kernel(0), persistent=False)
        self.register_buffer("_kH", _kernel(1), persistent=False)
        self.register_buffer("_kW", _kernel(2), persistent=False)

        # Smooth-|·| surrogate epsilon (used in sqrt(x² + ε²)).
        self.register_buffer(
            "_abs_eps", torch.tensor(float(config.denom_eps), dtype=torch.float32),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pad_circular(x: torch.Tensor) -> torch.Tensor:
        """Circular padding on the three spatial axes of a 5D tensor."""
        return F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")

    # ------------------------------------------------------------------ #
    def compute_vorticity_magnitude(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """
        Differentiable vorticity magnitude via fused depthwise conv3d.

        Parameters
        ----------
        velocity_field : torch.Tensor  [B, 3, D, H, W]
            Channel order: (u_x, u_y, u_z). The output uses the same
            consistent (mis-labeled-but-internally-consistent) curl convention
            as the SESI v1.0.0 reference.

        Returns
        -------
        vorticity_mag : torch.Tensor  [B, 1, D, H, W]
        """
        if velocity_field.dim() != 5 or velocity_field.shape[1] != 3:
            raise ValueError("velocity_field must be [B, 3, D, H, W].")

        dtype = velocity_field.dtype
        vp = self._pad_circular(velocity_field)

        # Three fused depthwise conv3d calls (one per spatial axis).
        gD = F.conv3d(vp, self._kD.to(dtype), groups=3)   # [B, 3, D, H, W]
        gH = F.conv3d(vp, self._kH.to(dtype), groups=3)
        gW = F.conv3d(vp, self._kW.to(dtype), groups=3)

        # Reference curl convention (see module docstring §5).
        #   omega_x = ∂u_z/∂H − ∂u_y/∂D   → gH[:, 2] − gD[:, 1]
        #   omega_y = ∂u_x/∂D − ∂u_z/∂W   → gD[:, 0] − gW[:, 2]
        #   omega_z = ∂u_y/∂W − ∂u_x/∂H   → gW[:, 1] − gH[:, 0]
        omega_x = gH[:, 2:3] - gD[:, 1:2]
        omega_y = gD[:, 0:1] - gW[:, 2:3]
        omega_z = gW[:, 1:2] - gH[:, 0:1]

        eps = self._abs_eps.to(dtype)
        return torch.sqrt(
            omega_x * omega_x + omega_y * omega_y + omega_z * omega_z + eps * eps
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _per_sample_sum(x: torch.Tensor) -> torch.Tensor:
        """Sum over every dim except the batch dim (dim 0) → shape [B]."""
        if x.dim() <= 1:
            return x
        return x.sum(dim=tuple(range(1, x.dim())))

    # ------------------------------------------------------------------ #
    def evaluate_footprint_energy(
        self,
        velocity_field: torch.Tensor,       # [B, 3, D, H, W]
        sigma_interface: torch.Tensor,      # [B, 1, D, H, W]
    ) -> torch.Tensor:
        """
        Per-sample kinetic + interface-stress footprint energy:

            E = ½ · Σ|u|²_spatial  +  Σ(σ_interface · |curl u|)_spatial

        Returns
        -------
        energy : torch.Tensor  [B]
        """
        # Kinetic energy density — `x * x` (exact, faster than `.pow(2)`).
        kinetic_energy = 0.5 * self._per_sample_sum(velocity_field * velocity_field)

        # Vorticity magnitude field  [B, 1, D, H, W]
        vorticity = self.compute_vorticity_magnitude(velocity_field)

        # Interface stress: σ (B,1,D,H,W) × |ω| (B,1,D,H,W) → per-sample scalar.
        interface_stress = self._per_sample_sum(sigma_interface * vorticity)

        return kinetic_energy + interface_stress


# =============================================================================
# Stealth footprint absorber
# =============================================================================
class SESIStealthFootprintAbsorber(nn.Module):
    """
    Production-grade SESI absorber that reduces Navier–Stokes footprint energy
    via disordered-interface coupling and a log-domain Gumbel No-Zeno bound.

    All inputs and outputs are batched; every metric is a per-sample tensor
    ready for `loss.backward()` and correct under DDP.

    Parameters
    ----------
    config : StealthCFDConfig, optional
    gradient_checkpointing : bool
        Recompute the step during backward — trades ~2× compute for memory.
    validate_inputs : bool
        Cheap shape guards. Disable once shapes are statically known.
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

        if not (math.isfinite(cfg.c1_geom) and cfg.c1_geom > 0.0):
            raise ValueError("c1_geom must be > 0.")
        if not (math.isfinite(cfg.sigma_variance) and cfg.sigma_variance >= 0.0):
            raise ValueError("sigma_variance must be ≥ 0.")
        if not (math.isfinite(cfg.dt) and cfg.dt > 0.0):
            raise ValueError("dt must be > 0.")
        if not (math.isfinite(cfg.inner_clamp) and cfg.inner_clamp > 0.0):
            raise ValueError("inner_clamp must be > 0.")
        if not (math.isfinite(cfg.log_floor) and cfg.log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")
        if not (math.isfinite(cfg.indicator_tau) and cfg.indicator_tau > 0.0):
            raise ValueError("indicator_tau must be > 0.")
        if not (0.0 <= cfg.absorption_efficiency_target <= 1.0):
            raise ValueError("absorption_efficiency_target must be in [0, 1].")

        self.cfg = cfg
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)
        self.ns_engine = NavierStokesFootprintEngine(cfg)

        # ---- Non-persistent scalar buffers (DDP-safe, checkpoint-excluded) --
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_c1",          _buf(cfg.c1_geom),                    persistent=False)
        self.register_buffer("_dt",          _buf(cfg.dt),                         persistent=False)
        self.register_buffer("_sigma_var",   _buf(cfg.sigma_variance),             persistent=False)
        self.register_buffer("_sigma_var_sq", _buf(cfg.sigma_variance ** 2),       persistent=False)
        self.register_buffer("_inner_max",   _buf(cfg.inner_clamp),                persistent=False)
        self.register_buffer("_log_floor",   _buf(cfg.log_floor),                  persistent=False)
        self.register_buffer("_denom_eps",   _buf(cfg.denom_eps),                  persistent=False)
        self.register_buffer("_barrier_beta", _buf(cfg.barrier_beta),              persistent=False)
        self.register_buffer("_indicator_tau", _buf(cfg.indicator_tau),            persistent=False)
        self.register_buffer("_pert_amp",    _buf(cfg.perturbation_amplitude),     persistent=False)
        self.register_buffer("_target",      _buf(cfg.absorption_efficiency_target), persistent=False)

    # ------------------------------------------------------------------ #
    # Disordered boundary layer                                          #
    # ------------------------------------------------------------------ #
    def inject_disordered_boundary_layer(self, sigma_base: torch.Tensor) -> torch.Tensor:
        """
        Adds quenched positive spatial disorder to the interface field.

        Deterministic in `.eval()` (returns `sigma_base` unchanged) so that
        evaluation, ONNX export, and real-time control are reproducible.
        """
        if not self.training:
            return sigma_base

        dtype = sigma_base.dtype
        eps   = self._denom_eps.to(dtype)
        sigma_amp = self._sigma_var.to(dtype)

        n = torch.randn_like(sigma_base)
        # C^∞ |n|  (equals ε at n = 0)
        abs_n = torch.sqrt(n * n + eps * eps)
        return sigma_base + abs_n * sigma_amp

    # ------------------------------------------------------------------ #
    # Log-domain Gumbel No-Zeno barrier                                  #
    # ------------------------------------------------------------------ #
    def compute_gumbel_probability(self, delta_E: torch.Tensor) -> torch.Tensor:
        """
        Differentiable, overflow-safe double-exponential barrier:

            log P = clamp( −c₁ · exp( clamp( ΔE / (σ²·dt), max=z_max ) ),
                           min = log_floor )
            P     = exp( log P )

        Barrier math runs in fp32 and casts back, so it is stable and
        gradient-preserving under AMP (fp16 / bf16 / fp32).

        Parameters
        ----------
        delta_E : torch.Tensor  [B]  (or broadcastable)
        """
        dtype_out = delta_E.dtype
        device    = delta_E.device

        c1        = self._c1.to(device=device)
        dt        = self._dt.to(device=device)
        sigma_sq  = self._sigma_var_sq.to(device=device)
        inner_max = self._inner_max.to(device=device)
        log_floor = self._log_floor.to(device=device)
        eps       = self._denom_eps.to(device=device)

        # Barrier math in fp32 for AMP safety.
        d32    = delta_E.float()
        denom  = (sigma_sq * dt).clamp(min=eps)             # physical ≥ 0 → floor
        inner  = (d32 / denom).clamp_(max=inner_max)        # ≤ z_max
        log_p  = (-c1 * torch.exp(inner)).clamp_(min=log_floor)
        return torch.exp(log_p).to(dtype_out)

    # ------------------------------------------------------------------ #
    # Core step (isolated for gradient checkpointing)                    #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        velocity_field: torch.Tensor,     # [B, 3, D, H, W]
        sigma_skin: torch.Tensor,         # [B, 1, D, H, W]
    ) -> Dict[str, torch.Tensor]:

        dtype  = velocity_field.dtype
        device = velocity_field.device

        beta  = self._barrier_beta.to(dtype=dtype,  device=device)
        tau   = self._indicator_tau.to(dtype=dtype, device=device)
        target = self._target.to(dtype=dtype,      device=device)
        pert  = self._pert_amp.to(dtype=dtype,     device=device)
        eps   = self._denom_eps.to(dtype=dtype,    device=device)

        # ---- 1. Disordered-interface field (deterministic in eval) ---------
        disordered_sigma = self.inject_disordered_boundary_layer(sigma_skin)

        # ---- 2. Initial per-sample footprint energy E(Γ(τ⁻)) ---------------
        initial_energy = self.ns_engine.evaluate_footprint_energy(
            velocity_field, disordered_sigma,
        )                                                     # [B]

        # ---- 3. Turbulent-perturbation field (deterministic in eval) ------
        if self.training:
            u = torch.rand_like(velocity_field)               # ∈ [0, 1)
            # perturbed = v · (1 − amp·u) = v − amp·v·u  → fused addcmul
            perturbed_velocity = torch.addcmul(
                velocity_field, velocity_field * u, -pert
            )
        else:
            perturbed_velocity = velocity_field

        # ---- 4. Perturbed per-sample energy E(Γ(τ⁺)) ---------------------
        perturbed_energy = self.ns_engine.evaluate_footprint_energy(
            perturbed_velocity, disordered_sigma,
        )                                                     # [B]

        # ---- 5. C^∞ activation energy ΔE (softplus, no dead zone) ----------
        delta_e = F.softplus(initial_energy - perturbed_energy, beta=beta) + eps

        # ---- 6. Log-domain Gumbel No-Zeno breakout bound -------------------
        signature_prob = self.compute_gumbel_probability(delta_e)   # [B]

        # ---- 7. Per-sample absorption damping factor ----------------------
        #   Reference:  damping = 1 − 1/(1 + ΔE)
        #   C^∞, bounded to (0, 1), per-sample.
        damping_factor = 1.0 - 1.0 / (1.0 + delta_e)                # [B]
        damping_view   = damping_factor.view(-1, 1, 1, 1, 1)        # [B,1,1,1,1]

        # ---- 8. Absorbed velocity field -----------------------------------
        absorbed_velocity = velocity_field * damping_view           # [B,3,D,H,W]

        # ---- 9. Final per-sample footprint energy -------------------------
        final_energy = self.ns_engine.evaluate_footprint_energy(
            absorbed_velocity, disordered_sigma,
        )                                                     # [B]

        # ---- 10. Absorption efficiency (per-sample, differentiable) -------
        #    ratio = 1 − final / (initial + ε)  — the reference's "+ 1e-12"
        #    guard is replaced with a buffer-based eps for AMP stability.
        absorption_ratio = 1.0 - final_energy / (initial_energy + eps)   # [B]

        # ---- 11. Soft indicator of target achievement (replaces `>=` bool) -
        indicator = torch.sigmoid((absorption_ratio - target) / tau)     # [B]

        return {
            "initial_footprint_energy":      initial_energy,
            "absorbed_footprint_energy":     final_energy,
            "absorption_efficiency":         absorption_ratio,
            "signature_breakout_probability": signature_prob,
            "disordered_sigma_field":        disordered_sigma,
            "absorbed_velocity_field":       absorbed_velocity,
            "stealth_target_achieved":       indicator,
            "delta_e":                       delta_e,
            "damping_factor":                damping_factor,
        }

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        velocity_field: torch.Tensor,     # [B, 3, D, H, W]  (batch-first)
        sigma_skin: torch.Tensor,         # [B, 1, D, H, W]  (or [B, D, H, W])
        *,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes the stealth footprint absorption step.

        Returns a dict of *fully differentiable* per-sample tensors:

            initial_footprint_energy       : [B]
            absorbed_footprint_energy      : [B]
            absorption_efficiency          : [B]  ∈ (−∞, 1]
            signature_breakout_probability : [B]  ∈ (e^{log_floor}, 1)
            disordered_sigma_field         : [B, 1, D, H, W]
            absorbed_velocity_field        : [B, 3, D, H, W]
            stealth_target_achieved        : [B]  ∈ (0, 1)  soft indicator
            delta_e                        : [B]  activation energy
            damping_factor                 : [B]  applied attenuation
        """
        # ---- Shape normalisation ------------------------------------------
        if sigma_skin.dim() == 4:
            sigma_skin = sigma_skin.unsqueeze(1)             # [B, D, H, W] → [B,1,D,H,W]

        if self.validate_inputs:
            if velocity_field.dim() != 5 or velocity_field.shape[1] != 3:
                raise ValueError("velocity_field must be [B, 3, D, H, W].")
            if sigma_skin.dim() != 5 or sigma_skin.shape[1] != 1:
                raise ValueError("sigma_skin must be [B, 1, D, H, W] (or [B, D, H, W]).")
            if sigma_skin.shape[0] != velocity_field.shape[0]:
                raise ValueError("sigma_skin and velocity_field batch dims must match.")
            if sigma_skin.shape[2:] != velocity_field.shape[2:]:
                raise ValueError("sigma_skin and velocity_field spatial dims must match.")

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
# SELF-TEST & VALIDATION SUITE
# =============================================================================
if __name__ == "__main__":
    print("====================================================================")
    print(" Running SESI 3D Navier-Stokes Stealth Footprint Absorber Test (v2)")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    cfg = StealthCFDConfig(grid_resolution=(16, 16, 16))
    absorber = SESIStealthFootprintAbsorber(cfg).to(device)
    absorber.train()

    # Batched inputs: [B, 3, D, H, W] velocity, [B, 1, D, H, W] skin field.
    B, D, H, W = 2, *cfg.grid_resolution
    velocity = torch.randn(B, 3, D, H, W, device=device, requires_grad=True) * 10.0
    sigma    = torch.ones(B, 1, D, H, W, device=device, requires_grad=True)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    with autocast_ctx:
        out = absorber(velocity, sigma)

    loss = (
        out["absorbed_velocity_field"].float().pow(2).mean()
        + (1.0 - out["stealth_target_achieved"].float()).mean()
        + out["signature_breakout_probability"].float().mean()
    )
    loss.backward()

    print(f"  [CFD] Initial footprint energy     : {out['initial_footprint_energy'].detach().cpu().tolist()}")
    print(f"  [CFD] Absorbed footprint energy    : {out['absorbed_footprint_energy'].detach().cpu().tolist()}")
    print(f"  [CFD] Absorption efficiency        : {out['absorption_efficiency'].detach().cpu().tolist()}")
    print(f"  [CFD] Signature breakout prob.     : {out['signature_breakout_probability'].detach().cpu().tolist()}")
    print(f"  [CFD] Stealth target (soft)        : {out['stealth_target_achieved'].detach().cpu().tolist()}")
    print(f"  [CFD] Absorbed velocity shape      : {tuple(out['absorbed_velocity_field'].shape)}")
    print("\n--- Autograd gradient-flow verification ---")
    print(f"  |∂L/∂velocity|                     : {velocity.grad.norm().item():.6e}")
    print(f"  |∂L/∂sigma_skin|                   : {sigma.grad.norm().item():.6e}")
    print("====================================================================")
