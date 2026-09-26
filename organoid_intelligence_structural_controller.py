# =============================================================================
# Organoid Intelligence (OI) Structural Controller — PyTorch-Native SESI
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Framework    : Self-Evolving Structural Interfaces (SESI)
# Module       : Autopilot & Aero-Topological Interface Controller
# Developer    : PAI and Yoon A Limsuwan — MSPS NETWORK
#                "My Soul Move By Power of Holy Spirit"
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Production / DDP-Ready)
# =============================================================================
"""
Fully differentiable PyTorch controller coupling
3D structural calculus (PFC3D, CH3D, ThinFilm3D, Dynamic Langevin SDE)
with Controlled Self-Organized Criticality (CSOC).

Production guarantees
---------------------
1. Full C^∞ differentiability — every hard `clamp`, `abs`, and `pow` has been
   replaced by a smooth surrogate that preserves the gradient in the physically
   active regime:
       clamp(x, min=c)      →  c + softplus(x − c, β)         (smooth floor)
       clamp(x, max=c)      →  c − softplus(c − x, β)         (smooth ceiling)
       |x|                  →  sqrt(x² + ε²)
       pow(x, 3)            →  x · x · x                      (faster, exact)
2. Rheological barrier is now numerically *bounded* in log-space — the original
   `stress / (σ_c − stress)` could overflow AMP (fp16/bf16) as σ → σ_c. The
   new smooth positive surrogate `σ / (ε + softplus(σ_c − σ, β))` stays finite,
   differentiable everywhere, and matches the physical limit σ ≪ σ_c.
3. Viscosity is evaluated in log-domain with a smooth saturating ceiling to
   guarantee `exp(·) ≤ e^{log_eta_max}` under bf16/fp16/fp32.
4. All constants (kernels, floors, ceilings) live in non-persistent buffers so
   the module is DDP-clean and excluded from checkpoints.
5. `torch.compile(mode="max-autotune")` friendly — no Python-scalar graph
   breaks, no data-dependent control flow, no forced AMP decorator.
6. Deterministic in `.eval()` — stochastic Langevin forcing is zero-mean
   Gaussian in training, identically zero in inference.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    ctrl = DifferentiableOIController(cfg).to(rank)
    ctrl = torch.compile(ctrl, mode="max-autotune")                 # optional
    ctrl = torch.nn.parallel.DistributedDataParallel(
        ctrl, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = ctrl(state, measured_tau)
    (out["phi"].pow(2).mean() + out["csoc_loss"]).backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PhysicsConfig",
    "DifferentiableSpatialOperators3D",
    "DifferentiableOIController",
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class PhysicsConfig:
    """Physical domain, solver, and CSOC parameters (SI-consistent units)."""
    grid_size: Tuple[int, int, int] = (32, 32, 32)
    dx: float = 1.0
    dt: float = 0.01

    # Physical stress / rheology
    sigma_critical: float = 10.0      # yield-stress boundary
    eta_0: float = 1.0                # base viscosity
    T_0: float = 0.5                  # base noise temperature
    gamma_stress: float = 0.5         # stress–viscosity amplification
    kappa_temp: float = 0.2           # temperature–gradient coupling

    # CSOC scaling law  τ̂(α) = τ_∞ + A · exp(−k·α)
    tau_target: float = 1.5
    tau_inf: float = 1.2
    A_csoc: float = 0.8
    k_csoc: float = 0.5

    # ---- C^∞ surrogates (new; safe defaults) -----------------------------
    barrier_beta: float = 100.0       # sharpness of the smooth stress barrier
    mobility_beta: float = 200.0      # sharpness of the smooth film mobility
    stress_eps: float = 1e-4          # ε in sqrt(σ² + ε²) and floor of denom
    mobility_floor: float = 1e-4      # minimum thin-film thickness
    log_eta_max: float = 20.0         # ceiling on log η (bf16-safe)


# =============================================================================
# 3D finite-difference operators (single fused conv3d per derivative)
# =============================================================================
class DifferentiableSpatialOperators3D(nn.Module):
    """
    C^∞ 3D finite-difference stencils implemented as fused conv3d kernels.

    Improvements over the reference implementation
    ----------------------------------------------
    · Laplacian and each gradient component are computed by a single conv3d
      over a circularly-padded input → 1 kernel launch vs. 6 slice kernels.
    · Pre-divided by `dx²` (Laplacian) or `2·dx` (central differences) once at
      construction, so no per-step scalar multiplication appears in the graph.
    · Kernels are non-persistent buffers; they follow the input device/dtype.
    """

    def __init__(self, dx: float = 1.0) -> None:
        super().__init__()
        if not (math.isfinite(dx) and dx > 0.0):
            raise ValueError("dx must be a positive finite scalar.")
        self.dx = float(dx)

        # 7-point Laplacian, pre-divided by dx²
        lap = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        lap[0, 0, 1, 1, 1] = -6.0
        lap[0, 0, 0, 1, 1] = lap[0, 0, 2, 1, 1] = 1.0
        lap[0, 0, 1, 0, 1] = lap[0, 0, 1, 2, 1] = 1.0
        lap[0, 0, 1, 1, 0] = lap[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("_lap_kernel", lap / (dx * dx), persistent=False)

        # Central-difference kernels per axis, pre-divided by 2·dx
        inv_2dx = 1.0 / (2.0 * dx)

        def _grad(axis: int) -> torch.Tensor:
            k = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
            if axis == 0:
                k[0, 0, 0, 1, 1], k[0, 0, 2, 1, 1] = -1.0, 1.0
            elif axis == 1:
                k[0, 0, 1, 0, 1], k[0, 0, 1, 2, 1] = -1.0, 1.0
            else:
                k[0, 0, 1, 1, 0], k[0, 0, 1, 1, 2] = -1.0, 1.0
            return k * inv_2dx

        self.register_buffer("_grad_d", _grad(0), persistent=False)
        self.register_buffer("_grad_h", _grad(1), persistent=False)
        self.register_buffer("_grad_w", _grad(2), persistent=False)

    # ------------------------------------------------------------------ #
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        # Circular (periodic) padding — matches the SESI reference BC.
        return F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """∇²x via a single fused conv3d (circular BC)."""
        return F.conv3d(self._pad(x), self._lap_kernel.to(x.dtype))

    def gradient_components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (∂_D x, ∂_H x, ∂_W x) as three fused conv3d calls."""
        xp = self._pad(x)
        gd = F.conv3d(xp, self._grad_d.to(x.dtype))
        gh = F.conv3d(xp, self._grad_h.to(x.dtype))
        gw = F.conv3d(xp, self._grad_w.to(x.dtype))
        return gd, gh, gw

    def gradient_sq_norm(self, x: torch.Tensor) -> torch.Tensor:
        """|∇x|² via three fused conv3d calls + addcmul (no intermediates)."""
        gd, gh, gw = self.gradient_components(x)
        return torch.addcmul(torch.addcmul(gd * gd, gh, gh), gw, gw)


# =============================================================================
# Differentiable OI controller
# =============================================================================
class DifferentiableOIController(nn.Module):
    """
    Fully differentiable SESI controller for Organoid Intelligence.

    Learnable parameters
    --------------------
    alpha_kernel : scalar  — CSOC avalanche-exponent controller.

    All other quantities are non-persistent buffers or Python dataclass fields,
    so DDP's gradient all-reduce effectively syncs a single scalar per step —
    maximally cheap for multi-GPU training.
    """

    def __init__(self, config: PhysicsConfig, *, validate_inputs: bool = True) -> None:
        super().__init__()
        self.cfg = config
        self.ops = DifferentiableSpatialOperators3D(dx=config.dx)
        self.validate_inputs = bool(validate_inputs)

        # ---- Trainable CSOC parameter (the only learnable state) ----------
        self.alpha_kernel = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # ---- Non-persistent scalar buffers (DDP-safe, checkpoint-excluded) --
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_sigma_c",   _buf(config.sigma_critical), persistent=False)
        self.register_buffer("_eta_0",     _buf(config.eta_0),          persistent=False)
        self.register_buffer("_T_0",       _buf(config.T_0),            persistent=False)
        self.register_buffer("_gamma_s",   _buf(config.gamma_stress),   persistent=False)
        self.register_buffer("_kappa_t",   _buf(config.kappa_temp),     persistent=False)
        self.register_buffer("_tau_inf",   _buf(config.tau_inf),        persistent=False)
        self.register_buffer("_A_csoc",    _buf(config.A_csoc),         persistent=False)
        self.register_buffer("_k_csoc",    _buf(config.k_csoc),         persistent=False)
        self.register_buffer("_tau_target", _buf(config.tau_target),    persistent=False)
        self.register_buffer("_log_eta_0", _buf(math.log(config.eta_0)), persistent=False)
        self.register_buffer("_log_eta_max", _buf(config.log_eta_max),  persistent=False)
        self.register_buffer("_stress_eps",  _buf(config.stress_eps),   persistent=False)
        self.register_buffer("_mob_floor",   _buf(config.mobility_floor), persistent=False)
        self.register_buffer("_dt",          _buf(config.dt),           persistent=False)

        # Betas are stored as fp32 buffers so they can be cast per-call without
        # triggering Python-float graph breaks under torch.compile.
        self.register_buffer("_barrier_beta",  _buf(config.barrier_beta),  persistent=False)
        self.register_buffer("_mobility_beta", _buf(config.mobility_beta), persistent=False)

    # ------------------------------------------------------------------ #
    # C^∞ primitives                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _smooth_abs(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """C^∞ |x|  (equals eps at x = 0)."""
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def _smooth_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """C^∞ max(x, floor) — exact for x ≫ floor, differentiable at x = floor."""
        return floor + F.softplus(x - floor, beta=beta)

    @staticmethod
    def _smooth_ceiling(x: torch.Tensor, ceil: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """C^∞ min(x, ceil) — exact for x ≪ ceil, differentiable at x = ceil."""
        return ceil - F.softplus(ceil - x, beta=beta)

    # ------------------------------------------------------------------ #
    # Stress field                                                       #
    # ------------------------------------------------------------------ #
    def compute_stress_field(self, phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Local mechanical stress  σ(x) = |∇φ|² + |ψ|.

        Replaces the original hard `clamp(stress, max=σ_c − 1e-4)` — which
        killed gradients above the ceiling — with a smooth saturating ceiling
        that remains differentiable in the entire physical range.
        """
        eps = self._stress_eps.to(phi.dtype)
        sigma_c = self._sigma_c.to(phi.dtype)
        beta = self._barrier_beta.to(phi.dtype)

        grad_phi_sq = self.ops.gradient_sq_norm(phi)
        psi_abs = self._smooth_abs(psi, eps)
        stress_raw = grad_phi_sq + psi_abs

        # C^∞ ceiling just below σ_c — preserves gradient everywhere.
        return self._smooth_ceiling(stress_raw, sigma_c - eps, beta)

    # ------------------------------------------------------------------ #
    # Rheological Langevin forcing                                       #
    # ------------------------------------------------------------------ #
    def compute_rheological_langevin(
        self,
        stress: torch.Tensor,
        noise_seed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        State-dependent Langevin forcing with the reparameterization trick.

        Barrier behaviour
        -----------------
        Original code:  η(σ) = η₀ · exp(γ · σ / (σ_c − σ))   ← overflows near σ_c.
        Now:            η(σ) = exp( log(η₀) + γ · σ / (ε + softplus(σ_c − σ, β)) )
                        with `log η` smoothly capped at `log_eta_max`.

        This reproduces σ/(σ_c − σ) exactly in the physical limit σ ≪ σ_c and
        degrades smoothly (rather than diverging) as the system approaches the
        critical yield boundary.
        """
        dtype = stress.dtype
        eps   = self._stress_eps.to(dtype)
        sigma_c = self._sigma_c.to(dtype)
        beta  = self._barrier_beta.to(dtype)
        gamma = self._gamma_s.to(dtype)
        T_0   = self._T_0.to(dtype)
        kappa = self._kappa_t.to(dtype)
        log_eta_0 = self._log_eta_0.to(dtype)
        log_eta_max = self._log_eta_max.to(dtype)

        # ---- Smooth positive denominator: σ_c − σ → softplus ----
        denom = eps + F.softplus(sigma_c - stress, beta=beta)
        stress_ratio = stress / denom

        # ---- Viscosity in log-domain with smooth saturating ceiling --------
        log_eta_raw = log_eta_0 + gamma * stress_ratio
        log_eta = log_eta_max - F.softplus(log_eta_max - log_eta_raw, beta=beta)
        eta_sigma = torch.exp(log_eta)

        # ---- Temperature modulated by stress gradient ----------------------
        grad_stress_sq = self.ops.gradient_sq_norm(stress)
        T_sigma = T_0 * (1.0 + kappa * grad_stress_sq)

        # ---- Reparameterized fluctuation: sqrt(2T/η) · ε,  ε ~ N(0, I) -----
        stochastic_amplitude = torch.sqrt(2.0 * T_sigma / eta_sigma)

        if not self.training:
            # Deterministic inference — zero forcing, no RNG state touched.
            stochastic_forcing = torch.zeros_like(stress)
        else:
            eps_noise = noise_seed if noise_seed is not None else torch.randn_like(stress)
            stochastic_forcing = stochastic_amplitude * eps_noise

        return stochastic_forcing, eta_sigma

    # ------------------------------------------------------------------ #
    # Structural field updates                                           #
    # ------------------------------------------------------------------ #
    def update_structural_pfc3d(self, psi: torch.Tensor) -> torch.Tensor:
        """Phase-Field-Crystal (Swift–Hohenberg form):  ∂_t ψ = ∇²ψ + ψ − ψ³."""
        lap_psi = self.ops.laplacian(psi)
        dt = self._dt.to(psi.dtype)
        rhs = lap_psi + psi - psi * psi * psi
        return torch.addcmul(psi, rhs, dt)

    def update_structural_ch3d(self, c: torch.Tensor) -> torch.Tensor:
        """Cahn–Hilliard conserved gradient flow:  ∂_t c = ∇²(c³ − c − ∇²c)."""
        lap_c = self.ops.laplacian(c)
        mu = c * c * c - c - lap_c
        ch_flux = self.ops.laplacian(mu)
        dt = self._dt.to(c.dtype)
        return torch.addcmul(c, ch_flux, dt)

    def update_structural_thinfilm3d(self, h: torch.Tensor) -> torch.Tensor:
        """
        Thin-film lubrication:  ∂_t h = −∇²( h³ ∇²h ).

        The original `clamp(h, min=1e-4)` is replaced by a C^∞ smooth floor so
        gradients survive the physically relevant regime h > 0 and behave
        gracefully as h → 0.
        """
        dtype = h.dtype
        beta  = self._mobility_beta.to(dtype)
        floor = self._mob_floor.to(dtype)

        h_pos = self._smooth_floor(h, floor, beta)
        mobility = h_pos * h_pos * h_pos

        lap_h = self.ops.laplacian(h)
        thinfilm_rhs = self.ops.laplacian(mobility * lap_h)

        dt = self._dt.to(dtype)
        return torch.addcmul(h, thinfilm_rhs, -dt)

    # ------------------------------------------------------------------ #
    # CSOC criticality loss                                              #
    # ------------------------------------------------------------------ #
    def compute_csoc_loss(self, measured_tau: torch.Tensor) -> torch.Tensor:
        """
        Differentiable CSOC loss:

            τ̂(α) = τ_∞ + A · exp(−k · α)
            L     = (τ̂(α) − τ_target)² + (τ_meas − τ_target)²
        """
        tau_inf  = self._tau_inf.to(measured_tau.dtype)
        A_csoc   = self._A_csoc.to(measured_tau.dtype)
        k_csoc   = self._k_csoc.to(measured_tau.dtype)
        target   = self._tau_target.to(measured_tau.dtype)

        predicted_tau = tau_inf + A_csoc * torch.exp(-k_csoc * self.alpha_kernel.to(measured_tau.dtype))

        # Explicit squared errors — cheaper than F.mse_loss (no reduction graph).
        return (predicted_tau - target).pow(2) + (measured_tau - target).pow(2)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        measured_tau: torch.Tensor,
        noise_seed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single fully differentiable step.

        Parameters
        ----------
        state : dict with 5D tensors [B, C, D, H, W]
            'phi' : neural potential field
            'psi' : phase-field-crystal order parameter
            'c'   : Cahn–Hilliard concentration
            'h'   : thin-film substrate height (≥ 0)
        measured_tau : torch.Tensor
            Measured avalanche exponent (scalar or broadcastable).
        noise_seed : torch.Tensor, optional
            Pre-sampled standard-normal noise, same shape as `phi`.
            If `None` and `self.training == True`, a fresh sample is drawn.
        """
        phi, psi, c, h = state["phi"], state["psi"], state["c"], state["h"]

        if self.validate_inputs:
            for name, t in (("phi", phi), ("psi", psi), ("c", c), ("h", h)):
                if t.dim() != 5:
                    raise ValueError(f"state['{name}'] must be 5D [B, C, D, H, W].")
            if not (phi.shape[2:] == psi.shape[2:] == c.shape[2:] == h.shape[2:]):
                raise ValueError("state tensors must share spatial shape.")

        # ---- 1. Structural field updates ----------------------------------
        psi_next = self.update_structural_pfc3d(psi)
        c_next   = self.update_structural_ch3d(c)
        h_next   = self.update_structural_thinfilm3d(h)

        # ---- 2. Stress field (C^∞ saturating ceiling) ----------------------
        stress = self.compute_stress_field(phi, psi_next)

        # ---- 3. Rheological Langevin forcing (reparameterized) -----------
        noise_term, eta_field = self.compute_rheological_langevin(stress, noise_seed=noise_seed)

        # ---- 4. Neural potential dynamics (fused addcmul) ------------------
        #   ∂_t φ = η⁻¹ · (−φ + c·h) + ξ(x, t)
        deterministic_force = torch.addcmul(-phi, c_next, h_next)
        dt = self._dt.to(phi.dtype)
        dphi_dt = deterministic_force / eta_field + noise_term
        phi_next = torch.addcmul(phi, dphi_dt, dt)

        # ---- 5. Differentiable CSOC criticality loss ----------------------
        csoc_loss = self.compute_csoc_loss(measured_tau)

        return {
            "phi":        phi_next,
            "psi":        psi_next,
            "c":          c_next,
            "h":          h_next,
            "stress":     stress,
            "eta":        eta_field,
            "csoc_loss":  csoc_loss,
        }


# =============================================================================
# Verification & autograd gradient-flow smoke test
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-OI v2] Running Differentiable OI Controller on: {device}")

    cfg = PhysicsConfig(grid_size=(16, 16, 16))
    controller = DifferentiableOIController(cfg).to(device)
    controller.train()  # enable stochastic Langevin forcing

    shape = (1, 1, 16, 16, 16)
    state = {
        "phi": torch.randn(shape, device=device, requires_grad=True),
        "psi": torch.randn(shape, device=device, requires_grad=True),
        "c":   torch.randn(shape, device=device, requires_grad=True),
        "h":   torch.ones (shape, device=device, requires_grad=True),
    }
    measured_tau = torch.tensor(1.45, device=device, requires_grad=True)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
        out = controller(state, measured_tau)

    total_loss = out["phi"].float().pow(2).mean() + out["csoc_loss"].float()
    total_loss.backward()

    print("\n--- Autograd gradient-flow verification ---")
    print(f"  total loss              : {total_loss.item():.6f}")
    print(f"  |∂L/∂phi|  (mean)       : {state['phi'].grad.abs().mean().item():.6e}")
    print(f"  |∂L/∂psi|  (mean)       : {state['psi'].grad.abs().mean().item():.6e}")
    print(f"  |∂L/∂c|    (mean)       : {state['c'].grad.abs().mean().item():.6e}")
    print(f"  |∂L/∂h|    (mean)       : {state['h'].grad.abs().mean().item():.6e}")
    print(f"  ∂L/∂alpha_kernel        : {controller.alpha_kernel.grad.item():.6f}")
    print(f"  ∂L/∂measured_tau        : {measured_tau.grad.item():.6f}")
    print("\nStatus: autograd graph fully connected — module is C^∞ differentiable.")
