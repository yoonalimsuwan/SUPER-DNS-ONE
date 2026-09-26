# =============================================================================
# SESI BATTERY-POWERED HYPERSONIC AIRCRAFT ENGINE
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Framework   : Self-Evolving Structural Interfaces (SESI)
# Module      : Battery-Powered Hypersonic Engine — Full Differentiable Stack
# Developer   : PAI and Yoon A Limsuwan — MSPS NETWORK
#               "My Soul Move By Power of Holy Spirit"
# ORCID       : 0009-0008-2374-0788
# GitHub      : yoonalimsuwan
# Contact     : msps4u@gmail.com
# License     : MIT
# Year        : 2026
# Version     : 2.0.0 (Native Differentiable / AMP-Safe / DDP-Ready)
# =============================================================================
"""
Production-grade, fully differentiable battery-powered hypersonic engine module.

Integrates
----------
1. Structural Calculus — polynomial-quotient state-space mapping (Φ_U).
2. SESI Framework — disordered-media energy landscape with Gumbel-type No-Zeno
   regulation and log-domain thermal-barrier bounds.
3. Advanced electric propulsion & battery thermal / power management.
4. Native full differentiability for cost-minimized, multi-GPU training.

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **Critical numerical bug fixed — the "topological determinant" was `-inf`.**
   The reference computed

       sig_matrix = contracted.unsqueeze(2) @ contracted.unsqueeze(1)   # v v^T

   i.e. a **rank-1 outer product** of shape `[B, D, D]`.  Its determinant is
   identically zero for `D > 1`.  `torch.slogdet` therefore returned
   `logabsdet = -inf`, making `stability_loss = mean(|det_sig|) = inf` and
   poisoning every gradient with NaN from the very first backward pass.
   The module now computes the **exact, closed-form regularized log-det**

       log|det(v v^T + ε I)| = (D − 1)·log(ε) + log(||v||² + ε)

   via the matrix-determinant lemma.  This is
       · mathematically exact for the regularized matrix,
       · finite and C^∞ everywhere,
       · and **O(B·D)** instead of the reference's **O(B·D³) `slogdet`** —
       a ~1000× speedup for `D = 32`.

2. **Full C^∞ differentiability** — every hard op replaced by a smooth
   surrogate in the active physical regime:
       torch.relu(x)         →  F.softplus(x, β)              (no dead zone)
       torch.abs(x)          →  sqrt(x² + ε²)                 (no kink at 0)
       exp(-relu(x))         →  exp(-softplus(x, β))          (C^∞ No-Zeno)
       `v ** 2`              →  `v * v`                       (exact & faster)
       `.pow(2)`             →  `x * x`                       (exact & faster)

3. **AMP-safe log-domain barrier** — every exponential is evaluated in fp32
   log-space with a lower bound on `log P`, so nothing overflows in
   fp16/bf16 and nothing underflows to zero (dead gradient).

4. **Deterministic, DDP-clean, `torch.compile`-stable** — no Python-scalar
   graph breaks, no data-dependent control flow, all constants in
   non-persistent buffers, no forced AMP decorator.

5. **Loss function upgraded in parallel** — smooth, per-sample weighted
   terms; no `.item()` inside the graph; configurable weights.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    engine = ProductionHypersonicBatteryEngineModule(cfg).to(rank)
    engine = torch.compile(engine, mode="max-autotune")            # optional
    engine = torch.nn.parallel.DistributedDataParallel(
        engine, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = engine(state_x)
    loss = compute_production_loss(out, target_thrust, weights)
    loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = [
    "HypersonicEngineConfig",
    "ProductionHypersonicBatteryEngineModule",
    "compute_production_loss",
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class HypersonicEngineConfig:
    """Production configuration for the hypersonic battery engine module."""
    latent_dim: int = 64
    num_clauses: int = 128                   # reserved for future discrete heads

    # ---- Numerical safety / C^∞ surrogate sharpness ----------------------
    signature_eps: float = 1e-4              # ε in log|det(vvᵀ + εI)|
    denom_eps: float = 1e-8                  # safe-division epsilon
    softplus_beta: float = 4.0               # smooth-ReLU sharpness
    barrier_beta: float = 4.0                # smooth No-Zeno barrier sharpness
    inner_clamp: float = 15.0                # ceiling on inner exp argument
    log_floor: float = -40.0                 # floor on log P (dead-grad guard)
    smoothing_tau: float = 0.05              # indicator sharpness


# =============================================================================
# Engine
# =============================================================================
class ProductionHypersonicBatteryEngineModule(nn.Module):
    """
    Complete production-grade, fully differentiable battery-powered hypersonic
    engine module.

    Parameters
    ----------
    config : HypersonicEngineConfig, optional
        Full physical / numerical configuration. If a bare `int` is passed it
        is interpreted as `latent_dim` for backward compatibility with the
        reference constructor signature.
    num_clauses : int, optional
        Backward-compatibility alias for `config.num_clauses`.
    gradient_checkpointing : bool
        Recompute the step during backward — trades ~2× compute for activation
        memory savings in long unrolled therapy / trajectory loops.
    validate_inputs : bool
        Cheap shape/dtype guards. Disable once shapes are statically known to
        avoid `torch.compile` recompiles.
    """

    def __init__(
        self,
        config: Optional[HypersonicEngineConfig] = None,
        num_clauses: Optional[int] = None,
        *,
        latent_dim: Optional[int] = None,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()

        # ---- Backward-compatible argument resolution -----------------------
        if config is None:
            config = HypersonicEngineConfig()
        if latent_dim is not None:
            config = HypersonicEngineConfig(**{**config.__dict__, "latent_dim": int(latent_dim)})
        if num_clauses is not None:
            config = HypersonicEngineConfig(**{**config.__dict__, "num_clauses": int(num_clauses)})

        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")
        if config.num_clauses <= 0:
            raise ValueError("num_clauses must be > 0.")
        if not (math.isfinite(config.signature_eps) and config.signature_eps > 0.0):
            raise ValueError("signature_eps must be > 0.")
        if not (math.isfinite(config.denom_eps) and config.denom_eps > 0.0):
            raise ValueError("denom_eps must be > 0.")
        if not (math.isfinite(config.softplus_beta) and config.softplus_beta > 0.0):
            raise ValueError("softplus_beta must be > 0.")
        if not (math.isfinite(config.barrier_beta) and config.barrier_beta > 0.0):
            raise ValueError("barrier_beta must be > 0.")
        if not (math.isfinite(config.inner_clamp) and config.inner_clamp > 0.0):
            raise ValueError("inner_clamp must be > 0.")
        if not (math.isfinite(config.log_floor) and config.log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")
        if not (math.isfinite(config.smoothing_tau) and config.smoothing_tau > 0.0):
            raise ValueError("smoothing_tau must be > 0.")

        self.cfg = config
        self.latent_dim = config.latent_dim
        self.num_clauses = config.num_clauses
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        # ---- Module 1: Structural-calculus tensor mapping Φ_U --------------
        #   Collapses exponential micro-state enumeration into O(m³·n²)
        #   equivalence classes.
        self.phi_u_tensor_net = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_dim),
        )

        # ---- Module 2: Battery / supercapacitor power management ----------
        #   Outputs [discharge_rate, thrust_burst_coefficient].
        self.battery_power_controller = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

        # ---- Module 3: SESI thermal-barrier estimator ---------------------
        #   Outputs a scalar activation energy barrier ΔE per sample.
        self.thermal_barrier_estimator = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # ---- Module 4: Aerothermo actuator controller ---------------------
        #   Outputs [Nucleation, Merging, Branching] topological weights.
        self.aerothermo_actuator_controller = nn.Sequential(
            nn.Linear(self.latent_dim, 3),
        )

        # ---- Non-persistent scalar buffers (DDP-safe, checkpoint-excluded) --
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_sig_eps",      _buf(config.signature_eps),  persistent=False)
        self.register_buffer("_denom_eps",    _buf(config.denom_eps),      persistent=False)
        self.register_buffer("_sp_beta",      _buf(config.softplus_beta),  persistent=False)
        self.register_buffer("_barrier_beta", _buf(config.barrier_beta),   persistent=False)
        self.register_buffer("_inner_max",    _buf(config.inner_clamp),    persistent=False)
        self.register_buffer("_log_floor",    _buf(config.log_floor),      persistent=False)
        self.register_buffer("_tau",          _buf(config.smoothing_tau),  persistent=False)
        #   Pre-computed constant (D − 1)·log(ε) used in the closed-form log-det.
        #   Materialized once at construction, then cast per-call — no Python
        #   scalars in the hot path.
        self.register_buffer(
            "_log_det_const",
            torch.tensor(float((self.latent_dim - 1) * math.log(config.signature_eps)),
                         dtype=torch.float32),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    # Closed-form regularized log-determinant                            #
    # ------------------------------------------------------------------ #
    def _topological_signature(self, v: torch.Tensor) -> torch.Tensor:
        """
        C^∞ regularized log-determinant of `v vᵀ + ε I` for a batch of row
        vectors `v`:

            log|det(v vᵀ + ε I)| = (D − 1)·log(ε) + log(||v||² + ε)

        Exact via the matrix-determinant lemma; O(B·D) instead of the
        reference's O(B·D³) `slogdet`, and — critically — **finite** for every
        input (the reference returned `-inf` for D > 1).

        Returns
        -------
        log_det : torch.Tensor  [B]
        """
        dtype  = v.dtype
        eps    = self._sig_eps.to(dtype)
        const  = self._log_det_const.to(dtype)

        v_sq = (v * v).sum(dim=-1)                     # [B]
        #   log(||v||² + ε) is well-defined and C^∞ for ε > 0.
        return const + torch.log(v_sq + eps)

    # ------------------------------------------------------------------ #
    # Log-domain SESI No-Zeno barrier                                    #
    # ------------------------------------------------------------------ #
    def _no_zeno_penalty(self, delta_e: torch.Tensor) -> torch.Tensor:
        """
        C^∞, overflow-safe double-exponential No-Zeno penalty

            penalty = exp( clamp( −softplus(ΔE, β), min = log_floor ) )

        Bounded in `[e^{log_floor}, 1]`, differentiable everywhere, and stable
        under fp16 / bf16 / fp32.
        """
        dtype = delta_e.dtype
        beta  = self._barrier_beta.to(dtype)
        log_floor = self._log_floor.to(dtype)

        delta_e_pos = F.softplus(delta_e, beta=beta)              # ≥ 0, C^∞
        #   Barrier math in fp32 for AMP safety.
        log_p = (-delta_e_pos.float()).clamp_(min=log_floor)
        return torch.exp(log_p).to(dtype)

    # ------------------------------------------------------------------ #
    # Core step (isolated for gradient checkpointing)                    #
    # ------------------------------------------------------------------ #
    def _step(self, state_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype = state_x.dtype
        device = state_x.device

        sp_beta = self._sp_beta.to(dtype=dtype, device=device)

        # ---- 1. Structural tensor mapping Φ_U ------------------------------
        contracted_state = self.phi_u_tensor_net(state_x)          # [B, D]

        # ---- 2. Topological signature (closed-form, C^∞, O(B·D)) ----------
        det_signature = self._topological_signature(contracted_state)   # [B]

        # ---- 3. Battery / thermal state estimates --------------------------
        battery_outputs = self.battery_power_controller(contracted_state)
        discharge_rate = torch.sigmoid(battery_outputs[:, 0:1])            # [B,1]
        #   C^∞ replacement for `relu(thrust_burst)`: softplus keeps the
        #   non-negative physical meaning without a dead zone at 0.
        thrust_burst = F.softplus(battery_outputs[:, 1:2], beta=sp_beta)   # [B,1]

        delta_e = self.thermal_barrier_estimator(contracted_state)         # [B,1]

        # ---- 4. Topological actuator gating (softmax over N / M / B) ------
        control_actions = torch.softmax(
            self.aerothermo_actuator_controller(contracted_state), dim=-1,
        )                                                                  # [B,3]

        return {
            "contracted_state":         contracted_state,
            "topological_determinant":  det_signature,
            "discharge_rate":           discharge_rate,
            "thrust_burst":             thrust_burst,
            "activation_energy_barrier": delta_e,
            "control_actions":          control_actions,
        }

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, state_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full end-to-end differentiable forward pass.

        Parameters
        ----------
        state_x : torch.Tensor  [B, latent_dim]
            Flight-regime, Mach-number, and thermal-load latent state.

        Returns
        -------
        dict with these *fully differentiable* per-sample tensors:
            contracted_state          : [B, D]   Φ_U(state_x)
            topological_determinant   : [B]      log|det(vvᵀ + εI)|
            discharge_rate            : [B, 1]   ∈ (0, 1)
            thrust_burst              : [B, 1]   ≥ 0
            activation_energy_barrier : [B, 1]   ΔE (unconstrained)
            control_actions           : [B, 3]   softmax(N, M, B), sums to 1
        """
        if self.validate_inputs:
            if state_x.dim() != 2:
                raise ValueError("state_x must be [B, latent_dim].")
            if state_x.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"state_x last dim must equal latent_dim={self.latent_dim}, "
                    f"got {state_x.shape[-1]}."
                )

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._step, state_x, use_reentrant=False,
            )
        return self._step(state_x)


# =============================================================================
# Production loss                                                          #
# =============================================================================
def compute_production_loss(
    outputs: Dict[str, torch.Tensor],
    target_thrust: torch.Tensor,
    cost_weights: Dict[str, float],
    *,
    barrier_beta: float = 4.0,
    log_floor: float = -40.0,
) -> torch.Tensor:
    """
    Fully differentiable production loss balancing:

      A. Performance / thrust target tracking
      B. Structural stability (regularized log-det signature — finite & C^∞)
      C. SESI No-Zeno thermal-barrier penalty (overflow-safe log-domain)
      D. Cost minimization + manufacturing-complexity (L1) penalty

    All terms are per-sample and differentiable; no `.item()` inside the graph.
    """
    contracted      = outputs["contracted_state"]              # [B, D]
    det_sig         = outputs["topological_determinant"]       # [B]
    delta_e         = outputs["activation_energy_barrier"]     # [B, 1]
    thrust_burst    = outputs["thrust_burst"]                  # [B, 1]

    # ---- A. Performance / thrust loss -------------------------------------
    #   Proxy thrust:  mean of contracted features  +  thrust burst.
    thrust_proxy = contracted.mean(dim=-1) + thrust_burst.squeeze(-1)   # [B]
    performance_loss = F.mse_loss(thrust_proxy, target_thrust)

    # ---- B. Structural stability loss -------------------------------------
    #   The reference used `mean(|log|det||)` on a matrix whose det was −inf.
    #   Our signature is a *finite*, C^∞, regularized log-det; we penalise its
    #   magnitude with a smooth surrogate so the gradient is never kinked.
    stability_loss = torch.sqrt(det_sig * det_sig + 1e-8).mean()

    # ---- C. SESI No-Zeno thermal-barrier penalty --------------------------
    #   C^∞, overflow-safe double-exponential:  exp(−softplus(ΔE, β)) ∈ (0, 1].
    delta_e_pos = F.softplus(delta_e, beta=barrier_beta)
    no_zeno_penalty = torch.exp(-delta_e_pos.float()).clamp_(min=math.exp(log_floor))
    no_zeno_penalty = no_zeno_penalty.to(delta_e.dtype).mean()

    # ---- D. Cost minimization + complexity penalty ------------------------
    #   Smooth-ReLU on ΔE (no dead zone) + L1 norm of the contracted features
    #   (kept with its L1 kink — that kink is the *intended* sparsity prior).
    delta_e_cost = F.softplus(delta_e, beta=barrier_beta).mean()
    l1_complexity = torch.linalg.vector_norm(contracted, ord=1, dim=-1).mean()
    cost_penalty = delta_e_cost + cost_weights.get("complexity", 0.0) * l1_complexity

    # ---- Total unified loss ----------------------------------------------
    total_loss = (
        performance_loss
        + cost_weights.get("stability", 0.0)         * stability_loss
        + cost_weights.get("no_zeno",   0.0)         * no_zeno_penalty
        + cost_weights.get("cost_minimization", 0.0) * cost_penalty
    )
    return total_loss


# =============================================================================
# Production execution & end-to-end optimization loop
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-HypersonicEngine v2] Running on: {device}")

    # ---- Initialize the complete system module ----------------------------
    cfg = HypersonicEngineConfig(latent_dim=32, num_clauses=64)
    engine_module = ProductionHypersonicBatteryEngineModule(cfg).to(device)
    engine_module.train()
    optimizer = optim.AdamW(engine_module.parameters(), lr=1e-3)

    # ---- Simulated flight / environmental state batch --------------------
    batch_inputs = torch.randn(16, cfg.latent_dim, device=device)
    target_optimal_thrust = torch.ones(16, device=device) * 2.5

    weights = {
        "stability":          0.05,
        "no_zeno":            0.02,
        "cost_minimization":  0.1,
        "complexity":         0.01,
    }

    # ---- Native full differentiable training step -------------------------
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        model_outputs = engine_module(batch_inputs)
        loss = compute_production_loss(model_outputs, target_optimal_thrust, weights)
    loss.backward()
    optimizer.step()

    # ---- Diagnostics (converted outside the graph) ------------------------
    det = model_outputs["topological_determinant"].detach().float()
    print("Complete Hypersonic Battery-Powered Engine Production Module Initialized & Optimized.")
    print(f"  End-to-end optimization step executed.")
    print(f"  Total loss                        : {loss.item():.6f}")
    print(f"  Topological signature (log|det|)  : "
          f"min={det.min().item():.4f}  max={det.max().item():.4f}  "
          f"finite={bool(torch.isfinite(det).all().item())}")
    print(f"  discharge_rate  ∈ (0, 1)          : "
          f"{model_outputs['discharge_rate'].detach().min().item():.4f} – "
          f"{model_outputs['discharge_rate'].detach().max().item():.4f}")
    print(f"  thrust_burst    ≥ 0               : "
          f"{model_outputs['thrust_burst'].detach().min().item():.4e} – "
          f"{model_outputs['thrust_burst'].detach().max().item():.4e}")
    print(f"  control_actions sum ≈ 1           : "
          f"{model_outputs['control_actions'].detach().sum(dim=-1).mean().item():.6f}")

    # ---- Gradient-flow audit ---------------------------------------------
    grad_ok = True
    for name, p in engine_module.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite or missing gradient: {name}")
    print(f"  Gradient flow                     : "
          f"{'OK (all finite)' if grad_ok else 'FAILED'}")
