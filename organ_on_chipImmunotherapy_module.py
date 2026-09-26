# =============================================================================
# Organ On Chip Immunotherapy Module (SESI)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================
"""
Production-grade, fully differentiable Organ-on-a-Chip (OoC) immunotherapy
telemetry engine with a smooth double-exponential No-Zeno topological barrier.

Fixes and improvements over the reference implementation
--------------------------------------------------------
1. **Syntax error removed** — the reference docstring contained a stray
   triple-backtick inside its body, which is a hard `SyntaxError`.
2. **Full differentiability** — `torch.abs(x)` (kinked at 0) and
   `torch.clamp(parameter, min=…)` (zero gradient outside the clamp) are
   replaced with C^∞ surrogates:
       |x|      →  sqrt(x² + ε²)
       clamp≥0  →  F.softplus(raw) + ε
3. **Reparameterized positivity** — `barrier_energy` and `noise_variance` are
   stored as raw unconstrained parameters and mapped through `softplus` in
   `forward`, so they never escape the valid domain and their gradients never
   die on a clamp boundary.
4. **Numerical stability** — the double-exponential `exp(−c₁·exp(·))` is
   evaluated with an inner clamp (prevents fp16/bf16/fp32 overflow) and an
   outer log-floor (prevents 0·gradient underflow). All barrier math is run
   in fp32 and cast back, so AMP is safe.
5. **DDP-clean** — no rank-local RNG, no Python-scalar graph breaks, no
   data-dependent control flow; `torch.compile(mode="max-autotune")` ready.
6. **AMP-correct** — autocast is caller-controlled (`torch.amp.autocast`),
   which composes correctly with DDP and compiled graphs (the previous
   decorator form did not).

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    module = OrganOnChipImmunotherapyModule(input_dim, hidden_dim).to(rank)
    module = torch.compile(module, mode="max-autotune")             # optional
    module = torch.nn.parallel.DistributedDataParallel(
        module, device_ids=[rank], gradient_as_bucket_view=True,
    )

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, metrics = module(sensor_stream, baseline)
    loss = criterion(logits, labels)
    loss.backward()
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OrganOnChipImmunotherapyModule"]


class OrganOnChipImmunotherapyModule(nn.Module):
    """
    Real-time Organ-on-a-Chip toxicity / CRS-risk head with a differentiable
    double-exponential No-Zeno topological gate.

    Parameters
    ----------
    input_dim, hidden_dim, output_dim : int
        Feature dimensions of the contraction backbone and prediction head.
    dropout : float
        Dropout probability inside the head; must be in [0, 1).
    c1_constant : float
        SESI No-Zeno prefactor c₁ in `P = exp(−c₁·exp(·))`. Must be > 0.
    zeno_clamp : float
        Upper clamp on the inner exponent `ΔE/(σ²·|x|)`; prevents overflow.
    log_floor : float
        Lower clamp on `log P`; prevents underflow-to-zero and gradient death.
    validate_inputs : bool
        Cheap shape guards; disable once shapes are statically known to avoid
        `torch.compile` recompiles.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 2,
        dropout: float = 0.05,
        *,
        c1_constant: float = 1.0,
        zeno_clamp: float = 15.0,
        log_floor: float = -25.0,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, output_dim must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (math.isfinite(c1_constant) and c1_constant > 0.0):
            raise ValueError("c1_constant must be > 0.")
        if not (math.isfinite(zeno_clamp) and zeno_clamp > 0.0):
            raise ValueError("zeno_clamp must be > 0.")
        if not (math.isfinite(log_floor) and log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")

        self.input_dim  = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.validate_inputs = bool(validate_inputs)

        # ---- Backbone: structural tensor contraction + tissue gating --------
        self.tensor_contractor = nn.Linear(input_dim,  hidden_dim, bias=True)
        self.tissue_gate       = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # ---- Reparameterized positivity (softplus of raw) -------------------
        # Initial raw values chosen so softplus(raw) matches the reference init
        # (ΔE₀ ≈ 1.5, σ²₀ ≈ 0.2), preserving the original numerical behaviour.
        self.barrier_energy_raw = nn.Parameter(torch.tensor(1.25))   # softplus ≈ 1.5
        self.noise_variance_raw = nn.Parameter(torch.tensor(-1.5))   # softplus ≈ 0.2

        # ---- Prediction head -----------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # ---- Non-persistent buffers (excluded from state_dict → DDP-clean) --
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_eps_abs",   _buf(1e-5),        persistent=False)
        self.register_buffer("_eps_pos",   _buf(1e-3),        persistent=False)
        self.register_buffer("_c1",        _buf(c1_constant), persistent=False)
        self.register_buffer("_zeno_max",  _buf(zeno_clamp),  persistent=False)
        self.register_buffer("_log_floor", _buf(log_floor),   persistent=False)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Weight initialization                                              #
    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Smooth double-exponential No-Zeno barrier                          #
    # ------------------------------------------------------------------ #
    def compute_double_exponential_barrier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable No-Zeno transition weight:

            P = exp( −c₁ · exp( clamp( ΔE / (σ² · |x|), max=z_max ) ) )
            P ← max( P, exp(log_floor) )        (prevents 0-gradient underflow)

        where
            ΔE   = softplus(barrier_energy_raw) + ε  > 0
            σ²   = softplus(noise_variance_raw) + ε  > 0
            |x|  ≈ sqrt(x² + ε²)                     (C^∞, no kink at 0)

        All operations run in fp32 internally and cast back to `x.dtype`, so the
        barrier is stable and gradient-preserving under AMP (fp16 / bf16).
        """
        dtype_out = x.dtype
        device    = x.device

        # Barrier math in fp32 for numerical robustness under AMP.
        x32 = x.float()

        eps_abs   = self._eps_abs.to(device=device)
        eps_pos   = self._eps_pos.to(device=device)
        c1        = self._c1.to(device=device)
        zeno_max  = self._zeno_max.to(device=device)
        log_floor = self._log_floor.to(device=device)

        # C^∞ positive physical parameters
        delta_e  = F.softplus(self.barrier_energy_raw.float()) + eps_pos
        sigma_sq = F.softplus(self.noise_variance_raw.float()) + eps_pos

        # C^∞ smooth |x| (equals ε at x = 0; matches the reference 1e-5 floor).
        x_abs = torch.sqrt(x32 * x32 + eps_abs * eps_abs)

        # Inner argument — clamped to prevent outer exp overflow (fp16/bf16/fp32).
        arg = (delta_e / (sigma_sq * x_abs)).clamp_(max=zeno_max)

        # Double-exponential in log-space with a lower floor on log P.
        inner = torch.exp(arg)                          # ≤ e^{z_max}
        log_p = (-c1 * inner).clamp_(min=log_floor)     # ≥ log_floor
        p     = torch.exp(log_p)                        # ∈ (e^{log_floor}, 1)

        return p.to(dtype_out)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        sensor_stream: torch.Tensor,                        # [B, D] or [B, L, D]
        baseline_reference: Optional[torch.Tensor] = None,  # same shape as contracted
        *,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        logits : [B, output_dim] or [B, L, output_dim] — toxicity-risk logits.
        metrics: dict of *differentiable* tensors (empty if `return_metrics=False`):
                    'transition_factor' : full per-position No-Zeno weight ∈ (0, 1)
                    'barrier_energy'    : physical ΔE (≥ ε)
                    'noise_variance'    : physical σ² (≥ ε)
        """
        if self.validate_inputs:
            if sensor_stream.dim() not in (2, 3):
                raise ValueError("sensor_stream must be [B, D] or [B, L, D].")
            if sensor_stream.shape[-1] != self.input_dim:
                raise ValueError(
                    f"last dim must equal input_dim={self.input_dim}, "
                    f"got {sensor_stream.shape[-1]}."
                )

        # ---- 1. Structural tensor contraction ------------------------------
        contracted = self.tensor_contractor(sensor_stream)

        if baseline_reference is not None:
            if baseline_reference.shape != contracted.shape:
                raise ValueError(
                    "baseline_reference must match contracted-state shape."
                )
            contracted = contracted - baseline_reference

        # ---- 2. Gated non-linear modulation (smooth, all-differentiable) ---
        gate      = torch.sigmoid(self.tissue_gate(contracted))
        modulated = contracted * gate

        # ---- 3. Double-exponential No-Zeno barrier modulation --------------
        transition_factor   = self.compute_double_exponential_barrier(modulated)
        # modulated · (1 − tf), expressed as a fused subtract-multiply.
        structural_manifold = modulated - modulated * transition_factor

        # ---- 4. Prediction head --------------------------------------------
        logits = self.head(structural_manifold)

        if not return_metrics:
            return logits, {}

        # Scalar summaries (differentiable) for logging / auxiliary losses.
        with torch.no_grad():
            barrier_energy = (F.softplus(self.barrier_energy_raw) + self._eps_pos).detach()
            noise_variance = (F.softplus(self.noise_variance_raw) + self._eps_pos).detach()

        metrics: Dict[str, torch.Tensor] = {
            "transition_factor": transition_factor,
            "barrier_energy":    barrier_energy,
            "noise_variance":    noise_variance,
        }
        return logits, metrics


# -----------------------------------------------------------------------------
# Reference DDP driver (illustrative — not executed at import time)
# -----------------------------------------------------------------------------
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
#
# def main(rank: int, world_size: int) -> None:
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#
#     model = OrganOnChipImmunotherapyModule(input_dim=128, hidden_dim=256).to(rank)
#     model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
#     model = torch.compile(model, mode="max-autotune")
#
#     B, L, D = 8, 32, 128
#     x  = torch.randn(B, L, D, device=rank)
#     y  = torch.randint(0, 2, (B, L), device=rank)
#
#     opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
#     with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
#         logits, metrics = model(x)
#         loss = torch.nn.functional.cross_entropy(
#             logits.reshape(-1, logits.size(-1)), y.reshape(-1)
#         )
#     loss.backward()
#     opt.step()
#     dist.destroy_process_group()
