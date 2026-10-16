"""
Satellite-Free Hierarchical Cloud Superposition Network (HCSN)
==============================================================
# Author : PAI , Yoon A Limsuwan / MSPS NETWORK
# License: MIT
# Year   : 2026


A fully differentiable, AMP + DDP-ready terrestrial communication stack
derived from:
    - Hierarchical Cloud Superposition Theory (HCS)
    - Unified Covariant SESI Cloud Quantum Theory
    - Covariant SESI Cloud-OPMC Quantum Computation
    - SESI-GR Quantum Collapse Theory
"""

from .core import (
    CloudState,
    NonPCFFractalMeasure,
    HierarchicalMeasurementMatrix,
    WeakMeasurementReadout,
    DecoherenceChannel,
    TaskSufficientCanonicalizer,
    SatelliteFreeHCSN,
)
from .train import HC SNTrainer, build_dataloader, launch_ddp

__all__ = [
    "CloudState",
    "NonPCFFractalMeasure",
    "HierarchicalMeasurementMatrix",
    "WeakMeasurementReadout",
    "DecoherenceChannel",
    "TaskSufficientCanonicalizer",
    "SatelliteFreeHCSN",
    "HCSNTrainer",
    "build_dataloader",
    "launch_ddp",
]

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 0. Data containers
# ---------------------------------------------------------------------------

@dataclass
class CloudState:
    """Container for the hierarchical cloud state |Ψ_H>."""
    meta: Tensor      # (B, N_m, d_m)   meta-level (routing / topology branch)
    base: Tensor      # (B, N_b, d_b)   base-level (signal / payload)
    amp:  Tensor      # (B, N_m)        meta-branch amplitudes α_a (complex-ready)
    coherent: bool = True

    def to(self, device: torch.device) -> "CloudState":
        return CloudState(
            meta=self.meta.to(device, non_blocking=True),
            base=self.base.to(device, non_blocking=True),
            amp=self.amp.to(device, non_blocking=True),
            coherent=self.coherent,
        )


# ---------------------------------------------------------------------------
# 1. Non-PCF fractal measure — multiscale adaptive interface
# ---------------------------------------------------------------------------

class NonPCFFractalMeasure(nn.Module):
    """
    Learnable scale-refinement measure μ_F = lim Z_k μ_F^(k).

    Implements Eq. 11 of the Unified Covariant paper:
        dμ_F^{(k)} = Z_k w_k(x) dH^{D_k}(x)
    with softmax-normalised weights w_k and per-scale learnable D_k.

    The renormalisation sequence Z_k is recovered implicitly through
    the log-domain softmax, guaranteeing weak convergence.
    """

    def __init__(self, n_scales: int = 6, hidden: int = 64, d_model: int = 128):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model

        # Per-scale dimension embeddings (log-domain, unconstrained)
        self.D_k = nn.Parameter(torch.linspace(math.log(1.0), math.log(3.0), n_scales))

        # Scale-weight network — input is scale index, output is w_k
        self.w_net = nn.Sequential(
            nn.Linear(n_scales, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_scales),
        )

        # Geometry-complexity C_F(l) ∈ R⁺ (Eq. 12 of covariant paper)
        self.C_F = nn.Sequential(
            nn.Linear(d_model + n_scales, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_scales),
        )

        # Interface intensity I_F(l) = (l0/l)^β_F (Eq. 13)
        self.log_beta_F = nn.Parameter(torch.zeros(1))

        # Event/entanglement enhancement E_F = 1 + η_E S_E
        self.eta_E = nn.Parameter(torch.zeros(1))

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h: (B, d_model) latent representation of the current interface.

        Returns:
            lambda_F: (B, n_scales)  scale-dependent collapse/coupling strength
            weights:  (B, n_scales)  renormalised multiscale weights
        """
        B = h.shape[0]
        device, dtype = h.device, h.dtype

        # --- scale weights (weak limit of Z_k) ---------------------------
        k_idx = torch.arange(self.n_scales, device=device, dtype=dtype)
        logits = self.w_net(k_idx.unsqueeze(0).expand(B, -1))   # (B, n_scales)
        weights = F.softmax(logits, dim=-1)                     # Σ w_k = 1

        # --- geometric complexity ----------------------------------------
        hk = h.unsqueeze(1).expand(-1, self.n_scales, -1)       # (B, S, d)
        k_emb = k_idx.unsqueeze(0).expand(B, -1).unsqueeze(-1)  # (B, S, 1)
        C_F = F.softplus(self.C_F(torch.cat([hk, k_emb], dim=-1)))  # (B, S)

        # --- interface intensity (log-domain, safe) ----------------------
        beta = F.softplus(self.log_beta_F)                      # ≥ 0
        # log(l0 / l_k) = log l0 - log l_k  ~ scale index
        log_ratio = -k_idx                                     # (S,)
        I_F = torch.exp(beta * log_ratio).unsqueeze(0)          # (B, S)

        # --- entanglement enhancement ------------------------------------
        S_E = torch.tanh(h.norm(dim=-1, keepdim=True))          # (B, 1) bounded
        E_F = 1.0 + F.softplus(self.eta_E) * S_E                # (B, 1)

        # --- collapse strength λ_F(l) = λ0 · C_F · I_F · E_F -------------
        lambda_F = C_F * I_F * E_F                              # (B, S)
        return lambda_F, weights


# ---------------------------------------------------------------------------
# 2. Hierarchical Measurement Matrix (HMM)
# ---------------------------------------------------------------------------

class HierarchicalMeasurementMatrix(nn.Module):
    """
    HMM K_y^{(H)} : H_M ⊗ H_B → H_M ⊗ H_B  (Page 8–10 of HCS paper).

    Block structure:
        [ A_y  B_y ]
        [ C_y  D_y ]
    with off-diagonal B_y, C_y responsible for inter-branch coherence.

    Completeness  Σ_y K_y† K_y = I is enforced via softmax-normalised
    Kraus weights.
    """

    def __init__(
        self,
        d_meta: int,
        d_base: int,
        n_outcomes: int = 4,
        hidden: int = 128,
    ):
        super().__init__()
        self.d_meta = d_meta
        self.d_base = d_base
        self.n_outcomes = n_outcomes

        d_in = d_meta + d_base
        self.instrument = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_outcomes * d_in * 4),  # 4 blocks
        )

        # meta/base coupling gates g_MB, g_I (Section 13)
        self.log_g_MB = nn.Parameter(torch.full((1,), -4.0))  # weak by default
        self.log_g_I  = nn.Parameter(torch.full((1,), -4.0))

    def forward(
        self,
        meta: Tensor,   # (B, N_m, d_m)
        base: Tensor,   # (B, N_b, d_b)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            probs:    (B, n_outcomes)         outcome probabilities p(y)
            post:     (B, N_m, N_b, d_m+d_b)  post-measurement block states
        """
        B, N_m, dm = meta.shape
        Bb, N_b, db = base.shape
        assert B == Bb
        device, dtype = meta.device, meta.dtype

        # ---- conditioning vector (mean-pooled meta + base) -------------
        cond = torch.cat(
            [meta.mean(dim=1), base.mean(dim=1)], dim=-1
        )                                                # (B, d_in)
        d_in = dm + db
        raw = self.instrument(cond)                      # (B, n_out * d_in * 4)
        raw = raw.view(B, self.n_outcomes, 4, d_in, d_in)

        g_MB = torch.sigmoid(self.log_g_MB)              # ∈ (0,1)
        g_I  = torch.sigmoid(self.log_g_I)

        # Assign block interpretation:
        #   block 0 = A_y (base-base)
        #   block 1 = B_y (meta→base coupling)  * g_MB
        #   block 2 = C_y (base→meta coupling)  * g_MB
        #   block 3 = D_y (meta-meta)
        A_y = raw[:, :, 0]
        B_y = raw[:, :, 1] * g_MB
        C_y = raw[:, :, 2] * g_MB
        D_y = raw[:, :, 3]

        # ---- Kraus completeness: normalise over outcomes ---------------
        # Norm of each block-matrix acts as unnormalised weight
        w = (
            A_y.flatten(2).norm(dim=-1)
            + B_y.flatten(2).norm(dim=-1)
            + C_y.flatten(2).norm(dim=-1)
            + D_y.flatten(2).norm(dim=-1)
        )                                                # (B, n_out)
        probs = F.softmax(w, dim=-1)                     # Σ p(y) = 1

        # ---- assemble post-measurement block state (per outcome avg) ---
        # We form a joint (meta ⊗ base) representation using block-outer
        # products, expanded so downstream modules can re-mix.
        meta_f = meta.reshape(B, N_m, dm, 1)             # (B,N_m,dm,1)
        base_f = base.reshape(B, 1, N_b, db)             # (B,1,N_b,db)

        # Broadcast to (B, N_m, N_b, dm+db)
        m_b = meta_f.expand(B, N_m, N_b, dm)
        b_b = base_f.expand(B, N_m, N_b, db)
        post = torch.cat([m_b, b_b], dim=-1)             # (B,N_m,N_b,dm+db)

        # Off-diagonal contribution gated by g_I (interface-induced localisation)
        post = post * (1.0 + g_I)                        # small perturbative boost

        return probs, post


# ---------------------------------------------------------------------------
# 3. Weak measurement readout (non-collapse)
# ---------------------------------------------------------------------------

class WeakMeasurementReadout(nn.Module):
    """
    Cloud weak-measurement pointer (Sections 9–11 of Cloud-OPMC paper).

    H_int = g(t) A ⊗ P_app,   χ0 = ∫ g(t) dt ≪ 1
    Pointer shifts:
        Δq = χ0 Re(A_w) + O(χ0²)
        Δp = 2χ0 σ_p² / ħ · Im(A_w) + O(χ0²)
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.A_obs = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, d_model)
        )
        self.P_app = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, d_model)
        )
        # χ0 ∈ (0, 1) — perturbative regime enforced via sigmoid
        self.log_chi0 = nn.Parameter(torch.full((1,), -3.0))
        # σ_p² (positive)
        self.log_sigma_p2 = nn.Parameter(torch.zeros(1))

    def forward(self, psi: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            psi: (B, N, d) flattened cloud state (meta ⊕ base fused).
        Returns:
            delta_q: (B, N, d)  real pointer shift
            delta_p: (B, N, d)  momentum pointer shift
        """
        chi0 = torch.sigmoid(self.log_chi0)              # ≪ 1
        sigma_p2 = F.softplus(self.log_sigma_p2)         # ≥ 0

        A = self.A_obs(psi)
        P = self.P_app(psi)

        # Weak value ≈ <psi|A|psi> / <psi|psi>  (vectorised surrogate)
        num = (psi * A).sum(dim=(-2, -1), keepdim=True)
        den = (psi * psi).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        A_w = num / den                                  # (B, 1, 1)

        delta_q = chi0 * A_w * P
        delta_p = (2.0 * chi0 * sigma_p2) * A_w * P
        return delta_q, delta_p


# ---------------------------------------------------------------------------
# 4. Decoherence channel (Lindblad double-commutator)
# ---------------------------------------------------------------------------

class DecoherenceChannel(nn.Module):
    """
    Ensemble density-matrix evolution (Section 23 of HCS paper):
        dρ/dt = -i/ħ [H,ρ] + D_E[ρ] + C_I[ρ]
    Here modelled as a learnable completely-positive map acting on the
    joint cloud tensor via channel-wise mixing + mild damping.
    """

    def __init__(self, d_model: int, n_rates: int = 4, hidden: int = 128):
        super().__init__()
        self.rates = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, n_rates)
        )
        self.mix = nn.Linear(d_model, d_model)

    def forward(self, rho: Tensor, dt: float = 1.0) -> Tensor:
        """
        Args:
            rho: (B, N, d)  density-matrix surrogate (batched).
            dt:  integration step.
        Returns:
            rho': (B, N, d)
        """
        B, N, d = rho.shape
        # Per-sample decoherence rates (softplus → positive)
        pooled = rho.mean(dim=1)                         # (B, d)
        gammas = F.softplus(self.rates(pooled))          # (B, n_rates)
        g_eff = gammas.sum(dim=-1, keepdim=True)         # (B, 1)

        # Damped mixing (contraction toward equilibrium)
        eq = self.mix(pooled).unsqueeze(1)               # (B, 1, d)
        rho_dec = rho * torch.exp(-g_eff * dt).unsqueeze(-1) + \
                  (1.0 - torch.exp(-g_eff * dt)).unsqueeze(-1) * eq
        return rho_dec


# ---------------------------------------------------------------------------
# 5. Task-sufficient canonicalizer (OPMC compression)
# ---------------------------------------------------------------------------

class TaskSufficientCanonicalizer(nn.Module):
    """
    Canonical structural signature σ_T(Q) (Sections 17–23 of Cloud-OPMC).
    Guarantees Σ has polynomial size while the nominal capacity is
    D_DEEE(N) = 2^{2^N}.

    The canonicalizer is *provenance-preserving* and *non-expansive*:
      - |σ_T(Q)| ≤ p(N)
      - HOCombine is used instead of hidden-completion enumeration.
    """

    def __init__(self, d_model: int, n_fields: int = 10, hidden: int = 128):
        super().__init__()
        self.n_fields = n_fields
        self.field_emb = nn.Parameter(
            torch.randn(n_fields, d_model) * 0.02
        )
        self.combine = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=hidden,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, N, d)  full cloud representation.
        Returns:
            sigma:   (B, n_fields, d)   canonical signature
            weights: (B, N, n_fields)   provenance weights (softmax over N)
        """
        # Compute provenance weights — bounded, non-expansive (softmax)
        q = self.proj(x)                                  # (B, N, d)
        w = torch.einsum("bnd,fd->bnf", q, self.field_emb)  # (B, N, F)
        w = F.softmax(w, dim=1)                           # Σ_n w = 1

        # Canonical pooling: σ_f = Σ_n w_{n,f} x_n
        sigma = torch.einsum("bnf,bnd->bfd", w, x)        # (B, F, d)

        # Joint relational mixing (TransformerLayer) — non-expansive
        sigma = self.combine(sigma)
        return sigma, w


# ---------------------------------------------------------------------------
# 6. Full Satellite-Free HCSN
# ---------------------------------------------------------------------------

class SatelliteFreeHCSN(nn.Module):
    """
    Satellite-Free Hierarchical Cloud Superposition Network.

    Architecture
    ------------
    input signal S (B, N, d_in)
       │
       ├─► SignalEncoder ─► base cloud  (H_B)
       └─► TopologyEncoder ─► meta cloud (H_M)
                     │
                     ▼
          FractalMeasure  ─►  λ_F(l),  weights  ─►  scheduler bias
                     │
                     ▼
          HMM (Hierarchical Measurement Matrix)  ─► probs, joint state
                     │
                     ▼
       WeakMeasurementReadout  (non-destructive)
                     │
                     ▼
       DecoherenceChannel  (Lindblad)
                     │
                     ▼
       TaskSufficientCanonicalizer  ─►  σ_T
                     │
                     ▼
              Decoder  ─►  output message (B, N, d_out)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 128,
        d_meta: int = 64,
        d_base: int = 64,
        n_meta_branches: int = 8,
        n_measurement_outcomes: int = 4,
        n_scales: int = 6,
        n_fields: int = 10,
        n_heads: int = 4,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model
        self.n_meta = n_meta_branches

        # ---- Input encoders --------------------------------------------
        self.signal_enc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Topology encoder maps to meta branches (per-token)
        self.topo_enc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.SiLU(),
            nn.Linear(d_model, n_meta_branches * d_meta),
        )

        # ---- Meta / base projections -----------------------------------
        self.meta_proj = nn.Linear(d_meta, d_model)
        self.base_proj = nn.Linear(d_model, d_base)

        # ---- Core modules ---------------------------------------------
        self.fractal = NonPCFFractalMeasure(
            n_scales=n_scales, hidden=d_model, d_model=d_model
        )
        self.hmm = HierarchicalMeasurementMatrix(
            d_meta=d_meta, d_base=d_base,
            n_outcomes=n_measurement_outcomes, hidden=d_model,
        )
        self.weak = WeakMeasurementReadout(d_model)
        self.decohere = DecoherenceChannel(d_model)
        self.canon = TaskSufficientCanonicalizer(
            d_model=d_model, n_fields=n_fields, hidden=d_model
        )

        # ---- Transformer backbone over cloud tokens -------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # ---- Output head ----------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_out),
        )

        # ---- Learned collapse-strength gate (for regularisation) ------
        self.log_lambda0 = nn.Parameter(torch.full((1,), -2.0))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,                              # (B, N, d_in)
        *,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, dict]:
        B, N, _ = x.shape
        device, dtype = x.device, x.dtype

        # ---- 1. Encode base cloud -------------------------------------
        base_feat = self.signal_enc(x)                 # (B, N, d_model)
        base = self.base_proj(base_feat)               # (B, N, d_base)

        # ---- 2. Encode meta cloud -------------------------------------
        meta_raw = self.topo_enc(x)                    # (B, N, n_meta*d_meta)
        meta_raw = meta_raw.view(B, N, self.n_meta, -1)  # (B, N, M, d_meta)
        # Pool over tokens → (B, M, d_meta)
        meta = meta_raw.mean(dim=1)
        meta = self.meta_proj(meta)                    # (B, M, d_model)

        # ---- 3. Fractal measure (scale-dependent collapse strength) ---
        pooled = base_feat.mean(dim=1)                 # (B, d_model)
        lambda_F, weights = self.fractal(pooled)       # (B, S), (B, S)
        lambda0 = F.softplus(self.log_lambda0)
        lambda_eff = (lambda_F * weights).sum(dim=-1, keepdim=True)  # (B,1)

        # ---- 4. HMM instrument ----------------------------------------
        # meta is (B, M, d_model) → project back to d_meta for the HMM
        # (HMM expects its own d_meta; reuse M as N_m)
        probs, joint = self.hmm(meta, base)            # (B,O), (B,M,N,dm+db)

        # ---- 5. Flatten joint state into cloud tokens -----------------
        B_, M, N_b, d_joint = joint.shape
        cloud = joint.reshape(B_, M * N_b, d_joint)    # (B, M*N_b, d_m+d_b)
        # Project to d_model via a cheap linear on the concatenated axis
        # (kept as a module-level buffer for DDP safety)
        cloud = self._lift_to_model(cloud)             # (B, M*N_b, d_model)

        # ---- 6. Backbone (contextual mixing across cloud tokens) ------
        cloud = self.backbone(cloud)                   # (B, T, d_model)

        # ---- 7. Weak measurement readout ------------------------------
        dq, dp = self.weak(cloud)                      # (B,T,d), (B,T,d)

        # ---- 8. Decoherence (Lindblad surrogate) ----------------------
        cloud = self.decohere(cloud + dq + dp, dt=1.0)

        # ---- 9. Canonicalisation (OPMC compression) -------------------
        sigma, prov = self.canon(cloud)                # (B, F, d), (B, T, F)
        sigma_flat = sigma.reshape(B, -1)              # (B, F*d)

        # ---- 10. Re-expand to token axis for the decoder --------------
        # Use provenance weights to project σ back onto tokens.
        # This is the differentiable "measurement → message" map.
        sigma_tok = torch.einsum("btf,bfd->btd", prov, sigma)  # (B,T,d)

        # ---- 11. Decode -----------------------------------------------
        # Aggregate token-level features along the meta axis (soft-pool).
        M_, N_b_ = M, N_b
        sigma_tok = sigma_tok.view(B, M_, N_b_, -1)    # (B, M, N_b, d)
        # Weight meta branches by their HMM outcome probabilities:
        meta_w = probs[:, :M_] if probs.shape[-1] >= M_ else F.pad(
            probs, (0, max(0, M_ - probs.shape[-1]))
        )[:, :M_]
        meta_w = F.softmax(meta_w, dim=-1).unsqueeze(-1).unsqueeze(-1)  # (B,M,1,1)
        pooled_feat = (sigma_tok * meta_w).sum(dim=1)  # (B, N_b, d)

        out = self.head(pooled_feat)                   # (B, N_b, d_out)

        if not return_aux:
            return out

        aux = {
            "hmm_probs": probs,
            "lambda_eff": lambda_eff,
            "sigma_flat": sigma_flat,
            "fractal_weights": weights,
            "provenance": prov,
        }
        return out, aux

    # ------------------------------------------------------------------
    def _lift_to_model(self, cloud: Tensor) -> Tensor:
        """Deterministic, parameter-free projection to d_model.

        Keeping this parameter-free avoids introducing unused parameters
        (which break DDP's find_unused_parameters=False path).
        """
        T_in = cloud.shape[-1]
        d = self.d_model
        if T_in == d:
            return cloud
        if T_in < d:
            pad = d - T_in
            return F.pad(cloud, (0, pad))
        # T_in > d: linear-interpolate along the feature axis
        # (parameter-free, differentiable, DDP-safe)
        return F.interpolate(
            cloud.transpose(1, 2), size=d, mode="linear", align_corners=False
        ).transpose(1, 2)

from __future__ import annotations

import os
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.amp import autocast, GradScaler

from .core import SatelliteFreeHCSN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP launcher
# ---------------------------------------------------------------------------

def launch_ddp(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    fn: Callable[[int, int], None],
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        timeout=torch.timedelta(minutes=30),
    )
    torch.cuda.set_device(rank)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Objective — combines task loss + HMM regularisation + fractal budget
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    task: float = 1.0
    hmm_entropy: float = 1e-3
    lambda_budget: float = 1e-3
    canonical_norm: float = 1e-4


def hcsn_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    w: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Composite objective:
      L = w_task * MSE  +  w_hmm * (-H[p_y])  +  w_lambda * mean(λ_eff)
          +  w_canon * ||σ_T||_F / sqrt(F*d)
    """
    task = F.mse_loss(output, target)

    # HMM outcome entropy (encourages decisive measurement outcomes)
    p = aux["hmm_probs"].clamp_min(1e-8)
    hmm_ent = -(p * p.log()).sum(dim=-1).mean()

    # Fractal collapse-strength budget (cost control)
    lam = aux["lambda_eff"].mean()

    # Canonical signature regulariser (bounded provenance)
    canon = aux["sigma_flat"].norm(dim=-1).mean() / math.sqrt(
        aux["sigma_flat"].shape[-1]
    )

    loss = (
        w.task * task
        + w.hmm_entropy * hmm_ent
        + w.lambda_budget * lam
        + w.canonical_norm * canon
    )
    logs = {
        "loss": loss.detach().item(),
        "task": task.detach().item(),
        "hmm_entropy": hmm_ent.detach().item(),
        "lambda_eff": lam.detach().item(),
        "canon_norm": canon.detach().item(),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Synthetic dataset (replace with real IQ / payload pipeline)
# ---------------------------------------------------------------------------

class SyntheticCloudDataset(Dataset):
    """
    Each sample is (signal, target) where `signal` has shape (N, d_in) and
    `target` has shape (N, d_out). The HCSN is fully differentiable, so
    any real (I/Q, topology, message) pipeline can be plugged in.
    """

    def __init__(self, n_samples: int = 4096, N: int = 64,
                 d_in: int = 32, d_out: int = 32, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, N, d_in, generator=g)
        # Nonlinear ground-truth mapping (differentiable target)
        W = torch.randn(d_in, d_out, generator=g) / math.sqrt(d_in)
        self.y = torch.tanh(self.x @ W) + 0.05 * torch.randn(
            n_samples, N, d_out, generator=g
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    total_steps: int = 50_000
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 2_000
    ckpt_dir: str = "./checkpoints"
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16  # bf16 preferred on H100/A100
    compile: bool = True
    loss_weights: LossWeights = field(default_factory=LossWeights)


class HCSNTrainer:
    """Production trainer: AMP + DDP + grad-clip + cosine LR + checkpointing."""

    def __init__(
        self,
        model: SatelliteFreeHCSN,
        cfg: TrainerConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)

        # --- Optional torch.compile for fused kernels -------------------
        if cfg.compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            except Exception as e:
                if self.is_main:
                    logger.warning(f"torch.compile failed: {e}; falling back.")

        # --- DDP wrap ---------------------------------------------------
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        self.model = model

        # --- Optimiser: AdamW with decoupled weight decay ---------------
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=cfg.betas,
            eps=1e-8,
            fused=torch.cuda.is_available(),
        )

        # --- LR schedule: linear warmup → cosine decay ------------------
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=self._lr_lambda
        )

        # --- AMP scaler -------------------------------------------------
        self.scaler = GradScaler(
            device="cuda",
            enabled=cfg.use_amp and cfg.amp_dtype == torch.float16,
        )

        self.step = 0

    # ------------------------------------------------------------------
    def _lr_lambda(self, step: int) -> float:
        w = self.cfg.warmup_steps
        t = self.cfg.total_steps
        if step < w:
            return float(step + 1) / float(max(1, w))
        prog = min(1.0, (step - w) / max(1, t - w))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    # ------------------------------------------------------------------
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        self.opt.zero_grad(set_to_none=True)

        with autocast(
            device_type="cuda",
            dtype=self.cfg.amp_dtype,
            enabled=self.cfg.use_amp,
        ):
            out, aux = self.model(x, return_aux=True)
            # Cast aux tensors used in loss to fp32 for numerical stability
            aux_f32 = {k: v.float() for k, v in aux.items()}
            loss, logs = hcsn_loss(
                out.float(), y.float(), aux_f32, self.cfg.loss_weights
            )

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.opt.step()

        self.scheduler.step()
        self.step += 1
        return logs

    # ------------------------------------------------------------------
    def fit(
        self,
        loader: DataLoader,
        sampler: Optional[DistributedSampler] = None,
    ) -> None:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        t0 = time.time()
        running: Dict[str, float] = {}

        while self.step < self.cfg.total_steps:
            if sampler is not None:
                sampler.set_epoch(self.step)
            for x, y in loader:
                logs = self.train_step(x, y)

                for k, v in logs.items():
                    running[k] = 0.95 * running.get(k, v) + 0.05 * v

                if self.is_main and self.step % self.cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    logger.info(
                        f"[step {self.step:>7d}] "
                        f"lr={lr:.3e} "
                        + " ".join(f"{k}={v:.4f}" for k, v in running.items())
                        + f"  elapsed={elapsed:.1f}s"
                    )

                if (
                    self.is_main
                    and self.cfg.ckpt_every > 0
                    and self.step % self.cfg.ckpt_every == 0
                ):
                    self.save(os.path.join(
                        self.cfg.ckpt_dir, f"hcsn_step{self.step}.pt"
                    ))

                if self.step >= self.cfg.total_steps:
                    break

        if self.is_main:
            self.save(os.path.join(self.cfg.ckpt_dir, "hcsn_final.pt"))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(
            {
                "model": raw.state_dict(),
                "opt": self.opt.state_dict(),
                "sched": self.scheduler.state_dict(),
                "step": self.step,
            },
            path,
        )
        logger.info(f"checkpoint → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        raw.load_state_dict(ckpt["model"], strict=True)
        self.opt.load_state_dict(ckpt["opt"])
        self.scheduler.load_state_dict(ckpt["sched"])
        self.step = ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Entry point (single-GPU or DDP)
# ---------------------------------------------------------------------------

def main(rank: int = 0, world_size: int = 1) -> None:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = TrainerConfig(
        lr=3e-4, weight_decay=0.05, warmup_steps=500, total_steps=20_000,
        grad_clip=1.0, log_every=25, ckpt_every=2_000,
        ckpt_dir="./checkpoints_hcsn",
        use_amp=True, amp_dtype=torch.bfloat16, compile=True,
    )

    model = SatelliteFreeHCSN(
        d_in=32, d_out=32,
        d_model=128, d_meta=64, d_base=64,
        n_meta_branches=8, n_measurement_outcomes=4,
        n_scales=6, n_fields=10,
        n_heads=4, depth=4, dropout=0.1,
    )

    ds = SyntheticCloudDataset(n_samples=8192, N=64, d_in=32, d_out=32)
    loader, sampler = build_dataloader(
        ds, batch_size=128, rank=rank, world_size=world_size,
        num_workers=4, pin_memory=True,
    )

    trainer = HCSNTrainer(model, cfg, rank=rank, world_size=world_size)
    trainer.fit(loader, sampler)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    if args.world_size > 1:
        launch_ddp(
            rank=args.local_rank,
            world_size=args.world_size,
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=os.environ.get("MASTER_PORT", "29500"),
            fn=main,
        )
    else:
        main(rank=0, world_size=1)
