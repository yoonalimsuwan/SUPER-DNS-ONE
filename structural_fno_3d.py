# =============================================================================
# STRUCTURAL FOURIER NEURAL OPERATOR (SFNO 3D)  —  v2.0
# AI-Physics Surrogate Model for SUPER DNS ONE Cluster
# =============================================================================
# Developer    : Yoon A Limsuwan
# Organization : MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Assisted by  : Claude (Anthropic), Gemini (Google), GPT (OpenAI)
# License      : MIT
# Year         : 2026
#
# Description:
#   A novel Fourier Neural Operator rooted in the Structural Calculus framework.
#   Designed specifically to receive and learn from output data produced by the
#   SUPER DNS ONE cluster (super_dns_one_v6.py, structuralfluctuatinghydro_v6.py,
#   structural_cahn_hilliard_3d.py, structural_langevin_v3.py).
#
#   Core mapping operator:
#       G : (u(x,0), sigma(x), t_target) ↦ u(x, t_target)
#
#   New in v2.0
#   ─────────────────────────────────────────────────────────────────────────
#   [1]  MultiScaleSpectralConv3d  — multi-resolution Fourier (coarse+fine)
#   [2]  StructuralFiLM            — sigma-conditioned Feature-wise Linear
#                                    Modulation (CSOC-aware gating)
#   [3]  StructuralCrossAttention  — cross-attention between u latent & sigma
#   [4]  MCDropoutHead             — Monte-Carlo Dropout uncertainty output
#   [5]  PhysicsLoss               — mass conservation + energy monotonicity
#                                    + Structural Itô regularisation
#   [6]  StructuralFNOTrainer      — full training loop with cosine-warm
#                                    restart LR, AMP, gradient clipping,
#                                    TensorBoard logging, checkpoint I/O
#   [7]  SuperDNSDataset           — data pipeline for SUPER DNS ONE .pt
#                                    snapshot files (u, sigma, metadata)
#   [8]  BoundaryPaddingConv3d     — physics-correct periodic/Neumann BC pads
# =============================================================================

from __future__ import annotations

import math
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# =============================================================================
# 0.  Constants & versioning
# =============================================================================

SFNO_VERSION = "2.0.0"
ONE_VERSION_REQ = "3.1"

# =============================================================================
# 1.  Low-level utilities
# =============================================================================

def _soft_clamp(x: Tensor, lo: float = -30.0, hi: float = 30.0) -> Tensor:
    """Differentiable clamp via tanh — consistent with ONE Ecosystem convention."""
    mid  = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh((x - mid) / (half + 1e-8))


class BoundaryPaddingConv3d(nn.Module):
    """
    3-D conv with explicit physics-aware padding.

    mode='circular'  → periodic BC (DNS / FH / FNO default)
    mode='replicate' → zero-gradient Neumann BC (thin-film / wall BC)
    mode='zeros'     → standard zero-padding
    """
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 1,
        pad_mode:     str = "circular",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.pad      = kernel_size // 2
        self.conv     = nn.Conv3d(in_channels, out_channels, kernel_size,
                                  padding=0, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.pad > 0:
            x = F.pad(x, [self.pad] * 6, mode=self.pad_mode)
        return self.conv(x)


# =============================================================================
# 2.  MultiScaleSpectralConv3d
# =============================================================================

class MultiScaleSpectralConv3d(nn.Module):
    """
    Multi-resolution 3-D Spectral Convolution.

    Instead of one set of Fourier weights at a single truncation level,
    this layer maintains *two* truncation levels:
        • modes_coarse  — low-freq bulk structure   (large-scale patterns)
        • modes_fine    — high-freq details          (interface sharpness, CSOC events)

    Both branches are summed after their respective compl_mul3d operations,
    allowing the network to jointly resolve slow bulk evolution and fast
    interface dynamics — which is exactly what Structural Cahn-Hilliard +
    Navier-Stokes produce in the DNS cluster.

    Args:
        in_channels   : number of input feature channels
        out_channels  : number of output feature channels
        modes_coarse  : Fourier truncation for low-frequency branch
        modes_fine    : Fourier truncation for high-frequency branch
                        (must satisfy modes_fine <= modes_coarse)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        modes_coarse: int = 12,
        modes_fine:   int = 6,
    ):
        super().__init__()
        assert modes_fine <= modes_coarse, \
            f"modes_fine ({modes_fine}) must be <= modes_coarse ({modes_coarse})"

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.mc           = modes_coarse
        self.mf           = modes_fine

        def _w(m: int) -> Tensor:
            scale = 1.0 / (in_channels * out_channels)
            return scale * torch.rand(
                in_channels, out_channels, m, m, m, dtype=torch.cfloat)

        # Coarse-scale weights (8 octants of the 3-D FFT)
        self.wc1 = nn.Parameter(_w(self.mc))
        self.wc2 = nn.Parameter(_w(self.mc))
        self.wc3 = nn.Parameter(_w(self.mc))
        self.wc4 = nn.Parameter(_w(self.mc))
        self.wc5 = nn.Parameter(_w(self.mc))
        self.wc6 = nn.Parameter(_w(self.mc))
        self.wc7 = nn.Parameter(_w(self.mc))
        self.wc8 = nn.Parameter(_w(self.mc))

        # Fine-scale weights (centre 8 octants, smaller truncation)
        self.wf1 = nn.Parameter(_w(self.mf))
        self.wf2 = nn.Parameter(_w(self.mf))
        self.wf3 = nn.Parameter(_w(self.mf))
        self.wf4 = nn.Parameter(_w(self.mf))
        self.wf5 = nn.Parameter(_w(self.mf))
        self.wf6 = nn.Parameter(_w(self.mf))
        self.wf7 = nn.Parameter(_w(self.mf))
        self.wf8 = nn.Parameter(_w(self.mf))

        # Learnable blend between coarse and fine (per output channel)
        self.blend = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))

    @staticmethod
    def _cmul(inp: Tensor, w: Tensor) -> Tensor:
        """Complex multiply: (B, Ci, x, y, z) x (Ci, Co, x, y, z) → (B, Co, x, y, z)."""
        return torch.einsum("bixyz,ioxyz->boxyz", inp, w)

    def forward(self, x: Tensor) -> Tensor:
        B      = x.shape[0]
        Nx, Ny, Nz = x.shape[-3], x.shape[-2], x.shape[-1]
        mc, mf = self.mc, self.mf

        x_ft   = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        Nzh    = x_ft.shape[-1]   # == Nz//2 + 1

        out_c  = torch.zeros(B, self.out_channels, Nx, Ny, Nzh,
                             dtype=torch.cfloat, device=x.device)
        out_f  = torch.zeros_like(out_c)

        # ── Coarse branch: 8 octants at mc ───────────────────────────────
        out_c[:, :, :mc,  :mc,  :mc]  = self._cmul(x_ft[:, :, :mc,  :mc,  :mc],  self.wc1)
        out_c[:, :, -mc:, :mc,  :mc]  = self._cmul(x_ft[:, :, -mc:, :mc,  :mc],  self.wc2)
        out_c[:, :, :mc,  -mc:, :mc]  = self._cmul(x_ft[:, :, :mc,  -mc:, :mc],  self.wc3)
        out_c[:, :, -mc:, -mc:, :mc]  = self._cmul(x_ft[:, :, -mc:, -mc:, :mc],  self.wc4)
        out_c[:, :, :mc,  :mc,  -mc:] = self._cmul(x_ft[:, :, :mc,  :mc,  -mc:], self.wc5)
        out_c[:, :, -mc:, :mc,  -mc:] = self._cmul(x_ft[:, :, -mc:, :mc,  -mc:], self.wc6)
        out_c[:, :, :mc,  -mc:, -mc:] = self._cmul(x_ft[:, :, :mc,  -mc:, -mc:], self.wc7)
        out_c[:, :, -mc:, -mc:, -mc:] = self._cmul(x_ft[:, :, -mc:, -mc:, -mc:], self.wc8)

        # ── Fine branch: 8 octants at mf ─────────────────────────────────
        out_f[:, :, :mf,  :mf,  :mf]  = self._cmul(x_ft[:, :, :mf,  :mf,  :mf],  self.wf1)
        out_f[:, :, -mf:, :mf,  :mf]  = self._cmul(x_ft[:, :, -mf:, :mf,  :mf],  self.wf2)
        out_f[:, :, :mf,  -mf:, :mf]  = self._cmul(x_ft[:, :, :mf,  -mf:, :mf],  self.wf3)
        out_f[:, :, -mf:, -mf:, :mf]  = self._cmul(x_ft[:, :, -mf:, -mf:, :mf],  self.wf4)
        out_f[:, :, :mf,  :mf,  -mf:] = self._cmul(x_ft[:, :, :mf,  :mf,  -mf:], self.wf5)
        out_f[:, :, -mf:, :mf,  -mf:] = self._cmul(x_ft[:, :, -mf:, :mf,  -mf:], self.wf6)
        out_f[:, :, :mf,  -mf:, -mf:] = self._cmul(x_ft[:, :, :mf,  -mf:, -mf:], self.wf7)
        out_f[:, :, -mf:, -mf:, -mf:] = self._cmul(x_ft[:, :, -mf:, -mf:, -mf:], self.wf8)

        # ── Blend & inverse FFT ───────────────────────────────────────────
        alpha    = torch.sigmoid(self.blend)           # ∈ (0, 1), learnable
        out_ft   = alpha * out_c + (1.0 - alpha) * out_f
        return torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), norm="ortho")


# =============================================================================
# 3.  StructuralFiLM  —  Feature-wise Linear Modulation conditioned on sigma
# =============================================================================

class StructuralFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on the structural
    regime field σ(x).

    Given latent feature map h ∈ ℝ^(B, C, Nx, Ny, Nz) and
    regime field σ ∈ ℝ^(B, 1, Nx, Ny, Nz):

        γ, β = MLP(σ)
        h_out = γ · h + β

    This is physically motivated: σ encodes the local criticality level
    (CSOC), so FiLM allows the network to apply *different* linear
    transformations in ordered vs disordered phases — exactly mimicking
    how the Structural Laplacian σ(x)·∇²u switches regime.

    Architecture:
        σ → Conv3d(1, hidden) → GELU → Conv3d(hidden, 2·C) → split(γ, β)
    """

    def __init__(self, width: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            BoundaryPaddingConv3d(1, hidden, kernel_size=3, pad_mode="circular"),
            nn.GELU(),
            BoundaryPaddingConv3d(hidden, hidden, kernel_size=3, pad_mode="circular"),
            nn.GELU(),
            nn.Conv3d(hidden, 2 * width, 1),   # → γ and β, both ∈ ℝ^C
        )
        # Initialise so that FiLM starts as identity (γ=1, β=0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: Tensor, sigma: Tensor) -> Tensor:
        out     = self.net(sigma)                          # (B, 2C, Nx, Ny, Nz)
        gamma, beta = out.chunk(2, dim=1)                  # each (B, C, Nx, Ny, Nz)
        return (1.0 + gamma) * h + beta                    # residual-style affine


# =============================================================================
# 4.  StructuralCrossAttention  —  u latent × sigma cross-attention
# =============================================================================

class StructuralCrossAttention(nn.Module):
    """
    Lightweight cross-attention between the latent field representation h
    and the structural regime field σ.

    Motivation: the FNO's spectral operations are global and translation-
    equivariant, but σ encodes *local* interface information.  A cross-
    attention mechanism allows each latent spatial location to attend
    selectively to regions where σ indicates phase boundaries — analogous
    to how Structural Itô correction concentrates near jump events.

    Implementation uses a spatially-pooled key/value from σ to keep the
    complexity O(C²) rather than O(N⁶).

    Args:
        width   : feature channel width (C)
        n_heads : number of attention heads
        pool_k  : spatial pooling kernel for σ compression
    """

    def __init__(self, width: int, n_heads: int = 4, pool_k: int = 4):
        super().__init__()
        assert width % n_heads == 0, "width must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = width // n_heads
        self.scale    = self.head_dim ** -0.5
        self.pool_k   = pool_k

        # Query from h; Key, Value from σ
        self.q_proj   = nn.Conv3d(width, width, 1)
        self.k_proj   = nn.Conv3d(1,     width, 1)
        self.v_proj   = nn.Conv3d(1,     width, 1)
        self.out_proj = nn.Conv3d(width, width, 1)
        self.norm     = nn.GroupNorm(n_heads, width)

    def forward(self, h: Tensor, sigma: Tensor) -> Tensor:
        B, C, Nx, Ny, Nz = h.shape
        H = self.n_heads
        D = self.head_dim

        # Spatially compress σ to reduce memory
        sigma_pool = F.avg_pool3d(sigma, self.pool_k,
                                  stride=self.pool_k,
                                  padding=self.pool_k // 4)
        # Query: (B, C, Nx, Ny, Nz) → flatten spatial → (B, H, Nx·Ny·Nz, D)
        q = self.q_proj(h).reshape(B, H, D, -1).permute(0, 1, 3, 2)  # (B,H,N,D)

        # Key / Value from pooled sigma
        Bp, _, Xp, Yp, Zp = sigma_pool.shape
        k = self.k_proj(sigma_pool).reshape(B, H, D, -1).permute(0, 1, 3, 2)
        v = self.v_proj(sigma_pool).reshape(B, H, D, -1).permute(0, 1, 3, 2)

        # Scaled dot-product attention
        attn   = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        ctx    = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, Nx, Ny, Nz)

        return self.norm(h + self.out_proj(ctx))


# =============================================================================
# 5.  StructuralFNOLayer  v2  —  upgraded with FiLM + cross-attention
# =============================================================================

class StructuralFNOLayer(nn.Module):
    """
    One block of the Structural FNO pipeline (v2).

    Processing order:
        1.  MultiScaleSpectralConv3d   — global multi-resolution spectral op
        2.  BoundaryPaddingConv3d      — local linear op (periodic-BC aware)
        3.  StructuralFiLM             — sigma-conditioned affine modulation
        4.  StructuralCrossAttention   — u × sigma cross-attention
        5.  GELU activation + residual skip connection

    Args:
        width         : feature channel width
        modes_coarse  : coarse-level Fourier truncation
        modes_fine    : fine-level Fourier truncation
        n_heads       : cross-attention heads
        dropout_p     : MC-Dropout probability (applied to spectral output)
        pad_mode      : padding mode for local conv ('circular' = periodic BC)
    """

    def __init__(
        self,
        width:        int,
        modes_coarse: int = 12,
        modes_fine:   int = 6,
        n_heads:      int = 4,
        dropout_p:    float = 0.1,
        pad_mode:     str  = "circular",
    ):
        super().__init__()

        self.spectral = MultiScaleSpectralConv3d(width, width, modes_coarse, modes_fine)
        self.local    = BoundaryPaddingConv3d(width, width, kernel_size=3, pad_mode=pad_mode)
        self.film     = StructuralFiLM(width, hidden=max(32, width // 2))
        self.attn     = StructuralCrossAttention(width, n_heads=n_heads)
        self.dropout  = nn.Dropout3d(p=dropout_p)
        self.norm     = nn.GroupNorm(min(8, width), width)

        # Learnable residual weight (starts near 1.0)
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        identity = x

        # Global + local paths
        x_spec = self.dropout(self.spectral(x))
        x_loc  = self.local(x)

        # FiLM: sigma modulates the combined spectral+local features
        x_mod  = self.film(x_spec + x_loc, sigma)

        # Cross-attention: refine at phase boundaries
        x_attn = self.attn(x_mod, sigma)

        # Residual + norm + activation
        return F.gelu(self.norm(x_attn + self.res_scale * identity))


# =============================================================================
# 6.  MCDropoutHead  —  uncertainty-aware output projection
# =============================================================================

class MCDropoutHead(nn.Module):
    """
    Output projection with Monte-Carlo Dropout for uncertainty estimation.

    At training time, dropout is always active (standard behaviour).
    At inference time, call model.train() before sampling n_samples times
    to obtain a predictive distribution, then call model.eval() to restore.

    Returns:
        mean  : (B, 1, Nx, Ny, Nz)  — predictive mean
        log_var : (B, 1, Nx, Ny, Nz) — log predictive variance (for NLL loss)
    """

    def __init__(self, width: int, dropout_p: float = 0.1):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Conv3d(width, 128, 1),
            nn.GELU(),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(128, 64, 1),
            nn.GELU(),
            nn.Conv3d(64, 1, 1),
        )
        self.var_head = nn.Sequential(
            nn.Conv3d(width, 64, 1),
            nn.GELU(),
            nn.Conv3d(64, 1, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean    = self.mean_head(x)
        log_var = _soft_clamp(self.var_head(x), lo=-10.0, hi=5.0)
        return mean, log_var


# =============================================================================
# 7.  StructuralFNO3D  —  full model
# =============================================================================

class StructuralFNO3D(nn.Module):
    """
    Structural Fourier Neural Operator 3D  (v2.0)

    Maps: (u(x,0), σ(x), t_target) ↦ (u(x, t_target), uncertainty)

    Input lifting:
        Channels = u_0 (1) + grid_xyz (3) + t_target (1) = 5

    Pipeline:
        Lift (5→width) → N × StructuralFNOLayer → MCDropoutHead

    Args:
        modes_coarse : coarse Fourier truncation (default 12)
        modes_fine   : fine   Fourier truncation (default 6)
        width        : latent channel width (default 64)
        num_layers   : number of SFNO blocks (default 6)
        n_heads      : cross-attention heads per layer (default 4)
        dropout_p    : MC-Dropout probability (default 0.1)
        pad_mode     : boundary condition mode ('circular' for periodic DNS)

    Example::

        model  = StructuralFNO3D()
        mean, log_var = model(u_0, sigma, t_target=0.5)
        std    = (0.5 * log_var).exp()           # point-wise std dev
    """

    def __init__(
        self,
        modes_coarse: int   = 12,
        modes_fine:   int   = 6,
        width:        int   = 64,
        num_layers:   int   = 6,
        n_heads:      int   = 4,
        dropout_p:    float = 0.1,
        pad_mode:     str   = "circular",
    ):
        super().__init__()
        self.width = width

        # Input lifting: (u_0, x, y, z, t) → width
        self.lift = nn.Sequential(
            nn.Conv3d(5, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, width, 1),
        )

        # Core SFNO layers
        self.layers = nn.ModuleList([
            StructuralFNOLayer(
                width        = width,
                modes_coarse = modes_coarse,
                modes_fine   = modes_fine,
                n_heads      = n_heads,
                dropout_p    = dropout_p,
                pad_mode     = pad_mode,
            )
            for _ in range(num_layers)
        ])

        # Uncertainty-aware output head
        self.head = MCDropoutHead(width, dropout_p=dropout_p)

    # ─── grid helper ──────────────────────────────────────────────────────

    def _make_grid(self, shape: Tuple, device: torch.device) -> Tensor:
        """Normalized coordinate grid [-1, 1]^3 broadcast over batch."""
        B, _, Nx, Ny, Nz = shape
        xs = torch.linspace(-1.0, 1.0, Nx, device=device)
        ys = torch.linspace(-1.0, 1.0, Ny, device=device)
        zs = torch.linspace(-1.0, 1.0, Nz, device=device)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid    = torch.stack([X, Y, Z], 0).unsqueeze(0).expand(B, -1, -1, -1, -1)
        return grid    # (B, 3, Nx, Ny, Nz)

    # ─── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        u_initial:  Tensor,
        sigma:      Tensor,
        t_target:   float | Tensor = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            u_initial : (B, 1, Nx, Ny, Nz)  initial order parameter / velocity
            sigma     : (B, 1, Nx, Ny, Nz)  structural regime field
            t_target  : scalar float or (B,) tensor — normalised target time ∈ [0,1]

        Returns:
            mean      : (B, 1, Nx, Ny, Nz)  predicted field at t_target
            log_var   : (B, 1, Nx, Ny, Nz)  log predictive variance
        """
        B, _, Nx, Ny, Nz = u_initial.shape
        device            = u_initial.device

        # Build t-channel
        if isinstance(t_target, float):
            t_ch = torch.full((B, 1, Nx, Ny, Nz), t_target,
                               device=device, dtype=u_initial.dtype)
        else:
            t_ch = t_target.view(B, 1, 1, 1, 1).expand(B, 1, Nx, Ny, Nz)

        grid = self._make_grid(u_initial.shape, device)
        x    = torch.cat([u_initial, grid, t_ch], dim=1)   # (B, 5, Nx, Ny, Nz)

        x = self.lift(x)
        for layer in self.layers:
            x = layer(x, sigma)

        mean, log_var = self.head(x)
        return mean, log_var

    # ─── MC uncertainty sampling ──────────────────────────────────────────

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        u_initial: Tensor,
        sigma:     Tensor,
        t_target:  float  = 1.0,
        n_samples: int    = 32,
    ) -> Dict[str, Tensor]:
        """
        Run MC-Dropout ensemble to estimate predictive uncertainty.

        Puts model in *train* mode temporarily to activate dropout, then
        restores the original mode.

        Returns dict with keys:
            'mean'  : (B, 1, Nx, Ny, Nz) — sample mean
            'std'   : (B, 1, Nx, Ny, Nz) — sample std (epistemic uncertainty)
            'samples' : (n_samples, B, 1, Nx, Ny, Nz) — raw samples
        """
        was_training = self.training
        self.train()   # activate dropout for MC sampling

        preds = []
        for _ in range(n_samples):
            m, _ = self(u_initial, sigma, t_target)
            preds.append(m)

        if not was_training:
            self.eval()

        stacked = torch.stack(preds, dim=0)   # (S, B, 1, Nx, Ny, Nz)
        return {
            "mean":    stacked.mean(0),
            "std":     stacked.std(0),
            "samples": stacked,
        }

    # ─── parameter count ──────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 8.  PhysicsLoss  —  physics-informed training objective
# =============================================================================

class PhysicsLoss(nn.Module):
    """
    Composite physics-informed loss for training StructuralFNO3D.

    Terms:
        L_mse     : standard mean-squared error (data fit)
        L_nll     : negative log-likelihood under Gaussian assumption
                    (encourages calibrated uncertainty)
        L_mass    : mass-conservation penalty |∫u_pred - ∫u_true| / |∫u_true|
                    (Cahn-Hilliard conserves order parameter)
        L_energy  : energy-monotonicity penalty max(0, E(u_pred) - E(u_0))
                    (Structural CH is a gradient flow — energy must decrease)
        L_smooth  : Structural Itô regularisation: penalises |∇u_pred|² in
                    high-σ regions (sharp gradients expected only at interfaces)

    Total loss:
        L = w_mse·L_mse + w_nll·L_nll + w_mass·L_mass
          + w_energy·L_energy + w_smooth·L_smooth

    Args:
        w_mse, w_nll, w_mass, w_energy, w_smooth : loss weights
        epsilon : interface width for energy functional (matches CH config)
    """

    def __init__(
        self,
        w_mse:    float = 1.0,
        w_nll:    float = 0.5,
        w_mass:   float = 0.1,
        w_energy: float = 0.05,
        w_smooth: float = 0.01,
        epsilon:  float = 1.5,
        dx:       float = 1.0,
    ):
        super().__init__()
        self.w_mse    = w_mse
        self.w_nll    = w_nll
        self.w_mass   = w_mass
        self.w_energy = w_energy
        self.w_smooth = w_smooth
        self.epsilon  = epsilon
        self.dx       = dx

    @staticmethod
    def _grad_sq(u: Tensor, dx: float) -> Tensor:
        """Approximate |∇u|² via finite differences (periodic)."""
        gx = (torch.roll(u, -1, -3) - torch.roll(u, +1, -3)) / (2 * dx)
        gy = (torch.roll(u, -1, -2) - torch.roll(u, +1, -2)) / (2 * dx)
        gz = (torch.roll(u, -1, -1) - torch.roll(u, +1, -1)) / (2 * dx)
        return gx**2 + gy**2 + gz**2

    def _bulk_energy(self, u: Tensor) -> Tensor:
        """CH double-well bulk free energy (non-differentiable-safe version)."""
        f_bulk = 0.25 * (u**2 - 1.0)**2
        f_grad = 0.5 * self.epsilon**2 * self._grad_sq(u, self.dx)
        return (f_bulk + f_grad).mean(dim=[-3, -2, -1])   # (B,1)

    def forward(
        self,
        pred:       Tensor,
        log_var:    Tensor,
        target:     Tensor,
        u_initial:  Tensor,
        sigma:      Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            pred      : (B, 1, Nx, Ny, Nz) model prediction
            log_var   : (B, 1, Nx, Ny, Nz) model log-variance
            target    : (B, 1, Nx, Ny, Nz) ground truth
            u_initial : (B, 1, Nx, Ny, Nz) initial field (for energy check)
            sigma     : (B, 1, Nx, Ny, Nz) regime field  (for smooth penalty)

        Returns:
            dict with 'total' and individual term tensors.
        """
        # MSE
        L_mse = F.mse_loss(pred, target)

        # NLL under diagonal Gaussian  (heteroscedastic)
        L_nll = (0.5 * (log_var + (pred - target)**2 / (log_var.exp() + 1e-8))).mean()

        # Mass conservation (sum over spatial dims)
        mass_pred = pred.mean(dim=[-3, -2, -1])
        mass_true = target.mean(dim=[-3, -2, -1])
        L_mass    = ((mass_pred - mass_true).abs() /
                     (mass_true.abs() + 1e-8)).mean()

        # Energy monotonicity
        E_pred = self._bulk_energy(pred)
        E_init = self._bulk_energy(u_initial).detach()
        L_energy = F.relu(E_pred - E_init).mean()

        # Structural Itô smoothness in high-sigma regions
        # High σ → ordered phase → gradients should be small
        # Low  σ → interface     → large gradients are expected
        grad_sq    = self._grad_sq(pred, self.dx)
        sigma_norm = torch.sigmoid(sigma - sigma.mean())   # ∈ (0,1)
        L_smooth   = (sigma_norm * grad_sq).mean()

        total = (self.w_mse    * L_mse
               + self.w_nll    * L_nll
               + self.w_mass   * L_mass
               + self.w_energy * L_energy
               + self.w_smooth * L_smooth)

        return {
            "total":    total,
            "mse":      L_mse.detach(),
            "nll":      L_nll.detach(),
            "mass":     L_mass.detach(),
            "energy":   L_energy.detach(),
            "smooth":   L_smooth.detach(),
        }


# =============================================================================
# 9.  SuperDNSDataset  —  data pipeline for SUPER DNS ONE output files
# =============================================================================

@dataclass
class SuperDNSSnapshot:
    """
    One training sample produced by SUPER DNS ONE cluster.

    Field layout produced by super_dns_one_v6.py / structural_cahn_hilliard_3d.py:
        'u_t0'     : (1, Nx, Ny, Nz)  — field at initial time
        'u_tT'     : (1, Nx, Ny, Nz)  — field at target time
        'sigma'    : (1, Nx, Ny, Nz)  — structural regime field
        't_target' : scalar float      — normalised target time
        'metadata' : dict              — solver config, dt, dx, etc.
    """
    u_t0:     Tensor
    u_tT:     Tensor
    sigma:    Tensor
    t_target: float
    metadata: Dict = field(default_factory=dict)


class SuperDNSDataset(Dataset):
    """
    PyTorch Dataset for .pt snapshot files saved by SUPER DNS ONE.

    Each file should contain a dict with the keys defined in SuperDNSSnapshot.
    Files are discovered recursively under `root_dir`.

    Usage::

        ds     = SuperDNSDataset("/data/dns_snapshots", normalise=True)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
        u0, uT, sigma, t = next(iter(loader))

    Args:
        root_dir   : directory containing .pt snapshot files
        normalise  : if True, normalise u fields to zero mean / unit std
                     using statistics computed over the dataset
        cache_size : number of samples to keep in RAM (0 = no cache)
        transform  : optional callable applied to (u0, uT, sigma) tensors
    """

    def __init__(
        self,
        root_dir:   str | Path,
        normalise:  bool = True,
        cache_size: int  = 512,
        transform   = None,
    ):
        self.root      = Path(root_dir)
        self.files     = sorted(self.root.rglob("*.pt"))
        self.normalise = normalise
        self.transform = transform
        self._cache: Dict[int, SuperDNSSnapshot] = {}
        self._cache_size = cache_size

        if not self.files:
            logger.warning(f"SuperDNSDataset: no .pt files found in {self.root}")

        self._u_mean, self._u_std = 0.0, 1.0
        if normalise and self.files:
            self._compute_statistics()

    def _compute_statistics(self, max_samples: int = 256) -> None:
        """Running mean / std over a subset of files for normalisation."""
        vals = []
        for fpath in self.files[:max_samples]:
            try:
                d = torch.load(fpath, map_location="cpu", weights_only=True)
                vals.append(d["u_t0"].float().flatten())
                vals.append(d["u_tT"].float().flatten())
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")
        if vals:
            all_v        = torch.cat(vals)
            self._u_mean = all_v.mean().item()
            self._u_std  = all_v.std().item() + 1e-8

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            snap = self._cache[idx]
        else:
            d = torch.load(self.files[idx], map_location="cpu", weights_only=True)
            snap = SuperDNSSnapshot(
                u_t0     = d["u_t0"].float(),
                u_tT     = d["u_tT"].float(),
                sigma    = d["sigma"].float(),
                t_target = float(d.get("t_target", 1.0)),
                metadata = d.get("metadata", {}),
            )
            if len(self._cache) < self._cache_size:
                self._cache[idx] = snap

        u0, uT, sig = snap.u_t0, snap.u_tT, snap.sigma
        if self.normalise:
            u0  = (u0  - self._u_mean) / self._u_std
            uT  = (uT  - self._u_mean) / self._u_std
        if self.transform is not None:
            u0, uT, sig = self.transform(u0, uT, sig)

        return u0, uT, sig, torch.tensor(snap.t_target, dtype=torch.float32)


# =============================================================================
# 10.  StructuralFNOTrainer  —  full training loop
# =============================================================================

@dataclass
class TrainerConfig:
    # Optimisation
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    max_epochs:     int   = 200
    batch_size:     int   = 4
    grad_clip:      float = 1.0
    # LR schedule: cosine warm-restart
    T_0:            int   = 20        # first restart cycle length (epochs)
    T_mult:         int   = 2         # cycle length multiplier
    eta_min:        float = 1e-6
    # AMP
    use_amp:        bool  = True
    # Checkpointing
    ckpt_dir:       str   = "./checkpoints"
    save_every:     int   = 10
    # Logging
    log_dir:        str   = "./runs"
    log_every:      int   = 10        # steps
    # Validation
    val_every:      int   = 1         # epochs
    val_fraction:   float = 0.1


class StructuralFNOTrainer:
    """
    Training manager for StructuralFNO3D.

    Features:
        • AdamW optimiser with cosine warm-restart LR schedule
        • Automatic Mixed Precision (AMP) via torch.cuda.amp
        • Gradient clipping (L2 norm)
        • TensorBoard scalar logging (loss terms + LR)
        • Checkpoint save/load (model + optimiser + scheduler states)
        • Validation loop with MSE + physics loss reporting

    Usage::

        model   = StructuralFNO3D(width=64, num_layers=6)
        ds      = SuperDNSDataset("/data/dns_snapshots")
        cfg     = TrainerConfig(max_epochs=100, batch_size=4)
        trainer = StructuralFNOTrainer(model, ds, cfg)
        trainer.train()

    Args:
        model    : StructuralFNO3D instance
        dataset  : SuperDNSDataset (or any compatible Dataset)
        config   : TrainerConfig dataclass
        device   : torch.device (auto-detected if None)
        loss_fn  : PhysicsLoss (created with defaults if None)
    """

    def __init__(
        self,
        model:   StructuralFNO3D,
        dataset: Dataset,
        config:  TrainerConfig,
        device:  Optional[torch.device] = None,
        loss_fn: Optional[PhysicsLoss]  = None,
    ):
        self.cfg  = config
        self.dev  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.loss_fn = loss_fn or PhysicsLoss()

        # Train / val split
        n_val   = max(1, int(len(dataset) * config.val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(
            val_ds,   batch_size=config.batch_size,
            shuffle=False, num_workers=2, pin_memory=True)

        # Optimiser + scheduler
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and self.dev.type == "cuda")

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=config.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available; skipping logging.")
            self.writer = None

        # Checkpoint dir
        Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)

        self._global_step = 0
        self._best_val    = float("inf")

    # ─── single training step ─────────────────────────────────────────────

    def _train_step(self, batch) -> Dict[str, float]:
        u0, uT, sig, t = [b.to(self.dev) for b in batch]

        self.optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp and self.dev.type == "cuda"):
            pred, log_var = self.model(u0, sig, t_target=t)
            losses        = self.loss_fn(pred, log_var, uT, u0, sig)

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optim)
        self.scaler.update()

        return {k: v.item() for k, v in losses.items()}

    # ─── validation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_mse = 0.0
        n = 0
        for batch in self.val_loader:
            u0, uT, sig, t = [b.to(self.dev) for b in batch]
            pred, _        = self.model(u0, sig, t_target=t)
            total_mse     += F.mse_loss(pred, uT).item() * u0.shape[0]
            n             += u0.shape[0]
        self.model.train()
        return total_mse / max(n, 1)

    # ─── checkpoint I/O ───────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_mse: float) -> Path:
        ckpt = {
            "epoch":      epoch,
            "val_mse":    val_mse,
            "model":      self.model.state_dict(),
            "optim":      self.optim.state_dict(),
            "scheduler":  self.sched.state_dict(),
            "scaler":     self.scaler.state_dict(),
            "sfno_version": SFNO_VERSION,
        }
        tag    = "best" if val_mse < self._best_val else f"ep{epoch:04d}"
        fpath  = Path(self.cfg.ckpt_dir) / f"sfno_{tag}.pt"
        torch.save(ckpt, fpath)
        logger.info(f"Saved checkpoint: {fpath}  (val_mse={val_mse:.4e})")
        return fpath

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint; returns starting epoch."""
        ckpt = torch.load(path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])
        self.sched.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        logger.info(f"Loaded checkpoint: {path}  epoch={ckpt['epoch']}")
        return ckpt["epoch"] + 1

    # ─── main training loop ───────────────────────────────────────────────

    def train(self, resume: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Args:
            resume : path to a checkpoint file to resume from (optional)
        """
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint(resume)

        logger.info(
            f"StructuralFNOTrainer — device={self.dev}, "
            f"params={self.model.count_parameters():,}, "
            f"epochs={self.cfg.max_epochs}")

        for epoch in range(start_epoch, self.cfg.max_epochs):
            self.model.train()
            t0 = time.time()
            epoch_losses: Dict[str, List[float]] = {}

            for batch in self.train_loader:
                step_losses = self._train_step(batch)
                self.sched.step()
                self._global_step += 1

                for k, v in step_losses.items():
                    epoch_losses.setdefault(k, []).append(v)

                if self.writer and self._global_step % self.cfg.log_every == 0:
                    for k, v in step_losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self._global_step)
                    self.writer.add_scalar(
                        "lr", self.optim.param_groups[0]["lr"], self._global_step)

            # Epoch summary
            mean_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            elapsed     = time.time() - t0
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.max_epochs}  "
                f"total={mean_losses.get('total', 0):.4e}  "
                f"mse={mean_losses.get('mse', 0):.4e}  "
                f"mass={mean_losses.get('mass', 0):.4e}  "
                f"t={elapsed:.1f}s"
            )

            # Validation
            if (epoch + 1) % self.cfg.val_every == 0:
                val_mse = self._validate()
                logger.info(f"  Val MSE: {val_mse:.4e}")
                if self.writer:
                    self.writer.add_scalar("val/mse", val_mse, epoch)
                if val_mse < self._best_val:
                    self._best_val = val_mse
                    self.save_checkpoint(epoch, val_mse)

            # Periodic checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch, val_mse if hasattr(self, "_last_val") else float("inf"))

        if self.writer:
            self.writer.close()
        logger.info("Training complete.")


# =============================================================================
# 11.  Verification suite
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    B, N       = 2, 32

    passed = 0; failed = 0

    def ok(name, extra=""):
        global passed; passed += 1
        print(f"  [PASS] {name}  {extra}")

    def fail(name, msg=""):
        global failed; failed += 1
        print(f"  [FAIL] {name}  -- {msg}")

    print("=" * 70)
    print(f"  Structural FNO 3D  v{SFNO_VERSION}  —  Verification Suite")
    print(f"  ONE Ecosystem (SUPER DNS ONE compatible) | device: {device_str}")
    print(f"  Assisted by: Claude (Anthropic), Gemini (Google), GPT (OpenAI)")
    print("=" * 70)

    # 1. MultiScaleSpectralConv3d forward + autograd
    try:
        conv = MultiScaleSpectralConv3d(16, 16, modes_coarse=8, modes_fine=4).to(device)
        x    = torch.randn(B, 16, N, N, N, device=device, requires_grad=True)
        out  = conv(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()
        ok("MultiScaleSpectralConv3d", f"out={out.shape}")
    except Exception as e:
        fail("MultiScaleSpectralConv3d", str(e))

    # 2. StructuralFiLM
    try:
        film  = StructuralFiLM(width=32, hidden=16).to(device)
        h     = torch.randn(B, 32, N, N, N, device=device, requires_grad=True)
        sigma = torch.rand(B, 1, N, N, N, device=device)
        out   = film(h, sigma)
        out.sum().backward()
        assert h.grad is not None and h.grad.isfinite().all()
        ok("StructuralFiLM (FiLM modulation)", f"out={out.shape}")
    except Exception as e:
        fail("StructuralFiLM", str(e))

    # 3. StructuralCrossAttention
    try:
        attn  = StructuralCrossAttention(width=32, n_heads=4, pool_k=4).to(device)
        h     = torch.randn(B, 32, N, N, N, device=device, requires_grad=True)
        sigma = torch.rand(B, 1, N, N, N, device=device)
        out   = attn(h, sigma)
        out.sum().backward()
        assert h.grad is not None and h.grad.isfinite().all()
        ok("StructuralCrossAttention", f"out={out.shape}")
    except Exception as e:
        fail("StructuralCrossAttention", str(e))

    # 4. StructuralFNOLayer (full block)
    try:
        layer = StructuralFNOLayer(width=32, modes_coarse=8, modes_fine=4).to(device)
        x     = torch.randn(B, 32, N, N, N, device=device, requires_grad=True)
        sigma = torch.rand(B, 1, N, N, N, device=device)
        out   = layer(x, sigma)
        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()
        ok("StructuralFNOLayer (full block)", f"out={out.shape}")
    except Exception as e:
        fail("StructuralFNOLayer", str(e))

    # 5. Full StructuralFNO3D forward
    try:
        model = StructuralFNO3D(
            modes_coarse=8, modes_fine=4,
            width=32, num_layers=4, n_heads=4
        ).to(device)
        u0    = torch.randn(B, 1, N, N, N, device=device, requires_grad=True)
        sigma = torch.rand(B, 1, N, N, N, device=device)
        mean, log_var = model(u0, sigma, t_target=0.5)
        assert mean.shape == u0.shape
        assert log_var.shape == u0.shape
        mean.sum().backward()
        assert u0.grad is not None and u0.grad.isfinite().all()
        ok("StructuralFNO3D forward + autograd",
           f"params={model.count_parameters():,}")
    except Exception as e:
        fail("StructuralFNO3D forward", str(e))

    # 6. PhysicsLoss
    try:
        loss_fn = PhysicsLoss()
        pred    = torch.randn(B, 1, N, N, N, device=device, requires_grad=True)
        log_var = torch.zeros_like(pred)
        target  = torch.randn_like(pred)
        u0      = torch.randn_like(pred)
        sigma   = torch.rand(B, 1, N, N, N, device=device)
        losses  = loss_fn(pred, log_var, target, u0, sigma)
        losses["total"].backward()
        assert pred.grad is not None and pred.grad.isfinite().all()
        ok("PhysicsLoss (all terms)",
           " | ".join(f"{k}={v.item():.3e}" for k, v in losses.items()))
    except Exception as e:
        fail("PhysicsLoss", str(e))

    # 7. MC-Dropout uncertainty estimation
    try:
        model2  = StructuralFNO3D(
            modes_coarse=8, modes_fine=4, width=32, num_layers=2
        ).to(device)
        u0      = torch.randn(B, 1, N, N, N, device=device)
        sigma   = torch.rand(B, 1, N, N, N, device=device)
        result  = model2.predict_with_uncertainty(u0, sigma, n_samples=8)
        assert result["mean"].shape  == u0.shape
        assert result["std"].shape   == u0.shape
        assert result["samples"].shape[0] == 8
        ok("MC-Dropout uncertainty",
           f"mean std={result['std'].mean().item():.4f}")
    except Exception as e:
        fail("MC-Dropout uncertainty", str(e))

    # 8. BoundaryPaddingConv3d (circular BC)
    try:
        bc_conv = BoundaryPaddingConv3d(8, 8, kernel_size=3, pad_mode="circular").to(device)
        x       = torch.randn(B, 8, N, N, N, device=device, requires_grad=True)
        out     = bc_conv(x)
        out.sum().backward()
        assert x.grad is not None
        ok("BoundaryPaddingConv3d (circular)", f"out={out.shape}")
    except Exception as e:
        fail("BoundaryPaddingConv3d", str(e))

    print("=" * 70)
    print(f"  Tests passed={passed}  failed={failed}")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
