"""
Production-grade Neuro-Sensory Enhancement Module (NSEM).

Branches (all differentiable end-to-end):
  1. Carbon <-> Silicon interface  (spike ingestion, normalization)
  2. Photonic routing              (vision + audition, high bandwidth)
  3. Quantum-inspired cognition    (complex amplitudes via dual real streams)
  4. In-memory recurrence          (stateful memory retention)
  5. OTA motor control + chemo     (muscle state, olfaction, gustation)

# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

Design notes
------------
* Complex ops are implemented as two real matmuls so the whole graph is
  AMP-compatible (autocast does not support complex64).
* The vision head emits a compact latent (vision_latent_dim); downstream
  neural-field / diffusion decoders sample it.  The original single Linear
  to 3x2160x3840 would be ~2.5e10 parameters and is not trainable.
* No registered buffers, all parameters used every forward -> DDP with
  find_unused_parameters=False and static_graph=True.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class NSEMConfig:
    biological_channels: int = 1024
    hidden_dim: int = 512
    vision_latent_dim: int = 1024       # downstream decoder consumes this
    audio_bands: int = 256              # cochlear frequency bins
    chemical_receptors: int = 128       # olfactory + gustatory
    motor_dof: int = 256                # normalized muscle activations
    num_quantum_layers: int = 2
    dropout: float = 0.05
    gradient_checkpointing: bool = False


# ---------------------------------------------------------------------------
# Complex linear as dual real streams  (AMP-safe)
# ---------------------------------------------------------------------------
class DualStreamLinear(nn.Module):
    """
    Complex-valued Linear:  (xr + i*xi) @ (Wr + i*Wi)^T + (br + i*bi)

    Implemented as 4 real matmuls + 2 adds.  Never materializes complex64,
    so it composes cleanly with torch.autocast and DDP.
    """

    __constants__ = ["in_features", "out_features", "has_bias"]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Complex He init: Var(|W|) = 1/in  ->  Var(Re) = Var(Im) = 1/(2*in)
        std = math.sqrt(1.0 / (2.0 * self.in_features))
        with torch.no_grad():
            nn.init.normal_(self.weight_real, 0.0, std)
            nn.init.normal_(self.weight_imag, 0.0, std)
            if self.bias_real is not None:
                nn.init.zeros_(self.bias_real)
                nn.init.zeros_(self.bias_imag)

    def forward(
        self, xr: torch.Tensor, xi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # complex multiply:  (xr + i xi)(Wr + i Wi)
        #  real:  xr Wr - xi Wi
        #  imag:  xr Wi + xi Wr
        yr = F.linear(xr, self.weight_real) - F.linear(xi, self.weight_imag)
        yi = F.linear(xr, self.weight_imag) + F.linear(xi, self.weight_real)
        if self.bias_real is not None:
            yr = yr + self.bias_real
            yi = yi + self.bias_imag
        return yr, yi

    extra_repr = lambda self: (
        f"in_features={self.in_features}, out_features={self.out_features}, "
        f"bias={self.has_bias}, complex=True"
    )


# ---------------------------------------------------------------------------
# Quantum-inspired cognition branch
# ---------------------------------------------------------------------------
class QuantumCognitionBranch(nn.Module):
    """
    Simulated state-vector evolution as a complex-valued MLP.
    Measurement step returns |psi| = sqrt(Re^2 + Im^2 + eps).
    """

    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DualStreamLinear(dim, dim) for _ in range(num_layers)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x, torch.zeros_like(x)
        for layer in self.layers:
            xr, xi = layer(xr, xi)
            xr = self.drop(self.act(xr))
            xi = self.drop(self.act(xi))
        # |psi|^2 measurement (sqrt for downstream scale stability)
        magnitude = torch.sqrt(xr * xr + xi * xi + 1e-6)
        return self.proj(self.norm(magnitude))


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class NeuroSensoryEnhancementModule(nn.Module):
    def __init__(self, config: Optional[NSEMConfig] = None, **kwargs) -> None:
        super().__init__()
        self.cfg = config if config is not None else NSEMConfig(**kwargs)
        H = self.cfg.hidden_dim

        # ---- 1. Carbon <-> Silicon interface -----------------------------
        self.interface = nn.Sequential(
            nn.Linear(self.cfg.biological_channels, H * 2, bias=False),
            nn.GELU(),
            nn.LayerNorm(H * 2),
            nn.Linear(H * 2, H, bias=False),
            nn.LayerNorm(H),
        )

        # ---- 2. Photonic routing -----------------------------------------
        self.photonic_router = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.LayerNorm(H),
        )
        self.vision_latent = nn.Linear(H, self.cfg.vision_latent_dim)
        self.audio_decoder = nn.Linear(H, self.cfg.audio_bands)

        # ---- 3. Quantum-inspired cognition -------------------------------
        self.cognition = QuantumCognitionBranch(
            dim=H, num_layers=self.cfg.num_quantum_layers, dropout=self.cfg.dropout
        )

        # ---- 4. In-memory recurrence -------------------------------------
        self.memory_cell = nn.GRUCell(H, H)
        self.memory_norm = nn.LayerNorm(H)

        # ---- 5. OTA motor + chemosensory ---------------------------------
        self.motor_head = nn.Sequential(
            nn.Linear(H, self.cfg.motor_dof),
            nn.Tanh(),  # normalized activation in [-1, 1]
        )
        self.chemical_head = nn.Linear(H, self.cfg.chemical_receptors)

        self.dropout = nn.Dropout(self.cfg.dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        neural_spike_stream: torch.Tensor,           # [B, bio_channels]
        memory_state: Optional[torch.Tensor] = None, # [B, H] or None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        B = neural_spike_stream.shape[0]
        H = self.cfg.hidden_dim
        ckpt = self.cfg.gradient_checkpointing and self.training

        # ---- 1. Interface ------------------------------------------------
        base = (
            checkpoint(self.interface, neural_spike_stream, use_reentrant=False)
            if ckpt else self.interface(neural_spike_stream)
        )
        base = self.dropout(base)

        # ---- 2. Photonic -------------------------------------------------
        photonic = self.photonic_router(base)
        vision_latent = self.vision_latent(photonic)
        auditory = self.audio_decoder(photonic)

        # ---- 3. Quantum cognition ---------------------------------------
        cognitive = (
            checkpoint(self.cognition, base, use_reentrant=False)
            if ckpt else self.cognition(base)
        )

        # ---- 4. In-memory ------------------------------------------------
        if memory_state is None:
            memory_state = base.new_zeros(B, H)
        next_memory = self.memory_norm(self.memory_cell(base, memory_state))

        # ---- 5. OTA motor + chemosensory --------------------------------
        motor = self.motor_head(base)
        chemical = torch.sigmoid(self.chemical_head(base))

        outputs = {
            "vision_latent":        vision_latent,   # feed to neural-field decoder
            "auditory":             auditory,
            "cognitive_feedback":   cognitive,
            "motor_control":        motor,
            "olfactory_gustatory":  chemical,
            "memory_readout":       next_memory,
        }
        return outputs, next_memory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def initial_memory(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        p = next(self.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device, dtype=dtype)

    def parameter_groups(
        self,
        weight_decay: float = 0.01,
        no_decay_keywords: Tuple[str, ...] = ("bias", "norm", "layernorm"),
    ):
        """
        Splits parameters into decay / no-decay groups (drop-in for AdamW).
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            lowered = name.lower()
            if p.ndim == 1 or any(k in lowered for k in no_decay_keywords):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
