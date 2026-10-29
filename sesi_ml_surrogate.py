"""
=============================================================================
SESI MACHINE LEARNING SURROGATE MODELS
Neural operators for fast inference, calibrated against reference solvers
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[C/D] Train neural operator surrogates (FNO-style) to approximate:
  • CFD solutions (Navier-Stokes, etc.)
  • EM field distributions
  • Quantum wavefunctions
  • Any PDE-based simulation

The surrogate is CALIBRATED against a reference solver (e.g., FDTD, FEM)
to ensure accuracy within a specified tolerance.

ARCHITECTURE
------------
  Input: (x, t, params) → Output: u(x, t)
  
  Neural Operator (FNO-style):
    • Lift: R^d_in → R^d_hidden
    • Spectral layers: K(θ) · FFT(u) + W · u
    • Project: R^d_hidden → R^d_out

CALIBRATION
-----------
  1. Generate training data from reference solver
  2. Train surrogate on (input, output) pairs
  3. Validate on hold-out set
  4. Report error metrics (L2, L∞, relative error)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Surrogate is an APPROXIMATION, not an exact solver.
• Accuracy depends on training data quality and quantity.
• Extrapolation beyond training distribution is unreliable.
• Must validate against reference solver before deployment.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.ml_surrogate")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "SurrogateConfig",
    "SpectralConv1d",
    "SpectralConv2d",
    "FNO1d",
    "FNO2d",
    "SurrogateTrainer",
    "SurrogateValidator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SurrogateConfig:
    """Configuration for neural operator surrogate."""
    # Architecture
    input_dim: int = 3  # (x, t, params)
    output_dim: int = 1  # u
    hidden_dim: int = 64
    n_layers: int = 4
    n_modes: int = 16  # Fourier modes

    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 32

    # Calibration
    validation_split: float = 0.2
    error_tolerance: float = 1e-2  # relative L2 error

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [C] SPECTRAL CONVOLUTION LAYERS (FNO-style)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class SpectralConv1d(nn.Module):
        """
        [C] 1D spectral convolution (Fourier Neural Operator layer).

        Applies linear transformation in Fourier space:
            K(θ) · FFT(u)
        """

        def __init__(self, in_channels: int, out_channels: int, n_modes: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_modes = n_modes

            # Fourier weights (complex)
            scale = 1 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, n_modes, dtype=torch.cfloat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, channels, grid_size)

            Returns:
                (batch, out_channels, grid_size)
            """
            batch_size = x.shape[0]

            # FFT
            x_fft = torch.fft.rfft(x, dim=-1)

            # Multiply by weights (only first n_modes)
            out_fft = torch.einsum("bix,iox->box",
                                   x_fft[..., :self.n_modes],
                                   self.weights)

            # Pad with zeros for remaining modes
            padding = x_fft.shape[-1] - self.n_modes
            out_fft = F.pad(out_fft, (0, padding))

            # Inverse FFT
            return torch.fft.irfft(out_fft, n=x.shape[-1], dim=-1)

    class SpectralConv2d(nn.Module):
        """
        [C] 2D spectral convolution (Fourier Neural Operator layer).
        """

        def __init__(self, in_channels: int, out_channels: int,
                     n_modes_x: int, n_modes_y: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_modes_x = n_modes_x
            self.n_modes_y = n_modes_y

            scale = 1 / (in_channels * out_channels)
            self.weights1 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, n_modes_x, n_modes_y, dtype=torch.cfloat)
            )
            self.weights2 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, n_modes_x, n_modes_y, dtype=torch.cfloat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, channels, grid_x, grid_y)

            Returns:
                (batch, out_channels, grid_x, grid_y)
            """
            batch_size = x.shape[0]

            # 2D FFT
            x_fft = torch.fft.rfft2(x, dim=(-2, -1))

            # Multiply by weights (upper-left quadrant)
            out_fft1 = torch.einsum("bixy,ioxy->boxy",
                                    x_fft[..., :self.n_modes_x, :self.n_modes_y],
                                    self.weights1)

            # Multiply by weights (lower-left quadrant)
            out_fft2 = torch.einsum("bixy,ioxy->boxy",
                                    x_fft[..., -self.n_modes_x:, :self.n_modes_y],
                                    self.weights2)

            # Pad with zeros
            out_fft = torch.zeros_like(x_fft)
            out_fft[..., :self.n_modes_x, :self.n_modes_y] = out_fft1
            out_fft[..., -self.n_modes_x:, :self.n_modes_y] = out_fft2

            # Inverse FFT
            return torch.fft.irfft2(out_fft, s=x.shape[-2:], dim=(-2, -1))


# ═══════════════════════════════════════════════════════════════════════
# [C/D] FNO MODELS
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class FNO1d(nn.Module):
        """
        [C] 1D Fourier Neural Operator.

        Architecture:
          • Lift: Linear(input_dim → hidden_dim)
          • Spectral layers: SpectralConv1d + Linear
          • Project: Linear(hidden_dim → output_dim)
        """

        def __init__(self, cfg: SurrogateConfig):
            super().__init__()
            self.cfg = cfg

            # Lift
            self.lift = nn.Linear(cfg.input_dim, cfg.hidden_dim)

            # Spectral layers
            self.spectral_layers = nn.ModuleList([
                SpectralConv1d(cfg.hidden_dim, cfg.hidden_dim, cfg.n_modes)
                for _ in range(cfg.n_layers)
            ])
            self.linear_layers = nn.ModuleList([
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, 1)
                for _ in range(cfg.n_layers)
            ])

            # Project
            self.project = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, grid_size, input_dim)

            Returns:
                (batch, grid_size, output_dim)
            """
            # Lift
            x = self.lift(x)  # (batch, grid_size, hidden_dim)
            x = x.permute(0, 2, 1)  # (batch, hidden_dim, grid_size)

            # Spectral layers
            for spectral, linear in zip(self.spectral_layers, self.linear_layers):
                x1 = spectral(x)
                x2 = linear(x)
                x = x1 + x2
                x = F.gelu(x)

            # Project
            x = x.permute(0, 2, 1)  # (batch, grid_size, hidden_dim)
            x = self.project(x)  # (batch, grid_size, output_dim)

            return x

    class FNO2d(nn.Module):
        """
        [C] 2D Fourier Neural Operator.
        """

        def __init__(self, cfg: SurrogateConfig):
            super().__init__()
            self.cfg = cfg

            self.lift = nn.Linear(cfg.input_dim, cfg.hidden_dim)

            self.spectral_layers = nn.ModuleList([
                SpectralConv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.n_modes, cfg.n_modes)
                for _ in range(cfg.n_layers)
            ])
            self.linear_layers = nn.ModuleList([
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1)
                for _ in range(cfg.n_layers)
            ])

            self.project = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, grid_x, grid_y, input_dim)

            Returns:
                (batch, grid_x, grid_y, output_dim)
            """
            batch_size, gx, gy, _ = x.shape

            # Lift
            x = self.lift(x)  # (batch, gx, gy, hidden_dim)
            x = x.permute(0, 3, 1, 2)  # (batch, hidden_dim, gx, gy)

            # Spectral layers
            for spectral, linear in zip(self.spectral_layers, self.linear_layers):
                x1 = spectral(x)
                x2 = linear(x)
                x = x1 + x2
                x = F.gelu(x)

            # Project
            x = x.permute(0, 2, 3, 1)  # (batch, gx, gy, hidden_dim)
            x = self.project(x)  # (batch, gx, gy, output_dim)

            return x


# ═══════════════════════════════════════════════════════════════════════
# [D] SURROGATE TRAINER
# ═══════════════════════════════════════════════════════════════════════

class SurrogateTrainer:
    """
    [D] Train neural operator surrogate on reference solver data.
    """

    def __init__(self, cfg: SurrogateConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor],
              val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
              ) -> Dict[str, Any]:
        """
        Train surrogate model.

        Args:
            train_data: (inputs, outputs) training tensors
            val_data: optional validation tensors

        Returns:
            dict with training history
        """
        inputs, outputs = train_data
        inputs = inputs.to(self.cfg.device)
        outputs = outputs.to(self.cfg.device)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Forward
            pred = self.model(inputs)
            loss = F.mse_loss(pred, outputs)

            # Backward
            loss.backward()
            self.optimizer.step()

            history["train_loss"].append(loss.item())

            # Validation
            if val_data is not None:
                val_inputs, val_outputs = val_data
                val_inputs = val_inputs.to(self.cfg.device)
                val_outputs = val_outputs.to(self.cfg.device)

                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(val_inputs)
                    val_loss = F.mse_loss(val_pred, val_outputs)
                history["val_loss"].append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.cfg.n_epochs}, "
                           f"Train Loss: {loss.item():.6f}")

        return history


# ═══════════════════════════════════════════════════════════════════════
# [D] SURROGATE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class SurrogateValidator:
    """
    [D] Validate surrogate accuracy against reference solver.
    """

    def __init__(self, cfg: SurrogateConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model

    def validate(self, test_data: Tuple[torch.Tensor, torch.Tensor]
                 ) -> Dict[str, Any]:
        """
        Validate surrogate on test data.

        Returns:
            dict with error metrics (L2, L∞, relative error)
        """
        inputs, outputs = test_data
        inputs = inputs.to(self.cfg.device)
        outputs = outputs.to(self.cfg.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(inputs)

        # Error metrics
        l2_error = torch.norm(pred - outputs, p=2).item()
        linf_error = torch.norm(pred - outputs, p=float('inf')).item()
        rel_l2_error = l2_error / (torch.norm(outputs, p=2).item() + 1e-10)

        passes = rel_l2_error < self.cfg.error_tolerance

        return {
            "l2_error": l2_error,
            "linf_error": linf_error,
            "relative_l2_error": rel_l2_error,
            "passes_tolerance": passes,
            "tolerance": self.cfg.error_tolerance,
            "note": ("Surrogate is an APPROXIMATION; accuracy depends on "
                     "training data quality and quantity."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI ML Surrogate Models — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available: {HAS_TORCH}")

    if not HAS_TORCH:
        print("  [SKIP] PyTorch required")
        return

    # Config
    cfg = SurrogateConfig(
        input_dim=3,
        output_dim=1,
        hidden_dim=32,
        n_layers=2,
        n_modes=8,
        n_epochs=20,
        batch_size=16,
    )

    # Create model
    print("\n  ── FNO 1D Model ──")
    model = FNO1d(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")

    # Generate synthetic training data
    print("\n  ── Training Data ──")
    n_train = 100
    grid_size = 64
    inputs = torch.randn(n_train, grid_size, cfg.input_dim)
    outputs = torch.randn(n_train, grid_size, cfg.output_dim)

    print(f"    Training samples: {n_train}")
    print(f"    Grid size: {grid_size}")

    # Train
    print("\n  ── Training ──")
    trainer = SurrogateTrainer(cfg, model)
    history = trainer.train((inputs, outputs))
    print(f"    Final train loss: {history['train_loss'][-1]:.6f}")

    # Validate
    print("\n  ── Validation ──")
    validator = SurrogateValidator(cfg, model)
    val_result = validator.validate((inputs[:20], outputs[:20]))
    print(f"    Relative L2 error: {val_result['relative_l2_error']:.6f}")
    print(f"    Passes tolerance: {val_result['passes_tolerance']}")
    print(f"    {val_result['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Surrogate is an APPROXIMATION, not an exact solver")
    print("  • Accuracy depends on training data quality")
    print("  • Extrapolation beyond training distribution is unreliable")
    print("  • Must validate against reference solver before deployment")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
