"""
=============================================================================
SESI OpenFOAM ↔ DIFFERENTIABLE SURROGATE — End-to-End Pipeline
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

END-TO-END ARCHITECTURE
───────────────────────
    ┌─────────────────┐
    │  OpenFOAM case  │  (run offline: simpleFoam / pisoFoam)
    │  0/, constant/, │
    │  system/, proc*/ │
    └────────┬────────┘
             │  parse field files (U, p, k, epsilon, nut)
             ▼
    ┌─────────────────┐
    │ OpenFOAMDataset │  PyTorch Dataset of (input, field) pairs
    └────────┬────────┘
             │  supervised training
             ▼
    ┌─────────────────────────┐
    │ DifferentiableCFDSurrogate │  FNO / U-Net style neural operator
    │ (fully autograd-native)    │
    └────────┬────────────────┘
             │  differentiable physics loss (NS residual)
             ▼
    ┌─────────────────┐
    │  Optimizer      │  geometry/BC params → grad → improved design
    └─────────────────┘

    The surrogate is 100–10000× faster than OpenFOAM and fully
    differentiable, enabling gradient-based design optimization.

HONEST SCOPE
────────────
• The surrogate is a DATA-DRIVEN APPROXIMATION of OpenFOAM, not OpenFOAM itself.
• Accuracy is bounded by training data coverage and resolution.
• Physics-informed loss helps but does not guarantee NS satisfaction.
• OpenFOAM must be installed separately to generate training data.
• For production CFD safety cases, always validate against OpenFOAM/experiment.
=============================================================================
"""

from __future__ import annotations

import os
import re
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.openfoam_surrogate")

__all__ = [
    "OpenFOAMSurrogateConfig",
    "OpenFOAMCaseParser",
    "OpenFOAMDataset",
    "SpectralConv2d",
    "DifferentiableCFDSurrogate",
    "PhysicsInformedLoss",
    "OpenFOAMSurrogateTrainer",
    "EndToEndCFDOptimizer",
]


def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OpenFOAMSurrogateConfig:
    # ── OpenFOAM case ─────────────────────────────────────────────────────
    case_dir: str = "./openfoam_case"
    field_names: List[str] = field(default_factory=lambda: ["U", "p"])
    time_dirs: Optional[List[str]] = None  # None = auto-detect
    # ── Grid / surrogate ──────────────────────────────────────────────────
    grid_h: int = 64
    grid_w: int = 64
    in_channels: int = 3          # e.g. boundary-condition / geometry encoding
    out_channels: int = 3         # Ux, Uy, p
    fourier_modes: int = 16       # FNO truncation
    fno_width: int = 32
    n_fno_layers: int = 4
    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 4
    lr: float = 1e-3
    epochs: int = 100
    w_data: float = 1.0           # data-fit weight
    w_physics: float = 0.1        # NS-residual weight
    nu: float = 1e-3              # kinematic viscosity for NS residual
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# 1. OpenFOAM CASE PARSER
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMCaseParser:
    """
    Parses OpenFOAM field files into numpy arrays.

    Handles the standard OpenFOAM ASCII field format:
        FoamFile { ... }
        dimensions [0 2 -2 0 0 0 0];
        internalField uniform 0;
        internalField nonuniform List<scalar> N ( v0 v1 ... );
        internalField nonuniform List<vector> N ( (ux uy uz) ... );
        boundaryField { ... }

    Also reads mesh points from constant/polyMesh/points to build a
    structured grid (nearest-neighbour resampling onto config grid).
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig):
        self.cfg = cfg
        self.case_dir = Path(cfg.case_dir)

    def list_time_dirs(self) -> List[str]:
        """Return sorted list of numeric time directories."""
        if self.cfg.time_dirs is not None:
            return self.cfg.time_dirs
        times = []
        for d in self.case_dir.iterdir():
            if d.is_dir() and re.match(r"^[\d.]+$", d.name):
                times.append(d.name)
        times.sort(key=float)
        return times

    def parse_points(self) -> Optional[np.ndarray]:
        """Parse constant/polyMesh/points → (N, 3) array."""
        points_file = self.case_dir / "constant" / "polyMesh" / "points"
        if not points_file.exists():
            logger.warning(f"points file not found: {points_file}")
            return None
        text = points_file.read_text(errors="ignore")
        match = re.search(r"\(\s*\n(.*?)\n\s*\)", text, re.DOTALL)
        if not match:
            return None
        vecs = re.findall(
            r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)",
            match.group(1),
        )
        return np.array(vecs, dtype=np.float64)

    def parse_scalar_field(self, field_path: Path) -> Optional[np.ndarray]:
        """Parse a scalar field (e.g. p) from a time directory."""
        if not field_path.exists():
            return None
        text = field_path.read_text(errors="ignore")

        # uniform
        m = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", text)
        if m:
            n = self._count_cells()
            return np.full(n, float(m.group(1)))

        # nonuniform List<scalar>
        m = re.search(
            r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;",
            text, re.DOTALL,
        )
        if m:
            values = np.fromstring(m.group(2), sep=" ", dtype=np.float64)
            return values
        return None

    def parse_vector_field(self, field_path: Path) -> Optional[np.ndarray]:
        """Parse a vector field (e.g. U) → (N, 3)."""
        if not field_path.exists():
            return None
        text = field_path.read_text(errors="ignore")

        # uniform vector
        m = re.search(
            r"internalField\s+uniform\s+\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*;",
            text,
        )
        if m:
            n = self._count_cells()
            vec = np.array([float(m.group(i)) for i in (1, 2, 3)])
            return np.tile(vec, (n, 1))

        # nonuniform List<vector>
        m = re.search(
            r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*\)\s*;",
            text, re.DOTALL,
        )
        if m:
            raw = m.group(2)
            vecs = re.findall(
                r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", raw
            )
            return np.array(vecs, dtype=np.float64)
        return None

    def _count_cells(self) -> int:
        """Best-effort cell count from polyMesh/owner."""
        owner_file = self.case_dir / "constant" / "polyMesh" / "owner"
        if owner_file.exists():
            text = owner_file.read_text(errors="ignore")
            m = re.search(r"note:.*?nCells:(\d+)", text)
            if m:
                return int(m.group(1))
        return 0

    def resample_to_grid(
        self, points: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        Resample scattered OpenFOAM cell-centre data onto a regular
        (grid_h, grid_w) grid via nearest-neighbour.

        Args:
            points: (N, 3) cell centres
            values: (N,) or (N, C) field values

        Returns:
            (grid_h, grid_w, C) array
        """
        H, W = self.cfg.grid_h, self.cfg.grid_w
        if values.ndim == 1:
            values = values[:, None]
        C = values.shape[1]

        # Normalize x, y to [0, 1]
        x = points[:, 0]
        y = points[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        xi = ((x - x_min) / x_span * (W - 1)).astype(int).clip(0, W - 1)
        yi = ((y - y_min) / y_span * (H - 1)).astype(int).clip(0, H - 1)

        grid = np.zeros((H, W, C), dtype=np.float64)
        count = np.zeros((H, W, 1), dtype=np.float64)
        np.add.at(grid, (yi, xi), values)
        np.add.at(count, (yi, xi), 1.0)
        count = np.maximum(count, 1.0)
        grid /= count
        return grid

    def load_snapshot(self, time_dir: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all configured fields at one timestep and resample to grid.

        Returns dict: {field_name: (H, W, C)} or None on failure.
        """
        points = self.parse_points()
        if points is None:
            return None

        snapshot = {}
        for name in self.cfg.field_names:
            fpath = self.case_dir / time_dir / name
            if name == "U":
                raw = self.parse_vector_field(fpath)   # (N, 3)
            else:
                raw = self.parse_scalar_field(fpath)   # (N,)
            if raw is None:
                logger.warning(f"Could not parse field {name} at t={time_dir}")
                return None
            if len(raw) != len(points):
                logger.warning(
                    f"Field {name} has {len(raw)} values but {len(points)} points; skipping."
                )
                return None
            snapshot[name] = self.resample_to_grid(points, raw)
        return snapshot


# ═══════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMDataset(Dataset):
    """
    PyTorch Dataset of OpenFOAM snapshots.

    Each sample is (input_encoding, target_field):
        input_encoding : (in_channels, H, W)  — geometry / BC encoding
        target_field   : (out_channels, H, W) — [Ux, Uy, p]

    If no real OpenFOAM data is found, generates a synthetic
    lid-driven-cavity-like dataset so the pipeline can be exercised.
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig):
        self.cfg = cfg
        self.parser = OpenFOAMCaseParser(cfg)
        self.samples = self._load_or_synthesize()

    def _load_or_synthesize(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        time_dirs = self.parser.list_time_dirs()
        samples = []

        if time_dirs:
            logger.info(f"Found {len(time_dirs)} OpenFOAM time directories")
            for td in time_dirs:
                snap = self.parser.load_snapshot(td)
                if snap is None:
                    continue
                # Assemble target: [Ux, Uy, p]
                U = snap.get("U")          # (H, W, 3)
                p = snap.get("p")          # (H, W, 1)
                if U is None:
                    continue
                target = np.concatenate(
                    [U[..., :2], p if p is not None else np.zeros((*U.shape[:2], 1))],
                    axis=-1,
                )  # (H, W, 3)
                target = np.transpose(target, (2, 0, 1))  # (3, H, W)
                # Input encoding: simple channel of ones + positional (placeholder)
                inp = self._make_input_encoding()
                samples.append((inp, target.astype(np.float32)))

        if not samples:
            logger.warning(
                "No OpenFOAM data loaded. Generating synthetic dataset. "
                "Install/run OpenFOAM and point case_dir at a real case for "
                "meaningful surrogate training."
            )
            samples = self._synthesize(n=32)

        return samples

    def _make_input_encoding(self) -> np.ndarray:
        H, W = self.cfg.grid_h, self.cfg.grid_w
        C = self.cfg.in_channels
        enc = np.zeros((C, H, W), dtype=np.float32)
        # Channel 0: constant 1 (bias-like)
        enc[0] = 1.0
        if C >= 2:
            # Channel 1: normalized x coordinate
            x = np.linspace(0, 1, W)[None, :]
            enc[1] = np.broadcast_to(x, (H, W))
        if C >= 3:
            # Channel 2: normalized y coordinate
            y = np.linspace(0, 1, H)[:, None]
            enc[2] = np.broadcast_to(y, (H, W))
        return enc

    def _synthesize(self, n: int = 32) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Synthetic Taylor-Green-vortex-like velocity + pressure fields."""
        samples = []
        H, W = self.cfg.grid_h, self.cfg.grid_w
        x = np.linspace(0, 2 * np.pi, W)
        y = np.linspace(0, 2 * np.pi, H)
        X, Y = np.meshgrid(x, y)

        rng = np.random.RandomState(42)
        for i in range(n):
            k = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            Ux = np.sin(k * X + phase) * np.cos(k * Y)
            Uy = -np.cos(k * X) * np.sin(k * Y + phase)
            p = 0.25 * (np.cos(2 * k * X + phase) + np.cos(2 * k * Y))
            target = np.stack([Ux, Uy, p], axis=0).astype(np.float32)
            inp = self._make_input_encoding()
            # Vary input slightly so the surrogate has something to condition on
            inp = inp + rng.randn(*inp.shape).astype(np.float32) * 0.01
            samples.append((inp, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return {
            "input": torch.tensor(inp, dtype=torch.float32),
            "target": torch.tensor(tgt, dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════════
# 3. FOURIER NEURAL OPERATOR (FNO) BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → truncated spectral transform → iFFT."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class DifferentiableCFDSurrogate(nn.Module):
    """
    Fourier Neural Operator surrogate for incompressible flow.

    Fully differentiable end-to-end:
        input (BC/geometry encoding) → FNO → [Ux, Uy, p]

    Because everything here is autograd-native, gradients flow from any
    downstream objective (drag, mixing, pressure drop) all the way back
    to the input encoding — enabling gradient-based design optimization.
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig):
        super().__init__()
        self.cfg = cfg

        self.lift = nn.Linear(cfg.in_channels, cfg.fno_width)

        self.spectral_layers = nn.ModuleList()
        for _ in range(cfg.n_fno_layers):
            self.spectral_layers.append(
                SpectralConv2d(cfg.fno_width, cfg.fno_width,
                               cfg.fourier_modes, cfg.fourier_modes)
            )
        self.pointwise = nn.ModuleList([
            nn.Conv2d(cfg.fno_width, cfg.fno_width, 1)
            for _ in range(cfg.n_fno_layers)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, cfg.fno_width) for _ in range(cfg.n_fno_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(cfg.fno_width, cfg.fno_width, 1),
            nn.GELU(),
            nn.Conv2d(cfg.fno_width, cfg.out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H, W)  →  [Ux, Uy, p]
        """
        # Lift channels: (B, C, H, W) → (B, H, W, width) → permute
        x = x.permute(0, 2, 3, 1)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)

        for spec, pw, norm in zip(self.spectral_layers, self.pointwise, self.norms):
            x1 = spec(x)
            x2 = pw(x)
            x = x1 + x2
            x = norm(x)
            x = F.gelu(x)

        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════
# 4. PHYSICS-INFORMED LOSS (Navier–Stokes residual)
# ═══════════════════════════════════════════════════════════════════════

class PhysicsInformedLoss(nn.Module):
    """
    Combined data-fit + physics loss.

    Data loss:
        MSE between surrogate output and OpenFOAM ground truth.

    Physics loss (incompressible Navier–Stokes residual):
        Continuity:  ∇·u = 0
        Momentum (steady, advective form):
            (u·∇)u + ∇p − ν∇²u = 0
        We compute these with autograd-friendly finite differences and
        penalize the residual. This is a soft constraint; it does NOT
        guarantee the surrogate exactly satisfies NS.
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _grad(field: Tensor, dx: float, dim: int) -> Tensor:
        """Central finite difference along a spatial dim."""
        return (torch.roll(field, -1, dims=dim) - torch.roll(field, 1, dims=dim)) / (2 * dx)

    @staticmethod
    def _laplacian(field: Tensor, dx: float) -> Tensor:
        lap = (
            torch.roll(field, 1, dims=-2) + torch.roll(field, -1, dims=-2)
            + torch.roll(field, 1, dims=-1) + torch.roll(field, -1, dims=-1)
            - 4 * field
        ) / (dx * dx)
        return lap

    def ns_residual(self, pred: Tensor) -> Tensor:
        """
        Compute continuity + momentum residual.

        Args:
            pred: (B, 3, H, W) → [Ux, Uy, p]
        Returns:
            scalar mean residual
        """
        ux = pred[:, 0:1]
        uy = pred[:, 1:2]
        p = pred[:, 2:3]

        dx = 1.0 / self.cfg.grid_w
        nu = self.cfg.nu

        # Continuity: dUx/dx + dUy/dy
        dux_dx = self._grad(ux, dx, dim=-1)
        duy_dy = self._grad(uy, dx, dim=-2)
        continuity = dux_dx + duy_dy

        # Momentum x: ux dUx/dx + uy dUx/dy + dp/dx - nu ∇²Ux
        dux_dy = self._grad(ux, dx, dim=-2)
        dp_dx = self._grad(p, dx, dim=-1)
        lap_ux = self._laplacian(ux, dx)
        mom_x = ux * dux_dx + uy * dux_dy + dp_dx - nu * lap_ux

        # Momentum y
        duy_dx = self._grad(uy, dx, dim=-1)
        dp_dy = self._grad(p, dx, dim=-2)
        lap_uy = self._laplacian(uy, dx)
        mom_y = ux * duy_dx + uy * duy_dy + dp_dy - nu * lap_uy

        residual = (
            continuity.pow(2).mean()
            + mom_x.pow(2).mean()
            + mom_y.pow(2).mean()
        )
        return residual

    def forward(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        data_loss = F.mse_loss(pred, target)
        physics_loss = self.ns_residual(pred)
        total = self.cfg.w_data * data_loss + self.cfg.w_physics * physics_loss
        return {
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "total": total,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. TRAINER
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMSurrogateTrainer:
    """Train the differentiable surrogate on OpenFOAM data."""

    def __init__(self, cfg: OpenFOAMSurrogateConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = DifferentiableCFDSurrogate(cfg).to(self.device)
        self.loss_fn = PhysicsInformedLoss(cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

    def train(self, dataset: OpenFOAMDataset) -> Dict[str, List[float]]:
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        history = {"data_loss": [], "physics_loss": [], "total": []}

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_stats = {"data_loss": 0.0, "physics_loss": 0.0, "total": 0.0}
            n_batches = 0

            for batch in loader:
                x = batch["input"].to(self.device)
                y = batch["target"].to(self.device)

                pred = self.model(x)
                losses = self.loss_fn(pred, y)

                self.opt.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                for k in epoch_stats:
                    epoch_stats[k] += losses[k].item()
                n_batches += 1

            for k in epoch_stats:
                epoch_stats[k] /= max(n_batches, 1)
                history[k].append(epoch_stats[k])

            if (epoch + 1) % max(1, self.cfg.epochs // 10) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.epochs}  "
                    f"data={epoch_stats['data_loss']:.4e}  "
                    f"physics={epoch_stats['physics_loss']:.4e}  "
                    f"total={epoch_stats['total']:.4e}"
                )

        return history

    @torch.no_grad()
    def evaluate(self, dataset: OpenFOAMDataset) -> Dict[str, float]:
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        self.model.eval()
        tot_data, tot_phys, n = 0.0, 0.0, 0
        for batch in loader:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)
            pred = self.model(x)
            losses = self.loss_fn(pred, y)
            tot_data += losses["data_loss"].item()
            tot_phys += losses["physics_loss"].item()
            n += 1
        return {
            "data_loss": tot_data / max(n, 1),
            "physics_loss": tot_phys / max(n, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# 6. END-TO-END DIFFERENTIABLE CFD OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

class EndToEndCFDOptimizer:
    """
    Gradient-based design optimization through the differentiable surrogate.

    Because the surrogate is fully autograd-native, we can optimize an
    upstream design variable (e.g. inlet profile, geometry encoding) to
    minimize/maximize a flow objective WITHOUT re-running OpenFOAM.

    Example objective: minimize mean kinetic energy in a region, or
    maximize pressure drop, or minimize drag proxy.

    This is the "end-to-end" promise:
        design_params → surrogate → flow field → objective → ∇ → updated params

    HONEST SCOPE:
      • Optimizes the SURROGATE, not the true NS solution.
      • The surrogate is only trustworthy within the distribution of its
        training data. Out-of-distribution designs give unreliable gradients.
      • Always validate an optimized design with a real OpenFOAM run.
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig, surrogate: DifferentiableCFDSurrogate):
        self.cfg = cfg
        self.device = next(surrogate.parameters()).device
        self.surrogate = surrogate.to(self.device)
        self.surrogate.eval()
        # Freeze surrogate weights during design optimization
        for p in self.surrogate.parameters():
            p.requires_grad_(False)

    def objective(self, flow: Tensor) -> Tensor:
        """
        Example objective: minimize total kinetic energy (proxy for drag/loss).
        Override this for your real design goal.

        Args:
            flow: (B, 3, H, W) → [Ux, Uy, p]
        Returns:
            scalar objective
        """
        ux = flow[:, 0]
        uy = flow[:, 1]
        ke = 0.5 * (ux.pow(2) + uy.pow(2))
        return ke.mean()

    def optimize(
        self,
        initial_design: Tensor,
        n_steps: int = 200,
        lr: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        Optimize a design-encoding tensor through the surrogate.

        Args:
            initial_design: (1, in_channels, H, W) starting encoding
            n_steps: optimization iterations
            lr: step size

        Returns:
            dict with optimized design, objective history, final flow
        """
        design = initial_design.clone().to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([design], lr=lr)

        history = []
        for step in range(n_steps):
            optimizer.zero_grad()
            flow = self.surrogate(design)
            obj = self.objective(flow)
            obj.backward()
            optimizer.step()
            history.append(obj.item())

            if (step + 1) % max(1, n_steps // 10) == 0:
                logger.info(f"Opt step {step+1}/{n_steps}  objective={obj.item():.6e}")

        with torch.no_grad():
            final_flow = self.surrogate(design)

        return {
            "optimized_design": design.detach(),
            "final_flow": final_flow.detach(),
            "objective_history": history,
            "final_objective": history[-1],
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI OpenFOAM ↔ Differentiable Surrogate — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(0)

    cfg = OpenFOAMSurrogateConfig(
        grid_h=32, grid_w=32,
        in_channels=3, out_channels=3,
        fourier_modes=8, fno_width=16, n_fno_layers=3,
        batch_size=4, epochs=5, device=str(device),
    )

    # Dataset (will fall back to synthetic if no OpenFOAM case present)
    ds = OpenFOAMDataset(cfg)
    print(f"  Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"  Input shape : {tuple(sample['input'].shape)}")
    print(f"  Target shape: {tuple(sample['target'].shape)}")

    # Surrogate forward
    model = DifferentiableCFDSurrogate(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Surrogate parameters: {n_par:,}")

    x = sample["input"].unsqueeze(0).to(device)
    pred = model(x)
    print(f"  Surrogate output shape: {tuple(pred.shape)}")

    # Physics loss
    loss_fn = PhysicsInformedLoss(cfg).to(device)
    y = sample["target"].unsqueeze(0).to(device)
    losses = loss_fn(pred, y)
    print(f"  Data loss    : {losses['data_loss'].item():.4e}")
    print(f"  Physics loss : {losses['physics_loss'].item():.4e}")

    # Backward through surrogate (verify differentiability)
    losses["total"].backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    print(f"  Surrogate differentiable: {'PASS' if grad_ok else 'FAIL'}")

    # End-to-end design optimization
    print("\n  ── End-to-end design optimization ──")
    optimizer = EndToEndCFDOptimizer(cfg, model)
    initial_design = torch.randn(1, cfg.in_channels, cfg.grid_h, cfg.grid_w)
    result = optimizer.optimize(initial_design, n_steps=20, lr=1e-2)
    print(f"  Initial objective: {result['objective_history'][0]:.6e}")
    print(f"  Final objective  : {result['final_objective']:.6e}")
    print(f"  Optimized design requires_grad: {result['optimized_design'].requires_grad}")

    print("\n" + "=" * 72)
    print("  SMOKE TEST COMPLETE")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Surrogate accuracy depends on OpenFOAM training data coverage.")
    print("  • Physics loss is a soft constraint, not an exact NS solve.")
    print("  • Optimized designs MUST be validated with a real OpenFOAM run.")
    print("  • OpenFOAM itself must be installed separately to generate data.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
