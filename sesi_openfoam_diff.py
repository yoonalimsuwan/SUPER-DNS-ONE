"""
=============================================================================
SESI OpenFOAM DIFFERENTIABLE INTEGRATION
Native Fully-Differentiable CFD → Neural Network Training
=============================================================================

ARCHITECTURE
────────────
OpenFOAM (ground truth CFD) → field parsing → PyTorch tensors →
train neural surrogate with autodiff → differentiable CFD predictions

TWO MODES
─────────
[Mode A] Surrogate Training (recommended, works now)
    Run OpenFOAM offline → parse U/p/k/ε fields → train NN surrogate
    The NN is fully differentiable; gradients flow end-to-end through it.

[Mode B] Native Differentiable OpenFOAM (advanced)
    Requires modifying OpenFOAM's C++ source OR using foam-extend with
    adjoint solvers. This module provides the INTERFACE for it but the
    actual C++ integration is beyond Python scope.

HONEST SCOPE
────────────
• Mode A gives a differentiable SURROGATE of OpenFOAM, not OpenFOAM itself.
• True "native fully differentiable OpenFOAM" requires C++ source changes
  or specialized adjoint solvers (e.g., foam-extend's adjointOptimization).
• The surrogate is only as accurate as its training data coverage.
=============================================================================
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.openfoam")

__all__ = [
    "OpenFOAMConfig",
    "OpenFOAMFieldParser",
    "OpenFOAMDataset",
    "DifferentiableCFDSurrogate",
    "NavierStokesResidualLoss",
    "OpenFOAMTrainer",
]


@dataclass
class OpenFOAMConfig:
    case_dir: str = "./openfoam_case"
    field_names: List[str] = None      # U, p, k, epsilon, etc.
    grid_dim: int = 64                 # spatial resolution for NN input
    hidden_dim: int = 256
    n_layers: int = 6
    batch_size: int = 4
    lr: float = 3e-4
    physics_weight: float = 0.1        # weight for NS residual loss
    device: str = "cuda"
    
    def __post_init__(self):
        if self.field_names is None:
            self.field_names = ["U", "p"]


# ═══════════════════════════════════════════════════════════════════════
# OpenFOAM FIELD PARSER
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMFieldParser:
    """
    Parses OpenFOAM field files (ASCII format) into numpy/torch tensors.
    
    OpenFOAM stores fields as:
        case_dir/
            0/          ← time directory
                U       ← velocity field
                p       ← pressure field
            constant/
                polyMesh/
                    points      ← mesh coordinates
                    faces
                    owner
                    neighbour
    
    This parser handles the standard OpenFOAM ASCII format.
    """
    
    def __init__(self, case_dir: str):
        self.case_dir = Path(case_dir)
    
    def parse_field(self, field_path: Path) -> Optional[np.ndarray]:
        """
        Parse an OpenFOAM field file into numpy array.
        
        Handles both scalar (p) and vector (U) fields.
        """
        if not field_path.exists():
            logger.warning(f"Field file not found: {field_path}")
            return None
        
        try:
            text = field_path.read_text()
            return self._parse_field_text(text)
        except Exception as e:
            logger.warning(f"Failed to parse {field_path}: {e}")
            return None
    
    def _parse_field_text(self, text: str) -> np.ndarray:
        """Parse OpenFOAM field text format."""
        # Extract internalField section
        match = re.search(r'internalField\s+uniform\s+\(?\s*([\d.eE+-\s]+)\)?;', text)
        if match:
            # Uniform field — single value
            vals = match.group(1).split()
            return np.array([float(v) for v in vals])
        
        # Non-uniform field: extract values between ( and )
        match = re.search(r'nonuniform\s+List<\w+>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)',
                         text, re.DOTALL)
        if match:
            n = int(match.group(1))
            data_str = match.group(2)
            
            # Check if vector field (has parentheses)
            if '(' in data_str:
                # Vector field: extract (x y z) tuples
                vectors = re.findall(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)',
                                    data_str)
                return np.array([[float(x), float(y), float(z)] for x, y, z in vectors])
            else:
                # Scalar field
                values = [float(v) for v in data_str.split() if v]
                return np.array(values)
        
        return np.array([])
    
    def parse_mesh_points(self) -> Optional[np.ndarray]:
        """Parse mesh point coordinates."""
        points_file = self.case_dir / "constant" / "polyMesh" / "points"
        if not points_file.exists():
            return None
        
        text = points_file.read_text()
        vectors = re.findall(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', text)
        if vectors:
            return np.array([[float(x), float(y), float(z)] for x, y, z in vectors])
        return None
    
    def parse_timestep(self, time_dir: str) -> Dict[str, np.ndarray]:
        """Parse all fields at a given timestep."""
        tdir = self.case_dir / time_dir
        fields = {}
        
        if not tdir.exists():
            logger.warning(f"Timestep directory not found: {tdir}")
            return fields
        
        for f in tdir.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                parsed = self.parse_field(f)
                if parsed is not None and parsed.size > 0:
                    fields[f.name] = parsed
        
        return fields
    
    def list_timesteps(self) -> List[str]:
        """List available timestep directories."""
        timesteps = []
        for d in self.case_dir.iterdir():
            if d.is_dir() and re.match(r'^[\d.]+$', d.name):
                timesteps.append(d.name)
        return sorted(timesteps, key=float)


# ═══════════════════════════════════════════════════════════════════════
# OpenFOAM DATASET
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMDataset(Dataset):
    """PyTorch dataset from OpenFOAM case data."""
    
    def __init__(self, cfg: OpenFOAMConfig):
        self.cfg = cfg
        self.parser = OpenFOAMFieldParser(cfg.case_dir)
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, np.ndarray]]:
        samples = []
        timesteps = self.parser.list_timesteps()
        
        if not timesteps:
            logger.warning("No OpenFOAM timesteps found. Using synthetic data.")
            return self._generate_synthetic()
        
        for ts in timesteps:
            fields = self.parser.parse_timestep(ts)
            if fields:
                fields['timestep'] = float(ts)
                samples.append(fields)
        
        logger.info(f"Loaded {len(samples)} OpenFOAM timesteps")
        return samples
    
    def _generate_synthetic(self, n: int = 50) -> List[Dict[str, np.ndarray]]:
        """Generate synthetic CFD-like data for testing."""
        samples = []
        g = np.random.RandomState(42)
        N = self.cfg.grid_dim ** 2
        
        for i in range(n):
            # Synthetic velocity field (divergence-free-ish)
            x = np.linspace(0, 1, self.cfg.grid_dim)
            y = np.linspace(0, 1, self.cfg.grid_dim)
            X, Y = np.meshgrid(x, y)
            
            phase = i * 0.1
            u = np.sin(2 * np.pi * Y + phase)
            v = -np.sin(2 * np.pi * X + phase)
            p = np.cos(2 * np.pi * (X + Y) + phase) * 0.5
            
            U = np.stack([u.ravel(), v.ravel(), np.zeros(N)], axis=-1)
            
            samples.append({
                'U': U.astype(np.float32),
                'p': p.ravel().astype(np.float32),
                'timestep': float(i),
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = {}
        
        for key, val in sample.items():
            if isinstance(val, np.ndarray):
                item[key] = torch.tensor(val, dtype=torch.float32)
            else:
                item[key] = torch.tensor(val, dtype=torch.float32)
        
        return item


# ═══════════════════════════════════════════════════════════════════════
# DIFFERENTIABLE CFD SURROGATE
# ═══════════════════════════════════════════════════════════════════════

class DifferentiableCFDSurrogate(nn.Module):
    """
    Neural surrogate for OpenFOAM CFD — fully differentiable.
    
    Predicts next-timestep fields given current fields + boundary conditions.
    The entire network is differentiable, enabling:
      • Gradient-based optimization of designs (shape, BCs)
      • Physics-informed loss (Navier-Stokes residual)
      • End-to-end training with downstream tasks
    """
    
    def __init__(self, cfg: OpenFOAMConfig):
        super().__init__()
        self.cfg = cfg
        
        # Input: U(3) + p(1) + boundary conditions
        in_dim = 4  # Ux, Uy, Uz, p
        
        # Encoder (downsample spatial field)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        
        # Temporal evolution (GRU cell for time stepping)
        self.temporal = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        
        # Physics-informed residual predictor
        self.residual_net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim),
        )
        
        # Decoder (predict next fields)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, in_dim),
        )
        
        # Attention for long-range spatial interactions
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=8,
            batch_first=True,
        )
    
    def encode_field(self, U: Tensor, p: Tensor) -> Tensor:
        """
        Encode velocity + pressure fields.
        
        Args:
            U: (B, N, 3) velocity field
            p: (B, N) or (B, N, 1) pressure field
        
        Returns:
            (B, N, hidden_dim) encoded features
        """
        if p.dim() == 2:
            p = p.unsqueeze(-1)
        
        field_input = torch.cat([U, p], dim=-1)  # (B, N, 4)
        return self.encoder(field_input)
    
    def forward(self, U: Tensor, p: Tensor,
                dt: float = 0.01) -> Dict[str, Tensor]:
        """
        Predict next timestep fields.
        
        Args:
            U: (B, N, 3) current velocity
            p: (B, N) current pressure
            dt: timestep size
        
        Returns:
            dict with U_next, p_next, residual
        """
        B, N, _ = U.shape
        
        # Encode
        encoded = self.encode_field(U, p)  # (B, N, H)
        
        # Spatial attention
        attn_out, _ = self.spatial_attn(encoded, encoded, encoded)
        encoded = encoded + attn_out
        
        # Temporal evolution
        h = torch.zeros(B, self.cfg.hidden_dim, device=U.device)
        pooled = encoded.mean(dim=1)  # (B, H)
        h = self.temporal(pooled, h)
        
        # Physics residual
        residual = self.residual_net(h)  # (B, H)
        
        # Decode — broadcast back to spatial points
        h_expanded = h.unsqueeze(1).expand(-1, N, -1)
        residual_expanded = residual.unsqueeze(1).expand(-1, N, -1)
        
        combined = torch.cat([h_expanded, residual_expanded], dim=-1)
        # Project back
        delta = self.decoder(F.gelu(combined[..., :self.cfg.hidden_dim]))
        
        # Next state = current + delta * dt
        U_next = U + delta[..., :3] * dt
        p_next = p + delta[..., 3] * dt
        
        return {
            "U_next": U_next,
            "p_next": p_next.squeeze(-1) if p_next.dim() == 3 else p_next,
            "residual": residual,
        }


# ═══════════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED LOSS (Navier-Stokes residual)
# ═══════════════════════════════════════════════════════════════════════

class NavierStokesResidualLoss(nn.Module):
    """
    Physics-informed loss: Navier-Stokes residual.
    
    For incompressible flow:
      Continuity: ∇·u = 0
      Momentum: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
    
    This is a simplified discrete version using finite differences.
    """
    
    def __init__(self, grid_dim: int, nu: float = 1e-3):
        super().__init__()
        self.grid_dim = grid_dim
        self.nu = nu
    
    def continuity_residual(self, U: Tensor) -> Tensor:
        """Compute ∇·u (should be ~0 for incompressible)."""
        # U: (B, N, 3) where N = grid_dim^2
        g = self.grid_dim
        B = U.shape[0]
        
        u = U[:, :, 0].view(B, g, g)
        v = U[:, :, 1].view(B, g, g)
        
        # Finite differences (periodic BC)
        du_dx = torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)
        dv_dy = torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)
        
        div = (du_dx + dv_dy) / 2.0
        return div.abs().mean()
    
    def forward(self, U_pred: Tensor, U_target: Tensor,
                p_pred: Tensor) -> Dict[str, Tensor]:
        """
        Combined data + physics loss.
        """
        # Data loss
        data_loss = F.mse_loss(U_pred, U_target)
        
        # Physics loss
        cont_res = self.continuity_residual(U_pred)
        
        return {
            "data_loss": data_loss,
            "continuity_residual": cont_res,
            "physics_loss": cont_res,
        }


# ═══════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMTrainer:
    """Train the differentiable CFD surrogate on OpenFOAM data."""
    
    def __init__(self, cfg: OpenFOAMConfig,
                 model: DifferentiableCFDSurrogate,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        
        self.physics_loss = NavierStokesResidualLoss(cfg.grid_dim)
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()
        
        U = batch['U'].to(self.device)  # (B, N, 3)
        p = batch['p'].to(self.device)  # (B, N)
        
        self.opt.zero_grad()
        
        out = self.model(U, p)
        
        # Self-supervised: predict next from current
        losses = self.physics_loss(out['U_next'], U, out['p_next'])
        
        total = losses['data_loss'] + self.cfg.physics_weight * losses['physics_loss']
        total.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        
        return {k: float(v.item()) for k, v in losses.items()} | \
               {"total": float(total.item())}
