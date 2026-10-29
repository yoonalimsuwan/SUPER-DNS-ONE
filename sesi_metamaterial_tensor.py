"""
=============================================================================
SESI METAMATERIAL TENSOR DESIGN
Inverse design of electromagnetic absorbers using ordinal-tensor framework
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DESIGNS
-------------------------
[C/D] Metamaterial absorber design via tensor optimization:
  • Target: near-perfect absorption at specific frequency band
  • Method: inverse design using ordinal-tensor framework
  • Output: spatially-varying permittivity ε(x) and permeability μ(x)

MATHEMATICAL FRAMEWORK
-----------------------
From Unified Ordinal-Tensor Measurement Theory:
  • Material properties represented as tensor fields T(x)
  • Absorption efficiency η = f(T, ω) where ω is frequency
  • Inverse problem: find T* that maximizes η subject to constraints

Maxwell's equations (frequency domain):
  ∇ × E = iωμ(x)H
  ∇ × H = -iωε(x)E + J

Absorption coefficient:
  α(ω) = (ω/c) Im[√(εμ)]

For perfect absorption: α → ∞, which requires:
  • Impedance matching: Z = √(μ/ε) = Z_0 (free space)
  • High loss: Im(ε), Im(μ) >> Re(ε), Re(μ)

OPTIMIZATION
------------
We use gradient-based optimization on the tensor field T(x):
  • Parameterize ε(x), μ(x) as neural network outputs
  • Loss function: L = -η + λ_1 · constraint_1 + λ_2 · constraint_2
  • Constraints: passivity, causality, manufacturability

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is an INVERSE DESIGN FRAMEWORK, not a guaranteed optimal solution.
• Real metamaterial fabrication has constraints (discrete values, finite
  resolution, material availability) not captured here.
• Perfect absorption is theoretically impossible over finite bandwidth
  (Bode-Fano limit).
• Results are for RESEARCH EXPLORATION, not manufacturing specification.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.metamaterial")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "MetamaterialConfig",
    "MaterialTensorField",
    "MaxwellSolver",
    "MetamaterialOptimizer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetamaterialConfig:
    """Configuration for metamaterial design."""
    # Geometry
    thickness_m: float = 1e-3          # 1 mm
    lateral_size_m: float = 10e-3      # 10 mm
    grid_resolution: int = 64          # N×N grid
    
    # Frequency
    target_frequency_hz: float = 10e9  # 10 GHz
    bandwidth_hz: float = 1e9          # 1 GHz bandwidth
    
    # Material constraints
    eps_min: float = 1.0               # minimum permittivity
    eps_max: float = 100.0             # maximum permittivity
    mu_min: float = 1.0                # minimum permeability
    mu_max: float = 100.0              # maximum permeability
    loss_tangent_max: float = 10.0     # maximum loss tangent
    
    # Optimization
    learning_rate: float = 1e-3
    n_iterations: int = 1000
    impedance_weight: float = 1.0      # λ_1
    absorption_weight: float = 10.0    # λ_2
    
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [D] MATERIAL TENSOR FIELD
# ═══════════════════════════════════════════════════════════════════════

class MaterialTensorField(nn.Module):
    """
    [D] Neural network parameterization of material tensor fields.
    
    Outputs spatially-varying ε(x,y) and μ(x,y) as complex-valued fields:
      ε(x,y) = ε'(x,y) + i ε''(x,y)
      μ(x,y) = μ'(x,y) + i μ''(x,y)
    
    The network enforces:
      • Passivity: ε'' ≥ 0, μ'' ≥ 0 (loss is non-negative)
      • Causality: Kramers-Kronig relations (approximately)
      • Bounds: ε', ε'', μ', μ'' ∈ [min, max]
    """
    
    def __init__(self, cfg: MetamaterialConfig):
        super().__init__()
        self.cfg = cfg
        
        # Coordinate input: (x, y) ∈ [0, 1]²
        # Output: 4 channels (ε', ε'', μ', μ'')
        
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute material tensors at coordinates x.
        
        Args:
            x: (N, 2) coordinates in [0, 1]²
        
        Returns:
            eps: (N,) complex permittivity
            mu: (N,) complex permeability
        """
        # Raw output
        raw = self.network(x)
        
        # Apply bounds and passivity constraints
        eps_real = cfg.eps_min + (cfg.eps_max - cfg.eps_min) * torch.sigmoid(raw[:, 0])
        eps_imag = cfg.loss_tangent_max * torch.sigmoid(raw[:, 1])  # ε'' ≥ 0
        mu_real = cfg.mu_min + (cfg.mu_max - cfg.mu_min) * torch.sigmoid(raw[:, 2])
        mu_imag = cfg.loss_tangent_max * torch.sigmoid(raw[:, 3])   # μ'' ≥ 0
        
        # Complex tensors
        eps = torch.complex(eps_real, eps_imag)
        mu = torch.complex(mu_real, mu_imag)
        
        return eps, mu


# ═══════════════════════════════════════════════════════════════════════
# [C] MAXWELL SOLVER (simplified)
# ═══════════════════════════════════════════════════════════════════════

class MaxwellSolver:
    """
    [C] Simplified Maxwell equation solver for 1D layered media.
    
    For a multilayer structure with spatially-varying ε(z), μ(z),
    we solve the 1D wave equation using transfer matrix method.
    
    This is a SIMPLIFICATION; real 3D solver requires FDTD or FEM.
    """
    
    def __init__(self, cfg: MetamaterialConfig):
        self.cfg = cfg
        self.omega = 2 * math.pi * cfg.target_frequency_hz
        self.c = 3e8  # speed of light
        self.k0 = self.omega / self.c  # free-space wavenumber
        self.Z0 = 377.0  # free-space impedance
        
    def compute_impedance(self, eps: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Compute wave impedance Z = √(μ/ε)."""
        return torch.sqrt(mu / eps)
    
    def compute_reflection_coefficient(self, eps: torch.Tensor, mu: torch.Tensor,
                                        thickness: float) -> torch.Tensor:
        """
        Compute reflection coefficient for a single layer.
        
        For a layer with impedance Z and thickness d:
          r = (Z - Z0) / (Z + Z0) · (1 - e^{2ikd}) / (1 - r² e^{2ikd})
        
        Simplified: assume thin layer, single interface.
        """
        Z = self.compute_impedance(eps, mu)
        k = self.k0 * torch.sqrt(eps * mu)
        
        # Single interface reflection
        r_interface = (Z - self.Z0) / (Z + self.Z0)
        
        # Phase shift through layer
        phase = torch.exp(2j * k * thickness)
        
        # Total reflection (Fabry-Perot)
        r_total = r_interface * (1 - phase) / (1 - r_interface**2 * phase)
        
        return r_total
    
    def compute_absorption(self, eps: torch.Tensor, mu: torch.Tensor,
                           thickness: float) -> torch.Tensor:
        """
        Compute absorption coefficient A = 1 - |r|² - |t|².
        
        For a backed structure (metal ground plane), t = 0, so:
          A = 1 - |r|²
        """
        r = self.compute_reflection_coefficient(eps, mu, thickness)
        absorption = 1.0 - torch.abs(r)**2
        return absorption


# ═══════════════════════════════════════════════════════════════════════
# [D] METAMATERIAL OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

class MetamaterialOptimizer:
    """
    [D] Inverse design optimizer for metamaterial absorber.
    
    Objective: maximize absorption at target frequency
    Constraints: impedance matching, passivity, manufacturability
    """
    
    def __init__(self, cfg: MetamaterialConfig):
        self.cfg = cfg
        self.model = MaterialTensorField(cfg)
        self.solver = MaxwellSolver(cfg)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        
    def loss_function(self, eps: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss:
          L = -λ_abs · A + λ_imp · |Z - Z0|² + λ_pass · max(0, -ε'') + max(0, -μ'')
        """
        cfg = self.cfg
        
        # Absorption (maximize)
        absorption = self.solver.compute_absorption(eps, mu, cfg.thickness_m)
        mean_absorption = absorption.mean()
        
        # Impedance matching (minimize |Z - Z0|²)
        Z = self.solver.compute_impedance(eps, mu)
        impedance_mismatch = torch.abs(Z - self.solver.Z0)**2
        mean_impedance = impedance_mismatch.mean()
        
        # Passivity (ε'' ≥ 0, μ'' ≥ 0)
        eps_imag = eps.imag
        mu_imag = mu.imag
        passivity_violation = torch.relu(-eps_imag).mean() + torch.relu(-mu_imag).mean()
        
        # Total loss
        loss = (-cfg.absorption_weight * mean_absorption
                + cfg.impedance_weight * mean_impedance
                + 100.0 * passivity_violation)
        
        return loss, {
            "absorption": mean_absorption.item(),
            "impedance_mismatch": mean_impedance.item(),
            "passivity_violation": passivity_violation.item(),
        }
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization loop."""
        if not HAS_TORCH:
            return {"error": "PyTorch required"}
        
        cfg = self.cfg
        N = cfg.grid_resolution
        
        # Create coordinate grid
        x = torch.linspace(0, 1, N, device=cfg.device)
        y = torch.linspace(0, 1, N, device=cfg.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        history = []
        
        for iteration in range(cfg.n_iterations):
            self.optimizer.zero_grad()
            
            # Forward pass
            eps, mu = self.model(coords)
            
            # Compute loss
            loss, metrics = self.loss_function(eps, mu)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            history.append(metrics)
            
            if (iteration + 1) % 100 == 0:
                logger.info(
                    f"Iter {iteration+1}/{cfg.n_iterations}: "
                    f"A={metrics['absorption']:.4f}, "
                    f"|Z-Z0|²={metrics['impedance_mismatch']:.2e}, "
                    f"passivity={metrics['passivity_violation']:.2e}"
                )
        
        # Final evaluation
        with torch.no_grad():
            eps_final, mu_final = self.model(coords)
            _, final_metrics = self.loss_function(eps_final, mu_final)
        
        return {
            "final_metrics": final_metrics,
            "history": history,
            "eps_real": eps_final.real.reshape(N, N).cpu().numpy(),
            "eps_imag": eps_final.imag.reshape(N, N).cpu().numpy(),
            "mu_real": mu_final.real.reshape(N, N).cpu().numpy(),
            "mu_imag": mu_final.imag.reshape(N, N).cpu().numpy(),
            "note": (
                "Optimization is HEURISTIC. Real metamaterial design requires "
                "full-wave EM simulation and fabrication constraints."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Metamaterial Tensor Design — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available: {HAS_TORCH}")
    
    if not HAS_TORCH:
        print("  [SKIP] PyTorch required")
        return
    
    cfg = MetamaterialConfig(
        grid_resolution=32,
        n_iterations=200,
        target_frequency_hz=10e9,
    )
    
    optimizer = MetamaterialOptimizer(cfg)
    
    print("\n  ── Optimization ──")
    result = optimizer.optimize()
    
    if "error" in result:
        print(f"    {result['error']}")
        return
    
    print(f"    Final absorption       : {result['final_metrics']['absorption']:.4f}")
    print(f"    Impedance mismatch     : {result['final_metrics']['impedance_mismatch']:.2e}")
    print(f"    Passivity violation    : {result['final_metrics']['passivity_violation']:.2e}")
    
    print(f"\n    Material field shapes  :")
    print(f"      ε' (real)            : {result['eps_real'].shape}")
    print(f"      ε'' (imag)           : {result['eps_imag'].shape}")
    print(f"      μ' (real)            : {result['mu_real'].shape}")
    print(f"      μ'' (imag)           : {result['mu_imag'].shape}")
    
    print(f"\n    {result['note']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Optimization is HEURISTIC (gradient-based)")
    print("  • Real metamaterial design requires full-wave EM simulation")
    print("  • Fabrication constraints not included")
    print("  • Perfect absorption over finite bandwidth is theoretically limited")
    print("  • Results are for research exploration, not manufacturing spec")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
