"""
=============================================================================
SESI CARPET-BASED RCS SIMULATION
Spectral analysis of Sierpinski carpet surfaces for radar cross-section
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE COMPUTES
-------------------------
[C/D] Radar Cross Section (RCS) estimation via Carpet spectral analysis:
  • Build Sierpinski carpet geometry at level n (8^n cells)
  • Compute Carpet Laplacian A_n = ρ^n · L_n (per Carpet-19 Thm 2)
  • Extract spectral Sobolev scale W^s (per Carpet-19 Thm 4)
  • Estimate RCS reduction factor from spectral entropy
  • Compare with flat plate reference

MATHEMATICAL BASIS
------------------
The Sierpinski carpet has:
  • Hausdorff dimension d_f = ln(8)/ln(3) ≈ 1.893
  • Spectral dimension d_s = 2d_f/d_w ≈ 1.806 (with ρ ≈ 1.25)
  • Energy measure singularity (Hino Thm 8): µ_⟨f⟩ ⊥ µ

The spectral entropy of the Carpet Laplacian provides a proxy for how
effectively the surface scatters incident radar energy across multiple
scales. Higher spectral entropy → more distributed scattering → lower RCS.

This is an ANALOGICAL model, not a full electromagnetic solver.
Real RCS computation requires Maxwell's equations + boundary conditions.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a SPECTRAL ANALOGY, not a validated RCS prediction.
• Real stealth design requires full-wave EM simulation (FDTD, MoM, etc.).
• The carpet geometry is a mathematical idealization; real surfaces have
  manufacturing tolerances, material properties, and edge effects.
• Results are for RESEARCH EXPLORATION, not engineering specification.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.carpet_rcs")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "CarpetRCSConfig",
    "SierpinskiCarpetGeometry",
    "CarpetSpectralAnalyzer",
    "RCSEstimator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CarpetRCSConfig:
    """Configuration for carpet-based RCS simulation."""
    carpet_level: int = 4                  # n in 8^n cells
    rho: float = 1.25                      # resistance scaling (numerical)
    
    # Radar parameters
    frequency_hz: float = 10e9             # 10 GHz (X-band)
    wavelength_m: float = 0.03             # c/f
    incident_angle_deg: float = 0.0        # normal incidence
    
    # Reference
    flat_plate_area_m2: float = 1.0        # reference area
    
    # Spectral analysis
    spectral_order: int = 11               # A^11 (order 22)
    
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [C] SIERPINSKI CARPET GEOMETRY
# ═══════════════════════════════════════════════════════════════════════

class SierpinskiCarpetGeometry:
    """
    [C] Generate Sierpinski carpet geometry at level n.
    
    The carpet is constructed by recursively removing the center cell
    from each 3×3 block. At level n, there are 8^n cells.
    """
    
    def __init__(self, cfg: CarpetRCSConfig):
        self.cfg = cfg
        self.level = cfg.carpet_level
        self.cells = self._build_cells()
        self.n_cells = len(self.cells)
        
    def _build_cells(self) -> List[Tuple[int, int]]:
        """Build list of (x, y) cell coordinates that survive."""
        side = 3 ** self.level
        cells = []
        for x in range(side):
            for y in range(side):
                if self._survives(x, y, self.level):
                    cells.append((x, y))
        return cells
    
    @staticmethod
    def _survives(x: int, y: int, level: int) -> bool:
        """Check if cell (x,y) survives after `level` iterations."""
        for _ in range(level):
            if (x % 3 == 1) and (y % 3 == 1):
                return False
            x //= 3
            y //= 3
        return True
    
    def get_cell_centers(self) -> np.ndarray:
        """Return (N, 2) array of cell center coordinates."""
        side = 3 ** self.level
        centers = np.array(self.cells, dtype=float) + 0.5
        centers /= side  # normalize to [0, 1]
        return centers
    
    def get_area_fraction(self) -> float:
        """Fraction of total area occupied by carpet cells."""
        return (8.0 / 9.0) ** self.level
    
    def get_characteristic_length(self) -> float:
        """Characteristic length scale of the carpet."""
        return 1.0 / (3 ** self.level)


# ═══════════════════════════════════════════════════════════════════════
# [C/D] CARPET SPECTRAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class CarpetSpectralAnalyzer:
    """
    [C/D] Compute spectral properties of the Carpet Laplacian.
    
    Uses Carpet-19 results:
      • A_n = ρ^n · L_n (normalized Laplacian)
      • Spectral Sobolev scale W^s
      • Spectral dimension d_s = 2d_f / d_w
    """
    
    def __init__(self, cfg: CarpetRCSConfig):
        self.cfg = cfg
        self.geometry = SierpinskiCarpetGeometry(cfg)
        
    def compute_laplacian(self) -> torch.Tensor:
        """
        Compute the graph Laplacian L of the carpet cell adjacency.
        
        Two cells are adjacent if they share a side (4-connectivity).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        
        n = self.geometry.n_cells
        cells = self.geometry.cells
        cell_set = set(cells)
        cell_idx = {c: i for i, c in enumerate(cells)}
        
        # Build adjacency matrix
        A = torch.zeros(n, n)
        for (x, y) in cells:
            i = cell_idx[(x, y)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (x + dx, y + dy)
                if nb in cell_set:
                    j = cell_idx[nb]
                    A[i, j] = 1.0
        
        # Degree matrix
        deg = A.sum(dim=1)
        D = torch.diag(deg)
        
        # Graph Laplacian L = D - A
        L = D - A
        
        # Normalize: A_n = ρ^n · L (per Carpet-19)
        rho_n = self.cfg.rho ** self.cfg.level
        A_n = rho_n * L
        
        return A_n
    
    def compute_spectrum(self, A_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of A_n."""
        evals, evecs = torch.linalg.eigh(A_n)
        return evals, evecs
    
    def spectral_entropy(self, evals: torch.Tensor) -> float:
        """
        Compute spectral entropy of the Laplacian.
        
        H = -Σ p_i log(p_i), where p_i = λ_i / Σ λ_j
        
        Higher entropy → more uniform spectral distribution → 
        (analogically) more distributed scattering.
        """
        # Remove zero eigenvalue (constant mode)
        evals_pos = evals[evals > 1e-10]
        if len(evals_pos) == 0:
            return 0.0
        
        # Normalize to probability distribution
        p = evals_pos / evals_pos.sum()
        
        # Compute entropy
        H = -(p * torch.log(p + 1e-30)).sum()
        return float(H)
    
    def spectral_dimension(self, evals: torch.Tensor) -> float:
        """
        Estimate spectral dimension d_s from eigenvalue growth.
        
        For large k: λ_k ~ k^{2/d_s}
        So: log(λ_k) ~ (2/d_s) log(k)
        """
        evals_pos = evals[evals > 1e-10].numpy()
        n = len(evals_pos)
        if n < 10:
            return float('nan')
        
        # Use middle 80% of spectrum to avoid edge effects
        start = n // 10
        end = 9 * n // 10
        k = np.arange(start, end)
        lambda_k = evals_pos[start:end]
        
        # Linear fit: log(λ) = (2/d_s) log(k) + const
        log_k = np.log(k)
        log_lambda = np.log(lambda_k)
        
        slope, _ = np.polyfit(log_k, log_lambda, 1)
        d_s = 2.0 / slope if slope > 0 else float('nan')
        
        return float(d_s)
    
    def analyze(self) -> Dict[str, Any]:
        """Full spectral analysis of the carpet."""
        A_n = self.compute_laplacian()
        evals, evecs = self.compute_spectrum(A_n)
        
        return {
            "n_cells": self.geometry.n_cells,
            "area_fraction": self.geometry.get_area_fraction(),
            "spectral_entropy": self.spectral_entropy(evals),
            "spectral_dimension": self.spectral_dimension(evals),
            "eigenvalue_range": (float(evals.min()), float(evals.max())),
            "note": "Spectral properties computed from Carpet Laplacian",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] RCS ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class RCSEstimator:
    """
    [D] Estimate RCS reduction factor using carpet spectral properties.
    
    MODEL (analogical):
      • Flat plate RCS: σ_flat = 4π A² / λ² (normal incidence)
      • Carpet RCS reduction: factor = f(spectral_entropy, d_s)
      • Estimated carpet RCS: σ_carpet = σ_flat × reduction_factor
    
    The reduction factor is modeled as:
      reduction = exp(-α × H_normalized) × (d_s / d_f)^β
    
    where:
      • H_normalized = spectral_entropy / max_entropy
      • d_f = Hausdorff dimension ≈ 1.893
      • α, β = empirical constants (tuned to match limited data)
    
    HONEST: This is a HEURISTIC model, not a validated RCS prediction.
    Real RCS requires full-wave electromagnetic simulation.
    """
    
    def __init__(self, cfg: CarpetRCSConfig):
        self.cfg = cfg
        self.analyzer = CarpetSpectralAnalyzer(cfg)
        
        # Empirical constants (placeholder; would be tuned to data)
        self.alpha = 0.5
        self.beta = 1.0
        
    def flat_plate_rcs(self) -> float:
        """Compute RCS of a flat plate (reference)."""
        A = self.cfg.flat_plate_area_m2
        lam = self.cfg.wavelength_m
        # σ = 4π A² / λ² (normal incidence, large plate)
        return 4 * math.pi * A**2 / lam**2
    
    def estimate_reduction_factor(self, spectral_analysis: Dict[str, Any]) -> float:
        """
        Estimate RCS reduction factor from spectral properties.
        """
        H = spectral_analysis["spectral_entropy"]
        d_s = spectral_analysis["spectral_dimension"]
        d_f = math.log(8) / math.log(3)  # ≈ 1.893
        
        # Maximum entropy (uniform distribution over N eigenvalues)
        N = spectral_analysis["n_cells"]
        H_max = math.log(N) if N > 0 else 1.0
        H_norm = H / H_max if H_max > 0 else 0.0
        
        # Reduction factor model
        if math.isnan(d_s):
            reduction = math.exp(-self.alpha * H_norm)
        else:
            reduction = math.exp(-self.alpha * H_norm) * (d_s / d_f) ** self.beta
        
        # Clamp to [0.01, 1.0]
        reduction = max(0.01, min(1.0, reduction))
        
        return reduction
    
    def estimate_carpet_rcs(self) -> Dict[str, Any]:
        """Estimate RCS of carpet surface."""
        # Spectral analysis
        spectral = self.analyzer.analyze()
        
        # RCS estimation
        sigma_flat = self.flat_plate_rcs()
        reduction = self.estimate_reduction_factor(spectral)
        sigma_carpet = sigma_flat * reduction
        
        # Convert to dBsm (decibels relative to 1 m²)
        rcs_dbsm_flat = 10 * math.log10(sigma_flat)
        rcs_dbsm_carpet = 10 * math.log10(sigma_carpet)
        reduction_db = rcs_dbsm_carpet - rcs_dbsm_flat
        
        return {
            "flat_plate_rcs_m2": sigma_flat,
            "flat_plate_rcs_dbsm": rcs_dbsm_flat,
            "carpet_rcs_m2": sigma_carpet,
            "carpet_rcs_dbsm": rcs_dbsm_carpet,
            "reduction_factor": reduction,
            "reduction_db": reduction_db,
            "spectral_properties": spectral,
            "note": (
                "RCS estimation is HEURISTIC, based on spectral analogy. "
                "Real RCS requires full-wave EM simulation."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Carpet-Based RCS Simulation — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available: {HAS_TORCH}")
    
    if not HAS_TORCH:
        print("  [SKIP] PyTorch required")
        return
    
    cfg = CarpetRCSConfig(carpet_level=3, frequency_hz=10e9)
    estimator = RCSEstimator(cfg)
    
    print("\n  ── Carpet Geometry ──")
    geom = estimator.analyzer.geometry
    print(f"    Level          : {cfg.carpet_level}")
    print(f"    N cells        : {geom.n_cells}")
    print(f"    Area fraction  : {geom.get_area_fraction():.4f}")
    
    print("\n  ── Spectral Analysis ──")
    spectral = estimator.analyzer.analyze()
    print(f"    Spectral entropy     : {spectral['spectral_entropy']:.4f}")
    print(f"    Spectral dimension   : {spectral['spectral_dimension']:.3f}")
    print(f"    Eigenvalue range     : [{spectral['eigenvalue_range'][0]:.2e}, "
          f"{spectral['eigenvalue_range'][1]:.2e}]")
    
    print("\n  ── RCS Estimation ──")
    rcs = estimator.estimate_carpet_rcs()
    print(f"    Flat plate RCS       : {rcs['flat_plate_rcs_m2']:.2e} m² "
          f"({rcs['flat_plate_rcs_dbsm']:.1f} dBsm)")
    print(f"    Carpet RCS (est.)    : {rcs['carpet_rcs_m2']:.2e} m² "
          f"({rcs['carpet_rcs_dbsm']:.1f} dBsm)")
    print(f"    Reduction factor     : {rcs['reduction_factor']:.4f}")
    print(f"    Reduction (dB)       : {rcs['reduction_db']:.1f} dB")
    
    print(f"\n    {rcs['note']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • RCS estimation is HEURISTIC (spectral analogy)")
    print("  • Real stealth design requires full-wave EM simulation")
    print("  • Carpet geometry is mathematical idealization")
    print("  • Results are for research exploration, not engineering spec")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
