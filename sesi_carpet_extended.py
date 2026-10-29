"""
=============================================================================
SESI CARPET OPERATOR — EXTENDED TO ORDER 22 (A^4 → A^11)
Full polyharmonic hierarchy with spectral Sobolev scale
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS EXTENDS
-----------------
From Carpet-19 Prop 4 + Cor 4:
    A^k = ∂*(∂∂*)^(k-1)∂ = (-1)^k ∆^k

This module implements:
  • A^4 (order 8)   — 4th-order polyharmonic
  • A^5 (order 10)  — 5th-order polyharmonic
  • A^6 (order 12)  — 6th-order polyharmonic
  • A^7 (order 14)  — 7th-order polyharmonic
  • A^8 (order 16)  — 8th-order polyharmonic
  • A^9 (order 18)  — 9th-order polyharmonic
  • A^10 (order 20) — 10th-order polyharmonic
  • A^11 (order 22) — 11th-order polyharmonic

APPLICATIONS
------------
  • High-order PDE solvers (e.g., Cahn-Hilliard, phase-field)
  • Spectral analysis of fractal structures
  • Machine learning feature extraction (multi-scale)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Higher-order operators are numerically challenging (conditioning).
• Spectral Sobolev scale W^s embeds into Hölder space (Carpet-19 Thm 4(f)).
• No lower-order terms (Cor 4): A^(2k) = (-1)^k ∆^k identically.
• Must validate against known solutions before deployment.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("sesi.carpet_extended")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sesi_carpet_operator import CarpetGraph, CarpetLaplacian
    HAS_CARPET = True
except ImportError:
    HAS_CARPET = False

__all__ = [
    "ExtendedCarpetConfig",
    "ExtendedCarpetOperator",
    "SpectralSobolevAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtendedCarpetConfig:
    """Configuration for extended Carpet operator."""
    carpet_level: int = 3
    rho: float = 1.25  # resistance scaling

    # Operator order
    min_order: int = 8   # A^4
    max_order: int = 22  # A^11

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [P] EXTENDED CARPET OPERATOR (A^4 → A^11)
# ═══════════════════════════════════════════════════════════════════════

class ExtendedCarpetOperator(nn.Module):
    """
    [P] Extended Carpet operator supporting A^k for k=4,...,11.

    Uses spectral decomposition:
        A^k u = Σ λ_j^k ⟨u, φ_j⟩ φ_j

    where (λ_j, φ_j) are eigenpairs of the normalized Carpet Laplacian.
    """

    def __init__(self, cfg: ExtendedCarpetConfig):
        super().__init__()
        self.cfg = cfg

        if not HAS_CARPET:
            raise RuntimeError("sesi_carpet_operator required")

        # Base Carpet Laplacian
        self.carpet = CarpetLaplacian(cfg.carpet_level, cfg.rho)
        self.n_vertices = self.carpet.graph.n_vertices

        # Precompute spectrum
        self.evals, self.evecs = self.carpet.spectrum()

    def apply_power(self, u: torch.Tensor, k: int) -> torch.Tensor:
        """
        [P] Apply A^k to signal u.

        Per Carpet-19 Cor 4: A^k = (-1)^k ∆^k (no lower-order terms).

        Args:
            u: (n_vertices,) signal on carpet
            k: power (4 ≤ k ≤ 11)

        Returns:
            A^k u
        """
        if k < 4 or k > 11:
            raise ValueError(f"Order must be in [4, 11]; got {k}")

        # Spectral decomposition
        coeffs = self.evecs.T @ u
        return self.evecs @ (self.evals.pow(k) * coeffs)

    def apply_all_orders(self, u: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Apply A^k for all k in [4, 11].

        Returns dict mapping k → A^k u.
        """
        results = {}
        for k in range(4, 12):
            results[k] = self.apply_power(u, k)
        return results

    def spectral_sobolev_norm(self, u: torch.Tensor, s: float) -> float:
        """
        [P] Compute spectral Sobolev norm ||u||_{W^s}.

        Per Carpet-19 Thm 4: W^s = D((I + A)^{s/2}) with norm
            ||u||_{W^s}^2 = Σ (1 + λ_j)^s |⟨u, φ_j⟩|^2
        """
        coeffs = self.evecs.T @ u
        weights = (1 + self.evals).pow(s)
        return float(torch.sqrt(torch.sum(weights * coeffs.pow(2))).item())


# ═══════════════════════════════════════════════════════════════════════
# [D] SPECTRAL SOBOLEV ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class SpectralSobolevAnalyzer:
    """
    [D] Analyze signal regularity across spectral Sobolev scale W^s.

    Computes:
      • Sobolev norms ||u||_{W^s} for s = 0, 1, 2, ..., 11
      • Regularity index (decay rate of Sobolev norms)
      • Hölder embedding (Carpet-19 Thm 4(f): W^s ↪→ C^{0,β/2})
    """

    def __init__(self, operator: ExtendedCarpetOperator):
        self.operator = operator

    def analyze(self, u: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze signal regularity.

        Returns dict with Sobolev norms, regularity index, etc.
        """
        # Compute Sobolev norms for s = 0, 1, ..., 11
        norms = []
        for s in range(12):
            norm = self.operator.spectral_sobolev_norm(u, float(s))
            norms.append(norm)

        # Regularity index: decay rate of norms
        # If ||u||_{W^s} grows slowly, signal is smooth
        if norms[0] > 0:
            growth_rates = [norms[s] / norms[s-1] if norms[s-1] > 0 else 0
                           for s in range(1, 12)]
            avg_growth = np.mean(growth_rates)
        else:
            avg_growth = 0.0

        # Hölder embedding: W^1 ↪→ C^{0,β/2}
        # β = log(ρ)/log(3) ≈ 0.322 for ρ ≈ 1.25
        beta = np.log(self.operator.cfg.rho) / np.log(3)
        holder_exponent = beta / 2

        return {
            "sobolev_norms": norms,
            "regularity_index": avg_growth,
            "holder_exponent": holder_exponent,
            "is_smooth": avg_growth < 2.0,  # heuristic threshold
            "note": ("Lower regularity index → smoother signal. "
                     "Hölder embedding guarantees continuity for W^1."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Extended Carpet Operator (A^4 → A^11) — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available  : {HAS_TORCH}")
    print(f"  Carpet module      : {HAS_CARPET}")

    if not (HAS_TORCH and HAS_CARPET):
        print("  [SKIP] Required modules not available")
        return

    # Config
    cfg = ExtendedCarpetConfig(carpet_level=2, rho=1.25)
    operator = ExtendedCarpetOperator(cfg)

    print(f"\n  ── Carpet Operator ──")
    print(f"    Carpet level: {cfg.carpet_level}")
    print(f"    Vertices: {operator.n_vertices}")
    print(f"    Supported orders: A^4 to A^11 (order 8 to 22)")

    # Test signal
    u = torch.randn(operator.n_vertices)

    # Apply all orders
    print(f"\n  ── Applying A^k for k=4,...,11 ──")
    results = operator.apply_all_orders(u)
    for k, ak_u in results.items():
        norm = torch.norm(ak_u).item()
        print(f"    A^{k:2d} (order {2*k:2d}): ||A^{k}u|| = {norm:.6e}")

    # Spectral Sobolev analysis
    print(f"\n  ── Spectral Sobolev Analysis ──")
    analyzer = SpectralSobolevAnalyzer(operator)
    analysis = analyzer.analyze(u)

    print(f"    Sobolev norms (s=0,...,11):")
    for s, norm in enumerate(analysis["sobolev_norms"]):
        print(f"      ||u||_W^{s:2d} = {norm:.6e}")

    print(f"    Regularity index: {analysis['regularity_index']:.3f}")
    print(f"    Hölder exponent: {analysis['holder_exponent']:.3f}")
    print(f"    Is smooth: {analysis['is_smooth']}")
    print(f"    {analysis['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE (per Carpet-19)")
    print("  • A^k = (-1)^k ∆^k (no lower-order terms, Cor 4)")
    print("  • W^s ↪→ C^{0,β/2} for s ≥ 1 (Thm 4(f))")
    print("  • Higher-order operators are numerically challenging")
    print("  • Must validate against known solutions")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
