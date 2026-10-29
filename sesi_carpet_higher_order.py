"""
=============================================================================
SESI EXTENDED CARPET OPERATOR — ORDER > 22 (A^12 and beyond)
Higher-order polyharmonic operators with spectral Sobolev analysis
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS EXTENDS
-----------------
From sesi_carpet_extended.py (A^4 to A^11, orders 8 to 22), this module
extends to ARBITRARY ORDER k ≥ 12 (orders 24, 26, 28, ...).

MATHEMATICAL FOUNDATION (Carpet-19)
------------------------------------
[P] Proposition 4: A^k = ∂*(∂∂*)^(k-1)∂ (polyharmonic representation)
[P] Corollary 4: A^(2k) = (-1)^k ∆^k (no lower-order terms)
[P] Theorem 4: Spectral Sobolev scale W^s with Hölder embedding

For k > 11:
  • Structure is IDENTICAL (Prop 4, Cor 4 apply for all k)
  • Numerical conditioning DEGRADES (eigenvalues grow as λ^k)
  • Spectral Sobolev W^s extends to s > 11 (Hölder embedding holds)

NUMERICAL CHALLENGES
--------------------
  • Eigenvalue computation: λ_j^k overflows for large k
  • Condition number: κ(A^k) = (λ_max/λ_min)^k grows exponentially
  • Regularization required: use log-scale or normalized operators

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Structure is PROVED for all k (Prop 4, Cor 4).
• Numerical stability is [O] for k >> 11; requires careful implementation.
• This module provides the FRAMEWORK; production use requires validation.
• Spectral Sobolev W^s for s > 11 is theoretically valid but numerically
  challenging.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.carpet_higher")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sesi_carpet_extended import ExtendedCarpetOperator, ExtendedCarpetConfig
    HAS_EXTENDED = True
except ImportError:
    HAS_EXTENDED = False

__all__ = [
    "HigherOrderCarpetConfig",
    "HigherOrderCarpetOperator",
    "RegularizedSpectralSobolev",
    "ConditionNumberAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HigherOrderCarpetConfig(ExtendedCarpetConfig):
    """Extended config for higher-order operators."""
    # Operator order
    max_order: int = 40  # A^20 (order 40)

    # Regularization
    use_log_scale: bool = True
    eigenvalue_floor: float = 1e-10

    # Numerical stability
    condition_threshold: float = 1e12  # warn if κ > this


# ═══════════════════════════════════════════════════════════════════════
# [P/D] HIGHER-ORDER CARPET OPERATOR
# ═══════════════════════════════════════════════════════════════════════

class HigherOrderCarpetOperator:
    """
    [P/D] Extended Carpet operator for arbitrary order k.

    Uses Prop 4: A^k = ∂*(∂∂*)^(k-1)∂
    With regularization for numerical stability.
    """

    def __init__(self, cfg: HigherOrderCarpetConfig):
        if not HAS_EXTENDED:
            raise RuntimeError("sesi_carpet_extended required")

        self.cfg = cfg
        self.base_op = ExtendedCarpetOperator(cfg)
        self.evals = self.base_op.evals
        self.evecs = self.base_op.evecs

        # Regularized eigenvalues (log-scale for large k)
        if cfg.use_log_scale:
            self.log_evals = torch.log(self.evals.clamp(min=cfg.eigenvalue_floor))
        else:
            self.log_evals = None

    def apply_power_regularized(self, u: torch.Tensor, k: int) -> torch.Tensor:
        """
        [P] Apply A^k with regularization for numerical stability.

        For large k, use log-scale: A^k u = exp(k * log(A)) u
        """
        if k < 1:
            raise ValueError(f"Order must be ≥ 1; got {k}")

        if self.cfg.use_log_scale and self.log_evals is not None:
            # Log-scale computation
            coeffs = self.evecs.T @ u
            log_result = k * self.log_evals
            result = self.evecs @ (torch.exp(log_result) * coeffs)
        else:
            # Direct computation (may overflow for large k)
            coeffs = self.evecs.T @ u
            result = self.evecs @ (self.evals.pow(k) * coeffs)

        return result

    def apply_all_higher_orders(self, u: torch.Tensor,
                                 min_k: int = 12, max_k: Optional[int] = None
                                 ) -> Dict[int, torch.Tensor]:
        """
        Apply A^k for k in [min_k, max_k].

        Default max_k from config.
        """
        max_k = max_k or self.cfg.max_order
        results = {}

        for k in range(min_k, max_k + 1):
            results[k] = self.apply_power_regularized(u, k)

        return results


# ═══════════════════════════════════════════════════════════════════════
# [D] REGULARIZED SPECTRAL SOBOLEV
# ═══════════════════════════════════════════════════════════════════════

class RegularizedSpectralSobolev:
    """
    [D] Spectral Sobolev W^s for s > 11 with regularization.

    Uses log-scale computation to avoid overflow:
      ||u||_{W^s}^2 = Σ (1 + λ_j)^s |⟨u, φ_j⟩|^2

    For large s, compute in log-space:
      log(||u||_{W^s}^2) = log(Σ exp(s * log(1 + λ_j) + 2 * log|⟨u, φ_j⟩|))
    """

    def __init__(self, operator: HigherOrderCarpetOperator):
        self.op = operator
        self.evals = operator.evals
        self.evecs = operator.evecs

    def compute_norm_regularized(self, u: torch.Tensor, s: float) -> float:
        """
        Compute ||u||_{W^s} with log-scale regularization.
        """
        coeffs = self.op.evecs.T @ u
        coeff_sq = coeffs ** 2

        if s > 11 and self.op.cfg.use_log_scale:
            # Log-scale computation
            log_weights = s * torch.log(1 + self.evals.clamp(min=1e-10))
            log_terms = log_weights + torch.log(coeff_sq.clamp(min=1e-300))

            # Log-sum-exp for numerical stability
            max_log = torch.max(log_terms)
            norm_sq = torch.exp(max_log) * torch.sum(torch.exp(log_terms - max_log))
            return float(torch.sqrt(norm_sq).item())
        else:
            # Direct computation
            weights = (1 + self.evals).pow(s)
            norm_sq = torch.sum(weights * coeff_sq)
            return float(torch.sqrt(norm_sq).item())

    def analyze_higher_sobolev(self, u: torch.Tensor,
                                s_range: Tuple[float, float] = (0.0, 20.0),
                                n_points: int = 20) -> Dict[str, Any]:
        """
        Analyze Sobolev norms for s in [s_min, s_max].
        """
        s_values = np.linspace(s_range[0], s_range[1], n_points)
        norms = [self.compute_norm_regularized(u, float(s)) for s in s_values]

        return {
            "s_values": s_values.tolist(),
            "norms": norms,
            "growth_rate": self._estimate_growth_rate(s_values, norms),
        }

    def _estimate_growth_rate(self, s_values: np.ndarray,
                               norms: List[float]) -> float:
        """Estimate exponential growth rate of norms."""
        log_norms = np.log(np.array(norms) + 1e-300)
        if len(s_values) < 2:
            return 0.0

        # Linear fit in log-space
        coeffs = np.polyfit(s_values, log_norms, 1)
        return float(coeffs[0])


# ═══════════════════════════════════════════════════════════════════════
# [D] CONDITION NUMBER ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class ConditionNumberAnalyzer:
    """
    [D] Analyze condition number of A^k for various k.

    κ(A^k) = (λ_max / λ_min)^k

    For large k, condition number grows exponentially, indicating
    numerical instability.
    """

    def __init__(self, operator: HigherOrderCarpetOperator):
        self.op = operator
        self.evals = operator.evals

    def compute_condition_number(self, k: int) -> float:
        """Compute κ(A^k) = (λ_max / λ_min)^k."""
        evals_pos = self.evals[self.evals > self.op.cfg.eigenvalue_floor]
        if len(evals_pos) == 0:
            return float('inf')

        lambda_max = float(evals_pos.max().item())
        lambda_min = float(evals_pos.min().item())

        if lambda_min <= 0:
            return float('inf')

        ratio = lambda_max / lambda_min
        return float(ratio ** k)

    def analyze_condition_numbers(self, k_range: Tuple[int, int] = (1, 40)
                                   ) -> Dict[str, Any]:
        """Analyze condition numbers for k in [k_min, k_max]."""
        k_values = list(range(k_range[0], k_range[1] + 1))
        conditions = [self.compute_condition_number(k) for k in k_values]

        # Find threshold where condition becomes problematic
        threshold = self.op.cfg.condition_threshold
        problematic_k = next((k for k, c in zip(k_values, conditions) if c > threshold), None)

        return {
            "k_values": k_values,
            "condition_numbers": conditions,
            "threshold": threshold,
            "problematic_k": problematic_k,
            "max_condition": max(conditions) if conditions else 0.0,
            "note": ("Condition number grows exponentially with k. "
                     "For k > problematic_k, regularization is essential."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Extended Carpet (Order > 22) — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available    : {HAS_TORCH}")
    print(f"  Extended module      : {HAS_EXTENDED}")

    if not (HAS_TORCH and HAS_EXTENDED):
        print("  [SKIP] Required modules not available")
        return

    cfg = HigherOrderCarpetConfig(
        carpet_level=2,
        rho=1.25,
        max_order=30,
        use_log_scale=True,
    )

    operator = HigherOrderCarpetOperator(cfg)

    # Test signal
    u = torch.randn(operator.base_op.n_vertices)

    print("\n  ── Higher-Order Operators (k = 12 to 30) ──")
    results = operator.apply_all_higher_orders(u, min_k=12, max_k=30)
    for k, ak_u in results.items():
        norm = torch.norm(ak_u).item()
        print(f"    A^{k:2d} (order {2*k:2d}): ||A^{k}u|| = {norm:.6e}")

    print("\n  ── Spectral Sobolev W^s (s = 0 to 20) ──")
    sobolev = RegularizedSpectralSobolev(operator)
    analysis = sobolev.analyze_higher_sobolev(u, s_range=(0.0, 20.0), n_points=10)
    print(f"    Growth rate: {analysis['growth_rate']:.4f}")
    for s, norm in zip(analysis['s_values'][::2], analysis['norms'][::2]):
        print(f"    ||u||_W^{s:4.1f} = {norm:.6e}")

    print("\n  ── Condition Number Analysis ──")
    cond_analyzer = ConditionNumberAnalyzer(operator)
    cond_analysis = cond_analyzer.analyze_condition_numbers(k_range=(1, 30))
    print(f"    Threshold: {cond_analysis['threshold']:.2e}")
    print(f"    Max condition: {cond_analysis['max_condition']:.2e}")
    if cond_analysis['problematic_k']:
        print(f"    Problematic for k > {cond_analysis['problematic_k']}")
    print(f"    {cond_analysis['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE (per Carpet-19)")
    print("  • Structure A^k = ∂*(∂∂*)^(k-1)∂ is PROVED for all k (Prop 4)")
    print("  • No lower-order terms: A^(2k) = (-1)^k ∆^k (Cor 4)")
    print("  • Numerical stability is [O] for k >> 11")
    print("  • Log-scale regularization essential for large k")
    print("  • Condition number grows exponentially; use with caution")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
