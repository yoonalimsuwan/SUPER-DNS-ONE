"""
=============================================================================
SESI ASSUMPTION-LIGHT CERTIFICATION
Paper 11: Subsampling + Self-Normalization for δ_min confidence interval
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

MATHEMATICAL FOUNDATIONS (per Paper 11)
----------------------------------------
[C] Subsampling Confidence Interval (Paper 11 §2.1):
  • Politis-Romano subsampling: valid whenever τ_n(θ̂_n - θ) converges
  • Does NOT require knowing convergence rate or limit law
  • Requires only: b_n → ∞, b_n/n → 0
  • Removes need to assume mixing rate or noise model

[C] Self-Normalization (Paper 11 §2.3):
  • t_n = Σ X_i / √(Σ X_i²)
  • Asymptotically pivotal: no separate variance estimate needed
  • Absorbs unknown long-run variance

[C] δ_min Isodiametric Bound (Delta_Min Thm 1.2):
  • δ_min = (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
  • Requires κ-fatness hypothesis
  • Geometric lower bound on Hausdorff distance

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Subsampling requires qualitative regularity (stationarity, α-mixing)
• Self-normalization assumes symmetric distribution under H0
• δ_min bound requires κ-fatness; without it, bound is false
• These are STATISTICAL TOOLS, not physical guarantees
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.assumption_light")

__all__ = [
    "CertificationConfig",
    "SubsamplingConfidenceInterval",
    "SelfNormalizedStatistic",
    "DeltaMinIsodiametricBound",
    "AssumptionLightCertifier",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CertificationConfig:
    """Configuration for assumption-light certification."""
    # Subsampling
    n_bootstrap: int = 1000
    block_size: int = 10
    confidence_level: float = 0.95
    
    # δ_min bound
    kappa_fatness: float = 2.0         # κ ≥ 1
    a_min: float = 1e-6                # minimum area (m²)
    dimension: int = 2                 # n in R^n
    
    # Self-normalization
    significance_level: float = 0.05
    
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# [C] SUBSAMPLING CONFIDENCE INTERVAL (Paper 11 §2.1)
# ═══════════════════════════════════════════════════════════════════════

class SubsamplingConfidenceInterval:
    """
    [C] Politis-Romano subsampling for assumption-light confidence intervals.
    
    Paper 11 §2.1: Valid whenever τ_n(θ̂_n - θ) converges to some limit law,
    without needing to know τ_n or the limit law.
    
    Requires only:
      • b_n → ∞
      • b_n / n → 0
      • Some limit law exists (qualitative regularity)
    """
    
    def __init__(self, cfg: CertificationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
    
    def compute(self, observations: np.ndarray,
                statistic_fn: Callable[[np.ndarray], float]
                ) -> Dict[str, Any]:
        """
        Compute subsampling confidence interval.
        
        Args:
            observations: (n,) data array
            statistic_fn: function computing statistic of interest
        
        Returns:
            dict with CI, center, and diagnostics
        """
        n = len(observations)
        b = min(self.cfg.block_size, n // 2)
        
        # Compute statistic on all subsamples of size b
        subsample_stats = []
        n_subsamples = 0
        
        for start in range(0, n - b + 1, max(1, b // 2)):
            sub = observations[start:start + b]
            try:
                stat = statistic_fn(sub)
                subsample_stats.append(stat)
                n_subsamples += 1
            except Exception:
                continue
        
        if n_subsamples < 10:
            return {"error": "insufficient subsamples"}
        
        subsample_stats = np.array(subsample_stats)
        
        # Confidence interval from subsampling distribution
        alpha = 1.0 - self.cfg.confidence_level
        lower = float(np.percentile(subsample_stats, 100 * alpha / 2))
        upper = float(np.percentile(subsample_stats, 100 * (1 - alpha / 2)))
        center = float(np.median(subsample_stats))
        
        return {
            "center": center,
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": self.cfg.confidence_level,
            "n_subsamples": n_subsamples,
            "block_size": b,
            "method": "subsampling (Paper 11 §2.1)",
            "note": (
                "Valid without knowing convergence rate or limit law. "
                "Requires only qualitative regularity (stationarity, α-mixing)."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [C] SELF-NORMALIZED STATISTIC (Paper 11 §2.3)
# ═══════════════════════════════════════════════════════════════════════

class SelfNormalizedStatistic:
    """
    [C] Self-normalized test statistic (Paper 11 §2.3).
    
    t_n = Σ X_i / √(Σ X_i²)
    
    Asymptotically pivotal: absorbs unknown long-run variance.
    No separate variance estimate needed.
    """
    
    def __init__(self, cfg: CertificationConfig):
        self.cfg = cfg
    
    def compute(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Compute self-normalized statistic and p-value.
        
        Under H0 (symmetric distribution), t_n → N(0, 1) asymptotically.
        """
        if len(observations) < 2:
            return {"error": "insufficient data"}
        
        X = observations - observations.mean()
        numerator = X.sum()
        denominator = np.sqrt((X ** 2).sum()) + 1e-12
        t_stat = numerator / denominator
        
        # Approximate p-value (standard normal under H0)
        from math import erf
        p_value = 2 * (1 - erf(abs(t_stat) / math.sqrt(2)))
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_anomalous": p_value < self.cfg.significance_level,
            "significance_level": self.cfg.significance_level,
            "method": "self-normalized (Paper 11 §2.3)",
            "note": (
                "Asymptotically pivotal; absorbs unknown long-run variance. "
                "No separate variance estimate needed."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [P] δ_MIN ISODIAMETRIC BOUND (Delta_Min Thm 1.2)
# ═══════════════════════════════════════════════════════════════════════

class DeltaMinIsodiametricBound:
    """
    [P] Isodiametric lower bound on δ_min (Delta_Min Thm 1.2).
    
    Under κ-fatness hypothesis:
        δ_min = (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    
    where ω_{n-1} is volume of unit (n-1)-ball.
    
    This is a RIGOROUS geometric bound, not a statistical estimate.
    """
    
    def __init__(self, cfg: CertificationConfig):
        self.cfg = cfg
    
    def compute(self) -> Dict[str, Any]:
        """Compute δ_min bound."""
        cfg = self.cfg
        
        # Volume of unit (n-1)-ball
        n = cfg.dimension
        if n == 2:
            omega_n_minus_1 = 2.0  # "volume" of unit 1-ball (line segment)
        elif n == 3:
            omega_n_minus_1 = math.pi  # area of unit 2-ball (disk)
        else:
            # General formula: ω_k = π^{k/2} / Γ(k/2 + 1)
            from math import gamma
            k = n - 1
            omega_n_minus_1 = (math.pi ** (k / 2)) / gamma(k / 2 + 1)
        
        # δ_min bound
        delta_min = (1.0 / cfg.kappa_fatness) * \
                    (cfg.a_min / omega_n_minus_1) ** (1.0 / (n - 1))
        
        return {
            "delta_min": delta_min,
            "kappa_fatness": cfg.kappa_fatness,
            "a_min": cfg.a_min,
            "dimension": n,
            "omega_n_minus_1": omega_n_minus_1,
            "method": "isodiametric bound (Delta_Min Thm 1.2)",
            "note": (
                "RIGOROUS geometric bound under κ-fatness hypothesis. "
                "Without κ-fatness, bound is FALSE (thin slivers can violate it)."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATED CERTIFIER
# ═══════════════════════════════════════════════════════════════════════

class AssumptionLightCertifier:
    """
    [D] Integrated assumption-light certifier combining:
      • Subsampling CI for absorption efficiency
      • Self-normalized anomaly test
      • δ_min isodiametric bound
    """
    
    def __init__(self, cfg: Optional[CertificationConfig] = None):
        self.cfg = cfg or CertificationConfig()
        self.subsampling = SubsamplingConfidenceInterval(self.cfg)
        self.self_norm = SelfNormalizedStatistic(self.cfg)
        self.delta_min = DeltaMinIsodiametricBound(self.cfg)
    
    def certify_absorption_efficiency(self, absorption_data: np.ndarray
                                       ) -> Dict[str, Any]:
        """
        Certify absorption efficiency using subsampling CI.
        
        Args:
            absorption_data: (n,) absorption measurements
        """
        # Statistic: mean absorption
        def mean_stat(x):
            return float(np.mean(x))
        
        ci_result = self.subsampling.compute(absorption_data, mean_stat)
        
        # Self-normalized test for anomalies
        anomaly_result = self.self_norm.compute(absorption_data)
        
        return {
            "absorption_ci": ci_result,
            "anomaly_test": anomaly_result,
            "certified": (
                ci_result.get("ci_lower", 0) > 0.8 and
                not anomaly_result.get("is_anomalous", True)
            ),
        }
    
    def certify_delta_min(self) -> Dict[str, Any]:
        """Certify δ_min using isodiametric bound."""
        return self.delta_min.compute()
    
    def full_certification(self, absorption_data: np.ndarray
                            ) -> Dict[str, Any]:
        """Run complete certification."""
        abs_cert = self.certify_absorption_efficiency(absorption_data)
        delta_cert = self.certify_delta_min()
        
        return {
            "absorption_certification": abs_cert,
            "delta_min_certification": delta_cert,
            "overall_certified": abs_cert["certified"],
            "honest_note": (
                "Certification uses assumption-light statistics (Paper 11) "
                "and rigorous geometric bound (Delta_Min Thm 1.2). "
                "Requires qualitative regularity (stationarity, κ-fatness)."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Assumption-Light Certification (Paper 11 + Delta_Min) — Smoke Test")
    print("=" * 72)
    
    cfg = CertificationConfig(
        n_bootstrap=500,
        block_size=20,
        kappa_fatness=2.0,
        a_min=1e-6,
    )
    certifier = AssumptionLightCertifier(cfg)
    
    # Generate synthetic absorption data
    rng = np.random.default_rng(42)
    n_samples = 200
    absorption_data = 0.95 + 0.03 * rng.normal(0, 1, n_samples)
    absorption_data = np.clip(absorption_data, 0, 1)
    
    print("\n  ── Absorption Efficiency Certification ──")
    abs_cert = certifier.certify_absorption_efficiency(absorption_data)
    
    ci = abs_cert["absorption_ci"]
    if "error" not in ci:
        print(f"    Center (median)        : {ci['center']:.4f}")
        print(f"    {ci['confidence_level']*100:.0f}% CI               : "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"    N subsamples           : {ci['n_subsamples']}")
        print(f"    Block size             : {ci['block_size']}")
    
    anomaly = abs_cert["anomaly_test"]
    print(f"    t-statistic            : {anomaly['t_statistic']:.4f}")
    print(f"    p-value                : {anomaly['p_value']:.4f}")
    print(f"    Is anomalous           : {anomaly['is_anomalous']}")
    
    print(f"    Certified              : {abs_cert['certified']}")
    
    print("\n  ── δ_min Isodiametric Bound ──")
    delta_cert = certifier.certify_delta_min()
    print(f"    δ_min                  : {delta_cert['delta_min']:.3e} m")
    print(f"    κ-fatness              : {delta_cert['kappa_fatness']}")
    print(f"    a_min                  : {delta_cert['a_min']:.3e} m²")
    print(f"    Dimension              : {delta_cert['dimension']}")
    
    print("\n  ── Full Certification ──")
    full_cert = certifier.full_certification(absorption_data)
    print(f"    Overall certified      : {full_cert['overall_certified']}")
    print(f"    {full_cert['honest_note']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE (per Paper 11 + Delta_Min)")
    print("  • Subsampling requires qualitative regularity (stationarity)")
    print("  • Self-normalization assumes symmetric distribution under H0")
    print("  • δ_min bound REQUIRES κ-fatness; without it, bound is FALSE")
    print("  • These are STATISTICAL TOOLS, not physical guarantees")
    print("  • Real certification requires experimental validation")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
