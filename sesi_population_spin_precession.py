"""
=============================================================================
SESI SPIN-PRECESION POPULATION MODEL
Aligned/isotropic spin-tilt mixture + effective precession parameter χ_p
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS ADDS TO sesi_population_full.py
-----------------------------------------
The full population model included spin MAGNITUDE and a simple tilt
distribution. This module adds the PRECESSION-astrophysics layer used in
LVK population analyses:

  [S1] Spin-tilt mixture (DefaultSpinTilt, LVK standard):
       p(cos t | ξ_t) = ξ_t · Uniform(-1,1)
                      + (1-ξ_t) · TruncGaussian(cos t; μ=1, σ_σt, [-1,1])
       ξ_t = fraction with ISOTROPIC tilts (dynamical-formation signature)
       (1-ξ_t) = fraction ALIGNED near cos t = 1 (isolated binary signature)

  [S2] Effective precession parameter χ_p (LVK definition):
       χ_p = max[ χ_1 sin t_1,  (4q+3)/(3q+4) · q · χ_2 sin t_2 ]
       This single number captures the dominant precession effects in the
       waveform and is what PE actually constrains.

  [S3] Hierarchical inference of ξ_t from an observed χ_p / tilt sample.

ASTROPHYSICAL MEANING [C: LVK population papers]
------------------------------------------------
• ξ_t ≈ 1  → spins isotropically oriented → DYNAMICAL assembly (globular
  clusters, nuclear star clusters).
• ξ_t ≈ 0  → spins aligned with orbital angular momentum → ISOLATED binary
  evolution through common-envelope.
Therefore ξ_t is a formation-channel diagnostic, not just a nuisance.

LABELS: [C] LVK population framework  [D] derived implementation  [O] open

HONEST SCOPE
------------
• χ_p is a PHENOMENOLOGICAL summary parameter; it is NOT a full solution
  of the spin-precession equations of motion.
• The tilt mixture is a model ASSUMPTION; the true tilt distribution could
  be more complex (e.g., spin-orbit misalignment from supernova kicks).
• Formation-channel inference from ξ_t is probabilistic, not deterministic:
  a mixed population can mimic either channel.
• With few events, ξ_t is weakly constrained; report the credible interval.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.spin_prec_pop")

__all__ = ["SpinTiltConfig", "SpinTiltMixtureModel", "EffectivePrecessionParameter",
           "PrecessionPopulationInference"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SpinTiltConfig:
    # Tilt mixture hyperparameters
    xi_t: float = 0.5          # fraction with isotropic tilts
    sigma_t: float = 1.0       # width of the aligned Gaussian component
    # Magnitude model (reuse power-law from full population)
    xi_spin: float = 1.0       # p(χ) ∝ χ^{ξ-1}
    # Grid
    n_quad: int = 2000
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# [S1] SPIN-TILT MIXTURE MODEL
# ═══════════════════════════════════════════════════════════════════════

class SpinTiltMixtureModel:
    """
    [C/D] The LVK DefaultSpinTilt mixture on cos(tilt) ∈ [-1, 1]:
        p(cos t | ξ_t) = ξ_t/2 + (1-ξ_t) · G(cos t; μ=1, σ)
    where G is a Gaussian truncated to [-1,1] and renormalized.

    ξ_t → 1: isotropic (dynamical); ξ_t → 0: aligned (isolated).
    """

    def __init__(self, cfg: SpinTiltConfig):
        self.cfg = cfg
        # Precompute the truncated-Gaussian normalization on a grid
        self._grid = np.linspace(-1.0, 1.0, cfg.n_quad)
        self._gauss_unnorm = np.exp(-0.5 * ((self._grid - 1.0) / cfg.sigma_t) ** 2)
        self._gauss_norm = np.trapz(self._gauss_unnorm, self._grid)

    def prob(self, cos_t: np.ndarray, xi_t: Optional[float] = None) -> np.ndarray:
        """Normalized mixture density on cos_t."""
        xi = self.cfg.xi_t if xi_t is None else xi_t
        cos_t = np.asarray(cos_t, dtype=float)
        inrange = (cos_t >= -1.0) & (cos_t <= 1.0)

        iso = 0.5 * np.ones_like(cos_t)  # uniform on [-1,1]
        gauss = np.exp(-0.5 * ((cos_t - 1.0) / self.cfg.sigma_t) ** 2) \
                / max(self._gauss_norm, 1e-30)

        p = np.where(inrange, xi * iso + (1.0 - xi) * gauss, 0.0)
        return p

    def log_prob(self, cos_t: np.ndarray, xi_t: Optional[float] = None) -> np.ndarray:
        return np.log(np.maximum(self.prob(cos_t, xi_t), 1e-300))

    def sample(self, n: int, xi_t: Optional[float] = None,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw n samples of cos(tilt)."""
        xi = self.cfg.xi_t if xi_t is None else xi_t
        rng = rng or np.random.default_rng(self.cfg.seed)
        # Choose component
        is_iso = rng.uniform(0, 1, n) < xi
        cos_t = np.empty(n)
        # Isotropic component: uniform on [-1,1]
        cos_t[is_iso] = rng.uniform(-1, 1, is_iso.sum())
        # Aligned component: truncated Gaussian at μ=1
        n_aligned = (~is_iso).sum()
        if n_aligned > 0:
            draws = rng.normal(1.0, self.cfg.sigma_t, n_aligned)
            cos_t[~is_iso] = np.clip(draws, -1.0, 1.0)
        return cos_t


# ═══════════════════════════════════════════════════════════════════════
# [S2] EFFECTIVE PRECESSION PARAMETER χ_p
# ═══════════════════════════════════════════════════════════════════════

class EffectivePrecessionParameter:
    """
    [C] The LVK effective precession parameter:
        χ_p = max[ χ_1 sin t_1,  (4q+3)/(3q+4) · q · χ_2 sin t_2 ]
    with q = m_2/m_1 ≤ 1 (convention m_1 ≥ m_2).

    χ_p ∈ [0, 1]; large χ_p → strong precession. It summarizes the
    in-plane spin that drives precession in the waveform.
    """

    @staticmethod
    def compute(chi1: np.ndarray, cos_t1: np.ndarray,
                chi2: np.ndarray, cos_t2: np.ndarray,
                q: np.ndarray) -> np.ndarray:
        """χ_p for arrays of sources (q ≤ 1)."""
        sin_t1 = np.sqrt(np.maximum(1.0 - cos_t1 ** 2, 0.0))
        sin_t2 = np.sqrt(np.maximum(1.0 - cos_t2 ** 2, 0.0))
        term1 = chi1 * sin_t1
        factor = (4.0 * q + 3.0) / (3.0 * q + 4.0) * q
        term2 = factor * chi2 * sin_t2
        return np.maximum(term1, term2)

    @staticmethod
    def is_precessing(chi_p: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """Heuristic: sources with χ_p above threshold are 'precessing'."""
        return chi_p > threshold


# ═══════════════════════════════════════════════════════════════════════
# [S3] HIERARCHICAL INFERENCE OF ξ_t
# ═══════════════════════════════════════════════════════════════════════

class PrecessionPopulationInference:
    """
    [D] Infer the isotropic fraction ξ_t from an observed sample of
    cos(tilt) values (or χ_p-derived tilts).

    Uses a simple grid posterior:
        p(ξ_t | {cos t_i}) ∝ π(ξ_t) Π_i p(cos t_i | ξ_t)
    with π(ξ_t) = Uniform(0,1).

    HONEST: for few events the posterior on ξ_t is wide. This returns the
    full posterior, not just a point estimate, and reports a credible
    interval so the uncertainty is never hidden.
    """

    def __init__(self, cfg: SpinTiltConfig):
        self.cfg = cfg
        self.model = SpinTiltMixtureModel(cfg)

    def infer(self, cos_t_obs: np.ndarray, n_grid: int = 200) -> Dict[str, Any]:
        """Grid posterior on ξ_t given observed cos(tilt) samples."""
        xi_grid = np.linspace(0.0, 1.0, n_grid)
        log_lik = np.zeros(n_grid)

        for i, xi in enumerate(xi_grid):
            # log-likelihood of all observations under this ξ_t
            lp = self.model.log_prob(cos_t_obs, xi_t=xi)
            log_lik[i] = float(np.sum(lp))

        # Uniform prior, normalize posterior
        log_lik -= np.max(log_lik)
        post = np.exp(log_lik)
        Z = np.trapz(post, xi_grid)
        if Z <= 0:
            return {"error": "degenerate posterior"}
        post /= Z

        cdf = np.cumsum(post * np.gradient(xi_grid))
        xi_med = float(xi_grid[np.searchsorted(cdf, 0.5)])
        xi_lo = float(xi_grid[np.searchsorted(cdf, 0.05)])
        xi_hi = float(xi_grid[np.searchsorted(cdf, 0.95)])

        # Formation-channel interpretation (probabilistic!)
        if xi_med > 0.66:
            channel = "likely dynamical (isotropic spins dominate)"
        elif xi_med < 0.33:
            channel = "likely isolated (aligned spins dominate)"
        else:
            channel = "mixed / inconclusive formation channel"

        return {
            "xi_grid": xi_grid,
            "posterior": post,
            "xi_median": xi_med,
            "xi_90ci": [xi_lo, xi_hi],
            "n_events": len(cos_t_obs),
            "channel_interpretation": channel,
            "honest_note": (
                "Channel interpretation is PROBABILISTIC; a mixed population "
                "can mimic either channel. Report the 90% credible interval."),
        }

    def simulate_and_infer(self, xi_true: float, n_events: int = 50,
                           rng: Optional[np.random.Generator] = None
                           ) -> Dict[str, Any]:
        """Draw from a known ξ_t and recover it (self-consistency check)."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        cos_t = self.model.sample(n_events, xi_t=xi_true, rng=rng)
        result = self.infer(cos_t)
        result["xi_true"] = xi_true
        result["recovered_within_90ci"] = (
            result.get("xi_90ci", [0, 1])[0] <= xi_true <= result.get("xi_90ci", [0, 1])[1])
        return result


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Spin-Precession Population — Smoke Test")
    print("=" * 72)

    cfg = SpinTiltConfig(xi_t=0.5, sigma_t=1.0)

    # ── [S1] Tilt mixture normalization ──
    model = SpinTiltMixtureModel(cfg)
    grid = np.linspace(-1, 1, 2000)
    norm = np.trapz(model.prob(grid, xi_t=0.5), grid)
    print(f"\n  ── [S1] Tilt mixture ──")
    print(f"    ∫p(cos t) = {norm:.4f}  (expect ~1)")
    print(f"    [{'PASS' if abs(norm-1)<0.02 else 'FAIL'}] normalized")

    # ── [S2] χ_p ──
    chi1 = np.array([0.8, 0.3]); cos_t1 = np.array([0.0, 0.9])
    chi2 = np.array([0.5, 0.2]); cos_t2 = np.array([0.2, 1.0])
    q = np.array([0.8, 0.6])
    chi_p = EffectivePrecessionParameter.compute(chi1, cos_t1, chi2, cos_t2, q)
    print(f"\n  ── [S2] Effective precession χ_p ──")
    print(f"    χ_p = {chi_p}")
    print(f"    precessing (>0.2): {EffectivePrecessionParameter.is_precessing(chi_p)}")

    # ── [S3] Recover ξ_t from simulated data ──
    print(f"\n  ── [S3] Hierarchical ξ_t recovery ──")
    prec = PrecessionPopulationInference(cfg)
    for xi_true in [0.1, 0.5, 0.9]:
        res = prec.simulate_and_infer(xi_true, n_events=80)
        print(f"    ξ_true={xi_true:.1f} → ξ_med={res['xi_median']:.2f} "
              f"90%CI=[{res['xi_90ci'][0]:.2f},{res['xi_90ci'][1]:.2f}] "
              f"within_CI={res['recovered_within_90ci']} "
              f"→ {res['channel_interpretation']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • χ_p is a PHENOMENOLOGICAL summary, not full precession dynamics")
    print("  • Tilt mixture is a model assumption; real distributions may differ")
    print("  • Formation-channel inference is probabilistic, not deterministic")
    print("  • Few events → wide ξ_t credible interval (always report it)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
