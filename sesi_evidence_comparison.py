"""
=============================================================================
SESI BAYESIAN MODEL EVIDENCE COMPARISON via dynesty (Bayes factors)
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS COMPUTES
------------------
The Bayesian evidence (marginal likelihood) for a model M:

    Z = p(d | M) = ∫ L(θ) π(θ) dθ

via dynesty nested sampling [C: Speagle 2020, MNRAS 493:3132]. The ratio
of evidences between two models M1, M2 is the BAYES FACTOR:

    B_12 = Z_1 / Z_2 ,    ln B_12 = logz_1 - logz_2

interpreted on the Kass & Raftery (1995) scale.

NESTED SAMPLING [C]
-------------------
Transforms the evidence integral into a 1D integral over likelihood
contours X(λ) = ∫_{L>λ} π(θ) dθ:
    Z = ∫ L dX
dynesty estimates this by evolving live points that "climb" in likelihood,
simultaneously yielding posterior samples (weighted) and the evidence.

LABELS: [C] dynesty + Kass-Raftery  [D] derived comparison/report  [O] open

HONEST SCOPE (critical — prior dependence)
------------------------------------------
• The evidence INTEGRATES over the prior, so the Bayes factor depends on
  prior VOLUME. Widening a prior penalizes the model (automatic Occam
  factor). Evidence comparisons are only meaningful when the priors are
  themselves physically justified.
• This compares MODELS given a likelihood; it does not establish that
  either model is physically true (Carpet-19 discipline: a conditional
  statement, not an absolute one).
• Validation uses the analytic Gaussian evidence (closed form) BEFORE
  trusting dynesty on a real GW likelihood.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.evidence")

try:
    import dynesty
    from dynesty.utils import resample_equal
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

__all__ = ["EvidenceConfig", "AnalyticGaussianEvidence",
           "ModelEvidenceCalculator", "BayesFactorComparator"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceConfig:
    nlive: int = 500                 # dynesty live points
    dlogz: float = 0.01             # termination criterion
    bound: str = "multi"            # multi-ellipsoid (multimodal-friendly)
    sample: str = "auto"
    ndim: int = 2
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [P] ANALYTIC GAUSSIAN EVIDENCE (ground truth for validation)
# ═══════════════════════════════════════════════════════════════════════

class AnalyticGaussianEvidence:
    """
    [P] Closed-form evidence for a 1-D conjugate Gaussian model.

    Model:  d ~ N(θ, σ²),   θ ~ N(μ0, τ²)
    Evidence:
        Z = ∫ N(d;θ,σ²) N(θ;μ0,τ²) dθ = N(d; μ0, σ²+τ²)

    This is the reference value the dynesty estimate must reproduce.
    """

    @staticmethod
    def log_evidence(d: float, mu0: float, sigma: float, tau: float) -> float:
        var = sigma**2 + tau**2
        return -0.5 * (math.log(2 * math.pi * var) + (d - mu0)**2 / var)

    @staticmethod
    def log_likelihood(theta: np.ndarray, d: float, sigma: float) -> float:
        return -0.5 * ((d - theta[0]) / sigma) ** 2

    @staticmethod
    def log_prior(theta: np.ndarray, mu0: float, tau: float) -> float:
        return -0.5 * ((theta[0] - mu0) / tau) ** 2


# ═══════════════════════════════════════════════════════════════════════
# [C/D] MODEL EVIDENCE CALCULATOR (dynesty wrapper)
# ═══════════════════════════════════════════════════════════════════════

class ModelEvidenceCalculator:
    """
    [C/D] Run dynesty nested sampling for one model and return the
    log-evidence, its uncertainty, and weighted posterior samples.
    """

    def __init__(self, cfg: EvidenceConfig):
        self.cfg = cfg

    def compute(self,
                log_likelihood: Callable[[np.ndarray], float],
                prior_transform: Callable[[np.ndarray], np.ndarray],
                label: str = "model") -> Dict[str, Any]:
        """
        Args:
            log_likelihood: θ -> log L(θ)
            prior_transform: u ∈ [0,1]^d -> θ in prior support
        """
        if not HAS_DYNESTY:
            return {"error": "dynesty not installed (pip install dynesty)"}

        try:
            sampler = dynesty.NestedSampler(
                log_likelihood, prior_transform,
                ndim=self.cfg.ndim, nlive=self.cfg.nlive,
                bound=self.cfg.bound, sample=self.cfg.sample,
            )
            sampler.run_nested(dlogz=self.cfg.dlogz)
            res = sampler.results

            # weighted posterior samples
            weights = np.exp(res.logwt - np.max(res.logwt))
            weights /= weights.sum()
            post_samples = resample_equal(res.samples, weights)

            return {
                "label": label,
                "logz": float(res.logz[-1]),
                "logzerr": float(res.logzerr[-1]),
                "n_iter": int(res.niter),
                "posterior_samples": post_samples,
                "weights": weights,
            }
        except Exception as e:
            logger.warning(f"dynesty failed for {label}: {e}")
            return {"error": str(e), "label": label}


# ═══════════════════════════════════════════════════════════════════════
# [C/D] BAYES FACTOR COMPARATOR (Kass-Raftery)
# ═══════════════════════════════════════════════════════════════════════

class BayesFactorComparator:
    """
    [C/D] Compare two models via their Bayes factor and interpret on the
    Kass & Raftery (1995) scale.

    ln B_12 :  <0   → favors M2
               0-1  → not worth more than a bare mention
               1-2.5→ positive evidence for M1
               2.5-5→ strong
               >5   → very strong
    """

    def __init__(self, cfg: EvidenceConfig):
        self.cfg = cfg
        self.calc = ModelEvidenceCalculator(cfg)

    def compare(self,
                model1: Dict[str, Callable],
                model2: Dict[str, Callable]) -> Dict[str, Any]:
        """
        model1/model2: {"log_likelihood": fn, "prior_transform": fn, "label": str}
        """
        r1 = self.calc.compute(model1["log_likelihood"],
                               model1["prior_transform"], model1.get("label", "M1"))
        r2 = self.calc.compute(model2["log_likelihood"],
                               model2["prior_transform"], model2.get("label", "M2"))

        if "error" in r1 or "error" in r2:
            return {"error": f"M1={r1.get('error')} M2={r2.get('error')}"}

        lnB = r1["logz"] - r2["logz"]
        # evidence error propagation (add in quadrature)
        lnB_err = math.sqrt(r1["logzerr"]**2 + r2["logzerr"]**2)

        return {
            "model1": r1["label"], "model2": r2["label"],
            "logz1": r1["logz"], "logz1_err": r1["logzerr"],
            "logz2": r2["logz"], "logz2_err": r2["logzerr"],
            "ln_bayes_factor": lnB,
            "ln_bayes_factor_err": lnB_err,
            "bayes_factor": math.exp(lnB) if lnB < 700 else float("inf"),
            "interpretation": self._kass_raftery(lnB),
            "preferred_model": r1["label"] if lnB > 0 else r2["label"],
            "honest_note": (
                "Bayes factor depends on PRIOR VOLUME (Occam factor). "
                "This compares models given their priors; it does not "
                "establish either model is physically true."),
        }

    @staticmethod
    def _kass_raftery(lnB: float) -> str:
        a = abs(lnB)
        favored = "M1" if lnB > 0 else "M2"
        if a < 1.0:
            return f"not worth more than a bare mention (|ln B|<1, favors {favored})"
        elif a < 2.5:
            return f"positive evidence for {favored}"
        elif a < 5.0:
            return f"strong evidence for {favored}"
        else:
            return f"very strong evidence for {favored}"


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST — analytic Gaussian validation
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Bayesian Evidence Comparison — Smoke Test")
    print("=" * 72)
    print(f"  dynesty installed: {HAS_DYNESTY}")

    if not HAS_DYNESTY:
        print("  [SKIP] pip install dynesty")
        return

    # Analytic reference
    d, mu0, sigma, tau = 2.0, 0.0, 1.0, 3.0
    logZ_true = AnalyticGaussianEvidence.log_evidence(d, mu0, sigma, tau)
    print(f"\n  Analytic log-evidence (true): {logZ_true:.4f}")

    # Build the same model for dynesty
    lo, hi = mu0 - 6*tau, mu0 + 6*tau
    def loglike(theta):
        return AnalyticGaussianEvidence.log_likelihood(theta, d, sigma)
    def prior_transform(u):
        return np.array([lo + u[0] * (hi - lo)])

    cfg = EvidenceConfig(ndim=1, nlive=300, dlogz=0.05)
    calc = ModelEvidenceCalculator(cfg)
    res = calc.compute(loglike, prior_transform, label="gaussian_1d")

    if "error" in res:
        print(f"  [FAIL] {res['error']}")
        return

    logZ_est = res["logz"]
    err = abs(logZ_est - logZ_true)
    print(f"  dynesty log-evidence (est)  : {logZ_est:.4f} ± {res['logzerr']:.4f}")
    print(f"  |est - true|                : {err:.4f}")
    print(f"  [{'PASS' if err < 0.2 else 'FAIL'}] dynesty reproduces analytic evidence")

    # Two-model comparison: same data, two likelihood widths
    print(f"\n  ── Two-model comparison ──")
    def loglike_narrow(theta):
        return AnalyticGaussianEvidence.log_likelihood(theta, d, 0.5)
    def loglike_wide(theta):
        return AnalyticGaussianEvidence.log_likelihood(theta, d, 2.0)

    m1 = {"log_likelihood": loglike_narrow, "prior_transform": prior_transform,
          "label": "narrow_sigma"}
    m2 = {"log_likelihood": loglike_wide, "prior_transform": prior_transform,
          "label": "wide_sigma"}

    comp = BayesFactorComparator(cfg)
    result = comp.compare(m1, m2)
    print(f"    ln B (narrow/wide) : {result['ln_bayes_factor']:.3f} "
          f"± {result['ln_bayes_factor_err']:.3f}")
    print(f"    interpretation     : {result['interpretation']}")
    print(f"    {result['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Evidence integrates over the PRIOR → Bayes factor is prior-")
    print("    volume sensitive (automatic Occam factor)")
    print("  • Compares models GIVEN a likelihood; does not prove either true")
    print("  • Validate against analytic evidence before trusting on real GW")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
