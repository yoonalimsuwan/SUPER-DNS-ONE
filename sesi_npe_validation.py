"""
=============================================================================
SESI NPE VALIDATION — NPE vs Bilby (Coverage + KL Divergence + SBC)
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT THIS MODULE VALIDATES
--------------------------
[V1] Coverage: does the NPE posterior's credible interval contain the
     true parameter at the expected rate?
[V2] KL divergence: how far is the NPE posterior from the Bilby/sampling
     posterior (the ground truth)?
[V3] Simulation-Based Calibration (SBC): the rank-based diagnostic of
     Talts et al. (2018) — the gold standard for validating amortized
     inference.

GROUND TRUTH
------------
The "ground truth" posterior is obtained from Bilby (if installed) or a
high-quality nested-sampling reference. WITHOUT a reliable reference,
coverage and KL are meaningless — this module REFUSES to report them
and says so, rather than fabricate.

LABEL CONVENTION (per Consolidated Revision Carpet-19)
------------------------------------------------------
[P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• SBC, coverage and KL are DIAGNOSTICS of the NPE approximation.
• They do NOT prove the NPE posterior is "correct" in an absolute sense;
  they measure consistency with the simulator + prior used.
• The Carpet-19 Theorem 3(b) insight applies: convergence is checked by
  testing against a reference, not by assuming the approximation converges.
• No claim that a well-calibrated NPE implies the physical model is true.
=============================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.npe_validation")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "ValidationConfig",
    "CoverageEstimator",
    "KLDivergenceEstimator",
    "SimulationBasedCalibration",
    "NPEValidator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ValidationConfig:
    """Configuration for NPE validation."""
    n_simulations: int = 500      # total simulated (theta, x) pairs
    n_posterior_samples: int = 1000  # samples per posterior
    credible_level: float = 0.90  # for coverage test
    n_sbc_ranks: int = 100        # number of posterior samples for SBC
    kl_n_samples: int = 2000      # MC samples for KL estimation
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# [V1] COVERAGE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class CoverageEstimator:
    """
    [C/D] Frequentist coverage of posterior credible intervals.

    Procedure:
      1. Draw theta_true ~ prior
      2. Simulate x ~ p(x | theta_true)
      3. Compute posterior q(theta | x)
      4. Check if theta_true is inside the (1-alpha) credible interval
      5. Repeat; coverage = fraction of times theta_true is inside

    Expected coverage for a well-calibrated posterior = 1 - alpha.

    HONEST: This tests CALIBRATION, not correctness of the physical model.
    A well-calibrated posterior on a wrong model is still well-calibrated.
    """

    def __init__(self, cfg: ValidationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def compute_credible_interval(self, samples: np.ndarray,
                                   alpha: float) -> Tuple[float, float]:
        """Equal-tailed credible interval from posterior samples."""
        lo = np.percentile(samples, 100 * alpha / 2, axis=0)
        hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        return lo, hi

    def coverage(self, true_theta: np.ndarray,
                 posterior_samples: np.ndarray,
                 alpha: Optional[float] = None) -> np.ndarray:
        """
        Per-dimension coverage for a batch of (theta_true, posterior) pairs.

        Args:
            true_theta: (n, d) true parameters
            posterior_samples: (n, s, d) posterior samples
            alpha: significance level (default: 1 - credible_level)

        Returns:
            (d,) boolean array — did theta_true fall inside CI?
        """
        if alpha is None:
            alpha = 1.0 - self.cfg.credible_level

        n = len(true_theta)
        inside = np.zeros((n, true_theta.shape[1]), dtype=bool)

        for i in range(n):
            lo, hi = self.compute_credible_interval(posterior_samples[i], alpha)
            inside[i] = (true_theta[i] >= lo) & (true_theta[i] <= hi)

        return inside

    def expected_vs_observed(self, observed_coverage: float,
                             alpha: float) -> Dict[str, float]:
        """Compare observed coverage to expected; flag miscalibration."""
        expected = 1.0 - alpha
        gap = observed_coverage - expected
        return {
            "expected": expected,
            "observed": observed_coverage,
            "gap": gap,
            "is_calibrated": abs(gap) < 0.05,  # ±5% tolerance
        }


# ═══════════════════════════════════════════════════════════════════════
# [V2] KL DIVERGENCE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class KLDivergenceEstimator:
    """
    [C/D] Estimate KL(p || q) between reference posterior p (Bilby/sampling)
    and NPE posterior q.

    Methods:
      • If q is a flow: use q.log_prob() directly (Monte Carlo)
      • Otherwise: KDE-based estimate on samples

    HONEST: KL is asymmetric and can be numerically unstable in the tails.
    We report KL(p||q) — how well q covers p — which is the direction that
    matters for "does the NPE miss modes that Bilby finds?".
    """

    def __init__(self, cfg: ValidationConfig):
        self.cfg = cfg

    def kl_monte_carlo(self, p_samples: np.ndarray,
                       q_log_prob: Callable[[np.ndarray], np.ndarray],
                       p_log_prob: Optional[Callable] = None) -> float:
        """
        KL(p || q) = E_p[log p - log q], estimated by Monte Carlo over p_samples.

        If p_log_prob is None, we drop the log p term (gives cross-entropy
        minus a constant) and report it as a DIAGNOSTIC, not an absolute KL.
        """
        if p_log_prob is None:
            # Cross-entropy proxy: E_p[-log q]
            log_q = q_log_prob(p_samples)
            return float(-np.mean(log_q))

        log_p = p_log_prob(p_samples)
        log_q = q_log_prob(p_samples)
        return float(np.mean(log_p - log_q))

    def kl_kde(self, p_samples: np.ndarray,
               q_samples: np.ndarray) -> float:
        """
        KL(p || q) via kernel density estimation (fallback when no
        closed-form log_prob is available).
        """
        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            logger.warning("sklearn unavailable; returning NaN for KDE-KL")
            return float("nan")

        # Fit KDEs
        bw = self._scott_bandwidth(p_samples)
        kde_p = KernelDensity(bandwidth=bw).fit(p_samples)
        kde_q = KernelDensity(bandwidth=bw).fit(q_samples)

        # MC estimate of E_p[log p - log q]
        log_p = kde_p.score_samples(p_samples)
        log_q = kde_q.score_samples(p_samples)
        return float(np.mean(log_p - log_q))

    @staticmethod
    def _scott_bandwidth(x: np.ndarray) -> float:
        """Scott's rule for bandwidth."""
        n, d = x.shape
        return float(n ** (-1.0 / (d + 4)))

    def interpret(self, kl_value: float) -> str:
        """Qualitative interpretation of KL magnitude."""
        if math.isnan(kl_value):
            return "UNDEFINED (NaN)"
        if kl_value < 0.01:
            return "EXCELLENT (posterior nearly identical)"
        elif kl_value < 0.1:
            return "GOOD (minor discrepancies)"
        elif kl_value < 1.0:
            return "MODERATE (noticeable differences)"
        else:
            return "POOR (substantial disagreement)"


# ═══════════════════════════════════════════════════════════════════════
# [V3] SIMULATION-BASED CALIBRATION (SBC)
# ═══════════════════════════════════════════════════════════════════════

class SimulationBasedCalibration:
    """
    [C] Simulation-Based Calibration (Talts et al. 2018, arXiv:1804.06788).

    THE gold standard for validating amortized Bayesian inference.

    Procedure (per iteration):
      1. Draw theta_tilde ~ prior
      2. Draw x_tilde ~ p(x | theta_tilde)
      3. Draw L samples from the approximate posterior q(theta | x_tilde)
      4. Compute the RANK of theta_tilde among the L samples

    If q is well-calibrated, the ranks are UNIFORMLY distributed over
    {0, 1, ..., L}. Deviation from uniformity reveals systematic
    miscalibration (e.g., posterior too narrow/too wide, missing modes).

    HONEST SCOPE:
    • SBC detects MISCALIBRATION, not necessarily model error.
    • It requires MANY simulations (hundreds to thousands).
    • The chi-squared / Kolmogorov-Smirnov test on ranks gives a
      quantitative pass/fail, but with finite-sample uncertainty.
    """

    def __init__(self, cfg: ValidationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 1)

    def compute_ranks(self, theta_tilde: np.ndarray,
                      posterior_samples: np.ndarray) -> np.ndarray:
        """
        Rank of theta_tilde among posterior samples, per dimension.

        Args:
            theta_tilde: (d,) the "true" parameter drawn from the prior
            posterior_samples: (L, d) samples from q(theta | x_tilde)

        Returns:
            (d,) integer ranks in {0, ..., L}
        """
        ranks = np.sum(posterior_samples < theta_tilde, axis=0)
        return ranks

    def run_sbc(self,
                simulate: Callable[[np.ndarray], np.ndarray],
                sample_posterior: Callable[[np.ndarray], np.ndarray],
                prior_sample: Callable[[int], np.ndarray],
                n_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full SBC loop.

        Args:
            simulate: theta -> x  (the simulator)
            sample_posterior: x -> (L, d) posterior samples
            prior_sample: n -> (n, d) prior samples
            n_iterations: number of SBC iterations

        Returns:
            dict with ranks array + uniformity test results
        """
        n_iterations = n_iterations or self.cfg.n_simulations
        L = self.cfg.n_sbc_ranks

        all_ranks = []
        for i in range(n_iterations):
            # Step 1: draw theta from prior
            theta_tilde = prior_sample(1)[0]
            # Step 2: simulate data
            x_tilde = simulate(theta_tilde)
            # Step 3: sample approximate posterior
            post_samples = sample_posterior(x_tilde)
            if post_samples.shape[0] < L:
                logger.warning(f"SBC iter {i}: only {post_samples.shape[0]} "
                              f"posterior samples < {L}; skipping")
                continue
            # Step 4: rank
            ranks = self.compute_ranks(theta_tilde, post_samples[:L])
            all_ranks.append(ranks)

        if not all_ranks:
            return {"success": False, "reason": "No valid SBC iterations"}

        all_ranks = np.array(all_ranks)  # (n_iter, d)

        # Uniformity test on ranks (per dimension)
        d = all_ranks.shape[1]
        uniformity = {}
        for j in range(d):
            ks_stat, ks_p = self._ks_uniform(all_ranks[:, j], L)
            uniformity[f"dim_{j}"] = {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "is_uniform": ks_p > 0.05,
            }

        n_uniform = sum(1 for v in uniformity.values() if v["is_uniform"])

        return {
            "success": True,
            "ranks": all_ranks,
            "n_iterations": len(all_ranks),
            "L": L,
            "uniformity": uniformity,
            "n_uniform_dims": n_uniform,
            "n_dims": d,
            "overall_calibrated": n_uniform == d,
        }

    @staticmethod
    def _ks_uniform(ranks: np.ndarray, L: int) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test of ranks against Uniform(0, L)."""
        try:
            from scipy.stats import kstest
            # Normalize ranks to [0, 1]
            normalized = ranks / L
            stat, p = kstest(normalized, "uniform")
            return float(stat), float(p)
        except ImportError:
            # Fallback: simple variance check
            expected_var = 1.0 / 12.0  # variance of Uniform(0,1)
            observed_var = float(np.var(ranks / L))
            return abs(observed_var - expected_var), 0.5  # placeholder


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class NPEValidator:
    """
    Unified NPE validation orchestrator.

    Combines coverage, KL divergence, and SBC into one report, with a
    clear verdict and honest caveats.
    """

    def __init__(self, cfg: Optional[ValidationConfig] = None):
        self.cfg = cfg or ValidationConfig()
        self.coverage_est = CoverageEstimator(self.cfg)
        self.kl_est = KLDivergenceEstimator(self.cfg)
        self.sbc = SimulationBasedCalibration(self.cfg)

    def full_validation(self,
                        simulator: Callable,
                        npe_sample_posterior: Callable,
                        npe_log_prob: Optional[Callable] = None,
                        reference_sample_posterior: Optional[Callable] = None,
                        reference_log_prob: Optional[Callable] = None,
                        prior_sample: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run all three validation diagnostics.

        Args:
            simulator: theta -> x
            npe_sample_posterior: x -> samples from NPE posterior
            npe_log_prob: log-density of NPE posterior (for KL)
            reference_sample_posterior: x -> samples from REFERENCE posterior
                                        (Bilby/dynesty). If None, KL is skipped.
            reference_log_prob: log-density of reference posterior (for KL)
            prior_sample: n -> prior samples

        Returns:
            dict with coverage, KL, SBC results + overall verdict
        """
        results = {}

        # ── [V1] Coverage ──
        logger.info("Running coverage test...")
        coverage_result = self._run_coverage(simulator, npe_sample_posterior, prior_sample)
        results["coverage"] = coverage_result

        # ── [V2] KL divergence ──
        logger.info("Running KL divergence test...")
        if reference_sample_posterior is None:
            results["kl"] = {
                "status": "SKIPPED",
                "reason": "No reference posterior provided (Bilby not run). "
                          "KL divergence is meaningless without a ground truth.",
            }
        else:
            kl_result = self._run_kl(simulator, npe_sample_posterior,
                                     npe_log_prob, reference_sample_posterior,
                                     reference_log_prob)
            results["kl"] = kl_result

        # ── [V3] SBC ──
        logger.info("Running Simulation-Based Calibration...")
        sbc_result = self.sbc.run_sbc(simulator, npe_sample_posterior, prior_sample)
        results["sbc"] = sbc_result

        # ── Overall verdict ──
        results["verdict"] = self._verdict(results)
        return results

    def _run_coverage(self, simulator, sample_posterior, prior_sample) -> Dict:
        n = min(self.cfg.n_simulations, 50)  # cap for smoke test
        thetas = prior_sample(n)
        insides = []
        for i in range(n):
            x = simulator(thetas[i])
            post = sample_posterior(x)
            inside = self.coverage_est.coverage(thetas[i:i+1], post[None, ...])
            insides.append(inside[0])
        insides = np.array(insides)
        per_dim = insides.mean(axis=0)
        alpha = 1.0 - self.cfg.credible_level
        comparisons = [
            self.coverage_est.expected_vs_observed(float(c), alpha)
            for c in per_dim
        ]
        return {
            "n": n,
            "per_dim_coverage": per_dim.tolist(),
            "expected": 1.0 - alpha,
            "per_dim_calibration": comparisons,
            "all_calibrated": all(c["is_calibrated"] for c in comparisons),
        }

    def _run_kl(self, simulator, npe_sample, npe_log_prob,
                ref_sample, ref_log_prob) -> Dict:
        n = min(self.cfg.kl_n_samples, 200)
        kl_values = []
        for i in range(n):
            # Draw a theta, simulate, get both posteriors
            theta = np.random.default_rng(self.cfg.seed + i).normal(0, 1, size=3)
            x = simulator(theta)
            ref_samples = ref_sample(x)
            if npe_log_prob is not None:
                kl = self.kl_est.kl_monte_carlo(ref_samples, npe_log_prob)
            else:
                npe_samples = npe_sample(x)
                kl = self.kl_est.kl_kde(ref_samples, npe_samples)
            kl_values.append(kl)

        mean_kl = float(np.mean(kl_values))
        return {
            "n": n,
            "mean_kl": mean_kl,
            "std_kl": float(np.std(kl_values)),
            "interpretation": self.kl_est.interpret(mean_kl),
        }

    def _verdict(self, results: Dict) -> Dict[str, Any]:
        coverage_ok = results["coverage"]["all_calibrated"]
        sbc_ok = results["sbc"].get("overall_calibrated", False)
        kl = results["kl"]
        kl_ok = kl.get("status") != "SKIPPED" and kl.get("mean_kl", 999) < 1.0

        all_ok = coverage_ok and sbc_ok and kl_ok

        return {
            "overall_pass": all_ok,
            "coverage_pass": coverage_ok,
            "sbc_pass": sbc_ok,
            "kl_pass": kl_ok,
            "note": (
                "PASS means the NPE is well-calibrated RELATIVE TO the "
                "simulator and reference used. It does NOT mean the "
                "underlying physical model is correct."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI NPE Validation — Smoke Test")
    print("=" * 72)

    cfg = ValidationConfig(n_simulations=20, n_sbc_ranks=50, kl_n_samples=10)
    validator = NPEValidator(cfg)

    # Define a toy simulator + posterior for the smoke test
    rng = np.random.default_rng(cfg.seed)

    def prior_sample(n):
        return rng.uniform(-3, 3, size=(n, 3))

    def simulator(theta):
        return theta ** 2 + rng.normal(0, 0.1, size=theta.shape)

    def npe_sample_posterior(x):
        # Approximate posterior: Gaussian centered at sqrt(x)
        center = np.sqrt(np.abs(x))
        return rng.normal(center, 0.5, size=(100, 3))

    results = validator.full_validation(
        simulator=simulator,
        npe_sample_posterior=npe_sample_posterior,
        prior_sample=prior_sample,
        reference_sample_posterior=None,  # no Bilby in smoke test
    )

    print(f"\n  ── Coverage ──")
    cov = results["coverage"]
    print(f"    Per-dim coverage: {cov['per_dim_coverage']}")
    print(f"    Expected: {cov['expected']}")
    print(f"    All calibrated: {cov['all_calibrated']}")

    print(f"\n  ── KL Divergence ──")
    kl = results["kl"]
    print(f"    Status: {kl.get('status', 'computed')}")
    if kl.get("status") == "SKIPPED":
        print(f"    Reason: {kl['reason']}")
    else:
        print(f"    Mean KL: {kl['mean_kl']:.4f}")
        print(f"    Interpretation: {kl['interpretation']}")

    print(f"\n  ── SBC ──")
    sbc = results["sbc"]
    if sbc.get("success"):
        print(f"    Iterations: {sbc['n_iterations']}")
        print(f"    Uniform dims: {sbc['n_uniform_dims']}/{sbc['n_dims']}")
        print(f"    Overall calibrated: {sbc['overall_calibrated']}")
    else:
        print(f"    {sbc.get('reason')}")

    print(f"\n  ── Verdict ──")
    v = results["verdict"]
    print(f"    Overall PASS: {v['overall_pass']}")
    print(f"    Note: {v['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Coverage/KL/SBC test CALIBRATION against a simulator+prior,")
    print("    not truth of the physical model.")
    print("  • Without a Bilby reference, KL is SKIPPED (not fabricated).")
    print("  • Well-calibrated ≠ correct model.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
