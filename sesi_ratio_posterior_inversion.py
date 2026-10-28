"""
=============================================================================
SESI RATIO → POSTERIOR INVERSION
Invert a learned likelihood ratio into a posterior via Bayes' rule
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

THE KEY IDENTITY [P]
--------------------
Bayes' rule relates the likelihood ratio r(x,θ)=p(x|θ)/p(x) to the
posterior:

    p(θ | x) = r(x, θ) × p(θ)

because  p(θ|x) = p(x|θ)p(θ)/p(x) = [p(x|θ)/p(x)] p(θ).

So a trained SNRE ratio network s_φ(x,θ) ≈ log r(x,θ) + c(x) gives the
UNNORMALIZED log-posterior directly:

    log p(θ|x) = s_φ(x,θ) + log p(θ) + const

The additive constant is θ-independent and is fixed by normalization.
This inversion is exact given an exact ratio; with a learned ratio the
posterior inherits the ratio's approximation error.

LABELS: [P] Bayes identity  [D] sampling/inversion here  [O] open

HONEST SCOPE
------------
• The SNRE ratio is learned only up to a θ-independent function of x;
  that is harmless for the posterior shape but means the ABSOLUTE
  evidence p(x) is NOT obtained from this inversion.
• A PRIOR MUST BE SUPPLIED. The inversion cannot produce a posterior
  without one, and the result is prior-sensitive in low-data regimes.
• In the SNRE-B case the ratio is trained against the PROPOSAL, so the
  inversion is most trustworthy inside the proposal support. We flag
  samples that fall outside it.
• Validate against an analytic posterior (linear-Gaussian case) before
  trusting the inversion on real data.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.ratio_inv")

__all__ = [
    "RatioToPosterior",
    "RejectionPosteriorSampler",
    "AnalyticGaussianValidator",
]


# ═══════════════════════════════════════════════════════════════════════
# [P/D] RATIO → POSTERIOR
# ═══════════════════════════════════════════════════════════════════════

class RatioToPosterior:
    """
    [D] Convert a learned log-ratio into an unnormalized log-posterior.

        log_posterior(θ) = log_ratio(x_obs, θ) + log_prior(θ)

    The result is unnormalized; use RejectionPosteriorSampler to draw
    samples, or normalize on a grid for low-dimensional inspection.
    """

    def __init__(self,
                 log_ratio: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 log_prior: Callable[[np.ndarray], np.ndarray],
                 prior_low: np.ndarray,
                 prior_high: np.ndarray):
        """
        Args:
            log_ratio:  (x, theta) -> scalar log r(x,θ)
            log_prior:  theta -> scalar log p(θ)  (-inf outside support)
            prior_low/high: box bounds of the prior support
        """
        self.log_ratio = log_ratio
        self.log_prior = log_prior
        self.prior_low = np.asarray(prior_low, dtype=float)
        self.prior_high = np.asarray(prior_high, dtype=float)

    def _in_support(self, theta: np.ndarray) -> bool:
        return bool(np.all(theta >= self.prior_low) and
                    np.all(theta <= self.prior_high))

    def log_posterior_unnorm(self, x_obs: np.ndarray,
                             theta: np.ndarray) -> float:
        """Unnormalized log-posterior; -inf outside prior support."""
        if not self._in_support(theta):
            return -math.inf
        lp = float(self.log_prior(theta))
        if not math.isfinite(lp):
            return -math.inf
        try:
            lr = float(self.log_ratio(x_obs, theta))
        except Exception as e:
            logger.warning(f"log_ratio failed: {e}")
            return -math.inf
        if not math.isfinite(lr):
            return -math.inf
        return lp + lr

    def grid_posterior(self, x_obs: np.ndarray,
                       grid: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate the (normalized) posterior on a grid of θ samples.

        grid: (N, d) parameter samples.
        Returns (normalized_density_weights, log_evidence_estimate).
        """
        logp = np.array([self.log_posterior_unnorm(x_obs, g) for g in grid])
        # log-sum-exp normalization
        m = np.max(logp[np.isfinite(logp)]) if np.any(np.isfinite(logp)) else 0.0
        w = np.exp(logp - m)
        w[~np.isfinite(logp)] = 0.0
        Z = w.sum()
        if Z <= 0:
            raise RuntimeError("Posterior has zero mass on the grid.")
        weights = w / Z
        # crude log-evidence estimate (needs a prior volume factor)
        prior_volume = float(np.prod(self.prior_high - self.prior_low))
        log_evidence = m + math.log(Z / max(len(grid), 1)) + math.log(prior_volume)
        return weights, log_evidence


# ═══════════════════════════════════════════════════════════════════════
# [D] REJECTION SAMPLER FROM THE INVERTED POSTERIOR
# ═══════════════════════════════════════════════════════════════════════

class RejectionPosteriorSampler:
    """
    [D] Draw posterior samples by rejection sampling with the prior as
    proposal.

    For θ ~ prior:
        accept with probability  p(θ|x) / (c × prior(θ))
                                 = r(x,θ) / c
    since p(θ|x) = r(x,θ) p(θ). The constant c must upper-bound
    r(x,θ) over the prior support; we estimate it adaptively.

    HONEST: rejection sampling is simple and exact (no autocorrelation
    bias), but inefficient when r is sharply peaked. For production use,
    swap in MCMC/nested sampling. Efficiency is reported, not hidden.
    """

    def __init__(self, rtp: RatioToPosterior, seed: int = 0):
        self.rtp = rtp
        self.rng = np.random.default_rng(seed)

    def _prior_sample(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.rtp.prior_low, self.rtp.prior_high,
                                size=(n, len(self.rtp.prior_low)))

    def sample(self, x_obs: np.ndarray, n_target: int,
               c_safety: float = 1.2, max_proposals: int = 200000
               ) -> Dict[str, Any]:
        """
        Draw ~n_target posterior samples.

        Returns dict with samples, acceptance rate, and the bound c used.
        """
        # Step 1: estimate c = sup_θ r(x,θ) over prior draws
        probe = self._prior_sample(min(2000, max_proposals))
        logrs = np.array([
            self.rtp.log_posterior_unnorm(x_obs, p) - self.rtp.log_prior(p)
            if self.rtp._in_support(p) else -math.inf
            for p in probe
        ])
        finite = logrs[np.isfinite(logrs)]
        if len(finite) == 0:
            return {"samples": None, "error": "no finite log-ratio found"}
        log_c = float(np.max(finite)) + math.log(c_safety)
        c = math.exp(log_c)

        # Step 2: rejection loop
        samples = []
        n_proposed = 0
        batch = 512
        while len(samples) < n_target and n_proposed < max_proposals:
            cand = self._prior_sample(batch)
            n_proposed += batch
            for p in cand:
                if len(samples) >= n_target:
                    break
                log_r = (self.rtp.log_posterior_unnorm(x_obs, p)
                         - self.rtp.log_prior(p))
                if not math.isfinite(log_r):
                    continue
                # accept with prob r/c = exp(log_r - log_c)
                log_u = math.log(self.rng.uniform())
                if log_u < log_r - log_c:
                    samples.append(p)

        samples = np.array(samples) if samples else None
        acc = (len(samples) / n_proposed) if samples is not None else 0.0

        return {
            "samples": samples,
            "n_samples": 0 if samples is None else len(samples),
            "n_proposed": n_proposed,
            "acceptance_rate": acc,
            "log_c_bound": log_c,
            "note": ("rejection sampling with prior proposal; swap to MCMC "
                     "for sharply peaked posteriors"),
        }


# ═══════════════════════════════════════════════════════════════════════
# [P] ANALYTIC GAUSSIAN VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class AnalyticGaussianValidator:
    """
    [P] Validate the inversion on a linear-Gaussian model where the
    posterior is known in closed form.

    Model:  x = θ + ε,   ε ~ N(0, σ²),   θ ~ N(μ0, τ²)   (1-D)
    Exact posterior:
        θ | x ~ N( m, s² ),
        s² = 1/(1/τ² + 1/σ²),   m = s²(μ0/τ² + x/σ²)
    Exact log-ratio (up to x-only constant):
        log r(x,θ) = log N(x; θ, σ²) + const(θ-independent)

    We build log_ratio from the Gaussian likelihood, run the inversion +
    sampler, and compare recovered mean/variance against the analytic
    posterior. This is the ground-truth check the learned SNRE must
    reproduce before being trusted.
    """

    def __init__(self, mu0: float = 0.0, tau: float = 3.0, sigma: float = 1.0):
        self.mu0, self.tau, self.sigma = mu0, tau, sigma

    def analytic_posterior(self, x: float) -> Tuple[float, float]:
        """Exact posterior (mean, variance)."""
        s2 = 1.0 / (1.0 / self.tau**2 + 1.0 / self.sigma**2)
        m = s2 * (self.mu0 / self.tau**2 + x / self.sigma**2)
        return m, s2

    def log_ratio(self, x_obs: np.ndarray, theta: np.ndarray) -> float:
        """log N(x; θ, σ²) — the likelihood (θ-dependent part)."""
        xx = float(x_obs[0]); th = float(theta[0])
        return -0.5 * ((xx - th) / self.sigma) ** 2

    def log_prior(self, theta: np.ndarray) -> float:
        th = float(theta[0])
        return -0.5 * ((th - self.mu0) / self.tau) ** 2

    def run_validation(self, x_obs_val: float, n_samples: int = 4000
                       ) -> Dict[str, Any]:
        lo = np.array([self.mu0 - 6 * self.tau])
        hi = np.array([self.mu0 + 6 * self.tau])
        rtp = RatioToPosterior(self.log_ratio, self.log_prior, lo, hi)
        sampler = RejectionPosteriorSampler(rtp, seed=1)

        result = sampler.sample(np.array([x_obs_val]), n_samples)
        m_true, s2_true = self.analytic_posterior(x_obs_val)

        if result["samples"] is None:
            return {"status": "FAILED", "reason": "no samples drawn"}

        th = result["samples"][:, 0]
        m_est, s2_est = float(th.mean()), float(th.var())

        return {
            "status": "RUN",
            "x_obs": x_obs_val,
            "analytic_mean": m_true,
            "analytic_var": s2_true,
            "recovered_mean": m_est,
            "recovered_var": s2_est,
            "mean_abs_err": abs(m_est - m_true),
            "var_rel_err": abs(s2_est - s2_true) / s2_true,
            "acceptance_rate": result["acceptance_rate"],
            "passes": (abs(m_est - m_true) < 0.1 and
                       abs(s2_est - s2_true) / s2_true < 0.2),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Ratio→Posterior Inversion — Smoke Test")
    print("=" * 72)

    # Analytic validation (the ground-truth check)
    print("\n  ── [P] Analytic Gaussian validation ──")
    val = AnalyticGaussianValidator(mu0=0.0, tau=3.0, sigma=1.0)
    res = val.run_validation(x_obs_val=2.0, n_samples=3000)
    if res["status"] == "RUN":
        print(f"    x_obs               : {res['x_obs']}")
        print(f"    analytic  mean/var  : {res['analytic_mean']:.3f} / {res['analytic_var']:.3f}")
        print(f"    recovered mean/var  : {res['recovered_mean']:.3f} / {res['recovered_var']:.3f}")
        print(f"    mean abs err        : {res['mean_abs_err']:.3f}")
        print(f"    var relative err    : {res['var_rel_err']:.3f}")
        print(f"    acceptance rate     : {res['acceptance_rate']:.3f}")
        print(f"    [{'PASS' if res['passes'] else 'FAIL'}] inversion matches analytic posterior")
    else:
        print(f"    FAILED: {res.get('reason')}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • p(θ|x)=r(x,θ)p(θ) is exact [P], but the LEARNED r carries error")
    print("  • A prior is REQUIRED; result is prior-sensitive")
    print("  • Absolute evidence p(x) is NOT obtained (ratio is θ-relative)")
    print("  • Rejection sampling is exact but inefficient for sharp peaks")
    print("  • Validate on this analytic case BEFORE trusting a learned SNRE")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
