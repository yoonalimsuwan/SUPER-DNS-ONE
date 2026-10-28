"""
=============================================================================
SESI MCMC RATIO INVERSION — Metropolis-Hastings for sharp posteriors
Replaces rejection sampling (sesi_ratio_posterior_inversion.py) where the
learned SNRE ratio produces a sharply peaked posterior.
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHY MCMC INSTEAD OF REJECTION
------------------------------
The prior module stated honestly:
    "rejection sampling is exact but inefficient for sharply peaked
     posteriors."
For a narrow posterior, the prior-as-proposal acceptance rate collapses
(often << 1%), because almost every prior draw misses the peak. MCMC
random-walk Metropolis fixes this by walking TOWARD high-density regions,
at the cost of CORRELATED samples.

THE KEY HONEST TRADE-OFF (per Six Problems §6 discipline)
----------------------------------------------------------
Rejection: independent samples, exact, but acceptance → 0 for sharp peaks.
MCMC:      works for sharp peaks, but samples are AUTOCORRELATED.
           The relevant quantity is EFFECTIVE SAMPLE SIZE (ESS), not the
           raw count. Running MCMC longer does not prove convergence;
           diagnostics (ESS, R-hat, acceptance) must be REPORTED.

TARGET DISTRIBUTION (Bayes inversion, same identity as before)
--------------------------------------------------------------
    log p(θ|x) = log r(x,θ) + log p(θ) + const
MCMC needs only the UNNORMALIZED log-posterior — it never requires the
evidence p(x). This is exactly the advantage over grid normalization.

LABELS: [P] MH detailed balance  [D] adaptive/diagnostics here  [O] open

HONEST SCOPE
------------
• This is self-contained random-walk Metropolis for PROTOTYPING.
  Production CBC inference uses dynesty/nested sampling or NUTS (Bilby).
• Convergence is DIAGNOSED, never assumed. ESS and R-hat are computed
  and must be checked by the user before trusting any credible interval.
• The learned SNRE ratio still carries approximation error; MCMC samples
  the posterior IMPLIED BY that ratio, not the true posterior.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.mcmc_inv")

__all__ = [
    "MCMCConfig",
    "AdaptiveMetropolisSampler",
    "MCMCDiagnostics",
    "MCMCRatioInverter",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCMCConfig:
    n_samples: int = 4000          # post-warmup samples to keep
    warmup: int = 1000             # burn-in steps (adaptation happens here)
    thin: int = 1                  # keep every thin-th sample
    n_chains: int = 3              # for R-hat
    target_accept: float = 0.30    # RW-MH sweet spot in low dim
    adapt_interval: int = 50
    proposal_scale: float = 0.5    # initial scale relative to prior width
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [D] ADAPTIVE RANDOM-WALK METROPOLIS
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveMetropolisSampler:
    """
    [D] Random-walk Metropolis-Hastings with warmup adaptation.

    [P] Correctness rests on detailed balance for the fixed-target MH step.
    Adaptation is applied ONLY during warmup; post-warmup the proposal is
    frozen so the chain is a proper Markov chain with the correct invariant
    distribution. (Adapting forever would break detailed balance.)
    """

    def __init__(self, cfg: MCMCConfig,
                 log_posterior: Callable[[np.ndarray], float],
                 prior_low: np.ndarray, prior_high: np.ndarray,
                 seed: int = 0):
        self.cfg = cfg
        self.log_post = log_posterior
        self.lo = np.asarray(prior_low, dtype=float)
        self.hi = np.asarray(prior_high, dtype=float)
        self.dim = len(self.lo)
        self.rng = np.random.default_rng(seed)
        # initial isotropic proposal covariance
        span = (self.hi - self.lo) * cfg.proposal_scale
        self.cov = np.diag((span / math.sqrt(self.dim)) ** 2)

    def _log_post_safe(self, theta: np.ndarray) -> float:
        if not (np.all(theta >= self.lo) and np.all(theta <= self.hi)):
            return -math.inf
        try:
            lp = float(self.log_post(theta))
        except Exception:
            return -math.inf
        return lp if math.isfinite(lp) else -math.inf

    def sample_chain(self, theta0: np.ndarray
                     ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run one chain. Returns (samples, stats)."""
        cfg = self.cfg
        total = cfg.warmup + cfg.n_samples * cfg.thin
        samples = np.zeros((cfg.n_samples, self.dim))
        theta = np.clip(np.array(theta0, dtype=float), self.lo, self.hi)
        lp = self._log_post_safe(theta)

        n_accept = 0
        n_adapt = 0
        recent_accept = 0

        kept = 0
        for step in range(total):
            # propose
            theta_prop = self.rng.multivariate_normal(theta, self.cov)
            lp_prop = self._log_post_safe(theta_prop)

            # MH accept/reject
            log_alpha = lp_prop - lp
            if math.isfinite(lp) or math.isfinite(lp_prop):
                if math.log(self.rng.uniform()) < log_alpha:
                    theta, lp = theta_prop, lp_prop
                    n_accept += 1
                    recent_accept += 1

            # adapt ONLY during warmup (preserves post-warmup Markov property)
            if step < cfg.warmup and step > 0 and step % cfg.adapt_interval == 0:
                rate = recent_accept / cfg.adapt_interval
                recent_accept = 0
                # simple multiplicative tuning toward target acceptance
                factor = math.exp(0.5 * (rate - cfg.target_accept))
                factor = min(max(factor, 0.5), 2.0)
                self.cov = self.cov * factor
                n_adapt += 1

            # store post-warmup, thinned
            if step >= cfg.warmup and (step - cfg.warmup) % cfg.thin == 0:
                if kept < cfg.n_samples:
                    samples[kept] = theta
                    kept += 1

        stats = {
            "acceptance_rate": n_accept / total,
            "n_adaptations": n_adapt,
            "final_mean_step": float(np.sqrt(np.trace(self.cov))),
        }
        return samples[:kept], stats

    def sample(self, theta0: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run n_chains independent chains for R-hat."""
        if theta0 is None:
            theta0 = self.rng.uniform(self.lo, self.hi)
        all_chains, all_stats = [], []
        for c in range(self.cfg.n_chains):
            start = self.rng.uniform(self.lo, self.hi) if c > 0 else theta0
            ch, st = self.sample_chain(start)
            all_chains.append(ch)
            all_stats.append(st)
        # pad to common length
        min_len = min(len(ch) for ch in all_chains)
        chains = np.stack([ch[:min_len] for ch in all_chains])  # (chains, n, dim)
        acc = float(np.mean([s["acceptance_rate"] for s in all_stats]))
        return chains, {"acceptance_rate": acc,
                        "n_chains": self.cfg.n_chains,
                        "per_chain": all_stats}


# ═══════════════════════════════════════════════════════════════════════
# [D] MCMC DIAGNOSTICS — ESS, autocorrelation, R-hat
# ═══════════════════════════════════════════════════════════════════════

class MCMCDiagnostics:
    """
    [D] Convergence and efficiency diagnostics.

    These DIAGNOSE the sampler; they do not PROVE the posterior is correct.
    """

    @staticmethod
    def autocorrelation(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
        """Sample autocorrelation up to max_lag."""
        n = len(x)
        x = x - x.mean()
        var = np.sum(x ** 2)
        if var <= 0:
            return np.zeros(min(max_lag, n))
        acf = np.zeros(min(max_lag, n))
        for k in range(len(acf)):
            acf[k] = np.sum(x[:n - k] * x[k:]) / var
        return acf

    @staticmethod
    def integrated_autocorr_time(x: np.ndarray) -> float:
        """
        τ_int = 1 + 2 Σ_k ρ_k, with an automatic window (stop at first
        non-positive ρ to avoid summing noise).
        """
        acf = MCMCDiagnostics.autocorrelation(x)
        tau = 1.0
        for k in range(1, len(acf)):
            if acf[k] <= 0:
                break
            tau += 2.0 * acf[k]
        return tau

    @staticmethod
    def effective_sample_size(x: np.ndarray) -> float:
        """ESS = N / τ_int."""
        tau = MCMCDiagnostics.integrated_autocorr_time(x)
        return float(len(x) / max(tau, 1.0))

    @staticmethod
    def rhat(chains: np.ndarray) -> np.ndarray:
        """
        [C] Gelman-Rubin R-hat per dimension. chains: (n_chains, n, dim).
        R-hat → 1 as chains converge; > 1.1 signals non-convergence.
        """
        m, n, d = chains.shape
        if m < 2 or n < 2:
            return np.full(d, np.nan)
        chain_means = chains.mean(axis=1)          # (m, d)
        chain_vars = chains.var(axis=1, ddof=1)    # (m, d)
        overall_mean = chain_means.mean(axis=0)
        B = n * ((chain_means - overall_mean) ** 2).sum(axis=0) / (m - 1)
        W = chain_vars.mean(axis=0)
        var_plus = (n - 1) / n * W + B / n
        with np.errstate(divide="ignore", invalid="ignore"):
            rhat = np.sqrt(var_plus / np.maximum(W, 1e-30))
        return rhat

    @staticmethod
    def full_report(chains: np.ndarray, stats: Dict) -> Dict[str, Any]:
        """Combined report across all chains and dimensions."""
        m, n, d = chains.shape
        pooled = chains.reshape(-1, d)
        ess = np.array([MCMCDiagnostics.effective_sample_size(pooled[:, j])
                        for j in range(d)])
        rhat = MCMCDiagnostics.rhat(chains)
        return {
            "n_chains": m,
            "n_per_chain": n,
            "acceptance_rate": stats.get("acceptance_rate"),
            "ess_per_dim": ess.tolist(),
            "min_ess": float(ess.min()),
            "total_ess_fraction": float(ess.min() / pooled.shape[0]),
            "rhat_per_dim": rhat.tolist(),
            "max_rhat": float(np.nanmax(rhat)),
            "converged_heuristic": bool(np.all(np.nan_to_num(rhat, nan=2.0) < 1.1)),
            "note": ("ESS and R-hat DIAGNOSE the sampler; they do not prove "
                     "the learned ratio equals the true posterior."),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] MCMC RATIO INVERTER (main interface)
# ═══════════════════════════════════════════════════════════════════════

class MCMCRatioInverter:
    """
    [D] Invert a learned likelihood ratio into posterior samples via MCMC.

        log_posterior(θ) = log_ratio(x_obs, θ) + log_prior(θ)

    Drop-in replacement for RejectionPosteriorSampler when the posterior is
    sharply peaked and rejection acceptance collapses.
    """

    def __init__(self,
                 log_ratio: Callable[[np.ndarray, np.ndarray], float],
                 log_prior: Callable[[np.ndarray], float],
                 prior_low: np.ndarray, prior_high: np.ndarray,
                 cfg: Optional[MCMCConfig] = None):
        self.log_ratio = log_ratio
        self.log_prior = log_prior
        self.lo = np.asarray(prior_low, dtype=float)
        self.hi = np.asarray(prior_high, dtype=float)
        self.cfg = cfg or MCMCConfig()
        self.diagnostics = MCMCDiagnostics()

    def _log_posterior(self, x_obs: np.ndarray) -> Callable[[np.ndarray], float]:
        def lp(theta: np.ndarray) -> float:
            if not (np.all(theta >= self.lo) and np.all(theta <= self.hi)):
                return -math.inf
            lpr = float(self.log_prior(theta))
            if not math.isfinite(lpr):
                return -math.inf
            try:
                lr = float(self.log_ratio(x_obs, theta))
            except Exception:
                return -math.inf
            return lpr + lr if math.isfinite(lr) else -math.inf
        return lp

    def invert(self, x_obs: np.ndarray,
               theta0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run MCMC posterior inversion.

        Returns dict with pooled samples, per-chain array, and diagnostics.
        """
        lp = self._log_posterior(x_obs)
        sampler = AdaptiveMetropolisSampler(
            self.cfg, lp, self.lo, self.hi, seed=self.cfg.seed)
        chains, stats = sampler.sample(theta0)
        report = self.diagnostics.full_report(chains, stats)

        pooled = chains.reshape(-1, chains.shape[-1])
        return {
            "samples": pooled,
            "chains": chains,
            "diagnostics": report,
            "method": "adaptive_random_walk_metropolis",
            "honest_note": ("Posterior is that implied by the LEARNED ratio; "
                            "check ESS/R-hat before using credible intervals."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST — analytic Gaussian (where posterior is known)
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI MCMC Ratio Inversion — Smoke Test")
    print("=" * 72)

    # Analytic linear-Gaussian model: x = θ + ε, ε~N(0,σ²), θ~N(μ0,τ²)
    mu0, tau, sigma, x_obs = 0.0, 3.0, 1.0, 2.0
    s2_true = 1.0 / (1.0 / tau**2 + 1.0 / sigma**2)
    m_true = s2_true * (mu0 / tau**2 + x_obs / sigma**2)
    print(f"\n  Analytic posterior: N(mean={m_true:.3f}, var={s2_true:.3f})")

    def log_ratio(x, theta):
        # likelihood N(x; θ, σ²) — θ-dependent part
        return -0.5 * ((x[0] - theta[0]) / sigma) ** 2

    def log_prior(theta):
        return -0.5 * ((theta[0] - mu0) / tau) ** 2

    cfg = MCMCConfig(n_samples=3000, warmup=800, n_chains=3, seed=7)
    inverter = MCMCRatioInverter(
        log_ratio, log_prior,
        prior_low=np.array([mu0 - 6 * tau]),
        prior_high=np.array([mu0 + 6 * tau]),
        cfg=cfg)

    result = inverter.invert(np.array([x_obs]))
    samples = result["samples"][:, 0]
    diag = result["diagnostics"]

    print(f"\n  ── MCMC results ──")
    print(f"    recovered mean   : {samples.mean():.3f}  (true {m_true:.3f})")
    print(f"    recovered var    : {samples.var():.3f}  (true {s2_true:.3f})")
    print(f"    acceptance rate  : {diag['acceptance_rate']:.3f}")
    print(f"    min ESS          : {diag['min_ess']:.0f}")
    print(f"    max R-hat        : {diag['max_rhat']:.4f}")
    print(f"    converged?       : {diag['converged_heuristic']}")

    mean_ok = abs(samples.mean() - m_true) < 0.1
    var_ok = abs(samples.var() - s2_true) / s2_true < 0.2
    print(f"\n    [{'PASS' if mean_ok else 'FAIL'}] mean matches analytic")
    print(f"    [{'PASS' if var_ok else 'FAIL'}] variance matches analytic")
    print(f"    [{'PASS' if diag['converged_heuristic'] else 'FAIL'}] R-hat converged")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • MCMC targets the UNNORMALIZED posterior (no evidence needed)")
    print("  • Samples are AUTOCORRELATED — ESS, not raw count, is what matters")
    print("  • Convergence is DIAGNOSED (ESS/R-hat), never assumed")
    print("  • The posterior is that of the LEARNED ratio, not ground truth")
    print("  • Production CBC inference uses dynesty/NUTS (Bilby), not RW-MH")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
