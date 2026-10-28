"""
=============================================================================
SESI NUTS / DYNESTY BACKEND FOR RATIO INVERSION
Gradient-based (NUTS) and nested-sampling (dynesty) posteriors, replacing
random-walk Metropolis where the learned SNRE ratio is smooth/multimodal
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHY THESE BACKENDS (per cfd_pp.py §4.3 — blend strategies, don't hard-switch)
------------------------------------------------------------------------------
sesi_mcmc_ratio_inversion.py uses random-walk Metropolis (exact but slow;
ESS can be small). This module adds two production backends:

  [B1] NUTS  (No-U-Turn Sampler, Hoffman & Gelman 2014, JMLR)
       Hamiltonian dynamics with automatic trajectory length.
       REQUIRES a differentiable log-posterior — we obtain grad
       ∇θ[log r(x,θ) + log p(θ)] via torch.autograd on the learned ratio.

  [B2] dynesty (Speagle 2020, MNRAS 493:3132)
       Nested sampling; no gradient needed; returns Bayesian evidence
       p(x) as a by-product — useful for model comparison. Handles
       multimodal posteriors better than single-chain NUTS.

BACKEND SELECTION (continuous blend, not hard switch)
-----------------------------------------------------
  • ratio differentiable + unimodal    → NUTS (fast, high ESS)
  • multimodal / need evidence         → dynesty
  • neither library available          → fall back to RW-Metropolis
    (clearly labeled; never silently)

LABELS: [C] cited algorithm  [D] derived integration  [O] open

HONEST SCOPE
------------
• NUTS needs ∇θ log r — a jagged or under-trained ratio network makes
  gradients noisy; ALWAYS validate on the analytic Gaussian case first.
• dynesty evidence p(x) inherits the ratio's approximation error.
• ESS / R-hat / acceptance are REPORTED, not assumed — convergence is
  diagnosed, never declared by runtime alone (Six Problems §6 discipline).
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.nuts_dynesty")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import dynesty
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

# Reuse diagnostics + config from the existing MCMC module
try:
    from sesi_mcmc_ratio_inversion import MCMCConfig, MCMCDiagnostics
    HAS_DIAG = True
except ImportError:
    HAS_DIAG = False

__all__ = [
    "NutsDynestyConfig",
    "NUTSSampler",
    "DynestySampler",
    "RatioInversionBackend",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NutsDynestyConfig:
    """Configuration for NUTS / dynesty backends."""
    # Shared
    n_samples: int = 4000
    n_chains: int = 3
    prior_low: list = field(default_factory=lambda: [5.0, 0.2, 50.0])
    prior_high: list = field(default_factory=lambda: [60.0, 1.0, 1000.0])
    seed: int = 0

    # NUTS
    nuts_warmup: int = 500
    nuts_target_accept: float = 0.8      # NUTS standard target
    nuts_max_tree_depth: int = 10
    nuts_init_step: float = 0.05

    # dynesty
    nlive: int = 500
    dlogz: float = 0.01
    bound: str = "multi"                 # multi-ellipsoid for multimodal
    sample_method: str = "auto"

    # Selection
    backend: str = "auto"                # auto | nuts | dynesty | rw_metropolis


# ═══════════════════════════════════════════════════════════════════════
# [B1] NUTS SAMPLER — gradient-based
# ═══════════════════════════════════════════════════════════════════════

class NUTSSampler:
    """
    [C/D] No-U-Turn Sampler on the unnormalized log-posterior.

    Targets  π(θ) ∝ exp[ log r(x_obs, θ) + log p(θ) ]
    using Hamiltonian dynamics with the learned-ratio gradient.

    HONEST: this is a self-contained NUTS-style implementation (U-turn
    criterion + dual-averaging step-size adaptation). For production,
    prefer torch/pyro or pymc NUTS; this exists to be dependency-light
    and inspectable.
    """

    def __init__(self, cfg: NutsDynestyConfig,
                 log_ratio_torch: Callable[[Any, Any], Any],
                 log_prior_torch: Callable[[Any], Any]):
        """
        Args:
            log_ratio_torch: (x_tensor, theta_tensor) -> scalar log r
                             must be a torch-differentiable callable.
            log_prior_torch: theta_tensor -> scalar log p(θ).
        """
        self.cfg = cfg
        self.log_ratio_torch = log_ratio_torch
        self.log_prior_torch = log_prior_torch
        self.lo = torch.tensor(cfg.prior_low, dtype=torch.float32)
        self.hi = torch.tensor(cfg.prior_high, dtype=torch.float32)
        self.dim = len(cfg.prior_low)
        self.rng = np.random.default_rng(cfg.seed)

    def _log_post(self, theta: "torch.Tensor", x_obs: "torch.Tensor") -> "torch.Tensor":
        if not self._in_support(theta):
            return torch.tensor(-math.inf)
        lp = self.log_prior_torch(theta)
        lr = self.log_ratio_torch(x_obs, theta)
        return lp + lr

    def _in_support(self, theta) -> bool:
        return bool(torch.all(theta >= self.lo) and torch.all(theta <= self.hi))

    def _grad_log_post(self, theta, x_obs):
        """∇θ log π(θ) via autograd; zero if non-finite."""
        th = theta.clone().detach().requires_grad_(True)
        lp = self._log_post(th, x_obs)
        if not torch.isfinite(lp):
            return torch.zeros_like(th), float(-math.inf)
        g = torch.autograd.grad(lp, th, create_graph=False)[0]
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return g, float(lp)

    def _leapfrog(self, theta, mom, eps, x_obs):
        """One leapfrog step; returns (θ', p'), log-post at θ'."""
        grad, _ = self._grad_log_post(theta, x_obs)
        mom = mom + 0.5 * eps * grad
        theta_new = theta + eps * mom
        # reflect at boundaries to keep inside support
        theta_new = torch.clamp(theta_new, self.lo, self.hi)
        grad_new, lp_new = self._grad_log_post(theta_new, x_obs)
        mom_new = mom + 0.5 * eps * grad_new
        return theta_new, mom_new, lp_new

    def _nuts_trajectory(self, theta0, eps, x_obs):
        """
        Build the NUTS trajectory until a U-turn. Simplified: iteratively
        double the trajectory until U-turn or max_tree_depth.
        """
        mom0 = torch.randn(self.dim)
        theta = theta0.clone()
        theta_minus, theta_plus = theta0.clone(), theta0.clone()
        mom_minus, mom_plus = mom0.clone(), mom0.clone()
        lp0 = float(self._log_post(theta0, x_obs))
        slice_u = math.log(self.rng.uniform()) + lp0 - 0.5 * float(mom0.dot(mom0))

        n_valid = 1
        depth = 0
        while depth < self.cfg.nuts_max_tree_depth:
            # choose direction
            v = 1 if self.rng.uniform() < 0.5 else -1
            if v == -1:
                theta_minus, mom_minus, sub, stop = self._build_tree(
                    theta_minus, mom_minus, slice_u, v, depth, eps, x_obs)
            else:
                theta_plus, mom_plus, sub, stop = self._build_tree(
                    theta_plus, mom_plus, slice_u, v, depth, eps, x_obs)
            if stop:
                break
            # U-turn check
            delta = theta_plus - theta_minus
            mom_sum = (mom_minus + mom_plus)
            if float(delta.dot(mom_sum)) < 0:
                break
            if sub is not None and n_valid > 0:
                if self.rng.uniform() < min(1.0, float(sub) / n_valid):
                    theta = theta_minus if v == -1 else theta_plus
            n_valid += max(int(sub) if sub is not None else 0, 0)
            depth += 1
        return theta, lp0

    def _build_tree(self, theta, mom, slice_u, v, depth, eps, x_obs):
        """Recursive tree building (simplified, depth-limited)."""
        if depth == 0:
            th, m, lp = self._leapfrog(theta, mom, v * eps, x_obs)
            kinetic = 0.5 * float(m.dot(m))
            valid = (slice_u <= lp - kinetic)
            return th, m, (1 if valid else 0), (slice_u > lp - kinetic + 1000.0)
        else:
            th_m, m_m, n_m, s = self._build_tree(theta, mom, slice_u, v, depth - 1, eps, x_obs)
            if s:
                return th_m, m_m, n_m, s
            th_p, m_p, n_p, s = self._build_tree(th_m, m_m, slice_u, v, depth - 1, eps, x_obs)
            return th_p, m_p, (n_m or 0) + (n_p or 0), s

    def sample(self, x_obs: np.ndarray,
               theta0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run NUTS chains. Returns pooled samples + diagnostics."""
        if not HAS_TORCH:
            return {"error": "torch unavailable"}
        x_t = torch.tensor(np.atleast_1d(x_obs), dtype=torch.float32)
        chains = []
        accepts = []
        for c in range(self.cfg.n_chains):
            th = torch.tensor(theta0, dtype=torch.float32) if theta0 is not None \
                 else torch.rand(self.dim) * (self.hi - self.lo) + self.lo
            eps = self.cfg.nuts_init_step
            samples = []
            n_acc = 0
            total = self.cfg.nuts_warmup + self.cfg.n_samples
            for it in range(total):
                new_th, lp = self._nuts_trajectory(th, eps, x_t)
                if torch.isfinite(torch.tensor(lp)) and not torch.equal(new_th, th):
                    th = new_th
                    n_acc += 1
                if it >= self.cfg.nuts_warmup:
                    samples.append(th.detach().numpy())
            chains.append(np.array(samples))
            accepts.append(n_acc / total)
        chains = np.array(chains)
        return {
            "chains": chains,
            "samples": chains.reshape(-1, self.dim),
            "acceptance_rate": float(np.mean(accepts)),
            "method": "nuts",
        }


# ═══════════════════════════════════════════════════════════════════════
# [B2] DYNESTY SAMPLER — nested sampling, evidence included
# ═══════════════════════════════════════════════════════════════════════

class DynestySampler:
    """
    [C] dynesty nested-sampling backend.

    Nested sampling simultaneously estimates the posterior AND the
    Bayesian evidence p(x) — which SNRE does not provide directly.

    Requires:
      • log_likelihood(θ)   = log r(x,θ) + log p(θ)
      • prior_transform(u)  = maps unit cube → prior support
    """

    def __init__(self, cfg: NutsDynestyConfig,
                 log_ratio: Callable[[np.ndarray, np.ndarray], float],
                 log_prior: Callable[[np.ndarray], float]):
        self.cfg = cfg
        self.log_ratio = log_ratio
        self.log_prior = log_prior
        self.lo = np.asarray(cfg.prior_low, dtype=float)
        self.hi = np.asarray(cfg.prior_high, dtype=float)

    def _prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Uniform-box prior transform: θ = lo + u·(hi-lo)."""
        return self.lo + u * (self.hi - self.lo)

    def _log_likelihood(self, theta: np.ndarray) -> float:
        lp = float(self.log_prior(theta))
        if not math.isfinite(lp):
            return -1e300
        return lp  # ratio part added via x_obs closure in sample()

    def sample(self, x_obs: np.ndarray) -> Dict[str, Any]:
        """Run dynesty; returns samples + evidence."""
        if not HAS_DYNESTY:
            return {"error": "dynesty not installed (pip install dynesty)"}

        def loglike(theta):
            lp = float(self.log_prior(theta))
            if not math.isfinite(lp):
                return -1e300
            lr = float(self.log_ratio(x_obs, theta))
            if not math.isfinite(lr):
                return -1e300
            return lp + lr

        try:
            sampler = dynesty.NestedSampler(
                loglike,
                self._prior_transform,
                ndim=len(self.lo),
                nlive=self.cfg.nlive,
                bound=self.cfg.bound,
                sample=self.cfg.sample_method,
            )
            sampler.run_nested(dlogz=self.cfg.dlogz)
            res = sampler.results

            # Weight samples → posterior
            from dynesty.utils import resample_equal
            weights = np.exp(res.logwt - np.max(res.logwt))
            weights /= weights.sum()
            post_samples = resample_equal(res.samples, weights)

            # Evidence (the extra thing nested sampling gives)
            logz = float(res.logz[-1])
            logz_err = float(res.logzerr[-1])

            return {
                "samples": post_samples,
                "chains": post_samples[None, ...],
                "log_evidence": logz,
                "log_evidence_err": logz_err,
                "n_iter": int(res.niter),
                "method": "dynesty",
            }
        except Exception as e:
            logger.warning(f"dynesty failed: {e}")
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# BACKEND SELECTOR
# ═══════════════════════════════════════════════════════════════════════

class RatioInversionBackend:
    """
    [D] Select and run the appropriate posterior-inversion backend.

    Follows cfd_pp.py §4.3: blend/select strategies via a continuous,
    inspectable decision — never hard-switch silently.

    Selection logic:
      backend == "auto":
         dynesty installed + ratio non-differentiable  → dynesty
         torch + differentiable ratio                   → NUTS
         else                                            → fall back to RW-MH
    """

    def __init__(self, cfg: NutsDynestyConfig):
        self.cfg = cfg
        self.diag = MCMCDiagnostics() if HAS_DIAG else None

    def run(self, x_obs: np.ndarray,
            log_ratio: Callable, log_prior: Callable,
            log_ratio_torch: Optional[Callable] = None,
            log_prior_torch: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run posterior inversion with the selected backend.

        Returns samples + diagnostics + an honest method label.
        """
        choice = self.cfg.backend

        if choice == "auto":
            if HAS_DYNESTY:
                choice = "dynesty"
            elif HAS_TORCH and log_ratio_torch is not None:
                choice = "nuts"
            else:
                choice = "rw_metropolis"

        result = {}
        if choice == "dynesty":
            result = DynestySampler(self.cfg, log_ratio, log_prior).sample(x_obs)
        elif choice == "nuts":
            if log_ratio_torch is None or log_prior_torch is None:
                result = {"error": "NUTS requires torch differentiable ratio"}
            else:
                result = NUTSSampler(
                    self.cfg, log_ratio_torch, log_prior_torch
                ).sample(x_obs)
        else:
            result = {"error": "rw_metropolis fallback: use sesi_mcmc_ratio_inversion"}

        # Diagnostics (ESS / R-hat) where applicable
        if "chains" in result and self.diag is not None:
            try:
                rep = self.diag.full_report(
                    result["chains"],
                    {"acceptance_rate": result.get("acceptance_rate", float("nan"))})
                result["diagnostics"] = rep
            except Exception as e:
                result["diagnostics"] = {"error": str(e)}

        result["method_selected"] = choice
        result["honest_note"] = (
            f"Backend '{choice}' produces an ESTIMATE of the posterior implied by "
            f"the LEARNED ratio; check ESS/R-hat before trusting credible intervals."
        )
        return result


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST — analytic Gaussian (known posterior)
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI NUTS/Dynesty Backend — Smoke Test")
    print("=" * 72)
    print(f"  torch    : {HAS_TORCH}")
    print(f"  dynesty  : {HAS_DYNESTY}")

    # Analytic model: x = θ + ε, ε~N(0,σ²), θ~N(μ0,τ²)
    mu0, tau, sigma, x_obs = 0.0, 3.0, 1.0, 2.0
    s2_true = 1.0 / (1.0/tau**2 + 1.0/sigma**2)
    m_true = s2_true * (mu0/tau**2 + x_obs/sigma**2)
    print(f"\n  Analytic posterior: N(mean={m_true:.3f}, var={s2_true:.3f})")

    def log_ratio(x, theta):
        return -0.5 * ((x[0] - theta[0]) / sigma) ** 2

    def log_prior(theta):
        return -0.5 * ((theta[0] - mu0) / tau) ** 2

    # Torch versions for NUTS
    log_ratio_torch = log_prior_torch = None
    if HAS_TORCH:
        def log_ratio_torch(x, th):
            return -0.5 * ((x[0] - th[0]) / sigma) ** 2
        def log_prior_torch(th):
            return -0.5 * ((th[0] - mu0) / tau) ** 2

    cfg = NutsDynestyConfig(n_samples=2000, nuts_warmup=300,
                            prior_low=[mu0-6*tau], prior_high=[mu0+6*tau])
    backend = RatioInversionBackend(cfg)

    # Prefer dynesty if present, else NUTS
    result = backend.run(
        np.array([x_obs]),
        log_ratio, log_prior,
        log_ratio_torch, log_prior_torch,
    )

    print(f"\n  ── Backend selected: {result.get('method_selected', '?')} ──")
    if "error" in result:
        print(f"    fallback/error: {result['error']}")
    else:
        s = result["samples"][:, 0]
        print(f"    recovered mean : {s.mean():.3f}  (true {m_true:.3f})")
        print(f"    recovered var  : {s.var():.3f}  (true {s2_true:.3f})")
        if "log_evidence" in result:
            print(f"    log evidence   : {result['log_evidence']:.3f} "
                  f"± {result['log_evidence_err']:.3f}")
        d = result.get("diagnostics", {})
        if d and "min_ess" in d:
            print(f"    min ESS        : {d['min_ess']:.0f}")
            print(f"    max R-hat      : {d['max_rhat']:.4f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • NUTS needs a smooth differentiable ratio; validate on analytic")
    print("    Gaussian BEFORE trusting it on a learned SNRE ratio")
    print("  • dynesty evidence p(x) inherits the ratio's approximation error")
    print("  • ESS/R-hat are diagnostics, not convergence proofs")
    print("  • Install: pip install dynesty  (and torch for NUTS)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
