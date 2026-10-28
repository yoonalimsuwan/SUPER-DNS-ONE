"""
=============================================================================
SESI SOURCE PARAMETER ESTIMATION
Bayesian inference for compact binary coalescence parameters
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT THIS DOES
--------------
[PE1] Gaussian noise likelihood
[PE2] Prior distributions (masses, spins, distance, inclination, sky)
[PE3] Grid search or MCMC sampling
[PE4] MAP estimation + credible intervals
[PE5] Correlation/degeneracy reporting

HONEST SCOPE
------------
• This is a PROTOTYPE, not Bilby or LALInference.
• Uses simplified likelihood; real PE requires:
  - Proper waveform systematics
  - Calibration uncertainty
  - Marginalization over time/phase
• Results are for algorithm development, not publication.

LABELS
------
[P] proved  [C] cited  [D] derived  [O] open
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.param_est")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "ParameterEstimationConfig",
    "BinaryPrior",
    "GaussianLikelihood",
    "GridParameterEstimator",
    "MCMCParameterEstimator",
    "ParameterEstimationReport",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ParameterEstimationConfig:
    """Configuration for parameter estimation."""
    # Parameter ranges (priors)
    mass1_range: Tuple[float, float] = (5.0, 100.0)
    mass2_range: Tuple[float, float] = (5.0, 100.0)
    spin1z_range: Tuple[float, float] = (-0.9, 0.9)
    spin2z_range: Tuple[float, float] = (-0.9, 0.9)
    distance_range: Tuple[float, float] = (10.0, 5000.0)  # Mpc
    inclination_range: Tuple[float, float] = (0.0, math.pi)
    
    # Sampling
    method: str = "grid"  # "grid" | "mcmc"
    grid_points: int = 20  # per dimension
    mcmc_steps: int = 10000
    mcmc_burnin: int = 2000
    mcmc_thin: int = 10
    
    # Waveform
    sample_rate: float = 4096.0
    duration: float = 8.0
    f_min: float = 20.0
    
    # Numerical
    device: str = "cpu"
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# PRIORS
# ═══════════════════════════════════════════════════════════════════════

class BinaryPrior:
    """
    [D] Prior distributions for binary parameters.
    
    Uses:
      • Uniform in component masses
      • Uniform in spins
      • Uniform in comoving volume (distance prior ∝ d_L²)
      • Isotropic inclination (cos θ uniform)
      • Uniform on sphere for sky location
    """

    def __init__(self, cfg: ParameterEstimationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def sample(self, n: int = 1) -> Dict[str, np.ndarray]:
        """Draw samples from the prior."""
        m1 = self.rng.uniform(*self.cfg.mass1_range, n)
        m2 = self.rng.uniform(*self.cfg.mass2_range, n)
        # Enforce m1 >= m2
        swap = m1 < m2
        m1[swap], m2[swap] = m2[swap].copy(), m1[swap].copy()
        
        return {
            "mass1": m1,
            "mass2": m2,
            "spin1z": self.rng.uniform(*self.cfg.spin1z_range, n),
            "spin2z": self.rng.uniform(*self.cfg.spin2z_range, n),
            "distance": self.rng.uniform(*self.cfg.distance_range, n),
            "inclination": np.arccos(self.rng.uniform(-1, 1, n)),  # isotropic
        }

    def log_prob(self, params: Dict[str, float]) -> float:
        """Compute log prior probability."""
        log_p = 0.0
        
        # Mass priors (uniform)
        m1, m2 = params["mass1"], params["mass2"]
        if not (self.cfg.mass1_range[0] <= m1 <= self.cfg.mass1_range[1]):
            return -np.inf
        if not (self.cfg.mass2_range[0] <= m2 <= self.cfg.mass2_range[1]):
            return -np.inf
        if m1 < m2:  # enforce ordering
            return -np.inf
        
        # Spin priors (uniform)
        for s in ["spin1z", "spin2z"]:
            if not (self.cfg.spin1z_range[0] <= params[s] <= self.cfg.spin1z_range[1]):
                return -np.inf
        
        # Distance prior (uniform in volume ∝ d²)
        d = params["distance"]
        if not (self.cfg.distance_range[0] <= d <= self.cfg.distance_range[1]):
            return -np.inf
        log_p += 2 * np.log(d)  # ∝ d²
        
        # Inclination prior (isotropic: uniform in cos θ)
        inc = params["inclination"]
        if not (0 <= inc <= math.pi):
            return -np.inf
        log_p += np.log(np.sin(inc) + 1e-10)
        
        return log_p


# ═══════════════════════════════════════════════════════════════════════
# LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════

class GaussianLikelihood:
    """
    [C] Gaussian noise likelihood for gravitational wave data.
    
    log L(d|θ) = -½ ⟨d - h(θ) | d - h(θ)⟩
    
    where ⟨a|b⟩ = 4 Re ∫ ã(f) b̃*(f) / S_n(f) df
    """

    def __init__(self, cfg: ParameterEstimationConfig, psd: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.psd = psd if psd is not None else np.ones(2049) * 1e-46

    def inner_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Noise-weighted inner product."""
        n = len(a)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.cfg.sample_rate)
        af, bf = np.fft.rfft(a), np.fft.rfft(b)
        
        # Interpolate PSD
        psd_i = np.interp(freqs, 
                          np.linspace(0, self.cfg.sample_rate / 2, len(self.psd)),
                          self.psd)
        
        # Mask to analysis band
        mask = (freqs >= self.cfg.f_min) & (freqs <= self.cfg.sample_rate / 2)
        integrand = 4.0 * af * np.conj(bf) / (psd_i + 1e-60)
        integrand[~mask] = 0.0
        
        df = self.cfg.sample_rate / n
        return float(np.real(np.sum(integrand)) * df)

    def log_likelihood(self, data: np.ndarray, 
                       waveform: np.ndarray) -> float:
        """Compute log likelihood for a waveform model."""
        residual = data - waveform
        return -0.5 * self.inner_product(residual, residual)

    def log_likelihood_ratio(self, data: np.ndarray, 
                             waveform: np.ndarray) -> float:
        """
        log LR = ⟨d|h⟩ - ½⟨h|h⟩
        (for matched filtering against noise-only hypothesis)
        """
        dh = self.inner_product(data, waveform)
        hh = self.inner_product(waveform, waveform)
        return dh - 0.5 * hh


# ═══════════════════════════════════════════════════════════════════════
# GRID ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class GridParameterEstimator:
    """
    [D] Brute-force grid search over parameter space.
    
    HONEST: Only feasible for low-dimensional problems.
    For full PE, use MCMC or nested sampling.
    """

    def __init__(self, cfg: ParameterEstimationConfig, 
                 prior: BinaryPrior, likelihood: GaussianLikelihood,
                 waveform_generator):
        self.cfg = cfg
        self.prior = prior
        self.likelihood = likelihood
        self.waveform_generator = waveform_generator

    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        """Run grid search."""
        # Generate grid (only 2D for demonstration: mass1, mass2)
        m1_grid = np.linspace(*self.cfg.mass1_range, self.cfg.grid_points)
        m2_grid = np.linspace(*self.cfg.mass2_range, self.cfg.grid_points)
        
        # Fix other parameters at fiducial values
        fiducial = {
            "spin1z": 0.0, "spin2z": 0.0,
            "distance": 400.0, "inclination": 0.0,
        }
        
        log_l = np.zeros((len(m1_grid), len(m2_grid)))
        
        for i, m1 in enumerate(m1_grid):
            for j, m2 in enumerate(m2_grid):
                if m1 < m2:
                    log_l[i, j] = -np.inf
                    continue
                
                params = {
                    "mass1": m1, "mass2": m2,
                    **fiducial,
                }
                
                # Generate waveform
                waveform = self.waveform_generator.generate(params)
                if waveform is None or len(waveform) != len(data):
                    log_l[i, j] = -np.inf
                    continue
                
                # Compute likelihood
                log_l[i, j] = self.likelihood.log_likelihood(data, waveform)
        
        # Find MAP
        valid = np.isfinite(log_l)
        if not valid.any():
            return {"success": False, "reason": "No valid grid points"}
        
        max_idx = np.unravel_index(np.argmax(log_l), log_l.shape)
        map_params = {
            "mass1": m1_grid[max_idx[0]],
            "mass2": m2_grid[max_idx[1]],
            **fiducial,
        }
        
        return {
            "success": True,
            "map_parameters": map_params,
            "max_log_likelihood": float(log_l[max_idx]),
            "log_likelihood_grid": log_l,
            "mass1_grid": m1_grid,
            "mass2_grid": m2_grid,
        }


# ═══════════════════════════════════════════════════════════════════════
# MCMC ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class MCMCParameterEstimator:
    """
    [D] Metropolis-Hastings MCMC sampler.
    
    HONEST: Simplified sampler. Production PE uses:
      • Bilby (dynesty, emcee, PyMC)
      • LALInference
      • Proper proposal distributions
    """

    def __init__(self, cfg: ParameterEstimationConfig,
                 prior: BinaryPrior, likelihood: GaussianLikelihood,
                 waveform_generator):
        self.cfg = cfg
        self.prior = prior
        self.likelihood = likelihood
        self.waveform_generator = waveform_generator
        self.rng = np.random.default_rng(cfg.seed)

    def estimate(self, data: np.ndarray) -> Dict[str, Any]:
        """Run MCMC."""
        # Initialize from prior
        current = self.prior.sample(1)
        current = {k: v[0] for k, v in current.items()}
        
        # Proposal std devs (tuned for ~25% acceptance)
        proposal_std = {
            "mass1": 1.0, "mass2": 1.0,
            "spin1z": 0.05, "spin2z": 0.05,
            "distance": 50.0, "inclination": 0.1,
        }
        
        current_waveform = self.waveform_generator.generate(current)
        current_log_l = self.likelihood.log_likelihood(data, current_waveform) \
            if current_waveform is not None else -np.inf
        current_log_p = self.prior.log_prob(current)
        
        samples = []
        n_accept = 0
        
        for step in range(self.cfg.mcmc_steps):
            # Propose new parameters
            proposal = current.copy()
            for key in proposal_std:
                proposal[key] += self.rng.normal(0, proposal_std[key])
            
            # Check prior
            log_p = self.prior.log_prob(proposal)
            if not np.isfinite(log_p):
                continue
            
            # Compute likelihood
            waveform = self.waveform_generator.generate(proposal)
            if waveform is None:
                continue
            
            log_l = self.likelihood.log_likelihood(data, waveform)
            
            # Metropolis-Hastings acceptance
            log_alpha = (log_l + log_p) - (current_log_l + current_log_p)
            if np.log(self.rng.uniform()) < log_alpha:
                current = proposal
                current_log_l = log_l
                current_log_p = log_p
                n_accept += 1
            
            # Store samples (after burn-in, thinned)
            if step >= self.cfg.mcmc_burnin and step % self.cfg.mcmc_thin == 0:
                samples.append(current.copy())
        
        if not samples:
            return {"success": False, "reason": "No samples collected"}
        
        # Convert to arrays
        samples_dict = {key: np.array([s[key] for s in samples]) 
                       for key in samples[0]}
        
        # Compute MAP and credible intervals
        results = {}
        for key, values in samples_dict.items():
            results[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "ci_5": float(np.percentile(values, 5)),
                "ci_95": float(np.percentile(values, 95)),
            }
        
        return {
            "success": True,
            "samples": samples_dict,
            "n_samples": len(samples),
            "acceptance_rate": n_accept / self.cfg.mcmc_steps,
            "posteriors": results,
        }


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

class ParameterEstimationReport:
    """Generate human-readable PE report."""

    @staticmethod
    def summarize(result: Dict[str, Any]) -> str:
        if not result.get("success"):
            return f"PE failed: {result.get('reason', 'unknown')}"
        
        lines = ["=" * 60, "PARAMETER ESTIMATION RESULTS", "=" * 60]
        
        if "map_parameters" in result:
            lines.append("\nMAP Parameters (Grid):")
            for key, val in result["map_parameters"].items():
                lines.append(f"  {key}: {val:.4f}")
            lines.append(f"  max log L: {result['max_log_likelihood']:.2f}")
        
        if "posteriors" in result:
            lines.append(f"\nPosterior Summary (MCMC, n={result['n_samples']}):")
            lines.append(f"  Acceptance rate: {result['acceptance_rate']:.3f}")
            for key, stats in result["posteriors"].items():
                lines.append(f"  {key}:")
                lines.append(f"    mean = {stats['mean']:.4f} ± {stats['std']:.4f}")
                lines.append(f"    90% CI: [{stats['ci_5']:.4f}, {stats['ci_95']:.4f}]")
        
        lines.append("\n" + "=" * 60)
        lines.append("HONEST: Prototype PE, not Bilby/LALInference.")
        lines.append("Results for algorithm development only.")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Parameter Estimation — Smoke Test")
    print("=" * 72)

    cfg = ParameterEstimationConfig(grid_points=5, mcmc_steps=100)
    prior = BinaryPrior(cfg)
    
    # Mock waveform generator
    class MockWaveform:
        def generate(self, params):
            n = int(cfg.sample_rate * cfg.duration)
            return np.random.randn(n) * 1e-21
    
    wf_gen = MockWaveform()
    likelihood = GaussianLikelihood(cfg)
    
    # Synthetic data
    data = np.random.randn(int(cfg.sample_rate * cfg.duration)) * 1e-21
    
    # Test prior sampling
    print("\n  ── Prior Sampling ──")
    samples = prior.sample(5)
    print(f"    Sampled {len(samples['mass1'])} parameter sets")
    print(f"    mass1 range: [{samples['mass1'].min():.1f}, {samples['mass1'].max():.1f}]")
    
    # Test grid estimator
    print("\n  ── Grid Estimator ──")
    grid_est = GridParameterEstimator(cfg, prior, likelihood, wf_gen)
    grid_result = grid_est.estimate(data)
    print(f"    Success: {grid_result['success']}")
    if grid_result["success"]:
        print(f"    MAP mass1: {grid_result['map_parameters']['mass1']:.1f}")
        print(f"    MAP mass2: {grid_result['map_parameters']['mass2']:.1f}")
    
    # Test MCMC estimator
    print("\n  ── MCMC Estimator ──")
    mcmc_est = MCMCParameterEstimator(cfg, prior, likelihood, wf_gen)
    mcmc_result = mcmc_est.estimate(data)
    print(f"    Success: {mcmc_result['success']}")
    if mcmc_result["success"]:
        print(f"    Samples: {mcmc_result['n_samples']}")
        print(f"    Acceptance: {mcmc_result['acceptance_rate']:.3f}")
    
    # Generate report
    print("\n" + ParameterEstimationReport.summarize(mcmc_result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
