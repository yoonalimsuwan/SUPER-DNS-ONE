"""
=============================================================================
SESI GRAVITON-MASS DISPERSION TEST — beyond-GR via massive graviton
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

PHYSICAL BASIS [C: Will 1998, Phys. Rev. D 57, 2061]
-----------------------------------------------------
If the graviton has a nonzero rest mass m_g, gravitational waves become
dispersive: the dispersion relation changes from
    ω² = c²k²                      (massless, GR)
to
    ω² = c²k² + m_g² c⁴/ℏ²         (massive)
so different frequencies propagate at different speeds. Over a cosmological
distance D, this accumulates a frequency-dependent phase correction to the
GW signal. At leading order the correction to the phasing is

    δΨ(f) = - (π D / ((1+z) λ_g²)) · f⁻¹

where λ_g = h/(m_g c) is the graviton Compton wavelength. This is EXACTLY
the structure used in LVK "testing GR" analyses to place the bound
m_g ≤ 1.27 × 10⁻²³ eV/c² (GWTC-3).

RELATION TO sesi_gr_vs_beyond_gr.py
-----------------------------------
The ppE module used a generic phase deviation β_ppE · u^b. This module is
the SPECIALIZED case b = -1 (since u ∝ f^{1/3}, u^b gives f^{-1/3}... 
actually here we implement the physical f⁻¹ dispersion directly), with
the parameter being m_g itself rather than a phenomenological β. Both are
Bayesian single-parameter beyond-GR probes.

LABELS: [C] Will (1998) dispersion  [D] derived implementation  [O] open

HONEST SCOPE
------------
• This bounds ONE parameter (m_g). It does NOT test the full consistency
  of GR, and a bound on m_g is not a test of "all modified gravity".
• The f⁻¹ phase correction is leading-order; real analyses include the
  full PN phasing and detector response.
• Without lalsuite, waveforms are analytic surrogates.
• An upper limit is a STATEMENT ABOUT THE PRIOR + DATA + MODEL, not a
  proof that the graviton has that mass or is exactly massless.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.graviton_mass")

try:
    import dynesty
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

__all__ = ["GravitonMassConfig", "MassiveGravitonWaveform",
           "GravitonMassLikelihood", "GravitonMassUpperLimit"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GravitonMassConfig:
    # Source
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    redshift: float = 0.1
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256

    # Noise
    noise_psd: float = 1e-46

    # Prior on m_g (eV/c²). LVK bound ~1.27e-23; we allow wider for test.
    mu_g_max_ev: float = 1e-22

    # Numerics
    seed: int = 0
    nlive: int = 400

    # Physical constants
    HBAR_EV_S: float = 6.582119569e-16    # ℏ in eV·s
    C_M_S: float = 2.99792458e8           # m/s
    MPC_M: float = 3.085677581e22         # meters per Mpc


# ═══════════════════════════════════════════════════════════════════════
# [C/D] MASSIVE-GRAVITON WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class MassiveGravitonWaveform:
    """
    [C/D] Inspiral waveform with the massive-graviton dispersion phase.

    The GR chirp phase gets an extra term:
        Ψ_total(f) = Ψ_GR(f) + δΨ(f),   δΨ = -β_D / f
        β_D = π D / ((1+z) λ_g²)
    where λ_g = h/(m_g c). As m_g → 0, λ_g → ∞, δΨ → 0 (recovers GR).
    """

    MSUN_SEC = 4.925490947e-6

    def __init__(self, cfg: GravitonMassConfig):
        self.cfg = cfg
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)

    def _gr_phase(self, Mc_sec: float) -> np.ndarray:
        """Leading-order GR chirp phase."""
        return -(3.0 / 128.0) * (math.pi * Mc_sec * self.freqs) ** (-5.0 / 3.0)

    def dispersion_phase(self, mu_g_ev: float) -> np.ndarray:
        """
        δΨ(f) = -β_D / f from the massive-graviton dispersion.

        Returns zeros if mu_g == 0 (pure GR).
        """
        cfg = self.cfg
        if mu_g_ev <= 0.0:
            return np.zeros_like(self.freqs)

        # Compton wavelength λ_g = ℏ / (m_g c), in meters
        # m_g in kg = mu_g_ev * 1.78266192e-36 kg/eV
        m_g_kg = mu_g_ev * 1.78266192e-36
        lambda_g = cfg.HBAR_EV_S * 1.602176634e-19 / (m_g_kg * cfg.C_M_S) \
                   if m_g_kg > 0 else np.inf

        # Propagation distance in meters
        D_m = cfg.distance_mpc * cfg.MPC_M

        # β_D = π D / ((1+z) λ_g²)  [dimensionless·Hz]
        beta_D = math.pi * D_m / ((1.0 + cfg.redshift) * lambda_g ** 2)

        return -beta_D / self.freqs

    def waveform(self, mu_g_ev: float = 0.0) -> np.ndarray:
        """Frequency-domain strain h̃(f) with optional dispersion phase."""
        cfg = self.cfg
        Mc_sec = cfg.Mc_solar * self.MSUN_SEC
        phase = self._gr_phase(Mc_sec) + self.dispersion_phase(mu_g_ev)
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(cfg.distance_mpc, 1.0)) * 1e-21
        return amp * np.exp(1j * phase)

    def make_data(self, mu_g_true: float = 0.0, snr_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulated data d = h(μ_g_true) + noise."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        h = self.waveform(mu_g_ev=mu_g_true)
        noise_std = math.sqrt(self.cfg.noise_psd * (self.freqs[1] - self.freqs[0]))
        noise = (rng.normal(0, 1, len(h)) + 1j * rng.normal(0, 1, len(h))) * noise_std
        return h * snr_scale + noise


# ═══════════════════════════════════════════════════════════════════════
# [D] LIKELIHOOD FOR m_g
# ═══════════════════════════════════════════════════════════════════════

class GravitonMassLikelihood:
    """
    [D] Gaussian likelihood over m_g given the data, plus a uniform prior
    on m_g ∈ [0, μ_max].
    """

    def __init__(self, cfg: GravitonMassConfig):
        self.cfg = cfg
        self.wf = MassiveGravitonWaveform(cfg)
        self.psd = np.full(cfg.n_freq, cfg.noise_psd)

    def log_likelihood(self, data: np.ndarray, mu_g_ev: float) -> float:
        h = self.wf.waveform(mu_g_ev=mu_g_ev)
        resid = data - h
        df = self.cfg.freqs[1] - self.cfg.freqs[0]
        return float(-2.0 * df * np.sum(np.abs(resid) ** 2 / self.psd))

    def log_prior(self, mu_g_ev: float) -> float:
        """Uniform prior in m_g on [0, μ_max]."""
        if 0.0 <= mu_g_ev <= self.cfg.mu_g_max_ev:
            return -math.log(self.cfg.mu_g_max_ev)
        return -math.inf


# ═══════════════════════════════════════════════════════════════════════
# [D] UPPER-LIMIT ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class GravitonMassUpperLimit:
    """
    [D] Compute the Bayesian 90% upper limit on m_g.

    If dynesty is available, runs nested sampling over m_g and reads the
    90th percentile of the posterior. Otherwise falls back to a grid-based
    posterior (clearly labeled as coarser).

    HONEST: the upper limit depends on the prior range μ_max. Report it.
    """

    def __init__(self, cfg: GravitonMassConfig):
        self.cfg = cfg
        self.likelihood = GravitonMassLikelihood(cfg)

    def compute(self, data: np.ndarray, method: str = "auto") -> Dict[str, Any]:
        mu_max = self.cfg.mu_g_max_ev

        if method == "auto":
            method = "dynesty" if HAS_DYNESTY else "grid"

        if method == "dynesty" and HAS_DYNESTY:
            def loglike(theta):
                return self.likelihood.log_likelihood(data, float(theta[0]))
            def prior_transform(u):
                return np.array([u[0] * mu_max])
            try:
                sampler = dynesty.NestedSampler(
                    loglike, prior_transform, ndim=1,
                    nlive=self.cfg.nlive)
                sampler.run_nested(dlogz=0.05)
                res = sampler.results
                from dynesty.utils import resample_equal
                w = np.exp(res.logwt - np.max(res.logwt)); w /= w.sum()
                samples = resample_equal(res.samples, w)[:, 0]
                return {
                    "method": "dynesty",
                    "mu_g_90_upper_ev": float(np.percentile(samples, 90)),
                    "mu_g_median_ev": float(np.median(samples)),
                    "posterior_samples": samples,
                    "prior_range_ev": [0.0, mu_max],
                }
            except Exception as e:
                logger.warning(f"dynesty failed ({e}); falling back to grid")
                method = "grid"

        # Grid fallback
        grid = np.linspace(0, mu_max, 400)
        logL = np.array([self.likelihood.log_likelihood(data, m) for m in grid])
        logL -= np.max(logL)
        post = np.exp(logL)
        post /= np.trapz(post, grid) + 1e-300
        cdf = np.cumsum(post * np.gradient(grid))
        idx90 = np.searchsorted(cdf, 0.9)
        return {
            "method": "grid",
            "mu_g_90_upper_ev": float(grid[min(idx90, len(grid)-1)]),
            "mu_g_median_ev": float(grid[np.searchsorted(cdf, 0.5)]),
            "posterior_grid": (grid, post),
            "prior_range_ev": [0.0, mu_max],
            "note": "Coarse grid posterior; dynesty preferred.",
        }

    def validate_gr_limit(self) -> Dict[str, Any]:
        """
        Consistency check: with data generated by GR (μ_g=0), the upper
        limit should be small and the recovered posterior peaked near 0.
        """
        rng = np.random.default_rng(self.cfg.seed)
        data = self.likelihood.wf.make_data(mu_g_true=0.0, snr_scale=12.0, rng=rng)
        result = self.compute(data)
        result["injected_mu_g"] = 0.0
        result["consistency"] = (result["mu_g_median_ev"] < 0.5 * self.cfg.mu_g_max_ev)
        return result


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Graviton-Mass Dispersion Test — Smoke Test")
    print("=" * 72)
    print(f"  dynesty installed: {HAS_DYNESTY}")

    cfg = GravitonMassConfig(mu_g_max_ev=1e-22)
    wf = MassiveGravitonWaveform(cfg)

    # Dispersion phase sanity: μ_g=0 → zero phase
    d0 = wf.dispersion_phase(0.0)
    d1 = wf.dispersion_phase(1e-23)
    print(f"\n  ── Dispersion phase ──")
    print(f"    μ_g=0      : max|δΨ| = {np.max(np.abs(d0)):.2e} (should be 0)")
    print(f"    μ_g=1e-23  : max|δΨ| = {np.max(np.abs(d1)):.2e} (should be >0)")
    print(f"    [{'PASS' if np.allclose(d0, 0) else 'FAIL'}] GR limit is dispersion-free")

    # Upper limit on GR-injected data
    print(f"\n  ── Upper limit (GR-injected data) ──")
    limiter = GravitonMassUpperLimit(cfg)
    result = limiter.validate_gr_limit()
    print(f"    method           : {result['method']}")
    print(f"    injected μ_g     : {result['injected_mu_g']:.2e} eV")
    print(f"    90% upper limit  : {result['mu_g_90_upper_ev']:.3e} eV")
    print(f"    median           : {result['mu_g_median_ev']:.3e} eV")
    print(f"    prior range      : {result['prior_range_ev']}")
    print(f"    consistent (posterior near 0): {result['consistency']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Bounds ONE parameter m_g; not a full GR consistency test")
    print("  • f⁻¹ correction is leading-order; real analyses use full PN")
    print("  • Upper limit depends on prior range μ_max (report it)")
    print("  • LVK reference bound: m_g ≤ 1.27e-23 eV/c² (GWTC-3)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
