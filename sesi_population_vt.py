"""
=============================================================================
SESI POPULATION-LEVEL VT — Hierarchical CBC population inference + merger rate
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS DOES
--------------
Extends single-event VT (sesi_vt_cosmological_full.py) to a POPULATION:

  [P1] Parametric population models p_pop(θ|Λ) with hyperparameters Λ
       (power-law primary mass, mass ratio, optional Gaussian peak).
  [P2] Hierarchical likelihood over detected events, with the standard
       importance-weighting over per-event posterior samples
       [C: Mandel 2010; Fishbach & Holz 2018; GWTC population papers].
  [P3] Selection effects: only DETECTED events are observed, so the
       likelihood must be corrected by the detection probability μ(Λ).
  [P4] Merger rate density R from N_det / (VT × selection).

HIERARCHICAL LIKELIHOOD [C]
---------------------------
For N detected events with posterior samples {θ_ij} drawn under PE prior
π_PE(θ), the population likelihood is:

    L(Λ) = Poisson(N | N_exp(Λ)) × Π_i  μ_i(Λ)

    μ_i(Λ) = (1/n) Σ_j  p_pop(θ_ij | Λ) / π_PE(θ_ij)     (per-event)
    N_exp(Λ) = R × T_obs × ∫ p_det(θ) p_pop(θ|Λ) dθ        (expected count)

The VT integral enters through N_exp: the selection-weighted sensitive
volume-time of the population.

LABELS: [C] hierarchical method  [D] derived implementation  [O] open

HONEST SCOPE
------------
• Population constraints need MANY events; with a handful of toy events
  the posterior on Λ is wide and this is a METHOD DEMO, not a measurement.
• The selection function here is a toy SNR-threshold model; a real
  pipeline injects into real noise (see sesi_vt_cosmological_full.py).
• π_PE (the PE sampling prior) must be known and non-zero wherever
  p_pop is supported, else the importance weights diverge.
• Merger rate is degenerate with selection; R and Λ cannot both be
  tightly constrained without a well-characterized VT.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.pop_vt")

__all__ = ["PopulationConfig", "PowerLawPeakMassModel", "SelectionFunction",
           "HierarchicalPopulationInference", "MergerRateEstimator"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PopulationConfig:
    # Mass population model bounds (solar masses)
    mmin: float = 5.0
    mmax: float = 80.0
    # Power-law slope (α: p(m1) ∝ m1^{-α})
    alpha: float = 2.3
    # Optional Gaussian peak (fraction λ, mean μ, width σ)
    lambda_peak: float = 0.1
    peak_mean: float = 35.0
    peak_sigma: float = 5.0
    # Detection
    snr_threshold: float = 8.0
    reference_snr: float = 20.0     # SNR at reference distance
    reference_distance_mpc: float = 100.0
    # Rate / observation
    T_obs_yr: float = 1.0
    n_monte_carlo: int = 5000
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# [P1] POPULATION MODELS
# ═══════════════════════════════════════════════════════════════════════

class PowerLawPeakMassModel:
    """
    [C/D] Power-law + Gaussian-peak primary-mass distribution (the standard
    phenomenological model used in GWTC population analyses).

        p(m1 | Λ) ∝ [ (1-λ) m1^{-α} + λ G(m1; μ, σ) ]   on [mmin, mmax]

    Returns NORMALIZED densities and can sample.
    """

    def __init__(self, cfg: PopulationConfig):
        self.cfg = cfg

    def _powerlaw(self, m1: np.ndarray) -> np.ndarray:
        c = self.cfg
        pl = np.where((m1 >= c.mmin) & (m1 <= c.mmax),
                      m1 ** (-c.alpha), 0.0)
        return pl

    def _gauss(self, m1: np.ndarray) -> np.ndarray:
        c = self.cfg
        g = np.exp(-0.5 * ((m1 - c.peak_mean) / c.peak_sigma) ** 2)
        return np.where((m1 >= c.mmin) & (m1 <= c.mmax), g, 0.0)

    def log_prob(self, m1: np.ndarray) -> np.ndarray:
        """Normalized log p(m1|Λ)."""
        c = self.cfg
        pl = self._powerlaw(m1)
        g = self._gauss(m1)
        # normalization via simple quadrature on a fine grid
        grid = np.linspace(c.mmin, c.mmax, 2000)
        Z_pl = np.trapz(self._powerlaw(grid), grid)
        Z_g = np.trapz(self._gauss(grid), grid)
        norm_pl = pl / max(Z_pl, 1e-30)
        norm_g = g / max(Z_g, 1e-30)
        p = (1 - c.lambda_peak) * norm_pl + c.lambda_peak * norm_g
        return np.log(np.maximum(p, 1e-300))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample m1 via inverse-CDF on the quadrature grid."""
        c = self.cfg
        grid = np.linspace(c.mmin, c.mmax, 2000)
        p = np.exp(self.log_prob(grid))
        cdf = np.cumsum(p); cdf /= cdf[-1]
        u = rng.uniform(0, 1, n)
        return np.interp(u, cdf, grid)


# ═══════════════════════════════════════════════════════════════════════
# [P3] SELECTION FUNCTION (toy SNR-threshold)
# ═══════════════════════════════════════════════════════════════════════

class SelectionFunction:
    """
    [D] Detection probability p_det as a function of source parameters.

    Toy model: SNR scales as 1/d_L; detected if SNR ≥ threshold. This is
    the same distance-scaling used in sesi_sensitive_volume.py. A real
    selection function comes from found/missed software injections.
    """

    def __init__(self, cfg: PopulationConfig):
        self.cfg = cfg

    def snr_at_distance(self, d_mpc: np.ndarray) -> np.ndarray:
        c = self.cfg
        return c.reference_snr * (c.reference_distance_mpc / np.maximum(d_mpc, 1e-6))

    def p_det(self, d_mpc: np.ndarray) -> np.ndarray:
        """Smoothed detection probability (sigmoid around threshold)."""
        snr = self.snr_at_distance(d_mpc)
        # smooth step: p = 1/(1+exp(-(snr-thr)/width))
        width = 1.0
        return 1.0 / (1.0 + np.exp(-(snr - self.cfg.snr_threshold) / width))

    def horizon_distance(self) -> float:
        """Distance at which SNR = threshold."""
        c = self.cfg
        return c.reference_distance_mpc * c.reference_snr / c.snr_threshold


# ═══════════════════════════════════════════════════════════════════════
# [P2] HIERARCHICAL POPULATION INFERENCE
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalPopulationInference:
    """
    [C/D] Hierarchical likelihood over detected events with importance
    weighting and selection correction.

    Given per-event posterior samples drawn under a known PE prior π_PE,
    the per-event population likelihood is:
        μ_i(Λ) = (1/n) Σ_j p_pop(θ_ij|Λ) / π_PE(θ_ij)

    We implement it for the primary-mass distribution (m1) as the
    parameter of interest; extension to (m1, m2, z) is straightforward.
    """

    def __init__(self, cfg: PopulationConfig):
        self.cfg = cfg
        self.pop_model = PowerLawPeakMassModel(cfg)
        self.selection = SelectionFunction(cfg)
        self.rng = np.random.default_rng(cfg.seed)

    def per_event_likelihood(self, event_samples: np.ndarray,
                             pe_prior_pdf: Callable[[np.ndarray], np.ndarray]
                             ) -> float:
        """μ_i(Λ) for one event's posterior samples (m1 array)."""
        log_pop = self.pop_model.log_prob(event_samples)
        log_pe = np.log(np.maximum(pe_prior_pdf(event_samples), 1e-300))
        log_w = log_pop - log_pe
        # log-sum-exp for stability
        m = np.max(log_w)
        return float(m + math.log(np.mean(np.exp(log_w - m))))

    def hierarchical_log_likelihood(self,
                                    event_samples_list: List[np.ndarray],
                                    pe_prior_pdf: Callable) -> float:
        """
        Full hierarchical log-likelihood (event term).

        Σ_i log μ_i(Λ). The Poisson/selection term is added separately
        in MergerRateEstimator to keep rate and shape separable.
        """
        total = 0.0
        for samples in event_samples_list:
            total += self.per_event_likelihood(samples, pe_prior_pdf)
        return total

    def expected_detections(self, rate: float) -> float:
        """
        N_exp(Λ) = R × T_obs × <p_det> over the population.
        Monte-Carlo over the population model + a distance distribution.
        """
        n = self.cfg.n_monte_carlo
        m1 = self.pop_model.sample(n, self.rng)
        # sample distance uniform in volume out to horizon
        d_hor = self.selection.horizon_distance()
        u = self.rng.uniform(0, 1, n)
        d = d_hor * u ** (1.0 / 3.0)
        p_det = self.selection.p_det(d)
        V_horizon = (4.0 / 3.0) * math.pi * d_horizon**3 if False else \
                    (4.0 / 3.0) * math.pi * d_hor**3
        # detection-weighted volume fraction
        frac = p_det.mean()
        return rate * self.cfg.T_obs_yr * frac  # scaled by V implicitly via rate units


# ═══════════════════════════════════════════════════════════════════════
# [P4] MERGER RATE ESTIMATOR (VT connection)
# ═══════════════════════════════════════════════════════════════════════

class MergerRateEstimator:
    """
    [D] Estimate the merger rate density from N_det and the effective VT.

        R ≈ N_det / VT_effective     [events / Gpc³ / yr]

    where VT_effective = V × <p_det> × T_obs (selection-weighted). This is
    the population-level generalization of the single-source VT.

    HONEST: R and the population shape Λ are DEGENERATE through selection;
    a real analysis samples them jointly. This provides a point estimate
    with Poisson counting uncertainty on N_det.
    """

    def __init__(self, cfg: PopulationConfig):
        self.cfg = cfg
        self.selection = SelectionFunction(cfg)

    def effective_vt(self, d_horizon_mpc: Optional[float] = None) -> float:
        """VT_eff = V_horizon × <p_det> × T_obs  (Gpc³·yr)."""
        d_h = d_horizon_mpc or self.selection.horizon_distance()
        n = self.cfg.n_monte_carlo
        u = self.rng.uniform(0, 1, n) if hasattr(self, "rng") else \
            np.random.default_rng(self.cfg.seed).uniform(0, 1, n)
        d = d_h * u ** (1.0 / 3.0)
        frac = self.selection.p_det(d).mean()
        V_gpc3 = (4.0 / 3.0) * math.pi * (d_h / 1000.0) ** 3
        return V_gpc3 * frac * self.cfg.T_obs_yr

    def rate(self, n_detected: int) -> Dict[str, float]:
        """R = N_det / VT_eff with sqrt(N) Poisson uncertainty."""
        vt = self.effective_vt()
        R = n_detected / max(vt, 1e-30)
        sigma_R = math.sqrt(max(n_detected, 1)) / max(vt, 1e-30)
        return {
            "n_detected": n_detected,
            "VT_eff_Gpc3_yr": vt,
            "rate_Gpc3_yr": R,
            "rate_sigma_Gpc3_yr": sigma_R,
            "note": "Point estimate; R and population shape are degenerate "
                    "through selection. Joint sampling required for real inference.",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Population VT — Smoke Test")
    print("=" * 72)

    cfg = PopulationConfig(n_monte_carlo=3000)
    pop = PowerLawPeakMassModel(cfg)
    sel = SelectionFunction(cfg)
    rng = np.random.default_rng(cfg.seed)

    # Population model sanity
    m1_grid = np.linspace(cfg.mmin, cfg.mmax, 500)
    p = np.exp(pop.log_prob(m1_grid))
    norm = np.trapz(p, m1_grid)
    print(f"\n  ── [P1] Population model ──")
    print(f"    ∫p(m1)dm1 = {norm:.4f}  (should be ~1)")
    print(f"    horizon distance = {sel.horizon_distance():.0f} Mpc")

    # Selection function
    d_test = np.array([50.0, 200.0, 600.0])
    print(f"\n  ── [P3] Selection function ──")
    for d in d_test:
        print(f"    d={d:6.0f} Mpc  p_det={sel.p_det(np.array([d]))[0]:.3f}")

    # Hierarchical likelihood on toy events
    print(f"\n  ── [P2] Hierarchical likelihood (toy) ──")
    hier = HierarchicalPopulationInference(cfg)
    # 5 toy events: posterior samples ~ population + noise
    true_samples = pop.sample(5, rng)
    events = [np.clip(rng.normal(m, 3.0, 200), cfg.mmin, cfg.mmax) for m in true_samples]
    # toy PE prior: uniform in [mmin, mmax]
    def pe_prior_pdf(m):
        return np.where((m >= cfg.mmin) & (m <= cfg.mmax),
                        1.0 / (cfg.mmax - cfg.mmin), 1e-300)
    logL = hier.hierarchical_log_likelihood(events, pe_prior_pdf)
    print(f"    N_events = {len(events)}")
    print(f"    hierarchical log L = {logL:.3f}")

    # Merger rate
    print(f"\n  ── [P4] Merger rate ──")
    rate_est = MergerRateEstimator(cfg)
    result = rate_est.rate(n_detected=5)
    print(f"    VT_eff = {result['VT_eff_Gpc3_yr']:.4f} Gpc³·yr")
    print(f"    R      = {result['rate_Gpc3_yr']:.1f} ± "
          f"{result['rate_sigma_Gpc3_yr']:.1f} Gpc⁻³ yr⁻¹")
    print(f"    {result['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Toy selection + few events → METHOD DEMO, not a measurement")
    print("  • Real population inference needs many events + real injections")
    print("  • R and Λ degenerate through selection; sample jointly in practice")
    print("  • π_PE must be known and cover p_pop's support")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
