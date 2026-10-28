"""
=============================================================================
SESI FULL CBC POPULATION MODEL — (m1, m2, z, spins) hierarchical inference
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS EXTENDS
-----------------
sesi_population_vt.py modeled only the primary mass m1. This module
implements the FULL standard CBC population model used by the LVK
population analyses [C: GWTC population papers, Fishbach et al., Mandel]:

    p(θ | Λ) = p(m1 | Λ) · p(q | m1, Λ) · p(z | Λ) · p(χ1, t1 | Λ) · p(χ2, t2 | Λ)

with hyperparameters Λ:
    mass   : α (power-law slope), mmin, mmax, λ_peak, μ_peak, σ_peak
    ratio  : β_q (mass-ratio power-law slope)
    redshift : κ_z (merger-rate evolution ∝ (1+z)^κ)
    spin   : ξ_χ (magnitude power-law), σ_t (tilt concentration)

HIERARCHICAL LIKELIHOOD (importance weighting) [C]
--------------------------------------------------
With per-event posterior samples {θ_ij} drawn under a known PE prior π_PE:

    μ_i(Λ) = (1/n) Σ_j  p_pop(θ_ij | Λ) / π_PE(θ_ij)

    log L({d}|Λ) = Σ_i log μ_i(Λ)  −  N_exp(Λ)   [Poisson/selection term]

LABELS: [C] LVK population framework  [D] derived implementation  [O] open

HONEST SCOPE
------------
• π_PE must be NON-ZERO wherever p_pop is supported, else weights diverge.
  We clip and REPORT the fraction of clipped weights as a diagnostic.
• Few events → wide Λ posteriors. This is a method scaffold; real
  population inference needs O(10–100) events and careful PE priors.
• Redshift evolution here is the simple (1+z)^κ power law, not the full
  Madau-Dickinson SFR; selection uses the toy SNR model unless wired to
  sesi_vt_cosmological_full.py.
• All results inherit the PIPELINE + waveform + cosmology choices.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.pop_full")

__all__ = [
    "FullPopulationConfig",
    "MassModel",
    "MassRatioModel",
    "RedshiftModel",
    "SpinModel",
    "FullPopulationModel",
    "HierarchicalPopulationFull",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — hyperparameter defaults (LVK-like fiducial)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FullPopulationConfig:
    # mass
    mmin: float = 5.0
    mmax: float = 80.0
    alpha: float = 2.3               # p(m1) ∝ m1^{-α}
    lambda_peak: float = 0.1         # Gaussian peak fraction
    mu_peak: float = 35.0
    sigma_peak: float = 5.0
    # mass ratio q = m2/m1
    beta_q: float = 1.0              # p(q) ∝ q^{β_q}
    # redshift
    z_max: float = 2.0
    kappa_z: float = 2.7             # ∝ (1+z)^κ
    # spin
    xi_spin: float = 1.0             # p(χ) ∝ χ^{ξ-1}  (ξ=1 → uniform)
    sigma_t: float = 2.0             # tilt concentration (large → isotropic)
    # numerics
    n_quad: int = 2000
    weight_clip: float = 1e3         # clip p_pop/π_PE to flag divergence
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# [C/D] COMPONENT MODELS
# ═══════════════════════════════════════════════════════════════════════

class MassModel:
    """
    [C/D] Primary mass: truncated power-law + Gaussian peak (LVK standard).
        p(m1) ∝ (1-λ)·m1^{-α} + λ·N(m1; μ, σ)   on [mmin, mmax]
    """

    def __init__(self, cfg: FullPopulationConfig):
        self.cfg = cfg
        grid = np.linspace(cfg.mmin, cfg.mmax, cfg.n_quad)
        self._grid = grid
        self._norm = self._unnorm(grid)
        self._Z = np.trapz(self._norm, grid)

    def _unnorm(self, m1: np.ndarray) -> np.ndarray:
        c = self.cfg
        inrange = (m1 >= c.mmin) & (m1 <= c.mmax)
        pl = np.where(inrange, m1 ** (-c.alpha), 0.0)
        g = np.where(inrange,
                     np.exp(-0.5 * ((m1 - c.mu_peak) / c.sigma_peak) ** 2), 0.0)
        return (1 - c.lambda_peak) * pl + c.lambda_peak * g

    def log_prob(self, m1: np.ndarray) -> np.ndarray:
        p = self._unnorm(np.asarray(m1)) / self._Z
        return np.log(np.maximum(p, 1e-300))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cdf = np.cumsum(self._unnorm(self._grid)); cdf /= cdf[-1]
        return np.interp(rng.uniform(0, 1, n), cdf, self._grid)


class MassRatioModel:
    """
    [C/D] Mass ratio q = m2/m1 ∈ [mmin/m1, 1],  p(q) ∝ q^{β_q}.
    """

    def __init__(self, cfg: FullPopulationConfig):
        self.cfg = cfg

    def log_prob(self, q: np.ndarray, m1: np.ndarray) -> np.ndarray:
        c = self.cfg
        qmin = np.full_like(q, c.mmin) / np.maximum(m1, 1e-6)
        qmin = np.clip(qmin, 1e-6, 1.0)
        inrange = (q >= qmin) & (q <= 1.0)
        # normalization ∫_{qmin}^{1} q^{β} dq
        beta = c.beta_q
        if abs(beta + 1) < 1e-9:
            norm = np.log(1.0 / qmin)
        else:
            norm = (1.0 - qmin ** (beta + 1)) / (beta + 1)
        p = np.where(inrange, q ** beta, 0.0) / np.maximum(norm, 1e-30)
        return np.log(np.maximum(p, 1e-300))

    def sample(self, m1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        qmin = np.clip(c.mmin / np.maximum(m1, 1e-6), 1e-6, 1.0)
        beta = c.beta_q
        u = rng.uniform(0, 1, len(m1))
        if abs(beta + 1) < 1e-9:
            return qmin * (1.0 / qmin) ** u
        return (qmin ** (beta + 1) + u * (1.0 - qmin ** (beta + 1))) ** (1.0 / (beta + 1))


class RedshiftModel:
    """
    [C/D] Redshift evolution p(z) ∝ (1+z)^{κ_z} on [0, z_max].
    (Simplified; the full LVK model uses the Madau-Dickinson SFR.)
    """

    def __init__(self, cfg: FullPopulationConfig):
        self.cfg = cfg
        grid = np.linspace(0, cfg.z_max, cfg.n_quad)
        self._grid = grid
        self._unn = (1 + grid) ** cfg.kappa_z
        self._Z = np.trapz(self._unn, grid)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        inrange = (z >= 0) & (z <= self.cfg.z_max)
        p = np.where(inrange, (1 + np.asarray(z)) ** self.cfg.kappa_z, 0.0) / self._Z
        return np.log(np.maximum(p, 1e-300))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cdf = np.cumsum(self._unn); cdf /= cdf[-1]
        return np.interp(rng.uniform(0, 1, n), cdf, self._grid)


class SpinModel:
    """
    [C/D] Spin magnitude χ ∈ [0,1],  p(χ) ∝ χ^{ξ-1};
    tilt via cosθ drawn from a truncated Gaussian of width σ_t
    (σ_t → ∞ = isotropic).
    """

    def __init__(self, cfg: FullPopulationConfig):
        self.cfg = cfg

    def log_prob_mag(self, chi: np.ndarray) -> np.ndarray:
        xi = self.cfg.xi_spin
        inrange = (chi >= 0) & (chi <= 1)
        # norm ∫_0^1 χ^{ξ-1} dχ = 1/ξ
        p = np.where(inrange, chi ** (xi - 1), 0.0) * xi
        return np.log(np.maximum(p, 1e-300))

    def log_prob_tilt(self, cos_t: np.ndarray) -> np.ndarray:
        st = self.cfg.sigma_t
        if st >= 1e3:
            return np.where((cos_t >= -1) & (cos_t <= 1), math.log(0.5), -math.inf)
        # truncated Gaussian on cos_tilt
        p = np.exp(-0.5 * (cos_t / st) ** 2)
        p = np.where((cos_t >= -1) & (cos_t <= 1), p, 0.0)
        Z = np.trapz(np.exp(-0.5 * (np.linspace(-1, 1, 1000) / st) ** 2),
                     np.linspace(-1, 1, 1000))
        return np.log(np.maximum(p / max(Z, 1e-30), 1e-300))

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        xi = self.cfg.xi_spin
        chi = rng.uniform(0, 1, n) ** (1.0 / xi)
        st = self.cfg.sigma_t
        if st >= 1e3:
            cos_t = rng.uniform(-1, 1, n)
        else:
            cos_t = np.clip(rng.normal(0, st, n), -1, 1)
        return chi, cos_t


# ═══════════════════════════════════════════════════════════════════════
# [D] FULL JOINT POPULATION MODEL
# ═══════════════════════════════════════════════════════════════════════

class FullPopulationModel:
    """
    [D] The joint population density p(m1, q, z, χ1, t1, χ2, t2 | Λ).

    log p_pop = log p(m1) + log p(q|m1) + log p(z)
               + log p(χ1) + log p(t1) + log p(χ2) + log p(t2)
    """

    def __init__(self, cfg: FullPopulationConfig):
        self.cfg = cfg
        self.mass = MassModel(cfg)
        self.q = MassRatioModel(cfg)
        self.z = RedshiftModel(cfg)
        self.spin = SpinModel(cfg)
        self.rng = np.random.default_rng(cfg.seed)

    def log_prob(self, theta: Dict[str, np.ndarray]) -> np.ndarray:
        """theta keys: m1, q, z, chi1, ct1, chi2, ct2 (arrays)."""
        lp = (self.mass.log_prob(theta["m1"])
              + self.q.log_prob(theta["q"], theta["m1"])
              + self.z.log_prob(theta["z"])
              + self.spin.log_prob_mag(theta["chi1"])
              + self.spin.log_prob_tilt(theta["ct1"])
              + self.spin.log_prob_mag(theta["chi2"])
              + self.spin.log_prob_tilt(theta["ct2"]))
        return lp

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """Draw n population draws (for selection/monte-Carlo)."""
        m1 = self.mass.sample(n, self.rng)
        q = self.q.sample(m1, self.rng)
        z = self.z.sample(n, self.rng)
        chi1, ct1 = self.spin.sample(n, self.rng)
        chi2, ct2 = self.spin.sample(n, self.rng)
        return {"m1": m1, "q": q, "z": z,
                "chi1": chi1, "ct1": ct1, "chi2": chi2, "ct2": ct2}


# ═══════════════════════════════════════════════════════════════════════
# [D] FULL HIERARCHICAL INFERENCE
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalPopulationFull:
    """
    [D] Hierarchical likelihood over full-dimension per-event posteriors.

    μ_i(Λ) = (1/n) Σ_j  exp[ log p_pop(θ_ij|Λ) − log π_PE(θ_ij) ]

    with weight clipping + a diagnostic for the clipped fraction (a large
    clipped fraction means π_PE does not cover p_pop — a real problem).
    """

    KEYS = ["m1", "q", "z", "chi1", "ct1", "chi2", "ct2"]

    def __init__(self, cfg: FullPopulationConfig):
        self.cfg = cfg
        self.pop = FullPopulationModel(cfg)

    def per_event_log_mu(self, event_samples: Dict[str, np.ndarray],
                         pe_log_prior: Callable[[Dict[str, np.ndarray]], np.ndarray]
                         ) -> Tuple[float, float]:
        """
        Returns (log μ_i, clipped_fraction).
        """
        log_pop = self.pop.log_prob(event_samples)
        log_pe = pe_log_prior(event_samples)
        log_w = log_pop - log_pe

        # clip extreme weights (π_PE not covering p_pop) and count them
        clip = self.cfg.weight_clip
        n_clipped = int(np.sum(log_w > math.log(clip)))
        log_w = np.clip(log_w, None, math.log(clip))

        m = np.max(log_w)
        log_mu = m + math.log(np.mean(np.exp(log_w - m)))
        clipped_frac = n_clipped / max(len(log_w), 1)
        return log_mu, clipped_frac

    def total_log_likelihood(self, events: List[Dict[str, np.ndarray]],
                             pe_log_prior: Callable) -> Dict[str, Any]:
        """Σ_i log μ_i + clipping diagnostics."""
        total = 0.0
        clip_fracs = []
        for ev in events:
            log_mu, cf = self.per_event_log_mu(ev, pe_log_prior)
            total += log_mu
            clip_fracs.append(cf)
        return {
            "log_likelihood": total,
            "n_events": len(events),
            "mean_clipped_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "max_clipped_fraction": float(np.max(clip_fracs)) if clip_fracs else 0.0,
            "warning": ("π_PE does not fully cover p_pop — increase PE prior "
                        "support or reweight" if (clip_fracs and max(clip_fracs) > 0.05)
                        else None),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Full Population Model — Smoke Test")
    print("=" * 72)

    cfg = FullPopulationConfig(n_quad=1000)
    pop = FullPopulationModel(cfg)

    # Sampling sanity
    draws = pop.sample(2000)
    print(f"\n  ── Population draws (n=2000) ──")
    print(f"    m1 : [{draws['m1'].min():.1f}, {draws['m1'].max():.1f}] "
          f"mean={draws['m1'].mean():.1f}")
    print(f"    q  : [{draws['q'].min():.3f}, {draws['q'].max():.3f}]")
    print(f"    z  : [{draws['z'].min():.3f}, {draws['z'].max():.3f}]")
    print(f"    chi1: mean={draws['chi1'].mean():.3f}")

    # Normalization check (each component integrates ~1)
    m1g = np.linspace(cfg.mmin, cfg.mmax, 2000)
    norm_m1 = np.trapz(np.exp(pop.mass.log_prob(m1g)), m1g)
    zg = np.linspace(0, cfg.z_max, 2000)
    norm_z = np.trapz(np.exp(pop.z.log_prob(zg)), zg)
    print(f"\n  ── Normalization ──")
    print(f"    ∫p(m1)dm1 = {norm_m1:.4f} (expect ~1)")
    print(f"    ∫p(z)dz   = {norm_z:.4f} (expect ~1)")

    # Hierarchical likelihood on toy events
    print(f"\n  ── Hierarchical likelihood (toy) ──")
    hier = HierarchicalPopulationFull(cfg)
    events = [pop.sample(300) for _ in range(5)]   # toy "posteriors"

    def pe_log_prior(theta):
        # toy PE prior: uniform box covering the population support
        lp = np.zeros_like(theta["m1"])
        lp = np.where((theta["m1"] >= cfg.mmin) & (theta["m1"] <= cfg.mmax), lp, -1e30)
        lp = np.where((theta["z"] >= 0) & (theta["z"] <= cfg.z_max), lp, -1e30)
        return lp

    result = hier.total_log_likelihood(events, pe_log_prior)
    print(f"    log L            = {result['log_likelihood']:.2f}")
    print(f"    n_events         = {result['n_events']}")
    print(f"    clipped fraction = {result['mean_clipped_fraction']:.4f}")
    if result["warning"]:
        print(f"    WARNING: {result['warning']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • π_PE must cover p_pop support; clipped fraction is reported")
    print("  • Few events → wide Λ posterior; this is a scaffold, not a measurement")
    print("  • (1+z)^κ is simplified; real LVK uses Madau-Dickinson SFR")
    print("  • Selection effects must come from a real injection campaign")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
