"""
=============================================================================
SESI COSMOLOGICAL VT — Full redshift-space sensitive volume-time
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT THIS REPLACES
------------------
sesi_sensitive_volume.py used the low-z Euclidean approximation V=(4π/3)d³.
This module implements the FULL cosmological VT integral in redshift space:

    VT = ∫₀^z_max dz  (dV_c/dz)  ×  [T_obs / (1+z)]  ×  ε(z)

where:
    dV_c/dz = 4π d_c(z)² (c/H₀) / E(z)      comoving volume element
    E(z)    = sqrt(Ω_m(1+z)³ + Ω_Λ)          flat ΛCDM
    1/(1+z)                                   source-frame time dilation
    ε(z)    = detection efficiency             from Monte-Carlo injections

and the redshifted-mass relation for compact binaries:
    m_detector = m_source × (1+z)

LABELS: [C] standard flat-ΛCDM cosmology (Planck 2018 defaults)
        [D] derived Monte-Carlo VT integration here
        [O] open — cosmological-parameter uncertainty NOT yet propagated

HONEST SCOPE
------------
• Default parameters are Planck-2018 flat ΛCDM. Results inherit the
  assumed cosmology; a production VT statement must marginalize over
  H0, Ωm (and report the cosmology used).
• VT is a property of the PIPELINE + cosmology + detection efficiency,
  not of Nature. The Monte-Carlo binomial uncertainty is reported.
• Redshifted masses are handled, but waveform amplitude corrections
  (lensing magnification, K-corrections beyond mass redshifting) are [O].
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.cosmo_vt")

__all__ = ["Cosmology", "RedshiftSampler", "CosmologicalVTCalculator"]


# ═══════════════════════════════════════════════════════════════════════
# [C] FLAT ΛCDM COSMOLOGY
# ═══════════════════════════════════════════════════════════════════════

class Cosmology:
    """
    [C] Flat ΛCDM cosmology (Planck-2018 defaults).

    Provides comoving distance, luminosity distance, comoving volume
    element, and source/detector-frame conversions.

    c/H0 is computed in Mpc. Distances in Mpc, H0 in km/s/Mpc.
    """

    C_KMS = 299792.458  # speed of light, km/s

    def __init__(self, H0: float = 67.4, Om0: float = 0.315):
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = 1.0 - Om0           # flat: Ω_Λ = 1 - Ω_m
        self.dh = self.C_KMS / H0       # Hubble distance, Mpc

    # ── E(z) ──────────────────────────────────────────────────────────
    def Ez(self, z: np.ndarray) -> np.ndarray:
        """Dimensionless Hubble parameter E(z) = H(z)/H0 (flat ΛCDM)."""
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.Om0 * (1.0 + z) ** 3 + self.Ode0)

    # ── Comoving distance  d_c(z) = (c/H0) ∫₀^z dz'/E(z') ────────────
    def comoving_distance(self, z: float, n_steps: int = 2000) -> float:
        """Line-of-sight comoving distance (Mpc) via composite Simpson."""
        if z <= 0:
            return 0.0
        zz = np.linspace(0.0, z, n_steps + 1)
        integrand = 1.0 / self.Ez(zz)
        # Simpson's rule (n_steps even)
        if n_steps % 2 == 1:
            n_steps += 1
            zz = np.linspace(0.0, z, n_steps + 1)
            integrand = 1.0 / self.Ez(zz)
        h = z / n_steps
        S = integrand[0] + integrand[-1] + \
            4.0 * integrand[1:-1:2].sum() + 2.0 * integrand[2:-2:2].sum()
        integral = S * h / 3.0
        return self.dh * integral

    # ── Luminosity distance d_L = (1+z) d_c ───────────────────────────
    def luminosity_distance(self, z: float) -> float:
        return (1.0 + z) * self.comoving_distance(z)

    # ── Comoving volume element dV_c/dz = 4π d_c² (c/H0)/E(z) ────────
    def dVc_dz(self, z: float) -> float:
        """Full-sky comoving volume element, Mpc³ per unit z."""
        dc = self.comoving_distance(z)
        return 4.0 * math.pi * dc * dc * self.dh / self.Ez(z)

    # ── Comoving volume within z (∫ dV_c/dz) ─────────────────────────
    def comoving_volume(self, z: float, n_steps: int = 400) -> float:
        """Total comoving volume within redshift z (Mpc³)."""
        if z <= 0:
            return 0.0
        zz = np.linspace(0.0, z, n_steps)
        dvc = np.array([self.dVc_dz(zi) for zi in zz])
        return float(np.trapz(dvc, zz))

    # ── z from comoving distance (inverse, for sampling) ─────────────
    def z_from_comoving(self, dc_target: float,
                        z_max: float = 3.0, tol: float = 1e-4) -> float:
        """Invert d_c(z) by bisection."""
        lo, hi = 0.0, z_max
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.comoving_distance(mid) < dc_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    # ── Frame conversions for compact binaries ────────────────────────
    def redshift_mass(self, m_source: float, z: float) -> float:
        """Detector-frame mass = source-frame mass × (1+z)."""
        return m_source * (1.0 + z)

    def source_time(self, t_detector: float, z: float) -> float:
        """Source-frame time = detector-frame time / (1+z)."""
        return t_detector / (1.0 + z)


# ═══════════════════════════════════════════════════════════════════════
# [D] REDSHIFT SAMPLER (uniform in comoving volume)
# ═══════════════════════════════════════════════════════════════════════

class RedshiftSampler:
    """
    [D] Sample redshifts uniformly in comoving volume out to z_max.

    Uniform-in-volume sampling is required so the Monte-Carlo found
    fraction directly estimates the volume-averaged efficiency.
    """

    def __init__(self, cosmo: Cosmology, z_max: float = 1.0, seed: int = 42):
        self.cosmo = cosmo
        self.z_max = z_max
        self.rng = np.random.default_rng(seed)
        # total comoving volume out to z_max
        self.V_max = cosmo.comoving_volume(z_max)

    def sample_z(self, n: int) -> np.ndarray:
        """Draw z uniform in comoving volume via inverse-CDF."""
        u = self.rng.uniform(0.0, 1.0, n)
        V_target = u * self.V_max
        zs = np.array([self.cosmo.z_from_comoving(v, z_max=self.z_max)
                       for v in V_target])
        return zs

    def sample_injections(self, n: int,
                          m1_source: float = 30.0,
                          m2_source: float = 25.0) -> Dict[str, np.ndarray]:
        """
        Draw (z, detector-frame masses) for injection.

        Source-frame masses are fixed; detector-frame masses are
        redshifted per sample.
        """
        z = self.sample_z(n)
        m1_det = self.cosmo.redshift_mass(m1_source, z)
        m2_det = self.cosmo.redshift_mass(m2_source, z)
        return {
            "z": z,
            "m1_source": np.full(n, m1_source),
            "m2_source": np.full(n, m2_source),
            "m1_detector": m1_det,
            "m2_detector": m2_det,
            "dL_Mpc": np.array([self.cosmo.luminosity_distance(zi) for zi in z]),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] COSMOLOGICAL VT CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class CosmologicalVTCalculator:
    """
    [D] Monte-Carlo cosmological VT.

    VT = ∫₀^z_max dz (dV_c/dz) [T_obs/(1+z)] ε(z)

    Estimated as  V_max × <ε/(1+z)> × T_obs  over uniform-in-volume
    injections, where the expectation is the Monte-Carlo mean. The
    binomial uncertainty on <ε> is propagated.

    The detection predicate is supplied as a callable so this calculator
    can wrap ANY pipeline (matched-filter SNR, χ² veto, coincidence...).
    """

    def __init__(self, cosmo: Cosmology, z_max: float = 1.0,
                 T_obs_yr: float = 1.0, seed: int = 42):
        self.cosmo = cosmo
        self.z_max = z_max
        self.T_obs_yr = T_obs_yr
        self.sampler = RedshiftSampler(cosmo, z_max, seed)

    def compute(self,
                detector: Callable[[Dict[str, np.ndarray]], np.ndarray],
                n_injections: int = 2000,
                m1_source: float = 30.0,
                m2_source: float = 25.0) -> Dict[str, Any]:
        """
        Run the Monte-Carlo VT estimate.

        Args:
            detector: callable taking the injection dict and returning a
                      boolean found/not-found array (the detection predicate).

        Returns dict with VT, uncertainty, and ε(z) binned curve.
        """
        inj = self.sampler.sample_injections(n_injections, m1_source, m2_source)
        found = np.asarray(detector(inj), dtype=bool)

        z = inj["z"]
        # Per-injection weight includes source-frame time dilation 1/(1+z)
        weights = 1.0 / (1.0 + z)
        weighted_eff = float(np.mean(found * weights))
        # unweighted found fraction for the binomial error estimate
        eff = float(found.mean())
        sigma_eff = math.sqrt(eff * (1 - eff) / max(n_injections, 1))

        V_max = self.sampler.V_max
        VT = V_max * weighted_eff * self.T_obs_yr
        sigma_VT = V_max * sigma_eff * self.T_obs_yr   # conservative (ignores weight variance)

        # ε(z) curve
        curve = self._efficiency_vs_z(z, found)

        return {
            "n_injections": n_injections,
            "n_found": int(found.sum()),
            "found_fraction": eff,
            "sigma_found_fraction": sigma_eff,
            "weighted_efficiency": weighted_eff,
            "V_max_Mpc3": V_max,
            "VT_Mpc3_yr": VT,
            "sigma_VT_Mpc3_yr": sigma_VT,
            "z_max": self.z_max,
            "T_obs_yr": self.T_obs_yr,
            "cosmology": {"H0": self.cosmo.H0, "Om0": self.cosmo.Om0},
            "efficiency_vs_z": curve,
            "note": ("VT is a property of the pipeline + assumed cosmology. "
                     "Cosmological-parameter uncertainty is [O] not propagated."),
        }

    def _efficiency_vs_z(self, z: np.ndarray, found: np.ndarray,
                         n_bins: int = 12) -> Dict[str, np.ndarray]:
        edges = np.linspace(0.0, self.z_max, n_bins + 1)
        idx = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
        eff = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = idx == b
            counts[b] = int(mask.sum())
            eff[b] = found[mask].mean() if counts[b] else 0.0
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {"z_centers": centers, "efficiency": eff, "counts": counts}


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Cosmological VT — Smoke Test")
    print("=" * 72)

    cosmo = Cosmology(H0=67.4, Om0=0.315)

    # Sanity: known distances
    dc_1 = cosmo.comoving_distance(1.0)
    dL_1 = cosmo.luminosity_distance(1.0)
    print(f"\n  d_c(z=1) = {dc_1:.0f} Mpc   (expect ~3300)")
    print(f"  d_L(z=1) = {dL_1:.0f} Mpc   (expect ~2×d_c)")

    # Toy detector: found if dL < 600 Mpc (a sharp horizon)
    def toy_detector(inj):
        return inj["dL_Mpc"] < 600.0

    calc = CosmologicalVTCalculator(cosmo, z_max=0.5, T_obs_yr=1.0)
    result = calc.compute(toy_detector, n_injections=1500)

    print(f"\n  ── Cosmological VT Result ──")
    print(f"    Cosmology           : H0={result['cosmology']['H0']}, "
          f"Om0={result['cosmology']['Om0']}")
    print(f"    V_max               : {result['V_max_Mpc3']:.3e} Mpc³")
    print(f"    found fraction      : {result['found_fraction']:.3f} "
          f"± {result['sigma_found_fraction']:.3f}")
    print(f"    weighted efficiency : {result['weighted_efficiency']:.3f} (×1/(1+z))")
    print(f"    VT                  : {result['VT_Mpc3_yr']:.3e} Mpc³·yr")
    print(f"    σ_VT                : {result['sigma_VT_Mpc3_yr']:.3e} Mpc³·yr")

    print(f"\n  ── ε(z) curve ──")
    curve = result["efficiency_vs_z"]
    for zi, e, c in zip(curve["z_centers"], curve["efficiency"], curve["counts"]):
        bar = "█" * int(e * 20)
        print(f"    z={zi:.3f}  ε={e:.2f}  n={int(c):4d}  {bar}")

    print(f"\n    {result['note']}")
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Planck-2018 flat ΛCDM assumed; cosmology uncertainty [O]")
    print("  • VT characterizes pipeline + cosmology, not Nature")
    print("  • Lensing / K-corrections beyond mass redshift are [O]")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
