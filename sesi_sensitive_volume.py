"""
=============================================================================
SESI SENSITIVE VOLUME (VT) CALCULATION
Monte-Carlo detection efficiency → volume-time sensitivity figure of merit
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT VT IS [C]
--------------
The sensitive volume-time  VT  is the standard figure of merit for a
gravitational-wave search (used throughout LIGO/Virgo sensitivity papers
and the GWTC catalog papers):

    VT = T_obs × ∫ p_detect(θ)  dV_c(θ)

where the integral runs over source parameters θ (distance, sky position,
orientation, masses...) and dV_c is the comoving volume element. It answers:
"over what volume, for how long, is this search sensitive?"

The integral is evaluated by MONTE CARLO:
  1. Inject N signals UNIFORMLY IN COMOVING VOLUME out to d_max
     (i.e. sample distance with p(d) ∝ d_c² · |dd_c/dz|, or simply p(d)∝d²
     for the low-redshift Euclidean approximation).
  2. Run each through the detection pipeline.
  3. VT = V(d_max) × (N_found / N_total) × T_obs

The detection efficiency ε(d) = found/total as a function of distance gives
the SENSITIVITY CURVE; its integral gives the volume.

LABELS: [C] standard LVK methodology  [D] derived implementation  [O] open

HONEST SCOPE
------------
• VT computed here inherits the limitations of the detection pipeline it
  wraps (template bank, SNR threshold, χ² veto). It is a property of the
  PIPELINE, not of Nature.
• Statistical uncertainty from finite Monte-Carlo samples is reported
  (binomial error on the found fraction). It must NOT be ignored.
• The low-redshift Euclidean volume element is an approximation; a
  production calculation uses the full cosmological dV_c/dz.
• VT on synthetic/toy data is for METHOD VALIDATION, not a published
  sensitivity statement.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.vt")

__all__ = [
    "VTConfig",
    "UniformVolumeSampler",
    "DetectionEfficiencyEstimator",
    "VTCalculator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VTConfig:
    """Configuration for the VT calculation."""
    # Volume / distance
    d_max_mpc: float = 500.0        # maximum injection distance
    n_injections: int = 2000        # Monte-Carlo injections

    # Detection criterion
    snr_threshold: float = 8.0      # network SNR threshold

    # Observation time
    T_obs_years: float = 1.0

    # Monte-Carlo
    seed: int = 42

    # Cosmology (low-z approximation toggle)
    use_lowz_euclidean: bool = True


# ═══════════════════════════════════════════════════════════════════════
# UNIFORM-IN-VOLUME SAMPLER
# ═══════════════════════════════════════════════════════════════════════

class UniformVolumeSampler:
    """
    [C/D] Sample source parameters uniformly in comoving volume.

    For the low-redshift (Euclidean) approximation, uniform-in-volume
    means  p(d) ∝ d², achieved by  d = d_max · u^{1/3},  u ~ U(0,1).
    Sky position is uniform on the sphere; orientation angles are drawn
    from their physically correct distributions.
    """

    def __init__(self, cfg: VTConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def sample_distance(self, n: int) -> np.ndarray:
        """Distance uniform in volume: d = d_max · u^{1/3}."""
        u = self.rng.uniform(0, 1, n)
        return self.cfg.d_max_mpc * np.cbrt(u)

    def sample_sky(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform on the sphere → (ra, dec) in degrees."""
        ra = self.rng.uniform(0, 360, n)
        dec = np.degrees(np.arcsin(self.rng.uniform(-1, 1, n)))
        return ra, dec

    def sample_orientation(self, n: int) -> Dict[str, np.ndarray]:
        """
        Inclination uniform in cos(iota); polarization uniform in [0, 2π).
        """
        cos_iota = self.rng.uniform(-1, 1, n)
        iota = np.arccos(cos_iota)
        psi = self.rng.uniform(0, 2 * math.pi, n)
        phase = self.rng.uniform(0, 2 * math.pi, n)
        return {"iota": iota, "psi": psi, "phase": phase}

    def sample_injections(self) -> Dict[str, np.ndarray]:
        """Draw the full set of Monte-Carlo injections."""
        n = self.cfg.n_injections
        distance = self.sample_distance(n)
        ra, dec = self.sample_sky(n)
        orient = self.sample_orientation(n)
        return {
            "distance_mpc": distance,
            "ra": ra, "dec": dec,
            **orient,
        }


# ═══════════════════════════════════════════════════════════════════════
# DETECTION EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════

class DetectionEfficiencyEstimator:
    """
    [D] Estimate detection efficiency ε(d) from injections.

    Detection criterion (default): optimal SNR at the injection's distance
    exceeds the network threshold. The SNR scales as 1/d, so an injection
    at distance d with reference SNR ρ_ref at d_ref has
        ρ(d) = ρ_ref · (d_ref / d).

    This is a simplified, distance-only efficiency. A production calculation
    marginalizes over sky, orientation, masses, and applies χ² and
    coincidence vetoes via the full pipeline.
    """

    def __init__(self, cfg: VTConfig, reference_snr: float = 20.0,
                 reference_distance_mpc: float = 100.0):
        self.cfg = cfg
        self.reference_snr = reference_snr
        self.reference_distance = reference_distance_mpc

    def snr_at_distance(self, distance_mpc: np.ndarray) -> np.ndarray:
        """ρ(d) = ρ_ref · (d_ref / d)."""
        return self.reference_snr * (self.reference_distance /
                                     np.maximum(distance_mpc, 1e-6))

    def is_detected(self, distance_mpc: np.ndarray) -> np.ndarray:
        """Boolean detection mask from the SNR threshold."""
        return self.snr_at_distance(distance_mpc) >= self.cfg.snr_threshold

    def efficiency_vs_distance(self, injections: Dict[str, np.ndarray],
                               n_bins: int = 20) -> Dict[str, np.ndarray]:
        """
        Binned detection efficiency ε(d) for the sensitivity curve.
        """
        d = injections["distance_mpc"]
        detected = self.is_detected(d)

        bin_edges = np.linspace(0, self.cfg.d_max_mpc, n_bins + 1)
        bin_idx = np.digitize(d, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        efficiency = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            counts[b] = mask.sum()
            if counts[b] > 0:
                efficiency[b] = detected[mask].mean()

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return {
            "bin_centers_mpc": bin_centers,
            "efficiency": efficiency,
            "counts": counts,
        }


# ═══════════════════════════════════════════════════════════════════════
# VT CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class VTCalculator:
    """
    [C/D] Monte-Carlo sensitive volume-time.

        VT = V(d_max) × (N_found / N_total) × T_obs

    with a binomial uncertainty on the found fraction:
        σ_ε = sqrt( ε(1-ε) / N_total )
        σ_VT = V × T_obs × σ_ε
    """

    def __init__(self, cfg: VTConfig):
        self.cfg = cfg
        self.sampler = UniformVolumeSampler(cfg)
        self.efficiency = DetectionEfficiencyEstimator(cfg)

    def volume_within(self, d_mpc: float) -> float:
        """
        Comoving volume within distance d (Mpc³).
        Low-z Euclidean: V = (4π/3) d³.
        """
        if self.cfg.use_lowz_euclidean:
            return (4.0 / 3.0) * math.pi * d_mpc ** 3
        # Production: integrate the cosmological dV_c/dz here.
        return (4.0 / 3.0) * math.pi * d_mpc ** 3

    def compute(self, injections: Optional[Dict[str, np.ndarray]] = None
                ) -> Dict[str, Any]:
        """Run the Monte-Carlo VT estimate."""
        if injections is None:
            injections = self.sampler.sample_injections()

        d = injections["distance_mpc"]
        detected = self.efficiency.is_detected(d)

        n_total = len(d)
        n_found = int(detected.sum())
        found_fraction = n_found / max(n_total, 1)

        # Binomial (Wilson-like) uncertainty on the fraction
        eps = found_fraction
        sigma_eps = math.sqrt(eps * (1 - eps) / max(n_total, 1))

        V_max = self.volume_within(self.cfg.d_max_mpc)
        VT = V_max * found_fraction * self.cfg.T_obs_years
        sigma_VT = V_max * sigma_eps * self.cfg.T_obs_years

        # Sensitivity curve (efficiency vs distance)
        curve = self.efficiency.efficiency_vs_distance(injections)

        # Range: the distance at which efficiency drops to 50%
        d_half = self._half_efficiency_distance(curve)

        return {
            "n_injections": n_total,
            "n_found": n_found,
            "found_fraction": found_fraction,
            "sigma_found_fraction": sigma_eps,
            "V_max_Mpc3": V_max,
            "VT_Mpc3_yr": VT,
            "sigma_VT_Mpc3_yr": sigma_VT,
            "half_efficiency_distance_mpc": d_half,
            "sensitivity_curve": curve,
            "note": ("VT is a property of the PIPELINE on these injections; "
                     "statistical uncertainty from finite samples is reported "
                     "and must not be ignored."),
        }

    @staticmethod
    def _half_efficiency_distance(curve: Dict[str, np.ndarray]) -> Optional[float]:
        """Distance where efficiency crosses 0.5 (interpolated)."""
        centers = curve["bin_centers_mpc"]
        eff = curve["efficiency"]
        for i in range(len(eff) - 1):
            if eff[i] >= 0.5 >= eff[i + 1]:
                frac = (eff[i] - 0.5) / max(eff[i] - eff[i + 1], 1e-12)
                return float(centers[i] + frac * (centers[i + 1] - centers[i]))
        return None


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Sensitive Volume (VT) — Smoke Test")
    print("=" * 72)

    cfg = VTConfig(d_max_mpc=500.0, n_injections=2000, snr_threshold=8.0)
    calc = VTCalculator(cfg)

    result = calc.compute()

    print(f"\n  ── Monte-Carlo VT Result ──")
    print(f"    Injections          : {result['n_injections']}")
    print(f"    Found               : {result['n_found']}")
    print(f"    Found fraction      : {result['found_fraction']:.3f} "
          f"± {result['sigma_found_fraction']:.3f}")
    print(f"    V(d_max)            : {result['V_max_Mpc3']:.3e} Mpc³")
    print(f"    VT                  : {result['VT_Mpc3_yr']:.3e} Mpc³·yr")
    print(f"    σ_VT                : {result['sigma_VT_Mpc3_yr']:.3e} Mpc³·yr")
    d_half = result["half_efficiency_distance_mpc"]
    print(f"    Half-eff. distance  : "
          f"{d_half:.1f} Mpc" if d_half is not None else
          f"    Half-eff. distance  : not reached")

    print(f"\n  ── Sensitivity Curve ε(d) ──")
    curve = result["sensitivity_curve"]
    for c, e, n in zip(curve["bin_centers_mpc"], curve["efficiency"], curve["counts"]):
        bar = "█" * int(e * 20)
        print(f"    d={c:6.1f} Mpc  ε={e:.2f}  n={int(n):4d}  {bar}")

    print(f"\n    Note: {result['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • VT characterizes the PIPELINE on these injections, not Nature")
    print("  • Low-z Euclidean volume is an approximation (no cosmology yet)")
    print("  • Distance-only efficiency; production marginalizes over all params")
    print("  • Statistical uncertainty is reported and must be propagated")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
