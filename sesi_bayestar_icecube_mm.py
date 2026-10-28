"""
=============================================================================
SESI BAYESTAR × ICECUBE MULTI-MESSENGER COINCIDENCE
Spatial overlap (HEALPix) + temporal window + joint significance,
with a No-Zeno alert throttle per cfd_pp.py §3.3
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT THIS CONNECTS
------------------
  • Bayestar GW sky map  p_GW(Ω)        — HEALPix probability (sesi_bayestar)
  • IceCube localization p_ν(Ω)         — track (Gaussian ~0.1-1°) or
                                           cascade (~10°) containment region
  → spatial overlap integral   ∫ p_GW(Ω) p_ν(Ω) dΩ
  → temporal coincidence window
  → joint background estimate + trials factor
  → No-Zeno throttled alert emission

LABELS: [C] HEALPix/Bayestar/IceCube conventions  [D] derived here  [O] open

THE DISCIPLINE FROM cfd_pp.py §3.3 (applied to alerts)
------------------------------------------------------
The No-Zeno minimum-event-spacing controls ALERT EMISSION only:
  • It throttles duplicate/near-duplicate alerts (event chatter).
  • It does NOT suppress candidate detections — candidates are still
    recorded; only the outbound alert may be deferred.
  • It NEVER suppresses a system-failure warning (non-finite data,
    failed FITS read). Warnings are recorded, not swallowed.
This mirrors the CFD paper: "the event interval MUST NOT override the
stability limit, and a no-Zeno rule must never suppress evidence of
numerical failure."

HONEST SCOPE (critical)
-----------------------
• As of current public catalogs there is NO confirmed joint GW–high-energy
  neutrino detection. A small coincidence p-value for one event is NOT a
  discovery without the trials factor and an independent background.
• The background here is an ANALYTIC Poisson/Gaussian approximation, not
  the time-slide empirical background a real search uses.
• "Coincidence candidate" ≠ "astrophysical association" — exactly the
  Moral Order paper's discipline: a description/statistic is not the thing.
=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.bayestar_icecube")

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

__all__ = [
    "IceCubeLocalization",
    "SpatialOverlap",
    "MultiMessengerCoincidence",
    "NoZenoAlertThrottler",
]


# ═══════════════════════════════════════════════════════════════════════
# [C/D] ICECUBE LOCALIZATION MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IceCubeLocalization:
    """
    [D] Model an IceCube event localization on the sky.

    event_type:
      • "track"    : muon-track, good containment σ ~ 0.1–1 deg
      • "cascade"  : shower, broad containment σ ~ 5–15 deg
    """
    ra_deg: float
    dec_deg: float
    sigma_deg: float
    event_type: str = "track"
    energy_gev: Optional[float] = None
    timestamp: Optional[float] = None   # Unix or GPS seconds

    def probability_map(self, nside: int) -> np.ndarray:
        """Build a normalized HEALPix probability map (2-D Gaussian)."""
        if not HAS_HEALPY:
            raise RuntimeError("healpy required for HEALPix maps")
        npix = hp.nside2npix(nside)
        pix = np.arange(npix)
        theta, phi = hp.pix2ang(nside, pix)
        dec = 90.0 - np.degrees(theta)
        ra = np.degrees(phi)
        sep = self._ang_sep(ra, dec, self.ra_deg, self.dec_deg)
        prob = np.exp(-0.5 * (sep / self.sigma_deg) ** 2)
        s = prob.sum()
        return prob / s if s > 0 else np.full(npix, 1.0 / npix)

    @staticmethod
    def _ang_sep(ra1, dec1, ra2, dec2) -> np.ndarray:
        r1, d1 = np.radians(ra1), np.radians(dec1)
        r2, d2 = np.radians(ra2), np.radians(dec2)
        c = (np.sin(d1) * np.sin(d2)
             + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2))
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ═══════════════════════════════════════════════════════════════════════
# [C/D] SPATIAL OVERLAP ON HEALPIX
# ═══════════════════════════════════════════════════════════════════════

class SpatialOverlap:
    """
    [D] Compute the spatial overlap integral between the Bayestar GW map
    and the IceCube localization:

        O = ∫ p_GW(Ω) p_ν(Ω) dΩ   ≈   Σ_pix p_GW(pix) p_ν(pix) ΔΩ

    and the probability that a uniformly-distributed neutrino would fall
    inside the GW credible region.
    """

    def __init__(self, nside: int = 64):
        self.nside = nside
        self.pix_area_deg2 = (41253.0 / (12 * nside * nside)) if True else 0.0

    def overlap(self, gw_prob: np.ndarray, nu: IceCubeLocalization) -> float:
        """Overlap integral O = Σ p_GW p_ν ΔΩ (dimensionless)."""
        if not HAS_HEALPY:
            return self._gaussian_overlap_fallback(nu)
        # ensure gw_prob is normalized and at the right nside
        gw = np.asarray(gw_prob, dtype=float)
        gw_nside = hp.npix2nside(len(gw))
        if gw_nside != self.nside:
            gw = hp.ud_grade(gw, self.nside)
        gw = gw / gw.sum()
        nu_prob = nu.probability_map(self.nside)
        pix_area_sr = hp.nside2pixarea(self.nside)
        return float(np.sum(gw * nu_prob) * pix_area_sr)

    def prob_in_credible_region(self, gw_prob: np.ndarray,
                                nu: IceCubeLocalization,
                                credible: float = 0.9) -> float:
        """P(neutrino lies inside the GW credible region)."""
        if not HAS_HEALPY:
            return float("nan")
        gw = np.asarray(gw_prob, dtype=float)
        gw_nside = hp.npix2nside(len(gw))
        if gw_nside != self.nside:
            gw = hp.ud_grade(gw, self.nside)
        gw = gw / gw.sum()
        # smallest region containing `credible` probability
        order = np.argsort(-gw)
        cum = np.cumsum(gw[order])
        region = set(order[: int(np.searchsorted(cum, credible)) + 1])
        # neutrino probability mass inside that region
        nu_prob = nu.probability_map(self.nside)
        return float(sum(nu_prob[p] for p in region))

    def _gaussian_overlap_fallback(self, nu: IceCubeLocalization) -> float:
        """Analytic Gaussian fallback when healpy is absent."""
        # treat GW map as a Gaussian too (caller supplies sigma via attr)
        gw_sigma = getattr(self, "_gw_sigma_deg", 10.0)
        s_eff = math.sqrt(gw_sigma**2 + nu.sigma_deg**2)
        sep = IceCubeLocalization._ang_sep(
            nu.ra_deg, nu.dec_deg, nu.ra_deg, nu.dec_deg)  # 0 if same center
        return float(math.exp(-0.5 * (sep / s_eff)**2) / (2 * math.pi * s_eff**2))


# ═══════════════════════════════════════════════════════════════════════
# [D] MULTI-MESSENGER COINCIDENCE
# ═══════════════════════════════════════════════════════════════════════

class MultiMessengerCoincidence:
    """
    [D] Joint GW–neutrino coincidence: spatial overlap × temporal window,
    with an analytic background estimate and a trials factor.

    HONEST: this is an analytic approximation, NOT an empirical time-slide
    background. Use it for pipeline prototyping, not for a discovery claim.
    """

    def __init__(self, time_window_s: float = 500.0,
                 n_gw_trials: int = 1, n_nu_trials: int = 1):
        self.time_window_s = time_window_s
        self.n_gw_trials = n_gw_trials
        self.n_nu_trials = n_nu_trials
        self.overlap_calc = SpatialOverlap()

    def evaluate(self, gw_prob: np.ndarray, nu: IceCubeLocalization,
                 gw_time: float, nu_time: float) -> Dict[str, Any]:
        """Full coincidence evaluation for one GW–ν pair."""
        # 1. spatial overlap
        O = self.overlap_calc.overlap(gw_prob, nu)
        p_in_region = self.overlap_calc.prob_in_credible_region(gw_prob, nu)

        # 2. temporal
        dt = abs(nu_time - gw_time)
        temporal_ok = dt <= self.time_window_s

        # 3. analytic spatial p-value:
        #    P(uniform ν gives overlap >= O) ~ O × (4π) [dimensionless proxy]
        #    We use the fraction of the sky the GW credible region covers.
        sky_area_90 = self._gw_sky_area_deg2(gw_prob, 0.9)
        p_spatial = min(1.0, sky_area_90 / 41253.0)

        # 4. temporal p-value: window / total search time (assume 1 yr)
        T_search = 365.25 * 24 * 3600.0
        p_temporal = min(1.0, 2.0 * self.time_window_s / T_search)

        # 5. joint + trials
        trials = self.n_gw_trials * self.nu_trials
        p_joint = min(1.0, p_spatial * p_temporal * trials)

        return {
            "overlap_integral": O,
            "p_nu_in_gw_90_region": p_in_region,
            "dt_s": dt,
            "temporal_ok": temporal_ok,
            "gw_sky_area_90_deg2": sky_area_90,
            "p_spatial": p_spatial,
            "p_temporal": p_temporal,
            "trials_factor": trials,
            "p_joint_trials": p_joint,
            "is_candidate": bool(temporal_ok and p_joint < 1e-2),
            "note": ("analytic background approximation; NOT a discovery "
                     "statistic; no confirmed GW–ν joint detection exists"),
        }

    def _gw_sky_area_deg2(self, gw_prob: np.ndarray, credible: float) -> float:
        if not HAS_HEALPY:
            return float("nan")
        gw = np.asarray(gw_prob, dtype=float)
        nside = hp.npix2nside(len(gw))
        gw = gw / gw.sum()
        order = np.argsort(-gw)
        cum = np.cumsum(gw[order])
        npix_region = int(np.searchsorted(cum, credible)) + 1
        return float(npix_region * hp.nside2pixarea(nside, degrees=True))


# ═══════════════════════════════════════════════════════════════════════
# [D] NO-ZENO ALERT THROTTLER  (cfd_pp.py §3.3 discipline)
# ═══════════════════════════════════════════════════════════════════════

class NoZenoAlertThrottler:
    """
    [D] Minimum-spacing throttle for multi-messenger ALERT EMISSION.

    Enforces  N(T) ≤ 1 + floor(T/τ_e)  (cfd_pp.py Prop 1) on outbound
    alerts, while preserving the §3.3 discipline:
      • candidates are always RECORDED (never dropped);
      • only the outbound ALERT may be deferred;
      • system-failure warnings are recorded WITHOUT consuming a slot and
        are NEVER suppressed by spacing.
    """

    def __init__(self, tau_e: float = 1.0):
        self.tau_e = tau_e
        self.last_alert_t = -math.inf
        self.candidate_log: List[Dict[str, Any]] = []
        self.alert_log: List[Dict[str, Any]] = []
        self.deferred_log: List[Dict[str, Any]] = []
        self.warning_log: List[str] = []

    def record_candidate(self, candidate: Dict[str, Any], t: float) -> None:
        """Always record the candidate — throttling never drops detections."""
        candidate = dict(candidate); candidate["_recorded_t"] = t
        self.candidate_log.append(candidate)

    def try_emit(self, candidate: Dict[str, Any], t: float) -> bool:
        """Emit an alert if τ_e has elapsed; else defer (do not drop)."""
        self.record_candidate(candidate, t)
        if t - self.last_alert_t >= self.tau_e:
            self.last_alert_t = t
            self.alert_log.append(candidate)
            return True
        self.deferred_log.append(candidate)
        return False

    def record_warning(self, msg: str) -> None:
        """Record a system warning WITHOUT consuming an event slot."""
        self.warning_log.append(msg)
        logger.warning(f"[MM-SYSTEM WARNING — NOT THROTTLED] {msg}")

    def bound_check(self, T: float) -> Dict[str, Any]:
        """Verify N(T) ≤ 1 + floor(T/τ_e) — cfd_pp.py Prop 1."""
        bound = 1 + math.floor(T / self.tau_e)
        return {
            "N_alerts": len(self.alert_log),
            "bound": bound,
            "invariant_holds": len(self.alert_log) <= bound,
            "n_candidates_recorded": len(self.candidate_log),
            "n_deferred": len(self.deferred_log),
            "n_warnings": len(self.warning_log),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Bayestar × IceCube Multi-Messenger — Smoke Test")
    print("=" * 72)
    print(f"  healpy installed: {HAS_HEALPY}")

    # Build a synthetic GW sky map (a real run would load a Bayestar FITS)
    nside = 32
    if HAS_HEALPY:
        gw_prob = np.zeros(hp.nside2npix(nside))
        center = hp.ang2pix(nside, math.radians(60.0), math.radians(120.0))
        gw_prob[center] = 1.0
        # spread a little
        gw_prob = hp.smoothing(gw_prob, sigma=math.radians(5.0))
        gw_prob = gw_prob / gw_prob.sum()
    else:
        gw_prob = np.ones(12 * nside * nside) / (12 * nside * nside)

    # IceCube track event near the GW center
    nu = IceCubeLocalization(ra_deg=120.5, dec_deg=29.8, sigma_deg=0.5,
                             event_type="track", energy_gev=2e5,
                             timestamp=1e9 + 100.0)

    mm = MultiMessengerCoincidence(time_window_s=500.0,
                                   n_gw_trials=10, n_nu_trials=5)
    res = mm.evaluate(gw_prob, nu, gw_time=1e9, nu_time=nu.timestamp)

    print(f"\n  ── Spatial overlap ──")
    print(f"    overlap integral       : {res['overlap_integral']:.3e}")
    print(f"    P(ν in GW 90% region)  : {res['p_nu_in_gw_90_region']:.3f}")
    print(f"    GW 90% sky area        : {res['gw_sky_area_90_deg2']:.0f} deg²")
    print(f"\n  ── Temporal ──")
    print(f"    Δt                     : {res['dt_s']:.1f} s")
    print(f"    within window          : {res['temporal_ok']}")
    print(f"\n  ── Significance (analytic background) ──")
    print(f"    p_spatial              : {res['p_spatial']:.3e}")
    print(f"    p_temporal             : {res['p_temporal']:.3e}")
    print(f"    trials factor          : {res['trials_factor']}")
    print(f"    p_joint (with trials)  : {res['p_joint_trials']:.3e}")
    print(f"    is_candidate           : {res['is_candidate']}")
    print(f"    note: {res['note']}")

    # No-Zeno alert throttler
    print(f"\n  ── No-Zeno alert throttler (cfd_pp.py §3.3) ──")
    throttler = NoZenoAlertThrottler(tau_e=100.0)
    for i in range(5):
        emitted = throttler.try_emit({"id": f"cand_{i}"}, t=i * 30.0)
        print(f"    t={i*30:3d}s  emitted={emitted}")
    throttler.record_warning("FITS read returned NaN — flagged, not suppressed")
    chk = throttler.bound_check(T=150.0)
    print(f"    alerts={chk['N_alerts']}  bound={chk['bound']}  "
          f"holds={chk['invariant_holds']}")
    print(f"    candidates recorded (never dropped)={chk['n_candidates_recorded']}")
    print(f"    deferred={chk['n_deferred']}  warnings={chk['n_warnings']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • NO confirmed joint GW–high-energy-ν detection exists to date")
    print("  • background is ANALYTIC, not an empirical time-slide")
    print("  • a coincidence candidate is NOT an astrophysical association")
    print("  • No-Zeno throttles ALERT EMISSION only; candidates are recorded")
    print("  • system warnings are NEVER suppressed by spacing")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
