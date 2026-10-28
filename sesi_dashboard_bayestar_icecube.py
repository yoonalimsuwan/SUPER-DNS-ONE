"""
=============================================================================
SESI DASHBOARD × BAYESTAR-ICECUBE INTEGRATION
Real spatial-overlap multi-messenger dashboard
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS CONNECTS
------------------
sesi_mm_dashboard.py          → SkyPanel, CoincidenceTable, AlertStreamPanel
sesi_bayestar_icecube_mm.py   → SpatialOverlap, MultiMessengerCoincidence,
                                 IceCubeLocalization, NoZenoAlertThrottler
sesi_bayestar_realdata.py     → BayestarSkyMapLoader (HEALPix FITS / synthetic)

Instead of the previous synthetic Gaussian-only demo, this dashboard drives
coincidence evaluation through the REAL MultiMessengerCoincidence engine:
  • GW Bayestar HEALPix sky map (real FITS if available, else labeled synthetic)
  • IceCube localization (track/cascade Gaussian)
  • SpatialOverlap.overlap()  →  ∫ p_GW(Ω) p_ν(Ω) dΩ
  • temporal window + trials factor + joint p-value
  • No-Zeno throttled alert emission (cfd_pp.py §3.3 discipline)

NO-ZENO DISCIPLINE (preserved from cfd_pp.py §3.3)
--------------------------------------------------
  • throttling controls ALERT EMISSION, never candidate recording
  • deferred candidates are stored, never dropped
  • system failures (FITS read error, healpy missing, non-finite data)
    are recorded WITHOUT consuming an event slot

LABELS: [C] HEALPix/Bayestar/IceCube conventions  [D] integration here
        [O] live GraceDB/GCN/Kafka streams

HONEST SCOPE
------------
• Without a real Bayestar FITS, the GW sky map is SYNTHETIC and flagged.
• No confirmed joint GW–high-energy-ν detection exists; a low joint
  p-value for one pair is a CANDIDATE, not an association.
• Live-stream ingestion is [O] and requires credentials.
=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.dash_mm")

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Real engines from the previous modules
try:
    from sesi_bayestar_icecube_mm import (
        IceCubeLocalization, SpatialOverlap,
        MultiMessengerCoincidence, NoZenoAlertThrottler,
    )
    HAS_MM = True
except ImportError:
    HAS_MM = False

try:
    from sesi_bayestar_realdata import BayestarConfig, BayestarSkyMapLoader
    HAS_BAYESTAR = True
except ImportError:
    HAS_BAYESTAR = False

__all__ = [
    "BayestarIceCubeSource",
    "SpatialOverlapPanel",
    "UnifiedMMDashboard",
]


# ═══════════════════════════════════════════════════════════════════════
# DATA SOURCE — real spatial-overlap engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MMSourceConfig:
    nside: int = 32
    tau_e: float = 5.0                  # No-Zeno min alert spacing (s)
    time_window_s: float = 500.0
    n_gw_trials: int = 10
    n_nu_trials: int = 5
    bayestar_fits: Optional[str] = None  # real FITS path → real sky map
    seed: int = 0


class BayestarIceCubeSource:
    """
    [D/O] Data source that produces (GW sky map, IceCube event) pairs and
    evaluates them through the REAL MultiMessengerCoincidence engine.

    If a Bayestar FITS path is given and healpy is available, the GW map
    is REAL; otherwise it is a clearly-labeled synthetic Gaussian.
    """

    def __init__(self, cfg: MMSourceConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.coinc = MultiMessengerCoincidence(
            time_window_s=cfg.time_window_s,
            n_gw_trials=cfg.n_gw_trials,
            n_nu_trials=cfg.n_nu_trials,
        ) if HAS_MM else None
        self.loader = self._build_loader()

    def _build_loader(self):
        if not HAS_BAYESTAR:
            return None
        bcfg = BayestarConfig(
            fits_path=self.cfg.bayestar_fits,
            nside_fallback=self.cfg.nside,
        )
        return BayestarSkyMapLoader(bcfg)

    def next_pair(self) -> Dict[str, Any]:
        """
        Produce one GW-sky-map + IceCube-event pair and evaluate their
        spatial+temporal coincidence via the real engine.
        """
        # --- GW sky map ---
        if self.loader is not None:
            status = self.loader.load()
            gw_prob = self.loader.prob_map
            gw_synthetic = self.loader.synthetic
            gw_center = self.loader.peak_position()
        else:
            # minimal fallback if the bayestar module is unavailable
            gw_prob, gw_synthetic, gw_center = self._synthetic_gw_map()

        # --- IceCube event ---
        # For a real run, ra/dec/sigma come from the alert; here we draw
        # near/around the GW peak to exercise the overlap integral.
        nu_type = self.rng.choice(["track", "cascade"], p=[0.6, 0.4])
        sigma = float(self.rng.uniform(0.3, 1.0)) if nu_type == "track" \
                else float(self.rng.uniform(5.0, 12.0))
        offset = 0.0 if self.rng.uniform() < 0.5 else float(self.rng.uniform(0, 20))
        nu = IceCubeLocalization(
            ra_deg=(gw_center[0] + offset) % 360.0,
            dec_deg=float(np.clip(gw_center[1] + self.rng.uniform(-10, 10), -90, 90)),
            sigma_deg=sigma,
            event_type=nu_type,
            energy_gev=float(10 ** self.rng.uniform(4, 7)),
            timestamp=time.time(),
        )

        # --- real coincidence evaluation ---
        result = {}
        if self.coinc is not None and gw_prob is not None:
            result = self.coinc.evaluate(gw_prob, nu,
                                         gw_time=time.time(),
                                         nu_time=nu.timestamp)
        result["gw_synthetic"] = gw_synthetic
        result["nu_event"] = {
            "ra": nu.ra_deg, "dec": nu.dec_deg,
            "sigma_deg": nu.sigma_deg, "type": nu.event_type,
            "energy_gev": nu.energy_gev,
        }
        result["gw_map"] = gw_prob
        result["gw_center"] = gw_center
        return result

    def _synthetic_gw_map(self):
        """Fallback synthetic GW map when the bayestar module is missing."""
        if not HAS_HEALPY:
            return None, True, (180.0, 30.0)
        nside = self.cfg.nside
        npix = hp.nside2npix(nside)
        gw = np.zeros(npix)
        center_pix = hp.ang2pix(nside, math.radians(60.0), math.radians(180.0))
        gw[center_pix] = 1.0
        gw = hp.smoothing(gw, sigma=math.radians(8.0))
        gw = gw / gw.sum()
        return gw, True, (180.0, 30.0)


# ═══════════════════════════════════════════════════════════════════════
# SPATIAL OVERLAP PANEL
# ═══════════════════════════════════════════════════════════════════════

class SpatialOverlapPanel:
    """
    [D] Render the GW sky map + neutrino position + overlap region.
    """

    def render(self, pair_result: Dict[str, Any],
               path: str = "mm_overlap.png") -> Optional[str]:
        gw = pair_result.get("gw_map")
        nu = pair_result["nu_event"]
        if not HAS_HEALPY or gw is None or not HAS_MPL:
            return None

        fig = plt.figure(figsize=(12, 5))
        # GW probability map (Mollweide)
        ax1 = fig.add_subplot(121, projection="mollweide")
        # healpy mollview renders into its own figure; use pcolormesh instead
        nside = hp.npix2nside(len(gw))
        theta, phi = hp.pix2ang(nside, np.arange(len(gw)))
        ra = np.where(np.degrees(phi) > 180, np.degrees(phi) - 360, np.degrees(phi))
        dec = 90.0 - np.degrees(theta)
        sc = ax1.scatter(np.radians(ra), np.radians(dec),
                         c=gw, cmap="inferno", s=2, alpha=0.8)
        ax1.set_title(f"GW Bayestar map ({'SYNTHETIC' if pair_result['gw_synthetic'] else 'REAL'})")
        plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)

        # Neutrino marker on the same sky
        ax1.scatter(np.radians(nu["ra"] if nu["ra"] <= 180 else nu["ra"] - 360),
                    np.radians(nu["dec"]),
                    marker="*", s=300, c="cyan", edgecolors="white",
                    label=f"ν ({nu['type']})")
        ax1.legend(loc="upper right")

        # Right panel: overlap diagnostic text
        ax2 = fig.add_subplot(122)
        ax2.axis("off")
        lines = [
            f"Overlap integral O     : {pair_result.get('overlap_integral', float('nan')):.3e}",
            f"P(ν in GW 90% region)  : {pair_result.get('p_nu_in_gw_90_region', float('nan')):.3f}",
            f"GW 90% sky area        : {pair_result.get('gw_sky_area_90_deg2', float('nan')):.0f} deg²",
            f"Angular separation     : (ν vs GW peak)",
            f"Δt                     : {pair_result.get('dt_s', float('nan')):.1f} s",
            f"p_spatial              : {pair_result.get('p_spatial', float('nan')):.3e}",
            f"p_joint (trials)       : {pair_result.get('p_joint_trials', float('nan')):.3e}",
            f"Is candidate           : {pair_result.get('is_candidate', False)}",
            "",
            "NOTE: a candidate is NOT an astrophysical association.",
        ]
        ax2.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                 family="monospace", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED MULTI-MESSENGER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class UnifiedMMDashboard:
    """
    [D] Real-time multi-messenger dashboard driven by the REAL
    Bayestar×IceCube coincidence engine and the No-Zeno alert throttler.

    Per refresh cycle:
      1. source.next_pair() → GW map + ν event + real coincidence result
      2. if is_candidate → throttler.try_emit() (No-Zeno emission control)
      3. log candidate + alert state for the panels
    """

    def __init__(self, cfg: Optional[MMSourceConfig] = None):
        self.cfg = cfg or MMSourceConfig()
        self.source = BayestarIceCubeSource(self.cfg)
        self.overlap_panel = SpatialOverlapPanel()
        self.throttler = NoZenoAlertThrottler(tau_e=self.cfg.tau_e) if HAS_MM else None
        self.history: List[Dict[str, Any]] = []
        self.n_refresh = 0

    def refresh(self) -> Dict[str, Any]:
        """One dashboard refresh cycle."""
        self.n_refresh += 1
        pair = self.source.next_pair()

        is_candidate = bool(pair.get("is_candidate", False))
        alert_emitted = False
        if is_candidate and self.throttler is not None:
            # record + maybe emit, per No-Zeno discipline
            alert_emitted = self.throttler.try_emit(pair, t=time.time())
        elif is_candidate:
            # no throttler (mm module missing) — still record
            alert_emitted = False

        self.history.append({
            "refresh": self.n_refresh,
            "is_candidate": is_candidate,
            "alert_emitted": alert_emitted,
            "p_joint": pair.get("p_joint_trials"),
            "gw_synthetic": pair.get("gw_synthetic"),
        })
        self._last_pair = pair
        return pair

    def render_frame(self, outdir: str = ".") -> Optional[str]:
        """Render the latest pair's overlap panel."""
        if not hasattr(self, "_last_pair"):
            return None
        return self.overlap_panel.render(
            self._last_pair, path=f"{outdir}/mm_overlap.png")

    def summary(self) -> Dict[str, Any]:
        """Dashboard summary with No-Zeno throttle state."""
        n_cand = sum(1 for h in self.history if h["is_candidate"])
        n_emit = sum(1 for h in self.history if h["alert_emitted"])
        n_synth = sum(1 for h in self.history if h["gw_synthetic"])
        out = {
            "n_refresh": self.n_refresh,
            "n_candidates": n_cand,
            "n_alerts_emitted": n_emit,
            "n_synthetic_gw_maps": n_synth,
        }
        if self.throttler is not None:
            # No-Zeno bound check (cfd_pp.py Prop 1 discipline)
            T = self.n_refresh * 1.0
            chk = self.throttler.bound_check(T)
            out["throttler"] = chk
            out["throttle_invariant_holds"] = chk["invariant_holds"]
        return out


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Dashboard × Bayestar-IceCube Integration — Smoke Test")
    print("=" * 72)
    print(f"  healpy           : {HAS_HEALPY}")
    print(f"  matplotlib       : {HAS_MPL}")
    print(f"  mm engine module : {HAS_MM}")
    print(f"  bayestar module  : {HAS_BAYESTAR}")

    cfg = MMSourceConfig(nside=16, tau_e=0.0, n_gw_trials=10, n_nu_trials=5)
    dash = UnifiedMMDashboard(cfg)

    print("\n  ── Running refresh cycles ──")
    for i in range(6):
        pair = dash.refresh()
        print(f"    cycle {i+1}: candidate={pair.get('is_candidate', False):<5} "
              f"p_joint={pair.get('p_joint_trials', float('nan')):.2e} "
              f"gw_synthetic={pair.get('gw_synthetic')}")

    # Render one frame
    print("\n  ── Rendering overlap frame ──")
    path = dash.render_frame(outdir=".")
    print(f"    saved to: {path if path else 'healpy/matplotlib unavailable'}")

    # Summary with No-Zeno bound
    print("\n  ── Summary ──")
    s = dash.summary()
    print(f"    refreshes          : {s['n_refresh']}")
    print(f"    candidates         : {s['n_candidates']}")
    print(f"    alerts emitted     : {s['n_alerts_emitted']}")
    print(f"    synthetic GW maps  : {s['n_synthetic_gw_maps']}")
    if "throttler" in s:
        print(f"    throttle invariant : holds={s['throttle_invariant_holds']}")
        print(f"    candidates recorded (never dropped): "
              f"{s['throttler']['n_candidates_recorded']}")
        print(f"    deferred           : {s['throttler']['n_deferred']}")
        print(f"    warnings           : {s['throttler']['n_warnings']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Without a real Bayestar FITS, GW maps are SYNTHETIC + flagged")
    print("  • No confirmed joint GW–ν detection exists; candidates ≠ association")
    print("  • No-Zeno throttles ALERT EMISSION, never candidate recording")
    print("  • Live GraceDB/GCN/Kafka streams are [O] (need credentials)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
