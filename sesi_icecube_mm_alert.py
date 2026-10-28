"""
=============================================================================
SESI IceCube MULTI-MESSENGER ALERT CLIENT + COINCIDENCE
GCN Circulars -> neutrino/GW spatial-temporal coincidence
=============================================================================
[C] GCN Circulars  : https://gcn.nasa.gov/circulars/
[C] IceCube alerts : https://icecube.wisc.edu/science/alerts/
[C] GW alerts      : https://gracedb.ligo.org/

HONEST SCOPE
------------
* Real public GCN data; IceCube's full real-time stream needs credentials.
* Coincidence below is an APPROXIMATION (Gaussian sky + Poisson background),
  NOT a published IceCube/GW search pipeline.
* Known fact: NO high-energy neutrino was coincident with GW170817.
  The canonical IceCube multimessenger event is IceCube-170922A /
  TXS 0506+056 (neutrino + blazar flare), NOT a GW event.
* A low p-value is NOT evidence of new physics or artificial sources.
=============================================================================
"""

from __future__ import annotations

import re
import math
import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.icecube_mm")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import astropy.io.fits as fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

__all__ = ["AlertConfig", "NeutrinoAlert", "GCNCircularClient",
           "CoincidenceEngine", "MultiMessengerBroker"]


@dataclass
class AlertConfig:
    gcn_api: str = "https://gcn.nasa.gov/circulars/api/circulars"
    cache_dir: str = "./data/gcn_cache"
    timeout: int = 60
    time_window_s: float = 1000.0
    neutrino_rate_per_day: float = 1.0
    trials_factor: int = 1
    min_significance_sigma: float = 3.0

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class NeutrinoAlert:
    alert_id: str
    timestamp: float          # GPS seconds
    ra_deg: float
    dec_deg: float
    error_radius_deg: float
    energy_gev: float
    detector: str = "IceCube"
    synthetic: bool = False


# ═══════════════════════════════════════════════════════════════════════
# GCN CLIENT
# ═══════════════════════════════════════════════════════════════════════

class GCNCircularClient:
    """Fetch and parse public GCN circulars for neutrino alerts."""

    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg

    def fetch_circulars(self, limit: int = 50) -> List[Dict]:
        cache = Path(self.cfg.cache_dir) / f"circulars_{limit}.json"
        if cache.exists():
            return json.loads(cache.read_text())

        if not HAS_REQUESTS:
            return self._fixtures()

        try:
            r = requests.get(self.cfg.gcn_api,
                             params={"limit": limit},
                             timeout=self.cfg.timeout,
                             headers={"Accept": "application/json"})
            if r.status_code == 200:
                data = r.json()
                items = data.get("items", data if isinstance(data, list) else [])
                cache.write_text(json.dumps(items))
                return items
        except Exception as e:
            logger.warning(f"GCN fetch failed: {e}")
        return self._fixtures()

    def extract_neutrino_alerts(self, circulars: List[Dict]) -> List[NeutrinoAlert]:
        """Parse IceCube-like alerts from circular text/structured fields."""
        out = []
        for c in circulars:
            text = str(c.get("text", "")) + " " + str(c.get("subject", ""))
            if not re.search(r"icecube|IC2\d{5}|neutrino", text, re.I):
                continue

            ra = self._number(text, r"RA\s*[=:]\s*([+-]?\d+\.?\d*)")
            dec = self._number(text, r"Dec\s*[=:]\s*([+-]?\d+\.?\d*)")
            err = self._number(text, r"(?:Error|Uncertainty)\s*radius\s*[=:]\s*(\d+\.?\d*)")
            if ra is None or dec is None:
                continue

            out.append(NeutrinoAlert(
                alert_id=str(c.get("circular_id", c.get("id", "unknown"))),
                timestamp=float(c.get("time_unix_gps", c.get("timestamp", 0.0))),
                ra_deg=float(ra), dec_deg=float(dec),
                error_radius_deg=float(err or 1.0),
                energy_gev=float(self._number(text, r"energy\s*[=:]\s*([\d.eE+-]+)") or 1e5),
            ))
        return out

    @staticmethod
    def _number(text: str, pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.I)
        return float(m.group(1)) if m else None

    def _fixtures(self) -> List[Dict]:
        """Clearly-labeled OFFLINE test fixtures, not real alerts."""
        logger.warning("Using OFFLINE GCN fixtures (not real alerts)")
        return [
            {"circular_id": "FIXTURE-IC-001", "time_unix_gps": 1187008882.0,
             "subject": "IceCube test alert",
             "text": "IceCube alert: RA = 180.0 Dec = -25.0 Error radius = 1.5 Energy = 2e5 GeV"},
            {"circular_id": "FIXTURE-IC-002", "time_unix_gps": 1187009100.0,
             "subject": "IceCube test alert 2",
             "text": "IceCube alert: RA = 30.0 Dec = 60.0 Error radius = 2.0 Energy = 8e5 GeV"},
        ]


# ═══════════════════════════════════════════════════════════════════════
# SKY LOCALIZATION (FITS when available)
# ═══════════════════════════════════════════════════════════════════════

class SkyLocalization:
    """Gaussian fallback or bayestar FITS probability reading."""

    @staticmethod
    def angular_separation(ra1, dec1, ra2, dec2) -> float:
        r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
        c = (math.sin(d1)*math.sin(d2)
             + math.cos(d1)*math.cos(d2)*math.cos(r1-r2))
        return math.degrees(math.acos(float(np.clip(c, -1., 1.))))

    def probability_within(self, sep_deg: float,
                           sigma_gw_deg: float,
                           sigma_nu_deg: float) -> float:
        """
        Approximate probability that a random sky position lies within
        sep_deg, under a combined Gaussian error model.
        """
        sigma_eff = math.sqrt(sigma_gw_deg**2 + sigma_nu_deg**2) + 1e-6
        return float(-math.expm1(-0.5 * (sep_deg / sigma_eff)**2))


# ═══════════════════════════════════════════════════════════════════════
# COINCIDENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════

class CoincidenceEngine:
    """
    Approximate spatiotemporal coincidence with Poisson background.

    p_coinc = p_spatial * p_temporal
    lambda_bg = R_nu * T_obs * p_coinc * trials
    p_value = 1 - exp(-lambda_bg)
    """

    def __init__(self, cfg: AlertConfig, sky: Optional[SkyLocalization] = None):
        self.cfg, self.sky = cfg, sky or SkyLocalization()

    def evaluate(self, gw_event: Dict, nu: NeutrinoAlert,
                 observation_time_s: float = 3.156e7) -> Dict[str, Any]:
        sep = self.sky.angular_separation(
            gw_event["ra"], gw_event["dec"], nu.ra_deg, nu.dec_deg)
        dt = abs(gw_event["gps_time"] - nu.timestamp)

        p_space = self.sky.probability_within(
            sep, gw_event.get("sigma_deg", 10.0), nu.error_radius_deg)
        p_time = min(1.0, 2.0 * self.cfg.time_window_s / max(observation_time_s, 1.0))
        p_coinc = p_space * p_time

        lambda_bg = (self.cfg.neutrino_rate_per_day * observation_time_s / 86400.0
                     * p_coinc * self.cfg.trials_factor)
        p_value = float(-math.expm1(-lambda_bg))

        significant = (dt <= self.cfg.time_window_s
                       and p_value < 1e-3
                       and sep < 3.0 * nu.error_radius_deg)

        return {
            "alert_id": nu.alert_id,
            "angular_separation_deg": sep,
            "time_difference_s": dt,
            "p_spatial": p_space,
            "p_temporal": p_time,
            "p_coincidence": p_coinc,
            "background_lambda": lambda_bg,
            "p_value": p_value,
            "candidate": significant,
            "note": "Approximate Gaussian+Poisson; NOT a published search pipeline",
        }


# ═══════════════════════════════════════════════════════════════════════
# BROKER
# ═══════════════════════════════════════════════════════════════════════

class MultiMessengerBroker:
    def __init__(self, cfg: Optional[AlertConfig] = None):
        self.cfg = cfg or AlertConfig()
        self.client = GCNCircularClient(self.cfg)
        self.engine = CoincidenceEngine(self.cfg)

    def scan(self, gw_events: List[Dict]) -> List[Dict]:
        circulars = self.client.fetch_circulars()
        alerts = self.client.extract_neutrino_alerts(circulars)
        results = []
        for gw in gw_events:
            for nu in alerts:
                results.append(self.engine.evaluate(gw, nu))
        return results


def smoke_test():
    print("=" * 72)
    print(" IceCube Multi-Messenger Alerts — Smoke Test")
    print("=" * 72)
    broker = MultiMessengerBroker()
    gws = [{"name": "GW170817-fixture", "gps_time": 1187008882.4,
            "ra": 180.0, "dec": -25.0, "sigma_deg": 10.0}]
    results = broker.scan(gws)
    for r in results:
        print(f"  {r['alert_id']}: sep={r['angular_separation_deg']:.2f} deg, "
              f"dt={r['time_difference_s']:.1f} s, "
              f"p={r['p_value']:.2e}, candidate={r['candidate']}")
    print("\n  [HONEST] No high-energy neutrino was coincident with GW170817.")
    print("  [HONEST] Fixtures are OFFLINE TEST DATA, not real alerts.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
