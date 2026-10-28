"""
=============================================================================
SESI ICECUBE MULTI-MESSENGER ALERT CONNECTOR
Real-time neutrino alerts + cross-messenger correlation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DATA SOURCES
------------
[I1] IceCube Real-Time Alerts (public stream)
     https://icecube.wisc.edu/realtime-alerts/
     • HESE (High-Energy Starting Events)
     • GFU (Gamma-Ray Follow-Up) alerts
     • Track-like events (muon neutrinos, best direction)

[I2] GCN (General Coordinates Network, formerly GRB Coordinates Network)
     https://gcn.nasa.gov/
     • VOEvents (XML) — machine-readable alerts
     • Kafka stream — newer format
     • Includes IceCube, Fermi-GBM, Swift, LIGO/Virgo notices

[I3] AMON (Astrophysical Multimessenger Observatory Network)
     https://www.amon.psu.edu/
     • Requires registration for full access
     • Cross-correlation of sub-threshold events

[I4] IceCube Public Data Releases
     • 10-year point-source search
     • 7-year all-sky diffuse map

LANDMARK MULTI-MESSENGER EVENTS [C]
------------------------------------
• IceCube-170922A + TXS 0506+056 (2017): First high-confidence
  neutrino-blazar association. Science 361, 147 (2018).
• GW170817 + GRB 170817A + AT2017gfo (2017): First BNS multi-messenger.
• IceCube-200107A + NGC 1068 correlation (2022).

HONEST SCOPE
------------
• Public alert streams have latency of seconds to minutes.
• Full IceCube alert API requires registration.
• GCN Kafka requires credentials for production use.
• Synthetic fallback generates physically plausible alerts.
• Cross-messenger correlation is STATISTICAL, not causal proof.
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger("sesi.icecube")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__all__ = [
    "IceCubeConfig",
    "NeutrinoAlert",
    "IceCubeAlertConnector",
    "GCNConnector",
    "MultiMessengerCorrelator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IceCubeConfig:
    """Configuration for IceCube data access."""
    cache_dir: str = "./data/icecube_cache"
    timeout: int = 60

    # IceCube public endpoints
    realtime_alerts_url: str = "https://icecube.wisc.edu/realtime-alerts"
    gcn_voevent_url: str = "https://gcn.nasa.gov/voevent"

    # Alert types
    alert_types: List[str] = field(default_factory=lambda: [
        "HESE",          # High-Energy Starting Events
        "GFU",           # Gamma-Ray Follow-Up
        "track",         # Track-like (muon neutrino)
        "cascade",       # Cascade-like (electron/tau neutrino)
    ])

    # Correlation parameters
    time_window_seconds: float = 300.0   # 5 minutes
    angular_window_deg: float = 2.0      # positional coincidence

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class NeutrinoAlert:
    """A neutrino alert event."""
    alert_id: str
    timestamp: float              # Unix time
    ra: float                     # degrees
    dec: float                    # degrees
    angular_uncertainty_deg: float
    energy_gev: float
    alert_type: str               # HESE, GFU, track, cascade
    p_value: float                # chance coincidence probability
    signalness: float             # astrophysical signal probability
    synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mjd(self) -> float:
        """Modified Julian Date."""
        return self.timestamp / 86400.0 + 40587.0


# ═══════════════════════════════════════════════════════════════════════
# [I1] ICECUBE ALERT CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class IceCubeAlertConnector:
    """
    Connector for IceCube real-time alerts.

    HONEST SCOPE:
    • The full alert stream requires IceCube registration.
    • Public notices are available via GCN (see GCNConnector).
    • This connector demonstrates the interface and falls back to
      synthetic alerts when the API is unavailable.
    """

    def __init__(self, cfg: IceCubeConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "alerts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_recent_alerts(self, hours: int = 24) -> List[NeutrinoAlert]:
        """
        Fetch recent neutrino alerts.

        Attempts real API first; falls back to synthetic.
        """
        cache_key = f"alerts_{hours}h_{int(time.time() // 3600)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return [NeutrinoAlert(**a) for a in data]
            except Exception:
                pass

        if not HAS_REQUESTS:
            logger.warning("requests unavailable; synthetic fallback")
            return self._synthetic_alerts(hours)

        try:
            # IceCube real-time alert JSON endpoint
            url = f"{self.cfg.realtime_alerts_url}/alert-stream"
            resp = requests.get(url, timeout=self.cfg.timeout,
                                headers={"User-Agent": "SESI-Research/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                alerts = self._parse_alerts(data)
                cache_file.write_text(json.dumps([a.__dict__ for a in alerts]))
                logger.info(f"Fetched {len(alerts)} real IceCube alerts")
                return alerts
            else:
                logger.warning(f"IceCube API returned {resp.status_code}")
                return self._synthetic_alerts(hours)
        except Exception as e:
            logger.warning(f"IceCube fetch failed: {e}")
            return self._synthetic_alerts(hours)

    def _parse_alerts(self, raw: Dict) -> List[NeutrinoAlert]:
        """Parse raw alert JSON into NeutrinoAlert objects."""
        alerts = []
        for item in raw.get("alerts", []):
            try:
                alerts.append(NeutrinoAlert(
                    alert_id=item.get("id", "unknown"),
                    timestamp=item.get("timestamp", time.time()),
                    ra=item.get("ra", 0.0),
                    dec=item.get("dec", 0.0),
                    angular_uncertainty_deg=item.get("angular_uncertainty", 1.0),
                    energy_gev=item.get("energy_gev", 1e5),
                    alert_type=item.get("alert_type", "unknown"),
                    p_value=item.get("p_value", 0.01),
                    signalness=item.get("signalness", 0.5),
                    synthetic=False,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse alert: {e}")
        return alerts

    def _synthetic_alerts(self, hours: int) -> List[NeutrinoAlert]:
        """
        Generate synthetic alerts following realistic distributions.

        Based on IceCube public statistics:
        • HESE rate: ~10/year above 100 TeV
        • GFU rate: ~1/day above 100 TeV
        • Track fraction: ~30%
        """
        rng = np.random.RandomState(42)
        n_alerts = max(1, int(hours / 24 * 3))  # ~3/day

        alerts = []
        now = time.time()
        for i in range(n_alerts):
            alert_type = rng.choice(["HESE", "GFU", "track", "cascade"],
                                    p=[0.1, 0.3, 0.3, 0.3])
            # Isotropic sky distribution (sin(dec) weighting)
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
            ra = rng.uniform(0, 360)

            # Energy: power-law distribution
            energy = 10 ** rng.uniform(4.5, 7.5)  # 30 TeV to 30 PeV

            alerts.append(NeutrinoAlert(
                alert_id=f"IC_SYN_{int(now)}_{i:03d}",
                timestamp=now - rng.uniform(0, hours * 3600),
                ra=ra,
                dec=dec,
                angular_uncertainty_deg=rng.uniform(0.1, 2.0),
                energy_gev=float(energy),
                alert_type=alert_type,
                p_value=float(rng.uniform(0.001, 0.1)),
                signalness=float(rng.uniform(0.3, 0.95)),
                synthetic=True,
            ))

        logger.info(f"Generated {len(alerts)} synthetic IceCube alerts")
        return alerts


# ═══════════════════════════════════════════════════════════════════════
# [I2] GCN CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class GCNConnector:
    """
    GCN (General Coordinates Network) connector.

    GCN distributes multi-messenger alerts from:
    • IceCube (neutrinos)
    • Fermi-GBM / Swift-BAT (gamma rays)
    • LIGO/Virgo/KAGRA (gravitational waves)
    • Optical surveys (transients)

    HONEST SCOPE:
    • VOEvent XML parsing simplified here.
    • Kafka stream requires gcn-kafka-python package + credentials.
    • This is a demonstration connector.
    """

    def __init__(self, cfg: IceCubeConfig):
        self.cfg = cfg

    def fetch_recent_notices(self, notice_types: Optional[List[str]] = None,
                             hours: int = 24) -> List[Dict]:
        """
        Fetch recent GCN notices.

        Args:
            notice_types: filter by type (e.g., ["ICECUBE", "FERMI GBM"])
            hours: lookback window

        Returns:
            List of notice dicts.
        """
        if not HAS_REQUESTS:
            return self._synthetic_notices(hours)

        try:
            # GCN has a REST-like interface for recent notices
            url = f"{self.cfg.gcn_voevent_url}/recent"
            params = {"hours": hours}
            if notice_types:
                params["types"] = ",".join(notice_types)

            resp = requests.get(url, params=params, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                return resp.json().get("notices", [])
            return self._synthetic_notices(hours)
        except Exception as e:
            logger.warning(f"GCN fetch failed: {e}")
            return self._synthetic_notices(hours)

    def _synthetic_notices(self, hours: int) -> List[Dict]:
        """Synthetic GCN notices for offline testing."""
        rng = np.random.RandomState(43)
        now = time.time()
        notices = []

        # Simulate a mix of notice types
        types = ["ICECUBE", "FERMI GBM", "LVC_PRELIMINARY", "SWIFT"]
        for i in range(5):
            notices.append({
                "notice_id": f"GCN_SYN_{int(now)}_{i:03d}",
                "notice_type": rng.choice(types),
                "timestamp": now - rng.uniform(0, hours * 3600),
                "ra": rng.uniform(0, 360),
                "dec": np.degrees(np.arcsin(rng.uniform(-1, 1))),
                "synthetic": True,
            })

        return notices


# ═══════════════════════════════════════════════════════════════════════
# [I3] MULTI-MESSENGER CORRELATOR
# ═══════════════════════════════════════════════════════════════════════

class MultiMessengerCorrelator:
    """
    [D] Cross-messenger temporal and spatial correlation.

    Given events from multiple messengers (GW, neutrino, EM, cosmic ray),
    finds coincidences within time and angular windows.

    HONEST SCOPE:
    • Coincidence ≠ physical association.
    • Chance coincidence probability must be computed for significance.
    • This is a search tool, not a discovery claim.
    """

    def __init__(self, cfg: IceCubeConfig):
        self.cfg = cfg
        self.events: Dict[str, List[Any]] = {
            "neutrino": [],
            "gravitational_wave": [],
            "electromagnetic": [],
            "cosmic_ray": [],
        }

    def add_events(self, messenger: str, events: List[Any]) -> None:
        """Add events from a messenger."""
        if messenger not in self.events:
            raise ValueError(f"Unknown messenger: {messenger}")
        self.events[messenger].extend(events)
        logger.info(f"Added {len(events)} {messenger} events")

    def angular_separation(self, ra1: float, dec1: float,
                           ra2: float, dec2: float) -> float:
        """
        Great-circle angular separation in degrees.

        Uses the Vincenty formula for numerical stability.
        """
        ra1_r, dec1_r = np.radians(ra1), np.radians(dec1)
        ra2_r, dec2_r = np.radians(ra2), np.radians(dec2)
        dra = ra2_r - ra1_r

        num = np.sqrt(
            (np.cos(dec2_r) * np.sin(dra)) ** 2 +
            (np.cos(dec1_r) * np.sin(dec2_r) -
             np.sin(dec1_r) * np.cos(dec2_r) * np.cos(dra)) ** 2
        )
        den = (np.sin(dec1_r) * np.sin(dec2_r) +
               np.cos(dec1_r) * np.cos(dec2_r) * np.cos(dra))

        return np.degrees(np.arctan2(num, den))

    def find_coincidences(self) -> List[Dict[str, Any]]:
        """
        Find multi-messenger coincidences.

        Returns list of coincidence records.
        """
        coincidences = []

        nu_events = self.events["neutrino"]
        gw_events = self.events["gravitational_wave"]
        em_events = self.events["electromagnetic"]

        # Neutrino-GW coincidences
        for nu in nu_events:
            for gw in gw_events:
                dt = abs(nu.timestamp - gw.timestamp)
                if dt > self.cfg.time_window_seconds:
                    continue

                # GW events may have large sky localization
                gw_ra = getattr(gw, "ra", None)
                gw_dec = getattr(gw, "dec", None)
                if gw_ra is None or gw_ra < 0:
                    continue  # not localized

                d_ang = self.angular_separation(nu.ra, nu.dec, gw_ra, gw_dec)
                # GW localization is typically 10-1000 deg², so use
                # a larger window scaled by uncertainty
                gw_unc = getattr(gw, "error_radius_deg", 10.0)
                effective_window = max(self.cfg.angular_window_deg, gw_unc)

                if d_ang < effective_window:
                    coincidences.append({
                        "type": "neutrino-gw",
                        "neutrino": nu,
                        "gw": gw,
                        "time_diff_s": dt,
                        "angular_sep_deg": d_ang,
                        "chance_probability": self._chance_coincidence(
                            dt, d_ang, effective_window),
                    })

        # Neutrino-EM coincidences
        for nu in nu_events:
            for em in em_events:
                dt = abs(nu.timestamp - em.get("timestamp", 0))
                if dt > self.cfg.time_window_seconds:
                    continue

                d_ang = self.angular_separation(nu.ra, nu.dec,
                                                em.get("ra", 0), em.get("dec", 0))
                if d_ang < self.cfg.angular_window_deg:
                    coincidences.append({
                        "type": "neutrino-em",
                        "neutrino": nu,
                        "em": em,
                        "time_diff_s": dt,
                        "angular_sep_deg": d_ang,
                        "chance_probability": self._chance_coincidence(
                            dt, d_ang, self.cfg.angular_window_deg),
                    })

        # Sort by chance probability (lowest = most significant)
        coincidences.sort(key=lambda c: c["chance_probability"])
        return coincidences

    def _chance_coincidence(self, dt: float, d_ang: float,
                            window_deg: float) -> float:
        """
        [D] Estimate chance coincidence probability.

        P_chance ≈ (time_window / T_obs) × (solid_angle / 4π)

        This is a rough estimate; real analyses use trial factors
        and background estimation from time-shifted data.
        """
        # Solid angle fraction
        omega = 2.0 * math.pi * (1.0 - math.cos(math.radians(window_deg)))
        omega_frac = omega / (4.0 * math.pi)

        # Time fraction (assume 1 year observation)
        t_frac = dt / (365.0 * 86400.0)

        return omega_frac * t_frac * len(self.events["neutrino"])


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI IceCube Multi-Messenger Connector — Smoke Test")
    print("=" * 72)

    cfg = IceCubeConfig()

    # ── [I1] IceCube alerts ─────────────────────────────────────────
    print("\n  ── [I1] IceCube Alerts ──")
    ic = IceCubeAlertConnector(cfg)
    alerts = ic.fetch_recent_alerts(hours=24)
    print(f"    Alerts fetched: {len(alerts)}")
    if alerts:
        a = alerts[0]
        print(f"    First alert: {a.alert_id}")
        print(f"      Type: {a.alert_type}")
        print(f"      Energy: {a.energy_gev:.2e} GeV")
        print(f"      RA/Dec: ({a.ra:.2f}, {a.dec:.2f})")
        print(f"      Synthetic: {a.synthetic}")

    # ── [I2] GCN notices ────────────────────────────────────────────
    print("\n  ── [I2] GCN Notices ──")
    gcn = GCNConnector(cfg)
    notices = gcn.fetch_recent_notices(hours=24)
    print(f"    Notices fetched: {len(notices)}")
    if notices:
        types = set(n.get("notice_type") for n in notices)
        print(f"    Notice types: {types}")

    # ── [I3] Multi-messenger correlation ────────────────────────────
    print("\n  ── [I3] Multi-Messenger Correlation ──")
    correlator = MultiMessengerCorrelator(cfg)
    correlator.add_events("neutrino", alerts)

    # Add synthetic GW events for testing
    from dataclasses import dataclass
    @dataclass
    class MockGWEvent:
        timestamp: float
        ra: float
        dec: float
        error_radius_deg: float = 30.0

    # Place a GW event near one neutrino (if any)
    if alerts:
        mock_gw = MockGWEvent(
            timestamp=alerts[0].timestamp + 10,  # 10s later
            ra=alerts[0].ra + 0.5,
            dec=alerts[0].dec + 0.5,
            error_radius_deg=30.0,
        )
        correlator.add_events("gravitational_wave", [mock_gw])

    coincidences = correlator.find_coincidences()
    print(f"    Coincidences found: {len(coincidences)}")
    if coincidences:
        c = coincidences[0]
        print(f"    Best coincidence:")
        print(f"      Type: {c['type']}")
        print(f"      Δt: {c['time_diff_s']:.1f} s")
        print(f"      Δθ: {c['angular_sep_deg']:.2f}°")
        print(f"      P_chance: {c['chance_probability']:.2e}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS")
    print("  • Full IceCube alert API requires registration")
    print("  • GCN Kafka stream requires credentials")
    print("  • Synthetic alerts used when offline")
    print("  • Coincidence ≠ physical association")
    print("  • Chance probability is rough estimate")
    print("  • Real analysis needs trial factors + background estimation")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
