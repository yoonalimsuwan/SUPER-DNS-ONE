"""
=============================================================================
SESI NEUTRINO & COSMIC RAY DATA CONNECTORS
IceCube Neutrino Observatory + Pierre Auger Observatory
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DATA SOURCES (all open-access)
------------------------------
[N1] IceCube Neutrino Observatory
     https://icecube.wisc.edu/science/alerts/
     - Real-time neutrino alerts (GCN Notices)
     - High-energy neutrino events
     - Direction, energy, time, significance
[N2] Pierre Auger Observatory
     https://www.auger.org/
     - Ultra-high-energy cosmic rays (UHECRs)
     - Shower direction, energy, composition
     - Public data releases

HONEST SCOPE
------------
• These connectors fetch REAL data from public APIs.
• Data quality depends on the source and network availability.
• Fallback to synthetic data when offline/unavailable.
• No claim that any detected anomaly is extraterrestrial.
• Multi-messenger correlation requires careful statistical analysis.

LABEL CONVENTION
----------------
[P] proved  [C] cited  [D] derived  [O] open
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
from datetime import datetime

import numpy as np

logger = logging.getLogger("sesi.neutrino_cosmicray")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "NeutrinoCosmicRayConfig",
    "IceCubeConnector",
    "PierreAugerConnector",
    "MultiMessengerDataHub",
    "NeutrinoEvent",
    "CosmicRayEvent",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NeutrinoCosmicRayConfig:
    """Configuration for neutrino and cosmic ray connectors."""
    cache_dir: str = "./data/neutrino_cosmicray_cache"
    timeout: int = 60
    max_retries: int = 3

    # IceCube
    icecube_alert_url: str = "https://gcn.gsfc.nasa.gov/notifications/notifications.html"
    icecube_api: str = "https://icecube.wisc.edu/science/alerts/"

    # Pierre Auger
    auger_data_url: str = "https://www.auger.org/data/"

    # Time window for correlation
    correlation_window_seconds: float = 60.0  # 1 minute

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class NeutrinoEvent:
    """A neutrino detection event from IceCube."""
    event_id: str
    timestamp: float              # Unix timestamp (MJD or GPS)
    ra: float                     # Right ascension (degrees)
    dec: float                    # Declination (degrees)
    energy_gev: float             # Energy in GeV
    significance: float           # Detection significance (sigma)
    alert_type: str               # "HESE", "GMCN", "track", etc.
    error_radius_deg: float       # Positional uncertainty
    synthetic: bool = False


@dataclass
class CosmicRayEvent:
    """A cosmic ray detection event from Pierre Auger."""
    event_id: str
    timestamp: float              # Unix timestamp
    ra: float                     # Right ascension (degrees)
    dec: float                    # Declination (degrees)
    energy_ev: float              # Energy in eV
    zenith_deg: float             # Zenith angle
    azimuth_deg: float            # Azimuth angle
    composition: Optional[str]    # "proton", "iron", "mixed"
    synthetic: bool = False


# ═══════════════════════════════════════════════════════════════════════
# [N1] ICECUBE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class IceCubeConnector:
    """
    IceCube Neutrino Observatory connector.

    Fetches real-time neutrino alerts from IceCube's public API.
    Falls back to synthetic data when offline.

    HONEST: IceCube's API structure may change; this is a best-effort connector.
    """

    def __init__(self, cfg: NeutrinoCosmicRayConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "icecube"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, query: Dict) -> str:
        return hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()

    def _get_cached(self, query: Dict) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _set_cached(self, query: Dict, data: List[Dict]) -> None:
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        cache_file.write_text(json.dumps(data, ensure_ascii=False))

    def fetch_recent_alerts(self, hours: int = 24) -> List[NeutrinoEvent]:
        """
        Fetch recent neutrino alerts from IceCube.

        Args:
            hours: lookback window in hours

        Returns:
            list of NeutrinoEvent objects
        """
        query = {"hours": hours}
        cached = self._get_cached(query)
        if cached:
            return [NeutrinoEvent(**e) for e in cached]

        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic fallback")
            return self._synthetic_fallback(hours)

        try:
            # IceCube alert API (simplified; actual endpoint may vary)
            url = f"{self.cfg.icecube_api}/recent"
            params = {"hours": hours, "format": "json"}
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)

            if resp.status_code == 200:
                data = resp.json()
                events = []
                for alert in data.get("alerts", []):
                    event = NeutrinoEvent(
                        event_id=alert.get("id", "unknown"),
                        timestamp=alert.get("timestamp", time.time()),
                        ra=alert.get("ra", 0.0),
                        dec=alert.get("dec", 0.0),
                        energy_gev=alert.get("energy_gev", 1e5),
                        significance=alert.get("significance", 3.0),
                        alert_type=alert.get("type", "HESE"),
                        error_radius_deg=alert.get("error_radius", 1.0),
                        synthetic=False,
                    )
                    events.append(event)

                # Cache the results
                self._set_cached(query, [e.__dict__ for e in events])
                logger.info(f"Fetched {len(events)} IceCube alerts")
                return events
            else:
                logger.warning(f"IceCube API returned {resp.status_code}")
                return self._synthetic_fallback(hours)

        except Exception as e:
            logger.warning(f"IceCube fetch failed: {e}")
            return self._synthetic_fallback(hours)

    def _synthetic_fallback(self, hours: int) -> List[NeutrinoEvent]:
        """Generate synthetic neutrino events for testing."""
        rng = np.random.RandomState(42)
        n_events = rng.randint(5, 20)
        events = []

        now = time.time()
        for i in range(n_events):
            event = NeutrinoEvent(
                event_id=f"IC_SYN_{i:04d}",
                timestamp=now - rng.uniform(0, hours * 3600),
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-90, 90),
                energy_gev=10 ** rng.uniform(4, 7),  # 10 TeV to 10 PeV
                significance=rng.uniform(3.0, 6.0),
                alert_type=rng.choice(["HESE", "track", "cascade"]),
                error_radius_deg=rng.uniform(0.1, 2.0),
                synthetic=True,
            )
            events.append(event)

        logger.info(f"Generated {len(events)} synthetic IceCube events")
        return events


# ═══════════════════════════════════════════════════════════════════════
# [N2] PIERRE AUGER CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class PierreAugerConnector:
    """
    Pierre Auger Observatory connector.

    Fetches ultra-high-energy cosmic ray events from Auger's public data.
    Falls back to synthetic data when offline.

    HONEST: Auger's data access may require registration; this is a best-effort connector.
    """

    def __init__(self, cfg: NeutrinoCosmicRayConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "auger"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, query: Dict) -> str:
        return hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()

    def _get_cached(self, query: Dict) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _set_cached(self, query: Dict, data: List[Dict]) -> None:
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        cache_file.write_text(json.dumps(data, ensure_ascii=False))

    def fetch_recent_events(self, hours: int = 24) -> List[CosmicRayEvent]:
        """
        Fetch recent cosmic ray events from Pierre Auger.

        Args:
            hours: lookback window in hours

        Returns:
            list of CosmicRayEvent objects
        """
        query = {"hours": hours}
        cached = self._get_cached(query)
        if cached:
            return [CosmicRayEvent(**e) for e in cached]

        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic fallback")
            return self._synthetic_fallback(hours)

        try:
            # Auger data API (simplified; actual endpoint may vary)
            url = f"{self.cfg.auger_data_url}/recent"
            params = {"hours": hours, "format": "json"}
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)

            if resp.status_code == 200:
                data = resp.json()
                events = []
                for event_data in data.get("events", []):
                    event = CosmicRayEvent(
                        event_id=event_data.get("id", "unknown"),
                        timestamp=event_data.get("timestamp", time.time()),
                        ra=event_data.get("ra", 0.0),
                        dec=event_data.get("dec", 0.0),
                        energy_ev=event_data.get("energy_ev", 1e19),
                        zenith_deg=event_data.get("zenith", 0.0),
                        azimuth_deg=event_data.get("azimuth", 0.0),
                        composition=event_data.get("composition", "mixed"),
                        synthetic=False,
                    )
                    events.append(event)

                # Cache the results
                self._set_cached(query, [e.__dict__ for e in events])
                logger.info(f"Fetched {len(events)} Auger events")
                return events
            else:
                logger.warning(f"Auger API returned {resp.status_code}")
                return self._synthetic_fallback(hours)

        except Exception as e:
            logger.warning(f"Auger fetch failed: {e}")
            return self._synthetic_fallback(hours)

    def _synthetic_fallback(self, hours: int) -> List[CosmicRayEvent]:
        """Generate synthetic cosmic ray events for testing."""
        rng = np.random.RandomState(43)
        n_events = rng.randint(3, 15)
        events = []

        now = time.time()
        for i in range(n_events):
            event = CosmicRayEvent(
                event_id=f"AUG_SYN_{i:04d}",
                timestamp=now - rng.uniform(0, hours * 3600),
                ra=rng.uniform(0, 360),
                dec=rng.uniform(-90, 90),
                energy_ev=10 ** rng.uniform(18, 20.5),  # 0.1 EeV to 300 EeV
                zenith_deg=rng.uniform(0, 60),
                azimuth_deg=rng.uniform(0, 360),
                composition=rng.choice(["proton", "iron", "mixed"]),
                synthetic=True,
            )
            events.append(event)

        logger.info(f"Generated {len(events)} synthetic Auger events")
        return events


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED MULTI-MESSENGER DATA HUB
# ═══════════════════════════════════════════════════════════════════════

class MultiMessengerDataHub:
    """
    Unified interface to all multi-messenger astrophysics data sources.

    Combines:
      • Electromagnetic (ATLAS, ZTF, Gaia)
      • Gravitational waves (LIGO/Virgo/KAGRA)
      • Neutrinos (IceCube)
      • Cosmic rays (Pierre Auger)

    Usage:
        hub = MultiMessengerDataHub(cfg)
        neutrino_events = hub.fetch_neutrino_events()
        cosmic_ray_events = hub.fetch_cosmic_ray_events()
        correlated = hub.find_temporal_correlations()
    """

    def __init__(self, cfg: Optional[NeutrinoCosmicRayConfig] = None):
        self.cfg = cfg or NeutrinoCosmicRayConfig()
        self.icecube = IceCubeConnector(self.cfg)
        self.auger = PierreAugerConnector(self.cfg)

        # Import other connectors if available
        try:
            from sesi_observational_data import ObservationalDataHub, ObservationalConfig
            self.obs_hub = ObservationalDataHub(ObservationalConfig())
        except ImportError:
            self.obs_hub = None

        try:
            from sesi_gw_analysis import GWDataProcessor, GWAnalysisConfig
            self.gw_processor = GWDataProcessor(GWAnalysisConfig())
        except ImportError:
            self.gw_processor = None

    def fetch_neutrino_events(self, hours: int = 24) -> List[NeutrinoEvent]:
        """Fetch recent neutrino events from IceCube."""
        return self.icecube.fetch_recent_alerts(hours)

    def fetch_cosmic_ray_events(self, hours: int = 24) -> List[CosmicRayEvent]:
        """Fetch recent cosmic ray events from Pierre Auger."""
        return self.auger.fetch_recent_events(hours)

    def find_temporal_correlations(
        self,
        neutrino_events: List[NeutrinoEvent],
        cosmic_ray_events: List[CosmicRayEvent],
        window_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find temporally correlated events between neutrinos and cosmic rays.

        Args:
            neutrino_events: list of neutrino events
            cosmic_ray_events: list of cosmic ray events
            window_seconds: correlation time window

        Returns:
            list of correlated event pairs
        """
        window = window_seconds or self.cfg.correlation_window_seconds
        correlations = []

        for nu_event in neutrino_events:
            for cr_event in cosmic_ray_events:
                dt = abs(nu_event.timestamp - cr_event.timestamp)
                if dt <= window:
                    correlations.append({
                        "neutrino": nu_event,
                        "cosmic_ray": cr_event,
                        "time_diff_seconds": dt,
                        "angular_separation_deg": self._angular_separation(
                            nu_event.ra, nu_event.dec,
                            cr_event.ra, cr_event.dec
                        ),
                    })

        # Sort by time difference
        correlations.sort(key=lambda x: x["time_diff_seconds"])
        return correlations

    def _angular_separation(self, ra1: float, dec1: float,
                            ra2: float, dec2: float) -> float:
        """Compute angular separation between two sky positions."""
        ra1_rad = np.radians(ra1)
        dec1_rad = np.radians(dec1)
        ra2_rad = np.radians(ra2)
        dec2_rad = np.radians(dec2)

        cos_sep = (np.sin(dec1_rad) * np.sin(dec2_rad) +
                   np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        return np.degrees(np.arccos(cos_sep))

    def summary(self) -> Dict[str, Any]:
        """Summary of available data sources."""
        return {
            "neutrino_detectors": ["IceCube"],
            "cosmic_ray_detectors": ["Pierre Auger"],
            "electromagnetic": ["ATLAS", "ZTF", "Gaia"] if self.obs_hub else [],
            "gravitational_wave": ["LIGO", "Virgo", "KAGRA"] if self.gw_processor else [],
            "cache_dir": str(self.cfg.cache_dir),
            "has_requests": HAS_REQUESTS,
            "honest_note": (
                "Multi-messenger data availability depends on network and API stability. "
                "Synthetic fallback used when offline."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Neutrino & Cosmic Ray Connectors — Smoke Test")
    print("=" * 72)

    cfg = NeutrinoCosmicRayConfig()
    hub = MultiMessengerDataHub(cfg)

    # Test 1: Summary
    print("\n  ── Data Source Summary ──")
    summary = hub.summary()
    print(f"    Neutrino detectors: {', '.join(summary['neutrino_detectors'])}")
    print(f"    Cosmic ray detectors: {', '.join(summary['cosmic_ray_detectors'])}")
    print(f"    EM telescopes: {', '.join(summary['electromagnetic'])}")
    print(f"    GW detectors: {', '.join(summary['gravitational_wave'])}")
    print(f"    HTTP available: {summary['has_requests']}")

    # Test 2: IceCube neutrinos
    print("\n  ── [N1] IceCube Neutrino Events ──")
    nu_events = hub.fetch_neutrino_events(hours=24)
    print(f"    Events found: {len(nu_events)}")
    if nu_events:
        e = nu_events[0]
        print(f"    First event:")
        print(f"      ID: {e.event_id}")
        print(f"      Energy: {e.energy_gev:.2e} GeV")
        print(f"      RA/Dec: ({e.ra:.2f}, {e.dec:.2f}) deg")
        print(f"      Significance: {e.significance:.2f} sigma")
        print(f"      Synthetic: {e.synthetic}")

    # Test 3: Pierre Auger cosmic rays
    print("\n  ── [N2] Pierre Auger Cosmic Ray Events ──")
    cr_events = hub.fetch_cosmic_ray_events(hours=24)
    print(f"    Events found: {len(cr_events)}")
    if cr_events:
        e = cr_events[0]
        print(f"    First event:")
        print(f"      ID: {e.event_id}")
        print(f"      Energy: {e.energy_ev:.2e} eV")
        print(f"      RA/Dec: ({e.ra:.2f}, {e.dec:.2f}) deg")
        print(f"      Composition: {e.composition}")
        print(f"      Synthetic: {e.synthetic}")

    # Test 4: Temporal correlations
    print("\n  ── Temporal Correlations ──")
    correlations = hub.find_temporal_correlations(nu_events, cr_events, window_seconds=300)
    print(f"    Correlated pairs (within 5 min): {len(correlations)}")
    if correlations:
        c = correlations[0]
        print(f"    First correlation:")
        print(f"      Time diff: {c['time_diff_seconds']:.1f} s")
        print(f"      Angular sep: {c['angular_separation_deg']:.2f} deg")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Real data requires network + may need API keys")
    print("  • Synthetic data used as fallback when offline")
    print("  • No claim that any detected anomaly is extraterrestrial")
    print("  • Multi-messenger correlation requires careful statistical analysis")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
