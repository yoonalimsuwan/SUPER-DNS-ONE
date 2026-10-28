"""
=============================================================================
SESI OBSERVATIONAL DATA CONNECTORS
Real telescope data + Gravitational wave data integration
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DATA SOURCES (all open-access)
------------------------------
[T1] ATLAS (Asteroid Terrestrial-impact Last Alert System)
     https://fallingstar-data.com/forcedphot/
     - Transient alerts, light curves
[T2] ZTF (Zwicky Transient Facility)
     https://www.ztf.caltech.edu/page/dr
     - Public data releases
[T3] Gaia (ESA)
     https://gea.esac.esa.int/archive/
     - Astrometry, photometry
[T4] Pan-STARRS
     https://catalogs.mast.stsci.edu/panstarrs/
     - Deep sky survey
[T5] GWOSC (LIGO/Virgo/KAGRA Open Science Center)
     https://gwosc.org/
     - Gravitational wave strain data (HDF5)

HONEST SCOPE
------------
• These connectors fetch REAL data from public APIs.
• Data quality and availability depend on the source.
• Fallback to synthetic data when offline/unavailable.
• No claim that any detected anomaly is extraterrestrial.
• All results require independent verification.

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

import numpy as np

logger = logging.getLogger("sesi.observational")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "ObservationalConfig",
    "TelescopeDataConnector",
    "ATLASConnector",
    "ZTFConnector",
    "GaiaConnector",
    "GWOSCConnector",
    "ObservationalDataHub",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ObservationalConfig:
    """Configuration for observational data connectors."""
    cache_dir: str = "./data/observational_cache"
    timeout: int = 60
    max_retries: int = 3

    # Telescope queries
    ra_range: Tuple[float, float] = (0.0, 360.0)      # degrees
    dec_range: Tuple[float, float] = (-90.0, 90.0)    # degrees
    time_range: Tuple[float, float] = (2459000.0, 2460000.0)  # JD

    # GW data
    gw_event: str = "GW150914"  # first detected GW event
    gw_sample_rate: int = 4096  # Hz

    # Fallback
    use_synthetic_fallback: bool = True

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# BASE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class TelescopeDataConnector:
    """Base class for telescope data connectors."""

    def __init__(self, cfg: ObservationalConfig, name: str):
        self.cfg = cfg
        self.name = name
        self.cache_dir = Path(cfg.cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, query: Dict) -> str:
        return hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()

    def _get_cached(self, query: Dict) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    def _set_cached(self, query: Dict, data: Dict) -> None:
        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        cache_file.write_text(json.dumps(data, ensure_ascii=False))

    def fetch(self, query: Dict) -> Optional[Dict]:
        """Fetch data from the telescope API."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# [T1] ATLAS CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class ATLASConnector(TelescopeDataConnector):
    """
    ATLAS (Asteroid Terrestrial-impact Last Alert System) connector.

    ATLAS is a robotic telescope system for near-Earth object detection.
    Public API provides transient alerts and light curves.

    HONEST: API structure may change; this is a best-effort connector.
    """

    BASE_URL = "https://fallingstar-data.com/forcedphot"

    def __init__(self, cfg: ObservationalConfig):
        super().__init__(cfg, "atlas")

    def fetch_transients(self, ra: float, dec: float,
                         radius_arcmin: float = 10.0) -> Optional[Dict]:
        """Fetch transient alerts near a sky position."""
        query = {"ra": ra, "dec": dec, "radius": radius_arcmin}
        cached = self._get_cached(query)
        if cached:
            return cached

        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic fallback")
            return self._synthetic_fallback(query)

        try:
            # ATLAS API endpoint (structure may vary)
            url = f"{self.BASE_URL}/query"
            params = {
                "ra": ra,
                "dec": dec,
                "radius": radius_arcmin,
                "format": "json",
            }
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cached(query, data)
                return data
            else:
                logger.warning(f"ATLAS API returned {resp.status_code}")
                return self._synthetic_fallback(query)
        except Exception as e:
            logger.warning(f"ATLAS fetch failed: {e}")
            return self._synthetic_fallback(query)

    def _synthetic_fallback(self, query: Dict) -> Dict:
        """Generate synthetic transient data for testing."""
        rng = np.random.RandomState(42)
        n_transients = rng.randint(5, 20)
        transients = []
        for i in range(n_transients):
            transients.append({
                "id": f"ATLAS_SYN_{i:04d}",
                "ra": query["ra"] + rng.uniform(-0.1, 0.1),
                "dec": query["dec"] + rng.uniform(-0.1, 0.1),
                "mjd": rng.uniform(59000, 60000),
                "mag": rng.uniform(15, 22),
                "mag_err": rng.uniform(0.01, 0.1),
                "filter": rng.choice(["orange", "cyan"]),
            })
        return {"transients": transients, "synthetic": True}


# ═══════════════════════════════════════════════════════════════════════
# [T2] ZTF CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class ZTFConnector(TelescopeDataConnector):
    """
    ZTF (Zwicky Transient Facility) connector.

    ZTF is an optical time-domain survey at Palomar Observatory.
    Public data releases available via IRSA.

    HONEST: Full data access may require registration.
    """

    BASE_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

    def __init__(self, cfg: ObservationalConfig):
        super().__init__(cfg, "ztf")

    def fetch_light_curve(self, object_id: str) -> Optional[Dict]:
        """Fetch light curve for a specific ZTF object."""
        query = {"object_id": object_id}
        cached = self._get_cached(query)
        if cached:
            return cached

        if not HAS_REQUESTS:
            return self._synthetic_fallback(query)

        try:
            url = f"{self.BASE_URL}?object_id={object_id}&format=json"
            resp = requests.get(url, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cached(query, data)
                return data
            else:
                return self._synthetic_fallback(query)
        except Exception as e:
            logger.warning(f"ZTF fetch failed: {e}")
            return self._synthetic_fallback(query)

    def _synthetic_fallback(self, query: Dict) -> Dict:
        """Generate synthetic light curve."""
        rng = np.random.RandomState(hash(query["object_id"]) % (2**31))
        n_points = 100
        mjd = np.linspace(59000, 59100, n_points)
        # Simulate a variable star with noise
        period = rng.uniform(1, 10)
        mag = 18.0 + 0.5 * np.sin(2 * np.pi * mjd / period) + rng.normal(0, 0.05, n_points)
        mag_err = np.full(n_points, 0.05)
        return {
            "object_id": query["object_id"],
            "mjd": mjd.tolist(),
            "mag": mag.tolist(),
            "mag_err": mag_err.tolist(),
            "synthetic": True,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T3] GAIA CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class GaiaConnector(TelescopeDataConnector):
    """
    Gaia (ESA) connector.

    Gaia provides astrometry, photometry, and radial velocities for
    over 1 billion stars.

    HONEST: Gaia archive requires ADQL queries; this is a simplified interface.
    """

    BASE_URL = "https://gea.esac.esa.int/tap-server/tap"

    def __init__(self, cfg: ObservationalConfig):
        super().__init__(cfg, "gaia")

    def query_region(self, ra: float, dec: float,
                     radius_deg: float = 0.1) -> Optional[Dict]:
        """Query Gaia sources in a circular region."""
        query = {"ra": ra, "dec": dec, "radius": radius_deg}
        cached = self._get_cached(query)
        if cached:
            return cached

        if not HAS_REQUESTS:
            return self._synthetic_fallback(query)

        try:
            # ADQL query for Gaia DR3
            adql = f"""
            SELECT TOP 100 source_id, ra, dec, phot_g_mean_mag, parallax
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
            ) = 1
            """
            url = f"{self.BASE_URL}/sync"
            params = {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "json",
                "QUERY": adql,
            }
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                data = resp.json()
                self._set_cached(query, data)
                return data
            else:
                return self._synthetic_fallback(query)
        except Exception as e:
            logger.warning(f"Gaia query failed: {e}")
            return self._synthetic_fallback(query)

    def _synthetic_fallback(self, query: Dict) -> Dict:
        """Generate synthetic Gaia-like catalog."""
        rng = np.random.RandomState(42)
        n_sources = 50
        sources = []
        for i in range(n_sources):
            sources.append({
                "source_id": f"GAIA_SYN_{i:010d}",
                "ra": query["ra"] + rng.uniform(-query["radius"], query["radius"]),
                "dec": query["dec"] + rng.uniform(-query["radius"], query["radius"]),
                "phot_g_mean_mag": rng.uniform(10, 20),
                "parallax": rng.uniform(0.1, 10.0),  # mas
            })
        return {"sources": sources, "synthetic": True}


# ═══════════════════════════════════════════════════════════════════════
# [T5] GWOSC CONNECTOR (Gravitational Wave)
# ═══════════════════════════════════════════════════════════════════════

class GWOSCConnector:
    """
    GWOSC (LIGO/Virgo/KAGRA Open Science Center) connector.

    Provides gravitational wave strain data in HDF5 format.
    Events include GW150914, GW151226, GW170817, etc.

    HONEST: Requires h5py for HDF5 parsing.
    """

    BASE_URL = "https://gwosc.org/archive"

    def __init__(self, cfg: ObservationalConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "gwosc"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_event(self, event_name: str = "GW150914",
                    detector: str = "H1") -> Optional[Dict]:
        """
        Fetch strain data for a GW event.

        Args:
            event_name: GW event name (e.g., "GW150914")
            detector: "H1" (Hanford), "L1" (Livingston), or "V1" (Virgo)

        Returns:
            dict with strain data, sample rate, GPS time
        """
        cache_file = self.cache_dir / f"{event_name}_{detector}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        if not HAS_REQUESTS:
            return self._synthetic_fallback(event_name, detector)

        try:
            # GWOSC API for event metadata
            url = f"{self.cfg.gwosc_base_url}/{event_name}/filelist/"
            resp = requests.get(url, timeout=self.cfg.timeout)
            if resp.status_code != 200:
                return self._synthetic_fallback(event_name, detector)

            filelist = resp.json()
            # Find HDF5 file for the requested detector
            hdf5_url = None
            for f in filelist.get("files", []):
                if detector in f.get("detector", "") and f["filename"].endswith(".hdf5"):
                    hdf5_url = f["url"]
                    break

            if not hdf5_url:
                return self._synthetic_fallback(event_name, detector)

            # Download HDF5 file
            hdf5_cache = self.cache_dir / f"{event_name}_{detector}.hdf5"
            if not hdf5_cache.exists():
                hdf5_resp = requests.get(hdf5_url, timeout=300)
                hdf5_cache.write_bytes(hdf5_resp.content)

            # Parse HDF5
            if HAS_H5PY:
                with h5py.File(hdf5_cache, "r") as f:
                    strain = f["strain/Strain"][:]
                    sample_rate = f["strain/Strain"].attrs["Xspacing"]
                    gps_start = f["strain/Strain"].attrs["Xstart"]
                data = {
                    "strain": strain.tolist(),
                    "sample_rate": float(sample_rate),
                    "gps_start": float(gps_start),
                    "event": event_name,
                    "detector": detector,
                    "synthetic": False,
                }
                cache_file.write_text(json.dumps(data))
                return data
            else:
                return self._synthetic_fallback(event_name, detector)

        except Exception as e:
            logger.warning(f"GWOSC fetch failed: {e}")
            return self._synthetic_fallback(event_name, detector)

    def _synthetic_fallback(self, event_name: str, detector: str) -> Dict:
        """Generate synthetic GW strain data (chirp-like signal)."""
        rng = np.random.RandomState(hash(event_name + detector) % (2**31))
        sample_rate = self.cfg.gw_sample_rate
        duration = 1.0  # seconds
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # Simulate a chirp signal (simplified inspiral)
        f0 = 35.0  # initial frequency (Hz)
        f1 = 250.0  # final frequency
        chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / duration))

        # Add noise
        noise = rng.normal(0, 1e-21, n_samples)
        strain = chirp * 1e-21 + noise

        return {
            "strain": strain.tolist(),
            "sample_rate": sample_rate,
            "gps_start": 1126259462.4,  # GW150914 GPS time
            "event": event_name,
            "detector": detector,
            "synthetic": True,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED OBSERVATIONAL DATA HUB
# ═══════════════════════════════════════════════════════════════════════

class ObservationalDataHub:
    """
    Unified interface to all observational data sources.

    Usage:
        hub = ObservationalDataHub(cfg)
        atlas_data = hub.fetch_atlas(ra=180.0, dec=45.0)
        gw_data = hub.fetch_gw_event("GW150914")
    """

    def __init__(self, cfg: Optional[ObservationalConfig] = None):
        self.cfg = cfg or ObservationalConfig()
        self.atlas = ATLASConnector(self.cfg)
        self.ztf = ZTFConnector(self.cfg)
        self.gaia = GaiaConnector(self.cfg)
        self.gwosc = GWOSCConnector(self.cfg)

    def fetch_atlas(self, ra: float, dec: float, **kwargs) -> Dict:
        """Fetch ATLAS transient data."""
        return self.atlas.fetch_transients(ra, dec, **kwargs)

    def fetch_ztf_light_curve(self, object_id: str) -> Dict:
        """Fetch ZTF light curve."""
        return self.ztf.fetch_light_curve(object_id)

    def fetch_gaia_region(self, ra: float, dec: float, **kwargs) -> Dict:
        """Query Gaia sources in a region."""
        return self.gaia.query_region(ra, dec, **kwargs)

    def fetch_gw_event(self, event_name: str, detector: str = "H1") -> Dict:
        """Fetch GW strain data."""
        return self.gwosc.fetch_event(event_name, detector)

    def summary(self) -> Dict[str, Any]:
        """Summary of available data sources."""
        return {
            "telescopes": ["ATLAS", "ZTF", "Gaia"],
            "gw_detectors": ["LIGO H1/L1", "Virgo V1", "KAGRA K1"],
            "cache_dir": str(self.cfg.cache_dir),
            "has_requests": HAS_REQUESTS,
            "has_h5py": HAS_H5PY,
            "honest_note": (
                "Data availability depends on network and API stability. "
                "Synthetic fallback used when offline."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Observational Data Connectors — Smoke Test")
    print("=" * 72)

    cfg = ObservationalConfig()
    hub = ObservationalDataHub(cfg)

    # Test 1: Summary
    print("\n  ── Data Source Summary ──")
    summary = hub.summary()
    print(f"    Telescopes: {', '.join(summary['telescopes'])}")
    print(f"    GW detectors: {', '.join(summary['gw_detectors'])}")
    print(f"    HTTP available: {summary['has_requests']}")
    print(f"    HDF5 available: {summary['has_h5py']}")

    # Test 2: ATLAS (synthetic fallback)
    print("\n  ── [T1] ATLAS Transients ──")
    atlas_data = hub.fetch_atlas(ra=180.0, dec=45.0)
    print(f"    Transients found: {len(atlas_data.get('transients', []))}")
    print(f"    Synthetic: {atlas_data.get('synthetic', False)}")

    # Test 3: ZTF light curve
    print("\n  ── [T2] ZTF Light Curve ──")
    ztf_data = hub.fetch_ztf_light_curve("ZTF21aaqjwmg")
    print(f"    Data points: {len(ztf_data.get('mjd', []))}")
    print(f"    Synthetic: {ztf_data.get('synthetic', False)}")

    # Test 4: Gaia region query
    print("\n  ── [T3] Gaia Region Query ──")
    gaia_data = hub.fetch_gaia_region(ra=180.0, dec=45.0, radius_deg=0.1)
    print(f"    Sources found: {len(gaia_data.get('sources', []))}")
    print(f"    Synthetic: {gaia_data.get('synthetic', False)}")

    # Test 5: GW event
    print("\n  ── [T5] GWOSC Event (GW150914) ──")
    gw_data = hub.fetch_gw_event("GW150914", detector="H1")
    print(f"    Strain samples: {len(gw_data.get('strain', []))}")
    print(f"    Sample rate: {gw_data.get('sample_rate')} Hz")
    print(f"    Synthetic: {gw_data.get('synthetic', False)}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Real data requires network + may need API keys")
    print("  • Synthetic data used as fallback when offline")
    print("  • No claim that any detected anomaly is extraterrestrial")
    print("  • All results require independent verification")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
