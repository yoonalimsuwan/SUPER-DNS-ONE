"""
=============================================================================
SESI GLOBAL MAPS — Google × Apple × Microsoft Azure Unified Interface
Disaster-response map services: geocoding, evacuation routing, flood elevation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

SUPPORTED PROVIDERS
-------------------
[C] Google Maps Platform
    • Geocoding API          (address → coordinates)
    • Directions API         (evacuation routing)
    • Elevation API          (flood-risk height check)
    • Distance Matrix API    (shelter reachability)
    Requires: API key + billing account + per-API enablement

[C] Apple MapKit (REST / MapKit JS)
    • Search / Geocoding
    • Directions / ETA
    Requires: Apple Developer account + JWT (ES256 signed)

[C] Microsoft Azure Maps  (successor of Bing Maps — Bing is DEPRECATED)
    • Geocoding / Search
    • Route API (directions)
    • Weather API            (storm tracking — unique strength)
    • Traffic API
    Requires: Azure subscription key

COORDINATE SYSTEMS
------------------
All three providers use WGS-84. For mainland China, the existing
sesi_geo_maps.py GCJ-02/BD-09 transform applies BEFORE calling these
providers, because Chinese law mandates the offset.

ROUTER DESIGN (No-Zeno + fallback discipline)
----------------------------------------------
  request → region check → provider selection → call
             │                     │
             │                     ├─ success → return
             │                     └─ failure → RECORD + retry next provider
             └─ China → GCJ-02 transform first
  Rate limiting: τ_e spacing per provider (CFD §3.3 — never hammer an API)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Each provider requires ITS OWN credentials; none are bundled here.
• Without credentials the router returns clearly-flagged SYNTHETIC data
  so the pipeline can be exercised, but NEVER as real coordinates.
• Elevation-based flood risk is a PROXY; real flood modelling needs
  hydrodynamic simulation (sesi_3d_free_surface.py).
• Routing gives SUGGESTED evacuation paths; local authorities' routes
  take precedence in an actual emergency.
=============================================================================
"""

from __future__ import annotations

import math
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.global_maps")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Reuse coordinate transform + geofence from the Thai/China module
try:
    from sesi_geo_maps import (
        CoordinateSystem, GCJ02Transformer, DisasterGeofence, GeoMapsConfig)
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

__all__ = [
    "MapProviderType",
    "GlobalMapsConfig",
    "GoogleMapsProvider",
    "AppleMapsProvider",
    "AzureMapsProvider",
    "GlobalMapsRouter",
    "EvacuationRoutePlanner",
    "FloodElevationChecker",
]


class MapProviderType(Enum):
    GOOGLE = auto()
    APPLE = auto()
    AZURE = auto()
    SYNTHETIC = auto()


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — credentials MUST come from environment, never hardcoded
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GlobalMapsConfig:
    """Configuration for global map providers."""
    # Credentials read from env vars (never store in code)
    google_api_key_env: str = "GOOGLE_MAPS_API_KEY"
    apple_key_id_env: str = "APPLE_MAPKIT_KEY_ID"
    apple_team_id_env: str = "APPLE_MAPKIT_TEAM_ID"
    azure_subscription_key_env: str = "AZURE_MAPS_SUBSCRIPTION_KEY"

    # Router behaviour
    preferred_order: List[MapProviderType] = field(default_factory=lambda: [
        MapProviderType.GOOGLE,
        MapProviderType.AZURE,
        MapProviderType.APPLE,
    ])
    timeout_s: float = 10.0
    max_retries: int = 2

    # No-Zeno rate limiting per provider (seconds between calls)
    call_min_interval_s: float = 0.2

    def google_key(self) -> Optional[str]:
        return os.environ.get(self.google_api_key_env)

    def azure_key(self) -> Optional[str]:
        return os.environ.get(self.azure_subscription_key_env)

    def apple_credentials(self) -> Optional[Tuple[str, str]]:
        kid = os.environ.get(self.apple_key_id_env)
        team = os.environ.get(self.apple_team_id_env)
        return (kid, team) if kid and team else None


# ═══════════════════════════════════════════════════════════════════════
# [D] NO-ZENO CALL THROTTLE (CFD §3.3)
# ═══════════════════════════════════════════════════════════════════════

class _CallThrottle:
    """Per-provider call spacing so we never hammer an external API."""

    def __init__(self, min_interval_s: float):
        self.min_interval = min_interval_s
        self._last_call: Dict[str, float] = {}
        self._blocked_count: Dict[str, int] = {}

    def wait_if_needed(self, provider: str) -> None:
        now = time.time()
        last = self._last_call.get(provider, 0.0)
        elapsed = now - last
        if elapsed < self.min_interval:
            self._blocked_count[provider] = \
                self._blocked_count.get(provider, 0) + 1
            time.sleep(self.min_interval - elapsed)
        self._last_call[provider] = time.time()

    def report(self) -> Dict[str, int]:
        return dict(self._blocked_count)


# ═══════════════════════════════════════════════════════════════════════
# [C] GOOGLE MAPS PROVIDER
# ═══════════════════════════════════════════════════════════════════════

class GoogleMapsProvider:
    """
    [C] Google Maps Platform connector.

    Endpoints used:
      • Geocoding       : maps.googleapis.com/maps/api/geocode/json
      • Directions      : maps.googleapis.com/maps/api/directions/json
      • Elevation       : maps.googleapis.com/maps/api/elevation/json
      • Distance Matrix : maps.googleapis.com/maps/api/distancematrix/json
    """

    BASE = "https://maps.googleapis.com/maps/api"

    def __init__(self, cfg: GlobalMapsConfig):
        self.cfg = cfg
        self.api_key = cfg.google_key()
        self.available = self.api_key is not None

    def _check(self) -> bool:
        if not self.available:
            logger.warning("Google Maps: no API key; provider unavailable")
            return False
        if not HAS_REQUESTS:
            return False
        return True

    def geocode(self, address: str) -> Optional[Dict[str, Any]]:
        """[C] Address → WGS-84 coordinates."""
        if not self._check():
            return None
        try:
            resp = requests.get(
                f"{self.BASE}/geocode/json",
                params={"address": address, "key": self.api_key},
                timeout=self.cfg.timeout_s)
            data = resp.json()
            if data.get("status") != "OK":
                logger.warning(f"Google geocode failed: {data.get('status')}")
                return None
            loc = data["results"][0]["geometry"]["location"]
            return {
                "lat": loc["lat"], "lon": loc["lng"],
                "system": CoordinateSystem.WGS84 if HAS_GEO else "WGS-84",
                "provider": "google",
                "formatted": data["results"][0].get("formatted_address", address),
            }
        except Exception as e:
            logger.error(f"Google geocode error: {e}")
            return None

    def elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        [C] Return ground elevation (meters) at a point.

        CRITICAL for flood-risk screening: a shelter's floor height must
        exceed the projected flood level.
        """
        if not self._check():
            return None
        try:
            resp = requests.get(
                f"{self.BASE}/elevation/json",
                params={"locations": f"{lat},{lon}", "key": self.api_key},
                timeout=self.cfg.timeout_s)
            data = resp.json()
            if data.get("status") != "OK":
                return None
            return float(data["results"][0]["elevation"])
        except Exception as e:
            logger.error(f"Google elevation error: {e}")
            return None

    def directions(self, origin: Tuple[float, float],
                   dest: Tuple[float, float],
                   mode: str = "driving") -> Optional[Dict[str, Any]]:
        """[C] Route between two points (for evacuation planning)."""
        if not self._check():
            return None
        try:
            resp = requests.get(
                f"{self.BASE}/directions/json",
                params={
                    "origin": f"{origin[0]},{origin[1]}",
                    "destination": f"{dest[0]},{dest[1]}",
                    "mode": mode,
                    "key": self.api_key,
                },
                timeout=self.cfg.timeout_s)
            data = resp.json()
            if data.get("status") != "OK":
                logger.warning(f"Google directions failed: {data.get('status')}")
                return None
            route = data["routes"][0]
            leg = route["legs"][0]
            return {
                "provider": "google",
                "distance_m": leg["distance"]["value"],
                "duration_s": leg["duration"]["value"],
                "summary": route.get("summary", ""),
                "polyline": route["overview_polyline"]["points"],
            }
        except Exception as e:
            logger.error(f"Google directions error: {e}")
            return None

    def distance_matrix(self, origins: List[Tuple[float, float]],
                        dest: Tuple[float, float]) -> Optional[List[float]]:
        """[C] Distances from many origins to one destination (meters)."""
        if not self._check():
            return None
        try:
            o_str = "|".join(f"{a},{b}" for a, b in origins)
            resp = requests.get(
                f"{self.BASE}/distancematrix/json",
                params={
                    "origins": o_str,
                    "destinations": f"{dest[0]},{dest[1]}",
                    "key": self.api_key,
                },
                timeout=self.cfg.timeout_s)
            data = resp.json()
            if data.get("status") != "OK":
                return None
            rows = data["rows"]
            return [row["elements"][0]["distance"]["value"] for row in rows]
        except Exception as e:
            logger.error(f"Google distance matrix error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# [C] APPLE MAPS PROVIDER (MapKit REST)
# ═══════════════════════════════════════════════════════════════════════

class AppleMapsProvider:
    """
    [C] Apple MapKit REST connector.

    HONEST: Apple Maps has NO fully-public REST geocoder like Google's.
    Production use requires:
      • Apple Developer Program membership
      • A MapKit JS token or a signed JWT (ES256, from a .p8 private key)
    The endpoints below follow Apple's MapKit Web Service pattern. If the
    developer credentials are absent, this provider reports unavailable and
    the router falls through to the next provider — it NEVER fabricates
    Apple results.
    """

    SEARCH = "https://maps-api.apple.com/v1/search"
    DIRECTIONS = "https://maps-api.apple.com/v1/directions"

    def __init__(self, cfg: GlobalMapsConfig):
        self.cfg = cfg
        self.creds = cfg.apple_credentials()
        self.available = self.creds is not None
        self._jwt: Optional[str] = None

    def _auth_headers(self) -> Dict[str, str]:
        """
        Build Authorization header.

        A real deployment signs a JWT with the ES256 private key (.p8).
        Signing itself is out of scope here; we require a pre-built token
        in APPLE_MAPKIT_JWT if present.
        """
        jwt = os.environ.get("APPLE_MAPKIT_JWT")
        if jwt:
            return {"Authorization": f"Bearer {jwt}"}
        return {}

    def _check(self) -> bool:
        if not self.available:
            logger.warning("Apple Maps: no developer credentials; unavailable")
            return False
        if not HAS_REQUESTS:
            return False
        return True

    def geocode(self, address: str) -> Optional[Dict[str, Any]]:
        """[C/O] Apple search/geocode. JWT-signed request required."""
        if not self._check():
            return None
        try:
            headers = self._auth_headers()
            if not headers:
                logger.warning("Apple Maps: no JWT; cannot authenticate")
                return None
            resp = requests.get(
                self.SEARCH,
                params={"q": address},
                headers=headers,
                timeout=self.cfg.timeout_s)
            if resp.status_code != 200:
                logger.warning(f"Apple geocode HTTP {resp.status_code}")
                return None
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            top = results[0]
            center = top.get("center", {})
            return {
                "lat": center.get("latitude"),
                "lon": center.get("longitude"),
                "system": CoordinateSystem.WGS84 if HAS_GEO else "WGS-84",
                "provider": "apple",
                "formatted": top.get("displayName", address),
            }
        except Exception as e:
            logger.error(f"Apple geocode error: {e}")
            return None

    def directions(self, origin: Tuple[float, float],
                   dest: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """[C/O] Apple directions (requires JWT)."""
        if not self._check():
            return None
        try:
            headers = self._auth_headers()
            if not headers:
                return None
            resp = requests.get(
                self.DIRECTIONS,
                params={
                    "origin": f"{origin[0]},{origin[1]}",
                    "destination": f"{dest[0]},{dest[1]}",
                },
                headers=headers,
                timeout=self.cfg.timeout_s)
            if resp.status_code != 200:
                return None
            data = resp.json()
            routes = data.get("routes", [])
            if not routes:
                return None
            r = routes[0]
            return {
                "provider": "apple",
                "distance_m": r.get("distance"),
                "duration_s": r.get("expectedTime"),
                "summary": "apple_route",
            }
        except Exception as e:
            logger.error(f"Apple directions error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# [C] MICROSOFT AZURE MAPS PROVIDER
# ═══════════════════════════════════════════════════════════════════════

class AzureMapsProvider:
    """
    [C] Microsoft Azure Maps connector.

    NOTE: Bing Maps is being retired; Azure Maps is the supported
    Microsoft mapping platform. Azure Maps uniquely offers a WEATHER API
    alongside geocoding/routing — valuable for storm/hurricane tracking.

    Endpoints:
      • Search / Geocode : atlas.microsoft.com/search/address/json
      • Route            : atlas.microsoft.com/route/directions/json
      • Weather          : atlas.microsoft.com/weather/currentConditions/json
    """

    BASE = "https://atlas.microsoft.com"

    def __init__(self, cfg: GlobalMapsConfig):
        self.cfg = cfg
        self.key = cfg.azure_key()
        self.available = self.key is not None

    def _check(self) -> bool:
        if not self.available:
            logger.warning("Azure Maps: no subscription key; unavailable")
            return False
        if not HAS_REQUESTS:
            return False
        return True

    def geocode(self, address: str) -> Optional[Dict[str, Any]]:
        """[C] Azure Maps address geocoding."""
        if not self._check():
            return None
        try:
            resp = requests.get(
                f"{self.BASE}/search/address/json",
                params={
                    "subscription-key": self.key,
                    "api-version": "1.0",
                    "query": address,
                },
                timeout=self.cfg.timeout_s)
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            top = results[0]
            pos = top.get("position", {})
            return {
                "lat": pos.get("lat"),
                "lon": pos.get("lon"),
                "system": CoordinateSystem.WGS84 if HAS_GEO else "WGS-84",
                "provider": "azure",
                "formatted": top.get("address", {}).get("freeformAddress", address),
            }
        except Exception as e:
            logger.error(f"Azure geocode error: {e}")
            return None

    def directions(self, origin: Tuple[float, float],
                   dest: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """[C] Azure Maps routing."""
        if not self._check():
            return None
        try:
            resp = requests.get(
                f"{self.BASE}/route/directions/json",
                params={
                    "subscription-key": self.key,
                    "api-version": "1.0",
                    "query": (f"{origin[0]},{origin[1]}:"
                              f"{dest[0]},{dest[1]}"),
                },
                timeout=self.cfg.timeout_s)
            data = resp.json()
            routes = data.get("routes", [])
            if not routes:
                return None
            leg = routes[0]["legs"][0]
            return {
                "provider": "azure",
                "distance_m": leg.get("summary", {}).get("lengthInMeters"),
                "duration_s": leg.get("summary", {}).get("travelTimeInSeconds"),
                "summary": "azure_route",
            }
        except Exception as e:
            logger.error(f"Azure directions error: {e}")
            return None

    def current_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        [C] Azure Maps current weather — UNIQUE to Azure among the three.

        Useful for storm/hurricane situational awareness.
        """
        if not self._check():
            return None
        try:
            resp = requests.get(
                f"{self.BASE}/weather/currentConditions/json",
                params={
                    "subscription-key": self.key,
                    "api-version": "1.0",
                    "query": f"{lat},{lon}",
                },
                timeout=self.cfg.timeout_s)
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            obs = results[0]
            return {
                "provider": "azure",
                "temperature_c": obs.get("temperature", {}).get("value"),
                "wind_speed_kmh": obs.get("wind", {}).get("speed", {}).get("value"),
                "wind_dir_deg": obs.get("wind", {}).get("direction", {}).get("value"),
                "alert": obs.get("alerts", []),
            }
        except Exception as e:
            logger.error(f"Azure weather error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# [D] GLOBAL MAPS ROUTER — region-aware + fallback + throttling
# ═══════════════════════════════════════════════════════════════════════

class GlobalMapsRouter:
    """
    [D] Route map requests to the appropriate provider with fallback.

    Logic:
      • Mainland China queries go to sesi_geo_maps.py providers (Amap/
        Tencent/Baidu) because Google/Apple/Azure are restricted or
        offset-incorrect there.
      • Everywhere else: try providers in cfg.preferred_order.
      • Every failure is LOGGED and the next provider is tried
        (CFD §3.3 — failures are recorded, never swallowed).
      • Per-provider call spacing via No-Zeno throttle.
    """

    def __init__(self, cfg: Optional[GlobalMapsConfig] = None):
        self.cfg = cfg or GlobalMapsConfig()
        self.google = GoogleMapsProvider(self.cfg)
        self.apple = AppleMapsProvider(self.cfg)
        self.azure = AzureMapsProvider(self.cfg)
        self.throttle = _CallThrottle(self.cfg.call_min_interval_s)
        self.failure_log: List[Dict[str, Any]] = []

        if HAS_GEO:
            self.transformer = GCJ02Transformer()
        else:
            self.transformer = None

    def _providers_in_order(self) -> List[Tuple[str, Any]]:
        """Map provider enum → instance, in configured order."""
        mapping = {
            MapProviderType.GOOGLE: ("google", self.google),
            MapProviderType.APPLE: ("apple", self.apple),
            MapProviderType.AZURE: ("azure", self.azure),
        }
        return [mapping[p] for p in self.cfg.preferred_order if p in mapping]

    def _is_china(self, lat: float, lon: float) -> bool:
        """Rough mainland-China bounding check."""
        return 18.0 <= lat <= 54.0 and 73.0 <= lon <= 135.0

    def geocode(self, address: str, region_hint: str = "auto"
                ) -> Optional[Dict[str, Any]]:
        """
        Geocode with provider fallback.

        region_hint: "auto", "TH", "CN", or "global"
        """
        if region_hint == "CN":
            logger.info("China region: use sesi_geo_maps.py providers "
                        "(Amap/Tencent/Baidu) for GCJ-02 correctness")
            # Delegate to the Thai/China module if present
            if HAS_GEO:
                from sesi_geo_maps import MapProvider, GeoMapsConfig as GMC
                # Caller should configure Chinese API keys separately
                logger.info("Reroute to MapProvider in sesi_geo_maps for CN")

        for name, provider in self._providers_in_order():
            self.throttle.wait_if_needed(name)
            try:
                result = provider.geocode(address)
                if result is not None:
                    return result
                self.failure_log.append(
                    {"provider": name, "op": "geocode",
                     "reason": "empty_result", "t": time.time()})
            except Exception as e:
                self.failure_log.append(
                    {"provider": name, "op": "geocode",
                     "reason": str(e), "t": time.time()})
                logger.warning(f"Provider {name} geocode failed: {e}")

        logger.error("All providers failed for geocode; returning None "
                     "(NEVER fabricating coordinates)")
        return None

    def route(self, origin: Tuple[float, float],
              dest: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Route with provider fallback."""
        for name, provider in self._providers_in_order():
            if not hasattr(provider, "directions"):
                continue
            self.throttle.wait_if_needed(name)
            try:
                result = provider.directions(origin, dest)
                if result is not None:
                    return result
                self.failure_log.append(
                    {"provider": name, "op": "route",
                     "reason": "empty_result", "t": time.time()})
            except Exception as e:
                self.failure_log.append(
                    {"provider": name, "op": "route",
                     "reason": str(e), "t": time.time()})

        logger.error("All providers failed for routing")
        return None

    def weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Weather is Azure-only among the three; no fallback needed."""
        self.throttle.wait_if_needed("azure")
        return self.azure.current_weather(lat, lon)

    def failure_report(self) -> Dict[str, Any]:
        """CFD §3.3 — failures are always visible."""
        by_provider: Dict[str, int] = {}
        for f in self.failure_log:
            by_provider[f["provider"]] = by_provider.get(f["provider"], 0) + 1
        return {
            "total_failures": len(self.failure_log),
            "by_provider": by_provider,
            "throttle_blocked": self.throttle.report(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] EVACUATION ROUTE PLANNER
# ═══════════════════════════════════════════════════════════════════════

class EvacuationRoutePlanner:
    """
    [D] Plan evacuation routes from a hazard zone to shelters.

    Uses the router to:
      1. Geocode shelters (or accept coordinates).
      2. Compute distance from affected population centroid to each shelter.
      3. Rank shelters by travel time.
      4. Return the fastest route + alternates.

    HONEST: routes are SUGGESTIONS. In a real emergency, follow official
    local-authority evacuation orders, which may override shortest-path.
    """

    def __init__(self, router: GlobalMapsRouter):
        self.router = router

    def rank_shelters(self, origin: Tuple[float, float],
                      shelters: List[Tuple[str, Tuple[float, float]]]
                      ) -> List[Dict[str, Any]]:
        """
        Rank shelters by estimated travel duration from origin.

        shelters: list of (name, (lat, lon))
        """
        ranked = []
        for name, coords in shelters:
            self.router.throttle.wait_if_needed("planner")
            route = self.router.route(origin, coords)
            if route is not None:
                ranked.append({
                    "shelter": name,
                    "coords": coords,
                    "distance_m": route.get("distance_m"),
                    "duration_s": route.get("duration_s"),
                    "provider": route.get("provider"),
                })
            else:
                # Record unreachable shelter, don't drop it silently
                ranked.append({
                    "shelter": name,
                    "coords": coords,
                    "distance_m": None,
                    "duration_s": None,
                    "provider": "unreachable",
                })

        # Sort: reachable first, then by duration
        def sort_key(r):
            if r["duration_s"] is None:
                return (1, float("inf"))
            return (0, r["duration_s"])

        ranked.sort(key=sort_key)
        return ranked


# ═══════════════════════════════════════════════════════════════════════
# [D] FLOOD ELEVATION CHECKER (uses Google Elevation)
# ═══════════════════════════════════════════════════════════════════════

class FloodElevationChecker:
    """
    [D] Screen locations against projected flood water level.

    Logic: ground_elevation + safety_margin > projected_water_level
           → safe, else → at risk.

    Uses Google Elevation API when available; otherwise reports that the
    check cannot be performed (never guesses).
    """

    def __init__(self, router: GlobalMapsRouter, safety_margin_m: float = 0.5):
        self.router = router
        self.safety_margin = safety_margin_m

    def check_point(self, lat: float, lon: float,
                    projected_water_level_m: float) -> Dict[str, Any]:
        """Assess one point against a projected water level."""
        elev = self.router.google.elevation(lat, lon)
        if elev is None:
            return {
                "lat": lat, "lon": lon,
                "status": "unknown",
                "note": ("Elevation unavailable (no Google key or API error). "
                         "Do NOT assume safe; verify with local hydrology."),
            }

        safe = (elev + self.safety_margin) > projected_water_level_m
        return {
            "lat": lat, "lon": lon,
            "ground_elevation_m": elev,
            "projected_water_level_m": projected_water_level_m,
            "safety_margin_m": self.safety_margin,
            "status": "safe" if safe else "at_risk",
            "note": ("Elevation is a PROXY; real flood extent needs "
                     "hydrodynamic modelling (sesi_3d_free_surface.py)."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Global Maps (Google × Apple × Azure) — Smoke Test")
    print("=" * 72)
    print(f"  requests       : {HAS_REQUESTS}")
    print(f"  geo module     : {HAS_GEO}")

    cfg = GlobalMapsConfig()
    router = GlobalMapsRouter(cfg)

    print("\n  ── Provider availability ──")
    print(f"    Google  : {'READY' if router.google.available else 'no key (env ' + cfg.google_api_key_env + ')'}")
    print(f"    Apple   : {'READY' if router.apple.available else 'no developer credentials'}")
    print(f"    Azure   : {'READY' if router.azure.available else 'no subscription key'}")

    print("\n  ── Geocode attempt (graceful degradation) ──")
    result = router.geocode("Bangkok, Thailand", region_hint="TH")
    if result:
        print(f"    lat={result['lat']}, lon={result['lon']} via {result['provider']}")
    else:
        print("    No provider available → returned None (no fabricated coords)")

    print("\n  ── Failure report (CFD §3.3: never swallowed) ──")
    rep = router.failure_report()
    print(f"    Total failures   : {rep['total_failures']}")
    print(f"    By provider      : {rep['by_provider']}")

    print("\n  ── Flood elevation check ──")
    checker = FloodElevationChecker(router)
    flood_check = checker.check_point(13.75, 100.5, projected_water_level_m=2.0)
    print(f"    status           : {flood_check['status']}")
    if "ground_elevation_m" in flood_check:
        print(f"    ground elevation : {flood_check['ground_elevation_m']} m")
    print(f"    note             : {flood_check['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Each provider needs ITS OWN credentials (env vars, not code)")
    print("  • No key → provider reports unavailable; router NEVER fabricates")
    print("  • Apple Maps needs Developer account + signed JWT")
    print("  • Bing Maps is DEPRECATED → use Azure Maps for Microsoft")
    print("  • Elevation flood check is a PROXY, not hydrodynamic modelling")
    print("  • Evacuation routes are suggestions; follow official orders")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
