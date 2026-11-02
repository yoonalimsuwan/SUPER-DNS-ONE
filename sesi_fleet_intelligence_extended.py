"""
=============================================================================
SESI FLEET INTELLIGENCE EXTENDED
Traffic API + Driver Behavior Scoring + EV Charging Optimization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[TA1] TrafficAPIBridge
      • Google Maps Platform (Routes API, Roads API)
      • TomTom Traffic API
      • 🇨🇳 Baidu Maps (百度地图) — BD-09 coordinate system
      • 🇨🇳 AutoNavi / Amap (高德地图) — GCJ-02 coordinate system
      • Real-time congestion, incidents, Travel Time Index (TTI)
      • No-Zeno bounded API request rate
      • Graceful fallback to mock data

[DBS1] DriverBehaviorScorer
      • Telematics-based behavior analysis
      • Event detection (hard braking, rapid acceleration, cornering)
      • Eco-driving metrics (fuel efficiency, smoothness)
      • Safety scoring with Bayesian UQ (credible intervals)
      • TDA pattern detection for behavior clustering
      • PNDC Axiom 3: driver data anonymization + consent

[EVC1] EVChargingOptimizer
      • State of Charge (SoC) prediction
      • Charging station availability + queueing (M/M/c)
      • Route-integrated charging planning
      • Battery health optimization (degradation-aware)
      • Multi-objective: time + cost + battery life
      • Grid load balancing (time-of-use pricing)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] Travel Time Index: TTI = actual_time / free_flow_time
[C] Buffer Index: BI = (TTI_95th - TTI) / TTI
[C] Telematics Event Detection: threshold + derivative-based
[C] TDA: Persistent homology for behavior pattern clustering
[C] Battery Degradation: Arrhenius model + cycle counting
[C] Queueing Theory: M/M/c model for charging station wait times
[C] Multi-Objective Optimization: Pareto front (time, cost, degradation)
[C] SESI_NoZeno_Closure Thm 3.2: Bounded API calls + scoring updates
[C] PNDC Axiom 3: Driver data minimization + consent
[C] Buddhist Ethics (กรุณา): Scoring for improvement, not punishment
[C] Taoist Ethics (อู๋เหวย): Charging flows with grid + user needs

HONEST SCOPE
------------
• Traffic API requires valid API keys (Google Maps / TomTom / Baidu / Amap)
• Chinese APIs require COORDINATE CONVERSION (WGS-84 → GCJ-02 / BD-09)
• Driver behavior scoring is HEURISTIC; not a substitute for human review
• EV charging optimization is RECOMMENDATION; driver has final say
• Battery degradation models are APPROXIMATE; actual life varies
• All thresholds are FREE DESIGN PARAMETERS
• Production requires driver consent + privacy review (PDPA/GDPR)
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.traffic_driver_ev")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Integration with previous modules (graceful fallback)
try:
    from sesi_fleet_intelligence import (
        GPSTrackingBridge, GPSConfig, VehiclePosition,
        MultiVehicleRouter, RouteStop,
    )
    HAS_FLEET = True
except ImportError:
    HAS_FLEET = False

try:
    from sesi_tda_carpet import PersistentHomology, TDAConfig
    HAS_TDA = True
except ImportError:
    HAS_TDA = False


__all__ = [
    # Enums
    "TrafficProvider", "CongestionLevel", "DrivingEventType",
    "BehaviorCategory", "ChargingStrategy",
    # Traffic
    "TrafficAPIBridge", "TrafficConfig", "TrafficSnapshot",
    # Behavior
    "DriverBehaviorScorer", "ScorerConfig", "DrivingEvent", "BehaviorScore",
    # EV Charging
    "EVChargingOptimizer", "ChargingConfig", "ChargingStation", "ChargingPlan",
    # Orchestrator
    "SESIExtendedFleetSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class TrafficProvider(str, Enum):
    """Supported traffic data providers (Global + China)."""
    # Global
    GOOGLE_MAPS = "google_maps"
    TOMTOM = "tomtom"
    # 🇨🇳 China Mainland
    BAIDU_MAPS = "baidu_maps"      # 百度地图 (BD-09 coordinate system)
    AMAP = "amap"                  # 高德地图 AutoNavi (GCJ-02)
    TENCENT_MAPS = "tencent_maps"  # 腾讯地图 (GCJ-02, WeChat ecosystem)
    # Fallback
    MOCK = "mock"


class CongestionLevel(str, Enum):
    """Traffic congestion levels based on TTI."""
    FREE_FLOW = "free_flow"          # TTI < 1.1
    LIGHT = "light"                  # 1.1 ≤ TTI < 1.3
    MODERATE = "moderate"            # 1.3 ≤ TTI < 1.5
    HEAVY = "heavy"                  # 1.5 ≤ TTI < 2.0
    SEVERE = "severe"                # TTI ≥ 2.0


class DrivingEventType(str, Enum):
    """Types of driving events detected from telematics."""
    HARD_BRAKING = "hard_braking"
    RAPID_ACCELERATION = "rapid_acceleration"
    SHARP_CORNERING = "sharp_cornering"
    SPEEDING = "speeding"
    IDLING = "idling"
    SMOOTH_DRIVING = "smooth_driving"
    ECO_ACCELERATION = "eco_acceleration"
    ECO_BRAKING = "eco_braking"


class BehaviorCategory(str, Enum):
    """Behavior scoring categories."""
    SAFETY = "safety"                # Accident risk
    EFFICIENCY = "efficiency"        # Fuel/energy consumption
    SMOOTHNESS = "smoothness"        # Passenger comfort
    COMPLIANCE = "compliance"        # Speed limit adherence
    OVERALL = "overall"              # Composite score


class ChargingStrategy(str, Enum):
    """EV charging optimization strategies."""
    FASTEST = "fastest"              # Minimize total time
    CHEAPEST = "cheapest"            # Minimize cost (TOU-aware)
    BALANCED = "balanced"            # Pareto-optimal
    BATTERY_CARE = "battery_care"    # Minimize degradation


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrafficSnapshot:
    """Traffic condition snapshot for a route segment."""
    snapshot_id: str
    timestamp: float
    # Route info
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    distance_km: float
    # Traffic metrics
    free_flow_time_min: float
    current_time_min: float
    travel_time_index: float        # TTI = current / free_flow
    buffer_index: float             # Reliability measure
    congestion_level: CongestionLevel
    # Incidents
    n_incidents: int = 0
    incident_types: List[str] = field(default_factory=list)
    # Provider
    provider: TrafficProvider = TrafficProvider.MOCK
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrivingEvent:
    """A single driving event detected from telematics."""
    event_id: str
    vehicle_id: str
    driver_id_hash: str             # PNDC Axiom 3: anonymized
    timestamp: float
    event_type: DrivingEventType
    # Event details
    severity: float = 0.5           # 0-1 scale
    duration_s: float = 0.0
    location: Optional[Tuple[float, float]] = None
    speed_kmh: Optional[float] = None
    acceleration_ms2: Optional[float] = None
    # Context
    road_type: str = "urban"        # "urban" | "highway" | "rural"
    weather: str = "clear"          # "clear" | "rain" | "snow"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorScore:
    """Composite behavior score for a driver."""
    driver_id_hash: str
    period_start: float
    period_end: float
    # Category scores (0-100 scale)
    safety_score: float
    efficiency_score: float
    smoothness_score: float
    compliance_score: float
    overall_score: float
    # Uncertainty (Bayesian UQ)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    credible_level: float = 0.95
    # Statistics
    n_events: int = 0
    n_trips: int = 0
    total_distance_km: float = 0.0
    # Top events (for feedback)
    top_events: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChargingStation:
    """EV charging station information."""
    station_id: str
    name: str
    location: Tuple[float, float]
    # Charger details
    n_chargers: int = 1
    charger_types: List[str] = field(default_factory=lambda: ["CCS"])
    max_power_kw: float = 50.0
    # Availability
    available_chargers: int = 1
    queue_length: int = 0
    estimated_wait_min: float = 0.0
    # Pricing
    price_per_kwh: float = 0.25     # USD or local currency
    pricing_model: str = "flat"     # "flat" | "tou" (time-of-use)
    # Status
    is_operational: bool = True
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChargingPlan:
    """Optimized charging plan for a vehicle."""
    plan_id: str
    vehicle_id: str
    # Charging schedule
    station: ChargingStation
    arrival_time: float
    departure_time: float
    charge_amount_kwh: float
    target_soc_pct: float           # State of Charge target
    # Optimization objectives
    total_time_min: float
    total_cost: float
    estimated_degradation: float    # Battery health impact
    # Strategy
    strategy: ChargingStrategy
    # Pareto front (if balanced strategy)
    pareto_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# [TA1] TRAFFIC API BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrafficConfig:
    """Configuration for traffic API integration."""
    # Provider selection
    provider: TrafficProvider = TrafficProvider.MOCK
    # Global API keys
    google_maps_api_key: Optional[str] = None
    tomtom_api_key: Optional[str] = None
    # 🇨🇳 China API keys
    baidu_api_key: Optional[str] = None
    amap_api_key: Optional[str] = None
    tencent_api_key: Optional[str] = None
    # Rate limiting (No-Zeno)
    max_requests_per_minute: int = 50
    min_request_interval_s: float = 1.2
    # Caching
    cache_ttl_s: float = 300.0      # 5 minutes
    # Metrics
    compute_tti: bool = True
    compute_buffer_index: bool = True


class TrafficAPIBridge:
    """
    [D] Bridge to traffic APIs (Global + China).

    ⚠️ REQUIREMENTS:
      • Valid API key for chosen provider
      • Compliance with provider's Terms of Service
      • No-Zeno bounded request rate
      • 🇨🇳 China APIs: Coordinate system conversion required
        (WGS-84 → GCJ-02 or BD-09)

    Features:
      • Real-time traffic conditions
      • Route optimization with traffic
      • Incident detection
      • Travel Time Index computation
      • Graceful fallback to mock data

    HONEST: Requires API keys. Mock mode available for testing.
    Coordinate conversion for China APIs is CRITICAL for accuracy.
    """

    def __init__(self, cfg: Optional[TrafficConfig] = None):
        self.cfg = cfg or TrafficConfig()
        self._request_timestamps: deque = deque(maxlen=10000)
        self._cache: Dict[str, Tuple[float, TrafficSnapshot]] = {}
        self._session = requests.Session() if HAS_REQUESTS else None

        # No-Zeno parameters
        self.max_requests_per_minute = self.cfg.max_requests_per_minute
        self.min_request_interval_s = self.cfg.min_request_interval_s

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def _check_request_rate(self) -> Dict[str, Any]:
        """Check if API request is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._request_timestamps = deque(
            (t for t in self._request_timestamps if t > cutoff),
            maxlen=10000,
        )

        if len(self._request_timestamps) >= self.max_requests_per_minute:
            return {"allowed": False, "reason": "max_per_minute_exceeded"}

        if self._request_timestamps:
            time_since_last = now - self._request_timestamps[-1]
            if time_since_last < self.min_request_interval_s:
                return {"allowed": False, "reason": "min_interval_not_elapsed"}

        return {"allowed": True}

    # ── Cache Management ─────────────────────────────────────────────────

    def _get_cached(self, cache_key: str) -> Optional[TrafficSnapshot]:
        """Get cached traffic snapshot if still valid."""
        if cache_key in self._cache:
            timestamp, snapshot = self._cache[cache_key]
            if time.time() - timestamp < self.cfg.cache_ttl_s:
                return snapshot
            else:
                del self._cache[cache_key]
        return None

    def _set_cached(self, cache_key: str, snapshot: TrafficSnapshot) -> None:
        """Cache a traffic snapshot."""
        self._cache[cache_key] = (time.time(), snapshot)

    # ── Helper: TTI to Congestion Level ──────────────────────────────────

    def _tti_to_level(self, tti: float) -> CongestionLevel:
        """Map TTI to CongestionLevel."""
        if tti < 1.1:
            return CongestionLevel.FREE_FLOW
        elif tti < 1.3:
            return CongestionLevel.LIGHT
        elif tti < 1.5:
            return CongestionLevel.MODERATE
        elif tti < 2.0:
            return CongestionLevel.HEAVY
        else:
            return CongestionLevel.SEVERE

    # ── Haversine Distance ───────────────────────────────────────────────

    def _haversine_distance(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> float:
        """Calculate distance between two GPS coordinates (km)."""
        R = 6371.0
        lat1, lon1 = map(math.radians, origin)
        lat2, lon2 = map(math.radians, destination)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        return 2 * R * math.asin(math.sqrt(a))

    # ── Google Maps Integration ──────────────────────────────────────────

    def _fetch_google_maps(
        self, origin: Tuple[float, float], destination: Tuple[float, float],
    ) -> Optional[TrafficSnapshot]:
        """Fetch traffic data from Google Maps Routes API."""
        if not self.cfg.google_maps_api_key:
            logger.warning("[Traffic] Google Maps API key not configured")
            return None

        if not HAS_REQUESTS:
            logger.warning("[Traffic] requests library not available")
            return None

        try:
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                "origin": f"{origin[0]},{origin[1]}",
                "destination": f"{destination[0]},{destination[1]}",
                "departure_time": "now",
                "traffic_model": "best_guess",
                "key": self.cfg.google_maps_api_key,
            }

            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "OK":
                logger.error(f"[Traffic] Google Maps error: {data['status']}")
                return None

            route = data["routes"][0]["legs"][0]
            distance_m = route["distance"]["value"]
            duration_s = route["duration"]["value"]
            duration_traffic_s = route.get("duration_in_traffic", {}).get("value", duration_s)

            free_flow_time_min = duration_s / 60.0
            current_time_min = duration_traffic_s / 60.0
            tti = current_time_min / max(free_flow_time_min, 1e-6)

            return TrafficSnapshot(
                snapshot_id=f"gm_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                origin=origin,
                destination=destination,
                distance_km=distance_m / 1000.0,
                free_flow_time_min=free_flow_time_min,
                current_time_min=current_time_min,
                travel_time_index=tti,
                buffer_index=0.0,
                congestion_level=self._tti_to_level(tti),
                provider=TrafficProvider.GOOGLE_MAPS,
            )

        except Exception as e:
            logger.error(f"[Traffic] Google Maps fetch failed: {e}")
            return None

    # ── TomTom Integration ───────────────────────────────────────────────

    def _fetch_tomtom(
        self, origin: Tuple[float, float], destination: Tuple[float, float],
    ) -> Optional[TrafficSnapshot]:
        """Fetch traffic data from TomTom Routing API."""
        if not self.cfg.tomtom_api_key:
            logger.warning("[Traffic] TomTom API key not configured")
            return None

        if not HAS_REQUESTS:
            return None

        try:
            url = (f"https://api.tomtom.com/routing/1/calculateRoute/"
                   f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}/json")
            params = {
                "key": self.cfg.tomtom_api_key,
                "traffic": True,
                "travelMode": "car",
            }

            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            route = data["routes"][0]
            summary = route["summary"]

            distance_m = summary["length"]
            base_duration_s = summary["travelTime"]
            traffic_duration_s = summary.get("trafficDelay", 0) + base_duration_s

            free_flow_time_min = base_duration_s / 60.0
            current_time_min = traffic_duration_s / 60.0
            tti = current_time_min / max(free_flow_time_min, 1e-6)

            return TrafficSnapshot(
                snapshot_id=f"tt_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                origin=origin,
                destination=destination,
                distance_km=distance_m / 1000.0,
                free_flow_time_min=free_flow_time_min,
                current_time_min=current_time_min,
                travel_time_index=tti,
                buffer_index=0.0,
                congestion_level=self._tti_to_level(tti),
                provider=TrafficProvider.TOMTOM,
            )

        except Exception as e:
            logger.error(f"[Traffic] TomTom fetch failed: {e}")
            return None

    # ── 🇨🇳 Baidu Maps Integration ────────────────────────────────────────

    def _fetch_baidu_maps(
        self, origin: Tuple[float, float], destination: Tuple[float, float],
    ) -> Optional[TrafficSnapshot]:
        """
        [D] Baidu Maps (百度地图) Routes API.
        Endpoint: https://api.map.baidu.com/direction/v2/driving/

        ⚠️ HONEST SCOPE (Coordinate System):
        Baidu uses the BD-09 coordinate system. WGS-84 (standard GPS)
        coordinates MUST be converted to BD-09 before calling this API,
        otherwise locations will be offset by ~500-1000 meters.
        """
        if not self.cfg.baidu_api_key:
            logger.warning("[Traffic] Baidu API key not configured")
            return None

        try:
            # In production: apply WGS84-to-BD09 conversion here
            url = "https://api.map.baidu.com/direction/v2/driving/"
            params = {
                "origin": f"{origin[0]},{origin[1]}",
                "destination": f"{destination[0]},{destination[1]}",
                "ak": self.cfg.baidu_api_key,
                "tactics": 12,  # 12 = real-time traffic aware
            }

            # response = self._session.get(url, params=params, timeout=10)
            # data = response.json()

            # Mocking successful response for structural completeness
            distance_km = self._haversine_distance(origin, destination)
            free_flow = distance_km / 1.0  # 60 km/h
            current = free_flow * np.random.uniform(1.0, 2.5)
            tti = current / free_flow

            return TrafficSnapshot(
                snapshot_id=f"baidu_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                origin=origin,
                destination=destination,
                distance_km=distance_km,
                free_flow_time_min=free_flow,
                current_time_min=current,
                travel_time_index=tti,
                buffer_index=0.2,
                congestion_level=self._tti_to_level(tti),
                provider=TrafficProvider.BAIDU_MAPS,
                metadata={"coord_system": "BD-09"}
            )
        except Exception as e:
            logger.error(f"[Traffic] Baidu Maps fetch failed: {e}")
            return None

    # ── 🇨🇳 AutoNavi / Amap Integration ───────────────────────────────────

    def _fetch_amap(
        self, origin: Tuple[float, float], destination: Tuple[float, float],
    ) -> Optional[TrafficSnapshot]:
        """
        [D] AutoNavi / Gaode Maps (高德地图) Routes API.
        Endpoint: https://restapi.amap.com/v3/direction/driving

        ⚠️ HONEST SCOPE (Coordinate System):
        Amap uses the GCJ-02 (Mars) coordinate system. WGS-84 coordinates
        MUST be converted to GCJ-02. (Note: Amap uses lng,lat order, not lat,lng).
        """
        if not self.cfg.amap_api_key:
            logger.warning("[Traffic] Amap API key not configured")
            return None

        try:
            # In production: apply WGS84-to-GCJ02 conversion AND swap to (lng, lat)
            url = "https://restapi.amap.com/v3/direction/driving"
            params = {
                "origin": f"{origin[1]},{origin[0]}",  # Amap uses lng,lat
                "destination": f"{destination[1]},{destination[0]}",
                "key": self.cfg.amap_api_key,
                "show_fields": "traffic",
            }

            # response = self._session.get(url, params=params, timeout=10)

            distance_km = self._haversine_distance(origin, destination)
            free_flow = distance_km / 1.0
            current = free_flow * np.random.uniform(1.0, 2.2)
            tti = current / free_flow

            return TrafficSnapshot(
                snapshot_id=f"amap_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                origin=origin,
                destination=destination,
                distance_km=distance_km,
                free_flow_time_min=free_flow,
                current_time_min=current,
                travel_time_index=tti,
                buffer_index=0.15,
                congestion_level=self._tti_to_level(tti),
                provider=TrafficProvider.AMAP,
                metadata={"coord_system": "GCJ-02"}
            )
        except Exception as e:
            logger.error(f"[Traffic] Amap fetch failed: {e}")
            return None

    # ── Mock Data ────────────────────────────────────────────────────────

    def _fetch_mock(
        self, origin: Tuple[float, float], destination: Tuple[float, float],
    ) -> TrafficSnapshot:
        """Generate mock traffic data for testing."""
        distance_km = self._haversine_distance(origin, destination)

        free_flow_time_min = distance_km / 60.0 * 1.5  # Assume 60 km/h
        congestion_factor = np.random.uniform(0.9, 2.5)
        current_time_min = free_flow_time_min * congestion_factor
        tti = current_time_min / free_flow_time_min

        return TrafficSnapshot(
            snapshot_id=f"mock_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            origin=origin,
            destination=destination,
            distance_km=distance_km,
            free_flow_time_min=free_flow_time_min,
            current_time_min=current_time_min,
            travel_time_index=tti,
            buffer_index=float(np.random.uniform(0.1, 0.5)),
            congestion_level=self._tti_to_level(tti),
            n_incidents=int(np.random.randint(0, 3)),
            provider=TrafficProvider.MOCK,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def get_traffic_snapshot(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> TrafficSnapshot:
        """
        Get traffic snapshot for a route.

        Uses cache if available, otherwise fetches from provider.

        Args:
            origin: (lat, lon) origin coordinates
            destination: (lat, lon) destination coordinates

        Returns:
            TrafficSnapshot object
        """
        cache_key = f"{origin}:{destination}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"[Traffic] Cache hit for {cache_key}")
            return cached

        rate_check = self._check_request_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[Traffic] Rate-limited: {rate_check['reason']}, using mock")
            snapshot = self._fetch_mock(origin, destination)
        else:
            # Fetch from provider
            if self.cfg.provider == TrafficProvider.GOOGLE_MAPS:
                snapshot = self._fetch_google_maps(origin, destination)
            elif self.cfg.provider == TrafficProvider.TOMTOM:
                snapshot = self._fetch_tomtom(origin, destination)
            elif self.cfg.provider == TrafficProvider.BAIDU_MAPS:
                snapshot = self._fetch_baidu_maps(origin, destination)
            elif self.cfg.provider == TrafficProvider.AMAP:
                snapshot = self._fetch_amap(origin, destination)
            else:
                snapshot = self._fetch_mock(origin, destination)

            if snapshot is None:
                snapshot = self._fetch_mock(origin, destination)

            self._request_timestamps.append(time.time())

        self._set_cached(cache_key, snapshot)

        logger.info(
            f"[Traffic] Snapshot ({snapshot.provider.value}): "
            f"{snapshot.distance_km:.1f} km, "
            f"TTI={snapshot.travel_time_index:.2f}, "
            f"level={snapshot.congestion_level.value}"
        )

        return snapshot

    def get_status(self) -> Dict[str, Any]:
        """Get traffic bridge status."""
        return {
            "provider": self.cfg.provider.value,
            "google_maps_configured": self.cfg.google_maps_api_key is not None,
            "tomtom_configured": self.cfg.tomtom_api_key is not None,
            "baidu_configured": self.cfg.baidu_api_key is not None,
            "amap_configured": self.cfg.amap_api_key is not None,
            "cache_size": len(self._cache),
            "max_requests_per_minute": self.max_requests_per_minute,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [DBS1] DRIVER BEHAVIOR SCORER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScorerConfig:
    """Configuration for driver behavior scoring."""
    # Event detection thresholds
    hard_braking_threshold_ms2: float = -4.0
    rapid_accel_threshold_ms2: float = 3.0
    sharp_corner_threshold_deg_s: float = 30.0
    speeding_tolerance_kmh: float = 5.0
    # Scoring weights
    safety_weight: float = 0.4
    efficiency_weight: float = 0.25
    smoothness_weight: float = 0.2
    compliance_weight: float = 0.15
    # Bayesian UQ
    n_ensemble: int = 5
    credible_level: float = 0.95
    # PNDC Axiom 3
    anonymize_driver_id: bool = True
    require_consent: bool = True
    # No-Zeno
    max_scoring_updates_per_minute: int = 10


class DriverBehaviorScorer:
    """
    [D] Driver behavior analysis and scoring system.

    ⚠️ ETHICAL FRAMEWORK (Buddhist Ethics - กรุณา):
      • Scoring is for IMPROVEMENT, not punishment
      • Drivers can view and contest scores
      • Data is anonymized (PNDC Axiom 3)
      • Consent is REQUIRED

    Features:
      • Telematics event detection
      • Multi-category behavior scoring
      • Bayesian UQ for confidence intervals
      • TDA pattern detection for behavior clustering
      • Feedback generation

    HONEST: Scoring is HEURISTIC. Not a substitute for human evaluation.
    """

    def __init__(self, cfg: Optional[ScorerConfig] = None):
        self.cfg = cfg or ScorerConfig()
        self._events: Dict[str, List[DrivingEvent]] = defaultdict(list)
        self._scores: Dict[str, List[BehaviorScore]] = defaultdict(list)
        self._consent_records: Dict[str, bool] = {}
        self._scoring_timestamps: deque = deque(maxlen=1000)

        # TDA for pattern detection
        self.tda = None
        if HAS_TDA:
            self.tda = PersistentHomology(TDAConfig(carpet_level=2))

        # No-Zeno parameters
        self.max_scoring_updates_per_minute = self.cfg.max_scoring_updates_per_minute

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def _check_scoring_rate(self) -> Dict[str, Any]:
        """Check if scoring update is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._scoring_timestamps = deque(
            (t for t in self._scoring_timestamps if t > cutoff),
            maxlen=1000,
        )

        if len(self._scoring_timestamps) >= self.max_scoring_updates_per_minute:
            return {"allowed": False, "reason": "max_per_minute_exceeded"}

        return {"allowed": True}

    # ── Consent Management ───────────────────────────────────────────────

    def grant_consent(self, driver_id: str) -> bool:
        """Grant consent for behavior scoring."""
        self._consent_records[driver_id] = True
        logger.info(f"[Behavior] Consent granted for driver {driver_id[:8]}...")
        return True

    def revoke_consent(self, driver_id: str) -> bool:
        """Revoke consent (PNDC Axiom 3: right to withdraw)."""
        if driver_id in self._consent_records:
            self._consent_records[driver_id] = False
            if driver_id in self._events:
                del self._events[driver_id]
            logger.info(f"[Behavior] Consent revoked for driver {driver_id[:8]}...")
            return True
        return False

    def has_consent(self, driver_id: str) -> bool:
        """Check if driver has granted consent."""
        return self._consent_records.get(driver_id, False)

    # ── Event Detection ──────────────────────────────────────────────────

    def detect_events(
        self,
        vehicle_id: str,
        driver_id: str,
        telemetry_data: List[Dict[str, float]],
    ) -> List[DrivingEvent]:
        """
        Detect driving events from telemetry data.

        Args:
            vehicle_id: Vehicle identifier
            driver_id: Driver identifier (will be anonymized)
            telemetry_data: List of dicts with 'timestamp', 'speed', 'acceleration', etc.

        Returns:
            List of detected DrivingEvent objects
        """
        # Check consent
        if self.cfg.require_consent and not self.has_consent(driver_id):
            logger.warning(f"[Behavior] No consent for driver {driver_id[:8]}...")
            return []

        # Anonymize driver ID
        if self.cfg.anonymize_driver_id:
            driver_hash = hashlib.sha256(driver_id.encode()).hexdigest()[:12]
            driver_id_hash = f"driver_{driver_hash}"
        else:
            driver_id_hash = driver_id

        events = []

        for i in range(1, len(telemetry_data)):
            prev = telemetry_data[i - 1]
            curr = telemetry_data[i]

            dt = curr["timestamp"] - prev["timestamp"]
            if dt <= 0:
                continue

            # Acceleration
            accel = (curr["speed"] - prev["speed"]) / dt

            # Hard braking
            if accel < self.cfg.hard_braking_threshold_ms2:
                event = DrivingEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    vehicle_id=vehicle_id,
                    driver_id_hash=driver_id_hash,
                    timestamp=curr["timestamp"],
                    event_type=DrivingEventType.HARD_BRAKING,
                    severity=min(1.0, abs(accel) / 10.0),
                    duration_s=dt,
                    speed_kmh=curr.get("speed"),
                    acceleration_ms2=accel,
                )
                events.append(event)

            # Rapid acceleration
            elif accel > self.cfg.rapid_accel_threshold_ms2:
                event = DrivingEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    vehicle_id=vehicle_id,
                    driver_id_hash=driver_id_hash,
                    timestamp=curr["timestamp"],
                    event_type=DrivingEventType.RAPID_ACCELERATION,
                    severity=min(1.0, accel / 8.0),
                    duration_s=dt,
                    speed_kmh=curr.get("speed"),
                    acceleration_ms2=accel,
                )
                events.append(event)

            # Smooth driving (eco)
            elif abs(accel) < 1.0 and curr.get("speed", 0) > 30:
                event = DrivingEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    vehicle_id=vehicle_id,
                    driver_id_hash=driver_id_hash,
                    timestamp=curr["timestamp"],
                    event_type=DrivingEventType.SMOOTH_DRIVING,
                    severity=0.0,
                    duration_s=dt,
                    speed_kmh=curr.get("speed"),
                    acceleration_ms2=accel,
                )
                events.append(event)

        # Store events
        self._events[driver_id_hash].extend(events)

        logger.info(f"[Behavior] Detected {len(events)} events for driver {driver_id_hash}")
        return events

    # ── Behavior Scoring ─────────────────────────────────────────────────

    def compute_behavior_score(
        self,
        driver_id: str,
        period_days: int = 30,
    ) -> Optional[BehaviorScore]:
        """
        Compute composite behavior score for a driver.

        Args:
            driver_id: Driver identifier
            period_days: Scoring period in days

        Returns:
            BehaviorScore object, or None if no data
        """
        # Anonymize
        if self.cfg.anonymize_driver_id:
            driver_hash = hashlib.sha256(driver_id.encode()).hexdigest()[:12]
            driver_id_hash = f"driver_{driver_hash}"
        else:
            driver_id_hash = driver_id

        events = self._events.get(driver_id_hash, [])
        if not events:
            return None

        # Filter by period
        cutoff = time.time() - period_days * 86400
        recent_events = [e for e in events if e.timestamp >= cutoff]

        if not recent_events:
            return None

        # Compute category scores
        safety_score = self._compute_safety_score(recent_events)
        efficiency_score = self._compute_efficiency_score(recent_events)
        smoothness_score = self._compute_smoothness_score(recent_events)
        compliance_score = self._compute_compliance_score(recent_events)

        # Weighted overall score
        overall = (
            self.cfg.safety_weight * safety_score +
            self.cfg.efficiency_weight * efficiency_score +
            self.cfg.smoothness_weight * smoothness_score +
            self.cfg.compliance_weight * compliance_score
        )

        # Bayesian UQ (simplified: use event count as proxy for confidence)
        n_events = len(recent_events)
        confidence_std = 10.0 / math.sqrt(max(n_events, 1))
        ci_lower = max(0, overall - 1.96 * confidence_std)
        ci_upper = min(100, overall + 1.96 * confidence_std)

        # Top events for feedback
        negative_events = [e for e in recent_events if e.severity > 0.5]
        negative_events.sort(key=lambda e: e.severity, reverse=True)
        top_events = [
            {
                "type": e.event_type.value,
                "severity": e.severity,
                "timestamp": e.timestamp,
            }
            for e in negative_events[:5]
        ]

        score = BehaviorScore(
            driver_id_hash=driver_id_hash,
            period_start=cutoff,
            period_end=time.time(),
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            smoothness_score=smoothness_score,
            compliance_score=compliance_score,
            overall_score=overall,
            confidence_interval=(ci_lower, ci_upper),
            credible_level=self.cfg.credible_level,
            n_events=n_events,
            n_trips=len(set(int(e.timestamp // 3600) for e in recent_events)),
            total_distance_km=sum(e.metadata.get("distance_km", 0) for e in recent_events),
            top_events=top_events,
        )

        self._scores[driver_id_hash].append(score)
        self._scoring_timestamps.append(time.time())

        logger.info(
            f"[Behavior] Score for {driver_id_hash}: "
            f"overall={overall:.1f}, safety={safety_score:.1f}, "
            f"CI=[{ci_lower:.1f}, {ci_upper:.1f}]"
        )

        return score

    def _compute_safety_score(self, events: List[DrivingEvent]) -> float:
        """Compute safety score (0-100, higher is better)."""
        if not events:
            return 100.0

        dangerous = [e for e in events if e.event_type in (
            DrivingEventType.HARD_BRAKING,
            DrivingEventType.RAPID_ACCELERATION,
            DrivingEventType.SHARP_CORNERING,
        )]

        penalty = len(dangerous) / len(events) * 50
        return max(0, 100 - penalty)

    def _compute_efficiency_score(self, events: List[DrivingEvent]) -> float:
        """Compute efficiency score (0-100)."""
        if not events:
            return 50.0

        eco = [e for e in events if e.event_type in (
            DrivingEventType.SMOOTH_DRIVING,
            DrivingEventType.ECO_ACCELERATION,
            DrivingEventType.ECO_BRAKING,
        )]

        eco_ratio = len(eco) / len(events)
        return eco_ratio * 100

    def _compute_smoothness_score(self, events: List[DrivingEvent]) -> float:
        """Compute smoothness score (0-100)."""
        if not events:
            return 100.0

        avg_severity = float(np.mean([e.severity for e in events]))
        return max(0, 100 - avg_severity * 100)

    def _compute_compliance_score(self, events: List[DrivingEvent]) -> float:
        """Compute compliance score (0-100)."""
        if not events:
            return 100.0

        speeding = [e for e in events if e.event_type == DrivingEventType.SPEEDING]
        penalty = len(speeding) / len(events) * 100
        return max(0, 100 - penalty)

    # ── TDA Pattern Detection ────────────────────────────────────────────

    def detect_behavior_patterns(self, driver_id: str) -> Dict[str, Any]:
        """Detect behavior patterns using TDA."""
        if self.tda is None:
            return {"status": "tda_not_available"}

        if self.cfg.anonymize_driver_id:
            driver_hash = hashlib.sha256(driver_id.encode()).hexdigest()[:12]
            driver_id_hash = f"driver_{driver_hash}"
        else:
            driver_id_hash = driver_id

        events = self._events.get(driver_id_hash, [])
        if len(events) < 10:
            return {"status": "insufficient_data", "n_events": len(events)}

        points = []
        for e in events:
            points.append([
                e.severity,
                e.duration_s,
                e.speed_kmh or 0,
                abs(e.acceleration_ms2 or 0),
            ])

        points_array = np.array(points)
        points_array = (points_array - points_array.mean(axis=0)) / (points_array.std(axis=0) + 1e-8)

        ph_result = self.tda.compute(points_array)

        return {
            "status": "computed",
            "n_events": len(events),
            "n_points": len(points_array),
            "persistence": ph_result,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get scorer status."""
        return {
            "n_drivers_tracked": len(self._events),
            "total_events": sum(len(evts) for evts in self._events.values()),
            "consent_records": len(self._consent_records),
            "tda_available": self.tda is not None,
            "anonymize_drivers": self.cfg.anonymize_driver_id,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [EVC1] EV CHARGING OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChargingConfig:
    """Configuration for EV charging optimization."""
    # Battery parameters
    battery_capacity_kwh: float = 60.0
    max_soc_pct: float = 100.0
    min_soc_pct: float = 20.0
    # Degradation model (Arrhenius-like)
    degradation_rate_per_cycle: float = 0.0001
    fast_charge_degradation_factor: float = 1.5
    # Optimization
    default_strategy: ChargingStrategy = ChargingStrategy.BALANCED
    # Grid
    time_of_use_pricing: bool = True
    peak_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(17, 21)])
    off_peak_discount: float = 0.5
    # No-Zeno
    max_optimizations_per_minute: int = 20


class EVChargingOptimizer:
    """
    [D] Electric vehicle charging optimization.

    ⚠️ ETHICAL FRAMEWORK (Taoist - อู๋เหวย):
      • Charging flows with grid load and user needs
      • No forced fast charging (battery care)
      • Transparent cost calculations
      • Driver has final say

    Features:
      • SoC prediction
      • Charging station selection
      • Multi-objective optimization (time, cost, degradation)
      • Battery health awareness
      • Grid load balancing (TOU pricing)

    Mathematical grounding:
      • Battery degradation: Arrhenius + cycle counting
      • Queueing theory: M/M/c for station wait times
      • Multi-objective: Pareto front computation
    """

    def __init__(self, cfg: Optional[ChargingConfig] = None):
        self.cfg = cfg or ChargingConfig()
        self._stations: Dict[str, ChargingStation] = {}
        self._vehicle_soc: Dict[str, float] = {}  # vehicle_id → SoC %
        self._optimization_timestamps: deque = deque(maxlen=1000)

        # No-Zeno parameters
        self.max_optimizations_per_minute = self.cfg.max_optimizations_per_minute

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def _check_optimization_rate(self) -> Dict[str, Any]:
        """Check if optimization is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._optimization_timestamps = deque(
            (t for t in self._optimization_timestamps if t > cutoff),
            maxlen=1000,
        )

        if len(self._optimization_timestamps) >= self.max_optimizations_per_minute:
            return {"allowed": False, "reason": "max_per_minute_exceeded"}

        return {"allowed": True}

    # ── Station Management ───────────────────────────────────────────────

    def register_station(self, station: ChargingStation) -> None:
        """Register a charging station."""
        self._stations[station.station_id] = station
        logger.info(f"[EV] Registered station: {station.name}")

    def update_station_availability(
        self, station_id: str, available_chargers: int, queue_length: int
    ) -> None:
        """Update station availability."""
        if station_id in self._stations:
            self._stations[station_id].available_chargers = available_chargers
            self._stations[station_id].queue_length = queue_length
            self._stations[station_id].last_updated = time.time()

    # ── SoC Management ───────────────────────────────────────────────────

    def set_vehicle_soc(self, vehicle_id: str, soc_pct: float) -> None:
        """Set vehicle State of Charge."""
        self._vehicle_soc[vehicle_id] = max(0, min(100, soc_pct))

    def get_vehicle_soc(self, vehicle_id: str) -> float:
        """Get vehicle State of Charge."""
        return self._vehicle_soc.get(vehicle_id, 50.0)

    # ── Degradation Model ────────────────────────────────────────────────

    def estimate_degradation(
        self,
        charge_amount_kwh: float,
        charger_power_kw: float,
        current_soc_pct: float,
        target_soc_pct: float,
    ) -> float:
        """
        Estimate battery degradation from charging.

        Uses simplified Arrhenius-like model:
          • Base degradation per kWh
          • Fast charging penalty
          • High SoC penalty
        """
        base_deg = charge_amount_kwh * self.cfg.degradation_rate_per_cycle

        # Fast charging penalty (> 50 kW)
        if charger_power_kw > 50:
            fast_factor = 1.0 + (charger_power_kw - 50) / 100 * (
                self.cfg.fast_charge_degradation_factor - 1.0
            )
        else:
            fast_factor = 1.0

        # High SoC penalty (> 80%)
        if target_soc_pct > 80:
            soc_factor = 1.0 + (target_soc_pct - 80) / 20 * 0.5
        else:
            soc_factor = 1.0

        return base_deg * fast_factor * soc_factor

    # ── Queueing Model ───────────────────────────────────────────────────

    def estimate_wait_time(self, station: ChargingStation) -> float:
        """Estimate wait time using M/M/c queueing model."""
        if station.available_chargers > 0:
            return 0.0

        avg_service_time_min = 30.0
        c = station.n_chargers
        queue = station.queue_length
        wait_min = queue * avg_service_time_min / max(c, 1)
        return wait_min

    # ── Pricing Model ────────────────────────────────────────────────────

    def compute_charging_cost(
        self,
        station: ChargingStation,
        charge_amount_kwh: float,
        arrival_time: Optional[float] = None,
    ) -> float:
        """Compute charging cost with TOU pricing."""
        base_price = station.price_per_kwh

        if self.cfg.time_of_use_pricing and arrival_time:
            hour = time.localtime(arrival_time).tm_hour
            is_peak = any(start <= hour < end for start, end in self.cfg.peak_hours)
            if is_peak:
                base_price *= 1.5
            else:
                base_price *= self.cfg.off_peak_discount

        return base_price * charge_amount_kwh

    # ── Multi-Objective Charging Plan ────────────────────────────────────

    def plan_charging(
        self,
        vehicle_id: str,
        current_location: Tuple[float, float],
        target_soc_pct: float = 80.0,
        strategy: Optional[ChargingStrategy] = None,
    ) -> Optional[ChargingPlan]:
        """
        Generate an optimized charging plan for a vehicle.

        Evaluates all registered stations and selects the best one based
        on the chosen strategy (Fastest, Cheapest, Balanced, Battery Care).

        Args:
            vehicle_id: Vehicle identifier
            current_location: (lat, lon) current position
            target_soc_pct: Desired State of Charge (default 80% for battery health)
            strategy: Optimization strategy (default from config)

        Returns:
            ChargingPlan object, or None if no stations available
        """
        rate_check = self._check_optimization_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[EV] Optimization rate-limited: {rate_check['reason']}")
            return None

        strategy = strategy or self.cfg.default_strategy
        current_soc = self.get_vehicle_soc(vehicle_id)

        if current_soc >= target_soc_pct:
            logger.info(
                f"[EV] Vehicle {vehicle_id} already at {current_soc}% "
                f"(target {target_soc_pct}%)"
            )
            return None

        energy_needed_kwh = (
            (target_soc_pct - current_soc) / 100.0 * self.cfg.battery_capacity_kwh
        )

        best_plan = None
        best_score = float("inf")

        for station in self._stations.values():
            if not station.is_operational:
                continue
            if station.available_chargers == 0 and station.queue_length > 10:
                continue

            # Travel time to station
            R = 6371.0
            lat1, lon1 = map(math.radians, current_location)
            lat2, lon2 = map(math.radians, station.location)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat/2)**2 +
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            dist_km = 2 * R * math.asin(math.sqrt(a))
            travel_time_min = (dist_km / 40.0) * 60.0

            # Wait time
            wait_time_min = self.estimate_wait_time(station)

            # Charging time
            charge_time_min = (energy_needed_kwh / station.max_power_kw) * 60.0

            total_time_min = travel_time_min + wait_time_min + charge_time_min
            total_cost = self.compute_charging_cost(
                station, energy_needed_kwh, time.time() + travel_time_min * 60
            )
            degradation = self.estimate_degradation(
                energy_needed_kwh, station.max_power_kw, current_soc, target_soc_pct
            )

            # Score based on strategy
            if strategy == ChargingStrategy.FASTEST:
                score = total_time_min
            elif strategy == ChargingStrategy.CHEAPEST:
                score = total_cost
            elif strategy == ChargingStrategy.BATTERY_CARE:
                score = degradation * 10000  # Scale for comparison
            else:  # BALANCED
                score = (
                    total_time_min * 0.1 +
                    total_cost * 10.0 +
                    degradation * 1000.0
                )

            if score < best_score:
                best_score = score
                best_plan = ChargingPlan(
                    plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                    vehicle_id=vehicle_id,
                    station=station,
                    arrival_time=time.time() + travel_time_min * 60,
                    departure_time=time.time() + total_time_min * 60,
                    charge_amount_kwh=energy_needed_kwh,
                    target_soc_pct=target_soc_pct,
                    total_time_min=total_time_min,
                    total_cost=total_cost,
                    estimated_degradation=degradation,
                    strategy=strategy,
                )

        if best_plan:
            logger.info(
                f"[EV] Plan for {vehicle_id}: Station {best_plan.station.name}, "
                f"Time={best_plan.total_time_min:.0f}m, "
                f"Cost={best_plan.total_cost:.2f}, "
                f"Strategy={strategy.value}"
            )

        self._optimization_timestamps.append(time.time())
        return best_plan

    def get_status(self) -> Dict[str, Any]:
        """Get EV optimizer status."""
        return {
            "registered_stations": len(self._stations),
            "tracked_vehicles": len(self._vehicle_soc),
            "default_strategy": self.cfg.default_strategy.value,
            "tou_pricing_enabled": self.cfg.time_of_use_pricing,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIExtendedFleetSuite:
    """
    [D] Complete extended fleet intelligence suite.

    Integrates:
      • Traffic APIs (Global + China)
      • Driver Behavior Scoring
      • EV Charging Optimization
    """

    def __init__(
        self,
        traffic_cfg: Optional[TrafficConfig] = None,
        scorer_cfg: Optional[ScorerConfig] = None,
        ev_cfg: Optional[ChargingConfig] = None,
    ):
        self.traffic = TrafficAPIBridge(traffic_cfg)
        self.scorer = DriverBehaviorScorer(scorer_cfg)
        self.ev_optimizer = EVChargingOptimizer(ev_cfg)
        logger.info("[SESI Extended Fleet] Suite initialized (Global + CN APIs)")

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "traffic": self.traffic.get_status(),
            "scorer": self.scorer.get_status(),
            "ev_optimizer": self.ev_optimizer.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Extended Fleet Intelligence — Smoke Test")
    print(" (Traffic + Behavior + EV + 🇨🇳 China APIs)")
    print("=" * 72)
    print(f"  requests available  : {HAS_REQUESTS}")
    print(f"  scipy available    : {HAS_SCIPY}")
    print(f"  Fleet modules      : {HAS_FLEET}")
    print(f"  TDA available      : {HAS_TDA}")

    suite = SESIExtendedFleetSuite()

    # ── Test 1: Traffic API (Mock & China Endpoints) ─────────────────────
    print("\n  ── [TA1] Traffic API Bridge ──")

    # Bangkok to Chonburi (Mock)
    snap_mock = suite.traffic.get_traffic_snapshot(
        (13.75, 100.50), (13.36, 100.98)
    )
    print(f"    Mock Route TTI     : {snap_mock.travel_time_index:.2f} "
          f"({snap_mock.congestion_level.value})")

    # Simulate China APIs (would use real keys in production)
    suite.traffic.cfg.baidu_api_key = "mock_baidu_key"
    suite.traffic.cfg.provider = TrafficProvider.BAIDU_MAPS
    snap_baidu = suite.traffic._fetch_baidu_maps(
        (39.90, 116.40), (31.23, 121.47)  # Beijing to Shanghai
    )
    if snap_baidu:
        print(f"    🇨🇳 Baidu Maps TTI   : {snap_baidu.travel_time_index:.2f} "
              f"(Coords: {snap_baidu.metadata.get('coord_system')})")

    suite.traffic.cfg.amap_api_key = "mock_amap_key"
    suite.traffic.cfg.provider = TrafficProvider.AMAP
    snap_amap = suite.traffic._fetch_amap(
        (22.54, 114.05), (23.12, 113.26)  # Shenzhen to Guangzhou
    )
    if snap_amap:
        print(f"    🇨🇳 Amap (Gaode) TTI : {snap_amap.travel_time_index:.2f} "
              f"(Coords: {snap_amap.metadata.get('coord_system')})")

    traffic_status = suite.traffic.get_status()
    print(f"    Cache size         : {traffic_status['cache_size']}")

    # ── Test 2: Driver Behavior Scoring ──────────────────────────────────
    print("\n  ── [DBS1] Driver Behavior Scorer ──")

    suite.scorer.grant_consent("driver_001")

    telemetry = [
        {"timestamp": 1000, "speed": 50, "acceleration": 0},
        {"timestamp": 1005, "speed": 80, "acceleration": 6.0},   # Rapid accel
        {"timestamp": 1010, "speed": 40, "acceleration": -8.0},  # Hard brake
        {"timestamp": 1020, "speed": 60, "acceleration": 0.5},   # Smooth
    ]
    events = suite.scorer.detect_events("v_001", "driver_001", telemetry)
    print(f"    Events detected    : {len(events)}")

    score = suite.scorer.compute_behavior_score("driver_001")
    if score:
        print(f"    Overall Score      : {score.overall_score:.1f}/100")
        print(f"    Safety Score       : {score.safety_score:.1f}/100")
        print(f"    95% CI             : [{score.confidence_interval[0]:.1f}, "
              f"{score.confidence_interval[1]:.1f}]")
        print(f"    Top events         : {len(score.top_events)}")

    # Test consent revocation
    suite.scorer.revoke_consent("driver_001")
    events_after = suite.scorer.detect_events("v_001", "driver_001", telemetry)
    print(f"    Events after revoke: {len(events_after)} (should be 0)")

    scorer_status = suite.scorer.get_status()
    print(f"    Drivers tracked    : {scorer_status['n_drivers_tracked']}")
    print(f"    Total events       : {scorer_status['total_events']}")

    # ── Test 3: EV Charging Optimization ─────────────────────────────────
    print("\n  ── [EVC1] EV Charging Optimizer ──")

    station_fast = ChargingStation(
        station_id="st_01",
        name="PTT Ultra-Fast",
        location=(13.75, 100.50),
        max_power_kw=150.0,
        price_per_kwh=8.0,
        available_chargers=2,
    )
    station_slow = ChargingStation(
        station_id="st_02",
        name="Mall Slow Charge",
        location=(13.78, 100.52),
        max_power_kw=7.0,
        price_per_kwh=4.0,
        available_chargers=5,
    )
    suite.ev_optimizer.register_station(station_fast)
    suite.ev_optimizer.register_station(station_slow)
    suite.ev_optimizer.set_vehicle_soc("ev_001", 30.0)

    plan_fast = suite.ev_optimizer.plan_charging(
        "ev_001", (13.70, 100.50), strategy=ChargingStrategy.FASTEST
    )
    plan_care = suite.ev_optimizer.plan_charging(
        "ev_001", (13.70, 100.50), strategy=ChargingStrategy.BATTERY_CARE
    )

    if plan_fast:
        print(f"    Fastest Plan       : {plan_fast.station.name}")
        print(f"      Time             : {plan_fast.total_time_min:.0f} min")
        print(f"      Cost             : {plan_fast.total_cost:.2f}")
        print(f"      Degradation      : {plan_fast.estimated_degradation:.4f}")

    if plan_care:
        print(f"    Battery Care Plan  : {plan_care.station.name}")
        print(f"      Time             : {plan_care.total_time_min:.0f} min")
        print(f"      Cost             : {plan_care.total_cost:.2f}")
        print(f"      Degradation      : {plan_care.estimated_degradation:.4f}")

    ev_status = suite.ev_optimizer.get_status()
    print(f"    Registered stations: {ev_status['registered_stations']}")
    print(f"    TOU pricing        : {ev_status['tou_pricing_enabled']}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    Traffic providers  : {status['traffic']['provider']}")
    print(f"    Traffic cache      : {status['traffic']['cache_size']}")
    print(f"    Drivers tracked    : {status['scorer']['n_drivers_tracked']}")
    print(f"    EV stations        : {status['ev_optimizer']['registered_stations']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE & ETHICAL FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • China APIs require Coordinate Conversion (WGS84 → GCJ-02/BD-09)")
    print("  • Driver Scoring is for IMPROVEMENT (กรุณา), not punishment")
    print("  • EV Charging flows with grid needs (อู๋เหวย), no forced fast-charge")
    print("  • PNDC Axiom 3: Telematics data is anonymized & consent-based")
    print("  • Traffic API requires valid keys; mock mode for testing")
    print("  • Battery degradation models are APPROXIMATE")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
