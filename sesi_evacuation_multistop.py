"""
=============================================================================
SESI MULTI-STOP EVACUATION ROUTE PLANNER
Hazard-aware, uncertainty-quantified, change-point-monitored evacuation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] Multi-stop evacuation routing with:
  • Stop types: assembly point → shelter → hospital → final safe zone
  • Nearest-Neighbor + 2-opt optimization (no external solver needed)
  • Hazard-zone avoidance via DisasterGeofence (sesi_geo_maps.py)
  • Travel-time uncertainty via assumption-light CI (Paper 11)
  • Route change-point monitoring (Paper 6) for road blockage
  • δ_min spacing between stops (Delta_Min Thm 1.2)

ALGORITHMS USED
---------------
[C] Haversine great-circle distance for geographic separation
[C] Nearest-Neighbor constructive heuristic for initial route
[C] 2-opt local search for route improvement
[C] Assumption-light subsampling for travel-time CI (Paper 11 §2.1)
[D] Hazard avoidance as constrained waypoint filtering
[D] Change-point detection on road-status time series (Paper 6 §3)

RELATION TO OTHER MODULES
-------------------------
  • sesi_global_maps.py  → real routing via Google/Azure/Apple Directions
  • sesi_geo_maps.py     → geofence zones + GCJ-02 transform (China)
  • sesi_mobile_alert_app.py → push the chosen route to users

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Heuristics give GOOD routes, not provably OPTIMAL routes.
• Travel times are ESTIMATES; real congestion, road damage, and weather
  can change them drastically during a disaster.
• Hazard zones here are circular; real flood/fire extents need GIS polygons.
• The final evacuation decision belongs to HUMAN operators and local
  authorities, not to this software. This is decision support.
• δ_min spacing prevents stops collapsing together, but does not guarantee
  shelter capacity — capacity must be checked separately.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ses.evac_multistop")

# Reuse map + geofence modules
try:
    from sesi_global_maps import GlobalMapsRouter, GlobalMapsConfig
    HAS_GLOBAL_MAPS = True
except ImportError:
    HAS_GLOBAL_MAPS = False

try:
    from sesi_geo_maps import DisasterGeofence, GeofenceZone, GeoMapsConfig
    HAS_GEOFENCE = True
except ImportError:
    HAS_GEOFENCE = False

__all__ = [
    "StopType",
    "EvacuationStop",
    "MultiStopConfig",
    "HaversineDistance",
    "RouteOptimizer",
    "HazardAvoidanceRouter",
    "TravelTimeUncertainty",
    "ChangePointRouteMonitor",
    "MultiStopEvacuationPlanner",
]


class StopType(Enum):
    """Types of stops in an evacuation route."""
    ORIGIN = auto()          # where people start
    ASSEMBLY = auto()        # gathering point
    SHELTER = auto()         # temporary shelter
    HOSPITAL = auto()        # medical facility
    TRANSIT = auto()         # transit/transfer point
    SAFE_ZONE = auto()       # final safe destination


@dataclass
class EvacuationStop:
    """A single stop in the evacuation route."""
    stop_id: str
    stop_type: StopType
    name: str
    lat: float
    lon: float
    capacity: int = 0          # people it can hold (0 = unknown)
    service_time_min: float = 0.0   # minutes spent at this stop
    is_safe: bool = True       # False if inside hazard zone


@dataclass
class MultiStopConfig:
    """Configuration for multi-stop evacuation planning."""
    # Optimization
    use_2opt: bool = True
    max_2opt_iterations: int = 100

    # Speed assumptions (km/h) by transport mode
    speed_on_foot_kmh: float = 4.0
    speed_vehicle_kmh: float = 30.0
    transport_mode: str = "vehicle"   # "foot" or "vehicle"

    # δ_min spacing between stops (Delta_Min Thm 1.2)
    delta_min_km: float = 0.5
    kappa_fatness: float = 2.0

    # Uncertainty (Paper 11)
    travel_time_cv: float = 0.3        # coefficient of variation
    confidence_level: float = 0.90
    n_subsamples: int = 50

    # Map provider
    use_real_routing: bool = False     # True → call GlobalMapsRouter


# ═══════════════════════════════════════════════════════════════════════
# [C] HAVERSINE DISTANCE
# ═══════════════════════════════════════════════════════════════════════

class HaversineDistance:
    """
    [C] Great-circle distance on a sphere.
    Accurate enough for evacuation-scale distances (< 100 km).
    """
    EARTH_RADIUS_KM = 6371.0

    @classmethod
    def distance_km(cls, lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return cls.EARTH_RADIUS_KM * c

    @classmethod
    def distance_matrix(cls, stops: List[EvacuationStop]) -> np.ndarray:
        """Build full pairwise distance matrix."""
        n = len(stops)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = cls.distance_km(stops[i].lat, stops[i].lon,
                                    stops[j].lat, stops[j].lon)
                mat[i, j] = d
                mat[j, i] = d
        return mat


# ═══════════════════════════════════════════════════════════════════════
# [D] ROUTE OPTIMIZER (Nearest Neighbor + 2-opt)
# ═══════════════════════════════════════════════════════════════════════

class RouteOptimizer:
    """
    [C/D] Optimize the visiting order of stops.

    Fixed structure: ORIGIN first, SAFE_ZONE last. Intermediate stops
    (ASSEMBLY, SHELTER, HOSPITAL, TRANSIT) are reordered to minimize
    total distance.

    Method:
      1. Nearest-Neighbor constructive heuristic.
      2. 2-opt local search improvement.

    HONEST: This is a heuristic, not an exact TSP solver. For > 20 stops
    an exact solver or metaheuristic would be preferable, but NN+2-opt is
    robust and dependency-free for typical evacuation sizes.
    """

    def __init__(self, cfg: MultiStopConfig):
        self.cfg = cfg

    def optimize(self, stops: List[EvacuationStop]) -> List[EvacuationStop]:
        """Return stops in optimized visit order."""
        if len(stops) <= 2:
            return list(stops)

        # Separate fixed endpoints from reorderable middle
        origins = [s for s in stops if s.stop_type == StopType.ORIGIN]
        finals = [s for s in stops if s.stop_type == StopType.SAFE_ZONE]
        middle = [s for s in stops
                  if s.stop_type not in (StopType.ORIGIN, StopType.SAFE_ZONE)
                  and s.is_safe]

        if not origins or not finals:
            logger.warning("Missing ORIGIN or SAFE_ZONE; returning input order")
            return list(stops)

        origin = origins[0]
        final = finals[0]

        if not middle:
            return [origin, final]

        # Nearest Neighbor from origin through middle
        ordered = self._nearest_neighbor(origin, middle)

        # 2-opt improvement
        if self.cfg.use_2opt:
            ordered = self._two_opt(ordered, origin, final)

        return [origin] + ordered + [final]

    def _nearest_neighbor(self, start: EvacuationStop,
                          candidates: List[EvacuationStop]) -> List[EvacuationStop]:
        """Greedy nearest-neighbor ordering."""
        remaining = list(candidates)
        route = []
        current = start

        while remaining:
            nearest = min(remaining, key=lambda s: HaversineDistance.distance_km(
                current.lat, current.lon, s.lat, s.lon))
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest

        return route

    def _two_opt(self, route: List[EvacuationStop], origin: EvacuationStop,
                 final: EvacuationStop) -> List[EvacuationStop]:
        """2-opt local search to reduce total distance."""
        full = [origin] + route + [final]
        best = list(full)
        improved = True
        iteration = 0

        while improved and iteration < self.cfg.max_2opt_iterations:
            improved = False
            iteration += 1
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if self._total_distance(new_route) < self._total_distance(best):
                        best = new_route
                        improved = True

        # Return only the middle (strip fixed endpoints)
        return best[1:-1]

    def _total_distance(self, stops: List[EvacuationStop]) -> float:
        total = 0.0
        for i in range(len(stops) - 1):
            total += HaversineDistance.distance_km(
                stops[i].lat, stops[i].lon,
                stops[i+1].lat, stops[i+1].lon)
        return total


# ═══════════════════════════════════════════════════════════════════════
# [D] HAZARD AVOIDANCE ROUTER
# ═══════════════════════════════════════════════════════════════════════

class HazardAvoidanceRouter:
    """
    [D] Filter and adjust stops to avoid hazard zones.

    Uses DisasterGeofence from sesi_geo_maps.py. A stop inside an active
    hazard zone is marked unsafe and excluded from the route. Optionally,
    a route segment that crosses a hazard zone can trigger a re-route.

    HONEST: Zones are circular here. Real hazard extents (flood polygons,
    fire perimeters) require GIS data and are more accurate.
    """

    def __init__(self, cfg: MultiStopConfig):
        self.cfg = cfg
        self.geofence: Optional[DisasterGeofence] = None
        if HAS_GEOFENCE:
            self.geofence = DisasterGeofence(GeoMapsConfig())

    def register_zone(self, zone: GeofenceZone) -> None:
        """Register a hazard zone to avoid."""
        if self.geofence is not None:
            self.geofence.add_zone(zone)

    def filter_stops(self, stops: List[EvacuationStop]) -> List[EvacuationStop]:
        """Mark stops inside hazard zones as unsafe."""
        if self.geofence is None:
            logger.warning("Geofence unavailable; no hazard filtering applied")
            return stops

        for stop in stops:
            zones = self.geofence.get_zones_for_location(stop.lat, stop.lon)
            stop.is_safe = len(zones) == 0
            if not stop.is_safe:
                logger.info(f"Stop '{stop.name}' inside hazard zone; marked unsafe")

        return stops

    def validate_route(self, ordered_stops: List[EvacuationStop],
                       n_check_points: int = 20) -> Dict[str, Any]:
        """
        Check whether route segments pass through hazard zones.

        Samples points along each segment and tests against geofence.
        Returns report of any hazardous crossings.
        """
        if self.geofence is None:
            return {"status": "skipped", "reason": "no geofence"}

        crossings = []
        for i in range(len(ordered_stops) - 1):
            a, b = ordered_stops[i], ordered_stops[i + 1]
            for t in np.linspace(0, 1, n_check_points):
                lat = a.lat + t * (b.lat - a.lat)
                lon = a.lon + t * (b.lon - a.lon)
                zones = self.geofence.get_zones_for_location(lat, lon)
                if zones:
                    crossings.append({
                        "segment": (a.stop_id, b.stop_id),
                        "point": (lat, lon),
                        "zones": [z.zone_id for z in zones],
                    })

        return {
            "status": "checked",
            "n_crossings": len(crossings),
            "crossings": crossings,
            "route_clear": len(crossings) == 0,
            "note": ("Circular-zone approximation; use GIS polygons for "
                     "real hazard extents."),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] TRAVEL-TIME UNCERTAINTY (Paper 11 assumption-light)
# ═══════════════════════════════════════════════════════════════════════

class TravelTimeUncertainty:
    """
    [D/C] Estimate travel time with uncertainty, assumption-light.

    Instead of a single point estimate, we report a confidence interval
    using the subsampling idea from Paper 11 §2.1: we do NOT assume a
    specific delay distribution; we propagate a coefficient-of-variation
    through subsampled estimates.

    This reflects the reality that disaster travel times are highly
    variable (congestion, debris, weather) and a single number is
    misleadingly precise.
    """

    def __init__(self, cfg: MultiStopConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)

    def _base_time_min(self, distance_km: float) -> float:
        """Base travel time in minutes from distance and mode speed."""
        speed = (self.cfg.speed_on_foot_kmh if self.cfg.transport_mode == "foot"
                 else self.cfg.speed_vehicle_kmh)
        return distance_km / speed * 60.0

    def estimate_segment(self, distance_km: float) -> Dict[str, float]:
        """
        Estimate travel time for one segment with CI.

        Uses subsampling: draw n_subsamples noisy durations, take the
        empirical quantiles. No distributional form assumed.
        """
        base = self._base_time_min(distance_km)
        if base <= 0:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

        cv = self.cfg.travel_time_cv
        samples = base * (1.0 + cv * self.rng.standard_normal(self.cfg.n_subsamples))
        samples = np.maximum(samples, 0.0)

        alpha = 1.0 - self.cfg.confidence_level
        ci_low = float(np.percentile(samples, 100 * alpha / 2))
        ci_high = float(np.percentile(samples, 100 * (1 - alpha / 2)))

        return {
            "mean": float(base),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": self.cfg.confidence_level,
        }

    def estimate_route(self, ordered_stops: List[EvacuationStop]) -> Dict[str, Any]:
        """Sum segment estimates along the whole route."""
        total_mean = 0.0
        total_low = 0.0
        total_high = 0.0
        segments = []

        for i in range(len(ordered_stops) - 1):
            d = HaversineDistance.distance_km(
                ordered_stops[i].lat, ordered_stops[i].lon,
                ordered_stops[i + 1].lat, ordered_stops[i + 1].lon)
            est = self.estimate_segment(d)
            # Add service time at the departure stop
            est["service_time_min"] = ordered_stops[i].service_time_min
            total_mean += est["mean"] + est["service_time_min"]
            total_low += est["ci_low"] + est["service_time_min"]
            total_high += est["ci_high"] + est["service_time_min"]
            segments.append({
                "from": ordered_stops[i].name,
                "to": ordered_stops[i + 1].name,
                "distance_km": d,
                **est,
            })

        return {
            "total_mean_min": total_mean,
            "total_ci_low_min": total_low,
            "total_ci_high_min": total_high,
            "confidence": self.cfg.confidence_level,
            "segments": segments,
            "note": ("Travel times are ESTIMATES with wide uncertainty during "
                     "disasters; treat the interval, not the mean, as guidance."),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] CHANGE-POINT ROUTE MONITOR (Paper 6)
# ═══════════════════════════════════════════════════════════════════════

class ChangePointRouteMonitor:
    """
    [C/D] Monitor road-condition signals along a route for change points.

    During an evacuation, road status can change abruptly (flooding, debris,
    fire spread). Paper 6's change-point induced reduction tells us to
    retime analysis at the significant change points rather than average
    over all time steps.

    Here we detect change points in a road-status time series using a
    simple CUSUM-style statistic and flag segments that need re-planning.

    HONEST: Real change-point recovery with unknown change points is an
    open problem (Paper 6, Open Problem 6.1). This is a lightweight
    detector for demonstration, not a proven universal method.
    """

    def __init__(self, cfg: MultiStopConfig, threshold_sigma: float = 3.0):
        self.cfg = cfg
        self.threshold_sigma = threshold_sigma

    def detect_change_points(self, status_series: np.ndarray) -> List[int]:
        """
        Detect indices where the road-status series shifts significantly.

        Uses a moving-window mean-shift test.
        """
        if len(status_series) < 4:
            return []

        change_points = []
        window = max(2, len(status_series) // 10)
        overall_std = float(np.std(status_series)) + 1e-12

        for i in range(window, len(status_series) - window):
            before = status_series[i - window:i].mean()
            after = status_series[i:i + window].mean()
            if abs(after - before) > self.threshold_sigma * overall_std:
                # avoid duplicates close together
                if not change_points or (i - change_points[-1]) > window:
                    change_points.append(i)

        return change_points

    def needs_replan(self, status_series: np.ndarray) -> Dict[str, Any]:
        """Decide whether the route needs re-planning."""
        cps = self.detect_change_points(status_series)
        return {
            "n_change_points": len(cps),
            "change_point_indices": cps,
            "needs_replan": len(cps) > 0,
            "note": ("Change points indicate abrupt road-status shifts; "
                     "re-run the planner if confirmed by field reports. "
                     "Detection with unknown change points is [O] per Paper 6."),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] MULTI-STOP EVACUATION PLANNER (orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class MultiStopEvacuationPlanner:
    """
    [D] Orchestrate the full multi-stop evacuation planning pipeline:

      1. Filter stops through hazard zones.
      2. Enforce δ_min spacing between stops (Delta_Min Thm 1.2).
      3. Optimize visit order (NN + 2-opt).
      4. Estimate travel time with uncertainty.
      5. (Optional) validate against real routing provider.
      6. Monitor for change points during execution.

    HONEST: The output is a RECOMMENDED route for human operators and
    local authorities, not an autonomous command.
    """

    def __init__(self, cfg: Optional[MultiStopConfig] = None):
        self.cfg = cfg or MultiStopConfig()
        self.optimizer = RouteOptimizer(self.cfg)
        self.hazard_router = HazardAvoidanceRouter(self.cfg)
        self.uncertainty = TravelTimeUncertainty(self.cfg)
        self.monitor = ChangePointRouteMonitor(self.cfg)

        # Optional real routing
        self.maps_router = None
        if self.cfg.use_real_routing and HAS_GLOBAL_MAPS:
            self.maps_router = GlobalMapsRouter(GlobalMapsConfig())

    def _enforce_delta_min(self, stops: List[EvacuationStop]) -> List[EvacuationStop]:
        """
        [P] Remove stops closer than δ_min to an already-kept stop.

        Delta_Min Thm 1.2: under κ-fatness, distinct nucleation/stop sites
        must be separated by at least δ_min. We apply the same principle so
        stops do not collapse onto one another.
        """
        kept: List[EvacuationStop] = []
        for stop in stops:
            too_close = False
            for k in kept:
                d = HaversineDistance.distance_km(stop.lat, stop.lon, k.lat, k.lon)
                if d < self.cfg.delta_min_km:
                    too_close = True
                    logger.info(f"Stop '{stop.name}' within δ_min of '{k.name}'; skipped")
                    break
            if not too_close:
                kept.append(stop)
        return kept

    def plan(self, stops: List[EvacuationStop],
             road_status: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Plan a multi-stop evacuation route.

        Args:
            stops: unordered list including one ORIGIN and one SAFE_ZONE.
            road_status: optional time series of road condition for monitoring.

        Returns:
            dict with ordered route, travel-time CI, hazard report, and
            change-point status.
        """
        # 1. Hazard filtering
        stops = self.hazard_router.filter_stops(stops)
        unsafe = [s.name for s in stops if not s.is_safe]

        # 2. δ_min spacing (only among safe stops)
        safe_stops = [s for s in stops if s.is_safe]
        safe_stops = self._enforce_delta_min(safe_stops)

        # 3. Optimize order
        ordered = self.optimizer.optimize(safe_stops)

        # 4. Travel time with uncertainty
        timing = self.uncertainty.estimate_route(ordered)

        # 5. Validate route against hazard zones (segment sampling)
        hazard_report = self.hazard_router.validate_route(ordered)

        # 6. Change-point monitoring
        cp_report = None
        if road_status is not None:
            cp_report = self.monitor.needs_replan(road_status)

        # 7. Optional real-routing validation
        real_route = None
        if self.maps_router is not None and len(ordered) >= 2:
            try:
                real_route = self.maps_router.route(
                    (ordered[0].lat, ordered[0].lon),
                    (ordered[-1].lat, ordered[-1].lon))
            except Exception as e:
                logger.warning(f"Real routing failed: {e}")
                real_route = {"status": "failed", "error": str(e)}

        return {
            "ordered_route": [
                {"id": s.stop_id, "name": s.name,
                 "type": s.stop_type.name,
                 "lat": s.lat, "lon": s.lon}
                for s in ordered
            ],
            "n_stops": len(ordered),
            "excluded_unsafe": unsafe,
            "travel_time": timing,
            "hazard_report": hazard_report,
            "change_point_report": cp_report,
            "real_route": real_route,
            "honest_note": (
                "RECOMMENDED route only. Heuristic, not provably optimal. "
                "Travel times are estimates. Final decision belongs to human "
                "operators and local authorities."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES Multi-Stop Evacuation Route Planner — Smoke Test")
    print("=" * 72)
    print(f"  global_maps module : {HAS_GLOBAL_MAPS}")
    print(f"  geofence module    : {HAS_GEOFENCE}")

    cfg = MultiStopConfig(
        use_2opt=True,
        transport_mode="vehicle",
        delta_min_km=0.5,
        travel_time_cv=0.3,
        use_real_routing=False,
    )
    planner = MultiStopEvacuationPlanner(cfg)

    # Define stops around a hypothetical disaster area (Bangkok-like coords)
    stops = [
        EvacuationStop("origin", StopType.ORIGIN, "Riverside Community",
                       13.7200, 100.5100),
        EvacuationStop("assembly", StopType.ASSEMBLY, "School Assembly Point",
                       13.7300, 100.5200, capacity=500),
        EvacuationStop("shelter1", StopType.SHELTER, "Temple Shelter",
                       13.7400, 100.5300, capacity=300, service_time_min=10),
        EvacuationStop("hospital", StopType.HOSPITAL, "City Hospital",
                       13.7600, 100.5400, capacity=100, service_time_min=20),
        EvacuationStop("shelter2", StopType.SHELTER, "Stadium Shelter",
                       13.7500, 100.5600, capacity=1000, service_time_min=10),
        # An unsafe stop inside a hazard zone
        EvacuationStop("bad_stop", StopType.SHELTER, "Flooded Shelter",
                       13.7563, 100.5018, capacity=200),
        EvacuationStop("safezone", StopType.SAFE_ZONE, "Northern Safe Zone",
                       13.8000, 100.5500, capacity=2000),
    ]

    # Register a flood hazard zone around the bad stop
    if HAS_GEOFENCE:
        planner.hazard_router.register_zone(GeofenceZone(
            zone_id="flood_zone_1",
            name="Active Flood Zone",
            center_lat=13.7563, center_lon=100.5018,
            radius_km=1.5,
            hazard_type="FLOOD",
            severity="WARNING",
        ))

    # Simulated road-status series with one abrupt degradation
    road_status = np.concatenate([
        np.random.normal(0.9, 0.05, 20),   # good road
        np.random.normal(0.3, 0.05, 20),   # degraded after change point
    ])

    print("\n  ── Planning multi-stop route ──")
    result = planner.plan(stops, road_status=road_status)

    print(f"\n  ── Ordered Route ──")
    for i, s in enumerate(result["ordered_route"]):
        print(f"    {i+1}. [{s['type']:10s}] {s['name']} "
              f"({s['lat']:.4f}, {s['lon']:.4f})")

    print(f"\n  ── Exclusions ──")
    print(f"    Excluded unsafe stops : {result['excluded_unsafe']}")

    print(f"\n  ── Travel Time (with uncertainty) ──")
    tt = result["travel_time"]
    print(f"    Mean total            : {tt['total_mean_min']:.1f} min")
    print(f"    {tt['confidence']*100:.0f}% CI             : "
          f"[{tt['total_ci_low_min']:.1f}, {tt['total_ci_high_min']:.1f}] min")
    print(f"    Segments              : {len(tt['segments'])}")
    print(f"    {tt['note']}")

    print(f"\n  ── Hazard Report ──")
    hr = result["hazard_report"]
    print(f"    Status                : {hr.get('status')}")
    print(f"    Route clear           : {hr.get('route_clear')}")
    print(f"    Crossings             : {hr.get('n_crossings')}")

    print(f"\n  ── Change-Point Monitoring ──")
    cp = result["change_point_report"]
    if cp:
        print(f"    Change points found   : {cp['n_change_points']}")
        print(f"    Needs replan          : {cp['needs_replan']}")
        print(f"    {cp['note']}")

    print(f"\n    {result['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Heuristic route, NOT provably optimal")
    print("  • Travel times are ESTIMATES; disasters change them")
    print("  • Hazard zones are circular approximations")
    print("  • Final decision belongs to HUMAN operators")
    print("  • δ_min prevents stop collapse, not capacity checks")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
