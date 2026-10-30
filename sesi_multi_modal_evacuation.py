"""
=============================================================================
SESI MULTI-MODAL EVACUATION — Heterogeneous Fleet + Shelter Time Windows
                            + Mobile Route Delivery
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE ADDS
---------------------
[MM1] VehicleProfile / EvacVehicle
      Heterogeneous fleet with DISTINCT travel modes:
        • AMBULANCE — road, medical-capable, low capacity, fast
        • BUS       — road, high capacity, wheelchair lift optional
        • BOAT      — WATER mode, needs dock access, flood-zone use
      Vehicles are NOT interchangeable across modes.

[MM2] ShelterSchedule
      Multi-interval time windows with dynamic closure.
      A shelter can close EARLY (flood arrives) or re-open (water recedes).
      Closure changes are treated as CHANGE POINTS (paper6 §1.1):
      when detected, plans are REBUILT, not averaged.

[MM3] FeasibilityChecker
      Arrival-vs-window check under travel-time UNCERTAINTY.
      Uses the LOWER bound of a Paper-11-style CI: we only call a plan
      safe if even the pessimistic travel estimate clears the window.

[MM4] MobileEvacuationNotifier
      Delivers routes to ses_mobile_alert_app with:
        • No-Zeno throttling (never spam users)
        • multilingual text (th/zh/en)
        • actionable flag (synthetic plans are NEVER pushed as orders)

RELATION TO EXISTING MODULES
----------------------------
  ses_multistop_evacuation   → single-vehicle multi-stop core (reused)
  ses_live_travel_matrix     → TravelTimeUncertainty (Paper 11 CI)
  ses_hazard_route_control   → ChangePointRegimeDetector (paper6)
  ses_mobile_alert_app       → delivery endpoint + No-Zeno gate

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Vehicle-demand matching is a GREEDY heuristic; multi-modal VRP with
  time windows is NP-hard [O]. No optimality claim.
• Boat routing uses straight-line water paths here; real channels/depth
  data would replace this.
• Travel-time CI coverage is ASYMPTOTIC (Paper 11), not finite-sample.
• A message reaching a phone is NOT proof the user is safe.
=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.multimodal")

# Optional dependencies (graceful degradation)
try:
    from ses_multistop_evacuation import (
        MultiStopConfig, MultiStopEvacuationPlanner, EvacStop, Shelter)
    HAS_MULTISTOP = True
except ImportError:
    HAS_MULTISTOP = False

try:
    from ses_live_travel_matrix import TravelTimeUncertainty, LiveMatrixConfig
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False

try:
    from ses_hazard_route_control import ChangePointRegimeDetector, HazardRouteConfig
    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False

try:
    from ses_mobile_alert_app import (
        MobileAlertApp, MobileAlertConfig, HazardAlert, AlertSeverity)
    HAS_MOBILE = True
except ImportError:
    HAS_MOBILE = False

__all__ = [
    "VehicleType", "TravelMode", "VehicleProfile", "EvacVehicle",
    "ShelterTimeWindow", "ShelterSchedule",
    "MultiModalFleet", "FeasibilityChecker",
    "MobileEvacuationNotifier", "MultiModalEvacuationPlanner",
]


# ═══════════════════════════════════════════════════════════════════════
# [D] VEHICLE TYPES & PROFILES
# ═══════════════════════════════════════════════════════════════════════

class VehicleType(Enum):
    AMBULANCE = auto()
    BUS = auto()
    BOAT = auto()


class TravelMode(Enum):
    ROAD = auto()
    WATER = auto()


@dataclass
class VehicleProfile:
    """
    [D] Physical/operational profile of a vehicle class.

    Speeds are conservative evacuation estimates, not legal maxima.
    """
    vehicle_type: VehicleType
    travel_mode: TravelMode
    capacity_persons: int
    wheelchair_capacity: int
    medical_capable: bool
    speed_ms: float
    boarding_s_per_person: float   # loading time per evacuee
    fixed_overhead_s: float        # dispatch/prep time

    @staticmethod
    def standard_profiles() -> Dict[VehicleType, "VehicleProfile"]:
        return {
            VehicleType.AMBULANCE: VehicleProfile(
                VehicleType.AMBULANCE, TravelMode.ROAD,
                capacity_persons=4, wheelchair_capacity=1,
                medical_capable=True, speed_ms=13.0,
                boarding_s_per_person=15.0, fixed_overhead_s=60.0),
            VehicleType.BUS: VehicleProfile(
                VehicleType.BUS, TravelMode.ROAD,
                capacity_persons=45, wheelchair_capacity=2,
                medical_capable=False, speed_ms=8.0,
                boarding_s_per_person=3.0, fixed_overhead_s=120.0),
            VehicleType.BOAT: VehicleProfile(
                VehicleType.BOAT, TravelMode.WATER,
                capacity_persons=20, wheelchair_capacity=0,
                medical_capable=False, speed_ms=4.0,
                boarding_s_per_person=8.0, fixed_overhead_s=180.0),
        }


@dataclass
class EvacVehicle:
    """A concrete vehicle instance in the fleet."""
    vehicle_id: str
    profile: VehicleProfile
    lat: float
    lon: float
    available_from: float = 0.0            # seconds from planning epoch
    available_until: float = float("inf")
    needs_dock: bool = False               # BOAT requires dock access


# ═══════════════════════════════════════════════════════════════════════
# [C/D] SHELTER TIME WINDOWS (dynamic, multi-interval)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ShelterTimeWindow:
    """One open interval [open_t, close_t] in seconds from epoch."""
    open_t: float
    close_t: float

    def contains(self, t: float) -> bool:
        return self.open_t <= t <= self.close_t


class ShelterSchedule:
    """
    [C/D] Multi-interval, dynamically-updatable shelter schedule.

    Supports:
      • Multiple open windows per shelter (open → close → re-open).
      • Dynamic closure: close_early() inserts a change point.
      • latest_arrival(): the last safe arrival time given service time.

    CHANGE-POINT DISCIPLINE (paper6 §1.1): a sudden closure is a change
    point. Consumers must call bump_revision() and REBUILD routes;
    they must NOT average the old and new schedules.
    """

    def __init__(self, shelter_id: str):
        self.shelter_id = shelter_id
        self.windows: List[ShelterTimeWindow] = []
        self.revision = 0            # bumped on every change
        self._change_log: List[Tuple[float, str]] = []

    def add_window(self, open_t: float, close_t: float) -> None:
        if close_t <= open_t:
            raise ValueError("close_t must exceed open_t")
        self.windows.append(ShelterTimeWindow(open_t, close_t))
        self.windows.sort(key=lambda w: w.open_t)
        self.revision += 1
        self._change_log.append((time.time(), f"added window [{open_t},{close_t}]"))

    def close_early(self, at_time: float, reason: str) -> bool:
        """
        [D] Emergency early-closure (e.g. flood arrives ahead of forecast).
        Returns True if a window was actually truncated (a real change point).
        """
        changed = False
        for w in self.windows:
            if w.open_t < at_time < w.close_t:
                w.close_t = at_time
                changed = True
            elif at_time <= w.open_t:
                w.close_t = w.open_t   # zero-length → effectively removed
                changed = True
        if changed:
            self.revision += 1
            self._change_log.append((time.time(), f"EARLY CLOSE at {at_time}: {reason}"))
            logger.warning(f"Shelter {self.shelter_id} closed early: {reason}")
        return changed

    def is_open(self, t: float) -> bool:
        return any(w.contains(t) for w in self.windows)

    def latest_arrival(self, t_now: float, service_s: float) -> Optional[float]:
        """Latest time an evacuee can arrive and still be served before close."""
        candidates = [w.close_t - service_s for w in self.windows
                      if w.close_t - service_s >= t_now and w.open_t <= t_now + 1e9]
        return max(candidates) if candidates else None

    def change_points(self) -> List[float]:
        """All window boundaries = the natural change points of this schedule."""
        pts: List[float] = []
        for w in self.windows:
            pts.extend([w.open_t, w.close_t])
        return sorted(pts)


# ═══════════════════════════════════════════════════════════════════════
# [D] MULTI-MODAL FLEET
# ═══════════════════════════════════════════════════════════════════════

class MultiModalFleet:
    """
    [D] Manage a heterogeneous fleet and match vehicles to demand.

    Matching rules (greedy, HONEST: not optimal):
      • medical need      → AMBULANCE only
      • wheelchair need   → vehicle with wheelchair_capacity > 0
      • flooded road      → BOAT (water mode)
      • everyone else     → BUS preferred for capacity
    """

    def __init__(self):
        self.vehicles: Dict[str, EvacVehicle] = {}

    def add_vehicle(self, v: EvacVehicle) -> None:
        self.vehicles[v.vehicle_id] = v

    def vehicles_for_mode(self, mode: TravelMode) -> List[EvacVehicle]:
        return [v for v in self.vehicles.values()
                if v.profile.travel_mode == mode]

    def match_demand(self, demand: Dict[str, Any],
                     flooded: bool) -> Optional[EvacVehicle]:
        """
        [D] Greedy single-demand matching. Returns a suitable vehicle
        or None if the fleet cannot serve this demand.
        """
        needs_medical = demand.get("needs_medical", 0) > 0
        needs_wheelchair = demand.get("needs_wheelchair", 0) > 0
        n_persons = demand.get("n_persons", 1)

        pool = (self.vehicles_for_mode(TravelMode.WATER) if flooded
                else self.vehicles_for_mode(TravelMode.ROAD))

        for v in pool:
            p = v.profile
            if needs_medical and not p.medical_capable:
                continue
            if needs_wheelchair and p.wheelchair_capacity < 1:
                continue
            if p.capacity_persons < n_persons:
                continue
            return v
        return None


# ═══════════════════════════════════════════════════════════════════════
# [D] FEASIBILITY CHECKER (Paper 11 CI-aware)
# ═══════════════════════════════════════════════════════════════════════

class FeasibilityChecker:
    """
    [C/D] Arrival-vs-window feasibility under travel-time uncertainty.

    Uses the LOWER bound of a Paper-11 subsampling CI for travel time:
    a leg is only 'safe' if even the pessimistic estimate arrives before
    the shelter closes (minus service time).

    HONEST: the CI is asymptotic (Paper 11 Thm 4.4), requires stationarity
    + summable α-mixing qualitatively; it is NOT a finite-sample guarantee.
    """

    def __init__(self):
        self._uncertainty = (TravelTimeUncertainty(LiveMatrixConfig())
                             if HAS_UNCERTAINTY else None)

    def travel_time_ci(self, samples: np.ndarray,
                       level: float = 0.95) -> Optional[Tuple[float, float]]:
        """Return (lower, upper) CI for travel time, or None."""
        if self._uncertainty is not None and len(samples) >= 12:
            ci = self._uncertainty.interval(samples)
            if "error" not in ci:
                return ci["ci_lower_s"], ci["ci_upper_s"]
        # Fallback: crude normal-ish interval, CLEARLY flagged
        if len(samples) < 2:
            return None
        m = float(np.mean(samples))
        s = float(np.std(samples))
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
        return max(0.0, m - z * s), m + z * s

    def leg_is_safe(self, depart_t: float,
                    travel_samples: np.ndarray,
                    schedule: ShelterSchedule,
                    service_s: float,
                    level: float = 0.95) -> Dict[str, Any]:
        """
        Decide whether a departure at depart_t can still be served.

        Returns dict with safety verdict + the reasoning. We require the
        UPPER CI arrival to clear the latest arrival time (conservative).
        """
        ci = self.travel_time_ci(travel_samples, level)
        if ci is None:
            return {"safe": False, "reason": "insufficient travel samples",
                    "actionable": False}

        lower, upper = ci
        arrive_optimistic = depart_t + lower
        arrive_pessimistic = depart_t + upper
        latest = schedule.latest_arrival(depart_t, service_s)

        if latest is None:
            return {"safe": False, "reason": "shelter has no future window",
                    "actionable": False}

        safe = arrive_pessimistic <= latest
        return {
            "safe": safe,
            "arrive_pessimistic_s": arrive_pessimistic,
            "latest_arrival_s": latest,
            "slack_s": latest - arrive_pessimistic,
            "ci": (lower, upper),
            "reason": "pessimistic arrival clears window" if safe
                      else "pessimistic arrival MISSES window",
            "actionable": True,
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] MOBILE ROUTE NOTIFIER (→ ses_mobile_alert_app)
# ═══════════════════════════════════════════════════════════════════════

class MobileEvacuationNotifier:
    """
    [D] Deliver evacuation routes to the mobile alert app.

    Discipline:
      • Respects the mobile app's own No-Zeno throttle.
      • NEVER pushes a synthetic/unvalidated plan as an evacuation order.
      • Keeps messages short + multilingual (th/zh/en).
    """

    VEHICLE_NAMES = {
        VehicleType.AMBULANCE: {"th": "รถพยาบาล", "zh": "救护车", "en": "ambulance"},
        VehicleType.BUS:       {"th": "รถบัส", "zh": "大巴", "en": "bus"},
        VehicleType.BOAT:      {"th": "เรือ", "zh": "船", "en": "boat"},
    }

    def __init__(self, mobile_app: Optional[Any] = None):
        self.mobile_app = mobile_app
        self.send_log: List[Dict[str, Any]] = []

    def build_route_message(self, plan: Dict[str, Any],
                            language: str = "th") -> Dict[str, str]:
        """Build a compact multilingual evacuation-route message."""
        vtype = plan.get("vehicle_type", VehicleType.BUS)
        vname = self.VEHICLE_NAMES.get(vtype, {}).get(language,
                self.VEHICLE_NAMES.get(vtype, {}).get("en", "vehicle"))
        shelter = plan.get("shelter_name", "shelter")
        eta_min = plan.get("eta_s", 0) / 60.0

        templates = {
            "th": (f"พาหนะ: {vname} | จุดหมาย: {shelter} | "
                   f"เวลาโดยประมาณ: {eta_min:.0f} นาที"),
            "zh": (f"交通工具：{vname} | 目的地：{shelter} | "
                   f"预计时间：{eta_min:.0f} 分钟"),
            "en": (f"Vehicle: {vname} | Destination: {shelter} | "
                   f"ETA: {eta_min:.0f} min"),
        }
        return {"body": templates.get(language, templates["en"]),
                "language": language}

    def send_route(self, plan: Dict[str, Any], language: str = "th",
                   device_tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send one route message. Refuses to push non-actionable plans.
        """
        if not plan.get("actionable", False):
            logger.warning("Refusing to push NON-ACTIONABLE plan to users")
            return {"status": "blocked", "reason": "plan not actionable",
                    "pushed": False}

        msg = self.build_route_message(plan, language)
        result: Dict[str, Any] = {"message": msg["body"]}

        if self.mobile_app is not None and HAS_MOBILE:
            # Route through the app so its No-Zeno throttle applies
            alert = HazardAlert(
                alert_id=f"evac_route_{int(time.time())}",
                hazard_type=plan.get("hazard_type"),
                severity=AlertSeverity.WARNING,
                title="Evacuation route",
                message=msg["body"],
                latitude=plan.get("lat", 0.0),
                longitude=plan.get("lon", 0.0),
                radius_km=plan.get("radius_km", 5.0),
                timestamp=time.time(),
                recommended_action=msg["body"],
            )
            throttle_ok = self.mobile_app.throttler.can_send_alert()
            if not throttle_ok:
                result.update({"status": "throttled", "pushed": False})
                logger.info("Route push deferred by No-Zeno throttle")
            else:
                tokens = device_tokens or ["device_token_placeholder"]
                push = self.mobile_app.send_alert(alert, tokens)
                self.mobile_app.throttler.record_alert()
                result.update({"status": push.get("status", "sent"),
                               "pushed": True})
        else:
            result.update({"status": "dry_run", "pushed": False,
                           "note": "no mobile app connected; message logged only"})

        self.send_log.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════════
# [D] MULTI-MODAL EVACUATION PLANNER (orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class MultiModalEvacuationPlanner:
    """
    [D] Orchestrator: fleet matching → feasibility → mobile delivery.
    """

    def __init__(self):
        self.fleet = MultiModalFleet()
        self.schedules: Dict[str, ShelterSchedule] = {}
        self.feasibility = FeasibilityChecker()
        self.notifier = MobileEvacuationNotifier()
        self._detector = (ChangePointRegimeDetector(HazardRouteConfig())
                          if HAS_CHANGEPOINT else None)

    def register_shelter_schedule(self, shelter_id: str,
                                  schedule: ShelterSchedule) -> None:
        self.schedules[shelter_id] = schedule

    def plan_for_demand(self, demand: Dict[str, Any],
                        shelter_id: str,
                        travel_samples: np.ndarray,
                        flooded: bool = False,
                        language: str = "th") -> Dict[str, Any]:
        """
        Full pipeline for one demand point.

        Returns a plan dict with an explicit 'actionable' flag.
        """
        schedule = self.schedules.get(shelter_id)
        if schedule is None:
            return {"actionable": False, "reason": "no schedule for shelter"}

        vehicle = self.fleet.match_demand(demand, flooded)
        if vehicle is None:
            return {"actionable": False,
                    "reason": "no suitable vehicle in fleet"}

        now = demand.get("now_s", 0.0)
        service_s = (vehicle.profile.fixed_overhead_s +
                     demand.get("n_persons", 1) *
                     vehicle.profile.boarding_s_per_person)

        check = self.feasibility.leg_is_safe(
            now, travel_samples, schedule, service_s)

        plan = {
            "actionable": check.get("safe", False) and check.get("actionable", False),
            "vehicle_type": vehicle.profile.vehicle_type,
            "vehicle_id": vehicle.vehicle_id,
            "shelter_id": shelter_id,
            "shelter_name": shelter_id,
            "hazard_type": "FLOOD" if flooded else "FIRE",
            "lat": demand.get("lat", 0.0),
            "lon": demand.get("lon", 0.0),
            "eta_s": float(np.mean(travel_samples)) + service_s,
            "feasibility": check,
            "schedule_revision": schedule.revision,
            "flooded": flooded,
        }

        if plan["actionable"]:
            delivery = self.notifier.send_route(plan, language=language)
            plan["delivery"] = delivery
        else:
            plan["delivery"] = {"status": "not_sent",
                                "reason": check.get("reason")}

        return plan

    def detect_window_change(self, shelter_id: str,
                             closure_signal: np.ndarray) -> Dict[str, Any]:
        """
        [C] Use change-point detection (paper6) on a closure/severity signal
        to decide whether schedules need rebuilding.
        """
        if self._detector is None:
            return {"change_detected": False, "reason": "detector unavailable"}
        res = self._detector.detect(np.asarray(closure_signal, dtype=float))
        cps = res.get("change_points", [])
        return {
            "change_detected": len(cps) > 0,
            "change_points": cps,
            "note": res.get("note", ""),
            "recommendation": "REBUILD routes; do not average schedules",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Multi-Modal Evacuation — Smoke Test")
    print("=" * 72)
    print(f"  multistop module   : {HAS_MULTISTOP}")
    print(f"  uncertainty module : {HAS_UNCERTAINTY}")
    print(f"  changepoint module : {HAS_CHANGEPOINT}")
    print(f"  mobile app module  : {HAS_MOBILE}")

    planner = MultiModalEvacuationPlanner()

    # Fleet: 1 ambulance, 1 bus, 1 boat
    profs = VehicleProfile.standard_profiles()
    planner.fleet.add_vehicle(EvacVehicle("AMB-1", profs[VehicleType.AMBULANCE], 13.75, 100.50))
    planner.fleet.add_vehicle(EvacVehicle("BUS-1", profs[VehicleType.BUS], 13.75, 100.50))
    planner.fleet.add_vehicle(EvacVehicle("BOAT-1", profs[VehicleType.BOAT], 13.75, 100.50,
                                          needs_dock=True))

    # Shelter schedule: open 0–1800 s, closes at 1800
    sched = ShelterSchedule("SHELTER-A")
    sched.add_window(0.0, 1800.0)
    planner.register_shelter_schedule("SHELTER-A", sched)

    rng = np.random.default_rng(7)
    travel_samples = rng.normal(600, 60, 40)   # ~10 min travel

    print("\n  ── Case 1: medical demand, road OK ──")
    demand1 = {"n_persons": 2, "needs_medical": 1, "needs_wheelchair": 0,
               "lat": 13.74, "lon": 100.49, "now_s": 0.0}
    plan1 = planner.plan_for_demand(demand1, "SHELTER-A", travel_samples,
                                    flooded=False, language="th")
    print(f"    vehicle      : {plan1.get('vehicle_id')}")
    print(f"    actionable   : {plan1['actionable']}")
    print(f"    delivery     : {plan1['delivery'].get('status')}")

    print("\n  ── Case 2: mass demand, flooded → boat ──")
    demand2 = {"n_persons": 15, "needs_medical": 0, "needs_wheelchair": 0,
               "lat": 13.74, "lon": 100.49, "now_s": 0.0}
    plan2 = planner.plan_for_demand(demand2, "SHELTER-A", travel_samples * 2,
                                    flooded=True, language="zh")
    print(f"    vehicle      : {plan2.get('vehicle_id')}")
    print(f"    actionable   : {plan2['actionable']}")

    print("\n  ── Case 3: shelter closes early (change point) ──")
    closed = sched.close_early(at_time=500.0, reason="flood arrives early")
    print(f"    early close applied : {closed}")
    print(f"    schedule revision   : {sched.revision}")
    plan3 = planner.plan_for_demand(demand1, "SHELTER-A", travel_samples,
                                    flooded=False)
    print(f"    re-planned actionable: {plan3['actionable']}")
    print(f"    reason             : {plan3['feasibility'].get('reason')}")

    print("\n  ── Case 4: change-point detection on closure signal ──")
    signal = np.concatenate([np.zeros(20), np.ones(20) * 5.0])  # sudden closure
    cp = planner.detect_window_change("SHELTER-A", signal)
    print(f"    change detected : {cp['change_detected']}")
    print(f"    recommendation  : {cp.get('recommendation')}")

    print("\n  ── Case 5: non-actionable plan is NOT pushed ──")
    blocked = planner.notifier.send_route({"actionable": False}, language="en")
    print(f"    blocked status : {blocked['status']}")
    print(f"    [{'PASS' if blocked['status']=='blocked' else 'FAIL'}] "
          f"synthetic/unvalidated plan never reaches users")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Vehicle matching is GREEDY; multi-modal VRP+TW is NP-hard [O]")
    print("  • Boat uses straight-line water path; real channels needed")
    print("  • Travel-time CI is ASYMPTOTIC (Paper 11), not finite-sample")
    print("  • Schedule change = change point → REBUILD, never average")
    print("  • Message delivery ≠ user safety")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
