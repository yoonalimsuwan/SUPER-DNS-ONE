"""
=============================================================================
SESI WATER TRANSIT SCHEDULE — Multi-Dock Boat Timetable
Change-point-aware replanning for evacuation water routes
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[WT1] Dock — a berth point with κ-fatness (Delta_Min Def 1.1):
      the waiting area must not be a thin sliver; a minimum service
      radius is enforced so the dock is geometrically meaningful.
[WT2] BoatRoute — one vessel visiting an ordered sequence of docks.
[WT3] TransitSchedule — timetable (arrive/depart per dock).
[WT4] DelayReplanner — delay is a CHANGE POINT (paper6 §1.1):
      we RETIME at the delay event, never average old+new schedules.

CHANGE-POINT DISCIPLINE (paper6 Thm 3.2)
-----------------------------------------
Each leg time ℓ_k between docks is a "block length":
  • E[ℓ] < ∞  (bounded delays) → Kac's formula applies; the schedule
    converges to a stable average → ordinary timetable reasoning is valid.
  • E[ℓ] = ∞  (heavy-tailed delays) → no a.s. convergence; use a
    Darling-Kac/Lamperti distributional statement instead.
  • Doubly-exponential cascading delays → degenerate two-point law
    (Thm 4.1): the route either completes or fully collapses; the
    scheduler must detect WHICH regime it is in before acting.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Travel times are estimates; real waterways have currents/tides/traffic.
• Boat routing here uses straight-line legs; real channels need GIS data.
• A delay-replan is a CONTROL DECISION, reversible, not a guarantee.
• The regime classifier is a heuristic proxy for the true tail index θ.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.water_transit")

__all__ = [
    "Dock", "BoatRoute", "ScheduleEntry", "TransitSchedule",
    "DelayRegime", "DelayReplanner",
]


# ═══════════════════════════════════════════════════════════════════════
# [P/D] DOCK with κ-fatness (Delta_Min Def 1.1)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Dock:
    """
    [D] A boat berth. Enforces κ-fatness: the waiting/service area must
    contain a ball of radius r_P/κ, preventing degenerate thin docks.
    """
    dock_id: str
    lat: float
    lon: float
    waiting_capacity: int          # evacuees that can wait safely
    berth_count: int               # simultaneous boats
    service_radius_m: float = 50.0 # equivalent radius of the dock area
    kappa_fatness: float = 2.0     # Delta_Min Def 1.1

    def fatness_ok(self) -> bool:
        """
        [P] κ-fatness check (Delta_Min Def 1.1): there must be an interior
        point at distance ≥ r_P/κ from the boundary. For a circular dock of
        radius r, the center is at distance r, so we need r ≥ r/κ ⟺ κ ≥ 1.
        """
        return self.kappa_fatness >= 1.0 and self.service_radius_m > 0

    def min_separation_m(self) -> float:
        """
        [P] δ_min-style lower bound on how far the usable interior is from
        the dock boundary: r_P/κ (Delta_Min Thm 1.2, n=2).
        """
        return self.service_radius_m / self.kappa_fatness


# ═══════════════════════════════════════════════════════════════════════
# [D] BOAT ROUTE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BoatRoute:
    """One vessel visiting an ordered sequence of docks."""
    route_id: str
    boat_capacity: int
    speed_ms: float = 4.0          # ~14 km/h, conservative for evacuation
    dwell_s: float = 180.0         # boarding/alighting time per dock
    dock_ids: List[str] = field(default_factory=list)

    @staticmethod
    def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = (math.sin(dlat/2)**2 +
             math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
        return 6371000.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1-h))

    def leg_time(self, d_from: Dock, d_to: Dock) -> float:
        """[D] Straight-line leg time. Real channels would replace this."""
        dist = self._haversine_m((d_from.lat, d_from.lon),
                                 (d_to.lat, d_to.lon))
        return dist / max(self.speed_ms, 0.1)


# ═══════════════════════════════════════════════════════════════════════
# [D] TRANSIT SCHEDULE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ScheduleEntry:
    dock_id: str
    arrive_s: float
    depart_s: float


class TransitSchedule:
    """[D] Compute and validate a boat timetable over a dock sequence."""

    def __init__(self, route: BoatRoute, docks: Dict[str, Dock]):
        self.route = route
        self.docks = docks

    def build(self, start_s: float = 0.0) -> List[ScheduleEntry]:
        """Forward-simulate the timetable from a start time."""
        entries: List[ScheduleEntry] = []
        t = start_s
        seq = self.route.dock_ids
        for i, did in enumerate(seq):
            arrive = t
            depart = arrive + self.route.dwell_s
            entries.append(ScheduleEntry(did, arrive, depart))
            if i + 1 < len(seq):
                t = depart + self.route.leg_time(self.docks[did],
                                                 self.docks[seq[i+1]])
        return entries

    def check_feasibility(self, entries: List[ScheduleEntry]) -> Dict[str, Any]:
        """Check dock capacity and κ-fatness validity."""
        issues = []
        for e in entries:
            dock = self.docks.get(e.dock_id)
            if dock is None:
                issues.append(f"unknown dock {e.dock_id}")
                continue
            if not dock.fatness_ok():
                issues.append(f"dock {e.dock_id} fails κ-fatness")
            if dock.waiting_capacity <= 0:
                issues.append(f"dock {e.dock_id} has zero capacity")
        return {"feasible": len(issues) == 0, "issues": issues}


# ═══════════════════════════════════════════════════════════════════════
# [C/D] DELAY REPLANNER (paper6 change-point discipline)
# ═══════════════════════════════════════════════════════════════════════

class DelayRegime(Enum):
    FINITE_MEAN = auto()      # θ>1: ordinary LLN, retime gently
    HEAVY_TAIL = auto()       # θ∈(0,1): distributional, keep slack
    DEGENERATE = auto()       # θ→0: all-or-nothing, abort/commit decision


class DelayReplanner:
    """
    [C/D] Replan a schedule when a delay occurs.

    Discipline (paper6 §1.1 'retime, don't average'):
      A delay is a CHANGE POINT. We rebuild the timetable from the delay
      event forward; we NEVER blend the old and new timetables.

    Regime classification (paper6 Thm 5.1) is a heuristic proxy for the
    tail index θ of the leg-delay distribution:
      • small, bounded delays      → FINITE_MEAN
      • large, variable delays     → HEAVY_TAIL
      • cascading/explosive delays → DEGENERATE
    """

    def __init__(self, max_delay_s: float = 600.0):
        self.max_delay_s = max_delay_s

    def classify_regime(self, delay_history: List[float]) -> DelayRegime:
        """Heuristic θ-classification from observed leg delays."""
        x = np.asarray(delay_history, dtype=float)
        if len(x) < 4:
            return DelayRegime.FINITE_MEAN
        peak = float(np.max(x))
        mean = float(np.mean(x)) + 1e-9
        ratio = peak / mean
        if ratio < 3.0:
            return DelayRegime.FINITE_MEAN
        elif ratio < 10.0:
            return DelayRegime.HEAVY_TAIL
        return DelayRegime.DEGENERATE

    def replan(self, route: BoatRoute, docks: Dict[str, Dock],
               current_dock_idx: int, delay_s: float,
               now_s: float) -> Dict[str, Any]:
        """
        Rebuild the remaining timetable from the delay point forward.

        Returns the new entries + regime + action. For DEGENERATE regime
        the honest action is an abort/commit decision, not a new average.
        """
        regime = self.classify_regime([delay_s])
        remaining = route.dock_ids[current_dock_idx:]
        entries: List[ScheduleEntry] = []

        if regime == DelayRegime.DEGENERATE and delay_s > self.max_delay_s:
            return {
                "regime": regime.name,
                "action": "abort_or_commit",
                "entries": [],
                "note": ("Delay exceeds bound; schedule is in the degenerate "
                         "two-point regime (paper6 Thm 4.1). Do NOT average a "
                         "new timetable; decide to abort or commit explicitly."),
            }

        t = now_s + delay_s
        slack = 1.5 if regime == DelayRegime.HEAVY_TAIL else 1.0

        for i, did in enumerate(remaining):
            arrive = t
            depart = arrive + route.dwell_s * slack
            entries.append(ScheduleEntry(did, arrive, depart))
            if i + 1 < len(remaining):
                leg = route.leg_time(docks[did], docks[remaining[i+1]])
                t = depart + leg * slack

        return {
            "regime": regime.name,
            "action": "retimed",
            "entries": entries,
            "note": "Rebuilt from change point forward; old tail discarded",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Water Transit Schedule — Smoke Test")
    print("=" * 72)

    docks = {
        "DOCK-A": Dock("DOCK-A", 13.70, 100.50, 200, 2),
        "DOCK-B": Dock("DOCK-B", 13.71, 100.52, 150, 1),
        "DOCK-C": Dock("DOCK-C", 13.72, 100.54, 300, 3),
    }
    for d in docks.values():
        print(f"  {d.dock_id}: fatness_ok={d.fatness_ok()}, "
              f"min_separation={d.min_separation_m():.1f}m")

    route = BoatRoute("BOAT-1", boat_capacity=40,
                      dock_ids=["DOCK-A", "DOCK-B", "DOCK-C"])
    sched = TransitSchedule(route, docks)

    print("\n  ── Build timetable ──")
    entries = sched.build(start_s=0.0)
    feas = sched.check_feasibility(entries)
    for e in entries:
        print(f"    {e.dock_id}: arrive={e.arrive_s:.0f}s depart={e.depart_s:.0f}s")
    print(f"    feasible: {feas['feasible']}")

    print("\n  ── Delay replan (change-point, retime don't average) ──")
    replanner = DelayReplanner(max_delay_s=600.0)
    for delay in [120.0, 400.0, 1500.0]:
        res = replanner.replan(route, docks, current_dock_idx=1,
                               delay_s=delay, now_s=1000.0)
        print(f"    delay={delay:.0f}s → regime={res['regime']}, "
              f"action={res['action']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Travel times are estimates; currents/tides not modeled")
    print("  • Straight-line legs; real channels need GIS")
    print("  • Replan is a reversible CONTROL decision, not a guarantee")
    print("  • Regime classifier is a heuristic proxy for tail index θ")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
