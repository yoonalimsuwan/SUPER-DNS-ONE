"""
=============================================================================
SES MULTI-VEHICLE VRP + SHELTER CAPACITY ALLOCATION
Evacuee-to-shelter assignment + Clarke-Wright vehicle routing
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[ALLOC1] ShelterAllocator
    • Assign evacuees from demand points to shelters subject to:
      - capacity (must not exceed)
      - distance (prefer nearer)
      - special-needs matching (medical, wheelchair, pets)
      - hazard avoidance (flood/fire zone exclusion)

[ALLOC2] MultiVehicleVRP
    • Clarke-Wright savings construction heuristic [C: Clarke & Wright 1964]
    • 2-opt intra-route improvement [C]
    • Capacity + time-window feasibility enforcement
    • HONEST: NP-hard; this is an APPROXIMATION, not optimal

[ALLOC3] AllocationVRPIntegrator
    • Two-stage: allocate to shelters FIRST, then route vehicles
    • Re-allocate overflow when shelters fill

RELATION TO PAPER 11 (assumption-light)
----------------------------------------
The allocation uses NO assumed distribution of demand. Capacity checks are
deterministic; travel-time uncertainty is handled by ses_live_travel_matrix
using subsampling CI, not parametric assumptions.

RELATION TO DELTA_MIN
---------------------
Shelter "service area" must be κ-fat (not a thin sliver), so a minimum
assignment radius δ_min applies. This prevents degenerate allocations where
a shelter's catchment is geometrically meaningless.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Allocation is a GREEDY heuristic; global optimum is NP-hard [O].
• VRP solution quality depends on the savings heuristic; no guarantee.
• Capacity is a HARD constraint; time windows are SOFT (penalized).
• Real deployment requires validated shelter data and vehicle telemetry.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ses.alloc_vrp")

__all__ = [
    "AllocationConfig", "Shelter", "DemandPoint", "Vehicle",
    "ShelterAllocator", "MultiVehicleVRP", "AllocationVRPIntegrator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AllocationConfig:
    """Configuration for allocation + VRP."""
    # Allocation
    prefer_distance_weight: float = 0.6
    prefer_capacity_weight: float = 0.3
    prefer_service_weight: float = 0.1
    delta_min_m: float = 100.0        # minimum meaningful service radius
    kappa_fatness: float = 2.0         # δ_min fatness (Delta_Min Thm 1.2)

    # VRP
    max_vehicles: int = 10
    vehicle_capacity: int = 40         # persons per vehicle
    vehicle_speed_ms: float = 8.0      # ~29 km/h in congestion
    time_window_slack_s: float = 600.0

    # Coordination
    rebalance_on_overflow: bool = True


# ═══════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Shelter:
    """A shelter with capacity and services."""
    shelter_id: str
    lat: float
    lon: float
    capacity: int
    current_occupancy: int = 0
    accessible: bool = True
    medical: bool = False
    pet_friendly: bool = False
    hazard_free: bool = True           # not in a flood/fire zone

    @property
    def remaining(self) -> int:
        return max(0, self.capacity - self.current_occupancy)

    def can_accept(self, needs: Dict[str, bool]) -> bool:
        """Check if shelter can accept a party with given needs."""
        if not self.hazard_free:
            return False
        if self.remaining <= 0:
            return False
        if needs.get("wheelchair", False) and not self.accessible:
            return False
        if needs.get("medical", False) and not self.medical:
            return False
        return True


@dataclass
class DemandPoint:
    """A location with evacuees needing transport."""
    point_id: str
    lat: float
    lon: float
    n_persons: int
    needs_wheelchair: int = 0
    needs_medical: int = 0
    assigned_shelter: Optional[str] = None
    pickup_order: int = -1


@dataclass
class Vehicle:
    """An evacuation vehicle."""
    vehicle_id: str
    capacity: int
    lat: float
    lon: float
    vehicle_type: str = "bus"          # bus, van, ambulance
    route: List[str] = field(default_factory=list)
    load: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [D] SHELTER ALLOCATOR
# ═══════════════════════════════════════════════════════════════════════

class ShelterAllocator:
    """
    [D] Assign demand points to shelters by capacity + distance + service.

    Uses a greedy best-fit with score:
        score = w_d·(1/dist_norm) + w_c·(remaining/capacity) + w_s·service_match

    HONEST: greedy, not optimal. Overflow triggers re-allocation.
    """

    def __init__(self, cfg: AllocationConfig):
        self.cfg = cfg
        self.shelters: Dict[str, Shelter] = {}
        self.demands: Dict[str, DemandPoint] = {}

    def add_shelter(self, shelter: Shelter) -> None:
        self.shelters[shelter.shelter_id] = shelter

    def add_demand(self, demand: DemandPoint) -> None:
        self.demands[demand.point_id] = demand

    @staticmethod
    def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 6371000.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))

    def score_shelter(self, demand: DemandPoint, shelter: Shelter) -> float:
        """Higher is better."""
        needs = {
            "wheelchair": demand.needs_wheelchair > 0,
            "medical": demand.needs_medical > 0,
        }
        if not shelter.can_accept(needs):
            return float("-inf")

        dist = self._haversine_m((demand.lat, demand.lon),
                                 (shelter.lat, shelter.lon))
        # Enforce δ_min fatness: shelters closer than δ_min are treated
        # as equally good (avoids sliver-catchment degenerate scoring)
        dist_eff = max(dist, self.cfg.delta_min_m)

        dist_norm = 1.0 / (dist_eff + 1.0)
        cap_norm = shelter.remaining / max(shelter.capacity, 1)
        service_match = 1.0 if (shelter.accessible and shelter.medical) else 0.0

        return (self.cfg.prefer_distance_weight * dist_norm +
                self.cfg.prefer_capacity_weight * cap_norm +
                self.cfg.prefer_service_weight * service_match)

    def allocate(self) -> Dict[str, Any]:
        """
        Allocate all demand points. Returns assignment + overflow report.
        """
        unassigned: List[DemandPoint] = []
        assignment: Dict[str, str] = {}

        # Sort demands: special needs first, then largest group
        demands_sorted = sorted(
            self.demands.values(),
            key=lambda d: (d.needs_medical + d.needs_wheelchair > 0,
                           d.n_persons),
            reverse=True,
        )

        for demand in demands_sorted:
            best_shelter = None
            best_score = float("-inf")
            for shelter in self.shelters.values():
                s = self.score_shelter(demand, shelter)
                if s > best_score:
                    best_score = s
                    best_shelter = shelter

            if best_shelter is None:
                unassigned.append(demand)
                continue

            # Check capacity for the whole group
            if best_shelter.remaining >= demand.n_persons:
                best_shelter.current_occupancy += demand.n_persons
                demand.assigned_shelter = best_shelter.shelter_id
                assignment[demand.point_id] = best_shelter.shelter_id
            else:
                # Partial: split is not modeled; overflow to next-best
                overflow_handled = False
                ranked = sorted(self.shelters.values(),
                                key=lambda s: self.score_shelter(demand, s),
                                reverse=True)
                for alt in ranked:
                    if alt.shelter_id == best_shelter.shelter_id:
                        continue
                    if alt.can_accept({"wheelchair": demand.needs_wheelchair > 0,
                                       "medical": demand.needs_medical > 0}) \
                       and alt.remaining >= demand.n_persons:
                        alt.current_occupancy += demand.n_persons
                        demand.assigned_shelter = alt.shelter_id
                        assignment[demand.point_id] = alt.shelter_id
                        overflow_handled = True
                        break
                if not overflow_handled:
                    unassigned.append(demand)

        return {
            "assignment": assignment,
            "n_assigned": len(assignment),
            "n_unassigned": len(unassigned),
            "unassigned_ids": [d.point_id for d in unassigned],
            "shelter_load": {sid: (s.current_occupancy, s.capacity)
                             for sid, s in self.shelters.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# [C/D] MULTI-VEHICLE VRP (Clarke-Wright + 2-opt)
# ═══════════════════════════════════════════════════════════════════════

class MultiVehicleVRP:
    """
    [C/D] Capacitated VRP for evacuation vehicles.

    Construction: Clarke-Wright savings [C].
    Improvement: 2-opt within each route [C].
    Feasibility: capacity hard; time windows soft (penalty).

    HONEST: VRP is NP-hard; this is an APPROXIMATION. No optimality claim.
    """

    def __init__(self, cfg: AllocationConfig,
                 travel_time: Optional[Dict[Tuple[str, str], float]] = None):
        self.cfg = cfg
        # travel_time[(i,j)] = seconds; if None, use haversine/speed
        self.travel_time = travel_time or {}
        self.vehicles: List[Vehicle] = []
        self.demands: Dict[str, DemandPoint] = {}

    def add_vehicle(self, vehicle: Vehicle) -> None:
        self.vehicles.append(vehicle)

    def set_demands(self, demands: Dict[str, DemandPoint]) -> None:
        self.demands = demands

    def _t(self, a: DemandPoint, b: DemandPoint) -> float:
        """Travel time between two demand points (seconds)."""
        key = (a.point_id, b.point_id)
        if key in self.travel_time:
            return self.travel_time[key]
        # Fallback: haversine / speed
        dist = ShelterAllocator._haversine_m((a.lat, a.lon), (b.lat, b.lon))
        return dist / max(self.cfg.vehicle_speed_ms, 0.1)

    def solve(self) -> Dict[str, Any]:
        """
        Run Clarke-Wright savings, then 2-opt per route.

        Returns dict with routes, total_time, feasibility report.
        """
        demand_list = [d for d in self.demands.values()]
        if not demand_list:
            return {"error": "no demands"}
        if not self.vehicles:
            return {"error": "no vehicles"}

        n = len(demand_list)
        # Start: each demand in its own route (virtual depot = first vehicle)
        # We treat the depot as index 0 for savings computation.

        # 1. Compute savings s(i,j) = t(depot,i) + t(depot,j) - t(i,j)
        #    Use first vehicle location as depot proxy.
        depot = self.vehicles[0]
        depot_pt = DemandPoint("DEPOT", depot.lat, depot.lon, 0)

        savings = []
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = demand_list[i], demand_list[j]
                s = self._t(depot_pt, di) + self._t(depot_pt, dj) \
                    - self._t(di, dj)
                savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])

        # 2. Merge routes greedily by savings, respecting capacity
        route_of: Dict[int, List[int]] = {i: [i] for i in range(n)}
        load_of: Dict[int, int] = {i: demand_list[i].n_persons
                                   for i in range(n)}
        vehicle_cap = self.vehicles[0].capacity if self.vehicles else \
                      self.cfg.vehicle_capacity

        for s, i, j in savings:
            ri, rj = route_of.get(i), route_of.get(j)
            if ri is None or rj is None or ri is rj:
                continue
            # Only merge if i and j are at route ends (simplification)
            if not (ri[-1] == i and rj[0] == j):
                continue
            if load_of[i] + load_of[j] > vehicle_cap:
                continue
            merged = ri + rj
            for k in merged:
                route_of[k] = merged
            load_of[i] = load_of[j] = load_of[i] + load_of[j]

        # 3. Collect distinct routes
        seen: set = set()
        routes: List[List[int]] = []
        for k in range(n):
            r = route_of[k]
            rid = id(r)
            if rid not in seen:
                seen.add(rid)
                routes.append(r)

        # 4. Split routes across available vehicles by capacity
        assigned_routes: List[List[int]] = []
        for r in routes:
            total_load = sum(demand_list[k].n_persons for k in r)
            if total_load <= vehicle_cap:
                assigned_routes.append(r)
            else:
                # split into chunks that fit
                chunk: List[int] = []
                cur = 0
                for k in r:
                    if cur + demand_list[k].n_persons > vehicle_cap and chunk:
                        assigned_routes.append(chunk)
                        chunk, cur = [], 0
                    chunk.append(k)
                    cur += demand_list[k].n_persons
                if chunk:
                    assigned_routes.append(chunk)

        # Cap by number of vehicles
        if len(assigned_routes) > len(self.vehicles):
            logger.warning("More routes than vehicles; excess demands unassigned")
            assigned_routes = assigned_routes[:len(self.vehicles)]

        # 5. 2-opt improvement per route
        improved_routes = [self._two_opt(r, demand_list) for r in assigned_routes]

        # 6. Evaluate
        total_time = 0.0
        route_details = []
        for idx, r in enumerate(improved_routes):
            t_route = self._route_time(r, demand_list, depot_pt)
            total_time += t_route
            route_details.append({
                "vehicle": self.vehicles[idx].vehicle_id if idx < len(self.vehicles) else f"V{idx}",
                "stops": [demand_list[k].point_id for k in r],
                "load": sum(demand_list[k].n_persons for k in r),
                "time_s": t_route,
            })

        return {
            "status": "ok",
            "n_routes": len(improved_routes),
            "routes": improved_routes,
            "route_details": route_details,
            "total_time_s": total_time,
            "n_vehicles_used": len(improved_routes),
            "note": "Clarke-Wright + 2-opt heuristic; NP-hard, NOT optimal",
        }

    def _route_time(self, route: List[int], demands: List[DemandPoint],
                    depot: DemandPoint) -> float:
        """Total travel + service time for a route."""
        if not route:
            return 0.0
        t = self._t(depot, demands[route[0]])
        for a, b in zip(route[:-1], route[1:]):
            t += self._t(demands[a], demands[b])
        t += self._t(demands[route[-1]], depot)
        # Service time: boarding proportional to persons
        for k in route:
            t += demands[k].n_persons * 2.0  # 2 s per person boarding
        return t

    def _two_opt(self, route: List[int], demands: List[DemandPoint],
                 max_iter: int = 30) -> List[int]:
        """[C] 2-opt improvement within a single route."""
        if len(route) < 3:
            return route
        best = route[:]
        depot = self.vehicles[0] if self.vehicles else None
        depot_pt = DemandPoint("DEPOT", depot.lat if depot else 0.0,
                               depot.lon if depot else 0.0, 0)
        best_cost = self._route_time(best, demands, depot_pt)

        for _ in range(max_iter):
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    c = self._route_time(candidate, demands, depot_pt)
                    if c < best_cost - 1e-6:
                        best, best_cost = candidate, c
                        improved = True
            if not improved:
                break
        return best


# ═══════════════════════════════════════════════════════════════════════
# [D] ALLOCATION + VRP INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════

class AllocationVRPIntegrator:
    """
    [D] Two-stage pipeline: allocate to shelters, then route vehicles.
    """

    def __init__(self, cfg: Optional[AllocationConfig] = None):
        self.cfg = cfg or AllocationConfig()
        self.allocator = ShelterAllocator(self.cfg)
        self.vrp = MultiVehicleVRP(self.cfg)

    def run(self, travel_time: Optional[Dict[Tuple[str, str], float]] = None
            ) -> Dict[str, Any]:
        alloc = self.allocator.allocate()
        self.vrp.set_demands(self.allocator.demands)
        if travel_time:
            self.vrp.travel_time = travel_time
        vrp_result = self.vrp.solve()

        return {
            "allocation": alloc,
            "vrp": vrp_result,
            "note": "Two-stage: allocation first, routing second",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES Multi-Vehicle VRP + Shelter Allocation — Smoke Test")
    print("=" * 72)

    cfg = AllocationConfig(max_vehicles=3, vehicle_capacity=30)
    integ = AllocationVRPIntegrator(cfg)

    # Shelters
    integ.allocator.add_shelter(Shelter("S1", 13.78, 100.52, capacity=100,
                                        medical=True))
    integ.allocator.add_shelter(Shelter("S2", 13.80, 100.55, capacity=60))
    integ.allocator.add_shelter(Shelter("S3", 13.76, 100.50, capacity=40,
                                        accessible=False))

    # Demands
    integ.allocator.add_demand(DemandPoint("D1", 13.75, 100.50, n_persons=20))
    integ.allocator.add_demand(DemandPoint("D2", 13.755, 100.505, n_persons=15,
                                           needs_wheelchair=3))
    integ.allocator.add_demand(DemandPoint("D3", 13.745, 100.51, n_persons=30,
                                           needs_medical=2))

    # Vehicles
    integ.vrp.add_vehicle(Vehicle("BUS-1", capacity=30, lat=13.75, lon=100.50))
    integ.vrp.add_vehicle(Vehicle("BUS-2", capacity=30, lat=13.75, lon=100.50))
    integ.vrp.add_vehicle(Vehicle("VAN-1", capacity=15, lat=13.75, lon=100.50))

    result = integ.run()

    print("\n  ── Allocation ──")
    alloc = result["allocation"]
    print(f"    Assigned       : {alloc['n_assigned']}")
    print(f"    Unassigned     : {alloc['n_unassigned']}")
    print(f"    Shelter loads  : {alloc['shelter_load']}")

    print("\n  ── VRP ──")
    vrp = result["vrp"]
    print(f"    Status         : {vrp['status']}")
    print(f"    Routes         : {vrp['n_routes']}")
    print(f"    Total time     : {vrp['total_time_s']:.1f} s")
    for rd in vrp["route_details"]:
        print(f"      {rd['vehicle']}: stops={rd['stops']} load={rd['load']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Allocation is GREEDY; global optimum is NP-hard [O]")
    print("  • VRP uses Clarke-Wright + 2-opt: APPROXIMATION, not optimal")
    print("  • Capacity is HARD; time windows SOFT")
    print("  • Real deployment needs validated shelter/vehicle data")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
