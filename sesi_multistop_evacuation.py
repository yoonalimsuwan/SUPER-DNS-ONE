"""
=============================================================================
SESI MULTI-STOP EVACUATION ROUTE PLANNER
Hazard-aware, time-windowed, capacity/accessible-aware multi-stop routing
=============================================================================
Developer    : PAI AND Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

PROBLEM
-------
Given:
  • origin (affected population / user location)
  • a set of intermediate stops (pickup, medical, water, assembly)
  • a final shelter (must be last)
  • time windows [open_j, close_j], service time, accessibility
  • a time-varying hazard field (flood depth / fire / ash / wind)

Find a visit order minimizing:
    objective 1 = total elapsed time (travel + wait + service)
    objective 2 = cumulative hazard exposure
subject to:
    • all REQUIRED intermediate stops visited exactly once
    • every time window respected
    • final stop is the shelter
    • accessibility / capacity constraints satisfied

This is a time-windowed, bi-objective TSP/VRP variant => NP-hard in general.

ALGORITHMS
----------
[A1] Exact label-setting DP (Held-Karp style) for small n
     State = (subset of required stops, last stop)
     Label = (departure_time, cumulative_risk, path)
     Pareto-prune labels; earlier departure + lower risk dominates.
     VALID because travel times are static within a plan and waiting is allowed.

[A2] Heuristic for larger n: nearest-neighbor + 2-opt + archive of
     non-dominated routes. No optimality guarantee.

MATHEMATICAL DISCIPLINE (from the supplied papers)
--------------------------------------------------
• Unified_Ordinal §11: only the ITERATION COUNT is bounded under fixed
  dimension/per-step cost. We do NOT claim O(1) physical complexity, and we
  do NOT claim the re-planning map is a contraction.
• Paper 6 Def 3.1: retime at change points -> induced (length, mark) pairs.
  Open Problem 6.1: estimated change points carry error/delay.
• Paper 11 Thm 4.4: subsampling CI for a quantile needs stationarity and
  summable alpha-mixing (qualitative); coverage is asymptotic.
• sesi_unified_amr: Dorfler bulk marking captures >= theta of squared error.
• No-Zeno (CFD): bound on EVENT COUNT, not on solver accuracy.

LABELS: [P] proved under stated assumptions
        [C] cited classical result
        [D] derived engineering implementation
        [O] open / requires real data

HONEST SCOPE
------------
• Routes are SUGGESTIONS. Official local-authority orders take precedence.
• Without map API credentials, the matrix is SYNTHETIC and flagged.
• ETA uncertainty is a CI under assumptions, NOT a safety guarantee.
• Hazard field is a proxy; real hazard extent needs CFD/hydrodyn simulation.
• If the DP label cap is hit, the result is APPROXIMATE and marked so.
=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.multistop")

try:
    from sesi_global_maps import GlobalMapsConfig, GlobalMapsRouter
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False

try:
    from sesi_geo_maps import GCJ02Transformer, CoordinateSystem
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

__all__ = [
    "MultiStopConfig", "EvacStop", "RouteCandidate",
    "HazardCell", "HazardField",
    "CUSUMChangePointDetector", "ChangePointInducedView",
    "AssumptionLightETA",
    "TravelMatrixBuilder",
    "MultiStopEvacuationPlanner",
    "NoZenoReplanner",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MultiStopConfig:
    """Configuration for multi-stop evacuation planning."""
    # Exact solver
    max_exact_stops: int = 10          # required intermediate stops + endpoints
    label_cap_per_state: int = 4000    # if hit => NOT exact anymore

    # Heuristic
    n_multistarts: int = 8
    max_local_search_iters: int = 200
    seed: int = 42

    # Hazard / geometry
    hazard_buffer_m: float = 200.0     # extra standoff from hazard boundary
    kappa_fatness: float = 2.0         # Delta_Min-style fatness of zone
    delta_min_m: Optional[float] = None  # from Paper 11 estimator if available
    n_hazard_samples_per_leg: int = 12

    # Dörfler marking
    dorfler_theta: float = 0.5

    # Re-planning (No-Zeno)
    replan_min_interval_s: float = 30.0
    max_replans_per_horizon: int = 20

    # Uncertainty (Paper 11)
    ci_level: float = 0.95
    tail_p: float = 0.01
    subsample_block: Optional[int] = None   # None => automatic rule-of-thumb

    # Routing backend
    use_live_router: bool = True
    average_speed_ms: float = 8.0      # evacuation traffic (~29 km/h)


# ═══════════════════════════════════════════════════════════════════════
# STOP MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvacStop:
    """One node in the evacuation graph."""
    stop_id: str
    name: str
    lat: float
    lon: float
    kind: str = "waypoint"        # origin / shelter / medical / water / pickup
    required: bool = True
    accessible: bool = True       # wheelchair / stretcher access
    capacity_persons: int = 1000
    open_t: float = 0.0           # seconds from planning epoch
    close_t: float = float("inf")
    service_s: float = 0.0        # time spent at stop (loading, triage, etc.)

    def window_ok(self, service_start: float) -> bool:
        return self.open_t <= service_start <= self.close_t


@dataclass
class RouteCandidate:
    """An evaluated candidate route."""
    order: List[str]
    total_time_s: float
    total_risk: float
    feasible: bool
    arrivals: List[float] = field(default_factory=list)
    waits_s: List[float] = field(default_factory=list)
    exact: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def objective(self) -> Tuple[float, float]:
        return (self.total_time_s, self.total_risk)


# ═══════════════════════════════════════════════════════════════════════
# HAZARD FIELD + DÖRFLER MARKING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HazardCell:
    """A circular hazard footprint (proxy, not a CFD output)."""
    lat: float
    lon: float
    radius_m: float
    intensity: float = 1.0        # 0..1
    valid_from: float = 0.0
    valid_until: float = float("inf")
    hazard_type: str = "FLOOD"


class HazardField:
    """
    [D] Time-dependent hazard footprint collection.

    Exposure along a leg is sampled at n points; a cell contributes
    intensity x overlap if the sample is inside radius + buffer.
    """

    def __init__(self, cfg: MultiStopConfig):
        self.cfg = cfg
        self.cells: List[HazardCell] = []

    def add(self, cell: HazardCell) -> None:
        self.cells.append(cell)

    @staticmethod
    def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 6371000.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))

    def exposure(self, origin: Tuple[float, float],
                 dest: Tuple[float, float], t_depart: float) -> float:
        """
        [D] Dimensionless cumulative exposure proxy along a straight leg.

        HONEST: straight-line sampling ignores road geometry; real exposure
        must be evaluated on the actual route polyline from the map provider.
        """
        total = 0.0
        n = self.cfg.n_hazard_samples_per_leg
        for k in range(n + 1):
            w = k / n
            p = (origin[0] + w * (dest[0] - origin[0]),
                 origin[1] + w * (dest[1] - origin[1]))
            t = t_depart + w * 0.0  # placeholder: leg duration supplied elsewhere
            for cell in self.cells:
                if not (cell.valid_from <= t_depart <= cell.valid_until):
                    continue
                d = self._haversine_m(p, (cell.lat, cell.lon))
                effective_r = cell.radius_m + self.cfg.hazard_buffer_m
                if d <= effective_r:
                    # linear decay inside the (fat) zone
                    total += cell.intensity * (1.0 - d / max(effective_r, 1e-9))
        return total / max(n, 1)

    def dorfler_mark(self, indicators: np.ndarray,
                     theta: Optional[float] = None) -> List[int]:
        """
        [C] Dörfler bulk chasing: mark smallest-set-by-greedy until
            sum_marked eta^2 >= theta * sum_all eta^2.

        This guarantees the captured ERROR FRACTION, not correctness of
        the underlying hazard model.
        """
        theta = self.cfg.dorfler_theta if theta is None else theta
        eta2 = np.asarray(indicators, dtype=float) ** 2
        total = float(eta2.sum())
        if total <= 0:
            return []
        order = np.argsort(-eta2)
        acc = 0.0
        marked: List[int] = []
        for i in order:
            if acc >= theta * total:
                break
            acc += float(eta2[i])
            marked.append(int(i))
        return sorted(marked)


# ═══════════════════════════════════════════════════════════════════════
# [C] CUSUM CHANGE-POINT DETECTOR  (Page 1954)
# ═══════════════════════════════════════════════════════════════════════

class CUSUMChangePointDetector:
    """
    [C] Two-sided CUSUM for hazard intensity drift.

        g+_t = max(0, g+_{t-1} + (x_t - mu0 - k))
        g-_t = max(0, g-_{t-1} + (-x_t + mu0 - k))
        signal when max(g+, g-) > h

    HONEST (Paper 6 Open Problem 6.1): estimated change points have
    delay and false alarms. Outputs are CANDIDATES, not confirmed events.
    """

    def __init__(self, mu0: float = 0.0, drift_k: float = 0.5,
                 threshold_h: float = 5.0):
        self.mu0, self.k, self.h = mu0, drift_k, threshold_h

    def detect(self, series: np.ndarray) -> Dict[str, Any]:
        gp = gm = 0.0
        signals: List[int] = []
        for t, x in enumerate(np.asarray(series, dtype=float)):
            gp = max(0.0, gp + (x - self.mu0 - self.k))
            gm = max(0.0, gm + (-x + self.mu0 - self.k))
            if max(gp, gm) > self.h:
                signals.append(t)
                gp = gm = 0.0
        return {
            "candidate_change_points": signals,
            "n_signals": len(signals),
            "note": ("CUSUM candidates only; estimation delay/false alarms "
                     "per Paper 6 Open Problem 6.1"),
        }


# ═══════════════════════════════════════════════════════════════════════
# [C] CHANGE-POINT INDUCED VIEW  (Paper 6 Def 3.1)
# ═══════════════════════════════════════════════════════════════════════

class ChangePointInducedView:
    """
    [C] Retime, don't average.

    Given change points tau_k and piecewise-constant marks eps_k, build
    the induced sequence (ell_k, eps_k), ell_k = tau_{k+1} - tau_k.

    This is a LOSSLESS re-encoding when the process is piecewise constant.
    Stationarity of the INDUCED sequence says nothing automatic about the
    real-time Cesaro average (Paper 6 shows the latter may not exist).
    """

    @staticmethod
    def build(taus: List[int], marks: List[float]) -> Dict[str, Any]:
        if len(taus) < 2 or len(marks) != len(taus) - 1:
            return {"error": "need >= 2 change points and one mark per block"}
        blocks = [(int(taus[k + 1] - taus[k]), float(marks[k]))
                  for k in range(len(marks))]
        eps = np.array([m for _, m in blocks])

        # Simple induced-sequence diagnostics (NOT a stationarity proof)
        acf1 = (float(np.corrcoef(eps[:-1], eps[1:])[0, 1])
                if len(eps) > 2 and np.std(eps) > 0 else float("nan"))
        return {
            "blocks": blocks,
            "n_blocks": len(blocks),
            "mean_block_length": float(np.mean([l for l, _ in blocks])),
            "induced_mark_lag1_corr": acf1,
            "note": ("Induced-sequence diagnostic only; does NOT establish "
                     "real-time stationarity (Paper 6 §4-5)"),
        }


# ═══════════════════════════════════════════════════════════════════════
# [C/D] ASSUMPTION-LIGHT ETA INTERVAL  (Paper 11 Thm 4.4)
# ═══════════════════════════════════════════════════════════════════════

class AssumptionLightETA:
    """
    [C/D] Subsampling confidence interval for a travel-time quantile.

    Mirrors Paper 11 Def 4.3 / Thm 4.4:
      • estimator = empirical (1-p) quantile of pooled samples
      • recompute on all length-b subsamples
      • CI half-width from empirical quantile of sqrt(b)*(quantile_b - quantile_n)

    REQUIRED (qualitative, per Paper 11):
      stationarity + alpha-mixing at SOME summable rate.
    NO numeric mixing rate is assumed or estimated.

    HONEST: coverage is ASYMPTOTIC. The automatic block-length rule here is
    a rule-of-thumb [D], not a full Politis-White implementation.
    """

    def __init__(self, cfg: MultiStopConfig):
        self.cfg = cfg

    def _auto_block(self, n: int) -> int:
        if self.cfg.subsample_block is not None:
            return max(2, min(int(self.cfg.subsample_block), n // 2))
        # [D] rule-of-thumb b ~ n^(1/3), bounded to a usable range
        return max(2, min(int(round(n ** (1.0 / 3.0))), max(2, n // 2)))

    def interval(self, samples: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(samples, dtype=float).ravel()
        n = len(x)
        if n < 12:
            return {"error": "insufficient samples for subsampling (need >= 12)"}

        p = self.cfg.tail_p
        b = self._auto_block(n)
        q_hat = float(np.quantile(x, 1.0 - p))

        rng = np.random.default_rng(self.cfg.seed)
        starts = range(0, n - b + 1)
        deviations = []
        for s in starts:
            sub = x[s:s + b]
            q_b = float(np.quantile(sub, 1.0 - p))
            deviations.append(math.sqrt(b) * (q_b - q_hat))
        if not deviations:
            return {"error": "block length too large for this sample size"}

        devs = np.array(deviations)
        alpha = 1.0 - self.cfg.ci_level
        half = float(np.quantile(np.abs(devs), 1.0 - alpha))

        return {
            "point_estimate_s": q_hat,
            "ci_lower_s": max(0.0, q_hat - half),
            "ci_upper_s": q_hat + half,
            "confidence_level": self.cfg.ci_level,
            "block_length_b": b,
            "n_subsamples": len(devs),
            "assumptions": [
                "stationary excursion/travel-time process",
                "alpha-mixing at some summable rate (value not assumed)",
            ],
            "note": ("Asymptotic coverage; finite-sample guarantee NOT claimed. "
                     "Block-length rule here is a rule-of-thumb [D]."),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] TRAVEL MATRIX BUILDER  (live router or flagged synthetic)
# ═══════════════════════════════════════════════════════════════════════

class TravelMatrixBuilder:
    """
    [D] Build static travel-time / risk matrices for one planning epoch.

    Live mode delegates to GlobalMapsRouter (Google/Azure/Apple) with its
    No-Zeno per-provider throttle and failure log.
    Offline mode returns an explicitly SYNTHETIC straight-line matrix.
    """

    def __init__(self, cfg: MultiStopConfig,
                 router: Optional[Any] = None):
        self.cfg = cfg
        self.router = router
        self.is_synthetic = True

    def build(self, stops: List[EvacStop],
              hazard: Optional[HazardField] = None
              ) -> Dict[str, Any]:
        n = len(stops)
        T = np.full((n, n), np.inf)
        R = np.zeros((n, n))
        pts = [(s.lat, s.lon) for s in stops]

        live_ok = (self.cfg.use_live_router and self.router is not None
                   and getattr(self.router, "google", None) is not None
                   and getattr(self.router.google, "available", False))
        self.is_synthetic = not live_ok

        for i in range(n):
            T[i, i] = 0.0
            for j in range(n):
                if i == j:
                    continue
                if live_ok:
                    route = self.router.route(pts[i], pts[j])
                    if route and route.get("duration_s"):
                        T[i, j] = float(route["duration_s"])
                        self.is_synthetic = False
                    else:
                        T[i, j] = self._fallback_time(pts[i], pts[j])
                else:
                    T[i, j] = self._fallback_time(pts[i], pts[j])

                if hazard is not None:
                    R[i, j] = hazard.exposure(pts[i], pts[j], t_depart=0.0)

        return {
            "time_s": T,
            "risk": R,
            "synthetic": self.is_synthetic,
            "note": ("SYNTHETIC straight-line matrix — NOT real road travel "
                     "times" if self.is_synthetic else
                     "Live matrix via GlobalMapsRouter; traffic snapshot only"),
        }

    def _fallback_time(self, a: Tuple[float, float],
                       b: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        dist = 6371000.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
        # detour factor 1.3 for street network (assumption, flagged)
        return float(dist * 1.3 / max(self.cfg.average_speed_ms, 0.1))


# ═══════════════════════════════════════════════════════════════════════
# [P/D] MULTI-STOP PLANNER
# ═══════════════════════════════════════════════════════════════════════

class MultiStopEvacuationPlanner:
    """
    [P/D] Time-windowed bi-objective multi-stop evacuation planner.

    Convention:
      stops[0]  = origin (must be first)
      stops[-1] = shelter (must be last)
      middle    = required pickups / medical / water stops

    [P] Exact DP validity conditions:
      (i)   travel-time matrix static during the plan,
      (ii)  waiting at a stop is allowed,
      (iii) service start = max(arrival, open), infeasible if > close,
      (iv)  Pareto labels (departure_time, cumulative_risk) are kept,
            and a label with both components <= dominates.
      Under (i)-(iv), keeping only non-dominated labels per
      (subset, last) loses no optimal feasible completion.

    If label_cap_per_state is reached, exact=False and the result is [D].
    """

    def __init__(self, cfg: Optional[MultiStopConfig] = None,
                 router: Optional[Any] = None):
        self.cfg = cfg or MultiStopConfig()
        self.matrix_builder = TravelMatrixBuilder(self.cfg, router)
        self.rng = np.random.default_rng(self.cfg.seed)

    # ── evaluate a fixed order ────────────────────────────────────────
    def evaluate(self, order_idx: List[int], stops: List[EvacStop],
                 matrix: Dict[str, Any]) -> RouteCandidate:
        T, R = matrix["time_s"], matrix["risk"]
        t = 0.0
        risk = 0.0
        arrivals: List[float] = []
        waits: List[float] = []
        notes: List[str] = []
        feasible = True

        prev = order_idx[0]
        s0 = stops[prev]
        start0 = max(t, s0.open_t)
        if not s0.window_ok(start0):
            feasible = False
        t = start0 + s0.service_s
        arrivals.append(start0)
        waits.append(start0 - t if start0 > t else 0.0)

        for idx in order_idx[1:]:
            s = stops[idx]
            if not math.isfinite(T[prev, idx]):
                feasible = False
                notes.append(f"unreachable edge {prev}->{idx}")
                break
            travel = float(T[prev, idx])
            arrival = t + travel
            service_start = max(arrival, s.open_t)
            wait = service_start - arrival
            if not s.window_ok(service_start):
                feasible = False
                notes.append(f"time window missed at {s.stop_id}")
            if s.kind == "shelter" and not s.accessible:
                notes.append("shelter not marked accessible")
            t = service_start + s.service_s
            risk += float(R[prev, idx])
            arrivals.append(arrival)
            waits.append(wait)
            prev = idx

        return RouteCandidate(
            order=[stops[i].stop_id for i in order_idx],
            total_time_s=float(t),
            total_risk=float(risk),
            feasible=feasible,
            arrivals=arrivals,
            waits_s=waits,
            notes=notes + (["SYNTHETIC travel matrix"]
                           if matrix.get("synthetic") else []),
        )

    # ── Pareto label helpers ──────────────────────────────────────────
    @staticmethod
    def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1] and
                (a[0] < b[0] or a[1] < b[1]))

    def _prune(self, labels: List[Tuple[float, float, Tuple[int, ...]]]
               ) -> Tuple[List[Tuple[float, float, Tuple[int, ...]]], bool]:
        """Return Pareto frontier and whether any cap was applied."""
        labels.sort(key=lambda z: (z[0], z[1]))
        front: List[Tuple[float, float, Tuple[int, ...]]] = []
        for lab in labels:
            if not any(self._dominates(o[:2], lab[:2]) for o in front):
                front = [o for o in front
                         if not self._dominates(lab[:2], o[:2])]
                front.append(lab)
        capped = False
        if len(front) > self.cfg.label_cap_per_state:
            front = front[:self.cfg.label_cap_per_state]
            capped = True
        return front, capped

    # ── [P] exact DP ──────────────────────────────────────────────────
    def solve_exact(self, stops: List[EvacStop],
                    matrix: Dict[str, Any]) -> Dict[str, Any]:
        """
        [P] Label-setting DP over required middle stops.
        Returns Pareto set of complete routes.
        """
        n = len(stops)
        if n < 2:
            return {"error": "need origin and shelter"}
        middle = [i for i in range(1, n - 1) if stops[i].required]
        m = len(middle)
        if n > self.cfg.max_exact_stops:
            return {"error": f"n={n} exceeds max_exact_stops="
                             f"{self.cfg.max_exact_stops}; use heuristic"}

        T, R = matrix["time_s"], matrix["risk"]
        full = (1 << m) - 1

        # state: (mask, last_index) -> list of (departure, risk, path)
        states: Dict[Tuple[int, int], List[Tuple[float, float, Tuple[int, ...]]]] = {}
        s0 = stops[0]
        dep0 = max(0.0, s0.open_t) + s0.service_s
        states[(0, 0)] = [(dep0, 0.0, (0,))]
        truncated = False

        for mask in range(full + 1):
            for last in range(n):
                labels = states.get((mask, last))
                if not labels:
                    continue
                for j_pos, j in enumerate(middle):
                    if mask & (1 << j_pos):
                        continue
                    sj = stops[j]
                    if not math.isfinite(T[last, j]):
                        continue
                    new_labels = []
                    for dep, risk, path in labels:
                        arrival = dep + float(T[last, j])
                        start = max(arrival, sj.open_t)
                        if start > sj.close_t:
                            continue
                        new_dep = start + sj.service_s
                        new_risk = risk + float(R[last, j])
                        new_labels.append((new_dep, new_risk, path + (j,)))
                    if new_labels:
                        key = (mask | (1 << j_pos), j)
                        merged = states.get(key, []) + new_labels
                        pruned, cap = self._prune(merged)
                        states[key] = pruned
                        truncated = truncated or cap

        # close at shelter (index n-1)
        finals: List[RouteCandidate] = []
        shelter = stops[-1]
        for (mask, last), labels in states.items():
            if mask != full:
                continue
            if not math.isfinite(T[last, n - 1]):
                continue
            for dep, risk, path in labels:
                arrival = dep + float(T[last, n - 1])
                start = max(arrival, shelter.open_t)
                if start > shelter.close_t:
                    continue
                end = start + shelter.service_s
                full_path = path + (n - 1,)
                cand = self.evaluate(list(full_path), stops, matrix)
                cand.total_time_s = float(end)
                cand.total_risk = float(risk + float(R[last, n - 1]))
                cand.exact = not truncated
                finals.append(cand)

        # Pareto over complete routes
        archive: List[RouteCandidate] = []
        for c in sorted(finals, key=lambda c: c.objective):
            if not any(self._dominates(o.objective, c.objective)
                       for o in archive):
                archive = [o for o in archive
                           if not self._dominates(c.objective, o.objective)]
                archive.append(c)

        return {
            "status": "ok" if archive else "infeasible",
            "pareto_routes": archive,
            "exact": not truncated,
            "truncated": truncated,
            "note": ("Exact under static-matrix + waiting-allowed assumptions"
                     if not truncated else
                     "LABEL CAP HIT => APPROXIMATE, not exact"),
        }

    # ── [D] heuristic for larger instances ────────────────────────────
    def solve_heuristic(self, stops: List[EvacStop],
                        matrix: Dict[str, Any]) -> Dict[str, Any]:
        n = len(stops)
        middle = [i for i in range(1, n - 1) if stops[i].required]
        T = matrix["time_s"]
        archive: List[RouteCandidate] = []

        def consider(order: List[int]) -> None:
            cand = self.evaluate(order, stops, matrix)
            if not cand.feasible:
                return
            if not any(self._dominates(o.objective, cand.objective)
                       for o in archive):
                archive[:] = [o for o in archive
                              if not self._dominates(cand.objective, o.objective)]
                archive.append(cand)

        for restart in range(self.cfg.n_multistarts):
            remaining = middle.copy()
            self.rng.shuffle(remaining) if restart else None
            order = [0]
            cur = 0
            while remaining:
                # greedy by travel time, tie-broken by index
                nxt = min(remaining,
                          key=lambda j: (T[cur, j] if math.isfinite(T[cur, j])
                                         else float("inf"), j))
                order.append(nxt)
                remaining.remove(nxt)
                cur = nxt
            order.append(n - 1)
            consider(order)

            # 2-opt on the middle segment
            best = order.copy()
            best_c = self.evaluate(best, stops, matrix)
            for _ in range(self.cfg.max_local_search_iters):
                improved = False
                for a in range(1, len(best) - 2):
                    for b in range(a + 1, len(best) - 1):
                        trial = best[:a] + best[a:b + 1][::-1] + best[b + 1:]
                        tc = self.evaluate(trial, stops, matrix)
                        if (tc.feasible and
                                tc.total_time_s < best_c.total_time_s - 1e-9):
                            best, best_c = trial, tc
                            improved = True
                if not improved:
                    break
            consider(best)

        return {
            "status": "ok" if archive else "infeasible",
            "pareto_routes": archive,
            "exact": False,
            "note": "Heuristic (NN + 2-opt); NO optimality guarantee",
        }

    def solve(self, stops: List[EvacStop],
              hazard: Optional[HazardField] = None,
              matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Full plan: build matrix, choose solver, return Pareto set."""
        if len(stops) < 2 or stops[0].kind != "origin" or stops[-1].kind != "shelter":
            return {"error": "stops[0] must be origin, stops[-1] must be shelter"}
        mat = matrix or self.matrix_builder.build(stops, hazard)
        n_required = 2 + sum(1 for s in stops[1:-1] if s.required)
        if n_required <= self.cfg.max_exact_stops:
            result = self.solve_exact(stops, mat)
            if result.get("status") == "infeasible":
                # try heuristic as a second chance (different pruning)
                alt = self.solve_heuristic(stops, mat)
                if alt["status"] == "ok":
                    result = alt
        else:
            result = self.solve_heuristic(stops, mat)
        result["matrix_synthetic"] = mat.get("synthetic", True)
        result["matrix_note"] = mat.get("note", "")
        return result


# ═══════════════════════════════════════════════════════════════════════
# [P/D] NO-ZENO RE-PLANNER
# ═══════════════════════════════════════════════════════════════════════

class NoZenoReplanner:
    """
    [P/D] Bound the NUMBER of re-plans, not the accuracy of each plan.

    N(T) <= 1 + floor(T / tau_e)  and  N <= max_replans_per_horizon.

    Deferred re-plan requests are RECORDED, never dropped
    (CFD §3.3 discipline). This is an engineering guarantee about the
    event counter; it is NOT a contraction or O(1)-complexity claim.
    """

    def __init__(self, cfg: MultiStopConfig):
        self.cfg = cfg
        self.last_t: Optional[float] = None
        self.n_done = 0
        self.deferred: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

    def request(self, reason: str, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        if self.last_t is not None and now - self.last_t < self.cfg.replan_min_interval_s:
            self.deferred.append({"reason": reason, "t": now})
            return {"status": "deferred", "n_done": self.n_done}
        if self.n_done >= self.cfg.max_replans_per_horizon:
            self.deferred.append({"reason": reason, "t": now,
                                  "why": "horizon cap"})
            return {"status": "capped", "n_done": self.n_done}
        self.last_t = now
        self.n_done += 1
        self.history.append({"reason": reason, "t": now, "n": self.n_done})
        return {"status": "accepted", "n_done": self.n_done}

    def bound_check(self, t_start: float) -> Dict[str, Any]:
        if self.last_t is None:
            return {"invariant_holds": True, "n_done": 0, "bound": 1}
        T = self.last_t - t_start
        bound = 1 + int(T / self.cfg.replan_min_interval_s)
        return {
            "invariant_holds": self.n_done <= bound,
            "n_done": self.n_done,
            "bound": bound,
            "n_deferred": len(self.deferred),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Multi-Stop Evacuation Planner — Smoke Test")
    print("=" * 72)
    print(f"  GlobalMaps router available : {HAS_ROUTER}")
    print(f"  Geo transform available     : {HAS_GEO}")

    cfg = MultiStopConfig(use_live_router=False, max_exact_stops=8)
    planner = MultiStopEvacuationPlanner(cfg)

    stops = [
        EvacStop("O", "Affected community", 13.750, 100.500, kind="origin"),
        EvacStop("P1", "Elder pickup", 13.760, 100.510, kind="pickup",
                 service_s=120.0, close_t=1800.0),
        EvacStop("M", "Triage post", 13.770, 100.505, kind="medical",
                 service_s=180.0, open_t=60.0),
        EvacStop("W", "Water point", 13.765, 100.520, kind="water",
                 required=False, service_s=60.0),
        EvacStop("S", "Shelter (high ground)", 13.790, 100.530,
                 kind="shelter", accessible=True, close_t=3600.0),
    ]

    hazard = HazardField(cfg)
    hazard.add(HazardCell(13.765, 100.510, radius_m=400.0,
                          intensity=0.8, hazard_type="FLOOD"))

    print("\n  ── Hazard field + Dörfler marking ──")
    indicators = np.array([0.1, 0.9, 0.2, 0.7, 0.05])
    marked = hazard.dorfler_mark(indicators, theta=0.5)
    total = float(np.sum(indicators ** 2))
    captured = float(np.sum(indicators[marked] ** 2)) / total
    print(f"    Marked cells         : {marked}")
    print(f"    Captured error share : {captured:.3f} (theta=0.5)")
    print(f"    [{'PASS' if captured >= 0.5 else 'FAIL'}] Dörfler bulk condition")

    print("\n  ── Multi-stop solve (synthetic matrix, FLAGGED) ──")
    result = planner.solve(stops, hazard=hazard)
    print(f"    Status               : {result['status']}")
    print(f"    Exact                : {result.get('exact')}")
    print(f"    Matrix synthetic     : {result.get('matrix_synthetic')}")
    for cand in result.get("pareto_routes", [])[:3]:
        print(f"    Route: {' -> '.join(cand.order)}")
        print(f"      time={cand.total_time_s:.1f}s risk={cand.total_risk:.3f} "
              f"feasible={cand.feasible}")

    print("\n  ── Exact DP vs brute force (small instance) ──")
    small = [stops[0], stops[1], stops[2], stops[4]]
    mat = planner.matrix_builder.build(small, None)
    exact = planner.solve_exact(small, mat)
    T, R = mat["time_s"], mat["risk"]
    brute_best = float("inf")
    for perm in itertools.permutations([1, 2]):
        order = [0, *perm, 3]
        c = planner.evaluate(order, small, mat)
        if c.feasible:
            brute_best = min(brute_best, c.total_time_s)
    dp_best = min((c.total_time_s for c in exact["pareto_routes"]
                   if c.feasible), default=float("inf"))
    print(f"    Brute force best     : {brute_best:.3f}")
    print(f"    DP best              : {dp_best:.3f}")
    print(f"    [{'PASS' if abs(dp_best-brute_best) < 1e-6 else 'FAIL'}] "
          f"DP matches brute force")

    print("\n  ── CUSUM + induced change-point view (Paper 6) ──")
    series = np.concatenate([np.ones(20) * 0.1,
                             np.ones(30) * 1.5,
                             np.ones(20) * 0.1])
    det = CUSUMChangePointDetector(mu0=0.1, drift_k=0.3, threshold_h=3.0)
    cp = det.detect(series)
    print(f"    Candidate change pts : {cp['candidate_change_points']}")
    taus = [0, 20, 50, 70]
    induced = ChangePointInducedView.build(taus, [0.1, 1.5, 0.1])
    print(f"    Induced blocks       : {induced.get('blocks')}")
    print(f"    Note                 : {induced.get('note')}")

    print("\n  ── Assumption-light ETA CI (Paper 11) ──")
    rng = np.random.default_rng(7)
    travel_samples = rng.normal(600, 80, 120)
    eta = AssumptionLightETA(cfg)
    ci = eta.interval(travel_samples)
    if "error" not in ci:
        print(f"    Point estimate       : {ci['point_estimate_s']:.1f}s")
        print(f"    CI                   : [{ci['ci_lower_s']:.1f}, "
              f"{ci['ci_upper_s']:.1f}] at {ci['confidence_level']:.0%}")
        print(f"    Block b              : {ci['block_length_b']}")
        print(f"    Note                 : {ci['note']}")

    print("\n  ── No-Zeno re-plan bound ──")
    replanner = NoZenoReplanner(cfg)
    t0 = 1000.0
    for k, reason in enumerate(["flood_expand", "road_closed", "new_pickup"]):
        r = replanner.request(reason, now=t0 + k * 5.0)
        print(f"    request {k+1}: {r['status']}")
    chk = replanner.bound_check(t0)
    print(f"    bound check          : {chk}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Routes are suggestions; official evacuation orders override")
    print("  • Synthetic matrix is FLAGGED, never presented as real traffic")
    print("  • Exact DP only under static matrix + waiting allowed")
    print("  • Label cap hit => APPROXIMATE, explicitly marked")
    print("  • CUSUM change points are CANDIDATES (Paper 6 Open Problem 6.1)")
    print("  • ETA CI coverage is ASYMPTOTIC, not a safety guarantee")
    print("  • No-Zeno bounds EVENT COUNT, not solver accuracy or O(1) cost")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
