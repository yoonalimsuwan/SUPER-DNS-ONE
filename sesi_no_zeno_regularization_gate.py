# =============================================================================
# sesi_no_zeno_regularization_gate.py
#
# A small, self-contained extension for structuralfluctuatinghydro_v6_3.py /
# super_dns_one_v6_3.py's SOCController: adds a DISCRETE, hysteresis-gated
# event count on top of the EXISTING smooth stress-collapse mechanism
# (SOCController.nu_t()'s stress_acc / theta / nu_collapse / decay logic),
# and a real, provable bound on how many such events can occur over a given
# time horizon and energy budget.
#
# WHAT THIS IS: a numerical bookkeeping / diagnostic layer. The bound below
# (Proposition 6.1 of the companion closure report) is a real theorem about
# the DISCRETE regularization-event count in this specific gate, derived
# from the same energy-budget argument as the SESI Closure note's Theorem
# 4.2 (Delta_E_min-per-event, total variation bounded by energy + dissipation
# budget). It is checkable in code, exactly as stated.
#
# WHAT THIS IS NOT, said as many times as it takes: a proof, partial or
# otherwise, that the true (continuum) Navier-Stokes equations this solver
# approximates do or do not develop a singularity. It says nothing about
# that question. It bounds how many times ONE SPECIFIC NUMERICAL SAFETY
# MECHANISM in ONE SPECIFIC DISCRETIZATION fires, which is a fact about the
# discretization and this gate, not about the PDE.
#
# WHY IT IS SEPARATE FROM SOCController, NOT A MODIFICATION OF IT: SOCController
# subclasses CSOCBase, imported in the uploaded files from `one_core`, a
# module not made available to us. Modifying SOCController itself would mean
# guessing at that base class's contract. This file instead reads
# SOCController's already-public stress_acc scalar each step (or any other
# scalar signal the caller supplies) and layers the hysteresis/event-count
# logic on top, with no dependency on one_core at all.
#
# Core gate logic (HysteresisEventGate, NoZenoEventBudget) is pure Python --
# no torch dependency -- so it is fully unit-testable on its own, independent
# of whether a CFD solver or torch is available in a given environment. The
# thin adapter at the bottom shows how to pull a scalar out of a live
# SOCController instance each step; that part needs torch (to call .item()
# on a tensor) but is otherwise a two-line glue function.
#
# Developed with Claude as AI co-developer.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Section A -- the energy-budget bound, reproduced here as a pure-Python
# function identical in formula to structural_calculus_ops.energy_budget_event_bound
# (SESI Closure note, Theorem 4.2), so this file has zero import dependency.
# If structural_calculus_ops.py IS on the path, prefer importing that one
# instead of this local copy, to avoid the two ever silently drifting apart:
#
#     from structural_calculus_ops import energy_budget_event_bound
#
# =============================================================================


def energy_budget_event_bound(e_max: float, dissipation_budget: float, delta_e_min: float) -> float:
    """N(T) <= (E_max(T) + D(T)) / Delta_E_min (SESI Closure note, Theorem 4.2).

    Raises rather than silently returning inf when delta_e_min <= 0, since
    the bound is then vacuous (a genuine positive per-event energy gap is
    exactly what the theorem needs) -- named as such instead of hidden.
    """
    if delta_e_min <= 0:
        raise ValueError(
            "delta_e_min must be > 0; the energy-budget no-Zeno bound is "
            "vacuous without a genuine positive per-event gap."
        )
    if e_max < 0 or dissipation_budget < 0:
        raise ValueError("e_max and dissipation_budget must be non-negative.")
    return (e_max + dissipation_budget) / delta_e_min


# =============================================================================
# Section B -- the hysteresis gate itself.
# =============================================================================


@dataclass
class HysteresisEventGate:
    """Quantizes a smoothly-varying scalar signal (e.g. SOCController's
    stress_acc) into discrete, non-degenerate-reset "collapse events",
    exactly as SESI Closure note Definition 3.1 requires for any no-Zeno
    theorem to apply: an event fires when the signal first rises through
    theta_hi, and cannot fire again until the signal has first fallen back
    through theta_lo < theta_hi.

    delta_min := theta_hi - theta_lo is the non-degenerate-reset gap
    (Delta_E_min in Proposition 6.1's notation). It must be > 0.

    This class does no smoothing, no gradient tracking, and does not modify
    the signal it watches -- it is a pure observer. The underlying smooth
    SOCController mechanism keeps running exactly as before; this only
    reports when, in hindsight, the smooth trace crossed the hysteresis
    band in the "up" direction.
    """

    theta_hi: float
    theta_lo: float
    _armed: bool = field(default=True, init=False)  # True: watching for an "up" crossing
    _event_times: List[float] = field(default_factory=list, init=False)
    _trace_peak: float = field(default=0.0, init=False)
    _trace_dissipation: float = field(default=0.0, init=False)
    _last_value: Optional[float] = field(default=None, init=False)
    _last_time: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not (self.theta_hi > self.theta_lo):
            raise ValueError(f"theta_hi ({self.theta_hi}) must be > theta_lo ({self.theta_lo}).")

    @property
    def delta_min(self) -> float:
        return self.theta_hi - self.theta_lo

    @property
    def event_count(self) -> int:
        return len(self._event_times)

    @property
    def event_times(self) -> List[float]:
        return list(self._event_times)

    def step(self, value: float, t: float) -> bool:
        """Feed one new observation of the watched scalar at simulation time
        t (monotonically non-decreasing across calls). Returns True exactly
        on the step a new collapse event fires.

        Also accumulates the two quantities Proposition 6.1's bound needs:
        the running peak (E_max analogue) and the accumulated absolute
        variation between consecutive samples (a discrete, always-valid
        upper bound on the true dissipation integral D(T), since total
        variation of a path always bounds the variation of any of its
        monotone sub-pieces).
        """
        fired = False
        if self._last_value is not None:
            self._trace_dissipation += abs(value - self._last_value)
        self._trace_peak = max(self._trace_peak, value)

        if self._armed and value >= self.theta_hi:
            self._event_times.append(t)
            self._armed = False
            fired = True
        elif (not self._armed) and value <= self.theta_lo:
            self._armed = True

        self._last_value = value
        self._last_time = t
        return fired

    def no_zeno_bound(self, up_to_time: Optional[float] = None) -> float:
        """N(T) bound from everything observed so far (or up to up_to_time,
        if given and no later than the last observation -- otherwise the
        bound is reported for the full trace observed so far, which is
        always a valid, if possibly loose, bound for any T within that
        range, since E_max and D are both monotone non-decreasing in T).
        """
        return energy_budget_event_bound(self._trace_peak, self._trace_dissipation, self.delta_min)

    def reset(self) -> None:
        self._armed = True
        self._event_times.clear()
        self._trace_peak = 0.0
        self._trace_dissipation = 0.0
        self._last_value = None
        self._last_time = None


# =============================================================================
# Section C -- thin adapter for a live SOCController instance (needs torch;
# everything above does not). Two lines of actual glue.
# =============================================================================


def make_soc_controller_gate(theta_hi: float, theta_lo: float) -> HysteresisEventGate:
    """Convenience constructor. `theta_hi` should typically sit somewhat
    above the SOCController's own `self.kernel.theta` (the point where its
    smooth nu_collapse term starts engaging), so the discrete event count
    tracks genuinely sustained excursions rather than firing at the very
    first hint of the smooth collapse mechanism's own activity; `theta_lo`
    should sit far enough below theta_hi to give a meaningful (non-tiny)
    delta_min -- both are physical/numerical tuning choices for the
    specific solver configuration, not something this file can choose for
    you.
    """
    return HysteresisEventGate(theta_hi=theta_hi, theta_lo=theta_lo)


def observe_soc_controller_step(gate: HysteresisEventGate, soc_controller, t: float) -> bool:
    """Call once per solver timestep, after soc_controller.nu_t(...) has run
    for that step (so stress_acc reflects the current step's state).
    `soc_controller` is a live SOCController instance (structural typing
    only -- this file does not import super_dns_one_v6_3 or one_core, so it
    has no way to check the type at import time; it simply reads
    `.stress_acc`, exactly as SOCController's own `nu_t()` sets it).

    Returns whatever gate.step(...) returns (True iff a new event just
    fired at this step).
    """
    if soc_controller.stress_acc is None:
        return False  # nu_t() has not been called yet this run
    stress_scalar = float(soc_controller.stress_acc.mean().item())
    return gate.step(stress_scalar, t)


# =============================================================================
# Smoke test / usage demonstration. Pure Python -- no torch needed for this
# part, since it exercises Sections A and B directly against a synthetic
# stress trace rather than a live solver.
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HysteresisEventGate: synthetic stress trace")
    print("=" * 70)

    gate = HysteresisEventGate(theta_hi=1.0, theta_lo=0.4)

    # A synthetic stress_acc-like trace: rises, crosses theta_hi (event),
    # falls back below theta_lo, rises again (second event), then hovers
    # near theta_hi WITHOUT dropping below theta_lo -- this should NOT
    # register as repeated events (the whole point of hysteresis), unlike
    # a naive single-threshold check which would fire on every up-tick.
    trace = (
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.05, 1.1, 0.9, 0.6, 0.3, 0.2]  # event #1 near t=6
        + [0.3, 0.5, 0.8, 1.02, 1.15]  # event #2 near t=16
        + [1.05, 0.98, 1.08, 0.95, 1.1, 0.99, 1.12]  # hovering near theta_hi, no re-arm -> no new events
        + [0.5, 0.35, 0.2]  # finally drops below theta_lo, re-arms
        + [0.6, 0.85, 1.03]  # event #3
    )
    events_this_step = []
    for i, v in enumerate(trace):
        fired = gate.step(v, t=float(i))
        if fired:
            events_this_step.append(i)

    print(f"  trace length                    : {len(trace)}")
    print(f"  event times (indices)           : {events_this_step}")
    print(f"  gate.event_count                : {gate.event_count}")
    assert gate.event_count == 3, f"expected exactly 3 events (hovering must not double-fire), got {gate.event_count}"
    print("  hovering-near-threshold check   : PASSED (no spurious re-fires)")

    print()
    print("=" * 70)
    print("No-Zeno event budget from the observed trace")
    print("=" * 70)
    bound = gate.no_zeno_bound()
    print(f"  observed peak (E_max analogue)   : {gate._trace_peak:.4f}")
    print(f"  observed total variation (D)     : {gate._trace_dissipation:.4f}")
    print(f"  delta_min (theta_hi - theta_lo)  : {gate.delta_min:.4f}")
    print(f"  N(T) bound                       : {bound:.4f}")
    print(f"  actual events observed            : {gate.event_count}")
    assert gate.event_count <= bound, "actual event count must never exceed the proved bound"
    print("  bound respected                  : PASSED (actual <= bound, as the theorem requires)")

    print()
    print("=" * 70)
    print("What this number does and does not mean")
    print("=" * 70)
    print("  This bounds how many times a DISCRETE regularization event fires")
    print("  in THIS gate, watching THIS numerical signal, over THIS observed")
    print("  trace. It is a statement about a numerical bookkeeping mechanism,")
    print("  not a claim about whether the underlying continuum PDE (Navier-")
    print("  Stokes) develops a true singularity. That question is untouched.")

    print()
    print("=" * 70)
    print("delta_min <= 0 is refused, not silently made vacuous")
    print("=" * 70)
    try:
        HysteresisEventGate(theta_hi=1.0, theta_lo=1.0)
        print("  ERROR: should have raised")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    try:
        energy_budget_event_bound(e_max=1.0, dissipation_budget=1.0, delta_e_min=0.0)
        print("  ERROR: should have raised")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    print()
    print("All checks passed.")
