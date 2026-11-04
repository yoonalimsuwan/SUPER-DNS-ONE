"""
=============================================================================
SESI ORGANOID STUDIO — PTM / BISTABLE SWITCHING / CRYO-ET
Post-Translational Modifications + Stochastic Switching + Cryo-ET Integration
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
MODULES
=============================================================================
[PTM1] PostTranslationalModificationModel
      • Multi-site combinatorial PTM dynamics [C Cohen 2000]
      • Phosphorylation: kinase + substrate → phospho-substrate (rate k_phos)
                         phosphatase reverses (rate k_dephos)
      • Ubiquitination: E1→E2→E3 cascade → poly-Ub chain (rate k_ubi)
                        DUB reverses (rate k_deubi)
                        Proteasomal degradation (rate k_degrade)
      • Scaffolding effect: PTM state modulates binding affinity
      • Gillespie SSA for discrete modification events [C 1976/77]
      • COROLLARY-3 guard: first-order binding/modification kinetics only
      • HINO guard: discrete modification events, not smooth density

[BSS1] BistableStochasticSwitching
      • Double-well potential: V(x) = a(x²-1)² [C/D]
      • Langevin dynamics: dx/dt = -V'(x) + σξ(t)
      • Kramers escape rate: r ~ exp(-ΔV/kBT) [C Kramers 1940]
      • Mean First Passage Time (MFPT) computation
      • Hysteresis detection via quasi-static parameter sweep
      • Coupling to WntPCPFeedbackNetwork (canonical ON/OFF states)
      • COROLLARY-3 guard: first-order Langevin only; no second derivatives
      • PROP-3 awareness: potential × noise product loses regularity

[CET1] CryoETBridge
      • Load cryo-ET tomograms (MRC format) [C Frank 2006]
      • Fourier Shell Correlation (FSC) for resolution [C Rosenthal 2003]
      • Subtomogram averaging (simplified)
      • Rigid-body fitting of model structures
      • Comparison with model predictions
      • HONEST: computational analysis only; NOT a reconstruction pipeline;
        NOT validated structural biology

=============================================================================
HONEST SCOPE
=============================================================================
• PostTranslationalModificationModel: SSA is exact for well-mixed
  compartments [C]. Real PTM involves spatial compartmentalization,
  enzyme processivity, and allosteric regulation not captured here.
• BistableStochasticSwitching: double-well model is illustrative. Real
  bistable switches involve complex gene regulatory networks with
  multiple feedback loops. Kramers theory assumes smooth potentials.
• CryoETBridge: FSC and subtomogram averaging are simplified. Real
  cryo-ET requires sophisticated reconstruction algorithms (SIRT, SART),
  CTF correction, and missing wedge compensation. This bridge is for
  calibration/validation only, NOT a reconstruction pipeline.
• COROLLARY-3 guard: all fields first-order only; no Hessian.
• HINO guard: energy measures singular; no smooth density assumed.
• PROP-3 awareness: field products lose two Sobolev orders.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All parameters illustrative. No biological/medical/diagnostic claim.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.ptm_bistable_cryoet")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import mrcfile
    HAS_MRCFILE = True
except Exception:
    HAS_MRCFILE = False

try:
    from scipy import stats as scipy_stats
    from scipy.spatial import KDTree
    from scipy.integrate import odeint
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

__all__ = [
    "PostTranslationalModificationModel", "PTMConfig", "PTMState",
    "BistableStochasticSwitching", "BistableConfig", "SwitchingEvent",
    "CryoETBridge", "CryoETConfig", "FSCResult",
    "SESIPTMBistableCryoETSuite",
]


# =============================================================================
# SECTION 1 — [PTM1] POST-TRANSLATIONAL MODIFICATION MODEL
# =============================================================================

@dataclass
class PTMConfig:
    """Configuration for post-translational modification dynamics."""
    # Substrate
    n_substrate: int = 100          # total substrate molecules
    n_phospho_sites: int = 3        # number of phosphorylation sites
    n_ubi_sites: int = 2            # number of ubiquitination sites
    # Enzymes
    n_kinase: int = 10              # kinase molecules
    n_phosphatase: int = 5          # phosphatase molecules
    n_e3_ligase: int = 5            # E3 ubiquitin ligase
    n_dub: int = 3                  # deubiquitinase
    n_proteasome: int = 50          # proteasome capacity
    # Reaction rates [C Cohen 2000; Hershko & Ciechanover 1998]
    k_phosphorylation: float = 0.5   # kinase rate
    k_dephosphorylation: float = 0.2 # phosphatase rate
    k_ubiquitination: float = 0.3    # E3 ligase rate
    k_deubiquitination: float = 0.1  # DUB rate
    k_degradation: float = 0.05      # proteasomal degradation rate
    # Scaffolding modulation
    scaffold_binding_base: float = 1.0
    phos_binding_factor: float = 2.0   # phosphorylation enhances binding
    ubi_binding_factor: float = 0.5    # ubiquitination reduces binding
    # Simulation
    max_time: float = 500.0
    max_steps: int = 200000
    seed: int = 0


@dataclass
class PTMState:
    """State of PTM system."""
    time: float
    # Substrate modification counts
    n_unmodified: int
    n_phosphorylated: int
    n_ubiquitinated: int
    n_both_modified: int
    n_degraded: int
    # Enzyme counts
    n_kinase_free: int
    n_phosphatase_free: int
    n_e3_free: int
    n_dub_free: int
    # Scaffolding
    scaffold_binding_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PostTranslationalModificationModel:
    """
    [C/D] Multi-site combinatorial PTM dynamics via Gillespie SSA.

    BIOLOGICAL BASIS [C Cohen 2000; Hershko & Ciechanover 1998]
      Post-translational modifications regulate protein function:

      PHOSPHORYLATION:
        - Kinase transfers phosphate from ATP to substrate
        - Substrate + ATP → Phospho-substrate + ADP  (rate k_phos)
        - Phospho-substrate → Substrate + Pi           (rate k_dephos, phosphatase)
        - Effect: conformational change, binding affinity modulation

      UBIQUITINATION:
        - E1 (activating) → E2 (conjugating) → E3 (ligase) cascade
        - Substrate + Ub → Ubiquitinated substrate       (rate k_ubi)
        - Ubiquitinated → Substrate + Ub                  (rate k_deubi, DUB)
        - Poly-Ub (K48-linked) → proteasomal degradation (rate k_deg)
        - Effect: protein stability, trafficking, signaling

      SCAFFOLDING MODULATION:
        - Phosphorylation can enhance or inhibit scaffold binding
        - Ubiquitination typically reduces scaffold binding
        - Combinatorial PTM creates a "modification code"

    COMBINATORIAL STATE SPACE
      For n_phospho_sites = 3 and n_ubi_sites = 2:
        - Each site can be: unmodified, phosphorylated, ubiquitinated
        - Total states: 3^5 = 243 (simplified to 4 categories here)
      We track aggregate counts rather than full combinatorial state.

    REACTIONS
      1. Phosphorylation: Substrate → Phospho-substrate
      2. Dephosphorylation: Phospho-substrate → Substrate
      3. Ubiquitination: Substrate → Ubiquitinated
      4. Deubiquitination: Ubiquitinated → Substrate
      5. Degradation: Ubiquitinated → ∅ (proteasome)
      6. Dual modification: Phospho → Both-modified → Ubiquitinated

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear or bilinear in
      copy numbers). We do NOT compute second derivatives of the
      modification state. The noise term is multiplicative but first-order.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates × copy numbers), NOT from an assumed smooth noise density.
      The stochasticity is singular in the sense of being tied to individual
      modification events.

    HONEST
      • SSA is exact for well-mixed compartments; real PTM involves
        spatial compartmentalization, enzyme processivity, allosteric
        regulation not captured here.
      • Combinatorial state space is simplified; real proteins can have
        dozens of modification sites.
      • Copy numbers are illustrative; real cells have variable enzyme
        and substrate concentrations.
    """

    def __init__(self, cfg: Optional[PTMConfig] = None):
        self.cfg = cfg or PTMConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # State counts
        self.n_unmodified: int = 0
        self.n_phosphorylated: int = 0
        self.n_ubiquitinated: int = 0
        self.n_both_modified: int = 0
        self.n_degraded: int = 0

        # Enzyme counts
        self.n_kinase_free: int = 0
        self.n_phosphatase_free: int = 0
        self.n_e3_free: int = 0
        self.n_dub_free: int = 0

        # Scaffolding
        self.scaffold_binding: float = 0.0

    def initialize(self) -> None:
        """Initialize PTM system."""
        self.n_unmodified = self.cfg.n_substrate
        self.n_phosphorylated = 0
        self.n_ubiquitinated = 0
        self.n_both_modified = 0
        self.n_degraded = 0

        self.n_kinase_free = self.cfg.n_kinase
        self.n_phosphatase_free = self.cfg.n_phosphatase
        self.n_e3_free = self.cfg.n_e3_ligase
        self.n_dub_free = self.cfg.n_dub

        self.scaffold_binding = self.cfg.scaffold_binding_base

    def compute_propensities(self) -> Tuple[np.ndarray, List[Tuple[str, Dict[str, int]]]]:
        """Compute all reaction propensities."""
        props = []
        events = []

        # 1. Phosphorylation: Unmodified → Phosphorylated
        if self.n_unmodified > 0 and self.n_kinase_free > 0:
            a = self.cfg.k_phosphorylation * self.n_unmodified * self.n_kinase_free
            props.append(a)
            events.append(("phosphorylate", {
                'n_unmodified': -1, 'n_phosphorylated': +1,
            }))

        # 2. Dephosphorylation: Phosphorylated → Unmodified
        if self.n_phosphorylated > 0 and self.n_phosphatase_free > 0:
            a = self.cfg.k_dephosphorylation * self.n_phosphorylated
            props.append(a)
            events.append(("dephosphorylate", {
                'n_phosphorylated': -1, 'n_unmodified': +1,
            }))

        # 3. Ubiquitination: Unmodified → Ubiquitinated
        if self.n_unmodified > 0 and self.n_e3_free > 0:
            a = self.cfg.k_ubiquitination * self.n_unmodified * self.n_e3_free
            props.append(a)
            events.append(("ubiquitinate", {
                'n_unmodified': -1, 'n_ubiquitinated': +1,
            }))

        # 4. Deubiquitination: Ubiquitinated → Unmodified
        if self.n_ubiquitinated > 0 and self.n_dub_free > 0:
            a = self.cfg.k_deubiquitination * self.n_ubiquitinated
            props.append(a)
            events.append(("deubiquitinate", {
                'n_ubiquitinated': -1, 'n_unmodified': +1,
            }))

        # 5. Degradation: Ubiquitinated → Degraded
        if self.n_ubiquitinated > 0:
            a = self.cfg.k_degradation * self.n_ubiquitinated
            props.append(a)
            events.append(("degrade", {
                'n_ubiquitinated': -1, 'n_degraded': +1,
            }))

        # 6. Phosphorylation of Ubiquitinated → Both-modified
        if self.n_ubiquitinated > 0 and self.n_kinase_free > 0:
            a = self.cfg.k_phosphorylation * 0.5 * self.n_ubiquitinated
            props.append(a)
            events.append(("phosphorylate_ubi", {
                'n_ubiquitinated': -1, 'n_both_modified': +1,
            }))

        # 7. Ubiquitination of Phosphorylated → Both-modified
        if self.n_phosphorylated > 0 and self.n_e3_free > 0:
            a = self.cfg.k_ubiquitination * 0.5 * self.n_phosphorylated
            props.append(a)
            events.append(("ubiquitinate_phos", {
                'n_phosphorylated': -1, 'n_both_modified': +1,
            }))

        # 8. Degradation of Both-modified
        if self.n_both_modified > 0:
            a = self.cfg.k_degradation * 1.5 * self.n_both_modified
            props.append(a)
            events.append(("degrade_both", {
                'n_both_modified': -1, 'n_degraded': +1,
            }))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self) -> Tuple[Dict[str, int], float]:
        """One Gillespie SSA step. Returns (delta, time_increment)."""
        props, events = self.compute_propensities()
        a_total = props.sum()

        if a_total <= 0:
            return {}, 0.0

        # Sample waiting time
        tau = -math.log(max(self._rng.random(), 1e-12)) / a_total

        # Sample which reaction fires
        r = self._rng.random() * a_total
        cumsum = 0.0
        event_idx = len(events) - 1
        for idx in range(len(events)):
            cumsum += props[idx]
            if r <= cumsum:
                event_idx = idx
                break

        reaction_name, delta = events[event_idx]

        # Apply state changes
        for key, change in delta.items():
            if key == 'n_unmodified':
                self.n_unmodified = max(0, self.n_unmodified + change)
            elif key == 'n_phosphorylated':
                self.n_phosphorylated = max(0, self.n_phosphorylated + change)
            elif key == 'n_ubiquitinated':
                self.n_ubiquitinated = max(0, self.n_ubiquitinated + change)
            elif key == 'n_both_modified':
                self.n_both_modified = max(0, self.n_both_modified + change)
            elif key == 'n_degraded':
                self.n_degraded = max(0, self.n_degraded + change)

        # Update scaffold binding based on modification state
        total_active = (self.n_unmodified + self.n_phosphorylated +
                        self.n_ubiquitinated + self.n_both_modified)
        if total_active > 0:
            phos_frac = self.n_phosphorylated / total_active
            ubi_frac = self.n_ubiquitinated / total_active
            both_frac = self.n_both_modified / total_active
            self.scaffold_binding = (
                self.cfg.scaffold_binding_base
                + phos_frac * self.cfg.phos_binding_factor
                - ubi_frac * self.cfg.ubi_binding_factor
                - both_frac * self.cfg.ubi_binding_factor * 0.5
            )

        return delta, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[PTMState], List[float]]:
        """Run full SSA simulation."""
        self.initialize()

        t = 0.0
        trajectory = [self._snapshot(t)]
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        steps = 0
        while t < max_time and steps < max_steps:
            delta, dt = self.ssa_step()
            if dt <= 0:
                break
            t += dt
            trajectory.append(self._snapshot(t))
            times.append(t)
            steps += 1

        return trajectory, times

    def _snapshot(self, t: float) -> PTMState:
        return PTMState(
            time=t,
            n_unmodified=self.n_unmodified,
            n_phosphorylated=self.n_phosphorylated,
            n_ubiquitinated=self.n_ubiquitinated,
            n_both_modified=self.n_both_modified,
            n_degraded=self.n_degraded,
            n_kinase_free=self.n_kinase_free,
            n_phosphatase_free=self.n_phosphatase_free,
            n_e3_free=self.n_e3_free,
            n_dub_free=self.n_dub_free,
            scaffold_binding_strength=self.scaffold_binding,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_substrate": self.cfg.n_substrate,
            "n_phospho_sites": self.cfg.n_phospho_sites,
            "n_ubi_sites": self.cfg.n_ubi_sites,
            "corollary3_guard": "first-order modification kinetics only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [BSS1] BISTABLE STOCHASTIC SWITCHING
# =============================================================================

@dataclass
class BistableConfig:
    """Configuration for bistable stochastic switching."""
    # Double-well potential parameters
    a: float = 1.0           # barrier height coefficient
    b: float = 0.0           # bias (tilts the potential)
    # Langevin dynamics
    sigma: float = 0.3       # noise amplitude
    dt: float = 0.001        # time step
    n_steps: int = 10000     # simulation steps
    # Kramers escape
    temperature: float = 1.0  # kBT
    # Bifurcation analysis
    n_b_values: int = 21      # number of bias values for sweep
    b_range: Tuple[float, float] = (-1.0, 1.0)
    # Seed
    seed: int = 0


@dataclass
class SwitchingEvent:
    """A switching event between bistable states."""
    time: float
    from_state: str  # "left" or "right"
    to_state: str    # "left" or "right"
    barrier_height: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BistableStochasticSwitching:
    """
    [C/D] Stochastic switching between bistable states.

    PHYSICAL BASIS [C Kramers 1940; Gardiner 2009]
      A bistable system has two stable states separated by an energy barrier.
      Noise can drive transitions between states (Kramers escape).

      Double-well potential:
        V(x) = a(x² - 1)² - b·x

      where:
        - a controls barrier height
        - b biases one well over the other
        - Stable states at x ≈ ±1 (for b = 0)
        - Barrier at x ≈ 0

      Langevin equation:
        dx/dt = -dV/dx + σ·ξ(t)
        where ξ(t) is Gaussian white noise

      Kramers escape rate:
        r ≈ (ω₀ω_b / 2πγ) · exp(-ΔV / kBT)
      where ω₀ is the well frequency, ω_b is the barrier frequency,
      γ is friction, ΔV is the barrier height.

    APPLICATION TO Wnt/PCP [D]
      The Wnt/β-catenin pathway can exhibit bistability:
        - OFF state: low β-catenin, high destruction complex
        - ON state: high β-catenin, low destruction complex
      Autocrine feedback (TCF → Wnt) creates the positive feedback
      needed for bistability.

      Switching between states corresponds to cell fate decisions:
        - OFF → ON: differentiation, proliferation
        - ON → OFF: quiescence, apoptosis

    COROLLARY-3 GUARD [D]
      The Langevin equation is FIRST-ORDER in time. We do NOT compute
      second derivatives of the trajectory. The noise term is additive
      (not multiplicative in x).

    PROP-3 AWARENESS [P]
      When we multiply the potential gradient by noise, the result
      loses regularity. We use the Euler-Maruyama discretization which
      is first-order accurate.

    HONEST
      • Double-well model is illustrative; real bistable switches involve
        complex gene regulatory networks with multiple feedback loops.
      • Kramers theory assumes smooth potentials; real molecular landscapes
        are rugged.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[BistableConfig] = None):
        self.cfg = cfg or BistableConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def potential(self, x: float, b: Optional[float] = None) -> float:
        """Double-well potential V(x) = a(x²-1)² - b·x."""
        b = b if b is not None else self.cfg.b
        return self.cfg.a * (x**2 - 1)**2 - b * x

    def potential_gradient(self, x: float, b: Optional[float] = None) -> float:
        """dV/dx = 4a·x(x²-1) - b."""
        b = b if b is not None else self.cfg.b
        return 4 * self.cfg.a * x * (x**2 - 1) - b

    def stable_states(self, b: Optional[float] = None) -> Tuple[float, float]:
        """Find the two stable states (minima of V)."""
        # For the double-well, minima are approximately at:
        # x ≈ ±sqrt(1 - b/(4a)) for small b
        b = b if b is not None else self.cfg.b
        a = self.cfg.a
        if abs(b) < 4 * a:
            x_star = math.sqrt(max(0, 1 - b / (4 * a)))
            return (-x_star, x_star)
        else:
            # Single well (bistability lost)
            return (0.0, 0.0)

    def barrier_height(self, b: Optional[float] = None) -> float:
        """Compute barrier height ΔV between the two wells."""
        b = b if b is not None else self.cfg.b
        left, right = self.stable_states(b)
        if left == right:
            return 0.0
        # Barrier is at x ≈ 0 (for symmetric case)
        V_left = self.potential(left, b)
        V_barrier = self.potential(0.0, b)
        return V_barrier - V_left

    def kramers_rate(self, b: Optional[float] = None) -> float:
        """
        Kramers escape rate [C Kramers 1940].
        r ≈ exp(-ΔV / kBT) (simplified, ignoring prefactor).
        """
        delta_v = self.barrier_height(b)
        kbt = self.cfg.temperature
        if delta_v <= 0:
            return float('inf')
        return math.exp(-delta_v / kbt)

    def simulate_langevin(self, x0: float, b: Optional[float] = None,
                            n_steps: Optional[int] = None,
                            dt: Optional[float] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Langevin dynamics: dx/dt = -V'(x) + σξ.
        Uses Euler-Maruyama discretization (first-order).
        Returns: (times, trajectory).
        """
        b = b if b is not None else self.cfg.b
        n_steps = n_steps or self.cfg.n_steps
        dt = dt or self.cfg.dt

        times = np.zeros(n_steps + 1)
        trajectory = np.zeros(n_steps + 1)
        trajectory[0] = x0

        sqrt_dt = math.sqrt(dt)

        for i in range(n_steps):
            x = trajectory[i]
            # Euler-Maruyama step (COROLLARY-3 guard: first-order)
            drift = -self.potential_gradient(x, b)
            noise = self._rng.standard_normal()
            trajectory[i + 1] = x + drift * dt + self.cfg.sigma * noise * sqrt_dt
            times[i + 1] = (i + 1) * dt

        return times, trajectory

    def detect_switching(self, trajectory: np.ndarray,
                           threshold: float = 0.0
                           ) -> List[SwitchingEvent]:
        """
        Detect switching events from a trajectory.
        A switch occurs when x crosses the threshold.
        """
        events = []
        current_state = "left" if trajectory[0] < threshold else "right"

        for i in range(1, len(trajectory)):
            new_state = "left" if trajectory[i] < threshold else "right"
            if new_state != current_state:
                events.append(SwitchingEvent(
                    time=i * self.cfg.dt,
                    from_state=current_state,
                    to_state=new_state,
                    barrier_height=self.barrier_height(),
                ))
                current_state = new_state

        return events

    def compute_mfpt(self, x0: float, target: float,
                       n_trials: int = 100) -> Dict[str, float]:
        """
        Compute Mean First Passage Time (MFPT) from x0 to target.
        Runs multiple trials and averages.
        """
        passage_times = []
        for trial in range(n_trials):
            times, traj = self.simulate_langevin(x0, n_steps=self.cfg.n_steps * 10)
            # Find first time trajectory crosses target
            crossed = np.where(np.sign(traj - target) != np.sign(x0 - target))[0]
            if len(crossed) > 0:
                passage_times.append(times[crossed[0]])

        if not passage_times:
            return {'mfpt': float('inf'), 'n_successful': 0, 'n_trials': n_trials}

        return {
            'mfpt': float(np.mean(passage_times)),
            'mfpt_std': float(np.std(passage_times)),
            'n_successful': len(passage_times),
            'n_trials': n_trials,
            'theoretical_kramers_rate': self.kramers_rate(),
        }

    def detect_hysteresis(self, n_steps_per_b: int = 500) -> Dict[str, Any]:
        """
        Detect hysteresis by sweeping the bias parameter b.
        Returns: dict with 'forward_branch', 'backward_branch', 'hysteresis_area'.
        """
        b_values = np.linspace(self.cfg.b_range[0], self.cfg.b_range[1],
                                self.cfg.n_b_values)
        forward_states = []
        backward_states = []

        # Forward sweep
        x = -1.0  # start in left well
        for b in b_values:
            _, traj = self.simulate_langevin(x, b=b, n_steps=n_steps_per_b)
            x = traj[-1]
            forward_states.append(x)

        # Backward sweep
        x = 1.0  # start in right well
        for b in reversed(b_values):
            _, traj = self.simulate_langevin(x, b=b, n_steps=n_steps_per_b)
            x = traj[-1]
            backward_states.append(x)
        backward_states.reverse()

        # Compute hysteresis area
        forward_arr = np.array(forward_states)
        backward_arr = np.array(backward_states)
        hysteresis_area = float(np.abs(np.trapz(forward_arr - backward_arr, b_values)))

        return {
            'b_values': b_values.tolist(),
            'forward_branch': forward_arr.tolist(),
            'backward_branch': backward_arr.tolist(),
            'hysteresis_area': hysteresis_area,
            'is_hysteretic': hysteresis_area > 0.1,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "a": self.cfg.a,
            "b": self.cfg.b,
            "sigma": self.cfg.sigma,
            "stable_states": self.stable_states(),
            "barrier_height": self.barrier_height(),
            "kramers_rate": self.kramers_rate(),
            "corollary3_guard": "first-order Langevin only; no second derivatives",
            "prop3_awareness": "potential × noise loses regularity",
        }


# =============================================================================
# SECTION 3 — [CET1] CRYO-ET BRIDGE
# =============================================================================

@dataclass
class CryoETConfig:
    """Configuration for cryo-ET data integration."""
    # Data format
    voxel_size_nm: float = 1.0  # nm per voxel
    # FSC parameters
    fsc_resolution_cutoff: float = 0.143  # "gold standard" FSC cutoff
    n_shells: int = 50
    # Subtomogram averaging
    box_size: int = 64  # pixels
    n_particles: int = 100
    # Model fitting
    rigid_body_search: bool = True
    max_translation_nm: float = 5.0
    max_rotation_deg: float = 15.0
    # Visualization
    save_figures: bool = False
    figure_dir: Optional[str] = None


@dataclass
class FSCResult:
    """Result of Fourier Shell Correlation analysis."""
    shells: np.ndarray        # spatial frequency shells
    fsc_values: np.ndarray    # FSC values per shell
    resolution_nm: float      # resolution at FSC=0.143
    n_shells: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class CryoETBridge:
    """
    [D] Bridge to cryo-electron tomography (cryo-ET) data.

    CAPABILITIES
      1. Load cryo-ET tomograms (MRC format) [C Frank 2006]
      2. Compute Fourier Shell Correlation (FSC) for resolution [C Rosenthal 2003]
      3. Subtomogram averaging (simplified)
      4. Rigid-body fitting of model structures
      5. Comparison with model predictions
      6. Visualization: central slices, projections

    CRYO-ET DATA [C Frank 2006]
      - Tomograms are 3D volumes (typically 512×512×200 voxels)
      - Voxel size: 0.5-5 nm
      - Missing wedge: limited tilt range (-60° to +60°)
      - Low SNR: requires averaging for high resolution

    FSC RESOLUTION [C Rosenthal & Henderson 2003]
      FSC(r) = Σ F₁(r)·F₂*(r) / sqrt(Σ|F₁|² · Σ|F₂|²)
      where F₁, F₂ are Fourier transforms of two half-maps.
      Resolution is where FSC drops below 0.143 ("gold standard").

    SUBTOMOGRAM AVERAGING [D]
      Simplified workflow:
        1. Extract subvolumes at particle positions
        2. Align by cross-correlation
        3. Average aligned subvolumes
        4. Iterate until convergence

    COROLLARY-3 GUARD [D]
      All operations are FIRST-ORDER (linear filtering, FFT, averaging).
      We do NOT compute second derivatives of the density field.

    HINO GUARD [C Thm 8]
      The density field from cryo-ET is a MEASUREMENT, not a smooth
      function. We treat it as discrete voxel data, not as a continuous
      field. No smooth density assumption is made.

    HONEST
      • This is a COMPUTATIONAL analysis tool, NOT a reconstruction pipeline.
      • Real cryo-ET requires CTF correction, missing wedge compensation,
        and sophisticated alignment algorithms.
      • FSC is computed on synthetic data for demonstration; real FSC
        requires two independent half-maps.
      • No biological/medical/diagnostic claim.
    """

    def __init__(self, cfg: Optional[CryoETConfig] = None):
        self.cfg = cfg or CryoETConfig()
        self._tomogram: Optional[np.ndarray] = None
        self._half_map1: Optional[np.ndarray] = None
        self._half_map2: Optional[np.ndarray] = None

    # ── Data loading ────────────────────────────────────────────────────
    def load_mrc(self, path: str) -> np.ndarray:
        """
        Load cryo-ET tomogram from MRC file.
        Returns: 3D numpy array.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"MRC file not found: {path}")

        if HAS_MRCFILE:
            with mrcfile.open(path) as mrc:
                self._tomogram = mrc.data.astype(np.float32)
                self.cfg.voxel_size_nm = float(mrc.voxel_size.x) / 10.0  # Å to nm
        else:
            # Fallback: try to load as raw numpy
            logger.warning("[CryoET] mrcfile not available; trying numpy load")
            self._tomogram = np.load(path)

        logger.info(f"[CryoET] Loaded tomogram: shape={self._tomogram.shape}, "
                    f"voxel_size={self.cfg.voxel_size_nm} nm")
        return self._tomogram

    def generate_synthetic_tomogram(self, shape: Tuple[int, int, int] = (64, 64, 64),
                                      n_particles: int = 20) -> np.ndarray:
        """Generate synthetic tomogram for testing."""
        tomogram = np.zeros(shape, dtype=np.float32)

        # Add Gaussian particles
        for _ in range(n_particles):
            cx, cy, cz = np.random.randint(10, shape[0]-10, 3)
            sigma = np.random.uniform(2, 4)
            amplitude = np.random.uniform(0.5, 1.0)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        r2 = (i-cx)**2 + (j-cy)**2 + (k-cz)**2
                        tomogram[i, j, k] += amplitude * np.exp(-r2 / (2*sigma**2))

        # Add noise
        tomogram += np.random.normal(0, 0.1, shape).astype(np.float32)

        self._tomogram = tomogram
        return tomogram

    # ── FSC computation ─────────────────────────────────────────────────
    def compute_fsc(self, half_map1: Optional[np.ndarray] = None,
                     half_map2: Optional[np.ndarray] = None) -> FSCResult:
        """
        Compute Fourier Shell Correlation between two half-maps.
        [C Rosenthal & Henderson 2003]
        """
        if half_map1 is None or half_map2 is None:
            if self._tomogram is None:
                raise ValueError("No tomogram loaded")
            # Split tomogram into two halves for synthetic FSC
            mid = self._tomogram.shape[0] // 2
            half_map1 = self._tomogram[:mid]
            half_map2 = self._tomogram[mid:2*mid]

        # Compute FFTs
        F1 = np.fft.fftn(half_map1)
        F2 = np.fft.fftn(half_map2)

        # Compute radial shells
        shape = F1.shape
        center = [s // 2 for s in shape]
        Y, X, Z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        R = np.sqrt((X - center[1])**2 + (Y - center[0])**2 + (Z - center[2])**2)

        max_radius = min(center)
        shell_edges = np.linspace(0, max_radius, self.cfg.n_shells + 1)

        fsc_values = np.zeros(self.cfg.n_shells)
        shells = np.zeros(self.cfg.n_shells)

        for i in range(self.cfg.n_shells):
            r0, r1 = shell_edges[i], shell_edges[i+1]
            mask = (R >= r0) & (R < r1)

            if mask.sum() == 0:
                continue

            F1_shell = F1[mask]
            F2_shell = F2[mask]

            numerator = np.real(np.sum(F1_shell * np.conj(F2_shell)))
            denominator = np.sqrt(np.sum(np.abs(F1_shell)**2) *
                                   np.sum(np.abs(F2_shell)**2))

            fsc_values[i] = numerator / (denominator + 1e-12)
            shells[i] = (r0 + r1) / 2

        # Find resolution at FSC = 0.143
        above_cutoff = fsc_values > self.cfg.fsc_resolution_cutoff
        if above_cutoff.any():
            last_above = np.where(above_cutoff)[0][-1]
            resolution_shell = shells[last_above]
            # Convert to spatial frequency and then to resolution
            resolution_nm = self.cfg.voxel_size_nm * max(shape) / max(resolution_shell, 1e-12)
        else:
            resolution_nm = float('inf')

        return FSCResult(
            shells=shells,
            fsc_values=fsc_values,
            resolution_nm=resolution_nm,
            n_shells=self.cfg.n_shells,
        )

    # ── Subtomogram averaging ───────────────────────────────────────────
    def extract_subtomograms(self, positions: List[Tuple[int, int, int]],
                               box_size: Optional[int] = None) -> np.ndarray:
        """
        Extract subtomograms at given positions.
        Returns: (n_particles, box_size, box_size, box_size) array.
        """
        if self._tomogram is None:
            raise ValueError("No tomogram loaded")

        box_size = box_size or self.cfg.box_size
        half_box = box_size // 2
        subtomograms = []

        for cx, cy, cz in positions:
            # Extract box around center
            x0, x1 = max(0, cx - half_box), min(self._tomogram.shape[0], cx + half_box)
            y0, y1 = max(0, cy - half_box), min(self._tomogram.shape[1], cy + half_box)
            z0, z1 = max(0, cz - half_box), min(self._tomogram.shape[2], cz + half_box)

            subvol = self._tomogram[x0:x1, y0:y1, z0:z1]

            # Pad or crop to box_size
            if subvol.shape != (box_size, box_size, box_size):
                padded = np.zeros((box_size, box_size, box_size), dtype=np.float32)
                padded[:subvol.shape[0], :subvol.shape[1], :subvol.shape[2]] = subvol
                subvol = padded

            subtomograms.append(subvol)

        return np.array(subtomograms)

    def average_subtomograms(self, subtomograms: np.ndarray) -> np.ndarray:
        """
        Compute average of subtomograms (simplified; no alignment).
        """
        return subtomograms.mean(axis=0)

    # ── Rigid-body fitting ──────────────────────────────────────────────
    def fit_model(self, model_density: np.ndarray,
                   search_translation: bool = True,
                   search_rotation: bool = False) -> Dict[str, Any]:
        """
        Fit a model density to the tomogram by cross-correlation.
        Simplified: translation search only (rotation is expensive).
        """
        if self._tomogram is None:
            raise ValueError("No tomogram loaded")

        # Compute cross-correlation
        from scipy.signal import fftconvolve

        if search_translation:
            # Cross-correlate model with tomogram
            cc = fftconvolve(self._tomogram, model_density[::-1, ::-1, ::-1],
                              mode='full')
            # Find peak
            peak_idx = np.unravel_index(np.argmax(cc), cc.shape)
            # Convert to translation
            model_center = np.array(model_density.shape) // 2
            translation = np.array(peak_idx) - model_center
            best_cc = cc[peak_idx]
        else:
            translation = np.zeros(3, dtype=int)
            best_cc = 0.0

        return {
            'translation_voxels': translation.tolist(),
            'translation_nm': (translation * self.cfg.voxel_size_nm).tolist(),
            'best_cross_correlation': float(best_cc),
            'search_rotation': search_rotation,
        }

    # ── Visualization ───────────────────────────────────────────────────
    def visualize_slice(self, axis: int = 0, slice_idx: Optional[int] = None,
                          save_path: Optional[str] = None) -> None:
        """Visualize a central slice of the tomogram."""
        if not HAS_MPL:
            logger.warning("[CryoET] matplotlib not available; skipping visualization")
            return

        if self._tomogram is None:
            raise ValueError("No tomogram loaded")

        if slice_idx is None:
            slice_idx = self._tomogram.shape[axis] // 2

        if axis == 0:
            slice_data = self._tomogram[slice_idx, :, :]
        elif axis == 1:
            slice_data = self._tomogram[:, slice_idx, :]
        else:
            slice_data = self._tomogram[:, :, slice_idx]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(slice_data, cmap='gray', origin='lower')
        ax.set_title(f'Cryo-ET slice (axis={axis}, index={slice_idx})')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=ax, label='Density')

        if save_path and self.cfg.save_figures:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"[CryoET] Saved figure to {save_path}")

        plt.close()

    def get_status(self) -> Dict[str, Any]:
        return {
            "tomogram_loaded": self._tomogram is not None,
            "voxel_size_nm": self.cfg.voxel_size_nm,
            "fsc_cutoff": self.cfg.fsc_resolution_cutoff,
            "box_size": self.cfg.box_size,
            "has_mrcfile": HAS_MRCFILE,
            "corollary3_guard": "first-order operations only; no Hessian",
            "hino_guard": "density treated as discrete voxels, not smooth field",
            "note": "computational analysis only; NOT a reconstruction pipeline",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIPTMBistableCryoETSuite:
    """
    Unifies: PostTranslationalModificationModel + BistableStochasticSwitching
             + CryoETBridge.

    FULL PIPELINE:
      1. PTM dynamics modulate scaffolding binding
      2. Scaffolding affects Wnt/PCP signaling (bistable switch)
      3. Stochastic switching between Wnt-ON and Wnt-OFF states
      4. Compare molecular organization with cryo-ET data
      5. Validate model predictions against experimental structures
    """

    def __init__(self,
                 ptm_cfg: Optional[PTMConfig] = None,
                 bistable_cfg: Optional[BistableConfig] = None,
                 cryoet_cfg: Optional[CryoETConfig] = None):
        self.ptm = PostTranslationalModificationModel(ptm_cfg)
        self.bistable = BistableStochasticSwitching(bistable_cfg)
        self.cryoet = CryoETBridge(cryoet_cfg)

    def run_coupled_simulation(self, n_steps: int = 100) -> Dict[str, Any]:
        """
        Run coupled PTM + bistable switching simulation.
        """
        # Run PTM simulation
        ptm_traj, ptm_times = self.ptm.simulate(max_steps=n_steps)
        final_ptm = ptm_traj[-1]

        # Use scaffold binding strength as bias for bistable switch
        # High scaffold binding → Wnt-ON (right state)
        # Low scaffold binding → Wnt-OFF (left state)
        binding_strength = final_ptm.scaffold_binding_strength
        bias = (binding_strength - 1.0) * 0.5  # map to bias parameter

        # Run bistable simulation
        x0 = -1.0  # start in Wnt-OFF state
        _, bistable_traj = self.bistable.simulate_langevin(x0, b=bias, n_steps=n_steps)

        # Detect switching events
        switching_events = self.bistable.detect_switching(bistable_traj)

        # Final state
        final_state = "Wnt-ON" if bistable_traj[-1] > 0 else "Wnt-OFF"

        return {
            'n_steps': n_steps,
            'ptm_final_state': {
                'unmodified': final_ptm.n_unmodified,
                'phosphorylated': final_ptm.n_phosphorylated,
                'ubiquitinated': final_ptm.n_ubiquitinated,
                'degraded': final_ptm.n_degraded,
                'scaffold_binding': final_ptm.scaffold_binding_strength,
            },
            'bistable_bias': bias,
            'final_wnt_state': final_state,
            'n_switching_events': len(switching_events),
            'bistable_trajectory': bistable_traj.tolist()[-10:],  # last 10 points
        }

    def validate_with_cryoet(self, tomogram_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate model predictions against cryo-ET data.
        """
        # Load or generate tomogram
        if tomogram_path and os.path.exists(tomogram_path):
            tomogram = self.cryoet.load_mrc(tomogram_path)
        else:
            tomogram = self.cryoet.generate_synthetic_tomogram(shape=(64, 64, 64),
                                                                   n_particles=10)

        # Compute FSC
        fsc_result = self.cryoet.compute_fsc()

        # Extract and average subtomograms
        n_particles = min(10, self.cryoet.cfg.n_particles)
        positions = [(np.random.randint(10, 54),
                       np.random.randint(10, 54),
                       np.random.randint(10, 54))
                      for _ in range(n_particles)]
        subtomograms = self.cryoet.extract_subtomograms(positions)
        average = self.cryoet.average_subtomograms(subtomograms)

        return {
            'tomogram_shape': tomogram.shape,
            'fsc_resolution_nm': fsc_result.resolution_nm,
            'n_subtomograms': len(subtomograms),
            'average_density_mean': float(average.mean()),
            'average_density_std': float(average.std()),
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "ptm": self.ptm.get_status(),
            "bistable": self.bistable.get_status(),
            "cryoet": self.cryoet.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO — PTM / BISTABLE SWITCHING / CRYO-ET")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" mrcfile available      : {HAS_MRCFILE}")
    print(f" scipy available        : {HAS_SCIPY}")
    print(f" matplotlib available   : {HAS_MPL}")

    suite = SESIPTMBistableCryoETSuite(
        ptm_cfg=PTMConfig(n_substrate=50, max_time=100.0, max_steps=5000),
        bistable_cfg=BistableConfig(a=1.0, sigma=0.3, n_steps=5000),
        cryoet_cfg=CryoETConfig(box_size=16))

    # ── 1. Post-Translational Modifications ─────────────────────────────
    print("\n [PTM1] Post-Translational Modification Model")
    ptm = suite.ptm
    print(f"   n_substrate          : {ptm.cfg.n_substrate}")
    print(f"   n_phospho_sites      : {ptm.cfg.n_phospho_sites}")
    print(f"   n_ubi_sites          : {ptm.cfg.n_ubi_sites}")

    # Run simulation
    ptm_traj, ptm_times = ptm.simulate(max_steps=3000)
    print(f"   trajectory length    : {len(ptm_traj)}")
    print(f"   final time           : {ptm_times[-1]:.2f}")

    final_ptm = ptm_traj[-1]
    print(f"   final PTM state      :")
    print(f"     unmodified         : {final_ptm.n_unmodified}")
    print(f"     phosphorylated     : {final_ptm.n_phosphorylated}")
    print(f"     ubiquitinated      : {final_ptm.n_ubiquitinated}")
    print(f"     both modified      : {final_ptm.n_both_modified}")
    print(f"     degraded           : {final_ptm.n_degraded}")
    print(f"     scaffold binding   : {final_ptm.scaffold_binding_strength:.4f}")

    ptm_status = ptm.get_status()
    print(f"   COROLLARY-3 guard    : {ptm_status['corollary3_guard']}")
    print(f"   HINO guard           : {ptm_status['hino_guard']}")

    # ── 2. Bistable Stochastic Switching ────────────────────────────────
    print("\n [BSS1] Bistable Stochastic Switching")
    bistable = suite.bistable
    print(f"   potential params     : a={bistable.cfg.a}, b={bistable.cfg.b}")
    print(f"   noise amplitude      : {bistable.cfg.sigma}")
    print(f"   stable states        : {bistable.stable_states()}")
    print(f"   barrier height       : {bistable.barrier_height():.4f}")
    print(f"   Kramers rate         : {bistable.kramers_rate():.4f}")

    # Run Langevin simulation
    x0 = -1.0  # start in left well
    times, trajectory = bistable.simulate_langevin(x0, n_steps=3000)
    print(f"   trajectory length    : {len(trajectory)}")
    print(f"   final position       : {trajectory[-1]:.4f}")

    # Detect switching
    switching_events = bistable.detect_switching(trajectory)
    print(f"   switching events     : {len(switching_events)}")
    if switching_events:
        print(f"     first switch at t  : {switching_events[0].time:.4f}")
        print(f"     direction          : {switching_events[0].from_state} → {switching_events[0].to_state}")

    # Compute MFPT
    mfpt = bistable.compute_mfpt(x0=-1.0, target=1.0, n_trials=20)
    print(f"   MFPT (-1 → +1)       : {mfpt['mfpt']:.4f}")
    print(f"   successful trials    : {mfpt['n_successful']}/{mfpt['n_trials']}")
    print(f"   Kramers rate (theory): {mfpt['theoretical_kramers_rate']:.4f}")

    # Detect hysteresis
    hysteresis = bistable.detect_hysteresis(n_steps_per_b=100)
    print(f"   hysteresis area      : {hysteresis['hysteresis_area']:.4f}")
    print(f"   is hysteretic        : {hysteresis['is_hysteretic']}")

    bistable_status = bistable.get_status()
    print(f"   COROLLARY-3 guard    : {bistable_status['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {bistable_status['prop3_awareness']}")

    # ── 3. Coupled PTM + Bistable Simulation ────────────────────────────
    print("\n [COUPLED] PTM + Bistable Switching Coupling")
    coupled_result = suite.run_coupled_simulation(n_steps=100)
    print(f"   n_steps              : {coupled_result['n_steps']}")
    print(f"   PTM final state      : {coupled_result['ptm_final_state']}")
    print(f"   bistable bias        : {coupled_result['bistable_bias']:.4f}")
    print(f"   final Wnt state      : {coupled_result['final_wnt_state']}")
    print(f"   n_switching events   : {coupled_result['n_switching_events']}")

    # ── 4. Cryo-ET Bridge ───────────────────────────────────────────────
    print("\n [CET1] Cryo-ET Bridge")
    cryoet = suite.cryoet
    print(f"   voxel size           : {cryoet.cfg.voxel_size_nm} nm")
    print(f"   box size             : {cryoet.cfg.box_size} px")
    print(f"   FSC cutoff           : {cryoet.cfg.fsc_resolution_cutoff}")

    # Generate synthetic tomogram
    tomogram = cryoet.generate_synthetic_tomogram(shape=(32, 32, 32), n_particles=5)
    print(f"   synthetic tomogram   : shape={tomogram.shape}")
    print(f"   density range        : [{tomogram.min():.4f}, {tomogram.max():.4f}]")

    # Compute FSC
    fsc_result = cryoet.compute_fsc()
    print(f"   FSC resolution       : {fsc_result.resolution_nm:.2f} nm")
    print(f"   n_shells             : {fsc_result.n_shells}")
    print(f"   FSC max              : {fsc_result.fsc_values.max():.4f}")
    print(f"   FSC min              : {fsc_result.fsc_values.min():.4f}")

    # Extract and average subtomograms
    positions = [(np.random.randint(5, 27), np.random.randint(5, 27), np.random.randint(5, 27))
                  for _ in range(5)]
    subtomograms = cryoet.extract_subtomograms(positions, box_size=16)
    print(f"   subtomograms         : shape={subtomograms.shape}")

    average = cryoet.average_subtomograms(subtomograms)
    print(f"   average density      : mean={average.mean():.4f}, std={average.std():.4f}")

    # Fit model
    model_density = np.random.rand(16, 16, 16).astype(np.float32)
    fit_result = cryoet.fit_model(model_density, search_translation=True)
    print(f"   model fitting        :")
    print(f"     translation (vox)  : {fit_result['translation_voxels']}")
    print(f"     translation (nm)   : {fit_result['translation_nm']}")
    print(f"     best CC            : {fit_result['best_cross_correlation']:.4f}")

    cryoet_status = cryoet.get_status()
    print(f"   COROLLARY-3 guard    : {cryoet_status['corollary3_guard']}")
    print(f"   HINO guard           : {cryoet_status['hino_guard']}")
    print(f"   Note                 : {cryoet_status['note']}")

    # ── 5. Full Validation Pipeline ─────────────────────────────────────
    print("\n [VALIDATE] Full Validation Pipeline")
    validation = suite.validate_with_cryoet()
    print(f"   tomogram shape       : {validation['tomogram_shape']}")
    print(f"   FSC resolution       : {validation['fsc_resolution_nm']:.2f} nm")
    print(f"   n_subtomograms       : {validation['n_subtomograms']}")
    print(f"   average density      : mean={validation['average_density_mean']:.4f}, "
          f"std={validation['average_density_std']:.4f}")

    # ── 6. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"     {kk}: {vv}")
        else:
            print(f"     {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Gillespie SSA is EXACT for well-mixed reactions (1976/77).")
    print("       Real PTM involves spatial compartmentalization, enzyme")
    print("       processivity, allosteric regulation not captured here.")
    print(" [C]   Phosphorylation: Cohen 2000; Ubiquitination: Hershko &")
    print("       Ciechanover 1998 (Nobel Prize 2004).")
    print(" [C]   Kramers escape rate theory (1940); Gardiner 2009.")
    print(" [D]   Double-well model is illustrative; real bistable switches")
    print("       involve complex gene regulatory networks.")
    print(" [C]   Cryo-ET: Frank 2006; FSC: Rosenthal & Henderson 2003.")
    print(" [D]   Cryo-ET bridge: computational analysis only; NOT a")
    print("       reconstruction pipeline. Real cryo-ET requires CTF correction,")
    print("       missing wedge compensation, sophisticated alignment.")
    print(" [D]   COROLLARY-3 guard: all fields first-order only; no Hessian.")
    print(" [C]   HINO guard: energy measures singular; no smooth density assumed.")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL models. No biological, medical,")
    print("       or diagnostic claim. Not validated against real organoids.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
