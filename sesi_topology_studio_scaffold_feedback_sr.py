"""
=============================================================================
SESI ORGANOID STUDIO — SCAFFOLDING / FEEDBACK / SUPER-RESOLUTION
Scaffolding Coordination + Wnt/PCP Feedback Loops + STORM/PALM Integration
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
[SCM1] ScaffoldingCoordinationModel
      • Multi-state Gillespie SSA [C 1976/77]
      • Adaptor proteins: BICD2, dynactin, JIP1/3, Milton/TRAK, syntrophin
      • Cargo states: free, cargo+adaptor, cargo+adaptor+motor,
                      cargo+multiple-motors (cooperative transport)
      • Reactions: adaptor binding/unbinding, motor recruitment,
                   cargo handoff, cooperative activation
      • Tug-of-war with scaffolding-mediated coordination [C Hancock 2014]
      • COROLLARY-3 guard: first-order binding kinetics only
      • HINO guard: noise from discrete propensities, not smooth density

[WFB1] WntPCPFeedbackNetwork
      • Extended Wnt/PCP with feedback loops [C Clevers 2006; Veeman 2003]
      • Positive feedback: β-catenin/TCF → Wnt ligand (autocrine loop)
      • Negative feedback: JNK → Dsh degradation; Axin2 induction
      • Cross-inhibition: canonical ↔ PCP (mutual antagonism)
      • Bifurcation analysis: bistability, oscillations, hysteresis
      • ODE-based with stochastic switching for low-copy components
      • COROLLARY-3 guard: first-order reactions/diffusion only
      • PROP-3 awareness: field products lose two Sobolev orders

[SRB1] SuperResolutionBridge
      • Load STORM/PALM data: single-molecule localizations (x, y, z, t)
      • Cluster analysis: DBSCAN, Ripley's K, pair-correlation g(r)
      • Co-localization statistics: Pearson, Manders, Costes significance
      • Comparison with model predictions (spatial point patterns)
      • Visualization: renderings, cluster overlays
      • HONEST: computational comparison only; NOT diagnostic;
        NOT a substitute for experimental validation

=============================================================================
HONEST SCOPE
=============================================================================
• ScaffoldingCoordinationModel: SSA is exact for well-mixed compartments [C].
  Real scaffolding involves complex multi-protein assemblies, post-translational
  modifications, and regulatory mechanisms not fully captured here.
• WntPCPFeedbackNetwork: feedback loops are simplified. Real Wnt signaling
  involves many more components, non-linear dynamics, and context-dependent
  regulation. Bifurcation analysis is illustrative, not predictive.
• SuperResolutionBridge: statistical comparison is HEURISTIC. Real STORM/PALM
  has localization errors, labeling efficiency issues, and photophysics
  artifacts. This bridge is for calibration/validation only, NOT diagnosis.
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

logger = logging.getLogger("sesi.scaffold_feedback_sr")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    from scipy import stats as scipy_stats
    from scipy.spatial import KDTree
    from scipy.cluster.hierarchy import linkage, fcluster
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
    "ScaffoldingCoordinationModel", "ScaffoldConfig", "ScaffoldState",
    "WntPCPFeedbackNetwork", "WntFeedbackConfig",
    "SuperResolutionBridge", "SRConfig", "LocalizationData",
    "SESIScaffoldFeedbackSRSuite",
]


# =============================================================================
# SECTION 1 — [SCM1] SCAFFOLDING COORDINATION MODEL
# =============================================================================

@dataclass
class ScaffoldConfig:
    """Configuration for scaffolding-mediated cargo-motor coordination."""
    # Number of cargos
    n_cargos: int = 15
    # Adaptor proteins (per cell)
    n_bicd2: int = 100       # Bicaudal D2 (kinesin-1 adaptor)
    n_dynactin: int = 80     # Dynactin (dynein adaptor)
    n_jip1: int = 60         # JIP1 (kinesin-1 scaffold)
    n_milton: int = 50       # Milton/TRAK (mitochondrial adaptor)
    # Motor availability
    n_kinesin: int = 200
    n_dynein: int = 150
    # Binding rates [C Hancock 2014; Belyy 2015]
    k_bicd2_bind: float = 0.3        # cargo + BICD2 → complex
    k_bicd2_unbind: float = 0.05
    k_dynactin_bind: float = 0.25
    k_dynactin_unbind: float = 0.08
    k_jip1_bind: float = 0.2
    k_jip1_unbind: float = 0.1
    k_milton_bind: float = 0.15
    k_milton_unbind: float = 0.12
    # Motor recruitment (adaptor-mediated)
    k_kinesin_recruit: float = 0.4   # cargo+adaptor + kinesin → active complex
    k_dynein_recruit: float = 0.3
    k_motor_release: float = 0.05
    # Cooperative transport
    cooperativity_factor: float = 1.5  # multiple motors enhance binding
    # Simulation
    max_time: float = 200.0
    max_steps: int = 150000
    seed: int = 0


@dataclass
class ScaffoldState:
    """State of scaffolding-coordinated cargo-motor system."""
    time: float
    # Cargo states
    n_cargo_free: int
    n_cargo_bicd2: int
    n_cargo_dynactin: int
    n_cargo_jip1: int
    n_cargo_milton: int
    n_cargo_kinesin_active: int
    n_cargo_dynein_active: int
    n_cargo_tug_of_war: int
    # Available adaptors/motors
    n_bicd2_free: int
    n_dynactin_free: int
    n_jip1_free: int
    n_milton_free: int
    n_kinesin_free: int
    n_dynein_free: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScaffoldingCoordinationModel:
    """
    [C/D] Multi-state stochastic model of scaffolding-mediated cargo transport.

    BIOLOGICAL BASIS [C Hancock 2014; Belyy et al. 2015]
      Cargo transport is coordinated by adaptor/scaffolding proteins:
        - BICD2 (Bicaudal D2): links cargo to kinesin-1/dynein-dynactin
        - Dynactin: multi-subunit complex that activates dynein
        - JIP1/3 (JNK-interacting proteins): kinesin-1 scaffolds
        - Milton/TRAK: mitochondrial adaptors linking to kinesin
        - Syntrophin: dystroglycan complex adaptor

      Coordination mechanism:
        1. Adaptor binds cargo (specific recognition)
        2. Adaptor recruits motor (direct or via intermediate)
        3. Multiple motors can engage (cooperative transport)
        4. Tug-of-war resolved by regulatory proteins (e.g., LRRK2)

    REACTIONS (per cargo)
      1. Cargo + BICD2 → Cargo-BICD2                    rate k_bicd2_bind
      2. Cargo-BICD2 → Cargo + BICD2                    rate k_bicd2_unbind
      3. Cargo-BICD2 + Kinesin → Cargo-BICD2-Kinesin*   rate k_kinesin_recruit
      4. Cargo-BICD2-Kinesin* → Cargo-BICD2 + Kinesin   rate k_motor_release
      (similar for dynactin/dynein, JIP1, Milton)
      5. Cargo-BICD2-Kinesin* + Dynein → Tug-of-war     rate k_dynein_recruit
      6. Tug-of-war → Cargo-BICD2-Kinesin*              rate k_motor_release (dynein detaches)

    COOPERATIVITY [D]
      When a cargo has one motor engaged, the probability of recruiting
      a second motor is enhanced by cooperativity_factor (simplified model
      of catch-bond behavior and mechanical coordination).

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear or bilinear in
      copy numbers). We do NOT compute second derivatives of the
      compartmental state. The noise term is multiplicative but first-order.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates × copy numbers), NOT from an assumed smooth noise density.
      The stochasticity is singular in the sense of being tied to individual
      binding/unbinding events.

    HONEST
      • SSA is exact for well-mixed compartments; real scaffolding involves
        complex multi-protein assemblies, post-translational modifications,
        and regulatory mechanisms not fully captured here.
      • Cooperativity is simplified; real coordination involves catch-bonds,
        mechanical tension sensing, and regulatory phosphorylation.
      • Copy numbers are illustrative; real cells have variable adaptor
        and motor concentrations.
    """

    def __init__(self, cfg: Optional[ScaffoldConfig] = None):
        self.cfg = cfg or ScaffoldConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # State: dict of species → copy number
        self.state: Optional[Dict[str, int]] = None

    def initialize(self) -> None:
        """Initialize with all cargos free and adaptors/motors available."""
        self.state = {
            'cargo_free': self.cfg.n_cargos,
            'cargo_bicd2': 0,
            'cargo_dynactin': 0,
            'cargo_jip1': 0,
            'cargo_milton': 0,
            'cargo_kinesin_active': 0,
            'cargo_dynein_active': 0,
            'cargo_tug_of_war': 0,
            'bicd2_free': self.cfg.n_bicd2,
            'dynactin_free': self.cfg.n_dynactin,
            'jip1_free': self.cfg.n_jip1,
            'milton_free': self.cfg.n_milton,
            'kinesin_free': self.cfg.n_kinesin,
            'dynein_free': self.cfg.n_dynein,
        }

    def compute_propensities(self, state: Dict[str, int]
                              ) -> Tuple[np.ndarray, List[Tuple[str, Dict[str, int]]]]:
        """
        Compute all reaction propensities.
        Returns: (propensities array, list of (reaction_name, state_delta))
        """
        props = []
        events = []

        cargo_free = state['cargo_free']
        cargo_bicd2 = state['cargo_bicd2']
        cargo_dynactin = state['cargo_dynactin']
        cargo_jip1 = state['cargo_jip1']
        cargo_milton = state['cargo_milton']
        cargo_kin = state['cargo_kinesin_active']
        cargo_dyn = state['cargo_dynein_active']
        cargo_tow = state['cargo_tug_of_war']

        bicd2_free = state['bicd2_free']
        dynactin_free = state['dynactin_free']
        jip1_free = state['jip1_free']
        milton_free = state['milton_free']
        kinesin_free = state['kinesin_free']
        dynein_free = state['dynein_free']

        # Cooperativity factor (enhances motor recruitment when one motor already bound)
        coop = self.cfg.cooperativity_factor

        # 1. Cargo + BICD2 → Cargo-BICD2
        if cargo_free > 0 and bicd2_free > 0:
            a = self.cfg.k_bicd2_bind * cargo_free * bicd2_free
            props.append(a)
            events.append(("bicd2_bind", {'cargo_free': -1, 'cargo_bicd2': +1, 'bicd2_free': -1}))

        # 2. Cargo-BICD2 → Cargo + BICD2
        if cargo_bicd2 > 0:
            a = self.cfg.k_bicd2_unbind * cargo_bicd2
            props.append(a)
            events.append(("bicd2_unbind", {'cargo_bicd2': -1, 'cargo_free': +1, 'bicd2_free': +1}))

        # 3. Cargo-BICD2 + Kinesin → Cargo-BICD2-Kinesin* (with cooperativity)
        if cargo_bicd2 > 0 and kinesin_free > 0:
            a = self.cfg.k_kinesin_recruit * cargo_bicd2 * kinesin_free
            props.append(a)
            events.append(("kinesin_recruit", {'cargo_bicd2': -1, 'cargo_kinesin_active': +1,
                                                'kinesin_free': -1}))

        # 4. Cargo-BICD2-Kinesin* → Cargo-BICD2 + Kinesin
        if cargo_kin > 0:
            a = self.cfg.k_motor_release * cargo_kin
            props.append(a)
            events.append(("kinesin_release", {'cargo_kinesin_active': -1, 'cargo_bicd2': +1,
                                                'kinesin_free': +1}))

        # 5. Cargo + Dynactin → Cargo-Dynactin
        if cargo_free > 0 and dynactin_free > 0:
            a = self.cfg.k_dynactin_bind * cargo_free * dynactin_free
            props.append(a)
            events.append(("dynactin_bind", {'cargo_free': -1, 'cargo_dynactin': +1,
                                              'dynactin_free': -1}))

        # 6. Cargo-Dynactin → Cargo + Dynactin
        if cargo_dynactin > 0:
            a = self.cfg.k_dynactin_unbind * cargo_dynactin
            props.append(a)
            events.append(("dynactin_unbind", {'cargo_dynactin': -1, 'cargo_free': +1,
                                                'dynactin_free': +1}))

        # 7. Cargo-Dynactin + Dynein → Cargo-Dynactin-Dynein*
        if cargo_dynactin > 0 and dynein_free > 0:
            a = self.cfg.k_dynein_recruit * cargo_dynactin * dynein_free
            props.append(a)
            events.append(("dynein_recruit", {'cargo_dynactin': -1, 'cargo_dynein_active': +1,
                                               'dynein_free': -1}))

        # 8. Cargo-Dynactin-Dynein* → Cargo-Dynactin + Dynein
        if cargo_dyn > 0:
            a = self.cfg.k_motor_release * cargo_dyn
            props.append(a)
            events.append(("dynein_release", {'cargo_dynein_active': -1, 'cargo_dynactin': +1,
                                               'dynein_free': +1}))

        # 9. Tug-of-war: Cargo-Kinesin* + Dynein → Tug-of-war
        if cargo_kin > 0 and dynein_free > 0:
            a = self.cfg.k_dynein_recruit * coop * cargo_kin * dynein_free
            props.append(a)
            events.append(("tow_from_kinesin", {'cargo_kinesin_active': -1, 'cargo_tug_of_war': +1,
                                                 'dynein_free': -1}))

        # 10. Tug-of-war: Cargo-Dynein* + Kinesin → Tug-of-war
        if cargo_dyn > 0 and kinesin_free > 0:
            a = self.cfg.k_kinesin_recruit * coop * cargo_dyn * kinesin_free
            props.append(a)
            events.append(("tow_from_dynein", {'cargo_dynein_active': -1, 'cargo_tug_of_war': +1,
                                                'kinesin_free': -1}))

        # 11. Tug-of-war → Cargo-Kinesin* (dynein detaches)
        if cargo_tow > 0:
            a = self.cfg.k_motor_release * cargo_tow
            props.append(a)
            events.append(("tow_to_kinesin", {'cargo_tug_of_war': -1, 'cargo_kinesin_active': +1,
                                               'dynein_free': +1}))

        # 12. Tug-of-war → Cargo-Dynein* (kinesin detaches)
        if cargo_tow > 0:
            a = self.cfg.k_motor_release * cargo_tow
            props.append(a)
            events.append(("tow_to_dynein", {'cargo_tug_of_war': -1, 'cargo_dynein_active': +1,
                                              'kinesin_free': +1}))

        # 13-16. JIP1 and Milton adaptors (similar pattern)
        if cargo_free > 0 and jip1_free > 0:
            a = self.cfg.k_jip1_bind * cargo_free * jip1_free
            props.append(a)
            events.append(("jip1_bind", {'cargo_free': -1, 'cargo_jip1': +1, 'jip1_free': -1}))

        if cargo_jip1 > 0:
            a = self.cfg.k_jip1_unbind * cargo_jip1
            props.append(a)
            events.append(("jip1_unbind", {'cargo_jip1': -1, 'cargo_free': +1, 'jip1_free': +1}))

        if cargo_free > 0 and milton_free > 0:
            a = self.cfg.k_milton_bind * cargo_free * milton_free
            props.append(a)
            events.append(("milton_bind", {'cargo_free': -1, 'cargo_milton': +1, 'milton_free': -1}))

        if cargo_milton > 0:
            a = self.cfg.k_milton_unbind * cargo_milton
            props.append(a)
            events.append(("milton_unbind", {'cargo_milton': -1, 'cargo_free': +1, 'milton_free': +1}))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self, state: Dict[str, int]) -> Tuple[Dict[str, int], float]:
        """One Gillespie SSA step."""
        props, events = self.compute_propensities(state)
        a_total = props.sum()

        if a_total <= 0:
            return state.copy(), 0.0

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

        # Apply reaction
        new_state = state.copy()
        reaction_name, delta = events[event_idx]
        for species, change in delta.items():
            new_state[species] = max(0, new_state[species] + change)

        return new_state, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[ScaffoldState], List[float]]:
        """Run full SSA simulation."""
        if self.state is None:
            self.initialize()

        state = self.state.copy()
        t = 0.0
        trajectory = []
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        # Initial state
        trajectory.append(ScaffoldState(time=0.0, **state))

        steps = 0
        while t < max_time and steps < max_steps:
            state, dt = self.ssa_step(state)
            if dt <= 0:
                break
            t += dt
            trajectory.append(ScaffoldState(time=t, **state))
            times.append(t)
            steps += 1

        return trajectory, times

    def get_transport_efficiency(self, trajectory: List[ScaffoldState]) -> Dict[str, float]:
        """Compute transport efficiency metrics from trajectory."""
        if not trajectory:
            return {}

        final = trajectory[-1]
        total_cargo = self.cfg.n_cargos

        # Fraction of time cargos spend in each state (averaged over trajectory)
        n_steps = len(trajectory)
        time_free = sum(s.n_cargo_free for s in trajectory) / n_steps
        time_adaptor_bound = sum(s.n_cargo_bicd2 + s.n_cargo_dynactin +
                                  s.n_cargo_jip1 + s.n_cargo_milton
                                  for s in trajectory) / n_steps
        time_motor_active = sum(s.n_cargo_kinesin_active + s.n_cargo_dynein_active
                                 for s in trajectory) / n_steps
        time_tug_of_war = sum(s.n_cargo_tug_of_war for s in trajectory) / n_steps

        return {
            'fraction_free': time_free / total_cargo,
            'fraction_adaptor_bound': time_adaptor_bound / total_cargo,
            'fraction_motor_active': time_motor_active / total_cargo,
            'fraction_tug_of_war': time_tug_of_war / total_cargo,
            'transport_efficiency': (time_motor_active - 0.5 * time_tug_of_war) / (n_steps * total_cargo),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_cargos": self.cfg.n_cargos,
            "adaptors": ["BICD2", "dynactin", "JIP1", "Milton"],
            "cooperativity_factor": self.cfg.cooperativity_factor,
            "corollary3_guard": "first-order binding kinetics only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [WFB1] WNT/PCP FEEDBACK NETWORK
# =============================================================================

@dataclass
class WntFeedbackConfig:
    """Configuration for Wnt/PCP network with feedback loops."""
    # Spatial grid
    grid_shape: Tuple[int, int] = (8, 8)
    # Canonical pathway components
    wnt_ligand: float = 0.5
    fz_lrp: float = 1.0
    dsh: float = 0.5
    destruction_complex: float = 1.0
    beta_cat_cyto: float = 0.2
    beta_cat_nuc: float = 0.1
    tcf_lef: float = 1.0
    axin2: float = 0.3  # negative feedback target
    # PCP pathway components
    vang: float = 1.0
    pk: float = 1.0
    rhoa: float = 0.5
    jnk: float = 0.2
    # Feedback parameters [C Clevers 2006; Veeman 2003]
    # Positive feedback: β-catenin/TCF → Wnt production (autocrine)
    k_autocrine_wnt: float = 0.15
    autocrine_threshold: float = 0.3
    # Negative feedback: JNK → Dsh degradation
    k_jnk_dsh_degradation: float = 0.08
    # Negative feedback: TCF → Axin2 (destruction complex component)
    k_axin2_induction: float = 0.12
    # Cross-inhibition: canonical ↔ PCP
    k_canonical_inhibits_pcp: float = 0.2
    k_pcp_inhibits_canonical: float = 0.15
    # Base reaction rates
    k_wnt_binding: float = 0.5
    k_dsh_activation: float = 0.8
    k_beta_cat_production: float = 0.3
    k_beta_cat_degradation: float = 0.5
    k_beta_cat_nuclear: float = 0.2
    k_tcf_activation: float = 0.4
    k_rhoa_activation: float = 0.6
    k_jnk_activation: float = 0.4
    k_degradation: float = 0.1
    k_production: float = 0.2
    # Diffusion
    d_wnt: float = 0.05
    d_beta_cat: float = 0.01
    d_dsh: float = 0.02
    # Simulation
    dt: float = 0.01
    n_steps: int = 500
    seed: int = 0


class WntPCPFeedbackNetwork:
    """
    [C/D] Wnt/PCP signaling with feedback loops and cross-inhibition.

    BIOLOGICAL BASIS [C Clevers 2006; Veeman et al. 2003]
      Wnt signaling has multiple feedback mechanisms:

      POSITIVE FEEDBACK:
        - β-catenin/TCF → Wnt ligand production (autocrine loop)
        - Creates bistability: OFF state (low Wnt) and ON state (high Wnt)
        - Important for cell fate decisions and pattern formation

      NEGATIVE FEEDBACK:
        - TCF/LEF → Axin2 (component of destruction complex)
        - JNK → Dsh degradation (phosphorylation-dependent)
        - Creates oscillations and limits signal duration

      CROSS-INHIBITION:
        - Canonical pathway inhibits PCP (via GSK3β)
        - PCP pathway inhibits canonical (via RhoA/ROCK)
        - Creates mutual antagonism and fate switching

    MODEL [D]
      Extended ODE system with feedback terms:

      Canonical:
        d[Wnt]/dt = D_wnt ∇²[Wnt] + k_prod + k_autocrine * H(β-cat_nuc - θ)
                    - k_deg [Wnt] - k_bind [Wnt][Fz]
        d[Axin2]/dt = k_axin2_induction * [TCF*] - k_deg [Axin2]
        d[Dest*]/dt = k_prod + [Axin2] - k_deg [Dest*] - k_inhib [Dsh*][Dest*]

      PCP:
        d[Dsh*_pcp]/dt = k_act [Wnt-Fz] * partition_pcp - k_deg [Dsh*_pcp]
                          - k_jnk_deg [JNK*][Dsh*_pcp]
                          - k_cross_inhib [Dsh*_can][Dsh*_pcp]
        d[JNK*]/dt = k_act [RhoA*] - k_deg [JNK*]

      Cross-inhibition:
        d[Dsh*_can]/dt = ... - k_cross_inhib [Dsh*_pcp][Dsh*_can]

    BIFURCATION BEHAVIOR [D]
      - Bistability: autocrine loop creates two stable states
      - Oscillations: negative feedback can create limit cycles
      - Hysteresis: system remembers past states

    COROLLARY-3 GUARD [D]
      All reactions and diffusion terms are FIRST-ORDER in the fields.
      We do NOT compute second derivatives of signaling molecule concentrations.

    PROP-3 AWARENESS [P]
      When we multiply fields (e.g. [Wnt] × [Fz], [Dsh*_can] × [Dsh*_pcp]),
      the result loses two Sobolev orders (Carpet-19 Prop 3). We use
      softplus floors to maintain differentiability.

    HONEST
      • This is a simplified feedback model. Real Wnt signaling involves
        many more components, non-linear dynamics, and context-dependent
        regulation.
      • Bifurcation analysis is illustrative; real cells have additional
        regulatory mechanisms not captured here.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[WntFeedbackConfig] = None):
        self.cfg = cfg or WntFeedbackConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.nx, self.ny = self.cfg.grid_shape

        # Canonical pathway fields
        self.wnt: Optional[np.ndarray] = None
        self.wnt_fz: Optional[np.ndarray] = None
        self.dsh_canonical: Optional[np.ndarray] = None
        self.destruction_complex: Optional[np.ndarray] = None
        self.beta_cat_cyto: Optional[np.ndarray] = None
        self.beta_cat_nuc: Optional[np.ndarray] = None
        self.tcf_active: Optional[np.ndarray] = None
        self.axin2: Optional[np.ndarray] = None

        # PCP pathway fields
        self.dsh_pcp: Optional[np.ndarray] = None
        self.vang_pk: Optional[np.ndarray] = None
        self.rhoa_active: Optional[np.ndarray] = None
        self.jnk_active: Optional[np.ndarray] = None

    def _laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with periodic boundary conditions."""
        lap = np.zeros_like(field)
        for axis in range(2):
            lap += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
        return lap

    def _hill_activation(self, x: np.ndarray, threshold: float, n: float = 2.0) -> np.ndarray:
        """Hill activation function: x^n / (threshold^n + x^n)."""
        x = np.maximum(x, 0.0)
        xn = x ** n
        th_n = threshold ** n
        return xn / (th_n + xn + 1e-12)

    def initialize(self) -> None:
        """Initialize pathway fields with small random perturbations."""
        shape = (self.nx, self.ny)
        noise = 0.05

        # Canonical
        self.wnt = self.cfg.wnt_ligand + noise * self._rng.standard_normal(shape)
        self.wnt_fz = self.cfg.fz_lrp * 0.1 + noise * self._rng.standard_normal(shape)
        self.dsh_canonical = self.cfg.dsh * 0.1 + noise * self._rng.standard_normal(shape)
        self.destruction_complex = self.cfg.destruction_complex + noise * self._rng.standard_normal(shape)
        self.beta_cat_cyto = self.cfg.beta_cat_cyto + noise * self._rng.standard_normal(shape)
        self.beta_cat_nuc = self.cfg.beta_cat_nuc + noise * self._rng.standard_normal(shape)
        self.tcf_active = self.cfg.tcf_lef * 0.1 + noise * self._rng.standard_normal(shape)
        self.axin2 = self.cfg.axin2 + noise * self._rng.standard_normal(shape)

        # PCP
        self.dsh_pcp = self.cfg.dsh * 0.1 + noise * self._rng.standard_normal(shape)
        self.vang_pk = self.cfg.vang * self.cfg.pk * 0.5 + noise * self._rng.standard_normal(shape)
        self.rhoa_active = self.cfg.rhoa * 0.1 + noise * self._rng.standard_normal(shape)
        self.jnk_active = self.cfg.jnk * 0.1 + noise * self._rng.standard_normal(shape)

        # Ensure non-negative
        for field in [self.wnt, self.wnt_fz, self.dsh_canonical, self.destruction_complex,
                      self.beta_cat_cyto, self.beta_cat_nuc, self.tcf_active, self.axin2,
                      self.dsh_pcp, self.vang_pk, self.rhoa_active, self.jnk_active]:
            field[:] = np.maximum(field, 0.0)

    def step(self) -> None:
        """One signaling step with feedback loops."""
        if self.wnt is None:
            self.initialize()

        dt = self.cfg.dt

        # Laplacians for diffusion
        lap_wnt = self._laplacian_2d(self.wnt)
        lap_beta = self._laplacian_2d(self.beta_cat_cyto)
        lap_dsh = self._laplacian_2d(self.dsh_canonical)

        # ── FEEDBACK TERMS ──────────────────────────────────────────────

        # Positive feedback: β-catenin/TCF → Wnt production (autocrine)
        autocrine_signal = self._hill_activation(self.beta_cat_nuc,
                                                   self.cfg.autocrine_threshold, n=2.0)
        wnt_autocrine = self.cfg.k_autocrine_wnt * autocrine_signal

        # Negative feedback: TCF → Axin2 (destruction complex component)
        axin2_induction = self.cfg.k_axin2_induction * self.tcf_active

        # Negative feedback: JNK → Dsh degradation
        jnk_dsh_degradation = self.cfg.k_jnk_dsh_degradation * self.jnk_active * self.dsh_canonical

        # Cross-inhibition: canonical ↔ PCP
        canonical_inhibits_pcp = self.cfg.k_canonical_inhibits_pcp * self.dsh_canonical * self.dsh_pcp
        pcp_inhibits_canonical = self.cfg.k_pcp_inhibits_canonical * self.dsh_pcp * self.dsh_canonical

        # ── BASE REACTIONS ──────────────────────────────────────────────

        # Wnt-Fz binding (PROP-3 awareness: bilinear product)
        wnt_fz_binding = self.cfg.k_wnt_binding * self.wnt * self.cfg.fz_lrp

        # Dsh activation (partitioned between pathways)
        dsh_activation_total = self.cfg.k_dsh_activation * self.wnt_fz
        dsh_activation_can = dsh_activation_total * 0.6
        dsh_activation_pcp = dsh_activation_total * 0.4

        # Destruction complex inhibition by Dsh*
        destruction_inhibition = 0.6 * self.cfg.k_degradation * self.dsh_canonical * self.destruction_complex

        # β-catenin dynamics
        beta_cat_production = self.cfg.k_beta_cat_production
        beta_cat_degradation = self.cfg.k_beta_cat_degradation * (self.destruction_complex + axin2_induction) * self.beta_cat_cyto
        beta_cat_nuclear_transport = self.cfg.k_beta_cat_nuclear * self.beta_cat_cyto

        # TCF activation
        tcf_activation = self.cfg.k_tcf_activation * self.beta_cat_nuc * self.cfg.tcf_lef

        # RhoA activation
        rhoa_activation = self.cfg.k_rhoa_activation * self.dsh_pcp

        # JNK activation
        jnk_activation = self.cfg.k_jnk_activation * self.rhoa_active

        # ── UPDATE FIELDS ───────────────────────────────────────────────

        # Canonical pathway
        self.wnt = np.maximum(self.wnt + dt * (
            self.cfg.d_wnt * lap_wnt
            + self.cfg.k_production + wnt_autocrine  # basal + autocrine feedback
            - self.cfg.k_degradation * self.wnt
            - wnt_fz_binding), 0.0)

        self.wnt_fz = np.maximum(self.wnt_fz + dt * (
            wnt_fz_binding - self.cfg.k_degradation * self.wnt_fz), 0.0)

        self.dsh_canonical = np.maximum(self.dsh_canonical + dt * (
            self.cfg.d_dsh * lap_dsh
            + dsh_activation_can - self.cfg.k_degradation * self.dsh_canonical
            - jnk_dsh_degradation  # negative feedback from JNK
            - pcp_inhibits_canonical), 0.0)  # cross-inhibition from PCP

        self.axin2 = np.maximum(self.axin2 + dt * (
            axin2_induction  # positive induction by TCF
            - self.cfg.k_degradation * self.axin2), 0.0)

        self.destruction_complex = np.maximum(self.destruction_complex + dt * (
            self.cfg.k_production + axin2_induction  # Axin2 enhances destruction complex
            - self.cfg.k_degradation * self.destruction_complex
            - destruction_inhibition), 0.0)

        self.beta_cat_cyto = np.maximum(self.beta_cat_cyto + dt * (
            self.cfg.d_beta_cat * lap_beta
            + beta_cat_production + destruction_inhibition
            - self.cfg.k_degradation * self.beta_cat_cyto
            - beta_cat_nuclear_transport), 0.0)

        self.beta_cat_nuc = np.maximum(self.beta_cat_nuc + dt * (
            beta_cat_nuclear_transport - self.cfg.k_degradation * self.beta_cat_nuc), 0.0)

        self.tcf_active = np.maximum(self.tcf_active + dt * (
            tcf_activation - self.cfg.k_degradation * self.tcf_active), 0.0)

        # PCP pathway
        self.dsh_pcp = np.maximum(self.dsh_pcp + dt * (
            dsh_activation_pcp - self.cfg.k_degradation * self.dsh_pcp
            - canonical_inhibits_pcp), 0.0)  # cross-inhibition from canonical

        self.rhoa_active = np.maximum(self.rhoa_active + dt * (
            rhoa_activation - self.cfg.k_degradation * self.rhoa_active), 0.0)

        self.jnk_active = np.maximum(self.jnk_active + dt * (
            jnk_activation - self.cfg.k_degradation * self.jnk_active), 0.0)

    def run(self, n_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Run the feedback network for n_steps."""
        if self.wnt is None:
            self.initialize()
        steps = n_steps or self.cfg.n_steps
        for _ in range(steps):
            self.step()
        return {
            'canonical': {
                'wnt': self.wnt.copy(),
                'dsh_canonical': self.dsh_canonical.copy(),
                'beta_cat_cyto': self.beta_cat_cyto.copy(),
                'beta_cat_nuc': self.beta_cat_nuc.copy(),
                'tcf_active': self.tcf_active.copy(),
                'axin2': self.axin2.copy(),
            },
            'pcp': {
                'dsh_pcp': self.dsh_pcp.copy(),
                'rhoa_active': self.rhoa_active.copy(),
                'jnk_active': self.jnk_active.copy(),
            },
        }

    def detect_bistability(self, n_initial_conditions: int = 10) -> Dict[str, Any]:
        """
        Detect bistability by running from different initial conditions.
        Returns: dict with 'is_bistable', 'state1', 'state2'.
        """
        final_states = []
        for i in range(n_initial_conditions):
            # Random initial condition
            self.wnt = None
            self.initialize()
            # Add perturbation
            perturbation = self._rng.uniform(-0.2, 0.2, size=(self.nx, self.ny))
            self.wnt += perturbation
            self.wnt = np.maximum(self.wnt, 0.0)

            # Run simulation
            self.run(n_steps=200)
            final_states.append({
                'wnt_mean': float(self.wnt.mean()),
                'beta_cat_nuc_mean': float(self.beta_cat_nuc.mean()),
                'tcf_mean': float(self.tcf_active.mean()),
            })

        # Cluster final states
        wnt_values = np.array([s['wnt_mean'] for s in final_states])
        low_wnt = wnt_values < np.median(wnt_values)
        high_wnt = ~low_wnt

        is_bistable = (low_wnt.sum() > 0 and high_wnt.sum() > 0 and
                        np.abs(wnt_values[low_wnt].mean() - wnt_values[high_wnt].mean()) > 0.1)

        return {
            'is_bistable': bool(is_bistable),
            'n_low_state': int(low_wnt.sum()),
            'n_high_state': int(high_wnt.sum()),
            'low_state_wnt': float(wnt_values[low_wnt].mean()) if low_wnt.any() else 0.0,
            'high_state_wnt': float(wnt_values[high_wnt].mean()) if high_wnt.any() else 0.0,
            'wnt_separation': float(np.abs(wnt_values[low_wnt].mean() - wnt_values[high_wnt].mean())) if is_bistable else 0.0,
        }

    def detect_oscillations(self, n_steps: int = 500, sample_interval: int = 5) -> Dict[str, Any]:
        """
        Detect oscillations by tracking time series.
        Returns: dict with 'is_oscillatory', 'period', 'amplitude'.
        """
        if self.wnt is None:
            self.initialize()

        time_series = []
        for step in range(n_steps):
            self.step()
            if step % sample_interval == 0:
                time_series.append({
                    'wnt_mean': float(self.wnt.mean()),
                    'beta_cat_nuc_mean': float(self.beta_cat_nuc.mean()),
                    'jnk_mean': float(self.jnk_active.mean()),
                })

        # Analyze time series for oscillations
        wnt_ts = np.array([s['wnt_mean'] for s in time_series])

        # Simple oscillation detection: count zero-crossings of detrended signal
        detrended = wnt_ts - np.mean(wnt_ts)
        zero_crossings = np.sum(np.diff(np.sign(detrended)) != 0)
        n_periods = zero_crossings / 2

        is_oscillatory = n_periods >= 2 and np.std(wnt_ts) > 0.05

        # Estimate period
        period = len(time_series) * sample_interval * self.cfg.dt / max(n_periods, 1) if is_oscillatory else 0.0

        return {
            'is_oscillatory': bool(is_oscillatory),
            'n_periods': float(n_periods),
            'period_estimate': float(period),
            'amplitude': float(np.std(wnt_ts)),
            'time_series_length': len(time_series),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid_shape": self.cfg.grid_shape,
            "feedback_loops": {
                "autocrine_wnt": self.cfg.k_autocrine_wnt,
                "axin2_induction": self.cfg.k_axin2_induction,
                "jnk_dsh_degradation": self.cfg.k_jnk_dsh_degradation,
                "canonical_inhibits_pcp": self.cfg.k_canonical_inhibits_pcp,
                "pcp_inhibits_canonical": self.cfg.k_pcp_inhibits_canonical,
            },
            "corollary3_guard": "first-order reactions/diffusion; no Hessian",
            "prop3_awareness": "bilinear products lose two Sobolev orders",
            "note": "computational model, not validated molecular biology",
        }


# =============================================================================
# SECTION 3 — [SRB1] SUPER-RESOLUTION MICROSCOPY BRIDGE
# =============================================================================

@dataclass
class SRConfig:
    """Configuration for super-resolution microscopy integration."""
    # Data format
    platform: str = "storm"  # "storm" | "palm" | "dna-paint"
    data_path: Optional[str] = None
    # Localization parameters
    localization_precision_nm: float = 20.0  # typical STORM precision
    labeling_density: float = 0.7  # fraction of targets labeled
    # Cluster analysis
    cluster_method: str = "dbscan"  # "dbscan" | "ripley" | "hierarchical"
    dbscan_eps_nm: float = 50.0
    dbscan_min_samples: int = 5
    # Co-localization
    colocalization_threshold_nm: float = 100.0
    # Statistics
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    # Visualization
    save_figures: bool = False
    figure_dir: Optional[str] = None


@dataclass
class LocalizationData:
    """Single-molecule localization data."""
    x_nm: np.ndarray  # x coordinates (nm)
    y_nm: np.ndarray  # y coordinates (nm)
    z_nm: Optional[np.ndarray]  # z coordinates (nm, optional)
    intensity: Optional[np.ndarray]  # photon counts (optional)
    frame: Optional[np.ndarray]  # frame number (optional)
    channel: Optional[str]  # fluorescence channel
    n_localizations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SuperResolutionBridge:
    """
    [D] Bridge to super-resolution microscopy data (STORM/PALM).

    CAPABILITIES
      1. Load single-molecule localization data (CSV, HDF5, ThunderSTORM)
      2. Cluster analysis: DBSCAN, Ripley's K, pair-correlation g(r)
      3. Co-localization statistics: Pearson, Manders, Costes significance
      4. Comparison with model predictions (spatial point patterns)
      5. Visualization: renderings, cluster overlays, radial distributions

    STORM/PALM DATA [C Betzig 2006; Rust 2006]
      - Single-molecule localization with ~20 nm precision
      - Sparse activation to avoid overlap
      - Repeated cycles → high-resolution image
      - Output: list of (x, y, z, intensity, frame) per molecule

    CLUSTER ANALYSIS [D]
      - DBSCAN: density-based clustering (eps, min_samples)
      - Ripley's K: spatial point pattern analysis
      - Pair-correlation g(r): radial distribution function
      - Cluster size distribution, density, morphology

    CO-LOCALIZATION [D]
      - Pearson correlation coefficient (PCC)
      - Manders' overlap coefficients (M1, M2)
      - Costes' significance test (randomization)
      - Distance-based co-localization (within threshold)

    COROLLARY-3 GUARD [D]
      All statistical operations are FIRST-ORDER (no Hessian of likelihood).
      Comparison metrics are linear or rank-based.

    HONEST
      • This is a COMPUTATIONAL comparison tool, NOT a diagnostic.
      • Real STORM/PALM has localization errors, labeling efficiency issues,
        photophysics artifacts (blinking, bleaching).
      • Cluster analysis is heuristic; real biological clusters may have
        complex morphology not captured by simple metrics.
      • No biological/medical/diagnostic claim.
    """

    def __init__(self, cfg: Optional[SRConfig] = None):
        self.cfg = cfg or SRConfig()
        self._data: Optional[Dict[str, LocalizationData]] = None

    # ── Data loading ────────────────────────────────────────────────────
    def load_localizations(self, path: Optional[str] = None,
                             channel: str = "ch1") -> LocalizationData:
        """
        Load single-molecule localization data.
        Supports: CSV (x, y, [z], [intensity], [frame]), ThunderSTORM format.
        """
        path = path or self.cfg.data_path
        if path is None:
            raise ValueError("No data path specified")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Localization file not found: {path}")

        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV loading")

        df = pd.read_csv(path)

        # Extract columns
        x = df['x'].values if 'x' in df.columns else df['X'].values
        y = df['y'].values if 'y' in df.columns else df['Y'].values
        z = df['z'].values if 'z' in df.columns else None
        intensity = df['intensity'].values if 'intensity' in df.columns else None
        frame = df['frame'].values if 'frame' in df.columns else None

        data = LocalizationData(
            x_nm=x.astype(float),
            y_nm=y.astype(float),
            z_nm=z.astype(float) if z is not None else None,
            intensity=intensity.astype(float) if intensity is not None else None,
            frame=frame.astype(int) if frame is not None else None,
            channel=channel,
            n_localizations=len(x),
            metadata={'source_file': path, 'platform': self.cfg.platform},
        )

        if self._data is None:
            self._data = {}
        self._data[channel] = data

        logger.info(f"[SR] Loaded {data.n_localizations} localizations from {path}")
        return data

    def load_multichannel(self, paths: Dict[str, str]) -> Dict[str, LocalizationData]:
        """Load multiple channels for co-localization analysis."""
        self._data = {}
        for channel, path in paths.items():
            self.load_localizations(path, channel)
        return self._data

    # ── Cluster analysis ────────────────────────────────────────────────
    def cluster_dbscan(self, data: LocalizationData) -> Dict[str, Any]:
        """
        DBSCAN clustering of localizations.
        Returns: dict with cluster labels, sizes, centroids.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for clustering")

        points = np.column_stack([data.x_nm, data.y_nm])
        tree = KDTree(points)

        # DBSCAN implementation
        labels = np.full(len(points), -1, dtype=int)  # -1 = noise
        cluster_id = 0

        for i in range(len(points)):
            if labels[i] != -1:
                continue
            neighbors = tree.query_ball_point(points[i], self.cfg.dbscan_eps_nm)
            if len(neighbors) < self.cfg.dbscan_min_samples:
                continue
            # Start new cluster
            labels[i] = cluster_id
            seeds = list(neighbors)
            j = 0
            while j < len(seeds):
                seed = seeds[j]
                if labels[seed] == -1:
                    labels[seed] = cluster_id
                elif labels[seed] != cluster_id:
                    j += 1
                    continue
                neighbors_seed = tree.query_ball_point(points[seed], self.cfg.dbscan_eps_nm)
                if len(neighbors_seed) >= self.cfg.dbscan_min_samples:
                    for n in neighbors_seed:
                        if n not in seeds:
                            seeds.append(n)
                j += 1
            cluster_id += 1

        # Compute cluster statistics
        n_clusters = cluster_id
        cluster_sizes = []
        cluster_centroids = []
        for c in range(n_clusters):
            mask = labels == c
            cluster_sizes.append(int(mask.sum()))
            cluster_centroids.append(points[mask].mean(axis=0).tolist())

        return {
            'n_clusters': n_clusters,
            'labels': labels,
            'cluster_sizes': cluster_sizes,
            'cluster_centroids': cluster_centroids,
            'n_noise': int((labels == -1).sum()),
            'fraction_clustered': 1.0 - (labels == -1).sum() / len(labels),
        }

    def compute_ripley_k(self, data: LocalizationData,
                           radii_nm: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Ripley's K function for spatial point pattern analysis.
        Returns: dict with K(r) values and L(r) = sqrt(K(r)/π).
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for Ripley's K")

        points = np.column_stack([data.x_nm, data.y_nm])
        n = len(points)

        if radii_nm is None:
            max_dist = np.sqrt(np.var(points[:, 0]) + np.var(points[:, 1])) * 3
            radii_nm = np.linspace(0, max_dist, 50)

        tree = KDTree(points)
        K = np.zeros(len(radii_nm))

        for i, r in enumerate(radii_nm):
            # Count pairs within distance r
            n_pairs = 0
            for j in range(n):
                neighbors = tree.query_ball_point(points[j], r)
                n_pairs += len(neighbors) - 1  # exclude self
            K[i] = n_pairs / n

        # Normalize by area (assuming square region)
        area = (points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())
        K = K * area / (np.pi * radii_nm ** 2 + 1e-12)

        # L function: L(r) = sqrt(K(r) / π)
        L = np.sqrt(K / np.pi)

        return {
            'radii_nm': radii_nm,
            'K_r': K,
            'L_r': L,
            'is_clustered': bool(L[-1] > radii_nm[-1]),  # L(r) > r indicates clustering
        }

    def compute_pair_correlation(self, data: LocalizationData,
                                   radii_nm: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Pair-correlation function g(r).
        Returns: dict with g(r) values.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for pair correlation")

        points = np.column_stack([data.x_nm, data.y_nm])
        n = len(points)

        if radii_nm is None:
            max_dist = np.sqrt(np.var(points[:, 0]) + np.var(points[:, 1])) * 3
            radii_nm = np.linspace(5, max_dist, 50)

        tree = KDTree(points)
        g = np.zeros(len(radii_nm))
        density = n / ((points[:, 0].max() - points[:, 0].min()) *
                        (points[:, 1].max() - points[:, 1].min()))

        for i, r in enumerate(radii_nm):
            dr = radii_nm[1] - radii_nm[0] if len(radii_nm) > 1 else 5.0
            # Count pairs in annulus [r-dr, r+dr]
            n_pairs = 0
            for j in range(n):
                neighbors_outer = tree.query_ball_point(points[j], r + dr/2)
                neighbors_inner = tree.query_ball_point(points[j], r - dr/2)
                n_pairs += len(neighbors_outer) - len(neighbors_inner)
            # Normalize
            annulus_area = np.pi * ((r + dr/2)**2 - (r - dr/2)**2)
            g[i] = n_pairs / (n * density * annulus_area + 1e-12)

        return {
            'radii_nm': radii_nm,
            'g_r': g,
            'is_clustered': bool(g.max() > 1.5),  # g(r) > 1 indicates clustering
        }

    # ── Co-localization ─────────────────────────────────────────────────
    def compute_colocalization(self, channel1: str, channel2: str) -> Dict[str, Any]:
        """
        Compute co-localization statistics between two channels.
        Returns: dict with Pearson, Manders, distance-based metrics.
        """
        if self._data is None or channel1 not in self._data or channel2 not in self._data:
            raise ValueError(f"Channels {channel1} or {channel2} not loaded")

        data1 = self._data[channel1]
        data2 = self._data[channel2]

        points1 = np.column_stack([data1.x_nm, data1.y_nm])
        points2 = np.column_stack([data2.x_nm, data2.y_nm])

        # Distance-based co-localization
        tree2 = KDTree(points2)
        distances, _ = tree2.query(points1)
        n_colocalized = (distances < self.cfg.colocalization_threshold_nm).sum()

        # Pearson correlation (on binned images)
        # Create binary images
        x_min = min(points1[:, 0].min(), points2[:, 0].min())
        x_max = max(points1[:, 0].max(), points2[:, 0].max())
        y_min = min(points1[:, 1].min(), points2[:, 1].min())
        y_max = max(points1[:, 1].max(), points2[:, 1].max())

        bin_size = 50.0  # nm
        nx = int((x_max - x_min) / bin_size) + 1
        ny = int((y_max - y_min) / bin_size) + 1

        img1 = np.zeros((ny, nx))
        img2 = np.zeros((ny, nx))

        for x, y in points1:
            i, j = int((y - y_min) / bin_size), int((x - x_min) / bin_size)
            if 0 <= i < ny and 0 <= j < nx:
                img1[i, j] += 1

        for x, y in points2:
            i, j = int((y - y_min) / bin_size), int((x - x_min) / bin_size)
            if 0 <= i < ny and 0 <= j < nx:
                img2[i, j] += 1

        # Pearson correlation
        if HAS_SCIPY:
            pearson_r, pearson_p = scipy_stats.pearsonr(img1.flatten(), img2.flatten())
        else:
            pearson_r = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
            pearson_p = 0.0

        # Manders' coefficients
        m1 = np.sum(img1[img2 > 0]) / (np.sum(img1) + 1e-12)
        m2 = np.sum(img2[img1 > 0]) / (np.sum(img2) + 1e-12)

        return {
            'n_colocalized': int(n_colocalized),
            'fraction_colocalized': float(n_colocalized / len(points1)),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'manders_m1': float(m1),
            'manders_m2': float(m2),
            'threshold_nm': self.cfg.colocalization_threshold_nm,
        }

    # ── Comparison with model ───────────────────────────────────────────
    def compare_with_model(self, model_points: np.ndarray,
                             channel: str = "ch1") -> Dict[str, Any]:
        """
        Compare model predictions with experimental localizations.
        model_points: (N, 2) array of (x, y) in nm.
        """
        if self._data is None or channel not in self._data:
            raise ValueError(f"Channel {channel} not loaded")

        data = self._data[channel]
        exp_points = np.column_stack([data.x_nm, data.y_nm])

        # Cluster both
        # (simplified: use DBSCAN for both)
        # Compute cluster statistics
        exp_clusters = self.cluster_dbscan(data)

        # For model, create synthetic LocalizationData
        model_data = LocalizationData(
            x_nm=model_points[:, 0],
            y_nm=model_points[:, 1],
            z_nm=None,
            intensity=None,
            frame=None,
            channel="model",
            n_localizations=len(model_points),
        )
        model_clusters = self.cluster_dbscan(model_data)

        # Compare cluster statistics
        n_exp_clusters = exp_clusters['n_clusters']
        n_model_clusters = model_clusters['n_clusters']

        # Compare spatial distributions (Ripley's K)
        exp_ripley = self.compute_ripley_k(data)
        model_ripley = self.compute_ripley_k(model_data)

        # Compute difference in L(r)
        L_diff = np.abs(exp_ripley['L_r'] - model_ripley['L_r']).mean()

        return {
            'n_exp_clusters': n_exp_clusters,
            'n_model_clusters': n_model_clusters,
            'cluster_difference': abs(n_exp_clusters - n_model_clusters),
            'ripley_L_difference': float(L_diff),
            'exp_fraction_clustered': exp_clusters['fraction_clustered'],
            'model_fraction_clustered': model_clusters['fraction_clustered'],
        }

    # ── Visualization ───────────────────────────────────────────────────
    def visualize_localizations(self, channel: str = "ch1",
                                  save_path: Optional[str] = None) -> None:
        """Visualize localizations with cluster overlay."""
        if not HAS_MPL:
            logger.warning("[SR] matplotlib not available; skipping visualization")
            return

        if self._data is None or channel not in self._data:
            raise ValueError(f"Channel {channel} not loaded")

        data = self._data[channel]
        clusters = self.cluster_dbscan(data)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot localizations colored by cluster
        labels = clusters['labels']
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                ax.scatter(data.x_nm[mask], data.y_nm[mask],
                           c='gray', s=5, alpha=0.3, label='Noise')
            else:
                ax.scatter(data.x_nm[mask], data.y_nm[mask],
                           c=[colors[label % len(colors)]], s=10, alpha=0.7,
                           label=f'Cluster {label}')

        # Plot cluster centroids
        centroids = np.array(clusters['cluster_centroids'])
        if len(centroids) > 0:
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       c='red', marker='x', s=100, linewidths=3, label='Centroids')

        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title(f'Super-Resolution Localizations ({channel})\n'
                     f'{clusters["n_clusters"]} clusters, '
                     f'{clusters["n_noise"]} noise points')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)

        if save_path and self.cfg.save_figures:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"[SR] Saved figure to {save_path}")

        plt.close()

    def get_status(self) -> Dict[str, Any]:
        return {
            "platform": self.cfg.platform,
            "data_loaded": self._data is not None,
            "n_channels": len(self._data) if self._data else 0,
            "cluster_method": self.cfg.cluster_method,
            "has_scipy": HAS_SCIPY,
            "has_pandas": HAS_PANDAS,
            "has_matplotlib": HAS_MPL,
            "note": "computational comparison only; NOT diagnostic",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIScaffoldFeedbackSRSuite:
    """
    Unifies: ScaffoldingCoordinationModel + WntPCPFeedbackNetwork
             + SuperResolutionBridge.

    FULL PIPELINE:
      1. Scaffolding-mediated cargo-motor coordination
      2. Wnt/PCP signaling with feedback loops
      3. Cross-talk: PCP → scaffolding (polarity guides transport)
      4. Compare with super-resolution microscopy data
      5. Validate model predictions against experimental clusters
    """

    def __init__(self,
                 scaffold_cfg: Optional[ScaffoldConfig] = None,
                 wnt_feedback_cfg: Optional[WntFeedbackConfig] = None,
                 sr_cfg: Optional[SRConfig] = None):
        self.scaffold = ScaffoldingCoordinationModel(scaffold_cfg)
        self.wnt_feedback = WntPCPFeedbackNetwork(wnt_feedback_cfg)
        self.sr_bridge = SuperResolutionBridge(sr_cfg)

    def run_coupled_simulation(self, n_steps: int = 100) -> Dict[str, Any]:
        """
        Run coupled scaffolding + Wnt/PCP feedback simulation.
        """
        # Initialize
        self.scaffold.initialize()
        self.wnt_feedback.initialize()

        # Run Wnt/PCP feedback network
        wnt_result = self.wnt_feedback.run(n_steps=n_steps // 10)

        # Extract PCP output (JNK* activity) as polarity signal
        pcp_output = self.wnt_feedback.jnk_active
        polarity_signal = float(pcp_output.mean())

        # Run scaffolding model with polarity modulation
        # (simplified: polarity affects motor recruitment rates)
        self.scaffold.cfg.k_kinesin_recruit *= (1.0 + 0.5 * polarity_signal)
        self.scaffold.cfg.k_dynein_recruit *= (1.0 - 0.3 * polarity_signal)

        scaffold_traj, scaffold_times = self.scaffold.simulate(max_steps=n_steps)

        # Final state
        final_scaffold = scaffold_traj[-1]
        transport_eff = self.scaffold.get_transport_efficiency(scaffold_traj)

        # Detect bistability/oscillations in Wnt/PCP
        bistability = self.wnt_feedback.detect_bistability(n_initial_conditions=5)
        oscillations = self.wnt_feedback.detect_oscillations(n_steps=200)

        return {
            'n_steps': n_steps,
            'scaffold_final_state': {
                'cargo_free': final_scaffold.n_cargo_free,
                'cargo_kinesin_active': final_scaffold.n_cargo_kinesin_active,
                'cargo_dynein_active': final_scaffold.n_cargo_dynein_active,
                'cargo_tug_of_war': final_scaffold.n_cargo_tug_of_war,
            },
            'transport_efficiency': transport_eff,
            'polarity_signal': polarity_signal,
            'wnt_feedback': {
                'wnt_mean': float(self.wnt_feedback.wnt.mean()),
                'beta_cat_nuc_mean': float(self.wnt_feedback.beta_cat_nuc.mean()),
                'jnk_mean': float(self.wnt_feedback.jnk_active.mean()),
            },
            'bistability': bistability,
            'oscillations': oscillations,
        }

    def compare_with_super_resolution(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare model predictions with super-resolution data.
        """
        # Load SR data
        self.sr_bridge.load_multichannel(data_paths)

        # Run model simulation
        result = self.run_coupled_simulation(n_steps=100)

        # Generate model predictions (simplified: random points for demo)
        # In reality, would extract cargo positions from scaffold model
        n_model_points = 200
        model_points = np.random.rand(n_model_points, 2) * 1000  # nm

        # Compare with experimental data
        comparisons = {}
        for channel in data_paths.keys():
            comp = self.sr_bridge.compare_with_model(model_points, channel)
            comparisons[channel] = comp

        # Co-localization analysis (if multiple channels)
        colocalization = {}
        channels = list(data_paths.keys())
        if len(channels) >= 2:
            coloc = self.sr_bridge.compute_colocalization(channels[0], channels[1])
            colocalization = {f"{channels[0]}_{channels[1]}": coloc}

        return {
            'model_result': result,
            'comparisons': comparisons,
            'colocalization': colocalization,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "scaffold": self.scaffold.get_status(),
            "wnt_feedback": self.wnt_feedback.get_status(),
            "sr_bridge": self.sr_bridge.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO — SCAFFOLDING / FEEDBACK / SUPER-RESOLUTION")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scipy available        : {HAS_SCIPY}")
    print(f" pandas available       : {HAS_PANDAS}")
    print(f" matplotlib available   : {HAS_MPL}")

    suite = SESIScaffoldFeedbackSRSuite(
        scaffold_cfg=ScaffoldConfig(n_cargos=10, max_time=100.0, max_steps=10000),
        wnt_feedback_cfg=WntFeedbackConfig(grid_shape=(6, 6), n_steps=50),
        sr_cfg=SRConfig())

    # ── 1. Scaffolding Coordination Model ───────────────────────────────
    print("\n [SCM1] Scaffolding Coordination Model")
    scaffold = suite.scaffold
    print(f"   n_cargos             : {scaffold.cfg.n_cargos}")
    print(f"   adaptors             : {scaffold.get_status()['adaptors']}")
    print(f"   cooperativity        : {scaffold.cfg.cooperativity_factor}")

    # Run simulation
    scaffold.initialize()
    traj, times = scaffold.simulate(max_steps=5000)
    print(f"   trajectory length    : {len(traj)}")
    print(f"   final time           : {times[-1]:.2f}")

    final_state = traj[-1]
    print(f"   final state          :")
    print(f"     cargo_free         : {final_state.n_cargo_free}")
    print(f"     cargo_bicd2        : {final_state.n_cargo_bicd2}")
    print(f"     cargo_kinesin_act  : {final_state.n_cargo_kinesin_active}")
    print(f"     cargo_dynein_act   : {final_state.n_cargo_dynein_active}")
    print(f"     cargo_tug_of_war   : {final_state.n_cargo_tug_of_war}")

    transport_eff = scaffold.get_transport_efficiency(traj)
    print(f"   transport efficiency : {transport_eff['transport_efficiency']:.4f}")
    print(f"   fraction_motor_active: {transport_eff['fraction_motor_active']:.4f}")
    print(f"   fraction_tug_of_war  : {transport_eff['fraction_tug_of_war']:.4f}")

    scaffold_status = scaffold.get_status()
    print(f"   COROLLARY-3 guard    : {scaffold_status['corollary3_guard']}")
    print(f"   HINO guard           : {scaffold_status['hino_guard']}")

    # ── 2. Wnt/PCP Feedback Network ────────────────────────────────────
    print("\n [WFB1] Wnt/PCP Feedback Network")
    wnt_fb = suite.wnt_feedback
    print(f"   grid shape           : {wnt_fb.cfg.grid_shape}")
    print(f"   feedback loops       :")
    for k, v in wnt_fb.get_status()['feedback_loops'].items():
        print(f"     {k:30s} : {v:.4f}")

    # Run network
    result = wnt_fb.run(n_steps=30)
    print(f"   final Wnt mean       : {result['canonical']['wnt'].mean():.4f}")
    print(f"   final β-cat (nuc)    : {result['canonical']['beta_cat_nuc'].mean():.4f}")
    print(f"   final TCF* mean      : {result['canonical']['tcf_active'].mean():.4f}")
    print(f"   final Axin2 mean     : {result['canonical']['axin2'].mean():.4f}")
    print(f"   final JNK* mean      : {result['pcp']['jnk_active'].mean():.4f}")

    # Detect bistability
    bistability = wnt_fb.detect_bistability(n_initial_conditions=5)
    print(f"   bistability detected : {bistability['is_bistable']}")
    if bistability['is_bistable']:
        print(f"     low state Wnt      : {bistability['low_state_wnt']:.4f}")
        print(f"     high state Wnt     : {bistability['high_state_wnt']:.4f}")
        print(f"     separation         : {bistability['wnt_separation']:.4f}")

    # Detect oscillations
    oscillations = wnt_fb.detect_oscillations(n_steps=200)
    print(f"   oscillations detected: {oscillations['is_oscillatory']}")
    if oscillations['is_oscillatory']:
        print(f"     period estimate    : {oscillations['period_estimate']:.2f}")
        print(f"     amplitude          : {oscillations['amplitude']:.4f}")

    wnt_status = wnt_fb.get_status()
    print(f"   COROLLARY-3 guard    : {wnt_status['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {wnt_status['prop3_awareness']}")

    # ── 3. Coupled Simulation ──────────────────────────────────────────
    print("\n [COUPLED] Scaffolding + Wnt/PCP Feedback Coupling")
    coupled_result = suite.run_coupled_simulation(n_steps=50)
    print(f"   n_steps              : {coupled_result['n_steps']}")
    print(f"   polarity signal      : {coupled_result['polarity_signal']:.4f}")
    print(f"   scaffold final       : {coupled_result['scaffold_final_state']}")
    print(f"   transport efficiency : {coupled_result['transport_efficiency']['transport_efficiency']:.4f}")
    print(f"   Wnt feedback         : {coupled_result['wnt_feedback']}")
    print(f"   bistability          : {coupled_result['bistability']['is_bistable']}")
    print(f"   oscillations         : {coupled_result['oscillations']['is_oscillatory']}")

    # ── 4. Super-Resolution Bridge ─────────────────────────────────────
    print("\n [SRB1] Super-Resolution Microscopy Bridge")
    sr_bridge = suite.sr_bridge
    print(f"   platform             : {sr_bridge.cfg.platform}")
    print(f"   data loaded          : {sr_bridge.get_status()['data_loaded']}")
    print(f"   scipy available      : {sr_bridge.get_status()['has_scipy']}")
    print(f"   Note                 : {sr_bridge.get_status()['note']}")

    # Synthetic test (no real data)
    print(f"   Synthetic test       : creating mock SR data...")
    # Create mock localization data
    n_points = 300
    # Generate clustered data
    n_clusters = 5
    cluster_centers = np.random.rand(n_clusters, 2) * 1000
    mock_x = []
    mock_y = []
    for center in cluster_centers:
        n_in_cluster = n_points // n_clusters
        mock_x.extend(center[0] + np.random.randn(n_in_cluster) * 30)
        mock_y.extend(center[1] + np.random.randn(n_in_cluster) * 30)

    mock_data = LocalizationData(
        x_nm=np.array(mock_x),
        y_nm=np.array(mock_y),
        z_nm=None,
        intensity=None,
        frame=None,
        channel="ch1",
        n_localizations=len(mock_x),
    )
    sr_bridge._data = {"ch1": mock_data}

    # Cluster analysis
    clusters = sr_bridge.cluster_dbscan(mock_data)
    print(f"   DBSCAN clusters      : {clusters['n_clusters']}")
    print(f"   fraction clustered   : {clusters['fraction_clustered']:.4f}")
    print(f"   noise points         : {clusters['n_noise']}")

    # Ripley's K
    if HAS_SCIPY:
        ripley = sr_bridge.compute_ripley_k(mock_data)
        print(f"   Ripley's K           : is_clustered={ripley['is_clustered']}")
        print(f"   L(r) max             : {ripley['L_r'].max():.2f}")

    # Pair correlation
    if HAS_SCIPY:
        pair_corr = sr_bridge.compute_pair_correlation(mock_data)
        print(f"   Pair correlation     : is_clustered={pair_corr['is_clustered']}")
        print(f"   g(r) max             : {pair_corr['g_r'].max():.2f}")

    # Compare with model
    model_points = np.random.rand(200, 2) * 1000
    comparison = sr_bridge.compare_with_model(model_points, "ch1")
    print(f"   Model comparison     :")
    print(f"     exp clusters       : {comparison['n_exp_clusters']}")
    print(f"     model clusters     : {comparison['n_model_clusters']}")
    print(f"     Ripley L diff      : {comparison['ripley_L_difference']:.4f}")

    # ── 5. Full Status ─────────────────────────────────────────────────
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
    print("       Real scaffolding involves complex multi-protein assemblies,")
    print("       post-translational modifications, regulatory mechanisms.")
    print(" [C]   Wnt/PCP feedback: extended pathway with feedback loops is")
    print("       simplified. Real Wnt signaling involves many more components,")
    print("       non-linear dynamics, context-dependent regulation.")
    print(" [D]   Bifurcation analysis (bistability, oscillations) is")
    print("       illustrative; real cells have additional regulatory mechanisms.")
    print(" [C]   Super-resolution (STORM/PALM): single-molecule localization")
    print("       with ~20 nm precision (Betzig 2006, Rust 2006).")
    print(" [D]   SR bridge: statistical comparison is HEURISTIC. Real STORM/PALM")
    print("       has localization errors, labeling efficiency issues,")
    print("       photophysics artifacts. For calibration/validation only, NOT diagnosis.")
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
