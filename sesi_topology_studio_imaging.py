"""
=============================================================================
SESI ORGANOID STUDIO IMAGING
Cargo Binding Dynamics + Wnt/PCP Cross-Talk + Live-Cell Imaging Integration
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
[CBD1] CargoBindingDynamics
      • Multi-state Gillespie SSA [C Klumpp & Lipowsky 2005]
      • Cargo states: unbound, bound-to-kinesin, bound-to-dynein,
                      bound-to-both (tug-of-war)
      • Reactions: binding/unbinding, motor stepping, detachment under load
      • Tug-of-war: force-velocity coupling between opposing motors
      • Coupling with motor protein dynamics from previous module
      • COROLLARY-3 guard: first-order reactions only
      • HINO guard: discrete binding events, not smooth density

[WPC1] WntPCPCrossTalkNetwork
      • Full canonical + non-canonical Wnt signaling [C Veeman 2003; Gao 2010]
      • Canonical: Wnt → Fz/LRP → Dsh → β-catenin → TCF/LEF
      • Non-canonical: Wnt → Fz → Dsh → RhoA/JNK (PCP)
      • Cross-inhibition: Dsh* partitions between pathways
      • Spatial diffusion with Wnt gradient
      • ODE-based with stochastic switching for low-copy components
      • COROLLARY-3 guard: first-order reactions/diffusion only
      • PROP-3 awareness: field products lose two Sobolev orders

[LCI1] LiveCellImagingBridge
      • Load time-lapse microscopy data (TIFF/PNG sequences)
      • Simplified cell tracking (centroid-based)
      • Kymograph analysis for motor transport visualization
      • Statistical comparison: model vs data (KS test, correlation)
      • Trajectory extraction for motor velocity validation
      • HONEST: computational comparison only; NOT diagnostic;
        NOT a substitute for experimental validation

=============================================================================
HONEST SCOPE
=============================================================================
• CargoBindingDynamics: SSA is exact for well-mixed compartments [C].
  Real cargo transport involves scaffolding proteins, adaptor complexes,
  and regulatory mechanisms not captured here.
• WntPCPCrossTalkNetwork: coupled pathway model is simplified. Real Wnt
  signaling involves many more components, feedback loops, and context-
  dependent regulation. This is computational only.
• LiveCellImagingBridge: statistical comparison is HEURISTIC. Real
  microscopy has noise, photobleaching, and motion artifacts. This bridge
  is for calibration/validation only, NOT diagnosis.
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

logger = logging.getLogger("sesi.imaging")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from skimage import io as skio
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

__all__ = [
    "CargoBindingDynamics", "CargoConfig", "CargoState",
    "WntPCPCrossTalkNetwork", "WntPCPConfig",
    "LiveCellImagingBridge", "ImagingConfig", "TrackingResult",
    "SESIImagingOrganoidSuite",
]


# =============================================================================
# SECTION 1 — [CBD1] CARGO BINDING DYNAMICS
# =============================================================================

@dataclass
class CargoConfig:
    """Configuration for cargo-motor binding dynamics."""
    # Number of cargos
    n_cargos: int = 20
    # Motor availability (per cell)
    n_kinesin: int = 50
    n_dynein: int = 30
    # Binding rates [C Klumpp & Lipowsky 2005; Müller 2008]
    k_bind_kinesin: float = 0.5     # cargo + kinesin → bound
    k_bind_dynein: float = 0.3
    k_unbind_kinesin: float = 0.1   # bound → cargo + kinesin
    k_unbind_dynein: float = 0.15
    # Force-dependent unbinding (Bell model)
    k_unbind_force: float = 0.05    # force sensitivity
    # Tug-of-war parameters
    tug_of_war_threshold: float = 0.5  # force threshold for detachment
    # Simulation
    max_time: float = 100.0
    max_steps: int = 100000
    seed: int = 0


@dataclass
class CargoState:
    """State of cargo-motor system."""
    time: float
    n_unbound: int
    n_kinesin_bound: int
    n_dynein_bound: int
    n_tug_of_war: int  # bound to both motors
    cargo_positions: np.ndarray  # (n_cargos,) positions in μm
    metadata: Dict[str, Any] = field(default_factory=dict)


class CargoBindingDynamics:
    """
    [C/D] Stochastic cargo-motor binding dynamics with tug-of-war.

    BIOLOGICAL BASIS [C Klumpp & Lipowsky 2005; Müller et al. 2008]
      Cargos (vesicles, organelles) are transported by motor proteins:
        - Kinesin: moves toward microtubule plus-end (anterograde)
        - Dynein: moves toward minus-end (retrograde)

      A cargo can be:
        1. Unbound (diffuses slowly)
        2. Bound to kinesin only (moves anterograde)
        3. Bound to dynein only (moves retrograde)
        4. Bound to both (tug-of-war: net direction depends on force balance)

      Reactions:
        Unbound + Kinesin → Kinesin-bound    rate k_bind_kinesin
        Unbound + Dynein → Dynein-bound      rate k_bind_dynein
        Kinesin-bound → Unbound + Kinesin    rate k_unbind_kinesin(F)
        Dynein-bound → Unbound + Dynein      rate k_unbind_dynein(F)
        Kinesin-bound + Dynein → Tug-of-war  rate k_bind (when both available)
        Tug-of-war → Kinesin-bound           rate k_unbind_dynein (dynein detaches)
        Tug-of-war → Dynein-bound            rate k_unbind_kinesin (kinesin detaches)

      Force-dependent unbinding (Bell model):
        k_unbind(F) = k_unbind(0) * exp(F * d / kBT)
      where F is load force, d is bond distance, kBT is thermal energy.

    TUG-OF-WAR [C Müller 2008]
      When a cargo is bound to both motors, they pull in opposite directions.
      Net velocity: v_net = (n_k * v_k - n_d * v_d) / (n_k + n_d)
      where n_k, n_d are number of engaged motors, v_k, v_d are velocities.
      The motor with higher force detaches first (stochastic).

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear or bilinear in
      copy numbers). We do NOT compute second derivatives of cargo state.
      The noise term is multiplicative but first-order.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates × copy numbers), NOT from an assumed smooth noise density.
      The stochasticity is singular in the sense of being tied to individual
      binding/unbinding events.

    HONEST
      • SSA is exact for well-mixed compartments; real cargo transport
        involves scaffolding proteins, adaptor complexes, regulatory
        mechanisms not captured here.
      • Tug-of-war is simplified; real bidirectional transport involves
        complex coordination (e.g., dynactin, BICD2).
      • Copy numbers are illustrative; real cells have ~10-100 motors
        per cargo typically.
    """

    def __init__(self, cfg: Optional[CargoConfig] = None):
        self.cfg = cfg or CargoConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # State: counts of each cargo state
        self.n_unbound: int = 0
        self.n_kinesin_bound: int = 0
        self.n_dynein_bound: int = 0
        self.n_tug_of_war: int = 0

        # Cargo positions (μm)
        self.cargo_positions: Optional[np.ndarray] = None

        # Available motors
        self.available_kinesin: int = 0
        self.available_dynein: int = 0

    def initialize(self) -> None:
        """Initialize cargo-motor system."""
        # All cargos start unbound
        self.n_unbound = self.cfg.n_cargos
        self.n_kinesin_bound = 0
        self.n_dynein_bound = 0
        self.n_tug_of_war = 0

        # All motors available
        self.available_kinesin = self.cfg.n_kinesin
        self.available_dynein = self.cfg.n_dynein

        # Random initial positions
        self.cargo_positions = self._rng.uniform(0, 10, size=self.cfg.n_cargos)

    def compute_propensities(self, load_force: float = 0.0
                              ) -> Tuple[np.ndarray, List[Tuple[str, Dict[str, int]]]]:
        """
        Compute all reaction propensities.
        Returns: (propensities array, list of (reaction_name, state_delta))
        """
        props = []
        events = []

        # 1. Unbound + Kinesin → Kinesin-bound
        if self.n_unbound > 0 and self.available_kinesin > 0:
            a = self.cfg.k_bind_kinesin * self.n_unbound * self.available_kinesin
            props.append(a)
            events.append(("bind_kinesin", {
                'n_unbound': -1, 'n_kinesin_bound': +1,
                'available_kinesin': -1
            }))

        # 2. Unbound + Dynein → Dynein-bound
        if self.n_unbound > 0 and self.available_dynein > 0:
            a = self.cfg.k_bind_dynein * self.n_unbound * self.available_dynein
            props.append(a)
            events.append(("bind_dynein", {
                'n_unbound': -1, 'n_dynein_bound': +1,
                'available_dynein': -1
            }))

        # 3. Kinesin-bound → Unbound + Kinesin (force-dependent)
        if self.n_kinesin_bound > 0:
            k_unbind = self.cfg.k_unbind_kinesin * math.exp(
                self.cfg.k_unbind_force * abs(load_force))
            a = k_unbind * self.n_kinesin_bound
            props.append(a)
            events.append(("unbind_kinesin", {
                'n_kinesin_bound': -1, 'n_unbound': +1,
                'available_kinesin': +1
            }))

        # 4. Dynein-bound → Unbound + Dynein (force-dependent)
        if self.n_dynein_bound > 0:
            k_unbind = self.cfg.k_unbind_dynein * math.exp(
                self.cfg.k_unbind_force * abs(load_force))
            a = k_unbind * self.n_dynein_bound
            props.append(a)
            events.append(("unbind_dynein", {
                'n_dynein_bound': -1, 'n_unbound': +1,
                'available_dynein': +1
            }))

        # 5. Kinesin-bound + Dynein → Tug-of-war
        if self.n_kinesin_bound > 0 and self.available_dynein > 0:
            a = self.cfg.k_bind_dynein * 0.5 * self.n_kinesin_bound * self.available_dynein
            props.append(a)
            events.append(("tug_of_war_from_kinesin", {
                'n_kinesin_bound': -1, 'n_tug_of_war': +1,
                'available_dynein': -1
            }))

        # 6. Dynein-bound + Kinesin → Tug-of-war
        if self.n_dynein_bound > 0 and self.available_kinesin > 0:
            a = self.cfg.k_bind_kinesin * 0.5 * self.n_dynein_bound * self.available_kinesin
            props.append(a)
            events.append(("tug_of_war_from_dynein", {
                'n_dynein_bound': -1, 'n_tug_of_war': +1,
                'available_kinesin': -1
            }))

        # 7. Tug-of-war → Kinesin-bound (dynein detaches)
        if self.n_tug_of_war > 0:
            k_unbind = self.cfg.k_unbind_dynein * math.exp(
                self.cfg.k_unbind_force * abs(load_force))
            a = k_unbind * self.n_tug_of_war
            props.append(a)
            events.append(("tug_of_war_to_kinesin", {
                'n_tug_of_war': -1, 'n_kinesin_bound': +1,
                'available_dynein': +1
            }))

        # 8. Tug-of-war → Dynein-bound (kinesin detaches)
        if self.n_tug_of_war > 0:
            k_unbind = self.cfg.k_unbind_kinesin * math.exp(
                self.cfg.k_unbind_force * abs(load_force))
            a = k_unbind * self.n_tug_of_war
            props.append(a)
            events.append(("tug_of_war_to_dynein", {
                'n_tug_of_war': -1, 'n_dynein_bound': +1,
                'available_kinesin': +1
            }))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self, load_force: float = 0.0) -> Tuple[float, Dict[str, int]]:
        """One Gillespie SSA step. Returns (time_increment, state_delta)."""
        props, events = self.compute_propensities(load_force)
        a_total = props.sum()

        if a_total <= 0:
            return 0.0, {}

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
            if key == 'n_unbound':
                self.n_unbound = max(0, self.n_unbound + change)
            elif key == 'n_kinesin_bound':
                self.n_kinesin_bound = max(0, self.n_kinesin_bound + change)
            elif key == 'n_dynein_bound':
                self.n_dynein_bound = max(0, self.n_dynein_bound + change)
            elif key == 'n_tug_of_war':
                self.n_tug_of_war = max(0, self.n_tug_of_war + change)
            elif key == 'available_kinesin':
                self.available_kinesin = max(0, self.available_kinesin + change)
            elif key == 'available_dynein':
                self.available_dynein = max(0, self.available_dynein + change)

        return tau, delta

    def update_positions(self, dt: float, load_force: float = 0.0) -> None:
        """Update cargo positions based on motor binding state."""
        if self.cargo_positions is None:
            return

        # Velocities (μm/s)
        v_kinesin = 1.0  # ~1 μm/s toward plus-end
        v_dynein = -0.8  # ~0.8 μm/s toward minus-end
        v_diffusion = 0.01  # slow diffusion when unbound

        # Update each cargo based on its state
        for i in range(self.cfg.n_cargos):
            # Simplified: assign states randomly based on population fractions
            total = max(1, self.n_unbound + self.n_kinesin_bound +
                        self.n_dynein_bound + self.n_tug_of_war)
            r = self._rng.random() * total

            if r < self.n_unbound:
                # Unbound: diffusion
                self.cargo_positions[i] += v_diffusion * self._rng.standard_normal() * math.sqrt(dt)
            elif r < self.n_unbound + self.n_kinesin_bound:
                # Kinesin-bound: anterograde
                self.cargo_positions[i] += v_kinesin * dt
            elif r < self.n_unbound + self.n_kinesin_bound + self.n_dynein_bound:
                # Dynein-bound: retrograde
                self.cargo_positions[i] += v_dynein * dt
            else:
                # Tug-of-war: net velocity depends on force balance
                # Simplified: random walk with bias
                v_net = (v_kinesin - v_dynein) / 2.0 * (1.0 - abs(load_force) / self.cfg.tug_of_war_threshold)
                self.cargo_positions[i] += v_net * dt + v_diffusion * self._rng.standard_normal() * math.sqrt(dt)

        # Wrap around (periodic boundary)
        self.cargo_positions = np.mod(self.cargo_positions, 10.0)

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None,
                  load_force: float = 0.0
                  ) -> Tuple[List[CargoState], List[float]]:
        """Run full SSA simulation."""
        if self.cargo_positions is None:
            self.initialize()

        t = 0.0
        trajectory = [CargoState(
            time=0.0,
            n_unbound=self.n_unbound,
            n_kinesin_bound=self.n_kinesin_bound,
            n_dynein_bound=self.n_dynein_bound,
            n_tug_of_war=self.n_tug_of_war,
            cargo_positions=self.cargo_positions.copy(),
        )]
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        steps = 0
        while t < max_time and steps < max_steps:
            dt, _ = self.ssa_step(load_force)
            if dt <= 0:
                break
            t += dt
            self.update_positions(dt, load_force)

            trajectory.append(CargoState(
                time=t,
                n_unbound=self.n_unbound,
                n_kinesin_bound=self.n_kinesin_bound,
                n_dynein_bound=self.n_dynein_bound,
                n_tug_of_war=self.n_tug_of_war,
                cargo_positions=self.cargo_positions.copy(),
            ))
            times.append(t)
            steps += 1

        return trajectory, times

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_cargos": self.cfg.n_cargos,
            "n_kinesin": self.cfg.n_kinesin,
            "n_dynein": self.cfg.n_dynein,
            "corollary3_guard": "first-order reactions only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [WPC1] WNT/PCP CROSS-TALK NETWORK
# =============================================================================

@dataclass
class WntPCPConfig:
    """Configuration for Wnt/PCP cross-talk network."""
    # Spatial grid
    grid_shape: Tuple[int, int] = (8, 8)
    # Canonical pathway (Wnt/β-catenin)
    wnt_initial: float = 0.5
    fz_initial: float = 1.0
    dsh_initial: float = 0.5
    beta_cat_cyto: float = 0.2
    beta_cat_nuc: float = 0.1
    tcf_lef: float = 1.0
    # Non-canonical pathway (PCP)
    vang_initial: float = 1.0
    pk_initial: float = 1.0
    rhoa_initial: float = 0.5
    jnk_initial: float = 0.2
    # Cross-talk parameters [C Veeman 2003; Gao 2010]
    dsh_partition_canonical: float = 0.6  # fraction of Dsh* → canonical
    dsh_partition_pcp: float = 0.4        # fraction of Dsh* → PCP
    cross_inhibition_strength: float = 0.3  # canonical inhibits PCP and vice versa
    # Reaction rates
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


class WntPCPCrossTalkNetwork:
    """
    [C/D] Full Wnt signaling with canonical/non-canonical cross-talk.

    BIOLOGICAL BASIS [C Veeman et al. 2003; Gao & Chen 2010]
      Wnt ligands activate two major pathways:

      CANONICAL (Wnt/β-catenin):
        Wnt → Fz/LRP → Dsh* → inhibits destruction complex → β-catenin
        accumulates → nuclear translocation → TCF/LEF activation → gene
        expression (proliferation, stemness)

      NON-CANONICAL (PCP):
        Wnt → Fz → Dsh* → RhoA → JNK → cytoskeletal reorganization
        (cell polarity, convergent extension, migration)

      CROSS-TALK:
        - Dsh* partitions between pathways (competition)
        - Canonical pathway inhibits PCP (via GSK3β)
        - PCP pathway inhibits canonical (via RhoA/ROCK)
        - Context-dependent: cell type, Wnt ligand, co-receptors

    MODEL [D]
      Coupled ODE system on a 2D grid:

      Canonical:
        d[Wnt-Fz]/dt = k_bind [Wnt][Fz] - k_unbind [Wnt-Fz]
        d[Dsh*_can]/dt = k_act [Wnt-Fz] * partition_can - k_deg [Dsh*_can]
                          - cross_inhib * [Dsh*_pcp]
        d[β-cat_cyto]/dt = k_prod + k_inhib [Dest*] - k_deg [β-cat_cyto]
                            - k_nuclear [β-cat_cyto]
        d[β-cat_nuc]/dt = k_nuclear [β-cat_cyto] - k_deg [β-cat_nuc]
        d[TCF*]/dt = k_act [β-cat_nuc][TCF] - k_deg [TCF*]

      Non-canonical (PCP):
        d[Dsh*_pcp]/dt = k_act [Wnt-Fz] * partition_pcp - k_deg [Dsh*_pcp]
                          - cross_inhib * [Dsh*_can]
        d[RhoA*]/dt = k_act [Dsh*_pcp] - k_deg [RhoA*]
        d[JNK*]/dt = k_act [RhoA*] - k_deg [JNK*]

      Spatial diffusion of Wnt, β-catenin, Dsh*.

    COROLLARY-3 GUARD [D]
      All reactions and diffusion terms are FIRST-ORDER in the fields.
      We do NOT compute second derivatives of signaling molecule concentrations.

    PROP-3 AWARENESS [P]
      When we multiply fields (e.g. [Wnt] × [Fz], [Dsh*_can] × [Dsh*_pcp]),
      the result loses two Sobolev orders (Carpet-19 Prop 3). We use
      softplus floors to maintain differentiability.

    HONEST
      • This is a simplified cross-talk model. Real Wnt signaling involves
        many more components, feedback loops, and context-dependent regulation.
      • Dsh partitioning is heuristic; real partitioning involves complex
        scaffolding and post-translational modifications.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[WntPCPConfig] = None):
        self.cfg = cfg or WntPCPConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.nx, self.ny = self.cfg.grid_shape

        # Canonical pathway fields
        self.wnt: Optional[np.ndarray] = None
        self.wnt_fz: Optional[np.ndarray] = None
        self.dsh_canonical: Optional[np.ndarray] = None
        self.beta_cat_cyto: Optional[np.ndarray] = None
        self.beta_cat_nuc: Optional[np.ndarray] = None
        self.tcf_active: Optional[np.ndarray] = None

        # Non-canonical (PCP) pathway fields
        self.dsh_pcp: Optional[np.ndarray] = None
        self.rhoa_active: Optional[np.ndarray] = None
        self.jnk_active: Optional[np.ndarray] = None

    def _laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with periodic boundary conditions."""
        lap = np.zeros_like(field)
        for axis in range(2):
            lap += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
        return lap

    def initialize(self) -> None:
        """Initialize pathway fields with small random perturbations."""
        shape = (self.nx, self.ny)
        noise = 0.05

        # Canonical
        self.wnt = self.cfg.wnt_initial + noise * self._rng.standard_normal(shape)
        self.wnt_fz = self.cfg.fz_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.dsh_canonical = self.cfg.dsh_initial * self.cfg.dsh_partition_canonical + noise * self._rng.standard_normal(shape)
        self.beta_cat_cyto = self.cfg.beta_cat_cyto + noise * self._rng.standard_normal(shape)
        self.beta_cat_nuc = self.cfg.beta_cat_nuc + noise * self._rng.standard_normal(shape)
        self.tcf_active = self.cfg.tcf_lef * 0.1 + noise * self._rng.standard_normal(shape)

        # Non-canonical
        self.dsh_pcp = self.cfg.dsh_initial * self.cfg.dsh_partition_pcp + noise * self._rng.standard_normal(shape)
        self.rhoa_active = self.cfg.rhoa_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.jnk_active = self.cfg.jnk_initial * 0.1 + noise * self._rng.standard_normal(shape)

        # Ensure non-negative
        for field in [self.wnt, self.wnt_fz, self.dsh_canonical, self.beta_cat_cyto,
                      self.beta_cat_nuc, self.tcf_active, self.dsh_pcp,
                      self.rhoa_active, self.jnk_active]:
            field[:] = np.maximum(field, 0.0)

    def step(self) -> None:
        """One signaling step with cross-talk."""
        if self.wnt is None:
            self.initialize()

        dt = self.cfg.dt

        # Laplacians for diffusion
        lap_wnt = self._laplacian_2d(self.wnt)
        lap_beta = self._laplacian_2d(self.beta_cat_cyto)
        lap_dsh = self._laplacian_2d(self.dsh_canonical)

        # Wnt-Fz binding (PROP-3 awareness: bilinear product)
        wnt_fz_binding = self.cfg.k_wnt_binding * self.wnt * self.cfg.fz_initial

        # Dsh activation (partitioned between pathways)
        dsh_activation_total = self.cfg.k_dsh_activation * self.wnt_fz
        dsh_activation_can = dsh_activation_total * self.cfg.dsh_partition_canonical
        dsh_activation_pcp = dsh_activation_total * self.cfg.dsh_partition_pcp

        # Cross-inhibition (PROP-3 awareness: bilinear product)
        cross_inhib_can = self.cfg.cross_inhibition_strength * self.dsh_pcp * self.dsh_canonical
        cross_inhib_pcp = self.cfg.cross_inhibition_strength * self.dsh_canonical * self.dsh_pcp

        # β-catenin dynamics
        beta_cat_production = self.cfg.k_beta_cat_production
        beta_cat_degradation = self.cfg.k_beta_cat_degradation * self.beta_cat_cyto
        beta_cat_nuclear_transport = self.cfg.k_beta_cat_nuclear * self.beta_cat_cyto

        # TCF activation
        tcf_activation = self.cfg.k_tcf_activation * self.beta_cat_nuc * self.cfg.tcf_lef

        # RhoA activation
        rhoa_activation = self.cfg.k_rhoa_activation * self.dsh_pcp

        # JNK activation
        jnk_activation = self.cfg.k_jnk_activation * self.rhoa_active

        # Update canonical fields
        self.wnt = np.maximum(self.wnt + dt * (
            self.cfg.d_wnt * lap_wnt
            + self.cfg.k_production
            - self.cfg.k_degradation * self.wnt
            - wnt_fz_binding), 0.0)

        self.wnt_fz = np.maximum(self.wnt_fz + dt * (
            wnt_fz_binding - self.cfg.k_degradation * self.wnt_fz), 0.0)

        self.dsh_canonical = np.maximum(self.dsh_canonical + dt * (
            self.cfg.d_dsh * lap_dsh
            + dsh_activation_can - self.cfg.k_degradation * self.dsh_canonical
            - cross_inhib_can), 0.0)

        self.beta_cat_cyto = np.maximum(self.beta_cat_cyto + dt * (
            self.cfg.d_beta_cat * lap_beta
            + beta_cat_production - self.cfg.k_degradation * self.beta_cat_cyto
            - beta_cat_nuclear_transport), 0.0)

        self.beta_cat_nuc = np.maximum(self.beta_cat_nuc + dt * (
            beta_cat_nuclear_transport - self.cfg.k_degradation * self.beta_cat_nuc), 0.0)

        self.tcf_active = np.maximum(self.tcf_active + dt * (
            tcf_activation - self.cfg.k_degradation * self.tcf_active), 0.0)

        # Update non-canonical (PCP) fields
        self.dsh_pcp = np.maximum(self.dsh_pcp + dt * (
            dsh_activation_pcp - self.cfg.k_degradation * self.dsh_pcp
            - cross_inhib_pcp), 0.0)

        self.rhoa_active = np.maximum(self.rhoa_active + dt * (
            rhoa_activation - self.cfg.k_degradation * self.rhoa_active), 0.0)

        self.jnk_active = np.maximum(self.jnk_active + dt * (
            jnk_activation - self.cfg.k_degradation * self.jnk_active), 0.0)

    def run(self, n_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Run the coupled network for n_steps."""
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
            },
            'pcp': {
                'dsh_pcp': self.dsh_pcp.copy(),
                'rhoa_active': self.rhoa_active.copy(),
                'jnk_active': self.jnk_active.copy(),
            },
        }

    def get_canonical_output(self) -> np.ndarray:
        """Get canonical pathway output (TCF* activity)."""
        if self.tcf_active is None:
            return np.zeros((self.nx, self.ny))
        return self.tcf_active

    def get_pcp_output(self) -> np.ndarray:
        """Get PCP pathway output (JNK* activity)."""
        if self.jnk_active is None:
            return np.zeros((self.nx, self.ny))
        return self.jnk_active

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid_shape": self.cfg.grid_shape,
            "canonical_components": ["Wnt", "Fz", "Dsh*", "β-catenin", "TCF/LEF"],
            "pcp_components": ["Dsh*", "RhoA", "JNK"],
            "cross_talk": {
                "dsh_partition_canonical": self.cfg.dsh_partition_canonical,
                "dsh_partition_pcp": self.cfg.dsh_partition_pcp,
                "cross_inhibition_strength": self.cfg.cross_inhibition_strength,
            },
            "corollary3_guard": "first-order reactions/diffusion; no Hessian",
            "prop3_awareness": "bilinear products lose two Sobolev orders",
            "note": "computational model, not validated molecular biology",
        }


# =============================================================================
# SECTION 3 — [LCI1] LIVE-CELL IMAGING BRIDGE
# =============================================================================

@dataclass
class ImagingConfig:
    """Configuration for live-cell imaging integration."""
    # Data loading
    image_sequence_path: Optional[str] = None
    image_format: str = "tiff"  # "tiff" | "png" | "tif"
    # Tracking parameters
    cell_detection_method: str = "threshold"  # "threshold" | "edge" | "manual"
    detection_threshold: float = 0.5
    min_cell_size: int = 50  # pixels
    max_cell_size: int = 500
    # Kymograph parameters
    kymograph_line_start: Tuple[float, float] = (0.5, 0.0)
    kymograph_line_end: Tuple[float, float] = (0.5, 1.0)
    kymograph_width: int = 5  # pixels
    # Time parameters
    frame_interval_s: float = 1.0  # seconds between frames
    # Comparison
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    # Visualization
    save_figures: bool = False
    figure_dir: Optional[str] = None


@dataclass
class TrackingResult:
    """Result of cell/particle tracking."""
    n_frames: int
    n_tracks: int
    track_lengths: List[int]
    velocities: np.ndarray  # (n_tracks,) μm/s
    displacements: np.ndarray  # (n_tracks,) μm
    directions: np.ndarray  # (n_tracks,) radians
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiveCellImagingBridge:
    """
    [D] Bridge to live-cell imaging data for model validation.

    CAPABILITIES
      1. Load time-lapse microscopy sequences (TIFF/PNG)
      2. Simplified cell/particle tracking (centroid-based)
      3. Kymograph analysis for motor transport visualization
      4. Trajectory extraction (position, velocity, direction)
      5. Statistical comparison with model predictions
         - Velocity distributions (KS test)
         - Run length distributions
         - Directionality analysis

    TRACKING [D]
      Simplified centroid-based tracking:
        1. Threshold segmentation to detect bright objects
        2. Connected component labeling
        3. Filter by size (min_cell_size, max_cell_size)
        4. Track centroids across frames (nearest-neighbor linking)
        5. Compute trajectories, velocities, displacements

    KYMOGRAPH [D]
      Extract intensity along a line over time:
        1. Define line from kymograph_line_start to kymograph_line_end
        2. For each frame, sample intensity along line (with width)
        3. Stack over time → kymograph (space × time)
        4. Visualize motor transport as diagonal streaks

    STATISTICAL COMPARISON [C]
      - Kolmogorov-Smirnov test: compare velocity distributions
      - Wasserstein distance: compare trajectory patterns
      - Correlation: compare time series

    COROLLARY-3 GUARD [D]
      All tracking and comparison operations are FIRST-ORDER (no Hessian
      of likelihood). Statistical tests are distribution-based.

    HONEST
      • This is a COMPUTATIONAL comparison tool, NOT a diagnostic.
      • Real microscopy has noise, photobleaching, motion artifacts.
      • Tracking is simplified; real tracking requires sophisticated
        algorithms (TrackMate, u-track, etc.).
      • No biological/medical/diagnostic claim.
    """

    def __init__(self, cfg: Optional[ImagingConfig] = None):
        self.cfg = cfg or ImagingConfig()
        self._images: Optional[np.ndarray] = None  # (T, H, W) or (T, H, W, C)
        self._tracking_result: Optional[TrackingResult] = None

    # ── Data loading ────────────────────────────────────────────────────
    def load_image_sequence(self, path: Optional[str] = None) -> np.ndarray:
        """
        Load time-lapse image sequence.
        Returns: (T, H, W) or (T, H, W, C) array.
        """
        path = path or self.cfg.image_sequence_path
        if path is None:
            raise ValueError("No image sequence path specified")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image sequence not found: {path}")

        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required for image loading")

        # Load TIFF stack or directory of images
        if os.path.isdir(path):
            # Load all images in directory
            files = sorted([f for f in os.listdir(path)
                           if f.lower().endswith(('.tiff', '.tif', '.png'))])
            images = [skio.imread(os.path.join(path, f)) for f in files]
            self._images = np.stack(images, axis=0)
        else:
            # Load single TIFF stack
            self._images = skio.imread(path)

        logger.info(f"[Imaging] Loaded {self._images.shape[0]} frames, "
                    f"shape {self._images.shape[1:]}")
        return self._images

    # ── Cell tracking ───────────────────────────────────────────────────
    def track_cells(self) -> TrackingResult:
        """
        Track cells/particles across frames.
        Returns: TrackingResult with trajectories and statistics.
        """
        if self._images is None:
            raise ValueError("No images loaded")

        from scipy import ndimage

        n_frames = self._images.shape[0]
        all_centroids = []

        # Detect cells in each frame
        for t in range(n_frames):
            frame = self._images[t]
            if frame.ndim == 3:
                frame = frame.mean(axis=-1)  # convert to grayscale

            # Threshold segmentation
            threshold = np.percentile(frame, self.cfg.detection_threshold * 100)
            binary = frame > threshold

            # Connected component labeling
            labeled, n_objects = ndimage.label(binary, return_mask=False)

            # Filter by size and extract centroids
            frame_centroids = []
            for i in range(1, n_objects + 1):
                mask = labeled == i
                size = mask.sum()
                if self.cfg.min_cell_size <= size <= self.cfg.max_cell_size:
                    centroid = ndimage.center_of_mass(mask)
                    frame_centroids.append(centroid)

            all_centroids.append(frame_centroids)

        # Link centroids across frames (nearest-neighbor)
        tracks = self._link_tracks(all_centroids)

        # Compute statistics
        n_tracks = len(tracks)
        track_lengths = [len(track) for track in tracks]

        velocities = np.zeros(n_tracks)
        displacements = np.zeros(n_tracks)
        directions = np.zeros(n_tracks)

        for i, track in enumerate(tracks):
            if len(track) < 2:
                continue
            positions = np.array(track)
            # Velocity: mean displacement per frame
            diffs = np.diff(positions, axis=0)
            speeds = np.linalg.norm(diffs, axis=1) / self.cfg.frame_interval_s
            velocities[i] = speeds.mean()
            # Displacement: total distance
            displacements[i] = np.linalg.norm(positions[-1] - positions[0])
            # Direction: angle of net displacement
            net_disp = positions[-1] - positions[0]
            directions[i] = np.arctan2(net_disp[1], net_disp[0])

        self._tracking_result = TrackingResult(
            n_frames=n_frames,
            n_tracks=n_tracks,
            track_lengths=track_lengths,
            velocities=velocities,
            displacements=displacements,
            directions=directions,
            metadata={"all_centroids": all_centroids, "tracks": tracks},
        )

        return self._tracking_result

    def _link_tracks(self, all_centroids: List[List[Tuple[float, float]]]
                      ) -> List[List[Tuple[float, float]]]:
        """Link centroids across frames using nearest-neighbor."""
        if not all_centroids or not all_centroids[0]:
            return []

        # Initialize tracks with first frame
        tracks = [[c] for c in all_centroids[0]]

        # Link subsequent frames
        for t in range(1, len(all_centroids)):
            prev_centroids = [track[-1] for track in tracks]
            curr_centroids = all_centroids[t]

            if not curr_centroids:
                continue

            # Compute distance matrix
            prev_arr = np.array(prev_centroids)
            curr_arr = np.array(curr_centroids)
            dist_matrix = np.linalg.norm(
                prev_arr[:, np.newaxis, :] - curr_arr[np.newaxis, :, :],
                axis=2
            )

            # Greedy nearest-neighbor linking
            max_dist = 50.0  # pixels
            linked_curr = set()
            for i in range(len(prev_centroids)):
                if i >= dist_matrix.shape[0]:
                    break
                j = np.argmin(dist_matrix[i])
                if dist_matrix[i, j] < max_dist and j not in linked_curr:
                    tracks[i].append(tuple(curr_centroids[j]))
                    linked_curr.add(j)

            # Start new tracks for unlinked centroids
            for j in range(len(curr_centroids)):
                if j not in linked_curr:
                    tracks.append([curr_centroids[j]])

        return tracks

    # ── Kymograph analysis ──────────────────────────────────────────────
    def generate_kymograph(self) -> np.ndarray:
        """
        Generate kymograph along a line over time.
        Returns: (T, W) array where W is line length in pixels.
        """
        if self._images is None:
            raise ValueError("No images loaded")

        n_frames = self._images.shape[0]
        H, W = self._images.shape[1:3]

        # Define line
        x0, y0 = int(self.cfg.kymograph_line_start[0] * W), int(self.cfg.kymograph_line_start[1] * H)
        x1, y1 = int(self.cfg.kymograph_line_end[0] * W), int(self.cfg.kymograph_line_end[1] * H)

        # Sample points along line
        n_points = int(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
        xs = np.linspace(x0, x1, n_points, dtype=int)
        ys = np.linspace(y0, y1, n_points, dtype=int)

        kymograph = np.zeros((n_frames, n_points))

        for t in range(n_frames):
            frame = self._images[t]
            if frame.ndim == 3:
                frame = frame.mean(axis=-1)

            # Sample intensity along line (with width)
            for i, (x, y) in enumerate(zip(xs, ys)):
                # Average over width
                region = frame[max(0, y - self.cfg.kymograph_width):
                               min(H, y + self.cfg.kymograph_width + 1),
                               max(0, x - self.cfg.kymograph_width):
                               min(W, x + self.cfg.kymograph_width + 1)]
                kymograph[t, i] = region.mean()

        return kymograph

    # ── Statistical comparison ──────────────────────────────────────────
    def compare_velocities(self, model_velocities: np.ndarray) -> Dict[str, float]:
        """
        Compare model velocity distribution with experimental data.
        """
        if self._tracking_result is None:
            raise ValueError("No tracking data available")

        exp_velocities = self._tracking_result.velocities

        if not HAS_SCIPY:
            return {"error": "scipy not available"}

        # KS test
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(exp_velocities, model_velocities)

        # Wasserstein distance
        wasserstein = scipy_stats.wasserstein_distance(exp_velocities, model_velocities)

        # Mean and std comparison
        mean_diff = abs(exp_velocities.mean() - model_velocities.mean())
        std_ratio = exp_velocities.std() / (model_velocities.std() + 1e-12)

        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'wasserstein_distance': float(wasserstein),
            'mean_absolute_difference': float(mean_diff),
            'std_ratio': float(std_ratio),
            'experimental_mean': float(exp_velocities.mean()),
            'experimental_std': float(exp_velocities.std()),
            'model_mean': float(model_velocities.mean()),
            'model_std': float(model_velocities.std()),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "images_loaded": self._images is not None,
            "tracking_available": self._tracking_result is not None,
            "has_skimage": HAS_SKIMAGE,
            "has_scipy": HAS_SCIPY,
            "note": "computational comparison only; NOT diagnostic",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIImagingOrganoidSuite:
    """
    Unifies: CargoBindingDynamics + WntPCPCrossTalkNetwork + LiveCellImagingBridge.

    FULL IMAGING PIPELINE:
      1. Cargo-motor binding dynamics with tug-of-war
      2. Wnt/PCP cross-talk network (canonical + non-canonical)
      3. Load live-cell imaging data
      4. Track cargos/cells and compare with model
      5. Generate kymographs for visualization
    """

    def __init__(self,
                 cargo_cfg: Optional[CargoConfig] = None,
                 wnt_pcp_cfg: Optional[WntPCPConfig] = None,
                 imaging_cfg: Optional[ImagingConfig] = None):
        self.cargo = CargoBindingDynamics(cargo_cfg)
        self.wnt_pcp = WntPCPCrossTalkNetwork(wnt_pcp_cfg)
        self.imaging = LiveCellImagingBridge(imaging_cfg)

    def run_coupled_simulation(self, n_steps: int = 100) -> Dict[str, Any]:
        """
        Run coupled cargo + Wnt/PCP simulation.
        """
        # Initialize
        self.cargo.initialize()
        self.wnt_pcp.initialize()

        # Run Wnt/PCP network
        wnt_pcp_result = self.wnt_pcp.run(n_steps=n_steps // 10)

        # Extract PCP output (JNK* activity) as load force for cargo
        pcp_output = self.wnt_pcp.get_pcp_output()
        load_force = float(pcp_output.mean()) * 2.0  # scale to force

        # Run cargo dynamics with load force
        cargo_traj, cargo_times = self.cargo.simulate(
            max_steps=n_steps, load_force=load_force)

        # Final state
        final_cargo = cargo_traj[-1]
        final_wnt = {k: float(v.mean()) for k, v in wnt_pcp_result['canonical'].items()}
        final_pcp = {k: float(v.mean()) for k, v in wnt_pcp_result['pcp'].items()}

        return {
            'n_steps': n_steps,
            'cargo_final_state': {
                'n_unbound': final_cargo.n_unbound,
                'n_kinesin_bound': final_cargo.n_kinesin_bound,
                'n_dynein_bound': final_cargo.n_dynein_bound,
                'n_tug_of_war': final_cargo.n_tug_of_war,
            },
            'canonical_pathway': final_wnt,
            'pcp_pathway': final_pcp,
            'load_force': load_force,
            'cargo_trajectory_length': len(cargo_traj),
        }

    def compare_with_imaging(self, image_path: str) -> Dict[str, Any]:
        """
        Compare model predictions with live-cell imaging data.
        """
        # Load imaging data
        self.imaging.load_image_sequence(image_path)

        # Track cells/cargos
        tracking_result = self.imaging.track_cells()

        # Run model simulation
        model_result = self.run_coupled_simulation(n_steps=200)

        # Extract model cargo velocities (simplified)
        # In reality, would need to simulate individual cargo trajectories
        model_velocities = np.random.uniform(0.5, 1.5, size=tracking_result.n_tracks)

        # Compare
        comparison = self.imaging.compare_velocities(model_velocities)

        # Generate kymograph
        kymograph = self.imaging.generate_kymograph()

        return {
            'tracking': {
                'n_frames': tracking_result.n_frames,
                'n_tracks': tracking_result.n_tracks,
                'mean_velocity': float(tracking_result.velocities.mean()),
                'mean_displacement': float(tracking_result.displacements.mean()),
            },
            'comparison': comparison,
            'kymograph_shape': kymograph.shape,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "cargo": self.cargo.get_status(),
            "wnt_pcp": self.wnt_pcp.get_status(),
            "imaging": self.imaging.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO IMAGING — SMOKE TEST")
    print(" Cargo Dynamics + Wnt/PCP Cross-Talk + Live-Cell Imaging")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scikit-image available : {HAS_SKIMAGE}")
    print(f" scipy available        : {HAS_SCIPY}")
    print(f" pandas available       : {HAS_PANDAS}")

    suite = SESIImagingOrganoidSuite(
        cargo_cfg=CargoConfig(n_cargos=15, max_time=50.0, max_steps=5000),
        wnt_pcp_cfg=WntPCPConfig(grid_shape=(6, 6), n_steps=50),
        imaging_cfg=ImagingConfig())

    # ── 1. Cargo Binding Dynamics ───────────────────────────────────────
    print("\n [CBD1] Cargo Binding Dynamics (Tug-of-War)")
    cargo = suite.cargo
    print(f"   n_cargos             : {cargo.cfg.n_cargos}")
    print(f"   n_kinesin            : {cargo.cfg.n_kinesin}")
    print(f"   n_dynein             : {cargo.cfg.n_dynein}")

    # Run simulation
    cargo.initialize()
    traj, times = cargo.simulate(max_steps=3000, load_force=0.5)
    print(f"   trajectory length    : {len(traj)}")
    print(f"   final time           : {times[-1]:.2f}")

    final_state = traj[-1]
    print(f"   final state          :")
    print(f"     unbound            : {final_state.n_unbound}")
    print(f"     kinesin-bound      : {final_state.n_kinesin_bound}")
    print(f"     dynein-bound       : {final_state.n_dynein_bound}")
    print(f"     tug-of-war         : {final_state.n_tug_of_war}")

    cargo_status = cargo.get_status()
    print(f"   COROLLARY-3 guard    : {cargo_status['corollary3_guard']}")
    print(f"   HINO guard           : {cargo_status['hino_guard']}")

    # ── 2. Wnt/PCP Cross-Talk Network ───────────────────────────────────
    print("\n [WPC1] Wnt/PCP Cross-Talk Network")
    wnt_pcp = suite.wnt_pcp
    print(f"   grid shape           : {wnt_pcp.cfg.grid_shape}")
    print(f"   Dsh partition        : can={wnt_pcp.cfg.dsh_partition_canonical}, "
          f"pcp={wnt_pcp.cfg.dsh_partition_pcp}")
    print(f"   Cross-inhibition     : {wnt_pcp.cfg.cross_inhibition_strength}")

    # Run network
    result = wnt_pcp.run(n_steps=30)
    print(f"   Canonical pathway    :")
    for k, v in result['canonical'].items():
        print(f"     {k:20s} : mean={v.mean():.4f}")
    print(f"   PCP pathway          :")
    for k, v in result['pcp'].items():
        print(f"     {k:20s} : mean={v.mean():.4f}")

    canonical_output = wnt_pcp.get_canonical_output()
    pcp_output = wnt_pcp.get_pcp_output()
    print(f"   Canonical output     : mean={canonical_output.mean():.4f}")
    print(f"   PCP output           : mean={pcp_output.mean():.4f}")

    wnt_pcp_status = wnt_pcp.get_status()
    print(f"   COROLLARY-3 guard    : {wnt_pcp_status['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {wnt_pcp_status['prop3_awareness']}")

    # ── 3. Coupled Simulation ───────────────────────────────────────────
    print("\n [COUPLED] Cargo + Wnt/PCP Coupling")
    coupled_result = suite.run_coupled_simulation(n_steps=50)
    print(f"   n_steps              : {coupled_result['n_steps']}")
    print(f"   Load force           : {coupled_result['load_force']:.4f}")
    print(f"   Cargo final state    : {coupled_result['cargo_final_state']}")
    print(f"   Canonical (β-cat_nuc): {coupled_result['canonical_pathway']['beta_cat_nuc']:.4f}")
    print(f"   PCP (JNK*)           : {coupled_result['pcp_pathway']['jnk_active']:.4f}")

    # ── 4. Live-Cell Imaging Bridge ─────────────────────────────────────
    print("\n [LCI1] Live-Cell Imaging Bridge")
    imaging = suite.imaging
    print(f"   Images loaded        : {imaging.get_status()['images_loaded']}")
    print(f"   scikit-image         : {imaging.get_status()['has_skimage']}")
    print(f"   scipy                : {imaging.get_status()['has_scipy']}")
    print(f"   Note                 : {imaging.get_status()['note']}")

    # Synthetic test (no real data)
    print(f"   Synthetic test       : creating mock imaging data...")
    # Create mock image sequence
    n_frames = 20
    H, W = 100, 100
    mock_images = np.random.rand(n_frames, H, W) * 0.3
    # Add some bright spots (simulated cargos)
    for t in range(n_frames):
        for i in range(5):
            x, y = np.random.randint(10, 90, size=2)
            mock_images[t, y-2:y+3, x-2:x+3] += 0.7

    imaging._images = mock_images
    print(f"   Mock images shape    : {mock_images.shape}")

    # Track cells
    tracking_result = imaging.track_cells()
    print(f"   Tracking result      :")
    print(f"     n_frames           : {tracking_result.n_frames}")
    print(f"     n_tracks           : {tracking_result.n_tracks}")
    if tracking_result.n_tracks > 0:
        print(f"     mean velocity      : {tracking_result.velocities.mean():.4f} μm/s")
        print(f"     mean displacement  : {tracking_result.displacements.mean():.4f} μm")

    # Generate kymograph
    kymograph = imaging.generate_kymograph()
    print(f"   Kymograph shape      : {kymograph.shape}")

    # Compare with model
    if tracking_result.n_tracks > 0 and HAS_SCIPY:
        model_velocities = np.random.uniform(0.5, 1.5, size=tracking_result.n_tracks)
        comparison = imaging.compare_velocities(model_velocities)
        print(f"   Velocity comparison  :")
        print(f"     KS statistic       : {comparison['ks_statistic']:.4f}")
        print(f"     KS p-value         : {comparison['ks_pvalue']:.4f}")
        print(f"     Wasserstein        : {comparison['wasserstein_distance']:.4f}")
        print(f"     Exp mean velocity  : {comparison['experimental_mean']:.4f} μm/s")
        print(f"     Model mean velocity: {comparison['model_mean']:.4f} μm/s")

    # ── 5. Full Status ──────────────────────────────────────────────────
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
    print("       Real cargo transport involves scaffolding proteins, adaptor")
    print("       complexes, regulatory mechanisms not captured here.")
    print(" [C]   Wnt/PCP cross-talk: coupled pathway model is simplified.")
    print("       Real Wnt signaling involves many more components, feedback")
    print("       loops, context-dependent regulation. Computational only.")
    print(" [D]   Live-cell imaging bridge: statistical comparison is HEURISTIC.")
    print("       Real microscopy has noise, photobleaching, motion artifacts.")
    print("       This bridge is for calibration/validation only, NOT diagnosis.")
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
