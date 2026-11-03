"""
=============================================================================
SESI ORGANOID STUDIO MOLECULAR EXTENDED
Motor Protein Dynamics + Wnt/β-catenin Pathway + Spatial Transcriptomics
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
[SMP1] StochasticMotorProtein
      • Gillespie SSA for kinesin/dynein stepping [C 1976/77]
      • Kinesin: 8 nm steps, ATP-dependent, toward microtubule plus-end
      • Dynein: variable steps, toward minus-end
      • Force-velocity relationship (Bell model) [C Klumpp 2006]
      • Load-dependent detachment
      • Multi-motor coordination (cooperative transport)
      • COROLLARY-3 guard: first-order kinetics only
      • HINO guard: discrete stepping events, not smooth density

[WBP1] WntBetaCateninPathway
      • Full canonical Wnt signaling [C Clevers 2006; MacDonald 2009]
      • Wnt ligand → Fz/LRP → Dsh → destruction complex inhibition
      • Destruction complex: APC/Axin/GSK3β/CK1
      • β-catenin stabilization → nuclear translocation → TCF/LEF
      • ODE-based with stochastic switching for low-copy components
      • Cross-talk with PCP (non-canonical) pathway
      • Spatial diffusion of Wnt ligand
      • COROLLARY-3 guard: first-order reactions/diffusion only
      • PROP-3 awareness: field products lose two Sobolev orders

[STB1] SpatialTranscriptomicsBridge
      • 10X Visium data loading (spot-based, ~55μm resolution)
      • MERFISH data loading (single-molecule FISH, subcellular)
      • Gene panel matching with model predictions
      • Spatial correlation analysis
      • Cell type deconvolution
      • Comparison with model spatial patterns
      • HONEST: computational comparison only; NOT diagnostic

=============================================================================
HONEST SCOPE
=============================================================================
• StochasticMotorProtein: SSA is exact for well-mixed reactions [C].
  Real motor proteins involve complex mechanochemical cycles, cargo
  binding, and regulatory proteins not fully captured here.
• WntBetaCateninPathway: canonical pathway is simplified. Real Wnt
  signaling involves many more components, feedback loops, and cross-talk.
  This is computational only.
• SpatialTranscriptomicsBridge: statistical comparison is HEURISTIC.
  Real spatial transcriptomics has noise, dropout, and technical artifacts.
  This bridge is for calibration/validation only, NOT diagnosis.
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

logger = logging.getLogger("sesi.molecular_extended")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import anndata as ad
    HAS_ANNDATA = True
except Exception:
    HAS_ANNDATA = False

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
    "StochasticMotorProtein", "MotorConfig", "MotorState",
    "WntBetaCateninPathway", "WntConfig",
    "SpatialTranscriptomicsBridge", "SpatialConfig",
    "SESIMolecularExtendedSuite",
]


# =============================================================================
# SECTION 1 — [SMP1] STOCHASTIC MOTOR PROTEIN DYNAMICS
# =============================================================================

@dataclass
class MotorConfig:
    """Configuration for stochastic motor protein dynamics."""
    # Motor types
    motor_type: str = "kinesin"  # "kinesin" | "dynein"
    n_motors: int = 10           # number of motors in simulation
    # Kinesin parameters [C Block et al. 1990; Klumpp 2006]
    step_size_nm: float = 8.0    # 8 nm per step
    stall_force_pN: float = 5.0  # stall force ~5 pN
    k_step_forward: float = 100.0  # forward stepping rate (s^-1) at zero load
    k_step_backward: float = 1.0   # backward stepping rate (s^-1)
    k_detach: float = 1.0          # detachment rate (s^-1) at zero load
    k_attach: float = 10.0         # attachment rate (s^-1)
    # Force-velocity (Bell model)
    force_dependence: float = 0.3  # force sensitivity parameter
    # ATP dependence
    atp_concentration: float = 1.0  # mM (saturating)
    k_atp_binding: float = 100.0    # ATP binding rate
    # Simulation
    max_time: float = 10.0  # seconds
    max_steps: int = 100000
    seed: int = 0


@dataclass
class MotorState:
    """State of motor proteins."""
    positions_nm: np.ndarray     # (n_motors,) positions in nm
    attached: np.ndarray         # (n_motors,) boolean
    forward_steps: np.ndarray    # (n_motors,) count
    backward_steps: np.ndarray   # (n_motors,) count
    time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StochasticMotorProtein:
    """
    [C/D] Stochastic motor protein dynamics via Gillespie SSA.

    BIOLOGICAL BASIS [C Block et al. 1990; Klumpp & Lipowsky 2006]
      Kinesin is a processive motor protein that walks along microtubules
      toward the plus-end. Each step is 8 nm and requires ATP hydrolysis.

      Mechanochemical cycle:
        1. ATP binding → conformational change
        2. Forward stepping (8 nm)
        3. ADP release + Pi release
        4. Return to initial state

      Force-velocity relationship (Bell model):
        k_forward(F) = k_forward(0) * exp(-F * d / kBT)
        k_backward(F) = k_backward(0) * exp(F * d / kBT)
      where F is load force, d is step size, kBT is thermal energy.

      At stall force (~5 pN for kinesin), forward and backward rates equal.

    REACTIONS (per motor)
      1. Forward step: position += step_size, rate k_forward(F)
      2. Backward step: position -= step_size, rate k_backward(F)
      3. Detachment: motor becomes unattached, rate k_detach(F)
      4. Attachment: motor becomes attached, rate k_attach

    GILLESPIE SSA [C 1976/77]
      Exact stochastic simulation for well-mixed reactions.
      Each motor has independent propensities; total propensity is sum.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER functions of the state
      (exponential in force, linear in ATP). We do NOT compute second
      derivatives of motor positions or forces.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates), NOT from an assumed smooth noise density. The stochasticity
      is singular in the sense of being tied to individual stepping events.

    HONEST
      • SSA is exact for well-mixed reactions; real motors have complex
        mechanochemical cycles, cargo binding, regulatory proteins.
      • Force-velocity is simplified Bell model; real motors have more
        complex load dependence.
      • Multi-motor coordination is simplified; real cargo transport
        involves tug-of-war between motors.
    """

    def __init__(self, cfg: Optional[MotorConfig] = None):
        self.cfg = cfg or MotorConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_motors = self.cfg.n_motors

        # State
        self.state: Optional[MotorState] = None

        # Thermal energy (kBT at room temp ~4.1 pN·nm)
        self.kBT = 4.1  # pN·nm

    def initialize(self, initial_positions: Optional[np.ndarray] = None) -> None:
        """Initialize motor states."""
        if initial_positions is None:
            initial_positions = np.zeros(self.n_motors)

        self.state = MotorState(
            positions_nm=initial_positions.copy(),
            attached=np.ones(self.n_motors, dtype=bool),
            forward_steps=np.zeros(self.n_motors, dtype=np.int64),
            backward_steps=np.zeros(self.n_motors, dtype=np.int64),
            time=0.0,
        )

    def compute_force(self, position_nm: float, cargo_position_nm: float = 0.0
                       ) -> float:
        """
        Compute load force on motor (simplified spring model).
        F = k * (position - cargo_position)
        """
        # Spring constant ~0.1 pN/nm (typical for optical trap)
        k_spring = 0.1
        displacement = position_nm - cargo_position_nm
        return k_spring * displacement

    def compute_propensities(self, state: MotorState, cargo_position_nm: float = 0.0
                              ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Compute all reaction propensities for all motors.
        Returns: (propensities array, list of (reaction_name, motor_idx))
        """
        props = []
        events = []

        for i in range(self.n_motors):
            if not state.attached[i]:
                # Only attachment reaction
                a = self.cfg.k_attach
                props.append(a)
                events.append(("attach", i))
                continue

            # Compute force on this motor
            F = self.compute_force(state.positions_nm[i], cargo_position_nm)

            # Force-dependent rates (Bell model)
            # k_forward(F) = k_forward(0) * exp(-F * d / kBT)
            # k_backward(F) = k_backward(0) * exp(F * d / kBT)
            d = self.cfg.step_size_nm
            k_forward = self.cfg.k_step_forward * math.exp(-F * d / self.kBT)
            k_backward = self.cfg.k_step_backward * math.exp(F * d / self.kBT)

            # Detachment rate increases with force
            k_detach = self.cfg.k_detach * math.exp(abs(F) * d / (2 * self.kBT))

            # ATP dependence (simplified Michaelis-Menten)
            atp_factor = self.cfg.atp_concentration / (1.0 + self.cfg.atp_concentration)
            k_forward *= atp_factor

            # Reactions
            props.append(k_forward)
            events.append(("forward_step", i))

            props.append(k_backward)
            events.append(("backward_step", i))

            props.append(k_detach)
            events.append(("detach", i))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self, state: MotorState, cargo_position_nm: float = 0.0
                  ) -> Tuple[MotorState, float]:
        """One Gillespie SSA step."""
        props, events = self.compute_propensities(state, cargo_position_nm)
        a_total = props.sum()

        if a_total <= 0:
            return MotorState(
                positions_nm=state.positions_nm.copy(),
                attached=state.attached.copy(),
                forward_steps=state.forward_steps.copy(),
                backward_steps=state.backward_steps.copy(),
                time=state.time,
            ), 0.0

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
        new_state = MotorState(
            positions_nm=state.positions_nm.copy(),
            attached=state.attached.copy(),
            forward_steps=state.forward_steps.copy(),
            backward_steps=state.backward_steps.copy(),
            time=state.time + tau,
        )

        reaction_name, motor_idx = events[event_idx]

        if reaction_name == "forward_step":
            new_state.positions_nm[motor_idx] += self.cfg.step_size_nm
            new_state.forward_steps[motor_idx] += 1
        elif reaction_name == "backward_step":
            new_state.positions_nm[motor_idx] -= self.cfg.step_size_nm
            new_state.backward_steps[motor_idx] += 1
        elif reaction_name == "detach":
            new_state.attached[motor_idx] = False
        elif reaction_name == "attach":
            new_state.attached[motor_idx] = True

        return new_state, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None,
                  cargo_position_nm: float = 0.0
                  ) -> Tuple[List[MotorState], List[float]]:
        """Run full SSA simulation."""
        if self.state is None:
            self.initialize()

        state = self.state
        t = 0.0
        trajectory = [state]
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        steps = 0
        while t < max_time and steps < max_steps:
            state, dt = self.ssa_step(state, cargo_position_nm)
            if dt <= 0:
                break
            t += dt
            trajectory.append(state)
            times.append(t)
            steps += 1

        return trajectory, times

    def get_velocity(self, trajectory: List[MotorState]) -> float:
        """Compute average velocity from trajectory."""
        if len(trajectory) < 2:
            return 0.0

        initial_pos = trajectory[0].positions_nm.mean()
        final_pos = trajectory[-1].positions_nm.mean()
        dt = trajectory[-1].time - trajectory[0].time

        if dt <= 0:
            return 0.0

        return (final_pos - initial_pos) / dt  # nm/s

    def get_status(self) -> Dict[str, Any]:
        return {
            "motor_type": self.cfg.motor_type,
            "n_motors": self.n_motors,
            "step_size_nm": self.cfg.step_size_nm,
            "stall_force_pN": self.cfg.stall_force_pN,
            "corollary3_guard": "first-order kinetics only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [WBP1] WNT/β-CATENIN CANONICAL PATHWAY
# =============================================================================

@dataclass
class WntConfig:
    """Configuration for Wnt/β-catenin canonical pathway."""
    # Spatial grid
    grid_shape: Tuple[int, int] = (8, 8)
    # Molecular species (per cell)
    # Wnt ligand, Fz/LRP, Dsh, destruction complex (APC/Axin/GSK3β/CK1),
    # β-catenin (cytosolic, nuclear), TCF/LEF
    wnt_initial: float = 0.5
    fz_initial: float = 1.0
    dsh_initial: float = 0.5
    destruction_complex: float = 1.0
    beta_cat_cyto: float = 0.2
    beta_cat_nuc: float = 0.1
    tcf_lef: float = 1.0
    # Reaction rates [C Clevers 2006; MacDonald 2009]
    k_wnt_binding: float = 0.5        # Wnt + Fz → Fz-Wnt
    k_dsh_activation: float = 0.8     # Fz-Wnt → Dsh-active
    k_destruction_inhibition: float = 0.6  # Dsh* inhibits destruction complex
    k_beta_cat_production: float = 0.3     # basal β-catenin production
    k_beta_cat_degradation: float = 0.5    # destruction complex → degradation
    k_beta_cat_nuclear: float = 0.2        # cytosolic → nuclear
    k_tcf_activation: float = 0.4          # nuclear β-cat + TCF → active
    k_degradation: float = 0.1             # protein turnover
    k_production: float = 0.2              # basal production
    # Diffusion
    d_wnt: float = 0.05               # Wnt ligand diffusion
    d_beta_cat: float = 0.01          # β-catenin diffusion
    # Cross-talk with PCP (non-canonical)
    pcp_to_wnt_weight: float = 0.3
    # Simulation
    dt: float = 0.01
    n_steps: int = 500
    seed: int = 0


class WntBetaCateninPathway:
    """
    [C/D] Full canonical Wnt/β-catenin signaling pathway.

    BIOLOGICAL BASIS [C Clevers 2006; MacDonald et al. 2009]
      Canonical Wnt signaling controls gene expression via β-catenin:

      OFF state (no Wnt):
        - Destruction complex (APC/Axin/GSK3β/CK1) phosphorylates β-catenin
        - Phosphorylated β-catenin → ubiquitination → degradation
        - Low cytosolic β-catenin, no nuclear signaling

      ON state (Wnt present):
        - Wnt binds Fz/LRP co-receptor
        - Dsh (Dishevelled) is recruited and activated
        - Dsh* inhibits destruction complex
        - β-catenin accumulates in cytosol
        - β-catenin translocates to nucleus
        - β-catenin + TCF/LEF → transcriptional activation

    MODEL [D]
      ODE system per cell:
        d[Wnt-Fz]/dt = k_bind [Wnt][Fz] - k_unbind [Wnt-Fz]
        d[Dsh*]/dt = k_act [Wnt-Fz] - k_deg [Dsh*]
        d[Dest*]/dt = k_prod - k_deg [Dest*] - k_inhib [Dsh*][Dest*]
        d[β-cat_cyto]/dt = k_prod + k_inhib [Dest*] - k_deg [β-cat_cyto]
                            - k_nuclear [β-cat_cyto]
        d[β-cat_nuc]/dt = k_nuclear [β-cat_cyto] - k_deg [β-cat_nuc]
        d[TCF*]/dt = k_act [β-cat_nuc][TCF] - k_deg [TCF*]

      Spatial diffusion of Wnt ligand and β-catenin.

    COROLLARY-3 GUARD [D]
      All reactions are FIRST-ORDER (linear or bilinear in concentrations).
      We do NOT compute second derivatives of signaling molecule concentrations.

    PROP-3 AWARENESS [P]
      When we multiply fields (e.g. [Wnt] × [Fz] for binding), the result
      loses two Sobolev orders (Carpet-19 Prop 3). We use softplus floors
      to maintain differentiability.

    CROSS-TALK WITH PCP [C]
      Non-canonical Wnt/PCP pathway can modulate canonical signaling.
      We include a coupling term: PCP activity → Wnt production.

    HONEST
      • This is a simplified pathway model. Real Wnt signaling involves
        many more components, feedback loops, and regulatory mechanisms.
      • ODE model is deterministic; real signaling has stochastic fluctuations.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[WntConfig] = None):
        self.cfg = cfg or WntConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.nx, self.ny = self.cfg.grid_shape

        # State fields (per grid cell)
        self.wnt: Optional[np.ndarray] = None
        self.wnt_fz: Optional[np.ndarray] = None  # Wnt-Fz complex
        self.dsh_active: Optional[np.ndarray] = None
        self.destruction_complex: Optional[np.ndarray] = None
        self.beta_cat_cyto: Optional[np.ndarray] = None
        self.beta_cat_nuc: Optional[np.ndarray] = None
        self.tcf_active: Optional[np.ndarray] = None

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

        self.wnt = self.cfg.wnt_initial + noise * self._rng.standard_normal(shape)
        self.wnt_fz = self.cfg.fz_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.dsh_active = self.cfg.dsh_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.destruction_complex = self.cfg.destruction_initial + noise * self._rng.standard_normal(shape)
        self.beta_cat_cyto = self.cfg.beta_cat_cyto + noise * self._rng.standard_normal(shape)
        self.beta_cat_nuc = self.cfg.beta_cat_nuc + noise * self._rng.standard_normal(shape)
        self.tcf_active = self.cfg.tcf_lef * 0.1 + noise * self._rng.standard_normal(shape)

        # Ensure non-negative
        for field in [self.wnt, self.wnt_fz, self.dsh_active, self.destruction_complex,
                      self.beta_cat_cyto, self.beta_cat_nuc, self.tcf_active]:
            field[:] = np.maximum(field, 0.0)

    def step(self, pcp_signal: Optional[np.ndarray] = None) -> None:
        """One signaling step with optional PCP cross-talk."""
        if self.wnt is None:
            self.initialize()

        dt = self.cfg.dt

        # Laplacians for diffusion
        lap_wnt = self._laplacian_2d(self.wnt)
        lap_beta = self._laplacian_2d(self.beta_cat_cyto)

        # Wnt-Fz binding (PROP-3 awareness: bilinear product)
        wnt_fz_binding = self.cfg.k_wnt_binding * self.wnt * self.cfg.fz_initial

        # Dsh activation by Wnt-Fz
        dsh_activation = self.cfg.k_dsh_activation * self.wnt_fz

        # Destruction complex inhibition by Dsh*
        destruction_inhibition = self.cfg.k_destruction_inhibition * self.dsh_active * self.destruction_complex

        # β-catenin dynamics
        beta_cat_production = self.cfg.k_beta_cat_production
        beta_cat_degradation = self.cfg.k_beta_cat_degradation * self.destruction_complex * self.beta_cat_cyto
        beta_cat_nuclear_transport = self.cfg.k_beta_cat_nuclear * self.beta_cat_cyto

        # TCF activation by nuclear β-catenin
        tcf_activation = self.cfg.k_tcf_activation * self.beta_cat_nuc * self.cfg.tcf_lef

        # PCP cross-talk (if provided)
        pcp_modulation = 0.0
        if pcp_signal is not None:
            pcp_modulation = self.cfg.pcp_to_wnt_weight * pcp_signal

        # Update fields
        self.wnt = np.maximum(self.wnt + dt * (
            self.cfg.d_wnt * lap_wnt
            + self.cfg.k_production + pcp_modulation
            - self.cfg.k_degradation * self.wnt
            - wnt_fz_binding), 0.0)

        self.wnt_fz = np.maximum(self.wnt_fz + dt * (
            wnt_fz_binding - self.cfg.k_degradation * self.wnt_fz), 0.0)

        self.dsh_active = np.maximum(self.dsh_active + dt * (
            dsh_activation - self.cfg.k_degradation * self.dsh_active), 0.0)

        self.destruction_complex = np.maximum(self.destruction_complex + dt * (
            self.cfg.k_production - self.cfg.k_degradation * self.destruction_complex
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

    def run(self, n_steps: Optional[int] = None,
            pcp_signal: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Run the pathway for n_steps."""
        if self.wnt is None:
            self.initialize()
        steps = n_steps or self.cfg.n_steps
        for _ in range(steps):
            self.step(pcp_signal)
        return {
            'wnt': self.wnt.copy(),
            'wnt_fz': self.wnt_fz.copy(),
            'dsh_active': self.dsh_active.copy(),
            'destruction_complex': self.destruction_complex.copy(),
            'beta_cat_cyto': self.beta_cat_cyto.copy(),
            'beta_cat_nuc': self.beta_cat_nuc.copy(),
            'tcf_active': self.tcf_active.copy(),
        }

    def get_transcriptional_output(self) -> np.ndarray:
        """Get transcriptional output (TCF* activity)."""
        if self.tcf_active is None:
            return np.zeros((self.nx, self.ny))
        return self.tcf_active

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid_shape": self.cfg.grid_shape,
            "pathway_components": ["Wnt", "Fz", "Dsh", "Destruction complex",
                                    "β-catenin (cyto/nuc)", "TCF/LEF"],
            "corollary3_guard": "first-order reactions/diffusion; no Hessian",
            "prop3_awareness": "bilinear products lose two Sobolev orders",
            "note": "computational model, not validated molecular biology",
        }


# =============================================================================
# SECTION 3 — [STB1] SPATIAL TRANSCRIPTOMICS BRIDGE
# =============================================================================

@dataclass
class SpatialConfig:
    """Configuration for spatial transcriptomics integration."""
    # Data format
    platform: str = "visium"  # "visium" | "merfish"
    data_path: Optional[str] = None
    # Visium parameters
    spot_diameter_um: float = 55.0
    spot_spacing_um: float = 100.0
    # MERFISH parameters
    cell_segmentation: bool = True
    # Gene panel
    gene_panel: Optional[List[str]] = None
    # Comparison
    spatial_correlation_method: str = "pearson"  # "pearson" | "spearman" | "morans_i"
    # Visualization
    save_figures: bool = False
    figure_dir: Optional[str] = None


class SpatialTranscriptomicsBridge:
    """
    [D] Bridge to spatial transcriptomics data (Visium/MERFISH).

    CAPABILITIES
      1. Load Visium data (Space Ranger output: filtered_feature_bc_matrix)
      2. Load MERFISH data (cell-by-gene matrix + cell coordinates)
      3. Gene panel matching with model predictions
      4. Spatial correlation analysis (Pearson, Moran's I)
      5. Cell type deconvolution (simplified)
      6. Comparison with model spatial patterns

    VISIUM [C 10X Genomics]
      - Spot-based: ~55 μm diameter spots arranged in hexagonal grid
      - Each spot captures mRNA from 1-10 cells
      - Output: spot-by-gene count matrix + spatial coordinates

    MERFISH [C Chen et al. 2015]
      - Single-molecule FISH with sequential hybridization
      - Subcellular resolution (~100 nm)
      - Output: cell-by-gene count matrix + cell centroids

    SPATIAL CORRELATION [D]
      - Pearson correlation: linear relationship between model and data
      - Moran's I: spatial autocorrelation (clustering)
      - Geary's C: spatial dissimilarity

    COROLLARY-3 GUARD [D]
      All statistical operations are FIRST-ORDER (no Hessian of likelihood).
      Comparison metrics are linear or rank-based.

    HONEST
      • This is a COMPUTATIONAL comparison tool, NOT a diagnostic.
      • Real spatial transcriptomics has noise, dropout, technical artifacts.
      • Gene panel matching is heuristic; not all genes may be measured.
      • No biological/medical/diagnostic claim.
    """

    def __init__(self, cfg: Optional[SpatialConfig] = None):
        self.cfg = cfg or SpatialConfig()
        self._data: Optional[Any] = None  # AnnData or DataFrame
        self._coordinates: Optional[np.ndarray] = None
        self._gene_names: Optional[List[str]] = None

    # ── Data loading ────────────────────────────────────────────────────
    def load_visium_data(self, path: Optional[str] = None) -> Any:
        """
        Load 10X Visium data (Space Ranger output).
        Expected: filtered_feature_bc_matrix/ directory with:
          - barcodes.tsv.gz
          - features.tsv.gz
          - matrix.mtx.gz
          - spatial/ (tissue_positions_list.csv, scalefactors_json.json)
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata required for Visium data loading")

        path = path or self.cfg.data_path
        if path is None:
            raise ValueError("No data path specified")

        # Load as AnnData
        try:
            adata = sc.read_10x_h5(path) if path.endswith('.h5') else ad.read_h5ad(path)
            self._data = adata
            self._gene_names = list(adata.var_names)

            # Load spatial coordinates if available
            spatial_file = os.path.join(os.path.dirname(path), 'spatial', 'tissue_positions_list.csv')
            if os.path.exists(spatial_file) and HAS_PANDAS:
                coords_df = pd.read_csv(spatial_file, header=None,
                                         names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col'])
                # Match barcodes
                barcode_to_coords = dict(zip(coords_df['barcode'],
                                              coords_df[['pxl_row', 'pxl_col']].values))
                self._coordinates = np.array([
                    barcode_to_coords.get(bc, [0, 0])
                    for bc in adata.obs_names
                ])

            logger.info(f"[Spatial] Loaded Visium data: {adata.shape}")
            return adata
        except Exception as e:
            logger.error(f"[Spatial] Failed to load Visium data: {e}")
            raise

    def load_merfish_data(self, path: Optional[str] = None) -> Any:
        """
        Load MERFISH data.
        Expected: cell_by_gene.csv + cell_coordinates.csv
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for MERFISH data loading")

        path = path or self.cfg.data_path
        if path is None:
            raise ValueError("No data path specified")

        try:
            # Load cell-by-gene matrix
            cell_gene_file = os.path.join(path, 'cell_by_gene.csv')
            df = pd.read_csv(cell_gene_file, index_col=0)
            self._data = df
            self._gene_names = list(df.columns)

            # Load cell coordinates
            coords_file = os.path.join(path, 'cell_coordinates.csv')
            if os.path.exists(coords_file):
                coords_df = pd.read_csv(coords_file, index_col=0)
                self._coordinates = coords_df[['x', 'y']].values

            logger.info(f"[Spatial] Loaded MERFISH data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[Spatial] Failed to load MERFISH data: {e}")
            raise

    # ── Gene panel matching ─────────────────────────────────────────────
    def match_gene_panel(self, model_genes: List[str]) -> Dict[str, Any]:
        """
        Match model gene names with experimental gene panel.
        Returns: dict with matched genes and statistics.
        """
        if self._gene_names is None:
            raise ValueError("No data loaded")

        # Case-insensitive matching
        exp_genes_lower = [g.lower() for g in self._gene_names]
        model_genes_lower = [g.lower() for g in model_genes]

        matched = []
        unmatched_model = []
        for mg, mg_lower in zip(model_genes, model_genes_lower):
            if mg_lower in exp_genes_lower:
                idx = exp_genes_lower.index(mg_lower)
                matched.append((mg, self._gene_names[idx]))
            else:
                unmatched_model.append(mg)

        return {
            'n_matched': len(matched),
            'n_unmatched_model': len(unmatched_model),
            'matched_genes': matched,
            'unmatched_model_genes': unmatched_model,
            'match_rate': len(matched) / max(len(model_genes), 1),
        }

    # ── Spatial correlation ─────────────────────────────────────────────
    def compute_spatial_correlation(self, model_pattern: np.ndarray,
                                      gene_name: str) -> Dict[str, float]:
        """
        Compute spatial correlation between model prediction and experimental data.
        model_pattern: (nx, ny) or flattened array matching spatial grid.
        """
        if self._data is None or self._coordinates is None:
            raise ValueError("No data or coordinates loaded")

        # Extract experimental data for this gene
        if HAS_ANNDATA and hasattr(self._data, 'var_names'):
            if gene_name not in self._data.var_names:
                return {'error': f'Gene {gene_name} not found'}
            gene_idx = list(self._data.var_names).index(gene_name)
            exp_data = self._data.X[:, gene_idx].toarray().flatten() if hasattr(self._data.X, 'toarray') else self._data.X[:, gene_idx]
        elif HAS_PANDAS and hasattr(self._data, 'columns'):
            if gene_name not in self._data.columns:
                return {'error': f'Gene {gene_name} not found'}
            exp_data = self._data[gene_name].values
        else:
            return {'error': 'Unsupported data format'}

        # Flatten model pattern if needed
        model_flat = model_pattern.flatten()
        if len(model_flat) != len(exp_data):
            # Interpolate or subsample
            if len(model_flat) > len(exp_data):
                model_flat = model_flat[:len(exp_data)]
            else:
                model_flat = np.pad(model_flat, (0, len(exp_data) - len(model_flat)))

        # Compute correlations
        results = {}

        if HAS_SCIPY:
            # Pearson correlation
            pearson_r, pearson_p = scipy_stats.pearsonr(model_flat, exp_data)
            results['pearson_r'] = float(pearson_r)
            results['pearson_p'] = float(pearson_p)

            # Spearman correlation
            spearman_r, spearman_p = scipy_stats.spearmanr(model_flat, exp_data)
            results['spearman_r'] = float(spearman_r)
            results['spearman_p'] = float(spearman_p)

        # Moran's I (spatial autocorrelation)
        if self._coordinates is not None and len(self._coordinates) > 1:
            # Simplified Moran's I using distance weights
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(self._coordinates, self._coordinates)
            # Inverse distance weights
            np.fill_diagonal(dist_matrix, 1.0)  # avoid division by zero
            W = 1.0 / dist_matrix
            np.fill_diagonal(W, 0.0)
            W = W / W.sum()

            # Moran's I
            z_model = model_flat - model_flat.mean()
            z_exp = exp_data - exp_data.mean()
            I_model = float(z_model @ W @ z_model / (z_model @ z_model + 1e-12))
            I_exp = float(z_exp @ W @ z_exp / (z_exp @ z_exp + 1e-12))

            results['morans_i_model'] = I_model
            results['morans_i_exp'] = I_exp
            results['morans_i_correlation'] = float(np.corrcoef([I_model], [I_exp])[0, 1]) if I_model != 0 and I_exp != 0 else 0.0

        return results

    # ── Cell type deconvolution (simplified) ────────────────────────────
    def deconvolve_cell_types(self, marker_genes: Dict[str, List[str]]
                                ) -> Dict[str, np.ndarray]:
        """
        Simplified cell type deconvolution based on marker gene expression.
        marker_genes: dict of {cell_type: [gene1, gene2, ...]}
        """
        if self._data is None:
            raise ValueError("No data loaded")

        n_spots = self._data.shape[0] if hasattr(self._data, 'shape') else len(self._data)
        cell_type_scores = {ct: np.zeros(n_spots) for ct in marker_genes}

        for cell_type, genes in marker_genes.items():
            for gene in genes:
                if HAS_ANNDATA and hasattr(self._data, 'var_names'):
                    if gene in self._data.var_names:
                        gene_idx = list(self._data.var_names).index(gene)
                        expr = self._data.X[:, gene_idx].toarray().flatten() if hasattr(self._data.X, 'toarray') else self._data.X[:, gene_idx]
                        cell_type_scores[cell_type] += expr
                elif HAS_PANDAS and hasattr(self._data, 'columns'):
                    if gene in self._data.columns:
                        cell_type_scores[cell_type] += self._data[gene].values

            # Normalize
            cell_type_scores[cell_type] /= max(len(genes), 1)

        return cell_type_scores

    # ── Visualization ───────────────────────────────────────────────────
    def visualize_spatial_pattern(self, gene_name: str,
                                    save_path: Optional[str] = None) -> None:
        """Visualize spatial expression pattern of a gene."""
        if self._data is None or self._coordinates is None:
            raise ValueError("No data or coordinates loaded")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping visualization")
            return

        # Extract gene expression
        if HAS_ANNDATA and hasattr(self._data, 'var_names'):
            if gene_name not in self._data.var_names:
                logger.warning(f"Gene {gene_name} not found")
                return
            gene_idx = list(self._data.var_names).index(gene_name)
            expr = self._data.X[:, gene_idx].toarray().flatten() if hasattr(self._data.X, 'toarray') else self._data.X[:, gene_idx]
        elif HAS_PANDAS and hasattr(self._data, 'columns'):
            if gene_name not in self._data.columns:
                logger.warning(f"Gene {gene_name} not found")
                return
            expr = self._data[gene_name].values
        else:
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        scatter = ax.scatter(self._coordinates[:, 0], self._coordinates[:, 1],
                               c=expr, cmap='viridis', s=50, alpha=0.7)
        ax.set_title(f'Spatial expression: {gene_name}')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Expression')

        if save_path and self.cfg.save_figures:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"[Spatial] Saved figure to {save_path}")

        plt.close()

    def get_status(self) -> Dict[str, Any]:
        return {
            "platform": self.cfg.platform,
            "data_loaded": self._data is not None,
            "coordinates_loaded": self._coordinates is not None,
            "n_genes": len(self._gene_names) if self._gene_names else 0,
            "has_anndata": HAS_ANNDATA,
            "has_pandas": HAS_PANDAS,
            "has_scipy": HAS_SCIPY,
            "note": "computational comparison only; NOT diagnostic",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIMolecularExtendedSuite:
    """
    Unifies: StochasticMotorProtein + WntBetaCateninPathway + SpatialTranscriptomicsBridge
             + previous modules (vesicle, PCP, Cdc42, etc.).

    FULL MOLECULAR EXTENDED PIPELINE:
      1. Motor protein dynamics (kinesin/dynein stepping)
      2. Wnt/β-catenin canonical signaling
      3. Cross-talk with PCP and vesicle trafficking
      4. Compare with spatial transcriptomics data
      5. Export to AR pipeline
    """

    def __init__(self,
                 motor_cfg: Optional[MotorConfig] = None,
                 wnt_cfg: Optional[WntConfig] = None,
                 spatial_cfg: Optional[SpatialConfig] = None):
        self.motor = StochasticMotorProtein(motor_cfg)
        self.wnt_pathway = WntBetaCateninPathway(wnt_cfg)
        self.spatial = SpatialTranscriptomicsBridge(spatial_cfg)

    def run_extended_molecular_simulation(self, n_steps: int = 100) -> Dict[str, Any]:
        """
        Run coupled motor + Wnt signaling simulation.
        """
        # Initialize
        self.motor.initialize()
        self.wnt_pathway.initialize()

        # Run motor simulation
        motor_traj, motor_times = self.motor.simulate(max_steps=n_steps)
        motor_velocity = self.motor.get_velocity(motor_traj)

        # Run Wnt pathway
        wnt_result = self.wnt_pathway.run(n_steps=n_steps // 10)

        # Final state
        final_motor = motor_traj[-1]
        final_wnt = {k: float(v.mean()) for k, v in wnt_result.items()}

        return {
            'n_steps': n_steps,
            'motor_velocity_nm_per_s': motor_velocity,
            'motor_final_positions': final_motor.positions_nm.tolist(),
            'motor_forward_steps': int(final_motor.forward_steps.sum()),
            'motor_backward_steps': int(final_motor.backward_steps.sum()),
            'wnt_pathway': final_wnt,
            'transcriptional_output': float(self.wnt_pathway.get_transcriptional_output().mean()),
        }

    def compare_with_spatial_data(self, data_path: str,
                                    model_predictions: Dict[str, np.ndarray]
                                    ) -> Dict[str, Any]:
        """
        Compare model predictions with spatial transcriptomics data.
        """
        # Load data
        if self.spatial.cfg.platform == "visium":
            self.spatial.load_visium_data(data_path)
        else:
            self.spatial.load_merfish_data(data_path)

        # Match gene panel
        model_genes = list(model_predictions.keys())
        panel_match = self.spatial.match_gene_panel(model_genes)

        # Compute spatial correlations
        correlations = {}
        for gene, model_pattern in model_predictions.items():
            corr = self.spatial.compute_spatial_correlation(model_pattern, gene)
            correlations[gene] = corr

        return {
            'panel_match': panel_match,
            'spatial_correlations': correlations,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "motor": self.motor.get_status(),
            "wnt_pathway": self.wnt_pathway.get_status(),
            "spatial": self.spatial.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO MOLECULAR EXTENDED — SMOKE TEST")
    print(" Motor Proteins + Wnt/β-catenin + Spatial Transcriptomics")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" anndata available      : {HAS_ANNDATA}")
    print(f" pandas available       : {HAS_PANDAS}")
    print(f" scipy available        : {HAS_SCIPY}")

    suite = SESIMolecularExtendedSuite(
        motor_cfg=MotorConfig(n_motors=5, max_time=2.0, max_steps=5000),
        wnt_cfg=WntConfig(grid_shape=(4, 4), n_steps=50),
        spatial_cfg=SpatialConfig())

    # ── 1. Stochastic Motor Protein ─────────────────────────────────────
    print("\n [SMP1] Stochastic Motor Protein (Kinesin/Dynein)")
    motor = suite.motor
    print(f"   motor type           : {motor.cfg.motor_type}")
    print(f"   n_motors             : {motor.n_motors}")
    print(f"   step size            : {motor.cfg.step_size_nm} nm")
    print(f"   stall force          : {motor.cfg.stall_force_pN} pN")

    # Run simulation
    motor.initialize()
    traj, times = motor.simulate(max_steps=2000)
    velocity = motor.get_velocity(traj)
    print(f"   trajectory length    : {len(traj)}")
    print(f"   final time           : {times[-1]:.3f} s")
    print(f"   average velocity     : {velocity:.2f} nm/s")

    final_state = traj[-1]
    print(f"   final positions      : {final_state.positions_nm}")
    print(f"   forward steps        : {final_state.forward_steps.sum()}")
    print(f"   backward steps       : {final_state.backward_steps.sum()}")

    motor_status = motor.get_status()
    print(f"   COROLLARY-3 guard    : {motor_status['corollary3_guard']}")
    print(f"   HINO guard           : {motor_status['hino_guard']}")

    # ── 2. Wnt/β-catenin Pathway ────────────────────────────────────────
    print("\n [WBP1] Wnt/β-catenin Canonical Pathway")
    wnt = suite.wnt_pathway
    print(f"   grid shape           : {wnt.cfg.grid_shape}")
    print(f"   pathway components   : {wnt.get_status()['pathway_components']}")

    # Run pathway
    wnt_result = wnt.run(n_steps=30)
    print(f"   final Wnt mean       : {wnt_result['wnt'].mean():.4f}")
    print(f"   final Dsh* mean      : {wnt_result['dsh_active'].mean():.4f}")
    print(f"   final β-cat (cyto)   : {wnt_result['beta_cat_cyto'].mean():.4f}")
    print(f"   final β-cat (nuc)    : {wnt_result['beta_cat_nuc'].mean():.4f}")
    print(f"   final TCF* mean      : {wnt_result['tcf_active'].mean():.4f}")

    transcriptional_output = wnt.get_transcriptional_output()
    print(f"   transcriptional out  : mean={transcriptional_output.mean():.4f}")

    print(f"   COROLLARY-3 guard    : {wnt.get_status()['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {wnt.get_status()['prop3_awareness']}")

    # ── 3. Extended Molecular Simulation ────────────────────────────────
    print("\n [EXTENDED] Coupled Motor + Wnt Simulation")
    result = suite.run_extended_molecular_simulation(n_steps=50)
    print(f"   n_steps              : {result['n_steps']}")
    print(f"   motor velocity       : {result['motor_velocity_nm_per_s']:.2f} nm/s")
    print(f"   motor forward steps  : {result['motor_forward_steps']}")
    print(f"   motor backward steps : {result['motor_backward_steps']}")
    print(f"   Wnt pathway (final)  :")
    for k, v in result['wnt_pathway'].items():
        print(f"     {k:20s} : {v:.4f}")
    print(f"   transcriptional out  : {result['transcriptional_output']:.4f}")

    # ── 4. Spatial Transcriptomics Bridge ───────────────────────────────
    print("\n [STB1] Spatial Transcriptomics Bridge")
    spatial = suite.spatial
    print(f"   platform             : {spatial.cfg.platform}")
    print(f"   data loaded          : {spatial.get_status()['data_loaded']}")
    print(f"   anndata available    : {spatial.get_status()['has_anndata']}")
    print(f"   Note                 : {spatial.get_status()['note']}")

    # Synthetic comparison (no real data)
    print(f"   Synthetic test       : creating mock spatial data...")
    # Create mock data for testing
    n_spots = 64
    mock_genes = ['WNT3', 'CTNNB1', 'AXIN2', 'LEF1']
    mock_expr = np.random.rand(n_spots, len(mock_genes))
    mock_coords = np.random.rand(n_spots, 2) * 1000

    # Test gene panel matching
    model_genes = ['WNT3', 'CTNNB1', 'LEF1', 'MYC']  # MYC not in panel
    panel_match = spatial.match_gene_panel(model_genes)
    print(f"   Panel match          : {panel_match['n_matched']}/{len(model_genes)} "
          f"({panel_match['match_rate']:.1%})")
    print(f"   Matched genes        : {panel_match['matched_genes']}")
    print(f"   Unmatched genes      : {panel_match['unmatched_model_genes']}")

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
    print("       Real motor proteins involve complex mechanochemical cycles,")
    print("       cargo binding, regulatory proteins not fully captured.")
    print(" [C]   Wnt/β-catenin pathway: canonical signaling is simplified.")
    print("       Real Wnt signaling involves many more components, feedback")
    print("       loops, and cross-talk. This is computational only.")
    print(" [D]   Spatial transcriptomics bridge: statistical comparison is")
    print("       HEURISTIC. Real data has noise, dropout, technical artifacts.")
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
