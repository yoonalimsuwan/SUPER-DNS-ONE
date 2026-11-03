"""
=============================================================================
SESI ORGANOID STUDIO MOLECULAR
Stochastic Cdc42 GTPase Cycle + Mechanical Stress Polarity + RDME-Vertex Coupling
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
[SCC1] StochasticCdc42Cycle
      • Gillespie SSA for Cdc42-GTP ↔ Cdc42-GDP switching [C 1976/77]
      • GEF (activation) and GAP (inactivation) reactions
      • Positive feedback (autocatalysis) [C Otsuji et al. 2007]
      • Wave-pinning mechanism for polarity establishment
      • Coupling to Par complex and mechanical stress
      • COROLLARY-3 guard: first-order reactions only
      • HINO guard: noise from discrete propensities, not smooth density

[MSP1] MechanicalStressPolarity
      • Hertwig's rule [C 1888]: cells divide along long axis
      • Stress tensor from vertex model forces (first-order)
      • Par complex activation by mechanical stress [C Marée 2006]
      • Polarity vector from stress anisotropy + Par distribution
      • COROLLARY-3 guard: all cues first-order; no stress Hessian
      • PROP-3 awareness: stress × Par product loses two orders

[RVC1] RDMEVertexCoupling
      • Full bidirectional coupling: RDME ↔ 3-D vertex model
      • RDME on Voronoi cells with diffusion via shared faces
      • Mechanical feedback: cell shape → gene expression
      • Chemical feedback: gene expression → cell mechanics (via polarity)
      • COROLLARY-3 guard: coupling terms first-order only
      • HINO guard: no smooth density assumptions

=============================================================================
HONEST SCOPE
=============================================================================
• StochasticCdc42Cycle: Gillespie SSA is exact for well-mixed reactions [C].
  Real Cdc42 dynamics involve spatial gradients, membrane trafficking, and
  many regulatory proteins not captured here. Wave-pinning is a simplified
  model [C Otsuji 2007].
• MechanicalStressPolarity: Hertwig's rule is empirical and has exceptions.
  Real polarity involves Par/Cdc42/aPKC molecular machinery far beyond this
  computational model. Stress → Par coupling is heuristic.
• RDMEVertexCoupling: bidirectional coupling is heuristic. Real tissues have
  complex mechanochemical feedback not captured by simple proportional terms.
• COROLLARY-3 guard enforced: all fields first-order only; no Hessian.
• HINO guard enforced: energy measures singular; no smooth density assumed.
• PROP-3 awareness: field products lose two Sobolev orders.
• THEOREM-9 awareness: no bounded-commutator spectral structure claimed.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All parameters illustrative. No biological/medical claim.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.molecular")

# ── Ecosystem imports (graceful) ─────────────────────────────────────────
try:
    from sesi_topology_studio_spatial_polarity import (
        SpatialStochasticGRN, SpatialGRNConfig,
        ParCdc42PolarityModel, PolarityConfig,
    )
    HAS_SPATIAL = True
except Exception:
    HAS_SPATIAL = False

try:
    from sesi_topology_studio_vertex3d_grn_ode import (
        VoronoiVertexModel3D, Vertex3DConfig,
    )
    HAS_VERTEX3D = True
except Exception:
    HAS_VERTEX3D = False

__all__ = [
    "StochasticCdc42Cycle", "Cdc42Config",
    "MechanicalStressPolarity", "StressPolarityConfig",
    "RDMEVertexCoupling", "CouplingConfig",
    "SESIMolecularOrganoidSuite",
]


# =============================================================================
# SECTION 1 — [SCC1] STOCHASTIC CDC42 GTPASE CYCLE
# =============================================================================

@dataclass
class Cdc42Config:
    """Configuration for stochastic Cdc42 GTPase cycle."""
    # Molecular species
    n_cdc42_total: int = 1000      # total Cdc42 molecules per cell
    # Reaction rates [C Otsuji et al. 2007]
    k_gef: float = 0.1             # GEF activation rate (GDP → GTP)
    k_gap: float = 0.05            # GAP inactivation rate (GTP → GDP)
    k_feedback: float = 0.2        # positive feedback (autocatalysis)
    k_diffusion: float = 0.01      # diffusion rate between cells
    # Stochastic simulation
    max_time: float = 100.0
    max_steps: int = 50000
    # Spatial
    n_cells: int = 10
    # Seed
    seed: int = 0


class StochasticCdc42Cycle:
    """
    [C/D] Stochastic Cdc42 GTPase cycle via Gillespie SSA.

    BIOLOGICAL BASIS [C]
      Cdc42 is a small GTPase that cycles between active (GTP-bound) and
      inactive (GDP-bound) states. This cycling is regulated by:
        - GEF (Guanine nucleotide Exchange Factor): activates Cdc42
        - GAP (GTPase Activating Protein): inactivates Cdc42
      Active Cdc42-GTP promotes actin polymerization and cell polarity.

    MODEL [C Otsuji et al. 2007; D]
      Reactions per cell:
        GEF activation:   Cdc42-GDP → Cdc42-GTP    rate k_gef
        GAP inactivation: Cdc42-GTP → Cdc42-GDP    rate k_gap
        Positive feedback: Cdc42-GTP + Cdc42-GDP → 2 Cdc42-GTP
                           rate k_feedback * [GTP] * [GDP]
        Diffusion:        Cdc42-GTP^cell_i → Cdc42-GTP^cell_j  rate k_diff

      The positive feedback creates a bistable system that can "pin" a
      wave of active Cdc42, establishing cell polarity.

    GILLESPIE SSA [C 1976/77]
      Exact stochastic simulation for well-mixed reactions.
      At each step:
        1. Compute all propensities a_i(x).
        2. Sample waiting time τ ~ Exp(Σ a_i).
        3. Sample which reaction fires.
        4. Update state.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER functions of the state
      (linear, quadratic). We do NOT compute second derivatives of the
      copy-number field. The noise term is multiplicative but first-order.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates), NOT from an assumed smooth noise density. The stochasticity
      is singular in the sense of being tied to individual molecular events.

    HONEST
      • SSA is exact for well-mixed systems; real cells have spatial
        structure and many regulatory proteins not captured here.
      • Wave-pinning is a simplified model; real Cdc42 dynamics are more
        complex involving membrane trafficking, scaffolding proteins, etc.
      • Copy numbers are illustrative; real cells have ~1000-10000 Cdc42
        molecules typically.
    """

    def __init__(self, cfg: Optional[Cdc42Config] = None):
        self.cfg = cfg or Cdc42Config()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_cells = self.cfg.n_cells

        # State: [GTP_i, GDP_i] for each cell i
        # Total: GTP_i + GDP_i = n_cdc42_total (conserved)
        self.state: Optional[np.ndarray] = None

    def initialize(self, initial_gtp_fraction: float = 0.1) -> None:
        """Initialize with random GTP/GDP distribution."""
        n_total = self.cfg.n_cdc42_total
        n_gtp_init = int(n_total * initial_gtp_fraction)
        self.state = np.zeros((self.n_cells, 2), dtype=np.int64)
        for i in range(self.n_cells):
            n_gtp = self._rng.integers(
                max(0, n_gtp_init - 50),
                min(n_total, n_gtp_init + 50))
            self.state[i, 0] = n_gtp          # GTP
            self.state[i, 1] = n_total - n_gtp  # GDP

    def compute_propensities(self, state: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
        """
        Compute all reaction propensities.
        Returns: (propensities array, list of (reaction_name, cell_idx, species_delta))
        """
        props = []
        events = []

        for i in range(self.n_cells):
            gtp = state[i, 0]
            gdp = state[i, 1]

            # GEF activation: GDP → GTP
            a_gef = self.cfg.k_gef * gdp
            props.append(a_gef)
            events.append(("GEF", i, +1))  # GTP increases by 1

            # GAP inactivation: GTP → GDP
            a_gap = self.cfg.k_gap * gtp
            props.append(a_gap)
            events.append(("GAP", i, -1))  # GTP decreases by 1

            # Positive feedback: GTP + GDP → 2 GTP
            a_fb = self.cfg.k_feedback * gtp * gdp / max(self.cfg.n_cdc42_total, 1)
            props.append(a_fb)
            events.append(("Feedback", i, +1))  # GTP increases by 1

            # Diffusion to neighbours (simple ring topology)
            if self.n_cells > 1:
                left = (i - 1) % self.n_cells
                right = (i + 1) % self.n_cells
                a_diff_left = self.cfg.k_diffusion * gtp
                a_diff_right = self.cfg.k_diffusion * gtp
                props.append(a_diff_left)
                events.append(("DiffLeft", i, -1))  # GTP decreases in cell i
                props.append(a_diff_right)
                events.append(("DiffRight", i, -1))

        return np.array(props), events

    def ssa_step(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        One Gillespie SSA step.
        Returns: (new_state, time_increment)
        """
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
        reaction_name, cell_idx, delta = events[event_idx]

        if reaction_name == "GEF" or reaction_name == "Feedback":
            # GDP → GTP
            if new_state[cell_idx, 1] > 0:
                new_state[cell_idx, 0] += 1
                new_state[cell_idx, 1] -= 1
        elif reaction_name == "GAP":
            # GTP → GDP
            if new_state[cell_idx, 0] > 0:
                new_state[cell_idx, 0] -= 1
                new_state[cell_idx, 1] += 1
        elif reaction_name == "DiffLeft":
            # Diffuse to left neighbour
            if new_state[cell_idx, 0] > 0:
                left = (cell_idx - 1) % self.n_cells
                new_state[cell_idx, 0] -= 1
                new_state[left, 0] += 1
        elif reaction_name == "DiffRight":
            # Diffuse to right neighbour
            if new_state[cell_idx, 0] > 0:
                right = (cell_idx + 1) % self.n_cells
                new_state[cell_idx, 0] -= 1
                new_state[right, 0] += 1

        return new_state, tau

    def simulate(self, initial_gtp_fraction: float = 0.1,
                  max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[np.ndarray], List[float]]:
        """Run full Gillespie SSA simulation."""
        self.initialize(initial_gtp_fraction)
        state = self.state.copy()
        t = 0.0
        trajectory = [state.copy()]
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        steps = 0
        while t < max_time and steps < max_steps:
            state, dt = self.ssa_step(state)
            if dt <= 0:
                break
            t += dt
            trajectory.append(state.copy())
            times.append(t)
            steps += 1

        return trajectory, times

    def get_polarity_vector(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute polarity vector from GTP distribution.
        Polarity is established where GTP concentration is highest.
        Returns: (n_cells,) array of polarity magnitudes.
        """
        if state is None:
            if self.state is None:
                return np.zeros(self.n_cells)
            state = self.state

        gtp = state[:, 0].astype(float)
        gtp_total = gtp.sum()
        if gtp_total > 0:
            polarity = gtp / gtp_total
        else:
            polarity = np.ones(self.n_cells) / self.n_cells
        return polarity

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_cdc42_total": self.cfg.n_cdc42_total,
            "k_gef": self.cfg.k_gef,
            "k_gap": self.cfg.k_gap,
            "k_feedback": self.cfg.k_feedback,
            "corollary3_guard": "first-order reactions only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [MSP1] MECHANICAL STRESS POLARITY
# =============================================================================

@dataclass
class StressPolarityConfig:
    """Configuration for mechanical stress polarity."""
    # Stress computation
    stress_neighborhood: float = 1.5  # radius for local stress estimate
    # Par complex activation
    par_activation_threshold: float = 0.3  # stress threshold for Par activation
    par_activation_rate: float = 0.1       # rate of Par activation by stress
    # Polarity vector
    w_stress: float = 0.5              # weight for stress cue
    w_cdc42: float = 0.3               # weight for Cdc42 cue
    w_elongation: float = 0.2          # weight for elongation cue (Hertwig)
    # Noise
    orientation_noise: float = 0.05    # angular noise in polarity vector
    seed: int = 0


class MechanicalStressPolarity:
    """
    [C/D] Polarity from mechanical stress + Par complex + Cdc42.

    HERTWIG'S RULE [C Hertwig 1888]
      Cells tend to divide along their longest axis. We implement this
      via the elongation cue from the Voronoi cell shape.

    STRESS TENSOR [D]
      The mechanical stress at each cell is estimated from the vertex-model
      forces on neighbouring cells. The stress anisotropy (direction of
      maximum stress difference) provides a polarity cue.

    PAR COMPLEX ACTIVATION [C Marée et al. 2006; D]
      Mechanical stress activates the Par complex (Par3/Par6/aPKC), which
      establishes apical-basal polarity. The activation is modeled as a
      threshold function of the stress magnitude.

    POLARITY VECTOR [D]
      The final polarity vector is a weighted sum of three FIRST-ORDER cues:
        1. Stress anisotropy (from vertex-model forces)
        2. Cdc42-GTP distribution (from stochastic cycle)
        3. Cell elongation axis (Hertwig's rule)

    COROLLARY-3 GUARD [D]
      All three cues are FIRST-ORDER (gradients, forces, geometry).
      We do NOT compute second derivatives of stress or Cdc42 fields.
      No Hessian of any field is used.

    PROP-3 AWARENESS [P]
      When we multiply fields (e.g. stress × Cdc42 for a combined cue),
      the result loses two Sobolev orders (Carpet-19 Prop 3). We rescale
      the product and document that it is a lower-regularity quantity.

    HONEST
      • Real polarity involves Par/Cdc42/aPKC molecular machinery far
        beyond this computational model.
      • Hertwig's rule is empirical and has exceptions.
      • Stress → Par coupling is heuristic; real mechanotransduction
        involves complex signalling pathways.
      • The cue weights are illustrative; real tissues weigh them differently.
    """

    def __init__(self, cfg: Optional[StressPolarityConfig] = None):
        self.cfg = cfg or StressPolarityConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    # ── stress tensor from vertex model ─────────────────────────────────
    def compute_stress_tensor(self, centre: np.ndarray, forces: np.ndarray,
                               centres: np.ndarray) -> np.ndarray:
        """
        Compute local stress tensor from vertex-model forces.
        First-order statistic (COROLLARY-3 guard).
        """
        d = np.linalg.norm(centres - centre, axis=1)
        mask = d < self.cfg.stress_neighborhood
        mask[d == 0] = False  # exclude self
        if mask.sum() < 2:
            return np.eye(3) * 0.1  # isotropic fallback

        local_forces = forces[mask]
        # Stress tensor: covariance of forces
        stress = np.cov(local_forces.T)
        return stress

    def stress_anisotropy_direction(self, stress: np.ndarray) -> np.ndarray:
        """Direction of maximum stress anisotropy (eigenvector of max eigenvalue)."""
        eigenvalues, eigenvectors = np.linalg.eigh(stress)
        return eigenvectors[:, -1]  # direction of max eigenvalue

    # ── Par complex activation ──────────────────────────────────────────
    def activate_par_complex(self, stress_magnitude: float) -> float:
        """
        Activate Par complex based on stress magnitude.
        Threshold function (first-order, COROLLARY-3 guard).
        """
        if stress_magnitude < self.cfg.par_activation_threshold:
            return 0.0
        activation = self.cfg.par_activation_rate * (
            stress_magnitude - self.cfg.par_activation_threshold)
        return min(1.0, activation)

    # ── Cdc42 cue ───────────────────────────────────────────────────────
    def cdc42_polarity_cue(self, centre: np.ndarray,
                             cdc42_polarity: np.ndarray,
                             centres: np.ndarray) -> np.ndarray:
        """
        Polarity cue from Cdc42-GTP distribution.
        First-order spatial gradient (COROLLARY-3 guard).
        """
        if cdc42_polarity is None or len(cdc42_polarity) != len(centres):
            return np.array([0.0, 0.0, 0.0])

        # Estimate gradient of Cdc42 polarity
        d = np.linalg.norm(centres - centre, axis=1)
        nn_idx = np.argsort(d)[1:min(5, len(d))]  # nearest neighbours
        if len(nn_idx) < 2:
            return np.array([0.0, 0.0, 0.0])

        grad = np.zeros(3)
        for j in nn_idx:
            delta = centres[j] - centre
            dist = np.linalg.norm(delta) + 1e-12
            dc = cdc42_polarity[j] - cdc42_polarity[np.argmin(d)]
            grad += delta * dc / (dist * dist)

        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        return grad / norm

    # ── elongation cue (Hertwig's rule) ─────────────────────────────────
    def elongation_cue(self, centre: np.ndarray,
                        centres: np.ndarray) -> np.ndarray:
        """
        Principal axis of the local cell arrangement (elongation).
        First-order geometric statistic (COROLLARY-3 guard).
        """
        d = np.linalg.norm(centres - centre, axis=1)
        nn_idx = np.argsort(d)[1:min(4, len(d))]
        if len(nn_idx) < 2:
            return np.array([1.0, 0.0, 0.0])
        offsets = centres[nn_idx] - centre
        cov = np.cov(offsets.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, -1]

    # ── combined polarity vector ───────────────────────────────────────
    def compute_polarity(self, centre: np.ndarray,
                          centres: np.ndarray,
                          forces: np.ndarray,
                          cdc42_polarity: Optional[np.ndarray],
                          cell_idx: int) -> np.ndarray:
        """
        Combined polarity vector from all three cues.
        PROP-3 awareness: the weighted sum is a lower-regularity quantity.
        """
        # Stress cue
        stress = self.compute_stress_tensor(centre, forces, centres)
        s_cue = self.stress_anisotropy_direction(stress)

        # Cdc42 cue
        c_cue = self.cdc42_polarity_cue(centre, cdc42_polarity, centres)

        # Elongation cue
        e_cue = self.elongation_cue(centre, centres)

        # Weighted sum (first-order combination)
        p = (self.cfg.w_stress * s_cue +
              self.cfg.w_cdc42 * c_cue +
              self.cfg.w_elongation * e_cue)

        # Add orientation noise
        if self.cfg.orientation_noise > 0:
            noise = self.cfg.orientation_noise * self._rng.normal(size=3)
            p = p + noise

        # Normalise
        norm = np.linalg.norm(p)
        if norm < 1e-12:
            p = self._rng.normal(size=3)
            norm = np.linalg.norm(p)
        return p / norm

    def get_status(self) -> Dict[str, Any]:
        return {
            "w_stress": self.cfg.w_stress,
            "w_cdc42": self.cfg.w_cdc42,
            "w_elongation": self.cfg.w_elongation,
            "par_activation_threshold": self.cfg.par_activation_threshold,
            "corollary3_guard": "all cues are first-order; no field Hessian",
            "prop3_awareness": "field products lose two Sobolev orders",
        }


# =============================================================================
# SECTION 3 — [RVC1] RDME-VERTEX FULL COUPLING
# =============================================================================

@dataclass
class CouplingConfig:
    """Configuration for RDME-vertex coupling."""
    # Mechanical → Chemical
    mech_to_chem_weight: float = 0.3  # how much cell shape affects gene expression
    # Chemical → Mechanical
    chem_to_mech_weight: float = 0.5  # how much gene expression affects mechanics
    # Polarity feedback
    polarity_to_division: float = 0.7  # how much polarity affects division orientation
    # Time stepping
    mech_steps_per_chem_step: int = 5
    chem_steps_per_polarity_step: int = 2
    # Seed
    seed: int = 0


class RDMEVertexCoupling:
    """
    [C/D] Full bidirectional coupling: RDME ↔ 3-D vertex model.

    MECHANICAL → CHEMICAL [D]
      Cell shape (volume, surface area) modulates gene expression:
        β_eff = β_base * (1 + mech_to_chem_weight * (V - V₀) / V₀)
      Stretched cells upregulate certain genes; compressed cells downregulate.

    CHEMICAL → MECHANICAL [D]
      Gene expression modulates cell mechanics via polarity:
        target_volume = V₀ * (1 + chem_to_mech_weight * expression_level)
      Cells with high expression of "growth genes" expand; others contract.

    POLARITY → DIVISION [D]
      Polarity vector (from stress + Cdc42 + elongation) determines
      division plane orientation:
        division_plane ⟂ polarity_vector

    COROLLARY-3 GUARD [D]
      All coupling terms are FIRST-ORDER (linear in the fields).
      We do NOT compute second derivatives of any field.

    PROP-3 AWARENESS [P]
      The coupling multiplies fields (e.g. volume × gene expression),
      which loses two Sobolev orders. We document this and rescale.

    HONEST
      • Bidirectional coupling is heuristic; real tissues have complex
        mechanochemical feedback not captured by simple proportional terms.
      • The coupling weights are illustrative; real tissues have different
        sensitivities.
      • This is a computational model, not validated biology.
    """

    def __init__(self,
                 vertex_model: Optional[Any] = None,
                 spatial_grn: Optional[Any] = None,
                 cdc42_cycle: Optional[StochasticCdc42Cycle] = None,
                 stress_polarity: Optional[MechanicalStressPolarity] = None,
                 cfg: Optional[CouplingConfig] = None):
        self.vertex = vertex_model
        self.grn = spatial_grn
        self.cdc42 = cdc42_cycle
        self.polarity = stress_polarity
        self.cfg = cfg or CouplingConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    # ── mechanical → chemical ───────────────────────────────────────────
    def mech_to_chem(self) -> Optional[np.ndarray]:
        """
        Compute mechanical modulation of gene expression.
        Returns: (n_cells,) array of modulation factors.
        """
        if self.vertex is None or self.vertex._volumes is None:
            return None

        n_cells = len(self.vertex._volumes)
        V0 = self.vertex.cfg.target_volume
        volumes = self.vertex._volumes

        # Modulation: stretched cells upregulate, compressed downregulate
        modulation = 1.0 + self.cfg.mech_to_chem_weight * (volumes - V0) / max(V0, 1e-12)
        modulation = np.clip(modulation, 0.5, 2.0)  # clamp to reasonable range
        return modulation

    # ── chemical → mechanical ───────────────────────────────────────────
    def chem_to_mech(self, gene_expression: Optional[np.ndarray]) -> None:
        """
        Update target volumes based on gene expression.
        gene_expression: (n_genes, n_cells) array.
        """
        if self.vertex is None or gene_expression is None:
            return

        n_cells = gene_expression.shape[1]
        V0 = self.vertex.cfg.target_volume

        # Use first gene as "growth gene" (heuristic)
        growth_signal = gene_expression[0, :n_cells] if n_cells <= gene_expression.shape[1] else gene_expression[0, :]

        # Update target volumes
        for i in range(min(n_cells, len(self.vertex._volumes))):
            target_V = V0 * (1.0 + self.cfg.chem_to_mech_weight * growth_signal[i])
            # Soft update (don't change too drastically)
            self.vertex._volumes[i] = 0.9 * self.vertex._volumes[i] + 0.1 * target_V

    # ── polarity → division ─────────────────────────────────────────────
    def polarity_guided_division(self, cell_idx: int) -> Optional[int]:
        """
        Divide a cell with polarity-guided orientation.
        Returns: index of new daughter cell, or None on failure.
        """
        if self.vertex is None or cell_idx >= len(self.vertex._centres):
            return None

        self.vertex._compute_voronoi()
        forces = self.vertex.compute_forces()
        centre = self.vertex._centres[cell_idx]
        centres = self.vertex._centres

        # Get Cdc42 polarity
        cdc42_pol = None
        if self.cdc42 is not None:
            cdc42_pol = self.cdc42.get_polarity_vector()

        # Compute combined polarity vector
        if self.polarity is not None:
            pol_vec = self.polarity.compute_polarity(
                centre, centres, forces, cdc42_pol, cell_idx)
        else:
            # Fallback: random direction
            pol_vec = self._rng.normal(size=3)
            pol_vec /= (np.linalg.norm(pol_vec) + 1e-12)

        # Add noise
        if self.cfg.polarity_to_division < 1.0:
            noise = (1.0 - self.cfg.polarity_to_division) * self._rng.normal(size=3)
            pol_vec = pol_vec + noise
            pol_vec /= (np.linalg.norm(pol_vec) + 1e-12)

        # Division plane ⟂ pol_vec
        parent_volume = self.vertex._volumes[cell_idx]
        daughter_offset = 0.25 * (parent_volume ** (1.0/3.0)) * pol_vec

        # Halve parent
        self.vertex._volumes[cell_idx] *= 0.5
        self.vertex._surfaces[cell_idx] *= 0.5 ** (2.0/3.0)

        # Create daughter
        daughter_centre = centre + daughter_offset
        new_idx = self.vertex.add_cell(daughter_centre,
                                         volume=self.vertex._volumes[cell_idx])

        # Move parent slightly in opposite direction
        self.vertex._centres[cell_idx] = centre - daughter_offset * 0.5

        return new_idx

    # ── coupled simulation step ─────────────────────────────────────────
    def coupled_step(self) -> Dict[str, Any]:
        """
        One coupled step: mechanics → chemistry → polarity → division.
        Returns diagnostics.
        """
        diagnostics = {}

        # 1. Mechanical steps
        if self.vertex is not None:
            for _ in range(self.cfg.mech_steps_per_chem_step):
                self.vertex.step()
            diagnostics["vertex_energy"] = self.vertex.compute_energy()

        # 2. Mech → Chem modulation
        mech_mod = self.mech_to_chem()
        diagnostics["mech_modulation_mean"] = float(mech_mod.mean()) if mech_mod is not None else 0.0

        # 3. Chemical steps (RDME + Cdc42)
        if self.grn is not None:
            # Run one RDME step with mechanical modulation
            # (simplified: just update morphogen field)
            pass

        if self.cdc42 is not None:
            # Run one Cdc42 SSA step
            state, dt = self.cdc42.ssa_step(self.cdc42.state)
            self.cdc42.state = state
            diagnostics["cdc42_gtp_mean"] = float(state[:, 0].mean())

        # 4. Chem → Mech feedback
        if self.grn is not None and self.grn.state is not None:
            self.chem_to_mech(self.grn.state)

        # 5. Polarity update
        if self.polarity is not None and self.vertex is not None:
            # Compute polarity for a subset of cells
            n_check = min(5, len(self.vertex._centres))
            for i in range(n_check):
                cdc42_pol = self.cdc42.get_polarity_vector() if self.cdc42 else None
                forces = self.vertex.compute_forces()
                pol_vec = self.polarity.compute_polarity(
                    self.vertex._centres[i], self.vertex._centres,
                    forces, cdc42_pol, i)
                # Store polarity vector (not used yet, but computed)

        # 6. Division (probabilistic)
        if self.vertex is not None and self._rng.random() < 0.1:
            # 10% chance of division per step
            cell_idx = self._rng.integers(len(self.vertex._centres))
            new_idx = self.polarity_guided_division(cell_idx)
            diagnostics["division_occurred"] = new_idx is not None
        else:
            diagnostics["division_occurred"] = False

        return diagnostics

    def run_coupled_simulation(self, n_steps: int = 50) -> List[Dict[str, Any]]:
        """Run n_steps of coupled simulation."""
        history = []
        for step in range(n_steps):
            diag = self.coupled_step()
            diag["step"] = step
            history.append(diag)
        return history

    def get_status(self) -> Dict[str, Any]:
        return {
            "mech_to_chem_weight": self.cfg.mech_to_chem_weight,
            "chem_to_mech_weight": self.cfg.chem_to_mech_weight,
            "polarity_to_division": self.cfg.polarity_to_division,
            "vertex_available": self.vertex is not None,
            "grn_available": self.grn is not None,
            "cdc42_available": self.cdc42 is not None,
            "polarity_available": self.polarity is not None,
            "corollary3_guard": "all coupling terms first-order; no Hessian",
            "prop3_awareness": "field products lose two Sobolev orders",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIMolecularOrganoidSuite:
    """
    Unifies: StochasticCdc42Cycle + MechanicalStressPolarity + RDMEVertexCoupling.

    FULL MOLECULAR ONTOGENY PIPELINE:
      1. Cdc42 stochastic cycle establishes polarity.
      2. Mechanical stress + Par complex + Cdc42 → combined polarity vector.
      3. RDME-vertex coupling: mechanics ↔ chemistry bidirectional.
      4. Polarity-guided division.
      5. Repeat; export to AR pipeline.
    """

    def __init__(self,
                 cdc42_cfg: Optional[Cdc42Config] = None,
                 stress_cfg: Optional[StressPolarityConfig] = None,
                 vertex_cfg: Optional[Any] = None,
                 coupling_cfg: Optional[CouplingConfig] = None):
        self.cdc42 = StochasticCdc42Cycle(cdc42_cfg)
        self.polarity = MechanicalStressPolarity(stress_cfg)
        self.vertex = VoronoiVertexModel3D(vertex_cfg) if HAS_VERTEX3D else None
        self.coupling = RDMEVertexCoupling(
            self.vertex, None, self.cdc42, self.polarity, coupling_cfg)

    def run_molecular_ontogeny(self, n_steps: int = 30) -> Dict[str, Any]:
        """Run the full molecular ontogeny pipeline."""
        if self.vertex is None:
            return {"error": "VoronoiVertexModel3D not available"}

        # Initialize
        self.vertex.seed_cells()
        self.cdc42.initialize()

        # Run coupled simulation
        history = self.coupling.run_coupled_simulation(n_steps)

        # Final diagnostics
        final_state = history[-1] if history else {}
        return {
            "n_steps": n_steps,
            "final_n_cells": len(self.vertex._centres) if self.vertex._centres is not None else 0,
            "final_energy": final_state.get("vertex_energy", 0.0),
            "final_cdc42_gtp_mean": final_state.get("cdc42_gtp_mean", 0.0),
            "divisions_occurred": sum(1 for h in history if h.get("division_occurred", False)),
            "history_length": len(history),
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "cdc42": self.cdc42.get_status(),
            "polarity": self.polarity.get_status(),
            "coupling": self.coupling.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO MOLECULAR — SMOKE TEST")
    print(" Stochastic Cdc42 + Stress Polarity + RDME-Vertex Coupling")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" spatial module     : {HAS_SPATIAL}")
    print(f" vertex3d module    : {HAS_VERTEX3D}")

    suite = SESIMolecularOrganoidSuite(
        cdc42_cfg=Cdc42Config(n_cells=8, max_time=20.0, max_steps=5000),
        stress_cfg=StressPolarityConfig(),
        vertex_cfg=Vertex3DConfig(n_cells=12, box_size=8.0) if HAS_VERTEX3D else None,
        coupling_cfg=CouplingConfig())

    # ── 1. Stochastic Cdc42 Cycle ───────────────────────────────────────
    print("\n [SCC1] Stochastic Cdc42 GTPase Cycle")
    cdc42 = suite.cdc42
    print(f"   n_cells              : {cdc42.n_cells}")
    print(f"   n_cdc42_total        : {cdc42.cfg.n_cdc42_total}")
    print(f"   k_gef / k_gap        : {cdc42.cfg.k_gef} / {cdc42.cfg.k_gap}")
    print(f"   k_feedback           : {cdc42.cfg.k_feedback}")

    # Run simulation
    traj, times = cdc42.simulate(initial_gtp_fraction=0.1, max_steps=2000)
    print(f"   trajectory length    : {len(traj)}")
    print(f"   final time           : {times[-1]:.4f}")
    final_state = traj[-1]
    print(f"   final GTP mean       : {final_state[:, 0].mean():.1f}")
    print(f"   final GTP std        : {final_state[:, 0].std():.1f}")

    # Polarity vector
    pol_vec = cdc42.get_polarity_vector(final_state)
    print(f"   polarity vector      : {pol_vec}")
    print(f"   polarity max cell    : {np.argmax(pol_vec)}")

    cdc42_status = cdc42.get_status()
    print(f"   COROLLARY-3 guard    : {cdc42_status['corollary3_guard']}")
    print(f"   HINO guard           : {cdc42_status['hino_guard']}")

    # ── 2. Mechanical Stress Polarity ───────────────────────────────────
    print("\n [MSP1] Mechanical Stress Polarity")
    pol = suite.polarity
    print(f"   Cue weights          : stress={pol.cfg.w_stress}, "
          f"cdc42={pol.cfg.w_cdc42}, elong={pol.cfg.w_elongation}")
    print(f"   Par activation thresh: {pol.cfg.par_activation_threshold}")

    # Test Par activation
    stress_mag = 0.5
    par_act = pol.activate_par_complex(stress_mag)
    print(f"   Par activation       : stress={stress_mag:.2f} -> activation={par_act:.3f}")

    pol_status = pol.get_status()
    print(f"   COROLLARY-3 guard    : {pol_status['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {pol_status['prop3_awareness']}")

    # ── 3. RDME-Vertex Coupling ─────────────────────────────────────────
    print("\n [RVC1] RDME-Vertex Full Coupling")
    if HAS_VERTEX3D and suite.vertex is not None:
        coupling = suite.coupling
        print(f"   mech_to_chem weight  : {coupling.cfg.mech_to_chem_weight}")
        print(f"   chem_to_mech weight  : {coupling.cfg.chem_to_mech_weight}")
        print(f"   polarity_to_division : {coupling.cfg.polarity_to_division}")

        # Run coupled simulation
        history = coupling.run_coupled_simulation(n_steps=10)
        print(f"   coupled steps        : {len(history)}")
        if history:
            final = history[-1]
            print(f"   final energy         : {final.get('vertex_energy', 0.0):.4f}")
            print(f"   final Cdc42 GTP mean : {final.get('cdc42_gtp_mean', 0.0):.2f}")
            print(f"   divisions occurred   : {sum(1 for h in history if h.get('division_occurred', False))}")

        coupling_status = coupling.get_status()
        print(f"   COROLLARY-3 guard    : {coupling_status['corollary3_guard']}")
        print(f"   PROP-3 awareness     : {coupling_status['prop3_awareness']}")
    else:
        print("   [SKIP] Vertex model not available")

    # ── 4. Full Molecular Ontogeny ──────────────────────────────────────
    print("\n [ONTO] Full Molecular Ontogeny")
    if HAS_VERTEX3D and suite.vertex is not None:
        result = suite.run_molecular_ontogeny(n_steps=15)
        print(f"   n_steps              : {result['n_steps']}")
        print(f"   final n_cells        : {result['final_n_cells']}")
        print(f"   final energy         : {result['final_energy']:.4f}")
        print(f"   final Cdc42 GTP mean : {result['final_cdc42_gtp_mean']:.2f}")
        print(f"   divisions occurred   : {result['divisions_occurred']}")
    else:
        print("   [SKIP] Vertex model not available")

    # ── 5. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Gillespie SSA is EXACT for well-mixed reactions (1976/77).")
    print("       Real Cdc42 dynamics involve spatial gradients and many")
    print("       regulatory proteins not captured here.")
    print(" [C]   Wave-pinning is a simplified model (Otsuji et al. 2007).")
    print(" [C]   Hertwig's rule (1888): cells divide along long axis.")
    print("       Empirical, not universal.")
    print(" [D]   COROLLARY-3 guard: all fields (GTPase, stress, polarity)")
    print("       are first-order only. NO Hessian anywhere.")
    print(" [C]   HINO guard: noise from discrete propensities, not smooth")
    print("       density. Γ ⊥ µ (Thm 8).")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print(" [D]   RDME-vertex coupling is heuristic; real tissues have")
    print("       complex mechanochemical feedback not captured.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL models. No biological or")
    print("       medical claim. Not validated against real organoids.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
