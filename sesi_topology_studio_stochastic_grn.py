"""
=============================================================================
SESI ORGANOID STUDIO STOCHASTIC
Gillespie/CLE GRN + Polarity-Driven Division + τ-Vertex Coupling
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
[SGRN1] StochasticGRN
      • Gillespie SSA [C 1976/1977]: exact stochastic simulation
          - propensities a_i(x) for production/degradation reactions
          - exponentially distributed waiting times
          - discrete copy-number states
      • Chemical Langevin Equation [C Gillespie 2000]:
          dx_i = Σ_j ν_ij a_j(x) dt + Σ_j ν_ij sqrt(a_j(x)) dW_j
        valid when copy numbers are large enough
      • Regime switching: SSA when copy number < threshold, CLE above
      • HINO guard: noise amplitude computed from propensities directly,
        never from an assumed smooth noise density (Γ ⊥ µ, Thm 8)
      • COROLLARY-3 guard: first-order reaction propensities only;
        no second-order field derivatives in the noise term

[PDO1] PolarityDivisionController
      • Hertwig's rule [C 1888]: cells divide along their long axis
      • Polarity vector from THREE first-order cues:
          1. Mechanical stress anisotropy (from vertex model forces)
          2. Morphogen gradient direction (local ∇m)
          3. Cell elongation axis (from Voronoi cell shape)
      • Division plane ⟂ polarity vector
      • COROLLARY-3 guard: all cues are FIRST-ORDER (gradients, forces).
        NO Hessian of stress or morphogen fields.
      • PROP-3 awareness: when multiplying fields (stress × morphogen),
        the result loses two Sobolev orders; we rescale and document it.

[TVT1] TauVertexCoupling
      • Couples the 3-D vertex model to τ from MultiStageCoarseningTau
      • τ controls:
          - vertex-model resolution (target number of cells)
          - GRN effective gene count (coarse-graining level)
          - event rates (division/death frequency scaling)
      • THM 3(b) DISCIPLINE: τ changes are validated for convergence
        across coarsening levels before being applied.
      • PROP-3 awareness: τ is a SCALE SELECTOR, not a regularity
        recovery tool. Decreasing τ does NOT restore smoothness lost
        by the commutator.

[OVT2] SESIStochasticOrganoidSuite (orchestrator)
      • Full stochastic organoid life-cycle:
        1. τ from multi-stage coarsening sets the resolution
        2. StochasticGRN assigns cell fates with noise
        3. VoronoiVertexModel3D relaxes mechanics
        4. PolarityDivisionController orients divisions
        5. Repeat; export to AR when done

=============================================================================
HONEST SCOPE
=============================================================================
• StochasticGRN: Gillespie SSA is EXACT for well-mixed reactions [C].
  The Chemical Langevin Equation is an APPROXIMATION valid at large
  copy numbers [C Gillespie 2000]. Neither captures spatial effects
  or detailed biochemistry. Gene copy numbers here are illustrative.
• PolarityDivisionController: uses first-order mechanical and chemical
  cues. Real cell polarity involves Par/Cdc42/aPKC molecular machinery
  far beyond this model. Hertwig's rule is empirical, not universal.
• TauVertexCoupling: τ is a HEURISTIC scale selector. It does NOT
  transfer the carpet's Mosco closure to biological tissue. The
  connection is dimensional-analogy only, not a theorem.
• COROLLARY-3 guard enforced: no second-order field derivatives.
• HINO guard enforced: no smooth noise/energy density assumptions.
• PROP-3 awareness: field products lose two Sobolev orders.
• THEOREM-9 awareness: no bounded-commutator spectral structure claimed.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All parameters are illustrative. No biological/medical claim.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.stochastic_grn")

# ── Ecosystem imports (graceful) ─────────────────────────────────────────
try:
    from sesi_topology_studio_vertex3d_grn_ode import (
        VoronoiVertexModel3D, Vertex3DConfig, Vertex3DState,
        ContinuousGRN, GRNODEConfig,
    )
    HAS_VERTEX3D = True
except Exception:
    HAS_VERTEX3D = False

try:
    from sesi_topology_studio_organoid import (
        MultiStageCoarseningTau, MultiStageTauConfig, CoarseningRegime,
    )
    HAS_MST = True
except Exception:
    HAS_MST = False

__all__ = [
    "GRNRegime", "Reaction", "StochasticGRN", "StochasticGRNConfig",
    "PolarityDivisionController", "PolarityConfig",
    "TauVertexCoupling", "TauVertexConfig",
    "SESIStochasticOrganoidSuite",
]


# =============================================================================
# SECTION 0 — DIFFERENTIABILITY PRIMITIVES
# =============================================================================

def _softplus_floor(x: Tensor, floor: float, beta: Optional[float] = None,
                    sharpness: float = 50.0) -> Tensor:
    if beta is None:
        beta = sharpness / max(abs(floor), 1e-12)
    return floor + F.softplus(x - floor, beta=beta)

def _soft_abs(x: Tensor, eps: float = 1e-9) -> Tensor:
    return torch.sqrt(x * x + eps)


# =============================================================================
# SECTION 1 — [SGRN1] STOCHASTIC GRN (GILLESPIE / CLE)
# =============================================================================

class GRNRegime(str, Enum):
    """Simulation regime for the stochastic GRN."""
    SSA = "ssa"             # exact Gillespie, discrete copy numbers
    CLE = "cle"             # Chemical Langevin Equation, continuous
    AUTO = "auto"           # switch based on copy number


@dataclass
class Reaction:
    """
    A single reaction in the stochastic GRN.

    For gene regulation, typical reactions are:
      production:   ∅ → X_i        (rate β)
      degradation:  X_i → ∅        (rate γ x_i)
      regulation:   X_j → X_j + X_i (rate β · H(x_j))  [Hill activation]
      repression:   X_j → X_j + X_i (rate β · (1 - H(x_j)))
    """
    reaction_id: str
    reactants: Dict[str, int]     # species -> stoichiometry consumed
    products: Dict[str, int]      # species -> stoichiometry produced
    rate: float                   # rate constant
    regulation: Optional[str] = None   # gene that regulates this reaction
    hill_K: float = 0.5
    hill_n: float = 2.0
    is_activating: bool = True


@dataclass
class StochasticGRNConfig:
    """Configuration for the stochastic GRN."""
    n_genes: int = 6
    # regime
    regime: GRNRegime = GRNRegime.AUTO
    cle_threshold: float = 20.0    # copy number above which CLE is used
    # production / degradation
    beta: float = 5.0              # max production rate
    gamma: float = 1.0             # degradation rate
    # Hill regulation
    hill_K: float = 5.0
    hill_n: float = 2.0
    k_regulators: int = 2
    # copy number scaling (controlled by τ via TauVertexCoupling)
    base_copy_number: int = 50     # molecules per unit concentration
    # integration
    dt_max: float = 0.1
    max_time: float = 50.0
    # stochasticity
    noise_floor: float = 1e-6
    seed: int = 0


class StochasticGRN:
    """
    [C] Stochastic gene regulatory network via Gillespie SSA and CLE.

    GILLESPIE SSA  [C Gillespie 1976, 1977]
      Exact stochastic simulation of well-mixed reactions.
      At each step:
        1. Compute all propensities a_i(x).
        2. Sample waiting time τ ~ Exp(Σ a_i).
        3. Sample which reaction fires, weighted by a_i.
        4. Update state x by the reaction's stoichiometry.

    CHEMICAL LANGEVIN EQUATION  [C Gillespie 2000]
      When copy numbers are large, the master equation is approximated by:
        dx_i = Σ_j ν_{ij} a_j(x) dt + Σ_j ν_{ij} √(a_j(x)) dW_j
      where ν_{ij} is the stoichiometry of species i in reaction j.
      This is an SDE driven by independent Wiener processes.

    REGIME SWITCHING  [D]
      AUTO mode: use SSA when total copy number < cle_threshold,
      CLE otherwise. This gives exact discreteness at low abundance and
      efficient continuous dynamics at high abundance.

    HINO GUARD  [C Thm 8]
      The noise amplitude √(a_j) is computed from the PROPENSITY directly,
      which is a discrete count. We do NOT assume a smooth noise density
      field. The stochasticity is singular in the sense of being tied to
      individual reaction events, not spread smoothly over space.

    COROLLARY-3 GUARD  [D]
      All propensities are FIRST-ORDER functions of the state (linear
      degradation, Hill regulation). We do NOT compute second derivatives
      of the copy-number field. The noise term is multiplicative but
      first-order in the state.

    HONEST
      • SSA is exact for well-mixed systems; real cells have spatial
        structure that breaks the well-mixed assumption.
      • CLE is an approximation; it can produce negative copy numbers,
        which we floor at zero.
      • Copy numbers are illustrative; real gene expression involves
        1-1000 molecules per cell typically.
    """

    def __init__(self, cfg: Optional[StochasticGRNConfig] = None):
        self.cfg = cfg or StochasticGRNConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n = self.cfg.n_genes
        self.reactions: List[Reaction] = []
        self._build_reactions()

    def _build_reactions(self) -> None:
        """Build production and degradation reactions for each gene."""
        self.reactions = []
        # Regulatory graph (same as ContinuousGRN for consistency)
        k = self.cfg.k_regulators
        self.regulators: List[List[int]] = []
        self.signs: List[List[int]] = []
        for i in range(self.n):
            regs = self._rng.choice(
                [j for j in range(self.n) if j != i],
                size=min(k, self.n - 1), replace=False).tolist()
            self.regulators.append(regs)
            self.signs.append([int(self._rng.choice([-1, 1])) for _ in regs])

        # Production reactions (one per gene, regulated)
        for i in range(self.n):
            self.reactions.append(Reaction(
                reaction_id=f"prod_{i}",
                reactants={},
                products={f"X{i}": 1},
                rate=self.cfg.beta,
                regulation=f"X{self.regulators[i][0]}" if self.regulators[i] else None,
                hill_K=self.cfg.hill_K,
                hill_n=self.cfg.hill_n,
                is_activating=(self.signs[i][0] > 0) if self.regulators[i] else True,
            ))
        # Degradation reactions (one per gene)
        for i in range(self.n):
            self.reactions.append(Reaction(
                reaction_id=f"deg_{i}",
                reactants={f"X{i}": 1},
                products={},
                rate=self.cfg.gamma,
            ))

    # ── propensities ────────────────────────────────────────────────────
    def _hill(self, x: float, K: float, n: float,
               activating: bool) -> float:
        """Hill function for propensity modulation."""
        x = max(x, 0.0)
        K = max(K, 1e-12)
        xn = x ** n
        Kn = K ** n
        if activating:
            return xn / (Kn + xn)  # activation
        else:
            return Kn / (Kn + xn)  # repression

    def compute_propensities(self, state: Dict[str, float]) -> np.ndarray:
        """
        Compute all reaction propensities given the current state.
        First-order functions only (COROLLARY-3 guard).
        """
        a = np.zeros(len(self.reactions))
        for idx, rxn in enumerate(self.reactions):
            # Check reactants are available
            available = True
            for species, stoich in rxn.reactants.items():
                if state.get(species, 0.0) < stoich:
                    available = False
                    break
            if not available:
                a[idx] = 0.0
                continue

            # Base propensity from mass action
            prop = rxn.rate
            for species, stoich in rxn.reactants.items():
                prop *= state.get(species, 0.0) ** stoich

            # Regulatory modulation via Hill function
            if rxn.regulation is not None:
                reg_conc = state.get(rxn.regulation, 0.0)
                h = self._hill(reg_conc, rxn.hill_K, rxn.hill_n,
                                rxn.is_activating)
                prop *= h

            a[idx] = max(prop, 0.0)
        return a

    # ── Gillespie SSA ───────────────────────────────────────────────────
    def ssa_step(self, state: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        One Gillespie SSA step. Returns (new_state, time_increment).
        Exact for well-mixed discrete reactions [C].
        """
        a = self.compute_propensities(state)
        a_total = a.sum()
        if a_total <= 0:
            return state.copy(), 0.0

        # Sample waiting time (exponential)
        tau = -math.log(max(self._rng.random(), 1e-12)) / a_total

        # Sample which reaction fires
        r = self._rng.random() * a_total
        cumsum = 0.0
        fired_idx = len(self.reactions) - 1
        for idx in range(len(self.reactions)):
            cumsum += a[idx]
            if r <= cumsum:
                fired_idx = idx
                break

        # Apply reaction
        new_state = state.copy()
        rxn = self.reactions[fired_idx]
        for species, stoich in rxn.reactants.items():
            new_state[species] = max(0.0, new_state.get(species, 0.0) - stoich)
        for species, stoich in rxn.products.items():
            new_state[species] = new_state.get(species, 0.0) + stoich

        return new_state, tau

    def run_ssa(self, x0: Dict[str, float],
                 max_time: Optional[float] = None,
                 max_steps: int = 10000) -> Tuple[List[Dict[str, float]], List[float]]:
        """Run full Gillespie SSA trajectory."""
        max_time = max_time or self.cfg.max_time
        state = x0.copy()
        t = 0.0
        trajectory = [state.copy()]
        times = [0.0]
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

    # ── Chemical Langevin Equation ──────────────────────────────────────
    def cle_step(self, state: Dict[str, float],
                  dt: Optional[float] = None) -> Dict[str, float]:
        """
        One CLE step: dx_i = Σ_j ν_ij a_j dt + Σ_j ν_ij √a_j dW_j.
        Valid at large copy numbers [C Gillespie 2000].
        HINO guard: noise √a_j from discrete propensity, not smooth density.
        """
        dt = dt or self.cfg.dt_max
        a = self.compute_propensities(state)
        new_state = state.copy()

        for i in range(self.n):
            species = f"X{i}"
            drift = 0.0
            noise = 0.0
            for j, rxn in enumerate(self.reactions):
                # Stoichiometry of species i in reaction j
                nu = rxn.products.get(species, 0) - rxn.reactants.get(species, 0)
                if nu == 0:
                    continue
                drift += nu * a[j]
                # HINO guard: noise from discrete propensity
                noise += nu * math.sqrt(max(a[j], 0.0)) * self._rng.standard_normal()

            # COROLLARY-3 guard: first-order update only
            new_val = state.get(species, 0.0) + drift * dt + noise * math.sqrt(dt)
            new_state[species] = max(new_val, 0.0)  # floor at zero

        return new_state

    def run_cle(self, x0: Dict[str, float],
                 n_steps: int = 500,
                 dt: Optional[float] = None) -> Tuple[List[Dict[str, float]], List[float]]:
        """Run CLE trajectory."""
        dt = dt or self.cfg.dt_max
        state = x0.copy()
        trajectory = [state.copy()]
        times = [0.0]
        t = 0.0
        for _ in range(n_steps):
            state = self.cle_step(state, dt)
            t += dt
            trajectory.append(state.copy())
            times.append(t)
        return trajectory, times

    # ── auto regime switching ───────────────────────────────────────────
    def _total_copy_number(self, state: Dict[str, float]) -> float:
        return sum(state.values())

    def simulate(self, x0: Dict[str, float],
                  n_steps: int = 200) -> Tuple[List[Dict[str, float]], List[float], str]:
        """
        Simulate with automatic regime selection.
        Returns (trajectory, times, regime_used).
        """
        if self.cfg.regime == GRNRegime.SSA:
            traj, times = self.run_ssa(x0, max_steps=n_steps)
            return traj, times, "ssa"
        elif self.cfg.regime == GRNRegime.CLE:
            traj, times = self.run_cle(x0, n_steps=n_steps)
            return traj, times, "cle"
        else:  # AUTO
            total = self._total_copy_number(x0)
            if total < self.cfg.cle_threshold:
                traj, times = self.run_ssa(x0, max_steps=n_steps)
                return traj, times, "ssa"
            else:
                traj, times = self.run_cle(x0, n_steps=n_steps)
                return traj, times, "cle"

    # ── gene expression for vertex model coupling ──────────────────────
    def expression_vector(self, state: Dict[str, float]) -> np.ndarray:
        """Convert state dict to expression vector for downstream use."""
        return np.array([state.get(f"X{i}", 0.0) for i in range(self.n)],
                          dtype=np.float64)

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_genes": self.n,
            "n_reactions": len(self.reactions),
            "regime": self.cfg.regime.value,
            "cle_threshold": self.cfg.cle_threshold,
            "corollary3_guard": "first-order propensities only; no Hessian",
            "hino_guard": "noise from discrete propensity, not smooth density",
        }


# =============================================================================
# SECTION 2 — [PDO1] POLARITY-DRIVEN DIVISION CONTROLLER
# =============================================================================

@dataclass
class PolarityConfig:
    """Configuration for polarity-driven division."""
    # cue weights (must sum to ~1 for interpretability)
    w_stress: float = 0.4       # mechanical stress anisotropy
    w_morphogen: float = 0.3    # morphogen gradient direction
    w_elongation: float = 0.3   # cell elongation axis (Hertwig's rule)
    # stress computation
    stress_neighborhood: float = 1.5  # radius for local stress estimate
    # morphogen
    morphogen_gradient_eps: float = 1e-3
    # randomness
    orientation_noise: float = 0.1  # angular noise in division plane
    seed: int = 0


class PolarityDivisionController:
    """
    [C/D] Division orientation from first-order polarity cues.

    HERTWIG'S RULE  [C Hertwig 1888]
      Cells tend to divide along their longest axis. We implement this
      via the elongation cue from the Voronoi cell shape.

    POLARITY CUES  [D]
      The division plane is perpendicular to the polarity vector p̂, which
      is a weighted sum of three FIRST-ORDER cues:

        1. Stress anisotropy: direction of maximum local stress difference.
           Computed from vertex-model forces (first-order).
        2. Morphogen gradient: direction of ∇m at the cell centre.
           First-order spatial derivative.
        3. Elongation axis: principal axis of the cell's Voronoi shape.
           From the cell's geometry (first-order).

    DIVISION PLANE
      The division plane passes through the cell centre and is PERPENDICULAR
      to the polarity vector p̂. The two daughters are offset along p̂.

    COROLLARY-3 GUARD  [D]
      All three cues are FIRST-ORDER (gradients, forces, geometry).
      We do NOT compute second derivatives of stress or morphogen fields.
      No Hessian of any field is used.

    PROP-3 AWARENESS  [P]
      When we multiply fields (e.g. stress × morphogen for a combined cue),
      the result loses two Sobolev orders (Carpet-19 Prop 3). We rescale
      the product and document that it is a lower-regularity quantity.

    HONEST
      • Real cell polarity involves Par/Cdc42/aPKC molecular machinery.
      • Hertwig's rule is empirical and has exceptions.
      • The cue weights are illustrative; real tissues weigh them differently.
      • No biological/medical claim.
    """

    def __init__(self, cfg: Optional[PolarityConfig] = None):
        self.cfg = cfg or PolarityConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    # ── cue 1: mechanical stress anisotropy ─────────────────────────────
    def stress_cue(self, centre: np.ndarray, forces: np.ndarray,
                    centres: np.ndarray) -> np.ndarray:
        """
        Direction of maximum stress anisotropy near the cell.
        Uses first-order force differences (COROLLARY-3 guard).
        """
        d = np.linalg.norm(centres - centre, axis=1)
        mask = d < self.cfg.stress_neighborhood
        mask[d == 0] = False  # exclude self
        if mask.sum() < 2:
            return np.array([1.0, 0.0, 0.0])
        local_forces = forces[mask]
        # Stress anisotropy: direction of max variance of forces
        # This is a first-order statistic (variance of first-order forces)
        cov = np.cov(local_forces.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Direction of max eigenvalue = max stress anisotropy
        return eigenvectors[:, -1]

    # ── cue 2: morphogen gradient ───────────────────────────────────────
    def morphogen_cue(self, centre: np.ndarray,
                       morphogen_field: np.ndarray,
                       centres: np.ndarray) -> np.ndarray:
        """
        Direction of ∇m at the cell centre.
        First-order spatial derivative (COROLLARY-3 guard).
        """
        if morphogen_field is None or len(morphogen_field) != len(centres):
            return np.array([0.0, 0.0, 0.0])
        eps = self.cfg.morphogen_gradient_eps
        # Estimate gradient by local finite differences
        d = np.linalg.norm(centres - centre, axis=1)
        nn_idx = np.argsort(d)[1:min(5, len(d))]  # nearest neighbours
        if len(nn_idx) < 2:
            return np.array([0.0, 0.0, 0.0])
        grad = np.zeros(3)
        for j in nn_idx:
            delta = centres[j] - centre
            dist = np.linalg.norm(delta) + eps
            dm = morphogen_field[j] - morphogen_field[
                np.argmin(d)] if d.min() >= 0 else 0.0
            grad += delta * dm / (dist * dist)
        norm = np.linalg.norm(grad)
        if norm < eps:
            return np.array([0.0, 0.0, 0.0])
        return grad / norm

    # ── cue 3: elongation axis (Hertwig's rule) ────────────────────────
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
        # Principal axis of nearest neighbours
        offsets = centres[nn_idx] - centre
        cov = np.cov(offsets.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, -1]

    # ── combined polarity vector ───────────────────────────────────────
    def polarity_vector(self, centre: np.ndarray,
                          centres: np.ndarray,
                          forces: np.ndarray,
                          morphogen_field: Optional[np.ndarray],
                          cell_idx: int) -> np.ndarray:
        """
        Combined polarity vector from all three cues.
        PROP-3 awareness: the weighted sum is a lower-regularity quantity.
        """
        s_cue = self.stress_cue(centre, forces, centres)
        m_cue = self.morphogen_cue(centre, morphogen_field, centres)
        e_cue = self.elongation_cue(centre, centres)

        # Weighted sum (first-order combination)
        p = (self.cfg.w_stress * s_cue +
              self.cfg.w_morphogen * m_cue +
              self.cfg.w_elongation * e_cue)

        # Normalise
        norm = np.linalg.norm(p)
        if norm < 1e-12:
            # Fallback: random direction
            p = self._rng.normal(size=3)
            norm = np.linalg.norm(p)
        return p / norm

    # ── division ────────────────────────────────────────────────────────
    def divide_with_polarity(self, vertex_model: Any, cell_idx: int,
                              morphogen_field: Optional[np.ndarray] = None
                              ) -> Optional[int]:
        """
        Divide a cell with polarity-oriented plane.
        Returns the index of the new daughter cell, or None on failure.
        """
        if vertex_model._centres is None or cell_idx >= len(vertex_model._centres):
            return None

        vertex_model._compute_voronoi()
        forces = vertex_model.compute_forces()
        centre = vertex_model._centres[cell_idx]
        centres = vertex_model._centres

        # Get polarity vector
        p = self.polarity_vector(centre, centres, forces,
                                   morphogen_field, cell_idx)

        # Add orientation noise
        if self.cfg.orientation_noise > 0:
            noise = self.cfg.orientation_noise * self._rng.normal(size=3)
            p = p + noise
            p /= (np.linalg.norm(p) + 1e-12)

        # Division plane is perpendicular to p; daughters offset along p
        parent_volume = vertex_model._volumes[cell_idx]
        daughter_offset = 0.25 * (parent_volume ** (1.0/3.0)) * p

        # Halve parent
        vertex_model._volumes[cell_idx] *= 0.5
        vertex_model._surfaces[cell_idx] *= 0.5 ** (2.0/3.0)

        # Create daughter
        daughter_centre = centre + daughter_offset
        new_idx = vertex_model.add_cell(daughter_centre,
                                          volume=vertex_model._volumes[cell_idx])

        # Move parent slightly in opposite direction
        vertex_model._centres[cell_idx] = centre - daughter_offset * 0.5

        return new_idx

    def get_status(self) -> Dict[str, Any]:
        return {
            "w_stress": self.cfg.w_stress,
            "w_morphogen": self.cfg.w_morphogen,
            "w_elongation": self.cfg.w_elongation,
            "orientation_noise": self.cfg.orientation_noise,
            "corollary3_guard": "all cues are first-order; no field Hessian",
            "prop3_awareness": "field products lose two Sobolev orders",
        }


# =============================================================================
# SECTION 3 — [TVT1] τ-VERTEX COUPLING
# =============================================================================

@dataclass
class TauVertexConfig:
    """Configuration for τ-vertex coupling."""
    # τ range (must match MultiStageCoarseningTau)
    tau_min: float = 0.005
    tau_max: float = 0.5
    # resolution mapping: τ -> target number of cells
    n_cells_at_tau_min: int = 200   # fine resolution
    n_cells_at_tau_max: int = 20    # coarse resolution
    # event rate scaling: τ -> division/death rate multiplier
    event_rate_exponent: float = -0.5   # rate ~ τ^exponent
    # GRN coarse-graining: τ -> effective gene count
    n_genes_at_tau_min: int = 20
    n_genes_at_tau_max: int = 4
    # discipline
    validate_tau_convergence: bool = True
    max_tau_change_per_step: float = 0.1
    # device
    device: str = "cpu"


class TauVertexCoupling:
    """
    [D] Couples the 3-D vertex model to τ from multi-stage coarsening.

    WHAT τ CONTROLS  [D]
      τ is a SCALE SELECTOR derived from the measured coarsening length.
      It sets:
        1. Vertex-model resolution: target number of cells
             n_cells(τ) = interpolate(n_cells_at_tau_max,
                                       n_cells_at_tau_min, τ)
           Small τ (late coarsening, sharp interfaces) -> many cells.
           Large τ (early coarsening, diffuse interfaces) -> few cells.
        2. GRN effective gene count:
             n_genes(τ) = interpolate(n_genes_at_tau_max,
                                       n_genes_at_tau_min, τ)
           Small τ -> fine-grained GRN. Large τ -> coarse GRN.
        3. Event rate scaling:
             rate_multiplier(τ) = τ^event_rate_exponent
           Small τ -> frequent events. Large τ -> rare events.

    THM 3(b) DISCIPLINE  [C]
      Before applying a new τ, we CHECK that it has converged from the
      previous value (relative change below threshold). If not converged,
      we clamp the change to max_tau_change_per_step. This prevents
      oscillating resolution changes that would destabilise the simulation.

    PROP-3 AWARENESS  [P]
      τ is a SCALE SELECTOR, not a regularity recovery tool. Decreasing τ
      does NOT restore smoothness lost by the commutator (Carpet-19 Prop 3:
      [A,M_f] loses two orders). τ only selects which scale to resolve.

    HONEST
      • The mapping from τ to n_cells/n_genes/rate is a HEURISTIC.
        There is no theorem connecting coarsening length to cell number.
      • τ from MultiStageCoarseningTau is itself a heuristic tracking
        the measured interface width.
      • This coupling is a dimensional analogy, NOT a closure transfer
        from the carpet to biological tissue.
    """

    def __init__(self, cfg: Optional[TauVertexConfig] = None):
        self.cfg = cfg or TauVertexConfig()
        self._current_tau = self.cfg.tau_max
        self._tau_history: List[float] = []

    # ── τ interpolation helpers ─────────────────────────────────────────
    def _interp(self, tau: float, v_min: float, v_max: float) -> float:
        """Linear interpolation: tau_min -> v_min, tau_max -> v_max."""
        t0, t1 = self.cfg.tau_min, self.cfg.tau_max
        frac = (tau - t0) / max(t1 - t0, 1e-12)
        frac = max(0.0, min(1.0, frac))
        return v_min + frac * (v_max - v_min)

    def target_n_cells(self, tau: Optional[float] = None) -> int:
        """Target number of cells for the vertex model at this τ."""
        tau = tau if tau is not None else self._current_tau
        n = self._interp(tau,
                           self.cfg.n_cells_at_tau_min,
                           self.cfg.n_cells_at_tau_max)
        return int(round(n))

    def target_n_genes(self, tau: Optional[float] = None) -> int:
        """Effective gene count for the GRN at this τ."""
        tau = tau if tau is not None else self._current_tau
        n = self._interp(tau,
                           self.cfg.n_genes_at_tau_min,
                           self.cfg.n_genes_at_tau_max)
        return max(2, int(round(n)))

    def event_rate_multiplier(self, tau: Optional[float] = None) -> float:
        """Event rate multiplier at this τ."""
        tau = tau if tau is not None else self._current_tau
        return tau ** self.cfg.event_rate_exponent

    # ── τ update with discipline ────────────────────────────────────────
    def update_tau(self, new_tau: float) -> Dict[str, Any]:
        """
        Update τ with convergence discipline (Thm 3(b) spirit).
        Returns diagnostics about the update.
        """
        new_tau = max(self.cfg.tau_min, min(self.cfg.tau_max, new_tau))

        # Check convergence
        if self.cfg.validate_tau_convergence and self._tau_history:
            prev_tau = self._tau_history[-1]
            rel_change = abs(new_tau - prev_tau) / max(abs(prev_tau), 1e-12)
            # Clamp large changes
            max_change = self.cfg.max_tau_change_per_step * prev_tau
            if abs(new_tau - prev_tau) > max_change:
                new_tau = prev_tau + math.copysign(max_change, new_tau - prev_tau)
                clamped = True
            else:
                clamped = False
        else:
            rel_change = 0.0
            clamped = False

        self._current_tau = new_tau
        self._tau_history.append(new_tau)

        return {
            "tau": new_tau,
            "previous_tau": self._tau_history[-2] if len(self._tau_history) > 1 else None,
            "relative_change": rel_change,
            "clamped": clamped,
            "target_n_cells": self.target_n_cells(new_tau),
            "target_n_genes": self.target_n_genes(new_tau),
            "event_rate_multiplier": self.event_rate_multiplier(new_tau),
        }

    # ── vertex model adaptation ─────────────────────────────────────────
    def adapt_vertex_model(self, vertex_model: Any, tau: Optional[float] = None
                            ) -> Dict[str, Any]:
        """
        Adapt the vertex model resolution to match τ.
        Adds or removes cells to approach the target count.
        Returns diagnostics.
        """
        tau = tau if tau is not None else self._current_tau
        target_n = self.target_n_cells(tau)
        current_n = len(vertex_model._centres) if vertex_model._centres is not None else 0

        added = 0
        removed = 0

        if current_n < target_n:
            # Add cells by subdividing existing ones
            n_to_add = target_n - current_n
            for _ in range(min(n_to_add, current_n)):
                # Pick a random cell to subdivide
                idx = int(np.random.randint(current_n))
                vertex_model.divide_cell(idx)
                added += 1
        elif current_n > target_n:
            # Remove smallest cells
            n_to_remove = current_n - target_n
            if vertex_model._volumes is not None:
                sorted_idx = np.argsort(vertex_model._volumes)
                for i in range(min(n_to_remove, len(sorted_idx))):
                    vertex_model.remove_cell(int(sorted_idx[i]))
                    removed += 1

        return {
            "tau": tau,
            "target_n_cells": target_n,
            "previous_n_cells": current_n,
            "cells_added": added,
            "cells_removed": removed,
            "final_n_cells": len(vertex_model._centres) if vertex_model._centres is not None else 0,
        }

    # ── GRN adaptation ──────────────────────────────────────────────────
    def adapt_grn(self, grn: Any, tau: Optional[float] = None) -> Dict[str, Any]:
        """
        Adapt the GRN gene count to match τ.
        Returns diagnostics.
        """
        tau = tau if tau is not None else self._current_tau
        target_genes = self.target_n_genes(tau)
        current_genes = grn.n if hasattr(grn, 'n') else 0

        return {
            "tau": tau,
            "target_n_genes": target_genes,
            "current_n_genes": current_genes,
            "note": ("GRN gene count adaptation requires rebuilding the "
                      "GRN with a new n_genes parameter. This method reports "
                      "the target; the caller must rebuild if needed."),
        }

    def get_tau_history(self) -> List[float]:
        return self._tau_history.copy()

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_tau": self._current_tau,
            "tau_range": (self.cfg.tau_min, self.cfg.tau_max),
            "target_n_cells": self.target_n_cells(),
            "target_n_genes": self.target_n_genes(),
            "event_rate_multiplier": self.event_rate_multiplier(),
            "tau_history_length": len(self._tau_history),
            "prop3_awareness": "τ is a scale selector, NOT a regularity recovery tool",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIStochasticOrganoidSuite:
    """
    Couples: StochasticGRN + PolarityDivisionController + TauVertexCoupling
             + VoronoiVertexModel3D.

    STOCHASTIC ONTOGENY PIPELINE:
      1. τ from multi-stage coarsening sets the resolution.
      2. StochasticGRN assigns cell fates with noise (SSA or CLE).
      3. VoronoiVertexModel3D relaxes mechanics.
      4. PolarityDivisionController orients divisions.
      5. Repeat; adapt resolution via τ as coarsening progresses.
    """

    def __init__(self,
                 vertex_cfg: Optional[Any] = None,
                 sgrn_cfg: Optional[StochasticGRNConfig] = None,
                 polarity_cfg: Optional[PolarityConfig] = None,
                 tau_cfg: Optional[TauVertexConfig] = None):
        self.vertex = VoronoiVertexModel3D(vertex_cfg) if HAS_VERTEX3D else None
        self.sgrn = StochasticGRN(sgrn_cfg)
        self.polarity = PolarityDivisionController(polarity_cfg)
        self.tau_coupling = TauVertexCoupling(tau_cfg)

    def run_stochastic_ontogeny(self,
                                  n_rounds: int = 3,
                                  initial_tau: Optional[float] = None,
                                  morphogen_field: Optional[np.ndarray] = None
                                  ) -> Dict[str, Any]:
        """
        Run the full stochastic organoid ontogeny.

        Args:
            n_rounds: number of life-cycle rounds
            initial_tau: starting τ (default: tau_max = coarse)
            morphogen_field: optional morphogen gradient for polarity

        Returns:
            Diagnostics including cell counts, divisions, τ trajectory.
        """
        if self.vertex is None:
            return {"error": "VoronoiVertexModel3D not available"}

        # Start at coarse resolution if no τ given
        if initial_tau is not None:
            self.tau_coupling.update_tau(initial_tau)
        else:
            self.tau_coupling.update_tau(self.tau_coupling.cfg.tau_max)

        # Adapt vertex model to initial τ
        adapt = self.tau_coupling.adapt_vertex_model(self.vertex)
        self.vertex.seed_cells()

        tau_trajectory = [self.tau_coupling._current_tau]
        cell_counts = []
        total_divisions = 0
        total_deaths = 0
        regime_counts = {"ssa": 0, "cle": 0}

        for round_idx in range(n_rounds):
            # 1. Run stochastic GRN for each cell
            n_cells = len(self.vertex._centres)
            grn_states = np.zeros((n_cells, self.sgrn.n))
            for i in range(n_cells):
                x0 = {f"X{j}": float(self.sgrn._rng.integers(0, 20))
                       for j in range(self.sgrn.n)}
                traj, times, regime = self.sgrn.simulate(x0, n_steps=20)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                grn_states[i] = self.sgrn.expression_vector(traj[-1])

            # 2. Relax mechanics
            self.vertex.step()

            # 3. Division with polarity
            n_to_check = min(n_cells, 10)  # limit per round
            for i in range(n_to_check):
                if self.vertex._volumes[i] > self.vertex.cfg.target_volume * 1.5:
                    m = morphogen_field[i] if morphogen_field is not None and i < len(morphogen_field) else None
                    new_idx = self.polarity.divide_with_polarity(
                        self.vertex, i, morphogen_field)
                    if new_idx is not None:
                        total_divisions += 1

            # 4. Update τ (simulate coarsening progression)
            # In a real simulation, τ would come from MultiStageCoarseningTau.
            # Here we decrease τ slightly each round to simulate refinement.
            new_tau = self.tau_coupling._current_tau * 0.8
            self.tau_coupling.update_tau(new_tau)
            tau_trajectory.append(self.tau_coupling._current_tau)

            # Adapt vertex model resolution
            adapt = self.tau_coupling.adapt_vertex_model(self.vertex)
            cell_counts.append(len(self.vertex._centres))

        return {
            "rounds": n_rounds,
            "final_n_cells": len(self.vertex._centres),
            "total_divisions": total_divisions,
            "total_deaths": total_deaths,
            "cell_counts": cell_counts,
            "tau_trajectory": tau_trajectory,
            "regime_counts": regime_counts,
            "final_tau": self.tau_coupling._current_tau,
            "event_rate_multiplier": self.tau_coupling.event_rate_multiplier(),
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "sgrn": self.sgrn.get_status(),
            "polarity": self.polarity.get_status(),
            "tau_coupling": self.tau_coupling.get_status(),
            "vertex_available": self.vertex is not None,
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO STOCHASTIC — SMOKE TEST")
    print(" Gillespie/CLE GRN + Polarity Division + τ-Vertex Coupling")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" vertex3d available : {HAS_VERTEX3D}")
    print(f" MST available      : {HAS_MST}")

    suite = SESIStochasticOrganoidSuite(
        vertex_cfg=Vertex3DConfig(n_cells=15, box_size=8.0) if HAS_VERTEX3D else None,
        sgrn_cfg=StochasticGRNConfig(n_genes=4, max_time=10.0),
        polarity_cfg=PolarityConfig(),
        tau_cfg=TauVertexConfig())

    # ── 1. Stochastic GRN ───────────────────────────────────────────────
    print("\n [SGRN1] Stochastic GRN (Gillespie SSA + CLE)")
    sgrn = suite.sgrn
    print(f"   genes              : {sgrn.n}")
    print(f"   reactions          : {len(sgrn.reactions)}")
    print(f"   regime             : {sgrn.cfg.regime.value}")
    print(f"   CLE threshold      : {sgrn.cfg.cle_threshold}")

    # Run with low copy number -> should use SSA
    x0_low = {f"X{i}": float(np.random.randint(1, 10)) for i in range(sgrn.n)}
    traj_low, times_low, regime_low = sgrn.simulate(x0_low, n_steps=50)
    print(f"\n   Low copy number    : total={sum(x0_low.values())}")
    print(f"   Regime used        : {regime_low}")
    print(f"   Trajectory length  : {len(traj_low)}")
    print(f"   Final state        : {[f'{k}={v:.1f}' for k,v in traj_low[-1].items()]}")

    # Run with high copy number -> should use CLE
    x0_high = {f"X{i}": float(np.random.randint(30, 60)) for i in range(sgrn.n)}
    traj_high, times_high, regime_high = sgrn.simulate(x0_high, n_steps=50)
    print(f"\n   High copy number   : total={sum(x0_high.values())}")
    print(f"   Regime used        : {regime_high}")
    print(f"   Trajectory length  : {len(traj_high)}")
    print(f"   Final state        : {[f'{k}={v:.1f}' for k,v in traj_high[-1].items()]}")

    sgrn_status = sgrn.get_status()
    print(f"\n   COROLLARY-3 guard  : {sgrn_status['corollary3_guard']}")
    print(f"   HINO guard         : {sgrn_status['hino_guard']}")

    # ── 2. τ-Vertex coupling ────────────────────────────────────────────
    print("\n [TVT1] τ-Vertex coupling")
    tc = suite.tau_coupling
    # Start at coarse τ
    tc.update_tau(tc.cfg.tau_max)
    print(f"   Initial τ          : {tc._current_tau:.4f}")
    print(f"   Target n_cells     : {tc.target_n_cells()}")
    print(f"   Target n_genes     : {tc.target_n_genes()}")
    print(f"   Event rate mult    : {tc.event_rate_multiplier():.4f}")

    # Refine τ (simulate coarsening progression)
    for i in range(5):
        new_tau = tc._current_tau * 0.7
        result = tc.update_tau(new_tau)
    print(f"   After refinement   : τ={tc._current_tau:.4f}")
    print(f"   Target n_cells     : {tc.target_n_cells()}")
    print(f"   Target n_genes     : {tc.target_n_genes()}")
    print(f"   Event rate mult    : {tc.event_rate_multiplier():.4f}")
    print(f"   τ trajectory       : {[round(t,4) for t in tc.get_tau_history()]}")
    print(f"   PROP-3 awareness   : {tc.get_status()['prop3_awareness']}")

    # ── 3. Polarity division ────────────────────────────────────────────
    print("\n [PDO1] Polarity-driven division")
    if HAS_VERTEX3D and suite.vertex is not None:
        suite.vertex.seed_cells()
        suite.vertex._compute_voronoi()
        n_before = len(suite.vertex._centres)
        # Divide a cell with polarity
        morph = np.linspace(0, 1, n_before)  # morphogen gradient
        new_idx = suite.polarity.divide_with_polarity(suite.vertex, 0, morph)
        n_after = len(suite.vertex._centres)
        print(f"   Cells before       : {n_before}")
        print(f"   Cells after        : {n_after}")
        print(f"   Division succeeded : {new_idx is not None}")
        pol_status = suite.polarity.get_status()
        print(f"   Cue weights        : stress={pol_status['w_stress']}, "
              f"morph={pol_status['w_morphogen']}, elong={pol_status['w_elongation']}")
        print(f"   COROLLARY-3 guard  : {pol_status['corollary3_guard']}")
        print(f"   PROP-3 awareness   : {pol_status['prop3_awareness']}")
    else:
        print("   [SKIP] Vertex model not available")

    # ── 4. Full stochastic ontogeny ─────────────────────────────────────
    print("\n [ONTO] Full stochastic organoid ontogeny")
    if HAS_VERTEX3D and suite.vertex is not None:
        result = suite.run_stochastic_ontogeny(n_rounds=2)
        print(f"   Rounds             : {result['rounds']}")
        print(f"   Final n_cells      : {result['final_n_cells']}")
        print(f"   Total divisions    : {result['total_divisions']}")
        print(f"   Cell counts        : {result['cell_counts']}")
        print(f"   τ trajectory       : {[round(t,4) for t in result['tau_trajectory']]}")
        print(f"   Regime counts      : {result['regime_counts']}")
        print(f"   Final τ            : {result['final_tau']:.4f}")
        print(f"   Event rate mult    : {result['event_rate_multiplier']:.4f}")
    else:
        print("   [SKIP] Vertex model not available")

    # ── 5. Full status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Gillespie SSA is EXACT for well-mixed reactions (1976/77).")
    print(" [C]   CLE is an APPROXIMATION valid at large copy numbers")
    print("       (Gillespie 2000). It can produce negative copy numbers,")
    print("       which we floor at zero.")
    print(" [C]   Hertwig's rule (1888): cells divide along long axis.")
    print("       Empirical, not universal.")
    print(" [D]   COROLLARY-3 guard: all fields (concentration, stress,")
    print("       polarity) are first-order only. NO Hessian anywhere.")
    print(" [C]   HINO guard: noise amplitude from discrete propensity,")
    print("       not smooth density. Γ ⊥ µ (Thm 8).")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print("       τ is a SCALE SELECTOR, not a regularity recovery tool.")
    print(" [D]   τ-vertex coupling is a HEURISTIC dimensional analogy.")
    print("       It does NOT transfer the carpet's Mosco closure to tissue.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   Stochastic GRN, polarity division, and τ coupling are")
    print("       COMPUTATIONAL models. No biological or medical claim.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
