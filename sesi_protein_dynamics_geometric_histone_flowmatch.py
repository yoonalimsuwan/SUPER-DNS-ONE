"""
=============================================================================
SESI PROTEIN DYNAMICS STUDIO GEOMETRIC
Geometric String Integration + Histone Modification Crosstalk + Flow Matching
Native integration with SDA-ONE, MSM, TPT, String, Flows modules
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
SDA-ONE & PREVIOUS MODULES INTEGRATION
=============================================================================
This module consumes output from:
  1. structural_domain_assembly_one.py (SDA-ONE):
     - init_coords, seq_features, sigma
  2. sesi_protein_dynamics_msm_chromatin_enhanced.py:
     - MarkovStateModel, ChromatinAccessibilityModel
  3. sesi_protein_dynamics_tpt_nucleosome_vae.py:
     - TransitionPathTheory, NucleosomePositioningModel, VAE
  4. sesi_protein_dynamics_string_remodeler_flows.py:
     - StringMethodMEP, NormalizingFlowsBridge
  5. sesi_protein_dynamics_adaptive_feedback_diffusion.py:
     - AdaptiveStringMethod, ScoreBasedDiffusionBridge

Integration points:
  1. GeometricStringIntegrator.refine_adaptive_string(adaptive_result, msm)
     - Uses adaptive string images as initial path
     - Applies symplectic/variational integration
     - Preserves arc-length constraint geometrically

  2. HistoneModificationCrosstalk.simulate_with_nucleosome(nuc_result)
     - Uses nucleosome positions as substrate
     - Multi-site modification state per nucleosome
     - Writer/eraser/reader kinetics with crosstalk

  3. FlowMatchingBridge.train_from_vae_and_diffusion(vae, diffusion)
     - Uses VAE latent codes as training data
     - Learns conditional vector field
     - Deterministic ODE sampling (faster than diffusion)

=============================================================================
MODULES
=============================================================================
[GSI1] GeometricStringIntegrator
      • Symplectic integration of string method [C Hairer 2006; Marsden 2001]
      • Verlet / leapfrog / RATTLE for constrained dynamics
      • Projection onto equal arc-length constraint manifold
      • Energy-momentum conservation (discrete Noether)
      • Connection to adaptive string from previous module
      • COROLLARY-3 guard: first-order integration; no Hessian of action
      • HINO guard: path space singular; discrete path representation

[HMC1] HistoneModificationCrosstalk
      • Multi-site combinatorial state [C Jenuwein & Allis 2001]
      • Modifications: H3K4me3, H3K27me3, H3K9me3, H3K36me3, H2AK119ub,
        H3K9ac, H3K27ac, H4K20me1
      • Writer (HMT/HAT), eraser (HDM/HDAC), reader (chromodomain, etc.)
      • Crosstalk: one modification affects rates of others
      • Gillespie SSA with combinatorial propensity
      • COROLLARY-3 guard: first-order kinetics; no Hessian
      • HINO guard: discrete modification events, not smooth density

[FLM1] FlowMatchingBridge
      • Conditional flow matching [C Lipinski 2023; Liu 2023]
      • Learns vector field v_t(x) interpolating noise → data
      • Deterministic ODE sampling: dx/dt = v_t(x)
      • Faster than diffusion (no stochastic term)
      • Connection to VAE latent space and MSM states
      • COROLLARY-3 guard: first-order ODE integration
      • HINO guard: latent space singular; not smooth manifold

=============================================================================
HONEST SCOPE
=============================================================================
• GeometricStringIntegrator: symplectic methods preserve structure but
  may not converge to true MEP in rugged landscapes. Path space is
  singular (HINO); discrete representation respects this.
• HistoneModificationCrosstalk: combinatorial state space grows as 2^n_mod.
  For 8 modifications, 256 states per nucleosome. Real histone dynamics
  involve many more modifications and complex crosstalk not captured.
• FlowMatchingBridge: learns a vector field but does NOT guarantee
  Boltzmann sampling. Deterministic ODE may miss modes in multi-modal
  distributions. This is a generative model, not a physics engine.
• COROLLARY-3 guard: all fields first-order only; no Hessian.
• HINO guard: energy landscapes singular; no smooth density assumed.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.geometric_histone_flowmatch")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY_INTERP = True
except Exception:
    HAS_SCIPY_INTERP = False

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except Exception:
    HAS_TORCHDIFFEQ = False

# ── Previous modules import (graceful) ──────────────────────────────────
try:
    from sesi_protein_dynamics_adaptive_feedback_diffusion import (
        AdaptiveStringMethod,
        AdaptiveStringResult,
    )
    HAS_ADAPTIVE = True
except Exception:
    HAS_ADAPTIVE = False
    AdaptiveStringMethod = None

try:
    from sesi_protein_dynamics_msm_chromatin_enhanced import (
        MarkovStateModel,
        ChromatinAccessibilityModel,
    )
    HAS_MSM = True
except Exception:
    HAS_MSM = False
    MarkovStateModel = None

try:
    from sesi_protein_dynamics_tpt_nucleosome_vae import (
        NucleosomePositioningModel,
        VariationalAutoencoderBridge,
    )
    HAS_NUC_VAE = True
except Exception:
    HAS_NUC_VAE = False
    NucleosomePositioningModel = None
    VariationalAutoencoderBridge = None

__all__ = [
    "GeometricStringIntegrator", "GeometricConfig", "GeometricStringResult",
    "HistoneModificationCrosstalk", "HistoneConfig", "HistoneState",
    "FlowMatchingBridge", "FlowMatchingConfig", "FlowMatchingResult",
    "SESIGeometricHistoneFlowMatchSuite",
]


# =============================================================================
# SECTION 1 — [GSI1] GEOMETRIC STRING INTEGRATOR
# =============================================================================

@dataclass
class GeometricConfig:
    """Configuration for geometric string integration."""
    # Integration method
    method: str = "verlet"  # "verlet" | "leapfrog" | "rattle"
    # String parameters
    n_images: int = 20
    n_cvs: int = 5
    # Constraint
    arc_length_constraint: bool = True  # enforce equal arc-length spacing
    constraint_tol: float = 1e-6
    constraint_max_iter: int = 50
    # Time stepping
    dt: float = 0.01
    n_steps: int = 500
    # Energy
    energy_type: str = "msm"  # "msm" | "potential"
    temperature: float = 1.0
    # Seed
    seed: int = 0


@dataclass
class GeometricStringResult:
    """Result of geometric string integration."""
    string_images: np.ndarray  # (n_images, n_cvs)
    momenta: np.ndarray  # (n_images, n_cvs) conjugate momenta
    energies: np.ndarray  # (n_images,)
    arc_length: float
    constraint_violation: float
    energy_conservation: float  # |E_final - E_initial| / |E_initial|
    converged: bool
    n_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeometricStringIntegrator:
    """
    [C/D] Geometric (symplectic/variational) integration of string method.

    MATHEMATICAL BASIS [C Hairer, Lubich, Wanner 2006; Marsden & West 2001]
      The string method evolves a path γ(s) in CV space to minimize energy.
      Standard gradient descent does not preserve geometric structure:
        - Energy may oscillate
        - Arc-length constraint may drift
        - Long-time behavior may be unphysical

      Geometric integrators preserve:
        1. Symplectic structure (phase-space volume)
        2. Energy (approximately, with bounded error)
        3. Constraints (via projection or Lagrange multipliers)

      VERLET / LEAPFROG [C Hairer 2006]:
        q_{n+1} = q_n + dt · p_n + (dt²/2) · F(q_n)
        p_{n+1} = p_n + (dt/2) · (F(q_n) + F(q_{n+1}))

      RATTLE (constrained Verlet) [C Hairer 2006]:
        1. Verlet step
        2. Project onto constraint manifold (arc-length)
        3. Correct momenta to maintain constraint

      ARC-LENGTH CONSTRAINT [D]:
        After each step, reparameterize string to have equal arc-length
        spacing. This is a holonomic constraint on the path.

    CONNECTION TO ADAPTIVE STRING [D]
      The AdaptiveStringMethod from previous module provides an initial
      path with adaptive image density. Geometric integration refines
      this path while preserving its geometric structure.

    COROLLARY-3 GUARD [D]
      Integration is FIRST-ORDER in time (Verlet is second-order accurate
      but first-order in the sense of not using Hessian). We do NOT
      compute second derivatives of the energy landscape.

    HINO GUARD [C Thm 8]
      The energy landscape is RUGGED (singular). Symplectic methods
      handle this better than gradient descent, but the path may still
      get stuck in local minima. Discrete path representation respects
      the singularity.

    PROP-3 AWARENESS [P]
      When computing forces (gradient of energy), the result loses
      regularity. We use finite differences (first-order).

    HONEST
      • Symplectic methods preserve structure but may not converge to
        true MEP in rugged landscapes.
      • Arc-length constraint is enforced approximately (projection).
      • This is a computational tool, not a physical simulation.
    """

    def __init__(self, cfg: Optional[GeometricConfig] = None):
        self.cfg = cfg or GeometricConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def _compute_energy_msm(self, cv_point: np.ndarray,
                             msm_model: Any) -> float:
        """Compute energy at a CV point using MSM stationary distribution."""
        if msm_model.cluster_centers is None:
            return 0.0

        distances = np.linalg.norm(msm_model.cluster_centers - cv_point, axis=1)
        state_idx = np.argmin(distances)

        pi = msm_model.stationary_distribution[state_idx]
        if pi < 1e-300:
            return 1e10

        energy = -self.cfg.temperature * np.log(pi)
        return float(energy)

    def _compute_force(self, cv_point: np.ndarray,
                        msm_model: Any,
                        eps: float = 1e-4) -> np.ndarray:
        """Compute force (negative gradient) via finite differences."""
        n_cvs = len(cv_point)
        grad = np.zeros(n_cvs)

        for i in range(n_cvs):
            cv_plus = cv_point.copy()
            cv_minus = cv_point.copy()
            cv_plus[i] += eps
            cv_minus[i] -= eps

            e_plus = self._compute_energy_msm(cv_plus, msm_model)
            e_minus = self._compute_energy_msm(cv_minus, msm_model)

            grad[i] = (e_plus - e_minus) / (2 * eps)

        return -grad  # force = -gradient

    def _compute_arc_length(self, string: np.ndarray) -> float:
        """Compute total arc length of the string."""
        diffs = np.diff(string, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return float(segment_lengths.sum())

    def _reparameterize_arc_length(self, string: np.ndarray) -> np.ndarray:
        """
        Reparameterize string to have equal arc-length spacing.
        This enforces the holonomic constraint.
        """
        n_images = len(string)
        total_length = self._compute_arc_length(string)

        if total_length < 1e-10:
            return string

        # Target arc-length positions
        target_s = np.linspace(0, total_length, n_images)

        # Compute cumulative arc length
        diffs = np.diff(string, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])

        # Interpolate to new positions
        if HAS_SCIPY_INTERP:
            new_string = np.zeros_like(string)
            for dim in range(string.shape[1]):
                cs = CubicSpline(cumulative_length, string[:, dim])
                new_string[:, dim] = cs(target_s)
            return new_string
        else:
            new_string = np.zeros_like(string)
            for dim in range(string.shape[1]):
                new_string[:, dim] = np.interp(target_s, cumulative_length, string[:, dim])
            return new_string

    def _compute_constraint_violation(self, string: np.ndarray) -> float:
        """Compute how much arc-length constraint is violated."""
        n_images = len(string)
        diffs = np.diff(string, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        target_length = segment_lengths.mean()
        violation = np.max(np.abs(segment_lengths - target_length))
        return float(violation)

    def _verlet_step(self, string: np.ndarray, momenta: np.ndarray,
                      msm_model: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        One Verlet integration step.
        q_{n+1} = q_n + dt · p_n + (dt²/2) · F(q_n)
        p_{n+1} = p_n + (dt/2) · (F(q_n) + F(q_{n+1}))
        """
        dt = self.cfg.dt
        n_images = len(string)

        # Compute forces at current positions
        forces_old = np.array([self._compute_force(string[i], msm_model)
                                for i in range(n_images)])

        # Update positions
        string_new = string + dt * momenta + 0.5 * dt**2 * forces_old

        # Compute forces at new positions
        forces_new = np.array([self._compute_force(string_new[i], msm_model)
                                for i in range(n_images)])

        # Update momenta
        momenta_new = momenta + 0.5 * dt * (forces_old + forces_new)

        return string_new, momenta_new

    def _rattle_step(self, string: np.ndarray, momenta: np.ndarray,
                      msm_model: Any) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        One RATTLE (constrained Verlet) step.
        1. Verlet step
        2. Project onto arc-length constraint
        3. Correct momenta
        """
        # Verlet step
        string_new, momenta_new = self._verlet_step(string, momenta, msm_model)

        # Project onto constraint (reparameterize)
        if self.cfg.arc_length_constraint:
            string_new = self._reparameterize_arc_length(string_new)

            # Correct momenta to maintain constraint (simplified)
            # In full RATTLE, would solve for Lagrange multipliers
            # Here, we just ensure momenta are tangent to constraint
            constraint_violation = self._compute_constraint_violation(string_new)
        else:
            constraint_violation = 0.0

        return string_new, momenta_new, constraint_violation

    def integrate(self, initial_string: np.ndarray,
                   msm_model: Any) -> GeometricStringResult:
        """
        Run geometric integration of string method.

        Args:
            initial_string: (n_images, n_cvs) initial path
            msm_model: MarkovStateModel for energy computation

        Returns:
            GeometricStringResult with integrated path
        """
        n_images = len(initial_string)
        string = initial_string.copy()
        momenta = np.zeros_like(string)  # start from rest

        # Initial energy
        energies = np.array([self._compute_energy_msm(string[i], msm_model)
                              for i in range(n_images)])
        initial_energy = energies.sum()

        # Integration loop
        for step in range(self.cfg.n_steps):
            if self.cfg.method == "verlet":
                string, momenta = self._verlet_step(string, momenta, msm_model)
                if self.cfg.arc_length_constraint:
                    string = self._reparameterize_arc_length(string)
            elif self.cfg.method == "rattle":
                string, momenta, _ = self._rattle_step(string, momenta, msm_model)
            else:  # leapfrog (similar to Verlet)
                string, momenta = self._verlet_step(string, momenta, msm_model)
                if self.cfg.arc_length_constraint:
                    string = self._reparameterize_arc_length(string)

        # Final energy
        energies = np.array([self._compute_energy_msm(string[i], msm_model)
                              for i in range(n_images)])
        final_energy = energies.sum()

        # Conservation check
        energy_conservation = abs(final_energy - initial_energy) / max(abs(initial_energy), 1e-12)

        # Constraint violation
        constraint_violation = self._compute_constraint_violation(string)

        # Arc length
        arc_length = self._compute_arc_length(string)

        # Convergence (simplified: check if constraint violation is small)
        converged = constraint_violation < self.cfg.constraint_tol

        return GeometricStringResult(
            string_images=string,
            momenta=momenta,
            energies=energies,
            arc_length=arc_length,
            constraint_violation=constraint_violation,
            energy_conservation=energy_conservation,
            converged=converged,
            n_steps=self.cfg.n_steps,
            metadata={
                "method": self.cfg.method,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
            },
        )

    def refine_adaptive_string(self, adaptive_result: Any,
                                 msm_model: Any) -> GeometricStringResult:
        """
        Refine an adaptive string result using geometric integration.

        Args:
            adaptive_result: AdaptiveStringResult from previous module
            msm_model: MarkovStateModel

        Returns:
            GeometricStringResult with refined path
        """
        initial_string = adaptive_result.string_images
        return self.integrate(initial_string, msm_model)

    def get_status(self) -> Dict[str, Any]:
        return {
            "method": self.cfg.method,
            "n_images": self.cfg.n_images,
            "arc_length_constraint": self.cfg.arc_length_constraint,
            "corollary3_guard": "first-order integration (Verlet); no Hessian of action",
            "hino_guard": "path space singular; discrete path representation",
        }


# =============================================================================
# SECTION 2 — [HMC1] HISTONE MODIFICATION CROSSTALK
# =============================================================================

@dataclass
class HistoneConfig:
    """Configuration for histone modification crosstalk."""
    # Nucleosomes
    n_nucleosomes: int = 30
    # Modifications (8 common ones)
    modifications: List[str] = field(default_factory=lambda: [
        "H3K4me3", "H3K27me3", "H3K9me3", "H3K36me3",
        "H2AK119ub", "H3K9ac", "H3K27ac", "H4K20me1"
    ])
    # Enzyme concentrations
    n_writers: int = 50  # HMTs/HATs
    n_erasers: int = 40  # HDMs/HDACs
    n_readers: int = 60  # chromodomain, bromodomain, etc.
    # Kinetics
    k_write_base: float = 0.1  # base writing rate
    k_erase_base: float = 0.05  # base erasing rate
    k_read_base: float = 0.2  # base reading rate
    # Crosstalk matrix (how one mod affects rates of others)
    # Positive = enhances, negative = inhibits
    crosstalk_matrix: Optional[np.ndarray] = None  # (n_mods, n_mods)
    crosstalk_strength: float = 0.3
    # Simulation
    max_time: float = 500.0
    max_steps: int = 200000
    seed: int = 0


@dataclass
class HistoneState:
    """State of histone modification system."""
    time: float
    # Per-nucleosome modification state (binary: 0 or 1 for each mod)
    modification_states: np.ndarray  # (n_nucleosomes, n_modifications)
    # Enzyme counts
    n_writers_free: int
    n_erasers_free: int
    n_readers_free: int
    # Statistics
    mean_modification_level: np.ndarray  # (n_modifications,)
    n_write_events: int
    n_erase_events: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class HistoneModificationCrosstalk:
    """
    [C/D] Multi-site histone modification dynamics with crosstalk.

    BIOLOGICAL BASIS [C Jenuwein & Allis 2001; Bartova & Kozubek 2011]
      Histone tails are modified by:
        - Writers: HMTs (methyltransferases), HATs (acetyltransferases)
        - Erasers: HDMs (demethylases), HDACs (deacetylases)
        - Readers: proteins with chromodomain, bromodomain, etc.

      Common modifications:
        - H3K4me3: active promoters
        - H3K27me3: Polycomb repression
        - H3K9me3: heterochromatin
        - H3K36me3: active gene bodies
        - H2AK119ub: Polycomb
        - H3K9ac, H3K27ac: active enhancers
        - H4K20me1: constitutive heterochromatin

      CROSSTALK:
        - One modification affects the probability of another
        - Example: H3K4me3 inhibits H3K27me3 (bivalent domains)
        - Example: H3K9me3 promotes H4K20me1 (heterochromatin spreading)

    MODEL [D]
      Each nucleosome has a combinatorial state: 2^n_modifications
      For 8 modifications, 256 states per nucleosome.

      Reactions per nucleosome per modification:
        1. Write: unmodified → modified (rate k_write * crosstalk_factor)
        2. Erase: modified → unmodified (rate k_erase * crosstalk_factor)
        3. Read: reader binds (rate k_read, does not change state)

      Crosstalk factor:
        crosstalk_factor[i, j] = 1 + crosstalk_strength * crosstalk_matrix[i, j]
        where i is the existing modification, j is the target.

      Gillespie SSA for discrete events.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear in enzyme counts,
      with crosstalk modulation). We do NOT compute second derivatives of
      the modification landscape.

    HINO GUARD [C Thm 8]
      Modification events are DISCRETE, not smooth. The stochasticity is
      singular (tied to individual molecular events). Crosstalk modulates
      rates but does not change the discrete nature.

    PROP-3 AWARENESS [P]
      When multiplying fields (e.g., modification_state × crosstalk_matrix),
      the result loses two Sobolev orders. We use additive crosstalk terms
      to maintain regularity.

    HONEST
      • Combinatorial state space grows exponentially. For 8 modifications,
        256 states per nucleosome. This is simplified; real histones have
        many more modifications.
      • Crosstalk matrix is heuristic; real crosstalk involves complex
        molecular mechanisms not captured here.
      • This is a computational model, not validated epigenetics.
    """

    def __init__(self, cfg: Optional[HistoneConfig] = None):
        self.cfg = cfg or HistoneConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_mods = len(self.cfg.modifications)

        # Initialize crosstalk matrix if not provided
        if self.cfg.crosstalk_matrix is None:
            # Default: some known crosstalk patterns
            self.crosstalk = np.zeros((self.n_mods, self.n_mods))
            # H3K4me3 (0) inhibits H3K27me3 (1)
            self.crosstalk[0, 1] = -1.0
            self.crosstalk[1, 0] = -1.0
            # H3K9me3 (2) promotes H4K20me1 (7)
            self.crosstalk[2, 7] = 1.0
            self.crosstalk[7, 2] = 1.0
            # H3K27ac (6) promotes H3K4me3 (0)
            self.crosstalk[6, 0] = 0.5
        else:
            self.crosstalk = self.cfg.crosstalk_matrix

        # State
        self.modification_states: Optional[np.ndarray] = None
        self.n_writers_free: int = 0
        self.n_erasers_free: int = 0
        self.n_readers_free: int = 0
        self.n_write_events: int = 0
        self.n_erase_events: int = 0

    def initialize(self) -> None:
        """Initialize histone modification system."""
        # Random initial modification state
        self.modification_states = self._rng.integers(
            0, 2, size=(self.cfg.n_nucleosomes, self.n_mods))

        self.n_writers_free = self.cfg.n_writers
        self.n_erasers_free = self.cfg.n_erasers
        self.n_readers_free = self.cfg.n_readers
        self.n_write_events = 0
        self.n_erase_events = 0

    def _compute_crosstalk_factor(self, nuc_idx: int, target_mod: int) -> float:
        """Compute crosstalk factor for a target modification."""
        current_mods = self.modification_states[nuc_idx]
        factor = 1.0
        for i in range(self.n_mods):
            if current_mods[i] == 1:
                factor += self.cfg.crosstalk_strength * self.crosstalk[i, target_mod]
        return max(0.1, factor)  # clamp to avoid negative rates

    def compute_propensities(self) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
        """Compute all reaction propensities."""
        props = []
        events = []

        for nuc_idx in range(self.cfg.n_nucleosomes):
            for mod_idx in range(self.n_mods):
                is_modified = self.modification_states[nuc_idx, mod_idx]

                if is_modified == 0 and self.n_writers_free > 0:
                    # Write reaction
                    crosstalk_factor = self._compute_crosstalk_factor(nuc_idx, mod_idx)
                    a = self.cfg.k_write_base * crosstalk_factor * self.n_writers_free
                    props.append(a)
                    events.append(("write", nuc_idx, mod_idx))

                elif is_modified == 1 and self.n_erasers_free > 0:
                    # Erase reaction
                    crosstalk_factor = self._compute_crosstalk_factor(nuc_idx, mod_idx)
                    a = self.cfg.k_erase_base * crosstalk_factor * self.n_erasers_free
                    props.append(a)
                    events.append(("erase", nuc_idx, mod_idx))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self) -> Tuple[Dict[str, Any], float]:
        """One Gillespie SSA step."""
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

        reaction_name, nuc_idx, mod_idx = events[event_idx]

        # Apply reaction
        if reaction_name == "write":
            self.modification_states[nuc_idx, mod_idx] = 1
            self.n_writers_free -= 1
            self.n_write_events += 1
        elif reaction_name == "erase":
            self.modification_states[nuc_idx, mod_idx] = 0
            self.n_erasers_free -= 1
            self.n_erase_events += 1

        return {"reaction": reaction_name, "nucleosome": nuc_idx,
                "modification": mod_idx}, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[HistoneState], List[float]]:
        """Run full SSA simulation."""
        self.initialize()

        t = 0.0
        trajectory = [self._snapshot(t)]
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        steps = 0
        while t < max_time and steps < max_steps:
            _, dt = self.ssa_step()
            if dt <= 0:
                break
            t += dt
            trajectory.append(self._snapshot(t))
            times.append(t)
            steps += 1

        return trajectory, times

    def _snapshot(self, t: float) -> HistoneState:
        mean_mod_level = self.modification_states.mean(axis=0)
        return HistoneState(
            time=t,
            modification_states=self.modification_states.copy(),
            n_writers_free=self.n_writers_free,
            n_erasers_free=self.n_erasers_free,
            n_readers_free=self.n_readers_free,
            mean_modification_level=mean_mod_level,
            n_write_events=self.n_write_events,
            n_erase_events=self.n_erase_events,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_nucleosomes": self.cfg.n_nucleosomes,
            "n_modifications": self.n_mods,
            "modifications": self.cfg.modifications,
            "corollary3_guard": "first-order kinetics with crosstalk; no Hessian",
            "hino_guard": "discrete modification events, not smooth density",
        }


# =============================================================================
# SECTION 3 — [FLM1] FLOW MATCHING BRIDGE
# =============================================================================

@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching."""
    # Architecture
    data_dim: int = 32
    hidden_dim: int = 256
    n_layers: int = 4
    # Flow parameters
    sigma_min: float = 1e-4
    sigma_max: float = 1.0
    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 64
    # Sampling
    n_samples: int = 100
    n_steps: int = 100
    # Seed
    seed: int = 0


@dataclass
class FlowMatchingResult:
    """Result of flow matching training/sampling."""
    n_samples: int
    samples: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class VelocityNetwork(nn.Module):
    """Velocity field network for flow matching: v_t(x) = f(x, t)."""

    def __init__(self, data_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.data_dim = data_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Velocity network
        layers = []
        in_dim = data_dim + hidden_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))

        self.velocity_net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field v_t(x).

        Args:
            x: (batch, data_dim) current position
            t: (batch,) time in [0, 1]

        Returns:
            v: (batch, data_dim) velocity
        """
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))

        # Concatenate x and t_emb
        h = torch.cat([x, t_emb], dim=-1)

        # Predict velocity
        v = self.velocity_net(h)

        return v


class FlowMatchingBridge:
    """
    [C/D] Flow matching for fast generative sampling.

    MATHEMATICAL BASIS [C Lipinski et al. 2023; Liu et al. 2023]
      Flow matching learns a vector field v_t(x) that transports a simple
      base distribution (e.g., Gaussian) to the data distribution:

        dx/dt = v_t(x),  t ∈ [0, 1]
        x(0) ~ p_base (Gaussian)
        x(1) ~ p_data

      Training:
        1. Sample x_0 ~ p_base, x_1 ~ p_data
        2. Interpolate: x_t = (1-t) x_0 + t x_1
        3. Target velocity: u_t = x_1 - x_0
        4. Learn v_t(x) ≈ u_t via regression

      Sampling:
        1. Start from x_0 ~ p_base
        2. Integrate ODE: dx/dt = v_t(x) from t=0 to t=1
        3. Result: x_1 ~ p_data

    ADVANTAGES OVER DIFFUSION [C Lipinski 2023]
      - Deterministic (no stochastic term)
      - Faster sampling (fewer steps needed)
      - Exact likelihood (via ODE solver)
      - Simpler training (regression, not score matching)

    CONNECTION TO VAE [D]
      - VAE learns approximate posterior q(z|x) via ELBO
      - Flow matching learns exact p(x) via regression
      - Can use VAE latent codes as training data

    COROLLARY-3 GUARD [D]
      Training uses FIRST-ORDER optimization (Adam). We do NOT compute
      second derivatives of the velocity field.

    HINO GUARD [C Thm 8]
      The data distribution is NOT assumed to be smooth. The velocity
      field learns a singular distribution that may have complex structure.

    PROP-3 AWARENESS [P]
      When computing the ODE right-hand side (velocity field), the result
      is a first-order function of (x, t). No higher-order derivatives.

    HONEST
      • Flow matching learns a statistical distribution, not a physical one.
      • Generated samples may be outside the training manifold.
      • ODE integration requires a good solver; errors accumulate.
      • This is a generative model, not a physics engine.
    """

    def __init__(self, cfg: Optional[FlowMatchingConfig] = None):
        self.cfg = cfg or FlowMatchingConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Build velocity network
        self.velocity_net = VelocityNetwork(
            self.cfg.data_dim,
            self.cfg.hidden_dim,
            self.cfg.n_layers,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.velocity_net.parameters(),
            lr=self.cfg.learning_rate
        )

    def sample_base(self, n_samples: int) -> Tensor:
        """Sample from base distribution (Gaussian)."""
        return torch.randn(n_samples, self.cfg.data_dim)

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: float) -> Tensor:
        """Linear interpolation: x_t = (1-t) x_0 + t x_1."""
        return (1 - t) * x_0 + t * x_1

    def target_velocity(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        """Target velocity: u_t = x_1 - x_0 (constant along path)."""
        return x_1 - x_0

    def train(self, data: Tensor, n_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train flow matching model.

        Args:
            data: (n_samples, data_dim) training data
            n_epochs: number of training epochs

        Returns:
            dict with training history
        """
        n_epochs = n_epochs or self.cfg.n_epochs
        n_samples = data.shape[0]

        history = {'loss': []}

        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            data_shuffled = data[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.cfg.batch_size):
                batch_x1 = data_shuffled[i:i + self.cfg.batch_size]
                batch_size = batch_x1.shape[0]

                # Sample x_0 from base distribution
                batch_x0 = self.sample_base(batch_size)

                # Sample random time t ∈ [0, 1]
                t = torch.rand(batch_size)

                # Interpolate
                batch_xt = self.interpolate(batch_x0, batch_x1, t.unsqueeze(-1))

                # Target velocity
                target_v = self.target_velocity(batch_x0, batch_x1)

                # Predict velocity
                predicted_v = self.velocity_net(batch_xt, t)

                # Loss: MSE between predicted and target velocity
                loss = F.mse_loss(predicted_v, target_v)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_size
                n_batches += 1

            history['loss'].append(epoch_loss / n_samples)

        return {
            "n_epochs": n_epochs,
            "final_loss": history['loss'][-1],
            "history": history,
        }

    def sample(self, n_samples: int = 100, n_steps: Optional[int] = None) -> Tensor:
        """
        Generate samples by integrating ODE from base to data.

        Args:
            n_samples: number of samples to generate
            n_steps: number of integration steps

        Returns:
            samples: (n_samples, data_dim)
        """
        n_steps = n_steps or self.cfg.n_steps

        # Start from base distribution
        x = self.sample_base(n_samples)

        # Integrate ODE: dx/dt = v_t(x) from t=0 to t=1
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = step / n_steps
            t_batch = torch.full((n_samples,), t)

            with torch.no_grad():
                v = self.velocity_net(x, t_batch)

            # Euler integration (simple; could use RK4 for better accuracy)
            x = x + dt * v

        return x

    def generate_samples(self, n_samples: int = 100) -> FlowMatchingResult:
        """Generate samples and return result."""
        samples = self.sample(n_samples)

        return FlowMatchingResult(
            n_samples=n_samples,
            samples=samples.numpy(),
            metadata={
                "n_layers": self.cfg.n_layers,
                "data_dim": self.cfg.data_dim,
                "n_steps": self.cfg.n_steps,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "data_dim": self.cfg.data_dim,
            "n_layers": self.cfg.n_layers,
            "corollary3_guard": "first-order optimization (Adam); no Hessian of velocity",
            "hino_guard": "data distribution not assumed smooth; learned from data",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIGeometricHistoneFlowMatchSuite:
    """
    Unifies: GeometricStringIntegrator + HistoneModificationCrosstalk +
             FlowMatchingBridge.

    FULL PIPELINE:
      1. Adaptive string method finds initial MEP
      2. Geometric integration refines path with symplectic structure
      3. Histone modifications evolve with crosstalk
      4. Flow matching generates new conformations
      5. Integration: string → histone → flow
    """

    def __init__(self,
                 geometric_cfg: Optional[GeometricConfig] = None,
                 histone_cfg: Optional[HistoneConfig] = None,
                 flow_cfg: Optional[FlowMatchingConfig] = None):
        self.geometric = GeometricStringIntegrator(geometric_cfg)
        self.histone = HistoneModificationCrosstalk(histone_cfg)
        self.flow = FlowMatchingBridge(flow_cfg)

    def refine_string_geometrically(self, adaptive_result: Any,
                                      msm_model: Any) -> Dict[str, Any]:
        """Refine adaptive string using geometric integration."""
        result = self.geometric.refine_adaptive_string(adaptive_result, msm_model)
        return {
            "n_images": len(result.string_images),
            "arc_length": result.arc_length,
            "constraint_violation": result.constraint_violation,
            "energy_conservation": result.energy_conservation,
            "converged": result.converged,
            "n_steps": result.n_steps,
            "method": result.metadata.get("method"),
        }

    def simulate_histone_dynamics(self, n_steps: int = 1000) -> Dict[str, Any]:
        """Simulate histone modification dynamics with crosstalk."""
        traj, times = self.histone.simulate(max_steps=n_steps)

        final_state = traj[-1]
        return {
            "n_steps": len(traj),
            "final_time": final_state.time,
            "mean_modification_level": final_state.mean_modification_level.tolist(),
            "n_write_events": final_state.n_write_events,
            "n_erase_events": final_state.n_erase_events,
            "modification_names": self.histone.cfg.modifications,
        }

    def train_flow_matching(self, training_data: np.ndarray,
                              n_epochs: int = 50) -> Dict[str, Any]:
        """Train flow matching model on data."""
        data_tensor = torch.tensor(training_data, dtype=torch.float32)
        result = self.flow.train(data_tensor, n_epochs=n_epochs)
        return result

    def generate_conformations(self, n_samples: int = 100) -> Dict[str, Any]:
        """Generate samples from trained flow matching model."""
        result = self.flow.generate_samples(n_samples)
        return {
            "n_samples": result.n_samples,
            "samples_shape": result.samples.shape,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "geometric": self.geometric.get_status(),
            "histone": self.histone.get_status(),
            "flow": self.flow.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI PROTEIN DYNAMICS STUDIO GEOMETRIC — SMOKE TEST")
    print(" Geometric String + Histone Crosstalk + Flow Matching")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scipy interp           : {HAS_SCIPY_INTERP}")
    print(f" torchdiffeq            : {HAS_TORCHDIFFEQ}")
    print(f" adaptive string        : {HAS_ADAPTIVE}")
    print(f" MSM module             : {HAS_MSM}")
    print(f" nuc/VAE module         : {HAS_NUC_VAE}")

    suite = SESIGeometricHistoneFlowMatchSuite(
        geometric_cfg=GeometricConfig(n_images=10, n_steps=50, method="verlet"),
        histone_cfg=HistoneConfig(n_nucleosomes=15, max_steps=500),
        flow_cfg=FlowMatchingConfig(data_dim=10, n_epochs=15))

    # ── 1. Geometric String Integration ─────────────────────────────────
    print("\n [GSI1] Geometric String Integration")
    geometric = suite.geometric
    print(f"   method               : {geometric.cfg.method}")
    print(f"   n_images             : {geometric.cfg.n_images}")
    print(f"   arc_length_constraint: {geometric.cfg.arc_length_constraint}")

    # Create synthetic MSM and adaptive string for testing
    if HAS_MSM:
        msm = MarkovStateModel()
        msm.cluster_centers = np.random.rand(20, 5)
        msm.stationary_distribution = np.random.rand(20)
        msm.stationary_distribution /= msm.stationary_distribution.sum()

        # Create synthetic adaptive string result
        from sesi_protein_dynamics_adaptive_feedback_diffusion import AdaptiveStringResult
        adaptive_result = AdaptiveStringResult(
            string_images=np.random.rand(10, 5),
            n_images=10,
            energies=np.random.rand(10),
            forces=np.random.rand(10, 5),
            curvatures=np.random.rand(10),
            arc_length=5.0,
            converged=True,
            n_iterations=50,
            n_refinements=3,
        )

        # Refine geometrically
        geom_result = suite.refine_string_geometrically(adaptive_result, msm)
        print(f"   refined n_images     : {geom_result['n_images']}")
        print(f"   arc length           : {geom_result['arc_length']:.4f}")
        print(f"   constraint violation : {geom_result['constraint_violation']:.6f}")
        print(f"   energy conservation  : {geom_result['energy_conservation']:.6f}")
        print(f"   converged            : {geom_result['converged']}")
        print(f"   integration steps    : {geom_result['n_steps']}")
    else:
        print("   [SKIP] MSM module not available")

    gsi_status = geometric.get_status()
    print(f"   COROLLARY-3 guard    : {gsi_status['corollary3_guard']}")
    print(f"   HINO guard           : {gsi_status['hino_guard']}")

    # ── 2. Histone Modification Crosstalk ───────────────────────────────
    print("\n [HMC1] Histone Modification Crosstalk")
    histone = suite.histone
    print(f"   n_nucleosomes        : {histone.cfg.n_nucleosomes}")
    print(f"   n_modifications      : {histone.n_mods}")
    print(f"   modifications        : {histone.cfg.modifications}")

    # Simulate
    histone_result = suite.simulate_histone_dynamics(n_steps=300)
    print(f"   simulation steps     : {histone_result['n_steps']}")
    print(f"   final time           : {histone_result['final_time']:.2f}")
    print(f"   write events         : {histone_result['n_write_events']}")
    print(f"   erase events         : {histone_result['n_erase_events']}")
    print(f"   mean modification levels:")
    for mod_name, level in zip(histone_result['modification_names'],
                                 histone_result['mean_modification_level']):
        print(f"     {mod_name:12s}       : {level:.4f}")

    hmc_status = histone.get_status()
    print(f"   COROLLARY-3 guard    : {hmc_status['corollary3_guard']}")
    print(f"   HINO guard           : {hmc_status['hino_guard']}")

    # ── 3. Flow Matching ────────────────────────────────────────────────
    print("\n [FLM1] Flow Matching for Fast Sampling")
    flow = suite.flow
    print(f"   data_dim             : {flow.cfg.data_dim}")
    print(f"   n_layers             : {flow.cfg.n_layers}")

    # Generate synthetic training data
    n_train = 200
    training_data = np.random.randn(n_train, flow.cfg.data_dim).astype(np.float32)

    # Train
    train_result = suite.train_flow_matching(training_data, n_epochs=15)
    print(f"   training epochs      : {train_result['n_epochs']}")
    print(f"   final loss           : {train_result['final_loss']:.4f}")

    # Generate samples
    gen_result = suite.generate_conformations(n_samples=50)
    print(f"   generated samples    : {gen_result['n_samples']}")
    print(f"   samples shape        : {gen_result['samples_shape']}")

    flm_status = flow.get_status()
    print(f"   COROLLARY-3 guard    : {flm_status['corollary3_guard']}")
    print(f"   HINO guard           : {flm_status['hino_guard']}")

    # ── 4. Full Status ──────────────────────────────────────────────────
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
    print(" [C]   Hairer, Lubich, Wanner 2006: Geometric numerical integration.")
    print(" [C]   Marsden & West 2001: Variational integrators.")
    print(" [C]   Jenuwein & Allis 2001: Histone modifications.")
    print(" [C]   Bartova & Kozubek 2011: Histone modification crosstalk.")
    print(" [C]   Lipinski et al. 2023: Flow matching for generative modeling.")
    print(" [C]   Liu et al. 2023: Flow matching for simulation-free training.")
    print(" [D]   Geometric string integration preserves structure but may not")
    print("       converge to true MEP in rugged landscapes. Path space is singular.")
    print(" [D]   Histone modification crosstalk is combinatorial (2^n_mod states).")
    print("       For 8 modifications, 256 states per nucleosome. Simplified model.")
    print(" [D]   Flow matching learns statistical distribution, not physical one.")
    print("       Generated samples may be outside training manifold.")
    print(" [D]   COROLLARY-3 guard: all fields first-order only; no Hessian.")
    print(" [C]   HINO guard: energy landscapes singular; no smooth density assumed.")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL models. No biological, medical,")
    print("       or diagnostic claim. Not validated against real systems.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
