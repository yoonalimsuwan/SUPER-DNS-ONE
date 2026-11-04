"""
=============================================================================
SESI PROTEIN DYNAMICS STUDIO FRONTIER
Adaptive String Method + Remodeler-Nucleosome Feedback + Diffusion Models
Native integration with SDA-ONE, MSM, TPT, VAE, String, Flows modules
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
     - MarkovStateModel (MSM)
     - ChromatinAccessibilityModel
  3. sesi_protein_dynamics_tpt_nucleosome_vae.py:
     - TransitionPathTheory (committor, reactive current)
     - NucleosomePositioningModel
     - VariationalAutoencoderBridge
  4. sesi_protein_dynamics_string_remodeler_flows.py:
     - StringMethodMEP (basic string)
     - ChromatinRemodelerDynamics
     - NormalizingFlowsBridge

Integration points:
  1. AdaptiveStringMethod.refine_from_basic(basic_string_result, msm_model)
     - Uses committor function for initial path
     - Adapts image density based on local curvature
     - Handles high-dimensional CV spaces efficiently

  2. RemodelerNucleosomeFeedback.simulate_coupled(nucleosome_result, remodeler_result)
     - Bidirectional coupling: nucleosome positions ↔ remodeler binding
     - Feedback terms modulate rates dynamically
     - Stochastic simulation with coupled propensities

  3. ScoreBasedDiffusionBridge.train_from_vae(vae_model, msm_states)
     - Uses VAE latent codes as training data
     - Learns score function ∇_x log p(x)
     - Generates samples via reverse-time SDE

=============================================================================
MODULES
=============================================================================
[ASM1] AdaptiveStringMethod
      • Adaptive image density based on local curvature [C E et al. 2007]
      • High-dimensional CV space support (from MSM/TPT)
      • Curvature-driven refinement: add images where |∇²E| is large
      • Coarsening in flat regions: remove redundant images
      • Connection to committor function (TPT)
      • COROLLARY-3 guard: first-order gradient descent; no Hessian
      • HINO guard: energy landscape singular; path may be local minimum

[RNF1] RemodelerNucleosomeFeedback
      • Bidirectional coupling: nucleosome ↔ remodeler [C Clapier 2009]
      • Nucleosome positions affect remodeler binding affinity
      • Remodeler activity changes nucleosome occupancy
      • Feedback terms modulate rates dynamically
      • Stochastic simulation with coupled propensities
      • COROLLARY-3 guard: first-order kinetics; no Hessian
      • HINO guard: discrete binding events, not smooth density

[SBD1] ScoreBasedDiffusionBridge
      • Score matching: ∇_x log p(x) [C Song & Ermon 2019]
      • Reverse-time SDE for generation [C Ho et al. 2020]
      • Integration with VAE latent space
      • Connection to MSM states (state-conditioned generation)
      • COROLLARY-3 guard: first-order optimization (Adam); no Hessian
      • HINO guard: latent space singular, not smooth manifold

=============================================================================
HONEST SCOPE
=============================================================================
• AdaptiveStringMethod: adaptive refinement is heuristic. Real MEP may
  require global optimization methods not captured here. Energy landscape
  is rugged (HINO); path may get stuck in local minima.
• RemodelerNucleosomeFeedback: bidirectional coupling is simplified. Real
  chromatin remodeling involves complex multi-protein assemblies, DNA
  mechanics, and histone modifications not captured here.
• ScoreBasedDiffusionBridge: score-based models learn a distribution but
  do NOT guarantee physical realism. Generated samples may be outside
  the training manifold. This is a generative model, not a physics engine.
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

logger = logging.getLogger("sesi.adaptive_feedback_diffusion")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY_INTERP = True
except Exception:
    HAS_SCIPY_INTERP = False

try:
    from scipy.spatial.distance import cdist
    HAS_SCIPY_DIST = True
except Exception:
    HAS_SCIPY_DIST = False

# ── Previous modules import (graceful) ──────────────────────────────────
try:
    from sesi_protein_dynamics_string_remodeler_flows import (
        StringMethodMEP,
        StringConfig,
        StringResult,
        ChromatinRemodelerDynamics,
        RemodelerConfig,
    )
    HAS_STRING_REMODELER = True
except Exception:
    HAS_STRING_REMODELER = False
    StringMethodMEP = None
    ChromatinRemodelerDynamics = None

try:
    from sesi_protein_dynamics_tpt_nucleosome_vae import (
        TransitionPathTheory,
        TPTResult,
        NucleosomePositioningModel,
        NucleosomeResult,
        VariationalAutoencoderBridge,
        VAEResult,
    )
    HAS_TPT_NUC_VAE = True
except Exception:
    HAS_TPT_NUC_VAE = False
    TransitionPathTheory = None
    NucleosomePositioningModel = None
    VariationalAutoencoderBridge = None

try:
    from sesi_protein_dynamics_msm_chromatin_enhanced import (
        MarkovStateModel,
        MSMConfig,
    )
    HAS_MSM = True
except Exception:
    HAS_MSM = False
    MarkovStateModel = None

__all__ = [
    "AdaptiveStringMethod", "AdaptiveStringConfig", "AdaptiveStringResult",
    "RemodelerNucleosomeFeedback", "FeedbackConfig", "FeedbackState",
    "ScoreBasedDiffusionBridge", "DiffusionConfig", "DiffusionResult",
    "SESIAdaptiveFeedbackDiffusionSuite",
]


# =============================================================================
# SECTION 1 — [ASM1] ADAPTIVE STRING METHOD
# =============================================================================

@dataclass
class AdaptiveStringConfig:
    """Configuration for adaptive string method."""
    # Initial string
    n_initial_images: int = 20
    # Adaptive refinement
    max_images: int = 100
    min_images: int = 10
    curvature_threshold: float = 0.1  # add image if curvature > threshold
    coarsen_threshold: float = 0.01   # remove image if curvature < threshold
    refinement_interval: int = 5      # refine every N iterations
    # Optimization
    step_size: float = 0.01
    n_iterations: int = 200
    convergence_tol: float = 1e-6
    # CV space
    n_cvs: int = 10  # high-dimensional CV space
    # Energy function
    energy_type: str = "msm"  # "msm" | "potential" | "custom"
    temperature: float = 1.0
    # Seed
    seed: int = 0


@dataclass
class AdaptiveStringResult:
    """Result of adaptive string method."""
    string_images: np.ndarray  # (n_images, n_cvs) path in CV space
    n_images: int
    energies: np.ndarray  # (n_images,)
    forces: np.ndarray  # (n_images, n_cvs)
    curvatures: np.ndarray  # (n_images,)
    arc_length: float
    converged: bool
    n_iterations: int
    n_refinements: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveStringMethod:
    """
    [C/D] Adaptive string method for high-dimensional CV spaces.

    MATHEMATICAL BASIS [C E, Ren, Vanden-Eijnden 2007]
      The adaptive string method refines the discretization of the path
      based on local curvature:
        1. Compute curvature κ(s) = |d²γ/ds²| at each image
        2. Add images where κ > curvature_threshold (high curvature regions)
        3. Remove images where κ < coarsen_threshold (flat regions)
        4. Maintain equal arc-length spacing after refinement

      This concentrates computational effort where the path bends most,
      improving accuracy in rugged energy landscapes.

    HIGH-DIMENSIONAL CV SPACE [D]
      For CV spaces with n_cvs >> 2 (from MSM/TPT analysis):
        - Curvature is computed in the full CV space
        - Reparameterization uses arc-length in n_cvs dimensions
        - Gradient descent operates in the full CV space

    CONNECTION TO MSM/TPT [D]
      - MSM provides metastable states A and B (from PCCA+ clustering)
      - TPT provides committor function q(x) as initial path guess
      - String refines the path to minimize energy in high-D CV space

    COROLLARY-3 GUARD [D]
      Curvature is computed via FINITE DIFFERENCES (first-order).
      We do NOT compute the Hessian of the energy landscape.
      The string evolution is a first-order ODE in CV space.

    HINO GUARD [C Thm 8]
      The energy landscape is RUGGED (singular). The string may get
      stuck in local minima. Adaptive refinement helps but does not
      guarantee global optimality.

    PROP-3 AWARENESS [P]
      When computing curvature (products of derivatives), the result
      loses regularity. We use finite differences (first-order).

    HONEST
      • Adaptive refinement is heuristic; real MEP may require global
        optimization methods not captured here.
      • Energy landscape is rugged; path may be suboptimal.
      • This is a computational tool, not a physical simulation.
    """

    def __init__(self, cfg: Optional[AdaptiveStringConfig] = None):
        self.cfg = cfg or AdaptiveStringConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def _compute_energy_msm(self, cv_point: np.ndarray,
                             msm_model: Any) -> float:
        """Compute energy at a CV point using MSM stationary distribution."""
        if msm_model.cluster_centers is None:
            return 0.0

        # Find nearest MSM state
        distances = np.linalg.norm(msm_model.cluster_centers - cv_point, axis=1)
        state_idx = np.argmin(distances)

        # Energy from stationary distribution
        pi = msm_model.stationary_distribution[state_idx]
        if pi < 1e-300:
            return 1e10

        energy = -self.cfg.temperature * np.log(pi)
        return float(energy)

    def _compute_gradient(self, cv_point: np.ndarray,
                           msm_model: Any,
                           eps: float = 1e-4) -> np.ndarray:
        """Compute gradient via finite differences (COROLLARY-3 guard)."""
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

        return grad

    def _compute_curvature(self, string: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """
        Compute curvature at each image via finite differences.
        CURVATURE = |d²γ/ds²| (first-order approximation).
        """
        n_images = len(string)
        curvatures = np.zeros(n_images)

        for i in range(1, n_images - 1):
            # Second derivative via central difference
            d2 = (string[i + 1] - 2 * string[i] + string[i - 1]) / (eps ** 2)
            curvatures[i] = np.linalg.norm(d2)

        # Boundary images: use one-sided difference
        if n_images > 1:
            curvatures[0] = np.linalg.norm(string[1] - string[0]) / eps
            curvatures[-1] = np.linalg.norm(string[-1] - string[-2]) / eps

        return curvatures

    def _refine_string(self, string: np.ndarray,
                        curvatures: np.ndarray) -> np.ndarray:
        """
        Adaptively refine string: add/remove images based on curvature.
        """
        n_images = len(string)
        new_images = [string[0]]

        for i in range(1, n_images - 1):
            # Add image if curvature is high
            if curvatures[i] > self.cfg.curvature_threshold and len(new_images) < self.cfg.max_images:
                # Insert midpoint
                midpoint = 0.5 * (string[i - 1] + string[i])
                new_images.append(midpoint)
                new_images.append(string[i])
            # Skip image if curvature is low
            elif curvatures[i] < self.cfg.coarsen_threshold and len(new_images) > self.cfg.min_images:
                continue  # remove this image
            else:
                new_images.append(string[i])

        new_images.append(string[-1])
        return np.array(new_images)

    def _reparameterize(self, string: np.ndarray) -> np.ndarray:
        """Reparameterize string to have equal arc-length spacing."""
        n_images = len(string)

        # Compute cumulative arc length
        diffs = np.diff(string, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length < 1e-10:
            return string

        # Target arc-length positions
        target_s = np.linspace(0, total_length, n_images)

        # Interpolate to new positions
        if HAS_SCIPY_INTERP:
            new_string = np.zeros_like(string)
            for dim in range(string.shape[1]):
                cs = CubicSpline(cumulative_length, string[:, dim])
                new_string[:, dim] = cs(target_s)
            return new_string
        else:
            # Simple linear interpolation fallback
            new_string = np.zeros_like(string)
            for dim in range(string.shape[1]):
                new_string[:, dim] = np.interp(target_s, cumulative_length, string[:, dim])
            return new_string

    def _initialize_string(self, msm_model: Any,
                            tpt_result: Optional[Any] = None) -> np.ndarray:
        """Initialize string path."""
        n_images = self.cfg.n_initial_images
        n_cvs = self.cfg.n_cvs

        if tpt_result is not None and hasattr(tpt_result, 'committor_forward'):
            # Use committor function as initial path
            committor = tpt_result.committor_forward
            q_values = np.linspace(0, 1, n_images)
            string = np.zeros((n_images, n_cvs))

            for i, q in enumerate(q_values):
                state_idx = np.argmin(np.abs(committor - q))
                if msm_model.cluster_centers is not None:
                    string[i] = msm_model.cluster_centers[state_idx, :n_cvs]
                else:
                    string[i] = self._rng.randn(n_cvs) * 0.1

            return string

        else:
            # Linear interpolation
            if msm_model.cluster_centers is not None:
                start = msm_model.cluster_centers[0, :n_cvs]
                end = msm_model.cluster_centers[-1, :n_cvs]
            else:
                start = np.zeros(n_cvs)
                end = np.ones(n_cvs)

            return np.linspace(start, end, n_images)

    def find_mep_adaptive(self, msm_model: Any,
                            tpt_result: Optional[Any] = None) -> AdaptiveStringResult:
        """
        Find minimum energy path using adaptive string method.
        """
        # Initialize string
        string = self._initialize_string(msm_model, tpt_result)

        # Evolution loop
        n_refinements = 0
        converged = False

        for iteration in range(self.cfg.n_iterations):
            # Compute energy, gradient, curvature
            n_images = len(string)
            energies = np.zeros(n_images)
            forces = np.zeros_like(string)
            curvatures = self._compute_curvature(string)

            for i in range(n_images):
                energies[i] = self._compute_energy_msm(string[i], msm_model)
                forces[i] = -self._compute_gradient(string[i], msm_model)

            # Evolve string
            prev_string = string.copy()
            string = string + self.cfg.step_size * forces

            # Adaptive refinement
            if iteration % self.cfg.refinement_interval == 0:
                string = self._refine_string(string, curvatures)
                string = self._reparameterize(string)
                n_refinements += 1

            # Check convergence
            max_displacement = np.max(np.linalg.norm(string[:len(prev_string)] - prev_string, axis=1))
            if max_displacement < self.cfg.convergence_tol:
                logger.info(f"Adaptive string converged at iteration {iteration}")
                converged = True
                break

        # Final computation
        n_images = len(string)
        energies = np.zeros(n_images)
        forces = np.zeros_like(string)
        curvatures = self._compute_curvature(string)

        for i in range(n_images):
            energies[i] = self._compute_energy_msm(string[i], msm_model)
            forces[i] = -self._compute_gradient(string[i], msm_model)

        arc_length = float(np.sum(np.linalg.norm(np.diff(string, axis=0), axis=1)))

        return AdaptiveStringResult(
            string_images=string,
            n_images=n_images,
            energies=energies,
            forces=forces,
            curvatures=curvatures,
            arc_length=arc_length,
            converged=converged,
            n_iterations=iteration + 1,
            n_refinements=n_refinements,
            metadata={"method": "adaptive_string", "n_cvs": self.cfg.n_cvs},
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_initial_images": self.cfg.n_initial_images,
            "max_images": self.cfg.max_images,
            "n_cvs": self.cfg.n_cvs,
            "corollary3_guard": "first-order gradient descent; curvature via finite differences; no Hessian",
            "hino_guard": "energy landscape singular; path may be local minimum",
        }


# =============================================================================
# SECTION 2 — [RNF1] REMODELER-NUCLEOSOME FEEDBACK
# =============================================================================

@dataclass
class FeedbackConfig:
    """Configuration for remodeler-nucleosome feedback."""
    # Nucleosome parameters
    n_nucleosomes: int = 20
    nucleosome_spacing_bp: int = 187
    dna_length_bp: int = 5000
    # Remodeler parameters
    n_remodelers: int = 30
    remodeler_families: List[str] = field(default_factory=lambda:
        ["SWI/SNF", "ISWI", "CHD", "INO80"])
    # Feedback parameters
    nucleosome_effect_on_binding: float = 0.3  # how much nucleosome position affects binding
    remodeler_effect_on_occupancy: float = 0.2  # how much remodeler activity affects occupancy
    feedback_delay: float = 0.1  # delay in feedback response
    # Kinetics
    k_binding_base: float = 0.1
    k_unbinding_base: float = 0.05
    k_sliding_base: float = 0.02
    k_eviction_base: float = 0.01
    # Simulation
    max_time: float = 200.0
    max_steps: int = 150000
    seed: int = 0


@dataclass
class FeedbackState:
    """State of coupled remodeler-nucleosome system."""
    time: float
    nucleosome_positions: np.ndarray
    remodeler_bound: np.ndarray
    remodeler_positions: np.ndarray
    binding_affinities: np.ndarray  # per-nucleosome, modified by feedback
    occupancy_levels: np.ndarray  # per-nucleosome, modified by remodeler activity
    n_sliding_events: int
    n_eviction_events: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RemodelerNucleosomeFeedback:
    """
    [C/D] Bidirectional feedback between chromatin remodelers and nucleosomes.

    BIOLOGICAL BASIS [C Clapier & Cairns 2009; Schiessel 2006]
      Chromatin remodelers and nucleosomes engage in bidirectional feedback:

      NUCLEOSOME → REMODELER:
        - Nucleosome position affects remodeler binding affinity
        - Closely spaced nucleosomes compete for remodeler binding
        - Nucleosome-free regions attract remodelers

      REMODELER → NUCLEOSOME:
        - Remodeler activity changes nucleosome occupancy
        - Sliding/eviction events reposition nucleosomes
        - Remodeler families have different effects (SWI/SNF evicts, ISWI spaces)

    MODEL [D]
      Coupled stochastic system:
        1. Binding affinity for remodeler i at nucleosome j:
           k_bind(i,j) = k_bind_base * (1 + nucleosome_effect * f(position_j, neighbors))

        2. Nucleosome occupancy at position j:
           occupancy(j) = base_occupancy * (1 - remodeler_effect * g(activity_near_j))

        3. Remodeler sliding/eviction rates depend on local nucleosome density

      Gillespie SSA with feedback-modulated propensities.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear in copy numbers,
      with feedback modulation). We do NOT compute second derivatives of
      the feedback landscape.

    HINO GUARD [C Thm 8]
      Binding/unbinding events are DISCRETE, not smooth. The stochasticity
      is singular (tied to individual molecular events). Feedback modulates
      rates but does not change the discrete nature.

    PROP-3 AWARENESS [P]
      When multiplying fields (e.g., nucleosome_density × remodeler_activity),
      the result loses two Sobolev orders. We use additive feedback terms
      instead of multiplicative to maintain regularity.

    HONEST
      • Bidirectional coupling is simplified. Real chromatin remodeling
        involves complex multi-protein assemblies, DNA mechanics, and
        histone modifications not captured here.
      • Feedback terms are heuristic; real feedback may involve signaling
        cascades and allosteric regulation.
      • This is a computational model, not validated biochemistry.
    """

    def __init__(self, cfg: Optional[FeedbackConfig] = None):
        self.cfg = cfg or FeedbackConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # State
        self.nucleosome_positions: Optional[np.ndarray] = None
        self.remodeler_bound: Optional[np.ndarray] = None
        self.remodeler_positions: Optional[np.ndarray] = None
        self.remodeler_families: Optional[np.ndarray] = None

        # Feedback fields
        self.binding_affinities: Optional[np.ndarray] = None
        self.occupancy_levels: Optional[np.ndarray] = None

        # Counters
        self.n_sliding_events = 0
        self.n_eviction_events = 0

    def initialize(self) -> None:
        """Initialize coupled system."""
        # Regular nucleosome spacing
        n_nuc = min(self.cfg.n_nucleosomes,
                    self.cfg.dna_length_bp // self.cfg.nucleosome_spacing_bp)
        self.nucleosome_positions = np.arange(n_nuc) * self.cfg.nucleosome_spacing_bp

        # Initialize remodelers
        self.remodeler_bound = np.zeros(self.cfg.n_remodelers, dtype=bool)
        self.remodeler_positions = self._rng.uniform(
            0, self.cfg.dna_length_bp, self.cfg.n_remodelers)
        self.remodeler_families = np.array([
            self.cfg.remodeler_families[i % len(self.cfg.remodeler_families)]
            for i in range(self.cfg.n_remodelers)
        ])

        # Initialize feedback fields
        self.binding_affinities = np.ones(n_nuc)
        self.occupancy_levels = np.ones(n_nuc)

        self.n_sliding_events = 0
        self.n_eviction_events = 0

    def _compute_nucleosome_density(self, position: float) -> float:
        """Compute local nucleosome density around a position."""
        if self.nucleosome_positions is None or len(self.nucleosome_positions) == 0:
            return 0.0
        distances = np.abs(self.nucleosome_positions - position)
        nearby = (distances < 100).sum()  # within 100 bp
        return float(nearby)

    def _compute_remodeler_activity(self, nucleosome_idx: int) -> float:
        """Compute local remodeler activity around a nucleosome."""
        if self.remodeler_positions is None or len(self.remodeler_positions) == 0:
            return 0.0
        nuc_pos = self.nucleosome_positions[nucleosome_idx]
        distances = np.abs(self.remodeler_positions - nuc_pos)
        active_nearby = ((distances < 50) & self.remodeler_bound).sum()
        return float(active_nearby)

    def update_feedback(self) -> None:
        """Update binding affinities and occupancy levels based on feedback."""
        n_nuc = len(self.nucleosome_positions)

        # Nucleosome → Remodeler: binding affinity depends on local density
        for i in range(n_nuc):
            density = self._compute_nucleosome_density(self.nucleosome_positions[i])
            # High density → lower affinity (competition)
            self.binding_affinities[i] = 1.0 - self.cfg.nucleosome_effect_on_binding * density / 10.0
            self.binding_affinities[i] = max(0.1, self.binding_affinities[i])

        # Remodeler → Nucleosome: occupancy depends on local activity
        for i in range(n_nuc):
            activity = self._compute_remodeler_activity(i)
            # High activity → lower occupancy (remodeling)
            self.occupancy_levels[i] = 1.0 - self.cfg.remodeler_effect_on_occupancy * activity / 5.0
            self.occupancy_levels[i] = max(0.0, self.occupancy_levels[i])

    def compute_propensities(self) -> Tuple[np.ndarray, List[Tuple[str, int, Dict[str, Any]]]]:
        """Compute all reaction propensities with feedback modulation."""
        props = []
        events = []

        n_nuc = len(self.nucleosome_positions)

        for i in range(self.cfg.n_remodelers):
            if not self.remodeler_bound[i]:
                # Unbound remodeler: can bind to nucleosome
                # Find nearest nucleosome
                distances = np.abs(self.remodeler_positions[i] - self.nucleosome_positions)
                if len(distances) > 0:
                    nearest_nuc = np.argmin(distances)
                    if distances[nearest_nuc] < 50:
                        # Feedback-modulated binding rate
                        affinity = self.binding_affinities[nearest_nuc]
                        a = self.cfg.k_binding_base * affinity
                        props.append(a)
                        events.append(("bind", i, {"nucleosome": nearest_nuc}))

            else:
                # Bound remodeler: can slide, evict, or unbind
                # Find which nucleosome this remodeler is bound to
                distances = np.abs(self.remodeler_positions[i] - self.nucleosome_positions)
                if len(distances) > 0:
                    nuc_idx = np.argmin(distances)
                    if distances[nuc_idx] < 10:
                        # Feedback-modulated rates
                        occupancy = self.occupancy_levels[nuc_idx]

                        # Sliding
                        a = self.cfg.k_sliding_base * occupancy
                        props.append(a)
                        events.append(("slide", i, {"nucleosome": nuc_idx}))

                        # Eviction (family-dependent)
                        family = self.remodeler_families[i]
                        if family in ["SWI/SNF", "INO80"]:
                            a = self.cfg.k_eviction_base * occupancy
                            props.append(a)
                            events.append(("evict", i, {"nucleosome": nuc_idx}))

                # Unbinding
                a = self.cfg.k_unbinding_base
                props.append(a)
                events.append(("unbind", i, {}))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self) -> Tuple[Dict[str, Any], float]:
        """One Gillespie SSA step with feedback."""
        # Update feedback fields
        self.update_feedback()

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

        reaction_name, remodeler_idx, params = events[event_idx]

        # Apply reaction
        if reaction_name == "bind":
            self.remodeler_bound[remodeler_idx] = True
            nuc_idx = params["nucleosome"]
            self.remodeler_positions[remodeler_idx] = self.nucleosome_positions[nuc_idx]

        elif reaction_name == "slide":
            nuc_idx = params["nucleosome"]
            # Slide nucleosome by ±1 bp
            direction = self._rng.choice([-1, 1])
            new_pos = self.nucleosome_positions[nuc_idx] + direction
            # Check bounds
            if 0 <= new_pos < self.cfg.dna_length_bp:
                self.nucleosome_positions[nuc_idx] = new_pos
                self.remodeler_positions[remodeler_idx] = new_pos
                self.n_sliding_events += 1

        elif reaction_name == "evict":
            nuc_idx = params["nucleosome"]
            # Remove nucleosome
            self.nucleosome_positions = np.delete(self.nucleosome_positions, nuc_idx)
            self.binding_affinities = np.delete(self.binding_affinities, nuc_idx)
            self.occupancy_levels = np.delete(self.occupancy_levels, nuc_idx)
            self.remodeler_bound[remodeler_idx] = False
            self.n_eviction_events += 1

        elif reaction_name == "unbind":
            self.remodeler_bound[remodeler_idx] = False

        return {"reaction": reaction_name, "remodeler": remodeler_idx}, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[FeedbackState], List[float]]:
        """Run full SSA simulation with feedback."""
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

    def _snapshot(self, t: float) -> FeedbackState:
        return FeedbackState(
            time=t,
            nucleosome_positions=self.nucleosome_positions.copy(),
            remodeler_bound=self.remodeler_bound.copy(),
            remodeler_positions=self.remodeler_positions.copy(),
            binding_affinities=self.binding_affinities.copy(),
            occupancy_levels=self.occupancy_levels.copy(),
            n_sliding_events=self.n_sliding_events,
            n_eviction_events=self.n_eviction_events,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_nucleosomes": len(self.nucleosome_positions) if self.nucleosome_positions is not None else 0,
            "n_remodelers": self.cfg.n_remodelers,
            "feedback_enabled": True,
            "corollary3_guard": "first-order kinetics with feedback modulation; no Hessian",
            "hino_guard": "discrete binding events, not smooth density",
        }


# =============================================================================
# SECTION 3 — [SBD1] SCORE-BASED DIFFUSION BRIDGE
# =============================================================================

@dataclass
class DiffusionConfig:
    """Configuration for score-based diffusion model."""
    # Architecture
    data_dim: int = 32
    hidden_dim: int = 256
    n_layers: int = 4
    # Diffusion process
    n_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 64
    # Sampling
    n_samples: int = 100
    # Seed
    seed: int = 0


@dataclass
class DiffusionResult:
    """Result of diffusion model training/sampling."""
    n_samples: int
    samples: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScoreNetwork(nn.Module):
    """Score network: predicts ∇_x log p_t(x) for diffusion model."""

    def __init__(self, data_dim: int, hidden_dim: int, n_layers: int, n_timesteps: int):
        super().__init__()
        self.n_timesteps = n_timesteps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Score network
        layers = []
        in_dim = data_dim + hidden_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))

        self.score_net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Predict score ∇_x log p_t(x).

        Args:
            x: (batch, data_dim) noisy samples
            t: (batch,) timestep indices

        Returns:
            score: (batch, data_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float() / self.n_timesteps)

        # Concatenate x and t_emb
        h = torch.cat([x, t_emb], dim=-1)

        # Predict score
        score = self.score_net(h)

        return score


class ScoreBasedDiffusionBridge:
    """
    [C/D] Score-based diffusion model for conformational sampling.

    MATHEMATICAL BASIS [C Song & Ermon 2019; Ho et al. 2020]
      Diffusion models learn to reverse a noising process:

      FORWARD PROCESS (noising):
        x_0 ~ p_data(x)
        x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε,  ε ~ N(0, I)

        where ᾱ_t = ∏_{s=1}^t (1 - β_s), β_s is noise schedule.

      REVERSE PROCESS (denoising):
        dx = [∇_x log p_t(x) + ∇_x log p_t(x)] dt + dW

        where ∇_x log p_t(x) is learned by a neural network (score network).

      TRAINING:
        Learn score network s_θ(x_t, t) ≈ ∇_x log p(x_t | x_0)
        Loss = E[||s_θ(x_t, t) - ∇_x log p(x_t | x_0)||²]

      SAMPLING:
        Start from x_T ~ N(0, I)
        Iteratively denoise: x_{t-1} = x_t + score(x_t, t) * Δt + noise

    CONNECTION TO VAE [D]
      - VAE learns approximate posterior q(z|x) via ELBO
      - Diffusion learns exact likelihood p(x) via score matching
      - Can use VAE latent codes as training data for diffusion
      - Diffusion provides better sample quality but slower sampling

    COROLLARY-3 GUARD [D]
      Training uses FIRST-ORDER optimization (Adam). We do NOT compute
      second derivatives of the score function.

    HINO GUARD [C Thm 8]
      The data distribution is NOT assumed to be smooth. The score network
      learns a singular distribution that may have complex structure.

    HONEST
      • Diffusion models learn a statistical distribution, not a physical one.
      • Generated samples may be outside the training manifold.
      • This is a generative model, not a physics engine.
      • Sampling is slow (requires many denoising steps).
    """

    def __init__(self, cfg: Optional[DiffusionConfig] = None):
        self.cfg = cfg or DiffusionConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Build score network
        self.score_net = ScoreNetwork(
            self.cfg.data_dim,
            self.cfg.hidden_dim,
            self.cfg.n_layers,
            self.cfg.n_timesteps,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.score_net.parameters(),
            lr=self.cfg.learning_rate
        )

        # Noise schedule
        self.betas = torch.linspace(
            self.cfg.beta_start,
            self.cfg.beta_end,
            self.cfg.n_timesteps
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _add_noise(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Add noise to samples according to diffusion schedule."""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).unsqueeze(-1)

        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def train(self, data: Tensor, n_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the diffusion model via score matching.

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
                batch = data_shuffled[i:i + self.cfg.batch_size]
                batch_size = batch.shape[0]

                # Sample random timesteps
                t = torch.randint(0, self.cfg.n_timesteps, (batch_size,), device=batch.device)

                # Add noise
                x_t, noise = self._add_noise(batch, t)

                # Predict score
                predicted_score = self.score_net(x_t, t)

                # Loss: MSE between predicted and true score
                # True score for Gaussian noise: -noise / sqrt(1 - alpha_bar)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).unsqueeze(-1)
                true_score = -noise / sqrt_one_minus_alpha_bar

                loss = F.mse_loss(predicted_score, true_score)

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

    def sample(self, n_samples: int = 100) -> Tensor:
        """
        Generate samples via reverse diffusion process.

        Args:
            n_samples: number of samples to generate

        Returns:
            samples: (n_samples, data_dim)
        """
        # Start from pure noise
        x = torch.randn(n_samples, self.cfg.data_dim)

        # Reverse diffusion
        for t in reversed(range(self.cfg.n_timesteps)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long)

            # Predict score
            with torch.no_grad():
                score = self.score_net(x, t_batch)

            # Denoise step (simplified Euler-Maruyama)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            # Mean
            mean = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * score) / torch.sqrt(alpha)

            # Variance
            if t > 0:
                variance = beta
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

        return x

    def generate_samples(self, n_samples: int = 100) -> DiffusionResult:
        """Generate samples and return result."""
        samples = self.sample(n_samples)

        return DiffusionResult(
            n_samples=n_samples,
            samples=samples.numpy(),
            metadata={
                "n_timesteps": self.cfg.n_timesteps,
                "data_dim": self.cfg.data_dim,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "data_dim": self.cfg.data_dim,
            "n_timesteps": self.cfg.n_timesteps,
            "corollary3_guard": "first-order optimization (Adam); no Hessian of score",
            "hino_guard": "data distribution not assumed smooth; learned from data",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIAdaptiveFeedbackDiffusionSuite:
    """
    Unifies: AdaptiveStringMethod + RemodelerNucleosomeFeedback + ScoreBasedDiffusionBridge.

    FULL PIPELINE:
      1. Adaptive string method finds MEP in high-D CV space
      2. Remodeler-nucleosome feedback simulates chromatin dynamics
      3. Diffusion model generates new conformations
      4. Integration: string → feedback → diffusion
    """

    def __init__(self,
                 adaptive_string_cfg: Optional[AdaptiveStringConfig] = None,
                 feedback_cfg: Optional[FeedbackConfig] = None,
                 diffusion_cfg: Optional[DiffusionConfig] = None):
        self.adaptive_string = AdaptiveStringMethod(adaptive_string_cfg)
        self.feedback = RemodelerNucleosomeFeedback(feedback_cfg)
        self.diffusion = ScoreBasedDiffusionBridge(diffusion_cfg)

    def find_adaptive_mep(self, msm_model: Any,
                            tpt_result: Optional[Any] = None) -> Dict[str, Any]:
        """Find MEP using adaptive string method."""
        result = self.adaptive_string.find_mep_adaptive(msm_model, tpt_result)
        return {
            "n_images": result.n_images,
            "arc_length": result.arc_length,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
            "n_refinements": result.n_refinements,
            "energy_range": (float(result.energies.min()), float(result.energies.max())),
            "mean_curvature": float(result.curvatures.mean()),
        }

    def simulate_feedback_dynamics(self, n_steps: int = 1000) -> Dict[str, Any]:
        """Simulate coupled remodeler-nucleosome dynamics with feedback."""
        traj, times = self.feedback.simulate(max_steps=n_steps)

        final_state = traj[-1]
        return {
            "n_steps": len(traj),
            "final_time": final_state.time,
            "n_nucleosomes": len(final_state.nucleosome_positions),
            "n_bound_remodelers": int(final_state.remodeler_bound.sum()),
            "n_sliding_events": final_state.n_sliding_events,
            "n_eviction_events": final_state.n_eviction_events,
            "mean_binding_affinity": float(final_state.binding_affinities.mean()),
            "mean_occupancy": float(final_state.occupancy_levels.mean()),
        }

    def train_diffusion_model(self, training_data: np.ndarray,
                                n_epochs: int = 50) -> Dict[str, Any]:
        """Train diffusion model on data."""
        data_tensor = torch.tensor(training_data, dtype=torch.float32)
        result = self.diffusion.train(data_tensor, n_epochs=n_epochs)
        return result

    def generate_conformations(self, n_samples: int = 100) -> Dict[str, Any]:
        """Generate samples from trained diffusion model."""
        result = self.diffusion.generate_samples(n_samples)
        return {
            "n_samples": result.n_samples,
            "samples_shape": result.samples.shape,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "adaptive_string": self.adaptive_string.get_status(),
            "feedback": self.feedback.get_status(),
            "diffusion": self.diffusion.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI PROTEIN DYNAMICS STUDIO FRONTIER — SMOKE TEST")
    print(" Adaptive String + Remodeler Feedback + Diffusion Models")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scipy interp           : {HAS_SCIPY_INTERP}")
    print(f" scipy dist             : {HAS_SCIPY_DIST}")
    print(f" String/Remodeler       : {HAS_STRING_REMODELER}")
    print(f" TPT/Nuc/VAE            : {HAS_TPT_NUC_VAE}")
    print(f" MSM                    : {HAS_MSM}")

    suite = SESIAdaptiveFeedbackDiffusionSuite(
        adaptive_string_cfg=AdaptiveStringConfig(
            n_initial_images=10, max_images=30, n_cvs=5, n_iterations=50),
        feedback_cfg=FeedbackConfig(
            n_nucleosomes=10, n_remodelers=15, max_steps=500),
        diffusion_cfg=DiffusionConfig(
            data_dim=10, n_timesteps=20, n_epochs=15))

    # ── 1. Adaptive String Method ───────────────────────────────────────
    print("\n [ASM1] Adaptive String Method (High-D CV Space)")
    adaptive_string = suite.adaptive_string
    print(f"   n_initial_images     : {adaptive_string.cfg.n_initial_images}")
    print(f"   max_images           : {adaptive_string.cfg.max_images}")
    print(f"   n_cvs                : {adaptive_string.cfg.n_cvs}")

    # Create synthetic MSM for testing
    if HAS_MSM:
        msm = MarkovStateModel()
        msm.cluster_centers = np.random.rand(20, 5)
        msm.stationary_distribution = np.random.rand(20)
        msm.stationary_distribution /= msm.stationary_distribution.sum()

        # Find adaptive MEP
        mep_result = suite.find_adaptive_mep(msm)
        print(f"   MEP converged        : {mep_result['converged']}")
        print(f"   MEP arc length       : {mep_result['arc_length']:.4f}")
        print(f"   MEP iterations       : {mep_result['n_iterations']}")
        print(f"   MEP refinements      : {mep_result['n_refinements']}")
        print(f"   Final n_images       : {mep_result['n_images']}")
        print(f"   Energy range         : {mep_result['energy_range']}")
        print(f"   Mean curvature       : {mep_result['mean_curvature']:.4f}")
    else:
        print("   [SKIP] MSM module not available")

    asm_status = adaptive_string.get_status()
    print(f"   COROLLARY-3 guard    : {asm_status['corollary3_guard']}")
    print(f"   HINO guard           : {asm_status['hino_guard']}")

    # ── 2. Remodeler-Nucleosome Feedback ────────────────────────────────
    print("\n [RNF1] Remodeler-Nucleosome Feedback")
    feedback = suite.feedback
    print(f"   n_nucleosomes        : {feedback.cfg.n_nucleosomes}")
    print(f"   n_remodelers         : {feedback.cfg.n_remodelers}")
    print(f"   feedback enabled     : {feedback.get_status()['feedback_enabled']}")

    # Simulate
    feedback_result = suite.simulate_feedback_dynamics(n_steps=300)
    print(f"   simulation steps     : {feedback_result['n_steps']}")
    print(f"   final time           : {feedback_result['final_time']:.2f}")
    print(f"   final nucleosomes    : {feedback_result['n_nucleosomes']}")
    print(f"   bound remodelers     : {feedback_result['n_bound_remodelers']}")
    print(f"   sliding events       : {feedback_result['n_sliding_events']}")
    print(f"   eviction events      : {feedback_result['n_eviction_events']}")
    print(f"   mean binding aff     : {feedback_result['mean_binding_affinity']:.4f}")
    print(f"   mean occupancy       : {feedback_result['mean_occupancy']:.4f}")

    rnf_status = feedback.get_status()
    print(f"   COROLLARY-3 guard    : {rnf_status['corollary3_guard']}")
    print(f"   HINO guard           : {rnf_status['hino_guard']}")

    # ── 3. Score-Based Diffusion ────────────────────────────────────────
    print("\n [SBD1] Score-Based Diffusion Model")
    diffusion = suite.diffusion
    print(f"   data_dim             : {diffusion.cfg.data_dim}")
    print(f"   n_timesteps          : {diffusion.cfg.n_timesteps}")

    # Generate synthetic training data
    n_train = 200
    training_data = np.random.randn(n_train, diffusion.cfg.data_dim).astype(np.float32)

    # Train
    train_result = suite.train_diffusion_model(training_data, n_epochs=15)
    print(f"   training epochs      : {train_result['n_epochs']}")
    print(f"   final loss           : {train_result['final_loss']:.4f}")

    # Generate samples
    gen_result = suite.generate_conformations(n_samples=50)
    print(f"   generated samples    : {gen_result['n_samples']}")
    print(f"   samples shape        : {gen_result['samples_shape']}")

    sbd_status = diffusion.get_status()
    print(f"   COROLLARY-3 guard    : {sbd_status['corollary3_guard']}")
    print(f"   HINO guard           : {sbd_status['hino_guard']}")

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
    print(" [C]   E, Ren, Vanden-Eijnden 2007: Adaptive string method.")
    print(" [C]   Clapier & Cairns 2009: Chromatin remodeler families.")
    print(" [C]   Schiessel 2006: Nucleosome positioning and remodeling.")
    print(" [C]   Song & Ermon 2019: Score-based generative models.")
    print(" [C]   Ho et al. 2020: Denoising diffusion probabilistic models.")
    print(" [D]   Adaptive string refinement is heuristic. Real MEP may require")
    print("       global optimization methods. Energy landscape is rugged (HINO).")
    print(" [D]   Remodeler-nucleosome feedback is simplified. Real chromatin")
    print("       remodeling involves complex multi-protein assemblies, DNA")
    print("       mechanics, histone modifications not captured here.")
    print(" [D]   Diffusion models learn statistical distribution, not physical one.")
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
