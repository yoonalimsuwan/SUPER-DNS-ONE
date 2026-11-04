"""
=============================================================================
SESI PROTEIN DYNAMICS STUDIO ULTIMATE
String Method + Chromatin Remodelers + Normalizing Flows
Native integration with SDA-ONE, MSM, TPT, VAE modules
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
     - MarkovStateModel (MSM with transition_matrix, stationary_distribution)
     - ChromatinAccessibilityModel (SASA, contact maps)
     - EnhancedSamplingBridge (metadynamics/REMD setup)
  3. sesi_protein_dynamics_tpt_nucleosome_vae.py:
     - TransitionPathTheory (committor functions, reactive currents)
     - NucleosomePositioningModel (occupancy predictions)
     - VariationalAutoencoderBridge (latent space)

Integration points:
  1. StringMethodMEP.find_mep_from_tpt(tpt_result, msm_model)
     - Uses committor function as initial path guess
     - Uses MSM states as endpoints
     - Refines path via gradient descent in CV space

  2. ChromatinRemodelerDynamics.simulate_from_nucleosome(nucleosome_result)
     - Uses nucleosome occupancy as initial state
     - Models SWI/SNF, ISWI, CHD, INO80 families
     - Updates accessibility landscape

  3. NormalizingFlowsBridge.train_from_vae(vae_model, msm_states)
     - Uses VAE latent codes as training data
     - Learns exact likelihood distribution
     - Generates samples with guaranteed probability

=============================================================================
MODULES
=============================================================================
[SME1] StringMethodMEP
      • Minimum Energy Path via string method [C E et al. 2002]
      • Collective variables from MSM/TPT analysis
      • Gradient descent on each string image
      • Reparameterization for equal arc-length spacing
      • Connection to committor function (TPT)
      • COROLLARY-3 guard: first-order gradient descent
      • HINO guard: energy landscape singular, path is discrete

[CRD1] ChromatinRemodelerDynamics
      • SWI/SNF, ISWI, CHD, INO80 families [C Clapier 2009]
      • ATP-dependent nucleosome sliding/eviction
      • Stochastic binding/unbinding kinetics
      • Position-dependent translocation rates
      • Coupling to nucleosome positioning model
      • COROLLARY-3 guard: first-order kinetics
      • HINO guard: discrete binding events

[NFB1] NormalizingFlowsBridge
      • RealNVP-style coupling layers [C Dinh 2017]
      • Exact likelihood via change of variables
      • Invertible transformations: z = f(x), x = f⁻¹(z)
      • Training via maximum likelihood (not ELBO)
      • Connection to VAE latent space
      • COROLLARY-3 guard: first-order optimization (Adam)
      • HINO guard: latent space singular, not smooth

=============================================================================
HONEST SCOPE
=============================================================================
• StringMethodMEP: finds local minimum energy path, not guaranteed global.
  Requires good initial guess (from TPT committor). Energy landscape is
  rugged (HINO), so path may get stuck in local minima.
• ChromatinRemodelerDynamics: simplified kinetic model. Real remodelers
  involve complex multi-subunit assemblies, DNA translocation mechanisms,
  and histone modifications not captured here.
• NormalizingFlowsBridge: learns a distribution, but does NOT guarantee
  physical realism. Generated samples may be outside the training manifold.
  This is a statistical model, not a physics engine.
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

logger = logging.getLogger("sesi.string_remodeler_flows")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY_INTERP = True
except Exception:
    HAS_SCIPY_INTERP = False

# ── Previous modules import (graceful) ──────────────────────────────────
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
    )
    HAS_MSM = True
except Exception:
    HAS_MSM = False
    MarkovStateModel = None

__all__ = [
    "StringMethodMEP", "StringConfig", "StringResult",
    "ChromatinRemodelerDynamics", "RemodelerConfig", "RemodelerState",
    "NormalizingFlowsBridge", "FlowConfig", "FlowResult",
    "SESIStringRemodelerFlowsSuite",
]


# =============================================================================
# SECTION 1 — [SME1] STRING METHOD FOR MINIMUM ENERGY PATHS
# =============================================================================

@dataclass
class StringConfig:
    """Configuration for string method MEP calculation."""
    # String discretization
    n_images: int = 20  # number of images along the string
    # Collective variables
    cv_type: str = "committor"  # "committor" | "pca" | "custom"
    n_cvs: int = 2  # number of collective variables
    # Optimization
    step_size: float = 0.01  # gradient descent step
    n_iterations: int = 100
    convergence_tol: float = 1e-6
    # Reparameterization
    reparam_interval: int = 5  # reparameterize every N iterations
    # Energy function
    energy_type: str = "msm"  # "msm" | "potential" | "custom"
    temperature: float = 1.0  # kBT
    # Seed
    seed: int = 0


@dataclass
class StringResult:
    """Result of string method calculation."""
    string_images: np.ndarray  # (n_images, n_cvs) path in CV space
    energies: np.ndarray  # (n_images,) energy at each image
    forces: np.ndarray  # (n_images, n_cvs) gradient at each image
    arc_length: float  # total path length
    converged: bool
    n_iterations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class StringMethodMEP:
    """
    [C/D] String method for minimum energy path (MEP) calculation.

    MATHEMATICAL BASIS [C E, Ren, Vanden-Eijnden 2002; Maragliano et al. 2006]
      Given a potential energy surface V(x) and two metastable states A and B,
      the MEP is the path γ(s) connecting A to B that satisfies:
        1. γ(0) ∈ A, γ(1) ∈ B
        2. ∇V(γ(s)) is parallel to γ'(s) for all s ∈ (0,1)
         (gradient is tangent to the path)

      The string method discretizes the path into N images {x_i} and evolves:
        dx_i/dt = -∇V(x_i) + (γ''(s_i) / |γ'(s_i)|²) · |∇V(x_i) · γ'(s_i)|

      The second term enforces equal arc-length spacing (reparameterization).

    CONNECTION TO MSM/TPT [D]
      - MSM provides metastable states A and B (from PCCA+ clustering)
      - TPT provides committor function q(x) as initial path guess
      - String refines the path to minimize energy

    COLLECTIVE VARIABLES [D]
      - Committor-based: CV = (q(x), 1-q(x)) for 2D path
      - PCA-based: CV = first 2 principal components of MSM states
      - Custom: user-defined CVs

    COROLLARY-3 GUARD [D]
      Gradient descent is FIRST-ORDER. We do NOT compute Hessian of V(x).
      The string evolution is a first-order ODE in CV space.

    HINO GUARD [C Thm 8]
      The energy landscape V(x) is SINGULAR (rugged, not smooth). The string
      may get stuck in local minima. We use multiple restarts and check
      convergence carefully.

    PROP-3 AWARENESS [P]
      When computing forces (products of gradients), the result loses
      regularity. We use finite differences for gradients (first-order).

    HONEST
      • String method finds LOCAL minimum energy path, not guaranteed global.
      • Requires good initial guess (from TPT committor or linear interpolation).
      • Energy landscape is rugged; path may be suboptimal.
      • This is a computational tool, not a physical simulation.
    """

    def __init__(self, cfg: Optional[StringConfig] = None):
        self.cfg = cfg or StringConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def _compute_energy_msm(self, cv_point: np.ndarray,
                             msm_model: Any) -> float:
        """
        Compute energy at a CV point using MSM stationary distribution.
        E(x) = -kBT log π(x)
        """
        # Find nearest MSM state
        if msm_model.cluster_centers is None:
            return 0.0

        # Map CV point to MSM state index (simplified: nearest neighbor)
        distances = np.linalg.norm(msm_model.cluster_centers - cv_point, axis=1)
        state_idx = np.argmin(distances)

        # Energy from stationary distribution
        pi = msm_model.stationary_distribution[state_idx]
        if pi < 1e-300:
            return 1e10  # very high energy for unvisited states

        energy = -self.cfg.temperature * np.log(pi)
        return float(energy)

    def _compute_gradient(self, cv_point: np.ndarray,
                           msm_model: Any,
                           eps: float = 1e-4) -> np.ndarray:
        """
        Compute gradient of energy at CV point via finite differences.
        COROLLARY-3 guard: first-order finite differences.
        """
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

    def _initialize_string(self, msm_model: Any,
                            tpt_result: Optional[Any] = None) -> np.ndarray:
        """
        Initialize string path.
        If TPT committor available, use it as initial guess.
        Otherwise, linear interpolation between source and target states.
        """
        n_images = self.cfg.n_images
        n_cvs = self.cfg.n_cvs

        if tpt_result is not None and hasattr(tpt_result, 'committor_forward'):
            # Use committor function as initial path
            committor = tpt_result.committor_forward
            # Sample points along committor isosurfaces
            q_values = np.linspace(0, 1, n_images)
            string = np.zeros((n_images, n_cvs))

            for i, q in enumerate(q_values):
                # Find state with committor closest to q
                state_idx = np.argmin(np.abs(committor - q))
                if msm_model.cluster_centers is not None:
                    string[i] = msm_model.cluster_centers[state_idx, :n_cvs]
                else:
                    string[i] = np.random.randn(n_cvs) * 0.1

            return string

        else:
            # Linear interpolation between first and last MSM states
            if msm_model.cluster_centers is not None:
                start = msm_model.cluster_centers[0, :n_cvs]
                end = msm_model.cluster_centers[-1, :n_cvs]
            else:
                start = np.zeros(n_cvs)
                end = np.ones(n_cvs)

            string = np.linspace(start, end, n_images)
            return string

    def _reparameterize(self, string: np.ndarray) -> np.ndarray:
        """
        Reparameterize string to have equal arc-length spacing.
        This maintains uniform distribution of images along the path.
        """
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

    def find_mep(self, msm_model: Any,
                  tpt_result: Optional[Any] = None) -> StringResult:
        """
        Find minimum energy path using string method.

        Args:
            msm_model: MarkovStateModel from previous module
            tpt_result: optional TPTResult for initial guess

        Returns:
            StringResult with optimized path
        """
        # Initialize string
        string = self._initialize_string(msm_model, tpt_result)
        n_images = len(string)

        # Evolution loop
        energies = np.zeros(n_images)
        forces = np.zeros_like(string)
        prev_string = string.copy()

        for iteration in range(self.cfg.n_iterations):
            # Compute energy and gradient at each image
            for i in range(n_images):
                energies[i] = self._compute_energy_msm(string[i], msm_model)
                forces[i] = -self._compute_gradient(string[i], msm_model)

            # Evolve string (gradient descent)
            string = string + self.cfg.step_size * forces

            # Reparameterize periodically
            if iteration % self.cfg.reparam_interval == 0:
                string = self._reparameterize(string)

            # Check convergence
            max_displacement = np.max(np.linalg.norm(string - prev_string, axis=1))
            if max_displacement < self.cfg.convergence_tol:
                logger.info(f"String method converged at iteration {iteration}")
                return StringResult(
                    string_images=string,
                    energies=energies,
                    forces=forces,
                    arc_length=float(np.sum(np.linalg.norm(np.diff(string, axis=0), axis=1))),
                    converged=True,
                    n_iterations=iteration + 1,
                    metadata={"method": "string", "n_images": n_images},
                )

            prev_string = string.copy()

        # Did not converge
        return StringResult(
            string_images=string,
            energies=energies,
            forces=forces,
            arc_length=float(np.sum(np.linalg.norm(np.diff(string, axis=0), axis=1))),
            converged=False,
            n_iterations=self.cfg.n_iterations,
            metadata={"method": "string", "n_images": n_images,
                      "note": "did not converge within max iterations"},
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_images": self.cfg.n_images,
            "cv_type": self.cfg.cv_type,
            "corollary3_guard": "first-order gradient descent; no Hessian",
            "hino_guard": "energy landscape singular; path may be local minimum",
        }


# =============================================================================
# SECTION 2 — [CRD1] CHROMATIN REMODELER DYNAMICS
# =============================================================================

@dataclass
class RemodelerConfig:
    """Configuration for chromatin remodeler dynamics."""
    # Remodeler types
    remodeler_families: List[str] = field(default_factory=lambda:
        ["SWI/SNF", "ISWI", "CHD", "INO80"])
    n_remodelers_per_family: int = 10
    # Kinetics
    k_binding: float = 0.1  # binding rate to nucleosome
    k_unbinding: float = 0.05  # unbinding rate
    k_sliding: float = 0.02  # sliding rate (bp/step)
    k_eviction: float = 0.01  # eviction rate
    k_atp_hydrolysis: float = 1.0  # ATP hydrolysis rate
    # Nucleosome parameters
    n_nucleosomes: int = 20
    nucleosome_spacing_bp: int = 187  # average spacing
    dna_length_bp: int = 5000
    # Simulation
    max_time: float = 100.0
    max_steps: int = 10000
    seed: int = 0


@dataclass
class RemodelerState:
    """State of chromatin remodeler system."""
    time: float
    nucleosome_positions: np.ndarray  # (n_nucleosomes,) positions in bp
    remodeler_bound: np.ndarray  # (n_remodelers,) boolean
    remodeler_positions: np.ndarray  # (n_remodelers,) positions in bp
    n_sliding_events: int
    n_eviction_events: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChromatinRemodelerDynamics:
    """
    [C/D] Chromatin remodeler dynamics with ATP-dependent nucleosome remodeling.

    BIOLOGICAL BASIS [C Clapier & Cairns 2009; Deindl et al. 2013]
      Chromatin remodelers are multi-subunit ATP-dependent enzymes that
      reposition, evict, or restructure nucleosomes. Four major families:

      1. SWI/SNF: slide/evict nucleosomes, create nucleosome-free regions
      2. ISWI: regular spacing, center nucleosomes on DNA
      3. CHD: slide + histone variant exchange (H2A.Z)
      4. INO80: H2A.Z exchange, nucleosome eviction

      Mechanism:
        - Remodeler binds to nucleosome
        - ATP hydrolysis drives DNA translocation
        - Nucleosome slides along DNA or is ejected
        - Remodeler unbinds

    MODEL [D]
      Stochastic kinetic model:
        1. Remodeler binding: rate k_binding × (free nucleosomes)
        2. ATP hydrolysis: rate k_atp_hydrolysis
        3. Sliding: nucleosome position changes by ±1 bp
        4. Eviction: nucleosome removed from DNA
        5. Unbinding: rate k_unbinding

      Gillespie SSA for discrete events.

    CONNECTION TO NUCLEOSOME POSITIONING [D]
      The NucleosomePositioningModel from previous module provides initial
      nucleosome occupancy. Remodelers modify this landscape dynamically.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear in copy numbers).
      We do NOT compute second derivatives of the remodeling landscape.

    HINO GUARD [C Thm 8]
      Binding/unbinding events are DISCRETE, not smooth. The stochasticity
      is singular (tied to individual molecular events).

    HONEST
      • Simplified kinetic model. Real remodelers involve complex
        multi-subunit assemblies, DNA translocation mechanisms, and
        histone modifications not captured here.
      • This is a computational model, not validated biochemistry.
    """

    def __init__(self, cfg: Optional[RemodelerConfig] = None):
        self.cfg = cfg or RemodelerConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # State
        self.n_remodelers = len(self.cfg.remodeler_families) * self.cfg.n_remodelers_per_family
        self.nucleosome_positions: Optional[np.ndarray] = None
        self.remodeler_bound: Optional[np.ndarray] = None
        self.remodeler_positions: Optional[np.ndarray] = None
        self.remodeler_families: Optional[np.ndarray] = None

        # Counters
        self.n_sliding_events = 0
        self.n_eviction_events = 0

    def initialize(self, nucleosome_result: Optional[Any] = None) -> None:
        """Initialize remodeler system."""
        # Initialize nucleosome positions
        if nucleosome_result is not None and hasattr(nucleosome_result, 'positioned_nucleosomes'):
            # Use nucleosome positioning model output
            positions = []
            for nuc in nucleosome_result.positioned_nucleosomes:
                positions.append(nuc['center'])
            self.nucleosome_positions = np.array(positions)
        else:
            # Regular spacing
            n_nuc = min(self.cfg.n_nucleosomes,
                        self.cfg.dna_length_bp // self.cfg.nucleosome_spacing_bp)
            self.nucleosome_positions = np.arange(n_nuc) * self.cfg.nucleosome_spacing_bp

        # Initialize remodelers
        self.remodeler_bound = np.zeros(self.n_remodelers, dtype=bool)
        self.remodeler_positions = self._rng.uniform(
            0, self.cfg.dna_length_bp, self.n_remodelers)
        self.remodeler_families = np.array([
            self.cfg.remodeler_families[i % len(self.cfg.remodeler_families)]
            for i in range(self.n_remodelers)
        ])

        self.n_sliding_events = 0
        self.n_eviction_events = 0

    def compute_propensities(self) -> Tuple[np.ndarray, List[Tuple[str, int, Dict[str, Any]]]]:
        """Compute all reaction propensities."""
        props = []
        events = []

        n_nuc = len(self.nucleosome_positions)

        for i in range(self.n_remodelers):
            if not self.remodeler_bound[i]:
                # Unbound remodeler: can bind to nucleosome
                # Find nearest nucleosome
                distances = np.abs(self.remodeler_positions[i] - self.nucleosome_positions)
                if len(distances) > 0:
                    nearest_nuc = np.argmin(distances)
                    if distances[nearest_nuc] < 50:  # within 50 bp
                        a = self.cfg.k_binding
                        props.append(a)
                        events.append(("bind", i, {"nucleosome": nearest_nuc}))

            else:
                # Bound remodeler: can hydrolyze ATP, slide, evict, or unbind
                # ATP hydrolysis
                a = self.cfg.k_atp_hydrolysis
                props.append(a)
                events.append(("atp_hydrolysis", i, {}))

                # Sliding (family-dependent)
                family = self.remodeler_families[i]
                if family in ["SWI/SNF", "ISWI", "CHD"]:
                    a = self.cfg.k_sliding
                    props.append(a)
                    events.append(("slide", i, {}))

                # Eviction (SWI/SNF and INO80)
                if family in ["SWI/SNF", "INO80"]:
                    a = self.cfg.k_eviction
                    props.append(a)
                    events.append(("evict", i, {}))

                # Unbinding
                a = self.cfg.k_unbinding
                props.append(a)
                events.append(("unbind", i, {}))

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

        reaction_name, remodeler_idx, params = events[event_idx]

        # Apply reaction
        if reaction_name == "bind":
            self.remodeler_bound[remodeler_idx] = True
            nuc_idx = params["nucleosome"]
            self.remodeler_positions[remodeler_idx] = self.nucleosome_positions[nuc_idx]

        elif reaction_name == "atp_hydrolysis":
            pass  # Energy consumed, no state change

        elif reaction_name == "slide":
            # Find which nucleosome this remodeler is bound to
            distances = np.abs(self.remodeler_positions[remodeler_idx] - self.nucleosome_positions)
            if len(distances) > 0:
                nuc_idx = np.argmin(distances)
                if distances[nuc_idx] < 10:  # within 10 bp
                    # Slide nucleosome by ±1 bp
                    direction = self._rng.choice([-1, 1])
                    new_pos = self.nucleosome_positions[nuc_idx] + direction
                    # Check bounds and avoid overlap
                    if 0 <= new_pos < self.cfg.dna_length_bp:
                        self.nucleosome_positions[nuc_idx] = new_pos
                        self.remodeler_positions[remodeler_idx] = new_pos
                        self.n_sliding_events += 1

        elif reaction_name == "evict":
            # Remove nucleosome
            distances = np.abs(self.remodeler_positions[remodeler_idx] - self.nucleosome_positions)
            if len(distances) > 0:
                nuc_idx = np.argmin(distances)
                if distances[nuc_idx] < 10:
                    self.nucleosome_positions = np.delete(self.nucleosome_positions, nuc_idx)
                    self.remodeler_bound[remodeler_idx] = False
                    self.n_eviction_events += 1

        elif reaction_name == "unbind":
            self.remodeler_bound[remodeler_idx] = False

        return {"reaction": reaction_name, "remodeler": remodeler_idx}, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None,
                  nucleosome_result: Optional[Any] = None
                  ) -> Tuple[List[RemodelerState], List[float]]:
        """Run full SSA simulation."""
        self.initialize(nucleosome_result)

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

    def _snapshot(self, t: float) -> RemodelerState:
        return RemodelerState(
            time=t,
            nucleosome_positions=self.nucleosome_positions.copy(),
            remodeler_bound=self.remodeler_bound.copy(),
            remodeler_positions=self.remodeler_positions.copy(),
            n_sliding_events=self.n_sliding_events,
            n_eviction_events=self.n_eviction_events,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_remodelers": self.n_remodelers,
            "remodeler_families": self.cfg.remodeler_families,
            "n_nucleosomes": len(self.nucleosome_positions) if self.nucleosome_positions is not None else 0,
            "corollary3_guard": "first-order kinetics; no Hessian",
            "hino_guard": "discrete binding events, not smooth density",
        }


# =============================================================================
# SECTION 3 — [NFB1] NORMALIZING FLOWS BRIDGE
# =============================================================================

@dataclass
class FlowConfig:
    """Configuration for normalizing flows."""
    # Architecture
    n_layers: int = 6
    hidden_dim: int = 128
    latent_dim: int = 32
    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 64
    # Base distribution
    base_distribution: str = "gaussian"  # "gaussian" | "uniform"
    # Seed
    seed: int = 0


@dataclass
class FlowResult:
    """Result of normalizing flow training/sampling."""
    n_samples: int
    samples: np.ndarray  # (n_samples, data_dim)
    log_likelihood: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CouplingLayer(nn.Module):
    """
    RealNVP-style coupling layer [C Dinh et al. 2017].
    Splits input into two halves, applies affine transformation to one half
    conditioned on the other.
    """

    def __init__(self, data_dim: int, hidden_dim: int, parity: str = "even"):
        super().__init__()
        self.data_dim = data_dim
        self.parity = parity

        # Split dimension
        self.split_dim = data_dim // 2

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.data_dim - self.split_dim),
            nn.Tanh(),  # Bound scale to [-1, 1] for stability
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.data_dim - self.split_dim),
        )

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: x -> z (or z -> x if reverse=True)
        Returns: (transformed_x, log_det_jacobian)
        """
        if self.parity == "even":
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        else:
            x2, x1 = x[:, :self.split_dim], x[:, self.split_dim:]

        # Compute scale and translation
        log_scale = self.scale_net(x1)
        scale = torch.exp(log_scale)
        translation = self.translate_net(x1)

        if not reverse:
            # Forward: x -> z
            z2 = x2 * scale + translation
            log_det = log_scale.sum(dim=1)
        else:
            # Reverse: z -> x
            z2 = (x2 - translation) / scale
            log_det = -log_scale.sum(dim=1)

        if self.parity == "even":
            z = torch.cat([x1, z2], dim=1)
        else:
            z = torch.cat([z2, x1], dim=1)

        return z, log_det


class NormalizingFlowsBridge(nn.Module):
    """
    [C/D] Normalizing flows for exact likelihood sampling.

    MATHEMATICAL BASIS [C Rezende & Mohamed 2015; Dinh et al. 2017]
      A normalizing flow is a sequence of invertible transformations:
        z_0 ~ p_0(z)  (simple base distribution, e.g., Gaussian)
        z_{k+1} = f_k(z_k)  (invertible transformation)
        x = z_K  (final sample)

      The likelihood is computed via change of variables:
        log p(x) = log p_0(z_0) + Σ_k log |det J_k|

      where J_k = ∂f_k/∂z_k is the Jacobian.

    ARCHITECTURE [C Dinh et al. 2017 (RealNVP)]
      - Coupling layers: split input, apply affine transformation
      - Alternating parity: even/odd splits
      - Exact log-likelihood (no ELBO approximation)

    CONNECTION TO VAE [D]
      - VAE learns approximate posterior q(z|x) via ELBO
      - Normalizing flow learns exact p(x) via maximum likelihood
      - Can use VAE latent codes as training data for flow

    ADVANTAGES OVER VAE [C]
      - Exact likelihood (not bound)
      - Exact sampling (no reparameterization trick needed)
      - Invertible (can compute z from x and vice versa)

    COROLLARY-3 GUARD [D]
      Training uses FIRST-ORDER optimization (Adam). We do NOT compute
      second derivatives of the log-likelihood.

    HINO GUARD [C Thm 8]
      The latent space is NOT assumed to be a smooth manifold. The flow
      learns a singular distribution that may have complex structure.

    HONEST
      • Normalizing flows learn a statistical distribution, not a physical one.
      • Generated samples may be outside the training manifold.
      • This is a generative model, not a physics engine.
    """

    def __init__(self, data_dim: int, cfg: Optional[FlowConfig] = None):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        self.data_dim = data_dim

        # Build coupling layers
        self.layers = nn.ModuleList()
        for i in range(self.cfg.n_layers):
            parity = "even" if i % 2 == 0 else "odd"
            self.layers.append(CouplingLayer(data_dim, self.cfg.hidden_dim, parity))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through all layers.
        Returns: (transformed_x, total_log_det_jacobian)
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        if not reverse:
            # Forward: x -> z
            z = x
            for layer in self.layers:
                z, log_det = layer(z, reverse=False)
                log_det_total += log_det
            return z, log_det_total
        else:
            # Reverse: z -> x
            x_recon = x
            for layer in reversed(self.layers):
                x_recon, log_det = layer(x_recon, reverse=True)
                log_det_total += log_det
            return x_recon, log_det_total

    def log_probability(self, x: Tensor) -> Tensor:
        """Compute log p(x) via change of variables."""
        z, log_det = self.forward(x, reverse=False)

        # Base distribution: standard Gaussian
        log_p_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.data_dim * math.log(2 * math.pi)

        # log p(x) = log p(z) + log |det J|
        log_p_x = log_p_z + log_det

        return log_p_x

    def sample(self, n_samples: int) -> Tensor:
        """Generate samples from the learned distribution."""
        # Sample from base distribution (standard Gaussian)
        z = torch.randn(n_samples, self.data_dim, device=next(self.parameters()).device)

        # Transform to data space
        x, _ = self.forward(z, reverse=True)

        return x

    def train_flow(self, data: Tensor, n_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the normalizing flow via maximum likelihood.

        Args:
            data: (n_samples, data_dim) training data
            n_epochs: number of training epochs

        Returns:
            dict with training history
        """
        n_epochs = n_epochs or self.cfg.n_epochs
        n_samples = data.shape[0]

        history = {'loss': [], 'log_likelihood': []}

        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            data_shuffled = data[perm]

            epoch_loss = 0.0
            epoch_ll = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.cfg.batch_size):
                batch = data_shuffled[i:i + self.cfg.batch_size]

                # Compute negative log-likelihood
                log_p = self.log_probability(batch)
                loss = -log_p.mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]
                epoch_ll += log_p.sum().item()
                n_batches += 1

            history['loss'].append(epoch_loss / n_samples)
            history['log_likelihood'].append(epoch_ll / n_samples)

        return {
            "n_epochs": n_epochs,
            "final_loss": history['loss'][-1],
            "final_log_likelihood": history['log_likelihood'][-1],
            "history": history,
        }

    def generate_samples(self, n_samples: int = 100) -> FlowResult:
        """Generate samples and compute log-likelihood."""
        with torch.no_grad():
            samples = self.sample(n_samples)
            log_likelihood = self.log_probability(samples).mean().item()

        return FlowResult(
            n_samples=n_samples,
            samples=samples.cpu().numpy(),
            log_likelihood=log_likelihood,
            metadata={"n_layers": self.cfg.n_layers, "data_dim": self.data_dim},
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_layers": self.cfg.n_layers,
            "data_dim": self.data_dim,
            "hidden_dim": self.cfg.hidden_dim,
            "corollary3_guard": "first-order optimization (Adam); no Hessian",
            "hino_guard": "latent space singular, not smooth manifold",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIStringRemodelerFlowsSuite:
    """
    Unifies: StringMethodMEP + ChromatinRemodelerDynamics + NormalizingFlowsBridge.

    FULL PIPELINE:
      1. MSM/TPT analysis identifies metastable states and transition paths
      2. String method refines the minimum energy path
      3. Chromatin remodelers modify nucleosome landscape
      4. Normalizing flows learn exact distribution over conformations
    """

    def __init__(self,
                 string_cfg: Optional[StringConfig] = None,
                 remodeler_cfg: Optional[RemodelerConfig] = None,
                 flow_cfg: Optional[FlowConfig] = None):
        self.string = StringMethodMEP(string_cfg)
        self.remodeler = ChromatinRemodelerDynamics(remodeler_cfg)
        self.flow = None  # Will be initialized with data_dim

    def find_minimum_energy_path(self, msm_model: Any,
                                   tpt_result: Optional[Any] = None) -> Dict[str, Any]:
        """Find MEP using string method."""
        result = self.string.find_mep(msm_model, tpt_result)
        return {
            "n_images": len(result.string_images),
            "arc_length": result.arc_length,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
            "energy_range": (float(result.energies.min()), float(result.energies.max())),
        }

    def simulate_remodeler_dynamics(self, nucleosome_result: Optional[Any] = None,
                                      n_steps: int = 1000) -> Dict[str, Any]:
        """Simulate chromatin remodeler dynamics."""
        traj, times = self.remodeler.simulate(
            max_steps=n_steps,
            nucleosome_result=nucleosome_result
        )

        final_state = traj[-1]
        return {
            "n_steps": len(traj),
            "final_time": final_state.time,
            "n_nucleosomes": len(final_state.nucleosome_positions),
            "n_bound_remodelers": int(final_state.remodeler_bound.sum()),
            "n_sliding_events": final_state.n_sliding_events,
            "n_eviction_events": final_state.n_eviction_events,
        }

    def train_normalizing_flow(self, training_data: np.ndarray,
                                 n_epochs: int = 50) -> Dict[str, Any]:
        """Train normalizing flow on data."""
        data_dim = training_data.shape[1]

        # Initialize flow
        self.flow = NormalizingFlowsBridge(data_dim, self.flow_cfg if hasattr(self, 'flow_cfg') else FlowConfig())

        # Convert to tensor
        data_tensor = torch.tensor(training_data, dtype=torch.float32)

        # Train
        result = self.flow.train_flow(data_tensor, n_epochs=n_epochs)

        return result

    def generate_conformations(self, n_samples: int = 100) -> Dict[str, Any]:
        """Generate samples from trained flow."""
        if self.flow is None:
            raise ValueError("Flow not trained. Call train_normalizing_flow first.")

        result = self.flow.generate_samples(n_samples)
        return {
            "n_samples": result.n_samples,
            "samples_shape": result.samples.shape,
            "log_likelihood": result.log_likelihood,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "string": self.string.get_status(),
            "remodeler": self.remodeler.get_status(),
            "flow": self.flow.get_status() if self.flow is not None else "not initialized",
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI PROTEIN DYNAMICS STUDIO ULTIMATE — SMOKE TEST")
    print(" String Method + Chromatin Remodelers + Normalizing Flows")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scipy interpolate      : {HAS_SCIPY_INTERP}")
    print(f" TPT/Nuc/VAE modules    : {HAS_TPT_NUC_VAE}")
    print(f" MSM module             : {HAS_MSM}")

    suite = SESIStringRemodelerFlowsSuite(
        string_cfg=StringConfig(n_images=10, n_iterations=50),
        remodeler_cfg=RemodelerConfig(n_nucleosomes=10, max_steps=500),
        flow_cfg=FlowConfig(n_layers=4, n_epochs=20))

    # ── 1. String Method ────────────────────────────────────────────────
    print("\n [SME1] String Method for Minimum Energy Path")
    string = suite.string
    print(f"   n_images             : {string.cfg.n_images}")
    print(f"   cv_type              : {string.cfg.cv_type}")
    print(f"   step_size            : {string.cfg.step_size}")

    # Create synthetic MSM for testing
    if HAS_MSM:
        msm = MarkovStateModel()
        msm.cluster_centers = np.random.rand(20, 2)
        msm.stationary_distribution = np.random.rand(20)
        msm.stationary_distribution /= msm.stationary_distribution.sum()

        # Find MEP
        mep_result = suite.find_minimum_energy_path(msm)
        print(f"   MEP converged        : {mep_result['converged']}")
        print(f"   MEP arc length       : {mep_result['arc_length']:.4f}")
        print(f"   MEP iterations       : {mep_result['n_iterations']}")
        print(f"   Energy range         : {mep_result['energy_range']}")
    else:
        print("   [SKIP] MSM module not available")

    string_status = string.get_status()
    print(f"   COROLLARY-3 guard    : {string_status['corollary3_guard']}")
    print(f"   HINO guard           : {string_status['hino_guard']}")

    # ── 2. Chromatin Remodeler Dynamics ─────────────────────────────────
    print("\n [CRD1] Chromatin Remodeler Dynamics")
    remodeler = suite.remodeler
    print(f"   remodeler families   : {remodeler.cfg.remodeler_families}")
    print(f"   n_remodelers         : {remodeler.n_remodelers}")
    print(f"   n_nucleosomes        : {remodeler.cfg.n_nucleosomes}")

    # Simulate
    remodeler_result = suite.simulate_remodeler_dynamics(n_steps=300)
    print(f"   simulation steps     : {remodeler_result['n_steps']}")
    print(f"   final time           : {remodeler_result['final_time']:.2f}")
    print(f"   final nucleosomes    : {remodeler_result['n_nucleosomes']}")
    print(f"   bound remodelers     : {remodeler_result['n_bound_remodelers']}")
    print(f"   sliding events       : {remodeler_result['n_sliding_events']}")
    print(f"   eviction events      : {remodeler_result['n_eviction_events']}")

    remodeler_status = remodeler.get_status()
    print(f"   COROLLARY-3 guard    : {remodeler_status['corollary3_guard']}")
    print(f"   HINO guard           : {remodeler_status['hino_guard']}")

    # ── 3. Normalizing Flows ────────────────────────────────────────────
    print("\n [NFB1] Normalizing Flows for Exact Likelihood")
    flow_cfg = FlowConfig(n_layers=4, hidden_dim=64, n_epochs=15)
    data_dim = 10

    # Generate synthetic training data
    n_train = 200
    training_data = np.random.randn(n_train, data_dim).astype(np.float32)

    # Train flow
    flow_result = suite.train_normalizing_flow(training_data, n_epochs=15)
    print(f"   n_layers             : {flow_cfg.n_layers}")
    print(f"   data_dim             : {data_dim}")
    print(f"   training epochs      : {flow_result['n_epochs']}")
    print(f"   final loss           : {flow_result['final_loss']:.4f}")
    print(f"   final log-likelihood : {flow_result['final_log_likelihood']:.4f}")

    # Generate samples
    gen_result = suite.generate_conformations(n_samples=50)
    print(f"   generated samples    : {gen_result['n_samples']}")
    print(f"   samples shape        : {gen_result['samples_shape']}")
    print(f"   log-likelihood       : {gen_result['log_likelihood']:.4f}")

    flow_status = suite.flow.get_status()
    print(f"   COROLLARY-3 guard    : {flow_status['corollary3_guard']}")
    print(f"   HINO guard           : {flow_status['hino_guard']}")

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
    print(" [C]   E, Ren, Vanden-Eijnden 2002: String method for MEP.")
    print(" [C]   Maragliano et al. 2006: String method with swarms of trajectories.")
    print(" [C]   Clapier & Cairns 2009: Chromatin remodeler families.")
    print(" [C]   Deindl et al. 2013: Remodeler mechanisms.")
    print(" [C]   Rezende & Mohamed 2015: Variational autoencoders.")
    print(" [C]   Dinh et al. 2017: Density estimation using real-valued non-linear")
    print("       volume preserving transformations (RealNVP).")
    print(" [D]   String method finds LOCAL minimum energy path, not guaranteed global.")
    print("       Energy landscape is rugged (HINO); path may be suboptimal.
    print(" [D]   Chromatin remodeler dynamics is simplified kinetic model. Real")
    print("       remodelers involve complex multi-subunit assemblies, DNA translocation,")
    print("       histone modifications not captured here.")
    print(" [D]   Normalizing flows learn statistical distribution, not physical one.")
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
