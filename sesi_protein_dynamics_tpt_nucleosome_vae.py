"""
=============================================================================
SESI PROTEIN DYNAMICS STUDIO ADVANCED
Transition Path Theory + Nucleosome Positioning + VAE Latent Sampling
Native integration with SDA-ONE and MSM/Chromatin modules
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
SDA-ONE & MSM INTEGRATION
=============================================================================
This module is designed to consume output from:
  1. structural_domain_assembly_one.py (SDA-ONE):
     - init_coords: (N, 3) assembled Cα coordinates
     - seq_features: (N, hidden_dim) per-residue latents
     - sigma: (N, 1) structural regime field

  2. sesi_protein_dynamics_msm_chromatin_enhanced.py:
     - MarkovStateModel: MSM built from trajectories
     - ChromatinAccessibilityModel: SASA and contact maps
     - EnhancedSamplingBridge: metadynamics/REMD setup

Integration points:
  1. TransitionPathTheory.analyze_from_msm(msm_model)
     - Uses transition_matrix and stationary_distribution
     - Computes committor functions and reactive currents
     - Identifies dominant pathways between metastable states

  2. NucleosomePositioningModel.predict_from_sequence(sequence)
     - Uses DNA sequence to predict nucleosome occupancy
     - Can be coupled with chromatin accessibility data

  3. VariationalAutoencoderBridge.encode_structure(coords)
     - Uses SDA-ONE init_coords as input
     - Generates latent representations for sampling

=============================================================================
MODULES
=============================================================================
[TPT1] TransitionPathTheory
      • Committor function q^+(x) = P(reach B before A | start at x)
      • Reactive current: f_ij = π_i T_ij q_j^+ [C Weinan 2006]
      • Transition path probability and flux
      • Dominant pathway extraction (max-flow / min-cut)
      • Connection to MSM metastable states
      • COROLLARY-3 guard: first-order Markov dynamics only
      • HINO guard: discrete states, not smooth density

[NPM1] NucleosomePositioningModel
      • DNA sequence → nucleosome affinity [C Segal 2006]
      • Periodic 10.5 bp signal (helical repeat)
      • Thermodynamic model: Boltzmann distribution over positions
      • Barrier model: poly(dA:dT) tracts exclude nucleosomes
      • Connection to chromatin accessibility
      • COROLLARY-3 guard: first-order sequence statistics
      • HINO guard: positioning landscape rugged, not smooth

[VAE1] VariationalAutoencoderBridge
      • Encoder: structure → (μ, log σ²) [C Kingma & Welling 2014]
      • Decoder: latent → structure
      • Latent space sampling for conformational diversity
      • Connection to MSM states (state-conditioned VAE)
      • KL divergence regularization
      • COROLLARY-3 guard: first-order ELBO optimization
      • HINO guard: latent space singular, not smooth manifold

=============================================================================
HONEST SCOPE
=============================================================================
• TransitionPathTheory: TPT requires a well-sampled MSM. With limited
  data, committor estimates are noisy. This is computational analysis,
  not experimental kinetics.
• NucleosomePositioningModel: Sequence-based prediction is statistical.
  Real nucleosome positioning involves chromatin remodelers, histone
  modifications, and transcription factor binding not captured here.
• VariationalAutoencoderBridge: VAE learns a latent representation but
  does not guarantee Boltzmann sampling. The decoder may produce
  physically implausible structures. This is a generative model, not
  a physics engine.
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

logger = logging.getLogger("sesi.tpt_nucleosome_vae")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from scipy.sparse.csgraph import dijkstra, shortest_path
    from scipy.sparse import csr_matrix
    HAS_SCIPY_GRAPH = True
except Exception:
    HAS_SCIPY_GRAPH = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY_OPT = True
except Exception:
    HAS_SCIPY_OPT = False

# ── SDA-ONE import (graceful) ───────────────────────────────────────────
try:
    from structural_domain_assembly_one import (
        StructuralDomainAssembly,
        DomainAssemblyConfig,
    )
    HAS_SDA = True
except Exception:
    HAS_SDA = False
    StructuralDomainAssembly = None
    DomainAssemblyConfig = None

# ── MSM/Chromatin import (graceful) ─────────────────────────────────────
try:
    from sesi_protein_dynamics_msm_chromatin_enhanced import (
        MarkovStateModel,
        MSMConfig,
        ChromatinAccessibilityModel,
        ChromatinConfig,
    )
    HAS_MSM = True
except Exception:
    HAS_MSM = False
    MarkovStateModel = None
    ChromatinAccessibilityModel = None

__all__ = [
    "TransitionPathTheory", "TPTConfig", "TPTResult",
    "NucleosomePositioningModel", "NucleosomeConfig", "NucleosomeResult",
    "VariationalAutoencoderBridge", "VAEConfig", "VAEResult",
    "SESIProteinDynamicsAdvancedSuite",
]


# =============================================================================
# SECTION 1 — [TPT1] TRANSITION PATH THEORY
# =============================================================================

@dataclass
class TPTConfig:
    """Configuration for Transition Path Theory analysis."""
    # State sets
    source_states: Optional[List[int]] = None  # A states
    target_states: Optional[List[int]] = None  # B states
    # Analysis options
    compute_committor: bool = True
    compute_reactive_current: bool = True
    compute_dominant_pathways: bool = True
    n_dominant_pathways: int = 5
    # Numerical
    convergence_tol: float = 1e-8
    max_iterations: int = 1000
    seed: int = 0


@dataclass
class TPTResult:
    """Result of Transition Path Theory analysis."""
    committor_forward: np.ndarray  # q^+(x) for each state
    committor_backward: np.ndarray  # q^-(x) for each state
    reactive_current: np.ndarray  # f_ij reactive current matrix
    transition_rate: float  # total A→B transition rate
    dominant_pathways: List[Dict[str, Any]]  # top pathways
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransitionPathTheory:
    """
    [C/D] Transition Path Theory for reaction mechanism analysis.

    MATHEMATICAL BASIS [C Weinan & Vanden-Eijnden 2006; Metzner et al. 2009]
      Given a Markov chain with transition matrix T and stationary
      distribution π, TPT characterizes the ensemble of reactive
      trajectories from set A to set B.

      Key quantities:
        1. Forward committor: q^+(x) = P(reach B before A | start at x)
           Satisfies: q^+ = T q^+ in A^c ∩ B^c, q^+ = 0 on A, q^+ = 1 on B

        2. Backward committor: q^-(x) = P(came from A before B | at x)
           For reversible chains: q^-(x) = 1 - q^+(x)

        3. Reactive current: f_ij = π_i T_ij q_j^+ (for i ∉ A, j ∉ B)
           Net reactive current: F_ij = max(0, f_ij - f_ji)

        4. Transition rate: k_AB = Σ_{i∈A} Σ_j π_i T_ij q_j^+

        5. Dominant pathways: paths with maximum reactive flux

    CONNECTION TO MSM [D]
      The MSM from sesi_protein_dynamics_msm_chromatin_enhanced.py provides:
        - transition_matrix: T_ij
        - stationary_distribution: π_i
        - metastable_states: A and B sets

      TPT then identifies:
        - Which conformational changes are on-pathway vs off-pathway
        - The dominant transition mechanisms
        - Rate-limiting steps (bottlenecks)

    COROLLARY-3 GUARD [D]
      TPT uses FIRST-ORDER Markov dynamics. We do NOT compute second
      derivatives of the committor function. The committor equation is
      solved iteratively (Gauss-Seidel), not via Newton's method.

    HINO GUARD [C Thm 8]
      The energy landscape is RUGGED (singular). We represent it as
      DISCRETE states with transition probabilities, not a continuous
      potential. This respects the singularity of the energy measure.

    PROP-3 AWARENESS [P]
      When computing reactive current f_ij = π_i T_ij q_j^+, the product
      of three fields loses regularity. We compute this as a matrix
      operation, which is first-order.

    HONEST
      • TPT requires a well-sampled MSM. With limited data, committor
        estimates are noisy.
      • Dominant pathway extraction is approximate (greedy max-flow).
      • This is computational analysis, not experimental kinetics.
    """

    def __init__(self, cfg: Optional[TPTConfig] = None):
        self.cfg = cfg or TPTConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def compute_committor(self, T: np.ndarray, pi: np.ndarray,
                           source_states: List[int],
                           target_states: List[int]) -> np.ndarray:
        """
        Compute forward committor q^+(x) by solving the linear system:
          q^+ = T q^+ in A^c ∩ B^c
          q^+ = 0 on A
          q^+ = 1 on B

        Uses Gauss-Seidel iteration (COROLLARY-3 guard: first-order).

        Args:
            T: (n_states, n_states) transition matrix
            pi: (n_states,) stationary distribution
            source_states: list of state indices in A
            target_states: list of state indices in B

        Returns:
            q_plus: (n_states,) forward committor
        """
        n_states = T.shape[0]
        q = np.zeros(n_states)

        # Boundary conditions
        q[source_states] = 0.0
        q[target_states] = 1.0

        # Interior states
        interior = [i for i in range(n_states)
                    if i not in source_states and i not in target_states]

        # Gauss-Seidel iteration
        for iteration in range(self.cfg.max_iterations):
            q_old = q.copy()
            for i in interior:
                # q_i = Σ_j T_ij q_j
                q_new = np.sum(T[i, :] * q)
                q[i] = q_new

            # Check convergence
            diff = np.abs(q - q_old).max()
            if diff < self.cfg.convergence_tol:
                logger.debug(f"Committor converged in {iteration+1} iterations")
                break

        return q

    def compute_reactive_current(self, T: np.ndarray, pi: np.ndarray,
                                   q_plus: np.ndarray,
                                   source_states: List[int],
                                   target_states: List[int]) -> np.ndarray:
        """
        Compute reactive current f_ij = π_i T_ij q_j^+.

        PROP-3 AWARENESS: this is a product of three fields (π, T, q),
        computed as matrix operations (first-order).

        Args:
            T: (n_states, n_states) transition matrix
            pi: (n_states,) stationary distribution
            q_plus: (n_states,) forward committor
            source_states: list of state indices in A
            target_states: list of state indices in B

        Returns:
            f: (n_states, n_states) reactive current matrix
        """
        n_states = T.shape[0]

        # f_ij = π_i T_ij q_j^+
        # Only for transitions from non-A to non-B
        f = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i in source_states:
                continue
            for j in range(n_states):
                if j in target_states:
                    continue
                f[i, j] = pi[i] * T[i, j] * q_plus[j]

        return f

    def compute_net_reactive_current(self, f: np.ndarray) -> np.ndarray:
        """
        Compute net reactive current F_ij = max(0, f_ij - f_ji).

        This removes backtracking and gives the effective flux.
        """
        n_states = f.shape[0]
        F = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(i + 1, n_states):
                net = f[i, j] - f[j, i]
                if net > 0:
                    F[i, j] = net
                else:
                    F[j, i] = -net
        return F

    def compute_transition_rate(self, T: np.ndarray, pi: np.ndarray,
                                  q_plus: np.ndarray,
                                  source_states: List[int]) -> float:
        """
        Compute total A→B transition rate:
          k_AB = Σ_{i∈A} Σ_j π_i T_ij q_j^+
        """
        rate = 0.0
        for i in source_states:
            for j in range(T.shape[1]):
                rate += pi[i] * T[i, j] * q_plus[j]
        return float(rate)

    def find_dominant_pathways(self, F: np.ndarray,
                                 source_states: List[int],
                                 target_states: List[int],
                                 n_pathways: int = 5) -> List[Dict[str, Any]]:
        """
        Find dominant pathways using greedy flux decomposition.

        This is an approximation of the max-flow decomposition.
        COROLLARY-3 guard: greedy algorithm, not optimization with Hessian.

        Args:
            F: (n_states, n_states) net reactive current
            source_states: list of state indices in A
            target_states: list of state indices in B
            n_pathways: number of dominant pathways to extract

        Returns:
            list of pathway dicts with 'path', 'flux', 'bottleneck'
        """
        pathways = []
        F_remaining = F.copy()

        for _ in range(n_pathways):
            # Find path with maximum bottleneck flux
            # Greedy: start from source, follow max flux edge
            best_path = None
            best_flux = 0.0

            for start in source_states:
                current = start
                path = [start]
                visited = {start}
                path_flux = float('inf')

                while current not in target_states:
                    # Find outgoing edge with max flux
                    next_states = np.where(F_remaining[current, :] > 0)[0]
                    next_states = [s for s in next_states if s not in visited]

                    if len(next_states) == 0:
                        break

                    next_fluxes = F_remaining[current, next_states]
                    best_next_idx = np.argmax(next_fluxes)
                    next_state = next_states[best_next_idx]
                    edge_flux = F_remaining[current, next_state]

                    path_flux = min(path_flux, edge_flux)
                    path.append(next_state)
                    visited.add(next_state)
                    current = next_state

                if current in target_states and path_flux > best_flux:
                    best_path = path
                    best_flux = path_flux

            if best_path is None or best_flux <= 1e-12:
                break

            # Subtract this pathway's flux
            for i in range(len(best_path) - 1):
                F_remaining[best_path[i], best_path[i + 1]] -= best_flux

            # Identify bottleneck (minimum flux edge)
            bottleneck_idx = 0
            bottleneck_flux = float('inf')
            for i in range(len(best_path) - 1):
                edge_flux = F[best_path[i], best_path[i + 1]]
                if edge_flux < bottleneck_flux:
                    bottleneck_flux = edge_flux
                    bottleneck_idx = i

            pathways.append({
                "path": best_path,
                "flux": float(best_flux),
                "bottleneck_edge": (best_path[bottleneck_idx],
                                     best_path[bottleneck_idx + 1]),
                "bottleneck_flux": float(bottleneck_flux),
            })

        return pathways

    def analyze_from_msm(self, msm_model: Any,
                          source_states: Optional[List[int]] = None,
                          target_states: Optional[List[int]] = None) -> TPTResult:
        """
        Run full TPT analysis on an MSM model.

        Args:
            msm_model: MarkovStateModel from previous module
            source_states: A states (default: first metastable state)
            target_states: B states (default: last metastable state)

        Returns:
            TPTResult with committor, reactive current, pathways
        """
        if msm_model.transition_matrix is None:
            raise ValueError("MSM model has no transition matrix. Run build_from_sda first.")

        T = msm_model.transition_matrix
        pi = msm_model.stationary_distribution

        # Default source/target from metastable states
        if source_states is None or target_states is None:
            metastable = msm_model.get_metastable_states()
            if len(metastable) >= 2:
                source_states = source_states or metastable[0]
                target_states = target_states or metastable[-1]
            else:
                # Fallback: use first and last states
                source_states = [0]
                target_states = [T.shape[0] - 1]

        # Compute committor
        q_plus = self.compute_committor(T, pi, source_states, target_states)

        # Backward committor (for reversible chains: q^- = 1 - q^+)
        q_minus = 1.0 - q_plus

        # Reactive current
        f = self.compute_reactive_current(T, pi, q_plus, source_states, target_states)
        F = self.compute_net_reactive_current(f)

        # Transition rate
        rate = self.compute_transition_rate(T, pi, q_plus, source_states)

        # Dominant pathways
        pathways = []
        if self.cfg.compute_dominant_pathways:
            pathways = self.find_dominant_pathways(
                F, source_states, target_states,
                self.cfg.n_dominant_pathways)

        return TPTResult(
            committor_forward=q_plus,
            committor_backward=q_minus,
            reactive_current=F,
            transition_rate=rate,
            dominant_pathways=pathways,
            metadata={
                "source_states": source_states,
                "target_states": target_states,
                "n_states": T.shape[0],
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "corollary3_guard": "first-order Markov dynamics; Gauss-Seidel iteration; no Hessian",
            "hino_guard": "discrete states, not smooth density; energy landscape singular",
        }


# =============================================================================
# SECTION 2 — [NPM1] NUCLEOSOME POSITIONING MODEL
# =============================================================================

@dataclass
class NucleosomeConfig:
    """Configuration for nucleosome positioning model."""
    # Nucleosome parameters
    nucleosome_size_bp: int = 147  # DNA wrapped around histone octamer
    linker_size_bp: int = 50       # average linker DNA length
    # Sequence features
    use_periodic_signal: bool = True  # 10.5 bp helical repeat
    use_gc_content: bool = True
    use_poly_dA_dT: bool = True       # barrier signal
    # Thermodynamic model
    temperature: float = 1.0          # kBT units
    binding_energy_scale: float = 5.0  # scale for sequence-dependent energy
    # Prediction
    window_size_bp: int = 147
    step_size_bp: int = 1
    # Seed
    seed: int = 0


@dataclass
class NucleosomeResult:
    """Result of nucleosome positioning prediction."""
    occupancy: np.ndarray  # (sequence_length,) nucleosome occupancy probability
    affinity: np.ndarray   # (sequence_length,) binding affinity
    positioned_nucleosomes: List[Dict[str, Any]]  # discrete nucleosome positions
    metadata: Dict[str, Any] = field(default_factory=dict)


class NucleosomePositioningModel:
    """
    [C/D] Nucleosome positioning model from DNA sequence.

    BIOLOGICAL BASIS [C Segal et al. 2006; Kaplan et al. 2009]
      Nucleosomes are the basic units of chromatin: ~147 bp of DNA wrapped
      around a histone octamer. Their positioning along the genome is
      influenced by:
        1. DNA sequence preferences (AA/TT/TA dinucleotides bend easily)
        2. Periodic 10.5 bp signal (helical repeat matching histone surface)
        3. Poly(dA:dT) tracts (rigid, exclude nucleosomes)
        4. Chromatin remodelers and transcription factors (not modeled here)

    MODEL [D]
      We use a simplified thermodynamic model:
        1. Compute sequence-dependent binding energy E(i) for each position i
        2. Occupancy: P(i) = exp(-E(i)/kBT) / Z (Boltzmann distribution)
        3. Periodic signal: AA/TT/TA at 10.5 bp intervals enhance binding
        4. Barrier signal: poly(dA:dT) tracts reduce binding

    ENERGY FUNCTION [D]
      E(i) = -binding_energy_scale * [
          periodic_score(i) + gc_score(i) - poly_dA_dT_penalty(i)
      ]

      where:
        periodic_score(i) = correlation of dinucleotide pattern with 10.5 bp period
        gc_score(i) = GC content contribution
        poly_dA_dT_penalty(i) = penalty for rigid poly(dA:dT) tracts

    COROLLARY-3 GUARD [D]
      All calculations are FIRST-ORDER (linear combinations of sequence
      features). We do NOT compute second derivatives of the energy landscape.

    HINO GUARD [C Thm 8]
      The nucleosome positioning landscape is RUGGED (singular). We represent
      it as discrete occupancy probabilities, not a continuous density.
      Poly(dA:dT) barriers create sharp transitions.

    PROP-3 AWARENESS [P]
      When computing products of sequence features (e.g., periodic × GC),
      the result loses regularity. We use additive combinations instead.

    CONNECTION TO CHROMATIN ACCESSIBILITY [D]
      The ChromatinAccessibilityModel from the previous module computes
      SASA from 3D structure. Nucleosome positioning can be used to
      interpret which regions are accessible vs occluded.

    HONEST
      • Sequence-based prediction is statistical. Real nucleosome positioning
        involves chromatin remodelers, histone modifications, transcription
        factor binding, and active processes not captured here.
      • This is a computational model, not experimental data.
    """

    def __init__(self, cfg: Optional[NucleosomeConfig] = None):
        self.cfg = cfg or NucleosomeConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Dinucleotide preferences (simplified from Segal et al. 2006)
        # AA, TT, TA are favorable; GC is less favorable
        self.dinucleotide_prefs = {
            'AA': 1.0, 'TT': 1.0, 'TA': 0.8,
            'AT': 0.5, 'GC': 0.3, 'CG': 0.3,
            'GA': 0.6, 'TC': 0.6, 'AG': 0.6, 'CT': 0.6,
            'CA': 0.7, 'TG': 0.7, 'GT': 0.7, 'AC': 0.7,
            'GG': 0.4, 'CC': 0.4,
        }

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to numerical encoding."""
        encoding = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return np.array([encoding.get(c, 0) for c in sequence.upper()], dtype=np.int8)

    def compute_periodic_score(self, sequence: str, position: int,
                                 window_size: int = 147) -> float:
        """
        Compute periodic signal score at a position.

        The 10.5 bp helical repeat means certain dinucleotides should
        appear at specific phases to bend around the histone.
        """
        if position + window_size > len(sequence):
            return 0.0

        window = sequence[position:position + window_size].upper()
        score = 0.0

        # Check for favorable dinucleotides at 10.5 bp intervals
        period = 10.5
        for i in range(len(window) - 1):
            dinuc = window[i:i + 2]
            if dinuc in self.dinucleotide_prefs:
                # Phase relative to helical repeat
                phase = (i % period) / period
                # Favorable phases: AA/TT at minor groove facing inward
                if dinuc in ['AA', 'TT', 'TA']:
                    # These are favorable when at specific phases
                    phase_score = math.cos(2 * math.pi * phase) ** 2
                    score += self.dinucleotide_prefs[dinuc] * phase_score

        return score / max(len(window) - 1, 1)

    def compute_gc_score(self, sequence: str, position: int,
                          window_size: int = 147) -> float:
        """Compute GC content contribution."""
        if position + window_size > len(sequence):
            return 0.0

        window = sequence[position:position + window_size].upper()
        gc_count = window.count('G') + window.count('C')
        gc_content = gc_count / max(len(window), 1)

        # Moderate GC is favorable; extreme GC is not
        return 1.0 - abs(gc_content - 0.5) * 2

    def compute_poly_dA_dT_penalty(self, sequence: str, position: int,
                                     window_size: int = 147) -> float:
        """
        Compute penalty for poly(dA:dT) tracts.

        Poly(dA:dT) tracts are rigid and exclude nucleosomes.
        """
        if position + window_size > len(sequence):
            return 0.0

        window = sequence[position:position + window_size].upper()
        penalty = 0.0

        # Count runs of A or T >= 4 bp
        current_run = 1
        for i in range(1, len(window)):
            if window[i] == window[i - 1] and window[i] in 'AT':
                current_run += 1
            else:
                if current_run >= 4:
                    penalty += current_run
                current_run = 1
        if current_run >= 4:
            penalty += current_run

        return penalty / max(len(window), 1)

    def compute_binding_energy(self, sequence: str, position: int) -> float:
        """Compute total binding energy at a position."""
        energy = 0.0

        if self.cfg.use_periodic_signal:
            energy += self.compute_periodic_score(sequence, position)

        if self.cfg.use_gc_content:
            energy += 0.5 * self.compute_gc_score(sequence, position)

        if self.cfg.use_poly_dA_dT:
            energy -= self.compute_poly_dA_dT_penalty(sequence, position)

        return -self.cfg.binding_energy_scale * energy

    def predict_occupancy(self, sequence: str) -> NucleosomeResult:
        """
        Predict nucleosome occupancy along a DNA sequence.

        Args:
            sequence: DNA sequence (A, T, G, C)

        Returns:
            NucleosomeResult with occupancy and affinity arrays
        """
        n = len(sequence)
        window_size = self.cfg.window_size_bp
        step_size = self.cfg.step_size_bp

        # Compute binding energy at each position
        n_positions = (n - window_size) // step_size + 1
        energies = np.zeros(n_positions)

        for i, pos in enumerate(range(0, n - window_size + 1, step_size)):
            energies[i] = self.compute_binding_energy(sequence, pos)

        # Convert to occupancy via Boltzmann distribution
        # P(i) ∝ exp(-E(i) / kBT)
        kBT = self.cfg.temperature
        boltzmann = np.exp(-energies / kBT)
        occupancy = boltzmann / (boltzmann.sum() + 1e-12)

        # Map back to sequence coordinates
        occupancy_full = np.zeros(n)
        affinity_full = np.zeros(n)

        for i, pos in enumerate(range(0, n - window_size + 1, step_size)):
            # Spread occupancy across the nucleosome window
            for j in range(window_size):
                if pos + j < n:
                    occupancy_full[pos + j] += occupancy[i] / window_size
                    affinity_full[pos + j] += -energies[i] / window_size

        # Identify positioned nucleosomes (local maxima of occupancy)
        positioned = []
        threshold = occupancy_full.mean() + 2 * occupancy_full.std()
        i = 0
        while i < n:
            if occupancy_full[i] > threshold:
                # Find peak
                peak_start = i
                peak_end = i
                while peak_end < n and occupancy_full[peak_end] > threshold:
                    peak_end += 1
                peak_center = (peak_start + peak_end) // 2
                positioned.append({
                    "start": peak_start,
                    "end": peak_end,
                    "center": peak_center,
                    "occupancy": float(occupancy_full[peak_center]),
                })
                i = peak_end
            else:
                i += 1

        return NucleosomeResult(
            occupancy=occupancy_full,
            affinity=affinity_full,
            positioned_nucleosomes=positioned,
            metadata={
                "sequence_length": n,
                "n_positions": n_positions,
                "window_size": window_size,
            },
        )

    def predict_from_sda(self, sda_output: Dict[str, Any],
                          sequence: Optional[str] = None) -> NucleosomeResult:
        """
        Predict nucleosome positioning, optionally using SDA-ONE output.

        Args:
            sda_output: dict from StructuralDomainAssembly.forward()
            sequence: DNA sequence (if not provided, uses synthetic)

        Returns:
            NucleosomeResult
        """
        if sequence is None:
            # Generate synthetic sequence from sigma field
            sigma = sda_output.get("sigma")
            if sigma is not None:
                if isinstance(sigma, torch.Tensor):
                    sigma = sigma.detach().cpu().numpy()
                # Use sigma to generate a pseudo-sequence
                n_residues = sigma.shape[0]
                # Map sigma to nucleotides (simplified)
                sequence = ''.join(['A' if s < 0.25 else 'T' if s < 0.5
                                     else 'G' if s < 0.75 else 'C'
                                     for s in sigma.flatten()[:n_residues]])
            else:
                sequence = "ATGC" * 100  # fallback

        return self.predict_occupancy(sequence)

    def get_status(self) -> Dict[str, Any]:
        return {
            "nucleosome_size_bp": self.cfg.nucleosome_size_bp,
            "use_periodic_signal": self.cfg.use_periodic_signal,
            "use_poly_dA_dT": self.cfg.use_poly_dA_dT,
            "corollary3_guard": "first-order sequence statistics; no Hessian",
            "hino_guard": "positioning landscape rugged; discrete occupancy",
        }


# =============================================================================
# SECTION 3 — [VAE1] VARIATIONAL AUTOENCODER BRIDGE
# =============================================================================

@dataclass
class VAEConfig:
    """Configuration for VAE latent space sampling."""
    # Architecture
    input_dim: int = 3 * 100  # flattened coordinates (N_residues * 3)
    hidden_dim: int = 256
    latent_dim: int = 32
    n_residues: int = 100
    # Training
    kl_weight: float = 0.1  # β-VAE weight
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 32
    # Sampling
    n_samples: int = 100
    # SDA integration
    use_seq_features: bool = True
    seq_feature_dim: int = 64
    seed: int = 0


@dataclass
class VAEResult:
    """Result of VAE encoding/decoding."""
    latent_mean: np.ndarray  # (n_samples, latent_dim)
    latent_logvar: np.ndarray  # (n_samples, latent_dim)
    reconstructed: np.ndarray  # (n_samples, n_residues, 3)
    kl_divergence: float
    reconstruction_loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VariationalAutoencoderBridge(nn.Module):
    """
    [C/D] Variational Autoencoder for protein conformational sampling.

    BIOLOGICAL BASIS [C Kingma & Welling 2014; Bowman et al. 2018]
      VAEs learn a latent representation of data that can be sampled
      to generate new examples. For proteins:
        - Encoder maps structure → latent distribution (μ, σ²)
        - Decoder maps latent code → structure
        - Sampling in latent space generates conformational diversity

    ARCHITECTURE [D]
      Encoder:
        input (N, 3) → flatten → Linear → ReLU → Linear → (μ, log σ²)

      Decoder:
        latent (latent_dim,) → Linear → ReLU → Linear → reshape (N, 3)

    LOSS FUNCTION [C Kingma & Welling 2014]
      ELBO = E[log p(x|z)] - β · KL(q(z|x) || p(z))

      where:
        - Reconstruction loss: MSE between input and output
        - KL divergence: regularizes latent space to be Gaussian

    CONNECTION TO MSM [D]
      Each MSM state can be encoded into latent space. Interpolation
      between states in latent space generates transition pathways.

    CONNECTION TO SDA-ONE [D]
      SDA-ONE's init_coords serve as input structures.
      SDA-ONE's seq_features can condition the VAE for
      sequence-aware generation.

    COROLLARY-3 GUARD [D]
      VAE training uses FIRST-ORDER gradient descent (Adam optimizer).
      We do NOT compute second derivatives (no natural gradient, no
      Newton's method). The ELBO is optimized with stochastic gradients.

    HINO GUARD [C Thm 8]
      The latent space is NOT assumed to be a smooth manifold. We
      represent it as a discrete set of samples from the learned
      distribution. The energy landscape in latent space may be singular.

    PROP-3 AWARENESS [P]
      When computing KL divergence (products of distributions), the
      result loses regularity. We use the closed-form Gaussian KL,
      which is well-defined.

    HONEST
      • VAE learns a latent representation but does NOT guarantee
        Boltzmann sampling. The decoder may produce physically
        implausible structures.
      • This is a generative model, not a physics engine.
      • Latent space interpolation does not follow true reaction pathways.
    """

    def __init__(self, cfg: Optional[VAEConfig] = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.cfg.hidden_dim, self.cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.cfg.hidden_dim, self.cfg.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.input_dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: (batch, input_dim) flattened coordinates

        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)

        COROLLARY-3 guard: this is a first-order transformation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent code to structure.

        Args:
            z: (batch, latent_dim)

        Returns:
            x_recon: (batch, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: (batch, input_dim)

        Returns:
            x_recon, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x: Tensor, x_recon: Tensor,
                       mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute VAE loss: reconstruction + β * KL.

        KL(q(z|x) || p(z)) for Gaussian q and p:
          KL = -0.5 * Σ(1 + log σ² - μ² - σ²)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

        # Total loss with β weight
        total_loss = recon_loss + self.cfg.kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def train(self, data: Tensor, n_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the VAE on a dataset of structures.

        Args:
            data: (n_samples, n_residues, 3) coordinates

        Returns:
            dict with training history
        """
        n_epochs = n_epochs or self.cfg.n_epochs
        n_samples = data.shape[0]

        # Flatten for encoder
        data_flat = data.reshape(n_samples, -1)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)

        history = {'loss': [], 'recon': [], 'kl': []}

        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            data_shuffled = data_flat[perm]

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            # Mini-batch training
            for i in range(0, n_samples, self.cfg.batch_size):
                batch = data_shuffled[i:i + self.cfg.batch_size]

                optimizer.zero_grad()
                x_recon, mu, logvar = self.forward(batch)
                loss, recon, kl = self.loss_function(batch, x_recon, mu, logvar)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]
                epoch_recon += recon.item() * batch.shape[0]
                epoch_kl += kl.item() * batch.shape[0]

            history['loss'].append(epoch_loss / n_samples)
            history['recon'].append(epoch_recon / n_samples)
            history['kl'].append(epoch_kl / n_samples)

        return history

    def sample_latent(self, n_samples: int = 1) -> Tensor:
        """Sample from the prior p(z) = N(0, I)."""
        return torch.randn(n_samples, self.cfg.latent_dim)

    def generate_structures(self, n_samples: int = 100,
                              latent_codes: Optional[Tensor] = None) -> Tensor:
        """
        Generate protein structures from latent space.

        Args:
            n_samples: number of structures to generate
            latent_codes: optional specific latent codes (else sample from prior)

        Returns:
            structures: (n_samples, n_residues, 3)
        """
        if latent_codes is None:
            latent_codes = self.sample_latent(n_samples)

        with torch.no_grad():
            generated = self.decode(latent_codes)

        return generated.reshape(n_samples, self.cfg.n_residues, 3)

    def encode_structure(self, coords: np.ndarray) -> VAEResult:
        """
        Encode a structure to latent space.

        Args:
            coords: (n_residues, 3) coordinates

        Returns:
            VAEResult with latent mean, logvar, reconstruction
        """
        # Ensure correct shape
        if coords.ndim == 2:
            coords = coords.reshape(1, coords.shape[0], coords.shape[1])

        n_samples = coords.shape[0]
        coords_flat = torch.tensor(coords, dtype=torch.float32).reshape(n_samples, -1)

        with torch.no_grad():
            mu, logvar = self.encode(coords_flat)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)

        recon = x_recon.reshape(n_samples, self.cfg.n_residues, 3)

        # Compute losses
        _, recon_loss, kl_loss = self.loss_function(coords_flat, x_recon, mu, logvar)

        return VAEResult(
            latent_mean=mu.numpy(),
            latent_logvar=logvar.numpy(),
            reconstructed=recon.numpy(),
            kl_divergence=float(kl_loss),
            reconstruction_loss=float(recon_loss),
        )

    def interpolate_between_states(self, coords_a: np.ndarray,
                                     coords_b: np.ndarray,
                                     n_intermediates: int = 10) -> Tensor:
        """
        Interpolate between two structures in latent space.

        This generates a putative transition pathway (NOT a true reaction path).

        Args:
            coords_a: (n_residues, 3) starting structure
            coords_b: (n_residues, 3) ending structure
            n_intermediates: number of intermediate structures

        Returns:
            interpolated: (n_intermediates, n_residues, 3)
        """
        # Encode both structures
        result_a = self.encode_structure(coords_a.reshape(1, -1, 3))
        result_b = self.encode_structure(coords_b.reshape(1, -1, 3))

        z_a = torch.tensor(result_a.latent_mean[0], dtype=torch.float32)
        z_b = torch.tensor(result_b.latent_mean[0], dtype=torch.float32)

        # Interpolate in latent space
        alphas = torch.linspace(0, 1, n_intermediates)
        interpolated = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z_a + alpha * z_b
            with torch.no_grad():
                coords_interp = self.decode(z_interp.unsqueeze(0))
            interpolated.append(coords_interp.reshape(self.cfg.n_residues, 3))

        return torch.stack(interpolated)

    def get_status(self) -> Dict[str, Any]:
        return {
            "input_dim": self.cfg.input_dim,
            "latent_dim": self.cfg.latent_dim,
            "kl_weight": self.cfg.kl_weight,
            "corollary3_guard": "first-order gradient descent; no Hessian",
            "hino_guard": "latent space not assumed smooth; discrete samples",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIProteinDynamicsAdvancedSuite:
    """
    Unifies: TransitionPathTheory + NucleosomePositioningModel +
             VariationalAutoencoderBridge.

    FULL PIPELINE:
      1. SDA-ONE produces init_coords, seq_features, sigma
      2. MSM analyzes conformational dynamics
      3. TPT identifies dominant transition pathways
      4. Nucleosome positioning predicts chromatin organization
      5. VAE generates new conformations via latent sampling
    """

    def __init__(self,
                 tpt_cfg: Optional[TPTConfig] = None,
                 nucleosome_cfg: Optional[NucleosomeConfig] = None,
                 vae_cfg: Optional[VAEConfig] = None):
        self.tpt = TransitionPathTheory(tpt_cfg)
        self.nucleosome = NucleosomePositioningModel(nucleosome_cfg)
        self.vae = VariationalAutoencoderBridge(vae_cfg)

    def run_full_analysis(self, sda_output: Dict[str, Any],
                            msm_model: Optional[Any] = None,
                            sequence: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full protein dynamics analysis pipeline.

        Args:
            sda_output: dict from StructuralDomainAssembly.forward()
            msm_model: optional MarkovStateModel (for TPT)
            sequence: optional DNA sequence (for nucleosome positioning)

        Returns:
            dict with all analysis results
        """
        results = {}

        # 1. VAE encoding of SDA structure
        init_coords = sda_output["init_coords"]
        if isinstance(init_coords, torch.Tensor):
            init_coords = init_coords.detach().cpu().numpy()

        # Adjust VAE config for actual structure size
        n_residues = init_coords.shape[0]
        self.vae.cfg.n_residues = n_residues
        self.vae.cfg.input_dim = n_residues * 3

        vae_result = self.vae.encode_structure(init_coords)
        results["vae"] = {
            "latent_dim": self.vae.cfg.latent_dim,
            "kl_divergence": vae_result.kl_divergence,
            "reconstruction_loss": vae_result.reconstruction_loss,
        }

        # 2. TPT analysis (if MSM available)
        if msm_model is not None and msm_model.transition_matrix is not None:
            tpt_result = self.tpt.analyze_from_msm(msm_model)
            results["tpt"] = {
                "transition_rate": tpt_result.transition_rate,
                "n_dominant_pathways": len(tpt_result.dominant_pathways),
                "dominant_pathways": tpt_result.dominant_pathways[:3],
            }
        else:
            results["tpt"] = {"note": "No MSM model provided"}

        # 3. Nucleosome positioning
        nucleosome_result = self.nucleosome.predict_from_sda(sda_output, sequence)
        results["nucleosome"] = {
            "n_positioned_nucleosomes": len(nucleosome_result.positioned_nucleosomes),
            "mean_occupancy": float(nucleosome_result.occupancy.mean()),
            "max_occupancy": float(nucleosome_result.occupancy.max()),
        }

        return results

    def generate_conformational_ensemble(self, sda_output: Dict[str, Any],
                                           n_samples: int = 10) -> Dict[str, Any]:
        """
        Generate a conformational ensemble using VAE latent sampling.

        Args:
            sda_output: dict from StructuralDomainAssembly.forward()
            n_samples: number of conformations to generate

        Returns:
            dict with generated structures
        """
        init_coords = sda_output["init_coords"]
        if isinstance(init_coords, torch.Tensor):
            init_coords = init_coords.detach().cpu().numpy()

        n_residues = init_coords.shape[0]
        self.vae.cfg.n_residues = n_residues
        self.vae.cfg.input_dim = n_residues * 3

        # Generate new structures
        generated = self.vae.generate_structures(n_samples=n_samples)

        return {
            "n_samples": n_samples,
            "structures": generated.numpy(),
            "mean_rmsd_from_original": float(
                np.sqrt(((generated.numpy() - init_coords.reshape(1, n_residues, 3))**2).sum(-1).mean())
            ),
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "tpt": self.tpt.get_status(),
            "nucleosome": self.nucleosome.get_status(),
            "vae": self.vae.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI PROTEIN DYNAMICS STUDIO ADVANCED — SMOKE TEST")
    print(" Transition Path Theory + Nucleosome Positioning + VAE")
    print(" Native SDA-ONE & MSM Integration")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scipy graph available  : {HAS_SCIPY_GRAPH}")
    print(f" SDA-ONE available      : {HAS_SDA}")
    print(f" MSM module available   : {HAS_MSM}")

    suite = SESIProteinDynamicsAdvancedSuite(
        tpt_cfg=TPTConfig(n_dominant_pathways=3),
        nucleosome_cfg=NucleosomeConfig(),
        vae_cfg=VAEConfig(n_residues=50, input_dim=150, latent_dim=16))

    # ── 1. Create synthetic SDA-ONE output ──────────────────────────────
    print("\n [SDA] Synthetic SDA-ONE Output")
    n_residues = 50
    sda_output = {
        "init_coords": np.random.randn(n_residues, 3) * 10,
        "seq_features": np.random.randn(n_residues, 64),
        "sigma": np.random.rand(n_residues, 1),
    }
    print(f"   n_residues           : {n_residues}")
    print(f"   init_coords shape    : {sda_output['init_coords'].shape}")
    print(f"   seq_features shape   : {sda_output['seq_features'].shape}")
    print(f"   sigma shape          : {sda_output['sigma'].shape}")

    # ── 2. Transition Path Theory ───────────────────────────────────────
    print("\n [TPT1] Transition Path Theory")
    tpt = suite.tpt

    # Create synthetic MSM for testing
    n_states = 20
    T_synthetic = np.random.rand(n_states, n_states)
    T_synthetic = np.random.rand(n_states, n_states)
    T_synthetic = T_synthetic / T_synthetic.sum(axis=1, keepdims=True)
    pi_synthetic = np.ones(n_states) / n_states

    source_states = [0, 1, 2]
    target_states = [17, 18, 19]

    # Compute committor
    q_plus = tpt.compute_committor(T_synthetic, pi_synthetic, source_states, target_states)
    print(f"   committor q+ range   : [{q_plus.min():.4f}, {q_plus.max():.4f}]")
    print(f"   q+ on source states  : {q_plus[source_states].mean():.4f} (should be 0)")
    print(f"   q+ on target states  : {q_plus[target_states].mean():.4f} (should be 1)")

    # Compute reactive current
    f = tpt.compute_reactive_current(T_synthetic, pi_synthetic, q_plus, source_states, target_states)
    F = tpt.compute_net_reactive_current(f)
    print(f"   reactive current max : {F.max():.6f}")

    # Compute transition rate
    rate = tpt.compute_transition_rate(T_synthetic, pi_synthetic, q_plus, source_states)
    print(f"   transition rate      : {rate:.6f}")

    # Find dominant pathways
    pathways = tpt.find_dominant_pathways(F, source_states, target_states, n_pathways=3)
    print(f"   dominant pathways    : {len(pathways)}")
    for i, pw in enumerate(pathways[:2]):
        print(f"     pathway {i}: flux={pw['flux']:.4f}, path_length={len(pw['path'])}")

    tpt_status = tpt.get_status()
    print(f"   COROLLARY-3 guard    : {tpt_status['corollary3_guard']}")
    print(f"   HINO guard           : {tpt_status['hino_guard']}")

    # ── 3. Nucleosome Positioning ───────────────────────────────────────
    print("\n [NPM1] Nucleosome Positioning Model")
    nucleosome = suite.nucleosome
    print(f"   nucleosome size      : {nucleosome.cfg.nucleosome_size_bp} bp")
    print(f"   use periodic signal  : {nucleosome.cfg.use_periodic_signal}")
    print(f"   use poly(dA:dT)      : {nucleosome.cfg.use_poly_dA_dT}")

    # Generate synthetic DNA sequence
    dna_sequence = "ATGCATGCATGCATGCATGC" * 50  # 1000 bp
    print(f"   sequence length      : {len(dna_sequence)} bp")

    # Predict occupancy
    nuc_result = nucleosome.predict_occupancy(dna_sequence)
    print(f"   occupancy shape      : {nuc_result.occupancy.shape}")
    print(f"   mean occupancy       : {nuc_result.occupancy.mean():.4f}")
    print(f"   max occupancy        : {nuc_result.occupancy.max():.4f}")
    print(f"   positioned nucleosomes: {len(nuc_result.positioned_nucleosomes)}")

    # Predict from SDA output
    nuc_from_sda = nucleosome.predict_from_sda(sda_output)
    print(f"   from SDA: positioned : {len(nuc_from_sda.positioned_nucleosomes)}")

    nuc_status = nucleosome.get_status()
    print(f"   COROLLARY-3 guard    : {nuc_status['corollary3_guard']}")
    print(f"   HINO guard           : {nuc_status['hino_guard']}")

    # ── 4. Variational Autoencoder ──────────────────────────────────────
    print("\n [VAE1] Variational Autoencoder Bridge")
    vae = suite.vae
    print(f"   input_dim            : {vae.cfg.input_dim}")
    print(f"   latent_dim           : {vae.cfg.latent_dim}")
    print(f"   kl_weight (β)        : {vae.cfg.kl_weight}")

    # Encode SDA structure
    vae_result = vae.encode_structure(sda_output["init_coords"])
    print(f"   latent_mean shape    : {vae_result.latent_mean.shape}")
    print(f"   latent_logvar shape  : {vae_result.latent_logvar.shape}")
    print(f"   KL divergence        : {vae_result.kl_divergence:.4f}")
    print(f"   reconstruction loss  : {vae_result.reconstruction_loss:.4f}")

    # Generate new structures
    generated = vae.generate_structures(n_samples=5)
    print(f"   generated shape      : {generated.shape}")
    print(f"   generated finite     : {torch.isfinite(generated).all().item()}")

    # Interpolate between two structures
    coords_a = sda_output["init_coords"]
    coords_b = coords_a + np.random.randn(n_residues, 3) * 2
    interpolated = vae.interpolate_between_states(coords_a, coords_b, n_intermediates=5)
    print(f"   interpolated shape   : {interpolated.shape}")

    vae_status = vae.get_status()
    print(f"   COROLLARY-3 guard    : {vae_status['corollary3_guard']}")
    print(f"   HINO guard           : {vae_status['hino_guard']}")

    # ── 5. Full Analysis Pipeline ───────────────────────────────────────
    print("\n [PIPELINE] Full Analysis Pipeline")
    # Create a simple MSM for testing
    if HAS_MSM:
        msm = MarkovStateModel(MSMConfig(n_states=20))
        msm.transition_matrix = T_synthetic
        msm.stationary_distribution = pi_synthetic
        msm.cluster_centers = np.random.rand(20, 10)
        msm.state_assignments = np.random.randint(0, 20, 100)
    else:
        msm = None

    full_result = suite.run_full_analysis(sda_output, msm_model=msm, sequence=dna_sequence)
    print(f"   VAE latent_dim       : {full_result['vae']['latent_dim']}")
    print(f"   VAE KL divergence    : {full_result['vae']['kl_divergence']:.4f}")
    print(f"   TPT transition rate  : {full_result['tpt'].get('transition_rate', 'N/A')}")
    print(f"   TPT pathways         : {full_result['tpt'].get('n_dominant_pathways', 'N/A')}")
    print(f"   Nucleosomes          : {full_result['nucleosome']['n_positioned_nucleosomes']}")

    # ── 6. Conformational Ensemble Generation ───────────────────────────
    print("\n [ENSEMBLE] Conformational Ensemble Generation")
    ensemble_result = suite.generate_conformational_ensemble(sda_output, n_samples=10)
    print(f"   n_samples            : {ensemble_result['n_samples']}")
    print(f"   structures shape     : {ensemble_result['structures'].shape}")
    print(f"   mean RMSD from orig  : {ensemble_result['mean_rmsd_from_original']:.4f} Å")

    # ── 7. Full Status ──────────────────────────────────────────────────
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
    print(" [C]   Weinan & Vanden-Eijnden 2006: Transition Path Theory.")
    print(" [C]   Metzner, Schütte, Vanden-Eijnden 2009: TPT for Markov chains.")
    print(" [C]   Segal et al. 2006: Nucleosome positioning from sequence.")
    print(" [C]   Kaplan et al. 2009: DNA sequence dictates nucleosome organization.")
    print(" [C]   Kingma & Welling 2014: Variational Autoencoders.")
    print(" [C]   Bowman et al. 2018: VAE for protein conformational dynamics.")
    print(" [D]   TPT requires well-sampled MSM. With limited data, committor")
    print("       estimates are noisy. This is computational analysis, not")
    print("       experimental kinetics.")
    print(" [D]   Nucleosome positioning is sequence-based statistical model.")
    print("       Real positioning involves remodelers, histone modifications,")
    print("       transcription factors not captured here.")
    print(" [D]   VAE does NOT guarantee Boltzmann sampling. Decoder may produce")
    print("       physically implausible structures. Latent interpolation is NOT")
    print("       a true reaction pathway.")
    print(" [D]   COROLLARY-3 guard: all fields first-order only; no Hessian.")
    print(" [C]   HINO guard: energy landscapes singular; discrete states/samples,")
    print("       not smooth densities.")
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
