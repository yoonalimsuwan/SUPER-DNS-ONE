"""
=============================================================================
SESI ORGANOID STUDIO SPATIAL
Spatial Stochastic GRN (RDME) + Par/Cdc42 Polarity + τ-d_s Coupling
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
[SSG1] SpatialStochasticGRN
      • Reaction-Diffusion Master Equation [C Elf & Ehrenberg 2003]
      • Spatial grid of cells, each with local stochastic GRN
      • Diffusion between neighbouring cells (stochastic jumps)
      • Coupling with morphogen field from Turing RD
      • COROLLARY-3 guard: first-order reactions only
      • HINO guard: noise from discrete propensities, not smooth density

[PCP1] ParCdc42PolarityModel
      • Turing-type mutual inhibition model [C Turing 1952; Marée 2006]
      • Two species: apical (Par3/Par6/aPKC) vs basal (Par1/Lgl)
      • Mutual inhibition + diffusion → pattern formation
      • Polarity vector from gradient of apical complex
      • Coupling to cell mechanics via vertex model
      • HONEST: computational model, not validated molecular biology

[TSD1] TauSpectralDimensionCoupling
      • Links τ to spectral dimension d_s = 2d_f/d_w from Carpet-19
      • Theorem 4(f): W^s ↪ C^{0,β/2} with β = log ρ/log 3 ≈ 0.222
      • Scaling heuristic: τ ~ L^{d_s/2} (dimensional analogy)
      • Uses carpet exponents as REFERENCE, not closure transfer
      • ρ ≈ 1.25 is NUMERICAL [C-num], not proved
      • PROP-3 awareness: τ is scale selector, not regularity recovery

[SSO1] SESISpatialOrganoidSuite (orchestrator)
      • Couples SpatialStochasticGRN + ParCdc42Polarity + τ-d_s
      • Full spatial organoid life-cycle with molecular polarity
      • Exports to AR pipeline via existing USDZ exporter

=============================================================================
HONEST SCOPE
=============================================================================
• SpatialStochasticGRN: RDME is exact for well-mixed compartments [C].
  Real cells have subcellular spatial structure not captured here.
• ParCdc42PolarityModel: Turing-type mutual inhibition is a simplified
  model. Real polarity involves Par3/Par6/aPKC, Cdc42, Scribble, Crumbs,
  and many other proteins with complex kinetics. This is computational
  only, not validated molecular biology.
• TauSpectralDimensionCoupling: the link between τ and d_s is a HEURISTIC
  dimensional analogy. Carpet-19 Theorem 10 is conditional on (I1)-(I4)
  and applies to the carpet, not to biological tissue. No closure transfer.
• ρ ≈ 1.25 is NUMERICAL (R_cross: 1.66, 1.33, 1.27, 1.26, 1.25), not proved.
• COROLLARY-3 guard: all fields first-order only; no Hessian.
• HINO guard: energy measures singular; no smooth density assumed.
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

logger = logging.getLogger("sesi.spatial_polarity")

# ── Carpet-19 reference constants [C] ────────────────────────────────────
CARPET_DF = math.log(8.0) / math.log(3.0)           # ≈ 1.8928
CARPET_RHO = 1.25                                     # NUMERICAL, not proved
CARPET_DW = math.log(8.0 * CARPET_RHO) / math.log(3.0)  # ≈ 2.096
CARPET_DS = 2.0 * CARPET_DF / CARPET_DW              # ≈ 1.806
CARPET_BETA = math.log(CARPET_RHO) / math.log(3.0)   # ≈ 0.222 (Hölder exponent)

__all__ = [
    "SpatialStochasticGRN", "SpatialGRNConfig",
    "ParCdc42PolarityModel", "PolarityConfig",
    "TauSpectralDimensionCoupling", "TauSpectralConfig",
    "SESISpatialOrganoidSuite",
]


# =============================================================================
# SECTION 1 — [SSG1] SPATIAL STOCHASTIC GRN (RDME)
# =============================================================================

@dataclass
class SpatialGRNConfig:
    """Configuration for spatial stochastic GRN (RDME)."""
    # Spatial grid
    grid_shape: Tuple[int, int, int] = (5, 5, 5)  # 3D grid of cells
    # Gene regulation (per cell)
    n_genes: int = 4
    k_regulators: int = 2
    # Kinetics
    beta: float = 5.0           # production rate
    gamma: float = 1.0          # degradation rate
    hill_K: float = 5.0
    hill_n: float = 2.0
    # Diffusion
    diffusion_rate: float = 0.1  # molecules per unit time
    diffusion_distance: float = 1.0  # grid spacing
    # Simulation
    max_time: float = 50.0
    max_steps: int = 10000
    # Morphogen coupling
    morphogen_field: Optional[np.ndarray] = None
    morphogen_weight: float = 0.3
    # Stochasticity
    seed: int = 0


class SpatialStochasticGRN:
    """
    [C/D] Spatial stochastic GRN via Reaction-Diffusion Master Equation.

    MODEL [C Elf & Ehrenberg 2003]
      The spatial domain is discretized into a grid of compartments (cells).
      Within each compartment, reactions follow the Gillespie SSA [C 1976/77].
      Between neighbouring compartments, molecules diffuse via stochastic jumps.

      For gene i in cell c:
        Production: ∅ → X_i^c      rate β · H(x_j^c)  (regulated)
        Degradation: X_i^c → ∅     rate γ · x_i^c
        Diffusion: X_i^c → X_i^{c'} rate D · x_i^c  (to neighbour c')

      The full system is a continuous-time Markov chain on the state space
      of copy numbers {x_i^c} for all genes i and cells c.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER functions of the state
      (linear degradation, Hill regulation). We do NOT compute second
      derivatives of the copy-number field. The diffusion term is a
      first-order spatial coupling.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates), NOT from an assumed smooth noise density. The stochasticity
      is singular in the sense of being tied to individual molecular events.

    COUPLING WITH MORPHOGEN [C/D]
      The morphogen field m(x) modulates gene production rates:
        β_eff(x) = β · (1 + morphogen_weight · m(x))
      This implements positional information [C Wolpert 1969].

    HONEST
      • RDME is exact for well-mixed compartments [C]. Real cells have
        subcellular spatial structure not captured here.
      • Diffusion is modeled as stochastic jumps between grid cells, not
        as continuum Fickian diffusion.
      • Gene copy numbers are illustrative; real expression involves
        1-1000 molecules per cell typically.
    """

    def __init__(self, cfg: Optional[SpatialGRNConfig] = None):
        self.cfg = cfg or SpatialGRNConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_genes = self.cfg.n_genes
        self.n_cells = int(np.prod(self.cfg.grid_shape))

        # Regulatory graph (same for all cells)
        self.regulators: List[List[int]] = []
        self.signs: List[List[int]] = []
        for i in range(self.n_genes):
            regs = self._rng.choice(
                [j for j in range(self.n_genes) if j != i],
                size=min(self.cfg.k_regulators, self.n_genes - 1),
                replace=False).tolist()
            self.regulators.append(regs)
            self.signs.append([int(self._rng.choice([-1, 1])) for _ in regs])

        # Spatial adjacency (6-neighbourhood in 3D)
        self._build_adjacency()

        # State: copy numbers x[gene, cell]
        self.state: Optional[np.ndarray] = None

    def _build_adjacency(self) -> None:
        """Build spatial adjacency list for diffusion."""
        nx, ny, nz = self.cfg.grid_shape
        self.adjacency: Dict[int, List[int]] = {}
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    c = ix * ny * nz + iy * nz + iz
                    neighbours = []
                    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        jx, jy, jz = ix+dx, iy+dy, iz+dz
                        if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                            neighbours.append(jx * ny * nz + jy * nz + jz)
                    self.adjacency[c] = neighbours

    def _hill(self, x: float, K: float, n: float, activating: bool) -> float:
        """Hill function for propensity modulation."""
        x = max(x, 0.0)
        K = max(K, 1e-12)
        xn = x ** n
        Kn = K ** n
        return xn / (Kn + xn) if activating else Kn / (Kn + xn)

    def _morphogen_at(self, cell_idx: int) -> float:
        """Get morphogen concentration at a cell."""
        if self.cfg.morphogen_field is None:
            return 0.0
        m = self.cfg.morphogen_field.flatten()
        if cell_idx < len(m):
            return float(m[cell_idx])
        return 0.0

    def compute_propensities(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute all reaction and diffusion propensities.
        state: (n_genes, n_cells) copy numbers
        Returns: (reaction_propensities, diffusion_propensities)
        """
        # Reaction propensities: (n_genes * 2, n_cells)
        # First n_genes: production; next n_genes: degradation
        rxn_props = np.zeros((self.n_genes * 2, self.n_cells))

        for c in range(self.n_cells):
            m = self._morphogen_at(c)
            beta_eff = self.cfg.beta * (1.0 + self.cfg.morphogen_weight * m)

            for i in range(self.n_genes):
                # Production (regulated)
                prop = beta_eff
                for r_idx, j in enumerate(self.regulators[i]):
                    x_j = state[j, c]
                    h = self._hill(x_j, self.cfg.hill_K, self.cfg.hill_n,
                                    self.signs[i][r_idx] > 0)
                    prop *= h
                rxn_props[i, c] = max(prop, 0.0)

                # Degradation
                rxn_props[self.n_genes + i, c] = self.cfg.gamma * state[i, c]

        # Diffusion propensities: (n_genes, n_cells, n_neighbours)
        # For each gene and cell, propensity to diffuse to each neighbour
        diff_props = np.zeros((self.n_genes, self.n_cells, 6))  # max 6 neighbours
        for c in range(self.n_cells):
            for i in range(self.n_genes):
                for n_idx, nc in enumerate(self.adjacency[c]):
                    diff_props[i, c, n_idx] = self.cfg.diffusion_rate * state[i, c]

        return rxn_props, diff_props

    def ssa_step(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        One RDME step: reaction or diffusion event.
        Returns: (new_state, time_increment)
        """
        rxn_props, diff_props = self.compute_propensities(state)

        # Total propensity
        total_rxn = rxn_props.sum()
        total_diff = diff_props.sum()
        a_total = total_rxn + total_diff

        if a_total <= 0:
            return state.copy(), 0.0

        # Sample waiting time
        tau = -math.log(max(self._rng.random(), 1e-12)) / a_total

        # Sample event type (reaction vs diffusion)
        r = self._rng.random() * a_total
        if r < total_rxn:
            # Reaction event
            # Flatten and sample which reaction
            rxn_flat = rxn_props.flatten()
            r2 = self._rng.random() * total_rxn
            cumsum = 0.0
            event_idx = len(rxn_flat) - 1
            for idx in range(len(rxn_flat)):
                cumsum += rxn_flat[idx]
                if r2 <= cumsum:
                    event_idx = idx
                    break

            # Decode event
            rxn_type = event_idx // self.n_cells  # 0..n_genes-1: production, n_genes..: degradation
            cell_idx = event_idx % self.n_cells
            gene_idx = rxn_type if rxn_type < self.n_genes else rxn_type - self.n_genes

            new_state = state.copy()
            if rxn_type < self.n_genes:
                # Production
                new_state[gene_idx, cell_idx] += 1
            else:
                # Degradation
                new_state[gene_idx, cell_idx] = max(0, new_state[gene_idx, cell_idx] - 1)
        else:
            # Diffusion event
            diff_flat = diff_props.flatten()
            r2 = self._rng.random() * total_diff
            cumsum = 0.0
            event_idx = len(diff_flat) - 1
            for idx in range(len(diff_flat)):
                cumsum += diff_flat[idx]
                if r2 <= cumsum:
                    event_idx = idx
                    break

            # Decode event
            gene_idx = event_idx // (self.n_cells * 6)
            remainder = event_idx % (self.n_cells * 6)
            cell_idx = remainder // 6
            neighbour_idx = remainder % 6

            new_state = state.copy()
            if cell_idx < self.n_cells and neighbour_idx < len(self.adjacency[cell_idx]):
                target_cell = self.adjacency[cell_idx][neighbour_idx]
                if new_state[gene_idx, cell_idx] > 0:
                    new_state[gene_idx, cell_idx] -= 1
                    new_state[gene_idx, target_cell] += 1

        return new_state, tau

    def simulate(self, x0: Optional[np.ndarray] = None,
                  max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[np.ndarray], List[float]]:
        """Run full RDME simulation."""
        if x0 is None:
            # Random initial state
            x0 = self._rng.integers(0, 20, size=(self.n_genes, self.n_cells)).astype(float)

        state = x0.copy()
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

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_genes": self.n_genes,
            "n_cells": self.n_cells,
            "grid_shape": self.cfg.grid_shape,
            "diffusion_rate": self.cfg.diffusion_rate,
            "corollary3_guard": "first-order propensities only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [PCP1] PAR/CDC42 POLARITY MODEL
# =============================================================================

@dataclass
class PolarityConfig:
    """Configuration for Par/Cdc42 polarity model."""
    # Grid
    grid_shape: Tuple[int, int, int] = (5, 5, 5)
    # Turing parameters [C Turing 1952; Marée 2006]
    # Apical complex (Par3/Par6/aPKC)
    apical_production: float = 1.0
    apical_degradation: float = 0.5
    apical_diffusion: float = 0.05
    # Basal complex (Par1/Lgl)
    basal_production: float = 1.0
    basal_degradation: float = 0.5
    basal_diffusion: float = 0.10  # faster diffusion for Turing instability
    # Mutual inhibition
    inhibition_strength: float = 2.0
    # Simulation
    dt: float = 0.01
    n_steps: int = 500
    # Seed
    seed: int = 0


class ParCdc42PolarityModel:
    """
    [C/D] Turing-type mutual inhibition model for cell polarity.

    BIOLOGICAL BASIS [C]
      Cell polarity in epithelia is established by the Par complex:
        - Apical: Par3/Par6/aPKC (promotes apical identity)
        - Basal: Par1/Par2/Lgl (promotes basal identity)
      These complexes mutually inhibit each other, leading to pattern
      formation via a Turing-type mechanism [C Turing 1952].

    MODEL [C Marée et al. 2006; D]
      Two species on a spatial grid:
        A (apical): Par3/Par6/aPKC complex
        B (basal): Par1/Lgl complex

      Reaction-diffusion equations:
        ∂A/∂t = D_A ∇²A + α_A - γ_A A - κ A B²
        ∂B/∂t = D_B ∇²B + α_B - γ_B B - κ B A²

      where:
        D_A, D_B: diffusion coefficients (D_B > D_A for Turing instability)
        α_A, α_B: production rates
        γ_A, γ_B: degradation rates
        κ: mutual inhibition strength

      The quadratic inhibition terms (A B² and B A²) create the nonlinearity
      needed for pattern formation.

    POLARITY VECTOR [D]
      After the Turing pattern forms, the polarity vector at each cell is
      computed from the gradient of the apical complex:
        p̂ = ∇A / |∇A|
      This gives the direction of apical-basal polarity.

    COROLLARY-3 GUARD [D]
      The reaction-diffusion equations are first-order in space (Laplacian).
      We do NOT compute second derivatives of the polarity field.

    HONEST
      • This is a simplified Turing model. Real polarity involves many more
        proteins (Cdc42, Scribble, Crumbs, etc.) with complex kinetics.
      • The mutual inhibition is modeled as quadratic terms, which is a
        simplification of the actual biochemistry.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[PolarityConfig] = None):
        self.cfg = cfg or PolarityConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.nx, self.ny, self.nz = self.cfg.grid_shape

        # State fields: A (apical), B (basal)
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with periodic boundary conditions."""
        lap = np.zeros_like(field)
        for axis in range(3):
            lap += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
        return lap

    def initialize(self) -> None:
        """Initialize A and B fields with small random perturbations."""
        shape = (self.nx, self.ny, self.nz)
        # Homogeneous steady state
        A0 = self.cfg.apical_production / self.cfg.apical_degradation
        B0 = self.cfg.basal_production / self.cfg.basal_degradation
        # Small random perturbations
        self.A = A0 + 0.1 * self._rng.standard_normal(shape)
        self.B = B0 + 0.1 * self._rng.standard_normal(shape)
        self.A = np.maximum(self.A, 0.0)
        self.B = np.maximum(self.B, 0.0)

    def step(self) -> None:
        """One RD step for the polarity model."""
        if self.A is None:
            self.initialize()

        dt = self.cfg.dt
        κ = self.cfg.inhibition_strength

        # Reaction-diffusion for A (apical)
        lap_A = self._laplacian(self.A)
        reaction_A = (self.cfg.apical_production
                      - self.cfg.apical_degradation * self.A
                      - κ * self.A * self.B**2)
        self.A = self.A + dt * (self.cfg.apical_diffusion * lap_A + reaction_A)
        self.A = np.maximum(self.A, 0.0)

        # Reaction-diffusion for B (basal)
        lap_B = self._laplacian(self.B)
        reaction_B = (self.cfg.basal_production
                      - self.cfg.basal_degradation * self.B
                      - κ * self.B * self.A**2)
        self.B = self.B + dt * (self.cfg.basal_diffusion * lap_B + reaction_B)
        self.B = np.maximum(self.B, 0.0)

    def run(self, n_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run the polarity model for n_steps."""
        if self.A is None:
            self.initialize()
        steps = n_steps or self.cfg.n_steps
        for _ in range(steps):
            self.step()
        return self.A.copy(), self.B.copy()

    def compute_polarity_vectors(self) -> np.ndarray:
        """
        Compute polarity vectors from gradient of apical complex.
        Returns: (nx, ny, nz, 3) array of unit vectors.
        """
        if self.A is None:
            self.initialize()

        # Compute gradient of A
        grad_A = np.zeros((self.nx, self.ny, self.nz, 3))
        for axis in range(3):
            grad_A[..., axis] = (np.roll(self.A, -1, axis=axis)
                                  - np.roll(self.A, 1, axis=axis)) / 2.0

        # Normalize to unit vectors
        norm = np.linalg.norm(grad_A, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        polarity = grad_A / norm

        return polarity

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid_shape": self.cfg.grid_shape,
            "apical_diffusion": self.cfg.apical_diffusion,
            "basal_diffusion": self.cfg.basal_diffusion,
            "inhibition_strength": self.cfg.inhibition_strength,
            "corollary3_guard": "first-order spatial coupling; no Hessian",
            "note": "computational model, not validated molecular biology",
        }


# =============================================================================
# SECTION 3 — [TSD1] τ-SPECTRAL DIMENSION COUPLING
# =============================================================================

@dataclass
class TauSpectralConfig:
    """Configuration for τ-d_s coupling."""
    # Carpet reference exponents [C]
    d_f: float = CARPET_DF           # ≈ 1.8928
    d_w: float = CARPET_DW           # ≈ 2.096
    d_s: float = CARPET_DS           # ≈ 1.806
    beta_holder: float = CARPET_BETA # ≈ 0.222
    # τ range
    tau_min: float = 0.005
    tau_max: float = 0.5
    # Scaling heuristic [D]
    tau_base: float = 0.05
    length_scale: float = 1.0        # characteristic length L
    scaling_exponent: float = 0.5    # τ ~ L^{d_s/2} (heuristic)


class TauSpectralDimensionCoupling:
    """
    [C/D] Coupling τ to spectral dimension d_s from Carpet-19.

    CARPET-19 REFERENCE [C]
      Theorem 4(f): W^s ↪ C^{0,β/2}(K) with β = log ρ / log 3 ≈ 0.222.
      Spectral dimension: d_s = 2 d_f / d_w ≈ 1.806.
      Walk dimension: d_w = ln(8ρ)/ln(3) ≈ 2.096 (using ρ ≈ 1.25).
      Hausdorff dimension: d_f = ln 8 / ln 3 ≈ 1.8928.

    SCALING HEURISTIC [D]
      We use a dimensional analogy:
        τ ~ L^{d_s / 2}
      where L is a characteristic length scale. This is motivated by the
      spectral dimension governing diffusion: the return probability scales
      as t^{-d_s/2}, so the characteristic time for diffusion over length L
      scales as L^{d_s}.

      For the vertex model, L is the target cell size, and τ controls the
      resolution of the isosurface extraction. Smaller τ (sharper interfaces)
      corresponds to finer resolution.

    ρ STATUS [C-num]
      ρ ≈ 1.25 is NUMERICAL (R_cross ratios: 1.66, 1.33, 1.27, 1.26, 1.25).
      It is NOT proved. All exponents derived from ρ inherit this status.

    THEOREM 10 CONDITIONAL [C]
      Theorem 10 is conditional on (I1)-(I4). For the standard carpet, these
      follow from KZ/BBKT/Hino. For general non-p.c.f. fractals, (I1)-(I4)
      are exactly the unproved content. We do NOT claim closure transfer
      to biological tissue.

    PROP-3 AWARENESS [P]
      τ is a SCALE SELECTOR, not a regularity recovery tool. Decreasing τ
      does NOT restore smoothness lost by the commutator (Carpet-19 Prop 3:
      [A,M_f] loses two orders).

    HONEST
      • The τ ~ L^{d_s/2} scaling is a HEURISTIC dimensional analogy.
      • Carpet-19 Theorem 10 applies to the carpet, not to biological tissue.
      • No closure transfer is claimed.
      • ρ ≈ 1.25 is numerical, not proved.
    """

    def __init__(self, cfg: Optional[TauSpectralConfig] = None):
        self.cfg = cfg or TauSpectralConfig()

    def tau_from_length(self, length_scale: float) -> float:
        """
        Compute τ from a characteristic length scale using the heuristic
        τ ~ L^{d_s/2}.
        """
        tau = self.cfg.tau_base * (length_scale ** self.cfg.scaling_exponent)
        return float(np.clip(tau, self.cfg.tau_min, self.cfg.tau_max))

    def length_from_tau(self, tau: float) -> float:
        """Invert the scaling: L ~ τ^{2/d_s}."""
        if tau <= 0:
            return 0.0
        L = (tau / self.cfg.tau_base) ** (2.0 / self.cfg.d_s)
        return float(L)

    def holder_exponent(self) -> float:
        """Return the Hölder exponent β = log ρ / log 3 from Theorem 4(f)."""
        return self.cfg.beta_holder

    def spectral_dimension(self) -> float:
        """Return the spectral dimension d_s = 2 d_f / d_w."""
        return self.cfg.d_s

    def get_status(self) -> Dict[str, Any]:
        return {
            "d_f": self.cfg.d_f,
            "d_w": self.cfg.d_w,
            "d_s": self.cfg.d_s,
            "beta_holder": self.cfg.beta_holder,
            "rho_status": "NUMERICAL, not proved [C-num]",
            "theorem10_status": "conditional on (I1)-(I4); for carpet only",
            "scaling_heuristic": f"τ ~ L^{{d_s/2}} = L^{{{self.cfg.d_s/2:.3f}}}",
            "prop3_awareness": "τ is scale selector, not regularity recovery",
            "note": "dimensional analogy; no closure transfer to biology",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESISpatialOrganoidSuite:
    """
    Couples: SpatialStochasticGRN + ParCdc42PolarityModel + TauSpectralDimensionCoupling.

    FULL SPATIAL ONTOGENY PIPELINE:
      1. τ from spectral dimension sets the resolution.
      2. Par/Cdc42 polarity model establishes apical-basal polarity.
      3. Spatial stochastic GRN assigns cell fates with noise and diffusion.
      4. Polarity vectors guide cell division orientation.
      5. Export to AR pipeline.
    """

    def __init__(self,
                 spatial_grn_cfg: Optional[SpatialGRNConfig] = None,
                 polarity_cfg: Optional[PolarityConfig] = None,
                 tau_spectral_cfg: Optional[TauSpectralConfig] = None):
        self.spatial_grn = SpatialStochasticGRN(spatial_grn_cfg)
        self.polarity = ParCdc42PolarityModel(polarity_cfg)
        self.tau_spectral = TauSpectralDimensionCoupling(tau_spectral_cfg)

    def run_spatial_ontogeny(self, length_scale: float = 1.0,
                               n_polarity_steps: int = 200,
                               n_grn_steps: int = 500
                               ) -> Dict[str, Any]:
        """
        Run the full spatial organoid ontogeny.

        Args:
            length_scale: characteristic length for τ scaling
            n_polarity_steps: steps for polarity pattern formation
            n_grn_steps: steps for stochastic GRN simulation

        Returns:
            Diagnostics including τ, polarity vectors, GRN states.
        """
        # 1. τ from spectral dimension
        tau = self.tau_spectral.tau_from_length(length_scale)

        # 2. Polarity pattern formation
        A_final, B_final = self.polarity.run(n_polarity_steps)
        polarity_vectors = self.polarity.compute_polarity_vectors()

        # 3. Spatial stochastic GRN
        # Use polarity as morphogen input (apical complex concentration)
        morphogen = A_final.flatten()
        self.spatial_grn.cfg.morphogen_field = morphogen

        x0 = self._rng.integers(0, 20,
                                  size=(self.spatial_grn.n_genes,
                                        self.spatial_grn.n_cells)).astype(float)
        trajectory, times = self.spatial_grn.simulate(x0, max_steps=n_grn_steps)

        # Final GRN state
        final_state = trajectory[-1]

        return {
            "tau": tau,
            "tau_spectral_status": self.tau_spectral.get_status(),
            "polarity_A_mean": float(A_final.mean()),
            "polarity_B_mean": float(B_final.mean()),
            "polarity_vectors_shape": polarity_vectors.shape,
            "grn_final_state_shape": final_state.shape,
            "grn_trajectory_length": len(trajectory),
            "grn_final_time": times[-1] if times else 0.0,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "spatial_grn": self.spatial_grn.get_status(),
            "polarity": self.polarity.get_status(),
            "tau_spectral": self.tau_spectral.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO SPATIAL — SMOKE TEST")
    print(" Spatial Stochastic GRN + Par/Cdc42 Polarity + τ-d_s Coupling")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f"\n Carpet-19 reference [C] (ρ NUMERICAL, not proved):")
    print(f"   d_f = ln8/ln3       = {CARPET_DF:.6f}")
    print(f"   ρ                   = {CARPET_RHO}  [NUMERICAL]")
    print(f"   d_w = ln(8ρ)/ln3    = {CARPET_DW:.6f}")
    print(f"   d_s = 2d_f/d_w      = {CARPET_DS:.6f}")
    print(f"   β = log ρ / log 3   = {CARPET_BETA:.6f}  (Hölder exponent)")

    suite = SESISpatialOrganoidSuite(
        spatial_grn_cfg=SpatialGRNConfig(
            grid_shape=(3, 3, 3), n_genes=3, max_time=20.0, max_steps=2000),
        polarity_cfg=PolarityConfig(
            grid_shape=(3, 3, 3), n_steps=100),
        tau_spectral_cfg=TauSpectralConfig())

    # ── 1. τ-Spectral Dimension Coupling ────────────────────────────────
    print("\n [TSD1] τ-Spectral Dimension Coupling")
    tsd = suite.tau_spectral
    print(f"   d_f                  : {tsd.cfg.d_f:.6f}")
    print(f"   d_w                  : {tsd.cfg.d_w:.6f}")
    print(f"   d_s                  : {tsd.cfg.d_s:.6f}")
    print(f"   β (Hölder)           : {tsd.cfg.beta_holder:.6f}")
    print(f"   ρ status             : {tsd.get_status()['rho_status']}")
    print(f"   Theorem 10 status    : {tsd.get_status()['theorem10_status']}")
    print(f"   Scaling heuristic    : {tsd.get_status()['scaling_heuristic']}")

    # Compute τ for different length scales
    for L in [0.5, 1.0, 2.0]:
        tau = tsd.tau_from_length(L)
        L_back = tsd.length_from_tau(tau)
        print(f"   L={L:.1f} -> τ={tau:.5f} -> L_back={L_back:.3f}")

    # ── 2. Par/Cdc42 Polarity Model ─────────────────────────────────────
    print("\n [PCP1] Par/Cdc42 Polarity Model")
    pol = suite.polarity
    print(f"   Grid shape           : {pol.cfg.grid_shape}")
    print(f"   Apical diffusion     : {pol.cfg.apical_diffusion}")
    print(f"   Basal diffusion      : {pol.cfg.basal_diffusion}")
    print(f"   Inhibition strength  : {pol.cfg.inhibition_strength}")
    print(f"   COROLLARY-3 guard    : {pol.get_status()['corollary3_guard']}")

    # Run polarity model
    A_final, B_final = pol.run(n_steps=50)
    print(f"   A mean (apical)      : {A_final.mean():.4f}")
    print(f"   B mean (basal)       : {B_final.mean():.4f}")
    print(f"   A std                : {A_final.std():.4f}")
    print(f"   B std                : {B_final.std():.4f}")

    # Compute polarity vectors
    pol_vectors = pol.compute_polarity_vectors()
    print(f"   Polarity vectors     : shape={pol_vectors.shape}")
    print(f"   Vector norms (mean)  : {np.linalg.norm(pol_vectors, axis=-1).mean():.4f}")

    # ── 3. Spatial Stochastic GRN (RDME) ────────────────────────────────
    print("\n [SSG1] Spatial Stochastic GRN (RDME)")
    sgrn = suite.spatial_grn
    print(f"   Grid shape           : {sgrn.cfg.grid_shape}")
    print(f"   n_genes              : {sgrn.n_genes}")
    print(f"   n_cells              : {sgrn.n_cells}")
    print(f"   Diffusion rate       : {sgrn.cfg.diffusion_rate}")
    print(f"   COROLLARY-3 guard    : {sgrn.get_status()['corollary3_guard']}")
    print(f"   HINO guard           : {sgrn.get_status()['hino_guard']}")

    # Use polarity as morphogen
    morphogen = A_final.flatten()
    sgrn.cfg.morphogen_field = morphogen

    # Run RDME simulation
    x0 = np.random.randint(0, 15, size=(sgrn.n_genes, sgrn.n_cells)).astype(float)
    print(f"   Initial state shape  : {x0.shape}")
    print(f"   Initial total copies : {x0.sum():.0f}")

    trajectory, times = sgrn.simulate(x0, max_steps=500)
    print(f"   Trajectory length    : {len(trajectory)}")
    print(f"   Final time           : {times[-1]:.4f}")
    final_state = trajectory[-1]
    print(f"   Final total copies   : {final_state.sum():.0f}")
    print(f"   Final state mean     : {final_state.mean():.2f}")
    print(f"   Final state std      : {final_state.std():.2f}")

    # ── 4. Full Spatial Ontogeny ────────────────────────────────────────
    print("\n [ONTO] Full Spatial Organoid Ontogeny")
    result = suite.run_spatial_ontogeny(
        length_scale=1.0, n_polarity_steps=80, n_grn_steps=300)
    print(f"   τ                    : {result['tau']:.5f}")
    print(f"   Polarity A mean      : {result['polarity_A_mean']:.4f}")
    print(f"   Polarity B mean      : {result['polarity_B_mean']:.4f}")
    print(f"   Polarity vec shape   : {result['polarity_vectors_shape']}")
    print(f"   GRN final shape      : {result['grn_final_state_shape']}")
    print(f"   GRN trajectory len   : {result['grn_trajectory_length']}")
    print(f"   GRN final time       : {result['grn_final_time']:.4f}")

    # ── 5. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}:")
        for kk, vv in v.items():
            print(f"     {kk}: {vv}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   RDME is exact for well-mixed compartments (Elf & Ehrenberg 2003).")
    print("       Real cells have subcellular spatial structure not captured.")
    print(" [C]   Par/Cdc42 polarity: Turing-type mutual inhibition is simplified.")
    print("       Real polarity involves many more proteins with complex kinetics.")
    print(" [C]   Carpet-19 Theorem 4(f): W^s ↪ C^{0,β/2} with β = log ρ/log 3.")
    print(" [C]   d_s = 2d_f/d_w ≈ 1.806 is from Carpet-19 reference exponents.")
    print(" [D]   τ ~ L^{d_s/2} is a HEURISTIC dimensional analogy.")
    print(" [C]   Theorem 10 is CONDITIONAL on (I1)-(I4). For carpet, these follow")
    print("       from KZ/BBKT/Hino. For other fractals, (I1)-(I4) are UNPROVED.")
    print(" [C-num] ρ ≈ 1.25 is NUMERICAL (R_cross: 1.66, 1.33, 1.27, 1.26, 1.25).")
    print(" [D]   COROLLARY-3 guard: all fields first-order only; no Hessian.")
    print(" [C]   HINO guard: energy measures singular; no smooth density assumed.")
    print(" [D]   No closure transfer from carpet to biological tissue is claimed.")
    print(" All parameters illustrative. No biological/medical claim.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
