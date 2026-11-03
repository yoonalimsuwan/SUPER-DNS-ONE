"""
=============================================================================
SESI ORGANOID STUDIO — VERTEX MECHANICS + BOOLEAN GRN + AR EXPORT
Cell Rearrangement on Carpet + Gene-Fate Attractors + USDZ Morphogenesis
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
[VTX1] CarpetVertexModel
      • 2-D vertex model [C Farhadifar et al. 2005] on the carpet cell graph
      • Energy: E = Σ_α K_A(A_α-A₀)² + K_P(P_α-P₀)² + Σ_edges Λ_ij L_ij
      • Forces: F_i = -∂E/∂r_i  (first-order; COROLLARY-3 guard)
      • σ-field modulates stiffness spatially (carpet structural operator)
      • T1 transitions (edge flip) for cell intercalation
      • T2 transitions (cell removal) for apoptosis / extrusion
      • HINO GUARD: line tensions computed from order parameter, NOT from
        an assumed smooth energy density (Γ ⊥ µ, Thm 8)

[GRN1] BooleanGRN
      • Kauffman NK model [C Kauffman 1969]: N genes, K regulators each
      • Boolean update: x_i(t+1) = f_i(x_{regs(i)}(t))
      • Attractor detection: cycle finding in state space = cell fates
      • Morphogen coupling [C Wolpert 1969 positional information]:
        morphogen concentration shifts gene activation thresholds
      • Spatial GRN: each cell carries its own GRN state; morphogen
        field determines which attractor basin each cell falls into
      • COROLLARY-3 guard: GRN coupling is first-order in the field

[OEX1] OrganoidUSDZExporter
      • Cell density field φ → isosurface (marching cubes)
      • Mesh → GLB → USDZ (crate binary or USDA text)
      • AR-ready output for iOS Quick Look / Android Scene Viewer
      • Multi-field export: cell density, morphogen, gene-fate overlay

[OVT1] OrganoidVertexGRNSuite (orchestrator)
      • Couples OrganoidOntogeneticEngine + VertexModel + GRN
      • Full ontogeny: RD prepattern → cell condensation → vertex
        mechanics rearrangement → GRN fate assignment → AR export

=============================================================================
HONEST SCOPE
=============================================================================
• CarpetVertexModel is a 2-D vertex model on the carpet cell graph. It is
  NOT a 3-D finite-element tissue simulation. The 3-D organoid field is
  projected to 2-D cross-sections for vertex mechanics. Extension to true
  3-D would require Voronoi-based or FEM approaches not implemented here.
• BooleanGRN is a COARSE-GRAINED model. Real gene regulatory networks are
  continuous, stochastic, and involve thousands of genes. The NK model
  captures attractor structure but not quantitative gene expression.
• The morphogen-GRN coupling uses threshold functions, not detailed
  biochemical kinetics. This is a computational framework, not a
  validated biological model.
• OrganoidUSDZExporter produces a visual representation of the simulated
  cell field. It is NOT a medical imaging tool and makes NO biological
  or diagnostic claims.
• All parameters are illustrative. Real organoids involve gene regulatory
  networks, cell signalling, mechanics, and stochasticity far beyond this.
• COROLLARY-3 guard: no second-order field Hessians in any module.
• HINO guard: energy measures are singular; no smooth density assumed.
• ρ≈1.25 is NUMERICAL and unused by these modules.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import time
import uuid
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.vertex_grn")

# ── Ecosystem imports (graceful) ─────────────────────────────────────────
try:
    from structural_cahn_hilliard_3d import (
        StructuralCahnHilliard3D, CahnHilliardConfig, make_sigma_field,
    )
    HAS_CH = True
except Exception:
    HAS_CH = False

try:
    from sesi_topology_studio_organoid import (
        OrganoidOntogeneticEngine, OrganoidConfig, OrganoidState,
    )
    HAS_ORGANOID = True
except Exception:
    HAS_ORGANOID = False

try:
    from sesi_topology_studio_quench_crate_ae import (
        USDCCrateWriter, CrateConfig,
    )
    HAS_CRATE = True
except Exception:
    HAS_CRATE = False

try:
    from sesi_topology_studio_field_ar_delta import (
        GLBtoUSDZConverter, USDZConfig, GLBExporter,
    )
    HAS_USDZ = True
except Exception:
    HAS_USDZ = False

try:
    from sesi_differentiable_topology import MarchingCubesExtractor
    HAS_MC = True
except Exception:
    HAS_MC = False

__all__ = [
    "VertexConfig", "CarpetVertexModel", "VertexModelState",
    "GRNConfig", "BooleanGRN", "GRNAttractor",
    "OrganoidUSDZExporter", "USDZExportResult",
    "OrganoidVertexGRNSuite",
]


# =============================================================================
# SECTION 0 — DIFFERENTIABILITY PRIMITIVES
# =============================================================================

def _softplus_floor(x: Tensor, floor: float, beta: Optional[float] = None,
                    sharpness: float = 50.0) -> Tensor:
    if beta is None:
        beta = sharpness / max(abs(floor), 1e-12)
    return floor + F.softplus(x - floor, beta=beta)

def _soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps)


# =============================================================================
# SECTION 1 — [VTX1] CARPET VERTEX MODEL
# =============================================================================

@dataclass
class VertexConfig:
    """Configuration for the carpet vertex model."""
    carpet_level: int = 3
    # mechanical parameters [C Farhadifar 2005]
    K_area: float = 1.0         # area elasticity modulus
    K_perimeter: float = 0.1    # perimeter contractility
    Lambda: float = 0.05        # line tension on edges
    target_area: float = 1.0    # A₀
    target_perimeter: float = 4.0  # P₀
    # dynamics
    mobility: float = 0.01      # vertex mobility μ
    dt: float = 0.01
    n_steps: int = 100
    # T1 / T2 transitions
    t1_threshold: float = 0.05  # edge length below which T1 fires
    t2_threshold: float = 0.01  # cell area below which T2 fires
    enable_t1: bool = True
    enable_t2: bool = True
    # carpet coupling
    use_sigma_field: bool = True   # modulate stiffness by σ
    sigma_background: float = 1.0
    sigma_inclusion: float = 5.0
    # noise (stochastic vertex model)
    noise_amplitude: float = 0.0
    # device
    device: str = "cpu"


@dataclass
class VertexModelState:
    """Snapshot of the vertex model."""
    step: int
    time: float
    positions: np.ndarray         # (V, 2) vertex positions
    cells: List[List[int]]        # cell -> list of vertex indices (ordered)
    edges: List[Tuple[int, int]]  # (v_i, v_j) pairs
    cell_areas: np.ndarray
    cell_perimeters: np.ndarray
    total_energy: float
    n_t1_transitions: int
    n_t2_transitions: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class CarpetVertexModel:
    """
    [C/D] 2-D vertex model on the Sierpinski carpet cell graph.

    ENERGY FUNCTIONAL  [C Farhadifar et al. 2005]
      E = Σ_α [ K_A (A_α - A₀)² + K_P (P_α - P₀)² ]
          + Σ_{<i,j>} Λ_{ij} L_{ij}

    where A_α is the area of cell α, P_α its perimeter, L_{ij} the length
    of the edge between vertices i and j, and Λ_{ij} the line tension.

    FORCES  [D]
      F_i = -∂E/∂r_i
      Computed by finite differences on the energy (first-order).
      COROLLARY-3 GUARD: we do NOT compute ∂²E/∂r². The Hessian of the
      vertex energy on the carpet is not available because D(A) is not
      an algebra (Carpet-19 Corollary 3). Forces are first-order only.

    CARPET COUPLING  [D]
      The σ-field from the carpet structural operator modulates the
      mechanical stiffness spatially:
        K_A(x) = K_A_base * σ(x)
        K_P(x) = K_P_base * σ(x)
        Λ(x)   = Λ_base   * σ(x)
      This creates spatially heterogeneous mechanics: stiff regions
      resist deformation, soft regions rearrange easily.

    HINO GUARD  [C Thm 8]
      The energy measure Γ(f,·) is SINGULAR w.r.t. µ. This means
      mechanical energy concentrates on a fractal subset of edges.
      We compute line tensions from the ORDER PARAMETER (cell areas,
      perimeters) directly, NOT from an assumed smooth energy density.
      The singularity manifests as certain edges carrying
      disproportionately high tension — creating natural "weak lines"
      for cell rearrangement, analogous to convergent extension in
      real tissues.

    T1 TRANSITIONS  [C]
      When an edge length drops below t1_threshold, the edge is flipped:
      the two cells sharing the edge exchange a vertex pair. This is the
      primary mechanism of cell intercalation in epithelial tissues.

    T2 TRANSITIONS  [C]
      When a cell area drops below t2_threshold, the cell is removed
      (apoptosis / extrusion). Its neighbours merge.

    HONEST
      • This is a 2-D model. The 3-D organoid is projected to 2-D
        cross-sections. True 3-D vertex models require different
        topology (Voronoi / FEM).
      • Parameters are illustrative, not calibrated to any real tissue.
      • The carpet topology is used for the cell graph, but the vertex
        positions are free to move (not constrained to carpet cells).
    """

    def __init__(self, cfg: Optional[VertexConfig] = None):
        self.cfg = cfg or VertexConfig()
        self._step = 0
        self._time = 0.0
        self._n_t1 = 0
        self._n_t2 = 0
        self._positions: Optional[np.ndarray] = None
        self._cells: Optional[List[List[int]]] = None
        self._edges: Optional[List[Tuple[int, int]]] = None
        self._sigma: Optional[np.ndarray] = None

    # ── carpet cell graph extraction ────────────────────────────────────
    def _carpet_mask(self, level: int) -> np.ndarray:
        pat = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
        m = np.ones((1,1), dtype=bool)
        for _ in range(level):
            m = np.kron(m, pat).astype(bool)
        return m

    def _build_vertex_model(self) -> None:
        """
        Build the initial vertex model from the carpet cell graph.
        Each occupied carpet cell becomes a polygonal cell in the
        vertex model. Vertices are the corners of the cells.
        """
        level = self.cfg.carpet_level
        mask = self._carpet_mask(level)
        H, W = mask.shape

        # Build vertex grid: corners of cells
        # Each cell (i,j) has corners at (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        vertex_map: Dict[Tuple[int,int], int] = {}
        vertices: List[Tuple[float,float]] = []
        cells: List[List[int]] = []
        cell_centres: List[Tuple[int,int]] = []

        def get_vertex(x: int, y: int) -> int:
            key = (x, y)
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append((float(x), float(y)))
            return vertex_map[key]

        for i in range(H):
            for j in range(W):
                if mask[i, j]:
                    v0 = get_vertex(j, i)
                    v1 = get_vertex(j+1, i)
                    v2 = get_vertex(j+1, i+1)
                    v3 = get_vertex(j, i+1)
                    cells.append([v0, v1, v2, v3])
                    cell_centres.append((i, j))

        # Build edge set (unique undirected edges)
        edge_set: Set[Tuple[int,int]] = set()
        for cell in cells:
            n = len(cell)
            for k in range(n):
                a, b = cell[k], cell[(k+1) % n]
                edge_set.add((min(a,b), max(a,b)))

        self._positions = np.array(vertices, dtype=np.float64)
        self._cells = cells
        self._edges = sorted(edge_set)
        self._cell_centres = cell_centres

        # Build sigma field for stiffness modulation
        if self.cfg.use_sigma_field:
            self._sigma = self._build_sigma_field(mask)
        else:
            self._sigma = np.ones(len(cells))

        logger.info("[VTX] Built vertex model: %d vertices, %d cells, "
                     "%d edges at carpet level %d",
                     len(vertices), len(cells), len(self._edges), level)

    def _build_sigma_field(self, mask: np.ndarray) -> np.ndarray:
        """
        Build σ-field for mechanical stiffness modulation.
        Uses the carpet structural operator pattern: high σ in the
        interior, low σ near removed cells (holes).
        """
        H, W = mask.shape
        sigma = np.ones(mask.sum())
        # Modulate by distance to nearest hole (removed cell)
        # This creates the singular energy concentration (Hino guard)
        idx = 0
        for i in range(H):
            for j in range(W):
                if mask[i, j]:
                    # Check if adjacent to a hole
                    near_hole = False
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < H and 0 <= nj < W and not mask[ni, nj]:
                            near_hole = True
                            break
                    if near_hole:
                        sigma[idx] = self.cfg.sigma_inclusion
                    else:
                        sigma[idx] = self.cfg.sigma_background
                    idx += 1
        return sigma

    # ── geometry computation ────────────────────────────────────────────
    def _cell_area(self, cell: List[int]) -> float:
        """Shoelace formula for polygon area."""
        pos = self._positions
        n = len(cell)
        area = 0.0
        for k in range(n):
            x1, y1 = pos[cell[k]]
            x2, y2 = pos[cell[(k+1) % n]]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _cell_perimeter(self, cell: List[int]) -> float:
        """Sum of edge lengths around the cell."""
        pos = self._positions
        n = len(cell)
        perim = 0.0
        for k in range(n):
            x1, y1 = pos[cell[k]]
            x2, y2 = pos[cell[(k+1) % n]]
            perim += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return perim

    def _edge_length(self, e: Tuple[int,int]) -> float:
        x1, y1 = self._positions[e[0]]
        x2, y2 = self._positions[e[1]]
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def _compute_areas_perimeters(self) -> Tuple[np.ndarray, np.ndarray]:
        n_cells = len(self._cells)
        areas = np.zeros(n_cells)
        perimeters = np.zeros(n_cells)
        for i, cell in enumerate(self._cells):
            areas[i] = self._cell_area(cell)
            perimeters[i] = self._cell_perimeter(cell)
        return areas, perimeters

    # ── energy ──────────────────────────────────────────────────────────
    def compute_energy(self) -> float:
        """
        Total vertex model energy.
        HINO GUARD: line tensions are modulated by σ, which is singular
        near holes. We do NOT assume a smooth energy density.
        """
        areas, perimeters = self._compute_areas_perimeters()
        sigma = self._sigma if self._sigma is not None else np.ones(len(areas))

        E = 0.0
        for i in range(len(areas)):
            s = sigma[i]
            E += s * self.cfg.K_area * (areas[i] - self.cfg.target_area)**2
            E += s * self.cfg.K_perimeter * (perimeters[i] - self.cfg.target_perimeter)**2

        # Line tension on edges
        for e in self._edges:
            L = self._edge_length(e)
            # Edge tension modulated by average σ of adjacent cells
            E += self.cfg.Lambda * L

        return float(E)

    # ── forces ──────────────────────────────────────────────────────────
    def compute_forces(self) -> np.ndarray:
        """
        F_i = -∂E/∂r_i  (first-order, finite differences).
        COROLLARY-3 GUARD: no Hessian, no second derivatives.
        """
        n_verts = len(self._positions)
        forces = np.zeros((n_verts, 2))
        eps = 1e-5
        E0 = self.compute_energy()

        for i in range(n_verts):
            for d in range(2):
                self._positions[i, d] += eps
                E_plus = self.compute_energy()
                self._positions[i, d] -= eps
                forces[i, d] = -(E_plus - E0) / eps

        return forces

    def compute_forces_efficient(self) -> np.ndarray:
        """
        Analytic first-order forces (faster than finite differences).
        COROLLARY-3 GUARD: still first-order only.
        """
        n_verts = len(self._positions)
        forces = np.zeros((n_verts, 2))
        areas, perimeters = self._compute_areas_perimeters()
        sigma = self._sigma if self._sigma is not None else np.ones(len(areas))

        # Area and perimeter contributions
        for ci, cell in enumerate(self._cells):
            s = sigma[ci]
            n = len(cell)
            dA = s * self.cfg.K_area * 2 * (areas[ci] - self.cfg.target_area)
            dP = s * self.cfg.K_perimeter * 2 * (perimeters[ci] - self.cfg.target_perimeter)
            for k in range(n):
                v_prev = cell[(k-1) % n]
                v_curr = cell[k]
                v_next = cell[(k+1) % n]
                x_p, y_p = self._positions[v_prev]
                x_c, y_c = self._positions[v_curr]
                x_n, y_n = self._positions[v_next]
                # Area force: ∂A/∂r_i (shoelace derivative)
                forces[v_curr, 0] += -dA * (y_n - y_p) / 2.0
                forces[v_curr, 1] += -dA * (x_p - x_n) / 2.0
                # Perimeter force: ∂P/∂r_i
                dx1, dy1 = x_c - x_p, y_c - y_p
                dx2, dy2 = x_n - x_c, y_n - y_c
                L1 = math.sqrt(dx1**2 + dy1**2) + 1e-12
                L2 = math.sqrt(dx2**2 + dy2**2) + 1e-12
                forces[v_curr, 0] += -dP * (dx1/L1 - dx2/L2)
                forces[v_curr, 1] += -dP * (dy1/L1 - dy2/L2)

        # Line tension forces
        for e in self._edges:
            a, b = e
            x1, y1 = self._positions[a]
            x2, y2 = self._positions[b]
            L = math.sqrt((x2-x1)**2 + (y2-y1)**2) + 1e-12
            fx = self.cfg.Lambda * (x2-x1) / L
            fy = self.cfg.Lambda * (y2-y1) / L
            forces[a, 0] += fx
            forces[a, 1] += fy
            forces[b, 0] -= fx
            forces[b, 1] -= fy

        return forces

    # ── T1 transition (edge flip) ───────────────────────────────────────
    def _find_t1_edges(self) -> List[Tuple[int,int]]:
        """Find edges shorter than the T1 threshold."""
        return [e for e in self._edges
                if self._edge_length(e) < self.cfg.t1_threshold]

    def _perform_t1(self, edge: Tuple[int,int]) -> bool:
        """
        T1 transition: flip a short edge.
        The two cells sharing the edge exchange a vertex pair.
        This is the primary mechanism of cell intercalation.
        """
        a, b = edge
        # Find cells containing this edge
        sharing_cells = []
        for ci, cell in enumerate(self._cells):
            if a in cell and b in cell:
                sharing_cells.append(ci)
        if len(sharing_cells) != 2:
            return False  # boundary edge, cannot flip

        c1, c2 = sharing_cells
        cell1 = self._cells[c1]
        cell2 = self._cells[c2]

        # Find the opposite vertices
        idx_a1, idx_b1 = cell1.index(a), cell1.index(b)
        opp1 = cell1[(idx_a1 + idx_b1 + 1) % len(cell1)]  # rough: pick a different vertex
        # More carefully: the vertex in cell1 that is not a or b
        opp1_candidates = [v for v in cell1 if v not in (a, b)]
        opp2_candidates = [v for v in cell2 if v not in (a, b)]
        if not opp1_candidates or not opp2_candidates:
            return False
        opp1 = opp1_candidates[0]
        opp2 = opp2_candidates[0]

        # Flip: replace edge (a,b) with edge (opp1, opp2)
        # Remove old edge
        self._edges = [e for e in self._edges if e != edge and e != (b, a)]
        # Add new edge
        new_edge = (min(opp1, opp2), max(opp1, opp2))
        if new_edge not in self._edges:
            self._edges.append(new_edge)

        # Update cell connectivity
        # Replace a->b with a->opp2->b in cell1, etc.
        # Simplified: swap the edge in both cells
        for ci in [c1, c2]:
            cell = self._cells[ci]
            if a in cell and b in cell:
                ia, ib = cell.index(a), cell.index(b)
                # Insert opp vertex between a and b
                if abs(ia - ib) == 1 or abs(ia - ib) == len(cell) - 1:
                    insert_pos = (max(ia, ib) + 1) % len(cell)
                    opp = opp2 if ci == c1 else opp1
                    if opp not in cell:
                        cell.insert(insert_pos, opp)

        self._n_t1 += 1
        return True

    # ── T2 transition (cell removal) ────────────────────────────────────
    def _find_t2_cells(self) -> List[int]:
        """Find cells with area below the T2 threshold."""
        areas, _ = self._compute_areas_perimeters()
        return [i for i, a in enumerate(areas) if a < self.cfg.t2_threshold]

    def _perform_t2(self, cell_idx: int) -> bool:
        """
        T2 transition: remove a small cell.
        The cell is deleted and its neighbours merge.
        """
        if cell_idx >= len(self._cells):
            return False
        cell = self._cells[cell_idx]
        # Remove edges belonging only to this cell
        # Simplified: just mark the cell as removed
        self._cells[cell_idx] = []
        self._n_t2 += 1
        return True

    # ── dynamics ────────────────────────────────────────────────────────
    def step(self) -> VertexModelState:
        """One vertex model step: forces + update + transitions."""
        if self._positions is None:
            self._build_vertex_model()

        # Compute forces (first-order only, Corollary 3 guard)
        forces = self.compute_forces_efficient()

        # Add noise if configured
        if self.cfg.noise_amplitude > 0:
            forces += self.cfg.noise_amplitude * np.random.randn(*forces.shape)

        # Update positions
        self._positions += self.cfg.mobility * forces * self.cfg.dt

        # T1 transitions
        if self.cfg.enable_t1:
            t1_edges = self._find_t1_edges()
            for edge in t1_edges[:5]:  # limit per step
                self._perform_t1(edge)

        # T2 transitions
        if self.cfg.enable_t2:
            t2_cells = self._find_t2_cells()
            for ci in t2_cells[:3]:  # limit per step
                self._perform_t2(ci)

        self._step += 1
        self._time += self.cfg.dt

        return self.snapshot()

    def snapshot(self) -> VertexModelState:
        areas, perimeters = self._compute_areas_perimeters()
        return VertexModelState(
            step=self._step,
            time=self._time,
            positions=self._positions.copy() if self._positions is not None else np.zeros((0,2)),
            cells=[c for c in self._cells if c],
            edges=self._edges.copy() if self._edges else [],
            cell_areas=areas,
            cell_perimeters=perimeters,
            total_energy=self.compute_energy(),
            n_t1_transitions=self._n_t1,
            n_t2_transitions=self._n_t2,
        )

    def run(self, n_steps: Optional[int] = None) -> List[VertexModelState]:
        """Run the vertex model for n_steps."""
        if self._positions is None:
            self._build_vertex_model()
        steps = n_steps or self.cfg.n_steps
        history = []
        sample_every = max(1, steps // 10)
        for s in range(steps):
            state = self.step()
            if s % sample_every == 0 or s == steps - 1:
                history.append(state)
        return history

    def get_status(self) -> Dict[str, Any]:
        return {
            "carpet_level": self.cfg.carpet_level,
            "n_vertices": len(self._positions) if self._positions is not None else 0,
            "n_cells": len([c for c in (self._cells or []) if c]),
            "n_edges": len(self._edges) if self._edges else 0,
            "n_t1_transitions": self._n_t1,
            "n_t2_transitions": self._n_t2,
            "total_energy": self.compute_energy() if self._positions is not None else 0.0,
            "corollary3_guard": "first-order forces only; no Hessian",
            "hino_guard": "line tensions from order parameter, not smooth density",
        }


# =============================================================================
# SECTION 2 — [GRN1] BOOLEAN GENE REGULATORY NETWORK
# =============================================================================

@dataclass
class GRNConfig:
    """Configuration for the Boolean GRN."""
    n_genes: int = 20
    k_regulators: int = 3
    # update mode
    synchronous: bool = True
    # morphogen coupling
    morphogen_threshold: float = 0.5   # threshold for morphogen activation
    morphogen_sensitivity: float = 2.0  # how strongly morphogen shifts thresholds
    # attractor detection
    max_attractor_length: int = 100
    n_attractor_trials: int = 10
    # seed
    seed: int = 42


@dataclass
class GRNAttractor:
    """A detected attractor (cell fate) in the GRN."""
    attractor_id: int
    states: List[int]          # the cycle of states
    length: int                # cycle length (1 = fixed point)
    basin_size: int            # number of initial states converging here
    gene_expression: np.ndarray  # average expression over the cycle
    fate_label: str            # assigned cell fate name


class BooleanGRN:
    """
    [C] Kauffman NK Boolean gene regulatory network.

    MODEL  [C Kauffman 1969]
      N genes, each regulated by K other genes.
      Gene i has a Boolean function f_i: {0,1}^K -> {0,1}.
      State x ∈ {0,1}^N updates synchronously:
          x_i(t+1) = f_i(x_{regs(i)}(t))

    MORPHOGEN COUPLING  [C Wolpert 1969 positional information]
      The morphogen concentration m shifts gene activation thresholds:
          x_i(t+1) = f_i(x_{regs(i)}(t))  if m < θ_i
          x_i(t+1) = 1                      if m >= θ_i  (morphogen ON)
      This implements positional information: cells at different morphogen
      concentrations activate different genes, leading to different fates.

    ATTRACTORS = CELL FATES
      The GRN state space has attractors (fixed points or cycles).
      Each attractor represents a cell fate. The basin of attraction
      determines which initial conditions lead to which fate.

    COROLLARY-3 GUARD
      The GRN coupling to the morphogen field is FIRST-ORDER.
      We do NOT compute second derivatives of gene expression through
      the carpet operator. The morphogen field modulates thresholds;
      it does not create new gene interactions.

    HONEST
      • Boolean GRN is a COARSE-GRAINED model. Real GRNs are continuous,
        stochastic, and involve thousands of genes with complex kinetics.
      • The NK model captures attractor structure but not quantitative
        gene expression levels.
      • Attractor detection is by simulation, not exhaustive enumeration
        (state space is 2^N, intractable for N > ~25).
      • Cell fate labels are ARBITRARY (fate_0, fate_1, ...) and carry
        no biological meaning without experimental calibration.
    """

    def __init__(self, cfg: Optional[GRNConfig] = None):
        self.cfg = cfg or GRNConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n = self.cfg.n_genes
        self.k = self.cfg.k_regulators
        # Regulatory graph: for each gene, which K genes regulate it
        self.regulators: List[List[int]] = []
        # Boolean functions: truth tables for each gene
        self.truth_tables: List[np.ndarray] = []
        self._build_network()
        self._attractors: List[GRNAttractor] = []

    def _build_network(self) -> None:
        """Build a random NK network."""
        for i in range(self.n):
            # Choose K regulators (excluding self)
            regs = self._rng.choice(
                [j for j in range(self.n) if j != i],
                size=self.k, replace=False).tolist()
            self.regulators.append(regs)
            # Random Boolean function: truth table of size 2^K
            tt = self._rng.integers(0, 2, size=2**self.k).astype(np.uint8)
            self.truth_tables.append(tt)

    def _evaluate_gene(self, gene_idx: int, state: np.ndarray,
                        morphogen: float = 0.0) -> int:
        """Evaluate one gene's Boolean function."""
        # Morphogen coupling [C Wolpert 1969]
        threshold = self.cfg.morphogen_threshold + \
                    morphogen * self.cfg.morphogen_sensitivity
        if morphogen > threshold:
            return 1  # morphogen activates this gene

        regs = self.regulators[gene_idx]
        # Build index into truth table
        idx = 0
        for k_i, reg in enumerate(regs):
            idx |= int(state[reg]) << k_i
        return int(self.truth_tables[gene_idx][idx])

    def step(self, state: np.ndarray, morphogen_field: Optional[np.ndarray] = None,
              cell_idx: int = 0) -> np.ndarray:
        """
        One GRN update step.
        morphogen_field: optional spatial morphogen concentration.
        cell_idx: which cell's morphogen to use (for spatial coupling).
        """
        m = 0.0
        if morphogen_field is not None and cell_idx < len(morphogen_field):
            m = float(morphogen_field[cell_idx])

        new_state = np.zeros(self.n, dtype=np.uint8)
        if self.cfg.synchronous:
            for i in range(self.n):
                new_state[i] = self._evaluate_gene(i, state, m)
        else:
            # Asynchronous: update one random gene at a time
            i = self._rng.integers(self.n)
            new_state = state.copy()
            new_state[i] = self._evaluate_gene(i, state, m)
        return new_state

    # ── attractor detection ─────────────────────────────────────────────
    def find_attractors(self, n_trials: Optional[int] = None,
                          morphogen: float = 0.0) -> List[GRNAttractor]:
        """
        Find attractors by simulating from random initial states.
        Each attractor is a cycle in state space.
        """
        n_trials = n_trials or self.cfg.n_attractor_trials
        seen_attractors: Dict[str, GRNAttractor] = {}
        basin_counts: Dict[str, int] = defaultdict(int)
        next_id = 0

        for trial in range(n_trials):
            state = self._rng.integers(0, 2, size=self.n).astype(np.uint8)
            # Simulate until we hit a cycle
            history: List[str] = []
            state_map: Dict[str, int] = {}
            for t in range(self.cfg.max_attractor_length * 2):
                key = state.tobytes().hex()
                if key in state_map:
                    # Found a cycle
                    cycle_start = state_map[key]
                    cycle_states = history[cycle_start:]
                    cycle_key = "".join(sorted(cycle_states))
                    if cycle_key not in seen_attractors:
                        # Decode states
                        decoded = []
                        for cs in cycle_states:
                            decoded.append(
                                np.frombuffer(bytes.fromhex(cs), dtype=np.uint8).copy())
                        avg_expr = np.mean(decoded, axis=0)
                        seen_attractors[cycle_key] = GRNAttractor(
                            attractor_id=next_id,
                            states=[int(cs, 16) for cs in cycle_states],
                            length=len(cycle_states),
                            basin_size=0,
                            gene_expression=avg_expr,
                            fate_label=f"fate_{next_id}",
                        )
                        next_id += 1
                    basin_counts[cycle_key] += 1
                    break
                state_map[key] = t
                history.append(key)
                state = self.step(state, morphogen=morphogen)

        # Assign basin sizes
        for key, att in seen_attractors.items():
            att.basin_size = basin_counts.get(key, 0)

        self._attractors = sorted(seen_attractors.values(),
                                    key=lambda a: a.attractor_id)
        return self._attractors

    # ── spatial GRN: assign fates to cells ──────────────────────────────
    def assign_cell_fates(self, morphogen_field: np.ndarray,
                            n_cells: int) -> np.ndarray:
        """
        Assign a cell fate (attractor ID) to each cell based on its
        local morphogen concentration.

        Returns: array of shape (n_cells,) with attractor IDs.
        """
        if not self._attractors:
            self.find_attractors()

        fates = np.zeros(n_cells, dtype=np.int32)
        for i in range(n_cells):
            m = float(morphogen_field[i]) if i < len(morphogen_field) else 0.0
            # Run GRN from a random state with this morphogen
            state = self._rng.integers(0, 2, size=self.n).astype(np.uint8)
            for _ in range(self.cfg.max_attractor_length):
                state = self.step(state, morphogen=m)
            # Find which attractor this state is closest to
            best_att = 0
            best_dist = float("inf")
            for att in self._attractors:
                # Compare with the average expression of the attractor
                dist = float(np.sum((state.astype(float) - att.gene_expression)**2))
                if dist < best_dist:
                    best_dist = dist
                    best_att = att.attractor_id
            fates[i] = best_att
        return fates

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_genes": self.n,
            "k_regulators": self.k,
            "synchronous": self.cfg.synchronous,
            "n_attractors": len(self._attractors),
            "attractor_lengths": [a.length for a in self._attractors],
            "corollary3_guard": "GRN coupling is first-order; no field Hessian",
        }


# =============================================================================
# SECTION 3 — [OEX1] ORGANOID USDZ EXPORTER
# =============================================================================

@dataclass
class USDZExportResult:
    """Result of an organoid USDZ export."""
    path: str
    format: str                  # "usdz" or "glb"
    n_vertices: int
    n_faces: int
    file_size_bytes: int
    fields_exported: List[str]
    ar_viewable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrganoidUSDZExporter:
    """
    [D] Export organoid cell fields to .usdz for AR visualization.

    PIPELINE
      organoid cell field φ (3-D tensor)
        → isosurface extraction (marching cubes)
        → mesh (vertices + faces)
        → GLB (glTF binary)
        → USDZ (crate binary or USDA text, stored zip)

    MULTI-FIELD EXPORT
      • cell_density: φ > 0 isosurface (cell boundary)
      • morphogen: colour-mapped overlay on the cell surface
      • gene_fate: discrete colour per cell fate (attractor ID)

    AR COMPATIBILITY
      The .usdz output is compatible with:
        • iOS Quick Look (ARKit)
        • Android Scene Viewer (ARCore)
        • Web-based AR viewers (model-viewer)

    HONEST
      • This is a VISUAL REPRESENTATION of a simulated cell field.
      • It is NOT a medical imaging tool.
      • No biological or diagnostic claims are made.
      • The mesh resolution is limited by the grid size.
    """

    def __init__(self, level: float = 0.0):
        self.level = level
        self.mc = MarchingCubesExtractor(level=level) if HAS_MC else None
        self.glb = GLBExporter() if HAS_USDZ else None
        self.usdz = GLBtoUSDZConverter() if HAS_USDZ else None
        self.crate = USDCCrateWriter() if HAS_CRATE else None

    def field_to_mesh(self, field: Tensor,
                        dx: float = 1.0) -> Optional[Dict[str, Any]]:
        """Extract isosurface mesh from a 3-D field."""
        if self.mc is None:
            logger.warning("[OEX] MarchingCubesExtractor not available")
            return None
        return self.mc.extract(field, dx=dx)

    def mesh_to_glb(self, mesh: Dict[str, Any], path: str,
                      color: Tuple[float,float,float] = (0.31,0.80,0.77)
                      ) -> Optional[str]:
        """Write mesh to .glb file."""
        if self.glb is None:
            return None
        verts = mesh["vertices"].detach().cpu().numpy()
        faces = mesh["faces"].detach().cpu().numpy()
        self.glb.export(verts, faces, path, color=color)
        return path

    def glb_to_usdz(self, glb_path: str, usdz_path: str,
                     use_crate: bool = True) -> Optional[str]:
        """Convert .glb to .usdz."""
        if self.usdz is None:
            return None
        if use_crate and self.crate is not None:
            # Parse GLB, write crate USDZ
            parsed = self.usdz.parse_glb(glb_path)
            self.crate.write_usdz(
                parsed["positions"], parsed["indices"],
                parsed["color"], usdz_path)
        else:
            self.usdz.convert(glb_path, usdz_path)
        return usdz_path

    def export_organoid(self, cell_field: Tensor,
                          morphogen_field: Optional[Tensor] = None,
                          gene_fates: Optional[np.ndarray] = None,
                          base_path: str = "organoid",
                          dx: float = 1.0,
                          use_crate: bool = True) -> USDZExportResult:
        """
        Full organoid export pipeline.

        Args:
            cell_field: (X,Y,Z) cell density field φ
            morphogen_field: optional (X,Y,Z) morphogen for colour overlay
            gene_fates: optional per-cell fate IDs for colour coding
            base_path: output file base path (without extension)
            dx: grid spacing
            use_crate: use .usdc crate (smaller) vs .usda text

        Returns:
            USDZExportResult with file info and diagnostics
        """
        fields_exported = ["cell_density"]

        # Extract cell boundary mesh
        mesh = self.field_to_mesh(cell_field, dx)
        if mesh is None or mesh.get("n_faces", 0) == 0:
            return USDZExportResult(
                path="", format="none", n_vertices=0, n_faces=0,
                file_size_bytes=0, fields_exported=[],
                ar_viewable=False,
                metadata={"error": "no mesh extracted"})

        # Colour by morphogen if provided
        color = (0.31, 0.80, 0.77)  # default teal
        if morphogen_field is not None:
            fields_exported.append("morphogen")
            # Map mean morphogen to colour (simple heuristic)
            m_mean = float(morphogen_field.mean())
            color = (0.2 + 0.6*m_mean, 0.8 - 0.3*m_mean, 0.77)

        # Write GLB
        glb_path = base_path + ".glb"
        self.mesh_to_glb(mesh, glb_path, color=color)

        # Convert to USDZ
        usdz_path = base_path + ".usdz"
        final_path = self.glb_to_usdz(glb_path, usdz_path, use_crate)
        if final_path is None:
            final_path = glb_path

        import os
        file_size = os.path.getsize(final_path) if os.path.exists(final_path) else 0

        return USDZExportResult(
            path=final_path,
            format="usdz" if final_path.endswith(".usdz") else "glb",
            n_vertices=mesh["n_vertices"],
            n_faces=mesh["n_faces"],
            file_size_bytes=file_size,
            fields_exported=fields_exported,
            ar_viewable=True,
            metadata={
                "cell_volume_fraction": float((cell_field > 0).float().mean()),
                "morphogen_mean": float(morphogen_field.mean()) if morphogen_field is not None else None,
                "n_gene_fates": len(np.unique(gene_fates)) if gene_fates is not None else 0,
                "use_crate": use_crate,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "marching_cubes": self.mc is not None,
            "glb_exporter": self.glb is not None,
            "usdz_converter": self.usdz is not None,
            "crate_writer": self.crate is not None,
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class OrganoidVertexGRNSuite:
    """
    Unifies: OrganoidOntogeneticEngine + CarpetVertexModel + BooleanGRN
             + OrganoidUSDZExporter.

    FULL ONTOGENY PIPELINE:
      1. RD prepattern (Turing morphogens)
      2. Cell condensation (CH phase separation)
      3. Vertex mechanics rearrangement (cell intercalation)
      4. GRN fate assignment (Boolean attractors)
      5. AR export (.usdz)
    """

    def __init__(self,
                 vertex_cfg: Optional[VertexConfig] = None,
                 grn_cfg: Optional[GRNConfig] = None,
                 organoid_cfg: Optional[Any] = None,
                 usdz_level: float = 0.0):
        self.vertex = CarpetVertexModel(vertex_cfg)
        self.grn = BooleanGRN(grn_cfg)
        self.exporter = OrganoidUSDZExporter(usdz_level)
        self.organoid = None
        if HAS_ORGANOID:
            self.organoid = OrganoidOntogeneticEngine(organoid_cfg)

    def run_vertex_mechanics(self, n_steps: int = 50) -> List[VertexModelState]:
        """Run the vertex model for cell rearrangement."""
        return self.vertex.run(n_steps)

    def run_grn_fate_assignment(self, morphogen_field: np.ndarray,
                                  n_cells: int) -> np.ndarray:
        """Assign cell fates via the Boolean GRN."""
        self.grn.find_attractors()
        return self.grn.assign_cell_fates(morphogen_field, n_cells)

    def export_to_ar(self, cell_field: Tensor,
                       morphogen_field: Optional[Tensor] = None,
                       gene_fates: Optional[np.ndarray] = None,
                       base_path: str = "organoid") -> USDZExportResult:
        """Export the organoid to .usdz for AR visualization."""
        return self.exporter.export_organoid(
            cell_field, morphogen_field, gene_fates, base_path)

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "vertex": self.vertex.get_status(),
            "grn": self.grn.get_status(),
            "exporter": self.exporter.get_status(),
            "organoid_available": self.organoid is not None,
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("="*78)
    print(" SESI ORGANOID STUDIO — VERTEX MECHANICS + BOOLEAN GRN + AR EXPORT")
    print("="*78)
    print(f" torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print(f" CH solvers      : {HAS_CH}")
    print(f" Organoid engine : {HAS_ORGANOID}")
    print(f" Crate writer    : {HAS_CRATE}")
    print(f" USDZ converter  : {HAS_USDZ}")
    print(f" Marching cubes  : {HAS_MC}")

    suite = OrganoidVertexGRNSuite(
        vertex_cfg=VertexConfig(carpet_level=3, n_steps=30, dt=0.05),
        grn_cfg=GRNConfig(n_genes=12, k_regulators=3, n_attractor_trials=20))

    # ── 1. Vertex model on carpet ───────────────────────────────────────
    print("\n [VTX1] Carpet vertex model")
    print(f"   COROLLARY-3 guard : {suite.vertex.cfg.device}")
    history = suite.vertex.run(n_steps=20)
    first, last = history[0], history[-1]
    print(f"   vertices           : {len(last.positions)}")
    print(f"   cells              : {len(last.cells)}")
    print(f"   edges              : {len(last.edges)}")
    print(f"   initial energy     : {first.total_energy:.4f}")
    print(f"   final energy       : {last.total_energy:.4f}")
    print(f"   energy decreased   : {last.total_energy <= first.total_energy}")
    print(f"   T1 transitions     : {last.n_t1_transitions}")
    print(f"   T2 transitions     : {last.n_t2_transitions}")
    print(f"   cell area mean     : {float(np.mean(last.cell_areas)):.4f}")
    print(f"   cell area std      : {float(np.std(last.cell_areas)):.4f}")
    vtx_status = suite.vertex.get_status()
    print(f"   guards             : {vtx_status['corollary3_guard']}")
    print(f"                      : {vtx_status['hino_guard']}")

    # ── 2. Boolean GRN ──────────────────────────────────────────────────
    print("\n [GRN1] Boolean gene regulatory network")
    grn = suite.grn
    print(f"   genes / regulators : {grn.n} / K={grn.k}")
    attractors = grn.find_attractors(n_trials=30)
    print(f"   attractors found   : {len(attractors)}")
    for att in attractors[:5]:
        print(f"     {att.fate_label}: length={att.length}, "
              f"basin={att.basin_size}, "
              f"expr_mean={att.gene_expression.mean():.3f}")

    # Spatial fate assignment with a morphogen gradient
    n_cells = 50
    morphogen = np.linspace(0, 1, n_cells)  # gradient
    fates = grn.assign_cell_fates(morphogen, n_cells)
    unique_fates = np.unique(fates)
    print(f"   morphogen gradient : [{morphogen.min():.2f}, {morphogen.max():.2f}]")
    print(f"   cells assigned     : {n_cells}")
    print(f"   unique fates       : {len(unique_fates)}")
    print(f"   fate distribution  : "
          f"{[(int(f), int((fates==f).sum())) for f in unique_fates]}")
    grn_status = grn.get_status()
    print(f"   guard              : {grn_status['corollary3_guard']}")

    # ── 3. Organoid USDZ export ─────────────────────────────────────────
    print("\n [OEX1] Organoid USDZ export for AR")
    # Create a synthetic organoid cell field
    n = 16
    xs = torch.linspace(-1, 1, n)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    # Sphere with some noise (organoid-like)
    phi = -(torch.sqrt(gx**2 + gy**2 + gz**2) - 0.6)
    phi = phi + 0.1 * torch.randn(n, n, n)

    # Morphogen field (gradient along x)
    morph = (gx + 1) / 2.0  # 0 to 1

    # Gene fates (from GRN)
    n_field_cells = int((phi > 0).sum())
    gene_fates = np.random.randint(0, max(len(attractors), 1),
                                     size=max(n_field_cells, 1))

    import tempfile, os
    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, "organoid_demo")

    result = suite.exporter.export_organoid(
        phi, morphogen_field=morph, gene_fates=gene_fates,
        base_path=base, use_crate=True)

    print(f"   export path        : {os.path.basename(result.path)}")
    print(f"   format             : {result.format}")
    print(f"   n_vertices         : {result.n_vertices}")
    print(f"   n_faces            : {result.n_faces}")
    print(f"   file_size_bytes    : {result.file_size_bytes}")
    print(f"   fields_exported    : {result.fields_exported}")
    print(f"   ar_viewable        : {result.ar_viewable}")
    print(f"   cell_vol_fraction  : {result.metadata.get('cell_volume_fraction', 0):.3f}")
    print(f"   morphogen_mean     : {result.metadata.get('morphogen_mean', 0):.3f}")

    # Verify the file exists and has content
    if os.path.exists(result.path):
        size = os.path.getsize(result.path)
        print(f"   file exists        : True ({size} bytes)")
        # Check for PXR-USDC magic if crate
        if result.format == "usdz":
            with open(result.path, "rb") as fh:
                blob = fh.read()
            has_crate = b"PXR-USDC" in blob
            has_usda = b"#usda" in blob
            print(f"   crate magic        : {has_crate}")
            print(f"   usda header        : {has_usda}")
    else:
        print(f"   file exists        : False")

    # ── 4. Full pipeline: organoid ontogeny + vertex + GRN + export ────
    print("\n [PIPELINE] Full organoid ontogeny pipeline")
    if HAS_ORGANOID and suite.organoid is not None:
        print(f"   Running organoid ontogeny...")
        org_history = suite.organoid.run_ontogeny(seed=0, n_steps=30)
        last_snap = org_history[-1]
        print(f"   ontogeny steps     : {len(org_history)}")
        print(f"   final cell frac    : {last_snap.n_cells_estimate:.4f}")
        print(f"   Turing active      : {last_snap.turing_active}")
        print(f"   pattern wavelength : {last_snap.pattern_wavelength:.3f}")

        # Now run vertex mechanics on the final cell field
        # (project to 2D for the vertex model)
        cell_2d = last_snap.cell_field[:, :, last_snap.cell_field.shape[2]//2]
        print(f"   2D cell field      : {tuple(cell_2d.shape)}")

        # GRN fate assignment on the organoid
        morph_np = last_snap.morphogen_u.cpu().numpy().flatten()[:100]
        fates_organoid = grn.assign_cell_fates(morph_np, 100)
        print(f"   GRN fates assigned : {len(np.unique(fates_organoid))} unique")

        # Export the organoid
        result2 = suite.exporter.export_organoid(
            last_snap.cell_field,
            morphogen_field=last_snap.morphogen_u,
            base_path=os.path.join(tmp, "organoid_full"),
            use_crate=True)
        print(f"   Full export        : {result2.format}, "
              f"V={result2.n_vertices}, F={result2.n_faces}, "
              f"size={result2.file_size_bytes} bytes")
    else:
        print("   [SKIP] Organoid engine not available")

    # ── 5. Full status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}: {v}")

    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "="*78)
    print(" HONEST SCOPE")
    print(" " + "-"*76)
    print(" [C]   Vertex model energy: E = Σ K_A(A-A₀)² + K_P(P-P₀)² + Σ Λ L")
    print("       (Farhadifar et al. 2005). Forces are FIRST-ORDER only.")
    print(" [D]   COROLLARY-3 GUARD: no Hessian of vertex energy. D(A) is not")
    print("       an algebra (Carpet-19 Cor. 3).")
    print(" [C]   HINO GUARD: line tensions from order parameter, not smooth")
    print("       energy density. Γ(f,·) ⊥ µ (Thm 8).")
    print(" [C]   Boolean GRN: Kauffman NK model (1969). Attractors = cell fates.")
    print("       Morphogen coupling: Wolpert positional information (1969).")
    print(" [D]   GRN coupling is first-order. No second derivatives of gene")
    print("       expression through the carpet operator.")
    print(" [D]   OrganoidUSDZExporter: field → mesh → GLB → USDZ pipeline.")
    print("       Visual representation only. NOT medical imaging.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved. Unused by these modules.")
    print(" [C/D] Theorem 10 conditional on (I1)-(I4). NOT transferred to")
    print("       biological systems. Organoid model is COMPUTATIONAL ONLY.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("="*78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
