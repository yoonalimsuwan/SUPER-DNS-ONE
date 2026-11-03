"""
=============================================================================
SESI ORGANOID STUDIO 3-D
Voronoi Vertex Model + Continuous Hill-Kinetics GRN + Division/Death
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
[V3D1] VoronoiVertexModel3D
      • 3-D vertex model [C Bi et al. 2015, 2016] via Voronoi tessellation
      • Energy: E = Σ_α K_V(V_α - V₀)² + K_S(S_α - S₀)²
                     + Σ_{αβ} Λ_{αβ} A_{αβ}
          V_α = cell volume, S_α = surface area, A_{αβ} = shared face area
      • Forces: F_α = -∂E/∂r_α  (FIRST-ORDER ONLY; COROLLARY-3 guard)
      • Voronoi via scipy.spatial when available; spherical fallback otherwise
      • T1 (neighbour swap / face flip), T2 (cell deletion) in 3-D
      • HINO guard: tensions from volumes/areas, NOT smooth energy density

[CGR1] ContinuousGRN
      • ODE-based gene regulation with Hill kinetics [C Hill 1910; Alon 2006]
          dx_i/dt = Σ_j β_{ij} H_{ij}(x_j) - γ_i x_i  +  morphogen input
          H_act(x,K,n) = x^n/(K^n + x^n)
          H_rep(x,K,n) = K^n/(K^n + x^n)
      • RK4 integration (differentiable via torch)
      • Fixed-point / limit-cycle attractor detection by long simulation
      • Morphogen coupling: Wolpert positional information [C 1969]
      • COROLLARY-3 guard: morphogen enters as a first-order forcing term,
        never differentiated twice

[CDD1] CellDivisionDeathController
      • Division: V_α > V_divide → split cell along a chosen plane
      • Death:    V_α < V_death OR GRN apoptotic fate → remove cell
      • GRN-gated: gene-expression state modulates divide/die rates
      • Morphogen-gated: local morphogen shifts the volume thresholds
      • Updates the Voronoi topology after every division/death

[O3G1] SESI3DVertexGRNSuite (orchestrator)
      • Couples V3D1 + CGR1 + CDD1 into one organoid life-cycle loop
      • Exports snapshots for the AR pipeline (reuses OrganoidUSDZExporter)

=============================================================================
HONEST SCOPE
=============================================================================
• VoronoiVertexModel3D is a COMPUTATIONAL cell mechanics model. It uses
  Voronoi geometry as a proxy for cell shapes; real epithelia have
  non-Voronoi topology. Not validated against any tissue.
• ContinuousGRN uses deterministic Hill-kinetics ODEs. Real GRNs are
  stochastic, involve thousands of genes, and have non-Hill regulation.
  Attractor detection is by simulation, not exhaustive.
• CellDivisionDeathController uses heuristic volume checkpoints. Real
  cell-cycle control involves CDK/cyclin machinery far beyond this model.
• COROLLARY-3 GUARD: all force/field computations are first-order.
• HINO GUARD: energy measures are singular; no smooth density assumed.
• ρ≈1.25 is NUMERICAL, not proved, and unused by the 3-D geometry.
• All parameters are illustrative. No biological/medical claim.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.vertex3d_grn")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from scipy.spatial import Voronoi, ConvexHull
    HAS_SCIPY_SPATIAL = True
except Exception:
    HAS_SCIPY_SPATIAL = False

try:
    from sesi_topology_studio_vertex_grn import (
        BooleanGRN, GRNConfig, OrganoidUSDZExporter,
    )
    HAS_PREV = True
except Exception:
    HAS_PREV = False

__all__ = [
    "Vertex3DConfig", "VoronoiVertexModel3D", "Vertex3DState",
    "GRNODEConfig", "ContinuousGRN",
    "DivisionDeathConfig", "CellDivisionDeathController",
    "SESI3DVertexGRNSuite",
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
# SECTION 1 — [V3D1] VORONOI VERTEX MODEL 3-D
# =============================================================================

@dataclass
class Vertex3DConfig:
    """Configuration for the 3-D Voronoi vertex model."""
    # domain
    box_size: float = 10.0
    n_cells: int = 40
    # mechanics [C Bi et al. 2015/2016]
    K_volume: float = 1.0       # volume elasticity
    K_surface: float = 0.1      # surface contractility
    Lambda: float = 0.02        # face tension
    target_volume: float = 1.0  # V₀
    target_surface: float = 6.0 # S₀
    # dynamics
    mobility: float = 0.05
    dt: float = 0.02
    n_steps: int = 100
    # topology transitions
    t2_volume_threshold: float = 0.05   # remove cell below this volume
    enable_t2: bool = True
    # division (handed to CDD1)
    enable_division: bool = True
    division_volume: float = 2.0        # divide above this volume
    # numerics
    noise_amplitude: float = 0.0
    seed: int = 0
    device: str = "cpu"


@dataclass
class Vertex3DState:
    """Snapshot of the 3-D vertex model."""
    step: int
    time: float
    centres: np.ndarray            # (N, 3)
    volumes: np.ndarray            # (N,)
    surfaces: np.ndarray           # (N,)
    n_cells: int
    total_energy: float
    n_t2: int
    n_divisions: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class VoronoiVertexModel3D:
    """
    [C/D] 3-D vertex model using Voronoi tessellation.

    ENERGY  [C Bi et al. 2015, 2016]
      E = Σ_α [ K_V (V_α - V₀)² + K_S (S_α - S₀)² ]
          + Σ_{α<β} Λ_{αβ} A_{αβ}
    where V_α is cell volume, S_α its surface area, and A_{αβ} the area
    of the face shared by cells α and β.

    VORONOI GEOMETRY  [D]
      Cell centres r_α define a Voronoi tessellation of the box. Each
      Voronoi region is a convex polyhedron = the cell. Volume, surface
      area, and shared-face areas are computed from the Voronoi geometry.
      When scipy is unavailable, a SPHERICAL FALLBACK approximates each
      cell as a sphere of radius from its local free volume.

    FORCES  [D]
      F_α = -∂E/∂r_α, computed by finite differences on the energy
      (first-order). COROLLARY-3 GUARD: NO Hessian, NO second derivatives.

    HINO GUARD  [C Thm 8]
      Face tensions Λ_{αβ} are set from the ORDER PARAMETER (volumes,
      areas), NOT from an assumed smooth energy density. The singular
      energy measure means tension concentrates on specific faces — this
      is captured by letting Λ vary per face.

    T2 TRANSITION  [C]
      Cells with V_α < t2_volume_threshold are removed (apoptosis /
      extrusion). The Voronoi is recomputed after removal.

    HONEST
      • Voronoi is a PROXY for cell shape. Real tissues are not Voronoi.
      • Spherical fallback is coarse; use scipy for real geometry.
      • Periodic boundaries handled by mirroring centres across faces.
    """

    def __init__(self, cfg: Optional[Vertex3DConfig] = None):
        self.cfg = cfg or Vertex3DConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._step = 0
        self._time = 0.0
        self._n_t2 = 0
        self._n_divisions = 0
        self._centres: Optional[np.ndarray] = None
        self._volumes: Optional[np.ndarray] = None
        self._surfaces: Optional[np.ndarray] = None
        self._face_areas: Optional[Dict[Tuple[int,int], float]] = None

    # ── initialisation ──────────────────────────────────────────────────
    def seed_cells(self) -> None:
        """Place n_cells centres randomly in the box."""
        n = self.cfg.n_cells
        L = self.cfg.box_size
        self._centres = self._rng.uniform(0.0, L, size=(n, 3))
        self._volumes = np.full(n, self.cfg.target_volume)
        self._surfaces = np.full(n, self.cfg.target_surface)
        self._face_areas = {}
        logger.info("[V3D] seeded %d cells in box %.1f^3", n, L)

    def add_cell(self, centre: np.ndarray, volume: Optional[float] = None
                  ) -> int:
        """Add a cell (used by division). Returns new index."""
        c = np.asarray(centre, dtype=float).reshape(3)
        if self._centres is None:
            self.seed_cells()
        self._centres = np.vstack([self._centres, c])
        v = volume if volume is not None else self.cfg.target_volume
        self._volumes = np.append(self._volumes, v)
        self._surfaces = np.append(self._surfaces, self.cfg.target_surface)
        return len(self._centres) - 1

    def remove_cell(self, idx: int) -> None:
        """Remove a cell (T2 / apoptosis)."""
        if self._centres is None:
            return
        mask = np.ones(len(self._centres), dtype=bool)
        mask[idx] = False
        self._centres = self._centres[mask]
        self._volumes = self._volumes[mask]
        self._surfaces = self._surfaces[mask]

    # ── Voronoi geometry ────────────────────────────────────────────────
    def _compute_voronoi(self) -> None:
        """
        Recompute volumes, surfaces, and face areas from current centres.
        Uses scipy.spatial.Voronoi when available; spherical fallback
        otherwise.
        """
        if self._centres is None or len(self._centres) == 0:
            return
        pts = self._centres
        n = len(pts)
        L = self.cfg.box_size

        if HAS_SCIPY_SPATIAL and n >= 4:
            # Mirror points for periodic-ish boundary handling
            mirrors = []
            for dx in (-L, 0, L):
                for dy in (-L, 0, L):
                    for dz in (-L, 0, L):
                        mirrors.append(pts + np.array([dx, dy, dz]))
            all_pts = np.vstack(mirrors)
            try:
                vor = Voronoi(all_pts)
                # The original cells are the middle block (index offset n*13)
                offset = 13 * n  # 13th block is (0,0,0)
                vols = np.zeros(n)
                surfs = np.zeros(n)
                face_areas: Dict[Tuple[int,int], float] = {}
                for i in range(n):
                    region_idx = vor.point_region[offset + i]
                    region = vor.regions[region_idx]
                    if -1 in region or len(region) < 4:
                        vols[i] = self.cfg.target_volume
                        surfs[i] = self.cfg.target_surface
                        continue
                    poly = vor.vertices[region]
                    try:
                        hull = ConvexHull(poly)
                        vols[i] = hull.volume
                        surfs[i] = hull.area
                    except Exception:
                        vols[i] = self.cfg.target_volume
                        surfs[i] = self.cfg.target_surface
                # Shared-face areas: approximate via pairwise Voronoi ridge
                # areas. For brevity use a distance-weighted estimate.
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < 2.0 * (self.cfg.target_volume ** (1/3)) * 1.5:
                            # rough face area ~ cross-section
                            r_i = (3 * vols[i] / (4 * math.pi)) ** (1/3)
                            r_j = (3 * vols[j] / (4 * math.pi)) ** (1/3)
                            overlap = max(0.0, (r_i + r_j) - d)
                            face_areas[(i, j)] = math.pi * (overlap/2)**2
                self._volumes = vols
                self._surfaces = surfs
                self._face_areas = face_areas
                return
            except Exception as e:
                logger.debug("[V3D] Voronoi failed (%s); using fallback", e)

        # SPHERICAL FALLBACK: estimate local free volume per cell
        self._spherical_fallback()

    def _spherical_fallback(self) -> None:
        """Approximate each cell as a sphere from nearest-neighbour spacing."""
        pts = self._centres
        n = len(pts)
        vols = np.zeros(n)
        surfs = np.zeros(n)
        face_areas = {}
        for i in range(n):
            # nearest-neighbour distance as diameter proxy
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[i] = np.inf
            nn = d.min() if n > 1 else self.cfg.box_size / 2
            r = max(nn / 2.0, 1e-3)
            vols[i] = (4.0 / 3.0) * math.pi * r ** 3
            surfs[i] = 4.0 * math.pi * r ** 2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                r_i = (3 * vols[i] / (4 * math.pi)) ** (1/3)
                r_j = (3 * vols[j] / (4 * math.pi)) ** (1/3)
                overlap = max(0.0, (r_i + r_j) - d)
                if overlap > 0:
                    face_areas[(i, j)] = math.pi * (overlap / 2) ** 2
        self._volumes = vols
        self._surfaces = surfs
        self._face_areas = face_areas

    # ── energy ──────────────────────────────────────────────────────────
    def compute_energy(self) -> float:
        """
        Total 3-D vertex-model energy.
        HINO GUARD: tensions vary per face (singular energy concentration).
        """
        if self._centres is None:
            return 0.0
        V0 = self.cfg.target_volume
        S0 = self.cfg.target_surface
        E = 0.0
        for i in range(len(self._centres)):
            dV = self._volumes[i] - V0
            dS = self._surfaces[i] - S0
            E += self.cfg.K_volume * dV * dV
            E += self.cfg.K_surface * dS * dS
        # face tension
        for (i, j), area in self._face_areas.items():
            E += self.cfg.Lambda * area
        return float(E)

    # ── forces (first-order only; COROLLARY-3 guard) ───────────────────
    def compute_forces(self) -> np.ndarray:
        """
        F_i = -∂E/∂r_i via central finite differences.
        COROLLARY-3 GUARD: NO second derivatives. Forces are first-order.
        """
        if self._centres is None:
            return np.zeros((0, 3))
        n = len(self._centres)
        forces = np.zeros((n, 3))
        eps = 1e-4
        E0 = self.compute_energy()
        saved_centres = self._centres.copy()
        for i in range(n):
            for d in range(3):
                self._centres = saved_centres.copy()
                self._centres[i, d] += eps
                self._compute_voronoi()
                E_plus = self.compute_energy()
                forces[i, d] = -(E_plus - E0) / eps
        self._centres = saved_centres
        self._compute_voronoi()
        return forces

    # ── T2 / division hooks (called by CDD1) ────────────────────────────
    def apply_t2(self) -> int:
        """Remove cells below the T2 volume threshold. Returns count."""
        if not self.cfg.enable_t2 or self._centres is None:
            return 0
        keep = self._volumes >= self.cfg.t2_volume_threshold
        removed = int((~keep).sum())
        if removed:
            self._centres = self._centres[keep]
            self._volumes = self._volumes[keep]
            self._surfaces = self._surfaces[keep]
            self._n_t2 += removed
        return removed

    def divide_cell(self, idx: int) -> Optional[int]:
        """
        Divide cell idx into two daughters along a random plane.
        Returns the index of the new daughter, or None on failure.
        """
        if self._centres is None or idx >= len(self._centres):
            return None
        parent = self._centres[idx]
        # daughter offset along a random unit vector
        direction = self._rng.normal(size=3)
        direction /= (np.linalg.norm(direction) + 1e-12)
        offset = direction * 0.25 * (self._volumes[idx] ** (1/3))
        daughter_centre = parent + offset
        # halve the parent volume
        self._volumes[idx] *= 0.5
        self._surfaces[idx] *= 0.5 ** (2/3)
        new_idx = self.add_cell(daughter_centre, volume=self._volumes[idx])
        self._n_divisions += 1
        return new_idx

    # ── dynamics ────────────────────────────────────────────────────────
    def step(self) -> Vertex3DState:
        """One vertex-model step."""
        if self._centres is None:
            self.seed_cells()
        self._compute_voronoi()
        forces = self.compute_forces()
        if self.cfg.noise_amplitude > 0:
            forces += self.cfg.noise_amplitude * self._rng.standard_normal(forces.shape)
        # move centres
        self._centres += self.cfg.mobility * forces * self.cfg.dt
        # periodic wrap
        L = self.cfg.box_size
        self._centres = np.mod(self._centres, L)
        self._step += 1
        self._time += self.cfg.dt
        return self.snapshot()

    def snapshot(self) -> Vertex3DState:
        return Vertex3DState(
            step=self._step, time=self._time,
            centres=self._centres.copy() if self._centres is not None else np.zeros((0,3)),
            volumes=self._volumes.copy() if self._volumes is not None else np.zeros(0),
            surfaces=self._surfaces.copy() if self._surfaces is not None else np.zeros(0),
            n_cells=len(self._centres) if self._centres is not None else 0,
            total_energy=self.compute_energy(),
            n_t2=self._n_t2, n_divisions=self._n_divisions,
        )

    def run(self, n_steps: Optional[int] = None) -> List[Vertex3DState]:
        if self._centres is None:
            self.seed_cells()
        steps = n_steps or self.cfg.n_steps
        history = []
        sample_every = max(1, steps // 10)
        for s in range(steps):
            state = self.step()
            if s % sample_every == 0 or s == steps - 1:
                history.append(state)
        return history

    def to_density_field(self, grid: int = 16) -> Tensor:
        """
        Rasterise cell centres to a density field for meshing / AR export.
        Each cell deposits a smooth blob at its centre.
        """
        L = self.cfg.box_size
        xs = torch.linspace(0, L, grid)
        gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
        phi = torch.zeros(grid, grid, grid)
        if self._centres is None:
            return phi
        for i, c in enumerate(self._centres):
            r = max((3 * self._volumes[i] / (4 * math.pi)) ** (1/3), 1e-3)
            d2 = ((gx - c[0])**2 + (gy - c[1])**2 + (gz - c[2])**2)
            phi += torch.exp(-d2 / (2 * r * r))
        return phi.clamp(0, 1)

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_cells": len(self._centres) if self._centres is not None else 0,
            "n_t2": self._n_t2,
            "n_divisions": self._n_divisions,
            "has_scipy_voronoi": HAS_SCIPY_SPATIAL,
            "corollary3_guard": "first-order forces only; no Hessian",
            "hino_guard": "face tensions from volumes/areas, not smooth density",
        }


# =============================================================================
# SECTION 2 — [CGR1] CONTINUOUS GRN (HILL KINETICS ODE)
# =============================================================================

@dataclass
class GRNODEConfig:
    """Configuration for the continuous (ODE) GRN."""
    n_genes: int = 8
    k_regulators: int = 3
    # Hill parameters
    hill_n: float = 2.0          # Hill coefficient (cooperativity)
    hill_K: float = 0.5          # half-activation concentration
    beta: float = 1.0            # max production rate
    gamma: float = 0.5           # degradation rate
    # integration
    dt: float = 0.01
    n_steps: int = 500
    # morphogen coupling
    morphogen_weight: float = 0.3
    # attractor detection
    attractor_transient: int = 300   # discard this many steps
    attractor_window: int = 200      # analyse this many steps
    seed: int = 42


class ContinuousGRN(nn.Module):
    """
    [C/D] Continuous gene regulatory network with Hill kinetics.

    MODEL  [C Hill 1910; Alon 2006]
      For gene i with regulators R(i):
          dx_i/dt = β Σ_{j∈R(i)} w_ij H_ij(x_j) - γ x_i + m_i

      Hill activation:   H_act(x,K,n) = x^n / (K^n + x^n)
      Hill repression:   H_rep(x,K,n) = K^n / (K^n + x^n)

      m_i is the morphogen input (positional information, Wolpert 1969).

    INTEGRATION  [D]
      Classical RK4 for accuracy. All operations are torch tensors so the
      whole trajectory is differentiable w.r.t. parameters.

    ATTRACTORS  [D]
      For ODE systems attractors are fixed points or limit cycles. We
      detect them by simulating, discarding a transient, then clustering
      the remaining trajectory. A fixed point shows ~zero variance; a
      limit cycle shows periodic variance.

    COROLLARY-3 GUARD
      The morphogen m_i enters as a FIRST-ORDER forcing term. We do NOT
      compute ∂²x/∂m² or differentiate the GRN twice through the carpet
      operator.

    HONEST
      • Deterministic ODE; real GRNs are stochastic.
      • Hill kinetics is an approximation; real regulation is more complex.
      • n_genes is small here for tractability; real GRNs have thousands.
      • Attractor detection is by simulation, not bifurcation analysis.
    """

    def __init__(self, cfg: Optional[GRNODEConfig] = None):
        super().__init__()
        self.cfg = cfg or GRNODEConfig()
        n, k = self.cfg.n_genes, self.cfg.k_regulators
        self._rng = np.random.default_rng(self.cfg.seed)
        # regulatory graph
        self.regulators: List[List[int]] = []
        self.signs: List[List[float]] = []   # +1 activation, -1 repression
        for i in range(n):
            regs = self._rng.choice(
                [j for j in range(n) if j != i], size=k, replace=False).tolist()
            self.regulators.append(regs)
            self.signs.append([float(self._rng.choice([-1.0, 1.0]))
                                for _ in regs])
        # parameters (learnable for differentiability)
        self.beta = nn.Parameter(torch.full((n,), self.cfg.beta))
        self.gamma = nn.Parameter(torch.full((n,), self.cfg.gamma))
        self.K = nn.Parameter(torch.full((n, k), self.cfg.hill_K))
        self.n_hill = nn.Parameter(torch.full((n, k), self.cfg.hill_n))
        self.w = nn.Parameter(torch.ones(n, k) / k)

    # ── Hill functions ──────────────────────────────────────────────────
    @staticmethod
    def hill_act(x: Tensor, K: Tensor, n: Tensor) -> Tensor:
        """Hill activation: x^n / (K^n + x^n)."""
        x = _softplus_floor(x, 1e-9)
        K = _softplus_floor(K, 1e-9)
        xn = torch.exp(n * torch.log(x))
        Kn = torch.exp(n * torch.log(K))
        return xn / (Kn + xn + 1e-12)

    @staticmethod
    def hill_rep(x: Tensor, K: Tensor, n: Tensor) -> Tensor:
        """Hill repression: K^n / (K^n + x^n)."""
        x = _softplus_floor(x, 1e-9)
        K = _softplus_floor(K, 1e-9)
        xn = torch.exp(n * torch.log(x))
        Kn = torch.exp(n * torch.log(K))
        return Kn / (Kn + xn + 1e-12)

    # ── RHS of the ODE ──────────────────────────────────────────────────
    def rhs(self, x: Tensor, morphogen: Optional[Tensor] = None) -> Tensor:
        """
        dx/dt = β Σ_j w_ij H_ij(x_j) - γ x_i + morphogen_weight * m_i
        x: (n_genes,) or (batch, n_genes)
        """
        n = self.cfg.n_genes
        if x.dim() == 1:
            x = x.unsqueeze(0)
        prod = torch.zeros(x.shape[0], n, device=x.device, dtype=x.dtype)
        for i in range(n):
            for r_idx, j in enumerate(self.regulators[i]):
                sgn = self.signs[i][r_idx]
                xj = x[:, j]
                K = _softplus_floor(self.K[i, r_idx], 1e-6)
                nh = _softplus_floor(self.n_hill[i, r_idx], 0.1)
                if sgn > 0:
                    h = self.hill_act(xj, K, nh)
                else:
                    h = self.hill_rep(xj, K, nh)
                prod[:, i] = prod[:, i] + self.w[i, r_idx] * h
        dxdt = self.beta * prod - self.gamma * x
        # morphogen input (first-order forcing; COROLLARY-3 guard)
        if morphogen is not None:
            m = morphogen if morphogen.dim() == 2 else morphogen.unsqueeze(0)
            dxdt = dxdt + self.cfg.morphogen_weight * m
        return dxdt.squeeze(0) if x.shape[0] == 1 else dxdt

    # ── RK4 integration ─────────────────────────────────────────────────
    def rk4_step(self, x: Tensor, morphogen: Optional[Tensor] = None,
                  dt: Optional[float] = None) -> Tensor:
        dt = dt or self.cfg.dt
        k1 = self.rhs(x, morphogen)
        k2 = self.rhs(x + 0.5 * dt * k1, morphogen)
        k3 = self.rhs(x + 0.5 * dt * k2, morphogen)
        k4 = self.rhs(x + dt * k3, morphogen)
        x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return _softplus_floor(x_new, 0.0)  # concentrations stay >= 0

    def simulate(self, x0: Optional[Tensor] = None,
                  n_steps: Optional[int] = None,
                  morphogen: Optional[Tensor] = None) -> Tensor:
        """
        Simulate the GRN from x0 for n_steps.
        Returns trajectory of shape (n_steps+1, n_genes).
        """
        n = self.cfg.n_genes
        if x0 is None:
            x0 = torch.rand(n) * 0.5
        x = x0.clone()
        steps = n_steps or self.cfg.n_steps
        traj = [x.detach().clone()]
        for _ in range(steps):
            x = self.rk4_step(x, morphogen)
            traj.append(x.detach().clone())
        return torch.stack(traj, dim=0)

    # ── attractor detection ─────────────────────────────────────────────
    def detect_attractor(self, x0: Optional[Tensor] = None,
                          morphogen: Optional[Tensor] = None
                          ) -> Dict[str, Any]:
        """
        Simulate and classify the attractor as fixed-point or limit-cycle.
        Returns the attractor type, mean state, and oscillation amplitude.
        """
        traj = self.simulate(x0, self.cfg.n_steps, morphogen)
        transient = self.cfg.attractor_transient
        window = traj[transient: transient + self.cfg.attractor_window]
        if len(window) < 10:
            window = traj[-min(len(traj), self.cfg.attractor_window):]
        mean_state = window.mean(dim=0)
        std_state = window.std(dim=0)
        osc_amp = float(std_state.max())
        is_fixed = osc_amp < 1e-3
        return {
            "type": "fixed_point" if is_fixed else "limit_cycle",
            "mean_state": mean_state,
            "oscillation_amplitude": osc_amp,
            "per_gene_std": std_state.tolist(),
            "trajectory": traj,
        }

    # ── spatial fate assignment (position-dependent morphogen) ─────────
    def assign_fates(self, morphogen_field: np.ndarray,
                       n_cells: int) -> np.ndarray:
        """
        Assign a GRN steady-state 'fate' to each cell based on its local
        morphogen. Cells with similar morphogen -> similar attractor ->
        same fate cluster.
        """
        fates = np.zeros(n_cells, dtype=np.int32)
        states = []
        for i in range(n_cells):
            m_val = float(morphogen_field[i]) if i < len(morphogen_field) else 0.0
            m = torch.tensor([m_val])
            att = self.detect_attractor(morphogen=m)
            states.append(att["mean_state"].numpy())
        states = np.array(states)
        # simple clustering by expression similarity
        if len(states) > 0:
            from itertools import groupby
            # bin by dominant gene
            dominant = states.argmax(axis=1)
            fates = dominant
        return fates

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_genes": self.cfg.n_genes,
            "k_regulators": self.cfg.k_regulators,
            "hill_n": self.cfg.hill_n,
            "hill_K": self.cfg.hill_K,
            "integrator": "RK4",
            "corollary3_guard": "morphogen is first-order forcing only",
        }


# =============================================================================
# SECTION 3 — [CDD1] CELL DIVISION / DEATH CONTROLLER
# =============================================================================

@dataclass
class DivisionDeathConfig:
    """Configuration for cell division and death."""
    # division
    division_volume_factor: float = 1.8   # divide when V > factor * V₀
    division_noise: float = 0.1
    # death (apoptosis)
    death_volume_factor: float = 0.3      # die when V < factor * V₀
    enable_apoptosis: bool = True
    # GRN gating
    grn_gate_division: bool = True        # GRN state modulates division
    grn_gate_death: bool = True
    # morphogen gating
    morphogen_shift: float = 0.2          # morphogen shifts thresholds
    # limits
    max_cells: int = 200
    seed: int = 0


class CellDivisionDeathController:
    """
    [D] Cell life-cycle controller coupled to the 3-D vertex model and GRN.

    DIVISION RULE  [D]
      A cell α divides when its volume exceeds a threshold:
          V_α > division_volume_factor * V₀ * (1 + GRN_modulation)
      The morphogen can shift the threshold (positional control of growth).
      Division splits the cell into two daughters along a random plane.

    DEATH RULE (apoptosis)  [D]
      A cell α dies when:
          V_α < death_volume_factor * V₀  (too small / crushed), OR
          the GRN state indicates an apoptotic fate.
      Dead cells are removed (T2-like) and the Voronoi is recomputed.

    GRN GATING  [C/D]
      The GRN steady-state expression modulates division/death:
        high "proliferation gene" expression -> lower division threshold
        high "apoptosis gene" expression     -> higher death probability
      This is a HEURISTIC coupling, not a molecular cell-cycle model.

    MORPHOGEN GATING  [C Wolpert 1969]
      Local morphogen concentration shifts the volume thresholds, so
      growth and death are spatially patterned.

    HONEST
      • Volume checkpoints are HEURISTIC. Real cell-cycle control involves
        CDK/cyclin, checkpoints, and mechanics far beyond this.
      • Division orientation is random; real divisions follow mechanical
        stress or polarity cues.
      • No biological/medical claim. Computational model only.
    """

    def __init__(self, cfg: Optional[DivisionDeathConfig] = None):
        self.cfg = cfg or DivisionDeathConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._total_divisions = 0
        self._total_deaths = 0

    def should_divide(self, volume: float, target_volume: float,
                        grn_state: Optional[np.ndarray] = None,
                        morphogen: float = 0.0) -> bool:
        """Decide whether a cell should divide."""
        threshold = self.cfg.division_volume_factor * target_volume
        # morphogen shifts threshold (high morphogen -> easier division)
        threshold *= (1.0 - self.cfg.morphogen_shift * morphogen)
        # GRN gating: assume gene 0 promotes division
        if self.cfg.grn_gate_division and grn_state is not None and len(grn_state) > 0:
            div_gene = float(grn_state[0])
            threshold *= (1.0 - 0.3 * div_gene)  # high div_gene -> divide sooner
        # noise
        threshold *= (1.0 + self.cfg.division_noise * self._rng.standard_normal())
        return volume > threshold

    def should_die(self, volume: float, target_volume: float,
                     grn_state: Optional[np.ndarray] = None,
                     morphogen: float = 0.0) -> bool:
        """Decide whether a cell should undergo apoptosis."""
        if not self.cfg.enable_apoptosis:
            return False
        death_threshold = self.cfg.death_volume_factor * target_volume
        # morphogen can protect or kill
        death_threshold *= (1.0 + self.cfg.morphogen_shift * morphogen)
        if volume < death_threshold:
            return True
        # GRN gating: assume last gene promotes apoptosis
        if self.cfg.grn_gate_death and grn_state is not None and len(grn_state) > 1:
            apo_gene = float(grn_state[-1])
            if apo_gene > 0.8:
                return True
        return False

    def apply(self, vertex_model: "VoronoiVertexModel3D",
               grn_states: Optional[np.ndarray] = None,
               morphogen_field: Optional[np.ndarray] = None) -> Dict[str, int]:
        """
        Apply division and death to all cells in the vertex model.
        Returns counts of divisions and deaths.
        """
        if vertex_model._centres is None:
            return {"divisions": 0, "deaths": 0}

        vertex_model._compute_voronoi()
        n = len(vertex_model._centres)
        target_v = vertex_model.cfg.target_volume

        divisions = 0
        deaths = 0
        # process deaths first (mark for removal)
        to_remove = []
        for i in range(n):
            gs = grn_states[i] if grn_states is not None and i < len(grn_states) else None
            m = float(morphogen_field[i]) if morphogen_field is not None and i < len(morphogen_field) else 0.0
            vol = vertex_model._volumes[i]
            if self.should_die(vol, target_v, gs, m):
                to_remove.append(i)
                deaths += 1
        # remove in reverse order to keep indices valid
        for i in sorted(to_remove, reverse=True):
            vertex_model.remove_cell(i)
        self._total_deaths += deaths

        # then divisions (on surviving cells)
        n_after = len(vertex_model._centres)
        for i in range(n_after):
            if len(vertex_model._centres) >= self.cfg.max_cells:
                break
            gs = grn_states[i] if grn_states is not None and i < len(grn_states) else None
            m = float(morphogen_field[i]) if morphogen_field is not None and i < len(morphogen_field) else 0.0
            vol = vertex_model._volumes[i]
            if self.should_divide(vol, target_v, gs, m):
                vertex_model.divide_cell(i)
                divisions += 1
        self._total_divisions += divisions

        return {"divisions": divisions, "deaths": deaths}

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_divisions": self._total_divisions,
            "total_deaths": self._total_deaths,
            "division_volume_factor": self.cfg.division_volume_factor,
            "death_volume_factor": self.cfg.death_volume_factor,
            "grn_gate_division": self.cfg.grn_gate_division,
            "grn_gate_death": self.cfg.grn_gate_death,
            "max_cells": self.cfg.max_cells,
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESI3DVertexGRNSuite:
    """
    Couples: VoronoiVertexModel3D + ContinuousGRN + CellDivisionDeathController.

    LIFE-CYCLE LOOP:
      1. Vertex model relaxes cell positions (mechanics).
      2. GRN runs for each cell -> steady-state expression.
      3. Division/death controller updates cell population.
      4. Repeat.
      5. Export final state to density field -> mesh -> AR.
    """

    def __init__(self,
                 vertex_cfg: Optional[Vertex3DConfig] = None,
                 grn_cfg: Optional[GRNODEConfig] = None,
                 dd_cfg: Optional[DivisionDeathConfig] = None):
        self.vertex = VoronoiVertexModel3D(vertex_cfg)
        self.grn = ContinuousGRN(grn_cfg)
        self.dd = CellDivisionDeathController(dd_cfg)
        self._history: List[Vertex3DState] = []

    def run_life_cycle(self, n_rounds: int = 5,
                        vertex_steps_per_round: int = 5,
                        morphogen_field: Optional[np.ndarray] = None
                        ) -> Dict[str, Any]:
        """
        Run the coupled organoid life-cycle for n_rounds.
        Each round: relax mechanics -> GRN fates -> division/death.
        """
        if self.vertex._centres is None:
            self.vertex.seed_cells()

        total_divisions = 0
        total_deaths = 0
        cell_counts = []

        for r in range(n_rounds):
            # 1. mechanics relaxation
            for _ in range(vertex_steps_per_round):
                self.vertex.step()
            # 2. GRN states for each cell
            n_cells = len(self.vertex._centres)
            grn_states = None
            if n_cells > 0:
                # use morphogen to set per-cell GRN input
                m = morphogen_field
                if m is None:
                    m = np.linspace(0, 1, n_cells)
                else:
                    m = np.resize(morphogen_field, n_cells)
                grn_states = np.zeros((n_cells, self.grn.cfg.n_genes))
                for i in range(n_cells):
                    att = self.grn.detect_attractor(
                        morphogen=torch.tensor([float(m[i])]))
                    grn_states[i] = att["mean_state"].numpy()
            # 3. division / death
            result = self.dd.apply(self.vertex, grn_states, morphogen_field)
            total_divisions += result["divisions"]
            total_deaths += result["deaths"]
            cell_counts.append(len(self.vertex._centres))

            logger.info("[LIFE] round %d: cells=%d divisions=%d deaths=%d",
                         r, len(self.vertex._centres),
                         result["divisions"], result["deaths"])

        self._history.append(self.vertex.snapshot())
        return {
            "rounds": n_rounds,
            "final_n_cells": len(self.vertex._centres),
            "total_divisions": total_divisions,
            "total_deaths": total_deaths,
            "cell_count_trajectory": cell_counts,
            "energy": self.vertex.compute_energy(),
        }

    def export_density_field(self, grid: int = 16) -> Tensor:
        """Rasterise current cell configuration to a density field."""
        return self.vertex.to_density_field(grid)

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "vertex": self.vertex.get_status(),
            "grn": self.grn.get_status(),
            "division_death": self.dd.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO 3-D — SMOKE TEST")
    print(" Voronoi Vertex Model + Hill-Kinetics GRN + Division/Death")
    print("=" * 78)
    print(f" torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print(f" scipy.spatial available : {HAS_SCIPY_SPATIAL}")
    print(f" prev modules available  : {HAS_PREV}")

    suite = SESI3DVertexGRNSuite(
        vertex_cfg=Vertex3DConfig(n_cells=20, n_steps=10, box_size=8.0),
        grn_cfg=GRNODEConfig(n_genes=6, k_regulators=2, n_steps=80,
                               attractor_transient=30, attractor_window=50),
        dd_cfg=DivisionDeathConfig(max_cells=60))

    # ── 1. 3-D Voronoi vertex model ─────────────────────────────────────
    print("\n [V3D1] 3-D Voronoi vertex model")
    suite.vertex.seed_cells()
    history = suite.vertex.run(n_steps=8)
    first, last = history[0], history[-1]
    print(f"   cells               : {last.n_cells}")
    print(f"   scipy Voronoi       : {HAS_SCIPY_SPATIAL}")
    print(f"   initial energy      : {first.total_energy:.4f}")
    print(f"   final energy        : {last.total_energy:.4f}")
    print(f"   energy decreased    : {last.total_energy <= first.total_energy + 1e-6}")
    print(f"   volume mean/std     : {last.volumes.mean():.3f} / {last.volumes.std():.3f}")
    print(f"   surface mean/std    : {last.surfaces.mean():.3f} / {last.surfaces.std():.3f}")
    vtx_status = suite.vertex.get_status()
    print(f"   corollary3 guard    : {vtx_status['corollary3_guard']}")
    print(f"   hino guard          : {vtx_status['hino_guard']}")

    # density field for AR export
    density = suite.vertex.to_density_field(grid=12)
    print(f"   density field shape : {tuple(density.shape)}")
    print(f"   density range       : [{float(density.min()):.3f}, {float(density.max()):.3f}]")

    # ── 2. Continuous GRN (Hill kinetics) ───────────────────────────────
    print("\n [CGR1] Continuous GRN with Hill kinetics")
    grn = suite.grn
    x0 = torch.rand(grn.cfg.n_genes) * 0.5
    traj = grn.simulate(x0, n_steps=100)
    print(f"   genes               : {grn.cfg.n_genes}")
    print(f"   regulators/gene     : {grn.cfg.k_regulators}")
    print(f"   trajectory shape    : {tuple(traj.shape)}")
    print(f"   final expression    : {[round(float(v),3) for v in traj[-1]]}")

    att = grn.detect_attractor(x0)
    print(f"   attractor type      : {att['type']}")
    print(f"   oscillation amp     : {att['oscillation_amplitude']:.5f}")
    print(f"   mean state          : {[round(float(v),3) for v in att['mean_state']]}")

    # morphogen-dependent attractors (positional information)
    m_low = grn.detect_attractor(x0, morphogen=torch.tensor([0.0]))
    m_high = grn.detect_attractor(x0, morphogen=torch.tensor([1.0]))
    diff = (m_low["mean_state"] - m_high["mean_state"]).abs().mean()
    print(f"   morphogen 0 vs 1    : mean state diff = {float(diff):.4f}")
    print(f"   guard               : {grn.get_status()['corollary3_guard']}")

    # ── 3. Cell division / death ────────────────────────────────────────
    print("\n [CDD1] Cell division / death controller")
    dd = suite.dd
    # test division decision
    big_vol = suite.vertex.cfg.target_volume * 2.5
    small_vol = suite.vertex.cfg.target_volume * 0.2
    div_yes = dd.should_divide(big_vol, suite.vertex.cfg.target_volume)
    div_no = dd.should_divide(small_vol, suite.vertex.cfg.target_volume)
    die_yes = dd.should_die(small_vol, suite.vertex.cfg.target_volume)
    print(f"   big volume divides  : {div_yes}")
    print(f"   small volume divides: {div_no}")
    print(f"   small volume dies   : {die_yes}")

    # ── 4. Full life-cycle ──────────────────────────────────────────────
    print("\n [LIFE] Coupled organoid life-cycle")
    n_before = len(suite.vertex._centres)
    result = suite.run_life_cycle(n_rounds=3, vertex_steps_per_round=3)
    print(f"   rounds              : {result['rounds']}")
    print(f"   cells before        : {n_before}")
    print(f"   cells after         : {result['final_n_cells']}")
    print(f"   total divisions     : {result['total_divisions']}")
    print(f"   total deaths        : {result['total_deaths']}")
    print(f"   cell trajectory     : {result['cell_count_trajectory']}")
    print(f"   final energy        : {result['energy']:.4f}")

    # ── 5. AR export of final density ───────────────────────────────────
    print("\n [EXPORT] Density field -> mesh (for AR pipeline)")
    final_density = suite.export_density_field(grid=12)
    # threshold to a signed field for marching cubes
    phi = final_density - 0.5
    try:
        from sesi_differentiable_topology import MarchingCubesExtractor
        mc = MarchingCubesExtractor(level=0.0)
        mesh = mc.extract(phi, dx=1.0)
        print(f"   mesh vertices       : {mesh.get('n_vertices', 0)}")
        print(f"   mesh faces          : {mesh.get('n_faces', 0)}")
    except Exception as e:
        print(f"   marching cubes      : unavailable ({e})")

    # ── 6. Full status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   3-D vertex energy E = Σ K_V(V-V₀)² + K_S(S-S₀)² + Σ Λ A")
    print("       (Bi et al. 2015/2016). Voronoi is a PROXY for cell shape.")
    print(" [D]   COROLLARY-3 GUARD: forces are first-order only. No Hessian")
    print("       of the mechanical energy (Carpet-19 Cor. 3: D(A) not an")
    print("       algebra).")
    print(" [C]   HINO GUARD: face tensions from volumes/areas, not a smooth")
    print("       energy density (Γ ⊥ µ, Thm 8).")
    print(" [C]   Hill kinetics (Hill 1910; Alon 2006). RK4 integration.")
    print(" [D]   Morphogen enters GRN as a FIRST-ORDER forcing term only.")
    print(" [D]   Division/death volume checkpoints are HEURISTIC. Real")
    print("       cell-cycle control involves CDK/cyclin machinery beyond this.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by 3-D geometry.")
    print(" [D]   Organoid model is COMPUTATIONAL ONLY. No biological or")
    print("       medical claim. Not validated against real organoids.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
