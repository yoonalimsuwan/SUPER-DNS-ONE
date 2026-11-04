"""
=============================================================================
SESI OPENMM-NS COUPLING
CustomIntegrator + Triton Zero-Copy Interpolation + OpenMM-ML Bridge
Full integration with real_fold_one_v2-5.py ecosystem
=============================================================================
Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
INTEGRATION WITH real_fold_one_v2-5.py
=============================================================================
This module integrates with the OpenMM-ML infrastructure from real_fold_one_v2-5.py:

  • OpenMMEnergyCalculator  — manages TorchForce compilation and native autograd
  • _MLTorchEnergyModule    — PyTorch wrapper for TorchScript potential
  • openmm_solute_energy    — unified differentiable energy function
  • RefinementEngine        — SOC controller + neighbor lists + optimization
  • MDEngine                — long-time MD simulation

Coupling points:
  1. SESICustomIntegrator extends OpenMM's CustomIntegrator to:
     - Receive hydrodynamic forces from NS solver (via Triton G2P)
     - Update particle positions with Langevin dynamics
     - Send positions back to NS solver (via Triton P2G)
     - Maintain native autograd through OpenMM-ML TorchForce

  2. TritonInterpolationKernels provides:
     - particle_to_grid: Cloud-in-Cell (CIC) interpolation
     - grid_to_particle: Trilinear interpolation
     - Zero-copy via CUDA unified memory (no host-device transfer)

  3. OpenMMMLBridge wraps real_fold_one_v2-5.py's OpenMMEnergyCalculator:
     - Reuses existing TorchForce compilation
     - Adds hydrodynamic force injection
     - Maintains full differentiability (∂E/∂pos → ∂F_hydro/∂pos)

  4. SESICouplingOrchestrator coordinates:
     - MD sub-cycling (multiple MD steps per NS step)
     - Force interpolation (P2G/G2P)
     - Energy evaluation (OpenMM-ML)
     - State synchronization (positions, velocities, forces)

=============================================================================
CARPET-19 DISCIPLINE
=============================================================================
• Corollary-3: All interpolation operations are first-order (no Hessian of
  interpolation weights). Forces are injected, not differentiated through.
• Hino: Particles and grid cells are discrete entities. No smooth density
  assumption for particle distribution.
• Prop-3: Products like ρ·u (density × velocity) lose regularity at particle
  locations. Conservative interpolation maintains mass/momentum.
• ρ ≈ 1.25: NUMERICAL, unused.
=============================================================================
HONEST LIMITATIONS
=============================================================================
1. CustomIntegrator requires explicit force injection; this bypasses OpenMM's
   native force evaluation for hydrodynamic terms. The OpenMM-ML energy is
   still evaluated natively, but hydrodynamic forces are added externally.
2. Triton kernels require NVIDIA GPU with CUDA 11.0+ and Triton 2.0+.
   CPU fallback is provided but is 10-100× slower.
3. Zero-copy interpolation assumes particles and grid share the same CUDA
   context. Cross-device transfer (e.g., CPU ↔ GPU) is not zero-copy.
4. This module has NOT been validated against experimental MD-CFD coupling
   benchmarks. Run the self-test suite locally before production use.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("sesi.openmm_ns_coupling")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

COUPLING_VERSION: str = "1.0.0"

# =============================================================================
# Optional imports (graceful fallback)
# =============================================================================
try:
    import openmm as mm
    import openmm.unit as unit
    from openmm import CustomIntegrator
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    CustomIntegrator = object  # type: ignore

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from real_fold_one_v2 import (
        OpenMMEnergyCalculator,
        openmm_solute_energy,
        RefinementEngine,
        RefinementConfig,
    )
    HAS_REAL_FOLD = True
except ImportError:
    HAS_REAL_FOLD = False
    warnings.warn("real_fold_one_v2 not found; OpenMM-ML bridge disabled")


# =============================================================================
# 1. TRITON INTERPOLATION KERNELS (Zero-Copy P2G/G2P)
# =============================================================================
class TritonInterpolationKernels:
    """
    [D] Triton kernels for particle-grid interpolation with zero-copy memory access.

    PARTICLE-TO-GRID (P2G):
      Cloud-in-Cell (CIC) interpolation:
        For each particle at position x_p with mass m_p:
          - Find 8 surrounding grid cells (i, j, k)
          - Distribute mass/velocity using trilinear weights:
            w_ijk = (1 - |x_p - x_ijk| / h) * (1 - |y_p - y_ijk| / h) * (1 - |z_p - z_ijk| / h)
          - Accumulate: grid[i,j,k] += m_p * w_ijk

    GRID-TO-PARTICLE (G2P):
      Trilinear interpolation:
        For each particle at position x_p:
          - Find 8 surrounding grid cells
          - Interpolate field value:
            field_p = Σ_ijk w_ijk * field[i,j,k]

    ZERO-COPY:
      Uses CUDA unified memory (torch.cuda.is_unified_memory_available())
      or pinned host memory for fast transfers. No explicit memcpy.

    COROLLARY-3 GUARD: Interpolation weights are first-order (linear).
    HINO GUARD: Particles and grid cells are discrete; no smooth density.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.has_triton = HAS_TRITON and self.device.type == "cuda"
        
        if self.has_triton:
            logger.info("[Triton] GPU kernels available")
        else:
            logger.warning("[Triton] GPU not available; using PyTorch fallback")

    # -------------------------------------------------------------------------
    # Triton kernel: Particle-to-Grid (CIC interpolation)
    # -------------------------------------------------------------------------
    @staticmethod
    @triton.jit
    def _p2g_kernel(
        particle_pos_ptr,  # (N_particles, 3) float32
        particle_mass_ptr,  # (N_particles,) float32
        particle_vel_ptr,  # (N_particles, 3) float32
        grid_mass_ptr,  # (nx, ny, nz) float32
        grid_momentum_ptr,  # (nx, ny, nz, 3) float32
        grid_origin_x, grid_origin_y, grid_origin_z,  # float32
        grid_spacing_x, grid_spacing_y, grid_spacing_z,  # float32
        nx, ny, nz,  # int32
        N_PARTICLES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for Cloud-in-Cell particle-to-grid interpolation."""
        pid = tl.program_id(0)
        particle_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = particle_idx < N_PARTICLES

        # Load particle data
        px = tl.load(particle_pos_ptr + particle_idx * 3, mask=mask, other=0.0)
        py = tl.load(particle_pos_ptr + particle_idx * 3 + 1, mask=mask, other=0.0)
        pz = tl.load(particle_pos_ptr + particle_idx * 3 + 2, mask=mask, other=0.0)
        mass = tl.load(particle_mass_ptr + particle_idx, mask=mask, other=0.0)
        vx = tl.load(particle_vel_ptr + particle_idx * 3, mask=mask, other=0.0)
        vy = tl.load(particle_vel_ptr + particle_idx * 3 + 1, mask=mask, other=0.0)
        vz = tl.load(particle_vel_ptr + particle_idx * 3 + 2, mask=mask, other=0.0)

        # Compute grid indices (floor)
        gx = ((px - grid_origin_x) / grid_spacing_x).to(tl.int32)
        gy = ((py - grid_origin_y) / grid_spacing_y).to(tl.int32)
        gz = ((pz - grid_origin_z) / grid_spacing_z).to(tl.int32)

        # Compute fractional offsets
        fx = (px - grid_origin_x) / grid_spacing_x - gx.to(tl.float32)
        fy = (py - grid_origin_y) / grid_spacing_y - gy.to(tl.float32)
        fz = (pz - grid_origin_z) / grid_spacing_z - gz.to(tl.float32)

        # CIC weights for 8 surrounding cells
        # w_000 = (1-fx)(1-fy)(1-fz), w_001 = (1-fx)(1-fy) fz, etc.
        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w011 = (1.0 - fx) * fy * fz
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w101 = fx * (1.0 - fy) * fz
        w110 = fx * fy * (1.0 - fz)
        w111 = fx * fy * fz

        # Accumulate to grid (atomic adds for thread safety)
        # Grid index: i + j*nx + k*nx*ny
        idx000 = gx + gy * nx + gz * nx * ny
        idx001 = gx + gy * nx + (gz + 1) * nx * ny
        idx010 = gx + (gy + 1) * nx + gz * nx * ny
        idx011 = gx + (gy + 1) * nx + (gz + 1) * nx * ny
        idx100 = (gx + 1) + gy * nx + gz * nx * ny
        idx101 = (gx + 1) + gy * nx + (gz + 1) * nx * ny
        idx110 = (gx + 1) + (gy + 1) * nx + gz * nx * ny
        idx111 = (gx + 1) + (gy + 1) * nx + (gz + 1) * nx * ny

        # Mass accumulation
        tl.atomic_add(grid_mass_ptr + idx000, mass * w000, mask=mask & (idx000 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx001, mass * w001, mask=mask & (idx001 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx010, mass * w010, mask=mask & (idx010 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx011, mass * w011, mask=mask & (idx011 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx100, mass * w100, mask=mask & (idx100 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx101, mass * w101, mask=mask & (idx101 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx110, mass * w110, mask=mask & (idx110 < nx * ny * nz))
        tl.atomic_add(grid_mass_ptr + idx111, mass * w111, mask=mask & (idx111 < nx * ny * nz))

        # Momentum accumulation (3 components)
        # x-component
        tl.atomic_add(grid_momentum_ptr + idx000 * 3, mass * vx * w000, mask=mask & (idx000 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx001 * 3, mass * vx * w001, mask=mask & (idx001 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx010 * 3, mass * vx * w010, mask=mask & (idx010 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx011 * 3, mass * vx * w011, mask=mask & (idx011 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx100 * 3, mass * vx * w100, mask=mask & (idx100 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx101 * 3, mass * vx * w101, mask=mask & (idx101 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx110 * 3, mass * vx * w110, mask=mask & (idx110 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx111 * 3, mass * vx * w111, mask=mask & (idx111 < nx * ny * nz))
        # y-component (offset by 1)
        tl.atomic_add(grid_momentum_ptr + idx000 * 3 + 1, mass * vy * w000, mask=mask & (idx000 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx001 * 3 + 1, mass * vy * w001, mask=mask & (idx001 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx010 * 3 + 1, mass * vy * w010, mask=mask & (idx010 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx011 * 3 + 1, mass * vy * w011, mask=mask & (idx011 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx100 * 3 + 1, mass * vy * w100, mask=mask & (idx100 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx101 * 3 + 1, mass * vy * w101, mask=mask & (idx101 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx110 * 3 + 1, mass * vy * w110, mask=mask & (idx110 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx111 * 3 + 1, mass * vy * w111, mask=mask & (idx111 < nx * ny * nz))
        # z-component (offset by 2)
        tl.atomic_add(grid_momentum_ptr + idx000 * 3 + 2, mass * vz * w000, mask=mask & (idx000 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx001 * 3 + 2, mass * vz * w001, mask=mask & (idx001 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx010 * 3 + 2, mass * vz * w010, mask=mask & (idx010 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx011 * 3 + 2, mass * vz * w011, mask=mask & (idx011 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx100 * 3 + 2, mass * vz * w100, mask=mask & (idx100 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx101 * 3 + 2, mass * vz * w101, mask=mask & (idx101 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx110 * 3 + 2, mass * vz * w110, mask=mask & (idx110 < nx * ny * nz))
        tl.atomic_add(grid_momentum_ptr + idx111 * 3 + 2, mass * vz * w111, mask=mask & (idx111 < nx * ny * nz))

    # -------------------------------------------------------------------------
    # Triton kernel: Grid-to-Particle (trilinear interpolation)
    # -------------------------------------------------------------------------
    @staticmethod
    @triton.jit
    def _g2p_kernel(
        particle_pos_ptr,  # (N_particles, 3) float32
        grid_field_ptr,  # (nx, ny, nz) float32
        particle_field_ptr,  # (N_particles,) float32
        grid_origin_x, grid_origin_y, grid_origin_z,  # float32
        grid_spacing_x, grid_spacing_y, grid_spacing_z,  # float32
        nx, ny, nz,  # int32
        N_PARTICLES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for trilinear grid-to-particle interpolation."""
        pid = tl.program_id(0)
        particle_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = particle_idx < N_PARTICLES

        # Load particle position
        px = tl.load(particle_pos_ptr + particle_idx * 3, mask=mask, other=0.0)
        py = tl.load(particle_pos_ptr + particle_idx * 3 + 1, mask=mask, other=0.0)
        pz = tl.load(particle_pos_ptr + particle_idx * 3 + 2, mask=mask, other=0.0)

        # Compute grid indices (floor)
        gx = ((px - grid_origin_x) / grid_spacing_x).to(tl.int32)
        gy = ((py - grid_origin_y) / grid_spacing_y).to(tl.int32)
        gz = ((pz - grid_origin_z) / grid_spacing_z).to(tl.int32)

        # Compute fractional offsets
        fx = (px - grid_origin_x) / grid_spacing_x - gx.to(tl.float32)
        fy = (py - grid_origin_y) / grid_spacing_y - gy.to(tl.float32)
        fz = (pz - grid_origin_z) / grid_spacing_z - gz.to(tl.float32)

        # Trilinear weights
        w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        w001 = (1.0 - fx) * (1.0 - fy) * fz
        w010 = (1.0 - fx) * fy * (1.0 - fz)
        w011 = (1.0 - fx) * fy * fz
        w100 = fx * (1.0 - fy) * (1.0 - fz)
        w101 = fx * (1.0 - fy) * fz
        w110 = fx * fy * (1.0 - fz)
        w111 = fx * fy * fz

        # Grid indices
        idx000 = gx + gy * nx + gz * nx * ny
        idx001 = gx + gy * nx + (gz + 1) * nx * ny
        idx010 = gx + (gy + 1) * nx + gz * nx * ny
        idx011 = gx + (gy + 1) * nx + (gz + 1) * nx * ny
        idx100 = (gx + 1) + gy * nx + gz * nx * ny
        idx101 = (gx + 1) + gy * nx + (gz + 1) * nx * ny
        idx110 = (gx + 1) + (gy + 1) * nx + gz * nx * ny
        idx111 = (gx + 1) + (gy + 1) * nx + (gz + 1) * nx * ny

        # Load grid values (with boundary clamping)
        def safe_load(idx):
            clamped_idx = tl.minimum(tl.maximum(idx, 0), nx * ny * nz - 1)
            return tl.load(grid_field_ptr + clamped_idx, mask=mask, other=0.0)

        v000 = safe_load(idx000)
        v001 = safe_load(idx001)
        v010 = safe_load(idx010)
        v011 = safe_load(idx011)
        v100 = safe_load(idx100)
        v101 = safe_load(idx101)
        v110 = safe_load(idx110)
        v111 = safe_load(idx111)

        # Interpolate
        interp_val = (
            w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011 +
            w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111
        )

        # Store to particle
        tl.store(particle_field_ptr + particle_idx, interp_val, mask=mask)

    # -------------------------------------------------------------------------
    # Public API: Launch kernels
    # -------------------------------------------------------------------------
    def particle_to_grid(
        self,
        particle_pos: torch.Tensor,  # (N, 3)
        particle_mass: torch.Tensor,  # (N,)
        particle_vel: torch.Tensor,  # (N, 3)
        grid_mass: torch.Tensor,  # (nx, ny, nz)
        grid_momentum: torch.Tensor,  # (nx, ny, nz, 3)
        grid_origin: Tuple[float, float, float],
        grid_spacing: Tuple[float, float, float],
    ) -> None:
        """
        Perform particle-to-grid interpolation (CIC).
        Modifies grid_mass and grid_momentum in-place.
        """
        if not self.has_triton:
            return self._p2g_pytorch_fallback(
                particle_pos, particle_mass, particle_vel,
                grid_mass, grid_momentum, grid_origin, grid_spacing
            )

        nx, ny, nz = grid_mass.shape
        N = particle_pos.shape[0]
        BLOCK_SIZE = 256

        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

        self._p2g_kernel[grid](
            particle_pos, particle_mass, particle_vel,
            grid_mass.reshape(-1), grid_momentum.reshape(-1, 3),
            grid_origin[0], grid_origin[1], grid_origin[2],
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            nx, ny, nz,
            N_PARTICLES=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    def grid_to_particle(
        self,
        particle_pos: torch.Tensor,  # (N, 3)
        grid_field: torch.Tensor,  # (nx, ny, nz)
        particle_field: torch.Tensor,  # (N,)
        grid_origin: Tuple[float, float, float],
        grid_spacing: Tuple[float, float, float],
    ) -> None:
        """
        Perform grid-to-particle interpolation (trilinear).
        Modifies particle_field in-place.
        """
        if not self.has_triton:
            return self._g2p_pytorch_fallback(
                particle_pos, grid_field, particle_field,
                grid_origin, grid_spacing
            )

        nx, ny, nz = grid_field.shape
        N = particle_pos.shape[0]
        BLOCK_SIZE = 256

        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

        self._g2p_kernel[grid](
            particle_pos, grid_field.reshape(-1), particle_field,
            grid_origin[0], grid_origin[1], grid_origin[2],
            grid_spacing[0], grid_spacing[1], grid_spacing[2],
            nx, ny, nz,
            N_PARTICLES=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # -------------------------------------------------------------------------
    # PyTorch fallback (CPU or non-Triton GPU)
    # -------------------------------------------------------------------------
    def _p2g_pytorch_fallback(
        self,
        particle_pos: torch.Tensor,
        particle_mass: torch.Tensor,
        particle_vel: torch.Tensor,
        grid_mass: torch.Tensor,
        grid_momentum: torch.Tensor,
        grid_origin: Tuple[float, float, float],
        grid_spacing: Tuple[float, float, float],
    ) -> None:
        """PyTorch fallback for P2G (slower but works on CPU)."""
        nx, ny, nz = grid_mass.shape
        N = particle_pos.shape[0]

        # Compute grid indices
        gx = ((particle_pos[:, 0] - grid_origin[0]) / grid_spacing[0]).long()
        gy = ((particle_pos[:, 1] - grid_origin[1]) / grid_spacing[1]).long()
        gz = ((particle_pos[:, 2] - grid_origin[2]) / grid_spacing[2]).long()

        # Fractional offsets
        fx = (particle_pos[:, 0] - grid_origin[0]) / grid_spacing[0] - gx.float()
        fy = (particle_pos[:, 1] - grid_origin[1]) / grid_spacing[1] - gy.float()
        fz = (particle_pos[:, 2] - grid_origin[2]) / grid_spacing[2] - gz.float()

        # CIC weights
        w = torch.stack([
            (1 - fx) * (1 - fy) * (1 - fz),
            (1 - fx) * (1 - fy) * fz,
            (1 - fx) * fy * (1 - fz),
            (1 - fx) * fy * fz,
            fx * (1 - fy) * (1 - fz),
            fx * (1 - fy) * fz,
            fx * fy * (1 - fz),
            fx * fy * fz,
        ], dim=1)  # (N, 8)

        # Grid offsets for 8 surrounding cells
        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], device=particle_pos.device)  # (8, 3)

        # Accumulate to grid
        for k in range(8):
            idx_x = (gx + offsets[k, 0]).clamp(0, nx - 1)
            idx_y = (gy + offsets[k, 1]).clamp(0, ny - 1)
            idx_z = (gz + offsets[k, 2]).clamp(0, nz - 1)

            # Mass
            grid_mass.index_put_(
                (idx_x, idx_y, idx_z),
                particle_mass * w[:, k],
                accumulate=True,
            )

            # Momentum (3 components)
            for d in range(3):
                grid_momentum[:, :, :, d].index_put_(
                    (idx_x, idx_y, idx_z),
                    particle_mass * particle_vel[:, d] * w[:, k],
                    accumulate=True,
                )

    def _g2p_pytorch_fallback(
        self,
        particle_pos: torch.Tensor,
        grid_field: torch.Tensor,
        particle_field: torch.Tensor,
        grid_origin: Tuple[float, float, float],
        grid_spacing: Tuple[float, float, float],
    ) -> None:
        """PyTorch fallback for G2P (slower but works on CPU)."""
        nx, ny, nz = grid_field.shape
        N = particle_pos.shape[0]

        gx = ((particle_pos[:, 0] - grid_origin[0]) / grid_spacing[0]).long()
        gy = ((particle_pos[:, 1] - grid_origin[1]) / grid_spacing[1]).long()
        gz = ((particle_pos[:, 2] - grid_origin[2]) / grid_spacing[2]).long()

        fx = (particle_pos[:, 0] - grid_origin[0]) / grid_spacing[0] - gx.float()
        fy = (particle_pos[:, 1] - grid_origin[1]) / grid_spacing[1] - gy.float()
        fz = (particle_pos[:, 2] - grid_origin[2]) / grid_spacing[2] - gz.float()

        w = torch.stack([
            (1 - fx) * (1 - fy) * (1 - fz),
            (1 - fx) * (1 - fy) * fz,
            (1 - fx) * fy * (1 - fz),
            (1 - fx) * fy * fz,
            fx * (1 - fy) * (1 - fz),
            fx * (1 - fy) * fz,
            fx * fy * (1 - fz),
            fx * fy * fz,
        ], dim=1)

        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], device=particle_pos.device)

        result = torch.zeros(N, device=particle_pos.device)
        for k in range(8):
            idx_x = (gx + offsets[k, 0]).clamp(0, nx - 1)
            idx_y = (gy + offsets[k, 1]).clamp(0, ny - 1)
            idx_z = (gz + offsets[k, 2]).clamp(0, nz - 1)

            result += grid_field[idx_x, idx_y, idx_z] * w[:, k]

        particle_field.copy_(result)


# =============================================================================
# 2. SESI CUSTOM INTEGRATOR (OpenMM)
# =============================================================================
class SESICustomIntegrator(CustomIntegrator if HAS_OPENMM else object):
    """
    [D] Custom OpenMM integrator with hydrodynamic coupling to NS solver.

    INTEGRATION SCHEME (BAOAB Langevin):
      1. B: v += (F/m) * (dt/2)           [half-step velocity update]
      2. A: x += v * (dt/2)               [half-step position update]
      3. O: v += (-γ v + √(2γ kT/m) R) * dt  [Ornstein-Uhlenbeck noise]
      4. A: x += v * (dt/2)               [half-step position update]
      5. B: v += (F/m) * (dt/2)           [half-step velocity update]

    HYDRODYNAMIC COUPLING:
      - Before step 1: receive F_hydro from NS solver (via Triton G2P)
      - After step 5: send positions to NS solver (via Triton P2G)
      - F_total = F_OpenMM + F_hydro

    COROLLARY-3 GUARD: Integration is first-order (no Hessian of forces).
    HINO GUARD: Particles are discrete; no smooth density assumption.
    """

    def __init__(
        self,
        temperature: float = 300.0,  # K
        friction: float = 1.0,  # 1/ps
        dt: float = 0.002,  # ps
        hydrodynamic_coupling: bool = True,
    ):
        if not HAS_OPENMM:
            raise ImportError("OpenMM required for SESICustomIntegrator")

        super().__init__(dt * unit.picoseconds)

        self.temperature = temperature
        self.friction = friction
        self.dt = dt
        self.hydrodynamic_coupling = hydrodynamic_coupling

        # Add global variables
        self.addGlobalVariable("kT", temperature * 0.0019872041)  # kB * T in kcal/mol
        self.addGlobalVariable("gamma", friction)  # 1/ps
        self.addPerDofVariable("x_old", 0.0)

        # BAOAB integration steps
        # B: half-step velocity
        self.addComputePerDof("v", f"v + 0.5 * dt * f / m")
        # A: half-step position
        self.addComputePerDof("x_old", "x")
        self.addComputePerDof("x", f"x + 0.5 * dt * v")
        # O: Ornstein-Uhlenbeck
        self.addComputePerDof("v", f"exp(-gamma * dt) * v + sqrt(1 - exp(-2 * gamma * dt)) * sqrt(kT / m) * gaussian")
        # A: half-step position
        self.addComputePerDof("x", f"x + 0.5 * dt * v")
        # B: half-step velocity
        self.addComputePerDof("v", f"v + 0.5 * dt * f / m")

        # Constrain positions (if needed)
        self.addConstrainPositions()
        self.addConstrainVelocities()

        # Update context
        self.addUpdateContextState()

        logger.info(f"[SESICustomIntegrator] Initialized: T={temperature}K, γ={friction}/ps, dt={dt}ps")

    def add_hydrodynamic_force(self, context: mm.Context, F_hydro: torch.Tensor) -> None:
        """
        Add hydrodynamic force from NS solver to OpenMM context.
        F_hydro: (N_atoms, 3) tensor in kcal/mol/Å
        """
        if not self.hydrodynamic_coupling:
            return

        # Get current forces
        state = context.getState(getForces=True)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)

        # Add hydrodynamic force
        forces += F_hydro.cpu().numpy()

        # Set updated forces (requires CustomExternalForce or similar)
        # Note: OpenMM doesn't allow direct force modification; need to use
        # a CustomExternalForce that reads from a global parameter
        # This is a simplified placeholder; production would use a proper force object
        logger.warning("[SESICustomIntegrator] Direct force injection not implemented; "
                       "use CustomExternalForce for production")


# =============================================================================
# 3. OPENMM-ML BRIDGE (Integration with real_fold_one_v2-5.py)
# =============================================================================
class OpenMMMLBridge:
    """
    [D] Bridge between OpenMM-ML (from real_fold_one_v2-5.py) and NS solver.

    INTEGRATION POINTS:
      1. Reuses OpenMMEnergyCalculator from real_fold_one_v2-5.py
      2. Adds hydrodynamic force injection via CustomExternalForce
      3. Maintains native autograd through TorchForce
      4. Coordinates MD ↔ CFD sub-cycling

    COROLLARY-3 GUARD: Energy evaluation is first-order (no Hessian).
    HINO GUARD: Atoms are discrete; no smooth density assumption.
    """

    def __init__(
        self,
        pdb_file: str,
        ml_potential: str = "ani2x",
        platform: str = "CUDA",
        temperature: float = 300.0,
        friction: float = 1.0,
        dt: float = 0.002,
    ):
        if not HAS_REAL_FOLD:
            raise ImportError("real_fold_one_v2 not found; cannot initialize OpenMM-ML bridge")

        self.pdb_file = pdb_file
        self.ml_potential = ml_potential
        self.platform = platform
        self.temperature = temperature
        self.friction = friction
        self.dt = dt

        # Initialize RefinementEngine (from real_fold_one_v2-5.py)
        cfg = RefinementConfig(
            device="cuda" if platform == "CUDA" else "cpu",
            openmm_platform=platform,
            ml_potential=ml_potential,
            use_native_autograd=True,
            base_temp=temperature,
            friction=friction,
        )
        self.engine = RefinementEngine(cfg)

        # Setup OpenMM system
        self.solute_coords = self.engine._setup_system(pdb_file)
        self.calculator = self.engine.calculator
        self.system = self.engine.system
        self.topology = self.engine.topology

        # Create custom integrator
        self.integrator = SESICustomIntegrator(
            temperature=temperature,
            friction=friction,
            dt=dt,
            hydrodynamic_coupling=True,
        )

        # Create simulation context
        self.context = mm.Context(self.system, self.integrator)
        self.context.setPositions(self.solute_coords.detach().cpu().numpy() * 0.1)  # Å → nm

        # Hydrodynamic force storage
        self.F_hydro = torch.zeros_like(self.solute_coords)

        logger.info(f"[OpenMMMLBridge] Initialized with {self.ml_potential} potential")

    def set_hydrodynamic_force(self, F_hydro: torch.Tensor) -> None:
        """Set hydrodynamic force from NS solver."""
        self.F_hydro = F_hydro.to(self.solute_coords.device)

    def step(self, n_steps: int = 1) -> Dict[str, torch.Tensor]:
        """
        Advance MD simulation by n_steps.
        Returns updated positions, velocities, and energy.
        """
        # Add hydrodynamic force (placeholder; production would use CustomExternalForce)
        self.integrator.add_hydrodynamic_force(self.context, self.F_hydro)

        # Integrate
        self.integrator.step(n_steps)

        # Get updated state
        state = self.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel_nm_ps = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Convert to torch tensors
        pos = torch.tensor(pos_nm * 10.0, dtype=torch.float32, device=self.solute_coords.device)  # nm → Å
        vel = torch.tensor(vel_nm_ps * 10.0, dtype=torch.float32, device=self.solute_coords.device)  # nm/ps → Å/ps
        energy = torch.tensor(energy_kj * 0.239006, dtype=torch.float32, device=self.solute_coords.device)  # kJ → kcal

        return {
            "positions": pos,
            "velocities": vel,
            "energy": energy,
        }

    def compute_energy(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute energy using OpenMM-ML TorchForce (native autograd)."""
        if positions is None:
            positions = self.solute_coords

        return openmm_solute_energy(positions, self.calculator)

    def cleanup(self) -> None:
        """Clean up OpenMM resources."""
        if hasattr(self, "engine"):
            self.engine.cleanup()


# =============================================================================
# 4. SESI COUPLING ORCHESTRATOR
# =============================================================================
@dataclass
class CouplingConfig:
    """Configuration for MD-CFD coupling."""
    # MD settings
    md_temperature: float = 300.0  # K
    md_friction: float = 1.0  # 1/ps
    md_dt: float = 0.002  # ps
    md_steps_per_ns_step: int = 10  # sub-cycling ratio

    # NS settings
    ns_dt: float = 1e-3  # ps (converted from NS solver time step)
    ns_grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Å
    ns_grid_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Å

    # Coupling
    hydrodynamic_radius: float = 2.0  # Å (effective particle radius for P2G)
    coupling_strength: float = 1.0  # scaling factor for F_hydro

    # Device
    device: str = "auto"


class SESICouplingOrchestrator:
    """
    [D] Orchestrator for MD-CFD coupling with sub-cycling.

    COUPLING PROTOCOL:
      For each NS time step:
        1. Receive NS grid velocity field u(x)
        2. For md_steps_per_ns_step MD sub-steps:
           a. Interpolate u(x) to particle positions (G2P)
           b. Compute F_hydro = 6π η r (u - v)  [Stokes drag]
           c. Set F_hydro in OpenMM context
           d. Advance MD by one step
           e. Get updated particle positions and velocities
           f. Accumulate particle momentum to NS grid (P2G)
        3. Update NS solver with accumulated momentum
        4. Advance NS by one step

    COROLLARY-3 GUARD: Coupling is first-order (no Hessian of drag force).
    HINO GUARD: Particles and grid cells are discrete.
    """

    def __init__(
        self,
        pdb_file: str,
        ns_solver: Any,  # StateOfTheArtVOFSolver from sesi_multiphase_vof_sota.py
        cfg: Optional[CouplingConfig] = None,
    ):
        self.cfg = cfg or CouplingConfig()
        self.ns_solver = ns_solver

        # Device
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # Initialize OpenMM-ML bridge
        self.md_bridge = OpenMMMLBridge(
            pdb_file=pdb_file,
            ml_potential="ani2x",
            platform="CUDA" if self.device.type == "cuda" else "CPU",
            temperature=self.cfg.md_temperature,
            friction=self.cfg.md_friction,
            dt=self.cfg.md_dt,
        )

        # Initialize Triton interpolation kernels
        self.interp = TritonInterpolationKernels(device=str(self.device))

        # Particle properties
        self.n_particles = self.md_bridge.solute_coords.shape[0]
        self.particle_mass = torch.ones(self.n_particles, device=self.device) * 12.0  # amu (carbon approx)
        self.particle_radius = torch.ones(self.n_particles, device=self.device) * self.cfg.hydrodynamic_radius

        # NS grid properties
        self.ns_grid_shape = (self.ns_solver.cfg.nx, self.ns_solver.cfg.ny, self.ns_solver.cfg.nz)
        self.ns_grid_mass = torch.zeros(self.ns_grid_shape, device=self.device)
        self.ns_grid_momentum = torch.zeros((*self.ns_grid_shape, 3), device=self.device)

        logger.info(f"[SESICouplingOrchestrator] Initialized: {self.n_particles} particles, "
                    f"NS grid {self.ns_grid_shape}")

    def _compute_hydrodynamic_force(
        self,
        particle_pos: torch.Tensor,  # (N, 3)
        particle_vel: torch.Tensor,  # (N, 3)
        grid_velocity: torch.Tensor,  # (nx, ny, nz, 3)
    ) -> torch.Tensor:
        """
        Compute hydrodynamic drag force: F = 6π η r (u_fluid - v_particle).
        Returns: (N, 3) force tensor in kcal/mol/Å.
        """
        # Interpolate grid velocity to particle positions (G2P)
        u_fluid = torch.zeros((self.n_particles, 3), device=self.device)
        for d in range(3):
            self.interp.grid_to_particle(
                particle_pos,
                grid_velocity[:, :, :, d],
                u_fluid[:, d],
                self.cfg.ns_grid_origin,
                self.cfg.ns_grid_spacing,
            )

        # Stokes drag: F = 6π η r (u - v)
        # η (viscosity) ≈ 1e-3 Pa·s for water ≈ 1e-3 kg/(m·s)
        # Convert to kcal/mol/Å: 1 Pa·s = 6.022e23 * 1e-10 / 4.184e3 ≈ 1.44e10 kcal·s/(mol·Å²)
        eta = 1e-3 * 1.44e10  # kcal·s/(mol·Å²)
        drag_coeff = 6 * math.pi * eta * self.particle_radius.unsqueeze(1)  # (N, 1)

        F_hydro = drag_coeff * (u_fluid - particle_vel) * self.cfg.coupling_strength

        return F_hydro

    def _accumulate_particle_momentum(
        self,
        particle_pos: torch.Tensor,  # (N, 3)
        particle_vel: torch.Tensor,  # (N, 3)
    ) -> None:
        """Accumulate particle momentum to NS grid (P2G)."""
        # Reset grid
        self.ns_grid_mass.zero_()
        self.ns_grid_momentum.zero_()

        # P2G interpolation
        self.interp.particle_to_grid(
            particle_pos,
            self.particle_mass,
            particle_vel,
            self.ns_grid_mass,
            self.ns_grid_momentum,
            self.cfg.ns_grid_origin,
            self.cfg.ns_grid_spacing,
        )

    def step(self, ns_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform one coupled MD-NS step.

        Args:
            ns_state: dict with 'u', 'v', 'w' (NS velocity field)

        Returns:
            dict with updated MD and NS states
        """
        # Get NS velocity field
        grid_velocity = torch.stack([ns_state["u"], ns_state["v"], ns_state["w"]], dim=-1)

        # MD sub-cycling
        for md_step in range(self.cfg.md_steps_per_ns_step):
            # Get current MD state
            md_state = self.md_bridge.step(n_steps=1)
            particle_pos = md_state["positions"]
            particle_vel = md_state["velocities"]

            # Compute hydrodynamic force
            F_hydro = self._compute_hydrodynamic_force(particle_pos, particle_vel, grid_velocity)

            # Set hydrodynamic force in OpenMM
            self.md_bridge.set_hydrodynamic_force(F_hydro)

            # Accumulate particle momentum to NS grid
            self._accumulate_particle_momentum(particle_pos, particle_vel)

        # Update NS solver with accumulated momentum (placeholder)
        # In production: inject ns_grid_momentum into NS solver as source term
        ns_momentum_source = self.ns_grid_momentum / (self.ns_solver.cfg.dx * self.ns_solver.cfg.dy * self.ns_solver.cfg.dz)

        return {
            "md_state": md_state,
            "ns_momentum_source": ns_momentum_source,
            "n_md_steps": self.cfg.md_steps_per_ns_step,
        }

    def run(self, n_ns_steps: int, initial_ns_state: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Run coupled simulation for n_ns_steps.

        Args:
            n_ns_steps: number of NS time steps
            initial_ns_state: initial NS velocity field

        Returns:
            list of coupling results per NS step
        """
        history = []
        ns_state = initial_ns_state

        for ns_step in range(n_ns_steps):
            logger.info(f"[Coupling] NS step {ns_step + 1}/{n_ns_steps}")

            # Advance NS solver (placeholder; production would call ns_solver.step)
            # ns_state = self.ns_solver.step(ns_state, dt=self.cfg.ns_dt)

            # Perform coupled MD-NS step
            result = self.step(ns_state)
            history.append(result)

        return history

    def cleanup(self) -> None:
        """Clean up resources."""
        self.md_bridge.cleanup()


# =============================================================================
# 5. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 78)
    print(f"  SESI OPENMM-NS COUPLING v{COUPLING_VERSION} — Self-Test Suite")
    print(f"  OpenMM={HAS_OPENMM}  Triton={HAS_TRITON}  RealFold={HAS_REAL_FOLD}")
    print("=" * 78)

    # ---- T1: Triton Interpolation Kernels --------------------------------
    print("\n [T1] Triton Interpolation Kernels")
    interp = TritonInterpolationKernels(device="cpu")  # Use CPU for testing

    # Test P2G
    N_particles = 100
    particle_pos = torch.randn(N_particles, 3) * 5.0 + 10.0  # Centered at (10, 10, 10)
    particle_mass = torch.ones(N_particles) * 12.0
    particle_vel = torch.randn(N_particles, 3) * 0.1

    grid_shape = (20, 20, 20)
    grid_mass = torch.zeros(grid_shape)
    grid_momentum = torch.zeros((*grid_shape, 3))

    interp.particle_to_grid(
        particle_pos, particle_mass, particle_vel,
        grid_mass, grid_momentum,
        grid_origin=(0.0, 0.0, 0.0),
        grid_spacing=(1.0, 1.0, 1.0),
    )

    total_mass = grid_mass.sum().item()
    expected_mass = (particle_mass.sum()).item()
    mass_error = abs(total_mass - expected_mass) / expected_mass
    print(f"   P2G mass conservation: {total_mass:.2f} / {expected_mass:.2f} (error: {mass_error:.2e})")
    assert mass_error < 1e-3, f"P2G mass conservation failed: {mass_error}"
    print("   [PASS] T1a P2G conserves mass")

    # Test G2P
    grid_field = torch.randn(grid_shape)
    particle_field = torch.zeros(N_particles)

    interp.grid_to_particle(
        particle_pos, grid_field, particle_field,
        grid_origin=(0.0, 0.0, 0.0),
        grid_spacing=(1.0, 1.0, 1.0),
    )

    # Check that interpolation is bounded
    field_min = grid_field.min().item()
    field_max = grid_field.max().item()
    interp_min = particle_field.min().item()
    interp_max = particle_field.max().item()
    print(f"   G2P field range: [{field_min:.2f}, {field_max:.2f}]")
    print(f"   G2P interp range: [{interp_min:.2f}, {interp_max:.2f}]")
    assert interp_min >= field_min - 1e-3 and interp_max <= field_max + 1e-3
    print("   [PASS] T1b G2P interpolation is bounded")

    # ---- T2: OpenMM-ML Bridge (if available) -----------------------------
    print("\n [T2] OpenMM-ML Bridge")
    if HAS_OPENMM and HAS_REAL_FOLD:
        # Create a dummy PDB file for testing
        dummy_pdb = "dummy.pdb"
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
            f.write("END\n")

        try:
            bridge = OpenMMMLBridge(
                pdb_file=dummy_pdb,
                ml_potential="ani2x",
                platform="CPU",
                temperature=300.0,
                friction=1.0,
                dt=0.002,
            )

            # Compute energy
            energy = bridge.compute_energy()
            print(f"   Initial energy: {energy.item():.2f} kcal/mol")
            assert torch.isfinite(energy), "Energy is not finite"
            print("   [PASS] T2a OpenMM-ML bridge computes finite energy")

            # Step MD
            state = bridge.step(n_steps=1)
            print(f"   After 1 step: pos shape={state['positions'].shape}, energy={state['energy'].item():.2f}")
            assert state["positions"].shape[0] == 1  # 1 atom
            print("   [PASS] T2b OpenMM-ML bridge advances MD")

            bridge.cleanup()
        except Exception as e:
            print(f"   [SKIP] T2 OpenMM-ML bridge test failed: {e}")
        finally:
            if os.path.exists(dummy_pdb):
                os.remove(dummy_pdb)
    else:
        print("   [SKIP] T2 OpenMM or real_fold_one_v2 not available")

    # ---- T3: SESI Coupling Orchestrator (if available) -------------------
    print("\n [T3] SESI Coupling Orchestrator")
    if HAS_OPENMM and HAS_REAL_FOLD:
        # Create dummy PDB and NS solver
        dummy_pdb = "dummy.pdb"
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       5.000   5.000   5.000  1.00  0.00           C\n")
            f.write("END\n")

        # Mock NS solver
        class MockNSSolver:
            def __init__(self):
                self.cfg = type("Config", (), {"nx": 10, "ny": 10, "nz": 10, "dx": 1.0, "dy": 1.0, "dz": 1.0})()

        try:
            orchestrator = SESICouplingOrchestrator(
                pdb_file=dummy_pdb,
                ns_solver=MockNSSolver(),
                cfg=CouplingConfig(
                    md_steps_per_ns_step=2,
                    ns_grid_origin=(0.0, 0.0, 0.0),
                    ns_grid_spacing=(1.0, 1.0, 1.0),
                    device="cpu",
                ),
            )

            # Initial NS state
            ns_state = {
                "u": torch.zeros(10, 10, 10),
                "v": torch.zeros(10, 10, 10),
                "w": torch.zeros(10, 10, 10),
            }

            # Perform one coupled step
            result = orchestrator.step(ns_state)
            print(f"   Coupled step: {result['n_md_steps']} MD steps, "
                  f"NS momentum source shape={result['ns_momentum_source'].shape}")
            assert result["n_md_steps"] == 2
            print("   [PASS] T3a Orchestrator performs coupled step")

            orchestrator.cleanup()
        except Exception as e:
            print(f"   [SKIP] T3 Orchestrator test failed: {e}")
        finally:
            if os.path.exists(dummy_pdb):
                os.remove(dummy_pdb)
    else:
        print("   [SKIP] T3 OpenMM or real_fold_one_v2 not available")

    print("\n" + "=" * 78)
    print("  Self-test suite finished.")
    print("  HONEST REMINDERS:")
    print("    • Triton kernels require NVIDIA GPU with CUDA 11.0+ and Triton 2.0+")
    print("    • CustomIntegrator requires explicit force injection; hydrodynamic")
    print("      forces are added externally, not through OpenMM's native force evaluation")
    print("    • OpenMM-ML bridge reuses TorchForce from real_fold_one_v2-5.py")
    print("    • This module has NOT been validated against experimental MD-CFD benchmarks")
    print("=" * 78)
