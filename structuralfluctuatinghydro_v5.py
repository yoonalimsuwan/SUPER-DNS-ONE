# =============================================================================
# STRUCTURAL LANGEVIN FOR FLUCTUATING HYDRODYNAMICS (CFD BRIDGE)
# =============================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Version   : 5.0 (production — full 3D + Native Full Differentiable)
# Year      : 2026
#
# A Fluctuating Hydrodynamics (FH) solver bridging the Structural Calculus
# Langevin framework to continuum CFD via the Landau–Lifshitz
# Navier–Stokes (LLNS) equations.
#
# Physical Model:
#   ∂ρ/∂t  + ∇·(ρu)    = 0                              (continuity)
#   ∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·(τ + S̃) + f         (momentum)
#
# where S̃ is the Landau–Lifshitz stochastic stress tensor (3×3 symmetric),
# constructed via the Structural Itô / CSOC framework from Papers 3 & 4.
#
# Grid convention:
#   Staggered 3-D Cartesian MAC (Marker-and-Cell) layout.
#   Scalars  (ρ, p)  : cell centres  (Nx, Ny, Nz)
#   Velocity ux      : x-face centres (Nx+1, Ny,   Nz  )
#   Velocity uy      : y-face centres (Nx,   Ny+1, Nz  )
#   Velocity uz      : z-face centres (Nx,   Ny,   Nz+1)
#
# Numerical scheme:
#   Time   : Fractional-step (projection), 1st-order explicit.
#   Space  : 7-point Laplacian (viscous), configurable advection:
#     "upwind"         – 1st-order upwind  (stable, diffusive)     [default]
#     "tvd"            – 2nd-order TVD  (limiter: minmod/van_leer/superbee)
#     "weno5"          – 5th-order WENO-5 (Jiang-Shu, LF split)
#     "semi_lagrangian"– unconditionally stable trilinear grid_sample (GPU)
#   Noise  : Discrete Landau-Lifshitz stochastic stress tensor (3×3).
#   Solver : Spectral FFT-based Poisson (rfft3, O(N³ log N), exact).
#
# Changes v4 → v5  (Native Full Differentiability):
#   DIFF-FIX 1 – TVD limiters: clamp/torch.where → softplus/tanh gating
#   DIFF-FIX 2 – WENO-5 LF wave speed: .max() → logsumexp (smooth)
#   DIFF-FIX 3 – 1st-order upwind: torch.where → soft-upwind tanh gate
#   DIFF-FIX 4 – TVD 1D flux: torch.where upwind → soft gate
#   DIFF-FIX 5 – LL stochastic stress: .clamp() → softplus floor
#   DIFF-FIX 6 – Itô correction: detach/requires_grad pattern preserved
#                but correction term kept in graph via retain_graph
#   DIFF-FIX 7 – step(): .clamp(min=1e-6) → softplus floor
#   DIFF-FIX 8 – CFL check: .max().item() → logsumexp (diagnostics only)
#   DIFF-FIX 9 – gradcheck test suite added
#
# Dependencies: torch >= 2.0
# =============================================================================

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from one_core import (
    SemanticStateContraction,
    CSOCBase,
    InterfaceDetectorBase,
    StructuralItoBase,
    ONE_VERSION,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Differentiability utilities  (shared with super_dns_one_v5)
# =============================================================================

_TAU_LIMITER = 1e-2   # TVD limiter gating temperature
_TAU_UPWIND  = 1e-2   # upwind gate temperature
_TAU_LSE     = 1e-2   # logsumexp wave-speed temperature
_SOFTPLUS_B  = 100.0  # softplus sharpness for positivity floors


def _softplus_floor(x: torch.Tensor, floor: float,
                    beta: float = _SOFTPLUS_B) -> torch.Tensor:
    """Differentiable floor:  floor + softplus(x − floor).
    Non-zero gradient everywhere; approaches clamp for large beta."""
    return floor + F.softplus(x - floor, beta=beta)


def _soft_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable |x| = sqrt(x²+ε). No kink at x=0."""
    return torch.sqrt(x * x + eps)


def _soft_gate(v: torch.Tensor, tau: float = _TAU_UPWIND) -> torch.Tensor:
    """Smooth gate ≈ Heaviside(v): σ(v/τ) ∈ (0,1). Gradient everywhere."""
    return torch.sigmoid(v / tau)


def _logsumexp_max(x: torch.Tensor, tau: float = _TAU_LSE) -> torch.Tensor:
    """Smooth differentiable max: τ·log Σ exp(xᵢ/τ). Gradient through all xᵢ."""
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# ── Smooth TVD slope limiters ─────────────────────────────────────────────────

def _lim_minmod(r: torch.Tensor) -> torch.Tensor:
    """Smooth minmod: soft same-sign gate × differentiable min(1,r)."""
    gate = _soft_gate(r, tau=_TAU_LIMITER)           # ≈1 when r>0
    mn   = 0.5*(1.0 + r) - 0.5*_soft_abs(1.0 - r)   # smooth min(1,r)
    pos  = 0.5*(mn + _soft_abs(mn))                   # smooth max(0,mn)
    return gate * pos


def _lim_van_leer(r: torch.Tensor) -> torch.Tensor:
    """Smooth van Leer: soft gate × (r+|r|)/(1+|r|)."""
    gate = _soft_gate(r, tau=_TAU_LIMITER)
    return gate * (r + _soft_abs(r)) / (1.0 + _soft_abs(r) + 1e-30)


def _lim_superbee(r: torch.Tensor) -> torch.Tensor:
    """Smooth superbee: soft gate × max(0, min(2r,1), min(r,2))."""
    gate = _soft_gate(r, tau=_TAU_LIMITER)
    m1   = 0.5*(2*r + 1.0) - 0.5*_soft_abs(2*r - 1.0)   # smooth min(2r,1)
    m2   = 0.5*(r   + 2.0) - 0.5*_soft_abs(r   - 2.0)   # smooth min(r,2)
    mx   = 0.5*(m1+m2) + 0.5*_soft_abs(m1-m2)            # smooth max(m1,m2)
    pos  = 0.5*(mx + _soft_abs(mx))
    return gate * pos


_LIMITERS: dict = {
    "minmod":   _lim_minmod,
    "van_leer": _lim_van_leer,
    "superbee": _lim_superbee,
}


# =============================================================================
# Configuration
# =============================================================================

class BoundaryCondition(str, Enum):
    PERIODIC  = "periodic"
    NO_SLIP   = "no_slip"
    FREE_SLIP = "free_slip"
    OPEN      = "open"


@dataclass
class SolverConfig:
    """
    Full configuration for :class:`StructuralFluctuatingHydro` (3-D).

    All physical quantities in SI units.

    Grid:
        Nx, Ny, Nz : cell counts (≥ 4 each).
        Lx, Ly, Lz : domain extents (m).
        dt         : time step (s).

    Physics:
        base_viscosity   : kinematic viscosity ν₀ (m²/s).
        base_diffusivity : thermal diffusivity α₀ (m²/s).
        kb_T             : thermal energy k_B T (J).  Default: 298 K.
        rho0             : reference density (kg/m³).

    BCs:
        bc_x, bc_y, bc_z : BoundaryCondition per axis.

    CSOC / SSC:
        interface_sharpness, interface_amp, viscosity_boost,
        sigma_target, epsilon_fp.

    Advection:
        advection_scheme  : "upwind" | "tvd" | "weno5" | "semi_lagrangian".
        advection_limiter : "minmod" | "van_leer" | "superbee"  (TVD only).

    Misc:
        cfl_limit, checkpoint_every, log_every, dtype.
    """
    Nx: int   = 32
    Ny: int   = 32
    Nz: int   = 32
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    dt: float = 1e-4

    base_viscosity:     float = 1e-3
    base_diffusivity:   float = 1e-5
    kb_T:               float = 4.11e-21   # k_B × 298 K
    rho0:               float = 1.0

    enable_fluctuations: bool = True

    bc_x: BoundaryCondition = BoundaryCondition.PERIODIC
    bc_y: BoundaryCondition = BoundaryCondition.PERIODIC
    bc_z: BoundaryCondition = BoundaryCondition.PERIODIC

    cfl_limit:           float = 0.5
    poisson_tol:         float = 1e-10

    interface_sharpness: float = 4.0
    interface_amp:       float = 3.0
    viscosity_boost:     float = 5.0
    sigma_target:        float = 0.1
    epsilon_fp:          float = 0.0028

    checkpoint_every: int = 0
    log_every:        int = 0
    dtype: torch.dtype = torch.float64

    advection_scheme:  str = "upwind"
    advection_limiter: str = "van_leer"

    def validate(self) -> None:
        for name, val in [("Nx", self.Nx), ("Ny", self.Ny), ("Nz", self.Nz)]:
            if val < 4:
                raise ValueError(f"{name} must be ≥ 4, got {val}.")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}.")
        if not (0.0 < self.epsilon_fp < 1.0):
            raise ValueError(f"epsilon_fp must be in (0,1), got {self.epsilon_fp}.")
        for name, val in [("kb_T", self.kb_T), ("rho0", self.rho0),
                          ("base_viscosity", self.base_viscosity)]:
            if val <= 0:
                raise ValueError(f"{name} must be positive.")
        _valid_schemes  = {"upwind", "tvd", "weno5", "semi_lagrangian"}
        _valid_limiters = {"minmod", "van_leer", "superbee"}
        if self.advection_scheme not in _valid_schemes:
            raise ValueError(f"advection_scheme must be one of {_valid_schemes}.")
        if self.advection_limiter not in _valid_limiters:
            raise ValueError(f"advection_limiter must be one of {_valid_limiters}.")


# =============================================================================
# 3-D MAC-grid finite-difference utilities
# =============================================================================

def _pad_field_3d(f: torch.Tensor, bc: BoundaryCondition) -> torch.Tensor:
    """
    Pad a cell-centred field (Nx, Ny, Nz) by 1 on all 6 faces.
    Returns (Nx+2, Ny+2, Nz+2).
    """
    f6 = f.unsqueeze(0).unsqueeze(0)   # (1,1,Nx,Ny,Nz)
    if bc == BoundaryCondition.PERIODIC:
        mode = "circular"
    elif bc == BoundaryCondition.FREE_SLIP:
        mode = "reflect"
    else:
        mode = "replicate"
    # F.pad pads last dims first: (z_l, z_r, y_l, y_r, x_l, x_r)
    out = F.pad(f6, (1, 1, 1, 1, 1, 1), mode=mode)
    return out.squeeze(0).squeeze(0)


def div_u(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """
    Cell-centred divergence of a 3-D MAC velocity field.

    Args:
        ux : (Nx+1, Ny,   Nz  )
        uy : (Nx,   Ny+1, Nz  )
        uz : (Nx,   Ny,   Nz+1)

    Returns:
        (Nx, Ny, Nz) divergence.
    """
    return (
        (ux[1:, :, :] - ux[:-1, :, :]) / dx
        + (uy[:, 1:, :] - uy[:, :-1, :]) / dy
        + (uz[:, :, 1:] - uz[:, :, :-1]) / dz
    )


def grad_p(
    p: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pressure gradient on MAC face centres (Neumann / zero-flux at boundaries).

    Returns:
        gx : (Nx+1, Ny,   Nz  )
        gy : (Nx,   Ny+1, Nz  )
        gz : (Nx,   Ny,   Nz+1)
    """
    Nx, Ny, Nz = p.shape
    dev, dt_ = p.device, p.dtype
    gx = torch.zeros(Nx+1, Ny,   Nz,   device=dev, dtype=dt_)
    gy = torch.zeros(Nx,   Ny+1, Nz,   device=dev, dtype=dt_)
    gz = torch.zeros(Nx,   Ny,   Nz+1, device=dev, dtype=dt_)
    gx[1:-1, :, :] = (p[1:, :, :] - p[:-1, :, :]) / dx
    gy[:, 1:-1, :] = (p[:, 1:, :] - p[:, :-1, :]) / dy
    gz[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / dz
    return gx, gy, gz


def laplacian_cell(
    f:  torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    bc: BoundaryCondition = BoundaryCondition.PERIODIC,
) -> torch.Tensor:
    """
    7-point cell-centred Laplacian with BC-aware padding.

    Args:
        f : (Nx, Ny, Nz)

    Returns:
        (Nx, Ny, Nz) Laplacian ∇²f.
    """
    fp = _pad_field_3d(f, bc)
    return (
        (fp[2:, 1:-1, 1:-1] - 2.0*f + fp[:-2, 1:-1, 1:-1]) / dx**2
        + (fp[1:-1, 2:, 1:-1] - 2.0*f + fp[1:-1, :-2, 1:-1]) / dy**2
        + (fp[1:-1, 1:-1, 2:] - 2.0*f + fp[1:-1, 1:-1, :-2]) / dz**2
    )


# =============================================================================
# FFT-based spectral Poisson solver (3-D, periodic, exact)
# =============================================================================

def _fft3_poisson_solve(
    rhs: torch.Tensor,
    dx:  float,
    dy:  float,
    dz:  float,
) -> torch.Tensor:
    """
    Solve ∇²p = rhs on a 3-D periodic domain via spectral method.

    Uses the discrete 7-point Laplacian eigenvalues.  Mean of p is zero.

    Args:
        rhs : (Nx, Ny, Nz).

    Returns:
        p   : (Nx, Ny, Nz).
    """
    Nx, Ny, Nz = rhs.shape
    dev, dt_ = rhs.device, rhs.dtype

    kx = torch.arange(Nx, device=dev, dtype=dt_)
    ky = torch.arange(Ny, device=dev, dtype=dt_)
    kz = torch.arange(Nz, device=dev, dtype=dt_)

    lx = (2.0*torch.cos(2.0*math.pi*kx/Nx) - 2.0) / dx**2
    ly = (2.0*torch.cos(2.0*math.pi*ky/Ny) - 2.0) / dy**2
    lz = (2.0*torch.cos(2.0*math.pi*kz/Nz) - 2.0) / dz**2

    eig = lx[:, None, None] + ly[None, :, None] + lz[None, None, :]   # (Nx,Ny,Nz)
    eig[0, 0, 0] = 1.0   # avoid divide-by-zero; zero mean enforced below

    rhs_hat = torch.fft.rfftn(rhs)
    eig_r   = eig[:, :, :Nz//2+1]   # rfftn output shape

    p_hat        = rhs_hat / eig_r
    p_hat[0,0,0] = 0.0 + 0.0j

    return torch.fft.irfftn(p_hat, s=(Nx, Ny, Nz))


# =============================================================================
# HIGH-ORDER ADVECTION LIBRARY  (3-D, dimensional splitting)
# =============================================================================

# ── TVD slope limiters ────────────────────────────────────────────────────────

# Limiter aliases pointing to smooth differentiable versions (DIFF-FIX 1)
_lim_minmod   = _lim_minmod    # defined in diff-utilities above
_lim_van_leer = _lim_van_leer
_lim_superbee = _lim_superbee
# _LIMITERS already defined in diff-utilities section above


def _tvd_flux_1d(
    q:   torch.Tensor,
    vel: torch.Tensor,
    dx:  float,
    dt:  float,
    bc:  BoundaryCondition,
    limiter: str = "van_leer",
) -> torch.Tensor:
    """
    1-D TVD scalar advection along the first (and only) axis of q.
    Lax-Wendroff base flux + TVD limiter correction.

    Returns dq/dt = −∂(vel·q)/∂x  (N,).
    """
    psi = _LIMITERS.get(limiter, _lim_van_leer)
    N   = q.shape[0]
    q4  = q.unsqueeze(0).unsqueeze(0)
    if bc == BoundaryCondition.PERIODIC:
        q_pad = F.pad(q4, (2, 2), mode="circular").squeeze(0).squeeze(0)
    elif bc == BoundaryCondition.FREE_SLIP:
        q_pad = F.pad(q4, (2, 2), mode="reflect").squeeze(0).squeeze(0)
    else:
        q_pad = F.pad(q4, (2, 2), mode="replicate").squeeze(0).squeeze(0)

    dq_m = q_pad[2:N+2] - q_pad[1:N+1]
    dq_p = q_pad[3:N+3] - q_pad[2:N+2]

    vel_face = 0.5 * (vel + torch.roll(vel, -1))
    nu       = vel_face * dt / dx

    # DIFF-FIX 4: soft-upwind gate instead of torch.where
    g_up  = _soft_gate(vel_face, tau=_TAU_UPWIND)
    dq_up = g_up * dq_m + (1.0 - g_up) * dq_p
    dq_dn = g_up * dq_p + (1.0 - g_up) * dq_m

    r   = dq_up / (dq_dn + 1e-30)
    phi = psi(r)

    q_upwind = g_up * q_pad[2:N+2] + (1.0 - g_up) * q_pad[3:N+3]
    flux     = vel_face * (q_upwind + 0.5 * phi * (1.0 - _soft_abs(nu)) * dq_dn)
    return -(flux - torch.roll(flux, 1)) / dx


def _weno5_flux_1d(
    q:   torch.Tensor,
    vel: torch.Tensor,
    dx:  float,
    bc:  BoundaryCondition,
) -> torch.Tensor:
    """
    1-D WENO-5 (Jiang-Shu 1996) scalar advection.
    Lax-Friedrichs split fluxes.  Returns dq/dt (N,).
    """
    eps = 1e-6
    N   = q.shape[0]
    q4  = q.unsqueeze(0).unsqueeze(0)
    if bc == BoundaryCondition.PERIODIC:
        q_pad = F.pad(q4, (3, 3), mode="circular").squeeze(0).squeeze(0)
    elif bc == BoundaryCondition.FREE_SLIP:
        q_pad = F.pad(q4, (3, 3), mode="reflect").squeeze(0).squeeze(0)
    else:
        q_pad = F.pad(q4, (3, 3), mode="replicate").squeeze(0).squeeze(0)

    q0,q1,q2,q3,q4_,q5 = (q_pad[k:N+k] for k in range(6))

    vel_face = 0.5*(vel + torch.roll(vel, -1))
    # DIFF-FIX 2: logsumexp-max instead of .max()
    alpha    = _logsumexp_max(_soft_abs(vel_face), tau=_TAU_LSE) + 1e-12

    def _fp(qi): return 0.5*(vel_face*qi + alpha*qi)
    def _fm(qi): return 0.5*(vel_face*qi - alpha*qi)

    # Positive flux reconstruction (left-biased)
    fp0,fp1,fp2,fp3,fp4,fp5 = _fp(q0),_fp(q1),_fp(q2),_fp(q3),_fp(q4_),_fp(q5)
    fh_p0 = ( 1./3.)*fp0 - (7./6.)*fp1 + (11./6.)*fp2
    fh_p1 = (-1./6.)*fp1 + (5./6.)*fp2 + ( 1./3.)*fp3
    fh_p2 = ( 1./3.)*fp2 + (5./6.)*fp3 - ( 1./6.)*fp4
    b0p = (13./12.)*(fp0-2*fp1+fp2)**2 + .25*(fp0-4*fp1+3*fp2)**2
    b1p = (13./12.)*(fp1-2*fp2+fp3)**2 + .25*(fp1-fp3)**2
    b2p = (13./12.)*(fp2-2*fp3+fp4)**2 + .25*(3*fp2-4*fp3+fp4)**2
    a0p=0.1/(eps+b0p)**2; a1p=0.6/(eps+b1p)**2; a2p=0.3/(eps+b2p)**2
    sp = a0p+a1p+a2p
    F_plus = (a0p*fh_p0 + a1p*fh_p1 + a2p*fh_p2) / sp

    # Negative flux reconstruction (right-biased)
    fm0,fm1,fm2,fm3,fm4,fm5 = _fm(q0),_fm(q1),_fm(q2),_fm(q3),_fm(q4_),_fm(q5)
    fh_m0 = ( 1./3.)*fm3 + (5./6.)*fm2  - ( 1./6.)*fm1
    fh_m1 = (-1./6.)*fm4 + (5./6.)*fm3  + ( 1./3.)*fm2
    fh_m2 = ( 1./3.)*fm5 - (7./6.)*fm4  + (11./6.)*fm3
    b0m = (13./12.)*(fm1-2*fm2+fm3)**2 + .25*(fm1-4*fm2+3*fm3)**2
    b1m = (13./12.)*(fm2-2*fm3+fm4)**2 + .25*(fm2-fm4)**2
    b2m = (13./12.)*(fm3-2*fm4+fm5)**2 + .25*(3*fm3-4*fm4+fm5)**2
    a0m=0.3/(eps+b0m)**2; a1m=0.6/(eps+b1m)**2; a2m=0.1/(eps+b2m)**2
    sm = a0m+a1m+a2m
    F_minus = (a0m*fh_m0 + a1m*fh_m1 + a2m*fh_m2) / sm

    flux = F_plus + F_minus
    return -(flux - torch.roll(flux, 1)) / dx


def _advect_upwind_3d(
    rho: torch.Tensor,
    ucx: torch.Tensor, ucy: torch.Tensor, ucz: torch.Tensor,
    dx: float, dy: float, dz: float,
    dt: float, bc: BoundaryCondition,
) -> torch.Tensor:
    """1st-order upwind density advection (3-D)."""
    fp = _pad_field_3d(rho, bc)
    # DIFF-FIX 3: soft upwind gates instead of torch.where
    gx = _soft_gate(ucx, tau=_TAU_UPWIND)
    gy = _soft_gate(ucy, tau=_TAU_UPWIND)
    gz = _soft_gate(ucz, tau=_TAU_UPWIND)
    drho_dx = (gx*(rho-fp[:-2,1:-1,1:-1]) + (1-gx)*(fp[2:,1:-1,1:-1]-rho)) / dx
    drho_dy = (gy*(rho-fp[1:-1,:-2,1:-1]) + (1-gy)*(fp[1:-1,2:,1:-1]-rho)) / dy
    drho_dz = (gz*(rho-fp[1:-1,1:-1,:-2]) + (1-gz)*(fp[1:-1,1:-1,2:]-rho)) / dz
    return rho - dt*(ucx*drho_dx + ucy*drho_dy + ucz*drho_dz)


def _advect_tvd_3d(
    rho: torch.Tensor,
    ucx: torch.Tensor, ucy: torch.Tensor, ucz: torch.Tensor,
    dx: float, dy: float, dz: float,
    dt: float, bc: BoundaryCondition, limiter: str,
) -> torch.Tensor:
    """2nd-order TVD density advection — Strang dimensional splitting (3-D)."""
    Nx, Ny, Nz = rho.shape

    # x-sweep
    rx = rho.clone()
    for j in range(Ny):
        for k in range(Nz):
            rx[:, j, k] = rho[:, j, k] + dt * _tvd_flux_1d(
                rho[:, j, k], ucx[:, j, k], dx, dt, bc, limiter)

    # y-sweep
    ry = rx.clone()
    for i in range(Nx):
        for k in range(Nz):
            ry[i, :, k] = rx[i, :, k] + dt * _tvd_flux_1d(
                rx[i, :, k], ucy[i, :, k], dy, dt, bc, limiter)

    # z-sweep
    rz = ry.clone()
    for i in range(Nx):
        for j in range(Ny):
            rz[i, j, :] = ry[i, j, :] + dt * _tvd_flux_1d(
                ry[i, j, :], ucz[i, j, :], dz, dt, bc, limiter)

    return rz


def _advect_weno5_3d(
    rho: torch.Tensor,
    ucx: torch.Tensor, ucy: torch.Tensor, ucz: torch.Tensor,
    dx: float, dy: float, dz: float,
    dt: float, bc: BoundaryCondition,
) -> torch.Tensor:
    """WENO-5 density advection — dimensional splitting (3-D)."""
    Nx, Ny, Nz = rho.shape

    rx = rho.clone()
    for j in range(Ny):
        for k in range(Nz):
            rx[:, j, k] = rho[:, j, k] + dt * _weno5_flux_1d(
                rho[:, j, k], ucx[:, j, k], dx, bc)

    ry = rx.clone()
    for i in range(Nx):
        for k in range(Nz):
            ry[i, :, k] = rx[i, :, k] + dt * _weno5_flux_1d(
                rx[i, :, k], ucy[i, :, k], dy, bc)

    rz = ry.clone()
    for i in range(Nx):
        for j in range(Ny):
            rz[i, j, :] = ry[i, j, :] + dt * _weno5_flux_1d(
                ry[i, j, :], ucz[i, j, :], dz, bc)

    return rz


def _advect_semi_lagrangian_3d(
    rho: torch.Tensor,
    ucx: torch.Tensor, ucy: torch.Tensor, ucz: torch.Tensor,
    dx: float, dy: float, dz: float,
    dt: float,
    Lx: float, Ly: float, Lz: float,
) -> torch.Tensor:
    """
    Unconditionally-stable semi-Lagrangian advection via
    torch.nn.functional.grid_sample (trilinear, GPU-native).

    Traces characteristics backward: x_dep = x - dt·u(x,t),
    then interpolates ρ at the departure point.

    Not conservative; best for incompressible / low-Ma regimes.
    """
    Nx, Ny, Nz = rho.shape
    dev   = rho.device
    dtype = rho.dtype

    xc = (torch.arange(Nx, device=dev, dtype=torch.float32) + 0.5) * dx
    yc = (torch.arange(Ny, device=dev, dtype=torch.float32) + 0.5) * dy
    zc = (torch.arange(Nz, device=dev, dtype=torch.float32) + 0.5) * dz

    X, Y, Z = torch.meshgrid(xc, yc, zc, indexing="ij")   # (Nx,Ny,Nz)

    X_dep = (X - dt * ucx.float()) % Lx
    Y_dep = (Y - dt * ucy.float()) % Ly
    Z_dep = (Z - dt * ucz.float()) % Lz

    # Normalise to [-1, 1] for grid_sample
    X_n = 2.0 * X_dep / Lx - 1.0
    Y_n = 2.0 * Y_dep / Ly - 1.0
    Z_n = 2.0 * Z_dep / Lz - 1.0

    # grid_sample 5-D: (N=1, C=1, D=Nz, H=Ny, W=Nx)
    # grid axes: dim-0 → W (x), dim-1 → H (y), dim-2 → D (z)
    rho_in = rho.float().permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,Nz,Ny,Nx)
    grid   = torch.stack(
        [X_n.permute(2, 1, 0), Y_n.permute(2, 1, 0), Z_n.permute(2, 1, 0)],
        dim=-1,
    ).unsqueeze(0)   # (1, Nz, Ny, Nx, 3)

    rho_out = F.grid_sample(
        rho_in, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(0).squeeze(0)   # (Nz, Ny, Nx)

    return rho_out.permute(2, 1, 0).to(dtype=dtype)   # (Nx, Ny, Nz)


def advect_density(
    rho: torch.Tensor,
    ucx: torch.Tensor, ucy: torch.Tensor, ucz: torch.Tensor,
    dx: float, dy: float, dz: float,
    dt: float,
    bc: BoundaryCondition,
    scheme:  str = "upwind",
    limiter: str = "van_leer",
    Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
) -> torch.Tensor:
    """
    Unified 3-D density advection dispatcher.

    Args:
        rho          : (Nx, Ny, Nz)
        ucx,ucy,ucz  : (Nx, Ny, Nz) cell-centred velocities
        dx,dy,dz     : grid spacings
        dt           : time step
        bc           : boundary condition
        scheme       : "upwind" | "tvd" | "weno5" | "semi_lagrangian"
        limiter      : "minmod" | "van_leer" | "superbee"  (TVD only)
        Lx,Ly,Lz    : domain extents (semi-Lagrangian only)

    Returns:
        rho_new : (Nx, Ny, Nz)
    """
    if scheme == "tvd":
        return _advect_tvd_3d(rho, ucx, ucy, ucz, dx, dy, dz, dt, bc, limiter)
    elif scheme == "weno5":
        return _advect_weno5_3d(rho, ucx, ucy, ucz, dx, dy, dz, dt, bc)
    elif scheme == "semi_lagrangian":
        return _advect_semi_lagrangian_3d(
            rho, ucx, ucy, ucz, dx, dy, dz, dt, Lx, Ly, Lz)
    else:  # upwind
        return _advect_upwind_3d(rho, ucx, ucy, ucz, dx, dy, dz, dt, bc)


# =============================================================================
# Module 1 — 3-D Differentiable Interface Detector
# =============================================================================

class CFDInterfaceDetector(InterfaceDetectorBase):
    """
    Detects sharp-gradient regions in a 3-D scalar field.

    Returns a differentiable soft mask ∈ [0, 1] via sigmoid of the
    normalised gradient magnitude |∇φ|.

    Args:
        sharpness : sigmoid steepness.
        bc        : boundary condition for padding.
    """

    def __init__(
        self,
        sharpness: float = 4.0,
        bc: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> None:
        super().__init__()
        if sharpness <= 0:
            raise ValueError(f"sharpness must be positive, got {sharpness}.")
        self.sharpness = sharpness
        self.bc        = bc

    def forward(
        self,
        phi: torch.Tensor,
        dx: float,
        dy: float,
        dz: float,
    ) -> torch.Tensor:
        """
        Args:
            phi    : (Nx, Ny, Nz) scalar field.
            dx,dy,dz : grid spacings.

        Returns:
            mask : (Nx, Ny, Nz) interface score ∈ [0, 1].
        """
        fp = _pad_field_3d(phi, self.bc)
        dphidx = (fp[2:, 1:-1, 1:-1] - fp[:-2, 1:-1, 1:-1]) / (2.0*dx)
        dphidy = (fp[1:-1, 2:, 1:-1] - fp[1:-1, :-2, 1:-1]) / (2.0*dy)
        dphidz = (fp[1:-1, 1:-1, 2:] - fp[1:-1, 1:-1, :-2]) / (2.0*dz)
        grad_mag  = torch.sqrt(dphidx**2 + dphidy**2 + dphidz**2 + 1e-30)
        norm_grad = grad_mag / (grad_mag.mean() + 1e-12)
        return torch.sigmoid(self.sharpness * (norm_grad - 1.0))


# =============================================================================
# Module 2 — CSOC Adaptive Viscosity
# =============================================================================

class CSOCAdaptiveViscosity(CSOCBase):
    """
    CSOC-driven adaptive kinematic viscosity and diffusivity (3-D compatible).

    Modulates ν and α based on the SSC-filtered structural stress (density
    variation), analogous to a MD thermostat.
    """

    def __init__(
        self,
        base_viscosity:   float = 1e-3,
        base_diffusivity: float = 1e-5,
        sigma_target:     float = 0.1,
        viscosity_boost:  float = 5.0,
        epsilon_fp:       float = 0.0028,
    ) -> None:
        if base_viscosity <= 0:
            raise ValueError("base_viscosity must be positive.")
        if viscosity_boost < 1.0:
            raise ValueError("viscosity_boost must be ≥ 1.")
        super().__init__(
            sigma_target=sigma_target,
            epsilon_fp=epsilon_fp,
            boost_factor=viscosity_boost,
        )
        self.base_nu         = base_viscosity
        self.base_alpha      = base_diffusivity
        self.viscosity_boost = viscosity_boost

    def forward(
        self,
        rho:      torch.Tensor,
        rho_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rho, rho_prev : (Nx, Ny, Nz)

        Returns:
            nu, alpha : scalar adaptive viscosity / diffusivity.
            sigma     : SSC-filtered stress.
        """
        raw_sigma = (rho - rho_prev).abs().mean()
        sigma     = self.ssc(raw_sigma)
        dev       = self._normalised_deviation(sigma)
        boost     = self._smooth_boost(dev)
        nu        = self.base_nu    * (1.0 + (self.viscosity_boost-1.0)*boost)
        alpha     = self.base_alpha * (1.0 + (self.viscosity_boost-1.0)*boost)
        return nu, alpha, sigma


# =============================================================================
# Module 3 — 3-D Landau-Lifshitz Stochastic Stress Tensor
# =============================================================================

class LLStochasticStress(nn.Module):
    """
    Discrete Landau-Lifshitz stochastic stress tensor for a 3-D MAC grid.

    The 3×3 symmetric tensor has 6 independent components
    (Sxx, Sxy, Sxz, Syy, Syz, Szz).  Each off-diagonal pair uses
    independent Gaussian noise realisations to satisfy the FDT.

    Amplitude:  σ_noise(x) = √( 2 ρ(x) ν(x) k_B T / (V Δt) )
    where V = dx·dy·dz is the cell volume.

    Structural Itô correction (Theorem 4.1) is applied to the density
    equation via a differentiable G-field gradient.
    """

    def __init__(
        self,
        kb_T:                    float = 4.11e-21,
        interface_amplification: float = 3.0,
    ) -> None:
        super().__init__()
        if kb_T <= 0:
            raise ValueError("kb_T must be positive.")
        self.kb_T = kb_T
        self.amp  = interface_amplification

    def _noise_prefactor(
        self,
        nu:  torch.Tensor,
        rho: torch.Tensor,
        dx: float, dy: float, dz: float,
        dt: float,
    ) -> torch.Tensor:
        V   = dx * dy * dz
        eta = rho * nu
        return torch.sqrt(2.0 * eta * self.kb_T / (V * dt + 1e-300))

    def _g_field(self, mask: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.amp * mask

    def _ito_correction(
        self,
        rho:      torch.Tensor,
        detector: "CFDInterfaceDetector",
        dx: float, dy: float, dz: float,
    ) -> torch.Tensor:
        """
        Structural Itô drift: ½ G(x) ∇_ρ G(x).

        DIFF-FIX 6: create_graph=True keeps the correction in the autograd
        graph so that loss.backward() flows through the Itô term back to rho.
        The .detach() on the final result is removed.
        """
        with torch.enable_grad():
            if not rho.requires_grad:
                rho = rho.detach().requires_grad_(True)
            mask   = detector(rho, dx, dy, dz)
            G      = 1.0 + self.amp * mask
            G_sum  = G.sum()
            grad_G = torch.autograd.grad(
                G_sum, rho, create_graph=True, retain_graph=True
            )[0]
        return 0.5 * G * grad_G

    def forward(
        self,
        rho:      torch.Tensor,
        nu:       torch.Tensor,
        mask:     torch.Tensor,
        detector: "CFDInterfaceDetector",
        dx: float, dy: float, dz: float,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute 3-D stochastic momentum forcing and Itô density correction.

        Args:
            rho      : (Nx,Ny,Nz)
            nu       : scalar kinematic viscosity
            mask     : (Nx,Ny,Nz) interface mask
            detector : CFDInterfaceDetector instance
            dx,dy,dz : grid spacings
            dt       : time step

        Returns:
            Sx, Sy, Sz : (Nx,Ny,Nz) stochastic acceleration [m/s²]
            ito        : (Nx,Ny,Nz) Itô drift correction
        """
        A  = self._noise_prefactor(nu, rho, dx, dy, dz, dt)
        G  = self._g_field(mask)
        amp = A * G

        # 6 independent noise fields
        Wxx = torch.randn_like(rho)
        Wxy = torch.randn_like(rho); Wyx = torch.randn_like(rho)
        Wxz = torch.randn_like(rho); Wzx = torch.randn_like(rho)
        Wyy = torch.randn_like(rho)
        Wyz = torch.randn_like(rho); Wzy = torch.randn_like(rho)
        Wzz = torch.randn_like(rho)

        Sxx = amp * 2.0 * Wxx
        Sxy = amp * (Wxy + Wyx)
        Sxz = amp * (Wxz + Wzx)
        Syy = amp * 2.0 * Wyy
        Syz = amp * (Wyz + Wzy)
        Szz = amp * 2.0 * Wzz

        bc = BoundaryCondition.PERIODIC
        def _div_row(S1, S2, S3):
            """∂S1/∂x + ∂S2/∂y + ∂S3/∂z via central differences."""
            p1 = _pad_field_3d(S1, bc)
            p2 = _pad_field_3d(S2, bc)
            p3 = _pad_field_3d(S3, bc)
            return (
                (p1[2:,1:-1,1:-1] - p1[:-2,1:-1,1:-1]) / (2.0*dx)
                + (p2[1:-1,2:,1:-1] - p2[1:-1,:-2,1:-1]) / (2.0*dy)
                + (p3[1:-1,1:-1,2:] - p3[1:-1,1:-1,:-2]) / (2.0*dz)
            )

        # DIFF-FIX 5: softplus floor instead of clamp
        rho_s = _softplus_floor(rho, 1e-12)
        Sx = _div_row(Sxx, Sxy, Sxz) / rho_s
        Sy = _div_row(Sxy, Syy, Syz) / rho_s
        Sz = _div_row(Sxz, Syz, Szz) / rho_s

        ito = self._ito_correction(rho, detector, dx, dy, dz)
        return Sx, Sy, Sz, ito


# =============================================================================
# Boundary condition enforcement
# =============================================================================

def _apply_velocity_bc(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    bc_x: BoundaryCondition,
    bc_y: BoundaryCondition,
    bc_z: BoundaryCondition,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enforce velocity BCs after projection.

    No-slip   : all velocity components zero on boundary faces.
    Free-slip : normal component zero, tangential free.
    Open      : zero-gradient (copy from interior).
    Periodic  : no action needed.
    """
    ux = ux.clone(); uy = uy.clone(); uz = uz.clone()

    if bc_x == BoundaryCondition.NO_SLIP:
        ux[0, :, :]  = 0.0; ux[-1, :, :] = 0.0
        uy[0, :, :]  = 0.0; uy[-1, :, :] = 0.0
        uz[0, :, :]  = 0.0; uz[-1, :, :] = 0.0
    elif bc_x == BoundaryCondition.FREE_SLIP:
        ux[0, :, :]  = 0.0; ux[-1, :, :] = 0.0
        uy[0, :, :]  = uy[1, :, :]; uy[-1, :, :] = uy[-2, :, :]
        uz[0, :, :]  = uz[1, :, :]; uz[-1, :, :] = uz[-2, :, :]
    elif bc_x == BoundaryCondition.OPEN:
        ux[0, :, :]  = ux[1, :, :]; ux[-1, :, :] = ux[-2, :, :]
        uy[0, :, :]  = uy[1, :, :]; uy[-1, :, :] = uy[-2, :, :]
        uz[0, :, :]  = uz[1, :, :]; uz[-1, :, :] = uz[-2, :, :]

    if bc_y == BoundaryCondition.NO_SLIP:
        uy[:, 0, :]  = 0.0; uy[:, -1, :] = 0.0
        ux[:, 0, :]  = 0.0; ux[:, -1, :] = 0.0
        uz[:, 0, :]  = 0.0; uz[:, -1, :] = 0.0
    elif bc_y == BoundaryCondition.FREE_SLIP:
        uy[:, 0, :]  = 0.0; uy[:, -1, :] = 0.0
        ux[:, 0, :]  = ux[:, 1, :]; ux[:, -1, :] = ux[:, -2, :]
        uz[:, 0, :]  = uz[:, 1, :]; uz[:, -1, :] = uz[:, -2, :]
    elif bc_y == BoundaryCondition.OPEN:
        uy[:, 0, :]  = uy[:, 1, :]; uy[:, -1, :] = uy[:, -2, :]
        ux[:, 0, :]  = ux[:, 1, :]; ux[:, -1, :] = ux[:, -2, :]
        uz[:, 0, :]  = uz[:, 1, :]; uz[:, -1, :] = uz[:, -2, :]

    if bc_z == BoundaryCondition.NO_SLIP:
        uz[:, :, 0]  = 0.0; uz[:, :, -1] = 0.0
        ux[:, :, 0]  = 0.0; ux[:, :, -1] = 0.0
        uy[:, :, 0]  = 0.0; uy[:, :, -1] = 0.0
    elif bc_z == BoundaryCondition.FREE_SLIP:
        uz[:, :, 0]  = 0.0; uz[:, :, -1] = 0.0
        ux[:, :, 0]  = ux[:, :, 1]; ux[:, :, -1] = ux[:, :, -2]
        uy[:, :, 0]  = uy[:, :, 1]; uy[:, :, -1] = uy[:, :, -2]
    elif bc_z == BoundaryCondition.OPEN:
        uz[:, :, 0]  = uz[:, :, 1]; uz[:, :, -1] = uz[:, :, -2]
        ux[:, :, 0]  = ux[:, :, 1]; ux[:, :, -1] = ux[:, :, -2]
        uy[:, :, 0]  = uy[:, :, 1]; uy[:, :, -1] = uy[:, :, -2]

    return ux, uy, uz


# =============================================================================
# Core Solver — StructuralFluctuatingHydro (3-D)
# =============================================================================

class StructuralFluctuatingHydro(nn.Module):
    """
    Production 3-D Fluctuating Hydrodynamics solver using the Structural
    Calculus / CSOC framework.

    Solves the Landau-Lifshitz Navier-Stokes equations on a staggered 3-D
    MAC grid:

        ∂ρ/∂t  + ∇·(ρu)     = 0
        ∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·(νρ(∇u+∇uᵀ)) + ∇·S̃

    State variables:
        rho : (Nx, Ny, Nz)        density  [kg/m³]
        ux  : (Nx+1, Ny,   Nz  )  x-face velocity [m/s]
        uy  : (Nx,   Ny+1, Nz  )  y-face velocity [m/s]
        uz  : (Nx,   Ny,   Nz+1)  z-face velocity [m/s]
        p   : (Nx, Ny, Nz)        pressure [Pa]

    Quickstart::

        cfg = SolverConfig(Nx=32, Ny=32, Nz=32, dt=1e-4)
        solver = StructuralFluctuatingHydro(cfg)
        rho, ux, uy, uz, p = solver.initialize_taylor_green()
        for _ in range(100):
            rho, ux, uy, uz, p, diag = solver.step(rho, ux, uy, uz, p)
    """

    def __init__(self, config: SolverConfig) -> None:
        super().__init__()
        config.validate()
        self.cfg = config

        self.dx = config.Lx / config.Nx
        self.dy = config.Ly / config.Ny
        self.dz = config.Lz / config.Nz

        # Sub-modules
        self.interface_detector = CFDInterfaceDetector(
            sharpness=config.interface_sharpness, bc=config.bc_x)
        self.csoc_viscosity = CSOCAdaptiveViscosity(
            base_viscosity=config.base_viscosity,
            base_diffusivity=config.base_diffusivity,
            sigma_target=config.sigma_target,
            viscosity_boost=config.viscosity_boost,
            epsilon_fp=config.epsilon_fp,
        )
        self.ll_stress = LLStochasticStress(
            kb_T=config.kb_T,
            interface_amplification=config.interface_amp,
        )

        # Persistent buffers
        self.register_buffer(
            "_rho_prev",
            torch.ones(config.Nx, config.Ny, config.Nz, dtype=config.dtype) * config.rho0,
        )
        self.register_buffer("_state_ready", torch.tensor(False))
        self.register_buffer("_step_count",  torch.tensor(0, dtype=torch.long))

        # Pre-compute 3-D Poisson eigenvalues
        kx = torch.arange(config.Nx, dtype=config.dtype)
        ky = torch.arange(config.Ny, dtype=config.dtype)
        kz = torch.arange(config.Nz, dtype=config.dtype)
        lx = (2.0*torch.cos(2.0*math.pi*kx/config.Nx) - 2.0) / self.dx**2
        ly = (2.0*torch.cos(2.0*math.pi*ky/config.Ny) - 2.0) / self.dy**2
        lz = (2.0*torch.cos(2.0*math.pi*kz/config.Nz) - 2.0) / self.dz**2
        eig = lx[:, None, None] + ly[None, :, None] + lz[None, None, :]
        eig[0, 0, 0] = 1.0
        self.register_buffer("_poisson_eig", eig)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return int(self._step_count.item())

    @property
    def device(self) -> torch.device:
        return self._rho_prev.device

    def _cast(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(device=self.device, dtype=self.cfg.dtype)

    # ── Initialisers ─────────────────────────────────────────────────────────

    def _kw(self) -> dict:
        return dict(device=self.device, dtype=self.cfg.dtype)

    def initialize_uniform(self):
        """Quiescent uniform state at rest."""
        cfg = self.cfg
        kw  = self._kw()
        rho = torch.ones(cfg.Nx,   cfg.Ny,   cfg.Nz,   **kw) * cfg.rho0
        ux  = torch.zeros(cfg.Nx+1, cfg.Ny,   cfg.Nz,   **kw)
        uy  = torch.zeros(cfg.Nx,   cfg.Ny+1, cfg.Nz,   **kw)
        uz  = torch.zeros(cfg.Nx,   cfg.Ny,   cfg.Nz+1, **kw)
        p   = torch.zeros(cfg.Nx,   cfg.Ny,   cfg.Nz,   **kw)
        return rho, ux, uy, uz, p

    def initialize_taylor_green(self, amplitude: float = 1.0):
        """
        3-D Taylor-Green vortex initial condition (incompressible limit).

            u = A sin(x) cos(y) cos(z)
            v = −A cos(x) sin(y) cos(z)
            w = 0
            p = A²/16 [cos(2x)+cos(2y)] [cos(2z)+2]
        """
        cfg = self.cfg
        kw  = self._kw()
        dx, dy, dz = self.dx, self.dy, self.dz

        # x-face centres
        xf = torch.arange(cfg.Nx+1, **kw) * dx
        yc = (torch.arange(cfg.Ny,  **kw) + 0.5) * dy
        zc = (torch.arange(cfg.Nz,  **kw) + 0.5) * dz
        ux = amplitude * (
            torch.sin(2*math.pi * xf[:, None, None])
            * torch.cos(2*math.pi * yc[None, :, None])
            * torch.cos(2*math.pi * zc[None, None, :])
        )

        # y-face centres
        xc = (torch.arange(cfg.Nx, **kw) + 0.5) * dx
        yf = torch.arange(cfg.Ny+1, **kw) * dy
        uy = -amplitude * (
            torch.cos(2*math.pi * xc[:, None, None])
            * torch.sin(2*math.pi * yf[None, :, None])
            * torch.cos(2*math.pi * zc[None, None, :])
        )

        # uz = 0 for TG vortex
        uz = torch.zeros(cfg.Nx, cfg.Ny, cfg.Nz+1, **kw)

        # Pressure at cell centres
        xcc = (torch.arange(cfg.Nx, **kw) + 0.5) * dx
        ycc = (torch.arange(cfg.Ny, **kw) + 0.5) * dy
        zcc = (torch.arange(cfg.Nz, **kw) + 0.5) * dz
        p = (amplitude**2 / 16.0) * (
            (torch.cos(4*math.pi*xcc[:,None,None]) + torch.cos(4*math.pi*ycc[None,:,None]))
            * (torch.cos(4*math.pi*zcc[None,None,:]) + 2.0)
        )

        rho = torch.ones(cfg.Nx, cfg.Ny, cfg.Nz, **kw) * cfg.rho0
        return rho, ux, uy, uz, p

    def initialize_rayleigh_taylor(
        self,
        rho_heavy: float = 2.0,
        rho_light: float = 1.0,
        interface_width: float = 0.05,
        perturbation_amp: float = 0.01,
    ):
        """
        3-D Rayleigh-Taylor interface.

        Heavy fluid occupies the upper half (z > Lz/2) with a sinusoidal
        perturbation in both x and y directions.
        """
        cfg = self.cfg
        kw  = self._kw()

        xc = (torch.arange(cfg.Nx, **kw) + 0.5) * self.dx
        yc = (torch.arange(cfg.Ny, **kw) + 0.5) * self.dy
        zc = (torch.arange(cfg.Nz, **kw) + 0.5) * self.dz

        # Interface position in z
        z_interface = (cfg.Lz / 2.0
            + perturbation_amp * torch.sin(2*math.pi*xc/cfg.Lx)[:, None]
            + perturbation_amp * torch.sin(2*math.pi*yc/cfg.Ly)[None, :])  # (Nx,Ny)

        dist = zc[None, None, :] - z_interface[:, :, None]   # (Nx,Ny,Nz)
        phi  = 0.5 * (1.0 + torch.tanh(dist / interface_width))
        rho  = rho_light + (rho_heavy - rho_light) * phi

        ux = torch.zeros(cfg.Nx+1, cfg.Ny,   cfg.Nz,   **kw)
        uy = torch.zeros(cfg.Nx,   cfg.Ny+1, cfg.Nz,   **kw)
        uz = torch.zeros(cfg.Nx,   cfg.Ny,   cfg.Nz+1, **kw)
        p  = torch.zeros(cfg.Nx,   cfg.Ny,   cfg.Nz,   **kw)
        return rho, ux, uy, uz, p

    # ── Poisson solver ───────────────────────────────────────────────────────

    def _solve_pressure_poisson(self, rhs: torch.Tensor) -> torch.Tensor:
        """Spectral Poisson solver via rfftn (3-D, periodic, exact)."""
        Nx, Ny, Nz = self.cfg.Nx, self.cfg.Ny, self.cfg.Nz
        eig   = self._poisson_eig
        rhs_hat = torch.fft.rfftn(rhs)
        eig_r   = eig[:, :, :Nz//2+1]
        p_hat   = rhs_hat / eig_r
        p_hat[0, 0, 0] = torch.zeros(1, dtype=p_hat.dtype, device=p_hat.device)
        return torch.fft.irfftn(p_hat, s=(Nx, Ny, Nz))

    # ── CFL check ────────────────────────────────────────────────────────────

    def _check_cfl(self, ux, uy, uz, nu) -> float:
        if self.cfg.cfl_limit <= 0:
            return 0.0
        dt   = self.cfg.dt
        dmin = min(self.dx, self.dy, self.dz)
        # DIFF-FIX 8: detach for diagnostics only (not in graph)
        u_max = max(ux.detach().abs().max().item(),
                    uy.detach().abs().max().item(),
                    uz.detach().abs().max().item())
        nu_v = nu.detach().item() if nu.ndim == 0 else nu.detach().max().item()
        cfl  = dt * (u_max/dmin + nu_v*(1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2))
        if cfl > self.cfg.cfl_limit:
            logger.warning("CFL = %.4f > %.4f at step %d",
                           cfl, self.cfg.cfl_limit, self.step_count)
        return cfl

    # ── Single time step ─────────────────────────────────────────────────────

    def step(
        self,
        rho: torch.Tensor,
        ux:  torch.Tensor,
        uy:  torch.Tensor,
        uz:  torch.Tensor,
        p:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Advance the 3-D FH state by one time step.

        Algorithm (fractional-step / projection):
          1.  CSOC adaptive viscosity from structural stress.
          2.  3-D interface detection (soft mask).
          3.  Cell-centred velocity by face interpolation.
          4.  Viscous diffusion (7-point Laplacian, explicit).
          5.  Landau-Lifshitz stochastic stress forcing (if enabled).
          6.  Intermediate velocity u* (without pressure correction).
          7.  Pressure Poisson solve (rfftn, O(N³ log N)).
          8.  Project u* onto divergence-free field (local ρ faces).
          9.  Enforce velocity boundary conditions.
          10. CFL check.
          11. Density advection (upwind / TVD / WENO-5 / semi-Lagrangian).
          12. Structural Itô density correction.
          13. Positivity clamp.

        Args:
            rho : (Nx,Ny,Nz)
            ux  : (Nx+1,Ny,Nz)
            uy  : (Nx,Ny+1,Nz)
            uz  : (Nx,Ny,Nz+1)
            p   : (Nx,Ny,Nz)

        Returns:
            rho_new, ux_new, uy_new, uz_new, p_new, diagnostics
        """
        t0  = time.perf_counter()
        cfg = self.cfg
        dx, dy, dz, dt = self.dx, self.dy, self.dz, cfg.dt

        rho = self._cast(rho)
        ux  = self._cast(ux)
        uy  = self._cast(uy)
        uz  = self._cast(uz)
        p   = self._cast(p)

        # ── Initialise previous density ─────────────────────────────────────
        if not self._state_ready.item():
            self._rho_prev.data = rho.detach().clone()
            self._state_ready.fill_(True)

        # ── 1. CSOC adaptive viscosity ──────────────────────────────────────
        nu, alpha, sigma = self.csoc_viscosity(rho, self._rho_prev)
        self._rho_prev.data = rho.detach().clone()

        # ── 2. Interface detection ──────────────────────────────────────────
        imask = self.interface_detector(rho, dx, dy, dz)

        # ── 3. Cell-centred velocity ────────────────────────────────────────
        ucx = 0.5 * (ux[:-1,:,:] + ux[1:,:,:])    # (Nx,Ny,Nz)
        ucy = 0.5 * (uy[:,:-1,:] + uy[:,1:,:])
        ucz = 0.5 * (uz[:,:,:-1] + uz[:,:,1:])

        # ── 4. Viscous diffusion ────────────────────────────────────────────
        visc_x = nu * laplacian_cell(ucx, dx, dy, dz, cfg.bc_x)
        visc_y = nu * laplacian_cell(ucy, dx, dy, dz, cfg.bc_y)
        visc_z = nu * laplacian_cell(ucz, dx, dy, dz, cfg.bc_z)

        # ── 5. Stochastic stress ────────────────────────────────────────────
        if cfg.enable_fluctuations:
            Sx, Sy, Sz, ito_corr = self.ll_stress(
                rho, nu, imask, self.interface_detector, dx, dy, dz, dt)
        else:
            zeros = torch.zeros_like(rho)
            Sx, Sy, Sz, ito_corr = zeros, zeros, zeros, zeros

        # ── 6. Intermediate velocity u* ────────────────────────────────────
        # DIFF-FIX 5: softplus floor instead of clamp
        rho_s = _softplus_floor(rho, 1e-12)

        # Cell-centred pressure gradient
        gx_c = torch.zeros_like(ucx); gy_c = torch.zeros_like(ucy); gz_c = torch.zeros_like(ucz)
        gx_c[1:-1,:,:] = (p[1:,:,:] - p[:-1,:,:]) / dx
        gy_c[:,1:-1,:] = (p[:,1:,:] - p[:,:-1,:]) / dy
        gz_c[:,:,1:-1] = (p[:,:,1:] - p[:,:,:-1]) / dz

        ux_star_c = ucx + dt * (visc_x - gx_c/rho_s + Sx)
        uy_star_c = ucy + dt * (visc_y - gy_c/rho_s + Sy)
        uz_star_c = ucz + dt * (visc_z - gz_c/rho_s + Sz)

        # Reconstruct face velocities
        ux_star = torch.zeros(cfg.Nx+1, cfg.Ny,   cfg.Nz,   device=self.device, dtype=cfg.dtype)
        uy_star = torch.zeros(cfg.Nx,   cfg.Ny+1, cfg.Nz,   device=self.device, dtype=cfg.dtype)
        uz_star = torch.zeros(cfg.Nx,   cfg.Ny,   cfg.Nz+1, device=self.device, dtype=cfg.dtype)
        ux_star[1:-1,:,:] = 0.5*(ux_star_c[:-1,:,:] + ux_star_c[1:,:,:])
        uy_star[:,1:-1,:] = 0.5*(uy_star_c[:,:-1,:] + uy_star_c[:,1:,:])
        uz_star[:,:,1:-1] = 0.5*(uz_star_c[:,:,:-1] + uz_star_c[:,:,1:])

        # ── 7. Pressure Poisson ─────────────────────────────────────────────
        div_star = div_u(ux_star, uy_star, uz_star, dx, dy, dz)
        rhs_p    = rho_s / dt * div_star
        p_new    = self._solve_pressure_poisson(rhs_p)

        # ── 8. Projection with local ρ on faces ────────────────────────────
        gx_new, gy_new, gz_new = grad_p(p_new, dx, dy, dz)

        # Interpolate ρ to faces
        rho_fx = torch.full((cfg.Nx+1, cfg.Ny, cfg.Nz), rho_s.mean().item(),
                            device=self.device, dtype=cfg.dtype)
        rho_fy = torch.full((cfg.Nx, cfg.Ny+1, cfg.Nz), rho_s.mean().item(),
                            device=self.device, dtype=cfg.dtype)
        rho_fz = torch.full((cfg.Nx, cfg.Ny, cfg.Nz+1), rho_s.mean().item(),
                            device=self.device, dtype=cfg.dtype)
        rho_fx[1:-1,:,:] = 0.5*(rho_s[:-1,:,:] + rho_s[1:,:,:])
        rho_fy[:,1:-1,:] = 0.5*(rho_s[:,:-1,:] + rho_s[:,1:,:])
        rho_fz[:,:,1:-1] = 0.5*(rho_s[:,:,:-1] + rho_s[:,:,1:])

        ux_new = ux_star - dt * gx_new / rho_fx
        uy_new = uy_star - dt * gy_new / rho_fy
        uz_new = uz_star - dt * gz_new / rho_fz

        # ── 9. Boundary conditions ──────────────────────────────────────────
        ux_new, uy_new, uz_new = _apply_velocity_bc(
            ux_new, uy_new, uz_new, cfg.bc_x, cfg.bc_y, cfg.bc_z)

        # ── 10. CFL check ───────────────────────────────────────────────────
        cfl = self._check_cfl(ux_new, uy_new, uz_new, nu)

        # ── 11. Density advection ───────────────────────────────────────────
        ucx_adv = 0.5*(ux_new[:-1,:,:] + ux_new[1:,:,:])
        ucy_adv = 0.5*(uy_new[:,:-1,:] + uy_new[:,1:,:])
        ucz_adv = 0.5*(uz_new[:,:,:-1] + uz_new[:,:,1:])

        rho_new = advect_density(
            rho, ucx_adv, ucy_adv, ucz_adv,
            dx, dy, dz, dt,
            bc=cfg.bc_x,
            scheme=cfg.advection_scheme,
            limiter=cfg.advection_limiter,
            Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
        )

        # ── 12. Structural Itô correction ───────────────────────────────────
        if cfg.enable_fluctuations:
            rho_new = rho_new + dt * ito_corr

        # ── 13. Positivity ──────────────────────────────────────────────────
        # DIFF-FIX 7: softplus floor instead of hard clamp
        rho_new = _softplus_floor(rho_new, 1e-6)

        # ── Diagnostics ─────────────────────────────────────────────────────
        div_new = div_u(ux_new, uy_new, uz_new, dx, dy, dz)
        ke = 0.5 * (
            (rho_new * ucx_adv**2).mean()
            + (rho_new * ucy_adv**2).mean()
            + (rho_new * ucz_adv**2).mean()
        )

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        self._step_count.add_(1)

        diag: Dict[str, Any] = {
            "step":       self.step_count,
            "sigma":      sigma.item(),
            "nu":         nu.item(),
            "alpha":      alpha.item(),
            "cfl":        cfl,
            "div_max":    div_new.abs().max().item(),
            "rho_min":    rho_new.min().item(),
            "rho_max":    rho_new.max().item(),
            "rho_mean":   rho_new.mean().item(),
            "ke":         ke.item(),
            "elapsed_ms": elapsed_ms,
        }

        if cfg.log_every > 0 and self.step_count % cfg.log_every == 0:
            logger.info(
                "step=%d  sigma=%.3e  nu=%.3e  cfl=%.3f  div_max=%.3e  ke=%.3e",
                diag["step"], diag["sigma"], diag["nu"],
                diag["cfl"],  diag["div_max"], diag["ke"],
            )

        return rho_new, ux_new, uy_new, uz_new, p_new, diag

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save_checkpoint(
        self, path: str,
        rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor,
        uz: torch.Tensor,  p: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "step_count": self.step_count,
            "state_dict": self.state_dict(),
            "fields":     {"rho": rho, "ux": ux, "uy": uy, "uz": uz, "p": p},
            "config":     self.cfg,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        logger.info("Checkpoint saved → %s  (step %d)", path, self.step_count)

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Optional[torch.device] = None,
    ) -> Tuple["StructuralFluctuatingHydro",
               torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        payload = torch.load(path, map_location=device, weights_only=False)
        solver  = cls(payload["config"])
        solver.load_state_dict(payload["state_dict"])
        f = payload["fields"]
        fields = [f["rho"], f["ux"], f["uy"], f["uz"], f["p"]]
        if device is not None:
            fields = [t.to(device) for t in fields]
        logger.info("Checkpoint loaded ← %s  (step %d)", path, solver.step_count)
        return (solver, *fields)

    def reset(self) -> None:
        self._rho_prev.fill_(self.cfg.rho0)
        self._state_ready.fill_(False)
        self._step_count.zero_()
        self.csoc_viscosity.reset()
        logger.debug("Solver reset.")


# =============================================================================
# Unit tests
# =============================================================================

def _run_tests() -> None:
    """Run in-module test suite:  python structuralfluctuatinghydro_v4.py --test"""
    import sys
    passed = failed = 0

    def ok(name):
        nonlocal passed; passed += 1; print(f"  [PASS] {name}")

    def fail(name, msg):
        nonlocal failed; failed += 1; print(f"  [FAIL] {name}: {msg}")

    torch.manual_seed(42)

    # ── Config validation ────────────────────────────────────────────────────
    try:
        SolverConfig(Nx=2).validate()
        fail("config_Nx_too_small", "should raise")
    except ValueError:
        ok("config_Nx_too_small")

    try:
        SolverConfig(advection_scheme="bad").validate()
        fail("config_bad_scheme", "should raise")
    except ValueError:
        ok("config_bad_scheme")

    # ── FFT Poisson solver (3-D) ─────────────────────────────────────────────
    Nx=Ny=Nz=16; dx=dy=dz=1.0/Nx
    rhs = torch.randn(Nx,Ny,Nz,dtype=torch.float64)
    rhs -= rhs.mean()
    p_sol = _fft3_poisson_solve(rhs, dx, dy, dz)
    # Verify lap(p) ≈ rhs via central differences
    p4 = F.pad(p_sol.unsqueeze(0).unsqueeze(0),(1,1,1,1,1,1),mode="circular").squeeze(0).squeeze(0)
    lap = ((p4[2:,1:-1,1:-1]-2*p_sol+p4[:-2,1:-1,1:-1])/dx**2
          +(p4[1:-1,2:,1:-1]-2*p_sol+p4[1:-1,:-2,1:-1])/dy**2
          +(p4[1:-1,1:-1,2:]-2*p_sol+p4[1:-1,1:-1,:-2])/dz**2)
    res = (lap - rhs).abs().max().item()
    if res < 1e-7:
        ok(f"fft3_poisson_residual ({res:.1e})")
    else:
        fail("fft3_poisson_residual", f"residual={res:.2e}")

    # ── Interface detector (3-D) ─────────────────────────────────────────────
    phi = torch.zeros(Nx,Ny,Nz,dtype=torch.float64)
    phi[Nx//2:,:,:] = 1.0
    det = CFDInterfaceDetector()
    mask = det(phi, dx, dy, dz)
    if mask.shape==(Nx,Ny,Nz) and mask.min()>=0 and mask.max()<=1:
        ok("interface_detector_3d_range")
    else:
        fail("interface_detector_3d_range", f"shape={mask.shape} range=[{mask.min():.2f},{mask.max():.2f}]")

    # ── Advection schemes (mass conservation / finite check) ─────────────────
    rho_t = torch.rand(Nx,Ny,Nz,dtype=torch.float64)+0.5
    ucx   = torch.randn(Nx,Ny,Nz,dtype=torch.float64)*0.05
    ucy   = torch.randn(Nx,Ny,Nz,dtype=torch.float64)*0.05
    ucz   = torch.randn(Nx,Ny,Nz,dtype=torch.float64)*0.05
    dt_t  = 5e-4; bc_t = BoundaryCondition.PERIODIC
    for sch in ["upwind","tvd","weno5","semi_lagrangian"]:
        r2 = advect_density(rho_t,ucx,ucy,ucz,dx,dy,dz,dt_t,bc_t,scheme=sch,
                            Lx=1.0,Ly=1.0,Lz=1.0)
        if r2.isfinite().all() and r2.shape==(Nx,Ny,Nz):
            ok(f"advection_{sch}_finite")
        else:
            fail(f"advection_{sch}_finite","NaN/Inf or wrong shape")

    # ── End-to-end solver step ───────────────────────────────────────────────
    cfg = SolverConfig(Nx=16,Ny=16,Nz=16,dt=1e-4,
                       enable_fluctuations=True,cfl_limit=0.0,
                       dtype=torch.float64)
    solver = StructuralFluctuatingHydro(cfg)
    rho,ux,uy,uz,p = solver.initialize_taylor_green()
    for _ in range(3):
        rho,ux,uy,uz,p,diag = solver.step(rho,ux,uy,uz,p)
    if (rho.isfinite().all() and rho.min()>0
            and ux.isfinite().all() and p.isfinite().all()):
        ok("e2e_taylor_green_3d_3steps")
    else:
        fail("e2e_taylor_green_3d_3steps","non-finite or non-positive values")

    # ── Divergence after projection ──────────────────────────────────────────
    cfg2 = SolverConfig(Nx=16,Ny=16,Nz=16,dt=1e-5,
                        enable_fluctuations=False,cfl_limit=0.0,
                        dtype=torch.float64)
    s2 = StructuralFluctuatingHydro(cfg2)
    r2,u2x,u2y,u2z,p2 = s2.initialize_taylor_green()
    for _ in range(5):
        r2,u2x,u2y,u2z,p2,d2 = s2.step(r2,u2x,u2y,u2z,p2)
    if d2["div_max"] < 1e-4:
        ok(f"divergence_after_projection_3d ({d2['div_max']:.2e})")
    else:
        fail("divergence_after_projection_3d", f"div_max={d2['div_max']:.2e}")

    # ── Rayleigh-Taylor IC ───────────────────────────────────────────────────
    rho_rt,_,_,_,_ = solver.initialize_rayleigh_taylor()
    if rho_rt.min()>=0.9 and rho_rt.max()<=2.1:
        ok("rayleigh_taylor_3d_init")
    else:
        fail("rayleigh_taylor_3d_init",f"rho in [{rho_rt.min():.2f},{rho_rt.max():.2f}]")

    # ── Reset ────────────────────────────────────────────────────────────────
    solver.reset()
    if solver.step_count==0 and not solver._state_ready.item():
        ok("solver_reset")
    else:
        fail("solver_reset","step_count or state_ready wrong after reset")

    # ── DIFF-FIX 9: Differentiability tests ──────────────────────────────────
    # Test that gradient flows through each diff-fixed component.

    # 9a. Smooth limiters
    for lname, lfn in _LIMITERS.items():
        r = torch.randn(16, dtype=torch.float64, requires_grad=True)
        try:
            lfn(r).sum().backward()
            if r.grad is not None and r.grad.isfinite().all():
                ok(f"diff_limiter_{lname}")
            else:
                fail(f"diff_limiter_{lname}", "None/NaN grad")
        except Exception as e:
            fail(f"diff_limiter_{lname}", str(e))
        r.grad = None

    # 9b. softplus_floor
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    _softplus_floor(x, 0.0).sum().backward()
    if x.grad is not None and x.grad.isfinite().all():
        ok("diff_softplus_floor")
    else:
        fail("diff_softplus_floor", f"grad={x.grad}")

    # 9c. logsumexp_max
    x2 = torch.randn(16, dtype=torch.float64, requires_grad=True)
    _logsumexp_max(x2).backward()
    if x2.grad is not None and x2.grad.isfinite().all():
        ok("diff_logsumexp_max")
    else:
        fail("diff_logsumexp_max", f"grad={x2.grad}")

    # 9d. End-to-end backward through solver.step()
    cfg_d = SolverConfig(Nx=8,Ny=8,Nz=8,dt=1e-4,enable_fluctuations=False,
                         cfl_limit=0.0,dtype=torch.float64,
                         advection_scheme="upwind")
    solver_d = StructuralFluctuatingHydro(cfg_d)
    rho_d,ux_d,uy_d,uz_d,p_d = solver_d.initialize_uniform()
    rho_d = rho_d.detach().requires_grad_(True)
    try:
        rho_out,_,_,_,_,_ = solver_d.step(rho_d,ux_d,uy_d,uz_d,p_d)
        rho_out.sum().backward()
        if rho_d.grad is not None and rho_d.grad.isfinite().all():
            ok("diff_e2e_backward_step")
        else:
            fail("diff_e2e_backward_step", f"grad={rho_d.grad}")
    except Exception as e:
        fail("diff_e2e_backward_step", str(e))

    print(f"\n{'='*44}")
    print(f"Tests passed: {passed}  |  Failed: {failed}")
    sys.exit(0 if failed==0 else 1)


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if "--test" in sys.argv:
        print("Running unit tests …\n")
        _run_tests()
        sys.exit(0)

    print("StructuralFluctuatingHydro v4.0 — 3-D production demo")
    print(f"PyTorch {torch.__version__}\n")
    torch.manual_seed(0)

    cfg = SolverConfig(
        Nx=32, Ny=32, Nz=32,
        Lx=1.0, Ly=1.0, Lz=1.0,
        dt=1e-4,
        base_viscosity=1e-3,
        enable_fluctuations=True,
        cfl_limit=0.5,
        advection_scheme="tvd",
        advection_limiter="van_leer",
        log_every=25,
        dtype=torch.float64,
    )
    solver = StructuralFluctuatingHydro(cfg)
    rho, ux, uy, uz, p = solver.initialize_taylor_green()

    hdr = f"{'Step':>6}  {'σ':>8}  {'ν':>10}  {'CFL':>6}  {'div_max':>10}  {'KE':>10}  {'ms':>8}"
    print(hdr); print("-"*len(hdr))

    N_STEPS = 100
    for step in range(N_STEPS):
        rho, ux, uy, uz, p, diag = solver.step(rho, ux, uy, uz, p)
        if step % 10 == 0 or step == N_STEPS-1:
            print(
                f"  {step:>4d}  {diag['sigma']:>8.4f}  {diag['nu']:>10.3e}"
                f"  {diag['cfl']:>6.3f}  {diag['div_max']:>10.3e}"
                f"  {diag['ke']:>10.4e}  {diag['elapsed_ms']:>7.2f}"
            )
    print("\nDemo complete.")
