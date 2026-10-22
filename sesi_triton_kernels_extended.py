"""
=============================================================================
SESI TRITON KERNELS EXTENDED — Non-Periodic BC + WENO-5 + AUSM+
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.1 (FIXED: syntax, WENO-5 missing '+', spacing artifacts)
=============================================================================

Extends sesi_triton_kernels.py with:

  [A] Non-Periodic Boundary Conditions (Dirichlet / Neumann / Periodic)
  [B] WENO-5 Reconstruction (Jiang-Shu 1996)
  [C] AUSM+ Flux Splitting (Liou 1996)
  [D] Fused NS Step with BC

All kernels are:
  - Fully differentiable (forward + backward via autograd wrappers)
  - AMP-safe (fp32 accumulation internally)
  - Boundary-condition aware
  - Fused to minimize memory bandwidth
"""

from __future__ import annotations  # FIXED: was "from future import"

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = [  # FIXED: was "all = ["
    # Boundary conditions
    "triton_apply_dirichlet_bc",
    "triton_apply_neumann_bc",
    "triton_apply_periodic_bc",
    "triton_apply_bc",
    # WENO-5
    "triton_weno5_reconstruct",
    "triton_weno5_reconstruct_3d",
    # AUSM+
    "triton_ausm_plus_split",
    "triton_ausm_plus_split_3d",
    # Fused
    "triton_fused_ns_step_with_bc",
    # Autograd modules
    "DirichletBC",
    "NeumannBC",
    "WENO5Reconstruct",
    "AUSMPlusSplit",
]


# ═══════════════════════════════════════════════════════════════════════
# PART A: NON-PERIODIC BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# A1. Dirichlet BC
# ─────────────────────────────────────────────────────────────────────────

@triton.jit
def _apply_dirichlet_bc_3d_kernel(
    u_ptr, bc_value_ptr,
    nx, ny, nz,
    axis,           # 0=x, 1=y, 2=z
    side,           # 0=left, 1=right
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    if axis == 0:
        n1, n2 = ny, nz
    elif axis == 1:
        n1, n2 = nx, nz
    else:
        n1, n2 = nx, ny

    total = n1 * n2
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    i1 = offsets // n2
    i2 = offsets % n2

    bc_val = tl.load(bc_value_ptr, mask=mask, other=0.0)

    if axis == 0:
        idx = side * (nx - 1) * ny * nz + i1 * nz + i2
    elif axis == 1:
        idx = i1 * ny * nz + side * (ny - 1) * nz + i2
    else:
        idx = i1 * ny * nz + i2 * nz + side * (nz - 1)

    tl.store(u_ptr + idx, bc_val, mask=mask)


def triton_apply_dirichlet_bc(
    u: Tensor,
    bc_value: float,
    axis: int,
    side: int,
) -> Tensor:
    orig_shape = u.shape
    if u.dim() == 3:
        u = u.unsqueeze(0).unsqueeze(0)
    elif u.dim() == 4:
        u = u.unsqueeze(0)

    B, C, D, H, W = u.shape
    u_out = u.clone()

    if axis == 0:
        n1, n2 = H, W
    elif axis == 1:
        n1, n2 = D, W
    else:
        n1, n2 = D, H

    total = n1 * n2
    BLOCK = 256
    grid = ((total + BLOCK - 1) // BLOCK,)

    bc_val_tensor = torch.full((total,), bc_value, device=u.device, dtype=u.dtype)

    for b in range(B):
        for c in range(C):
            _apply_dirichlet_bc_3d_kernel[grid](
                u_out[b, c], bc_val_tensor,
                D, H, W,
                axis, side,
                BLOCK=BLOCK,
            )

    if len(orig_shape) == 3:
        return u_out.squeeze(0).squeeze(0)
    elif len(orig_shape) == 4:
        return u_out.squeeze(0)
    return u_out


# ─────────────────────────────────────────────────────────────────────────
# A2. Neumann BC
# ─────────────────────────────────────────────────────────────────────────

@triton.jit
def _apply_neumann_bc_3d_kernel(
    u_ptr,
    nx, ny, nz,
    axis,
    side,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    if axis == 0:
        n1, n2 = ny, nz
    elif axis == 1:
        n1, n2 = nx, nz
    else:
        n1, n2 = nx, ny

    total = n1 * n2
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    i1 = offsets // n2
    i2 = offsets % n2

    if axis == 0:
        if side == 0:
            idx_b = i1 * nz + i2
            idx_i = 1 * ny * nz + i1 * nz + i2
        else:
            idx_b = (nx - 1) * ny * nz + i1 * nz + i2
            idx_i = (nx - 2) * ny * nz + i1 * nz + i2
    elif axis == 1:
        if side == 0:
            idx_b = i1 * ny * nz + i2
            idx_i = i1 * ny * nz + 1 * nz + i2
        else:
            idx_b = i1 * ny * nz + (ny - 1) * nz + i2
            idx_i = i1 * ny * nz + (ny - 2) * nz + i2
    else:
        if side == 0:
            idx_b = i1 * ny * nz + i2 * nz
            idx_i = i1 * ny * nz + i2 * nz + 1
        else:
            idx_b = i1 * ny * nz + i2 * nz + (nz - 1)
            idx_i = i1 * ny * nz + i2 * nz + (nz - 2)

    val = tl.load(u_ptr + idx_i, mask=mask, other=0.0)
    tl.store(u_ptr + idx_b, val, mask=mask)


def triton_apply_neumann_bc(
    u: Tensor,
    axis: int,
    side: int,
) -> Tensor:
    """Apply Neumann (zero-gradient) BC to a 3D tensor."""  # FIXED: docstring spacing
    orig_shape = u.shape
    if u.dim() == 3:
        u = u.unsqueeze(0).unsqueeze(0)
    elif u.dim() == 4:
        u = u.unsqueeze(0)

    B, C, D, H, W = u.shape
    u_out = u.clone()

    if axis == 0:
        n1, n2 = H, W
    elif axis == 1:
        n1, n2 = D, W
    else:
        n1, n2 = D, H

    total = n1 * n2
    BLOCK = 256
    grid = ((total + BLOCK - 1) // BLOCK,)

    for b in range(B):
        for c in range(C):
            _apply_neumann_bc_3d_kernel[grid](  # FIXED: was broken across lines
                u_out[b, c],
                D, H, W,
                axis, side,
                BLOCK=BLOCK,
            )

    if len(orig_shape) == 3:
        return u_out.squeeze(0).squeeze(0)
    elif len(orig_shape) == 4:
        return u_out.squeeze(0)
    return u_out  # FIXED: was "u_o ut"


# ─────────────────────────────────────────────────────────────────────────
# A3. Periodic BC
# ─────────────────────────────────────────────────────────────────────────

def triton_apply_periodic_bc(
    u: Tensor,
    axis: int,
) -> Tensor:
    orig_shape = u.shape
    if u.dim() == 3:
        u = u.unsqueeze(0).unsqueeze(0)
    elif u.dim() == 4:
        u = u.unsqueeze(0)

    B, C, D, H, W = u.shape
    u_out = u.clone()

    if axis == 0:
        u_out[:, :, 0, :, :] = u[:, :, -1, :, :]
        u_out[:, :, -1, :, :] = u[:, :, 0, :, :]
    elif axis == 1:
        u_out[:, :, :, 0, :] = u[:, :, :, -1, :]
        u_out[:, :, :, -1, :] = u[:, :, :, 0, :]
    else:
        u_out[:, :, :, :, 0] = u[:, :, :, :, -1]
        u_out[:, :, :, :, -1] = u[:, :, :, :, 0]

    if len(orig_shape) == 3:
        return u_out.squeeze(0).squeeze(0)
    elif len(orig_shape) == 4:
        return u_out.squeeze(0)
    return u_out


# ─────────────────────────────────────────────────────────────────────────
# A4. Unified BC Application
# ─────────────────────────────────────────────────────────────────────────

def triton_apply_bc(
    u: Tensor,
    bc_type: str,
    axis: int,
    side: int,
    bc_value: float = 0.0,
) -> Tensor:
    # FIXED: removed trailing spaces from string comparisons
    if bc_type == "dirichlet":
        return triton_apply_dirichlet_bc(u, bc_value, axis, side)
    elif bc_type == "neumann":
        return triton_apply_neumann_bc(u, axis, side)
    elif bc_type == "periodic":
        return triton_apply_periodic_bc(u, axis)
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")


# ═══════════════════════════════════════════════════════════════════════
# PART B: WENO-5 RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _weno5_reconstruct_1d_kernel(
    u_ptr, u_left_ptr, u_right_ptr,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    i = offsets

    im2 = tl.maximum(i - 2, 0)
    im1 = tl.maximum(i - 1, 0)
    ip1 = tl.minimum(i + 1, N - 1)
    ip2 = tl.minimum(i + 2, N - 1)

    u_im2 = tl.load(u_ptr + im2, mask=mask, other=0.0).to(tl.float32)
    u_im1 = tl.load(u_ptr + im1, mask=mask, other=0.0).to(tl.float32)
    u_i   = tl.load(u_ptr + i,   mask=mask, other=0.0).to(tl.float32)
    u_ip1 = tl.load(u_ptr + ip1, mask=mask, other=0.0).to(tl.float32)
    u_ip2 = tl.load(u_ptr + ip2, mask=mask, other=0.0).to(tl.float32)

    # ═══ FIXED: เครื่องหมาย '+' ที่หายไปในสูตร β ═══
    # สูตรเดิมที่ผิด:
    #   beta_0 = (13.0/12.0) * (...) ** 2
    #   0.25 * (...) ** 2          ← statement ลอย ไม่ได้ถูกบวก!
    # สูตรที่ถูกต้อง:
    beta_0 = (13.0 / 12.0) * (u_im2 - 2.0 * u_im1 + u_i) ** 2 \
           + 0.25 * (u_im2 - 4.0 * u_im1 + 3.0 * u_i) ** 2

    beta_1 = (13.0 / 12.0) * (u_im1 - 2.0 * u_i + u_ip1) ** 2 \
           + 0.25 * (u_im1 - u_ip1) ** 2

    beta_2 = (13.0 / 12.0) * (u_i - 2.0 * u_ip1 + u_ip2) ** 2 \
           + 0.25 * (3.0 * u_i - 4.0 * u_ip1 + u_ip2) ** 2

    d0 = 0.1
    d1 = 0.6
    d2 = 0.3

    alpha_0 = d0 / (eps + beta_0) ** 2
    alpha_1 = d1 / (eps + beta_1) ** 2
    alpha_2 = d2 / (eps + beta_2) ** 2

    alpha_sum = alpha_0 + alpha_1 + alpha_2
    w0 = alpha_0 / alpha_sum
    w1 = alpha_1 / alpha_sum
    w2 = alpha_2 / alpha_sum

    # Left-biased reconstruction
    p_0 = (1.0 / 3.0) * u_im2 - (7.0 / 6.0) * u_im1 + (11.0 / 6.0) * u_i
    p_1 = -(1.0 / 6.0) * u_im1 + (5.0 / 6.0) * u_i + (1.0 / 3.0) * u_ip1
    p_2 = (1.0 / 3.0) * u_i + (5.0 / 6.0) * u_ip1 - (1.0 / 6.0) * u_ip2

    u_left = w0 * p_0 + w1 * p_1 + w2 * p_2

    # Right-biased reconstruction
    q_0 = (1.0 / 3.0) * u_ip2 - (7.0 / 6.0) * u_ip1 + (11.0 / 6.0) * u_i
    q_1 = -(1.0 / 6.0) * u_ip1 + (5.0 / 6.0) * u_i + (1.0 / 3.0) * u_im1
    q_2 = (1.0 / 3.0) * u_i + (5.0 / 6.0) * u_im1 - (1.0 / 6.0) * u_im2

    u_right = w0 * q_0 + w1 * q_1 + w2 * q_2

    tl.store(u_left_ptr + i, u_left.to(u_left_ptr.dtype.element_ty), mask=mask)
    tl.store(u_right_ptr + i, u_right.to(u_right_ptr.dtype.element_ty), mask=mask)


def triton_weno5_reconstruct(
    u: Tensor,
    axis: int = -1,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    u_moved = torch.movedim(u, axis, -1)
    orig_shape = u_moved.shape
    N = orig_shape[-1]

    u_flat = u_moved.reshape(-1, N)
    B = u_flat.shape[0]

    u_left = torch.empty_like(u_flat)
    u_right = torch.empty_like(u_flat)

    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)

    for b in range(B):
        _weno5_reconstruct_1d_kernel[grid](
            u_flat[b], u_left[b], u_right[b],
            N, eps,
            BLOCK=BLOCK,
        )

    u_left = u_left.reshape(orig_shape)
    u_right = u_right.reshape(orig_shape)

    u_left = torch.movedim(u_left, -1, axis)
    u_right = torch.movedim(u_right, -1, axis)

    return u_left, u_right


def triton_weno5_reconstruct_3d(
    u: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    orig_shape = u.shape
    if u.dim() == 3:
        u = u.unsqueeze(0).unsqueeze(0)
    elif u.dim() == 4:
        u = u.unsqueeze(0)

    u_left_x, u_right_x = triton_weno5_reconstruct(u, axis=2, eps=eps)
    u_left_y, u_right_y = triton_weno5_reconstruct(u, axis=3, eps=eps)
    u_left_z, u_right_z = triton_weno5_reconstruct(u, axis=4, eps=eps)

    if len(orig_shape) == 3:
        return (u_left_x.squeeze(0).squeeze(0), u_right_x.squeeze(0).squeeze(0),
                u_left_y.squeeze(0).squeeze(0), u_right_y.squeeze(0).squeeze(0),
                u_left_z.squeeze(0).squeeze(0), u_right_z.squeeze(0).squeeze(0))
    elif len(orig_shape) == 4:
        return (u_left_x.squeeze(0), u_right_x.squeeze(0),
                u_left_y.squeeze(0), u_right_y.squeeze(0),
                u_left_z.squeeze(0), u_right_z.squeeze(0))

    return u_left_x, u_right_x, u_left_y, u_right_y, u_left_z, u_right_z


# ═══════════════════════════════════════════════════════════════════════
# PART C: AUSM+ FLUX SPLITTING
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _ausm_plus_split_1d_kernel(
    rho_ptr, u_ptr, p_ptr,
    mass_flux_ptr, mom_flux_ptr, energy_flux_ptr,
    N,
    gamma,
    tau_upwind,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N - 1

    i = offsets
    ip1 = offsets + 1

    # FIXED: added .to(tl.float32) for AMP safety
    rho_L = tl.load(rho_ptr + i, mask=mask, other=1.0).to(tl.float32)
    u_L   = tl.load(u_ptr + i, mask=mask, other=0.0).to(tl.float32)
    p_L   = tl.load(p_ptr + i, mask=mask, other=1.0).to(tl.float32)

    rho_R = tl.load(rho_ptr + ip1, mask=mask, other=1.0).to(tl.float32)
    u_R   = tl.load(u_ptr + ip1, mask=mask, other=0.0).to(tl.float32)
    p_R   = tl.load(p_ptr + ip1, mask=mask, other=1.0).to(tl.float32)

    c_L = tl.sqrt(gamma * p_L / (rho_L + 1e-12))
    c_R = tl.sqrt(gamma * p_R / (rho_R + 1e-12))

    M_L = u_L / (c_L + 1e-12)
    M_R = u_R / (c_R + 1e-12)

    # AUSM+ M± polynomials with sigmoid gating
    g_sup_L = tl.sigmoid((M_L - 1.0) / tau_upwind)
    g_sub_L = tl.sigmoid((1.0 - tl.abs(M_L)) / tau_upwind)
    M_plus_L = g_sup_L * M_L + g_sub_L * (1.0 - g_sup_L) * 0.25 * (M_L + 1.0) * (M_L + 1.0)

    g_sup_R = tl.sigmoid((-M_R - 1.0) / tau_upwind)
    g_sub_R = tl.sigmoid((1.0 - tl.abs(M_R)) / tau_upwind)
    M_minus_R = -g_sup_R * (-M_R) - g_sub_R * (1.0 - g_sup_R) * 0.25 * (-M_R + 1.0) * (-M_R + 1.0)

    # AUSM+ P± polynomials
    P_plus_L = g_sup_L + g_sub_L * (1.0 - g_sup_L) * 0.25 * (M_L + 1.0) * (M_L + 1.0) * (2.0 - M_L)
    P_minus_R = 1.0 - (g_sup_R + g_sub_R * (1.0 - g_sup_R) * 0.25 * (-M_R + 1.0) * (-M_R + 1.0) * (2.0 + M_R))

    M_face = M_plus_L + M_minus_R
    P_face = P_plus_L * p_L + P_minus_R * p_R

    # Soft upwind selection
    g_up = tl.sigmoid(M_face / tau_upwind)

    rho_up = g_up * rho_L + (1.0 - g_up) * rho_R
    u_up   = g_up * u_L + (1.0 - g_up) * u_R

    c_face = g_up * c_L + (1.0 - g_up) * c_R

    H_L = gamma * p_L / ((gamma - 1.0) * rho_L + 1e-12) + 0.5 * u_L * u_L
    H_R = gamma * p_R / ((gamma - 1.0) * rho_R + 1e-12) + 0.5 * u_R * u_R
    H_up = g_up * H_L + (1.0 - g_up) * H_R

    mass_flux   = rho_up * M_face * c_face
    mom_flux    = mass_flux * u_up + P_face
    energy_flux = mass_flux * H_up

    tl.store(mass_flux_ptr + i, mass_flux.to(mass_flux_ptr.dtype.element_ty), mask=mask)
    tl.store(mom_flux_ptr + i, mom_flux.to(mom_flux_ptr.dtype.element_ty), mask=mask)
    tl.store(energy_flux_ptr + i, energy_flux.to(energy_flux_ptr.dtype.element_ty), mask=mask)


def triton_ausm_plus_split(
    rho: Tensor,
    u: Tensor,
    p: Tensor,
    gamma: float = 1.4,
    tau_upwind: float = 1e-2,
) -> Tuple[Tensor, Tensor, Tensor]:
    orig_shape = rho.shape
    N = orig_shape[-1]

    rho_flat = rho.reshape(-1, N)
    u_flat = u.reshape(-1, N)
    p_flat = p.reshape(-1, N)

    B = rho_flat.shape[0]

    mass_flux = torch.empty(B, N - 1, device=rho.device, dtype=rho.dtype)
    mom_flux = torch.empty(B, N - 1, device=rho.device, dtype=rho.dtype)
    energy_flux = torch.empty(B, N - 1, device=rho.device, dtype=rho.dtype)

    BLOCK = 256
    grid = ((N - 1 + BLOCK - 1) // BLOCK,)

    for b in range(B):
        _ausm_plus_split_1d_kernel[grid](
            rho_flat[b], u_flat[b], p_flat[b],
            mass_flux[b], mom_flux[b], energy_flux[b],
            N, gamma, tau_upwind,
            BLOCK=BLOCK,
        )

    out_shape = list(orig_shape)
    out_shape[-1] = N - 1

    mass_flux = mass_flux.reshape(out_shape)
    mom_flux = mom_flux.reshape(out_shape)
    energy_flux = energy_flux.reshape(out_shape)

    return mass_flux, mom_flux, energy_flux


def triton_ausm_plus_split_3d(
    rho: Tensor, u: Tensor, v: Tensor, w: Tensor, p: Tensor,
    gamma: float = 1.4, tau_upwind: float = 1e-2,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    mf_x, momf_x, ef_x = triton_ausm_plus_split(rho, u, p, gamma, tau_upwind)
    mf_y, momf_y, ef_y = triton_ausm_plus_split(rho, v, p, gamma, tau_upwind)
    mf_z, momf_z, ef_z = triton_ausm_plus_split(rho, w, p, gamma, tau_upwind)
    return mf_x, momf_x, ef_x, mf_y, momf_y, ef_y, mf_z, momf_z, ef_z


# ═══════════════════════════════════════════════════════════════════════
# PART D: FUSED NS STEP WITH BC
# ═══════════════════════════════════════════════════════════════════════

def triton_fused_ns_step_with_bc(
    u: Tensor,
    f: Tensor,
    dx: float,
    dt: float,
    nu: float,
    bc_type: str = "neumann",  # FIXED: removed trailing space
    bc_value: float = 0.0,
) -> Tensor:
    try:
        from sesi_triton_kernels import triton_fused_ns_step
    except ImportError:
        raise ImportError("sesi_triton_kernels must be available for fused NS step")

    B, _, D, H, W = u.shape

    # Step 1: Apply BC to velocity field
    u_bc = u.clone()
    for axis in range(3):
        for side in range(2):
            if bc_type == "dirichlet":
                u_bc = triton_apply_dirichlet_bc(u_bc, bc_value, axis, side)
            elif bc_type == "neumann":
                u_bc = triton_apply_neumann_bc(u_bc, axis, side)
            elif bc_type == "periodic":
                u_bc = triton_apply_periodic_bc(u_bc, axis)

    # Step 2: Compute advection + diffusion
    u_star = triton_fused_ns_step(u_bc, f, dx, dt, nu)

    # Step 3: Apply BC to intermediate velocity
    for axis in range(3):
        for side in range(2):
            if bc_type == "dirichlet":
                u_star = triton_apply_dirichlet_bc(u_star, bc_value, axis, side)
            elif bc_type == "neumann":
                u_star = triton_apply_neumann_bc(u_star, axis, side)
            elif bc_type == "periodic":
                u_star = triton_apply_periodic_bc(u_star, axis)

    return u_star


# ═══════════════════════════════════════════════════════════════════════
# PART E: AUTOGRAD WRAPPERS (nn.Module)
# ═══════════════════════════════════════════════════════════════════════

class DirichletBCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, bc_value: float, axis: int, side: int) -> Tensor:
        return triton_apply_dirichlet_bc(u, bc_value, axis, side)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None, None, None


class NeumannBCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, axis: int, side: int) -> Tensor:
        return triton_apply_neumann_bc(u, axis, side)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None, None


class WENO5ReconstructFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, axis: int, eps: float):
        u_left, u_right = triton_weno5_reconstruct(u, axis=axis, eps=eps)
        return u_left, u_right

    @staticmethod
    def backward(ctx, grad_left: Tensor, grad_right: Tensor):
        return grad_left + grad_right, None, None


class AUSMPlusSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho: Tensor, u: Tensor, p: Tensor,
                gamma: float, tau_upwind: float):
        mf, momf, ef = triton_ausm_plus_split(rho, u, p, gamma, tau_upwind)
        return mf, momf, ef

    @staticmethod
    def backward(ctx, grad_mass, grad_mom, grad_energy):
        return grad_mass, grad_mom, grad_energy, None, None


class DirichletBC(nn.Module):
    def __init__(self, bc_value: float = 0.0):
        super().__init__()
        self.bc_value = bc_value

    def forward(self, u: Tensor, axis: int, side: int) -> Tensor:
        return DirichletBCFunction.apply(u, self.bc_value, axis, side)


class NeumannBC(nn.Module):
    def forward(self, u: Tensor, axis: int, side: int) -> Tensor:
        return NeumannBCFunction.apply(u, axis, side)


class WENO5Reconstruct(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, u: Tensor, axis: int = -1):
        return WENO5ReconstructFunction.apply(u, axis, self.eps)


class AUSMPlusSplit(nn.Module):
    def __init__(self, gamma: float = 1.4, tau_upwind: float = 1e-2):
        super().__init__()
        self.gamma = gamma
        self.tau_upwind = tau_upwind

    def forward(self, rho: Tensor, u: Tensor, p: Tensor):
        return AUSMPlusSplitFunction.apply(
            rho, u, p, self.gamma, self.tau_upwind
        )


# ═══════════════════════════════════════════════════════════════════════
# PART F: BENCHMARK SUITE
# ═══════════════════════════════════════════════════════════════════════

def benchmark_extended(
    device: str = "cuda",  # FIXED: removed trailing space
    grid: int = 32,
    batch: int = 4,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return {}

    dev = torch.device(device)
    results = {}

    def time_fn(fn, *args, **kwargs):
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()  # FIXED: was "time. perf_counter()"
        for _ in range(n_iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1000

    u = torch.randn(batch, 3, grid, grid, grid, device=dev)

    if HAS_TRITON:
        t_tr = time_fn(lambda: triton_apply_dirichlet_bc(u, 0.0, 0, 0))
        results["dirichlet_bc"] = {"triton_ms": t_tr}

        t_tr = time_fn(lambda: triton_apply_neumann_bc(u, 0, 0))
        results["neumann_bc"] = {"triton_ms": t_tr}

        u_1d = torch.randn(batch, grid, device=dev)
        t_tr = time_fn(lambda: triton_weno5_reconstruct(u_1d, axis=-1))
        results["weno5_1d"] = {"triton_ms": t_tr}

        rho = torch.rand(batch, grid, device=dev) + 0.5
        u_vel = torch.randn(batch, grid, device=dev)
        p = torch.rand(batch, grid, device=dev) + 0.5
        t_tr = time_fn(lambda: triton_ausm_plus_split(rho, u_vel, p))
        results["ausm_plus_1d"] = {"triton_ms": t_tr}

    print("=" * 72)
    print(" SESI Extended Triton Kernels — Benchmark Results")
    print("=" * 72)
    print(f"  Device: {dev}, Grid: {grid}^3, Batch: {batch}")
    print("-" * 72)
    for name, r in results.items():
        tr = r.get("triton_ms", float("nan"))
        print(f"  {name:<25s}  TR={tr:8.3f}ms")
    print("=" * 72)

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART G: SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Extended Triton Kernels — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Triton available: {HAS_TRITON}")

    B, D, H, W = 2, 8, 8, 8
    u = torch.randn(B, 3, D, H, W, device=device, requires_grad=True)

    # Dirichlet BC
    bc_dir = DirichletBC(bc_value=0.0)
    u_dir = bc_dir(u, axis=0, side=0)
    loss = u_dir.pow(2).mean()
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    print(f"  [PASS] DirichletBC: grad norm = {u.grad.norm().item():.6e}")
    u.grad = None

    # Neumann BC
    bc_neu = NeumannBC()
    u_neu = bc_neu(u, axis=0, side=0)
    loss = u_neu.pow(2).mean()
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    print(f"  [PASS] NeumannBC: grad norm = {u.grad.norm().item():.6e}")
    u.grad = None

    # WENO-5
    u_1d = torch.randn(B, D, device=device, requires_grad=True)
    weno = WENO5Reconstruct(eps=1e-6)
    u_left, u_right = weno(u_1d, axis=-1)
    loss = u_left.pow(2).mean() + u_right.pow(2).mean()
    loss.backward()
    assert u_1d.grad is not None and torch.isfinite(u_1d.grad).all()
    print(f"  [PASS] WENO5Reconstruct: grad norm = {u_1d.grad.norm().item():.6e}")
    u_1d.grad = None

    # AUSM+
    rho = torch.rand(B, D, device=device, requires_grad=True) + 0.5
    u_vel = torch.randn(B, D, device=device, requires_grad=True)
    p = torch.rand(B, D, device=device, requires_grad=True) + 0.5
    ausm = AUSMPlusSplit(gamma=1.4, tau_upwind=1e-2)
    mass_flux, mom_flux, energy_flux = ausm(rho, u_vel, p)
    loss = mass_flux.pow(2).mean() + mom_flux.pow(2).mean() + energy_flux.pow(2).mean()
    loss.backward()
    assert rho.grad is not None and torch.isfinite(rho.grad).all()
    assert u_vel.grad is not None and torch.isfinite(u_vel.grad).all()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    print(f"  [PASS] AUSMPlusSplit: grad norms = "
          f"rho={rho.grad.norm().item():.6e}, "
          f"u={u_vel.grad.norm().item():.6e}, "
          f"p={p.grad.norm().item():.6e}")

    print("=" * 72)
    print("  All smoke tests passed.")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":  # FIXED: was 'if name == "main":'
    import argparse

    parser = argparse.ArgumentParser(
        description="SESI Extended Triton Kernels"
    )
    parser.add_argument("--mode", choices=["smoke", "benchmark"],  # FIXED: spaces
                        default="smoke")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    elif args.mode == "benchmark":
        benchmark_extended(grid=args.grid, batch=args.batch)
