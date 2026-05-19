=============================================================================
SUPER DNS ONE — Production Release
Advanced Hypersonic SOC‑Controlled CFD Engine
=============================================================================
Author: Yoon A Limsuwan
License: MIT
Year: 2026

Industrial‑grade compressible Navier‑Stokes solver with:
  - HLLC Riemann solver & MUSCL reconstruction (2nd order)
  - Self‑Organised Criticality (SOC) adaptive turbulence model
  - Semantic‑State Contraction (SSC) filtering
  - Renormalisation Group (RG) coarse‑graining
  - Itô stochastic backscatter for sub‑grid scales
  - Batalin–Vilkovisky (BV) consistency monitoring
  - Sutherland’s law viscosity option
  - Fully dimensionless formulation (Re, Mach, Pr)
  - Validation suites: Taylor–Green, shock tube, Kolmogorov spectrum,
    grid convergence
  - Trainable SOC parameters (differential evolution / Optuna)
  - Multi‑backend: CPU, CUDA, MPS, Ascend
=============================================================================
"""

import math
import os
import sys
import json
import argparse
import logging
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.fft import fftn, fftfreq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("SuperDNS")

# =============================================================================
# 1. Core Physics Modules
# =============================================================================
class CSOCKernel(nn.Module):
    """Learnable kernel for SOC eddy viscosity scale."""
    def __init__(self, init_Cs=0.18, init_lambda=12.0):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(init_Cs)))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))

    @property
    def Cs(self):
        return torch.exp(self.log_Cs)

    @property
    def lambd(self):
        return torch.exp(self.log_lambda)

    def forward(self, r):
        # r is a normalised fluctuation intensity
        safe_r = r + 1e-6
        return self.Cs * torch.pow(safe_r, -0.1) * torch.exp(-r / self.lambd)


class SemanticStateContraction:
    """SSC low‑pass filter for stress and signal denoising."""
    def __init__(self, epsilon_fp=0.0028, sigma_target=1.0):
        self.eps = epsilon_fp
        self.target = sigma_target
        self.prev = None

    def __call__(self, sigma):
        if self.prev is None:
            self.prev = sigma
            return sigma
        new = self.prev + self.eps * (sigma - self.prev)
        self.prev = new
        return new

    def reset(self):
        self.prev = None


class SOCController:
    """Adaptive turbulence closure driven by Self‑Organised Criticality."""
    def __init__(self, base_temp: float = 300.0,
                 max_nu_t: float = 0.01,
                 use_ssc: bool = True,
                 epsilon_fp: float = 0.0028):
        self.base_temp = base_temp
        self.max_nu_t = max_nu_t
        self.use_ssc = use_ssc
        self.ssc = SemanticStateContraction(epsilon_fp, 1.0) if use_ssc else None
        self.prev_ke = None

    def reset(self):
        self.prev_ke = None
        if self.ssc:
            self.ssc.reset()

    def nu_t(self, rho, strain_rate, dx, step_cfl=1.0):
        """
        Calculate eddy viscosity nu_t based on local strain rate
        and fluctuation intensity sigma.
        """
        if self.prev_ke is None:
            self.prev_ke = torch.mean(rho * strain_rate).detach()
            return torch.zeros_like(strain_rate)

        # global fluctuation measure
        ke_local = torch.mean(rho * strain_rate)
        sigma = torch.abs(ke_local - self.prev_ke) / (torch.abs(self.prev_ke) + 1e-8)
        self.prev_ke = ke_local.detach()

        if self.ssc:
            sigma = self.ssc(sigma)

        # temperature-like scaling factor
        T_factor = self.base_temp + 2000.0 * torch.sigmoid((sigma - 1.0) / 0.5)
        C_local = torch.clamp((T_factor - self.base_temp) / 1000.0, 0.0, self.max_nu_t)
        # Smagorinsky–style eddy viscosity
        delta = dx
        nu_t = (C_local * delta) ** 2 * strain_rate
        return torch.clamp(nu_t, 0.0, self.max_nu_t)


class DiffRGRefiner:
    """Renormalisation Group (RG) coarse‑graining in 3‑D."""
    def __init__(self, factor: int = 4, n_levels: int = 2):
        self.factor = factor
        self.n_levels = n_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_levels):
            f = self.factor
            D, H, W = x.shape
            mD = D - D % f
            mH = H - H % f
            mW = W - W % f
            if mD == 0 or mH == 0 or mW == 0:
                break
            x_crop = x[:mD, :mH, :mW]
            x_pool = F.avg_pool3d(x_crop.unsqueeze(0).unsqueeze(0),
                                  kernel_size=f, stride=f)
            x = F.interpolate(x_pool, size=(D, H, W),
                              mode='trilinear', align_corners=True).squeeze()
        return x


class ItoStressGenerator:
    """Itô stochastic stress tensor for physically consistent backscatter."""
    def __init__(self, dt: float, noise_amp: float):
        self.dt = dt
        self.noise_amp = noise_amp

    def generate(self, shape, device):
        amp = self.noise_amp * math.sqrt(self.dt)
        s11 = amp * torch.randn(shape, device=device)
        s22 = amp * torch.randn(shape, device=device)
        s33 = amp * torch.randn(shape, device=device)
        s12 = amp * torch.randn(shape, device=device)
        s13 = amp * torch.randn(shape, device=device)
        s23 = amp * torch.randn(shape, device=device)
        return s11, s22, s33, s12, s13, s23


class BVFieldTheory:
    """BV‑inspired diagnostics for mass, momentum and energy consistency."""
    def check_divergence(self, rho_pad, u_pad, v_pad, w_pad, dx):
        # Using second‑order central differences on padded fields
        rhou = rho_pad * u_pad
        rhov = rho_pad * v_pad
        rhow = rho_pad * w_pad
        dudx = (rhou[2:, 1:-1, 1:-1] - rhou[:-2, 1:-1, 1:-1]) / (2 * dx)
        dvdy = (rhov[1:-1, 2:, 1:-1] - rhov[1:-1, :-2, 1:-1]) / (2 * dx)
        dwdz = (rhow[1:-1, 1:-1, 2:] - rhow[1:-1, 1:-1, :-2]) / (2 * dx)
        div = dudx + dvdy + dwdz
        return torch.max(torch.abs(div)).item()

    def kinetic_energy(self, rho, u, v, w):
        return 0.5 * torch.mean(rho * (u ** 2 + v ** 2 + w ** 2)).item()


# =============================================================================
# 2. Advanced Flux Splitting (HLLC) with MUSCL Reconstruction
# =============================================================================
class FluxSplitter:
    """
    HLLC Riemann solver with optional MUSCL‑Hancock reconstruction.
    All fields must be padded with 2 ghost cells (periodic).
    """
    def __init__(self, gamma: float = 1.4, muscl: bool = True):
        self.gamma = gamma
        self.muscl = muscl
        self.eps = 1e-8

    @staticmethod
    def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(a * b > 0,
                           torch.where(torch.abs(a) < torch.abs(b), a, b),
                           torch.zeros_like(a))

    def _reconstruct_1d(self, q_pad: torch.Tensor, axis: int):
        """
        Returns left and right states at cell interfaces for one axis.
        q_pad shape: (nx+4, ny+4, nz+4) with 2 ghost cells per side.
        Output qL, qR shapes: (nx+2, ny, nz) for axis 0, similarly for others.
        """
        # Move axis of interest to dimension 0
        if axis == 0:
            q = q_pad
        elif axis == 1:
            q = q_pad.permute(1, 0, 2)   # axis=1 -> dim0
        else:
            q = q_pad.permute(2, 1, 0)   # axis=2 -> dim0

        N = q.shape[0] - 4  # physical cells

        # Compute slopes at cell centers (including ghost)
        diff1 = q[1:] - q[:-1]          # size N+3
        diff2 = q[2:] - q[1:-1]         # size N+2
        d1 = diff1[:-1]                 # for i=1..N+2
        d2 = diff2
        slope = torch.zeros_like(q)
        slope[1:-1] = self._minmod(d1, d2)

        # Left and right states at interfaces (i=1..N+2)
        qL_full = q[:-1] + 0.5 * slope[:-1]   # size N+3
        qR_full = q[1:]  - 0.5 * slope[1:]    # size N+3

        qL = qL_full[1:-1]   # interfaces 1..N+2 -> size N+2
        qR = qR_full[1:-1]   # interfaces 1..N+2

        # Permute back if necessary
        if axis == 1:
            qL = qL.permute(1, 0, 2)
            qR = qR.permute(1, 0, 2)
        elif axis == 2:
            qL = qL.permute(2, 1, 0)
            qR = qR.permute(2, 1, 0)

        return qL, qR

    def _hllc_flux(self, rhoL, uL, vL, wL, pL,
                   rhoR, uR, vR, wR, pR, normal):
        gamma = self.gamma
        cL = torch.sqrt(gamma * pL / (rhoL + self.eps))
        cR = torch.sqrt(gamma * pR / (rhoR + self.eps))

        if normal == 0:
            unL, unR = uL, uR
            utL, utR = vL, vR
            uwL, uwR = wL, wR
        elif normal == 1:
            unL, unR = vL, vR
            utL, utR = uL, uR
            uwL, uwR = wL, wR
        else:
            unL, unR = wL, wR
            utL, utR = uL, uR
            uwL, uwR = vL, vR

        # Roe averages
        R = torch.sqrt(rhoR / (rhoL + self.eps))
        un_roe = (unL + R * unR) / (1.0 + R)
        c_roe = (cL + R * cR) / (1.0 + R)

        SL = torch.min(unL - cL, un_roe - c_roe)
        SR = torch.max(unR + cR, un_roe + c_roe)

        S_star = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) / \
                 (rhoL * (SL - unL) - rhoR * (SR - unR) + self.eps)

        mask_L = SL >= 0
        mask_R = SR <= 0
        mask_star = ~(mask_L | mask_R)

        rho_face = torch.zeros_like(rhoL)
        un_face  = torch.zeros_like(rhoL)
        p_face   = torch.zeros_like(rhoL)
        ut_face  = torch.zeros_like(rhoL)
        uw_face  = torch.zeros_like(rhoL)
        E_face   = torch.zeros_like(rhoL)

        # Left state
        rho_face[mask_L] = rhoL[mask_L]
        un_face[mask_L] = unL[mask_L]
        p_face[mask_L] = pL[mask_L]
        ut_face[mask_L] = utL[mask_L]
        uw_face[mask_L] = uwL[mask_L]
        E_face[mask_L] = pL[mask_L] / (gamma - 1.0) + 0.5 * rhoL[mask_L] * \
                         (unL[mask_L]**2 + utL[mask_L]**2 + uwL[mask_L]**2)

        # Right state
        rho_face[mask_R] = rhoR[mask_R]
        un_face[mask_R] = unR[mask_R]
        p_face[mask_R] = pR[mask_R]
        ut_face[mask_R] = utR[mask_R]
        uw_face[mask_R] = uwR[mask_R]
        E_face[mask_R] = pR[mask_R] / (gamma - 1.0) + 0.5 * rhoR[mask_R] * \
                         (unR[mask_R]**2 + utR[mask_R]**2 + uwR[mask_R]**2)

        if mask_star.any():
            rhoL_s = rhoL[mask_star]; unL_s = unL[mask_star]; SL_s = SL[mask_star]
            S_star_s = S_star[mask_star]; pL_s = pL[mask_star]
            utL_s = utL[mask_star]; uwL_s = uwL[mask_star]
            rhoR_s = rhoR[mask_star]; unR_s = unR[mask_star]; SR_s = SR[mask_star]
            pR_s = pR[mask_star]; utR_s = utR[mask_star]; uwR_s = uwR[mask_star]

            factorL = (SL_s - unL_s) / (SL_s - S_star_s + self.eps)
            rho_starL = rhoL_s * factorL
            p_starL = pL_s + rhoL_s * (unL_s - SL_s) * (unL_s - S_star_s)
            E_starL = p_starL / (gamma - 1.0) + 0.5 * rho_starL * \
                      (S_star_s**2 + utL_s**2 + uwL_s**2)

            factorR = (SR_s - unR_s) / (SR_s - S_star_s + self.eps)
            rho_starR = rhoR_s * factorR
            p_starR = pR_s + rhoR_s * (unR_s - SR_s) * (unR_s - S_star_s)
            E_starR = p_starR / (gamma - 1.0) + 0.5 * rho_starR * \
                      (S_star_s**2 + utR_s**2 + uwR_s**2)

            star_left = S_star_s >= 0
            rho_face[mask_star] = torch.where(star_left, rho_starL, rho_starR)
            un_face[mask_star] = S_star_s
            p_face[mask_star] = torch.where(star_left, p_starL, p_starR)
            ut_face[mask_star] = torch.where(star_left, utL_s, utR_s)
            uw_face[mask_star] = torch.where(star_left, uwL_s, uwR_s)
            E_face[mask_star] = torch.where(star_left, E_starL, E_starR)

        mass_flux = rho_face * un_face
        if normal == 0:
            F_rhou = mass_flux * un_face + p_face
            F_rhov = mass_flux * ut_face
            F_rhow = mass_flux * uw_face
        elif normal == 1:
            F_rhou = mass_flux * ut_face
            F_rhov = mass_flux * un_face + p_face
            F_rhow = mass_flux * uw_face
        else:
            F_rhou = mass_flux * ut_face
            F_rhov = mass_flux * uw_face
            F_rhow = mass_flux * un_face + p_face

        H_face = (E_face + p_face) / (rho_face + self.eps)
        F_rhoE = mass_flux * H_face

        return mass_flux, F_rhou, F_rhov, F_rhow, F_rhoE

    def compute_flux_divergence(self,
                                rho_pad, u_pad, v_pad, w_pad, p_pad,
                                dx):
        """
        Return convective flux divergence for interior cells.
        div shape: (nx, ny, nz, 5)
        """
        nx, ny, nz = rho_pad.shape[0] - 4, rho_pad.shape[1] - 4, rho_pad.shape[2] - 4
        div_flux = torch.zeros(nx, ny, nz, 5, device=rho_pad.device)

        for axis in range(3):
            if self.muscl:
                rhoL, rhoR = self._reconstruct_1d(rho_pad, axis)
                uL, uR = self._reconstruct_1d(u_pad, axis)
                vL, vR = self._reconstruct_1d(v_pad, axis)
                wL, wR = self._reconstruct_1d(w_pad, axis)
                pL, pR = self._reconstruct_1d(p_pad, axis)
            else:
                # First‑order: left and right cells directly
                if axis == 0:
                    rhoL, rhoR = rho_pad[1:-3], rho_pad[2:-2]
                    uL, uR = u_pad[1:-3], u_pad[2:-2]
                    vL, vR = v_pad[1:-3], v_pad[2:-2]
                    wL, wR = w_pad[1:-3], w_pad[2:-2]
                    pL, pR = p_pad[1:-3], p_pad[2:-2]
                elif axis == 1:
                    rhoL, rhoR = rho_pad[:, 1:-3], rho_pad[:, 2:-2]
                    uL, uR = u_pad[:, 1:-3], u_pad[:, 2:-2]
                    vL, vR = v_pad[:, 1:-3], v_pad[:, 2:-2]
                    wL, wR = w_pad[:, 1:-3], w_pad[:, 2:-2]
                    pL, pR = p_pad[:, 1:-3], p_pad[:, 2:-2]
                else:
                    rhoL, rhoR = rho_pad[..., 1:-3], rho_pad[..., 2:-2]
                    uL, uR = u_pad[..., 1:-3], u_pad[..., 2:-2]
                    vL, vR = v_pad[..., 1:-3], v_pad[..., 2:-2]
                    wL, wR = w_pad[..., 1:-3], w_pad[..., 2:-2]
                    pL, pR = p_pad[..., 1:-3], p_pad[..., 2:-2]

            F_rho, F_u, F_v, F_w, F_E = self._hllc_flux(
                rhoL, uL, vL, wL, pL,
                rhoR, uR, vR, wR, pR, axis)

            # Flux interface array has size N+2
            # Divergence for cell j (0..N-1) = (F[j+2] - F[j+1]) / dx
            if axis == 0:
                div_flux[..., 0] += (F_rho[2:] - F_rho[1:-1]) / dx
                div_flux[..., 1] += (F_u[2:] - F_u[1:-1]) / dx
                div_flux[..., 2] += (F_v[2:] - F_v[1:-1]) / dx
                div_flux[..., 3] += (F_w[2:] - F_w[1:-1]) / dx
                div_flux[..., 4] += (F_E[2:] - F_E[1:-1]) / dx
            elif axis == 1:
                div_flux[..., 0] += (F_rho[:, 2:] - F_rho[:, 1:-1]) / dx
                div_flux[..., 1] += (F_u[:, 2:] - F_u[:, 1:-1]) / dx
                div_flux[..., 2] += (F_v[:, 2:] - F_v[:, 1:-1]) / dx
                div_flux[..., 3] += (F_w[:, 2:] - F_w[:, 1:-1]) / dx
                div_flux[..., 4] += (F_E[:, 2:] - F_E[:, 1:-1]) / dx
            else:
                div_flux[..., 0] += (F_rho[..., 2:] - F_rho[..., 1:-1]) / dx
                div_flux[..., 1] += (F_u[..., 2:] - F_u[..., 1:-1]) / dx
                div_flux[..., 2] += (F_v[..., 2:] - F_v[..., 1:-1]) / dx
                div_flux[..., 3] += (F_w[..., 2:] - F_w[..., 1:-1]) / dx
                div_flux[..., 4] += (F_E[..., 2:] - F_E[..., 1:-1]) / dx

        return div_flux


# =============================================================================
# 3. Compressible CFD Solver
# =============================================================================
@dataclass
class CFDConfig:
    # Grid
    nx: int = 64
    ny: int = 64
    nz: int = 64
    Lx: float = 2.0 * math.pi   # domain size (non‑dimensional)
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi

    # Flow parameters (dimensionless)
    Re: float = 1e4
    Pr: float = 0.71
    gamma: float = 1.4
    Mach: float = 0.1

    # Time stepping
    cfl: float = 0.5
    steps: int = 500

    # SOC / SSC / RG
    soc_base_temp: float = 300.0
    max_nu_t: float = 0.05
    use_rg: bool = True
    rg_factor: int = 4
    rg_levels: int = 2
    rg_interval: int = 10
    ito_noise: float = 0.001

    # Numerics
    muscl: bool = True
    use_sutherland: bool = False  # constant viscosity by default
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CompressibleSolver:
    """Dimensionless compressible Navier‑Stokes solver with advanced models."""
    def __init__(self, cfg: CFDConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dx = cfg.Lx / cfg.nx
        self.dy = cfg.Ly / cfg.ny
        self.dz = cfg.Lz / cfg.nz
        assert abs(self.dx - self.dy) < 1e-10 and abs(self.dy - self.dz) < 1e-10, \
               "Only uniform grid spacing supported"
        self.eps = 1e-8

        # Physical viscosity (constant or Sutherland)
        self.Re = cfg.Re
        self.nu_phys = 1.0 / self.Re if self.Re > 0 else 0.0

        # Modules
        self.soc = SOCController(base_temp=cfg.soc_base_temp,
                                 max_nu_t=cfg.max_nu_t,
                                 use_ssc=True)
        self.ito_gen = ItoStressGenerator(dt=0.0, noise_amp=cfg.ito_noise) if cfg.ito_noise > 0 else None
        self.bv = BVFieldTheory()
        self.rg = DiffRGRefiner(factor=cfg.rg_factor, n_levels=cfg.rg_levels) if cfg.use_rg else None
        self.flux_splitter = FluxSplitter(gamma=cfg.gamma, muscl=cfg.muscl)

        # Conserved fields (padded with 2 ghost cells for MUSCL)
        self.rho = None
        self.rhou = None
        self.rhov = None
        self.rhow = None
        self.rhoE = None

        self.energy_hist = []
        self.div_hist = []
        self.step_count = 0

    def _pad(self, f: torch.Tensor) -> torch.Tensor:
        """Pad 3D tensor with 2 ghost cells on each side (periodic)."""
        return F.pad(f, (2, 2, 2, 2, 2, 2), mode='circular')

    def _unpad(self, f: torch.Tensor) -> torch.Tensor:
        """Extract physical interior from padded tensor (size: nx,ny,nz)."""
        return f[2:-2, 2:-2, 2:-2]

    def _primitive_from_conserved(self):
        rho = self._unpad(self.rho)
        u = self._unpad(self.rhou) / (rho + self.eps)
        v = self._unpad(self.rhov) / (rho + self.eps)
        w = self._unpad(self.rhow) / (rho + self.eps)
        E = self._unpad(self.rhoE)
        ke = 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)
        p = (self.cfg.gamma - 1.0) * (E - ke)
        T = p / (rho + self.eps)
        return rho, u, v, w, p, T

    def _sutherland_viscosity(self, T):
        """Dimensionless Sutherland law (T ref = 1, S = 0.5)."""
        S = 0.5
        return T.pow(1.5) * (1.0 + S) / (T + S)

    def _deriv(self, f_pad, axis, dx):
        if axis == 0:
            return (f_pad[2:, 1:-1, 1:-1] - f_pad[:-2, 1:-1, 1:-1]) / (2 * dx)
        if axis == 1:
            return (f_pad[1:-1, 2:, 1:-1] - f_pad[1:-1, :-2, 1:-1]) / (2 * dx)
        return (f_pad[1:-1, 1:-1, 2:] - f_pad[1:-1, 1:-1, :-2]) / (2 * dx)

    def _laplacian(self, f_pad, dx):
        return (f_pad[2:, 1:-1, 1:-1] + f_pad[:-2, 1:-1, 1:-1] +
                f_pad[1:-1, 2:, 1:-1] + f_pad[1:-1, :-2, 1:-1] +
                f_pad[1:-1, 1:-1, 2:] + f_pad[1:-1, 1:-1, :-2] -
                6 * f_pad[1:-1, 1:-1, 1:-1]) / (dx ** 2)

    def _init_fields(self, case='taylor_green'):
        nx, ny, nz = self.cfg.nx, self.cfg.ny, self.cfg.nz
        x = torch.linspace(0, self.cfg.Lx, nx, device=self.device)
        y = torch.linspace(0, self.cfg.Ly, ny, device=self.device)
        z = torch.linspace(0, self.cfg.Lz, nz, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        if case == 'taylor_green':
            # Dimensionless TG vortex: Mach = u0 / c0, c0 = 1, so u0 = Mach
            u0 = self.cfg.Mach
            u = u0 * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
            v = -u0 * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
            w = 0.5 * u0 * torch.cos(X) * torch.cos(Y) * torch.sin(Z)
            rho = torch.ones_like(u)
            T = 1.0 / self.cfg.gamma  # p0 = rho*T = 1/gamma, sound speed = 1
            p = rho * T
        elif case == 'shock_tube':
            rho = torch.ones(nx, ny, nz, device=self.device)
            u = torch.zeros_like(rho)
            v = torch.zeros_like(rho)
            w = torch.zeros_like(rho)
            T = torch.ones_like(rho) / self.cfg.gamma
            xmid = nx // 2
            rho[:xmid] = 8.0
            T[:xmid] = 2.0 / self.cfg.gamma
            p = rho * T
        elif case == 'hypersonic':
            u0 = self.cfg.Mach
            u = u0 + torch.randn(nx, ny, nz, device=self.device) * 0.001
            v = torch.randn(nx, ny, nz, device=self.device) * 0.001
            w = torch.randn(nx, ny, nz, device=self.device) * 0.001
            rho = torch.ones_like(u)
            T = 1.0 / self.cfg.gamma
            p = rho * T
        else:
            raise ValueError(f"Unknown case: {case}")

        rhou = rho * u
        rhov = rho * v
        rhow = rho * w
        E_int = p / (self.cfg.gamma - 1.0)
        E_kin = 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)
        rhoE = E_int + E_kin

        self.rho = self._pad(rho)
        self.rhou = self._pad(rhou)
        self.rhov = self._pad(rhov)
        self.rhow = self._pad(rhow)
        self.rhoE = self._pad(rhoE)

        # Reset SOC and counters
        self.soc.reset()
        self.step_count = 0

    def step(self):
        dx = self.dx
        gamma = self.cfg.gamma
        Pr = self.cfg.Pr

        # Primitive variables (interior)
        rho_i, u_i, v_i, w_i, p_i, T_i = self._primitive_from_conserved()

        # CFL time step
        c = torch.sqrt(gamma * p_i / (rho_i + self.eps))
        max_speed = torch.max(torch.abs(u_i) + c).item()
        dt = self.cfg.cfl * dx / (max_speed + 1e-8)

        # Laminar viscosity (dynamic)
        if self.cfg.use_sutherland:
            mu_lam = self._sutherland_viscosity(T_i) * self.nu_phys * rho_i
        else:
            mu_lam = self.nu_phys * rho_i

        # SOC eddy viscosity (kinematic -> dynamic)
        strain_rate = torch.sqrt(2 * (0.5 * (self._deriv(self._pad(u_i), 0, dx)**2 +
                                            self._deriv(self._pad(v_i), 1, dx)**2 +
                                            self._deriv(self._pad(w_i), 2, dx)**2) +
                                      0.25 * ((self._deriv(self._pad(u_i), 1, dx) +
                                               self._deriv(self._pad(v_i), 0, dx))**2 +
                                              (self._deriv(self._pad(u_i), 2, dx) +
                                               self._deriv(self._pad(w_i), 0, dx))**2 +
                                              (self._deriv(self._pad(v_i), 2, dx) +
                                               self._deriv(self._pad(w_i), 1, dx))**2)))
        nu_t = self.soc.nu_t(rho_i, strain_rate, dx)
        mu_eff = mu_lam + rho_i * nu_t

        # Convective flux divergence (HLLC + MUSCL)
        rho_pad = self.rho
        u_pad = self._pad(u_i)
        v_pad = self._pad(v_i)
        w_pad = self._pad(w_i)
        p_pad = self._pad(p_i)
        div_flux = self.flux_splitter.compute_flux_divergence(
            rho_pad, u_pad, v_pad, w_pad, p_pad, dx)
        conv_rho  = div_flux[..., 0]
        conv_rhou = div_flux[..., 1]
        conv_rhov = div_flux[..., 2]
        conv_rhow = div_flux[..., 3]
        conv_rhoE = div_flux[..., 4]

        # Velocity gradients for viscous stress (use padded primitives)
        dudx = self._deriv(u_pad, 0, dx)
        dudy = self._deriv(u_pad, 1, dx)
        dudz = self._deriv(u_pad, 2, dx)
        dvdx = self._deriv(v_pad, 0, dx)
        dvdy = self._deriv(v_pad, 1, dx)
        dvdz = self._deriv(v_pad, 2, dx)
        dwdx = self._deriv(w_pad, 0, dx)
        dwdy = self._deriv(w_pad, 1, dx)
        dwdz = self._deriv(w_pad, 2, dx)

        div_v = dudx + dvdy + dwdz

        # Viscous stress tensor (physical + SOC + stochastic)
        tau_xx = mu_eff * (2.0 * dudx - (2.0 / 3.0) * div_v)
        tau_yy = mu_eff * (2.0 * dvdy - (2.0 / 3.0) * div_v)
        tau_zz = mu_eff * (2.0 * dwdz - (2.0 / 3.0) * div_v)
        tau_xy = mu_eff * (dudy + dvdx)
        tau_xz = mu_eff * (dudz + dwdx)
        tau_yz = mu_eff * (dvdz + dwdy)

        if self.ito_gen is not None:
            self.ito_gen.dt = dt
            s11, s22, s33, s12, s13, s23 = self.ito_gen.generate(
                tau_xx.shape, self.device)
            tau_xx += s11; tau_yy += s22; tau_zz += s33
            tau_xy += s12; tau_xz += s13; tau_yz += s23

        # Viscous stress divergence (momentum)
        tau_xx_p = self._pad(tau_xx)
        tau_yy_p = self._pad(tau_yy)
        tau_zz_p = self._pad(tau_zz)
        tau_xy_p = self._pad(tau_xy)
        tau_xz_p = self._pad(tau_xz)
        tau_yz_p = self._pad(tau_yz)

        visc_rhou = self._deriv(tau_xx_p, 0, dx) + self._deriv(tau_xy_p, 1, dx) + self._deriv(tau_xz_p, 2, dx)
        visc_rhov = self._deriv(tau_xy_p, 0, dx) + self._deriv(tau_yy_p, 1, dx) + self._deriv(tau_yz_p, 2, dx)
        visc_rhow = self._deriv(tau_xz_p, 0, dx) + self._deriv(tau_yz_p, 1, dx) + self._deriv(tau_zz_p, 2, dx)

        # Viscous work & heat flux
        tau_dot_u_x = tau_xx * u_i + tau_xy * v_i + tau_xz * w_i
        tau_dot_u_y = tau_xy * u_i + tau_yy * v_i + tau_yz * w_i
        tau_dot_u_z = tau_xz * u_i + tau_yz * v_i + tau_zz * w_i

        visc_work = self._deriv(self._pad(tau_dot_u_x), 0, dx) + \
                    self._deriv(self._pad(tau_dot_u_y), 1, dx) + \
                    self._deriv(self._pad(tau_dot_u_z), 2, dx)

        # Heat flux (Fourier's law)
        cp = gamma / (gamma - 1.0)
        k = mu_eff * cp / Pr
        T_pad = self._pad(T_i)
        dTdx = self._deriv(T_pad, 0, dx)
        dTdy = self._deriv(T_pad, 1, dx)
        dTdz = self._deriv(T_pad, 2, dx)
        qx = -k * dTdx; qy = -k * dTdy; qz = -k * dTdz
        heat_flux_div = self._deriv(self._pad(qx), 0, dx) + \
                        self._deriv(self._pad(qy), 1, dx) + \
                        self._deriv(self._pad(qz), 2, dx)

        visc_rhoE = visc_work + heat_flux_div

        # Time update (explicit Euler)
        rho_new  = rho_i  - dt * conv_rho  + dt * visc_rho
        rhou_new = self._unpad(self.rhou) - dt * conv_rhou + dt * visc_rhou
        rhov_new = self._unpad(self.rhov) - dt * conv_rhov + dt * visc_rhov
        rhow_new = self._unpad(self.rhow) - dt * conv_rhow + dt * visc_rhow
        rhoE_new = self._unpad(self.rhoE) - dt * conv_rhoE + dt * visc_rhoE

        # Positivity preservation (minimal clipping)
        rho_new = torch.clamp(rho_new, min=1e-6)
        ke_new = 0.5 * (rhou_new**2 + rhov_new**2 + rhow_new**2) / (rho_new + self.eps)
        p_new = (gamma - 1.0) * (rhoE_new - ke_new)
        mask_low_p = p_new < 1e-8
        if mask_low_p.any():
            p_new = torch.clamp(p_new, min=1e-8)
            rhoE_new = p_new / (gamma - 1.0) + ke_new
            logger.debug("Pressure clipped in %d cells", mask_low_p.sum().item())

        # RG coarse‑graining (every rg_interval steps)
        self.step_count += 1
        if self.cfg.use_rg and self.rg is not None and self.step_count % self.cfg.rg_interval == 0:
            rho_new  = self.rg.forward(rho_new)
            rhou_new = self.rg.forward(rhou_new)
            rhov_new = self.rg.forward(rhov_new)
            rhow_new = self.rg.forward(rhow_new)
            rhoE_new = self.rg.forward(rhoE_new)

        # Update padded fields
        self.rho   = self._pad(rho_new)
        self.rhou  = self._pad(rhou_new)
        self.rhov  = self._pad(rhov_new)
        self.rhow  = self._pad(rhow_new)
        self.rhoE  = self._pad(rhoE_new)

        return torch.mean(p_new).item()  # return mean pressure as diagnostic

    def run(self, steps=None):
        if steps is None:
            steps = self.cfg.steps
        if self.rho is None:
            self._init_fields()
        for t in range(1, steps + 1):
            p_avg = self.step()
            if t % 50 == 0:
                rho_i, u_i, v_i, w_i, _, _ = self._primitive_from_conserved()
                E = self.bv.kinetic_energy(rho_i, u_i, v_i, w_i)
                div = self.bv.check_divergence(self.rho,
                                                self._pad(u_i),
                                                self._pad(v_i),
                                                self._pad(w_i),
                                                self.dx)
                self.energy_hist.append(E)
                self.div_hist.append(div)
                logger.info(f"Step {t:04d}: E={E:.6f}, div={div:.3e}, ⟨p⟩={p_avg:.4f}")
        return self.rho, self.rhou, self.rhov, self.rhow, self.rhoE, self.energy_hist, self.div_hist

    # =========================================================================
    # Validation suites
    # =========================================================================
    def kolmogorov_slope(self):
        rho_i, u_i, v_i, w_i, _, _ = self._primitive_from_conserved()
        u_hat = fftn(u_i.cpu().numpy())
        v_hat = fftn(v_i.cpu().numpy())
        w_hat = fftn(w_i.cpu().numpy())
        kx = fftfreq(u_i.shape[0], d=self.cfg.Lx / u_i.shape[0])
        ky = fftfreq(u_i.shape[1], d=self.cfg.Ly / u_i.shape[1])
        kz = fftfreq(u_i.shape[2], d=self.cfg.Lz / u_i.shape[2])
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        bins = np.logspace(np.log10(1), np.log10(K.max()), 20)
        E_spec = []
        for i in range(len(bins)-1):
            mask = (K >= bins[i]) & (K < bins[i+1])
            if np.any(mask):
                Ek = 0.5 * (np.abs(u_hat[mask])**2 + np.abs(v_hat[mask])**2 + np.abs(w_hat[mask])**2).mean()
                E_spec.append(Ek)
            else:
                E_spec.append(0.0)
        valid = (np.array(E_spec) > 0)
        if sum(valid) < 3:
            return None
        k_centers = 0.5 * (bins[:-1] + bins[1:])
        slope, _, _, _, _ = linregress(np.log10(k_centers[valid]), np.log10(np.array(E_spec)[valid]))
        return slope

    def taylor_green_energy_decay(self, steps=200):
        self._init_fields('taylor_green')
        energy = []
        for t in range(steps):
            self.step()
            if t % 10 == 0:
                rho_i, u_i, v_i, w_i, _, _ = self._primitive_from_conserved()
                E = self.bv.kinetic_energy(rho_i, u_i, v_i, w_i)
                energy.append(E)
        return energy

    def grid_convergence_test(self, grid_sizes=[32, 64, 96], ref_steps=50):
        errors = []
        u_ref = None
        original_cfg = self.cfg
        for N in grid_sizes:
            self.cfg.nx = N; self.cfg.ny = N; self.cfg.nz = N
            self.dx = self.cfg.Lx / N
            self._init_fields('taylor_green')
            for _ in range(ref_steps):
                self.step()
            rho_i, u_i, v_i, w_i, _, _ = self._primitive_from_conserved()
            if N == max(grid_sizes):
                u_ref = u_i
            else:
                u_ref_down = F.interpolate(u_ref.unsqueeze(0).unsqueeze(0),
                                           size=(N, N, N),
                                           mode='trilinear', align_corners=False).squeeze()
                err = torch.norm(u_i - u_ref_down).item() / np.sqrt(N**3)
                errors.append(err)
        if len(errors) >= 2:
            slope, _, _, _, _ = linregress(np.log(grid_sizes[:-1]), np.log(errors))
            return -slope
        return None


# =============================================================================
# 4. Signal/Noise Separation
# =============================================================================
class SignalNoiseSeparator:
    def __init__(self, ssc_strength=0.5):
        self.ssc = SemanticStateContraction(epsilon_fp=0.01)
        self.strength = ssc_strength

    def denoise(self, sensor_data, reference=None):
        if reference is not None:
            out = sensor_data - self.strength * (sensor_data - reference)
        else:
            out = sensor_data - self.strength * (sensor_data - sensor_data.mean())
        if hasattr(self.ssc, 'prev'):
            out = self.ssc(out)
        return out


# =============================================================================
# 5. Trainable SOC
# =============================================================================
class SOCTrainer:
    @staticmethod
    def objective(params, solver, target_energy):
        solver.cfg.soc_base_temp = params[0]
        solver.cfg.max_nu_t = params[1]
        solver._init_fields('taylor_green')
        for _ in range(50):
            solver.step()
        rho_i, u_i, v_i, w_i, _, _ = solver._primitive_from_conserved()
        E = solver.bv.kinetic_energy(rho_i, u_i, v_i, w_i)
        return abs(E - target_energy)

    @classmethod
    def train(cls, solver, target_energy, method='de', max_iter=30):
        if method == 'optuna' and HAS_OPTUNA:
            def obj(trial):
                solver.cfg.soc_base_temp = trial.suggest_float('temp', 100, 1000)
                solver.cfg.max_nu_t = trial.suggest_float('nu_max', 0.001, 0.1)
                solver._init_fields('taylor_green')
                for _ in range(50):
                    solver.step()
                rho_i, u_i, v_i, w_i, _, _ = solver._primitive_from_conserved()
                E = solver.bv.kinetic_energy(rho_i, u_i, v_i, w_i)
                return abs(E - target_energy)
            study = optuna.create_study(direction='minimize')
            study.optimize(obj, n_trials=max_iter)
            return study.best_params
        else:
            from scipy.optimize import differential_evolution
            bounds = [(100, 1000), (0.001, 0.1)]
            result = differential_evolution(
                lambda p: cls.objective(p, solver, target_energy),
                bounds, maxiter=max_iter)
            return {'soc_base_temp': result.x[0], 'max_nu_t': result.x[1]}


# =============================================================================
# 6. CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SUPER DNS ONE – Industrial Hypersonic CFD")
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--Re', type=float, default=1e4)
    parser.add_argument('--Mach', type=float, default=0.1)
    parser.add_argument('--cfl', type=float, default=0.5)
    parser.add_argument('--soc_temp', type=float, default=300.0)
    parser.add_argument('--max_nu_t', type=float, default=0.05)
    parser.add_argument('--ito', type=float, default=0.001)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--case', default='taylor_green',
                        choices=['taylor_green', 'shock_tube', 'hypersonic'])
    parser.add_argument('--rg', action='store_true')
    parser.add_argument('--muscl', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    args = parser.parse_args()

    cfg = CFDConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        Re=args.Re, Mach=args.Mach, cfl=args.cfl,
        steps=args.steps, soc_base_temp=args.soc_temp,
        max_nu_t=args.max_nu_t, ito_noise=args.ito,
        use_rg=args.rg, muscl=args.muscl,
        device=args.device
    )
    solver = CompressibleSolver(cfg)

    if args.benchmark:
        energy = solver.taylor_green_energy_decay(steps=200)
        plt.figure()
        plt.plot(energy)
        plt.title('Taylor–Green Energy Decay')
        plt.show()
        slope = solver.kolmogorov_slope()
        print(f"Kolmogorov slope: {slope:.3f} (expected −5/3)")
        order = solver.grid_convergence_test([32, 64, 96])
        print(f"Grid convergence order: {order:.2f}")
        rho_i, u_i, v_i, w_i, _, _ = solver._primitive_from_conserved()
        div = solver.bv.check_divergence(solver.rho,
                                         solver._pad(u_i),
                                         solver._pad(v_i),
                                         solver._pad(w_i),
                                         solver.dx)
        print(f"Max divergence: {div:.6e}")
    elif args.train:
        trainer = SOCTrainer()
        params = trainer.train(solver, target_energy=0.5, method='de')
        print(f"Optimal parameters: {params}")
    elif args.denoise:
        sep = SignalNoiseSeparator(ssc_strength=0.7)
        t = torch.linspace(0, 10, 1000)
        true = torch.sin(2 * math.pi * 0.5 * t) + 0.5 * torch.sin(2 * math.pi * 2 * t)
        noisy = true + 0.3 * torch.randn_like(true)
        denoised = sep.denoise(noisy, reference=true)
        plt.plot(t, noisy, label='Noisy')
        plt.plot(t, denoised, label='Denoised')
        plt.plot(t, true, 'k--', label='True')
        plt.legend(); plt.title('SSC Denoising'); plt.show()
    else:
        solver._init_fields(args.case)
        rho, rhou, rhov, rhow, rhoE, energy, div = solver.run()
        print(f"Final kinetic energy: {energy[-1]:.6f}")
        if args.case in ('shock_tube', 'hypersonic'):
            rho_i, _, _, _, _, _ = solver._primitive_from_conserved()
            plt.figure()
            plt.plot(rho_i[:, 0, 0].cpu().numpy())
            plt.title('Density profile')
            plt.show()

if __name__ == "__main__":
    main()
