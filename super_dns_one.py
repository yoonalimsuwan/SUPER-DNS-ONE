# =============================================================================
# SUPER DNS ONE — Industrial‑Grade 3D Compressible DNS / LES Solver
# =============================================================================
# Author : Yoon A Limsuwan
# License: MIT
# Year   : 2026
#
# First industrial release – fully differentiable, multi‑physics
# Navier‑Stokes solver designed for hypersonic civilian aviation,
# cardiovascular & respiratory flows, and atmospheric boundary layers.
#
# Built on solid open‑source foundations:
#   • PyTorch (BSD‑style) – automatic differentiation, GPU back‑end
#   • NumPy (BSD‑3‑Clause) – array operations for post‑processing
#   • SciPy (BSD‑3‑Clause) – FFT, statistics, differential evolution, Wiener filter
#   • Matplotlib (PSF‑based) – visualisation (optional)
#   • Optuna (MIT) – Bayesian hyper‑parameter tuning (optional)
#   • CoolProp (MIT) – real‑gas equations of state for hypersonic flows
#   • PyWavelets (BSD‑3‑Clause) – wavelet denoising for advanced signal processing
#
# All physical models are original implementations of published methods:
#   • AUSM+ and HLLC Riemann solvers (Liou, Toro)
#   • MUSCL reconstruction with minmod limiter (van Leer)
#   • 3rd‑order TVD Runge‑Kutta (Gottlieb & Shu)
#   • Self‑Organised Criticality (SOC) dynamic sub‑grid model
#   • Semantic‑State Contraction (SSC) low‑pass filter for stress denoising
#   • Itô stochastic backscatter (LES)
#   • Renormalisation Group (RG) conservative spectral truncation
#   • Compressibility correction (Sarkar)
#   • Ducros shock sensor for adaptive artificial viscosity
#   • Werner–Wengle wall model for high‑Re LES
#   • Characteristic non‑reflecting boundary conditions (Poinsot & Lele)
#   • Batalin–Vilkovisky (BV) consistency diagnostics
#   • Immersed boundary method (volume penalisation, Angot et al.)
#   • Wavelet‑based denoising (via PyWavelets) for sensor data
#   • Real‑gas equation of state (via CoolProp) for hypersonic conditions
#
# Features:
#   • Conservative finite‑volume discretisation on structured grids
#   • 2nd‑order MUSCL reconstruction, AUSM+/HLLC Riemann solvers
#   • Low‑storage 3rd‑order TVD Runge‑Kutta time integration
#   • SOC‑adaptive eddy viscosity with 5‑parameter learnable kernel
#   • Itô stochastic backscatter for LES
#   • RG conservative high‑wavenumber truncation
#   • Ducros sensor + artificial viscosity for robust shock capturing
#   • Werner–Wengle wall model (optional) for high‑Re wall‑bounded flows
#   • High‑order ghost cell treatment for all boundary conditions
#   • Mixed precision (FP16/FP32) via PyTorch AMP
#   • Multi‑backend: CPU, CUDA, MPS, Ascend NPU
#   • Differentiable end‑to‑end (autograd‑compatible)
#   • Trainable SOC kernel (Differential Evolution / Optuna)
#   • Grid convergence test & Kolmogorov spectral analysis
#   • Checkpoint / restart support
#   • Real‑gas thermodynamics for hypersonic flows (CoolProp)
#   • Immersed boundary method for complex medical geometries
#   • Wavelet‑based denoising for sensor signals
#   • Multi‑GPU distributed memory parallelism via domain decomposition
#
# This software is intended exclusively for peaceful civilian applications.
# =============================================================================

import math
import sys
import argparse
import logging
import warnings
import os
import pickle
from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.fft import fftn, fftfreq
from scipy.optimize import differential_evolution
from scipy import signal as scipy_signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Optional real‑gas EOS (CoolProp)
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

# Optional wavelet library for advanced denoising
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SuperDNS")

# =============================================================================
# 0. Device / Backend Utilities
# =============================================================================
def get_device(preferred: str = "cuda") -> torch.device:
    """Select the appropriate compute device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "ascend":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. Boundary Conditions & Ghost‑Cell Treatment
# =============================================================================
class BoundaryCondition:
    """Base class for boundary condition treatment on physical cells."""
    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        raise NotImplementedError

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        raise NotImplementedError

class PeriodicBC(BoundaryCondition):
    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        pass
    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        pass

class SupersonicInflowBC(BoundaryCondition):
    def __init__(self, rho_inf, u_inf, v_inf, w_inf, p_inf):
        self.rho_inf = rho_inf
        self.u_inf = u_inf
        self.v_inf = v_inf
        self.w_inf = w_inf
        self.p_inf = p_inf

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        nx, ny, nz = rho.shape
        if axis == 0:
            i = 0 if side == 'left' else nx - 1
            rho[i] = self.rho_inf
            rhou[i] = self.rho_inf * self.u_inf
            rhov[i] = self.rho_inf * self.v_inf
            rhow[i] = self.rho_inf * self.w_inf
            rhoE[i] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                self.u_inf ** 2 + self.v_inf ** 2 + self.w_inf ** 2
            )
        elif axis == 1:
            j = 0 if side == 'left' else ny - 1
            rho[:, j] = self.rho_inf
            rhou[:, j] = self.rho_inf * self.u_inf
            rhov[:, j] = self.rho_inf * self.v_inf
            rhow[:, j] = self.rho_inf * self.w_inf
            rhoE[:, j] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                self.u_inf ** 2 + self.v_inf ** 2 + self.w_inf ** 2
            )
        else:
            k = 0 if side == 'left' else nz - 1
            rho[:, :, k] = self.rho_inf
            rhou[:, :, k] = self.rho_inf * self.u_inf
            rhov[:, :, k] = self.rho_inf * self.v_inf
            rhow[:, :, k] = self.rho_inf * self.w_inf
            rhoE[:, :, k] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                self.u_inf ** 2 + self.v_inf ** 2 + self.w_inf ** 2
            )

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    i = 0 - g
                    rho[i] = self.rho_inf
                    rhou[i] = self.rho_inf * self.u_inf
                    rhov[i] = self.rho_inf * self.v_inf
                    rhow[i] = self.rho_inf * self.w_inf
                    rhoE[i] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                        self.u_inf**2 + self.v_inf**2 + self.w_inf**2)
            else:
                for g in range(1, n_ghost + 1):
                    i = nx - 1 + g
                    rho[i] = self.rho_inf
                    rhou[i] = self.rho_inf * self.u_inf
                    rhov[i] = self.rho_inf * self.v_inf
                    rhow[i] = self.rho_inf * self.w_inf
                    rhoE[i] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                        self.u_inf**2 + self.v_inf**2 + self.w_inf**2)
        elif axis == 1:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    j = 0 - g
                    rho[:, j] = self.rho_inf
                    rhou[:, j] = self.rho_inf * self.u_inf
                    rhov[:, j] = self.rho_inf * self.v_inf
                    rhow[:, j] = self.rho_inf * self.w_inf
                    rhoE[:, j] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                        self.u_inf**2 + self.v_inf**2 + self.w_inf**2)
            else:
                for g in range(1, n_ghost + 1):
                    j = ny - 1 + g
                    rho[:, j] = self.rho_inf
                    rhou[:, j] = self.rho_inf * self.u_inf
                    rhov[:, j] = self.rho_inf * self.v_inf
                    rhow[:, j] = self.rho_inf * self.w_inf
                    rhoE[:, j] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                        self.u_inf**2 + self.v_inf**2 + self.w_inf**2)
        else:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    k = 0 - g
                    rho[:, :, k] = self.rho_inf
                    rhou[:, :, k] = self.rho_inf * self.u_inf
                    rhov[:, :, k] = self.rho_inf * self.v_inf
                    rhow[:, :, k] = self.rho_inf * self.w_inf
                    rhoE[:, :, k] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                        self.u_inf**2 + self.v_inf**2 + self.w_inf**2)
            else:
                for g in range(1, n_ghost + 1):
                    k = nz - 1 + g
                    rho[:, :, k] = self.rho_inf
                    rhou[:, :, k] = self.rho_inf * self.u_inf
                    rhov[:, :, k] = self.rho_inf * self.v_inf
                    rhow[:, :, k] = self.rho_inf * self.w_inf
                    rhoE[:, :, k] = self.p_inf / (gamma - 1) + 0.5 * self.rho_inf * (
                        self.u_inf**2 + self.v_inf**2 + self.w_inf**2)


class SubsonicOutflowBC(BoundaryCondition):
    def __init__(self, p_out):
        self.p_out = p_out

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        nx, ny, nz = rho.shape
        if axis == 0:
            i = nx - 1 if side == 'right' else 0
            i_int = nx - 2 if side == 'right' else 1
            rho[i] = rho[i_int]
            rhou[i] = rhou[i_int]
            rhov[i] = rhov[i_int]
            rhow[i] = rhow[i_int]
            ke = 0.5 * (rhou[i] ** 2 + rhov[i] ** 2 + rhow[i] ** 2) / (rho[i] + 1e-8)
            rhoE[i] = self.p_out / (gamma - 1) + ke
        elif axis == 1:
            j = ny - 1 if side == 'right' else 0
            j_int = ny - 2 if side == 'right' else 1
            rho[:, j] = rho[:, j_int]
            rhou[:, j] = rhou[:, j_int]
            rhov[:, j] = rhov[:, j_int]
            rhow[:, j] = rhow[:, j_int]
            ke = 0.5 * (rhou[:, j] ** 2 + rhov[:, j] ** 2 + rhow[:, j] ** 2) / (rho[:, j] + 1e-8)
            rhoE[:, j] = self.p_out / (gamma - 1) + ke
        else:
            k = nz - 1 if side == 'right' else 0
            k_int = nz - 2 if side == 'right' else 1
            rho[:, :, k] = rho[:, :, k_int]
            rhou[:, :, k] = rhou[:, :, k_int]
            rhov[:, :, k] = rhov[:, :, k_int]
            rhow[:, :, k] = rhow[:, :, k_int]
            ke = 0.5 * (rhou[:, :, k] ** 2 + rhov[:, :, k] ** 2 + rhow[:, :, k] ** 2) / (rho[:, :, k] + 1e-8)
            rhoE[:, :, k] = self.p_out / (gamma - 1) + ke

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            if side == 'right':
                boundary_idx = nx - 1
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx + g
                    rho[idx] = rho[boundary_idx]
                    rhou[idx] = rhou[boundary_idx]
                    rhov[idx] = rhov[boundary_idx]
                    rhow[idx] = rhow[boundary_idx]
                    rhoE[idx] = rhoE[boundary_idx]
            else:
                boundary_idx = 0
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx - g
                    rho[idx] = rho[boundary_idx]
                    rhou[idx] = rhou[boundary_idx]
                    rhov[idx] = rhov[boundary_idx]
                    rhow[idx] = rhow[boundary_idx]
                    rhoE[idx] = rhoE[boundary_idx]
        elif axis == 1:
            if side == 'right':
                boundary_idx = ny - 1
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx + g
                    rho[:, idx] = rho[:, boundary_idx]
                    rhou[:, idx] = rhou[:, boundary_idx]
                    rhov[:, idx] = rhov[:, boundary_idx]
                    rhow[:, idx] = rhow[:, boundary_idx]
                    rhoE[:, idx] = rhoE[:, boundary_idx]
            else:
                boundary_idx = 0
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx - g
                    rho[:, idx] = rho[:, boundary_idx]
                    rhou[:, idx] = rhou[:, boundary_idx]
                    rhov[:, idx] = rhov[:, boundary_idx]
                    rhow[:, idx] = rhow[:, boundary_idx]
                    rhoE[:, idx] = rhoE[:, boundary_idx]
        else:
            if side == 'right':
                boundary_idx = nz - 1
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx + g
                    rho[:, :, idx] = rho[:, :, boundary_idx]
                    rhou[:, :, idx] = rhou[:, :, boundary_idx]
                    rhov[:, :, idx] = rhov[:, :, boundary_idx]
                    rhow[:, :, idx] = rhow[:, :, boundary_idx]
                    rhoE[:, :, idx] = rhoE[:, :, boundary_idx]
            else:
                boundary_idx = 0
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx - g
                    rho[:, :, idx] = rho[:, :, boundary_idx]
                    rhou[:, :, idx] = rhou[:, :, boundary_idx]
                    rhov[:, :, idx] = rhov[:, :, boundary_idx]
                    rhow[:, :, idx] = rhow[:, :, boundary_idx]
                    rhoE[:, :, idx] = rhoE[:, :, boundary_idx]


class NoSlipIsothermalWallBC(BoundaryCondition):
    def __init__(self, T_wall):
        self.T_wall = T_wall

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        nx, ny, nz = rho.shape
        if axis == 0:
            i = 0 if side == 'left' else nx - 1
            i_int = 1 if side == 'left' else nx - 2
            p_int = (gamma - 1) * (rhoE[i_int] - 0.5 * (
                rhou[i_int] ** 2 + rhov[i_int] ** 2 + rhow[i_int] ** 2
            ) / (rho[i_int] + 1e-8))
            rho[i] = p_int / self.T_wall
            rhou[i] = rhov[i] = rhow[i] = 0.0
            rhoE[i] = p_int / (gamma - 1)
        elif axis == 1:
            j = 0 if side == 'left' else ny - 1
            j_int = 1 if side == 'left' else ny - 2
            p_int = (gamma - 1) * (rhoE[:, j_int] - 0.5 * (
                rhou[:, j_int] ** 2 + rhov[:, j_int] ** 2 + rhow[:, j_int] ** 2
            ) / (rho[:, j_int] + 1e-8))
            rho[:, j] = p_int / self.T_wall
            rhou[:, j] = rhov[:, j] = rhow[:, j] = 0.0
            rhoE[:, j] = p_int / (gamma - 1)
        else:
            k = 0 if side == 'left' else nz - 1
            k_int = 1 if side == 'left' else nz - 2
            p_int = (gamma - 1) * (rhoE[:, :, k_int] - 0.5 * (
                rhou[:, :, k_int] ** 2 + rhov[:, :, k_int] ** 2 + rhow[:, :, k_int] ** 2
            ) / (rho[:, :, k_int] + 1e-8))
            rho[:, :, k] = p_int / self.T_wall
            rhou[:, :, k] = rhov[:, :, k] = rhow[:, :, k] = 0.0
            rhoE[:, :, k] = p_int / (gamma - 1)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    i = -g
                    i_int = g
                    rho[i] = rho[i_int]
                    rhoE[i] = rhoE[i_int]
                    rhou[i] = -rhou[i_int]
                    rhov[i] = -rhov[i_int]
                    rhow[i] = -rhow[i_int]
            else:
                boundary = nx - 1
                for g in range(1, n_ghost + 1):
                    i = boundary + g
                    i_int = boundary - g
                    rho[i] = rho[i_int]
                    rhoE[i] = rhoE[i_int]
                    rhou[i] = -rhou[i_int]
                    rhov[i] = -rhov[i_int]
                    rhow[i] = -rhow[i_int]
        elif axis == 1:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    j = -g
                    j_int = g
                    rho[:, j] = rho[:, j_int]
                    rhoE[:, j] = rhoE[:, j_int]
                    rhou[:, j] = -rhou[:, j_int]
                    rhov[:, j] = -rhov[:, j_int]
                    rhow[:, j] = -rhow[:, j_int]
            else:
                boundary = ny - 1
                for g in range(1, n_ghost + 1):
                    j = boundary + g
                    j_int = boundary - g
                    rho[:, j] = rho[:, j_int]
                    rhoE[:, j] = rhoE[:, j_int]
                    rhou[:, j] = -rhou[:, j_int]
                    rhov[:, j] = -rhov[:, j_int]
                    rhow[:, j] = -rhow[:, j_int]
        else:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    k = -g
                    k_int = g
                    rho[:, :, k] = rho[:, :, k_int]
                    rhoE[:, :, k] = rhoE[:, :, k_int]
                    rhou[:, :, k] = -rhou[:, :, k_int]
                    rhov[:, :, k] = -rhov[:, :, k_int]
                    rhow[:, :, k] = -rhow[:, :, k_int]
            else:
                boundary = nz - 1
                for g in range(1, n_ghost + 1):
                    k = boundary + g
                    k_int = boundary - g
                    rho[:, :, k] = rho[:, :, k_int]
                    rhoE[:, :, k] = rhoE[:, :, k_int]
                    rhou[:, :, k] = -rhou[:, :, k_int]
                    rhov[:, :, k] = -rhov[:, :, k_int]
                    rhow[:, :, k] = -rhow[:, :, k_int]


class WernerWengleWallModelBC(BoundaryCondition):
    def __init__(self, T_wall, A=8.3, B=1.0/7.0):
        self.T_wall = T_wall
        self.A = A
        self.B = B

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        NoSlipIsothermalWallBC(self.T_wall).apply(rho, rhou, rhov, rhow, rhoE,
                                                  axis, side, gamma, dx)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        NoSlipIsothermalWallBC(self.T_wall).ghost_cells(
            rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx, n_ghost)


class MovingWallBC(BoundaryCondition):
    def __init__(self, u_wall, v_wall, w_wall, T_wall):
        self.u_wall = u_wall
        self.v_wall = v_wall
        self.w_wall = w_wall
        self.T_wall = T_wall

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        nx, ny, nz = rho.shape
        if axis == 0:
            i = 0 if side == 'left' else nx - 1
            i_int = 1 if side == 'left' else nx - 2
            p_int = (gamma - 1) * (rhoE[i_int] - 0.5 * (
                rhou[i_int] ** 2 + rhov[i_int] ** 2 + rhow[i_int] ** 2
            ) / (rho[i_int] + 1e-8))
            rho[i] = p_int / self.T_wall
            rhou[i] = rho[i] * self.u_wall
            rhov[i] = rho[i] * self.v_wall
            rhow[i] = rho[i] * self.w_wall
            rhoE[i] = p_int / (gamma - 1) + 0.5 * rho[i] * (
                self.u_wall ** 2 + self.v_wall ** 2 + self.w_wall ** 2
            )
        elif axis == 1:
            j = 0 if side == 'left' else ny - 1
            j_int = 1 if side == 'left' else ny - 2
            p_int = (gamma - 1) * (rhoE[:, j_int] - 0.5 * (
                rhou[:, j_int] ** 2 + rhov[:, j_int] ** 2 + rhow[:, j_int] ** 2
            ) / (rho[:, j_int] + 1e-8))
            rho[:, j] = p_int / self.T_wall
            rhou[:, j] = rho[:, j] * self.u_wall
            rhov[:, j] = rho[:, j] * self.v_wall
            rhow[:, j] = rho[:, j] * self.w_wall
            rhoE[:, j] = p_int / (gamma - 1) + 0.5 * rho[:, j] * (
                self.u_wall ** 2 + self.v_wall ** 2 + self.w_wall ** 2
            )
        else:
            k = 0 if side == 'left' else nz - 1
            k_int = 1 if side == 'left' else nz - 2
            p_int = (gamma - 1) * (rhoE[:, :, k_int] - 0.5 * (
                rhou[:, :, k_int] ** 2 + rhov[:, :, k_int] ** 2 + rhow[:, :, k_int] ** 2
            ) / (rho[:, :, k_int] + 1e-8))
            rho[:, :, k] = p_int / self.T_wall
            rhou[:, :, k] = rho[:, :, k] * self.u_wall
            rhov[:, :, k] = rho[:, :, k] * self.v_wall
            rhow[:, :, k] = rho[:, :, k] * self.w_wall
            rhoE[:, :, k] = p_int / (gamma - 1) + 0.5 * rho[:, :, k] * (
                self.u_wall ** 2 + self.v_wall ** 2 + self.w_wall ** 2
            )

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    i = -g
                    i_int = g
                    rho[i] = rho[i_int]
                    rhoE[i] = rhoE[i_int]
                    rhou[i] = 2 * self.u_wall * rho[i] - rhou[i_int]
                    rhov[i] = 2 * self.v_wall * rho[i] - rhov[i_int]
                    rhow[i] = 2 * self.w_wall * rho[i] - rhow[i_int]
            else:
                boundary = nx - 1
                for g in range(1, n_ghost + 1):
                    i = boundary + g
                    i_int = boundary - g
                    rho[i] = rho[i_int]
                    rhoE[i] = rhoE[i_int]
                    rhou[i] = 2 * self.u_wall * rho[i] - rhou[i_int]
                    rhov[i] = 2 * self.v_wall * rho[i] - rhov[i_int]
                    rhow[i] = 2 * self.w_wall * rho[i] - rhow[i_int]
        elif axis == 1:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    j = -g
                    j_int = g
                    rho[:, j] = rho[:, j_int]
                    rhoE[:, j] = rhoE[:, j_int]
                    rhou[:, j] = 2 * self.u_wall * rho[:, j] - rhou[:, j_int]
                    rhov[:, j] = 2 * self.v_wall * rho[:, j] - rhov[:, j_int]
                    rhow[:, j] = 2 * self.w_wall * rho[:, j] - rhow[:, j_int]
            else:
                boundary = ny - 1
                for g in range(1, n_ghost + 1):
                    j = boundary + g
                    j_int = boundary - g
                    rho[:, j] = rho[:, j_int]
                    rhoE[:, j] = rhoE[:, j_int]
                    rhou[:, j] = 2 * self.u_wall * rho[:, j] - rhou[:, j_int]
                    rhov[:, j] = 2 * self.v_wall * rho[:, j] - rhov[:, j_int]
                    rhow[:, j] = 2 * self.w_wall * rho[:, j] - rhow[:, j_int]
        else:
            if side == 'left':
                for g in range(1, n_ghost + 1):
                    k = -g
                    k_int = g
                    rho[:, :, k] = rho[:, :, k_int]
                    rhoE[:, :, k] = rhoE[:, :, k_int]
                    rhou[:, :, k] = 2 * self.u_wall * rho[:, :, k] - rhou[:, :, k_int]
                    rhov[:, :, k] = 2 * self.v_wall * rho[:, :, k] - rhov[:, :, k_int]
                    rhow[:, :, k] = 2 * self.w_wall * rho[:, :, k] - rhow[:, :, k_int]
            else:
                boundary = nz - 1
                for g in range(1, n_ghost + 1):
                    k = boundary + g
                    k_int = boundary - g
                    rho[:, :, k] = rho[:, :, k_int]
                    rhoE[:, :, k] = rhoE[:, :, k_int]
                    rhou[:, :, k] = 2 * self.u_wall * rho[:, :, k] - rhou[:, :, k_int]
                    rhov[:, :, k] = 2 * self.v_wall * rho[:, :, k] - rhov[:, :, k_int]
                    rhow[:, :, k] = 2 * self.w_wall * rho[:, :, k] - rhow[:, :, k_int]


class FarFieldBC(BoundaryCondition):
    def __init__(self, rho_inf, u_inf, v_inf, w_inf, p_inf):
        self.rho_inf = rho_inf
        self.u_inf = u_inf
        self.v_inf = v_inf
        self.w_inf = w_inf
        self.p_inf = p_inf

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx):
        nx, ny, nz = rho.shape
        c_inf = math.sqrt(gamma * self.p_inf / self.rho_inf)
        if axis == 0:
            i = 0 if side == 'left' else nx - 1
            i_int = 1 if side == 'left' else nx - 2
            u = rhou[i_int] / (rho[i_int] + 1e-8)
            v = rhov[i_int] / (rho[i_int] + 1e-8)
            w = rhow[i_int] / (rho[i_int] + 1e-8)
            p = (gamma - 1) * (rhoE[i_int] - 0.5 * rho[i_int] * (u ** 2 + v ** 2 + w ** 2))
            c = torch.sqrt(gamma * p / (rho[i_int] + 1e-8))
            R_plus = u + 2 * c / (gamma - 1)
            R_minus = self.u_inf - 2 * c_inf / (gamma - 1)
            u_b = 0.5 * (R_plus + R_minus)
            c_b = 0.25 * (gamma - 1) * (R_plus - R_minus)
            s = p / (rho[i_int] ** gamma + 1e-8)
            p_b = (c_b ** 2 * s / gamma) ** (gamma / (gamma - 1))
            rho_b = p_b / (c_b ** 2 / gamma)
            rho[i] = rho_b
            rhou[i] = rho_b * u_b
            rhov[i] = rho_b * v
            rhow[i] = rho_b * w
            rhoE[i] = p_b / (gamma - 1) + 0.5 * rho_b * (u_b ** 2 + v ** 2 + w ** 2)
        elif axis == 1:
            j = 0 if side == 'left' else ny - 1
            j_int = 1 if side == 'left' else ny - 2
            u = rhou[:, j_int] / (rho[:, j_int] + 1e-8)
            v = rhov[:, j_int] / (rho[:, j_int] + 1e-8)
            w = rhow[:, j_int] / (rho[:, j_int] + 1e-8)
            p = (gamma - 1) * (rhoE[:, j_int] - 0.5 * rho[:, j_int] * (u ** 2 + v ** 2 + w ** 2))
            c = torch.sqrt(gamma * p / (rho[:, j_int] + 1e-8))
            R_plus = v + 2 * c / (gamma - 1)
            R_minus = self.v_inf - 2 * c_inf / (gamma - 1)
            v_b = 0.5 * (R_plus + R_minus)
            c_b = 0.25 * (gamma - 1) * (R_plus - R_minus)
            s = p / (rho[:, j_int] ** gamma + 1e-8)
            p_b = (c_b ** 2 * s / gamma) ** (gamma / (gamma - 1))
            rho_b = p_b / (c_b ** 2 / gamma)
            rho[:, j] = rho_b
            rhou[:, j] = rho_b * u
            rhov[:, j] = rho_b * v_b
            rhow[:, j] = rho_b * w
            rhoE[:, j] = p_b / (gamma - 1) + 0.5 * rho_b * (u ** 2 + v_b ** 2 + w ** 2)
        else:
            k = 0 if side == 'left' else nz - 1
            k_int = 1 if side == 'left' else nz - 2
            u = rhou[:, :, k_int] / (rho[:, :, k_int] + 1e-8)
            v = rhov[:, :, k_int] / (rho[:, :, k_int] + 1e-8)
            w = rhow[:, :, k_int] / (rho[:, :, k_int] + 1e-8)
            p = (gamma - 1) * (rhoE[:, :, k_int] - 0.5 * rho[:, :, k_int] * (u ** 2 + v ** 2 + w ** 2))
            c = torch.sqrt(gamma * p / (rho[:, :, k_int] + 1e-8))
            R_plus = w + 2 * c / (gamma - 1)
            R_minus = self.w_inf - 2 * c_inf / (gamma - 1)
            w_b = 0.5 * (R_plus + R_minus)
            c_b = 0.25 * (gamma - 1) * (R_plus - R_minus)
            s = p / (rho[:, :, k_int] ** gamma + 1e-8)
            p_b = (c_b ** 2 * s / gamma) ** (gamma / (gamma - 1))
            rho_b = p_b / (c_b ** 2 / gamma)
            rho[:, :, k] = rho_b
            rhou[:, :, k] = rho_b * u
            rhov[:, :, k] = rho_b * v
            rhow[:, :, k] = rho_b * w_b
            rhoE[:, :, k] = p_b / (gamma - 1) + 0.5 * rho_b * (u ** 2 + v ** 2 + w_b ** 2)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx,
                    n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            if side == 'right':
                boundary_idx = nx - 1
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx + g
                    rho[idx] = rho[boundary_idx]
                    rhou[idx] = rhou[boundary_idx]
                    rhov[idx] = rhov[boundary_idx]
                    rhow[idx] = rhow[boundary_idx]
                    rhoE[idx] = rhoE[boundary_idx]
            else:
                boundary_idx = 0
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx - g
                    rho[idx] = rho[boundary_idx]
                    rhou[idx] = rhou[boundary_idx]
                    rhov[idx] = rhov[boundary_idx]
                    rhow[idx] = rhow[boundary_idx]
                    rhoE[idx] = rhoE[boundary_idx]
        elif axis == 1:
            if side == 'right':
                boundary_idx = ny - 1
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx + g
                    rho[:, idx] = rho[:, boundary_idx]
                    rhou[:, idx] = rhou[:, boundary_idx]
                    rhov[:, idx] = rhov[:, boundary_idx]
                    rhow[:, idx] = rhow[:, boundary_idx]
                    rhoE[:, idx] = rhoE[:, boundary_idx]
            else:
                boundary_idx = 0
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx - g
                    rho[:, idx] = rho[:, boundary_idx]
                    rhou[:, idx] = rhou[:, boundary_idx]
                    rhov[:, idx] = rhov[:, boundary_idx]
                    rhow[:, idx] = rhow[:, boundary_idx]
                    rhoE[:, idx] = rhoE[:, boundary_idx]
        else:
            if side == 'right':
                boundary_idx = nz - 1
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx + g
                    rho[:, :, idx] = rho[:, :, boundary_idx]
                    rhou[:, :, idx] = rhou[:, :, boundary_idx]
                    rhov[:, :, idx] = rhov[:, :, boundary_idx]
                    rhow[:, :, idx] = rhow[:, :, boundary_idx]
                    rhoE[:, :, idx] = rhoE[:, :, boundary_idx]
            else:
                boundary_idx = 0
                for g in range(1, n_ghost + 1):
                    idx = boundary_idx - g
                    rho[:, :, idx] = rho[:, :, boundary_idx]
                    rhou[:, :, idx] = rhou[:, :, boundary_idx]
                    rhov[:, :, idx] = rhov[:, :, boundary_idx]
                    rhow[:, :, idx] = rhow[:, :, boundary_idx]
                    rhoE[:, :, idx] = rhoE[:, :, boundary_idx]


# =============================================================================
# 2. Riemann Solvers (Face‑Centred, Conservative)
# =============================================================================
class RiemannSolverBase:
    def __init__(self, gamma):
        self.gamma = gamma

    def _muscl_face_states(self, q_pad, axis):
        nx, ny, nz = q_pad.shape[0] - 4, q_pad.shape[1] - 4, q_pad.shape[2] - 4
        if axis == 0:
            q_im1 = q_pad[0:nx+2, 2:ny+2, 2:nz+2]
            q_i   = q_pad[1:nx+3, 2:ny+2, 2:nz+2]
            q_ip1 = q_pad[2:nx+4, 2:ny+2, 2:nz+2]
            d1 = q_i - q_im1
            d2 = q_ip1 - q_i
            slope = self._minmod(d1, d2)
            qL = q_i + 0.5 * slope
            slope_ip1 = torch.cat([slope[1:], slope[-1:]], dim=0)
            qR = q_ip1 - 0.5 * slope_ip1
            qL = qL[:nx+1]
            qR = qR[:nx+1]
        elif axis == 1:
            q_im1 = q_pad[2:nx+2, 0:ny+2, 2:nz+2]
            q_i   = q_pad[2:nx+2, 1:ny+3, 2:nz+2]
            q_ip1 = q_pad[2:nx+2, 2:ny+4, 2:nz+2]
            d1 = q_i - q_im1
            d2 = q_ip1 - q_i
            slope = self._minmod(d1, d2)
            qL = q_i + 0.5 * slope
            slope_ip1 = torch.cat([slope[:, 1:], slope[:, -1:]], dim=1)
            qR = q_ip1 - 0.5 * slope_ip1
            qL = qL[:, :ny+1]
            qR = qR[:, :ny+1]
        else:
            q_im1 = q_pad[2:nx+2, 2:ny+2, 0:nz+2]
            q_i   = q_pad[2:nx+2, 2:ny+2, 1:nz+3]
            q_ip1 = q_pad[2:nx+2, 2:ny+2, 2:nz+4]
            d1 = q_i - q_im1
            d2 = q_ip1 - q_i
            slope = self._minmod(d1, d2)
            qL = q_i + 0.5 * slope
            slope_ip1 = torch.cat([slope[:, :, 1:], slope[:, :, -1:]], dim=2)
            qR = q_ip1 - 0.5 * slope_ip1
            qL = qL[:, :, :nz+1]
            qR = qR[:, :, :nz+1]
        return qL, qR

    @staticmethod
    def _minmod(a, b):
        return torch.where(a * b > 0, torch.where(torch.abs(a) < torch.abs(b), a, b), torch.zeros_like(a))


class AUSMPlusFlux(RiemannSolverBase):
    def __init__(self, gamma=1.4):
        super().__init__(gamma)

    def compute_face_flux(self, rho_pad, u_pad, v_pad, w_pad, p_pad, axis):
        rhoL, rhoR = self._muscl_face_states(rho_pad, axis)
        uL, uR     = self._muscl_face_states(u_pad, axis)
        vL, vR     = self._muscl_face_states(v_pad, axis)
        wL, wR     = self._muscl_face_states(w_pad, axis)
        pL, pR     = self._muscl_face_states(p_pad, axis)

        gamma = self.gamma
        cL = torch.sqrt(gamma * pL / (rhoL + 1e-8))
        cR = torch.sqrt(gamma * pR / (rhoR + 1e-8))
        c_face = 0.5 * (cL + cR)

        if axis == 0:
            unL, unR = uL, uR
            utL, utR = vL, vR
            uwL, uwR = wL, wR
        elif axis == 1:
            unL, unR = vL, vR
            utL, utR = uL, uR
            uwL, uwR = wL, wR
        else:
            unL, unR = wL, wR
            utL, utR = uL, uR
            uwL, uwR = vL, vR

        M_L = unL / (c_face + 1e-8)
        M_R = unR / (c_face + 1e-8)

        def M_plus(M):
            return torch.where(M >= 1, M, 0.25 * (M + 1) ** 2)
        def M_minus(M):
            return torch.where(M <= -1, M, -0.25 * (M - 1) ** 2)
        def p_plus(M):
            return torch.where(M >= 1, torch.ones_like(M), 0.25 * (M + 1) ** 2 * (2 - M))
        def p_minus(M):
            return torch.where(M <= -1, torch.zeros_like(M), 0.25 * (M - 1) ** 2 * (2 + M))

        M_face = M_plus(M_L) + M_minus(M_R)
        p_face = p_plus(M_L) * pL + p_minus(M_R) * pR

        mass_flux = c_face * (torch.where(M_face >= 0, M_face * rhoL, M_face * rhoR))

        flux_n  = torch.where(M_face >= 0, mass_flux * unL, mass_flux * unR) + p_face
        flux_t1 = torch.where(M_face >= 0, mass_flux * utL, mass_flux * utR)
        flux_t2 = torch.where(M_face >= 0, mass_flux * uwL, mass_flux * uwR)

        EL = pL / (gamma - 1) + 0.5 * rhoL * (uL ** 2 + vL ** 2 + wL ** 2)
        ER = pR / (gamma - 1) + 0.5 * rhoR * (uR ** 2 + vR ** 2 + wR ** 2)
        HL = (EL + pL) / (rhoL + 1e-8)
        HR = (ER + pR) / (rhoR + 1e-8)
        flux_E = torch.where(M_face >= 0, mass_flux * HL, mass_flux * HR)

        if axis == 0:
            return mass_flux, flux_n, flux_t1, flux_t2, flux_E
        elif axis == 1:
            return mass_flux, flux_t1, flux_n, flux_t2, flux_E
        else:
            return mass_flux, flux_t1, flux_t2, flux_n, flux_E


class HLLCFlux(RiemannSolverBase):
    def __init__(self, gamma=1.4):
        super().__init__(gamma)

    def compute_face_flux(self, rho_pad, u_pad, v_pad, w_pad, p_pad, axis):
        rhoL, rhoR = self._muscl_face_states(rho_pad, axis)
        uL, uR     = self._muscl_face_states(u_pad, axis)
        vL, vR     = self._muscl_face_states(v_pad, axis)
        wL, wR     = self._muscl_face_states(w_pad, axis)
        pL, pR     = self._muscl_face_states(p_pad, axis)

        gamma = self.gamma
        if axis == 0:
            unL, unR = uL, uR; utL, utR = vL, vR; uwL, uwR = wL, wR
        elif axis == 1:
            unL, unR = vL, vR; utL, utR = uL, uR; uwL, uwR = wL, wR
        else:
            unL, unR = wL, wR; utL, utR = uL, uR; uwL, uwR = vL, vR

        cL = torch.sqrt(gamma * pL / (rhoL + 1e-8))
        cR = torch.sqrt(gamma * pR / (rhoR + 1e-8))
        R = torch.sqrt(rhoR / (rhoL + 1e-8))
        un_roe = (unL + R * unR) / (1 + R)
        c_roe  = (cL + R * cR) / (1 + R)
        SL = torch.min(unL - cL, un_roe - c_roe)
        SR = torch.max(unR + cR, un_roe + c_roe)
        S_star = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) / \
                 (rhoL * (SL - unL) - rhoR * (SR - unR) + 1e-8)

        mask_L = SL >= 0
        mask_R = SR <= 0
        mask_star = ~(mask_L | mask_R)

        rho_face = torch.zeros_like(rhoL)
        un_face  = torch.zeros_like(rhoL)
        p_face   = torch.zeros_like(pL)
        ut_face  = torch.zeros_like(rhoL)
        uw_face  = torch.zeros_like(rhoL)
        E_face   = torch.zeros_like(rhoL)

        rho_face[mask_L] = rhoL[mask_L]
        un_face[mask_L]  = unL[mask_L]
        p_face[mask_L]   = pL[mask_L]
        ut_face[mask_L]  = utL[mask_L]
        uw_face[mask_L]  = uwL[mask_L]
        E_face[mask_L]   = pL[mask_L] / (gamma - 1) + 0.5 * rhoL[mask_L] * (
            unL[mask_L] ** 2 + utL[mask_L] ** 2 + uwL[mask_L] ** 2)

        rho_face[mask_R] = rhoR[mask_R]
        un_face[mask_R]  = unR[mask_R]
        p_face[mask_R]   = pR[mask_R]
        ut_face[mask_R]  = utR[mask_R]
        uw_face[mask_R]  = uwR[mask_R]
        E_face[mask_R]   = pR[mask_R] / (gamma - 1) + 0.5 * rhoR[mask_R] * (
            unR[mask_R] ** 2 + utR[mask_R] ** 2 + uwR[mask_R] ** 2)

        if mask_star.any():
            rhoL_s = rhoL[mask_star]; unL_s = unL[mask_star]; SL_s = SL[mask_star]
            S_star_s = S_star[mask_star]; pL_s = pL[mask_star]
            utL_s = utL[mask_star]; uwL_s = uwL[mask_star]
            rhoR_s = rhoR[mask_star]; unR_s = unR[mask_star]; SR_s = SR[mask_star]
            pR_s = pR[mask_star]; utR_s = utR[mask_star]; uwR_s = uwR[mask_star]

            factorL = (SL_s - unL_s) / (SL_s - S_star_s + 1e-8)
            rho_starL = rhoL_s * factorL
            p_starL = pL_s + rhoL_s * (unL_s - SL_s) * (unL_s - S_star_s)
            E_starL = p_starL / (gamma - 1) + 0.5 * rho_starL * (S_star_s ** 2 + utL_s ** 2 + uwL_s ** 2)

            factorR = (SR_s - unR_s) / (SR_s - S_star_s + 1e-8)
            rho_starR = rhoR_s * factorR
            p_starR = pR_s + rhoR_s * (unR_s - SR_s) * (unR_s - S_star_s)
            E_starR = p_starR / (gamma - 1) + 0.5 * rho_starR * (S_star_s ** 2 + utR_s ** 2 + uwR_s ** 2)

            star_left = S_star_s >= 0
            rho_face[mask_star] = torch.where(star_left, rho_starL, rho_starR)
            un_face[mask_star]  = S_star_s
            p_face[mask_star]   = torch.where(star_left, p_starL, p_starR)
            ut_face[mask_star]  = torch.where(star_left, utL_s, utR_s)
            uw_face[mask_star]  = torch.where(star_left, uwL_s, uwR_s)
            E_face[mask_star]   = torch.where(star_left, E_starL, E_starR)

        mass_flux = rho_face * un_face
        flux_n = mass_flux * un_face + p_face
        flux_t1 = mass_flux * ut_face
        flux_t2 = mass_flux * uw_face
        H_face = (E_face + p_face) / (rho_face + 1e-8)
        flux_E = mass_flux * H_face

        if axis == 0:
            return mass_flux, flux_n, flux_t1, flux_t2, flux_E
        elif axis == 1:
            return mass_flux, flux_t1, flux_n, flux_t2, flux_E
        else:
            return mass_flux, flux_t1, flux_t2, flux_n, flux_E


# =============================================================================
# 3. Sub‑grid Scale Models (SOC, Itô, RG, SSC)
# =============================================================================
class CSOCKernel(nn.Module):
    def __init__(self, init_Cs=0.18, init_lambda=12.0, init_alpha=0.5,
                 init_theta=1.0, init_tau=10.0, device='cpu'):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(init_Cs), device=device))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda), device=device))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), device=device))
        self.log_theta = nn.Parameter(torch.tensor(math.log(init_theta), device=device))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), device=device))

    @property
    def Cs(self): return torch.exp(self.log_Cs)
    @property
    def lambd(self): return torch.exp(self.log_lambda)
    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def theta(self): return torch.exp(self.log_theta)
    @property
    def tau(self): return torch.exp(self.log_tau)

    def forward(self, r):
        return self.Cs * torch.pow(r + 1e-6, -self.alpha) * torch.exp(-r / self.lambd)


class SemanticStateContraction:
    """SSC low‑pass filter for stress sigma."""
    def __init__(self, epsilon_fp=0.0028, sigma_target=1.0):
        self.eps = epsilon_fp
        self.target = sigma_target
        self.prev = torch.tensor(0.0)

    def __call__(self, sigma):
        if self.prev.item() == 0.0 or self.prev.device != sigma.device:
            self.prev = sigma.detach().to(sigma.device)
            return sigma
        new = self.prev + self.eps * (sigma - self.prev)
        self.prev = new.detach()
        return new


class SOCController:
    def __init__(self, base_temp=300.0, max_nu_t=0.01, use_ssc=True,
                 epsilon_fp=0.0028, compressibility_correction=True, device='cpu'):
        self.base_temp = base_temp
        self.max_nu_t = max_nu_t
        self.compressibility_correction = compressibility_correction
        self.kernel = CSOCKernel(device=device).to(device)
        self.ssc = SemanticStateContraction(epsilon_fp) if use_ssc else None
        self.stress_acc = None
        self.device = device

    def reset(self):
        self.stress_acc = None

    def nu_t(self, rho, strain_rate_mag, dilatation, dx, dt, c):
        mean_S = torch.mean(strain_rate_mag) + 1e-8
        r = strain_rate_mag / mean_S
        Cs_local = self.kernel(r)
        nu_t_base = (Cs_local * dx) ** 2 * strain_rate_mag

        if self.compressibility_correction:
            M_t = torch.sqrt(2.0 * nu_t_base * strain_rate_mag) / (c + 1e-8)
            f_dil = 1.0 / (1.0 + 2.0 * M_t ** 2)
            nu_t_base = nu_t_base * f_dil

        if self.stress_acc is None:
            self.stress_acc = torch.zeros_like(strain_rate_mag)
        tau = self.kernel.tau
        dS = strain_rate_mag ** 2 - (1.0 / tau) * self.stress_acc
        self.stress_acc = self.stress_acc + dt * dS
        self.stress_acc = torch.clamp(self.stress_acc, min=0.0)
        if self.ssc is not None:
            ssc_val = self.ssc(self.stress_acc.mean())
        theta = self.kernel.theta
        excess = torch.clamp(self.stress_acc - theta, min=0.0)
        nu_collapse = 0.1 * excess * dx ** 2
        self.stress_acc = torch.where(excess > 0, theta * 0.5, self.stress_acc)

        nu_t_total = nu_t_base + nu_collapse
        return torch.clamp(nu_t_total, 0.0, self.max_nu_t)


class ItoStressGenerator:
    def __init__(self, noise_amp=0.001):
        self.noise_amp = noise_amp

    def generate(self, shape, device, dt):
        amp = self.noise_amp * math.sqrt(dt)
        return (amp * torch.randn(shape, device=device),
                amp * torch.randn(shape, device=device),
                amp * torch.randn(shape, device=device),
                amp * torch.randn(shape, device=device),
                amp * torch.randn(shape, device=device),
                amp * torch.randn(shape, device=device))


class DiffRGRefiner:
    def __init__(self, keep_fraction=0.5):
        self.keep_fraction = keep_fraction

    def forward(self, x):
        nx, ny, nz = x.shape
        x_hat = torch.fft.rfftn(x)
        kx = torch.fft.fftfreq(nx, d=1.0, device=x.device)
        ky = torch.fft.fftfreq(ny, d=1.0, device=x.device)
        kz = torch.fft.rfftfreq(nz, d=1.0, device=x.device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
        mask = K_mag <= (self.keep_fraction * K_mag.max())
        mask[0, 0, 0] = True
        return torch.fft.irfftn(x_hat * mask.to(x_hat.dtype), s=(nx, ny, nz))


# =============================================================================
# 4. Real‑Gas Equation of State (CoolProp wrapper)
# =============================================================================
class RealGasEOS:
    """
    Real‑gas thermodynamics using CoolProp.
    Supports vectorised evaluation via numpy and returns torch tensors.
    Falls back to ideal gas if CoolProp is not available.
    """
    def __init__(self, fluid='Air', gamma=1.4):
        self.fluid = fluid
        self.gamma = gamma
        self.use_real = HAS_COOLPROP
        if not self.use_real:
            logger.warning("CoolProp not found. Falling back to ideal gas EOS.")

    def pressure(self, rho, e):
        """Compute pressure [Pa] from density [kg/m3] and internal energy [J/kg]."""
        if not self.use_real:
            return (self.gamma - 1) * rho * e
        rho_np = rho.detach().cpu().numpy().ravel()
        e_np = e.detach().cpu().numpy().ravel()
        p_np = np.array([CP.PropsSI('P', 'D', d, 'U', u, self.fluid)
                         for d, u in zip(rho_np, e_np)])
        p_np = p_np.reshape(rho.shape)
        return torch.tensor(p_np, device=rho.device, dtype=rho.dtype)

    def sound_speed(self, rho, e):
        """Speed of sound [m/s] from density and internal energy."""
        if not self.use_real:
            p = self.pressure(rho, e)
            return torch.sqrt(self.gamma * p / (rho + 1e-8))
        rho_np = rho.detach().cpu().numpy().ravel()
        e_np = e.detach().cpu().numpy().ravel()
        c_np = np.array([CP.PropsSI('A', 'D', d, 'U', u, self.fluid)
                         for d, u in zip(rho_np, e_np)])
        c_np = c_np.reshape(rho.shape)
        return torch.tensor(c_np, device=rho.device, dtype=rho.dtype)


# =============================================================================
# 5. Immersed Boundary Method (Volume Penalisation)
# =============================================================================
class ImmersedBoundary:
    """
    Immersed boundary method using volume penalisation.
    Adds forcing terms to momentum and energy equations inside the solid.
    """
    def __init__(self, mask: torch.Tensor, eta=1e4, u_target=(0.,0.,0.),
                 T_target=None, eta_T=1e4, gamma=1.4):
        self.mask = mask.to(torch.bool)
        self.eta = eta
        self.u_target = u_target
        self.T_target = T_target
        self.eta_T = eta_T
        self.gamma = gamma

    def apply_forcing(self, rho, rhou, rhov, rhow, rhoE, dt):
        u = rhou / (rho + 1e-8)
        v = rhov / (rho + 1e-8)
        w = rhow / (rho + 1e-8)
        force_x = -self.eta * self.mask * (u - self.u_target[0])
        force_y = -self.eta * self.mask * (v - self.u_target[1])
        force_z = -self.eta * self.mask * (w - self.u_target[2])
        rhou = rhou + dt * force_x * rho
        rhov = rhov + dt * force_y * rho
        rhow = rhow + dt * force_z * rho

        if self.T_target is not None:
            ke = 0.5 * (rhou**2 + rhov**2 + rhow**2) / (rho + 1e-8)
            p = (self.gamma - 1) * (rhoE - ke)
            T = p / (rho + 1e-8)
            rhoE_target = rho * self.T_target / (self.gamma - 1)
            penalty_E = -self.eta_T * self.mask * (rhoE - rhoE_target)
            rhoE = rhoE + dt * penalty_E

        return rhou, rhov, rhow, rhoE


# =============================================================================
# 6. Configuration & Main Solver (with distributed memory parallelism)
# =============================================================================
class CFDConfig:
    def __init__(self, nx=64, ny=64, nz=64,
                 Lx=2*math.pi, Ly=2*math.pi, Lz=2*math.pi,
                 Re=1e4, Pr=0.71, gamma=1.4, Mach=0.1, cfl=0.5, steps=500,
                 soc_base_temp=300.0, max_nu_t=0.05,
                 use_rg=False, rg_keep_frac=0.5, rg_interval=10,
                 ito_noise=0.0, muscl=True, use_sutherland=True,
                 device='cuda', flux_scheme='ausm', shock_capturing=False,
                 compressibility_correction=True, ssc_epsilon=0.0028,
                 dtype=torch.float32,
                 bc_x_min='periodic', bc_x_max='periodic',
                 bc_y_min='periodic', bc_y_max='periodic',
                 bc_z_min='periodic', bc_z_max='periodic',
                 inflow_rho=1.0, inflow_u=0.0, inflow_v=0.0, inflow_w=0.0, inflow_p=1.0,
                 outflow_p=1.0, wall_temp=300.0,
                 farfield_rho=1.0, farfield_u=0.0, farfield_v=0.0,
                 farfield_w=0.0, farfield_p=1.0,
                 moving_wall_u=0.0, moving_wall_v=0.0, moving_wall_w=0.0,
                 use_wall_model=False, wm_A=8.3, wm_B=1.0/7.0,
                 eos_model='ideal', fluid_name='Air',
                 ib_mask_file=None, ib_eta=1e4, ib_T_target=None, ib_eta_T=1e4,
                 distributed=False):
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz
        if abs(self.dx - self.dy) > 1e-10 or abs(self.dy - self.dz) > 1e-10:
            raise ValueError("Uniform grid spacing is required.")


class CompressibleSolver:
    def __init__(self, cfg: CFDConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.dtype = cfg.dtype
        self.dx = cfg.dx
        self.gamma = cfg.gamma
        self.nu_phys = 1.0 / cfg.Re if cfg.Re > 0 else 0.0
        self.Pr = cfg.Pr

        # Distributed parallel setup
        self.distributed = cfg.distributed
        if self.distributed:
            if not dist.is_initialized():
                raise RuntimeError("Distributed environment not initialized. Call init_process_group before creating the solver.")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            # Domain decomposition along the z‑axis
            if cfg.nz % self.world_size != 0:
                raise ValueError(f"nz ({cfg.nz}) must be divisible by world_size ({self.world_size})")
            self.local_nz = cfg.nz // self.world_size
            self.z_start = self.rank * self.local_nz
            self.z_end   = self.z_start + self.local_nz
            # Determine neighbors for halo exchange (periodic in z if both BCs are periodic)
            if cfg.bc_z_min == 'periodic' and cfg.bc_z_max == 'periodic':
                self.neighbor_left  = (self.rank - 1) % self.world_size
                self.neighbor_right = (self.rank + 1) % self.world_size
                self.z_periodic_dist = True
            else:
                self.neighbor_left  = self.rank - 1 if self.rank > 0 else -1
                self.neighbor_right = self.rank + 1 if self.rank < self.world_size - 1 else -1
                self.z_periodic_dist = False
            logger.info(f"Rank {self.rank}/{self.world_size}: local_nz = {self.local_nz}, "
                        f"neighbors left={self.neighbor_left}, right={self.neighbor_right}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_nz = cfg.nz

        self.soc = SOCController(
            base_temp=cfg.soc_base_temp,
            max_nu_t=cfg.max_nu_t,
            use_ssc=True,
            epsilon_fp=cfg.ssc_epsilon,
            compressibility_correction=cfg.compressibility_correction,
            device=self.device
        )
        self.ito_gen = ItoStressGenerator(noise_amp=cfg.ito_noise) if cfg.ito_noise > 0 else None
        self.rg = DiffRGRefiner(keep_fraction=cfg.rg_keep_frac) if cfg.use_rg else None

        if cfg.flux_scheme == 'ausm':
            self.flux_solver = AUSMPlusFlux(gamma=cfg.gamma)
        else:
            self.flux_solver = HLLCFlux(gamma=cfg.gamma)

        self._init_bc_objects()
        self.eos = RealGasEOS(cfg.fluid_name, gamma=cfg.gamma) if cfg.eos_model == 'real' else None

        self.ib = None
        if cfg.ib_mask_file:
            full_mask = np.load(cfg.ib_mask_file)
            if self.distributed:
                # Slice the mask according to the local z-slab
                local_mask = full_mask[:, :, self.z_start:self.z_end]
            else:
                local_mask = full_mask
            mask = torch.tensor(local_mask, dtype=torch.bool, device=self.device)
            self.ib = ImmersedBoundary(mask, eta=cfg.ib_eta,
                                       T_target=cfg.ib_T_target,
                                       eta_T=cfg.ib_eta_T,
                                       gamma=cfg.gamma)

        self.rho  = None
        self.rhou = None
        self.rhov = None
        self.rhow = None
        self.rhoE = None
        self.step_count = 0
        self.time = 0.0
        self.energy_hist = []
        self.div_hist = []

        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.use_amp = False

        self.wall_model_faces = []
        for face, bc_type in [
            ('xmin', cfg.bc_x_min), ('xmax', cfg.bc_x_max),
            ('ymin', cfg.bc_y_min), ('ymax', cfg.bc_y_max),
            ('zmin', cfg.bc_z_min), ('zmax', cfg.bc_z_max)
        ]:
            if bc_type == 'werner_wengle':
                self.wall_model_faces.append(face)

    def _init_bc_objects(self):
        self.bc_objects = {}
        bc_map = {
            'xmin': self.cfg.bc_x_min, 'xmax': self.cfg.bc_x_max,
            'ymin': self.cfg.bc_y_min, 'ymax': self.cfg.bc_y_max,
            'zmin': self.cfg.bc_z_min, 'zmax': self.cfg.bc_z_max
        }
        for face, bc_type in bc_map.items():
            if bc_type == 'periodic':
                self.bc_objects[face] = PeriodicBC()
            elif bc_type == 'supersonic_inflow':
                self.bc_objects[face] = SupersonicInflowBC(
                    self.cfg.inflow_rho, self.cfg.inflow_u, self.cfg.inflow_v,
                    self.cfg.inflow_w, self.cfg.inflow_p)
            elif bc_type == 'subsonic_outflow':
                self.bc_objects[face] = SubsonicOutflowBC(self.cfg.outflow_p)
            elif bc_type == 'noslip_isothermal':
                self.bc_objects[face] = NoSlipIsothermalWallBC(self.cfg.wall_temp)
            elif bc_type == 'werner_wengle':
                self.bc_objects[face] = WernerWengleWallModelBC(self.cfg.wall_temp,
                                                                self.cfg.wm_A, self.cfg.wm_B)
            elif bc_type == 'moving_wall':
                self.bc_objects[face] = MovingWallBC(
                    self.cfg.moving_wall_u, self.cfg.moving_wall_v,
                    self.cfg.moving_wall_w, self.cfg.wall_temp)
            elif bc_type == 'farfield':
                self.bc_objects[face] = FarFieldBC(
                    self.cfg.farfield_rho, self.cfg.farfield_u, self.cfg.farfield_v,
                    self.cfg.farfield_w, self.cfg.farfield_p)
            else:
                raise ValueError(f"Unknown BC type: {bc_type}")

    def _apply_bc_to_boundary_cells(self, rho, rhou, rhov, rhow, rhoE):
        for axis, side, face in [(0, 'left', 'xmin'), (0, 'right', 'xmax'),
                                 (1, 'left', 'ymin'), (1, 'right', 'ymax'),
                                 (2, 'left', 'zmin'), (2, 'right', 'zmax')]:
            # In distributed mode, only apply physical BCs on the actual domain boundaries,
            # not on the inter‑processor boundaries (they are handled by halo exchange).
            if self.distributed and axis == 2:
                if side == 'left' and self.rank != 0:
                    continue
                if side == 'right' and self.rank != self.world_size - 1:
                    continue
            self.bc_objects[face].apply(rho, rhou, rhov, rhow, rhoE,
                                        axis, side, self.gamma, self.dx)

    def _is_periodic_dim(self, dim):
        if dim == 0:
            return self.cfg.bc_x_min == 'periodic' and self.cfg.bc_x_max == 'periodic'
        elif dim == 1:
            return self.cfg.bc_y_min == 'periodic' and self.cfg.bc_y_max == 'periodic'
        else:
            return self.cfg.bc_z_min == 'periodic' and self.cfg.bc_z_max == 'periodic'

    def _pad_field(self, f):
        # Pad with 2 ghost cells on each side in every direction.
        # In distributed mode, ghost cells in z will be filled later by halo exchange.
        if all(self._is_periodic_dim(d) for d in range(3)) and not self.distributed:
            return F.pad(f, (2,2,2,2,2,2), mode='circular')
        padded = torch.zeros(f.shape[0]+4, f.shape[1]+4, f.shape[2]+4,
                             dtype=f.dtype, device=f.device)
        padded[2:-2, 2:-2, 2:-2] = f
        return padded

    def _exchange_halo_z(self, *fields):
        """
        Exchange ghost layers in z‑direction using send/recv.
        Each field is a 3D tensor (nx, ny, local_nz+4) after padding.
        We exchange two layers on each side.
        """
        if not self.distributed or self.world_size == 1:
            return
        # fields: tuples of (rho_p, rhou_p, rhov_p, rhow_p, rhoE_p, u_p, v_p, w_p, p_p)
        # For simplicity, we pack all fields into a single buffer to reduce communication calls.
        # Here we just loop over fields; more efficient packing possible but okay for demonstration.
        for f in fields:
            # Send to left neighbor (lower rank)
            if self.neighbor_left >= 0:
                send_buf = f[:, :, 2:4].contiguous()  # two inner layers
                dist.send(send_buf, self.neighbor_left)
            # Receive from right neighbor (higher rank) into left ghost
            if self.neighbor_right >= 0:
                recv_buf = torch.zeros_like(f[:, :, 0:2])
                dist.recv(recv_buf, self.neighbor_right)
                f[:, :, 0:2] = recv_buf
            # Send to right neighbor
            if self.neighbor_right >= 0:
                send_buf = f[:, :, -4:-2].contiguous()
                dist.send(send_buf, self.neighbor_right)
            # Receive from left neighbor into right ghost
            if self.neighbor_left >= 0:
                recv_buf = torch.zeros_like(f[:, :, -2:])
                dist.recv(recv_buf, self.neighbor_left)
                f[:, :, -2:] = recv_buf
            # For periodic distributed z, wrap‑around is handled by the neighbor indices already.
            # No additional synchronization needed because sends/recvs are paired.

    def _fill_ghost_cells(self, rho_p, rhou_p, rhov_p, rhow_p, rhoE_p):
        # Apply physical BC ghost cells for all axes except the ones handled by halo exchange.
        for axis, side, face in [(0, 'left', 'xmin'), (0, 'right', 'xmax'),
                                 (1, 'left', 'ymin'), (1, 'right', 'ymax'),
                                 (2, 'left', 'zmin'), (2, 'right', 'zmax')]:
            if self.distributed and axis == 2:
                # In distributed mode, the physical BC ghost filling is only applied on the true domain boundaries.
                if side == 'left' and self.rank != 0:
                    continue
                if side == 'right' and self.rank != self.world_size - 1:
                    continue
            if not self._is_periodic_dim(axis):
                self.bc_objects[face].ghost_cells(
                    rho_p, rhou_p, rhov_p, rhow_p, rhoE_p, axis, side, self.gamma, self.dx)

    def _apply_wall_model(self, rho, rhou, rhov, rhow, rhoE, dt):
        if not self.wall_model_faces:
            return
        gamma = self.gamma
        dx = self.dx
        T_ref = 300.0
        S = 110.4 / T_ref

        face_map = {
            'xmin': (0, 'left'), 'xmax': (0, 'right'),
            'ymin': (1, 'left'), 'ymax': (1, 'right'),
            'zmin': (2, 'left'), 'zmax': (2, 'right')
        }

        for face in self.wall_model_faces:
            axis, side = face_map[face]
            # In distributed mode, skip zmin/zmax if not the owning rank
            if self.distributed and axis == 2:
                if side == 'left' and self.rank != 0:
                    continue
                if side == 'right' and self.rank != self.world_size - 1:
                    continue
            nx, ny, nz = rho.shape
            # ... same implementation as before, but using local nz ...
            # (the wall model code remains unchanged, operating on the local slab)
            if axis == 0:
                if side == 'left':
                    i_cell = 1
                else:
                    i_cell = nx - 2
                rho_slice = rho[i_cell, :, :]
                u_slice   = rhou[i_cell, :, :] / (rho_slice + 1e-8)
                v_slice   = rhov[i_cell, :, :] / (rho_slice + 1e-8)
                w_slice   = rhow[i_cell, :, :] / (rho_slice + 1e-8)
                ut = torch.sqrt(v_slice**2 + w_slice**2)
                y = 0.5 * dx
                ke = 0.5 * rho_slice * (u_slice**2 + v_slice**2 + w_slice**2)
                p = (gamma - 1) * (rhoE[i_cell, :, :] - ke)
                T = p / (rho_slice + 1e-8)
                mu_lam = self.nu_phys * rho_slice * T.pow(1.5) * (1 + S) / (T + S)
                nu = mu_lam / (rho_slice + 1e-8)
                u_tau = torch.sqrt(nu * ut / (y + 1e-8) + 1e-8)
                for _ in range(5):
                    y_plus = y * u_tau / (nu + 1e-8)
                    u_plus = torch.where(y_plus <= 11.81, y_plus, self.cfg.wm_A * y_plus**self.cfg.wm_B)
                    u_tau_new = ut / (u_plus + 1e-8)
                    u_tau = 0.5 * (u_tau + u_tau_new)
                tau_w = rho_slice * u_tau**2
                ut_mag = ut + 1e-8
                tau_y = -tau_w * (v_slice / ut_mag)
                tau_z = -tau_w * (w_slice / ut_mag)
                src_v = tau_y / dx
                src_w = tau_z / dx
                rhov[i_cell, :, :] += dt * src_v * rho_slice
                rhow[i_cell, :, :] += dt * src_w * rho_slice
            elif axis == 1:
                if side == 'left':
                    j_cell = 1
                else:
                    j_cell = ny - 2
                rho_slice = rho[:, j_cell, :]
                u_slice   = rhou[:, j_cell, :] / (rho_slice + 1e-8)
                v_slice   = rhov[:, j_cell, :] / (rho_slice + 1e-8)
                w_slice   = rhow[:, j_cell, :] / (rho_slice + 1e-8)
                ut = torch.sqrt(u_slice**2 + w_slice**2)
                y = 0.5 * dx
                ke = 0.5 * rho_slice * (u_slice**2 + v_slice**2 + w_slice**2)
                p = (gamma - 1) * (rhoE[:, j_cell, :] - ke)
                T = p / (rho_slice + 1e-8)
                mu_lam = self.nu_phys * rho_slice * T.pow(1.5) * (1 + S) / (T + S)
                nu = mu_lam / (rho_slice + 1e-8)
                u_tau = torch.sqrt(nu * ut / (y + 1e-8) + 1e-8)
                for _ in range(5):
                    y_plus = y * u_tau / (nu + 1e-8)
                    u_plus = torch.where(y_plus <= 11.81, y_plus, self.cfg.wm_A * y_plus**self.cfg.wm_B)
                    u_tau_new = ut / (u_plus + 1e-8)
                    u_tau = 0.5 * (u_tau + u_tau_new)
                tau_w = rho_slice * u_tau**2
                ut_mag = ut + 1e-8
                tau_x = -tau_w * (u_slice / ut_mag)
                tau_z = -tau_w * (w_slice / ut_mag)
                src_u = tau_x / dx
                src_w = tau_z / dx
                rhou[:, j_cell, :] += dt * src_u * rho_slice
                rhow[:, j_cell, :] += dt * src_w * rho_slice
            else:  # axis == 2
                if side == 'left':
                    k_cell = 1
                else:
                    k_cell = nz - 2
                rho_slice = rho[:, :, k_cell]
                u_slice   = rhou[:, :, k_cell] / (rho_slice + 1e-8)
                v_slice   = rhov[:, :, k_cell] / (rho_slice + 1e-8)
                w_slice   = rhow[:, :, k_cell] / (rho_slice + 1e-8)
                ut = torch.sqrt(u_slice**2 + v_slice**2)
                y = 0.5 * dx
                ke = 0.5 * rho_slice * (u_slice**2 + v_slice**2 + w_slice**2)
                p = (gamma - 1) * (rhoE[:, :, k_cell] - ke)
                T = p / (rho_slice + 1e-8)
                mu_lam = self.nu_phys * rho_slice * T.pow(1.5) * (1 + S) / (T + S)
                nu = mu_lam / (rho_slice + 1e-8)
                u_tau = torch.sqrt(nu * ut / (y + 1e-8) + 1e-8)
                for _ in range(5):
                    y_plus = y * u_tau / (nu + 1e-8)
                    u_plus = torch.where(y_plus <= 11.81, y_plus, self.cfg.wm_A * y_plus**self.cfg.wm_B)
                    u_tau_new = ut / (u_plus + 1e-8)
                    u_tau = 0.5 * (u_tau + u_tau_new)
                tau_w = rho_slice * u_tau**2
                ut_mag = ut + 1e-8
                tau_x = -tau_w * (u_slice / ut_mag)
                tau_y = -tau_w * (v_slice / ut_mag)
                src_u = tau_x / dx
                src_v = tau_y / dx
                rhou[:, :, k_cell] += dt * src_u * rho_slice
                rhov[:, :, k_cell] += dt * src_v * rho_slice

    def _init_fields(self, case='taylor_green'):
        nx, ny = self.cfg.nx, self.cfg.ny
        # Use local_nz instead of cfg.nz
        nz = self.local_nz
        x = torch.linspace(0, self.cfg.Lx, nx, device=self.device, dtype=self.dtype)
        y = torch.linspace(0, self.cfg.Ly, ny, device=self.device, dtype=self.dtype)
        # z coordinate for this slab
        z_global = torch.linspace(0, self.cfg.Lz, self.cfg.nz, device=self.device, dtype=self.dtype)
        z = z_global[self.z_start:self.z_end] if self.distributed else z_global
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        if case == 'taylor_green':
            u0 = self.cfg.Mach
            u = u0 * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
            v = -u0 * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
            w = 0.0 * u0 * torch.cos(X) * torch.cos(Y) * torch.sin(Z)
            rho = torch.ones_like(u)
            T = 1.0 / self.gamma
            p = rho * T
        elif case == 'hypersonic_bnd':
            u0 = self.cfg.Mach * math.sqrt(self.gamma)
            u = u0 * (1 - torch.exp(-Y * 5))
            v = torch.zeros_like(u)
            w = torch.zeros_like(u)
            rho = torch.ones_like(u)
            T = 1.0 / self.gamma
            p = rho * T
        else:
            u = self.cfg.Mach * torch.ones_like(X)
            v = torch.zeros_like(X)
            w = torch.zeros_like(X)
            rho = torch.ones_like(X)
            T = 1.0 / self.gamma
            p = rho * T

        rhou = rho * u
        rhov = rho * v
        rhow = rho * w
        rhoE = p / (self.gamma - 1) + 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)

        self.rho = rho
        self.rhou = rhou
        self.rhov = rhov
        self.rhow = rhow
        self.rhoE = rhoE
        self.soc.reset()
        self.step_count = 0
        self.time = 0.0

    def _compute_rhs(self, rho, rhou, rhov, rhow, rhoE, dt):
        dx = self.dx
        gamma = self.gamma
        nx, ny, nz = rho.shape  # local_nz in distributed case

        u = rhou / (rho + 1e-8)
        v = rhov / (rho + 1e-8)
        w = rhow / (rho + 1e-8)
        ke = 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)
        if self.eos is not None and self.eos.use_real:
            e = (rhoE - ke) / (rho + 1e-8)
            p = self.eos.pressure(rho, e)
        else:
            p = (gamma - 1) * (rhoE - ke)
        T = p / (rho + 1e-8)
        if self.eos is not None and self.eos.use_real:
            c = self.eos.sound_speed(rho, e)
        else:
            c = torch.sqrt(gamma * p / (rho + 1e-8))

        if self.cfg.use_sutherland:
            S = 110.4 / 300.0
            mu_lam = self.nu_phys * rho * T.pow(1.5) * (1 + S) / (T + S)
        else:
            mu_lam = self.nu_phys * rho

        rho_p  = self._pad_field(rho)
        u_p    = self._pad_field(u)
        v_p    = self._pad_field(v)
        w_p    = self._pad_field(w)
        p_p    = self._pad_field(p)
        rhou_pad = self._pad_field(rhou)
        rhov_pad = self._pad_field(rhov)
        rhow_pad = self._pad_field(rhow)
        rhoE_pad = self._pad_field(rhoE)

        # Fill ghost cells: physical BCs + distributed halo exchange
        self._fill_ghost_cells(rho_p, rhou_pad, rhov_pad, rhow_pad, rhoE_pad)
        if self.distributed:
            # Exchange ghost layers for primitive variables too
            # Recompute u_p, v_p, w_p, p_p from padded conserved variables.
            u_p = rhou_pad / (rho_p + 1e-8)
            v_p = rhov_pad / (rho_p + 1e-8)
            w_p = rhow_pad / (rho_p + 1e-8)
            ke_p = 0.5 * rho_p * (u_p**2 + v_p**2 + w_p**2)
            if self.eos is not None and self.eos.use_real:
                e_p = (rhoE_pad - ke_p) / (rho_p + 1e-8)
                p_p = self.eos.pressure(rho_p, e_p)
            else:
                p_p = (gamma - 1) * (rhoE_pad - ke_p)
            # Now exchange halo for primitive fields as well
            self._exchange_halo_z(rho_p, u_p, v_p, w_p, p_p)
        else:
            # In single GPU, ghost cells are already correct from _fill_ghost_cells.
            u_p = rhou_pad / (rho_p + 1e-8)
            v_p = rhov_pad / (rho_p + 1e-8)
            w_p = rhow_pad / (rho_p + 1e-8)
            ke_p = 0.5 * rho_p * (u_p**2 + v_p**2 + w_p**2)
            if self.eos is not None and self.eos.use_real:
                e_p = (rhoE_pad - ke_p) / (rho_p + 1e-8)
                p_p = self.eos.pressure(rho_p, e_p)
            else:
                p_p = (gamma - 1) * (rhoE_pad - ke_p)

        def ddx(f): return (f[3:nx+3, 2:ny+2, 2:nz+2] - f[1:nx+1, 2:ny+2, 2:nz+2]) / (2*dx)
        def ddy(f): return (f[2:nx+2, 3:ny+3, 2:nz+2] - f[2:nx+2, 1:ny+1, 2:nz+2]) / (2*dx)
        def ddz(f): return (f[2:nx+2, 2:ny+2, 3:nz+3] - f[2:nx+2, 2:ny+2, 1:nz+1]) / (2*dx)

        S11 = ddx(u_p); S22 = ddy(v_p); S33 = ddz(w_p)
        S12 = 0.5 * (ddy(u_p) + ddx(v_p))
        S13 = 0.5 * (ddz(u_p) + ddx(w_p))
        S23 = 0.5 * (ddz(v_p) + ddy(w_p))
        strain_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)))
        dilatation = S11 + S22 + S33

        nu_t = self.soc.nu_t(rho, strain_mag, dilatation, dx, dt, c)
        mu_eff = mu_lam + rho * nu_t

        if self.cfg.shock_capturing:
            vorticity = torch.sqrt(
                (ddz(v_p) - ddy(w_p))**2 +
                (ddx(w_p) - ddz(u_p))**2 +
                (ddy(u_p) - ddx(v_p))**2
            )
            shock_sensor = torch.clamp(-dilatation, min=0) / (
                strain_mag + 1e-8) * torch.clamp(1 - vorticity / (strain_mag + 1e-8), min=0)
            mu_shock = rho * dx * c * 0.1 * shock_sensor
            mu_eff = mu_eff + mu_shock

        conv_rho  = torch.zeros_like(rho)
        conv_rhou = torch.zeros_like(rho)
        conv_rhov = torch.zeros_like(rho)
        conv_rhow = torch.zeros_like(rho)
        conv_rhoE = torch.zeros_like(rho)

        for axis in range(3):
            mass_flux, f_u, f_v, f_w, f_E = self.flux_solver.compute_face_flux(
                rho_p, u_p, v_p, w_p, p_p, axis)
            if axis == 0:
                conv_rho  += (mass_flux[1:] - mass_flux[:-1]) / dx
                conv_rhou += (f_u[1:] - f_u[:-1]) / dx
                conv_rhov += (f_v[1:] - f_v[:-1]) / dx
                conv_rhow += (f_w[1:] - f_w[:-1]) / dx
                conv_rhoE += (f_E[1:] - f_E[:-1]) / dx
            elif axis == 1:
                conv_rho  += (mass_flux[:, 1:] - mass_flux[:, :-1]) / dx
                conv_rhou += (f_u[:, 1:] - f_u[:, :-1]) / dx
                conv_rhov += (f_v[:, 1:] - f_v[:, :-1]) / dx
                conv_rhow += (f_w[:, 1:] - f_w[:, :-1]) / dx
                conv_rhoE += (f_E[:, 1:] - f_E[:, :-1]) / dx
            else:
                conv_rho  += (mass_flux[:, :, 1:] - mass_flux[:, :, :-1]) / dx
                conv_rhou += (f_u[:, :, 1:] - f_u[:, :, :-1]) / dx
                conv_rhov += (f_v[:, :, 1:] - f_v[:, :, :-1]) / dx
                conv_rhow += (f_w[:, :, 1:] - f_w[:, :, :-1]) / dx
                conv_rhoE += (f_E[:, :, 1:] - f_E[:, :, :-1]) / dx

        div_v = S11 + S22 + S33
        tau_xx = mu_eff * (2*S11 - (2/3)*div_v)
        tau_yy = mu_eff * (2*S22 - (2/3)*div_v)
        tau_zz = mu_eff * (2*S33 - (2/3)*div_v)
        tau_xy = mu_eff * (ddy(u_p) + ddx(v_p))
        tau_xz = mu_eff * (ddz(u_p) + ddx(w_p))
        tau_yz = mu_eff * (ddz(v_p) + ddy(w_p))

        if self.ito_gen is not None:
            s11, s22, s33, s12, s13, s23 = self.ito_gen.generate(tau_xx.shape, self.device, dt)
            tau_xx += s11; tau_yy += s22; tau_zz += s33
            tau_xy += s12; tau_xz += s13; tau_yz += s23

        pad_tau_xx = self._pad_field(tau_xx); pad_tau_xy = self._pad_field(tau_xy)
        pad_tau_xz = self._pad_field(tau_xz); pad_tau_yy = self._pad_field(tau_yy)
        pad_tau_yz = self._pad_field(tau_yz); pad_tau_zz = self._pad_field(tau_zz)

        # For viscous terms, we also need ghost cells for tau. We'll use the same halo exchange.
        # Since tau is based on velocity gradients, we can exchange the needed ghost layers.
        # For simplicity, we exchange tau components as well.
        if self.distributed:
            self._exchange_halo_z(pad_tau_xx, pad_tau_xy, pad_tau_xz,
                                  pad_tau_yy, pad_tau_yz, pad_tau_zz)

        visc_rhou = ddx(pad_tau_xx) + ddy(pad_tau_xy) + ddz(pad_tau_xz)
        visc_rhov = ddx(pad_tau_xy) + ddy(pad_tau_yy) + ddz(pad_tau_yz)
        visc_rhow = ddx(pad_tau_xz) + ddy(pad_tau_yz) + ddz(pad_tau_zz)

        k_eff = mu_eff * gamma / (gamma - 1) / self.Pr
        T_p = self._pad_field(T)
        if self.distributed:
            self._exchange_halo_z(T_p)
        qx = k_eff * ddx(T_p); qy = k_eff * ddy(T_p); qz = k_eff * ddz(T_p)
        qx_p = self._pad_field(qx); qy_p = self._pad_field(qy); qz_p = self._pad_field(qz)
        if self.distributed:
            self._exchange_halo_z(qx_p, qy_p, qz_p)
        heat_div = ddx(qx_p) + ddy(qy_p) + ddz(qz_p)

        tau_dot_u_x = tau_xx*u + tau_xy*v + tau_xz*w
        tau_dot_u_y = tau_xy*u + tau_yy*v + tau_yz*w
        tau_dot_u_z = tau_xz*u + tau_yz*v + tau_zz*w
        pad_tau_dot_x = self._pad_field(tau_dot_u_x)
        pad_tau_dot_y = self._pad_field(tau_dot_u_y)
        pad_tau_dot_z = self._pad_field(tau_dot_u_z)
        if self.distributed:
            self._exchange_halo_z(pad_tau_dot_x, pad_tau_dot_y, pad_tau_dot_z)
        work_div = ddx(pad_tau_dot_x) + ddy(pad_tau_dot_y) + ddz(pad_tau_dot_z)

        visc_rhoE = work_div + heat_div

        rhs_rho  = -conv_rho
        rhs_rhou = -conv_rhou + visc_rhou
        rhs_rhov = -conv_rhov + visc_rhov
        rhs_rhow = -conv_rhow + visc_rhow
        rhs_rhoE = -conv_rhoE + visc_rhoE

        return rhs_rho, rhs_rhou, rhs_rhov, rhs_rhow, rhs_rhoE

    def step(self, dt=None):
        rho, rhou, rhov, rhow, rhoE = self.rho, self.rhou, self.rhov, self.rhow, self.rhoE
        gamma = self.gamma
        dx = self.dx

        if dt is None:
            u = rhou / (rho + 1e-8); v = rhov / (rho + 1e-8); w = rhow / (rho + 1e-8)
            p = (gamma - 1) * (rhoE - 0.5*rho*(u**2+v**2+w**2))
            c = torch.sqrt(gamma * p / (rho + 1e-8))
            speed = torch.sqrt(u**2+v**2+w**2) + c
            dt = self.cfg.cfl * dx / (speed.max() + 1e-8)
            if self.distributed:
                # Reduce maximum across processes to get global minimum dt
                dt_local = dt.clone()
                dist.all_reduce(dt, op=dist.ReduceOp.MIN)

        # Stage 1
        k1 = self._compute_rhs(rho, rhou, rhov, rhow, rhoE, dt)
        rho1   = rho   + dt * k1[0]; rhou1  = rhou  + dt * k1[1]
        rhov1  = rhov  + dt * k1[2]; rhow1  = rhow  + dt * k1[3]; rhoE1  = rhoE  + dt * k1[4]
        self._apply_bc_to_boundary_cells(rho1, rhou1, rhov1, rhow1, rhoE1)
        if self.ib is not None:
            rhou1, rhov1, rhow1, rhoE1 = self.ib.apply_forcing(rho1, rhou1, rhov1, rhow1, rhoE1, dt)
        self._apply_wall_model(rho1, rhou1, rhov1, rhow1, rhoE1, dt)

        # Stage 2
        k2 = self._compute_rhs(rho1, rhou1, rhov1, rhow1, rhoE1, dt)
        rho2   = 0.75 * rho   + 0.25 * (rho1   + dt * k2[0])
        rhou2  = 0.75 * rhou  + 0.25 * (rhou1  + dt * k2[1])
        rhov2  = 0.75 * rhov  + 0.25 * (rhov1  + dt * k2[2])
        rhow2  = 0.75 * rhow  + 0.25 * (rhow1  + dt * k2[3])
        rhoE2  = 0.75 * rhoE  + 0.25 * (rhoE1  + dt * k2[4])
        self._apply_bc_to_boundary_cells(rho2, rhou2, rhov2, rhow2, rhoE2)
        if self.ib is not None:
            rhou2, rhov2, rhow2, rhoE2 = self.ib.apply_forcing(rho2, rhou2, rhov2, rhow2, rhoE2, dt)
        self._apply_wall_model(rho2, rhou2, rhov2, rhow2, rhoE2, dt)

        # Stage 3
        k3 = self._compute_rhs(rho2, rhou2, rhov2, rhow2, rhoE2, dt)
        rho_new   = (1/3) * rho   + (2/3) * (rho2   + dt * k3[0])
        rhou_new  = (1/3) * rhou  + (2/3) * (rhou2  + dt * k3[1])
        rhov_new  = (1/3) * rhov  + (2/3) * (rhov2  + dt * k3[2])
        rhow_new  = (1/3) * rhow  + (2/3) * (rhow2  + dt * k3[3])
        rhoE_new  = (1/3) * rhoE  + (2/3) * (rhoE2  + dt * k3[4])
        self._apply_bc_to_boundary_cells(rho_new, rhou_new, rhov_new, rhow_new, rhoE_new)
        if self.ib is not None:
            rhou_new, rhov_new, rhow_new, rhoE_new = self.ib.apply_forcing(
                rho_new, rhou_new, rhov_new, rhow_new, rhoE_new, dt)
        self._apply_wall_model(rho_new, rhou_new, rhov_new, rhow_new, rhoE_new, dt)

        rho_new = torch.clamp(rho_new, min=1e-6)

        ke_new  = 0.5 * (rhou_new**2 + rhov_new**2 + rhow_new**2) / (rho_new + 1e-8)
        if self.eos is not None and self.eos.use_real:
            e_new = (rhoE_new - ke_new) / (rho_new + 1e-8)
            p_new = self.eos.pressure(rho_new, e_new)
        else:
            p_new = (gamma - 1) * (rhoE_new - ke_new)

        p_min = p_new.min().item()
        if p_min < 1e-10:
            logger.warning(f"Low pressure detected: min(p) = {p_min:.3e} at step {self.step_count}. Clamping applied.")
        p_new = torch.clamp(p_new, min=1e-8)
        if self.eos is None or not self.eos.use_real:
            rhoE_new = p_new / (gamma - 1) + ke_new

        self.rho  = rho_new
        self.rhou = rhou_new
        self.rhov = rhov_new
        self.rhow = rhow_new
        self.rhoE = rhoE_new
        self.step_count += 1
        self.time += dt

        if self.rg is not None and self.step_count % self.cfg.rg_interval == 0:
            self.rho  = self.rg.forward(self.rho)
            self.rhou = self.rg.forward(self.rhou)
            self.rhov = self.rg.forward(self.rhov)
            self.rhow = self.rg.forward(self.rhow)
            self.rhoE = self.rg.forward(self.rhoE)

        return p_new.mean()

    def run(self, steps=None):
        if steps is None:
            steps = self.cfg.steps
        if self.rho is None:
            self._init_fields()
        for t in range(1, steps + 1):
            p_avg = self.step()
            if t % 50 == 0 or t == 1:
                u = self.rhou / (self.rho + 1e-8)
                v = self.rhov / (self.rho + 1e-8)
                w = self.rhow / (self.rho + 1e-8)
                ke_local = 0.5 * torch.mean(self.rho * (u**2 + v**2 + w**2)).item()
                # Reduce KE across processes
                if self.distributed:
                    ke_tensor = torch.tensor(ke_local, device=self.device)
                    dist.all_reduce(ke_tensor, op=dist.ReduceOp.SUM)
                    ke_global = ke_tensor.item() / self.world_size
                else:
                    ke_global = ke_local
                self.energy_hist.append(ke_global)
                if self.rank == 0:
                    logger.info(f"Step {t:04d}, time={self.time:.6f}, KE={ke_global:.6f}, ⟨p⟩={p_avg.item():.4f}")
        if self.distributed:
            dist.barrier()
        return self.rho, self.rhou, self.rhov, self.rhow, self.rhoE, self.energy_hist, self.div_hist

    def taylor_green_test(self, steps=200):
        logger.info("=== Taylor–Green vortex test ===")
        self._init_fields('taylor_green')
        self.run(steps=steps)
        u = self.rhou / (self.rho + 1e-8)
        v = self.rhov / (self.rho + 1e-8)
        w = self.rhow / (self.rho + 1e-8)
        ke_local = 0.5 * torch.mean(self.rho * (u**2 + v**2 + w**2)).item()
        if self.distributed:
            ke_tensor = torch.tensor(ke_local, device=self.device)
            dist.all_reduce(ke_tensor, op=dist.ReduceOp.SUM)
            ke_final = ke_tensor.item() / self.world_size
        else:
            ke_final = ke_local
        if self.rank == 0:
            logger.info(f"Final kinetic energy = {ke_final:.6f}")
        return ke_final, self.energy_hist

    def save_checkpoint(self, filepath):
        # Save local state; for distributed restart all ranks must load their part.
        state = {
            'cfg': self.cfg,
            'step': self.step_count,
            'time': self.time,
            'rho': self.rho.cpu(),
            'rhou': self.rhou.cpu(),
            'rhov': self.rhov.cpu(),
            'rhow': self.rhow.cpu(),
            'rhoE': self.rhoE.cpu(),
            'energy_hist': self.energy_hist,
            'div_hist': self.div_hist,
        }
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath} (rank {self.rank})")

    def load_checkpoint(self, filepath):
        state = torch.load(filepath, map_location='cpu')
        # Ensure the loaded configuration matches the current decomposition
        if self.distributed:
            # The saved state contains full domain? We saved local slab, so fine.
            pass
        self.cfg = state['cfg']
        self.step_count = state['step']
        self.time = state['time']
        self.rho  = state['rho'].to(self.device)
        self.rhou = state['rhou'].to(self.device)
        self.rhov = state['rhov'].to(self.device)
        self.rhow = state['rhow'].to(self.device)
        self.rhoE = state['rhoE'].to(self.device)
        self.energy_hist = state['energy_hist']
        self.div_hist = state['div_hist']
        logger.info(f"Checkpoint loaded from {filepath}, step={self.step_count}, time={self.time}")

    def kolmogorov_slope(self):
        # Gather full 3D field to rank 0 for spectral analysis
        if self.distributed:
            # Gather all z-slabs along dim 2
            local_u = self.rhou / (self.rho + 1e-8)
            local_v = self.rhov / (self.rho + 1e-8)
            local_w = self.rhow / (self.rho + 1e-8)
            # Gather tensors from all ranks
            gather_list_u = [torch.zeros_like(local_u) for _ in range(self.world_size)] if self.rank == 0 else None
            gather_list_v = [torch.zeros_like(local_v) for _ in range(self.world_size)] if self.rank == 0 else None
            gather_list_w = [torch.zeros_like(local_w) for _ in range(self.world_size)] if self.rank == 0 else None
            dist.gather(local_u, gather_list_u, dst=0)
            dist.gather(local_v, gather_list_v, dst=0)
            dist.gather(local_w, gather_list_w, dst=0)
            if self.rank != 0:
                return None
            u_full = torch.cat(gather_list_u, dim=2)
            v_full = torch.cat(gather_list_v, dim=2)
            w_full = torch.cat(gather_list_w, dim=2)
        else:
            u_full = self.rhou / (self.rho + 1e-8)
            v_full = self.rhov / (self.rho + 1e-8)
            w_full = self.rhow / (self.rho + 1e-8)

        u_hat = fftn(u_full.cpu().numpy())
        v_hat = fftn(v_full.cpu().numpy())
        w_hat = fftn(w_full.cpu().numpy())
        kx = fftfreq(u_full.shape[0], d=self.cfg.Lx / u_full.shape[0])
        ky = fftfreq(u_full.shape[1], d=self.cfg.Ly / u_full.shape[1])
        kz = fftfreq(u_full.shape[2], d=self.cfg.Lz / u_full.shape[2])
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        bins = np.logspace(np.log10(1), np.log10(K.max()), 20)
        E_spec = []
        for i in range(len(bins)-1):
            mask = (K >= bins[i]) & (K < bins[i+1])
            if np.any(mask):
                E = 0.5 * (np.abs(u_hat[mask])**2 + np.abs(v_hat[mask])**2 + np.abs(w_hat[mask])**2).mean()
                E_spec.append(E)
            else:
                E_spec.append(0.0)
        valid = np.array(E_spec) > 0
        if sum(valid) < 3:
            return None
        kc = 0.5 * (bins[:-1] + bins[1:])
        slope, _, _, _, _ = linregress(np.log10(kc[valid]), np.log10(np.array(E_spec)[valid]))
        return slope

    def grid_convergence_test(self, grid_sizes=[32,64,96], ref_steps=50):
        # Not adapted for distributed; warn if called
        if self.distributed:
            logger.warning("Grid convergence test not fully supported in distributed mode. Run on single GPU.")
            return None
        errors = []
        u_ref = None
        original_nx = self.cfg.nx
        for N in grid_sizes:
            self.cfg.nx = N; self.cfg.ny = N; self.cfg.nz = N
            self.cfg.dx = self.cfg.Lx / N
            self.dx = self.cfg.dx
            self._init_fields('taylor_green')
            for _ in range(ref_steps):
                self.step()
            u = self.rhou / (self.rho + 1e-8)
            if N == max(grid_sizes):
                u_ref = u.clone()
            else:
                u_ref_down = F.interpolate(u_ref.unsqueeze(0).unsqueeze(0),
                                           size=(N, N, N), mode='trilinear',
                                           align_corners=False).squeeze()
                err = torch.norm(u - u_ref_down).item() / np.sqrt(N**3)
                errors.append(err)
        self.cfg.nx = original_nx
        self.cfg.dx = self.cfg.Lx / original_nx
        self.dx = self.cfg.dx
        if len(errors) >= 2:
            slope, _, _, _, _ = linregress(np.log(grid_sizes[:-1]), np.log(errors))
            return -slope
        return None


# =============================================================================
# 7. Advanced Signal Denoising
# =============================================================================
class SignalDenoiser:
    def __init__(self, method='ssc', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def denoise(self, data: torch.Tensor) -> torch.Tensor:
        if self.method == 'ssc':
            ssc = SemanticStateContraction(**self.kwargs)
            return ssc(data.mean())
        elif self.method == 'wiener':
            data_np = data.cpu().numpy()
            from scipy.signal import wiener
            denoised = wiener(data_np, **self.kwargs)
            return torch.tensor(denoised, device=data.device)
        elif self.method == 'wavelet' and HAS_PYWT:
            data_np = data.cpu().numpy()
            coeffs = pywt.wavedec(data_np, self.kwargs.get('wavelet', 'db4'), level=self.kwargs.get('level', 4))
            threshold = self.kwargs.get('threshold', 0.1)
            coeffs[1:] = [pywt.threshold(c, threshold) for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, self.kwargs.get('wavelet', 'db4'))
            return torch.tensor(denoised[:data.shape[0]], device=data.device)
        else:
            return data


# =============================================================================
# 8. SOC Kernel Trainer
# =============================================================================
class SOCTrainer:
    @staticmethod
    def objective(params, solver, target_energy):
        solver.soc.kernel.log_Cs.data     = torch.tensor(math.log(params[0]), device=solver.device)
        solver.soc.kernel.log_lambda.data = torch.tensor(math.log(params[1]), device=solver.device)
        solver.soc.kernel.log_alpha.data  = torch.tensor(math.log(params[2]), device=solver.device)
        solver.soc.kernel.log_theta.data  = torch.tensor(math.log(params[3]), device=solver.device)
        solver.soc.kernel.log_tau.data    = torch.tensor(math.log(params[4]), device=solver.device)
        solver._init_fields('taylor_green')
        for _ in range(50):
            solver.step()
        u = solver.rhou / (solver.rho + 1e-8)
        v = solver.rhov / (solver.rho + 1e-8)
        w = solver.rhow / (solver.rho + 1e-8)
        ke = 0.5 * torch.mean(solver.rho * (u**2 + v**2 + w**2)).item()
        return abs(ke - target_energy)

    @classmethod
    def train(cls, solver, target_energy, method='de', max_iter=50):
        if method == 'optuna' and HAS_OPTUNA:
            def obj(trial):
                solver.soc.kernel.log_Cs.data = torch.tensor(
                    trial.suggest_float('log_Cs', math.log(0.05), math.log(0.3)),
                    device=solver.device)
                solver.soc.kernel.log_lambda.data = torch.tensor(
                    trial.suggest_float('log_lambda', math.log(5), math.log(30)),
                    device=solver.device)
                solver.soc.kernel.log_alpha.data = torch.tensor(
                    trial.suggest_float('log_alpha', math.log(0.1), math.log(2.0)),
                    device=solver.device)
                solver.soc.kernel.log_theta.data = torch.tensor(
                    trial.suggest_float('log_theta', math.log(0.5), math.log(5.0)),
                    device=solver.device)
                solver.soc.kernel.log_tau.data = torch.tensor(
                    trial.suggest_float('log_tau', math.log(1.0), math.log(50.0)),
                    device=solver.device)
                solver._init_fields('taylor_green')
                for _ in range(50):
                    solver.step()
                u = solver.rhou / (solver.rho + 1e-8)
                v = solver.rhov / (solver.rho + 1e-8)
                w = solver.rhow / (solver.rho + 1e-8)
                ke = 0.5 * torch.mean(solver.rho * (u**2+v**2+w**2)).item()
                return abs(ke - target_energy)
            study = optuna.create_study(direction='minimize')
            study.optimize(obj, n_trials=max_iter)
            return study.best_params
        else:
            bounds = [(0.05, 0.3), (5, 30), (0.1, 2.0), (0.5, 5.0), (1, 50)]
            result = differential_evolution(
                lambda p: cls.objective(p, solver, target_energy),
                bounds, maxiter=max_iter, popsize=10, tol=1e-6, disp=False)
            return {'Cs': result.x[0], 'lambda': result.x[1],
                    'alpha': result.x[2], 'theta': result.x[3], 'tau': result.x[4]}


# =============================================================================
# 9. Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SUPER DNS ONE – Industrial Compressible 3D DNS")
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--Lx', type=float, default=2*math.pi)
    parser.add_argument('--Ly', type=float, default=2*math.pi)
    parser.add_argument('--Lz', type=float, default=2*math.pi)
    parser.add_argument('--Re', type=float, default=1e4)
    parser.add_argument('--Pr', type=float, default=0.71)
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--Mach', type=float, default=0.1)
    parser.add_argument('--cfl', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--soc-temp', type=float, default=300.0)
    parser.add_argument('--max-nu-t', type=float, default=0.05)
    parser.add_argument('--rg', action='store_true', help='Enable RG refinement')
    parser.add_argument('--rg-keep', type=float, default=0.5)
    parser.add_argument('--ito', type=float, default=0.0)
    parser.add_argument('--muscl', action='store_true', default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--flux', default='ausm', choices=['ausm', 'hllc'])
    parser.add_argument('--shock-capturing', action='store_true')
    parser.add_argument('--compress-corr', action='store_true', default=True)
    parser.add_argument('--ssc-epsilon', type=float, default=0.0028)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--bc-x-min', default='periodic')
    parser.add_argument('--bc-x-max', default='periodic')
    parser.add_argument('--bc-y-min', default='periodic')
    parser.add_argument('--bc-y-max', default='periodic')
    parser.add_argument('--bc-z-min', default='periodic')
    parser.add_argument('--bc-z-max', default='periodic')
    parser.add_argument('--inflow-rho', type=float, default=1.0)
    parser.add_argument('--inflow-u', type=float, default=0.0)
    parser.add_argument('--inflow-v', type=float, default=0.0)
    parser.add_argument('--inflow-w', type=float, default=0.0)
    parser.add_argument('--inflow-p', type=float, default=1.0)
    parser.add_argument('--outflow-p', type=float, default=1.0)
    parser.add_argument('--wall-temp', type=float, default=300.0)
    parser.add_argument('--farfield-rho', type=float, default=1.0)
    parser.add_argument('--farfield-u', type=float, default=0.0)
    parser.add_argument('--farfield-v', type=float, default=0.0)
    parser.add_argument('--farfield-w', type=float, default=0.0)
    parser.add_argument('--farfield-p', type=float, default=1.0)
    parser.add_argument('--moving-wall-u', type=float, default=0.0)
    parser.add_argument('--moving-wall-v', type=float, default=0.0)
    parser.add_argument('--moving-wall-w', type=float, default=0.0)
    parser.add_argument('--wall-model', action='store_true')
    parser.add_argument('--wm-A', type=float, default=8.3)
    parser.add_argument('--wm-B', type=float, default=1.0/7.0)
    parser.add_argument('--eos-model', default='ideal', choices=['ideal', 'real'])
    parser.add_argument('--fluid', default='Air')
    parser.add_argument('--ib-mask', help='Path to numpy mask file for immersed boundary')
    parser.add_argument('--ib-eta', type=float, default=1e4)
    parser.add_argument('--ib-T-target', type=float, default=None)
    parser.add_argument('--ib-eta-T', type=float, default=1e4)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--denoise-method', default='ssc', choices=['ssc', 'wiener', 'wavelet'])
    parser.add_argument('--wavelet', default='db4')
    parser.add_argument('--denoise-level', type=int, default=4)
    parser.add_argument('--denoise-threshold', type=float, default=0.1)
    parser.add_argument('--case', default='taylor_green', choices=['taylor_green', 'hypersonic_bnd', 'uniform'])
    parser.add_argument('--save-checkpoint', default=None)
    parser.add_argument('--load-checkpoint', default=None)
    parser.add_argument('--train-soc', action='store_true')
    parser.add_argument('--target-energy', type=float, default=0.1)
    parser.add_argument('--grid-convergence', action='store_true')
    parser.add_argument('--distributed', action='store_true', help='Enable multi‑GPU distributed parallelism')
    parser.add_argument('--local_rank', type=int, default=None, help='Local rank for distributed launch')

    args = parser.parse_args()

    # Initialize distributed environment if requested
    if args.distributed:
        # Use environment variables set by torchrun / mpirun
        dist.init_process_group(backend='nccl')
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)

    cfg = CFDConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        Re=args.Re, Pr=args.Pr, gamma=args.gamma,
        Mach=args.Mach, cfl=args.cfl, steps=args.steps,
        soc_base_temp=args.soc_temp, max_nu_t=args.max_nu_t,
        use_rg=args.rg, rg_keep_frac=args.rg_keep,
        ito_noise=args.ito, muscl=args.muscl,
        device=args.device, flux_scheme=args.flux,
        shock_capturing=args.shock_capturing,
        compressibility_correction=args.compress_corr,
        ssc_epsilon=args.ssc_epsilon, dtype=torch.float32,
        bc_x_min=args.bc_x_min, bc_x_max=args.bc_x_max,
        bc_y_min=args.bc_y_min, bc_y_max=args.bc_y_max,
        bc_z_min=args.bc_z_min, bc_z_max=args.bc_z_max,
        inflow_rho=args.inflow_rho, inflow_u=args.inflow_u,
        inflow_v=args.inflow_v, inflow_w=args.inflow_w, inflow_p=args.inflow_p,
        outflow_p=args.outflow_p, wall_temp=args.wall_temp,
        farfield_rho=args.farfield_rho, farfield_u=args.farfield_u,
        farfield_v=args.farfield_v, farfield_w=args.farfield_w, farfield_p=args.farfield_p,
        moving_wall_u=args.moving_wall_u, moving_wall_v=args.moving_wall_v,
        moving_wall_w=args.moving_wall_w,
        use_wall_model=args.wall_model, wm_A=args.wm_A, wm_B=args.wm_B,
        eos_model=args.eos_model, fluid_name=args.fluid,
        ib_mask_file=args.ib_mask, ib_eta=args.ib_eta,
        ib_T_target=args.ib_T_target, ib_eta_T=args.ib_eta_T,
        distributed=args.distributed
    )

    solver = CompressibleSolver(cfg)

    if args.load_checkpoint:
        solver.load_checkpoint(args.load_checkpoint)

    if args.train_soc:
        SOCTrainer.train(solver, args.target_energy)
    elif args.grid_convergence:
        rate = solver.grid_convergence_test()
        if rate is not None and solver.rank == 0:
            logger.info(f"Estimated convergence rate: {rate}")
    elif args.denoise:
        denoiser = SignalDenoiser(method=args.denoise_method,
                                  wavelet=args.wavelet,
                                  level=args.denoise_level,
                                  threshold=args.denoise_threshold)
        solver._init_fields(args.case)
        solver.run()
        u = solver.rhou / (solver.rho + 1e-8)
        denoised_u = denoiser.denoise(u)
        if solver.rank == 0:
            logger.info(f"Denoised velocity range: {denoised_u.min():.3f} - {denoised_u.max():.3f}")
    else:
        if args.case == 'taylor_green':
            solver.taylor_green_test(steps=args.steps)
        else:
            solver._init_fields(args.case)
            solver.run()

    if args.save_checkpoint:
        solver.save_checkpoint(args.save_checkpoint)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
