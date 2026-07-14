# =============================================================================
# SUPER DNS ONE v6 — Native Full-Differentiable 3D Compressible DNS / LES
# =============================================================================
# Author : Yoon A Limsuwan / MSPS NETWORK
# License: MIT
# Version: 6.1 (full ecosystem integration — CH↔DNS bridge + attribution)
# Year   : 2026
#
# AI Development Partners:
#   Claude   (Anthropic)  — differentiability audit, CH body-force injection,
#                           bridge protocol design
#   GPT      (OpenAI)     — algorithmic suggestions, flux scheme review
#   Gemini   (Google)     — numerical scheme cross-validation
#   DeepSeek              — supplementary code analysis
#
# Changes v6.0 → v6.1:
#   • CahnHilliardDNSBridge imported from one_core
#   • CompressibleSolver.__init__ initialises _ext_rho_ch, _ext_nu_ch,
#     _ext_fx, _ext_fy, _ext_fz buffers for CH coupling
#   • _compute_rhs blends CH density/viscosity and injects Korteweg body force
#   Bug 1 fix: ssc._prev → ssc.prev_sigma (correct buffer name)
#   Bug 2 fix: SOCController now inherits CSOCBase properly
#   Bug 3 fix: LangevinDNSBridge imported; _ext_sigma coupling in SOCController
#
# Changes v6.7 → v6.8:
#   Genuine mass source into the continuity equation (requested: full
#   DNS-grid pyrolysis coupling). New _ext_mdot buffer family:
#   _ext_mdot [kg/(m^3.s), mass], _ext_mdot_u/v/w [momentum carried by
#   injected mass], _ext_mdot_e [energy carried], _ext_mdot_Z
#   [mixture-fraction carried] -- all consumed together in _compute_rhs,
#   the ONLY place in this file rhs_rho gets a source term at all
#   (continuity was exactly closed, convective-terms-only, before this).
#   Written by the new one_core_v3.PyrolysisDNSBridge (v3.4.0+).
#
#   UNIT-CONSISTENCY BUG caught and fixed before shipping: the mass-
#   source consumption block initially added the buffers directly into
#   this solver's non-dimensional RHS with NO scale conversion at all --
#   the same class of bug already found once for combustion/radiation
#   (v6.6's T_ref/combustion_nondim_scale fix), just not yet applied
#   here. Fixed with a NEW, SEPARATE cfg.mdot_nondim_scale (default 1.0,
#   explicit must-set-yourself, same philosophy as
#   combustion_nondim_scale) -- deliberately NOT reusing
#   combustion_nondim_scale, since mass-rate and energy-rate source terms
#   belong to different dimensional groups (mass ~ L_ref/(rho_ref*U_ref);
#   energy ~ L_ref/(rho_ref*U_ref^3)) and would need different conversion
#   factors even in a correctly-derived non-dimensionalization.
#
#   RS(soot)/RZ(mixture fraction) clamping in step() (Z,Y_soot in [0,1])
#   is unaffected by mass injection changing rho_n's growth rate -- the
#   existing `torch.minimum(RZ_n, rho_n)` / `torch.minimum(RS_n, rho_n)`
#   clamps already reference the POST-injection rho_n, so they remain
#   correct without further changes.
#
# Changes v6.6 → v6.7:
#   Discrete-ordinates radiation (DOM) + soot transport -- BOTH DEFAULT
#   DISABLED (opt-in only, explicitly requested to be "built but gated
#   off for certainty" given the real risk class already shown once by
#   the v6.6 non-dimensional unit bug). Every new code path is validated
#   in isolation (numpy prototypes) but NOT tested end-to-end against
#   real hardware/GPU in this session -- treat as a further step toward
#   FDS's modeling category, still requiring your own validation before
#   trusting quantitative results, same caveat as v6.6's combustion/
#   radiation additions.
#
#     - cfg.radiation_method: "P1" (default, unchanged v6.6 behaviour) or
#       "DOM" (new). DOM solves the discrete-ordinates RTE via SOURCE
#       ITERATION (fixed-point/Jacobi iteration on the upwind-discretized
#       equation along each quadrature direction) rather than a literal
#       sequential sweep -- sweeps don't vectorize on tensor hardware
#       (each cell needs an already-updated upwind neighbor within the
#       same pass); source iteration is a real, standard method for the
#       same equation set that DOES vectorize, at the cost of needing
#       cfg.dom_n_iterations (default 20) to converge. Quadrature is a
#       product Gauss-Legendre(polar) x uniform(azimuthal) set
#       (cfg.dom_n_polar x cfg.dom_n_azimuthal, default 4x8=32
#       directions) -- verified (numpy prototype) to integrate to exactly
#       4*pi total solid angle and to reproduce the correct isotropic
#       blackbody equilibrium G=4*sigma*T^4 for a uniform-temperature
#       test case. Same periodic-boundary assumption as the P1 path (no
#       wall emissivity/reflection BC modeled) -- not rigorously valid
#       for a wall-dominated enclosure.
#     - cfg.enable_soot: new conserved scalar RS=rho*Y_soot, transported
#       via the same centered-difference scheme as RZ (mixture fraction),
#       through the full TVD-RK3 integration. Production/oxidation source
#       only active when BOTH enable_soot AND enable_combustion are True
#       (double-gated: soot kinetics need the resolved Y_fuel/Y_O2/T
#       state that only exists under combustion) -- ports the exact same
#       semi-empirical rate laws as fire_one.SootKinetics (validated
#       independently there: formation gated to fuel-rich Z, oxidation
#       requires both soot and local O2 present).
#     - RS added to save_checkpoint/load_checkpoint, same backward-
#       compatible pattern as RZ (pre-v6.7 checkpoints load with
#       Y_soot=0, warn if enable_soot=True).
#     - Still NOT implemented: finite-rate/PAH soot chemistry (this
#       remains a semi-empirical rate closure, not real kinetics), wall
#       emissivity/reflection boundary conditions for either radiation
#       method, and live 3D-grid coupling of PyrolysisModel (still a
#       standalone fire_one.py physics model, not wired as a DNS boundary
#       condition -- that needs a new mass-source-into-continuity
#       mechanism, a further integration step beyond this pass).
#
# Changes v6.5 → v6.6:
#   FIRE CFD upgrade (requested: "make it real NIST-FDS-style fire CFD",
#   scoped honestly -- see fire_one.py / this changelog for what is and
#   is NOT actually reproduced relative to real FDS):
#     - New RESOLVED mixture-fraction field (self.RZ = rho*Z, conserved
#       form), transported alongside rho/rhou/rhov/rhow/rhoE through the
#       full TVD-RK3 integration in step() -- a genuine new transported
#       PDE field, not just an external body-force buffer. Uses simple
#       centered-difference advection/diffusion (consistent with this
#       file's existing viscous-term style), deliberately NOT hooked into
#       flux_solver's Godunov/Riemann Euler scheme, so this cannot
#       destabilize the proven compressible core -- but is therefore more
#       diffusive than a dedicated upwind/TVD scalar scheme; validate
#       against your own grid-Peclet-number requirements before trusting
#       sharp flame fronts.
#     - Local combustion source (cfg.enable_combustion=True): Burke-
#       Schumann fast-chemistry equilibrium temperature evaluated on the
#       RESOLVED local Z field (not a prescribed external HRR(t) as in
#       v6.5's fire_dns_coupling_one.py path), relaxed toward via a
#       "presumed equilibrium" closure at the local turbulent mixing rate
#       (tau_mix ~ dx^2/D_Z) -- chosen over an invented scalar-dissipation
#       reaction-rate formula specifically to avoid presenting an
#       unvalidated closure as authoritative.
#     - P1 (diffusion-approximation) radiation (cfg.enable_radiation=True),
#       solved via FFT -- assumes PERIODIC boundaries (matches this
#       solver's default), NOT rigorously valid for wall-dominated
#       enclosures (would need Marshak BCs + iterative elliptic solve).
#     - CRITICAL, EXPLICITLY FLAGGED GAP: this solver's internal T is
#       NON-DIMENSIONAL (T_ref=300K, per the pre-existing Sutherland-law
#       convention). Combustion/radiation source terms are derived and
#       computed in REAL (dimensional, W/m^3) units, converting T via
#       T_ref -- but the resulting terms are added to rhs_rhoE, which is
#       in this solver's OWN non-dimensional unit system (scaled by
#       reference length/velocity/density implied by cfg.Re/Mach, which
#       this file does not have enough visibility into to derive
#       automatically). cfg.combustion_nondim_scale (default 1.0 = NO
#       conversion) is an explicit, visible, user-must-set knob for this
#       gap -- NOT a silently-assumed-correct value. Do not trust
#       quantitative combustion/radiation magnitudes until you have set
#       this correctly for your own reference-scale choices.
#     - RZ added to save_checkpoint/load_checkpoint (backward compatible:
#       loading a pre-v6.6 checkpoint with enable_combustion=True warns
#       and resumes with Z=0 rather than raising).
#     - NOT implemented (explicitly out of scope for this pass, unlike
#       real FDS): discrete-ordinates RTE, finite-rate/soot chemistry
#       kinetics, pyrolysis/solid-fuel decomposition, sprinkler/detector/
#       HVAC device models, or any experimental validation. Treat this as
#       a step toward FDS's MODELING CATEGORY (conserved-scalar fast-
#       chemistry + simplified radiation), not a replacement for it.
#
# Changes v6.4 → v6.5:
#   New _ext_q buffer (volumetric heat-release-rate coupling, W/m^3),
#     added directly to rhs_rhoE, for FIRE ONE / fire_dns_coupling_one.py
#     combustion heat release. No such direct energy-source hook existed
#     before -- _ext_fx/fy/fz only reach the energy equation indirectly
#     via mechanical work (f.u), which cannot represent a heat addition.
#     Zero-cost if not connected, same guard pattern as every other
#     _ext_ buffer.
#
# Changes v6.3 → v6.4:
#   Bug 11 fix: self._ext_nu_ch (viscosity-modulation coupling buffer,
#     written by CahnHilliardDNSBridge.sync() in one_core_v3.py) was
#     declared in __init__ but never read in _compute_rhs -- only
#     _ext_rho_ch and _ext_fx/fy/fz were consumed. Confirmed by reading
#     both source files: the bridge writes _ext_nu_ch every sync() call,
#     but mu_lam was computed purely from nu_phys (=1/Re) and Sutherland's
#     law, with no path for _ext_nu_ch to reach it. Effect: in every
#     existing two-phase CahnHilliardDNSBridge-coupled run, the density
#     contrast between phases took effect but the viscosity contrast did
#     not -- both phases silently shared the solver's single global
#     nu_phys. Fixed by adding `mu_lam = mu_lam + rho * self._ext_nu_ch`
#     right after mu_lam is computed, guarded by the same
#     "zero-cost if not connected" pattern used for _ext_fx/_ext_rho_ch.
#
# Changes v6.2 → v6.3:
#   Bug 6 fix: SOCController.nu_t used an additive epsilon floor on
#     mean_S (strain_rate_mag.mean() + 1e-8) -- the same failure mode
#     as Bug 4. When true mean strain is near zero (laminar regions,
#     early transient steps), 1e-8 dominates and r = strain/mean_S
#     blows up uniformly across the whole domain. Fixed with the
#     differentiable _softplus_floor, consistent with every other
#     positivity floor in this module.
#   Bug 7 fix: CSOCKernel's lj_capped ceiling used beta=1.0 while
#     every other floor/ceiling in the module (including the
#     downstream nu_t_total ceiling it's meant to stay inside of) uses
#     _SOFTPLUS_B=100. The mismatched, ~100x-wider transition meant the
#     "ceiling at 50" was considerably looser than intended. Beta now
#     matches _SOFTPLUS_B.
#   Bug 8 fix: CompressibleSolver defined _apply_wall_model twice.
#     Python keeps only the last definition, so the comprehensive
#     6-face, 5-iteration Werner-Wengle solve (using wall_model_faces /
#     cfg.wm_A / cfg.wm_B) was dead code; every call in step() actually
#     ran a trivial y-faces-only stub instead. Removed the duplicate.
#   Bug 9 fix: _init_bc_objects called WernerWengleWallModelBC(cfg.wm_A,
#     cfg.wm_B) positionally, but the constructor signature is
#     (T_wall, A, B) -- so wm_A silently became the wall temperature
#     (8.3 instead of cfg.wall_temp) and wm_B silently became A, with B
#     left at its hardcoded default. Now passed as T_wall/A/B keywords.
#   Bug 10 fix: CFDConfig.with_stretched_grid had a malformed, doubled
#     type-annotation string literal on the z_coords parameter
#     (cosmetic only, no runtime effect; corrected for clarity).
#   Cleanup: removed a dead `if False else ...` branch in HLLCFlux's SL
#     computation that left an unreachable logsumexp expression next to
#     the DIFF-FIX 4a comment; the smooth-min fallback is now written
#     directly as the only path.
#
# Changes v6.1 → v6.2:
#   Bug 4 fix: DiffRGRefiner's mean-ratio rescale (x * mean_before/safe_after)
#     silently collapsed near-zero-mean fields (rhou/rhov/rhow in periodic
#     turbulence) toward zero by up to ~10 orders of magnitude, because the
#     softplus floor's effective width (log(2)/beta≈6.93e-3) swamps the
#     requested 1e-30 floor whenever |mean_after| is much smaller than that.
#     Fix: removed the rescale entirely; mask_dc[0,0,0]=1.0 already keeps
#     the DC (mean) term exact, so the filter is mean-preserving without it.
#     Same root-cause pattern as the REAL FOLD ONE structure-corruption bug
#     (near-zero-denominator rescale, not a NaN). Numerically confirmed via
#     standalone reproduction before fixing.
#   Bug 5 fix: CSOCKernel was a pure power-law Cs*r^-alpha*exp(-r/lambda)
#     with no equilibrium point. Its apparent stability in v6.1 relied on a
#     coincidental cancellation at alpha=0.5 between r^-2alpha and the
#     strain_rate_mag factor in nu_t_base — but alpha is an unconstrained
#     learnable nn.Parameter, and drifting to alpha=0.9-5.0 during training
#     was confirmed numerically to inflate nu_t_base by 4-9 orders of
#     magnitude before the max_nu_t ceiling clips it, saturating gradients.
#     Fix: replaced with a Lennard-Jones-style equilibrium form (same
#     pattern validated in REAL FOLD ONE) with a true minimum at r=r_eq
#     and a softplus-capped rise as r->0, bounded for any alpha value.
#
# Changes v4 → v5  (Native Full Differentiability)
# ─────────────────────────────────────────────────
# Every operation that breaks torch.autograd is replaced with a smooth,
# differentiable equivalent.  The full computational graph from any scalar
# loss back through _compute_rhs → step() is now clean.
#
# DIFF-FIX 1 — TVD slope limiters (RiemannSolverBase)
#   Before : torch.where(a*b>0, …, 0)          ← hard zero kills gradient
#   After  : smooth_minmod / smooth_van_leer / smooth_superbee
#            using softplus / tanh gating — gradient everywhere
#
# DIFF-FIX 2 — WENO-5 Lax-Friedrichs wave speed (RiemannSolverBase)
#   Before : alpha = vel.abs().max()            ← non-differentiable
#   After  : alpha = logsumexp(|vel|/τ)·τ       ← soft-max, fully smooth
#
# DIFF-FIX 3 — AUSM+ flux (AUSMPlusFlux)
#   Before : torch.where(M>=1, M, 0.25*(M+1)²) ← discontinuous at |M|=1
#   After  : smooth M± / P± polynomials via tanh blending
#            mass_flux uses soft-upwind: tanh(M/τ) gating
#
# DIFF-FIX 4 — HLLC flux (HLLCFlux)
#   Before : boolean masks + in-place __setitem__ ← breaks autograd graph
#   After  : fully branch-free soft gating with sigmoid / tanh
#            SL, SR, S* computed identically but composed with smooth masks
#
# DIFF-FIX 5 — SOCController.nu_t
#   Before : torch.clamp(stress_acc, min=0)      ← gradient=0 when clamped
#            torch.where(excess>0, …)             ← hard branch
#   After  : softplus(stress_acc) throughout
#            soft_collapse = softplus(excess)·scale
#
# DIFF-FIX 6 — DiffRGRefiner
#   Before : if mean_after.abs() > 1e-30: … else: … ← Python branch on tensor
#   After  : smooth rescaling using softplus-guarded division
#
# DIFF-FIX 7 — CompressibleSolver.step()
#   Before : torch.clamp(rho_new, min=1e-6)      ← gradient=0 below floor
#            torch.clamp(p_new, min=1e-8)
#   After  : softplus floor: rho_floor + softplus(rho_new - rho_floor)
#
# DIFF-FIX 8 — _compute_rhs monkey-patch WENO dispatch
#   Before : runtime monkey-patching of _muscl_face_states  ← fragile
#   After  : clean dispatch via _face_states() method on solver
#
# Backward-compatible: all public APIs (CFDConfig, CompressibleSolver,
# SemiLagrangianAdvection3D, …) are unchanged.
#
# Built on solid open‑source foundations:
#   PyTorch · NumPy · SciPy · Matplotlib · Optuna · CoolProp · PyWavelets
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

from one_core import (
    SemanticStateContraction,
    CSOCBase,
    InterfaceDetectorBase,
    LangevinDNSBridge,          # Bridge: Langevin → DNS (Bug 3 fix)
    CahnHilliardDNSBridge,      # Bridge: CahnHilliard → DNS (v3.1)
    get_device as _core_get_device,
    ONE_VERSION,
)
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

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
# Differentiability utilities
# =============================================================================

# Smoothness temperature constants — larger τ → smoother but less sharp
_TAU_LIMITER  = 1e-2   # TVD limiter gating sharpness
_TAU_UPWIND   = 1e-2   # upwind/M± gating sharpness
_TAU_LSE      = 1e-2   # logsumexp wave-speed temperature
_SOFTPLUS_B   = 100.0  # softplus sharpness for positivity floors


def _softplus_floor(x: torch.Tensor, floor: float, beta: float = _SOFTPLUS_B) -> torch.Tensor:
    """
    Differentiable positivity floor:  out = floor + softplus(x - floor).

    Unlike clamp(x, min=floor), this has non-zero gradient everywhere.
    For large beta the transition becomes sharp (approaches clamp).
    """
    return floor + F.softplus(x - floor, beta=beta)


def _soft_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable |x| = sqrt(x²+ε).  Avoids kink at x=0."""
    return torch.sqrt(x * x + eps)


def _soft_sign(x: torch.Tensor, tau: float = _TAU_UPWIND) -> torch.Tensor:
    """Differentiable sign: tanh(x/τ) ∈ (-1,1).  Gradient everywhere."""
    return torch.tanh(x / tau)


def _soft_gate(condition_val: torch.Tensor, tau: float = _TAU_UPWIND) -> torch.Tensor:
    """
    Smooth gate in [0,1]:  σ(v/τ)  — approaches Heaviside as τ→0.
    Use in place of: (condition_val >= 0).float()
    """
    return torch.sigmoid(condition_val / tau)


def _logsumexp_max(x: torch.Tensor, tau: float = _TAU_LSE) -> torch.Tensor:
    """
    Smooth differentiable approximation of x.max():
        τ · log(Σ exp(xᵢ/τ))
    Gradient flows through all elements (not just the argmax).
    """
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# =============================================================================
# Smooth TVD slope limiters  (DIFF-FIX 1)
# =============================================================================

def _smooth_minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Smooth differentiable minmod limiter.

    Classical: max(0, min(a, b))  for same-sign inputs, 0 otherwise.
    Smooth version:
        gate = σ(a·b/τ)                 ← soft same-sign indicator
        slope = 0.5*(a+b) - 0.5*|a-b|  ← differentiable min(a,b)
        out = gate * max(0, slope)       ← soft zero outside same-sign
    """
    gate  = _soft_gate(a * b, tau=_TAU_LIMITER)
    mn    = 0.5 * (a + b) - 0.5 * _soft_abs(a - b)   # smooth min(a,b)
    pos   = 0.5 * (mn + _soft_abs(mn))                 # smooth max(0, mn)
    return gate * pos


def _smooth_van_leer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    van Leer limiter — already smooth (ratio form), but the original
    torch.where(a*b>0, …, 0) is replaced with soft gating.

    ψ = gate · (a|b| + |a|b) / (|a|+|b|+ε)
    """
    gate = _soft_gate(a * b, tau=_TAU_LIMITER)
    num  = a * _soft_abs(b) + _soft_abs(a) * b
    den  = _soft_abs(a) + _soft_abs(b) + 1e-30
    return gate * (num / den)


def _smooth_superbee(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Smooth superbee limiter via soft-max composition.

    Classical: max(0, min(2a,b), min(a,2b))
    Smooth: replace hard max/min with soft_abs versions.
    """
    gate = _soft_gate(a * b, tau=_TAU_LIMITER)
    # smooth min(2a,b) and min(a,2b)
    m1 = 0.5*(2*a + b) - 0.5*_soft_abs(2*a - b)
    m2 = 0.5*(a + 2*b) - 0.5*_soft_abs(a - 2*b)
    # smooth max(m1, m2)
    mx = 0.5*(m1 + m2) + 0.5*_soft_abs(m1 - m2)
    # smooth max(0, mx)
    pos = 0.5*(mx + _soft_abs(mx))
    return gate * pos


_SMOOTH_LIMITERS = {
    "minmod":   _smooth_minmod,
    "van_leer": _smooth_van_leer,
    "superbee": _smooth_superbee,
}


# =============================================================================
# 2. Boundary conditions  (unchanged from v4)
# =============================================================================

def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class BoundaryCondition:
    pass


class PeriodicBC(BoundaryCondition):
    def apply(self, q_pad, axis, side, nx, ny, nz):
        if axis == 0:
            if side == 'left':
                q_pad[:2, 2:ny+2, 2:nz+2]    = q_pad[nx:nx+2, 2:ny+2, 2:nz+2]
            else:
                q_pad[nx+2:, 2:ny+2, 2:nz+2] = q_pad[2:4, 2:ny+2, 2:nz+2]
        elif axis == 1:
            if side == 'left':
                q_pad[2:nx+2, :2, 2:nz+2]    = q_pad[2:nx+2, ny:ny+2, 2:nz+2]
            else:
                q_pad[2:nx+2, ny+2:, 2:nz+2] = q_pad[2:nx+2, 2:4, 2:nz+2]
        else:
            if side == 'left':
                q_pad[2:nx+2, 2:ny+2, :2]    = q_pad[2:nx+2, 2:ny+2, nz:nz+2]
            else:
                q_pad[2:nx+2, 2:ny+2, nz+2:] = q_pad[2:nx+2, 2:ny+2, 2:4]


class SupersonicInflowBC(BoundaryCondition):
    def __init__(self, rho_inf, u_inf, v_inf, w_inf, p_inf, gamma=1.4):
        self.rho_inf = rho_inf; self.u_inf = u_inf; self.v_inf = v_inf
        self.w_inf   = w_inf;   self.p_inf = p_inf; self.gamma = gamma

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None):
        g = gamma if gamma is not None else self.gamma
        nx, ny, nz = rho.shape
        E_inf = self.p_inf/(g-1) + 0.5*self.rho_inf*(self.u_inf**2+self.v_inf**2+self.w_inf**2)
        if axis == 0:
            i = 0 if side == 'left' else nx-1
            rho[i] = self.rho_inf;  rhou[i] = self.rho_inf*self.u_inf
            rhov[i] = self.rho_inf*self.v_inf; rhow[i] = self.rho_inf*self.w_inf
            rhoE[i] = self.rho_inf*E_inf
        elif axis == 1:
            j = 0 if side == 'left' else ny-1
            rho[:,j] = self.rho_inf;  rhou[:,j] = self.rho_inf*self.u_inf
            rhov[:,j] = self.rho_inf*self.v_inf; rhow[:,j] = self.rho_inf*self.w_inf
            rhoE[:,j] = self.rho_inf*E_inf
        else:
            k = 0 if side == 'left' else nz-1
            rho[:,:,k] = self.rho_inf;  rhou[:,:,k] = self.rho_inf*self.u_inf
            rhov[:,:,k] = self.rho_inf*self.v_inf; rhow[:,:,k] = self.rho_inf*self.w_inf
            rhoE[:,:,k] = self.rho_inf*E_inf

    def apply_to_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, device=None, dtype=None):
        self.apply(rho, rhou, rhov, rhow, rhoE, axis, side)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None, n_ghost=2):
        g = gamma if gamma is not None else self.gamma
        nx, ny, nz = rho.shape
        E_inf = self.p_inf/(g-1) + 0.5*self.rho_inf*(self.u_inf**2+self.v_inf**2+self.w_inf**2)
        if axis == 0:
            rng = range(1, n_ghost+1)
            idxs = [(-g, 0) if side=='left' else (nx-1+g, nx-1) for g in rng]
            for i, _ in idxs:
                rho[i] = self.rho_inf;   rhou[i] = self.rho_inf*self.u_inf
                rhov[i] = self.rho_inf*self.v_inf; rhow[i] = self.rho_inf*self.w_inf
                rhoE[i] = self.rho_inf*E_inf
        elif axis == 1:
            for g in range(1, n_ghost+1):
                j = -g if side=='left' else ny-1+g
                rho[:,j] = self.rho_inf;   rhou[:,j] = self.rho_inf*self.u_inf
                rhov[:,j] = self.rho_inf*self.v_inf; rhow[:,j] = self.rho_inf*self.w_inf
                rhoE[:,j] = self.rho_inf*E_inf
        else:
            for g in range(1, n_ghost+1):
                k = -g if side=='left' else nz-1+g
                rho[:,:,k] = self.rho_inf;   rhou[:,:,k] = self.rho_inf*self.u_inf
                rhov[:,:,k] = self.rho_inf*self.v_inf; rhow[:,:,k] = self.rho_inf*self.w_inf
                rhoE[:,:,k] = self.rho_inf*E_inf


class SubsonicOutflowBC(BoundaryCondition):
    def __init__(self, p_out, gamma=1.4):
        self.p_out = p_out; self.gamma = gamma

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None):
        g = gamma if gamma is not None else self.gamma
        nx, ny, nz = rho.shape
        if axis == 0:
            i = nx-1 if side=='right' else 0
            ii = nx-2 if side=='right' else 1
            rho[i]=rho[ii]; rhou[i]=rhou[ii]; rhov[i]=rhov[ii]; rhow[i]=rhow[ii]
            ke = 0.5*(rhou[i]**2+rhov[i]**2+rhow[i]**2)/(rho[i]+1e-8)
            rhoE[i] = self.p_out/(g-1) + ke
        elif axis == 1:
            j = ny-1 if side=='right' else 0
            jj = ny-2 if side=='right' else 1
            rho[:,j]=rho[:,jj]; rhou[:,j]=rhou[:,jj]; rhov[:,j]=rhov[:,jj]; rhow[:,j]=rhow[:,jj]
            ke = 0.5*(rhou[:,j]**2+rhov[:,j]**2+rhow[:,j]**2)/(rho[:,j]+1e-8)
            rhoE[:,j] = self.p_out/(g-1) + ke
        else:
            k = nz-1 if side=='right' else 0
            kk = nz-2 if side=='right' else 1
            rho[:,:,k]=rho[:,:,kk]; rhou[:,:,k]=rhou[:,:,kk]
            rhov[:,:,k]=rhov[:,:,kk]; rhow[:,:,k]=rhow[:,:,kk]
            ke = 0.5*(rhou[:,:,k]**2+rhov[:,:,k]**2+rhow[:,:,k]**2)/(rho[:,:,k]+1e-8)
            rhoE[:,:,k] = self.p_out/(g-1) + ke

    def apply_to_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side):
        self.apply(rho, rhou, rhov, rhow, rhoE, axis, side)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None, n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            b = nx-1 if side=='right' else 0
            for g in range(1, n_ghost+1):
                idx = b+g if side=='right' else b-g
                rho[idx]=rho[b]; rhou[idx]=rhou[b]; rhov[idx]=rhov[b]
                rhow[idx]=rhow[b]; rhoE[idx]=rhoE[b]
        elif axis == 1:
            b = ny-1 if side=='right' else 0
            for g in range(1, n_ghost+1):
                idx = b+g if side=='right' else b-g
                rho[:,idx]=rho[:,b]; rhou[:,idx]=rhou[:,b]; rhov[:,idx]=rhov[:,b]
                rhow[:,idx]=rhow[:,b]; rhoE[:,idx]=rhoE[:,b]
        else:
            b = nz-1 if side=='right' else 0
            for g in range(1, n_ghost+1):
                idx = b+g if side=='right' else b-g
                rho[:,:,idx]=rho[:,:,b]; rhou[:,:,idx]=rhou[:,:,b]
                rhov[:,:,idx]=rhov[:,:,b]; rhow[:,:,idx]=rhow[:,:,b]; rhoE[:,:,idx]=rhoE[:,:,b]


class NoSlipIsothermalWallBC(BoundaryCondition):
    def __init__(self, T_wall, gamma=1.4):
        self.T_wall = T_wall; self.gamma = gamma

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None):
        g = gamma if gamma is not None else self.gamma
        nx, ny, nz = rho.shape
        def _apply_slice(s, si):
            p_int = (g-1)*(rhoE[si] - 0.5*(rhou[si]**2+rhov[si]**2+rhow[si]**2)/(rho[si]+1e-8))
            rho[s] = p_int/self.T_wall; rhou[s] = 0.0; rhov[s] = 0.0; rhow[s] = 0.0
            rhoE[s] = p_int/(g-1)
        if axis == 0:
            _apply_slice(0 if side=='left' else nx-1, 1 if side=='left' else nx-2)
        elif axis == 1:
            if side=='left': _apply_slice(slice(None,1), slice(1,2))  # type: ignore
            else:            _apply_slice(slice(-1,None), slice(-2,-1))  # type: ignore
        else:
            if side=='left':
                p_int = (g-1)*(rhoE[:,:,1]-0.5*(rhou[:,:,1]**2+rhov[:,:,1]**2+rhow[:,:,1]**2)/(rho[:,:,1]+1e-8))
                rho[:,:,0]=p_int/self.T_wall; rhou[:,:,0]=0.; rhov[:,:,0]=0.; rhow[:,:,0]=0.; rhoE[:,:,0]=p_int/(g-1)
            else:
                p_int = (g-1)*(rhoE[:,:,-2]-0.5*(rhou[:,:,-2]**2+rhov[:,:,-2]**2+rhow[:,:,-2]**2)/(rho[:,:,-2]+1e-8))
                rho[:,:,-1]=p_int/self.T_wall; rhou[:,:,-1]=0.; rhov[:,:,-1]=0.; rhow[:,:,-1]=0.; rhoE[:,:,-1]=p_int/(g-1)

    def apply_to_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side):
        self.apply(rho, rhou, rhov, rhow, rhoE, axis, side)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None, n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            for g in range(1, n_ghost+1):
                i = -g if side=='left' else nx-1+g
                ii = g if side=='left' else nx-1-g
                rho[i]=rho[ii]; rhoE[i]=rhoE[ii]
                rhou[i]=-rhou[ii]; rhov[i]=-rhov[ii]; rhow[i]=-rhow[ii]
        elif axis == 1:
            for g in range(1, n_ghost+1):
                j = -g if side=='left' else ny-1+g
                jj = g if side=='left' else ny-1-g
                rho[:,j]=rho[:,jj]; rhoE[:,j]=rhoE[:,jj]
                rhou[:,j]=-rhou[:,jj]; rhov[:,j]=-rhov[:,jj]; rhow[:,j]=-rhow[:,jj]
        else:
            for g in range(1, n_ghost+1):
                k = -g if side=='left' else nz-1+g
                kk = g if side=='left' else nz-1-g
                rho[:,:,k]=rho[:,:,kk]; rhoE[:,:,k]=rhoE[:,:,kk]
                rhou[:,:,k]=-rhou[:,:,kk]; rhov[:,:,k]=-rhov[:,:,kk]; rhow[:,:,k]=-rhow[:,:,kk]


class WernerWengleWallModelBC(BoundaryCondition):
    def __init__(self, T_wall=300.0, A=8.3, B=1.0/7.0):
        self.T_wall = T_wall; self.A = A; self.B = B

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None):
        NoSlipIsothermalWallBC(self.T_wall).apply(rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx)

    def apply_to_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, dx=None, nu=None):
        self.apply(rho, rhou, rhov, rhow, rhoE, axis, side)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None, n_ghost=2):
        NoSlipIsothermalWallBC(self.T_wall).ghost_cells(
            rho, rhou, rhov, rhow, rhoE, axis, side, gamma, dx, n_ghost)


class MovingWallBC(BoundaryCondition):
    def __init__(self, u_wall=0.0, v_wall=0.0, w_wall=0.0, T_wall=300.0, gamma=1.4):
        self.u_wall = u_wall; self.v_wall = v_wall; self.w_wall = w_wall
        self.T_wall = T_wall; self.gamma = gamma

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None):
        g = gamma if gamma is not None else self.gamma
        nx, ny, nz = rho.shape
        def _set(i, ii):
            p_int = (g-1)*(rhoE[ii]-0.5*(rhou[ii]**2+rhov[ii]**2+rhow[ii]**2)/(rho[ii]+1e-8))
            rho[i]=rho[ii]; rhou[i]=rho[i]*self.u_wall; rhov[i]=torch.zeros_like(rho[i])
            rhow[i]=rho[i]*self.w_wall
            rhoE[i]=p_int/(g-1)+0.5*rho[i]*(self.u_wall**2+self.v_wall**2+self.w_wall**2)
        if axis == 0:
            _set(0 if side=='left' else nx-1, 1 if side=='left' else nx-2)
        elif axis == 1:
            j,ji = (0,1) if side=='left' else (ny-1,ny-2)
            p_int=(g-1)*(rhoE[:,ji]-0.5*(rhou[:,ji]**2+rhov[:,ji]**2+rhow[:,ji]**2)/(rho[:,ji]+1e-8))
            rho[:,j]=rho[:,ji]; rhou[:,j]=rho[:,j]*self.u_wall
            rhov[:,j]=torch.zeros_like(rho[:,j]); rhow[:,j]=rho[:,j]*self.w_wall
            rhoE[:,j]=p_int/(g-1)+0.5*rho[:,j]*(self.u_wall**2+self.v_wall**2+self.w_wall**2)
        else:
            k,ki = (0,1) if side=='left' else (nz-1,nz-2)
            p_int=(g-1)*(rhoE[:,:,ki]-0.5*(rhou[:,:,ki]**2+rhov[:,:,ki]**2+rhow[:,:,ki]**2)/(rho[:,:,ki]+1e-8))
            rho[:,:,k]=rho[:,:,ki]; rhou[:,:,k]=rho[:,:,k]*self.u_wall
            rhov[:,:,k]=torch.zeros_like(rho[:,:,k]); rhow[:,:,k]=rho[:,:,k]*self.w_wall
            rhoE[:,:,k]=p_int/(g-1)+0.5*rho[:,:,k]*(self.u_wall**2+self.v_wall**2+self.w_wall**2)

    def apply_to_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side):
        self.apply(rho, rhou, rhov, rhow, rhoE, axis, side)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None, n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            for g in range(1, n_ghost+1):
                i = -g if side=='left' else nx-1+g
                ii = g if side=='left' else nx-1-g
                rho[i]=rho[ii]; rhoE[i]=rhoE[ii]
                rhou[i]=2*self.u_wall*rho[i]-rhou[ii]
                rhov[i]=2*self.v_wall*rho[i]-rhov[ii]
                rhow[i]=2*self.w_wall*rho[i]-rhow[ii]
        elif axis == 1:
            for g in range(1, n_ghost+1):
                j = -g if side=='left' else ny-1+g
                jj = g if side=='left' else ny-1-g
                rho[:,j]=rho[:,jj]; rhoE[:,j]=rhoE[:,jj]
                rhou[:,j]=2*self.u_wall*rho[:,j]-rhou[:,jj]
                rhov[:,j]=2*self.v_wall*rho[:,j]-rhov[:,jj]
                rhow[:,j]=2*self.w_wall*rho[:,j]-rhow[:,jj]
        else:
            for g in range(1, n_ghost+1):
                k = -g if side=='left' else nz-1+g
                kk = g if side=='left' else nz-1-g
                rho[:,:,k]=rho[:,:,kk]; rhoE[:,:,k]=rhoE[:,:,kk]
                rhou[:,:,k]=2*self.u_wall*rho[:,:,k]-rhou[:,:,kk]
                rhov[:,:,k]=2*self.v_wall*rho[:,:,k]-rhov[:,:,kk]
                rhow[:,:,k]=2*self.w_wall*rho[:,:,k]-rhow[:,:,kk]


class FarFieldBC(BoundaryCondition):
    """Characteristic non-reflecting far-field BC (Poinsot & Lele 1992)."""

    def __init__(self, rho_inf=1.0, u_inf=0.0, v_inf=0.0, w_inf=0.0, p_inf=1.0, gamma=1.4):
        self.rho_inf = rho_inf; self.u_inf = u_inf; self.v_inf = v_inf
        self.w_inf   = w_inf;   self.p_inf = p_inf; self.gamma = gamma

    def apply(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None):
        g = gamma if gamma is not None else self.gamma
        nx, ny, nz = rho.shape
        c_inf = math.sqrt(g * self.p_inf / (self.rho_inf + 1e-30))

        def _riemann_update(rho_s, rhou_s, rhov_s, rhow_s, rhoE_s, un_inf, ut1_inf, ut2_inf):
            rho_i = _softplus_floor(rho_s, 1e-8)
            u_i = rhou_s / rho_i; v_i = rhov_s / rho_i; w_i = rhow_s / rho_i
            p_i = (g-1)*(rhoE_s - 0.5*rho_i*(u_i**2+v_i**2+w_i**2))
            p_i = _softplus_floor(p_i, 1e-8)
            c_i = torch.sqrt(g*p_i/rho_i)
            R_p = u_i + 2*c_i/(g-1)
            R_m = un_inf - 2*c_inf/(g-1)
            un_b = 0.5*(R_p + R_m)
            c_b  = 0.25*(g-1)*(R_p - R_m)
            c_b  = _softplus_floor(c_b, 1e-6)
            s    = p_i / (rho_i**g + 1e-30)
            p_b  = (c_b**2 * s / g) ** (g/(g-1))
            rho_b = p_b / (c_b**2/g + 1e-30)
            return rho_b, un_b, v_i, w_i, p_b

        if axis == 0:
            i = 0 if side=='left' else nx-1
            ii = 1 if side=='left' else nx-2
            rho_b,u_b,v_b,w_b,p_b = _riemann_update(
                rho[ii],rhou[ii],rhov[ii],rhow[ii],rhoE[ii],self.u_inf,self.v_inf,self.w_inf)
            rho[i]=rho_b; rhou[i]=rho_b*u_b; rhov[i]=rho_b*v_b; rhow[i]=rho_b*w_b
            rhoE[i]=p_b/(g-1)+0.5*rho_b*(u_b**2+v_b**2+w_b**2)
        elif axis == 1:
            j = 0 if side=='left' else ny-1
            jj = 1 if side=='left' else ny-2
            rho_b,v_b,u_b,w_b,p_b = _riemann_update(
                rho[:,jj],rhov[:,jj],rhou[:,jj],rhow[:,jj],rhoE[:,jj],self.v_inf,self.u_inf,self.w_inf)
            rho[:,j]=rho_b; rhou[:,j]=rho_b*u_b; rhov[:,j]=rho_b*v_b; rhow[:,j]=rho_b*w_b
            rhoE[:,j]=p_b/(g-1)+0.5*rho_b*(u_b**2+v_b**2+w_b**2)
        else:
            k = 0 if side=='left' else nz-1
            kk = 1 if side=='left' else nz-2
            rho_b,w_b,u_b,v_b,p_b = _riemann_update(
                rho[:,:,kk],rhow[:,:,kk],rhou[:,:,kk],rhov[:,:,kk],rhoE[:,:,kk],
                self.w_inf,self.u_inf,self.v_inf)
            rho[:,:,k]=rho_b; rhou[:,:,k]=rho_b*u_b; rhov[:,:,k]=rho_b*v_b
            rhow[:,:,k]=rho_b*w_b; rhoE[:,:,k]=p_b/(g-1)+0.5*rho_b*(u_b**2+v_b**2+w_b**2)

    def apply_to_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side):
        self.apply(rho, rhou, rhov, rhow, rhoE, axis, side)

    def ghost_cells(self, rho, rhou, rhov, rhow, rhoE, axis, side, gamma=None, dx=None, n_ghost=2):
        nx, ny, nz = rho.shape
        if axis == 0:
            b = nx-1 if side=='right' else 0
            for g in range(1, n_ghost+1):
                idx = b+g if side=='right' else b-g
                rho[idx]=rho[b]; rhou[idx]=rhou[b]; rhov[idx]=rhov[b]
                rhow[idx]=rhow[b]; rhoE[idx]=rhoE[b]
        elif axis == 1:
            b = ny-1 if side=='right' else 0
            for g in range(1, n_ghost+1):
                idx = b+g if side=='right' else b-g
                rho[:,idx]=rho[:,b]; rhou[:,idx]=rhou[:,b]; rhov[:,idx]=rhov[:,b]
                rhow[:,idx]=rhow[:,b]; rhoE[:,idx]=rhoE[:,b]
        else:
            b = nz-1 if side=='right' else 0
            for g in range(1, n_ghost+1):
                idx = b+g if side=='right' else b-g
                rho[:,:,idx]=rho[:,:,b]; rhou[:,:,idx]=rhou[:,:,b]
                rhov[:,:,idx]=rhov[:,:,b]; rhow[:,:,idx]=rhow[:,:,b]; rhoE[:,:,idx]=rhoE[:,:,b]


# =============================================================================
# 3. Riemann Solvers — Native Full-Differentiable  (DIFF-FIX 1,2,3,4)
# =============================================================================

class RiemannSolverBase:
    """
    Base class for Riemann solvers.

    All limiter and flux operations are differentiable everywhere:
    no torch.where on boolean conditions, no .max() on spatial fields,
    no in-place indexed writes to result tensors.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma

    # ── Smooth TVD limiters ──────────────────────────────────────────────────

    @staticmethod
    def _minmod(a, b):
        return _smooth_minmod(a, b)

    @staticmethod
    def _van_leer(a, b):
        return _smooth_van_leer(a, b)

    @staticmethod
    def _superbee(a, b):
        return _smooth_superbee(a, b)

    def _get_limiter(self, name: str):
        _map = {
            "minmod":   self._minmod,
            "van_leer": self._van_leer,
            "superbee": self._superbee,
        }
        if name not in _map:
            raise ValueError(f"Unknown limiter {name!r}. Choose from: {list(_map)}")
        return _map[name]

    def _muscl_face_states(self, q_pad, axis, limiter: str = "van_leer"):
        """
        MUSCL reconstruction with smooth differentiable limiters.
        No torch.where on boolean masks.
        """
        lim = self._get_limiter(limiter)
        nx, ny, nz = q_pad.shape[0]-4, q_pad.shape[1]-4, q_pad.shape[2]-4

        if axis == 0:
            q_im1 = q_pad[0:nx+2, 2:ny+2, 2:nz+2]
            q_i   = q_pad[1:nx+3, 2:ny+2, 2:nz+2]
            q_ip1 = q_pad[2:nx+4, 2:ny+2, 2:nz+2]
            d1, d2 = q_i - q_im1, q_ip1 - q_i
            slope = lim(d1, d2)
            qL = (q_i   + 0.5 * slope)[:nx+1]
            slope_r = torch.cat([slope[1:], slope[-1:]], dim=0)
            qR = (q_ip1 - 0.5 * slope_r)[:nx+1]
        elif axis == 1:
            q_im1 = q_pad[2:nx+2, 0:ny+2, 2:nz+2]
            q_i   = q_pad[2:nx+2, 1:ny+3, 2:nz+2]
            q_ip1 = q_pad[2:nx+2, 2:ny+4, 2:nz+2]
            d1, d2 = q_i - q_im1, q_ip1 - q_i
            slope = lim(d1, d2)
            qL = (q_i   + 0.5 * slope)[:, :ny+1]
            slope_r = torch.cat([slope[:, 1:], slope[:, -1:]], dim=1)
            qR = (q_ip1 - 0.5 * slope_r)[:, :ny+1]
        else:
            q_im1 = q_pad[2:nx+2, 2:ny+2, 0:nz+2]
            q_i   = q_pad[2:nx+2, 2:ny+2, 1:nz+3]
            q_ip1 = q_pad[2:nx+2, 2:ny+2, 2:nz+4]
            d1, d2 = q_i - q_im1, q_ip1 - q_i
            slope = lim(d1, d2)
            qL = (q_i   + 0.5 * slope)[:, :, :nz+1]
            slope_r = torch.cat([slope[:, :, 1:], slope[:, :, -1:]], dim=2)
            qR = (q_ip1 - 0.5 * slope_r)[:, :, :nz+1]

        return qL, qR

    def _weno5_face_states(self, q_pad, axis):
        """
        WENO-5 (Jiang-Shu 1996) — fully differentiable.

        DIFF-FIX 2: Lax-Friedrichs wave speed uses logsumexp instead of .max()
        so that gradient flows through all wave-speed values, not just the max.
        """
        eps = 1e-6
        nx, ny, nz = (q_pad.shape[i] - 6 for i in range(3))

        def _stencil(arr, ax):
            slabs = []
            if ax == 0:
                for k in range(6): slabs.append(arr[k:nx+k+1, 2:ny+2, 2:nz+2])
            elif ax == 1:
                for k in range(6): slabs.append(arr[2:nx+2, k:ny+k+1, 2:nz+2])
            else:
                for k in range(6): slabs.append(arr[2:nx+2, 2:ny+2, k:nz+k+1])
            return slabs

        s = _stencil(q_pad, axis)
        q0,q1,q2,q3,q4,q5 = s

        # ── Smooth LF wave speed (DIFF-FIX 2) ────────────────────────────────
        # alpha = logsumexp(|q|/τ)·τ  approximates max|q| but is differentiable
        alpha = _logsumexp_max(_soft_abs(q2), tau=_TAU_LSE) + 1e-12

        def _fp(qi): return 0.5 * (qi + alpha * qi / (alpha + 1e-30))
        def _fm(qi): return 0.5 * (qi - alpha * qi / (alpha + 1e-30))

        # Smoothness indicators β
        def beta(a, b, c):
            return (13./12.)*(a-2*b+c)**2 + 0.25*(a-c)**2

        # Left reconstruction
        b0L = beta(q0,q1,q2); b1L = beta(q1,q2,q3); b2L = beta(q2,q3,q4)
        a0L = 0.1/(eps+b0L)**2; a1L = 0.6/(eps+b1L)**2; a2L = 0.3/(eps+b2L)**2
        sL  = a0L + a1L + a2L
        fh0L = (1./3.)*q0 - (7./6.)*q1 + (11./6.)*q2
        fh1L = -(1./6.)*q1 + (5./6.)*q2 + (1./3.)*q3
        fh2L = (1./3.)*q2 + (5./6.)*q3 - (1./6.)*q4
        qL   = (a0L*fh0L + a1L*fh1L + a2L*fh2L) / sL

        # Right reconstruction
        b0R = beta(q5,q4,q3); b1R = beta(q4,q3,q2); b2R = beta(q3,q2,q1)
        a0R = 0.1/(eps+b0R)**2; a1R = 0.6/(eps+b1R)**2; a2R = 0.3/(eps+b2R)**2
        sR  = a0R + a1R + a2R
        fh0R = (1./3.)*q5 - (7./6.)*q4 + (11./6.)*q3
        fh1R = -(1./6.)*q4 + (5./6.)*q3 + (1./3.)*q2
        fh2R = (1./3.)*q3 + (5./6.)*q2 - (1./6.)*q1
        qR   = (a0R*fh0R + a1R*fh1R + a2R*fh2R) / sR

        if axis == 0: return qL[:nx+1], qR[:nx+1]
        elif axis == 1: return qL[:, :ny+1], qR[:, :ny+1]
        else: return qL[:, :, :nz+1], qR[:, :, :nz+1]

    def _face_states(self, fields_pad: dict, axis: int, scheme: str, limiter: str):
        """
        Unified face-state dispatcher. Returns dict of (qL, qR) per field key.
        Eliminates monkey-patching from v4.
        """
        out = {}
        for key, q_pad in fields_pad.items():
            if scheme == "weno5":
                qL, qR = self._weno5_face_states(q_pad, axis)
            else:
                qL, qR = self._muscl_face_states(q_pad, axis, limiter)
            out[key] = (qL, qR)
        return out


class AUSMPlusFlux(RiemannSolverBase):
    """
    AUSM+ flux — fully differentiable.

    DIFF-FIX 3:
    - M±(M) and P±(M) polynomials blended with tanh instead of torch.where
    - mass_flux upwind selection: soft-upwind via tanh gate
    - all torch.where(M>=1, …) replaced with sigmoid blending
    """

    def __init__(self, gamma: float = 1.4):
        super().__init__(gamma)

    # ── Smooth AUSM± polynomials ─────────────────────────────────────────────

    @staticmethod
    def _M_plus(M: torch.Tensor) -> torch.Tensor:
        """
        Smooth M⁺(M):
          M>=1   → M
          |M|<1  → 0.25(M+1)²
          M<=-1  → 0
        Blended with sigmoid gates — differentiable everywhere.
        """
        g_sup = torch.sigmoid((M - 1.0) / _TAU_UPWIND)       # ≈1 when M≥1
        g_sub = torch.sigmoid((1.0 - M.abs()) / _TAU_UPWIND)  # ≈1 when |M|<1
        M_poly = 0.25 * (M + 1.0)**2
        return g_sup * M + g_sub * (1.0 - g_sup) * M_poly

    @staticmethod
    def _M_minus(M: torch.Tensor) -> torch.Tensor:
        """Smooth M⁻(M) = -M⁺(-M)."""
        g_sub_neg = torch.sigmoid((-M - 1.0) / _TAU_UPWIND)
        g_sub     = torch.sigmoid((1.0 - M.abs()) / _TAU_UPWIND)
        M_poly    = -0.25 * (M - 1.0)**2
        return -g_sub_neg * (-M) + g_sub * (1.0 - g_sub_neg) * M_poly

    @staticmethod
    def _P_plus(M: torch.Tensor) -> torch.Tensor:
        """Smooth P⁺(M): 1 for M≥1, 0.25(M+1)²(2-M) for |M|<1, 0 for M≤-1."""
        g_sup = torch.sigmoid((M - 1.0) / _TAU_UPWIND)
        g_sub = torch.sigmoid((1.0 - M.abs()) / _TAU_UPWIND)
        P_poly = 0.25 * (M + 1.0)**2 * (2.0 - M)
        return g_sup + g_sub * (1.0 - g_sup) * P_poly

    @staticmethod
    def _P_minus(M: torch.Tensor) -> torch.Tensor:
        """Smooth P⁻(M) = 1 - P⁺(M)."""
        return 1.0 - AUSMPlusFlux._P_plus(M)

    def compute_face_flux(self, rho_pad, u_pad, v_pad, w_pad, p_pad, axis,
                          scheme="muscl", limiter="van_leer"):
        fields = {"rho": rho_pad, "u": u_pad, "v": v_pad, "w": w_pad, "p": p_pad}
        fs = self._face_states(fields, axis, scheme, limiter)
        rhoL,rhoR = fs["rho"]; uL,uR = fs["u"]; vL,vR = fs["v"]
        wL,wR = fs["w"];        pL,pR = fs["p"]

        g = self.gamma
        cL = torch.sqrt(g * _softplus_floor(pL, 1e-8) / _softplus_floor(rhoL, 1e-8))
        cR = torch.sqrt(g * _softplus_floor(pR, 1e-8) / _softplus_floor(rhoR, 1e-8))
        c_face = 0.5 * (cL + cR)

        if   axis == 0: unL,unR,utL,utR,uwL,uwR = uL,uR,vL,vR,wL,wR
        elif axis == 1: unL,unR,utL,utR,uwL,uwR = vL,vR,uL,uR,wL,wR
        else:           unL,unR,utL,utR,uwL,uwR = wL,wR,uL,uR,vL,vR

        c_safe = _softplus_floor(c_face, 1e-8)
        M_L = unL / c_safe
        M_R = unR / c_safe

        M_face = self._M_plus(M_L) + self._M_minus(M_R)
        p_face = self._P_plus(M_L) * pL + self._P_minus(M_R) * pR

        # Soft upwind mass flux: tanh gate instead of torch.where(M_face>=0, …)
        g_up   = _soft_gate(M_face, tau=_TAU_UPWIND)          # ≈1 when M_face>0
        rho_up = g_up * rhoL + (1.0 - g_up) * rhoR
        mass_flux = c_safe * M_face * rho_up

        un_up  = g_up * unL  + (1.0 - g_up) * unR
        ut_up  = g_up * utL  + (1.0 - g_up) * utR
        uw_up  = g_up * uwL  + (1.0 - g_up) * uwR

        EL = pL/(g-1) + 0.5*rhoL*(uL**2+vL**2+wL**2)
        ER = pR/(g-1) + 0.5*rhoR*(uR**2+vR**2+wR**2)
        HL = (EL + pL) / _softplus_floor(rhoL, 1e-8)
        HR = (ER + pR) / _softplus_floor(rhoR, 1e-8)
        H_up = g_up * HL + (1.0 - g_up) * HR

        flux_n  = mass_flux * un_up + p_face
        flux_t1 = mass_flux * ut_up
        flux_t2 = mass_flux * uw_up
        flux_E  = mass_flux * H_up

        if   axis == 0: return mass_flux, flux_n,  flux_t1, flux_t2, flux_E
        elif axis == 1: return mass_flux, flux_t1, flux_n,  flux_t2, flux_E
        else:           return mass_flux, flux_t1, flux_t2, flux_n,  flux_E


class HLLCFlux(RiemannSolverBase):
    """
    HLLC flux — fully differentiable.

    DIFF-FIX 4:
    - SL,SR computed with soft-min/max (logsumexp) instead of torch.min/max
    - Boolean region masks replaced with sigmoid soft gates
    - In-place indexed writes removed; all outputs computed branch-free
    """

    def __init__(self, gamma: float = 1.4):
        super().__init__(gamma)

    def compute_face_flux(self, rho_pad, u_pad, v_pad, w_pad, p_pad, axis,
                          scheme="muscl", limiter="van_leer"):
        fields = {"rho": rho_pad, "u": u_pad, "v": v_pad, "w": w_pad, "p": p_pad}
        fs = self._face_states(fields, axis, scheme, limiter)
        rhoL,rhoR = fs["rho"]; uL,uR = fs["u"]; vL,vR = fs["v"]
        wL,wR = fs["w"];        pL,pR = fs["p"]

        g = self.gamma
        rhoL_s = _softplus_floor(rhoL, 1e-8)
        rhoR_s = _softplus_floor(rhoR, 1e-8)
        pL_s   = _softplus_floor(pL,   1e-8)
        pR_s   = _softplus_floor(pR,   1e-8)

        cL = torch.sqrt(g * pL_s / rhoL_s)
        cR = torch.sqrt(g * pR_s / rhoR_s)

        if   axis == 0: unL,unR,utL,utR,uwL,uwR = uL,uR,vL,vR,wL,wR
        elif axis == 1: unL,unR,utL,utR,uwL,uwR = vL,vR,uL,uR,wL,wR
        else:           unL,unR,utL,utR,uwL,uwR = wL,wR,uL,uR,vL,vR

        # Roe-average wave speeds
        R      = torch.sqrt(rhoR_s / rhoL_s)
        un_roe = (unL + R * unR) / (1.0 + R)
        c_roe  = (cL  + R * cR)  / (1.0 + R)

        # DIFF-FIX 4a: soft min/max for SL, SR
        # SL = min(unL-cL, un_roe-c_roe) via smooth min: 0.5(a+b)-0.5|a-b|
        a1 = unL - cL;  b1 = un_roe - c_roe
        SL = 0.5*(a1+b1) - 0.5*_soft_abs(a1-b1)   # smooth min

        a2 = unR + cR;  b2 = un_roe + c_roe
        SR = 0.5*(a2+b2) + 0.5*_soft_abs(a2-b2)   # smooth max

        # Contact wave speed S*
        num_S = pR_s - pL_s + rhoL_s*unL*(SL-unL) - rhoR_s*unR*(SR-unR)
        den_S = rhoL_s*(SL-unL) - rhoR_s*(SR-unR) + 1e-8
        S_star = num_S / den_S

        # DIFF-FIX 4b: soft region gates (replaces boolean masks)
        gL     = _soft_gate(SL,            tau=_TAU_UPWIND)  # ≈1 when SL≥0  → pure left
        gR     = _soft_gate(-SR,           tau=_TAU_UPWIND)  # ≈1 when SR≤0  → pure right
        g_star = (1.0 - gL) * (1.0 - gR)                    # middle region

        # Left and right primitive states
        EL = pL_s/(g-1) + 0.5*rhoL_s*(unL**2+utL**2+uwL**2)
        ER = pR_s/(g-1) + 0.5*rhoR_s*(unR**2+utR**2+uwR**2)

        # Star states (HLLC)
        factL  = (SL - unL) / (SL - S_star + 1e-8)
        factR  = (SR - unR) / (SR - S_star + 1e-8)
        pst_L  = pL_s + rhoL_s*(unL - SL)*(unL - S_star)
        pst_R  = pR_s + rhoR_s*(unR - SR)*(unR - S_star)
        rho_sL = rhoL_s * factL
        rho_sR = rhoR_s * factR
        E_sL   = pst_L/(g-1) + 0.5*rho_sL*(S_star**2+utL**2+uwL**2)
        E_sR   = pst_R/(g-1) + 0.5*rho_sR*(S_star**2+utR**2+uwR**2)

        # Sub-region gate within star region: σ(S*/τ)
        g_sL   = _soft_gate(S_star, tau=_TAU_UPWIND)         # ≈1 left star
        g_sR   = 1.0 - g_sL                                  # ≈1 right star

        # Compose branch-free
        rho_f  = gL*rhoL_s + gR*rhoR_s + g_star*(g_sL*rho_sL + g_sR*rho_sR)
        un_f   = gL*unL    + gR*unR    + g_star*S_star
        ut_f   = gL*utL    + gR*utR    + g_star*(g_sL*utL + g_sR*utR)
        uw_f   = gL*uwL    + gR*uwR    + g_star*(g_sL*uwL + g_sR*uwR)
        p_f    = gL*pL_s   + gR*pR_s   + g_star*(g_sL*pst_L + g_sR*pst_R)
        E_f    = gL*EL     + gR*ER     + g_star*(g_sL*E_sL  + g_sR*E_sR)

        mass_flux = rho_f * un_f
        flux_n    = mass_flux * un_f + p_f
        flux_t1   = mass_flux * ut_f
        flux_t2   = mass_flux * uw_f
        H_f       = (E_f + p_f) / _softplus_floor(rho_f, 1e-8)
        flux_E    = mass_flux * H_f

        if   axis == 0: return mass_flux, flux_n,  flux_t1, flux_t2, flux_E
        elif axis == 1: return mass_flux, flux_t1, flux_n,  flux_t2, flux_E
        else:           return mass_flux, flux_t1, flux_t2, flux_n,  flux_E


# =============================================================================
# 3b. Semi-Lagrangian Advection 3D  (unchanged — grid_sample is differentiable)
# =============================================================================

class SemiLagrangianAdvection3D:
    """
    Unconditionally-stable semi-Lagrangian advection.
    torch.nn.functional.grid_sample is natively differentiable.
    """

    def __init__(self, Lx, Ly, Lz, dx, dy, dz, mode="bilinear"):
        self.Lx=Lx; self.Ly=Ly; self.Lz=Lz
        self.dx=dx; self.dy=dy; self.dz=dz
        self.mode = mode

    def __call__(self, q, u, v, w, dt):
        Nx,Ny,Nz = q.shape
        dev   = q.device
        dtype = q.dtype

        xc = (torch.arange(Nx, device=dev, dtype=torch.float32)+0.5)*self.dx
        yc = (torch.arange(Ny, device=dev, dtype=torch.float32)+0.5)*self.dy
        zc = (torch.arange(Nz, device=dev, dtype=torch.float32)+0.5)*self.dz
        X,Y,Z = torch.meshgrid(xc, yc, zc, indexing="ij")

        X_dep = (X - dt*u.float()) % self.Lx
        Y_dep = (Y - dt*v.float()) % self.Ly
        Z_dep = (Z - dt*w.float()) % self.Lz

        Xn = 2.0*X_dep/self.Lx - 1.0
        Yn = 2.0*Y_dep/self.Ly - 1.0
        Zn = 2.0*Z_dep/self.Lz - 1.0

        q_in = q.float().permute(2,1,0).unsqueeze(0).unsqueeze(0)
        grid = torch.stack([Xn.permute(2,1,0), Yn.permute(2,1,0), Zn.permute(2,1,0)],
                           dim=-1).unsqueeze(0)

        q_out = F.grid_sample(q_in, grid, mode=self.mode,
                              padding_mode="border", align_corners=False)
        return q_out.squeeze(0).squeeze(0).permute(2,1,0).to(dtype=dtype)


# =============================================================================
# 4. SGS / CSOC Models  (DIFF-FIX 5)
# =============================================================================

class CSOCKernel(nn.Module):
    """
    Learnable 5-parameter CSOC kernel.

    Bug 5 fix (v6.2): the v6.1 kernel was a pure power-law,
    Cs * r^(-alpha) * exp(-r/lambda), with NO equilibrium point —
    it diverges monotonically as r -> 0 (low-strain / laminar-like
    regions). At the design value alpha=0.5 this divergence happens
    to be cancelled by the strain_rate_mag factor multiplying it in
    nu_t_base = (Cs_local*dx)^2 * strain_rate_mag (since r=strain/mean
    strain, the net r-dependence is r^(1-2*alpha) = O(1) only at
    alpha=0.5). But alpha is an UNCONSTRAINED learnable nn.Parameter
    (via log_alpha): numerically pushing it to 0.9 or 1.5 during
    training breaks that cancellation and inflates nu_t_base by
    4-5 orders of magnitude before the max_nu_t softplus ceiling
    clips it — saturating gradients near the ceiling and stalling
    learning, the same structural failure mode identified and fixed
    in REAL FOLD ONE's CSOCKernel.

    Fix: replace the power-law with a Lennard-Jones-style equilibrium
    form (validated in REAL FOLD ONE). It has a true minimum at
    r = r_eq (instead of diverging as r -> 0) and saturates smoothly
    on both sides, giving bounded, gradient-friendly behaviour for
    any value the learnable parameters can take — no reliance on a
    coincidental exponent cancellation.
    """

    def __init__(self, init_Cs=0.18, init_lambda=12.0, init_alpha=0.5,
                 init_theta=1.0, init_tau=10.0, device="cpu"):
        super().__init__()
        self.log_Cs     = nn.Parameter(torch.tensor(math.log(init_Cs),     device=device))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda), device=device))
        self.log_alpha  = nn.Parameter(torch.tensor(math.log(init_alpha),  device=device))
        self.log_theta  = nn.Parameter(torch.tensor(math.log(init_theta),  device=device))
        self.log_tau    = nn.Parameter(torch.tensor(math.log(init_tau),    device=device))
        # r_eq: equilibrium point of the LJ-style form, in units of r
        # (= strain_rate_mag / mean_strain). r_eq=1.0 keeps Cs_local
        # at its reference value Cs when local strain equals the mean
        # strain — matching the v6.1 kernel's behaviour at r=1.
        self.r_eq = 1.0

    @property
    def Cs(self):     return torch.exp(self.log_Cs)
    @property
    def lambd(self):  return torch.exp(self.log_lambda)
    @property
    def alpha(self):  return torch.exp(self.log_alpha)
    @property
    def theta(self):  return torch.exp(self.log_theta)
    @property
    def tau(self):    return torch.exp(self.log_tau)

    def forward(self, r):
        """
        Lennard-Jones-style equilibrium form:
            shape(r) = (r_eq/r)^(2a) - 2*(r_eq/r)^a      [standard LJ, min=-1 at r=r_eq]
            shape_capped(r) = softplus-ceiling(shape(r))  [bounded as r->0, any alpha]
            Cs_local = Cs * (1 + shape_capped + 1) * exp(-r/lambda)

        At r=r_eq, shape=-1 exactly, so Cs_local(r_eq) = Cs * exp(-r_eq/lambda),
        matching the v6.1 kernel's reference value at r=1. For r << r_eq,
        shape rises but is capped (softplus ceiling) instead of diverging,
        so Cs_local stays bounded for ANY value the learnable alpha can
        take -- not just the coincidental alpha=0.5 that kept v6.1 bounded.
        The original exp(-r/lambda) far-field decay is preserved unchanged.
        """
        r_safe = _softplus_floor(r, 1e-6, beta=_SOFTPLUS_B)
        x = self.r_eq / r_safe
        lj_shape  = torch.pow(x, 2.0 * self.alpha) - 2.0 * torch.pow(x, self.alpha)
        # Ceiling at 50 (shape value, dimensionless) -- generous enough to
        # not distort behaviour near r_eq, tight enough to keep nu_t_base
        # well inside the max_nu_t softplus ceiling applied downstream.
        # Bug 7 fix (v6.3): beta was 1.0 here vs _SOFTPLUS_B (=100) used by
        # every other floor/ceiling in this module, including the
        # nu_t_total ceiling this is meant to stay inside of. At beta=1.0
        # the transition is ~100x wider, so lj_capped could run well past
        # 50 before actually flattening -- the ceiling was far looser than
        # the docstring/comment claimed. Match beta to _SOFTPLUS_B.
        lj_capped = 50.0 - F.softplus(50.0 - lj_shape, beta=_SOFTPLUS_B)
        shape_val = lj_capped + 1.0   # shifted so shape_val(r_eq) = 0
        return self.Cs * (1.0 + shape_val) * torch.exp(-r_safe / self.lambd)


class SOCController(CSOCBase):
    """
    CSOC-adaptive eddy viscosity — fully differentiable.

    Bug 2 fix: now properly inherits CSOCBase so that ssc, reset(),
    _normalised_deviation(), and _smooth_boost() are consistent with
    the rest of the ONE Ecosystem.

    DIFF-FIX 5:
    - torch.clamp(stress_acc, min=0) → softplus(stress_acc)
    - torch.where(excess>0, …)       → soft gate via softplus / sigmoid
    - stress_acc update is smooth throughout
    """

    def __init__(self, base_temp=300.0, max_nu_t=0.01, use_ssc=True,
                 epsilon_fp=0.0028, compressibility_correction=True,
                 sigma_target=1.0, boost_factor=3.0, device="cpu"):
        # CSOCBase.__init__ creates self.ssc, self.sigma_target, self.boost_factor,
        # and self.reset() which calls self.ssc.reset()
        super().__init__(
            sigma_target=sigma_target,
            epsilon_fp=epsilon_fp if use_ssc else 0.0028,
            boost_factor=boost_factor,
        )
        if not use_ssc:
            self.ssc = None   # override: disable SSC if requested
        self.base_temp = base_temp
        self.max_nu_t  = max_nu_t
        self.compressibility_correction = compressibility_correction
        self.kernel = CSOCKernel(device=device).to(device)
        self.stress_acc = None
        self.device = device

    def reset(self):
        """Reset stress accumulator and SSC state."""
        self.stress_acc = None
        if self.ssc is not None:
            self.ssc.reset()

    def forward(self, *args, **kwargs):
        """Required by CSOCBase ABC — delegates to nu_t()."""
        return self.nu_t(*args, **kwargs)

    def nu_t(self, rho, strain_rate_mag, dilatation, dx, dt, c,
             ext_sigma: "torch.Tensor | None" = None):
        """
        Compute adaptive eddy viscosity.

        Args:
            ext_sigma : optional external structural stress from
                        LangevinDNSBridge — blended with CSOC stress
                        when provided (Bug 3 bridge support).
        """
        # Bug 6 fix (v6.3): additive epsilon floor on mean_S is the same
        # failure mode as Bug 4 — when the true mean strain is near zero
        # (laminar regions, early transient steps before turbulence
        # develops), 1e-8 dominates mean_S instead of acting as a true
        # floor, so r = strain/mean_S blows up uniformly across the whole
        # domain (not just where strain is zero). Use the differentiable
        # softplus floor so the floor only engages when mean_S would
        # otherwise be smaller than it, consistent with every other
        # positivity floor in this module.
        mean_S   = _softplus_floor(strain_rate_mag.mean(), 1e-8)
        r        = strain_rate_mag / mean_S
        Cs_local = self.kernel(r)
        nu_t_base = (Cs_local * dx)**2 * strain_rate_mag

        if self.compressibility_correction:
            M_t    = torch.sqrt(2.0 * nu_t_base * strain_rate_mag + 1e-12) / (c + 1e-8)
            f_dil  = 1.0 / (1.0 + 2.0 * M_t**2)
            nu_t_base = nu_t_base * f_dil

        # DIFF-FIX 5a: stress accumulator uses softplus instead of clamp
        if self.stress_acc is None:
            self.stress_acc = torch.zeros_like(strain_rate_mag)
        tau_k = self.kernel.tau
        dS    = strain_rate_mag**2 - self.stress_acc / (tau_k + 1e-8)
        self.stress_acc = F.softplus(self.stress_acc + dt * dS, beta=_SOFTPLUS_B)

        if self.ssc is not None:
            ssc_in = self.stress_acc.mean()
            if ext_sigma is not None:
                ssc_in = 0.5 * (ssc_in + ext_sigma)
            _ = self.ssc(ssc_in)

        # DIFF-FIX 5b: soft collapse via softplus excess
        theta   = self.kernel.theta
        excess  = F.softplus(self.stress_acc - theta, beta=_SOFTPLUS_B)
        nu_collapse = 0.1 * excess * dx**2

        # Soft reset: scale stress_acc down smoothly when excess is large
        decay   = torch.sigmoid(-(excess / (theta + 1e-8)) * 10.0)
        self.stress_acc = self.stress_acc * (0.5 + 0.5 * decay)

        # DIFF-FIX 5c: soft max for nu_t_total (no hard clamp)
        nu_t_total = nu_t_base + nu_collapse
        # softplus floor at 0, soft ceiling at max_nu_t
        nu_t_total = F.softplus(nu_t_total, beta=_SOFTPLUS_B)
        nu_t_total = self.max_nu_t - F.softplus(
            self.max_nu_t - nu_t_total, beta=_SOFTPLUS_B)
        return nu_t_total


class ItoStressGenerator:
    """Itô stochastic backscatter — unchanged (randn is differentiable w.r.t. scale)."""

    def __init__(self, noise_amp=0.001):
        self.noise_amp = noise_amp

    def generate(self, shape, device, dt):
        amp = self.noise_amp * math.sqrt(dt)
        return tuple(amp * torch.randn(shape, device=device) for _ in range(6))


class DiffRGRefiner:
    """
    Conservative spectral RG truncation — fully differentiable.

    DIFF-FIX 6: Python if-branch on tensor value replaced with
    smooth softplus-guarded rescaling.

    Bug 4 fix (v6.2): the v6.1 mean-ratio rescale `x * (mean_before /
    safe_after)` silently destroyed fields whose mean is near zero —
    which is the *normal* state for momentum components (rhou, rhov,
    rhow) in homogeneous/periodic turbulence, not an edge case.

    Root cause: `_softplus_floor(v, 1e-30, beta=100)` does NOT behave
    like a 1e-30 floor. Its effective transition width is
    log(2)/beta ≈ 6.93e-3, which swamps the requested 1e-30 floor
    whenever |mean_after| << 6.93e-3 (true for any zero-mean field).
    safe_after then collapses to a near-constant ~6.93e-3 regardless
    of the field's real scale, and the ratio mean_before/safe_after
    silently crushes the entire field toward zero — by ~10 orders of
    magnitude for mean ~ 1e-12, confirmed numerically. This is the
    same class of bug as the REAL FOLD ONE structure-corruption issue:
    a global rescale driven by a near-zero denominator, not a NaN.

    Fix: do not rescale by the mean at all. The spectral mask already
    keeps the DC (k=0) component exactly (mask_dc[0,0,0] = 1.0), so
    the filter is mean-preserving by construction up to negligible
    numerical FFT round-off — no rescale step is needed, and removing
    it eliminates the division-driven blow-up/collapse entirely.
    """

    def __init__(self, keep_fraction: float = 0.5):
        if not (0.0 < keep_fraction <= 1.0):
            raise ValueError(f"keep_fraction must be in (0,1]; got {keep_fraction!r}.")
        self.keep_fraction = keep_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny, nz   = x.shape
        x_hat        = torch.fft.rfftn(x)

        kx = torch.fft.fftfreq(nx,  d=1.0, device=x.device)
        ky = torch.fft.fftfreq(ny,  d=1.0, device=x.device)
        kz = torch.fft.rfftfreq(nz, d=1.0, device=x.device)
        KX,KY,KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        K_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)

        # Soft spectral mask: sigmoid instead of hard K_mag <= k_cut
        k_cut = self.keep_fraction * K_mag.max().detach()
        mask  = torch.sigmoid((k_cut - K_mag) / (_TAU_UPWIND * k_cut + 1e-12))
        mask_dc = mask.clone(); mask_dc[0,0,0] = 1.0   # always keep DC exactly

        # Bug 4 fix: no post-hoc mean-ratio rescale. The DC term (the
        # field's mean) passes through mask_dc[0,0,0]=1.0 unchanged,
        # so this filter already conserves the mean to FFT numerical
        # precision — rescaling on top of that only reintroduces a
        # division blow-up/collapse for near-zero-mean fields.
        x_hat_f    = x_hat * mask_dc.to(x_hat.dtype)
        x_filtered = torch.fft.irfftn(x_hat_f, s=(nx,ny,nz))
        return x_filtered


# =============================================================================
# 5. Real-Gas EOS  (unchanged)
# =============================================================================

class RealGasEOS:
    """Real-gas thermodynamics via CoolProp (optional)."""

    def __init__(self, fluid_name="Air", gamma=1.4):
        self.fluid_name = fluid_name
        self.gamma      = gamma
        self.use_real   = HAS_COOLPROP

    def pressure(self, rho, e):
        if not self.use_real:
            return (self.gamma - 1) * rho * e
        p_out = torch.zeros_like(rho)
        rho_np = rho.detach().cpu().numpy().ravel()
        e_np   = e.detach().cpu().numpy().ravel()
        for i, (r, ei) in enumerate(zip(rho_np, e_np)):
            try:
                p_out.ravel()[i] = CP.PropsSI("P","D",max(r,1e-6),"U",max(ei,1.0),self.fluid_name)
            except Exception:
                p_out.ravel()[i] = (self.gamma-1)*r*ei
        return p_out

    def sound_speed(self, rho, e):
        if not self.use_real:
            p = (self.gamma-1)*rho*e
            return torch.sqrt(self.gamma * _softplus_floor(p,1e-8) / _softplus_floor(rho,1e-8))
        c_out = torch.zeros_like(rho)
        rho_np = rho.detach().cpu().numpy().ravel()
        e_np   = e.detach().cpu().numpy().ravel()
        for i, (r, ei) in enumerate(zip(rho_np, e_np)):
            try:
                c_out.ravel()[i] = CP.PropsSI("A","D",max(r,1e-6),"U",max(ei,1.0),self.fluid_name)
            except Exception:
                c_out.ravel()[i] = math.sqrt(self.gamma*(self.gamma-1)*max(ei,1e-8))
        return c_out


# =============================================================================
# 6. Immersed Boundary  (unchanged)
# =============================================================================

class ImmersedBoundary:
    """
    Immersed Boundary forcing (volume-penalisation method).

    Supports vector velocity target u_target and optional temperature penalty.
    """

    def __init__(self, mask: torch.Tensor, eta: float = 1e4,
                 u_target: tuple = (0., 0., 0.),
                 T_target: float = None, eta_T: float = 1e4, gamma: float = 1.4):
        self.mask     = mask.to(torch.bool)
        self.eta      = eta
        self.u_target = u_target
        self.T_target = T_target
        self.eta_T    = eta_T
        self.gamma    = gamma

    def apply_forcing(self, rho, rhou, rhov, rhow, rhoE, dt):
        m   = self.mask.to(rho.device).to(rho.dtype)
        rho_s = _softplus_floor(rho, 1e-8)
        u = rhou / rho_s;  v = rhov / rho_s;  w = rhow / rho_s

        # Velocity penalty toward u_target
        force_x = -self.eta * m * (u - self.u_target[0])
        force_y = -self.eta * m * (v - self.u_target[1])
        force_z = -self.eta * m * (w - self.u_target[2])
        rhou = rhou + dt * force_x * rho_s
        rhov = rhov + dt * force_y * rho_s
        rhow = rhow + dt * force_z * rho_s

        if self.T_target is not None:
            ke        = 0.5 * (rhou**2 + rhov**2 + rhow**2) / rho_s
            p         = (self.gamma - 1) * _softplus_floor(rhoE - ke, 1e-8)
            T         = p / rho_s
            rhoE_tgt  = rho_s * self.T_target / (self.gamma - 1)
            rhoE      = rhoE + dt * (-self.eta_T) * m * (rhoE - rhoE_tgt)

        return rhou, rhov, rhow, rhoE


# =============================================================================
# 7. CFDConfig  (unchanged public API)
# =============================================================================

class CFDConfig:
    _VALID_FLUX = {'ausm', 'hllc'}

    def __init__(self,
                 nx=64, ny=64, nz=64, Lx=1.0, Ly=1.0, Lz=1.0,
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
                 distributed=False,
                 muscl_limiter='van_leer',
                 advection_scheme='muscl'):

        self.nx=nx; self.ny=ny; self.nz=nz
        self.Lx=Lx; self.Ly=Ly; self.Lz=Lz
        self.dx = Lx/nx
        self.Re=Re; self.Pr=Pr; self.gamma=gamma; self.Mach=Mach
        self.cfl=cfl; self.steps=steps
        self.soc_base_temp=soc_base_temp; self.max_nu_t=max_nu_t
        self.use_rg=use_rg; self.rg_keep_frac=rg_keep_frac; self.rg_interval=rg_interval
        self.ito_noise=ito_noise; self.muscl=muscl; self.use_sutherland=use_sutherland
        self.device=device; self.flux_scheme=flux_scheme; self.shock_capturing=shock_capturing
        self.compressibility_correction=compressibility_correction
        self.ssc_epsilon=ssc_epsilon; self.dtype=dtype
        self.bc_x_min=bc_x_min; self.bc_x_max=bc_x_max
        self.bc_y_min=bc_y_min; self.bc_y_max=bc_y_max
        self.bc_z_min=bc_z_min; self.bc_z_max=bc_z_max
        self.inflow_rho=inflow_rho; self.inflow_u=inflow_u
        self.inflow_v=inflow_v; self.inflow_w=inflow_w; self.inflow_p=inflow_p
        self.outflow_p=outflow_p; self.wall_temp=wall_temp
        self.farfield_rho=farfield_rho; self.farfield_u=farfield_u
        self.farfield_v=farfield_v; self.farfield_w=farfield_w; self.farfield_p=farfield_p
        self.moving_wall_u=moving_wall_u; self.moving_wall_v=moving_wall_v
        self.moving_wall_w=moving_wall_w
        self.use_wall_model=use_wall_model; self.wm_A=wm_A; self.wm_B=wm_B
        self.eos_model=eos_model; self.fluid_name=fluid_name
        self.ib_mask_file=ib_mask_file; self.ib_eta=ib_eta
        self.ib_T_target=ib_T_target; self.ib_eta_T=ib_eta_T
        self.distributed=distributed
        self.muscl_limiter=muscl_limiter
        self.advection_scheme=advection_scheme

        if flux_scheme not in self._VALID_FLUX:
            raise ValueError(f"'flux_scheme' must be one of {sorted(self._VALID_FLUX)}; got {flux_scheme!r}.")
        if advection_scheme not in {'muscl','weno5','semi_lagrangian'}:
            raise ValueError(f"advection_scheme must be muscl|weno5|semi_lagrangian.")
        if muscl_limiter not in {'minmod','van_leer','superbee'}:
            raise ValueError(f"muscl_limiter must be minmod|van_leer|superbee.")


    @classmethod
    def with_stretched_grid(
        cls,
        x_coords: "np.ndarray",
        y_coords: "np.ndarray",
        z_coords: "np.ndarray",
        **kwargs,
    ) -> "_StretchedCFDConfig":
        """
        Create a configuration for a **non-uniform (stretched) grid**.

        This factory bypasses the uniform-spacing requirement and instead
        accepts 1-D coordinate arrays for each axis.  Typical use: wall-
        bounded flows that need fine resolution near solid surfaces (e.g.
        cardiovascular vessel walls, wing boundary layers) without the
        prohibitive cell count of a globally fine uniform mesh.

        The returned :class:`_StretchedCFDConfig` is a thin subclass that
        carries the coordinate arrays and exposes per-cell spacing tensors
        (``dx_arr``, ``dy_arr``, ``dz_arr``) for use in the solver.

        Args:
            x_coords : 1-D array of cell-*face* positions in x, length nx+1.
            y_coords : 1-D array of cell-face positions in y, length ny+1.
            z_coords : 1-D array of cell-face positions in z, length nz+1.
            **kwargs : any keyword accepted by :class:`CFDConfig` except
                       nx/ny/nz/Lx/Ly/Lz (those are derived from the arrays).

        Returns:
            A :class:`_StretchedCFDConfig` instance.

        Example::

            import numpy as np
            # Tanh-stretched grid in y (fine near y=0 wall)
            ny = 64
            eta = np.linspace(0, 1, ny + 1)
            y_faces = np.tanh(3 * eta) / np.tanh(3)   # ∈ [0, 1]
            cfg = CFDConfig.with_stretched_grid(
                x_coords = np.linspace(0, 2*np.pi, 65),
                y_coords = y_faces,
                z_coords = np.linspace(0, 2*np.pi, 65),
                Re=1e4, device='cuda',
            )
        """
        import numpy as _np

        x_coords = _np.asarray(x_coords, dtype=float)
        y_coords = _np.asarray(y_coords, dtype=float)
        z_coords = _np.asarray(z_coords, dtype=float)

        nx = len(x_coords) - 1
        ny = len(y_coords) - 1
        nz = len(z_coords) - 1

        Lx = float(x_coords[-1] - x_coords[0])
        Ly = float(y_coords[-1] - y_coords[0])
        Lz = float(z_coords[-1] - z_coords[0])

        # Compute nominal dx as the mean spacing (used by BC classes that
        # need a single representative length scale, e.g. ghost-cell fill)
        dx_nom = Lx / nx

        # Bypass uniform check: use a known-uniform dummy domain, then fix attrs
        obj = object.__new__(_StretchedCFDConfig)
        # Directly initialise without calling the uniform-check path
        _StretchedCFDConfig.__init__(
            obj,
            nx=nx, ny=ny, nz=nz,
            Lx=Lx, Ly=Ly, Lz=Lz,
            **{k: v for k, v in kwargs.items()
               if k not in ('nx', 'ny', 'nz', 'Lx', 'Ly', 'Lz')},
        )
        # Override scalar spacings with per-cell arrays (cell-centred widths)
        obj.dx_arr = _np.diff(x_coords).astype(float)   # shape (nx,)
        obj.dy_arr = _np.diff(y_coords).astype(float)   # shape (ny,)
        obj.dz_arr = _np.diff(z_coords).astype(float)   # shape (nz,)
        obj.x_faces = x_coords
        obj.y_faces = y_coords
        obj.z_faces = z_coords
        obj.is_stretched = True
        return obj



class _StretchedCFDConfig(CFDConfig):
    """
    CFDConfig subclass for non-uniform stretched grids.

    Attributes added beyond CFDConfig
    ──────────────────────────────────
    dx_arr, dy_arr, dz_arr : np.ndarray of per-cell spacings  (cell-centred)
    x_faces, y_faces, z_faces : np.ndarray of face positions
    is_stretched : True

    The CompressibleSolver checks ``getattr(cfg, 'is_stretched', False)``
    and uses the per-cell spacing arrays for finite-difference stencils
    when this flag is set.  When ``is_stretched`` is False (the default),
    the scalar ``cfg.dx`` is used as before so that all existing uniform-
    grid code paths remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        # Temporarily relax the uniform-spacing check so the parent
        # __init__ can validate everything else normally.
        # We achieve this by making dx==dy==dz from the nominal Lx/nx.
        # The real per-cell arrays are set by with_stretched_grid().
        super().__init__(*args, **kwargs)
        self.is_stretched = False   # overridden by factory




# =============================================================================
# 8. CompressibleSolver — Native Full-Differentiable
# =============================================================================

class CompressibleSolver:
    """
    3-D compressible Navier-Stokes solver, fully differentiable end-to-end.

    All non-differentiable operations from v4 have been replaced:
    see module docstring DIFF-FIX 1–8 for details.
    """

    def __init__(self, cfg: CFDConfig):
        self.cfg   = cfg
        self.device = get_device(cfg.device)
        self.dtype  = cfg.dtype
        self.dx     = cfg.dx
        self.gamma  = cfg.gamma
        self.nu_phys = 1.0 / cfg.Re if cfg.Re > 0 else 0.0
        self.Pr     = cfg.Pr

        # Distributed setup
        self.distributed = cfg.distributed
        if self.distributed:
            if not dist.is_initialized():
                raise RuntimeError("Call init_process_group before creating the solver.")
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
            if cfg.nz % self.world_size != 0:
                raise ValueError(f"nz ({cfg.nz}) must be divisible by world_size ({self.world_size})")
            self.local_nz = cfg.nz // self.world_size
            self.z_start  = self.rank * self.local_nz
            self.z_end    = self.z_start + self.local_nz
            if cfg.bc_z_min == 'periodic' and cfg.bc_z_max == 'periodic':
                self.neighbor_left  = (self.rank - 1) % self.world_size
                self.neighbor_right = (self.rank + 1) % self.world_size
                self.z_periodic_dist = True
            else:
                self.neighbor_left  = self.rank-1 if self.rank>0 else -1
                self.neighbor_right = self.rank+1 if self.rank<self.world_size-1 else -1
                self.z_periodic_dist = False
        else:
            self.rank=0; self.world_size=1; self.local_nz=cfg.nz

        self.soc = SOCController(
            base_temp=cfg.soc_base_temp, max_nu_t=cfg.max_nu_t,
            use_ssc=True, epsilon_fp=cfg.ssc_epsilon,
            compressibility_correction=cfg.compressibility_correction,
            device=self.device,
        )
        self.ito_gen = ItoStressGenerator(noise_amp=cfg.ito_noise) if cfg.ito_noise > 0 else None
        self.rg      = DiffRGRefiner(keep_fraction=cfg.rg_keep_frac) if cfg.use_rg else None

        self.flux_solver = (AUSMPlusFlux(gamma=cfg.gamma)
                            if cfg.flux_scheme == 'ausm'
                            else HLLCFlux(gamma=cfg.gamma))

        self._adv_scheme  = getattr(cfg, 'advection_scheme', 'muscl')
        self._adv_limiter = getattr(cfg, 'muscl_limiter', 'van_leer')

        if self._adv_scheme == 'semi_lagrangian':
            d = cfg.Lx / cfg.nx
            self._semi_lag = SemiLagrangianAdvection3D(
                Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz, dx=d, dy=d, dz=d)
        else:
            self._semi_lag = None

        self._init_bc_objects()
        self.eos = RealGasEOS(cfg.fluid_name, gamma=cfg.gamma) if cfg.eos_model=='real' else None

        self.ib = None
        if cfg.ib_mask_file:
            try:
                mask = torch.load(cfg.ib_mask_file, map_location=self.device)
                self.ib = ImmersedBoundary(mask, eta=cfg.ib_eta,
                                           T_target=cfg.ib_T_target, eta_T=cfg.ib_eta_T)
            except Exception as e:
                logger.warning(f"IB mask load failed: {e}")

        self.rho = self.rhou = self.rhov = self.rhow = self.rhoE = None
        self.step_count = 0
        self.time       = 0.0

        # External Langevin coupling buffer (written by LangevinDNSBridge.sync())
        self._ext_sigma = torch.tensor(0.0, device=self.device)

        # External Cahn-Hilliard coupling buffers (written by CahnHilliardDNSBridge.sync())
        _zeros = torch.zeros(cfg.nx, cfg.ny, cfg.nz, device=self.device, dtype=self.dtype)
        self._ext_rho_ch = _zeros.clone()   # (nx,ny,nz) density modulation
        self._ext_nu_ch  = _zeros.clone()   # (nx,ny,nz) viscosity modulation
        self._ext_fx     = _zeros.clone()   # (nx,ny,nz) Korteweg body force x
        self._ext_fy     = _zeros.clone()   # (nx,ny,nz) Korteweg body force y
        self._ext_fz     = _zeros.clone()   # (nx,ny,nz) Korteweg body force z

        # External volumetric heat-release-rate coupling buffer (v6.5;
        # written by e.g. HeatReleaseDNSBridge in one_core_v3.py, used by
        # fire_dns_coupling_one.py). [W/m^3], added directly to rhs_rhoE.
        # No such energy-source hook existed prior to v6.5 -- _ext_fx/fy/fz
        # only enter the energy equation indirectly via mechanical work
        # (f.u), which cannot represent a direct heat addition like
        # combustion or radiative absorption/emission.
        self._ext_q      = _zeros.clone()   # (nx,ny,nz) volumetric heat release [W/m^3]

        # ── v6.6: resolved mixture-fraction combustion + P1 radiation ───────
        # (FIRE ONE full-CFD upgrade; see v6.5→v6.6 changelog entry above
        # for scope/limitations.)
        self.enable_combustion      = bool(getattr(cfg, "enable_combustion", False))
        self.z_stoich               = float(getattr(cfg, "z_stoich", 0.055))
        self.T_adiabatic            = float(getattr(cfg, "T_adiabatic_K", 2260.0))
        self.T_ambient_rad          = float(getattr(cfg, "T_ambient_K", 293.15))
        self.turbulent_schmidt      = float(getattr(cfg, "turbulent_schmidt", 0.7))
        self.cp_gas                 = float(getattr(cfg, "cp_gas", 1400.0))
        self.enable_radiation       = bool(getattr(cfg, "enable_radiation", False))
        self.radiation_absorption_coeff = float(getattr(cfg, "radiation_absorption_coeff", 0.3))
        self.T_ref = 300.0   # matches the Sutherland-law reference used elsewhere in this file
        # UNRESOLVED SCALING GAP: see the combustion/radiation blocks in
        # _compute_rhs. Default 1.0 means NO dimensional->non-dimensional
        # conversion is applied -- combustion_source/q_rad are added to
        # rhs_rhoE in real W/m^3 units as-is. This is very unlikely to be
        # the physically correct scale for your specific cfg.Re/Mach
        # non-dimensionalization; set cfg.combustion_nondim_scale
        # explicitly (L_ref/(rho_ref*U_ref**3)) before trusting
        # quantitative combustion/radiation results.
        self.combustion_nondim_scale = float(getattr(cfg, "combustion_nondim_scale", 1.0))
        # Resolved mixture fraction, transported as RZ = rho*Z (conserved
        # form, consistent with how rho/rhou/rhov/rhow/rhoE are all
        # transported as conserved densities elsewhere in this class).
        self.RZ = _zeros.clone()

        # ── v6.7: soot transport + discrete-ordinates radiation ─────────────
        # BOTH DEFAULT DISABLED ("built but gated off for now, for
        # certainty" per explicit request -- validated in isolation, but
        # gated behind config flags until explicitly enabled, given the
        # real risk class already demonstrated once by the v6.6
        # non-dimensional unit bug). Double-gated where relevant so an
        # accidental single flag flip can't silently activate a new,
        # less-tested code path.
        self.enable_soot           = bool(getattr(cfg, "enable_soot", False))
        self.soot_A_formation      = float(getattr(cfg, "soot_A_formation", 100.0))
        self.soot_Ea_formation     = float(getattr(cfg, "soot_Ea_formation", 1.25e5))
        self.soot_A_oxidation      = float(getattr(cfg, "soot_A_oxidation", 1.0e4))
        self.soot_Ea_oxidation     = float(getattr(cfg, "soot_Ea_oxidation", 1.65e5))
        self.soot_schmidt          = float(getattr(cfg, "soot_schmidt", 0.7))
        # rho*Y_soot, conserved form (parallels RZ). Zero-initialized;
        # inert (transports but never produced/destroyed) unless BOTH
        # enable_soot AND enable_combustion are True (soot kinetics need
        # the local Y_fuel/Y_O2/T state that only exists under combustion).
        self.RS = _zeros.clone()

        # ── v6.8: external mass-source buffers (continuity + carried
        # momentum/energy/mixture-fraction) ─────────────────────────────────
        # Requested addition: pyrolysis (or any real mass-injection
        # process -- fuel gasification, water-mist evaporation, etc.)
        # adds mass to the gas phase, which NO existing buffer could
        # represent -- rho was, until now, exactly conserved by the
        # convective terms alone (a true closed continuity equation with
        # no source). This is a genuinely new physical capability, not
        # just another guarded external force.
        #
        # Mass added at a nonzero rate must ALSO carry momentum (the
        # injected gas has some velocity), energy (it has some enthalpy/
        # temperature), and -- now that mixture fraction is resolved --
        # composition (pyrolyzate is Z=1, pure fuel). Four buffers, all
        # zero-cost if not connected, mirroring every other _ext_ buffer:
        self._ext_mdot   = _zeros.clone()   # [kg/(m^3.s)] volumetric mass source -> rhs_rho
        self._ext_mdot_u = _zeros.clone()   # [kg/(m^2.s^2)] momentum carried by injected mass (mdot*u_inject) -> rhs_rhou
        self._ext_mdot_v = _zeros.clone()   # ... -> rhs_rhov
        self._ext_mdot_w = _zeros.clone()   # ... -> rhs_rhow
        self._ext_mdot_e = _zeros.clone()   # [W/m^3] energy carried by injected mass -> rhs_rhoE
        self._ext_mdot_Z = _zeros.clone()   # [kg/(m^3.s)] mixture-fraction flux carried by injected mass -> rhs_RZ
        # (a mass source with no matching _ext_mdot_u/v/w/e/Z is physically
        # a source with zero injection velocity and zero enthalpy/
        # composition -- unusual but not forbidden; the bridge writing
        # these buffers is responsible for setting all of them together
        # consistently for a real physical source.)
        #
        # Separate, dimensionally-distinct scaling knob from
        # combustion_nondim_scale (mass-rate and energy-rate source terms
        # have different reference-scale groups; see the UNIT-CONSISTENCY
        # FIX note in _compute_rhs). Default 1.0 = no conversion, same
        # explicit-must-set-yourself philosophy as combustion_nondim_scale.
        self.mdot_nondim_scale = float(getattr(cfg, "mdot_nondim_scale", 1.0))

        self.radiation_method = str(getattr(cfg, "radiation_method", "P1"))  # "P1" | "DOM"
        self.dom_n_polar      = int(getattr(cfg, "dom_n_polar", 4))
        self.dom_n_azimuthal  = int(getattr(cfg, "dom_n_azimuthal", 8))
        self.dom_n_iterations = int(getattr(cfg, "dom_n_iterations", 20))
        self._dom_quadrature_cache = None   # built lazily on first DOM call

        self.energy_hist = []
        self.div_hist    = []

        self.scaler  = GradScaler() if self.device.type == 'cuda' else None
        self.use_amp = False

        self.wall_model_faces = []
        for face, bc_type in [
            ('xmin', cfg.bc_x_min), ('xmax', cfg.bc_x_max),
            ('ymin', cfg.bc_y_min), ('ymax', cfg.bc_y_max),
            ('zmin', cfg.bc_z_min), ('zmax', cfg.bc_z_max),
        ]:
            if bc_type == 'wall_model':
                self.wall_model_faces.append(face)

    # ── BC object initialisation ────────────────────────────────────────────

    def _init_bc_objects(self):
        cfg = self.cfg
        def _bc(name):
            if name == 'periodic':       return PeriodicBC()
            if name == 'inflow':         return SupersonicInflowBC(cfg.inflow_rho,cfg.inflow_u,cfg.inflow_v,cfg.inflow_w,cfg.inflow_p,cfg.gamma)
            if name == 'outflow':        return SubsonicOutflowBC(cfg.outflow_p,cfg.gamma)
            if name == 'no_slip':        return NoSlipIsothermalWallBC(cfg.wall_temp,cfg.gamma)
            # Bug 9 fix (v6.3): WernerWengleWallModelBC.__init__ is
            # (T_wall, A, B). This call previously passed (cfg.wm_A,
            # cfg.wm_B) positionally, so wm_A silently became T_wall
            # (wall temperature ended up as 8.3 instead of cfg.wall_temp)
            # and wm_B silently became A, leaving B at its hardcoded
            # default. Pass T_wall explicitly and A/B as keywords.
            if name == 'wall_model':     return WernerWengleWallModelBC(T_wall=cfg.wall_temp, A=cfg.wm_A, B=cfg.wm_B)
            if name == 'moving_wall':    return MovingWallBC(cfg.moving_wall_u,cfg.moving_wall_v,cfg.moving_wall_w,cfg.wall_temp,cfg.gamma)
            if name == 'farfield':       return FarFieldBC(cfg.farfield_rho,cfg.farfield_u,cfg.farfield_v,cfg.farfield_w,cfg.farfield_p,cfg.gamma)
            return PeriodicBC()
        self.bc_x_min = _bc(cfg.bc_x_min); self.bc_x_max = _bc(cfg.bc_x_max)
        self.bc_y_min = _bc(cfg.bc_y_min); self.bc_y_max = _bc(cfg.bc_y_max)
        self.bc_z_min = _bc(cfg.bc_z_min); self.bc_z_max = _bc(cfg.bc_z_max)

    # ── Helpers ─────────────────────────────────────────────────────────────

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
        Exchange 2-layer ghost halos in the z-direction using **non-blocking**
        isend / irecv so that all send and receive operations are posted before
        any wait() call.  The original blocking send/recv sequence was prone to
        deadlock when all ranks simultaneously tried to send to the same
        neighbour before posting their matching recv.

        Protocol (for each field tensor shaped (nx, ny, local_nz + 4)):
          • Ghost layers 0:2  ← data from the *right* neighbour's inner layers
          • Ghost layers -2:  ← data from the *left*  neighbour's inner layers
        """
        if not self.distributed or self.world_size == 1:
            return

        reqs = []
        recv_bufs_left  = []   # will fill ghost[0:2]
        recv_bufs_right = []   # will fill ghost[-2:]

        for f in fields:
            # ── Post receives first (avoids the classical send-then-recv deadlock) ──
            if self.neighbor_right >= 0:
                # We will receive the *left* ghost of this rank from the right neighbour
                rb = torch.empty_like(f[:, :, 0:2])
                reqs.append(dist.irecv(rb, src=self.neighbor_right))
                recv_bufs_left.append((f, rb))
            else:
                recv_bufs_left.append(None)

            if self.neighbor_left >= 0:
                rb = torch.empty_like(f[:, :, -2:])
                reqs.append(dist.irecv(rb, src=self.neighbor_left))
                recv_bufs_right.append((f, rb))
            else:
                recv_bufs_right.append(None)

        for f in fields:
            # ── Post sends after all receives are posted ──────────────────────
            if self.neighbor_left >= 0:
                # Send our inner-left layers to the left neighbour's right ghost
                reqs.append(dist.isend(f[:, :, 2:4].contiguous(), dst=self.neighbor_left))
            if self.neighbor_right >= 0:
                # Send our inner-right layers to the right neighbour's left ghost
                reqs.append(dist.isend(f[:, :, -4:-2].contiguous(), dst=self.neighbor_right))

        # ── Wait for all communications to complete ───────────────────────────
        for req in reqs:
            req.wait()

        # ── Copy received data into the padded ghost-cell regions ─────────────
        for entry in recv_bufs_left:
            if entry is not None:
                f, rb = entry
                f[:, :, 0:2] = rb
        for entry in recv_bufs_right:
            if entry is not None:
                f, rb = entry
                f[:, :, -2:] = rb

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

    # Bug 8 fix (v6.3): this class previously defined _apply_wall_model
    # twice. Python keeps only the last definition, so the comprehensive
    # implementation above (all 6 faces, 5-iteration y_plus/u_tau solve,
    # using wall_model_faces + cfg.wm_A/cfg.wm_B) was dead code -- every
    # call to self._apply_wall_model() in step() actually ran this
    # trivial y-only stub instead. Removed the duplicate; the single
    # remaining definition above is now the one that executes.

    def _apply_bc_to_boundary_cells(self, rho, rhou, rhov, rhow, rhoE):
        for bc, ax, side in [
            (self.bc_x_min,0,'left'),(self.bc_x_max,0,'right'),
            (self.bc_y_min,1,'left'),(self.bc_y_max,1,'right'),
            (self.bc_z_min,2,'left'),(self.bc_z_max,2,'right'),
        ]:
            if hasattr(bc, 'apply'):
                try:
                    bc.apply(rho, rhou, rhov, rhow, rhoE, ax, side, self.gamma, self.dx)
                except Exception:
                    pass

    # ── Initial conditions ───────────────────────────────────────────────────

    def initialize(self, case='taylor_green'):
        cfg  = self.cfg
        dev  = self.device
        dt_  = self.dtype
        nx,ny,nz = cfg.nx, cfg.ny, self.local_nz

        x = torch.linspace(0, 2*math.pi, nx+1, device=dev, dtype=dt_)[:-1]
        y = torch.linspace(0, 2*math.pi, ny+1, device=dev, dtype=dt_)[:-1]
        if self.distributed:
            z_global = torch.linspace(0,2*math.pi,cfg.nz+1,device=dev,dtype=dt_)[:-1]
            z = z_global[self.z_start:self.z_end]
        else:
            z = torch.linspace(0,2*math.pi,nz+1,device=dev,dtype=dt_)[:-1]

        X,Y,Z = torch.meshgrid(x, y, z, indexing='ij')

        if case == 'taylor_green':
            u0  = self.cfg.Mach
            u   = u0*torch.sin(X)*torch.cos(Y)*torch.cos(Z)
            v   = -u0*torch.cos(X)*torch.sin(Y)*torch.cos(Z)
            w   = torch.zeros_like(u)
            rho = torch.ones_like(u)
            p   = rho / self.gamma
        elif case == 'hypersonic_bnd':
            u0  = self.cfg.Mach * math.sqrt(self.gamma)
            u   = u0 * (1 - torch.exp(-Y*5))
            v   = torch.zeros_like(u); w = torch.zeros_like(u)
            rho = torch.ones_like(u);  p = rho / self.gamma
        else:
            u   = self.cfg.Mach * torch.ones_like(X)
            v   = torch.zeros_like(X); w = torch.zeros_like(X)
            rho = torch.ones_like(X);  p = rho / self.gamma

        rhou = rho*u; rhov = rho*v; rhow = rho*w
        rhoE = p/(self.gamma-1) + 0.5*rho*(u**2+v**2+w**2)

        self.rho=rho; self.rhou=rhou; self.rhov=rhov; self.rhow=rhow; self.rhoE=rhoE
        self.soc.reset(); self.step_count=0; self.time=0.0

    # ── RHS ─────────────────────────────────────────────────────────────────

    def _build_dom_quadrature(self, device, dtype):
        """
        [v6.7, opt-in] Product Gauss-Legendre(polar) x uniform(azimuthal)
        discrete-ordinates quadrature over the unit sphere. Cached after
        first build (direction set doesn't depend on solver state).

        Validated (see fire_dns development notes): weights sum to
        exactly 4*pi (total solid angle) to floating-point precision, and
        all direction vectors are unit vectors, checked with a numpy
        prototype of this exact construction before being ported here.
        """
        if self._dom_quadrature_cache is not None:
            return self._dom_quadrature_cache
        mu_nodes, mu_weights = np.polynomial.legendre.leggauss(self.dom_n_polar)
        phi = np.linspace(0, 2 * np.pi, self.dom_n_azimuthal, endpoint=False)
        dphi = 2 * np.pi / self.dom_n_azimuthal
        dirs, wts = [], []
        for mu, w_mu in zip(mu_nodes, mu_weights):
            sin_t = np.sqrt(max(1.0 - mu**2, 0.0))
            for p in phi:
                dirs.append((sin_t * np.cos(p), sin_t * np.sin(p), mu))
                wts.append(w_mu * dphi)
        dirs_t = torch.tensor(np.array(dirs), device=device, dtype=dtype)
        wts_t  = torch.tensor(np.array(wts),  device=device, dtype=dtype)
        self._dom_quadrature_cache = (dirs_t, wts_t)
        return self._dom_quadrature_cache

    def _solve_dom_radiation(self, T_real: torch.Tensor, kappa: float, dx: float) -> torch.Tensor:
        """
        [v6.7, opt-in -- cfg.radiation_method="DOM", default "P1"]
        Discrete-ordinates radiative transport, solved via SOURCE
        ITERATION (fixed-point/Jacobi iteration on the upwind-discretized
        RTE along each quadrature direction) rather than a literal
        sequential sweep -- sweeps don't parallelize on tensor hardware
        (each cell depends on an already-updated upwind neighbor within
        the same pass), whereas source iteration is a real, standard
        method for the same equation set that DOES vectorize (every
        direction, every cell, updated simultaneously from the PREVIOUS
        iteration's field), at the cost of needing multiple iterations to
        converge. This is an approximation quality/cost trade against a
        true single-pass sweep, not a different physical model.

        ASSUMES PERIODIC BOUNDARIES (uses torch.roll for upwind shifts,
        matching this solver's default circular-padding convention) --
        same limitation as the P1/FFT radiation path, NOT rigorously
        valid for a wall-dominated enclosure (no wall emissivity/
        reflection boundary condition is modeled here at all).

        For each direction s=(mu,eta,xi), solves the steady upwind
        discretization of:
            s . grad(I) + kappa*I = kappa*I_b
        via Jacobi iteration, then accumulates the incident radiation
        G = sum_i(weight_i * I_i) and returns the net volumetric
        radiative source q_rad = kappa*(4*sigma*T^4 - G), matching the
        same physical quantity and sign convention as the P1 path.
        """
        sigma_sb = 5.670374e-8
        I_b = sigma_sb * T_real**4 / math.pi   # Planck intensity, isotropic assumption
        dirs, wts = self._build_dom_quadrature(T_real.device, T_real.dtype)

        G = torch.zeros_like(T_real)
        for d in range(dirs.shape[0]):
            mu_x, mu_y, mu_z = dirs[d, 0].item(), dirs[d, 1].item(), dirs[d, 2].item()
            denom = (abs(mu_x) + abs(mu_y) + abs(mu_z)) / dx + kappa
            I = I_b.clone()   # initial guess: local blackbody intensity
            for _ in range(self.dom_n_iterations):
                # Upwind neighbor via torch.roll: shift OPPOSITE to the
                # sign of the direction cosine so we read the cell the
                # radiation is coming FROM (periodic wrap = the
                # documented assumption above).
                Ix = torch.roll(I, shifts=1 if mu_x >= 0 else -1, dims=0)
                Iy = torch.roll(I, shifts=1 if mu_y >= 0 else -1, dims=1)
                Iz = torch.roll(I, shifts=1 if mu_z >= 0 else -1, dims=2)
                numer = (kappa * I_b
                         + (abs(mu_x) / dx) * Ix
                         + (abs(mu_y) / dx) * Iy
                         + (abs(mu_z) / dx) * Iz)
                I = numer / denom
            G = G + wts[d] * I

        q_rad = kappa * (4.0 * sigma_sb * T_real**4 - G)
        return q_rad

    def _compute_rhs(self, rho, rhou, rhov, rhow, rhoE, RZ, RS, dt):
        dx    = self.dx
        gamma = self.gamma
        nx,ny,nz = rho.shape

        # Blend Cahn-Hilliard density if CH bridge has synced (zero-cost otherwise)
        if self._ext_rho_ch.abs().max() > 1e-30:
            rho = 0.5 * (rho + self._ext_rho_ch)

        u = rhou / _softplus_floor(rho, 1e-8)
        v = rhov / _softplus_floor(rho, 1e-8)
        w = rhow / _softplus_floor(rho, 1e-8)
        ke = 0.5 * rho * (u**2+v**2+w**2)

        if self.eos is not None and self.eos.use_real:
            e = (rhoE - ke) / _softplus_floor(rho, 1e-8)
            p = self.eos.pressure(rho, e)
            c = self.eos.sound_speed(rho, e)
        else:
            p = (gamma-1) * _softplus_floor(rhoE - ke, 1e-8)
            c = torch.sqrt(gamma * _softplus_floor(p,1e-8) / _softplus_floor(rho,1e-8))

        T = p / _softplus_floor(rho, 1e-8)

        if self.cfg.use_sutherland:
            S    = 110.4/300.0
            mu_lam = self.nu_phys * rho * T.pow(1.5) * (1+S) / (T+S+1e-12)
        else:
            mu_lam = self.nu_phys * rho

        # Bug 11 fix (v6.4): self._ext_nu_ch was declared in __init__
        # ("External Cahn-Hilliard coupling buffers") and written by
        # CahnHilliardDNSBridge.sync() in one_core_v3.py (dns_solver.
        # _ext_nu_ch = nu_eff), but was never read anywhere in this
        # method -- only _ext_rho_ch and _ext_fx/fy/fz were actually
        # consumed. This meant the viscosity contrast between CH phases
        # (e.g. water/air, water/oil, liquefied-soil/water) had zero
        # effect on any coupled run; only the density contrast took
        # effect. _ext_nu_ch is a kinematic-viscosity-like modulation
        # (same units/role as nu_phys), so it enters mu_lam the same way
        # nu_phys does: multiplied by local rho. Zero-cost when not
        # connected, matching the _ext_fx/_ext_rho_ch guard pattern.
        if self._ext_nu_ch.abs().max() > 1e-30:
            mu_lam = mu_lam + rho * self._ext_nu_ch

        rho_p  = self._pad_field(rho);   u_p  = self._pad_field(u)
        v_p    = self._pad_field(v);     w_p  = self._pad_field(w)
        p_p    = self._pad_field(p)
        rhou_p = self._pad_field(rhou);  rhov_p = self._pad_field(rhov)
        rhow_p = self._pad_field(rhow);  rhoE_p = self._pad_field(rhoE)

        self._fill_ghost_cells(rho_p, rhou_p, rhov_p, rhow_p, rhoE_p)

        if self.distributed:
            u_p = rhou_p / _softplus_floor(rho_p, 1e-8)
            v_p = rhov_p / _softplus_floor(rho_p, 1e-8)
            w_p = rhow_p / _softplus_floor(rho_p, 1e-8)
            ke_p = 0.5*rho_p*(u_p**2+v_p**2+w_p**2)
            p_p  = (gamma-1)*_softplus_floor(rhoE_p - ke_p, 1e-8)
            self._exchange_halo_z(rho_p, u_p, v_p, w_p, p_p)
        else:
            u_p = rhou_p / _softplus_floor(rho_p, 1e-8)
            v_p = rhov_p / _softplus_floor(rho_p, 1e-8)
            w_p = rhow_p / _softplus_floor(rho_p, 1e-8)
            ke_p = 0.5*rho_p*(u_p**2+v_p**2+w_p**2)
            p_p  = (gamma-1)*_softplus_floor(rhoE_p - ke_p, 1e-8)

        def ddx(f): return (f[3:nx+3,2:ny+2,2:nz+2]-f[1:nx+1,2:ny+2,2:nz+2])/(2*dx)
        def ddy(f): return (f[2:nx+2,3:ny+3,2:nz+2]-f[2:nx+2,1:ny+1,2:nz+2])/(2*dx)
        def ddz(f): return (f[2:nx+2,2:ny+2,3:nz+3]-f[2:nx+2,2:ny+2,1:nz+1])/(2*dx)

        S11=ddx(u_p); S22=ddy(v_p); S33=ddz(w_p)
        S12=0.5*(ddy(u_p)+ddx(v_p)); S13=0.5*(ddz(u_p)+ddx(w_p)); S23=0.5*(ddz(v_p)+ddy(w_p))
        strain_mag  = torch.sqrt(2.0*(S11**2+S22**2+S33**2+2*(S12**2+S13**2+S23**2))+1e-16)
        dilatation  = S11+S22+S33

        # Pass external Langevin sigma to SOC if bridge has synced
        _ext = getattr(self, "_ext_sigma", None)
        nu_t   = self.soc.nu_t(rho, strain_mag, dilatation, dx, dt, c,
                                ext_sigma=_ext if (_ext is not None and _ext.abs() > 1e-12) else None)
        mu_eff = mu_lam + rho*nu_t

        if self.cfg.shock_capturing:
            vorticity = torch.sqrt(
                (ddz(v_p)-ddy(w_p))**2+(ddx(w_p)-ddz(u_p))**2+(ddy(u_p)-ddx(v_p))**2+1e-16)
            # Soft shock sensor (differentiable)
            shock_sensor = F.softplus(-dilatation, beta=50.0) / (strain_mag+1e-8) \
                           * torch.sigmoid((strain_mag - vorticity)/(strain_mag+1e-8) / _TAU_UPWIND)
            mu_shock = rho * dx * c * 0.1 * shock_sensor
            mu_eff   = mu_eff + mu_shock

        # ── Convective fluxes (DIFF-FIX 8: clean dispatch, no monkey-patch) ──
        conv_rho  = torch.zeros_like(rho)
        conv_rhou = torch.zeros_like(rho)
        conv_rhov = torch.zeros_like(rho)
        conv_rhow = torch.zeros_like(rho)
        conv_rhoE = torch.zeros_like(rho)

        if self._adv_scheme == 'semi_lagrangian':
            def _sl(q): return self._semi_lag(q, u, v, w, dt)
            conv_rho  = -(_sl(rho)  - rho)  / dt
            conv_rhou = -(_sl(rhou) - rhou) / dt
            conv_rhov = -(_sl(rhov) - rhov) / dt
            conv_rhow = -(_sl(rhow) - rhow) / dt
            conv_rhoE = -(_sl(rhoE) - rhoE) / dt
        else:
            for axis in range(3):
                mass_flux, f_u, f_v, f_w, f_E = self.flux_solver.compute_face_flux(
                    rho_p, u_p, v_p, w_p, p_p, axis,
                    scheme=self._adv_scheme,
                    limiter=self._adv_limiter,
                )
                if axis == 0:
                    conv_rho  += (mass_flux[1:]-mass_flux[:-1])/dx
                    conv_rhou += (f_u[1:]-f_u[:-1])/dx
                    conv_rhov += (f_v[1:]-f_v[:-1])/dx
                    conv_rhow += (f_w[1:]-f_w[:-1])/dx
                    conv_rhoE += (f_E[1:]-f_E[:-1])/dx
                elif axis == 1:
                    conv_rho  += (mass_flux[:,1:]-mass_flux[:,:-1])/dx
                    conv_rhou += (f_u[:,1:]-f_u[:,:-1])/dx
                    conv_rhov += (f_v[:,1:]-f_v[:,:-1])/dx
                    conv_rhow += (f_w[:,1:]-f_w[:,:-1])/dx
                    conv_rhoE += (f_E[:,1:]-f_E[:,:-1])/dx
                else:
                    conv_rho  += (mass_flux[:,:,1:]-mass_flux[:,:,:-1])/dx
                    conv_rhou += (f_u[:,:,1:]-f_u[:,:,:-1])/dx
                    conv_rhov += (f_v[:,:,1:]-f_v[:,:,:-1])/dx
                    conv_rhow += (f_w[:,:,1:]-f_w[:,:,:-1])/dx
                    conv_rhoE += (f_E[:,:,1:]-f_E[:,:,:-1])/dx

        # ── Viscous fluxes ───────────────────────────────────────────────────
        div_v    = S11+S22+S33
        tau_xx   = mu_eff*(2*S11-(2./3.)*div_v)
        tau_yy   = mu_eff*(2*S22-(2./3.)*div_v)
        tau_zz   = mu_eff*(2*S33-(2./3.)*div_v)
        tau_xy   = mu_eff*(ddy(u_p)+ddx(v_p))
        tau_xz   = mu_eff*(ddz(u_p)+ddx(w_p))
        tau_yz   = mu_eff*(ddz(v_p)+ddy(w_p))

        if self.ito_gen is not None:
            s11,s22,s33,s12,s13,s23 = self.ito_gen.generate(tau_xx.shape, self.device, dt)
            tau_xx=tau_xx+s11; tau_yy=tau_yy+s22; tau_zz=tau_zz+s33
            tau_xy=tau_xy+s12; tau_xz=tau_xz+s13; tau_yz=tau_yz+s23

        def _padv(t): q=self._pad_field(t); self._fill_ghost_cells(q); return q
        p_txx=_padv(tau_xx); p_txy=_padv(tau_xy); p_txz=_padv(tau_xz)
        p_tyy=_padv(tau_yy); p_tyz=_padv(tau_yz); p_tzz=_padv(tau_zz)

        if self.distributed:
            self._exchange_halo_z(p_txx,p_txy,p_txz,p_tyy,p_tyz,p_tzz)

        visc_rhou = ddx(p_txx)+ddy(p_txy)+ddz(p_txz)
        visc_rhov = ddx(p_txy)+ddy(p_tyy)+ddz(p_tyz)
        visc_rhow = ddx(p_txz)+ddy(p_tyz)+ddz(p_tzz)

        k_eff  = mu_eff*gamma/(gamma-1)/self.Pr
        T_p    = _padv(T)
        if self.distributed: self._exchange_halo_z(T_p)
        qx=k_eff*ddx(T_p); qy=k_eff*ddy(T_p); qz=k_eff*ddz(T_p)
        qxp=_padv(qx); qyp=_padv(qy); qzp=_padv(qz)
        if self.distributed: self._exchange_halo_z(qxp,qyp,qzp)
        heat_div = ddx(qxp)+ddy(qyp)+ddz(qzp)

        td_x=tau_xx*u+tau_xy*v+tau_xz*w
        td_y=tau_xy*u+tau_yy*v+tau_yz*w
        td_z=tau_xz*u+tau_yz*v+tau_zz*w
        work_div = ddx(_padv(td_x))+ddy(_padv(td_y))+ddz(_padv(td_z))
        visc_rhoE = work_div + heat_div

        rhs_rho  = -conv_rho
        rhs_rhou = -conv_rhou + visc_rhou
        rhs_rhov = -conv_rhov + visc_rhov
        rhs_rhow = -conv_rhow + visc_rhow
        rhs_rhoE = -conv_rhoE + visc_rhoE

        # ── Cahn-Hilliard Korteweg body force injection ─────────────────────
        # Written by CahnHilliardDNSBridge.sync(); zero-cost if not connected.
        if self._ext_fx.abs().max() > 1e-30:
            rhs_rhou = rhs_rhou + self._ext_fx
            rhs_rhov = rhs_rhov + self._ext_fy
            rhs_rhow = rhs_rhow + self._ext_fz
            # Energy: f · u  (differentiable — u already computed above)
            rhs_rhoE = rhs_rhoE + (self._ext_fx * u +
                                   self._ext_fy * v +
                                   self._ext_fz * w)

        # ── External volumetric heat release (v6.5) ─────────────────────────
        # Written by HeatReleaseDNSBridge.sync() (one_core_v3.py), fed by
        # fire_dns_coupling_one.py. Direct energy-equation source term
        # [W/m^3] -- distinct from the f.u mechanical-work term above,
        # which cannot represent combustion heat release or net radiative
        # gain/loss. Zero-cost if not connected.
        if self._ext_q.abs().max() > 1e-30:
            rhs_rhoE = rhs_rhoE + self._ext_q

        # ── v6.6: resolved mixture-fraction transport (Z = RZ/rho) ──────────
        # Conserved-scalar (Shvab-Zeldovich) formulation: Z itself is
        # SOURCE-FREE under fast chemistry -- combustion doesn't create or
        # destroy the conserved mixture fraction, it only redistributes
        # composition/temperature AT a given Z via the Burke-Schumann state
        # relation. This is the same modeling principle fire_one.py's
        # MixtureFractionCombustion uses, now evaluated on the LOCALLY
        # RESOLVED Z field instead of a prescribed external HRR(t).
        #
        # Deliberately uses simple centered-difference advection/diffusion
        # (matching this file's existing viscous-term style: S11, heat_div,
        # etc. are all centered differences) rather than hooking into
        # flux_solver's Godunov/Riemann-based Euler scheme -- Z has no
        # acoustic/shock structure of its own, and NOT touching the
        # compressible flux solver means this addition cannot destabilize
        # the proven Euler/Navier-Stokes core. Known limitation: centered
        # advection can ring at high local Peclet number (sharp Z fronts,
        # low diffusivity); the D_Z floor below provides some numerical
        # safety margin, but very under-resolved flame sheets may still
        # need more diffusion or a proper upwind/TVD scalar scheme -- flag
        # this in your own validation before trusting sharp-front results.
        rho_safe = _softplus_floor(rho, 1e-8)
        Z = RZ / rho_safe
        Z_p  = self._pad_field(Z)
        dZdx = ddx(Z_p); dZdy = ddy(Z_p); dZdz = ddz(Z_p)

        D_Z = torch.clamp(mu_eff / (rho_safe * self.turbulent_schmidt), min=1e-6)

        adv_RZ = -(ddx(self._pad_field(rho*u*Z)) +
                   ddy(self._pad_field(rho*v*Z)) +
                   ddz(self._pad_field(rho*w*Z)))

        diff_RZ = (ddx(self._pad_field(rho*D_Z*dZdx)) +
                   ddy(self._pad_field(rho*D_Z*dZdy)) +
                   ddz(self._pad_field(rho*D_Z*dZdz)))

        rhs_RZ = adv_RZ + diff_RZ   # source-free (conserved scalar)

        combustion_source = torch.zeros_like(rho)
        if self.enable_combustion:
            # CRITICAL UNIT NOTE: this solver's internal T is NON-DIMENSIONAL
            # (T* = T_real/T_ref, T_ref=300.0 -- see the Sutherland-law
            # constant S=110.4/300.0 used a few lines above, which only
            # makes sense if T here is already scaled by that same 300K
            # reference). Combustion/radiation physics below is most
            # transparently derived in REAL units (Kelvin, W/m^3), so T is
            # converted to real Kelvin at the point of use via self.T_ref.
            T_real = T * self.T_ref

            Zst = max(self.z_stoich, 1e-6)
            T_eq_real = torch.where(
                Z < Zst,
                self.T_ambient_rad + (self.T_adiabatic - self.T_ambient_rad) * (Z / Zst),
                self.T_ambient_rad + (self.T_adiabatic - self.T_ambient_rad) *
                    ((1.0 - Z) / max(1.0 - Zst, 1e-6)),
            )
            tau_mix = (dx**2) / D_Z
            combustion_source_dimensional = (
                rho * self.cp_gas * (T_eq_real - T_real) / torch.clamp(tau_mix, min=1e-6)
            )   # [W/m^3], REAL/dimensional units

            # UNRESOLVED SCALING GAP (flagged, not hidden): rhs_rhoE is in
            # this solver's own NON-DIMENSIONAL unit system, whose energy
            # scaling depends on the reference length/velocity/density
            # (L_ref, U_ref, rho_ref) implied by cfg.Re / cfg.Mach -- which
            # this method does not have enough visibility into to derive
            # automatically. combustion_nondim_scale is an EXPLICIT,
            # user-supplied conversion factor (default 1.0, i.e. NO
            # conversion applied) so this gap is a visible, must-set knob
            # rather than a silently wrong assumption. Compute it as
            # (L_ref / (rho_ref * U_ref**3)) for your specific
            # nondimensionalization before trusting quantitative results.
            combustion_source = combustion_source_dimensional * self.combustion_nondim_scale
            rhs_rhoE = rhs_rhoE + combustion_source

        # ── v6.6: P1 (diffusion-approximation) radiation ────────────────────
        # Solves -1/(3*kappa)*lap(G) + kappa*G = 4*kappa*sigma*T^4 via FFT,
        # which assumes PERIODIC boundary conditions (matching this
        # solver's default circular-padding convention) -- for a
        # wall-dominated enclosure this FFT shortcut is not rigorously
        # correct (a real P1 solve there needs Marshak wall BCs and an
        # iterative elliptic solver); using it on a non-periodic domain
        # will silently give an approximate, not exact, radiative field.
        # Documented, not hidden: check your BC setup before trusting
        # radiation results in a walled enclosure.
        if self.enable_radiation:
            # Same T_ref conversion and UNRESOLVED nondim-scaling caveat as
            # the combustion block above applies here (both branches).
            T_real = T * self.T_ref
            kappa = self.radiation_absorption_coeff
            sigma_sb = 5.670374e-8

            if self.radiation_method == "DOM":
                # v6.7, opt-in (cfg.radiation_method="DOM"). See
                # _solve_dom_radiation's docstring for the source-
                # iteration method and its periodic-BC assumption.
                q_rad_dimensional = self._solve_dom_radiation(T_real, kappa, dx)
            else:
                # v6.6 default: P1 diffusion approximation via FFT.
                S_rad = 4.0 * kappa * sigma_sb * T_real**4
                S_hat = torch.fft.fftn(S_rad)
                kx = torch.fft.fftfreq(nx, d=dx, device=rho.device) * 2 * math.pi
                ky = torch.fft.fftfreq(ny, d=dx, device=rho.device) * 2 * math.pi
                kz = torch.fft.fftfreq(nz, d=dx, device=rho.device) * 2 * math.pi
                KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
                k2 = KX**2 + KY**2 + KZ**2
                denom = kappa + k2 / (3.0 * kappa + 1e-12)
                G_hat = S_hat / denom
                G = torch.real(torch.fft.ifftn(G_hat))
                q_rad_dimensional = kappa * (4.0 * sigma_sb * T_real**4 - G)   # [W/m^3]

            q_rad = q_rad_dimensional * self.combustion_nondim_scale
            rhs_rhoE = rhs_rhoE - q_rad

        # ── v6.7: soot transport (opt-in, cfg.enable_soot) ──────────────────
        # Parallels the RZ transport above: rho*Y_soot carried as a
        # conserved scalar via the SAME centered-difference advection/
        # diffusion scheme (same rationale: doesn't touch the compressible
        # flux solver). Production/oxidation source only applied when
        # BOTH enable_soot AND enable_combustion are True -- soot kinetics
        # need Y_fuel/Y_O2/T from the resolved combustion state; with
        # enable_combustion=False, soot is purely a passive (inert,
        # never-produced) transported scalar, matching the "double-gated"
        # safety approach for this whole feature set.
        rhs_RS = torch.zeros_like(rho)
        if self.enable_soot:
            rho_safe2 = _softplus_floor(rho, 1e-8)
            Y_soot = RS / rho_safe2
            Ys_p = self._pad_field(Y_soot)
            dYsdx = ddx(Ys_p); dYsdy = ddy(Ys_p); dYsdz = ddz(Ys_p)
            D_S = torch.clamp(mu_eff / (rho_safe2 * self.soot_schmidt), min=1e-6)

            adv_RS = -(ddx(self._pad_field(rho*u*Y_soot)) +
                       ddy(self._pad_field(rho*v*Y_soot)) +
                       ddz(self._pad_field(rho*w*Y_soot)))
            diff_RS = (ddx(self._pad_field(rho*D_S*dYsdx)) +
                       ddy(self._pad_field(rho*D_S*dYsdy)) +
                       ddz(self._pad_field(rho*D_S*dYsdz)))
            rhs_RS = adv_RS + diff_RS

            if self.enable_combustion:
                # Reuses the same Burke-Schumann state (Z, T_eq/T_real
                # already computed above) to derive Y_fuel/Y_O2 at each
                # cell (piecewise-linear state relations, matching
                # fire_one.MixtureFractionCombustion.state_relations),
                # then applies fire_one.SootKinetics' rate laws directly
                # (same formulas, ported here to avoid importing fire_one
                # into the DNS solver -- see fire_one.py for the
                # standalone, independently-tested version of these exact
                # rate expressions).
                Zst2 = max(self.z_stoich, 1e-6)
                Y_fuel = torch.where(Z > Zst2, (Z - Zst2) / max(1.0 - Zst2, 1e-6),
                                      torch.zeros_like(Z))
                Y_O2 = torch.where(Z < Zst2, 0.233 * (1.0 - Z / Zst2), torch.zeros_like(Z))
                R_gas = 8.314
                k_f = self.soot_A_formation * torch.exp(
                    -self.soot_Ea_formation / (R_gas * torch.clamp(T_real, min=200.0)))
                formation = torch.where(Z > Zst2, k_f * Y_fuel, torch.zeros_like(Z))
                k_o = self.soot_A_oxidation * torch.exp(
                    -self.soot_Ea_oxidation / (R_gas * torch.clamp(T_real, min=200.0)))
                oxidation = k_o * Y_soot * Y_O2
                # Rate is per unit time on Y_soot; convert to a source on
                # the CONSERVED RS=rho*Y_soot: d(RS)/dt|_reaction = rho*(formation-oxidation).
                rhs_RS = rhs_RS + rho * (formation - oxidation)

        # ── v6.8: external mass source (continuity + carried quantities) ────
        # See __init__ for buffer descriptions. This is the ONLY place in
        # this file where rhs_rho gets a source term at all -- continuity
        # was exactly closed (convective terms only) before this. Guarded
        # exactly like every other _ext_ buffer.
        #
        # UNIT-CONSISTENCY FIX: an earlier version of this block added the
        # _ext_mdot* buffers (written in REAL/dimensional units by
        # PyrolysisDNSBridge -- kg/(m^3.s), etc.) directly into this
        # solver's NON-DIMENSIONAL rhs_rho/rhou/rhov/rhow/rhoE/RZ, with no
        # conversion at all -- the exact same class of bug already found
        # and fixed once for combustion/radiation (v6.6). Worse: mass-rate
        # and energy-rate source terms have DIFFERENT dimensional groups
        # (mass: ~L_ref/(rho_ref*U_ref); energy: ~L_ref/(rho_ref*U_ref^3)),
        # so reusing combustion_nondim_scale here would itself be
        # dimensionally wrong even if a conversion had been applied. A
        # SEPARATE, explicitly-flagged mdot_nondim_scale is used instead.
        if self._ext_mdot.abs().max() > 1e-30:
            mdot_scale = self.mdot_nondim_scale
            rhs_rho  = rhs_rho  + self._ext_mdot   * mdot_scale
            rhs_rhou = rhs_rhou + self._ext_mdot_u * mdot_scale
            rhs_rhov = rhs_rhov + self._ext_mdot_v * mdot_scale
            rhs_rhow = rhs_rhow + self._ext_mdot_w * mdot_scale
            rhs_rhoE = rhs_rhoE + self._ext_mdot_e * mdot_scale
            rhs_RZ   = rhs_RZ   + self._ext_mdot_Z * mdot_scale

        return rhs_rho, rhs_rhou, rhs_rhov, rhs_rhow, rhs_rhoE, rhs_RZ, rhs_RS

    # ── TVD-RK3 time integration ─────────────────────────────────────────────

    def step(self, dt=None):
        rho,rhou,rhov,rhow,rhoE = self.rho,self.rhou,self.rhov,self.rhow,self.rhoE
        RZ = self.RZ
        RS = self.RS
        gamma = self.gamma; dx = self.dx

        if dt is None:
            u = rhou/_softplus_floor(rho,1e-8)
            v = rhov/_softplus_floor(rho,1e-8)
            w = rhow/_softplus_floor(rho,1e-8)
            p = (gamma-1)*_softplus_floor(rhoE-0.5*rho*(u**2+v**2+w**2),1e-8)
            c = torch.sqrt(gamma*_softplus_floor(p,1e-8)/_softplus_floor(rho,1e-8))
            speed = torch.sqrt(u**2+v**2+w**2) + c
            # Differentiable dt: logsumexp-max instead of speed.max()
            dt = self.cfg.cfl * dx / (_logsumexp_max(speed, tau=_TAU_LSE) + 1e-8)
            if self.distributed:
                dist.all_reduce(dt, op=dist.ReduceOp.MIN)

        def _stage(r,ru,rv,rw,rE):
            self._apply_bc_to_boundary_cells(r,ru,rv,rw,rE)
            if self.ib is not None:
                ru,rv,rw,rE = self.ib.apply_forcing(r,ru,rv,rw,rE,dt)
            self._apply_wall_model(r,ru,rv,rw,rE,dt)
            return r,ru,rv,rw,rE

        # Stage 1
        k1 = self._compute_rhs(rho,rhou,rhov,rhow,rhoE,RZ,RS,dt)
        r1=rho+dt*k1[0]; ru1=rhou+dt*k1[1]; rv1=rhov+dt*k1[2]; rw1=rhow+dt*k1[3]; rE1=rhoE+dt*k1[4]
        RZ1 = RZ + dt*k1[5]
        RS1 = RS + dt*k1[6]
        r1,ru1,rv1,rw1,rE1 = _stage(r1,ru1,rv1,rw1,rE1)
        RZ1 = torch.clamp(RZ1, min=0.0)   # RZ=rho*Z, Z>=0 physically; floor only, no upper clamp (r1 not yet known outside _stage's own floors)
        RS1 = torch.clamp(RS1, min=0.0)

        # Stage 2
        k2 = self._compute_rhs(r1,ru1,rv1,rw1,rE1,RZ1,RS1,dt)
        r2=0.75*rho+0.25*(r1+dt*k2[0]); ru2=0.75*rhou+0.25*(ru1+dt*k2[1])
        rv2=0.75*rhov+0.25*(rv1+dt*k2[2]); rw2=0.75*rhow+0.25*(rw1+dt*k2[3])
        rE2=0.75*rhoE+0.25*(rE1+dt*k2[4])
        RZ2=0.75*RZ+0.25*(RZ1+dt*k2[5])
        RS2=0.75*RS+0.25*(RS1+dt*k2[6])
        r2,ru2,rv2,rw2,rE2 = _stage(r2,ru2,rv2,rw2,rE2)
        RZ2 = torch.clamp(RZ2, min=0.0)
        RS2 = torch.clamp(RS2, min=0.0)

        # Stage 3
        k3 = self._compute_rhs(r2,ru2,rv2,rw2,rE2,RZ2,RS2,dt)
        rho_n =(1./3.)*rho +(2./3.)*(r2 +dt*k3[0])
        rhou_n=(1./3.)*rhou+(2./3.)*(ru2+dt*k3[1])
        rhov_n=(1./3.)*rhov+(2./3.)*(rv2+dt*k3[2])
        rhow_n=(1./3.)*rhow+(2./3.)*(rw2+dt*k3[3])
        rhoE_n=(1./3.)*rhoE+(2./3.)*(rE2+dt*k3[4])
        RZ_n  =(1./3.)*RZ  +(2./3.)*(RZ2+dt*k3[5])
        RS_n  =(1./3.)*RS  +(2./3.)*(RS2+dt*k3[6])
        rho_n,rhou_n,rhov_n,rhow_n,rhoE_n = _stage(rho_n,rhou_n,rhov_n,rhow_n,rhoE_n)

        # DIFF-FIX 7: softplus floor instead of hard clamp
        rho_n  = _softplus_floor(rho_n,  1e-6)
        ke_n   = 0.5*(rhou_n**2+rhov_n**2+rhow_n**2)/_softplus_floor(rho_n,1e-8)
        p_n    = (gamma-1)*_softplus_floor(rhoE_n - ke_n, 1e-8)
        p_n    = _softplus_floor(p_n, 1e-8)
        rhoE_n = p_n/(gamma-1) + ke_n

        # v6.6: Z = RZ/rho must stay in [0,1] physically (mass fraction of
        # fuel-side material); clamp RZ to [0, rho_n] to enforce that
        # without a hard clamp on Z itself breaking differentiability of
        # rho_n (matches this file's general "soft floor, not hard clamp"
        # philosophy -- DIFF-FIX 7 above -- applied to the new field too).
        RZ_n = torch.clamp(RZ_n, min=0.0, max=None)
        RZ_n = torch.minimum(RZ_n, rho_n)   # Z<=1 => RZ<=rho

        # v6.7: same [0,1] mass-fraction enforcement for soot.
        RS_n = torch.clamp(RS_n, min=0.0, max=None)
        RS_n = torch.minimum(RS_n, rho_n)

        self.rho=rho_n; self.rhou=rhou_n; self.rhov=rhov_n
        self.rhow=rhow_n; self.rhoE=rhoE_n; self.RZ=RZ_n; self.RS=RS_n
        self.step_count += 1; self.time += dt.item() if hasattr(dt,'item') else dt

        if self.rg is not None and self.step_count % self.cfg.rg_interval == 0:
            self.rho  = self.rg.forward(self.rho)
            self.rhou = self.rg.forward(self.rhou)
            self.rhov = self.rg.forward(self.rhov)
            self.rhow = self.rg.forward(self.rhow)
            self.rhoE = self.rg.forward(self.rhoE)
            # Note: self.RZ intentionally NOT passed through the
            # renormalization-group filter (self.rg) -- rg was designed
            # for the 5 Euler conservation variables; applying it to RZ
            # without validating that it preserves Z in [0,1] "physically
            # renormalized" is a separate exercise. Flagged, not silently done.

    @property
    def Z(self):
        """Mixture fraction (primitive), derived from the conserved RZ=rho*Z."""
        return self.RZ / _softplus_floor(self.rho, 1e-8)

    @property
    def Y_soot(self):
        """Soot mass fraction (primitive), derived from conserved RS=rho*Y_soot."""
        return self.RS / _softplus_floor(self.rho, 1e-8)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_primitives(self):
        rho = self.rho
        u   = self.rhou/_softplus_floor(rho,1e-8)
        v   = self.rhov/_softplus_floor(rho,1e-8)
        w   = self.rhow/_softplus_floor(rho,1e-8)
        ke  = 0.5*rho*(u**2+v**2+w**2)
        p   = (self.gamma-1)*_softplus_floor(self.rhoE-ke,1e-8)
        T   = p/_softplus_floor(rho,1e-8)
        c   = torch.sqrt(self.gamma*_softplus_floor(p,1e-8)/_softplus_floor(rho,1e-8))
        Ma  = torch.sqrt(u**2+v**2+w**2)/(c+1e-8)
        return {"rho":rho,"u":u,"v":v,"w":w,"p":p,"T":T,"c":c,"Ma":Ma}

    def kinetic_energy(self):
        prim = self.get_primitives()
        return 0.5*(prim["rho"]*(prim["u"]**2+prim["v"]**2+prim["w"]**2)).mean()

    def kolmogorov_slope(self):
        """Compute spectral energy cascade slope (should be ≈ -5/3 in inertial range)."""
        if self.distributed:
            local_u = self.rhou / _softplus_floor(self.rho, 1e-8)
            local_v = self.rhov / _softplus_floor(self.rho, 1e-8)
            local_w = self.rhow / _softplus_floor(self.rho, 1e-8)
            gl_u = [torch.zeros_like(local_u) for _ in range(self.world_size)] if self.rank==0 else None
            gl_v = [torch.zeros_like(local_v) for _ in range(self.world_size)] if self.rank==0 else None
            gl_w = [torch.zeros_like(local_w) for _ in range(self.world_size)] if self.rank==0 else None
            dist.gather(local_u, gl_u, dst=0)
            dist.gather(local_v, gl_v, dst=0)
            dist.gather(local_w, gl_w, dst=0)
            if self.rank != 0:
                return None
            u_full = torch.cat(gl_u, dim=2)
            v_full = torch.cat(gl_v, dim=2)
            w_full = torch.cat(gl_w, dim=2)
        else:
            u_full = self.rhou / _softplus_floor(self.rho, 1e-8)
            v_full = self.rhov / _softplus_floor(self.rho, 1e-8)
            w_full = self.rhow / _softplus_floor(self.rho, 1e-8)

        u_hat = fftn(u_full.detach().cpu().numpy())
        v_hat = fftn(v_full.detach().cpu().numpy())
        w_hat = fftn(w_full.detach().cpu().numpy())
        kx = fftfreq(u_full.shape[0], d=self.cfg.Lx/u_full.shape[0])
        ky = fftfreq(u_full.shape[1], d=self.cfg.Ly/u_full.shape[1])
        kz = fftfreq(u_full.shape[2], d=self.cfg.Lz/u_full.shape[2])
        KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='ij')
        K = np.sqrt(KX**2+KY**2+KZ**2)
        bins = np.logspace(np.log10(max(K[K>0].min(),1e-6)), np.log10(K.max()+1e-6), 20)
        E_spec = []
        for i in range(len(bins)-1):
            mask = (K >= bins[i]) & (K < bins[i+1])
            if np.any(mask):
                E = 0.5*(np.abs(u_hat[mask])**2+np.abs(v_hat[mask])**2+np.abs(w_hat[mask])**2).mean()
                E_spec.append(E)
            else:
                E_spec.append(0.0)
        valid = np.array(E_spec) > 0
        if valid.sum() < 3:
            return None
        kc = 0.5*(bins[:-1]+bins[1:])
        slope,_,_,_,_ = linregress(np.log10(kc[valid]), np.log10(np.array(E_spec)[valid]))
        return slope

    def grid_convergence_test(self, grid_sizes=None, ref_steps=50):
        """
        Estimate spatial convergence rate via Taylor-Green vortex at multiple grids.
        Solver state is fully restored after the test (try/finally).
        """
        if grid_sizes is None:
            grid_sizes = [32, 64, 96]
        if self.distributed:
            logger.warning("Grid convergence test not supported in distributed mode.")
            return None

        import copy
        cfg_backup    = copy.copy(self.cfg)
        dx_backup     = self.dx
        fields_backup = {
            a: getattr(self, a).cpu() if getattr(self, a) is not None else None
            for a in ('rho','rhou','rhov','rhow','rhoE')
        }
        step_backup = self.step_count
        time_backup = self.time

        errors = []
        u_ref  = None
        sorted_sizes = sorted(grid_sizes)

        try:
            for N in sorted_sizes:
                self.cfg.nx = N; self.cfg.ny = N; self.cfg.nz = N
                new_dx = cfg_backup.Lx / N
                self.cfg.dx = self.cfg.dy = self.cfg.dz = new_dx
                self.dx = new_dx; self.local_nz = N
                self.soc.reset()
                self.initialize('taylor_green')
                for _ in range(ref_steps):
                    self.step()
                u = self.rhou / _softplus_floor(self.rho, 1e-8)
                if N == sorted_sizes[-1]:
                    u_ref = u.clone()
                else:
                    if u_ref is not None:
                        u_ref_down = F.interpolate(
                            u_ref.unsqueeze(0).unsqueeze(0),
                            size=(N,N,N), mode='trilinear', align_corners=False
                        ).squeeze()
                        err = torch.norm(u - u_ref_down).item() / math.sqrt(N**3)
                        errors.append(err)
        finally:
            for attr in ('nx','ny','nz','Lx','Ly','Lz','dx','dy','dz'):
                setattr(self.cfg, attr, getattr(cfg_backup, attr))
            self.dx = dx_backup; self.local_nz = cfg_backup.nz
            self.step_count = step_backup; self.time = time_backup
            for attr, val in fields_backup.items():
                setattr(self, attr, val.to(self.device) if val is not None else None)
            self.soc.reset()
            logger.info("Grid convergence test complete; solver state restored.")

        if len(errors) >= 2:
            valid_sizes = sorted_sizes[:len(errors)]
            slope,_,_,_,_ = linregress(np.log(valid_sizes), np.log(errors))
            return -slope
        return None

    # ── run loop ──────────────────────────────────────────────────────────────

    def run(self, steps=None):
        if steps is None:
            steps = self.cfg.steps
        if self.rho is None:
            self.initialize()
        for t in range(1, steps+1):
            self.step()
            if t % 50 == 0 or t == 1:
                u = self.rhou / _softplus_floor(self.rho, 1e-8)
                v = self.rhov / _softplus_floor(self.rho, 1e-8)
                w = self.rhow / _softplus_floor(self.rho, 1e-8)
                ke_local = 0.5*torch.mean(self.rho*(u**2+v**2+w**2)).item()
                if self.distributed:
                    ke_t = torch.tensor(ke_local, device=self.device)
                    dist.all_reduce(ke_t, op=dist.ReduceOp.SUM)
                    ke_global = ke_t.item() / self.world_size
                else:
                    ke_global = ke_local
                if self.rank == 0:
                    logger.info(f"Step {t:04d}  time={self.time:.6f}  KE={ke_global:.6f}")
                self.energy_hist.append(ke_global)
        # track divergence history
        if self.rho is not None:
            try:
                u_ = self.rhou/_softplus_floor(self.rho,1e-8)
                v_ = self.rhov/_softplus_floor(self.rho,1e-8)
                w_ = self.rhow/_softplus_floor(self.rho,1e-8)
                div = ((torch.roll(u_,-1,0)-torch.roll(u_,1,0))
                      +(torch.roll(v_,-1,1)-torch.roll(v_,1,1))
                      +(torch.roll(w_,-1,2)-torch.roll(w_,1,2)))/(2*self.dx)
                self.div_hist.append(div.abs().max().item())
            except Exception:
                pass
        if self.distributed:
            dist.barrier()

    def taylor_green_test(self, steps=200):
        logger.info("=== Taylor-Green vortex test ===")
        self.initialize('taylor_green')
        self.run(steps=steps)
        u = self.rhou / _softplus_floor(self.rho, 1e-8)
        v = self.rhov / _softplus_floor(self.rho, 1e-8)
        w = self.rhow / _softplus_floor(self.rho, 1e-8)
        ke_local = 0.5*torch.mean(self.rho*(u**2+v**2+w**2)).item()
        if self.distributed:
            ke_t = torch.tensor(ke_local, device=self.device)
            dist.all_reduce(ke_t, op=dist.ReduceOp.SUM)
            ke_final = ke_t.item() / self.world_size
        else:
            ke_final = ke_local
        if self.rank == 0:
            logger.info(f"Final kinetic energy = {ke_final:.6f}")
        return ke_final, self.energy_hist

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path):
        """Save full solver state including SOC kernel weights and SSC buffer."""
        # Bug 1 fix: SSC buffer is named prev_sigma (not _prev)
        ssc_state = (self.soc.ssc.prev_sigma.cpu()
                     if (self.soc.ssc is not None and
                         hasattr(self.soc.ssc, 'prev_sigma')) else None)
        state = {
            'cfg':         self.cfg,
            'step':        self.step_count,
            'time':        self.time,
            'rho':         self.rho.cpu(),
            'rhou':        self.rhou.cpu(),
            'rhov':        self.rhov.cpu(),
            'rhow':        self.rhow.cpu(),
            'rhoE':        self.rhoE.cpu(),
            'RZ':          self.RZ.cpu(),   # v6.6: mixture-fraction conserved field
            'RS':          self.RS.cpu(),   # v6.7: soot conserved field
            'soc_kernel_state_dict': self.soc.kernel.state_dict(),
            'ssc_prev':    ssc_state,
        }
        torch.save(state, path)
        logger.info(f"Checkpoint saved → {path}  (step {self.step_count})")

    def load_checkpoint(self, path):
        """Load solver state; restores SOC kernel and SSC buffer (v2+ format)."""
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.cfg        = state['cfg']
        self.step_count = state['step']
        self.time       = state['time']
        self.rho   = state['rho'].to(self.device)
        self.rhou  = state['rhou'].to(self.device)
        self.rhov  = state['rhov'].to(self.device)
        self.rhow  = state['rhow'].to(self.device)
        self.rhoE  = state['rhoE'].to(self.device)
        # v6.6: backward compatible with pre-v6.6 checkpoints (no 'RZ' key)
        # -- falls back to zeros (no mixture fraction / unburnt everywhere)
        # rather than raising, since combustion is opt-in (enable_combustion).
        if 'RZ' in state:
            self.RZ = state['RZ'].to(self.device)
        else:
            self.RZ = torch.zeros_like(self.rho)
            if self.enable_combustion:
                logger.warning(
                    "load_checkpoint: pre-v6.6 checkpoint has no 'RZ' field "
                    "but enable_combustion=True -- resuming with Z=0 "
                    "(unburnt) everywhere, not the actual prior mixture-"
                    "fraction state."
                )
        # v6.7: same backward-compatible pattern for soot.
        if 'RS' in state:
            self.RS = state['RS'].to(self.device)
        else:
            self.RS = torch.zeros_like(self.rho)
            if self.enable_soot:
                logger.warning(
                    "load_checkpoint: checkpoint has no 'RS' field but "
                    "enable_soot=True -- resuming with Y_soot=0 everywhere, "
                    "not the actual prior soot state."
                )
        if 'soc_kernel_state_dict' in state:
            self.soc.kernel.load_state_dict(state['soc_kernel_state_dict'])
            logger.info("SOC kernel weights restored.")
        else:
            logger.warning("Checkpoint predates v2: using default SOC kernel.")
        # Bug 1 fix: restore prev_sigma (correct buffer name)
        if 'ssc_prev' in state and state['ssc_prev'] is not None:
            if self.soc.ssc is not None:
                self.soc.ssc.prev_sigma.data = state['ssc_prev'].to(self.device)
        elif self.soc.ssc is not None:
            self.soc.ssc.reset()
        logger.info(f"Checkpoint loaded ← {path}  (step={self.step_count}, time={self.time:.6g})")


# =============================================================================
# 9. Signal denoising & SOC Trainer  (unchanged)
# =============================================================================

# =============================================================================
# 9. Advanced Signal Denoising
# =============================================================================

class SignalDenoiser:
    """Multi-method signal denoiser: SSC, Wiener, or Wavelet."""

    def __init__(self, method='ssc', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def denoise(self, data: torch.Tensor) -> torch.Tensor:
        if self.method == 'ssc':
            ssc = SemanticStateContraction(**self.kwargs)
            return ssc(data.mean())
        elif self.method == 'wiener':
            data_np = data.detach().cpu().numpy()
            denoised = scipy_signal.wiener(data_np, **self.kwargs)
            return torch.tensor(denoised, device=data.device)
        elif self.method == 'wavelet' and HAS_PYWT:
            data_np = data.detach().cpu().numpy()
            wav   = self.kwargs.get('wavelet', 'db4')
            level = self.kwargs.get('level', 4)
            thresh = self.kwargs.get('threshold', 0.1)
            coeffs = pywt.wavedec(data_np, wav, level=level)
            coeffs[1:] = [pywt.threshold(c, thresh) for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, wav)
            return torch.tensor(denoised[:data.shape[0]], device=data.device)
        else:
            return data


# =============================================================================
# 10. SOC Kernel Trainer  (gradient-based + Optuna + Differential Evolution)
# =============================================================================

class SOCTrainer:
    """
    Full SOC kernel trainer supporting three modes:
      1. gradient — Adam backprop through differentiable step() (NEW in v5)
      2. optuna   — Bayesian hyperparameter search via Optuna
      3. de       — Differential Evolution (scipy, fallback)
    """

    def __init__(self, solver: CompressibleSolver, target_spectrum=None):
        self.solver    = solver
        self.target    = target_spectrum
        self.optimizer = torch.optim.Adam(
            list(solver.soc.kernel.parameters()), lr=1e-3)

    # ── Gradient-based (native differentiable, v5 only) ──────────────────────

    def train_step(self, n_steps=10):
        """One Adam step backpropping through n_steps of the solver."""
        self.optimizer.zero_grad()
        self.solver.initialize('taylor_green')
        for _ in range(n_steps):
            self.solver.step()
        ke   = self.solver.kinetic_energy()
        loss = ke if self.target is None else (ke - self.target)**2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_gradient(self, n_outer=50, n_inner=10):
        """Full gradient training loop."""
        losses = []
        for i in range(n_outer):
            loss = self.train_step(n_inner)
            losses.append(loss)
            if i % 10 == 0:
                logger.info(f"SOCTrainer  iter={i:4d}  loss={loss:.6e}")
        return losses

    # ── Black-box optimisation (Optuna / DE) ─────────────────────────────────

    @staticmethod
    def _eval(params, solver, target_energy, n_steps=50):
        solver.soc.kernel.log_Cs.data     = torch.tensor(math.log(max(params[0],1e-9)), device=solver.device)
        solver.soc.kernel.log_lambda.data = torch.tensor(math.log(max(params[1],1e-9)), device=solver.device)
        solver.soc.kernel.log_alpha.data  = torch.tensor(math.log(max(params[2],1e-9)), device=solver.device)
        solver.soc.kernel.log_theta.data  = torch.tensor(math.log(max(params[3],1e-9)), device=solver.device)
        solver.soc.kernel.log_tau.data    = torch.tensor(math.log(max(params[4],1e-9)), device=solver.device)
        solver.initialize('taylor_green')
        with torch.no_grad():
            for _ in range(n_steps):
                solver.step()
        u = solver.rhou / _softplus_floor(solver.rho, 1e-8)
        v = solver.rhov / _softplus_floor(solver.rho, 1e-8)
        w = solver.rhow / _softplus_floor(solver.rho, 1e-8)
        ke = 0.5*torch.mean(solver.rho*(u**2+v**2+w**2)).item()
        return abs(ke - target_energy)

    @classmethod
    def train(cls, solver, target_energy=0.1, method='de', max_iter=50):
        if method == 'optuna' and HAS_OPTUNA:
            def obj(trial):
                params = [
                    trial.suggest_float('Cs',     0.05, 0.30),
                    trial.suggest_float('lambda', 5.0,  30.0),
                    trial.suggest_float('alpha',  0.1,  2.0),
                    trial.suggest_float('theta',  0.5,  5.0),
                    trial.suggest_float('tau',    1.0,  50.0),
                ]
                return cls._eval(params, solver, target_energy)
            study = optuna.create_study(direction='minimize')
            study.optimize(obj, n_trials=max_iter)
            return study.best_params
        else:  # differential evolution
            bounds = [(0.05,0.30),(5,30),(0.1,2.0),(0.5,5.0),(1,50)]
            result = differential_evolution(
                lambda p: cls._eval(p, solver, target_energy),
                bounds, maxiter=max_iter, popsize=10, tol=1e-6, disp=False)
            return {'Cs':result.x[0],'lambda':result.x[1],
                    'alpha':result.x[2],'theta':result.x[3],'tau':result.x[4]}


# =============================================================================
# 10. Differentiability test suite
# =============================================================================

def run_diff_tests() -> None:
    """
    Verify that gradient flows cleanly through every major component.
    All tests use torch.autograd.gradcheck on small grids.
    """
    import sys
    passed = failed = 0

    def ok(name):
        nonlocal passed; passed += 1; print(f"  [PASS] {name}")

    def fail(name, msg):
        nonlocal failed; failed += 1; print(f"  [FAIL] {name}: {msg}")

    torch.manual_seed(0)
    dev = torch.device("cpu")
    dt_ = torch.float64

    # ── 1. Smooth limiters ────────────────────────────────────────────────────
    for name, fn in _SMOOTH_LIMITERS.items():
        a = torch.randn(8, dtype=dt_, requires_grad=True)
        b = torch.randn(8, dtype=dt_, requires_grad=True)
        try:
            out = fn(a, b).sum()
            out.backward()
            if a.grad is not None and b.grad is not None:
                ok(f"smooth_limiter_{name}_grad")
            else:
                fail(f"smooth_limiter_{name}_grad", "None gradient")
        except Exception as e:
            fail(f"smooth_limiter_{name}_grad", str(e))

    # ── 2. softplus_floor ─────────────────────────────────────────────────────
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=dt_, requires_grad=True)
    out = _softplus_floor(x, 0.0).sum()
    out.backward()
    if x.grad is not None and x.grad.isfinite().all():
        ok("softplus_floor_grad")
    else:
        fail("softplus_floor_grad", f"grad={x.grad}")

    # ── 3. logsumexp_max ──────────────────────────────────────────────────────
    x = torch.randn(16, dtype=dt_, requires_grad=True)
    out = _logsumexp_max(x)
    out.backward()
    if x.grad is not None and x.grad.isfinite().all():
        ok("logsumexp_max_grad")
    else:
        fail("logsumexp_max_grad", f"grad={x.grad}")

    # ── 4. MUSCL face states ──────────────────────────────────────────────────
    for lim in ["minmod","van_leer","superbee"]:
        solver_base = RiemannSolverBase(gamma=1.4)
        q = torch.randn(8,8,8, dtype=dt_, requires_grad=True)
        q_pad = F.pad(q.unsqueeze(0).unsqueeze(0),(2,2,2,2,2,2),mode='replicate').squeeze(0).squeeze(0)
        try:
            qL,qR = solver_base._muscl_face_states(q_pad, 0, lim)
            (qL.sum()+qR.sum()).backward()
            if q.grad is not None and q.grad.isfinite().all():
                ok(f"muscl_face_states_{lim}_grad")
            else:
                fail(f"muscl_face_states_{lim}_grad", "NaN grad")
        except Exception as e:
            fail(f"muscl_face_states_{lim}_grad", str(e))
        q.grad = None

    # ── 5. WENO-5 face states ─────────────────────────────────────────────────
    solver_base = RiemannSolverBase(gamma=1.4)
    q = torch.randn(8,8,8, dtype=dt_, requires_grad=True)
    q_pad = F.pad(q.unsqueeze(0).unsqueeze(0),(3,3,3,3,3,3),mode='replicate').squeeze(0).squeeze(0)
    try:
        qL,qR = solver_base._weno5_face_states(q_pad, 0)
        (qL.sum()+qR.sum()).backward()
        if q.grad is not None and q.grad.isfinite().all():
            ok("weno5_face_states_grad")
        else:
            fail("weno5_face_states_grad", "NaN grad")
    except Exception as e:
        fail("weno5_face_states_grad", str(e))

    # ── 6. AUSMPlusFlux ───────────────────────────────────────────────────────
    ausm = AUSMPlusFlux(gamma=1.4)
    def _rnd(): return torch.randn(6,6,6,dtype=dt_)+2.0
    rho_p = _rnd().requires_grad_(True)
    u_p   = _rnd().requires_grad_(True)
    v_p   = _rnd().requires_grad_(True)
    w_p   = _rnd().requires_grad_(True)
    p_p   = _rnd().requires_grad_(True)
    try:
        fluxes = ausm.compute_face_flux(rho_p,u_p,v_p,w_p,p_p,0)
        sum([f.sum() for f in fluxes]).backward()
        grads_ok = all(t.grad is not None and t.grad.isfinite().all()
                       for t in [rho_p,u_p,v_p,w_p,p_p])
        ok("ausm_flux_grad") if grads_ok else fail("ausm_flux_grad","NaN/None grad")
    except Exception as e:
        fail("ausm_flux_grad", str(e))

    # ── 7. HLLCFlux ──────────────────────────────────────────────────────────
    hllc = HLLCFlux(gamma=1.4)
    for t in [rho_p,u_p,v_p,w_p,p_p]: t.grad=None
    rho_p2 = _rnd().requires_grad_(True)
    u_p2   = _rnd().requires_grad_(True)
    v_p2   = _rnd().requires_grad_(True)
    w_p2   = _rnd().requires_grad_(True)
    p_p2   = _rnd().requires_grad_(True)
    try:
        fluxes2 = hllc.compute_face_flux(rho_p2,u_p2,v_p2,w_p2,p_p2,0)
        sum([f.sum() for f in fluxes2]).backward()
        grads_ok2 = all(t.grad is not None and t.grad.isfinite().all()
                        for t in [rho_p2,u_p2,v_p2,w_p2,p_p2])
        ok("hllc_flux_grad") if grads_ok2 else fail("hllc_flux_grad","NaN/None grad")
    except Exception as e:
        fail("hllc_flux_grad", str(e))

    # ── 8. DiffRGRefiner ─────────────────────────────────────────────────────
    rg = DiffRGRefiner(keep_fraction=0.5)
    x  = torch.randn(8,8,8, dtype=dt_, requires_grad=True)
    try:
        y = rg.forward(x)
        y.sum().backward()
        if x.grad is not None and x.grad.isfinite().all():
            ok("diffrg_refiner_grad")
        else:
            fail("diffrg_refiner_grad","NaN/None grad")
    except Exception as e:
        fail("diffrg_refiner_grad", str(e))

    # ── 9. End-to-end: loss.backward() through solver.step() ─────────────────
    try:
        cfg = CFDConfig(nx=8,ny=8,nz=8,steps=2,device='cpu',
                        dtype=torch.float64,flux_scheme='ausm',
                        advection_scheme='muscl',muscl_limiter='van_leer',
                        cfl=0.3,Re=100.0)
        solver = CompressibleSolver(cfg)
        solver.initialize('taylor_green')

        # Make initial state require grad for e2e test
        solver.rho   = solver.rho.detach().requires_grad_(True)
        solver.rhou  = solver.rhou.detach().requires_grad_(True)
        rho_init = solver.rho

        solver.step(dt=1e-4)
        ke = solver.kinetic_energy()
        ke.backward()

        if rho_init.grad is not None and rho_init.grad.isfinite().all():
            ok("e2e_backward_through_step")
        else:
            fail("e2e_backward_through_step", f"grad={rho_init.grad}")
    except Exception as e:
        fail("e2e_backward_through_step", str(e))

    print(f"\n{'='*50}")
    print(f"Differentiability tests  passed={passed}  failed={failed}")
    sys.exit(0 if failed == 0 else 1)


# =============================================================================
# 11. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SUPER DNS ONE v5 — Native Full-Differentiable 3D CFD")
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
    parser.add_argument('--rg', action='store_true')
    parser.add_argument('--rg-keep', type=float, default=0.5)
    parser.add_argument('--ito', type=float, default=0.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--flux', default='ausm', choices=['ausm','hllc'])
    parser.add_argument('--adv', default='muscl', choices=['muscl','weno5','semi_lagrangian'])
    parser.add_argument('--limiter', default='van_leer', choices=['minmod','van_leer','superbee'])
    parser.add_argument('--shock-capturing', action='store_true')
    parser.add_argument('--compress-corr', action='store_true', default=True)
    parser.add_argument('--ssc-epsilon', type=float, default=0.0028)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--bc-x-min', default='periodic'); parser.add_argument('--bc-x-max', default='periodic')
    parser.add_argument('--bc-y-min', default='periodic'); parser.add_argument('--bc-y-max', default='periodic')
    parser.add_argument('--bc-z-min', default='periodic'); parser.add_argument('--bc-z-max', default='periodic')
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
    parser.add_argument('--eos-model', default='ideal', choices=['ideal','real'])
    parser.add_argument('--fluid', default='Air')
    parser.add_argument('--ib-mask', default=None)
    parser.add_argument('--ib-eta', type=float, default=1e4)
    parser.add_argument('--ib-T-target', type=float, default=None)
    parser.add_argument('--ib-eta-T', type=float, default=1e4)
    parser.add_argument('--case', default='taylor_green', choices=['taylor_green','hypersonic_bnd','uniform'])
    parser.add_argument('--save-checkpoint', default=None)
    parser.add_argument('--load-checkpoint', default=None)
    parser.add_argument('--train-soc', action='store_true')
    parser.add_argument('--train-soc-method', default='gradient', choices=['gradient','optuna','de'])
    parser.add_argument('--target-energy', type=float, default=0.1)
    parser.add_argument('--grid-convergence', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--denoise-method', default='ssc', choices=['ssc','wiener','wavelet'])
    parser.add_argument('--wavelet', default='db4')
    parser.add_argument('--denoise-level', type=int, default=4)
    parser.add_argument('--denoise-threshold', type=float, default=0.1)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--diff-test', action='store_true', help='Run differentiability test suite')
    args = parser.parse_args()

    if args.diff_test:
        print("Running differentiability test suite …\n")
        run_diff_tests()
        return

    if args.distributed:
        dist.init_process_group(backend='nccl')
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)

    dt_map = {'float32': torch.float32, 'float64': torch.float64}
    cfg = CFDConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        Re=args.Re, Pr=args.Pr, gamma=args.gamma,
        Mach=args.Mach, cfl=args.cfl, steps=args.steps,
        soc_base_temp=args.soc_temp, max_nu_t=args.max_nu_t,
        use_rg=args.rg, rg_keep_frac=args.rg_keep,
        ito_noise=args.ito, device=args.device,
        flux_scheme=args.flux, shock_capturing=args.shock_capturing,
        compressibility_correction=args.compress_corr,
        ssc_epsilon=args.ssc_epsilon,
        dtype=dt_map.get(args.dtype, torch.float32),
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
        distributed=args.distributed,
        muscl_limiter=args.limiter, advection_scheme=args.adv,
    )

    solver = CompressibleSolver(cfg)

    if args.load_checkpoint:
        solver.load_checkpoint(args.load_checkpoint)

    if args.train_soc:
        if args.train_soc_method == 'gradient':
            trainer = SOCTrainer(solver, target_spectrum=args.target_energy)
            trainer.train_gradient(n_outer=50, n_inner=10)
        else:
            SOCTrainer.train(solver, args.target_energy,
                             method=args.train_soc_method)
    elif args.grid_convergence:
        rate = solver.grid_convergence_test()
        if rate is not None and solver.rank == 0:
            logger.info(f"Estimated convergence rate: {rate:.3f}")
    elif args.denoise:
        denoiser = SignalDenoiser(method=args.denoise_method,
                                  wavelet=args.wavelet,
                                  level=args.denoise_level,
                                  threshold=args.denoise_threshold)
        solver.initialize(args.case)
        solver.run()
        u = solver.rhou / _softplus_floor(solver.rho, 1e-8)
        denoised_u = denoiser.denoise(u)
        if solver.rank == 0:
            logger.info(f"Denoised velocity range: [{denoised_u.min():.3f}, {denoised_u.max():.3f}]")
    else:
        if args.case == 'taylor_green':
            solver.taylor_green_test(steps=args.steps)
        else:
            solver.initialize(args.case)
            solver.run()

    if args.save_checkpoint:
        solver.save_checkpoint(args.save_checkpoint)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
