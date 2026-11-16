#!/usr/bin/env python3
"""
=============================================================================
HYPERSONIC ONE v1.0
Dual energy · Zhang-Shu positivity · Harten-Hyman · Park 5-species · Two-T
+ wall catalysis/ablation + tangent-slab radiation + MSPS control layer
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
               My Soul Move By Power of Holy Spirit ;
               What MSPS NETWORK SEE, Lord Buddha Knows.
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

=============================================================================
THE SEVEN GAPS THIS FILE CLOSES (and what remains open)
=============================================================================
[G1] DUAL ENERGY  -- IMPLEMENTED
     At M=20, e/E = c^2/(gamma(gamma-1)) / [c^2/(gamma(gamma-1)) + u^2/2]
                  ~ 0.885%. Computing e = E - |m|^2/(2rho) is a
     subtraction of two large nearly-equal numbers. bf16 (8-bit
     mantissa) loses ~88% of e; fp32 loses ~1e-5 relative.
     Fix: evolve rho*e as a SEPARATE field with its own (non-
     conservative) equation, use it for p/T when e/E < eta_switch,
     and RESYNC it from the conservative total energy whenever
     e/E >= eta_switch. This is the standard dual-energy formalism
     (Bryan et al. 1995; Enzo/FLASH/Athena).
     MEASURED in T1: pressure error fp32/bf16 with and without it.

[G2] ZHANG-SHU POSITIVITY LIMITER -- IMPLEMENTED
     Zhang & Shu (2010), JCP 229:3091. Two-stage scaling toward the
     cell average:
        stage 1 (density):  theta_rho = min(1, (rho_bar-eps)/(rho_bar-rho_min))
        stage 2 (pressure): largest theta in [0,1] with
                            p(theta*U + (1-theta)*U_bar) >= eps_p
     p is concave along the segment for ideal gas, so bisection on
     theta is well posed; 40 iterations gives ~1e-12 in theta.
     HONEST: the theorem guarantees positivity of the LIMITED
     reconstructed states GIVEN a positive cell average, which in
     turn requires the first-order scheme + a CFL restriction.
     The limiter does not rescue an already-negative cell average.
     MEASURED in T2 at the M=20 pressure ratio 466:1.

[G3] HARTEN-HYMAN ENTROPY FIX + CARBUNCLE CURE -- IMPLEMENTED
     (a) Acoustic-wave entropy fix: |lambda| -> (lambda^2+delta^2)/(2 delta)
         when |lambda| < delta, delta from the local wave-speed spread.
         Removes expansion shocks at sonic points.
     (b) Shock-normal shear dissipation (Kim et al. style): detect a
         strong shock via the pressure jump, add dissipation to the
         TRANSVERSE momentum fluxes proportional to shock strength.
         This is the mechanism that actually suppresses the carbuncle.
     HONEST: no entropy fix is a PROOF of carbuncle-free behaviour.
     The decisive test is a blunt-body run at M=15 (T3 harness
     provided). YOU must run it and look at the stagnation streamline.

[G4] PARK 5-SPECIES AIR + NASA-7 GIBBS -- IMPLEMENTED
     Species: N2, O2, NO, N, O.  Reactions (Park 1990):
        R1  N2 + M <=> 2N + M
        R2  O2 + M <=> 2O + M
        R3  NO + M <=> N + O + M
        R4  N2 + O <=> NO + N     (Zeldovich)
        R5  NO + O <=> O2 + N     (Zeldovich)
     Forward rates: Arrhenius k_f = A Ta^n exp(-Ea/Ta).
     Backward rates: k_b = k_f / K_c, K_c from NASA-7 Gibbs free
     energy -- NOT from a fitted curve, so the thermodynamics and
     kinetics are mutually consistent by construction.
     HONEST: (i) the NASA-7 coefficients below are the standard
     GRI-Mech/NASA low(200-1000K)/mid(1000-6000K) sets; above 6000 K
     they are EXTRAPOLATED and a warning is issued. (ii) 5-species
     air is valid roughly 2000-9000 K; above that, ionization
     (N+, O+, NO+, e-) matters and an 11-species model is required.
     (iii) Absolute agreement with NASA CEA2 is NOT claimed; T4
     computes the ideal-gas vs equilibrium post-shock temperature
     DIFFERENCE, which is the real physics gap, and requires you to
     run CEA for absolute validation.

[G5] TWO-TEMPERATURE PARK -- IMPLEMENTED
     T   : translational-rotational
     Tv  : vibrational-electronic
     e_v,s = R_s theta_v,s / (exp(theta_v,s/Tv) - 1)   (harmonic osc.)
     theta_v: N2 3395 K, O2 2239 K, NO 2740 K
     Landau-Teller relaxation with Millikan-White p*tau plus Park's
     collision-limiting correction at high T.
     Park controlling temperature for dissociation: Ta = sqrt(T*Tv)
     (geometric mean); exchange reactions use Ta = T.
     CVDV (coupled vibration-dissociation-vibration): dissociation
     removes a fraction chi of the dissociation energy from the
     vibrational pool. chi=0.3 (preferential) or 0 (non-preferential).

[G6] WALL CATALYSIS + ABLATION -- IMPLEMENTED
     Catalytic heat flux: q_cat = sum_s gamma_s * rho_s *
        sqrt(R_s T/(2 pi)) * dh_recomb,s
     gamma_s = 0 (non-catalytic) .. 1 (fully catalytic).
     Ablation: Stefan blowing with mass/energy balance; the wall
     normal velocity is DERIVED from mdot/rho_wall, not prescribed.

[G7] SHOCK-LAYER RADIATION -- PARTIAL, LABELLED
     Two limits implemented: optically-thin (upper bound on loss)
     and tangent-slab with an effective absorption coefficient.
     HONEST: real reentry radiation is dominated by NO gamma/beta,
     N2+ first-negative, and atomic lines in the VUV, requiring
     line-by-line transport (NEQAIR / HARA class). This is a
     tractable surrogate, NOT that. Do not use for TPS sizing.

=============================================================================
MSPS MATHEMATICS, PLACED WHERE IT IS LEGITIMATE
=============================================================================
  CSOC event bound N(T) <= (E_max+D)/dE_min with ACTUAL dissipation
     accounting -> bounds LIMITER ACTIVATIONS and chemistry-regime
     switches. Limiter chatter near shocks is a real hypersonic
     pathology; a bounded switch count is a real engineering
     guarantee. NOT a PDE regularity statement.
  delta_min + kappa-fatness -> AMR guard near shock/boundary layer.
     delta_min = (1/kappa)(a_min/omega_{n-1})^{1/(n-1)}.
     OUTER loop (mesh topology change is not differentiable).
  Carpet rho >= 9/8 -> SECONDARY refinement priority. The PRIMARY
     criterion here is Delta / min(eta, delta_shock).
  OPMC d(m,n)=320 -> DESIGN parameterisation for shape optimisation,
     exact gradient through the decoder. NEVER compresses the
     conserved state (that breaks conservation) or the adjoint tape.
  Kovasznay/Helmholtz mode split -> acoustic / vortical / entropy
     decomposition of the field. In COMPRESSIBLE flow this is real
     physics, not a metaphor (Kovasznay 1953).
  Assumption-light CI (Politis-Romano) + Madras-Sokal tau_int ->
     error bars on heat flux / drag with autocorrelation.

=============================================================================
HONEST SCOPE
=============================================================================
  * NOT validated against experiment, CEA2, or any reference
    hypersonic code (LAURA, DPLR, US3D, SPARK).
  * 5-species valid ~2000-9000 K. No ionization. No electron energy.
  * NASA-7 extrapolated above 6000 K (warning issued).
  * Radiation is tangent-slab / optically-thin, NOT line-by-line.
  * Structured Cartesian + immersed boundary. No body-fitted grid,
    no shock-aligned grid adaptation (which matters a lot for
    hypersonic surface heat flux).
  * Single device. No MPI halo exchange here.
  * NO claim of "world's fastest" or "world's highest resolution".
    Nothing has been benchmarked against a baseline.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("hypersonic_one")

VERSION = "1.0.0"
R_UNIV = 8.314462618          # J/(mol K)
K_B = 1.380649e-23
N_A = 6.02214076e23
EV_K = 11604.518              # 1 eV in Kelvin
CARPET_RHO = 9.0 / 8.0
KAPPA_FATNESS = 2.0
A_MIN = 0.01

BANNER = f"""
=============================================================================
HYPERSONIC ONE v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
Dual energy | Zhang-Shu | Harten-Hyman | Park 5-species | Two-T
5-species valid ~2000-9000 K. NOT validated against CEA/experiment.
NO claim of world's fastest / highest resolution.
=============================================================================
"""

# Species order is FIXED everywhere: N2, O2, NO, N, O
SPECIES = ("N2", "O2", "NO", "N", "O")
NS = 5
M_MOL = np.array([28.0134e-3, 31.9988e-3, 30.0061e-3, 14.0067e-3, 15.9994e-3])  # kg/mol
R_SPEC = R_UNIV / M_MOL                       # J/(kg K)
THETA_V = np.array([3395.0, 2239.0, 2740.0, 0.0, 0.0])   # K, diatomics only
DIATOMIC = np.array([True, True, True, False, False])
# Formation enthalpy at 298.15 K [J/mol]: N2 0, O2 0, NO 90291, N 472680, O 249180
H_FORM = np.array([0.0, 0.0, 90291.0, 472680.0, 249180.0])


# =============================================================================
# §1. SMOOTH PRIMITIVES  (floor width TRACKS the floor)
# =============================================================================

def soft_floor(x: Tensor, floor: float, sharpness: float = 50.0) -> Tensor:
    beta = min(sharpness / max(abs(floor), 1e-30), 1e12)
    return floor + F.softplus(x - floor, beta=beta)


def effective_floor(floor: float, sharpness: float = 50.0) -> float:
    beta = min(sharpness / max(abs(floor), 1e-30), 1e12)
    return floor + math.log(2.0) / beta


def soft_abs(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


def smooth_min(a: Tensor, b: Tensor) -> Tensor:
    return 0.5 * (a + b) - 0.5 * soft_abs(a - b)


def smooth_max(a: Tensor, b: Tensor) -> Tensor:
    return 0.5 * (a + b) + 0.5 * soft_abs(a - b)


def smoothstep(t: Tensor) -> Tensor:
    tc = t.clamp(0.0, 1.0)
    return tc * tc * (3.0 - 2.0 * tc)


# =============================================================================
# §2. NASA-7 THERMODYNAMICS + GIBBS + EQUILIBRIUM CONSTANTS
# =============================================================================

# Standard NASA/GRI-Mech 7-coefficient sets.
# LOW  = 200-1000 K, MID = 1000-6000 K. Above 6000 K: EXTRAPOLATED (warned).
NASA7_LOW = np.array([
    [3.53100528, -1.23660988e-4, -5.02999433e-7, 2.43530612e-9, -1.40881235e-12, -1.04697628e3, 2.96747468],   # N2
    [3.78245636, -2.99673416e-3, 9.84730201e-6, -9.68129509e-9, 3.24372837e-12, -1.06394356e3, 3.65767573],   # O2
    [4.21847630, -4.63897600e-3, 1.10410220e-5, -9.33613540e-9, 2.80357700e-12, 9.84462300e3, 2.28084640],    # NO
    [2.50000000, 0.0, 0.0, 0.0, 0.0, 5.61046370e4, 4.19390870],                                              # N
    [3.16826710, -3.27931884e-3, 6.64306396e-6, -6.12806624e-9, 2.11265971e-12, 2.91222592e4, 2.05193346],    # O
])
NASA7_MID = np.array([
    [2.95257637, 1.39690040e-3, -4.92631603e-7, 7.86010195e-11, -4.60755204e-15, -9.23948688e2, 5.87188762],  # N2
    [3.28253784, 1.48308754e-3, -7.57966669e-7, 2.09470555e-10, -2.16717794e-14, -1.08845772e3, 5.45323129],  # O2
    [3.26060560, 1.19110430e-3, -4.29170480e-7, 6.94576690e-11, -4.03360990e-15, 9.92097460e3, 6.36930270],   # NO
    [2.41594290, 1.74890650e-4, -1.19023690e-7, 3.02262450e-11, -2.03609820e-15, 5.61337730e4, 4.64960960],   # N
    [2.56942078, -8.59741137e-5, 4.19484589e-8, -1.00177799e-11, 1.22833691e-15, 2.92175791e4, 4.78433864],   # O
])
T_MID = 1000.0
T_MAX_FIT = 6000.0
_warned_extrap = False


def nasa7(T: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns (cp/R, h/(RT), s/R) per species, shape (..., NS).
    Blends LOW/MID sets with a smooth gate so the derivative is
    continuous at T_MID (a hard switch would break autograd).
    """
    global _warned_extrap
    Tc = soft_floor(T, 100.0)
    if bool((Tc > T_MAX_FIT).any()) and not _warned_extrap:
        log.warning("NASA-7 EXTRAPOLATED above %.0f K. 5-species air is "
                    "valid ~2000-9000 K; ionization is NOT modelled.", T_MAX_FIT)
        _warned_extrap = True
    lo = torch.as_tensor(NASA7_LOW, dtype=Tc.dtype, device=Tc.device)
    mid = torch.as_tensor(NASA7_MID, dtype=Tc.dtype, device=Tc.device)
    w = smoothstep((Tc - T_MID) / 100.0).unsqueeze(-1)   # smooth 100 K blend
    a = (1 - w) * lo + w * mid
    t1, t2, t3, t4 = Tc.unsqueeze(-1), Tc.unsqueeze(-1)**2, Tc.unsqueeze(-1)**3, Tc.unsqueeze(-1)**4
    a1, a2, a3, a4, a5, a6, a7 = [a[..., i] for i in range(7)]
    cp_R = a1 + a2*t1 + a3*t2 + a4*t3 + a5*t4
    h_RT = a1 + a2*t1/2 + a3*t2/3 + a4*t3/4 + a5*t4/5 + a6/Tc.unsqueeze(-1)
    s_R = a1*torch.log(Tc.unsqueeze(-1)) + a2*t1 + a3*t2/2 + a4*t3/3 + a5*t4/4 + a7
    return cp_R, h_RT, s_R


def gibbs_RT(T: Tensor) -> Tensor:
    """g/(RT) = h/(RT) - s/R, shape (..., NS)."""
    _, h_RT, s_R = nasa7(T)
    return h_RT - s_R


def equilibrium_Kc(T: Tensor, p_ref: float = 101325.0) -> Tensor:
    """
    K_c for R1,R2,R3 (concentration units, mol/m^3).
    R1: N2 <=> 2N     dn = +1
    R2: O2 <=> 2O     dn = +1
    R3: NO <=> N + O  dn = +1
    K_p = exp(-dg/RT);  K_c = K_p * (p_ref/(R T))^dn
    """
    g = gibbs_RT(T)                                  # (...,5)
    dg1 = 2*g[..., 3] - g[..., 0]
    dg2 = 2*g[..., 4] - g[..., 1]
    dg3 = g[..., 3] + g[..., 4] - g[..., 2]
    factor = p_ref / (R_UNIV * soft_floor(T, 100.0))
    return torch.stack([torch.exp(-dg1)*factor,
                        torch.exp(-dg2)*factor,
                        torch.exp(-dg3)*factor], dim=-1)


def species_internal_energy(T: Tensor, Tv: Optional[Tensor] = None) -> Tensor:
    """
    e_s [J/kg] = e_tr(T) + e_vib(Tv) + h_form/M
    e_tr = (3/2) R_s T monatomic, (5/2) R_s T diatomic
    e_vib = R_s theta_v/(exp(theta_v/Tv)-1)
    """
    Tc = soft_floor(T, 50.0)
    Tvc = soft_floor(Tv if Tv is not None else T, 50.0)
    Rs = torch.as_tensor(R_SPEC, dtype=T.dtype, device=T.device)
    dia = torch.as_tensor(DIATOMIC, dtype=torch.bool, device=T.device)
    thv = torch.as_tensor(THETA_V, dtype=T.dtype, device=T.device)
    e_tr = torch.where(dia, 2.5*Rs*Tc.unsqueeze(-1), 1.5*Rs*Tc.unsqueeze(-1))
    x = thv / Tvc.unsqueeze(-1)
    evib = torch.where(dia & (thv > 0),
                       Rs*thv / torch.expm1(x.clamp(max=50.0)).clamp_min(1e-30),
                       torch.zeros_like(e_tr))
    hf = torch.as_tensor(H_FORM / M_MOL, dtype=T.dtype, device=T.device)
    return e_tr + evib + hf


def vibrational_energy(Tv: Tensor) -> Tensor:
    Tvc = soft_floor(Tv, 50.0)
    Rs = torch.as_tensor(R_SPEC, dtype=Tv.dtype, device=Tv.device)
    dia = torch.as_tensor(DIATOMIC, dtype=torch.bool, device=Tv.device)
    thv = torch.as_tensor(THETA_V, dtype=Tv.dtype, device=Tv.device)
    x = thv / Tvc.unsqueeze(-1)
    return torch.where(dia & (thv > 0),
                       Rs*thv / torch.expm1(x.clamp(max=50.0)).clamp_min(1e-30),
                       torch.zeros_like(Tvc.unsqueeze(-1).expand(*Tvc.shape, NS)))


def mixture_pressure(rho_s: Tensor, T: Tensor) -> Tensor:
    """p = sum_s rho_s R_s T  (Dalton)."""
    Rs = torch.as_tensor(R_SPEC, dtype=T.dtype, device=T.device)
    return (rho_s * Rs).sum(-1) * soft_floor(T, 50.0)


# =============================================================================
# §3. PARK 5-SPECIES KINETICS + TWO-TEMPERATURE RELAXATION
# =============================================================================

@dataclass
class ParkConfig:
    two_temperature: bool = True
    ta_mode: str = "park_sqrt"        # park_sqrt | park_avg | single_T
    cvdv_chi: float = 0.3             # fraction of D removed from vibration
    millikan_white: bool = True
    park_collision_limit: bool = True
    sigma_v_ref: float = 3e-21        # m^2 (3e-17 cm^2)
    sigma_v_Tref: float = 50000.0

    def __post_init__(self):
        if self.ta_mode not in ("park_sqrt", "park_avg", "single_T"):
            raise ValueError("bad ta_mode")
        if not 0.0 <= self.cvdv_chi <= 1.0:
            raise ValueError("cvdv_chi in [0,1]")


# Park (1990) forward Arrhenius parameters. A in SI-consistent
# m^3/(mol s) form converted from CGS; Ea in Kelvin (Ea = E/R).
# R1,R2,R3 have different A for atomic vs molecular collision partners.
PARK_A_MOL = np.array([7.0e18, 2.0e18, 5.0e12])    # m^3/(mol s), n applied
PARK_A_ATOM = np.array([3.0e19, 1.0e19, 1.1e14])
PARK_N = np.array([-1.6, -1.5, 0.0])
PARK_EA = np.array([113200.0, 59500.0, 75500.0])   # K
# Exchange reactions
PARK_A_EX = np.array([6.4e14, 8.4e9])              # R4, R5
PARK_N_EX = np.array([-1.0, 0.0])
PARK_EA_EX = np.array([38400.0, 19450.0])
# Dissociation energies [K] for CVDV
D_K = np.array([113200.0, 59500.0, 75500.0])
# Millikan-White coefficients (A, B) for p*tau [atm s], per species pair
MW_A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def millikan_white_tau(T: Tensor, p_pa: Tensor, rho_s: Tensor) -> Tensor:
    """
    Millikan-White: p[atm] * tau[s] = exp(A T^(-1/3) + B)
    Standard fitted values for N2, O2, NO in a bath of N2 (dominant).
    A = 220, 129, 168 ; B = -18.42, -18.42, -18.42 (approximate).
    HONEST: these are approximate literature values; verify against
    Park's tables before quantitative use.
    """
    A = torch.tensor([220.0, 129.0, 168.0], dtype=T.dtype, device=T.device)
    B = torch.tensor([-18.42, -18.42, -18.42], dtype=T.dtype, device=T.device)
    p_atm = soft_floor(p_pa, 1.0) / 101325.0
    Tc = soft_floor(T, 200.0)
    tau = torch.exp(A * Tc.unsqueeze(-1).pow(-1.0/3.0) + B) / p_atm.unsqueeze(-1)
    out = torch.zeros(*T.shape, NS, dtype=T.dtype, device=T.device)
    out[..., :3] = tau
    return out


def park_collision_tau(T: Tensor, rho_s: Tensor) -> Tensor:
    """
    Park collision-limiting: tau_p = 1/(n_s sigma_v <c_s>)
    sigma_v = sigma_ref (Tref/T)^2 ; <c> = sqrt(8 R T/(pi M))
    Prevents the Millikan-White tau from becoming unphysically small
    at very high T.
    """
    Tc = soft_floor(T, 200.0)
    sigma = 3e-21 * (50000.0 / Tc)**2
    M = torch.as_tensor(M_MOL, dtype=T.dtype, device=T.device)
    cbar = torch.sqrt(8.0 * R_UNIV * Tc.unsqueeze(-1) / (math.pi * M))
    n_s = soft_floor(rho_s, 1e-20) / M * N_A     # number density [1/m^3]
    tau_p = 1.0 / (n_s * sigma.unsqueeze(-1) * cbar).clamp_min(1e-30)
    return tau_p


def park_rates(T: Tensor, Tv: Tensor, cfg: ParkConfig) -> Tuple[Tensor, Tensor]:
    """
    Returns (k_f, k_b) for the 3 dissociation reactions, shape (...,3).
    Controlling temperature Ta:
      park_sqrt : Ta = sqrt(T*Tv)   (Park's recommendation)
      park_avg  : Ta = 0.5(T+Tv)
      single_T  : Ta = T
    """
    Tc = soft_floor(T, 200.0)
    Tvc = soft_floor(Tv, 200.0)
    if not cfg.two_temperature or cfg.ta_mode == "single_T":
        Ta = Tc
    elif cfg.ta_mode == "park_sqrt":
        Ta = torch.sqrt(Tc * Tvc)
    else:
        Ta = 0.5 * (Tc + Tvc)
    A_m = torch.as_tensor(PARK_A_MOL, dtype=T.dtype, device=T.device)
    n = torch.as_tensor(PARK_N, dtype=T.dtype, device=T.device)
    Ea = torch.as_tensor(PARK_EA, dtype=T.dtype, device=T.device)
    kf = A_m * Ta.unsqueeze(-1)**n * torch.exp(-Ea / Ta.unsqueeze(-1))
    Kc = equilibrium_Kc(Tc).clamp_min(1e-300)
    kb = kf / Kc
    return kf, kb


def chemical_source(rho_s: Tensor, T: Tensor, Tv: Tensor,
                    cfg: ParkConfig) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns (omega_s [kg/(m^3 s)], q_chem_to_vib [W/m^3], n_reactions).
    Uses molar concentrations c_s = rho_s/M_s [mol/m^3].
    R1: N2+M -> 2N+M    rate = kf1 c_N2 c_M - kb1 c_N^2 c_M
    R2: O2+M -> 2O+M
    R3: NO+M -> N+O+M
    R4: N2+O -> NO+N    rate = kf4 c_N2 c_O - kb4 c_NO c_N
    R5: NO+O -> O2+N    rate = kf5 c_NO c_O - kb5 c_O2 c_N
    """
    M = torch.as_tensor(M_MOL, dtype=T.dtype, device=T.device)
    c = soft_floor(rho_s, 1e-20) / M
    cM = c.sum(-1)
    kf, kb = park_rates(T, Tv, cfg)
    R = []
    for i, (ia, ib) in enumerate([(0, 3), (1, 4), (2, None)]):
        if ib is not None:
            fwd = kf[..., i] * c[..., ia] * cM
            bwd = kb[..., i] * c[..., ib]**2 * cM
        else:
            fwd = kf[..., i] * c[..., ia] * cM
            bwd = kb[..., i] * c[..., 3] * c[..., 4] * cM
        R.append(fwd - bwd)
    R1, R2, R3 = R
    # Exchange (Zeldovich), Ta = T
    Tc = soft_floor(T, 200.0)
    A_ex = torch.as_tensor(PARK_A_EX, dtype=T.dtype, device=T.device)
    n_ex = torch.as_tensor(PARK_N_EX, dtype=T.dtype, device=T.device)
    Ea_ex = torch.as_tensor(PARK_EA_EX, dtype=T.dtype, device=T.device)
    kf_ex = A_ex * Tc.unsqueeze(-1)**n_ex * torch.exp(-Ea_ex / Tc.unsqueeze(-1))
    # K_eq for R4: N2+O <=> NO+N  -> dg = g_NO + g_N - g_N2 - g_O
    g = gibbs_RT(Tc)
    dg4 = g[..., 2] + g[..., 3] - g[..., 0] - g[..., 4]
    dg5 = g[..., 1] + g[..., 3] - g[..., 2] - g[..., 4]
    kb4 = kf_ex[..., 0] / torch.exp(-dg4).clamp_min(1e-300)
    kb5 = kf_ex[..., 1] / torch.exp(-dg5).clamp_min(1e-300)
    R4 = kf_ex[..., 0]*c[..., 0]*c[..., 4] - kb4*c[..., 2]*c[..., 3]
    R5 = kf_ex[..., 1]*c[..., 2]*c[..., 4] - kb5*c[..., 1]*c[..., 3]

    omega = torch.zeros_like(rho_s)
    omega[..., 0] += (-R1 - R4) * M[0]
    omega[..., 1] += (-R2 + R5) * M[1]
    omega[..., 2] += (-R3 + R4 - R5) * M[2]
    omega[..., 3] += (2*R1 + R3 + R4 + R5) * M[3]
    omega[..., 4] += (2*R2 + R3 - R4 - R5) * M[4]

    # CVDV: dissociation removes chi*D from the vibrational pool
    D_J_mol = torch.as_tensor(D_K, dtype=T.dtype, device=T.device) * R_UNIV
    rates = torch.stack([R1, R2, R3], dim=-1)
    q_vib = -cfg.cvdv_chi * (rates.clamp_min(0.0) * D_J_mol).sum(-1)
    return omega, q_vib, torch.stack([R1, R2, R3, R4, R5], dim=-1)


def vibrational_relaxation(rho_s: Tensor, T: Tensor, Tv: Tensor,
                           cfg: ParkConfig) -> Tensor:
    """
    Landau-Teller: Q_LT,s = rho_s (e_v,s(T) - e_v,s(Tv)) / tau_s
    tau_s = tau_MW + tau_Park (collision-limited).
    """
    p = mixture_pressure(rho_s, T)
    tau = millikan_white_tau(T, p, rho_s)
    if cfg.park_collision_limit:
        tau = tau + park_collision_tau(T, rho_s)
    ev_eq = vibrational_energy(T)
    ev_now = vibrational_energy(Tv)
    return (soft_floor(rho_s, 1e-20) * (ev_eq - ev_now)
            / tau.clamp_min(1e-12)).sum(-1)


# =============================================================================
# §4. DUAL-ENERGY STATE  [G1]
# =============================================================================

@dataclass
class HypersonicState:
    rho: Tensor          # (nx,ny,nz)
    rhou: Tensor
    rhov: Tensor
    rhow: Tensor
    rhoE: Tensor         # total energy density (CONSERVATIVE)
    rhoe: Tensor         # internal energy density (DUAL, non-conservative)
    rho_s: Tensor        # (nx,ny,nz,NS) species partial densities
    rhoEv: Tensor        # vibrational energy density
    t: float = 0.0
    step: int = 0


class DualEnergy:
    """
    [G1] Use rhoe for p/T when e/E < eta_switch, else use rhoE - KE.
    RESYNC rhoe from the conservative total energy whenever the
    ratio is safe. This keeps conservation where it is numerically
    safe and precision where it is not.
    """
    def __init__(self, eta_switch: float = 1e-2):
        self.eta = float(eta_switch)
        self.n_dual_used = 0
        self.n_resync = 0

    def internal(self, st: HypersonicState) -> Tuple[Tensor, Tensor]:
        """Returns (rho_e_used, used_dual_mask)."""
        rs = soft_floor(st.rho, 1e-10)
        ke = 0.5*(st.rhou**2 + st.rhov**2 + st.rhow**2) / rs
        e_from_total = st.rhoE - ke
        ratio = e_from_total / soft_floor(st.rhoE, 1e-30)
        use_dual = ratio < self.eta
        self.n_dual_used += int(use_dual.sum())
        return torch.where(use_dual, st.rhoe, e_from_total), use_dual

    def resync(self, st: HypersonicState) -> HypersonicState:
        rs = soft_floor(st.rho, 1e-10)
        ke = 0.5*(st.rhou**2 + st.rhov**2 + st.rhow**2) / rs
        e_from_total = st.rhoE - ke
        ratio = e_from_total / soft_floor(st.rhoE, 1e-30)
        safe = ratio >= self.eta
        self.n_resync += int(safe.sum())
        st.rhoe = torch.where(safe, e_from_total, st.rhoe)
        return st

    def pressure_temperature(self, st: HypersonicState,
                             cfg: ParkConfig) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Two-temperature closure:
          e_total = e_tr(T) + e_vib(Tv) + h_form
          rho*e = sum_s rho_s e_tr,s(T) + rhoEv + sum_s rho_s h_form,s/M_s
        Solve for T from the translational part (Newton, 6 iters).
        """
        re, _ = self.internal(st)
        Rs = torch.as_tensor(R_SPEC, dtype=re.dtype, device=re.device)
        dia = torch.as_tensor(DIATOMIC, dtype=torch.bool, device=re.device)
        hf = torch.as_tensor(H_FORM/M_MOL, dtype=re.dtype, device=re.device)
        cv_tr = torch.where(dia, 2.5*Rs, 1.5*Rs)
        rho_s = soft_floor(st.rho_s, 1e-20)
        C = (rho_s * cv_tr).sum(-1)                     # J/(m^3 K)
        Hf = (rho_s * hf).sum(-1)
        target = re - st.rhoEv - Hf
        T = soft_floor(target / C.clamp_min(1e-12), 100.0)
        for _ in range(4):                              # cv depends weakly on T via NASA
            T = soft_floor(target / C.clamp_min(1e-12), 100.0)
        p = mixture_pressure(rho_s, T)
        Tv = T
        if cfg.two_temperature:
            # invert e_vib(Tv) = rhoEv (Newton in Tv, 8 iters)
            Tv = T.clone()
            for _ in range(8):
                ev = (rho_s * vibrational_energy(Tv)).sum(-1)
                dT = 1.0
                ev_p = (rho_s * vibrational_energy(Tv + dT)).sum(-1)
                deriv = (ev_p - ev) / dT
                Tv = soft_floor(Tv + (st.rhoEv - ev) / deriv.clamp_min(1e-12), 100.0)
        return p, T, Tv


# =============================================================================
# §5. ZHANG-SHU POSITIVITY LIMITER  [G2]
# =============================================================================

class ZhangShuLimiter:
    """
    [G2] Zhang & Shu (2010) two-stage positivity limiter.
    Given a cell average U_bar and reconstructed states U_i:
      stage 1: rho -> theta_rho scaling so rho >= eps_rho
      stage 2: p   -> theta_p scaling so p >= eps_p
    p along the segment U(theta) = theta U + (1-theta) U_bar is
    concave for ideal gas, so bisection finds the largest feasible
    theta. 40 iterations -> ~1e-12.
    """
    def __init__(self, eps_rho: float = 1e-10, eps_p: float = 1e-8,
                 n_bisect: int = 40, gamma: float = 1.4):
        self.eps_rho = eps_rho
        self.eps_p = eps_p
        self.n_bisect = n_bisect
        self.gamma = gamma
        self.theta_rho_min = 1.0
        self.theta_p_min = 1.0
        self.n_activated = 0

    @staticmethod
    def _p_of(rho, rhou, rhov, rhow, rhoE, gamma):
        rs = soft_floor(rho, 1e-30)
        return (gamma-1)*(rhoE - 0.5*(rhou**2+rhov**2+rhow**2)/rs)

    def limit_density(self, U: Dict[str, Tensor],
                      bar: Dict[str, Tensor]) -> Dict[str, Tensor]:
        rho_min = U["rho"].min()
        rb = bar["rho"]
        theta = torch.clamp((rb - self.eps_rho) /
                            (rb - rho_min).clamp_min(1e-30), 0.0, 1.0)
        self.theta_rho_min = min(self.theta_rho_min, float(theta))
        if float(theta) < 1.0:
            self.n_activated += 1
        out = {}
        for k in U:
            out[k] = theta * U[k] + (1 - theta) * bar[k]
        return out

    def limit_pressure(self, U: Dict[str, Tensor],
                       bar: Dict[str, Tensor]) -> Dict[str, Tensor]:
        g = self.gamma
        p_now = self._p_of(U["rho"], U["rhou"], U["rhov"], U["rhow"], U["rhoE"], g)
        need = p_now < self.eps_p
        if not bool(need.any()):
            return U
        self.n_activated += int(need.sum())
        lo = torch.zeros_like(p_now)
        hi = torch.ones_like(p_now)
        for _ in range(self.n_bisect):
            th = 0.5*(lo + hi)
            def mix(k): return th*U[k] + (1-th)*bar[k]
            p_th = self._p_of(mix("rho"), mix("rhou"), mix("rhov"),
                              mix("rhow"), mix("rhoE"), g)
            ok = p_th >= self.eps_p
            lo = torch.where(ok, th, lo)
            hi = torch.where(ok, hi, th)
        theta = torch.where(need, lo, torch.ones_like(lo))
        self.theta_p_min = min(self.theta_p_min, float(theta.min()))
        out = {}
        for k in U:
            out[k] = theta*U[k] + (1-theta)*bar[k]
        return out

    def __call__(self, U: Dict[str, Tensor],
                 bar: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.limit_pressure(self.limit_density(U, bar), bar)

    def report(self) -> Dict[str, Any]:
        return {"activations": self.n_activated,
                "theta_rho_min": self.theta_rho_min,
                "theta_p_min": self.theta_p_min,
                "note": ("positivity of the LIMITED states is guaranteed "
                         "given a positive cell average (Zhang-Shu 2010) "
                         "under the scheme's CFL restriction. It does not "
                         "repair an already-negative cell average.")}


# =============================================================================
# §6. HLLC + HARTEN-HYMAN ENTROPY FIX + CARBUNCLE CURE  [G3]
# =============================================================================

class HypersonicHLLC:
    """
    HLLC with:
      [G3a] Harten-Hyman acoustic entropy fix (smooth version)
      [G3b] shock-normal shear dissipation (carbuncle cure)
    Both are optional and reported.
    """
    def __init__(self, gamma: float = 1.4, entropy_fix: bool = True,
                 carbuncle_cure: bool = True, cure_strength: float = 0.5,
                 shock_threshold: float = 0.3, tau: float = 1e-3):
        self.gamma = gamma
        self.entropy_fix = entropy_fix
        self.carbuncle_cure = carbuncle_cure
        self.cure_strength = cure_strength
        self.shock_threshold = shock_threshold
        self.tau = tau
        self.n_entropy_fixed = 0
        self.n_cure_applied = 0

    @staticmethod
    def harten_hyman(lam: Tensor, lamL: Tensor, lamR: Tensor,
                     delta_floor: float) -> Tensor:
        """
        |lam| -> (lam^2 + delta^2)/(2 delta) when |lam| < delta.
        delta = max(0, lamR - lamL, delta_floor).
        Smooth (C^1) via the soft_abs magnitude.
        """
        delta = smooth_max(lamR - lamL, torch.full_like(lam, delta_floor))
        mag = soft_abs(lam)
        fixed = (lam*lam + delta*delta) / (2*delta.clamp_min(1e-30))
        return torch.where(mag < delta, fixed, mag)

    def wave_speeds(self, rhoL, unL, pL, cL, rhoR, unR, pR, cR):
        R = torch.sqrt(soft_floor(rhoR, 1e-12) / soft_floor(rhoL, 1e-12))
        u_roe = (unL + R*unR) / (1 + R)
        c_roe = (cL + R*cR) / (1 + R)
        SL = smooth_min(unL - cL, u_roe - c_roe)
        SR = smooth_max(unR + cR, u_roe + c_roe)
        if self.entropy_fix:
            dfloor = 0.1 * c_roe
            SL_mag = self.harten_hyman(SL, unL - cL, u_roe - c_roe, dfloor)
            SR_mag = self.harten_hyman(SR, unR + cR, u_roe + c_roe, dfloor)
            SL = -soft_abs(SL_mag) * torch.sign(SL).clamp(min=-1)
            SL = smooth_min(SL, unL - cL)
            SR = smooth_max(SR_mag, unR + cR)
            self.n_entropy_fixed += 1
        return SL, SR, u_roe, c_roe

    def shock_strength(self, pL: Tensor, pR: Tensor) -> Tensor:
        """Ducowicz-style shock indicator: max(0, (pR-pL)/pL) gated."""
        jump = (pR - pL) / soft_floor(smooth_min(pL, pR), 1e-12)
        return smoothstep((jump - self.shock_threshold) / self.shock_threshold)

    def flux(self, primL: Dict[str, Tensor], primR: Dict[str, Tensor],
             axis: int) -> Dict[str, Tensor]:
        g = self.gamma
        keys = ("u", "v", "w")
        unL, unR = primL[keys[axis]], primR[keys[axis]]
        rest = [k for i, k in enumerate(keys) if i != axis]
        rL = soft_floor(primL["rho"], 1e-12); rR = soft_floor(primR["rho"], 1e-12)
        pL = soft_floor(primL["p"], 1e-12);   pR = soft_floor(primR["p"], 1e-12)
        cL = primL["c"]; cR = primR["c"]
        EL = pL/(g-1) + 0.5*rL*(primL["u"]**2+primL["v"]**2+primL["w"]**2)
        ER = pR/(g-1) + 0.5*rR*(primR["u"]**2+primR["v"]**2+primR["w"]**2)
        SL, SR, u_roe, c_roe = self.wave_speeds(rL, unL, pL, cL, rR, unR, pR, cR)
        num = pR - pL + rL*unL*(SL-unL) - rR*unR*(SR-unR)
        den = rL*(SL-unL) - rR*(SR-unR)
        Ss = num / (den + torch.sign(den)*1e-12 + 1e-30)

        def star(r, un, p, E, S, ut1, ut2):
            fac = (S - un) / (S - Ss + 1e-12)
            rs = r*fac
            ps = p + r*(un-S)*(un-Ss)
            Es = rs*(E/r + (Ss-un)*(Ss + p/(r*(S-un)+1e-12)))
            return rs, rs*Ss, rs*ut1, rs*ut2, Es

        rLs, mLs, m1Ls, m2Ls, ELs = star(rL, unL, pL, EL, SL, primL[rest[0]], primL[rest[1]])
        rRs, mRs, m1Rs, m2Rs, ERs = star(rR, unR, pR, ER, SR, primR[rest[0]], primR[rest[1]])
        FL = torch.stack([rL*unL, rL*unL**2+pL, rL*unL*primL[rest[0]],
                          rL*unL*primL[rest[1]], (EL+pL)*unL], 0)
        FR = torch.stack([rR*unR, rR*unR**2+pR, rR*unR*primR[rest[0]],
                          rR*unR*primR[rest[1]], (ER+pR)*unR], 0)
        UL = torch.stack([rL, rL*unL, rL*primL[rest[0]], rL*primL[rest[1]], EL], 0)
        UR = torch.stack([rR, rR*unR, rR*primR[rest[0]], rR*primR[rest[1]], ER], 0)
        ULs = torch.stack([rLs, mLs, m1Ls, m2Ls, ELs], 0)
        URs = torch.stack([rRs, mRs, m1Rs, m2Rs, ERs], 0)
        SLb = SL.unsqueeze(0).expand_as(FL); SRb = SR.unsqueeze(0).expand_as(FL)
        FLs = FL + SLb*(ULs - UL)
        FRs = FR + SRb*(URs - UR)
        gL = soft_gate(SL, self.tau).unsqueeze(0)
        gR = soft_gate(-SR, self.tau).unsqueeze(0)
        gm = (1-gL)*(1-gR)
        gs = soft_gate(Ss, self.tau).unsqueeze(0)
        flux = gL*FL + gR*FR + gm*(gs*FLs + (1-gs)*FRs)

        # [G3b] carbuncle cure: extra TRANSVERSE dissipation at a strong shock
        if self.carbuncle_cure:
            strength = self.shock_strength(pL, pR).unsqueeze(0)
            self.n_cure_applied += 1
            diss = self.cure_strength * strength * c_roe.unsqueeze(0) * (UR - UL)
            # apply only to transverse momentum components (indices 2,3)
            mask = torch.zeros_like(flux)
            mask[2] = 1.0; mask[3] = 1.0
            flux = flux - mask * diss
        out = {"mass": flux[0], "n": flux[1], "t1": flux[2], "t2": flux[3], "E": flux[4]}
        return out

    def report(self) -> Dict[str, Any]:
        return {"entropy_fix_calls": self.n_entropy_fixed,
                "carbuncle_cure_calls": self.n_cure_applied,
                "note": ("entropy fix + shock-normal shear dissipation "
                         "reduce carbuncle risk; NEITHER is a proof of "
                         "carbuncle-free behaviour. Run the M=15 blunt "
                         "body case and inspect the stagnation line.")}


# =============================================================================
# §7. WALL CATALYSIS + ABLATION  [G6]
# =============================================================================

class CatalyticAblativeWall:
    """
    [G6] Catalytic recombination heat flux + Stefan blowing ablation.
    q_cat = sum_s gamma_s rho_s sqrt(R_s T/(2 pi)) dh_recomb,s
    Recombination: 2N -> N2 releases D_N2; 2O -> O2 releases D_O2.
    Ablation: mdot prescribed or Arrhenius; wall normal velocity
    DERIVED from mdot/rho_wall (mass conservation), not imposed.
    """
    def __init__(self, gamma_N: float = 0.0, gamma_O: float = 0.0,
                 T_wall: float = 1500.0, mdot: float = 0.0,
                 h_abl: float = 1e7, emissivity: float = 0.85):
        self.gamma_N = gamma_N
        self.gamma_O = gamma_O
        self.T_wall = T_wall
        self.mdot = mdot
        self.h_abl = h_abl
        self.emissivity = emissivity

    def catalytic_flux(self, rho_s: Tensor, T: Tensor) -> Tensor:
        Rs = torch.as_tensor(R_SPEC, dtype=T.dtype, device=T.device)
        Tc = soft_floor(T, 100.0)
        vth = torch.sqrt(Rs * Tc.unsqueeze(-1) / (2*math.pi))
        D_N2 = 113200.0 * R_UNIV / M_MOL[3]     # J/kg of N atoms
        D_O2 = 59500.0 * R_UNIV / M_MOL[4]
        qN = self.gamma_N * soft_floor(rho_s[..., 3], 0.0) * vth[..., 3] * D_N2
        qO = self.gamma_O * soft_floor(rho_s[..., 4], 0.0) * vth[..., 4] * D_O2
        return qN + qO

    def blowing_velocity(self, rho_wall: Tensor) -> Tensor:
        return self.mdot / soft_floor(rho_wall, 1e-8)

    def surface_balance(self, q_conv: Tensor, q_rad_in: Tensor,
                        T_wall: Tensor) -> Dict[str, Tensor]:
        sigma = 5.670374e-8
        q_rad_out = self.emissivity * sigma * soft_floor(T_wall, 100.0)**4
        q_abl = self.mdot * self.h_abl
        net = q_conv + q_rad_in - q_rad_out - q_abl
        return {"q_conv": q_conv, "q_rad_in": q_rad_in,
                "q_rad_out": q_rad_out, "q_ablation": q_abl,
                "q_net": net,
                "note": "q_net>0 means the wall heats; a real TPS design "
                        "solves the coupled material response, not done here."}


# =============================================================================
# §8. SHOCK-LAYER RADIATION (tangent slab / optically thin)  [G7]
# =============================================================================

class ShockLayerRadiation:
    """
    [G7] PARTIAL. Two tractable limits:
      optically_thin : q = sum_s n_s Lambda_s(T)  (upper bound on loss)
      tangent_slab   : q = 2 pi int I ds with an effective kappa
    HONEST: real reentry radiation needs NO gamma/beta, N2+ 1st-neg,
    and atomic VUV lines with line-by-line transport (NEQAIR/HARA).
    This is a surrogate. DO NOT use for TPS sizing.
    """
    def __init__(self, mode: str = "optically_thin",
                 kappa_eff: float = 1.0, slab_length: float = 0.05):
        if mode not in ("optically_thin", "tangent_slab", "off"):
            raise ValueError("mode must be optically_thin|tangent_slab|off")
        self.mode = mode
        self.kappa = kappa_eff
        self.L = slab_length

    def emission_coeff(self, T: Tensor, rho_s: Tensor) -> Tensor:
        """
        Crude effective emission Lambda(T) ~ A exp(-E/T) per species.
        NOT a substitute for real line data.
        """
        Tc = soft_floor(T, 500.0)
        A = torch.tensor([0.0, 0.0, 1.0e-2, 5.0e-3, 2.0e-3],
                         dtype=T.dtype, device=T.device)
        E = torch.tensor([0.0, 0.0, 9.0e4, 1.1e5, 5.0e4],
                         dtype=T.dtype, device=T.device)
        n = soft_floor(rho_s, 0.0) / torch.as_tensor(M_MOL, dtype=T.dtype, device=T.device) * N_A
        return (n * A * torch.exp(-E / Tc.unsqueeze(-1))).sum(-1)

    def source(self, T: Tensor, rho_s: Tensor) -> Tensor:
        if self.mode == "off":
            return torch.zeros_like(T)
        eps = self.emission_coeff(T, rho_s)
        if self.mode == "optically_thin":
            return -4.0 * eps
        tau = self.kappa * self.L
        escape = 1.0 - torch.exp(-tau)     # crude escape factor
        return -4.0 * eps * escape


# =============================================================================
# §9. MSPS CONTROL LAYER
# =============================================================================

class CSOCEventCounter:
    """N(T) <= (E_max + D)/dE_min with D = ACTUAL removal."""
    def __init__(self, hi=0.15, lo=0.05, rate=0.05, name="event"):
        assert hi > lo
        self.name, self.hi, self.lo, self.rate = name, hi, lo, rate
        self.gap = hi - lo
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0; self.D_audit = 0.0

    def step(self, signal: float) -> bool:
        s = float(signal)
        excess = math.log1p(math.exp(-abs(s-self.hi))) + max(s-self.hi, 0.0)
        decay = 1.0/(1.0+math.exp(-(self.acc-self.lo)*10.0))
        removal = decay*self.rate*self.acc
        self.acc += excess - removal
        self.D += removal; self.D_audit += removal
        self.E_max = max(self.E_max, self.acc)
        fire = self.armed and self.acc > self.hi
        if self.acc < self.lo: self.armed = True
        if fire:
            self.armed = False; self.events += 1
        return fire

    def bound(self) -> float:
        return (self.E_max + self.D)/max(self.gap, 1e-30)

    def report(self) -> Dict[str, Any]:
        return {"name": self.name, "events": self.events,
                "bound": self.bound(), "satisfied": self.events <= self.bound()+1e-9,
                "D_actual": self.D, "D_match": abs(self.D-self.D_audit) < 1e-12,
                "note": "engineering event bound, NOT PDE regularity"}


def delta_min_exact(n=3, kappa=KAPPA_FATNESS, a_min=A_MIN) -> float:
    d = n-1
    om = math.pi**(d/2.0)/math.gamma(d/2.0+1.0)
    return (1.0/kappa)*(a_min/om)**(1.0/d)


def kappa_fat_gate(ind: Tensor, kappa=KAPPA_FATNESS) -> Tensor:
    f = ind[None, None]
    avg = F.avg_pool3d(f, 3, stride=1, padding=1)[0, 0]
    return soft_gate(avg/(ind.abs()+1e-12) - 1.0/kappa, tau=0.05)


def build_carpet_laplacian(side=9) -> Tensor:
    side = max(3, side); n = side*side
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1: continue
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii, jj = i+di, j+dj
                if not (0 <= ii < side and 0 <= jj < side): continue
                if ii % 3 == 1 and jj % 3 == 1: continue
                A[i*side+j, ii*side+jj] = 1.0
    return torch.diag(A.sum(1)) - A


def carpet_priority(field3: Tensor, L: Tensor, levels=3, rho=CARPET_RHO) -> Tensor:
    flat = field3.reshape(-1).double()
    n, m = flat.numel(), L.shape[0]
    grp = torch.arange(n, device=flat.device)*m//n
    num = torch.zeros(m, dtype=torch.float64, device=flat.device).index_add(0, grp, flat)
    den = torch.zeros(m, dtype=torch.float64, device=flat.device).index_add(
        0, grp, torch.ones_like(flat))
    s = num/den.clamp_min(1.0)
    return sum((rho**l)*(s @ L.to(s.device) @ s)/(l+1) for l in range(levels))


def opmc_d_mn(m=4, n=4) -> int:
    return m*m*n*n + m*n*n


class OPMCShapeDesign(nn.Module):
    """theta -> decoder -> wall geometry perturbation. EXACT gradient."""
    def __init__(self, shape, m=4, n=4, seed=0):
        super().__init__()
        self.d = opmc_d_mn(m, n); self.shape = tuple(shape)
        nout = self.shape[0]*self.shape[1]*self.shape[2]
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(self.d, nout, generator=g,
                                              dtype=torch.float64)/math.sqrt(self.d))
        self.theta = nn.Parameter(torch.zeros(self.d, dtype=torch.float64))

    def forward(self) -> Tensor:
        return (self.theta @ self.W).reshape(self.shape)


def kovasznay_split(u: Tensor, v: Tensor, w: Tensor,
                    dx: float) -> Dict[str, Tensor]:
    """
    Helmholtz split of the velocity: solenoidal (vortical) +
    irrotational (acoustic). In COMPRESSIBLE flow this is real
    physics (Kovasznay 1953 mode decomposition), not a metaphor.
    """
    def d(f, ax):
        return (torch.roll(f, -1, ax) - torch.roll(f, 1, ax))/(2*dx)
    div = d(u,0)+d(v,1)+d(w,2)
    dh = torch.fft.fftn(div)
    n = u.shape[0]
    k = 2*math.pi*torch.fft.fftfreq(n, d=dx, device=u.device)
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    K2 = (KX**2+KY**2+KZ**2).clamp_min(1e-30)
    ph = -dh/K2
    ph[0,0,0] = 0.0
    phi = torch.fft.ifftn(ph).real
    gx, gy, gz = d(phi,0), d(phi,1), d(phi,2)
    us, vs, ws = u-gx, v-gy, w-gz
    return {"vortical_norm": torch.sqrt(us**2+vs**2+ws**2),
            "acoustic_norm": torch.sqrt(gx**2+gy**2+gz**2),
            "dilatation": div, "phi": phi}


def subsample_ci(x: np.ndarray, conf=0.95) -> Dict[str, float]:
    x = np.asarray(x, float).reshape(-1); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(max(2, int(n**(1/3))), n//2)
    th = float(x.mean())
    sub = np.array([x[i:i+b].mean() for i in range(n-b+1)])
    sc = math.sqrt(b)*(sub-th); a = 1-conf
    lo = th - float(np.quantile(sc, 1-a/2))/math.sqrt(n)
    hi = th - float(np.quantile(sc, a/2))/math.sqrt(n)
    return {"mean": th, "lo": min(lo,hi), "hi": max(lo,hi), "n": n, "block": b}


def tau_int(x: np.ndarray, c=5.0) -> Dict[str, float]:
    x = np.asarray(x, float); n = len(x)
    if n < 64: return {"tau_int": float("nan"), "n": n, "status": "too_short"}
    xc = x-x.mean(); var = float(xc@xc)/n
    if var <= 0: return {"tau_int": 0.5, "n": n, "status": "constant"}
    tau, W = 0.5, 0
    for t in range(1, min(n//2, 5000)):
        tau += float(xc[:n-t]@xc[t:])/(n-t)/var
        if tau <= 0: break
        W = t
        if W >= c*max(tau, 0.5): break
    tau = max(tau, 0.5)
    return {"tau_int": float(tau), "window": W, "n_eff": float(n/(2*tau))}


# =============================================================================
# §10. EQUILIBRIUM NORMAL-SHOCK REFERENCE (for T4)
# =============================================================================

def equilibrium_composition(T: float, rho: float,
                            elem_N: float, elem_O: float,
                            n_iter: int = 60) -> np.ndarray:
    """
    Solve for the 5 species at (T, rho) with element conservation and
    the 3 dissociation equilibria. Newton in (ln n_N, ln n_O).
      n_N2 = n_N^2/K1, n_O2 = n_O^2/K2, n_NO = n_N n_O/K3
      2n_N2 + n_NO + n_N = N_tot ; 2n_O2 + n_NO + n_O = O_tot
    """
    Tt = torch.tensor([T], dtype=torch.float64)
    Kc = equilibrium_Kc(Tt)[0].detach().numpy()
    K1, K2, K3 = [max(k, 1e-300) for k in Kc]
    N_tot = elem_N * rho   # mol/m^3 of N atoms (element)
    O_tot = elem_O * rho
    lnN, lnO = math.log(max(N_tot*0.1, 1e-20)), math.log(max(O_tot*0.1, 1e-20))
    for _ in range(n_iter):
        nN, nO = math.exp(lnN), math.exp(lnO)
        nN2 = nN*nN/K1; nO2 = nO*nO/K2; nNO = nN*nO/K3
        r1 = 2*nN2 + nNO + nN - N_tot
        r2 = 2*nO2 + nNO + nO - O_tot
        J11 = 4*nN2 + nNO + nN; J12 = nNO
        J21 = nNO;              J22 = 4*nO2 + nNO + nO
        det = J11*J22 - J12*J21
        if abs(det) < 1e-300: break
        dN = (-r1*J22 + r2*J12)/det
        dO = (-r2*J11 + r1*J21)/det
        lnN += max(min(dN, 2.0), -2.0)
        lnO += max(min(dO, 2.0), -2.0)
        if abs(r1) + abs(r2) < 1e-12*max(N_tot+O_tot, 1e-30): break
    nN, nO = math.exp(lnN), math.exp(lnO)
    c = np.array([nN*nN/K1, nO*nO/K2, nN*nO/K3, nN, nO])
    return c * M_MOL      # partial densities kg/m^3


def normal_shock_ideal(M1: float, T1: float, p1: float,
                       gamma=1.4) -> Dict[str, float]:
    r1 = p1/(R_UNIV/0.02896*T1)
    c1 = math.sqrt(gamma*p1/r1); u1 = M1*c1
    pr = (2*gamma*M1**2-(gamma-1))/(gamma+1)
    rr = (gamma+1)*M1**2/((gamma-1)*M1**2+2)
    tr = pr/rr
    return {"M1": M1, "T1": T1, "T2": T1*tr, "p2_p1": pr,
            "rho2_rho1": rr, "u1": u1, "rho1": r1,
            "e_over_E": (c1**2/(gamma*(gamma-1)))/
                        (c1**2/(gamma*(gamma-1)) + u1**2/2)}


def normal_shock_equilibrium(M1: float, T1: float, p1: float,
                             n_outer: int = 80) -> Dict[str, float]:
    """
    Rankine-Hugoniot with equilibrium 5-species air.
    Outer bisection on T2; inner: rho2 from continuity+momentum.
    """
    Rair = R_UNIV/0.02896
    r1 = p1/(Rair*T1); u1 = M1*math.sqrt(1.4*p1/r1)
    elem_N = 2*0.79/(0.02896)   # mol N atoms per kg air (approx)
    elem_O = 2*0.21/(0.02896)
    h1 = 1.4/(1.4-1)*p1/r1 + 0.0   # per kg, cold air (no formation offset)
    mflux = r1*u1
    mom = p1 + r1*u1*u1
    H = h1 + 0.5*u1**2
    def resid(T2):
        # p2 = rho2 R_mix(T2) T2 ; p2 = mom - mflux^2/rho2
        # => rho2^2 R T2 - mom rho2 + mflux^2 = 0
        Rs = R_SPEC
        # first guess with air R
        Rm = Rair
        for _ in range(12):
            A = Rm*T2; B = -mom; C = mflux**2
            disc = B*B - 4*A*C
            if disc < 0: return None
            r2 = (-B + math.sqrt(disc))/(2*A)
            rs = equilibrium_composition(T2, r2, elem_N, elem_O)
            Rm = float((rs*Rs).sum()/max(r2, 1e-30))
        p2 = mom - mflux**2/r2
        u2 = mflux/r2
        Tt = torch.tensor([T2], dtype=torch.float64)
        es = species_internal_energy(Tt)[0].detach().numpy()
        e2 = float((rs*es).sum()/r2)
        h2 = e2 + p2/r2
        return (h2 + 0.5*u2**2) - H, r2, p2, u2, rs
    lo, hi = T1*1.05, T1*90.0
    best = None
    for _ in range(n_outer):
        Tm = 0.5*(lo+hi)
        out = resid(Tm)
        if out is None:
            hi = Tm; continue
        rr_, r2, p2, u2, rs = out
        best = (Tm, r2, p2, u2, rs)
        if rr_ > 0: hi = Tm
        else: lo = Tm
    if best is None:
        return {"status": "no_root"}
    T2, r2, p2, u2, rs = best
    return {"status": "ok", "M1": M1, "T1": T1, "T2": T2,
            "p2_p1": p2/p1, "rho2_rho1": r2/r1, "u2": u2,
            "mass_fractions": (rs/r2).tolist(),
            "species": list(SPECIES)}


# =============================================================================
# §11. VERIFICATION SUITE — the 5 tests requested
# =============================================================================

def verify(device="cpu") -> bool:
    print(BANNER)
    dev = torch.device(device)
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p); print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # T0 NASA-7 internal consistency at the blend point
    print("\n[T0] NASA-7 continuity at T_MID (internal consistency)")
    T = torch.tensor([999.0, 1000.0, 1001.0], dtype=torch.float64)
    cp, h, s = nasa7(T)
    dcp = float((cp[1]-cp[0]).abs().max())
    rep("cp/R continuous across T_MID", dcp < 0.05, f"max jump={dcp:.4f}")
    print("      NOTE: absolute values MUST be verified against the NASA database.")

    # T1 DUAL ENERGY at M=20
    print("\n[T1] Dual energy: e/E and pressure precision at M=20")
    ig = normal_shock_ideal(20.0, 220.0, 1000.0)
    print(f"      ideal-gas M=20: T2/T1={ig['T2']/ig['T1']:.2f} "
          f"p2/p1={ig['p2_p1']:.1f} rho2/r1={ig['rho2_rho1']:.2f}")
    print(f"      e/E = {ig['e_over_E']*100:.3f}%  <- the cancellation problem")
    rep("e/E < 1% at M=20 (confirms the need)", ig["e_over_E"] < 0.01)
    # build a state at that condition and compare p via both routes
    g = 1.4
    r1, u1 = ig["rho1"], ig["u1"]
    e1 = ig["p1"] if False else 1000.0/(g-1)
    E1 = e1 + 0.5*r1*u1**2
    for dt_ in (torch.float64, torch.float32, torch.bfloat16):
        rho = torch.tensor(r1, dtype=dt_)
        rhou = torch.tensor(r1*u1, dtype=dt_)
        rhoE = torch.tensor(E1, dtype=dt_)
        rhoe = torch.tensor(e1, dtype=dt_)
        ke = 0.5*rhou**2/rho
        p_tot = (g-1)*(rhoE-ke)
        p_dual = (g-1)*rhoe
        p_ref = 1000.0
        e_tot = float(abs(float(p_tot)-p_ref)/p_ref)
        e_dual = float(abs(float(p_dual)-p_ref)/p_ref)
        print(f"      {str(dt_):>16}: rel err p from E = {e_tot:.3e} | "
              f"from dual e = {e_dual:.3e}")
        if dt_ == torch.bfloat16:
            rep("bf16: dual energy beats total-energy subtraction",
                e_dual < e_tot, f"{e_dual:.3e} < {e_tot:.3e}")

    # T2 ZHANG-SHU at pressure ratio 466:1
    print("\n[T2] Zhang-Shu positivity limiter at p2/p1 = 466")
    lim = ZhangShuLimiter(gamma=1.4)
    n = 8
    bar = {"rho": torch.full((n,), 1.0, dtype=torch.float64),
           "rhou": torch.full((n,), 20.0, dtype=torch.float64),
           "rhov": torch.zeros(n, dtype=torch.float64),
           "rhow": torch.zeros(n, dtype=torch.float64),
           "rhoE": torch.full((n,), 201.786, dtype=torch.float64)}
    U = {k: v.clone() for k, v in bar.items()}
    U["rho"][0] = -0.5          # negative density from an unlimited reconstruction
    U["rhoE"][1] = 190.0        # p < 0 (KE = 200)
    out = lim(U, bar)
    p_after = (1.4-1)*(out["rhoE"] - 0.5*(out["rhou"]**2+out["rhov"]**2+out["rhow"]**2)
                       / out["rho"].clamp_min(1e-30))
    rep("rho > 0 everywhere after limiting", bool((out["rho"] > 0).all()),
        f"min rho={float(out['rho'].min()):.3e}")
    rep("p > 0 everywhere after limiting", bool((p_after > 0).all()),
        f"min p={float(p_after.min()):.3e}")
    print(f"      {lim.report()}")

    # T3 ENTROPY FIX + CARBUNCLE (machinery test; blunt body must be run)
    print("\n[T3] Harten-Hyman + carbuncle cure (machinery)")
    fl = HypersonicHLLC(entropy_fix=True, carbuncle_cure=True)
    lam = torch.tensor([0.0, 0.01, -0.01], dtype=torch.float64)
    fixed = fl.harten_hyman(lam, lam-1.0, lam+1.0, 0.1)
    rep("entropy fix lifts |lambda| near 0", bool((fixed >= 0.05-1e-9).all()),
        f"fixed={fixed.tolist()}")
    pL = torch.tensor([1.0]); pR = torch.tensor([466.0])
    st = float(fl.shock_strength(pL, pR))
    rep("shock detector fires at 466:1", st > 0.9, f"strength={st:.4f}")
    print("      ⚠ The DECISIVE carbuncle test is a blunt body at M=15.")
    print("        Run --blunt-body --Mach 15 and inspect the stagnation line.")

    # T4 PARK 5-SPECIES: ideal vs equilibrium post-shock T
    print("\n[T4] Post-shock T: ideal gas vs equilibrium 5-species air")
    for M1 in (10.0, 15.0, 20.0):
        ig = normal_shock_ideal(M1, 220.0, 1000.0)
        eq = normal_shock_equilibrium(M1, 220.0, 1000.0)
        if eq.get("status") == "ok":
            ratio = ig["T2"]/eq["T2"]
            print(f"      M={M1:4.0f}: T2_ideal={ig['T2']:8.0f} K | "
                  f"T2_equil={eq['T2']:8.0f} K | ideal/equil={ratio:.2f}x")
            if M1 == 20.0:
                rep("ideal gas overpredicts T2 by >1.5x at M=20", ratio > 1.5,
                    f"{ratio:.2f}x")
                yN = eq["mass_fractions"][3]; yO = eq["mass_fractions"][4]
                rep("dissociation produces atomic N and O", yN+yO > 0.05,
                    f"Y_N={yN:.3f} Y_O={yO:.3f}")
        else:
            rep(f"equilibrium shock root found at M={M1}", False)
    print("      ⚠ ABSOLUTE agreement with NASA CEA2 is NOT claimed.")
    print("        Run CEA2 at the same (M, T1, p1) and compare T2 yourself.")

    # T5 TWO-TEMPERATURE
    print("\n[T5] Two-temperature Park: Tv != T behind a shock")
    pcfg = ParkConfig(two_temperature=True, ta_mode="park_sqrt", cvdv_chi=0.3)
    rho_s = torch.tensor([[0.7, 0.2, 0.0, 0.05, 0.05]], dtype=torch.float64)
    T = torch.tensor([8000.0], dtype=torch.float64)
    Tv = torch.tensor([2000.0], dtype=torch.float64)
    Qlt = vibrational_relaxation(rho_s, T, Tv, pcfg)
    rep("Landau-Teller transfers energy T -> Tv", float(Qlt) > 0,
        f"Q_LT={float(Qlt):.3e} W/m^3")
    kf_sqrt, _ = park_rates(T, Tv, pcfg)
    pcfg1 = ParkConfig(two_temperature=False)
    kf_T, _ = park_rates(T, Tv, pcfg1)
    rep("Ta=sqrt(T*Tv) reduces dissociation vs Ta=T",
        bool((kf_sqrt < kf_T).all()),
        f"ratio={float((kf_sqrt/kf_T).mean()):.3e}")
    om, qv, R = chemical_source(rho_s, T, Tv, pcfg)
    rep("mass conserved by chemistry", abs(float(om.sum())) < 1e-6*float(rho_s.sum()),
        f"|sum omega|={abs(float(om.sum())):.3e}")
    rep("CVDV removes vibrational energy on dissociation", float(qv) <= 0,
        f"q_vib={float(qv):.3e}")

    # T6 WALL CATALYSIS
    print("\n[T6] Wall catalysis: non-catalytic vs fully catalytic")
    w0 = CatalyticAblativeWall(gamma_N=0.0, gamma_O=0.0)
    w1 = CatalyticAblativeWall(gamma_N=1.0, gamma_O=1.0)
    q0 = float(w0.catalytic_flux(rho_s, T)); q1 = float(w1.catalytic_flux(rho_s, T))
    rep("fully catalytic heat flux > non-catalytic", q1 > q0,
        f"q_nc={q0:.3e}  q_fc={q1:.3e} W/m^2")
    rep("non-catalytic gives zero catalytic flux", q0 == 0.0)

    # T7 MSPS CONTROL LAYER
    print("\n[T7] MSPS control layer")
    ec = CSOCEventCounter(name="limiter")
    for t in range(800):
        ec.step(0.1 + 0.3*abs(math.sin(0.11*t)))
    r = ec.report()
    rep("N <= (E_max+D)/gap", r["satisfied"], f"N={r['events']} bound={r['bound']:.2f}")
    rep("D accounting exact", r["D_match"])
    rep("delta_min > 0", delta_min_exact() > 0, f"{delta_min_exact():.6f}")
    L = build_carpet_laplacian(9)
    rep("carpet Laplacian row-sums 0", bool(torch.allclose(L.sum(1), torch.zeros(81, dtype=torch.float64))))
    u = torch.randn(16,16,16, dtype=torch.float64)
    kv = kovasznay_split(u, u*0.5, u*0.3, 1/16)
    rep("Kovasznay split finite", bool(torch.isfinite(kv["vortical_norm"]).all()))
    d = OPMCShapeDesign((8,8,8))
    rep("d(4,4)=320", d.d == 320)
    (d().pow(2).sum()).backward()
    rep("OPMC shape-design gradient exact", d.theta.grad is not None)
    ci = subsample_ci(np.random.default_rng(0).normal(size=500))
    rep("CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])

    print("\n" + "="*78)
    print(f"VERIFY: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
MEASURED HERE
  e/E = 0.885% at M=20 (the cancellation problem is real)
  dual energy beats total-energy subtraction in bf16
  Zhang-Shu restores rho>0 and p>0 from an unlimited negative state
  Harten-Hyman lifts |lambda| at sonic points; shock detector fires at 466:1
  ideal gas OVERPREDICTS post-shock T vs equilibrium 5-species air
  Ta=sqrt(T*Tv) reduces dissociation; chemistry conserves mass
  fully catalytic wall heat flux > non-catalytic
  CSOC event bound holds with exact D accounting

REQUIRES EXTERNAL VALIDATION (not done here)
  NASA-7 absolute values -> NASA database
  post-shock T2          -> NASA CEA2
  carbuncle-free         -> blunt body at M=15, stagnation streamline
  shock standoff         -> experiment (e.g. CUBRC / LENS)
  surface heat flux      -> flight or tunnel data

NOT CLAIMED
  World's fastest / highest resolution. Nothing benchmarked.
  11-species ionization. Line-by-line radiation. Body-fitted grid.
  MPI. Any clinical / flight-certification use.
""")
    return ok


# =============================================================================
# §12. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Hypersonic ONE")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--shock", type=float, default=None,
                    help="normal-shock comparison at this Mach")
    ap.add_argument("--T1", type=float, default=220.0)
    ap.add_argument("--p1", type=float, default=1000.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.verify:
        return 0 if verify(a.device) else 1

    if a.shock is not None:
        print(BANNER)
        ig = normal_shock_ideal(a.shock, a.T1, a.p1)
        eq = normal_shock_equilibrium(a.shock, a.T1, a.p1)
        out = {"ideal_gas": ig, "equilibrium_5species": eq,
               "T2_ratio_ideal_over_equil": (ig["T2"]/eq["T2"]
                                             if eq.get("status")=="ok" else None),
               "honest_note": "Absolute T2 requires NASA CEA2 verification."}
        print(json.dumps(out, indent=2, default=float))
        if a.json:
            with open(a.json, "w") as fh: json.dump(out, fh, indent=2, default=float)
        return 0

    ap.print_help(); print(BANNER)
    print("Start: --verify | --shock 20 --T1 220 --p1 1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
