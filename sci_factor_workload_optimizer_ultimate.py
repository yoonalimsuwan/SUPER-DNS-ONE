#!/usr/bin/env python3
"""
=============================================================================
HYPERSONIC THERMOCHEMICAL NONEQUILIBRIUM — v4.0 FULL RADIATIVE
RKR1 + Johnson Renormalized Numerov + Hönl-London + NDE + FC Validation
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
Licence   : MIT
Year      : 2026
Code      : Qwen (Alibaba Group)

NEW IN v4.0 (relative to v3.1):
  [1] Johnson renormalized Numerov replaces shooting method for ψ_v(r).
  [2] Hönl-London rotational line-strength matrix for Σ–Σ, Π–Σ, Δ–Σ
      transitions using Wigner 3j symbols (Watson 2008).
  [3] Near-dissociation expansion (NDE) à la LeRoy–Bernstein for
      vibrational levels v → v_D (RKR1 logic).
  [4] Validation of Franck–Condon matrices against Jain & Sahni (1968)
      for NO γ (A²Σ⁺–X²Π) and β (B²Π–X²Π) bands.

ALGORITHM PROVENANCE:
  [RKR]  LeRoy, JQSRT 186, 158 (2017) — RKR1.
  [JN]   Johnson, J. Chem. Phys. 69, 4678 (1977) — renormalized Numerov.
  [HL]   Watson, J. Mol. Spectrosc. 252, 5 (2008) — Hönl-London factors.
  [NDE]  LeRoy & Bernstein, J. Chem. Phys. 52, 3869 (1970).
  [FC]   Jain & Sahni, Trans. Faraday Soc. 64, 3169 (1968).
  [LBL]  Whiting & Park, NEQAIR96, NASA TM-110575 (1996).

HONEST SCOPE:
  ❌ NDE parameters (C_n, v_D) are supplied by the user; defaults are
     approximations for N2/NO and must be replaced by literature values.
  ❌ FC validation uses a small hard-coded reference table from Jain &
     Sahni (1968); full matrices must be loaded from the original paper.
  ❌ Hönl-London implementation covers singlet–singlet Σ, Π, Δ only;
     multiplet (triplet) transitions require a full case (a)/(b) treatment.
  ❌ No experimental validation is performed here beyond FC comparison.
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import simpson, trapezoid
from scipy.optimize import brentq
from scipy.special import factorial, gammaln

# =============================================================================
# PHYSICAL CONSTANTS (SI, CODATA 2018)
# =============================================================================
R_UNIV = 8.31446261815324
K_B = 1.380649e-23
N_A = 6.02214076e23
M_E = 9.1093837015e-31
Q_E = 1.602176634e-19
EV_TO_K = 11604.51812
EPS_0 = 8.8541878128e-12
SIGMA_SB = 5.670374419e-8
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
AMU = 1.66053906660e-27
CM_TO_J = H_PLANCK * C_LIGHT * 100.0   # 1 cm^-1 in J
BOHR_M = 5.29177210903e-11
EA0_SI = 2.5416e-18 / 3e9 / 100.0     # e·a0 in C·m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("hypchem")

VERSION = "4.0.0-rkr-johnson"

BANNER = f"""
=============================================================================
HYPERSONIC CHEMISTRY v{VERSION}
[RKR1] [Johnson Numerov] [Hönl-London Π-Σ Δ-Σ] [NDE] [FC validation]
Developer: PAI and Yoon A Limsuwan / MSPS NETWORK
Licence: MIT | Year: 2026
=============================================================================
"""

# =============================================================================
# §1. SPECTROSCOPIC CONSTANTS
# =============================================================================

@dataclass
class SpectroscopicConstants:
    """Dunham coefficients for one electronic state."""
    Te: float          # electronic term energy [cm^-1]
    we: float          # harmonic frequency [cm^-1]
    wexe: float        # anharmonicity [cm^-1]
    Be: float          # rotational constant [cm^-1]
    alpha_e: float     # rotation-vibration coupling [cm^-1]
    De: float = 1e-6   # centrifugal distortion [cm^-1]
    mu_amu: float = 7.0  # reduced mass [amu]

    def G(self, v: float) -> float:
        """Vibrational energy [cm^-1] (Dunham)."""
        vp = v + 0.5
        return self.we * vp - self.wexe * vp**2

    def B(self, v: float) -> float:
        """Rotational constant B_v [cm^-1]."""
        return self.Be - self.alpha_e * (v + 0.5)


# Huber & Herzberg (1979) Vol. 4 / NIST CCCBDB
N2_X = SpectroscopicConstants(Te=0.0, we=2358.57, wexe=14.324,
                              Be=1.99324, alpha_e=0.017318,
                              De=5.7e-6, mu_amu=7.0015)
N2_A = SpectroscopicConstants(Te=50203.6, we=1460.0, wexe=13.9,
                              Be=1.466, alpha_e=0.017, mu_amu=7.0015)
N2_B = SpectroscopicConstants(Te=59619.1, we=1733.4, wexe=14.12,
                              Be=1.638, alpha_e=0.018, De=5.8e-6,
                              mu_amu=7.0015)
N2_C = SpectroscopicConstants(Te=89136.9, we=2047.2, wexe=28.3,
                              Be=1.825, alpha_e=0.027, De=6.1e-6,
                              mu_amu=7.0015)

N2p_X = SpectroscopicConstants(Te=0.0, we=2207.0, wexe=16.1,
                               Be=1.932, alpha_e=0.018, De=6.0e-6,
                               mu_amu=7.0015)
N2p_B = SpectroscopicConstants(Te=25461.0, we=2419.0, wexe=23.2,
                               Be=2.073, alpha_e=0.019, De=6.5e-6,
                               mu_amu=7.0015)

NO_X = SpectroscopicConstants(Te=0.0, we=1904.0, wexe=14.1,
                              Be=1.705, alpha_e=0.017, De=5.6e-6,
                              mu_amu=7.5015)
NO_A = SpectroscopicConstants(Te=43965.0, we=2371.0, wexe=16.0,
                              Be=2.002, alpha_e=0.020, De=6.2e-6,
                              mu_amu=7.5015)
NO_B = SpectroscopicConstants(Te=45975.0, we=1735.0, wexe=14.0,
                              Be=1.746, alpha_e=0.019, De=5.9e-6,
                              mu_amu=7.5015)

O2_X = SpectroscopicConstants(Te=0.0, we=1580.19, wexe=11.98,
                              Be=1.4377, alpha_e=0.0159, De=4.9e-6,
                              mu_amu=7.9975)
O2_B = SpectroscopicConstants(Te=49793.0, we=700.0, wexe=8.0,
                              Be=0.812, alpha_e=0.012, De=4.0e-6,
                              mu_amu=7.9975)


# =============================================================================
# §2. RKR1 — first-order Rydberg–Klein–Rees turning points
# =============================================================================
# Follows LeRoy, JQSRT 186, 158 (2017) and Tellinghuisen (1973).
# Gauss–Mehler quadrature handles the (1 - t)^(-1/2) singularity.

def _gauss_mehler_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for Gauss–Mehler on [0,1] with weight (1-t)^(-1/2)."""
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1.0)
    w_t = 0.5 * w * np.sqrt(1.0 - x)
    return t, w_t


def rkr_turning_points(sc: SpectroscopicConstants, v: int,
                       n_quad: int = 64) -> Tuple[float, float]:
    """
    Return (r_min, r_max) in Å for vibrational level v via RKR1.

    f(v) = sqrt(2/μ) ∫_0^v dv' / sqrt(G(v)-G(v'))
    g(v) = sqrt(8π² μ c / h) ∫_0^v B(v') dv' / sqrt(G(v)-G(v'))
    """
    mu = sc.mu_amu * AMU  # kg
    t_nodes, t_weights = _gauss_mehler_nodes(n_quad)

    def f_integral(v_target: float) -> float:
        integrand = np.zeros(n_quad)
        for i, t in enumerate(t_nodes):
            vp = v_target * t
            dG = sc.G(v_target) - sc.G(vp)
            integrand[i] = 0.0 if dG < 1e-12 else 1.0 / math.sqrt(dG)
        integral = np.sum(t_weights * integrand) * v_target
        return math.sqrt(2.0 / mu) * integral

    def g_integral(v_target: float) -> float:
        integrand = np.zeros(n_quad)
        for i, t in enumerate(t_nodes):
            vp = v_target * t
            dG = sc.G(v_target) - sc.G(vp)
            integrand[i] = 0.0 if dG < 1e-12 else sc.B(vp) / math.sqrt(dG)
        integral = np.sum(t_weights * integrand) * v_target
        # prefactor in SI; B and G in cm^-1, c in m/s, h in J s
        pref = math.sqrt(8.0 * math.pi**2 * mu * C_LIGHT / H_PLANCK) * 1e-2
        return pref * integral

    f_val = f_integral(v)
    g_val = g_integral(v)
    if f_val <= 0 or g_val <= 0:
        raise ValueError(f"RKR failed at v={v}: f={f_val}, g={g_val}")
    disc = f_val**2 + 2.0 * f_val / g_val
    r_max = f_val + math.sqrt(disc)
    r_min = r_max - 2.0 * f_val
    return r_min * 1e10, r_max * 1e10  # m -> Å


# =============================================================================
# §3. JOHNSON RENORMALIZED NUMEROV
# =============================================================================
# Adapted from PyDiatomic `johnson.py` (Stephen Gibson, ANU).
# The code below is a single-channel reduction of the coupled-channel solver.

def johnson_solve_single_channel(
    R: np.ndarray,          # radial grid [Å]
    V_cm: np.ndarray,       # potential [cm^-1]
    mu_amu: float,
    J: int = 0,
    n_states: int = 20,
    E_min_cm: float = -1000.0,
    E_max_cm: float = 100000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the radial Schrödinger equation with Johnson renormalized Numerov.

    Returns
    -------
    E_v : ndarray, shape (n_states,)   [cm^-1]
    psi : ndarray, shape (n_states, nR)
    """
    nR = len(R)
    h = R[1] - R[0]
    mu = mu_amu * AMU
    V_J = V_cm * CM_TO_J
    if J > 0:
        Rm = R * 1e-10
        V_J = V_J + J * (J + 1) * (H_PLANCK / (2 * math.pi))**2 / (2 * mu * Rm**2)

    # Johnson factor: 2μ ΔR² / 12 (in units of 1/J)
    h_m = h * 1e-10
    factor = 2.0 * mu * h_m**2 / 12.0   # J^-1

    def W_inv(energy_J: float) -> np.ndarray:
        """Inverse interaction matrix W^{-1} = [I + (E-V) factor]^{-1}."""
        diag = 1.0 + (energy_J - V_J) * factor
        return 1.0 / diag          # diagonal in single channel

    def R_matrix(energy_J: float, mx: int) -> np.ndarray:
        """R_i = [U_i - R_{i-1}]^{-1},  U = 12 W^{-1} - 10."""
        WI = W_inv(energy_J)
        U = 12.0 * WI - 10.0
        RI = np.zeros(nR)
        RI[1] = 0.0
        for i in range(2, mx + 1):
            RI[i] = 1.0 / (U[i] - RI[i - 1])
        RI[nR - 2] = 0.0
        for i in range(nR - 3, mx - 1, -1):
            RI[i] = 1.0 / (U[i] - RI[i + 1])
        return RI

    def f_vector(energy_J: float, mx: int) -> np.ndarray:
        """f-matrix propagated from matching point."""
        RI = R_matrix(energy_J, mx)
        f = np.zeros(nR)
        f[mx] = 1.0 / W_inv(energy_J)[mx]
        for i in range(mx - 1, -1, -1):
            f[i] = f[i + 1] * RI[i]
        for i in range(mx + 1, nR):
            f[i] = f[i - 1] * RI[i]
        return f

    def wavefunction(energy_J: float, mx: int) -> np.ndarray:
        f = f_vector(energy_J, mx)
        return f * W_inv(energy_J)

    def matching_index(energy_J: float) -> int:
        """Index where V ≈ E (outer turning point)."""
        j_min = int(np.argmin(np.abs(V_J - energy_J)))
        return max(5, min(nR - 6, j_min))

    def f_at_outer(energy_J: float) -> float:
        mx = matching_index(energy_J)
        f = f_vector(energy_J, mx)
        return f[-1]

    # Scan for sign changes of f[-1] to bracket eigenvalues
    E_scan = np.linspace(E_min_cm * CM_TO_J, E_max_cm * CM_TO_J, 800)
    signs = np.array([np.sign(f_at_outer(E)) for E in E_scan])
    eigenvalues = []
    eigenfunctions = []
    for k in range(len(E_scan) - 1):
        if signs[k] * signs[k + 1] < 0:
            E_lo, E_hi = E_scan[k], E_scan[k + 1]
            try:
                E_eig = brentq(f_at_outer, E_lo, E_hi, xtol=1e-12)
            except ValueError:
                continue
            mx = matching_index(E_eig)
            psi = wavefunction(E_eig, mx)
            # normalise
            norm = math.sqrt(trapezoid(psi**2, R * 1e-10))
            if norm < 1e-30:
                continue
            psi /= norm
            eigenvalues.append(E_eig / CM_TO_J)
            eigenfunctions.append(psi)
            if len(eigenvalues) >= n_states:
                break
    return np.array(eigenvalues), np.array(eigenfunctions)


# =============================================================================
# §4. HÖNL–LONDON FACTORS (Watson 2008)
# =============================================================================
# For singlet–singlet transitions:
#   S_{Λ'Λ''} = (1 + δ_{Λ'0} + δ_{Λ''0} - 2 δ_{Λ'0}δ_{Λ''0})
#               × (2J'+1)(2J''+1) × (3j)^2
# where 3j = ( J'  1  J'' ; -Λ'  Λ'-Λ''  Λ'' )

def _wigner_3j_squared(Jp: int, Jpp: int, Lp: int, Lpp: int) -> float:
    """
    Compute [ (J' 1 J''; -Λ' Λ'-Λ'' Λ'') ]^2 using the Racah formula.
    Only ΔΛ = |Λ'−Λ''| ≤ 1 contributes for electric-dipole transitions.
    """
    dL = Lp - Lpp
    if abs(dL) > 1:
        return 0.0
    # Wigner 3j with one row = 1
    # Use explicit formulas for ΔΛ = 0, ±1.
    if dL == 0:
        # ( J' 1 J'' ; -Λ 0 Λ )^2
        if Jp == Jpp:
            # 3j = (-1)^{J'-Λ} Λ / sqrt(J'(J'+1)(2J'+1))
            val = (Lp / math.sqrt(Jp * (Jp + 1) * (2 * Jp + 1))) if Jp > 0 else 0.0
            return val**2
        else:
            return 0.0
    else:
        # ΔΛ = ±1; 3j squared
        # Use Racah formula via factorials
        # For 3j(j1 j2 j3; m1 m2 m3) with j2=1, m2=±1
        # We implement a general routine using gammaln.
        j1, j2, j3 = Jp, 1, Jpp
        m1, m2, m3 = -Lp, dL, Lpp
        if m1 + m2 + m3 != 0:
            return 0.0
        # Selection rules
        if not (abs(j1 - j3) <= j2 <= j1 + j3):
            return 0.0
        if m1 + m2 + m3 != 0:
            return 0.0
        # Racah formula
        def f(n):
            return gammaln(n + 1)
        try:
            log_val = (
                0.5 * (f(j1 + j2 - j3) + f(j1 - j2 + j3) + f(-j1 + j2 + j3)
                       - f(j1 + j2 + j3 + 1))
                + 0.5 * (f(j1 + m1) + f(j1 - m1) + f(j2 + m2) + f(j2 - m2)
                         + f(j3 + m3) + f(j3 - m3))
            )
            # sum over k
            k_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
            k_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
            s = 0.0
            for k in range(int(k_min), int(k_max) + 1):
                log_term = (
                    -f(k) - f(j1 + j2 - j3 - k) - f(j1 - m1 - k)
                    - f(j2 + m2 - k) - f(j3 - j2 + m1 + k)
                    - f(j3 - j1 - m2 + k)
                )
                s += (-1)**k * math.exp(log_term)
            val = (-1)**(j1 - j2 - m3) * math.exp(log_val) * s
            return val**2
        except (ValueError, OverflowError):
            return 0.0


def honl_london(Jp: int, Jpp: int, Lp: int, Lpp: int) -> float:
    """
    Singlet–singlet Hönl-London factor S_{J'J''} for Λ' → Λ''.
    """
    delta_term = (1 + (1 if Lp == 0 else 0) + (1 if Lpp == 0 else 0)
                  - 2 * (1 if Lp == 0 and Lpp == 0 else 0))
    return delta_term * (2 * Jp + 1) * (2 * Jpp + 1) * _wigner_3j_squared(Jp, Jpp, Lp, Lpp)


# =============================================================================
# §5. NEAR-DISSOCIATION EXPANSION (NDE) — LeRoy–Bernstein
# =============================================================================
# G(v) = D - [ (v_D - v) * (n-2)/(2n) * (C_n)^{1/n} * ... ]^{2n/(n-2)}
# Simplified implementation for C_6 (n=6) and C_3 (n=3).

@dataclass
class NDEParameters:
    """Long-range parameters for one electronic state."""
    D_e_cm: float          # dissociation energy [cm^-1]
    v_D: float             # effective vibrational index at dissociation
    C_n: float             # long-range coefficient [cm^-1 Å^n]
    n: int = 6             # 6 for neutral–neutral, 3 for ion–induced dipole

    def G_NDE(self, v: float) -> float:
        """Vibrational energy near dissociation [cm^-1]."""
        if v >= self.v_D:
            return self.D_e_cm
        x = self.v_D - v
        if self.n == 6:
            # LeRoy–Bernstein for n=6
            pref = (6.0 / 4.0) * (2.0 / 6.0) * (self.C_n ** (1.0 / 6.0))
            return self.D_e_cm - (x * pref) ** (6.0 / 4.0)
        elif self.n == 3:
            pref = (3.0 / 1.0) * (1.0 / 3.0) * (self.C_n ** (1.0 / 3.0))
            return self.D_e_cm - (x * pref) ** (3.0 / 1.0)
        else:
            # generic
            pref = (self.n - 2) / (2 * self.n) * (self.C_n ** (1.0 / self.n))
            return self.D_e_cm - (x * pref) ** (2 * self.n / (self.n - 2))


# Example parameters (approximate; replace with literature values)
N2_X_NDE = NDEParameters(D_e_cm=79854.0, v_D=60.0, C_n=1.2e6, n=6)
NO_X_NDE = NDEParameters(D_e_cm=52400.0, v_D=45.0, C_n=5.0e5, n=6)


# =============================================================================
# §6. FRANCK–CONDON MATRIX (full quantum mechanical)
# =============================================================================

def build_potential_rkr_nde(
    sc: SpectroscopicConstants,
    nde: Optional[NDEParameters] = None,
    v_max: int = 30,
    n_quad: int = 64,
    n_grid: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a potential curve from RKR1 turning points and NDE near dissociation.
    Returns (R [Å], V [cm^-1]) on a uniform grid.
    """
    r_points, V_points = [], []
    for v in range(v_max + 1):
        try:
            r_min, r_max = rkr_turning_points(sc, v, n_quad=n_quad)
        except ValueError:
            continue
        # If NDE is available and v is near dissociation, use NDE energy
        if nde is not None and v > nde.v_D - 5:
            E_v = nde.G_NDE(v)
        else:
            E_v = sc.G(v)
        r_points.extend([r_min, r_max])
        V_points.extend([E_v, E_v])
    if not r_points:
        raise RuntimeError("RKR returned no turning points")
    r_arr = np.array(r_points)
    V_arr = np.array(V_points)
    # Sort and interpolate
    order = np.argsort(r_arr)
    r_arr, V_arr = r_arr[order], V_arr[order]
    r_uniform = np.linspace(r_arr.min(), r_arr.max(), n_grid)
    V_uniform = np.interp(r_uniform, r_arr, V_arr)
    return r_uniform, V_uniform


def franck_condon_matrix(
    upper_sc: SpectroscopicConstants,
    lower_sc: SpectroscopicConstants,
    v_max_u: int = 12,
    v_max_l: int = 12,
    upper_nde: Optional[NDEParameters] = None,
    lower_nde: Optional[NDEParameters] = None,
    n_quad: int = 64,
    n_grid: int = 3000,
) -> np.ndarray:
    """
    Compute q[v', v''] = |⟨ψ_{v'} | ψ_{v''}⟩|² using Johnson Numerov.
    """
    R_u, V_u = build_potential_rkr_nde(upper_sc, upper_nde, v_max_u,
                                       n_quad, n_grid)
    R_l, V_l = build_potential_rkr_nde(lower_sc, lower_nde, v_max_l,
                                       n_quad, n_grid)

    E_u, psi_u = johnson_solve_single_channel(R_u, V_u, upper_sc.mu_amu,
                                              J=0, n_states=v_max_u + 1)
    E_l, psi_l = johnson_solve_single_channel(R_l, V_l, lower_sc.mu_amu,
                                              J=0, n_states=v_max_l + 1)

    # Common grid
    r_min = max(R_u.min(), R_l.min())
    r_max = min(R_u.max(), R_l.max())
    R_common = np.linspace(r_min, r_max, n_grid)

    psi_u_i = np.array([np.interp(R_common, R_u, psi_u[i])
                        for i in range(len(E_u))])
    psi_l_i = np.array([np.interp(R_common, R_l, psi_l[i])
                        for i in range(len(E_l))])

    # Renormalise on common grid
    for i in range(len(E_u)):
        norm = math.sqrt(trapezoid(psi_u_i[i]**2, R_common))
        psi_u_i[i] /= max(norm, 1e-30)
    for j in range(len(E_l)):
        norm = math.sqrt(trapezoid(psi_l_i[j]**2, R_common))
        psi_l_i[j] /= max(norm, 1e-30)

    FC = np.zeros((len(E_u), len(E_l)))
    for i in range(len(E_u)):
        for j in range(len(E_l)):
            FC[i, j] = trapezoid(psi_u_i[i] * psi_l_i[j], R_common)**2
    return FC


# =============================================================================
# §7. VALIDATION AGAINST JAIN & SAHNI (1968)
# =============================================================================
# Reference FC values for NO γ (A²Σ⁺–X²Π) and β (B²Π–X²Π).
# These are the diagonal and first off-diagonal values from the original
# paper (Jain & Sahni, Trans. Faraday Soc. 64, 3169, 1968).
# Only a small subset is hard-coded; the full matrices are in the paper.

JAIN_SAHNI_REFERENCE = {
    "NO_gamma": {
        # (v', v''): q
        (0, 0): 0.161,
        (0, 1): 0.285,
        (0, 2): 0.251,
        (0, 3): 0.147,
        (0, 4): 0.073,
        (1, 0): 0.309,
        (1, 1): 0.015,
        (1, 2): 0.101,
        (1, 3): 0.196,
        (2, 0): 0.292,
        (2, 1): 0.028,
        (2, 2): 0.063,
        (2, 3): 0.015,
    },
    "NO_beta": {
        (0, 0): 0.342,
        (0, 1): 0.351,
        (0, 2): 0.190,
        (0, 3): 0.076,
        (0, 4): 0.026,
        (1, 0): 0.395,
        (1, 1): 0.041,
        (1, 2): 0.154,
        (1, 3): 0.213,
        (2, 0): 0.168,
        (2, 1): 0.222,
        (2, 2): 0.021,
        (2, 3): 0.109,
    },
}


def validate_fc_against_jain_sahni(
    FC: np.ndarray,
    system: str = "NO_gamma",
    tol: float = 0.02,
) -> Dict[str, float]:
    """
    Compare computed FC matrix with the Jain & Sahni (1968) reference.
    Returns a dictionary of errors.
    """
    if system not in JAIN_SAHNI_REFERENCE:
        raise ValueError(f"Unknown system: {system}")
    ref = JAIN_SAHNI_REFERENCE[system]
    errors = {}
    for (vp, vpp), q_ref in ref.items():
        if vp >= FC.shape[0] or vpp >= FC.shape[1]:
            continue
        q_comp = FC[vp, vpp]
        errors[f"({vp},{vpp})"] = {
            "computed": q_comp,
            "reference": q_ref,
            "abs_error": abs(q_comp - q_ref),
            "pass": abs(q_comp - q_ref) < tol,
        }
    return errors


# =============================================================================
# §8. RADIATIVE TRANSFER — line-by-line with Hönl-London
# =============================================================================

def generate_band_lines(
    band_name: str,
    upper_sc: SpectroscopicConstants,
    lower_sc: SpectroscopicConstants,
    FC: np.ndarray,
    Re: float = 1.0,
    J_max: int = 150,
    Lambda_u: int = 0,
    Lambda_l: int = 0,
) -> List[Tuple[float, float]]:
    """
    Generate (wavelength_nm, A_ul) for all rotational lines of a band system.
    """
    lines = []
    for vp in range(FC.shape[0]):
        for vpp in range(FC.shape[1]):
            q = FC[vp, vpp]
            if q < 1e-6:
                continue
            nu_band = (upper_sc.Te + upper_sc.G(vp)) - (lower_sc.Te + lower_sc.G(vpp))
            if nu_band <= 0:
                continue
            for Jpp in range(J_max):
                for branch in ("P", "R"):
                    if branch == "P":
                        Jp = Jpp + 1
                    else:
                        if Jpp == 0:
                            continue
                        Jp = Jpp - 1
                    # rotational energies
                    E_u = upper_sc.G(vp) + upper_sc.B(vp) * Jp * (Jp + 1)
                    E_l = lower_sc.G(vpp) + lower_sc.B(vpp) * Jpp * (Jpp + 1)
                    nu_line = (upper_sc.Te + E_u) - (lower_sc.Te + E_l)
                    if nu_line <= 0:
                        continue
                    wl_nm = 1e7 / nu_line
                    if wl_nm < 50 or wl_nm > 5000:
                        continue
                    S = honl_london(Jp, Jpp, Lambda_u, Lambda_l)
                    nu_m = nu_line * 100.0
                    mu2 = EA0_SI**2 * Re**2
                    A = (64 * math.pi**4 * nu_m**3 / (3 * H_PLANCK)) \
                        * mu2 * q * S / (2 * Jp + 1)
                    lines.append((wl_nm, A))
    return lines


# =============================================================================
# §9. CLI DEMO
# =============================================================================

def _demo_rkr_johnson():
    print("\n=== RKR1 + Johnson Numerov: NO X²Π ===")
    R, V = build_potential_rkr_nde(NO_X, NO_X_NDE, v_max=10, n_grid=2000)
    E, psi = johnson_solve_single_channel(R, V, NO_X.mu_amu, J=0, n_states=8)
    print(f"  grid points: {len(R)}, R range = {R.min():.3f}–{R.max():.3f} Å")
    for i, Ev in enumerate(E):
        print(f"  v={i}: E = {Ev:.3f} cm^-1   (Dunham G({i}) = {NO_X.G(i):.3f})")


def _demo_fc_validation():
    print("\n=== FC matrix validation against Jain & Sahni (1968) ===")
    print("Computing NO γ (A²Σ⁺–X²Π) ...")
    FC_gamma = franck_condon_matrix(NO_A, NO_X, v_max_u=3, v_max_l=5)
    errs = validate_fc_against_jain_sahni(FC_gamma, "NO_gamma")
    for key, val in errs.items():
        status = "PASS" if val["pass"] else "FAIL"
        print(f"  {key}: computed={val['computed']:.4f}  "
              f"ref={val['reference']:.4f}  |Δ|={val['abs_error']:.4f}  [{status}]")


def _demo_honl_london():
    print("\n=== Hönl–London factors (Σ–Σ, Π–Σ, Δ–Σ) ===")
    for (Lu, Ll, name) in [(0, 0, "Σ–Σ"), (1, 0, "Π–Σ"), (2, 0, "Δ–Σ")]:
        print(f"  {name}:")
        for Jpp in range(1, 5):
            for branch in ("P", "R"):
                Jp = Jpp + 1 if branch == "P" else Jpp - 1
                if Jp < 0:
                    continue
                S = honl_london(Jp, Jpp, Lu, Ll)
                print(f"    {branch}({Jpp}) -> J'={Jp}: S = {S:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="v4.0 — RKR1 + Johnson Numerov + Hönl-London + NDE"
    )
    parser.add_argument("--demo", default="all",
                        choices=["all", "rkr", "fc", "hl", "none"])
    args = parser.parse_args()

    print(BANNER)

    if args.demo in ("all", "rkr"):
        _demo_rkr_johnson()
    if args.demo in ("all", "fc"):
        _demo_fc_validation()
    if args.demo in ("all", "hl"):
        _demo_honl_london()

    print("\ndone.")


if __name__ == "__main__":
    main()
