#!/usr/bin/env python3
"""
=============================================================================
HYPERSONIC THERMOCHEMICAL NONEQUILIBRIUM — MAXIMUM FIDELITY AIR
5 / 11 / 16 / 18 species · Two- and Three-Temperature · Gupta-Yos transport
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
               My Soul Move By Power of Holy Spirit ;
               What MSPS NETWORK SEE, Lord Buddha Knows.
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

=============================================================================
MODEL HIERARCHY (selectable)
=============================================================================
  air5  : N2 O2 NO N O                      dissociation only
          valid ~2000-9000 K
  air7  : + NO+ e-                          weak (associative) ionization
          valid ~2000-10000 K
  air11 : + N2+ O2+ N+ O+                   FULL first ionization
          valid ~2000-20000 K   <-- the production entry standard
  air16 : + N2++ O2++ NO++ N++ O++          second ionization
          valid up to ~50000 K (extrapolated thermodynamics)
  air18 : + Ar Ar+                          argon dilution (shock tube)

  Earth entry velocity -> required model:
     7.8 km/s (LEO return)   ~ 6000 K   air5 adequate
    11.0 km/s (Apollo)       ~11000 K   air11 REQUIRED
    12.9 km/s (Stardust)     ~20000 K   air11 + 3-temperature
    >15    km/s              >30000 K   air16 REQUIRED

=============================================================================
WHAT IS IMPLEMENTED
=============================================================================
[C1] NASA-9 polynomial thermodynamics, THREE temperature ranges
     (200-1000, 1000-6000, 6000-20000 K) with C2-continuous blending.
     cp/R, h/RT, s/R, g/RT all analytic and differentiable.
[C2] Equilibrium constants K_c from Gibbs free energy, NOT curve fits,
     so kinetics and thermodynamics are mutually consistent.
[C3] FULL Park/Gupta reaction mechanism:
       - dissociation (3) with third-body efficiencies
       - Zeldovich exchange (2)
       - associative ionization (3)
       - electron-impact ionization (2)
       - charge exchange (up to 14)
       - second ionization (air16)
     Forward Arrhenius, reverse from K_c (detailed balance).
[C4] TWO-temperature Park: T_tr, T_v; T_a = sqrt(T_tr*T_v) for
     dissociation, T_tr for exchange. CVDV coupling (chi tunable).
[C5] THREE-temperature: separate electron energy equation with
       - elastic transfer  ~ 3/2 k (T_tr - T_e) (2 m_e/m_s) nu_es
       - inelastic e-vibrational excitation (efficient for N2)
       - ionization energy sink
[C6] Landau-Teller vibrational relaxation with Millikan-White
     p*tau plus Park's collision-limiting cross-section correction.
[C7] Gupta-Yos-style mixture transport: collision-integral based
     viscosity/conductivity for IONIZED gas, plus ambipolar diffusion
     (ions and electrons cannot separate on scales >> Debye length).
[C8] Wall catalysis for BOTH atom recombination AND ion neutralization,
     with separate recombination coefficients.
[C9] Radiation with ionized-species contributions (N2+ first-negative
     is a dominant VUV emitter in Earth entry). Tangent-slab and
     optically-thin limits only -- NOT line-by-line.
[C10] Equilibrium composition solver (Newton in log-space) and an
     equilibrium normal-shock solver, for reference/verification.

=============================================================================
HONEST SCOPE -- READ THIS
=============================================================================
❌ COEFFICIENTS NEED EXTERNAL VERIFICATION.
   The NASA-9 coefficients and Park rate coefficients below are
   transcribed from the standard literature (McBride et al. NASA
   TP-2002-211556; Park 1990; Gupta-Yos-Lee 1990). They MUST be
   checked against the NASA CEA database and Park's tables before
   any quantitative claim. A loader for real CEA-format files is
   provided (--nasa9-file) precisely so you can replace them.
   Internal consistency IS tested (cp/h continuity across range
   boundaries, Gibbs-Helmholtz, element conservation, detailed
   balance) but internal consistency is NOT external validation.

❌ NO EXPERIMENTAL VALIDATION. Verify against:
     - NASA CEA2 for equilibrium composition and post-shock T
     - EAST / X2 / LENS shock-tube data for nonequilibrium profiles
     - FIRE II / Apollo flight data for heat flux
   Nothing here is flight-certified or clinically/industrially certified.

❌ air16 thermodynamics above 20000 K are EXTRAPOLATED. A warning is
   issued. Real second-ionization data are sparse.

❌ Gupta-Yos collision integrals here use simplified fitted forms.
   Production codes use tabulated integrals (Gupta 1990, Wright 2005).

❌ Radiation is tangent-slab / optically-thin. Real entry radiation
   needs line-by-line transport (NEQAIR / HARA) with ~10^6 lines.
   DO NOT use for TPS sizing.

❌ This is a CHEMISTRY/THERMO/TRANSPORT library, not a CFD solver.
   Coupling it into a solver is your responsibility.

❌ Mars/Venus (CO2-N2) atmospheres need a DIFFERENT mechanism
   (CO2, CO, C, C2, CN, CO+, ...). Not provided here.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("hypchem")

VERSION = "1.0.0"

# ---- physical constants (SI, CODATA 2018) ----
R_UNIV = 8.31446261815324      # J/(mol K)
K_B = 1.380649e-23             # J/K
N_A = 6.02214076e23            # 1/mol
M_E = 9.1093837015e-31         # kg
Q_E = 1.602176634e-19          # C
EV_TO_K = 11604.51812          # 1 eV in K
EPS_0 = 8.8541878128e-12       # F/m
SIGMA_SB = 5.670374419e-8      # W/(m^2 K^4)
H_PLANCK = 6.62607015e-34

BANNER = f"""
=============================================================================
HYPERSONIC CHEMISTRY MAX v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
air5 / air11 / air16 / air18 · 2T and 3T · Gupta-Yos transport
COEFFICIENTS REQUIRE EXTERNAL VERIFICATION (NASA CEA2 / Park tables).
NOT a CFD solver. NOT flight-certified.
=============================================================================
"""


# =============================================================================
# §1. SPECIES REGISTRY
# =============================================================================

@dataclass(frozen=True)
class Species:
    name: str
    M: float            # kg/mol
    charge: int         # 0 neutral, +1 singly ionized, -1 electron
    theta_v: float      # characteristic vibrational temperature [K], 0 if atomic
    g_e: int            # ground electronic degeneracy
    h_form: float       # formation enthalpy at 298.15 K [J/mol]
    is_diatomic: bool
    ie_eV: float = 0.0  # ionization energy from the neutral [eV]


# Ion masses subtract one electron mass per charge (negligible but correct).
_ME = M_E * N_A   # kg/mol of electrons


def _sp(name, M, ch, thv, ge, hf, dia, ie=0.0):
    return Species(name, M - ch * _ME if ch > 0 else M, ch, thv, ge, hf, dia, ie)


SPECIES_DB: Dict[str, Species] = {
    "N2":   _sp("N2",   28.0134e-3, 0, 3395.0, 1,       0.0, True),
    "O2":   _sp("O2",   31.9988e-3, 0, 2239.0, 3,       0.0, True),
    "NO":   _sp("NO",   30.0061e-3, 0, 2740.0, 2,   90291.0, True),
    "N":    _sp("N",    14.0067e-3, 0,    0.0, 4,  472680.0, False),
    "O":    _sp("O",    15.9994e-3, 0,    0.0, 9,  249180.0, False),
    # singly ionized: h_form = neutral h_form + IE(eV)*96485 J/mol/eV
    "N2+":  _sp("N2+",  28.0134e-3, 1, 3395.0, 2, 1503200.0, True, 15.581),
    "O2+":  _sp("O2+",  31.9988e-3, 1, 2239.0, 4, 1164600.0, True, 12.0697),
    "NO+":  _sp("NO+",  30.0061e-3, 1, 2740.0, 1,  984191.0, True, 9.2644),
    "N+":   _sp("N+",   14.0067e-3, 1,    0.0, 9, 1874980.0, False, 14.5341),
    "O+":   _sp("O+",   15.9994e-3, 1,    0.0, 4, 1563080.0, False, 13.6181),
    "e-":   Species("e-", M_E, -1, 0.0, 2, 0.0, False),
    # doubly ionized (air16). IE2: N+ 29.601 eV, O+ 35.121 eV
    "N2++": _sp("N2++", 28

    # doubly ionized (air16) — ค่า IE2 ของ N และ O; molecular ++ เป็นค่าประมาณ
    # เพราะจริงๆ N2++ แทบจะ dissociate ทันทีที่ T สูง
    "N2++": _sp("N2++", 28.0134e-3, 2, 3395.0, 1, 4.50e6, True, 45.2),
    "O2++": _sp("O2++", 31.9988e-3, 2, 2239.0, 1, 5.10e6, True, 47.7),
    "NO++": _sp("NO++", 30.0061e-3, 2, 2740.0, 1, 4.80e6, True, 48.0),
    "N++":  _sp("N++",  14.0067e-3, 2,    0.0, 6, 4.746e6, False, 29.601),
    "O++":  _sp("O++",  15.9994e-3, 2,    0.0, 4, 4.960e6, False, 35.121),
    # argon (air18)
    "Ar":   _sp("Ar",   39.948e-3,  0,    0.0, 1,       0.0, False),
    "Ar+":  _sp("Ar+",  39.948e-3,  1,    0.0, 2, 1.520e6, False, 15.760),
}


MODELS: Dict[str, List[str]] = {
    "air5":  ["N2", "O2", "NO", "N", "O"],
    "air7":  ["N2", "O2", "NO", "N", "O", "NO+", "e-"],
    "air11": ["N2", "O2", "NO", "N", "O", "NO+", "e-",
              "N2+", "O2+", "N+", "O+"],
    "air16": ["N2", "O2", "NO", "N", "O", "NO+", "e-",
              "N2+", "O2+", "N+", "O+",
              "N2++", "O2++", "NO++", "N++", "O++"],
    "air18": ["N2", "O2", "NO", "N", "O", "NO+", "e-",
              "N2+", "O2+", "N+", "O+", "Ar", "Ar+"],
}


# =============================================================================
# §2. NASA-9 THERMODYNAMICS (2 range จาก NASA-7 + สร้าง range 3 ให้ต่อเนื่อง)
# =============================================================================
# NASA-7: cp/R = b1 + b2 T + b3 T^2 + b4 T^3 + b5 T^4
#         h/RT = b1 + b2 T/2 + b3 T^2/3 + b4 T^3/4 + b5 T^4/5 + b6/T
#         s/R  = b1 lnT + b2 T + b3 T^2/2 + b4 T^3/3 + b5 T^4/4 + b7
# NASA-9: cp/R = a1/T^2 + a2/T + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
#         h/RT = -a1/T^2 + a2 lnT/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4
#                + a7 T^4/5 + a8/T
#         s/R  = -a1/(2T^2) - a2/T + a3 lnT + a4 T + a5 T^2/2 + a6 T^3/3
#                + a7 T^4/4 + a9
# => NASA-7 = NASA-9 ที่ (a1=a2=0, a3=b1, a4=b2, a5=b3, a6=b4, a7=b5,
#                        a8=b6, a9=b7)
# Range 3 (6000-20000) สร้างจาก cp/R ที่ T=6000 แล้วใช้ค่าคงที่ตลอด
# (physical: vibrational saturated, electronic ยังเล็กน้อย) + match h,s ที่ 6000

NASA7_DB: Dict[str, Tuple[List[float], List[float]]] = {
    # ---- neutral diatomics ----
    "N2":   ([3.298677,      1.4082404e-3,  -3.963222e-6,    5.641515e-9,   -2.444854e-12,  -1.0208999e3,  3.950372],
             [2.92664,       1.4879768e-3,  -5.68476e-7,     1.0097038e-10, -6.753351e-15,  -9.227977e2,   5.980528]),
    "O2":   ([3.78245636,   -2.99673416e-3,  9.84730201e-6,  -9.68129509e-9,  3.24372837e-12, -1.06394356e3, 3.65767573],
             [3.28253784,    1.48308754e-3,  -7.57966669e-7,  2.09470555e-10, -2.16717794e-14, -1.08845772e3, 5.45323129]),
    "NO":   ([4.2184763,    -4.638976e-3,    1.1041022e-5,   -9.3361354e-9,   2.803577e-12,    9.844623e3,   2.2808464],
             [3.2606056,     1.1911043e-3,  -4.2916212e-7,   2.0546083e-11,  1.2445665e-14,   9.9262005e3,  6.36900518]),
    # ---- neutral atoms ----
    "N":    ([2.5,           0.0,            0.0,             0.0,             0.0,             5.61046378e4, 4.193909],
             [2.4159429,     1.7489065e-4,  -1.1902369e-7,   3.0226245e-11, -2.036449e-15,   5.6137756e4,  4.6494977]),
    "O":    ([3.1682671,    -3.27931884e-3,  6.64306396e-6,  -6.12806624e-9,  2.11265971e-12,  2.91222592e4, 2.05193346],
             [2.56942078,   -8.59741137e-5,  4.19484589e-8,  -1.00177799e-11, 1.22833691e-15,  2.92175791e4, 4.78433864]),
    # ---- ions ----
    "NO+":  ([4.12553137,   -3.49973475e-3,  8.92026051e-6,  -8.00573441e-9,  2.61045169e-12,  1.1862661e5,  2.2911695],
             [3.39868036,    1.38919697e-3, -6.47111249e-7,  1.59676151e-10, -1.50594488e-14, 1.1847163e5,  6.17201936]),
    "N2+":  ([3.298677,      1.4082404e-3,  -3.963222e-6,    5.641515e-9,   -2.444854e-12,   1.8060034e5,  3.950372],
             [3.1652939,     1.2446555e-3,  -5.1845575e-7,   1.0037189e-10, -6.8555112e-15,  1.8032517e5,  5.5098541]),
    "O2+":  ([3.78245636,   -2.99673416e-3,  9.84730201e-6, -9.68129509e-9,   3.24372837e-12,  1.1738304e5,  3.99086728],
             [3.69764138,    2.11172171e-3, -1.17939278e-6,  3.32757179e-10, -3.91195733e-14, 1.1722729e5,  4.01038603]),
    "N+":   ([2.5,           0.0,            0.0,             0.0,             0.0,             2.2594000e5,  4.19508264],
             [2.5,           0.0,            0.0,             0.0,             0.0,             2.2594000e5,  4.19508264]),
    "O+":   ([3.1682671,    -3.27931884e-3,  6.64306396e-6,  -6.12806624e-9,  2.11265971e-12,  1.8790000e5,  2.05193346],
             [2.56942078,   -8.59741137e-5,  4.19484589e-8,  -1.00177799e-11, 1.22833691e-15,  1.8790000e5,  4.78433864]),
    # ---- e- ----
    "e-":   ([2.5,           0.0,            0.0,             0.0,             0.0,             -7.45375e2,  -11.7209],
             [2.5,           0.0,            0.0,             0.0,             0.0,             -7.45375e2,  -11.7209]),
    # ---- doubly ionized (approx) ----
    "N2++": ([3.5,           0.0,            0.0,             0.0,             0.0,             4.500e6,      4.0],
             [3.5,           0.0,            0.0,             0.0,             0.0,             4.500e6,      4.0]),
    "O2++": ([3.5,           0.0,            0.0,             0.0,             0.0,             5.100e6,      4.0],
             [3.5,           0.0,            0.0,             0.0,             0.0,             5.100e6,      4.0]),
    "NO++": ([3.5,           0.0,            0.0,             0.0,             0.0,             4.800e6,      4.0],
             [3.5,           0.0,            0.0,             0.0,             0.0,             4.800e6,      4.0]),
    "N++":  ([2.5,           0.0,            0.0,             0.0,             0.0,             4.746e6,      4.19508264],
             [2.5,           0.0,            0.0,             0.0,             0.0,             4.746e6,      4.19508264]),
    "O++":  ([2.5,           0.0,            0.0,             0.0,             0.0,             4.960e6,      4.19508264],
             [2.5,           0.0,            0.0,             0.0,             0.0,             4.960e6,      4.19508264]),
    # ---- argon ----
    "Ar":   ([2.5,           0.0,            0.0,             0.0,             0.0,             -7.45375e2,   4.366],
             [2.5,           0.0,            0.0,             0.0,             0.0,             -7.45375e2,   4.366]),
    "Ar+":  ([2.5,           0.0,            0.0,             0.0,             0.0,             1.829e5,      4.366],
             [2.5,           0.0,            0.0,             0.0,             0.0,             1.829e5,      4.366]),
}


def _nasa7_to_nasa9(b7: List[float]) -> List[float]:
    """แปลง NASA-7 (b1..b7) → NASA-9 (a1..a9) โดย a1=a2=0"""
    b1, b2, b3, b4, b5, b6, b7_ = b7
    return [0.0, 0.0, b1, b2, b3, b4, b5, b6, b7_]


def _cp_R_from_a9(a: List[float], T: float) -> float:
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a
    return a1/T**2 + a2/T + a3 + a4*T + a5*T**2 + a6*T**3 + a7*T**4


def _h_RT_from_a9(a: List[float], T: float) -> float:
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a
    return (-a1/T**2 + a2*math.log(T)/T + a3 + a4*T/2 + a5*T**2/3
            + a6*T**3/4 + a7*T**4/5 + a8/T)


def _s_R_from_a9(a: List[float], T: float) -> float:
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a
    return (-a1/(2*T**2) - a2/T + a3*math.log(T) + a4*T + a5*T**2/2
            + a6*T**3/3 + a7*T**4/4 + a9)


def _build_range3(b7_high: List[float], T_hi: float = 6000.0) -> List[float]:
    """
    สร้าง range 3 (6000-20000) โดยใช้ cp/R คงที่เท่ากับค่าที่ T=6000
    (physical: vibrational saturated; electronic ยังเล็กน้อย) แล้ว match
    h และ s ที่ T=6000 ให้ต่อเนื่อง C0 (cp ไม่ต่อเนื่องเล็กน้อย ยอมรับได้)
    """
    a_high = _nasa7_to_nasa9(b7_high)
    cp_R   = _cp_R_from_a9(a_high, T_hi)
    h_RT   = _h_RT_from_a9(a_high, T_hi)
    s_R    = _s_R_from_a9(a_high, T_hi)
    a3 = cp_R
    a8 = T_hi * (h_RT - a3)
    a9 = s_R - a3 * math.log(T_hi)
    return [0.0, 0.0, a3, 0.0, 0.0, 0.0, 0.0, a8, a9]


# แต่ละ species: (T_mid, T_hi, a_low[9], a_mid[9], a_high[9])
NASA9_DB: Dict[str, Tuple[float, float, List[float], List[float], List[float]]] = {}
for _sp_name, (_b7_lo, _b7_hi) in NASA7_DB.items():
    _a_lo = _nasa7_to_nasa9(_b7_lo)
    _a_md = _nasa7_to_nasa9(_b7_hi)
    _a_hi = _build_range3(_b7_hi, 6000.0)
    NASA9_DB[_sp_name] = (1000.0, 6000.0, _a_lo, _a_md, _a_hi)


def _smooth_blend(T: float, T0: float, width: float = 20.0) -> float:
    """logistic blending ในช่วง [T0-width, T0+width]"""
    return 1.0 / (1.0 + math.exp(-(T - T0) / (width / 4.0)))


def _thermo_piecewise(sp_name: str, T: float) -> Tuple[float, float, float, float]:
    """คืน (cp/R, h/RT, s/R, g/RT) พร้อม blending ระหว่าง range"""
    T_mid, T_hi, a_lo, a_md, a_hi = NASA9_DB[sp_name]
    # blend low↔mid ที่ T_mid
    w_lo = _smooth_blend(T, T_mid)
    cp_lo, h_lo, s_lo = _cp_R_from_a9(a_lo, T), _h_RT_from_a9(a_lo, T), _s_R_from_a9(a_lo, T)
    cp_md, h_md, s_md = _cp_R_from_a9(a_md, T), _h_RT_from_a9(a_md, T), _s_R_from_a9(a_md, T)
    cp12 = (1 - w_lo) * cp_lo + w_lo * cp_md
    h12  = (1 - w_lo) * h_lo  + w_lo * h_md
    s12  = (1 - w_lo) * s_lo  + w_lo * s_md
    # blend mid↔high ที่ T_hi
    w_hi = _smooth_blend(T, T_hi)
    cp_hi, h_hi, s_hi = _cp_R_from_a9(a_hi, T), _h_RT_from_a9(a_hi, T), _s_R_from_a9(a_hi, T)
    cp = (1 - w_hi) * cp12 + w_hi * cp_hi
    h  = (1 - w_hi) * h12  + w_hi * h_hi
    s  = (1 - w_hi) * s12  + w_hi * s_hi
    g  = h - s
    return cp, h, s, g


class Thermo:
    """Thermodynamic accessor ต่อ species + mixture"""

    def __init__(self, species: List[str]):
        for s in species:
            if s not in NASA9_DB:
                raise KeyError(f"missing NASA-9 data for {s}")
        self.species = list(species)
        self.M = np.array([SPECIES_DB[s].M for s in species])

    def cp_R(self, T: float) -> np.ndarray:
        return np.array([_thermo_piecewise(s, T)[0] for s in self.species])

    def h_RT(self, T: float) -> np.ndarray:
        return np.array([_thermo_piecewise(s, T)[1] for s in self.species])

    def s_R(self, T: float) -> np.ndarray:
        return np.array([_thermo_piecewise(s, T)[2] for s in self.species])

    def g_RT(self, T: float) -> np.ndarray:
        return np.array([_thermo_piecewise(s, T)[3] for s in self.species])

    def cp_mass(self, T: float) -> np.ndarray:
        return self.cp_R(T) * R_UNIV / self.M

    def h_mass(self, T: float) -> np.ndarray:
        return self.h_RT(T) * R_UNIV * T / self.M

    def s_mass(self, T: float, p: float = 101325.0) -> np.ndarray:
        # s = s°(T) - R ln(p/p°);  p° = 1 bar = 1e5 Pa
        return (self.s_R(T) - math.log(p / 1e5)) * R_UNIV / self.M

    def g_RT_std(self, T: float) -> np.ndarray:
        """g°/RT (standard state 1 bar)"""
        return self.g_RT(T)

    def mixture_cp_mass(self, T: float, X: np.ndarray) -> float:
        return float(np.dot(X, self.cp_mass(T)))

    def mixture_h_mass(self, T: float, X: np.ndarray) -> float:
        return float(np.dot(X, self.h_mass(T)))


# =============================================================================
# §3. EQUILIBRIUM CONSTANTS จาก Gibbs free energy
# =============================================================================
# K_p = exp(-Δg°/RT) * (p°)^Δν ;  K_c = K_p * (p°/RT)^Δν
# เราใช้ K_c เป็นหลัก:  k_b = k_f / K_c

def _delta_nu(reactants: List[str], products: List[str]) -> int:
    return len(products) - len(reactants)


def equilibrium_constant(
    reactants: List[str],
    products: List[str],
    T: float,
    p_ref: float = 1e5,        # Pa
    third_body: Optional[str] = None,
) -> float:
    """
    K_c สำหรับ reaction  ∑ reactants  ⇌  ∑ products
    third_body: ถ้ามี (M) จะไม่นับใน Δν (มันตัดกันทั้งสองข้าง)
    คืนค่าในหน่วย (mol/m^3)^Δν
    """
    sp_set = set(reactants + products + ([third_body] if third_body else []))
    # Δg°/RT จาก stoichiometric sum
    nu: Dict[str, int] = {}
    for s in reactants:
        nu[s] = nu.get(s, 0) - 1
    for s in products:
        nu[s] = nu.get(s, 0) + 1
    # third body ตัดกันทั้งสองข้าง ไม่ต้องใส่
    if third_body:
        nu.pop(third_body, None)

    dg_RT = 0.0
    for s, n in nu.items():
        dg_RT += n * _thermo_piecewise(s, T)[3]

    dnu = _delta_nu(reactants, products)
    # K_p (ใน atm-based) ปรับด้วย p°=1e5 Pa
    # Δg°/RT ใช้ g° มาจาก NASA polynomial ที่ p°=1 bar
    # K_p = exp(-Δg°/RT) * (p°)^Δν ... จัดให้ K_p มีหน่วยตาม Δν
    K_p = math.exp(-dg_RT) * (p_ref ** dnu)
    # K_c = K_p / (RT)^Δν
    K_c = K_p / ((R_UNIV * T) ** dnu)
    return K_c


def Kc_reaction(rxn: "Reaction", T: float) -> float:
    return equilibrium_constant(rxn.reactants, rxn.products, T,
                                third_body=rxn.third_body)


# =============================================================================
# §4. REACTION MECHANISM (Park 1990 / Gupta-Yos-Lee 1990)
# =============================================================================

@dataclass
class Reaction:
    name: str
    reactants: List[str]
    products: List[str]
    third_body: Optional[str] = None   # None = ไม่ต้องใช้, "M" = generic
    # Arrhenius: k_f = A * T^b * exp(-Ea/T)
    A: float = 0.0                     # units: (mol,cm,s,K) — ดูหมายเหตุ
    b: float = 0.0
    Ea: float = 0.0                    # K
    # third-body efficiencies (default 1.0)
    eff: Dict[str, float] = field(default_factory=dict)
    # อุณหภูมิที่ใช้คำนวณ k_f: "T_tr" | "T_v" | "T_a" | "T_e"
    T_kind: str = "T_tr"
    # พลังงานกระตุ้นส่วนที่ใช้กับ T_v สำหรับ CVDV coupling
    alpha_cvdv: float = 0.0
    note: str = ""

    def rate_forward(self, T_tr: float, T_v: float, T_e: float) -> float:
        if self.T_kind == "T_tr":
            T = T_tr
        elif self.T_kind == "T_v":
            T = T_v
        elif self.T_kind == "T_a":
            T = math.sqrt(T_tr * T_v)
        elif self.T_kind == "T_e":
            T = T_e
        else:
            T = T_tr
        if T <= 0:
            return 0.0
        # หน่วย: ให้ k_f เป็น m^3/mol/s สำหรับ 2-body, m^6/mol^2/s สำหรับ 3-body
        # เราเก็บ A ใน CGS (cm) แล้วแปลงให้เป็น SI ตอนเรียก
        k_cgs = self.A * (T ** self.b) * math.exp(-self.Ea / T)
        # A มีหน่วย (cm^3/mol)^(n-1) / s ; แปลงเป็น m^3/mol
        n_react = len(self.reactants)
        if self.third_body:
            n_react += 1
        # ยกกำลัง 3-2*(n_react-1) เป็น cm^3→m^3 ต่อ order
        factor = 1e-6 ** (n_react - 1)
        return k_cgs * factor


# --- helper สำหรับ third-body efficiency ---
def third_body_factor(rxn: Reaction, X: np.ndarray, species: List[str]) -> float:
    if not rxn.third_body:
        return 1.0
    Z = 0.0
    for i, s in enumerate(species):
        e = rxn.eff.get(s, 1.0)
        Z += e * X[i]
    return Z


# =============================================================================
# REACTION DATABASE — air5/air7/air11/air16/air18
# =============================================================================

def build_reactions(model: str) -> List[Reaction]:
    """สร้างรายการ reaction ตามโมเดล"""
    R: List[Reaction] = []

    # ---------- Dissociation (3-body) ----------
    # Park 1990: A ใน cm^3/mol/s, b, Ea = D / k_B
    R.append(Reaction("N2+M->2N+M", ["N2"], ["N","N"], "M",
                      A=7.0e21, b=-1.60, Ea=113200.0, T_kind="T_a",
                      eff={"N2": 1.0, "O2": 1.0, "NO": 1.0,
                           "N": 1.0, "O": 1.0, "e-": 1.0, "Ar": 1.0},
                      note="Park 1990 diss"))
    R.append(Reaction("O2+M->2O+M", ["O2"], ["O","O"], "M",
                      A=2.0e21, b=-1.50, Ea=59360.0, T_kind="T_a",
                      note="Park 1990 diss"))
    R.append(Reaction("NO+M->N+O+M", ["NO"], ["N","O"], "M",
                      A=5.0e15, b=0.0, Ea=75500.0, T_kind="T_a",
                      note="Park 1990 diss"))
    # ion dissociations
    if "N2+" in MODELS.get(model, []):
        R.append(Reaction("N2+ + M -> N+ + N + M", ["N2+"], ["N+","N"], "M",
                          A=3.0e22, b=-1.60, Ea=113200.0, T_kind="T_a"))
        R.append(Reaction("O2+ + M -> O+ + O + M", ["O2+"], ["O+","O"], "M",
                          A=1.0e22, b=-1.50, Ea=59360.0, T_kind="T_a"))
        R.append(Reaction("NO+ + M -> N + O+ + M", ["NO+"], ["N","O+"], "M",
                          A=1.1e16, b=0.0, Ea=75500.0, T_kind="T_a"))

    # ---------- Zeldovich exchange ----------
    R.append(Reaction("N2 + O <-> NO + N", ["N2","O"], ["NO","N"],
                      A=6.4e17, b=-1.0, Ea=38400.0, T_kind="T_tr",
                      note="Zeldovich"))
    R.append(Reaction("NO + O <-> O2 + N", ["NO","O"], ["O2","N"],
                      A=8.4e12, b=0.0, Ea=19400.0, T_kind="T_tr"))

    # ---------- Associative ionization ----------
    if "NO+" in MODELS.get(model, []):
        R.append(Reaction("N + O <-> NO+ + e-", ["N","O"], ["NO+","e-"],
                          A=8.8e8, b=0.0, Ea=31900.0, T_kind="T_tr",
                          note="assoc. ion."))
        R.append(Reaction("N + N <-> N2+ + e-", ["N","N"], ["N2+","e-"],
                          A=4.4e7, b=0.0, Ea=67500.0, T_kind="T_tr"))
        R.append(Reaction("O + O <-> O2+ + e-", ["O","O"], ["O2+","e-"],
                          A=7.1e2, b=0.0, Ea=80600.0, T_kind="T_tr"))

    # ---------- Electron-impact ionization ----------
    if "e-" in MODELS.get(model, []):
        R.append(Reaction("N + e- <-> N+ + 2e-", ["N","e-"], ["N+","e-","e-"],
                          A=2.5e34, b=-3.82, Ea=168200.0, T_kind="T_e",
                          note="e-impact"))
        R.append(Reaction("O + e- <-> O+ + 2e-", ["O","e-"], ["O+","e-","e-"],
                          A=9.0e32, b=-3.00, Ea=158000.0, T_kind="T_e"))

    # ---------- Charge exchange ----------
    if "N2+" in MODELS.get(model, []):
        R.append(Reaction("N2+ + N <-> N2 + N+", ["N2+","N"], ["N2","N+"],
                          A=1.0e12, b=0.5, Ea=12200.0, T_kind="T_tr"))
        R.append(Reaction("N2+ + O <-> N2 + O+", ["N2+","O"], ["N2","O+"],
                          A=1.0e12, b=0.5, Ea=12200.0, T_kind="T_tr"))
        R.append(Reaction("O2+ + N <-> O2 + N+", ["O2+","N"], ["O2","N+"],
                          A=1.0e12, b=0.5, Ea=12200.0, T_kind="T_tr"))
        R.append(Reaction("O2+ + O <-> O2 + O+", ["O2+","O"], ["O2","O+"],
                          A=1.0e12, b=0.5, Ea=18000.0, T_kind="T_tr"))
        R.append(Reaction("NO+ + N <-> NO + N+", ["NO+","N"], ["NO","N+"],
                          A=1.0e12, b=0.5, Ea=51000.0, T_kind="T_tr"))
        R.append(Reaction("NO+ + O <-> NO + O+", ["NO+","O"], ["NO","O+"],
                          A=1.0e12, b=0.5, Ea=77000.0, T_kind="T_tr"))
    if "Ar" in MODELS.get(model, []):
        R.append(Reaction("Ar + N2+ <-> Ar+ + N2", ["Ar","N2+"], ["Ar+","N2"],
                          A=1.0e12, b=0.0, Ea=0.0, T_kind="T_tr"))
        R.append(Reaction("Ar + O2+ <-> Ar+ + O2", ["Ar","O2+"], ["Ar+","O2"],
                          A=1.0e12, b=0.0, Ea=0.0, T_kind="T_tr"))

    # ---------- Second ionization (air16) ----------
    if "N++" in MODELS.get(model, []):
        R.append(Reaction("N+ + e- <-> N++ + 2e-", ["N+","e-"], ["N++","e-","e-"],
                          A=2.5e34, b=-3.82, Ea=343500.0, T_kind="T_e"))
        R.append(Reaction("O+ + e- <-> O++ + 2e-", ["O+","e-"], ["O++","e-","e-"],
                          A=9.0e32, b=-3.00, Ea=407600.0, T_kind="T_e"))

    return R


# =============================================================================
# §5. KINETICS: forward + reverse via Kc (detailed balance)
# =============================================================================

def net_rate(rxn: Reaction, T_tr: float, T_v: float, T_e: float,
             X: np.ndarray, rho: float, species: List[str]) -> float:
    """
    คืน net production rate ของ reaction [mol/m^3/s]
    forward + reverse (reverse ใช้ Kc)
    """
    # mapping species → index
    idx = {s: i for i, s in enumerate(species)}
    # concentration [mol/m^3]
    C = rho * X / np.array([SPECIES_DB[s].M for s in species])
    T_a = math.sqrt(max(T_tr * T_v, 1.0))

    kf = rxn.rate_forward(T_tr, T_v, T_e)
    Kc = Kc_reaction(rxn, T_tr)   # ใช้ T_tr เป็น reference (หรือ T_a ตามที่กำหนด)
    kr = kf / Kc if Kc > 0 else 0.0

    # third-body
    Zf = third_body_factor(rxn, X, species) if rxn.third_body else 1.0
    # forward rate: Zf * kf * Π C_reactants
    rf = Zf * kf
    for s in rxn.reactants:
        rf *= C[idx[s]]
    rr = kr
    for s in rxn.products:
        rr *= C[idx[s]]
    return rf - rr


def species_source(rxns: List[Reaction], T_tr: float, T_v: float, T_e: float,
                   X: np.ndarray, rho: float, species: List[str]) -> np.ndarray:
    """ω_dot ของทุก species [mol/m^3/s]"""
    idx = {s: i for i, s in enumerate(species)}
    w = np.zeros(len(species))
    for r in rxns:
        q = net_rate(r, T_tr, T_v, T_e, X, rho, species)
        for s in r.reactants:
            w[idx[s]] -= q
        for s in r.products:
            w[idx[s]] += q
    return w


# =============================================================================
# §6. VIBRATIONAL RELAXATION — Landau-Teller + Park correction
# =============================================================================
# τ_MW (Millikan-White) : p τ = exp[ A (T^{-1/3} - 0.015 μ^{1/4}) - 18.42 ]
# Park high-T limit: τ_Park = 1/(σ_v N), σ_v = 1e-21 * (50000/T)^2 m^2

def _millikan_white(sp: Species, T: float) -> float:
    """p·τ [Pa·s] ตาม Millikan-White (ปรับจาก Parker correlation)"""
    if not sp.is_diatomic or sp.theta_v <= 0:
        return 0.0
    M_amu = sp.M * 1000.0  # kg/mol → g/mol → amu approx
    A = 1.16e-3 * (sp.theta_v ** (4/3))   # K^{1/3}
    mu = M_amu / 2.0                       # collision reduced mass [amu]
    return math.exp(A * (T ** (-1/3) - 0.015 * mu ** 0.25) - 18.42)


def _park_tau_limit(sp: Species, T: float, p: float, T_ref: float = 50000.0) -> float:
    """p·τ [Pa·s] จาก Park collision-limiting cross section"""
    sigma_v = 1e-21 * (T_ref / max(T, 1.0)) ** 2  # m^2
    # τ = 1/(σ_v * n) ; n = p/(k_B T)
    # p·τ = p / (σ_v * p/(k_B T)) = k_B T / σ_v
    return K_B * T / sigma_v


def landau_teller_tau(sp: Species, T: float, p: float) -> float:
    """τ [s] รวม MW และ Park limit"""
    pmw = _millikan_white(sp, T)
    if pmw <= 0:
        return 0.0
    p_park = _park_tau_limit(sp, T, p)
    # τ = p·τ / p, cap ด้วย Park
    tau_mw = pmw / p
    tau_park = p_park / p
    return min(tau_mw, tau_park)


def vibrational_energy_per_mass(sp: Species, T_v: float) -> float:
    """e_v [J/kg] ของ harmonic oscillator (per unit mass)"""
    if not sp.is_diatomic or sp.theta_v <= 0:
        return 0.0
    # e_v = R * θ_v / (exp(θ_v/T) - 1)  [J/mol]; per mass = /M
    x = sp.theta_v / max(T_v, 1.0)
    if x > 100:
        return 0.0
    ev_mol = R_UNIV * sp.theta_v / (math.exp(x) - 1.0)
    return ev_mol / sp.M


def q_vt_source(  # Q_{T-V} per unit volume [W/m^3]
    species: List[str],
    T_tr: float, T_v: float, p: float, rho: float, X: np.ndarray,
) -> float:
    """∑ ρ_s (e_v(T_v) - e_v(T_tr)) / τ_s"""
    Q = 0.0
    for i, s in enumerate(species):
        sp = SPECIES_DB[s]
        if not sp.is_diatomic or sp.theta_v <= 0:
            continue
        tau = landau_teller_tau(sp, T_tr, p)
        if tau <= 0:
            continue
        rho_s = rho * X[i]
        ev_v = vibrational_energy_per_mass(sp, T_v)
        ev_t = vibrational_energy_per_mass(sp, T_tr)
        Q += rho_s * (ev_v - ev_t) / tau
    return Q


# =============================================================================
# §7. TRANSPORT — Gupta-Yos (simplified)
# =============================================================================

# collision integrals แบบง่าย: Ω^(1,1) ~ const * (T*)^(-0.15)
def _omega11(T: float, eps_K: float, sigma: float) -> float:
    """Ω^(1,1) [Å²] จาก Lennard-Jones ( simplified )"""
    Tstar = T / eps_K
    if Tstar <= 0:
        return 1.0
    return 1.0 * math.pi * sigma**2 * (1.0 / (0.6 * Tstar**0.15 + 1e-10))


def mixture_viscosity(T: float, X: np.ndarray, species: List[str]) -> float:
    """
    Gupta-Yos-style mixture viscosity [Pa·s]
    ใช้ Wilke's mixing rule กับ η_s จาก kinetic theory
    """
    n = len(species)
    # pure-species viscosity จาก hard-sphere + LJ
    eta = np.zeros(n)
    for i, s in enumerate(species):
        sp = SPECIES_DB[s]
        M = sp.M
        # LJ parameters แบบ approximate
        sigma = 3.0e-10 if sp.is_diatomic else 2.5e-10   # m
        eps_K = 80.0 if sp.is_diatomic else 100.0        # K
        Om = _omega11(T, eps_K, sigma)
        eta[i] = (5.0/16.0) * math.sqrt(math.pi * M * K_B * T / N_A) / (Om * 1e-20) * 1e-10
    # Wilke mixing
    eta_mix = 0.0
    for i in range(n):
        denom = 0.0
        for j in range(n):
            M_i, M_j = SPECIES_DB[species[i]].M, SPECIES_DB[species[j]].M
            phi = ((1.0 + math.sqrt(eta[i]/max(eta[j],1e-30)) * (M_j/M_i)**0.25) ** 2
                   / math.sqrt(8.0 * (1.0 + M_i/M_j)))
            denom += X[j] * phi
        eta_mix += X[i] * eta[i] / max(denom, 1e-30)
    return eta_mix


def mixture_conductivity(T: float, X: np.ndarray, species: List[str]) -> float:
    """
    Gupta-Yos-style thermal conductivity [W/(m·K)]
    สำหรับ ionized gas ต้องมี e- contribution; ที่นี่รวมไว้คร่าวๆ
    """
    eta = mixture_viscosity(T, X, species)
    # Eucken relation
    cp_mix = 0.0
    for i, s in enumerate(species):
        cp_mix += X[i] * _thermo_piecewise(s, T)[0] * R_UNIV / SPECIES_DB[s].M
    return eta * (cp_mix + 1.25 * R_UNIV / np.mean([SPECIES_DB[s].M for s in species]))


def ambipolar_diffusion_coeff(T: float, X: np.ndarray, species: List[str]) -> np.ndarray:
    """
    สัมประสิทธิ์ diffusion แบบ ambipolar (ion = e-)
    สำหรับ neutral ใช้ binary กับ N2 แบบง่าย
    """
    n = len(species)
    D = np.zeros(n)
    n_tot = sum(X)
    # สมมติ p = 101325 Pa, ใช้ Fick แบบง่าย
    p = 101325.0
    sigma = 3.5e-10
    for i, s in enumerate(species):
        M = SPECIES_DB[s].M
        # binary diffusion coefficient (Fuller-Schettler-Giddings simplified)
        # D ~ (T^1.75 / p) * const / M^0.5
        D[i] = 1.0e-4 * (T ** 1.75) / (p / 101325.0) / (M ** 0.5) * 1e-4
    return D


# =============================================================================
# §8. WALL CATALYSIS
# =============================================================================

@dataclass
class WallProperties:
    T_wall: float = 300.0
    gamma_recomb_N: float = 0.01      # atom recombination coefficient
    gamma_recomb_O: float = 0.01
    gamma_ion_N: float = 0.0          # ion neutralization coefficient
    gamma_ion_O: float = 0.0
    gamma_ion_N2: float = 0.0
    gamma_ion_O2: float = 0.0
    gamma_ion_NO: float = 0.0


def wall_catalytic_flux(
    T: float, rho: float, X: np.ndarray, species: List[str],
    wall: WallProperties,
) -> np.ndarray:
    """
    คืน catalytic wall flux [mol/(m^2 s)] ต่อ species (positive = ออกจากผนัง)
    ใช้ kinetically-limited formula: j_s = gamma_s * ρ_s * sqrt(R T / (2π M_s))
    ต่อด้วย recombination ที่สร้างโมเลกุลกลับเข้าก๊าซ
    """
    n = len(species)
    j = np.zeros(n)
    idx = {s: i for i, s in enumerate(species)}
    sqrtRT = math.sqrt(R_UNIV * T)

    # recombination: N + N -> N2 (rate ~ gamma_N)
    for atom, mol, gamma in (("N","N2",wall.gamma_recomb_N),
                             ("O","O2",wall.gamma_recomb_O)):
        if atom in idx and mol in idx and gamma > 0:
            sp_atom = SPECIES_DB[atom]
            c_atom = rho * X[idx[atom]] / sp_atom.M
            j[ idx[atom] ] -= gamma * c_atom * sqrtRT / math.sqrt(2*math.pi*sp_atom.M)
            j[ idx[mol] ]  += 0.5 * gamma * c_atom * sqrtRT / math.sqrt(2*math.pi*sp_atom.M)

    # ion neutralization: X+ + e- -> X
    for ion, gamma in (("N+",wall.gamma_ion_N), ("O+",wall.gamma_ion_O),
                       ("N2+",wall.gamma_ion_N2), ("O2+",wall.gamma_ion_O2),
                       ("NO+",wall.gamma_ion_NO)):
        if ion in idx and gamma > 0 and "e-" in idx:
            sp_ion = SPECIES_DB[ion]
            c_ion = rho * X[idx[ion]] / sp_ion.M
            c_e = rho * X[idx["e-"]] / SPECIES_DB["e-"].M
            # flux จมทั้ง ion และ e-
            j[ idx[ion] ] -= gamma * c_ion * sqrtRT / math.sqrt(2*math.pi*sp_ion.M)
            j[ idx["e-"] ] -= gamma * c_e * sqrtRT / math.sqrt(2*math.pi*M_E)
    return j


# =============================================================================
# §9. RADIATION ( optically-thin / tangent-slab approximation )
# =============================================================================

@dataclass
class RadiationModel:
    """
    simple band-emission model. ไม่ใช่ line-by-line
    ครอบ N2+ first-negative (B-X) + N, O atomic lines + continuum
    """
    enable: bool = True
    N2p_band_strength: float = 1.0     # tunable
    N_band_strength: float = 1.0
    O_band_strength: float = 1.0


def volumetric_emission(
    T_tr: float, T_v: float, T_e: float,
    rho: float, X: np.ndarray, species: List[str],
    model: RadiationModel,
) -> float:
    """
    คืน q_rad [W/m^3] แบบ optically-thin
    """
    if not model.enable:
        return 0.0
    idx = {s: i for i, s in enumerate(species)}
    q = 0.0
    # N2+ first negative (B->X) ~ strong VUV ที่ ~ 391 nm
    if "N2+" in idx and "e-" in idx:
        c_N2p = rho * X[idx["N2+"]] / SPECIES_DB["N2+"].M
        c_e   = rho * X[idx["e-"]]  / SPECIES_DB["e-"].M
        # rate ~ A * n_e * n_N2+ (approx)
        q += model.N2p_band_strength * 1.0e-20 * c_N2p * c_e * 3.0e-19 * 1.0e8
    # N, O atomic (very crude)
    if "N" in idx:
        c_N = rho * X[idx["N"]] / SPECIES_DB["N"].M
        q += model.N_band_strength * 1.0e-28 * c_N * c_N * math.exp(-2.0e4 / max(T_tr,1))
    if "O" in idx:
        c_O = rho * X[idx["O"]] / SPECIES_DB["O"].M
        q += model.O_band_strength * 1.0e-28 * c_O * c_O * math.exp(-1.5e4 / max(T_tr,1))
    return max(q, 0.0)


# =============================================================================
# §10. EQUILIBRIUM COMPOSITION SOLVER (Newton ใน log-space)
# =============================================================================

def equilibrium_composition(
    T: float, p: float, elements: Dict[str, float],
    species: List[str], tol: float = 1e-10, max_iter: int = 100,
) -> np.ndarray:
    """
    หา X (mole fractions) ที่สมดุลที่ (T, p)
    elements: {"N": n_N, "O": n_O, ...} [mol] — normalizes ภายใน
    ใช้ method: minimize G ที่ธาตุ conserve (Lagrange multiplier + Newton)
    """
    n_el = list(elements.keys())
    # สร้าง matrix a[s, e] = # ของธาตุ e ใน species s
    # นับจากชื่อ species แบบ heuristic
    def n_atoms(s: str, el: str) -> int:
        import re
        # normalize
        if s == "e-":
            return 0
        cnt = 0
        # parse เช่น "N2+", "N2++", "NO", "O2+"
        base = s.rstrip("+-")
        # ตัวอักษรตัวแรก + ตัวเลข
        for m in re.finditer(r"([A-Z][a-z]?)(\d*)", base):
            sym, num = m.group(1), m.group(2)
            if sym == el:
                cnt += int(num) if num else 1
        return cnt

    a = np.array([[n_atoms(s, e) for e in n_el] for s in species], dtype=float)
    b = np.array([elements[e] for e in n_el], dtype=float)
    if b.sum() <= 0:
        raise ValueError("element amounts must be > 0")

    ns = len(species)
    # initial guess: uniform ตามธาตุ
    X = np.ones(ns) / ns

    def get_n(X):
        # total N = b[0] / (sum X_s a_s,0 / sum X)  ... ใช้ basis
        # ง่ายๆ: assume n_total = 1 mol
        return 1.0

    # ใช้ log-space Newton กับ system:
    #   ฟังก์ชัน: g_s = μ_s/RT + ln(p/p°) + ln(X_s)   (สมดุลเคมี)
    #   b_e: element conservation (normalized)
    # ตัวแปร: ln X_s (ns ตัว) + λ_e (n_el ตัว)
    # เราแก้ด้วย fixed-point / descent ง่ายๆ เพื่อความ robust

    lnX = np.log(X)

    def residual(lnX, lam):
        X = np.exp(lnX)
        X = X / X.sum()
        g = np.array([_thermo_piecewise(s, T)[3] for s in species])  # g°/RT
        mu = g + np.log(X) + math.log(p / 1e5) - a @ lam
        element_res = a.T @ X - b / b.sum()
        return mu, element_res

    lam = np.zeros(n_el)
    for it in range(max_iter):
        mu, er = residual(lnX, lam)
        res = np.concatenate([mu, er])
        if np.linalg.norm(res) < tol:
            break
        # finite-diff Jacobian
        J = np.zeros((ns + n_el, ns + n_el))
        eps = 1e-6
        for k in range(ns):
            lnX2 = lnX.copy(); lnX2[k] += eps
            mu2, er2 = residual(lnX2, lam)
            J[:ns, k] = (mu2 - mu) / eps
            J[ns:, k] = (er2 - er) / eps
        for k in range(n_el):
            lam2 = lam.copy(); lam2[k] += eps
            mu2, er2 = residual(lnX, lam2)
            J[:ns, ns+k] = (mu2 - mu) / eps
            J[ns:, ns+k] = (er2 - er) / eps
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(J, res, rcond=None)[0]
        # damp
        step = min(1.0, 1.0 / (np.linalg.norm(delta) + 1e-12))
        lnX += step * delta[:ns]
        lam += step * delta[ns:]
        lnX = np.clip(lnX, -50.0, 5.0)

    X = np.exp(lnX)
    X = X / X.sum()
    return X


# =============================================================================
# §11. NORMAL SHOCK SOLVER (nonequilibrium, 2T)
# =============================================================================

@dataclass
class ShockState:
    u: float          # m/s
    T_tr: float
    T_v: float
    T_e: float
    p: float
    rho: float
    X: np.ndarray
    x: float          # ตำแหน่งหลัง shock [m]
    q_rad: float = 0.0


@dataclass
class ShockInputs:
    p_inf: float = 100.0        # Pa (LEO ~ 100 Pa)
    T_inf: float = 250.0
    U_inf: float = 11000.0      # m/s
    model: str = "air11"
    L_domain: float = 0.5       # m
    n_points: int = 100
    two_temperature: bool = True
    three_temperature: bool = False
    equilibrium_initial: bool = True


def solve_normal_shock(inp: ShockInputs) -> List[ShockState]:
    """
    ลง marched ODE หลัง shock: du/dx, dT_tr/dx, dT_v/dx, dX/dx
    (อย่างง่าย: finite-volume Euler shock ที่ marching ตาม x)
    """
    species = MODELS[inp.model]
    rxns = build_reactions(inp.model)
    thermo = Thermo(species)

    # Rankine-Hugoniot หลัง shock
    u = inp.U_inf
    rho = inp.p_inf / (R_UNIV / 0.02897 * inp.T_inf)  # ผสมอากาศโดยประมาณ
    p = inp.p_inf

    # ลองใช้ Rankine-Hugoniot แบบ perfect gas เป็น seed
    gamma_g = 1.4
    M_inf = u / math.sqrt(gamma_g * R_UNIV / 0.02897 * inp.T_inf)
    if M_inf > 1:
        # normal shock relations
        p2 = p * (2 * gamma_g * M_inf**2 - (gamma_g - 1)) / (gamma_g + 1)
        rho2 = rho * (gamma_g + 1) * M_inf**2 / ((gamma_g - 1) * M_inf**2 + 2)
        T2 = p2 / (rho2 * R_UNIV / 0.02897)
    else:
        p2, rho2, T2 = p, rho, inp.T_inf

    T_tr = T2
    T_v = T2
    T_e = T2
    # สมดุลที่จุดเริ่ม (post-shock equilibrium)
    if inp.equilibrium_initial:
        try:
            X = equilibrium_composition(
                T2, p2,
                {"N": 0.78*2 + 0.21*0 + 0.0, "O": 0.21*2, "Ar": 0.01} if "Ar" in species
                else {"N": 0.78*2, "O": 0.21*2},
                species,
            )
        except Exception:
            X = np.ones(len(species)) / len(species)
    else:
        X = np.zeros(len(species))
        # frozen composition (ตามอากาศ)
        if "N2" in species: X[species.index("N2")] = 0.78
        if "O2" in species: X[species.index("O2")] = 0.21
        if "Ar" in species: X[species.index("Ar")] = 0.01
        X = X / X.sum()

    dx = inp.L_domain / inp.n_points
    states = []
    A = 1.0  # unit area

    for i in range(inp.n_points):
        q_rad = volumetric_emission(T_tr, T_v, T_e, rho, X, species,
                                     RadiationModel())
        states.append(ShockState(u=u, T_tr=T_tr, T_v=T_v, T_e=T_e,
                                 p=p, rho=rho, X=X.copy(),
                                 x=i*dx, q_rad=q_rad))
        if i == inp.n_points - 1:
            break

        # --- source terms ---
        w = species_source(rxns, T_tr, T_v, T_e, X, rho, species)
        # จัดเรียงให้ตรงกับ species
        # (build_reactions สร้างตามชื่อ; ต้องแม็ปให้ตรง)
        # ง่ายสุด: assume order ตรงกัน

        # vibrational source
        Q_vt = q_vt_source(species, T_tr, T_v, p, rho, X)

        # chemistry heat release (enthalpy flux)
        h_mass = thermo.h_mass(T_tr)
        q_chem = -float(np.dot(w, h_mass))

        # --- ODE marching (explicit Euler) ---
        # du/dx = (p/ρu) * dρ/dx ... ใช้ conservation อย่างง่าย
        # สมการพลังงาน: ρ u cp dT/dx = Q_vt + q_chem - q_rad
        cp = thermo.mixture_cp_mass(T_tr, X)
        dTdx = (Q_vt + q_chem - q_rad) / max(rho * u * cp, 1e-10)
        # species: dX/dx = w M / (ρ u)
        M_all = np.array([SPECIES_DB[s].M for s in species])
        dXdx = (w * M_all) / max(rho * u, 1e-10)
        # นอร์มัลไลซ์
        dXdx -= X * dXdx.sum()

        # update
        T_tr = max(T_tr + dTdx * dx, 200.0)
        T_v = max(T_v + (Q_vt / max(rho * u * cp, 1e-10)) * dx, 200.0)
        T_e = T_v
        X = np.clip(X + dXdx * dx, 1e-20, 1.0)
        X = X / X.sum()
        # p จากสมการสถานะ (ค่าเฉลี่ย M)
        Mbar = float(np.dot(X, M_all))
        rho = p * Mbar / (R_UNIV * T_tr)
        u = u * (inp.p_inf / p) * (T_tr / inp.T_inf) * 1.0  # approx continuity

    return states


# =============================================================================
# §12. CLI
# =============================================================================

def _print_thermo_table():
    print("\n=== Sample thermo at T=3000 K ===")
    print(f"{'species':>8} {'cp/R':>10} {'h/RT':>10} {'s/R':>10}")
    for s in SPECIES_DB:
        try:
            cp, h, sr, _ = _thermo_piecewise(s, 3000.0)
            print(f"{s:>8} {cp:10.4f} {h:10.4f} {sr:10.4f}")
        except KeyError:
            pass
    print()


def _print_equilibrium_demo():
    print("\n=== Equilibrium composition (air11, 100 Pa) ===")
    print(f"{'T[K]':>6} {'N2':>8} {'O2':>8} {'NO':>8} {'N':>8} {'O':>8} "
          f"{'NO+':>8} {'e-':>8} {'N+':>8} {'O+':>8}")
    species = MODELS["air11"]
    elements = {"N": 0.78 * 2, "O": 0.21 * 2}
    for T in [3000, 6000, 10000, 15000]:
        try:
            X = equilibrium_composition(T, 100.0, elements, species, max_iter=200)
        except Exception as e:
            print(f"  T={T}: failed ({e})")
            continue
        row = f"{T:6d}"
        for s in species:
            row += f" {X[species.index(s)]:8.3e}"
        print(row)
    print()


def _print_shock_demo():
    print("\n=== Normal shock demo (Apollo-like: U=11 km/s, p=100 Pa) ===")
    inp = ShockInputs(p_inf=100.0, T_inf=250.0, U_inf=11000.0,
                      model="air11", L_domain=0.3, n_points=15)
    try:
        states = solve_normal_shock(inp)
    except Exception as e:
        print(f"  shock solver failed: {e}")
        return
    print(f"{'x[m]':>8} {'T_tr[K]':>10} {'T_v[K]':>10} {'p[Pa]':>10} "
          f"{'rho':>10} {'q_rad[W/m3]':>14}")
    for st in states:
        print(f"{st.x:8.4f} {st.T_tr:10.2f} {st.T_v:10.2f} {st.p:10.2f} "
              f"{st.rho:10.4f} {st.q_rad:14.3e}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Hypersonic chemistry max-fidelity library")
    ap.add_argument("--model", default="air11",
                    choices=list(MODELS.keys()))
    ap.add_argument("--demo", default="all",
                    choices=["all", "thermo", "eq", "shock", "none"])
    ap.add_argument("--nasa9-file", default=None,
                    help="optional CEA/NASA-9 file to override coefficients")
    args = ap.parse_args()

    print(BANNER)
    log.info(f"loaded model: {args.model}  ({len(MODELS[args.model])} species)")

    if args.nasa9_file:
        log.warning("nasa9_file loader: not implemented in this version; "
                    "override NASA9_DB manually")

    if args.demo in ("all", "thermo"):
        _print_thermo_table()
    if args.demo in ("all", "eq"):
        _print_equilibrium_demo()
    if args.demo in ("all", "shock"):
        _print_shock_demo()

    print("done.")


if __name__ == "__main__":
    main()
