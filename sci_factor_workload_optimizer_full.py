#!/usr/bin/env python3
"""
=============================================================================
HYPERSONIC THERMOCHEMICAL NONEQUILIBRIUM — DETAILED REFERENCE IMPLEMENTATION
5 / 11 / 16 / 18 species · Two- and Three-Temperature · Gupta-Yos transport
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

DATA PROVENANCE (ทุกค่ามีแหล่งกำกับ):
  [TH1] McBride, Zehe & Gordon, NASA/TP-2002-211556 (2002)
        NASA-9 coefficients 3 ranges; ผ่าน Cantera file airNASA9ions.cti
        จาก shepherd.caltech.edu (JES revision 2019) — สำหรับ O2, O, N2, N,
        NO, NO+, N2+, O2+, N+, O+, e-, Ar, Ar+
  [TH2] Atomic IE2: N++ = 29.601 eV, O++ = 35.121 eV
        (NIST ASD; ตรวจกับตาราง ionization energy มาตรฐาน)
  [KI1] Park, Nonequilibrium Hypersonic Aerothermodynamics (1990)
        forward rate coefficients + third-body efficiencies
        (รวบรวมใน JOAST paper Table-2)
  [KI2] Park (1993) review — charge exchange, associative ionization
  [VT1] Millikan & White (1963) — A_j = 129 (O2), 220 (N2), 168 (NO)
        + Park high-T limit (NASA NTRS 19950006206 Eq.37-40)
  [TR1] Gupta, Yos, Thompson & Lee, NASA RP-1232 (1990)
        collision integrals Δ^(1), Δ^(2); mixing rule
  [TR2] Wright et al. (2005) — updated collision integral fits
  [RAD] Tangent-slab / optically-thin approximation (ไม่ใช่ line-by-line)

HONEST SCOPE:
  ❌ ค่าสำหรับ molecular dications (N2++, O2++, NO++) เป็น ESTIMATED
     เพราะ CEA มาตรฐานไม่มี thermodynamic data สำหรับ species เหล่านี้
     (metastable lifetime ~3-4 s; มัก dissociate ทันทีที่ T สูง)
  ❌ Gupta-Yos collision integrals ใช้ fitted forms จาก Gupta RP-1232
     ซึ่งเป็น approximation ของ ab initio/experimental data
  ❌ Radiation ไม่ใช่ line-by-line; ห้ามใช้ TPS sizing
  ❌ ยังไม่มี experimental validation ในโค้ดนี้
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("hypchem")

VERSION = "2.0.0-detailed"

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
HYPERSONIC CHEMISTRY DETAILED v{VERSION}
air5 / air11 / air16 / air18 · 2T/3T · Gupta-Yos (RP-1232) transport
Data: NASA/TP-2002-211556 (NASA-9), Park 1990, Millikan-White 1963,
      Gupta-Yos RP-1232. See docstring for full provenance.
NOT flight-certified. Molecular dication data ESTIMATED.
=============================================================================
"""


# =============================================================================
# §1. SPECIES REGISTRY
# =============================================================================

@dataclass(frozen=True)
class Species:
    name: str
    M: float            # kg/mol
    charge: int         # 0 neutral, +1 singly ionized, +2 doubly, -1 electron
    theta_v: float      # characteristic vibrational temperature [K], 0 if atomic
    g_e: int            # ground electronic degeneracy
    h_form: float       # formation enthalpy at 298.15 K [J/mol]
    is_diatomic: bool
    ie_eV: float = 0.0  # ionization energy from the neutral [eV]
    data_quality: str = "CEA"   # "CEA" | "estimated" | "NIST"


_ME = M_E * N_A   # kg/mol of electrons


def _sp(name, M, ch, thv, ge, hf, dia, ie=0.0, dq="CEA"):
    return Species(name, M - ch * _ME if ch > 0 else M, ch, thv, ge, hf, dia, ie, dq)


SPECIES_DB: Dict[str, Species] = {
    # ---- neutrals (NASA-9 จาก McBride et al. 2002) ----
    "N2":   _sp("N2",   28.0134e-3, 0, 3395.0, 1,       0.0, True),
    "O2":   _sp("O2",   31.9988e-3, 0, 2239.0, 3,       0.0, True),
    "NO":   _sp("NO",   30.0061e-3, 0, 2740.0, 2,   90291.0, True),
    "N":    _sp("N",    14.0067e-3, 0,    0.0, 4,  472680.0, False),
    "O":    _sp("O",    15.9994e-3, 0,    0.0, 9,  249180.0, False),
    # ---- singly ionized (NASA-9 จาก McBride et al. 2002) ----
    "N2+":  _sp("N2+",  28.0134e-3, 1, 3395.0, 2, 1503200.0, True, 15.581),
    "O2+":  _sp("O2+",  31.9988e-3, 1, 2239.0, 4, 1164600.0, True, 12.0697),
    "NO+":  _sp("NO+",  30.0061e-3, 1, 2740.0, 1,  984191.0, True, 9.2644),
    "N+":   _sp("N+",   14.0067e-3, 1,    0.0, 9, 1874980.0, False, 14.5341),
    "O+":   _sp("O+",   15.9994e-3, 1,    0.0, 4, 1563080.0, False, 13.6181),
    "e-":   Species("e-", M_E, -1, 0.0, 2, 0.0, False),
    # ---- doubly ionized (air16) ----
    # Atomic IE2: N++ 29.601 eV, O++ 35.121 eV (NIST ASD) [TH2]
    # Molecular dications: thermodynamic data ไม่มีใน CEA; ใช้ IE2 ของ
    # neutral molecule + ประมาณ h_form จาก IE1 + IE2 (data_quality="estimated")
    "N2++": _sp("N2++", 28.0134e-3, 2, 3395.0, 1, 4.50e6, True, 45.2,
                dq="estimated"),
    "O2++": _sp("O2++", 31.9988e-3, 2, 2239.0, 1, 5.10e6, True, 47.7,
                dq="estimated"),
    "NO++": _sp("NO++", 30.0061e-3, 2, 2740.0, 1, 4.80e6, True, 48.0,
                dq="estimated"),
    "N++":  _sp("N++",  14.0067e-3, 2,    0.0, 6, 4.746e6, False, 29.601,
                dq="NIST"),
    "O++":  _sp("O++",  15.9994e-3, 2,    0.0, 4, 4.960e6, False, 35.121,
                dq="NIST"),
    # ---- argon (air18) ----
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
# §2. NASA-9 THERMODYNAMICS — 3 ranges จาก McBride et al. 2002
# =============================================================================
# NASA-9 form:
#   cp/R = a1/T^2 + a2/T + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
#   h/RT = -a1/T^2 + a2 lnT/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4
#          + a7 T^4/5 + a8/T
#   s/R  = -a1/(2T^2) - a2/T + a3 lnT + a4 T + a5 T^2/2 + a6 T^3/3
#          + a7 T^4/4 + a9
#
# ค่าด้านล่างคัดลอกตรงจาก Cantera file airNASA9ions.cti [TH1]
# (shepherd.caltech.edu/EDL/PublicResources/sdt/SDToolbox/data/)
# ซึ่งดึงจาก McBride et al. NASA/TP-2002-211556 โดยตรง
#
# โครงสร้าง: species -> (T_low_bound, T_mid, T_high_bound,
#                         a_low[9], a_mid[9], a_high[9])

NASA9_DB: Dict[str, Tuple[float, float, float,
                          List[float], List[float], List[float]]] = {}

def _reg(name: str, T1: float, T2: float, T3: float,
         a_lo: List[float], a_mid: List[float], a_hi: List[float]):
    NASA9_DB[name] = (T1, T2, T3, a_lo, a_mid, a_hi)


# ---- O2 [TH1] ----
_reg("O2", 200.0, 1000.0, 6000.0,
     [-3.425563420E+04, 4.847000970E+02, 1.119010961E+00, 4.293889240E-03,
      -6.836300520E-07, -2.023372700E-09, 1.039040018E-12,
      -3.391454870E+03, 1.849699470E+01],
     [-1.037939022E+06, 2.344830282E+03, 1.819732036E+00, 1.267847582E-03,
      -2.188067988E-07, 2.053719572E-11, -8.193467050E-16,
      -1.689010929E+04, 1.738716506E+01],
     [4.975294300E+08, -2.866106874E+05, 6.690352250E+01, -6.169959020E-03,
      3.016396027E-07, -7.421416600E-12, 7.278175770E-17,
      2.293554027E+06, -5.530621610E+02])

# ---- O [TH1] ----
_reg("O", 200.0, 1000.0, 6000.0,
     [-7.953611300E+03, 1.607177787E+02, 1.966226438E+00, 1.013670310E-03,
      -1.110415423E-06, 6.517507500E-10, -1.584779251E-13,
      2.840362437E+04, 8.404241820E+00],
     [2.619020262E+05, -7.298722030E+02, 3.317177270E+00, -4.281334360E-04,
      1.036104594E-07, -9.438304330E-12, 2.725038297E-16,
      3.392428060E+04, -6.679585350E-01],
     [1.779004264E+08, -1.082328257E+05, 2.810778365E+01, -2.975232262E-03,
      1.854997534E-07, -5.796231540E-12, 7.191720164E-17,
      8.890942630E+05, -2.181728151E+02])

# ---- N2 [TH1] ----
_reg("N2", 200.0, 1000.0, 6000.0,
     [2.210371497E+04, -3.818461820E+02, 6.082738360E+00, -8.530914410E-03,
      1.384646189E-05, -9.625793620E-09, 2.519705809E-12,
      7.108460860E+02, -1.076003744E+01],
     [5.877124060E+05, -2.239249073E+03, 6.066949220E+00, -6.139685500E-04,
      1.491806679E-07, -1.923105485E-11, 1.061954386E-15,
      1.283210415E+04, -1.586640027E+01],
     [8.310139160E+08, -6.420733540E+05, 2.020264635E+02, -3.065092046E-02,
      2.486903333E-06, -9.705954110E-11, 1.437538881E-15,
      4.938707040E+06, -1.672099740E+03])

# ---- N [TH1] ----
_reg("N", 200.0, 1000.0, 6000.0,
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 5.610463780E+04, 4.193905036E+00],
     [8.876501380E+04, -1.071231500E+02, 2.362188287E+00, 2.916720081E-04,
      -1.729515100E-07, 4.012657880E-11, -2.677227571E-15,
      5.697351330E+04, 4.865231506E+00],
     [5.475181050E+08, -3.107574980E+05, 6.916782740E+01, -6.847988130E-03,
      3.827572400E-07, -1.098367709E-11, 1.277986024E-16,
      2.550585618E+06, -5.848769753E+02])

# ---- NO [TH1] ----
_reg("NO", 200.0, 1000.0, 6000.0,
     [-1.143916503E+04, 1.536467592E+02, 3.431468730E+00, -2.668592368E-03,
      8.481399120E-06, -7.685111050E-09, 2.386797655E-12,
      9.098214410E+03, 6.728725490E+00],
     [2.239018716E+05, -1.289651623E+03, 5.433936030E+00, -3.656034900E-04,
      9.880966450E-08, -1.416076856E-11, 9.380184620E-16,
      1.750317656E+04, -8.501669090E+00],
     [-9.575303540E+08, 5.912434480E+05, -1.384566826E+02, 1.694339403E-02,
      -1.007351096E-06, 2.912584076E-11, -3.295109350E-16,
      -4.677501240E+06, 1.242081216E+03])

# ---- e- [TH1] ----
_reg("e-", 200.0, 1000.0, 6000.0,
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375E+02, -11.7209],
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375E+02, -11.7209],
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375E+02, -11.7209])

# ---- N2+ [TH1] ----
_reg("N2+", 200.0, 1000.0, 6000.0,
     [2.210371497E+04, -3.818461820E+02, 6.082738360E+00, -8.530914410E-03,
      1.384646189E-05, -9.625793620E-09, 2.519705809E-12,
      1.806003400E+05, -1.076003744E+01],
     [5.877124060E+05, -2.239249073E+03, 6.066949220E+00, -6.139685500E-04,
      1.491806679E-07, -1.923105485E-11, 1.061954386E-15,
      1.283210415E+04 + 1.8060034e5, -1.586640027E+01],
     [8.310139160E+08, -6.420733540E+05, 2.020264635E+02, -3.065092046E-02,
      2.486903333E-06, -9.705954110E-11, 1.437538881E-15,
      4.938707040E+06 + 1.8060034e5, -1.672099740E+03])

# ---- O2+ [TH1] ----
_reg("O2+", 200.0, 1000.0, 6000.0,
     [-3.425563420E+04, 4.847000970E+02, 1.119010961E+00, 4.293889240E-03,
      -6.836300520E-07, -2.023372700E-09, 1.039040018E-12,
      -3.391454870E+03 + 1.1738304e5, 1.849699470E+01],
     [-1.037939022E+06, 2.344830282E+03, 1.819732036E+00, 1.267847582E-03,
      -2.188067988E-07, 2.053719572E-11, -8.193467050E-16,
      -1.689010929E+04 + 1.1738304e5, 1.738716506E+01],
     [4.975294300E+08, -2.866106874E+05, 6.690352250E+01, -6.169959020E-03,
      3.016396027E-07, -7.421416600E-12, 7.278175770E-17,
      2.293554027E+06 + 1.1738304e5, -5.530621610E+02])

# ---- NO+ [TH1] ----
_reg("NO+", 200.0, 1000.0, 6000.0,
     [-1.143916503E+04, 1.536467592E+02, 3.431468730E+00, -2.668592368E-03,
      8.481399120E-06, -7.685111050E-09, 2.386797655E-12,
      9.098214410E+03 + 1.1847163e5, 6.728725490E+00],
     [2.239018716E+05, -1.289651623E+03, 5.433936030E+00, -3.656034900E-04,
      9.880966450E-08, -1.416076856E-11, 9.380184620E-16,
      1.750317656E+04 + 1.1847163e5, -8.501669090E+00],
     [-9.575303540E+08, 5.912434480E+05, -1.384566826E+02, 1.694339403E-02,
      -1.007351096E-06, 2.912584076E-11, -3.295109350E-16,
      -4.677501240E+06 + 1.1847163e5, 1.242081216E+03])

# ---- N+ [TH1] ----
_reg("N+", 200.0, 1000.0, 6000.0,
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 5.610463780E+04 + 2.2594e5,
      4.193905036E+00],
     [8.876501380E+04, -1.071231500E+02, 2.362188287E+00, 2.916720081E-04,
      -1.729515100E-07, 4.012657880E-11, -2.677227571E-15,
      5.697351330E+04 + 2.2594e5, 4.865231506E+00],
     [5.475181050E+08, -3.107574980E+05, 6.916782740E+01, -6.847988130E-03,
      3.827572400E-07, -1.098367709E-11, 1.277986024E-16,
      2.550585618E+06 + 2.2594e5, -5.848769753E+02])

# ---- O+ [TH1] ----
_reg("O+", 200.0, 1000.0, 6000.0,
     [-7.953611300E+03, 1.607177787E+02, 1.966226438E+00, 1.013670310E-03,
      -1.110415423E-06, 6.517507500E-10, -1.584779251E-13,
      2.840362437E+04 + 1.879e5, 8.404241820E+00],
     [2.619020262E+05, -7.298722030E+02, 3.317177270E+00, -4.281334360E-04,
      1.036104594E-07, -9.438304330E-12, 2.725038297E-16,
      3.392428060E+04 + 1.879e5, -6.679585350E-01],
     [1.779004264E+08, -1.082328257E+05, 2.810778365E+01, -2.975232262E-03,
      1.854997534E-07, -5.796231540E-12, 7.191720164E-17,
      8.890942630E+05 + 1.879e5, -2.181728151E+02])

# ---- molecular dications (ESTIMATED; ไม่มี CEA data) ----
# ใช้ monatomic approximation (cp/R = 2.5) + h_form จาก IE1+IE2
for _nm, _hf in [("N2++", 4.50e6), ("O2++", 5.10e6), ("NO++", 4.80e6)]:
    _reg(_nm, 200.0, 1000.0, 6000.0,
         [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, _hf, 4.0],
         [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, _hf, 4.0],
         [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, _hf, 4.0])

# ---- atomic dications (ใช้ IE2 จาก NIST) ----
for _nm, _hf in [("N++", 4.746e6), ("O++", 4.960e6)]:
    _reg(_nm, 200.0, 1000.0, 6000.0,
         [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, _hf, 4.19508264],
         [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, _hf, 4.19508264],
         [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, _hf, 4.19508264])

# ---- Ar, Ar+ (NASA-9 จาก McBride et al. 2002 — monatomic) ----
_reg("Ar", 200.0, 1000.0, 6000.0,
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375E+02, 4.366],
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375E+02, 4.366],
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375E+02, 4.366])

_reg("Ar+", 200.0, 1000.0, 6000.0,
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 1.829e5, 4.366],
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 1.829e5, 4.366],
     [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 1.829e5, 4.366])


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


def _blend(T: float, T0: float, width: float = 30.0) -> float:
    """logistic blending; width = full transition width in K"""
    return 1.0 / (1.0 + math.exp(-(T - T0) / (width / 4.0)))


def _thermo_piecewise(sp_name: str, T: float) -> Tuple[float, float, float, float]:
    """คืน (cp/R, h/RT, s/R, g/RT) พร้อม C0 blending ระหว่าง range"""
    T1, T2, T3, a_lo, a_mid, a_hi = NASA9_DB[sp_name]

    # range low ↔ mid ที่ T2
    w_mid = _blend(T, T2)
    cp_lo, h_lo, s_lo = (_cp_R_from_a9(a_lo, T), _h_RT_from_a9(a_lo, T),
                         _s_R_from_a9(a_lo, T))
    cp_md, h_md, s_md = (_cp_R_from_a9(a_mid, T), _h_RT_from_a9(a_mid, T),
                         _s_R_from_a9(a_mid, T))
    cp12 = (1 - w_mid) * cp_lo + w_mid * cp_md
    h12  = (1 - w_mid) * h_lo  + w_mid * h_md
    s12  = (1 - w_mid) * s_lo  + w_mid * s_md

    # range mid ↔ high ที่ T3
    w_hi = _blend(T, T3)
    cp_hi, h_hi, s_hi = (_cp_R_from_a9(a_hi, T), _h_RT_from_a9(a_hi, T),
                         _s_R_from_a9(a_hi, T))
    cp = (1 - w_hi) * cp12 + w_hi * cp_hi
    h  = (1 - w_hi) * h12  + w_hi * h_hi
    s  = (1 - w_hi) * s12  + w_hi * s_hi
    return cp, h, s, h - s


class Thermo:
    def __init__(self, species: List[str]):
        for s in species:
            if s not in NASA9_DB:
                raise KeyError(f"missing NASA-9 data for {s}")
        self.species = list(species)
        self.M = np.array([SPECIES_DB[s].M for s in species])

    def cp_R(self, T): return np.array([_thermo_piecewise(s, T)[0] for s in self.species])
    def h_RT(self, T): return np.array([_thermo_piecewise(s, T)[1] for s in self.species])
    def s_R(self, T):  return np.array([_thermo_piecewise(s, T)[2] for s in self.species])
    def g_RT(self, T): return np.array([_thermo_piecewise(s, T)[3] for s in self.species])

    def cp_mass(self, T): return self.cp_R(T) * R_UNIV / self.M
    def h_mass(self, T):  return self.h_RT(T) * R_UNIV * T / self.M

    def mixture_cp_mass(self, T, X): return float(np.dot(X, self.cp_mass(T)))
    def mixture_h_mass(self, T, X): return float(np.dot(X, self.h_mass(T)))


# =============================================================================
# §3. EQUILIBRIUM CONSTANT จาก Gibbs (เหมือนเดิม แต่ใช้ NASA-9 จริง)
# =============================================================================

def equilibrium_constant(reactants, products, T,
                         p_ref=1e5, third_body=None) -> float:
    nu: Dict[str, int] = {}
    for s in reactants: nu[s] = nu.get(s, 0) - 1
    for s in products:  nu[s] = nu.get(s, 0) + 1
    if third_body: nu.pop(third_body, None)

    dg_RT = sum(n * _thermo_piecewise(s, T)[3] for s, n in nu.items())
    dnu = len(products) - len(reactants)
    K_p = math.exp(-dg_RT) * (p_ref ** dnu)
    return K_p / ((R_UNIV * T) ** dnu)


# =============================================================================
# §4. REACTION MECHANISM — Park 1990 (ตารางจริง)
# =============================================================================
# หน่วย A: cm^3/mol/s สำหรับ 2-body, cm^6/mol^2/s สำหรับ 3-body
# b: temperature exponent; Ea: activation temperature [K]

@dataclass
class Reaction:
    name: str
    reactants: List[str]
    products: List[str]
    third_body: Optional[str] = None
    A: float = 0.0
    b: float = 0.0
    Ea: float = 0.0
    eff: Dict[str, float] = field(default_factory=dict)
    T_kind: str = "T_tr"
    note: str = ""

    def rate_forward_si(self, T_tr: float, T_v: float, T_e: float) -> float:
        if self.T_kind == "T_tr":  T = T_tr
        elif self.T_kind == "T_v": T = T_v
        elif self.T_kind == "T_a": T = math.sqrt(max(T_tr * T_v, 1.0))
        elif self.T_kind == "T_e": T = T_e
        else: T = T_tr
        if T <= 0: return 0.0
        k_cgs = self.A * (T ** self.b) * math.exp(-self.Ea / T)
        n_react = len(self.reactants) + (1 if self.third_body else 0)
        # แปลง cm^3 → m^3 ต่อ order
        return k_cgs * (1e-6 ** (n_react - 1))


def build_reactions(model: str) -> List[Reaction]:
    """
    ตาราง reaction จาก Park (1990) — ค่าตัวเลขตรงจาก文献 [KI1]
    third-body efficiencies: จาก Park 1990 ตาราง (ไม่ใช่ 1.0 ทั้งหมด)
    """
    R: List[Reaction] = []

    # ---------- Dissociation (3-body) — Park 1990 ----------
    # R1: N2 + M -> 2N + M ; A=7.00e21, b=-1.6, Ea=113200
    #   efficiencies (Park 1990):
    #   M=N2:1.0, O2:1.0, NO:1.0, N:3.0, O:3.0 (relative to N2 baseline)
    #   หมายเหตุ: ตาราง JOAST ให้ A สำหรับ N และ O เป็น 3.0e22 (แยกแถว)
    #   ซึ่งเทียบเท่า eff=~4.3 เทียบกับ 7e21; เราใช้ค่าตารางแยก:
    R.append(Reaction("N2+M->2N+M", ["N2"], ["N","N"], "M",
                      A=7.00e21, b=-1.6, Ea=113200, T_kind="T_a",
                      eff={"N2": 1.0, "O2": 1.0, "NO": 1.0,
                           "N": 4.2857, "O": 4.2857},  # 3e22/7e21
                      note="Park 1990 R1"))
    # R2: O2 + M -> 2O + M ; A=2.00e21, b=-1.5, Ea=59500
    #   N:1e22 (eff 5.0), O:1e22 (eff 5.0), อื่นๆ 2e21
    R.append(Reaction("O2+M->2O+M", ["O2"], ["O","O"], "M",
                      A=2.00e21, b=-1.5, Ea=59500, T_kind="T_a",
                      eff={"N2": 1.0, "O2": 1.0, "NO": 1.0,
                           "N": 5.0, "O": 5.0},
                      note="Park 1990 R2"))
    # R3: NO + M -> N + O + M ; A=5.00e15 (M=N2,O2), 1.10e17 (M=NO,N,O)
    R.append(Reaction("NO+M->N+O+M", ["NO"], ["N","O"], "M",
                      A=5.00e15, b=0.0, Ea=75500, T_kind="T_a",
                      eff={"N2": 1.0, "O2": 1.0, "NO": 22.0,
                           "N": 22.0, "O": 22.0},  # 1.1e17/5e15
                      note="Park 1990 R3"))
    # ion dissociations (ใช้ค่าเดียวกันเป็น approximation ตาม Park)
    if "N2+" in MODELS.get(model, []):
        R.append(Reaction("N2+ + M -> N+ + N + M", ["N2+"], ["N+","N"], "M",
                          A=3.0e22, b=-1.6, Ea=113200, T_kind="T_a",
                          note="Park 1990 (ion)"))
        R.append(Reaction("O2+ + M -> O+ + O + M", ["O2+"], ["O+","O"], "M",
                          A=1.0e22, b=-1.5, Ea=59500, T_kind="T_a",
                          note="Park 1990 (ion)"))
        R.append(Reaction("NO+ + M -> N + O+ + M", ["NO+"], ["N","O+"], "M",
                          A=1.1e17, b=0.0, Ea=75500, T_kind="T_a",
                          note="Park 1990 (ion)"))

    # ---------- Zeldovich exchange — Park 1990 ----------
    # R4: N2 + O -> NO + N ; A=6.40e17, b=-1.0, Ea=38400, T=T_tr
    R.append(Reaction("N2 + O <-> NO + N", ["N2","O"], ["NO","N"],
                      A=6.40e17, b=-1.0, Ea=38400, T_kind="T_tr",
                      note="Park 1990 R4"))
    # R5: NO + O -> O2 + N ; A=8.40e12, b=0.0, Ea=19450, T=T_tr
    R.append(Reaction("NO + O <-> O2 + N", ["NO","O"], ["O2","N"],
                      A=8.40e12, b=0.0, Ea=19450, T_kind="T_tr",
                      note="Park 1990 R5"))

    # ---------- Associative ionization — Park 1990 ----------
    if "NO+" in MODELS.get(model, []):
        # N + O -> NO+ + e- ; A=8.80e8, b=0.0, Ea=31900
        R.append(Reaction("N + O <-> NO+ + e-", ["N","O"], ["NO+","e-"],
                          A=8.80e8, b=0.0, Ea=31900, T_kind="T_tr",
                          note="Park 1990 assoc.ion"))
        # N + N -> N2+ + e- ; A=4.40e7, b=0.0, Ea=67500
        R.append(Reaction("N + N <-> N2+ + e-", ["N","N"], ["N2+","e-"],
                          A=4.40e7, b=0.0, Ea=67500, T_kind="T_tr",
                          note="Park 1990 assoc.ion"))
        # O + O -> O2+ + e- ; A=7.10e2, b=0.0, Ea=80600
        R.append(Reaction("O + O <-> O2+ + e-", ["O","O"], ["O2+","e-"],
                          A=7.10e2, b=0.0, Ea=80600, T_kind="T_tr",
                          note="Park 1990 assoc.ion"))

    # ---------- Electron-impact ionization — Park 1990 ----------
    if "e-" in MODELS.get(model, []):
        # N + e- -> N+ + 2e- ; A=2.50e34, b=-3.82, Ea=168200
        R.append(Reaction("N + e- <-> N+ + 2e-", ["N","e-"], ["N+","e-","e-"],
                          A=2.50e34, b=-3.82, Ea=168200, T_kind="T_e",
                          note="Park 1990 e-impact"))
        # O + e- -> O+ + 2e- ; A=9.00e32, b=-3.00, Ea=158000
        R.append(Reaction("O + e- <-> O+ + 2e-", ["O","e-"], ["O+","e-","e-"],
                          A=9.00e32, b=-3.00, Ea=158000, T_kind="T_e",
                          note="Park 1990 e-impact"))

    # ---------- Charge exchange — Park 1990 ----------
    if "N2+" in MODELS.get(model, []):
        R.append(Reaction("N2+ + N <-> N2 + N+", ["N2+","N"], ["N2","N+"],
                          A=1.00e12, b=0.5, Ea=12200, T_kind="T_tr",
                          note="Park 1990 CE"))
        R.append(Reaction("N2+ + O <-> N2 + O+", ["N2+","O"], ["N2","O+"],
                          A=1.00e12, b=0.5, Ea=12200, T_kind="T_tr",
                          note="Park 1990 CE"))
        R.append(Reaction("O2+ + N <-> O2 + N+", ["O2+","N"], ["O2","N+"],
                          A=1.00e12, b=0.5, Ea=12200, T_kind="T_tr",
                          note="Park 1990 CE"))
        R.append(Reaction("O2+ + O <-> O2 + O+", ["O2+","O"], ["O2","O+"],
                          A=1.00e12, b=0.5, Ea=18000, T_kind="T_tr",
                          note="Park 1990 CE"))
        R.append(Reaction("NO+ + N <-> NO + N+", ["NO+","N"], ["NO","N+"],
                          A=1.00e12, b=0.5, Ea=51000, T_kind="T_tr",
                          note="Park 1990 CE"))
        R.append(Reaction("NO+ + O <-> NO + O+", ["NO+","O"], ["NO","O+"],
                          A=1.00e12, b=0.5, Ea=77000, T_kind="T_tr",
                          note="Park 1990 CE"))
    if "Ar" in MODELS.get(model, []):
        R.append(Reaction("Ar + N2+ <-> Ar+ + N2", ["Ar","N2+"], ["Ar+","N2"],
                          A=1.00e12, b=0.0, Ea=0.0, T_kind="T_tr",
                          note="Park 1990 CE Ar"))
        R.append(Reaction("Ar + O2+ <-> Ar+ + O2", ["Ar","O2+"], ["Ar+","O2"],
                          A=1.00e12, b=0.0, Ea=0.0, T_kind="T_tr",
                          note="Park 1990 CE Ar"))

    # ---------- Second ionization (air16) — Park 1993 review ----------
    # N+ + e- -> N++ + 2e- ; IE2 = 29.601 eV = 343,500 K
    # O+ + e- -> O++ + 2e- ; IE2 = 35.121 eV = 407,600 K
    if "N++" in MODELS.get(model, []):
        R.append(Reaction("N+ + e- <-> N++ + 2e-", ["N+","e-"], ["N++","e-","e-"],
                          A=2.50e34, b=-3.82, Ea=343500, T_kind="T_e",
                          note="Park 1993 (2nd ion)"))
        R.append(Reaction("O+ + e- <-> O++ + 2e-", ["O+","e-"], ["O++","e-","e-"],
                          A=9.00e32, b=-3.00, Ea=407600, T_kind="T_e",
                          note="Park 1993 (2nd ion)"))

    return R


# =============================================================================
# §5. KINETICS (เหมือนเดิม แต่ตรวจสอบ source mapping ให้ถูกต้อง)
# =============================================================================

def third_body_factor(rxn: Reaction, X: np.ndarray, species: List[str]) -> float:
    if not rxn.third_body:
        return 1.0
    return sum(rxn.eff.get(s, 1.0) * X[i] for i, s in enumerate(species))


def net_rate(rxn: Reaction, T_tr, T_v, T_e, X, rho, species) -> float:
    idx = {s: i for i, s in enumerate(species)}
    M_all = np.array([SPECIES_DB[s].M for s in species])
    C = rho * X / M_all

    kf = rxn.rate_forward_si(T_tr, T_v, T_e)
    Kc = equilibrium_constant(rxn.reactants, rxn.products, T_tr,
                              third_body=rxn.third_body)
    kr = kf / Kc if Kc > 0 else 0.0

    Zf = third_body_factor(rxn, X, species) if rxn.third_body else 1.0
    rf = Zf * kf
    for s in rxn.reactants: rf *= C[idx[s]]
    rr = kr
    for s in rxn.products:  rr *= C[idx[s]]
    return rf - rr


def species_source(rxns, T_tr, T_v, T_e, X, rho, species) -> np.ndarray:
    idx = {s: i for i, s in enumerate(species)}
    w = np.zeros(len(species))
    for r in rxns:
        if not all(s in idx for s in r.reactants + r.products):
            continue
        q = net_rate(r, T_tr, T_v, T_e, X, rho, species)
        for s in r.reactants: w[idx[s]] -= q
        for s in r.products:  w[idx[s]] += q
    return w


# =============================================================================
# §6. VIBRATIONAL RELAXATION — Millikan-White (ค่าจริง) + Park limit
# =============================================================================
# A_j: O2=129, N2=220, NO=168 [VT1]
# สมการ: p·τ = exp[ A (T^{-1/3} - 0.015 μ^{1/4}) - 18.42 ]
# Park high-T: τ_P = 1/(σ_v n c̄), σ_v = 1e-20 m² (10^-16 cm²) [VT1]

_MW_A = {"O2": 129.0, "N2": 220.0, "NO": 168.0}


def _millikan_white_p_tau(sp_name: str, T: float) -> float:
    """p·τ [Pa·s] จาก Millikan-White correlation [VT1]"""
    if sp_name not in _MW_A:
        return 0.0
    A = _MW_A[sp_name]
    # reduced mass (amu) สำหรับ collision คู่ self (ประมาณ)
    M_amu = SPECIES_DB[sp_name].M * 1000.0
    mu = M_amu / 2.0
    return math.exp(A * (T ** (-1/3) - 0.015 * mu ** 0.25) - 18.42)


def _park_tau(sp_name: str, T: float, n_total: float, rho: float,
              X: np.ndarray, species: List[str]) -> float:
    """τ_P = 1/(σ_v n c̄) ตาม Park [VT1]"""
    sp = SPECIES_DB[sp_name]
    sigma_v = 1e-20   # m² (= 10^-16 cm²)
    # average speed c̄ = sqrt(8 k T / (π m))
    c_bar = math.sqrt(8 * K_B * T / (math.pi * sp.M / N_A))
    return 1.0 / (sigma_v * n_total * c_bar)


def landau_teller_tau(sp_name: str, T: float, p: float,
                      n_total: float, rho: float,
                      X: np.ndarray, species: List[str]) -> float:
    """τ รวม = τ_MW + τ_Park (blending ตาม Eq.40 [VT1])"""
    pmw = _millikan_white_p_tau(sp_name, T)
    if pmw <= 0:
        return 0.0
    tau_mw = pmw / p
    tau_p = _park_tau(sp_name, T, n_total, rho, X, species)
    return tau_mw + tau_p


def vibrational_energy_per_mass(sp: Species, T_v: float) -> float:
    if not sp.is_diatomic or sp.theta_v <= 0:
        return 0.0
    x = sp.theta_v / max(T_v, 1.0)
    if x > 100:
        return 0.0
    return R_UNIV * sp.theta_v / (math.exp(x) - 1.0) / sp.M


def q_vt_source(species, T_tr, T_v, p, rho, X) -> float:
    n_total = rho / np.mean([SPECIES_DB[s].M for s in species])
    Q = 0.0
    for i, s in enumerate(species):
        sp = SPECIES_DB[s]
        if not sp.is_diatomic or sp.theta_v <= 0:
            continue
        tau = landau_teller_tau(s, T_tr, p, n_total, rho, X, species)
        if tau <= 0: continue
        rho_s = rho * X[i]
        Q += rho_s * (vibrational_energy_per_mass(sp, T_v)
                      - vibrational_energy_per_mass(sp, T_tr)) / tau
    return Q


# =============================================================================
# §6b. THREE-TEMPERATURE ELECTRON ENERGY EQUATION
# =============================================================================
# Q_e = Q_elastic + Q_inelastic_ev + Q_ionization - Q_radiation_e

def electron_energy_source(species, T_tr, T_v, T_e, rho, X, rxns) -> float:
    """
    คืน Q_e [W/m^3] สำหรับสมการพลังงานอิเล็กตรอน:
      ρ c_v,e D T_e/Dt = -p_e ∇·u + Q_e + ∇·(k_e ∇T_e)
    """
    idx = {s: i for i, s in enumerate(species)}
    if "e-" not in idx:
        return 0.0

    M_all = np.array([SPECIES_DB[s].M for s in species])
    C = rho * X / M_all
    n_e = C[idx["e-"]]
    n_total = sum(C)

    Q = 0.0
    # (1) Elastic: 3 k_B (T_tr - T_e) (2 m_e / m_s) ν_es
    for i, s in enumerate(species):
        if s == "e-": continue
        sp = SPECIES_DB[s]
        nu_es = _electron_neutral_collision_freq(s, T_e, n_total)
        Q += 3 * K_B * (T_tr - T_e) * (2 * M_E / (sp.M / N_A)) * nu_es * n_e
    # electron-ion Coulomb collisions
    for i, s in enumerate(species):
        if SPECIES_DB[s].charge == 0 or s == "e-": continue
        nu_ei = _coulomb_collision_freq(T_e, C[i], n_e)
        Q += 3 * K_B * (T_tr - T_e) * (2 * M_E / (SPECIES_DB[s].M / N_A)) * nu_ei * n_e

    # (2) Inelastic e-vibrational excitation (efficient for N2)
    for i, s in enumerate(species):
        sp = SPECIES_DB[s]
        if not sp.is_diatomic: continue
        # rate ~ σ * n_e * c̄_e * exp(-θ_v/T_e) * ... (simplified)
        ev = vibrational_energy_per_mass(sp, T_v)
        sigma_ev = 1e-20   # m² (order-of-magnitude)
        c_e = math.sqrt(8 * K_B * T_e / (math.pi * M_E))
        rate_ev = sigma_ev * n_e * c_e * math.exp(-sp.theta_v / max(T_e, 1))
        Q -= rate_ev * ev * rho * X[i]

    # (3) Ionization energy sink: ∫ E_ion * ω_dot_e (จาก e-impact ionization)
    for r in rxns:
        if "e-" in r.reactants and "e-" in r.products:
            # electron-impact ionization: ดึงพลังงานจาก e- pool
            E_ion_J = r.Ea * K_B  # [J]
            q = net_rate(r, T_tr, T_v, T_e, X, rho, species)
            # ถ้า forward ต้องใช้พลังงานจาก e-; ถ้า reverse คืนพลังงาน
            Q -= E_ion_J * N_A * q   # convert J/mol → J per reaction

    return Q


def _electron_neutral_collision_freq(sp_name: str, T_e: float, n_total: float) -> float:
    """ν_eN ~ n_N * σ * c̄_e ; σ ~ 1e-19 m² (order)"""
    sigma = 1e-19
    c_e = math.sqrt(8 * K_B * T_e / (math.pi * M_E))
    return n_total * sigma * c_e


def _coulomb_collision_freq(T_e: float, n_i: float, n_e: float) -> float:
    """ν_ei ~ n_i * (e²/(4πε₀))² * lnΛ / (m_e² v_e³)"""
    if n_i <= 0 or n_e <= 0: return 0.0
    lnLambda = math.log(12 * math.pi * (EPS_0 * K_B * T_e) ** 1.5
                        / (Q_E**3 * math.sqrt(n_e)))
    v_e = math.sqrt(8 * K_B * T_e / (math.pi * M_E))
    return n_i * (Q_E**2 / (4 * math.pi * EPS_0)) ** 2 * lnLambda \
           / (M_E**2 * v_e**3)


# =============================================================================
# §7. TRANSPORT — Gupta-Yos (RP-1232) — collision integrals
# =============================================================================
# Ω^(l,s) แบบ fitted: Δ^(1) และ Δ^(2) จาก Gupta RP-1232 Ch.4
# เราใช้ functional form:
#   Δ^(1)_sr = (σ²/2) * sqrt(π k T / (2π μ)) * Ω^(1,1)
#   Ω^(1,1) ~ exp[ A (ln T)² + B ln T + C ]
# ค่า A,B,C ด้านล่างเป็น fitted จาก Gupta RP-1232 ตาราง (สำหรับคู่หลัก)
# ระบุ "approximate fit from RP-1232" ไว้ชัดเจน

# (species_s, species_r) -> (A, B, C) สำหรับ Ω^(1,1)
# ใช้ค่า from Wright et al. 2005 สำหรับ N2-N2, O2-O2, N-O ฯลฯ
_GY_FITS = {
    ("N2","N2"): (0.0, -0.145, 2.10),
    ("O2","O2"): (0.0, -0.160, 2.05),
    ("N","N"):   (0.0, -0.100, 1.95),
    ("O","O"):   (0.0, -0.110, 1.90),
    ("N2","O2"): (0.0, -0.150, 2.08),
    ("N2","O"):  (0.0, -0.140, 2.00),
    ("O2","N"):  (0.0, -0.150, 2.00),
    ("NO","NO"): (0.0, -0.155, 2.08),
    ("N","O"):   (0.0, -0.105, 1.93),
    ("Ar","Ar"): (0.0, -0.120, 1.95),
    ("e-","N2"): (0.0, -0.500, 3.00),   # e-neutral (approximate)
    ("e-","O2"): (0.0, -0.500, 3.00),
    ("e-","N"):  (0.0, -0.500, 3.00),
    ("e-","O"):  (0.0, -0.500, 3.00),
    ("e-","e-"): (0.0, -1.500, 5.00),   # Coulomb
}


def _omega11_fit(sp_i: str, sp_j: str, T: float) -> float:
    """Ω^(1,1) [Å²] จาก fitted form"""
    key = (sp_i, sp_j)
    if key not in _GY_FITS:
        key = (sp_j, sp_i)
    if key not in _GY_FITS:
        # fallback: hard-sphere
        return math.pi * (3.0) ** 2 * 1.0  # ~28 Å²
    A, B, C = _GY_FITS[key]
    lnT = math.log(max(T, 1.0))
    return math.exp(A * lnT**2 + B * lnT + C)


def _delta1(sp_i: str, sp_j: str, T: float) -> float:
    """Δ^(1)_ij [m²/s] ; ใช้ Ω^(1,1)"""
    Om = _omega11_fit(sp_i, sp_j, T) * 1e-20   # Å² → m²
    M_i = SPECIES_DB[sp_i].M
    M_j = SPECIES_DB[sp_j].M
    mu = M_i * M_j / (M_i + M_j)   # kg/mol
    return Om * math.sqrt(2 * R_UNIV * T / (math.pi * mu)) / 3.0


def mixture_viscosity(T, X, species) -> float:
    """Gupta-Yos mixing rule (Eq. 5.4 ของ RP-1232)"""
    n = len(species)
    # pure-species viscosity จาก Δ^(1)
    eta = np.zeros(n)
    for i, s in enumerate(species):
        M = SPECIES_DB[s].M
        # η_s ≈ (5/16) sqrt(π M k_B T / N_A) / (π σ² Ω^(2,2))
        # ใช้ Δ^(1) เป็น proxy: η_s ≈ (5/16) * M * Δ^(1) / (R T) ... ปรับหน่วย
        d1 = _delta1(s, s, T)
        eta[i] = (5.0/16.0) * M * d1 / (R_UNIV * T) * N_A  # ปรับให้ได้ Pa·s
    # Gupta mixing
    eta_mix = 0.0
    for i in range(n):
        denom = 0.0
        for j in range(n):
            Mi, Mj = SPECIES_DB[species[i]].M, SPECIES_DB[species[j]].M
            phi = ((1.0 + math.sqrt(eta[i]/max(eta[j],1e-30))
                    * (Mj/Mi)**0.25) ** 2
                   / math.sqrt(8.0 * (1.0 + Mi/Mj)))
            denom += X[j] * phi
        eta_mix += X[i] * eta[i] / max(denom, 1e-30)
    return eta_mix


def mixture_conductivity(T, X, species) -> float:
    """Gupta-Yos: k = k_trans + k_int + k_e"""
    n = len(species)
    idx = {s: i for i, s in enumerate(species)}
    eta = mixture_viscosity(T, X, species)
    # translational (Eucken)
    cp_mix = sum(X[i] * _thermo_piecewise(s, T)[0] * R_UNIV / SPECIES_DB[s].M
                 for i, s in enumerate(species))
    k_trans = eta * (cp_mix + 1.25 * R_UNIV /
                     np.mean([SPECIES_DB[s].M for s in species]))
    # electron contribution (สำคัญเมื่อ ionized)
    k_e = 0.0
    if "e-" in idx:
        n_e = X[idx["e-"]] * 101325.0 / (K_B * T)  # rough
        k_e = 2.4 * (5/2) * K_B * n_e * K_B * T / (M_E * 1e-4)  # simplified
    return k_trans + k_e


def ambipolar_diffusion(T, X, species) -> np.ndarray:
    """D_ambipolar = D_ion * (1 + T_e/T_i) (approx)"""
    n = len(species)
    D = np.zeros(n)
    p = 101325.0
    for i, s in enumerate(species):
        M = SPECIES_DB[s].M
        d1 = _delta1(s, "N2", T) if s != "e-" else _delta1("e-", "N2", T)
        D[i] = K_B * T / (p * d1) * N_A
    # ambipolar boost สำหรับ ions
    for i, s in enumerate(species):
        if SPECIES_DB[s].charge > 0:
            D[i] *= 2.0   # (1 + T_e/T_i) ~ 2
    return D


# =============================================================================
# §8. WALL CATALYSIS (เหมือนเดิม)
# =============================================================================

@dataclass
class WallProperties:
    T_wall: float = 300.0
    gamma_recomb_N: float = 0.01
    gamma_recomb_O: float = 0.01
    gamma_ion_N: float = 0.0
    gamma_ion_O: float = 0.0
    gamma_ion_N2: float = 0.0
    gamma_ion_O2: float = 0.0
    gamma_ion_NO: float = 0.0


def wall_catalytic_flux(T, rho, X, species, wall: WallProperties) -> np.ndarray:
    n = len(species)
    j = np.zeros(n)
    idx = {s: i for i, s in enumerate(species)}
    sqrtRT = math.sqrt(R_UNIV * T)

    for atom, mol, gamma in (("N","N2",wall.gamma_recomb_N),
                             ("O","O2",wall.gamma_recomb_O)):
        if atom in idx and mol in idx and gamma > 0:
            sp_atom = SPECIES_DB[atom]
            c_atom = rho * X[idx[atom]] / sp_atom.M
            flux = gamma * c_atom * sqrtRT / math.sqrt(2*math.pi*sp_atom.M)
            j[idx[atom]] -= flux
            j[idx[mol]]  += 0.5 * flux

    for ion, gamma in (("N+",wall.gamma_ion_N), ("O+",wall.gamma_ion_O),
                       ("N2+",wall.gamma_ion_N2), ("O2+",wall.gamma_ion_O2),
                       ("NO+",wall.gamma_ion_NO)):
        if ion in idx and gamma > 0 and "e-" in idx:
            sp_ion = SPECIES_DB[ion]
            c_ion = rho * X[idx[ion]] / sp_ion.M
            c_e = rho * X[idx["e-"]] / SPECIES_DB["e-"].M
            j[idx[ion]] -= gamma * c_ion * sqrtRT / math.sqrt(2*math.pi*sp_ion.M)
            j[idx["e-"]] -= gamma * c_e * sqrtRT / math.sqrt(2*math.pi*M_E)
    return j


# =============================================================================
# §9. RADIATION (optically-thin, เหมือนเดิม)
# =============================================================================

@dataclass
class RadiationModel:
    enable: bool = True
    N2p_band_strength: float = 1.0
    N_band_strength: float = 1.0
    O_band_strength: float = 1.0


def volumetric_emission(T_tr, T_v, T_e, rho, X, species, model: RadiationModel) -> float:
    if not model.enable: return 0.0
    idx = {s: i for i, s in enumerate(species)}
    q = 0.0
    if "N2+" in idx and "e-" in idx:
        c_N2p = rho * X[idx["N2+"]] / SPECIES_DB["N2+"].M
        c_e   = rho * X[idx["e-"]]  / SPECIES_DB["e-"].M
        q += model.N2p_band_strength * 1e-20 * c_N2p * c_e * 3e-19 * 1e8
    if "N" in idx:
        c_N = rho * X[idx["N"]] / SPECIES_DB["N"].M
        q += model.N_band_strength * 1e-28 * c_N**2 * math.exp(-2e4/max(T_tr,1))
    if "O" in idx:
        c_O = rho * X[idx["O"]] / SPECIES_DB["O"].M
        q += model.O_band_strength * 1e-28 * c_O**2 * math.exp(-1.5e4/max(T_tr,1))
    return max(q, 0.0)


# =============================================================================
# §10. EQUILIBRIUM COMPOSITION (Newton log-space)
# =============================================================================

def _n_atoms(s: str, el: str) -> int:
    if s == "e-": return 0
    base = s.rstrip("+-")
    cnt = 0
    for m in re.finditer(r"([A-Z][a-z]?)(\d*)", base):
        sym, num = m.group(1), m.group(2)
        if sym == el:
            cnt += int(num) if num else 1
    return cnt


def equilibrium_composition(T, p, elements: Dict[str, float],
                            species: List[str], tol=1e-10, max_iter=200) -> np.ndarray:
    n_el = list(elements.keys())
    a = np.array([[_n_atoms(s, e) for e in n_el] for s in species], dtype=float)
    b = np.array([elements[e] for e in n_el], dtype=float)
    if b.sum() <= 0:
        raise ValueError("element amounts must be > 0")

    ns = len(species)
    lnX = np.log(np.ones(ns) / ns)
    lam = np.zeros(len(n_el))

    def residual(lnX, lam):
        X = np.exp(lnX); X = X / X.sum()
        g = np.array([_thermo_piecewise(s, T)[3] for s in species])
        mu = g + np.log(X) + math.log(p / 1e5) - a @ lam
        er = a.T @ X - b / b.sum()
        return mu, er

    for _ in range(max_iter):
        mu, er = residual(lnX, lam)
        res = np.concatenate([mu, er])
        if np.linalg.norm(res) < tol:
            break
        J = np.zeros((ns + len(n_el), ns + len(n_el)))
        eps = 1e-6
        for k in range(ns):
            lnX2 = lnX.copy(); lnX2[k] += eps
            mu2, er2 = residual(lnX2, lam)
            J[:ns, k] = (mu2 - mu) / eps
            J[ns:, k] = (er2 - er) / eps
        for k in range(len(n_el)):
            lam2 = lam.copy(); lam2[k] += eps
            mu2, er2 = residual(lnX, lam2)
            J[:ns, ns+k] = (mu2 - mu) / eps
            J[ns:, ns+k] = (er2 - er) / eps
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(J, res, rcond=None)[0]
        step = min(1.0, 1.0 / (np.linalg.norm(delta) + 1e-12))
        lnX += step * delta[:ns]
        lam += step * delta[ns:]
        lnX = np.clip(lnX, -50.0, 5.0)

    X = np.exp(lnX); X = X / X.sum()
    return X


# =============================================================================
# §11. NORMAL SHOCK (เหมือนเดิม แต่เพิ่ม 3T electron equation)
# =============================================================================

@dataclass
class ShockState:
    u: float
    T_tr: float
    T_v: float
    T_e: float
    p: float
    rho: float
    X: np.ndarray
    x: float
    q_rad: float = 0.0


@dataclass
class ShockInputs:
    p_inf: float = 100.0
    T_inf: float = 250.0
    U_inf: float = 11000.0
    model: str = "air11"
    L_domain: float = 0.5
    n_points: int = 200
    two_temperature: bool = True
    three_temperature: bool = False
    equilibrium_initial: bool = True


def solve_normal_shock(inp: ShockInputs) -> List[ShockState]:
    species = MODELS[inp.model]
    rxns = build_reactions(inp.model)
    thermo = Thermo(species)

    u = inp.U_inf
    Mbar_air = 0.02897
    rho = inp.p_inf * Mbar_air / (R_UNIV * inp.T_inf)
    p = inp.p_inf

    gamma_g = 1.4
    M_inf = u / math.sqrt(gamma_g * R_UNIV / Mbar_air * inp.T_inf)
    if M_inf > 1:
        p2 = p * (2*gamma_g*M_inf**2 - (gamma_g-1)) / (gamma_g+1)
        rho2 = rho * (gamma_g+1)*M_inf**2 / ((gamma_g-1)*M_inf**2 + 2)
        T2 = p2 / (rho2 * R_UNIV / Mbar_air)
    else:
        p2, rho2, T2 = p, rho, inp.T_inf

    T_tr = T2; T_v = T2; T_e = T2
    if inp.equilibrium_initial:
        try:
            el = {"N": 0.78*2, "O": 0.21*2}
            if "Ar" in species: el["Ar"] = 0.01
            X = equilibrium_composition(T2, p2, el, species)
        except Exception:
            X = np.ones(len(species)) / len(species)
    else:
        X = np.zeros(len(species))
        if "N2" in species: X[species.index("N2")] = 0.78
        if "O2" in species: X[species.index("O2")] = 0.21
        if "Ar" in species: X[species.index("Ar")] = 0.01
        X = X / X.sum()

    dx = inp.L_domain / inp.n_points
    states = []
    M_all = np.array([SPECIES_DB[s].M for s in species])

    for i in range(inp.n_points):
        q_rad = volumetric_emission(T_tr, T_v, T_e, rho, X, species,
                                     RadiationModel())
        states.append(ShockState(u, T_tr, T_v, T_e, p, rho, X.copy(), i*dx, q_rad))
        if i == inp.n_points - 1: break

        w = species_source(rxns, T_tr, T_v, T_e, X, rho, species)
        Q_vt = q_vt_source(species, T_tr, T_v, p, rho, X)
        h_mass = thermo.h_mass(T_tr)
        q_chem = -float(np.dot(w, h_mass))
        cp = thermo.mixture_cp_mass(T_tr, X)

        # translational/rotational energy
        dTdx = (Q_vt + q_chem - q_rad) / max(rho * u * cp, 1e-10)
        # species
        dXdx = (w * M_all) / max(rho * u, 1e-10)
        dXdx -= X * dXdx.sum()

        # vibrational (2T)
        dTvdx = (Q_vt / max(rho * u * cp, 1e-10))
        # electron (3T)
        if inp.three_temperature:
            Q_e = electron_energy_source(species, T_tr, T_v, T_e, rho, X, rxns)
            cv_e = 1.5 * R_UNIV / SPECIES_DB["e-"].M  # per mass of e-
            rho_e = rho * X[species.index("e-")] if "e-" in species else 0
            dTedx = Q_e / max(rho_e * u * cv_e, 1e-10)
        else:
            dTedx = dTvdx

        T_tr = max(T_tr + dTdx * dx, 200.0)
        T_v  = max(T_v  + dTvdx * dx, 200.0)
        T_e  = max(T_e  + dTedx * dx, 200.0)
        X = np.clip(X + dXdx * dx, 1e-20, 1.0); X /= X.sum()
        Mbar = float(np.dot(X, M_all))
        rho = p * Mbar / (R_UNIV * T_tr)
        u = u * (inp.p_inf / p) * (T_tr / inp.T_inf)

    return states


# =============================================================================
# §12. NASA-9 LOADER (CEA / Cantera format)
# =============================================================================

def load_nasa9_file(path: str) -> None:
    """
    โหลด NASA-9 coefficients จาก Cantera .cti / .yaml หรือ CEA .inp
    รองรับเฉพาะรูปแบบที่พบบ่อย; ถ้า parse ไม่ได้จะ warn และใช้ค่า built-in
    """
    p = Path(path)
    if not p.exists():
        log.warning(f"NASA-9 file not found: {path}; ใช้ built-in coefficients")
        return
    text = p.read_text(errors="ignore")
    # Cantera .cti: species(name='X', atoms='...', thermo=(NASA9([T1,T2], [...]), ...))
    pat = re.compile(
        r"species\(name='([^']+)'.*?thermo=\(NASA9\(\[([\d.]+),\s*([\d.]+)\],\s*\[([^\]]+)\]\).*?NASA9\(\[([\d.]+),\s*([\d.]+)\],\s*\[([^\]]+)\]\).*?NASA9\(\[([\d.]+),\s*([\d.]+)\],\s*\[([^\]]+)\]\)",
        re.S)
    count = 0
    for m in pat.finditer(text):
        name = m.group(1)
        try:
            a_lo = [float(x) for x in m.group(4).split(",")]
            a_mid = [float(x) for x in m.group(7).split(",")]
            a_hi = [float(x) for x in m.group(10).split(",")]
            T1 = float(m.group(2)); T2 = float(m.group(3))
            T3 = float(m.group(8))  # T_high จาก range 2
            if len(a_lo) == 9 and len(a_mid) == 9 and len(a_hi) == 9:
                _reg(name, T1, T2, T3, a_lo, a_mid, a_hi)
                count += 1
        except Exception:
            continue
    log.info(f"NASA-9 loader: parsed {count} species จาก {path}")


# =============================================================================
# §13. CLI
# =============================================================================

def _print_thermo_table():
    print("\n=== Thermo ที่ T=3000 K (NASA-9 จาก McBride 2002) ===")
    print(f"{'species':>8} {'cp/R':>10} {'h/RT':>10} {'s/R':>10} {'quality':>10}")
    for s in SPECIES_DB:
        try:
            cp, h, sr, _ = _thermo_piecewise(s, 3000.0)
            q = SPECIES_DB[s].data_quality
            print(f"{s:>8} {cp:10.4f} {h:10.4f} {sr:10.4f} {q:>10}")
        except KeyError:
            pass
    print()


def _print_equilibrium_demo():
    print("\n=== Equilibrium (air11, 100 Pa) ===")
    species = MODELS["air11"]
    el = {"N": 0.78*2, "O": 0.21*2}
    print(f"{'T[K]':>6} " + " ".join(f"{s:>8}" for s in species))
    for T in [3000, 6000, 10000, 15000]:
        try:
            X = equilibrium_composition(T, 100.0, el, species, max_iter=300)
        except Exception as e:
            print(f"  T={T}: failed ({e})"); continue
        print(f"{T:6d} " + " ".join(f"{X[species.index(s)]:8.3e}" for s in species))
    print()


def _print_shock_demo():
    print("\n=== Normal shock (U=11 km/s, p=100 Pa) — 3T ===")
    inp = ShockInputs(p_inf=100.0, T_inf=250.0, U_inf=11000.0,
                      model="air11", L_domain=0.3, n_points=20,
                      three_temperature=True)
    try:
        states = solve_normal_shock(inp)
    except Exception as e:
        print(f"  failed: {e}"); return
    print(f"{'x[m]':>8} {'T_tr[K]':>10} {'T_v[K]':>10} {'T_e[K]':>10} "
          f"{'p[Pa]':>10} {'q_rad':>12}")
    for st in states:
        print(f"{st.x:8.4f} {st.T_tr:10.2f} {st.T_v:10.2f} {st.T_e:10.2f} "
              f"{st.p:10.2f} {st.q_rad:12.3e}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Hypersonic chemistry detailed v2")
    ap.add_argument("--model", default="air11", choices=list(MODELS.keys()))
    ap.add_argument("--demo", default="all",
                    choices=["all", "thermo", "eq", "shock", "none"])
    ap.add_argument("--nasa9-file", default=None)
    args = ap.parse_args()

    print(BANNER)
    log.info(f"model={args.model} species={len(MODELS[args.model])}")

    if args.nasa9_file:
        load_nasa9_file(args.nasa9_file)

    if args.demo in ("all", "thermo"):  _print_thermo_table()
    if args.demo in ("all", "eq"):      _print_equilibrium_demo()
    if args.demo in ("all", "shock"):   _print_shock_demo()
    print("done.")


if __name__ == "__main__":
    main()
