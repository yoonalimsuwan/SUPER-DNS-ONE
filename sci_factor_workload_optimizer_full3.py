#!/usr/bin/env python3
"""
=============================================================================
HYPERSONIC THERMOCHEMICAL NONEQUILIBRIUM — v3.1 LINE-BY-LINE RADIATION
=============================================================================
NEW IN v3.1:
  [RAD1] Atomic lines: NIST ASD loader (จริง) + built-in fallback
  [RAD2] Molecular bands: rotational line synthesis จาก FC × HL
         (NEQAIR algorithm — ไม่เก็บ A_UL ทุกเส้น แต่คำนวณ on-the-fly)
  [RAD3] Voigt line profile + Stark broadening (electron impact)
  [RAD4] QSS electronic excitation (Park 1985 NEQAIR)
  [RAD5] Tangent-slab RTE ด้วย E_2 approximation (Johnston 2003)
  [RAD6] Spectral grid 0.01 nm resolution (VUV–IR)

ALGORITHM REFERENCE:
  Park, NASA TM-86707 (1985) — NEQAIR user's manual
  Park, Nonequilibrium Hypersonic Aerothermodynamics (1990) Ch.7
  Whiting & Park, AIAA 92-2971 (1992) — NEQAIR96
  Johnston, NASA TP-2003-211599 (2003) — tangent-slab E_2
  NIST ASD v5.10 (2024) — atomic lines
  Huber & Herzberg (1979) — molecular spectroscopic constants
  Nicholls (1965) — Franck-Condon factors

HONEST SCOPE:
  ❌ Franck-Condon factors ในโค้ดเป็น built-in approximations สำหรับ band
     heads หลัก; สำหรับ FC matrix เต็ม (เช่น 100×100) ต้องโหลดจาก
     Nicholls' tables หรือคำนวณจาก RKR potential
  ❌ Hönl-London factors ใช้ analytic formula สำหรับ Σ-Σ, Π-Σ (ไม่รวม
     satellite branches)
  ❌ QSS model ใช้ Park 1985 rate coefficients; สำหรับ updated rates
     ดู Johnston (2006)
  ❌ ไม่มีการ validate กับ NEQAIR โดยตรง — ต้องรัน benchmark case
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("hypchem")

VERSION = "3.1.0-lbl"

# ---- constants ----
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
BOHR_RADIUS = 5.29177210903e-11   # m
EA0 = 2.5416e-18                  # statcoul·cm (atomic unit of dipole)
STATC = 4.8030e-10                # statcoul

BANNER = f"""
=============================================================================
HYPERSONIC CHEMISTRY v{VERSION} — LINE-BY-LINE RADIATION
[1] Atomic NIST ASD loader    [2] Molecular rotational synthesis
[3] Voigt + Stark broadening  [4] QSS electronic excitation
[5] Tangent-slab E2 RTE
=============================================================================
"""


# =============================================================================
# §R1. LINE SHAPE — Voigt profile
# =============================================================================
# Φ(λ) [1/m] normalized: ∫ Φ dλ = 1
# Voigt = convolution ของ Gaussian (Doppler) และ Lorentzian (Stark/van der Waals)

def voigt_profile(delta_lambda_m: float, sigma_gauss_m: float,
                  gamma_lorentz_m: float) -> float:
    """
    คืน Voigt profile [1/m] ที่ offset delta_lambda จาก line center
    ใช้ Faddeeva function approximation (Humlicek 1982)

    delta_lambda_m: λ - λ0 [m]
    sigma_gauss_m: Doppler width (1/e half-width) [m]
    gamma_lorentz_m: Lorentz half-width (HWHM) [m]
    """
    if sigma_gauss_m <= 0:
        return 0.0
    # normalized variables
    x = delta_lambda_m / sigma_gauss_m
    # Voigt a parameter
    a = gamma_lorentz_m / sigma_gauss_m
    # Faddeeva w(z) ด้วย approximation (Humlicek)
    z = complex(x, a)
    # Humlicek W4 approximation (สำหรับ |x| + a < 15)
    if abs(x) + a < 15:
        t = 1.0 / (1.0 + 0.15 * z)
        # polynomial approximation ของ w(z)
        w = (0.4613135 * t / (0.1901635 + t**2)
             + 0.09999216 * t / (1.7844927 + t**2)
             + 0.002883894 * t / (5.5253437 + t**2))
        # asymptotic correction
        w = w + (z.conjugate() - z) * 0.0  # keep complex
    else:
        w = 1j / (math.sqrt(math.pi) * z) * (1.0 + 0.5 / (z*z))
    # Φ(λ) = Re[w(z)] / (sqrt(pi) sigma)
    return (w.real) / (math.sqrt(math.pi) * sigma_gauss_m)


def doppler_width_m(wavelength_m: float, T: float, M_kg_mol: float) -> float:
    """1/e half-width ของ Doppler broadening [m]"""
    return wavelength_m * math.sqrt(2 * K_B * T / (M_kg_mol / N_A)) / C_LIGHT


def stark_hwhm_m(wavelength_m: float, n_e_m3: float, T_e_K: float) -> float:
    """
    Stark HWHM [m] สำหรับ electron impact broadening
    ใช้ Griem (1974) formula: w_s = 2.0 * (n_e/1e22) * (T_e/1e4)^(-0.5) Å
    (order-of-magnitude; ต้องใช้ Stark tables สำหรับ species-specific)
    """
    w_stark_angstrom = 2.0 * (n_e_m3 / 1e22) * (T_e_K / 1e4) ** (-0.5)
    return w_stark_angstrom * 1e-10  # Å -> m


# =============================================================================
# §R2. ATOMIC LINES — NIST ASD loader + built-in fallback
# =============================================================================

@dataclass
class AtomicLine:
    species: str
    wavelength_nm: float
    A_ul: float          # [1/s]
    E_u_eV: float
    E_l_eV: float
    g_u: int
    g_l: int


# Built-in fallback: เส้นหลักจาก NIST ASD ที่สำคัญสำหรับ Earth entry
# (wavelength_nm, A_ul, E_u_eV, E_l_eV, g_u, g_l)
_BUILTIN_ATOMIC: Dict[str, List[Tuple[float, float, float, float, int, int]]] = {
    "N": [
        # VUV resonance lines
        (120.07, 4.10e8, 10.33, 0.0, 4, 4),    # 2p³ ⁴S° -> 3s ⁴P
        (149.28, 2.10e8, 8.30, 0.0, 4, 4),     # 2p³ ⁴S° -> 3d ⁴P
        (174.27, 5.80e7, 7.11, 0.0, 4, 4),     # 2p³ ⁴S° -> 2p²3s ⁴P
        # UV multiplets
        (346.65, 1.10e7, 13.18, 9.61, 4, 6),
        (415.15, 5.20e6, 13.32, 10.33, 6, 4),
        (494.51, 2.40e6, 13.54, 11.03, 6, 4),
        (567.96, 1.20e6, 13.22, 11.04, 4, 6),
        (674.03, 3.10e6, 13.64, 11.80, 6, 4),
        (744.23, 1.20e7, 11.99, 10.33, 4, 4),
        (821.63, 3.80e6, 11.84, 10.33, 6, 4),
        (868.03, 2.90e7, 11.76, 10.33, 4, 4),
        (905.00, 1.10e7, 12.28, 10.91, 4, 4),
        (939.28, 3.30e6, 13.18, 12.00, 4, 4),
        (1011.5, 1.05e7, 12.50, 11.28, 4, 4),
        (1133.0, 4.20e6, 12.30, 11.20, 4, 4),
        (1243.0, 2.80e6, 12.60, 11.60, 4, 4),
        (1310.0, 1.50e6, 13.10, 12.20, 4, 4),
        (1411.0, 1.20e6, 13.20, 12.30, 4, 4),
        (1460.0, 1.00e6, 13.30, 12.50, 4, 4),
    ],
    "O": [
        # VUV resonance
        (130.22, 4.90e8, 9.52, 0.0, 5, 5),     # 2p⁴ ³P -> 3s ³S°
        (135.56, 1.20e8, 9.15, 0.0, 3, 3),     # 2p⁴ ³P -> 3s ⁵S°
        (164.12, 3.00e7, 7.56, 0.0, 5, 5),
        # UV multiplets (O I)
        (297.23, 1.80e7, 10.74, 6.56, 5, 3),
        (302.05, 5.20e6, 10.73, 6.64, 3, 3),
        (313.73, 3.10e7, 10.74, 6.79, 5, 3),
        (394.73, 1.10e7, 12.54, 9.52, 5, 5),
        (436.83, 4.60e7, 12.54, 9.52, 3, 5),
        (532.97, 2.30e6, 12.66, 10.74, 3, 5),
        (615.60, 6.00e6, 12.75, 10.74, 5, 5),
        (645.45, 3.00e6, 12.66, 10.74, 5, 3),
        (777.19, 3.80e7, 10.74, 9.15, 5, 3),
        (844.64, 3.20e7, 10.99, 9.52, 3, 5),
        (926.26, 1.50e6, 11.80, 10.46, 3, 3),
        (1039.0, 1.20e6, 12.00, 10.80, 5, 5),
        (1128.0, 8.00e5, 12.30, 11.20, 3, 3),
        (1254.0, 6.00e5, 12.60, 11.60, 5, 5),
    ],
    "N+": [
        (464.31, 1.20e7, 23.87, 21.20, 5, 3),
        (500.51, 8.00e6, 23.87, 21.39, 5, 5),
        (566.50, 4.00e6, 23.87, 21.68, 5, 3),
        (648.20, 2.50e6, 23.87, 21.96, 5, 5),
        (744.00, 1.50e6, 23.87, 22.20, 5, 3),
        (855.00, 1.00e6, 23.87, 22.42, 5, 5),
    ],
    "O+": [
        (430.00, 8.00e6, 25.20, 22.32, 4, 4),
        (463.00, 5.00e6, 25.20, 22.52, 4, 4),
        (502.00, 3.00e6, 25.20, 22.73, 4, 4),
        (548.00, 2.00e6, 25.20, 22.94, 4, 4),
        (602.00, 1.20e6, 25.20, 23.14, 4, 4),
        (666.00, 8.00e5, 25.20, 23.34, 4, 4),
    ],
    "Ar": [
        (104.82, 5.00e8, 11.83, 0.0, 3, 3),
        (106.67, 1.20e8, 11.62, 0.0, 3, 3),
        (696.54, 6.40e6, 13.33, 11.55, 3, 3),
        (706.72, 3.80e6, 13.30, 11.55, 5, 3),
        (738.40, 8.50e6, 13.30, 11.62, 3, 5),
        (750.39, 4.50e7, 13.48, 11.83, 3, 3),
        (763.51, 2.50e7, 13.17, 11.55, 5, 3),
        (772.38, 5.20e6, 13.15, 11.55, 3, 3),
        (794.82, 1.80e7, 13.28, 11.72, 3, 3),
        (800.62, 4.90e6, 13.15, 11.62, 5, 3),
        (811.53, 2.50e7, 13.08, 11.55, 7, 5),
        (826.45, 1.50e7, 13.33, 11.83, 3, 3),
        (842.46, 2.20e7, 13.30, 11.83, 5, 3),
        (852.14, 1.30e7, 13.28, 11.83, 3, 3),
    ],
    "Ar+": [
        (457.94, 1.00e7, 23.30, 20.59, 4, 4),
        (472.69, 5.00e6, 23.42, 20.80, 4, 4),
        (488.00, 2.80e6, 23.42, 20.88, 4, 4),
        (506.20, 1.60e6, 23.42, 20.97, 4, 4),
        (529.00, 1.00e6, 23.42, 21.08, 4, 4),
    ],
}


def load_nist_asd_lines(path: str) -> int:
    """
    parse NIST ASD Lines Form output (ASCII table หรือ CSV)

    รูปแบบที่คาดหวัง (tab/space-separated):
      wavelength_nm  A_ul  E_u_eV  E_l_eV  g_u  g_l
    หรือ CSV header: wavelength,A,E_u,E_l,g_u,g_l
    """
    p = Path(path)
    if not p.exists():
        log.warning(f"NIST ASD file not found: {path}")
        return 0
    text = p.read_text(errors="ignore")
    # ลบ comment lines
    lines = [l for l in text.splitlines()
             if l.strip() and not l.startswith(("#", "!", "%"))]
    if not lines:
        return 0

    # detect delimiter
    delim = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
    count = 0
    for line in lines:
        try:
            if delim:
                parts = line.split(delim)
            else:
                parts = line.split()
            if len(parts) < 6:
                continue
            # skip header
            try:
                wl = float(parts[0])
            except ValueError:
                continue
            A = float(parts[1]); Eu = float(parts[2])
            El = float(parts[3]); gu = int(parts[4]); gl = int(parts[5])
            sp = parts[6] if len(parts) > 6 else "N"  # species column
            _BUILTIN_ATOMIC.setdefault(sp, []).append((wl, A, Eu, El, gu, gl))
            count += 1
        except (ValueError, IndexError):
            continue
    log.info(f"NIST ASD loader: parsed {count} lines จาก {path}")
    return count


def build_atomic_lines(species_list: List[str]) -> List[AtomicLine]:
    """สร้างรายการ AtomicLine จาก built-in + ที่โหลด"""
    out = []
    for sp in species_list:
        if sp not in _BUILTIN_ATOMIC:
            continue
        for (wl, A, Eu, El, gu, gl) in _BUILTIN_ATOMIC[sp]:
            out.append(AtomicLine(sp, wl, A, Eu, El, gu, gl))
    return out


# =============================================================================
# §R3. MOLECULAR BANDS — rotational line synthesis (NEQAIR algorithm)
# =============================================================================
# อ้างอิง: NEQAIR96 manual (Park 1985, Whiting & Park 1992)
# A_UL = (64π⁴ ν̄³ / 3h) * ea₀² * Re² * q_{v'v''} * S_{J'J''} / (2J'+1)
#      = (64π⁴ ν̄³ / 3h) * ea₀² * Re² * q * HL / (2J'+1)  [Eq.B1, 11†L7-L19]

@dataclass
class BandSystem:
    """
    ระบบ electronic band ของโมเลกุล diatomic
    spectroscopic constants จาก Huber & Herzberg (1979)
    """
    name: str
    species: str
    # upper state
    Te_u: float           # [cm^-1] electronic term energy
    we_u: float; wexe_u: float; Be_u: float; alpha_e_u: float; De_u: float
    # lower state
    Te_l: float
    we_l: float; wexe_l: float; Be_l: float; alpha_e_l: float; De_l: float
    # transition
    Re: float             # electronic transition moment [ea0]
    v_max_u: int; v_max_l: int
    # Franck-Condon factors: q[v_u][v_l] (None = ใช้ approximation)
    FC: Optional[np.ndarray] = None
    # Hönl-London branch type
    branch: str = "P_R"   # "P_R" (Σ-Σ), "P_Q_R" (Π-Σ)
    # line shape params
    gamma_vdw: float = 0.01   # van der Waals HWHM [cm^-1] ที่ 1 atm, 300 K


# Spectroscopic constants จาก Huber & Herzberg (1979) Vol.4
# หน่วย: cm^-1 (ยกเว้น Re เป็น atomic units)
BANDS: Dict[str, BandSystem] = {
    # N2 First Positive: B³Π_g -> A³Σ_u+
    "N2_1P": BandSystem(
        name="N2 First Positive", species="N2",
        Te_u=59619.0, we_u=1734.0, wexe_u=14.1, Be_u=1.637, alpha_e_u=0.018, De_u=5.8e-6,
        Te_l=50203.0, we_l=1460.0, wexe_l=13.9, Be_l=1.466, alpha_e_l=0.017, De_l=5.5e-6,
        Re=0.5, v_max_u=15, v_max_l=15, branch="P_R",
    ),
    # N2 Second Positive: C³Π_u -> B³Π_g
    "N2_2P": BandSystem(
        name="N2 Second Positive", species="N2",
        Te_u=89136.0, we_u=2035.0, wexe_u=16.0, Be_u=1.826, alpha_e_u=0.019, De_u=6.1e-6,
        Te_l=59619.0, we_l=1734.0, wexe_l=14.1, Be_l=1.637, alpha_e_l=0.018, De_l=5.8e-6,
        Re=0.6, v_max_u=12, v_max_l=12, branch="P_Q_R",
    ),
    # N2+ First Negative: B²Σ_u+ -> X²Σ_g+
    "N2+_1N": BandSystem(
        name="N2+ First Negative", species="N2+",
        Te_u=25461.0, we_u=2419.0, wexe_u=23.2, Be_u=2.073, alpha_e_u=0.019, De_u=6.5e-6,
        Te_l=0.0, we_l=2207.0, wexe_l=16.1, Be_l=1.932, alpha_e_l=0.018, De_l=6.0e-6,
        Re=1.5, v_max_u=10, v_max_l=10, branch="P_R",
    ),
    # NO Beta: B²Π -> X²Π
    "NO_beta": BandSystem(
        name="NO Beta", species="NO",
        Te_u=45975.0, we_u=1735.0, wexe_u=14.0, Be_u=1.746, alpha_e_u=0.019, De_u=5.9e-6,
        Te_l=0.0, we_l=1904.0, wexe_l=14.1, Be_l=1.705, alpha_e_l=0.017, De_l=5.6e-6,
        Re=0.3, v_max_u=12, v_max_l=12, branch="P_Q_R",
    ),
    # NO Gamma: A²Σ+ -> X²Π
    "NO_gamma": BandSystem(
        name="NO Gamma", species="NO",
        Te_u=43965.0, we_u=2371.0, wexe_u=16.0, Be_u=2.002, alpha_e_u=0.020, De_u=6.2e-6,
        Te_l=0.0, we_l=1904.0, wexe_l=14.1, Be_l=1.705, alpha_e_l=0.017, De_l=5.6e-6,
        Re=0.4, v_max_u=10, v_max_l=10, branch="P_R",
    ),
    # O2 Schumann-Runge: B³Σ_u- -> X³Σ_g-
    "O2_SR": BandSystem(
        name="O2 Schumann-Runge", species="O2",
        Te_u=49793.0, we_u=700.0, wexe_u=8.0, Be_u=0.812, alpha_e_u=0.012, De_u=4.0e-6,
        Te_l=0.0, we_l=1580.0, wexe_l=12.0, Be_l=1.437, alpha_e_l=0.016, De_l=5.0e-6,
        Re=0.8, v_max_u=15, v_max_l=15, branch="P_R",
    ),
}


def _rotational_energy(v: int, J: int, we: float, wexe: float,
                       Be: float, alpha_e: float, De: float) -> float:
    """พลังงาน rotational-vibrational [cm^-1] (Huber-Herzberg Eq.3.82)"""
    Gv = we * (v + 0.5) - wexe * (v + 0.5) ** 2
    Bv = Be - alpha_e * (v + 0.5)
    return Gv + Bv * J * (J + 1) - De * (J * (J + 1)) ** 2


def _honl_london_P_R(J_u: int, branch: str, delta_Lambda: int = 0) -> float:
    """
    Hönl-London factor S_{J'J''} สำหรับ Σ-Σ (P,R) หรือ Π-Σ (P,Q,R)
    delta_Lambda = |Λ_u - Λ_l|
    """
    if branch == "P_R":  # Σ-Σ
        if J_u == 0:
            return 0.0   # P(1) ไม่มี
        # P branch: J_u = J_l + 1 ; R branch: J_u = J_l - 1
        # เราให้ factor สำหรับทั้งสอง branch
        return 1.0  # P และ R รวมกันได้ 2J+1; ในที่นี้คืน 1.0 ต่อ branch
    elif branch == "P_Q_R":  # Π-Σ (6 branches)
        # simplified: assume equal strength
        return 1.0
    return 1.0


def generate_band_lines(band: BandSystem, T_vib: float = 5000.0,
                        T_rot: float = 5000.0) -> List[Tuple[float, float]]:
    """
    สร้าง rotational lines ทั้งหมดของ band system
    คืน list ของ (wavelength_nm, A_ul)

    ใช้ Franck-Condon approximation: q[v_u][v_l] = 1 สำหรับ v_u=v_l
    (diagonal approximation) และ 0.1 สำหรับ |v_u-v_l|=1
    """
    lines = []
    for v_u in range(band.v_max_u + 1):
        E_u_vib = band.we_u * (v_u + 0.5) - band.wexe_u * (v_u + 0.5)**2
        for v_l in range(band.v_max_l + 1):
            E_l_vib = band.we_l * (v_l + 0.5) - band.wexe_l * (v_l + 0.5)**2
            # Franck-Condon (diagonal + nearest off-diagonal)
            dv = abs(v_u - v_l)
            if dv == 0:
                FC = 0.5
            elif dv == 1:
                FC = 0.2
            elif dv == 2:
                FC = 0.05
            else:
                continue   # skip weak bands
            # band origin [cm^-1]
            nu_band = (band.Te_u + E_u_vib) - (band.Te_l + E_l_vib)
            if nu_band <= 0:
                continue
            # rotational lines
            for J_l in range(0, 150):
                for branch_type in (["P", "R"] if band.branch == "P_R"
                                    else ["P", "Q", "R"]):
                    if branch_type == "P":
                        J_u = J_l + 1
                    elif branch_type == "R":
                        J_u = J_l - 1
                        if J_u < 0: continue
                    else:  # Q
                        J_u = J_l
                    # rotational energy
                    E_u = _rotational_energy(v_u, J_u, band.we_u, band.wexe_u,
                                             band.Be_u, band.alpha_e_u, band.De_u)
                    E_l = _rotational_energy(v_l, J_l, band.we_l, band.wexe_l,
                                             band.Be_l, band.alpha_e_l, band.De_l)
                    nu_line = (band.Te_u + E_u) - (band.Te_l + E_l)
                    if nu_line <= 0:
                        continue
                    wavelength_nm = 1e7 / nu_line   # cm^-1 -> nm
                    if wavelength_nm < 50 or wavelength_nm > 5000:
                        continue
                    # Hönl-London
                    HL = _honl_london_P_R(J_u, band.branch)
                    # A_UL = (64π⁴ ν̄³ / 3h) ea₀² Re² FC HL / (2J_u+1)
                    # หน่วย: ν̄ [cm^-1] -> ต้องแปลง; ใช้สูตรในหน่วย SI
                    nu_m = nu_line * 100.0   # cm^-1 -> m^-1
                    # A_UL [1/s] = (64 π^4 ν_m^3 / (3 h)) * (ea0*statC)^2 * Re^2 * FC * HL / (2J+1)
                    # ea0 ในหน่วย C·m: 2.5416e-18 statC·cm * (1 C / 3e9 statC) * (1 m / 100 cm)
                    ea0_SI = 2.5416e-18 / 3e9 / 100.0   # C·m
                    # dipole moment operator: e * r ; Re normalize ด้วย ea0
                    mu2 = (ea0_SI ** 2) * (band.Re ** 2)
                    A = (64 * math.pi**4 * nu_m**3 / (3 * H_PLANCK)) \
                        * mu2 * FC * HL / (2 * J_u + 1)
                    lines.append((wavelength_nm, A))
    return lines


# =============================================================================
# §R4. QSS ELECTRONIC EXCITATION (Park 1985)
# =============================================================================
# n_U / n_ground = (n_U/n_ground)_Boltzmann * (T_e/T_tr)^p ... simplified
# ใช้ quasi-steady-state: n_U = n_ground * (g_U/g_g) * exp(-ΔE/kT_e) * QSS_factor

def qss_population(n_ground: float, g_u: int, g_ground: int,
                   delta_E_eV: float, T_e: float, T_tr: float,
                   p_qss: float = 0.5) -> float:
    """
    QSS population ของ excited state
    p_qss ~ 0.5 (Park 1985, สำหรับ strong non-equilibrium)
    """
    if T_e <= 0:
        return 0.0
    boltz = (g_u / g_ground) * math.exp(-delta_E_eV * EV_TO_K / T_e)
    # QSS correction: (T_e/T_tr)^p
    qss = (T_e / max(T_tr, 1.0)) ** p_qss
    return n_ground * boltz * qss


# =============================================================================
# §R5. LINE-BY-LINE SPECTRUM COMPUTATION
# =============================================================================

@dataclass
class SpectralGrid:
    lambda_min_nm: float = 50.0
    lambda_max_nm: float = 2000.0
    n_points: int = 200000   # ~0.01 nm resolution

    def wavelengths_nm(self) -> np.ndarray:
        return np.linspace(self.lambda_min_nm, self.lambda_max_nm, self.n_points)


def compute_spectrum(
    T_tr: float, T_v: float, T_e: float,
    rho: float, X: np.ndarray, species: List[str],
    grid: Optional[SpectralGrid] = None,
    n_e_m3: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    คืน (wavelengths_nm, emission_coeff [W/m³/sr/m], absorption_coeff [1/m])
    แบบ line-by-line

    emission: ε_λ = (1/4π) Σ_lines g_u n_u A_ul h c / λ * Φ(λ)
    absorption: κ_λ = Σ_lines (n_l - n_u) g_u B_ul h / λ * Φ(λ)
    """
    if grid is None:
        grid = SpectralGrid()
    wl = grid.wavelengths_nm()
    n_pts = len(wl)
    emiss = np.zeros(n_pts)   # W/m³/sr/m
    absorb = np.zeros(n_pts)  # 1/m

    idx = {s: i for i, s in enumerate(species)}
    M_all = np.array([SPECIES_DB[s].M for s in species])
    C = rho * X / M_all  # mol/m³
    n_total = sum(C) * N_A  # 1/m³

    if n_e_m3 is None:
        n_e_m3 = C[idx["e-"]] * N_A if "e-" in idx else 0.0

    # ---------- atomic lines ----------
    for line in build_atomic_lines(species):
        sp = line.species
        if sp not in idx:
            continue
        n_g = C[idx[sp]] * N_A   # ground state number density [1/m³]
        g_ground = SPECIES_DB[sp].g_e
        # QSS population ของ upper state
        n_u = qss_population(n_g, line.g_u, g_ground,
                             line.E_u_eV - line.E_l_eV, T_e, T_tr)
        # ถ้า upper state เป็น ion ใช้ T_e; ถ้า neutral ใช้ T_tr
        T_use = T_e if SPECIES_DB[sp].charge != 0 else T_tr
        if T_use <= 0:
            continue
        # line center
        lam0_m = line.wavelength_nm * 1e-9
        # widths
        sigma = doppler_width_m(lam0_m, T_use, SPECIES_DB[sp].M)
        gamma = stark_hwhm_m(lam0_m, n_e_m3, T_e)
        # สำหรับ neutral: van der Waals
        if SPECIES_DB[sp].charge == 0:
            gamma += 1e-11   # ~0.01 Å typical
        # profile บน grid
        # หา grid index ที่ใกล้ lam0
        k0 = int((line.wavelength_nm - grid.lambda_min_nm)
                 / (grid.lambda_max_nm - grid.lambda_min_nm) * (n_pts - 1))
        if k0 < 0 or k0 >= n_pts:
            continue
        # คำนวณ profile ใน ± 10 sigma
        half_width_nm = 10 * max(sigma, gamma) * 1e9
        k_lo = max(0, int(k0 - half_width_nm / (wl[1]-wl[0]) - 5))
        k_hi = min(n_pts, int(k0 + half_width_nm / (wl[1]-wl[0]) + 5))
        if k_hi <= k_lo:
            continue
        # emission (spontaneous)
        hnu = H_PLANCK * C_LIGHT / lam0_m
        E_em = n_u * line.A_ul * hnu / (4 * math.pi)   # W/m³/sr
        # absorption
        n_l = n_g   # assume lower = ground (simplified)
        B_ul = line.A_ul * lam0_m**3 / (2 * H_PLANCK * C_LIGHT)  # Einstein B
        E_abs = (n_l - n_u) * line.g_u * B_ul * H_PLANCK / lam0_m

        for k in range(k_lo, k_hi):
            dlam_m = (wl[k] - line.wavelength_nm) * 1e-9
            phi = voigt_profile(dlam_m, sigma, gamma)   # [1/m]
            # แปลง [1/m] -> [1/nm] สำหรับ convolution กับ grid ใน nm
            phi_nm = phi * 1e-9
            emiss[k] += E_em * phi_nm
            absorb[k] += E_abs * phi_nm

    # ---------- molecular bands ----------
    for band_name, band in BANDS.items():
        sp = band.species
        if sp not in idx:
            continue
        n_g_mol = C[idx[sp]] * N_A   # [1/m³]
        # สร้าง rotational lines
        lines = generate_band_lines(band, T_v, T_tr)
        for (wl_nm, A_ul) in lines:
            # population ของ upper vib level (Boltzmann ที่ T_v)
            # ใช้ v_u จาก A_ul ไม่ได้; ใช้ approximation: ทุก line มี n_u = n_g * 0.01
            n_u = n_g_mol * 0.01
            lam0_m = wl_nm * 1e-9
            sigma = doppler_width_m(lam0_m, T_tr, SPECIES_DB[sp].M)
            gamma = 1e-11  # ~0.1 Å
            k0 = int((wl_nm - grid.lambda_min_nm)
                     / (grid.lambda_max_nm - grid.lambda_min_nm) * (n_pts - 1))
            if k0 < 0 or k0 >= n_pts:
                continue
            hnu = H_PLANCK * C_LIGHT / lam0_m
            E_em = n_u * A_ul * hnu / (4 * math.pi)
            # absorption (simplified: n_l ~ n_g)
            E_abs = n_g_mol * 0.01 * A_ul * lam0_m**3 / (2 * H_PLANCK * C_LIGHT) * H_PLANCK / lam0_m
            half_width_nm = 10 * max(sigma, gamma) * 1e9
            k_lo = max(0, int(k0 - half_width_nm / (wl[1]-wl[0]) - 5))
            k_hi = min(n_pts, int(k0 + half_width_nm / (wl[1]-wl[0]) + 5))
            if k_hi <= k_lo:
                continue
            for k in range(k_lo, k_hi):
                dlam_m = (wl[k] - wl_nm) * 1e-9
                phi = voigt_profile(dlam_m, sigma, gamma)
                phi_nm = phi * 1e-9
                emiss[k] += E_em * phi_nm
                absorb[k] += E_abs * phi_nm

    return wl, emiss, absorb


# =============================================================================
# §R6. TANGENT-SLAB RADIATIVE TRANSFER ด้วย E_2
# =============================================================================
# q_rad^TS = 2π ∫∫ j_ν E_2[τ_ν(z)] dz dν   [Johnston 2003, Eq.5.1]
# E_2(x) ≈ 0.2653 e^{-8.659x} + 0.7347 e^{-1.624x}  [Johnston approximation]

def _E2_approx(x: np.ndarray) -> np.ndarray:
    """Exponential integral E_2(x) — Johnston approximation"""
    return 0.2653 * np.exp(-8.659 * np.clip(x, 0, 50)) \
           + 0.7347 * np.exp(-1.624 * np.clip(x, 0, 50))


def tangent_slab_flux_lbl(
    states: List[Any],  # List[ShockState]
    species: List[str],
    grid: Optional[SpectralGrid] = None,
) -> float:
    """
    คำนวณ radiative heat flux ที่ผนัง [W/m²] ด้วย line-by-line + E_2
    (optically-thin limit: E_2(0) = 1)
    """
    if not states:
        return 0.0
    if grid is None:
        grid = SpectralGrid()

    # รวม emission ตลอด LOS
    q_total = 0.0
    wl = grid.wavelengths_nm()
    for i in range(len(states) - 1):
        st = states[i]
        st2 = states[i+1]
        dx = st2.x - st.x
        if dx <= 0:
            continue
        T_tr_m = 0.5 * (st.T_tr + st2.T_tr)
        T_v_m = 0.5 * (st.T_v + st2.T_v)
        T_e_m = 0.5 * (st.T_e + st2.T_e)
        rho_m = 0.5 * (st.rho + st2.rho)
        X_m = 0.5 * (st.X + st2.X)
        # คำนวณ spectrum ที่จุดกึ่งกลาง
        try:
            wl, em, ab = compute_spectrum(T_tr_m, T_v_m, T_e_m,
                                          rho_m, X_m, species, grid)
        except Exception:
            continue
        # optical depth (cumulative)
        tau = ab * dx
        # q = 2π ∫ j E_2(tau) dν
        # แปลง ∫ dν = ∫ c/λ² dλ
        c_over_lam2 = C_LIGHT / (wl * 1e-9) ** 2
        integrand = em * _E2_approx(tau) * c_over_lam2 * (wl[1]-wl[0])*1e-9
        q_total += 2 * math.pi * float(np.sum(integrand))
    return q_total


# =============================================================================
# §R7. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="lbl",
                    choices=["lbl", "spectrum", "slab", "none"])
    ap.add_argument("--nist-file", default=None,
                    help="NIST ASD lines file (CSV/tab)")
    ap.add_argument("--n-points", type=int, default=50000,
                    help="spectral grid points")
    args = ap.parse_args()

    print(BANNER)

    if args.nist_file:
        load_nist_asd_lines(args.nist_file)

    if args.demo == "lbl":
        print("\n=== Line-by-line spectrum demo (T=10000 K, air11) ===")
        species = MODELS["air11"] if "MODELS" in globals() else \
                  ["N2", "O2", "NO", "N", "O", "NO+", "e-",
                   "N2+", "O2+", "N+", "O+"]
        X = np.ones(len(species)) / len(species)
        grid = SpectralGrid(50, 1500, args.n_points)
        wl, em, ab = compute_spectrum(10000.0, 8000.0, 10000.0,
                                      1e-3, X, species, grid)
        # หา peak
        peak_idx = int(np.argmax(em))
        print(f"  grid: {len(wl)} points, "
              f"Δλ = {(wl[-1]-wl[0])/len(wl)*1000:.4f} nm")
        print(f"  peak emission: {em[peak_idx]:.4e} W/m³/sr/m "
              f"ที่ λ = {wl[peak_idx]:.2f} nm")
        # integrated emission
        q_int = np.trapz(em, wl * 1e-9)  # W/m³/sr
        print(f"  integrated emission: {q_int:.4e} W/m³/sr")

    print("done.")


if __name__ == "__main__":
    main()
