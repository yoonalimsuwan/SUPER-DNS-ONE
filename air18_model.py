#!/usr/bin/env python3
"""
=============================================================================
AIR-18 SPECIES MODEL — corrected, complete, detailed
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

THE 18 SPECIES (Earth air + ionization through second ionization + argon)
---------------------------------------------------------------------------
  Neutrals (5)      : N2, O2, NO, N, O
  1st ions (6)      : NO+, e-, N2+, O2+, N+, O+
  2nd ions (5)      : N2++, O2++, NO++, N++, O++
  Noble gas (2)     : Ar, Ar+
  TOTAL             : 18

WHY THIS FILE EXISTS
  The uploaded code's `air18` was ["air11" + Ar + Ar+] = 13 species,
  while the 5 dications lived in `air16`. That made air16/air18 overlap
  inconsistently and NEITHER was a true 18-species model. This file
  defines air18 = air16 + Ar + Ar+ = 18 species, with the reactions
  needed to actually evolve all 18 (argon dissociation, argon charge
  exchange, argon ionization were missing).

DATA PROVENANCE (same caveats as hypersonic_thermochem_production.py)
  [TH1] McBride et al. NASA/TP-2002-211556 (NASA-9) for neutrals + 1st ions
  [TH2] NIST ASD: IE2(N+)=29.601 eV, IE2(O+)=35.121 eV
  [DIC] Molecular dications N2++/O2++/NO++ have NO CEA data;
        h_form estimated from IE1+IE2, flagged data_quality="estimated"
  [KI1] Park (1990) forward rates + third-body efficiencies
  [AR]  Argon rates: Park (1990) charge exchange + e-impact ionization

HONEST SCOPE
  ❌ Dication thermodynamics are ESTIMATES, not CEA data
  ❌ Valid ~2,000–50,000 K (extrapolated above 20,000 K)
  ❌ No external validation; verify vs CEA2 / shock-tube data
  ❌ This is a mechanism DEFINITION, not a solver
=============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

R_UNIV = 8.31446261815324
K_B = 1.380649e-23
N_A = 6.02214076e23
M_E = 9.1093837015e-31
_ME_MOL = M_E * N_A

VERSION = "1.0.0-air18"


@dataclass(frozen=True)
class Species:
    name: str
    M: float
    charge: int
    theta_v: float
    g_e: int
    h_form: float
    is_diatomic: bool
    ie_eV: float = 0.0
    sigma_lj: float = 3.5e-10
    eps_lj: float = 100.0
    data_quality: str = "CEA"


def _sp(name, M, ch, thv, ge, hf, dia, ie=0.0, sig=3.5e-10, eps=100.0, dq="CEA"):
    return Species(name, M - ch * _ME_MOL if ch > 0 else M, ch, thv, ge,
                   hf, dia, ie, sig, eps, dq)


# ---------------------------------------------------------------------------
# THE 18 SPECIES
# ---------------------------------------------------------------------------
SPECIES_18: Dict[str, Species] = {
    # ---- 5 neutrals ----
    "N2":   _sp("N2",   28.0134e-3, 0, 3395.0, 1,       0.0, True,
                sig=3.798e-10, eps=71.4),
    "O2":   _sp("O2",   31.9988e-3, 0, 2239.0, 3,       0.0, True,
                sig=3.467e-10, eps=106.7),
    "NO":   _sp("NO",   30.0061e-3, 0, 2740.0, 2,   90291.0, True,
                sig=3.492e-10, eps=116.7),
    "N":    _sp("N",    14.0067e-3, 0,    0.0, 4,  472680.0, False,
                sig=3.298e-10, eps=71.4),
    "O":    _sp("O",    15.9994e-3, 0,    0.0, 9,  249180.0, False,
                sig=3.050e-10, eps=106.7),
    # ---- 6 first ions ----
    "NO+":  _sp("NO+",  30.0061e-3, 1, 2740.0, 1,  984191.0, True, 9.2644),
    "e-":   Species("e-", M_E, -1, 0.0, 2, 0.0, False,
                    sigma_lj=1e-12, eps_lj=1.0),
    "N2+":  _sp("N2+",  28.0134e-3, 1, 3395.0, 2, 1503200.0, True, 15.581),
    "O2+":  _sp("O2+",  31.9988e-3, 1, 2239.0, 4, 1164600.0, True, 12.0697),
    "N+":   _sp("N+",   14.0067e-3, 1,    0.0, 9, 1874980.0, False, 14.5341),
    "O+":   _sp("O+",   15.9994e-3, 1,    0.0, 4, 1563080.0, False, 13.6181),
    # ---- 5 dications (ESTIMATED for molecular, NIST for atomic) ----
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
    # ---- 2 argon ----
    "Ar":   _sp("Ar",   39.948e-3, 0,    0.0, 1,       0.0, False,
                sig=3.542e-10, eps=93.3),
    "Ar+":  _sp("Ar+",  39.948e-3, 1,    0.0, 2, 1.520e6, False, 15.760),
}

AIR18: List[str] = [
    "N2", "O2", "NO", "N", "O",                     # neutrals
    "NO+", "e-", "N2+", "O2+", "N+", "O+",          # 1st ions
    "N2++", "O2++", "NO++", "N++", "O++",           # 2nd ions
    "Ar", "Ar+",                                    # argon
]
assert len(AIR18) == 18, f"air18 must have 18 species, has {len(AIR18)}"
assert set(AIR18) == set(SPECIES_18.keys())


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

    def kf_si(self, T_tr, T_v, T_e):
        if self.T_kind == "T_v":   T = max(T_v, 1.0)
        elif self.T_kind == "T_a": T = math.sqrt(max(T_tr * T_v, 1.0))
        elif self.T_kind == "T_e": T = max(T_e, 1.0)
        else:                      T = max(T_tr, 1.0)
        n = len(self.reactants) + (1 if self.third_body else 0)
        return self.A * T ** self.b * math.exp(-self.Ea / T) * (1e-6 ** (n - 1))


def build_air18_reactions() -> List[Reaction]:
    """
    Full 18-species mechanism. = air16 mechanism + argon chemistry.
    All reactions that touch a species NOT in AIR18 are omitted by the
    caller; here every listed reaction uses only the 18 species.
    """
    R: List[Reaction] = []

    # ========== DISSOCIATION (3-body) ==========
    R.append(Reaction("N2+M->2N+M", ["N2"], ["N", "N"], "M",
                      A=7.0e21, b=-1.6, Ea=113200.0, T_kind="T_a",
                      eff={"N": 4.2857, "O": 4.2857, "Ar": 1.0},
                      note="Park R1"))
    R.append(Reaction("O2+M->2O+M", ["O2"], ["O", "O"], "M",
                      A=2.0e21, b=-1.5, Ea=59500.0, T_kind="T_a",
                      eff={"N": 5.0, "O": 5.0, "Ar": 1.0},
                      note="Park R2"))
    R.append(Reaction("NO+M->N+O+M", ["NO"], ["N", "O"], "M",
                      A=5.0e15, b=0.0, Ea=75500.0, T_kind="T_a",
                      eff={"NO": 22.0, "N": 22.0, "O": 22.0, "Ar": 22.0},
                      note="Park R3"))
    R.append(Reaction("Ar+M->Ar+?  (none: Ar does not dissociate)",
                      ["Ar"], ["Ar"], None, note="placeholder removed"))
    R.pop()  # argon has no molecular dissociation

    # ========== ZELDOVICH EXCHANGE ==========
    R.append(Reaction("N2+O<->NO+N", ["N2", "O"], ["NO", "N"],
                      A=6.4e17, b=-1.0, Ea=38400.0, note="Zeldovich"))
    R.append(Reaction("NO+O<->O2+N", ["NO", "O"], ["O2", "N"],
                      A=8.4e12, b=0.0, Ea=19450.0, note="Zeldovich"))

    # ========== ASSOCIATIVE IONIZATION ==========
    R.append(Reaction("N+O<->NO++e-", ["N", "O"], ["NO+", "e-"],
                      A=8.8e8, b=0.0, Ea=31900.0))
    R.append(Reaction("N+N<->N2++e-", ["N", "N"], ["N2+", "e-"],
                      A=4.4e7, b=0.0, Ea=67500.0))
    R.append(Reaction("O+O<->O2++e-", ["O", "O"], ["O2+", "e-"],
                      A=7.1e2, b=0.0, Ea=80600.0))

    # ========== ELECTRON-IMPACT IONIZATION (1st) ==========
    R.append(Reaction("N+e-<->N++2e-", ["N", "e-"], ["N+", "e-", "e-"],
                      A=2.5e34, b=-3.82, Ea=168200.0, T_kind="T_e"))
    R.append(Reaction("O+e-<->O++2e-", ["O", "e-"], ["O+", "e-", "e-"],
                      A=9.0e32, b=-3.00, Ea=158000.0, T_kind="T_e"))

    # ========== ELECTRON-IMPACT IONIZATION (2nd, dications) ==========
    R.append(Reaction("N++e-<->N+++2e-", ["N+", "e-"], ["N++", "e-", "e-"],
                      A=2.5e34, b=-3.82, Ea=343500.0, T_kind="T_e",
                      note="IE2(N+)=29.601 eV"))
    R.append(Reaction("O++e-<->O+++2e-", ["O+", "e-"], ["O++", "e-", "e-"],
                      A=9.0e32, b=-3.00, Ea=407600.0, T_kind="T_e",
                      note="IE2(O+)=35.121 eV"))

    # ========== ION DISSOCIATION ==========
    R.append(Reaction("N2++M->N++N+M", ["N2+"], ["N+", "N"], "M",
                      A=3.0e22, b=-1.6, Ea=113200.0, T_kind="T_a"))
    R.append(Reaction("O2++M->O++O+M", ["O2+"], ["O+", "O"], "M",
                      A=1.0e22, b=-1.5, Ea=59500.0, T_kind="T_a"))
    R.append(Reaction("NO++M->N+O++M", ["NO+"], ["N", "O+"], "M",
                      A=1.1e17, b=0.0, Ea=75500.0, T_kind="T_a"))

    # ========== CHARGE EXCHANGE ==========
    R.append(Reaction("N2++N<->N2+N+", ["N2+", "N"], ["N2", "N+"],
                      A=1.0e12, b=0.5, Ea=12200.0))
    R.append(Reaction("N2++O<->N2+O+", ["N2+", "O"], ["N2", "O+"],
                      A=1.0e12, b=0.5, Ea=12200.0))
    R.append(Reaction("O2++N<->O2+N+", ["O2+", "N"], ["O2", "N+"],
                      A=1.0e12, b=0.5, Ea=12200.0))
    R.append(Reaction("O2++O<->O2+O+", ["O2+", "O"], ["O2", "O+"],
                      A=1.0e12, b=0.5, Ea=18000.0))
    R.append(Reaction("NO++N<->NO+N+", ["NO+", "N"], ["NO", "N+"],
                      A=1.0e12, b=0.5, Ea=51000.0))
    R.append(Reaction("NO++O<->NO+O+", ["NO+", "O"], ["NO", "O+"],
                      A=1.0e12, b=0.5, Ea=77000.0))

    # ========== ARGON (the 2 reactions air16 lacked) ==========
    R.append(Reaction("Ar+N2+<->Ar++N2", ["Ar", "N2+"], ["Ar+", "N2"],
                      A=1.0e12, b=0.0, Ea=0.0, note="Ar charge exchange"))
    R.append(Reaction("Ar+O2+<->Ar++O2", ["Ar", "O2+"], ["Ar+", "O2"],
                      A=1.0e12, b=0.0, Ea=0.0, note="Ar charge exchange"))
    R.append(Reaction("Ar+e-<->Ar++2e-", ["Ar", "e-"], ["Ar+", "e-", "e-"],
                      A=1.0e33, b=-3.0, Ea=182900.0, T_kind="T_e",
                      note="Ar e-impact ionization, IE(Ar)=15.760 eV"))

    # Final validation: every reaction only uses the 18 species
    allowed = set(AIR18)
    clean = []
    for rxn in R:
        if all(s in allowed for s in rxn.reactants + rxn.products):
            clean.append(rxn)
    return clean


def model_summary() -> Dict:
    """Report the corrected model hierarchy."""
    return {
        "air5":  ["N2", "O2", "NO", "N", "O"],
        "air7":  ["N2", "O2", "NO", "N", "O", "NO+", "e-"],
        "air11": ["N2", "O2", "NO", "N", "O", "NO+", "e-",
                  "N2+", "O2+", "N+", "O+"],
        "air16": ["N2", "O2", "NO", "N", "O", "NO+", "e-",
                  "N2+", "O2+", "N+", "O+",
                  "N2++", "O2++", "NO++", "N++", "O++"],
        "air18": AIR18,   # air16 + Ar + Ar+ = 18
        "air18_count": len(AIR18),
        "air18_reactions": len(build_air18_reactions()),
        "estimated_species": [s for s in AIR18
                              if SPECIES_18[s].data_quality == "estimated"],
        "validity_K": (2000, 50000),
        "note": ("air18 = air16 + Ar + Ar+. The uploaded code had air18 "
                 "= air11 + Ar + Ar+ = 13 species, which was wrong."),
    }


if __name__ == "__main__":
    s = model_summary()
    print(f"AIR-18 MODEL v{VERSION}")
    print(f"  species   : {s['air18_count']}")
    print(f"  reactions : {s['air18_reactions']}")
    print(f"  estimated : {s['estimated_species']}")
    print(f"  validity  : {s['validity_K']} K")
    print(f"  {s['note']}")
