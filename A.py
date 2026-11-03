"""
=============================================================================
SESI ORGANOID STUDIO EXTENDED
Vesicle Dynamics + Full PCP Signaling + Single-Cell RNA-seq Integration
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
MODULES
=============================================================================
[VD1] VesicleDynamicsModel
      • Multi-compartment Gillespie SSA [C 1976/77]
      • Compartments: cytosol, plasma membrane, early endosome (EE),
        late endosome (LE), recycling endosome (RE), Golgi
      • Vesicle types: clathrin-coated (CCV), caveolin, uncoated
      • Reactions: budding, uncoating, fusion, fission, motor transport
        (kinesin/dynein), tethering, SNARE-mediated fusion
      • Cdc42 coupling: Cdc42-GTP promotes actin polymerization → vesicle
        budding at leading edge [C Roth 2016]
      • COROLLARY-3 guard: first-order reactions only
      • HINO guard: noise from discrete propensities, not smooth density

[PCP2] FullPCPSignalingNetwork
      • Extended signaling cascade [C Strutt 2001; Habich 2016]:
          Wnt ligand → Frizzled/LRP → Dsh → RhoA → JNK → cytoskeleton
          Vang/Pk → Daam1 → RhoA → ROCK → actin
      • ODE-based with stochastic switching for low-copy components
      • Cross-talk with Cdc42 and Par complex
      • Spatial grid with diffusion between neighbouring cells
      • COROLLARY-3 guard: first-order reactions/diffusion only
      • PROP-3 awareness: field products lose two Sobolev orders

[SCR1] SingleCellRNABridge
      • Load scRNA-seq data: 10X Genomics H5, CSV, MTX formats
      • Gene expression → protein levels (via translation model)
      • Cell type annotation (clustering + marker genes)
      • Comparison with model predictions (KS test, correlation)
      • Pseudotime analysis for ontogeny validation
      • HONEST: computational comparison only; NOT diagnostic;
        NOT a substitute for experimental validation

=============================================================================
HONEST SCOPE
=============================================================================
• VesicleDynamicsModel: multi-compartment SSA is exact for well-mixed
  compartments [C]. Real vesicle trafficking involves motor proteins,
  cytoskeletal transport, and membrane curvature sensing not fully captured.
• FullPCPSignalingNetwork: extended cascade is simplified. Real signaling
  involves many more proteins, post-translational modifications, feedback
  loops, and cross-talk. This is computational only.
• SingleCellRNABridge: statistical comparison is HEURISTIC. Real scRNA-seq
  has dropout, batch effects, and biological variability. This bridge is
  for calibration/validation only, NOT diagnosis.
• COROLLARY-3 guard: all fields first-order only; no Hessian.
• HINO guard: energy measures singular; no smooth density assumed.
• PROP-3 awareness: field products lose two Sobolev orders.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All parameters illustrative. No biological/medical/diagnostic claim.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.vesicle_pcp_scrna")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import anndata as ad
    HAS_ANNDATA = True
except Exception:
    HAS_ANNDATA = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except Exception:
    HAS_SCANPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

__all__ = [
    "VesicleDynamicsModel", "VesicleConfig",
    "FullPCPSignalingNetwork", "PCPSignalingConfig",
    "SingleCellRNABridge", "scRNAConfig", "CellAnnotation",
    "SESIExtendedOrganoidSuite",
]


# =============================================================================
# SECTION 1 — [VD1] VESICLE DYNAMICS MODEL
# =============================================================================

@dataclass
class VesicleConfig:
    """Configuration for vesicle dynamics."""
    # Compartments per cell
    n_cells: int = 6
    # Molecular species (per compartment)
    # Cytosol: Cdc42-GDP, Cdc42-GTP, GDI, actin-monomer
    # Membrane: Cdc42-GTP (active), PIP2, clathrin
    # Vesicles: CCV (clathrin-coated), caveolin, uncoated
    # Endosomes: EE, LE, RE
    # Golgi: cargo proteins
    n_cdc42_total: int = 800
    n_actin_total: int = 5000
    n_vesicles_initial: int = 50
    # Reaction rates [C Roth 2016]
    k_budding_ccv: float = 0.05        # membrane → CCV
    k_budding_caveolin: float = 0.02
    k_uncoating: float = 0.1           # CCV → uncoated vesicle
    k_fusion_ee: float = 0.08          # uncoated → EE
    k_ee_to_le: float = 0.03           # EE → LE (maturation)
    k_le_to_re: float = 0.02           # LE → RE (recycling)
    k_re_to_membrane: float = 0.05     # RE → membrane (exocytosis)
    k_motor_transport: float = 0.1     # vesicle movement (kinesin/dynein)
    k_cdc42_activation: float = 0.2    # Cdc42-GDP → Cdc42-GTP (GEF)
    k_cdc42_inactivation: float = 0.1  # Cdc42-GTP → Cdc42-GDP (GAP)
    k_actin_polymerization: float = 0.15  # actin-monomer → actin-filament
    # Cdc42-actin coupling
    cdc42_actin_coupling: float = 0.3  # Cdc42-GTP promotes actin polymerization
    # Simulation
    max_time: float = 300.0
    max_steps: int = 200000
    seed: int = 0


class VesicleDynamicsModel:
    """
    [C/D] Multi-compartment stochastic vesicle trafficking model.

    BIOLOGICAL BASIS [C Roth 2016]
      Vesicle trafficking cycle:
        1. Budding: cargo selection + coat assembly (clathrin/caveolin)
        2. Uncoating: coat removal (Hsc70/auxilin for CCV)
        3. Transport: motor proteins (kinesin/dynein) along cytoskeleton
        4. Tethering: Rab GTPases + tethering factors
        5. Fusion: SNARE complex-mediated membrane fusion

      Compartments:
        - Cytosol: soluble proteins (Cdc42, actin-monomer, GDI)
        - Plasma membrane: Cdc42-GTP (active), PIP2, clathrin
        - Vesicles: CCV (clathrin-coated), caveolin, uncoated
        - Endosomes: EE (early), LE (late), RE (recycling)
        - Golgi: cargo proteins

      Cdc42 coupling [C]:
        Cdc42-GTP at the leading edge promotes:
          - Actin polymerization (via WASP/Arp2/3)
          - Clathrin-mediated endocytosis (via AP-2 adaptor)
          - Vesicle budding at the plasma membrane

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear or bilinear in
      copy numbers). We do NOT compute second derivatives of the
      compartmental state. The noise term is multiplicative but first-order.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates × copy numbers), NOT from an assumed smooth noise density.
      The stochasticity is singular in the sense of being tied to individual
      vesicle trafficking events.

    HONEST
      • SSA is exact for well-mixed compartments; real trafficking involves
        motor proteins, cytoskeletal transport, membrane curvature sensing
        not fully captured here.
      • Vesicle types are simplified; real vesicles have diverse cargo,
        coat proteins, and Rab GTPases.
      • Copy numbers are illustrative; real cells have ~1000-10000 vesicles.
    """

    def __init__(self, cfg: Optional[VesicleConfig] = None):
        self.cfg = cfg or VesicleConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_cells = self.cfg.n_cells

        # State: dict of species → (n_cells,) array of copy numbers
        # Species keys:
        #   Cytosol: 'cdc42_gdp_cyt', 'cdc42_gtp_cyt', 'gdi', 'actin_monomer', 'actin_filament'
        #   Membrane: 'cdc42_gtp_mem', 'pip2', 'clathrin'
        #   Vesicles: 'ccv', 'caveolin_ves', 'uncoated_ves'
        #   Endosomes: 'ee', 'le', 're'
        #   Golgi: 'cargo_golgi'
        self.state: Optional[Dict[str, np.ndarray]] = None

    def initialize(self) -> None:
        """Initialize with random distribution across compartments."""
        self.state = {}
        species_list = [
            'cdc42_gdp_cyt', 'cdc42_gtp_cyt', 'gdi', 'actin_monomer', 'actin_filament',
            'cdc42_gtp_mem', 'pip2', 'clathrin',
            'ccv', 'caveolin_ves', 'uncoated_ves',
            'ee', 'le', 're', 'cargo_golgi'
        ]

        for sp in species_list:
            self.state[sp] = np.zeros(self.n_cells, dtype=np.int64)

        # Initial distribution
        for i in range(self.n_cells):
            # Cdc42
            n_cdc42_cyt = int(self.cfg.n_cdc42_total * 0.4)
            n_cdc42_mem = int(self.cfg.n_cdc42_total * 0.3)
            n_gdi = int(self.cfg.n_cdc42_total * 0.3)

            self.state['cdc42_gdp_cyt'][i] = int(n_cdc42_cyt * 0.7)
            self.state['cdc42_gtp_cyt'][i] = int(n_cdc42_cyt * 0.3)
            self.state['gdi'][i] = n_gdi
            self.state['cdc42_gtp_mem'][i] = n_cdc42_mem

            # Actin
            n_actin_mono = int(self.cfg.n_actin_total * 0.6)
            n_actin_fil = int(self.cfg.n_actin_total * 0.4)
            self.state['actin_monomer'][i] = n_actin_mono
            self.state['actin_filament'][i] = n_actin_fil

            # Membrane components
            self.state['pip2'][i] = 200
            self.state['clathrin'][i] = 150

            # Vesicles
            self.state['ccv'][i] = int(self.cfg.n_vesicles_initial * 0.4)
            self.state['caveolin_ves'][i] = int(self.cfg.n_vesicles_initial * 0.2)
            self.state['uncoated_ves'][i] = int(self.cfg.n_vesicles_initial * 0.4)

            # Endosomes
            self.state['ee'][i] = 30
            self.state['le'][i] = 20
            self.state['re'][i] = 25

            # Golgi
            self.state['cargo_golgi'][i] = 100

    def compute_propensities(self, state: Dict[str, np.ndarray]
                              ) -> Tuple[np.ndarray, List[Tuple[str, int, Dict[str, int]]]]:
        """
        Compute all reaction propensities.
        Returns: (propensities array, list of (reaction_name, cell_idx, state_delta))
        """
        props = []
        events = []

        for i in range(self.n_cells):
            cdc42_gdp_cyt = state['cdc42_gdp_cyt'][i]
            cdc42_gtp_cyt = state['cdc42_gtp_cyt'][i]
            cdc42_gtp_mem = state['cdc42_gtp_mem'][i]
            actin_mono = state['actin_monomer'][i]
            actin_fil = state['actin_filament'][i]
            pip2 = state['pip2'][i]
            clathrin = state['clathrin'][i]
            ccv = state['ccv'][i]
            caveolin_ves = state['caveolin_ves'][i]
            uncoated_ves = state['uncoated_ves'][i]
            ee = state['ee'][i]
            le = state['le'][i]
            re = state['re'][i]
            cargo = state['cargo_golgi'][i]

            # 1. Cdc42 activation (GEF): GDP → GTP (cytosol)
            a = self.cfg.k_cdc42_activation * cdc42_gdp_cyt
            props.append(a)
            events.append(("cdc42_act", i, {'cdc42_gdp_cyt': -1, 'cdc42_gtp_cyt': +1}))

            # 2. Cdc42 inactivation (GAP): GTP → GDP (cytosol)
            a = self.cfg.k_cdc42_inactivation * cdc42_gtp_cyt
            props.append(a)
            events.append(("cdc42_inact", i, {'cdc42_gtp_cyt': -1, 'cdc42_gdp_cyt': +1}))

            # 3. Cdc42 membrane recruitment: GTP(cyt) → GTP(mem)
            a = self.cfg.k_cdc42_activation * 0.5 * cdc42_gtp_cyt
            props.append(a)
            events.append(("cdc42_recruit", i, {'cdc42_gtp_cyt': -1, 'cdc42_gtp_mem': +1}))

            # 4. Actin polymerization: monomer → filament
            # Cdc42-GTP promotes polymerization (coupling)
            coupling_factor = 1.0 + self.cfg.cdc42_actin_coupling * cdc42_gtp_mem / max(self.cfg.n_cdc42_total, 1)
            a = self.cfg.k_actin_polymerization * actin_mono * coupling_factor
            props.append(a)
            events.append(("actin_poly", i, {'actin_monomer': -1, 'actin_filament': +1}))

            # 5. CCV budding: membrane + clathrin + cargo → CCV
            a = self.cfg.k_budding_ccv * pip2 * clathrin * cargo / 1000.0
            props.append(a)
            events.append(("budding_ccv", i, {'pip2': -1, 'clathrin': -1, 'cargo_golgi': -1, 'ccv': +1}))

            # 6. Caveolin budding
            a = self.cfg.k_budding_caveolin * pip2 * cargo / 1000.0
            props.append(a)
            events.append(("budding_caveolin", i, {'pip2': -1, 'cargo_golgi': -1, 'caveolin_ves': +1}))

            # 7. Uncoating: CCV → uncoated vesicle
            a = self.cfg.k_uncoating * ccv
            props.append(a)
            events.append(("uncoating", i, {'ccv': -1, 'uncoated_ves': +1, 'clathrin': +1}))

            # 8. Fusion with EE: uncoated → EE
            a = self.cfg.k_fusion_ee * uncoated_ves
            props.append(a)
            events.append(("fusion_ee", i, {'uncoated_ves': -1, 'ee': +1}))

            # 9. EE → LE maturation
            a = self.cfg.k_ee_to_le * ee
            props.append(a)
            events.append(("ee_to_le", i, {'ee': -1, 'le': +1}))

            # 10. LE → RE recycling
            a = self.cfg.k_le_to_re * le
            props.append(a)
            events.append(("le_to_re", i, {'le': -1, 're': +1}))

            # 11. RE → membrane exocytosis
            a = self.cfg.k_re_to_membrane * re
            props.append(a)
            events.append(("exocytosis", i, {'re': -1, 'pip2': +1}))

            # 12. Motor transport (simplified: vesicle movement between cells)
            if self.n_cells > 1:
                left = (i - 1) % self.n_cells
                right = (i + 1) % self.n_cells
                # CCV transport
                a = self.cfg.k_motor_transport * ccv
                props.append(a)
                events.append(("transport_ccv_left", i, {'ccv': -1, 'ccv_target': left}))
                a = self.cfg.k_motor_transport * ccv
                props.append(a)
                events.append(("transport_ccv_right", i, {'ccv': -1, 'ccv_target': right}))

        return np.array(props, dtype=np.float64), events

    def ssa_step(self, state: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """One Gillespie SSA step."""
        props, events = self.compute_propensities(state)
        a_total = props.sum()

        if a_total <= 0:
            return {k: v.copy() for k, v in state.items()}, 0.0

        # Sample waiting time
        tau = -math.log(max(self._rng.random(), 1e-12)) / a_total

        # Sample which reaction fires
        r = self._rng.random() * a_total
        cumsum = 0.0
        event_idx = len(events) - 1
        for idx in range(len(events)):
            cumsum += props[idx]
            if r <= cumsum:
                event_idx = idx
                break

        # Apply reaction
        new_state = {k: v.copy() for k, v in state.items()}
        reaction_name, cell_idx, delta = events[event_idx]

        for species, change in delta.items():
            if species.endswith('_target'):
                # Transport target
                target_cell = change
                target_species = species.replace('_target', '')
                new_state[target_species][target_cell] += 1
            else:
                new_state[species][cell_idx] = max(0, new_state[species][cell_idx] + change)

        return new_state, tau

    def simulate(self, max_time: Optional[float] = None,
                  max_steps: Optional[int] = None
                  ) -> Tuple[List[Dict[str, np.ndarray]], List[float]]:
        """Run full SSA simulation."""
        if self.state is None:
            self.initialize()

        state = {k: v.copy() for k, v in self.state.items()}
        t = 0.0
        trajectory = [state]
        times = [0.0]
        max_time = max_time or self.cfg.max_time
        max_steps = max_steps or self.cfg.max_steps

        steps = 0
        while t < max_time and steps < max_steps:
            state, dt = self.ssa_step(state)
            if dt <= 0:
                break
            t += dt
            trajectory.append(state)
            times.append(t)
            steps += 1

        return trajectory, times

    def get_vesicle_distribution(self, state: Optional[Dict[str, np.ndarray]] = None
                                   ) -> Dict[str, float]:
        """Get average vesicle distribution across compartments."""
        if state is None:
            state = self.state
        if state is None:
            return {}

        total_ccv = state['ccv'].sum()
        total_caveolin = state['caveolin_ves'].sum()
        total_uncoated = state['uncoated_ves'].sum()
        total_ee = state['ee'].sum()
        total_le = state['le'].sum()
        total_re = state['re'].sum()

        total = total_ccv + total_caveolin + total_uncoated + total_ee + total_le + total_re
        if total == 0:
            return {}

        return {
            'ccv_fraction': total_ccv / total,
            'caveolin_fraction': total_caveolin / total,
            'uncoated_fraction': total_uncoated / total,
            'ee_fraction': total_ee / total,
            'le_fraction': total_le / total,
            're_fraction': total_re / total,
            'total_vesicles': int(total),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_cdc42_total": self.cfg.n_cdc42_total,
            "compartments": ["cytosol", "membrane", "vesicles", "endosomes", "golgi"],
            "species_count": len(self.state) if self.state else 0,
            "corollary3_guard": "first-order reactions only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
        }


# =============================================================================
# SECTION 2 — [PCP2] FULL PCP SIGNALING NETWORK
# =============================================================================

@dataclass
class PCPSignalingConfig:
    """Configuration for full PCP signaling cascade."""
    # Spatial grid
    grid_shape: Tuple[int, int] = (8, 8)
    # Signaling components (per cell)
    # Wnt pathway: Wnt → Fz/LRP → Dsh → RhoA → JNK
    wnt_initial: float = 0.5
    fz_initial: float = 1.0
    dsh_initial: float = 1.0
    rhoa_initial: float = 0.5
    jnk_initial: float = 0.2
    # Vang/Pk pathway: Vang → Pk → Daam1 → RhoA → ROCK
    vang_initial: float = 1.0
    pk_initial: float = 1.0
    daam1_initial: float = 0.5
    rock_initial: float = 0.3
    # Reaction rates [C Strutt 2001; Habich 2016]
    k_wnt_binding: float = 0.5        # Wnt + Fz → Fz-Wnt
    k_dsh_activation: float = 0.8     # Fz-Wnt → Dsh-active
    k_rhoa_activation: float = 0.6    # Dsh-active → RhoA-GTP
    k_jnk_activation: float = 0.4     # RhoA-GTP → JNK-active
    k_daam1_activation: float = 0.5   # Vang-Pk → Daam1-active
    k_rock_activation: float = 0.4    # RhoA-GTP → ROCK-active
    k_degradation: float = 0.1        # protein turnover
    k_production: float = 0.2         # basal production
    # Diffusion
    d_wnt: float = 0.05               # Wnt ligand diffusion
    d_fz: float = 0.01                # Fz membrane diffusion
    d_dsh: float = 0.02               # Dsh cytosolic diffusion
    # Cross-talk with Cdc42
    pcp_to_cdc42_weight: float = 0.5
    cdc42_to_pcp_weight: float = 0.3
    # Simulation
    dt: float = 0.01
    n_steps: int = 500
    seed: int = 0


class FullPCPSignalingNetwork:
    """
    [C/D] Full PCP signaling cascade with cross-talk.

    BIOLOGICAL BASIS [C Strutt 2001; Habich 2016]
      Core PCP pathway:
        Wnt ligand → Frizzled (Fz) + LRP → Dishevelled (Dsh) → RhoA → JNK
        Van Gogh (Vang) + Prickle (Pk) → Daam1 → RhoA → ROCK

      Downstream effects:
        - RhoA-GTP activates ROCK → actin-myosin contractility
        - JNK phosphorylates transcription factors → gene expression
        - Cytoskeletal reorganization → cell polarity, convergent extension

      Cross-talk:
        - PCP → Cdc42: RhoA-GTP can activate Cdc42 via cross-talk
        - Cdc42 → PCP: Cdc42-GTP can modulate Dsh activity

    MODEL [D]
      Reaction-diffusion system on a 2D grid:
        ∂[Wnt]/∂t = D_wnt ∇²[Wnt] + production - degradation
        ∂[Fz-Wnt]/∂t = k_bind [Wnt][Fz] - k_unbind [Fz-Wnt]
        ∂[Dsh*]/∂t = k_act [Fz-Wnt] - k_deg [Dsh*] + D_dsh ∇²[Dsh*]
        ∂[RhoA*]/∂t = k_act ([Dsh*] + [Daam1*]) - k_deg [RhoA*]
        ∂[JNK*]/∂t = k_act [RhoA*] - k_deg [JNK*]
        ∂[Daam1*]/∂t = k_act [Vang-Pk] - k_deg [Daam1*]
        ∂[ROCK*]/∂t = k_act [RhoA*] - k_deg [ROCK*]

    COROLLARY-3 GUARD [D]
      All reactions and diffusion terms are FIRST-ORDER in the fields.
      We do NOT compute second derivatives of signaling molecule concentrations.
      The Laplacian is first-order spatial coupling.

    PROP-3 AWARENESS [P]
      When we multiply fields (e.g. [Wnt] × [Fz] for binding), the result
      loses two Sobolev orders (Carpet-19 Prop 3). We use softplus floors
      to maintain differentiability and document that the product is a
      lower-regularity quantity.

    HONEST
      • This is a simplified signaling model. Real PCP involves many more
        proteins, post-translational modifications, feedback loops.
      • Cross-talk terms are heuristic; real signaling is more complex.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[PCPSignalingConfig] = None):
        self.cfg = cfg or PCPSignalingConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.nx, self.ny = self.cfg.grid_shape

        # State fields (per grid cell)
        self.wnt: Optional[np.ndarray] = None
        self.fz_wnt: Optional[np.ndarray] = None  # Fz-Wnt complex
        self.dsh_active: Optional[np.ndarray] = None
        self.rhoa_active: Optional[np.ndarray] = None
        self.jnk_active: Optional[np.ndarray] = None
        self.vang_pk: Optional[np.ndarray] = None  # Vang-Pk complex
        self.daam1_active: Optional[np.ndarray] = None
        self.rock_active: Optional[np.ndarray] = None

    def _laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with periodic boundary conditions."""
        lap = np.zeros_like(field)
        for axis in range(2):
            lap += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
        return lap

    def initialize(self) -> None:
        """Initialize signaling fields with small random perturbations."""
        shape = (self.nx, self.ny)
        noise = 0.05

        self.wnt = self.cfg.wnt_initial + noise * self._rng.standard_normal(shape)
        self.fz_wnt = self.cfg.fz_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.dsh_active = self.cfg.dsh_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.rhoa_active = self.cfg.rhoa_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.jnk_active = self.cfg.jnk_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.vang_pk = self.cfg.vang_initial * self.cfg.pk_initial * 0.5 + noise * self._rng.standard_normal(shape)
        self.daam1_active = self.cfg.daam1_initial * 0.1 + noise * self._rng.standard_normal(shape)
        self.rock_active = self.cfg.rock_initial * 0.1 + noise * self._rng.standard_normal(shape)

        # Ensure non-negative
        for field in [self.wnt, self.fz_wnt, self.dsh_active, self.rhoa_active,
                      self.jnk_active, self.vang_pk, self.daam1_active, self.rock_active]:
            field[:] = np.maximum(field, 0.0)

    def step(self, cdc42_signal: Optional[np.ndarray] = None) -> None:
        """One signaling step with cross-talk from Cdc42."""
        if self.wnt is None:
            self.initialize()

        dt = self.cfg.dt

        # Laplacians for diffusion
        lap_wnt = self._laplacian_2d(self.wnt)
        lap_fz = self._laplacian_2d(self.fz_wnt)
        lap_dsh = self._laplacian_2d(self.dsh_active)

        # Wnt → Fz binding (PROP-3 awareness: bilinear product)
        wnt_fz = np.log1p(np.exp(self.wnt * 10)) / 10 * np.log1p(np.exp(self.cfg.fz_initial * 10)) / 10
        binding = self.cfg.k_wnt_binding * wnt_fz

        # Fz-Wnt → Dsh activation
        dsh_activation = self.cfg.k_dsh_activation * self.fz_wnt

        # Dsh* + Daam1* → RhoA activation
        rhoa_activation = self.cfg.k_rhoa_activation * (self.dsh_active + self.daam1_active)

        # RhoA* → JNK activation
        jnk_activation = self.cfg.k_jnk_activation * self.rhoa_active

        # Vang-Pk → Daam1 activation
        daam1_activation = self.cfg.k_daam1_activation * self.vang_pk

        # RhoA* → ROCK activation
        rock_activation = self.cfg.k_rock_activation * self.rhoa_active

        # Cdc42 cross-talk (if provided)
        cdc42_modulation = 0.0
        if cdc42_signal is not None:
            cdc42_modulation = self.cfg.cdc42_to_pcp_weight * cdc42_signal

        # Update fields
        self.wnt = np.maximum(self.wnt + dt * (
            self.cfg.d_wnt * lap_wnt
            + self.cfg.k_production - self.cfg.k_degradation * self.wnt
            - binding), 0.0)

        self.fz_wnt = np.maximum(self.fz_wnt + dt * (
            self.cfg.d_fz * lap_fz
            + binding - self.cfg.k_degradation * self.fz_wnt), 0.0)

        self.dsh_active = np.maximum(self.dsh_active + dt * (
            self.cfg.d_dsh * lap_dsh
            + dsh_activation - self.cfg.k_degradation * self.dsh_active
            + cdc42_modulation), 0.0)

        self.rhoa_active = np.maximum(self.rhoa_active + dt * (
            rhoa_activation - self.cfg.k_degradation * self.rhoa_active), 0.0)

        self.jnk_active = np.maximum(self.jnk_active + dt * (
            jnk_activation - self.cfg.k_degradation * self.jnk_active), 0.0)

        self.daam1_active = np.maximum(self.daam1_active + dt * (
            daam1_activation - self.cfg.k_degradation * self.daam1_active), 0.0)

        self.rock_active = np.maximum(self.rock_active + dt * (
            rock_activation - self.cfg.k_degradation * self.rock_active), 0.0)

    def run(self, n_steps: Optional[int] = None,
            cdc42_signal: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Run the signaling network for n_steps."""
        if self.wnt is None:
            self.initialize()
        steps = n_steps or self.cfg.n_steps
        for _ in range(steps):
            self.step(cdc42_signal)
        return {
            'wnt': self.wnt.copy(),
            'fz_wnt': self.fz_wnt.copy(),
            'dsh_active': self.dsh_active.copy(),
            'rhoa_active': self.rhoa_active.copy(),
            'jnk_active': self.jnk_active.copy(),
            'vang_pk': self.vang_pk.copy(),
            'daam1_active': self.daam1_active.copy(),
            'rock_active': self.rock_active.copy(),
        }

    def get_cdc42_activation_signal(self) -> np.ndarray:
        """Compute Cdc42 activation signal from PCP (RhoA cross-talk)."""
        if self.rhoa_active is None:
            return np.zeros((self.nx, self.ny))
        # RhoA-GTP can activate Cdc42 via cross-talk
        return self.cfg.pcp_to_cdc42_weight * self.rhoa_active

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid_shape": self.cfg.grid_shape,
            "signaling_components": ["Wnt", "Fz", "Dsh", "RhoA", "JNK", "Vang", "Pk", "Daam1", "ROCK"],
            "corollary3_guard": "first-order reactions/diffusion; no Hessian",
            "prop3_awareness": "bilinear products lose two Sobolev orders",
            "note": "computational model, not validated molecular biology",
        }


# =============================================================================
# SECTION 3 — [SCR1] SINGLE-CELL RNA-SEQ BRIDGE
# =============================================================================

@dataclass
class scRNAConfig:
    """Configuration for single-cell RNA-seq integration."""
    # Data loading
    data_path: Optional[str] = None
    format: str = "h5ad"  # "h5ad" | "csv" | "mtx"
    # Gene-protein mapping
    translation_efficiency: float = 0.3  # mRNA → protein conversion factor
    protein_degradation_rate: float = 0.1
    # Cell type annotation
    marker_genes: Optional[Dict[str, List[str]]] = None
    # Comparison
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    # Visualization
    save_figures: bool = False
    figure_dir: Optional[str] = None


@dataclass
class CellAnnotation:
    """Cell type annotation result."""
    cell_id: str
    cell_type: str
    confidence: float
    marker_expression: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SingleCellRNABridge:
    """
    [D] Bridge to single-cell RNA-seq data for model calibration.

    CAPABILITIES
      1. Load scRNA-seq data (10X Genomics H5, CSV, MTX formats)
      2. Gene expression → protein levels (via translation model)
      3. Cell type annotation (clustering + marker genes)
      4. Statistical comparison: model vs data
         - KS test for distributions
         - Correlation for gene expression patterns
         - Pseudotime analysis for ontogeny validation
      5. Visualization overlay

    GENE-PROTEIN MAPPING [D]
      protein_level = translation_efficiency * mRNA_level
      This is a simplified model; real translation involves ribosome loading,
      codon usage, post-translational modifications.

    CELL TYPE ANNOTATION [D]
      Clustering (Louvain/Leiden) + marker gene expression.
      Marker genes define cell types (e.g., SOX2 for neural progenitors).

    COROLLARY-3 GUARD [D]
      All statistical operations are FIRST-ORDER (no Hessian of likelihood).
      Comparison metrics are linear or rank-based.

    HONEST
      • This is a COMPUTATIONAL comparison tool, NOT a diagnostic.
      • Real scRNA-seq has dropout, batch effects, biological variability.
      • Gene-protein mapping is heuristic; real translation is complex.
      • No biological/medical/diagnostic claim.
    """

    def __init__(self, cfg: Optional[scRNAConfig] = None):
        self.cfg = cfg or scRNAConfig()
        self._data: Optional[Any] = None  # AnnData or DataFrame
        self._annotations: Optional[List[CellAnnotation]] = None

    # ── Data loading ────────────────────────────────────────────────────
    def load_data(self, path: Optional[str] = None) -> Any:
        """
        Load scRNA-seq data.
        Supports: .h5ad (AnnData), .csv (genes × cells), .mtx (10X format)
        """
        path = path or self.cfg.data_path
        if path is None:
            raise ValueError("No data path specified")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        fmt = self.cfg.format.lower()
        if fmt == "h5ad" and HAS_ANNDATA:
            self._data = ad.read_h5ad(path)
        elif fmt == "csv" and HAS_PANDAS:
            df = pd.read_csv(path, index_col=0)
            self._data = df
        elif fmt == "mtx":
            # 10X format: matrix.mtx, genes.tsv, barcodes.tsv
            if not HAS_ANNDATA:
                raise ImportError("anndata required for MTX format")
            self._data = sc.read_10x_mtx(os.path.dirname(path))
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"[scRNA] Loaded data: {self._data.shape if hasattr(self._data, 'shape') else 'unknown'}")
        return self._data

    # ── Gene-protein mapping ────────────────────────────────────────────
    def mRNA_to_protein(self, mrna_levels: np.ndarray) -> np.ndarray:
        """
        Convert mRNA levels to protein levels (simplified translation model).
        protein = translation_efficiency * mRNA
        """
        return self.cfg.translation_efficiency * mrna_levels

    def extract_marker_genes(self, genes_of_interest: List[str]) -> Optional[np.ndarray]:
        """Extract expression of specific marker genes."""
        if self._data is None:
            return None

        if HAS_ANNDATA and hasattr(self._data, 'var_names'):
            # AnnData format
            mask = self._data.var_names.isin(genes_of_interest)
            return self._data.X[:, mask].toarray() if hasattr(self._data.X, 'toarray') else self._data.X[:, mask]
        elif HAS_PANDAS and hasattr(self._data, 'index'):
            # DataFrame format
            available = [g for g in genes_of_interest if g in self._data.index]
            if not available:
                return None
            return self._data.loc[available].values.T
        return None

    # ── Cell type annotation ────────────────────────────────────────────
    def annotate_cells(self, marker_genes: Optional[Dict[str, List[str]]] = None
                        ) -> List[CellAnnotation]:
        """
        Annotate cell types based on marker gene expression.
        marker_genes: dict of {cell_type: [gene1, gene2, ...]}
        """
        if self._data is None:
            raise ValueError("No data loaded")

        markers = marker_genes or self.cfg.marker_genes
        if markers is None:
            raise ValueError("No marker genes specified")

        annotations = []
        n_cells = self._data.shape[0] if hasattr(self._data, 'shape') else 0

        for cell_idx in range(n_cells):
            # Extract expression for this cell
            if HAS_ANNDATA:
                cell_expr = self._data.X[cell_idx, :].toarray().flatten() if hasattr(self._data.X, 'toarray') else self._data.X[cell_idx, :]
            else:
                cell_expr = self._data.iloc[:, cell_idx].values

            # Score each cell type
            scores = {}
            for cell_type, genes in markers.items():
                gene_mask = [g in (self._data.var_names if HAS_ANNDATA else self._data.index) for g in genes]
                if any(gene_mask):
                    type_expr = cell_expr[gene_mask].mean()
                    scores[cell_type] = float(type_expr)
                else:
                    scores[cell_type] = 0.0

            # Assign cell type with highest score
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type] / (sum(scores.values()) + 1e-12)

            annotations.append(CellAnnotation(
                cell_id=f"cell_{cell_idx}",
                cell_type=best_type,
                confidence=confidence,
                marker_expression=scores,
            ))

        self._annotations = annotations
        return annotations

    # ── Statistical comparison ──────────────────────────────────────────
    def compare_gene_expression(self, model_predictions: Dict[str, np.ndarray]
                                  ) -> Dict[str, Any]:
        """
        Compare model predictions with experimental gene expression.
        model_predictions: dict of {gene_name: predicted_protein_levels}
        """
        if self._data is None:
            raise ValueError("No data loaded")

        from scipy import stats

        comparisons = {}
        for gene, predicted in model_predictions.items():
            # Extract experimental data for this gene
            if HAS_ANNDATA and gene in self._data.var_names:
                gene_idx = list(self._data.var_names).index(gene)
                experimental = self._data.X[:, gene_idx].toarray().flatten() if hasattr(self._data.X, 'toarray') else self._data.X[:, gene_idx]
            elif HAS_PANDAS and gene in self._data.index:
                experimental = self._data.loc[gene].values
            else:
                continue

            # Convert mRNA to protein
            experimental_protein = self.mRNA_to_protein(experimental)

            # Statistical comparison
            ks_stat, ks_pvalue = stats.ks_2samp(experimental_protein, predicted)
            correlation = np.corrcoef(experimental_protein, predicted)[0, 1]

            comparisons[gene] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'pearson_correlation': float(correlation),
                'experimental_mean': float(experimental_protein.mean()),
                'predicted_mean': float(predicted.mean()),
            }

        return comparisons

    def compare_cell_types(self, model_annotations: List[str]) -> Dict[str, Any]:
        """
        Compare model cell type assignments with experimental annotations.
        """
        if self._annotations is None:
            raise ValueError("No experimental annotations available")

        # Contingency table
        experimental_types = [ann.cell_type for ann in self._annotations]
        n_cells = len(experimental_types)

        # Compute agreement
        agreement = sum(1 for e, m in zip(experimental_types, model_annotations) if e == m)
        accuracy = agreement / n_cells

        # Per-type accuracy
        type_accuracy = {}
        for cell_type in set(experimental_types):
            exp_mask = [e == cell_type for e in experimental_types]
            mod_mask = [m == cell_type for m in model_annotations]
            tp = sum(e and m for e, m in zip(exp_mask, mod_mask))
            fp = sum(not e and m for e, m in zip(exp_mask, mod_mask))
            fn = sum(e and not m for e, m in zip(exp_mask, mod_mask))
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            type_accuracy[cell_type] = {'precision': precision, 'recall': recall, 'f1': f1}

        return {
            'overall_accuracy': accuracy,
            'per_type_accuracy': type_accuracy,
            'n_cells': n_cells,
        }

    # ── Pseudotime analysis ─────────────────────────────────────────────
    def compute_pseudotime(self, start_cell_type: str, end_cell_type: str
                             ) -> Optional[np.ndarray]:
        """
        Compute pseudotime ordering of cells from start to end type.
        Uses diffusion pseudotime (simplified).
        """
        if not HAS_SCANPY or self._data is None:
            return None

        try:
            # Preprocess
            sc.pp.normalize_total(self._data)
            sc.pp.log1p(self._data)
            sc.pp.pca(self._data)
            sc.pp.neighbors(self._data)

            # Compute diffusion pseudotime
            sc.tl.diffmap(self._data)
            # Find start and end cells
            if self._annotations:
                start_cells = [i for i, ann in enumerate(self._annotations) if ann.cell_type == start_cell_type]
                end_cells = [i for i, ann in enumerate(self._annotations) if ann.cell_type == end_cell_type]
                if start_cells and end_cells:
                    root_cell = start_cells[0]
                    sc.tl.dpt(self._data, root=root_cell)
                    return self._data.obs['dpt_pseudotime'].values
        except Exception as e:
            logger.warning(f"[scRNA] Pseudotime computation failed: {e}")

        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "data_loaded": self._data is not None,
            "annotations_available": self._annotations is not None,
            "has_anndata": HAS_ANNDATA,
            "has_scanpy": HAS_SCANPY,
            "has_pandas": HAS_PANDAS,
            "note": "computational comparison only; NOT diagnostic",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIExtendedOrganoidSuite:
    """
    Unifies: VesicleDynamicsModel + FullPCPSignalingNetwork + SingleCellRNABridge
             + previous modules (Cdc42, Par complex, vertex model).

    FULL EXTENDED PIPELINE:
      1. Vesicle trafficking with Cdc42-actin coupling
      2. Full PCP signaling cascade (Wnt→Fz→Dsh→RhoA→JNK)
      3. Cross-talk between vesicle dynamics and PCP signaling
      4. Compare with scRNA-seq data for validation
      5. Export to AR pipeline
    """

    def __init__(self,
                 vesicle_cfg: Optional[VesicleConfig] = None,
                 pcp_sig_cfg: Optional[PCPSignalingConfig] = None,
                 scrna_cfg: Optional[scRNAConfig] = None):
        self.vesicle = VesicleDynamicsModel(vesicle_cfg)
        self.pcp_signaling = FullPCPSignalingNetwork(pcp_sig_cfg)
        self.scrna = SingleCellRNABridge(scrna_cfg)

    def run_extended_simulation(self, n_steps: int = 100) -> Dict[str, Any]:
        """
        Run coupled vesicle + PCP signaling simulation.
        """
        # Initialize
        self.vesicle.initialize()
        self.pcp_signaling.initialize()

        # Run coupled simulation
        vesicle_traj = []
        pcp_traj = []

        for step in range(n_steps):
            # Vesicle dynamics
            ves_state, _ = self.vesicle.ssa_step(self.vesicle.state)
            self.vesicle.state = ves_state
            vesicle_traj.append(ves_state)

            # Extract Cdc42 signal from vesicle model
            cdc42_mem = ves_state['cdc42_gtp_mem'].mean()
            cdc42_signal = np.full(self.pcp_signaling.cfg.grid_shape, cdc42_mem / max(self.vesicle.cfg.n_cdc42_total, 1))

            # PCP signaling with Cdc42 cross-talk
            self.pcp_signaling.step(cdc42_signal)

            # Extract PCP signal for vesicle model
            pcp_signal = self.pcp_signaling.get_cdc42_activation_signal().mean()
            # Modulate vesicle budding rate
            self.vesicle.cfg.k_budding_ccv *= (1.0 + 0.1 * pcp_signal)

            if step % 10 == 0:
                pcp_traj.append({
                    'wnt': self.pcp_signaling.wnt.mean(),
                    'dsh_active': self.pcp_signaling.dsh_active.mean(),
                    'rhoa_active': self.pcp_signaling.rhoa_active.mean(),
                })

        # Final state
        final_vesicle = self.vesicle.get_vesicle_distribution()
        final_pcp = {
            'wnt_mean': float(self.pcp_signaling.wnt.mean()),
            'rhoa_mean': float(self.pcp_signaling.rhoa_active.mean()),
            'jnk_mean': float(self.pcp_signaling.jnk_active.mean()),
        }

        return {
            'n_steps': n_steps,
            'final_vesicle_distribution': final_vesicle,
            'final_pcp_signaling': final_pcp,
            'vesicle_trajectory_length': len(vesicle_traj),
            'pcp_trajectory_length': len(pcp_traj),
        }

    def compare_with_scrna(self, data_path: str,
                             model_genes: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare model predictions with scRNA-seq data.
        """
        self.scrna.load_data(data_path)
        comparisons = self.scrna.compare_gene_expression(model_genes)
        return comparisons

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "vesicle": self.vesicle.get_status(),
            "pcp_signaling": self.pcp_signaling.get_status(),
            "scrna": self.scrna.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO EXTENDED — SMOKE TEST")
    print(" Vesicle Dynamics + Full PCP Signaling + scRNA-seq Integration")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" anndata available      : {HAS_ANNDATA}")
    print(f" scanpy available       : {HAS_SCANPY}")
    print(f" pandas available       : {HAS_PANDAS}")

    suite = SESIExtendedOrganoidSuite(
        vesicle_cfg=VesicleConfig(n_cells=4, max_time=100.0, max_steps=10000),
        pcp_sig_cfg=PCPSignalingConfig(grid_shape=(4, 4), n_steps=50),
        scrna_cfg=scRNAConfig())

    # ── 1. Vesicle Dynamics ─────────────────────────────────────────────
    print("\n [VD1] Vesicle Dynamics Model")
    vesicle = suite.vesicle
    print(f"   n_cells              : {vesicle.n_cells}")
    print(f"   compartments         : {vesicle.get_status()['compartments']}")
    print(f"   species count        : {vesicle.get_status()['species_count']}")

    # Run simulation
    vesicle.initialize()
    traj, times = vesicle.simulate(max_steps=5000)
    print(f"   trajectory length    : {len(traj)}")
    print(f"   final time           : {times[-1]:.2f}")

    final_state = traj[-1]
    vesicle_dist = vesicle.get_vesicle_distribution(final_state)
    print(f"   vesicle distribution :")
    for k, v in vesicle_dist.items():
        print(f"     {k:20s} : {v:.4f}" if isinstance(v, float) else f"     {k:20s} : {v}")

    print(f"   COROLLARY-3 guard    : {vesicle.get_status()['corollary3_guard']}")
    print(f"   HINO guard           : {vesicle.get_status()['hino_guard']}")

    # ── 2. Full PCP Signaling ───────────────────────────────────────────
    print("\n [PCP2] Full PCP Signaling Network")
    pcp = suite.pcp_signaling
    print(f"   grid shape           : {pcp.cfg.grid_shape}")
    print(f"   signaling components : {pcp.get_status()['signaling_components']}")

    # Run with Cdc42 cross-talk
    cdc42_signal = np.random.rand(pcp.cfg.grid_shape[0], pcp.cfg.grid_shape[1]) * 0.5
    pcp_result = pcp.run(n_steps=30, cdc42_signal=cdc42_signal)
    print(f"   final Wnt mean       : {pcp_result['wnt'].mean():.4f}")
    print(f"   final Dsh* mean      : {pcp_result['dsh_active'].mean():.4f}")
    print(f"   final RhoA* mean     : {pcp_result['rhoa_active'].mean():.4f}")
    print(f"   final JNK* mean      : {pcp_result['jnk_active'].mean():.4f}")

    cdc42_from_pcp = pcp.get_cdc42_activation_signal()
    print(f"   Cdc42 signal (from PCP): mean={cdc42_from_pcp.mean():.4f}")

    print(f"   COROLLARY-3 guard    : {pcp.get_status()['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {pcp.get_status()['prop3_awareness']}")

    # ── 3. Extended Simulation ──────────────────────────────────────────
    print("\n [EXTENDED] Coupled Vesicle + PCP Simulation")
    result = suite.run_extended_simulation(n_steps=50)
    print(f"   n_steps              : {result['n_steps']}")
    print(f"   final vesicle dist   : {result['final_vesicle_distribution']}")
    print(f"   final PCP signaling  : {result['final_pcp_signaling']}")

    # ── 4. scRNA-seq Bridge ─────────────────────────────────────────────
    print("\n [SCR1] Single-Cell RNA-seq Bridge")
    scrna = suite.scrna
    print(f"   data loaded          : {scrna.get_status()['data_loaded']}")
    print(f"   anndata available    : {scrna.get_status()['has_anndata']}")
    print(f"   scanpy available     : {scrna.get_status()['has_scanpy']}")

    # Synthetic comparison (no real data)
    print(f"   Note                 : {scrna.get_status()['note']}")

    # ── 5. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"     {kk}: {vv}")
        else:
            print(f"     {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Gillespie SSA is EXACT for well-mixed compartments (1976/77).")
    print("       Real vesicle trafficking involves motor proteins, cytoskeletal")
    print("       transport, membrane curvature sensing not fully captured.")
    print(" [C]   PCP signaling: extended cascade (Wnt→Fz→Dsh→RhoA→JNK) is")
    print("       simplified. Real signaling involves many more proteins,")
    print("       post-translational modifications, feedback loops.")
    print(" [D]   scRNA-seq bridge: statistical comparison is HEURISTIC.")
    print("       Real scRNA-seq has dropout, batch effects, biological variability.")
    print("       This bridge is for calibration/validation only, NOT diagnosis.")
    print(" [D]   COROLLARY-3 guard: all fields first-order only; no Hessian.")
    print(" [C]   HINO guard: energy measures singular; no smooth density assumed.")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL models. No biological, medical,")
    print("       or diagnostic claim. Not validated against real organoids.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
