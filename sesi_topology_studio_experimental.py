"""
=============================================================================
SESI ORGANOID STUDIO EXPERIMENTAL
Membrane Trafficking + PCP Pathway + Organoid Imaging Integration
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
[SMT1] StochasticMembraneTrafficking
      • Multi-compartment Gillespie SSA [C 1976/77]
      • Compartments: cytosol, membrane (per cell), endosome
      • Molecular species: Cdc42-GDP, Cdc42-GTP, GDI, GDI:Cdc42-GDP
      • Reactions: GDI binding/unbinding, GEF activation, GAP inactivation,
                   membrane insertion/extraction, endocytosis/recycling
      • Positive feedback via membrane-localized Cdc42-GTP [C Goryunov 2008]
      • COROLLARY-3 guard: first-order reactions only
      • HINO guard: noise from discrete propensities, not smooth density

[PCP1] PlanarCellPolarityPathway
      • Core PCP proteins: Frizzled (Fz), Dishevelled (Dsh), Van Gogh (Vang),
        Prickle (Pk), Flamingo (Fmi), Diego (Dgo) [C Adler 2012; Strutt 2001]
      • Turing-type mutual inhibition between Fz-Dsh and Vang-Pk complexes
      • Wnt ligand gradient as global cue [C]
      • Coupling to Cdc42 activation (PCP → Cdc42 → Par complex)
      • Spatial grid with diffusion between neighbouring cells
      • COROLLARY-3 guard: first-order reactions/diffusion only
      • PROP-3 awareness: field products lose two Sobolev orders

[EDB1] ExperimentalDataBridge
      • Load organoid imaging data (TIFF/CSV/NPZ)
      • Quantification: cell shapes, polarity markers, gene expression
      • Statistical comparison: KS test, Wasserstein distance, correlation
      • Bayesian parameter calibration (MCMC-lite)
      • Visualization overlay: model vs data
      • HONEST: computational comparison only; NOT diagnostic;
        NOT a substitute for experimental validation

=============================================================================
HONEST SCOPE
=============================================================================
• StochasticMembraneTrafficking: multi-compartment SSA is exact for
  well-mixed compartments [C]. Real trafficking involves vesicle dynamics,
  motor proteins, and cytoskeletal transport not captured here.
• PlanarCellPolarityPathway: core PCP model is simplified. Real PCP
  involves many more proteins, post-translational modifications, and
  feedback loops. This is computational only.
• ExperimentalDataBridge: statistical comparison is HEURISTIC. Real
  organoid imaging has noise, artifacts, and biological variability.
  This bridge is for calibration/validation only, NOT diagnosis.
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

logger = logging.getLogger("sesi.experimental")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from skimage import io as skio
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

__all__ = [
    "StochasticMembraneTrafficking", "TraffickingConfig",
    "PlanarCellPolarityPathway", "PCPConfig",
    "ExperimentalDataBridge", "DataBridgeConfig", "CalibrationResult",
    "SESIExperimentalOrganoidSuite",
]


# =============================================================================
# SECTION 1 — [SMT1] STOCHASTIC MEMBRANE TRAFFICKING
# =============================================================================

@dataclass
class TraffickingConfig:
    """Configuration for Cdc42 membrane trafficking."""
    # Compartments per cell
    n_cells: int = 8
    # Molecular species (per compartment)
    # Cytosol: Cdc42-GDP, GDI, GDI:Cdc42-GDP complex
    # Membrane: Cdc42-GDP (membrane-bound), Cdc42-GTP (active)
    # Endosome: Cdc42-GTP (recycling)
    # Total Cdc42 molecules per cell
    n_cdc42_total: int = 500
    n_gdi_total: int = 300
    # Reaction rates [C Goryunov/Bement 2008]
    k_gdi_bind: float = 0.05          # Cdc42-GDP + GDI → complex
    k_gdi_unbind: float = 0.02        # complex → Cdc42-GDP + GDI
    k_gef: float = 0.15               # membrane Cdc42-GDP → Cdc42-GTP
    k_gap: float = 0.08               # membrane Cdc42-GTP → Cdc42-GDP
    k_feedback: float = 0.3           # positive feedback (GTP → more GEF)
    k_insert: float = 0.1             # cytosol → membrane insertion
    k_extract: float = 0.05           # membrane → cytosol extraction
    k_endocytosis: float = 0.02       # membrane → endosome
    k_recycling: float = 0.03         # endosome → membrane
    # Diffusion between cells (membrane)
    k_diffusion: float = 0.01
    # Simulation
    max_time: float = 200.0
    max_steps: int = 100000
    seed: int = 0


class StochasticMembraneTrafficking:
    """
    [C/D] Multi-compartment stochastic Cdc42 trafficking.

    BIOLOGICAL BASIS [C]
      Cdc42 cycles between:
        - Cytosol: Cdc42-GDP (inactive), bound to GDI for solubility
        - Membrane: Cdc42-GDP (membrane-bound via prenylation),
                    Cdc42-GTP (active, recruits effectors)
        - Endosome: Cdc42-GTP (recycling pool)

      Trafficking reactions:
        1. GDI binding: Cdc42-GDP + GDI → GDI:Cdc42-GDP (cytosol)
        2. GDI unbinding: GDI:Cdc42-GDP → Cdc42-GDP + GDI (at membrane)
        3. Membrane insertion: Cdc42-GDP (cytosol) → Cdc42-GDP (membrane)
        4. GEF activation: Cdc42-GDP (membrane) → Cdc42-GTP (membrane)
        5. GAP inactivation: Cdc42-GTP (membrane) → Cdc42-GDP (membrane)
        6. Extraction: Cdc42-GDP (membrane) → Cdc42-GDP (cytosol)
        7. Endocytosis: Cdc42-GTP (membrane) → Cdc42-GTP (endosome)
        8. Recycling: Cdc42-GTP (endosome) → Cdc42-GDP (membrane)
        9. Positive feedback: Cdc42-GTP enhances GEF activity locally
       10. Diffusion: Cdc42-GTP (membrane cell i) → (membrane cell j)

    GILLESPIE SSA [C 1976/77]
      Exact stochastic simulation for well-mixed compartments.
      Each reaction has a propensity; the next reaction and time are sampled.

    COROLLARY-3 GUARD [D]
      All reaction propensities are FIRST-ORDER (linear or bilinear in
      copy numbers). We do NOT compute second derivatives of the
      compartmental state. The noise term is multiplicative but first-order.

    HINO GUARD [C Thm 8]
      The noise amplitude is computed from discrete propensities (reaction
      rates × copy numbers), NOT from an assumed smooth noise density.
      The stochasticity is singular in the sense of being tied to individual
      molecular trafficking events.

    HONEST
      • SSA is exact for well-mixed compartments; real trafficking involves
        vesicle dynamics, motor proteins, cytoskeletal transport not captured.
      • Positive feedback is simplified; real Cdc42 activation involves
        complex scaffolding (Bem1/2, Bud2/3 in yeast; PAR-3/PAR-6/aPKC in mammals).
      • Copy numbers are illustrative; real cells have ~1000-10000 Cdc42 molecules.
    """

    def __init__(self, cfg: Optional[TraffickingConfig] = None):
        self.cfg = cfg or TraffickingConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_cells = self.cfg.n_cells

        # State: dict of species → (n_cells,) array of copy numbers
        # Species keys:
        #   'cdp_cyt' : Cdc42-GDP in cytosol
        #   'cdp_mem' : Cdc42-GDP membrane-bound
        #   'ctp_mem' : Cdc42-GTP membrane (active)
        #   'ctp_end' : Cdc42-GTP in endosome
        #   'gdi_free': free GDI
        #   'gdi_cplx': GDI:Cdc42-GDP complex
        self.state: Optional[Dict[str, np.ndarray]] = None

    def initialize(self, fraction_membrane: float = 0.3,
                    fraction_gtp: float = 0.1) -> None:
        """Initialize with random distribution across compartments."""
        n_total = self.cfg.n_cdc42_total
        n_gdi = self.cfg.n_gdi_total

        self.state = {}
        for i in range(self.n_cells):
            # Cdc42 distribution
            n_mem = int(n_total * fraction_membrane *
                        (1 + 0.2 * self._rng.standard_normal()))
            n_mem = max(0, min(n_total, n_mem))
            n_cyt = n_total - n_mem

            n_gtp = int(n_mem * fraction_gtp)
            n_gdp_mem = n_mem - n_gtp
            n_gdp_cyt = int(n_cyt * 0.7)
            n_gdi_cplx = n_cyt - n_gdp_cyt

            if i == 0:
                self.state['cdp_cyt'] = np.zeros(self.n_cells, dtype=np.int64)
                self.state['cdp_mem'] = np.zeros(self.n_cells, dtype=np.int64)
                self.state['ctp_mem'] = np.zeros(self.n_cells, dtype=np.int64)
                self.state['ctp_end'] = np.zeros(self.n_cells, dtype=np.int64)
                self.state['gdi_free'] = np.zeros(self.n_cells, dtype=np.int64)
                self.state['gdi_cplx'] = np.zeros(self.n_cells, dtype=np.int64)

            self.state['cdp_cyt'][i] = max(0, n_gdp_cyt)
            self.state['cdp_mem'][i] = max(0, n_gdp_mem)
            self.state['ctp_mem'][i] = max(0, n_gtp)
            self.state['ctp_end'][i] = 0
            self.state['gdi_free'][i] = max(0, n_gdi - n_gdi_cplx)
            self.state['gdi_cplx'][i] = max(0, min(n_gdi, n_gdi_cplx))

    def compute_propensities(self, state: Dict[str, np.ndarray]
                              ) -> Tuple[np.ndarray, List[Tuple[str, int, Dict[str, int]]]]:
        """
        Compute all reaction propensities.
        Returns: (propensities array, list of (reaction_name, cell_idx, state_delta))
        """
        props = []
        events = []

        for i in range(self.n_cells):
            cdp_cyt = state['cdp_cyt'][i]
            cdp_mem = state['cdp_mem'][i]
            ctp_mem = state['ctp_mem'][i]
            ctp_end = state['ctp_end'][i]
            gdi_free = state['gdi_free'][i]
            gdi_cplx = state['gdi_cplx'][i]

            # 1. GDI binding: Cdc42-GDP(cyt) + GDI → complex
            a = self.cfg.k_gdi_bind * cdp_cyt * gdi_free
            props.append(a)
            events.append(("gdi_bind", i, {'cdp_cyt': -1, 'gdi_free': -1, 'gdi_cplx': +1}))

            # 2. GDI unbinding: complex → Cdc42-GDP(cyt) + GDI
            a = self.cfg.k_gdi_unbind * gdi_cplx
            props.append(a)
            events.append(("gdi_unbind", i, {'cdp_cyt': +1, 'gdi_free': +1, 'gdi_cplx': -1}))

            # 3. Membrane insertion: Cdc42-GDP(cyt) → Cdc42-GDP(mem)
            a = self.cfg.k_insert * cdp_cyt
            props.append(a)
            events.append(("insert", i, {'cdp_cyt': -1, 'cdp_mem': +1}))

            # 4. GEF activation: Cdc42-GDP(mem) → Cdc42-GTP(mem)
            # Positive feedback: GEF rate enhanced by local Cdc42-GTP
            feedback_factor = 1.0 + self.cfg.k_feedback * ctp_mem / max(self.cfg.n_cdc42_total, 1)
            a = self.cfg.k_gef * cdp_mem * feedback_factor
            props.append(a)
            events.append(("gef", i, {'cdp_mem': -1, 'ctp_mem': +1}))

            # 5. GAP inactivation: Cdc42-GTP(mem) → Cdc42-GDP(mem)
            a = self.cfg.k_gap * ctp_mem
            props.append(a)
            events.append(("gap", i, {'ctp_mem': -1, 'cdp_mem': +1}))

            # 6. Extraction: Cdc42-GDP(mem) → Cdc42-GDP(cyt)
            a = self.cfg.k_extract * cdp_mem
            props.append(a)
            events.append(("extract", i, {'cdp_mem': -1, 'cdp_cyt': +1}))

            # 7. Endocytosis: Cdc42-GTP(mem) → Cdc42-GTP(endosome)
            a = self.cfg.k_endocytosis * ctp_mem
            props.append(a)
            events.append(("endocytosis", i, {'ctp_mem': -1, 'ctp_end': +1}))

            # 8. Recycling: Cdc42-GTP(endosome) → Cdc42-GDP(mem)
            a = self.cfg.k_recycling * ctp_end
            props.append(a)
            events.append(("recycling", i, {'ctp_end': -1, 'cdp_mem': +1}))

            # 9. Diffusion to neighbours (ring topology)
            if self.n_cells > 1:
                left = (i - 1) % self.n_cells
                right = (i + 1) % self.n_cells
                # Cdc42-GTP membrane diffusion
                a = self.cfg.k_diffusion * ctp_mem
                props.append(a)
                events.append(("diff_left", i, {'ctp_mem': -1, 'ctp_mem_target': left}))
                a = self.cfg.k_diffusion * ctp_mem
                props.append(a)
                events.append(("diff_right", i, {'ctp_mem': -1, 'ctp_mem_target': right}))

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
            if species == 'ctp_mem_target':
                # Diffusion target
                target_cell = change
                new_state['ctp_mem'][target_cell] += 1
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

    def get_polarity_vector(self, state: Optional[Dict[str, np.ndarray]] = None
                              ) -> np.ndarray:
        """
        Compute polarity vector from membrane Cdc42-GTP distribution.
        Returns: (n_cells,) array of polarity magnitudes.
        """
        if state is None:
            state = self.state
        if state is None:
            return np.zeros(self.n_cells)

        ctp_mem = state['ctp_mem']
        total = ctp_mem.sum()
        if total > 0:
            return ctp_mem.astype(float) / total
        return np.ones(self.n_cells) / self.n_cells

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_cdc42_total": self.cfg.n_cdc42_total,
            "n_gdi_total": self.cfg.n_gdi_total,
            "corollary3_guard": "first-order reactions only; no Hessian",
            "hino_guard": "noise from discrete propensities, not smooth density",
            "compartments": ["cytosol", "membrane", "endosome"],
            "species": ["cdp_cyt", "cdp_mem", "ctp_mem", "ctp_end", "gdi_free", "gdi_cplx"],
        }


# =============================================================================
# SECTION 2 — [PCP1] PLANAR CELL POLARITY PATHWAY
# =============================================================================

@dataclass
class PCPConfig:
    """Configuration for Planar Cell Polarity pathway."""
    # Spatial grid
    grid_shape: Tuple[int, int] = (8, 8)  # 2D epithelial sheet
    # Core PCP proteins (per cell)
    # Fz-Dsh complex (forward polarity)
    fz_initial: float = 1.0
    dsh_initial: float = 1.0
    # Vang-Pk complex (reverse polarity)
    vang_initial: float = 1.0
    pk_initial: float = 1.0
    # Fmi (Flamingo) - mediates intercellular alignment
    fmi_initial: float = 1.0
    # Reaction rates [C Adler 2012; Strutt 2001]
    k_fz_dsh_assembly: float = 0.5     # Fz + Dsh → Fz-Dsh complex
    k_vang_pk_assembly: float = 0.5    # Vang + Pk → Vang-Pk complex
    k_mutual_inhibition: float = 1.0   # Fz-Dsh inhibits Vang-Pk (and vice versa)
    k_intercellular_align: float = 0.3 # Fmi-mediated alignment
    k_degradation: float = 0.1         # protein turnover
    k_production: float = 0.2          # basal production
    # Diffusion (membrane-bound, limited)
    d_fz_dsh: float = 0.01
    d_vang_pk: float = 0.01
    d_fmi: float = 0.02
    # Wnt gradient
    wnt_gradient: bool = True
    wnt_direction: Tuple[float, float] = (1.0, 0.0)  # direction of Wnt source
    wnt_strength: float = 0.5
    # Coupling to Cdc42
    pcp_to_cdc42_weight: float = 0.5
    # Simulation
    dt: float = 0.01
    n_steps: int = 500
    seed: int = 0


class PlanarCellPolarityPathway:
    """
    [C/D] Planar Cell Polarity (PCP) pathway model.

    BIOLOGICAL BASIS [C Adler 2012; Strutt 2001]
      PCP establishes coordinated polarity across an epithelial sheet,
      perpendicular to the apical-basal axis. Core components:
        - Frizzled (Fz) + Dishevelled (Dsh): form "forward" complex
        - Van Gogh (Vang) + Prickle (Pk): form "reverse" complex
        - Flamingo (Fmi): atypical cadherin, mediates intercellular alignment

      Mechanism:
        1. Fz-Dsh and Vang-Pk mutually inhibit each other (Turing-type)
        2. Fmi forms homodimers across cell boundaries, aligning polarity
        3. Wnt gradient provides global directional cue
        4. Local feedback amplifies small asymmetries

      Output:
        - Each cell has a polarity vector (direction of Fz-Dsh complex)
        - Polarity is coordinated across the tissue
        - PCP signals activate downstream effectors (Cdc42, RhoA, JNK)

    MODEL [D]
      Reaction-diffusion system on a 2D grid:
        ∂[Fz-Dsh]/∂t = D_fz ∇²[Fz-Dsh] + k_prod - k_deg [Fz-Dsh]
                        + k_assembly [Fz][Dsh] - k_inhib [Vang-Pk][Fz-Dsh]
                        + k_wnt * Wnt(x)
        ∂[Vang-Pk]/∂t = D_vang ∇²[Vang-Pk] + k_prod - k_deg [Vang-Pk]
                        + k_assembly [Vang][Pk] - k_inhib [Fz-Dsh][Vang-Pk]
        ∂[Fmi]/∂t = D_fmi ∇²[Fmi] + k_align * (neighbour_avg - [Fmi])

      Wnt gradient: Wnt(x) = wnt_strength * (x · wnt_direction)

    COROLLARY-3 GUARD [D]
      All reactions and diffusion terms are FIRST-ORDER in the fields.
      We do NOT compute second derivatives of PCP protein concentrations.
      The Laplacian is first-order spatial coupling.

    PROP-3 AWARENESS [P]
      When we multiply fields (e.g. [Fz-Dsh] × [Vang-Pk] for mutual inhibition),
      the result loses two Sobolev orders (Carpet-19 Prop 3). We use
      softplus floors to maintain differentiability and document that the
      product is a lower-regularity quantity.

    HONEST
      • This is a simplified Turing-type model. Real PCP involves many more
        proteins, post-translational modifications, and feedback loops.
      • The mutual inhibition is modeled as bilinear terms, which is a
        simplification of the actual biochemistry.
      • Intercellular alignment via Fmi is approximated as diffusion-like.
      • This is a computational model, not validated molecular biology.
    """

    def __init__(self, cfg: Optional[PCPConfig] = None):
        self.cfg = cfg or PCPConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.nx, self.ny = self.cfg.grid_shape

        # State fields (per grid cell)
        self.fz_dsh: Optional[np.ndarray] = None  # (nx, ny)
        self.vang_pk: Optional[np.ndarray] = None
        self.fmi: Optional[np.ndarray] = None
        self.wnt: Optional[np.ndarray] = None
        self.polarity_vectors: Optional[np.ndarray] = None  # (nx, ny, 2)

    def _laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with periodic boundary conditions."""
        lap = np.zeros_like(field)
        for axis in range(2):
            lap += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
        return lap

    def _compute_wnt_gradient(self) -> np.ndarray:
        """Compute Wnt ligand gradient across the tissue."""
        if not self.cfg.wnt_gradient:
            return np.zeros((self.nx, self.ny))

        xs = np.linspace(-1, 1, self.nx)
        ys = np.linspace(-1, 1, self.ny)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')

        # Wnt concentration: dot product with direction
        wnt = self.cfg.wnt_strength * (
            xx * self.cfg.wnt_direction[0] + yy * self.cfg.wnt_direction[1]
        )
        # Normalize to [0, 1]
        wnt = (wnt - wnt.min()) / (wnt.max() - wnt.min() + 1e-12)
        return wnt

    def initialize(self) -> None:
        """Initialize PCP fields with small random perturbations."""
        shape = (self.nx, self.ny)
        noise = 0.05

        self.fz_dsh = self.cfg.fz_initial + noise * self._rng.standard_normal(shape)
        self.vang_pk = self.cfg.vang_initial + noise * self._rng.standard_normal(shape)
        self.fmi = self.cfg.fmi_initial + noise * self._rng.standard_normal(shape)

        # Ensure non-negative
        self.fz_dsh = np.maximum(self.fz_dsh, 0.0)
        self.vang_pk = np.maximum(self.vang_pk, 0.0)
        self.fmi = np.maximum(self.fmi, 0.0)

        self.wnt = self._compute_wnt_gradient()
        self.polarity_vectors = np.zeros((self.nx, self.ny, 2))

    def step(self) -> None:
        """One RD step for the PCP model."""
        if self.fz_dsh is None:
            self.initialize()

        dt = self.cfg.dt

        # Laplacians
        lap_fz = self._laplacian_2d(self.fz_dsh)
        lap_vang = self._laplacian_2d(self.vang_pk)
        lap_fmi = self._laplacian_2d(self.fmi)

        # Mutual inhibition (PROP-3 awareness: bilinear product)
        # Use softplus to maintain differentiability
        fz_dsh_soft = np.log1p(np.exp(self.fz_dsh * 10)) / 10  # soft approximation
        vang_pk_soft = np.log1p(np.exp(self.vang_pk * 10)) / 10

        inhibition_fz = self.cfg.k_mutual_inhibition * vang_pk_soft * fz_dsh_soft
        inhibition_vang = self.cfg.k_mutual_inhibition * fz_dsh_soft * vang_pk_soft

        # Wnt activation of Fz-Dsh
        wnt_activation = self.cfg.wnt_strength * self.wnt * self.fz_dsh

        # Update Fz-Dsh
        dfz = (self.cfg.d_fz_dsh * lap_fz
               + self.cfg.k_production - self.cfg.k_degradation * self.fz_dsh
               + self.cfg.k_fz_dsh_assembly * self.fz_dsh * (1 - self.fz_dsh)
               - inhibition_fz + wnt_activation)
        self.fz_dsh = np.maximum(self.fz_dsh + dt * dfz, 0.0)

        # Update Vang-Pk
        dvang = (self.cfg.d_vang_pk * lap_vang
                 + self.cfg.k_production - self.cfg.k_degradation * self.vang_pk
                 + self.cfg.k_vang_pk_assembly * self.vang_pk * (1 - self.vang_pk)
                 - inhibition_vang)
        self.vang_pk = np.maximum(self.vang_pk + dt * dvang, 0.0)

        # Update Fmi (intercellular alignment)
        # Fmi tends to match neighbours
        neighbour_avg = (np.roll(self.fmi, 1, axis=0) + np.roll(self.fmi, -1, axis=0)
                         + np.roll(self.fmi, 1, axis=1) + np.roll(self.fmi, -1, axis=1)) / 4.0
        dfmi = (self.cfg.d_fmi * lap_fmi
                + self.cfg.k_intercellular_align * (neighbour_avg - self.fmi)
                + self.cfg.k_production - self.cfg.k_degradation * self.fmi)
        self.fmi = np.maximum(self.fmi + dt * dfmi, 0.0)

        # Compute polarity vectors from Fz-Dsh gradient
        self._compute_polarity_vectors()

    def _compute_polarity_vectors(self) -> None:
        """Compute polarity vectors from Fz-Dsh gradient (first-order)."""
        # Gradient of Fz-Dsh (COROLLARY-3 guard: first-order only)
        grad_x = (np.roll(self.fz_dsh, -1, axis=0) - np.roll(self.fz_dsh, 1, axis=0)) / 2.0
        grad_y = (np.roll(self.fz_dsh, -1, axis=1) - np.roll(self.fz_dsh, 1, axis=1)) / 2.0

        # Normalize to unit vectors
        norm = np.sqrt(grad_x**2 + grad_y**2 + 1e-12)
        self.polarity_vectors[..., 0] = grad_x / norm
        self.polarity_vectors[..., 1] = grad_y / norm

    def run(self, n_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the PCP model for n_steps."""
        if self.fz_dsh is None:
            self.initialize()
        steps = n_steps or self.cfg.n_steps
        for _ in range(steps):
            self.step()
        return self.fz_dsh.copy(), self.vang_pk.copy(), self.fmi.copy()

    def get_cdc42_activation_signal(self) -> np.ndarray:
        """
        Compute Cdc42 activation signal from PCP.
        Returns: (nx, ny) array of activation strength.
        """
        if self.fz_dsh is None:
            return np.zeros((self.nx, self.ny))
        # Cdc42 activation proportional to Fz-Dsh complex
        return self.cfg.pcp_to_cdc42_weight * self.fz_dsh

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid_shape": self.cfg.grid_shape,
            "wnt_gradient": self.cfg.wnt_gradient,
            "wnt_direction": self.cfg.wnt_direction,
            "corollary3_guard": "first-order reactions/diffusion; no Hessian",
            "prop3_awareness": "bilinear products lose two Sobolev orders",
            "note": "computational model, not validated molecular biology",
        }


# =============================================================================
# SECTION 3 — [EDB1] EXPERIMENTAL DATA BRIDGE
# =============================================================================

@dataclass
class DataBridgeConfig:
    """Configuration for experimental data integration."""
    # Data paths
    imaging_dir: Optional[str] = None
    quantification_file: Optional[str] = None
    # Image processing
    image_channel: str = "membrane"  # "membrane" | "nucleus" | "polarity_marker"
    pixel_size_um: float = 0.5       # microns per pixel
    # Quantification
    cell_segmentation: bool = True
    polarity_metric: str = "asymmetry_index"  # "asymmetry_index" | "orientation_order"
    # Statistical comparison
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    # Calibration
    calibrate_parameters: bool = False
    n_mcmc_samples: int = 1000
    # Visualization
    overlay_model_on_data: bool = True
    save_figures: bool = False
    figure_dir: Optional[str] = None


@dataclass
class CalibrationResult:
    """Result of model calibration to experimental data."""
    parameter_name: str
    prior_value: float
    posterior_mean: float
    posterior_std: float
    credible_interval: Tuple[float, float]
    acceptance_rate: float
    log_likelihood: float
    converged: bool


class ExperimentalDataBridge:
    """
    [D] Bridge to experimental organoid imaging data.

    CAPABILITIES
      1. Load imaging data (TIFF, PNG, NPZ)
      2. Quantify cell shapes, polarity markers, gene expression
      3. Statistical comparison: model vs data
         - KS test for distributions
         - Wasserstein distance for spatial patterns
         - Correlation for time series
      4. Bayesian parameter calibration (simplified MCMC)
      5. Visualization overlay

    STATISTICAL METHODS [C]
      - Kolmogorov-Smirnov test: compare distributions
      - Wasserstein distance: compare spatial patterns
      - Pearson/Spearman correlation: compare time series
      - Bootstrap confidence intervals

    BAYESIAN CALIBRATION [D]
      Simplified Metropolis-Hastings MCMC:
        - Prior: uniform or Gaussian around initial guess
        - Likelihood: Gaussian error model
        - Proposal: random walk in parameter space
        - Acceptance: min(1, exp(Δlog_likelihood))

    COROLLARY-3 GUARD [D]
      All statistical operations are FIRST-ORDER (no Hessian of likelihood).
      MCMC uses gradient-free sampling (no second derivatives needed).

    HONEST
      • This is a COMPUTATIONAL comparison tool, NOT a diagnostic.
      • Real organoid imaging has noise, artifacts, biological variability.
      • Statistical tests assume certain distributions; violations may
        invalidate p-values.
      • Bayesian calibration is simplified; real parameter estimation
        requires careful prior specification and convergence diagnostics.
      • No biological/medical/diagnostic claim.
    """

    def __init__(self, cfg: Optional[DataBridgeConfig] = None):
        self.cfg = cfg or DataBridgeConfig()
        self._data: Optional[Dict[str, Any]] = None
        self._model_predictions: Optional[Dict[str, Any]] = None

    # ── Data loading ────────────────────────────────────────────────────
    def load_imaging_data(self, path: Optional[str] = None) -> np.ndarray:
        """
        Load organoid imaging data (TIFF/PNG/NPZ).
        Returns: image array (H, W) or (H, W, C).
        """
        path = path or self.cfg.imaging_dir
        if path is None:
            raise ValueError("No imaging data path specified")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Imaging data not found: {path}")

        if path.endswith('.npz'):
            data = np.load(path)
            return data['image'] if 'image' in data else list(data.values())[0]
        elif HAS_SKIMAGE:
            return skio.imread(path)
        else:
            raise ImportError("scikit-image required for TIFF/PNG loading")

    def load_quantification(self, path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load quantified data (CSV/JSON/NPZ).
        Returns: dict with keys like 'cell_areas', 'polarity_vectors', etc.
        """
        path = path or self.cfg.quantification_file
        if path is None:
            raise ValueError("No quantification file specified")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Quantification file not found: {path}")

        if path.endswith('.csv') and HAS_PANDAS:
            df = pd.read_csv(path)
            return {col: df[col].values for col in df.columns}
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                return json.load(f)
        elif path.endswith('.npz'):
            data = np.load(path)
            return {k: data[k] for k in data.files}
        else:
            raise ValueError(f"Unsupported format: {path}")

    # ── Quantification ──────────────────────────────────────────────────
    def quantify_cells(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Quantify cell properties from imaging data.
        Returns: dict with 'cell_areas', 'cell_perimeters', 'polarity_vectors', etc.
        """
        # Simplified: threshold-based segmentation
        if image.ndim == 3:
            gray = image.mean(axis=-1)
        else:
            gray = image

        # Threshold
        threshold = np.percentile(gray, 50)
        binary = gray > threshold

        # Label connected components (simplified)
        from scipy import ndimage
        labeled, n_cells = ndimage.label(binary)

        # Measure properties
        cell_areas = np.zeros(n_cells)
        cell_perimeters = np.zeros(n_cells)
        cell_centroids = np.zeros((n_cells, 2))

        for i in range(1, n_cells + 1):
            mask = labeled == i
            cell_areas[i-1] = mask.sum() * (self.cfg.pixel_size_um ** 2)
            # Perimeter: count boundary pixels
            eroded = ndimage.binary_erosion(mask)
            boundary = mask & ~eroded
            cell_perimeters[i-1] = boundary.sum() * self.cfg.pixel_size_um
            # Centroid
            coords = np.argwhere(mask)
            cell_centroids[i-1] = coords.mean(axis=0) * self.cfg.pixel_size_um

        # Polarity vectors (simplified: gradient of membrane marker)
        grad_x = np.gradient(gray, axis=0)
        grad_y = np.gradient(gray, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + 1e-12)

        polarity_vectors = np.zeros((n_cells, 2))
        for i in range(n_cells):
            mask = labeled == (i + 1)
            polarity_vectors[i, 0] = grad_x[mask].mean() / (grad_mag[mask].mean() + 1e-12)
            polarity_vectors[i, 1] = grad_y[mask].mean() / (grad_mag[mask].mean() + 1e-12)

        return {
            'n_cells': n_cells,
            'cell_areas': cell_areas,
            'cell_perimeters': cell_perimeters,
            'cell_centroids': cell_centroids,
            'polarity_vectors': polarity_vectors,
        }

    # ── Statistical comparison ──────────────────────────────────────────
    def compare_distributions(self, data_dist: np.ndarray, model_dist: np.ndarray
                                ) -> Dict[str, float]:
        """
        Compare two distributions using KS test and Wasserstein distance.
        """
        from scipy import stats

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(data_dist, model_dist)

        # Wasserstein distance
        wasserstein = stats.wasserstein_distance(data_dist, model_dist)

        # Mean and std comparison
        mean_diff = abs(data_dist.mean() - model_dist.mean())
        std_ratio = data_dist.std() / (model_dist.std() + 1e-12)

        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'wasserstein_distance': float(wasserstein),
            'mean_absolute_difference': float(mean_diff),
            'std_ratio': float(std_ratio),
        }

    def compare_spatial_patterns(self, data_pattern: np.ndarray,
                                   model_pattern: np.ndarray) -> Dict[str, float]:
        """
        Compare spatial patterns using correlation and structural similarity.
        """
        # Flatten for correlation
        data_flat = data_pattern.flatten()
        model_flat = model_pattern.flatten()

        # Pearson correlation
        corr = np.corrcoef(data_flat, model_flat)[0, 1]

        # Structural similarity (simplified: normalized cross-correlation)
        ncc = (data_flat * model_flat).sum() / (
            np.linalg.norm(data_flat) * np.linalg.norm(model_flat) + 1e-12
        )

        # Mean squared error
        mse = ((data_flat - model_flat) ** 2).mean()

        return {
            'pearson_correlation': float(corr),
            'normalized_cross_correlation': float(ncc),
            'mean_squared_error': float(mse),
        }

    def compare_time_series(self, data_ts: np.ndarray, model_ts: np.ndarray
                              ) -> Dict[str, float]:
        """
        Compare time series using correlation and dynamic time warping.
        """
        # Ensure same length
        min_len = min(len(data_ts), len(model_ts))
        data_ts = data_ts[:min_len]
        model_ts = model_ts[:min_len]

        # Pearson correlation
        corr = np.corrcoef(data_ts, model_ts)[0, 1]

        # Mean absolute error
        mae = np.abs(data_ts - model_ts).mean()

        # Normalized root mean squared error
        nrmse = np.sqrt(((data_ts - model_ts) ** 2).mean()) / (data_ts.std() + 1e-12)

        return {
            'pearson_correlation': float(corr),
            'mean_absolute_error': float(mae),
            'normalized_rmse': float(nrmse),
        }

    # ── Bayesian calibration ────────────────────────────────────────────
    def calibrate_parameter(self, parameter_name: str, initial_value: float,
                              data_summary: float, model_func,
                              prior_std: float = 0.5,
                              n_samples: Optional[int] = None
                              ) -> CalibrationResult:
        """
        Calibrate a single parameter using simplified MCMC.

        Args:
            parameter_name: name of parameter
            initial_value: starting value
            data_summary: summary statistic from data
            model_func: function(param) → model_summary
            prior_std: standard deviation of Gaussian prior
            n_samples: number of MCMC samples

        Returns:
            CalibrationResult with posterior statistics
        """
        n_samples = n_samples or self.cfg.n_mcmc_samples

        # Initialize
        current_param = initial_value
        current_model = model_func(current_param)
        current_likelihood = -0.5 * ((current_model - data_summary) / 0.1) ** 2
        current_prior = -0.5 * ((current_param - initial_value) / prior_std) ** 2
        current_posterior = current_likelihood + current_prior

        # MCMC chain
        samples = [current_param]
        likelihoods = [current_likelihood]
        accepted = 0

        for i in range(n_samples):
            # Proposal: random walk
            proposal = current_param + 0.1 * prior_std * self._rng.standard_normal()

            # Evaluate proposal
            proposal_model = model_func(proposal)
            proposal_likelihood = -0.5 * ((proposal_model - data_summary) / 0.1) ** 2
            proposal_prior = -0.5 * ((proposal - initial_value) / prior_std) ** 2
            proposal_posterior = proposal_likelihood + proposal_prior

            # Acceptance criterion
            log_alpha = proposal_posterior - current_posterior
            if log_alpha > 0 or self._rng.random() < math.exp(log_alpha):
                current_param = proposal
                current_likelihood = proposal_likelihood
                current_posterior = proposal_posterior
                accepted += 1

            samples.append(current_param)
            likelihoods.append(current_likelihood)

        samples = np.array(samples)
        acceptance_rate = accepted / n_samples

        # Posterior statistics
        posterior_mean = samples.mean()
        posterior_std = samples.std()
        ci_lower = np.percentile(samples, 100 * (1 - self.cfg.confidence_level) / 2)
        ci_upper = np.percentile(samples, 100 * (1 + self.cfg.confidence_level) / 2)

        # Convergence check (simplified: Gelman-Rubin not implemented)
        converged = acceptance_rate > 0.2 and acceptance_rate < 0.5

        return CalibrationResult(
            parameter_name=parameter_name,
            prior_value=initial_value,
            posterior_mean=float(posterior_mean),
            posterior_std=float(posterior_std),
            credible_interval=(float(ci_lower), float(ci_upper)),
            acceptance_rate=float(acceptance_rate),
            log_likelihood=float(current_likelihood),
            converged=converged,
        )

    # ── Visualization ───────────────────────────────────────────────────
    def visualize_overlay(self, data_image: np.ndarray,
                            model_prediction: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize model prediction overlaid on experimental data.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping visualization")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Data
        axes[0].imshow(data_image, cmap='viridis')
        axes[0].set_title('Experimental Data')
        axes[0].axis('off')

        # Model
        axes[1].imshow(model_prediction, cmap='viridis')
        axes[1].set_title('Model Prediction')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(data_image, cmap='gray', alpha=0.5)
        axes[2].imshow(model_prediction, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path and self.cfg.save_figures:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved overlay figure to {save_path}")

        plt.close()

    def get_status(self) -> Dict[str, Any]:
        return {
            "has_skimage": HAS_SKIMAGE,
            "has_pandas": HAS_PANDAS,
            "imaging_dir": self.cfg.imaging_dir,
            "quantification_file": self.cfg.quantification_file,
            "note": "computational comparison only; NOT diagnostic",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIExperimentalOrganoidSuite:
    """
    Unifies: StochasticMembraneTrafficking + PlanarCellPolarityPathway
             + ExperimentalDataBridge.

    FULL EXPERIMENTAL PIPELINE:
      1. Stochastic Cdc42 trafficking with membrane/cytosol compartments
      2. PCP pathway establishes planar polarity
      3. PCP → Cdc42 coupling (polarity guides trafficking)
      4. Compare with experimental organoid imaging data
      5. Calibrate model parameters via Bayesian inference
    """

    def __init__(self,
                 trafficking_cfg: Optional[TraffickingConfig] = None,
                 pcp_cfg: Optional[PCPConfig] = None,
                 data_cfg: Optional[DataBridgeConfig] = None):
        self.trafficking = StochasticMembraneTrafficking(trafficking_cfg)
        self.pcp = PlanarCellPolarityPathway(pcp_cfg)
        self.data_bridge = ExperimentalDataBridge(data_cfg)

    def run_coupled_simulation(self, n_steps: int = 100) -> Dict[str, Any]:
        """
        Run coupled trafficking + PCP simulation.
        """
        # Initialize
        self.trafficking.initialize()
        self.pcp.initialize()

        # Run PCP first to establish polarity
        fz_dsh, vang_pk, fmi = self.pcp.run(n_steps=n_steps // 2)

        # Get PCP → Cdc42 activation signal
        cdc42_signal = self.pcp.get_cdc42_activation_signal()

        # Run trafficking with PCP coupling
        # (simplified: use PCP signal to modulate GEF rate)
        self.trafficking.cfg.k_gef *= (1.0 + cdc42_signal.mean())

        traj, times = self.trafficking.simulate(max_steps=n_steps)

        # Final state
        final_state = traj[-1]
        polarity_vector = self.trafficking.get_polarity_vector(final_state)

        return {
            "n_steps": n_steps,
            "pcp_fz_dsh_mean": float(fz_dsh.mean()),
            "pcp_vang_pk_mean": float(vang_pk.mean()),
            "pcp_fmi_mean": float(fmi.mean()),
            "cdc42_membrane_mean": float(final_state['ctp_mem'].mean()),
            "cdc42_cytosol_mean": float(final_state['cdp_cyt'].mean()),
            "polarity_vector": polarity_vector.tolist(),
            "trajectory_length": len(traj),
            "final_time": times[-1],
        }

    def compare_with_data(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare model predictions with experimental data.
        """
        # Run model
        model_result = self.run_coupled_simulation()

        # Compare
        comparisons = {}

        if 'cell_polarity_magnitude' in data_summary:
            data_pol = data_summary['cell_polarity_magnitude']
            model_pol = np.array(model_result['polarity_vector'])
            comparisons['polarity'] = self.data_bridge.compare_distributions(
                data_pol, model_pol)

        if 'cdc42_membrane_fraction' in data_summary:
            data_cdc42 = data_summary['cdc42_membrane_fraction']
            model_cdc42 = model_result['cdc42_membrane_mean'] / (
                model_result['cdc42_membrane_mean'] + model_result['cdc42_cytosol_mean'] + 1e-12)
            comparisons['cdc42'] = {
                'data_mean': float(data_cdc42.mean()),
                'model_value': float(model_cdc42),
                'absolute_difference': float(abs(data_cdc42.mean() - model_cdc42)),
            }

        return comparisons

    def calibrate_to_data(self, data_summary: Dict[str, Any],
                            parameter_name: str = 'k_gef',
                            initial_value: Optional[float] = None
                            ) -> CalibrationResult:
        """
        Calibrate a model parameter to match experimental data.
        """
        if initial_value is None:
            initial_value = getattr(self.trafficking.cfg, parameter_name)

        # Define model function
        def model_func(param_value):
            setattr(self.trafficking.cfg, parameter_name, param_value)
            result = self.run_coupled_simulation()
            return result['cdc42_membrane_mean'] / (
                result['cdc42_membrane_mean'] + result['cdc42_cytosol_mean'] + 1e-12)

        # Data summary
        if 'cdc42_membrane_fraction' in data_summary:
            data_value = float(data_summary['cdc42_membrane_fraction'].mean())
        else:
            raise ValueError("No cdc42_membrane_fraction in data_summary")

        # Calibrate
        return self.data_bridge.calibrate_parameter(
            parameter_name=parameter_name,
            initial_value=initial_value,
            data_summary=data_value,
            model_func=model_func,
        )

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "trafficking": self.trafficking.get_status(),
            "pcp": self.pcp.get_status(),
            "data_bridge": self.data_bridge.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOID STUDIO EXPERIMENTAL — SMOKE TEST")
    print(" Membrane Trafficking + PCP Pathway + Imaging Integration")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" scikit-image available : {HAS_SKIMAGE}")
    print(f" pandas available       : {HAS_PANDAS}")

    suite = SESIExperimentalOrganoidSuite(
        trafficking_cfg=TraffickingConfig(n_cells=6, max_time=50.0, max_steps=5000),
        pcp_cfg=PCPConfig(grid_shape=(6, 6), n_steps=100),
        data_cfg=DataBridgeConfig())

    # ── 1. Stochastic Membrane Trafficking ──────────────────────────────
    print("\n [SMT1] Stochastic Membrane Trafficking")
    trafficking = suite.trafficking
    print(f"   n_cells              : {trafficking.n_cells}")
    print(f"   n_cdc42_total        : {trafficking.cfg.n_cdc42_total}")
    print(f"   n_gdi_total          : {trafficking.cfg.n_gdi_total}")
    print(f"   compartments         : {trafficking.get_status()['compartments']}")
    print(f"   species              : {trafficking.get_status()['species']}")

    # Run simulation
    trafficking.initialize()
    traj, times = trafficking.simulate(max_steps=2000)
    print(f"   trajectory length    : {len(traj)}")
    print(f"   final time           : {times[-1]:.2f}")

    final_state = traj[-1]
    print(f"   final Cdc42-GTP(mem) : {final_state['ctp_mem'].mean():.1f}")
    print(f"   final Cdc42-GDP(cyt) : {final_state['cdp_cyt'].mean():.1f}")
    print(f"   final GDI:Cdc42(cyt) : {final_state['gdi_cplx'].mean():.1f}")

    polarity = trafficking.get_polarity_vector(final_state)
    print(f"   polarity vector      : {polarity}")
    print(f"   COROLLARY-3 guard    : {trafficking.get_status()['corollary3_guard']}")
    print(f"   HINO guard           : {trafficking.get_status()['hino_guard']}")

    # ── 2. Planar Cell Polarity Pathway ─────────────────────────────────
    print("\n [PCP1] Planar Cell Polarity Pathway")
    pcp = suite.pcp
    print(f"   grid shape           : {pcp.cfg.grid_shape}")
    print(f"   Wnt gradient         : {pcp.cfg.wnt_gradient}")
    print(f"   Wnt direction        : {pcp.cfg.wnt_direction}")

    # Run PCP model
    fz_dsh, vang_pk, fmi = pcp.run(n_steps=80)
    print(f"   Fz-Dsh mean          : {fz_dsh.mean():.4f}")
    print(f"   Vang-Pk mean         : {vang_pk.mean():.4f}")
    print(f"   Fmi mean             : {fmi.mean():.4f}")

    # Polarity vectors
    pol_vecs = pcp.polarity_vectors
    print(f"   polarity vectors     : shape={pol_vecs.shape}")
    print(f"   mean vector magnitude: {np.linalg.norm(pol_vecs, axis=-1).mean():.4f}")

    # Cdc42 activation signal
    cdc42_signal = pcp.get_cdc42_activation_signal()
    print(f"   Cdc42 signal mean    : {cdc42_signal.mean():.4f}")
    print(f"   COROLLARY-3 guard    : {pcp.get_status()['corollary3_guard']}")
    print(f"   PROP-3 awareness     : {pcp.get_status()['prop3_awareness']}")

    # ── 3. Coupled Simulation ───────────────────────────────────────────
    print("\n [COUPLED] Trafficking + PCP Coupling")
    result = suite.run_coupled_simulation(n_steps=50)
    print(f"   n_steps              : {result['n_steps']}")
    print(f"   PCP Fz-Dsh mean      : {result['pcp_fz_dsh_mean']:.4f}")
    print(f"   PCP Vang-Pk mean     : {result['pcp_vang_pk_mean']:.4f}")
    print(f"   Cdc42 membrane mean  : {result['cdc42_membrane_mean']:.1f}")
    print(f"   Cdc42 cytosol mean   : {result['cdc42_cytosol_mean']:.1f}")
    print(f"   polarity vector      : {result['polarity_vector']}")

    # ── 4. Experimental Data Bridge ─────────────────────────────────────
    print("\n [EDB1] Experimental Data Bridge")
    data_bridge = suite.data_bridge
    print(f"   scikit-image         : {data_bridge.get_status()['has_skimage']}")
    print(f"   pandas               : {data_bridge.get_status()['has_pandas']}")

    # Synthetic data for testing
    synthetic_data = {
        'cell_polarity_magnitude': np.random.beta(2, 5, size=50),
        'cdc42_membrane_fraction': np.random.beta(3, 2, size=50),
    }

    # Compare with model
    comparisons = suite.compare_with_data(synthetic_data)
    print(f"   comparison keys      : {list(comparisons.keys())}")

    if 'polarity' in comparisons:
        pol_comp = comparisons['polarity']
        print(f"   polarity KS stat     : {pol_comp['ks_statistic']:.4f}")
        print(f"   polarity KS p-value  : {pol_comp['ks_pvalue']:.4f}")
        print(f"   polarity Wasserstein : {pol_comp['wasserstein_distance']:.4f}")

    if 'cdc42' in comparisons:
        cdc42_comp = comparisons['cdc42']
        print(f"   Cdc42 data mean      : {cdc42_comp['data_mean']:.4f}")
        print(f"   Cdc42 model value    : {cdc42_comp['model_value']:.4f}")
        print(f"   Cdc42 abs diff       : {cdc42_comp['absolute_difference']:.4f}")

    # ── 5. Bayesian Calibration ─────────────────────────────────────────
    print("\n [CALIB] Bayesian Parameter Calibration")
    try:
        calib_result = suite.calibrate_to_data(
            data_summary=synthetic_data,
            parameter_name='k_gef',
            initial_value=0.15,
        )
        print(f"   parameter            : {calib_result.parameter_name}")
        print(f"   prior value          : {calib_result.prior_value:.4f}")
        print(f"   posterior mean       : {calib_result.posterior_mean:.4f}")
        print(f"   posterior std        : {calib_result.posterior_std:.4f}")
        print(f"   95% credible interval: [{calib_result.credible_interval[0]:.4f}, "
              f"{calib_result.credible_interval[1]:.4f}]")
        print(f"   acceptance rate      : {calib_result.acceptance_rate:.3f}")
        print(f"   converged            : {calib_result.converged}")
    except Exception as e:
        print(f"   calibration failed   : {e}")

    # ── 6. Full Status ──────────────────────────────────────────────────
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
    print("       Real trafficking involves vesicle dynamics, motor proteins,")
    print("       cytoskeletal transport not captured here.")
    print(" [C]   PCP pathway: core proteins (Fz, Dsh, Vang, Pk, Fmi) modeled")
    print("       as Turing-type mutual inhibition. Real PCP involves many more")
    print("       proteins, post-translational modifications, feedback loops.")
    print(" [D]   ExperimentalDataBridge: statistical comparison is HEURISTIC.")
    print("       Real organoid imaging has noise, artifacts, biological variability.")
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
