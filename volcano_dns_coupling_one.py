# =============================================================================
# VOLCANO DNS COUPLING ONE  —  v1.0
# Grids volcano_one.py's eruption-column/tephra/PDC output into SuperDNSSnapshot
# -compatible .pt training files for StructuralFNO3D.for_volcano()
# =============================================================================
# Developer    : Yoon A Limsuwan
# Organization : MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Assisted by  : Claude (Anthropic)
# License      : MIT
# Year         : 2026
#
# Description:
#   volcano_one.py's EruptionColumnModel (MTT plume-height scaling),
#   TephraTransportModel (settling velocity + single-particle-size
#   ashfall footprint), and PyroclasticDensityCurrentModel (energy-cone
#   runout) are all explicitly-flagged simplified/analytical models --
#   TephraTransportModel.simple_ashfall_radius()'s own docstring calls it
#   a "crude single-particle-size, no-diffusion estimate", and
#   PyroclasticDensityCurrentModel's own docstring calls its radial
#   hazard zone "NOT topography-aware". This module is the projection
#   layer flagged in structural_fno_3d_v2_4.py's changelog: it turns
#   their scalar outputs (column height, ashfall downwind distance, PDC
#   runout distance) into two gridded hazard fields -- airborne ash
#   concentration C_ash(x,y) and a PDC extent/intensity indicator
#   C_pdc(x,y) -- that StructuralFNO3D.for_volcano() can train on.
#
#   Both fields are built as simple radial/advected decay profiles
#   anchored to volcano_one.py's own scalar distance estimates (ashfall
#   downwind distance, PDC runout distance) -- NOT a real multiphase
#   ash-transport or PDC simulation (e.g. Ash3d/HYSPLIT/Tephra2 for ash;
#   TITAN2D/VolcFlow for PDC). Do not use this for real hazard-zone
#   mapping; it exists to give StructuralFNO3D a physically-anchored (if
#   simplified) spatial target to learn an operator against.
#
#   SCOPE NOTE: no real DEM/topography is used -- both fields are
#   radially symmetric about the vent (ash field additionally advected
#   downwind), matching the radial-symmetry assumption volcano_one.py's
#   PDC model docstring already states explicitly. Real topography
#   (valley-channelized PDC runout, terrain-modulated ashfall) would
#   need a DEM the same way flood_dns_coupling_one.py flags for its own
#   synthetic terrain.
#
#   3D EMBEDDING NOTE: as in the flood/storm coupling modules, both
#   fields are 2D (map-view) phenomena, extruded (tiled) across Nz thin
#   layers purely for shape compatibility with StructuralFNO3D's 3D grid
#   -- not a claim of real vertical structure in these two channels
#   (airborne column vertical structure is exactly what MTT plume theory
#   describes, but is not itself one of the two state channels here).
#
# Output snapshot layout (matches SuperDNSSnapshot / StructuralFNO3D.for_volcano()):
#   'u_t0'           : (2, Nx, Ny, Nz)  — [C_ash, C_pdc] pre-eruption (zero)
#   'u_tT'           : (2, Nx, Ny, Nz)  — [C_ash, C_pdc] at t_target into the eruption
#   'sigma'          : (1, Nx, Ny, Nz)  — normalised distance-from-vent field
#   't_target'       : float             — normalised position in the eruption timeline
#   'forcing_window' : (2, T)            — [MER(t), column_height(t)]
#   'metadata'       : dict              — eruption/grid parameters
# =============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "volcano_dns_coupling_one.py requires PyTorch (for .pt snapshot output). "
        "Install with: pip install torch"
    ) from e

try:
    from volcano_one import (
        EruptionColumnModel, TephraTransportModel, PyroclasticDensityCurrentModel,
    )
except ImportError as e:
    raise ImportError(
        "volcano_dns_coupling_one.py requires volcano_one.py to be importable "
        "(same directory or on PYTHONPATH)."
    ) from e

logger = logging.getLogger("volcano_dns_coupling_one")
logging.basicConfig(level=logging.INFO)

__version__ = "1.0.0"


# =====================================================================
# 1. GRID SPEC
# =====================================================================

@dataclass
class VolcanoGridSpec:
    nx: int = 32
    ny: int = 32
    nz: int = 4
    dx_m: float = 500.0   # near-vent scale (500 m cells)


# =====================================================================
# 2. COUPLING
# =====================================================================

class VolcanoDNSCoupling:
    """
    Orchestrates volcano_one.py's EruptionColumnModel / TephraTransportModel /
    PyroclasticDensityCurrentModel into gridded (C_ash, C_pdc) hazard
    fields for StructuralFNO3D.for_volcano().
    """

    def __init__(self, grid: VolcanoGridSpec,
                 column_model: Optional[EruptionColumnModel] = None,
                 pdc_model: Optional[PyroclasticDensityCurrentModel] = None,
                 median_ash_diameter_m: float = 1e-4,
                 wind_speed_m_s: float = 15.0,
                 wind_heading_deg: float = 90.0,
                 vent_xy: Optional[Tuple[float, float]] = None):
        self.grid = grid
        self.column_model = column_model or EruptionColumnModel()
        self.pdc_model = pdc_model or PyroclasticDensityCurrentModel()
        self.median_ash_diameter_m = median_ash_diameter_m
        self.wind_speed_m_s = wind_speed_m_s
        self.wind_heading = math.radians(wind_heading_deg)

        xs = (np.arange(grid.nx) - grid.nx // 2) * grid.dx_m
        ys = (np.arange(grid.ny) - grid.ny // 2) * grid.dx_m
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        self.vent_x, self.vent_y = vent_xy if vent_xy is not None else (0.0, 0.0)
        self.r_from_vent = np.sqrt((self.X - self.vent_x) ** 2 + (self.Y - self.vent_y) ** 2)

    def _ash_field(self, mass_eruption_rate_kg_s: float,
                    T_erupted_K: float = 1273.0) -> np.ndarray:
        """
        Airborne ash concentration proxy C_ash(x,y) in [0,1]: an
        advected-Gaussian-like footprint anchored to volcano_one.py's
        own simple_ashfall_radius() downwind distance (the along-wind
        e-folding decay length) and a fixed cross-wind spread fraction
        of that distance (illustrative, not a fitted diffusion
        coefficient -- see module scope note).
        """
        H = self.column_model.column_height(mass_eruption_rate_kg_s, T_erupted_K)
        footprint = TephraTransportModel.simple_ashfall_radius(
            H, self.median_ash_diameter_m, wind_speed_m_s=self.wind_speed_m_s)
        downwind_L = max(footprint["downwind_distance_m"], self.grid.dx_m)
        crosswind_L = 0.25 * downwind_L

        dx, dy = self.X - self.vent_x, self.Y - self.vent_y
        along = dx * math.cos(self.wind_heading) + dy * math.sin(self.wind_heading)
        across = -dx * math.sin(self.wind_heading) + dy * math.cos(self.wind_heading)

        along_decay = np.where(along >= 0, np.exp(-along / downwind_L), np.exp(along / (0.3 * downwind_L)))
        across_decay = np.exp(-0.5 * (across / max(crosswind_L, self.grid.dx_m)) ** 2)
        c_ash = along_decay * across_decay
        return np.clip(c_ash, 0.0, 1.0)

    def _pdc_field(self, column_collapse_height_m: float) -> np.ndarray:
        """
        PDC extent/intensity indicator C_pdc(x,y) in [0,1]: 1 at the
        vent, decaying to 0 at PyroclasticDensityCurrentModel's own
        energy-cone runout_distance, radially symmetric (matching that
        model's own stated radial-symmetry / non-topography-aware
        scope).
        """
        L = self.pdc_model.runout_distance(column_collapse_height_m)
        if L <= 0:
            return np.zeros_like(self.r_from_vent)
        c_pdc = np.clip(1.0 - self.r_from_vent / L, 0.0, 1.0)
        return c_pdc

    def _sigma_field(self) -> np.ndarray:
        """Normalised distance-from-vent -- near-vent is the high-hazard
        structural regime for both ash and PDC channels."""
        r = self.r_from_vent
        return (r - r.min()) / (r.max() - r.min() + 1e-8)

    def _extrude(self, field2d: np.ndarray) -> np.ndarray:
        return np.tile(field2d[:, :, None], (1, 1, self.grid.nz))

    def generate_snapshot(
        self,
        mass_eruption_rate_kg_s: float,
        eruption_duration_s: float,
        t_target_frac: float = 0.5,
        T_erupted_K: float = 1273.0,
        forcing_len: int = 64,
        mer_ramp_fraction: float = 0.3,
    ) -> Dict:
        """
        Simulates a MER ramp-up from 0 to `mass_eruption_rate_kg_s` over
        `mer_ramp_fraction` of `eruption_duration_s`, then sustained MER,
        and returns a SuperDNSSnapshot-shaped dict at t_target_frac of
        the eruption duration.

        NOTE: volcano_one.py does not itself provide an eruption
        time-history model, so the ramp-then-sustain profile here is an
        illustrative timeline construction (a standard simplifying
        assumption for a first eruptive phase), not an observed/forecast
        eruption chronology.
        """
        def mer_at(frac: float) -> float:
            frac = float(np.clip(frac, 0.0, 1.0))
            if frac >= mer_ramp_fraction:
                return mass_eruption_rate_kg_s
            return mass_eruption_rate_kg_s * (frac / mer_ramp_fraction)

        mer_t0 = mer_at(0.0)
        mer_tT = mer_at(t_target_frac)

        H_tT = self.column_model.column_height(max(mer_tT, 1.0), T_erupted_K)
        collapse = self.column_model.is_column_collapse_likely(
            max(mer_tT, 1.0), vent_radius_m=50.0, T_erupted_K=T_erupted_K)
        collapse_height_m = H_tT if collapse["collapse_likely"] else 0.0

        c_ash_t0 = np.zeros((self.grid.nx, self.grid.ny))
        c_pdc_t0 = np.zeros((self.grid.nx, self.grid.ny))
        c_ash_tT = self._ash_field(max(mer_tT, 1.0), T_erupted_K) if mer_tT > 0 else c_ash_t0
        c_pdc_tT = self._pdc_field(collapse_height_m) if collapse_height_m > 0 else c_pdc_t0

        u_t0 = np.stack([self._extrude(c_ash_t0), self._extrude(c_pdc_t0)], axis=0)
        u_tT = np.stack([self._extrude(c_ash_tT), self._extrude(c_pdc_tT)], axis=0)
        sigma = self._extrude(self._sigma_field())[None, ...]

        fracs = np.linspace(0.0, t_target_frac, forcing_len)
        mer_series = np.array([mer_at(f) for f in fracs])
        height_series = np.array([
            self.column_model.column_height(max(m, 1.0), T_erupted_K) for m in mer_series
        ])
        forcing_window = np.stack([mer_series, height_series], axis=0)   # (2, T)

        return {
            "u_t0":     torch.from_numpy(u_t0).float(),
            "u_tT":     torch.from_numpy(u_tT).float(),
            "sigma":    torch.from_numpy(sigma).float(),
            "t_target": float(t_target_frac),
            "forcing_window": torch.from_numpy(np.ascontiguousarray(forcing_window)).float(),
            "metadata": {
                "mass_eruption_rate_kg_s": mass_eruption_rate_kg_s,
                "eruption_duration_s": eruption_duration_s,
                "T_erupted_K": T_erupted_K,
                "column_height_m_at_tT": H_tT,
                "column_collapse_likely": collapse["collapse_likely"],
                "dx_m": self.grid.dx_m,
                "grid_shape": (self.grid.nx, self.grid.ny, self.grid.nz),
                "state_channels": ("C_ash", "C_pdc"),
                "source": "volcano_dns_coupling_one",
                "version": __version__,
            },
        }

    def save_snapshot(self, snapshot: Dict, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, out_path)
        return out_path


# =====================================================================
# 3. DATASET-GENERATION DRIVER
# =====================================================================

def generate_volcano_dataset(
    out_dir: str | Path,
    eruptions: Tuple[Tuple[str, float, float], ...],
    grid: Optional[VolcanoGridSpec] = None,
) -> int:
    """
    `eruptions`: tuples of (label, mass_eruption_rate_kg_s, eruption_duration_s).
    Writes one .pt file per eruption scenario.
    """
    grid = grid or VolcanoGridSpec()
    out_dir = Path(out_dir)
    coupling = VolcanoDNSCoupling(grid)
    n_written = 0
    for (label, mer, dur) in eruptions:
        snap = coupling.generate_snapshot(mer, dur)
        coupling.save_snapshot(snap, out_dir / f"volcano_{label}.pt")
        n_written += 1
    logger.info(f"Wrote {n_written} volcano snapshots to {out_dir}")
    return n_written


if __name__ == "__main__":
    print(f"volcano_dns_coupling_one.py v{__version__}")
    print("-" * 70)

    grid = VolcanoGridSpec(nx=32, ny=32, nz=4, dx_m=500.0)
    coupling = VolcanoDNSCoupling(grid)

    snap = coupling.generate_snapshot(
        mass_eruption_rate_kg_s=1.4e7, eruption_duration_s=3600.0,
    )
    print(f"u_t0  shape: {tuple(snap['u_t0'].shape)}")
    print(f"u_tT  shape: {tuple(snap['u_tT'].shape)}")
    print(f"sigma shape: {tuple(snap['sigma'].shape)}")
    print(f"forcing_window shape: {tuple(snap['forcing_window'].shape)}")
    print(f"t_target: {snap['t_target']}")
    print(f"metadata: {snap['metadata']}")

    out_path = coupling.save_snapshot(snap, "/tmp/volcano_snapshots/volcano_msh1980_demo.pt")
    print(f"Saved: {out_path}")

    n = generate_volcano_dataset(
        "/tmp/volcano_snapshots",
        eruptions=(
            ("VEI3", 1e5, 1800.0),
            ("VEI5_MSH1980", 1.4e7, 3600.0),
            ("VEI6_Pinatubo", 1e8, 7200.0),
        ),
        grid=grid,
    )
    print(f"Ensemble generation wrote {n} snapshot files.")
