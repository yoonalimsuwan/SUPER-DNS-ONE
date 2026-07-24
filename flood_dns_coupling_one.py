# =============================================================================
# FLOOD DNS COUPLING ONE  —  v1.0
# Grids flood_one.py's lumped hydrology/hydraulics output into SuperDNSSnapshot
# -compatible .pt training files for StructuralFNO3D.for_flood()
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
#   flood_one.py (WatershedHydrology / ChannelHydraulics / FloodRouting /
#   DamBreachModel / FloodplainModel / FloodOne) is a lumped-parameter /
#   analytical hydrology-hydraulics engine: it produces scalar and 1D
#   time-series quantities (runoff depth, peak discharge, a routed
#   hydrograph, a single water-surface elevation), not a gridded field.
#   This module is the missing projection layer flagged in
#   structural_fno_3d_v2_4.py's changelog: it takes flood_one.py's
#   scalar/1D outputs and projects them onto a spatial grid via the
#   HAND (Height Above Nearest Drainage) method already implemented in
#   FloodplainModel, producing (u_t0, u_tT, sigma, forcing_window)
#   snapshots StructuralFNO3D.for_flood() / SuperDNSDataset can train on.
#
#   SCOPE / HONESTY NOTE: there is no real DEM ingestion here yet --
#   `synthetic_valley_terrain()` below generates an illustrative
#   synthetic valley/floodplain elevation surface (sloped plane + a
#   channel trench + smooth low-frequency roughness) purely so this
#   pipeline is runnable end-to-end today. For real hazard-relevant
#   training data, replace `synthetic_valley_terrain()` with a real DEM
#   raster (e.g. loaded from a GeoTIFF via rasterio) resampled onto the
#   same (Nx, Ny) grid -- everything downstream (HAND, inundation,
#   Manning discharge, sigma) is DEM-format-agnostic and does not need
#   to change.
#
#   3D EMBEDDING NOTE: flood inundation is a 2D (surface) phenomenon.
#   StructuralFNO3D operates on a 3D (Nx,Ny,Nz) grid, so every 2D field
#   here is extruded (tiled, not physically stratified) across Nz thin
#   layers -- a "2.5D" embedding purely for shape compatibility with the
#   3D architecture, not a claim that flood depth varies with height.
#   Nz=4 by default; keep it small, since these layers are duplicates.
#
# Output snapshot layout (matches SuperDNSSnapshot / StructuralFNO3D.for_flood()):
#   'u_t0'           : (2, Nx, Ny, Nz)  — [h, q] before the storm (baseflow)
#   'u_tT'           : (2, Nx, Ny, Nz)  — [h, q] at t_target into the storm
#   'sigma'          : (1, Nx, Ny, Nz)  — normalised HAND field (structural regime)
#   't_target'       : float             — normalised position in the storm timeline
#   'forcing_window' : (1, T)            — rainfall intensity i(t) hyetograph
#   'metadata'       : dict              — storm/watershed parameters, dx, grid shape
# =============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "flood_dns_coupling_one.py requires PyTorch (for .pt snapshot output). "
        "Install with: pip install torch"
    ) from e

try:
    from flood_one import (
        WatershedHydrology, ChannelHydraulics, RainfallModel,
        FloodRouting, DamGeometry, DamBreachModel, FloodplainModel, FloodOne,
    )
except ImportError as e:
    raise ImportError(
        "flood_dns_coupling_one.py requires flood_one.py to be importable "
        "(same directory or on PYTHONPATH)."
    ) from e

logger = logging.getLogger("flood_dns_coupling_one")
logging.basicConfig(level=logging.INFO)

__version__ = "1.0.0"
G_ACCEL = 9.81


# =====================================================================
# 1. SYNTHETIC TERRAIN  (replace with a real DEM for real hazard work)
# =====================================================================

def synthetic_valley_terrain(nx: int, ny: int, dx_m: float,
                              valley_slope: float = 0.002,
                              channel_depth_m: float = 4.0,
                              channel_half_width_cells: int = 3,
                              roughness_amplitude_m: float = 0.5,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    Illustrative synthetic elevation grid [m]: a plane sloping down along
    x at `valley_slope` (typical small-watershed longitudinal slope),
    with a channel trench of `channel_depth_m` running along the y=ny/2
    centerline, plus smooth low-frequency roughness. NOT a real DEM --
    see the module-level scope note.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(nx) * dx_m
    plane = -valley_slope * x[:, None] * np.ones((1, ny))

    y_idx = np.arange(ny)
    channel_center = ny // 2
    dist_from_channel = np.abs(y_idx - channel_center)[None, :] * np.ones((nx, 1))
    channel_trench = -channel_depth_m * np.exp(
        -0.5 * (dist_from_channel / max(channel_half_width_cells, 1)) ** 2
    )

    # smooth low-frequency roughness via a small random field, blurred
    coarse = rng.normal(0, 1, size=(max(nx // 8, 2), max(ny // 8, 2)))
    roughness = np.kron(coarse, np.ones((math.ceil(nx / coarse.shape[0]),
                                          math.ceil(ny / coarse.shape[1]))))[:nx, :ny]
    roughness = roughness_amplitude_m * roughness / (np.abs(roughness).max() + 1e-8)

    elevation = plane + channel_trench + roughness
    elevation -= elevation.min()   # anchor to 0 at the lowest cell
    return elevation


# =====================================================================
# 2. GRID SPEC
# =====================================================================

@dataclass
class FloodGridSpec:
    nx: int = 32
    ny: int = 32
    nz: int = 4          # thin "2.5D" extrusion, see module docstring
    dx_m: float = 50.0   # cell size


# =====================================================================
# 3. COUPLING
# =====================================================================

class FloodDNSCoupling:
    """
    Orchestrates flood_one.py's design-storm pipeline -> gridded
    (h, q) snapshots for StructuralFNO3D.for_flood().
    """

    def __init__(self, watershed: WatershedHydrology, channel: ChannelHydraulics,
                 grid: FloodGridSpec, elevation_grid_m: Optional[np.ndarray] = None,
                 manning_n: float = 0.04, rainfall: Optional[RainfallModel] = None,
                 seed: Optional[int] = None):
        self.watershed = watershed
        self.channel = channel
        self.grid = grid
        self.manning_n = manning_n
        self.rainfall = rainfall or RainfallModel()
        self.flood_one = FloodOne(watershed, channel, self.rainfall)
        self.elevation = (
            elevation_grid_m if elevation_grid_m is not None
            else synthetic_valley_terrain(grid.nx, grid.ny, grid.dx_m, seed=seed)
        )
        # drainage (channel-bed) reference elevation = lowest cell, i.e. the
        # channel trench bottom -- HAND is computed relative to this.
        self.drainage_elev_m = float(self.elevation.min())
        # local terrain slope magnitude, for Manning discharge below
        gy, gx = np.gradient(self.elevation, grid.dx_m)
        self.slope_mag = np.clip(np.sqrt(gx ** 2 + gy ** 2), 1e-4, None)

    def _depth_field(self, water_surface_elevation_m: float) -> np.ndarray:
        return FloodplainModel.inundation_depth(self.elevation, water_surface_elevation_m)

    def _discharge_field(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Manning's-equation unit-discharge magnitude per cell [m^2/s]:
            q = (1/n) * h^(5/3) * sqrt(S)
        applied only where the cell is wet (h>0); zero elsewhere. This is
        a per-cell KINEMATIC approximation (each cell treated as a locally
        uniform-flow patch on the local terrain slope), not a routed 2D
        shallow-water solution -- adequate for giving the network a
        physically-grounded second state channel, not a substitute for a
        real 2D hydraulic model.
        """
        h = np.clip(depth_m, 0.0, None)
        q = (1.0 / self.manning_n) * h ** (5.0 / 3.0) * np.sqrt(self.slope_mag)
        return np.where(h > 0, q, 0.0)

    def _sigma_field(self) -> np.ndarray:
        """
        Structural regime field sigma(x): normalised HAND (Height Above
        Nearest Drainage), the natural flood-hazard analogue of SUPER DNS
        ONE's viscosity-contrast sigma -- low HAND (near the channel) is
        the high-hazard-sensitivity regime, high HAND (upland) is the
        low-sensitivity regime. Normalised to [0, 1] via min-max over the
        domain, with 0 = at the channel, 1 = furthest upland cell.
        """
        hand = self.elevation - self.drainage_elev_m
        hand_norm = (hand - hand.min()) / (hand.max() - hand.min() + 1e-8)
        return hand_norm

    def _extrude(self, field2d: np.ndarray) -> np.ndarray:
        return np.tile(field2d[:, :, None], (1, 1, self.grid.nz))

    def generate_snapshot(
        self,
        return_period_label: str,
        rainfall_depth_mm: float,
        duration_min: float,
        t_target_frac: float = 0.7,
        baseflow_frac_of_peak: float = 0.02,
        hyetograph_len: int = 64,
    ) -> Dict:
        """
        Runs one design storm through flood_one.py's pipeline and returns
        a SuperDNSSnapshot-shaped dict.

        t_target_frac in (0, 1]: fraction of the SCS triangular unit
        hydrograph's time base at which to take the u_tT snapshot (0.7 ~
        just past peak, a reasonable default "hazard" snapshot; pass 1.0
        for end-of-hydrograph, closer to peak-value use ~ time_to_peak/
        time_base for the true peak).
        """
        result = self.flood_one.run_design_storm(
            return_period_label, duration_min, rainfall_depth_mm)
        Q_peak = result["peak_discharge_m3_s"]
        y_normal_peak = result["normal_depth_m"]

        uh = self.watershed.scs_triangular_unit_hydrograph(result["runoff_depth_mm"])
        # (uh['base_time_hr'] available if a duration-aware snapshot timeline
        # is needed later; not required for the fixed-fraction scheme below.)

        # Discharge trajectory: simple triangular rise/fall to Q_peak, then
        # normal depth at each discharge via ChannelHydraulics.normal_depth
        # (reuses the exact same hydraulic relation flood_one.py already
        # validated for the peak; not a new/separate hydraulic model).
        def y_at_fraction(frac: float) -> float:
            frac = float(np.clip(frac, 0.0, 1.0))
            Q = Q_peak * (frac if frac <= 0.5 else (1.0 - frac) / 0.5 * 0.5 + 0.5 * (1 - (frac - 0.5) / 0.5))
            Q = max(Q, Q_peak * baseflow_frac_of_peak)
            return self.channel.normal_depth(Q)

        y_t0 = y_at_fraction(0.0)
        y_tT = y_at_fraction(t_target_frac)

        wse_t0 = self.drainage_elev_m + y_t0
        wse_tT = self.drainage_elev_m + max(y_tT, y_normal_peak * baseflow_frac_of_peak)

        h_t0, h_tT = self._depth_field(wse_t0), self._depth_field(wse_tT)
        q_t0, q_tT = self._discharge_field(h_t0), self._discharge_field(h_tT)

        u_t0 = np.stack([self._extrude(h_t0), self._extrude(q_t0)], axis=0)
        u_tT = np.stack([self._extrude(h_tT), self._extrude(q_tT)], axis=0)
        sigma = self._extrude(self._sigma_field())[None, ...]

        # Rainfall-intensity hyetograph as the single forcing channel:
        # use flood_one.RainfallModel's own Alternating-Block-Method
        # hyetograph (the same rainfall input already driving the runoff
        # computation above), converted from incremental depth to
        # intensity and resampled to hyetograph_len samples.
        dt_min = max(duration_min / 48.0, 1.0)
        hy = self.rainfall.design_hyetograph(duration_min, dt_min=dt_min)
        intensity_mm_hr = hy["incremental_depth_mm"] / dt_min * 60.0
        t_src = hy["time_min"]
        t_dst = np.linspace(t_src[0], t_src[-1], hyetograph_len)
        intensities = np.interp(t_dst, t_src, intensity_mm_hr)
        forcing_window = intensities[None, :]   # (1, T)

        return {
            "u_t0":     torch.from_numpy(u_t0).float(),
            "u_tT":     torch.from_numpy(u_tT).float(),
            "sigma":    torch.from_numpy(sigma).float(),
            "t_target": float(t_target_frac),
            "forcing_window": torch.from_numpy(np.ascontiguousarray(forcing_window)).float(),
            "metadata": {
                "return_period": return_period_label,
                "rainfall_depth_mm": rainfall_depth_mm,
                "duration_min": duration_min,
                "peak_discharge_m3_s": Q_peak,
                "dx_m": self.grid.dx_m,
                "grid_shape": (self.grid.nx, self.grid.ny, self.grid.nz),
                "state_channels": ("h", "q"),
                "source": "flood_dns_coupling_one",
                "version": __version__,
            },
        }

    def save_snapshot(self, snapshot: Dict, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, out_path)
        return out_path


# =====================================================================
# 4. DATASET-GENERATION DRIVER
# =====================================================================

def generate_flood_dataset(
    out_dir: str | Path,
    watershed: WatershedHydrology,
    channel: ChannelHydraulics,
    storms: Tuple[Tuple[str, float, float], ...],
    grid: Optional[FloodGridSpec] = None,
    n_terrain_variants: int = 4,
    manning_n: float = 0.04,
    seed: int = 0,
) -> int:
    """
    Generates an ensemble of flood snapshots: for each of
    `n_terrain_variants` synthetic terrain realisations, runs every
    (return_period_label, rainfall_depth_mm, duration_min) storm in
    `storms` and writes one .pt file per (terrain, storm) pair. Returns
    the number of files written.
    """
    grid = grid or FloodGridSpec()
    out_dir = Path(out_dir)
    n_written = 0
    for v in range(n_terrain_variants):
        terrain = synthetic_valley_terrain(grid.nx, grid.ny, grid.dx_m, seed=seed + v)
        coupling = FloodDNSCoupling(watershed, channel, grid,
                                     elevation_grid_m=terrain,
                                     manning_n=manning_n, seed=seed + v)
        for (label, depth_mm, dur_min) in storms:
            snap = coupling.generate_snapshot(label, depth_mm, dur_min)
            fname = f"flood_{label}_{depth_mm:.0f}mm_terrain{v}.pt"
            coupling.save_snapshot(snap, out_dir / fname)
            n_written += 1
    logger.info(f"Wrote {n_written} flood snapshots to {out_dir}")
    return n_written


if __name__ == "__main__":
    print(f"flood_dns_coupling_one.py v{__version__}")
    print("-" * 70)

    watershed = WatershedHydrology(
        curve_number=78, area_km2=25.0, flow_length_m=6000.0, avg_slope=0.01,
    )
    channel = ChannelHydraulics(
        bottom_width_m=8.0, side_slope_h_per_v=2.0, manning_n=0.035,
        channel_slope=0.002,
    )
    grid = FloodGridSpec(nx=32, ny=32, nz=4, dx_m=30.0)
    coupling = FloodDNSCoupling(watershed, channel, grid, seed=0)

    snap = coupling.generate_snapshot(
        return_period_label="100yr", rainfall_depth_mm=150.0, duration_min=180.0,
    )
    print(f"u_t0  shape: {tuple(snap['u_t0'].shape)}")
    print(f"u_tT  shape: {tuple(snap['u_tT'].shape)}")
    print(f"sigma shape: {tuple(snap['sigma'].shape)}")
    print(f"forcing_window shape: {tuple(snap['forcing_window'].shape)}")
    print(f"t_target: {snap['t_target']}")
    print(f"metadata: {snap['metadata']}")

    out_path = coupling.save_snapshot(snap, "/tmp/flood_snapshots/flood_100yr_150mm.pt")
    print(f"Saved: {out_path}")

    n = generate_flood_dataset(
        "/tmp/flood_snapshots", watershed, channel,
        storms=(("10yr", 80.0, 120.0), ("100yr", 150.0, 180.0), ("500yr", 220.0, 240.0)),
        grid=grid, n_terrain_variants=2, seed=1,
    )
    print(f"Ensemble generation wrote {n} snapshot files.")
