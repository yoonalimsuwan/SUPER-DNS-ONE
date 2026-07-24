# =============================================================================
# STORM DNS COUPLING ONE  —  v1.0
# Grids storm_one.py's parametric cyclone/surge output into SuperDNSSnapshot
# -compatible .pt training files for StructuralFNO3D.for_storm()
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
#   storm_one.py's TropicalCycloneModel (Holland gradient-wind radial
#   profile) and StormSurgeModel (inverse-barometer + 1D wind-setup) are
#   both explicitly-flagged-as-simplified PARAMETRIC / analytical models
#   — StormSurgeModel.total_surge_estimate() itself returns a single
#   SCALAR surge height, not a field, and its own docstring states it is
#   "NOT bathymetry/coastline-aware... use SLOSH/ADCIRC for real hazard
#   assessment." This module is the projection layer flagged in
#   structural_fno_3d_v2_4.py's changelog: it evaluates the cyclone's
#   radial wind field and the surge formula CELL-BY-CELL over a
#   synthetic coastal grid (using each cell's own distance-to-eye for
#   wind speed, and its own synthetic bathymetric depth for wind-setup),
#   producing a spatially-varying eta(x,y) field StructuralFNO3D.
#   for_storm() can train on. This is still a 1D-physics-per-cell
#   evaluation, NOT a real 2D hydrodynamic surge solve — it inherits
#   every limitation storm_one.py's own docstrings already flag (no
#   inter-cell momentum/continuity coupling, no real bathymetry/
#   coastline geometry). Do not use this for real coastal hazard
#   assessment; it exists to give StructuralFNO3D a physically-grounded
#   (if simplified) spatial target field to learn an operator against.
#
#   SCOPE NOTE: `synthetic_coastal_bathymetry()` below generates an
#   illustrative straight-coastline bathymetry (land above y=0, water
#   depth increasing linearly offshore) purely so the pipeline is
#   runnable end-to-end. Replace with real bathymetry (e.g. GEBCO,
#   resampled to the same grid) for anything beyond illustrative use.
#
#   3D EMBEDDING NOTE: storm surge is a 2D (sea-surface) phenomenon; as
#   in flood_dns_coupling_one.py, every 2D field here is extruded
#   (tiled, not physically stratified) across Nz thin layers purely for
#   shape compatibility with StructuralFNO3D's 3D grid.
#
# Output snapshot layout (matches SuperDNSSnapshot / StructuralFNO3D.for_storm()):
#   'u_t0'           : (1, Nx, Ny, Nz)  — eta before the storm (calm sea, ~0)
#   'u_tT'           : (1, Nx, Ny, Nz)  — eta at t_target into storm approach
#   'sigma'          : (1, Nx, Ny, Nz)  — normalised bathymetric depth (structural regime)
#   't_target'       : float             — normalised position in the approach timeline
#   'forcing_window' : (3, T)            — [wind_x, wind_y, pressure_deficit] at the coast
#   'metadata'       : dict              — cyclone/grid parameters
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
        "storm_dns_coupling_one.py requires PyTorch (for .pt snapshot output). "
        "Install with: pip install torch"
    ) from e

try:
    from storm_one import TropicalCycloneModel, StormSurgeModel, RHO_AIR_SEA_LEVEL
except ImportError as e:
    raise ImportError(
        "storm_dns_coupling_one.py requires storm_one.py to be importable "
        "(same directory or on PYTHONPATH)."
    ) from e

logger = logging.getLogger("storm_dns_coupling_one")
logging.basicConfig(level=logging.INFO)

__version__ = "1.0.0"


# =====================================================================
# 1. SYNTHETIC COASTAL BATHYMETRY  (replace with real bathymetry for real work)
# =====================================================================

def synthetic_coastal_bathymetry(nx: int, ny: int, dx_m: float,
                                  coast_row: Optional[int] = None,
                                  max_depth_m: float = 20.0,
                                  min_depth_m: float = 1.0) -> np.ndarray:
    """
    Illustrative straight-coastline bathymetry [m water depth, NaN-free
    -- land cells get `min_depth_m` as a floor so downstream formulas
    stay well-defined]: rows < coast_row are "land" (depth floored at
    min_depth_m, i.e. treated as a thin never-flooded fringe for this
    illustrative grid), rows >= coast_row are open water with depth
    increasing linearly offshore (a simple continental-shelf proxy).
    NOT a real bathymetry raster -- see module scope note.
    """
    coast_row = coast_row if coast_row is not None else ny // 3
    depth = np.full((nx, ny), min_depth_m)
    offshore_rows = np.arange(ny) - coast_row
    water_mask_1d = offshore_rows >= 0
    water_mask = np.tile(water_mask_1d[None, :], (nx, 1))
    ramp = np.clip(offshore_rows, 0, None) * dx_m / 20000.0   # deepens ~1m per 20km offshore
    depth_1d = min_depth_m + ramp * (max_depth_m - min_depth_m)
    depth_1d = np.clip(depth_1d, min_depth_m, max_depth_m)
    depth[:, water_mask_1d] = depth_1d[water_mask_1d][None, :]
    return depth, water_mask


# =====================================================================
# 2. GRID SPEC
# =====================================================================

@dataclass
class StormGridSpec:
    nx: int = 32
    ny: int = 32
    nz: int = 4
    dx_m: float = 2000.0   # coastal-scale cell size (2 km)


# =====================================================================
# 3. COUPLING
# =====================================================================

class StormDNSCoupling:
    """
    Orchestrates storm_one.py's TropicalCycloneModel + StormSurgeModel
    into a gridded eta(x,y) surge field for StructuralFNO3D.for_storm().
    """

    def __init__(self, cyclone: TropicalCycloneModel, grid: StormGridSpec,
                 water_depth_grid_m: Optional[np.ndarray] = None,
                 water_mask: Optional[np.ndarray] = None,
                 drag_coefficient: float = 0.0026):
        self.cyclone = cyclone
        self.grid = grid
        self.drag_coefficient = drag_coefficient
        if water_depth_grid_m is None:
            water_depth_grid_m, water_mask = synthetic_coastal_bathymetry(
                grid.nx, grid.ny, grid.dx_m)
        self.depth = water_depth_grid_m
        self.water_mask = (water_mask if water_mask is not None
                            else np.ones_like(water_depth_grid_m, dtype=bool))

        xs = (np.arange(grid.nx) - grid.nx // 2) * grid.dx_m
        ys = (np.arange(grid.ny) - grid.ny // 2) * grid.dx_m
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")

    def _wind_field(self, eye_x_m: float, eye_y_m: float) -> Dict[str, np.ndarray]:
        """
        Per-cell wind speed magnitude from the cyclone's Holland radial
        profile (TropicalCycloneModel.wind_speed), combined with a
        cyclonic tangential direction (same rotate-90-degree construction
        TornadoVortexModel uses for its Rankine vortex, reused here for
        directional consistency since storm_one.py does not itself
        expose a 2D vector wind field for the cyclone model).
        """
        dx, dy = self.X - eye_x_m, self.Y - eye_y_m
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)
        speed = self.cyclone.wind_speed(r)
        vx = -speed * np.sin(theta)
        vy = speed * np.cos(theta)
        return {"speed": speed, "vx": vx, "vy": vy, "r": r}

    def _eta_field(self, eye_x_m: float, eye_y_m: float) -> np.ndarray:
        """
        Per-cell total surge: uniform inverse-barometer term (a domain-
        wide sea-level rise, per storm_one.py's own definition) plus a
        per-cell wind-setup term using EACH CELL's own local wind speed
        and local water depth. Cells with assumption_valid=False (setup
        exceeds half local depth -- storm_one.py's own flagged failure
        mode) are clipped to that half-depth bound rather than allowed
        to return the unphysical unbounded value the raw formula would
        otherwise produce there.
        """
        wind = self._wind_field(eye_x_m, eye_y_m)
        ib = StormSurgeModel.inverse_barometer_setup(
            self.cyclone.Pc, self.cyclone.Pn)

        fetch_m = self.grid.dx_m * self.grid.ny * 0.5   # illustrative fixed fetch proxy
        eta = np.zeros_like(self.depth)
        flat_speed = wind["speed"].ravel()
        flat_depth = self.depth.ravel()
        flat_eta = np.zeros_like(flat_speed)
        for i in range(flat_speed.size):
            ws = StormSurgeModel.wind_setup(
                float(flat_speed[i]), fetch_m, float(flat_depth[i]),
                drag_coefficient=self.drag_coefficient)
            setup = ws["wind_setup_m"]
            if not ws["assumption_valid"]:
                setup = 0.5 * flat_depth[i]
            flat_eta[i] = setup
        eta = flat_eta.reshape(self.depth.shape) + ib
        eta[~self.water_mask] = 0.0   # no surge over "land" cells in this grid
        return eta

    def _sigma_field(self) -> np.ndarray:
        """Normalised local water depth -- shallow water is the high
        surge-sensitivity structural regime (wind-setup scales inversely
        with depth in storm_one.py's own formula)."""
        d = self.depth
        return (d - d.min()) / (d.max() - d.min() + 1e-8)

    def _extrude(self, field2d: np.ndarray) -> np.ndarray:
        return np.tile(field2d[:, :, None], (1, 1, self.grid.nz))

    def generate_snapshot(
        self,
        approach_fraction_t0: float = 0.1,
        approach_fraction_tT: float = 0.6,
        track_heading_deg: float = 0.0,
        forcing_len: int = 64,
    ) -> Dict:
        """
        Simulates the cyclone eye approaching along a straight track
        toward the domain center, at two points along that approach
        (t0 = far offshore / weak local wind at any given coastal cell,
        tT = closer / peak local impact), and returns a SuperDNSSnapshot
        -shaped dict. `approach_fraction` in [0,1]: 0 = eye at domain
        edge (far), 1 = eye at domain center.

        NOTE: storm_one.py does not itself provide a track/timing model,
        so the two-point "approach" here is a simple straight-line
        interpolation between an offshore and an onshore eye position --
        an illustrative timeline construction, not a real forecast track.
        """
        heading = math.radians(track_heading_deg)
        track_len_m = self.grid.dx_m * max(self.grid.nx, self.grid.ny)
        start = np.array([-track_len_m * math.cos(heading), -track_len_m * math.sin(heading)])
        end = np.array([0.0, 0.0])

        def eye_at(frac: float) -> Tuple[float, float]:
            p = start + frac * (end - start)
            return float(p[0]), float(p[1])

        ex0, ey0 = eye_at(approach_fraction_t0)
        exT, eyT = eye_at(approach_fraction_tT)

        eta_t0 = self._eta_field(ex0, ey0)
        eta_tT = self._eta_field(exT, eyT)
        sigma = self._extrude(self._sigma_field())[None, ...]

        u_t0 = self._extrude(eta_t0)[None, ...]
        u_tT = self._extrude(eta_tT)[None, ...]

        # Forcing window: [wind_x, wind_y, pressure_deficit] sampled at the
        # nearest-to-coast water cell along the track, across the approach.
        coast_cell = tuple(np.argwhere(self.water_mask)[0]) if self.water_mask.any() else (0, 0)
        fracs = np.linspace(approach_fraction_t0, approach_fraction_tT, forcing_len)
        wind_x = np.zeros(forcing_len)
        wind_y = np.zeros(forcing_len)
        dp = np.full(forcing_len, (self.cyclone.Pn - self.cyclone.Pc))
        cx, cy = self.X[coast_cell], self.Y[coast_cell]
        for i, f in enumerate(fracs):
            ex, ey = eye_at(float(f))
            dxc, dyc = cx - ex, cy - ey
            r = math.hypot(dxc, dyc)
            speed = float(self.cyclone.wind_speed(np.array([r]))[0])
            th = math.atan2(dyc, dxc)
            wind_x[i] = -speed * math.sin(th)
            wind_y[i] = speed * math.cos(th)
        forcing_window = np.stack([wind_x, wind_y, dp], axis=0)   # (3, T)

        return {
            "u_t0":     torch.from_numpy(u_t0).float(),
            "u_tT":     torch.from_numpy(u_tT).float(),
            "sigma":    torch.from_numpy(sigma).float(),
            "t_target": float(approach_fraction_tT),
            "forcing_window": torch.from_numpy(np.ascontiguousarray(forcing_window)).float(),
            "metadata": {
                "central_pressure_Pa": self.cyclone.Pc,
                "ambient_pressure_Pa": self.cyclone.Pn,
                "radius_max_wind_m": self.cyclone.Rmax,
                "track_heading_deg": track_heading_deg,
                "dx_m": self.grid.dx_m,
                "grid_shape": (self.grid.nx, self.grid.ny, self.grid.nz),
                "state_channels": ("eta",),
                "source": "storm_dns_coupling_one",
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

def generate_storm_dataset(
    out_dir: str | Path,
    cyclones: Tuple[Tuple[str, float, float, float, float], ...],
    grid: Optional[StormGridSpec] = None,
) -> int:
    """
    `cyclones`: tuples of (label, central_pressure_Pa, ambient_pressure_Pa,
    radius_max_wind_m, latitude_deg). Writes one .pt file per cyclone.
    """
    grid = grid or StormGridSpec()
    out_dir = Path(out_dir)
    depth, mask = synthetic_coastal_bathymetry(grid.nx, grid.ny, grid.dx_m)
    n_written = 0
    for (label, pc, pn, rmax, lat) in cyclones:
        cyclone = TropicalCycloneModel(pc, pn, rmax, lat)
        coupling = StormDNSCoupling(cyclone, grid, water_depth_grid_m=depth, water_mask=mask)
        snap = coupling.generate_snapshot()
        coupling.save_snapshot(snap, out_dir / f"storm_{label}.pt")
        n_written += 1
    logger.info(f"Wrote {n_written} storm snapshots to {out_dir}")
    return n_written


if __name__ == "__main__":
    print(f"storm_dns_coupling_one.py v{__version__}")
    print("-" * 70)

    cyclone = TropicalCycloneModel(
        central_pressure_Pa=94000.0, ambient_pressure_Pa=101300.0,
        radius_max_wind_m=40000.0, latitude_deg=25.0,
    )
    grid = StormGridSpec(nx=32, ny=32, nz=4, dx_m=2000.0)
    coupling = StormDNSCoupling(cyclone, grid)

    snap = coupling.generate_snapshot()
    print(f"u_t0  shape: {tuple(snap['u_t0'].shape)}")
    print(f"u_tT  shape: {tuple(snap['u_tT'].shape)}")
    print(f"sigma shape: {tuple(snap['sigma'].shape)}")
    print(f"forcing_window shape: {tuple(snap['forcing_window'].shape)}")
    print(f"t_target: {snap['t_target']}")
    print(f"metadata: {snap['metadata']}")

    out_path = coupling.save_snapshot(snap, "/tmp/storm_snapshots/storm_cat3_demo.pt")
    print(f"Saved: {out_path}")

    n = generate_storm_dataset(
        "/tmp/storm_snapshots",
        cyclones=(
            ("cat1", 98000.0, 101300.0, 50000.0, 22.0),
            ("cat3", 94000.0, 101300.0, 40000.0, 25.0),
            ("cat5", 88000.0, 101300.0, 25000.0, 20.0),
        ),
        grid=grid,
    )
    print(f"Ensemble generation wrote {n} snapshot files.")
