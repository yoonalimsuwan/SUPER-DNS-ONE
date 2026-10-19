"""
sesi_openfoam_bridge.py
=============================================================================
SESI OPENFOAM BRIDGE
Connects OpenFOAM CFD fields to the SESI Universal Connector.

Supported sources:
  1. NPZ file with keys:
       points : optional (N, 3)
       U      : velocity field
       p      : pressure field
       T      : temperature field
       rho    : density field

     Structured format:
       U   : (D, H, W, 3)
       p   : (D, H, W)
       T   : (D, H, W)
       rho : (D, H, W)

     Unstructured format:
       points : (N, 3)
       U      : (N, 3)
       p      : (N,)
       T      : (N,)
       rho    : (N,)

  2. HDF5 file with the same keys.

  3. VTK / VTU file exported from OpenFOAM, for example via:
       foamToVTK -latestTime

Output:
  StandardSample compatible with sesi_universal_connector.py

Optional dependencies:
  pip install numpy torch
  pip install scipy        # better interpolation
  pip install h5py         # HDF5 support
  pip install pyvista      # VTK/VTU support

Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
License   : MIT
Year      : 2026
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

#------------------------------------------------------------------------------
# Optional dependencies
#------------------------------------------------------------------------------
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    pv = None
    _HAS_PYVISTA = False

try:
    import h5py
    _HAS_H5PY = True
except Exception:
    h5py = None
    _HAS_H5PY = False

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    _HAS_SCIPY = True
except Exception:
    LinearNDInterpolator = None
    NearestNDInterpolator = None
    _HAS_SCIPY = False


#------------------------------------------------------------------------------
# StandardSample import with fallback
#------------------------------------------------------------------------------
try:
    from sesi_universal_connector import StandardSample
    _CONNECTOR_AVAILABLE = True
except Exception:
    _CONNECTOR_AVAILABLE = False

    @dataclass
    class StandardSample:
        """
        Minimal fallback StandardSample. If sesi_universal_connector.py is
        available, its StandardSample will be used instead.
        """

        # Core fields
        velocity_field: Optional[torch.Tensor] = None
        thermal_field: Optional[torch.Tensor] = None
        pressure_field: Optional[torch.Tensor] = None
        density_field: Optional[torch.Tensor] = None
        em_field: Optional[torch.Tensor] = None
        emf_coupling: Optional[torch.Tensor] = None
        sigma_skin: Optional[torch.Tensor] = None

        # Flight
        flight_state: Optional[torch.Tensor] = None
        delta_e_cmd: Optional[torch.Tensor] = None
        thrust: Optional[torch.Tensor] = None

        # EW
        ew_raw: Optional[torch.Tensor] = None
        ew_jam: Optional[torch.Tensor] = None

        # HCSN
        hcsn_signal: Optional[torch.Tensor] = None
        hcsn_target: Optional[torch.Tensor] = None

        # QMD
        qmd_D: Optional[torch.Tensor] = None
        qmd_E: Optional[torch.Tensor] = None

        # SPTC
        sptc_substrate: Optional[torch.Tensor] = None

        # Countermeasure
        target_position: Optional[torch.Tensor] = None
        target_velocity: Optional[torch.Tensor] = None
        target_mass: Optional[torch.Tensor] = None
        predicted_trajectory: Optional[torch.Tensor] = None
        gravimeter_weights: Optional[torch.Tensor] = None
        muon_sensor_grid: Optional[torch.Tensor] = None
        dew_focal_controls: Optional[torch.Tensor] = None

        def shapes(self) -> Dict[str, Tuple[int, ...]]:
            out = {}
            for k, v in self.__dict__.items():
                if torch.is_tensor(v):
                    out[k] = tuple(v.shape)
            return out

        def validate(self) -> Dict[str, Any]:
            errors = []
            batch_size = None

            if self.velocity_field is not None:
                batch_size = self.velocity_field.shape[0]

            core_5d = [
                self.velocity_field,
                self.thermal_field,
                self.pressure_field,
                self.density_field,
                self.em_field,
                self.emf_coupling,
                self.sigma_skin,
            ]

            for t in core_5d:
                if t is None:
                    continue
                if t.dim() != 5:
                    errors.append(f"tensor shape {tuple(t.shape)} is not 5D")
                if batch_size is None:
                    batch_size = t.shape[0]
                elif t.shape[0] != batch_size:
                    errors.append("batch mismatch")

            return {
                "ok": len(errors) == 0,
                "batch_size": batch_size,
                "errors": errors,
            }


#------------------------------------------------------------------------------
# Defaults and constants
#------------------------------------------------------------------------------
DEFAULT_T_K = 300.0
DEFAULT_P_PA = 101325.0
DEFAULT_RHO_KG_M3 = 1.225

_FIELD_NAMES = ("U", "p", "T", "rho")


#------------------------------------------------------------------------------
# Basic helpers
#------------------------------------------------------------------------------
def _as_float_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _pad_or_slice_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad or slice the last dimension to exactly `dim`."""
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]

    pad_size = dim - x.shape[-1]
    return F.pad(x, (0, pad_size))


def _repeat_batch(sample: StandardSample, batch_size: int) -> StandardSample:
    if batch_size <= 1:
        return sample

    updates = {}
    for k, v in sample.__dict__.items():
        if torch.is_tensor(v):
            reps = (batch_size,) + (1,) * max(0, v.dim() - 1)
            updates[k] = v.repeat(*reps)

    return replace(sample, **updates)


#------------------------------------------------------------------------------
# Structured array conversion
#------------------------------------------------------------------------------
def _scalar_to_bcdhw(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert scalar array to [1, 1, D, H, W].

    Accepted shapes:
      (D, H, W)
      (1, D, H, W)
      (D, H, W, 1)
    """
    arr = _as_float_np(arr)

    if arr.ndim == 3:
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 4:
        if arr.shape[0] == 1:
            t = torch.from_numpy(arr)
        elif arr.shape[-1] == 1:
            t = torch.from_numpy(arr[..., 0]).unsqueeze(0).unsqueeze(0)
        else:
            # Fallback: take first slice or mean over small last axis.
            if arr.shape[-1] <= 4:
                t = torch.from_numpy(arr.mean(axis=-1)).unsqueeze(0).unsqueeze(0)
            else:
                t = torch.from_numpy(arr[0:1])
    else:
        raise ValueError(f"Unsupported scalar field shape: {arr.shape}")

    if t.dim() == 4:
        t = t.unsqueeze(0)

    return t.to(device=device, dtype=dtype)


def _vector_to_bcdhw(
    arr: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    channels: int = 3,
) -> torch.Tensor:
    """
    Convert vector array to [1, C, D, H, W].

    Accepted shapes:
      (D, H, W, C)
      (C, D, H, W)
    """
    arr = _as_float_np(arr)

    if arr.ndim != 4:
        raise ValueError(f"Unsupported vector field shape: {arr.shape}")

    if arr.shape[-1] == channels:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] == channels:
        pass
    else:
        raise ValueError(
            f"Vector field must have {channels} channels; got shape {arr.shape}"
        )

    return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)


#------------------------------------------------------------------------------
# Interpolation from unstructured points to regular grid
#------------------------------------------------------------------------------
def _interpolate_points_to_grid(
    points: np.ndarray,
    values: np.ndarray,
    grid_shape: Tuple[int, int, int],
    bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Interpolate scattered points to a regular grid.

    Args:
        points: (N, 3)
        values: (N,) or (N, C)
        grid_shape: (nx, ny, nz)
        bbox: optional ((xmin,ymin,zmin), (xmax,ymax,zmax))
        fill_value: value outside convex hull

    Returns:
        If values is scalar: (nx, ny, nz)
        If values is vector: (nx, ny, nz, C)
    """
    points = _as_float_np(points)
    values = _as_float_np(values)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if values.ndim == 1:
        values = values[:, None]

    if values.shape[0] != points.shape[0]:
        raise ValueError("points and values must have same number of samples")

    nx, ny, nz = grid_shape

    if bbox is None:
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        pad = np.maximum((pmax - pmin) * 0.02, 1e-6)
        pmin = pmin - pad
        pmax = pmax + pad
    else:
        pmin = np.asarray(bbox[0], dtype=np.float32)
        pmax = np.asarray(bbox[1], dtype=np.float32)

    xs = np.linspace(pmin[0], pmax[0], nx, dtype=np.float32)
    ys = np.linspace(pmin[1], pmax[1], ny, dtype=np.float32)
    zs = np.linspace(pmin[2], pmax[2], nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    out = np.full((grid_points.shape[0], values.shape[1]), float(fill_value), dtype=np.float32)

    # -------------------------------------------------------------------------
    # scipy path
    # -------------------------------------------------------------------------
    if _HAS_SCIPY and points.shape[0] >= 8:
        for c in range(values.shape[1]):
            try:
                interp = LinearNDInterpolator(points, values[:, c], fill_value=np.nan)
                val = interp(grid_points)

                if np.isnan(val).any():
                    nearest = NearestNDInterpolator(points, values[:, c])
                    nan_mask = np.isnan(val)
                    val[nan_mask] = nearest(grid_points[nan_mask])

                out[:, c] = np.nan_to_num(val, nan=fill_value, posinf=fill_value, neginf=fill_value)
            except Exception:
                try:
                    nearest = NearestNDInterpolator(points, values[:, c])
                    out[:, c] = nearest(grid_points)
                except Exception:
                    out[:, c] = fill_value

    # -------------------------------------------------------------------------
    # torch nearest fallback
    # -------------------------------------------------------------------------
    else:
        pts_t = torch.from_numpy(points).float()
        vals_t = torch.from_numpy(values).float()
        grid_t = torch.from_numpy(grid_points).float()

        chunk = 4096
        for i in range(0, grid_t.shape[0], chunk):
            g = grid_t[i : i + chunk]
            d = torch.cdist(g, pts_t)
            idx = d.argmin(dim=1)
            out[i : i + chunk] = vals_t[idx].numpy()

    out = out.reshape(nx, ny, nz, values.shape[1])

    if out.shape[-1] == 1:
        return out[..., 0]

    return out


#------------------------------------------------------------------------------
# File loaders
#------------------------------------------------------------------------------
def load_npz_fields(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.keys()}


def load_h5_fields(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    if not _HAS_H5PY:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")

    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    fields = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            fields[k] = np.asarray(f[k])

    return fields


def load_vtk_fields(
    path: Union[str, Path],
    field_names: Tuple[str, ...] = _FIELD_NAMES,
) -> Dict[str, np.ndarray]:
    if not _HAS_PYVISTA:
        raise ImportError("pyvista is required for VTK/VTU support. Install with: pip install pyvista")

    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mesh = pv.read(path)

    # If MultiBlock, try to combine.
    if hasattr(mesh, "n_blocks"):
        try:
            mesh = mesh.combine()
        except Exception:
            pass

    # Prefer point data. Convert cell data if needed.
    try:
        if len(mesh.cell_data.keys()) > 0:
            mesh = mesh.cell_data_to_point_data()
    except Exception:
        pass

    fields: Dict[str, np.ndarray] = {
        "points": np.asarray(mesh.points, dtype=np.float32),
    }

    for name in field_names:
        arr = None

        if name in mesh.point_data:
            arr = np.asarray(mesh.point_data[name], dtype=np.float32)
        elif name in mesh.cell_data:
            arr = np.asarray(mesh.cell_data[name], dtype=np.float32)

        if arr is not None:
            fields[name] = arr

    return fields


#------------------------------------------------------------------------------
# Build StandardSample
#------------------------------------------------------------------------------
def _fields_to_standard_sample(
    fields: Dict[str, np.ndarray],
    grid_shape: Tuple[int, int, int],
    bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    device: torch.device,
    dtype: torch.dtype,
) -> StandardSample:
    """
    Convert loaded OpenFOAM-like fields into StandardSample.

    Expected fields:
      U   velocity
      p   pressure
      T   temperature
      rho density
    """
    # --------------------------------------------------------------------------
    # Unstructured point-cloud path
    # --------------------------------------------------------------------------
    if "points" in fields and "U" in fields and fields["U"].ndim == 2:
        points = fields["points"]

        U_grid = _interpolate_points_to_grid(points, fields["U"], grid_shape, bbox, fill_value=0.0)

        if "p" in fields and fields["p"].ndim == 1:
            p_grid = _interpolate_points_to_grid(points, fields["p"], grid_shape, bbox, fill_value=DEFAULT_P_PA)
        else:
            p_grid = np.full(grid_shape, DEFAULT_P_PA, dtype=np.float32)

        if "T" in fields and fields["T"].ndim == 1:
            T_grid = _interpolate_points_to_grid(points, fields["T"], grid_shape, bbox, fill_value=DEFAULT_T_K)
        else:
            T_grid = np.full(grid_shape, DEFAULT_T_K, dtype=np.float32)

        if "rho" in fields and fields["rho"].ndim == 1:
            rho_grid = _interpolate_points_to_grid(points, fields["rho"], grid_shape, bbox, fill_value=DEFAULT_RHO_KG_M3)
        else:
            rho_grid = np.full(grid_shape, DEFAULT_RHO_KG_M3, dtype=np.float32)

        velocity_field = _vector_to_bcdhw(U_grid, device=device, dtype=dtype, channels=3)
        pressure_field = _scalar_to_bcdhw(p_grid, device=device, dtype=dtype)
        thermal_field = _scalar_to_bcdhw(T_grid, device=device, dtype=dtype)
        density_field = _scalar_to_bcdhw(rho_grid, device=device, dtype=dtype)

    # --------------------------------------------------------------------------
    # Structured grid path
    # --------------------------------------------------------------------------
    else:
        if "U" in fields:
            U_arr = fields["U"]
        elif "p" in fields and fields["p"].ndim == 3:
            U_arr = np.zeros((*fields["p"].shape, 3), dtype=np.float32)
        else:
            U_arr = np.zeros((*grid_shape, 3), dtype=np.float32)

        velocity_field = _vector_to_bcdhw(U_arr, device=device, dtype=dtype, channels=3)
        spatial = tuple(velocity_field.shape[2:])

        def scalar_field(name: str, default: float) -> torch.Tensor:
            if name not in fields:
                arr = np.full(spatial, default, dtype=np.float32)
            else:
                arr = _as_float_np(fields[name])

            t = _scalar_to_bcdhw(arr, device=device, dtype=dtype)

            if tuple(t.shape[2:]) != spatial:
                t = F.interpolate(t, size=spatial, mode="trilinear", align_corners=False)

            return t

        pressure_field = scalar_field("p", DEFAULT_P_PA)
        thermal_field = scalar_field("T", DEFAULT_T_K)
        density_field = scalar_field("rho", DEFAULT_RHO_KG_M3)

    # --------------------------------------------------------------------------
    # Derived/default EM and material fields
    # --------------------------------------------------------------------------
    em_field = torch.zeros_like(velocity_field)
    emf_coupling = torch.zeros_like(thermal_field)
    sigma_skin = torch.ones_like(thermal_field)

    # Sanitize
    thermal_field = thermal_field.clamp_min(1.0)
    pressure_field = pressure_field.clamp_min(1.0)
    density_field = density_field.clamp_min(1e-6)
    sigma_skin = sigma_skin.clamp_min(1e-6)

    return StandardSample(
        velocity_field=velocity_field,
        thermal_field=thermal_field,
        pressure_field=pressure_field,
        density_field=density_field,
        em_field=em_field,
        emf_coupling=emf_coupling,
        sigma_skin=sigma_skin,
    )


#------------------------------------------------------------------------------
# Auxiliary augmentation for full connector compatibility
#------------------------------------------------------------------------------
def augment_standard_sample(
    sample: StandardSample,
    ew_dim: int = 64,
    hcsn_len: int = 16,
    hcsn_d_in: int = 16,
    hcsn_d_out: int = 16,
    qmd_len: int = 8,
    qmd_d_D: int = 8,
    qmd_d_E: int = 4,
    sptc_dim: int = 12,
    horizon: int = 12,
) -> StandardSample:
    """
    Fill missing optional fields with deterministic/synthetic values so that
    the universal connector can execute more modules.

    Replace these with real data whenever available.
    """
    if sample.velocity_field is None:
        return sample

    updates = {}
    B = sample.velocity_field.shape[0]
    device = sample.velocity_field.device
    dtype = sample.velocity_field.dtype

    # EM/material defaults
    if sample.em_field is None:
        updates["em_field"] = torch.zeros_like(sample.velocity_field)
    if sample.emf_coupling is None:
        updates["emf_coupling"] = torch.zeros_like(sample.thermal_field)
    if sample.sigma_skin is None:
        updates["sigma_skin"] = torch.ones_like(sample.thermal_field)

    # Compact pooled physics feature
    feat = sample.velocity_field.pow(2).mean(dim=(2, 3, 4))  # [B, 3]

    # Flight defaults
    if sample.flight_state is None:
        flight_state = torch.zeros(B, 12, device=device, dtype=dtype)
        flight_state[:, 2] = 30000.0
        flight_state[:, 3] = 2000.0
        updates["flight_state"] = flight_state

    if sample.delta_e_cmd is None:
        updates["delta_e_cmd"] = torch.zeros(B, device=device, dtype=dtype)

    if sample.thrust is None:
        updates["thrust"] = torch.full((B,), 5000.0, device=device, dtype=dtype)

    # EW defaults
    if sample.ew_raw is None:
        updates["ew_raw"] = feat.mean(dim=-1, keepdim=True).repeat(1, ew_dim)
    if sample.ew_jam is None:
        updates["ew_jam"] = 0.1 * feat.mean(dim=-1, keepdim=True).repeat(1, ew_dim)

    # HCSN defaults
    if sample.hcsn_signal is None:
        sig = feat.unsqueeze(1).repeat(1, hcsn_len, 1)
        sig = _pad_or_slice_last(sig, hcsn_d_in)
        updates["hcsn_signal"] = sig

    if sample.hcsn_target is None:
        target = torch.tanh(updates.get("hcsn_signal", sample.hcsn_signal))
        target = _pad_or_slice_last(target, hcsn_d_out)
        updates["hcsn_target"] = target

    # QMD defaults
    if sample.qmd_D is None:
        qd = feat.unsqueeze(1).repeat(1, qmd_len, 1)
        qd = _pad_or_slice_last(qd, qmd_d_D)
        updates["qmd_D"] = qd

    if sample.qmd_E is None:
        qe = feat.mean(dim=-1, keepdim=True).unsqueeze(1).repeat(1, qmd_len, 1)
        qe = _pad_or_slice_last(qe, qmd_d_E)
        updates["qmd_E"] = qe

    # SPTC defaults
    if sample.sptc_substrate is None:
        updates["sptc_substrate"] = _pad_or_slice_last(feat, sptc_dim)

    # Countermeasure defaults
    if sample.target_position is None:
        updates["target_position"] = torch.tensor(
            [[1.0e5, 2.0e5, 3.0e4]],
            device=device,
            dtype=dtype,
        ).repeat(B, 1)

    if sample.target_velocity is None:
        updates["target_velocity"] = torch.tensor(
            [[2000.0, 500.0, -100.0]],
            device=device,
            dtype=dtype,
        ).repeat(B, 1)

    if sample.target_mass is None:
        updates["target_mass"] = torch.tensor([1500.0], device=device, dtype=dtype).repeat(B)

    if sample.predicted_trajectory is None:
        pos = updates.get("target_position", sample.target_position)
        updates["predicted_trajectory"] = pos.unsqueeze(1).repeat(1, horizon, 1)

    if sample.gravimeter_weights is None:
        updates["gravimeter_weights"] = torch.ones(4, device=device, dtype=dtype)

    if sample.muon_sensor_grid is None:
        updates["muon_sensor_grid"] = torch.ones(4, device=device, dtype=dtype)

    if sample.dew_focal_controls is None:
        pos = updates.get("target_position", sample.target_position)
        updates["dew_focal_controls"] = pos + 100.0

    return replace(sample, **updates)


#------------------------------------------------------------------------------
# Main adapter
#------------------------------------------------------------------------------
class OpenFOAMFieldAdapter:
    """
    Adapter that loads OpenFOAM fields and produces a StandardSample.

    Example:
        adapter = OpenFOAMFieldAdapter(
            source_type="npz",
            path="openfoam_fields.npz",
            grid_shape=(32, 32, 32),
        )
        sample = adapter.to_standard_sample(augment=True)
    """

    def __init__(
        self,
        source_type: str = "npz",
        path: Optional[Union[str, Path]] = None,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.source_type = source_type.lower().strip()
        self.path = Path(path) if path is not None else None
        self.grid_shape = tuple(grid_shape)
        self.bbox = bbox
        self.device = torch.device(device)
        self.dtype = dtype

        if len(self.grid_shape) != 3:
            raise ValueError("grid_shape must be (D, H, W)")

    #--------------------------------------------------------------------------
    def load_fields(self) -> Dict[str, np.ndarray]:
        if self.path is None:
            raise ValueError("path is required")

        if self.source_type in {"npz", "numpy"}:
            return load_npz_fields(self.path)

        if self.source_type in {"h5", "hdf5"}:
            return load_h5_fields(self.path)

        if self.source_type in {"vtk", "vtu", "openfoam-vtk"}:
            return load_vtk_fields(self.path, field_names=_FIELD_NAMES)

        raise ValueError(f"Unsupported source_type: {self.source_type}")

    #--------------------------------------------------------------------------
    def to_standard_sample(
        self,
        batch_size: int = 1,
        augment: bool = False,
        **augment_kwargs: Any,
    ) -> StandardSample:
        fields = self.load_fields()

        sample = _fields_to_standard_sample(
            fields=fields,
            grid_shape=self.grid_shape,
            bbox=self.bbox,
            device=self.device,
            dtype=self.dtype,
        )

        if augment:
            sample = augment_standard_sample(sample, **augment_kwargs)

        sample = _repeat_batch(sample, batch_size)

        return sample


#------------------------------------------------------------------------------
# Save utilities
#------------------------------------------------------------------------------
def save_standard_sample_npz(sample: StandardSample, path: Union[str, Path]) -> None:
    """
    Save the first batch item of a StandardSample to NPZ.

    Saved keys:
      U   : (D, H, W, 3)
      p   : (D, H, W)
      T   : (D, H, W)
      rho : (D, H, W)
    """
    arrays = {}

    if sample.velocity_field is not None:
        arrays["U"] = (
            sample.velocity_field[0]
            .permute(1, 2, 3, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if sample.pressure_field is not None:
        arrays["p"] = (
            sample.pressure_field[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if sample.thermal_field is not None:
        arrays["T"] = (
            sample.thermal_field[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if sample.density_field is not None:
        arrays["rho"] = (
            sample.density_field[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    np.savez_compressed(str(path), **arrays)


#------------------------------------------------------------------------------
# Connector helper
#------------------------------------------------------------------------------
def run_with_universal_connector(
    sample: StandardSample,
    connector: Optional[Any] = None,
    module_paths: Optional[Dict[str, str]] = None,
    build: bool = True,
) -> Dict[str, Any]:
    """
    Run a StandardSample through sesi_universal_connector.UniversalSESIConnector.
    """
    if connector is None:
        from sesi_universal_connector import UniversalSESIConnector

        connector = UniversalSESIConnector(
            module_paths=module_paths,
            build=build,
        )

    return connector(sample)


#------------------------------------------------------------------------------
# Self-test
#------------------------------------------------------------------------------
def run_selftest() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "connector_available": _CONNECTOR_AVAILABLE,
        "pyvista_available": _HAS_PYVISTA,
        "h5py_available": _HAS_H5PY,
        "scipy_available": _HAS_SCIPY,
        "tests": {},
    }

    torch.manual_seed(0)
    np.random.seed(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ----------------------------------------------------------------------
        # Structured NPZ test
        # ----------------------------------------------------------------------
        D, H, W = 8, 8, 8

        U = np.random.randn(D, H, W, 3).astype(np.float32) * 100.0
        p = np.random.rand(D, H, W).astype(np.float32) * 1000.0 + 1000.0
        T = np.random.rand(D, H, W).astype(np.float32) * 40.0 + 250.0
        rho = np.random.rand(D, H, W).astype(np.float32) * 0.2 + 0.1

        structured_path = os.path.join(tmpdir, "structured.npz")
        np.savez_compressed(structured_path, U=U, p=p, T=T, rho=rho)

        adapter = OpenFOAMFieldAdapter(
            source_type="npz",
            path=structured_path,
            grid_shape=(D, H, W),
            device="cpu",
            dtype=torch.float32,
        )

        sample = adapter.to_standard_sample(batch_size=1, augment=True)

        report["tests"]["structured_npz"] = {
            "shapes": sample.shapes(),
            "validation": sample.validate(),
        }

        # ----------------------------------------------------------------------
        # Unstructured NPZ test
        # ----------------------------------------------------------------------
        N = 400
        points = np.random.rand(N, 3).astype(np.float32)
        points[:, 0] *= 10.0
        points[:, 1] *= 5.0
        points[:, 2] *= 2.0

        U_pts = np.random.randn(N, 3).astype(np.float32) * 80.0
        p_pts = np.random.rand(N).astype(np.float32) * 900.0 + 500.0
        T_pts = np.random.rand(N).astype(np.float32) * 50.0 + 220.0
        rho_pts = np.random.rand(N).astype(np.float32) * 0.3 + 0.05

        unstructured_path = os.path.join(tmpdir, "unstructured.npz")
        np.savez_compressed(
            unstructured_path,
            points=points,
            U=U_pts,
            p=p_pts,
            T=T_pts,
            rho=rho_pts,
        )

        adapter_unstructured = OpenFOAMFieldAdapter(
            source_type="npz",
            path=unstructured_path,
            grid_shape=(8, 8, 8),
            device="cpu",
            dtype=torch.float32,
        )

        sample_u = adapter_unstructured.to_standard_sample(batch_size=1, augment=True)

        report["tests"]["unstructured_npz"] = {
            "shapes": sample_u.shapes(),
            "validation": sample_u.validate(),
        }

        # ----------------------------------------------------------------------
        # Save/load roundtrip test
        # ----------------------------------------------------------------------
        out_path = os.path.join(tmpdir, "saved.npz")
        save_standard_sample_npz(sample, out_path)

        adapter_saved = OpenFOAMFieldAdapter(
            source_type="npz",
            path=out_path,
            grid_shape=(D, H, W),
            device="cpu",
            dtype=torch.float32,
        )

        sample_saved = adapter_saved.to_standard_sample(batch_size=1, augment=False)

        report["tests"]["save_load_roundtrip"] = {
            "shapes": sample_saved.shapes(),
            "validation": sample_saved.validate(),
        }

    return report


#------------------------------------------------------------------------------
# CLI
#------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SESI OpenFOAM bridge: convert OpenFOAM fields to StandardSample."
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["npz", "h5", "hdf5", "vtk", "vtu"],
        default=None,
        help="Input source type.",
    )

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to input file.",
    )

    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        metavar=("D", "H", "W"),
        help="Target grid shape.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size.",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Fill missing optional SESI fields with synthetic defaults.",
    )

    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="Optional path to save converted StandardSample as NPZ.",
    )

    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run built-in self-test.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.selftest or args.source is None:
        report = run_selftest()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.path is None:
        raise ValueError("--path is required unless --selftest is used.")

    adapter = OpenFOAMFieldAdapter(
        source_type=args.source,
        path=args.path,
        grid_shape=tuple(args.grid),
        device="cpu",
        dtype=torch.float32,
    )

    sample = adapter.to_standard_sample(
        batch_size=args.batch_size,
        augment=args.augment,
    )

    print("StandardSample shapes:")
    print(json.dumps(sample.shapes(), indent=2, default=str))

    print("\nValidation:")
    print(json.dumps(sample.validate(), indent=2, default=str))

    if args.save_npz:
        save_standard_sample_npz(sample, args.save_npz)
        print(f"\nSaved NPZ to: {args.save_npz}")


if __name__ == "__main__":
    main()
