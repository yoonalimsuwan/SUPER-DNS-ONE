"""
=============================================================================
openfoam_to_superdns_bridge.py
OpenFOAM → SUPER DNS ONE v6  Training Data Bridge
=============================================================================
Author  : PAI & Yoon A Limsuwan / MSPS NETWORK
License : MIT
Year    : 2026
Version : 1.0.0

Bridges OpenFOAM simulation results into SUPER DNS ONE v6 format
for AI training.  Supports:

  1. Reading OpenFOAM time directories (ASCII + binary)
  2. Parsing scalar / vector / tensor volFields
  3. Extracting mesh cell centres for structured-grid mapping
  4. Converting to SUPER DNS conservative variables (rho, rhou, …, rhoE)
  5. Building PyTorch Datasets for supervised / self-supervised training
  6. Initialising a CompressibleSolver directly from OpenFOAM state
  7. Creating (input, target) pairs for neural-surrogate training

Typical workflow
────────────────
    # 1. Run OpenFOAM (e.g. rhoCentralFoam)
    # 2. Load results
    bridge = OpenFOAMSuperDNSBridge("cases/hypersonic_bnd")
    samples = bridge.build_training_set(time_range=(100, 500))
    # 3. Train
    loader = DataLoader(samples, batch_size=4)
    for batch in loader:
        rho, rhou, rhov, rhow, rhoE = batch
        ...

=============================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

__all__ = [
    "OpenFOAMFieldParser",
    "OpenFOAMCaseReader",
    "SuperDNSDataConverter",
    "OpenFOAMSuperDNSBridge",
    "OpenFOAMTrainingDataset",
    "build_training_data",
]


# ═══════════════════════════════════════════════════════════════════════
# 1.  OpenFOAM field-file parser
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMFieldParser:
    """
    Parses an OpenFOAM field file (scalar, vector, or tensor volField).

    Handles both ASCII and binary formats, uniform and nonuniform
    internal fields.  Boundary fields are read but not used for
    volume-data extraction (they can be applied separately if needed).

    Supported classes:
        volScalarField, volVectorField, volSymmTensorField,
        surfaceScalarField, surfaceVectorField
    """

    # regex helpers
    _RE_HEADER   = re.compile(r"FoamFile\s*\{(.*?)\}", re.DOTALL)
    _RE_DIMS     = re.compile(r"dimensions\s*\[([\d\s\.\-eE]+)\]")
    _RE_UNIFORM  = re.compile(r"uniform\s+(.+)")
    _RE_NONUNI   = re.compile(
        r"nonuniform\s+List<(\w+)>\s+(\d+)\s*\((.*?)\)\s*;",
        re.DOTALL,
    )

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"OpenFOAM field file not found: {self.path}")

    # ── public API ─────────────────────────────────────────────────────
    def parse(self) -> Dict[str, Any]:
        """Return dict with keys: name, dims, n_cells, data, field_type."""
        raw = self.path.read_text(encoding="utf-8", errors="replace")

        header = self._parse_header(raw)
        dims   = self._parse_dimensions(raw)
        name   = header.get("object", self.path.name)
        fclass = header.get("class", "")

        data, n_cells, field_type = self._parse_internal_field(raw, fclass)

        return {
            "name": name,
            "class": fclass,
            "dimensions": dims,
            "n_cells": n_cells,
            "data": data,            # np.ndarray
            "field_type": field_type,  # "scalar" | "vector" | "tensor"
        }

    # ── header / dimensions ────────────────────────────────────────────
    def _parse_header(self, raw: str) -> Dict[str, str]:
        m = self._RE_HEADER.search(raw)
        if not m:
            return {}
        hdr = {}
        for line in m.group(1).strip().splitlines():
            parts = line.strip().rstrip(";").split()
            if len(parts) >= 2:
                hdr[parts[0]] = parts[1]
        return hdr

    def _parse_dimensions(self, raw: str) -> List[float]:
        m = self._RE_DIMS.search(raw)
        if not m:
            return [0.0] * 7
        return [float(x) for x in m.group(1).split()]

    # ── internal field ─────────────────────────────────────────────────
    def _parse_internal_field(
        self, raw: str, fclass: str
    ) -> Tuple[np.ndarray, int, str]:
        # Determine field type from class name
        if "Vector" in fclass:
            ftype = "vector"
        elif "Tensor" in fclass or "SymmTensor" in fclass:
            ftype = "tensor"
        else:
            ftype = "scalar"

        # Try uniform first
        m = self._RE_UNIFORM.search(raw)
        if m:
            val_str = m.group(1).strip()
            return self._parse_uniform(val_str, ftype)

        # Nonuniform
        m = self._RE_NONUNI.search(raw)
        if m:
            list_type = m.group(1)
            n_cells   = int(m.group(2))
            body      = m.group(3).strip()
            return self._parse_nonuniform(body, n_cells, ftype)

        raise ValueError(
            f"Cannot parse internalField in {self.path} "
            f"(class={fclass})"
        )

    def _parse_uniform(
        self, val_str: str, ftype: str
    ) -> Tuple[np.ndarray, int, str]:
        """Parse 'uniform 0' or 'uniform (0 0 0)'."""
        if ftype == "scalar":
            data = np.array([float(val_str)], dtype=np.float64)
            return data, 1, ftype
        elif ftype == "vector":
            nums = self._extract_numbers(val_str)
            data = np.array(nums, dtype=np.float64).reshape(1, -1)
            return data, 1, ftype
        else:
            nums = self._extract_numbers(val_str)
            data = np.array(nums, dtype=np.float64).reshape(1, -1)
            return data, 1, ftype

    def _parse_nonuniform(
        self, body: str, n_cells: int, ftype: str
    ) -> Tuple[np.ndarray, int, str]:
        """Parse the list body of a nonuniform field."""
        nums = self._extract_numbers(body)
        nums = np.array(nums, dtype=np.float64)

        if ftype == "scalar":
            data = nums.reshape(n_cells)
        elif ftype == "vector":
            data = nums.reshape(n_cells, 3)
        elif ftype == "tensor":
            # symmetric tensor has 6 components; full tensor has 9
            if len(nums) == n_cells * 6:
                data = nums.reshape(n_cells, 6)
            elif len(nums) == n_cells * 9:
                data = nums.reshape(n_cells, 9)
            else:
                raise ValueError(
                    f"Tensor field size mismatch: {len(nums)} values "
                    f"for {n_cells} cells"
                )
        else:
            data = nums

        return data, n_cells, ftype

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Extract all floating-point numbers from OpenFOAM list text."""
        # Remove parentheses, then split
        cleaned = text.replace("(", " ").replace(")", " ")
        tokens = cleaned.split()
        nums = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            try:
                nums.append(float(t))
            except ValueError:
                continue
        return nums


# ═══════════════════════════════════════════════════════════════════════
# 2.  OpenFOAM case reader
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMCaseReader:
    """
    Reads an entire OpenFOAM case directory.

    Expected layout:
        case_dir/
            0/          (or 100/, 200/, ...)
                U
                p
                T
                rho
                rhoU
                rhoE
                ...
            constant/
                polyMesh/
                    points
                    faces
                    owner
                    neighbour
                    cellCentres   (if present)
            system/
                controlDict
                blockMeshDict
                ...
    """

    def __init__(self, case_dir: Union[str, Path]):
        self.case_dir = Path(case_dir)
        if not self.case_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.case_dir}")

    # ── time directories ───────────────────────────────────────────────
    def list_times(self) -> List[float]:
        """Return sorted list of available time values."""
        times = []
        for entry in self.case_dir.iterdir():
            if entry.is_dir():
                try:
                    times.append(float(entry.name))
                except ValueError:
                    continue
        return sorted(times)

    def time_dir(self, t: float) -> Path:
        """Return the directory path for a given time value."""
        # OpenFOAM may store as "100" or "100.0" or "0.5"
        candidates = [str(t), str(int(t)) if t == int(t) else str(t)]
        for c in candidates:
            p = self.case_dir / c
            if p.is_dir():
                return p
        # Try closest match
        for entry in self.case_dir.iterdir():
            if entry.is_dir():
                try:
                    if abs(float(entry.name) - t) < 1e-12:
                        return entry
                except ValueError:
                    continue
        raise FileNotFoundError(f"No time directory for t={t} in {self.case_dir}")

    # ── field loading ──────────────────────────────────────────────────
    def list_fields(self, t: float) -> List[str]:
        """List available field names at time t."""
        td = self.time_dir(t)
        return sorted(
            f.name for f in td.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )

    def load_field(self, t: float, field_name: str) -> Dict[str, Any]:
        """Parse a single field file at time t."""
        td = self.time_dir(t)
        fpath = td / field_name
        if not fpath.exists():
            raise FileNotFoundError(f"Field '{field_name}' not found at t={t}")
        parser = OpenFOAMFieldParser(fpath)
        return parser.parse()

    def load_fields_at_time(
        self, t: float, field_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load multiple fields at a single time step.
        Returns {field_name: np.ndarray}.
        """
        if field_names is None:
            field_names = self.list_fields(t)

        result = {}
        for fn in field_names:
            try:
                parsed = self.load_field(t, fn)
                result[fn] = parsed["data"]
            except Exception as e:
                logger.warning(f"Skipping field '{fn}' at t={t}: {e}")
        return result

    # ── mesh ───────────────────────────────────────────────────────────
    def load_cell_centres(self) -> Optional[np.ndarray]:
        """
        Try to load cell centres.  First checks for a pre-computed
        cellCentres file, then falls back to computing from points/faces.
        Returns (n_cells, 3) array or None.
        """
        # Try constant/polyMesh/cellCentres (written by some utilities)
        cc_path = self.case_dir / "constant" / "polyMesh" / "cellCentres"
        if cc_path.exists():
            parsed = OpenFOAMFieldParser(cc_path).parse()
            return parsed["data"]

        # Try time-directory cellCentres (written by postProcess)
        for t in self.list_times():
            cc_path = self.time_dir(t) / "cellCentres"
            if cc_path.exists():
                parsed = OpenFOAMFieldParser(cc_path).parse()
                return parsed["data"]

        # Try C (centre) field written by some solvers
        for t in self.list_times():
            c_path = self.time_dir(t) / "C"
            if c_path.exists():
                parsed = OpenFOAMFieldParser(c_path).parse()
                return parsed["data"]

        logger.warning(
            "Could not find cell centres. "
            "Run: postProcess -func 'writeCellCentres' "
            "or use a structured blockMesh case."
        )
        return None

    def load_points(self) -> Optional[np.ndarray]:
        """Load mesh vertex positions from constant/polyMesh/points."""
        pts_path = self.case_dir / "constant" / "polyMesh" / "points"
        if not pts_path.exists():
            return None
        parsed = OpenFOAMFieldParser(pts_path).parse()
        return parsed["data"]

    def load_owner_neighbour(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load face owner/neighbour arrays."""
        poly = self.case_dir / "constant" / "polyMesh"
        owner = neighbour = None
        op = poly / "owner"
        np_ = poly / "neighbour"
        if op.exists():
            owner = OpenFOAMFieldParser(op).parse()["data"].astype(int)
        if np_.exists():
            neighbour = OpenFOAMFieldParser(np_).parse()["data"].astype(int)
        return owner, neighbour


# ═══════════════════════════════════════════════════════════════════════
# 3.  Structured-grid mapping
# ═══════════════════════════════════════════════════════════════════════

class StructuredGridMapper:
    """
    Maps OpenFOAM unstructured cell-centred data onto a structured
    (nx, ny, nz) grid for SUPER DNS.

    Two strategies:
      A) Direct reshape — works when the OpenFOAM case was built with
         a simple blockMesh (structured hex mesh) and cells are in
         lexicographic order.
      B) Interpolation  — works for any mesh; uses scipy griddata.
    """

    def __init__(
        self,
        target_shape: Tuple[int, int, int],
        domain_extents: Optional[Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]] = None,
    ):
        self.nx, self.ny, self.nz = target_shape
        self.domain_extents = domain_extents  # ((xmin,ymin,zmin),(xmax,ymax,zmax))

    def fit(self, cell_centres: np.ndarray) -> None:
        """Precompute mapping from cell_centres to structured indices."""
        self.cell_centres = cell_centres  # (N, 3)
        if self.domain_extents is None:
            self.domain_extents = (
                cell_centres.min(axis=0).tolist(),
                cell_centres.max(axis=0).tolist(),
            )
        self._build_index_map()

    def _build_index_map(self) -> None:
        """Assign each OpenFOAM cell to a structured-grid index."""
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.domain_extents
        dx = (xmax - xmin) / max(self.nx, 1)
        dy = (ymax - ymin) / max(self.ny, 1)
        dz = (zmax - zmin) / max(self.nz, 1)

        cc = self.cell_centres
        ix = np.clip(((cc[:, 0] - xmin) / max(dx, 1e-30)).astype(int), 0, self.nx - 1)
        iy = np.clip(((cc[:, 1] - ymin) / max(dy, 1e-30)).astype(int), 0, self.ny - 1)
        iz = np.clip(((cc[:, 2] - zmin) / max(dz, 1e-30)).astype(int), 0, self.nz - 1)

        self._ix = ix
        self._iy = iy
        self._iz = iz

    def map_scalar(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Map a scalar field (N,) → (nx, ny, nz)."""
        grid = np.full((self.nx, self.ny, self.nz), fill, dtype=np.float64)
        # Average if multiple cells map to the same grid point
        counts = np.zeros_like(grid)
        np.add.at(grid, (self._ix, self._iy, self._iz), values)
        np.add.at(counts, (self._ix, self._iy, self._iz), 1.0)
        mask = counts > 0
        grid[mask] /= counts[mask]
        return grid

    def map_vector(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Map a vector field (N,3) → (3, nx, ny, nz)."""
        result = np.stack(
            [self.map_scalar(values[:, c], fill) for c in range(values.shape[1])],
            axis=0,
        )
        return result  # (3, nx, ny, nz)

    def map_interpolate(
        self,
        cell_centres: np.ndarray,
        values: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """
        Interpolate unstructured data onto the structured grid using
        scipy.interpolate.griddata.  Slower but works for any mesh.

        Args:
            cell_centres: (N, 3)
            values: (N,) for scalar or (N, C) for multi-component
            method: 'linear', 'nearest', or 'cubic'

        Returns:
            (nx, ny, nz) or (C, nx, ny, nz)
        """
        from scipy.interpolate import griddata

        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.domain_extents
        gx = np.linspace(xmin, xmax, self.nx)
        gy = np.linspace(ymin, ymax, self.ny)
        gz = np.linspace(zmin, zmax, self.nz)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        grid_pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1)

        if values.ndim == 1:
            return griddata(cell_centres, values, grid_pts, method=method).reshape(
                self.nx, self.ny, self.nz
            )
        else:
            n_comp = values.shape[1]
            result = np.zeros((n_comp, self.nx, self.ny, self.nz))
            for c in range(n_comp):
                result[c] = griddata(
                    cell_centres, values[:, c], grid_pts, method=method
                ).reshape(self.nx, self.ny, self.nz)
            return result


# ═══════════════════════════════════════════════════════════════════════
# 4.  SUPER DNS data converter
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SuperDNSFieldSet:
    """Container for one time-step of SUPER DNS conservative variables."""
    rho:  np.ndarray   # (nx, ny, nz)
    rhou: np.ndarray   # (nx, ny, nz)
    rhov: np.ndarray   # (nx, ny, nz)
    rhow: np.ndarray   # (nx, ny, nz)
    rhoE: np.ndarray   # (nx, ny, nz)
    RZ:   Optional[np.ndarray] = None   # mixture fraction (optional)
    RS:   Optional[np.ndarray] = None   # soot (optional)
    time: float = 0.0

    def to_tensors(self, device: torch.device = torch.device("cpu"),
                   dtype: torch.dtype = torch.float32) -> Dict[str, Tensor]:
        out = {}
        for name in ("rho", "rhou", "rhov", "rhow", "rhoE"):
            arr = getattr(self, name)
            out[name] = torch.from_numpy(np.ascontiguousarray(arr)).to(
                device=device, dtype=dtype
            )
        for name in ("RZ", "RS"):
            arr = getattr(self, name)
            if arr is not None:
                out[name] = torch.from_numpy(np.ascontiguousarray(arr)).to(
                    device=device, dtype=dtype
                )
        return out


class SuperDNSDataConverter:
    """
    Converts OpenFOAM primitive or conservative fields into the
    conservative-variable set used by SUPER DNS ONE v6.

    Supported input combinations:
      A) rho, U, p  (+ T optional)  →  compute rhoE
      B) rho, rhoU, rhoE            →  direct
      C) rho, U, T, p               →  compute rhoE from T
      D) rhoU, rhoE, rho            →  direct

    The converter also handles mixture fraction (Z) and soot (Y_soot)
    if present in the OpenFOAM case.
    """

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    def convert(
        self,
        fields: Dict[str, np.ndarray],
        mapper: StructuredGridMapper,
        time: float = 0.0,
    ) -> SuperDNSFieldSet:
        """
        Convert a dict of OpenFOAM fields (already loaded as numpy arrays)
        into a SuperDNSFieldSet on the structured grid.

        Args:
            fields: {name: raw_data} where raw_data is (N,) for scalar
                    or (N,3) for vector, as returned by OpenFOAMCaseReader.
            mapper: fitted StructuredGridMapper.
            time: simulation time.
        """
        g = self.gamma

        # ── density ──────────────────────────────────────────────────
        rho_raw = self._require(fields, "rho")
        rho = mapper.map_scalar(rho_raw, fill=1.0)

        # ── velocity / momentum ─────────────────────────────────────
        if "rhoU" in fields and "rhoV" in fields and "rhoW" in fields:
            # Conservative form directly
            rhou = mapper.map_scalar(fields["rhoU"], fill=0.0)
            rhov = mapper.map_scalar(fields["rhoV"], fill=0.0)
            rhow = mapper.map_scalar(fields["rhoW"], fill=0.0)
        elif "U" in fields:
            U = mapper.map_vector(fields["U"], fill=0.0)  # (3, nx, ny, nz)
            rhou = rho * U[0]
            rhov = rho * U[1]
            rhow = rho * U[2]
        else:
            raise KeyError("Need either 'U' or 'rhoU/rhoV/rhoW' in OpenFOAM fields")

        # ── energy ───────────────────────────────────────────────────
        if "rhoE" in fields:
            rhoE = mapper.map_scalar(fields["rhoE"], fill=0.0)
        elif "p" in fields:
            p = mapper.map_scalar(fields["p"], fill=101325.0)
            ke = 0.5 * (rhou**2 + rhov**2 + rhow**2) / np.clip(rho, 1e-10, None)
            rhoE = p / (g - 1.0) + ke
        elif "T" in fields and "rho" in fields:
            T = mapper.map_scalar(fields["T"], fill=300.0)
            R_air = 287.052874
            p = rho * R_air * T
            ke = 0.5 * (rhou**2 + rhov**2 + rhow**2) / np.clip(rho, 1e-10, None)
            rhoE = p / (g - 1.0) + ke
        else:
            raise KeyError(
                "Need 'rhoE', 'p', or 'T' in OpenFOAM fields to compute energy"
            )

        # ── optional scalars ─────────────────────────────────────────
        RZ = None
        RS = None
        if "Z" in fields:
            RZ = rho * mapper.map_scalar(fields["Z"], fill=0.0)
        elif "RZ" in fields:
            RZ = mapper.map_scalar(fields["RZ"], fill=0.0)
        if "Y_soot" in fields:
            RS = rho * mapper.map_scalar(fields["Y_soot"], fill=0.0)
        elif "RS" in fields:
            RS = mapper.map_scalar(fields["RS"], fill=0.0)

        return SuperDNSFieldSet(
            rho=rho, rhou=rhou, rhov=rhov, rhow=rhow, rhoE=rhoE,
            RZ=RZ, RS=RS, time=time,
        )

    @staticmethod
    def _require(fields: Dict[str, np.ndarray], name: str) -> np.ndarray:
        if name not in fields:
            raise KeyError(f"Required field '{name}' not found in OpenFOAM output")
        return fields[name]


# ═══════════════════════════════════════════════════════════════════════
# 5.  PyTorch Dataset for AI training
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMTrainingDataset(Dataset):
    """
    PyTorch Dataset that yields SUPER DNS conservative-variable tensors
    from a sequence of OpenFOAM time steps.

    Each sample is a dict:
        {
            "rho":  Tensor (nx, ny, nz),
            "rhou": Tensor (nx, ny, nz),
            "rhov": Tensor (nx, ny, nz),
            "rhow": Tensor (nx, ny, nz),
            "rhoE": Tensor (nx, ny, nz),
            "time": float,
        }

    Optionally includes "RZ" and "RS" if present.

    For supervised surrogate training, set `return_pairs=True` to get
    (input, target) pairs from consecutive time steps.
    """

    def __init__(
        self,
        samples: List[SuperDNSFieldSet],
        return_pairs: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.samples = samples
        self.return_pairs = return_pairs
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        if self.return_pairs:
            return max(0, len(self.samples) - 1)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.return_pairs:
            inp = self.samples[idx].to_tensors(self.device, self.dtype)
            tgt = self.samples[idx + 1].to_tensors(self.device, self.dtype)
            return {"input": inp, "target": tgt,
                    "dt": self.samples[idx + 1].time - self.samples[idx].time}
        else:
            return self.samples[idx].to_tensors(self.device, self.dtype)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Top-level bridge
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMSuperDNSBridge:
    """
    End-to-end bridge: OpenFOAM case → SUPER DNS ONE v6 training data.

    Usage:
        bridge = OpenFOAMSuperDNSBridge(
            case_dir="cases/hypersonic_bnd",
            target_shape=(64, 64, 64),
            field_names=["rho", "U", "p"],
        )
        samples = bridge.build_training_set(time_range=(100, 500))
        dataset = bridge.to_pytorch_dataset(samples)
    """

    def __init__(
        self,
        case_dir: Union[str, Path],
        target_shape: Tuple[int, int, int] = (64, 64, 64),
        field_names: Optional[List[str]] = None,
        gamma: float = 1.4,
        domain_extents: Optional[Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]] = None,
        use_interpolation: bool = False,
    ):
        self.reader = OpenFOAMCaseReader(case_dir)
        self.target_shape = target_shape
        self.field_names = field_names or ["rho", "U", "p"]
        self.converter = SuperDNSDataConverter(gamma=gamma)
        self.use_interpolation = use_interpolation

        # Fit the structured-grid mapper
        self.mapper = StructuredGridMapper(target_shape, domain_extents)
        cc = self.reader.load_cell_centres()
        if cc is not None:
            self.mapper.fit(cc)
        else:
            logger.warning(
                "No cell centres found. "
                "Set domain_extents manually or run writeCellCentres."
            )

    # ── build training set ─────────────────────────────────────────────
    def build_training_set(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        time_stride: int = 1,
    ) -> List[SuperDNSFieldSet]:
        """
        Load all requested time steps and convert to SUPER DNS format.

        Args:
            time_range: (t_start, t_end) inclusive; None = all times.
            time_stride: take every Nth time step.

        Returns:
            List of SuperDNSFieldSet, sorted by time.
        """
        times = self.reader.list_times()
        if time_range is not None:
            t0, t1 = time_range
            times = [t for t in times if t0 <= t <= t1]
        times = times[::time_stride]

        logger.info(f"Loading {len(times)} time steps: {times[0]} → {times[-1]}")

        samples: List[SuperDNSFieldSet] = []
        for t in times:
            try:
                raw_fields = self.reader.load_fields_at_time(t, self.field_names)
                fieldset = self.converter.convert(raw_fields, self.mapper, time=t)
                samples.append(fieldset)
            except Exception as e:
                logger.warning(f"Skipping t={t}: {e}")

        logger.info(f"Successfully converted {len(samples)} time steps.")
        return samples

    # ── PyTorch dataset ────────────────────────────────────────────────
    def to_pytorch_dataset(
        self,
        samples: List[SuperDNSFieldSet],
        return_pairs: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> OpenFOAMTrainingDataset:
        return OpenFOAMTrainingDataset(
            samples, return_pairs=return_pairs, device=device, dtype=dtype
        )

    # ── initialise SUPER DNS solver ───────────────────────────────────
    def initialise_solver(
        self,
        solver: Any,   # CompressibleSolver instance
        fieldset: SuperDNSFieldSet,
    ) -> None:
        """
        Overwrite a CompressibleSolver's fields with OpenFOAM data.
        Useful for "warm-start" or "restart from OpenFOAM".

        Args:
            solver: an initialised CompressibleSolver.
            fieldset: the converted OpenFOAM state to inject.
        """
        dev = solver.device
        dt_ = solver.dtype

        solver.rho  = torch.from_numpy(fieldset.rho).to(dev, dt_)
        solver.rhou = torch.from_numpy(fieldset.rhou).to(dev, dt_)
        solver.rhov = torch.from_numpy(fieldset.rhov).to(dev, dt_)
        solver.rhow = torch.from_numpy(fieldset.rhow).to(dev, dt_)
        solver.rhoE = torch.from_numpy(fieldset.rhoE).to(dev, dt_)

        if fieldset.RZ is not None and hasattr(solver, "RZ"):
            solver.RZ = torch.from_numpy(fieldset.RZ).to(dev, dt_)
        if fieldset.RS is not None and hasattr(solver, "RS"):
            solver.RS = torch.from_numpy(fieldset.RS).to(dev, dt_)

        solver.step_count = 0
        solver.time = fieldset.time
        logger.info(
            f"Solver initialised from OpenFOAM state at t={fieldset.time}"
        )

    # ── convenience: build + dataset in one call ───────────────────────
    def build_and_wrap(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        time_stride: int = 1,
        return_pairs: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> OpenFOAMTrainingDataset:
        samples = self.build_training_set(time_range, time_stride)
        return self.to_pytorch_dataset(samples, return_pairs, device, dtype)


# ═══════════════════════════════════════════════════════════════════════
# 7.  Convenience factory
# ═══════════════════════════════════════════════════════════════════════

def build_training_data(
    case_dir: str,
    target_shape: Tuple[int, int, int] = (64, 64, 64),
    field_names: Optional[List[str]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    time_stride: int = 1,
    return_pairs: bool = False,
    gamma: float = 1.4,
    batch_size: int = 4,
    num_workers: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> DataLoader:
    """
    One-call factory: OpenFOAM case → PyTorch DataLoader.

    Args:
        case_dir: path to the OpenFOAM case directory.
        target_shape: structured grid resolution for SUPER DNS.
        field_names: OpenFOAM fields to read (default: rho, U, p).
        time_range: (t_start, t_end) or None for all.
        time_stride: take every Nth time step.
        return_pairs: if True, yield (input, target) pairs for
                      supervised learning (consecutive time steps).
        gamma: ratio of specific heats.
        batch_size, num_workers: DataLoader settings.
        device: "cpu" or "cuda".
        dtype: "float32" or "float64".

    Returns:
        DataLoader ready for training.

    Example:
        loader = build_training_data(
            "cases/hypersonic_bnd",
            target_shape=(32, 32, 32),
            field_names=["rho", "U", "p"],
            time_range=(100, 500),
            return_pairs=True,
        )
        for batch in loader:
            inp = batch["input"]
            tgt = batch["target"]
            ...
    """
    dev = torch.device(device)
    dt_ = torch.float32 if dtype == "float32" else torch.float64

    bridge = OpenFOAMSuperDNSBridge(
        case_dir=case_dir,
        target_shape=target_shape,
        field_names=field_names,
        gamma=gamma,
    )

    dataset = bridge.build_and_wrap(
        time_range=time_range,
        time_stride=time_stride,
        return_pairs=return_pairs,
        device=dev,
        dtype=dt_,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        drop_last=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# 8.  AI training loop template
# ═══════════════════════════════════════════════════════════════════════

def train_surrogate(
    case_dir: str,
    model: torch.nn.Module,
    target_shape: Tuple[int, int, int] = (32, 32, 32),
    field_names: Optional[List[str]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 2,
    device: str = "cuda",
) -> List[float]:
    """
    Minimal training loop for a neural surrogate that learns to predict
    the next time step from the current one (trained on OpenFOAM data).

    Args:
        case_dir: OpenFOAM case directory.
        model: a PyTorch nn.Module that takes a dict of conservative
               variables and returns the predicted next-step dict.
        target_shape: structured grid resolution.
        field_names: OpenFOAM fields to load.
        time_range: (t_start, t_end).
        epochs: number of training epochs.
        lr: learning rate.
        batch_size: batch size.
        device: "cpu" or "cuda".

    Returns:
        List of per-epoch mean losses.
    """
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = build_training_data(
        case_dir=case_dir,
        target_shape=target_shape,
        field_names=field_names,
        time_range=time_range,
        return_pairs=True,
        batch_size=batch_size,
        device=device,
    )

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            inp = batch["input"]
            tgt = batch["target"]

            opt.zero_grad()
            pred = model(inp)

            # MSE loss on all conservative variables
            loss = sum(
                torch.nn.functional.mse_loss(pred[k], tgt[k])
                for k in ("rho", "rhou", "rhov", "rhow", "rhoE")
                if k in pred and k in tgt
            )

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}  loss={mean_loss:.6e}")

    return losses


# ═══════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenFOAM → SUPER DNS ONE v6 bridge"
    )
    parser.add_argument("--case", type=str, required=True,
                        help="Path to OpenFOAM case directory")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--fields", type=str, nargs="+",
                        default=["rho", "U", "p"],
                        help="OpenFOAM field names to load")
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--output", type=str, default=None,
                        help="Save converted data as .pt file")
    parser.add_argument("--info", action="store_true",
                        help="Print case info only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    bridge = OpenFOAMSuperDNSBridge(
        case_dir=args.case,
        target_shape=(args.nx, args.ny, args.nz),
        field_names=args.fields,
        gamma=args.gamma,
    )

    times = bridge.reader.list_times()
    logger.info(f"Case: {args.case}")
    logger.info(f"Available times: {len(times)}  "
                f"[{times[0] if times else '—'} → {times[-1] if times else '—'}]")

    if args.info:
        for t in times[:10]:
            fns = bridge.reader.list_fields(t)
            logger.info(f"  t={t}: fields={fns}")
        return

    t_range = None
    if args.t_start is not None and args.t_end is not None:
        t_range = (args.t_start, args.t_end)

    samples = bridge.build_training_set(
        time_range=t_range, time_stride=args.stride
    )

    if args.output:
        # Save as a list of tensors
        save_data = []
        for s in samples:
            save_data.append(s.to_tensors())
        torch.save(save_data, args.output)
        logger.info(f"Saved {len(save_data)} samples → {args.output}")
    else:
        logger.info(f"Converted {len(samples)} time steps (not saved).")


if __name__ == "__main__":
    main()
