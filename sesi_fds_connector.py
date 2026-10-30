"""
=============================================================================
SESI FDS (FIRE DYNAMICS SIMULATOR) CONNECTOR
Integration with NIST's open-source fire simulation software
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT FDS IS [C]
---------------
FDS (Fire Dynamics Simulator) is an open-source CFD code developed by NIST
(National Institute of Standards and Technology) specifically for fire
applications. It solves:
  • Low-Mach number Navier-Stokes equations
  • Combustion (mixture fraction model)
  • Radiation (finite volume method)
  • Pyrolysis (solid fuel decomposition)
  • Smoke transport

FDS uses a structured Cartesian grid and outputs results in HDF5 format
compatible with Smokeview visualization.

WHAT THIS CONNECTOR DOES
-------------------------
[D] Generates FDS input files (.fds) from SESI FireCFDEngine scenarios
[D] Launches FDS simulations (if installed)
[D] Parses FDS HDF5 output (temperature, velocity, smoke, HRR)
[D] Feeds results back into SESI analysis pipeline

FDS INPUT FILE STRUCTURE
-------------------------
&HEAD CHID='case_id', TITLE='description' /
&GRID IBJK=64,64,32 XB=0,10,0,10,0,3 /
&TIME T_END=60.0, DT_DEVC_INIT=0.1 /
&DUMP SMOKE3D=.TRUE., DEVC_LINE=... /
&MATL ID='WOOD', DENSITY=600, CONDUCTIVITY=0.2, ... /
&SURF ID='furniture', MATL_ID='WOOD', THICKNESS=0.05 /
&OBST XB=2,3,2,3,0,1 SURF_ID='furniture' /
&VENT MB='XMIN', SURF_ID='OPEN' /
&TAIL /

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• FDS must be installed separately (https://pages.nist.gov/fds-smv/)
• FDS is a RESEARCH TOOL; results depend on grid resolution, turbulence
  model, and combustion parameters.
• This connector generates input files and parses outputs; it does NOT
  replace FDS itself.
• All results require experimental validation before engineering use.
• Furniture/pyrolysis models are simplified; real materials need
  calibrated properties.
=============================================================================
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.fds_connector")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

__all__ = [
    "FDSConfig",
    "FDSMaterial",
    "FDSGeometry",
    "FDSInputGenerator",
    "FDSRunner",
    "FDSOutputParser",
    "FDSConnector",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FDSConfig:
    """Configuration for FDS simulation."""
    # Case identification
    case_id: str = "sesi_fire"
    title: str = "SESI Fire Simulation"

    # Grid
    nx: int = 64
    ny: int = 64
    nz: int = 32
    domain_size: Tuple[float, float, float] = (10.0, 10.0, 3.0)  # meters

    # Time
    t_end: float = 60.0  # seconds
    dt_init: float = 0.01
    max_dt: float = 0.1

    # Output
    output_dir: str = "./fds_output"
    save_smoke3d: bool = True
    save_slice: bool = True
    save_devc: bool = True

    # FDS installation
    fds_binary: str = "fds"  # path to FDS executable
    n_processes: int = 1     # for MPI parallel run

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# [D] FDS MATERIAL LIBRARY
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FDSMaterial:
    """FDS material definition for solid fuel."""
    name: str
    density: float           # kg/m³
    conductivity: float      # W/(m·K)
    specific_heat: float     # J/(kg·K)
    # Pyrolysis parameters
    n_reactions: int = 1
    nu_spec: float = 1.0     # yield of fuel vapor
    heat_of_combustion: float = 18e6  # J/kg (wood default)
    reference_temperature: float = 600.0  # K

    def to_fds_matl(self) -> str:
        """Generate &MATL line."""
        return (f"&MATL ID='{self.name}',\n"
                f"      DENSITY={self.density},\n"
                f"      CONDUCTIVITY={self.conductivity},\n"
                f"      SPECIFIC_HEAT={self.specific_heat},\n"
                f"      N_REACTIONS={self.n_reactions},\n"
                f"      NU_SPEC={self.nu_spec},\n"
                f"      HEAT_OF_COMBUSTION={self.heat_of_combustion},\n"
                f"      REFERENCE_TEMPERATURE={self.reference_temperature} /\n")


class FDSMaterialLibrary:
    """[D] Library of common furniture materials."""

    MATERIALS = {
        "WOOD": FDSMaterial("WOOD", 600.0, 0.2, 2500.0, heat_of_combustion=18e6),
        "FABRIC": FDSMaterial("FABRIC", 100.0, 0.1, 1500.0, heat_of_combustion=20e6),
        "FOAM": FDSMaterial("FOAM", 30.0, 0.05, 1800.0, heat_of_combustion=25e6),
        "PLASTIC": FDSMaterial("PLASTIC", 900.0, 0.3, 2000.0, heat_of_combustion=40e6),
    }

    @classmethod
    def get(cls, name: str) -> FDSMaterial:
        name_upper = name.upper()
        if name_upper not in cls.MATERIALS:
            logger.warning(f"Unknown material '{name}'; using WOOD")
            name_upper = "WOOD"
        return cls.MATERIALS[name_upper]


# ═══════════════════════════════════════════════════════════════════════
# [D] FDS GEOMETRY
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FDSGeometry:
    """Geometry object for FDS (obstruction or vent)."""
    obj_type: str  # "OBST" or "VENT"
    xb: Tuple[float, float, float, float, float, float]  # x1,x2,y1,y2,z1,z2
    surf_id: str = "INERT"
    color: str = "GRAY"

    def to_fds_line(self) -> str:
        xb_str = ",".join(f"{v:.3f}" for v in self.xb)
        if self.obj_type == "OBST":
            return (f"&OBST XB={xb_str}, SURF_ID='{self.surf_id}', "
                    f"COLOR='{self.color}' /\n")
        else:  # VENT
            return f"&VENT XB={xb_str}, SURF_ID='{self.surf_id}' /\n"


# ═══════════════════════════════════════════════════════════════════════
# [D] FDS INPUT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class FDSInputGenerator:
    """
    [D] Generate FDS input files from SESI fire scenarios.

    Converts SESI FireCFDEngine furniture objects to FDS &OBST and &MATL.
    """

    def __init__(self, cfg: FDSConfig):
        self.cfg = cfg
        self.materials: List[FDSMaterial] = []
        self.geometries: List[FDSGeometry] = []
        self.surfaces: Dict[str, str] = {}  # surf_id -> matl_id

    def add_furniture(self, name: str, material: str,
                      position: Tuple[float, float, float],
                      size: Tuple[float, float, float]) -> None:
        """Add a furniture item as an FDS obstruction."""
        mat = FDSMaterialLibrary.get(material)
        if mat not in self.materials:
            self.materials.append(mat)

        x, y, z = position
        dx, dy, dz = size
        xb = (x, x + dx, y, y + dy, z, z + dz)

        surf_id = f"surf_{name}"
        self.surfaces[surf_id] = mat.name

        self.geometries.append(FDSGeometry("OBST", xb, surf_id))

    def add_open_boundaries(self) -> None:
        """Add open boundaries (vents) on all sides."""
        Lx, Ly, Lz = self.cfg.domain_size
        # Open at XMIN, XMAX, YMIN, YMAX, ZMAX
        self.geometries.append(FDSGeometry("VENT", (0, 0, 0, Ly, 0, Lz), "OPEN"))
        self.geometries.append(FDSGeometry("VENT", (Lx, Lx, 0, Ly, 0, Lz), "OPEN"))
        self.geometries.append(FDSGeometry("VENT", (0, Lx, 0, 0, 0, Lz), "OPEN"))
        self.geometries.append(FDSGeometry("VENT", (0, Lx, Ly, Ly, 0, Lz), "OPEN"))
        self.geometries.append(FDSGeometry("VENT", (0, Lx, 0, Ly, Lz, Lz), "OPEN"))

    def generate(self) -> str:
        """Generate complete FDS input file content."""
        cfg = self.cfg
        lines = []

        # Header
        lines.append(f"&HEAD CHID='{cfg.case_id}', TITLE='{cfg.title}' /\n")

        # Grid
        Lx, Ly, Lz = cfg.domain_size
        lines.append(f"&GRID IJK={cfg.nx},{cfg.ny},{cfg.nz} "
                     f"XB=0,{Lx},0,{Ly},0,{Lz} /\n")

        # Time
        lines.append(f"&TIME T_END={cfg.t_end}, DT_DEVC_INIT={cfg.dt_init} /\n")
        lines.append(f"&DUMP SMOKE3D={'.TRUE.' if cfg.save_smoke3d else '.FALSE.'}, "
                     f"DT_DEVC={cfg.max_dt} /\n")

        # Materials
        for mat in self.materials:
            lines.append(mat.to_fds_matl())

        # Surfaces (link material to geometry)
        for surf_id, matl_id in self.surfaces.items():
            lines.append(f"&SURF ID='{surf_id}', MATL_ID='{matl_id}', "
                         f"THICKNESS=0.05 /\n")

        # Geometries
        for geom in self.geometries:
            lines.append(geom.to_fds_line())

        # Ignition source (simple heat source)
        lines.append(f"&SURF ID='IGNITER', HRRPUA=500.0 /\n")  # 500 kW/m²
        # Place igniter at center-bottom
        lines.append(f"&VENT XB=0,0.5,0,0.5,0,0 SURF_ID='IGNITER' /\n")

        # Tail
        lines.append("&TAIL /\n")

        return "".join(lines)

    def write(self, output_path: Optional[str] = None) -> str:
        """Write FDS input file to disk."""
        content = self.generate()
        if output_path is None:
            output_path = Path(self.cfg.output_dir) / f"{self.cfg.case_id}.fds"
        Path(output_path).write_text(content)
        logger.info(f"FDS input written to {output_path}")
        return str(output_path)


# ═══════════════════════════════════════════════════════════════════════
# [D] FDS RUNNER
# ═══════════════════════════════════════════════════════════════════════

class FDSRunner:
    """
    [D] Launch FDS simulation.

    Requires FDS to be installed and in PATH (or specified via cfg.fds_binary).
    """

    def __init__(self, cfg: FDSConfig):
        self.cfg = cfg
        self.available = self._check_fds()

    def _check_fds(self) -> bool:
        """Check if FDS is available."""
        try:
            result = subprocess.run(
                [self.cfg.fds_binary, "--version"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("FDS not available; using synthetic fallback")
            return False

    def run(self, input_file: str) -> Dict[str, Any]:
        """
        Run FDS simulation.

        Returns dict with status and output paths.
        """
        if not self.available:
            return self._synthetic_run()

        try:
            # Build command
            if self.cfg.n_processes > 1:
                cmd = ["mpiexec", "-n", str(self.cfg.n_processes),
                       self.cfg.fds_binary, input_file]
            else:
                cmd = [self.cfg.fds_binary, input_file]

            logger.info(f"Running FDS: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.cfg.t_end * 10,  # generous timeout
                cwd=self.cfg.output_dir,
                text=True
            )

            if result.returncode == 0:
                return {
                    "status": "completed",
                    "backend": "fds",
                    "stdout": result.stdout[-1000:],  # last 1000 chars
                    "output_dir": self.cfg.output_dir,
                }
            else:
                return {
                    "status": "failed",
                    "backend": "fds",
                    "returncode": result.returncode,
                    "stderr": result.stderr[-1000:],
                }

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "backend": "fds"}
        except Exception as e:
            logger.error(f"FDS run failed: {e}")
            return {"status": "error", "error": str(e), "backend": "fds"}

    def _synthetic_run(self) -> Dict[str, Any]:
        """Generate synthetic results when FDS unavailable."""
        logger.warning("Generating SYNTHETIC FDS results")

        # Synthetic temperature field
        n_steps = 100
        temps = 293.0 + 500.0 * np.random.rand(n_steps, self.cfg.nz,
                                                  self.cfg.ny, self.cfg.nx)

        # Synthetic HRR
        hrr = 1e6 * np.exp(-np.arange(n_steps) / 30.0)

        return {
            "status": "synthetic",
            "backend": "synthetic",
            "temperature": temps,
            "hrr": hrr,
            "note": "SYNTHETIC results; FDS not available",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] FDS OUTPUT PARSER
# ═══════════════════════════════════════════════════════════════════════

class FDSOutputParser:
    """
    [D] Parse FDS HDF5 output files.

    FDS outputs:
      • {case_id}_3d.h5 — 3D smoke/temperature fields
      • {case_id}_devc.csv — device measurements
      • {case_id}_slice.vtk — slice files for visualization
    """

    def __init__(self, cfg: FDSConfig):
        self.cfg = cfg
        if not HAS_H5PY:
            logger.warning("h5py not available; parsing disabled")

    def parse_3d(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Parse 3D output file."""
        if not HAS_H5PY:
            return {"error": "h5py not available"}

        case_id = case_id or self.cfg.case_id
        h5_path = Path(self.cfg.output_dir) / f"{case_id}_3d.h5"

        if not h5_path.exists():
            return {"error": f"File not found: {h5_path}"}

        try:
            with h5py.File(h5_path, 'r') as f:
                # FDS 3D file structure (simplified)
                data = {}
                for key in f.keys():
                    if key in ['TEMP', 'U-VEL', 'V-VEL', 'W-VEL', 'SOOT_DENSITY']:
                        data[key] = f[key][:]
                return {"status": "parsed", "data": data}
        except Exception as e:
            return {"error": str(e)}

    def parse_devic(self, case_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Parse device output CSV."""
        case_id = case_id or self.cfg.case_id
        csv_path = Path(self.cfg.output_dir) / f"{case_id}_devc.csv"

        if not csv_path.exists():
            return {"error": f"File not found: {csv_path}"}

        try:
            data = np.genfromtxt(csv_path, delimiter=',', names=True)
            return {"status": "parsed", "data": data}
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATED FDS CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class FDSConnector:
    """
    [D] End-to-end FDS integration.

    Workflow:
      1. Convert SESI fire scenario to FDS input
      2. Run FDS (or synthetic fallback)
      3. Parse results
      4. Feed back to SESI analysis
    """

    def __init__(self, cfg: Optional[FDSConfig] = None):
        self.cfg = cfg or FDSConfig()
        self.generator = FDSInputGenerator(self.cfg)
        self.runner = FDSRunner(self.cfg)
        self.parser = FDSOutputParser(self.cfg)

    def simulate_fire(self, furniture_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate a fire scenario with given furniture.

        furniture_items: list of dicts with keys:
          • name, material, position, size
        """
        # Add furniture
        for item in furniture_items:
            self.generator.add_furniture(
                item["name"],
                item["material"],
                item["position"],
                item["size"],
            )
        self.generator.add_open_boundaries()

        # Write input
        input_path = self.generator.write()

        # Run
        result = self.runner.run(input_path)

        # Parse if real FDS
        if result.get("backend") == "fds" and result.get("status") == "completed":
            parsed = self.parser.parse_3d()
            result["parsed_output"] = parsed

        return result


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI FDS Connector — Smoke Test")
    print("=" * 72)
    print(f"  h5py available       : {HAS_H5PY}")

    cfg = FDSConfig(case_id="test_fire", output_dir="./fds_test_output")
    connector = FDSConnector(cfg)

    # Define furniture
    furniture = [
        {"name": "sofa", "material": "foam",
         "position": (2.0, 2.0, 0.0), "size": (2.0, 1.0, 0.8)},
        {"name": "table", "material": "wood",
         "position": (5.0, 3.0, 0.0), "size": (1.5, 1.0, 0.75)},
        {"name": "curtain", "material": "fabric",
         "position": (0.0, 4.0, 0.0), "size": (0.1, 2.0, 2.5)},
    ]

    print("\n  ── Generating FDS input ──")
    for item in furniture:
        connector.generator.add_furniture(
            item["name"], item["material"], item["position"], item["size"])
    connector.generator.add_open_boundaries()

    fds_content = connector.generator.generate()
    print(f"    FDS input length   : {len(fds_content)} chars")
    print(f"    First 3 lines      :")
    for line in fds_content.split('\n')[:3]:
        print(f"      {line}")

    # Write and check
    input_path = connector.generator.write()
    print(f"    Input file         : {input_path}")

    # Run simulation
    print("\n  ── Running simulation ──")
    result = connector.runner.run(input_path)
    print(f"    Backend            : {result.get('backend', 'unknown')}")
    print(f"    Status             : {result.get('status', 'unknown')}")
    if result.get("status") == "synthetic":
        print(f"    NOTE               : {result.get('note', '')}")

    # Check FDS availability
    print(f"\n    FDS available      : {connector.runner.available}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • FDS is a RESEARCH TOOL from NIST")
    print("  • Results depend on grid resolution and model parameters")
    print("  • This connector generates input and parses output")
    print("  • Synthetic fallback used when FDS not installed")
    print("  • All results require experimental validation")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
