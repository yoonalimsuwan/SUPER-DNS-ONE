"""
=============================================================================
SESI OpenFOAM CASE GENERATOR — สร้างเคสจริงที่รันได้
=============================================================================
สร้างโครงสร้างไดเรกทอรี OpenFOAM ที่สมบูรณ์:
  case_dir/
  ├── 0/                  ← initial & boundary conditions
  │   ├── U
  │   └── p
  ├── constant/
  │   ├── transportProperties
  │   ├── turbulenceProperties
  │   └── polyMesh/
  │       └── blockMeshDict
  ├── system/
  │   ├── controlDict
  │   ├── fvSchemes
  │   ├── fvSolution
  │   └── decomposeParDict
  ├── postProcess/          ← สำหรับแปลงเป็น PyTorch tensors
  │   └── extract_fields.py
  └── run.sh                ← สคริปต์รันแบบครบวงจร

รองรับ:
  • ลามินาร์ (icoFoam / pimpleFoam)
  • เทอร์บูเลนซ์ (simpleFoam / pisoFoam + k-epsilon, k-omega SST)
  • 2D / 3D
  • Inlet/outlet/wall/symmetry BCs
  • Parallel run (decomposePar)

HONEST SCOPE:
  • สร้างไฟล์ที่ถูกต้องตามรูปแบบ OpenFOAM
  • ต้องติดตั้ง OpenFOAM ก่อน (v2112+, v2206+, หรือ ESI/OpenCFD)
  • ผลลัพธ์จาก OpenFOAM ใช้เทรน differentiable surrogate ได้
=============================================================================
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.openfoam_case")

__all__ = [
    "OpenFOAMCaseConfig",
    "OpenFOAMCaseGenerator",
    "OpenFOAMFieldParser",
]


@dataclass
class OpenFOAMCaseConfig:
    """Configuration for generating an OpenFOAM case."""

    case_name: str = "sesi_cfd"
    case_dir: str = "./openfoam_cases"
    dimension: int = 3                    # 2 or 3

    # Geometry (blockMesh)
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 0.1
    z_min: float = 0.0
    z_max: float = 0.1
    nx: int = 100                         # cells in x
    ny: int = 20                          # cells in y
    nz: int = 20                          # cells in z (ignored if 2D)

    # Flow
    inlet_velocity: float = 1.0           # m/s
    kinematic_viscosity: float = 1e-5     # m^2/s
    density: float = 1000.0               # kg/m^3
    pressure_ref: float = 0.0             # Pa

    # Turbulence
    turbulence_model: str = "kOmegaSST"   # laminar | kEpsilon | kOmegaSST
    turbulence_intensity: float = 0.05
    turbulence_length_scale: float = 0.01

    # Solver
    solver: str = "simpleFoam"            # icoFoam | simpleFoam | pisoFoam | pimpleFoam
    transient: bool = False
    end_time: float = 100.0
    delta_t: float = 0.001
    write_interval: int = 10

    # Parallel
    n_processors: int = 4
    decompose_method: str = "scotch"

    # Output
    write_fields: List[str] = field(
        default_factory=lambda: ["U", "p", "k", "epsilon", "omega", "nut"]
    )


class OpenFOAMCaseGenerator:
    """สร้างเคส OpenFOAM ที่สมบูรณ์และรันได้จริง"""

    def __init__(self, cfg: OpenFOAMCaseConfig):
        self.cfg = cfg
        self.case_path = Path(cfg.case_dir) / cfg.case_name

    # ────────────────────────────────────────────────────────────────────
    # Directory structure
    # ────────────────────────────────────────────────────────────────────
    def create_directories(self) -> None:
        dirs = [
            self.case_path / "0",
            self.case_path / "constant" / "polyMesh",
            self.case_path / "system",
            self.case_path / "postProcess",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created case directories: {self.case_path}")

    # ────────────────────────────────────────────────────────────────────
    # 0/ — initial and boundary conditions
    # ────────────────────────────────────────────────────────────────────
    def write_U(self) -> Path:
        """เขียนไฟล์ 0/U — velocity field"""
        cfg = self.cfg
        dim = 3 if cfg.dimension == 3 else 2
        u_inlet = f"({cfg.inlet_velocity} 0 0)" if dim == 3 else f"({cfg.inlet_velocity} 0 0)"

        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform {u_inlet};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {u_inlet};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    topBottom
    {{
        type            symmetryPlane;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        path = self.case_path / "0" / "U"
        path.write_text(content)
        return path

    def write_p(self) -> Path:
        """เขียนไฟล์ 0/p — pressure field"""
        cfg = self.cfg
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {cfg.pressure_ref};

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {cfg.pressure_ref};
    }}
    walls
    {{
        type            zeroGradient;
    }}
    topBottom
    {{
        type            symmetryPlane;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        path = self.case_path / "0" / "p"
        path.write_text(content)
        return path

    def write_turbulence_fields(self) -> List[Path]:
        """เขียนไฟล์ 0/k, 0/epsilon หรือ 0/omega สำหรับ turbulence"""
        cfg = self.cfg
        paths = []

        if cfg.turbulence_model == "laminar":
            return paths

        # k
        k_val = 1.5 * (cfg.inlet_velocity * cfg.turbulence_intensity) ** 2
        k_content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_val};

boundaryField
{{
    inlet           {{ type fixedValue; value uniform {k_val}; }}
    outlet          {{ type zeroGradient; }}
    walls           {{ type kqRWallFunction; value uniform {k_val}; }}
    topBottom       {{ type symmetryPlane; }}
    frontAndBack    {{ type empty; }}
}}
"""
        k_path = self.case_path / "0" / "k"
        k_path.write_text(k_content)
        paths.append(k_path)

        if cfg.turbulence_model == "kEpsilon":
            eps_val = k_val ** 1.5 / cfg.turbulence_length_scale
            eps_content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      epsilon;
}}

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {eps_val};

boundaryField
{{
    inlet           {{ type fixedValue; value uniform {eps_val}; }}
    outlet          {{ type zeroGradient; }}
    walls           {{ type epsilonWallFunction; value uniform {eps_val}; }}
    topBottom       {{ type symmetryPlane; }}
    frontAndBack    {{ type empty; }}
}}
"""
            eps_path = self.case_path / "0" / "epsilon"
            eps_path.write_text(eps_content)
            paths.append(eps_path)

        elif cfg.turbulence_model == "kOmegaSST":
            omega_val = k_val ** 0.5 / (0.09 ** 0.25 * cfg.turbulence_length_scale)
            omega_content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega_val};

boundaryField
{{
    inlet           {{ type fixedValue; value uniform {omega_val}; }}
    outlet          {{ type zeroGradient; }}
    walls           {{ type omegaWallFunction; value uniform {omega_val}; }}
    topBottom       {{ type symmetryPlane; }}
    frontAndBack    {{ type empty; }}
}}
"""
            omega_path = self.case_path / "0" / "omega"
            omega_path.write_text(omega_content)
            paths.append(omega_path)

        # nut
        nut_content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet           {{ type calculated; value uniform 0; }}
    outlet          {{ type calculated; value uniform 0; }}
    walls           {{ type nutkWallFunction; value uniform 0; }}
    topBottom       {{ type symmetryPlane; }}
    frontAndBack    {{ type empty; }}
}}
"""
        nut_path = self.case_path / "0" / "nut"
        nut_path.write_text(nut_content)
        paths.append(nut_path)

        return paths

    # ────────────────────────────────────────────────────────────────────
    # constant/ — physical properties
    # ────────────────────────────────────────────────────────────────────
    def write_transport_properties(self) -> Path:
        cfg = self.cfg
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {cfg.kinematic_viscosity};
"""
        path = self.case_path / "constant" / "transportProperties"
        path.write_text(content)
        return path

    def write_turbulence_properties(self) -> Path:
        cfg = self.cfg
        if cfg.turbulence_model == "laminar":
            sim_type = "laminar"
            model_block = ""
        else:
            sim_type = "RAS"
            model_block = f"""
RASModel        {cfg.turbulence_model};

turbulence      on;

printCoeffs     on;
"""
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}

simulationType  {sim_type};
{model_block}
"""
        path = self.case_path / "constant" / "turbulenceProperties"
        path.write_text(content)
        return path

    def write_block_mesh_dict(self) -> Path:
        """เขียน constant/polyMesh/blockMeshDict"""
        cfg = self.cfg
        z_max = cfg.z_max if cfg.dimension == 3 else 0.01
        nz = cfg.nz if cfg.dimension == 3 else 1

        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({cfg.x_min} {cfg.y_min} {cfg.z_min})
    ({cfg.x_max} {cfg.y_min} {cfg.z_min})
    ({cfg.x_max} {cfg.y_max} {cfg.z_min})
    ({cfg.x_min} {cfg.y_max} {cfg.z_min})
    ({cfg.x_min} {cfg.y_min} {z_max})
    ({cfg.x_max} {cfg.y_min} {z_max})
    ({cfg.x_max} {cfg.y_max} {z_max})
    ({cfg.x_min} {cfg.y_max} {z_max})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({cfg.nx} {cfg.ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    walls
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    topBottom
    {{
        type symmetryPlane;
        faces ((3 7 6 2));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);

mergePatchPairs
(
);
"""
        path = self.case_path / "constant" / "polyMesh" / "blockMeshDict"
        path.write_text(content)
        return path

    # ────────────────────────────────────────────────────────────────────
    # system/ — solver configuration
    # ────────────────────────────────────────────────────────────────────
    def write_control_dict(self) -> Path:
        cfg = self.cfg
        app = cfg.solver
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {app};

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {cfg.end_time};
deltaT          {cfg.delta_t};
writeControl    timeStep;
writeInterval   {cfg.write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        path = self.case_path / "system" / "controlDict"
        path.write_text(content)
        return path

    def write_fv_schemes(self) -> Path:
        cfg = self.cfg
        # Upwind สำหรับ steady, limitedLinear สำหรับ transient
        div_scheme = "Gauss upwind" if not cfg.transient else "Gauss limitedLinear 1"
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}

ddtSchemes
{{
    default         Euler;
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      {div_scheme};
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
"""
        path = self.case_path / "system" / "fvSchemes"
        path.write_text(content)
        return path

    def write_fv_solution(self) -> Path:
        cfg = self.cfg
        if cfg.solver == "simpleFoam":
            solvers_block = """\
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    "(U|k|epsilon|omega|nut)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
}

relaxationFactors
{
    equations
    {
        U               0.9;
        k               0.7;
        epsilon         0.7;
        omega           0.7;
    }
}
"""
        else:
            solvers_block = """\
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    "(U|k|epsilon|omega|nut)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
}
"""
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

{solvers_block}
"""
        path = self.case_path / "system" / "fvSolution"
        path.write_text(content)
        return path

    def write_decompose_par_dict(self) -> Path:
        cfg = self.cfg
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {cfg.n_processors};

method          {cfg.decompose_method};

simpleCoeffs
{{
    n               (2 2 1);
    delta           0.001;
}}

hierarchicalCoeffs
{{
    n               (2 2 1);
    delta           0.001;
    order           xyz;
}}

manualCoeffs
{{
    dataFile        "";
}}
"""
        path = self.case_path / "system" / "decomposeParDict"
        path.write_text(content)
        return path

    # ────────────────────────────────────────────────────────────────────
    # run.sh — สคริปต์รันแบบครบวงจร
    # ────────────────────────────────────────────────────────────────────
    def write_run_script(self) -> Path:
        cfg = self.cfg
        content = f"""\
#!/bin/bash
#=============================================================================
# SESI OpenFOAM Case — Run Script
# Case: {cfg.case_name}
# Solver: {cfg.solver}
# Turbulence: {cfg.turbulence_model}
#=============================================================================
set -e

echo "=============================================="
echo "  SESI OpenFOAM Case: {cfg.case_name}"
echo "=============================================="

# 1. Generate mesh
echo "[1/4] Generating mesh with blockMesh..."
blockMesh

# 2. Check mesh quality
echo "[2/4] Checking mesh..."
checkMesh

# 3. Decompose for parallel (if n_processors > 1)
if [ {cfg.n_processors} -gt 1 ]; then
    echo "[3/4] Decomposing for parallel run ({cfg.n_processors} processors)..."
    decomposePar
    echo "[4/4] Running {cfg.solver} in parallel..."
    mpirun -np {cfg.n_processors} {cfg.solver} -parallel
    reconstructPar
else
    echo "[3/4] Running {cfg.solver}..."
    {cfg.solver}
    echo "[4/4] Done."
fi

echo "=============================================="
echo "  Simulation complete!"
echo "  Results in: {cfg.case_dir}/{cfg.case_name}"
echo "=============================================="
"""
        path = self.case_path / "run.sh"
        path.write_text(content)
        path.chmod(0o755)
        return path

    # ────────────────────────────────────────────────────────────────────
    # postProcess/extract_fields.py — แปลงผลลัพธ์เป็น PyTorch tensors
    # ────────────────────────────────────────────────────────────────────
    def write_extract_script(self) -> Path:
        content = '''\
"""
Extract OpenFOAM fields into PyTorch tensors for SESI training.
Run after simulation: python postProcess/extract_fields.py
"""
import os
import re
import sys
import torch
import numpy as np
from pathlib import Path

def parse_openfoam_scalar_field(filepath):
    """Parse an OpenFOAM scalar field file into numpy array."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find internalField
    match = re.search(r'internalField\\s+nonuniform\\s+List<scalar>\\s*\\n\\s*(\\d+)\\s*\\n\\(\\s*([\\s\\S]*?)\\)', content)
    if match:
        n = int(match.group(1))
        values = np.fromstring(match.group(2), sep=' ')
        return values

    # Uniform field
    match = re.search(r'internalField\\s+uniform\\s+([\\d.eE+-]+)', content)
    if match:
        return np.array([float(match.group(1))])

    return None


def parse_openfoam_vector_field(filepath):
    """Parse an OpenFOAM vector field file into numpy array (N, 3)."""
    with open(filepath, 'r') as f:
        content = f.read()

    match = re.search(
        r'internalField\\s+nonuniform\\s+List<vector>\\s*\\n\\s*(\\d+)\\s*\\n\\(\\s*([\\s\\S]*?)\\)',
        content
    )
    if match:
        n = int(match.group(1))
        raw = match.group(2)
        # Extract (x y z) tuples
        vectors = re.findall(r'\\(\\s*([\\d.eE+-]+)\\s+([\\d.eE+-]+)\\s+([\\d.eE+-]+)\\s*\\)', raw)
        return np.array(vectors, dtype=np.float64)

    return None


def extract_all_fields(case_dir, time_dir):
    """Extract all fields from a given time directory."""
    time_path = Path(case_dir) / time_dir
    fields = {}

    for f in time_path.iterdir():
        if f.is_file():
            name = f.name
            if name in ('U',):
                data = parse_openfoam_vector_field(f)
                if data is not None:
                    fields[name] = torch.tensor(data, dtype=torch.float32)
            else:
                data = parse_openfoam_scalar_field(f)
                if data is not None:
                    fields[name] = torch.tensor(data, dtype=torch.float32)

    return fields


if __name__ == "__main__":
    case_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    time_dirs = sorted([d for d in os.listdir(case_dir)
                        if d.replace('.','').isdigit()], key=float)

    if not time_dirs:
        print("No time directories found.")
        sys.exit(1)

    print(f"Found {len(time_dirs)} time directories")

    for td in time_dirs[-3:]:  # last 3 timesteps
        fields = extract_all_fields(case_dir, td)
        out_path = Path(case_dir) / "postProcess" / f"fields_t{td}.pt"
        torch.save(fields, out_path)
        print(f"Saved: {out_path} ({list(fields.keys())})")
'''
        path = self.case_path / "postProcess" / "extract_fields.py"
        path.write_text(content)
        return path

    # ────────────────────────────────────────────────────────────────────
    # Generate all
    # ────────────────────────────────────────────────────────────────────
    def generate_all(self) -> Dict[str, Path]:
        """สร้างไฟล์ทั้งหมด"""
        self.create_directories()
        files = {
            "U": self.write_U(),
            "p": self.write_p(),
            "transportProperties": self.write_transport_properties(),
            "turbulenceProperties": self.write_turbulence_properties(),
            "blockMeshDict": self.write_block_mesh_dict(),
            "controlDict": self.write_control_dict(),
            "fvSchemes": self.write_fv_schemes(),
            "fvSolution": self.write_fv_solution(),
            "decomposeParDict": self.write_decompose_par_dict(),
            "run.sh": self.write_run_script(),
            "extract_fields.py": self.write_extract_script(),
        }
        turb_files = self.write_turbulence_fields()
        for i, p in enumerate(turb_files):
            files[f"turb_{i}"] = p

        logger.info(f"Generated {len(files)} files for case '{self.cfg.case_name}'")
        return files


# ═══════════════════════════════════════════════════════════════════════
# OpenFOAM Field Parser — สำหรับโหลดผลลัพธ์มาเทรน
# ═══════════════════════════════════════════════════════════════════════

class OpenFOAMFieldParser:
    """
    โหลดผลลัพธ์จาก OpenFOAM case ที่รันเสร็จแล้ว
    แปลงเป็น PyTorch tensors สำหรับเทรน differentiable surrogate
    """

    def __init__(self, case_dir: str):
        self.case_dir = Path(case_dir)

    def list_timesteps(self) -> List[str]:
        """รายการ time directories"""
        dirs = []
        for d in self.case_dir.iterdir():
            if d.is_dir() and self._is_float(d.name):
                dirs.append(d.name)
        return sorted(dirs, key=float)

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def parse_field(self, field_path: Path) -> Optional[np.ndarray]:
        """Parse OpenFOAM field file (scalar หรือ vector)"""
        content = field_path.read_text()

        # Vector field
        if 'List<vector>' in content:
            match = re.search(
                r'internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\(\s*([\s\S]*?)\)',
                content
            )
            if match:
                raw = match.group(2)
                vectors = re.findall(
                    r'\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)', raw
                )
                return np.array(vectors, dtype=np.float64)

        # Scalar field
        match = re.search(
            r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(\s*([\s\S]*?)\)',
            content
        )
        if match:
            return np.fromstring(match.group(2), sep=' ', dtype=np.float64)

        # Uniform scalar
        match = re.search(r'internalField\s+uniform\s+([\d.eE+-]+)', content)
        if match:
            return np.array([float(match.group(1))])

        return None

    def load_timestep(self, time_dir: str) -> Dict[str, np.ndarray]:
        """โหลดทุก fields จาก timestep หนึ่ง"""
        tpath = self.case_dir / time_dir
        fields = {}
        if not tpath.exists():
            return fields

        for f in tpath.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                data = self.parse_field(f)
                if data is not None:
                    fields[f.name] = data
        return fields

    def to_torch_dataset(self, fields_list: List[Dict[str, np.ndarray]],
                         field_names: List[str] = None) -> Dict[str, 'torch.Tensor']:
        """แปลงเป็น PyTorch tensors"""
        import torch
        if field_names is None:
            field_names = ['U', 'p']

        dataset = {}
        for name in field_names:
            arrays = [f[name] for f in fields_list if name in f]
            if arrays:
                dataset[name] = torch.stack([
                    torch.tensor(a, dtype=torch.float32) for a in arrays
                ])
        return dataset
