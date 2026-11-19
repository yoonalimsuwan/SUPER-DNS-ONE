#!/usr/bin/env python3
"""
=============================================================================
OpenFOAM DNS Acceleration Bridge — Python orchestration layer (other Files see in src)
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the
better path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT THIS BRIDGE DOES
=============================================================================
[1] Honest resource estimation for DNS at Re_L = 1e5, 1e6, 1e7
    Bandwidth-based, not FLOP-based (DNS pressure solve is memory-bound).
[2] Generates OpenFOAM case dictionaries optimised for DNS
    (fvSolution with the GPU MG solver, fvSchemes with 2nd-order
     energy-conserving convection, controlDict with ADIOS2 I/O).
[3] Detects whether the acceleration path is applicable to your mesh.
[4] Benchmark harness: measures ACTUAL speedup vs GAMG.
[5] Feasibility report: tells you which Re you can actually run.

=============================================================================
HONEST SCOPE — THE THREE THINGS THAT MATTER
=============================================================================
1. SOFTWARE OPTIMISATION GIVES 3-20x, NOT 10^6x.
   Re_L = 1e7 needs ~24 PB of state. No kernel rewrite changes that.
   The bridge REPORTS this rather than promising it.

2. THE STRUCTURED FAST PATH REQUIRES A CARTESIAN MESH.
   Arbitrary polyhedral meshes fall back to GAMG. That is correct
   behaviour, not a failure. OpenFOAM's generality is its value.

3. NOTHING HERE IS VALIDATED AGAINST A REFERENCE DNS.
   The benchmark measures SPEED. Accuracy must be verified separately
   against a spectral code or a published DNS dataset (e.g. JHTDB).
   A fast wrong answer is worse than a slow right one.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("of_dns")

VERSION = "1.0.0"

BANNER = f"""
=============================================================================
OpenFOAM DNS ACCELERATION BRIDGE v{VERSION}
Author: PAI & Yoon A Limsuwan / MSPS NETWORK
Structured fast path + CUDA geometric MG + honest resource estimation
Software gives 3-20x. Re 1e7 needs exascale hardware. Both are true.
=============================================================================
"""


# =============================================================================
# §1. GPU / TOOL DETECTION
# =============================================================================

@dataclass
class GpuSpec:
    name: str
    memory_gb: float
    bw_gb_s: float          # achievable HBM bandwidth
    fp64_tflops: float

GPU_CATALOG = {
    "H100-SXM":  GpuSpec("H100 SXM 80GB",   80, 3350, 34.0),
    "H100-PCIe": GpuSpec("H100 PCIe 80GB",  80, 2000, 26.0),
    "A100-80":   GpuSpec("A100 SXM 80GB",   80, 2039,  9.7),
    "A100-40":   GpuSpec("A100 PCIe 40GB",  40, 1555,  9.7),
    "MI300X":    GpuSpec("AMD MI300X 192GB",192, 5300, 81.7),
    "MI250X":    GpuSpec("AMD MI250X 128GB",128, 3277, 47.9),
    "L40S":      GpuSpec("L40S 48GB",        48,  864,  1.4),
    "RTX4090":   GpuSpec("RTX 4090 24GB",    24, 1008,  1.3),
}


def detect_environment() -> Dict[str, Any]:
    """Probe what is actually installed. Report honestly."""
    env = {
        "openfoam": None, "openfoam_version": None,
        "nvcc": None, "cuda_version": None,
        "gpus": [], "n_gpus": 0,
        "mpirun": None, "adios2": False, "amgx": False, "petsc": False,
    }

    # OpenFOAM
    for exe in ("simpleFoam", "pisoFoam", "icoFoam", "dnsFoam"):
        p = shutil.which(exe)
        if p:
            env["openfoam"] = p
            break
    if env["openfoam"]:
        try:
            r = subprocess.run([env["openfoam"], "-help"],
                               capture_output=True, text=True, timeout=15)
            for line in (r.stdout + r.stderr).splitlines():
                if "OpenFOAM" in line and ("v" in line or "version" in line.lower()):
                    env["openfoam_version"] = line.strip()[:120]
                    break
        except Exception:
            pass
    if os.environ.get("WM_PROJECT_VERSION"):
        env["openfoam_version"] = os.environ["WM_PROJECT_VERSION"]

    # CUDA
    env["nvcc"] = shutil.which("nvcc")
    if env["nvcc"]:
        try:
            r = subprocess.run([env["nvcc"], "--version"],
                               capture_output=True, text=True, timeout=15)
            for line in r.stdout.splitlines():
                if "release" in line:
                    env["cuda_version"] = line.strip()
                    break
        except Exception:
            pass

    # GPUs
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
            for line in r.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2:
                    env["gpus"].append({
                        "name": parts[0],
                        "memory_mb": float(parts[1]),
                    })
            env["n_gpus"] = len(env["gpus"])
        except Exception:
            pass
    elif shutil.which("rocm-smi"):
        try:
            r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                               capture_output=True, text=True, timeout=20)
            n = r.stdout.count("GPU[")
            env["n_gpus"] = max(n, 0)
            env["gpus"] = [{"name": "AMD GPU (rocm-smi)", "memory_mb": 0}] * n
        except Exception:
            pass

    env["mpirun"] = shutil.which("mpirun") or shutil.which("mpiexec")
    env["adios2"] = bool(shutil.which("adios2_iotest") or
                         Path("/usr/lib/x86_64-linux-gnu/libadios2_core.so").exists())
    env["amgx"] = bool(os.environ.get("AMGX_DIR") or
                       Path("/opt/amgx").exists())
    env["petsc"] = bool(os.environ.get("PETSC_DIR"))
    return env


# =============================================================================
# §2. HONEST RESOURCE ESTIMATION  (bandwidth-based)
# =============================================================================

@dataclass
class DnsTarget:
    """A DNS target specified by the integral-scale Reynolds number."""
    Re_L: float = 1e5
    L: float = 1.0                  # integral length scale [m]
    u_prime: float = 1.0            # rms velocity [m/s]
    kmax_eta: float = 1.5           # resolution criterion (>=1.5 for DNS)
    n_fields: int = 5               # u,v,w,p + work
    bytes_per_scalar: int = 8       # fp64
    n_turnovers: float = 10.0       # simulated eddy turnover times
    cfl: float = 0.5
    mg_passes_per_step: int = 20    # V-cycle: ~5 levels x 4 sweeps
    bandwidth_efficiency: float = 0.70
    precision_note: str = "fp64 state, fp32 MG preconditioner"

    @property
    def n_per_axis(self) -> int:
        """
        k_max * eta >= kmax_eta, with k_max = pi*N/L and eta/L = Re^(-3/4):
            N >= (kmax_eta/pi) * Re^(3/4)
        Round UP to a multiple of 64 so MG coarsening and FFT both work.
        """
        n = (self.kmax_eta/math.pi) * self.Re_L**0.75
        return int(math.ceil(n/64.0)*64)

    @property
    def n_cells(self) -> int:
        return self.n_per_axis**3

    @property
    def state_bytes(self) -> int:
        return self.n_cells*self.n_fields*self.bytes_per_scalar

    @property
    def state_gb(self) -> float:
        return self.state_bytes/1e9

    @property
    def dx(self) -> float:
        return self.L/self.n_per_axis

    @property
    def dt(self) -> float:
        return self.cfl*self.dx/max(self.u_prime, 1e-12)

    @property
    def turnover_time(self) -> float:
        return self.L/max(self.u_prime, 1e-12)

    @property
    def n_steps(self) -> int:
        return int(math.ceil(self.n_turnovers*self.turnover_time/self.dt))

    def gpus_for_memory(self, gpu: GpuSpec,
                        headroom: float = 0.6) -> int:
        """Usable fraction of GPU memory (leave room for halos, MG levels)."""
        usable = gpu.memory_gb*headroom
        return max(1, int(math.ceil(self.state_gb/usable)))

    def seconds_per_step(self, gpu: GpuSpec, n_gpus: int) -> float:
        """
        BANDWIDTH model, not FLOP model. The pressure Poisson MG and the
        convection stencil are both memory-bound at these grid sizes.
            bytes moved / step ~ N * n_fields * 8 * mg_passes
        """
        bytes_step = (self.n_cells*self.n_fields*self.bytes_per_scalar
                      *self.mg_passes_per_step)
        agg_bw = n_gpus*gpu.bw_gb_s*1e9*self.bandwidth_efficiency
        return bytes_step/agg_bw

    def report(self, gpu_key: str = "H100-SXM") -> Dict[str, Any]:
        gpu = GPU_CATALOG[gpu_key]
        n_min = self.gpus_for_memory(gpu, headroom=1.0)
        n_real = self.gpus_for_memory(gpu, headroom=0.6)
        # Use 4x the memory-minimum as a realistic working configuration
        n_work = max(n_real, 4*n_min)
        sps = self.seconds_per_step(gpu, n_work)
        total = sps*self.n_steps
        return {
            "Re_L": self.Re_L,
            "n_per_axis": self.n_per_axis,
            "n_cells": self.n_cells,
            "n_cells_human": f"{self.n_cells:.3e}",
            "state_gb": self.state_gb,
            "state_human": (f"{self.state_gb/1e6:.2f} PB"
                            if self.state_gb > 1e6 else
                            f"{self.state_gb/1e3:.2f} TB"
                            if self.state_gb > 1e3 else
                            f"{self.state_gb:.1f} GB"),
            "dx_m": self.dx,
            "dt_s": self.dt,
            "n_steps": self.n_steps,
            "gpu": gpu.name,
            "gpus_memory_absolute_min": n_min,
            "gpus_memory_with_headroom": n_real,
            "gpus_working_estimate": n_work,
            "seconds_per_step": sps,
            "total_wall_seconds": total,
            "total_wall_human": self._human_time(total),
            "model": "BANDWIDTH-based (pressure MG is memory-bound)",
            "precision": self.precision_note,
            "NOT_INCLUDED": [
                "MPI halo exchange latency (dominates above ~1000 ranks)",
                "I/O time (use ADIOS2; can be 10-30% if unoptimised)",
                "checkpoint/restart overhead",
                "queue wait time",
            ],
        }

    @staticmethod
    def _human_time(s: float) -> str:
        if s < 120:      return f"{s:.1f} s"
        if s < 7200:     return f"{s/60:.1f} min"
        if s < 172800:   return f"{s/3600:.1f} h"
        if s < 3.15e7:   return f"{s/86400:.1f} days"
        return f"{s/3.15e7:.1f} years"


def feasibility_table(gpu_key: str = "H100-SXM") -> List[Dict[str, Any]]:
    out = []
    for re in (1e4, 1e5, 1e6, 1e7):
        t = DnsTarget(Re_L=re)
        out.append(t.report(gpu_key))
    return out


# =============================================================================
# §3. OPENFOAM CASE GENERATION (DNS-optimised dictionaries)
# =============================================================================

FV_SOLUTION_DNS = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM DNS Acceleration Bridge                |
|  \\    /   O peration     |                                                 |
|   \\  /    A nd           | Generated by openfoam_dns_bridge.py             |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    // ---------------------------------------------------------------
    // PRESSURE: the 60-80% bottleneck.
    // GpuPoisson is the CUDA geometric-MG solver from this bridge.
    // It self-detects structured topology; if the mesh is NOT
    // Cartesian it falls back internally to the smoother on the
    // ldu matrix (still correct, just not accelerated).
    // ---------------------------------------------------------------
    p
    {
        solver          GpuPoisson;
        nLevels         5;          // reduced automatically if grid too small
        preSweeps       2;
        postSweeps      2;
        maxOuter        30;         // FP64 iterative-refinement outer iters
        tolerance       1e-8;       // DNS: do NOT loosen this
        relTol          0;          // absolute, every timestep
        mixedPrecision  true;       // fp32 smoother, fp64 defect
        fallbackSolver  GAMG;       // used if structured detection fails
    }

    pFinal
    {
        $p;
        tolerance       1e-9;       // tighter on the final PISO corrector
    }

    // ---------------------------------------------------------------
    // MOMENTUM: diagonal-dominant, cheap. PBiCGStab + DILU is fine.
    // ---------------------------------------------------------------
    "(U|k|epsilon|omega)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0;
    }
}

PISO
{
    nCorrectors                 2;
    nNonOrthogonalCorrectors    0;   // Cartesian: exactly 0 needed
    pRefCell                    0;
    pRefValue                   0;
    consistent                  no;
}

// For PIMPLE (transient with outer correctors) use:
PIMPLE
{
    nOuterCorrectors            1;   // DNS: 1 is enough at CFL<0.5
    nCorrectors                 2;
    nNonOrthogonalCorrectors    0;
    momentumPredictor           yes;
}
// ************************************************************************* //
"""

FV_SCHEMES_DNS = r"""/*--------------------------------*- C++ -*----------------------------------*\
| OpenFOAM DNS Acceleration Bridge — fvSchemes                                |
|                                                                             |
| DNS SCHEME CHOICE — WHY                                                     |
|   div(phi,U)  Gauss linear;   <- 2nd-order central, ENERGY CONSERVING       |
|   NOT upwind / linearUpwind / limitedLinear: those add numerical            |
|   dissipation that competes with the physical dissipation you are           |
|   trying to resolve. At DNS resolution central is stable and correct.       |
|                                                                             |
|   laplacian   Gauss linear corrected;  <- 'corrected' is a no-op on a       |
|   Cartesian mesh (non-orthogonality = 0) but harmless.                      |
|                                                                             |
|   ddt         backward;  <- 2nd-order implicit. CrankNicolson 0.9 is        |
|   also acceptable; Euler is 1st-order and NOT recommended for DNS.          |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         backward;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;      // ENERGY-CONSERVING central
    div(phi,k)      Gauss linear;
    div(phi,epsilon) Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
// ************************************************************************* //
"""


def control_dict_dns(dt: float, end_time: float,
                     write_interval: float,
                     use_adios2: bool = True) -> str:
    io_block = (
        '    // ADIOS2 removes the collated-I/O bottleneck at scale\n'
        '    writeFormat     adios2;\n'
        '    writeCompression off;\n'
        if use_adios2 else
        '    writeFormat     binary;\n'
        '    writeCompression off;\n'
    )
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM DNS Acceleration Bridge — controlDict                              |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     pisoFoam;

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.6g};

deltaT          {dt:.6e};
writeControl    adjustableRunTime;
writeInterval   {write_interval:.6g};
purgeWrite      3;

{io_block}
timeFormat      general;
timePrecision   10;
runTimeModifiable yes;

adjustTimeStep  yes;
maxCo           0.5;
maxDeltaT       {dt*2:.6e};

libs
(
    "libGpuPoissonSolver.so"
);

functions
{{
    // Minimal in-situ diagnostics: do NOT write full fields every step
    fieldAverage1
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        fields
        (
            U {{ mean on; prime2Mean on; base time; }}
            p {{ mean on; prime2Mean on; base time; }}
        );
    }}

    // Energy spectrum + dissipation: the quantities you validate DNS with
    turbulenceFields1
    {{
        type            turbulenceFields;
        libs            (fieldFunctionObjects);
        fields          (k epsilon R);
        writeControl    writeTime;
    }}
}}
// ************************************************************************* //
"""


def blockmesh_dict_dns(n: int, L: float = 1.0,
                       periodic: Tuple[bool, bool, bool] = (True, True, True)
                       ) -> str:
    """Uniform Cartesian brick — the ONLY mesh the fast path accepts."""
    px, py, pz = periodic

    def patch(name, axis, side):
        if axis == 0:
            f = ("(0 4 7 3)" if side == 0 else "(1 2 6 5)")
        elif axis == 1:
            f = ("(0 1 5 4)" if side == 0 else "(3 7 6 2)")
        else:
            f = ("(0 3 2 1)" if side == 0 else "(4 5 6 7)")
        per = (px, py, pz)[axis]
        if per:
            nbr = {("xMin"): "xMax", ("xMax"): "xMin",
                   ("yMin"): "yMax", ("yMax"): "yMin",
                   ("zMin"): "zMax", ("zMax"): "zMin"}[name]
            return (f"    {name}\n    {{\n        type            cyclic;\n"
                    f"        neighbourPatch  {nbr};\n"
                    f"        faces\n        (\n            {f}\n        );\n    }}\n")
        return (f"    {name}\n    {{\n        type            wall;\n"
                f"        faces\n        (\n            {f}\n        );\n    }}\n")

    patches = "".join([
        patch("xMin", 0, 0), patch("xMax", 0, 1),
        patch("yMin", 1, 0), patch("yMax", 1, 1),
        patch("zMin", 2, 0), patch("zMax", 2, 1),
    ])

    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM DNS Acceleration Bridge — blockMeshDict                            |
| n = {n} per axis  ->  {n**3:.3e} cells                                       |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   {L:.8g};

vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 1) (1 0 1) (1 1 1) (0 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n} {n} {n}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
{patches});
// ************************************************************************* //
"""


def decompose_par_dict(n_gpus: int) -> str:
    """Cartesian decomposition — REQUIRED for the structured fast path."""
    # Factor n_gpus into a near-cubic triple
    best = (1, 1, n_gpus)
    best_score = float("inf")
    for a in range(1, int(round(n_gpus**(1/3)))+2):
        if n_gpus % a: continue
        rem = n_gpus//a
        for b in range(1, int(round(rem**0.5))+2):
            if rem % b: continue
            c = rem//b
            score = max(a, b, c)/min(a, b, c)
            if score < best_score:
                best_score, best = score, (a, b, c)
    a, b, c = best
    return f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object decomposeParDict;
}}
numberOfSubdomains  {n_gpus};
method              simple;      // Cartesian slab split — REQUIRED
coeffs
{{
    n   ({a} {b} {c});
}}
"""


def generate_case(out_dir: str, target: DnsTarget,
                  n_gpus: int, use_adios2: bool = True) -> Dict[str, str]:
    root = Path(out_dir)
    (root/"system").mkdir(parents=True, exist_ok=True)
    (root/"constant").mkdir(parents=True, exist_ok=True)
    (root/"0").mkdir(parents=True, exist_ok=True)

    files = {}
    n = target.n_per_axis
    writes = {
        "system/fvSolution":       FV_SOLUTION_DNS,
        "system/fvSchemes":        FV_SCHEMES_DNS,
        "system/controlDict":      control_dict_dns(
            target.dt, target.n_turnovers*target.turnover_time,
            target.turnover_time/2, use_adios2),
        "system/blockMeshDict":    blockmesh_dict_dns(n, target.L),
        "system/decomposeParDict": decompose_par_dict(n_gpus),
    }
    for rel, content in writes.items():
        p = root/rel
        p.write_text(content)
        files[rel] = str(p)

    log.info(f"[case] generated {len(files)} dictionaries in {root}")
    log.warning(f"[case] n={n} per axis -> {n**3:.3e} cells. "
                f"blockMesh will need {target.state_gb:.0f} GB just for the mesh. "
                f"Do NOT run blockMesh on a login node.")
    return files


# =============================================================================
# §4. BENCHMARK HARNESS — measures ACTUAL speedup
# =============================================================================

def benchmark_solver(case_dir: str, solver_exe: str = "pisoFoam",
                     n_steps: int = 20, parallel: int = 0,
                     solvers: Tuple[str, ...] = ("GAMG", "GpuPoisson")
                     ) -> Dict[str, Any]:
    """
    Run the SAME case with each pressure solver and time it.
    Reports the measured ratio, not a claimed one.
    """
    root = Path(case_dir)
    if not (root/"system/fvSolution").exists():
        return {"status": "ERROR", "reason": "not an OpenFOAM case"}
    exe = shutil.which(solver_exe)
    if not exe:
        return {"status": "UNAVAILABLE",
                "reason": f"{solver_exe} not on PATH; source OpenFOAM bashrc"}

    results = {}
    base = (root/"system/fvSolution").read_text()

    for s in solvers:
        # Patch the p solver block
        patched = base.replace("solver          GpuPoisson;",
                               f"solver          {s};")
        patched = patched.replace("solver          GAMG;",
                                  f"solver          {s};")
        (root/"system/fvSolution").write_text(patched)

        cmd = ([shutil.which("mpirun"), "-np", str(parallel), exe, "-parallel"]
               if parallel > 1 and shutil.which("mpirun")
               else [exe])
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=str(root), capture_output=True,
                               text=True, timeout=3600)
            wall = time.time() - t0
            results[s] = {
                "status": "OK" if r.returncode == 0 else "FAILED",
                "wall_s": wall,
                "returncode": r.returncode,
                "stderr_tail": r.stderr[-800:] if r.returncode else "",
            }
        except subprocess.TimeoutExpired:
            results[s] = {"status": "TIMEOUT"}
        except Exception as e:
            results[s] = {"status": "ERROR", "reason": str(e)}

    (root/"system/fvSolution").write_text(base)

    ok = {k: v for k, v in results.items() if v.get("status") == "OK"}
    if len(ok) >= 2:
        keys = list(ok)
        slow = max(ok, key=lambda k: ok[k]["wall_s"])
        fast = min(ok, key=lambda k: ok[k]["wall_s"])
        results["_speedup"] = {
            "baseline": slow, "accelerated": fast,
            "ratio": ok[slow]["wall_s"]/max(ok[fast]["wall_s"], 1e-9),
            "NOTE": ("MEASURED on this case/mesh/hardware. "
                     "Speed is not accuracy — validate the solution "
                     "separately against a reference DNS."),
        }
    return results


# =============================================================================
# §5. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 environment probe
    print("\n[T1] environment detection (honest)")
    env = detect_environment()
    for k in ("openfoam", "openfoam_version", "nvcc", "cuda_version",
              "n_gpus", "mpirun", "adios2", "amgx", "petsc"):
        print(f"      {k:20s} {env[k]}")
    rep("probe returns a dict", isinstance(env, dict))
    rep("no fabricated GPU count", env["n_gpus"] == len(env["gpus"])
        or env["n_gpus"] == 0)

    # T2 resolution scaling law
    print("\n[T2] Kolmogorov resolution scaling N ~ Re^(3/4)")
    for re in (1e4, 1e5, 1e6, 1e7):
        t = DnsTarget(Re_L=re)
        n = t.n_per_axis
        theory = (1.5/math.pi)*re**0.75
        print(f"      Re={re:.0e}  N/axis={n:>7d}  "
              f"(theory {theory:.0f})  cells={t.n_cells:.3e}  "
              f"state={t.report()['state_human']}")
        rep(f"Re={re:.0e}: N >= theory", n >= theory, f"{n} >= {theory:.0f}")
        rep(f"Re={re:.0e}: N divisible by 64", n % 64 == 0)

    # T3 the honest feasibility verdict
    print("\n[T3] feasibility — what you can ACTUALLY run")
    rows = feasibility_table("H100-SXM")
    print(f"      {'Re_L':>8} {'cells':>12} {'state':>12} "
          f"{'GPUs(min)':>10} {'GPUs(work)':>11} {'wall':>12}")
    for r in rows:
        print(f"      {r['Re_L']:>8.0e} {r['n_cells']:>12.2e} "
              f"{r['state_human']:>12} {r['gpus_memory_absolute_min']:>10d} "
              f"{r['gpus_working_estimate']:>11d} {r['total_wall_human']:>12}")
    rep("Re=1e5 needs >1 GPU", rows[1]["gpus_memory_absolute_min"] > 1)
    rep("Re=1e6 needs >1000 GPUs", rows[2]["gpus_memory_absolute_min"] > 1000)
    rep("Re=1e7 needs >100000 GPUs",
        rows[3]["gpus_memory_absolute_min"] > 100000,
        f"{rows[3]['gpus_memory_absolute_min']:,}")
    print("      → software optimisation gives 3-20x, NOT the 10^6x")
    print("        needed to make Re=1e7 fit on a small cluster.")

    # T4 timestep and step count
    print("\n[T4] CFL timestep and total steps")
    t = DnsTarget(Re_L=1e5)
    print(f"      dx={t.dx:.3e} m   dt={t.dt:.3e} s   "
          f"steps={t.n_steps:,} for {t.n_turnovers} turnovers")
    rep("dt < dx/u (CFL respected)", t.dt <= t.dx/t.u_prime)
    rep("step count is large (DNS is expensive)", t.n_steps > 10000,
        f"{t.n_steps:,}")

    # T5 case generation
    print("\n[T5] OpenFOAM case generation")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        small = DnsTarget(Re_L=1e3, n_turnovers=1.0)
        files = generate_case(td, small, n_gpus=8)
        rep("5 dictionaries written", len(files) == 5, f"{len(files)}")
        fvS = (Path(td)/"system/fvSolution").read_text()
        rep("GpuPoisson registered", "GpuPoisson" in fvS)
        rep("fallback to GAMG declared", "fallbackSolver  GAMG" in fvS)
        rep("DNS tolerance not loosened", "tolerance       1e-8" in fvS)
        fvSch = (Path(td)/"system/fvSchemes").read_text()
        rep("central (energy-conserving) convection",
            "div(phi,U)      Gauss linear;" in fvSch)
        rep("NOT upwind (would add numerical dissipation)",
            "upwind" not in fvSch.split("divSchemes")[1].split("laplacian")[0])
        bm = (Path(td)/"system/blockMeshDict").read_text()
        rep("uniform grading (required by fast path)",
            "simpleGrading (1 1 1)" in bm)
        dp = (Path(td)/"system/decomposeParDict").read_text()
        rep("Cartesian decomposition (simple, not scotch)",
            "method              simple" in dp)
        cd = (Path(td)/"system/controlDict").read_text()
        rep("plugin library declared", "libGpuPoissonSolver.so" in cd)

    # T6 decomposition factorisation
    print("\n[T6] Cartesian rank decomposition")
    for ng in (8, 64, 256, 1024):
        d = decompose_par_dict(ng)
        trip = d.split("n   (")[1].split(")")[0].split()
        a, b, c = map(int, trip)
        rep(f"{ng} ranks factorise exactly", a*b*c == ng, f"({a} {b} {c})")
        rep(f"{ng} ranks near-cubic", max(a,b,c)/min(a,b,c) <= 4.0,
            f"aspect {max(a,b,c)/min(a,b,c):.1f}")

    # T7 GPU catalogue sanity
    print("\n[T7] GPU catalogue (bandwidth model inputs)")
    for k, g in GPU_CATALOG.items():
        print(f"      {k:10s} {g.name:22s} {g.memory_gb:>4.0f} GB  "
              f"{g.bw_gb_s:>5.0f} GB/s  {g.fp64_tflops:>5.1f} TFLOP/s(fp64)")
    rep("H100 fp64 is NOT its bf16 tensor number",
        GPU_CATALOG["H100-SXM"].fp64_tflops < 100,
        "34 TFLOP/s fp64, not 990 TFLOP/s bf16")
    rep("RTX4090 fp64 is weak (consumer card)",
        GPU_CATALOG["RTX4090"].fp64_tflops < 2.0)

    # T8 benchmark harness refuses gracefully
    print("\n[T8] benchmark harness honesty")
    r = benchmark_solver("/tmp/definitely_not_a_case")
    rep("non-case reported as ERROR, not fabricated speedup",
        r.get("status") == "ERROR", str(r.get("reason", ""))[:60])

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
WHAT IS REAL
  Kolmogorov resolution scaling, CFL timestep, bandwidth-based cost model,
  Cartesian rank factorisation, DNS-appropriate OpenFOAM dictionaries
  (central convection, tight pressure tolerance, no non-orthogonal
   correctors, ADIOS2 I/O, structured-required decomposition).

WHAT THE PLUGIN GIVES (measured elsewhere, verify on YOUR case)
  structured fast path       2-4x   (Cartesian mesh only)
  CUDA geometric MG          5-20x  (pressure = 60-80% of runtime)
  mixed-precision MG         1.5-2x (fp32 smoother, fp64 defect)

WHAT IT DOES NOT GIVE
  Re_L = 1e7 on a small cluster. That needs ~24 PB of state and
  ~10^5-10^6 GPUs. No software rewrite changes the Kolmogorov scaling.

NOT VALIDATED
  Speed is measured by the benchmark harness. ACCURACY is not.
  Validate against a spectral DNS or JHTDB before believing any field.
""")
    return ok


# =============================================================================
# §6. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="OpenFOAM DNS acceleration bridge")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "probe", "feasibility",
                             "case", "benchmark"])
    ap.add_argument("--re", type=float, default=1e5, dest="re_l")
    ap.add_argument("--gpu", default="H100-SXM", choices=list(GPU_CATALOG))
    ap.add_argument("--n-gpus", type=int, default=64)
    ap.add_argument("--turnovers", type=float, default=10.0)
    ap.add_argument("--out", default="dns_case")
    ap.add_argument("--case", default=None)
    ap.add_argument("--no-adios2", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    out: Dict[str, Any] = {}

    if a.cmd == "probe":
        out = detect_environment()
        print(json.dumps(out, indent=2, default=str))

    elif a.cmd == "feasibility":
        rows = feasibility_table(a.gpu)
        print(BANNER)
        print(f"{'Re_L':>9} {'N/axis':>8} {'cells':>12} {'state':>11} "
              f"{'GPUmin':>8} {'GPUwork':>9} {'wall':>12}")
        print("-"*78)
        for r in rows:
            print(f"{r['Re_L']:>9.0e} {r['n_per_axis']:>8d} "
                  f"{r['n_cells']:>12.2e} {r['state_human']:>11} "
                  f"{r['gpus_memory_absolute_min']:>8d} "
                  f"{r['gpus_working_estimate']:>9d} "
                  f"{r['total_wall_human']:>12}")
        print("-"*78)
        print("Software optimisation: 3-20x. Kolmogorov scaling: Re^(9/4).")
        print("The second dominates. Re=1e7 is an exascale problem.")
        out = {"feasibility": rows}

    elif a.cmd == "case":
        t = DnsTarget(Re_L=a.re_l, n_turnovers=a.turnovers)
        rep = t.report(a.gpu)
        print(json.dumps(rep, indent=2, default=str))
        if rep["gpus_memory_absolute_min"] > a.n_gpus:
            print(f"\n*** REFUSING to generate: needs "
                  f"{rep['gpus_memory_absolute_min']} GPUs minimum for memory, "
                  f"you declared {a.n_gpus}. ***")
            print("    Generating the case would produce something that "
                  "cannot run. Lower --re or raise --n-gpus.")
            return 1
        files = generate_case(a.out, t, a.n_gpus, not a.no_adios2)
        out = {"target": rep, "files": files}
        print(f"\nCase written to {a.out}/")
        print("Next: blockMesh -> decomposePar -> mpirun -np "
              f"{a.n_gpus} pisoFoam -parallel")

    elif a.cmd == "benchmark":
        if not a.case:
            print("--case required"); return 1
        out = benchmark_solver(a.case, parallel=a.n_gpus)
        print(json.dumps(out, indent=2, default=str))

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
