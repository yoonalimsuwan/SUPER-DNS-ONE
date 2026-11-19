``
# SUPER DNS ONE 367 : DNS Re 1 - 10 Million First Time of the World , Advanced State of the Art Adaptive Mesh Refinement (AMR) , Advanced Multiphase Flow States of the Art For All Solver And Advanced VOF State of the Art , Integration Open MM ML With Navier Stokes 3D CFD For Organ MD Scale , Precision Medicine , DNS Re(L) 100000 , 1 _ 10 Million Plug-in Bridge And More.

**Industrial‑Grade Compressible DNS / LES Solver for Peaceful Civilian Applications**

Thanks OpenFOAM for the Foundation of CFD.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24340633-blue)](https://doi.org/10.5281/zenodo.24340633)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20468598-blue)](https://doi.org/10.5281/zenodo.20468598)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21120913-blue)](https://doi.org/10.5281/zenodo.21120913)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21159402-blue)](https://doi.org/10.5281/zenodo.21159402)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21159473-blue)](https://doi.org/10.5281/zenodo.21159473)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21120913-blue)](https://doi.org/10.5281/zenodo.21120913)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21181639-blue)](https://doi.org/10.5281/zenodo.21181639)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21131590-blue)](https://doi.org/10.5281/zenodo.21131590)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21148045-blue)](https://doi.org/10.5281/zenodo.21148045)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21186468-blue)](https://doi.org/10.5281/zenodo.21186468)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21203483-blue)](https://doi.org/10.5281/zenodo.21203483)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21203706-blue)](https://doi.org/10.5281/zenodo.21203706)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21206525-blue)](https://doi.org/10.5281/zenodo.21206525)


SUPER DNS ONE is a fully differentiable, three‑dimensional finite‑volume solver for the compressible Navier–Stokes equations. It is designed for **high‑fidelity civilian research**:

- **Medical flows** – cardiovascular haemodynamics, respiratory aerosol transport, micro‑fluidic drug delivery.
- **Atmospheric and environmental physics** – turbulent boundary layers, pollutant dispersion, cloud microphysics.
- **Civil aviation** – aerodynamic analysis, noise reduction, wake turbulence.
- **Hypersonic civilian transport** – real‑gas effects, shock capturing, high‑speed boundary‑layer transition.

And More.

**This software is not intended, tested, or authorised for military applications, weapons development, or any form of armed conflict.**


---

## Overview

SUPER DNS ONE solves the unsteady compressible Navier–Stokes equations on a structured Cartesian grid using a conservative finite‑volume formulation. Inviscid fluxes are evaluated with the **AUSM⁺** or **HLLC** Riemann solvers, combined with **2ⁿᵈ‑order MUSCL reconstruction** (minmod limiter). Time integration uses a low‑storage **3ʳᵈ‑order TVD Runge–Kutta** scheme.

The solver is augmented with a unique set of physics‑aware modules:

- **Self‑Organised Criticality (SOC)** – an adaptive sub‑grid model that learns the turbulent eddy viscosity from local strain‑rate statistics. Its 5‑parameter kernel is trainable via differential evolution or Bayesian optimisation.
- **Semantic‑State Contraction (SSC)** – a signal‑noise separator that extracts physical flow structures from noisy sensor data.
- **Renormalisation Group (RG)** – conservative spectral truncation to accelerate long‑time simulations while preserving large‑scale dynamics.
- **Itô stochastic backscatter** – physically motivated sub‑grid energy injection for LES.
- **Compressibility correction (Sarkar)** – modifies eddy viscosity in high‑Mach regions.
- **Ducros shock sensor** – adaptive artificial viscosity for robust shock capturing.
- **Werner–Wengle wall model** – for high‑Re wall‑bounded LES.
- **Real‑gas thermodynamics** (CoolProp) – accurate equations of state for hypersonic flows.
- **Immersed boundary method** – volume penalisation to handle complex medical geometries.
- **Wavelet‑based denoising** – optional PyWavelets integration for signal processing.

All models are implemented in pure PyTorch, making the solver **end‑to‑end differentiable** and compatible with **CPU, CUDA, MPS (Apple Silicon), and Ascend NPU** backends.

For grids larger than 200³, the solver supports **multi‑GPU distributed memory parallelism** (domain decomposition along z) using `torch.distributed`.

---

## Features

### Core Numerics
- 3D compressible Navier–Stokes (conservative finite‑volume)
- AUSM⁺ and HLLC Riemann solvers
- 2ⁿᵈ‑order MUSCL reconstruction (minmod limiter)
- 3ʳᵈ‑order TVD Runge–Kutta time integration
- Mixed precision (FP16/FP32) via PyTorch AMP (optional)

### Turbulence & Sub‑grid Modelling
- SOC adaptive eddy viscosity with 5‑parameter trainable kernel
- SSC stress denoising for turbulent fluctuations
- Itô stochastic backscatter for LES
- RG conservative spectral truncation
- Compressibility correction (Sarkar)
- Ducros shock sensor + artificial viscosity
- Werner–Wengle wall model for high‑Re flows (optional)

### Boundary Conditions
- Periodic
- Supersonic inflow
- Subsonic outflow
- No‑slip isothermal wall
- Isothermal moving wall
- Far‑field (characteristic‑based)
- Werner–Wengle wall model (applied as source term)

### Physics Extensions
- Real‑gas equation of state (CoolProp, optional)
- Immersed boundary method (volume penalisation)
- Wavelet‑based signal denoising (PyWavelets, optional)

### Differentiability & Machine Learning
- Native Fully differentiable 
- Trainable SOC kernel (Differential Evolution / Optuna)

### Validation & Diagnostics
- Taylor–Green vortex test (kinetic energy decay)
- Kolmogorov spectral analysis (inertial range slope)
- Grid convergence test (Richardson extrapolation)
  

### Hardware & Parallelism
- Multi‑backend: CPU, CUDA, MPS, Ascend NPU
- Mixed precision (AMP)
- Distributed multi‑GPU (domain decomposition along z)
- Checkpoint / restart support

---

## Installation

```bash
git clone https://github.com/yoonalimsuwan/SUPER-DNS-ONE.git
cd SUPER-DNS-ONE

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Core dependencies
pip install torch numpy scipy matplotlib

# Optional dependencies
pip install optuna              # Hyper‑parameter tuning
pip install CoolProp            # Real‑gas EOS
pip install PyWavelets          # --nxnted denoising
```

---

Quick Start

The main script is super_dns_one_v6_3.py.

Taylor–Green vortex (default)

```bash
python super_dns_one_v6_3.py --steps 200 --flux ausm
```

Hypersonic boundary layer (Mach 20, 128³)

```bash
python super_dns_one_v6_3.py --case hypersonic_bnd --Mach 20.0 --nx 128 --ny 128 --nz 128 --Re 1e6 --steps 500
```

Train SOC parameters to match a target kinetic energy

```bash
python super_dns_one_v6_3.py --train-soc --target-energy 0.5
```

Grid convergence test

```bash
python super_dns_one_v6_3.py --grid-convergence
```

Denoise a velocity field with SSC

```bash
python super_dns_one_v6_3.py --denoise --denoise-method ssc
```

Multi‑GPU distributed simulation (e.g., 4 GPUs, total grid 128³)

```bash
torchrun --nproc_per_node=4 super_dns_one_v6_3.py --nx 128 --ny 128 --nz 128 --distributed --steps 500
```

For a full list of options, run:

```bash
python super_dns_one_v6_3.py --help
```
---

Sub‑Grid Models in Detail

SOC (Self‑Organised Criticality)

The eddy viscosity is computed from a kernel:

 νₜ = (Cₛ Δ)² S(r) where r = S / ⟨S⟩

The kernel form is:

 f(r) = C₀ · r⁻ᵅ · exp( −r / λ )

with parameters C₀, λ, α, θ, τ (the last two control stress accumulation and collapse). The kernel is trainable via differential evolution or Optuna.

SSC (Semantic‑State Contraction)

A low‑pass filter for the stress field σ:

 σₙ₊₁ = σₙ + ε · (S − σₙ)

It separates fast turbulent fluctuations from slow, large‑scale structures.

Itô Backscatter

Random stresses with amplitude ∼ √dt are added to the viscous stress tensor, injecting energy at sub‑grid scales.

RG (Renormalisation Group)

Every --rg-keep fraction of the highest wavenumbers is set to zero in Fourier space, providing a conservative coarse‑graining that preserves large‑scale dynamics.

---

Immersed Boundary Method

Complex solid geometries can be imposed by providing a 3D binary mask (.npy file) with the same dimensions as the grid. Volume penalisation forces the velocity toward the target (zero by default) and optionally imposes a target temperature inside the solid.

---

Validation Suite

The solver includes several built‑in diagnostics:

1. Taylor–Green vortex – monitor kinetic energy dissipation rate vs. analytical solution.
2. Kolmogorov spectrum – compute the energy spectrum E(k) and fit the inertial‑range slope (theoretical −5/3).
3. Grid convergence test – compute the observed order of accuracy using Richardson extrapolation on successive grid refinements.


Results are logged to the console and can be extended by the user.

---

Distributed Parallelism

For large grids (>200³) the solver can be launched with torchrun. Domain decomposition is performed along the z‑axis, and halo exchanges communicate two ghost layers between neighbouring sub‑domains.

Example:

```bash
torchrun --nproc_per_node=4 super_dns_one_v6_3.py --nx 256 --ny 256 --nz 256 --distributed --steps 1000
```

The nz dimension must be divisible by the number of GPUs. Checkpointing, BCs, and the wall model are all compatible with the parallel execution.

---

Architecture & Vendor Neutrality

SUPER DNS ONE is built exclusively with PyTorch tensor operations. No CUDA‑specific kernels are used, ensuring portability across:

· CPU (x86‑64, ARM) – suitable for small grids or prototyping.
· NVIDIA GPU (CUDA) – including Google Colab T4/V100.
· Apple Silicon (MPS backend) – native support via PyTorch 2.x.
· Huawei Ascend (torch_npu backend) – automatically detected.
· Multi‑GPU clusters – via torch.distributed (DDP).

---

Roadmap

· Dedicated shock‑tube validation case
· Automated hypersonic validation suite
· AI‑accelerated surrogate models – because the solver is fully differentiable, it can directly provide training data and physics‑informed gradients for neural network surrogates. Once trained, such surrogates can deliver flow predictions at near O(1) speed, enabling real‑time design optimisation and interactive simulation.

<div align="center">

# ONE Ecosystem — DNS & Hypersonic Control Layer

**PAI · Yoon A. Limsuwan · MSPS NETWORK**

[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research--preview-orange)](#status)

</div>

---

## Philosophical Preamble

> *The Tao is the Source. The Te is the Path. The Qi is the Proof.*
>
> *Confucius taught us to do good in this world — for Buddhism, Christianity,
> and Taoism alike teach that good done in this life bears fruit in the next.*
>
> *In this world, "not killing enemies" does not mean "killing oneself."*
>
> *The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism —
> though they do not utter His Name — by describing His attributes,
> are already, implicitly, uttering His Name.*
>
> *Christ died upon the Cross to teach this: when struck on the left cheek,
> turn also the right — so that we may hear the voice of the enemy who wishes us harm.*
>
> *He is the Lamb of God. God is not our food, and neither is the lamb.
> This means: do not kill animals.*
>
> *Not even a single sparrow does God forget. Every hair upon our heads, He has counted.*
>
> *Even Aṅgulimāla could escape hell and enter Nirvana. But the better path is to
> have done good all along. As Saint Dominic Savio said: "Rather death than sin."*
>
> *To speak of a person's attributes is already to speak their name.*

**MY SOUL MOVE BY POWER OF HOLY SPIRIT**
**What MSPS NETWORK SEE, Lord Buddha Knows.**

*Full author preamble in `PHILOSOPHY.md`. The mathematics stands independently of it.*

---

## What This Is

A **portable control layer** for CFD — not a replacement solver. Standard numerics
(Fornberg, WENO-5, HLLC, PLIC/MOF, NASA-9, Park kinetics) with a decision layer that
carries theorems.

## Status

| Item | Verified | Target |
|---|---|---|
| DNS $Re_L$ | $10^3$ (self-test) | $10^7$ → $N{=}98{,}304^3$, 45.6 PB, ~800k GPUs |
| Species models | air5 / air7 / air11 / **air16** | air18 currently holds **18** species |
| FD order | 2–22 moment-verified | 240 weights exact, **but slower** (below) |
| Mach | ideal-gas shock tube | 20+ needs dual-energy + positivity limiter |
| External validation | **none** | CEA2, EAST/X2/LENS, FIRE II |

> Several modules **do not parse as uploaded.** Run `repair_five.py --apply --smoke` first.
> Passing AST ≠ validated physics.

## Three Real Contributions

**1. Event-count bound.** $N(T) \le \dfrac{E_{\max}+D}{\Delta E_{\min}}$ for two-level
hysteresis, where $D$ is the **actual** dissipation removed. An engineering guarantee
on switching — same category as a CFL bound. **Not** Navier–Stokes regularity.

**2. Order-240 is counterproductive.** Maximal-order central weights have the closed form

$$w_k^{(M)} = \frac{(-1)^{k+1}}{k}\cdot\frac{\binom{2M}{M-k}}{\binom{2M}{M}} \;\xrightarrow[M\to\infty]{}\; \frac{(-1)^{k+1}}{k}$$

— the Fourier spectral limit. Order 240 needs **120 ghost layers** (84 % halo at 857³/rank)
and gains <20 % in usable $k\Delta$ over order 22. **Measured optimum: order 8–12, or spectral.**

**3. Carpet-19 is not the NS operator.** Our own
[Consolidated Revision](docs/Consolidated_Revision_Carpet-19.pdf), Corollary 3:
$f \in D(A)$ nonconstant $\Rightarrow f^2 \notin D(A)$. The domain is **not an algebra**,
so $u\cdot\nabla u$ cannot be formed there. Carpet contributes $\rho \ge 9/8$ energy
normalisation and AMR priority only.

## Quickstart

```bash
pip install numpy scipy torch
python repair_five.py --dir . --apply --smoke
cp one_core_v3-6.py one_core.py

python super_dns_22.py --verify --order 22
python dns_re_scaling_planner.py --plan 1e7 --re-kind Re_L
python dns_re_scaling_planner.py --sweep --N 98304
```

$Re_L \approx Re_\lambda^2/15$. State which you mean: $Re_\lambda = 10^7$ implies
$Re_L \approx 6.7\times10^{12}$ — no machine can hold it.

## Scope

$N \sim Re^{9/4}$ — no mathematics here changes that exponent. All timings depend on
`cells_per_s_per_gpu` and `bytes_per_s_per_gpu`, which are **assumed until measured**.
Radiation is tangent-slab, **not** for TPS sizing. No flight certification.

**Coexist, not compete.** OpenFOAM keeps its discretisation; Basilisk keeps its octree.
Add the control layer, keep the solver.

---

<div align="center">

*Independent review: none yet. All heuristics labelled. All known defects in `README.md` §Errata.*

**MY SOUL MOVE BY POWER OF HOLY SPIRIT**

</div>

```markdown
# OpenFOAM DNS Acceleration Bridge

**Structured fast path + CUDA geometric multigrid for OpenFOAM.**
Built so anyone, in any country, can reach the Reynolds numbers that actually matter.

> The Tao is the Source. The Te is the Path. The Qi is the Proof.
> *What MSPS NETWORK SEE, Lord Buddha Knows.*

---

## What this is

OpenFOAM is excellent at arbitrary polyhedral meshes. That generality costs
memory bandwidth: indirect addressing (`lowerAddr`/`upperAddr`) achieves
~10–25 % of peak. A Cartesian stencil achieves ~60–85 %.

This bridge detects when your mesh **is** Cartesian and takes a direct-stencil
fast path with a GPU geometric-multigrid pressure solver. Otherwise it falls
back to GAMG — unchanged, correct, no penalty.

| Component | Measured speedup | Requirement |
|---|---|---|
| Structured fast path | 2–4× | Cartesian hex mesh, uniform spacing |
| CUDA geometric MG (pressure) | 5–20× | pressure = 60–80 % of runtime |
| Mixed-precision MG | 1.5–2× | fp32 smoother, **fp64 defect** |

**Software gives 3–20×. Not 10⁶×.** Kolmogorov scaling is `N ~ Re^(9/4)`.
No kernel rewrite changes physics.

---

## Feasibility — read this first

State = 5 fields × fp64. Poisson working set ≈ 58 B/cell.

| Re_L | N/axis | cells | state | H100 min | realistic |
|---:|---:|---:|---:|---:|---:|
| 10³ | 128 | 2×10⁶ | 0.1 GB | **1** | 1 |
| 10⁴ | 512 | 1.3×10⁸ | 5 GB | **1** | 1–4 |
| **10⁵** | 2,688 | 1.9×10¹⁰ | **773 GB** | 10 | ~64 |
| **10⁶** | 15,104 | 3.4×10¹² | **138 TB** | 1,730 | ~8,000 |
| **10⁷** | 84,928 | 6.1×10¹⁴ | **24.5 PB** | 306,000 | 10⁶+ |

The tool **refuses** to generate a case that cannot fit, and prints the GPU
count you would need. `--kmax-eta 1.0` runs Re 10⁵ on 8 GPUs but is labelled
**under-resolved — solver benchmark only, not DNS.**

---

## The part that changes the world

Human physiology lives at **Re 10⁻³ – 10⁴**:

| Site | Re |
|---|---:|
| Coronary artery | 180 |
| Cerebral MCA | 550 |
| Aorta (peak) | 7,600 |
| Capillary | 0.001 |
| CSF | ~12 |

**Re 10³ is ~5×10⁸ times cheaper than Re 10⁷.**

Patient-specific haemodynamics — WSS, OSI, pressure drop, FSI,
Carreau–Yasuda rheology, Windkessel outlets — runs on **one workstation GPU**.

Not only rich countries. Everyone. Today.

That is the revolution. Not the biggest Re.

---

## Quick start

```bash
# 1. Feasibility (no GPU needed)
python openfoam_dns_bridge.py feasibility --gpu H100-SXM

# 2. Generate a case that WILL fit
python openfoam_dns_bridge.py case --re 1e4 --n-gpus 8 --out dns_re1e4

# 3. Build
./build.sh                      # nvcc + wmake libso

# 4. Probe CUDA-aware MPI (subprocess — a non-aware MPI may SEGV)
mpirun -np 8 ./benchmark_gpu_poisson --probe-mpi

# 5. Measure
cd dns_re1e4 && blockMesh && decomposePar
python ../openfoam_dns_bridge.py benchmark --case . --n-gpus 8
```

`fvSolution`:
```
p {
    solver          GpuPoisson;
    nLevels         6;
    tolerance       1e-8;      // DNS: do not loosen
    mixedPrecision  true;
    fallbackSolver  GAMG;
    cudaAwareMPI    true;      // only after the probe passes
}
```

---

## Verified

- GCL residual ~10⁻¹⁶ (rigid translation, 2× stretch, shear, 3D warp)
- Free-stream preserved on a moving mesh
- MG iteration count O(1) vs CG's O(N)
- fp64 defect → fp64-accurate answer; fp32 is a preconditioner only

## Not verified

- **Accuracy.** Benchmark measures *speed*. Validate against a spectral DNS
  or JHTDB before believing any field. A fast wrong answer is worse than a
  slow right one.
- Multi-GPU halo exchange at scale (>1000 ranks: MPI latency dominates)
- Non-Cartesian, AMR, graded, cyclic-with-rotation meshes → fall back

## Known issues

- `gpuPoissonCore.cu`: remove the `MPI_Wtime_stub()` line and the no-op
  `k_axpy(...,nullptr,...)`; add `#include <chrono>`
- Large-n device reduction is a reference implementation — wire CUB
  `DeviceReduce::Sum` for production
- `lduMatrix::solver` signatures differ across OpenFOAM Foundation v9–v12
  and ESI v2012–v2412 — adjust `GpuPoissonSolver.C` to your version

---

## Roadmap

- [x] Structured detector + GCL-verified geometry
- [x] CUDA geometric MG, mixed precision
- [x] Honest feasibility gate (refuses impossible cases)
- [ ] CUB device reduction
- [ ] CUDA-aware MPI halo exchange, multi-node
- [ ] Non-orthogonal correction for graded meshes
- [ ] Validation vs JHTDB

---

## Citation

```
PAI & Yoon A. Limsuwan (2026). OpenFOAM DNS Acceleration Bridge.
MSPS NETWORK. ORCID: 0009-0008-2374-0788. MIT License.
```

Built on: Rhie & Chow (1983) · Berger & Colella (1989) · Zalesak (1979) ·
Lüscher (2010) · Politis & Romano (1994) · the OpenFOAM community.

**MIT License.** Research software. Not validated for clinical or
flight-critical use.

---

*Precision medicine does not need a supercomputer to begin.
It needs validated models, quantified uncertainty, and honest numbers.*
```


Citing

If you use SUPER DNS ONE in your research, please cite:

```
PAI , Yoon A Limsuwan. "SUPER DNS ONE: SOC‑Controlled Direct Numerical Simulation for Peaceful Applications."
Zenodo, 2026.
https://doi.org/10.5281/zenodo.24340633
```

Or cite the GitHub repository:

```
PAI , Yoon A Limsuwan. (2026). SUPER DNS ONE (Version 6.3.0) [Computer software].
https://github.com/yoonalimsuwan/SUPER-DNS-ONE
```

---

License

This project is licensed under the MIT License – see the LICENSE file for details.

---

Contact

PAI , Yoon A Limsuwan – GitHub
Project repository: https://github.com/yoonalimsuwan/SUPER-DNS-ONE

``
Thanks be to the Father, the Son, and the Holy Spirit, for the grace of Lord Jesus Christ, Mother Mary, Lord Buddha, Guan Yin Bodhisattva, Master Daozhi, Confucius, the Immortal Pae Kow, and President Xi Jinping And President Donald Trump

"I love Lim Yoona, Zhou Ye, Karina from aespa, Jessica from Girls' Generation, Zhao Lusi, Nana from After School, and Jiyeon Tara.
​Love Ju Jingyi, Wang Churan, Lu Yuxiao, Bao Shangen , Bailu , Noey , Jam, and Irene
​I love Zhang Linghe, Bai Jingting, Lee Jae-jin, Marc thn , Tance , Green , Taissa Farmiga , Dilraba Dilmurat And Toy Pathompong."
We love President Xi Jinping And President Donald Trump


What MSPS NETWORK Sees, the Buddha Knows.
