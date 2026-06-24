``
# SUPER DNS ONE

**Industrial‑Grade Compressible DNS / LES Solver for Peaceful Civilian Applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20821780-blue)](https://doi.org/10.5281/zenodo.20821780)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20468598-blue)](https://doi.org/10.5281/zenodo.20468598)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20633681-blue)](https://doi.org/10.5281/zenodo.20633681)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20623622-blue)](https://doi.org/10.5281/zenodo.20623622)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20755856-blue)](https://doi.org/10.5281/zenodo.20755856)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20755892-blue)](https://doi.org/10.5281/zenodo.20755892)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)


SUPER DNS ONE is a fully differentiable, three‑dimensional finite‑volume solver for the compressible Navier–Stokes equations. It is designed for **high‑fidelity civilian research**:

- **Medical flows** – cardiovascular haemodynamics, respiratory aerosol transport, micro‑fluidic drug delivery.
- **Atmospheric and environmental physics** – turbulent boundary layers, pollutant dispersion, cloud microphysics.
- **Civil aviation** – aerodynamic analysis, noise reduction, wake turbulence.
- **Hypersonic civilian transport** – real‑gas effects, shock capturing, high‑speed boundary‑layer transition.

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
- Batalin–Vilkovisky (BV) consistency monitors (energy, divergence, stress balance)

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
torchrun --nproc_per_node=4 super_dns_one_v6.2.py --nx 128 --ny 128 --nz 128 --distributed --steps 500
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
4. BV conservation checks – track maximum divergence, stress balance, and energy history.

Results are logged to the console and can be extended by the user.

---

Distributed Parallelism

For large grids (>200³) the solver can be launched with torchrun. Domain decomposition is performed along the z‑axis, and halo exchanges communicate two ghost layers between neighbouring sub‑domains.

Example:

```bash
torchrun --nproc_per_node=4 super_dns_one_v6.py --nx 256 --ny 256 --nz 256 --distributed --steps 1000
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
· Adaptive mesh refinement (AMR) support
· Multi‑phase flow extensions (volume‑of‑fluid)

---

Citing

If you use SUPER DNS ONE in your research, please cite:

```
Yoon A Limsuwan. "SUPER DNS ONE: SOC‑Controlled Direct Numerical Simulation for Peaceful Applications."
Zenodo, 2026.
https://doi.org/10.5281/zenodo.20821780
```

Or cite the GitHub repository:

```
Yoon A Limsuwan. (2026). SUPER DNS ONE (Version 1.0.0) [Computer software].
https://github.com/yoonalimsuwan/SUPER-DNS-ONE
```

---

License

This project is licensed under the MIT License – see the LICENSE file for details.

---

Contact

Yoon A Limsuwan – GitHub
Project repository: https://github.com/yoonalimsuwan/SUPER-DNS-ONE

```
