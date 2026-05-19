`
# SUPER DNS ONE

**SOC‑Controlled Direct Numerical Simulation for Peaceful Applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20283663-blue)](https://doi.org/10.5281/zenodo.20283663) <!-- replace with actual DOI when available -->

SUPER DNS ONE is a fully differentiable, three‑dimensional Direct Numerical
Simulation (DNS) engine that solves the compressible Navier–Stokes equations
on structured Cartesian grids.  It is intended exclusively for **peaceful,
civilian purposes**:

- **Medical research** – haemodynamics, respiratory aerosol transport,
  drug delivery through micro‑fluidic devices.
- **Weather and climate prediction** – high‑fidelity atmospheric boundary
  layer simulations, pollutant dispersion, cloud microphysics.
- **Civil aviation** – aerodynamic analysis of commercial aircraft, noise
  reduction studies, air‑traffic wake turbulence.

**This software is not designed, tested, or authorised for use in weapons,
military systems, or any form of armed conflict.**

---

## Overview

SUPER DNS ONE resolves the compressible flow equations directly on a
three‑dimensional Cartesian grid using a finite‑volume conservative
discretisation.  It is augmented with a unique set of mathematical tools
drawn from complex‑systems physics:

- **Self‑Organised Criticality (SOC)** – an adaptive eddy viscosity that
  reacts to local strain rates and stress accumulation, mimicking the
  turbulent energy cascade without fixed empirical constants.  The SOC
  kernel has **five trainable parameters**.
- **Semantic‑State Contraction (SSC)** – a signal‑noise separator that
  extracts physical flow structures from noisy sensor data.
- **Renormalisation Group (RG)** – optional conservative spectral
  truncation for coarse‑graining, accelerating long‑time simulations
  while preserving large‑scale dynamics.
- **Itô stochastic backscatter** – physically motivated sub‑grid energy
  injection that replaces traditional LES models.
- **Batalin–Vilkovisky (BV) consistency diagnostics** – monitors
  mass, momentum, and energy conservation in real time.

Compressible‑flow capabilities include:

- Riemann‑solver‑based inviscid fluxes: **AUSM+** and **HLLC** with
  optional **MUSCL reconstruction** (second‑order spatial accuracy).
- **Compressibility correction** for hypersonic turbulent viscosity.
- **Shock capturing** via dilatation‑based artificial viscosity.
- Reynolds numbers up to machine precision.
- Mixed precision (AMP) support on CUDA.

The solver is written in pure PyTorch and runs on **CPUs**, **NVIDIA GPUs**,
**Apple Silicon (MPS)**, **Huawei Ascend NPU**, and multi‑GPU clusters
without code changes.

---

## Features

- 3D compressible Navier–Stokes on structured grids (finite‑volume)
- AUSM+ and HLLC Riemann solvers (optional MUSCL reconstruction)
- SOC‑driven adaptive eddy viscosity (trainable 5‑parameter kernel)
- SSC‑based signal denoising and flow control
- Renormalisation‑Group (RG) spectral truncation (optional)
- Itô stochastic backscatter for LES
- BV consistency monitors (kinetic energy, divergence, stress balance)
- Trainable SOC parameters via **Differential Evolution** or **Optuna**
- Built‑in validation: Taylor–Green vortex, Kolmogorov –5/3 spectrum,
  grid‑convergence order, conservation checks
- Hypersonic boundary‑layer initialisation (`hypersonic_bnd`)
- Signal/noise separation module for sensor data
- Fully differentiable (PyTorch autograd) – ready for physics‑informed
  machine learning and surrogate model training
- Vendor‑neutral: CPU, CUDA, MPS, Ascend NPU
- MIT license – unrestricted peaceful use

---

## Installation

```bash
git clone https://github.com/yoonalimsuwan/SUPER-DNS-ONE.git
cd SUPER-DNS-ONE

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Core dependencies
pip install torch numpy scipy matplotlib

# Optional: Optuna for hyperparameter tuning
pip install optuna
```

---

Quick Start

The main script is super_dns_one.py.

```bash
# Taylor–Green vortex (default) with AUSM+
python super_dns_one.py --steps 200 --flux ausm

# Hypersonic boundary layer (Mach 20), 128³ grid
python super_dns_one.py --case hypersonic_bnd --Mach 20.0 --nx 128 --ny 128 --nz 128 --Re 1e6 --steps 500

# Train SOC parameters to match a target kinetic energy
python super_dns_one.py --train --target_energy 0.5 --tune_method de

# Denoise a synthetic velocity field with SSC
python super_dns_one.py --denoise
```

For a full list of command‑line arguments, use:

```bash
python super_dns_one.py --help
```

---

Validation & Benchmarks

The solver includes built‑in routines that can be activated with
--benchmark (or by calling them programmatically):

Test Description
Taylor–Green vortex Energy decay rate compared with analytical solution
Kolmogorov spectrum Slope of E(k) in the inertial range (expected –5/3)
Grid convergence Order of accuracy via Richardson extrapolation
BV conservation checks Maximum divergence of velocity, stress consistency, and kinetic energy history

Additionally, the hypersonic_bnd case provides a compressible boundary‑layer
initial condition suitable for testing high‑speed wall‑bounded flows.

Note: A dedicated shock‑tube test and automated hypersonic validation
suite are planned for a future release.

---

Architecture & Vendor Neutrality

SUPER DNS ONE is built exclusively on PyTorch tensor operations.  No
CUDA‑specific kernels are used.  This design guarantees portability across:

· CPU (x86‑64, ARM) – minimum 4 GB RAM for grid sizes up to 64³
· NVIDIA GPU (CUDA) – including Google Colab T4
· Apple Silicon (MPS backend)
· Huawei Ascend (torch_npu backend)
· Multi‑GPU clusters – ready for DataParallel / DistributedDataParallel

---

Roadmap & Planned Features

· Dedicated shock‑tube validation case
· AI surrogate model training – because the solver is fully
  differentiable, it can directly generate training data and provide
  physics‑informed gradients for neural network surrogates (O(1) inference).
· Real‑gas equations of state for hypersonic flows
· Immersed boundary method for complex medical geometries
· Distributed training wrapper for DDP

---

Citing SUPER DNS ONE

If you use this software in your research, please cite:

```
Yoon A Limsuwan. "SUPER DNS ONE: SOC‑Controlled Direct Numerical Simulation."
Zenodo, 2026. DOI: 10.5281/zenodo.20283663   (if available)
```

Otherwise, cite the GitHub repository:

```
Yoon A Limsuwan. (2026). SUPER DNS ONE (Version 1.0.0) [Computer software].
https://github.com/yoonalimsuwan/SUPER-DNS-ONE
```

---

License

This project is licensed under the MIT License – see the LICENSE file
for details.

---

Contact

Yoon A Limsuwan – GitHub
Project repository: https://github.com/yoonalimsuwan/SUPER-DNS-ONE

```

---
