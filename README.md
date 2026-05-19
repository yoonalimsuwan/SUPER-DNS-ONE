``
# SUPER DNS ONE

**SOC‑Controlled Direct Numerical Simulation for Peaceful Applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

SUPER DNS ONE is a fully differentiable, three‑dimensional Direct Numerical
Simulation (DNS) engine that solves the compressible Navier–Stokes equations
on structured grids.  It is intended exclusively for **peaceful, civilian
purposes**:

- **Medical research** – haemodynamics, respiratory aerosol transport,
  drug delivery through micro‑fluidic devices.
- **Weather and climate prediction** – high‑fidelity atmospheric boundary
  layer simulations, cloud microphysics, pollutant dispersion.
- **Civil aviation** – aerodynamic analysis of commercial aircraft, noise
  reduction studies, air‑traffic wake turbulence.

**This software is not designed, tested, or authorised for use in weapons,
military systems, or any form of armed conflict.**

---

## Overview

Instead of classical turbulence models (RANS, LES), SUPER DNS ONE resolves
all relevant flow scales directly on a three‑dimensional Cartesian grid.  The
solver is augmented with a unique set of mathematical tools that originate
from complex‑systems physics:

- **Self‑Organised Criticality (SOC)** – adaptive eddy viscosity that reacts
  to local stress, mimicking the energy cascade without empirical constants.
- **Semantic‑State Contraction (SSC)** – a low‑pass filter that extracts
  physical signals from noise and guides the flow towards target states.
- **Renormalisation Group (RG)** – optional coarse‑graining that accelerates
  convergence while preserving large‑scale dynamics.
- **Itô stochastic backscatter** – injects sub‑grid energy in a physically
  consistent manner, replacing heuristic sub‑grid models.
- **Batalin–Vilkovisky (BV) consistency** – monitors mass, momentum, and
  energy conservation during the simulation.

Advanced compressible‑flow capabilities include:

- Riemann‑solver‑based inviscid fluxes: **AUSM+** and **HLLC** (with optional
  MUSCL reconstruction for second‑order spatial accuracy).
- **Shock‑capturing** via SSC‑guided artificial viscosity.
- Hypersonic regime (Mach > 20) and Reynolds numbers up to machine precision.
- Trainable SOC parameters (Optuna / differential evolution) that can be
  calibrated against reference data.

The solver is written in pure PyTorch.  It runs on **CPUs (3 GB RAM)**,
**Google Colab T4**, **Apple Silicon (MPS)**, **Huawei Ascend NPU**, and
scales to multi‑GPU clusters and supercomputers without code changes.

---

## Features

- 3D compressible Navier–Stokes on structured grids
- AUSM+ and HLLC Riemann solvers (optional MUSCL reconstruction)
- SOC‑driven adaptive eddy viscosity
- SSC‑based flow control and signal denoising
- Renormalisation‑Group (RG) coarse‑graining / refinement
- Itô stochastic backscatter (sub‑grid scale model)
- Trainable SOC parameters (Optuna or differential evolution)
- Built‑in validation suites: Taylor–Green vortex, Kolmogorov –5/3 spectrum,
  grid‑convergence order, conservation checks, shock tube, hypersonic flow
- Signal/noise separation module for sensor data
- Fully differentiable (PyTorch autograd)
- Vendor‑neutral: CPU, GPU, Ascend NPU, Apple MPS

---

## Installation

```bash
git clone https://github.com/yoonalimsuwan/SUPER-DNS-ONE.git
cd super-dns-one

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install core dependencies
pip install torch numpy scipy matplotlib

# Optional: install Optuna for hyperparameter tuning
pip install optuna
```

---

Quick Start

```bash
# Run a Taylor–Green vortex benchmark with AUSM+
python super_dns_one.py --benchmark --flux ausm --steps 200

# Simulate a shock tube with HLLC
python super_dns_one.py --case shock_tube --flux hllc --nx 128 --steps 1000

# Hypersonic flow (Mach 20) with grid 128³
python super_dns_one.py --case hypersonic --Mach 20.0 --nx 128 --ny 128 --nz 128 --Re 1e6 --steps 500

# Train SOC parameters to match a target energy
python super_dns_one.py --train --target_energy 0.5 --tune_method optuna

# Denoise a synthetic sensor signal with SSC
python super_dns_one.py --denoise
```

---

Validation & Benchmarks

The solver includes built‑in validation routines:

· Taylor–Green vortex – energy decay rate compared with analytical
  solution.
· Kolmogorov spectrum – slope of E(k) in the inertial range
  (expected –5/3).
· Grid convergence – order of accuracy computed by Richardson
  extrapolation.
· Conservation laws – maximum divergence of mass flux and total
  kinetic energy.
· Shock tube – Sod’s problem with comparison to exact Riemann solution.
· Hypersonic flow – stability at Mach > 20, Re > 1e6.

These tests can be executed with --benchmark.

---

Architecture & Vendor Neutrality

SUPER DNS ONE is built entirely on PyTorch’s tensor operations.  No
CUDA‑specific kernels are used.  This design guarantees that the solver
runs on:

· CPU (x86‑64, ARM) – minimum 3 GB RAM
· NVIDIA GPU (CUDA)
· Apple Silicon (MPS backend)
· Huawei Ascend (torch_npu backend)
· Multi‑GPU clusters (DataParallel / DistributedDataParallel ready)
· Supercomputers (any platform with a PyTorch distribution)

---

Intended Use & Ethical Notice

SUPER DNS ONE is developed for peaceful scientific and engineering
applications only, including but not limited to:

· Medical fluid dynamics – blood flow, respiratory aerosols,
  micro‑fluidic drug delivery.
· Weather and climate modelling – atmospheric boundary layers,
  pollutant transport, cloud dynamics.
· Civil aviation – commercial aircraft aerodynamics, wake turbulence,
  noise prediction.

This software must not be used for the design, testing, or operation of
weapons, military vehicles, or any form of armed conflict.  The author
disclaims all liability for any such misuse.

### AI Integration – Toward O(1) Complexity

Because SUPER DNS ONE is written entirely in PyTorch and every operation—the
SOC kernel, SSC filter, RG smoothing, Itô forcing, and the Riemann solvers—is
**fully differentiable**, the solver can be directly embedded in a deep-learning
workflow.  This opens the door to training AI surrogate models that reproduce
the physics of the DNS at a tiny fraction of the cost.

**How the training works**

1. **Generate training data** – Run the SOC‑controlled DNS on a variety of
   initial conditions, Mach numbers, and Reynolds numbers.  At each time step
   (or every few steps), store the flow fields together with the SOC stress
   and the SSC contraction signal.
2. **Build a neural surrogate** – Train a 3D convolutional or
   SE(3)‑equivariant graph network to predict the next time step
   *directly* from the current fields, bypassing the iterative solution of
   the Navier–Stokes equations.
3. **Physics‑informed loss** – Because the solver is differentiable, the
   surrogate can also be trained using a **physics‑informed loss** that
   evaluates the residual of the compressible Navier–Stokes equations,
   guaranteeing that the predicted fields respect conservation laws.
4. **Inference** – Once trained, the neural network predicts one time step
   (or even a full turbulent trajectory) in a **single forward pass**,
   independently of the grid size.  The computational cost becomes
   **O(1)** with respect to the number of degrees of freedom, enabling
   real‑time medical flow visualisation, rapid weather ensemble forecasts,
   and interactive aerodynamic design.

The differentiability of SUPER DNS ONE ensures that gradients can flow
seamlessly from the physics loss into the surrogate network, creating a
self‑improving loop that continuously narrows the gap between classical
DNS and AI‑driven prediction.

---

Citing SUPER DNS ONE

If you use this software in your research, please cite:

```
Yoon A Limsuwan. "SUPER DNS ONE: SOC‑Controlled Direct Numerical Simulation."
Zenodo, 2026. DOI: 10.5281/zenodo.XXXXXXX
```

---

License

This project is licensed under the MIT License – see LICENSE for
details.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/yoonalimsuwan/SUPER-DNS-ONE


```
