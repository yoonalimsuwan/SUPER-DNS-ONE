# Contributing to Super DNS ONE

Thank you for your interest in contributing to the **Super DNS ONE** solver and the broader **ONE Ecosystem** (Structural Calculus / CSOC framework).

This document explains how the codebase is structured, which parts are intentionally left simple, and where community contributions are most welcome.

---

## Philosophy

The core novelty of this project is the **Structural Itô Calculus**, **CSOC adaptive viscosity**, and **Landau–Lifshitz stochastic stress** framework.  These are the parts that are new to science.

Everything else — advection schemes, Riemann solvers, boundary conditions — is deliberately kept at a working baseline so that the physics is easy to read and verify.  **You are encouraged to replace or extend those parts.**

---

## Where to Contribute

### 1. Advection Schemes
**File:** `structuralfluctuatinghydro_v5.py` / `super_dns_one_v5.py`  
**Current baseline:** 1st-order upwind (stable, simple)  
**What the community can add:**
- 2nd-order TVD (minmod, van Leer, superbee limiters)
- WENO-7 for high-accuracy shock capturing
- `torch.nn.functional.grid_sample` for GPU-optimised semi-Lagrangian advection

To add a new scheme, find the advection block marked:
```python
# ── Density advection (1st-order upwind) ──
```
and replace or extend it.  Keep the interface: input `(rho, ux, uy, dx, dy, dt)`, output `rho_new`.

---

### 2. Riemann Solvers
**File:** `super_dns_one_v5.py`  
**Current options:** AUSM+, HLLC  
**What the community can add:**
- Roe solver with entropy fix
- AUSM+-up (low-Mach preconditioning)
- Rotated-hybrid HLLC-HLL for carbuncle suppression

New solvers should subclass `RiemannSolverBase` and implement `compute_face_flux(...)`.

---

### 3. OpenFOAM / External CFD Coupling
If you want to use this solver as a **stochastic stress source term** inside OpenFOAM or another C++ CFD engine:

- The recommended approach is **file-based or shared-memory coupling** (not direct linking), because PyTorch and OpenFOAM have incompatible memory models.
- A minimal coupling loop would: (1) read cell-centred fields from OpenFOAM via HDF5 or VTK, (2) run one CSOC/LL stress step here, (3) write the stochastic forcing back.
- This is non-trivial engineering work.  If you build it, please open a pull request with a `examples/openfoam_coupling/` directory.

---

### 4. Stretched / Body-Fitted Grids
**Current support:** uniform Cartesian grids + `CFDConfig.with_stretched_grid()` for 1-D tanh stretching  
**What the community can add:**
- Curvilinear coordinate transforms (metrics tensor approach)
- Unstructured mesh support via graph-based stencils
- Interfacing with mesh generators (Gmsh, Pointwise)

---

### 5. Boundary Conditions
New BC classes should subclass `BoundaryCondition` and implement both `apply(...)` and `ghost_cells(...)`.  See `NoSlipIsothermalWallBC` as the reference implementation.

---

### 6. Hardware Backends
The solver targets CPU, CUDA, MPS (Apple), and Huawei Ascend NPU.  If you have access to hardware that needs a specific kernel, contributions to `get_device()` and any device-specific workarounds are welcome.

---

## What NOT to Change

Please do **not** modify the following without opening a discussion first, as they implement the published theoretical framework:

- `LLStochasticStress` — Landau–Lifshitz stochastic stress tensor (Paper 3)
- `CSOCAdaptiveViscosity` / `SOCController` — CSOC thermostat
- `SemanticStateContraction` — SSC EMA filter
- `_structural_ito_correction` — Structural Itô drift term (Theorem 4.1)
- `DiffRGRefiner` — conservative RG spectral truncation

Changes to these affect the scientific claims of the accompanying papers.

---

## Code Style

- Python 3.10+, PyTorch 2.0+
- Type hints on all public functions
- Docstrings for every class and public method
- No `print()` in library code — use `logging.getLogger(__name__)`
- All new numerical methods should include at least one unit test

---

## Licence

All contributions are accepted under the existing **MIT licence**.

---

*Super DNS ONE is part of the ONE Ecosystem developed by Yoon A Limsuwan.*  
*ORCID: 0009-0008-2374-0788 — GitHub: yoonalimsuwan*
