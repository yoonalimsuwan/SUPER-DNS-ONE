# README_PLUS — ONE Ecosystem · Physics Cluster (v3.0.0)

> **Developer** : Yoon A Limsuwan / MSPS NETWORK  
> **ORCID** : 0009-0008-2374-0788  
> **GitHub** : yoonalimsuwan  
> **License** : MIT  
> **Year** : 2026

---

## Overview

The Physics Cluster consists of **four core files** that work together as a fully differentiable, multi-scale simulation system spanning from the atomic scale (Å) to compressible turbulence (cm–m). The theoretical foundation is the **Structural Calculus 4-Paper Framework** by Yoon A Limsuwan:

| Paper | Short Name | Core Contribution |
|-------|-----------|-------------------|
| Paper 1 | Structural Operators | Regime-Dependent Analytical Framework |
| Paper 2 | BV Jump Measures | Self-Evolving Interfaces |
| Paper 3 | Structural Itô Calculus | Multiplicative Noise & Drift Correction |
| Paper 4 | CSOC / SSC | Controlled Self-Organised Criticality |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    one_core_v3.py  (Shared Foundation)          │
│                                                                 │
│  SemanticStateContraction  CSOCBase  InterfaceDetectorBase      │
│  StructuralItoBase         get_device  ONE_VERSION              │
│  ──────────────────────────────────────────────────────────     │
│  LangevinFHBridge          LangevinDNSBridge  (Bridge Protocol) │
└────────────────────┬────────────────────┬───────────────────────┘
                     │ imports             │ imports
          ┌──────────▼──────────┐          │
          │ structural_langevin │          │
          │      _v3.py         │          │
          │                     │          │
          │  InterfaceDetector  │          │
          │  CSOCThermostat     │          │
          │  StructuralItoNoise │          │
          │  AdvancedStructural │          │
          │  Langevin (BAOAB)   │          │
          └──────┬────────┬─────┘          │
                 │        │                │
      LangevinFH │        │ LangevinDNS    │
         Bridge  │        │ Bridge         │
                 ▼        ▼                ▼
   ┌─────────────────┐  ┌──────────────────────────┐
   │ structuralfluc- │  │   super_dns_one_v6.py    │
   │ tuatinghydro_   │  │                          │
   │     v6.py       │  │  SOCController           │
   │                 │  │  CompressibleSolver      │
   │  CFDInterface-  │  │  AUSMPlusFlux / HLLCFlux │
   │  Detector       │  │  DiffRGRefiner           │
   │  CSOCAdaptive-  │  │  RealGasEOS              │
   │  Viscosity      │  │  ImmersedBoundary        │
   │  LLStochastic-  │  │  SOCTrainer              │
   │  Stress         │  └──────────────────────────┘
   │  Structural-    │
   │  Fluctuating-   │
   │  Hydro          │
   └─────────────────┘
```

**Scale hierarchy:**

```
Molecular (Å, ps)          Continuum-FH (μm–mm)     Compressible DNS (cm–m)
structural_langevin_v3  →  structuralfluctuating  →  super_dns_one_v6
                           hydro_v6
```

---

## 1. `one_core_v3.py` — Shared Foundation

### Role
This file is the **Single Source of Truth** for the entire Physics Cluster. No class defined here may be redefined in any other solver file. All solvers must import exclusively from this module.

### Key Classes and Components

#### `get_device(preferred)`
Automatically selects the best available hardware backend in priority order: CUDA → MPS (Apple) → NPU (Ascend) → CPU.

#### `SemanticStateContraction` (SSC) — Paper 4
EMA low-pass filter for structural stress σ:

```
σ_filtered[t] = σ_filtered[t-1] + ε · (σ_raw[t] − σ_filtered[t-1])
```

- Uses a boolean `_initialized` buffer instead of a zero-check, so the filter handles a true-zero first stress value correctly.
- Provides a `reset()` method for use between independent trajectories.

#### `CSOCBase` — Paper 4 (Abstract)
Abstract base class for all Adaptive Parameter Modules (Thermostat, Viscosity, SOCController). Provides the shared SSC filter, `_normalised_deviation()`, and `_smooth_boost()` logic.

#### `InterfaceDetectorBase` (Abstract)
Enforces a common `forward()` contract: all subclasses must return a fully differentiable tensor ∈ [0, 1].

#### `StructuralItoBase` — Papers 2 & 3 (Abstract)
Abstract base for Itô drift correction ½ G(x) ∇_x G(x), applicable at both per-atom (MD) and per-cell (CFD) resolution.

### Bridge Protocol (Cross-Solver Communication)

| Bridge Class | Direction | Payload |
|---|---|---|
| `LangevinFHBridge` | Langevin → FH Solver | `_ext_sigma` (grid), `_ext_mask` (interface) |
| `LangevinDNSBridge` | Langevin → DNS Solver | `_ext_sigma` (scalar) |

```python
# Bridge usage example
bridge_fh  = LangevinFHBridge(langevin_integrator, fh_solver, bandwidth=1.0)
bridge_dns = LangevinDNSBridge(langevin_integrator, dns_solver)

for step in range(num_steps):
    coords, velocities = langevin_integrator.full_step(...)
    bridge_fh.sync(coords, velocities)   # inject σ + mask → FH grid
    bridge_dns.sync(coords)              # inject σ → DNS controller
```

---

## 2. `structural_langevin_v3.py` — Molecular Dynamics Scale

### Role
Atomic-scale Langevin integrator using the BAOAB splitting scheme, implementing the full Structural Calculus framework across all four papers.

### Key Classes

#### `InterfaceDetector` ← `InterfaceDetectorBase`
Computes a per-atom soft interface mask ∈ [0, 1]:
- Counts neighbours within `r_cut` (Å) using soft sigmoid weighting.
- Uses weighted variance of pairwise distances as the interface score.
- Fully differentiable with respect to atomic coordinates via `autograd`.

#### `CSOCThermostat` ← `CSOCBase` — Paper 4
Adaptively modulates temperature and friction coefficient in real time:

```
T_eff = base_temp × (1 + boost · sigmoid(dev))
γ_eff = base_friction × (1 + friction_boost · sigmoid(dev))
```

where `dev = (σ_filtered − σ_target) / σ_target`.

#### `StructuralItoNoise` ← `StructuralItoBase` — Paper 3
Multiplicative noise field: G(x) = 1 + amp × mask(x).  
Computes the Itô drift correction ½ G(x) ∇_x G(x) via `torch.autograd.grad`.

#### `AdvancedStructuralLangevin` — Core Integrator

BAOAB splitting algorithm:

```
B  : v ← v + (F/m) · (dt/2)        [half momentum kick: bulk + jump forces]
A  : x ← x + v · (dt/2)            [half position drift]
O  : v ← v·e^{-γdt} + η√(...)      [Ornstein-Uhlenbeck: multiplicative noise]
A  : x ← x + v · (dt/2)            [half position drift]
B  : v ← v + (F/m) · (dt/2)        [half momentum kick]
```

```python
integrator = AdvancedStructuralLangevin(
    mass=1.0, dt=0.002, base_temp=300.0, base_friction=1.0
)

for step in range(N):
    force = -torch.autograd.grad(energy, coords)[0]
    interface_mask = integrator.interface_detector(coords)

    x_new, v_tilde, T, sigma = integrator.baoa_step(
        coords, velocities, force, jumps, interface_mask
    )
    velocities = integrator.final_b_step(v_tilde, new_force, jumps, interface_mask)
    coords = x_new.detach().requires_grad_(True)
```

### Factory Functions

```python
bridge = make_fh_bridge(integrator, fh_solver, bandwidth=1.0)
bridge = make_dns_bridge(integrator, dns_solver)
```

---

## 3. `structuralfluctuatinghydro_v6.py` — Fluctuating Hydrodynamics Scale

### Role
Continuum CFD solver for the Landau–Lifshitz Navier–Stokes equations on a 3D staggered MAC grid. Receives structural stress signals from the Langevin integrator via `LangevinFHBridge`.

### Governing Equations

```
∂ρ/∂t  + ∇·(ρu)     = 0                               (continuity)
∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·(νρ(∇u+∇uᵀ)) + ∇·S̃   (momentum)
```

where S̃ is the Landau–Lifshitz stochastic stress tensor (3×3 symmetric).

### Grid Convention (Staggered MAC)

```
ρ, p  → cell centres   (Nx,   Ny,   Nz  )
ux    → x-face centres (Nx+1, Ny,   Nz  )
uy    → y-face centres (Nx,   Ny+1, Nz  )
uz    → z-face centres (Nx,   Ny,   Nz+1)
```

### Key Classes

#### `CFDInterfaceDetector` ← `InterfaceDetectorBase`
Detects interfaces on grid scalar fields using differentiable gradient magnitude.

#### `CSOCAdaptiveViscosity` ← `CSOCBase` — Paper 4
Adaptively modulates viscosity and thermal diffusivity based on the filtered stress signal:

```python
csoc = CSOCAdaptiveViscosity(
    base_viscosity=1e-3, base_diffusivity=1e-5,
    sigma_target=1.0, viscosity_boost=3.0
)
nu, kappa = csoc(sigma_raw)
```

#### `LLStochasticStress` — Paper 3
Constructs the Landau–Lifshitz stochastic stress tensor:
- Noise amplitude set by the Fluctuation–Dissipation theorem: ∝ √(2 k_B T ν / dV dt).
- Noise is amplified at interfaces via G(x) = 1 + amp · mask.
- `softplus` floor replaces `.clamp()` everywhere for full differentiability (DIFF-FIX 5).

#### `StructuralFluctuatingHydro` — Core Solver

```python
cfg    = SolverConfig(Nx=32, Ny=32, Nz=32, dt=1e-4)
solver = StructuralFluctuatingHydro(cfg)
rho, ux, uy, uz, p = solver.initialize_taylor_green()

for _ in range(100):
    rho, ux, uy, uz, p, diag = solver.step(rho, ux, uy, uz, p)
```

### Advection Schemes (configurable)

| Scheme | Order | Notes |
|--------|-------|-------|
| `"upwind"` | 1st | Stable, diffusive (default) |
| `"tvd"` | 2nd | TVD limiters: minmod / van_leer / superbee (differentiable) |
| `"weno5"` | 5th | Jiang-Shu WENO-5, LF split (smooth logsumexp wave speed) |
| `"semi_lagrangian"` | — | Unconditionally stable trilinear GPU interpolation |

### Poisson Solver
Spectral FFT-based pressure projection using `rfftn` — O(N³ log N), exact solution.

### External Coupling Buffers
When `LangevinFHBridge.sync()` is called, it writes:
```
solver._ext_sigma → (Nx, Ny, Nz)  : uniform projection of MD SSC stress
solver._ext_mask  → (Nx, Ny, Nz)  : mean atomic interface score
```
Inside `step()`, these are blended with the local CSOC signal:
`σ_eff = 0.5 * (σ_local + _ext_sigma.mean())`

---

## 4. `super_dns_one_v6.py` — Compressible DNS / LES Scale

### Role
The largest solver in the cluster: a fully differentiable 3D compressible DNS/LES solver. Receives structural stress from the Langevin integrator via `LangevinDNSBridge`.

### Key Classes

#### `SOCController` ← `CSOCBase` — Paper 4
CSOC-adaptive eddy viscosity controller:
- `CSOCKernel`: 5-parameter learnable kernel — Cs · r^{-α} · e^{-r/λ}.
- Accepts `ext_sigma` from `LangevinDNSBridge` and blends it with local CSOC stress.
- All `.clamp()` calls replaced by `softplus` (DIFF-FIX 5).

#### Riemann Solvers (both fully differentiable)

| Class | Algorithm | Differentiability fix |
|-------|-----------|----------------------|
| `AUSMPlusFlux` | AUSM+ | Smooth M±/P± via tanh blending (DIFF-FIX 3) |
| `HLLCFlux` | HLLC | Branch-free soft gating via sigmoid (DIFF-FIX 4) |

#### Boundary Conditions (7 types)

```
PeriodicBC              SubsonicOutflowBC        NoSlipIsothermalWallBC
SupersonicInflowBC      WernerWengleWallModelBC  FarFieldBC
MovingWallBC
```

#### `SemiLagrangianAdvection3D`
Unconditionally stable advection using `grid_sample` GPU trilinear interpolation.

#### `DiffRGRefiner` — Paper 4 (RG)
Renormalization Group refinement step. DIFF-FIX 6 replaces the Python-branch rescaling with a smooth softplus-guarded division.

#### `RealGasEOS`
van der Waals equation of state for real-gas thermodynamic effects.

#### `ImmersedBoundary`
Immersed Boundary Method (IBM) for simulations over complex geometry.

#### `CompressibleSolver` — Core Solver

```python
cfg    = CFDConfig(Nx=64, Ny=64, Nz=64, dt=1e-5, mach=0.5)
solver = CompressibleSolver(cfg)
state  = solver.initialize()

for _ in range(steps):
    state, diag = solver.step(state)
```

#### `SOCTrainer`
Training loop for optimising CSOC kernel parameters via Optuna / gradient descent.

---

## Multi-Scale Coupling

### Full-Coupled Workflow

```python
from one_core_v3 import LangevinFHBridge, LangevinDNSBridge, get_device

device = get_device("cuda")

# 1. Initialise all solvers
md_integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=300.0).to(device)
fh_solver     = StructuralFluctuatingHydro(SolverConfig(Nx=32, Ny=32, Nz=32)).to(device)
dns_solver    = CompressibleSolver(CFDConfig(Nx=64, Ny=64, Nz=64)).to(device)

# 2. Create bridges (defined in one_core_v3)
bridge_fh  = LangevinFHBridge(md_integrator, fh_solver, bandwidth=1.0)
bridge_dns = LangevinDNSBridge(md_integrator, dns_solver)

# 3. Main simulation loop
coords     = torch.randn(N_atoms, 3, device=device, requires_grad=True)
velocities = torch.zeros(N_atoms, 3, device=device)
rho, ux, uy, uz, p = fh_solver.initialize_taylor_green()
dns_state  = dns_solver.initialize()

for step in range(total_steps):
    # --- Molecular Dynamics step ---
    force = -torch.autograd.grad(potential(coords), coords)[0]
    coords, velocities = md_integrator.full_step(coords, velocities, force, ...)

    # --- Bridge: propagate stress upward ---
    bridge_fh.sync(coords, velocities)    # → fh_solver._ext_sigma / _ext_mask
    bridge_dns.sync(coords)               # → dns_solver._ext_sigma

    # --- FH step (receives MD stress) ---
    rho, ux, uy, uz, p, fh_diag = fh_solver.step(rho, ux, uy, uz, p)

    # --- DNS step (receives MD stress) ---
    dns_state, dns_diag = dns_solver.step(dns_state)
```

### Data Flow Diagram

```
AdvancedStructuralLangevin
    │
    ├─ thermostat.ssc.prev_sigma  ─────────────────────────────┐
    │  (scalar SSC stress buffer)                               │
    │                                                           │
    ├─── LangevinFHBridge.sync() ──────────────────▶  StructuralFluctuatingHydro
    │        expand σ → (Nx, Ny, Nz)                   _ext_sigma blended in step()
    │        mean(mask_atomic) → _ext_mask              _ext_mask modulates S̃
    │
    └─── LangevinDNSBridge.sync() ─────────────────▶  CompressibleSolver
             scalar σ                                   SOCController.nu_t(ext_sigma=σ)
                                                        blend: ssc_in = 0.5*(local + σ)
```

---

## Differentiability Matrix

| Component | Autograd Safe | Mechanism |
|-----------|:---:|-----------|
| `SemanticStateContraction` | ✅ | EMA + buffer detach |
| `InterfaceDetector` (MD) | ✅ | sigmoid, sqrt+ε |
| `CFDInterfaceDetector` | ✅ | gradient magnitude |
| `CSOCThermostat` | ✅ | sigmoid boost |
| `CSOCAdaptiveViscosity` | ✅ | sigmoid boost |
| `SOCController` (DNS) | ✅ | softplus throughout (DIFF-FIX 5) |
| `StructuralItoNoise` | ✅ | autograd.grad + detached return |
| `LLStochasticStress` | ✅ | softplus floor (DIFF-FIX 5 FH) |
| TVD limiters (FH) | ✅ | softplus / tanh gating (DIFF-FIX 1 FH) |
| WENO-5 wave speed | ✅ | logsumexp smooth max (DIFF-FIX 2) |
| AUSM+ flux | ✅ | tanh smooth M± (DIFF-FIX 3) |
| HLLC flux | ✅ | sigmoid branch-free (DIFF-FIX 4) |
| `CompressibleSolver.step()` | ✅ | softplus floor on ρ, p (DIFF-FIX 7) |

---

## Critical Bug Fixes (v3.0.0 / v6.0)

| Bug | Original Problem | Fix Applied |
|-----|-----------------|-------------|
| Bug 1 | `ssc._prev` — incorrect attribute name | Renamed to `ssc.prev_sigma` (correct registered buffer) |
| Bug 2 | `SOCController` did not inherit `CSOCBase` | Added `super().__init__(sigma_target, epsilon_fp, boost_factor)` |
| Bug 3 | No communication interface between Langevin ↔ FH/DNS | Added `LangevinFHBridge` and `LangevinDNSBridge` in `one_core_v3.py` |
| SSC Init | `prev == 0.0` check failed when true stress was zero | Replaced with `_initialized` boolean buffer |
| Clamp | `.clamp()` zeroed gradients at boundary | Replaced with `softplus` floor at every occurrence |
| Hard branch | `torch.where()` created zero-gradient regions | Replaced with tanh / sigmoid smooth gating |

---

## Dependencies

```
torch >= 2.0   (core — autograd, rfft, grid_sample)
numpy          (diagnostics, I/O)
scipy          (spectral analysis in tests)
matplotlib     (visualisation)
```

Optional:
```
optuna         (SOCTrainer hyperparameter search)
CoolProp       (real-gas thermodynamics)
PyWavelets     (wavelet diagnostics)
```

---

## Quick-Start (Minimal Example)

```python
import torch
from one_core_v3                   import get_device, LangevinFHBridge
from structural_langevin_v3        import AdvancedStructuralLangevin
from structuralfluctuatinghydro_v6 import StructuralFluctuatingHydro, SolverConfig

device = get_device("cuda")

# Molecular Dynamics solver
md = AdvancedStructuralLangevin(dt=0.002, base_temp=300.0).to(device)

# Fluctuating Hydro solver
fh = StructuralFluctuatingHydro(SolverConfig(Nx=16, Ny=16, Nz=16, dt=1e-4)).to(device)
rho, ux, uy, uz, p = fh.initialize_taylor_green()

# Cross-scale bridge
bridge = LangevinFHBridge(md, fh)

# Simulation state
coords     = torch.randn(32, 3, device=device, requires_grad=True)
velocities = torch.zeros(32, 3, device=device)

for step in range(100):
    with torch.no_grad():
        bridge.sync(coords, velocities)
    rho, ux, uy, uz, p, diag = fh.step(rho, ux, uy, uz, p)
    if step % 10 == 0:
        print(f"step {step:4d} | ρ_mean={rho.mean():.4f} | CFL={diag['cfl']:.3f}")
```

---

## File Summary

| File | Physical Scale | Core Classes | Size |
|------|---------------|-------------|------|
| `one_core_v3.py` | Foundation (shared) | SSC, CSOCBase, Bridges | 13 KB |
| `structural_langevin_v3.py` | Molecular (Å, ps) | AdvancedStructuralLangevin, CSOCThermostat | 25 KB |
| `structuralfluctuatinghydro_v6.py` | Continuum FH (μm–mm) | StructuralFluctuatingHydro, LLStochasticStress | 64 KB |
| `super_dns_one_v6.py` | Turbulence DNS (cm–m) | CompressibleSolver, SOCController, AUSM+/HLLC | 123 KB |

---

*README_PLUS.md — ONE Ecosystem Physics Cluster v3.0.0*  
*Yoon A Limsuwan / MSPS NETWORK · MIT License · 2026*
