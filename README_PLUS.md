# SUPER DNS ONE Cluster — README_PLUS

**ONE Ecosystem · MSPS NETWORK**

> *"My Soul Move By Power Of Holy Spirit"*

---

## Overview

The **SUPER DNS ONE** cluster is a collection of six tightly integrated Python modules that form a multi-scale, fully differentiable scientific computing suite for structural physics simulation and AI surrogate modelling. Every module is built on the **Structural Calculus** framework — a regime-dependent mathematical formulation in which all differential operators are modulated by a spatially varying structural field σ(x).

The cluster spans four physical scales simultaneously and bridges them through a shared coupling protocol defined in `one_core_v3.py`:

```
Molecular scale  ←→  Continuum FH  ←→  Compressible DNS  ←→  Phase-Field CH
(Langevin MD)         (LLNS FH)         (Navier-Stokes)        (Cahn-Hilliard)
        ↑                  ↑                    ↑                      ↑
        └──────────────────┴────────────────────┴──────────────────────┘
                                   AI Surrogate
                              (Structural FNO 3D)
```

All six modules are:

- **PyTorch-native** — every operation is autograd-compatible; end-to-end gradients flow through the full multi-physics pipeline
- **Fully differentiable** — hard branches (`torch.where`, `.clamp`, `.max`) replaced throughout with smooth equivalents (`softplus`, `tanh`, `logsumexp`)
- **GPU-parallel** — single-GPU and multi-GPU (DDP) ready
- **MIT licensed** — free to use, modify, and distribute

---

## Repository Structure

```
SUPER DNS ONE/
├── one_core_v3.py                  # Shared foundation (import first)
├── structural_langevin_v3.py       # Molecular-scale Langevin MD
├── structuralfluctuatinghydro_v6.py # Continuum Fluctuating Hydrodynamics
├── structural_cahn_hilliard_3d.py  # Phase-field & interface dynamics
├── super_dns_one_v6.py             # Compressible DNS / LES solver
└── structural_fno_3d.py            # AI surrogate (Fourier Neural Operator)
```

---

## Module Reference

### 1. `one_core_v3.py` — ONE Core  `v3.1.0`

**Role:** Single source of truth for every shared component in the ONE Ecosystem.

All base classes, bridge protocols, and utility functions that appear in more than one solver file are defined **here and only here**. Individual solver files import from `one_core`; they must never redefine these classes locally.

#### Shared Components

| Symbol | Purpose |
|---|---|
| `ONE_VERSION` | Ecosystem-wide version string (`"3.1.0"`) |
| `get_device(preferred)` | Unified hardware-backend selector (CUDA → MPS → CPU) |
| `SemanticStateContraction` | SSC exponential moving-average low-pass filter (Paper 4) |
| `CSOCBase` | Abstract base for CSOC adaptive thermostat / viscosity controllers |
| `InterfaceDetectorBase` | Abstract base for interface detection subclasses |
| `StructuralItoBase` | Abstract base for Itô drift correction subclasses |
| `structural_biharmonic_n` | Recursive Δ_S^n operator — module-level utility |

#### Bridge Protocol

Bridges carry physical fields from one solver to another through well-defined coupling channels. All bridges are imported by downstream solver files from `one_core`:

| Bridge | Direction | Coupling channels |
|---|---|---|
| `LangevinFHBridge` | Langevin → FH | structural stress σ, interface mask |
| `LangevinDNSBridge` | Langevin → DNS | structural stress σ, interface mask |
| `CahnHilliardFHBridge` | CH → FH | effective density ρ_eff, viscosity ν_eff |
| `CahnHilliardDNSBridge` | CH → DNS | ρ_eff, ν_eff, Korteweg body force (fx, fy, fz) |

`CahnHilliardDNSBridge` supports two usage patterns:

```python
# Pattern A — two-solver coupling (pushes fields into dns_solver buffers)
bridge = CahnHilliardDNSBridge(ch_solver, dns_solver, korteweg_strength=1e-4)
bridge.sync(u_new)          # called each step

# Pattern B — standalone (no dns_solver required)
bridge = CahnHilliardDNSBridge(ch_solver, korteweg_strength=0.1)
u_new, rho_eff, nu_eff, fx, fy, fz = bridge.coupled_step(u, sigma)
```

#### Mathematical Foundation

The Structural Calculus operators used throughout the cluster:

```
grad_S u   = σ(x) · ∇u          Structural Gradient
div_S F    = ∇ · (σ(x) · F)     Structural Divergence
Δ_S u      = ∇ · (σ(x) · ∇u)   Structural Laplacian
Δ_S^n u    = Δ_S(Δ_S^(n-1) u)  Recursive Structural Bi-Laplacian
```

#### AI Development Partners

Claude (Anthropic) · GPT (OpenAI) · Gemini (Google) · DeepSeek

---

### 2. `structural_langevin_v3.py` — Structural Langevin MD  `v3.1`

**Role:** Molecular-dynamics integrator at the particle scale.

Implements the **BAOAB splitting** Langevin integrator extended with all four Structural Calculus papers. Produces structural stress fields that can be injected into the FH or DNS solvers via bridge objects.

#### Key Classes

| Class | Role |
|---|---|
| `InterfaceDetector` | Detects particle-scale interfaces; produces boolean mask and distance field |
| `CSOCThermostat` | CSOC-adaptive thermostat: adjusts T and friction γ near criticality |
| `StructuralItoNoise` | Multiplicative noise generator with Itô drift correction |
| `AdvancedStructuralLangevin` | Full BAOAB integrator — main entry point |

#### Integrated Physics

- **BAOAB Splitting** — half-B (force), full-A (drift), full-O (thermostat), half-A, half-B — guarantees correct thermodynamic sampling at second-order accuracy
- **BV Jump Measures** — concentrated stochastic increments at phase boundaries (Paper 2)
- **Structural Itô Correction** — explicit drift term that compensates for state-dependent noise (Paper 3)
- **CSOC Adaptive Control** — SemanticStateContraction EMA smooths the criticality signal; thermostat temperature and friction are adjusted dynamically (Paper 4)

#### Usage

```python
from structural_langevin_v3 import AdvancedStructuralLangevin, make_fh_bridge

integrator = AdvancedStructuralLangevin(
    mass=1.0, dt=0.002, base_temp=300.0
)

for step in range(n_steps):
    force = -torch.autograd.grad(energy(coords), coords)[0]
    interface_mask = integrator.interface_detector(coords)
    jumps = ...  # (N, 3) jump vectors at interfaces

    x_new, v_tilde, T, sigma = integrator.baoa_step(
        coords, velocities, force, jumps, interface_mask
    )
    new_force = -torch.autograd.grad(energy(x_new), x_new)[0]
    velocities = integrator.final_b_step(v_tilde, new_force, jumps, interface_mask)
    coords = x_new.detach().requires_grad_(True)

# Bridge to FH solver
bridge = make_fh_bridge(integrator, fh_solver, bandwidth=1.0)
bridge.sync(coords)     # push σ and interface mask into fh_solver
```

#### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `mass` | `1.0` | Atomic mass (amu or reduced units) |
| `dt` | `0.002` | Time step (ps) |
| `base_temp` | `300.0` | Reference temperature (K) |
| `base_friction` | `1.0` | Reference friction coefficient |
| `interface_r_cut` | `8.0` | Interface detection radius |

#### AI Development Partners

Claude (Anthropic) · GPT (OpenAI) · Gemini (Google) · DeepSeek

---

### 3. `structuralfluctuatinghydro_v6.py` — Structural Fluctuating Hydrodynamics  `v6.1`

**Role:** Continuum Fluctuating Hydrodynamics (FH) solver at the mesoscale.

Solves the **Landau–Lifshitz Navier–Stokes (LLNS)** equations on a 3-D staggered MAC grid with thermal fluctuations, CSOC-adaptive viscosity, and a spectral FFT Poisson solver. Acts as the bridge layer between Langevin MD (below) and compressible DNS (above).

#### Governing Equations

```
∂ρ/∂t  + ∇·(ρu)     = 0                     continuity
∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·(τ + S̃) + f  momentum
```

where S̃ is the Landau–Lifshitz stochastic stress tensor constructed via the Structural Itô / CSOC framework.

#### Key Classes

| Class | Role |
|---|---|
| `SolverConfig` | Full solver configuration (grid, physics, BC, advection) |
| `CFDInterfaceDetector` | Detects density-gradient-based interfaces on the continuum grid |
| `CSOCAdaptiveViscosity` | CSOC-driven ν(x,t) — boosts viscosity near critical regions |
| `LLStochasticStress` | Landau–Lifshitz 3×3 symmetric stress tensor with Itô correction |
| `StructuralFluctuatingHydro` | Main solver — time-steps the full LLNS system |

#### Numerical Scheme

| Component | Method |
|---|---|
| Time integration | Fractional-step (projection), 1st-order explicit |
| Pressure solve | Spectral FFT Poisson — O(N³ log N), exact for periodic domains |
| Advection | Configurable: `upwind` · `tvd` · `weno5` · `semi_lagrangian` |
| TVD limiters | `minmod` · `van_leer` · `superbee` |
| Viscous term | 7-point Laplacian, staggered MAC |
| Stochastic stress | Discrete Landau–Lifshitz, strength ∝ √(2 k_B T ν / dx³ dt) |

#### Grid Convention

```
Scalars  (ρ, p)  : cell centres  (Nx, Ny, Nz)
Velocity ux      : x-face centres (Nx+1, Ny,   Nz  )
Velocity uy      : y-face centres (Nx,   Ny+1, Nz  )
Velocity uz      : z-face centres (Nx,   Ny,   Nz+1)
```

#### Usage

```python
from structuralfluctuatinghydro_v6 import StructuralFluctuatingHydro, SolverConfig

cfg = SolverConfig(
    Nx=64, Ny=64, Nz=64,
    Lx=1.0, Ly=1.0, Lz=1.0,
    dt=1e-4,
    base_viscosity=1e-3,
    kb_T=4.11e-21,              # k_B × 298 K
    advection_scheme="weno5",
    enable_fluctuations=True,
)
solver = StructuralFluctuatingHydro(cfg)
solver.initialize_taylor_green(amplitude=1.0)

for step in range(n_steps):
    solver.step()
```

#### Key Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `Nx, Ny, Nz` | `32` | Grid cell counts |
| `dt` | `1e-4` | Time step (s) |
| `base_viscosity` | `1e-3` | Kinematic viscosity ν₀ (m²/s) |
| `kb_T` | `4.11e-21` | Thermal energy k_B T (J) |
| `advection_scheme` | `"upwind"` | `upwind` / `tvd` / `weno5` / `semi_lagrangian` |
| `interface_sharpness` | `4.0` | CSOC interface detection sharpness |
| `viscosity_boost` | `5.0` | CSOC viscosity amplification at interfaces |

#### AI Development Partners

Claude (Anthropic) · GPT (OpenAI) · Gemini (Google) · DeepSeek

---

### 4. `structural_cahn_hilliard_3d.py` — Structural Cahn–Hilliard 3D  `v2`

**Role:** Phase-field and interface dynamics at the mesoscale (Component #4 of SUPER DNS ONE cluster).

Implements a family of fourth- and sixth-order structural PDEs for phase separation, thin-film dynamics, and crystallisation. All solvers are fully differentiable and GPU-parallel.

#### Classes

| Class | Role |
|---|---|
| `CahnHilliardConfig` | Configuration for all CH variants |
| `StructuralCahnHilliard3D` | Standard structural Cahn–Hilliard (base class) |
| `ThinFilmStructuralCahnHilliard3D` | Degenerate mobility M(u) = softplus(u)³ + surface diffusion |
| `PhaseFieldCrystal3D` | 6th-order Phase-Field Crystal PDE (PFC) |
| `CahnHilliardDNSBridge` | Re-exported from `one_core` — Korteweg coupling to DNS |
| `make_sigma_field` | Utility to construct σ(x) fields with inclusions |

#### Governing Equations

**Standard Cahn–Hilliard:**
```
μ_R   = (u³ − u) − ε² Δ_S u        Chemical potential
∂u/∂t = Δ_S(μ_R)                    Phase evolution
```

**Thin-Film (degenerate mobility):**
```
∂u/∂t = div_S(M(u) · grad μ_R)
        [+ optional: −κ_s Δ_S(M(u) Δ_S u)]    surface diffusion
```

**Phase-Field Crystal (6th-order):**
```
μ_PFC = (r·u + u³) + u + 2Δ_S u + Δ_S² u
∂u/∂t = Δ_S(μ_PFC)
```

#### Laplacian Backends

The structural Laplacian Δ_S u = ∇·(σ(x)·∇u) has three interchangeable backends:

| Backend | Key | Notes |
|---|---|---|
| `_Conv3dLaplacian` | `"conv3d"` | Vectorised stencil via `F.conv3d`; ~4–8× faster on GPU than roll-based |
| `_FFTLaplacian` | `"fft"` | Spectral, O(N log N), exact for periodic domains; fully autograd-enabled |
| `_RollLaplacian` | `"roll"` | Classical finite-difference roll-based; v1 reference implementation |

#### Usage

```python
from structural_cahn_hilliard_3d import (
    CahnHilliardConfig, StructuralCahnHilliard3D,
    ThinFilmStructuralCahnHilliard3D, PhaseFieldCrystal3D,
    CahnHilliardDNSBridge, make_sigma_field,
)

# Standard CH
cfg = CahnHilliardConfig(
    dx=1.0, epsilon=1.5, dt=1e-5,
    laplacian="conv3d",    # or "fft" / "roll"
    scheme="imex",         # or "explicit"
    device="cuda",
)
ch = StructuralCahnHilliard3D(cfg).to("cuda")

sigma = make_sigma_field(64, 64, 64, background=1.0,
                         inclusions=[{"x0":20,"x1":44,"y0":20,"y1":44,
                                      "z0":20,"z1":44,"sigma":5.0}])
u_new = ch.step(u, sigma)

# Thin-Film
cfg_tf = CahnHilliardConfig(thin_film=True, surface_diffusion=True, kappa_s=0.01)
tf = ThinFilmStructuralCahnHilliard3D(cfg_tf).to("cuda")

# Phase-Field Crystal
cfg_pfc = CahnHilliardConfig(pfc_r=-0.5, laplacian="fft")
pfc = PhaseFieldCrystal3D(cfg_pfc).to("cuda")

# Multi-step evolution
u_final, energy_history = ch.evolve(u, sigma, n_steps=1000, log_interval=50)
```

#### Key Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `dx` | `1.0` | Isotropic grid spacing |
| `epsilon` | `1.5` | Interface-thickness parameter (Cahn number) |
| `dt` | `1e-5` | Time step |
| `mobility` | `1.0` | Isotropic mobility M |
| `scheme` | `"explicit"` | `explicit` or `imex` (implicit-explicit for stiff problems) |
| `laplacian` | `"conv3d"` | Laplacian backend |
| `thin_film` | `False` | Enable degenerate M(u) = softplus(u)³ |
| `pfc_r` | `-0.5` | PFC reduced temperature r |
| `ssc_stabilise` | `False` | Apply SSC low-pass filter after each PFC step |

#### AI Assistants

Claude (Anthropic) · Gemini (Google) · GPT (OpenAI) · DeepSeek

---

### 5. `super_dns_one_v6.py` — SUPER DNS ONE  `v6.1`

**Role:** 3-D compressible Direct Numerical Simulation / Large-Eddy Simulation solver at the continuum scale.

The largest and most comprehensive module in the cluster. Solves the **compressible Navier–Stokes equations** on a structured 3-D grid with high-order flux schemes, multiple boundary conditions, real-gas EOS, immersed boundary method, and full coupling to the CH and Langevin solvers via `one_core` bridges.

#### Key Classes

| Class | Role |
|---|---|
| `CFDConfig` | Full solver configuration (grid, physics, schemes, BC) |
| `CompressibleSolver` | Main solver — time-steps the compressible NS system |
| `RiemannSolverBase` | Base class for flux computation |
| `AUSMPlusFlux` | AUSM+ Riemann solver (compressible flows) |
| `HLLCFlux` | HLLC Riemann solver (shock-capturing) |
| `SemiLagrangianAdvection3D` | Unconditionally stable trilinear semi-Lagrangian advection |
| `SOCController` | CSOC-based adaptive eddy-viscosity for LES |
| `ItoStressGenerator` | Structural Itô stochastic stress injection |
| `DiffRGRefiner` | Differentiable renormalisation-group field refiner |
| `RealGasEOS` | Real-gas equation of state (CoolProp integration) |
| `ImmersedBoundary` | IBM body-force penalisation for complex geometries |
| `SignalDenoiser` | Wavelet-based diagnostics denoiser |

#### Boundary Conditions

| Class | Type |
|---|---|
| `PeriodicBC` | Periodic (DNS default) |
| `SupersonicInflowBC` | Supersonic inflow with Riemann characteristic prescription |
| `SubsonicOutflowBC` | Subsonic outflow (pressure-specified) |
| `NoSlipIsothermalWallBC` | Isothermal no-slip wall |
| `WernerWengleWallModelBC` | Wall-model LES (Werner–Wengle log-law) |
| `MovingWallBC` | Moving wall with velocity and temperature prescription |
| `FarFieldBC` | Far-field characteristic (external aerodynamics) |

#### Advection Schemes

| Scheme | Order | Notes |
|---|---|---|
| `upwind` | 1st | Stable, diffusive |
| `tvd` | 2nd | Slope-limited (minmod / van_leer / superbee) |
| `weno5` | 5th | Jiang–Shu WENO-5, Lax–Friedrichs splitting |
| `semi_lagrangian` | 3rd | Unconditionally stable, trilinear `grid_sample` (GPU) |

#### CH and Langevin Coupling

`CompressibleSolver` accepts external fields injected by bridge objects:

```python
# CH → DNS coupling (Korteweg body force + density/viscosity modulation)
from one_core import CahnHilliardDNSBridge

bridge = CahnHilliardDNSBridge(ch_solver, dns_solver, korteweg_strength=1e-4)
u_new = ch_solver.step(u, sigma)
bridge.sync(u_new)       # writes _ext_rho_ch, _ext_nu_ch, _ext_fx/fy/fz
dns_solver.step()        # _compute_rhs() reads and blends these buffers

# Langevin → DNS coupling
from one_core import LangevinDNSBridge
bridge = LangevinDNSBridge(langevin_integrator, dns_solver)
bridge.sync(coords)
```

#### Usage

```python
from super_dns_one_v6 import CompressibleSolver, CFDConfig
import numpy as np

nx, ny, nz = 128, 64, 64
xs = np.linspace(0, 2*np.pi, nx+1)
ys = np.linspace(0, np.pi,   ny+1)
zs = np.linspace(0, np.pi,   nz+1)

cfg = CFDConfig(
    x_coords=xs, y_coords=ys, z_coords=zs,
    advection_scheme="weno5",
    flux_scheme="hllc",
    gamma=1.4,
    device="cuda",
)
solver = CompressibleSolver(cfg)
solver.initialize_taylor_green(mach=0.1)

for step in range(n_steps):
    solver.step()
```

#### AI Development Partners

Claude (Anthropic) · GPT (OpenAI) · Gemini (Google) · DeepSeek

---

### 6. `structural_fno_3d.py` — Structural FNO 3D  `v2.0.0`

**Role:** AI surrogate model trained on SUPER DNS ONE cluster output.

A **Structural Fourier Neural Operator** that learns the mapping:

```
G : (u(x, 0), σ(x), t_target) ↦ u(x, t_target)
```

delivering O(1)-time predictions for any trained physical system — replacing hours of DNS/FH/CH simulation with a single neural network forward pass. Designed specifically to consume `.pt` snapshot files saved by the SUPER DNS ONE solvers.

#### Architecture

```
Input (B, 5, Nx, Ny, Nz)             u₀ + xyz grid + t_target
        │
   Lifting MLP                        Conv1×1×1 (5 → width×2 → width)
        │
   N × StructuralFNOLayer
   ├── MultiScaleSpectralConv3d       Coarse + fine Fourier branches
   ├── BoundaryPaddingConv3d          Local linear op (periodic-BC aware)
   ├── StructuralFiLM                 σ-conditioned Feature-wise Linear Mod.
   ├── StructuralCrossAttention       u × σ cross-attention at boundaries
   └── GroupNorm + GELU + residual
        │
   MCDropoutHead                      mean + log_variance output
        │
Output (B, 1, Nx, Ny, Nz) × 2        mean, log_var
```

#### Key Classes

| Class | Role |
|---|---|
| `MultiScaleSpectralConv3d` | Multi-resolution 3-D spectral convolution (coarse + fine Fourier modes with learnable blend) |
| `StructuralFiLM` | Feature-wise Linear Modulation conditioned on σ(x) |
| `StructuralCrossAttention` | Cross-attention between u latent features and σ field |
| `MCDropoutHead` | Uncertainty-aware output with Monte-Carlo Dropout |
| `StructuralFNOLayer` | Full SFNO block (spectral + local + FiLM + attention + residual) |
| `StructuralFNO3D` | Top-level model |
| `PhysicsLoss` | Physics-informed composite loss (MSE + NLL + mass + energy + smoothness) |
| `SuperDNSDataset` | Data pipeline for SUPER DNS ONE `.pt` snapshot files |
| `TrainerConfig` | Training hyperparameter configuration |
| `StructuralFNOTrainer` | Full training loop with AMP, cosine LR, checkpointing, TensorBoard |
| `BoundaryPaddingConv3d` | Conv3d with physics-aware BC padding (circular / replicate / zero) |

#### PhysicsLoss Terms

| Term | Symbol | Description |
|---|---|---|
| MSE | L_mse | Standard data-fit loss |
| NLL | L_nll | Negative log-likelihood (calibrates uncertainty) |
| Mass | L_mass | `|∫u_pred − ∫u_true| / |∫u_true|` — Cahn–Hilliard mass conservation |
| Energy | L_energy | `max(0, E(u_pred) − E(u₀))` — energy monotonicity (gradient flow) |
| Smoothness | L_smooth | Structural Itô regularisation — penalises gradients in high-σ (ordered) regions |

#### Data Pipeline

`SuperDNSDataset` expects `.pt` files with the following keys — matching the format produced by the SUPER DNS ONE solvers:

```python
{
    "u_t0":     Tensor,     # (1, Nx, Ny, Nz)  initial field
    "u_tT":     Tensor,     # (1, Nx, Ny, Nz)  target field
    "sigma":    Tensor,     # (1, Nx, Ny, Nz)  structural regime field
    "t_target": float,      # normalised target time ∈ [0, 1]
    "metadata": dict,       # solver config, dx, dt, etc. (optional)
}
```

#### Usage

```python
from structural_fno_3d import (
    StructuralFNO3D, TrainerConfig,
    StructuralFNOTrainer, SuperDNSDataset, PhysicsLoss,
)

# Build model
model = StructuralFNO3D(
    modes_coarse=12, modes_fine=6,
    width=64, num_layers=6,
    n_heads=4, dropout_p=0.1,
)

# One-shot inference
mean, log_var = model(u_initial, sigma, t_target=0.5)
std = (0.5 * log_var).exp()        # point-wise std dev

# MC uncertainty estimation
result = model.predict_with_uncertainty(u_initial, sigma, n_samples=32)
print(result["mean"].shape)        # (B, 1, Nx, Ny, Nz)
print(result["std"].shape)         # (B, 1, Nx, Ny, Nz)

# Training
ds = SuperDNSDataset("/data/dns_snapshots", normalise=True)
cfg = TrainerConfig(
    max_epochs=200, batch_size=4,
    lr=3e-4, use_amp=True,
    ckpt_dir="./checkpoints",
    log_dir="./runs",
)
trainer = StructuralFNOTrainer(model, ds, cfg)
trainer.train()

# Resume from checkpoint
trainer.train(resume="./checkpoints/sfno_best.pt")
```

#### AI Assistants

Claude (Anthropic) · Gemini (Google) · GPT (OpenAI)

---

## Cross-Solver Coupling Architecture

The full coupling graph between all six modules:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         one_core_v3.py                                   │
│  SemanticStateContraction · CSOCBase · InterfaceDetectorBase             │
│  StructuralItoBase · structural_biharmonic_n · ONE_VERSION               │
│  LangevinFHBridge · LangevinDNSBridge                                    │
│  CahnHilliardFHBridge · CahnHilliardDNSBridge                           │
└───────┬──────────────────┬──────────────────┬───────────────────────────┘
        │ imports           │ imports           │ imports
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────────────────────┐
│  Structural  │   │   Structural     │   │    SUPER DNS ONE v6.1       │
│  Langevin    │   │   Fluctuating    │   │    (CompressibleSolver)     │
│  v3.1        │   │   Hydro v6.1     │   │    Riemann: AUSM+, HLLC    │
│  (BAOAB MD)  │   │   (LLNS FH)      │   │    Advection: WENO-5, TVD  │
└──────┬───────┘   └────────┬─────────┘   └──────────────┬──────────────┘
       │                    │                              │
       │ LangevinFHBridge   │ CahnHilliardFHBridge         │ CahnHilliardDNSBridge
       │ LangevinDNSBridge  │                              │
       └──────────┬─────────┘                              │
                  │                                        │
                  ▼                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │              Structural Cahn–Hilliard 3D v2              │
        │   StructuralCahnHilliard3D                               │
        │   ThinFilmStructuralCahnHilliard3D                       │
        │   PhaseFieldCrystal3D                                    │
        │   Laplacians: Conv3d · FFT · Roll                        │
        └──────────────────────────┬───────────────────────────────┘
                                   │
                          Training data (.pt snapshots)
                                   │
                                   ▼
                 ┌─────────────────────────────────────┐
                 │       Structural FNO 3D v2.0         │
                 │  StructuralFNO3D · PhysicsLoss       │
                 │  SuperDNSDataset · StructuralFNOTrainer │
                 └─────────────────────────────────────┘
```

---

## Installation & Requirements

### Python Dependencies

```
torch >= 2.0
numpy
scipy
matplotlib
```

Optional (for full `super_dns_one_v6.py` features):

```
optuna          # Bayesian hyperparameter optimisation
CoolProp        # Real-gas equation of state
pywavelets      # Wavelet-based diagnostics
tensorboard     # Training visualisation (structural_fno_3d.py)
```

### Installation

```bash
pip install torch numpy scipy matplotlib
pip install optuna CoolProp PyWavelets tensorboard   # optional
```

### File Order

Always place `one_core_v3.py` in the same directory as the solver files (or on your `PYTHONPATH`). All other files import from it at module load time:

```python
from one_core import SemanticStateContraction, CSOCBase, ...
```

---

## Quick Start

### Run individual solver self-tests

Each module includes a built-in verification suite. Run any file directly:

```bash
python one_core_v3.py
python structural_langevin_v3.py
python structuralfluctuatinghydro_v6.py
python structural_cahn_hilliard_3d.py
python super_dns_one_v6.py
python structural_fno_3d.py
```

All verification suites print `[PASS]` / `[FAIL]` for each test case and exit with code `0` on success.

### Minimal two-solver pipeline (CH + DNS)

```python
import torch
from structural_cahn_hilliard_3d import (
    CahnHilliardConfig, StructuralCahnHilliard3D, make_sigma_field
)
from super_dns_one_v6 import CompressibleSolver, CFDConfig
from one_core import CahnHilliardDNSBridge
import numpy as np

N = 64
# Build solvers
ch_cfg = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-5,
                             laplacian="fft", device="cuda")
ch     = StructuralCahnHilliard3D(ch_cfg).to("cuda")
sigma  = make_sigma_field(N, N, N, background=1.0, device="cuda")

xs = np.linspace(0, 2*np.pi, N+1)
dns_cfg = CFDConfig(x_coords=xs, y_coords=xs, z_coords=xs,
                    advection_scheme="weno5", device="cuda")
dns = CompressibleSolver(dns_cfg)

# Bridge — Pattern A
bridge = CahnHilliardDNSBridge(ch, dns, korteweg_strength=1e-4)

u = torch.zeros(N, N, N, device="cuda", dtype=torch.float64)

for step in range(1000):
    u     = ch.step(u, sigma)
    bridge.sync(u, sigma)       # push density, viscosity, Korteweg force
    dns.step()                  # DNS reads the injected fields
```

### Train FNO surrogate on DNS snapshots

```python
from structural_fno_3d import (
    StructuralFNO3D, TrainerConfig, StructuralFNOTrainer, SuperDNSDataset
)

model   = StructuralFNO3D(modes_coarse=12, modes_fine=6, width=64, num_layers=6)
dataset = SuperDNSDataset("/path/to/snapshots", normalise=True)
cfg     = TrainerConfig(max_epochs=200, batch_size=4, use_amp=True)
trainer = StructuralFNOTrainer(model, dataset, cfg)
trainer.train()
```

---

## Theoretical Framework — Structural Calculus

The entire cluster is grounded in a four-paper series on **Structural Calculus**:

| Paper | Title | Key contribution |
|---|---|---|
| Paper 1 | Regime-Dependent Analytical Framework | σ(x)-modulated operators: grad_S, div_S, Δ_S |
| Paper 2 | BV Jump Measures & Self-Evolving Interfaces | Concentrated stochastic increments at phase boundaries |
| Paper 3 | Structural Itô Calculus & Multiplicative Noise | Itô drift correction for state-dependent noise |
| Paper 4 | Controlled Self-Organised Criticality (CSOC) & SSC | Adaptive thermostat / viscosity driven by EMA criticality signal |

**Key shared concept — the structural field σ(x):**

σ(x) is a spatially varying scalar field that encodes local phase information across all solvers. In the ordered phase, σ is large; at interfaces and critical points, σ is small. All Structural Calculus operators — Δ_S, grad_S, div_S — are weighted by σ(x), so diffusion, noise, and forcing are automatically concentrated where the physics demands it.

---

## Version History

| Module | Current version | Key milestone |
|---|---|---|
| `one_core_v3.py` | 3.1.0 | Full 5-solver bridge interoperability |
| `structural_langevin_v3.py` | 3.1 | BAOAB + CSOC + one_core integration |
| `structuralfluctuatinghydro_v6.py` | 6.1 | CH↔FH bridge + full differentiability (9 fixes) |
| `structural_cahn_hilliard_3d.py` | v2 | GPU Conv3d/FFT Laplacian, ThinFilm, PFC, CH↔DNS bridge |
| `super_dns_one_v6.py` | 6.1 | CH↔DNS bridge + full differentiability (8 fixes) |
| `structural_fno_3d.py` | 2.0.0 | MultiScale FFT, FiLM, CrossAttention, MC-Dropout, PhysicsLoss |

---

## Developer & Attribution

| Field | Value |
|---|---|
| Developer | Yoon A Limsuwan |
| Organization | MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT |
| ORCID | 0009-0008-2374-0788 |
| GitHub | yoonalimsuwan |
| Contact | msps4u@gmail.com |
| License | MIT |
| Year | 2026 |

### AI Development Partners

All six modules in the SUPER DNS ONE cluster were developed with the assistance of multiple AI systems:

| AI | Organization | Role |
|---|---|---|
| **Claude** | Anthropic | Architecture design, differentiability audits, bridge protocol, integration testing, documentation |
| **GPT** | OpenAI | Algorithmic suggestions, flux scheme review, code review |
| **Gemini** | Google | Numerical scheme cross-validation, operator scaffolding |
| **DeepSeek** | DeepSeek | Supplementary code analysis, stencil verification |

---

## License

All modules in the SUPER DNS ONE cluster are released under the **MIT License**.

```
Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
