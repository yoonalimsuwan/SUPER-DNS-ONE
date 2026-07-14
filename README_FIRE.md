# FIRE ONE — Fire Dynamics Engineering & Fire CFD Modules

Part of the **ONE Ecosystem**. Adds fire engineering analysis and
CFD-coupled fire/combustion physics to the platform, integrated with
**SUPER DNS ONE** via **ONE Core**.

| File | Role |
|---|---|
| `fire_one.py` | Standalone fire engineering: design fires, plume correlations, combustion state relations, radiation, tenability |
| `fire_dns_coupling_one.py` | Converts fire engineering outputs into CFD source terms (prescribed-HRR path) |
| `one_core_v3.py` | Houses `HeatReleaseDNSBridge` (canonical, shared bridge) |
| `super_dns_one_v6_3.py` → `v6.6` | CFD/DNS solver; patched with a resolved mixture-fraction combustion field and P1 radiation |

---

## 1. Life-safety notice — read first

Fire and tenability modeling directly informs evacuation timing and
firefighter safety decisions. Every correlation here is a published,
simplified engineering approximation (Heskestad, Purser, Jin, NFPA
design-fire curves) — the same category of tool used in the SFPE
Handbook and NIST FDS's own tenability post-processing, **not** a
replacement for validated CFD fire modeling (NIST FDS itself),
professional fire protection engineering review, or code-required
life-safety analysis. Do not use this module as the sole basis for an
actual evacuation plan, egress design, or firefighting decision.

---

## 2. Scope: what this is, and what it deliberately is not

This is **not** a reproduction of NIST FDS. FDS represents 20+ years of
dedicated development (discrete-ordinates radiative transport,
finite-rate/soot chemistry kinetics, pyrolysis models, sprinkler/
detector/HVAC devices, extensive experimental validation). What's
implemented here moves toward FDS's **modeling category** — conserved-
scalar (mixture-fraction) fast-chemistry combustion plus a simplified
radiation model — not a validated replacement for it.

**Not implemented** (explicitly out of scope):
- Discrete-ordinates radiative transport equation (RTE)
- Finite-rate/soot formation-oxidation chemistry kinetics
- Pyrolysis / solid-fuel decomposition models
- Sprinkler, detector, HVAC device models
- Any experimental validation against real fire test data

---

## 3. `fire_one.py` — core engineering module

### DesignFireCurve
Standard t-squared HRR growth: `Q(t) = alpha·t²`, with NFPA-standard
growth-rate categories (slow/medium/fast/ultrafast). Supports a
steady-burning plateau and simplified linear decay.

### PlumeCorrelations
Validated closed-form fire plume physics (Heskestad 1983; McCaffrey
1979), as used throughout the SFPE Handbook and NIST FDS verification
suites:
- Heskestad mean flame height
- Heskestad centerline temperature rise and velocity (with virtual
  origin correction)
- McCaffrey plume-regime classification (continuous flame /
  intermittent / buoyant plume)

### MixtureFractionCombustion
Burke-Schumann (fast, infinitely-fast diffusion-flame-sheet) state
relations — the same modeling approach used by NIST FDS's default
(non-finite-rate) combustion model:
- Stoichiometric mixture fraction from fuel formula (validated:
  propane's stoichiometric air/fuel mass ratio matches the known
  ~15.6:1 value)
- Adiabatic flame temperature (single-step energy balance)
- Piecewise-linear species/temperature state relations vs. Z

**Bug found and fixed during validation:** the adiabatic flame
temperature calculation initially used room-temperature air `cp`
(1005 J/kg·K), overshooting propane's known adiabatic flame temperature
(~2260-2390K) by ~800K (predicted 3077K). Combustion product gases have
roughly double that heat capacity when averaged from ambient to flame
temperature. Fixed using a representative `CP_COMBUSTION_PRODUCTS`
(1400 J/kg·K); validated result: 2292K, matching the literature value.

### RadiationModel
- Point-source incident flux (SFPE Handbook engineering method)
- Critical distance for a given flux threshold (piloted ignition,
  tenable exposure limits)
- Optically-thin volumetric radiative loss

### TenabilityAssessment
Fractional Effective Dose (FED) model (Purser; ISO 13571) — the
standard life-safety post-processing used with NIST FDS:
- FED contribution from CO exposure
- FED contribution from convective heat exposure
- Visibility via Jin's correlation from smoke extinction
- Full-timeline FED integration with time-to-incapacitation

### FireOne (orchestrator)
Chains design fire → plume analytics → combustion state → tenability,
with a plain-text summary.

---

## 4. `fire_dns_coupling_one.py` — CFD coupling (prescribed-HRR path)

Two mechanisms, both real verified hooks into SUPER DNS ONE:

### Buoyancy
Fire plumes are driven by buoyant acceleration. SUPER DNS ONE has no
gravity term anywhere in `_compute_rhs` (confirmed by reading the
source). Buoyancy is wired through the **same** `SeismicDNSBridge` used
for seismic body forces — gravity is just a constant "ground
acceleration" of magnitude `G_ACCEL`. No new bridge class needed.

### Heat release
Combustion is a direct volumetric energy-equation source, which
`SeismicDNSBridge`'s mechanical-work-only buffers cannot represent.
Wired through `one_core_v3.HeatReleaseDNSBridge` (v3.3.0+), writing the
new `_ext_q` buffer (v6.5+). `FireSourceField` shapes a `DesignFireCurve`
HRR(t) into a Gaussian-plume-shaped volumetric source at the fire's
location, sized by the fire diameter and instantaneous Heskestad flame
height.

**Validated:** the field's spatial integral matches the fire's
convective HRR to <0.001% error (grid-resolution-limited only); zero
heat below the fire base; zero source before ignition.

This path is a **prescribed** external HRR — the flame shape/intensity
is set from `fire_one.py`'s analytics, not computed from a locally
resolved flame. For locally-resolved combustion, see §5.

---

## 5. `super_dns_one_v6_3.py` v6.6 — resolved combustion + radiation

Requested upgrade: move from "prescribed HRR fed in externally" toward
FDS's actual default combustion model — mixture fraction as a genuinely
**resolved, transported field**, with heat release computed locally
from it.

### Mixture-fraction transport
`self.RZ = ρZ` (conserved form, consistent with how ρ, ρu, ρv, ρw, ρE
are all transported) is now carried through the full TVD-RK3
integration in `step()`, alongside the five Euler variables.

Deliberately uses simple centered-difference advection/diffusion
(matching this file's existing viscous-term style), **not** hooked into
`flux_solver`'s Godunov/Riemann Euler scheme — Z has no acoustic/shock
structure of its own, and avoiding the compressible flux solver means
this addition cannot destabilize the proven core. Trade-off: more
diffusive than a dedicated upwind/TVD scalar scheme; validate against
your own grid-Peclet-number requirements before trusting sharp flame
fronts.

### Local combustion source (`cfg.enable_combustion=True`)
Burke-Schumann fast-chemistry equilibrium temperature evaluated on the
**resolved local Z field**, relaxed toward via a "presumed equilibrium"
closure at the local turbulent mixing rate (`tau_mix ~ dx²/D_Z`) —
chosen deliberately over an invented scalar-dissipation reaction-rate
formula, to avoid presenting an unvalidated closure as authoritative.

Validated properties (checked algebraically):
- `T_eq(Z)` correctly returns ambient temperature at both unmixed limits
  (Z=0 pure air, Z=1 pure unburnt fuel) and peaks at the adiabatic flame
  temperature exactly at the stoichiometric mixture fraction
- Relaxation source is positive (heating) when local T lags the local
  equilibrium, and exactly zero once equilibrium is reached

### P1 radiation (`cfg.enable_radiation=True`)
Diffusion approximation to the radiative transport equation, solved via
FFT. Assumes **periodic** boundaries (matching this solver's default
circular-padding convention) — not rigorously valid for a wall-dominated
enclosure, which would need Marshak boundary conditions and an iterative
elliptic solve instead.

Validated properties (checked algebraically):
- Uniform temperature field → zero net radiative source (correct
  radiative equilibrium)
- Localized hot spot in cold surroundings → positive net radiative loss
  at the hot spot

### Critical bug found and fixed during implementation
This solver is **fully non-dimensionalized** (confirmed: the existing
Sutherland's-law viscosity formula uses a hardcoded `T_ref=300.0`
reference, meaning the solver's internal `T` variable is
`T_real / 300K`, not Kelvin). An early version of this combustion/
radiation implementation used real-Kelvin constants (293.15K ambient,
2260K adiabatic flame temperature, Stefan-Boltzmann with real K⁴)
directly against that non-dimensional `T` — a severe unit inconsistency.

**Fixed** by converting T to real Kelvin via `self.T_ref` at the point
of use. A second, **unresolved** gap remains and is explicitly flagged
rather than silently assumed: the resulting combustion/radiation source
terms are physically computed in real (W/m³) units, but added to
`rhs_rhoE`, which lives in the solver's own non-dimensional unit system
(scaled by reference length/velocity/density implied by `cfg.Re`/`cfg.Mach`).
This file does not have enough visibility into those reference scales to
derive the correct conversion factor automatically.

**`cfg.combustion_nondim_scale`** (default `1.0` = no conversion applied)
is an explicit, visible, user-must-set knob for this gap. **Do not trust
quantitative combustion/radiation magnitudes until you have computed and
set this correctly** (approximately `L_ref / (rho_ref · U_ref³)`) for
your specific non-dimensionalization choices.

### Checkpoint compatibility
`RZ` is now included in `save_checkpoint`/`load_checkpoint`. Loading a
pre-v6.6 checkpoint with `enable_combustion=True` logs a warning and
resumes with `Z=0` (unburnt) everywhere, rather than raising or silently
losing state.

---

## 6. `one_core_v3.py` v3.3.0 — `HeatReleaseDNSBridge`

Follows the same convention as `SeismicDNSBridge`/`CahnHilliardDNSBridge`:
duck-typed source (`q_dot` is `None`, a constant, or an object exposing
`.field(t) -> Tensor`), torch-only dependency, overwrite (not
accumulate) semantics on `sync()`.

**Deliberately raises at construction** if the solver doesn't have an
`_ext_q` buffer (i.e. predates v6.5) — this is the same lesson learned
from the earlier `_ext_nu_ch` dead-buffer bug (a buffer written but never
read, silently producing a no-op simulation). `HeatReleaseDNSBridge`
checks for the buffer's existence up front rather than repeating that
failure mode.

---

## 7. Usage example

```python
from fire_one import DesignFireCurve, FireOne, COMMON_FUELS
from fire_dns_coupling_one import (
    FireSourceField, make_buoyancy_dns_bridge, make_heat_release_dns_bridge,
)
from super_dns_one_v6_3 import CompressibleSolver, CFDConfig

# Prescribed-HRR path (simpler, no local Z resolution required)
fire_curve = DesignFireCurve("fast", q_peak_kw=2000.0)
fire = FireOne(fire_curve, COMMON_FUELS["polyurethane_foam"], fire_diameter_m=1.5)

solver = CompressibleSolver(CFDConfig(...))
solver.initialize("uniform")

buoyancy = make_buoyancy_dns_bridge(solver, vertical_axis="z")
source = FireSourceField(fire_curve, x0=0, y0=0, z0=0, diameter_m=1.5,
                          grid_x=..., grid_y=..., grid_z=...)
heat = make_heat_release_dns_bridge(solver, source)

for step in range(n_steps):
    buoyancy.sync()
    heat.sync()
    solver.step()

# Resolved-combustion path (v6.6): set cfg.enable_combustion=True,
# cfg.z_stoich, cfg.T_adiabatic_K, cfg.combustion_nondim_scale (!) before
# constructing CompressibleSolver, then just call solver.step() — no
# external bridge needed, combustion is now computed from the solver's
# own resolved mixture-fraction field each step.
```

---

## 8. Validation summary

| Check | Result |
|---|---|
| DesignFireCurve t-squared timing | matches growth-rate definition exactly |
| Heskestad flame height | plausible range for reference fire size |
| Heskestad centerline temperature decay with height | correct 5/3-power decay behavior |
| Propane stoichiometric air/fuel ratio | matches known ~15.6:1 value |
| Adiabatic flame temperature (after cp fix) | 2292K vs. literature ~2260-2390K |
| Burke-Schumann state relations | fuel/O2 mass fractions non-negative, correct limits at Z=0/1, peak T at Z_stoich |
| Point-source radiation inverse-square law | verified (flux at 2× distance = 1/4) |
| Critical-distance / point-source-flux inversion | exact round-trip |
| FED timeline monotonicity | verified non-decreasing |
| Visibility: light-emitting vs. reflecting signs | correct relative ordering |
| FireSourceField integral vs. convective HRR | <0.001% error |
| P1 radiation: uniform-T equilibrium | zero net source, verified |
| P1 radiation: hot-spot net loss | verified positive |
| Flame-sheet T_eq(Z) limits and peak | verified at Z=0, Z=1, Z=Z_stoich |

---

## 9. Known gaps / before production use

- **`cfg.combustion_nondim_scale` must be set correctly** for your own
  non-dimensionalization — this is not solvable generically without your
  reference length/velocity/density scales; defaulting to 1.0 is very
  unlikely to be physically correct for your case.
- **P1 radiation assumes periodic boundaries.** A walled enclosure needs
  a different (iterative, Marshak-BC) solve.
- **Z transport uses centered differences**, not a dedicated upwind/TVD
  scalar scheme — check for oscillations at under-resolved flame fronts.
- **No experimental validation** has been performed against real fire
  test data or NIST FDS itself.
- No GPU/torch was available in the environment these modules were
  developed in. Numerical logic was validated with algebraic/mock tests
  only (equilibrium limits, FFT radiation solve on plain arrays,
  source-field integrals) — run an actual smoke test (a few steps,
  checking for NaN/Inf in `rho`, `rhoE`, and `RZ`) on real hardware
  before a full run, especially with combustion and radiation both
  enabled together.

---

*Co-developed with Claude (Anthropic), as an AI co-developer, consistent
with the ONE Ecosystem attribution convention.*
