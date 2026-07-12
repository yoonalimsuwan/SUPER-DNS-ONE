# SEISMIC ONE — Earthquake & Tsunami Engineering Modules

Part of the **ONE Ecosystem**. Adds earthquake engineering and seismically-driven
fluid dynamics to the platform, coupled to **SUPER DNS ONE** (compressible
CFD/DNS/LES) via **ONE Core**.

| File | Role |
|---|---|
| `seismic_one.py` | Standalone earthquake engineering: ground motions, site response, structural response, liquefaction |
| `seismic_dns_coupling_one.py` | Converts seismic outputs into CFD-ready forcing/initial conditions |
| `one_core_v3.py` | Houses `SeismicDNSBridge` (canonical, shared across solvers) |
| `super_dns_one_v6_3.py` → `v6.4` | CFD/DNS solver; patched to actually consume coupling buffers |

---

## 1. Scope

`seismic_one.py` covers **structural and geotechnical earthquake engineering**
(elastodynamics, site response, structural dynamics) — it is **not** a
seismic-wave-propagation solver for the Earth's crust. It complements
SUPER DNS ONE for problems where shaking drives a *fluid* — tank sloshing,
tsunami generation, liquefied-soil flow — while structural response itself
stays in the solid-mechanics domain modeled here.

---

## 2. `seismic_one.py` — core module

### 2.1 GroundMotionEngine
- Loads or synthesizes ground-motion acceleration time histories
- Synthetic motions via a Clough–Penzien-filtered, Jennings-enveloped white-noise process, scaled to a target PGA
- Baseline correction (polynomial detrending)
- Response spectra (Sa, Sv, Sd) via the **exact Nigam–Jennings piecewise-linear recursion** — not a coarse numerical integrator

### 2.2 SiteResponseLayer
1D equivalent-linear site response analysis (SHAKE-style):
- Frequency-domain transfer function through a layered viscoelastic soil column over an elastic half-space (Kramer 1996 / Schnabel et al. 1972 formulation)
- Iterates shear modulus (G) and damping (D) to strain-compatible values using modulus-reduction / damping curves (default: modified-hyperbolic Darendeli-type)
- Returns surface motion, converged G/Gmax and damping per layer, peak shear strain, amplification factor

**Known limitation:** the default reference strain (γᵣ) is held constant rather than scaled with confining stress and plasticity index. For site-specific work, supply lab-measured or stress-adjusted `G_over_Gmax_curve` / `damping_curve` callables per layer.

### 2.3 StructuralResponseLayer
Lumped-mass MDOF shear-building model:
- Eigen-analysis for modal periods/shapes
- Nonlinear time-history response via **Newmark-β** (average acceleration, unconditionally stable) with **Rayleigh damping**
- **Bouc–Wen** smooth hysteretic inter-story springs, capturing stiffness degradation / damage under strong shaking
- Outputs: drift ratios, story forces, damage index per story (drift-based, Park–Ang-inspired proxy)

Validated: for a single-story elastic system, the Newmark-β time-history peak displacement matched an independently-computed SDOF response-spectrum value to **<0.1% error**.

### 2.4 LiquefactionAssessment
Simplified stress-based liquefaction triggering (Seed & Idriss 1971; Idriss & Boulanger 2008 conventions):
- Idriss (1999) depth-reduction factor rd(z)
- Magnitude Scaling Factor (MSF) correction to Mw = 7.5 equivalent
- CRR from SPT blow count via the Idriss & Boulanger (2008) correlation
- Returns Factor of Safety (FS = CRR/CSR) per depth; FS < 1 → liquefiable

### 2.5 SeismicOne (orchestrator)
Chains the pipeline: bedrock motion → site response → structural response,
and reports a plain-text summary.

---

## 3. `seismic_dns_coupling_one.py` — CFD coupling layer

Three physical coupling scenarios, each producing CFD-ready forcing/IC data:

### 3.1 Tank sloshing → `TankSloshingCoupling`
Non-inertial-frame body force for fluid in a tank rigidly attached to a
shaking structure:

```
f_volumetric(x,t) = −ρ(x,t) · a_ground(t)
```

added directly to the CFD momentum equation. Vertical component falls back
to a code-typical 2/3 × horizontal approximation if not separately supplied.

### 3.2 Tsunami generation → `SeabedDisplacementCoupling`
- Stylized rectangular-dislocation seabed uplift/subsidence (dipole
  approximation of the near-field Okada 1985 pattern — sign-correct,
  non-singular, **not** a substitute for a validated Okada implementation
  in real hazard work)
- Kajiura (1963) filtering: smooths seabed-displacement wavelengths short
  relative to water depth before they reach the free surface
- `to_ch_order_parameter()`: builds a 3D Cahn–Hilliard order-parameter
  field for direct use with the real `CahnHilliardDNSBridge`

### 3.3 Liquefied soil as non-Newtonian flow → `LiquefiedSoilForcingCoupling`
- Converts `LiquefactionAssessment` FS<1 zones into Herschel–Bulkley
  rheology (yield stress from an empirical fit to residual-strength vs.
  N₁₆₀cs trends; shear-thinning flow index)
- Non-liquefiable depths return infinite apparent viscosity (treated as solid)

### 3.4 DNS wiring
- `make_seismic_dns_bridge()` — factory that wires `TankSloshingCoupling`
  into the canonical `one_core_v3.SeismicDNSBridge`, handling the sign-
  convention conversion automatically
- `LiquefiedSoilDNSCoupling` — gated behind explicit confirmation that the
  solver has the `_ext_nu_ch` fix (see §4.2); refuses to run otherwise,
  to avoid a silent no-op simulation

---

## 4. Findings from reading the real solver source

Built against the actual `super_dns_one_v6_3.py` and `one_core_v3.py`
(not assumed APIs). Two real issues were found and fixed along the way:

### 4.1 Transfer-function sign/direction bug (`seismic_one.py`, caught during validation)
An early implementation of `SiteResponseLayer`'s layered transfer function
recursed bottom-up from bedrock, which is provably (and was numerically
confirmed) insensitive to the soil/rock impedance contrast — it produced a
flat, non-resonant response. Rewritten as the standard **top-down** SHAKE
recursion (free-surface condition at the top, H = 1/A_bedrock), validated
against the closed-form single-layer amplification formula to <1e-4 error.

### 4.2 `_ext_nu_ch` dead buffer (confirmed across both files)
`CahnHilliardDNSBridge.sync()` (`one_core_v3.py`) writes
`dns_solver._ext_nu_ch = nu_eff` every call, but `CompressibleSolver._compute_rhs`
(`super_dns_one_v6_3.py`, through v6.3) never read that buffer — only
`_ext_rho_ch` and `_ext_fx/fy/fz` were consumed. **Effect:** in every prior
two-phase CH-DNS run, the density contrast between phases took effect but
the viscosity contrast silently did not. **Fixed** in v6.4 by adding, right
after `mu_lam` is computed:

```python
if self._ext_nu_ch.abs().max() > 1e-30:
    mu_lam = mu_lam + rho * self._ext_nu_ch
```

### 4.3 No gravity term in the solver
`super_dns_one_v6_3.py` has no gravity/buoyancy body force anywhere in
`_compute_rhs`. If a tank-sloshing case needs hydrostatic gravity as well
as seismic forcing, add it separately — bridges **overwrite**
`_ext_fx/fy/fz` on each `sync()` call rather than accumulating.

---

## 5. End-to-end usage example

```python
from seismic_one import (
    GroundMotionEngine, SoilLayerProps, SiteResponseLayer, SeismicOne,
)
from seismic_dns_coupling_one import TankSloshingCoupling, make_seismic_dns_bridge
from super_dns_one_v6_3 import CompressibleSolver, CFDConfig

# 1. Ground motion + site response
gme = GroundMotionEngine()
bedrock = gme.synthetic_motion(duration=20, dt=0.005, pga_target=3.0, dominant_freq=2.5)

layers = [SoilLayerProps(thickness=8, density=1700, Gmax=50e6)]
site = SiteResponseLayer(layers, rock_density=2200, rock_Vs=760)

so = SeismicOne(site_layer=site)
results = so.run(bedrock, do_site_response=True, do_structure=False)
surface_motion = results["surface_motion"]
print(so.summary(results))

# 2. Couple to SUPER DNS ONE for tank sloshing
tank = TankSloshingCoupling(surface_motion)
solver = CompressibleSolver(CFDConfig(...))
solver.initialize("uniform")

bridge = make_seismic_dns_bridge(solver, tank, directions="xz")
for step in range(n_steps):
    bridge.sync()
    solver.step()
```

---

## 6. Validation summary

| Check | Result |
|---|---|
| Synthetic motion PGA scaling | matches target to <0.05 m/s² |
| SDOF Newmark-β vs. response spectrum | <0.1% relative error |
| Site response vs. closed-form single-layer amplification | <1.3e-4 absolute error |
| Weak-story drift concentration (nonlinear MDOF) | reproduced correctly |
| Liquefaction: loose vs. dense sand | correctly differentiated (FS<1 vs FS>1) |
| Herschel–Bulkley shear-thinning | apparent viscosity decreases with strain rate, verified |
| Okada dipole sign convention | flipping rake sign flips uplift/subsidence pattern, verified |
| `SeismicDNSBridge` unit conversion (ρ·a) | verified against mock tensor solver |
| `make_seismic_dns_bridge` sign unwrapping | verified against mocked `one_core_v3` |

---

## 7. Known gaps / recommended next steps

- **Not implemented:** full elastodynamic wave propagation through the
  crust (fault rupture → seismic wave field). The Okada patch is a
  stylized dipole, not a validated dislocation solver.
- **Not implemented:** tank *wall* flexibility (fluid-structure
  interaction on the shell itself) — the current tank-sloshing coupling
  assumes a rigid tank.
- **Recommended before production tsunami runs:** verify `_ext_rho_ch`'s
  50/50 blend behavior against your intended IC; for a clean tsunami
  source, set `solver.rho` directly rather than relying on the blend.
- **Recommended before production liquefaction-CFD runs:** confirm you
  are running the patched `super_dns_one_v6_3.py` (v6.4+) before setting
  `confirm_solver_patched=True`.
- No GPU/torch was available in the environment these modules were
  developed in — `SeismicDNSBridge` and `make_seismic_dns_bridge` were
  validated with mock tensors only. Run a short smoke test (a few steps,
  check `_ext_fx` for NaN/Inf) on real hardware before a full run.

---

*Co-developed with Claude (Anthropic), as an AI co-developer, consistent
with the ONE Ecosystem attribution convention.*
