# STORM ONE — Severe Storm, Tornado & Tropical Cyclone Engineering Module

Part of the **ONE Ecosystem**.

## 1. Life-safety notice — read first

Severe storm, tornado, and hurricane hazard assessment directly informs
warnings, evacuation orders, and structural design decisions. This
module follows the same two-tier honesty discipline established across
this ecosystem's other hazard modules (`fire_one.py`, `seismic_one.py`,
`volcano_one.py`, `flood_one.py`): formulas that are **thermodynamic/
kinematic definitions** (CAPE, storm-relative helicity, the Rankine
vortex, the Holland wind profile's functional form) are implemented
with high confidence; formulas that are **fitted empirical regressions**
(the Holland B-parameter estimate, pressure-wind relationships, the
Bunkers storm-motion deviation, official EF-scale/Saffir-Simpson wind
thresholds reproduced from memory) are explicitly flagged as needing
verification against current literature/official sources. This is
**not** a replacement for a real numerical weather prediction model, a
supercell/tornado simulation (a cloud-resolving model), a real
storm-surge model (SLOSH, ADCIRC — this module's surge estimate ignores
bathymetry entirely), or professional review by a meteorologist. Do not
use this as the sole basis for a warning decision, evacuation order, or
structural design wind speed.

## 2. Scope

| Class | Covers |
|---|---|
| `AtmosphericStabilityModel` | CAPE/CIN from a lifted-parcel sounding, theoretical max updraft velocity |
| `WindShearModel` | Bulk shear, Bunkers storm motion, storm-relative helicity (SRH), Supercell Composite Parameter |
| `TornadoVortexModel` | Rankine combined vortex wind field (with translation), EF-scale classification |
| `TropicalCycloneModel` | **This is the hurricane/typhoon model** — a hurricane, typhoon, and tropical cyclone are the same physical phenomenon under different regional names. Holland (1980) parametric wind profile, pressure-wind relationship, Saffir-Simpson category |
| `StormSurgeModel` | Simplified inverse-barometer + 1D wind-driven setup |
| `StormDamageAssessment` | Generic wind-speed-based damage fraction curve |
| `StormOne` | Orchestrator: tornado-environment assessment, tropical-cyclone assessment |

**Not implemented:** derecho/straight-line windstorms, winter storms/
blizzards, hailstorm physics, a real numerical weather model of any
kind, multi-vortex/tornado-genesis dynamics, real bathymetry-resolved
storm surge.

## 3. Bug found and fixed during this module's own development

`StormSurgeModel.wind_setup()`'s 1D linear formula produced **≈19-20
meters** of storm surge for a strong hurricane at a long-fetch,
constant-shallow-depth combination — a value never observed in the
historical record (the highest documented storm surges, in the most
extreme funnel-shaped bays, are around 10 m). The formula assumes
constant water depth over the entire fetch and a small-perturbation
(linear) surface response; both assumptions silently break down for a
long fetch over shallow water, since real continental shelves deepen
away from shore and a linear surge response cannot exceed the water
depth itself without violating its own premise. **Fixed** by adding an
explicit `assumption_valid` check (false whenever the computed wind
setup exceeds half the water depth) and a `warning` message in the
returned result, rather than silently returning the implausible number.
Verified: realistic shelf parameters (50 km fetch, 15 m depth) for the
same storm give 5.96 m total surge — matching the commonly cited 3–6 m
range for major hurricanes — while the original problematic parameter
combination now correctly reports `assumption_valid=False`.

## 4. Explicitly flagged, unverified constants

- `WindShearModel.bunkers_storm_motion()`'s deviation magnitude
  (D≈7.5 m/s) reproduces the commonly cited Bunkers et al. (2000)
  value from memory — verify against the original paper.
- `WindShearModel.supercell_composite_parameter()` reproduces the
  standard SCP formula's *structure*; verify the exact normalization
  constants (1000, 50, 20) against Thompson et al. (2003).
- `TornadoVortexModel.EF_SCALE_THRESHOLDS_MPH` and
  `TropicalCycloneModel.SAFFIR_SIMPSON_MPH` are official NWS/NHC scale
  definitions reproduced from memory — verify exact mph boundaries
  against the current official references before regulatory/operational
  use.
- `TropicalCycloneModel.estimate_holland_B()` and
  `pressure_wind_relationship()` reproduce the correct qualitative
  trend and order of magnitude, not verified fits to a specific
  published source (e.g. Vickery & Wadhera 2008; Atkinson & Holliday
  1977 / Knaff & Zehr 2007).
- `StormSurgeModel.wind_setup()`'s drag coefficient (0.0026) is
  representative at high wind speeds but real Cd is wind-speed-dependent
  and somewhat uncertain at extreme winds — verify against current
  literature (e.g. Powell et al. 2003).

## 5. Validation summary

| Check | Result |
|---|---|
| Coriolis parameter: zero at equator, increases toward pole | Verified |
| Saturation vapor pressure: monotonic in T, correct at 0°C reference | Verified |
| CAPE: higher in an unstable environment than a stable one | Verified |
| LCL height in a physically plausible range | Verified |
| Bulk shear / SRH computed from a realistic veering wind profile | Verified |
| SCP = 0 in a zero-shear environment | Verified exactly |
| Rankine vortex: linear (solid-body) increase inside the core | Verified |
| Rankine vortex: 1/r decay outside the core | Verified |
| Rankine vortex: continuous at r_max | Verified (<1e-9 discontinuity) |
| EF-scale classification at both weak and violent ends | Verified |
| Ground-relative peak speed = rotational max + translation speed | Verified exactly |
| Holland profile: peaks near the radius of maximum wind | Verified |
| Holland profile: decays well outside Rmax | Verified |
| Pressure-wind relationship: deeper storm → higher Vmax | Verified |
| Saffir-Simpson category boundaries | Verified at multiple wind speeds |
| Inverse-barometer setup: deeper storm → more setup | Verified |
| Storm surge: realistic shelf parameters give a plausible magnitude | Verified (5.96 m vs. the commonly cited 3–6 m range) |
| Storm surge: unphysical fetch/depth combination correctly flagged | Verified (`assumption_valid=False`) |
| Damage fraction: monotonic in wind speed, bounded [0,1] | Verified |

## 6. Usage example

```python
from storm_one import (
    AtmosphericStabilityModel, WindShearModel, WindProfile,
    TropicalCycloneModel, StormOne,
)
import numpy as np

# Tornado-environment assessment
stability = AtmosphericStabilityModel(surface_T_K=303.15, surface_p_Pa=101300.0,
                                       surface_dewpoint_K=295.15)
profile = WindProfile(
    z_m=np.array([0, 1000, 3000, 6000]),
    u_m_s=np.array([5, 15, 25, 35]),
    v_m_s=np.array([0, 5, 3, -5]),
)
shear_model = WindShearModel(profile)
storm = StormOne()
result = storm.assess_tornado_potential(
    stability, lambda z: 303.15 - 0.0075 * z, shear_model)
print(storm.summary_tornado(result))

# Hurricane / tropical cyclone assessment
tc = TropicalCycloneModel(central_pressure_Pa=93500.0, ambient_pressure_Pa=101300.0,
                           radius_max_wind_m=40000.0, latitude_deg=25.0)
tc_result = storm.assess_tropical_cyclone(tc, fetch_m=50000.0, water_depth_m=15.0)
print(storm.summary_tropical_cyclone(tc_result))
```

---

*Co-developed with Claude (Anthropic), as an AI co-developer, consistent
with the ONE Ecosystem attribution convention.*
