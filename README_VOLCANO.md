# VOLCANO ONE — Volcanic Eruption Engineering Module

Part of the **ONE Ecosystem**. A new physical domain (no prior volcanology
work existed elsewhere in this ecosystem to build on before this module).

## 1. Life-safety notice — read first

Volcanic hazard assessment directly informs evacuation zones, exclusion
radii, and aviation ash advisories. Every correlation here is a
published, simplified volcanological/engineering approximation
(Morton-Taylor-Turner plume theory, Mastin et al. column-height scaling,
Malin & Sheridan energy-cone PDC runout) — **not** a replacement for a
real volcanological hazard assessment, an actual eruption-column model
(e.g. Plumeria, ATHAM, or a 3D multiphase conduit-to-atmosphere code),
or professional review by a volcanologist. **Several empirical constants
were found to be wrong during this module's own development and were
corrected before shipping — see §4.** Do not use this as the sole basis
for an evacuation decision, an ash-cloud aviation advisory, or a
hazard-zone map.

## 2. Scope

`volcano_one.py` covers magma-to-atmosphere eruption physics at an
engineering-screening level:

| Class | Covers |
|---|---|
| `MagmaRheology` | Melt viscosity (Vogel-Fulcher-Tammann form), crystal-content correction (Einstein-Roscoe) |
| `ConduitFlowModel` | Magma ascent, volatile (H₂O) exsolution/degassing, choking (isothermal-Mach) check |
| `FragmentationModel` | Gas-volume-fraction fragmentation criterion, fragmentation depth estimate |
| `EruptionColumnModel` | Buoyant plume rise (Morton-Taylor-Turner theory), independent Mastin-form empirical cross-check, column-collapse screening |
| `TephraTransportModel` | Particle terminal settling velocity (standard drag-coefficient regimes), simple single-particle ashfall footprint |
| `PyroclasticDensityCurrentModel` | Energy-cone (Malin & Sheridan) PDC runout |
| `VEIAssessment` | Volcanic Explosivity Index from erupted volume |
| `VolcanoOne` | Orchestrator: conduit → fragmentation → column → hazard footprint → VEI |

**Not implemented:** lava-flow rheology/routing, full multiphase
conduit-to-atmosphere CFD, real (non-radially-symmetric,
topography-aware) PDC modeling, tephra dispersal with a full grain-size
distribution and turbulent diffusion (Tephra2/Ash3d/HYSPLIT-class
models), any experimental/observational validation beyond the single
reference eruption used for calibration below.

## 3. Conceptual links to other ONE Ecosystem modules

- **Eruption column ↔ fire plume**: both are buoyant-plume theory
  (Morton-Taylor-Turner ≈ the Heskestad correlations in `fire_one.py`);
  the difference is a volcanic column rises into a *stratified*
  atmosphere (Brunt-Väisälä frequency sets the height scale), whereas
  `fire_one.py`'s Heskestad correlations assume a uniform ambient.
- **Vent mass injection ↔ pyrolysis wall BC**: conceptually the same
  category of problem as `super_dns_one_v6_3.py`'s `PyrolysisWallBC`
  (Stefan-flow surface mass injection), but volcanic vent velocities can
  be far higher (approaching or exceeding choking) and the erupted
  material is a particle-laden multiphase mixture, not a single-phase
  gas — the existing `PyrolysisWallBC` formulation was NOT reused
  as-is; a real DNS-coupled volcanic vent BC would need its own
  derivation.
- **Volcano-tectonic earthquakes**: `seismic_one.py` can analyze the
  resulting ground shaking in principle, but has no volcanic source
  mechanism (magma movement, resonance, etc.) of its own.

## 4. Bugs found and fixed during this module's own development

This was the first volcanology work in the ecosystem, so no prior
cross-checked constants existed to build on. Validation testing (in
`test_volcano_one.py`) caught real, significant calibration errors
before shipping:

| Issue | Symptom | Fix |
|---|---|---|
| MTT theory column-height constant | Column heights of 0.05–1.6 km even at supervolcano-scale mass eruption rate (should be tens of km) — wrong by roughly two orders of magnitude | Recalibrated the leading constant (`k`) against the Mount St. Helens 18 May 1980 eruption (MER≈1.4×10⁷ kg/s, observed column height ≈15–24 km) |
| Mastin-form empirical height constant | Overpredicted by ≈5× at the same reference eruption (105 km vs. observed 15–24 km) | Recalibrated the same way |
| Exit-velocity estimate | Produced ≈287,000 m/s for a small vent radius — unphysical (real eruptions: <1000 m/s) | Capped at the local choked/sonic gas velocity |

**Positive validation signal:** after recalibrating both column-height
formulas at a *single* reference point (Mount St. Helens 1980), the two
independently-derived formulas (MTT theory vs. Mastin-style power law)
agree closely across the *entire* tested mass-eruption-rate range
(10³–10⁹ kg/s) — e.g. 1.84 km vs. 2.00 km at the low end, 58.1 km vs.
56.0 km at the high end. Two different functional forms agreeing well
beyond their shared calibration point is a reassuring (though not
independently conclusive) consistency check.

## 5. Explicitly flagged, unverified constants

Marked in the module's own docstrings — check against current
literature before quantitative use:

- `MagmaRheology`'s VFT viscosity parameterization (A/B/C vs. SiO₂ wt%)
  reproduces the correct *qualitative* trend and order of magnitude, not
  a verified fit to a specific published model (e.g. Giordano-Russell-
  Dingwell 2008).
- `ConduitFlowModel.solubility_water()`'s Henry's-law-like constant
  reproduces plausible wt% H₂O solubility at typical pressures, not a
  verified fit to a specific published solubility law (e.g. Liu et al.
  2005).
- `PyroclasticDensityCurrentModel`'s default Heim coefficient (H/L=0.2)
  is representative of commonly cited PDC values but varies significantly
  by current type (dilute vs. concentrated) and should be set from
  site/eruption-specific data for real hazard-zone mapping.

## 6. Validation summary

| Check | Result |
|---|---|
| Atmosphere profile: T, ρ decrease with height | Verified |
| Brunt-Väisälä frequency in typical tropospheric range | Verified (≈0.0105 s⁻¹) |
| Melt viscosity: basalt < andesite < dacite < rhyolite | Verified, monotonic |
| Melt viscosity decreases with increasing temperature | Verified |
| Einstein-Roscoe crystal correction monotonic in crystal fraction | Verified |
| Gas volume fraction increases monotonically during ascent (degassing) | Verified, bounded in [0,1] |
| Mach number increases with mass flux (conduit choking check) | Verified |
| Fragmentation threshold logic | Verified |
| Column height monotonic in mass eruption rate (both formulas) | Verified |
| Column height plausible order of magnitude vs. a real eruption | Verified (post-recalibration) |
| Larger vent radius → lower exit velocity (more collapse-prone) | Verified |
| Settling velocity increases with particle size | Verified |
| PDC runout scales linearly with collapse height (energy-cone) | Verified exactly |
| VEI threshold assignment | Verified |

## 7. Usage example

```python
from volcano_one import VolcanoOne, COMMON_MAGMAS

volcano = VolcanoOne(COMMON_MAGMAS["dacite"], conduit_radius_m=30.0,
                      conduit_length_m=6000.0, water_content_wt_pct=4.5)
results = volcano.run(mass_eruption_rate_kg_s=5e7, T_erupted_K=1173.0,
                       duration_s=3600 * 6)
print(volcano.summary(results))
```

---

*Co-developed with Claude (Anthropic), as an AI co-developer, consistent
with the ONE Ecosystem attribution convention.*
