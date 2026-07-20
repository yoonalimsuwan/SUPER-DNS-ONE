# FLOOD ONE — Flood Hydrology & Hydraulics Engineering Module

Part of the **ONE Ecosystem**.

## 1. Life-safety notice — read first

Flood hazard assessment directly informs evacuation zones, dam-failure
emergency action plans, and floodplain (insurance/zoning) maps. Every
correlation here is a standard, published hydrology/hydraulics
engineering method (SCS/NRCS Curve Number method, Manning's equation,
Muskingum routing, simplified parametric dam-breach regressions) —
**not** a replacement for a calibrated hydrologic/hydraulic model (e.g.
HEC-HMS, HEC-RAS, or a real 2D flood model), a site-specific Emergency
Action Plan, or professional review by a hydrologist/hydraulic engineer.
Do not use this as the sole basis for an evacuation decision, a dam
safety determination, or a floodplain regulatory map.

## 2. Scope

| Class | Covers |
|---|---|
| `RainfallModel` | Design-storm intensity-duration-frequency (IDF) curve, alternating-block design hyetograph |
| `WatershedHydrology` | SCS/NRCS Curve Number rainfall-runoff, Kirpich time of concentration, Rational Method, SCS triangular unit hydrograph |
| `ChannelHydraulics` | Manning's equation (trapezoidal channels), normal depth, critical depth, Froude number / flow-regime classification |
| `FloodRouting` | Muskingum channel routing (storage-based hydrologic routing) |
| `DamBreachModel` | Simplified parametric dam-breach outflow hydrograph (broad-crested-weir breach growth) |
| `FloodplainModel` | HAND (Height Above Nearest Drainage) inundation extent, generic depth-damage assessment |
| `FloodOne` | Orchestrator: design storm → watershed runoff → channel routing/flow state |

**Not implemented:** full 2D/hydrodynamic flood-wave propagation
(shallow-water-equation solver), real dam-breach erosion physics
(NWS BREACH-class models), a full grain-size/land-cover-resolved
watershed model, coupling to `super_dns_one_v6_3.py` for a
CFD-resolved flood/debris-flow simulation, structure-type-specific
depth-damage curves (e.g. FEMA/USACE HAZUS curves).

## 3. Why this module validated more cleanly than volcano_one.py

Several of this module's core relationships are **definitions with
tabulated coefficients** (Manning's equation's 2/3 and 1/2 exponents,
the Froude-number definition) rather than fitted empirical regressions
with an uncertain leading constant — the class of error that caused
real, caught-before-shipping bugs in `volcano_one.py`. Where this module
does rely on a fitted empirical relationship with real calibration
uncertainty (chiefly `DamBreachModel`'s breach-width/formation-time
regressions), that uncertainty is flagged explicitly in the docstrings,
following the same practice established after `volcano_one.py`'s
development.

## 4. Bug found and fixed during this module's own development

`RainfallModel.design_hyetograph()`'s alternating-block arrangement
raised an `IndexError` (an array index ran past the end of the block
array) in an earlier version, caught by this module's own validation
suite before shipping. Fixed by building the full block-index ordering
up front (a single pass generating a valid permutation of block
positions) rather than advancing two independent left/right pointers
with ad hoc boundary-case branches, which had a real edge case where
neither branch's guard condition held.

## 5. Explicitly flagged, unverified constants

- `DamBreachModel.breach_parameters()`'s final-breach-width and
  breach-formation-time regressions are illustrative forms in the
  spirit of Froehlich (1995) / MacDonald & Langridge-Monopolis (1984)
  type relationships, not verified reproductions of a specific published
  source — verify against current literature (e.g. Froehlich's 2016
  update) before real dam-safety use.
- `RainfallModel`'s default IDF parameters (a, b, c) are generic
  illustrative values, not parameters for any real gauge station —
  always use actual local IDF data (e.g. NOAA Atlas 14 in the US, or the
  relevant national meteorological service) for real work.
- `FloodplainModel.damage_fraction_generic()` is an illustrative smooth
  saturating curve, not a structure-type-specific published
  depth-damage function (e.g. FEMA/USACE HAZUS curves) — substitute a
  real curve for the structure type in question for real damage
  estimates.

## 6. Validation summary

| Check | Result |
|---|---|
| IDF intensity decreases with storm duration | Verified |
| SCS-CN: higher CN → more runoff at same rainfall | Verified |
| SCS-CN: runoff never exceeds rainfall | Verified |
| SCS-CN: runoff increases monotonically with rainfall depth | Verified |
| Kirpich Tc: plausible order of magnitude for a multi-km watershed | Verified |
| Kirpich Tc: steeper slope → shorter time of concentration | Verified |
| Manning's equation: matches an independent hand calculation | Verified **exactly** (<1e-6 relative error) |
| Manning's equation: discharge increases monotonically with depth | Verified |
| Normal-depth solver inverts Manning's equation accurately | Verified (<0.01 m³/s residual) |
| Critical-depth solver: Froude number at critical depth | Verified **≈1.0000** |
| Shallow depth at fixed discharge correctly classified supercritical | Verified |
| Muskingum routing: peak attenuated, not amplified | Verified |
| Muskingum routing: approximate volume conservation over a full event | Verified (<15% relative difference) |
| Dam breach: peak outflow order of magnitude vs. a real dam failure | Verified (Teton Dam 1976, ≈93 m: cited range 40,000–65,000 m³/s; this module's 50 m-dam estimate: ≈46,000 m³/s) |
| HAND inundation mask: correctly separates low/high terrain cells | Verified |
| Depth-damage curve: monotonic, zero at zero depth | Verified |

## 7. Usage example

```python
from flood_one import WatershedHydrology, ChannelHydraulics, FloodOne

watershed = WatershedHydrology(curve_number=75, area_km2=25.0,
                                flow_length_m=8000, avg_slope=0.015)
channel = ChannelHydraulics(bottom_width_m=15.0, side_slope_h_per_v=2.0,
                             manning_n=0.030, channel_slope=0.0015)

flood = FloodOne(watershed, channel)
results = flood.run_design_storm("100-year", duration_min=120,
                                  rainfall_depth_mm=150.0)
print(flood.summary(results))
```

---

*Co-developed with Claude (Anthropic), as an AI co-developer, consistent
with the ONE Ecosystem attribution convention.*
