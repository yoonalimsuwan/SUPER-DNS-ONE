# Developer  : PAI , Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT

import numpy as np
from flood_one import (
    RainfallModel, WatershedHydrology, ChannelHydraulics, FloodRouting,
    DamBreachModel, DamGeometry, FloodplainModel, FloodOne, G_ACCEL,
)

print("=" * 70)
print("TEST 1: RainfallModel — IDF curve monotonicity")
print("=" * 70)
rain = RainfallModel()
durations = np.array([5, 15, 30, 60, 120])
intensities = rain.intensity_mm_hr(durations)
print("Durations (min):", durations, " Intensities (mm/hr):", np.round(intensities, 1))
assert np.all(np.diff(intensities) < 0), "intensity should decrease with duration"
print("PASS: intensity decreases with duration (physically correct)")

hyeto = rain.design_hyetograph(60, dt_min=5)
print(f"Hyetograph total depth: {hyeto['total_depth_mm']:.1f} mm over 60min")
assert hyeto["total_depth_mm"] > 0
assert len(hyeto["time_min"]) == 12
print("PASS\n")

print("=" * 70)
print("TEST 2: WatershedHydrology — SCS-CN runoff, sanity vs known CN behavior")
print("=" * 70)
# Two watersheds: low CN (pervious, e.g. forest) vs high CN (impervious, urban)
ws_low = WatershedHydrology(curve_number=55, area_km2=10.0, flow_length_m=3000, avg_slope=0.02)
ws_high = WatershedHydrology(curve_number=95, area_km2=10.0, flow_length_m=3000, avg_slope=0.02)

P = np.array([25.0, 50.0, 100.0])  # mm rainfall
Q_low = ws_low.runoff_depth_mm(P)
Q_high = ws_high.runoff_depth_mm(P)
print("Rainfall (mm):", P)
print("Runoff, low CN=55:", np.round(Q_low, 2))
print("Runoff, high CN=95:", np.round(Q_high, 2))
assert np.all(Q_high > Q_low), "higher CN (more impervious) should produce more runoff for same rainfall"
assert np.all(Q_low <= P), "runoff cannot exceed rainfall"
assert np.all(Q_high <= P), "runoff cannot exceed rainfall"
print("PASS: higher CN gives more runoff, runoff never exceeds rainfall")

# Runoff should increase monotonically with rainfall depth
assert np.all(np.diff(Q_low) > 0)
print("PASS: runoff increases monotonically with rainfall\n")

print("=" * 70)
print("TEST 3: Time of concentration (Kirpich) — sanity check")
print("=" * 70)
tc = ws_low.time_of_concentration_min()
print(f"Tc for 3000m flow length, 2% slope: {tc:.1f} min ({tc/60:.2f} hr)")
# Sanity: for a few-km watershed with moderate slope, Tc should be
# on the order of tens of minutes to a couple hours -- not seconds, not days
assert 5 < tc < 500, "Tc should be in a physically plausible range for this watershed size"
print("PASS: Tc in plausible range")

# Steeper slope -> faster concentration (shorter Tc)
ws_steep = WatershedHydrology(curve_number=55, area_km2=10.0, flow_length_m=3000, avg_slope=0.10)
tc_steep = ws_steep.time_of_concentration_min()
print(f"Tc at 10% slope: {tc_steep:.1f} min (should be less than at 2% slope)")
assert tc_steep < tc, "steeper slope should give shorter time of concentration"
print("PASS: steeper slope correctly reduces Tc\n")

print("=" * 70)
print("TEST 4: SCS triangular unit hydrograph")
print("=" * 70)
uh = ws_low.scs_triangular_unit_hydrograph(rainfall_excess_mm=30.0)
print(uh)
assert uh["peak_discharge_m3_s"] > 0
assert uh["base_time_hr"] > uh["time_to_peak_hr"], "base time must exceed time to peak"
print("PASS\n")

print("=" * 70)
print("TEST 5: ChannelHydraulics — Manning's equation")
print("=" * 70)
channel = ChannelHydraulics(bottom_width_m=10.0, side_slope_h_per_v=2.0,
                             manning_n=0.030, channel_slope=0.001)
depths = np.array([0.5, 1.0, 2.0, 3.0])
Q = channel.discharge(depths)
print("Depths (m):", depths, " Discharge (m^3/s):", np.round(Q, 2))
assert np.all(np.diff(Q) > 0), "discharge should increase monotonically with depth"
print("PASS: discharge increases monotonically with depth")

# Verify Manning's equation directly against hand calculation at one depth
y_test = 2.0
A = (10.0 + 2.0*y_test)*y_test
P_wet = 10.0 + 2*y_test*math.sqrt(1+2.0**2) if False else 10.0 + 2*y_test*np.sqrt(1+2.0**2)
R = A/P_wet
Q_hand = (1/0.030)*A*R**(2/3)*0.001**0.5
Q_model = channel.discharge(np.array([y_test]))[0]
print(f"Hand calc Q at y=2m: {Q_hand:.4f}, model Q: {Q_model:.4f}")
assert abs(Q_hand - Q_model) < 1e-6, "Manning's equation implementation must match hand calculation exactly"
print("PASS: Manning's equation matches hand calculation exactly\n")

print("=" * 70)
print("TEST 6: Normal depth & critical depth solvers")
print("=" * 70)
Q_target = 50.0
y_n = channel.normal_depth(Q_target)
Q_check = channel.discharge(np.array([y_n]))[0]
print(f"Normal depth for Q={Q_target}: y_n={y_n:.4f}m, Q(y_n)={Q_check:.4f}")
assert abs(Q_check - Q_target) < 0.01, "normal depth solver should invert Manning's equation accurately"
print("PASS: normal depth solver accurate")

y_c = channel.critical_depth(Q_target)
Fr_c = channel.froude_number(y_c, Q_target)
print(f"Critical depth: y_c={y_c:.4f}m, Froude at y_c={Fr_c:.4f} (should be ~1.0)")
assert abs(Fr_c - 1.0) < 0.05, "Froude number at critical depth should be ~1.0"
print("PASS: critical depth solver gives Froude~1\n")

print("=" * 70)
print("TEST 7: Froude number / flow regime classification")
print("=" * 70)
# Deep slow flow -> subcritical; shallow fast flow -> supercritical
regime_deep = channel.flow_regime(y_n * 3, Q_target)   # much deeper than normal for same Q is unphysical for uniform flow, but tests the classifier directly
Fr_shallow = channel.froude_number(0.1, Q_target)
print(f"Froude at very shallow depth (0.1m) for Q={Q_target}: {Fr_shallow:.2f} (should be supercritical, Fr>1)")
assert Fr_shallow > 1.0, "very shallow depth at fixed discharge should be supercritical"
print("PASS: shallow flow correctly classified as supercritical\n")

print("=" * 70)
print("TEST 8: FloodRouting — Muskingum, mass conservation & attenuation")
print("=" * 70)
router = FloodRouting(K_hr=2.0, X=0.2)
t = np.arange(0, 48, 1.0)
inflow = 10.0 + 90.0 * np.exp(-0.5 * ((t - 12) / 4) ** 2)   # gaussian flood wave
outflow = router.route(inflow, dt_hr=1.0)
print(f"Peak inflow: {inflow.max():.1f}, peak outflow: {outflow.max():.1f}")
assert outflow.max() <= inflow.max() * 1.01, "routed peak should not exceed inflow peak (attenuation, not amplification)"
assert outflow.max() > inflow.min(), "outflow should show a real flood wave response"

# Approximate mass conservation (storage-routed, should conserve volume up to
# initial/final storage imbalance -- check total volumes are close)
vol_in = np.trapezoid(inflow, t)
vol_out = np.trapezoid(outflow, t)
rel_diff = abs(vol_in - vol_out) / vol_in
print(f"Inflow volume: {vol_in:.1f}, outflow volume: {vol_out:.1f}, rel diff: {rel_diff*100:.2f}%")
assert rel_diff < 0.15, "Muskingum routing should approximately conserve volume over a full event"
print("PASS: peak attenuated, volume approximately conserved")

stab = router.stability_check(dt_hr=1.0)
print("Stability check:", stab)
print("PASS\n")

print("=" * 70)
print("TEST 9: DamBreachModel")
print("=" * 70)
dam = DamGeometry(height_m=50.0, crest_length_m=300.0, reservoir_surface_area_km2=5.0, dam_type="earthen")
breach_model = DamBreachModel(dam)
params = breach_model.breach_parameters()
print("Breach params:", params)
assert params["final_breach_width_m"] > 0
assert params["breach_formation_time_hr"] > 0

hydro = breach_model.breach_outflow_hydrograph(reservoir_head_m=45.0)
print(f"Peak breach outflow: {hydro['peak_outflow_m3_s']:.0f} m^3/s")
assert hydro["peak_outflow_m3_s"] > 0
assert np.all(np.diff(hydro["outflow_m3_s"]) >= -1e-6), "triangular rise hydrograph should be non-decreasing"

# Sanity: for a real ~50m dam, peak breach outflow is typically in the
# thousands to tens of thousands of m^3/s range (order-of-magnitude check
# against well-known historical dam failures, e.g. Teton Dam 1976, ~93m
# high, peak outflow was on the order of 40,000-65,000 m^3/s)
print(f"Order-of-magnitude check: {hydro['peak_outflow_m3_s']:.0f} m^3/s for a 50m dam "
      f"(reference: Teton Dam ~93m -> ~40,000-65,000 m^3/s)")
assert 100 < hydro["peak_outflow_m3_s"] < 200000, "should be a plausible order of magnitude for a dam this size"
print("PASS\n")

print("=" * 70)
print("TEST 10: FloodplainModel")
print("=" * 70)
elev = np.array([[10, 12, 15], [9, 11, 14], [8, 10, 13]], dtype=float)
mask = FloodplainModel.inundation_extent_hand(elev, drainage_elevation_m=8.0, water_surface_elevation_m=11.0)
print("Elevation grid:\n", elev)
print("Inundation mask (water surface=11m):\n", mask)
assert mask[2, 0] == True, "lowest cell (elev=8) should be inundated at water surface 11m"
assert mask[0, 2] == False, "highest cell (elev=15) should not be inundated at water surface 11m"
print("PASS: HAND-based inundation mask correctly identifies low/high cells")

depth = FloodplainModel.inundation_depth(elev, water_surface_elevation_m=11.0)
print("Depth grid:\n", depth)
assert np.all(depth >= 0)
print("PASS")

damage = FloodplainModel.estimate_damage(np.array([0, 1, 3, 6]), np.array([200000]*4))
print("Damage estimate at depths [0,1,3,6]m, value=200000:", damage)
assert np.all(np.diff(damage) > 0), "damage should increase monotonically with depth"
assert damage[0] < 1e-6, "zero depth should give ~zero damage"
print("PASS: damage curve monotonic, zero at zero depth\n")

print("=" * 70)
print("TEST 11: FloodOne end-to-end orchestration")
print("=" * 70)
flood = FloodOne(ws_low, channel)
results = flood.run_design_storm("100-year", duration_min=60, rainfall_depth_mm=120.0)
print(flood.summary(results))
assert results["peak_discharge_m3_s"] > 0
assert results["normal_depth_m"] > 0
print("\nPASS\n")

print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
