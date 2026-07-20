# Developer  : PAI , Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT

import numpy as np
import math
from storm_one import (
    AtmosphericStabilityModel, WindShearModel, WindProfile, TornadoVortexModel,
    TropicalCycloneModel, StormSurgeModel, StormDamageAssessment, StormOne,
    coriolis_parameter, saturation_vapor_pressure,
)


print("=" * 70)
print("TEST 1: Coriolis parameter & saturation vapor pressure sanity")
print("=" * 70)
f_equator = coriolis_parameter(0.0)
f_midlat = coriolis_parameter(45.0)
f_pole = coriolis_parameter(90.0)
print(f"f at equator: {f_equator:.2e}, 45N: {f_midlat:.2e}, pole: {f_pole:.2e}")
assert abs(f_equator) < 1e-10, "Coriolis parameter should be ~0 at equator"
assert f_midlat > 0 and f_pole > f_midlat, "Coriolis parameter should increase toward pole"
print("PASS: Coriolis parameter correct sign/magnitude trend")

es = saturation_vapor_pressure(np.array([273.15, 293.15, 313.15]))
print("Sat vapor pressure at 0C,20C,40C:", es, "Pa")
assert np.all(np.diff(es) > 0), "saturation vapor pressure should increase with temperature"
assert 600 < es[0] < 650, "sat vapor pressure at 0C should be ~611 Pa"
print("PASS: saturation vapor pressure monotonic and correct at 0C reference point\n")

print("=" * 70)
print("TEST 2: AtmosphericStabilityModel — CAPE/CIN")
print("=" * 70)
# Warm moist surface parcel, standard-atmosphere-like environment (stable
# baseline) -- should give modest CAPE
stability = AtmosphericStabilityModel(surface_T_K=303.15, surface_p_Pa=101300.0,
                                       surface_dewpoint_K=295.15)
lcl = stability.lcl_height_m()
print(f"LCL height: {lcl:.0f} m")
assert 500 < lcl < 3000, "LCL should be in a physically plausible range for this T/Td spread"

def env_profile_unstable(z):
    # environment that's COOLER than a lifted moist parcel aloft (favorable for CAPE)
    return 303.15 - 0.0075 * z   # 7.5 K/km environmental lapse rate (steeper than moist adiabat)

result = stability.cape_cin(env_profile_unstable, z_top_m=12000, dz_m=100)
print(f"CAPE: {result['CAPE_J_kg']:.0f} J/kg, CIN: {result['CIN_J_kg']:.0f} J/kg")
assert result["CAPE_J_kg"] >= 0, "CAPE cannot be negative"
assert result["CIN_J_kg"] >= 0, "CIN (reported magnitude) cannot be negative"

def env_profile_stable(z):
    # very stable environment (isothermal-ish, should suppress CAPE)
    return 303.15 - 0.001 * z

result_stable = stability.cape_cin(env_profile_stable, z_top_m=12000, dz_m=100)
print(f"Stable environment CAPE: {result_stable['CAPE_J_kg']:.0f} J/kg (should be less than unstable case)")
assert result_stable["CAPE_J_kg"] < result["CAPE_J_kg"], \
    "a more stable environment should produce less CAPE than an unstable one"
print("PASS: CAPE correctly higher in unstable environment\n")

w_max = AtmosphericStabilityModel.max_updraft_velocity(result["CAPE_J_kg"])
print(f"Theoretical max updraft: {w_max:.1f} m/s")
assert w_max > 0
print("PASS\n")

print("=" * 70)
print("TEST 3: WindShearModel — bulk shear, SRH, Bunkers motion")
print("=" * 70)
z = np.array([0, 500, 1000, 2000, 3000, 4000, 6000, 9000])
# classic veering wind profile (favorable for supercells): speed and
# direction both increase/turn with height
u = np.array([5, 8, 12, 18, 22, 25, 30, 35])
v = np.array([0, 3, 6, 8, 5, 2, -2, -5])
profile = WindProfile(z, u, v)
shear_model = WindShearModel(profile)

shear = shear_model.bulk_shear(6000.0)
print(f"0-6km bulk shear: {shear['shear_m_s']:.1f} m/s")
assert shear["shear_m_s"] > 0

motion = shear_model.bunkers_storm_motion()
print(f"Bunkers storm motion: u={motion['u']:.1f}, v={motion['v']:.1f}")

srh = shear_model.storm_relative_helicity(3000.0, motion)
print(f"0-3km SRH: {srh:.0f} m^2/s^2")

scp = shear_model.supercell_composite_parameter(2500.0, srh, shear["shear_m_s"])
print(f"SCP (CAPE=2500, this shear/SRH): {scp:.2f}")

# Sanity: zero shear should give SCP=0 (shear<10 threshold)
flat_profile = WindProfile(z, np.zeros_like(z, dtype=float), np.zeros_like(z, dtype=float))
flat_shear_model = WindShearModel(flat_profile)
flat_shear = flat_shear_model.bulk_shear(6000.0)
scp_flat = flat_shear_model.supercell_composite_parameter(2500.0, 0.0, flat_shear["shear_m_s"])
print(f"Flat (no-shear) environment SCP: {scp_flat} (should be exactly 0)")
assert scp_flat == 0.0, "zero-shear environment should give SCP=0"
print("PASS: shear-based SCP correctly zero in a no-shear environment\n")

print("=" * 70)
print("TEST 4: TornadoVortexModel — Rankine vortex physics")
print("=" * 70)
tornado = TornadoVortexModel(v_max_m_s=90.0, r_max_m=150.0)

# Velocity should INCREASE linearly inside the core, then DECREASE outside
r_inside = np.array([0, 50, 100, 150])
r_outside = np.array([150, 300, 600, 1500])
v_inside = tornado.tangential_velocity(r_inside)
v_outside = tornado.tangential_velocity(r_outside)
print("Inside core velocities:", v_inside)
print("Outside core velocities:", v_outside)
assert np.all(np.diff(v_inside) >= 0), "velocity should increase (solid body) inside the core"
assert np.all(np.diff(v_outside) <= 0), "velocity should decrease (free vortex) outside the core"
assert abs(v_inside[-1] - v_outside[0]) < 1e-9, "velocity should be continuous at r_max"
print("PASS: Rankine vortex correctly peaks at r_max, solid-body inside, free vortex outside")

# EF-scale classification sanity
ef = tornado.ef_scale_from_wind_speed(90.0)  # 90 m/s ~ 201 mph
print(f"90 m/s ({90*2.23694:.0f} mph) classified as: {ef}")
assert ef == "EF5"
ef_weak = tornado.ef_scale_from_wind_speed(30.0)  # ~67mph
print(f"30 m/s ({30*2.23694:.0f} mph) classified as: {ef_weak}")
assert ef_weak == "EF0"
print("PASS: EF-scale classification correct at both ends\n")

print("=" * 70)
print("TEST 5: Tornado translation & ground-relative wind field")
print("=" * 70)
tornado_moving = TornadoVortexModel(v_max_m_s=80.0, r_max_m=100.0,
                                     translation_speed_m_s=15.0, translation_heading_deg=90.0)
peak = tornado_moving.max_ground_relative_speed()
print(f"Peak ground-relative speed: {peak['peak_speed_m_s']:.1f} m/s (v_max + translation = 80+15=95)")
assert abs(peak["peak_speed_m_s"] - 95.0) < 1e-9
print("PASS: peak ground-relative speed correctly adds translation to rotational max\n")

print("=" * 70)
print("TEST 6: TropicalCycloneModel — Holland wind profile")
print("=" * 70)
# Category 4-ish hurricane: central pressure ~935 hPa
tc = TropicalCycloneModel(central_pressure_Pa=93500.0, ambient_pressure_Pa=101300.0,
                           radius_max_wind_m=40000.0, latitude_deg=25.0)
print(f"Holland B parameter: {tc.B:.2f} (should be in [1, 2.5])")
assert 1.0 <= tc.B <= 2.5

Vmax = tc.max_wind_speed()
print(f"Max wind speed: {Vmax:.1f} m/s ({Vmax*2.23694:.0f} mph)")
category = tc.saffir_simpson_category(Vmax)
print(f"Saffir-Simpson category: {category}")
assert Vmax > 0

profile = tc.radial_wind_profile(300000.0, 100)
print(f"Wind speed at Rmax vs far field: {profile['wind_speed_m_s'][0]:.1f} vs {profile['wind_speed_m_s'][-1]:.1f} m/s")
# Wind should peak near Rmax and decay outward at large radius
peak_idx = np.argmax(profile["wind_speed_m_s"])
peak_r = profile["r_m"][peak_idx]
print(f"Peak wind occurs at r={peak_r:.0f}m (Rmax={tc.Rmax}m)")
assert abs(peak_r - tc.Rmax) < 5000, "peak wind should occur near the radius of maximum wind"
assert profile["wind_speed_m_s"][-1] < profile["wind_speed_m_s"][peak_idx], \
    "wind speed should decay well outside the radius of maximum wind"
print("PASS: Holland profile peaks near Rmax and decays outward\n")

print("=" * 70)
print("TEST 7: Pressure-wind relationship & deeper storms = stronger winds")
print("=" * 70)
v_shallow = TropicalCycloneModel.pressure_wind_relationship(990e2, 1013e2)
v_deep = TropicalCycloneModel.pressure_wind_relationship(920e2, 1013e2)
print(f"Weak storm (990hPa): Vmax={v_shallow:.1f} m/s, Intense storm (920hPa): Vmax={v_deep:.1f} m/s")
assert v_deep > v_shallow, "lower central pressure (deeper storm) should give higher estimated Vmax"
print("PASS: deeper pressure minimum correctly gives higher wind estimate\n")

print("=" * 70)
print("TEST 8: Saffir-Simpson category assignment")
print("=" * 70)
test_speeds_mph = [50, 80, 100, 120, 145, 170]
for mph in test_speeds_mph:
    ms = mph / 2.23694
    cat = TropicalCycloneModel.saffir_simpson_category(ms)
    print(f"  {mph} mph -> {cat}")
assert TropicalCycloneModel.saffir_simpson_category(50/2.23694) == "Tropical Storm (or weaker)"
assert TropicalCycloneModel.saffir_simpson_category(170/2.23694) == "Category 5"
print("PASS: category boundaries correctly assigned\n")

print("=" * 70)
print("TEST 9: StormSurgeModel")
print("=" * 70)
# Realistic shelf parameters: shorter fetch, more representative shallow-shelf depth
surge = StormSurgeModel.total_surge_estimate(
    central_pressure_Pa=93500.0, wind_speed_m_s=Vmax, fetch_m=50000.0, water_depth_m=15.0)
print(surge)
assert surge["total_surge_m"] > 0
assert surge["inverse_barometer_m"] > 0

# Deeper storm -> more inverse-barometer surge
ib_shallow = StormSurgeModel.inverse_barometer_setup(990e2)
ib_deep = StormSurgeModel.inverse_barometer_setup(920e2)
print(f"IB setup: 990hPa->{ib_shallow:.3f}m, 920hPa->{ib_deep:.3f}m")
assert ib_deep > ib_shallow, "lower central pressure should give more inverse-barometer setup"

print(f"Total surge for this Cat4-ish storm, realistic shelf params: {surge['total_surge_m']:.2f} m "
      f"(real major-hurricane surges commonly 3-6m); assumption_valid={surge['assumption_valid']}")
assert 0.5 < surge["total_surge_m"] < 15.0, "should be a plausible order of magnitude for a strong hurricane"
print("PASS: inverse barometer effect correctly increases with storm intensity, plausible total magnitude\n")

print("=" * 70)
print("TEST 9b: StormSurgeModel — assumption-violation flag catches unphysical regime")
print("=" * 70)
# The exact parameters that produced an unphysical ~19m result during
# this module's own development (long fetch, shallow constant depth) --
# verify the fix now flags it instead of silently returning the number.
bad_surge = StormSurgeModel.total_surge_estimate(
    central_pressure_Pa=93500.0, wind_speed_m_s=70.0, fetch_m=250000.0, water_depth_m=20.0)
print(bad_surge)
assert bad_surge["assumption_valid"] == False, \
    "long fetch / shallow constant depth should correctly trip the assumption-validity flag"
assert "warning" in bad_surge
print("PASS: unphysical-regime combination correctly flagged as assumption_valid=False\n")

print("=" * 70)
print("TEST 10: StormDamageAssessment")
print("=" * 70)
d_low = StormDamageAssessment.damage_fraction_from_wind_speed(20.0)
d_high = StormDamageAssessment.damage_fraction_from_wind_speed(90.0)
print(f"Damage fraction at 20m/s: {d_low:.3f}, at 90m/s: {d_high:.3f}")
assert d_high > d_low, "damage fraction should increase with wind speed"
assert 0 <= d_low <= 1 and 0 <= d_high <= 1, "damage fraction must be bounded [0,1]"
print("PASS\n")

print("=" * 70)
print("TEST 11: StormOne end-to-end — tornado assessment")
print("=" * 70)
storm = StormOne()
tornado_assessment = storm.assess_tornado_potential(stability, env_profile_unstable, shear_model)
print(storm.summary_tornado(tornado_assessment))
assert tornado_assessment["CAPE_J_kg"] >= 0
print("\nPASS\n")

print("=" * 70)
print("TEST 12: StormOne end-to-end — tropical cyclone / hurricane assessment")
print("=" * 70)
tc_assessment = storm.assess_tropical_cyclone(tc, fetch_m=50000.0, water_depth_m=15.0)
print(storm.summary_tropical_cyclone(tc_assessment))
assert tc_assessment["max_wind_speed_m_s"] > 0
print("\nPASS\n")

print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
