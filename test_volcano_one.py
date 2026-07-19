"""
Validation tests for volcano_one.py. Since this is a brand-new physical
domain with no prior cross-checked constants in this codebase, tests
focus heavily on dimensional consistency, monotonicity, and known
qualitative trends -- NOT precise agreement with specific published
numbers (several constants are explicitly flagged in the module as
needing literature verification before quantitative use).
"""
import numpy as np
from volcano_one import (
    MagmaRheology, COMMON_MAGMAS, ConduitFlowModel, FragmentationModel,
    EruptionColumnModel, TephraTransportModel, PyroclasticDensityCurrentModel,
    VEIAssessment, VolcanoOne, brunt_vaisala_frequency, atmosphere_profile,
)

print("=" * 70)
print("TEST 1: Atmosphere profile & Brunt-Vaisala frequency")
print("=" * 70)
atm = atmosphere_profile(np.array([0, 5000, 10000]))
print("T(z):", atm["T"], " rho(z):", atm["rho"])
assert np.all(np.diff(atm["T"]) < 0), "temperature should decrease with height"
assert np.all(np.diff(atm["rho"]) < 0), "density should decrease with height"
N = brunt_vaisala_frequency()
print(f"Brunt-Vaisala frequency: {N:.5f} 1/s (typical tropospheric range ~0.01-0.02)")
assert 0.005 < N < 0.03, "N should be in the physically typical tropospheric range"
print("PASS\n")

print("=" * 70)
print("TEST 2: MagmaRheology — viscosity trend basalt < andesite < dacite < rhyolite")
print("=" * 70)
T_test = 1273.0  # 1000C, typical eruptive temperature
viscosities = {}
for name, comp in COMMON_MAGMAS.items():
    rheo = MagmaRheology(comp)
    eta = rheo.melt_viscosity(np.array([T_test]))[0]
    viscosities[name] = eta
    print(f"  {name} (SiO2={comp.sio2_wt_pct}%): eta = {eta:.3e} Pa.s")

assert viscosities["basalt"] < viscosities["andesite"] < viscosities["dacite"] < viscosities["rhyolite"], \
    "viscosity should increase monotonically with silica content"
print("PASS: viscosity correctly increases with silica content (basalt->rhyolite)")

# Temperature dependence: viscosity should DECREASE as T increases (hotter = less viscous)
rheo_rhy = MagmaRheology(COMMON_MAGMAS["rhyolite"])
eta_hot = rheo_rhy.melt_viscosity(np.array([1200.0]))[0]
eta_cold = rheo_rhy.melt_viscosity(np.array([900.0]))[0]
print(f"Rhyolite viscosity at 1200K: {eta_hot:.3e}, at 900K: {eta_cold:.3e}")
assert eta_cold > eta_hot, "viscosity should increase as temperature decreases"
print("PASS: viscosity correctly decreases with increasing temperature")

# Crystal content should increase bulk viscosity monotonically
eta_bulk_0 = rheo_rhy.bulk_viscosity(np.array([1200.0]), phi_crystals=0.0)[0]
eta_bulk_3 = rheo_rhy.bulk_viscosity(np.array([1200.0]), phi_crystals=0.3)[0]
eta_bulk_5 = rheo_rhy.bulk_viscosity(np.array([1200.0]), phi_crystals=0.5)[0]
print(f"Bulk viscosity at phi=0,0.3,0.5: {eta_bulk_0:.2e}, {eta_bulk_3:.2e}, {eta_bulk_5:.2e}")
assert eta_bulk_0 < eta_bulk_3 < eta_bulk_5, "crystal content should monotonically increase viscosity"
print("PASS: Einstein-Roscoe crystal correction correctly monotonic\n")

print("=" * 70)
print("TEST 3: ConduitFlowModel — degassing and gas volume fraction")
print("=" * 70)
comp = COMMON_MAGMAS["dacite"]
rheo = MagmaRheology(comp)
conduit = ConduitFlowModel(conduit_radius_m=25.0, conduit_length_m=5000.0,
                            magma_rheology=rheo, water_content_wt_pct=4.0)

# Gas fraction should increase as pressure decreases (degassing during ascent)
phi_deep = conduit.gas_volume_fraction(200e6, 1173.0, comp.density_melt)
phi_mid  = conduit.gas_volume_fraction(50e6, 1173.0, comp.density_melt)
phi_shallow = conduit.gas_volume_fraction(1e6, 1173.0, comp.density_melt)
phi_vent = conduit.gas_volume_fraction(101325.0, 1173.0, comp.density_melt)
print(f"phi_gas at 200MPa: {phi_deep:.4f}, 50MPa: {phi_mid:.4f}, 1MPa: {phi_shallow:.4f}, vent: {phi_vent:.4f}")
assert phi_deep <= phi_mid <= phi_shallow <= phi_vent, \
    "gas volume fraction should monotonically increase as magma ascends (pressure drops)"
assert 0 <= phi_vent <= 1.0, "gas volume fraction must be physically bounded in [0,1]"
print("PASS: degassing trend correct, gas fraction bounded in [0,1]\n")

print("=" * 70)
print("TEST 4: ConduitFlowModel — choking check sanity")
print("=" * 70)
result_low = conduit.is_choked(1e4, 101325.0, 1173.0, comp.density_melt)
result_high = conduit.is_choked(1e7, 101325.0, 1173.0, comp.density_melt)
print(f"Low mass flux (1e4 kg/s): Mach={result_low['mach']:.3f}, choked={result_low['choked']}")
print(f"High mass flux (1e7 kg/s): Mach={result_high['mach']:.3f}, choked={result_high['choked']}")
assert result_high["mach"] > result_low["mach"], "higher mass flux should give higher Mach number"
print("PASS: Mach number correctly increases with mass flux\n")

print("=" * 70)
print("TEST 5: FragmentationModel")
print("=" * 70)
frag = FragmentationModel(gas_fraction_threshold=0.75)
assert frag.is_fragmented(0.8) == True
assert frag.is_fragmented(0.5) == False
print("PASS: fragmentation threshold logic correct")

depth = frag.fragmentation_depth(conduit, 1e6, 1173.0, comp.density_melt)
print(f"Fragmentation depth estimate: {depth} m" if depth else "No fragmentation found in search range")
if depth is not None:
    assert depth >= 0, "fragmentation depth must be non-negative"
print("PASS\n")

print("=" * 70)
print("TEST 6: EruptionColumnModel — height scaling monotonicity + cross-check")
print("=" * 70)
col = EruptionColumnModel()
mer_values = [1e3, 1e5, 1e7, 1e9]  # kg/s, spanning small to Plinian-scale
heights_mtt = [col.column_height(m) for m in mer_values]
heights_mastin = [col.mastin_height_scaling(m) for m in mer_values]
print("MER (kg/s):        ", mer_values)
print("H (MTT theory, km): ", [f"{h/1000:.2f}" for h in heights_mtt])
print("H (Mastin emp., km):", [f"{h/1000:.2f}" for h in heights_mastin])

assert all(heights_mtt[i] < heights_mtt[i+1] for i in range(len(heights_mtt)-1)), \
    "column height should increase monotonically with mass eruption rate"
assert all(heights_mastin[i] < heights_mastin[i+1] for i in range(len(heights_mastin)-1)), \
    "Mastin height should also increase monotonically"

# Sanity: known real eruptions -- Mount St. Helens 1980 (MER ~1.4e7 kg/s) had
# column height ~15-24 km; both formulas are now CALIBRATED against this
# exact reference point (see volcano_one.py's calibration notes -- the
# original memorized constants were off by 1-2 orders of magnitude and
# were caught and fixed here, not shipped). This checks the calibration
# was applied correctly and both formulas agree closely at their shared
# anchor point, plus stay in a plausible range at other scales.
h_msh_mtt = col.column_height(1.4e7) / 1000
h_msh_mastin = col.mastin_height_scaling(1.4e7) / 1000
print(f"\nMSH-1980-like MER (1.4e7 kg/s): H_mtt={h_msh_mtt:.1f}km, H_mastin={h_msh_mastin:.1f}km (observed: ~15-24km)")
assert 14 < h_msh_mtt < 26, "MTT formula should reproduce its MSH-1980 calibration point"
assert 14 < h_msh_mastin < 26, "Mastin-form formula should reproduce its MSH-1980 calibration point"
# At a much larger (supervolcano-scale) MER, height should still be plausible
# (real large Plinian/ultra-Plinian columns reach into the 40-55km range)
h_super = col.mastin_height_scaling(1e9) / 1000
print(f"Supervolcano-scale MER (1e9 kg/s): H_mastin={h_super:.1f}km")
assert 30 < h_super < 100, "should stay in a physically plausible stratospheric-injection range"
print("PASS: recalibrated formulas agree at anchor point and stay plausible across scales\n")

print("=" * 70)
print("TEST 7: Column collapse stability check")
print("=" * 70)
stab_small_vent = col.is_column_collapse_likely(1e8, vent_radius_m=20.0)
stab_large_vent = col.is_column_collapse_likely(1e8, vent_radius_m=500.0)
print("Small vent (20m):", stab_small_vent)
print("Large vent (500m):", stab_large_vent)
# Larger vent at same MER -> lower exit velocity -> more collapse-prone (physically expected)
assert stab_large_vent["exit_velocity_m_s"] < stab_small_vent["exit_velocity_m_s"], \
    "larger vent at same MER should give lower exit velocity"
print("PASS: larger vent radius correctly gives lower exit velocity (more collapse-prone)\n")

print("=" * 70)
print("TEST 8: TephraTransportModel — settling velocity trends")
print("=" * 70)
diameters = np.array([1e-5, 1e-4, 1e-3, 1e-2])  # 10um to 1cm
v_settle = TephraTransportModel.terminal_velocity(diameters, particle_density=2000.0)
print("Diameters (m):", diameters)
print("Settling velocities (m/s):", v_settle)
assert np.all(np.diff(v_settle) > 0), "larger particles should settle faster"
print("PASS: settling velocity correctly increases with particle size")

# Sanity check against well-known Stokes-regime approximate value:
# ~10um ash particle (density 2000) should settle around cm/s to a few cm/s range
assert 1e-4 < v_settle[0] < 0.1, "10um particle settling velocity should be in a physically plausible range"
print(f"10um particle: {v_settle[0]:.4f} m/s (physically plausible range)")

footprint = TephraTransportModel.simple_ashfall_radius(15000.0, 1e-4, wind_speed_m_s=15.0)
print("Ashfall footprint estimate (100um particle, 15km column, 15m/s wind):", footprint)
assert footprint["downwind_distance_m"] > 0
print("PASS\n")

print("=" * 70)
print("TEST 9: PyroclasticDensityCurrentModel — energy cone")
print("=" * 70)
pdc = PyroclasticDensityCurrentModel(heim_coefficient=0.2)
runout_1km = pdc.runout_distance(1000.0)
runout_2km = pdc.runout_distance(2000.0)
print(f"Runout from 1km collapse: {runout_1km/1000:.1f} km")
print(f"Runout from 2km collapse: {runout_2km/1000:.1f} km")
assert runout_2km == 2 * runout_1km, "runout should scale linearly with collapse height (energy-cone model)"
assert abs(runout_1km - 5000.0) < 1.0, "H/L=0.2 means L=H/0.2=5x H"
print("PASS: energy-cone linear scaling correct\n")

print("=" * 70)
print("TEST 10: VEIAssessment")
print("=" * 70)
assert VEIAssessment.vei_from_volume(1e-5) == 0
assert VEIAssessment.vei_from_volume(1e-3) == 2
assert VEIAssessment.vei_from_volume(1.0) == 5
assert VEIAssessment.vei_from_volume(1000.0) == 8
print("PASS: VEI thresholds correctly assigned")

vol = VEIAssessment.volume_from_mass_eruption_rate(1e7, 3600*10, bulk_density_kg_m3=1000.0)
print(f"Volume from MER=1e7 kg/s over 10hr: {vol:.4f} km^3, VEI={VEIAssessment.vei_from_volume(vol)}")
assert vol > 0
print("PASS\n")

print("=" * 70)
print("TEST 11: VolcanoOne end-to-end orchestration")
print("=" * 70)
volcano = VolcanoOne(COMMON_MAGMAS["dacite"], conduit_radius_m=30.0,
                      conduit_length_m=6000.0, water_content_wt_pct=4.5)
results = volcano.run(mass_eruption_rate_kg_s=5e7, T_erupted_K=1173.0, duration_s=3600*6)
print(volcano.summary(results))
assert results["VEI"] >= 0
assert results["column_height_m_mtt_theory"] > 0
assert results["column_height_m_mastin_empirical"] > 0
print("\nPASS\n")

print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print()
print("NOTE: Tests validate dimensional consistency, monotonicity, and")
print("order-of-magnitude plausibility against known real eruptions.")
print("Several constants are explicitly FLAGGED in volcano_one.py's")
print("docstrings as needing verification against current literature")
print("before any quantitative/operational use.")
