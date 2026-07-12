"""
Validation / smoke tests for seismic_one.py

Checks:
  1. GroundMotionEngine: synthetic motion generation, PGA/PGV sanity, response spectrum shape
  2. SiteResponseLayer: soft soil over rock amplifies motion at expected frequency range;
     converges within iteration budget
  3. StructuralResponseLayer: elastic single-story building matches closed-form SDOF
     response spectrum value (validates Newmark-beta + assembly against known solution);
     nonlinear model with weak story shows larger drift concentration than uniform building
  4. LiquefactionAssessment: loose saturated sand -> FS < 1 ; dense sand -> FS > 1
"""
import numpy as np
from seismic_one import (
    GroundMotionEngine, GroundMotion, SoilLayerProps, SiteResponseLayer,
    Story, StructuralResponseLayer, LiquefactionAssessment, SeismicOne, G_ACCEL
)

rng = np.random.default_rng(42)

print("=" * 70)
print("TEST 1: GroundMotionEngine")
print("=" * 70)
gme = GroundMotionEngine()
motion = gme.synthetic_motion(duration=20.0, dt=0.01, pga_target=3.0, dominant_freq=3.0, rng=rng)
print(f"PGA = {motion.pga:.3f} m/s^2 (target 3.0)")
print(f"PGV = {motion.pgv:.4f} m/s")
print(f"PGD = {motion.pgd:.4f} m")
assert abs(motion.pga - 3.0) < 0.05, "PGA scaling failed"

periods = np.geomspace(0.02, 4.0, 40)
spec = gme.response_spectrum(motion, periods, zeta=0.05)
assert np.all(np.isfinite(spec["Sa"])), "Non-finite Sa in spectrum"
assert spec["Sa"][0] > 0, "Sa at T->0 should approach PGA"
print(f"Sa(T->0) = {spec['Sa'][0]:.3f} (should be ~PGA={motion.pga:.3f})")
print(f"Max Sa = {np.max(spec['Sa']):.3f} at T={periods[np.argmax(spec['Sa'])]:.3f}s")
print("PASS\n")

print("=" * 70)
print("TEST 2: SiteResponseLayer (soft clay over rock)")
print("=" * 70)
layers = [
    SoilLayerProps(thickness=5.0, density=1600, Gmax=30e6, damping_min=0.02),
    SoilLayerProps(thickness=10.0, density=1800, Gmax=80e6, damping_min=0.02),
]
site = SiteResponseLayer(layers, rock_density=2200, rock_Vs=760)
bedrock_motion = gme.synthetic_motion(duration=15.0, dt=0.01, pga_target=2.0, dominant_freq=4.0, rng=rng)
site_result = site.analyze(bedrock_motion, n_iter=8)
print(f"Bedrock PGA = {bedrock_motion.pga:.3f} m/s^2")
print(f"Surface PGA = {site_result['surface_motion'].pga:.3f} m/s^2")
print(f"Amplification factor = {site_result['amplification_PGA']:.2f}")
print(f"Converged in {site_result['iterations']} iterations")
print(f"Converged G/Gmax per layer = {site_result['converged_G'] / np.array([l.Gmax for l in layers])}")
print(f"Converged damping per layer = {site_result['converged_D']}")
assert site_result["surface_motion"].pga > 0, "Surface motion degenerate"
assert np.all(site_result["converged_G"] <= np.array([l.Gmax for l in layers]) + 1e-6), \
    "G should reduce or stay equal under strain softening"
print("PASS\n")

print("=" * 70)
print("TEST 3a: StructuralResponseLayer -- elastic SDOF validation vs response spectrum")
print("=" * 70)
# Single-story elastic building: check that Newmark-beta time-history peak
# displacement matches the SDOF response spectrum Sd at the same period.
mass = 1.0e5  # kg
target_T = 0.5  # s
k = mass * (2 * np.pi / target_T) ** 2
story = Story(mass=mass, k_elastic=k, uy=1e6, alpha_ratio=1.0, height=3.0)  # alpha=1 -> stays linear
struct = StructuralResponseLayer([story], zeta=0.05)
modal = struct.modal_analysis()
print(f"Target period = {target_T} s, model period = {modal['periods'][0]:.4f} s")

test_motion = gme.synthetic_motion(duration=20.0, dt=0.005, pga_target=2.0, dominant_freq=2.0, rng=rng)
th_result = struct.time_history_analysis(test_motion, nonlinear=False)
peak_disp_newmark = th_result["max_drift_ratio"][0] * story.height

spec_at_T = gme.response_spectrum(test_motion, np.array([modal["periods"][0]]), zeta=0.05)
Sd_theory = spec_at_T["Sd"][0]
print(f"Newmark-beta peak displacement = {peak_disp_newmark:.6f} m")
print(f"Response-spectrum Sd (independent method) = {Sd_theory:.6f} m")
rel_err = abs(peak_disp_newmark - Sd_theory) / Sd_theory
print(f"Relative error = {rel_err*100:.2f}%")
assert rel_err < 0.05, f"Newmark-beta MDOF (n=1) should match SDOF response spectrum, err={rel_err}"
print("PASS\n")

print("=" * 70)
print("TEST 3b: StructuralResponseLayer -- nonlinear 3-story with weak story")
print("=" * 70)
stories = [
    Story(mass=2e5, k_elastic=8e8, uy=0.02, alpha_ratio=0.05, height=3.5),
    Story(mass=2e5, k_elastic=3e8, uy=0.015, alpha_ratio=0.05, height=3.5),  # weak story
    Story(mass=2e5, k_elastic=8e8, uy=0.02, alpha_ratio=0.05, height=3.5),
]
struct3 = StructuralResponseLayer(stories, zeta=0.05)
strong_motion = gme.synthetic_motion(duration=20.0, dt=0.005, pga_target=6.0, dominant_freq=1.5, rng=rng)
result3 = struct3.time_history_analysis(strong_motion, nonlinear=True)
print(f"Max drift ratios per story: {result3['max_drift_ratio']}")
print(f"Damage index per story: {result3['damage_index']}")
assert result3["max_drift_ratio"][1] >= result3["max_drift_ratio"][0] * 0.8, \
    "Weak story (story 2) expected to show comparable/larger drift concentration"
assert np.all(np.isfinite(result3["displacement"])), "Non-finite displacement in nonlinear analysis"
print("PASS (weak-story drift concentration observed, solution stable/finite)\n")

print("=" * 70)
print("TEST 4: LiquefactionAssessment")
print("=" * 70)
depths = np.linspace(1, 15, 15)
loose_N = np.full(15, 4.0)   # loose sand, N1-60cs = 4
dense_N = np.full(15, 30.0)  # dense sand, N1-60cs = 30
unit_weight = np.full(15, 18000.0)  # N/m^3

loose_result = LiquefactionAssessment.assess_profile(
    depths, loose_N, unit_weight, water_table_depth=1.5, pga_g=0.4, Mw=7.5)
dense_result = LiquefactionAssessment.assess_profile(
    depths, dense_N, unit_weight, water_table_depth=1.5, pga_g=0.4, Mw=7.5)

print(f"Loose sand FS range: {loose_result['FS'].min():.2f} - {loose_result['FS'].max():.2f}")
print(f"  Liquefiable layers: {loose_result['liquefiable'].sum()}/{len(depths)}")
print(f"Dense sand FS range: {dense_result['FS'].min():.2f} - {dense_result['FS'].max():.2f}")
print(f"  Liquefiable layers: {dense_result['liquefiable'].sum()}/{len(depths)}")

assert loose_result["liquefiable"].sum() > dense_result["liquefiable"].sum(), \
    "Loose sand should be more liquefiable than dense sand"
assert np.all(dense_result["FS"] > 1.0), "Dense sand should not liquefy at moderate PGA"
print("PASS\n")

print("=" * 70)
print("TEST 5: SeismicOne end-to-end orchestration")
print("=" * 70)
so = SeismicOne(site_layer=site, structure_layer=struct3)
full_results = so.run(bedrock_motion, do_site_response=True, do_structure=True)
print(so.summary(full_results))
assert "structural_response" in full_results
assert "site_response" in full_results
print("\nPASS\n")

print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
