# NSEM package – what was added / fixed

## Verified here (executed)
- `nsem/hardware/interlock.py` – 11 unit tests pass (`python -m unittest discover -s tests -t .`).
- Numeric checks (numpy mirrors): DAC unit fix, Pennes-CG vs dense solve (err ~1e-14).

## Written but NOT executed (sandbox has no torch / onnx / iree) – run these first
`tests/test_hardware_torch.py`, `tests/test_core_production_torch.py`, `tests/test_physics_torch.py`
(`pip install torch numpy` then the same unittest command).
`nsem/compile/lowering.py` and `nsem/runtime/dispatch.py` need IREE + onnxruntime.

## Bugs found in the uploaded code (now fixed)
| File | Bug | Effect |
|---|---|---|
| dac.py | charge uC computed with extra x1e3; 0.1 mm^2 taken as 1e-4 cm^2 | density 5e4-2e5 uC/cm^2 -> safety mask zeroed ~all output |
| dac.py | unipolar map of [-1,1] to [0,I_fs] | zero command = +50 uA |
| dac.py | anodic := -cathodic | residual charge always 0 (check vacuous) |
| interlock.py | NaN passes every `>` test; missing sensor defaults to 0; negative current unchecked | unsafe commands ALLOWed |
| interlock.py | sha256(repr+secret) "token"; nonce resets on restart; all-zero default key | forgeable / replayable |
| module_production.py | interlock got residual charge as "charge density"; validation skipped if state None | safety check bypassed |
| chemical.py | `.clamp(min=)` after softplus | dead gradients |
| dispatch.py | wrong IREE API; KeyError for non-compiled branches | could not run |
| mlir_lowering.py | passes/targets that do not exist (`--nsem-*`, `photonic-quake`) | pipeline could not run |
| sota_optogenetics_full.py | "implicit CG" was 8 Jacobi sweeps; `att+dif` heuristic | docstring != code |

## Still missing (cannot be filled by code alone)
- Real patch-clamp data + fit/validation; benchmarks vs pyrho / MCX.
- Custom MLIR dialects for photonic / quantum backends (those branches run as "reference").
- Electrode spec: `max_current_density_a_m2=20` is inconsistent with a 100 uA / 0.1 mm^2 electrode.
- Clinical limits (`max_chem_volume_nl` is a placeholder = model max). Not a certified safety function.
- The `.clamp_min` calls remaining in sota_optogenetics_full.py (spectrum, eff) contradict its "no clamps" claim.
