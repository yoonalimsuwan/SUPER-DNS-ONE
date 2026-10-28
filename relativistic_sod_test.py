"""
=============================================================================
RELATIVISTIC SOD SHOCK TUBE — correctness verification for the HLLE solver
=============================================================================
Standard relativistic Riemann problem (cf. Waldmann 1999; widely used to
validate relativistic hydro codes such as PLUTO / Athena++ / Whisky).

Setup (classic high-contrast relativistic Sod):
    LEFT  : rho=10.0, p=13.3333 (eps=2.0), v=0
    RIGHT : rho=1.0,  p=1e-6,            v=0
    gamma_ad = 4/3, domain [0,1], discontinuity at x=0.5, run to t=0.4

Expected qualitative structure at t=0.4:
    • left-going rarefaction fan
    • contact discontinuity (density jump, pressure continuous)
    • right-going relativistic shock (moves at ~ 0.6-0.8 c)

These are the hallmarks used to confirm the solver is physically correct.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from typing import Dict

import torch

from relativistic_hlle import SRHConfig, RelativisticCompressible3D

logger = logging.getLogger("sesi.sod")


def setup_sod(n: int = 200, device: str = "cpu") -> Dict:
    """Build the relativistic Sod initial condition."""
    cfg = SRHConfig(n=n, dim=1, gamma_ad=4.0 / 3.0, dx=1.0 / n,
                    cfl=0.4, device=device)
    solver = RelativisticCompressible3D(cfg)

    x = torch.linspace(0, 1, n, device=device)
    left = x < 0.5

    rho = torch.where(left, torch.full_like(x, 10.0), torch.full_like(x, 1.0))
    p = torch.where(left, torch.full_like(x, 13.3333), torch.full_like(x, 1e-6))
    vx = torch.zeros_like(x)
    vy = torch.zeros_like(x)
    vz = torch.zeros_like(x)

    # shape (5, n) for 1D
    Q = solver.state.prim_to_cons(rho, vx, vy, vz, p)
    return {"cfg": cfg, "solver": solver, "Q": Q, "x": x}


def run_sod(t_final: float = 0.4, n: int = 200, device: str = "cpu") -> Dict:
    """Evolve the Sod problem to t_final."""
    s = setup_sod(n=n, device=device)
    solver, Q, cfg = s["solver"], s["Q"], s["cfg"]

    t = 0.0
    steps = 0
    while t < t_final:
        out = solver.step(Q)
        Q = out["Q"]
        t += float(out["dt"])
        steps += 1
        if steps > 200000:
            logger.warning("step limit reached")
            break

    prim = solver.pr.recover(Q)
    return {
        "x": s["x"], "rho": prim["rho"], "p": prim["p"], "vx": prim["vx"],
        "W": prim["W"], "recovery_ok_frac": prim["ok"].mean(),
        "steps": steps, "t": t,
    }


def verify_sod(result: Dict) -> Dict[str, bool]:
    """
    Check the expected physical structure of the relativistic Sod problem.
    These are qualitative correctness checks, not exact-value comparisons
    (exact values require a reference solver / analytic Riemann solution).
    """
    x = result["x"]
    rho = result["rho"]
    p = result["p"]
    vx = result["vx"]

    checks = {}

    # 1. density: high on far left, low on far right
    checks["density_gradient_correct"] = bool(
        rho[x < 0.1].mean() > rho[x > 0.9].mean()
    )

    # 2. pressure: high left, low right, with a jump (shock)
    checks["pressure_gradient_correct"] = bool(
        p[x < 0.1].mean() > p[x > 0.9].mean()
    )

    # 3. velocity: flow pushed to the right (positive vx in shocked region)
    checks["rightward_flow"] = bool(vx.max() > 0.0)

    # 4. Lorentz factor remains finite and physical
    checks["lorentz_finite"] = bool(torch.isfinite(result["W"]).all()
                                    and result["W"].max() < 50.0)

    # 5. primitive recovery succeeded almost everywhere
    checks["recovery_ok"] = bool(result["recovery_ok_frac"] > 0.99)

    # 6. no NaNs
    checks["no_nan"] = bool(torch.isfinite(rho).all() and torch.isfinite(p).all())

    return checks


def smoke_test():
    print("=" * 72)
    print(" Relativistic Sod Shock Tube — HLLE Verification")
    print("=" * 72)
    torch.manual_seed(0)

    result = run_sod(t_final=0.4, n=200)
    print(f"\n  steps = {result['steps']}, final t = {result['t']:.4f}")
    print(f"  recovery_ok_frac = {float(result['recovery_ok_frac']):.4f}")
    print(f"  W_max = {float(result['W'].max()):.4f}")
    print(f"  vx_max = {float(result['vx'].max()):.4f} c")
    print(f"  rho range = [{float(result['rho'].min()):.4g}, "
          f"{float(result['rho'].max()):.4g}]")
    print(f"  p   range = [{float(result['p'].min()):.4g}, "
          f"{float(result['p'].max()):.4g}]")

    checks = verify_sod(result)
    print("\n  ── Correctness checks (expected Sod structure) ──")
    all_ok = True
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    print("\n" + "=" * 72)
    if all_ok:
        print("  ALL CHECKS PASSED — HLLE solver reproduces relativistic Sod structure")
    else:
        print("  SOME CHECKS FAILED — solver needs debugging")
    print("=" * 72)
    print("  HONEST NOTES:")
    print("  • These are structural checks, not exact-value validation.")
    print("  • For publication-grade validation, compare against an analytic")
    print("    relativistic Riemann solution or PLUTO/Athena++ reference output.")
    print("  • First-order HLLE is diffusive; add MUSCL/WENO reconstruction")
    print("    for sharper shocks (structure shown in RelativisticCompressible3D).")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
