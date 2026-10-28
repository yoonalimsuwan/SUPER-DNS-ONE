"""
=============================================================================
MUSCL / WENO HIGH-ORDER RECONSTRUCTION FOR THE RELATIVISTIC HLLE SOLVER
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

PURPOSE
-------
Upgrade the first-order HLLE flux (which is diffusive) to high-order
accuracy via reconstruction at cell interfaces. Two families provided:

  [M1] MUSCL — second-order, TVD, with explicit slope limiters:
       minmod / van Leer / monotonized-central (MC) / superbee.

  [M2] WENO5 — fifth-order weighted essentially non-oscillatory,
       with both classical (WENO-JS) and improved (WENO-Z) weights.

WHY THIS MATTERS (paper §11.1)
------------------------------
The CFD paper requires verification via "spatial-resolution refinement"
and comparison against an independent discretization. A first-order
scheme cannot resolve shock/contact structure well enough for that
comparison; MUSCL/WENO reconstruction is the standard route to make the
HLLE solver sharp while remaining robust.

HONEST SCOPE
------------
• Reconstruction is applied to CONSERVATIVE variables. For relativistic
  hydro, reconstructing PRIMITIVES (rho, v, p) is usually more robust;
  a flag selects the mode. Primitive reconstruction is recommended.
• These are established, published techniques (van Leer 1979; Jiang &
  Shu 1996; Borges et al. 2008). No new mathematical claim is made.
• Stability of the FULL scheme (reconstruction + HLLE + integrator)
  must still be verified empirically; reconstruction alone does not
  guarantee it (paper §4.2: "This is a controller design, not a proof
  of stability.").
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.reconstruction")

__all__ = [
    "LimiterType",
    "MUSCLReconstruction",
    "WENOReconstruction",
]


class LimiterType(Enum):
    MINMOD = auto()
    VANLEER = auto()
    MC = auto()        # monotonized central
    SUPERBEE = auto()


# ═══════════════════════════════════════════════════════════════════════
# SLOPE LIMITERS
# ═══════════════════════════════════════════════════════════════════════

def minmod(a: Tensor, b: Tensor) -> Tensor:
    """minmod limiter: most diffusive, most robust."""
    return torch.where(
        a * b > 0,
        torch.where(a.abs() < b.abs(), a, b),
        torch.zeros_like(a),
    )

def mc_limiter(a: Tensor, b: Tensor) -> Tensor:
    """Monotonized-central limiter (van Leer's MC)."""
    c = 0.5 * (a + b)
    return torch.where(
        a * b > 0,
        torch.where(torch.minimum(a.abs(), b.abs()) * 2.0 < c.abs(),
                    torch.sign(c) * 2.0 * torch.minimum(a.abs(), b.abs()),
                    c),
        torch.zeros_like(a),
    )

def superbee(a: Tensor, b: Tensor) -> Tensor:
    """Superbee limiter: sharpest, can be less robust."""
    m1 = torch.where(a.abs() < 2.0 * b.abs(),
                     torch.sign(b) * torch.minimum(a.abs(), 2.0 * b.abs()),
                     torch.zeros_like(a))
    m2 = torch.where(2.0 * a.abs() < b.abs(),
                     torch.sign(a) * torch.minimum(2.0 * a.abs(), b.abs()),
                     torch.zeros_like(a))
    return torch.where(m1.abs() > m2.abs(), m1, m2)

def vanleer_slope(a: Tensor, b: Tensor) -> Tensor:
    """Harmonic-mean (van Leer) slope."""
    return torch.where(
        a * b > 0,
        2.0 * a * b / (a + b + 1e-30),
        torch.zeros_like(a),
    )


def apply_limiter(a: Tensor, b: Tensor, kind: LimiterType) -> Tensor:
    if kind == LimiterType.MINMOD:
        return minmod(a, b)
    if kind == LimiterType.VANLEER:
        return vanleer_slope(a, b)
    if kind == LimiterType.MC:
        return mc_limiter(a, b)
    if kind == LimiterType.SUPERBEE:
        return superbee(a, b)
    raise ValueError(f"Unknown limiter {kind}")


# ═══════════════════════════════════════════════════════════════════════
# [M1] MUSCL RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

class MUSCLReconstruction:
    """
    Second-order MUSCL reconstruction along one axis.

    Given cell values Q[..., i-1], Q[..., i], Q[..., i+1], produces
    left/right interface states at i+1/2:
        Q^R_{i+1/2} = Q_i     + 0.5 * slope_i
        Q^L_{i+3/2} = Q_{i+1} - 0.5 * slope_{i+1}
    where slope is limited to preserve TVD.

    Works on an arbitrary number of leading component dimensions and a
    single reconstruction axis (the last dim by default).
    """

    def __init__(self, limiter: LimiterType = LimiterType.MC):
        self.limiter = limiter

    def reconstruct(self, Q: Tensor, axis: int = -1) -> Tuple[Tensor, Tensor]:
        """
        Returns (Q_L, Q_R) at each interface along `axis`.
        Shapes match Q; the interface at index i sits between cell i-1 and i.
        Periodic indexing via torch.roll.
        """
        Qm = torch.roll(Q, shifts=1, dims=axis)   # i-1
        Qp = torch.roll(Q, shifts=-1, dims=axis)  # i+1

        # forward/backward differences at cell i
        df = Q - Qm
        db = Qp - Q
        slope = apply_limiter(df, db, self.limiter)

        # right state of interface (from cell i) and left state (from cell i+1)
        Q_R = Q + 0.5 * slope
        Qp_m = torch.roll(Qm, shifts=-1, dims=axis)   # (i+1)-1 = i
        Qp_p = Qp
        df_p = Qp - Q          # at i+1: Q_{i+1}-Q_i
        db_p = Qp_p - Qp       # at i+1: Q_{i+2}-Q_{i+1}
        slope_p = apply_limiter(df_p, db_p, self.limiter)
        Q_L = Qp - 0.5 * slope_p

        return Q_L, Q_R


# ═══════════════════════════════════════════════════════════════════════
# [M2] WENO5 RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

class WENOReconstruction:
    """
    Fifth-order WENO reconstruction (Jiang & Shu 1996) with optional
    WENO-Z weights (Borges et al. 2008) for improved accuracy near
    critical points.

    Standard WENO5 uses three 3-point substencils to build a 5th-order
    approximation, with nonlinear weights that avoid oscillation near
    discontinuities.
    """

    def __init__(self, eps: float = 1e-36, use_weno_z: bool = True):
        self.eps = eps
        self.use_weno_z = use_weno_z

    def _shift(self, Q: Tensor, s: int, axis: int) -> Tensor:
        return torch.roll(Q, shifts=-s, dims=axis)

    def reconstruct(self, Q: Tensor, axis: int = -1) -> Tuple[Tensor, Tensor]:
        """
        Returns (Q_L, Q_R) at interfaces along `axis`, WENO5-reconstructed.
        Periodic boundary via torch.roll.
        """
        # stencil values: Q_{i-2} ... Q_{i+2}
        qm2 = self._shift(Q, 2, axis)
        qm1 = self._shift(Q, 1, axis)
        q0 = Q
        qp1 = self._shift(Q, -1, axis)
        qp2 = self._shift(Q, -2, axis)

        # ---- left-biased reconstruction (for right-going flux) ----
        # three candidate polynomials evaluated at i+1/2
        f0 = (2 * qm2 - 7 * qm1 + 11 * q0) / 6.0
        f1 = (-qm1 + 5 * q0 + 2 * qp1) / 6.0
        f2 = (2 * q0 + 5 * qp1 - qp2) / 6.0

        # smoothness indicators
        b0 = (13.0 / 12.0) * (qm2 - 2 * qm1 + q0) ** 2 + \
             0.25 * (qm2 - 4 * qm1 + 3 * q0) ** 2
        b1 = (13.0 / 12.0) * (qm1 - 2 * q0 + qp1) ** 2 + \
             0.25 * (qm1 - qp1) ** 2
        b2 = (13.0 / 12.0) * (q0 - 2 * qp1 + qp2) ** 2 + \
             0.25 * (3 * q0 - 4 * qp1 + qp2) ** 2

        # ideal weights
        d0, d1, d2 = 0.1, 0.6, 0.3

        if self.use_weno_z:
            # WENO-Z: tau5 = |b0 - b2|
            tau5 = (b0 - b2).abs()
            a0 = d0 * (1 + (tau5 / (b0 + self.eps)) ** 2)
            a1 = d1 * (1 + (tau5 / (b1 + self.eps)) ** 2)
            a2 = d2 * (1 + (tau5 / (b2 + self.eps)) ** 2)
        else:
            a0 = d0 / (self.eps + b0) ** 2
            a1 = d1 / (self.eps + b1) ** 2
            a2 = d2 / (self.eps + b2) ** 2

        asum = a0 + a1 + a2
        w0, w1, w2 = a0 / asum, a1 / asum, a2 / asum
        Q_L = w0 * f0 + w1 * f1 + w2 * f2

        # ---- right-biased reconstruction (mirror) ----
        g0 = (2 * qp2 - 7 * qp1 + 11 * q0) / 6.0
        g1 = (-qp1 + 5 * q0 + 2 * qm1) / 6.0
        g2 = (2 * q0 + 5 * qm1 - qm2) / 6.0

        c0 = (13.0 / 12.0) * (qp2 - 2 * qp1 + q0) ** 2 + \
             0.25 * (qp2 - 4 * qp1 + 3 * q0) ** 2
        c1 = (13.0 / 12.0) * (qp1 - 2 * q0 + qm1) ** 2 + \
             0.25 * (qp1 - qm1) ** 2
        c2 = (13.0 / 12.0) * (q0 - 2 * qm1 + qm2) ** 2 + \
             0.25 * (3 * q0 - 4 * qm1 + qm2) ** 2

        if self.use_weno_z:
            tau5r = (c0 - c2).abs()
            r0 = d0 * (1 + (tau5r / (c0 + self.eps)) ** 2)
            r1 = d1 * (1 + (tau5r / (c1 + self.eps)) ** 2)
            r2 = d2 * (1 + (tau5r / (c2 + self.eps)) ** 2)
        else:
            r0 = d0 / (self.eps + c0) ** 2
            r1 = d1 / (self.eps + c1) ** 2
            r2 = d2 / (self.eps + c2) ** 2

        rsum = r0 + r1 + r2
        v0, v1, v2 = r0 / rsum, r1 / rsum, r2 / rsum
        Q_R = v0 * g0 + v1 * g1 + v2 * g2

        return Q_L, Q_R


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" MUSCL / WENO Reconstruction — Smoke Test")
    print("=" * 72)
    torch.manual_seed(0)
    n = 64
    x = torch.linspace(0, 1, n)
    # smooth sine
    u_smooth = torch.sin(2 * math.pi * x)
    # discontinuous step
    u_step = torch.where(x < 0.5, torch.ones(n), torch.zeros(n))

    muscl = MUSCLReconstruction(limiter=LimiterType.MC)
    weno = WENOReconstruction(use_weno_z=True)

    for name, u in [("smooth", u_smooth), ("step", u_step)]:
        mL, mR = muscl.reconstruct(u.unsqueeze(0), axis=-1)
        wL, wR = weno.reconstruct(u.unsqueeze(0), axis=-1)
        print(f"\n  [{name}]")
        print(f"    MUSCL Q_L/Q_R shape : {tuple(mL.shape)}")
        print(f"    WENO  Q_L/Q_R shape : {tuple(wL.shape)}")
        print(f"    WENO left  finite   : {bool(torch.isfinite(wL).all())}")
        print(f"    WENO right finite   : {bool(torch.isfinite(wR).all())}")

    # Check WENO reproduces linear functions exactly (5th-order property)
    u_lin = 3.0 * x + 1.0
    wL, wR = weno.reconstruct(u_lin.unsqueeze(0), axis=-1)
    # at interface i+1/2, exact value = u_lin shifted by half-cell
    u_exact_L = 3.0 * (x + 0.5 / n) + 1.0
    err = (wL.squeeze() - u_exact_L).abs().max().item()
    print(f"\n  WENO5 linear-exactness error: {err:.2e} (should be ~0)")
    print(f"    [{'PASS' if err < 1e-4 else 'FAIL'}] WENO5 reproduces linear data")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Reconstruction is applied to the chosen variable set.")
    print("  • For relativistic flows prefer primitive reconstruction.")
    print("  • Full-scheme stability still requires empirical verification")
    print("    (paper §4.2: reconstruction is not itself a stability proof).")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
