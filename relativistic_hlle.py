"""
=============================================================================
RELATIVISTIC 3D COMPRESSIBLE CFD WITH HLLE RIEMANN SOLVER
Special-Relativistic Hydrodynamics — production-grade architecture
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

PHYSICS (special relativity, c=1, Minkowski metric)
---------------------------------------------------
Conservative variables:
    D   = W rho              (relativistic rest-mass density)
    S_i = W^2 rho h v_i      (momentum density)
    tau = W^2 rho h - p - D  (energy minus rest mass)
where W = 1/sqrt(1-v^2), h = 1 + eps + p/rho (specific enthalpy),
ideal-gas EOS: p = (gamma_ad - 1) rho eps.

Fluxes along direction j:
    F_D   = D v_j
    F_Si  = S_i v_j + p delta_ij
    F_tau = (tau + p) v_j

HLLE solver (Einfeldt 1988 / adapted for SRH):
    F_HLLE = (S^+ F_L - S^- F_R + S^+ S^- (U_R - U_L)) / (S^+ - S^-)

HONEST SCOPE
------------
• This is SPECIAL relativity (flat Minkowski), not GR, not "quantum".
• First-order HLLE is robust but diffusive; production codes (PLUTO,
  Athena++, Whisky) add reconstruction (MUSCL/WENO) — structure provided.
• Primitive recovery (cons->prim) is the delicate step; clamped & flagged.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.hlle")

__all__ = [
    "SRHConfig",
    "PrimitiveRecovery",
    "RelativisticState",
    "HLLESolver",
    "RelativisticCompressible3D",
]


@dataclass
class SRHConfig:
    n: int = 64                       # grid per dimension (1D/2D/3D)
    dim: int = 3                      # spatial dimensions
    gamma_ad: float = 4.0 / 3.0       # adiabatic index (relativistic gas)
    cfl: float = 0.4
    dx: float = 1.0
    newton_iters: int = 30
    newton_tol: float = 1e-10
    W_max: float = 50.0               # Lorentz factor clamp
    device: str = "cpu"


class PrimitiveRecovery:
    """
    Conservative -> primitive via Newton-Raphson on pressure.
    The numerically delicate heart of relativistic CFD.
    """

    def __init__(self, cfg: SRHConfig):
        self.cfg = cfg

    def lorentz(self, v2: Tensor) -> Tensor:
        one_minus = (1.0 - v2).clamp(min=1.0 / (self.cfg.W_max ** 2))
        return torch.rsqrt(one_minus).clamp(max=self.cfg.W_max)

    def recover(self, Q: Tensor) -> Dict[str, Tensor]:
        """
        Q: (5, ...) conserved [D, Sx, Sy, Sz, tau].
        Returns primitives {rho, vx, vy, vz, p, W, v2, cs2, mach, ok}.
        """
        c = self.cfg
        D, Sx, Sy, Sz, tau = Q[0], Q[1], Q[2], Q[3], Q[4]
        S2 = Sx * Sx + Sy * Sy + Sz * Sz

        # initial guess
        p = ((c.gamma_ad - 1.0) * tau.clamp(min=1e-10)).clamp(min=1e-12)

        def residual(p_cur: Tensor) -> Tensor:
            xi = (tau + D + p_cur).clamp(min=1e-10)     # = W^2 rho h
            v2 = (S2 / (xi * xi)).clamp(max=1.0 - 1e-8)
            W = self.lorentz(v2)
            rho = (D / W).clamp(min=1e-12)
            eps = xi / (W * W * rho) - 1.0 - p_cur / rho
            return p_cur - (c.gamma_ad - 1.0) * rho * eps

        # damped Newton
        for _ in range(c.newton_iters):
            f0 = residual(p)
            dp = (p.abs() + 1e-8) * 1e-5
            deriv = (residual(p + dp) - f0) / (dp + 1e-16)
            step = f0 / (deriv + 1e-14 * torch.sign(deriv + 1e-30))
            p_new = (p - 0.5 * step).clamp(min=1e-12)
            if torch.all((p_new - p).abs() < c.newton_tol * (p.abs() + 1e-8)):
                p = p_new
                break
            p = p_new

        xi = (tau + D + p).clamp(min=1e-10)
        v2 = (S2 / (xi * xi)).clamp(max=1.0 - 1e-8)
        W = self.lorentz(v2)
        rho = (D / W).clamp(min=1e-10)
        vx, vy, vz = Sx / xi, Sy / xi, Sz / xi
        eps = p / ((c.gamma_ad - 1.0) * rho)
        h = 1.0 + eps + p / rho
        cs2 = (c.gamma_ad * p / (rho * h + 1e-12)).clamp(min=1e-10, max=1.0)
        mach = torch.sqrt(v2) / (torch.sqrt(cs2) + 1e-12)
        ok = (residual(p).abs() < 1e-5).float()

        return {
            "rho": rho, "vx": vx, "vy": vy, "vz": vz, "p": p,
            "W": W, "v2": v2, "h": h, "cs2": cs2, "mach": mach, "ok": ok,
        }


class RelativisticState:
    """Pack/unpack conservative state and compute fluxes."""

    def __init__(self, cfg: SRHConfig, pr: PrimitiveRecovery):
        self.cfg = cfg
        self.pr = pr

    def prim_to_cons(self, rho, vx, vy, vz, p) -> Tensor:
        c = self.cfg
        v2 = (vx * vx + vy * vy + vz * vz).clamp(max=1.0 - 1e-6)
        W = self.pr.lorentz(v2)
        rho_s = rho.clamp(min=1e-12)
        eps = p / ((c.gamma_ad - 1.0) * rho_s)
        h = 1.0 + eps + p / rho_s
        W2rho_h = W * W * rho_s * h
        D = W * rho_s
        tau = W2rho_h - p - D
        return torch.stack([D, W2rho_h * vx, W2rho_h * vy, W2rho_h * vz, tau])

    def flux(self, Q: Tensor, prim: Dict[str, Tensor], axis: int) -> Tensor:
        """
        Physical flux along `axis` (0=x,1=y,2=z).
        F = [D v_j, S_i v_j + p delta_ij, (tau+p) v_j]
        """
        D, Sx, Sy, Sz, tau = Q[0], Q[1], Q[2], Q[3], Q[4]
        p = prim["p"]
        vj = [prim["vx"], prim["vy"], prim["vz"]][axis]
        S = [Sx, Sy, Sz]

        fD = D * vj
        fSx = Sx * vj + (p if axis == 0 else 0.0)
        fSy = Sy * vj + (p if axis == 1 else 0.0)
        fSz = Sz * vj + (p if axis == 2 else 0.0)
        ftau = (tau + p) * vj
        return torch.stack([fD, fSx, fSy, fSz, ftau])

    def sound_speed(self, prim: Dict[str, Tensor]) -> Tensor:
        """Relativistic sound speed cs = sqrt(gamma p / (rho h))."""
        return torch.sqrt(prim["cs2"].clamp(min=1e-10))


class HLLESolver:
    """
    HLLE approximate Riemann solver for SRH.

    Wave-speed estimates use the relativistic velocity-addition form
    for acoustic characteristics:
        lambda^+- = (v +- cs) / (1 +- v cs)
    (valid bound for the fastest signal along an axis).
    """

    def __init__(self, cfg: SRHConfig, pr: PrimitiveRecovery,
                 state: RelativisticState):
        self.cfg = cfg
        self.pr = pr
        self.state = state

    def _wave_speeds(self, primL, primR, v_L, v_R, axis) -> Tuple[Tensor, Tensor]:
        cs_L = self.state.sound_speed(primL)
        cs_R = self.state.sound_speed(primR)

        def lam_plus(v, cs):
            return ((v + cs) / (1.0 + v * cs)).clamp(min=-1.0 + 1e-8, max=1.0 - 1e-8)

        def lam_minus(v, cs):
            return ((v - cs) / (1.0 - v * cs)).clamp(min=-1.0 + 1e-8, max=1.0 - 1e-8)

        # robust envelope: take most extreme across both states
        a_plus = torch.maximum(lam_plus(v_L, cs_L), lam_plus(v_R, cs_R))
        a_minus = torch.minimum(lam_minus(v_L, cs_L), lam_minus(v_R, cs_R))
        a_plus = a_plus.clamp(min=0.0)
        a_minus = a_minus.clamp(max=0.0)
        return a_plus, a_minus

    def solve(self, QL: Tensor, QR: Tensor, axis: int) -> Tensor:
        """
        HLLE flux at an interface. QL, QR: (5, ...) conserved states
        on the left/right of the interface along `axis`.
        """
        primL = self.pr.recover(QL)
        primR = self.pr.recover(QR)
        v_L = [primL["vx"], primL["vy"], primL["vz"]][axis]
        v_R = [primR["vx"], primR["vy"], primR["vz"]][axis]

        FL = self.state.flux(QL, primL, axis)
        FR = self.state.flux(QR, primR, axis)
        a_plus, a_minus = self._wave_speeds(primL, primR, v_L, v_R, axis)

        # HLLE formula
        denom = (a_plus - a_minus).clamp(min=1e-12)
        F_hlle = (a_plus * FL - a_minus * FR + a_plus * a_minus * (QR - QL)) / denom

        # pick upwind-consistent branch for stability
        all_right = (a_minus >= 0.0)
        all_left = (a_plus <= 0.0)
        F = torch.where(all_right, FL, torch.where(all_left, FR, F_hlle))
        return F


class RelativisticCompressible3D:
    """
    Full 3D compressible relativistic update via HLLE, dimension-agnostic
    (works for 1D/2D/3D grids via torch.roll periodic BC).
    """

    def __init__(self, cfg: SRHConfig):
        self.cfg = cfg
        self.pr = PrimitiveRecovery(cfg)
        self.state = RelativisticState(cfg, self.pr)
        self.hl = HLLESolver(cfg, self.pr, self.state)

    def _roll(self, Q: Tensor, shift: int, axis: int) -> Tensor:
        # spatial axes are the last cfg.dim dims
        return torch.roll(Q, shifts=shift, dims=-(self.cfg.dim - axis))

    def compute_dt(self, Q: Tensor) -> Tensor:
        prim = self.pr.recover(Q)
        cs = self.state.sound_speed(prim)
        v2 = prim["v2"]
        vmax = torch.sqrt(v2.max())
        smax = torch.maximum(vmax, cs.max()).clamp(min=1e-8)
        return torch.tensor(self.cfg.cfl * self.cfg.dx / float(smax),
                            dtype=Q.dtype, device=Q.device)

    def step(self, Q: Tensor, dt: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """One conservative update: dQ/dt = -div(F) via HLLE fluxes."""
        if dt is None:
            dt = self.compute_dt(Q)

        prim = self.pr.recover(Q)
        divF = torch.zeros_like(Q)

        for axis in range(self.cfg.dim):
            QL = self._roll(Q, 1, axis)    # left neighbor
            QR = Q                          # current (interface QL->QR)
            F_iface = self.hl.solve(QL, QR, axis)
            F_right = self._roll(F_iface, -1, axis)
            # divergence approx (F_right - F_iface)/dx
            divF = divF + (F_right - F_iface) / self.cfg.dx

        Q_new = Q - dt * divF
        Q_new[0] = Q_new[0].clamp(min=1e-10)
        Q_new[4] = Q_new[4].clamp(min=1e-12)

        prim_new = self.pr.recover(Q_new)
        return {
            "Q": Q_new, "dt": dt,
            "W": prim_new["W"], "beta": torch.sqrt(prim_new["v2"].clamp(min=0)),
            "mach": prim_new["mach"], "p": prim_new["p"], "rho": prim_new["rho"],
            "recovery_ok_frac": prim_new["ok"].mean(),
        }
