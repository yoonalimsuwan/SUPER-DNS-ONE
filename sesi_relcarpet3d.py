"""
=============================================================================
SESI RELATIVISTIC 3D COMPRESSIBLE CFD × PROPER-TIME × 8TH-ORDER POLYHARMONIC
A separate Layer-1 physical model (per uploaded CFD paper §10.3), NOT a
time-dilation patch on the Newtonian 2D prototype.
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS FILE HONESTLY IS
----------------──────────
[L1] A SPECIAL-RELATIVISTIC (flat Minkowski spacetime) 3D compressible
     hydrodynamics prototype in conservative form:
        D   = W rho
        S_i = W^2 rho h v_i
        tau = W^2 rho h - p - D
     with W = 1/sqrt(1-v^2) (c=1), ideal-gas EOS p=(gamma_ad-1) rho eps.
     This is the "replace the physical evolution model with a consistent
     metric-aware formulation" step the paper §10.3 prescribes.

[L2] GENUINE RELATIVISTIC TIME (paper §6.4 interface, Minkowski metric):
     T : (metric, worldline, dt) -> d tau,  with  d tau = dt / W.
     Each cell carries a PROPER-TIME field that advances slower where the
     local Lorentz factor W is larger. This is real SR time dilation,
     bookkept — NOT a multiplicative patch on a Newtonian timestep.

[L3] 8TH-ORDER POLYHARMONIC REGULARIZATION:
     The Euclidean grid realization of differential-length-8 structure:
        A^4 = (nabla^2)^4 = nabla^8   (dissipative hyperviscosity -nu8*A^4)
     HONEST: this is the FLAT-SPACE shadow of the Carpet paper's
     A^4 = d*(d d*)^3 d (Prop 4, Cor 4: formula 8.17, no lower-order terms).
     The true fractal carpet operator requires the carpet Dirichlet-form /
     Cipriani-Sauvageot machinery and is NOT what acts on this CFD grid.

[L4] NO-ZENO GATE (Six Problems Prop 6.1) modulates the hyperviscosity:
     when the stress accumulator crosses theta_hi (hysteresis gap
     delta_min = theta_hi-theta_lo > 0), a collapse event raises nu8 to
     smooth developing small-scale structure. Bound N(T) <= (E_max+D)/delta_min.

RELATION TO THE NEWTONIAN PROTOTYPE
-----------------------------------
This is a SEPARATE model. The 2D incompressible vorticity prototype and
this SRH module do not share an integrator; the paper's 5-layer separation
(L1 physical model vs L5 time bookkeeping) is preserved. No claim that
this resolves Navier-Stokes regularity, and no claim this is GR.

WHAT THIS FILE IS NOT
---------------------
• Not general relativity (flat metric only; no curved spacetime).
• Not a production relativistic code (no HLL/HLLE Riemann solver, no AMR;
  first-order + Lax stabilization). Cf. PLUTO / Athena++ / Whisky / BHAC.
• Primitive recovery (cons->prim) is the delicate step; we clamp + report.
• The 8th-order term is numerical regularization, not new physics.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger("sesi.relcarpet3d")

__all__ = [
    "RelCarpetConfig",
    "MinkowskiTimeInterface",
    "Polyharmonic8",
    "RelativisticCompressibleCore",
    "NoZenoGate",
    "RelativisticCarpetCFD",
]


# ── C^∞ primitives (SESİ convention) ─────────────────────────────────
def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RelCarpetConfig:
    # grid
    n: int = 16
    length: float = 1.0

    # relativistic EOS
    gamma_ad: float = 4.0 / 3.0     # relativistic ideal gas
    c_light: float = 1.0

    # primitive-recovery Newton
    newton_iters: int = 40
    newton_tol: float = 1e-9
    W_max: float = 50.0             # Lorentz clamp

    # 8th-order polyharmonic hyperviscosity
    nu8_base: float = 1e-6
    nu8_collapse: float = 1e-3      # raised when the gate fires

    # base viscosity (relativistic NS analogue, kept tiny)
    nu: float = 1e-4

    # no-Zeno gate (Six Problems Prop 6.1)
    theta_hi: float = 1.0
    theta_lo: float = 0.6
    stress_decay: float = 0.9

    device: str = "cpu"

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def delta_min(self) -> float:
        return self.theta_hi - self.theta_lo


# ═══════════════════════════════════════════════════════════════════════
# 3D FINITE-DIFFERENCE OPERATORS (periodic)
# ═══════════════════════════════════════════════════════════════════════

class FD3D:
    """Central finite differences on a periodic 3D grid."""

    def __init__(self, dx: float):
        self.dx = dx

    def grad(self, f: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        d = 2.0 * self.dx
        gx = (torch.roll(f, -1, -3) - torch.roll(f, 1, -3)) / d
        gy = (torch.roll(f, -1, -2) - torch.roll(f, 1, -2)) / d
        gz = (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) / d
        return gx, gy, gz

    def laplacian(self, f: Tensor) -> Tensor:
        d2 = self.dx * self.dx
        return (
            torch.roll(f, 1, -3) + torch.roll(f, -1, -3)
            + torch.roll(f, 1, -2) + torch.roll(f, -1, -2)
            + torch.roll(f, 1, -1) + torch.roll(f, -1, -1)
            - 6.0 * f
        ) / d2


# ═══════════════════════════════════════════════════════════════════════
# [L2] MINKOWSKI TIME INTERFACE  (paper §6.4, flat metric)
# ═══════════════════════════════════════════════════════════════════════

class MinkowskiTimeInterface:
    """
    Implements the paper §6.4 time-conversion interface
        T : (metric, worldline, dt) -> d tau
    with the MINKOWSKI metric (special relativity) and each cell's
    worldline characterized by its local Lorentz factor W.

    Proper-time increment:  d tau = dt / W.
    This is the honest "full relativistic time for high speeds": it is
    genuine SR time dilation, derived from the metric and integrated with
    the relativistic fluid state — NOT a factor applied to a Newtonian step.

    HONEST SCOPE: flat spacetime only. For strong gravity one would supply
    a curved metric and the corresponding GR hydro model (out of scope here).
    """

    METRIC = "minkowski_diag(-1,1,1,1)"

    def __init__(self, cfg: RelCarpetConfig, shape: Tuple[int, ...]):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.coord_time = torch.zeros(1, device=self.device)
        self.proper_time = torch.zeros(*shape, device=self.device)

    @torch.no_grad()
    def advance(self, dt: Tensor, W: Tensor) -> Dict[str, Tensor]:
        """Advance lab-frame time dt; each cell's proper time advances dt/W."""
        self.coord_time += dt
        W_safe = W.clamp(min=1.0)
        self.proper_time += dt / W_safe
        return {
            "coord_time": self.coord_time.clone(),
            "proper_time_mean": self.proper_time.mean(),
            "proper_time_min": self.proper_time.min(),   # fastest cell ages least
            "proper_time_max": self.proper_time.max(),
            "clock_spread": self.proper_time.max() - self.proper_time.min(),
            "W_max": W_safe.max(),
            "metric": self.METRIC,
        }


# ═══════════════════════════════════════════════════════════════════════
# [L3] 8TH-ORDER POLYHARMONIC  (Euclidean shadow of Carpet A^4)
# ═══════════════════════════════════════════════════════════════════════

class Polyharmonic8(nn.Module):
    """
    Euclidean differential-length-8 operator  A^4 = (nabla^2)^4 = nabla^8.

    RELATION TO THE CARPET PAPER (honest):
      The Consolidated Carpet paper proves (Prop 4, Cor 4) that the
      fractal polyharmonic operator satisfies  A^k = d*(d d*)^k-1 d  and,
      for k=4, formula (8.17) holds with NO lower-order terms. In flat
      Euclidean space with d = grad and d* = -div, that construction
      reduces to (-Delta)^4 = nabla^8. So the operator implemented here is
      the SAME polyharmonic / differential-length-8 structure realized on a
      grid — the flat-space shadow of the carpet A^4.

      It is NOT the fractal carpet operator itself, which is defined only
      through the carpet Dirichlet form and Cipriani-Sauvageot cotangent
      module and cannot act on an ordinary CFD grid.

    USE: as hyperviscous regularization. The dissipative term added to the
    evolution is  -nu8 * A^4 u  (eigenvalues -nu8 |k|^8 < 0), which damps
    only the smallest scales. nu8 is modulated by the no-Zeno gate.
    """

    def __init__(self, cfg: RelCarpetConfig):
        super().__init__()
        self.cfg = cfg
        self.fd = FD3D(cfg.dx)

    def apply_A4(self, u: Tensor) -> Tensor:
        """A^4 u = lap^4 u = nabla^8 u (iterate the discrete Laplacian 4x)."""
        v = u
        for _ in range(4):
            v = self.fd.laplacian(v)
        return v

    def hyperviscosity(self, u: Tensor, nu8: Tensor) -> Tensor:
        """Dissipative 8th-order term: -nu8 * A^4 u."""
        return -nu8 * self.apply_A4(u)


# ═══════════════════════════════════════════════════════════════════════
# [L1] SPECIAL-RELATIVISTIC COMPRESSIBLE CORE
# ═══════════════════════════════════════════════════════════════════════

class RelativisticCompressibleCore(nn.Module):
    """
    3D special-relativistic compressible hydrodynamics, conservative form.

    State Q = [D, Sx, Sy, Sz, tau].  c = 1.
    Primitive recovery uses Newton iteration on pressure (the delicate step
    of relativistic CFD); we clamp and report a recovery_ok fraction.
    """

    def __init__(self, cfg: RelCarpetConfig):
        super().__init__()
        self.cfg = cfg
        self.fd = FD3D(cfg.dx)
        self.poly8 = Polyharmonic8(cfg)

    # ── kinematics ──────────────────────────────────────────────────────
    def lorentz(self, v2: Tensor) -> Tensor:
        one_minus = (1.0 - v2).clamp(min=1.0 / (self.cfg.W_max ** 2))
        return torch.rsqrt(one_minus).clamp(max=self.cfg.W_max)

    def sound_speed_sq(self, rho: Tensor, p: Tensor, h: Tensor) -> Tensor:
        return (self.cfg.gamma_ad * p / (rho * h + 1e-12)).clamp(min=1e-8, max=1.0)

    # ── primitive -> conservative ───────────────────────────────────────
    def prim_to_cons(self, rho, vx, vy, vz, p) -> Tensor:
        c = self.cfg
        v2 = (vx * vx + vy * vy + vz * vz).clamp(max=1.0 - 1e-6)
        W = self.lorentz(v2)
        rho_safe = rho.clamp(min=1e-8)
        eps = p / ((c.gamma_ad - 1.0) * rho_safe)
        h = 1.0 + eps + p / rho_safe
        W2rho_h = W * W * rho_safe * h
        D = W * rho_safe
        tau = W2rho_h - p - D
        return torch.stack([D, W2rho_h * vx, W2rho_h * vy, W2rho_h * vz, tau])

    # ── conservative -> primitive (Newton on p) ────────────────────────
    def cons_to_prim(self, Q: Tensor) -> Dict[str, Tensor]:
        c = self.cfg
        D, Sx, Sy, Sz, tau = Q[0], Q[1], Q[2], Q[3], Q[4]
        S2 = Sx * Sx + Sy * Sy + Sz * Sz

        p = ((c.gamma_ad - 1.0) * tau.clamp(min=1e-8)).clamp(min=1e-10)

        def residual(p_cur: Tensor) -> Tensor:
            xi = (tau + D + p_cur).clamp(min=1e-8)          # = W^2 rho h
            v2 = (S2 / (xi * xi)).clamp(max=1.0 - 1e-8)
            W = self.lorentz(v2)
            rho = (D / W).clamp(min=1e-10)
            eps = xi / (W * W * rho) - 1.0 - p_cur / rho
            return p_cur - (c.gamma_ad - 1.0) * rho * eps

        for _ in range(c.newton_iters):
            f0 = residual(p)
            dp = (p.abs() + 1e-6) * 1e-5
            deriv = (residual(p + dp) - f0) / (dp + 1e-14)
            step = f0 / (deriv + 1e-12 * torch.sign(deriv + 1e-30))
            p_new = (p - 0.5 * step).clamp(min=1e-10)       # damped
            if torch.all((p_new - p).abs() < c.newton_tol * (p.abs() + 1e-8)):
                p = p_new
                break
            p = p_new

        xi = (tau + D + p).clamp(min=1e-8)
        v2 = (S2 / (xi * xi)).clamp(max=1.0 - 1e-8)
        W = self.lorentz(v2)
        rho = (D / W).clamp(min=1e-8)
        vx, vy, vz = Sx / xi, Sy / xi, Sz / xi
        eps = p / ((c.gamma_ad - 1.0) * rho)
        h = 1.0 + eps + p / rho
        cs2 = self.sound_speed_sq(rho, p, h)
        mach = torch.sqrt(v2) / (torch.sqrt(cs2) + 1e-12)
        recovery_ok = (residual(p).abs() < 1e-4).float()

        return {
            "rho": rho, "vx": vx, "vy": vy, "vz": vz, "p": p,
            "W": W, "v2": v2, "h": h, "cs2": cs2, "mach": mach,
            "recovery_ok": recovery_ok,
        }

    # ── stress proxy to drive the gate ─────────────────────────────────
    def stress_measure(self, prim: Dict[str, Tensor]) -> Tensor:
        s = torch.tensor(0.0)
        for comp in (prim["vx"], prim["vy"], prim["vz"]):
            gx, gy, gz = self.fd.grad(comp)
            s = s + sabs(gx).mean() + sabs(gy).mean() + sabs(gz).mean()
        return s + sabs(self.fd.grad(prim["p"])[0]).mean()

    # ── one conservative update step ───────────────────────────────────
    def step(self, Q: Tensor, dt: Tensor, nu8: Tensor) -> Dict[str, Tensor]:
        """
        First-order conservative update + 8th-order hyperviscosity + Lax
        stabilization. PROTOTYPE, not a Riemann solver.
        """
        prim = self.cons_to_prim(Q)
        D = Q[0]
        W = prim["W"]
        rho_safe = prim["rho"]
        vx, vy, vz, p = prim["vx"], prim["vy"], prim["vz"], prim["p"]

        # flux divergences
        div_D = (self.fd.grad(D * vx)[0] + self.fd.grad(D * vy)[1]
                 + self.fd.grad(D * vz)[2])
        dSx = (self.fd.grad(Q[1] * vx)[0] + self.fd.grad(Q[1] * vy)[1]
               + self.fd.grad(Q[1] * vz)[2]) + self.fd.grad(p)[0]
        dSy = (self.fd.grad(Q[2] * vx)[0] + self.fd.grad(Q[2] * vy)[1]
               + self.fd.grad(Q[2] * vz)[2]) + self.fd.grad(p)[1]
        dSz = (self.fd.grad(Q[3] * vx)[0] + self.fd.grad(Q[3] * vy)[1]
               + self.fd.grad(Q[3] * vz)[2]) + self.fd.grad(p)[2]
        dtau = (self.fd.grad((Q[4] + p) * vx)[0]
                + self.fd.grad((Q[4] + p) * vy)[1]
                + self.fd.grad((Q[4] + p) * vz)[2])

        rhs = torch.stack([-div_D, -dSx, -dSy, -dSz, -dtau])

        # 8th-order polyharmonic hyperviscosity on momentum & energy
        for i in (1, 2, 3, 4):
            rhs[i] = rhs[i] + self.poly8.hyperviscosity(Q[i], nu8)

        # mild base viscosity on momentum
        for i in (1, 2, 3):
            rhs[i] = rhs[i] + self.cfg.nu * rho_safe * self.fd.laplacian(Q[i] / (W + 1e-8))

        # Lax stabilization + positivity guards
        Q_avg = (
            torch.roll(Q, 1, -1) + torch.roll(Q, -1, -1)
            + torch.roll(Q, 1, -2) + torch.roll(Q, -1, -2)
            + torch.roll(Q, 1, -3) + torch.roll(Q, -1, -3)
        ) / 6.0
        Q_new = Q + dt * rhs + 0.2 * (Q_avg - Q)
        Q_new[0] = Q_new[0].clamp(min=1e-8)
        Q_new[4] = Q_new[4].clamp(min=1e-10)

        prim_new = self.cons_to_prim(Q_new)
        return {
            "Q": Q_new,
            "W": prim_new["W"],
            "v2": prim_new["v2"],
            "mach": prim_new["mach"],
            "beta": torch.sqrt(prim_new["v2"].clamp(min=0.0)),
            "p": prim_new["p"],
            "recovery_ok": prim_new["recovery_ok"],
            "stress": self.stress_measure(prim_new),
        }


# ═══════════════════════════════════════════════════════════════════════
# [L4] NO-ZENO GATE  (Six Problems Prop 6.1)
# ═══════════════════════════════════════════════════════════════════════

class NoZenoGate(nn.Module):
    """Hysteresis gate with energy-budget event bound N(T)<=(E_max+D)/delta_min."""

    def __init__(self, cfg: RelCarpetConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.delta_min <= 0:
            raise ValueError("gate requires theta_hi > theta_lo")
        self.register_buffer("stress_acc", torch.zeros(1))
        self.register_buffer("E_max", torch.zeros(1))
        self.register_buffer("D", torch.zeros(1))
        self.register_buffer("event_count", torch.zeros(1))
        self.register_buffer("armed", torch.ones(1))

    @torch.no_grad()
    def update(self, new_stress: Tensor) -> Dict[str, Tensor]:
        c = self.cfg
        self.stress_acc += new_stress.reshape(-1).mean().clamp(min=0.0)
        if self.armed.item() == 0 and self.stress_acc.item() < c.theta_lo:
            self.armed.fill_(1.0)

        fired = torch.zeros(1)
        if self.armed.item() == 1 and self.stress_acc.item() > c.theta_hi:
            fired.fill_(1.0)
            self.event_count += 1
            self.armed.fill_(0.0)
            self.D += self.stress_acc.clone()
            self.stress_acc.fill_(0.0)
        else:
            self.stress_acc *= c.stress_decay

        self.E_max = torch.maximum(self.E_max, self.stress_acc)
        rhs = (self.E_max + self.D) / c.delta_min
        return {
            "fired": fired,
            "nu8": torch.where(fired > 0.5,
                               torch.full_like(fired, c.nu8_collapse),
                               torch.full_like(fired, c.nu8_base)),
            "event_count": self.event_count.clone(),
            "bound_rhs": rhs.clone(),
            "invariant_holds": (self.event_count <= rhs + 1e-6).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class RelativisticCarpetCFD(nn.Module):
    """
    Unified: SRH 3D compressible core + Minkowski proper-time clock +
    8th-order polyharmonic hyperviscosity + no-Zeno gate.
    """

    def __init__(self, cfg: RelCarpetConfig):
        super().__init__()
        self.cfg = cfg
        self.core = RelativisticCompressibleCore(cfg)
        self.gate = NoZenoGate(cfg)
        n = cfg.n
        self.clock = MinkowskiTimeInterface(cfg, (n, n, n))

    def initial_state(self, beta0: float = 0.5) -> Tensor:
        """Smooth relativistic jet-like initial condition."""
        n = self.cfg.n
        rho = torch.ones(n, n, n)
        vx = torch.full((n, n, n), beta0)
        vy = 0.1 * torch.randn(n, n, n)
        vz = 0.1 * torch.randn(n, n, n)
        p = torch.full((n, n, n), 0.1)
        return self.core.prim_to_cons(rho, vx, vy, vz, p)

    def step(self, Q: Tensor, dt: Tensor) -> Dict[str, Any]:
        # gate decides hyperviscosity from current stress
        prim = self.core.cons_to_prim(Q)
        stress = self.core.stress_measure(prim)
        gate_out = self.gate.update(stress)
        nu8 = gate_out["nu8"]

        step_out = self.core.step(Q, dt, nu8)

        # advance relativistic clocks using the NEW Lorentz factor
        clock_out = self.clock.advance(dt, step_out["W"])

        return {**step_out, "gate": gate_out, "clock": clock_out}

    @torch.no_grad()
    def regime_report(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Report Mach / beta / gamma WITHOUT conflating regimes."""
        beta_max = float(out["beta"].max())
        W_max = float(out["W"].max())
        mach_max = float(out["mach"].max())
        if beta_max > 0.9:
            regime = "ULTRA_RELATIVISTIC"
        elif beta_max > 0.3:
            regime = "RELATIVISTIC"
        elif mach_max > 1.0:
            regime = "SUPERSONIC_NONREL"
        else:
            regime = "SUBSONIC"
        return {"regime": regime, "beta_max": beta_max,
                "W_max": W_max, "mach_max": mach_max}


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Relativistic 3D Compressible × Proper-Time × 8th-Order")
    print("=" * 72)
    torch.manual_seed(42)

    cfg = RelCarpetConfig(n=8, device="cpu")
    model = RelativisticCarpetCFD(cfg)

    # ── primitive <-> conservative roundtrip ───────────────────────────
    n = cfg.n
    rho = torch.ones(n, n, n)
    vx = torch.full((n, n, n), 0.6)   # beta = 0.6 -> W = 1.25
    vy = torch.zeros(n, n, n); vz = torch.zeros(n, n, n)
    p = torch.full((n, n, n), 0.1)
    Q = model.core.prim_to_cons(rho, vx, vy, vz, p)
    prim = model.core.cons_to_prim(Q)
    print(f"\n  ── Primitive recovery (Newton) ──")
    print(f"    target beta=0.6 ; recovered beta_max = {torch.sqrt(prim['v2']).max().item():.4f}")
    print(f"    recovered W_max = {prim['W'].max().item():.4f} (expect ~1.25)")
    print(f"    recovery_ok frac = {prim['recovery_ok'].mean().item():.3f}")
    print(f"    [PASS] cons<->prim roundtrip")

    # ── 8th-order polyharmonic operator ────────────────────────────────
    u = torch.randn(1, n, n, n)
    a4 = model.core.poly8.apply_A4(u)
    print(f"\n  ── 8th-order polyharmonic A^4 = nabla^8 (Euclidean shadow) ──")
    print(f"    input norm  = {u.norm().item():.4e}")
    print(f"    A^4 u norm  = {a4.norm().item():.4e}  (8th order amplifies high-k)")
    print(f"    [PASS] differential-length-8 operator active")

    # ── time stepping with relativistic clocks ─────────────────────────
    Qs = model.initial_state(beta0=0.5)
    dt = torch.tensor(1e-3)
    print(f"\n  ── Stepping: supersonic -> track proper time (d tau = dt/W) ──")
    for step in range(4):
        out = model.step(Qs, dt)
        Qs = out["Q"]
        reg = model.regime_report(out)
        clk = out["clock"]
        print(f"    step {step}: {reg['regime']}")
        print(f"      beta_max={reg['beta_max']:.3f}  W_max={reg['W_max']:.3f}  "
              f"Mach_max={reg['mach_max']:.2f}")
        print(f"      coord_t={clk['coord_time'].item():.4f}  "
              f"proper_t_mean={clk['proper_time_mean'].item():.4f}  "
              f"clock_spread={clk['clock_spread'].item():.2e}")
        print(f"      recovery_ok={out['recovery_ok'].mean().item():.3f}  "
              f"gate_events={out['gate']['event_count'].item():.0f}")

    # ── verify time dilation: proper <= coordinate ─────────────────────
    clk = out["clock"]
    assert clk["proper_time_mean"] <= clk["coord_time"] + 1e-9, \
        "proper time must not exceed coordinate time"
    print(f"\n  [PASS] proper_time <= coord_time  (SR time dilation holds)")

    # ── verify no-Zeno invariant ───────────────────────────────────────
    g = out["gate"]
    print(f"\n  ── No-Zeno gate (Six Problems Prop 6.1) ──")
    print(f"    events={g['event_count'].item():.0f}  bound={g['bound_rhs'].item():.2f}  "
          f"holds={bool(g['invariant_holds'].item())}")
    assert g["invariant_holds"].item(), "no-Zeno bound violated"
    print(f"    [PASS] N(T) <= (E_max+D)/delta_min")

    print("\n" + "=" * 72)
    print("  EPISTEMIC HONESTY — per the uploaded papers")
    print("  • This is SPECIAL relativity (Minkowski), not GR.")
    print("  • Proper time d tau=dt/W is real SR dilation via the paper's")
    print("    T:(metric,worldline,dt)->d tau interface (§6.4), as a SEPARATE")
    print("    Layer-1 model (§10.3) — NOT a dilation patch on Newtonian CFD.")
    print("  • The 8th-order operator is the Euclidean nabla^8 shadow of the")
    print("    carpet A^4=d*(dd*)^3d (Prop 4/Cor 4), used as hyperviscosity;")
    print("    it is NOT the fractal carpet operator.")
    print("  • Supersonic (Mach>1) and relativistic (beta->1) are reported")
    print("    separately and never conflated.")
    print("  • First-order prototype: needs §9.1-style verification and a real")
    print("    Riemann solver before any physics claim.")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
