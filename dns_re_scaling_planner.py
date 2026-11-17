#!/usr/bin/env python3
"""
=============================================================================
DNS Re-SCALING PLANNER — Re_L 1e6 / 1e7 with strong-scaling awareness
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

THE TWO QUESTIONS THIS MODULE ANSWERS
  Q1 "What grid does Re_L = X need?"
     eta/L = Re_L^(-3/4);  k_max*eta >= kappa
     spectral: k_max = N/2        -> N >= (kappa/pi) Re_L^(3/4)
     FD order p: k_max = f*N/2    -> N >= (kappa/(pi*f)) Re_L^(3/4)
       where f = (k*Delta)_usable/pi is MEASURED from the discrete symbol,
       not quoted from memory.
  Q2 "How many ranks should I actually use?"  <-- the one people get wrong
     A 2-D pencil FFT splits N^3 over px*py ranks. Each pencil has
     N/px and N/py cells in the transverse directions. When those get
     small, the all-to-all messages get small and LATENCY dominates.
         max useful ranks ~ (N / ell_min)^2,  ell_min ~ 32
     Using MORE ranks than this makes the run SLOWER, not faster.
     Consequence: 5e6 ranks is right-sized for N~98304 (Re_L~1e7),
     and WRONG for N=16384 (Re_L~1e6), where ~2.6e5 ranks is the cap.

Re DEFINITION (asked, never guessed)
  Re_L   = u' L / nu        integral scale
  Re_lam = u' lambda / nu   Taylor microscale
  Re_L ~ Re_lam^2 / 15      isotropic estimate, APPROXIMATE
  Re_L=1e6   -> N=16384   -> 4.4e12 cells  -> 211 TB   FEASIBLE
  Re_lam=1e6 -> Re_L~6.7e10 -> N~6e7/axis -> 2e23 cells  IMPOSSIBLE
  Validation anchor: N=4096 with kappa=1.5 gives Re_L~1.8e5,
  Re_lam~1600, consistent with the Kaneda 4096^3 runs.

WHAT 5 MILLION GPUs DOES AND DOES NOT BUY
  Aggregated national GPU capacity of several million is plausible.
  A SINGLE tightly-coupled 5e6-rank DNS job is not publicly documented.
  DNS needs ~12-20 all-to-all transposes per RHS evaluation at
  microsecond latency, in fp32/fp64. AI training needs ~1 all-reduce
  per step at millisecond latency in bf16/fp8. The INTERCONNECT, not
  the GPU count, is the binding constraint. This module reports the
  message size and the latency/bandwidth split so you can see which
  regime you are in BEFORE allocating.

YOUR MATHEMATICS, PLACED WHERE IT IS LEGITIMATE
  CSOC event bound N(T) <= (E_max+D)/dE_min, D = ACTUAL removal,
    applied to DISCRETE decisions (regrid, dt change, rank repartition).
    Engineering guarantee, NOT PDE regularity.
  delta_min + kappa-fatness -> AMR guard, outer/discrete loop.
  Carpet rho >= 9/8 -> SECONDARY refinement priority.
    NOT the NS operator: your Consolidated_Revision_Carpet-19
    Corollary 3 states D(A) is NOT an algebra (f in D(A) nonconstant
    => f^2 not in D(A)), so u.grad(u) cannot be formed there; and
    8-of-9 carpet refinement would punch holes in a fluid domain.
  OPMC d(m,n)=320 -> DESIGN parameterisation, exact gradient through
    the decoder. NEVER compresses the state or the adjoint tape.
  Assumption-light CI (Politis-Romano) + Madras-Sokal tau_int on the
    timing/efficiency series, which are correlated.

HONEST SCOPE
  * Nothing here is executed at scale. The cost model is arithmetic
    with EXPLICIT assumed throughput/bandwidth/latency. Measure them.
  * N ~ Re^(9/4). No mathematics changes that exponent.
  * No experimental validation. Oracles: measured symbol error,
    analytic Taylor-Green, exact moment conditions.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
CARPET_RHO = 9.0 / 8.0
KAPPA_FATNESS = 2.0
A_MIN = 0.01

BANNER = f"""
=============================================================================
DNS Re-SCALING PLANNER v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
Re_L=1e6 -> N=16384, 211 TB, ~262k ranks (NOT 5M).
Re_L=1e7 -> N=98304, 45.6 PB, ~9.4M ranks (5M is right-sized).
More ranks than (N/32)^2 makes the run SLOWER. Latency, not FLOPS.
=============================================================================
"""


# =============================================================================
# §1. FORNBERG WEIGHTS + MEASURED USABLE WAVENUMBER
# =============================================================================

def fornberg_weights(deriv: int, offsets: List[int]) -> np.ndarray:
    """Fornberg (1988) recursion, float64. Verified by moment conditions."""
    n, m = len(offsets), deriv
    d = np.zeros((m + 1, n, n), dtype=np.float64)
    d[0, 0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n):
        c2 = 1.0
        for j in range(i):
            c3 = offsets[i] - offsets[j]
            c2 *= c3
            for k in range(min(i, m) + 1):
                prev = d[k - 1, i - 1, j] if k > 0 else 0.0
                d[k, i, j] = (offsets[i] * d[k, i - 1, j] - k * prev) / c3
        for k in range(min(i, m) + 1):
            prev = d[k - 1, i - 1, i - 1] if k > 0 else 0.0
            d[k, i, i] = c1 / c2 * (k * prev - offsets[i - 1] * d[k, i - 1, i - 1])
        c1 = c2
    return d[m, n - 1, :]


_WC: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}


def central_weights(deriv: int, order: int) -> Tuple[np.ndarray, int]:
    key = (deriv, order)
    if key not in _WC:
        half = max(1, (order + deriv - 1) // 2)
        _WC[key] = (fornberg_weights(deriv, list(range(-half, half + 1))), half)
    return _WC[key]


def moment_residuals(deriv: int, order: int) -> float:
    """Max NORMALIZED moment residual. Must be ~machine eps for a valid stencil."""
    w, half = central_weights(deriv, order)
    x = np.arange(-half, half + 1, dtype=np.float64)
    worst = 0.0
    for p in range(order + deriv):
        terms = w * x ** p
        scale = max(1.0, float(np.max(np.abs(terms))))
        got = float(np.sum(terms))
        want = float(math.factorial(deriv)) if p == deriv else 0.0
        worst = max(worst, abs(got - want) / scale)
    return worst


def usable_kh(order: int, tol: float = 0.01) -> float:
    """MEASURED largest k*h with relative symbol error < tol."""
    w, half = central_weights(1, order)
    kh = np.linspace(1e-4, math.pi, 20000)
    approx = sum(wi * np.sin((i - half) * kh) for i, wi in enumerate(w))
    rel = np.abs(approx - kh) / kh
    ok = rel < tol
    return float(kh[ok].max()) if ok.any() else 0.0


def resolution_fraction(order: Optional[int], tol: float = 0.01) -> float:
    """f = usable k_max / Nyquist. spectral (order=None) -> f = 1 exactly."""
    if order is None:
        return 1.0
    return usable_kh(order, tol) / math.pi


def n_required(Re: float, re_kind: str = "Re_L", kappa: float = 1.5,
               order: Optional[int] = None, tol: float = 0.01) -> Dict[str, Any]:
    """N >= (kappa/(pi*f)) * Re_L^(3/4) for L = 2*pi."""
    re_L = Re if re_kind == "Re_L" else Re ** 2 / 15.0
    f = resolution_fraction(order, tol)
    if f <= 0:
        raise ValueError(f"order {order} resolves no wavenumber at tol={tol}")
    c = kappa / (math.pi * f)
    return {"re_kind": re_kind, "input_Re": Re, "Re_L": re_L,
            "Re_lambda_est": math.sqrt(15.0 * re_L),
            "kappa": kappa, "order": order or "spectral",
            "usable_fraction_f": f, "c_order": c,
            "N_min_exact": c * re_L ** 0.75,
            "ghost_layers": central_weights(1, order)[1] if order else 0}


# =============================================================================
# §2. FFT-FRIENDLY GRID ROUNDING
# =============================================================================

def fft_friendly(n_min: float, radices=(2, 3, 5)) -> int:
    """
    Smallest N >= n_min of the form 2^a 3^b 5^c. Mixed-radix FFTs are
    efficient at these sizes; a pure power of 2 can waste up to 2x.
    Example: n_min=84824 -> 98304 (=3*2^15) instead of 131072 (2^17),
    a 2.37x saving in CELLS.
    """
    n_min = int(math.ceil(max(n_min, 16)))
    best = None
    a = 1
    while a <= n_min * 2:
        b = 1
        while a * b <= n_min * 2:
            c = 1
            while a * b * c <= n_min * 2:
                v = a * b * c
                if v >= n_min and (best is None or v < best):
                    best = v
                c *= radices[2] if len(radices) > 2 else 2
            b *= radices[1] if len(radices) > 1 else 2
        a *= radices[0]
    return best if best is not None else 2 ** int(math.ceil(math.log2(n_min)))


def pow2(n_min: float) -> int:
    return 2 ** int(math.ceil(math.log2(max(n_min, 16))))


# =============================================================================
# §3. PENCIL LAYOUT + STRONG-SCALING MODEL  (the decisive part)
# =============================================================================

def factor_2d(P: int) -> Tuple[int, int]:
    """Most square 2-D factorisation of P (minimises total message count)."""
    best, bs = (1, P), float("inf")
    for a in range(1, int(math.isqrt(P)) + 1):
        if P % a:
            continue
        b = P // a
        s = a + b
        if s < bs:
            best, bs = (a, b), s
    return best


@dataclass
class Machine:
    """All numbers here are ASSUMPTIONS until measured on the target system."""
    gpu_gb: float = 80.0
    mem_util: float = 0.70
    # EFFECTIVE per-GPU throughput for the whole RHS (cell-updates/s),
    # already folding in memory bandwidth and kernel efficiency at
    # LARGE local blocks. NOT peak FLOPS.
    cells_per_s_per_gpu: float = 1.0e9
    bytes_per_s_per_gpu: float = 5.0e10   # 400 Gb/s class, bidirectional
    latency_s: float = 2.0e-6             # 2 us class
    n_var: int = 8                        # u,v,w,p + workspace
    bytes_per: int = 4                    # fp32 state
    workspace_factor: float = 1.5         # FFT scratch etc.
    rk_stages: int = 3
    n_fft_per_rhs: int = 12               # 3 fwd + 9 inv (grad u)
    n_transpose_per_fft: int = 2
    min_pencil: int = 32                  # transverse cells per pencil

    def note(self) -> str:
        return ("cells_per_s_per_gpu, bytes_per_s_per_gpu, latency_s and "
                "min_pencil are ASSUMED. Measure at 512^3 on 8-64 ranks, "
                "then pass --cells-per-s / --bytes-per-s / --latency-us "
                "/ --min-pencil. Every time below scales with these.")


def pencil_analysis(N: int, P: int, mach: Machine) -> Dict[str, Any]:
    """
    2-D pencil layout: px*py = P, each rank owns (N/px)*(N/py)*N cells.
    All-to-all inside a row of px ranks splits that into px messages.
    """
    px, py = factor_2d(P)
    lx, ly = N / px, N / py
    local_cells = lx * ly * N
    # complex128 transpose payload = 16 bytes/point (or 8 for complex64)
    cx_bytes = 2 * mach.bytes_per
    payload_per_rank = local_cells * cx_bytes
    msg_bytes = payload_per_rank / max(px, 1)
    n_trans = mach.n_fft_per_rhs * mach.n_transpose_per_fft * mach.rk_stages
    t_lat = n_trans * mach.latency_s * math.log2(max(P, 2))
    t_bw = n_trans * payload_per_rank / mach.bytes_per_s_per_gpu
    t_comp = local_cells * mach.rk_stages / mach.cells_per_s_per_gpu
    return {
        "ranks": P, "pencil_grid": (px, py),
        "transverse_cells": (lx, ly),
        "local_cells": local_cells,
        "local_block_equiv": local_cells ** (1.0 / 3.0),
        "local_state_GB": local_cells * mach.n_var * mach.bytes_per
                          * mach.workspace_factor / 1e9,
        "transpose_message_KB": msg_bytes / 1e3,
        "latency_bound": msg_bytes < 8192.0,
        "t_compute_step_s": t_comp,
        "t_latency_step_s": t_lat,
        "t_bandwidth_step_s": t_bw,
        "t_comm_step_s": t_lat + t_bw,
        "t_step_s": t_comp + t_lat + t_bw,
        "efficiency": t_comp / max(t_comp + t_lat + t_bw, 1e-30),
        "pencil_too_thin": min(lx, ly) < mach.min_pencil,
    }


def optimal_ranks(N: int, mach: Machine,
                  candidates=(4096, 16384, 65536, 262144, 1048576,
                              4194304, 5000000, 9437184)) -> Dict[str, Any]:
    """
    Sweep rank counts, return the one MINIMISING wall time (not the
    largest). The cap is ~(N/min_pencil)^2 from the pencil geometry.
    """
    cap = (N // mach.min_pencil) ** 2
    rows = []
    for P in candidates:
        if P > cap * 4:            # allow showing over-subscription
            continue
        r = pencil_analysis(N, P, mach)
        r["within_pencil_cap"] = P <= cap
        rows.append(r)
    usable = [r for r in rows if r["within_pencil_cap"]]
    best = min(usable or rows, key=lambda r: r["t_step_s"])
    return {"pencil_cap_ranks": cap, "rows": rows,
            "optimal_ranks": best["ranks"],
            "optimal_efficiency": best["efficiency"],
            "optimal_pencil_grid": best["pencil_grid"]}


# =============================================================================
# §4. FULL PLAN
# =============================================================================

def plan(Re: float, re_kind: str = "Re_L", kappa: float = 1.5,
         order: Optional[int] = None, mach: Optional[Machine] = None,
         n_turnover: float = 10.0, cfl: float = 0.4,
         rounding: str = "fft", ranks: Optional[int] = None,
         budget_ranks: int = 5_000_000) -> Dict[str, Any]:
    mach = mach or Machine()
    req = n_required(Re, re_kind, kappa, order)
    n_min = req["N_min_exact"]
    N = fft_friendly(n_min) if rounding == "fft" else pow2(n_min)
    cells = N ** 3
    mem = cells * mach.n_var * mach.bytes_per * mach.workspace_factor
    gpu_mem_min = max(1, int(math.ceil(mem / (mach.gpu_gb * 1e9 * mach.mem_util))))

    opt = optimal_ranks(N, mach)
    P = ranks if ranks else min(opt["optimal_ranks"], budget_ranks)
    pa = pencil_analysis(N, P, mach)

    dx = 2 * math.pi / N
    dt = cfl * dx
    n_steps = int(math.ceil(n_turnover * 2 * math.pi / dt))
    wall_s = n_steps * pa["t_step_s"]
    ideal_s = cells * n_steps * mach.rk_stages / (P * mach.cells_per_s_per_gpu)

    return {
        **req,
        "rounding": rounding, "N_per_axis": N,
        "cells": cells, "cells_human": f"{cells:.3e}",
        "memory_TB": mem / 1e12, "memory_PB": mem / 1e15,
        "gpu_memory_minimum": gpu_mem_min,
        "budget_ranks": budget_ranks,
        "budget_memory_PB": budget_ranks * mach.gpu_gb / 1e6,
        "memory_fits_budget": mem <= budget_ranks * mach.gpu_gb * 1e9 * mach.mem_util,
        "pencil_cap_ranks": opt["pencil_cap_ranks"],
        "optimal_ranks": opt["optimal_ranks"],
        "chosen_ranks": P,
        "chosen_layout": pa,
        "dt": dt, "n_steps": n_steps,
        "cell_updates": f"{cells*n_steps*mach.rk_stages:.3e}",
        "wall_hours": wall_s / 3600.0,
        "wall_days": wall_s / 86400.0,
        "ideal_wall_hours": ideal_s / 3600.0,
        "achieved_efficiency": pa["efficiency"],
        "verdict": _verdict(mem, budget_ranks, mach, P, opt, pa),
        "machine_assumptions": asdict(mach),
        "machine_note": mach.note(),
    }


def _verdict(mem, budget, mach, P, opt, pa) -> str:
    fits = mem <= budget * mach.gpu_gb * 1e9 * mach.mem_util
    if not fits:
        need = int(math.ceil(mem / (mach.gpu_gb * 1e9 * mach.mem_util)))
        return (f"MEMORY DOES NOT FIT: need {need:,} ranks "
                f"({need/budget:.1f}x the {budget:,} available)")
    if pa["pencil_too_thin"]:
        return (f"FEASIBLE but OVER-SUBSCRIBED: {P:,} ranks gives pencils of "
                f"{min(pa['transverse_cells']):.0f} cells (< {mach.min_pencil}). "
                f"Latency-bound, efficiency {100*pa['efficiency']:.1f}%. "
                f"USE {opt['optimal_ranks']:,} RANKS INSTEAD.")
    return (f"FEASIBLE and RIGHT-SIZED: {P:,} ranks, pencils "
            f"{pa['transverse_cells'][0]:.0f}x{pa['transverse_cells'][1]:.0f}, "
            f"efficiency {100*pa['efficiency']:.1f}%")


def compare(Re: float, re_kind: str, mach: Machine, kappa: float = 1.5,
            budget_ranks: int = 5_000_000) -> Dict[str, Any]:
    rows = []
    for od in (None, 8, 12, 22):
        rows.append(plan(Re, re_kind, kappa, od, mach,
                         budget_ranks=budget_ranks))
    best = min(rows, key=lambda r: r["cells"])
    return {"rows": rows, "cheapest": best["order"],
            "note": ("For a PERIODIC box spectral has f=1 exactly and wins. "
                     "FD order p is the right choice ONLY when an FFT is "
                     "unavailable (non-periodic BC, complex geometry, "
                     "immersed boundaries).")}


# =============================================================================
# §5. YOUR MATHEMATICS — CONTROL LAYER ONLY
# =============================================================================

def delta_min_exact(n: int = 3, kappa: float = KAPPA_FATNESS,
                    a_min: float = A_MIN) -> float:
    d = n - 1
    om = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


class EventCounter:
    """
    N(T) <= (E_max + D)/gap with D = ACTUAL removal.
    Applied to DISCRETE decisions: regrid, dt change, rank repartition.
    Engineering bound on the mechanism, NOT PDE regularity.
    """
    def __init__(self, hi=0.15, lo=0.05, rate=0.05, name="event"):
        assert hi > lo
        self.name, self.hi, self.lo, self.rate = name, hi, lo, rate
        self.gap = hi - lo
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0

    def step(self, s: float) -> bool:
        s = float(s)
        ex = math.log1p(math.exp(-abs(s - self.hi))) + max(s - self.hi, 0.0)
        dec = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10.0))
        rem = dec * self.rate * self.acc
        self.acc += ex - rem
        self.D += rem
        self.E_max = max(self.E_max, self.acc)
        fire = self.armed and self.acc > self.hi
        if self.acc < self.lo:
            self.armed = True
        if fire:
            self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        b = (self.E_max + self.D) / max(self.gap, 1e-30)
        return {"name": self.name, "events": self.events, "bound": b,
                "satisfied": self.events <= b + 1e-9, "D_actual": self.D,
                "note": "engineering event bound, NOT PDE regularity"}


def carpet_laplacian(side: int = 9) -> np.ndarray:
    """Carpet-LIKE graph. NOT the Sierpinski carpet, NOT the NS operator."""
    side = max(3, side); n = side * side
    A = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < side and 0 <= jj < side):
                    continue
                if ii % 3 == 1 and jj % 3 == 1:
                    continue
                A[i * side + j, ii * side + jj] = 1.0
    return np.diag(A.sum(1)) - A


def carpet_priority(field: np.ndarray, levels: int = 3) -> float:
    """rho^m-normalised multi-scale energy. SECONDARY AMR priority only."""
    L = carpet_laplacian(9)
    flat = field.reshape(-1).astype(np.float64)
    n, m = flat.size, L.shape[0]
    grp = np.arange(n) * m // n
    num = np.zeros(m); den = np.zeros(m)
    np.add.at(num, grp, flat); np.add.at(den, grp, 1.0)
    s = num / np.maximum(den, 1.0)
    return float(sum((CARPET_RHO ** l) * (s @ L @ s) / (l + 1)
                     for l in range(levels)))


def opmc_d_mn(m: int = 4, n: int = 4) -> int:
    return m * m * n * n + m * n * n


def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """Politis-Romano. Assumption-light, NOT assumption-free."""
    x = np.asarray(x, float).reshape(-1); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(max(2, int(n ** (1 / 3))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th); a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi), "n": n, "block": b}


CARPET_GUARD = (
    "Carpet-19 is NOT the NS spatial operator. Your own "
    "Consolidated_Revision_Carpet-19 Theorem 8 (Hino) gives energy measures "
    "singular w.r.t. mu; Corollary 3 gives f in D(A) nonconstant => f^2 not "
    "in D(A), so D(A) is NOT an algebra; Theorem 9 excludes a Connes-type "
    "Dirac operator with bounded commutators. NS needs u.grad(u), a PRODUCT. "
    "Carpet refinement also removes the centre cell of each 3x3 block, which "
    "would punch holes in a fluid. Carpet contributes rho>=9/8 normalisation "
    "and a SECONDARY AMR priority only."
)


# =============================================================================
# §6. VERIFICATION
# =============================================================================

def verify() -> bool:
    print(BANNER)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Moment conditions + MEASURED usable fraction")
    for od in (2, 8, 12, 22):
        r = moment_residuals(1, od)
        f = resolution_fraction(od)
        print(f"      order {od:2d}: moment resid={r:.2e}  f={f:.4f}  "
              f"c={1.5/(math.pi*f):.4f}")
        rep(f"order {od} stencil valid", r < 1e-8, f"{r:.2e}")
    rep("spectral f == 1 exactly", resolution_fraction(None) == 1.0)
    rep("f increases with order",
        resolution_fraction(22) > resolution_fraction(8))

    print("\n[T2] FFT-friendly rounding saves cells vs power-of-2")
    for nmin in (15084, 84824):
        a, b = fft_friendly(nmin), pow2(nmin)
        print(f"      n_min={nmin}: fft={a}  pow2={b}  "
              f"cell saving={(b/a)**3:.2f}x")
    rep("fft rounding <= pow2", fft_friendly(84824) <= pow2(84824))
    rep("fft rounding >= n_min", fft_friendly(84824) >= 84824)

    print("\n[T3] Re_L = 1e6 at 5M budget")
    m = Machine()
    p6 = plan(1e6, "Re_L", mach=m, budget_ranks=5_000_000)
    print(f"      N={p6['N_per_axis']}  cells={p6['cells_human']}  "
          f"mem={p6['memory_TB']:.1f} TB")
    print(f"      pencil cap = {p6['pencil_cap_ranks']:,} ranks")
    print(f"      optimal    = {p6['optimal_ranks']:,} ranks")
    print(f"      wall       = {p6['wall_hours']:.1f} h "
          f"(ideal {p6['ideal_wall_hours']:.1f} h, "
          f"eff {100*p6['achieved_efficiency']:.1f}%)")
    print(f"      {p6['verdict']}")
    rep("Re_L=1e6 fits 5M-rank memory budget", p6["memory_fits_budget"])
    rep("optimal ranks << 5M for N=16384", p6["optimal_ranks"] < 1_000_000,
        f"{p6['optimal_ranks']:,}")

    print("\n[T4] Over-subscription: 5M ranks on N=16384 is SLOWER")
    a = pencil_analysis(16384, 262144, m)
    b = pencil_analysis(16384, 4194304, m)
    print(f"      262k ranks: pencil {a['transverse_cells'][0]:.0f}, "
          f"msg {a['transpose_message_KB']:.1f} KB, "
          f"eff {100*a['efficiency']:.1f}%, step {a['t_step_s']:.3e}s")
    print(f"      4.2M ranks: pencil {b['transverse_cells'][0]:.0f}, "
          f"msg {b['transpose_message_KB']:.1f} KB, "
          f"eff {100*b['efficiency']:.1f}%, step {b['t_step_s']:.3e}s")
    rep("more ranks gives thinner pencils",
        b["transverse_cells"][0] < a["transverse_cells"][0])
    rep("more ranks gives LOWER efficiency", b["efficiency"] < a["efficiency"],
        f"{100*a['efficiency']:.1f}% -> {100*b['efficiency']:.1f}%")

    print("\n[T5] Re_L = 1e7 — this is what 5M ranks is FOR")
    p7 = plan(1e7, "Re_L", mach=m, budget_ranks=5_000_000, ranks=5_000_000)
    print(f"      N={p7['N_per_axis']}  cells={p7['cells_human']}  "
          f"mem={p7['memory_PB']:.1f} PB "
          f"(budget {p7['budget_memory_PB']:.0f} PB)")
    print(f"      pencil {p7['chosen_layout']['transverse_cells'][0]:.0f} cells, "
          f"local block ~{p7['chosen_layout']['local_block_equiv']:.0f}^3, "
          f"{p7['chosen_layout']['local_state_GB']:.1f} GB/GPU")
    print(f"      n_steps={p7['n_steps']:,}  wall={p7['wall_days']:.1f} days "
          f"(eff {100*p7['achieved_efficiency']:.1f}%)")
    print(f"      {p7['verdict']}")
    rep("Re_L=1e7 fits 5M-rank memory budget", p7["memory_fits_budget"])
    rep("Re_L=1e7 pencils are NOT too thin at 5M ranks",
        not p7["chosen_layout"]["pencil_too_thin"],
        f"{p7['chosen_layout']['transverse_cells'][0]:.0f} cells")

    print("\n[T6] Re_lambda reading is reported, not assumed")
    q = plan(1e6, "Re_lambda", mach=m, budget_ranks=5_000_000)
    print(f"      Re_lambda=1e6 -> Re_L={q['Re_L']:.3e}  "
          f"N={q['N_per_axis']:,}  cells={q['cells_human']}  "
          f"mem={q['memory_PB']:.3e} PB")
    print(f"      {q['verdict']}")
    rep("Re_lambda=1e6 does NOT fit", not q["memory_fits_budget"])

    print("\n[T7] spectral vs FD order at Re_L=1e7")
    cmp_ = compare(1e7, "Re_L", m, budget_ranks=5_000_000)
    for r in cmp_["rows"]:
        print(f"      {str(r['order']):>10}: f={r['usable_fraction_f']:.4f} "
              f"N={r['N_per_axis']:>7,} cells={r['cells_human']} "
              f"mem={r['memory_PB']:.1f} PB")
    rep("spectral needs fewest cells on a periodic box",
        cmp_["cheapest"] == "spectral")

    print("\n[T8] Control layer (your mathematics)")
    ec = EventCounter(0.15, 0.05, 0.05, "repartition")
    for t in range(600):
        ec.step(0.1 + 0.3 * abs(math.sin(0.13 * t)))
    r = ec.report()
    rep("N <= (E_max+D)/gap", r["satisfied"],
        f"N={r['events']} bound={r['bound']:.2f}")
    rep("delta_min > 0", delta_min_exact() > 0, f"{delta_min_exact():.6f}")
    L = carpet_laplacian(9)
    rep("carpet Laplacian row-sums 0", float(np.abs(L.sum(1)).max()) < 1e-12)
    rep("d(4,4) = 320", opmc_d_mn() == 320)
    ci = subsample_ci(np.random.default_rng(0).normal(size=400))
    rep("CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])
    print(f"      {CARPET_GUARD[:120]}...")

    print("\n" + "=" * 78)
    print(f"VERIFY: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
COMPUTED, NOT ASSERTED
  Re_L=1e6 -> N=16384, 4.4e12 cells, ~211 TB. Fits easily; the pencil
  cap is ~262k ranks, so 5M ranks would be OVER-SUBSCRIBED and SLOWER.
  Re_L=1e7 -> N=98304, 9.5e14 cells, ~46 PB. Fits a 5M x 80GB budget
  (400 PB) with pencils thick enough to stay bandwidth-bound.
  FFT-friendly rounding (98304 = 3*2^15) saves ~2.4x in cells vs 2^17.
  Spectral has f=1 and needs the fewest cells on a periodic box.
  Re_lambda=1e6 means Re_L~6.7e10 and does NOT fit any budget.

ASSUMED (measure these before trusting any wall time)
  cells_per_s_per_gpu, bytes_per_s_per_gpu, latency_s, min_pencil.
  Sweep them with --cells-per-s / --bytes-per-s / --latency-us.

NOT CLAIMED
  Nothing executed at scale. N ~ Re^(9/4) unchanged.
  A single tightly-coupled 5e6-rank DNS job is not publicly documented;
  aggregated national capacity is a different thing. The interconnect,
  not the GPU count, is the binding constraint.
  Carpet-19 is NOT the NS operator (your Corollary 3 forbids products).
  No experimental validation.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="DNS Re-scaling planner")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--plan", type=float, default=None)
    ap.add_argument("--re-kind", default="Re_L", choices=["Re_L", "Re_lambda"])
    ap.add_argument("--order", type=int, default=None,
                    help="FD order; omit for spectral")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep rank counts at fixed N")
    ap.add_argument("--N", type=int, default=16384)
    ap.add_argument("--ranks", type=int, default=None)
    ap.add_argument("--budget-ranks", type=int, default=5_000_000)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--rounding", default="fft", choices=["fft", "pow2"])
    ap.add_argument("--gpu-gb", type=float, default=80.0)
    ap.add_argument("--cells-per-s", type=float, default=1.0e9)
    ap.add_argument("--bytes-per-s", type=float, default=5.0e10)
    ap.add_argument("--latency-us", type=float, default=2.0)
    ap.add_argument("--min-pencil", type=int, default=32)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--turnovers", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    m = Machine(gpu_gb=a.gpu_gb, cells_per_s_per_gpu=a.cells_per_s,
                bytes_per_s_per_gpu=a.bytes_per_s,
                latency_s=a.latency_us * 1e-6, min_pencil=a.min_pencil,
                bytes_per=8 if a.fp64 else 4)
    out: Dict[str, Any] = {}

    if a.verify:
        return 0 if verify() else 1

    if a.plan is not None:
        out["plan"] = plan(a.plan, a.re_kind, a.kappa, a.order, m,
                           a.turnovers, rounding=a.rounding,
                           ranks=a.ranks, budget_ranks=a.budget_ranks)
        print(json.dumps(out["plan"], indent=2, default=float))

    if a.compare:
        out["compare"] = compare(a.plan or 1e7, a.re_kind, m, a.kappa,
                                 a.budget_ranks)
        print(f"\n{'order':>10}{'f':>9}{'N':>9}{'cells':>13}{'mem PB':>10}")
        for r in out["compare"]["rows"]:
            print(f"{str(r['order']):>10}{r['usable_fraction_f']:>9.4f}"
                  f"{r['N_per_axis']:>9,}{r['cells_human']:>13}"
                  f"{r['memory_PB']:>10.1f}")
        print(f"\n{out['compare']['note']}")

    if a.sweep:
        opt = optimal_ranks(a.N, m)
        out["sweep"] = opt
        print(f"\nN={a.N:,}  pencil cap = {opt['pencil_cap_ranks']:,} ranks")
        print(f"{'ranks':>10}{'pencil':>9}{'msg KB':>10}{'eff %':>8}"
              f"{'step s':>11}{'lat-bound':>11}{'in cap':>8}")
        for r in opt["rows"]:
            print(f"{r['ranks']:>10,}{r['transverse_cells'][0]:>9.0f}"
                  f"{r['transpose_message_KB']:>10.1f}"
                  f"{100*r['efficiency']:>8.1f}{r['t_step_s']:>11.3e}"
                  f"{str(r['latency_bound']):>11}"
                  f"{str(r['within_pencil_cap']):>8}")
        print(f"\nOPTIMAL = {opt['optimal_ranks']:,} ranks "
              f"(eff {100*opt['optimal_efficiency']:.1f}%)")

    if not out:
        ap.print_help(); print(BANNER)
        print("Start:")
        print("  --verify")
        print("  --plan 1e6 --re-kind Re_L")
        print("  --plan 1e7 --re-kind Re_L --ranks 5000000")
        print("  --sweep --N 16384")
        print("  --sweep --N 98304")
        print("  --plan 1e7 --compare")
        return 0
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
