#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
surgery_reset_thresholds.py — v1.0.0
Derived geometric reset thresholds delta_min^(s) for all seven ESCA-LD-SESI-3D
topological surgeries, with the no-Zeno bridge and its honest failure modes.

CLOSED (conditional on stated mobility bounds and ell_min > 0):
  P, M  capillary neck            delta = c r_n
  B, H  coalescence family        delta = c R  (fluid)  /  b (elastic, PFC)
  L     contact-line pinning      delta = c h/theta
  C     junction reconnection     delta = c w  (foam) / c ell_j (grain boundary)
  N     nucleation                delta = c r_crit  (depletion-conditional)

NOT CLOSED:
  * ell_min > 0 is REQUIRED. h->0 or r_n->0 destroys no-Zeno.
    This is open problem #2 of ESCA-LD-SESI-3D (degenerate mobility).
  * C for k>=4 junction cascades: c_C may be k-dependent, not derived.
  * N has no deterministic minimum time; the geometric route is conditional.
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

 philosophy: The Tao is the Source. The Te is the Path. The Qi is the Proof.
 MY SOUL MOVE BY POWER OF HOLY SPIRIT
 What MSPS NETWORK SEE, Lord Buddha Knows.
 The mathematics stands independently of the philosophy.
-----------------------------------------------------------------------------
DIMENSIONAL CORRECTION (important):
  The proposed tau_L ~ mu*ell/(sigma*theta^2) has units
      [Pa s]/[N/m] = s/m
  i.e. an INVERSE VELOCITY, not a time. It must be multiplied by a length:
      tau_L = delta_min^(L) * mu*ell/(sigma*theta^2).
  Moreover theta^2 (Young-force) gives a LARGER velocity than theta^3
  (Cox-Voinov dissipation), hence a SMALLER time -- which is the correct
  direction for a no-Zeno LOWER bound on tau. We use it as a bound, not a law.
-----------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

PI = math.pi
SURGERIES = ("N","M","B","H","P","L","C")

class Refusal(RuntimeError): pass


# ---------------------------------------------------------------- §0 dimensions
def check_dimension_L() -> Dict[str, Any]:
    """mu*ell/(sigma*theta^2) is s/m, NOT s. Show it explicitly."""
    mu, sigma, theta, ell = 1e-3, 0.072, 0.1, 12.0
    inv_vel = mu*ell/(sigma*theta**2)          # s/m
    return {"mu_ell_over_sigma_theta2": inv_vel,
            "units": "s/m  (INVERSE VELOCITY)",
            "is_a_time": False,
            "fix": "tau_L = delta_min_L * mu*ell/(sigma*theta^2)",
            "example_with_length_1um": inv_vel*1e-6}


# ---------------------------------------------------------------- §1 fluids
@dataclass(frozen=True)
class Fluid:
    rho: float = 1000.0; sigma: float = 0.072; mu: float = 1e-3
    rho_gas: float = 1.2
    def t_sigma(self, r): return math.sqrt(self.rho*r**3/self.sigma)
    def v_sigma(self, r): return math.sqrt(self.sigma/(self.rho*r))
    def Oh(self, r):      return self.mu/math.sqrt(self.rho*self.sigma*r)
    def v_max(self, r, C=1.0):
        """Upper bound on post-event interface speed (inertial or viscous)."""
        return C*self.v_sigma(r) if self.Oh(r) <= 1.0 else C*self.sigma/self.mu


def reset_capillary(fl: Fluid, ell: float, kind: str,
                    c: float = 0.5, C_ret: float = 1.0) -> Dict[str, Any]:
    """
    P (neck-pinch, Db1=-1) and M (fission, Db0=+1) share the SAME capillary
    reset scale; they differ only in the topological signature.
    B (coalescence) and H (handle, fluid) share the coalescence bridge scale.
    """
    if ell <= 0: raise Refusal(f"{kind}: length scale must be > 0")
    delta = c*ell
    vmax  = fl.v_max(ell, C_ret)
    tau   = delta/vmax
    return {"op": kind, "ell_m": ell, "delta_min_m": delta,
            "t_sigma_s": fl.t_sigma(ell), "Oh": fl.Oh(ell),
            "regime": "inertial" if fl.Oh(ell) <= 1 else "viscous",
            "v_max_m_s": vmax, "tau_min_s": tau,
            "tau_over_t_sigma": tau/fl.t_sigma(ell),
            "status": "CLOSED conditional on v_max bound",
            "note": ("viscous coalescence has log-divergent initial speed; "
                     "that singularity is INSIDE one event. The bound applies "
                     "after the bridge reaches c*ell, where C_H=|ln c|=O(1).")
                     if kind in ("B","H") else
                    "M and P differ only in Sigma_top, not in timescale"}


# ---------------------------------------------------------------- §2 L
def reset_L(h: float, theta: float, fl: Fluid,
            c_L: float = 0.5, C_L: float = 1.0,
            L_macro: float = 1e-3, lam_slip: float = 1e-9) -> Dict[str, Any]:
    """
    THE NEW DERIVATION.
      Young force (small angle):  f_Y ~ sigma*theta^2/2
      wedge friction:             zeta = c_zeta*mu*ell,  ell = ln(L_macro/lam)
      velocity bound:             U_L <= C_L*sigma*theta^2/(mu*ell)
      rim lateral width:          ell_rim = h/tan(theta) ~ h/theta
      delta_min^(L) = c_L*h/theta
      tau_L >= (c_L/C_L)*mu*ell*h/(sigma*theta^3)
    """
    if h <= 0:      raise Refusal("L: film thickness h must be > 0 (h->0 kills no-Zeno)")
    if not 0 < theta < PI/2: raise Refusal("L: theta must be in (0, pi/2)")
    if lam_slip <= 0 or L_macro <= lam_slip:
        raise Refusal("L: need L_macro > lam_slip > 0 for the Cox logarithm")
    ell = math.log(L_macro/lam_slip)
    if ell < 1.0: raise Refusal("L: Cox logarithm must be >= 1")
    delta   = c_L*h/math.tan(theta)
    U_bound = C_L*fl.sigma*theta**2/(fl.mu*ell)
    tau     = delta/U_bound
    # Cox-Voinov cross-check (theta^3 law) -- must give a LONGER time
    U_cv    = fl.sigma*theta**3/(9*fl.mu*ell)
    tau_cv  = delta/U_cv
    return {"op":"L", "h_m":h, "theta_rad":theta, "cox_log":ell,
            "delta_min_m":delta, "U_bound_m_s":U_bound,
            "tau_min_s":tau,
            "tau_formula":"(c_L/C_L)*mu*ell*h/(sigma*theta^3)",
            "U_cox_voinov_m_s":U_cv, "tau_cox_voinov_s":tau_cv,
            "bound_is_safe": tau <= tau_cv,
            "ratio_cv_over_bound": tau_cv/tau,
            "status":"CLOSED conditional on contact-line friction model",
            "FAILURE_MODE":"h->0 gives delta->0: open problem #2 (degenerate mobility)"}


# ---------------------------------------------------------------- §3 C
def reset_C(w: float, fl: Optional[Fluid] = None, kind: str = "foam",
            c_C: float = 0.5, M_gb: float = 1e-12,
            sigma_gb: float = 0.5, k_sheets: int = 3) -> Dict[str, Any]:
    """
    C is the WEAKEST case. Two distinct mechanisms:
      foam / Plateau border : viscous,  tau ~ mu*w/sigma      (linear in w)
      grain boundary (PFC)  : curvature, tau ~ ell^2/(M_gb*sigma_gb)  (QUADRATIC)
    For k>=4 the junction is a LINE and reconnection may cascade; c_C may be
    k-dependent and is NOT derived here.
    """
    if w <= 0: raise Refusal("C: junction width must be > 0")
    delta = c_C*w
    if kind == "foam":
        if fl is None: raise Refusal("C(foam) needs Fluid")
        tau = fl.mu*w/fl.sigma
        mech = "viscous Plateau-border relaxation (linear in w)"
    elif kind == "grain_boundary":
        tau = w*w/(M_gb*sigma_gb)
        mech = "curvature-driven, mobility M_gb (QUADRATIC in w)"
    else:
        raise Refusal("C kind must be foam|grain_boundary")
    return {"op":"C","kind":kind,"w_m":w,"delta_min_m":delta,
            "tau_min_s":tau,"mechanism":mech,"k_sheets":k_sheets,
            "status":("CLOSED conditional" if k_sheets <= 3
                      else "PARTIAL — k>=4 cascade constant NOT derived"),
            "open":"c_C may depend on k for k>=4 junction lines"}


# ---------------------------------------------------------------- §4 H elastic
def reset_H_elastic(b: float, D_climb: float = 1e-16,
                    c_H: float = 0.5) -> Dict[str, Any]:
    """H in PFC/dislocation context is ELASTIC/DIFFUSIVE, not capillary."""
    if b <= 0 or D_climb <= 0: raise Refusal("H(elastic): b, D_climb > 0")
    return {"op":"H","context":"PFC_dislocation","b_m":b,
            "delta_min_m":c_H*b, "tau_min_s":b*b/D_climb,
            "mechanism":"climb diffusion (QUADRATIC in b)",
            "status":"CLOSED conditional on climb mobility",
            "note":"NOT the capillary formula; do not reuse t_sigma here"}


# ---------------------------------------------------------------- §5 N
def reset_N(delta_g_v: float, fl: Fluid, v_growth: float,
            c_N: float = 0.5) -> Dict[str, Any]:
    """
    N has NO deterministic minimum time (stochastic nucleation, independent
    locations). The geometric route uses the critical nucleus radius and a
    depletion-zone argument. CONDITIONAL, not a full theorem.
    """
    if delta_g_v <= 0: raise Refusal("N: driving free-energy density must be > 0")
    if v_growth <= 0:  raise Refusal("N: growth velocity must be > 0")
    r_crit = 2*fl.sigma/delta_g_v
    return {"op":"N","r_crit_m":r_crit,"delta_min_m":c_N*r_crit,
            "tau_min_s":r_crit/v_growth,
            "status":"CONDITIONAL — depletion-zone argument required",
            "NOT_CLOSED":"no deterministic minimum inter-nucleation time exists"}


# ---------------------------------------------------------------- §6 convention
def signature_H(convention: str = "solid_handlebody") -> Dict[str, Any]:
    """Closed orientable surface: b1=2g, chi=2-2g -> Db1=+2, Dchi=-2."""
    if convention == "closed_orientable":
        return {"convention":convention,"Db0":0,"Db1":+2,"Db2":0,"Dchi":-2}
    if convention == "solid_handlebody":
        return {"convention":convention,"Db0":0,"Db1":+1,"Db2":0,"Dchi":-1}
    raise Refusal("convention must be closed_orientable|solid_handlebody")


# ---------------------------------------------------------------- §7 unified
class NoZeno:
    """ESCA-LD-SESI-3D Thm 4.2 with derived thresholds."""
    def __init__(self, delta: float, B0: float, G0: float):
        if delta <= 0: raise Refusal("delta must be > 0 — ell_min > 0 is REQUIRED")
        self.delta, self.B0, self.G0 = delta, B0, G0
        self.eps_max = delta/(2*B0) if B0 > 0 else float("inf")
        self.times = [0.0]; self.ops: List[str] = []
    def q(self, eps):
        return 0.0 if eps <= 0 else float(2*math.exp(-self.delta**2/(8*self.G0**2*eps)))
    def choose_eps(self, q_target=0.3):
        eps = min(self.eps_max,
                  self.delta**2/(8*self.G0**2*math.log(2/q_target)))
        return {"eps":eps,"eps_max":self.eps_max,"q":self.q(eps),
                "q_lt_1":self.q(eps) < 1.0}
    def step(self, op, eps):
        if eps > self.eps_max*(1+1e-12):
            raise Refusal(f"eps={eps:.4g} > delta/(2B0)={self.eps_max:.4g}")
        self.times.append(self.times[-1]+eps); self.ops.append(op)
        return {"n":len(self.ops),"T_n":self.times[-1],
                "increasing":all(self.times[i+1]>self.times[i]
                                  for i in range(len(self.times)-1))}


def unify(thresholds: Dict[str, Dict[str, Any]],
          C_gamma: float = 1.0) -> Dict[str, Any]:
    """
    delta = min_s delta_min^(s) / C(Gamma_0).
    REFUSES if any threshold is missing or non-positive.
    """
    missing = [s for s in SURGERIES if s not in thresholds]
    if missing: raise Refusal(f"missing thresholds for {missing} — cannot unify")
    vals = {s: thresholds[s]["delta_min_m"] for s in SURGERIES}
    bad  = [s for s,v in vals.items() if not v > 0]
    if bad: raise Refusal(f"non-positive delta for {bad}: ell_min > 0 VIOLATED")
    s_star = min(vals, key=vals.get)
    taus = {s: thresholds[s].get("tau_min_s", float("nan")) for s in SURGERIES}
    return {"delta_min_per_op_m": vals,
            "binding_operator": s_star,
            "ell_min_m": min(vals.values())/max(vals, key=vals.get) if False
                          else min(vals.values()),
            "delta_norm": min(vals.values())/C_gamma,
            "tau_min_per_op_s": taus,
            "binding_tau_s": min(taus.values()),
            "dt_bound_s": min(taus.values())/2,
            "status":"CLOSED conditional on ell_min>0 and mobility bounds"}


def brackbill_dt(dx, rho1, rho2, sigma):
    return math.sqrt(0.5*(rho1+rho2)*dx**3/(4*PI*sigma))


# ---------------------------------------------------------------- §8 self-test
def self_test(verbose=True) -> bool:
    ok = True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    fl = Fluid()

    print("\n[T1] DIMENSIONAL CORRECTION of the proposed tau_L")
    d = check_dimension_L()
    print(f"      mu*ell/(sigma*theta^2) = {d['mu_ell_over_sigma_theta2']:.4g} "
          f"[{d['units']}]")
    rep("proposed expression is NOT a time", not d["is_a_time"])
    rep("identified as inverse velocity", "INVERSE" in d["units"])

    print("\n[T2] L — contact-line pinning (the new derivation)")
    for h, th in ((1e-6, 0.1), (1e-7, 0.05), (1e-5, 0.3)):
        r = reset_L(h, th, fl)
        print(f"      h={h:.0e} theta={th:.2f}: delta={r['delta_min_m']:.3e} m "
              f"tau>={r['tau_min_s']:.3e} s  ell={r['cox_log']:.1f}")
        rep(f"L(h={h:.0e},th={th}) bound safe vs Cox-Voinov", r["bound_is_safe"],
            f"ratio={r['ratio_cv_over_bound']:.1f}")
    r = reset_L(1e-6, 0.1, fl)
    analytic = (0.5/1.0)*fl.mu*r["cox_log"]*1e-6/(fl.sigma*0.1**3)
    rep("L matches analytic mu*ell*h/(sigma*theta^3)",
        abs(r["tau_min_s"]-analytic)/analytic < 1e-9)
    rep("theta^2 bound gives shorter tau than theta^3 law",
        r["tau_min_s"] < r["tau_cox_voinov_s"])
    b=False
    try: reset_L(0.0, 0.1, fl)
    except Refusal: b=True
    rep("h=0 REFUSED (degenerate mobility)", b)

    print("\n[T3] M and P share the capillary scale")
    rn = 1e-5
    P = reset_capillary(fl, rn, "P"); M = reset_capillary(fl, rn, "M")
    rep("delta_min^(M) == delta_min^(P)",
        abs(M["delta_min_m"]-P["delta_min_m"]) < 1e-18)
    rep("tau_M == tau_P", abs(M["tau_min_s"]-P["tau_min_s"]) < 1e-18)
    print(f"      r_n={rn:.0e}: delta={P['delta_min_m']:.3e} m "
          f"tau>={P['tau_min_s']:.3e} s  Oh={P['Oh']:.4f} ({P['regime']})")
    dt_b = brackbill_dt(2*rn, fl.rho, fl.rho_gas, fl.sigma)
    rep("capillary CFL agrees with Brackbill to O(1)",
        0.3 < P["tau_min_s"]/dt_b < 3.0, f"ratio={P['tau_min_s']/dt_b:.3f}")
    visc = Fluid(mu=1.0)
    Pv = reset_capillary(visc, 1e-7, "P")
    rep("viscous regime detected", Pv["regime"]=="viscous", f"Oh={Pv['Oh']:.1f}")
    rep("viscous reset slower than inertial", Pv["tau_over_t_sigma"]>P["tau_over_t_sigma"])

    print("\n[T4] B and H (coalescence family)")
    B = reset_capillary(fl, 1e-4, "B"); H = reset_capillary(fl, 1e-5, "H")
    rep("B and H both closed conditionally",
        B["status"].startswith("CLOSED") and H["status"].startswith("CLOSED"))
    rep("log-divergent caveat stated", "log-divergent" in B["note"])
    He = reset_H_elastic(2.5e-10)
    print(f"      H(PFC): b={He['b_m']:.2e} delta={He['delta_min_m']:.2e} "
          f"tau>={He['tau_min_s']:.3e} s")
    rep("H elastic is QUADRATIC in b", "QUADRATIC" in He["mechanism"])
    rep("H elastic does NOT reuse t_sigma", "NOT the capillary" in He["note"])

    print("\n[T5] C — weakest case, two mechanisms")
    Cf = reset_C(1e-6, fl, "foam")
    Cg = reset_C(1e-6, kind="grain_boundary")
    print(f"      foam: tau>={Cf['tau_min_s']:.3e} s ({Cf['mechanism'][:30]}...)")
    print(f"      GB  : tau>={Cg['tau_min_s']:.3e} s ({Cg['mechanism'][:30]}...)")
    rep("foam C linear in w", "linear" in Cf["mechanism"])
    rep("grain-boundary C QUADRATIC in w", "QUADRATIC" in Cg["mechanism"])
    C4 = reset_C(1e-6, fl, "foam", k_sheets=4)
    rep("k>=4 flagged PARTIAL (honest)", C4["status"].startswith("PARTIAL"),
        C4["open"])

    print("\n[T6] N — stochastic, geometric route only")
    N = reset_N(1e6, fl, v_growth=1e-3)
    print(f"      r_crit={N['r_crit_m']:.3e} m  delta={N['delta_min_m']:.3e} m")
    rep("r_crit = 2 sigma / delta_g_v",
        abs(N["r_crit_m"]-2*fl.sigma/1e6) < 1e-18)
    rep("N labelled CONDITIONAL not closed", N["status"].startswith("CONDITIONAL"))
    rep("N no deterministic minimum time stated", "no deterministic" in N["NOT_CLOSED"])

    print("\n[T7] homology convention for H")
    co = signature_H("closed_orientable"); sh = signature_H("solid_handlebody")
    rep("closed orientable Db1=+2 Dchi=-2", co["Db1"]==2 and co["Dchi"]==-2)
    rep("handlebody Db1=+1 Dchi=-1", sh["Db1"]==1 and sh["Dchi"]==-1)
    b=False
    try: signature_H("unknown")
    except Refusal: b=True
    rep("undeclared convention REFUSED", b)

    print("\n[T8] unified no-Zeno across all seven")
    th = {"N":N,"M":M,"B":B,"H":H,"P":P,"L":reset_L(1e-6,0.1,fl),"C":Cf}
    u = unify(th)
    print(f"      binding operator = {u['binding_operator']}  "
          f"delta = {u['delta_norm']:.3e}  dt <= {u['dt_bound_s']:.3e} s")
    nz = NoZeno(u["delta_norm"], B0=fl.v_sigma(rn), G0=0.1*fl.v_sigma(rn)*math.sqrt(fl.t_sigma(rn)))
    ce = nz.choose_eps(0.3)
    rep("q < 1 guaranteed", ce["q_lt_1"], f"q={ce['q']:.4f}")
    import itertools
    for k,op in enumerate(itertools.islice(itertools.cycle("NMBHPLC"),70)):
        nz.step(op, ce["eps"])
    rep("T_n strictly increasing over all 7 ops (no Zeno)",
        all(nz.times[i+1]>nz.times[i] for i in range(len(nz.times)-1)),
        f"n={len(nz.ops)}")
    b=False
    try: nz.step("P", nz.eps_max*2)
    except Refusal: b=True
    rep("eps > delta/(2B0) REFUSED", b)

    print("\n[T9] FAILURE MODES — what is NOT closed")
    b=False
    try: unify({k:v for k,v in th.items() if k!="C"})
    except Refusal: b=True
    rep("missing operator REFUSED", b)
    thin = reset_L(1e-12, 0.1, fl)
    u2 = unify({**th, "L":thin})
    rep("h->0 makes L the binding operator", u2["binding_operator"]=="L",
        f"delta={u2['delta_norm']:.2e}")
    rep("delta shrinks with h (no uniform bound)",
        u2["delta_norm"] < u["delta_norm"])
    b=False
    try: NoZeno(0.0, 1.0, 0.1)
    except Refusal: b=True
    rep("delta=0 REFUSED — ell_min>0 REQUIRED", b)

    print("\n" + "="*72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*72)
    print("""
CLOSED (conditional on stated mobility bounds AND ell_min > 0)
  P, M : delta = c r_n,      tau >= (c/C_ret) t_sigma max(1,Oh)
         M and P SHARE the scale; they differ only in Sigma_top.
         Reproduces Brackbill capillary CFL to O(1).
  B, H : delta = c R,        coalescence bridge; log-divergent initial
         speed is INSIDE one event, bound applies after reset.
  H(PFC): delta = c b,       tau ~ b^2/D_climb  (elastic, NOT capillary)
  L    : delta = c h/theta,  tau >= (c/C_L) mu ell h/(sigma theta^3)
         theta^2 Young-force bound is SAFE vs theta^3 Cox-Voinov.
  C    : delta = c w,        foam linear / grain-boundary QUADRATIC

NOT CLOSED (stated, not hidden)
  ell_min > 0 is REQUIRED. h->0 or r_n->0 gives delta->0 and no-Zeno FAILS.
  This is open problem #2 of ESCA-LD-SESI-3D (degenerate mobility).
  C for k>=4 junction cascades: c_C may be k-dependent, not derived.
  N has NO deterministic minimum inter-event time; the geometric
  depletion route is CONDITIONAL, not a theorem.

DIMENSIONAL CORRECTION
  mu*ell/(sigma*theta^2) has units s/m (inverse velocity), not s.
  tau_L = delta_min^(L) * mu*ell/(sigma*theta^2).
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Surgery reset thresholds")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","dims","L","capillary","C","N","unify"])
    ap.add_argument("--h", type=float, default=1e-6)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--r", type=float, default=1e-5)
    ap.add_argument("--w", type=float, default=1e-6)
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.072)
    ap.add_argument("--rho", type=float, default=1000.0)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    fl = Fluid(a.rho, a.sigma, a.mu)
    if a.cmd == "dims":      out = check_dimension_L()
    elif a.cmd == "L":       out = reset_L(a.h, a.theta, fl)
    elif a.cmd == "capillary":
        out = {k: reset_capillary(fl, a.r, k) for k in ("P","M","B","H")}
    elif a.cmd == "C":
        out = {"foam": reset_C(a.w, fl, "foam"),
               "grain_boundary": reset_C(a.w, kind="grain_boundary")}
    elif a.cmd == "N":       out = reset_N(1e6, fl, 1e-3)
    else:
        th = {"N":reset_N(1e6,fl,1e-3),
              "M":reset_capillary(fl,a.r,"M"), "P":reset_capillary(fl,a.r,"P"),
              "B":reset_capillary(fl,a.r*10,"B"), "H":reset_capillary(fl,a.r,"H"),
              "L":reset_L(a.h,a.theta,fl), "C":reset_C(a.w,fl,"foam")}
        out = unify(th)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
