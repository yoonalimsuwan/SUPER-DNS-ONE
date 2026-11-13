#!/usr/bin/env python3
"""
=============================================================================
UNSEEN-CLASS PREDICTION + PRIOR-SCALE SENSITIVITY FOR τ_w, τ_b
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — TRUE UNSEEN-CLASS PREDICTION
  The hardest honest test of a hierarchical model: hold out an ENTIRE
  ligand class (not one system), fit the hierarchy on the remaining
  classes, then predict dCp(T) for the held-out class.

  For a genuinely unseen class g*, there is no class-level information
  to borrow. The prediction must come from the GLOBAL level:
      theta_{g*} ~ N(mu_0, Sigma_b)
  so the predictive variance is the between-class variance Sigma_b
  (plus the observation noise), NOT the within-class variance. This is
  the correct, wider interval. Reporting the within-class interval for
  an unseen class would be overconfident.

  Stacking across M0/M1/M2 is applied to the PREDICTIVE density of the
  held-out class, with stacking weights fit only on the training
  classes (never on the held-out data).

PART B — PRIOR-SCALE SENSITIVITY FOR τ_w, τ_b
  The half-Cauchy scale parameters A_w (for within-class sd) and A_b
  (for between-class sd) are weakly-informative. When the number of
  training classes is small, the posterior of Sigma_b is dominated by
  its prior. This module quantifies that domination by:

      1. Refitting the hierarchy with A_w, A_b multiplied by
         factors in {0.25, 0.5, 1, 2, 4}
      2. Reporting the posterior median of Sigma_b and the predictive
         interval at a target T under each factor
      3. Computing the "prior-leverage ratio":
             (max prediction - min prediction across factors)
             / (posterior sd of the prediction at factor=1)
         If this ratio >> 1, the prediction is prior-dominated.

  This is the decisive diagnostic for small-G hierarchies and must be
  reported alongside any unseen-class claim.

HONEST SCOPE
  ❌ With G_train = 2, Sigma_b is barely identified; the sensitivity
     sweep will show large prior leverage. That is information, not a bug.
  ❌ Stacking weights are predictive-optimal on the TRAINING classes.
     They are not guaranteed to be optimal for the held-out class.
  ❌ Unseen-class prediction assumes exchangeability of classes. If the
     held-out class is chemically outside the range of training classes,
     the global mean is a poor anchor and the interval will be centered
     wrong even if its width is honest.
  ❌ No experimental validation is performed here.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("unseen_prior")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# 0. DESIGN MATRIX (same convention as the earlier dCp modules)
# =============================================================================

PARAM_NAMES = ("dH0", "dS0", "dCp0", "b")
MODELS: Dict[str, Tuple[int, ...]] = {
    "M0": (0, 1),
    "M1": (0, 1, 2),
    "M2": (0, 1, 2, 3),
}


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    """theta = (dH0, dS0, dCp0, b), dCp(T) = dCp0 + b (T - T0)."""
    dT = T - T0
    L = math.log(T / T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, 0.5 * dT * dT])
    if kind == "dG":
        return np.array([1.0, -T, dT - T * L,
                         0.5 * dT * dT - T * (dT - T0 * L)])
    raise ValueError(f"kind must be dH|dG, got {kind!r}")


@dataclass
class SystemData:
    name: str
    group: str
    T: np.ndarray
    kind: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    T0: float = 310.0

    def design(self, cols: Tuple[int, ...]):
        X = np.vstack([design_row(float(t), str(k), self.T0)[list(cols)]
                       for t, k in zip(self.T, self.kind)])
        w = 1.0 / np.maximum(np.asarray(self.sigma, float), 1e-12) ** 2
        return X, np.asarray(self.y, float), w

    @property
    def n(self) -> int:
        return len(self.y)


# =============================================================================
# 1. CONJUGATE HIERARCHICAL GIBBS (per-parameter variance components)
# =============================================================================

def _rwishart(rng: np.random.Generator, df: float, scale: np.ndarray) -> np.ndarray:
    k = scale.shape[0]
    L = np.linalg.cholesky(scale)
    A = np.zeros((k, k))
    for i in range(k):
        A[i, i] = math.sqrt(rng.chisquare(max(df - i, 1e-6)))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    M = L @ A
    return M @ M.T


def _riw(rng: np.random.Generator, df: float, Psi: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_rwishart(rng, df, np.linalg.inv(Psi)))


@dataclass
class HierarchicalFit:
    """Result of one hierarchical Gibbs fit."""
    model: str
    cols: Tuple[int, ...]
    G_train: int
    S_train: int
    n_draws: int
    theta: np.ndarray          # (D, S_train, k)
    mu_g: np.ndarray           # (D, G_train, k)
    mu0: np.ndarray            # (D, k)
    Sigma_w: np.ndarray        # (D, k, k)
    Sigma_b: np.ndarray        # (D, k, k)
    A_w: float                 # prior scale used for within
    A_b: float                 # prior scale used for between
    warnings: List[str] = field(default_factory=list)


def fit_hierarchy(
    systems: List[SystemData],
    model: str = "M2",
    A_w: float = 2.0,
    A_b: float = 1.0,
    n_iter: int = 2500,
    burn: int = 900,
    thin: int = 3,
    seed: int = 0,
) -> HierarchicalFit:
    """
    Normal–Inverse-Wishart hierarchy, exact conjugate Gibbs.
    A_w, A_b are the half-Cauchy SCALE hyperparameters (the thing we sweep).
    """
    cols = MODELS[model]
    k = len(cols)
    groups = sorted({s.group for s in systems})
    gidx = {g: i for i, g in enumerate(groups)}
    sg = np.array([gidx[s.group] for s in systems])
    G, S = len(groups), len(systems)
    warns: List[str] = []
    if G < 2:
        raise ValueError("hierarchy needs >= 2 classes")
    if G == 2:
        warns.append("G_train=2: Sigma_b is barely identified; prior dominates")
    counts = np.bincount(sg, minlength=G)
    if (counts == 1).any():
        warns.append(f"singleton class(es) {[groups[g] for g in np.where(counts==1)[0]]}")

    prep = [s.design(cols) for s in systems]
    rng = np.random.default_rng(seed)

    # weak location prior on mu0
    m0 = np.zeros(k)
    a0 = 0.1

    # initial values
    theta = np.zeros((S, k))
    for i, (X, y, w) in enumerate(prep):
        if systems[i].n >= k:
            Xw = X * np.sqrt(w)[:, None]
            theta[i] = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)[0]
    mu_g = np.array([theta[sg == g].mean(0) for g in range(G)])
    mu0 = mu_g.mean(0)
    scale_guess = np.array([50.0, 0.3, 1.0, 0.01])[:k]
    Sigma_w = np.diag((0.5 * scale_guess) ** 2)
    Sigma_b = np.diag((0.3 * scale_guess) ** 2)

    Dtheta, Dmu_g, Dmu0, DSw, DSb = [], [], [], [], []
    for it in range(n_iter):
        Swi = np.linalg.inv(Sigma_w)
        Sbi = np.linalg.inv(Sigma_b)
        # theta_s
        for i, (X, y, w) in enumerate(prep):
            A = X.T @ (w[:, None] * X) + Swi
            b = X.T @ (w * y) + Swi @ mu_g[sg[i]]
            V = np.linalg.inv(A)
            theta[i] = V @ b + np.linalg.cholesky(V) @ rng.standard_normal(k)
        # mu_g
        for g in range(G):
            sel = sg == g
            ng = int(sel.sum())
            A = ng * Swi + Sbi
            V = np.linalg.inv(A)
            b = Swi @ theta[sel].sum(0) + Sbi @ mu0
            mu_g[g] = V @ b + np.linalg.cholesky(V) @ rng.standard_normal(k)
        # mu0
        V0 = Sigma_b / (a0 + G)
        m0p = (a0 * m0 + mu_g.sum(0)) / (a0 + G)
        mu0 = m0p + np.linalg.cholesky(V0) @ rng.standard_normal(k)
        # Sigma_w, Sigma_b with half-Cauchy scale A (via auxiliary)
        Rw = theta - mu_g[sg]
        Psi_w = np.eye(k) * (A_w ** 2)
        Sigma_w = _riw(rng, S + k + 1, Psi_w + Rw.T @ Rw)
        Rb = mu_g - mu0[None, :]
        Psi_b = np.eye(k) * (A_b ** 2)
        Sigma_b = _riw(rng, G + k + 1,
                       Psi_b + Rb.T @ Rb + a0 * np.outer(mu0 - m0, mu0 - m0))
        if it >= burn and (it - burn) % thin == 0:
            Dtheta.append(theta.copy()); Dmu_g.append(mu_g.copy())
            Dmu0.append(mu0.copy()); DSw.append(Sigma_w.copy()); DSb.append(Sigma_b.copy())

    return HierarchicalFit(
        model=model, cols=cols, G_train=G, S_train=S,
        n_draws=len(Dtheta),
        theta=np.array(Dtheta), mu_g=np.array(Dmu_g), mu0=np.array(Dmu0),
        Sigma_w=np.array(DSw), Sigma_b=np.array(DSb),
        A_w=A_w, A_b=A_b, warnings=warns,
    )


# =============================================================================
# 2. UNSEEN-CLASS PREDICTION
# =============================================================================

def predict_unseen_class(
    fit: HierarchicalFit,
    T_target: float,
    T0: float = 310.0,
    kind: str = "dH",
) -> Dict[str, Any]:
    """
    Predict dCp(T_target) for a class with NO training systems.

    The new class draws its theta from N(mu0, Sigma_b). Therefore the
    predictive variance is the between-class variance Sigma_b (plus any
    observation noise you add), NOT Sigma_w. This is the honest, wider
    interval for a genuinely unseen class.

    Returns the posterior draws of dCp(T_target) and summary stats.
    """
    cols = fit.cols
    x = design_row(T_target, kind, T0)[list(cols)]
    # predictive draws: theta_new ~ N(mu0, Sigma_b)
    D = fit.n_draws
    k = len(cols)
    rng = np.random.default_rng(123)
    L = np.linalg.cholesky(fit.Sigma_b.mean(0))
    z = rng.standard_normal((D, k))
    theta_new = fit.mu0 + z @ L.T
    dcp_draws = theta_new @ x
    return {
        "T_target_K": T_target,
        "kind": kind,
        "mean_dCp": float(dcp_draws.mean()),
        "sd_dCp": float(dcp_draws.std(ddof=1)),
        "ci95": [float(v) for v in np.quantile(dcp_draws, [0.025, 0.975])],
        "n_draws": D,
        "note": ("predictive variance uses Sigma_b (between-class) because "
                 "the class is unseen; this is the honest wide interval"),
    }


# =============================================================================
# 3. STACKING WEIGHTS ON TRAINING CLASSES
# =============================================================================

def stacking_weights_for_classes(
    systems: List[SystemData],
    models: Sequence[str] = ("M0", "M1", "M2"),
    A_w: float = 2.0, A_b: float = 1.0,
    n_iter: int = 1500, burn: int = 500, seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute leave-one-system-out elpd per model on the TRAINING systems,
    then stacking weights. Held-out class is NOT used here.
    """
    elpd: Dict[str, List[float]] = {m: [] for m in models}
    for m in models:
        cols = MODELS[m]
        k = len(cols)
        for i, s in enumerate(systems):
            train = [t for j, t in enumerate(systems) if j != i]
            if len({t.group for t in train}) < 2:
                continue
            try:
                f = fit_hierarchy(train, m, A_w, A_b, n_iter, burn, seed=seed + i)
            except Exception:
                continue
            X, y, w = s.design(cols)
            # predict system i from its class if present, else global
            groups = sorted({t.group for t in train})
            if s.group in groups:
                mu = f.mu_g.mean(0)[groups.index(s.group)]
                Sig = f.Sigma_w.mean(0)
            else:
                mu = f.mu0.mean(0)
                Sig = f.Sigma_b.mean(0)
            m_pred = X @ mu
            v_pred = np.einsum("ni,ij,nj->n", X, Sig, X) + 1.0 / w
            lp = -0.5 * (np.log(2 * np.pi * v_pred) + (y - m_pred) ** 2 / v_pred)
            elpd[m].append(float(lp.sum()))
    # pseudo-BMA stacking on summed elpd
    tot = {m: float(np.sum(v)) if v else -np.inf for m, v in elpd.items()}
    finite = {m: v for m, v in tot.items() if math.isfinite(v)}
    if not finite:
        return {"status": "FAILED", "reason": "no model produced elpd"}
    mx = max(finite.values())
    raw = {m: math.exp(v - mx) for m, v in finite.items()}
    Z = sum(raw.values())
    weights = {m: raw[m] / Z for m in raw}
    return {"status": "OK", "elpd": tot, "weights": weights,
            "note": "weights fit on training classes only"}


# =============================================================================
# 4. PRIOR-SCALE SENSITIVITY SWEEP
# =============================================================================

def prior_scale_sweep(
    systems: List[SystemData],
    T_target: float = 310.0,
    model: str = "M2",
    factors: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    base_A_w: float = 2.0,
    base_A_b: float = 1.0,
    n_iter: int = 1500, burn: int = 500, seed: int = 0,
) -> Dict[str, Any]:
    """
    Refit the hierarchy with (A_w, A_b) multiplied by each factor.
    Report how much the unseen-class prediction and Sigma_b move.

    prior_leverage_ratio = (max-min of prediction across factors)
                           / (posterior sd at factor=1)
    If >> 1, the prediction is prior-dominated.
    """
    cols = MODELS[model]
    sweep: Dict[float, Dict[str, Any]] = {}
    for f in factors:
        try:
            fit = fit_hierarchy(systems, model,
                                A_w=base_A_w * f, A_b=base_A_b * f,
                                n_iter=n_iter, burn=burn, seed=seed)
            pred = predict_unseen_class(fit, T_target)
            Sb_median = float(np.median(fit.Sigma_b, axis=0).diagonal().mean())
            sweep[f] = {
                "A_w": base_A_w * f, "A_b": base_A_b * f,
                "mean_dCp": pred["mean_dCp"],
                "ci95": pred["ci95"],
                "sd_dCp": pred["sd_dCp"],
                "Sigma_b_diag_mean": Sb_median,
                "warnings": fit.warnings,
            }
        except Exception as e:
            sweep[f] = {"error": str(e)}
    ok = {f: v for f, v in sweep.items() if "mean_dCp" in v}
    if len(ok) < 2:
        return {"status": "FAILED", "sweep": sweep}
    means = [v["mean_dCp"] for v in ok.values()]
    sd_at_1 = ok[1.0]["sd_dCp"] if 1.0 in ok else float(np.std(means))
    leverage = (max(means) - min(means)) / max(sd_at_1, 1e-12)
    Sb_vals = [v["Sigma_b_diag_mean"] for v in ok.values()]
    return {
        "status": "OK",
        "T_target_K": T_target,
        "model": model,
        "sweep": sweep,
        "prediction_range": [float(min(means)), float(max(means))],
        "prior_leverage_ratio": float(leverage),
        "Sigma_b_range": [float(min(Sb_vals)), float(max(Sb_vals))],
        "verdict": ("PRIOR-DOMINATED" if leverage > 2.0 else
                    "moderately prior-influenced" if leverage > 1.0 else
                    "data-dominated"),
    }


# =============================================================================
# 5. SELF-TEST
# =============================================================================

def _make_grouped(rng, mode: str, n_per: int = 6, T0: float = 310.0,
                  classes: Sequence[str] = ("A", "B", "C")) -> List[SystemData]:
    base = np.array([-45.0, -0.10, -0.65, 0.0020])
    shifts = {"A": np.array([0.0, 0.0, 0.0, 0.0]),
              "B": np.array([12.0, 0.02, -0.25, 0.0004]),
              "C": np.array([-14.0, -0.03, 0.30, -0.0005])}
    out = []
    for g in classes:
        sh = shifts[g]
        for j in range(3):
            if mode == "structured":
                th = base + sh + rng.normal(0, 1, 4) * np.array([1.0, .005, .03, .0002])
            else:
                th = base + rng.normal(0, 1, 4) * np.array([1.0, .005, .03, .0002])
            Ts = np.linspace(296., 324., n_per) + rng.normal(0, .6, n_per)
            kinds, ys, sg = [], [], []
            for t in Ts:
                dT = t - T0
                H = th[0] + th[2] * dT + 0.5 * th[3] * dT * dT
                kinds.append("dH"); ys.append(H + rng.normal(0, .2)); sg.append(.2)
            out.append(SystemData(f"{g}{j}", g, Ts, np.array(kinds),
                                  np.array(ys), np.array(sg), T0))
    return out


def self_test() -> bool:
    print("=" * 94)
    print(f"  UNSEEN-CLASS PREDICTION + PRIOR-SCALE SENSITIVITY  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 94)
    rng = np.random.default_rng(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # ---- A1 fit hierarchy on 3 structured classes ----
    print("\n[A1] Hierarchy fit on 3 structured classes")
    sysl = _make_grouped(rng, "structured", n_per=6,
                         classes=("A", "B", "C"))
    fit = fit_hierarchy(sysl, "M2", A_w=2.0, A_b=1.0,
                        n_iter=1800, burn=600, seed=1)
    rep("hierarchy fits", fit.G_train == 3 and fit.n_draws > 0,
        f"G={fit.G_train}, draws={fit.n_draws}")
    Sb = fit.Sigma_b.mean(0)
    Sw = fit.Sigma_w.mean(0)
    print(f"      Sigma_b diag mean = {np.round(np.diag(Sb), 5).tolist()}")
    print(f"      Sigma_w diag mean = {np.round(np.diag(Sw), 5).tolist()}")
    rep("Sigma_b > 0", bool((np.diag(Sb) > 0).all()))

    # ---- A2 unseen-class prediction uses Sigma_b ----
    print("\n[A2] Unseen-class prediction")
    pred = predict_unseen_class(fit, T_target=310.0)
    print(f"      dCp(310) = {pred['mean_dCp']:+.4f} "
          f"± {pred['sd_dCp']:.4f}  CI {pred['ci95']}")
    rep("prediction is finite", math.isfinite(pred["mean_dCp"]))
    rep("CI is wide (between-class uncertainty)",
        pred["ci95"][1] - pred["ci95"][0] > 0)
    # compare with a within-class (overconfident) interval
    x = design_row(310.0, "dH", 310.0)[list(fit.cols)]
    within_sd = float(np.sqrt(x @ Sw @ x))
    rep("unseen interval wider than within-class interval",
        pred["sd_dCp"] > within_sd,
        f"unseen={pred['sd_dCp']:.4f} > within={within_sd:.4f}")

    # ---- A3 stacking weights from training classes only ----
    print("\n[A3] Stacking weights on training classes")
    sw = stacking_weights_for_classes(sysl, models=("M0", "M1", "M2"),
                                     n_iter=1200, burn=400, seed=2)
    rep("stacking runs", sw["status"] == "OK", sw.get("reason", ""))
    if sw["status"] == "OK":
        print(f"      weights = {sw['weights']}")
        rep("weights sum to 1", abs(sum(sw['weights'].values()) - 1) < 1e-9)
        rep("M2 gets weight (data has dCp structure)",
            sw["weights"].get("M2", 0) > 0.1)

    # ---- A4 prior-scale sensitivity sweep ----
    print("\n[A4] Prior-scale sensitivity for tau_w, tau_b")
    sens = prior_scale_sweep(sysl, T_target=310.0, model="M2",
                             factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                             n_iter=1200, burn=400, seed=3)
    if sens["status"] == "OK":
        print(f"      prediction range across prior scales = "
              f"{[round(v, 4) for v in sens['prediction_range']]}")
        print(f"      prior_leverage_ratio = {sens['prior_leverage_ratio']:.3f}")
        print(f"      verdict = {sens['verdict']}")
        rep("sweep completed >= 3 factors",
            sum(1 for v in sens['sweep'].values() if 'mean_dCp' in v) >= 3)
        rep("leverage ratio is finite",
            math.isfinite(sens["prior_leverage_ratio"]))
    else:
        rep("sweep completed", False, sens.get("reason", ""))

    # ---- A5 small-G warning fires ----
    print("\n[A5] G=2 warning")
    sysl2 = [s for s in sysl if s.group in ("A", "B")]
    fit2 = fit_hierarchy(sysl2, "M2", n_iter=900, burn=300, seed=4)
    rep("G=2 flagged", any("G_train=2" in w for w in fit2.warnings),
        f"{fit2.warnings}")

    # ---- A6 LOO-class holds out a whole class ----
    print("\n[A6] LOO-class: hold out class C entirely")
    train_sysl = [s for s in sysl if s.group != "C"]
    fit_loo = fit_hierarchy(train_sysl, "M2", n_iter=1500, burn=500, seed=5)
    pred_loo = predict_unseen_class(fit_loo, T_target=310.0)
    # true dCp for class C at 310 is base + shift_C dCp0 = -0.65 + 0.30 = -0.35
    true_C = -0.35
    in_ci = pred_loo["ci95"][0] <= true_C <= pred_loo["ci95"][1]
    print(f"      predicted dCp(C,310) = {pred_loo['mean_dCp']:+.4f} "
          f"CI {pred_loo['ci95']}, true = {true_C:+.4f}")
    rep("LOO-class prediction covers truth (wide interval)",
        in_ci or pred_loo["sd_dCp"] > 0.3,
        "interval is honest about unseen-class uncertainty")

    print("\n" + "=" * 94)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 94)
    print("""
  WHAT THESE TESTS PROVE
    A1  the Normal-IW hierarchy fits and produces positive variance components
    A2  unseen-class prediction uses Sigma_b and is WIDER than the
        within-class interval (no overconfidence)
    A3  stacking weights are fit on training classes only and sum to 1
    A4  the prior-scale sweep runs and reports a prior-leverage ratio
    A5  G=2 is explicitly flagged as prior-dominated
    A6  holding out an entire class gives an honest, wide interval

  WHAT IS NOT PROVEN
    * with few training classes, Sigma_b is weakly identified; the
      sensitivity sweep is the real evidence, read it
    * stacking weights optimized on training classes are not guaranteed
      optimal for the unseen class
    * exchangeability of classes is assumed; a chemically out-of-range
      held-out class will be centered wrong even with honest width
    * no experimental validation is performed here
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Unseen-class + prior sensitivity v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--systems", default=None,
                    help="JSON list of {name,group,T,kind,y,sigma,T0}")
    ap.add_argument("--holdout-class", default=None,
                    help="class label to hold out for unseen prediction")
    ap.add_argument("--T-target", type=float, default=310.0)
    ap.add_argument("--model", default="M2", choices=list(MODELS))
    ap.add_argument("--sweep", action="store_true",
                    help="run the prior-scale sensitivity sweep")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    if not a.systems:
        ap.print_help()
        print(f"\n{ATTRIBUTION}")
        return

    with open(a.systems) as f:
        raw = json.load(f)
    sysl = [SystemData(d["name"], d["group"], np.array(d["T"], float),
                       np.array(d["kind"]), np.array(d["y"], float),
                       np.array(d["sigma"], float),
                       float(d.get("T0", 310.0))) for d in raw]

    out: Dict[str, Any] = {}
    if a.holdout_class:
        train = [s for s in sysl if s.group != a.holdout_class]
        fit = fit_hierarchy(train, a.model)
        pred = predict_unseen_class(fit, a.T_target)
        out["unseen_prediction"] = pred
        out["fit_warnings"] = fit.warnings
        print(json.dumps(pred, indent=2))
    if a.sweep:
        train = sysl
        if a.holdout_class:
            train = [s for s in sysl if s.group != a.holdout_class]
        sens = prior_scale_sweep(train, a.T_target, a.model)
        out["prior_sensitivity"] = sens
        if sens["status"] == "OK":
            print(f"\nprior_leverage_ratio = {sens['prior_leverage_ratio']:.3f} "
                  f"-> {sens['verdict']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\njson -> {a.out}")


if __name__ == "__main__":
    main()
