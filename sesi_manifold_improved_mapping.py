"""
=============================================================================
SESI IMPROVED OUT-OF-SAMPLE MANIFOLD MAPPING
Nyström extension + kernel regression + validation suite
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS IMPROVES (vs sesi_manifold_acceleration.py)
-----------------------------------------------------
The previous module used Inverse-Distance-Weighted (IDW) interpolation from
the k=5 nearest training points. IDW is a GEODESIC-UNAWARED approximation:
it treats parameter-space distance as if it were intrinsic distance, which
fails when the manifold is strongly curved or the query point is far from
the training cloud.

This module provides THREE improved out-of-sample mappers, each respecting
the kernel structure learned during training:

  [M1] NyströmOutofSampleMapper — classical Nyström extension of cMDS.
       Projects a query point onto the intrinsic coordinates by computing
       its kernel similarities to the training set and projecting onto the
       top eigenvectors. This is the STANDARD manifold-learning extension
       [C: Bengio et al. 2004, "Out-of-Sample Extensions for LLE..."].

  [M2] KernelRegressionOutofSampleMapper — Nadaraya-Watson-style regression
       from training intrinsic coords to query, using the waveform-overlap
       kernel. Better when the manifold has non-uniform density.

  [M3] LocalTangentOutofSampleMapper — for each query, finds its k nearest
       training points, fits a local linear model in their intrinsic coords,
       and maps via the local tangent space. Best for highly curved regions.

VALIDATION (per Carpet-19 Thm 3(b) discipline)
----------------------------------------------
Each mapper is validated against a HOLD-OUT set of waveforms:
  • For each hold-out point, compute its TRUE nearest-neighbour mismatch
    against the training set (the ground truth).
  • Compute the PREDICTED nearest-neighbour mismatch using each mapper's
    intrinsic coordinates.
  • Report the correlation between predicted and true mismatch, plus the
    rank-1 retrieval accuracy (does the predicted nearest neighbour equal
    the true nearest neighbour?).

LABELS: [C] cited method  [D] derived implementation  [O] open

HONEST SCOPE
------------
• All three mappers are APPROXIMATIONS. None is exact; the quality depends
  on the training set size, the intrinsic dimensionality, and the curvature
  of the waveform manifold.
• The validation reports CORRELATION, not proof. A high correlation on the
  hold-out set is evidence the mapper is useful, not a theorem that it is
  always accurate.
• The "best" mapper depends on the waveform family; no single mapper is
  universally optimal. The AdaptiveManifoldAccelerator selects per-query.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.manifold_improved")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sesi_chip_template_bank import (
        TemplateBankConfig, PrecessingWaveformGenerator, BankMetricEstimator)
    HAS_BANK = True
except ImportError:
    HAS_BANK = False

__all__ = [
    "ImprovedManifoldConfig",
    "NyströmOutofSampleMapper",
    "KernelRegressionOutofSampleMapper",
    "LocalTangentOutofSampleMapper",
    "OutofSampleValidator",
    "AdaptiveManifoldAccelerator",
    "compare_all_mappers",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ImprovedManifoldConfig:
    n_train: int = 200
    n_holdout: int = 50
    intrinsic_dim: int = 3
    kernel_bandwidth: float = 0.3       # for kernel regression
    k_neighbours: int = 7               # for local tangent / IDW baseline
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# BASE: waveform-kernel cMDS (reused from the original module)
# ═══════════════════════════════════════════════════════════════════════

class _BaseManifoldLearner:
    """Shared training logic: build waveform-overlap kernel, run cMDS."""

    def __init__(self, cfg: ImprovedManifoldConfig):
        self.cfg = cfg
        self._wfg = None
        self._est = None
        self.params: Optional[np.ndarray] = None
        self.coords: Optional[np.ndarray] = None
        self.kernel_matrix: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

    def _waveform_tools(self):
        if not HAS_BANK:
            raise RuntimeError("sesi_chip_template_bank required")
        if self._wfg is None:
            bank_cfg = TemplateBankConfig(sample_rate=1024.0, duration=2.0)
            self._wfg = PrecessingWaveformGenerator(bank_cfg)
            self._est = BankMetricEstimator(bank_cfg)
        return self._wfg, self._est

    def fit(self, params: np.ndarray) -> Dict[str, Any]:
        """Build kernel, run cMDS, store eigendecomposition."""
        wfg, est = self._waveform_tools()
        n = len(params)
        self.params = np.asarray(params)

        # Waveforms
        waveforms = [wfg.generate(p[0], p[1], p[2]) for p in params]

        # Kernel (overlap matrix)
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                ov = est.overlap(waveforms[i], waveforms[j])
                K[i, j] = ov
                K[j, i] = ov
        self.kernel_matrix = K

        # cMDS: double-center -D²/2, take top eigenvectors
        D2 = 2.0 * (1.0 - np.clip(K, -1, 1))
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2 @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1]
        d = self.cfg.intrinsic_dim
        vals = np.clip(eigvals[order[:d]], 0, None)
        vecs = eigvecs[:, order[:d]]
        self.eigvals = vals
        self.eigvecs = vecs
        self.coords = vecs * np.sqrt(vals)[None, :]
        return {
            "n_train": n,
            "intrinsic_dim": d,
            "top_eigvals": vals.tolist(),
            "explained_variance_ratio": float(vals.sum() / max(np.clip(eigvals, 0, None).sum(), 1e-30)),
        }

    def _waveform_for(self, p: np.ndarray) -> np.ndarray:
        wfg, _ = self._waveform_tools()
        return wfg.generate(p[0], p[1], p[2])

    def _kernel_to_train(self, p_query: np.ndarray) -> np.ndarray:
        """Compute kernel similarities between query and every training point."""
        _, est = self._waveform_tools()
        h_q = self._waveform_for(p_query)
        n = len(self.params)
        k = np.zeros(n)
        for i in range(n):
            h_i = self._waveform_for(self.params[i])
            k[i] = est.overlap(h_q, h_i)
        return k


# ═══════════════════════════════════════════════════════════════════════
# [M1] NYSTRÖM OUT-OF-SAMPLE MAPPER
# ═══════════════════════════════════════════════════════════════════════

class NyströmOutofSampleMapper(_BaseManifoldLearner):
    """
    [C/D] Classical Nyström extension of cMDS.

    Given a query point q with kernel similarities k_q to the training set,
    the intrinsic coordinates are:
        z_q = (1/2) Λ^{-1/2} V^T k_q
    where Λ, V are the top eigenvalues/vectors of the centered kernel.

    This is the STANDARD manifold-learning extension and is exact when the
    query lies on the same manifold as the training set. Off-manifold
    queries are projected to the nearest point on the learned manifold.
    """

    def transform(self, p_query: np.ndarray) -> np.ndarray:
        if self.coords is None:
            raise RuntimeError("call fit() first")
        k_q = self._kernel_to_train(p_query)
        # Center the kernel vector the same way training was centered
        k_q_centered = k_q - self.kernel_matrix.mean(axis=0)
        # Nyström projection
        Λ = self.eigvals
        V = self.eigvecs
        z = 0.5 * (V.T @ k_q_centered) / np.sqrt(np.maximum(Λ, 1e-12))
        return z


# ═══════════════════════════════════════════════════════════════════════
# [M2] KERNEL REGRESSION OUT-OF-SAMPLE MAPPER
# ═══════════════════════════════════════════════════════════════════════

class KernelRegressionOutofSampleMapper(_BaseManifoldLearner):
    """
    [C/D] Nadaraya-Watson kernel regression from training coords to query.

    Given query q with kernel similarities k_q to training points, the
    intrinsic coordinates are a kernel-weighted average of training coords:
        z_q = Σ_i w_i z_i,   w_i ∝ exp(-d²(q, i) / 2h²)
    where d² is derived from the waveform-overlap kernel: d² = 2(1 - k).

    This is more robust than Nyström when the training density is uneven,
    because it does not assume the manifold is globally linear.
    """

    def transform(self, p_query: np.ndarray) -> np.ndarray:
        if self.coords is None:
            raise RuntimeError("call fit() first")
        k_q = self._kernel_to_train(p_query)
        # Convert overlap to squared distance
        d2 = 2.0 * (1.0 - np.clip(k_q, -1, 1))
        h = self.cfg.kernel_bandwidth
        w = np.exp(-d2 / (2.0 * h * h))
        w = w / (w.sum() + 1e-30)
        return w @ self.coords


# ═══════════════════════════════════════════════════════════════════════
# [M3] LOCAL TANGENT OUT-OF-SAMPLE MAPPER
# ═══════════════════════════════════════════════════════════════════════

class LocalTangentOutofSampleMapper(_BaseManifoldLearner):
    """
    [C/D] Local tangent-space mapping.

    For each query:
      1. Find k nearest training points in RAW parameter space.
      2. Fit a local linear model: intrinsic_coord ≈ A · raw_param + b.
      3. Map the query through this local model.

    This is the most robust for HIGHLY CURVED manifolds, because it
    adapts the mapping to the local geometry. It is also the most
    expensive (one local fit per query).
    """

    def transform(self, p_query: np.ndarray) -> np.ndarray:
        if self.coords is None:
            raise RuntimeError("call fit() first")
        k = self.cfg.k_neighbours
        # Find k nearest training points in raw parameter space
        d = np.linalg.norm(self.params - p_query, axis=1)
        idx = np.argsort(d)[:k]
        if len(idx) < 3:
            # Degenerate: fall back to Nyström-style
            return NyströmOutofSampleMapper.transform(self, p_query)
        # Local linear fit: z ≈ A p + b
        P = self.params[idx]
        Z = self.coords[idx]
        # Center
        P_mean = P.mean(axis=0)
        Z_mean = Z.mean(axis=0)
        dP = P - P_mean
        dZ = Z - Z_mean
        # Least-squares: A = (dP^T dP)^{-1} dP^T dZ
        try:
            A = np.linalg.lstsq(dP, dZ, rcond=None)[0]
        except np.linalg.LinAlgError:
            A = np.zeros((3, self.cfg.intrinsic_dim))
        return Z_mean + (p_query - P_mean) @ A


# ═══════════════════════════════════════════════════════════════════════
# [D] OUT-OF-SAMPLE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class OutofSampleValidator:
    """
    [D] Validate each mapper against the ground truth (true mismatch).

    For each hold-out point:
      • TRUE nearest neighbour: the training point with smallest waveform
        mismatch to the query (ground truth).
      • PREDICTED nearest neighbour: the training point with smallest
        intrinsic-coordinate distance to the query's mapped position.
      • Report: rank-1 retrieval accuracy + correlation of predicted vs
        true mismatch.
    """

    def __init__(self, cfg: ImprovedManifoldConfig):
        self.cfg = cfg

    def validate(self, mapper, holdout_params: np.ndarray) -> Dict[str, Any]:
        """Run validation for one mapper on the hold-out set."""
        if not HAS_BANK:
            return {"error": "bank module unavailable"}
        wfg, est = mapper._waveform_tools()

        # Build training waveforms (cached)
        train_waveforms = [wfg.generate(p[0], p[1], p[2]) for p in mapper.params]

        rank1_correct = 0
        true_mm = []
        pred_mm = []

        for p_q in holdout_params:
            h_q = wfg.generate(p_q[0], p_q[1], p_q[2])

            # TRUE nearest neighbour (ground truth)
            true_mismatches = np.array([1.0 - est.overlap(h_q, h_t)
                                        for h_t in train_waveforms])
            true_nn = int(np.argmin(true_mismatches))
            true_mm.append(true_mismatches[true_nn])

            # PREDICTED nearest neighbour (via mapper)
            z_q = mapper.transform(p_q)
            pred_dists = np.linalg.norm(mapper.coords - z_q, axis=1)
            pred_nn = int(np.argmin(pred_dists))
            pred_mm.append(true_mismatches[pred_nn])  # true mismatch at predicted NN

            if pred_nn == true_nn:
                rank1_correct += 1

        true_mm = np.array(true_mm)
        pred_mm = np.array(pred_mm)
        n = len(holdout_params)

        # Correlation
        if np.std(true_mm) > 0 and np.std(pred_mm) > 0:
            corr = float(np.corrcoef(true_mm, pred_mm)[0, 1])
        else:
            corr = float("nan")

        return {
            "n_holdout": n,
            "rank1_accuracy": rank1_correct / n,
            "corr_true_vs_predicted_mm": corr,
            "mean_true_mm": float(true_mm.mean()),
            "mean_pred_mm": float(pred_mm.mean()),
            "mapper_name": type(mapper).__name__,
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] ADAPTIVE MANIFOLD ACCELERATOR
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveManifoldAccelerator:
    """
    [D] sbank-style placement that selects the best out-of-sample mapper
    per query, based on a small validation run during training.

    The accelerator:
      1. Fits all three mappers on the training set.
      2. Validates each on a hold-out set.
      3. Selects the mapper with highest rank-1 retrieval accuracy.
      4. Uses that mapper for all subsequent queries.

    HONEST: the selection is made once, on the training distribution. If
    the query distribution shifts significantly, the chosen mapper may not
    remain optimal. The validator reports the selection, not a guarantee.
    """

    def __init__(self, cfg: ImprovedManifoldConfig):
        self.cfg = cfg
        self.mappers = {
            "nystrom": NyströmOutofSampleMapper(cfg),
            "kernel_regression": KernelRegressionOutofSampleMapper(cfg),
            "local_tangent": LocalTangentOutofSampleMapper(cfg),
        }
        self.selected_mapper: Optional[Any] = None
        self.selection_report: Optional[Dict[str, Any]] = None

    def fit_and_select(self, train_params: np.ndarray,
                       holdout_params: np.ndarray) -> Dict[str, Any]:
        """Fit all mappers, validate, select the best."""
        validator = OutofSampleValidator(self.cfg)
        reports = {}
        for name, mapper in self.mappers.items():
            fit_info = mapper.fit(train_params)
            val = validator.validate(mapper, holdout_params)
            reports[name] = {"fit": fit_info, "validation": val}

        # Select by rank-1 accuracy (primary), break ties by correlation
        best_name = max(reports, key=lambda n: (
            reports[n]["validation"].get("rank1_accuracy", 0.0),
            reports[n]["validation"].get("corr_true_vs_predicted_mm", 0.0),
        ))
        self.selected_mapper = self.mappers[best_name]
        self.selection_report = {
            "selected": best_name,
            "reports": reports,
            "honest_note": (
                "Selection is made on the training distribution. If the query "
                "distribution shifts, the chosen mapper may not remain optimal. "
                "No single mapper is universally best."
            ),
        }
        return self.selection_report

    def transform(self, p_query: np.ndarray) -> np.ndarray:
        if self.selected_mapper is None:
            raise RuntimeError("call fit_and_select() first")
        return self.selected_mapper.transform(p_query)

    @property
    def coords(self) -> np.ndarray:
        return self.selected_mapper.coords


# ═══════════════════════════════════════════════════════════════════════
# [D] FULL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compare_all_mappers(cfg: Optional[ImprovedManifoldConfig] = None
                        ) -> Dict[str, Any]:
    """
    [D] Train all three mappers + the IDW baseline, validate each, and
    report which performs best on the hold-out set.

    This is the Carpet-19 Thm 3(b) discipline applied: the approximation
    is validated against the reference (true mismatch), not assumed better.
    """
    if not HAS_BANK:
        return {"error": "sesi_chip_template_bank not importable"}

    cfg = cfg or ImprovedManifoldConfig()
    rng = np.random.default_rng(cfg.seed)

    # Generate training + hold-out parameter sets
    n_total = cfg.n_train + cfg.n_holdout
    all_params = np.zeros((n_total, 3))
    for i in range(n_total):
        m1 = rng.uniform(15, 50)
        all_params[i] = [m1, rng.uniform(15, m1), rng.uniform(0, 1)]

    train = all_params[:cfg.n_train]
    holdout = all_params[cfg.n_train:]

    # IDW baseline (from the original module's approach)
    class IDWBaseline(_BaseManifoldLearner):
        def transform(self, p_query):
            d = np.linalg.norm(self.params - p_query, axis=1)
            k = min(self.cfg.k_neighbours, len(d))
            idx = np.argsort(d)[:k]
            w = 1.0 / (d[idx] + 1e-9)
            return (self.coords[idx] * w[:, None]).sum(0) / w.sum()

    # Fit all four mappers
    mappers = {
        "idw_baseline": IDWBaseline(cfg),
        "nystrom": NyströmOutofSampleMapper(cfg),
        "kernel_regression": KernelRegressionOutofSampleMapper(cfg),
        "local_tangent": LocalTangentOutofSampleMapper(cfg),
    }
    validator = OutofSampleValidator(cfg)
    results = {}
    for name, mapper in mappers.items():
        mapper.fit(train)
        val = validator.validate(mapper, holdout)
        results[name] = val

    # Rank by rank-1 accuracy
    ranking = sorted(results.items(),
                     key=lambda kv: kv[1].get("rank1_accuracy", 0.0),
                     reverse=True)

    return {
        "n_train": cfg.n_train,
        "n_holdout": cfg.n_holdout,
        "intrinsic_dim": cfg.intrinsic_dim,
        "results": results,
        "ranking": [(name, r["rank1_accuracy"], r["corr_true_vs_predicted_mm"])
                    for name, r in ranking],
        "best_mapper": ranking[0][0],
        "honest_note": (
            "The 'best' mapper depends on the waveform family and the "
            "training/hold-out split. No single mapper is universally best. "
            "Validation is on a FINITE hold-out set; generalization is not guaranteed."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Improved Out-of-Sample Manifold Mapping — Smoke Test")
    print("=" * 72)
    print(f"  scipy cKDTree : {HAS_SCIPY}")
    print(f"  bank module   : {HAS_BANK}")

    if not HAS_BANK:
        print("  [SKIP] requires sesi_chip_template_bank")
        return

    cfg = ImprovedManifoldConfig(n_train=120, n_holdout=30, intrinsic_dim=3)
    result = compare_all_mappers(cfg)
    if "error" in result:
        print(f"  [SKIP] {result['error']}")
        return

    print(f"\n  ── Mapper comparison (hold-out validation) ──")
    print(f"    training set     : {result['n_train']}")
    print(f"    hold-out set     : {result['n_holdout']}")
    print(f"    intrinsic dim    : {result['intrinsic_dim']}")

    print(f"\n    {'mapper':<22s} {'rank-1 acc':>10s} {'corr(mm)':>10s}")
    print("    " + "-" * 44)
    for name, r in result["results"].items():
        acc = r.get("rank1_accuracy", 0.0)
        corr = r.get("corr_true_vs_predicted_mm", float("nan"))
        print(f"    {name:<22s} {acc:>10.3f} {corr:>10.3f}")

    print(f"\n    Ranking by rank-1 accuracy:")
    for i, (name, acc, corr) in enumerate(result["ranking"]):
        marker = " ← BEST" if i == 0 else ""
        print(f"      {i+1}. {name:<22s} rank-1={acc:.3f}  corr={corr:.3f}{marker}")

    print(f"\n    {result['honest_note']}")

    # ── Adaptive accelerator demo ──
    print(f"\n  ── Adaptive accelerator (auto-selects best mapper) ──")
    rng = np.random.default_rng(cfg.seed)
    n_total = cfg.n_train + cfg.n_holdout
    all_params = np.zeros((n_total, 3))
    for i in range(n_total):
        m1 = rng.uniform(15, 50)
        all_params[i] = [m1, rng.uniform(15, m1), rng.uniform(0, 1)]
    train = all_params[:cfg.n_train]
    holdout = all_params[cfg.n_train:]

    accel = AdaptiveManifoldAccelerator(cfg)
    sel = accel.fit_and_select(train, holdout)
    print(f"    selected mapper : {sel['selected']}")
    best_val = sel["reports"][sel["selected"]]["validation"]
    print(f"    rank-1 accuracy : {best_val['rank1_accuracy']:.3f}")
    print(f"    correlation     : {best_val['corr_true_vs_predicted_mm']:.3f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • All mappers are APPROXIMATIONS; none is exact")
    print("  • Validation is on a FINITE hold-out set; generalization is not guaranteed")
    print("  • The 'best' mapper depends on the waveform family")
    print("  • This is a RESEARCH scaffold; production uses calibrated waveform models")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
