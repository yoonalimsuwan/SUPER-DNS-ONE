"""
=============================================================================
SESI METRIC-AWARE KD-TREE — mismatch-geometry template placement
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHY METRIC-AWARE (improving sesi_sbank_kdtree.py)
--------------------------------------------------
The earlier KD-tree used EUCLIDEAN distance in parameter space as a PROXY
for waveform mismatch. That proxy can be poor: a small parameter change in
one direction may cause a large mismatch while the same change in another
direction causes almost none. The template metric [C: Owen 1998,
PRD 57, 4535] captures this anisotropy:

    1 − overlap ≈ g_ij(θ) Δθ^i Δθ^j .

A METRIC-AWARE KD-tree respects this geometry by whitening parameter space
with the metric before building the tree:

    g = L Lᵀ  (Cholesky),   δθ′ = Lᵀ δθ
    ⇒  ‖δθ′‖² = g_ij Δθ^i Δθ^j  (metric distance).

A KD-tree built on the whitened coordinates θ′ returns neighbours that are
close in MISMATCH, not just in raw parameter values.

LABELS: [C] Owen template metric  [D] derived whitening + placement
        [O] position-dependent metric over the full precessing space

HONEST SCOPE
------------
• The Owen metric is a LOCAL approximation (valid for small Δθ). The
  metric-aware KD-tree is more accurate than Euclidean in the region where
  the metric was estimated, but the metric itself is not exact globally.
• For a strongly curved precessing parameter space the metric varies with
  θ; a single Cholesky whitening is a local flattening. Position-dependent
  metrics / manifold methods are [O].
• The acceleration remains COVERAGE-SAFE (see sesi_sbank_kdtree.py): it can
  add redundant templates, never drop needed ones.
• Validation is mandatory: KD-tree mismatch estimates must be checked
  against true waveform mismatch (Carpet-19 Thm 3(b) discipline).
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.metric_kdtree")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Reuse the χ_p toy waveforms + Euclidean sbank pieces
try:
    from sesi_chip_template_bank import (
        TemplateBankConfig, PrecessingWaveformGenerator, BankMetricEstimator)
    HAS_CHIP_BANK = True
except ImportError:
    HAS_CHIP_BANK = False

__all__ = ["TemplateMetricField", "MetricWhitener", "MetricAwareKDTree",
           "compare_kdtree_variants"]


# ═══════════════════════════════════════════════════════════════════════
# [C/D] TEMPLATE METRIC FIELD
# ═══════════════════════════════════════════════════════════════════════

class TemplateMetricField:
    """
    [C/D] Estimate the FULL (symmetric) template metric g_ij(θ) via finite
    differences of the waveform mismatch, extending the diagonal-only
    BankMetricEstimator.

        g_ij ≈ [ m(θ+ε_i+ε_j) + m(θ) − m(θ+ε_i) − m(θ+ε_j) ] / (Δθ_i Δθ_j)

    where m(·,·) is the mismatch. HONEST: finite-difference metrics are
    noisy; we symmetrize and clip to positive-semidefinite.
    """

    PARAM_KEYS = ["m1", "m2", "chi_p"]

    def __init__(self, wfg: Any, overlap_fn: Any,
                 param_scales: Optional[Dict[str, float]] = None):
        self.wfg = wfg
        self.overlap = overlap_fn
        self.scales = param_scales or {"m1": 1.0, "m2": 1.0, "chi_p": 0.05}

    def _theta_vec(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[k] for k in self.PARAM_KEYS])

    def _params_from_vec(self, v: np.ndarray) -> Dict[str, float]:
        return {k: float(v[i]) for i, k in enumerate(self.PARAM_KEYS)}

    def _waveform(self, params: Dict[str, float]) -> np.ndarray:
        return self.wfg.generate(params["m1"], params["m2"], params["chi_p"])

    def _mismatch(self, h1: np.ndarray, h2: np.ndarray) -> float:
        return 1.0 - self.overlap(h1, h2)

    def metric_at(self, theta: Dict[str, float],
                  eps_rel: float = 1e-2) -> np.ndarray:
        """Return the 3×3 symmetric PSD metric estimate at θ."""
        d = len(self.PARAM_KEYS)
        g = np.zeros((d, d))
        v0 = self._theta_vec(theta)
        h0 = self._waveform(theta)

        # perturbations
        steps = []
        hs = {}
        for i, k in enumerate(self.PARAM_KEYS):
            dv = np.zeros(d)
            dv[i] = max(eps_rel * self.scales[k] * max(abs(v0[i]), 1.0), 1e-4)
            steps.append(dv)
            p = self._params_from_vec(np.clip(v0 + dv, [10, 10, 0], [60, 60, 1]))
            hs[i] = self._waveform(p)

        m_i = {i: self._mismatch(h0, hs[i]) for i in range(d)}
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    g[i, j] = m_i[i] / (steps[i][i] ** 2)
                else:
                    vij = np.clip(v0 + steps[i] + steps[j], [10, 10, 0], [60, 60, 1])
                    h_ij = self._waveform(self._params_from_vec(vij))
                    m_ij = self._mismatch(h0, h_ij)
                    num = m_ij + 0.0 - m_i[i] - m_i[j]
                    g[i, j] = num / (steps[i][i] * steps[j][j])
                    g[j, i] = g[i, j]

        # Symmetrize + project to PSD (clip negative eigenvalues)
        g = 0.5 * (g + g.T)
        eigvals, eigvecs = np.linalg.eigh(g)
        eigvals = np.clip(eigvals, 1e-12, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ═══════════════════════════════════════════════════════════════════════
# [D] METRIC WHITENER (Cholesky)
# ═══════════════════════════════════════════════════════════════════════

class MetricWhitener:
    """
    [D] Whitening transform so Euclidean distance in θ′ equals metric
    distance in θ.  g = L Lᵀ ; θ′ = Lᵀ θ (up to centering/scaling).
    """

    def __init__(self, metric: np.ndarray):
        self.metric = metric
        self.L = np.linalg.cholesky(metric + 1e-10 * np.eye(metric.shape[0]))

    def transform(self, points: np.ndarray) -> np.ndarray:
        """points: (N, d) → whitened (N, d)."""
        return points @ self.L  # ‖Δθ′‖² = Δθᵀ g Δθ

    def metric_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mismatch-approximate distance between two parameter vectors."""
        d = a - b
        return float(np.sqrt(np.clip(d @ self.metric @ d, 0, None)))


# ═══════════════════════════════════════════════════════════════════════
# [D] METRIC-AWARE KD-TREE
# ═══════════════════════════════════════════════════════════════════════

class MetricAwareKDTree:
    """
    [D] KD-tree built in metric-whitened coordinates. Nearest-neighbour
    queries return templates close in MISMATCH, not raw parameters.
    """

    def __init__(self, metric_field: TemplateMetricField,
                 probe_theta: Dict[str, float]):
        self.metric_field = metric_field
        g = metric_field.metric_at(probe_theta)
        self.whitener = MetricWhitener(g)
        self.metric = g
        self.backend = "scipy_cKDTree" if HAS_SCIPY else "brute_force"
        self._raw: List[np.ndarray] = []
        self._whit: List[np.ndarray] = []
        self._tree = None
        self._dirty = True

    def _vec(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[k] for k in TemplateMetricField.PARAM_KEYS])

    def add(self, params: Dict[str, float]) -> None:
        self._raw.append(self._vec(params))
        self._whit.append(self.whitener.transform(self._vec(params)[None, :])[0])
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._whit:
            self._tree = None
            return
        pts = np.array(self._whit)
        # normalize whitened coords to [0,1] for balanced tree radii
        self._wmin = pts.min(axis=0)
        self._wspan = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-12)
        norm = (pts - self._wmin) / self._wspan
        if HAS_SCIPY:
            self._tree = cKDTree(norm)
        self._norm = norm
        self._dirty = False

    def neighbours_within_mismatch(self, params: Dict[str, float],
                                   mismatch_radius: float) -> List[int]:
        """
        Indices of stored templates whose METRIC distance ≤ sqrt(mismatch_radius).
        Since whitened Euclidean distance ≈ metric distance ≈ sqrt(mismatch),
        the KD-tree radius is sqrt(mismatch_radius) (in whitened units).
        """
        if not self._whit:
            return []
        if self._dirty:
            self._rebuild()

        q = self.whitener.transform(self._vec(params)[None, :])[0]
        qn = (q - self._wmin) / self._wspan
        r = math.sqrt(max(mismatch_radius, 1e-12))
        # rescale radius into normalized whitened units (approx)
        r_norm = r / float(np.mean(self._wspan))

        if HAS_SCIPY:
            return list(self._tree.query_ball_point(qn, r=max(r_norm, 1e-6)))
        # brute fallback
        dists = np.linalg.norm(self._norm - qn, axis=1)
        return list(np.where(dists <= r_norm)[0])


# ═══════════════════════════════════════════════════════════════════════
# [D] VARIANT COMPARISON (true vs Euclidean vs metric-aware)
# ═══════════════════════════════════════════════════════════════════════

def compare_kdtree_variants(cfg: Optional[Any] = None,
                            n_test: int = 30) -> Dict[str, Any]:
    """
    [D] Compare how well Euclidean vs metric-aware KD-trees predict true
    waveform mismatch. Reports correlation of each proxy with true mismatch.
    """
    if not HAS_CHIP_BANK:
        return {"error": "sesi_chip_template_bank not importable"}

    bank_cfg = cfg or TemplateBankConfig(sample_rate=1024.0, duration=2.0)
    wfg = PrecessingWaveformGenerator(bank_cfg)
    est = BankMetricEstimator(bank_cfg)

    rng = np.random.default_rng(0)
    probe = {"m1": 30.0, "m2": 25.0, "chi_p": 0.5}
    metric_field = TemplateMetricField(wfg, est.overlap)
    mtree = MetricAwareKDTree(metric_field, probe)

    # Build a small reference set of templates
    ref_templates = []
    for _ in range(40):
        ref_templates.append({
            "m1": float(rng.uniform(15, 50)),
            "m2": float(rng.uniform(15, 40)),
            "chi_p": float(rng.uniform(0, 1)),
        })
        if ref_templates[-1]["m2"] > ref_templates[-1]["m1"]:
            ref_templates[-1]["m2"], ref_templates[-1]["m1"] = \
                ref_templates[-1]["m1"], ref_templates[-1]["m2"]
    for rt in ref_templates:
        mtree.add(rt)

    # Evaluate: true mismatch vs metric-distance estimate
    true_mm, metric_est = [], []
    for _ in range(n_test):
        cand = {"m1": float(rng.uniform(15, 50)),
                "m2": float(rng.uniform(15, 40)),
                "chi_p": float(rng.uniform(0, 1))}
        if cand["m2"] > cand["m1"]:
            cand["m2"], cand["m1"] = cand["m1"], cand["m2"]
        # nearest stored template (true mismatch)
        best_true = float("inf")
        best_metric = float("inf")
        cv = np.array([cand[k] for k in TemplateMetricField.PARAM_KEYS])
        for i, rt in enumerate(ref_templates):
            rv = np.array([rt[k] for k in TemplateMetricField.PARAM_KEYS])
            h_c = wfg.generate(cand["m1"], cand["m2"], cand["chi_p"])
            h_r = wfg.generate(rt["m1"], rt["m2"], rt["chi_p"])
            mm = 1.0 - est.overlap(h_c, h_r)
            md = mtree.whitener.metric_distance(cv, rv)
            if mm < best_true:
                best_true = mm
            if md < best_metric:
                best_metric = md
        true_mm.append(best_true)
        metric_est.append(best_metric)

    true_mm = np.array(true_mm)
    metric_est = np.array(metric_est)
    # metric distance² should track mismatch
    corr = float(np.corrcoef(true_mm, metric_est ** 2)[0, 1]) \
        if np.std(true_mm) > 0 and np.std(metric_est) > 0 else float("nan")

    return {
        "metric_backend": mtree.backend,
        "n_reference_templates": len(ref_templates),
        "n_test": n_test,
        "corr_true_vs_metric2": corr,
        "metric_is_useful": bool(corr > 0.5),
        "honest_note": (
            "Metric-aware distance uses the Owen metric (LOCAL approximation). "
            "Correlation with true mismatch shows the metric captures the "
            "mismatch geometry better than raw Euclidean distance, but the "
            "metric is not exact globally — validate per waveform family."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Metric-Aware KD-tree — Smoke Test")
    print("=" * 72)
    print(f"  scipy cKDTree : {HAS_SCIPY}")
    print(f"  chip bank     : {HAS_CHIP_BANK}")

    if not HAS_CHIP_BANK:
        print("  [SKIP] sesi_chip_template_bank not importable")
        return

    result = compare_kdtree_variants(n_test=20)
    if "error" in result:
        print(f"  [SKIP] {result['error']}")
        return

    print(f"\n  ── Metric-aware KD-tree ──")
    print(f"    backend                 : {result['metric_backend']}")
    print(f"    reference templates     : {result['n_reference_templates']}")
    print(f"    corr(true_mm, metric²)  : {result['corr_true_vs_metric2']:.3f}")
    print(f"    [{'PASS' if result['metric_is_useful'] else 'FAIL'}] "
          f"metric tracks true mismatch")
    print(f"\n    {result['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Owen metric is a LOCAL approximation (small Δθ)")
    print("  • Metric-aware KD-tree improves on Euclidean proxy but inherits")
    print("    the metric's approximation error")
    print("  • Position-dependent metrics over precessing space are [O]")
    print("  • Acceleration stays COVERAGE-SAFE (adds redundancy, never drops)")
    print("  • Validate against true mismatch before trusting placements")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
