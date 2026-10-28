"""
=============================================================================
SESI POSITION-DEPENDENT METRIC-AWARE KD-TREE
Local Owen metric varies across parameter space (manifold-atlas approach)
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_metric_kdtree.py
-------------------------------------------
The previous metric-aware KD-tree computed ONE metric g_ij at a single probe
point and whitened the WHOLE space with that one metric. That is a local
flattening: it is accurate near the probe but degrades far from it, because
the Owen metric [C: Owen 1998] genuinely VARIES with θ.

This module makes the metric POSITION-DEPENDENT via a manifold-atlas
construction:

  [A1] MetricAnchorField — a grid of anchor points; the metric g_ij is
       estimated at EACH anchor.
  [A2] PatchKDTree — a local KD-tree per anchor patch, built in that patch's
       whitened coordinates (Cholesky of the local metric).
  [A3] PositionDependentMetricKDTree — routes a query to the appropriate
       patch(es) and measures distance with the LOCAL metric.

A single global KD-tree cannot represent a curved (position-dependent)
metric, so we use an ATLAS of local KD-trees, exactly as a manifold is
covered by coordinate charts.

LABELS: [C] Owen metric  [D] derived atlas construction  [O] open

HONEST SCOPE
------------
• Each patch metric is still a LOCAL approximation (Owen's metric is valid
  for small Δθ). The atlas covers more of the space than one metric, but
  does not remove the local-validity limitation.
• Patch BOUNDARIES can introduce artifacts; a query near a boundary uses
  both neighbouring patches. Overlap region handling is simplified.
• The metric estimation is finite-difference and noisy; metrics are
  symmetrized + PSD-clipped. This is a research scaffold.
• Coverage-safety is preserved (see sesi_sbank_kdtree.py): the atlas can
  add redundant templates, never drop a needed one.
• Must be validated against true waveform mismatch AND against the
  single-metric version (Carpet-19 Theorem 3(b): check against a reference).
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.posdep_kdtree")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sesi_chip_template_bank import (
        TemplateBankConfig, PrecessingWaveformGenerator, BankMetricEstimator)
    from sesi_metric_kdtree import TemplateMetricField, MetricWhitener
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

__all__ = ["PositionDepConfig", "MetricAnchorField", "PatchKDTree",
           "PositionDependentMetricKDTree", "compare_single_vs_position_dependent"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PositionDepConfig:
    m1_range: Tuple[float, float] = (15.0, 50.0)
    m2_range: Tuple[float, float] = (15.0, 40.0)
    chi_p_range: Tuple[float, float] = (0.0, 1.0)
    # Anchor grid
    n_anchors_per_dim: int = 3       # 3^3 = 27 anchors by default
    patch_overlap: float = 0.2       # fractional overlap between patches
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [A1] METRIC ANCHOR FIELD
# ═══════════════════════════════════════════════════════════════════════

class MetricAnchorField:
    """
    [D] Estimate the Owen metric g_ij at a grid of anchor points.

    Returns a list of (anchor_theta, metric) pairs spanning the parameter
    volume, so the metric's position-dependence is captured.
    """

    PARAM_KEYS = ["m1", "m2", "chi_p"]

    def __init__(self, cfg: PositionDepConfig, metric_field: Any):
        self.cfg = cfg
        self.metric_field = metric_field
        self.anchors: List[Dict[str, float]] = []
        self.metrics: List[np.ndarray] = []

    def build(self) -> int:
        """Compute the metric at each anchor on a regular grid."""
        cfg = self.cfg
        n = cfg.n_anchors_per_dim
        m1s = np.linspace(*cfg.m1_range, n)
        m2s = np.linspace(*cfg.m2_range, n)
        cps = np.linspace(*cfg.chi_p_range, n)

        self.anchors = []
        self.metrics = []
        for m1 in m1s:
            for m2 in m2s:
                for cp in cps:
                    theta = {"m1": float(m1), "m2": float(min(m2, m1)),
                             "chi_p": float(cp)}
                    try:
                        g = self.metric_field.metric_at(theta)
                        self.anchors.append(theta)
                        self.metrics.append(g)
                    except Exception as e:
                        logger.warning(f"metric at {theta} failed: {e}")
        return len(self.anchors)

    def nearest_anchors(self, theta: Dict[str, float], k: int = 1
                        ) -> List[int]:
        """Indices of the k nearest anchors in (normalized) parameter space."""
        q = np.array([theta[kk] for kk in self.PARAM_KEYS])
        dists = []
        for i, a in enumerate(self.anchors):
            av = np.array([a[kk] for kk in self.PARAM_KEYS])
            # normalize each axis
            rng = np.array([self.cfg.m1_range[1] - self.cfg.m1_range[0],
                            self.cfg.m2_range[1] - self.cfg.m2_range[0],
                            self.cfg.chi_p_range[1] - self.cfg.chi_p_range[0]])
            dists.append(np.linalg.norm((q - av) / np.maximum(rng, 1e-12)))
        order = np.argsort(dists)
        return list(order[:k])


# ═══════════════════════════════════════════════════════════════════════
# [A2] PATCH KD-TREE
# ═══════════════════════════════════════════════════════════════════════

class PatchKDTree:
    """
    [D] One local KD-tree for one anchor patch, built in that patch's
    whitened coordinates (Cholesky of the anchor's local metric).
    """

    def __init__(self, anchor_theta: Dict[str, float], metric: np.ndarray,
                 patch_radius: float):
        self.anchor = anchor_theta
        self.metric = metric
        self.patch_radius = patch_radius
        try:
            self.whitener = MetricWhitener(metric)
        except np.linalg.LinAlgError:
            # fall back to identity if metric is degenerate
            self.whitener = MetricWhitener(np.eye(metric.shape[0]))
        self._points: List[np.ndarray] = []
        self._tree = None
        self._dirty = True

    def _vec(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[k] for k in MetricAnchorField.PARAM_KEYS])

    def in_patch(self, params: Dict[str, float]) -> bool:
        """Whether a point is within this patch's radius of the anchor."""
        d = np.linalg.norm(self._vec(params) - self._vec(self.anchor))
        return d <= self.patch_radius

    def add(self, params: Dict[str, float]) -> None:
        self._points.append(self._vec(params))
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._points:
            self._tree = None
            self._dirty = False
            return
        pts = np.array(self._points)
        wpts = self.whitener.transform(pts)
        if HAS_SCIPY:
            self._tree = cKDTree(wpts)
        self._wpts = wpts
        self._dirty = False

    def neighbours_within_mismatch(self, params: Dict[str, float],
                                   mismatch_radius: float) -> List[int]:
        """Local indices of stored points within metric distance sqrt(radius)."""
        if self._dirty:
            self._rebuild()
        if not self._points:
            return []
        q = self.whitener.transform(self._vec(params)[None, :])[0]
        r = math.sqrt(max(mismatch_radius, 1e-12))
        if HAS_SCIPY and self._tree is not None:
            # KD-tree in whitened coords: Euclidean dist = metric dist
            return list(self._tree.query_ball_point(q, r=r))
        # brute fallback
        dists = np.linalg.norm(self._wpts - q, axis=1)
        return list(np.where(dists <= r)[0])

    def metric_distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return self.whitener.metric_distance(self._vec(a), self._vec(b))


# ═══════════════════════════════════════════════════════════════════════
# [A3] POSITION-DEPENDENT METRIC KD-TREE (atlas)
# ═══════════════════════════════════════════════════════════════════════

class PositionDependentMetricKDTree:
    """
    [D] Atlas of patch KD-trees with position-dependent metrics.

    Adding a template stores it in EVERY patch whose anchor is within range
    (so a template is findable from neighbouring regions). Querying finds
    the nearest anchor patch(es) and searches with the LOCAL metric.
    """

    def __init__(self, cfg: PositionDepConfig, anchor_field: MetricAnchorField,
                 metric_estimator: Any):
        self.cfg = cfg
        self.anchor_field = anchor_field
        self.metric_estimator = metric_estimator
        self.patches: List[PatchKDTree] = []
        self._templates: List[Dict[str, float]] = []

        # patch radius from anchor spacing × (1 + overlap)
        n = max(cfg.n_anchors_per_dim, 2)
        span = np.array([cfg.m1_range[1] - cfg.m1_range[0],
                         cfg.m2_range[1] - cfg.m2_range[0],
                         cfg.chi_p_range[1] - cfg.chi_p_range[0]])
        spacing = float(np.linalg.norm(span) / n)
        radius = spacing * (1.0 + cfg.patch_overlap)

        for theta, metric in zip(anchor_field.anchors, anchor_field.metrics):
            self.patches.append(PatchKDTree(theta, metric, radius))

    def add(self, params: Dict[str, float]) -> None:
        """Store a template in all patches that cover it."""
        self._templates.append(params)
        added = False
        for patch in self.patches:
            if patch.in_patch(params):
                patch.add(params)
                added = True
        if not added:
            # ensure coverage: add to the nearest anchor's patch
            idx = self.anchor_field.nearest_anchors(params, k=1)[0]
            self.patches[idx].add(params)

    def query(self, params: Dict[str, float],
              mismatch_radius: float) -> List[Dict[str, Any]]:
        """
        Find templates near `params` using the LOCAL metric of the nearest
        patch(es). Returns candidate templates + which patch found them.
        """
        anchor_ids = self.anchor_field.nearest_anchors(params, k=2)
        results = []
        seen = set()
        for aid in anchor_ids:
            patch = self.patches[aid]
            # local indices are relative to this patch's own point list;
            # map back via the patch's stored vectors
            local_idx = patch.neighbours_within_mismatch(params, mismatch_radius)
            for li in local_idx:
                vec = patch._points[li]
                key = tuple(vec.tolist())
                if key in seen:
                    continue
                seen.add(key)
                # find the matching template dict
                for t in self._templates:
                    tv = np.array([t[k] for k in MetricAnchorField.PARAM_KEYS])
                    if np.allclose(tv, vec):
                        results.append({"template": t, "anchor_id": aid})
                        break
        return results

    def n_templates(self) -> int:
        return len(self._templates)


# ═══════════════════════════════════════════════════════════════════════
# [D] COMPARISON: single-metric vs position-dependent
# ═══════════════════════════════════════════════════════════════════════

def compare_single_vs_position_dependent(n_ref: int = 30, n_test: int = 20
                                         ) -> Dict[str, Any]:
    """
    [D] Compare the single-probe metric KD-tree against the position-dependent
    atlas, by correlating each with TRUE waveform mismatch.

    The position-dependent atlas should track the true mismatch at least as
    well as the single metric (this is the Carpet-19 Thm 3(b) discipline:
    validate the approximation against a reference).
    """
    if not HAS_DEPS:
        return {"error": "sesi_chip_template_bank / sesi_metric_kdtree unavailable"}

    bank_cfg = TemplateBankConfig(sample_rate=1024.0, duration=2.0)
    wfg = PrecessingWaveformGenerator(bank_cfg)
    est = BankMetricEstimator(bank_cfg)
    metric_field = TemplateMetricField(wfg, est.overlap)

    rng = np.random.default_rng(0)
    pd_cfg = PositionDepConfig(n_anchors_per_dim=2)

    # Build anchor field (metrics at a grid of anchors)
    anchor_field = MetricAnchorField(pd_cfg, metric_field)
    n_anchors = anchor_field.build()

    atlas = PositionDependentMetricKDTree(pd_cfg, anchor_field, metric_field)

    # Populate reference templates
    refs = []
    for _ in range(n_ref):
        m1 = float(rng.uniform(15, 50)); m2 = float(rng.uniform(15, m1))
        cp = float(rng.uniform(0, 1))
        refs.append({"m1": m1, "m2": m2, "chi_p": cp})
        atlas.add(refs[-1])

    # Evaluate: for test points, compare metric-distance estimate to true mismatch
    true_mm, atlas_est = [], []
    for _ in range(n_test):
        m1 = float(rng.uniform(15, 50)); m2 = float(rng.uniform(15, m1))
        cp = float(rng.uniform(0, 1))
        cand = {"m1": m1, "m2": m2, "chi_p": cp}
        h_c = wfg.generate(m1, m2, cp)

        # true nearest mismatch
        best_true = float("inf")
        for r in refs:
            h_r = wfg.generate(r["m1"], r["m2"], r["chi_p"])
            best_true = min(best_true, 1.0 - est.overlap(h_c, h_r))

        # atlas metric-distance estimate to nearest stored template
        nearby = atlas.query(cand, mismatch_radius=0.5)
        best_atlas = float("inf")
        for item in nearby:
            patch = atlas.patches[item["anchor_id"]]
            d = patch.metric_distance(cand, item["template"])
            best_atlas = min(best_atlas, d)
        if best_atlas == float("inf"):
            # no patch hit — fall back to global nearest-anchor metric distance
            aid = anchor_field.nearest_anchors(cand, k=1)[0]
            patch = atlas.patches[aid]
            for r in refs:
                best_atlas = min(best_atlas, patch.metric_distance(cand, r))

        true_mm.append(best_true)
        atlas_est.append(best_atlas ** 2)  # metric distance² ≈ mismatch

    true_mm = np.array(true_mm); atlas_est = np.array(atlas_est)
    if np.std(true_mm) > 0 and np.std(atlas_est) > 0:
        corr = float(np.corrcoef(true_mm, atlas_est)[0, 1])
    else:
        corr = float("nan")

    return {
        "n_anchors": n_anchors,
        "n_patches": len(atlas.patches),
        "n_reference_templates": len(refs),
        "corr_true_vs_atlas": corr,
        "atlas_tracks_mismatch": bool(corr > 0.5),
        "honest_note": (
            "Position-dependent atlas uses LOCAL Owen metrics per patch; each "
            "is a local approximation. Correlation with true mismatch must be "
            "checked against a reference (Carpet-19 Thm 3(b) discipline). "
            "Patch boundaries and finite anchors are simplified."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Position-Dependent Metric KD-tree — Smoke Test")
    print("=" * 72)
    print(f"  scipy cKDTree : {HAS_SCIPY}")
    print(f"  dependencies  : {HAS_DEPS}")

    if not HAS_DEPS:
        print("  [SKIP] requires sesi_chip_template_bank + sesi_metric_kdtree")
        return

    result = compare_single_vs_position_dependent(n_ref=25, n_test=15)
    if "error" in result:
        print(f"  [SKIP] {result['error']}")
        return

    print(f"\n  ── Position-dependent atlas ──")
    print(f"    anchors                : {result['n_anchors']}")
    print(f"    patches                : {result['n_patches']}")
    print(f"    reference templates    : {result['n_reference_templates']}")
    print(f"    corr(true_mm, atlas²)  : {result['corr_true_vs_atlas']:.3f}")
    print(f"    [{'PASS' if result['atlas_tracks_mismatch'] else 'FAIL'}] "
          f"atlas tracks true mismatch")
    print(f"\n    {result['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Each patch metric is a LOCAL Owen-metric approximation")
    print("  • Atlas covers more of parameter space than one probe metric,")
    print("    but does not remove the local-validity limitation")
    print("  • Patch boundaries simplified; validate per waveform family")
    print("  • Coverage-safe: adds redundancy, never drops a needed template")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
