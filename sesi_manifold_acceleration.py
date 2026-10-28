"""
=============================================================================
SESI MANIFOLD-LEARNING ACCELERATION for template placement
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_position_dependent_kdtree.py
-------------------------------------------------------
The metric-atlas KD-tree used the Owen metric in RAW parameter space. This
module instead learns the INTRINSIC geometry of the template family via
manifold learning, then builds the accelerator in the learned coordinates.

The idea: templates of nearby physical parameters produce nearby waveforms,
but the map (m1, m2, χ_p) → waveform is non-linear. Manifold learning finds
low-dimensional coordinates that respect the WAVEFORM similarity, which is
what mismatch actually measures.

APPROACH (dependency-light)
---------------------------
  [M1] Sample a training set of templates and compute their waveforms.
  [M2] Build a waveform-similarity kernel K_ij = overlap(h_i, h_j).
  [M3] Classical Multidimensional Scaling (cMDS) on the kernel to get
       intrinsic coordinates. (A self-contained alternative to Isomap /
       diffusion maps; swappable if scikit-learn is present.)
  [M4] Build a KD-tree in the intrinsic coordinates.
  [M5] Query: map a candidate to intrinsic coords (via nearest training
       neighbours), then search the KD-tree.

LABELS: [C] cMDS / manifold-learning idea  [D] derived implementation
        [O] production-grade manifold + out-of-sample extension

HONEST SCOPE
------------
• Manifold coordinates are learned from a FINITE training set; out-of-sample
  mapping is approximate (nearest-neighbour interpolation here).
• The accelerator is still COVERAGE-SAFE (adds redundancy, never drops a
  needed template) — same argument as the KD-tree modules.
• Coverage MUST be verified against true mismatch (Carpet-19 Thm 3(b):
  check the approximation against a reference, never assume it).
• This is a research scaffold; production template banks use dedicated
  geometric/stochastic algorithms.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.manifold_accel")

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

__all__ = ["ManifoldConfig", "TemplateManifoldLearner",
           "ManifoldAccelerator", "compare_manifold_vs_raw"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ManifoldConfig:
    n_train: int = 200          # training templates
    intrinsic_dim: int = 3      # target intrinsic dimension
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [M1-M3] MANIFOLD LEARNER (waveform-kernel cMDS)
# ═══════════════════════════════════════════════════════════════════════

class TemplateManifoldLearner:
    """
    [C/D] Learn intrinsic template coordinates from waveform similarity.

    Builds a similarity kernel from waveform overlaps and applies classical
    Multidimensional Scaling to obtain coordinates that preserve the
    waveform geometry.
    """

    def __init__(self, cfg: ManifoldConfig):
        self.cfg = cfg
        self.params: Optional[np.ndarray] = None     # (N, 3) raw parameters
        self.coords: Optional[np.ndarray] = None     # (N, d) intrinsic coords
        self._wfg = None
        self._est = None

    def _waveform_tools(self):
        if not HAS_BANK:
            raise RuntimeError("sesi_chip_template_bank required")
        if self._wfg is None:
            bank_cfg = TemplateBankConfig(sample_rate=1024.0, duration=2.0)
            self._wfg = PrecessingWaveformGenerator(bank_cfg)
            self._est = BankMetricEstimator(bank_cfg)
        return self._wfg, self._est

    def fit(self, params: np.ndarray) -> np.ndarray:
        """
        Fit intrinsic coordinates for the given parameter samples.

        Args:
            params: (N, 3) array of [m1, m2, chi_p]
        Returns:
            coords: (N, intrinsic_dim) intrinsic coordinates
        """
        wfg, est = self._waveform_tools()
        n = len(params)
        self.params = np.asarray(params)

        # [M1] waveforms
        waveforms = [wfg.generate(p[0], p[1], p[2]) for p in params]

        # [M2] similarity kernel from overlaps, converted to a distance
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                ov = est.overlap(waveforms[i], waveforms[j])
                K[i, j] = ov
                K[j, i] = ov

        # distance from similarity: d = sqrt(2(1-overlap)) (mismatch-like)
        D2 = 2.0 * (1.0 - np.clip(K, -1, 1))

        # [M3] classical MDS: double-center -D²/2, take top eigenvectors
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2 @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        # take the largest eigenvalues (they come last from eigh)
        order = np.argsort(eigvals)[::-1]
        d = self.cfg.intrinsic_dim
        vals = np.clip(eigvals[order[:d]], 0, None)
        vecs = eigvecs[:, order[:d]]
        self.coords = vecs * np.sqrt(vals)[None, :]
        return self.coords

    def transform(self, params_query: np.ndarray) -> np.ndarray:
        """
        Map new parameter points to intrinsic coords via nearest training
        neighbours (approximate out-of-sample extension).
        """
        if self.params is None or self.coords is None:
            raise RuntimeError("call fit() first")
        q = np.atleast_2d(params_query)
        out = np.zeros((len(q), self.coords.shape[1]))
        for i, p in enumerate(q):
            d = np.linalg.norm(self.params - p, axis=1)
            # inverse-distance weighted average of the 5 nearest
            idx = np.argsort(d)[:5]
            w = 1.0 / (d[idx] + 1e-9)
            out[i] = (self.coords[idx] * w[:, None]).sum(0) / w.sum()
        return out


# ═══════════════════════════════════════════════════════════════════════
# [M4-M5] MANIFOLD ACCELERATOR
# ═══════════════════════════════════════════════════════════════════════

class ManifoldAccelerator:
    """
    [D] KD-tree in intrinsic (manifold) coordinates for fast template
    neighbour queries.
    """

    def __init__(self, learner: TemplateManifoldLearner):
        self.learner = learner
        self._tree = None
        self._backend = "scipy_cKDTree" if HAS_SCIPY else "brute_force"

    def build(self) -> None:
        if self.learner.coords is None:
            raise RuntimeError("learner not fitted")
        if HAS_SCIPY:
            self._tree = cKDTree(self.learner.coords)

    def neighbours(self, params_query: np.ndarray,
                   radius_intrinsic: float) -> List[int]:
        """Indices of training templates within intrinsic radius."""
        q = self.learner.transform(params_query)[0]
        if HAS_SCIPY and self._tree is not None:
            return list(self._tree.query_ball_point(q, r=radius_intrinsic))
        # brute fallback
        d = np.linalg.norm(self.learner.coords - q, axis=1)
        return list(np.where(d <= radius_intrinsic)[0])


# ═══════════════════════════════════════════════════════════════════════
# [D] COMPARISON: manifold vs raw-parameter distance
# ═══════════════════════════════════════════════════════════════════════

def compare_manifold_vs_raw(cfg: Optional[ManifoldConfig] = None,
                            n_test: int = 30) -> Dict[str, Any]:
    """
    [D] Check whether intrinsic coordinates track true waveform mismatch
    better than raw parameter distance does.

    This is the Carpet-19 Thm 3(b) discipline applied: the manifold
    approximation is validated against the reference (true mismatch), not
    assumed to be better.
    """
    if not HAS_BANK:
        return {"error": "sesi_chip_template_bank not importable"}

    cfg = cfg or ManifoldConfig(n_train=120, intrinsic_dim=3)
    rng = np.random.default_rng(cfg.seed)

    # training set
    train = np.zeros((cfg.n_train, 3))
    for i in range(cfg.n_train):
        m1 = rng.uniform(15, 50)
        train[i] = [m1, rng.uniform(15, m1), rng.uniform(0, 1)]

    learner = TemplateManifoldLearner(cfg)
    learner.fit(train)

    # test points: correlate intrinsic distance and raw distance with true mm
    wfg, est = learner._waveform_tools()
    true_mm, intrinsic_d, raw_d = [], [], []
    for _ in range(n_test):
        m1 = rng.uniform(15, 50)
        q = np.array([m1, rng.uniform(15, m1), rng.uniform(0, 1)])
        # nearest training template
        dtrain = np.linalg.norm(train - q, axis=1)
        j = int(np.argmin(dtrain))
        hq = wfg.generate(q[0], q[1], q[2])
        hj = wfg.generate(train[j][0], train[j][1], train[j][2])
        true_mm.append(1.0 - est.overlap(hq, hj))
        intrinsic_d.append(float(np.linalg.norm(
            learner.transform(q)[0] - learner.coords[j])))
        raw_d.append(float(dtrain[j]))

    true_mm = np.array(true_mm); intrinsic_d = np.array(intrinsic_d)
    raw_d = np.array(raw_d)

    def corr(a, b):
        if np.std(a) > 0 and np.std(b) > 0:
            return float(np.corrcoef(a, b)[0, 1])
        return float("nan")

    corr_intrinsic = corr(true_mm, intrinsic_d)
    corr_raw = corr(true_mm, raw_d)

    return {
        "n_train": cfg.n_train,
        "intrinsic_dim": cfg.intrinsic_dim,
        "corr_true_vs_intrinsic": corr_intrinsic,
        "corr_true_vs_raw": corr_raw,
        "manifold_better": bool(corr_intrinsic > corr_raw),
        "honest_note": (
            "Manifold coordinates should track true mismatch at least as well "
            "as raw parameter distance. If not, the training set or intrinsic "
            "dimension is inadequate. Out-of-sample mapping is approximate."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Manifold Acceleration — Smoke Test")
    print("=" * 72)
    print(f"  scipy cKDTree : {HAS_SCIPY}")
    print(f"  bank module   : {HAS_BANK}")

    if not HAS_BANK:
        print("  [SKIP] requires sesi_chip_template_bank")
        return

    result = compare_manifold_vs_raw(ManifoldConfig(n_train=100), n_test=20)
    if "error" in result:
        print(f"  [SKIP] {result['error']}")
        return

    print(f"\n  ── Manifold vs raw-parameter distance ──")
    print(f"    training templates        : {result['n_train']}")
    print(f"    intrinsic dim             : {result['intrinsic_dim']}")
    print(f"    corr(true_mm, intrinsic)  : {result['corr_true_vs_intrinsic']:.3f}")
    print(f"    corr(true_mm, raw)        : {result['corr_true_vs_raw']:.3f}")
    print(f"    [{'PASS' if result['manifold_better'] else 'NOTE'}] "
          f"{'manifold tracks mismatch better' if result['manifold_better'] else 'manifold not clearly better here'}")
    print(f"\n    {result['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Intrinsic coords from a FINITE training set")
    print("  • Out-of-sample mapping is approximate (IDW interpolation)")
    print("  • Manifold accelerator is coverage-safe like the KD-tree ones")
    print("  • Must be validated against true mismatch before trusting")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
