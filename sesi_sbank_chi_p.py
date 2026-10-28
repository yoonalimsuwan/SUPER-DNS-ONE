"""
=============================================================================
SESI STOCHASTIC χ_p TEMPLATE PLACEMENT (sbank-style)
Metric-driven stochastic insertion + pruning + INDEPENDENT coverage audit
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

METHOD [C: Pürrer & Haster 2020, "Robust and efficient matching pursuit
        placement of gravitational-wave template banks"]
-------------------------------------------------------------------------
sbank places templates stochastically rather than on a regular lattice:

  1. Propose a random point θ in the physical parameter space.
  2. Compute its best overlap against the current bank.
  3. If the mismatch exceeds (1 − minimal_match), INSERT it.
  4. Reuse rejected/uncovered points as an adaptive proposal ("hot map").
  5. Prune redundant templates.
  6. Audit coverage on an INDEPENDENT validation set.

WHY THIS BEATS A LATTICE FOR χ_p
--------------------------------
Precessing parameter spaces are strongly non-uniform in metric and the
local-metric approximation fails globally. Stochastic placement discovers
the actual uncovered regions instead of trusting a diagonal metric.

INDEPENDENT AUDIT (Carpet-19 Thm 3(b) discipline)
-------------------------------------------------
Coverage is measured on a validation sample that was NEVER used to build
or prune the bank. Reporting coverage on the training points would be
self-congratulatory, not a convergence check.

LABELS: [C] sbank methodology  [D] implementation  [O] open

HONEST SCOPE
------------
• Waveforms here are TOY precessing surrogates, NOT IMRPhenomPv3 /
  SEOBNRv4PHM. Real banks must be built with lalsimulation.
• Overlap uses a white-noise inner product and a single global phase
  maximization; real placement uses noise-weighted, orientation-aware
  matches over the full extrinsic space.
• The achieved coverage is a MEASURED estimate on a finite validation set,
  not a certificate over the continuum parameter space.
• This is "sbank-style", not a reimplementation of the production sbank.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.sbank_chip")

__all__ = [
    "SbankConfig",
    "ToyPrecessingGenerator",
    "OverlapComputer",
    "StochasticChiPBank",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SbankConfig:
    # Physical space: (m1, q = m2/m1, χ_p)
    m1_range: Tuple[float, float] = (10.0, 80.0)
    q_range: Tuple[float, float] = (0.45, 1.0)
    chi_p_range: Tuple[float, float] = (0.0, 1.0)

    minimal_match: float = 0.97
    n_seed: int = 24
    n_iterations: int = 600
    n_validate: int = 250
    n_prune_probes: int = 80
    hot_map_fraction: float = 0.5   # fraction of proposals from uncovered points

    sample_rate: float = 1024.0
    duration: float = 2.0
    f_min: float = 20.0
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [D-TOY] PRECESSING WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class ToyPrecessingGenerator:
    """
    [D-TOY] Chirp with χ_p-controlled precession modulation.
    NOT a calibrated precessing approximant.
    """

    MSUN_SEC = 4.925490947e-6

    def __init__(self, cfg: SbankConfig):
        self.cfg = cfg
        self.n = int(cfg.sample_rate * cfg.duration)
        self.t = np.arange(self.n) / cfg.sample_rate

    def generate(self, theta: np.ndarray) -> np.ndarray:
        m1, q, chi_p = float(theta[0]), float(theta[1]), float(theta[2])
        m2 = m1 * q
        Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        Mc_sec = Mc * self.MSUN_SEC

        t_c = self.cfg.duration * 0.7
        tau = np.maximum(t_c - self.t, 1e-4)
        f = (1 / math.pi) * (5 / (256 * tau)) ** 0.375 * Mc_sec ** -0.625
        f = np.clip(f, self.cfg.f_min, self.cfg.sample_rate * 0.45)
        phase = 2 * math.pi * np.cumsum(f) / self.cfg.sample_rate

        f_prec = chi_p * f * 0.1
        prec = 2 * math.pi * np.cumsum(f_prec) / self.cfg.sample_rate
        amp_mod = 1.0 + 0.3 * chi_p * np.cos(prec)
        phase_mod = phase + 0.2 * chi_p * np.sin(prec)

        amp = Mc ** (5 / 6) * f ** (2 / 3) * 1e-21
        return amp * amp_mod * np.cos(phase_mod)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        m1 = rng.uniform(*c.m1_range, n)
        q = rng.uniform(*c.q_range, n)
        chi = rng.uniform(*c.chi_p_range, n)
        return np.stack([m1, q, chi], axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# [D] OVERLAP
# ═══════════════════════════════════════════════════════════════════════

class OverlapComputer:
    """[D] White-noise overlap with a single global time/phase maximization."""

    def __init__(self, cfg: SbankConfig):
        self.cfg = cfg
        self.wfg = ToyPrecessingGenerator(cfg)

    def overlap(self, h1: np.ndarray, h2: np.ndarray) -> float:
        n = len(h1)
        corr = np.fft.irfft(np.fft.rfft(h1) * np.conj(np.fft.rfft(h2)), n=n)
        n1 = math.sqrt(float(np.sum(h1 ** 2)))
        n2 = math.sqrt(float(np.sum(h2 ** 2)))
        if n1 < 1e-60 or n2 < 1e-60:
            return 0.0
        return float(np.max(np.abs(corr)) / (n1 * n2))

    def best_overlap(self, theta: np.ndarray,
                     bank_thetas: np.ndarray,
                     cache: Optional[Dict] = None) -> float:
        """Best overlap of θ against every bank template."""
        if len(bank_thetas) == 0:
            return 0.0
        h = self.wfg.generate(theta)
        best = 0.0
        for b in bank_thetas:
            key = tuple(np.round(b, 8))
            if cache is not None and key in cache:
                hb = cache[key]
            else:
                hb = self.wfg.generate(b)
                if cache is not None:
                    cache[key] = hb
            ov = self.overlap(h, hb)
            if ov > best:
                best = ov
        return best


# ═══════════════════════════════════════════════════════════════════════
# [D] STOCHASTIC BANK
# ═══════════════════════════════════════════════════════════════════════

class StochasticChiPBank:
    """
    [D] sbank-style stochastic insertion + adaptive hot map + pruning.

    The hot map is a memory of UNCOVERED proposals. Feeding it back makes
    the placement concentrate where the metric actually fails, instead of
    wasting proposals in already-covered volume.
    """

    def __init__(self, cfg: Optional[SbankConfig] = None):
        self.cfg = cfg or SbankConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.overlap = OverlapComputer(self.cfg)
        self.wfg = self.overlap.wfg
        self.bank: List[np.ndarray] = []
        self.hot_map: List[np.ndarray] = []
        self._cache: Dict[Tuple, np.ndarray] = {}

    # ── proposal ─────────────────────────────────────────────────────
    def _propose(self) -> np.ndarray:
        if self.hot_map and self.rng.uniform() < self.cfg.hot_map_fraction:
            idx = self.rng.integers(len(self.hot_map))
            # jitter around an uncovered point
            p = self.hot_map[idx].copy()
            scale = np.array([2.0, 0.03, 0.08])
            p += self.rng.normal(0, 1, 3) * scale
            c = self.cfg
            p[0] = np.clip(p[0], *c.m1_range)
            p[1] = np.clip(p[1], *c.q_range)
            p[2] = np.clip(p[2], *c.chi_p_range)
            return p
        return self.wfg.sample_uniform(1, self.rng)[0]

    # ── build ─────────────────────────────────────────────────────────
    def build(self) -> Dict[str, Any]:
        c = self.cfg
        max_mm = 1.0 - c.minimal_match
        bank_array = np.zeros((0, 3))
        n_inserted = n_rejected = 0

        for it in range(c.n_iterations):
            theta = self._propose()
            best_ov = self.overlap.best_overlap(theta, bank_array, self._cache)
            mm = 1.0 - best_ov

            if mm > max_mm:
                bank_array = np.vstack([bank_array, theta])
                n_inserted += 1
            else:
                n_rejected += 1
                # remember uncovered-ish / boundary points for the hot map
                if mm > 0.6 * max_mm and len(self.hot_map) < 400:
                    self.hot_map.append(theta.copy())

        self.bank = [b.copy() for b in bank_array]

        # ── prune against an INTERNAL probe set ───────────────────────
        probes = self.wfg.sample_uniform(c.n_prune_probes, self.rng)
        self._prune(probes)

        # ── INDEPENDENT audit (never used for build or prune) ─────────
        audit_rng = np.random.default_rng(c.seed + 99991)
        audit = self.wfg.sample_uniform(c.n_validate, audit_rng)
        audit_report = self.audit(audit)

        return {
            "n_templates": len(self.bank),
            "n_inserted": n_inserted,
            "n_rejected": n_rejected,
            "hot_map_size": len(self.hot_map),
            "minimal_match_target": c.minimal_match,
            "independent_audit": audit_report,
            "honest_note": (
                "TOY precessing waveforms + white-noise overlap. This is "
                "sbank-STYLE placement, not production sbank. Coverage is "
                "measured on an independent finite sample, not certified "
                "over the continuum space."),
        }

    # ── pruning ───────────────────────────────────────────────────────
    def _prune(self, probes: np.ndarray) -> None:
        """
        Remove a template if the probes still meet the match target
        without it. Greedy, order-fixed for reproducibility.
        """
        c = self.cfg
        target = c.minimal_match

        def covered(bank: List[np.ndarray]) -> bool:
            arr = np.array(bank) if bank else np.zeros((0, 3))
            for p in probes:
                if self.overlap.best_overlap(p, arr, self._cache) < target:
                    return False
            return True

        i = 0
        while i < len(self.bank):
            trial = self.bank[:i] + self.bank[i + 1:]
            if len(trial) > 0 and covered(trial):
                self.bank = trial
                # do not advance: indices shifted
            else:
                i += 1

    # ── independent audit ─────────────────────────────────────────────
    def audit(self, validation_points: np.ndarray) -> Dict[str, Any]:
        arr = np.array(self.bank) if self.bank else np.zeros((0, 3))
        mismatches, coverages = [], []
        for p in validation_points:
            ov = self.overlap.best_overlap(p, arr, self._cache)
            mismatches.append(1.0 - ov)
            coverages.append(ov >= self.cfg.minimal_match)

        mm = np.array(mismatches)
        return {
            "n_validation": len(validation_points),
            "coverage_fraction": float(np.mean(coverages)),
            "median_mismatch": float(np.median(mm)),
            "p95_mismatch": float(np.percentile(mm, 95)),
            "worst_mismatch": float(np.max(mm)),
            "meets_target": bool(np.mean(coverages) >= 0.99),
            "note": "independent sample; not used for build or pruning",
        }

    # ── coverage heat map (m1 × χ_p at median q) ──────────────────────
    def coverage_heatmap(self, n_bins: int = 12) -> Dict[str, Any]:
        c = self.cfg
        q_mid = 0.5 * (c.q_range[0] + c.q_range[1])
        m1_edges = np.linspace(*c.m1_range, n_bins + 1)
        chi_edges = np.linspace(*c.chi_p_range, n_bins + 1)
        heat = np.zeros((n_bins, n_bins))

        arr = np.array(self.bank) if self.bank else np.zeros((0, 3))
        for i in range(n_bins):
            for j in range(n_bins):
                m1 = 0.5 * (m1_edges[i] + m1_edges[i + 1])
                chi = 0.5 * (chi_edges[j] + chi_edges[j + 1])
                p = np.array([m1, q_mid, chi])
                ov = self.overlap.best_overlap(p, arr, self._cache)
                heat[i, j] = ov

        return {
            "m1_edges": m1_edges,
            "chi_p_edges": chi_edges,
            "overlap_heat": heat,
            "uncovered_cells": int(np.sum(heat < c.minimal_match)),
            "note": "diagnostic slice at q = median; full space is 3-D",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Stochastic χ_p Bank (sbank-style) — Smoke Test")
    print("=" * 72)

    cfg = SbankConfig(n_iterations=180, n_validate=80,
                      n_prune_probes=40, n_seed=12)
    bank = StochasticChiPBank(cfg)
    result = bank.build()

    print(f"\n  ── Placement ──")
    print(f"    templates        : {result['n_templates']}")
    print(f"    inserted         : {result['n_inserted']}")
    print(f"    rejected         : {result['n_rejected']}")
    print(f"    hot-map memory   : {result['hot_map_size']}")

    audit = result["independent_audit"]
    print(f"\n  ── INDEPENDENT coverage audit ──")
    print(f"    n_validation     : {audit['n_validation']}")
    print(f"    coverage         : {audit['coverage_fraction']:.3f}")
    print(f"    median mismatch  : {audit['median_mismatch']:.4f}")
    print(f"    p95 mismatch     : {audit['p95_mismatch']:.4f}")
    print(f"    worst mismatch   : {audit['worst_mismatch']:.4f}")
    print(f"    [{'PASS' if audit['meets_target'] else 'WARN'}] "
          f"coverage ≥ 99% target")

    heat = bank.coverage_heatmap(n_bins=8)
    print(f"\n  ── Coverage heat map (m1 × χ_p slice) ──")
    print(f"    uncovered cells  : {heat['uncovered_cells']} / 64")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • TOY waveforms + white-noise overlap, not IMRPhenomP/SEOBNR")
    print("  • Coverage audited on an INDEPENDENT sample (Carpet Thm 3(b))")
    print("  • Finite-sample estimate, not a continuum certificate")
    print("  • sbank-STYLE, not a reimplementation of production sbank")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
