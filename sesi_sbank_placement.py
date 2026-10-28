"""
=============================================================================
SESI STOCHASTIC TEMPLATE PLACEMENT (sbank-style) FOR χ_p BANKS
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHY STOCHASTIC INSTEAD OF METRIC (from sesi_chip_template_bank.py)
------------------------------------------------------------------
The earlier χ_p bank used a LOCAL metric + geometric lattice. That breaks
down for precessing banks because:
  • The parameter-space metric varies strongly across (m1, m2, χ_p, tilts).
  • Precessing waveform morphology changes non-linearly with χ_p.
  • A single probe-point metric cannot capture the global structure.

The sbank algorithm [C: Babak et al. 2010; Cokelaer & Fazi 2014] places
templates STOCHASTICALLY:
    1. Draw a candidate template from the target parameter distribution.
    2. Compute its mismatch against all ALREADY-ACCEPTED templates.
    3. If min-mismatch > threshold → ACCEPT; else REJECT.
    4. Repeat until the target number or coverage is reached.

This is robust to curvature and does not need an explicit metric, at the
cost of more templates and more pairwise mismatch evaluations.

LABELS: [C] sbank algorithm  [D] derived implementation  [O] open

HONEST SCOPE
------------
• Coverage is VERIFIED by holding out random test points and measuring
  their mismatch to the nearest accepted template — an ESTIMATE, not a
  guarantee over the continuum (Carpet-19 discipline: check against a
  reference, do not assume convergence).
• Pairwise mismatch scales as O(N²); this implementation caps candidates
  and reports the actual cost. Production sbank uses metric-assisted
  or KD-tree acceleration.
• The toy precessing waveforms from sesi_chip_template_bank are used;
  swap in lalsimulation IMRPhenomP for science use.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.sbank")

# Reuse the precessing waveform generator from the χ_p bank module
try:
    from sesi_chip_template_bank import (
        TemplateBankConfig, PrecessingWaveformGenerator, BankMetricEstimator,
    )
    HAS_CHIP_BANK = True
except ImportError:
    HAS_CHIP_BANK = False

__all__ = ["SBankConfig", "RandomTemplateSampler", "SBankPlacement",
           "BankCoverageReport"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SBankConfig:
    # Parameter space (must match the waveform generator)
    m1_range: Tuple[float, float] = (10.0, 60.0)
    m2_range: Tuple[float, float] = (10.0, 60.0)
    chi_p_range: Tuple[float, float] = (0.0, 1.0)

    # Placement
    minimal_match: float = 0.97        # accept if mismatch > (1 - this)
    max_candidates: int = 3000         # trial templates drawn
    max_templates: int = 500           # cap on accepted templates
    # Coverage verification
    n_test_points: int = 200
    # Waveform
    sample_rate: float = 1024.0
    duration: float = 2.0
    f_min: float = 20.0
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [D] RANDOM TEMPLATE SAMPLER
# ═══════════════════════════════════════════════════════════════════════

class RandomTemplateSampler:
    """
    [D] Draw candidate templates from the target parameter distribution.

    Uses uniform-in-volume sampling for the masses and uniform for χ_p.
    m2 ≤ m1 is enforced. (A production sampler would use the astrophysical
    population from sesi_population_full.py.)
    """

    def __init__(self, cfg: SBankConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def sample_one(self) -> Dict[str, float]:
        m1 = self.rng.uniform(*self.cfg.m1_range)
        m2 = self.rng.uniform(self.cfg.m2_range[0], m1)
        chi_p = self.rng.uniform(*self.cfg.chi_p_range)
        return {"m1": float(m1), "m2": float(m2), "chi_p": float(chi_p)}

    def sample_many(self, n: int) -> List[Dict[str, float]]:
        return [self.sample_one() for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════
# [D] SBANK PLACEMENT
# ═══════════════════════════════════════════════════════════════════════

class SBankPlacement:
    """
    [C/D] The sbank stochastic placement algorithm.

    Accepts a candidate only if its mismatch to EVERY accepted template
    exceeds the mismatch threshold (i.e. it is not redundant).
    """

    def __init__(self, cfg: SBankConfig,
                 waveform_gen: Optional[Any] = None,
                 metric_est: Optional[Any] = None):
        self.cfg = cfg
        if waveform_gen is not None:
            self.wfg = waveform_gen
            self.overlap_fn = metric_est.overlap if metric_est else None
        elif HAS_CHIP_BANK:
            wcfg = TemplateBankConfig(
                sample_rate=cfg.sample_rate, duration=cfg.duration, f_min=cfg.f_min)
            self.wfg = PrecessingWaveformGenerator(wcfg)
            self.overlap_fn = BankMetricEstimator(wcfg).overlap
        else:
            raise RuntimeError("sesi_chip_template_bank not available and no "
                               "waveform generator supplied")

        self.accepted: List[Dict[str, Any]] = []
        self._waveform_cache: Dict[int, np.ndarray] = {}

    def _waveform(self, params: Dict[str, float]) -> np.ndarray:
        return self.wfg.generate(params["m1"], params["m2"], params["chi_p"])

    def _mismatch(self, h1: np.ndarray, h2: np.ndarray) -> float:
        return 1.0 - self.overlap_fn(h1, h2)

    def place(self) -> Dict[str, Any]:
        """
        Run the sbank loop. Returns the accepted bank + placement statistics.
        """
        cfg = self.cfg
        sampler = RandomTemplateSampler(cfg, seed=cfg.seed)
        mismatch_threshold = 1.0 - cfg.minimal_match

        n_drawn = 0
        n_accepted = 0
        n_rejected = 0
        n_mismatch_evals = 0

        while (n_drawn < cfg.max_candidates and
               n_accepted < cfg.max_templates):
            cand = sampler.sample_one()
            h_cand = self._waveform(cand)
            n_drawn += 1

            # Check against all accepted templates
            is_redundant = False
            for acc in self.accepted:
                mm = self._mismatch(h_cand, acc["waveform"])
                n_mismatch_evals += 1
                if mm < mismatch_threshold:
                    is_redundant = True
                    break   # early exit: already covered

            if not is_redundant:
                self.accepted.append({
                    "m1": cand["m1"], "m2": cand["m2"],
                    "chi_p": cand["chi_p"], "waveform": h_cand,
                })
                n_accepted += 1
            else:
                n_rejected += 1

        return {
            "n_templates": len(self.accepted),
            "n_candidates_drawn": n_drawn,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "n_mismatch_evaluations": n_mismatch_evals,
            "minimal_match_target": cfg.minimal_match,
            "chi_p_values": sorted(set(
                round(t["chi_p"], 3) for t in self.accepted)),
        }

    def templates(self) -> List[Dict[str, float]]:
        """Return accepted templates (without waveforms)."""
        return [{"m1": t["m1"], "m2": t["m2"], "chi_p": t["chi_p"]}
                for t in self.accepted]


# ═══════════════════════════════════════════════════════════════════════
# [D] COVERAGE VERIFICATION (hold-out test points)
# ═══════════════════════════════════════════════════════════════════════

class BankCoverageReport:
    """
    [D] Verify bank coverage by measuring the mismatch of held-out random
    points to their nearest accepted template.

    HONEST: this is an ESTIMATE of coverage from a finite test sample, not
    a proof over the continuum. The worst-case test mismatch and the
    fraction of test points meeting the minimal-match target are reported.
    """

    def __init__(self, cfg: SBankConfig, placement: SBankPlacement):
        self.cfg = cfg
        self.placement = placement
        self.rng = np.random.default_rng(cfg.seed + 1)

    def evaluate(self) -> Dict[str, Any]:
        cfg = self.cfg
        sampler = RandomTemplateSampler(cfg, seed=cfg.seed + 1)
        mismatch_threshold = 1.0 - cfg.minimal_match

        worst_mm = 0.0
        n_covered = 0
        mms = []

        for _ in range(cfg.n_test_points):
            test = sampler.sample_one()
            h_test = self.placement._waveform(test)
            best_mm = float("inf")
            for acc in self.placement.accepted:
                mm = self.placement._mismatch(h_test, acc["waveform"])
                if mm < best_mm:
                    best_mm = mm
            mms.append(best_mm)
            worst_mm = max(worst_mm, best_mm)
            if best_mm <= mismatch_threshold:
                n_covered += 1

        mms = np.array(mms)
        return {
            "n_test_points": cfg.n_test_points,
            "worst_case_mismatch": float(worst_mm),
            "mean_mismatch": float(np.mean(mms)),
            "fraction_covered": n_covered / cfg.n_test_points,
            "meets_target": worst_mm <= mismatch_threshold,
            "minimal_match_target": cfg.minimal_match,
            "honest_note": (
                "Coverage estimated from a FINITE hold-out sample; not a "
                "guarantee over the continuum. Increase n_test_points and "
                "max_templates for a tighter bank."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Stochastic (sbank) Template Placement — Smoke Test")
    print("=" * 72)

    cfg = SBankConfig(
        minimal_match=0.90,          # relaxed for the toy demo
        max_candidates=400,
        max_templates=60,
        n_test_points=50,
        sample_rate=1024.0, duration=2.0,
    )

    # Placement
    print("\n  ── [1] sbank placement ──")
    placement = SBankPlacement(cfg)
    result = placement.place()
    print(f"    candidates drawn       : {result['n_candidates_drawn']}")
    print(f"    templates accepted     : {result['n_templates']}")
    print(f"    rejected (redundant)   : {result['n_rejected']}")
    print(f"    mismatch evaluations   : {result['n_mismatch_evaluations']}")
    print(f"    distinct χ_p values    : {len(result['chi_p_values'])}")

    # Coverage verification
    print("\n  ── [2] Coverage verification (hold-out) ──")
    report = BankCoverageReport(cfg, placement).evaluate()
    print(f"    worst-case mismatch    : {report['worst_case_mismatch']:.4f}")
    print(f"    mean mismatch          : {report['mean_mismatch']:.4f}")
    print(f"    fraction covered       : {report['fraction_covered']:.2%}")
    print(f"    meets target (MM={report['minimal_match_target']}) : "
          f"{report['meets_target']}")
    print(f"    {report['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • sbank is robust to parameter-space curvature (no global metric)")
    print("  • Coverage is a FINITE-SAMPLE estimate, not a continuum guarantee")
    print("  • Pairwise mismatch is O(N²); production uses acceleration")
    print("  • Toy precessing waveforms; use IMRPhenomP/SEOBNR for science")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
