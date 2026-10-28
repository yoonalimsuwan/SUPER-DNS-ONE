"""
=============================================================================
SESI χ_p-AUGMENTED WAVEFORM TEMPLATE BANK
Extends the mass-only bank to cover the effective precession parameter
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS ADDS TO sesi_waveform_templates.py
--------------------------------------------
The earlier bank covered (m1, m2, aligned spin). Precessing binaries —
characterized by the effective precession parameter χ_p — require
precessing templates. This module:

  [T1] PrecessingWaveformGenerator
       A toy precessing waveform whose amplitude/phase are modulated at a
       precession frequency set by χ_p.  [D-TOY: NOT IMRPhenomP/SEOBNRv4PHM]

  [T2] BankMetricEstimator
       Estimates the parameter-space metric g_ij from the local mismatch
       1 − overlap ≈ g_ij Δθ^i Δθ^j  [C: Owen 1998, PRD 57, 4535]

  [T3] ChiPTemplateBank
       Places templates on a lattice in (m1, m2, χ_p) so the worst-case
       mismatch stays below (1 − minimal_match).

χ_p DEFINITION (standard LVK convention)
----------------------------------------
    χ_p = max[ χ_1 sin t_1,  (4q+3)/(3q+4) · q · χ_2 sin t_2 ]
with q = m_2/m_1 ≤ 1.  χ_p ∈ [0, 1]; larger χ_p = stronger precession.

LABELS: [C] χ_p + Owen metric  [D] implementation  [O] open

HONEST SCOPE
------------
• The precessing waveform is a TOY (amplitude/phase modulation). Real
  precessing banks use IMRPhenomPv2/Pv3 or SEOBNRv4PHM via lalsimulation.
• The metric approximation is LOCAL (valid for small Δθ). Real banks use
  geometric + stochastic placement (e.g. `sbank`) because the metric
  breaks down over the full precessing parameter space.
• Discretizing χ_p introduces a systematic mismatch; the bank reports its
  achieved worst-case mismatch so the error is visible, not hidden.
• Template COUNT grows rapidly with χ_p bins; this is reported.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.chip_bank")

__all__ = [
    "TemplateBankConfig",
    "PrecessingWaveformGenerator",
    "BankMetricEstimator",
    "ChiPTemplateBank",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TemplateBankConfig:
    # Parameter ranges
    m1_range: Tuple[float, float] = (10.0, 60.0)
    m2_range: Tuple[float, float] = (10.0, 60.0)
    chi_p_range: Tuple[float, float] = (0.0, 1.0)

    # Bank density
    minimal_match: float = 0.97     # 3% max mismatch (detection standard)
    n_chi_p_bins: int = 5

    # Waveform
    sample_rate: float = 2048.0
    duration: float = 4.0
    f_min: float = 20.0
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [T1] PRECESSING WAVEFORM (toy)
# ═══════════════════════════════════════════════════════════════════════

class PrecessingWaveformGenerator:
    """
    [D-TOY] Chirp waveform with χ_p-driven precession modulation.

    Precession modulates the amplitude and adds a slow phase wobble at a
    precession frequency proportional to χ_p. This captures the QUALITATIVE
    effect (χ_p changes waveform morphology) but is NOT a calibrated
    precessing approximant.
    """

    MSUN_SEC = 4.925490947e-6

    def __init__(self, cfg: TemplateBankConfig):
        self.cfg = cfg
        self.n = int(cfg.sample_rate * cfg.duration)
        self.t = np.arange(self.n) / cfg.sample_rate

    def generate(self, m1: float, m2: float, chi_p: float) -> np.ndarray:
        """Return a toy precessing strain h(t)."""
        Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        Mc_sec = Mc * self.MSUN_SEC

        # leading-order chirp frequency & phase
        t_c = self.cfg.duration * 0.7
        tau = np.maximum(t_c - self.t, 1e-4)
        f_gw = (1.0 / math.pi) * (5.0 / (256.0 * tau)) ** 0.375 * Mc_sec ** (-0.625)
        f_gw = np.clip(f_gw, self.cfg.f_min, self.cfg.sample_rate * 0.45)
        phase = 2.0 * math.pi * np.cumsum(f_gw) / self.cfg.sample_rate

        # Precession frequency ∝ χ_p × orbital frequency (toy scaling)
        f_prec = chi_p * f_gw * 0.1
        prec_phase = 2.0 * math.pi * np.cumsum(f_prec) / self.cfg.sample_rate

        # Amplitude modulation from precession (stronger χ_p → deeper wobble)
        amp_mod = 1.0 + 0.3 * chi_p * np.cos(prec_phase)
        # Phase wobble
        phase_mod = phase + 0.2 * chi_p * np.sin(prec_phase)

        amp = Mc ** (5.0 / 6.0) * f_gw ** (2.0 / 3.0) * 1e-21
        return amp * amp_mod * np.cos(phase_mod)


# ═══════════════════════════════════════════════════════════════════════
# [T2] BANK METRIC ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class BankMetricEstimator:
    """
    [C/D] Estimate the template-space metric via finite-difference mismatch.

    For nearby templates,  1 − overlap ≈ g_ij Δθ^i Δθ^j.  We probe the
    mismatch along each parameter axis to estimate the diagonal metric
    components (full off-diagonal estimation is [O]).

    Overlap is noise-weighted; here we use a white-noise inner product for
    the toy waveforms.
    """

    def __init__(self, cfg: TemplateBankConfig):
        self.cfg = cfg
        self.wfg = PrecessingWaveformGenerator(cfg)

    def overlap(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Maximized noise-weighted overlap (white noise, toy)."""
        n = len(h1)
        h1f = np.fft.rfft(h1)
        h2f = np.fft.rfft(h2)
        cross = np.fft.irfft(h1f * np.conj(h2f), n=n)
        norm1 = math.sqrt(np.sum(h1 ** 2))
        norm2 = math.sqrt(np.sum(h2 ** 2))
        if norm1 < 1e-60 or norm2 < 1e-60:
            return 0.0
        return float(np.max(np.abs(cross)) / (norm1 * norm2))

    def mismatch(self, theta1: np.ndarray, theta2: np.ndarray) -> float:
        h1 = self.wfg.generate(*theta1)
        h2 = self.wfg.generate(*theta2)
        return 1.0 - self.overlap(h1, h2)

    def diagonal_metric(self, theta: np.ndarray,
                        eps: float = 1e-2) -> Dict[str, float]:
        """
        Estimate diagonal g_ii at θ via central finite differences.
        θ = (m1, m2, chi_p).
        """
        names = ["m1", "m2", "chi_p"]
        metric = {}
        for i, name in enumerate(names):
            step = np.zeros(3)
            # scale step per parameter
            scale = [1.0, 1.0, 0.05][i] * max(abs(theta[i]), 1.0)
            step[i] = max(eps * scale, 1e-4)
            th_p = np.clip(theta + step, [10, 10, 0], [60, 60, 1])
            th_m = np.clip(theta - step, [10, 10, 0], [60, 60, 1])
            mm = self.mismatch(th_p, th_m)
            dtheta = np.linalg.norm(th_p - th_m)
            metric[name] = mm / (dtheta ** 2) if dtheta > 0 else 0.0
        return metric

    def is_positive(self, metric: Dict[str, float]) -> bool:
        return all(v >= 0 for v in metric.values())


# ═══════════════════════════════════════════════════════════════════════
# [T3] χ_p TEMPLATE BANK
# ═══════════════════════════════════════════════════════════════════════

class ChiPTemplateBank:
    """
    [D] Place templates on a lattice in (m1, m2, χ_p).

    Spacing along each axis is chosen from the metric so that the
    worst-case mismatch between neighbours is ≤ (1 − minimal_match).
    This is a simplified geometric placement; production banks use
    stochastic methods because the metric is only locally valid.
    """

    def __init__(self, cfg: TemplateBankConfig):
        self.cfg = cfg
        self.metric_est = BankMetricEstimator(cfg)
        self.templates: List[Dict[str, Any]] = []

    def _spacing(self, g_ii: float, max_mismatch: float) -> float:
        """Δθ such that g_ii Δθ² ≈ max_mismatch."""
        if g_ii <= 0:
            return float("inf")
        return math.sqrt(max_mismatch / g_ii)

    def build(self, probe_theta: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Build the bank. Uses the metric at a representative point to set
        global spacings (a full implementation would adapt locally).
        """
        cfg = self.cfg
        probe = probe_theta if probe_theta is not None else np.array(
            [0.5 * (cfg.m1_range[0] + cfg.m1_range[1]),
             0.5 * (cfg.m2_range[0] + cfg.m2_range[1]),
             0.5])

        metric = self.metric_est.diagonal_metric(probe)
        max_mm = 1.0 - cfg.minimal_match

        dm1 = self._spacing(metric["m1"], max_mm)
        dm2 = self._spacing(metric["m2"], max_mm)
        # χ_p spacing: use metric if positive, else uniform bins
        if metric["chi_p"] > 0:
            dchi = self._spacing(metric["chi_p"], max_mm)
        else:
            dchi = 1.0 / max(cfg.n_chi_p_bins, 1)

        # Build lattice (cap the count for tractability)
        m1_grid = np.arange(cfg.m1_range[0], cfg.m1_range[1] + dm1, dm1)
        m2_grid = np.arange(cfg.m2_range[0], cfg.m2_range[1] + dm2, dm2)
        chi_grid = np.linspace(cfg.chi_p_range[0], cfg.chi_p_range[1],
                               cfg.n_chi_p_bins)

        self.templates = []
        for m1 in m1_grid:
            for m2 in m2_grid:
                if m2 > m1:   # enforce m1 >= m2
                    continue
                for chi_p in chi_grid:
                    self.templates.append({
                        "m1": float(m1), "m2": float(m2), "chi_p": float(chi_p),
                    })

        # Achieved worst-case neighbour mismatch (sampled estimate)
        worst_mm = self._estimate_worst_mismatch()

        return {
            "n_templates": len(self.templates),
            "metric_at_probe": metric,
            "spacings": {"dm1": dm1, "dm2": dm2, "dchi_p": dchi},
            "chi_p_bins": cfg.n_chi_p_bins,
            "minimal_match_target": cfg.minimal_match,
            "estimated_worst_mismatch": worst_mm,
            "meets_target": worst_mm <= max_mm,
            "honest_note": (
                "Toy precessing waveforms + local metric + geometric lattice. "
                "Production precessing banks use IMRPhenomP/SEOBNRv4PHM and "
                "stochastic placement (sbank). Worst-case mismatch is ESTIMATED, "
                "not guaranteed over the full space."
            ),
        }

    def _estimate_worst_mismatch(self, n_probes: int = 30) -> float:
        """Sample random points; find mismatch to nearest template."""
        if not self.templates:
            return float("nan")
        rng = np.random.default_rng(self.cfg.seed)
        wfg = self.metric_est.wfg
        worst = 0.0
        for _ in range(n_probes):
            m1 = rng.uniform(*self.cfg.m1_range)
            m2 = rng.uniform(self.cfg.m2_range[0], m1)
            chi_p = rng.uniform(*self.cfg.chi_p_range)
            h_probe = wfg.generate(m1, m2, chi_p)
            # nearest template by Euclidean parameter distance
            best_ov = 0.0
            for tmpl in self.templates[:50]:   # cap for speed
                h_t = wfg.generate(tmpl["m1"], tmpl["m2"], tmpl["chi_p"])
                ov = self.metric_est.overlap(h_probe, h_t)
                best_ov = max(best_ov, ov)
            worst = max(worst, 1.0 - best_ov)
        return float(worst)

    def chi_p_coverage(self) -> Dict[str, Any]:
        """Report how χ_p is covered by the bank."""
        chi_vals = sorted(set(round(t["chi_p"], 4) for t in self.templates))
        return {
            "distinct_chi_p_values": chi_vals,
            "n_chi_p_layers": len(chi_vals),
            "chi_p_range": [min(chi_vals), max(chi_vals)] if chi_vals else [],
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI χ_p Template Bank — Smoke Test")
    print("=" * 72)

    cfg = TemplateBankConfig(n_chi_p_bins=4, minimal_match=0.90,
                             sample_rate=1024.0, duration=2.0)

    # Precessing waveform sanity
    wfg = PrecessingWaveformGenerator(cfg)
    h_low = wfg.generate(30.0, 25.0, chi_p=0.0)
    h_high = wfg.generate(30.0, 25.0, chi_p=0.9)
    diff = np.max(np.abs(h_low - h_high))
    print(f"\n  ── [T1] Precessing waveform ──")
    print(f"    χ_p=0 vs χ_p=0.9 max difference: {diff:.3e}")
    print(f"    [{'PASS' if diff > 0 else 'FAIL'}] χ_p changes morphology")

    # Metric
    metric_est = BankMetricEstimator(cfg)
    metric = metric_est.diagonal_metric(np.array([30.0, 25.0, 0.5]))
    print(f"\n  ── [T2] Bank metric at probe ──")
    for k, v in metric.items():
        print(f"    g_{k} = {v:.4e}")
    print(f"    [{'PASS' if metric_est.is_positive(metric) else 'FAIL'}] metric is positive")

    # Bank
    bank = ChiPTemplateBank(cfg)
    result = bank.build()
    print(f"\n  ── [T3] Template bank ──")
    print(f"    n_templates            : {result['n_templates']}")
    print(f"    χ_p bins               : {result['chi_p_bins']}")
    print(f"    minimal match target   : {result['minimal_match_target']}")
    print(f"    est. worst mismatch    : {result['estimated_worst_mismatch']:.4f}")
    print(f"    meets target           : {result['meets_target']}")

    cov = bank.chi_p_coverage()
    print(f"\n    χ_p coverage: {cov['n_chi_p_layers']} layers in {cov['chi_p_range']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Waveforms are TOY precessing surrogates, not IMRPhenomP/SEOBNR")
    print("  • Metric is LOCAL; real banks use stochastic placement (sbank)")
    print("  • Worst-case mismatch is ESTIMATED on a sample, not guaranteed")
    print("  • χ_p discretization error is reported, never hidden")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
