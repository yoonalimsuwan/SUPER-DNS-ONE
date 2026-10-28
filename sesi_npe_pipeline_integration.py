"""
=============================================================================
SESI NPE ↔ LOW-LATENCY PIPELINE INTEGRATION
Posterior-based event ranking (SNR + χ² + NPE) for candidate ordering
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT THIS MODULE ADDS
---------------------
[N1] PosteriorRankingFeatureExtractor
     strain segment → fixed-size feature vector for the NPE
[N2] NPERankingStatistic
     trained NPE → scalar ranking from posterior concentration
[N3] HybridRankingStatistic
     combines reweighted SNR, χ², and NPE score with explicit weights
[N4] NPEAugmentedLowLatencyPipeline
     extends LowLatencyPipeline so triggers are ranked by the hybrid
     statistic, with per-stage latency accounting

DESIGN PRINCIPLES (per cfd_pp.py §3.3 & Six Problems §6)
----------------------------------------------------------
• NPE ranking is an ADDITIONAL feature, not a replacement for the proven
  matched-filter SNR + χ². The hybrid weight for NPE defaults to 0 until
  the NPE passes validation (see sesi_npe_validation).
• The throttler still controls ALERT EMISSION, not detection. NPE never
  suppresses a trigger; it only re-orders candidates.
• System failures (non-finite features, NPE sampling failure) are recorded
  as warnings and fall back to the SNR+χ² ranking — never silently swallowed.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• NPE posterior concentration is a HEURISTIC ranking feature. A trained
  NPE gives p(θ|x), NOT the Bayesian evidence p(x); a true likelihood-ratio
  ranking would need SNRE. We use concentration as a proxy and say so.
• The NPE must be pre-trained AND validated (coverage/KL/SBC) before its
  weight is raised above 0. This module enforces that default.
• Ranking changes candidate ORDER, not the underlying detection threshold,
  until a full sensitivity study is done.
=============================================================================
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.npe_pipeline")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Graceful import of the pipeline + NPE modules built earlier
try:
    from sesi_low_latency_pipeline import (
        PipelineConfig, LowLatencyPipeline, Trigger,
        MatchedFilterEngine, ChiSquaredDiscriminator,
    )
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False

try:
    from sesi_neural_posterior import NeuralPosteriorEstimator, NPEConfig, GWSimulator
    HAS_NPE = True
except ImportError:
    HAS_NPE = False

__all__ = [
    "NPEPipelineConfig",
    "PosteriorRankingFeatureExtractor",
    "NPERankingStatistic",
    "HybridRankingStatistic",
    "NPEAugmentedLowLatencyPipeline",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NPEPipelineConfig:
    """Configuration for NPE-augmented ranking."""
    # Feature extraction
    n_features: int = 16           # size of NPE input feature vector
    segment_seconds: float = 1.0

    # NPE ranking
    n_posterior_samples: int = 200  # samples for concentration estimate
    npe_enabled: bool = False       # MUST stay False until validated [O]
    npe_weight: float = 0.0         # weight of NPE score in hybrid ranking

    # Hybrid ranking weights
    snr_weight: float = 1.0
    chi2_weight: float = 1.0

    # Latency budget
    npe_latency_budget_ms: float = 50.0

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [N1] FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

class PosteriorRankingFeatureExtractor:
    """
    [D] Convert a strain segment into a fixed-size feature vector for the NPE.

    Features are summary statistics chosen to be informative about the
    presence and parameters of a compact-binary signal:
      • peak amplitude, RMS, kurtosis
      • band-limited power in several frequency bins
      • peak frequency

    HONEST: This is a hand-crafted summary, not a learned representation.
    It must be consistent with whatever the NPE was trained on.
    """

    def __init__(self, cfg: NPEPipelineConfig, sample_rate: float = 4096.0):
        self.cfg = cfg
        self.sample_rate = sample_rate

    def extract(self, strain: np.ndarray) -> np.ndarray:
        """Return a (n_features,) feature vector, or None on failure."""
        try:
            if not np.all(np.isfinite(strain)):
                return None  # system integrity issue — caller must log

            feats = []
            x = strain
            feats.append(float(np.max(np.abs(x))))          # peak amplitude
            feats.append(float(np.sqrt(np.mean(x ** 2))))   # RMS
            # kurtosis
            mu = x.mean()
            sd = x.std() + 1e-12
            feats.append(float(np.mean(((x - mu) / sd) ** 4)))

            # band power via FFT
            n = len(x)
            xf = np.fft.rfft(x)
            power = np.abs(xf) ** 2
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

            # split into bins
            n_bins = max(1, self.cfg.n_features - 5)
            edges = np.linspace(0, len(power), n_bins + 1).astype(int)
            for i in range(n_bins):
                lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
                feats.append(float(np.sum(power[lo:hi])))

            # peak frequency
            if len(power) > 0:
                feats.append(float(freqs[int(np.argmax(power))]))
            else:
                feats.append(0.0)

            # pad/truncate to n_features
            feats = np.array(feats, dtype=np.float32)
            if len(feats) < self.cfg.n_features:
                feats = np.pad(feats, (0, self.cfg.n_features - len(feats)))
            else:
                feats = feats[: self.cfg.n_features]

            if not np.all(np.isfinite(feats)):
                return None
            return feats

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# [N2] NPE RANKING STATISTIC
# ═══════════════════════════════════════════════════════════════════════

class NPERankingStatistic:
    """
    [D] Derive a scalar ranking feature from a trained NPE posterior.

    Given features x, sample θ ~ p(θ|x) and compute POSTERIOR CONCENTRATION:
        score = - entropy(posterior)  (normalized)
    A concentrated posterior (low entropy) means the data strongly constrains
    the source parameters → more signal-like.

    HONEST LIMITATIONS:
      • This uses posterior CONCENTRATION, not the Bayesian evidence p(x).
        NPE does not directly give p(x); SNRE would be needed for a true
        likelihood-ratio ranking. Concentration is a proxy. [O]
      • If the NPE was not validated (coverage/KL/SBC), this score is
        untrustworthy and its weight should remain 0.
    """

    def __init__(self, cfg: NPEPipelineConfig, npe: Optional[Any] = None):
        self.cfg = cfg
        self.npe = npe  # a trained NeuralPosteriorEstimator
        self._validated = False

    def mark_validated(self) -> None:
        """Flag that the NPE passed sesi_npe_validation checks."""
        self._validated = True

    @property
    def validated(self) -> bool:
        return self._validated

    def score(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Compute the NPE ranking score for one feature vector.

        Returns dict with 'score', 'ok' flag, and 'latency_ms'.
        """
        t0 = time.time()
        result = {"score": 0.0, "ok": False, "latency_ms": 0.0,
                  "reason": ""}

        if not self.cfg.npe_enabled:
            result["reason"] = "npe_enabled=False (requires validation)"
            result["latency_ms"] = (time.time() - t0) * 1000
            return result

        if self.npe is None:
            result["reason"] = "no trained NPE attached"
            result["latency_ms"] = (time.time() - t0) * 1000
            return result

        if not HAS_TORCH:
            result["reason"] = "torch unavailable"
            result["latency_ms"] = (time.time() - t0) * 1000
            return result

        try:
            # Set the observation and draw posterior samples
            self.npe._x_obs = features
            samples = self.npe.posterior_samples(n=self.cfg.n_posterior_samples)
            if samples is None or len(samples) < 10:
                result["reason"] = "insufficient posterior samples"
                result["latency_ms"] = (time.time() - t0) * 1000
                return result

            # Posterior concentration = negative normalized entropy
            score = self._concentration(samples)
            result["score"] = float(score)
            result["ok"] = True

        except Exception as e:
            # Per cfd_pp.py §3.3: do NOT suppress the failure
            result["reason"] = f"NPE scoring failed: {e}"
            logger.warning(result["reason"])

        result["latency_ms"] = (time.time() - t0) * 1000
        return result

    @staticmethod
    def _concentration(samples: np.ndarray) -> float:
        """
        Negative normalized entropy of the posterior samples.

        We estimate entropy per-dimension via histogram, average, then
        negate so that HIGHER score = MORE concentrated = more signal-like.
        """
        n, d = samples.shape
        entropies = []
        for j in range(d):
            col = samples[:, j]
            lo, hi = col.min(), col.max()
            if hi - lo < 1e-12:
                entropies.append(0.0)  # fully concentrated
                continue
            hist, _ = np.histogram(col, bins=20, range=(lo, hi))
            p = hist / hist.sum()
            p = p[p > 0]
            entropies.append(float(-np.sum(p * np.log(p))))
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        # normalize by log(n_bins) so 0 = max concentration, 1 = uniform
        max_entropy = math.log(20)
        normalized = mean_entropy / max_entropy
        return 1.0 - normalized  # higher = more concentrated


# ═══════════════════════════════════════════════════════════════════════
# [N3] HYBRID RANKING STATISTIC
# ═══════════════════════════════════════════════════════════════════════

class HybridRankingStatistic:
    """
    [D] Combine reweighted SNR, χ², and NPE score into one ranking value.

        ranking = w_snr * reweighted_snr + w_npe * npe_score

    where reweighted_snr down-weights candidates with large χ² (glitch-like):
        reweighted_snr = snr / sqrt((1 + chi2_reduced^3) / 2)   (PyCBC-style)

    The NPE weight defaults to 0 until validation. This keeps the pipeline
    on the PROVEN SNR+χ² footing by default.
    """

    def __init__(self, cfg: NPEPipelineConfig):
        self.cfg = cfg

    def reweighted_snr(self, snr: float, chi2_reduced: float) -> float:
        """[C-style] PyCBC-like reweighted SNR."""
        chi2_term = max(chi2_reduced, 0.0)
        denom = math.sqrt((1.0 + chi2_term ** 3) / 2.0)
        return snr / max(denom, 1e-12)

    def rank(self, snr: float, chi2_reduced: float,
             npe_score: float, npe_ok: bool) -> Dict[str, Any]:
        """Produce the final hybrid ranking and a breakdown."""
        rsnr = self.reweighted_snr(snr, chi2_reduced)

        # Only use NPE if it's both enabled AND produced a valid score
        use_npe = self.cfg.npe_enabled and npe_ok and (self.cfg.npe_weight > 0)
        npe_term = self.cfg.npe_weight * npe_score if use_npe else 0.0

        ranking = self.cfg.snr_weight * rsnr + npe_term

        return {
            "ranking": float(ranking),
            "reweighted_snr": float(rsnr),
            "npe_score": float(npe_score) if use_npe else None,
            "npe_used": use_npe,
            "breakdown": {
                "snr_weight": self.cfg.snr_weight,
                "chi2_weight": self.cfg.chi2_weight,
                "npe_weight": self.cfg.npe_weight if use_npe else 0.0,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [N4] NPE-AUGMENTED PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class NPEAugmentedLowLatencyPipeline:
    """
    [D] Wraps LowLatencyPipeline and replaces the plain ranking with the
    hybrid (SNR + χ² + NPE) ranking.

    Guarantees carried over from cfd_pp.py discipline:
      • NPE re-orders candidates; it does NOT drop triggers.
      • The No-Zeno throttler still gates alert EMISSION only.
      • Feature/NPE failures are logged and fall back to SNR+χ².
    """

    def __init__(self, cfg: NPEPipelineConfig,
                 base_cfg: Optional[Any] = None,
                 npe: Optional[Any] = None):
        self.cfg = cfg
        self.feature_extractor = PosteriorRankingFeatureExtractor(cfg)
        self.npe_ranker = NPERankingStatistic(cfg, npe)
        self.hybrid = HybridRankingStatistic(cfg)
        self.npe_failures: List[str] = []   # never silently swallowed
        self.npe_latencies: List[float] = []

        if HAS_PIPELINE:
            self.base = LowLatencyPipeline(base_cfg) if base_cfg else LowLatencyPipeline()
        else:
            self.base = None
            logger.warning("LowLatencyPipeline unavailable; ranking-only mode")

    def rank_trigger(self, trigger: "Trigger",
                     strain_segment: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Compute the hybrid ranking for one trigger.

        Args:
            trigger: a Trigger with .snr, .chi2
            strain_segment: the data around the trigger (for NPE features)

        Returns:
            dict with ranking + diagnostics
        """
        snr = trigger.snr
        # χ² stored on trigger is reduced chi2 already
        chi2_reduced = max(trigger.chi2, 0.0)

        npe_score = 0.0
        npe_ok = False

        if self.cfg.npe_enabled and strain_segment is not None:
            features = self.feature_extractor.extract(strain_segment)
            if features is None:
                # system integrity: log, don't swallow
                msg = "non-finite/inextractable features — fallback to SNR+chi2"
                self.npe_failures.append(msg)
                logger.warning(msg)
            else:
                s = self.npe_ranker.score(features)
                self.npe_latencies.append(s["latency_ms"])
                if s["ok"]:
                    npe_score = s["score"]
                    npe_ok = True
                else:
                    self.npe_failures.append(s["reason"])
                    logger.warning(f"NPE ranking fallback: {s['reason']}")

        ranked = self.hybrid.rank(snr, chi2_reduced, npe_score, npe_ok)
        ranked["trigger_detector"] = trigger.detector
        ranked["trigger_gps"] = trigger.gps_time
        return ranked

    def health_report(self) -> Dict[str, Any]:
        """Latency + failure accounting for the NPE stage."""
        n_lat = len(self.npe_latencies)
        return {
            "npe_enabled": self.cfg.npe_enabled,
            "npe_weight": self.cfg.npe_weight,
            "npe_validated": self.npe_ranker.validated,
            "n_ranked": n_lat,
            "npe_latency_mean_ms": float(np.mean(self.npe_latencies)) if n_lat else 0.0,
            "npe_latency_max_ms": float(np.max(self.npe_latencies)) if n_lat else 0.0,
            "npe_latency_budget_ms": self.cfg.npe_latency_budget_ms,
            "within_budget": (max(self.npe_latencies, default=0.0)
                              <= self.cfg.npe_latency_budget_ms),
            "n_npe_failures": len(self.npe_failures),
            "note": ("NPE ranking is a heuristic proxy (posterior concentration), "
                     "not a likelihood ratio. Keep npe_weight=0 until validated."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI NPE ↔ Low-Latency Pipeline Integration — Smoke Test")
    print("=" * 72)

    cfg = NPEPipelineConfig(npe_enabled=False)  # default: NPE OFF
    pipeline = NPEAugmentedLowLatencyPipeline(cfg)

    print(f"\n  ── Feature extraction ──")
    rng = np.random.default_rng(42)
    strain = rng.normal(0, 1.0, 4096)
    feats = pipeline.feature_extractor.extract(strain)
    print(f"    features shape : {feats.shape if feats is not None else None}")
    print(f"    all finite     : {bool(np.all(np.isfinite(feats))) if feats is not None else False}")

    # NaN strain must yield None (system integrity, not swallowed)
    bad_strain = strain.copy()
    bad_strain[10] = np.nan
    bad_feats = pipeline.feature_extractor.extract(bad_strain)
    print(f"    NaN input → None (correctly refused): {bad_feats is None}")

    print(f"\n  ── Hybrid ranking (NPE disabled → proven SNR+χ² only) ──")
    if HAS_PIPELINE:
        trig = Trigger(detector="H1", gps_time=1e9, snr=9.0, chi2=1.2,
                       template_idx=0, ranking=0.0)
        ranked = pipeline.rank_trigger(trig, strain)
        print(f"    snr=9.0, chi2=1.2 → ranking={ranked['ranking']:.3f} "
              f"(reweighted_snr={ranked['reweighted_snr']:.3f}, "
              f"npe_used={ranked['npe_used']})")
    else:
        # rank directly via hybrid
        ranked = pipeline.hybrid.rank(snr=9.0, chi2_reduced=1.2,
                                      npe_score=0.0, npe_ok=False)
        print(f"    ranking={ranked['ranking']:.3f} npe_used={ranked['npe_used']}")

    print(f"\n  ── Health report ──")
    report = pipeline.health_report()
    print(f"    npe_enabled      : {report['npe_enabled']}")
    print(f"    npe_weight       : {report['npe_weight']}")
    print(f"    n_npe_failures   : {report['n_npe_failures']}")
    print(f"    note: {report['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • NPE ranking uses posterior CONCENTRATION, a proxy — not evidence.")
    print("  • npe_weight defaults to 0 until NPE passes coverage/KL/SBC.")
    print("  • NPE re-orders candidates; it never drops a trigger.")
    print("  • Feature/NPE failures are logged, never silently swallowed.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
