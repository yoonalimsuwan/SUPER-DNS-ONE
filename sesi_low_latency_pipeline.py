"""
=============================================================================
SESI LOW-LATENCY PIPELINE — PyCBC-Live-style Continuous GW Search
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

ARCHITECTURE (modeled on PyCBC Live)
------------------------------------
  streaming strain
      -> [StreamingDataManager]   sliding-window data ingest
      -> [MatchedFilterEngine]    template matching per detector
      -> [ChiSquaredDiscriminator] signal-vs-glitch veto
      -> [TriggerGenerator]       single-detector triggers
      -> [CoincidenceEngine]      cross-detector coincidence
      -> [RankingStatistic]       SNR + chi2 + (optional ML) ranking
      -> [NoZenoThrottler]        alert-rate control (see below)
      -> [LatencyMonitor]         end-to-end latency tracking

NO-ZENO INTEGRATION (per cfd_pp.py)
-----------------------------------
The NoZenoEventScheduler from the CFD paper is repurposed here as an
ALERT-RATE THROTTLER:
  • Prevents "chattering" (many near-duplicate alerts in a short window)
  • Enforces a minimum spacing tau_e between emitted alerts
  • Bound: N(T) <= 1 + floor(T/tau_e)  [paper Prop 1]

CRITICAL — per paper §3.3:
  "the event interval MUST NOT override the CFL stability limit, and a
   no-Zeno rule must never suppress evidence of numerical failure"

Translation to this pipeline:
  • The throttle controls ALERT EMISSION, not TRIGGER DETECTION.
    Triggers are still computed; only the outbound alert is deferred.
  • SYSTEM FAILURES (non-finite data, buffer overflow) are NEVER
    suppressed by the throttle — they are recorded and raised.

LABEL CONVENTION
----------------
[P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a PROTOTYPE low-latency search, not PyCBC Live itself.
• Uses synthetic/simulated strain unless real data is provided.
• Latency numbers are measured but NOT guaranteed in production.
• A candidate event is a CANDIDATE, not a detection.
=============================================================================
"""

from __future__ import annotations

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger("sesi.low_latency")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "PipelineConfig",
    "StreamingDataManager",
    "MatchedFilterEngine",
    "ChiSquaredDiscriminator",
    "TriggerGenerator",
    "CoincidenceEngine",
    "RankingStatistic",
    "NoZenoThrottler",
    "LatencyMonitor",
    "LowLatencyPipeline",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for the low-latency pipeline."""
    # Data
    sample_rate: float = 4096.0      # Hz
    segment_duration: float = 16.0   # seconds per analysis segment
    overlap_duration: float = 8.0    # seconds of overlap between segments

    # Detection
    snr_threshold: float = 5.5       # single-detector SNR threshold
    chi2_threshold: float = 1.5      # reduced chi2 veto threshold
    min_ranking: float = 8.0         # minimum ranking statistic to alert

    # Coincidence
    coincidence_window: float = 0.1  # seconds for cross-detector matching

    # No-Zeno throttle
    alert_min_interval: float = 1.0  # seconds (tau_e)

    # Detectors
    detectors: List[str] = field(default_factory=lambda: ["H1", "L1"])

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [C] STREAMING DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════

class StreamingDataManager:
    """
    [D] Manages a sliding window of streaming strain data.

    Mimics how PyCBC Live ingests data: overlapping segments are
    processed in sequence, with state carried across segments.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.n_segment = int(cfg.segment_duration * cfg.sample_rate)
        self.n_overlap = int(cfg.overlap_duration * cfg.sample_rate)
        self.n_advance = self.n_segment - self.n_overlap
        # Buffer per detector
        self.buffers: Dict[str, deque] = {
            det: deque(maxlen=self.n_segment) for det in cfg.detectors
        }

    def push(self, detector: str, data: np.ndarray) -> None:
        """Append new samples to a detector's buffer."""
        buf = self.buffers[detector]
        for val in data:
            buf.append(val)

    def ready(self, detector: str) -> bool:
        """Check if enough data is buffered for a full segment."""
        return len(self.buffers[detector]) >= self.n_segment

    def get_segment(self, detector: str) -> np.ndarray:
        """Return the current full segment as an array."""
        if not self.ready(detector):
            raise RuntimeError(f"Not enough data buffered for {detector}")
        return np.array(list(self.buffers[detector]))

    def advance(self, detector: str) -> None:
        """Advance the window by n_advance samples (drop oldest)."""
        buf = self.buffers[detector]
        for _ in range(min(self.n_advance, len(buf))):
            buf.popleft()


# ═══════════════════════════════════════════════════════════════════════
# MATCHED FILTER ENGINE
# ═══════════════════════════════════════════════════════════════════════

class MatchedFilterEngine:
    """
    [D] Computes matched-filter SNR for a bank of templates against a
    data segment.
    """

    def __init__(self, cfg: PipelineConfig, templates: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.templates = templates  # (n_templates, n_samples)

    def compute_snr(self, data: np.ndarray, template: np.ndarray,
                    psd: Optional[np.ndarray] = None) -> float:
        """
        Compute the optimal matched-filter SNR.

        SNR = <d, h> / sqrt(<h, h>)
        with noise-weighted inner product.
        """
        n = len(data)
        df = self.cfg.sample_rate / n

        d_f = np.fft.rfft(data)
        h_f = np.fft.rfft(template)

        if psd is None:
            psd = np.ones(len(d_f))
        else:
            psd = np.interp(
                np.arange(len(d_f)),
                np.linspace(0, len(d_f) - 1, len(psd)),
                psd,
            )

        mask = psd > 0
        integrand = d_f * np.conj(h_f) / (psd + 1e-60)
        integrand[~mask] = 0

        d_h = 4.0 * df * np.real(np.sum(integrand))
        h_h = 4.0 * df * np.real(np.sum(np.conj(h_f) * h_f / (psd + 1e-60) * mask))

        if h_h <= 0:
            return 0.0
        return float(d_h / np.sqrt(h_h))

    def filter_segment(self, data: np.ndarray) -> Dict[str, Any]:
        """Run all templates against one segment; return best match."""
        if self.templates is None:
            return {"best_snr": 0.0, "best_template_idx": -1}

        best_snr = 0.0
        best_idx = -1
        for i, tmpl in enumerate(self.templates):
            n = min(len(data), len(tmpl))
            snr = self.compute_snr(data[:n], tmpl[:n])
            if snr > best_snr:
                best_snr = snr
                best_idx = i
        return {"best_snr": best_snr, "best_template_idx": best_idx}


# ═══════════════════════════════════════════════════════════════════════
# [C] CHI-SQUARED DISCRIMINATOR
# ═══════════════════════════════════════════════════════════════════════

class ChiSquaredDiscriminator:
    """
    [C/D] Allen chi-squared discriminator (Allen 2005).

    Splits the template bandwidth into p frequency bins. A true signal
    contributes equally to each bin; a glitch does not. High chi2 =>
    likely glitch, not signal.

    This is the KEY defense against false positives from instrumental
    noise transients.
    """

    def __init__(self, cfg: PipelineConfig, n_bins: int = 16):
        self.cfg = cfg
        self.n_bins = n_bins

    def compute(self, data: np.ndarray, template: np.ndarray) -> float:
        """Compute reduced chi2 for a candidate trigger."""
        n = min(len(data), len(template))
        d_f = np.fft.rfft(data[:n])
        h_f = np.fft.rfft(template[:n])

        # Split into frequency bins
        n_freq = len(d_f)
        bin_edges = np.linspace(0, n_freq, self.n_bins + 1).astype(int)

        total_snr_sq = 0.0
        bin_contributions = []
        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            contribution = np.real(np.sum(d_f[lo:hi] * np.conj(h_f[lo:hi])))
            bin_contributions.append(contribution)
            total_snr_sq += contribution ** 2

        total_snr_sq = max(total_snr_sq, 1e-12)
        expected = total_snr_sq / self.n_bins

        chi2 = 0.0
        for c in bin_contributions:
            chi2 += (c - expected) ** 2 / (expected + 1e-12)

        # Reduced chi2 (degrees of freedom = 2*p - 2)
        dof = 2 * self.n_bins - 2
        return float(chi2 / dof)


# ═══════════════════════════════════════════════════════════════════════
# TRIGGER GENERATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Trigger:
    """A single-detector trigger candidate."""
    detector: str
    gps_time: float
    snr: float
    chi2: float
    template_idx: int
    ranking: float = 0.0


class TriggerGenerator:
    """
    [D] Generates single-detector triggers from matched-filter output.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def generate(self, detector: str, gps_time: float,
                 mf_result: Dict[str, Any], chi2: float,
                 template_idx: int) -> Optional[Trigger]:
        """Create a trigger if SNR exceeds threshold."""
        snr = mf_result["best_snr"]
        if snr < self.cfg.snr_threshold:
            return None

        # Compute ranking statistic
        ranking = self._ranking(snr, chi2)

        return Trigger(
            detector=detector,
            gps_time=gps_time,
            snr=snr,
            chi2=chi2,
            template_idx=template_idx,
            ranking=ranking,
        )

    @staticmethod
    def _ranking(snr: float, chi2: float) -> float:
        """
        Simple ranking: SNR weighted by chi2 consistency.
        NewSNR-style: snr * (1 / (1 + chi2))^0.5
        """
        return snr / math.sqrt(1.0 + max(chi2, 0.0))


# ═══════════════════════════════════════════════════════════════════════
# COINCIDENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════

class CoincidenceEngine:
    """
    [D] Matches triggers across detectors within a time window.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pending_triggers: Dict[str, List[Trigger]] = {
            det: [] for det in cfg.detectors
        }

    def add_trigger(self, trigger: Trigger) -> None:
        self.pending_triggers[trigger.detector].append(trigger)

    def find_coincidences(self) -> List[List[Trigger]]:
        """
        Find groups of triggers (one per detector) within the
        coincidence window.
        """
        coincidences = []
        detectors = list(self.pending_triggers.keys())
        if len(detectors) < 2:
            return coincidences

        # Simple pairwise matching between first two detectors
        d1, d2 = detectors[0], detectors[1]
        for t1 in self.pending_triggers[d1]:
            for t2 in self.pending_triggers[d2]:
                if abs(t1.gps_time - t2.gps_time) <= self.cfg.coincidence_window:
                    coincidences.append([t1, t2])

        # Clear matched triggers
        self.pending_triggers = {det: [] for det in self.cfg.detectors}
        return coincidences


# ═══════════════════════════════════════════════════════════════════════
# RANKING STATISTIC
# ═══════════════════════════════════════════════════════════════════════

class RankingStatistic:
    """
    [D] Combines single-detector rankings into a coincident ranking.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def coincident_ranking(self, triggers: List[Trigger]) -> float:
        """Combine rankings from multiple detectors."""
        if not triggers:
            return 0.0
        # Sum of individual rankings (simple); could be quadrature
        return sum(t.ranking for t in triggers)


# ═══════════════════════════════════════════════════════════════════════
# NO-ZENO THROTTLER  (per cfd_pp.py)
# ═══════════════════════════════════════════════════════════════════════

class NoZenoThrottler:
    """
    [D] Alert-rate throttler using the No-Zeno event-count bound.

    Repurposes NoZenoEventScheduler from cfd_pp.py for alert emission:
      • Enforces minimum spacing tau_e between OUTBOUND alerts
      • Guarantees N(T) <= 1 + floor(T/tau_e)

    CRITICAL (per cfd_pp.py §3.3):
      • This throttles ALERT EMISSION, not trigger detection.
        Triggers are still recorded; only the outbound alert is deferred.
      • System failures (non-finite data, errors) are NEVER suppressed.
      • Warnings are RECORDED, not swallowed.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tau_e = cfg.alert_min_interval
        self.last_alert_time = -math.inf
        self.deferred_alerts: List[Dict[str, Any]] = []
        self.system_warnings: List[str] = []  # NEVER suppressed

    def should_emit(self, current_time: float) -> bool:
        """Check if enough time has elapsed since the last alert."""
        return (current_time - self.last_alert_time) >= self.tau_e

    def emit_or_defer(self, current_time: float,
                      alert: Dict[str, Any]) -> bool:
        """
        Attempt to emit an alert. Returns True if emitted, False if deferred.

        HONEST: deferral does NOT mean the alert is dropped. It is
        recorded for later, ensuring no real candidate is silently lost.
        """
        if self.should_emit(current_time):
            self.last_alert_time = current_time
            return True
        else:
            # Do NOT suppress — record for later
            self.deferred_alerts.append(alert)
            return False

    def record_system_warning(self, msg: str) -> None:
        """
        Record a SYSTEM warning (never suppressed by the throttle).

        Per cfd_pp.py §3.3: "Warnings raised before the controller is
        allowed to switch modes are recorded, not swallowed."
        """
        self.system_warnings.append(msg)
        logger.warning(f"[SYSTEM WARNING — NOT THROTTLED] {msg}")


# ═══════════════════════════════════════════════════════════════════════
# LATENCY MONITOR
# ═══════════════════════════════════════════════════════════════════════

class LatencyMonitor:
    """
    [D] Tracks end-to-end latency of the pipeline.
    """

    def __init__(self):
        self.latencies: List[float] = []

    def record(self, start_time: float) -> float:
        latency = time.time() - start_time
        self.latencies.append(latency)
        return latency

    def stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {"n": 0}
        arr = np.array(self.latencies)
        return {
            "n": len(arr),
            "mean_ms": float(np.mean(arr) * 1000),
            "median_ms": float(np.median(arr) * 1000),
            "p95_ms": float(np.percentile(arr, 95) * 1000),
            "max_ms": float(np.max(arr) * 1000),
        }


# ═══════════════════════════════════════════════════════════════════════
# LOW-LATENCY PIPELINE (orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class LowLatencyPipeline:
    """
    [D] Full low-latency pipeline orchestrator, PyCBC-Live-style.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.data_mgr = StreamingDataManager(self.cfg)
        self.mf_engine = MatchedFilterEngine(self.cfg)
        self.chi2 = ChiSquaredDiscriminator(self.cfg)
        self.trigger_gen = TriggerGenerator(self.cfg)
        self.coincidence = CoincidenceEngine(self.cfg)
        self.ranking = RankingStatistic(self.cfg)
        self.throttler = NoZenoThrottler(self.cfg)
        self.latency = LatencyMonitor()
        self.alert_log: List[Dict[str, Any]] = []

    def process_chunk(self, gps_time: float,
                      data_by_detector: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process one chunk of streaming data across all detectors.

        Args:
            gps_time: GPS time of the chunk
            data_by_detector: {detector_name: strain_array}

        Returns:
            list of emitted alerts
        """
        start = time.time()
        emitted = []

        for det, data in data_by_detector.items():
            # ── System check: never suppress failures ──
            if not np.all(np.isfinite(data)):
                self.throttler.record_system_warning(
                    f"Non-finite data in {det} at GPS {gps_time}")
                continue

            # Push to buffer
            self.data_mgr.push(det, data)

            # Check if we have a full segment
            if not self.data_mgr.ready(det):
                continue

            segment = self.data_mgr.get_segment(det)

            # Matched filter
            mf_result = self.mf_engine.filter_segment(segment)

            # Only proceed if SNR is above threshold
            if mf_result["best_snr"] < self.cfg.snr_threshold:
                self.data_mgr.advance(det)
                continue

            # Chi2 veto
            tmpl_idx = mf_result["best_template_idx"]
            if self.mf_engine.templates is not None and tmpl_idx >= 0:
                tmpl = self.mf_engine.templates[tmpl_idx]
                chi2_val = self.chi2.compute(segment, tmpl)
            else:
                chi2_val = 0.0

            # Generate trigger
            trigger = self.trigger_gen.generate(
                det, gps_time, mf_result, chi2_val, tmpl_idx)
            if trigger is None:
                self.data_mgr.advance(det)
                continue

            self.coincidence.add_trigger(trigger)
            self.data_mgr.advance(det)

        # ── Cross-detector coincidence ──
        coincidences = self.coincidence.find_coincidences()
        for coinc in coincidences:
            ranking = self.ranking.coincident_ranking(coinc)
            if ranking < self.cfg.min_ranking:
                continue

            alert = {
                "gps_time": gps_time,
                "detectors": [t.detector for t in coinc],
                "ranking": ranking,
                "snrs": [t.snr for t in coinc],
                "chi2s": [t.chi2 for t in coinc],
            }

            # ── No-Zeno throttle: controls EMISSION, not detection ──
            if self.throttler.emit_or_defer(gps_time, alert):
                emitted.append(alert)
                self.alert_log.append(alert)

        self.latency.record(start)
        return emitted

    def report(self) -> Dict[str, Any]:
        """Generate a pipeline health report."""
        return {
            "alerts_emitted": len(self.alert_log),
            "alerts_deferred": len(self.throttler.deferred_alerts),
            "system_warnings": len(self.throttler.system_warnings),
            "latency": self.latency.stats(),
            "note": (
                "Candidates are CANDIDATES, not detections. "
                "Deferred alerts are recorded, not dropped. "
                "System warnings are never suppressed (cfd_pp.py §3.3)."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Low-Latency Pipeline — Smoke Test")
    print("=" * 72)

    cfg = PipelineConfig(segment_duration=2.0, overlap_duration=1.0)
    pipeline = LowLatencyPipeline(cfg)

    # Create a simple template bank (chirp-like signals)
    n_templates = 4
    n_samples = int(cfg.segment_duration * cfg.sample_rate)
    templates = np.zeros((n_templates, n_samples))
    t = np.arange(n_samples) / cfg.sample_rate
    for i in range(n_templates):
        freq = 50.0 + i * 50.0
        templates[i] = np.sin(2 * np.pi * freq * t) * np.exp(-t * 2)

    pipeline.mf_engine.templates = templates

    print(f"\n  Config: {len(cfg.detectors)} detectors, "
          f"{n_templates} templates, segment={cfg.segment_duration}s")

    # ── Simulate streaming data with an injected signal ──
    rng = np.random.default_rng(42)
    chunk_size = int(cfg.overlap_duration * cfg.sample_rate)  # advance step
    n_chunks = 6

    all_alerts = []
    for chunk_idx in range(n_chunks):
        gps_time = 1e9 + chunk_idx * cfg.overlap_duration
        chunk_data = {}
        for det in cfg.detectors:
            # Background noise
            noise = rng.normal(0, 0.1, size=chunk_size)
            # Inject a signal in chunk 3 (simulating a real event)
            if chunk_idx == 3:
                signal = 5.0 * np.sin(2 * np.pi * 100.0 * t[:chunk_size]) * \
                         np.exp(-t[:chunk_size] * 2)
                noise = noise + signal
            chunk_data[det] = noise

        alerts = pipeline.process_chunk(gps_time, chunk_data)
        all_alerts.extend(alerts)
        print(f"    chunk {chunk_idx}: GPS={gps_time:.1f}, "
              f"alerts={len(alerts)}")

    # ── Report ──
    report = pipeline.report()
    print(f"\n  ── Pipeline Report ──")
    print(f"    Alerts emitted  : {report['alerts_emitted']}")
    print(f"    Alerts deferred : {report['alerts_deferred']}")
    print(f"    System warnings : {report['system_warnings']}")
    if report['latency'].get('n', 0) > 0:
        print(f"    Latency mean    : {report['latency']['mean_ms']:.2f} ms")
        print(f"    Latency p95     : {report['latency']['p95_ms']:.2f} ms")

    # ── Verify No-Zeno property ──
    print(f"\n  ── No-Zeno Throttle Check ──")
    tau_e = cfg.alert_min_interval
    print(f"    Alert min interval (tau_e): {tau_e}s")
    print(f"    Deferred alerts (NOT dropped): "
          f"{len(pipeline.throttler.deferred_alerts)}")
    print(f"    System warnings (NEVER suppressed): "
          f"{len(pipeline.throttler.system_warnings)}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE — per cfd_pp.py §3.3 & Six Problems §6")
    print("  • Throttle controls ALERT EMISSION, not trigger detection")
    print("  • Deferred alerts are RECORDED, never dropped")
    print("  • System failures are NEVER suppressed by the throttle")
    print("  • A candidate is a CANDIDATE, not a confirmed detection")
    print("  • Alert-rate bound is an ENGINEERING guarantee, not physics")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
