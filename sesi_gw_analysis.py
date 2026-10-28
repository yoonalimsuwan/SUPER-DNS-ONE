"""
=============================================================================
SESI GRAVITATIONAL WAVE ANALYSIS
Real GW data processing + matched filtering + anomaly detection
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[G1] Load real GW strain data from GWOSC (or synthetic fallback)
[G2] Compute power spectral density (PSD)
[G3] Matched filtering with template waveforms
[G4] Signal-to-noise ratio (SNR) computation
[G5] Anomaly detection via Carpet A^11 filter

HONEST SCOPE
------------
• This is a RESEARCH FRAMEWORK, not a production GW analysis pipeline.
• Production codes (PyCBC, LALSuite) use sophisticated template banks.
• We use simplified leading-order PN waveforms for demonstration.
• Anomaly detection is heuristic; no claim of detecting new physics.

LABEL CONVENTION
----------------
[P] proved  [C] cited  [D] derived  [O] open
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.gw_analysis")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy import signal as sp_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sesi_observational_data import ObservationalConfig, GWOSCConnector

__all__ = [
    "GWAnalysisConfig",
    "GWDataProcessor",
    "MatchedFilter",
    "GWAnomalyDetector",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GWAnalysisConfig:
    """Configuration for GW analysis."""
    # Data
    event_name: str = "GW150914"
    detector: str = "H1"
    sample_rate: int = 4096  # Hz

    # Analysis
    f_min: float = 20.0  # Hz
    f_max: float = 1024.0  # Hz
    segment_duration: float = 1.0  # seconds

    # Template
    mass1_solar: float = 36.0  # component mass 1
    mass2_solar: float = 29.0  # component mass 2
    distance_mpc: float = 410.0

    # Anomaly detection
    carpet_level: int = 3
    anomaly_threshold: float = 3.0  # sigma

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [G1-G2] GW DATA PROCESSOR
# ═══════════════════════════════════════════════════════════════════════

class GWDataProcessor:
    """
    Load and preprocess GW strain data.

    Steps:
      1. Load strain from GWOSC (or synthetic)
      2. Bandpass filter (f_min to f_max)
      3. Whiten using PSD estimate
      4. Segment into analysis windows
    """

    def __init__(self, cfg: GWAnalysisConfig):
        self.cfg = cfg
        self.gwosc = GWOSCConnector(ObservationalConfig())

    def load_strain(self) -> Tuple[np.ndarray, float]:
        """Load strain data and sample rate."""
        data = self.gwosc.fetch_event(self.cfg.event_name, self.cfg.detector)
        strain = np.array(data["strain"])
        sample_rate = data["sample_rate"]
        logger.info(f"Loaded {len(strain)} samples at {sample_rate} Hz")
        return strain, sample_rate

    def bandpass_filter(self, strain: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply bandpass filter (f_min to f_max)."""
        if not HAS_SCIPY:
            logger.warning("scipy not available; skipping bandpass filter")
            return strain

        nyquist = 0.5 * sample_rate
        low = self.cfg.f_min / nyquist
        high = min(self.cfg.f_max / nyquist, 0.99)
        b, a = sp_signal.butter(4, [low, high], btype="band")
        filtered = sp_signal.filtfilt(b, a, strain)
        return filtered

    def estimate_psd(self, strain: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate power spectral density using Welch's method."""
        if not HAS_SCIPY:
            # Fallback: flat PSD
            n = len(strain)
            freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
            psd = np.ones_like(freqs) * 1e-46
            return freqs, psd

        freqs, psd = sp_signal.welch(strain, fs=sample_rate, nperseg=1024)
        return freqs, psd

    def whiten(self, strain: np.ndarray, freqs: np.ndarray,
               psd: np.ndarray) -> np.ndarray:
        """Whiten strain data using PSD."""
        n = len(strain)
        strain_fft = np.fft.rfft(strain)
        # Interpolate PSD to match FFT frequencies
        fft_freqs = np.fft.rfftfreq(n, d=1.0/self.cfg.sample_rate)
        psd_interp = np.interp(fft_freqs, freqs, psd, left=psd[0], right=psd[-1])
        # Whiten
        whitened_fft = strain_fft / np.sqrt(psd_interp + 1e-60)
        whitened = np.fft.irfft(whitened_fft, n=n)
        return whitened

    def segment(self, strain: np.ndarray) -> List[np.ndarray]:
        """Segment strain into analysis windows."""
        n_samples = int(self.cfg.segment_duration * self.cfg.sample_rate)
        n_segments = len(strain) // n_samples
        segments = []
        for i in range(n_segments):
            start = i * n_samples
            end = start + n_samples
            segments.append(strain[start:end])
        return segments

    def process(self) -> Dict[str, Any]:
        """Full processing pipeline."""
        strain, sample_rate = self.load_strain()
        filtered = self.bandpass_filter(strain, sample_rate)
        freqs, psd = self.estimate_psd(filtered, sample_rate)
        whitened = self.whiten(filtered, freqs, psd)
        segments = self.segment(whitened)

        return {
            "strain_raw": strain,
            "strain_filtered": filtered,
            "strain_whitened": whitened,
            "freqs": freqs,
            "psd": psd,
            "segments": segments,
            "sample_rate": sample_rate,
            "n_segments": len(segments),
        }


# ═══════════════════════════════════════════════════════════════════════
# [G3-G4] MATCHED FILTER
# ═══════════════════════════════════════════════════════════════════════

class MatchedFilter:
    """
    Matched filtering for GW signal detection.

    Computes SNR by correlating data with template waveform.
    """

    def __init__(self, cfg: GWAnalysisConfig):
        self.cfg = cfg

    def generate_template(self, t: np.ndarray) -> np.ndarray:
        """
        Generate leading-order PN inspiral waveform.

        HONEST: This is a simplified chirp; production codes use
        IMRPhenom or SEOBNR templates.
        """
        # Chirp mass
        m1 = self.cfg.mass1_solar
        m2 = self.cfg.mass2_solar
        mc = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

        # Simplified chirp (Newtonian order)
        # f(t) ∝ (t_c - t)^(-3/8)
        t_c = t[-1] * 0.8  # coalescence time
        tau = np.maximum(t_c - t, 1e-3)
        f_gw = (5.0 / (96.0 * np.pi**(8/3)))**(-3/8) * mc**(-5/8) * tau**(-3/8)
        f_gw = np.clip(f_gw, self.cfg.f_min, self.cfg.f_max)

        # Phase
        phase = 2 * np.pi * np.cumsum(f_gw) / self.cfg.sample_rate

        # Amplitude (simplified)
        amp = mc**(5/6) * f_gw**(2/3) / (self.cfg.distance_mpc * 1e6)
        amp = amp / np.max(amp + 1e-30)  # normalize

        waveform = amp * np.cos(phase)
        return waveform

    def compute_snr(self, data: np.ndarray, template: np.ndarray,
                    psd: np.ndarray) -> float:
        """Compute optimal SNR via matched filtering."""
        n = len(data)
        data_fft = np.fft.rfft(data)
        template_fft = np.fft.rfft(template)
        freqs = np.fft.rfftfreq(n, d=1.0/self.cfg.sample_rate)

        # Interpolate PSD
        psd_interp = np.interp(freqs, np.linspace(0, self.cfg.sample_rate/2, len(psd)),
                               psd, left=psd[0], right=psd[-1])

        # Matched filter SNR
        snr_complex = 4.0 * np.sum(data_fft * np.conj(template_fft) / (psd_interp + 1e-60))
        snr_optimal = np.sqrt(np.abs(snr_complex))

        return float(snr_optimal)

    def analyze_segment(self, segment: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Analyze a single segment."""
        t = np.arange(len(segment)) / self.cfg.sample_rate
        template = self.generate_template(t)
        snr = self.compute_snr(segment, template, psd)
        return {
            "snr": snr,
            "template_max": float(np.max(np.abs(template))),
        }


# ═══════════════════════════════════════════════════════════════════════
# [G5] GW ANOMALY DETECTOR (Carpet A^11)
# ═══════════════════════════════════════════════════════════════════════

class GWAnomalyDetector:
    """
    Detect anomalies in GW data using Carpet A^11 filter.

    The high-order polyharmonic operator amplifies singular features,
    which may indicate anomalous signals.
    """

    def __init__(self, cfg: GWAnalysisConfig):
        self.cfg = cfg
        if HAS_TORCH:
            from sesi_anomalous_relativistic_detector import CarpetHighOrderFilter, DetectorConfig
            carpet_cfg = DetectorConfig(
                carpet_level=cfg.carpet_level,
                polyharmonic_order=22,  # A^11
                device=cfg.device,
            )
            self.carpet_filter = CarpetHighOrderFilter(carpet_cfg)
        else:
            self.carpet_filter = None

    def detect_anomaly(self, segment: np.ndarray) -> Dict[str, Any]:
        """Detect anomaly in a GW segment."""
        if not HAS_TORCH or self.carpet_filter is None:
            return {"anomaly_score": 0.0, "is_anomalous": False}

        # Convert to tensor
        signal = torch.tensor(segment, dtype=torch.float32)

        # Pad/truncate to match carpet size
        n_vertices = self.carpet_filter.carpet.graph.n_vertices
        if len(signal) < n_vertices:
            signal = torch.nn.functional.pad(signal, (0, n_vertices - len(signal)))
        else:
            signal = signal[:n_vertices]

        # Apply A^11 filter
        filtered = self.carpet_filter(signal)
        anomaly_score = self.carpet_filter.anomaly_score(signal)

        is_anomalous = float(anomaly_score) > self.cfg.anomaly_threshold

        return {
            "anomaly_score": float(anomaly_score),
            "is_anomalous": is_anomalous,
            "filtered_norm": float(filtered.norm()),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Gravitational Wave Analysis — Smoke Test")
    print("=" * 72)

    cfg = GWAnalysisConfig()

    # Test 1: Data processing
    print("\n  ── [G1-G2] GW Data Processing ──")
    processor = GWDataProcessor(cfg)
    data = processor.process()
    print(f"    Raw strain samples: {len(data['strain_raw'])}")
    print(f"    Segments: {data['n_segments']}")
    print(f"    Sample rate: {data['sample_rate']} Hz")

    # Test 2: Matched filter
    print("\n  ── [G3-G4] Matched Filter ──")
    mf = MatchedFilter(cfg)
    if data["segments"]:
        segment = data["segments"][0]
        result = mf.analyze_segment(segment, data["psd"])
        print(f"    SNR: {result['snr']:.2f}")
        print(f"    Template max: {result['template_max']:.4e}")

    # Test 3: Anomaly detection
    print("\n  ── [G5] Anomaly Detection (Carpet A^11) ──")
    detector = GWAnomalyDetector(cfg)
    if data["segments"]:
        segment = data["segments"][0]
        anomaly = detector.detect_anomaly(segment)
        print(f"    Anomaly score: {anomaly['anomaly_score']:.4f}")
        print(f"    Is anomalous: {anomaly['is_anomalous']}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • This is a RESEARCH FRAMEWORK, not production GW analysis")
    print("  • Template is simplified; real codes use IMRPhenom/SEOBNR")
    print("  • Anomaly detection is heuristic; no claim of new physics")
    print("  • All results require independent verification")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
