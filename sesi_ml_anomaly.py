"""
=============================================================================
SESI ML ANOMALY DETECTION FOR GRAVITATIONAL WAVE DATA
Autoencoder + Isolation Forest for glitch / unmodeled-burst detection
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DETECTS (HONESTLY)
------------------------------------
[M1] Instrumental glitches
     • Short-duration noise transients in detector strain
     • Scattering, blips, whistles, etc. (per Gravity Spy taxonomy)
[M2] Unmodeled bursts
     • Excess power events not matching BBH/BNS templates
     • Candidates for follow-up, NOT confirmed sources
[M3] Beyond-standard-model candidates
     • Signals inconsistent with GR waveforms
     • Requires independent verification before any claim

WHAT THIS MODULE DOES NOT DETECT
--------------------------------
• Extraterrestrial spacecraft or "UFOs"
• Anything requiring physics beyond what the data supports
• Confirmed sources (those are found by matched filtering)

ARCHITECTURE
------------
  Raw strain
      │
      ├─► [A] Spectrogram feature extraction (Q-transform style)
      │
      ├─► [B1] Autoencoder — reconstruction error as anomaly score
      │
      ├─► [B2] Isolation Forest — feature-space anomaly score
      │
      └─► [C] Ensemble score + threshold → anomaly candidates

HONEST SCOPE
------------
• Trained on simulated/real background strain.
• Anomaly scores are PROBABILISTIC, not binary detections.
• Real glitch classification uses Gravity Spy labels; this is a
  research prototype, not a production pipeline.
• No claim that anomalies = new physics or artificial sources.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.ml_anomaly")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "AnomalyConfig",
    "StrainFeaturizer",
    "GWAutoencoder",
    "IsolationForestDetector",
    "EnsembleAnomalyDetector",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyConfig:
    """Configuration for ML anomaly detection."""
    # Feature extraction
    sample_rate: int = 4096
    window_size: int = 4096          # 1 second windows
    hop_size: int = 2048             # 50% overlap
    n_mels: int = 64                 # mel-spectrogram bins

    # Autoencoder
    latent_dim: int = 16
    encoder_hidden: int = 256
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32

    # Anomaly threshold
    reconstruction_threshold: float = 3.0   # sigma above mean
    isolation_contamination: float = 0.05   # expected anomaly fraction

    # Ensemble
    autoencoder_weight: float = 0.6
    isolation_weight: float = 0.4

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [A] FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

class StrainFeaturizer:
    """
    [D] Extract features from strain time series.

    Converts h(t) into spectrogram-like features suitable for
    anomaly detection. Uses a simplified Q-transform / mel-spectrogram
    approach.
    """

    def __init__(self, cfg: AnomalyConfig):
        self.cfg = cfg

    def extract_windows(self, strain: np.ndarray) -> List[np.ndarray]:
        """Split strain into overlapping windows."""
        windows = []
        n = len(strain)
        for start in range(0, n - self.cfg.window_size, self.cfg.hop_size):
            window = strain[start:start + self.cfg.window_size]
            windows.append(window)
        return windows

    def spectrogram(self, window: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrogram of a window.

        Returns: (n_freq, n_time) array.
        """
        n_fft = self.cfg.window_size
        hop = n_fft // 4
        n_frames = 1 + (len(window) - n_fft) // hop

        spec = np.zeros((n_fft // 2 + 1, max(1, n_frames)))
        for i in range(max(1, n_frames)):
            start = i * hop
            frame = window[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            # Hann window
            frame = frame * np.hanning(n_fft)
            fft = np.fft.rfft(frame)
            spec[:, i] = np.abs(fft)

        return spec

    def whiten(self, strain: np.ndarray) -> np.ndarray:
        """
        Whiten strain by its amplitude spectral density.

        This flattens the noise spectrum so anomalies stand out.
        """
        n = len(strain)
        fft = np.fft.rfft(strain)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.cfg.sample_rate)

        # Estimate ASD via median of |fft|
        asd = np.median(np.abs(fft)) + 1e-30
        whitened_fft = fft / asd
        whitened = np.fft.irfft(whitened_fft, n=n)
        return whitened

    def featurize(self, strain: np.ndarray) -> np.ndarray:
        """
        Full feature pipeline: whiten → windows → spectrograms → flatten.

        Returns: (n_windows, feature_dim) array.
        """
        whitened = self.whiten(strain)
        windows = self.extract_windows(whitened)

        features = []
        for w in windows:
            spec = self.spectrogram(w)
            # Downsample frequency axis to n_mels for compactness
            n_freq = spec.shape[0]
            if n_freq > self.cfg.n_mels:
                idx = np.linspace(0, n_freq - 1, self.cfg.n_mels).astype(int)
                spec = spec[idx, :]
            # Summary statistics per window
            feat = np.array([
                spec.mean(),
                spec.std(),
                spec.max(),
                np.percentile(spec, 95),
                np.mean(spec > spec.mean() + 3 * spec.std()),  # excess power frac
            ])
            features.append(feat)

        return np.array(features) if features else np.zeros((1, 5))


# ═══════════════════════════════════════════════════════════════════════
# [B1] AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class GWAutoencoder(nn.Module):
        """
        [D] Autoencoder for anomaly detection.

        Trains on "normal" strain features. Anomalous inputs have
        high reconstruction error.
        """

        def __init__(self, cfg: AnomalyConfig, input_dim: int = 5):
            super().__init__()
            self.cfg = cfg
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, cfg.encoder_hidden),
                nn.ReLU(),
                nn.Linear(cfg.encoder_hidden, cfg.latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(cfg.latent_dim, cfg.encoder_hidden),
                nn.ReLU(),
                nn.Linear(cfg.encoder_hidden, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

        def reconstruction_error(self, x):
            """Per-sample reconstruction error."""
            with torch.no_grad():
                recon = self.forward(x)
                return F.mse_loss(recon, x, reduction="none").mean(dim=-1)

        def fit(self, features: np.ndarray) -> Dict[str, List[float]]:
            """Train the autoencoder on normal features."""
            device = torch.device(self.cfg.device)
            self.to(device)
            self.train()

            x = torch.tensor(features, dtype=torch.float32, device=device)
            # Normalize
            self.mean = x.mean(dim=0)
            self.std = x.std(dim=0) + 1e-8
            x_norm = (x - self.mean) / self.std

            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
            history = []

            for epoch in range(self.cfg.epochs):
                optimizer.zero_grad()
                # Mini-batch
                perm = torch.randperm(len(x_norm))
                epoch_loss = 0.0
                for i in range(0, len(x_norm), self.cfg.batch_size):
                    batch_idx = perm[i:i + self.cfg.batch_size]
                    batch = x_norm[batch_idx]
                    recon = self.forward(batch)
                    loss = F.mse_loss(recon, batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()

                history.append(epoch_loss / max(1, len(x_norm) // self.cfg.batch_size))

            self.eval()
            return {"loss_history": history}

        def anomaly_scores(self, features: np.ndarray) -> np.ndarray:
            """Compute anomaly scores (normalized reconstruction error)."""
            device = torch.device(self.cfg.device)
            x = torch.tensor(features, dtype=torch.float32, device=device)
            x_norm = (x - self.mean.to(device)) / self.std.to(device)
            errors = self.reconstruction_error(x_norm).cpu().numpy()
            return errors
else:
    class GWAutoencoder:
        """Placeholder when torch unavailable."""
        def __init__(self, *a, **kw):
            raise ImportError("torch required for GWAutoencoder")


# ═══════════════════════════════════════════════════════════════════════
# [B2] ISOLATION FOREST
# ═══════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    """
    [D] Isolation Forest anomaly detector.

    Uses sklearn if available; otherwise a minimal implementation.
    Anomalies are points that are "easy to isolate" in random trees.
    """

    def __init__(self, cfg: AnomalyConfig):
        self.cfg = cfg
        self.model = None

    def fit(self, features: np.ndarray) -> None:
        """Fit the isolation forest."""
        try:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=self.cfg.isolation_contamination,
                random_state=42,
            )
            self.model.fit(features)
        except ImportError:
            logger.warning("sklearn unavailable; using distance-based fallback")
            self._fit_fallback(features)

    def _fit_fallback(self, features: np.ndarray) -> None:
        """Fallback: store mean/std for distance-based anomaly."""
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0) + 1e-8

    def anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        if self.model is not None:
            # score_samples returns negative scores; negate for consistency
            return -self.model.score_samples(features)
        else:
            # Fallback: normalized distance from mean
            z = (features - self._mean) / self._std
            return np.linalg.norm(z, axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# [C] ENSEMBLE ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════

class EnsembleAnomalyDetector:
    """
    [D] Ensemble of autoencoder + isolation forest.

    Combines reconstruction-based and feature-space anomaly scores
    into a single calibrated anomaly score.
    """

    def __init__(self, cfg: AnomalyConfig):
        self.cfg = cfg
        self.featurizer = StrainFeaturizer(cfg)
        self.autoencoder = GWAutoencoder(cfg) if HAS_TORCH else None
        self.isolation = IsolationForestDetector(cfg)
        self._fitted = False

    def fit(self, strain_normal: np.ndarray) -> Dict[str, Any]:
        """
        Train on normal (background) strain.

        Args:
            strain_normal: strain time series WITHOUT injected signals
        """
        features = self.featurizer.featurize(strain_normal)
        logger.info(f"Extracted {len(features)} feature windows")

        result = {}

        if self.autoencoder is not None:
            ae_result = self.autoencoder.fit(features)
            result["autoencoder"] = ae_result

        self.isolation.fit(features)
        self._fitted = True
        result["isolation"] = "fitted"
        result["n_features"] = len(features)

        return result

    def score(self, strain: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Score a strain time series for anomalies.

        Returns per-window anomaly scores from each detector + ensemble.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")

        features = self.featurizer.featurize(strain)

        scores = {}

        # Autoencoder scores
        if self.autoencoder is not None:
            ae_scores = self.autoencoder.anomaly_scores(features)
            ae_norm = (ae_scores - ae_scores.mean()) / (ae_scores.std() + 1e-8)
            scores["autoencoder"] = ae_norm

        # Isolation forest scores
        iso_scores = self.isolation.anomaly_scores(features)
        iso_norm = (iso_scores - iso_scores.mean()) / (iso_scores.std() + 1e-8)
        scores["isolation"] = iso_norm

        # Ensemble
        if "autoencoder" in scores:
            ensemble = (self.cfg.autoencoder_weight * scores["autoencoder"] +
                        self.cfg.isolation_weight * scores["isolation"])
        else:
            ensemble = iso_norm
        scores["ensemble"] = ensemble

        return scores

    def detect(self, strain: np.ndarray, threshold_sigma: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies in strain.

        Returns windows flagged as anomalous (score > threshold sigma).
        """
        scores = self.score(strain)
        ensemble = scores["ensemble"]

        anomalies = ensemble > threshold_sigma
        n_anomalies = int(anomalies.sum())

        return {
            "scores": scores,
            "anomaly_mask": anomalies,
            "n_anomalies": n_anomalies,
            "n_windows": len(ensemble),
            "anomaly_rate": n_anomalies / max(len(ensemble), 1),
            "max_score": float(ensemble.max()) if len(ensemble) else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI ML Anomaly Detection — Smoke Test")
    print("=" * 72)

    cfg = AnomalyConfig(epochs=5, device="cpu")
    rng = np.random.RandomState(42)

    # Generate synthetic "normal" background strain (Gaussian noise)
    n_normal = cfg.sample_rate * 4  # 4 seconds
    strain_normal = rng.normal(0, 1e-21, n_normal)

    # Generate test strain with an injected "glitch" (short burst)
    strain_test = rng.normal(0, 1e-21, n_normal)
    glitch_start = n_normal // 2
    glitch_len = cfg.sample_rate // 16  # short burst
    strain_test[glitch_start:glitch_start + glitch_len] += (
        rng.uniform(-1, 1, glitch_len) * 1e-19  # much louder than noise
    )

    # Build and fit detector
    detector = EnsembleAnomalyDetector(cfg)
    print("\n  ── Fitting on background strain ──")
    fit_result = detector.fit(strain_normal)
    print(f"    Feature windows: {fit_result['n_features']}")

    # Score test strain
    print("\n  ── Scoring test strain (with injected glitch) ──")
    result = detector.detect(strain_test, threshold_sigma=2.0)
    print(f"    Windows analyzed: {result['n_windows']}")
    print(f"    Anomalies found: {result['n_anomalies']}")
    print(f"    Anomaly rate: {result['anomaly_rate']:.3f}")
    print(f"    Max score: {result['max_score']:.2f}")

    # The glitch should produce elevated scores near the middle
    ensemble = result["scores"]["ensemble"]
    if len(ensemble) > 2:
        glitch_region = ensemble[len(ensemble)//2 - 1:len(ensemble)//2 + 1]
        print(f"    Scores near glitch: {glitch_region}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Detects glitches/excess-power anomalies, NOT spacecraft")
    print("  • Trained on synthetic Gaussian noise for this smoke test")
    print("  • Real deployment needs real detector noise + Gravity Spy labels")
    print("  • Anomaly ≠ new physics; requires independent verification")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
