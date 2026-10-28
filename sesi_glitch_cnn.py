"""
=============================================================================
SESI GLITCH CLASSIFICATION CNN
Gravity Spy-style classification of instrumental glitches in GW strain data
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

BACKGROUND [C]
--------------
"Glitches" are short-duration instrumental noise transients in GW
detectors. The Gravity Spy project (Zevin et al. 2017, Class. Quantum
Grav. 34, 064003) uses CNNs to classify glitches into ~28 categories
based on their Q-transform spectrograms.

Common glitch classes (from Gravity Spy):
  • Blip: short, high-frequency, isolated
  • Chirp: frequency increases over time (like GW signal!)
  • Koi Fish: paired low-frequency features
  • Paired Doves: two symmetric frequency tracks
  • Scattered Light: arch-shaped features
  • Whistle: narrow-band frequency increase
  • Helix: spiral pattern
  • Scratchy: broadband noise bursts
  • Tomte: short low-frequency
  • Repeating Blips: series of blips
  • And ~18 more...

WHAT THIS MODULE PROVIDES
-------------------------
[G1] GlitchTaxonomy — class definitions with physics descriptions
[G2] QTransformSpectrogram — time-frequency representation
[G3] GlitchDataset — synthetic + real data loading
[G4] GlitchCNN1D — 1D convolutional network on raw strain
[G5] GlitchCNN2D — 2D CNN on spectrograms (Gravity Spy style)
[G6] GlitchClassifier — training + evaluation pipeline

HONEST SCOPE
------------
• Real Gravity Spy uses labeled human-classified glitches from
  LIGO Open Data. This module provides the ARCHITECTURE and
  SYNTHETIC data generation for offline development.
• Synthetic glitches are simplified models of real classes.
• Production glitch classification requires real detector data.
• This is NOT a substitute for Gravity Spy, but a compatible framework.

LABEL CONVENTION
----------------
[P] proved  [C] cited  [D] derived here  [O] open
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.glitch_cnn")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "GlitchClass",
    "GlitchTaxonomy",
    "QTransformSpectrogram",
    "GlitchDataset",
    "GlitchCNN1D",
    "GlitchCNN2D",
    "GlitchClassifier",
]


# ═══════════════════════════════════════════════════════════════════════
# [G1] GLITCH TAXONOMY
# ═══════════════════════════════════════════════════════════════════════

class GlitchClass(Enum):
    """
    [C] Gravity Spy glitch classes (subset).
    Full list has ~28 classes; we implement the 12 most common.
    """
    BLIP = auto()                  # Short high-frequency isolated
    CHIRP = auto()                 # Frequency increases (GW-like!)
    KOI_FISH = auto()              # Paired low-frequency
    PAIRED_DOVES = auto()          # Symmetric frequency tracks
    SCATTERED_LIGHT = auto()       # Arch-shaped
    WHISTLE = auto()               # Narrow-band rising
    HELIX = auto()                 # Spiral pattern
    SCRATCHY = auto()              # Broadband burst
    TOMTE = auto()                 # Short low-frequency
    REPEATING_BLIPS = auto()       # Series of blips
    POWER_LINE = auto()            # 60 Hz comb
    NONE = auto()                  # Clean Gaussian noise


class GlitchTaxonomy:
    """
    [C] Descriptions and physical characteristics of glitch classes.

    Based on Gravity Spy documentation and Zevin et al. (2017).
    """

    DESCRIPTIONS = {
        GlitchClass.BLIP: {
            "description": "Short duration (<50ms), high frequency (50-200 Hz), "
                          "isolated burst. One of the most common LIGO glitches.",
            "frequency_range": (50, 200),
            "duration_ms": (5, 50),
            "physics": "Unknown; possibly related to control system artifacts",
        },
        GlitchClass.CHIRP: {
            "description": "Frequency increases over time, similar to GW "
                          "inspiral signals. Important to distinguish from real GWs.",
            "frequency_range": (30, 300),
            "duration_ms": (100, 1000),
            "physics": "Can mimic binary inspiral; requires careful vetting",
        },
        GlitchClass.KOI_FISH: {
            "description": "Two paired low-frequency features resembling fish.",
            "frequency_range": (20, 60),
            "duration_ms": (200, 500),
            "physics": "Possibly related to suspension or seismic coupling",
        },
        GlitchClass.PAIRED_DOVES: {
            "description": "Symmetric frequency tracks, like two birds.",
            "frequency_range": (30, 100),
            "duration_ms": (100, 300),
            "physics": "Coupled oscillator artifact",
        },
        GlitchClass.SCATTERED_LIGHT: {
            "description": "Arch-shaped features from scattered laser light.",
            "frequency_range": (10, 100),
            "duration_ms": (500, 2000),
            "physics": "Light scattered from optics recombining with main beam",
        },
        GlitchClass.WHISTLE: {
            "description": "Narrow-band frequency that increases over time.",
            "frequency_range": (100, 500),
            "duration_ms": (50, 200),
            "physics": "Possibly electronics or control loop artifact",
        },
        GlitchClass.HELIX: {
            "description": "Spiral pattern in time-frequency plane.",
            "frequency_range": (20, 80),
            "duration_ms": (300, 1000),
            "physics": "Rotating machinery or coupled oscillation",
        },
        GlitchClass.SCRATCHY: {
            "description": "Broadband noise burst, no clear structure.",
            "frequency_range": (10, 500),
            "duration_ms": (10, 100),
            "physics": "Possibly seismic or anthropogenic noise",
        },
        GlitchClass.TOMTE: {
            "description": "Short low-frequency glitch.",
            "frequency_range": (20, 50),
            "duration_ms": (10, 50),
            "physics": "Named after Scandinavian folklore creature",
        },
        GlitchClass.REPEATING_BLIPS: {
            "description": "Series of blip glitches at regular intervals.",
            "frequency_range": (50, 150),
            "duration_ms": (100, 500),
            "physics": "Periodic instrumental artifact",
        },
        GlitchClass.POWER_LINE: {
            "description": "60 Hz (or 50 Hz) comb and harmonics.",
            "frequency_range": (50, 400),
            "duration_ms": (1000, 10000),
            "physics": "Electrical mains coupling",
        },
        GlitchClass.NONE: {
            "description": "Clean Gaussian noise, no glitch.",
            "frequency_range": (0, 0),
            "duration_ms": (0, 0),
            "physics": "Stationary detector noise",
        },
    }

    @classmethod
    def describe(cls, glitch_class: GlitchClass) -> Dict[str, Any]:
        return cls.DESCRIPTIONS[glitch_class]

    @classmethod
    def all_classes(cls) -> List[GlitchClass]:
        return list(cls.DESCRIPTIONS.keys())


# ═══════════════════════════════════════════════════════════════════════
# [G2] Q-TRANSFORM SPECTROGRAM
# ═══════════════════════════════════════════════════════════════════════

class QTransformSpectrogram:
    """
    [D] Q-transform spectrogram generator.

    The Q-transform is the standard time-frequency representation used
    by Gravity Spy. Unlike STFT, it uses constant-Q wavelets, giving
    better frequency resolution at low frequencies and better time
    resolution at high frequencies.

    HONEST SCOPE:
    • This is a simplified implementation using Gaussian wavelets.
    • Real LIGO Q-transform uses specific whitening and tiling.
    • Output is compatible in shape with Gravity Spy inputs.
    """

    def __init__(self, sample_rate: float = 4096.0,
                 f_min: float = 20.0, f_max: float = 1024.0,
                 q_range: Tuple[float, float] = (4.0, 64.0),
                 n_freq_bins: int = 64, n_time_bins: int = 64):
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.q_range = q_range
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins

    def transform(self, strain: np.ndarray,
                  t_center: Optional[float] = None) -> np.ndarray:
        """
        Compute Q-transform spectrogram.

        Args:
            strain: (n_samples,) time series
            t_center: center time (seconds); defaults to middle

        Returns:
            (n_freq_bins, n_time_bins) spectrogram (log-scaled power)
        """
        n = len(strain)
        if t_center is None:
            t_center = n / (2 * self.sample_rate)

        # Time window around center
        window_dur = 1.0  # 1 second window
        t_start = max(0, int((t_center - window_dur / 2) * self.sample_rate))
        t_end = min(n, t_start + int(window_dur * self.sample_rate))
        segment = strain[t_start:t_end]

        # FFT of segment
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / self.sample_rate)

        # Frequency bins (log-spaced for constant-Q)
        f_bins = np.geomspace(self.f_min, self.f_max, self.n_freq_bins)
        t_bins = np.linspace(0, window_dur, self.n_time_bins)

        spectrogram = np.zeros((self.n_freq_bins, self.n_time_bins))

        for i, f0 in enumerate(f_bins):
            # Q varies with frequency (constant-Q tiling)
            Q = self.q_range[0] + (self.q_range[1] - self.q_range[0]) * \
                (np.log(f0 / self.f_min) / np.log(self.f_max / self.f_min))

            # Time resolution: Δt = Q / (2π f0)
            dt = Q / (2 * math.pi * f0)
            n_window = max(3, int(dt * self.sample_rate))

            # Gaussian wavelet at frequency f0
            for j, t0 in enumerate(t_bins):
                # Extract local window
                i_center = int(t0 * self.sample_rate)
                i_start = max(0, i_center - n_window // 2)
                i_end = min(len(segment), i_start + n_window)
                if i_end <= i_start:
                    continue

                local = segment[i_start:i_end]
                # Multiply by Gaussian window
                t_local = np.arange(len(local)) / self.sample_rate - len(local) / (2 * self.sample_rate)
                gauss = np.exp(-t_local ** 2 / (2 * (dt / 2) ** 2))
                windowed = local * gauss[:len(local)]

                # Project onto frequency f0
                osc = np.exp(-2j * math.pi * f0 * t_local[:len(windowed)])
                coeff = np.abs(np.sum(windowed * osc)) / (np.sqrt(np.sum(gauss[:len(windowed)] ** 2)) + 1e-10)
                spectrogram[i, j] = coeff ** 2

        # Log-scale and normalize
        spectrogram = np.log1p(spectrogram / (spectrogram.max() + 1e-10))
        return spectrogram


# ═══════════════════════════════════════════════════════════════════════
# [G3] GLITCH DATASET
# ═══════════════════════════════════════════════════════════════════════

class GlitchDataset:
    """
    [D] Dataset for glitch classification.

    Supports:
    • Synthetic glitch generation (for offline development)
    • Real Gravity Spy data loading (if available)

    HONEST SCOPE:
    • Synthetic glitches are simplified models.
    • Real Gravity Spy data requires downloading from Zenodo/Gravity Spy.
    """

    def __init__(self, sample_rate: float = 4096.0,
                 duration: float = 1.0, seed: int = 42):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.rng = np.random.RandomState(seed)

    def generate_synthetic(self, glitch_class: GlitchClass) -> np.ndarray:
        """
        Generate a synthetic glitch of the specified class.

        Returns: (n_samples,) strain array with glitch embedded.
        """
        # Background Gaussian noise
        strain = self.rng.normal(0, 1e-22, self.n_samples)

        t = np.arange(self.n_samples) / self.sample_rate
        t_center = self.duration / 2

        if glitch_class == GlitchClass.BLIP:
            # Short high-frequency burst
            mask = (t > t_center - 0.01) & (t < t_center + 0.01)
            strain[mask] += 5e-22 * np.sin(2 * math.pi * 100 * t[mask]) * \
                           np.exp(-((t[mask] - t_center) / 0.005) ** 2)

        elif glitch_class == GlitchClass.CHIRP:
            # Frequency increases over time
            mask = (t > t_center - 0.2) & (t < t_center + 0.2)
            f0, f1 = 50.0, 200.0
            tau = t[mask] - (t_center - 0.2)
            f_inst = f0 + (f1 - f0) * tau / 0.4
            phase = 2 * math.pi * np.cumsum(f_inst) / self.sample_rate
            strain[mask] += 3e-22 * np.cos(phase) * np.exp(-((tau - 0.2) / 0.1) ** 2)

        elif glitch_class == GlitchClass.WHISTLE:
            # Narrow-band rising tone
            mask = (t > t_center - 0.1) & (t < t_center + 0.1)
            tau = t[mask] - (t_center - 0.1)
            f_inst = 200.0 + 300.0 * tau / 0.2
            phase = 2 * math.pi * np.cumsum(f_inst) / self.sample_rate
            strain[mask] += 4e-22 * np.sin(phase)

        elif glitch_class == GlitchClass.SCRATCHY:
            # Broadband burst
            mask = (t > t_center - 0.05) & (t < t_center + 0.05)
            strain[mask] += self.rng.normal(0, 5e-22, mask.sum())

        elif glitch_class == GlitchClass.TOMTE:
            # Short low-frequency
            mask = (t > t_center - 0.02) & (t < t_center + 0.02)
            strain[mask] += 6e-22 * np.sin(2 * math.pi * 30 * t[mask]) * \
                           np.exp(-((t[mask] - t_center) / 0.01) ** 2)

        elif glitch_class == GlitchClass.POWER_LINE:
            # 60 Hz + harmonics throughout
            for harmonic in [1, 2, 3, 5]:
                strain += 1e-22 / harmonic * np.sin(
                    2 * math.pi * 60 * harmonic * t)

        elif glitch_class == GlitchClass.KOI_FISH:
            # Two paired low-frequency features
            for offset in [-0.1, 0.1]:
                mask = (t > t_center + offset - 0.05) & (t < t_center + offset + 0.05)
                strain[mask] += 4e-22 * np.sin(2 * math.pi * 35 * t[mask]) * \
                               np.exp(-((t[mask] - t_center - offset) / 0.03) ** 2)

        elif glitch_class == GlitchClass.REPEATING_BLIPS:
            # Series of blips
            for k in range(5):
                t_k = t_center - 0.2 + k * 0.1
                mask = (t > t_k - 0.005) & (t < t_k + 0.005)
                strain[mask] += 3e-22 * np.sin(2 * math.pi * 80 * t[mask])

        elif glitch_class == GlitchClass.SCATTERED_LIGHT:
            # Arch-shaped (frequency sweeps down then up)
            mask = (t > t_center - 0.3) & (t < t_center + 0.3)
            tau = t[mask] - t_center
            f_inst = 50.0 + 50.0 * (tau / 0.3) ** 2
            phase = 2 * math.pi * np.cumsum(f_inst) / self.sample_rate
            strain[mask] += 3e-22 * np.cos(phase)

        elif glitch_class == GlitchClass.PAIRED_DOVES:
            # Two symmetric frequency tracks
            for sign in [-1, 1]:
                mask = (t > t_center - 0.1) & (t < t_center + 0.1)
                tau = t[mask] - t_center
                f_inst = 60.0 + sign * 20.0 * tau / 0.1
                phase = 2 * math.pi * np.cumsum(f_inst) / self.sample_rate
                strain[mask] += 2e-22 * np.cos(phase)

        elif glitch_class == GlitchClass.HELIX:
            # Spiral (frequency modulated)
            mask = (t > t_center - 0.25) & (t < t_center + 0.25)
            tau = t[mask] - (t_center - 0.25)
            f_inst = 40.0 + 30.0 * np.sin(2 * math.pi * 3 * tau / 0.5)
            phase = 2 * math.pi * np.cumsum(f_inst) / self.sample_rate
            strain[mask] += 3e-22 * np.cos(phase)

        # NONE: just noise
        return strain

    def create_training_set(self, n_per_class: int = 100) -> Dict[str, Any]:
        """
        Create a synthetic training set.

        Returns dict with 'strain' (N, n_samples) and 'labels' (N,).
        """
        classes = GlitchTaxonomy.all_classes()
        all_strain = []
        all_labels = []

        for cls in classes:
            for _ in range(n_per_class):
                strain = self.generate_synthetic(cls)
                all_strain.append(strain)
                all_labels.append(cls.value)

        return {
            "strain": np.array(all_strain),
            "labels": np.array(all_labels),
            "class_names": {cls.value: cls.name for cls in classes},
            "n_classes": len(classes),
        }


# ═══════════════════════════════════════════════════════════════════════
# [G4] GLITCH CNN 1D
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class GlitchCNN1D(nn.Module):
        """
        [D] 1D CNN for glitch classification on raw strain.

        Architecture:
          Conv1d(1→32) → ReLU → MaxPool
          Conv1d(32→64) → ReLU → MaxPool
          Conv1d(64→128) → ReLU → GlobalAvgPool
          FC(128→n_classes)

        HONEST: 1D CNNs work but spectrogram-based 2D CNNs (Gravity Spy)
        generally perform better due to the visual structure of glitches.
        """

        def __init__(self, n_classes: int = 12, n_samples: int = 4096):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=30)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(128, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, 1, n_samples) or (B, n_samples)."""
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
# [G5] GLITCH CNN 2D (Gravity Spy style)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class GlitchCNN2D(nn.Module):
        """
        [D] 2D CNN for glitch classification on Q-transform spectrograms.

        Architecture (Gravity Spy-inspired):
          Conv2d(1→32, 3×3) → ReLU → MaxPool(2×2)
          Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)
          Conv2d(64→128, 3×3) → ReLU → GlobalAvgPool
          FC(128→64) → ReLU → Dropout
          FC(64→n_classes)
        """

        def __init__(self, n_classes: int = 12, img_size: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, 1, img_size, img_size) or (B, img_size, img_size)."""
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = self.features(x)
            return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════
# [G6] GLITCH CLASSIFIER (Training + Evaluation)
# ═══════════════════════════════════════════════════════════════════════

class GlitchClassifier:
    """
    [D] Complete glitch classification pipeline.

    Supports both 1D (raw strain) and 2D (spectrogram) models.
    """

    def __init__(self, model_type: str = "2d", device: str = "cpu"):
        if not HAS_TORCH:
            raise ImportError("torch required for GlitchClassifier")
        self.model_type = model_type
        self.device = torch.device(device)
        self.model = None
        self.qtransform = QTransformSpectrogram()

    def prepare_data(self, dataset: Dict[str, Any],
                     use_spectrogram: bool = True) -> Tuple[Any, Any]:
        """
        Convert dataset to tensors, optionally via Q-transform.

        Args:
            dataset: from GlitchDataset.create_training_set()
            use_spectrogram: if True, convert to spectrograms

        Returns:
            (X, y) tensors
        """
        strain = dataset["strain"]
        labels = dataset["labels"]

        if use_spectrogram:
            # Convert each sample to spectrogram
            specs = []
            for s in strain:
                spec = self.qtransform.transform(s)
                specs.append(spec)
            X = torch.tensor(np.array(specs), dtype=torch.float32)
        else:
            # Normalize raw strain
            X = torch.tensor(strain, dtype=torch.float32)
            X = (X - X.mean()) / (X.std() + 1e-10)

        y = torch.tensor(labels, dtype=torch.long)
        return X, y

    def build_model(self, n_classes: int, img_size: int = 64) -> None:
        """Build the appropriate model."""
        if self.model_type == "1d":
            self.model = GlitchCNN1D(n_classes=n_classes).to(self.device)
        else:
            self.model = GlitchCNN2D(n_classes=n_classes, img_size=img_size).to(self.device)

    def train(self, X: torch.Tensor, y: torch.Tensor,
              n_classes: int, epochs: int = 10, batch_size: int = 32,
              lr: float = 1e-3) -> Dict[str, List[float]]:
        """
        Train the classifier.

        Returns training history.
        """
        if self.model is None:
            img_size = X.shape[-1] if X.dim() >= 3 else 64
            self.build_model(n_classes, img_size)

        X, y = X.to(self.device), y.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(X))
            epoch_loss, correct, total = 0.0, 0, 0

            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X[idx], y[idx]

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(xb)
                correct += (logits.argmax(-1) == yb).sum().item()
                total += len(xb)

            history["loss"].append(epoch_loss / total)
            history["accuracy"].append(correct / total)
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"loss={history['loss'][-1]:.4f}, "
                       f"acc={history['accuracy'][-1]:.3f}")

        return history

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Evaluate on test set."""
        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        logits = self.model(X)
        preds = logits.argmax(-1)

        acc = (preds == y).float().mean().item()
        return {
            "accuracy": acc,
            "n_correct": int((preds == y).sum()),
            "n_total": len(y),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Glitch Classification CNN — Smoke Test")
    print("=" * 72)

    # ── [G1] Taxonomy ───────────────────────────────────────────────
    print("\n  ── [G1] Glitch Taxonomy ──")
    classes = GlitchTaxonomy.all_classes()
    print(f"    Classes defined: {len(classes)}")
    for cls in classes[:5]:
        desc = GlitchTaxonomy.describe(cls)
        print(f"      {cls.name}: {desc['frequency_range']} Hz, "
              f"{desc['duration_ms']} ms")

    # ── [G2] Q-transform ────────────────────────────────────────────
    print("\n  ── [G2] Q-Transform Spectrogram ──")
    qt = QTransformSpectrogram(n_freq_bins=32, n_time_bins=32)
    test_strain = np.random.randn(4096) * 1e-22
    spec = qt.transform(test_strain)
    print(f"    Spectrogram shape: {spec.shape}")
    print(f"    Value range: [{spec.min():.3f}, {spec.max():.3f}]")

    # ── [G3] Dataset ────────────────────────────────────────────────
    print("\n  ── [G3] Synthetic Dataset ──")
    dataset_gen = GlitchDataset(sample_rate=4096.0, duration=1.0)
    dataset = dataset_gen.create_training_set(n_per_class=10)
    print(f"    Total samples: {len(dataset['strain'])}")
    print(f"    Classes: {dataset['n_classes']}")

    # ── [G4/G5] Models ──────────────────────────────────────────────
    if HAS_TORCH:
        print("\n  ── [G4/G5] CNN Models ──")

        # 2D model (Gravity Spy style)
        X, y = GlitchClassifier(model_type="2d").prepare_data(dataset, use_spectrogram=True)
        print(f"    Input shape (spectrogram): {tuple(X.shape)}")

        model_2d = GlitchCNN2D(n_classes=dataset["n_classes"], img_size=X.shape[-1])
        n_params = sum(p.numel() for p in model_2d.parameters())
        print(f"    GlitchCNN2D parameters: {n_params:,}")

        # Forward pass test
        out = model_2d(X[:4])
        print(f"    Output shape: {tuple(out.shape)}")

        # 1D model
        X1d, y1d = GlitchClassifier(model_type="1d").prepare_data(dataset, use_spectrogram=False)
        model_1d = GlitchCNN1D(n_classes=dataset["n_classes"], n_samples=X1d.shape[-1])
        out_1d = model_1d(X1d[:4])
        print(f"    GlitchCNN1D output: {tuple(out_1d.shape)}")

        # ── [G6] Training ───────────────────────────────────────────
        print("\n  ── [G6] Training (quick, 3 epochs) ──")
        classifier = GlitchClassifier(model_type="2d")
        history = classifier.train(X, y, n_classes=dataset["n_classes"],
                                   epochs=3, batch_size=16)
        print(f"    Final loss: {history['loss'][-1]:.4f}")
        print(f"    Final accuracy: {history['accuracy'][-1]:.3f}")

        # Evaluation
        eval_result = classifier.evaluate(X, y)
        print(f"    Eval accuracy: {eval_result['accuracy']:.3f}")
    else:
        print("\n  [SKIP] torch not available")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Synthetic glitches are SIMPLIFIED models")
    print("  • Real Gravity Spy uses human-labeled LIGO data")
    print("  • Production requires real detector strain")
    print("  • This is a compatible FRAMEWORK, not a replacement")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
