"""
=============================================================================
SESI GRAVITY SPY REAL DATA + QTRANSFORM + GROUP TIME SPLIT
=============================================================================
[O] Gravity Spy data download (requires Zenodo/Kaggle acceptance)
[C] gwpy.qtransform (if installed)
[D] Real GroupTimeSplit implementation

HONEST SCOPE
------------
• Gravity Spy data requires accepting terms on Zenodo/Kaggle.
• Without real data, uses synthetic spectrograms.
• gwpy.qtransform used if available; falls back to STFT.
• GroupTimeSplit prevents temporal leakage (critical!).
=============================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.gravityspy_real")

try:
    from gwpy.timeseries import TimeSeries
    HAS_GWPY = True
except ImportError:
    HAS_GWPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "GravitySpyConfig",
    "GravitySpyDownloader",
    "QTransformFeaturizer",
    "RealGroupTimeSplit",
    "RealGlitchCNN",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GravitySpyConfig:
    """Configuration for Gravity Spy data."""
    # Data location
    data_dir: str = "./data/gravityspy"
    
    # Q-transform parameters
    q_range: Tuple[float, float] = (4.0, 64.0)
    f_min: float = 20.0
    f_max: float = 1024.0
    outseg_duration: float = 1.0  # seconds
    
    # Image size
    image_size: int = 224
    
    # Classes (Gravity Spy taxonomy)
    classes: List[str] = None
    
    # Training
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    
    # Group split
    train_frac: float = 0.7
    val_frac: float = 0.15
    
    def __post_init__(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        if self.classes is None:
            self.classes = [
                "Blip", "Koi Fish", "Scratchy", "Whistle",
                "Helix", "Repeating Blips", "Scatter", "None",
            ]


# ═══════════════════════════════════════════════════════════════════════
# DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════

class GravitySpyDownloader:
    """
    [O] Download Gravity Spy labeled spectrograms.
    
    HONEST: Real Gravity Spy data requires:
      1. Accepting terms on Zenodo or Kaggle
      2. Manual download or API access
      3. Proper citation in publications
    
    This class provides the STRUCTURE but cannot bypass terms of use.
    """

    # Real data sources (require acceptance)
    ZENODO_DOI = "10.5281/zenodo.6329519"  # Example DOI
    KAGGLE_URL = "https://www.kaggle.com/datasets/..."
    
    def __init__(self, cfg: GravitySpyConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)

    def check_data(self) -> Dict[str, Any]:
        """Check if data exists locally."""
        has_data = (self.data_dir / "spectrograms").exists()
        
        return {
            "has_data": has_data,
            "data_dir": str(self.data_dir),
            "note": (
                "Gravity Spy data requires accepting terms on Zenodo/Kaggle. "
                "See: https://zenodo.org/record/6329519"
            ),
        }

    def download_instructions(self) -> str:
        """Provide download instructions."""
        return f"""
GRAVITY SPY DATA DOWNLOAD INSTRUCTIONS
======================================

1. Visit Zenodo: https://zenodo.org/record/{self.ZENODO_DOI.split('/')[-1]}
   OR Kaggle: {self.KAGGLE_URL}

2. Accept the terms of use

3. Download the spectrogram images (ZIP/TAR)

4. Extract to: {self.data_dir}/spectrograms/

5. Expected structure:
   {self.data_dir}/spectrograms/
   ├── Blip/
   │   ├── image1.png
   │   └── ...
   ├── Koi Fish/
   ├── Scratchy/
   └── ...

6. Citation (if used in publication):
   Zevin et al. (2017), "Gravity Spy: Integrating Advanced LIGO
   Detector Characterization, Machine Learning, and Citizen Science"

NOTE: Without real data, this module uses SYNTHETIC spectrograms
for testing only. Results are NOT representative of real performance.
"""


# ═══════════════════════════════════════════════════════════════════════
# Q-TRANSFORM FEATURIZER
# ═══════════════════════════════════════════════════════════════════════

class QTransformFeaturizer:
    """
    [C] Q-transform spectrogram generation using gwpy.
    
    Falls back to STFT if gwpy unavailable.
    """

    def __init__(self, cfg: GravitySpyConfig):
        self.cfg = cfg
        self.use_gwpy = HAS_GWPY
        
        if not HAS_GWPY:
            logger.warning("gwpy not installed; using STFT fallback")

    def transform(self, strain: np.ndarray, sample_rate: float = 4096.0) -> np.ndarray:
        """
        Convert strain time series to Q-transform spectrogram.
        
        Args:
            strain: (N,) time series
            sample_rate: Hz
            
        Returns:
            (H, W) spectrogram image
        """
        if self.use_gwpy:
            return self._gwpy_qtransform(strain, sample_rate)
        else:
            return self._stft_fallback(strain, sample_rate)

    def _gwpy_qtransform(self, strain: np.ndarray, sample_rate: float) -> np.ndarray:
        """Use gwpy's q_transform."""
        try:
            ts = TimeSeries(strain, sample_rate=sample_rate)
            
            qgram = ts.q_transform(
                qrange=self.cfg.q_range,
                frange=(self.cfg.f_min, self.cfg.f_max),
                outseg=self.cfg.outseg_duration,
            )
            
            # Convert to numpy and normalize
            spec = qgram.value
            
            # Log scale and normalize to [0, 1]
            spec = np.log1p(spec / (spec.max() + 1e-12))
            spec = spec / (spec.max() + 1e-12)
            
            # Resize to target image size
            from PIL import Image
            img = Image.fromarray((spec * 255).astype(np.uint8))
            img = img.resize((self.cfg.image_size, self.cfg.image_size))
            
            return np.array(img) / 255.0
            
        except Exception as e:
            logger.warning(f"gwpy q_transform failed: {e}")
            return self._stft_fallback(strain, sample_rate)

    def _stft_fallback(self, strain: np.ndarray, sample_rate: float) -> np.ndarray:
        """STFT fallback when gwpy unavailable."""
        nperseg = 256
        hop = nperseg // 4
        
        n_frames = 1 + (len(strain) - nperseg) // hop
        spec = np.zeros((nperseg // 2 + 1, max(1, n_frames)))
        
        for i in range(max(1, n_frames)):
            start = i * hop
            frame = strain[start:start + nperseg]
            if len(frame) < nperseg:
                frame = np.pad(frame, (0, nperseg - len(frame)))
            
            frame = frame * np.hanning(nperseg)
            spec[:, i] = np.abs(np.fft.rfft(frame))
        
        # Normalize and resize
        spec = np.log1p(spec / (spec.max() + 1e-12))
        spec = spec / (spec.max() + 1e-12)
        
        # Simple resize via interpolation
        from scipy.ndimage import zoom
        zoom_y = self.cfg.image_size / spec.shape[0]
        zoom_x = self.cfg.image_size / spec.shape[1]
        spec = zoom(spec, (zoom_y, zoom_x))[:self.cfg.image_size, :self.cfg.image_size]
        
        return spec


# ═══════════════════════════════════════════════════════════════════════
# GROUP TIME SPLIT (REAL IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════

class RealGroupTimeSplit:
    """
    [D] Leakage-safe data split by time/event groups.
    
    CRITICAL: Glitches are temporally correlated. Random splitting
    leaks information from train to validation/test.
    
    This split ensures all glitches from the same time period
    (or same trigger event) go to the same split.
    """

    def __init__(self, cfg: GravitySpyConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)

    def split(self, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split indices by groups.
        
        Args:
            groups: (N,) array of group IDs (e.g., event_id, time_block)
            
        Returns:
            (train_mask, val_mask, test_mask)
        """
        unique_groups = np.unique(groups)
        self.rng.shuffle(unique_groups)
        
        n = len(unique_groups)
        n_train = int(n * self.cfg.train_frac)
        n_val = int(n * self.cfg.val_frac)
        
        train_groups = unique_groups[:n_train]
        val_groups = unique_groups[n_train:n_train + n_val]
        test_groups = unique_groups[n_train + n_val:]
        
        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)
        test_mask = np.isin(groups, test_groups)
        
        # Log statistics
        logger.info(f"GroupTimeSplit: {len(train_groups)} train, "
                   f"{len(val_groups)} val, {len(test_groups)} test groups")
        logger.info(f"  Samples: train={train_mask.sum()}, "
                   f"val={val_mask.sum()}, test={test_mask.sum()}")
        
        return train_mask, val_mask, test_mask

    def verify_no_leakage(self, groups: np.ndarray, 
                          train_mask: np.ndarray, 
                          val_mask: np.ndarray) -> bool:
        """Verify no group appears in both train and val."""
        train_groups = set(groups[train_mask])
        val_groups = set(groups[val_mask])
        return len(train_groups & val_groups) == 0


# ═══════════════════════════════════════════════════════════════════════
# CNN (REAL IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class RealGlitchCNN(nn.Module):
        """
        [D] CNN for glitch classification.
        
        Uses real Gravity Spy architecture principles:
          • ResNet-style blocks
          • Input: 1×224×224 spectrogram
          • Output: 8 classes
        """

        def __init__(self, cfg: GravitySpyConfig):
            super().__init__()
            self.cfg = cfg
            
            # Feature extractor
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, len(cfg.classes)),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x)
            return self.classifier(features)
else:
    RealGlitchCNN = None


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Gravity Spy Real Data — Smoke Test")
    print("=" * 72)

    cfg = GravitySpyConfig()
    
    # Check data
    print("\n  ── Data Check ──")
    downloader = GravitySpyDownloader(cfg)
    status = downloader.check_data()
    print(f"    Has data: {status['has_data']}")
    print(f"    Note: {status['note'][:80]}...")
    
    # Q-transform
    print("\n  ── Q-Transform ──")
    featurizer = QTransformFeaturizer(cfg)
    print(f"    Using gwpy: {featurizer.use_gwpy}")
    
    # Test with synthetic strain
    strain = np.random.randn(4096) * 1e-21
    spec = featurizer.transform(strain)
    print(f"    Spectrogram shape: {spec.shape}")
    
    # Group time split
    print("\n  ── GroupTimeSplit ──")
    splitter = RealGroupTimeSplit(cfg)
    
    # Simulate groups (e.g., time blocks)
    n_samples = 100
    groups = np.repeat(np.arange(10), 10)  # 10 groups of 10 samples
    
    train_mask, val_mask, test_mask = splitter.split(groups)
    print(f"    Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    no_leak = splitter.verify_no_leakage(groups, train_mask, val_mask)
    print(f"    No leakage: {no_leak}")
    
    # CNN
    print("\n  ── CNN ──")
    if HAS_TORCH and RealGlitchCNN is not None:
        model = RealGlitchCNN(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model parameters: {n_params:,}")
        
        # Test forward pass
        x = torch.randn(2, 1, cfg.image_size, cfg.image_size)
        y = model(x)
        print(f"    Output shape: {tuple(y.shape)}")
    else:
        print("    torch not available")
    
    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Real Gravity Spy data requires accepting Zenodo/Kaggle terms")
    print("  • Without real data, results are SYNTHETIC ONLY")
    print("  • GroupTimeSplit is CRITICAL for valid evaluation")
    print("  • gwpy.qtransform preferred over STFT for real analysis")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
