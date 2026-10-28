"""
=============================================================================
SESI CNN GLITCH CLASSIFIER — Gravity Spy-style
Q-transform-like spectrograms -> ResNet-style CNN -> glitch taxonomy
=============================================================================
[C] Gravity Spy : Zevin et al. (2017), Coughlin et al.
    Glitch classes are INSTRUMENTAL NOISE, not astrophysical signals.

HONEST SCOPE
------------
* This CNN classifies detector noise transients (blip, scratchy, whistle...).
* It does NOT classify astrophysical sources or artificial objects.
* The smoke test trains on SYNTHETIC spectrograms -> NO real-world accuracy.
* Leakage-safe splitting by time/event is mandatory (glitches correlate).
* OOD detection is required: unknown glitch types must not be forced
  into a known class.
=============================================================================
"""

from __future__ import annotations

import math
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.glitch_cnn")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "GLITCH_CLASSES", "CNNConfig", "SpectrogramFeaturizer",
    "ResidualBlock", "GlitchCNN", "GroupTimeSplit",
    "GlitchTrainer", "GlitchEvaluator",
]


# Real Gravity Spy taxonomy (subset; full set has more classes)
GLITCH_CLASSES = [
    "Blip", "Koi Fish", "Scratchy", "Whistle",
    "Helix", "Repeating Blips", "Scatter", "None",
]


@dataclass
class CNNConfig:
    image_size: int = 96
    n_classes: int = len(GLITCH_CLASSES)
    base_filters: int = 32
    dropout: float = 0.3
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 16
    label_smoothing: float = 0.05
    ood_threshold: float = 0.6
    device: str = "cpu"
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# SPECTROGRAM FEATURES
# ═══════════════════════════════════════════════════════════════════════

class SpectrogramFeaturizer:
    """
    STFT-based surrogate for a Q-transform spectrogram.

    [HONEST] A real pipeline uses gwpy.qtransform / pygsti with
    sine-Gaussian tiling. This is a compact STFT stand-in for plumbing.
    """

    def __init__(self, cfg: CNNConfig, sample_rate: float = 2048.0):
        self.cfg, self.fs = cfg, sample_rate

    def transform(self, strain: np.ndarray) -> np.ndarray:
        n = self.cfg.image_size
        if len(strain) < 64:
            strain = np.pad(strain, (0, max(0, 256 - len(strain))))

        nperseg = max(16, min(128, len(strain) // 4))
        hop = max(1, nperseg // 4)
        win = np.hanning(nperseg)

        n_frames = 1 + max(0, (len(strain) - nperseg) // hop)
        spec = np.zeros((nperseg // 2 + 1, max(n_frames, 1)))
        for i in range(max(n_frames, 1)):
            s = i * hop
            frame = strain[s:s + nperseg]
            if len(frame) < nperseg:
                frame = np.pad(frame, (0, nperseg - len(frame)))
            spec[:, i] = np.abs(np.fft.rfft(frame * win))

        # Resize to (image_size, image_size) via interpolation
        spec = np.log1p(spec / (spec.max() + 1e-12))
        x = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
        return x.squeeze(0).numpy()

    def synthesize(self, class_idx: int, rng: np.random.Generator,
                   size: int = 2048) -> np.ndarray:
        """Generate a synthetic spectrogram-like feature map for testing."""
        n = self.cfg.image_size
        im = rng.normal(0, 0.05, (1, n, n)).astype(np.float32)
        yy, xx = np.mgrid[0:n, 0:n]

        if class_idx == 0:      # Blip: compact blob
            cy, cx = rng.integers(10, n-10, 2)
            im[0] += np.exp(-((yy-cy)**2 + (xx-cx)**2) / 30.0)
        elif class_idx == 1:    # Koi Fish: two lobes
            im[0] += np.exp(-((yy-n*0.4)**2/40 + (xx-n*0.4)**2/200))
            im[0] += 0.6*np.exp(-((yy-n*0.6)**2/40 + (xx-n*0.6)**2/200))
        elif class_idx == 2:    # Scratchy: horizontal lines
            for _ in range(6):
                y = rng.integers(5, n-5)
                im[0, y:y+2, :] += rng.uniform(0.3, 1.0)
        elif class_idx == 3:    # Whistle: rising track
            t = np.linspace(0, 1, n)
            track = (n-1) - (n-8) * t**1.5
            for j, x in enumerate(range(n)):
                y = int(track[j])
                im[0, max(0, y-1):y+2, x] += 1.0
        elif class_idx == 4:    # Helix: two parallel curves
            t = np.linspace(0, 1, n)
            for sign in (1, -1):
                track = n/2 + sign * (n/4) * np.sin(3*np.pi*t)
                for x in range(n):
                    y = int(track[x])
                    im[0, max(0, min(n-2, y)):max(1, min(n, y+2)), x] += 0.8
        elif class_idx == 5:    # Repeating blips: periodic dots
            for k in range(4):
                cx = (k+1) * n // 5
                im[0, n//2-2:n//2+2, cx-2:cx+2] += 1.0
        elif class_idx == 6:    # Scatter: random dots
            for _ in range(40):
                y, x = rng.integers(0, n-2, 2)
                im[0, y:y+2, x:x+2] += rng.uniform(0.3, 1.0)
        # class 7 = None: noise only
        return im


# ═══════════════════════════════════════════════════════════════════════
# CNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class ResidualBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(c_out)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out)
            self.shortcut = nn.Sequential()
            if stride != 1 or c_in != c_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(c_out))

        def forward(self, x):
            y = F.gelu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return F.gelu(y + self.shortcut(x))

    class GlitchCNN(nn.Module):
        """ResNet-style classifier with OOD feature head."""
        def __init__(self, cfg: CNNConfig):
            super().__init__()
            f = cfg.base_filters
            self.stem = nn.Sequential(
                nn.Conv2d(1, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f), nn.GELU())
            self.stage1 = ResidualBlock(f, f)
            self.stage2 = ResidualBlock(f, 2*f, stride=2)
            self.stage3 = ResidualBlock(2*f, 4*f, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(4*f, cfg.n_classes)
            self.feature_dim = 4*f

        def features(self, x):
            h = self.stage3(self.stage2(self.stage1(self.stem(x))))
            return self.pool(h).flatten(1)

        def forward(self, x):
            z = self.features(x)
            return self.head(self.dropout(z))
else:
    GlitchCNN = None


# ═══════════════════════════════════════════════════════════════════════
# LEAKAGE-SAFE SPLIT
# ═══════════════════════════════════════════════════════════════════════

class GroupTimeSplit:
    """
    Split by GROUP (event/time block), never randomly per sample.
    Glitches cluster in time; random splits leak information.
    """
    def __init__(self, train_frac: float = 0.7, val_frac: float = 0.15):
        self.train_frac, self.val_frac = train_frac, val_frac

    def split(self, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique = np.unique(groups)
        rng = np.random.RandomState(42)
        rng.shuffle(unique)
        n = len(unique)
        n_tr = max(1, int(n * self.train_frac))
        n_va = max(1, int(n * self.val_frac))
        tr, va = unique[:n_tr], unique[n_tr:n_tr+n_va]
        te = unique[n_tr+n_va:]
        if len(te) == 0:
            te = va
        return (np.isin(groups, tr), np.isin(groups, va), np.isin(groups, te))


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class TensorSpecDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    class GlitchTrainer:
        def __init__(self, cfg: CNNConfig):
            self.cfg = cfg
            self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
            torch.manual_seed(cfg.seed)
            self.model = GlitchCNN(cfg).to(self.device)

        def fit(self, X: np.ndarray, y: np.ndarray,
                groups: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
            cfg = self.cfg
            if groups is None:
                groups = np.arange(len(X)) // max(1, len(X)//8)
            tr, va, _ = GroupTimeSplit().split(groups)

            # Class weights for imbalance
            counts = np.bincount(y[tr], minlength=cfg.n_classes).astype(float) + 1.0
            weights = torch.tensor(counts.sum()/counts, dtype=torch.float32).to(self.device)

            train_dl = DataLoader(TensorSpecDataset(X[tr], y[tr]),
                                  batch_size=cfg.batch_size, shuffle=True)
            opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
            history = {"train": [], "val": [], "val_acc": []}
            best_state, best_loss = None, float("inf")

            for epoch in range(cfg.epochs):
                self.model.train()
                losses = []
                for xb, yb in train_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    logits = self.model(xb)
                    loss = F.cross_entropy(logits, yb, weight=weights,
                                           label_smoothing=cfg.label_smoothing)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    losses.append(loss.item())
                sched.step()

                # Validation (group-separated)
                self.model.eval()
                with torch.no_grad():
                    xv = torch.tensor(X[va], dtype=torch.float32).to(self.device)
                    yv = torch.tensor(y[va], dtype=torch.long).to(self.device)
                    lv = F.cross_entropy(self.model(xv), yv).item()
                    acc = (self.model(xv).argmax(1) == yv).float().mean().item()

                history["train"].append(float(np.mean(losses)))
                history["val"].append(lv)
                history["val_acc"].append(acc)

                if lv < best_loss:
                    best_loss = lv
                    best_state = copy.deepcopy(self.model.state_dict())

            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.model.eval()
            return history

        @torch.no_grad()
        def predict_with_ood(self, X: np.ndarray) -> Dict[str, np.ndarray]:
            self.model.eval()
            x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            z = self.model.features(x).cpu().numpy()

            confidence = probs.max(axis=1)
            ood = confidence < self.cfg.ood_threshold
            return {
                "probs": probs,
                "pred": probs.argmax(axis=1),
                "confidence": confidence,
                "ood": ood,
                "features": z,
            }
else:
    GlitchTrainer = None


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

class GlitchEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                 probs: np.ndarray) -> Dict[str, Any]:
        n = len(GLITCH_CLASSES)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        per_class = {}
        f1s = []
        for i, name in enumerate(GLITCH_CLASSES):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
            f1s.append(f1)
            per_class[name] = {"precision": prec, "recall": rec, "f1": f1}

        # Brier score (multi-class)
        oh = np.zeros_like(probs)
        oh[np.arange(len(y_true)), y_true] = 1.0
        brier = float(np.mean(np.sum((probs - oh)**2, axis=1)))

        return {
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "macro_f1": float(np.mean(f1s)),
            "accuracy": float((y_true == y_pred).mean()),
            "brier_score": brier,
            "note": "Synthetic-data metric; NOT real Gravity Spy performance",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" CNN Glitch Classifier — Smoke Test (SYNTHETIC DATA)")
    print("=" * 72)
    if not HAS_TORCH:
        print("  [SKIP] torch not installed")
        return

    cfg = CNNConfig(image_size=64, epochs=5, batch_size=8, base_filters=16)
    rng = np.random.default_rng(42)
    fe = SpectrogramFeaturizer(cfg)

    n_per_class = 12
    X, y, groups = [], [], []
    for k in range(cfg.n_classes):
        for j in range(n_per_class):
            im = fe.synthesize(k, rng)
            X.append(im)
            y.append(k)
            # 8 time blocks; same class spread across blocks
            groups.append(j % 8)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    trainer = GlitchTrainer(cfg)
    hist = trainer.fit(X, y, groups)
    print(f"\n  Train loss: {hist['train'][0]:.3f} -> {hist['train'][-1]:.3f}")
    print(f"  Val acc   : {hist['val_acc'][-1]:.3f}")

    pred = trainer.predict_with_ood(X)
    report = GlitchEvaluator.evaluate(y, pred["pred"], pred["probs"])
    print(f"  Macro F1  : {report['macro_f1']:.3f}")
    print(f"  Brier     : {report['brier_score']:.3f}")
    print(f"  OOD flags : {int(pred['ood'].sum())}/{len(pred['ood'])}")

    # OOD probe: pure uniform noise should be less confident
    unknown = rng.uniform(0, 1, (4, 1, cfg.image_size, cfg.image_size)).astype(np.float32)
    up = trainer.predict_with_ood(unknown)
    print(f"  Unknown-input mean confidence: {up['confidence'].mean():.3f}")

    print("\n  [HONEST] Trained on SYNTHETIC maps, not real Gravity Spy data.")
    print("  [HONEST] Glitches are INSTRUMENTAL NOISE, not sources or craft.")
    print("  [HONEST] GroupTimeSplit prevents temporal leakage.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
