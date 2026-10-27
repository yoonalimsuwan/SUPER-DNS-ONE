"""
=============================================================================
SESI TRAINING DASHBOARD
Real-time monitoring of training progress via TensorBoard + console summary
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

FEATURES
────────
  • TensorBoard integration (scalar, histogram, text logging)
  • Console progress display with key metrics
  • Automatic best-model tracking
  • Training-time estimation (ETA)
  • Export summary report as JSON

USAGE
─────
  # During training:
  dashboard = TrainingDashboard(log_dir="./logs/run_001")
  for step, loss_dict in training_loop:
      dashboard.log_step(step, loss_dict)
  dashboard.finish()

  # View in browser:
  tensorboard --logdir ./logs

HONEST SCOPE
────────────
• TensorBoard must be installed: pip install tensorboard
• Dashboard does not alter training; it only observes and records.
• ETA estimates assume roughly constant step time.
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("sesi.dashboard")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

__all__ = ["TrainingDashboard", "TrainingConfig"]


@dataclass
class TrainingConfig:
    """Minimal training config for the dashboard."""
    total_steps: int = 10000
    experiment_name: str = "sesi_run"
    log_dir: str = "./logs"


class TrainingDashboard:
    """
    Monitors and logs training progress.

    Integrates with TensorBoard for visualization and keeps an
    in-memory record of key metrics for the final summary report.
    """

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.log_dir = Path(cfg.log_dir) / cfg.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer (optional)
        self.writer = None
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoard logging to: {self.log_dir}")
        else:
            logger.warning("tensorboard not installed — console-only logging")

        # In-memory history
        self.history: Dict[str, List[float]] = {}
        self.best_metrics: Dict[str, float] = {}
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.step_times: List[float] = []

    # ── Logging ────────────────────────────────────────────────────────
    def log_step(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a single training step.

        Args:
            step: global step number
            metrics: dict of metric_name → scalar value
                     e.g. {"loss": 0.5, "lr": 3e-4, "accuracy": 0.8}
        """
        now = time.time()
        self.step_times.append(now - self.last_step_time)
        self.last_step_time = now

        for name, value in metrics.items():
            # Record history
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

            # Track best (for loss-like metrics, lower is better)
            lower_is_better = any(k in name.lower() for k in ["loss", "error", "mse"])
            if name not in self.best_metrics:
                self.best_metrics[name] = value
            elif lower_is_better and value < self.best_metrics[name]:
                self.best_metrics[name] = value
            elif not lower_is_better and value > self.best_metrics[name]:
                self.best_metrics[name] = value

            # TensorBoard scalar
            if self.writer is not None:
                self.writer.add_scalar(f"train/{name}", value, step)

    def log_histogram(self, step: int, name: str, values: torch.Tensor):
        """Log a histogram (e.g., weight distribution)."""
        if self.writer is not None and values.numel() > 0:
            self.writer.add_histogram(f"train/{name}", values, step)

    def log_text(self, step: int, tag: str, text: str):
        """Log a text message."""
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    # ── Progress display ───────────────────────────────────────────────
    def print_progress(self, step: int, metrics: Dict[str, float]):
        """Print a one-line progress update to the console."""
        total = self.cfg.total_steps
        pct = 100.0 * step / total if total > 0 else 0.0

        # ETA
        if self.step_times:
            avg_step = sum(self.step_times[-20:]) / len(self.step_times[-20:])
            remaining = (total - step) * avg_step
            eta_str = self._format_duration(remaining)
        else:
            eta_str = "?"

        # Elapsed
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_duration(elapsed)

        # Format metrics
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())

        bar_len = 30
        filled = int(bar_len * step / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\r[{bar}] {pct:5.1f}%  step {step}/{total}  "
              f"elapsed {elapsed_str}  ETA {eta_str}  {metric_str}",
              end="", flush=True)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as H:MM:SS."""
        if seconds < 0 or not (seconds == seconds):  # negative or NaN
            return "?"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # ── Checkpoint coordination ────────────────────────────────────────
    def is_best(self, metric_name: str, value: float) -> bool:
        """Check if the given value is the best seen so far for this metric."""
        if metric_name not in self.best_metrics:
            return True
        lower_is_better = any(k in metric_name.lower() for k in ["loss", "error", "mse"])
        if lower_is_better:
            return value <= self.best_metrics[metric_name]
        return value >= self.best_metrics[metric_name]

    # ── Final report ───────────────────────────────────────────────────
    def finish(self) -> Dict[str, Any]:
        """
        Finalize training: close TensorBoard, write summary report.

        Returns the summary dict.
        """
        total_time = time.time() - self.start_time

        summary = {
            "experiment": self.cfg.experiment_name,
            "total_steps": self.cfg.total_steps,
            "total_time_s": total_time,
            "total_time_human": self._format_duration(total_time),
            "best_metrics": self.best_metrics,
            "final_metrics": {k: v[-1] for k, v in self.history.items() if v},
            "avg_step_time_ms": (
                1000.0 * sum(self.step_times) / len(self.step_times)
                if self.step_times else 0.0
            ),
        }

        # Save summary
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to: {summary_path}")

        # Close TensorBoard
        if self.writer is not None:
            self.writer.close()

        # Print final summary
        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Experiment   : {summary['experiment']}")
        print(f"  Total time   : {summary['total_time_human']}")
        print(f"  Avg step     : {summary['avg_step_time_ms']:.1f} ms")
        print(f"  Best metrics :")
        for k, v in summary["best_metrics"].items():
            print(f"    {k}: {v:.6f}")
        print("=" * 70)

        return summary


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Demonstrate the dashboard with a synthetic training loop."""
    print("=" * 70)
    print("  SESI Training Dashboard — Smoke Test")
    print("=" * 70)

    cfg = TrainingConfig(
        total_steps=100,
        experiment_name="smoke_test",
        log_dir="/tmp/sesi_logs",
    )
    dashboard = TrainingDashboard(cfg)

    # Simulate training
    import math
    for step in range(1, 101):
        # Fake decreasing loss with noise
        loss = 2.0 * math.exp(-step / 30.0) + 0.1 * (hash(step) % 10) / 10.0
        accuracy = 1.0 - loss / 3.0
        metrics = {"loss": loss, "accuracy": accuracy, "lr": 3e-4}
        dashboard.log_step(step, metrics)
        dashboard.print_progress(step, metrics)
        time.sleep(0.01)  # simulate work

    summary = dashboard.finish()
    print(f"\n  Summary keys: {list(summary.keys())}")
    print(f"  Best loss: {summary['best_metrics'].get('loss', 'N/A')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
