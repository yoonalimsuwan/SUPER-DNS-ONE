"""
=============================================================================
SESI PRE-TRAINING VALIDATION SUITE
Verify everything is ready before committing to a (costly) training run
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS VALIDATES
───────────────────
  [1] Environment   — Python version, PyTorch, CUDA, required packages
  [2] Data          — files exist, shapes valid, no NaN/Inf, class balance
  [3] Model         — forward pass, backward pass, gradient flow, param count
  [4] Memory        — estimated model + batch memory fits on device
  [5] Integration   — data → model → loss → optimizer pipeline works end-to-end
  [6] Reproducibility — seed produces deterministic results

HONEST SCOPE
────────────
• This is a pre-flight check, not a guarantee of training success.
• Memory estimates are approximate (real usage varies with fragmentation).
• Passing all checks does NOT mean the model will converge — only that
  the plumbing is correct. Convergence depends on data quality,
  hyperparameters, and model architecture.
=============================================================================
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("sesi.pre_train")

__all__ = ["PreTrainValidator", "ValidationReport"]


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION REPORT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ValidationReport:
    """Aggregated results of all validation checks."""
    checks: List[Dict[str, Any]] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    duration_s: float = 0.0

    def add(self, name: str, status: str, message: str = "",
            details: Optional[Dict] = None):
        self.checks.append({
            "name": name,
            "status": status,       # PASS | FAIL | WARN
            "message": message,
            "details": details or {},
        })
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.warnings += 1

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        lines = ["=" * 70, "  PRE-TRAINING VALIDATION REPORT", "=" * 70]
        for c in self.checks:
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}.get(c["status"], "?")
            lines.append(f"  {icon} [{c['status']:4s}] {c['name']}")
            if c["message"]:
                lines.append(f"         └─ {c['message']}")
        lines.append("-" * 70)
        lines.append(f"  Passed: {self.passed}  |  Failed: {self.failed}  "
                     f"|  Warnings: {self.warnings}")
        lines.append(f"  Duration: {self.duration_s:.2f}s")
        lines.append("=" * 70)
        if self.all_passed:
            lines.append("  ✅ READY TO TRAIN")
        else:
            lines.append("  ❌ FIX FAILURES BEFORE TRAINING")
        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class PreTrainValidator:
    """
    Runs a battery of checks before training.

    Usage:
        validator = PreTrainValidator(model, dataset_path)
        report = validator.run_all()
        print(report.summary())
    """

    def __init__(
        self,
        model: nn.Module,
        data_path: Optional[str] = None,
        sample_input: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.data_path = data_path
        self.sample_input = sample_input
        self.device = device or self._pick_device()
        self.report = ValidationReport()

    def _pick_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── [1] Environment checks ─────────────────────────────────────────
    def check_environment(self):
        """Verify Python, PyTorch, CUDA, and package versions."""
        # Python version
        py_ver = sys.version_info
        if py_ver >= (3, 9):
            self.report.add("Python version", "PASS",
                            f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}")
        else:
            self.report.add("Python version", "WARN",
                            f"{py_ver.major}.{py_ver.minor} — recommend ≥3.9")

        # PyTorch version
        torch_ver = torch.__version__
        major, minor = map(int, torch_ver.split(".")[:2])
        if (major, minor) >= (2, 0):
            self.report.add("PyTorch version", "PASS", torch_ver)
        else:
            self.report.add("PyTorch version", "WARN",
                            f"{torch_ver} — recommend ≥2.0 for compile/FSDP")

        # CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            cuda_ver = torch.version.cuda
            self.report.add("CUDA/GPU", "PASS",
                            f"{gpu_name} ({gpu_mem:.1f} GB), CUDA {cuda_ver}")
        else:
            self.report.add("CUDA/GPU", "WARN",
                            "No GPU detected — training will use CPU (slow)")

        # Key packages
        for pkg in ["numpy", "biotite", "requests", "tensorboard"]:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                self.report.add(f"Package: {pkg}", "PASS", str(ver))
            except ImportError:
                self.report.add(f"Package: {pkg}", "WARN", "not installed")

    # ── [2] Data checks ────────────────────────────────────────────────
    def check_data(self):
        """Verify data files exist and are valid."""
        if self.data_path is None:
            self.report.add("Data", "WARN", "No data_path provided")
            return

        data_path = Path(self.data_path)
        if not data_path.exists():
            self.report.add("Data", "FAIL", f"Path not found: {data_path}")
            return

        # Load and inspect
        try:
            if data_path.suffix == ".npz":
                data = np.load(data_path, allow_pickle=True)
                expr = data["expression"]
            elif data_path.suffix == ".pt":
                data = torch.load(data_path, map_location="cpu")
                expr = data["expression"].numpy() if torch.is_tensor(data["expression"]) else data["expression"]
            else:
                self.report.add("Data", "WARN", f"Unknown format: {data_path.suffix}")
                return

            n_samples, n_genes = expr.shape
            self.report.add("Data shape", "PASS",
                            f"{n_samples} samples × {n_genes} genes")

            # Check for NaN/Inf
            n_nan = np.isnan(expr).sum()
            n_inf = np.isinf(expr).sum()
            if n_nan > 0 or n_inf > 0:
                self.report.add("Data cleanliness", "FAIL",
                                f"Found {n_nan} NaN, {n_inf} Inf values")
            else:
                self.report.add("Data cleanliness", "PASS", "No NaN/Inf")

            # Check variance (all-zero features are useless)
            zero_cols = (np.var(expr, axis=0) == 0).sum()
            if zero_cols > 0:
                self.report.add("Data variance", "WARN",
                                f"{zero_cols} genes have zero variance")
            else:
                self.report.add("Data variance", "PASS", "All genes vary")

            # Minimum sample count
            if n_samples < 32:
                self.report.add("Data volume", "WARN",
                                f"Only {n_samples} samples — consider more data")
            else:
                self.report.add("Data volume", "PASS", f"{n_samples} samples")

        except Exception as e:
            self.report.add("Data", "FAIL", f"Error loading data: {e}")

    # ── [3] Model checks ───────────────────────────────────────────────
    def check_model(self):
        """Verify model forward/backward pass and gradient flow."""
        if self.sample_input is None:
            self.report.add("Model", "WARN", "No sample_input provided")
            return

        self.model = self.model.to(self.device)
        x = self.sample_input.to(self.device)

        # Forward pass
        try:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x)
            if isinstance(out, dict):
                out_shape = {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
                             for k, v in out.items()}
            elif torch.is_tensor(out):
                out_shape = tuple(out.shape)
            else:
                out_shape = str(type(out))
            self.report.add("Model forward pass", "PASS", f"Output: {out_shape}")
        except Exception as e:
            self.report.add("Model forward pass", "FAIL", str(e))
            return

        # Parameter count
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.report.add("Model parameters", "PASS",
                        f"{n_params:,} total, {n_trainable:,} trainable")

        # Backward pass + gradient flow
        try:
            self.model.train()
            self.model.zero_grad()
            out = self.model(x)
            # Build a scalar loss from output
            if isinstance(out, dict):
                loss = sum(v.mean() for v in out.values() if torch.is_tensor(v) and v.is_floating_point())
            else:
                loss = out.mean() if torch.is_tensor(out) else out
            loss.backward()

            # Check gradients exist
            grads_ok = 0
            grads_missing = 0
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is not None and torch.isfinite(p.grad).all():
                        grads_ok += 1
                    else:
                        grads_missing += 1

            if grads_missing == 0:
                self.report.add("Gradient flow", "PASS",
                                f"All {grads_ok} params have finite gradients")
            else:
                self.report.add("Gradient flow", "WARN",
                                f"{grads_missing} params missing/NaN gradients")

        except Exception as e:
            self.report.add("Gradient flow", "FAIL", str(e))

    # ── [4] Memory estimate ────────────────────────────────────────────
    def check_memory(self):
        """Estimate whether model + batch fits in device memory."""
        n_params = sum(p.numel() for p in self.model.parameters())
        # Rough estimate: params (4B) + grads (4B) + optimizer states (8B) = 16B/param
        model_mem_gb = n_params * 16 / 1e9

        if self.device.type == "cuda":
            total_mem_gb = torch.cuda.get_device_properties(self.device).total_mem / 1e9
            # Leave 20% headroom
            available_gb = total_mem_gb * 0.8
            if model_mem_gb < available_gb:
                self.report.add("Memory estimate", "PASS",
                                f"Model ~{model_mem_gb:.2f} GB, available ~{available_gb:.1f} GB")
            else:
                self.report.add("Memory estimate", "FAIL",
                                f"Model ~{model_mem_gb:.2f} GB exceeds available ~{available_gb:.1f} GB. "
                                f"Reduce batch size or use gradient checkpointing/FSDP.")
        else:
            self.report.add("Memory estimate", "WARN",
                            f"CPU training, model ~{model_mem_gb:.2f} GB (check system RAM)")

    # ── [5] End-to-end integration ─────────────────────────────────────
    def check_integration(self):
        """Run one full train step: data → model → loss → optimizer."""
        if self.sample_input is None:
            self.report.add("Integration", "WARN", "No sample_input provided")
            return

        try:
            self.model = self.model.to(self.device)
            self.model.train()
            x = self.sample_input.to(self.device)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

            t0 = time.time()
            optimizer.zero_grad()
            out = self.model(x)
            if isinstance(out, dict):
                loss = sum(v.mean() for v in out.values()
                           if torch.is_tensor(v) and v.is_floating_point())
            else:
                loss = out.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            dt = time.time() - t0

            self.report.add("End-to-end train step", "PASS",
                            f"loss={loss.item():.4f}, step_time={dt*1000:.1f}ms")

        except Exception as e:
            self.report.add("End-to-end train step", "FAIL", str(e))

    # ── [6] Reproducibility ────────────────────────────────────────────
    def check_reproducibility(self):
        """Verify that seeding produces deterministic results."""
        if self.sample_input is None:
            self.report.add("Reproducibility", "WARN", "No sample_input provided")
            return

        try:
            self.model.eval()
            x = self.sample_input.to(self.device)

            torch.manual_seed(42)
            with torch.no_grad():
                out1 = self.model(x)
            torch.manual_seed(42)
            with torch.no_grad():
                out2 = self.model(x)

            # Compare
            if isinstance(out1, dict):
                same = all(
                    torch.allclose(out1[k], out2[k], atol=1e-6)
                    for k in out1 if torch.is_tensor(out1[k]) and torch.is_tensor(out2.get(k))
                )
            else:
                same = torch.allclose(out1, out2, atol=1e-6)

            if same:
                self.report.add("Reproducibility", "PASS",
                                "Same seed → same output")
            else:
                self.report.add("Reproducibility", "WARN",
                                "Outputs differ with same seed (check for dropout/nondeterminism)")

        except Exception as e:
            self.report.add("Reproducibility", "FAIL", str(e))

    # ── Run all checks ─────────────────────────────────────────────────
    def run_all(self) -> ValidationReport:
        """Execute all validation checks and return the report."""
        t0 = time.time()
        logger.info("Running pre-training validation...")

        self.check_environment()
        self.check_data()
        self.check_model()
        self.check_memory()
        self.check_integration()
        self.check_reproducibility()

        self.report.duration_s = time.time() - t0
        return self.report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Run validation on a tiny model to demonstrate the pipeline."""
    print("=" * 70)
    print("  SESI Pre-Training Validation — Smoke Test")
    print("=" * 70)

    # Build a tiny model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 10)
            )
        def forward(self, x):
            return self.net(x)

    model = TinyModel()
    sample_input = torch.randn(4, 64)

    validator = PreTrainValidator(
        model=model,
        data_path=None,       # no data file for smoke test
        sample_input=sample_input,
    )
    report = validator.run_all()
    print(report.summary())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
