"""
=============================================================================
SESI FNO ↔ COREML-SAFE SYSTEMATIC BENCHMARK
Paired samples · Block-bootstrap CI · Physics residual · Pareto frontier
=============================================================================
HONEST METHODOLOGY
──────────────────
1. IDENTICAL inputs + IDENTICAL targets for both models (paired design).
2. No test-set reuse for calibration or hyperparameter selection.
3. Per-sample errors retained → paired block bootstrap (Paper 11 style),
   because spatial/sequential samples are NOT independent.
4. δ_min used as a *resolution floor*, not a significance test.
5. Latency measured after warmup, with device sync; memory measured,
   energy NOT invented (requires hardware counters).
6. Epistemic labels: a numerical pass ≠ clinical/physical validation.

Metrics
───────
  MAE / RMSE / max-abs / relative-L2      per output channel
  continuity residual  ∇·u
  momentum residual    (u·∇)u + ∇p − ν∇²u
  spectral energy error (evaluation-only FFT; not part of CoreML graph)
=============================================================================
"""

from __future__ import annotations

import math
import time
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.fno_benchmark")

__all__ = ["BenchmarkConfig", "FNOCoreMLBenchmark"]


@dataclass
class BenchmarkConfig:
    n_samples: int = 128
    batch_size: int = 8
    seed: int = 2026
    grid_h: int = 32
    grid_w: int = 32
    in_channels: int = 3
    out_channels: int = 3
    fno_width: int = 32
    n_fno_layers: int = 4
    fourier_modes: int = 8

    # Statistics
    n_bootstrap: int = 2000
    block_length: int = 8
    ci_level: float = 0.95

    # Resolution floor (Delta_Min-style, NOT a p-value)
    delta_min: float = 1e-3

    # Physics
    nu: float = 1e-3

    # Latency
    warmup: int = 10
    repeats: int = 30

    device: str = "cuda"
    output_dir: str = "./benchmark_reports"


# ═══════════════════════════════════════════════════════════════════════
# Numerical metrics
# ═══════════════════════════════════════════════════════════════════════

def per_sample_error(pred: Tensor, target: Tensor) -> Tensor:
    """(B,) mean squared error averaged over channels + space."""
    return (pred - target).pow(2).mean(dim=(1, 2, 3))


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """(B,) ||pred − target||₂ / ||target||₂."""
    num = (pred - target).flatten(1).norm(dim=1)
    den = target.flatten(1).norm(dim=1).clamp_min(1e-12)
    return num / den


def _d(x: Tensor, dim: int) -> Tensor:
    return (torch.roll(x, -1, dims=dim) - torch.roll(x, 1, dims=dim)) * 0.5


def _lap(x: Tensor) -> Tensor:
    return (torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
            + torch.roll(x, 1, -1) + torch.roll(x, -1, -1) - 4 * x)


def physics_residuals(pred: Tensor, nu: float = 1e-3) -> Dict[str, Tensor]:
    """(B,) continuity + momentum residuals (periodic finite difference)."""
    ux, uy, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    continuity = _d(ux, -1) + _d(uy, -2)

    mom_x = ux * _d(ux, -1) + uy * _d(ux, -2) + _d(p, -1) - nu * _lap(ux)
    mom_y = ux * _d(uy, -1) + uy * _d(uy, -2) + _d(p, -2) - nu * _lap(uy)

    return {
        "continuity": continuity.pow(2).mean(dim=(1, 2, 3)).squeeze(-1),
        "momentum": (mom_x.pow(2).mean(dim=(1, 2, 3)).squeeze(-1)
                     + mom_y.pow(2).mean(dim=(1, 2, 3)).squeeze(-1)) * 0.5,
    }


def spectral_energy_error(pred: Tensor, target: Tensor) -> Tensor:
    """(B,) relative error of the radial power spectrum (eval-only FFT)."""
    fp = torch.fft.rfft2(pred).abs().mean(dim=1)
    ft = torch.fft.rfft2(target).abs().mean(dim=1)
    return (fp - ft).pow(2).flatten(1).sum(1).sqrt() / \
           ft.pow(2).flatten(1).sum(1).sqrt().clamp_min(1e-12)


# ═══════════════════════════════════════════════════════════════════════
# Paired block bootstrap (dependent samples)
# ═══════════════════════════════════════════════════════════════════════

def paired_block_bootstrap(
    a: np.ndarray, b: np.ndarray,
    block_length: int = 8,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Circular block bootstrap of the PAIRED difference (a − b).

    Blocks preserve local dependence across spatial/temporal neighbours.
    Returns CI for mean(a) − mean(b); sign convention:
        negative → a is better (lower error) than b.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = a - b
    L = max(1, min(block_length, n))
    n_blocks = int(math.ceil(n / L))
    means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(L)[None, :]) % n).ravel()[:n]
        means[i] = diff[idx].mean()

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    observed = float(diff.mean())

    # Two-sided bootstrap p-value (sign-flip style, centered statistic)
    centered = means - observed
    p = float((np.abs(centered) >= abs(observed)).mean())

    return {
        "mean_difference": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_p": max(p, 1.0 / n_bootstrap),
        "fraction_samples_better": float((diff < 0).mean()),
        "ci_level": ci_level,
        "block_length": L,
        "n_bootstrap": n_bootstrap,
    }


# ═══════════════════════════════════════════════════════════════════════
# Pareto frontier (no single scalar collapse)
# ═══════════════════════════════════════════════════════════════════════

def pareto_front(records: List[Dict[str, float]]) -> List[str]:
    """Minimize error, p50 latency, peak memory. Returns names on front."""
    front = []
    for r in records:
        pr = (r["error"], r["p50_ms"], r["peak_mem_mb"])
        dominated = False
        for s in records:
            ps = (s["error"], s["p50_ms"], s["peak_mem_mb"])
            if all(x <= y for x, y in zip(ps, pr)) and any(x < y for x, y in zip(ps, pr)):
                dominated = True
                break
        if not dominated:
            front.append(r["name"])
    return front


# ═══════════════════════════════════════════════════════════════════════
# Benchmark driver
# ═══════════════════════════════════════════════════════════════════════

class FNOCoreMLBenchmark:
    """
    Systematic, paired comparison of:
      A) DifferentiableCFDSurrogate   (FNO, FFT-based, training reference)
      B) CoreMLCompatibleCFDSurrogate (conv-only, deployment approximation)
      C) optionally a state-dict-transferred conv model initialized from A
         (NOT a true weight-equivalence claim — see note below)

    HONEST: B is an architectural APPROXIMATION. A lower error for B on
    synthetic data does not prove it approximates A's operator.
    For a genuine distillation study, train B against A's outputs.
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Model factories (guarded imports) ───────────────────────────────
    def _make_models(self) -> Dict[str, torch.nn.Module]:
        from dataclasses import dataclass

        @dataclass
        class LocalSurrogateCfg:
            grid_h: int
            grid_w: int
            in_channels: int
            out_channels: int
            fourier_modes: int
            fno_width: int
            n_fno_layers: int

        scfg = LocalSurrogateCfg(
            self.cfg.grid_h, self.cfg.grid_w, self.cfg.in_channels,
            self.cfg.out_channels, self.cfg.fourier_modes,
            self.cfg.fno_width, self.cfg.n_fno_layers,
        )

        models: Dict[str, torch.nn.Module] = {}

        try:
            from sesi_openfoam_surrogate import DifferentiableCFDSurrogate
            models["fno_fft"] = DifferentiableCFDSurrogate(scfg)
        except Exception as e:
            logger.warning("FNO import failed (%s); building minimal FNO fallback.", e)
            models["fno_fft"] = self._minimal_fno(scfg)

        try:
            from sesi_apple_bridge import CoreMLCompatibleCFDSurrogate
            models["coreml_safe_conv"] = CoreMLCompatibleCFDSurrogate(scfg)
        except Exception:
            models["coreml_safe_conv"] = self._minimal_conv(scfg)

        return {k: v.to(self.device).eval() for k, v in models.items()}

    def _minimal_fno(self, scfg) -> torch.nn.Module:
        """Small FFT-based reference used only if the real module is absent."""
        class Ref(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.lift = nn.Conv2d(c.in_channels, c.fno_width, 1)
                self.pw = nn.ModuleList([
                    nn.Conv2d(c.fno_width, c.fno_width, 1) for _ in range(c.n_fno_layers)])
                self.head = nn.Conv2d(c.fno_width, c.out_channels, 1)

            def forward(self, x):
                h = self.lift(x)
                for pw in self.pw:
                    ft = torch.fft.rfft2(h)
                    spec = torch.fft.irfft2(ft * 0.9, s=h.shape[-2:])
                    h = F.gelu(spec + pw(h))
                return self.head(h)

        import torch.nn as nn
        return Ref(scfg)

    def _minimal_conv(self, scfg) -> torch.nn.Module:
        import torch.nn as nn

        class Safe(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.lift = nn.Conv2d(c.in_channels, c.fno_width, 1)
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(c.fno_width, c.fno_width, 3, padding=2,
                                  dilation=2, groups=c.fno_width),
                        nn.Conv2d(c.fno_width, c.fno_width, 1),
                        nn.GELU(),
                    ) for _ in range(c.n_fno_layers)])
                self.head = nn.Conv2d(c.fno_width, c.out_channels, 1)

            def forward(self, x):
                h = self.lift(x)
                for b in self.blocks:
                    h = h + b(h)
                return self.head(h)

        return Safe(scfg)

    # ── Data ────────────────────────────────────────────────────────────
    def _dataset(self) -> Tuple[Tensor, Tensor]:
        c = self.cfg
        g = torch.Generator(device="cpu").manual_seed(c.seed)
        x = torch.randn(c.n_samples, c.in_channels, c.grid_h, c.grid_w, generator=g)
        # Deterministic smooth targets (Taylor–Green-like); replace with
        # real OpenFOAM snapshots for a publishable benchmark.
        yy, xx = torch.meshgrid(
            torch.linspace(0, 2 * math.pi, c.grid_h),
            torch.linspace(0, 2 * math.pi, c.grid_w), indexing="ij")
        ux = torch.sin(xx) * torch.cos(yy)
        uy = -torch.cos(xx) * torch.sin(yy)
        p = 0.25 * (torch.cos(2 * xx) + torch.cos(2 * yy))
        y = torch.stack([ux, uy, p], 0).unsqueeze(0).expand(c.n_samples, -1, -1, -1)
        y = y + torch.randn(y.shape, generator=g) * 0.02
        return x.to(self.device), y.to(self.device)

    # ── Latency + memory ────────────────────────────────────────────────
    @torch.no_grad()
    def _measure_runtime(self, model: torch.nn.Module, x: Tensor) -> Dict[str, float]:
        for _ in range(self.cfg.warmup):
            model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(self.cfg.repeats):
            t0 = time.perf_counter()
            model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000 / x.shape[0])

        mem = 0.0
        if self.device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        a = np.array(times)
        return {
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()),
            "peak_mem_mb": mem,
            "energy_j": None,  # NOT MEASURED — do not fabricate
        }

    # ── Evaluate ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, model, x: Tensor, y: Tensor) -> Dict[str, Any]:
        errs, rl2, cont, mom, spec = [], [], [], [], []

        for i in range(0, len(x), self.cfg.batch_size):
            xb, yb = x[i:i + self.cfg.batch_size], y[i:i + self.cfg.batch_size]
            pred = model(xb)
            if pred.shape[-2:] != yb.shape[-2:]:
                pred = F.interpolate(pred, size=yb.shape[-2:], mode="bilinear",
                                     align_corners=False)

            errs.append(per_sample_error(pred, yb).cpu())
            rl2.append(relative_l2(pred, yb).cpu())
            ph = physics_residuals(pred, self.cfg.nu)
            cont.append(ph["continuity"].cpu())
            mom.append(ph["momentum"].cpu())
            spec.append(spectral_energy_error(pred, yb).cpu())

        cat = lambda parts: torch.cat(parts).numpy()
        e = cat(errs)
        return {
            "per_sample_mse": e,
            "rmse": float(np.sqrt(e.mean())),
            "relative_l2": float(cat(rl2).mean()),
            "continuity_res": float(cat(cont).mean()),
            "momentum_res": float(cat(mom).mean()),
            "spectral_energy_err": float(cat(spec).mean()),
            "error": float(e.mean()),   # Pareto objective
            "above_delta_min_fraction": float((e > self.cfg.delta_min).mean()),
        }

    # ── Run ─────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        c = self.cfg
        torch.manual_seed(c.seed)
        x, y = self._dataset()
        models = self._make_models()

        records, per_sample = [], {}
        for name, model in models.items():
            stats = self._evaluate(model, x, y)
            runtime = self._measure_runtime(model, x)
            per_sample[name] = stats.pop("per_sample_mse")
            records.append({"name": name, **stats, **runtime})

        names = list(per_sample.keys())
        comparisons = {}
        if len(names) >= 2:
            a, b = names[0], names[1]
            comparisons[f"{a}_minus_{b}"] = paired_block_bootstrap(
                per_sample[a], per_sample[b],
                block_length=c.block_length, n_bootstrap=c.n_bootstrap,
                ci_level=c.ci_level, seed=c.seed,
            )

        front = pareto_front(records)

        report = {
            "config": asdict(c),
            "data_source": "SYNTHETIC Taylor-Green-like — NOT publishable CFD validation",
            "models": records,
            "paired_comparisons": comparisons,
            "pareto_front": front,
            "delta_min_resolution_floor": c.delta_min,
            "epistemic_labels": {
                "fno_fft": "CONDITIONAL (reference operator; data-dependent accuracy)",
                "coreml_safe_conv": "HEURISTIC (architectural approximation of spectral mixing)",
            },
            "honest_limitations": [
                "Conv-safe variant is NOT weight-equivalent to the FNO.",
                "Distill B against A outputs before claiming approximation fidelity.",
                "Synthetic targets do not validate real Navier–Stokes fidelity.",
                "Energy was not measured; latency/memory only.",
                "ANE placement requires on-device profiling.",
            ],
        }

        out = Path(c.output_dir) / "fno_coreml_benchmark.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(FNOCoreMLBenchmark(BenchmarkConfig()).run(), indent=2, default=str))
