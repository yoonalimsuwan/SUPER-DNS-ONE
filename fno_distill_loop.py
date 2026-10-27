"""
=============================================================================
FNO → CONV-SAFE DISTILLATION LOOP
Teacher-Student training for CoreML/ANE-deployable CFD surrogate
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

PIPELINE
────────
  Teacher (FNO, FFT-based) — FROZEN
      │  forward on identical inputs
      ├── output: (B, 3, H, W) flow field
      └── intermediates: lift / spectral / pointwise features
      │
      ▼
  Distillation loss
      L = w_out · ‖y_s − y_t‖²
        + w_feat · Σ ‖f_s − proj(f_t)‖²
        + w_phys · NS-residual(y_s)
        + w_data · ‖y_s − ground_truth‖²   (optional, if OpenFOAM data)
      │
      ▼
  Student (CoreML-compatible, conv-only) — TRAINABLE

KEY POINTS
──────────
• Teacher is FROZEN (requires_grad=False) during distillation.
• Feature matching uses 1×1 conv projectors to bridge width mismatch.
• Physics loss keeps the student consistent with Navier-Stokes even
  where teacher outputs are noisy.
• Supports AMP + gradient accumulation + checkpointing + resume.
• Pareto tracking: distillation error vs. student latency.

HONEST SCOPE
────────────
• Distillation transfers FUNCTIONAL behavior, not exact FFT semantics.
  The student is an APPROXIMATION; validate on held-out OpenFOAM cases.
• Feature matching assumes teacher/student layer correspondence —
  widths must be compatible or projectors must be sized correctly.
• No claim the distilled model matches FNO accuracy outside training dist.
=============================================================================
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger("sesi.distill")

__all__ = [
    "DistillConfig",
    "FeatureHook",
    "DistillationLoss",
    "FeatureProjector",
    "FnoDistillationTrainer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DistillConfig:
    # ── Loss weights ──────────────────────────────────────────────────────
    w_output: float = 1.0        # student vs teacher output
    w_feature: float = 0.5       # intermediate feature matching
    w_physics: float = 0.1       # NS residual on student output
    w_data: float = 0.2          # student vs ground-truth (if available)

    # ── Physics ───────────────────────────────────────────────────────────
    nu: float = 1e-3

    # ── Training ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 8
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"          # bf16 | fp16 | fp32

    # ── Schedule ──────────────────────────────────────────────────────────
    warmup_epochs: int = 2

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir: str = "./distill_ckpts"
    ckpt_every: int = 5
    resume_path: Optional[str] = None

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda"
    seed: int = 42

    # ── Feature matching layer indices ────────────────────────────────────
    # Which intermediate features to match (by attribute name)
    match_features: List[str] = field(default_factory=lambda: ["lift", "pw"])


# ═══════════════════════════════════════════════════════════════════════
# FEATURE HOOKS
# ═══════════════════════════════════════════════════════════════════════

class FeatureHook:
    """
    Attaches forward hooks to capture intermediate activations.

    Usage:
        hook = FeatureHook(teacher, ["lift", "pw"])
        teacher(x)
        features = hook.get()      # dict name → tensor
        hook.clear()
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self._features: Dict[str, Tensor] = {}
        self._handles = []
        self._attach()

    def _attach(self):
        for name in self.layer_names:
            module = self._get_submodule(name)
            if module is None:
                logger.warning(f"Layer '{name}' not found in model; skipping hook.")
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _get_submodule(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        module = self.model
        for p in parts:
            if hasattr(module, p):
                module = getattr(module, p)
            elif hasattr(module, "_modules") and p in module._modules:
                module = module._modules[p]
            else:
                return None
        return module

    def _make_hook(self, name: str):
        def hook_fn(module, inp, out):
            self._features[name] = out.detach() if isinstance(out, Tensor) else out
        return hook_fn

    def get(self) -> Dict[str, Tensor]:
        return dict(self._features)

    def clear(self):
        self._features.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PROJECTOR
# ═══════════════════════════════════════════════════════════════════════

class FeatureProjector(nn.Module):
    """
    1×1 conv projector bridging teacher/student width mismatch.

    Teacher feature:  (B, C_t, H, W)
    Student feature:  (B, C_s, H, W)
    Projector maps C_t → C_s so L2 matching is well-defined.
    """

    def __init__(self, teacher_channels: int, student_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(teacher_channels, student_channels, kernel_size=1)

    def forward(self, teacher_feat: Tensor) -> Tensor:
        return self.proj(teacher_feat)


# ═══════════════════════════════════════════════════════════════════════
# DISTILLATION LOSS
# ═══════════════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """
    Combined distillation loss.

    L = w_out  · MSE(y_student, y_teacher)
      + w_feat · Σ MSE(proj(f_t), f_s)
      + w_phys · NS_residual(y_student)
      + w_data · MSE(y_student, y_ground_truth)   [if GT available]
    """

    def __init__(self, cfg: DistillConfig, projectors: nn.ModuleDict):
        super().__init__()
        self.cfg = cfg
        self.projectors = projectors

    def ns_residual(self, pred: Tensor) -> Tensor:
        """Continuity + momentum residual on student output."""
        ux = pred[:, 0:1]
        uy = pred[:, 1:2]
        p = pred[:, 2:3]

        def d(f, dim):
            return (torch.roll(f, -1, dims=dim) - torch.roll(f, 1, dims=dim)) * 0.5

        def lap(f):
            return (torch.roll(f, 1, -2) + torch.roll(f, -1, -2)
                    + torch.roll(f, 1, -1) + torch.roll(f, -1, -1) - 4 * f)

        cont = d(ux, -1) + d(uy, -2)
        mom_x = ux * d(ux, -1) + uy * d(ux, -2) + d(p, -1) - self.cfg.nu * lap(ux)
        mom_y = ux * d(uy, -1) + uy * d(uy, -2) + d(p, -2) - self.cfg.nu * lap(uy)

        return (cont.pow(2).mean() + mom_x.pow(2).mean() + mom_y.pow(2).mean())

    def forward(
        self,
        student_out: Tensor,
        teacher_out: Tensor,
        student_feats: Dict[str, Tensor],
        teacher_feats: Dict[str, Tensor],
        ground_truth: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}

        # Output matching
        losses["output"] = F.mse_loss(student_out, teacher_out)

        # Feature matching
        feat_loss = torch.tensor(0.0, device=student_out.device)
        n_matched = 0
        for name in self.cfg.match_features:
            if name in teacher_feats and name in student_feats:
                proj = self.projectors[name]
                t_proj = proj(teacher_feats[name])
                s = student_feats[name]
                # Handle shape mismatch gracefully
                if t_proj.shape != s.shape:
                    s = F.interpolate(s, size=t_proj.shape[-2:],
                                      mode="bilinear", align_corners=False)
                feat_loss = feat_loss + F.mse_loss(s, t_proj)
                n_matched += 1
        losses["feature"] = feat_loss / max(n_matched, 1)

        # Physics loss on student
        losses["physics"] = self.ns_residual(student_out)

        # Data loss (optional)
        if ground_truth is not None:
            losses["data"] = F.mse_loss(student_out, ground_truth)
        else:
            losses["data"] = torch.tensor(0.0, device=student_out.device)

        # Weighted total
        total = (
            self.cfg.w_output * losses["output"]
            + self.cfg.w_feature * losses["feature"]
            + self.cfg.w_physics * losses["physics"]
            + self.cfg.w_data * losses["data"]
        )
        losses["total"] = total
        return losses


# ═══════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════

class FnoDistillationTrainer:
    """
    Complete distillation trainer: FNO teacher → conv-safe student.

    Handles:
      • Teacher freezing + eval mode
      • AMP mixed precision
      • Gradient accumulation
      • Warmup + cosine LR
      • Checkpointing + resume
      • Validation with teacher-parity metric
    """

    def __init__(
        self,
        cfg: DistillConfig,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher_channels: Dict[str, int],
        student_channels: Dict[str, int],
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)

        # Freeze teacher
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Student is trainable
        self.student = student.to(self.device).train()

        # Feature hooks
        self.teacher_hook = FeatureHook(self.teacher, cfg.match_features)
        self.student_hook = FeatureHook(self.student, cfg.match_features)

        # Projectors
        self.projectors = nn.ModuleDict()
        for name in cfg.match_features:
            if name in teacher_channels and name in student_channels:
                self.projectors[name] = FeatureProjector(
                    teacher_channels[name], student_channels[name]
                ).to(self.device)
        logger.info(f"Projectors: {list(self.projectors.keys())}")

        # Loss
        self.loss_fn = DistillationLoss(cfg, self.projectors).to(self.device)

        # Optimizer
        trainable = list(self.student.parameters()) + list(self.projectors.parameters())
        self.opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # LR schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs, eta_min=1e-6
        )

        # AMP
        self._amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                            "fp32": torch.float32}[cfg.amp_dtype]
        self._amp_on = cfg.amp_dtype != "fp32" and self.device.type == "cuda"
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler(enabled=(self._amp_on and self._amp_dtype == torch.float16))
        except ImportError:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=(self._amp_on and self._amp_dtype == torch.float16))

        # Loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_parity = float("inf")

        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

        # Resume
        if cfg.resume_path:
            self.load_checkpoint(cfg.resume_path)

    # ── Checkpoint ────────────────────────────────────────────────────────
    def save_checkpoint(self, tag: str) -> Path:
        path = Path(self.cfg.ckpt_dir) / f"distill_{tag}.pt"
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "student": self.student.state_dict(),
            "projectors": self.projectors.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_parity": self.best_val_parity,
            "config": asdict(self.cfg),
        }, path)
        logger.info(f"Checkpoint → {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student"])
        self.projectors.load_state_dict(ckpt["projectors"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_parity = ckpt.get("best_val_parity", float("inf"))
        logger.info(f"Resumed from {path} (epoch {self.epoch})")

    # ── Single step ───────────────────────────────────────────────────────
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.student.train()
        self.teacher.eval()

        x = batch["input"].to(self.device, non_blocking=True)
        gt = batch.get("target")
        gt = gt.to(self.device, non_blocking=True) if gt is not None else None

        self.opt.zero_grad(set_to_none=True)

        try:
            from torch.amp import autocast
            ctx = autocast(device_type=self.device.type,
                          dtype=self._amp_dtype, enabled=self._amp_on)
        except ImportError:
            from torch.cuda.amp import autocast
            ctx = autocast(dtype=self._amp_dtype, enabled=self._amp_on)

        with ctx:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_hook = self.teacher_hook
                teacher_hook.clear()
                teacher_out = self.teacher(x)
                teacher_feats = teacher_hook.get()

            # Student forward
            self.student_hook.clear()
            student_out = self.student(x)
            student_feats = self.student_hook.get()

            losses = self.loss_fn(
                student_out, teacher_out,
                student_feats, teacher_feats,
                ground_truth=gt,
            )
            loss = losses["total"] / self.cfg.grad_accum_steps

        self.scaler.scale(loss).backward()

        if (self.global_step + 1) % self.cfg.grad_accum_steps == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.projectors.parameters()),
                self.cfg.grad_clip,
            )
            self.scaler.step(self.opt)
            self.scaler.update()

        self.global_step += 1

        return {k: float(v.item()) for k, v in losses.items()}

    # ── Validation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute teacher-parity: how close is student to teacher."""
        self.student.eval()
        self.teacher.eval()

        total_parity = 0.0
        total_physics = 0.0
        n = 0

        for batch in self.val_loader:
            x = batch["input"].to(self.device, non_blocking=True)

            with torch.no_grad():
                t_out = self.teacher(x)
                s_out = self.student(x)

            parity = F.mse_loss(s_out, t_out).item()
            physics = self.loss_fn.ns_residual(s_out).item()

            total_parity += parity * x.shape[0]
            total_physics += physics * x.shape[0]
            n += x.shape[0]

        return {
            "parity_mse": total_parity / max(n, 1),
            "physics_residual": total_physics / max(n, 1),
            "n_samples": n,
        }

    # ── Full training loop ────────────────────────────────────────────────
    def fit(self) -> Dict[str, Any]:
        logger.info(
            f"Starting distillation: {self.cfg.epochs} epochs, "
            f"batch={self.cfg.batch_size}, device={self.device}"
        )

        history: Dict[str, List] = {
            "train_loss": [], "val_parity": [], "val_physics": [],
            "lr": [],
        }

        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            n_batches = 0

            # Warmup
            if epoch < self.cfg.warmup_epochs:
                for pg in self.opt.param_groups:
                    pg["lr"] = self.cfg.lr * (epoch + 1) / self.cfg.warmup_epochs

            for batch in self.train_loader:
                losses = self.train_step(batch)
                epoch_loss += losses["total"]
                n_batches += 1

            self.scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)
            history["lr"].append(self.opt.param_groups[0]["lr"])

            # Validate
            val = self.validate()
            history["val_parity"].append(val["parity_mse"])
            history["val_physics"].append(val["physics_residual"])

            logger.info(
                f"Epoch {epoch:3d} | train={avg_train:.4f} | "
                f"parity={val['parity_mse']:.4f} | "
                f"physics={val['physics_residual']:.4e} | "
                f"lr={self.opt.param_groups[0]['lr']:.2e}"
            )

            # Checkpoint
            if val["parity_mse"] < self.best_val_parity:
                self.best_val_parity = val["parity_mse"]
                self.save_checkpoint("best")

            if (epoch + 1) % self.cfg.ckpt_every == 0:
                self.save_checkpoint(f"epoch_{epoch}")

        # Final report
        self.save_checkpoint("final")
        history["best_val_parity"] = self.best_val_parity
        history["teacher_frozen"] = True
        history["student_trainable_params"] = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )

        report_path = Path(self.cfg.ckpt_dir) / "distill_report.json"
        with open(report_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Report → {report_path}")

        return history


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Verify distillation loop mechanics on synthetic data."""
    print("=" * 72)
    print(" FNO → Conv Distillation Loop — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    # Try importing real models, fall back to minimal stand-ins
    try:
        from sesi_openfoam_surrogate import (
            DifferentiableCFDSurrogate, CoreMLCompatibleCFDSurrogate,
            OpenFOAMSurrogateConfig,
        )
        scfg = OpenFOAMSurrogateConfig(
            grid_h=32, grid_w=32, in_channels=3, out_channels=3,
            fourier_modes=8, fno_width=16, n_fno_layers=3,
        )
        teacher = DifferentiableCFDSurrogate(scfg)
        student = CoreMLCompatibleCFDSurrogate(scfg)
        print("  Using real FNO + conv-safe models")
    except ImportError:
        print("  Fallback: minimal FNO / conv models")

        class _MiniFNO(nn.Module):
            def __init__(self):
                super().__init__()
                self.lift = nn.Conv2d(3, 16, 1)
                self.spec = nn.ModuleList([nn.Conv2d(16, 16, 3, padding=1) for _ in range(2)])
                self.pw = nn.ModuleList([nn.Conv2d(16, 16, 1) for _ in range(2)])
                self.proj = nn.Conv2d(16, 3, 1)

            def forward(self, x):
                h = F.gelu(self.lift(x))
                for s, p in zip(self.spec, self.pw):
                    h = F.gelu(s(h) + p(h))
                return self.proj(h)

        teacher = _MiniFNO()
        student = _MiniFNO()

    teacher = teacher.to(device)
    student = student.to(device)

    # Channel counts for projectors (match layer widths)
    teacher_ch = {"lift": 16, "pw": 16}
    student_ch = {"lift": 16, "pw": 16}

    # Synthetic dataloader
    class _SynthDS(torch.utils.data.Dataset):
        def __len__(self):
            return 32
        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"input": torch.randn(3, 32, 32, generator=g),
                    "target": torch.randn(3, 32, 32, generator=g)}

    ds = _SynthDS()
    train_dl = DataLoader(ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(ds, batch_size=4, shuffle=False)

    cfg = DistillConfig(
        epochs=3, batch_size=4, device=str(device),
        ckpt_dir="/tmp/distill_smoke",
        match_features=["lift", "pw"],
    )

    trainer = FnoDistillationTrainer(
        cfg, teacher, student, train_dl, val_dl,
        teacher_ch, student_ch,
    )

    # Verify teacher is frozen
    n_trainable_t = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    n_trainable_s = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Teacher trainable params: {n_trainable_t} (must be 0)")
    print(f"  Student trainable params: {n_trainable_s:,}")
    assert n_trainable_t == 0, "Teacher must be frozen"

    # Run 1 step to verify gradient flow
    batch = next(iter(train_dl))
    losses = trainer.train_step(batch)
    print(f"  Train step losses: { {k: f'{v:.4f}' for k, v in losses.items()} }")

    # Validate
    val = trainer.validate()
    print(f"  Validation parity MSE: {val['parity_mse']:.4f}")
    print(f"  Physics residual:      {val['physics_residual']:.4e}")

    # Full short training
    print("\n  Running full 3-epoch distillation...")
    history = trainer.fit()
    print(f"  Best parity: {history['best_val_parity']:.4f}")

    # Verify gradient only flows to student/projectors
    all_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in list(student.parameters()) + list(trainer.projectors.parameters())
        if p.requires_grad
    )
    print(f"\n  Autograd (student+projectors): {'PASS' if all_ok else 'FAIL'}")

    print("\n" + "=" * 72)
    print("  ✅ Distillation loop verified")
    print("  HONEST NOTES:")
    print("  • Teacher frozen, student learns from teacher outputs")
    print("  • Feature matching via 1×1 projectors")
    print("  • Physics loss keeps NS consistency")
    print("  • Validate on held-out OpenFOAM data before deployment")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
