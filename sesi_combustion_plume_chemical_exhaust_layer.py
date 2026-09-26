# =============================================================================
# COMBUSTION PLUME & CHEMICAL EXHAUST LAYER (SESI FRAMEWORK)
# SUPER DNS ONE Cluster / ONE Ecosystem — Production Release
# =============================================================================
# Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID     : 0009-0008-2374-0788
# License   : MIT · Year 2026 · Version 2.0.0-PROD
#
# Native fully-differentiable · DDP · AMP · torch.compile · checkpointing
# =============================================================================
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

logger = logging.getLogger("combustion_plume")

__all__ = ["CombustionPlumeTensorLayer", "CombustionConfig",
           "wrap_ddp", "initialize_distributed", "cleanup_distributed",
           "train_step", "EMA", "save_checkpoint", "load_checkpoint"]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class CombustionConfig:
    # --- Physics -------------------------------------------------------------
    c2: float = 1.0                   # Gumbel sharpness in the double-exp bound
    ste_tau: float = 0.05             # sigmoid temperature for hard classification
    mach_threshold: float = 5.0       # informational (kept for API parity)
    barrier_clamp: float = 20.0       # caps exp(exp(·)) argument — gradient safe

    # --- Performance ---------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    compile: bool = False
    compile_mode: str = "max-autotune"
    use_checkpointing: bool = False
    amp_dtype: Optional[torch.dtype] = None

    # --- Numerics ------------------------------------------------------------
    eps: float = 1e-8
    ste_grad_scale: float = 1.0

    # --- Distributed ---------------------------------------------------------
    distributed: bool = False
    backend: str = "nccl"
    seed: int = 42


# =============================================================================
# Distributed helpers
# =============================================================================
def initialize_distributed(cfg: CombustionConfig) -> None:
    if not cfg.distributed or dist.is_initialized():
        return
    if cfg.device.startswith("cuda"):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend=cfg.backend)
    logger.info("DDP init: rank=%d world=%d backend=%s",
                dist.get_rank(), dist.get_world_size(), cfg.backend)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _seeded_rand_like(x: torch.Tensor, step: int, seed: int) -> torch.Tensor:
    """Deterministic uniform noise per (rank, step); DDP-reproducible."""
    g = torch.Generator(device=x.device)
    g.manual_seed(seed + 7_919 * get_rank() + 31 * step)
    return torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype)


# =============================================================================
# Main layer
# =============================================================================
class CombustionPlumeTensorLayer(nn.Module):
    """
    Structural signature extractor for hypersonic chemical exhaust / combustion
    plumes. Uses O(N) pooling + double-exponential No-Zeno bound; no expensive
    3D convolutions. Fully differentiable end-to-end; batch-aware.

    Input shapes (channel-first):
        velocity_field     : [B, 3, Z, Y, X]
        thermal_field      : [B, 1, Z, Y, X]
        chemical_density   : [B, 1, Z, Y, X]
        ambient_temperature: [B, 1, Z, Y, X] or scalar tensor
        dt                 : python float / 0-dim tensor
    """

    def __init__(self, cfg: Optional[CombustionConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CombustionConfig()
        c = self.cfg

        # Learnable structural weights (dtype-aligned).
        self.plume_core_thermal_weight = nn.Parameter(
            torch.tensor(1.8, dtype=c.dtype))
        self.chemical_shear_weight = nn.Parameter(
            torch.tensor(1.4, dtype=c.dtype))
        self.exhaust_velocity_weight = nn.Parameter(
            torch.tensor(1.2, dtype=c.dtype))

        # Ultra-lightweight classifier head (per-sample [B, 3] -> [B, 1]).
        self.plume_classifier_head = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self._reset_parameters()

        self.register_buffer("step", torch.tensor(0, dtype=torch.long),
                             persistent=False)

        if c.compile:
            logger.info("torch.compile(mode=%s) on classifier head",
                        c.compile_mode)
            self._compiled_head = torch.compile(
                self.plume_classifier_head, mode=c.compile_mode)
        else:
            self._compiled_head = None

    def _reset_parameters(self) -> None:
        for m in self.plume_classifier_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _sum_over_channels(v: torch.Tensor) -> torch.Tensor:
        """|v|² per voxel for a vector field [B,C,Z,Y,X] -> [B,1,Z,Y,X]."""
        return (v * v).sum(dim=1, keepdim=True)

    @staticmethod
    def _broadcast_ambient(ambient: torch.Tensor,
                           ref: torch.Tensor) -> torch.Tensor:
        """Ambient can be scalar, [B,1,1,1,1], or full tensor."""
        if not torch.is_tensor(ambient):
            ambient = torch.as_tensor(ambient, device=ref.device, dtype=ref.dtype)
        return ambient.to(device=ref.device, dtype=ref.dtype)

    # --------------------------------------------------- double-exp Zeno bound
    def compute_plume_anomaly_bound(
        self,
        localized_heat_sq: torch.Tensor,
        chemical_variance: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Double-exponential anomaly bound (No-Zeno equivalent for plumes):
            log p_bound = -C2 · exp(  ΔT² / (χ² · dt) )
            p_bound     = exp(log p_bound)

        Returns (prob_bound, log_prob_bound). `log_prob_bound` is the
        gradient-safe quantity to use in losses; `prob_bound` underflows to 0
        for large arguments (as intended by the theorem).
        """
        c = self.cfg
        dt_t = torch.as_tensor(dt, device=localized_heat_sq.device,
                               dtype=localized_heat_sq.dtype)
        denominator = chemical_variance * dt_t + c.eps

        # Clamp the argument of the inner exp — prevents exp(exp(·)) overflow
        # and therefore NaN gradients, at the cost of a flat (zero) gradient
        # region for extremely strong signals (which is the intended physics).
        exponent = torch.clamp(localized_heat_sq / denominator,
                               min=0.0, max=c.barrier_clamp)
        log_prob_bound = -c.c2 * torch.exp(exponent)
        prob_bound = torch.exp(log_prob_bound)
        return prob_bound, log_prob_bound

    # ------------------------------------------------------------- core compute
    def _forward_impl(
        self,
        velocity_field: torch.Tensor,
        thermal_field: torch.Tensor,
        chemical_density: torch.Tensor,
        ambient_temperature: torch.Tensor,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        c = self.cfg
        step = int(self.step.item())

        # -------- 1. Smooth thermal excess (softplus — no ReLU dead zone) ----
        ambient = self._broadcast_ambient(ambient_temperature, thermal_field)
        delta_thermal = F.softplus(thermal_field - ambient, beta=20.0)
        localized_heat_sq = delta_thermal ** 2

        # -------- 2. Chemical density variance proxy -------------------------
        chemical_variance = chemical_density ** 2

        # -------- 3. Exhaust kinetic energy ----------------------------------
        v_sq = self._sum_over_channels(velocity_field)          # [B,1,Z,Y,X]
        exhaust_kinetic_energy = 0.5 * chemical_density * v_sq  # [B,1,Z,Y,X]

        # -------- 4. Zeno bound ----------------------------------------------
        prob_bound, log_prob_bound = self.compute_plume_anomaly_bound(
            localized_heat_sq, chemical_variance, dt)

        # -------- 5. Per-sample pooling (KEEP batch dim) --------------------
        reduce_dims = (-3, -2, -1)                              # Z,Y,X
        pooled_thermal_core = (
            (localized_heat_sq * prob_bound).mean(dim=reduce_dims)
            * self.plume_core_thermal_weight
        ).squeeze(-1)                                           # [B]
        pooled_chem_shear = (
            (chemical_variance * prob_bound).mean(dim=reduce_dims)
            * self.chemical_shear_weight
        ).squeeze(-1)                                           # [B]
        pooled_exhaust_ke = (
            (exhaust_kinetic_energy * prob_bound).mean(dim=reduce_dims)
            * self.exhaust_velocity_weight
        ).squeeze(-1)                                           # [B]

        # -------- 6. Feature stack for Tensor Network queue ------------------
        plume_features = torch.stack(
            [pooled_thermal_core, pooled_chem_shear, pooled_exhaust_ke],
            dim=-1)                                             # [B, 3]

        # -------- 7. Differentiable classification head ----------------------
        head = self._compiled_head or self.plume_classifier_head
        logit = head(plume_features).squeeze(-1)                # [B]
        soft = torch.sigmoid(logit / max(c.ste_tau, c.eps))     # [B]

        # -------- 8. Straight-Through Bernoulli ------------------------------
        u = _seeded_rand_like(soft, step=step, seed=c.seed)
        hard = (u < soft).to(soft.dtype)
        # forward: hard ; backward: grad flows through soft.
        final_prediction = hard.detach() - soft.detach() + c.ste_grad_scale * soft

        with torch.no_grad():
            self.step.add_(1)

        return {
            "is_combustion_target":   final_prediction,     # [B]
            "combustion_confidence":  soft,                 # [B]
            "plume_features_tensor":  plume_features,       # [B, 3]
            "probability_bound":      prob_bound,           # [B,1,Z,Y,X]
            "log_probability_bound":  log_prob_bound,       # [B,1,Z,Y,X]  (grad-safe)
        }

    # ------------------------------------------------------------- public API
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if self.cfg.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, *args,
                              use_reentrant=False, **kwargs)
        return self._forward_impl(*args, **kwargs)

    # ---- keep old public method name for API compatibility ----------------
    def compute_plume_anomaly_bound_legacy(
        self,
        localized_heat_sq: torch.Tensor,
        chemical_variance: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        return self.compute_plume_anomaly_bound(
            localized_heat_sq, chemical_variance, dt)[0]


# =============================================================================
# DDP wrapper & training utilities
# =============================================================================
def wrap_ddp(model: nn.Module, cfg: CombustionConfig,
             find_unused_parameters: bool = False) -> nn.Module:
    if not cfg.distributed:
        return model
    if not dist.is_initialized():
        raise RuntimeError("Call initialize_distributed(cfg) first.")
    local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
    if torch.cuda.is_available():
        model = model.to(torch.device(f"cuda:{local_rank}"))
    return DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
    )


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    batch: Dict[str, Any],
    loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    grad_accum_steps: int = 1,
    micro_step: int = 0,
    max_grad_norm: float = 1.0,
    amp_dtype: Optional[torch.dtype] = None,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    model.train()
    enabled = amp_dtype is not None and torch.cuda.is_available()

    with autocast(device_type="cuda", dtype=amp_dtype, enabled=enabled):
        out = model(**batch)
        loss = loss_fn(out) / grad_accum_steps

    if scaler is not None and enabled:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = {"loss": float(loss.detach()) * grad_accum_steps}

    if (micro_step + 1) % grad_accum_steps == 0:
        if scaler is not None and enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    return metrics


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[GradScaler], epoch: int, step: int,
                    ema: Optional[EMA] = None) -> None:
    if not is_main_process():
        return
    raw = model.module if isinstance(model, DDP) else model
    state: Dict[str, Any] = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch, "step": step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        state["ema"] = ema.shadow
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[GradScaler] = None,
                    ema: Optional[EMA] = None) -> Dict[str, int]:
    raw = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        device = next(raw.parameters()).device
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
    return {"epoch": ckpt.get("epoch", 0), "step": ckpt.get("step", 0)}


def enable_fast_math() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Demo
# =============================================================================
def _demo() -> None:
    cfg = CombustionConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        distributed=False,
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        compile=False,
    )
    enable_fast_math()
    initialize_distributed(cfg)

    model = wrap_ddp(CombustionPlumeTensorLayer(cfg).to(cfg.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda", enabled=False)

    B, Z, Y, X = 4, 16, 16, 16
    dev, dt = cfg.device, cfg.dtype
    batch = {
        "velocity_field":      torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 100.0,
        "thermal_field":       torch.full((B, 1, Z, Y, X), 1500.0, device=dev, dtype=dt)
                               + torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 50.0,
        "chemical_density":    torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt),
        "ambient_temperature": torch.full((B, 1, Z, Y, X), 300.0, device=dev, dtype=dt),
        "dt":                  1e-3,
    }

    def loss_fn(out):
        return (out["is_combustion_target"].pow(2).mean()
                + out["plume_features_tensor"].pow(2).mean()
                + 0.01 * out["log_probability_bound"].mean())

    for step in range(3):
        m = train_step(model, opt, scaler, batch, loss_fn,
                       amp_dtype=cfg.amp_dtype, micro_step=step)
        if is_main_process():
            print(f"[step {step}] loss={m['loss']:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _demo()
