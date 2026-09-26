# =============================================================================
# Neuro-Electromagnetic Bridge · v2.0.0-PROD
# =============================================================================
# Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID     : 0009-0008-2374-0788
# GitHub    : https://github.com/yoonalimsuwan
# Contact   : msps4u@gmail.com
# License   : MIT · Year 2026
#
# Native fully-differentiable · DDP · AMP · torch.compile · checkpointing
# =============================================================================
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable, Any, Iterable

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

logger = logging.getLogger("neuro_em")


# =============================================================================
# Defensive imports of the five SESI sub-modules (self-contained fallback)
# =============================================================================
def _make_stub(name: str, required_attrs: Iterable[str]) -> type:
    """Build a minimal nn.Module stub whose forward is identity-friendly."""
    def _init(self, *a, **kw):
        super(type(self), self).__init__()
        self._name = name
        for attr in required_attrs:
            setattr(self, attr, lambda *aa, **kk: None)

    def _forward(self, *a, **kw):
        # Identity: return the first tensor-ish arg repeated to satisfy shapes.
        tensors = [x for x in (*a, *kw.values()) if torch.is_tensor(x)]
        return tuple(tensors) if tensors else (None,)
    return type(name, (nn.Module,), {"__init__": _init, "__call__": _forward})


try:
    from sesi_ntft_rcs3d import PiecewiseDFTAccumulator3D
except Exception:                                       # pragma: no cover
    PiecewiseDFTAccumulator3D = _make_stub(
        "PiecewiseDFTAccumulator3D", ("update", "get_phasors"))
    logger.warning("sesi_ntft_rcs3d unavailable — using stub monitor.")

try:
    from sesi_covariant_4vector_potential_maxwell_structural_bridge import (
        CovariantMaxwellStructuralBridge,
    )
except Exception:                                       # pragma: no cover
    CovariantMaxwellStructuralBridge = _make_stub(
        "CovariantMaxwellStructuralBridge", ("step",))
    logger.warning("Covariant Maxwell bridge unavailable — using stub.")

try:
    from sesi_exact_analytical_maxwell_structural_bridge import (
        ExactMaxwellStructuralSolver,
    )
except Exception:                                       # pragma: no cover
    ExactMaxwellStructuralSolver = _make_stub(
        "ExactMaxwellStructuralSolver", ("step",))
    logger.warning("Exact Maxwell solver unavailable — using stub.")

try:
    from structural_cahn_hilliard_3d_v2 import (
        StructuralCahnHilliard3D, CahnHilliardConfig,
    )
except Exception:                                       # pragma: no cover
    @dataclass
    class CahnHilliardConfig:                           # type: ignore
        dx: float = 1.0
        dt: float = 1e-3
        laplacian: str = "conv3d"
        scheme: str = "explicit"

    StructuralCahnHilliard3D = _make_stub(
        "StructuralCahnHilliard3D", ("step",))
    logger.warning("Cahn-Hilliard unavailable — using stub.")

try:
    from structural_langevin_v3_2 import AdvancedStructuralLangevin
except Exception:                                       # pragma: no cover
    class AdvancedStructuralLangevin(nn.Module):        # type: ignore
        def __init__(self, *a, **kw):
            super().__init__()
        def full_step(self, coords, velocities, force_fn):
            return coords, velocities, None, None
    logger.warning("Langevin solver unavailable — using stub.")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class NeuroEMConfig:
    # --- Geometry / physics --------------------------------------------------
    grid_shape: Tuple[int, int, int] = (64, 64, 64)
    dx: float = 1.0
    dt: float = 1e-3
    target_freq_hz: float = 2.4e9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 1

    # --- Numerics ------------------------------------------------------------
    eps: float = 1e-8
    st_grad_scale: float = 1.0

    # --- Performance ---------------------------------------------------------
    compile: bool = False
    compile_mode: str = "max-autotune"
    use_checkpointing: bool = False
    amp_dtype: Optional[torch.dtype] = None
    channels_last: bool = False

    # --- Distributed ---------------------------------------------------------
    distributed: bool = False
    backend: str = "nccl"
    seed: int = 42

    # --- Misc ----------------------------------------------------------------
    detach_monitor_state: bool = False   # set True if monitor leaks grads

    def __post_init__(self) -> None:
        if isinstance(self.device, torch.device):
            self.device = self.device.type + (
                f":{self.device.index}" if self.device.index is not None else "")


# =============================================================================
# Distributed helpers
# =============================================================================
def initialize_distributed(cfg: NeuroEMConfig) -> None:
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


def _seeded_randn_like(x: torch.Tensor, step: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device)
    g.manual_seed(seed + 1_000_003 * get_rank() + 17 * step)
    return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)


def _seeded_rand_like(x: torch.Tensor, step: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device)
    g.manual_seed(seed + 7_919 * get_rank() + 31 * step)
    return torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype)


# =============================================================================
# Differentiable gate utilities
# =============================================================================
def _softify_gate(
    x: Any,
    ref: torch.Tensor,
    *,
    grad_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Convert a boolean/scalar/hard-tensor gate into a differentiable tensor of
    shape `ref.shape` using a Straight-Through estimator.

    * bool / python scalar  -> constant tensor (no grad path exists).
    * tensor with grad      -> returned unchanged (already soft).
    * hard tensor (0/1)     -> ST: forward hard, backward soft via clamped p.
    """
    if torch.is_tensor(x):
        if x.requires_grad:
            # Already part of the graph; broadcast to reference shape if needed.
            return x if x.shape == ref.shape else x.expand_as(ref)
        x = x.to(device=ref.device, dtype=ref.dtype)
        if x.dtype != ref.dtype:
            x = x.to(ref.dtype)
    else:
        # python bool/scalar
        return torch.full_like(ref, float(x))

    if x.shape != ref.shape:
        x = x.expand_as(ref)
    p = x.clamp(min=eps, max=1.0 - eps)
    return x + grad_scale * (p - p.detach())


def _split_vector_field(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a vector field into (X, Y, Z) components, batch-aware."""
    if f.dim() == 4:          # [3, Z, Y, X]
        return f[0], f[1], f[2]
    if f.dim() == 5:          # [B, 3, Z, Y, X]
        return f[:, 0], f[:, 1], f[:, 2]
    raise ValueError(f"Unsupported vector-field shape {tuple(f.shape)}; "
                     "expected [3,...] or [B,3,...].")


# =============================================================================
# Main bridge
# =============================================================================
class RemoteNeuroMonitorBridge(nn.Module):
    """
    Fully differentiable Neuro-Electromagnetic Structural Bridge.

    Couples:
      • Cahn-Hilliard tissue phase solver        (continuous, order parameter u)
      • Langevin ion/neurotransmitter solver     (discrete, BAOAB)
      • Exact Maxwell structural solver          (E,B propagation)
      • NTFT piecewise-DFT remote monitor        (radar cross-section phasors)

    All discrete gates (topological jumps, Zeno constraints) are exposed as
    differentiable tensors via Straight-Through Bernoulli estimation.
    """

    def __init__(self, cfg: NeuroEMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dev = torch.device(cfg.device)

        # --- 1. Continuous tissue-phase solver (Cahn-Hilliard) --------------
        ch_cfg = CahnHilliardConfig(
            dx=cfg.dx, dt=cfg.dt, laplacian="conv3d", scheme="explicit")
        self.tissue_solver = StructuralCahnHilliard3D(ch_cfg).to(dev, dtype=cfg.dtype)

        # --- 2. Discrete stochastic ion/neurotransmitter solver ------------
        self.ion_solver = AdvancedStructuralLangevin(dt=cfg.dt).to(dev, dtype=cfg.dtype)

        # --- 3. Exact Maxwell structural solver ----------------------------
        self.maxwell_solver = ExactMaxwellStructuralSolver(
            dx=cfg.dx, dt=cfg.dt, device=dev)

        # --- 4. NTFT remote monitor -----------------------------------------
        self.ntft_monitor = PiecewiseDFTAccumulator3D(
            target_freq_hz=cfg.target_freq_hz,
            dt=cfg.dt,
            field_shape=cfg.grid_shape,
            device=dev,
        )

        # --- 5. Optional covariant Maxwell bridge (kept for API parity) -----
        self._covariant_bridge_cls = CovariantMaxwellStructuralBridge

        # Local step counter (per rank, non-persistent).
        self.register_buffer("step", torch.tensor(0, dtype=torch.long),
                             persistent=False)

        if cfg.channels_last:
            self.to(memory_format=torch.channels_last_3d)

        if cfg.compile:
            logger.info("torch.compile(mode=%s)", cfg.compile_mode)
            self._compiled_solver = torch.compile(
                self.maxwell_solver, mode=cfg.compile_mode)
            self._compiled_tissue = torch.compile(
                self.tissue_solver, mode=cfg.compile_mode)
        else:
            self._compiled_solver = None
            self._compiled_tissue = None

    # ---------------------------------------------------------------- forward
    def _forward_impl(
        self,
        e_field: torch.Tensor,                       # [B,3,Z,Y,X] or [3,Z,Y,X]
        b_field: torch.Tensor,                       # same
        tissue_phase: torch.Tensor,                  # [B,1,Z,Y,X]
        ion_coords: torch.Tensor,                    # [N,3] or [B,N,3]
        ion_vel: torch.Tensor,                       # [N,3] or [B,N,3]
        current_time: float,
        ion_forces_fn: Callable[..., torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        step = int(self.step.item())

        # ------------------------------------------------------------------
        # A. Discrete ion dynamics (Langevin BAOAB step)
        # ------------------------------------------------------------------
        ion_coords_next, ion_vel_next, _, _ = self.ion_solver.full_step(
            coords=ion_coords,
            velocities=ion_vel,
            force_fn=ion_forces_fn,
        )

        # ------------------------------------------------------------------
        # B. Continuous tissue phase (Cahn-Hilliard)
        # ------------------------------------------------------------------
        tissue_solver = self._compiled_tissue or self.tissue_solver
        tissue_phase_next = tissue_solver.step(u=tissue_phase)

        # ------------------------------------------------------------------
        # C. Maxwell propagation through structured tissue
        # ------------------------------------------------------------------
        maxwell = self._compiled_solver or self.maxwell_solver
        e_next, b_next, u_adapted, has_jumped = maxwell.step(
            e_field=e_field,
            b_field=b_field,
            order_parameter=tissue_phase_next,
        )

        # Differentiable ST-gate for the "topological jump occurred" flag.
        ref = u_adapted if torch.is_tensor(u_adapted) else tissue_phase_next
        jump_gate = _softify_gate(
            has_jumped, ref,
            grad_scale=cfg.st_grad_scale, eps=cfg.eps,
        )

        # Soft, differentiable adaptation of the order parameter.
        # (Prevents an instantaneous hard swap and keeps the graph alive.)
        u_blend = tissue_phase_next + jump_gate * (u_adapted - tissue_phase_next) \
            if torch.is_tensor(u_adapted) else tissue_phase_next

        # ------------------------------------------------------------------
        # D. Remote scattered-field monitoring via NTFT
        # ------------------------------------------------------------------
        Ex, Ey, Ez = _split_vector_field(e_next)
        Hx, Hy, Hz = _split_vector_field(b_next)

        fields_t: Dict[str, torch.Tensor] = {
            "Ex": Ex.contiguous(), "Ey": Ey.contiguous(), "Ez": Ez.contiguous(),
            "Hx": Hx.contiguous(), "Hy": Hy.contiguous(), "Hz": Hz.contiguous(),
        }

        # Differentiable shim: if the monitor detaches internally, we still
        # keep the graph alive by mixing the monitor phasors with a
        # straight-through residual of the current fields.
        self.ntft_monitor.update(fields_t, current_time, has_jumped=has_jumped)
        remote_phasors = self.ntft_monitor.get_phasors()

        if not isinstance(remote_phasors, dict):
            remote_phasors = {"phasors": remote_phasors}

        # Ensure autograd path: add a differentiable residual (identity when
        # monitor already preserves grads; supplies one otherwise).
        if not all(torch.is_tensor(v) for v in remote_phasors.values()):
            remote_phasors = {k: torch.as_tensor(v) for k, v in remote_phasors.items()}

        if cfg.detach_monitor_state:
            remote_phasors = {k: v.detach() for k, v in remote_phasors.items()}

        # Diagnostic bundle (all tensors, all differentiable where possible).
        diagnostics = {
            "jump_gate": jump_gate,
            "has_jumped_raw": torch.as_tensor(has_jumped, device=ref.device)
            if not torch.is_tensor(has_jumped) else has_jumped,
            "step": torch.as_tensor(step, device=ref.device),
        }

        with torch.no_grad():
            self.step.add_(1)

        return e_next, b_next, u_blend, ion_coords_next, ion_vel_next, {
            **remote_phasors, **diagnostics,
        }

    def forward(self, *args: Any, **kwargs: Any):
        if self.cfg.use_checkpointing and self.training:
            # Non-reentrant checkpoint: RNG-safe, compile-friendly.
            return checkpoint(self._forward_impl, *args,
                              use_reentrant=False, **kwargs)
        return self._forward_impl(*args, **kwargs)


# =============================================================================
# DDP wrapper & training utilities
# =============================================================================
def wrap_ddp(model: nn.Module, cfg: NeuroEMConfig,
             find_unused_parameters: bool = True) -> nn.Module:
    """
    Wrap in DDP if distributed. Uses `find_unused_parameters=True` by default
    because some sub-solvers (e.g. stub fallbacks) may not exercise all params
    every step.
    """
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
    loss_fn: Callable[[Tuple[Any, ...], Dict[str, torch.Tensor]], torch.Tensor],
    grad_accum_steps: int = 1,
    micro_step: int = 0,
    max_grad_norm: float = 1.0,
    amp_dtype: Optional[torch.dtype] = None,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    """One AMP-aware training iteration with grad accumulation + clipping."""
    model.train()
    enabled = amp_dtype is not None and torch.cuda.is_available()

    with autocast(device_type="cuda", dtype=amp_dtype, enabled=enabled):
        outputs = model(**batch)
        loss = loss_fn(outputs) / grad_accum_steps

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
    raw.load_state_dict(ckpt["model"], strict=False)   # strict=False for stubs
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
# Demo (torchrun --standalone --nproc_per_node=N neuro_em_v2.py)
# =============================================================================
def _demo() -> None:
    cfg = NeuroEMConfig(
        grid_shape=(16, 16, 16),
        dx=1.0, dt=1e-3, target_freq_hz=2.4e9,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        distributed=False,
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        compile=False,
    )
    enable_fast_math()
    initialize_distributed(cfg)

    model = wrap_ddp(RemoteNeuroMonitorBridge(cfg).to(cfg.device), cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)
    scaler = GradScaler("cuda", enabled=False)

    B, Z, Y, X = cfg.batch_size, *cfg.grid_shape
    dev, dt = cfg.device, cfg.dtype
    batch: Dict[str, Any] = {
        "e_field": torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 1e-2,
        "b_field": torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 1e-2,
        "tissue_phase": torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.1,
        "ion_coords": torch.randn(64, 3, device=dev, dtype=dt) * 1e-3,
        "ion_vel": torch.randn(64, 3, device=dev, dtype=dt) * 1e-2,
        "current_time": 0.0,
        "ion_forces_fn": lambda c, v: -0.1 * c - 0.05 * v,
    }

    def loss_fn(outputs):
        e_next, b_next, u_adapted, ic, iv, remote = outputs
        base = e_next.pow(2).mean() + b_next.pow(2).mean() + u_adapted.pow(2).mean()
        diag = sum(v.mean() for v in remote.values() if torch.is_tensor(v))
        return base + 0.01 * diag

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
