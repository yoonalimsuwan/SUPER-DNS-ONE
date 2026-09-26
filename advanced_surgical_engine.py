# =============================================================================
# UNIFIED ADVANCED SURGICAL ENGINE (ANS-OS) · v2.0.0-PROD
# =============================================================================
# Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
#             MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License   : MIT · Year 2026 · ORCID: 0009-0008-2374-0788
# GitHub    : https://github.com/yoonalimsuwan
#
# Native fully differentiable · DDP · AMP · torch.compile · checkpointing
# =============================================================================
from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

logger = logging.getLogger("ansos")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class SurgicalConfig:
    # --- Grid / numerics -----------------------------------------------------
    dx: float = 1e-5
    dt: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    boundary: str = "replicate"            # replicate | circular | reflect | zero
    eps: float = 1e-8

    # --- Tissue physics ------------------------------------------------------
    tissue_density: float = 1050.0          # kg/m^3
    specific_heat: float = 3600.0           # J/(kg·K)
    blood_viscosity: float = 3.5e-3         # Pa·s
    thermal_diffusivity: float = 0.51       # W/(m·K)
    thermal_damage_threshold: float = 315.15  # K (42 °C)
    payload_trigger_temp: float = 314.15    # K (41 °C)

    # --- Nanobot SAR ---------------------------------------------------------
    sar_coeff: float = 2.0e-8
    magnetic_field: float = 1.5             # T
    rf_frequency: float = 1e5               # Hz
    macro_heat_coeff: float = 1e4           # K/s per unit tool intensity
    payload_release_base: float = 1.0e-4
    payload_release_heat: float = 0.05

    # --- SESI / Zeno ---------------------------------------------------------
    c_topo_bound: float = 5.0
    c1: float = 1.0
    sigma_sq: float = 0.05
    barrier_clamp: float = 30.0
    st_grad_scale: float = 1.0              # ST gradient magnitude

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

    def __post_init__(self) -> None:
        if self.boundary not in {"replicate", "circular", "reflect", "zero"}:
            raise ValueError(f"Unknown boundary: {self.boundary!r}")


# =============================================================================
# Distributed helpers
# =============================================================================
def initialize_distributed(cfg: SurgicalConfig) -> None:
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
# Vectorized 3D Laplacian (depthwise conv3d, single cuDNN launch)
# =============================================================================
class SpatialLaplacian3D(nn.Module):
    """∇² on [B, C, Z, Y, X] with one conv3d call — 5–15× faster than torch.roll."""

    def __init__(self, cfg: SurgicalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dev, dt = torch.device(cfg.device), cfg.dtype

        lap = torch.zeros(1, 1, 3, 3, 3, device=dev, dtype=dt)
        lap[0, 0, 1, 1, 0] = 1.0
        lap[0, 0, 1, 1, 2] = 1.0
        lap[0, 0, 1, 0, 1] = 1.0
        lap[0, 0, 1, 2, 1] = 1.0
        lap[0, 0, 0, 1, 1] = 1.0
        lap[0, 0, 2, 1, 1] = 1.0
        lap[0, 0, 1, 1, 1] = -6.0
        self.register_buffer("lap_kernel", lap / (cfg.dx ** 2), persistent=False)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        p = (1, 1, 1, 1, 1, 1)
        b = self.cfg.boundary
        if b == "circular":
            return F.pad(x, p, mode="circular")
        if b == "replicate":
            return F.pad(x, p, mode="replicate")
        if b == "reflect":
            return F.pad(x, p, mode="reflect")
        return F.pad(x, p, mode="constant", value=0.0)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        C = f.shape[1]
        k = self.lap_kernel.to(f.dtype).expand(C, 1, 3, 3, 3).contiguous()
        return F.conv3d(self._pad(f), k, groups=C)


# =============================================================================
# Main engine
# =============================================================================
class AdvancedSurgicalEngine(nn.Module):
    """
    Unified, fully-differentiable surgical engine.

    Phases (all differentiable, no Python control flow on tensors):
        1. Macro surgery  — thermal diffusion, nanobot SAR, macro tool heat
        2. Micro surgery  — nanobot payload release kinetics
        3. Topological    — SESI incision via No-Zeno ST-Bernoulli gate
        4. Biophysics     — hemodynamic viscous decay
    """

    def __init__(self, cfg: SurgicalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.laplacian = SpatialLaplacian3D(cfg)

        # Non-persistent constants (never synced by DDP, portable state_dict).
        for name, val in (
            ("tissue_density", cfg.tissue_density),
            ("specific_heat", cfg.specific_heat),
            ("c_topo_bound", cfg.c_topo_bound),
            ("blood_viscosity", cfg.blood_viscosity),
        ):
            self.register_buffer(name, torch.tensor(float(val)), persistent=False)

        # Global step counter (local per rank, non-persistent).
        self.register_buffer("step", torch.tensor(0, dtype=torch.long),
                             persistent=False)

        if cfg.channels_last:
            self.to(memory_format=torch.channels_last_3d)

        if cfg.compile:
            logger.info("torch.compile(mode=%s) on engine", cfg.compile_mode)
            # Compile instance — safer than decorating forward.
            self._compiled = torch.compile(self, mode=cfg.compile_mode)
        else:
            self._compiled = None

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _vector_magnitude(v: torch.Tensor, eps: float) -> torch.Tensor:
        """fp16-safe |v| for a channel-first vector field [B, C, ...]."""
        return torch.sqrt((v * v).sum(dim=1, keepdim=True) + eps)

    def _resolve_tool_activation(self, tool_activation: Optional[torch.Tensor],
                                 ref: torch.Tensor) -> torch.Tensor:
        if tool_activation is None:
            return torch.zeros_like(ref)
        if tool_activation.dim() == 0:
            return tool_activation.to(ref.dtype).expand_as(ref)
        return tool_activation

    # ----------------------------------------------------------- core routine
    def _forward_impl(
        self,
        tissue_geometry: torch.Tensor,        # [B, 1, Z, Y, X]  h(t)
        surgical_tool_field: torch.Tensor,    # [B, 3, Z, Y, X]  tool trajectory
        nanobot_density: torch.Tensor,        # [B, 1, Z, Y, X]  swarm concentration
        temperature_field: torch.Tensor,      # [B, 1, Z, Y, X]  Kelvin
        biomass_conc: torch.Tensor,           # [B, 6, Z, Y, X]  O2,Glc,Lac,ATP,enc,rel
        fluid_velocity: torch.Tensor,         # [B, 3, Z, Y, X]  m/s
        tool_activation: Optional[torch.Tensor] = None,   # [B,1,Z,Y,X] or scalar in [0,1]
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        eps = cfg.eps
        step = int(self.step.item())
        dt = cfg.dt

        act = self._resolve_tool_activation(tool_activation, tissue_geometry)

        # =====================================================================
        # 1. MACRO SURGERY — heat diffusion + nanobot SAR + macro tool heating
        # =====================================================================
        lap_T = self.laplacian(temperature_field)
        tool_intensity = self._vector_magnitude(surgical_tool_field, eps)  # [B,1,...]

        nanobot_sar = (
            cfg.sar_coeff
            * (cfg.magnetic_field ** 2)
            * (cfg.rf_frequency ** 2)
            * nanobot_density
        )
        macro_heat = tool_intensity * cfg.macro_heat_coeff * act

        heat_source = cfg.thermal_diffusivity * lap_T + nanobot_sar + macro_heat
        dT_dt = heat_source / (self.tissue_density * self.specific_heat)
        updated_temperature = temperature_field + dT_dt * dt

        # Smooth thermal-damage integral (no ReLU dead zone).
        thermal_damage = F.softplus(
            updated_temperature - cfg.thermal_damage_threshold, beta=20.0
        ) * dt

        # =====================================================================
        # 2. MICRO SURGERY — nanobot payload release (heat-triggered)
        # =====================================================================
        encapsulated = biomass_conc[:, 4:5]
        released = biomass_conc[:, 5:6]

        trigger = torch.sigmoid(
            (updated_temperature - cfg.payload_trigger_temp) / 0.5
        )
        release_rate = (
            cfg.payload_release_base + cfg.payload_release_heat * trigger
        ) * encapsulated
        new_released = released + release_rate * dt

        updated_biomass = torch.cat(
            [biomass_conc[:, 0:5], new_released], dim=1
        )

        # =====================================================================
        # 3. TOPOLOGICAL — continuous SDE + No-Zeno ST-Bernoulli jump
        # =====================================================================
        dW = _seeded_randn_like(tissue_geometry, step=step, seed=cfg.seed) \
             * math.sqrt(dt)
        lap_h = self.laplacian(tissue_geometry)
        h_continuous = tissue_geometry + lap_h * dt + dW

        # Extreme-value probability:  p_g = exp(−c1 · exp(ΔE / (σ²·dt)))
        delta_e = torch.abs(h_continuous - tissue_geometry) + eps
        inner = (delta_e / (cfg.sigma_sq * dt)).clamp(max=cfg.barrier_clamp)
        gumbel_prob = torch.exp(-cfg.c1 * torch.exp(inner))

        # Probabilistic OR with the tool gate:
        #   p_combined = 1 − (1 − p_gumbel)(1 − a_tool)
        p_combined = 1.0 - (1.0 - gumbel_prob) * (1.0 - act)
        p_combined = p_combined.clamp(min=eps, max=1.0 - eps)

        # Straight-Through Bernoulli gate — hard forward, soft gradient.
        u = _seeded_rand_like(p_combined, step=step, seed=cfg.seed + 1)
        hard = (u < p_combined).to(p_combined.dtype)
        gate = hard + cfg.st_grad_scale * (p_combined - p_combined.detach())

        # Differentiable topological operator (incision / severing).
        xi = _seeded_randn_like(h_continuous, step=step, seed=cfg.seed + 2)
        target = h_continuous + torch.abs(xi) * 0.2
        h_next = h_continuous + gate * (target - h_continuous)

        # Differentiable per-voxel chart re-centering (ALE reference update).
        gamma_0 = gate * h_next + (1.0 - gate) * tissue_geometry

        # =====================================================================
        # 4. BIOPHYSICS — hemodynamic viscous decay
        # =====================================================================
        velocity_decay = 1.0 - cfg.blood_viscosity * dt
        updated_velocity = fluid_velocity * velocity_decay

        with torch.no_grad():
            self.step.add_(1)

        return {
            "updated_geometry": h_next,
            "reference_chart_gamma_0": gamma_0,
            "updated_temperature": updated_temperature,
            "thermal_damage": thermal_damage,
            "released_payload": new_released,
            "updated_biomass": updated_biomass,
            "updated_velocity": updated_velocity,
            "jump_gate": gate,
            "jump_prob": p_combined,
            "topological_event_occurred": (gate > 0.5).any().detach(),
        }

    # ---------------------------------------------------------------- forward
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if self.cfg.use_checkpointing and self.training:
            # Non-reentrant checkpoint preserves RNG + is compile-friendly.
            return checkpoint(self._forward_impl, *args,
                              use_reentrant=False, **kwargs)
        if self._compiled is not None:
            return self._compiled._forward_impl(*args, **kwargs)
        return self._forward_impl(*args, **kwargs)


# =============================================================================
# DDP wrapper & training utilities
# =============================================================================
def wrap_ddp(model: nn.Module, cfg: SurgicalConfig,
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
        gradient_as_bucket_view=True,   # fewer copies, lower peak memory
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
    batch: Dict[str, torch.Tensor],
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
    raw.load_state_dict(ckpt["model"], strict=True)
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
# Demo (torchrun --standalone --nproc_per_node=N ans_os_v2.py)
# =============================================================================
def _demo() -> None:
    cfg = SurgicalConfig(
        dx=1e-5, dt=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        distributed=False,
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        compile=False,
        boundary="replicate",
    )
    enable_fast_math()
    initialize_distributed(cfg)

    model = wrap_ddp(AdvancedSurgicalEngine(cfg).to(cfg.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda", enabled=False)  # bf16 needs no scaler

    B, Z, Y, X = 1, 32, 32, 32
    dev, dt = cfg.device, cfg.dtype
    batch = {
        "tissue_geometry":    torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.1,
        "surgical_tool_field": torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 0.05,
        "nanobot_density":    torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt),
        "temperature_field":  torch.full((B, 1, Z, Y, X), 310.15, device=dev, dtype=dt),
        "biomass_conc":       torch.rand(B, 6, Z, Y, X, device=dev, dtype=dt) * 0.1 + 0.02,
        "fluid_velocity":     torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 1e-3,
        "tool_activation":    torch.zeros(B, 1, Z, Y, X, device=dev, dtype=dt),
    }

    def loss_fn(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            out["updated_geometry"].pow(2).mean()
            + out["updated_temperature"].pow(2).mean() * 1e-3
            + 0.05 * out["jump_prob"].mean()
            + 0.1 * out["thermal_damage"].mean()
        )

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
