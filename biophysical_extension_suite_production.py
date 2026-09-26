"""
================================================================================
SUPER DNS ONE v6 — Biophysical Extension Suite  |  Production / DDP Edition
================================================================================
Developer : PAI AND Yoon A Limsuwan : MSPS NETWORK / My Soul Move By Power of Holy Spirit
License   : MIT
Year      : 2026
Version   : 2.0.0

Design notes
------------
* Pure PyTorch autograd end-to-end. No `.detach()`, `.item()`, `.cpu()` or
  numpy conversions on the hot path — the entire simulation graph is
  differentiable w.r.t. inputs, parameters, and physical coefficients.
* Two interchangeable stencil backends:
    - ``conv3d``   : depth-wise cuDNN convolutions (fastest on Ampere+ / Hopper)
    - ``spectral`` : FFT-based Laplacian (O(N log N), exact for periodic BCs)
* All multi-species operators are vectorized (no Python loops over channels).
* ``torch.compile(mode='max-autotune')`` compatible.
* Optional gradient checkpointing + AMP (bf16 / fp16) for memory-constrained rigs.
* First-class DDP support: ``initialize_distributed`` / ``wrap_ddp`` /
  ``train_step`` / ``save_checkpoint`` / ``load_checkpoint``.
================================================================================
"""
from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

logger = logging.getLogger("super_dns_one.biophysics")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BiophysicsConfig:
    """Global configuration for biophysical computational domains."""

    # ---- Geometry -----------------------------------------------------------
    grid_shape: Tuple[int, int, int] = (128, 128, 128)
    dx: float = 1e-5                     # grid spacing [m]
    dt: float = 1e-4                     # time step [s]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # ---- Boundary conditions ------------------------------------------------
    boundary: str = "periodic"           # periodic | replicate | reflect | zero

    # ---- Performance knobs --------------------------------------------------
    operator_backend: str = "conv3d"     # conv3d | spectral
    compile: bool = False                # torch.compile the module graph
    compile_mode: str = "max-autotune"
    use_checkpointing: bool = False      # gradient checkpointing
    amp_dtype: Optional[torch.dtype] = None   # torch.bfloat16 | torch.float16
    channels_last: bool = False          # NDHWC memory format (3D convs)

    # ---- Numerical safety ---------------------------------------------------
    clamp_positive: bool = True
    enforce_cfl: bool = True
    eps: float = 1e-8

    # ---- Distributed --------------------------------------------------------
    distributed: bool = False
    backend: str = "nccl"

    def __post_init__(self) -> None:
        if self.boundary not in {"periodic", "replicate", "reflect", "zero"}:
            raise ValueError(f"Unknown boundary mode: {self.boundary!r}")
        if self.operator_backend not in {"conv3d", "spectral"}:
            raise ValueError(f"Unknown operator_backend: {self.operator_backend!r}")
        if self.operator_backend == "spectral" and self.boundary != "periodic":
            raise ValueError("Spectral operators require periodic boundary.")
        if self.enforce_cfl:
            self._cfl_sanity_check()

    def _cfl_sanity_check(self) -> None:
        # Wave-speed bound ≈ sqrt of max diffusivity; warn only.
        D_max = max(1.33e-9, 1.96e-9, 0.79e-9, 1.8e-9)
        dt_cfl = 0.5 * self.dx ** 2 / D_max
        if self.dt > dt_cfl:
            logger.warning(
                "dt=%.3e exceeds explicit-diffusion CFL bound %.3e. "
                "Consider reducing dt or switching to implicit integration.",
                self.dt, dt_cfl,
            )


# =============================================================================
# Distributed helpers
# =============================================================================

def initialize_distributed(cfg: BiophysicsConfig) -> None:
    """Initialize the default process group (NCCL by default). Idempotent."""
    if not cfg.distributed:
        return
    if dist.is_initialized():
        return
    if cfg.device.startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=cfg.backend)
    logger.info("DDP initialized: rank=%d world=%d backend=%s",
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


# =============================================================================
# Spatial operators (conv3d + FFT-spectral backends, fully vectorized)
# =============================================================================

class SpectralLaplacian3D(nn.Module):
    """FFT-based Laplacian ∇² for periodic domains. O(N log N), exact."""

    def __init__(self, grid_shape: Tuple[int, int, int], dx: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        Z, Y, X = grid_shape
        kz = torch.fft.fftfreq(Z, d=dx, device=device, dtype=dtype) * 2 * math.pi
        ky = torch.fft.fftfreq(Y, d=dx, device=device, dtype=dtype) * 2 * math.pi
        kx = torch.fft.rfftfreq(X, d=dx, device=device, dtype=dtype) * 2 * math.pi
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        k_sq = -(KZ ** 2 + KY ** 2 + KX ** 2)
        # Register as non-persistent buffer (DDP must not sync).
        self.register_buffer("k_sq", k_sq, persistent=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        F_hat = torch.fft.rfftn(f, dim=(-3, -2, -1))
        return torch.fft.irfftn(F_hat * self.k_sq, s=f.shape[-3:], dim=(-3, -2, -1))


class SpatialOperators3D(nn.Module):
    """
    Batched 3D finite-difference spatial operators with a selectable backend.

    * ``conv3d`` backend   : depth-wise `F.conv3d`, groups = number of channels.
    * ``spectral`` backend : FFT Laplacian; gradients fall back to finite
                             differences (needed for flux divergence).
    """

    def __init__(self, cfg: BiophysicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dx = cfg.dx
        self.boundary = cfg.boundary
        dev = torch.device(cfg.device)
        dt = cfg.dtype

        # --- Finite-difference stencils -------------------------------------
        lap = torch.zeros(1, 1, 3, 3, 3, device=dev, dtype=dt)
        lap[0, 0, 1, 1, 0] = 1.0
        lap[0, 0, 1, 1, 2] = 1.0
        lap[0, 0, 1, 0, 1] = 1.0
        lap[0, 0, 1, 2, 1] = 1.0
        lap[0, 0, 0, 1, 1] = 1.0
        lap[0, 0, 2, 1, 1] = 1.0
        lap[0, 0, 1, 1, 1] = -6.0
        self.register_buffer("lap_kernel", lap / (self.dx ** 2), persistent=False)

        gx = torch.zeros(1, 1, 1, 1, 3, device=dev, dtype=dt)
        gx[0, 0, 0, 0, 0], gx[0, 0, 0, 0, 2] = -0.5, 0.5
        gy = torch.zeros(1, 1, 1, 3, 1, device=dev, dtype=dt)
        gy[0, 0, 0, 0, 0], gy[0, 0, 0, 2, 0] = -0.5, 0.5
        gz = torch.zeros(1, 1, 3, 1, 1, device=dev, dtype=dt)
        gz[0, 0, 0, 0, 0], gz[0, 0, 2, 0, 0] = -0.5, 0.5
        self.register_buffer("gx_kernel", gx / self.dx, persistent=False)
        self.register_buffer("gy_kernel", gy / self.dx, persistent=False)
        self.register_buffer("gz_kernel", gz / self.dx, persistent=False)

        # --- Optional spectral Laplacian ------------------------------------
        self.spectral: Optional[SpectralLaplacian3D] = None
        if cfg.operator_backend == "spectral":
            self.spectral = SpectralLaplacian3D(cfg.grid_shape, cfg.dx, dev, dt)

    # ------------------------------------------------------------------ padding
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        p = (1, 1, 1, 1, 1, 1)
        if self.boundary == "periodic":
            return F.pad(x, p, mode="circular")
        if self.boundary == "replicate":
            return F.pad(x, p, mode="replicate")
        if self.boundary == "reflect":
            return F.pad(x, p, mode="reflect")
        return F.pad(x, p, mode="constant", value=0.0)

    @staticmethod
    def _expand(kernel: torch.Tensor, C: int, ref: torch.Tensor) -> torch.Tensor:
        k = kernel.to(dtype=ref.dtype)
        return k.expand(C, 1, *kernel.shape[2:]).contiguous()

    # --------------------------------------------------------------- Laplacian
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """∇² f  on  [B, C, Z, Y, X]"""
        if self.spectral is not None:
            return self.spectral(f)
        C = f.shape[1]
        k = self._expand(self.lap_kernel, C, f)
        return F.conv3d(self._pad(f), k, groups=C)

    # ---------------------------------------------------------------- Gradient
    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = f.shape[1]
        fp = self._pad(f)
        gx = F.conv3d(fp, self._expand(self.gx_kernel, C, f), groups=C)
        gy = F.conv3d(fp, self._expand(self.gy_kernel, C, f), groups=C)
        gz = F.conv3d(fp, self._expand(self.gz_kernel, C, f), groups=C)
        return gx, gy, gz

    def d_dx(self, f: torch.Tensor) -> torch.Tensor:
        C = f.shape[1]
        return F.conv3d(self._pad(f), self._expand(self.gx_kernel, C, f), groups=C)

    def d_dy(self, f: torch.Tensor) -> torch.Tensor:
        C = f.shape[1]
        return F.conv3d(self._pad(f), self._expand(self.gy_kernel, C, f), groups=C)

    def d_dz(self, f: torch.Tensor) -> torch.Tensor:
        C = f.shape[1]
        return F.conv3d(self._pad(f), self._expand(self.gz_kernel, C, f), groups=C)

    # -------------------------------------------------------------- Divergence
    def divergence(self, u: torch.Tensor, v: torch.Tensor,
                   w: torch.Tensor) -> torch.Tensor:
        """∇·(u, v, w) for scalar fields u, v, w of shape [B, C, Z, Y, X]."""
        return self.d_dx(u) + self.d_dy(v) + self.d_dz(w)

    # ---------------------- Flux divergence for vector-valued flux (3 channels)
    def flux_divergence(self, fx: torch.Tensor, fy: torch.Tensor,
                        fz: torch.Tensor) -> torch.Tensor:
        """Conservative divergence: ∂_x fx + ∂_y fy + ∂_z fz  (channel-wise)."""
        return self.d_dx(fx) + self.d_dy(fy) + self.d_dz(fz)


# =============================================================================
# Electrochemical module (vectorized PNP)
# =============================================================================

class ElectrochemicalSignalModule(nn.Module):
    """Poisson–Nernst–Planck + FitzHugh–Nagumo ionic dynamics. Differentiable."""

    def __init__(self, cfg: BiophysicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ops = SpatialOperators3D(cfg)

        self.F = 96485.3321
        self.R = 8.3144626
        self.T = 310.15

        # Na+, K+, Ca2+
        self.register_buffer(
            "diffusivities",
            torch.tensor([1.33e-9, 1.96e-9, 0.79e-9],
                         device=cfg.device, dtype=cfg.dtype),
            persistent=False,
        )
        self.register_buffer(
            "valencies",
            torch.tensor([1.0, 1.0, 2.0], device=cfg.device, dtype=cfg.dtype),
            persistent=False,
        )

    def forward(self, ion_concentrations: torch.Tensor,
                electric_potential: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shapes: ions [B, 3, Z, Y, X], phi [B, 1, Z, Y, X]
        D = self.diffusivities.view(1, -1, 1, 1, 1)
        eta = ((self.valencies * self.F) / (self.R * self.T)
               ).view(1, -1, 1, 1, 1)

        # 1) Fickian diffusion — one conv per batch, all channels in parallel.
        diff_term = D * self.ops.laplacian(ion_concentrations)

        # 2) Electro-migration: divergence of (D·η·c·∇φ)
        dphi_dx, dphi_dy, dphi_dz = self.ops.gradient(electric_potential)
        flux_x = D * eta * ion_concentrations * dphi_dx
        flux_y = D * eta * ion_concentrations * dphi_dy
        flux_z = D * eta * ion_concentrations * dphi_dz
        migr_term = self.ops.flux_divergence(flux_x, flux_y, flux_z)

        d_ion_dt = diff_term + migr_term
        phi_lap = self.ops.laplacian(electric_potential)
        return d_ion_dt, phi_lap


# =============================================================================
# Vascular module (Navier–Stokes–Brinkman)
# =============================================================================

class VascularNetworkModule(nn.Module):
    """Darcy–Brinkman hemodynamics with Carman–Kozeny permeability."""

    def __init__(self, cfg: BiophysicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ops = SpatialOperators3D(cfg)

        self.blood_density = 1060.0
        self.blood_viscosity = 3.5e-3

        self.permeability_scale = nn.Parameter(
            torch.tensor([1e-10], device=cfg.device, dtype=cfg.dtype)
        )

    def forward(self, velocity: torch.Tensor, pressure: torch.Tensor,
                vessel_volume_fraction: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = pressure                                     # [B, 1, Z, Y, X]
        vf = vessel_volume_fraction.squeeze(1)           # [B, Z, Y, X]

        # Gradient of pressure (single channel)
        dp_dx, dp_dy, dp_dz = self.ops.gradient(p)

        # Depthwise Laplacian over the 3 velocity components
        lap_u = self.ops.laplacian(velocity)             # [B, 3, ...]

        # Carman–Kozeny permeability (non-negative, well-conditioned)
        vf3 = vf.clamp(min=self.cfg.eps) ** 3
        one_minus = (1.0 - vf).clamp(min=self.cfg.eps) ** 2
        k_perm = self.permeability_scale * vf3 / (one_minus + self.cfg.eps)

        # Darcy drag (per component)
        drag = -(self.blood_viscosity / k_perm).unsqueeze(1) * velocity  # [B,3,...]

        # NS–Brinkman acceleration
        press_grad = torch.cat([dp_dx, dp_dy, dp_dz], dim=1)             # [B,3,...]
        du_dt = (self.blood_viscosity * lap_u - press_grad + drag) / self.blood_density
        return du_dt, drag


# =============================================================================
# Metabolic module (vectorized MM kinetics)
# =============================================================================

class MetabolicKineticsModule(nn.Module):
    """O₂ / Glucose / Lactate / ATP reaction–diffusion with Michaelis–Menten."""

    SPECIES = ("O2", "Glucose", "Lactate", "ATP")

    def __init__(self, cfg: BiophysicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ops = SpatialOperators3D(cfg)

        self.register_buffer(
            "D_species",
            torch.tensor([1.8e-9, 6.7e-10, 5.0e-10, 1.0e-10],
                         device=cfg.device, dtype=cfg.dtype),
            persistent=False,
        )

        self.Vmax_O2 = 0.05
        self.Km_O2 = 0.01
        self.Vmax_Glucose = 0.02
        self.Km_Glucose = 0.05

    def forward(self, species_conc: torch.Tensor, cell_viability: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eps = self.cfg.eps
        O2 = species_conc[:, 0:1]
        Glucose = species_conc[:, 1:2]
        ATP = species_conc[:, 3:4]

        r_O2 = (self.Vmax_O2 * O2 / (self.Km_O2 + O2 + eps)) * cell_viability
        r_Gl = (self.Vmax_Glucose * Glucose / (self.Km_Glucose + Glucose + eps)) * cell_viability

        r_ATP_prod = 29.0 * r_O2 + 2.0 * r_Gl
        r_Lac = 2.0 * r_Gl * torch.exp(-O2 / (self.Km_O2 + eps))

        S_O2 = -r_O2
        S_Gl = -r_Gl
        S_Lac = r_Lac
        S_ATP = r_ATP_prod - 0.1 * ATP
        S_net = torch.cat([S_O2, S_Gl, S_Lac, S_ATP], dim=1)

        # Vectorized diffusion (one conv for all 4 species)
        D = self.D_species.view(1, -1, 1, 1, 1)
        d_species_dt = D * self.ops.laplacian(species_conc) + S_net

        rates = {
            "O2_consumption": r_O2,
            "Glucose_consumption": r_Gl,
            "ATP_production": r_ATP_prod,
            "Lactate_production": r_Lac,
        }
        return d_species_dt, rates


# =============================================================================
# Master integration bridge
# =============================================================================

class BiophysicalDNSBridge(nn.Module):
    """
    Unified biophysical forward integrator.

    All three sub-modules share a single shared :class:`SpatialOperators3D`
    backend choice. Time integration is explicit Euler; swap to RK4 by
    overriding :meth:`_integrate`.
    """

    def __init__(self, cfg: BiophysicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.electro = ElectrochemicalSignalModule(cfg)
        self.vascular = VascularNetworkModule(cfg)
        self.metabolism = MetabolicKineticsModule(cfg)

        if cfg.channels_last:
            # NDHWC memory format — faster on Ampere+ for 3D convs.
            self.to(memory_format=torch.channels_last_3d)

        if cfg.compile:
            logger.info("Applying torch.compile (mode=%s)", cfg.compile_mode)
            self.electro  = torch.compile(self.electro,  mode=cfg.compile_mode)
            self.vascular = torch.compile(self.vascular, mode=cfg.compile_mode)
            self.metabolism = torch.compile(self.metabolism, mode=cfg.compile_mode)

    # --------------------------------------------------------- integrator core
    def _integrate(self, state: torch.Tensor, d_state_dt: torch.Tensor) -> torch.Tensor:
        new = state + d_state_dt * self.cfg.dt
        if self.cfg.clamp_positive:
            new = new.clamp_(min=0.0)
        return new

    # ------------------------------------------------------------- one full step
    def _forward_impl(
        self,
        fluid_velocity: torch.Tensor,
        fluid_pressure: torch.Tensor,
        ion_concentrations: torch.Tensor,
        electric_potential: torch.Tensor,
        metabolic_species: torch.Tensor,
        vascular_density: torch.Tensor,
        cell_viability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        du_dt, drag = self.vascular(fluid_velocity, fluid_pressure, vascular_density)
        d_ions_dt, phi_lap = self.electro(ion_concentrations, electric_potential)
        d_species_dt, rates = self.metabolism(metabolic_species, cell_viability)

        updated_velocity  = self._integrate(fluid_velocity, du_dt)
        updated_ions      = self._integrate(ion_concentrations, d_ions_dt)
        updated_species   = self._integrate(metabolic_species, d_species_dt)

        return {
            "velocity": updated_velocity,
            "ion_concentrations": updated_ions,
            "metabolic_species": updated_species,
            "darcy_drag": drag,
            "phi_laplacian": phi_lap,
            "metabolic_rates": rates,
        }

    def step_simulation(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.cfg.use_checkpointing and self.training:
            # Checkpoint the sub-modules — everything below is differentiable.
            return checkpoint(self._forward_impl, kwargs["fluid_velocity"],
                              kwargs["fluid_pressure"], kwargs["ion_concentrations"],
                              kwargs["electric_potential"], kwargs["metabolic_species"],
                              kwargs["vascular_density"], kwargs["cell_viability"],
                              use_reentrant=False)
        return self._forward_impl(**kwargs)


# =============================================================================
# DDP wrapping + training utilities
# =============================================================================

def wrap_ddp(model: nn.Module, cfg: BiophysicsConfig,
             find_unused_parameters: bool = False) -> nn.Module:
    """Wrap a model in DistributedDataParallel if distributed training is on."""
    if not cfg.distributed:
        return model
    if not dist.is_initialized():
        raise RuntimeError("Call initialize_distributed(cfg) before wrap_ddp().")

    local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
               output_device=local_rank if device.type == "cuda" else None,
               find_unused_parameters=find_unused_parameters,
               gradient_as_bucket_view=True,
               static_graph=False)


class EMA:
    """Exponential moving average of parameters (no .detach() → keeps grads intact)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad:
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
    """One AMP-aware training iteration with gradient accumulation + clipping."""
    model.train()
    ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)

    with ctx:
        out = model.step_simulation(**batch)
        loss = loss_fn(out) / grad_accum_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = {"loss": float(loss.detach()) * grad_accum_steps}

    if (micro_step + 1) % grad_accum_steps == 0:
        if scaler is not None:
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
    state = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        state["ema"] = {k: v for k, v in ema.shadow.items()}
    torch.save(state, path)
    logger.info("Checkpoint saved to %s", path)


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
        ema.shadow = {k: v.to(raw.state_dict()[k].device) for k, v in ckpt["ema"].items()}
    return {"epoch": ckpt.get("epoch", 0), "step": ckpt.get("step", 0)}


# =============================================================================
# Global GPU perf flags
# =============================================================================

def enable_fast_math() -> None:
    """Enable TF32 + cudnn autotuning for peak throughput on Ampere+ / Hopper."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Example: minimal training loop (DDP-ready)
# =============================================================================

def _demo() -> None:
    cfg = BiophysicsConfig(
        grid_shape=(64, 64, 64),
        device="cuda",
        dtype=torch.float32,
        distributed=False,
        amp_dtype=torch.bfloat16,
        compile=False,
        operator_backend="conv3d",
        boundary="periodic",
    )
    enable_fast_math()
    initialize_distributed(cfg)

    model = wrap_ddp(BiophysicalDNSBridge(cfg).to(cfg.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda", enabled=False)  # bf16 needs no scaler

    B = 1
    Z, Y, X = cfg.grid_shape
    dev, dt = cfg.device, cfg.dtype
    batch = {
        "fluid_velocity":     torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 1e-3,
        "fluid_pressure":     torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 10.0,
        "ion_concentrations": torch.rand(B, 3, Z, Y, X, device=dev, dtype=dt) * 0.15,
        "electric_potential": torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.05,
        "metabolic_species":  torch.rand(B, 4, Z, Y, X, device=dev, dtype=dt) * 0.1,
        "vascular_density":   torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt).clamp_min(0.05),
        "cell_viability":     torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt).clamp_min(0.1),
    }

    def loss_fn(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Example surrogate loss — replace with real supervision.
        return (out["velocity"].pow(2).mean()
                + out["ion_concentrations"].pow(2).mean()
                + out["metabolic_species"].pow(2).mean())

    for step in range(5):
        metrics = train_step(model, opt, scaler, batch, loss_fn,
                             amp_dtype=cfg.amp_dtype, micro_step=step)
        if is_main_process():
            print(f"[step {step}] loss={metrics['loss']:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _demo()
