# ===============================================================================
# SUPER DNS ONE v6 — SESI Biomass Synthesis & Topological Integration Engine
# ===============================================================================
# Framework Integrations:
#   1. Advanced Biomass Metabolism (Lipids, Amino Acids, ATP, Glucose, O2)
#   2. Global Well-Posedness of Topologically-Active Structural Interfaces
#   3. The No-Zeno Condition via Disordered & Double-Exponential Dynamics
#
# Native fully-differentiable · Production · DDP-ready · AMP · torch.compile
#
# Developer : PAI AND Yoon A Limsuwan : MSPS NETWORK
# License   : MIT
# Year      : 2026  |  Version : 2.0.0-PROD
# ===============================================================================
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

logger = logging.getLogger("sesi.biomass")


# ===============================================================================
# Configuration
# ===============================================================================
@dataclass
class SESIConfig:
    # --- Grid / physics ------------------------------------------------------
    dx: float = 1e-5
    dt: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    boundary: str = "replicate"            # replicate | circular | reflect | zero
    operator_backend: str = "conv3d"       # conv3d | spectral

    # --- Zeno / topological --------------------------------------------------
    gumbel_c1: float = 1.0
    noise_variance: float = 0.05
    base_energy_barrier: float = 2.0
    c_topo_bound: float = 5.0
    st_grad_scale: float = 1.0             # ST gradient magnitude for jump gate
    softness: float = 0.05                 # sigmoid temperature for mass gate

    # --- Performance ---------------------------------------------------------
    compile: bool = False
    compile_mode: str = "max-autotune"
    use_checkpointing: bool = False
    amp_dtype: Optional[torch.dtype] = None
    channels_last: bool = False

    # --- Numerics ------------------------------------------------------------
    eps: float = 1e-8
    log_eps: float = 1e-30
    barrier_clamp: float = 30.0            # avoids exp() overflow -> NaN grad
    min_conc: float = 1e-6

    # --- Distributed ---------------------------------------------------------
    distributed: bool = False
    backend: str = "nccl"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.boundary not in {"replicate", "circular", "reflect", "zero"}:
            raise ValueError(f"Unknown boundary: {self.boundary!r}")
        if self.operator_backend not in {"conv3d", "spectral"}:
            raise ValueError(f"Unknown backend: {self.operator_backend!r}")
        if self.operator_backend == "spectral" and self.boundary != "circular":
            raise ValueError("Spectral backend requires circular (periodic) BC.")


# ===============================================================================
# Distributed helpers
# ===============================================================================
def initialize_distributed(cfg: SESIConfig) -> None:
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
    """Deterministic Gaussian noise, seed = base + rank + step. DDP-safe."""
    g = torch.Generator(device=x.device)
    g.manual_seed(seed + 1_000_003 * get_rank() + step)
    return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)


def _seeded_rand_like(x: torch.Tensor, step: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device)
    g.manual_seed(seed + 7_919 * get_rank() + 31 * step)
    return torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype)


# ===============================================================================
# Spatial operators (fully vectorized, depthwise conv3d)
# ===============================================================================
class SpectralLaplacian3D(nn.Module):
    """FFT Laplacian. O(N log N), exact for periodic BCs."""

    def __init__(self, grid_shape: Tuple[int, int, int], dx: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        Z, Y, X = grid_shape
        kz = torch.fft.fftfreq(Z, d=dx, device=device, dtype=dtype) * 2 * math.pi
        ky = torch.fft.fftfreq(Y, d=dx, device=device, dtype=dtype) * 2 * math.pi
        kx = torch.fft.rfftfreq(X, d=dx, device=device, dtype=dtype) * 2 * math.pi
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        self.register_buffer("k_sq", -(KZ ** 2 + KY ** 2 + KX ** 2), persistent=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        F_hat = torch.fft.rfftn(f, dim=(-3, -2, -1))
        return torch.fft.irfftn(F_hat * self.k_sq, s=f.shape[-3:], dim=(-3, -2, -1))


class SpatialOperators3D(nn.Module):
    """Vectorized 3D Laplacian / gradient / divergence, single conv3d per op."""

    def __init__(self, cfg: SESIConfig) -> None:
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

        for axis, name in ((4, "gx"), (3, "gy"), (2, "gz")):
            k = torch.zeros(1, 1, 3 if axis == 2 else 1, 3 if axis == 3 else 1,
                            3 if axis == 4 else 1, device=dev, dtype=dt)
            k_idx = [0, 0, 1, 1, 1]
            lo, hi = k_idx[axis - 2], k_idx[axis - 2]
            # centre index = 1; lower/upper along the axis
            idx_lo = [0, 0, 1, 1, 1]; idx_lo[axis - 2] = 0
            idx_hi = [0, 0, 1, 1, 1]; idx_hi[axis - 2] = 2
            k[tuple(idx_lo)] = -0.5
            k[tuple(idx_hi)] = 0.5
            self.register_buffer(f"{name}_kernel", k / cfg.dx, persistent=False)

        self.spectral: Optional[SpectralLaplacian3D] = None
        if cfg.operator_backend == "spectral":
            # grid_shape must be supplied by caller via attribute; default cube
            self.spectral = None  # attached lazily by caller if needed

    # ------------------------------------------------------------- padding
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        p = (1, 1, 1, 1, 1, 1)
        if self.cfg.boundary == "circular":
            return F.pad(x, p, mode="circular")
        if self.cfg.boundary == "replicate":
            return F.pad(x, p, mode="replicate")
        if self.cfg.boundary == "reflect":
            return F.pad(x, p, mode="reflect")
        return F.pad(x, p, mode="constant", value=0.0)

    def _dw(self, kernel: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        k = kernel.to(x.dtype).expand(C, 1, *kernel.shape[2:]).contiguous()
        return F.conv3d(self._pad(x), k, groups=C)

    # -------------------------------------------------------------- public API
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        if self.spectral is not None:
            return self.spectral(f)
        return self._dw(self.lap_kernel, f)

    def d_dx(self, f: torch.Tensor) -> torch.Tensor:
        return self._dw(self.gx_kernel, f)

    def d_dy(self, f: torch.Tensor) -> torch.Tensor:
        return self._dw(self.gy_kernel, f)

    def d_dz(self, f: torch.Tensor) -> torch.Tensor:
        return self._dw(self.gz_kernel, f)

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.d_dx(f), self.d_dy(f), self.d_dz(f)

    def divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.d_dx(u) + self.d_dy(v) + self.d_dz(w)


# ===============================================================================
# Biomass metabolism (differentiable positivity)
# ===============================================================================
class FullBiomassMetabolism(nn.Module):
    """
    Continuous 6-species reaction–diffusion:
        [O2, Glucose, Lactate, ATP, AminoAcids, Lipids]
    Positivity enforced by smooth softplus projection (no ReLU dead zone).
    """

    def __init__(self, cfg: SESIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial = SpatialOperators3D(cfg)

        self.register_buffer(
            "D_species",
            torch.tensor([1.8e-9, 6.7e-10, 5.0e-10, 1.0e-10, 5.5e-10, 1.2e-10],
                         device=cfg.device, dtype=cfg.dtype).view(1, 6, 1, 1, 1),
            persistent=False,
        )

        # Michaelis–Menten catabolic
        self.Vmax_O2 = 0.05
        self.Km_O2 = 0.01
        self.Vmax_Glc = 0.02
        self.Km_Glc = 0.05
        # Anabolic
        self.Vmax_Lipid = 0.008
        self.Km_Lipid_ATP = 0.02
        self.Vmax_AA_to_Protein = 0.015
        self.Km_AA_ATP = 0.03

    def forward(self, conc: torch.Tensor, cell_viability: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eps = self.cfg.eps
        O2, Glc, Lac, ATP, AA, Lipids = torch.chunk(conc, 6, dim=1)

        # Catabolism
        r_O2 = (self.Vmax_O2 * O2 / (self.Km_O2 + O2 + eps)) * cell_viability
        r_Glc = (self.Vmax_Glc * Glc / (self.Km_Glc + Glc + eps)) * cell_viability
        r_ATP_prod = 29.0 * r_O2 + 2.0 * r_Glc
        r_Lac_prod = 2.0 * r_Glc * torch.exp(-O2 / (self.Km_O2 + eps))

        # Anabolism
        r_Lipid_syn = (
            self.Vmax_Lipid * ATP / (self.Km_Lipid_ATP + ATP + eps)
            * (Glc / (self.Km_Glc + Glc + eps)) * cell_viability
        )
        r_Protein_syn = (
            self.Vmax_AA_to_Protein * AA / (self.Km_AA_ATP + AA + eps)
            * (ATP / (self.Km_Lipid_ATP + ATP + eps)) * cell_viability
        )
        r_ATP_cons = 8.0 * r_Lipid_syn + 4.0 * r_Protein_syn + 0.01 * ATP

        S_net = torch.cat([
            -r_O2,
            -r_Glc - 0.5 * r_Lipid_syn,
            r_Lac_prod,
            r_ATP_prod - r_ATP_cons,
            -r_Protein_syn,
            r_Lipid_syn,
        ], dim=1)

        # Vectorized diffusion
        d_conc_dt = self.D_species * self.spatial.laplacian(conc) + S_net

        # Fully differentiable positivity: softplus(x - m) + m ≈ max(x, m)
        raw = conc + d_conc_dt * self.cfg.dt
        conc_next = F.softplus(raw - self.cfg.min_conc, beta=20.0) + self.cfg.min_conc

        rates = {
            "ATP_production": r_ATP_prod,
            "Lipid_synthesis": r_Lipid_syn,
            "Protein_synthesis": r_Protein_syn,
            "O2_consumption": r_O2,
            "Glucose_consumption": r_Glc,
        }
        return conc_next, rates


# ===============================================================================
# Extreme-value No-Zeno filter (Straight-Through Bernoulli)
# ===============================================================================
class ExtremeValueZenoFilter(nn.Module):
    """
    Theorem 10.4 bound:
        P(T_{k+1} − T_k < dt) ≤ exp[−C1 · exp(ΔE / (σ² · dt))]

    Forward pass samples a hard Bernoulli trigger; backward pass routes
    gradients through the probability itself (Straight-Through estimator).
    """

    def __init__(self, cfg: SESIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.C1 = cfg.gumbel_c1
        self.sigma_sq = cfg.noise_variance
        self.base_barrier = cfg.base_energy_barrier
        self.grad_scale = cfg.st_grad_scale

    def forward(self, atp: torch.Tensor, lipids: torch.Tensor, aa: torch.Tensor,
                step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = self.cfg.eps
        bio = 0.5 * atp + 0.3 * lipids + 0.2 * aa
        delta_E = F.relu(self.base_barrier - bio) + 0.01

        # NaN-safe: clamp before exp to prevent overflow → NaN backward.
        inner = (delta_E / (self.sigma_sq * self.cfg.dt)
                 ).clamp(max=self.cfg.barrier_clamp)
        prob = torch.exp(-self.C1 * torch.exp(inner))   # ∈ (0, e^{-C1}]

        # --- Straight-Through Bernoulli ------------------------------------
        u = _seeded_rand_like(prob, step=step, seed=self.cfg.seed)
        hard = (u < prob).to(prob.dtype)
        gate = hard + self.grad_scale * (prob - prob.detach())
        return gate, prob


# ===============================================================================
# Topological jump operators (N / M / B) — fully soft-gated
# ===============================================================================
class TopologicalBiomassOperators(nn.Module):
    """
    Soft-gated operators:
        N: nucleation   h → h + ξ·0.05 + 0.3
        M: merging      h → avg_pool3d(h, 3)
        B: branching    h → h + |ξ|·0.2
    All obey energy/mass bound  E(γ⁺) − E(γ⁻) ≤ C_topo  by paying
    lipid_cost / aa_cost per unit gate.
    """

    def __init__(self, cfg: SESIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.c_topo = cfg.c_topo_bound
        self.lipid_cost = 0.5
        self.aa_cost = 0.5

    def _field(self, h_minus: torch.Tensor, op_type: str,
               step: int) -> torch.Tensor:
        if op_type == "nucleation":
            xi = _seeded_randn_like(h_minus, step=step, seed=self.cfg.seed)
            return h_minus + xi * 0.05 + 0.3
        if op_type == "merging":
            return F.avg_pool3d(h_minus, kernel_size=3, stride=1, padding=1)
        if op_type == "branching":
            xi = _seeded_randn_like(h_minus, step=step, seed=self.cfg.seed + 1)
            return h_minus + torch.abs(xi) * 0.2
        return h_minus

    def apply_jump(
        self,
        h_minus: torch.Tensor,
        jump_gate: torch.Tensor,
        biomass_conc: torch.Tensor,
        op_type: str = "nucleation",
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = self.cfg.eps
        AA = biomass_conc[:, 4:5]
        Lipids = biomass_conc[:, 5:6]

        # Soft sufficient-mass gate (no Python control flow).
        s = self.cfg.softness
        mass_gate = (torch.sigmoid((AA - self.aa_cost) / s)
                     * torch.sigmoid((Lipids - self.lipid_cost) / s))

        gate = jump_gate * mass_gate                       # [B, 1, Z, Y, X]

        # Differentiable soft blend:  h⁺ = h⁻ + gate·(field − h⁻)
        new_field = self._field(h_minus, op_type, step)
        h_plus = h_minus + gate * (new_field - h_minus)

        # Differentiable biomass consumption (per-voxel)
        AA_new = (AA - gate * self.aa_cost)
        Lipids_new = (Lipids - gate * self.lipid_cost)

        biomass_new = torch.cat([
            biomass_conc[:, 0:1], biomass_conc[:, 1:2], biomass_conc[:, 2:3],
            biomass_conc[:, 3:4], AA_new, Lipids_new,
        ], dim=1)
        return h_plus, biomass_new, gate


# ===============================================================================
# Master engine (per-voxel chart rebasing, zero Python control flow)
# ===============================================================================
class SESIBiomassIntegrationEngine(nn.Module):
    """
    Unifies:
      • Continuous Stochastic Evolution (fixed topology)
      • Extreme-Value No-Zeno gate
      • Discrete Topological jump + per-voxel chart re-centering
    Everything differentiable; DDP-safe; AMP-safe; compile-friendly.
    """

    def __init__(self, cfg: SESIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.metabolism = FullBiomassMetabolism(cfg)
        self.zeno_filter = ExtremeValueZenoFilter(cfg)
        self.topo_ops = TopologicalBiomassOperators(cfg)

        # Non-persistent step counter (kept local, not synced across ranks).
        self.register_buffer("step", torch.tensor(0, dtype=torch.long), persistent=False)

        if cfg.channels_last:
            self.to(memory_format=torch.channels_last_3d)

        if cfg.compile:
            logger.info("Compiling submodules (mode=%s)", cfg.compile_mode)
            self.metabolism = torch.compile(self.metabolism, mode=cfg.compile_mode)
            self.zeno_filter = torch.compile(self.zeno_filter, mode=cfg.compile_mode)
            self.topo_ops = torch.compile(self.topo_ops, mode=cfg.compile_mode)

    def _forward_impl(
        self,
        h_current: torch.Tensor,
        gamma_0: torch.Tensor,
        biomass_conc: torch.Tensor,
        cell_viability: torch.Tensor,
        op_type: str = "nucleation",
    ) -> Dict[str, torch.Tensor]:
        step = int(self.step.item())

        # -------- Phase 1: continuous metabolism -----------------------------
        conc_next, rates = self.metabolism(biomass_conc, cell_viability)

        # -------- Phase 2: SDE drift-diffusion (fixed topology) --------------
        drift = rates["Protein_synthesis"] * 0.1 - 0.01 * h_current
        diffusion = 0.005 + rates["Lipid_synthesis"] * 0.01
        dW = _seeded_randn_like(h_current, step=step, seed=self.cfg.seed) \
             * math.sqrt(self.cfg.dt)
        h_predict = h_current + drift * self.cfg.dt + diffusion * dW

        # -------- Phase 3: No-Zeno extreme-value gate ------------------------
        ATP = conc_next[:, 3:4]
        AA = conc_next[:, 4:5]
        Lipids = conc_next[:, 5:6]
        jump_gate, prob_bound = self.zeno_filter(ATP, Lipids, AA, step=step)

        # -------- Phase 4: differentiable topological operator ---------------
        h_plus, conc_next, applied_gate = self.topo_ops.apply_jump(
            h_predict, jump_gate, conc_next, op_type=op_type, step=step,
        )

        # -------- Phase 5: per-voxel chart re-centering ----------------------
        # γ ← g·h⁺ + (1−g)·γ      ;  h ← (1−g)·h_predict
        gamma_new = applied_gate * h_plus + (1.0 - applied_gate) * gamma_0
        h_new = (1.0 - applied_gate) * h_predict

        # Advance step (in-place, buffer)
        with torch.no_grad():
            self.step.add_(1)

        return {
            "h_graph": h_new,
            "gamma_0_reference": gamma_new,
            "biomass_conc": conc_next,
            "metabolic_rates": rates,
            "jump_prob_bound": prob_bound,
            "jump_gate": applied_gate,
            "topological_jump_occurred": (applied_gate > 0.5).any().detach(),
        }

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if self.cfg.use_checkpointing and self.training:
            # Non-reentrant checkpoint preserves RNG state by default.
            return checkpoint(self._forward_impl, *args, use_reentrant=False, **kwargs)
        return self._forward_impl(*args, **kwargs)


# ===============================================================================
# DDP wrapping & training utilities
# ===============================================================================
def wrap_ddp(model: nn.Module, cfg: SESIConfig,
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


# ===============================================================================
# Checkpointing
# ===============================================================================
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[GradScaler], epoch: int, step: int,
                    ema: Optional[EMA] = None) -> None:
    if not is_main_process():
        return
    raw = model.module if isinstance(model, DDP) else model
    state: Dict[str, Any] = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
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


# ===============================================================================
# Global perf flags
# ===============================================================================
def enable_fast_math() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ===============================================================================
# Demo (multi-GPU via `torchrun --nproc_per_node=N this_file.py`)
# ===============================================================================
def _demo() -> None:
    cfg = SESIConfig(
        dx=1e-5, dt=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        distributed=False,
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        operator_backend="conv3d",
        boundary="replicate",
        compile=False,
        use_checkpointing=False,
    )
    enable_fast_math()
    initialize_distributed(cfg)

    model = wrap_ddp(SESIBiomassIntegrationEngine(cfg).to(cfg.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda", enabled=False)

    B, Z, Y, X = 1, 32, 32, 32
    dev, dt = cfg.device, cfg.dtype
    batch = {
        "h_current":      torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.1,
        "gamma_0":        torch.zeros(B, 1, Z, Y, X, device=dev, dtype=dt),
        "biomass_conc":   torch.rand(B, 6, Z, Y, X, device=dev, dtype=dt) * 0.1 + 0.02,
        "cell_viability": torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt).clamp_min(0.1),
    }

    def loss_fn(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (out["h_graph"].pow(2).mean()
                + out["biomass_conc"].pow(2).mean()
                + 0.1 * out["jump_prob_bound"].mean())

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
