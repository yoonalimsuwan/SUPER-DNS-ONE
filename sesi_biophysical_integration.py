# ===============================================================================
# SUPER DNS ONE v6 — SESI Biophysical Integration Suite
# ===============================================================================
# Framework   : Self-Evolving Structural Interfaces (SESI)
# Developer   : PAI AND Yoon A Limsuwan : MSPS NETWORK
# License     : MIT · Year 2026 · Version 2.0.0-PROD
#
# Native fully-differentiable · DDP · AMP · torch.compile · checkpointing
# ===============================================================================
from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

logger = logging.getLogger("sesi.biophys")


# ===============================================================================
# Configuration
# ===============================================================================
@dataclass
class SESIBiophysConfig:
    # --- Physics -------------------------------------------------------------
    dt: float = 1e-4
    dx: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    boundary: str = "replicate"            # replicate | circular | reflect | zero
    operator_backend: str = "conv3d"       # conv3d | spectral

    # --- SESI / Zeno ---------------------------------------------------------
    c_topo_bound: float = 5.0
    gumbel_c1: float = 1.0
    noise_variance: float = 0.05
    barrier_floor: float = 0.01
    barrier_clamp: float = 30.0            # prevents exp(exp(·)) overflow
    st_grad_scale: float = 1.0             # ST gradient magnitude for jump gate
    softness: float = 0.05                 # sigmoid temp for mass/mixing gates

    # --- Operators -----------------------------------------------------------
    bio_op_type: str = "branching"         # branching | merging | nucleation
    soft_operator_mix: bool = False        # if True, mix all three operators
    mix_weights: Tuple[float, float, float] = (0.5, 0.25, 0.25)  # B, M, N

    # --- Performance ---------------------------------------------------------
    compile: bool = False
    compile_mode: str = "max-autotune"
    use_checkpointing: bool = False
    amp_dtype: Optional[torch.dtype] = None
    channels_last: bool = False

    # --- Numerics ------------------------------------------------------------
    eps: float = 1e-8

    # --- Distributed ---------------------------------------------------------
    distributed: bool = False
    backend: str = "nccl"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.boundary not in {"replicate", "circular", "reflect", "zero"}:
            raise ValueError(f"Unknown boundary: {self.boundary!r}")
        if self.operator_backend not in {"conv3d", "spectral"}:
            raise ValueError(f"Unknown backend: {self.operator_backend!r}")
        if self.bio_op_type not in {"branching", "merging", "nucleation"}:
            raise ValueError(f"Unknown bio_op_type: {self.bio_op_type!r}")


# ===============================================================================
# Distributed helpers
# ===============================================================================
def initialize_distributed(cfg: SESIBiophysConfig) -> None:
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


# ===============================================================================
# Spatial operators (compact, conv3d-based)
# ===============================================================================
class SpatialLaplacian3D(nn.Module):
    """Vectorized 3D Laplacian via depth-wise conv3d, single launch per call."""

    def __init__(self, cfg: SESIBiophysConfig) -> None:
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


# ===============================================================================
# Disordered energy landscape (smooth, differentiable)
# ===============================================================================
class DisorderedEnergyLandscape(nn.Module):
    """
    Smooth realization of the quenched disordered potential landscape.
    Combines:
        • a mean-field double-well  V_mf(h) = a·h⁴ − b·h²
        • multi-scale disorder  ∑_i  A_i · cos(k_i · h + φ_i)
    where the disorder amplitudes A_i and phases φ_i are learnable
    (buffers, non-persistent) and the wavenumbers k_i define the scales.
    """

    def __init__(self, cfg: SESIBiophysConfig,
                 scales: Tuple[float, ...] = (3.0, 7.0, 13.0, 29.0)) -> None:
        super().__init__()
        self.cfg = cfg
        dev, dt = torch.device(cfg.device), cfg.dtype

        a = torch.tensor(0.25, device=dev, dtype=dt)         # quartic
        b = torch.tensor(0.50, device=dev, dtype=dt)         # quadratic
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)

        k = torch.tensor(scales, device=dev, dtype=dt)       # (K,)
        phase = torch.linspace(0.0, 2.0 * math.pi, len(scales),
                               device=dev, dtype=dt)         # deterministic init
        amp = torch.full((len(scales),), 1.0 / len(scales),
                         device=dev, dtype=dt)
        self.register_buffer("k", k.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("phase", phase.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("amp", amp.view(1, -1, 1, 1, 1), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : [B, 1, Z, Y, X]  →  ΔE : [B, 1, Z, Y, X]"""
        h_ = h.unsqueeze(1)                                  # [B,1,1,Z,Y,X]
        v_mf = self.a * h_.pow(4) - self.b * h_.pow(2)
        disorder = (self.amp * torch.cos(self.k * h_ + self.phase)).sum(dim=1)
        return v_mf.squeeze(1) + disorder


# ===============================================================================
# Double-exponential No-Zeno filter (ST-Bernoulli)
# ===============================================================================
class DoubleExponentialZenoFilter(nn.Module):
    """
    Theorem 10.4 bound:
        P(T_{k+1} − T_k < dt) ≤ exp[ −C1 · exp(ΔE_min / (σ² · dt)) ]
    Forward: hard Bernoulli sample.
    Backward: Straight-Through gradient routed through the probability.
    """

    def __init__(self, cfg: SESIBiophysConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.C1 = cfg.gumbel_c1
        self.sigma_sq = cfg.noise_variance
        self.floor = cfg.barrier_floor
        self.clamp = cfg.barrier_clamp
        self.grad_scale = cfg.st_grad_scale

    def forward(self, delta_e: torch.Tensor, step: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ΔE_min floor + smooth softplus to avoid ReLU dead zone.
        de = F.softplus(delta_e - self.floor, beta=20.0) + self.floor
        inner = (de / (self.sigma_sq * self.cfg.dt)).clamp(max=self.clamp)
        prob = torch.exp(-self.C1 * torch.exp(inner))       # ∈ (0, e^{-C1}]
        u = _seeded_rand_like(prob, step=step, seed=self.cfg.seed)
        hard = (u < prob).to(prob.dtype)
        gate = hard + self.grad_scale * (prob - prob.detach())
        return gate, prob


# ===============================================================================
# Bio-topological operators (soft-gated, differentiable)
# ===============================================================================
class BioTopologicalOperators(nn.Module):
    """
    Operators N (nucleation), M (fusion), B (branching) with differentiable
    per-voxel gating and an optional soft mixture of all three.
    """

    def __init__(self, cfg: SESIBiophysConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.c_topo = cfg.c_topo_bound
        w = torch.tensor(cfg.mix_weights, dtype=cfg.dtype,
                         device=cfg.device).clamp_min(cfg.eps)
        self.register_buffer("mix_w", w / w.sum(), persistent=False)
        # weights order: (branching, merging, nucleation)

    # ---- single-operator generators (all differentiable) --------------------
    def _branching(self, h: torch.Tensor, step: int) -> torch.Tensor:
        xi = _seeded_randn_like(h, step=step, seed=self.cfg.seed)
        return h + torch.abs(xi) * 0.15

    def _merging(self, h: torch.Tensor, step: int) -> torch.Tensor:
        # 3D avg-pool to honour 5D tensors (v1 used avg_pool2d on 3D data).
        return F.avg_pool3d(h, kernel_size=3, stride=1, padding=1)

    def _nucleation(self, h: torch.Tensor, step: int) -> torch.Tensor:
        xi = _seeded_randn_like(h, step=step, seed=self.cfg.seed + 1)
        return h + xi * 0.05 + 0.3

    # ---- forward ------------------------------------------------------------
    def forward(self, h_minus: torch.Tensor, gate: torch.Tensor, *,
                bio_op_type: Optional[str] = None, step: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        op = bio_op_type or self.cfg.bio_op_type

        if self.cfg.soft_operator_mix:
            hb = self._branching(h_minus, step)
            hm = self._merging(h_minus, step)
            hn = self._nucleation(h_minus, step)
            w = self.mix_w.view(1, 3, 1, 1, 1) if h_minus.dim() == 5 else \
                self.mix_w.view(1, 3, 1, 1)
            target = (w[:, 0:1] * hb + w[:, 1:2] * hm + w[:, 2:3] * hn)
        elif op == "branching":
            target = self._branching(h_minus, step)
        elif op == "merging":
            target = self._merging(h_minus, step)
        else:  # nucleation
            target = self._nucleation(h_minus, step)

        # Differentiable per-voxel blend.
        h_plus = h_minus + gate * (target - h_minus)
        return h_plus, gate


# ===============================================================================
# Master SESI biophysical engine
# ===============================================================================
class SESIBiophysicalEngine(nn.Module):
    """
    Unifies:
      • Continuous Biophysical Solver (fluid, ions, metabolites) — passed in.
      • Local SDE interface evolution  dh = b(h;u)dt + g(h;u)dW
      • Double-exponential No-Zeno gate   (Theorem 10.4)
      • Discrete bio-topological jump (B/M/N) + per-voxel chart re-centering.
    All operations are differentiable; no Python control flow on tensors.
    """

    def __init__(self, cfg: SESIBiophysConfig, biophysics_bridge: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.bio_solver = biophysics_bridge
        self.landscape = DisorderedEnergyLandscape(cfg)
        self.zeno_filter = DoubleExponentialZenoFilter(cfg)
        self.bio_topo_ops = BioTopologicalOperators(cfg)

        self.register_buffer("step", torch.tensor(0, dtype=torch.long),
                             persistent=False)

        if cfg.channels_last:
            self.to(memory_format=torch.channels_last_3d)

        if cfg.compile:
            logger.info("torch.compile (mode=%s)", cfg.compile_mode)
            self.landscape = torch.compile(self.landscape, mode=cfg.compile_mode)
            self.zeno_filter = torch.compile(self.zeno_filter, mode=cfg.compile_mode)
            self.bio_topo_ops = torch.compile(self.bio_topo_ops, mode=cfg.compile_mode)

    # ----------------------------------------------------------- drift/diff
    def compute_interface_drift_diffusion(
        self, h_t: torch.Tensor, fluid_vel: torch.Tensor,
        metabolic_rates: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        atp = metabolic_rates.get("ATP_production", torch.zeros_like(h_t))
        # fp16-safe vector magnitude.
        vel_mag = torch.sqrt((fluid_vel * fluid_vel).sum(dim=1, keepdim=True)
                             + self.cfg.eps)
        drift = vel_mag * 0.1 + atp * 0.05 - 0.01 * h_t
        diffusion = 0.01 + atp * 0.005
        return drift, diffusion

    # ---------------------------------------------------------- differentiable
    def _forward_impl(
        self,
        h_current: torch.Tensor,
        gamma_0: torch.Tensor,
        fluid_velocity: torch.Tensor,
        fluid_pressure: torch.Tensor,
        ion_concentrations: torch.Tensor,
        electric_potential: torch.Tensor,
        metabolic_species: torch.Tensor,
        vascular_density: torch.Tensor,
        cell_viability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        step = int(self.step.item())

        # ---- Phase 1: continuous biophysics --------------------------------
        bio_state = self.bio_solver.step_simulation(
            fluid_velocity, fluid_pressure, ion_concentrations,
            electric_potential, metabolic_species, vascular_density, cell_viability,
        )

        # ---- Phase 2: interface SDE (fixed topology) -----------------------
        drift, diffusion = self.compute_interface_drift_diffusion(
            h_current, bio_state["velocity"], bio_state["metabolic_rates"])
        dW = _seeded_randn_like(h_current, step=step, seed=self.cfg.seed) \
             * math.sqrt(self.cfg.dt)
        h_predict = h_current + drift * self.cfg.dt + diffusion * dW

        # ---- Phase 3: No-Zeno gate -----------------------------------------
        delta_e = self.landscape(h_predict)
        jump_gate, prob_bound = self.zeno_filter(delta_e, step=step)

        # ---- Phase 4: bio-topological operator (soft blend) ----------------
        h_plus, applied_gate = self.bio_topo_ops(
            h_predict, jump_gate, bio_op_type=self.cfg.bio_op_type, step=step,
        )

        # ---- Phase 5: per-voxel chart re-centering (differentiable) --------
        gamma_new = applied_gate * h_plus + (1.0 - applied_gate) * gamma_0
        h_new = (1.0 - applied_gate) * h_predict

        # Differentiable vascular-density response to jumps.
        vasc_updated = vascular_density + applied_gate * 0.1

        with torch.no_grad():
            self.step.add_(1)

        return {
            "h_graph": h_new,
            "gamma_0_reference": gamma_new,
            "velocity": bio_state["velocity"],
            "ion_concentrations": bio_state["ion_concentrations"],
            "metabolic_species": bio_state["metabolic_species"],
            "vascular_density_updated": vasc_updated,
            "jump_gate": applied_gate,
            "jump_prob_bound": prob_bound,
            "topological_jump_occurred": (applied_gate > 0.5).any().detach(),
        }

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if self.cfg.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, *args,
                              use_reentrant=False, **kwargs)
        return self._forward_impl(*args, **kwargs)


# ===============================================================================
# DDP wrapping & training utilities
# ===============================================================================
def wrap_ddp(model: nn.Module, cfg: SESIBiophysConfig,
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


# ===============================================================================
# Mock biophysics bridge (drop in your BiophysicalDNSBridge here)
# ===============================================================================
class _MockBiophysicsBridge(nn.Module):
    """Stand-in for BiophysicalDNSBridge — returns same-shaped outputs."""

    def step_simulation(self, velocity, pressure, ions, phi,
                        species, vasc, viability) -> Dict[str, torch.Tensor]:
        return {
            "velocity": velocity,
            "ion_concentrations": ions,
            "metabolic_species": species,
            "metabolic_rates": {
                "ATP_production": torch.ones_like(viability) * 0.02,
                "Lipid_synthesis": torch.ones_like(viability) * 0.005,
            },
        }


# ===============================================================================
# Demo (torchrun --standalone --nproc_per_node=N sesi_biophys_v2.py)
# ===============================================================================
def _demo() -> None:
    cfg = SESIBiophysConfig(
        dt=1e-4, dx=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        distributed=False,
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        compile=False,
        bio_op_type="branching",
        soft_operator_mix=False,
    )
    enable_fast_math()
    initialize_distributed(cfg)

    bridge = _MockBiophysicsBridge()
    model = wrap_ddp(SESIBiophysicalEngine(cfg, bridge).to(cfg.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda", enabled=False)

    B, Z, Y, X = 1, 32, 32, 32
    dev, dt = cfg.device, cfg.dtype
    batch = {
        "h_current":          torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.1,
        "gamma_0":            torch.zeros(B, 1, Z, Y, X, device=dev, dtype=dt),
        "fluid_velocity":     torch.randn(B, 3, Z, Y, X, device=dev, dtype=dt) * 1e-3,
        "fluid_pressure":     torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 10.0,
        "ion_concentrations": torch.rand(B, 3, Z, Y, X, device=dev, dtype=dt) * 0.15,
        "electric_potential": torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.05,
        "metabolic_species":  torch.rand(B, 4, Z, Y, X, device=dev, dtype=dt) * 0.1,
        "vascular_density":   torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt).clamp_min(0.05),
        "cell_viability":     torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt).clamp_min(0.1),
    }

    def loss_fn(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (out["h_graph"].pow(2).mean()
                + out["gamma_0_reference"].pow(2).mean()
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
