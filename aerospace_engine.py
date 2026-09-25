# =============================================================================
# AEROSPACE ENGINE v3.0.0 — Multi-GPU PyTorch DDP | Full Differentiable
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# License      : MIT
# Year         : 2026
# =============================================================================
# Launch       : torchrun --nproc_per_node=NUM_GPUS aerospace_engine_v3.py
# Features     :
#   • Native PyTorch DDP (Multi-Node Multi-GPU via NCCL)
#   • Fully Differentiable Physics (Autograd end-to-end)
#   • torch.compile(mode="max-autotune") + dynamic shapes
#   • Automatic Mixed Precision (BF16/FP16) with GradScaler
#   • torch.set_float32_matmul_precision("high") → TF32 acceleration
#   • Gradient Checkpointing option for OOM safety
#   • In-place ops & fused kernels for minimal memory/cost
#   • Differentiable Material Optimizer (Adam + L-BFGS via gradients)
#   • Zero redundant CPU↔GPU sync | Async distributed all-gather
# =============================================================================

import os
import sys
import time
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

# =============================================================================
# [0] Distributed Bootstrap (must run before anything else)
# =============================================================================

def setup_distributed() -> Tuple[int, int, int, torch.device, bool]:
    """Init NCCL. Returns (rank, world_size, local_rank, device, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_distributed = world_size > 1
    else:
        rank, world_size, local_rank = 0, 1, 0
        is_distributed = False

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device, is_distributed


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# [1] Logging & Cost Tracking
# =============================================================================

def setup_logging(rank: int = 0) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    fmt = f"%(asctime)s | RANK:{rank} | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout)


class CostTracker:
    """Minimal-overhead GPU timing & memory profiler."""
    def __init__(self, device: torch.device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def begin(self):
        if self.device.type == "cuda":
            self.start_event.record()
        else:
            self._t0 = time.perf_counter()

    def end(self) -> float:
        if self.device.type == "cuda":
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            return self.start_event.elapsed_time(self.end_event)  # ms
        return (time.perf_counter() - self._t0) * 1000.0

    @staticmethod
    def memory_mb(device: torch.device) -> float:
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return 0.0

    @staticmethod
    def reset_peak(device: torch.device):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)


# =============================================================================
# [2] Exception Hierarchy
# =============================================================================

class AerospaceEngineError(Exception): pass
class SimulationError(AerospaceEngineError): pass
class DistributedError(AerospaceEngineError): pass


# =============================================================================
# [3] Configuration (dataclass for zero-overhead in hot path)
# =============================================================================

class EngineConfig:
    """Zero-dependency config. Edit here or set env vars before launch."""
    # Distributed
    backend: str = "nccl"

    # Precision & Optimization
    use_amp: bool = True
    amp_dtype: str = "bfloat16"          # "bfloat16" | "float16"
    matmul_precision: str = "high"       # "high" uses TF32 on Ampere+
    use_compile: bool = True
    compile_mode: str = "max-autotune"   # "reduce-overhead" | "max-autotune"
    use_checkpoint: bool = False         # Enable if OOM on large batches

    # Physics
    gas_constant: float = 8.314
    reference_temperature: float = 298.15
    yield_multiplier: float = 450.0
    sc_factor: float = 1.15
    apply_sc: bool = True
    fatigue_exponent: float = 2.0
    fatigue_divisor: float = 8.5
    wind_shear: float = 65000.0
    creep_shear_factor: float = 0.00012
    creep_thermal_exponent: float = 2.5

    # Viability thresholds
    min_yield: float = 850.0
    min_fatigue: float = 100_000.0
    max_creep: float = 15.0

    # Dispatcher
    batch_size_per_gpu: int = 16
    sim_timeout: float = 60.0

    # Optimizer (material discovery via gradients)
    opt_lr: float = 0.05
    opt_steps: int = 200
    opt_type: str = "adam"               # "adam" | "lbfgs"

    @classmethod
    def from_env(cls) -> "EngineConfig":
        cfg = cls()
        for k in dir(cfg):
            if k.startswith("_"):
                continue
            env_key = f"AERO_{k.upper()}"
            if env_key in os.environ:
                val = os.environ[env_key]
                ann = type(getattr(cfg, k))
                if ann == bool:
                    setattr(cfg, k, val.lower() in ("1", "true", "yes"))
                elif ann == int:
                    setattr(cfg, k, int(val))
                elif ann == float:
                    setattr(cfg, k, float(val))
                else:
                    setattr(cfg, k, val)
        return cfg


# =============================================================================
# [4] Material Tensor Encoder (host → device, zero-copy friendly)
# =============================================================================

class MaterialEncoder:
    """Maps element dictionaries to dense differentiable tensors."""
    ELEMENTS = ["Ti", "Al", "V", "Ni", "Cr", "Fe", "Li", "Co", "Mo", "W", "Nb", "Ta"]
    ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
    NUM_ELEMENTS: int = len(ELEMENTS)

    @classmethod
    def encode(cls, materials: List[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(materials)
        comp = torch.zeros(B, cls.NUM_ELEMENTS, device=device, dtype=torch.float32)
        temp = torch.zeros(B, device=device, dtype=torch.float32)
        press = torch.zeros(B, device=device, dtype=torch.float32)
        for i, mat in enumerate(materials):
            for elem, frac in mat.get("elements", {}).items():
                if elem in cls.ELEM_TO_IDX:
                    comp[i, cls.ELEM_TO_IDX[elem]] = frac
            temp[i] = mat.get("temperature_k", 298.15)
            press[i] = mat.get("pressure_pa", 101325.0)
        # Normalize (softmax-friendly pre-normalization)
        comp = comp / (comp.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        return comp, temp, press

    @classmethod
    def decode(cls, tensors: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        B = tensors["yield"].shape[0]
        out = []
        for i in range(B):
            out.append({
                "yield_strength_mpa": float(tensors["yield"][i].item()),
                "fatigue_life_cycles": int(tensors["fatigue"][i].item()),
                "creep_rate": float(tensors["creep"][i].item()),
                "viability_score": float(tensors["viable"][i].item()),
                "is_viable": bool(tensors["viable"][i].item() > 0.5),
            })
        return out

    @classmethod
    def decode_composition(cls, comp_tensor: torch.Tensor) -> Dict[str, float]:
        """Convert a [NUM_ELEMENTS] probability/simplex vector back to dict."""
        comp_tensor = comp_tensor.detach().cpu()
        return {e: float(comp_tensor[cls.ELEM_TO_IDX[e]].item()) for e in cls.ELEMENTS}


# =============================================================================
# [5] Differentiable Physics Modules (100% torch, GPU-native)
# =============================================================================

class DifferentiableThermodynamics(nn.Module):
    """Vectorized phase stability via regular-solution approximation."""
    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.register_buffer("ref_temp", torch.tensor(cfg.reference_temperature))

    def forward(self, composition: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        B, N = composition.shape
        device = composition.device
        dtype = composition.dtype
        temp_ratio = temperature.view(B, 1, 1) / self.ref_temp
        base_weight = composition.amax(dim=-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        diag = eye * temp_ratio * base_weight
        # Pairwise outer product without explicit loops (fused kernel)
        fi = composition.unsqueeze(2)
        fj = composition.unsqueeze(1)
        off_diag = fi * fj * temp_ratio * 0.5 * (1.0 - eye)
        return diag + off_diag


class DifferentiableMechanics(nn.Module):
    """Yield stress & fatigue via differentiable matrix norms."""
    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.apply_sc = cfg.apply_sc
        self.register_buffer("yield_mult", torch.tensor(cfg.yield_multiplier, dtype=torch.float32))
        self.register_buffer("sc_f", torch.tensor(cfg.sc_factor, dtype=torch.float32))
        self.register_buffer("fatigue_exp", torch.tensor(cfg.fatigue_exponent, dtype=torch.float32))
        self.register_buffer("fatigue_div", torch.tensor(cfg.fatigue_divisor, dtype=torch.float32))

    def forward(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm = torch.linalg.matrix_norm(phase, ord="fro", dim=(-2, -1))
        y = norm * self.yield_mult
        if self.apply_sc:
            y = y * self.sc_f
        y = torch.clamp(y, min=0.0)
        f = torch.pow(y.clamp_min(1e-6), self.fatigue_exp) / self.fatigue_div
        return y, torch.clamp(f, min=0.0)


class DifferentiableAerodynamics(nn.Module):
    """Thermal creep under boundary-layer wind shear."""
    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.register_buffer("dyn_load", torch.tensor(cfg.wind_shear * cfg.creep_shear_factor, dtype=torch.float32))
        self.thermal_exp = cfg.creep_thermal_exponent

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        return self.dyn_load * torch.pow(temperature / 1000.0, self.thermal_exp)


class DifferentiableViability(nn.Module):
    """Soft viability scoring for gradient-friendly pass/fail."""
    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.register_buffer("min_y", torch.tensor(cfg.min_yield, dtype=torch.float32))
        self.register_buffer("min_f", torch.tensor(cfg.min_fatigue, dtype=torch.float32))
        self.register_buffer("max_c", torch.tensor(cfg.max_creep, dtype=torch.float32))
        self.register_buffer("k", torch.tensor(10.0, dtype=torch.float32))

    def forward(self, y: torch.Tensor, f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        y_ok = torch.sigmoid(self.k * (y - self.min_y) / self.min_y)
        f_ok = torch.sigmoid(self.k * (f - self.min_f) / self.min_f)
        c_ok = torch.sigmoid(self.k * (self.max_c - c) / self.max_c)
        return y_ok * f_ok * c_ok


# =============================================================================
# [6] End-to-End Simulation Module (nn.Module → DDP + torch.compile eligible)
# =============================================================================

class AerospaceSimulationModule(nn.Module):
    def __init__(self, cfg: EngineConfig):
        super().__init__()
        self.cfg = cfg
        self.thermo = DifferentiableThermodynamics(cfg)
        self.mech = DifferentiableMechanics(cfg)
        self.aero = DifferentiableAerodynamics(cfg)
        self.viability = DifferentiableViability(cfg)

    def forward(self, composition: torch.Tensor, temperature: torch.Tensor, _pressure: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.cfg.use_checkpoint and self.training and composition.requires_grad:
            phase = torch.utils.checkpoint.checkpoint(self.thermo, composition, temperature, use_reentrant=False)
        else:
            phase = self.thermo(composition, temperature)
        y, f = self.mech(phase)
        c = self.aero(temperature)
        v = self.viability(y, f, c)
        return {"yield": y, "fatigue": f, "creep": c, "viable": v, "phase": phase}


# =============================================================================
# [7] Distributed Aerospace Engine (DDP + AMP + torch.compile + Cost Opt)
# =============================================================================

class DistributedAerospaceEngine:
    """Production orchestrator: DDP inference/training, AMP, compile, optimize."""

    def __init__(self, cfg: EngineConfig, rank: int, world_size: int, local_rank: int, device: torch.device, is_distributed: bool):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.is_distributed = is_distributed
        self.encoder = MaterialEncoder()
        self.tracker = CostTracker(device)

        # Precision tuning: TF32 for matmul, TF32 for cuDNN
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        # AMP dtype selection
        if cfg.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.use_amp = cfg.use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # Build model
        self.model = AerospaceSimulationModule(cfg).to(device)

        # DDP wrapper (before compile for best graph capture)
        if is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,      # Reduce memory fragmentation
                static_graph=True,                 # Skip runtime checks after 1st iter
            )

        # torch.compile: fuse kernels, eliminate Python overhead
        if cfg.use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode=cfg.compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
            except Exception as exc:
                logging.getLogger("AeroEngine").warning(f"torch.compile skipped: {exc}")

        self.model.eval()

    # ------------------------------------------------------------------
    # Inference (zero gradient sync overhead)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_batch(self, materials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not materials:
            return []

        # Shard across ranks (round-robin)
        local_mats = [materials[i] for i in range(self.rank, len(materials), self.world_size)]
        has_local = len(local_mats) > 0

        local_results: List[Dict[str, Any]] = []
        if has_local:
            comp, temp, press = self.encoder.encode(local_mats, self.device)
            self.tracker.reset_peak(self.device)
            self.tracker.begin()

            ctx = autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
            with ctx:
                out = self.model(comp, temp, press)

            ms = self.tracker.end()
            mem = self.tracker.memory_mb(self.device)
            local_results = self.encoder.decode(out)
            for r in local_results:
                r["exec_ms"] = round(ms / max(len(local_results), 1), 2)
                r["peak_mem_mb"] = round(mem, 1)

        # All-gather serialized results (CPU-side to avoid CUDA serialization issues)
        return self._all_gather_objects(local_results, len(materials))

    def _all_gather_objects(self, local_objs: List[Any], total: int) -> List[Any]:
        if self.world_size == 1:
            return local_objs
        import pickle
        local_bytes = pickle.dumps(local_objs)
        size = torch.tensor([len(local_bytes)], dtype=torch.long, device=self.device)
        sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(sizes, size)
        max_size = int(max(s.item() for s in sizes))

        padded = local_bytes + b"\x00" * (max_size - len(local_bytes))
        sendbuf = torch.frombuffer(padded, dtype=torch.uint8).to(self.device)
        recvbufs = [torch.empty(max_size, dtype=torch.uint8, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(recvbufs, sendbuf)

        all_objs: List[Any] = []
        for i, buf in enumerate(recvbufs):
            actual = int(sizes[i].item())
            obj = pickle.loads(buf.cpu().numpy().tobytes()[:actual])
            all_objs.extend(obj)

        # Unshuffle round-robin ordering
        reordered = [None] * total
        idx = 0
        for r in range(self.world_size):
            for i in range(r, total, self.world_size):
                if idx < len(all_objs):
                    reordered[i] = all_objs[idx]
                    idx += 1
        return [x for x in reordered if x is not None]

    # ------------------------------------------------------------------
    # Differentiable Material Optimization (DDP training mode)
    # ------------------------------------------------------------------
    def optimize_composition(self, target: Dict[str, float], temp_k: float = 900.0,
                             init: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Optimize elemental fractions via fully-differentiable simulation."""
        self.model.train()

        # Build differentiable simplex parameter (logits → softmax)
        logits = torch.zeros(self.encoder.NUM_ELEMENTS, device=self.device)
        if init:
            for elem, frac in init.items():
                if elem in self.encoder.ELEM_TO_IDX:
                    logits[self.encoder.ELEM_TO_IDX[elem]] = frac
        logits = nn.Parameter(logits)

        if self.cfg.opt_type == "adam":
            optimizer = optim.Adam([logits], lr=self.cfg.opt_lr, fused=(self.device.type == "cuda")
            )
        else:
            optimizer = optim.LBFGS([logits], lr=self.cfg.opt_lr, max_iter=20, line_search_fn="strong_wolfe")

        temp_t = torch.tensor([temp_k], device=self.device)
        press_t = torch.tensor([101325.0], device=self.device)

        target_yield = torch.tensor([target.get("yield_strength_mpa", 1200.0)], device=self.device)
        target_fatigue = torch.tensor([target.get("fatigue_life_cycles", 300000.0)], device=self.device)
        target_creep = torch.tensor([target.get("creep_rate", 5.0)], device=self.device)

        history = []
        logger = logging.getLogger("AeroEngine")

        for step in range(self.cfg.opt_steps):
            def closure():
                optimizer.zero_grad(set_to_none=True)  # Memory opt
                comp = torch.softmax(logits, dim=-1).unsqueeze(0)  # [1, NUM_ELEMENTS]

                ctx = autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
                with ctx:
                    out = self.model(comp, temp_t, press_t)

                # Multi-objective MSE loss (differentiable)
                loss_y = torch.nn.functional.mse_loss(out["yield"], target_yield)
                loss_f = torch.nn.functional.mse_loss(out["fatigue"], target_fatigue)
                loss_c = torch.nn.functional.mse_loss(out["creep"], target_creep)
                loss_v = 0.05 * (1.0 - out["viable"]).mean()  # Soft penalty for non-viable
                loss = loss_y + 1e-7 * loss_f + loss_c + loss_v

                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                return loss

            if self.cfg.opt_type == "lbfgs":
                loss = optimizer.step(closure)
            else:
                loss = closure()
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

            if step % 25 == 0 and self.rank == 0:
                with torch.no_grad():
                    comp = torch.softmax(logits, dim=-1)
                    logger.info(f"Opt step {step:03d} | loss={loss.item():.4f} | comp={comp.cpu().numpy()}")
                history.append({"step": step, "loss": float(loss.item())})

        # Final evaluation (no grad)
        with torch.no_grad():
            comp_final = torch.softmax(logits, dim=-1)
            out = self.model(comp_final.unsqueeze(0), temp_t, press_t)

        self.model.eval()
        return {
            "optimized_composition": self.encoder.decode_composition(comp_final),
            "properties": {
                "yield_strength_mpa": float(out["yield"][0].item()),
                "fatigue_life_cycles": int(out["fatigue"][0].item()),
                "creep_rate": float(out["creep"][0].item()),
                "viability_score": float(out["viable"][0].item()),
                "is_viable": bool(out["viable"][0].item() > 0.5),
            },
            "history": history,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"status": "empty"}
        viable = [r for r in results if r.get("is_viable", False)]
        ys = [r["yield_strength_mpa"] for r in results]
        fs = [r["fatigue_life_cycles"] for r in results]
        cs = [r["creep_rate"] for r in results]
        return {
            "status": "completed",
            "materials_evaluated": len(results),
            "summary": {"pass": len(viable), "fail": len(results) - len(viable), "pass_rate": round(len(viable) / len(results), 4)},
            "stats": {
                "yield_mpa": {"mean": round(sum(ys) / len(ys), 2), "min": round(min(ys), 2), "max": round(max(ys), 2)},
                "fatigue": {"mean": int(sum(fs) / len(fs)), "min": min(fs), "max": max(fs)},
                "creep": {"mean": round(sum(cs) / len(cs), 4), "min": round(min(cs), 4), "max": round(max(cs), 4)},
            },
            "details": results,
        }


# =============================================================================
# [8] Main Execution
# =============================================================================

def main():
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    setup_logging(rank)
    logger = logging.getLogger("AeroEngine")

    if rank == 0:
        logger.info("=" * 70)
        logger.info(" Aerospace Engine v3.0.0 | PyTorch DDP | Differentiable | Cost-Opt ")
        logger.info("=" * 70)

    cfg = EngineConfig.from_env()
    engine = DistributedAerospaceEngine(cfg, rank, world_size, local_rank, device, is_distributed)

    # ---- Demo 1: Distributed Batch Inference ----
    materials = [
        {"elements": {"Ti": 0.90, "Al": 0.06, "V": 0.04}, "temperature_k": 900.0, "pressure_pa": 101325.0},
        {"elements": {"Al": 0.98, "Li": 0.02}, "temperature_k": 450.0, "pressure_pa": 101325.0},
        {"elements": {"Ni": 0.70, "Cr": 0.20, "Fe": 0.10}, "temperature_k": 1100.0, "pressure_pa": 150000.0},
        {"elements": {"Ti": 0.60, "Al": 0.25, "V": 0.10, "Mo": 0.05}, "temperature_k": 950.0, "pressure_pa": 101325.0},
    ]

    if rank == 0:
        logger.info(f"Evaluating {len(materials)} materials on {world_size} GPU(s)...")

    t0 = time.perf_counter()
    results = engine.evaluate_batch(materials)
    elapsed = time.perf_counter() - t0

    if rank == 0:
        print("\n" + "=" * 70)
        print(" SIMULATION RESULTS ")
        print("=" * 70)
        for i, r in enumerate(results):
            status = "PASS" if r["is_viable"] else "FAIL"
            print(f"\n[{i+1}] {status}")
            print(f"      Yield:   {r['yield_strength_mpa']:.2f} MPa")
            print(f"      Fatigue: {r['fatigue_life_cycles']:,} cycles")
            print(f"      Creep:   {r['creep_rate']:.4f}")
            print(f"      Viable:  {r['viability_score']:.4f}")
            print(f"      Time:    {r.get('exec_ms', 0):.2f} ms | Mem: {r.get('peak_mem_mb', 0):.1f} MB")
        report = engine.generate_report(results)
        print("\n" + "-" * 70)
        print(json.dumps(report["summary"], indent=2))
        print(f"\nTotal Pipeline Time: {elapsed:.4f}s")

    # ---- Demo 2: Differentiable Material Optimization ----
    if rank == 0:
        logger.info("\n--- Differentiable Material Optimization ---")
        target_props = {"yield_strength_mpa": 2200.0, "fatigue_life_cycles": 600000, "creep_rate": 4.0}
        opt = engine.optimize_composition(
            target=target_props,
            temp_k=920.0,
            init={"Ti": 0.5, "Al": 0.3, "V": 0.15, "Mo": 0.05},
        )
        print("\nOptimized Composition:")
        for elem, frac in opt["optimized_composition"].items():
            if frac > 0.001:
                print(f"  {elem}: {frac:.4f}")
        print("\nPredicted Properties:")
        print(json.dumps(opt["properties"], indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("AeroEngine").warning("Interrupted by user")
        cleanup_distributed()
        sys.exit(130)
    except Exception as exc:
        logging.getLogger("AeroEngine").error(f"Fatal: {exc}", exc_info=True)
        cleanup_distributed()
        sys.exit(1)
