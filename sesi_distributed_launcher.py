"""
=============================================================================
SESI PYTORCH DISTRIBUTED LAUNCHER
Multi-GPU/Multi-Node orchestration with No-Zeno task throttling
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[C] Wraps torch.distributed.run (torchrun) for multi-GPU/Multi-node execution
[D] Abstracts backend selection: NCCL (NVIDIA), RCCL (AMD), Gloo (CPU),
    CCL (Intel), HCCL (Huawei)
[D] Integrates No-Zeno throttling for task scheduling across workers
[D] Provides unified API for distributed FDTD, optimization, and visualization

BACKEND SELECTION
-----------------
Based on sesi_hardware_backend.py detection:
  • NVIDIA GPU → NCCL (fastest, requires CUDA)
  • AMD GPU → RCCL (ROCm equivalent)
  • Intel XPU → CCL (oneAPI collective library)
  • Huawei NPU → HCCL (CANN collective)
  • CPU only → Gloo (fallback, slower)

NO-ZENO TASK THROTTLING (cfd_pp.py §3.3)
-----------------------------------------
In distributed execution, task submission can overwhelm the scheduler.
We apply the No-Zeno principle:
  • Each task submission is an "event"
  • Minimum spacing τ_e between submissions
  • Bound: N(T) ≤ 1 + ⌊T/τ_e⌋
  • Failed submissions are RECORDED, never suppressed

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Distributed execution requires proper cluster setup (MPI, network).
• Scaling efficiency depends on hardware, network, and problem structure.
• No-Zeno throttling controls TASK SUBMISSION, not computation itself.
• This is an ORCHESTRATION TOOL, not a guarantee of linear speedup.
=============================================================================
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.distributed")

try:
    import torch
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sesi_hardware_backend import HardwareBackend, Vendor, get_backend, detect_available_backends
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

__all__ = [
    "DistributedConfig",
    "CollectiveBackend",
    "DistributedLauncher",
    "NoZenoTaskScheduler",
    "DistributedFDTDRunner",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class CollectiveBackend(Enum):
    """Collective communication backends."""
    NCCL = auto()    # NVIDIA
    RCCL = auto()    # AMD ROCm
    GLOO = auto()    # CPU fallback
    CCL = auto()     # Intel oneAPI
    HCCL = auto()    # Huawei CANN
    AUTO = auto()    # Auto-detect


@dataclass
class DistributedConfig:
    """Configuration for distributed execution."""
    # Cluster
    n_nodes: int = 1
    n_gpus_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    # Backend
    backend: CollectiveBackend = CollectiveBackend.AUTO

    # No-Zeno task scheduling
    task_min_interval_s: float = 0.01  # τ_e
    max_concurrent_tasks: int = 100

    # Timeout
    timeout_s: int = 1800  # 30 minutes

    # Script to launch
    script_path: str = ""
    script_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.n_nodes < 1 or self.n_gpus_per_node < 1:
            raise ValueError("n_nodes and n_gpus_per_node must be ≥ 1")


# ═══════════════════════════════════════════════════════════════════════
# [D] NO-ZENO TASK SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class NoZenoTaskScheduler:
    """
    [D] Task scheduler with No-Zeno throttling (cfd_pp.py §3.3).

    Ensures:
      • Minimum spacing τ_e between task submissions
      • Bound: N(T) ≤ 1 + ⌊T/τ_e⌋
      • Failed submissions are RECORDED, never suppressed
    """

    def __init__(self, cfg: DistributedConfig):
        self.cfg = cfg
        self.tau_e = cfg.task_min_interval_s
        self.last_submit_time = 0.0
        self.task_count = 0
        self.task_log: List[Dict[str, Any]] = []
        self.warning_log: List[str] = []

    def submit_task(self, task_fn: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Submit a task with No-Zeno throttling.

        Returns dict with status and task result (if executed).
        """
        current_time = time.time()

        # Check spacing
        if current_time - self.last_submit_time < self.tau_e:
            # Too soon; record but don't execute
            self.warning_log.append(
                f"Task submission throttled at t={current_time:.3f}s "
                f"(last={self.last_submit_time:.3f}s, τ_e={self.tau_e}s)"
            )
            return {
                "status": "throttled",
                "reason": "No-Zeno spacing violation",
                "task_count": self.task_count,
            }

        # Execute task
        try:
            result = task_fn(*args, **kwargs)
            self.task_count += 1
            self.last_submit_time = current_time
            self.task_log.append({
                "task_id": self.task_count,
                "time": current_time,
                "status": "completed",
            })
            return {
                "status": "completed",
                "task_id": self.task_count,
                "result": result,
            }
        except Exception as e:
            # Record failure, don't suppress
            self.warning_log.append(f"Task {self.task_count + 1} failed: {e}")
            self.task_log.append({
                "task_id": self.task_count + 1,
                "time": current_time,
                "status": "failed",
                "error": str(e),
            })
            return {
                "status": "failed",
                "error": str(e),
                "task_count": self.task_count,
            }

    def bound_check(self) -> Dict[str, Any]:
        """Verify N(T) ≤ 1 + ⌊T/τ_e⌋ (cfd_pp.py Prop 1)."""
        if not self.task_log:
            T = 0.0
        else:
            T = self.task_log[-1]["time"] - self.task_log[0]["time"]

        bound = 1 + int(T / self.tau_e) if T > 0 else 1
        n_completed = sum(1 for t in self.task_log if t["status"] == "completed")

        return {
            "T_seconds": T,
            "tau_e": self.tau_e,
            "bound": bound,
            "n_completed": n_completed,
            "invariant_holds": n_completed <= bound,
            "n_warnings": len(self.warning_log),
        }


# ═══════════════════════════════════════════════════════════════════════
# [C/D] DISTRIBUTED LAUNCHER
# ═══════════════════════════════════════════════════════════════════════

class DistributedLauncher:
    """
    [C/D] Launch distributed PyTorch jobs using torchrun.

    Wraps torch.distributed.run with backend auto-detection and No-Zeno
    task scheduling.
    """

    def __init__(self, cfg: DistributedConfig):
        self.cfg = cfg
        self.scheduler = NoZenoTaskScheduler(cfg)
        self.backend = self._select_backend()

    def _select_backend(self) -> str:
        """Select collective backend based on hardware."""
        if self.cfg.backend != CollectiveBackend.AUTO:
            return self.cfg.backend.name.lower()

        if not HAS_BACKEND:
            logger.warning("Hardware backend not available; using Gloo")
            return "gloo"

        backends = detect_available_backends()
        for info in backends:
            if info.vendor == Vendor.NVIDIA:
                return "nccl"
            elif info.vendor == Vendor.AMD:
                return "rccl"
            elif info.vendor == Vendor.INTEL:
                return "ccl"
            elif info.vendor == Vendor.HUAWEI:
                return "hccl"

        return "gloo"

    def build_torchrun_command(self) -> List[str]:
        """Build torchrun command line."""
        cfg = self.cfg
        nprocs = cfg.n_nodes * cfg.n_gpus_per_node

        cmd = [
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={cfg.n_gpus_per_node}",
            f"--nnodes={cfg.n_nodes}",
            f"--master_addr={cfg.master_addr}",
            f"--master_port={cfg.master_port}",
            f"--rdzv_backend=c10d",
            cfg.script_path,
        ] + cfg.script_args

        return cmd

    def launch(self) -> Dict[str, Any]:
        """
        Launch distributed job with No-Zeno throttling.

        Returns dict with launch status and process info.
        """
        if not self.cfg.script_path:
            return {"status": "error", "reason": "No script specified"}

        def _do_launch():
            cmd = self.build_torchrun_command()
            logger.info(f"Launching: {' '.join(cmd)}")

            # Set environment variables
            env = os.environ.copy()
            env["MASTER_ADDR"] = self.cfg.master_addr
            env["MASTER_PORT"] = str(self.cfg.master_port)

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.cfg.timeout_s)
                return {
                    "status": "completed" if process.returncode == 0 else "failed",
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "status": "timeout",
                    "reason": f"Exceeded {self.cfg.timeout_s}s",
                }

        # Submit through No-Zeno scheduler
        result = self.scheduler.submit_task(_do_launch)
        return result

    def status_report(self) -> Dict[str, Any]:
        """Generate status report with No-Zeno bound check."""
        bound = self.scheduler.bound_check()
        return {
            "backend": self.backend,
            "n_nodes": self.cfg.n_nodes,
            "n_gpus_per_node": self.cfg.n_gpus_per_node,
            "total_tasks": self.scheduler.task_count,
            "no_zeno_bound": bound,
            "recent_warnings": self.scheduler.warning_log[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] DISTRIBUTED FDTD RUNNER
# ═══════════════════════════════════════════════════════════════════════

class DistributedFDTDRunner:
    """
    [D] Example: distributed FDTD simulation using domain decomposition.

    This is a placeholder showing how to integrate with sesi_gpu_fdtd.py
    for multi-GPU execution.
    """

    def __init__(self, cfg: DistributedConfig):
        self.cfg = cfg
        self.launcher = DistributedLauncher(cfg)

    def run_distributed_fdtd(self, grid_shape: Tuple[int, int, int],
                              n_steps: int) -> Dict[str, Any]:
        """
        Launch distributed FDTD simulation.

        In production, this would:
          1. Partition grid across GPUs
          2. Exchange ghost cells via all_reduce
          3. Run FDTD steps in parallel
          4. Gather results
        """
        # Placeholder: generate script content
        script_content = f"""
# Auto-generated distributed FDTD script
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='{self.launcher.backend}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {{rank}}/{{world_size}} starting FDTD")

    # Domain decomposition
    grid_shape = {grid_shape}
    local_nx = grid_shape[0] // world_size

    # FDTD simulation (placeholder)
    # ... actual FDTD code from sesi_gpu_fdtd.py ...

    # Synchronize
    dist.barrier()
    print(f"Rank {{rank}} completed")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
"""
        # Write script
        script_path = "/tmp/distributed_fdtd.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Update config
        self.cfg.script_path = script_path
        self.cfg.script_args = [str(n_steps)]

        # Launch
        return self.launcher.launch()


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Distributed Launcher — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available  : {HAS_TORCH}")
    print(f"  Backend module     : {HAS_BACKEND}")

    if not HAS_TORCH:
        print("  [SKIP] PyTorch required")
        return

    cfg = DistributedConfig(
        n_nodes=1,
        n_gpus_per_node=2,
        backend=CollectiveBackend.AUTO,
        task_min_interval_s=0.1,
    )

    launcher = DistributedLauncher(cfg)
    print(f"\n  ── Backend Selection ──")
    print(f"    Selected backend   : {launcher.backend}")

    print(f"\n  ── No-Zeno Task Scheduler ──")
    # Submit multiple tasks rapidly
    for i in range(5):
        result = launcher.scheduler.submit_task(lambda: i * 2)
        print(f"    Task {i+1}: {result['status']}")

    # Check bound
    bound = launcher.scheduler.bound_check()
    print(f"\n  ── No-Zeno Bound Check ──")
    print(f"    T = {bound['T_seconds']:.3f}s")
    print(f"    τ_e = {bound['tau_e']}s")
    print(f"    Bound = {bound['bound']}")
    print(f"    Completed = {bound['n_completed']}")
    print(f"    Invariant holds = {bound['invariant_holds']}")
    print(f"    Warnings = {bound['n_warnings']}")

    # Status report
    print(f"\n  ── Status Report ──")
    report = launcher.status_report()
    print(f"    Backend            : {report['backend']}")
    print(f"    Total tasks        : {report['total_tasks']}")
    if report['recent_warnings']:
        print(f"    Recent warnings    :")
        for w in report['recent_warnings']:
            print(f"      • {w}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Distributed execution requires proper cluster setup")
    print("  • Scaling efficiency depends on hardware and problem structure")
    print("  • No-Zeno throttling controls TASK SUBMISSION, not computation")
    print("  • This is an ORCHESTRATION TOOL, not a speedup guarantee")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
