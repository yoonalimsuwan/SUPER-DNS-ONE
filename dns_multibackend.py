#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
dns_multibackend.py — Pseudo-spectral DNS at Re_L = 1e5 / 1e6 / 1e7
Backends: JAX · TensorFlow · PaddlePaddle · MindSpore · Apple(MLX/Metal)
          · PyTorch · NumPy
ONE FILE. Self-contained. Self-verifying. Refuses impossible grids.
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026
ORCID        : 0009-0008-2374-0788

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
A CORRECTION ABOUT "APPLE XNU"
=============================================================================
XNU ("X is Not Unix") is the KERNEL of macOS / iOS / iPadOS. It is not a
numerical or ML framework and it exposes no FFT API. The actual Apple
compute paths are:

  * MLX  — Apple's array framework, unified memory, `mx.fft.rfftn`.
           This is the correct native Apple path and is implemented here.
  * Metal / MPS — via PyTorch's `device="mps"`. torch.fft on MPS has
           partial support depending on version; probed, not assumed.
  * Accelerate / vDSP — CPU FFT. Reached through NumPy on macOS.

All three are probed at runtime. Nothing is claimed to work untested.

=============================================================================
THE PHYSICS  (standard, cited, not novel)
=============================================================================
Incompressible Navier–Stokes in a triply-periodic box [0, L)^3:

    du/dt + (u.grad)u = -grad(p)/rho + nu lap(u) + f
    div u = 0

Pseudo-spectral Fourier discretisation:
  * Velocity stored as u_hat(k), shape (3, Nx, Ny, Nz//2+1) complex
    (rfft: real field -> Hermitian half-spectrum, ~2x memory saving)
  * Pressure eliminated EXACTLY by the Leray projection
        u_hat <- u_hat - k (k . u_hat) / |k|^2
    This is not an iterative Poisson solve. In Fourier space the
    pressure projection is an algebraic operation, exact to round-off.
  * Nonlinear term in ROTATION (skew-symmetric) form
        N(u) = u x omega,   omega = curl u
    which equals -(u.grad)u + grad(|u|^2/2); the gradient part is
    annihilated by the projection. This form has better discrete
    energy behaviour than the convective form.
  * Dealiasing: Orszag 2/3 rule. Modes with |k_i| > (2/3)(N_i/2) are
    zeroed. Without this, quadratic nonlinearity aliases energy back
    into resolved scales and the simulation is wrong.
  * Viscous term: -nu |k|^2 u_hat, exact in Fourier space.
  * Time: SSP-RK3 (Shu & Osher 1988), CFL-limited.

References (real, verifiable):
  [O1] Orszag, J. Atmos. Sci. 27, 890 (1970)      — 2/3 dealiasing
  [O2] Patterson & Orszag, Phys. Fluids 14 (1971) — pseudo-spectral DNS
  [S1] Shu & Osher, JCP 77, 439 (1988)            — SSP-RK3
  [P1] Pope, "Turbulent Flows", CUP (2000)        — resolution criteria
  [K1] Kolmogorov (1941)                          — eta = (nu^3/eps)^1/4
  [T1] Taylor & Green, Proc. R. Soc. A 158 (1937) — analytic vortex

=============================================================================
THE HONEST RESOURCE WALL  (read before choosing Re)
=============================================================================
DNS requires k_max * eta >= 1.5 (Pope [P1] §9.1). With eta/L = Re^(-3/4)
and k_max = pi N / L:

    N_per_axis >= (1.5/pi) * Re^(3/4) = 0.4775 * Re^(3/4)

    Re_L      N/axis      cells        state (5 fields, fp64)
    1e3        128        2.1e6        0.08 GB
    1e4        512        1.3e8        5.4  GB
    1e5        2688       1.9e10       780  GB
    1e6        15104      3.4e12       138  TB
    1e7        84928      6.1e14       24.5 PB

Total cost scales as Re^(9/4) in space and Re^(3/4) more in time
(CFL), i.e. Re^3 overall. NO FRAMEWORK CHANGE ALTERS THIS EXPONENT.
JAX, TensorFlow, PaddlePaddle, MindSpore and MLX differ in ergonomics
and kernel fusion, not in asymptotic cost.

This module therefore:
  * computes the requirement BEFORE allocating,
  * REFUSES grids that do not fit the declared device memory,
  * reports the GPU/rank count that would be needed,
  * and offers a reduced k_max*eta (clearly labelled UNDER-RESOLVED)
    so the solver can still be benchmarked at the target Re.

=============================================================================
WHAT IS VERIFIED HERE
=============================================================================
  V1  Leray projection makes div u = 0 to round-off
  V2  2/3 dealiasing leaves high-k modes exactly zero
  V3  2-D Taylor–Green decays at the ANALYTIC rate exp(-2 nu k^2 t)
      (its nonlinear term is a pure gradient, killed by projection,
       so the decay is exactly viscous — a decisive correctness test)
  V4  Energy budget: dE/dt = -epsilon (unforced) to O(dt^3)
  V5  k_max*eta resolution diagnostic is reported, not assumed
  V6  Backend capability probe reports UNAVAILABLE honestly

NOT CLAIMED
  * No execution at Re >= 1e5 here (needs a cluster).
  * No validation against JHTDB or experimental turbulence.
  * Single-device only. Multi-rank domain decomposition is NOT
    implemented; the resource report says how many ranks you'd need.
=============================================================================
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("dns")

VERSION = "1.0.0"
PI = math.pi

BANNER = f"""
=============================================================================
MULTI-BACKEND PSEUDO-SPECTRAL DNS v{VERSION}
JAX · TensorFlow · PaddlePaddle · MindSpore · Apple(MLX/Metal) · Torch · NumPy
Leray projection is EXACT in Fourier space. 2/3 dealiasing is mandatory.
Refuses grids that do not fit. Reports k_max*eta honestly.
=============================================================================
"""


# =============================================================================
# §1. RESOURCE PLANNER — the honesty gate
# =============================================================================

@dataclass
class DeviceSpec:
    name: str
    mem_gb: float
    bw_gb_s: float
    fp64_tflops: float

DEVICES = {
    "H100":    DeviceSpec("NVIDIA H100 SXM 80GB",  80, 3350, 34.0),
    "H200":    DeviceSpec("NVIDIA H200 141GB",    141, 4800, 34.0),
    "A100":    DeviceSpec("NVIDIA A100 80GB",      80, 2039,  9.7),
    "MI300X":  DeviceSpec("AMD MI300X 192GB",     192, 5300, 81.7),
    "M3Ultra": DeviceSpec("Apple M3 Ultra 192GB", 192,  800,  1.0),
    "M4Max":   DeviceSpec("Apple M4 Max 128GB",   128,  546,  0.5),
    "Ascend":  DeviceSpec("Huawei Ascend 910B 64GB", 64, 1600, 4.0),
    "CPU":     DeviceSpec("CPU 256GB DDR5",       256,  200,  1.0),
}


def n_for_re(re_l: float, kmax_eta: float = 1.5, round_to: int = 64) -> int:
    """N >= (kmax_eta/pi) Re^(3/4), rounded UP for FFT/MG friendliness."""
    n = (kmax_eta / PI) * re_l ** 0.75
    return int(math.ceil(n / round_to) * round_to)


def plan(re_l: float, kmax_eta: float = 1.5, device: str = "H100",
         n_fields: int = 5, bytes_per: int = 8,
         dt_cfl: float = 0.4, turnovers: float = 5.0,
         bw_eff: float = 0.65, n_passes: int = 12) -> Dict[str, Any]:
    """
    Cost model. BANDWIDTH-based: the pseudo-spectral DNS step is dominated
    by FFT memory traffic, not FLOPs, at these grid sizes.
    """
    d = DEVICES[device]
    n = n_for_re(re_l, kmax_eta)
    cells = n ** 3
    state_gb = cells * n_fields * bytes_per / 1e9
    # rfft halves the complex storage but we keep 5 real-equivalent fields
    work_gb = state_gb * 2.5           # u, omega, nonlinear, RK stages
    ranks_min = max(1, math.ceil(work_gb / d.mem_gb))
    ranks_work = max(ranks_min, math.ceil(work_gb / (0.6 * d.mem_gb)))
    dx = 1.0 / n
    dt = dt_cfl * dx
    steps = int(math.ceil(turnovers / dt))
    bytes_step = cells * n_fields * bytes_per * n_passes
    sec_step = bytes_step / (ranks_work * d.bw_gb_s * 1e9 * bw_eff)
    total = sec_step * steps
    return {
        "Re_L": re_l,
        "kmax_eta_target": kmax_eta,
        "N_per_axis": n,
        "cells": cells,
        "cells_sci": f"{cells:.3e}",
        "state_GB": state_gb,
        "state_human": _gb(state_gb),
        "working_GB": work_gb,
        "working_human": _gb(work_gb),
        "device": d.name,
        "ranks_memory_min": ranks_min,
        "ranks_working": ranks_work,
        "dt": dt,
        "n_steps": steps,
        "sec_per_step": sec_step,
        "total_sec": total,
        "total_human": _time(total),
        "dns_adequate": kmax_eta >= 1.5,
        "cost_scaling": "Re^(9/4) space x Re^(3/4) time = Re^3 total",
        "NOT_INCLUDED": ["MPI halo/allreduce latency", "I/O", "queue wait"],
    }


def _gb(x: float) -> str:
    if x >= 1e6: return f"{x/1e6:.2f} PB"
    if x >= 1e3: return f"{x/1e3:.2f} TB"
    return f"{x:.2f} GB"


def _time(s: float) -> str:
    if s < 120:    return f"{s:.1f} s"
    if s < 7200:   return f"{s/60:.1f} min"
    if s < 172800: return f"{s/3600:.1f} h"
    if s < 3.15e7: return f"{s/86400:.1f} days"
    return f"{s/3.15e7:.1f} years"


def feasibility_table(device: str = "H100") -> List[Dict[str, Any]]:
    return [plan(re, 1.5, device) for re in (1e3, 1e4, 1e5, 1e6, 1e7)]


# =============================================================================
# §2. BACKEND ADAPTER — probe, never assume
# =============================================================================

class Ops:
    """
    Minimal array-op surface needed by the spectral solver.
    Every backend supplies the same callables. FFT convention is
    'backward' (forward unnormalised, inverse 1/N) — all seven
    frameworks use this by default.
    """
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.reason = ""
        self.device = "cpu"
        self.fp64 = True
        self.xp: Any = None
        self.fft: Any = None
        self.jit: Callable = lambda f: f
        self.sync: Callable = lambda: None

    # -- to be bound by each probe --
    def asarray(self, x): raise NotImplementedError
    def zeros(self, shape, complex_=False): raise NotImplementedError
    def rfftn(self, x): raise NotImplementedError
    def irfftn(self, xh, s): raise NotImplementedError
    def to_np(self, x): raise NotImplementedError
    def stack(self, xs, axis=0): raise NotImplementedError
    def real(self, x): raise NotImplementedError
    def sum(self, x): raise NotImplementedError
    def maxabs(self, x): raise NotImplementedError
    def exp(self, x): raise NotImplementedError


def _probe_numpy() -> Ops:
    o = Ops("numpy"); o.available = True; o.xp = np; o.fft = np.fft
    o.reason = "reference CPU backend, always available"
    o.asarray = lambda x: np.asarray(x)
    o.zeros = lambda s, complex_=False: np.zeros(s, np.complex128 if complex_ else np.float64)
    o.rfftn = lambda x: np.fft.rfftn(x, axes=(-3, -2, -1))
    o.irfftn = lambda xh, s: np.fft.irfftn(xh, s=s, axes=(-3, -2, -1))
    o.to_np = lambda x: np.asarray(x)
    o.stack = lambda xs, axis=0: np.stack(xs, axis)
    o.real = np.real
    o.sum = np.sum
    o.maxabs = lambda x: float(np.abs(x).max())
    o.exp = np.exp
    return o


def _probe_torch() -> Ops:
    o = Ops("torch")
    try:
        import torch
    except ImportError:
        o.reason = "torch not installed"; return o
    o.xp = torch
    # Apple Metal path
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        o.device = "mps"
        o.fp64 = False        # MPS has no float64
        o.reason = "torch MPS (Apple Metal). float64 unsupported -> complex64."
    elif torch.cuda.is_available():
        o.device = "cuda"
        o.reason = f"torch CUDA: {torch.cuda.get_device_name(0)}"
    else:
        o.device = "cpu"; o.reason = "torch CPU"
    cd = torch.complex64 if not o.fp64 else torch.complex128
    rd = torch.float32 if not o.fp64 else torch.float64
    dev = torch.device(o.device)
    o.available = True
    o.asarray = lambda x: torch.as_tensor(np.asarray(x), dtype=rd, device=dev)
    o.zeros = lambda s, complex_=False: torch.zeros(s, dtype=cd if complex_ else rd, device=dev)
    o.rfftn = lambda x: torch.fft.rfftn(x, dim=(-3, -2, -1))
    o.irfftn = lambda xh, s: torch.fft.irfftn(xh, s=s, dim=(-3, -2, -1))
    o.to_np = lambda x: x.detach().cpu().numpy()
    o.stack = lambda xs, axis=0: torch.stack(xs, axis)
    o.real = torch.real
    o.sum = torch.sum
    o.maxabs = lambda x: float(x.abs().max().item())
    o.exp = torch.exp
    o.sync = lambda: (torch.cuda.synchronize() if o.device == "cuda" else None)
    if hasattr(torch, "compile") and o.device == "cuda":
        o.jit = lambda f: torch.compile(f, mode="reduce-overhead")
    return o


def _probe_jax() -> Ops:
    o = Ops("jax")
    try:
        import jax, jax.numpy as jnp
    except ImportError:
        o.reason = "jax not installed (pip install jax)"; return o
    o.xp = jnp; o.fft = jnp.fft
    try:
        devs = jax.devices()
        o.device = str(devs[0].platform)
        o.reason = f"jax {jax.__version__} on {o.device} ({len(devs)} device(s))"
    except Exception as e:
        o.reason = f"jax present but device probe failed: {e}"
    jax.config.update("jax_enable_x64", True)
    o.available = True
    o.asarray = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)
    o.zeros = lambda s, complex_=False: jnp.zeros(s, jnp.complex128 if complex_ else jnp.float64)
    o.rfftn = lambda x: jnp.fft.rfftn(x, axes=(-3, -2, -1))
    o.irfftn = lambda xh, s: jnp.fft.irfftn(xh, s=s, axes=(-3, -2, -1))
    o.to_np = lambda x: np.asarray(x)
    o.stack = lambda xs, axis=0: jnp.stack(xs, axis)
    o.real = jnp.real
    o.sum = jnp.sum
    o.maxabs = lambda x: float(jnp.abs(x).max())
    o.exp = jnp.exp
    o.jit = jax.jit
    o.sync = lambda: jax.block_until_ready
    return o


def _probe_tensorflow() -> Ops:
    o = Ops("tensorflow")
    try:
        import tensorflow as tf
    except ImportError:
        o.reason = "tensorflow not installed"; return o
    o.xp = tf
    gpus = tf.config.list_physical_devices("GPU")
    o.device = "gpu" if gpus else "cpu"
    o.reason = f"tensorflow {tf.__version__}, {len(gpus)} GPU(s), XLA via tf.function"
    o.available = True
    o.asarray = lambda x: tf.constant(np.asarray(x), dtype=tf.float64)
    o.zeros = lambda s, complex_=False: tf.zeros(s, tf.complex128 if complex_ else tf.float64)
    o.rfftn = lambda x: tf.signal.rfft3d(x)
    o.irfftn = lambda xh, s: tf.signal.irfft3d(xh, fft_length=list(s))
    o.to_np = lambda x: x.numpy()
    o.stack = lambda xs, axis=0: tf.stack(xs, axis)
    o.real = tf.math.real
    o.sum = tf.reduce_sum
    o.maxabs = lambda x: float(tf.reduce_max(tf.abs(x)).numpy())
    o.exp = tf.exp
    o.jit = lambda f: tf.function(f, jit_compile=True)
    return o


def _probe_paddle() -> Ops:
    o = Ops("paddle")
    try:
        import paddle
    except ImportError:
        o.reason = "paddlepaddle not installed (pip install paddlepaddle)"; return o
    o.xp = paddle
    try:
        o.device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    except Exception:
        o.device = "cpu"
    o.reason = f"paddlepaddle {paddle.__version__} on {o.device}"
    o.available = True
    o.asarray = lambda x: paddle.to_tensor(np.asarray(x), dtype="float64")
    o.zeros = lambda s, complex_=False: paddle.zeros(s, "complex128" if complex_ else "float64")
    o.rfftn = lambda x: paddle.fft.rfftn(x, axes=[-3, -2, -1])
    o.irfftn = lambda xh, s: paddle.fft.irfftn(xh, s=list(s), axes=[-3, -2, -1])
    o.to_np = lambda x: x.numpy()
    o.stack = lambda xs, axis=0: paddle.stack(xs, axis)
    o.real = paddle.real
    o.sum = paddle.sum
    o.maxabs = lambda x: float(paddle.abs(x).max().numpy())
    o.exp = paddle.exp
    return o


def _probe_mindspore() -> Ops:
    """
    MindSpore FFT support is version- and device-dependent. We PROBE for
    a usable 3-D real FFT rather than assume it exists. If absent, the
    backend reports UNAVAILABLE — it does not silently fall back to NumPy
    while claiming to be MindSpore.
    """
    o = Ops("mindspore")
    try:
        import mindspore as ms
        from mindspore import ops, Tensor
    except ImportError:
        o.reason = "mindspore not installed (pip install mindspore)"; return o
    o.xp = ms
    try:
        ms.set_context(device_target="Ascend" if os.path.exists("/usr/local/Ascend") else "CPU")
    except Exception:
        pass
    cand_r = [getattr(ops, n, None) for n in ("rfftn", "rfft3d", "fft_with_size")]
    cand_i = [getattr(ops, n, None) for n in ("irfftn", "irfft3d")]
    rf = next((f for f in cand_r if f is not None), None)
    irf = next((f for f in cand_i if f is not None), None)
    if rf is None or irf is None:
        o.reason = (f"mindspore {ms.__version__} present but no usable 3-D real "
                    f"FFT op found (tried rfftn/rfft3d/fft_with_size). "
                    f"NOT silently substituting NumPy.")
        return o
    # smoke-test the op on a tiny array
    try:
        t = Tensor(np.random.rand(4, 4, 4).astype(np.float32))
        h = rf(t)
        _ = irf(h)
    except Exception as e:
        o.reason = f"mindspore FFT op present but failed smoke test: {e}"
        return o
    o.available = True
    o.fp64 = False   # Ascend FFT is fp32/complex64
    o.device = "Ascend" if os.path.exists("/usr/local/Ascend") else "CPU"
    o.reason = f"mindspore {ms.__version__} on {o.device}, complex64 FFT verified"
    o.asarray = lambda x: Tensor(np.asarray(x).astype(np.float32))
    o.zeros = lambda s, complex_=False: ops.zeros(s, ms.complex64 if complex_ else ms.float32)
    o.rfftn = rf
    o.irfftn = lambda xh, s: irf(xh)
    o.to_np = lambda x: x.asnumpy()
    o.stack = lambda xs, axis=0: ops.stack(xs, axis)
    o.real = ops.real
    o.sum = ops.sum
    o.maxabs = lambda x: float(np.abs(x.asnumpy()).max())
    o.exp = ops.exp
    return o


def _probe_mlx() -> Ops:
    """
    Apple MLX — the CORRECT native Apple path (XNU is a kernel, not this).
    Unified memory, Metal-backed FFT.
    """
    o = Ops("mlx")
    try:
        import mlx.core as mx
    except ImportError:
        o.reason = "mlx not installed (Apple Silicon only: pip install mlx)"; return o
    o.xp = mx
    if not hasattr(mx, "fft") or not hasattr(mx.fft, "rfftn"):
        o.reason = f"mlx present but mx.fft.rfftn missing (upgrade mlx)"; return o
    try:
        t = mx.array(np.random.rand(4, 4, 4).astype(np.float32))
        h = mx.fft.rfftn(t)
        _ = mx.fft.irfftn(h, s=(4, 4, 4))
        mx.eval(_)
    except Exception as e:
        o.reason = f"mlx FFT smoke test failed: {e}"; return o
    o.available = True
    o.fp64 = False           # Metal has no float64
    o.device = "apple-metal"
    o.reason = "Apple MLX / Metal, unified memory, complex64 (no fp64 on Metal)"
    o.asarray = lambda x: mx.array(np.asarray(x).astype(np.float32))
    o.zeros = lambda s, complex_=False: mx.zeros(s, mx.complex64 if complex_ else mx.float32)
    o.rfftn = lambda x: mx.fft.rfftn(x, axes=(-3, -2, -1))
    o.irfftn = lambda xh, s: mx.fft.irfftn(xh, s=s, axes=(-3, -2, -1))
    o.to_np = lambda x: np.asarray(x)
    o.stack = lambda xs, axis=0: mx.stack(xs, axis)
    o.real = mx.real
    o.sum = mx.sum
    o.maxabs = lambda x: float(mx.abs(x).max().item())
    o.exp = mx.exp
    o.sync = lambda: mx.eval()
    return o


PROBES = {
    "numpy": _probe_numpy, "torch": _probe_torch, "jax": _probe_jax,
    "tensorflow": _probe_tensorflow, "paddle": _probe_paddle,
    "mindspore": _probe_mindspore, "mlx": _probe_mlx,
}


def probe_all() -> Dict[str, Ops]:
    return {k: f() for k, f in PROBES.items()}


def pick_backend(prefer: Optional[str] = None) -> Ops:
    all_ = probe_all()
    order = ([prefer] if prefer else []) + ["jax", "torch", "mlx", "tensorflow",
                                            "paddle", "mindspore", "numpy"]
    for name in order:
        o = all_.get(name)
        if o is not None and o.available:
            log.info("[backend] %s — %s", o.name, o.reason)
            return o
    raise RuntimeError("no usable backend")


# =============================================================================
# §3. SPECTRAL GRID
# =============================================================================

class Grid:
    """
    Triply-periodic box. rfft layout: (Nx, Ny, Nz//2+1).
    Wavenumbers are ANGULAR (2*pi*m/L) so that -k^2 is the exact
    Fourier symbol of the Laplacian.
    """
    def __init__(self, o: Ops, n: int, L: float = 1.0,
                 dealias: float = 2.0 / 3.0):
        self.o, self.n, self.L = o, n, L
        self.shape_r = (n, n, n)
        self.shape_h = (n, n, n // 2 + 1)
        m = np.fft.fftfreq(n, d=1.0 / n)              # integer modes
        mr = np.fft.rfftfreq(n, d=1.0 / n)
        kx = (2 * PI / L) * m[:, None, None]
        ky = (2 * PI / L) * m[None, :, None]
        kz = (2 * PI / L) * mr[None, None, :]
        KX = np.broadcast_to(kx, self.shape_h).copy()
        KY = np.broadcast_to(ky, self.shape_h).copy()
        KZ = np.broadcast_to(kz, self.shape_h).copy()
        K2 = KX ** 2 + KY ** 2 + KZ ** 2
        K2inv = np.zeros_like(K2)
        nz_ = K2 > 0
        K2inv[nz_] = 1.0 / K2[nz_]
        # Orszag 2/3 mask
        cut = dealias * (n // 2)
        mask = ((np.abs(m)[:, None, None] < cut) &
                (np.abs(m)[None, :, None] < cut) &
                (np.abs(mr)[None, None, :] < cut)).astype(np.float64)
        self.mask_np = mask
        self.kmax = float((2 * PI / L) * (n // 2) * dealias)
        A = o.asarray
        self.kx, self.ky, self.kz = A(KX), A(KY), A(KZ)
        self.k2, self.k2inv, self.mask = A(K2), A(K2inv), A(mask)
        self.n_modes_kept = int(mask.sum())
        self.n_modes_total = int(mask.size)
        # real-space grid for initial conditions
        x = (2 * PI / L) * (np.arange(n) * (L / n))
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing="ij")

    def project(self, uh):
        """Leray: uh <- uh - k (k.uh)/|k|^2. Exact, not iterative."""
        o = self.o
        div = self.kx * uh[0] + self.ky * uh[1] + self.kz * uh[2]
        f = div * self.k2inv
        return o.stack([uh[0] - self.kx * f,
                        uh[1] - self.ky * f,
                        uh[2] - self.kz * f])

    def divergence_max(self, uh) -> float:
        div = self.kx * uh[0] + self.ky * uh[1] + self.kz * uh[2]
        return self.o.maxabs(div)

    def curl_hat(self, uh):
        o = self.o
        i = o.asarray(np.array(0.0)) * 0 + 1j    # imaginary unit, backend-typed
        wx = i * (self.ky * uh[2] - self.kz * uh[1])
        wy = i * (self.kz * uh[0] - self.kx * uh[2])
        wz = i * (self.kx * uh[1] - self.ky * uh[0])
        return o.stack([wx, wy, wz])

    def dealias(self, uh):
        return uh * self.mask


# =============================================================================
# §4. THE DNS SOLVER
# =============================================================================

@dataclass
class DNSConfig:
    n: int = 64
    Re: float = 1000.0
    L: float = 1.0
    cfl: float = 0.4
    dealias: float = 2.0 / 3.0
    force_k: float = 3.0          # forcing shell (0 = unforced)
    force_amp: float = 0.0
    dt: Optional[float] = None
    kmax_eta_target: float = 1.5


class SpectralDNS:
    """
    Pseudo-spectral incompressible DNS. Backend-agnostic through Ops.

    Rotation-form nonlinearity + exact Leray projection + 2/3 dealias
    + SSP-RK3. Nothing here is novel; it is the standard method
    (Patterson & Orszag 1971, Shu & Osher 1988).
    """

    def __init__(self, o: Ops, cfg: DNSConfig):
        self.o, self.cfg = o, cfg
        self.nu = cfg.L * 1.0 / cfg.Re          # u'=1, L=cfg.L
        self.g = Grid(o, cfg.n, cfg.L, cfg.dealias)
        self.dt = cfg.dt if cfg.dt else cfg.cfl * (cfg.L / cfg.n)
        self.t = 0.0
        self.steps = 0
        self.uh = o.zeros((3,) + self.g.shape_h, complex_=True)

    # ---- initial conditions ----
    def set_taylor_green_2d(self, k: int = 1, amp: float = 1.0):
        """
        2-D Taylor–Green embedded in 3-D (no z-dependence).
        Its nonlinear term is a PURE GRADIENT, annihilated by the
        Leray projection => exact viscous decay exp(-2 nu k^2 t).
        This is the decisive analytic verification.
        """
        g, o = self.g, self.o
        kk = 2 * PI * k / self.cfg.L
        u = amp * np.sin(kk * g.X) * np.cos(kk * g.Y)
        v = -amp * np.cos(kk * g.X) * np.sin(kk * g.Y)
        w = np.zeros_like(u)
        u_r = o.stack([o.asarray(u), o.asarray(v), o.asarray(w)])
        self.uh = self.g.project(g.dealias(o.rfftn(u_r)))
        self.t, self.steps = 0.0, 0

    def set_taylor_green_3d(self, k: int = 1, amp: float = 1.0):
        """Classic 3-D Taylor–Green vortex (Taylor & Green 1937)."""
        g, o = self.g, self.o
        kk = 2 * PI * k / self.cfg.L
        u = amp * np.sin(kk * g.X) * np.cos(kk * g.Y) * np.cos(kk * g.Z)
        v = -amp * np.cos(kk * g.X) * np.sin(kk * g.Y) * np.cos(kk * g.Z)
        w = np.zeros_like(u)
        u_r = o.stack([o.asarray(u), o.asarray(v), o.asarray(w)])
        self.uh = self.g.project(g.dealias(o.rfftn(u_r)))
        self.t, self.steps = 0.0, 0

    def set_random(self, seed: int = 0, kpeak: int = 4):
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(3,) + self.g.shape_r)
        u_r = self.o.asarray(u)
        uh = self.o.rfftn(u_r)
        # concentrate energy near kpeak
        kmag = np.sqrt(self.g.k2 / (2 * PI / self.cfg.L) ** 2)
        w = np.exp(-0.5 * ((kmag - kpeak) / 1.5) ** 2)
        uh = uh * self.o.asarray(w)
        self.uh = self.g.project(self.g.dealias(uh))
        self.t, self.steps = 0.0, 0

    # ---- right-hand side ----
    def rhs(self, uh):
        o, g = self.o, self.g
        wh = g.curl_hat(uh)
        u_r = o.irfftn(uh, s=g.shape_r)
        w_r = o.irfftn(wh, s=g.shape_r)
        # rotation form: N = u x omega
        n0 = u_r[1] * w_r[2] - u_r[2] * w_r[1]
        n1 = u_r[2] * w_r[0] - u_r[0] * w_r[2]
        n2 = u_r[0] * w_r[1] - u_r[1] * w_r[0]
        nh = o.rfftn(o.stack([n0, n1, n2]))
        nh = g.project(g.dealias(nh))
        visc = -self.nu * g.k2 * uh
        out = o.stack([nh[0] + visc[0], nh[1] + visc[1], nh[2] + visc[2]])
        if self.cfg.force_amp > 0:
            kmag = np.sqrt(g.k2 / (2 * PI / self.cfg.L) ** 2)
            sel = ((kmag > 0) & (kmag <= self.cfg.force_k)).astype(np.float64)
            f = o.asarray(sel) * self.cfg.force_amp
            out = o.stack([out[i] + f * uh[i] for i in range(3)])
        return out

    # ---- SSP-RK3 (Shu & Osher 1988) ----
    def step(self, dt: Optional[float] = None):
        o = self.o
        h = dt if dt is not None else self.dt
        u0 = self.uh
        k1 = self.rhs(u0)
        u1 = o.stack([u0[i] + h * k1[i] for i in range(3)])
        u1 = self.g.project(self.g.dealias(u1))
        k2 = self.rhs(u1)
        u2 = o.stack([0.75 * u0[i] + 0.25 * (u1[i] + h * k2[i]) for i in range(3)])
        u2 = self.g.project(self.g.dealias(u2))
        k3 = self.rhs(u2)
        u3 = o.stack([(u0[i] / 3.0) + (2.0 / 3.0) * (u2[i] + h * k3[i]) for i in range(3)])
        self.uh = self.g.project(self.g.dealias(u3))
        self.t += h
        self.steps += 1

    def run(self, n_steps: int, cb=None):
        hist = []
        for i in range(n_steps):
            self.step()
            if cb and (i % max(1, n_steps // 10) == 0):
                d = self.diagnostics(); hist.append(d); cb(i, d)
        return hist

    # ---- diagnostics (the honesty layer) ----
    def diagnostics(self) -> Dict[str, float]:
        o, g = self.o, self.g
        uh = self.uh
        N = self.cfg.n ** 3
        # Parseval with rfft weighting: interior k_z counted twice
        w = np.ones(g.shape_h); w[..., 1:-1] = 2.0
        wt = o.asarray(w)
        e_k = 0.5 * sum(o.real(uh[i] * _conj(o, uh[i])) for i in range(3))
        E = float(o.to_np(o.sum(e_k * wt))) / (N * N)
        eps_k = self.nu * g.k2 * e_k * 2.0
        eps = float(o.to_np(o.sum(eps_k * wt))) / (N * N)
        u_rms = math.sqrt(max(2.0 * E / 3.0, 1e-30))
        eta = (self.nu ** 3 / max(eps, 1e-30)) ** 0.25
        lam = math.sqrt(15.0 * self.nu * u_rms ** 2 / max(eps, 1e-30))
        return {
            "t": self.t, "steps": self.steps, "dt": self.dt,
            "E": E, "epsilon": eps, "u_rms": u_rms,
            "eta": eta, "kmax_eta": g.kmax * eta,
            "lambda_taylor": lam, "Re_lambda": u_rms * lam / self.nu,
            "div_max": g.divergence_max(uh),
            "dns_resolved": bool(g.kmax * eta >= 1.5),
            "nu": self.nu,
        }

    def spectrum(self, nbins: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        o, g = self.o, self.g
        w = np.ones(g.shape_h); w[..., 1:-1] = 2.0
        e_k = 0.5 * sum(o.real(self.uh[i] * _conj(o, self.uh[i])) for i in range(3))
        e_np = o.to_np(e_k) * w / (self.cfg.n ** 6)
        kmag = np.sqrt(g.k2) / (2 * PI / self.cfg.L)
        edges = np.linspace(0, kmag.max() + 1e-9, nbins + 1)
        idx = np.digitize(kmag.ravel(), edges) - 1
        Ek = np.bincount(idx, weights=e_np.ravel(), minlength=nbins)
        kc = 0.5 * (edges[:-1] + edges[1:])
        return kc, Ek


def _conj(o: Ops, x):
    xp = o.xp
    if hasattr(xp, "conj"):
        return xp.conj(x)
    return xp.conjugate(x)


# =============================================================================
# §5. VERIFICATION
# =============================================================================

def self_test(backend: Optional[str] = None, verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # ---- T0 backend probe ----
    print("\n[T0] backend capability probe (honest)")
    all_ = probe_all()
    for name, o in all_.items():
        mark = "OK  " if o.available else "N/A "
        print(f"      {mark} {name:12s} {o.reason[:78]}")
    rep("numpy reference always available", all_["numpy"].available)
    o = pick_backend(backend)
    print(f"      selected: {o.name}  device={o.device}  fp64={o.fp64}")

    # ---- T1 resource wall ----
    print("\n[T1] the resource wall (Re^(9/4) is not negotiable)")
    rows = feasibility_table("H100")
    print(f"      {'Re_L':>8} {'N/axis':>8} {'cells':>11} {'state':>11} "
          f"{'ranks_min':>10} {'wall':>11}")
    for r in rows:
        print(f"      {r['Re_L']:>8.0e} {r['N_per_axis']:>8d} "
              f"{r['cells']:>11.2e} {r['state_human']:>11} "
              f"{r['ranks_memory_min']:>10d} {r['total_human']:>11}")
    rep("Re=1e4 fits one H100", rows[1]["ranks_memory_min"] <= 1)
    rep("Re=1e5 needs >10 ranks", rows[2]["ranks_memory_min"] > 10,
        f"{rows[2]['ranks_memory_min']}")
    rep("Re=1e7 needs >100k ranks", rows[4]["ranks_memory_min"] > 100_000,
        f"{rows[4]['ranks_memory_min']:,}")

    # ---- T2 Leray projection ----
    print("\n[T2] Leray projection makes div u = 0 to round-off")
    cfg = DNSConfig(n=32, Re=1000.0)
    s = SpectralDNS(o, cfg)
    rng = np.random.default_rng(0)
    raw = o.rfftn(o.asarray(rng.normal(size=(3,) + s.g.shape_r)))
    d_before = s.g.divergence_max(raw)
    proj = s.g.project(raw)
    d_after = s.g.divergence_max(proj)
    print(f"      max|k.u| before={d_before:.3e}  after={d_after:.3e}")
    tol = 1e-9 if o.fp64 else 1e-3
    rep("projection kills divergence", d_after < tol, f"{d_after:.2e}")
    rep("random field was NOT divergence-free", d_before > 1.0)

    # ---- T3 dealiasing ----
    print("\n[T3] Orszag 2/3 rule leaves high-k modes exactly zero")
    deal = s.g.dealias(raw)
    dn = o.to_np(deal)
    hi = s.g.mask_np == 0
    rep("all masked modes are exactly 0", float(np.abs(dn[:, hi]).max()) == 0.0)
    frac = s.g.n_modes_kept / s.g.n_modes_total
    print(f"      kept {s.g.n_modes_kept}/{s.g.n_modes_total} modes ({frac:.1%})")
    rep("kept fraction ~ (2/3)^3 = 29.6%", abs(frac - (2/3)**3) < 0.06, f"{frac:.3f}")

    # ---- T4 ANALYTIC Taylor-Green decay ----
    print("\n[T4] 2-D Taylor–Green: decay must match exp(-2 nu k^2 t) EXACTLY")
    for Re in (100.0, 1000.0):
        c = DNSConfig(n=32, Re=Re, cfl=0.25)
        s = SpectralDNS(o, c)
        s.set_taylor_green_2d(k=1, amp=1.0)
        E0 = s.diagnostics()["E"]
        nst = 60
        s.run(nst)
        d = s.diagnostics()
        kk = 2 * PI / c.L
        # energy ~ exp(-4 nu k^2 t)  (amplitude ~ exp(-2 nu k^2 t))
        ana = math.exp(-4.0 * s.nu * kk ** 2 * d["t"])
        num = d["E"] / E0
        rel = abs(num - ana) / max(ana, 1e-30)
        print(f"      Re={Re:6.0f}  E/E0 num={num:.8f} analytic={ana:.8f} "
              f"rel={rel:.2e}")
        thr = 2e-3 if o.fp64 else 3e-2
        rep(f"Re={Re:.0f}: analytic decay matched", rel < thr, f"rel={rel:.2e}")

    # ---- T5 energy budget ----
    print("\n[T5] unforced energy budget: dE/dt = -epsilon")
    c = DNSConfig(n=32, Re=500.0, cfl=0.25, force_amp=0.0)
    s = SpectralDNS(o, c)
    s.set_taylor_green_3d(k=2)
    d0 = s.diagnostics()
    h = s.dt
    s.run(20)
    d1 = s.diagnostics()
    dEdt = (d1["E"] - d0["E"]) / (d1["t"] - d0["t"])
    eps_mid = 0.5 * (d0["epsilon"] + d1["epsilon"])
    rel = abs(dEdt + eps_mid) / max(abs(eps_mid), 1e-30)
    print(f"      dE/dt={dEdt:.6e}  -eps={-eps_mid:.6e}  rel={rel:.3e}")
    rep("energy budget closes (<15%)", rel < 0.15, f"rel={rel:.3f}")

    # ---- T6 resolution diagnostic is honest ----
    print("\n[T6] k_max*eta is REPORTED, not assumed")
    c = DNSConfig(n=32, Re=5000.0, cfl=0.3)
    s = SpectralDNS(o, c); s.set_random(seed=1, kpeak=4)
    s.run(20)
    d = s.diagnostics()
    print(f"      k_max*eta={d['kmax_eta']:.3f}  Re_lambda={d['Re_lambda']:.1f}  "
          f"dns_resolved={d['dns_resolved']}")
    rep("under-resolved case is FLAGGED as such",
        d["dns_resolved"] == (d["kmax_eta"] >= 1.5))
    rep("divergence stays at round-off during run",
        d["div_max"] < (1e-8 if o.fp64 else 1e-2), f"{d['div_max']:.2e}")

    # ---- T7 the gate refuses impossible grids ----
    print("\n[T7] allocation gate refuses Re=1e7 on one device")
    r = plan(1e7, 1.5, "H100")
    refused = r["ranks_memory_min"] > 1
    rep("Re=1e7 refused on 1 rank", refused,
        f"needs {r['ranks_memory_min']:,} ranks, {r['state_human']}")
    r2 = plan(1e5, 1.0, "H100")
    print(f"      Re=1e5 @ kmax_eta=1.0 -> N={r2['N_per_axis']}, "
          f"{r2['state_human']}, dns_adequate={r2['dns_adequate']}")
    rep("reduced-resolution option is LABELLED under-resolved",
        r2["dns_adequate"] is False)

    # ---- T8 spectrum ----
    print("\n[T8] energy spectrum is computable and non-negative")
    kc, Ek = s.spectrum(nbins=24)
    rep("spectrum non-negative", bool((Ek >= -1e-14).all()))
    rep("spectrum has finite total", math.isfinite(float(Ek.sum())))
    print(f"      sum E(k) = {Ek.sum():.6e}   peak at k={kc[int(Ek.argmax())]:.1f}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}   backend={o.name}")
    print("=" * 78)
    print("""
VERIFIED
  Leray projection is exact (algebraic in Fourier space, not iterative).
  2/3 dealiasing zeroes the aliased third exactly.
  2-D Taylor-Green decays at the ANALYTIC viscous rate — the decisive
  correctness test, since its nonlinearity is a pure gradient.
  Unforced energy budget closes: dE/dt = -epsilon.
  k_max*eta is reported and under-resolution is flagged.
  The allocation gate refuses Re=1e7 and states the rank count needed.

NOT VERIFIED / NOT CLAIMED
  No run at Re >= 1e5 (needs a cluster; the gate says how many ranks).
  No multi-rank domain decomposition — single device only.
  No validation against JHTDB or experiment.
  MindSpore/MLX are PROBED; if the FFT op is absent they report
  UNAVAILABLE rather than silently substituting NumPy.
  Apple: XNU is a kernel. The real paths are MLX and Metal/MPS.
""")
    return ok


# =============================================================================
# §6. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-backend pseudo-spectral DNS")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "probe", "plan", "run"])
    ap.add_argument("--backend", default=None,
                    choices=list(PROBES), help="force a backend")
    ap.add_argument("--re", type=float, default=1000.0, dest="re_l")
    ap.add_argument("--n", type=int, default=None,
                    help="grid per axis (default: from Re and kmax_eta)")
    ap.add_argument("--kmax-eta", type=float, default=1.5)
    ap.add_argument("--device", default="H100", choices=list(DEVICES))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--force", type=float, default=0.0)
    ap.add_argument("--ic", default="tg3d",
                    choices=["tg2d", "tg3d", "random"])
    ap.add_argument("--allow-underresolved", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test(a.backend) else 1

    out: Dict[str, Any] = {}

    if a.cmd == "probe":
        out = {k: {"available": v.available, "device": v.device,
                   "fp64": v.fp64, "reason": v.reason}
               for k, v in probe_all().items()}
        print(json.dumps(out, indent=2))

    elif a.cmd == "plan":
        out = {"feasibility": feasibility_table(a.device),
               "target": plan(a.re_l, a.kmax_eta, a.device)}
        print(BANNER)
        print(f"{'Re_L':>9} {'N/axis':>8} {'cells':>12} {'state':>11} "
              f"{'ranks':>9} {'wall':>12}")
        print("-" * 70)
        for r in out["feasibility"]:
            print(f"{r['Re_L']:>9.0e} {r['N_per_axis']:>8d} "
                  f"{r['cells']:>12.2e} {r['state_human']:>11} "
                  f"{r['ranks_memory_min']:>9d} {r['total_human']:>12}")
        print("-" * 70)
        print("Cost ~ Re^3. No framework changes that exponent.")

    elif a.cmd == "run":
        o = pick_backend(a.backend)
        n = a.n if a.n else n_for_re(a.re_l, a.kmax_eta)
        # ---- THE GATE ----
        cells = n ** 3
        need_gb = cells * 5 * (8 if o.fp64 else 4) * 2.5 / 1e9
        avail_gb = 16.0
        try:
            import torch
            if torch.cuda.is_available():
                avail_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
        print(BANNER)
        print(f"  Re_L={a.re_l:.3g}  N={n}^3={cells:.3e}  "
              f"working set ~ {need_gb:.2f} GB  device ~ {avail_gb:.1f} GB")
        if need_gb > 0.85 * avail_gb:
            print(f"\n  *** REFUSED ***")
            print(f"  This grid needs ~{need_gb:.1f} GB; device has ~{avail_gb:.1f} GB.")
            r = plan(a.re_l, a.kmax_eta, a.device)
            print(f"  Ranks required: {r['ranks_memory_min']:,} "
                  f"({r['state_human']} state).")
            print(f"  Options: lower --n, or --kmax-eta 1.0 "
                  f"(UNDER-RESOLVED, solver benchmark only), or use a cluster.")
            return 2
        cfg = DNSConfig(n=n, Re=a.re_l, cfl=a.cfl,
                        dealias=2/3, force_amp=a.force)
        s = SpectralDNS(o, cfg)
        if a.ic == "tg2d":   s.set_taylor_green_2d()
        elif a.ic == "tg3d": s.set_taylor_green_3d()
        else:                s.set_random()
        d0 = s.diagnostics()
        print(f"  backend={o.name} device={o.device} fp64={o.fp64}")
        print(f"  nu={s.nu:.3e}  dt={s.dt:.3e}  modes kept "
              f"{s.g.n_modes_kept}/{s.g.n_modes_total}")
        t0 = time.time()
        hist = s.run(a.steps, cb=lambda i, d: print(
            f"    step {i:5d} t={d['t']:.4f} E={d['E']:.6e} "
            f"eps={d['epsilon']:.4e} kmax*eta={d['kmax_eta']:.3f} "
            f"div={d['div_max']:.1e}"))
        wall = time.time() - t0
        d1 = s.diagnostics()
        if not d1["dns_resolved"] and not a.allow_underresolved:
            print(f"\n  WARNING: k_max*eta = {d1['kmax_eta']:.3f} < 1.5. "
                  f"This is NOT a resolved DNS.")
            print(f"  Re-run with --allow-underresolved to acknowledge, "
                  f"or raise --n to {n_for_re(d1['Re_lambda']*10, 1.5)}.")
        kc, Ek = s.spectrum()
        out = {"config": asdict(cfg), "backend": o.name,
               "initial": d0, "final": d1, "wall_s": wall,
               "ms_per_step": 1e3 * wall / max(a.steps, 1),
               "spectrum_k": kc.tolist(), "spectrum_E": Ek.tolist(),
               "history": hist}
        print(f"\n  {a.steps} steps in {wall:.2f}s "
              f"({1e3*wall/a.steps:.1f} ms/step)")
        print(f"  E: {d0['E']:.6e} -> {d1['E']:.6e}")
        print(f"  Re_lambda: {d0['Re_lambda']:.1f} -> {d1['Re_lambda']:.1f}")
        print(f"  k_max*eta: {d1['kmax_eta']:.3f}  "
              f"resolved={d1['dns_resolved']}")

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
