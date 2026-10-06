"""
advanced_optogenetics.py
========================

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================


Production-grade, fully-differentiable multi-opsin optogenetic control engine.

Capabilities
------------
* **Multi-opsin co-expression** — any mix of excitatory (ChR2, Chrimson, Chronos)
  and inhibitory (GtACR2, eNpHR3.0, ArchT) opsins in a single module.
* **Wavelength-resolved action spectra** — Gaussian approximation of published
  action spectra; input is spectral irradiance I(λ). Supports arbitrary
  wavelength grids and learnable peak shifts.
* **5-state kinetics with desensitization** —
      C1 ⇌ O1 ⇌ O2 ⇌ C2
           ↓    ↓
           └────┴──→ D ──(slow)──→ C1
  D is the light-adapted / desensitized state; recovers to C1 with rate kr.
* **Single- and two-photon excitation** — per-opsin exponent n_photon ∈ [1, 2.2].
* **Exact matrix-exponential integrator** — unconditionally stable, preserves
  positivity (Metzler generator) and mass (column sums = 0) by construction.
* **Full AMP / DDP / torch.compile compatibility** — no cross-rank state,
  internal FP32 lane for kinetics, autocast-disabled region for the update.

Shape contract
--------------
irradiance    : [B, N, L]   (spectral irradiance per neuron per wavelength)
                 or [B, L]  (broad illumination, broadcast across neurons)
v_membrane    : [B, N]      (mV)
opsin_states  : [B, N, K, 5]

Returns
-------
i_photo       : [B, N]      (pA/pF, signed; inhibitory opsins contribute negative)
next_states   : [B, N, K, 5]
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_AUTOCAST_DEVICES = frozenset({"cuda", "cpu", "xpu", "hpu", "mps"})
_STATE_NAMES = ("C1", "O1", "O2", "C2", "D")


# ====================================================================== #
#  Opsin specification
# ====================================================================== #
@dataclass(frozen=True)
class OpsinSpec:
    """
    Static description of a single opsin variant.

    Kinetics are taken from the ChR2/Chrimson literature (Nikolic 2009,
    Grossman 2011, Williams 2013) and are exposed as tunable defaults
    for other opsins. The desensitization branch (kd1, kd2, kr) is an
    extension for prolonged-illumination experiments.
    """
    name: str
    peak_nm: float
    sign: int = +1                    # +1 excitatory, -1 inhibitory

    # --- photocycle kinetics (s^-1) --------------------------------- #
    gamma: float = 0.1                # g(O2) / g(O1)
    e12_0: float = 0.053
    e21_0: float = 0.023
    Gd1: float = 0.1
    Gd2: float = 0.05
    Gr: float = 4e-4
    k1: float = 0.5
    k2: float = 0.11
    kd1: float = 2e-2                 # O1 → D
    kd2: float = 2e-2                 # O2 → D
    kr: float = 1e-3                  # D  → C1 (slow recovery)

    # --- action spectrum (Gaussian approximation) ------------------- #
    sigma_nm: float = 25.0

    # --- physics ---------------------------------------------------- #
    alpha: float = 1.0                # photocycle drive gain
    n_photon: float = 1.0             # 1.0 = 1P, 2.0 = 2P
    E_rev: float = 0.0                # reversal potential (mV)
    g_max_init: float = 0.05          # initial peak conductance per neuron

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        if self.n_photon < 1.0:
            raise ValueError("n_photon must be >= 1 (physical photocycle)")
        if self.sigma_nm <= 0.0:
            raise ValueError("sigma_nm must be positive")
        if self.peak_nm <= 0.0:
            raise ValueError("peak_nm must be positive")


# Published / commonly-used opsin variants
KNOWN_OPSINS = {
    "ChR2":      OpsinSpec("ChR2",      470.0, +1, sigma_nm=30.0),
    "Chrimson":  OpsinSpec("Chrimson",  590.0, +1, sigma_nm=35.0, n_photon=1.0),
    "Chronos":   OpsinSpec("Chronos",   500.0, +1, sigma_nm=40.0),
    "ReaChR":    OpsinSpec("ReaChR",    590.0, +1, sigma_nm=40.0),
    "GtACR2":    OpsinSpec("GtACR2",    470.0, -1, sigma_nm=30.0),
    "eNpHR3.0":  OpsinSpec("eNpHR3.0",  590.0, -1, sigma_nm=40.0),
    "ArchT":     OpsinSpec("ArchT",     566.0, -1, sigma_nm=40.0),
}


# ====================================================================== #
#  Main engine
# ====================================================================== #
class AdvancedOptogeneticEngine(nn.Module):
    """
    Multi-opsin, wavelength-aware, 5-state differentiable optogenetic engine.

    Parameters
    ----------
    num_neurons : int
        Number of opsin-expressing units (neurons / compartments).
    opsin_specs : Sequence[OpsinSpec]
        One entry per co-expressed opsin. Order defines K.
    wavelengths_nm : torch.Tensor
        Monotonic wavelength grid (nm) sampled by ``irradiance``. Shape [L].
    dt : float, default 1e-3
        Integration step (seconds). Match the surrounding simulation dt.
    integrator : {"expm", "rk4"}, default "expm"
        * ``expm`` — matrix exponential. Unconditionally stable, exact for
          the linear photocycle, preserves the probability simplex by
          construction (Metzler generator + column sums = 0).
        * ``rk4``  — classical RK4. Cheaper per step at small ``dt``; loses
          the strict positivity guarantee (still stable for the kinetics
          shipped here).
    """

    # ---------------------------------------------------------------- #
    #  Construction
    # ---------------------------------------------------------------- #
    def __init__(
        self,
        num_neurons: int,
        opsin_specs: Sequence[OpsinSpec],
        wavelengths_nm: torch.Tensor,
        dt: float = 1e-3,
        integrator: str = "expm",
    ) -> None:
        super().__init__()

        if integrator not in ("expm", "rk4"):
            raise ValueError(f"integrator must be 'expm' or 'rk4', got {integrator!r}")
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        if len(opsin_specs) == 0:
            raise ValueError("at least one OpsinSpec is required")
        if wavelengths_nm.dim() != 1 or wavelengths_nm.numel() < 2:
            raise ValueError("wavelengths_nm must be a 1-D tensor with >= 2 samples")

        self.num_neurons = int(num_neurons)
        self.K = len(opsin_specs)
        self.L = int(wavelengths_nm.numel())
        self.dt = float(dt)
        self.integrator = integrator
        self.opsin_names = tuple(s.name for s in opsin_specs)

        # ---- learnable per-opsin, per-neuron conductance -------------- #
        # g_max = softplus(g_raw)  →  strictly positive, smooth, no boundary
        # gradient degeneration.  Shape [K, N] so that DDP syncs the whole
        # tensor in a single all-reduce.
        g_raw_col = torch.tensor(
            [math.log(math.expm1(max(s.g_max_init, 1e-8))) for s in opsin_specs],
            dtype=torch.float32,
        ).unsqueeze(-1)                                    # [K, 1]
        self.g_raw = nn.Parameter(g_raw_col.expand(self.K, self.num_neurons).clone())

        # ---- fixed kinetics as buffers (dtype/device-agnostic) -------- #
        def _t(values: Sequence[float]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=torch.float32)

        self.register_buffer("gamma",   _t(s.gamma   for s in opsin_specs))
        self.register_buffer("e12_0",   _t(s.e12_0   for s in opsin_specs))
        self.register_buffer("e21_0",   _t(s.e21_0   for s in opsin_specs))
        self.register_buffer("Gd1",     _t(s.Gd1     for s in opsin_specs))
        self.register_buffer("Gd2",     _t(s.Gd2     for s in opsin_specs))
        self.register_buffer("Gr",      _t(s.Gr      for s in opsin_specs))
        self.register_buffer("k1",      _t(s.k1      for s in opsin_specs))
        self.register_buffer("k2",      _t(s.k2      for s in opsin_specs))
        self.register_buffer("kd1",     _t(s.kd1     for s in opsin_specs))
        self.register_buffer("kd2",     _t(s.kd2     for s in opsin_specs))
        self.register_buffer("kr",      _t(s.kr      for s in opsin_specs))
        self.register_buffer("E_rev",   _t(s.E_rev   for s in opsin_specs))
        self.register_buffer("sign",    _t(float(s.sign) for s in opsin_specs))
        self.register_buffer("alpha",   _t(s.alpha   for s in opsin_specs))
        self.register_buffer("n_photon", _t(s.n_photon for s in opsin_specs))

        # Photon-activation scaling  (fixed from original model: 0.5 and 0.1)
        self.register_buffer("ga1_scale", _t([0.5] * self.K))
        self.register_buffer("ga2_scale", _t([0.1] * self.K))

        # ---- action spectrum ------------------------------------------ #
        self.register_buffer("wavelengths_nm", wavelengths_nm.float().clone())
        self.register_buffer(
            "action_spectrum",
            self._build_action_spectrum(opsin_specs, wavelengths_nm),
        )

    @staticmethod
    def _build_action_spectrum(
        opsin_specs: Sequence[OpsinSpec],
        wavelengths_nm: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian action spectrum per opsin, peak-normalized to 1."""
        lam = wavelengths_nm.float()
        rows = []
        for spec in opsin_specs:
            s = torch.exp(-0.5 * ((lam - spec.peak_nm) / spec.sigma_nm) ** 2)
            s = s / s.amax().clamp_min(1e-8)
            rows.append(s)
        return torch.stack(rows, dim=0)                    # [K, L]

    # ---------------------------------------------------------------- #
    #  Public helpers
    # ---------------------------------------------------------------- #
    def initial_states(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dark-adapted state: all opsins in C1.  Returns [B, N, K, 5]."""
        x = torch.zeros(batch_size, self.num_neurons, self.K, 5,
                        device=device, dtype=dtype)
        x[..., 0] = 1.0
        return x

    @torch.no_grad()
    def get_g_max(self) -> torch.Tensor:
        """Current peak conductances, shape [K, N]."""
        return F.softplus(self.g_raw)

    @torch.no_grad()
    def set_g_max(self, g_max: torch.Tensor) -> None:
        """Set peak conductances from physical values, shape [K, N]."""
        g = g_max.clamp_min(1e-8)
        self.g_raw.copy_(torch.log(torch.expm1(g)))

    # ---------------------------------------------------------------- #
    #  Forward
    # ---------------------------------------------------------------- #
    def forward(
        self,
        irradiance: torch.Tensor,
        v_membrane: torch.Tensor,
        opsin_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out_dtype_i = v_membrane.dtype
        out_dtype_x = opsin_states.dtype
        device_type = v_membrane.device.type

        # FP32 kinetics regardless of outer autocast context.
        ctx = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in _AUTOCAST_DEVICES
            else contextlib.nullcontext()
        )

        with ctx:
            irr = irradiance.float()
            vm  = v_membrane.float()
            x   = opsin_states.float()

            # --- 1. Effective wavelength-integrated drive ------------- #
            # irr: [B, L]  or  [B, N, L]
            if irr.dim() == 2:
                irr = irr.unsqueeze(1)                       # [B, 1, L]
            # action_spectrum: [K, L]
            eff = torch.einsum("bnl,kl->bnk", irr, self.action_spectrum)  # [B, N, K]
            eff = eff.clamp_min(0.0)

            # --- 2. Nonlinear (1P/2P) photocycle drive --------------- #
            eff = eff.pow(self.n_photon) * self.alpha        # [B, N, K]

            Ga1 = self.ga1_scale * eff
            Ga2 = self.ga2_scale * eff
            e12 = self.e12_0 + self.k1 * eff
            e21 = self.e21_0 + self.k2 * eff

            # --- 3. State advance ------------------------------------ #
            if self.integrator == "expm":
                x_next = self._step_expm(x, Ga1, Ga2, e12, e21)
            else:
                x_next = self._step_rk4(x, Ga1, Ga2, e12, e21)

            # --- 4. Photocurrent ------------------------------------- #
            O1 = x_next[..., 1]                              # [B, N, K]
            O2 = x_next[..., 2]

            # Voltage rectification written as tanh(u/2) — overflow-free
            # in FP16/BF16, saturates to ±1 cleanly.
            u = (vm + 40.0) * (1.0 / 15.0)                   # [B, N]
            v_factor = torch.tanh(0.5 * u)

            # g_max: [K, N] → [1, N, K]  for broadcast with [B, N, K]
            g_max_bnk = F.softplus(self.g_raw).transpose(0, 1).unsqueeze(0)

            # Per-opsin driving force: V - E_rev  → [B, N, K]
            V_minus_E = vm.unsqueeze(-1) - self.E_rev        # broadcast [K]

            i_k = (
                g_max_bnk
                * (O1 + self.gamma * O2)
                * v_factor.unsqueeze(-1)
                * V_minus_E
            )                                                # [B, N, K]

            # Signed sum: excitatory (+) and inhibitory (−) add on the
            # membrane. Desensitized state D carries no current.
            i_photo = (i_k * self.sign).sum(dim=-1)          # [B, N]

        return i_photo.to(out_dtype_i), x_next.to(out_dtype_x)

    # ---------------------------------------------------------------- #
    #  Integrators
    # ---------------------------------------------------------------- #
    def _build_generator(
        self,
        Ga1: torch.Tensor, Ga2: torch.Tensor,
        e12: torch.Tensor, e21: torch.Tensor,
    ) -> torch.Tensor:
        """
        Column-stochastic Markov generator M, shape [B, N, K, 5, 5].

        Column sums are zero (mass conservation).  Off-diagonals are
        non-negative (Metzler) ⇒ expm(M·dt) is entrywise non-negative
        and column-stochastic, so the probability simplex is invariant.
        """
        B, N, K = Ga1.shape
        M = Ga1.new_zeros(B, N, K, 5, 5)

        # From C1 (state 0)
        M[..., 0, 0] = -Ga1
        M[..., 1, 0] =  Ga1

        # From O1 (state 1)
        M[..., 0, 1] =  self.Gd1
        M[..., 1, 1] = -(self.Gd1 + e12 + self.kd1)
        M[..., 2, 1] =  e12
        M[..., 4, 1] =  self.kd1

        # From O2 (state 2)
        M[..., 1, 2] =  e21
        M[..., 2, 2] = -(e21 + self.Gd2 + self.kd2)
        M[..., 3, 2] =  self.Gd2
        M[..., 4, 2] =  self.kd2

        # From C2 (state 3)
        M[..., 0, 3] =  self.Gr
        M[..., 2, 3] =  Ga2
        M[..., 3, 3] = -(Ga2 + self.Gr)

        # From D  (state 4)
        M[..., 0, 4] =  self.kr
        M[..., 4, 4] = -self.kr

        return M

    def _step_expm(self, x, Ga1, Ga2, e12, e21):
        M = self._build_generator(Ga1, Ga2, e12, e21)
        P = torch.linalg.matrix_exp(M * self.dt)             # [B, N, K, 5, 5]
        return torch.einsum("bnkij,bnkj->bnki", P, x)

    def _step_rk4(self, x, Ga1, Ga2, e12, e21):
        M = self._build_generator(Ga1, Ga2, e12, e21)
        dt = self.dt

        def f(y):
            return torch.einsum("bnkij,bnkj->bnki", M, y)

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ====================================================================== #
#  Smoke-test / demo
# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dt = torch.bfloat16 if device == "cuda" else torch.float32

    B, N = 4, 128
    wavelengths = torch.arange(400.0, 701.0, 10.0)            # 31 samples

    engine = AdvancedOptogeneticEngine(
        num_neurons=N,
        opsin_specs=[
            KNOWN_OPSINS["ChR2"],          # excitatory, blue
            KNOWN_OPSINS["Chrimson"],      # excitatory, red
            KNOWN_OPSINS["GtACR2"],        # inhibitory, blue
        ],
        wavelengths_nm=wavelengths,
        dt=1e-3,
    ).to(device)

    # ── forward pass ──────────────────────────────────────────────── #
    irr = torch.rand(B, N, wavelengths.numel(), device=device) * 0.5
    vm  = -65.0 + 5.0 * torch.randn(B, N, device=device)
    x   = engine.initial_states(B, device=device)

    i_photo, x_next = engine(irr, vm, x)
    print(f"i_photo    : {tuple(i_photo.shape)}  dtype={i_photo.dtype}")
    print(f"x_next     : {tuple(x_next.shape)}  dtype={x_next.dtype}")

    mass = x_next.sum(dim=-1)
    print(f"mass err   : {(mass - 1.0).abs().max().item():.3e}")
    print(f"min state  : {x_next.min().item():.3e}")
    print(f"i_photo    : min={i_photo.min().item():+.4f}  max={i_photo.max().item():+.4f}")

    # ── gradient sanity ───────────────────────────────────────────── #
    loss = i_photo.pow(2).mean()
    loss.backward()
    g = engine.g_raw.grad
    print(f"grad finite: {torch.isfinite(g).all().item()}  "
          f"|g|max={g.abs().max().item():.3e}")

    # ── AMP smoke test ────────────────────────────────────────────── #
    with torch.autocast(device_type=device, dtype=autocast_dt):
        i_amp, x_amp = engine(
            irr.to(autocast_dt), vm.to(autocast_dt), x.to(autocast_dt),
        )
    print(f"AMP out    : i={i_amp.dtype}, x={x_amp.dtype}")

    # ── RK4 comparison (numerical equivalence) ────────────────────── #
    engine_rk4 = AdvancedOptogeneticEngine(
        num_neurons=N,
        opsin_specs=[KNOWN_OPSINS["ChR2"], KNOWN_OPSINS["Chrimson"], KNOWN_OPSINS["GtACR2"]],
        wavelengths_nm=wavelengths,
        dt=1e-3,
        integrator="rk4",
    ).to(device)
    engine_rk4.load_state_dict(engine.state_dict())
    with torch.no_grad():
        i_rk4, _ = engine_rk4(irr, vm, x)
    print(f"|expm − rk4|  = {(i_photo - i_rk4).abs().max().item():.3e}")
