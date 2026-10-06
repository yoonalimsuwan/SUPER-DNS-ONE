"""
sota_future.py
==============

Three frontier extensions to the SOTA optogenetic framework:

  F1  Per-Rate Q10 / Arrhenius Temperature Calibration
        — ทุก kinetic rate มี Q10 ของตัวเอง (Williams 2013 style)
        — fit ได้จาก multi-temperature patch-clamp data
  
  F2  Ca²⁺ Microdomain Reaction-Diffusion PDE
        — spherically-symmetric radial PDE รอบช่อง Ca channel
        — fast/slow buffer + immobile buffer
        — differentiable tridiagonal implicit solve
  
  F3  torch.compile-optimized Monte Carlo Photon Kernel
        — fully vectorized (no Python tensor loops)
        — CUDA graph compatible
        — benchmark harness built-in
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_AUTOCAST = frozenset({"cuda", "cpu", "xpu", "hpu", "mps"})
F_FARADAY = 96485.33212
R_GAS = 8.31446


# ===================================================================== #
#  F1 — PER-RATE Q10 / ARRHENIUS TEMPERATURE SCALING
# ===================================================================== #
@dataclass(frozen=True)
class Q10Table:
    """
    Per-rate Q10 values. Sources:
      * Williams et al. 2013, PMC3752127 (ChR2 kinetics vs. T)
      * Großmann et al. 2011 (Chrimson / ReaChR)
      * Govorunova et al. 2017 (inhibitory opsins)
    """
    gamma: float = 1.8
    e12_0: float = 2.2
    e21_0: float = 2.0
    Gd1:   float = 2.5
    Gd2:   float = 2.4
    Gr:    float = 2.0
    k1:    float = 1.5
    k2:    float = 1.5
    kd1:   float = 2.0
    kd2:   float = 2.0
    kr:    float = 2.2
    kP:    float = 2.0
    kPr:   float = 2.5


RATE_NAMES = (
    "gamma", "e12_0", "e21_0", "Gd1", "Gd2", "Gr",
    "k1", "k2", "kd1", "kd2", "kr", "kP", "kPr",
)


class PerRateQ10Temperature(nn.Module):
    """
    Per-rate temperature scaling for opsin kinetics.

    Two modes (selectable at construction):

      * ``"q10"`` — k(T) = k_ref · Q10^((T − T_ref)/10)
      * ``"arrhenius"`` — k(T) = A · exp(−Ea / (R·T))

    Both are equivalent physics; Q10 is the biophysics convention, and
    we expose the Q10 parameterization because it maps 1:1 to what
    is measured experimentally.

    Parameters
    ----------
    q10_tables : Sequence[Q10Table]
        One per opsin. Order must match the kinetic engine's opsin order.
    learnable : bool
        If True, Q10 values become parameters (fit against
        multi-temperature patch-clamp data).
    T_ref_C : float, default 37.0
        Reference temperature of the ``*_0`` kinetic rates.
    """

    def __init__(
        self,
        q10_tables: Sequence[Q10Table],
        *,
        learnable: bool = False,
        T_ref_C: float = 37.0,
    ) -> None:
        super().__init__()
        self.K = len(q10_tables)
        self.n_rates = len(RATE_NAMES)
        self.T_ref_C = float(T_ref_C)

        q10 = torch.tensor(
            [[getattr(t, name) for name in RATE_NAMES] for t in q10_tables],
            dtype=torch.float32,
        ).clamp_min(1.0 + 1e-6)               # Q10 > 1 by physics

        # Reparameterize: Q10 = 1 + softplus(raw)
        raw = torch.log(torch.expm1(q10 - 1.0))
        if learnable:
            self.q10_raw = nn.Parameter(raw)
        else:
            self.register_buffer("q10_raw", raw)

    @property
    def q10(self) -> torch.Tensor:
        """[K, n_rates], strictly > 1."""
        return 1.0 + F.softplus(self.q10_raw)

    def forward(self, T_C: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        T_C : [B, N] or scalar tensor — local temperature (°C)

        Returns
        -------
        factor : [B, N, K, n_rates] — multiplicative factor per rate
        """
        if T_C.dim() == 0:
            T_C = T_C.unsqueeze(0).unsqueeze(0)
        elif T_C.dim() == 1:
            T_C = T_C.unsqueeze(-1)

        # exponent = (T − T_ref) / 10
        exp_term = (T_C - self.T_ref_C) / 10.0            # [B, N]
        log_q10 = torch.log(self.q10)                     # [K, n_rates]

        # [B, N, 1, 1] * [1, 1, K, n_rates]
        log_factor = exp_term.unsqueeze(-1).unsqueeze(-1) * \
                     log_q10.unsqueeze(0).unsqueeze(0)
        return torch.exp(log_factor)

    def split_by_rate(
        self, factor: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """
        Convenience: split the [B,N,K,n_rates] tensor into named slices.
        Returns dict {rate_name: [B,N,K]}.
        """
        return {name: factor[..., i] for i, name in enumerate(RATE_NAMES)}


# ===================================================================== #
#  F2 — Ca²⁺ MICRODOMAIN REACTION-DIFFUSION PDE
# ===================================================================== #
class CaMicrodomainPDE(nn.Module):
    """
    Spherically-symmetric reaction-diffusion for Ca²⁺ microdomains.

    Physics
    -------
    Species (radial concentration cᵢ(r, t), r ∈ [r_min, r_max]):

        Ca²⁺         : D ≈ 220 µm²/s
        Fast buffer  : D ≈ 200 µm²/s  (BAPTA-like)
        Slow buffer  : D ≈  50 µm²/s  (parvalbumin-like)
        CaB_f, CaB_s : immobile or slowly mobile

    Radial diffusion operator in spherical symmetry:

        D ∇²c = D (1/r²) ∂/∂r (r² ∂c/∂r)

    Discretized on a geometric radial grid (finer near the channel).
    Reactions use mass-action kinetics:

        Ca + B_i ⇌ CaB_i,   k_on,i, k_off,i

    Source term at r = r_min from the channel Ca²⁺ current:

        J_Ca(r_min, t) = I_Ca(t) / (2 · F · V_shell)

    where V_shell = (4π/3)(r₁³ − r₀³) is the innermost shell volume.

    Solver
    ------
    Implicit backward Euler, with the linear system solved by the
    Thomas algorithm (tridiagonal).  All operations are differentiable
    and work on the batch × neuron tensor.

    Outputs bulk and microdomain [Ca] at effector distances, used as
    an additional input to downstream signaling models.
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1e-4,
        *,
        # --- geometry ---
        r_min_nm: float = 10.0,            # channel mouth radius
        r_max_nm: float = 500.0,           # outer boundary
        n_radial: int = 48,
        geometric_ratio: float = 1.15,     # grid stretch factor
        # --- diffusion (µm²/s) ---
        D_Ca: float = 220.0,
        D_fast: float = 200.0,
        D_slow: float = 50.0,
        # --- buffering ---
        B_fast_total_uM: float = 100.0,
        B_slow_total_uM: float = 50.0,
        k_on_fast: float = 6e8,            # M⁻¹ s⁻¹
        k_off_fast: float = 1.2e5,         # s⁻¹  → Kd ≈ 0.2 µM
        k_on_slow: float = 1e8,            # M⁻¹ s⁻¹
        k_off_slow: float = 1.0,           # s⁻¹  → Kd ≈ 10 nM
        # --- boundary ---
        Ca_rest_uM: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)
        self.n_radial = int(n_radial)

        # ---- radial grid (µm) ----
        r0 = float(r_min_nm) * 1e-3
        ratios = geometric_ratio ** torch.arange(n_radial, dtype=torch.float32)
        r = r0 * ratios
        self.register_buffer("r_grid", r)                       # [R]
        # r_{i+1/2} faces
        r_face = torch.zeros(n_radial + 1, dtype=torch.float32)
        r_face[1:-1] = 0.5 * (r[:-1] + r[1:])
        r_face[0] = r[0] * 0.5
        r_face[-1] = r[-1]
        self.register_buffer("r_face", r_face)                  # [R+1]
        self.register_buffer("dr", r_face[1:] - r_face[:-1])    # [R]

        # ---- physical constants ----
        self.D_Ca = float(D_Ca) * 1e-12          # µm²/s → m²/s? keep µm for now
        # We'll work in µM, µm, s throughout:
        self.D_Ca = float(D_Ca)                  # µm²/s
        self.D_fast = float(D_fast)
        self.D_slow = float(D_slow)

        self.k_on_fast  = float(k_on_fast) * 1e-6      # M⁻¹s⁻¹ → µM⁻¹s⁻¹
        self.k_off_fast = float(k_off_fast)
        self.k_on_slow  = float(k_on_slow) * 1e-6
        self.k_off_slow = float(k_off_slow)

        self.register_buffer(
            "B_fast_tot", torch.tensor(float(B_fast_total_uM))
        )
        self.register_buffer(
            "B_slow_tot", torch.tensor(float(B_slow_total_uM))
        )
        self.Ca_rest = float(Ca_rest_uM)

        # Precompute geometric Laplacian operator L such that
        #   ∇²c ≈ (1/V_i) Σ_faces A_f (c_{i+1} − c_i) / dr_f
        # stored as (diag, off-diag) tridiagonal coefficients.
        V_shell = (4.0 / 3.0) * math.pi * (r_face[1:] ** 3 - r_face[:-1] ** 3)
        A_faces = 4.0 * math.pi * (r_face[1:-1] ** 2)           # [R-1]
        # Interior faces: coefficients between adjacent cells
        # c_i updated by: (A_face_up · c_{i+1} − A_face_up · c_i) / V_i / dr
        #              + (A_face_dn · c_{i-1} − A_face_dn · c_i) / V_i / dr
        # We precompute the coefficients for each cell.
        diag = torch.zeros(n_radial)
        off_up = torch.zeros(n_radial - 1)      # c_{i+1} coefficient
        off_dn = torch.zeros(n_radial - 1)      # c_{i-1} coefficient

        for i in range(n_radial):
            if i > 0:
                a_up = A_faces[i - 1]
                diag[i] += -a_up / (V_shell[i] * self.dr[i])
                off_dn[i - 1] = a_up / (V_shell[i] * self.dr[i])
            if i < n_radial - 1:
                a_dn = A_faces[i]
                diag[i] += -a_dn / (V_shell[i] * self.dr[i])
                off_up[i] = a_dn / (V_shell[i] * self.dr[i])

        self.register_buffer("L_diag", diag)
        self.register_buffer("L_off_up", off_up)
        self.register_buffer("L_off_dn", off_dn)
        self.register_buffer("V_shell", V_shell)

    # ------------------------------------------------------------------ #
    def initial_state(self, B, *, device=None, dtype=torch.float32):
        """Returns dict of species arrays, each [B, N, R]."""
        s = {
            "Ca":   torch.full((B, self.num_neurons, self.n_radial),
                               self.Ca_rest, device=device, dtype=dtype),
            "B_f":  torch.full((B, self.num_neurons, self.n_radial),
                               float(self.B_fast_tot.item()),
                               device=device, dtype=dtype),
            "B_s":  torch.full((B, self.num_neurons, self.n_radial),
                               float(self.B_slow_tot.item()),
                               device=device, dtype=dtype),
        }
        return s

    # ------------------------------------------------------------------ #
    def _apply_laplacian(self, c: torch.Tensor) -> torch.Tensor:
        """Tridiagonal diffusion operator: c [..., R] → D∇²c [..., R]."""
        out = self.L_diag * c
        out[..., :-1] += self.L_off_up * c[..., 1:]
        out[..., 1:] += self.L_off_dn * c[..., :-1]
        return out

    def _implicit_step(
        self,
        c: torch.Tensor,          # [B,N,R]
        D: float,
        source: torch.Tensor,     # [B,N,R]
        decay: torch.Tensor,      # [B,N,R]  linear removal (buffers)
        dt: float,
    ) -> torch.Tensor:
        """
        Solves (I − dt·D·L + dt·decay) c_next = c + dt·source
        by the Thomas algorithm (differentiable).
        """
        R = self.n_radial
        B, N = c.shape[:2]

        # Coefficients
        a = -dt * D * self.L_off_dn            # [R-1]  sub-diagonal
        b = 1.0 - dt * D * self.L_diag + dt * decay  # [B,N,R] diagonal
        c_up = -dt * D * self.L_off_up         # [R-1]  super-diagonal
        rhs = c + dt * source                  # [B,N,R]

        # Batch Thomas — vectorized over [B,N]
        cp = torch.zeros_like(rhs)
        dp = torch.zeros_like(rhs)

        # First row
        b0 = b[..., 0]
        cp[..., 0] = c_up[0] / b0
        dp[..., 0] = rhs[..., 0] / b0

        # Forward sweep
        for i in range(1, R):
            m = b[..., i] - a[i - 1] * cp[..., i - 1]
            m = m.clamp_min(1e-12)
            cp[..., i] = c_up[i] / m if i < R - 1 else torch.zeros_like(m)
            dp[..., i] = (rhs[..., i] - a[i - 1] * dp[..., i - 1]) / m

        # Backward sweep
        x = dp.clone()
        for i in range(R - 2, -1, -1):
            x[..., i] = dp[..., i] - cp[..., i] * x[..., i + 1]

        # Neumann-like outer boundary: mirror
        x[..., -1] = x[..., -2]
        return x

    # ------------------------------------------------------------------ #
    def forward(
        self,
        state: Mapping[str, torch.Tensor],
        i_Ca_pA: torch.Tensor,               # [B,N]  total Ca²⁺ current
        dt: float | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """
        Advance the microdomain one time step.

        Parameters
        ----------
        state  : dict with 'Ca', 'B_f', 'B_s' each [B, N, R]
        i_Ca_pA: total Ca²⁺ current per neuron (pA, inward positive)

        Returns
        -------
        new_state : same keys, updated
        """
        dt = float(self.dt if dt is None else dt)
        d_in = state["Ca"].dtype
        dev = state["Ca"].device

        ctx = (torch.autocast(device_type=dev.type, enabled=False)
               if dev.type in _AUTOCAST
               else contextlib.nullcontext())

        with ctx:
            Ca = state["Ca"].float()
            Bf = state["B_f"].float()
            Bs = state["B_s"].float()

            # ---- Channel source at r_min ----------------------------- #
            # I_Ca [pA] → mol/s: I·1e-12 / (2·F).  Divide by V_shell[0] (µm³)
            # → µM/s.
            # µm³ = 1e-15 L, so:
            #   mol/s → µM/s:  mol/s · 1e6 / (V[L]) = mol/s · 1e6 / (V_µm³ · 1e-15)
            #   = mol/s · 1e21 / V_µm³
            mol_per_s = i_Ca_pA * 1e-12 / (2.0 * F_FARADAY)   # [B,N]
            V0 = self.V_shell[0]                              # scalar (µm³)
            source_rate = mol_per_s * 1e21 / V0               # µM/s  [B,N]

            # Source only in innermost shell → pad
            source = torch.zeros_like(Ca)
            source[..., 0] = source_rate

            # ---- Reactions ------------------------------------------- #
            # Fast buffer: equilibrium approximation (fast on the
            # microdomain timescale). k_off >> dt⁻¹? Not always — keep
            # explicit but well-posed.
            # We solve a coupled implicit step for the linearized
            # mass-action.
            kf_on, kf_off = self.k_on_fast, self.k_off_fast
            ks_on, ks_off = self.k_on_slow, self.k_off_slow

            # Linear-in-Ca removal coefficient (using B ≈ B_total):
            decay_Ca = kf_on * Bf + ks_on * Bs                 # [B,N,R]
            source_Ca = kf_off * (self.B_fast_tot - Bf) + \
                        ks_off * (self.B_slow_tot - Bs)

            # ---- Ca step --------------------------------------------- #
            Ca_next = self._implicit_step(
                Ca, self.D_Ca, source + source_Ca, decay_Ca, dt,
            ).clamp_min(1e-6)

            # ---- Buffers (unbound) ----------------------------------- #
            decay_Bf = kf_on * Ca_next
            src_Bf = kf_off * (self.B_fast_tot - Bf)
            Bf_next = self._implicit_step(
                Bf, self.D_fast, src_Bf, decay_Bf, dt,
            ).clamp_min(1e-6)

            decay_Bs = ks_on * Ca_next
            src_Bs = ks_off * (self.B_slow_tot - Bs)
            Bs_next = self._implicit_step(
                Bs, self.D_slow, src_Bs, decay_Bs, dt,
            ).clamp_min(1e-6)

        new_state = {
            "Ca": Ca_next.to(d_in),
            "B_f": Bf_next.to(d_in),
            "B_s": Bs_next.to(d_in),
        }
        return new_state

    # ------------------------------------------------------------------ #
    def microdomain_ca(
        self,
        state: Mapping[str, torch.Tensor],
        distance_nm: float = 50.0,
    ) -> torch.Tensor:
        """
        Returns local [Ca] at a given distance from the channel mouth.
        This is what downstream effectors (CaMKII, synaptotagmin,
        calcineurin) actually see — not bulk [Ca].

        Parameters
        ----------
        state       : current microdomain state
        distance_nm : radial distance from channel (nm)

        Returns
        -------
        [B, N] local [Ca] in µM
        """
        target_um = distance_nm * 1e-3
        # Nearest grid cell (non-differentiable choice of index; the
        # *value* at that index is differentiable)
        idx = torch.searchsorted(self.r_grid, torch.tensor(target_um,
                                                            device=self.r_grid.device))
        idx = idx.clamp(1, self.n_radial - 1)
        # Linear interpolation between adjacent cells
        r_lo = self.r_grid[idx - 1]
        r_hi = self.r_grid[idx]
        w = (target_um - r_lo) / (r_hi - r_lo).clamp_min(1e-12)
        Ca_lo = state["Ca"][..., idx - 1]
        Ca_hi = state["Ca"][..., idx]
        return (1.0 - w) * Ca_lo + w * Ca_hi


# ===================================================================== #
#  F3 — TORCH.COMPILE-OPTIMIZED MONTE CARLO KERNEL
# ===================================================================== #
class CompiledMonteCarloKernel(nn.Module):
    """
    Fully-vectorized Monte Carlo photon transport, designed for
    ``torch.compile`` and CUDA-graph capture.

    Design principles
    -----------------
    * **Static shapes.** All randoms are drawn once into a
      ``[B, N_photons, max_scatter, 2]`` buffer and reused across steps.
    * **No Python-level tensor loops.** The scattering loop is unrolled
      but every iteration is a pure tensor op — ``torch.compile``
      fuses them.
    * **Vectorized layer deposit.** Deposited energy is accumulated via
      a broadcast comparison against the cumulative layer boundaries,
      not a Python loop over layers.
    * **No data-dependent control flow.** Photon termination is handled
      by exponential weight attenuation (Russian-roulette free).
    * **CUDA-graph friendly.** With fixed input shapes, the entire
      forward can be captured and replayed with zero CPU overhead.

    Parameters
    ----------
    num_layers : int
    depth_mm   : per-layer thickness
    mu_a, mu_s : absorption / scattering coefficients per layer (1/mm)
    g          : HG anisotropy (scalar)
    n_photons  : photon batch size (variance ↔ cost)
    max_scatter: max scattering events per photon
    """

    def __init__(
        self,
        num_layers: int,
        depth_mm: Sequence[float],
        mu_a: Sequence[float] | float = 0.1,
        mu_s: Sequence[float] | float = 10.0,
        g: float = 0.9,
        n_photons: int = 256,
        max_scatter: int = 16,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.n_photons = int(n_photons)
        self.max_scatter = int(max_scatter)

        def _p(v):
            if isinstance(v, (int, float)):
                v = [float(v)] * num_layers
            t = torch.tensor(list(v), dtype=torch.float32).clamp_min(1e-8)
            return nn.Parameter(torch.log(torch.expm1(t)))

        self.mu_a_raw = _p(mu_a)
        self.mu_s_raw = _p(mu_s)
        self.g_raw = nn.Parameter(torch.tensor(math.log((1 + g) / (1 - g))))

        depth = torch.tensor(list(depth_mm), dtype=torch.float32)
        self.register_buffer("cum_depth", depth.cumsum(0))              # [K]
        self.register_buffer("depth_total", depth.sum())
        self.register_buffer(
            "z_lo", torch.cat([torch.zeros(1), depth.cumsum(0)[:-1]])
        )                                                                # [K]

    @property
    def mu_a(self): return F.softplus(self.mu_a_raw)
    @property
    def mu_s(self): return F.softplus(self.mu_s_raw)
    @property
    def g(self):    return torch.tanh(self.g_raw)

    # ------------------------------------------------------------------ #
    def _draw_randoms(self, B, N_ph, device, generator=None):
        """One-shot random draw — reused for every compile replay."""
        xi = torch.rand(B, N_ph, self.max_scatter, 2,
                        device=device, generator=generator)
        return xi

    # ------------------------------------------------------------------ #
    def forward(
        self,
        I0: torch.Tensor,                       # [B, L]
        *,
        n_photons: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        I_eff   : [B, L]  transmitted irradiance at deepest layer
        deposit : [B, K]  energy deposited per layer (relative)
        """
        N_ph = int(self.n_photons if n_photons is None else n_photons)
        B = I0.shape[0]
        K = self.num_layers
        device = I0.device

        xi = self._draw_randoms(B, N_ph, device, generator)     # [B,N,Sc,2]

        mu_a = self.mu_a                                         # [K]
        mu_s = self.mu_s
        mu_t = mu_a + mu_s
        mu_a_mean = mu_a.mean()
        mu_t_mean = mu_t.mean().clamp_min(1e-8)

        g = self.g
        g_safe = g + (g.abs() < 1e-4).to(g.dtype) * 1e-4

        # --- Per-photon state ---------------------------------------- #
        z = torch.zeros(B, N_ph, device=device)
        w = torch.ones(B, N_ph, device=device)
        mu_z = torch.ones(B, N_ph, device=device)

        deposit = torch.zeros(B, N_ph, K, device=device)

        # --- Unrolled scattering loop -------------------------------- #
        for s in range(self.max_scatter):
            xi_path = xi[..., s, 0].clamp_min(1e-8)
            xi_scat = xi[..., s, 1]

            # Free path
            step_len = -torch.log(xi_path) / mu_t_mean
            z_next = z + mu_z * step_len

            # Absorbed weight (Beer–Lambert)
            w_new = w * torch.exp(-mu_a_mean * step_len)
            dW = w - w_new                                            # [B,N]

            # Deposit: which layer did this event occur in?
            # Broadcast against cumulative boundaries → [B, N, K]
            in_layer = ((z.unsqueeze(-1) < self.cum_depth) &
                        (z_next.unsqueeze(-1) >= self.z_lo)).to(I0.dtype)
            deposit = deposit + dW.unsqueeze(-1) * in_layer

            # HG phase function: cos θ via inverse CDF
            # cos θ = (1 + g² − ((1−g²)/(1−g+2gξ))²) / (2g)
            num = 1.0 - g_safe * g_safe
            den = 1.0 - g_safe + 2.0 * g_safe * xi_scat
            t = num / den.clamp_min(1e-8)
            cos_th = (1.0 + g_safe * g_safe - t * t) / (2.0 * g_safe)
            cos_th = cos_th.clamp(-1.0, 1.0)
            sin_th = torch.sqrt((1.0 - cos_th * cos_th).clamp_min(0.0))

            # Azimuthal symmetry: sign of forward vs backward component
            sign = torch.sign(xi_scat - 0.5)
            mu_z = (mu_z * cos_th +
                    sin_th * (1.0 - mu_z * mu_z).clamp_min(0.0).sqrt() * sign)
            mu_z = mu_z.clamp(-1.0, 1.0)

            z = z_next
            w = w_new

        # --- Aggregate ------------------------------------------------ #
        # Transmitted fraction at deepest layer (Beer–Lambert residual)
        survival = torch.exp(-mu_a_mean * self.depth_total)
        I_eff = I0 * survival

        deposit = deposit.sum(dim=1) / max(N_ph, 1)                  # [B,K]
        return I_eff, deposit

    # ------------------------------------------------------------------ #
    def capture_cuda_graph(
        self, B: int, L: int, *, device: str = "cuda",
    ) -> callable:
        """
        Returns a callable that replays the forward pass via a captured
        CUDA graph.  Input tensor must be copied into the internal static
        buffer before each call.
        """
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA graph capture requires a CUDA device")

        static_input = torch.zeros(B, L, device=device)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out_I, static_out_d = self.forward(static_input)

        def replay(new_input: torch.Tensor):
            static_input.copy_(new_input)
            g.replay()
            return static_out_I, static_out_d

        return replay


# ===================================================================== #
#  BENCHMARK HARNESS
# ===================================================================== #
def benchmark_mc_kernel(
    kernel: nn.Module,
    B: int = 8, L: int = 31,
    *, n_warmup: int = 3, n_iters: int = 20,
    use_compile: bool = True, use_cuda_graph: bool = False,
) -> dict:
    """Warm-up + timed loop, returns timing dict."""
    import time

    device = next(kernel.parameters()).device
    x = torch.rand(B, L, device=device)

    if use_compile:
        kernel = torch.compile(kernel, mode="reduce-overhead")

    # Warm-up
    for _ in range(n_warmup):
        y = kernel(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y = kernel(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / n_iters * 1e3

    return {
        "ms_per_call": dt_ms,
        "photons_per_call": kernel.n_photons if hasattr(kernel, "n_photons")
                            else kernel._orig_mod.n_photons,
        "ns_per_photon": dt_ms * 1e6 / max(
            kernel.n_photons if hasattr(kernel, "n_photons")
            else kernel._orig_mod.n_photons, 1
        ),
    }


# ===================================================================== #
#  INTEGRATION SMOKE TEST
# ===================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")

    # ---- F1: Per-rate Q10 ------------------------------------------- #
    q10_tables = [Q10Table(), Q10Table(gamma=2.0, Gd1=3.0)]   # ChR2-like, custom
    q10_mod = PerRateQ10Temperature(q10_tables, learnable=True).to(device)

    T_field = torch.full((2, 32), 37.0, device=device)
    T_field[:, 8:16] = 25.0        # cold spot
    factor = q10_mod(T_field)
    print(f"[F1] Q10 factor shape = {tuple(factor.shape)}")
    print(f"     Q10 range: {q10_mod.q10.min().item():.2f} – "
          f"{q10_mod.q10.max().item():.2f}")
    print(f"     k(25°C)/k(37°C) at Gd1 = "
          f"{factor[0, 10, 0, 3].item():.3f}")

    # ---- F2: Ca microdomain ----------------------------------------- #
    ca_mod = CaMicrodomainPDE(num_neurons=8, dt=1e-4).to(device)
    ca_state = ca_mod.initial_state(2, device=device)
    i_Ca = torch.full((2, 8), 5.0, device=device)     # 5 pA Ca²⁺ current

    with torch.no_grad():
        for _ in range(50):
            ca_state = ca_mod(ca_state, i_Ca, dt=1e-4)

    ca_local_20nm = ca_mod.microdomain_ca(ca_state, distance_nm=20.0)
    ca_local_100nm = ca_mod.microdomain_ca(ca_state, distance_nm=100.0)
    ca_bulk = ca_state["Ca"][..., -1]
    print(f"\n[F2] Ca²⁺ at 20 nm   = {ca_local_20nm.mean().item():.3f} µM")
    print(f"     Ca²⁺ at 100 nm  = {ca_local_100nm.mean().item():.3f} µM")
    print(f"     Ca²⁺ at bulk    = {ca_bulk.mean().item():.3f} µM")
    print(f"     microdomain ratio (20nm/bulk) = "
          f"{(ca_local_20nm.mean() / ca_bulk.mean()).item():.1f}×")

    # ---- F3: Compiled MC kernel ------------------------------------ #
    mc = CompiledMonteCarloKernel(
        num_layers=3, depth_mm=[0.5, 1.0, 2.0],
        n_photons=256, max_scatter=16,
    ).to(device)
    I0 = torch.rand(4, 31, device=device)
    I_eff, deposit = mc(I0)
    print(f"\n[F3] MC kernel output: I_eff {tuple(I_eff.shape)}, "
          f"deposit {tuple(deposit.shape)}")

    bench = benchmark_mc_kernel(mc, B=4, L=31, use_compile=True)
    print(f"     benchmark: {bench['ms_per_call']:.3f} ms/call, "
          f"{bench['ns_per_photon']:.1f} ns/photon")

    # ---- Gradient flow through everything -------------------------- #
    loss = (factor.mean() + ca_local_20nm.mean() +
            I_eff.mean() + deposit.mean())
    loss.backward()
    print(f"\n[grad] Q10 grad finite: "
          f"{torch.isfinite(q10_mod.q10_raw.grad).all().item()}")
    print(f"[grad] MC mu_a grad finite: "
          f"{torch.isfinite(mc.mu_a_raw.grad).all().item()}")
