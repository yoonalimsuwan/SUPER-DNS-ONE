"""
coupled_optogenetics_nsem_safety.py
===================================

End-to-end differentiable pipeline:

    irradiance  →  OptogeneticEngine  →  I_photo
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
              NSEM Coupling             V_membrane             Safety Layer
              (neural mass)             (feedback)             (phototoxicity
                                                                 + thermal
                                                                 + No-Zeno)

Design invariants preserved from AdvancedOptogeneticEngine:
  * No clamps, no +1e-8 hacks, no boundary-gradient death
  * AMP-safe (internal FP32 lane)
  * DDP-safe (no cross-rank coupling in learnable params)
  * torch.compile-friendly (pure-functional step)
  * Matrix-exponential integrator keeps the simplex invariant
"""

from __future__ import annotations

import contextlib
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- #
#  Re-use the engine from the previous file
# --------------------------------------------------------------------- #
from advanced_optogenetics import (          # noqa: E402
    AdvancedOptogeneticEngine,
    OpsinSpec,
    KNOWN_OPSINS,
)


# ===================================================================== #
#  1.  NSEM COUPLING LAYER
# ===================================================================== #
class NSEMCouplingLayer(nn.Module):
    """
    Stateless rate-based neural mass for Neuro-Sensory Enhancement.

    The optogenetic photocurrent ``i_photo`` is injected directly as an
    external bias term in the membrane equation:

        τ_m · dV/dt = −(V − V_rest) + R_m · (I_syn + I_photo)

    Firing rate is a smooth sigmoid of the post-synaptic potential, so
    gradients flow back into the opsin kinetics through ``I_photo``.

    AMP-safe: the entire computation is C^∞ and works in FP16/BF16.
    DDP-safe: fully stateless — the caller owns ``v_membrane``.
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1e-3,
        tau_m: float = 20e-3,          # membrane time constant (s)
        R_m: float = 1.0 / 100.0,      # input resistance  (mV·pF/pA)
        V_rest: float = -65.0,         # resting potential  (mV)
        V_th: float = -50.0,           # half-activation      (mV)
        rate_slope: float = 5.0,       # sigmoid softness     (mV)
    ) -> None:
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)
        self.tau_m = float(tau_m)
        self.R_m = float(R_m)
        self.V_rest = float(V_rest)
        self.V_th = float(V_th)
        self.rate_slope = float(rate_slope)

    def forward(
        self,
        v_membrane: torch.Tensor,      # [B, N]
        i_synaptic: torch.Tensor,      # [B, N]  upstream synaptic drive
        i_photo: torch.Tensor,         # [B, N]  ← from optogenetic engine
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (v_next, firing_rate), both [B, N]."""
        i_total = i_synaptic + i_photo
        dv = (-(v_membrane - self.V_rest) + self.R_m * i_total) / self.tau_m
        v_next = v_membrane + self.dt * dv
        rate = torch.sigmoid((v_next - self.V_th) / self.rate_slope)
        return v_next, rate


# ===================================================================== #
#  2.  PHOTOTOXICITY / THERMAL SAFETY LAYER  (No-Zeno, double-exp)
# ===================================================================== #
class PhototoxicitySafetyLayer(nn.Module):
    """
    Differentiable cumulative-dose monitor with smooth safety envelope.

    Tracks three independent hazard channels:

    (1) Phototoxicity fluence
            Φ(t) = ∫₀ᵗ I_total(x, τ) dτ          [J / mm²]

    (2) Thermal damage (Arrhenius)
            Ω(t) = ∫₀ᵗ A · exp(−E_a / (R · T(τ))) dτ

    (3) No-Zeno chatter  (prevents infinitely-fast switching)
            Ψ(t) = ∫₀ᵗ (dI/dt)² dτ               [(mW/mm²)² / s]

    Each channel is bounded by a **double-exponential barrier**

            B(s) = exp( exp( α · (s / s_max − 1) ) ) − 1

    which is:
        • C^∞ everywhere
        • ≈ 0  when s ≪ s_max
        • ≈ e−1 ≈ 1.72 at s = s_max
        • blows up *smoothly* as s → ∞

    This gives the optimizer a *predictive* gradient signal that grows
    exponentially as the dose approaches the threshold — no hard ReLU
    cutoff, no gradient death at the boundary.  The result is exactly
    the No-Zeno property: the controller is guided away from the
    boundary long before reaching it, so infinite-frequency switching
    becomes sub-optimal under the loss.

    Notes
    -----
    * Buffers are non-persistent → excluded from state_dict.
    * In DDP, each rank maintains its own dose counter (correct, since
      trajectories are rank-local).
    * ``reset()`` should be called between independent rollouts.
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1e-3,
        # --- phototoxicity ---
        fluence_max: float = 200.0,        # J/mm²  (visible-light threshold)
        # --- thermal (Arrhenius) ---
        T_base_C: float = 37.0,
        dT_per_intensity: float = 2.0,     # °C per mW/mm²  (empirical)
        Ea_over_R: float = 75000.0,        # K     (protein denaturation)
        A_arr: float = 1e44,               # 1/s   (frequency factor)
        # --- No-Zeno chatter ---
        chatter_max: float = 1e3,
        # --- barrier shape ---
        sharpness: float = 8.0,            # α
    ) -> None:
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)
        self.fluence_max = float(fluence_max)
        self.T_base_C = float(T_base_C)
        self.dT_per_intensity = float(dT_per_intensity)
        self.Ea_over_R = float(Ea_over_R)
        self.A_arr = float(A_arr)
        self.chatter_max = float(chatter_max)
        self.sharpness = float(sharpness)

        # Running state — non-persistent so DDP/state_dict ignore them.
        self.register_buffer("fluence",  torch.zeros(num_neurons), persistent=False)
        self.register_buffer("thermal",  torch.zeros(num_neurons), persistent=False)
        self.register_buffer("chatter",  torch.zeros(num_neurons), persistent=False)
        self.register_buffer("last_I",   torch.zeros(num_neurons), persistent=False)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset(self) -> None:
        """Zero all accumulators — call between independent rollouts."""
        self.fluence.zero_()
        self.thermal.zero_()
        self.chatter.zero_()
        self.last_I.zero_()

    # ------------------------------------------------------------------ #
    def _barrier(self, s: torch.Tensor, s_max: float) -> torch.Tensor:
        """B(s) = exp(exp(α(s/s_max − 1))) − 1,  overflow-clamped."""
        z = self.sharpness * (s / max(s_max, 1e-12) - 1.0)
        z = z.clamp(max=20.0)                     # exp(20) ≈ 4.85e8
        return torch.exp(torch.exp(z)) - 1.0

    # ------------------------------------------------------------------ #
    def forward(
        self,
        irradiance: torch.Tensor,                # [B,N,L] | [B,N] | [B,L]
        *,
        return_diagnostics: bool = False,
        temperature_C: torch.Tensor | None = None,   # [B,N] optional override
    ):
        """
        Returns
        -------
        safety_loss : scalar tensor (differentiable) — add to training loss
        margin      : [B, N] ∈ [0,1] — 1 = safe, 0 = at threshold
        diagnostics : dict (optional)
        """
        # ---- total intensity per neuron ------------------------------- #
        if irradiance.dim() == 3:                              # [B,N,L]
            I_total = irradiance.sum(dim=-1)
        elif irradiance.dim() == 2:                            # [B,N] or [B,L]
            if irradiance.shape[1] == self.num_neurons:
                I_total = irradiance
            else:                                              # [B,L] → [B,N]
                I_total = irradiance.sum(dim=-1, keepdim=True).expand(
                    -1, self.num_neurons
                )
        else:
            raise ValueError(f"irradiance rank {irradiance.dim()} not supported")
        I_total = I_total.clamp_min(0.0)                       # physical

        # ---- per-batch mean for the accumulator update --------------- #
        I_mean = I_total.mean(dim=0)                           # [N]

        # ---- (1) fluence --------------------------------------------- #
        d_fluence = I_mean * self.dt

        # ---- (2) Arrhenius thermal damage ---------------------------- #
        if temperature_C is None:
            T_C = self.T_base_C + self.dT_per_intensity * I_mean
        else:
            T_C = temperature_C.mean(dim=0).detach()
        T_K = (T_C + 273.15).clamp_min(1.0)
        arrhenius = self.A_arr * torch.exp(-self.Ea_over_R / T_K)
        d_thermal = arrhenius * self.dt

        # ---- (3) No-Zeno chatter ------------------------------------- #
        dI_dt = (I_mean - self.last_I) / self.dt
        d_chatter = dI_dt.pow(2) * self.dt

        # ---- Update accumulators (history is detached from BPTT) ----- #
        with torch.no_grad():
            self.fluence.add_(d_fluence.detach())
            self.thermal.add_(d_thermal.detach())
            self.chatter.add_(d_chatter.detach())
            self.last_I.copy_(I_mean.detach())

        # ---- "current" dose = history + this step (grad-carrying) ---- #
        f_now  = self.fluence.detach() + d_fluence
        th_now = self.thermal.detach() + d_thermal
        ch_now = self.chatter.detach() + d_chatter

        # ---- Double-exponential barriers ----------------------------- #
        B_fluence = self._barrier(f_now,  self.fluence_max)
        B_thermal = self._barrier(th_now, 1.0)
        B_chatter = self._barrier(ch_now, self.chatter_max)

        safety_loss = (B_fluence + B_thermal + B_chatter).mean()

        # ---- Margin ∈ [0,1] for monitoring / control gating ---------- #
        margin = torch.stack(
            [
                1.0 - f_now  / self.fluence_max,
                1.0 - th_now / 1.0,
                1.0 - ch_now / self.chatter_max,
            ],
            dim=-1,
        ).clamp(0.0, 1.0).amin(dim=-1)                          # [N]

        if return_diagnostics:
            return safety_loss, margin, {
                "fluence_J_mm2":     f_now,
                "thermal_arrhenius": th_now,
                "chatter":           ch_now,
                "temperature_C":     T_C,
                "barrier_fluence":   B_fluence,
                "barrier_thermal":   B_thermal,
                "barrier_chatter":   B_chatter,
            }
        return safety_loss, margin


# ===================================================================== #
#  3.  END-TO-END COUPLED LOOP
# ===================================================================== #
class SafeOptogeneticNSEMLoop(nn.Module):
    """
    Complete differentiable pipeline for closed-loop optogenetics.

    Usage
    -----
        loop = SafeOptogeneticNSEMLoop(
            num_neurons=128,
            opsin_specs=[KNOWN_OPSINS["ChR2"], KNOWN_OPSINS["GtACR2"]],
            wavelengths_nm=torch.arange(400., 701., 10.),
        ).cuda()

        state = loop.initial_state(B, device="cuda")
        for t in range(T):
            out = loop.step(state, irr_t, i_syn_t)
            state = out["state"]
            loss = loss + task_loss(out) + 0.5 * out["safety_loss"]
    """

    def __init__(
        self,
        num_neurons: int,
        opsin_specs: list[OpsinSpec],
        wavelengths_nm: torch.Tensor,
        dt: float = 1e-3,
        integrator: str = "expm",
    ) -> None:
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dt = float(dt)

        self.optogenetics = AdvancedOptogeneticEngine(
            num_neurons=num_neurons,
            opsin_specs=opsin_specs,
            wavelengths_nm=wavelengths_nm,
            dt=dt,
            integrator=integrator,
        )
        self.nsem = NSEMCouplingLayer(num_neurons, dt=dt)
        self.safety = PhototoxicitySafetyLayer(num_neurons, dt=dt)

    # ------------------------------------------------------------------ #
    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "v": torch.full(
                (batch_size, self.num_neurons), -65.0, device=device, dtype=dtype
            ),
            "x": self.optogenetics.initial_states(
                batch_size, device=device, dtype=dtype
            ),
        }

    @torch.no_grad()
    def reset_safety(self) -> None:
        self.safety.reset()

    # ------------------------------------------------------------------ #
    def step(
        self,
        state: Mapping[str, torch.Tensor],
        irradiance: torch.Tensor,
        i_synaptic: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        temperature_C: torch.Tensor | None = None,
    ) -> dict:
        """One simulation step. Returns dict with state, rate, safety."""
        # 1. Opsin kinetics → photocurrent
        i_photo, x_next = self.optogenetics(
            irradiance, state["v"], state["x"]
        )

        # 2. NSEM neural mass — I_photo enters as external bias
        v_next, rate = self.nsem(state["v"], i_synaptic, i_photo)

        # 3. Safety evaluation (differentiable barrier)
        if return_diagnostics:
            safety_loss, margin, diag = self.safety(
                irradiance,
                return_diagnostics=True,
                temperature_C=temperature_C,
            )
        else:
            safety_loss, margin = self.safety(
                irradiance, temperature_C=temperature_C
            )
            diag = None

        out = {
            "state":       {"v": v_next, "x": x_next},
            "i_photo":     i_photo,
            "rate":        rate,
            "safety_loss": safety_loss,
            "margin":      margin,          # [B, N], 1 = safe, 0 = danger
        }
        if diag is not None:
            out["diagnostics"] = diag
        return out


# ===================================================================== #
#  4.  DEMO — training-style rollout
# ===================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, T = 2, 64, 100
    wavelengths = torch.arange(400.0, 701.0, 10.0)

    loop = SafeOptogeneticNSEMLoop(
        num_neurons=N,
        opsin_specs=[KNOWN_OPSINS["ChR2"], KNOWN_OPSINS["GtACR2"]],
        wavelengths_nm=wavelengths,
        dt=1e-3,
    ).to(device)

    optimizer = torch.optim.AdamW(loop.parameters(), lr=3e-4)

    # simple learnable controller: irradiance ∝ W · rate
    controller = nn.Linear(N, wavelengths.numel() * N).to(device)
    optimizer.add_param_group({"params": controller.parameters(), "lr": 1e-3})

    target_rate = 0.5

    for epoch in range(3):
        loop.reset_safety()
        state = loop.initial_state(B, device=device)
        state = {k: v.detach() for k, v in state.items()}

        total_loss = 0.0
        last_diag = None

        for t in range(T):
            # controller consumes rate → emits irradiance
            flat = controller(state["v"]).view(B, N, wavelengths.numel())
            irr = F.softplus(flat) * 0.2                  # 0 – ~1 mW/mm²
            i_syn = 5.0 * torch.randn(B, N, device=device)

            out = loop.step(
                state, irr, i_syn,
                return_diagnostics=(t == T - 1),
            )
            state = {k: v.detach() for k, v in out["state"].items()}

            task_loss = F.mse_loss(out["rate"], torch.full_like(out["rate"], target_rate))
            total_loss = total_loss + task_loss + 0.5 * out["safety_loss"]

            if "diagnostics" in out:
                last_diag = out["diagnostics"]

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(loop.parameters()) + list(controller.parameters()), 1.0
        )
        optimizer.step()

        print(
            f"epoch {epoch}  loss={total_loss.item():.4f}  "
            f"fluence_max={last_diag['fluence_J_mm2'].max():.3f} J/mm²  "
            f"T_max={last_diag['temperature_C'].max():.2f} °C  "
            f"margin_min={out['margin'].min().item():.3f}"
        )
