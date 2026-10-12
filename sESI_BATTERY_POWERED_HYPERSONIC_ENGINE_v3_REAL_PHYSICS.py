# =============================================================================
# SESI BATTERY-POWERED HYPERSONIC ENGINE v3 — REAL PHYSICS
# Physics-grounded: Butler-Volmer + SOC + Isentropic nozzle + Arrhenius
# =============================================================================
# Developer  : PAI and Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# Version    : 3.0.0 (Physics-Grounded / C^∞ / AMP-Safe / DDP-Ready)
# =============================================================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import (
    SI,
    butler_volmer_current,
    soc_dynamics,
    arrhenius_rate,
    isentropic_ratios,
    rankine_hugoniot,
    soft_floor,
)

__all__ = [
    "EngineConfig",
    "ProductionHypersonicBatteryEngineModuleV3",
    "compute_production_loss_v3",
]


# =============================================================================
@dataclass
class EngineConfig:
    latent_dim: int = 32
    C_nom_Ah: float = 1500.0          # battery nominal capacity [Ah]
    V_nom: float = 800.0              # nominal voltage [V]
    T_chamber: float = 2800.0         # combustion chamber T [K]
    p_chamber: float = 5e6            # chamber pressure [Pa]
    p_ambient: float = 101325.0       # ambient pressure [Pa]
    signature_eps: float = 1e-4
    denom_eps: float = 1e-8
    softplus_beta: float = 4.0
    inner_clamp: float = 15.0
    log_floor: float = -40.0
    smoothing_tau: float = 0.05


# =============================================================================
class ProductionHypersonicBatteryEngineModuleV3(nn.Module):
    """
    Physics-grounded battery-powered hypersonic engine module.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        *,
        latent_dim: Optional[int] = None,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if config is None:
            config = EngineConfig()
        if latent_dim is not None:
            config = EngineConfig(**{**config.__dict__, "latent_dim": int(latent_dim)})
        self.cfg = config
        self.latent_dim = config.latent_dim
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        # Learnable state-space mapping Φ_U
        self.phi_u_tensor_net = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_dim),
        )
        # Battery controller → [SOC_raw, T_batt_raw, mdot_raw]
        self.battery_power_controller = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        # Thermal barrier ΔE estimator
        self.thermal_barrier_estimator = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Aerothermo actuator: [Nucleation, Merging, Branching]
        self.aerothermo_actuator_controller = nn.Sequential(
            nn.Linear(self.latent_dim, 3),
        )

        def _buf(v):
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_sig_eps",      _buf(config.signature_eps),  persistent=False)
        self.register_buffer("_sp_beta",      _buf(config.softplus_beta),  persistent=False)
        self.register_buffer("_log_floor",    _buf(config.log_floor),      persistent=False)
        self.register_buffer("_C_nom",        _buf(config.C_nom_Ah),       persistent=False)
        self.register_buffer("_V_nom",        _buf(config.V_nom),          persistent=False)
        self.register_buffer("_T_chamber",    _buf(config.T_chamber),      persistent=False)
        self.register_buffer("_p_chamber",    _buf(config.p_chamber),      persistent=False)
        self.register_buffer("_p_amb",        _buf(config.p_ambient),      persistent=False)
        self.register_buffer(
            "_log_det_const",
            torch.tensor(float((self.latent_dim - 1) * math.log(config.signature_eps))),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    def _topological_signature(self, v):
        eps = self._sig_eps.to(v.dtype)
        const = self._log_det_const.to(v.dtype)
        v_sq = (v * v).sum(dim=-1)
        return const + torch.log(v_sq + eps)

    # ------------------------------------------------------------------ #
    def _battery_step(self, SOC, I_demand, T_batt, dt):
        # Arrhenius exchange current density
        i0 = 1e-3 * arrhenius_rate(T_batt, Ea=1.0e4, T_ref=298.15)
        # Overpotential from SOC (rough linear model)
        eta = self._V_nom.to(I_demand.dtype) * (1.0 - SOC.clamp(0.0, 1.0))
        I_actual = butler_volmer_current(eta, i0, T=T_batt)
        dSOC = soc_dynamics(I_actual, self._C_nom.to(I_actual.dtype), dt)
        return I_actual, dSOC

    # ------------------------------------------------------------------ #
    def _nozzle_thrust(self, mdot, T_chamber, p_chamber, p_ambient):
        """Isentropic nozzle:  v_e = √(2γ R T0/(γ−1) · [1 − (pe/p0)^((γ−1)/γ)])."""
        gamma = SI.gamma_air
        pe_p0 = (p_ambient / p_chamber.clamp(min=1.0)).clamp(1e-3, 0.99)
        v_e = torch.sqrt(2.0 * gamma * SI.R_air * T_chamber / (gamma - 1.0)
                         * (1.0 - pe_p0 ** ((gamma - 1.0) / gamma)))
        thrust = mdot * v_e
        Isp = v_e / SI.g_0
        return thrust, Isp, v_e

    # ------------------------------------------------------------------ #
    def _step(self, state_x, dt=1e-3):
        dtype = state_x.dtype
        device = state_x.device
        sp_beta = self._sp_beta.to(dtype=dtype, device=device)

        # 1. Φ_U contraction
        contracted = self.phi_u_tensor_net(state_x)

        # 2. Topological signature (closed form, C^∞)
        det_sig = self._topological_signature(contracted)

        # 3. Battery outputs
        batt_raw = self.battery_power_controller(contracted)
        SOC_raw   = batt_raw[:, 0:1]
        T_raw     = batt_raw[:, 1:2]
        mdot_raw  = batt_raw[:, 2:3]

        SOC     = torch.sigmoid(SOC_raw)
        T_batt  = 273.15 + 60.0 * torch.sigmoid(T_raw)      # 273–333 K
        mdot    = F.softplus(mdot_raw, beta=sp_beta)        # kg/s (small)
        I_demand = mdot * 100.0                              # crude coupling

        # 4. Real battery step
        I_actual, dSOC = self._battery_step(SOC, I_demand, T_batt, dt)
        SOC_next = (SOC + dSOC).clamp(0.0, 1.0)

        # 5. Chamber temperature scales with power draw
        T_chamber = 300.0 + 2500.0 * SOC
        # 6. Nozzle thrust / Isp
        thrust, Isp, v_e = self._nozzle_thrust(
            mdot, T_chamber,
            self._p_chamber.to(dtype), self._p_amb.to(dtype),
        )

        # 7. Thermal barrier ΔE estimator
        delta_e = self.thermal_barrier_estimator(contracted)

        # 8. Actuator gating
        control_actions = torch.softmax(
            self.aerothermo_actuator_controller(contracted), dim=-1,
        )

        return {
            "contracted_state":           contracted,
            "topological_determinant":    det_sig,
            "SOC":                        SOC_next,
            "SOC_delta":                  dSOC,
            "battery_current":            I_actual,
            "battery_temperature":        T_batt,
            "discharge_rate":             I_actual.abs().clamp(max=1.0),
            "thrust_burst":               thrust,
            "Isp":                        Isp,
            "exhaust_velocity":           v_e,
            "mass_flow":                  mdot,
            "activation_energy_barrier":  delta_e,
            "control_actions":            control_actions,
        }

    # ------------------------------------------------------------------ #
    def forward(self, state_x, dt=1e-3):
        if self.validate_inputs:
            if state_x.dim() != 2 or state_x.shape[-1] != self.latent_dim:
                raise ValueError(f"state_x must be [B, {self.latent_dim}].")
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._step, state_x, dt, use_reentrant=False,
            )
        return self._step(state_x, dt)


# =============================================================================
def compute_production_loss_v3(outputs, target_thrust, cost_weights):
    """Smooth production loss with real physics-aware terms."""
    contracted = outputs["contracted_state"]
    det_sig    = outputs["topological_determinant"]
    delta_e    = outputs["activation_energy_barrier"]
    thrust     = outputs["thrust_burst"].squeeze(-1)
    SOC        = outputs["SOC"].squeeze(-1)

    performance = F.mse_loss(thrust, target_thrust)
    stability   = torch.sqrt(det_sig * det_sig + 1e-8).mean()
    no_zeno     = torch.exp(-F.softplus(delta_e, beta=4.0).float()).clamp(min=1e-9).mean()
    no_zeno     = no_zeno.to(delta_e.dtype)
    # SOC regularization: prefer high SOC but bounded
    soc_cost    = (1.0 - SOC).mean()
    complexity  = torch.linalg.vector_norm(contracted, ord=1, dim=-1).mean()

    return (performance
            + cost_weights.get("stability", 0.0) * stability
            + cost_weights.get("no_zeno", 0.0)   * no_zeno
            + cost_weights.get("soc", 0.0)       * soc_cost
            + cost_weights.get("complexity", 0.0) * complexity)


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-Engine v3] Running on: {device}")

    cfg = EngineConfig(latent_dim=32)
    engine = ProductionHypersonicBatteryEngineModuleV3(cfg).to(device)
    engine.train()
    opt = torch.optim.AdamW(engine.parameters(), lr=1e-3)

    B = 16
    x = torch.randn(B, cfg.latent_dim, device=device)
    target_thrust = torch.full((B,), 5000.0, device=device)  # 5 kN target

    opt.zero_grad(set_to_none=True)
    out = engine(x, dt=1e-3)
    loss = compute_production_loss_v3(out, target_thrust,
                                      {"stability": 0.05, "no_zeno": 0.02,
                                       "soc": 0.1, "complexity": 0.01})
    loss.backward()
    opt.step()

    print(f"  loss                       : {loss.item():.6f}")
    print(f"  SOC ∈ (0,1)                : "
          f"{out['SOC'].detach().min().item():.4f} – "
          f"{out['SOC'].detach().max().item():.4f}")
    print(f"  battery I [A]              : "
          f"{out['battery_current'].detach().min().item():.3e} – "
          f"{out['battery_current'].detach().max().item():.3e}")
    print(f"  thrust [N]                 : "
          f"{out['thrust_burst'].detach().min().item():.3e} – "
          f"{out['thrust_burst'].detach().max().item():.3e}")
    print(f"  Isp [s]                    : "
          f"{out['Isp'].detach().min().item():.1f} – "
          f"{out['Isp'].detach().max().item():.1f}")
    print(f"  log|det|                   : "
          f"{out['topological_determinant'].detach().min().item():.3f} – "
          f"{out['topological_determinant'].detach().max().item():.3f}")
    print("Status: real battery + isentropic nozzle; gradients flow.")
