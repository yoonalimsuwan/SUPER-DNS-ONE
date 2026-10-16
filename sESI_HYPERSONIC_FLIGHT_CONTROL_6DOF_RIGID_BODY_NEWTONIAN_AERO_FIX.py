# =============================================================================
# SESI HYPERSONIC FLIGHT CONTROL — 6-DOF RIGID BODY + NEWTONIAN AERO
# Physics-grounded v4.0.1 (FIXED: leaf-tensor in __main__)
# =============================================================================
# Developer  : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# =============================================================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sesi_physics_core import (
    SI,
    exponential_atmosphere_rho,
    rankine_hugoniot,
    isentropic_ratios,
    soft_floor,
)

__all__ = [
    "Flight6DOFConfig",
    "AeroLogGumbelEngine",
    "SESIHypersonicFlightController6DOF",
]


@dataclass
class Flight6DOFConfig:
    mass: float = 1200.0
    Ixx: float = 120.0
    Iyy: float = 1500.0
    Izz: float = 1600.0
    S_ref: float = 2.5
    L_ref: float = 4.0
    Cd0: float = 0.03
    Cm_alpha: float = -1.5
    Cm_q: float = -8.0
    Cl_delta_e: float = 0.8
    max_deflection_deg: float = 20.0

    c1_geom: float = 1.2
    dt_control_loop: float = 1e-3
    gumbel_safety_cutoff: float = 1e-9
    gating_temperature: float = 10.0
    inner_clamp: float = 15.0
    log_floor: float = -40.0
    abs_eps: float = 1e-4
    barrier_beta: float = 4.0
    energy_scale: float = 0.05


class AeroLogGumbelEngine(nn.Module):
    """Log-domain double-exponential SESI barrier."""

    def __init__(self, config: Flight6DOFConfig) -> None:
        super().__init__()
        self.cfg = config

        def _buf(v):
            return torch.tensor(float(v), dtype=torch.float32)

        p_min = config.gumbel_safety_cutoff
        target_inner = math.log(-math.log(p_min) / config.c1_geom)
        self.register_buffer("_c1",        _buf(config.c1_geom),            persistent=False)
        self.register_buffer("_dt",        _buf(config.dt_control_loop),    persistent=False)
        self.register_buffer("_inner_max", _buf(config.inner_clamp),        persistent=False)
        self.register_buffer("_log_floor", _buf(config.log_floor),          persistent=False)
        self.register_buffer("_log_cutoff",_buf(math.log(p_min)),           persistent=False)
        self.register_buffer("_target_inner", _buf(target_inner),           persistent=False)

    def compute_log_bound(self, delta_e, sigma_sq):
        dtype_out = delta_e.dtype
        device = delta_e.device
        d32 = delta_e.float()
        s32 = sigma_sq.float()
        c1        = self._c1.to(device=device)
        dt        = self._dt.to(device=device)
        inner_max = self._inner_max.to(device=device)
        log_floor = self._log_floor.to(device=device)
        denom = soft_floor(s32 * dt, 1e-8)
        inner = (d32 / denom).clamp_(max=inner_max)
        log_prob = (-c1 * torch.exp(inner)).clamp_(min=log_floor)
        return log_prob.to(dtype_out), inner.to(dtype_out)


class SESIHypersonicFlightController6DOF(nn.Module):

    def __init__(
        self,
        config: Optional[Flight6DOFConfig] = None,
        *,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        cfg = config or Flight6DOFConfig()
        self.cfg = cfg
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        self.ev_engine = AeroLogGumbelEngine(cfg)

        def _buf(v):
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_m",  _buf(cfg.mass),         persistent=False)
        self.register_buffer("_I",  torch.tensor([cfg.Ixx, cfg.Iyy, cfg.Izz],
                                                 dtype=torch.float32), persistent=False)
        self.register_buffer("_S",  _buf(cfg.S_ref),        persistent=False)
        self.register_buffer("_L",  _buf(cfg.L_ref),        persistent=False)
        self.register_buffer("_Cd0", _buf(cfg.Cd0),         persistent=False)
        self.register_buffer("_Cm_a", _buf(cfg.Cm_alpha),   persistent=False)
        self.register_buffer("_Cm_q", _buf(cfg.Cm_q),       persistent=False)
        self.register_buffer("_Cl_de", _buf(cfg.Cl_delta_e),persistent=False)
        self.register_buffer("_max_defl", _buf(math.radians(cfg.max_deflection_deg)),
                             persistent=False)
        self.register_buffer("_gate_T", _buf(cfg.gating_temperature),  persistent=False)
        self.register_buffer("_energy_scale", _buf(cfg.energy_scale),  persistent=False)
        self.register_buffer("_abs_eps", _buf(cfg.abs_eps),            persistent=False)
        self.register_buffer("_beta", _buf(cfg.barrier_beta),          persistent=False)

    def _atmos(self, alt):
        rho = exponential_atmosphere_rho(alt)
        T   = torch.full_like(alt, SI.T_0) - 0.0065 * alt.clamp(min=0.0, max=11000.0)
        p   = rho * SI.R_air * T
        a   = torch.sqrt(SI.gamma_air * SI.R_air * T.clamp(min=1.0))
        return rho, p, T, a

    def _aero_forces_moments(self, state, delta_e, delta_r, delta_a):
        u, v, w = state[..., 3], state[..., 4], state[..., 5]
        alt = state[..., 2].clamp(min=0.0)
        rho, p, T, a = self._atmos(alt)

        V = torch.sqrt(u*u + v*v + w*w + 1e-6)
        M = V / a.clamp(min=1.0)
        q_inf = 0.5 * rho * V * V
        S = self._S.to(q_inf.dtype)

        alpha = torch.atan2(w, u.clamp(min=1.0))
        s_a = torch.sin(alpha)
        CL = 2.0 * s_a * s_a * torch.cos(alpha)
        CD = self._Cd0.to(q_inf.dtype) + 2.0 * s_a ** 3

        L = q_inf * S * CL + q_inf * S * self._Cl_de.to(q_inf.dtype) * delta_e
        D = q_inf * S * CD

        Fx = -D * torch.cos(alpha) + L * torch.sin(alpha)
        Fz = -D * torch.sin(alpha) - L * torch.cos(alpha)
        Fy = torch.zeros_like(Fx)
        F_body = torch.stack([Fx, Fy, Fz], dim=-1)

        q_rate = state[..., 9]
        L_ref = self._L.to(q_inf.dtype)
        Cm = (self._Cm_a.to(q_inf.dtype) * alpha
              + self._Cm_q.to(q_inf.dtype) * q_rate * L_ref / V.clamp(min=1.0)
              + self._Cl_de.to(q_inf.dtype) * delta_e)
        My = q_inf * S * L_ref * Cm
        M_body = torch.stack([torch.zeros_like(My),
                              My,
                              torch.zeros_like(My)], dim=-1)
        aux = {"q_inf": q_inf, "M": M, "alpha": alpha, "L": L, "D": D, "T": T}
        return F_body, M_body, aux

    def _rhs(self, state, delta_e, delta_r, delta_a, thrust):
        u, v, w = state[..., 3], state[..., 4], state[..., 5]
        phi, theta, psi = state[..., 6], state[..., 7], state[..., 8]
        p, q, r = state[..., 9], state[..., 10], state[..., 11]

        m = self._m.to(state.dtype)
        I = self._I.to(state.dtype)

        F_body, M_body, aux = self._aero_forces_moments(state, delta_e, delta_r, delta_a)

        g_vec_ned = torch.stack([torch.zeros_like(u), torch.zeros_like(u),
                                 torch.full_like(u, SI.g_0)], dim=-1)
        cph, sph = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(theta), torch.sin(theta)
        cps, sps = torch.cos(psi), torch.sin(psi)
        R_nb = torch.stack([
            torch.stack([ cth*cps,                    cth*sps,                    -sth], -1),
            torch.stack([ sph*sth*cps - cph*sps,      sph*sth*sps + cph*cps,      sph*cth], -1),
            torch.stack([ cph*sth*cps + sph*sps,      cph*sth*sps - sph*cps,      cph*cth], -1),
        ], dim=-2)
        g_body = (R_nb @ g_vec_ned.unsqueeze(-1)).squeeze(-1) * m

        F_thrust = torch.stack([thrust, torch.zeros_like(thrust),
                                torch.zeros_like(thrust)], dim=-1)
        F_total = F_body + g_body + F_thrust

        omega = torch.stack([p, q, r], dim=-1)
        u_vec = torch.stack([u, v, w], dim=-1)
        omega_cross_u = torch.cross(omega, u_vec, dim=-1)
        du = F_total / m - omega_cross_u

        dphi   = p + (q * sph + r * cph) * torch.tan(theta.clamp(-1.4, 1.4))
        dtheta = q * cph - r * sph
        dpsi   = (q * sph + r * cph) / torch.cos(theta.clamp(-1.4, 1.4))

        I_omega = I * omega
        omega_cross_Iomega = torch.cross(omega, I_omega, dim=-1)
        domega = (M_body - omega_cross_Iomega) / I

        dpos = (R_nb @ u_vec.unsqueeze(-1)).squeeze(-1)

        dstate = torch.cat([
            dpos,
            du,
            torch.stack([dphi, dtheta, dpsi], dim=-1),
            domega,
        ], dim=-1)
        return dstate, aux

    def _rk4(self, state, delta_e, delta_r, delta_a, thrust, dt, substeps=4):
        h = dt / substeps
        for _ in range(substeps):
            k1, aux = self._rhs(state, delta_e, delta_r, delta_a, thrust)
            k2, _   = self._rhs(state + 0.5*h*k1, delta_e, delta_r, delta_a, thrust)
            k3, _   = self._rhs(state + 0.5*h*k2, delta_e, delta_r, delta_a, thrust)
            k4, _   = self._rhs(state +     h*k3, delta_e, delta_r, delta_a, thrust)
            state = state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return state, aux

    def _gate_and_adjust(self, state, delta_e_cmd):
        u, w = state[..., 3], state[..., 5]
        alt  = state[..., 2].clamp(min=0.0)
        rho, _, _, a = self._atmos(alt)
        V = torch.sqrt(u*u + w*w + 1e-6)
        q_inf = 0.5 * rho * V * V

        e_current = q_inf * torch.abs(torch.atan2(w, u.clamp(min=1.0)))
        if self.training:
            noise = 0.05 * torch.randn_like(e_current)
        else:
            noise = torch.zeros_like(e_current)
        e_transition = q_inf * torch.abs(torch.atan2(w + noise, u.clamp(min=1.0)))

        delta_e = F.softplus(e_transition - e_current,
                             beta=self._beta.to(e_current.dtype)) + 1e-3
        sigma_sq = (q_inf / rho.clamp(min=1e-12)) + 1e-12

        log_P, inner = self.ev_engine.compute_log_bound(delta_e, sigma_sq)
        log_cutoff = self.ev_engine._log_cutoff.to(log_P.dtype)
        logits = (log_P - log_cutoff) * self._gate_T.to(log_P.dtype)
        gate = torch.sigmoid(logits)

        target_inner = self.ev_engine._target_inner.to(delta_e.dtype)
        dt = self.ev_engine._dt.to(delta_e.dtype)
        required = target_inner * (sigma_sq * dt)
        deficit = F.softplus(required - delta_e, beta=self._beta.to(delta_e.dtype))
        raw = deficit * self._energy_scale.to(delta_e.dtype)
        adjust = self._max_defl.to(delta_e.dtype) * torch.tanh(
            raw / self._max_defl.to(delta_e.dtype)
        )

        corrected = delta_e_cmd + gate * adjust
        corrected = self._max_defl.to(delta_e.dtype) * torch.tanh(
            corrected / self._max_defl.to(delta_e.dtype)
        )
        return corrected, gate, delta_e, log_P

    def forward(
        self,
        state: torch.Tensor,
        delta_e_cmd: torch.Tensor,
        delta_r_cmd: torch.Tensor = None,
        delta_a_cmd: torch.Tensor = None,
        thrust: torch.Tensor = None,
        dt: float = 1e-3,
        *,
        substeps: int = 4,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.validate_inputs:
            if state.dim() != 2 or state.shape[-1] != 12:
                raise ValueError("state must be [B, 12].")
            if delta_e_cmd.shape != (state.shape[0],):
                raise ValueError("delta_e_cmd must be [B].")

        B = state.shape[0]
        if delta_r_cmd is None:
            delta_r_cmd = torch.zeros(B, device=state.device, dtype=state.dtype)
        if delta_a_cmd is None:
            delta_a_cmd = torch.zeros(B, device=state.device, dtype=state.dtype)
        if thrust is None:
            thrust = torch.zeros(B, device=state.device, dtype=state.dtype)

        delta_e_corrected, gate, delta_e_barrier, log_P = self._gate_and_adjust(
            state, delta_e_cmd,
        )

        state_next, aux = self._rk4(
            state, delta_e_corrected, delta_r_cmd, delta_a_cmd,
            thrust, dt, substeps=substeps,
        )

        if not return_metrics:
            return {"state_next": state_next,
                    "delta_e_corrected": delta_e_corrected}
        return {
            "state_next":                    state_next,
            "delta_e_corrected":             delta_e_corrected,
            "zeno_intervention_gate":        gate,
            "aerodynamic_activation_energy": delta_e_barrier,
            "log_shock_transition_probability": log_P,
            "q_inf":                         aux["q_inf"],
            "mach":                          aux["M"],
            "alpha":                         aux["alpha"],
            "lift":                          aux["L"],
            "drag":                          aux["D"],
            "atmos_T":                       aux["T"],
        }


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-Flight6DOF v4] Running on: {device}")

    cfg = Flight6DOFConfig()
    ctrl = SESIHypersonicFlightController6DOF(cfg).to(device)
    ctrl.train()

    B = 4

    # FIXED: สร้าง tensor fill ค่าก่อน แล้วค่อย .requires_grad_(True)
    state = torch.zeros(B, 12, device=device)
    state[:, 2] = 30000.0
    state[:, 3] = 2000.0
    state = state.requires_grad_(True)

    delta_e_cmd = torch.zeros(B, device=device)
    thrust = torch.full((B,), 5000.0, device=device)

    out = ctrl(state, delta_e_cmd, thrust=thrust, dt=1e-3, substeps=4)
    loss = out["state_next"][:, 3].pow(2).mean() \
           + out["zeno_intervention_gate"].mean() \
           + out["delta_e_corrected"].pow(2).mean()
    loss.backward()

    print(f"  loss                : {loss.item():.6f}")
    print(f"  Mach                : "
          f"{out['mach'].detach().min().item():.3f} – "
          f"{out['mach'].detach().max().item():.3f}")
    print(f"  q_inf [Pa]          : "
          f"{out['q_inf'].detach().min().item():.3e} – "
          f"{out['q_inf'].detach().max().item():.3e}")
    print(f"  lift [N]            : "
          f"{out['lift'].detach().min().item():.3e} – "
          f"{out['lift'].detach().max().item():.3e}")
    print(f"  Zeno gate           : "
          f"{out['zeno_intervention_gate'].detach().min().item():.4f} – "
          f"{out['zeno_intervention_gate'].detach().max().item():.4f}")
    print(f"  State_next[alt]     : {out['state_next'][:, 2].detach().tolist()}")
    print(f"  State_next[Vx]      : {out['state_next'][:, 3].detach().tolist()}")
    print(f"  |∂L/∂state|         : {state.grad.norm().item():.6e}")
    print("Status: 6-DOF RK4 integration + SESI gate; gradients flow.")
