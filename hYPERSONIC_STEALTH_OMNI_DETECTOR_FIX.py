# =============================================================================
# HYPERSONIC STEALTH OMNI-DETECTOR — FULL PHYSICS EDITION
# v3.0.1 (FIXED: _c_light → registered buffer)
# =============================================================================
# Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID     : 0009-0008-2374-0788
# License   : MIT  |  Year : 2026
# =============================================================================

from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HypersonicStealthOmniDetectorPhysics"]


class PhysConst:
    k_B    = 1.380649e-23
    h_P    = 6.62607015e-34
    m_e    = 9.1093837015e-31
    m_p    = 1.67262192369e-27
    e_ch   = 1.602176634e-19
    eps0   = 8.8541878128e-12
    sigma_SB = 5.670374419e-8
    R_air  = 287.052874
    gamma_air = 1.4
    T_ref  = 273.15
    mu_ref = 1.716e-5
    S_suth = 110.4
    E_ion_N2 = 15.58 * e_ch
    E_ion_O2 = 12.07 * e_ch
    G_N    = 6.67430e-11
    c_light = 2.99792458e8


class HypersonicStealthOmniDetectorPhysics(nn.Module):

    def __init__(
        self,
        *,
        body_length_m: float = 3.0,
        nose_radius_m: float = 0.05,
        emissivity: float = 0.85,
        wall_temp_K: float = 1200.0,
        radar_freq_Hz: float = 10e9,
        radar_power_W: float = 1e6,
        radar_gain_dB: float = 40.0,
        radar_target_rcs_m2: float = 1.0,
        noise_figure_dB: float = 3.0,
        bandwidth_Hz: float = 1e6,
        barrier_beta: float = 25.0,
        zeno_clamp: float = 15.0,
        log_floor: float = -30.0,
        denom_eps: float = 1e-12,
        hidden_dim: int = 32,
        use_hard_ste: bool = True,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()

        def _b(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_L",       _b(body_length_m),      persistent=False)
        self.register_buffer("_Rn",      _b(nose_radius_m),      persistent=False)
        self.register_buffer("_eps_w",   _b(emissivity),         persistent=False)
        self.register_buffer("_Tw",      _b(wall_temp_K),        persistent=False)
        self.register_buffer("_f_r",     _b(radar_freq_Hz),      persistent=False)
        self.register_buffer("_P_tx",    _b(radar_power_W),      persistent=False)
        self.register_buffer("_G_dB",    _b(radar_gain_dB),      persistent=False)
        self.register_buffer("_rcs0",    _b(radar_target_rcs_m2),persistent=False)
        self.register_buffer("_NF_dB",   _b(noise_figure_dB),    persistent=False)
        self.register_buffer("_B_rx",    _b(bandwidth_Hz),       persistent=False)
        self.register_buffer("_beta",    _b(barrier_beta),       persistent=False)
        self.register_buffer("_zeno_max",_b(zeno_clamp),         persistent=False)
        self.register_buffer("_log_flr", _b(log_floor),          persistent=False)
        self.register_buffer("_eps",     _b(denom_eps),          persistent=False)

        self.register_buffer("_kB",  _b(PhysConst.k_B),  persistent=False)
        self.register_buffer("_hP",  _b(PhysConst.h_P),  persistent=False)
        self.register_buffer("_me",  _b(PhysConst.m_e),  persistent=False)
        self.register_buffer("_ech", _b(PhysConst.e_ch), persistent=False)
        self.register_buffer("_eps0",_b(PhysConst.eps0), persistent=False)
        self.register_buffer("_sig", _b(PhysConst.sigma_SB), persistent=False)
        self.register_buffer("_R",   _b(PhysConst.R_air), persistent=False)
        self.register_buffer("_gam", _b(PhysConst.gamma_air), persistent=False)
        self.register_buffer("_Tre", _b(PhysConst.T_ref), persistent=False)
        self.register_buffer("_mure",_b(PhysConst.mu_ref), persistent=False)
        self.register_buffer("_Ss",  _b(PhysConst.S_suth), persistent=False)
        self.register_buffer("_Eion",_b(PhysConst.E_ion_N2), persistent=False)

        # === FIXED: _c_light เป็น registered buffer ไม่ใช่ instance attribute ===
        self.register_buffer("_c_light", _b(PhysConst.c_light), persistent=False)

        self.w_aero   = nn.Parameter(torch.tensor(1.0))
        self.w_plasma = nn.Parameter(torch.tensor(1.0))
        self.w_em     = nn.Parameter(torch.tensor(1.0))
        self.w_therm  = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.use_hard_ste = use_hard_ste
        self.validate_inputs = validate_inputs

    def sutherland_viscosity(self, T: torch.Tensor) -> torch.Tensor:
        T   = T.clamp_min(1.0)
        Tre = self._Tre.to(T.dtype)
        S   = self._Ss.to(T.dtype)
        mu  = self._mure.to(T.dtype)
        return mu * (T / Tre).pow(1.5) * (Tre + S) / (T + S)

    def speed_of_sound(self, T: torch.Tensor) -> torch.Tensor:
        g = self._gam.to(T.dtype); R = self._R.to(T.dtype)
        return torch.sqrt(g * R * T + 1e-12)

    def mach_number(self, v: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        v_mag = torch.sqrt((v * v).sum(dim=1, keepdim=True) + 1e-12)
        return v_mag / self.speed_of_sound(T)

    def rankine_hugoniot(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        g = self._gam.to(M.dtype); M2 = M * M
        p_ratio = (2 * g * M2 - (g - 1)) / (g + 1)
        T_ratio = ((2 * g * M2 - (g - 1)) * ((g - 1) * M2 + 2)) / ((g + 1) ** 2 * M2 + 1e-12)
        rho_ratio = ((g + 1) * M2) / ((g - 1) * M2 + 2)
        mach_angle = torch.asin(torch.clamp(1.0 / M.clamp_min(1.0001), max=1.0))
        return {"p_ratio": p_ratio, "T_ratio": T_ratio,
                "rho_ratio": rho_ratio, "mach_angle": mach_angle}

    def saha_ionization(self, T: torch.Tensor, n_total: torch.Tensor) -> torch.Tensor:
        T = T.clamp_min(300.0)
        kB  = self._kB.to(T.dtype);  hP  = self._hP.to(T.dtype)
        me  = self._me.to(T.dtype);  Eio = self._Eion.to(T.dtype)
        thermal_de_Broglie = (2 * math.pi * me * kB * T) / (hP * hP)
        rhs = thermal_de_Broglie.pow(1.5) * 2.0 * torch.exp(-Eio / (kB * T))
        K = rhs / (n_total + 1e-30)
        x = (-K + torch.sqrt(K * K + 4 * K)) / 2.0
        return x.clamp(0.0, 1.0)

    def plasma_frequency(self, n_e: torch.Tensor) -> torch.Tensor:
        e = self._ech.to(n_e.dtype); me = self._me.to(n_e.dtype)
        eps0 = self._eps0.to(n_e.dtype)
        return torch.sqrt((n_e * e * e) / (eps0 * me + 1e-30) + 1e-12)

    def debye_length(self, T_e: torch.Tensor, n_e: torch.Tensor) -> torch.Tensor:
        eps0 = self._eps0.to(T_e.dtype); kB = self._kB.to(T_e.dtype)
        e    = self._ech.to(T_e.dtype)
        return torch.sqrt((eps0 * kB * T_e) / (n_e * e * e + 1e-30) + 1e-12)

    def fay_riddell_heat_flux(
        self,
        rho_e: torch.Tensor,
        mu_e: torch.Tensor,
        T_e: torch.Tensor,
        v_inf: torch.Tensor,
        h_0: torch.Tensor,
        h_w: torch.Tensor,
        Pr: float = 0.71,
    ) -> torch.Tensor:
        Rn = self._Rn.to(rho_e.dtype)
        du_dx = (v_inf.abs() / (2.0 * Rn)).clamp_min(1e-6)
        return 0.763 * Pr ** (-0.6) * torch.sqrt(rho_e * mu_e) \
               * torch.sqrt(du_dx) * (h_0 - h_w).clamp_min(0.0)

    def radiative_equilibrium_temp(self, q_conv: torch.Tensor) -> torch.Tensor:
        sig = self._sig.to(q_conv.dtype); eps = self._eps_w.to(q_conv.dtype)
        return (q_conv.clamp_min(0.0) / (eps * sig + 1e-30)).pow(0.25)

    def radar_snr(
        self,
        range_m: torch.Tensor,
        rcs_eff: torch.Tensor,
        T_sys_K: torch.Tensor = None,
    ) -> torch.Tensor:
        # === FIXED: ใช้ self._c_light buffer แทน torch.tensor ทุกครั้ง ===
        c = self._c_light.to(range_m.dtype)
        f = self._f_r.to(range_m.dtype)
        lam = c / f
        G = 10.0 ** (self._G_dB.to(range_m.dtype) / 10.0)
        P_t = self._P_tx.to(range_m.dtype)
        B = self._B_rx.to(range_m.dtype)
        F_nf = 10.0 ** (self._NF_dB.to(range_m.dtype) / 10.0)
        kB = self._kB.to(range_m.dtype)
        if T_sys_K is None:
            T_sys_K = torch.full_like(range_m, 290.0)
        P_r = P_t * G * G * lam * lam * rcs_eff \
              / ((4 * math.pi) ** 3 * range_m.pow(4) + 1e-30)
        N   = kB * T_sys_K * B * F_nf
        return P_r / N

    def planck_radiance(self, T: torch.Tensor, lam_m: torch.Tensor) -> torch.Tensor:
        hP = self._hP.to(T.dtype)
        c = self._c_light.to(T.dtype)     # FIXED: buffer
        kB = self._kB.to(T.dtype)
        num = 2 * hP * c * c / (lam_m.pow(5) + 1e-30)
        den = torch.expm1((hP * c) / (lam_m * kB * T + 1e-30))
        return num / (den + 1e-30)

    def forward(
        self,
        velocity_field: torch.Tensor,
        thermal_field: torch.Tensor,
        pressure_field: torch.Tensor,
        density_field: torch.Tensor,
        em_field: torch.Tensor,
        emf_coupling: torch.Tensor,
        *,
        range_m: torch.Tensor = None,
        dt: float = 1e-3,
        return_metrics: bool = True,
    ) -> Dict[str, torch.Tensor]:

        dtype = velocity_field.dtype
        dev   = velocity_field.device

        M   = self.mach_number(velocity_field, thermal_field)
        RH  = self.rankine_hugoniot(M)
        mu  = self.sutherland_viscosity(thermal_field)
        a   = self.speed_of_sound(thermal_field)

        T2  = thermal_field * RH["T_ratio"]
        p2  = pressure_field * RH["p_ratio"]
        rho2 = density_field * RH["rho_ratio"]

        g = self._gam.to(dtype); R = self._R.to(dtype)
        cp = g * R / (g - 1.0)
        h0 = cp * T2 + 0.5 * (velocity_field * velocity_field).sum(dim=1, keepdim=True)
        hw = cp * self._Tw.to(dtype)

        v_mag = torch.sqrt((velocity_field * velocity_field).sum(dim=1, keepdim=True) + 1e-12)
        q_s = self.fay_riddell_heat_flux(rho2, mu, T2, v_mag, h0, hw)
        T_w_eq = self.radiative_equilibrium_temp(q_s)

        kB = self._kB.to(dtype)
        n_total = p2 / (kB * T2 + 1e-30)
        x_ion = self.saha_ionization(T2, n_total)
        n_e   = (x_ion * n_total).clamp_min(0.0)
        w_p   = self.plasma_frequency(n_e)
        lam_D = self.debye_length(T2, n_e + 1e-6)

        f_rad = self._f_r.to(dtype)
        f_p   = w_p / (2 * math.pi)
        rcs_plasma = self._rcs0.to(dtype) \
                     * torch.sigmoid((f_rad - f_p) / (f_rad * 0.1 + 1e-6))

        if range_m is None:
            range_m = torch.full_like(n_e, 100_000.0)
        else:
            range_m = range_m.to(dtype).view(-1, 1, 1, 1, 1) if range_m.dim() == 1 else range_m
        T_sys = torch.full_like(n_e, 290.0)
        snr   = self.radar_snr(range_m, rcs_plasma, T_sys)

        lam_ir = torch.full_like(T_w_eq, 4.3e-6)
        L_ir   = self.planck_radiance(T_w_eq, lam_ir)

        em_pow = (em_field * em_field).mean(dim=1, keepdim=True)
        emf_sq = emf_coupling * emf_coupling

        sd = (2, 3, 4)
        f1 = (q_s      ).mean(dim=sd) * self.w_aero
        f2 = (n_e      ).mean(dim=sd) * self.w_plasma
        f3 = (em_pow   ).mean(dim=sd) * self.w_em
        f4 = (emf_sq   ).mean(dim=sd)
        f5 = (T_w_eq   ).mean(dim=sd) * self.w_therm
        f6 = (snr      ).mean(dim=sd)
        f7 = (L_ir     ).mean(dim=sd)
        f8 = (lam_D    ).mean(dim=sd)

        feats = torch.stack([f1, f2, f3, f4, f5, f6, f7, f8], dim=-1)

        logits = self.head(feats)
        tau = 0.05
        soft = torch.sigmoid(logits / tau)
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
            g_noise = -torch.log(-torch.log(u))
            y_soft = torch.sigmoid((logits + g_noise) / tau)
            y_hard = (y_soft > 0.5).to(y_soft.dtype)
            pred = (y_hard - y_soft.detach() + y_soft) if self.use_hard_ste else y_soft
        else:
            pred = soft

        if not return_metrics:
            return {"is_hypersonic_drone": pred}

        return {
            "is_hypersonic_drone":        pred,
            "classification_confidence":  soft,
            "extracted_features":         feats,
            "mach_number":                M,
            "stagnation_heat_flux_W_m2":  q_s,
            "radiative_wall_temp_K":      T_w_eq,
            "electron_density_m3":        n_e,
            "ionization_fraction":        x_ion,
            "plasma_frequency_rad_s":     w_p,
            "debye_length_m":             lam_D,
            "radar_snr":                  snr,
            "ir_radiance":                L_ir,
            "logits":                     logits,
        }


# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-HypersonicDetectorFullPhysics] Running on: {device}")

    model = HypersonicStealthOmniDetectorPhysics().to(device)
    model.train()

    B, D, H, W = 2, 8, 8, 8
    velocity = torch.randn(B, 3, D, H, W, device=device) * 1500.0
    thermal  = torch.randn(B, 1, D, H, W, device=device) * 20.0 + 250.0
    pressure = torch.rand(B, 1, D, H, W, device=device) * 1000.0 + 100.0
    density  = torch.rand(B, 1, D, H, W, device=device) * 0.1 + 0.01
    em       = torch.randn(B, 3, D, H, W, device=device)
    emf      = torch.randn(B, 1, D, H, W, device=device) * 0.05

    out = model(velocity, thermal, pressure, density, em, emf, dt=1e-3)
    loss = out["is_hypersonic_drone"].float().mean() \
           + out["logits"].float().pow(2).mean()
    loss.backward()

    print(f"  loss                     : {loss.item():.6f}")
    print(f"  Mach range               : "
          f"{out['mach_number'].detach().min().item():.3f} – "
          f"{out['mach_number'].detach().max().item():.3f}")
    print(f"  stagnation q_s [W/m²]    : "
          f"{out['stagnation_heat_flux_W_m2'].detach().min().item():.3e} – "
          f"{out['stagnation_heat_flux_W_m2'].detach().max().item():.3e}")
    print(f"  n_e [m⁻³]                : "
          f"{out['electron_density_m3'].detach().min().item():.3e} – "
          f"{out['electron_density_m3'].detach().max().item():.3e}")
    print(f"  radar SNR                : "
          f"{out['radar_snr'].detach().min().item():.3e} – "
          f"{out['radar_snr'].detach().max().item():.3e}")
    print("Status: full physics chain executed; gradients flow.")
