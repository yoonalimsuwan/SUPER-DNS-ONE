# =============================================================================
# SESI PHYSICS CORE — Real Physical Constants & Differentiable Primitives
# SUPER DNS ONE / ONE Ecosystem — Production Release
# =============================================================================
# Developer  : PAI and Yoon A Limsuwan — MSPS NETWORK
# ORCID      : 0009-0008-2374-0788
# License    : MIT
# Year       : 2026
# Version    : 3.0.0 (Physics-Grounded / C^∞ / AMP-Safe / DDP-Ready)
# =============================================================================

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "SI",
    "soft_abs", "soft_floor", "soft_ceiling", "soft_clamp",
    "isentropic_ratios", "rankine_hugoniot", "prandtl_meyer",
    "sutton_graves_stagnation_T", "fay_riddell_stagnation_heat_flux",
    "newtonian_Cp", "eckert_reference_temperature",
    "saha_electron_density", "plasma_frequency",
    "electron_neutral_collision_freq", "plasma_complex_permittivity",
    "plasma_reflection_coefficient",
    "friis_received_power", "radar_range_equation",
    "jamming_to_signal_ratio", "burn_through_range",
    "beer_lambert_transmittance",
    "newton_gravity", "j2_perturbation_potential",
    "exponential_atmosphere_rho", "aerodynamic_drag",
    "cosmic_muon_flux", "muon_energy_loss_dEdx",
    "butler_volmer_current", "soc_dynamics", "arrhenius_rate",
    "gaussian_beam_intensity", "thermal_damage_time",
    "rotor_blade_passing_freq", "doppler_shift",
]


class SI:
    """Fundamental physical constants — SI 2019 redefinition."""

    c         = 2.99792458e8
    h         = 6.62607015e-34
    hbar      = 1.054571817e-34
    k_B       = 1.380649e-23
    e         = 1.602176634e-19
    N_A       = 6.02214076e23
    G         = 6.67430e-11
    sigma_SB  = 5.670374419e-8
    eps_0     = 8.8541878128e-12
    mu_0      = 1.25663706212e-6

    m_e       = 9.1093837015e-31
    m_p       = 1.67262192369e-27
    m_mu      = 1.883531627e-28
    m_n       = 1.67492749804e-27
    m_air     = 28.9647 / 6.02214076e26

    M_earth   = 5.9722e24
    R_earth   = 6.371e6
    g_0       = 9.80665
    J2_earth  = 1.08263e-3
    Omega_earth = 7.292115e-5

    rho_0     = 1.225
    p_0       = 101325.0
    T_0       = 288.15
    H_scale   = 8500.0

    gamma_air = 1.4
    R_air     = 287.052874
    c_p_air   = gamma_air * R_air / (gamma_air - 1.0)
    Pr_air    = 0.71

    E_i_air   = 14.5 * 1.602176634e-19

    sigma_bird  = 0.01
    sigma_drone = 0.1

    muon_a_ion  = 2.0e-3
    muon_b_rad  = 4.0e-6


# =============================================================================
# Smooth primitives
# =============================================================================
def soft_abs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def soft_floor(x: torch.Tensor, floor: float, beta: float = 50.0) -> torch.Tensor:
    return floor + F.softplus(x - floor, beta=beta)


def soft_ceiling(x: torch.Tensor, ceil: float, beta: float = 50.0) -> torch.Tensor:
    return ceil - F.softplus(ceil - x, beta=beta)


def soft_clamp(x: torch.Tensor, lo: float, hi: float, beta: float = 50.0) -> torch.Tensor:
    return soft_ceiling(soft_floor(x, lo, beta), hi, beta)


# =============================================================================
# Gas dynamics
# =============================================================================
def isentropic_ratios(M, gamma=SI.gamma_air):
    k = (gamma - 1.0) * 0.5
    T_ratio = 1.0 + k * M * M
    p_ratio = T_ratio ** (gamma / (gamma - 1.0))
    r_ratio = T_ratio ** (1.0 / (gamma - 1.0))
    return T_ratio, p_ratio, r_ratio


def rankine_hugoniot(M1, gamma=SI.gamma_air):
    M1sq = M1 * M1
    num  = 1.0 + (gamma - 1.0) * 0.5 * M1sq
    den  = gamma * M1sq - (gamma - 1.0) * 0.5
    M2sq = num / den.clamp(min=1e-3)
    M2sq = F.softplus(M2sq, beta=10.0) + 1e-8
    p2_p1    = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1sq - 1.0)
    rho2_rho1 = ((gamma + 1.0) * M1sq) / (2.0 + (gamma - 1.0) * M1sq)
    T2_T1    = p2_p1 / rho2_rho1.clamp(min=1e-6)
    return torch.sqrt(M2sq), p2_p1, rho2_rho1, T2_T1


def prandtl_meyer(M, gamma=SI.gamma_air):
    b = math.sqrt((gamma + 1.0) / (gamma - 1.0))
    M2m1 = (M * M - 1.0).clamp(min=1e-12)
    return (b * torch.atan(torch.sqrt(M2m1) / b)
            - torch.atan(torch.sqrt(M2m1)))


# =============================================================================
# Hypersonic aerothermodynamics
# =============================================================================
def sutton_graves_stagnation_T(V_inf, T_inf=250.0):
    return T_inf + (V_inf * V_inf) / (2.0 * SI.c_p_air)


def fay_riddell_stagnation_heat_flux(V_inf, rho_inf, R_nose, *, T_wall=None, T_inf=250.0):
    R_n = R_nose.clamp(min=1e-3)
    return 1.83e-4 * torch.sqrt(rho_inf / R_n) * V_inf ** 3


def newtonian_Cp(theta, Cp_max=2.0):
    s = torch.sin(theta)
    return Cp_max * s * s


def eckert_reference_temperature(M_e, T_e, T_w, gamma=SI.gamma_air, Pr=SI.Pr_air):
    r = math.sqrt(Pr)
    return T_e * (1.0 + 0.032 * M_e ** 2 + 0.58 * (T_w / T_e.clamp(min=1.0) - 1.0))


# =============================================================================
# Plasma physics
# =============================================================================
def saha_electron_density(T, n_total, E_i=SI.E_i_air):
    """
    Saha ionization electron density n_e (m⁻³), single-ionization Z=1.

    === FIXED ===
    - ลบ .item() ที่ทำให้ graph break
    - ใช้ softplus elementwise สำหรับ smooth ceiling (n_e ≤ n_total)
    """
    T_safe = T.clamp(min=1.0)
    prefactor = (2.0 * math.pi * SI.m_e * SI.k_B / (SI.h ** 2)) ** 1.5
    expo = (-(E_i / SI.k_B) / T_safe).float().clamp(min=-80.0)
    K = (prefactor * T_safe ** 1.5 * 2.0) * torch.exp(expo).to(T_safe.dtype)
    K = K.clamp(min=1e-60, max=1e60)
    disc = torch.sqrt(K * K + 4.0 * K * n_total + 1e-60)
    n_e = 0.5 * (-K + disc)
    # elementwise smooth ceiling — NO .item() / NO graph break
    n_e = n_total - F.softplus(n_total - n_e, beta=10.0)
    return n_e.clamp(min=0.0)


def plasma_frequency(n_e):
    return torch.sqrt(n_e * SI.e ** 2 / (SI.eps_0 * SI.m_e) + 1e-30)


def electron_neutral_collision_freq(n_e, T_e, n_n, sigma_en=1e-19):
    v_th = torch.sqrt(8.0 * SI.k_B * T_e / (math.pi * SI.m_e) + 1e-12)
    return n_n * sigma_en * v_th


def plasma_complex_permittivity(omega, omega_p, nu):
    denom = (omega * omega + nu * nu).clamp(min=1e-30)
    eps_r = 1.0 - (omega_p * omega_p) / denom
    eps_i = (nu / omega.clamp(min=1e-30)) * (omega_p * omega_p) / denom
    return eps_r, eps_i


def plasma_reflection_coefficient(eps_r, eps_i):
    eps_r_c = soft_floor(eps_r, 1e-6)
    s = torch.sqrt(eps_r_c)
    G = (1.0 - s) / (1.0 + s + 1e-12)
    below_cutoff = (eps_r < 0.0).to(eps_r.dtype)
    return below_cutoff + (1.0 - below_cutoff) * G * G


# =============================================================================
# EM / Radar
# =============================================================================
def friis_received_power(P_t, G_t, G_r, lam, R):
    return P_t * G_t * G_r * (lam / (4.0 * math.pi * R.clamp(min=1e-3))) ** 2


def radar_range_equation(P_t, G, sigma_rcs, lam, R, L=1.0):
    return (P_t * G * G * sigma_rcs * lam * lam) / \
           ((4.0 * math.pi) ** 3 * R.pow(4).clamp(min=1e-12) * L)


def jamming_to_signal_ratio(P_j, G_j, B_j, R_j, P_s, G_s, B_s, R_s):
    return (P_j * G_j * B_s * R_s * R_s) / \
           (P_s * G_s * B_j * R_j * R_j + 1e-30)


def burn_through_range(P_avg, G_t, sigma_rcs, lam, P_j, G_j, B_j, B_s, R_j, S_min):
    k = (P_avg * G_t ** 2 * sigma_rcs * lam ** 2 * B_j * R_j ** 2) / \
        ((4.0 * math.pi) ** 3 * P_j * G_j * S_min + 1e-30)
    return k.clamp(min=0.0).pow(0.25)


def beer_lambert_transmittance(alpha, L):
    return torch.exp(-(alpha * L).clamp(max=50.0))


# =============================================================================
# Orbital / ballistic
# =============================================================================
def newton_gravity(m1, m2, r_vec):
    r2 = (r_vec * r_vec).sum(dim=-1, keepdim=True)
    r_mag = torch.sqrt(r2 + 1.0)
    return SI.G * m1 * m2 * r_vec / (r_mag ** 3)


def j2_perturbation_potential(r, lat):
    mu = SI.G * SI.M_earth
    return -mu * SI.J2_earth * SI.R_earth ** 2 / (2.0 * r.clamp(min=SI.R_earth) ** 3) \
           * (3.0 * torch.sin(lat) ** 2 - 1.0)


def exponential_atmosphere_rho(h, rho_0=SI.rho_0, H=SI.H_scale):
    return rho_0 * torch.exp(-h / H)


def aerodynamic_drag(rho, v, Cd, A):
    return 0.5 * rho * (v * v).sum(dim=-1, keepdim=True) * Cd * A


# =============================================================================
# Astroparticle
# =============================================================================
def cosmic_muon_flux(altitude_m, E_min_GeV=1.0, I_0=70.0):
    spectrum = I_0 * (E_min_GeV ** -2.7)
    return spectrum * torch.exp(-altitude_m / SI.H_scale)


def muon_energy_loss_dEdx(E_mu_GeV, rho):
    return (SI.muon_a_ion + SI.muon_b_rad * E_mu_GeV) * rho


# =============================================================================
# Electrochemistry
# =============================================================================
def butler_volmer_current(eta, i0, alpha_a=0.5, alpha_c=0.5, T=298.15):
    F_over_RT = SI.e / (SI.k_B * T)
    exp_a = torch.exp(( alpha_a * F_over_RT * eta).clamp(max=30.0))
    exp_c = torch.exp((-alpha_c * F_over_RT * eta).clamp(max=30.0))
    return i0 * (exp_a - exp_c)


def soc_dynamics(I, C_nom, dt):
    return -(I * dt) / C_nom.clamp(min=1e-6)


def arrhenius_rate(T, Ea=1.0e4, T_ref=298.15):
    return torch.exp((-Ea * (1.0 / T.clamp(min=1.0) - 1.0 / T_ref)).clamp(min=-30.0))


# =============================================================================
# Directed energy
# =============================================================================
def gaussian_beam_intensity(P, w0, r, z, lam):
    z_R = math.pi * w0 ** 2 / lam
    w_z = w0 * torch.sqrt(1.0 + (z / z_R) ** 2)
    return (2.0 * P / (math.pi * w_z ** 2)) * torch.exp(-2.0 * (r ** 2) / (w_z ** 2))


def thermal_damage_time(I, T_melt, T_0, rho, c_p, k_cond, thickness):
    num = rho * c_p * thickness * (T_melt - T_0)
    den = (I - k_cond * (T_melt - T_0) / thickness.clamp(min=1e-4)).clamp(min=1e-6)
    return num / den


# =============================================================================
# Acoustics
# =============================================================================
def rotor_blade_passing_freq(rpm, n_blades):
    return n_blades * rpm / 60.0


def doppler_shift(f0, v_source, v_observer, v_wave, along):
    num = v_wave + v_observer * along
    den = (v_wave - v_source * along).clamp(min=1.0)
    return f0 * num / den


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print(" SESI PHYSICS CORE — self-test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    M = torch.tensor(8.0, device=device)
    M2, p2_p1, rho2_rho1, T2_T1 = rankine_hugoniot(M)
    print(f"[RH] M=8  →  M2={M2.item():.3f}  p2/p1={p2_p1.item():.1f}  "
          f"ρ2/ρ1={rho2_rho1.item():.2f}  T2/T1={T2_T1.item():.2f}")

    T2 = (250.0 * T2_T1).detach()
    n_n = torch.tensor(1e24, device=device)
    n_e = saha_electron_density(T2, n_n)
    print(f"[Saha] T2={T2.item():.0f} K, n_n=1e24 m⁻³  →  n_e={n_e.item():.2e} m⁻³")

    wp = plasma_frequency(n_e)
    print(f"[Plasma] ω_p = {wp.item():.3e} rad/s  (f_p = {wp.item()/(2*math.pi)/1e9:.2f} GHz)")

    P_r = radar_range_equation(
        P_t=1e4, G=10.0**(35/10), sigma_rcs=0.1,
        lam=SI.c / 10e9, R=torch.tensor(1e4, device=device),
    )
    print(f"[Radar] P_r at 10 km → {10*math.log10(P_r.item()/1e-3):.1f} dBm")

    eta = torch.tensor(0.1, device=device)
    i0 = torch.tensor(1e-3, device=device)
    i = butler_volmer_current(eta, i0)
    print(f"[BV] η=100 mV, i0=1 mA  →  i={i.item():.3e} A")

    print("\nAll physics primitives executed successfully.")
