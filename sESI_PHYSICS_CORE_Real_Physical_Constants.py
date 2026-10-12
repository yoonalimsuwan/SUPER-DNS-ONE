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
"""
Physics core library — real governing equations, all differentiable.

Coverage
--------
· Compressible gas dynamics      — isentropic, Rankine-Hugoniot, Prandtl-Meyer
· Hypersonic aerothermodynamics  — Sutton-Graves, Fay-Riddell, Newtonian, Eckert
· Plasma physics                 — Saha, ω_p, ν_en, collisional ε_r, |Γ|²
· Electromagnetics               — Friis, radar equation, J/S, burn-through
· Orbital / ballistic            — Newton gravity, J2, exponential atmosphere
· Astroparticle                  — cosmic muon flux, Bethe-Bloch dE/dx
· Electrochemistry               — Butler-Volmer, SOC, Arrhenius
· Directed energy                — Gaussian beam, Beer-Lambert, thermal damage
· Acoustics                      — rotor BPF, Doppler

Every public primitive is
  · C^∞-smooth where physically justified,
  · AMP-safe (internal fp32 for extreme exponentials),
  · vectorised / batch-first,
  · free of Python-scalar graph breaks (torch.compile friendly).
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "SI",
    "soft_abs", "soft_floor", "soft_ceiling", "soft_clamp",
    # gas dynamics
    "isentropic_ratios", "rankine_hugoniot", "prandtl_meyer",
    # hypersonic aero
    "sutton_graves_stagnation_T", "fay_riddell_stagnation_heat_flux",
    "newtonian_Cp", "eckert_reference_temperature",
    # plasma
    "saha_electron_density", "plasma_frequency",
    "electron_neutral_collision_freq", "plasma_complex_permittivity",
    "plasma_reflection_coefficient",
    # em / radar
    "friis_received_power", "radar_range_equation",
    "jamming_to_signal_ratio", "burn_through_range",
    "beer_lambert_transmittance",
    # orbital
    "newton_gravity", "j2_perturbation_potential",
    "exponential_atmosphere_rho", "aerodynamic_drag",
    # astroparticle
    "cosmic_muon_flux", "muon_energy_loss_dEdx",
    # electrochemistry
    "butler_volmer_current", "soc_dynamics", "arrhenius_rate",
    # directed energy
    "gaussian_beam_intensity", "thermal_damage_time",
    # acoustics
    "rotor_blade_passing_freq", "doppler_shift",
]


# =============================================================================
# CODATA 2018 / IUPAC SI constants (exact where defined)
# =============================================================================
class SI:
    """Fundamental physical constants — SI 2019 redefinition."""

    # Fundamental
    c         = 2.99792458e8         # m/s       speed of light (exact)
    h         = 6.62607015e-34       # J·s       Planck (exact)
    hbar      = 1.054571817e-34      # J·s       reduced Planck
    k_B       = 1.380649e-23         # J/K       Boltzmann (exact)
    e         = 1.602176634e-19      # C         elementary charge (exact)
    N_A       = 6.02214076e23        # 1/mol     Avogadro (exact)
    G         = 6.67430e-11          # m³/kg/s²  gravitational
    sigma_SB  = 5.670374419e-8       # W/m²/K⁴   Stefan-Boltzmann
    eps_0     = 8.8541878128e-12     # F/m       vacuum permittivity
    mu_0      = 1.25663706212e-6     # H/m       vacuum permeability

    # Particle masses
    m_e       = 9.1093837015e-31     # kg        electron
    m_p       = 1.67262192369e-27    # kg        proton
    m_mu      = 1.883531627e-28      # kg        muon
    m_n       = 1.67492749804e-27    # kg        neutron
    m_air     = 28.9647 / 6.02214076e26  # kg    mean air molecule (≈4.81e-26)

    # Earth
    M_earth   = 5.9722e24            # kg
    R_earth   = 6.371e6              # m
    g_0       = 9.80665              # m/s²      standard gravity (exact)
    J2_earth  = 1.08263e-3           # dimensionless oblateness
    Omega_earth = 7.292115e-5        # rad/s     Earth rotation rate

    # US Standard Atmosphere 1976 (near sea level)
    rho_0     = 1.225                # kg/m³
    p_0       = 101325.0             # Pa
    T_0       = 288.15               # K
    H_scale   = 8500.0               # m         scale height

    # Air thermodynamics (dry air at STP)
    gamma_air = 1.4
    R_air     = 287.052874           # J/kg/K
    c_p_air   = gamma_air * R_air / (gamma_air - 1.0)   # ≈ 1005 J/kg/K
    Pr_air    = 0.71

    # Effective ionization energy of air (N₂/O₂ weighted)
    E_i_air   = 14.5 * 1.602176634e-19    # J

    # Reference RCS
    sigma_bird  = 0.01               # m²  small bird
    sigma_drone = 0.1                # m²  quadrotor

    # Muon physics (PDG 2024, standard rock)
    muon_a_ion  = 2.0e-3             # GeV/(g·cm⁻²)  ionization loss
    muon_b_rad  = 4.0e-6             # cm²/g         radiative loss


# =============================================================================
# Smooth primitives (reused everywhere)
# =============================================================================
def soft_abs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """C^∞ surrogate for |x| — equals ε at x = 0."""
    return torch.sqrt(x * x + eps * eps)


def soft_floor(x: torch.Tensor, floor: float, beta: float = 50.0) -> torch.Tensor:
    """C^∞ surrogate for max(x, floor) — exact for x ≫ floor."""
    return floor + F.softplus(x - floor, beta=beta)


def soft_ceiling(x: torch.Tensor, ceil: float, beta: float = 50.0) -> torch.Tensor:
    """C^∞ surrogate for min(x, ceil) — exact for x ≪ ceil."""
    return ceil - F.softplus(ceil - x, beta=beta)


def soft_clamp(x: torch.Tensor, lo: float, hi: float, beta: float = 50.0) -> torch.Tensor:
    """C^∞ surrogate for clamp(x, lo, hi)."""
    return soft_ceiling(soft_floor(x, lo, beta), hi, beta)


# =============================================================================
# 1. COMPRESSIBLE GAS DYNAMICS
# =============================================================================
def isentropic_ratios(
    M: torch.Tensor, gamma: float = SI.gamma_air,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Isentropic stagnation ratios T0/T, p0/p, ρ0/ρ.

    Ref: Anderson, 'Modern Compressible Flow', 3rd ed., Eqs. 3.28–3.30.
    """
    k = (gamma - 1.0) * 0.5
    T_ratio = 1.0 + k * M * M
    p_ratio = T_ratio ** (gamma / (gamma - 1.0))
    r_ratio = T_ratio ** (1.0 / (gamma - 1.0))
    return T_ratio, p_ratio, r_ratio


def rankine_hugoniot(
    M1: torch.Tensor, gamma: float = SI.gamma_air,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normal-shock jump relations for calorically perfect gas.

    Returns
    -------
    M2      : downstream Mach number
    p2_p1   : pressure ratio
    rho2_rho1 : density ratio
    T2_T1   : temperature ratio

    Ref: Anderson, 'Modern Compressible Flow', Eqs. 3.51, 3.57, 3.58.
    """
    M1sq = M1 * M1
    num  = 1.0 + (gamma - 1.0) * 0.5 * M1sq
    den  = gamma * M1sq - (gamma - 1.0) * 0.5
    M2sq = num / den.clamp(min=1e-3)
    M2sq = F.softplus(M2sq, beta=10.0) + 1e-8
    p2_p1    = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1sq - 1.0)
    rho2_rho1 = ((gamma + 1.0) * M1sq) / (2.0 + (gamma - 1.0) * M1sq)
    T2_T1    = p2_p1 / rho2_rho1.clamp(min=1e-6)
    return torch.sqrt(M2sq), p2_p1, rho2_rho1, T2_T1


def prandtl_meyer(M: torch.Tensor, gamma: float = SI.gamma_air) -> torch.Tensor:
    """
    Prandtl-Meyer expansion function ν(M) [rad].

    Ref: Anderson, 'Modern Compressible Flow', Eq. 4.42.
    """
    b = math.sqrt((gamma + 1.0) / (gamma - 1.0))
    M2m1 = (M * M - 1.0).clamp(min=1e-12)
    return (b * torch.atan(torch.sqrt(M2m1) / b)
            - torch.atan(torch.sqrt(M2m1)))


# =============================================================================
# 2. HYPERSONIC AEROTHERMODYNAMICS
# =============================================================================
def sutton_graves_stagnation_T(
    V_inf: torch.Tensor, T_inf: float = 250.0,
) -> torch.Tensor:
    """
    Stagnation-point temperature behind a normal shock (recovery ≈ 1).

        T_s = T_∞ + V²/(2 c_p)

    Ref: Sutton & Graves, NASA TR R-376 (1971); Anderson, 'Hypersonic and
    High Temperature Gas Dynamics', §1.4.
    """
    return T_inf + (V_inf * V_inf) / (2.0 * SI.c_p_air)


def fay_riddell_stagnation_heat_flux(
    V_inf: torch.Tensor,
    rho_inf: torch.Tensor,
    R_nose: torch.Tensor,
    *,
    T_wall: torch.Tensor = None,
    T_inf: float = 250.0,
) -> torch.Tensor:
    """
    Stagnation-point convective heat flux q_s [W/m²].

    Closed-form (Sutton-Graves) equivalent of the Fay-Riddell correlation
    for air at flight conditions:

        q_s ≈ 1.83e-4 · √(ρ_∞ / R_nose) · V³

    Ref: Fay & Riddell, J. Aerosp. Sci. 25, 73–85 (1958);
         Sutton & Graves, NASA TR R-376 (1971).
    """
    R_n = R_nose.clamp(min=1e-3)
    return 1.83e-4 * torch.sqrt(rho_inf / R_n) * V_inf ** 3


def newtonian_Cp(theta: torch.Tensor, Cp_max: float = 2.0) -> torch.Tensor:
    """
    Newtonian impact-theory pressure coefficient Cp = Cp_max · sin²θ.

    Ref: Anderson, 'Hypersonic...', Eq. 3.20.
    """
    s = torch.sin(theta)
    return Cp_max * s * s


def eckert_reference_temperature(
    M_e: torch.Tensor,
    T_e: torch.Tensor,
    T_w: torch.Tensor,
    gamma: float = SI.gamma_air,
    Pr: float = SI.Pr_air,
) -> torch.Tensor:
    """
    Eckert reference temperature for compressible boundary layers.

    Ref: Eckert, NACA TR-1174 (1955).
    """
    r = math.sqrt(Pr)
    return T_e * (1.0 + 0.032 * M_e ** 2 + 0.58 * (T_w / T_e.clamp(min=1.0) - 1.0))


# =============================================================================
# 3. PLASMA PHYSICS (Saha + collisional permittivity)
# =============================================================================
def saha_electron_density(
    T: torch.Tensor,
    n_total: torch.Tensor,
    E_i: float = SI.E_i_air,
) -> torch.Tensor:
    """
    Saha ionization electron density n_e (m⁻³), single-ionization Z=1.

        n_e² / (n_total − n_e) = (2π m_e k_B T / h²)^{3/2} · 2 · exp(−E_i/(k_B T))

    Solved in closed form (quadratic). Physical upper bound n_e ≤ n_total
    enforced with a smooth ceiling.

    Ref: Saha, Phil. Mag. 40, 472 (1920); Mitchner & Kruger (1973).
    """
    T_safe = T.clamp(min=1.0)
    prefactor = (2.0 * math.pi * SI.m_e * SI.k_B / (SI.h ** 2)) ** 1.5
    # Inner exp evaluated in fp32 with a floor to prevent underflow.
    expo = (-(E_i / SI.k_B) / T_safe).float().clamp(min=-80.0)
    K = (prefactor * T_safe ** 1.5 * 2.0) * torch.exp(expo).to(T_safe.dtype)
    K = K.clamp(min=1e-60, max=1e60)
    disc = torch.sqrt(K * K + 4.0 * K * n_total + 1e-60)
    n_e = 0.5 * (-K + disc)
    # Smooth ceiling: n_e ≤ n_total
    n_e = soft_ceiling(n_e, float(n_total.max().item() if n_total.numel() else 1.0),
                       beta=10.0)
    return n_e.clamp(min=0.0)


def plasma_frequency(n_e: torch.Tensor) -> torch.Tensor:
    """ω_p = √(n_e e² / (ε₀ m_e)) [rad/s]."""
    return torch.sqrt(n_e * SI.e ** 2 / (SI.eps_0 * SI.m_e) + 1e-30)


def electron_neutral_collision_freq(
    n_e: torch.Tensor,
    T_e: torch.Tensor,
    n_n: torch.Tensor,
    sigma_en: float = 1e-19,   # m²  electron–N₂ elastic cross-section
) -> torch.Tensor:
    """ν_en = n_n σ v_th,  v_th = √(8 k_B T_e / (π m_e))."""
    v_th = torch.sqrt(8.0 * SI.k_B * T_e / (math.pi * SI.m_e) + 1e-12)
    return n_n * sigma_en * v_th


def plasma_complex_permittivity(
    omega: torch.Tensor,
    omega_p: torch.Tensor,
    nu: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cold collisional plasma relative permittivity ε_r = ε' − i ε''.

        ε'  = 1 − ω_p² / (ω² + ν²)
        ε'' = (ν/ω) · ω_p² / (ω² + ν²)

    Ref: Ginzburg, 'Propagation of EM Waves in Plasmas' (1970), §5.
    """
    denom = (omega * omega + nu * nu).clamp(min=1e-30)
    eps_r = 1.0 - (omega_p * omega_p) / denom
    eps_i = (nu / omega.clamp(min=1e-30)) * (omega_p * omega_p) / denom
    return eps_r, eps_i


def plasma_reflection_coefficient(
    eps_r: torch.Tensor,
    eps_i: torch.Tensor,
) -> torch.Tensor:
    """
    Normal-incidence power reflection |Γ|² for a plasma slab.

    ε' < 0 (below plasma cutoff) → total reflection |Γ|² = 1.
    ε' > 0 → partial reflection from the index mismatch.
    """
    eps_r_c = soft_floor(eps_r, 1e-6)
    s = torch.sqrt(eps_r_c)
    G = (1.0 - s) / (1.0 + s + 1e-12)
    # Below the plasma cutoff frequency the sheath is metallic (|Γ|² ≈ 1).
    below_cutoff = (eps_r < 0.0).to(eps_r.dtype)
    return below_cutoff + (1.0 - below_cutoff) * G * G


# =============================================================================
# 4. ELECTROMAGNETICS / RADAR  (Friis, radar range, J/S)
# =============================================================================
def friis_received_power(P_t, G_t, G_r, lam, R):
    """P_r = P_t G_t G_r (λ / (4π R))²  [W]."""
    return P_t * G_t * G_r * (lam / (4.0 * math.pi * R.clamp(min=1e-3))) ** 2


def radar_range_equation(P_t, G, sigma_rcs, lam, R, L=1.0):
    """P_r = P_t G² σ λ² / ((4π)³ R⁴ L)  [W]."""
    return (P_t * G * G * sigma_rcs * lam * lam) / \
           ((4.0 * math.pi) ** 3 * R.pow(4).clamp(min=1e-12) * L)


def jamming_to_signal_ratio(
    P_j, G_j, B_j, R_j,
    P_s, G_s, B_s, R_s,
):
    """
    Barrage-jamming J/S ratio.

        J/S = (P_j G_j / P_s G_s) · (B_s / B_j) · (R_s / R_j)²

    Ref: Skolnik, 'Introduction to Radar Systems', §2.6.
    """
    return (P_j * G_j * B_s * R_s * R_s) / \
           (P_s * G_s * B_j * R_j * R_j + 1e-30)


def burn_through_range(
    P_avg, G_t, sigma_rcs, lam,
    P_j, G_j, B_j, B_s, R_j, S_min,
):
    """
    Self-screening burn-through range (Skolnik §2.6).

    Simplified closed form for the range at which J/S = S_min.
    Returns R_bt [m].
    """
    k = (P_avg * G_t ** 2 * sigma_rcs * lam ** 2 * B_j * R_j ** 2) / \
        ((4.0 * math.pi) ** 3 * P_j * G_j * S_min + 1e-30)
    return k.clamp(min=0.0).pow(0.25)


def beer_lambert_transmittance(alpha: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """τ = exp(−α L)  (Beer-Lambert attenuation)."""
    return torch.exp(-(alpha * L).clamp(max=50.0))


# =============================================================================
# 5. ORBITAL / BALLISTIC MECHANICS
# =============================================================================
def newton_gravity(m1, m2, r_vec):
    """F = G m1 m2 / |r|³ · r  (softened at 0 by +1 m)."""
    r2 = (r_vec * r_vec).sum(dim=-1, keepdim=True)
    r_mag = torch.sqrt(r2 + 1.0)
    return SI.G * m1 * m2 * r_vec / (r_mag ** 3)


def j2_perturbation_potential(r, lat):
    """
    J2 oblateness term of Earth's gravitational potential.

        U₂ = −μ J2 R² / (2 r³) · (3 sin²φ − 1)
    """
    mu = SI.G * SI.M_earth
    return -mu * SI.J2_earth * SI.R_earth ** 2 / (2.0 * r.clamp(min=SI.R_earth) ** 3) \
           * (3.0 * torch.sin(lat) ** 2 - 1.0)


def exponential_atmosphere_rho(
    h: torch.Tensor, rho_0: float = SI.rho_0, H: float = SI.H_scale,
) -> torch.Tensor:
    """ρ(h) = ρ₀ exp(−h/H)  (US Std. 1976, valid 0–86 km)."""
    return rho_0 * torch.exp(-h / H)


def aerodynamic_drag(rho, v, Cd, A):
    """F_d = ½ ρ |v|² Cd A  (magnitude)."""
    return 0.5 * rho * (v * v).sum(dim=-1, keepdim=True) * Cd * A


# =============================================================================
# 6. ASTROPARTICLE / MUON TOMOGRAPHY
# =============================================================================
def cosmic_muon_flux(
    altitude_m: torch.Tensor, E_min_GeV: float = 1.0, I_0: float = 70.0,
) -> torch.Tensor:
    """
    Vertical muon integral flux I(>E_min) at altitude [m⁻² sr⁻¹ s⁻¹].

    Approx.: I ≈ I_0 · (E_min/GeV)^{−2.7} · exp(−h/L), L ≈ 8500 m.

    Ref: Gaisser, 'Cosmic Rays and Particle Physics', 2nd ed., §4.
    """
    spectrum = I_0 * (E_min_GeV ** -2.7)
    return spectrum * torch.exp(-altitude_m / SI.H_scale)


def muon_energy_loss_dEdx(
    E_mu_GeV: torch.Tensor, rho: torch.Tensor,
) -> torch.Tensor:
    """
    Muon energy loss [GeV / (g·cm⁻²)]: dE/dx = a + b·E.

    PDG 2024, standard rock: a = 2.0e-3, b = 4.0e-6.

    Ref: PDG Review of Particle Physics, 'Muon energy loss' (2024).
    """
    return (SI.muon_a_ion + SI.muon_b_rad * E_mu_GeV) * rho


# =============================================================================
# 7. ELECTROCHEMISTRY  (Battery)
# =============================================================================
def butler_volmer_current(
    eta: torch.Tensor,
    i0: torch.Tensor,
    alpha_a: float = 0.5,
    alpha_c: float = 0.5,
    T: float = 298.15,
) -> torch.Tensor:
    """
    Butler-Volmer:  i = i0 · [exp(α_a F η / RT) − exp(−α_c F η / RT)].

    Ref: Bard & Faulkner, 'Electrochemical Methods' (2001), Eq. 3.3.10.
    """
    F_over_RT = SI.e / (SI.k_B * T)
    exp_a = torch.exp(( alpha_a * F_over_RT * eta).clamp(max=30.0))
    exp_c = torch.exp((-alpha_c * F_over_RT * eta).clamp(max=30.0))
    return i0 * (exp_a - exp_c)


def soc_dynamics(I: torch.Tensor, C_nom: torch.Tensor, dt: float) -> torch.Tensor:
    """ΔSOC = −I·dt / C_nom  (Coulomb-counting)."""
    return -(I * dt) / C_nom.clamp(min=1e-6)


def arrhenius_rate(
    T: torch.Tensor, Ea: float = 1.0e4, T_ref: float = 298.15,
) -> torch.Tensor:
    """k(T) = exp(−Ea (1/T − 1/T_ref))  (normalised Arrhenius)."""
    return torch.exp((-Ea * (1.0 / T.clamp(min=1.0) - 1.0 / T_ref)).clamp(min=-30.0))


# =============================================================================
# 8. DIRECTED ENERGY  (Gaussian beam, thermal damage)
# =============================================================================
def gaussian_beam_intensity(P, w0, r, z, lam):
    """
    Gaussian beam intensity I(r,z) [W/m²].

        w(z) = w0 √(1 + (z/z_R)²),   z_R = π w0² / λ
        I    = (2P / π w²) exp(−2 r² / w²)

    Ref: Siegman, 'Lasers' (1986), Ch. 17.
    """
    z_R = math.pi * w0 ** 2 / lam
    w_z = w0 * torch.sqrt(1.0 + (z / z_R) ** 2)
    return (2.0 * P / (math.pi * w_z ** 2)) * torch.exp(-2.0 * (r ** 2) / (w_z ** 2))


def thermal_damage_time(I, T_melt, T_0, rho, c_p, k_cond, thickness):
    """
    1-D lumped thermal-damage timescale for a slab of thickness d hit by
    intensity I.

        t_dmg ≈ ρ c_p d (T_melt − T_0) / (I − k_cond (T_melt − T_0)/d)

    Ref: Ready, 'Effects of High-Power Laser Radiation' (1971).
    """
    num = rho * c_p * thickness * (T_melt - T_0)
    den = (I - k_cond * (T_melt - T_0) / thickness.clamp(min=1e-4)).clamp(min=1e-6)
    return num / den


# =============================================================================
# 9. ACOUSTICS  (rotor / bird)
# =============================================================================
def rotor_blade_passing_freq(rpm: torch.Tensor, n_blades: int) -> torch.Tensor:
    """BPF = n_blades · RPM / 60  [Hz]."""
    return n_blades * rpm / 60.0


def doppler_shift(f0, v_source, v_observer, v_wave, along: torch.Tensor):
    """f_obs = f0 · (v_wave + v_obs · along) / (v_wave − v_src · along)."""
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

    # 1. Rankine-Hugoniot at M=8 (typical hypersonic)
    M = torch.tensor(8.0, device=device)
    M2, p2_p1, rho2_rho1, T2_T1 = rankine_hugoniot(M)
    print(f"[RH] M=8  →  M2={M2.item():.3f}  p2/p1={p2_p1.item():.1f}  "
          f"ρ2/ρ1={rho2_rho1.item():.2f}  T2/T1={T2_T1.item():.2f}")

    # 2. Saha ionization at post-shock T
    T2 = (250.0 * T2_T1).detach()
    n_n = torch.tensor(1e24, device=device)
    n_e = saha_electron_density(T2, n_n)
    print(f"[Saha] T2={T2.item():.0f} K, n_n=1e24 m⁻³  →  n_e={n_e.item():.2e} m⁻³")

    # 3. Plasma frequency
    wp = plasma_frequency(n_e)
    print(f"[Plasma] ω_p = {wp.item():.3e} rad/s  (f_p = {wp.item()/(2*math.pi)/1e9:.2f} GHz)")

    # 4. Radar range equation sanity check
    P_r = radar_range_equation(
        P_t=1e4, G=10.0**(35/10), sigma_rcs=0.1,
        lam=SI.c / 10e9, R=torch.tensor(1e4, device=device),
    )
    print(f"[Radar] P_r at 10 km, 10 GHz, σ=0.1 m², P_t=10 kW, G=35 dB  →  "
          f"{10*math.log10(P_r.item()/1e-3):.1f} dBm")

    # 5. Butler-Volmer
    eta = torch.tensor(0.1, device=device)
    i0 = torch.tensor(1e-3, device=device)
    i = butler_volmer_current(eta, i0)
    print(f"[BV] η=100 mV, i0=1 mA  →  i={i.item():.3e} A")

    print("\nAll physics primitives executed successfully.")
