"""
FIRE ONE — Fire Dynamics Engineering Module for the ONE Ecosystem
====================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT

Fire engineering analysis, analogous in structure to seismic_one.py:
    1. DesignFireCurve       — t-squared HRR growth curves (NFPA-standard)
    2. PlumeCorrelations     — Heskestad/McCaffrey fire plume analytics
    3. MixtureFractionCombustion — Burke-Schumann fast-chemistry state relations
    4. RadiationModel        — point-source + optically-thin radiative loss
    5. TenabilityAssessment  — FED (CO, thermal), visibility/smoke extinction
    6. FireOne               — orchestrator

======================================================================
 LIFE-SAFETY NOTICE — READ BEFORE USE
======================================================================
Fire and tenability modeling directly informs evacuation timing and
firefighter safety decisions. Every correlation here is a published,
simplified engineering approximation (Heskestad, Purser, Jin, NFPA
design-fire curves) — the same class of tool used in the SFPE Handbook
and NIST FDS's own tenability post-processing, NOT a replacement for
validated CFD fire modeling (e.g. NIST FDS itself), professional fire
protection engineering review, or code-required life-safety analysis.
Do not use this module as the sole basis for an actual evacuation plan,
egress design, or firefighting decision. Treat all outputs as engineering
estimates requiring professional review, consistent with how this
ecosystem treats seismic_one.py's structural/liquefaction outputs.
======================================================================

Units: SI throughout (m, s, kg, K, W) unless noted otherwise.

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem attribution convention.
"""

from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Callable

__version__ = "1.0.0"
__all__ = [
    "DesignFireCurve",
    "PlumeCorrelations",
    "MixtureFractionCombustion",
    "RadiationModel",
    "TenabilityAssessment",
    "FireOne",
    "SootKinetics",
    "HeatDetectorModel",
    "SmokeDetectorModel",
    "PyrolysisModel",
]

G_ACCEL = 9.80665       # m/s^2
T_AMBIENT = 293.15      # K (20 C)
RHO_AMBIENT = 1.204     # kg/m^3 at 20C, 1 atm
CP_AIR = 1005.0         # J/(kg.K), ambient air -- used in Heskestad plume correlations
CP_COMBUSTION_PRODUCTS = 1400.0  # J/(kg.K), representative average for
                                   # combustion product gases integrated
                                   # from ambient to flame temperature


# =====================================================================
# 1. DESIGN FIRE CURVE  (t-squared growth)
# =====================================================================

class DesignFireCurve:
    """
    Standard t-squared fire growth model:  Q(t) = alpha * t^2   [kW, t in s]

    alpha values per NFPA 72 / SFPE-standard growth-rate categories
    (kW/s^2), reaching 1055 kW (1000 BTU/s) at the stated growth time:
        slow:       alpha = 0.00293   (1055 kW at ~600 s)
        medium:     alpha = 0.01172   (1055 kW at ~300 s)
        fast:       alpha = 0.0469    (1055 kW at ~150 s)
        ultrafast:  alpha = 0.1876    (1055 kW at ~75 s)
    """

    GROWTH_RATES = {
        "slow": 0.00293, "medium": 0.01172, "fast": 0.0469, "ultrafast": 0.1876,
    }

    def __init__(self, growth_rate: str = "medium", q_peak_kw: Optional[float] = None,
                 t_decay_start: Optional[float] = None):
        if growth_rate not in self.GROWTH_RATES:
            raise ValueError(f"growth_rate must be one of {list(self.GROWTH_RATES)}")
        self.alpha = self.GROWTH_RATES[growth_rate]
        self.growth_rate = growth_rate
        self.q_peak_kw = q_peak_kw
        self.t_decay_start = t_decay_start

    def hrr_kw(self, t: np.ndarray) -> np.ndarray:
        """
        Heat release rate [kW] at time t [s]. Growth phase is t-squared;
        if q_peak_kw is set, HRR plateaus there (fuel- or ventilation-
        limited steady burning); if t_decay_start is also set, HRR decays
        linearly to zero over the same duration as the growth phase took
        to reach peak (a common simplified decay assumption -- for a real
        fuel-controlled decay, replace with actual fuel-load burnout data).
        """
        t = np.asarray(t, dtype=float)
        q = self.alpha * t**2

        if self.q_peak_kw is not None:
            t_peak = np.sqrt(self.q_peak_kw / self.alpha)
            q = np.where(t < t_peak, q, self.q_peak_kw)
            if self.t_decay_start is not None:
                decay_duration = t_peak  # symmetric decay assumption
                decayed = self.q_peak_kw * np.clip(
                    1.0 - (t - self.t_decay_start) / decay_duration, 0.0, 1.0)
                q = np.where(t > self.t_decay_start, decayed, q)
        return q

    def time_to_hrr(self, q_kw: float) -> float:
        """Time [s] for the t-squared growth phase alone to reach q_kw."""
        return float(np.sqrt(q_kw / self.alpha))


# =====================================================================
# 2. PLUME CORRELATIONS  (Heskestad / McCaffrey)
# =====================================================================

class PlumeCorrelations:
    """
    Validated closed-form fire plume correlations (Heskestad 1983;
    McCaffrey 1979), as used throughout the SFPE Handbook and NIST FDS
    verification suites.
    """

    @staticmethod
    def flame_height(q_kw: float, D: float) -> float:
        """
        Heskestad mean flame height [m].
            Lf = 0.235 * Q^(2/5) - 1.02 * D
        Q in kW, D = effective diameter of the fire base [m].
        """
        return 0.235 * q_kw**0.4 - 1.02 * D

    @staticmethod
    def convective_hrr(q_kw: float, chi_r: float = 0.35) -> float:
        """
        Convective fraction of total HRR [kW]. chi_r = radiative fraction
        (0.2-0.4 typical for hydrocarbon pool/solid fires; SFPE Handbook
        default ~0.35 absent fuel-specific data).
        """
        return q_kw * (1.0 - chi_r)

    @classmethod
    def centerline_temperature_rise(cls, q_kw: float, z: float, D: float,
                                     chi_r: float = 0.35,
                                     z_origin_correction: bool = True) -> float:
        """
        Heskestad centerline excess temperature [K above ambient] at
        height z [m] above the fire base, valid in the plume region
        (z > flame height).

            dT0 = 9.1 * (T_inf / (g^2 * cp^2 * rho_inf^2))^(1/3)
                  * Qc^(2/3) * (z - z0)^(-5/3)

        z0 = virtual origin correction (Heskestad):
            z0 = 0.083*Q^(2/5) - 1.02*D
        """
        Qc = cls.convective_hrr(q_kw, chi_r) * 1000.0  # kW -> W
        z0 = 0.083 * q_kw**0.4 - 1.02 * D if z_origin_correction else 0.0
        dz = max(z - z0, 0.01)
        coeff = 9.1 * (T_AMBIENT / (G_ACCEL**2 * CP_AIR**2 * RHO_AMBIENT**2))**(1.0 / 3.0)
        return coeff * Qc**(2.0 / 3.0) * dz**(-5.0 / 3.0)

    @classmethod
    def centerline_velocity(cls, q_kw: float, z: float, D: float,
                             chi_r: float = 0.35,
                             z_origin_correction: bool = True) -> float:
        """
        Heskestad centerline plume velocity [m/s] at height z [m].
            u0 = 3.4 * (g / (cp * rho_inf * T_inf))^(1/3) * Qc^(1/3) * (z-z0)^(-1/3)
        """
        Qc = cls.convective_hrr(q_kw, chi_r) * 1000.0
        z0 = 0.083 * q_kw**0.4 - 1.02 * D if z_origin_correction else 0.0
        dz = max(z - z0, 0.01)
        coeff = 3.4 * (G_ACCEL / (CP_AIR * RHO_AMBIENT * T_AMBIENT))**(1.0 / 3.0)
        return coeff * Qc**(1.0 / 3.0) * dz**(-1.0 / 3.0)

    @staticmethod
    def mccaffrey_regime(q_kw: float, z: float) -> str:
        """
        Classifies plume height z into McCaffrey's three regimes via the
        normalized height z/Q^(2/5): continuous flame, intermittent, or
        buoyant plume (thermal).
        """
        zq = z / q_kw**0.4
        if zq < 0.08:
            return "continuous_flame"
        elif zq < 0.2:
            return "intermittent"
        else:
            return "buoyant_plume"


# =====================================================================
# 3. MIXTURE FRACTION COMBUSTION  (Burke-Schumann fast chemistry)
# =====================================================================

@dataclass
class FuelProperties:
    name: str
    formula_C: int          # carbon atoms per fuel molecule
    formula_H: int          # hydrogen atoms per fuel molecule
    formula_O: int = 0      # oxygen atoms per fuel molecule (0 for pure HC)
    delta_h_c: float = 4.0e7   # heat of combustion [J/kg], default ~propane-ish
    soot_yield: float = 0.015  # kg soot / kg fuel burned (typical HC ~0.01-0.1)
    co_yield: float = 0.02     # kg CO / kg fuel burned (under-ventilated higher)
    molar_mass: float = 44.1e-3  # kg/mol, default propane


# A few common fuels (typical literature yields; override for site-specific fuel)
COMMON_FUELS = {
    "propane":  FuelProperties("propane",  3, 8, 0, 4.637e7, 0.024, 0.005, 44.1e-3),
    "wood":     FuelProperties("wood",     1, 1.7, 0.7, 1.6e7, 0.015, 0.004, 27.0e-3),  # approx per-C basis
    "polyurethane_foam": FuelProperties("PU foam", 1, 1.8, 0.3, 2.3e7, 0.10, 0.03, 30.0e-3),
    "heptane":  FuelProperties("heptane",  7, 16, 0, 4.465e7, 0.037, 0.007, 100.2e-3),
}


class MixtureFractionCombustion:
    """
    Burke-Schumann (fast, infinitely-fast, diffusion-flame-sheet)
    combustion model: given a fuel's stoichiometry, computes the
    stoichiometric mixture fraction and piecewise-linear state relations
    for temperature and major species as a function of mixture fraction Z
    (Z=1 pure fuel, Z=0 pure air). This is the same modeling approach
    used by NIST FDS's default (non-finite-rate) combustion model.
    """

    def __init__(self, fuel: FuelProperties, y_o2_air: float = 0.233):
        self.fuel = fuel
        self.y_o2_air = y_o2_air

        # Stoichiometric O2 mass per unit fuel mass (complete combustion
        # to CO2 + H2O): CxHyOz + (x + y/4 - z/2) O2 -> x CO2 + (y/2) H2O
        nu_o2 = fuel.formula_C + fuel.formula_H / 4.0 - fuel.formula_O / 2.0
        self.o2_per_fuel_mass = nu_o2 * 32.0e-3 / fuel.molar_mass   # kg O2 / kg fuel
        self.r_stoich = self.o2_per_fuel_mass / y_o2_air             # kg air / kg fuel (stoich)
        self.z_stoich = 1.0 / (1.0 + self.r_stoich)

    def adiabatic_flame_temperature(self) -> float:
        """
        Simplified adiabatic flame temperature [K] at stoichiometric
        mixture fraction, from a single-step energy balance:
            T_ad = T_amb + delta_h_c / (cp_products * (1 + r_stoich))

        Uses CP_COMBUSTION_PRODUCTS (~1400 J/kg.K), a representative
        AVERAGE heat capacity for combustion product gases integrated
        from ambient to flame temperature -- NOT the room-temperature
        air value (CP_AIR=1005, correctly used elsewhere for the
        Heskestad plume correlations, which are validated specifically
        with ambient-air cp). Using ambient cp here was an earlier bug:
        cp of hot gas roughly doubles from 300K to 2500K, and using the
        room-temperature value overestimates T_ad by ~800K (predicting
        ~3080K vs. the well-known ~2260-2390K stoichiometric propane-air
        adiabatic flame temperature). This single-step balance still
        neglects dissociation, so it remains an upper-bound engineering
        estimate, not a chemical-equilibrium calculation.
        """
        return T_AMBIENT + self.fuel.delta_h_c / (CP_COMBUSTION_PRODUCTS * (1.0 + self.r_stoich))

    def state_relations(self, Z: np.ndarray) -> dict:
        """
        Burke-Schumann piecewise-linear state relations vs. mixture
        fraction Z ∈ [0,1].

        Fuel-lean side (Z < Z_st): O2 present, no fuel, products scale
        linearly with Z. Fuel-rich side (Z > Z_st): fuel present
        (unburned), no O2, products scale linearly down from stoichiometric
        peak to zero at Z=1 (pure fuel, no products -- adiabatic mixing
        with no oxidizer).

        Returns dict of arrays (same shape as Z): Y_fuel, Y_O2, Y_products,
        T (temperature via mixing-line + combustion energy release).
        """
        Z = np.clip(np.asarray(Z, dtype=float), 0.0, 1.0)
        Zst = self.z_stoich

        Y_O2 = np.where(Z < Zst, self.y_o2_air * (1.0 - Z / Zst), 0.0)
        Y_fuel = np.where(Z > Zst, (Z - Zst) / (1.0 - Zst), 0.0)
        Y_products = 1.0 - Y_O2 - Y_fuel - (1.0 - self.y_o2_air) * (1 - Z)  # simplified N2-exclusive accounting
        Y_products = np.clip(Y_products, 0.0, 1.0)

        # Temperature: linear mixing of sensible enthalpy + combustion
        # heat release, peaking at Z=Zst (Burke-Schumann flame sheet).
        T_ad = self.adiabatic_flame_temperature()
        T = np.where(
            Z < Zst,
            T_AMBIENT + (T_ad - T_AMBIENT) * (Z / Zst),
            T_AMBIENT + (T_ad - T_AMBIENT) * ((1.0 - Z) / (1.0 - Zst)),
        )

        return {"Z": Z, "Y_fuel": Y_fuel, "Y_O2": Y_O2, "Y_products": Y_products, "T": T}

    def mass_burning_rate_to_hrr(self, mdot_fuel_kg_s: float) -> float:
        """HRR [kW] from a fuel mass burning rate [kg/s]."""
        return mdot_fuel_kg_s * self.fuel.delta_h_c / 1000.0

    def hrr_to_yields(self, q_kw: float) -> dict:
        """
        Given total HRR, back out fuel consumption rate and CO/soot
        production rates via the fuel's yields.
        """
        mdot_fuel = q_kw * 1000.0 / self.fuel.delta_h_c   # kg/s
        return {
            "mdot_fuel_kg_s": mdot_fuel,
            "mdot_co_kg_s":   mdot_fuel * self.fuel.co_yield,
            "mdot_soot_kg_s": mdot_fuel * self.fuel.soot_yield,
        }


# =====================================================================
# 4. RADIATION MODEL
# =====================================================================

class RadiationModel:
    """
    Simplified radiative heat transfer: point-source model for far-field
    incident flux (standard SFPE Handbook engineering method) and
    optically-thin volumetric emission for CFD source-term coupling.
    """

    STEFAN_BOLTZMANN = 5.670374e-8   # W/(m^2.K^4)

    @staticmethod
    def point_source_flux(q_kw: float, chi_r: float, distance_m: float) -> float:
        """
        Incident radiant heat flux [kW/m^2] at a target `distance_m` from
        the fire, treating the fire as a point radiator (SFPE Handbook
        point-source method):
            q" = chi_r * Q / (4*pi*R^2)
        Valid for R greater than ~2.5x the fire diameter; under-predicts
        close to large fires (use a solid-flame model there instead).
        """
        Qr = chi_r * q_kw
        return Qr / (4.0 * np.pi * distance_m**2)

    @classmethod
    def critical_distance_for_flux(cls, q_kw: float, chi_r: float,
                                    critical_flux_kw_m2: float = 12.5) -> float:
        """
        Distance [m] at which incident flux falls to a critical threshold
        (12.5 kW/m^2 = piloted ignition of most common materials; 2.5
        kW/m^2 = tenable exposure limit for skin/personnel per SFPE).
        """
        Qr = chi_r * q_kw
        return np.sqrt(Qr / (4.0 * np.pi * critical_flux_kw_m2))

    @classmethod
    def optically_thin_volumetric_emission(cls, T_gas: np.ndarray,
                                             absorption_coeff: float = 0.3) -> np.ndarray:
        """
        Optically-thin-limit volumetric radiative loss [W/m^3]:
            q_rad''' = 4 * kappa * sigma * (T_gas^4 - T_amb^4)
        kappa = mean absorption coefficient [1/m] (soot-laden flame gases
        typically 0.1-1.0 /m; 0.3 /m is a common engineering default).
        This is the term that would populate a volumetric energy SINK in
        the DNS energy equation away from the flame sheet itself (net
        emission from hot gas to surroundings); the flame-sheet HRR itself
        is a separate SOURCE term (see FireOne.heat_release_source_field).
        """
        T_gas = np.asarray(T_gas, dtype=float)
        return 4.0 * absorption_coeff * cls.STEFAN_BOLTZMANN * (T_gas**4 - T_AMBIENT**4)


# =====================================================================
# 5. TENABILITY ASSESSMENT
# =====================================================================

class TenabilityAssessment:
    """
    Fractional Effective Dose (FED) tenability model (Purser; ISO 13571),
    the standard life-safety post-processing used with NIST FDS and in
    performance-based egress design. FED >= 1.0 predicts incapacitation
    for an average exposed occupant.

    LIFE-SAFETY CAVEAT: these are population-average, simplified
    correlations. Vulnerable individuals (children, elderly, those with
    respiratory/cardiac conditions) may be incapacitated well below
    FED=1.0. Do not treat FED=1.0 as a precise, individual-level safety
    threshold.
    """

    @staticmethod
    def fed_co_increment(co_ppm: float, dt_min: float, rmv_l_min: float = 25.0) -> float:
        """
        Incremental FED contribution from CO exposure over a time step
        (Purser's equation, as implemented in NIST FDS / ISO 13571):
            dFED_CO = (3.317e-5 * RMV * [CO]^1.036) * dt
        RMV = respiratory minute volume [L/min] (25 = light activity default).
        [CO] in ppm, dt in minutes.
        """
        return 3.317e-5 * rmv_l_min * co_ppm**1.036 * dt_min

    @staticmethod
    def fed_thermal_increment(T_celsius: float, dt_min: float) -> float:
        """
        Incremental FED from convective heat exposure (Purser):
            dFED_thermal = dt / t_I(T)
            t_I(T) = exp(5.1849 - 0.0273*T)   [min], T in Celsius
        """
        t_I = np.exp(5.1849 - 0.0273 * T_celsius)
        return dt_min / np.clip(t_I, 1e-6, None)

    @staticmethod
    def visibility_from_soot(soot_concentration_kg_m3: np.ndarray,
                              is_reflecting_sign: bool = False) -> np.ndarray:
        """
        Visibility [m] via Jin's correlation from smoke mass extinction:
            S = K_vis / (Cs * K_m)
        Cs = soot mass concentration [kg/m^3], K_m = mass extinction
        coefficient (~8.7 m^2/g for flaming combustion, per Mulholland/
        Jin), K_vis = 8 for light-emitting signs, 3 for reflecting signs.
        """
        K_m = 8700.0  # m^2/kg  (8.7 m^2/g)
        K_vis = 3.0 if is_reflecting_sign else 8.0
        Cs = np.clip(np.asarray(soot_concentration_kg_m3, dtype=float), 1e-9, None)
        return K_vis / (Cs * K_m)

    @classmethod
    def assess_timeline(cls, t: np.ndarray, co_ppm: np.ndarray, T_celsius: np.ndarray,
                         rmv_l_min: float = 25.0) -> dict:
        """
        Integrates FED_CO and FED_thermal over a full exposure timeline.

        Args:
            t: time array [s], must be monotonically increasing
            co_ppm, T_celsius: same-length arrays of conditions at each t
        Returns dict with cumulative FED arrays and time-to-incapacitation
        (None if FED never reaches 1.0 within the given timeline).
        """
        t = np.asarray(t, dtype=float)
        dt_min = np.diff(t, prepend=t[0]) / 60.0

        fed_co = np.cumsum([cls.fed_co_increment(c, d, rmv_l_min)
                             for c, d in zip(co_ppm, dt_min)])
        fed_thermal = np.cumsum([cls.fed_thermal_increment(T, d)
                                  for T, d in zip(T_celsius, dt_min)])
        fed_total = fed_co + fed_thermal

        idx_incap = np.argmax(fed_total >= 1.0) if np.any(fed_total >= 1.0) else None
        t_incap = float(t[idx_incap]) if idx_incap else None

        return {
            "time": t, "fed_co": fed_co, "fed_thermal": fed_thermal,
            "fed_total": fed_total, "time_to_incapacitation_s": t_incap,
        }


# =====================================================================
# 5b. SOOT KINETICS  (semi-empirical formation/oxidation rates)
# =====================================================================

class SootKinetics:
    """
    Semi-empirical soot formation/oxidation rate model, in the style of
    Lindstedt/Moss-Brookes-type engineering closures used in fire and
    combustion CFD (NOT finite-rate PAH/soot-precursor chemistry -- that
    requires a real chemical mechanism with dozens of species, well
    beyond an engineering-level model).

    Formation is modeled as first-order in local fuel mass fraction,
    Arrhenius-accelerated by temperature, active only on the fuel-rich
    side of the flame (Z > Z_stoich, matching where soot precursors
    physically form in a diffusion flame). Oxidation is modeled as
    first-order in soot mass fraction AND local O2 mass fraction,
    Arrhenius-accelerated, active on the oxidizer-exposed side.
    """

    def __init__(
        self,
        A_formation: float = 100.0,      # 1/s pre-exponential (formation)
        Ea_formation: float = 1.25e5,    # J/mol activation energy (formation)
        A_oxidation: float = 1.0e4,      # 1/s pre-exponential (oxidation)
        Ea_oxidation: float = 1.65e5,    # J/mol activation energy (oxidation)
    ):
        self.A_f, self.Ea_f = A_formation, Ea_formation
        self.A_o, self.Ea_o = A_oxidation, Ea_oxidation
        self.R_gas = 8.314  # J/(mol.K)

    def formation_rate(self, Y_fuel: np.ndarray, T: np.ndarray, Z: np.ndarray,
                        Z_stoich: float) -> np.ndarray:
        """Soot formation rate [1/s, mass fraction per second], fuel-rich side only."""
        Y_fuel = np.asarray(Y_fuel, dtype=float)
        T = np.asarray(T, dtype=float)
        rich_mask = np.asarray(Z, dtype=float) > Z_stoich
        k = self.A_f * np.exp(-self.Ea_f / (self.R_gas * np.clip(T, 200, None)))
        return np.where(rich_mask, k * Y_fuel, 0.0)

    def oxidation_rate(self, Y_soot: np.ndarray, Y_O2: np.ndarray,
                        T: np.ndarray) -> np.ndarray:
        """Soot oxidation rate [1/s], active wherever both soot and O2 are present."""
        Y_soot = np.asarray(Y_soot, dtype=float)
        Y_O2 = np.asarray(Y_O2, dtype=float)
        T = np.asarray(T, dtype=float)
        k = self.A_o * np.exp(-self.Ea_o / (self.R_gas * np.clip(T, 200, None)))
        return k * Y_soot * Y_O2

    def net_rate(self, Y_fuel, Y_soot, Y_O2, T, Z, Z_stoich) -> np.ndarray:
        """Net d(Y_soot)/dt [1/s] = formation - oxidation."""
        return (self.formation_rate(Y_fuel, T, Z, Z_stoich)
                - self.oxidation_rate(Y_soot, Y_O2, T))


# =====================================================================
# 5c. DETECTOR / SPRINKLER MODELS  (RTI correlation, obscuration)
# =====================================================================

class HeatDetectorModel:
    """
    Response Time Index (RTI) heat detector / sprinkler activation model
    -- the standard engineering correlation used throughout fire
    protection engineering (and internally by NIST FDS's own detector
    model):

        dT_link/dt = (sqrt(u_gas) / RTI) * (T_gas - T_link) - (C / RTI) * u_gas^0.5 * (T_link - T_ambient)

    Simplified here (C-factor/conduction loss term optional, defaults to
    0 = no conductive loss, the classic RTI-only form):

        dT_link/dt = (sqrt(u_gas) / RTI) * (T_gas - T_link)

    RTI [m^1/2 s^1/2] and activation temperature are device properties
    (typical sprinkler: RTI ~50-135 (m.s)^0.5, activation ~57-74 C for
    standard-response residential/commercial heads; fast-response heads
    have RTI as low as ~35-50).
    """

    def __init__(self, RTI: float = 100.0, activation_temp_C: float = 68.0,
                 C_factor: float = 0.0):
        self.RTI = RTI
        self.activation_temp_K = activation_temp_C + 273.15
        self.C_factor = C_factor
        self.T_link_K = T_AMBIENT
        self.activated = False
        self.activation_time_s: Optional[float] = None

    def step(self, t: float, dt: float, T_gas_K: float, u_gas_m_s: float,
              T_ambient_K: float = T_AMBIENT) -> bool:
        """
        Advances the link temperature by one time step (simple explicit
        Euler -- adequate given RTI dynamics are slow, seconds-to-minutes,
        relative to typical CFD/engineering dt). Returns True on the step
        activation first occurs.
        """
        u = max(u_gas_m_s, 0.1)  # avoid stall at exactly zero velocity
        dTdt = (math.sqrt(u) / self.RTI) * (T_gas_K - self.T_link_K)
        if self.C_factor > 0:
            dTdt -= (self.C_factor / self.RTI) * math.sqrt(u) * (self.T_link_K - T_ambient_K)
        self.T_link_K += dTdt * dt

        just_activated = False
        if not self.activated and self.T_link_K >= self.activation_temp_K:
            self.activated = True
            self.activation_time_s = t
            just_activated = True
        return just_activated

    def run_timeline(self, t: np.ndarray, T_gas_K: np.ndarray,
                       u_gas_m_s: np.ndarray) -> dict:
        """Runs step() over a full time series; returns link-temp history + activation time."""
        T_link_hist = np.zeros_like(t)
        for i in range(len(t)):
            dt = t[i] - t[i - 1] if i > 0 else t[0]
            self.step(t[i], max(dt, 1e-6), float(T_gas_K[i]), float(u_gas_m_s[i]))
            T_link_hist[i] = self.T_link_K
        return {"time": t, "T_link_K": T_link_hist, "activated": self.activated,
                "activation_time_s": self.activation_time_s}


class SmokeDetectorModel:
    """
    Obscuration-based smoke detector model with a characteristic
    response-lag time constant (first-order lag, matching NIST FDS's
    smoke detector model form):

        dY_c/dt = (Y_c,infinity - Y_c) / tau

    where Y_c,infinity is the instantaneous local smoke obscuration
    [%/m] and Y_c is the detector's internal (lagged) reading. Activates
    when Y_c exceeds the device's obscuration threshold.
    """

    def __init__(self, obscuration_threshold_pct_m: float = 3.28, tau_s: float = 4.5):
        """
        obscuration_threshold_pct_m: default 3.28 %/ft-equivalent
            (%/m converted), a common UL-268-ish residential smoke
            detector sensitivity; tau_s: characteristic lag (~2-9s
            typical spot detectors, per SFPE Handbook).
        """
        self.threshold = obscuration_threshold_pct_m
        self.tau = tau_s
        self.Y_c = 0.0
        self.activated = False
        self.activation_time_s: Optional[float] = None

    @staticmethod
    def obscuration_from_soot(soot_concentration_kg_m3: float, K_m: float = 8700.0) -> float:
        """
        Converts soot mass concentration [kg/m^3] to light obscuration
        [%/m]: Obs = 100*(1 - exp(-K_m * Cs))  ~=  100*K_m*Cs for small Cs.
        """
        return 100.0 * (1.0 - math.exp(-K_m * soot_concentration_kg_m3))

    def step(self, t: float, dt: float, soot_concentration_kg_m3: float) -> bool:
        Y_inf = self.obscuration_from_soot(soot_concentration_kg_m3)
        self.Y_c += (Y_inf - self.Y_c) / self.tau * dt
        just_activated = False
        if not self.activated and self.Y_c >= self.threshold:
            self.activated = True
            self.activation_time_s = t
            just_activated = True
        return just_activated


# =====================================================================
# 5d. PYROLYSIS  (1D in-depth conduction + Arrhenius decomposition)
# =====================================================================

@dataclass
class SolidFuelProperties:
    name: str
    density: float             # kg/m^3, virgin solid
    conductivity: float        # W/(m.K)
    specific_heat: float       # J/(kg.K)
    thickness: float           # m
    A_pyrolysis: float         # 1/s, Arrhenius pre-exponential
    Ea_pyrolysis: float        # J/mol, activation energy
    heat_of_pyrolysis: float   # J/kg, endothermic decomposition enthalpy
    char_fraction: float = 0.2  # residual mass fraction after full pyrolysis


class PyrolysisModel:
    """
    1D in-depth solid conduction coupled to single-step Arrhenius
    pyrolysis (the standard simplified solid-phase model used in fire
    engineering hand calculations and as the basis of FDS's own default
    single-reaction pyrolysis model, absent detailed TGA-derived
    multi-step kinetics):

        rho*cp * dT/dt = d/dx(k dT/dx) - heat_of_pyrolysis * reaction_rate
        d(rho_solid)/dt = -A*exp(-Ea/RT) * (rho_solid - rho_char)
        mdot_fuel'' = -d(rho_solid)/dt * dx   (integrated over the
                                                pyrolyzing depth)

    Explicit finite-difference in depth (1D), insulated back face by
    default (adiabatic backing, typical for a thick fuel package;
    override via back_face_bc for a conducting/cooled backing).

    NOT wired into the 3D DNS grid as a live boundary condition in this
    pass (that requires a new mass-source-into-continuity mechanism at
    the solver level, a further integration step) -- this class is a
    genuine, standalone, testable physics model producing mdot_fuel(t)
    and surface temperature history, which CAN be fed into
    fire_dns_coupling_one.FireSourceField as a physically-derived
    (rather than prescribed-t-squared) HRR(t) via
    fuel.delta_h_c * mdot_fuel(t) -- see run() below.
    """

    def __init__(self, fuel: SolidFuelProperties, n_nodes: int = 20,
                 incident_heat_flux_kw_m2: float = 25.0):
        self.fuel = fuel
        self.n = n_nodes
        self.dx = fuel.thickness / n_nodes
        self.T = np.full(n_nodes, T_AMBIENT)
        self.rho_solid = np.full(n_nodes, fuel.density)
        self.q_incident = incident_heat_flux_kw_m2 * 1000.0  # W/m^2
        self.R_gas = 8.314

    def _reaction_rate(self, T: np.ndarray, rho: np.ndarray) -> np.ndarray:
        rho_char = self.fuel.char_fraction * self.fuel.density
        available = np.clip(rho - rho_char, 0.0, None)
        k = self.fuel.A_pyrolysis * np.exp(
            -self.fuel.Ea_pyrolysis / (self.R_gas * np.clip(T, 200, None)))
        return k * available   # kg/(m^3.s), positive = mass loss rate

    def step(self, dt: float, T_gas_K: float, h_conv: float = 15.0) -> dict:
        """
        Advances the solid one time step. Surface (node 0) receives
        incident radiative/convective heat flux; back face (node n-1) is
        adiabatic. Returns dict with surface temperature and instantaneous
        fuel mass-loss rate per unit area [kg/(m^2.s)].
        """
        k, rho_cp = self.fuel.conductivity, self.fuel.density * self.fuel.specific_heat
        alpha = k / rho_cp
        dx = self.dx

        # Explicit stability check (simple diffusion CFL); if violated,
        # sub-cycle rather than silently going unstable.
        dt_stable = 0.4 * dx**2 / alpha
        n_sub = max(1, int(math.ceil(dt / dt_stable)))
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            T = self.T
            reaction = self._reaction_rate(T, self.rho_solid)
            self.rho_solid = np.clip(
                self.rho_solid - reaction * dt_sub, self.fuel.char_fraction * self.fuel.density, None)

            lap = np.zeros_like(T)
            lap[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2
            lap[0]  = 2 * (T[1] - T[0]) / dx**2      # Neumann-consistent one-sided
            lap[-1] = 2 * (T[-2] - T[-1]) / dx**2    # adiabatic back face

            pyrolysis_sink = reaction * self.fuel.heat_of_pyrolysis / rho_cp

            dTdt = alpha * lap - pyrolysis_sink
            # Surface flux boundary term (node 0): net flux in minus
            # re-radiation/convective loss to the gas.
            sigma_sb = 5.670374e-8
            q_net_surface = (self.q_incident
                              - h_conv * (T[0] - T_gas_K)
                              - 0.9 * sigma_sb * (T[0]**4 - T_gas_K**4))
            dTdt[0] += q_net_surface / (rho_cp * dx)

            self.T = T + dTdt * dt_sub

        mdot_area = float(np.sum(self._reaction_rate(self.T, self.rho_solid)) * dx)  # kg/(m^2.s)
        return {"T_surface_K": float(self.T[0]), "mdot_fuel_kg_m2_s": mdot_area,
                "T_profile_K": self.T.copy()}

    def run(self, duration_s: float, dt: float, T_gas_K: float = 800.0) -> dict:
        """Runs step() over a fixed duration; returns time series."""
        n_steps = int(duration_s / dt)
        t = np.arange(n_steps) * dt
        T_surf = np.zeros(n_steps)
        mdot = np.zeros(n_steps)
        for i in range(n_steps):
            r = self.step(dt, T_gas_K)
            T_surf[i] = r["T_surface_K"]
            mdot[i] = r["mdot_fuel_kg_m2_s"]
        return {"time": t, "T_surface_K": T_surf, "mdot_fuel_kg_m2_s": mdot}


# =====================================================================
# 6. ORCHESTRATOR
# =====================================================================

class FireOne:
    """
    Orchestrates the fire engineering pipeline: design fire -> plume
    analytics -> combustion state relations -> radiation -> tenability.
    Mirrors SeismicOne's role in seismic_one.py.
    """

    def __init__(self, fire_curve: DesignFireCurve, fuel: FuelProperties,
                 fire_diameter_m: float, chi_r: float = 0.35):
        self.fire_curve = fire_curve
        self.combustion = MixtureFractionCombustion(fuel)
        self.D = fire_diameter_m
        self.chi_r = chi_r

    def run(self, t: np.ndarray, assessment_heights: np.ndarray) -> dict:
        """
        Full pipeline over a time series, reporting plume conditions at
        each requested height.
        """
        q_kw = self.fire_curve.hrr_kw(t)
        flame_height = PlumeCorrelations.flame_height(np.maximum(q_kw, 1e-6), self.D)

        dT = np.zeros((len(t), len(assessment_heights)))
        for j, z in enumerate(assessment_heights):
            for i, q in enumerate(q_kw):
                if q > 1e-6:
                    dT[i, j] = PlumeCorrelations.centerline_temperature_rise(
                        q, z, self.D, self.chi_r)

        yields = self.combustion.hrr_to_yields(np.maximum(q_kw, 1e-9))

        return {
            "time": t, "hrr_kw": q_kw, "flame_height_m": flame_height,
            "centerline_dT_K": dT, "assessment_heights": assessment_heights,
            "mdot_co_kg_s": yields["mdot_co_kg_s"],
            "mdot_soot_kg_s": yields["mdot_soot_kg_s"],
            "adiabatic_flame_temp_K": self.combustion.adiabatic_flame_temperature(),
            "z_stoich": self.combustion.z_stoich,
        }

    def summary(self, results: dict) -> str:
        lines = []
        lines.append(f"Peak HRR: {np.max(results['hrr_kw']):.0f} kW")
        lines.append(f"Peak flame height: {np.max(results['flame_height_m']):.2f} m")
        lines.append(f"Adiabatic flame temperature: {results['adiabatic_flame_temp_K']:.0f} K "
                      f"({results['adiabatic_flame_temp_K']-273.15:.0f} C)")
        lines.append(f"Stoichiometric mixture fraction: {results['z_stoich']:.4f}")
        for j, z in enumerate(results["assessment_heights"]):
            peak_dT = np.max(results["centerline_dT_K"][:, j])
            lines.append(f"  z={z}m: peak centerline dT = {peak_dT:.1f} K")
        return "\n".join(lines)


if __name__ == "__main__":
    print(f"fire_one.py v{__version__} loaded OK")
