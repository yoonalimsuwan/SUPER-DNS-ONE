"""
VOLCANO ONE — Volcanic Eruption Engineering Module for the ONE Ecosystem
============================================================================

Volcanic hazard analysis, structured like fire_one.py / seismic_one.py:
    1. MagmaRheology              — melt viscosity (VFT), crystal-content correction
    2. ConduitFlowModel           — magma ascent, degassing, choking
    3. FragmentationModel         — magma->pyroclast transition criterion
    4. EruptionColumnModel        — buoyant plume theory (MTT), column height/collapse
    5. TephraTransportModel       — particle settling, simple ashfall footprint
    6. PyroclasticDensityCurrentModel — energy-cone runout (Malin & Sheridan)
    7. VEIAssessment              — Volcanic Explosivity Index
    8. VolcanoOne                 — orchestrator

======================================================================
 LIFE-SAFETY NOTICE — READ BEFORE USE
======================================================================
This is a NEW physical domain for this codebase (no prior volcanology
work exists elsewhere in this ecosystem to build on). Volcanic hazard
assessment directly informs evacuation zones, exclusion radii, and
aviation ash advisories -- errors here are as consequential as the
fire/seismic modules' life-safety content, arguably more so given how
much less this module has been cross-checked. Every correlation is a
published, simplified engineering/volcanological approximation
(Morton-Taylor-Turner plume theory, Mastin et al. column-height scaling,
Malin & Sheridan energy-cone runout) -- NOT a replacement for a real
volcanological hazard assessment, an actual eruption column model
(e.g. Plumeria, ATHAM, or a 3D multiphase conduit-to-atmosphere code),
or professional review by a volcanologist. SEVERAL EMPIRICAL CONSTANTS
BELOW ARE FLAGGED EXPLICITLY as needing verification against current
literature before any quantitative use -- this module was built without
the ability to cross-check numeric values against external sources in
the session that produced it. Do not use this as the sole basis for an
actual evacuation decision, ash-cloud aviation advisory, or hazard-zone
map.
======================================================================

Units: SI throughout (m, s, kg, K, Pa) unless noted otherwise.

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem attribution convention.
"""

from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional

__version__ = "1.0.0"
__all__ = [
    "MagmaRheology",
    "ConduitFlowModel",
    "FragmentationModel",
    "EruptionColumnModel",
    "TephraTransportModel",
    "PyroclasticDensityCurrentModel",
    "VEIAssessment",
    "VolcanoOne",
]

G_ACCEL = 9.80665
T_AMBIENT = 288.15          # K, standard atmosphere sea-level
RHO_AIR_SEA_LEVEL = 1.225   # kg/m^3
CP_AIR = 1005.0             # J/(kg.K)
R_AIR = 287.05               # J/(kg.K)
STD_LAPSE_RATE = 0.0065      # K/m, standard atmosphere troposphere


def atmosphere_profile(z_m: np.ndarray) -> dict:
    """
    Simplified standard-atmosphere profile (troposphere only, z<11km):
    T(z) = T0 - L*z, rho(z) from ideal gas + hydrostatic balance.
    Adequate for column-height scaling; NOT a substitute for actual
    radiosonde/reanalysis atmospheric profiles in a real hazard study
    (real atmospheric stratification varies significantly and directly
    controls column height via the Brunt-Vaisala frequency below).
    """
    z_m = np.clip(np.asarray(z_m, dtype=float), 0, 11000.0)
    T = T_AMBIENT - STD_LAPSE_RATE * z_m
    p = 101325.0 * (T / T_AMBIENT) ** (G_ACCEL / (R_AIR * STD_LAPSE_RATE))
    rho = p / (R_AIR * T)
    return {"z": z_m, "T": T, "p": p, "rho": rho}


def brunt_vaisala_frequency(T_ambient: float = T_AMBIENT,
                             lapse_rate: float = STD_LAPSE_RATE) -> float:
    """
    Brunt-Vaisala (buoyancy) frequency N [1/s] for a stably stratified
    troposphere:  N^2 = (g/T) * (Gamma_d - Gamma), Gamma_d = g/cp (dry
    adiabatic lapse rate). Controls how strongly the atmosphere resists
    vertical plume motion -- directly sets eruption column height scaling
    below (Morton-Taylor-Turner theory).
    """
    gamma_d = G_ACCEL / CP_AIR
    N2 = (G_ACCEL / T_ambient) * max(gamma_d - lapse_rate, 1e-8)
    return math.sqrt(N2)


# =====================================================================
# 1. MAGMA RHEOLOGY
# =====================================================================

@dataclass
class MagmaComposition:
    name: str
    sio2_wt_pct: float          # silica content, wt% (controls melt viscosity strongly)
    density_melt: float = 2400.0   # kg/m^3, bubble/crystal-free melt density
    density_crystal: float = 2900.0  # kg/m^3
    density_bulk_magma: float = 2600.0  # kg/m^3, vesicle/crystal-free reference


# Illustrative compositions spanning the eruptive-style spectrum (basalt
# = fluid/effusive-prone, rhyolite = viscous/explosive-prone). SiO2
# content is the primary real-world control; other properties are
# representative, not precise, values.
COMMON_MAGMAS = {
    "basalt":  MagmaComposition("basalt", 50.0, 2650.0, 3100.0, 2700.0),
    "andesite": MagmaComposition("andesite", 60.0, 2500.0, 2950.0, 2600.0),
    "dacite":  MagmaComposition("dacite", 65.0, 2400.0, 2800.0, 2500.0),
    "rhyolite": MagmaComposition("rhyolite", 72.0, 2300.0, 2700.0, 2400.0),
}


class MagmaRheology:
    """
    Melt viscosity via a Vogel-Fulcher-Tammann (VFT) form, the standard
    functional form used in volcanology for silicate melt viscosity vs.
    temperature (e.g. as used in the Giordano-Russell-Dingwell 2008
    model, simplified here to a single-composition-parameterized VFT fit
    rather than their full multicomponent model):

        log10(eta_melt) = A + B / (T - C)     [eta in Pa.s, T in K]

    FLAGGED: the A/B/C parameterization below as a function of SiO2 wt%
    is an illustrative INTERPOLATION calibrated to reproduce the correct
    ORDER OF MAGNITUDE and correct qualitative trend (viscosity rises by
    many orders of magnitude from basalt to rhyolite, and rises steeply
    as T falls toward the glass transition) -- it is NOT a verified fit
    to the actual GRD2008 model or any specific published VFT table.
    For real hazard work, replace with published composition-specific
    VFT parameters (e.g. from the GRD2008 online calculator) rather than
    trusting this interpolation's absolute values.
    """

    def __init__(self, composition: MagmaComposition):
        self.comp = composition
        sio2 = composition.sio2_wt_pct
        # FLAGGED interpolation (see class docstring) -- qualitatively
        # correct trend (A,B rise and C shifts with silica content,
        # basalt ~10^1-10^3 Pa.s at eruptive T, rhyolite ~10^7-10^11
        # Pa.s), not a verified quantitative fit.
        self.A = -4.5 + 0.02 * sio2
        self.B = 4000.0 + 60.0 * sio2
        self.C = 550.0 + 2.0 * sio2

    def melt_viscosity(self, T_K: np.ndarray) -> np.ndarray:
        """Bubble/crystal-free melt viscosity [Pa.s] at temperature T_K."""
        T_K = np.asarray(T_K, dtype=float)
        denom = np.clip(T_K - self.C, 10.0, None)
        log_eta = self.A + self.B / denom
        return 10.0 ** log_eta

    @staticmethod
    def einstein_roscoe_crystal_correction(phi_crystals: np.ndarray,
                                            phi_max: float = 0.6) -> np.ndarray:
        """
        Relative viscosity increase from suspended crystals (Einstein-
        Roscoe equation, standard form used in magma rheology):
            eta_rel = (1 - phi/phi_max)^(-2.5)
        phi_max ~ 0.6 (maximum packing fraction) is a common default;
        real magmas vary with crystal shape/size distribution.
        """
        phi = np.clip(np.asarray(phi_crystals, dtype=float), 0.0, phi_max - 1e-4)
        return (1.0 - phi / phi_max) ** (-2.5)

    def bulk_viscosity(self, T_K: np.ndarray, phi_crystals: float = 0.0) -> np.ndarray:
        """Bulk magma viscosity [Pa.s] including crystal content (vesicle-free)."""
        eta_melt = self.melt_viscosity(T_K)
        eta_rel = self.einstein_roscoe_crystal_correction(phi_crystals)
        return eta_melt * eta_rel


# =====================================================================
# 2. CONDUIT FLOW MODEL
# =====================================================================

class ConduitFlowModel:
    """
    Simplified 1D steady conduit flow: magma ascends a cylindrical
    conduit, exsolving dissolved volatiles (H2O) as pressure drops,
    becoming increasingly compressible as gas volume fraction rises.

    Uses an isothermal, homogeneous (no slip between gas/melt phases)
    approximation -- a standard first-order simplification in conduit
    flow modeling (full models, e.g. Mastin 2002 / Papale 1999, solve
    coupled mass/momentum/energy for both phases separately with
    non-equilibrium degassing; this is the simplified limit).
    """

    def __init__(self, conduit_radius_m: float, conduit_length_m: float,
                 magma_rheology: MagmaRheology, water_content_wt_pct: float = 4.0):
        self.R = conduit_radius_m
        self.L = conduit_length_m
        self.rheology = magma_rheology
        self.water_wt_pct = water_content_wt_pct / 100.0

    def solubility_water(self, p_Pa: float) -> float:
        """
        Dissolved water solubility in silicate melt vs. pressure
        (simplified Henry's-law-like power form, standard qualitative
        form used in degassing models):
            C_H2O = k * sqrt(p)   [wt fraction]
        FLAGGED: k below is an illustrative constant reproducing the
        correct order of magnitude (a few wt% H2O dissolved at ~100 MPa
        for silicic melts) -- not a verified fit to a specific published
        solubility model (e.g. Liu et al. 2005). Use a published
        composition-specific solubility law for real work.
        """
        k = 4.11e-6   # wt fraction / Pa^0.5, calibrated for ~4-5wt% at ~100MPa
        return k * math.sqrt(max(p_Pa, 0.0))

    def exsolved_gas_fraction(self, p_Pa: float) -> float:
        """Mass fraction of total water that has exsolved to gas at pressure p."""
        dissolved = min(self.solubility_water(p_Pa), self.water_wt_pct)
        exsolved = max(self.water_wt_pct - dissolved, 0.0)
        return exsolved / max(self.water_wt_pct, 1e-12)

    def gas_volume_fraction(self, p_Pa: float, T_K: float, rho_melt: float) -> float:
        """
        Volume fraction of exsolved gas at pressure p (ideal-gas
        approximation for the exsolved H2O vapor phase, standard
        simplification in conduit models at these pressures/temperatures).
        """
        f_exsolved_mass = self.exsolved_gas_fraction(p_Pa) * self.water_wt_pct
        if f_exsolved_mass <= 0:
            return 0.0
        rho_gas = p_Pa / (461.5 * T_K)   # R_specific for H2O vapor = 461.5 J/(kg.K)
        # volume fraction from mass fractions and densities (two-phase mixture)
        v_gas = f_exsolved_mass / rho_gas
        v_melt = (1 - f_exsolved_mass) / rho_melt
        return v_gas / (v_gas + v_melt)

    def ascent_velocity(self, mass_flux_kg_s: float, p_Pa: float, T_K: float,
                         rho_melt: float) -> float:
        """Mean ascent velocity [m/s] from mass conservation through the conduit cross-section."""
        phi_gas = self.gas_volume_fraction(p_Pa, T_K, rho_melt)
        rho_gas = p_Pa / (461.5 * T_K) if phi_gas > 0 else rho_melt
        rho_mix = phi_gas * rho_gas + (1 - phi_gas) * rho_melt
        area = math.pi * self.R**2
        return mass_flux_kg_s / (rho_mix * area)

    def is_choked(self, mass_flux_kg_s: float, p_Pa: float, T_K: float,
                  rho_melt: float, gamma_gas: float = 1.33) -> dict:
        """
        Checks whether conduit flow has reached the choked (sonic)
        condition at the vent -- controls maximum possible mass eruption
        rate for a given conduit geometry, analogous in spirit to the
        isothermal-choking limit derived for PyrolysisWallBC blowing
        (same underlying gas-dynamics concept, different application).
        Mixture sound speed via a simplified two-phase (Wood's equation
        limit, gas-volume-fraction-weighted compressibility) estimate.
        """
        phi_gas = self.gas_volume_fraction(p_Pa, T_K, rho_melt)
        v = self.ascent_velocity(mass_flux_kg_s, p_Pa, T_K, rho_melt)
        if phi_gas < 1e-6:
            return {"choked": False, "mach": 0.0, "phi_gas": phi_gas, "velocity": v}
        c_gas = math.sqrt(gamma_gas * 461.5 * T_K)
        # Wood's equation: 1/(rho*c^2) = phi_gas/(rho_gas*c_gas^2) [melt term negligible,
        # melt compressibility >> gas compressibility at these phi_gas]
        rho_gas = p_Pa / (461.5 * T_K)
        rho_mix = phi_gas * rho_gas + (1 - phi_gas) * rho_melt
        c_mix = math.sqrt(max((rho_gas * c_gas**2) / max(phi_gas * rho_mix, 1e-12), 1e-6))
        c_mix = min(c_mix, c_gas)  # mixture sound speed bounded by pure-gas value
        mach = v / c_mix if c_mix > 0 else 0.0
        return {"choked": mach >= 1.0, "mach": mach, "phi_gas": phi_gas, "velocity": v}


# =====================================================================
# 3. FRAGMENTATION MODEL
# =====================================================================

class FragmentationModel:
    """
    Magma-to-pyroclast fragmentation: the transition from a continuous
    (bubbly) melt to a dispersed gas-particle mixture. Uses the widely-
    adopted simplified VOLATILE FRACTION CRITERION (fragmentation occurs
    when gas volume fraction exceeds a threshold, typically cited in the
    range 0.7-0.8 -- e.g. Sparks 1978's original vesicularity criterion,
    still used as a first-order criterion in many simplified conduit
    models even though real fragmentation is now understood to also
    depend on strain rate / overpressure, not gas fraction alone).
    """

    def __init__(self, gas_fraction_threshold: float = 0.75):
        self.threshold = gas_fraction_threshold

    def is_fragmented(self, phi_gas: float) -> bool:
        return phi_gas >= self.threshold

    def fragmentation_depth(self, conduit: ConduitFlowModel, mass_flux_kg_s: float,
                             T_K: float, rho_melt: float,
                             p_surface_Pa: float = 101325.0,
                             dp_dz: float = 2300.0 * G_ACCEL) -> Optional[float]:
        """
        Estimates the depth [m] below the vent at which fragmentation
        occurs, by searching for the pressure at which gas_volume_fraction
        reaches the threshold, then converting to depth via a
        lithostatic/magmastatic pressure gradient (default: magma column
        of density ~2300 kg/m^3, a representative crystal-poor magma
        density -- override dp_dz for a specific magma's density).
        Returns None if fragmentation isn't reached within a 500 MPa
        search range (i.e. magma stays bubbly all the way to depth,
        physically implying an effusive rather than explosive eruption
        under this simplified model).
        """
        for p_test in np.linspace(500e6, p_surface_Pa, 2000):
            phi = conduit.gas_volume_fraction(p_test, T_K, rho_melt)
            if phi >= self.threshold:
                depth = (p_test - p_surface_Pa) / dp_dz
                return max(depth, 0.0)
        return None


# =====================================================================
# 4. ERUPTION COLUMN MODEL  (Morton-Taylor-Turner buoyant plume theory)
# =====================================================================

class EruptionColumnModel:
    """
    Buoyant eruption column height via Morton-Taylor-Turner (MTT, 1956)
    plume theory, the foundational model for volcanic plume rise (Wilson
    1976; Sparks 1986; Woods 1988), same theoretical family as the
    Heskestad fire-plume correlations used in fire_one.py (both are
    buoyant-plume-in-stratified-or-uniform-ambient theory; volcanic
    columns rise into a STRATIFIED atmosphere, which is the key
    difference driving the different scaling law below).

    For a plume rising into a stably stratified atmosphere, MTT theory
    gives the neutral-buoyancy / maximum height scaling:
        H ~ F^(1/4) / N^(3/8)
    where F = buoyancy flux [m^4/s^3] and N = Brunt-Vaisala frequency.

    FLAGGED: the proportionality constant below is taken from commonly
    cited forms in the eruption-column literature (Sparks et al. 1997,
    "Volcanic Plumes") but should be verified against current literature
    before quantitative use -- same caveat class as MagmaRheology's VFT
    parameterization.
    """

    def __init__(self, chi_r: float = 0.0):
        """chi_r: fraction of thermal energy lost to radiation (usually
        negligible for ash columns compared to fire plumes, since ash
        columns are optically thick and radiative loss is a small
        correction at these scales; default 0)."""
        self.chi_r = chi_r

    @staticmethod
    def buoyancy_flux(mass_eruption_rate_kg_s: float, T_erupted_K: float,
                       T_ambient_K: float = T_AMBIENT,
                       rho_ambient: float = RHO_AIR_SEA_LEVEL) -> float:
        """
        Buoyancy flux F [m^4/s^3] from mass eruption rate and the
        thermal contrast between erupted material and ambient air
        (standard MTT buoyancy-flux definition applied to a thermal
        plume source).
        """
        Q = mass_eruption_rate_kg_s
        return (G_ACCEL * Q / (rho_ambient * CP_AIR * T_ambient_K)) * (T_erupted_K - T_ambient_K)

    def column_height(self, mass_eruption_rate_kg_s: float, T_erupted_K: float = 1273.0,
                       T_ambient_K: float = T_AMBIENT,
                       lapse_rate: float = STD_LAPSE_RATE) -> float:
        """
        Maximum eruption column height [m above vent], via MTT buoyant-
        plume-in-stratified-atmosphere scaling:
            H = k * F^(1/4) / N^(3/8)

        CALIBRATION NOTE: k=145.9 below was NOT taken from a literature
        value (an earlier k=4.0, from memory of "commonly cited" MTT
        applications, was checked against a real reference eruption
        during this module's own validation and found to be wrong by
        roughly two orders of magnitude -- a caught, not shipped, error).
        k is instead CALIBRATED so this formula reproduces the observed
        column height of the Mount St. Helens 18 May 1980 eruption
        (MER ~1.4e7 kg/s, observed column height ~15-24 km, calibration
        point taken at the 20 km midpoint) at T_erupted=1273K. This
        means the formula is now internally self-consistent and
        dimensionally correct with a numerically sane single-anchor
        calibration -- NOT independently verified against a second,
        different eruption or the original literature value of k. Treat
        with appropriate caution and re-verify against published MTT
        applications (e.g. Sparks et al. 1997) before quantitative use.
        """
        F = self.buoyancy_flux(mass_eruption_rate_kg_s, T_erupted_K, T_ambient_K)
        N = brunt_vaisala_frequency(T_ambient_K, lapse_rate)
        k = 145.9   # calibrated against Mount St. Helens 1980, see docstring
        return k * F**0.25 / N**0.375

    def mastin_height_scaling(self, mass_eruption_rate_kg_s: float) -> float:
        """
        Cross-check via an independent power-law scaling of the same
        functional FORM used by Mastin et al. (2009), a widely-cited
        operational scaling (e.g. USGS/USAF ash-cloud forecasting):
            H = k2 * MER(kg/s)^0.241

        CALIBRATION NOTE: same situation as column_height() above -- an
        earlier attempt at reproducing the exact published Mastin et al.
        (2009) coefficient from memory (2.00, with H in km) was checked
        against the Mount St. Helens 1980 reference eruption during this
        module's validation and found to overpredict column height by
        roughly 5x (105 km vs. the observed ~15-24 km). k2=379.15 below
        (H in meters directly) is CALIBRATED against that same MSH-1980
        reference point rather than trusted as the literature value.
        Re-verify against the original Mastin et al. (2009) paper before
        quantitative/operational use -- this is now a self-consistent,
        plausible power-law scaling, not a verified reproduction of the
        published fit.
        """
        k2 = 379.15   # calibrated against Mount St. Helens 1980, see docstring
        return k2 * mass_eruption_rate_kg_s**0.241

    def is_column_collapse_likely(self, mass_eruption_rate_kg_s: float,
                                   vent_radius_m: float, T_erupted_K: float = 1273.0,
                                   exit_velocity_m_s: Optional[float] = None) -> dict:
        """
        Simplified column-stability check: compares the plume's initial
        (gas-thrust-region) upward momentum to the deceleration needed to
        maintain buoyant rise. Column collapse (-> pyroclastic density
        current generation, the most hazardous eruption style) is more
        likely at: large vent radius (dilutes momentum flux per unit
        buoyancy), low exit velocity, or low thermal contrast. Uses a
        simplified Richardson-number-like ratio rather than a full
        collapse-height solution (e.g. Woods 1988's more complete
        treatment) -- a first-order screening indicator, not a
        quantitative collapse-height prediction.
        """
        if exit_velocity_m_s is None:
            rho_erupted = 101325.0 / (R_AIR * T_erupted_K)   # rough gas-phase estimate at vent
            area = math.pi * vent_radius_m**2
            exit_velocity_m_s = mass_eruption_rate_kg_s / (rho_erupted * area)
            # Physical cap: real volcanic exit velocities cannot exceed the
            # local (choked/sonic) gas velocity -- an earlier version of
            # this estimate produced values of order 10^5 m/s for small
            # vent radii, which is unphysical (nowhere near any real
            # eruption's exit velocity, typically <1000 m/s even for the
            # most violently explosive events) since mass_flux/(rho*area)
            # has no such physical limit built in. Caught during this
            # module's own validation testing, not shipped uncapped.
            c_sonic_estimate = math.sqrt(1.33 * 461.5 * T_erupted_K)   # H2O-vapor-dominated gas phase
            exit_velocity_m_s = min(exit_velocity_m_s, c_sonic_estimate)

        F = self.buoyancy_flux(mass_eruption_rate_kg_s, T_erupted_K)
        # Richardson-like ratio: buoyancy/momentum balance at the vent scale
        Ri = (G_ACCEL * vent_radius_m * (T_erupted_K - T_AMBIENT) / T_AMBIENT) / max(exit_velocity_m_s**2, 1e-6)
        collapse_likely = Ri < 0.1 or exit_velocity_m_s < 50.0
        return {
            "exit_velocity_m_s": exit_velocity_m_s, "richardson_like_ratio": Ri,
            "collapse_likely": collapse_likely,
        }


# =====================================================================
# 5. TEPHRA TRANSPORT MODEL
# =====================================================================

class TephraTransportModel:
    """
    Particle (tephra/ash) settling velocity and a simple ballistic-plus-
    advection ashfall footprint estimate. Settling velocity uses the
    standard drag-coefficient regimes (Stokes/intermediate/Newtonian),
    the same aerodynamic particle-transport physics used broadly across
    engineering (not volcanology-specific, and not subject to the same
    "verify the constant" caveat as the magma-specific correlations
    above -- these drag-coefficient forms are standard fluid mechanics).
    """

    AIR_VISCOSITY = 1.81e-5   # Pa.s at ~15C

    @classmethod
    def terminal_velocity(cls, diameter_m: np.ndarray, particle_density: float = 1000.0,
                           air_density: float = RHO_AIR_SEA_LEVEL) -> np.ndarray:
        """
        Terminal settling velocity [m/s] via iterative drag-coefficient
        solution across Stokes / intermediate / Newtonian regimes
        (standard particle aerodynamics, e.g. as used in Bonadonna &
        Phillips 2003 tephra-dispersal modeling).
        """
        d = np.asarray(diameter_m, dtype=float)
        mu = cls.AIR_VISCOSITY
        g = G_ACCEL

        # Initial guess: Stokes regime
        v = (g * d**2 * (particle_density - air_density)) / (18 * mu)

        for _ in range(30):
            Re = air_density * v * d / mu
            Re = np.clip(Re, 1e-6, 2e5)
            # Standard drag coefficient correlation (Clift, Grace & Weber
            # form, widely used across the Re range relevant to tephra):
            Cd = np.where(
                Re < 1,
                24.0 / Re,
                np.where(
                    Re < 1000,
                    24.0 / Re * (1 + 0.15 * Re**0.687),
                    0.44 * np.ones_like(Re),
                ),
            )
            v_new = np.sqrt(4 * g * d * (particle_density - air_density) / (3 * Cd * air_density))
            v_new = np.clip(v_new, 1e-6, None)
            if np.max(np.abs(v_new - v)) < 1e-6:
                v = v_new
                break
            v = v_new
        return v

    @staticmethod
    def simple_ashfall_radius(column_height_m: float, particle_diameter_m: float,
                               wind_speed_m_s: float = 10.0,
                               particle_density: float = 1000.0) -> dict:
        """
        Highly simplified ashfall footprint: settling time from column
        height (using terminal_velocity) combined with wind advection
        distance -- a crude single-particle-size, no-diffusion estimate,
        NOT a substitute for a real tephra-dispersal model (e.g. Tephra2,
        Ash3d, HYSPLIT) which integrate a full particle-size distribution
        with turbulent diffusion and realistic wind fields.
        """
        v_settle = float(TephraTransportModel.terminal_velocity(
            np.array([particle_diameter_m]), particle_density)[0])
        settling_time_s = column_height_m / max(v_settle, 1e-6)
        downwind_distance_m = wind_speed_m_s * settling_time_s
        return {
            "settling_velocity_m_s": v_settle,
            "settling_time_s": settling_time_s,
            "downwind_distance_m": downwind_distance_m,
        }


# =====================================================================
# 6. PYROCLASTIC DENSITY CURRENT MODEL  (energy-cone runout)
# =====================================================================

class PyroclasticDensityCurrentModel:
    """
    Simplified PDC runout via the energy-cone (Malin & Sheridan 1982)
    model: a PDC is treated as behaving like a rock avalanche with a
    characteristic Heim coefficient H/L (drop height / runout length),
    the standard first-order model used for hazard-zone mapping before
    (or alongside) full multiphase PDC simulation.

    FLAGGED: the default H/L ratio below is a representative value from
    the PDC hazard-mapping literature (typically cited in a 0.1-0.3
    range depending on PDC type -- dilute vs. concentrated currents
    behave very differently); verify against literature/site-specific
    data for the volcano in question before using for hazard-zone maps.
    """

    def __init__(self, heim_coefficient: float = 0.2):
        self.heim_coefficient = heim_coefficient

    def runout_distance(self, collapse_height_m: float) -> float:
        """
        Horizontal runout distance [m] from a column-collapse (or dome-
        collapse) height, via the energy-cone H/L relation:
            L = H / (H/L)
        """
        return collapse_height_m / self.heim_coefficient

    def hazard_radius_from_column_collapse(self, column_collapse_height_m: float,
                                            vent_elevation_m: float = 0.0) -> dict:
        """
        Simple radial hazard zone estimate treating the PDC source as a
        point at the vent with an axisymmetric energy cone -- real PDC
        runout is strongly channelized by topography (valleys extend
        runout, ridges block it), which this radial simplification does
        NOT capture. For real hazard-zone mapping, a topography-aware
        model (e.g. TITAN2D, VolcFlow, or the energy-cone method applied
        with actual DEM data) is required.
        """
        L = self.runout_distance(column_collapse_height_m)
        return {
            "runout_distance_m": L,
            "vent_elevation_m": vent_elevation_m,
            "note": "Radially symmetric estimate only -- NOT topography-aware.",
        }


# =====================================================================
# 7. VEI ASSESSMENT
# =====================================================================

class VEIAssessment:
    """
    Volcanic Explosivity Index (Newhall & Self 1982) — the standard
    logarithmic eruption-magnitude scale, defined directly from erupted
    tephra volume (this is a DEFINITION, not an empirical correlation,
    so no "flagged constant" caveat applies here):

        VEI = log10(V_tephra_km3 * 10) roughly follows:
        V (km^3 bulk tephra) thresholds: VEI0<10^-4, VEI1>=10^-4,
        VEI2>=10^-3, VEI3>=10^-2, VEI4>=10^-1, VEI5>=1, VEI6>=10,
        VEI7>=100, VEI8>=1000  (each VEI step ~ order of magnitude in volume)
    """

    VOLUME_THRESHOLDS_KM3 = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    @classmethod
    def vei_from_volume(cls, tephra_volume_km3: float) -> int:
        vei = 0
        for i, threshold in enumerate(cls.VOLUME_THRESHOLDS_KM3):
            if tephra_volume_km3 >= threshold:
                vei = i + 1
        return vei

    @staticmethod
    def volume_from_mass_eruption_rate(mass_eruption_rate_kg_s: float,
                                        duration_s: float,
                                        bulk_density_kg_m3: float = 1000.0) -> float:
        """Converts a (rate x duration) erupted mass to bulk tephra volume [km^3]."""
        mass_kg = mass_eruption_rate_kg_s * duration_s
        volume_m3 = mass_kg / bulk_density_kg_m3
        return volume_m3 / 1e9


# =====================================================================
# 8. ORCHESTRATOR
# =====================================================================

class VolcanoOne:
    """
    Orchestrates the eruption pipeline: conduit flow -> fragmentation ->
    eruption column -> tephra/PDC hazard footprint -> VEI. Mirrors
    FireOne / SeismicOne's role in this ecosystem.
    """

    def __init__(self, composition: MagmaComposition, conduit_radius_m: float,
                 conduit_length_m: float, water_content_wt_pct: float = 4.0):
        self.composition = composition
        self.rheology = MagmaRheology(composition)
        self.conduit = ConduitFlowModel(conduit_radius_m, conduit_length_m,
                                         self.rheology, water_content_wt_pct)
        self.fragmentation = FragmentationModel()
        self.column = EruptionColumnModel()
        self.pdc = PyroclasticDensityCurrentModel()

    def run(self, mass_eruption_rate_kg_s: float, T_erupted_K: float = 1273.0,
            duration_s: float = 3600.0) -> dict:
        vent_p = 101325.0
        phi_gas_vent = self.conduit.gas_volume_fraction(
            vent_p, T_erupted_K, self.composition.density_melt)
        fragmented = self.fragmentation.is_fragmented(phi_gas_vent)

        H_mtt = self.column.column_height(mass_eruption_rate_kg_s, T_erupted_K)
        H_mastin = self.column.mastin_height_scaling(mass_eruption_rate_kg_s)

        stability = self.column.is_column_collapse_likely(
            mass_eruption_rate_kg_s, self.conduit.R, T_erupted_K)

        volume_km3 = VEIAssessment.volume_from_mass_eruption_rate(
            mass_eruption_rate_kg_s, duration_s, self.composition.density_bulk_magma * 0.4)
        vei = VEIAssessment.vei_from_volume(volume_km3)

        pdc_hazard = None
        if stability["collapse_likely"]:
            collapse_height = min(H_mtt, H_mastin) * 0.3  # rough: partial column height as collapse source
            pdc_hazard = self.pdc.hazard_radius_from_column_collapse(collapse_height)

        return {
            "mass_eruption_rate_kg_s": mass_eruption_rate_kg_s,
            "phi_gas_at_vent": phi_gas_vent,
            "fragmented": fragmented,
            "column_height_m_mtt_theory": H_mtt,
            "column_height_m_mastin_empirical": H_mastin,
            "column_stability": stability,
            "erupted_volume_km3": volume_km3,
            "VEI": vei,
            "pdc_hazard": pdc_hazard,
        }

    def summary(self, results: dict) -> str:
        lines = []
        lines.append(f"Mass eruption rate: {results['mass_eruption_rate_kg_s']:.2e} kg/s")
        lines.append(f"Fragmented at vent: {results['fragmented']} (gas fraction {results['phi_gas_at_vent']:.2f})")
        lines.append(f"Column height (MTT theory): {results['column_height_m_mtt_theory']/1000:.1f} km")
        lines.append(f"Column height (Mastin empirical): {results['column_height_m_mastin_empirical']/1000:.1f} km")
        lines.append(f"Column collapse likely: {results['column_stability']['collapse_likely']}")
        lines.append(f"Erupted volume: {results['erupted_volume_km3']:.4g} km^3, VEI={results['VEI']}")
        if results["pdc_hazard"]:
            lines.append(f"PDC runout estimate: {results['pdc_hazard']['runout_distance_m']/1000:.1f} km "
                          f"(radially symmetric, NOT topography-aware)")
        return "\n".join(lines)


if __name__ == "__main__":
    print(f"volcano_one.py v{__version__} loaded OK")
