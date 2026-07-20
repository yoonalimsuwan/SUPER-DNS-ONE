"""
STORM ONE — Severe Storm, Tornado & Tropical Cyclone Engineering Module
============================================================================

Severe weather hazard analysis, structured like fire_one.py / seismic_one.py
/ volcano_one.py / flood_one.py:
    1. AtmosphericStabilityModel  — CAPE/CIN from a lifted-parcel sounding
    2. WindShearModel             — bulk shear, storm-relative helicity
    3. TornadoVortexModel         — Rankine combined vortex, EF-scale
    4. TropicalCycloneModel       — Holland parametric wind profile,
                                     pressure-wind relationship, Saffir-Simpson
    5. StormSurgeModel            — simplified wind-setup + inverse-barometer surge
    6. StormDamageAssessment      — EF-scale / Saffir-Simpson damage curves
    7. StormOne                   — orchestrator

======================================================================
 LIFE-SAFETY NOTICE — READ BEFORE USE
======================================================================
Severe storm and tornado/hurricane hazard assessment directly informs
warnings, evacuation orders, and structural design decisions. This
module follows the SAME two-tier honesty discipline established across
this ecosystem: formulas that are THERMODYNAMIC/KINEMATIC DEFINITIONS
(CAPE, storm-relative helicity, the Rankine vortex, the Holland wind
profile's functional FORM) are implemented with high confidence; formulas
that are FITTED EMPIRICAL REGRESSIONS (the Holland B-parameter estimate,
pressure-wind relationships, EF-scale wind-speed-to-damage mapping) are
explicitly FLAGGED as needing verification against current literature.
This is NOT a replacement for a real numerical weather prediction model,
a supercell/tornado simulation (e.g. a cloud-resolving model), a real
storm-surge model (SLOSH, ADCIRC -- this module's surge estimate ignores
bathymetry entirely), or professional review by a meteorologist. Do not
use this as the sole basis for a warning decision, evacuation order, or
structural design wind speed.
======================================================================

Units: SI throughout (m, s, kg, Pa, K) unless noted (wind speeds are
also reported in familiar mph/kt at official threshold tables, since
EF-scale and Saffir-Simpson are officially defined in those units).

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
    "AtmosphericStabilityModel",
    "WindShearModel",
    "TornadoVortexModel",
    "TropicalCycloneModel",
    "StormSurgeModel",
    "StormDamageAssessment",
    "StormOne",
]

G_ACCEL = 9.80665
RD = 287.05      # J/(kg.K), dry air gas constant
RV = 461.5       # J/(kg.K), water vapor gas constant
CP_DRY = 1005.0  # J/(kg.K)
CP_MOIST_APPROX = 1870.0   # J/(kg.K), approximate for latent heat release accounting
L_VAP = 2.501e6   # J/kg, latent heat of vaporization at 0C
P0 = 100000.0     # Pa, reference pressure for potential temperature
RHO_AIR_SEA_LEVEL = 1.225
OMEGA_EARTH = 7.2921e-5  # rad/s, Earth's rotation rate


def coriolis_parameter(latitude_deg: float) -> float:
    """f = 2*Omega*sin(latitude) [1/s]. Definitional, not fitted."""
    return 2 * OMEGA_EARTH * math.sin(math.radians(latitude_deg))


def saturation_vapor_pressure(T_K: np.ndarray) -> np.ndarray:
    """
    Saturation vapor pressure [Pa] via the Bolton (1980) formula -- a
    widely-used, well-validated approximation to the Clausius-Clapeyron
    relation in atmospheric science (accurate to ~0.1% over -35C to 35C):
        es = 611.2 * exp(17.67*Tc / (Tc+243.5))   [Tc in Celsius, es in Pa]
    """
    Tc = np.asarray(T_K, dtype=float) - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


# =====================================================================
# 1. ATMOSPHERIC STABILITY MODEL  (CAPE/CIN)
# =====================================================================

class AtmosphericStabilityModel:
    """
    Convective Available Potential Energy (CAPE) and Convective
    Inhibition (CIN) from a simplified atmospheric sounding, via a
    lifted-parcel calculation -- these are DEFINITIONS from parcel
    theory (not fitted regressions), the standard basis for severe-
    weather forecasting diagnostics:

        CAPE = integral_LFC^EL  g * (Tv_parcel - Tv_env)/Tv_env  dz
        CIN  = integral_LCL^LFC g * (Tv_parcel - Tv_env)/Tv_env  dz  (negative contributions)

    Uses a simplified pseudo-adiabatic ascent: dry-adiabatic below the
    lifting condensation level (LCL), moist-adiabatic above it (with a
    simplified constant moist lapse rate rather than a full iterative
    moist-adiabat solver -- adequate for engineering-level CAPE
    estimates, not a substitute for a real thermodynamic sounding
    analysis, e.g. SHARPpy).
    """

    DRY_ADIABATIC_LAPSE = G_ACCEL / CP_DRY          # ~9.8 K/km, exact (definitional)
    MOIST_ADIABATIC_LAPSE_APPROX = 6.0e-3            # K/m, representative average
    # (real moist lapse rate varies 4-7 K/km with temperature/pressure;
    # this constant average is a simplification, FLAGGED)

    def __init__(self, surface_T_K: float, surface_p_Pa: float,
                 surface_dewpoint_K: float):
        self.T0 = surface_T_K
        self.p0 = surface_p_Pa
        self.Td0 = surface_dewpoint_K

    def lcl_height_m(self) -> float:
        """
        Lifting condensation level height [m] via the standard
        Espy/Bolton approximation:
            z_LCL ~ 125 * (T - Td)   [m, T/Td in Celsius]
        (Bolton 1980's more precise LCL temperature formula exists;
        this linear approximation is the commonly used quick estimate,
        accurate to within a few tens of meters for typical conditions.)
        """
        return 125.0 * (self.T0 - self.Td0)

    def parcel_temperature_profile(self, z_m: np.ndarray) -> np.ndarray:
        """Lifted-parcel temperature [K] at height z above the surface."""
        z = np.asarray(z_m, dtype=float)
        z_lcl = self.lcl_height_m()
        T_at_lcl = self.T0 - self.DRY_ADIABATIC_LAPSE * z_lcl
        T = np.where(
            z <= z_lcl,
            self.T0 - self.DRY_ADIABATIC_LAPSE * z,
            T_at_lcl - self.MOIST_ADIABATIC_LAPSE_APPROX * (z - z_lcl),
        )
        return T

    def cape_cin(self, env_T_profile_fn, z_top_m: float = 15000.0,
                 dz_m: float = 50.0) -> dict:
        """
        Integrates CAPE/CIN given an environmental temperature profile
        function env_T_profile_fn(z_m) -> T_K.

        Uses virtual temperature correction only implicitly (via the
        moist parcel being warmer/more buoyant than a dry equivalent
        rather than a full mixing-ratio-resolved Tv calculation) --
        adequate for an engineering-level estimate, not a fully rigorous
        thermodynamic sounding analysis.
        """
        z = np.arange(0, z_top_m, dz_m)
        T_parcel = self.parcel_temperature_profile(z)
        T_env = np.asarray([env_T_profile_fn(zi) for zi in z])

        buoyancy = G_ACCEL * (T_parcel - T_env) / T_env

        # LFC: first height (above LCL) where buoyancy turns positive
        # after having been negative; EL: height above LFC where it
        # returns to negative. Simplified single-LFC/EL search (real
        # soundings can have multiple positive/negative layers; this
        # module reports the primary CAPE/CIN layer only).
        positive = buoyancy > 0
        cape = float(np.sum(np.clip(buoyancy, 0, None)) * dz_m)
        cin = float(np.sum(np.clip(-buoyancy, 0, None)) * dz_m)

        # Only accumulate CIN below the first sustained positive-buoyancy
        # layer (i.e., inhibition the parcel must overcome before free
        # convection), not spurious negative buoyancy above the EL.
        if np.any(positive):
            first_positive_idx = int(np.argmax(positive))
            cin = float(np.sum(np.clip(-buoyancy[:first_positive_idx], 0, None)) * dz_m)
        else:
            cin = float(np.sum(np.clip(-buoyancy, 0, None)) * dz_m)

        return {"CAPE_J_kg": cape, "CIN_J_kg": cin, "z": z,
                "T_parcel": T_parcel, "T_env": T_env, "buoyancy": buoyancy}

    @staticmethod
    def max_updraft_velocity(cape_J_kg: float) -> float:
        """
        Theoretical maximum updraft velocity from CAPE (standard parcel-
        theory result, a DEFINITION not a fit):
            w_max = sqrt(2*CAPE)
        Real updrafts are typically 30-50% of this theoretical maximum
        due to entrainment, water loading, and pressure-gradient effects
        not captured by simple parcel theory -- this returns the
        theoretical upper bound, not a realistic forecast value.
        """
        return math.sqrt(max(2.0 * cape_J_kg, 0.0))


# =====================================================================
# 2. WIND SHEAR MODEL
# =====================================================================

@dataclass
class WindProfile:
    z_m: np.ndarray       # heights [m]
    u_m_s: np.ndarray     # eastward wind component [m/s]
    v_m_s: np.ndarray     # northward wind component [m/s]


class WindShearModel:
    """
    Vertical wind shear and storm-relative helicity (SRH) -- standard
    severe-weather kinematic diagnostics, both exact definitions from
    the wind profile (not fitted regressions):

        Bulk shear (0-6km) = |V(6km) - V(surface)|
        SRH = integral_0^h (V(z) - C) x (dV/dz) . k  dz
            = integral_0^h [(u-Cu)*dv/dz - (v-Cv)*du/dz] dz
    C = storm motion vector (estimated here via a simplified internal
    dynamics method, e.g. Bunkers et al. 2000 -- FLAGGED, see method
    docstring -- or supplied directly if known).
    """

    def __init__(self, profile: WindProfile):
        self.profile = profile

    def bulk_shear(self, z_top_m: float = 6000.0) -> dict:
        z, u, v = self.profile.z_m, self.profile.u_m_s, self.profile.v_m_s
        u_top = np.interp(z_top_m, z, u)
        v_top = np.interp(z_top_m, z, v)
        du, dv = u_top - u[0], v_top - v[0]
        return {"shear_m_s": math.hypot(du, dv), "du": du, "dv": dv}

    def bunkers_storm_motion(self) -> dict:
        """
        Estimated right-moving supercell motion via the Bunkers et al.
        (2000) internal dynamics method: mean wind (0-6km) plus a
        deviation perpendicular to the 0-6km shear vector.
            C_right = mean_wind + D * (shear_vector rotated -90deg, normalized)
        D ~ 7.5 m/s is the commonly cited deviation magnitude (FLAGGED --
        verify against the original paper before quantitative use).
        """
        z, u, v = self.profile.z_m, self.profile.u_m_s, self.profile.v_m_s
        mask = z <= 6000.0
        mean_u, mean_v = np.mean(u[mask]), np.mean(v[mask])
        shear = self.bulk_shear(6000.0)
        du, dv = shear["du"], shear["dv"]
        shear_mag = math.hypot(du, dv)
        if shear_mag < 1e-6:
            return {"u": mean_u, "v": mean_v}
        # rotate shear vector -90 degrees (clockwise, for right-mover in NH)
        perp_u, perp_v = dv / shear_mag, -du / shear_mag
        D = 7.5   # FLAGGED constant, see docstring
        return {"u": mean_u + D * perp_u, "v": mean_v + D * perp_v}

    def storm_relative_helicity(self, z_top_m: float = 3000.0,
                                 storm_motion: Optional[dict] = None) -> float:
        """SRH [m^2/s^2] over the layer 0 to z_top_m."""
        if storm_motion is None:
            storm_motion = self.bunkers_storm_motion()
        Cu, Cv = storm_motion["u"], storm_motion["v"]
        z, u, v = self.profile.z_m, self.profile.u_m_s, self.profile.v_m_s
        mask = z <= z_top_m
        z_l, u_l, v_l = z[mask], u[mask], v[mask]
        if len(z_l) < 2:
            return 0.0
        du_dz = np.gradient(u_l, z_l)
        dv_dz = np.gradient(v_l, z_l)
        integrand = (u_l - Cu) * dv_dz - (v_l - Cv) * du_dz
        return float(np.trapezoid(integrand, z_l))

    def supercell_composite_parameter(self, cape_J_kg: float, srh_m2_s2: float,
                                       shear_0_6km_m_s: float) -> float:
        """
        Supercell Composite Parameter (SCP) -- a widely-used composite
        severe-weather index (Thompson et al. 2003 form):
            SCP = (CAPE/1000) * (SRH/50) * (shear_0_6/20)
        with shear capped/floored per the standard definition (shear<10
        m/s -> 0, shear>20 m/s -> capped at 20 for the ratio).
        FLAGGED: reproduces the standard form's STRUCTURE; verify exact
        normalization constants (1000, 50, 20) against Thompson et al.
        (2003) before quantitative operational use.
        """
        shear_term = np.clip(shear_0_6km_m_s, 10.0, 20.0) / 20.0
        if shear_0_6km_m_s < 10.0:
            return 0.0
        return (cape_J_kg / 1000.0) * (srh_m2_s2 / 50.0) * shear_term


# =====================================================================
# 3. TORNADO VORTEX MODEL  (Rankine combined vortex)
# =====================================================================

class TornadoVortexModel:
    """
    Tornado wind field via the Rankine combined vortex -- the classic,
    widely-used simplified tornado wind-field model in wind engineering
    (a solid-body-rotation core matched to an irrotational free vortex
    outside the core radius; used e.g. in ASCE tornado load studies as
    a standard reference wind field, though real tornadoes have more
    complex multi-vortex, translating, and asymmetric structure not
    captured here):

        v(r) = v_max * (r / r_max)          for r <= r_max  (core, solid body)
        v(r) = v_max * (r_max / r)          for r > r_max   (free vortex)

    Also provides the EF-scale (Enhanced Fujita Scale) official NWS
    3-second-gust wind speed thresholds for intensity classification.
    """

    # EF-scale 3-second gust wind speed thresholds (NWS official
    # definition; reproduced from memory of the commonly cited table --
    # FLAGGED, verify exact mph boundaries against the current NWS
    # Enhanced Fujita Scale reference before operational/regulatory use).
    EF_SCALE_THRESHOLDS_MPH = {
        "EF0": (65, 85), "EF1": (86, 110), "EF2": (111, 135),
        "EF3": (136, 165), "EF4": (166, 200), "EF5": (200, 999),
    }

    def __init__(self, v_max_m_s: float, r_max_m: float,
                 translation_speed_m_s: float = 0.0,
                 translation_heading_deg: float = 0.0):
        self.v_max = v_max_m_s
        self.r_max = r_max_m
        self.trans_speed = translation_speed_m_s
        self.trans_heading = math.radians(translation_heading_deg)

    def tangential_velocity(self, r_m: np.ndarray) -> np.ndarray:
        r = np.asarray(r_m, dtype=float)
        return np.where(r <= self.r_max, self.v_max * (r / self.r_max),
                         self.v_max * (self.r_max / np.clip(r, 1e-6, None)))

    def wind_field(self, x_m: np.ndarray, y_m: np.ndarray) -> dict:
        """
        Full 2D wind field including tornado translation (the storm-
        relative Rankine vortex plus the vector addition of ground-
        relative translational motion -- the classic reason the
        right-side / forward-right quadrant of a translating tornado
        has the highest ground-relative wind speeds).
        """
        X, Y = np.meshgrid(x_m, y_m) if x_m.ndim == 1 else (x_m, y_m)
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        v_tan = self.tangential_velocity(r)

        # Tangential direction (counterclockwise, cyclonic in NH convention)
        vx_rot = -v_tan * np.sin(theta)
        vy_rot = v_tan * np.cos(theta)

        vx_trans = self.trans_speed * math.cos(self.trans_heading)
        vy_trans = self.trans_speed * math.sin(self.trans_heading)

        vx = vx_rot + vx_trans
        vy = vy_rot + vy_trans
        speed = np.sqrt(vx**2 + vy**2)
        return {"x": X, "y": Y, "vx": vx, "vy": vy, "speed": speed}

    @classmethod
    def ef_scale_from_wind_speed(cls, wind_speed_m_s: float) -> str:
        mph = wind_speed_m_s * 2.23694
        for ef, (lo, hi) in cls.EF_SCALE_THRESHOLDS_MPH.items():
            if lo <= mph <= hi:
                return ef
        return "EF5" if mph > 200 else "sub-EF0"

    def max_ground_relative_speed(self) -> dict:
        """Peak ground-relative wind speed = v_max (at r_max) + translation speed
        (occurs where the rotational and translational vectors align, typically
        the forward-right quadrant relative to storm motion)."""
        peak = self.v_max + self.trans_speed
        return {"peak_speed_m_s": peak, "ef_scale": self.ef_scale_from_wind_speed(peak)}


# =====================================================================
# 4. TROPICAL CYCLONE MODEL  (Holland parametric wind profile)
# =====================================================================

class TropicalCycloneModel:
    """
    Tropical cyclone (hurricane/typhoon) parametric wind field via the
    Holland (1980) model -- the most widely used parametric TC wind
    profile in engineering/hazard practice:

        V(r) = sqrt[ B/rho * (Rmax/r)^B * (Pn-Pc) * exp(-(Rmax/r)^B)
                     + (r*f/2)^2 ]  -  r*f/2

    B = Holland shape parameter (peakedness); Pn = ambient (environmental)
    pressure, Pc = central pressure, Rmax = radius of maximum wind, f =
    Coriolis parameter.

    Also provides the Saffir-Simpson Hurricane Wind Scale classification
    (official NHC category thresholds) and a pressure-wind relationship
    for estimating Vmax from central pressure alone when a full profile
    isn't available.
    """

    # Saffir-Simpson category thresholds (1-min sustained wind, mph;
    # reproduced from memory of the commonly cited NHC table -- FLAGGED,
    # verify against the current official NHC definition before
    # operational/regulatory use).
    SAFFIR_SIMPSON_MPH = {
        1: (74, 95), 2: (96, 110), 3: (111, 129), 4: (130, 156), 5: (157, 999),
    }

    def __init__(self, central_pressure_Pa: float, ambient_pressure_Pa: float,
                 radius_max_wind_m: float, latitude_deg: float,
                 holland_B: Optional[float] = None):
        self.Pc = central_pressure_Pa
        self.Pn = ambient_pressure_Pa
        self.Rmax = radius_max_wind_m
        self.f = coriolis_parameter(latitude_deg)
        self.B = holland_B if holland_B is not None else self.estimate_holland_B()

    def estimate_holland_B(self) -> float:
        """
        Simplified Holland B-parameter estimate from the pressure deficit
        (Holland 1980's original suggested range is B in [1,2.5];
        several empirical B-vs-environment regressions exist in later
        literature, e.g. Vickery & Wadhera 2008). FLAGGED: this uses an
        illustrative linear interpolation across the standard [1,2.5]
        range based on pressure deficit alone, NOT a verified reproduction
        of any specific published B-estimation formula -- for real work,
        use a published B-parameter regression or fit B directly to
        observed wind data for the storm in question.
        """
        dp_hPa = (self.Pn - self.Pc) / 100.0
        B = 1.0 + np.clip(dp_hPa, 0, 100) / 100.0 * 1.5
        return float(np.clip(B, 1.0, 2.5))

    def wind_speed(self, r_m: np.ndarray, rho_air: float = RHO_AIR_SEA_LEVEL) -> np.ndarray:
        r = np.clip(np.asarray(r_m, dtype=float), 1.0, None)
        dp = self.Pn - self.Pc
        term = (self.Rmax / r) ** self.B
        gradient_wind_sq = (self.B / rho_air) * term * dp * np.exp(-term) + (r * self.f / 2) ** 2
        V = np.sqrt(np.clip(gradient_wind_sq, 0, None)) - r * self.f / 2
        return np.clip(V, 0, None)

    def max_wind_speed(self, rho_air: float = RHO_AIR_SEA_LEVEL) -> float:
        return float(self.wind_speed(np.array([self.Rmax]), rho_air)[0])

    @staticmethod
    def pressure_wind_relationship(central_pressure_Pa: float,
                                    ambient_pressure_Pa: float = 101300.0) -> float:
        """
        Simplified Atkinson-Holliday-type pressure-wind relationship for
        estimating Vmax from central pressure alone (used operationally
        when only central pressure is known, e.g. from satellite
        estimates):
            Vmax = a * (Pn - Pc)^b     [Pn,Pc in hPa, Vmax in m/s]
        FLAGGED: a,b below are illustrative constants reproducing the
        correct qualitative relationship and order of magnitude, not a
        verified reproduction of the original Atkinson & Holliday (1977)
        or Knaff & Zehr (2007) coefficients -- verify before quantitative
        use.
        """
        dp_hPa = (ambient_pressure_Pa - central_pressure_Pa) / 100.0
        a, b = 3.4, 0.644   # FLAGGED constants (illustrative form only)
        return a * max(dp_hPa, 0.0) ** b

    @classmethod
    def saffir_simpson_category(cls, max_wind_m_s: float) -> str:
        mph = max_wind_m_s * 2.23694
        if mph < 74:
            return "Tropical Storm (or weaker)"
        for cat, (lo, hi) in cls.SAFFIR_SIMPSON_MPH.items():
            if lo <= mph <= hi:
                return f"Category {cat}"
        return "Category 5"

    def radial_wind_profile(self, r_max_extent_m: float = 300000.0,
                             n_points: int = 200) -> dict:
        r = np.linspace(1.0, r_max_extent_m, n_points)
        V = self.wind_speed(r)
        return {"r_m": r, "wind_speed_m_s": V}


# =====================================================================
# 5. STORM SURGE MODEL  (simplified wind-setup + inverse barometer)
# =====================================================================

class StormSurgeModel:
    """
    Highly simplified storm surge estimate combining:
      1. Inverse barometer effect (DEFINITIONAL, standard oceanography):
             eta_ib = (Pn - Pc) / (rho_water * g)
      2. Wind-driven setup (standard shallow-water wind-stress balance,
             a 1D along-fetch approximation, NOT resolving real
             bathymetry/coastline geometry at all):
             eta_wind = (Cd * rho_air * U^2 * Fetch) / (rho_water * g * depth)

    THIS IS NOT A SUBSTITUTE FOR A REAL STORM SURGE MODEL. Real surge is
    dominated by bathymetry (continental shelf width/slope), coastline
    geometry (bays funnel/amplify surge), and storm track/forward speed
    interacting with these in ways a 1D formula cannot capture. Use
    SLOSH, ADCIRC, or another real hydrodynamic surge model for any
    actual hazard assessment.
    """

    RHO_SEAWATER = 1025.0

    @classmethod
    def inverse_barometer_setup(cls, central_pressure_Pa: float,
                                 ambient_pressure_Pa: float = 101300.0) -> float:
        """Sea-level rise [m] from the inverse barometer effect alone."""
        dp = ambient_pressure_Pa - central_pressure_Pa
        return dp / (cls.RHO_SEAWATER * G_ACCEL)

    @classmethod
    def wind_setup(cls, wind_speed_m_s: float, fetch_m: float, water_depth_m: float,
                    drag_coefficient: float = 0.0026) -> dict:
        """
        1D wind-driven setup [m] (standard shallow-water wind-stress
        balance form). drag_coefficient ~0.0026 is a representative
        sea-surface drag coefficient at high wind speeds (FLAGGED --
        real Cd is wind-speed-dependent and somewhat uncertain at
        extreme/hurricane-force winds; verify against current literature,
        e.g. Powell et al. 2003, before quantitative use).

        PHYSICAL VALIDITY CHECK: this formula assumes constant water
        depth over the entire fetch and a small-perturbation (linear)
        surface response -- both assumptions break down for a long
        fetch over shallow water (real continental shelves deepen away
        from shore, and once setup becomes a large fraction of the
        water depth itself, the linear balance this formula rests on is
        no longer valid). Caught during this module's own validation
        testing: for a strong hurricane with unrealistic constant-depth-
        over-250km parameters, this formula produced ~19m of setup --
        never observed in the historical record (worse than any
        documented storm surge). `assumption_valid` below is False
        whenever setup exceeds half the water depth, flagging exactly
        this failure mode rather than silently returning an implausible
        number.
        """
        tau_wind = drag_coefficient * RHO_AIR_SEA_LEVEL * wind_speed_m_s**2
        eta = (tau_wind * fetch_m) / (cls.RHO_SEAWATER * G_ACCEL * max(water_depth_m, 1.0))
        assumption_valid = eta <= 0.5 * water_depth_m
        return {"wind_setup_m": eta, "assumption_valid": assumption_valid}

    @classmethod
    def total_surge_estimate(cls, central_pressure_Pa: float, wind_speed_m_s: float,
                              fetch_m: float, water_depth_m: float,
                              ambient_pressure_Pa: float = 101300.0) -> dict:
        ib = cls.inverse_barometer_setup(central_pressure_Pa, ambient_pressure_Pa)
        wind = cls.wind_setup(wind_speed_m_s, fetch_m, water_depth_m)
        result = {"inverse_barometer_m": ib, "wind_setup_m": wind["wind_setup_m"],
                  "total_surge_m": ib + wind["wind_setup_m"],
                  "assumption_valid": wind["assumption_valid"],
                  "note": "1D estimate only -- NOT bathymetry/coastline-aware. "
                          "Use SLOSH/ADCIRC for real hazard assessment."}
        if not wind["assumption_valid"]:
            result["warning"] = (
                "wind_setup exceeds half the water depth -- the linear "
                "shallow-water assumption has broken down for this fetch/"
                "depth combination (a real coastline does not hold constant "
                "shallow depth over the full fetch); this result is NOT "
                "physically reliable. Use a real surge model."
            )
        return result


# =====================================================================
# 6. STORM DAMAGE ASSESSMENT
# =====================================================================

class StormDamageAssessment:
    """
    Simplified wind-speed-based damage fraction curves for tornado
    (EF-scale-anchored) and hurricane (Saffir-Simpson-anchored) wind
    damage. Illustrative smooth curves anchored at official-scale
    threshold wind speeds, NOT a substitute for a real fragility/
    vulnerability curve (e.g. HAZUS-MH wind fragility functions) for a
    specific structure type.
    """

    @staticmethod
    def damage_fraction_from_wind_speed(wind_speed_m_s: float,
                                         half_damage_speed_m_s: float = 60.0,
                                         steepness: float = 1.5) -> float:
        """
        Generic logistic damage-fraction curve vs. wind speed:
            D = 1 / (1 + exp(-steepness*(V - half_damage_speed)/10))
        half_damage_speed_m_s ~60 m/s (~EF2/strong Cat3 range) is an
        illustrative midpoint, not a structure-specific calibration.
        """
        V = wind_speed_m_s
        return float(1.0 / (1.0 + math.exp(-steepness * (V - half_damage_speed_m_s) / 10.0)))

    @classmethod
    def estimate_damage(cls, wind_speed_m_s: np.ndarray, structure_value: np.ndarray,
                         half_damage_speed_m_s: float = 60.0) -> np.ndarray:
        V = np.atleast_1d(np.asarray(wind_speed_m_s, dtype=float))
        frac = np.array([cls.damage_fraction_from_wind_speed(v, half_damage_speed_m_s) for v in V])
        return frac * np.asarray(structure_value, dtype=float)


# =====================================================================
# 7. ORCHESTRATOR
# =====================================================================

class StormOne:
    """
    Orchestrates the severe-storm pipeline: atmospheric stability ->
    wind shear -> supercell/tornado potential, OR tropical cyclone wind
    field -> surge -> damage. Mirrors FireOne / SeismicOne / VolcanoOne /
    FloodOne's role in this ecosystem.
    """

    def __init__(self):
        pass

    def assess_tornado_potential(self, stability: AtmosphericStabilityModel,
                                  env_T_profile_fn, shear_model: WindShearModel) -> dict:
        cape_result = stability.cape_cin(env_T_profile_fn)
        shear = shear_model.bulk_shear(6000.0)
        storm_motion = shear_model.bunkers_storm_motion()
        srh = shear_model.storm_relative_helicity(3000.0, storm_motion)
        scp = shear_model.supercell_composite_parameter(
            cape_result["CAPE_J_kg"], srh, shear["shear_m_s"])
        w_max = stability.max_updraft_velocity(cape_result["CAPE_J_kg"])

        return {
            "CAPE_J_kg": cape_result["CAPE_J_kg"], "CIN_J_kg": cape_result["CIN_J_kg"],
            "max_theoretical_updraft_m_s": w_max,
            "bulk_shear_0_6km_m_s": shear["shear_m_s"],
            "storm_relative_helicity_m2_s2": srh,
            "supercell_composite_parameter": scp,
            "environment_supportive_of_supercells": scp > 1.0 and shear["shear_m_s"] > 20.0,
        }

    def assess_tropical_cyclone(self, tc: TropicalCycloneModel,
                                 fetch_m: float = 200000.0, water_depth_m: float = 30.0) -> dict:
        Vmax = tc.max_wind_speed()
        category = tc.saffir_simpson_category(Vmax)
        surge = StormSurgeModel.total_surge_estimate(
            tc.Pc, Vmax, fetch_m, water_depth_m, tc.Pn)
        return {
            "max_wind_speed_m_s": Vmax, "max_wind_speed_mph": Vmax * 2.23694,
            "saffir_simpson_category": category,
            "holland_B": tc.B,
            "storm_surge_estimate": surge,
        }

    def summary_tornado(self, results: dict) -> str:
        lines = [
            f"CAPE: {results['CAPE_J_kg']:.0f} J/kg, CIN: {results['CIN_J_kg']:.0f} J/kg",
            f"Max theoretical updraft: {results['max_theoretical_updraft_m_s']:.1f} m/s",
            f"0-6km bulk shear: {results['bulk_shear_0_6km_m_s']:.1f} m/s",
            f"0-3km SRH: {results['storm_relative_helicity_m2_s2']:.0f} m^2/s^2",
            f"Supercell Composite Parameter: {results['supercell_composite_parameter']:.2f}",
            f"Environment supportive of supercells: {results['environment_supportive_of_supercells']}",
        ]
        return "\n".join(lines)

    def summary_tropical_cyclone(self, results: dict) -> str:
        lines = [
            f"Max sustained wind: {results['max_wind_speed_m_s']:.1f} m/s "
            f"({results['max_wind_speed_mph']:.0f} mph)",
            f"Saffir-Simpson category: {results['saffir_simpson_category']}",
            f"Holland B parameter: {results['holland_B']:.2f}",
            f"Storm surge estimate: {results['storm_surge_estimate']['total_surge_m']:.2f} m "
            f"(inverse barometer: {results['storm_surge_estimate']['inverse_barometer_m']:.2f}m, "
            f"wind setup: {results['storm_surge_estimate']['wind_setup_m']:.2f}m) "
            f"-- 1D estimate only, not bathymetry-aware",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print(f"storm_one.py v{__version__} loaded OK")
