"""
FLOOD ONE — Flood Hydrology & Hydraulics Engineering Module for the ONE Ecosystem
=====================================================================================
# Developer  : PAI , Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT



Flood hazard analysis, structured like fire_one.py / seismic_one.py / volcano_one.py:
    1. RainfallModel           — design storm intensity-duration-frequency (IDF)
    2. WatershedHydrology      — SCS Curve Number rainfall-runoff, time of concentration
    3. ChannelHydraulics       — Manning's equation, normal/critical depth, Froude number
    4. FloodRouting            — Muskingum channel routing
    5. DamBreachModel          — simplified parametric dam-breach outflow hydrograph
    6. FloodplainModel         — HAND-based floodplain extent, depth-damage assessment
    7. FloodOne                — orchestrator

======================================================================
 LIFE-SAFETY NOTICE — READ BEFORE USE
======================================================================
Flood hazard assessment directly informs evacuation zones, dam-failure
emergency action plans, and floodplain (insurance/zoning) maps -- errors
here are as consequential as this ecosystem's fire/seismic/volcanic
modules. Every correlation is a published, standard hydrology/hydraulics
engineering method (SCS/NRCS Curve Number method, Manning's equation,
Muskingum routing, simplified parametric dam-breach regressions) -- NOT
a replacement for a calibrated hydrologic/hydraulic model (e.g. HEC-HMS,
HEC-RAS, or a real 2D flood model), a site-specific Emergency Action Plan,
or professional review by a hydrologist/hydraulic engineer. Some
empirical regression constants below (particularly DamBreachModel's
breach-formation parameters) are FLAGGED explicitly as needing
verification against current literature before quantitative use -- this
module was built without the ability to cross-check numeric values
against external sources in the session that produced it, the exact
situation that produced calibration errors (caught and fixed before
shipping) in this ecosystem's volcano_one.py module. Do not use this as
the sole basis for an actual evacuation decision, dam safety
determination, or floodplain regulatory map.
======================================================================

Units: SI throughout (m, s, kg, m^3/s) unless noted otherwise.

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
    "RainfallModel",
    "WatershedHydrology",
    "ChannelHydraulics",
    "FloodRouting",
    "DamBreachModel",
    "FloodplainModel",
    "FloodOne",
]

G_ACCEL = 9.80665
RHO_WATER = 1000.0


# =====================================================================
# 1. RAINFALL MODEL  (design storm IDF curves)
# =====================================================================

class RainfallModel:
    """
    Design-storm rainfall intensity via an Intensity-Duration-Frequency
    (IDF) relationship, the standard engineering hydrology form:
        i = a / (t_min + b)^c        [mm/hr]
    (Sherman/Bernard-type IDF equation; a, b, c are region/gauge-specific
    fitted parameters in real practice -- this class ships generic
    illustrative defaults, NOT parameters for any specific real gauge
    station. Always use actual local IDF parameters, e.g. from NOAA
    Atlas 14 (US) or a national meteorological service, for real work.)
    """

    def __init__(self, a: float = 1000.0, b: float = 10.0, c: float = 0.8):
        self.a, self.b, self.c = a, b, c

    def intensity_mm_hr(self, duration_min: np.ndarray) -> np.ndarray:
        t = np.asarray(duration_min, dtype=float)
        return self.a / (t + self.b) ** self.c

    def design_hyetograph(self, duration_min: float, dt_min: float = 5.0,
                           method: str = "alternating_block") -> dict:
        """
        Builds a design storm hyetograph (time series of incremental
        rainfall depth) from the IDF curve via the standard Alternating
        Block Method (NRCS/ASCE hydrology textbook method): compute
        cumulative depth at each duration from the IDF curve, take
        successive differences, and arrange blocks with the peak
        centered (standard convention placing the most intense block
        near but not necessarily exactly at the storm center, simplified
        here to center placement).
        """
        n_blocks = int(round(duration_min / dt_min))
        t = np.arange(1, n_blocks + 1) * dt_min
        cum_depth = self.intensity_mm_hr(t) * (t / 60.0)   # mm
        incremental = np.diff(np.concatenate([[0.0], cum_depth]))
        incremental_sorted = np.sort(incremental)[::-1]

        # Alternating block arrangement: build a list of block INDICES in
        # the standard AB pattern (peak block at center, next-largest
        # alternating right-then-left of it), then assign the sorted
        # depths to those indices. Building the index order up front
        # (rather than tracking two independently-advancing left/right
        # pointers with ad hoc fallback branches) avoids the out-of-
        # bounds edge case an earlier version of this method had --
        # caught by this module's own validation testing before shipping.
        mid = n_blocks // 2
        order = [mid]
        left, right = mid - 1, mid + 1
        toggle_right = True
        while len(order) < n_blocks:
            if toggle_right and right < n_blocks:
                order.append(right); right += 1
            elif not toggle_right and left >= 0:
                order.append(left); left -= 1
            elif right < n_blocks:
                order.append(right); right += 1
            elif left >= 0:
                order.append(left); left -= 1
            toggle_right = not toggle_right

        arranged = np.zeros(n_blocks)
        for idx, val in zip(order, incremental_sorted):
            arranged[idx] = val

        return {"time_min": t, "incremental_depth_mm": arranged,
                "total_depth_mm": float(np.sum(arranged))}


# =====================================================================
# 2. WATERSHED HYDROLOGY  (SCS/NRCS Curve Number method)
# =====================================================================

class WatershedHydrology:
    """
    Rainfall-runoff transformation via the SCS (NRCS) Curve Number
    method -- the standard US engineering-hydrology method for
    estimating direct runoff depth from storm rainfall and land-cover/
    soil characteristics (this is a well-defined, widely-tabulated
    method, not an uncertain fitted regression):

        S = (25400/CN) - 254                [mm, potential retention]
        Ia = 0.2 * S                         [mm, initial abstraction]
        Q = (P - Ia)^2 / (P - Ia + S)        [mm, direct runoff], for P > Ia; else Q=0

    Time of concentration via the Kirpich (1940) formula (a standard,
    widely-used empirical formula for small watersheds):
        Tc = 0.0195 * L^0.77 * S^(-0.385)    [minutes, L in meters,
                                               S = average slope, m/m]
    """

    def __init__(self, curve_number: float, area_km2: float,
                 flow_length_m: float, avg_slope: float):
        if not (30 <= curve_number <= 100):
            raise ValueError("curve_number should be in the tabulated NRCS range [30,100]")
        self.CN = curve_number
        self.area_km2 = area_km2
        self.L = flow_length_m
        self.slope = max(avg_slope, 1e-4)

    def potential_retention_mm(self) -> float:
        return (25400.0 / self.CN) - 254.0

    def runoff_depth_mm(self, rainfall_depth_mm: np.ndarray) -> np.ndarray:
        P = np.asarray(rainfall_depth_mm, dtype=float)
        S = self.potential_retention_mm()
        Ia = 0.2 * S
        Q = np.where(P > Ia, (P - Ia) ** 2 / (P - Ia + S), 0.0)
        return Q

    def time_of_concentration_min(self) -> float:
        """Kirpich (1940) formula. L in meters, slope in m/m, Tc in minutes."""
        return 0.0195 * self.L ** 0.77 * self.slope ** (-0.385)

    def peak_discharge_rational_method(self, rainfall_intensity_mm_hr: float,
                                        runoff_coefficient: Optional[float] = None) -> float:
        """
        Rational Method peak discharge (standard for small watersheds,
        <~2.5 km^2, where the storm duration equals Tc):
            Q_peak = C * i * A / 3.6     [m^3/s, i in mm/hr, A in km^2]
        If runoff_coefficient C is not given, estimates it from the CN
        value via a simple monotonic mapping (C roughly correlates with
        CN -- higher CN/more impervious surface -> higher C). This
        mapping is illustrative (a smooth interpolation reproducing the
        correct qualitative range C~0.1-0.95), not a verified regression
        -- use a tabulated C value for the actual land use for real work.
        """
        if runoff_coefficient is None:
            runoff_coefficient = np.clip((self.CN - 30) / 70.0, 0.05, 0.95)
        return runoff_coefficient * rainfall_intensity_mm_hr * self.area_km2 / 3.6

    def scs_triangular_unit_hydrograph(self, rainfall_excess_mm: float) -> dict:
        """
        SCS synthetic triangular unit hydrograph (standard NRCS method):
            Tp = (D/2) + 0.6*Tc         [time to peak, D = rainfall
                                          duration ~ 2*sqrt(Tc) heuristic
                                          if not otherwise specified]
            Qp = 2.08 * A / Tp          [m^3/s per mm of runoff, A in km^2,
                                          Tp in hours -- SCS dimensionless
                                          UH peak-rate factor 484 in
                                          imperial units converts to 2.08
                                          in SI (this conversion IS a
                                          standard, well-documented unit
                                          conversion, not an uncertain fit)]
            Tb = 2.67 * Tp              [base time, triangular UH]
        """
        Tc_hr = self.time_of_concentration_min() / 60.0
        D_hr = max(2 * math.sqrt(Tc_hr), 0.1)
        Tp_hr = D_hr / 2 + 0.6 * Tc_hr
        Qp_per_mm = 2.08 * self.area_km2 / Tp_hr   # m^3/s per mm runoff
        Tb_hr = 2.67 * Tp_hr
        Q_peak = Qp_per_mm * rainfall_excess_mm
        return {"time_to_peak_hr": Tp_hr, "base_time_hr": Tb_hr,
                "peak_discharge_m3_s": Q_peak, "unit_peak_m3_s_per_mm": Qp_per_mm}


# =====================================================================
# 3. CHANNEL HYDRAULICS  (Manning's equation)
# =====================================================================

class ChannelHydraulics:
    """
    Open-channel flow via Manning's equation -- the standard, universally
    used open-channel hydraulics formula (a definition with tabulated
    roughness coefficients n, not an uncertain fitted constant):
        V = (1/n) * R^(2/3) * S^(1/2)     [m/s, SI Manning's equation]
        Q = V * A
    R = hydraulic radius = A/P (flow area / wetted perimeter).

    Supports trapezoidal channel geometry (rectangular is the b=0 side-
    slope-zero special case).
    """

    # Representative Manning's n values (standard textbook table, e.g.
    # Chow 1959) -- illustrative defaults, use site-specific values for
    # real work.
    MANNING_N = {
        "concrete_lined": 0.013, "earth_channel_clean": 0.022,
        "earth_channel_weedy": 0.035, "natural_channel_clean": 0.030,
        "natural_channel_weedy": 0.070, "floodplain_light_brush": 0.050,
        "floodplain_heavy_brush": 0.100,
    }

    def __init__(self, bottom_width_m: float, side_slope_h_per_v: float,
                 manning_n: float, channel_slope: float):
        self.b = bottom_width_m
        self.z = side_slope_h_per_v   # horizontal:vertical, 0 = rectangular
        self.n = manning_n
        self.S0 = max(channel_slope, 1e-6)

    def geometry(self, depth_m: np.ndarray) -> dict:
        y = np.asarray(depth_m, dtype=float)
        area = (self.b + self.z * y) * y
        wetted_perimeter = self.b + 2 * y * math.sqrt(1 + self.z**2)
        top_width = self.b + 2 * self.z * y
        hydraulic_radius = np.where(wetted_perimeter > 0, area / wetted_perimeter, 0.0)
        return {"area": area, "wetted_perimeter": wetted_perimeter,
                "top_width": top_width, "hydraulic_radius": hydraulic_radius}

    def discharge(self, depth_m: np.ndarray) -> np.ndarray:
        """Manning's equation discharge [m^3/s] at a given flow depth."""
        g = self.geometry(depth_m)
        return (1.0 / self.n) * g["area"] * g["hydraulic_radius"] ** (2.0 / 3.0) * self.S0 ** 0.5

    def normal_depth(self, discharge_m3_s: float, y_max: float = 50.0) -> float:
        """
        Solves Manning's equation for normal depth given discharge, via
        bisection (Manning's Q(y) is monotonically increasing in y for
        a physically normal channel, so bisection is guaranteed to
        converge).
        """
        lo, hi = 1e-4, y_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.discharge(np.array([mid]))[0] < discharge_m3_s:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def critical_depth(self, discharge_m3_s: float, y_max: float = 50.0) -> float:
        """
        Critical depth: where Froude number = 1 (Q^2*T/(g*A^3) = 1),
        found via bisection on the same monotonic-in-y basis.
        """
        Q = discharge_m3_s

        def froude_sq_minus_1(y):
            g = self.geometry(np.array([y]))
            A, T = g["area"][0], g["top_width"][0]
            if A <= 0:
                return -1.0
            return (Q**2 * T) / (G_ACCEL * A**3) - 1.0

        lo, hi = 1e-4, y_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if froude_sq_minus_1(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def froude_number(self, depth_m: float, discharge_m3_s: float) -> float:
        g = self.geometry(np.array([depth_m]))
        A, T = g["area"][0], g["top_width"][0]
        if A <= 0:
            return 0.0
        V = discharge_m3_s / A
        D_hydraulic = A / T   # hydraulic depth
        return V / math.sqrt(G_ACCEL * D_hydraulic)

    def flow_regime(self, depth_m: float, discharge_m3_s: float) -> str:
        Fr = self.froude_number(depth_m, discharge_m3_s)
        if Fr < 0.95:
            return "subcritical"
        elif Fr > 1.05:
            return "supercritical"
        return "critical"


# =====================================================================
# 4. FLOOD ROUTING  (Muskingum method)
# =====================================================================

class FloodRouting:
    """
    Channel flood routing via the Muskingum method -- the standard
    hydrologic (storage-based, not full hydrodynamic) routing method for
    propagating an inflow hydrograph through a reach:
        S = K * [X*I + (1-X)*O]           [storage]
    Discretized routing equation (standard derivation, exact given K,X,Δt):
        O2 = C0*I2 + C1*I1 + C2*O1
        C0 = (Δt - 2*K*X) / (2*K*(1-X) + Δt)
        C1 = (Δt + 2*K*X) / (2*K*(1-X) + Δt)
        C2 = (2*K*(1-X) - Δt) / (2*K*(1-X) + Δt)
    K = travel time through the reach [hr], X = weighting factor
    (0 <= X <= 0.5, 0=reservoir-like, ~0.2-0.3 typical natural channels).
    """

    def __init__(self, K_hr: float, X: float):
        if not (0 <= X <= 0.5):
            raise ValueError("Muskingum X must be in [0, 0.5]")
        self.K = K_hr
        self.X = X

    def route(self, inflow_hydrograph: np.ndarray, dt_hr: float,
              initial_outflow: Optional[float] = None) -> np.ndarray:
        I = np.asarray(inflow_hydrograph, dtype=float)
        n = len(I)
        O = np.zeros(n)
        O[0] = initial_outflow if initial_outflow is not None else I[0]

        denom = 2 * self.K * (1 - self.X) + dt_hr
        C0 = (dt_hr - 2 * self.K * self.X) / denom
        C1 = (dt_hr + 2 * self.K * self.X) / denom
        C2 = (2 * self.K * (1 - self.X) - dt_hr) / denom

        for t in range(1, n):
            O[t] = C0 * I[t] + C1 * I[t - 1] + C2 * O[t - 1]
        return np.clip(O, 0.0, None)

    def stability_check(self, dt_hr: float) -> dict:
        """
        Standard Muskingum numerical-stability/accuracy guidance:
            2*K*X <= dt <= 2*K*(1-X)
        Violating the lower bound risks negative routing coefficients
        (C0<0, numerically valid but can produce non-physical
        oscillation); violating the upper bound loses attenuation
        accuracy.
        """
        lower = 2 * self.K * self.X
        upper = 2 * self.K * (1 - self.X)
        return {"dt_hr": dt_hr, "lower_bound_hr": lower, "upper_bound_hr": upper,
                "within_recommended_range": lower <= dt_hr <= upper}


# =====================================================================
# 5. DAM BREACH MODEL  (simplified parametric breach)
# =====================================================================

@dataclass
class DamGeometry:
    height_m: float             # dam height, crest to streambed
    crest_length_m: float
    reservoir_surface_area_km2: float   # at normal pool, for storage estimate
    dam_type: str = "earthen"   # 'earthen' or 'concrete'


class DamBreachModel:
    """
    Simplified parametric dam-breach outflow via a broad-crested-weir
    breach-growth model -- the standard simplified approach used for
    preliminary dam-safety screening (full analysis requires a real
    breach model, e.g. NWS BREACH, HEC-RAS breach module, or physically-
    based erosion modeling).

    Breach geometry parameters (final breach width, time to full breach)
    use simplified regression forms in the spirit of Froehlich (1995) /
    MacDonald & Langridge-Monopolis (1984) -- these are exactly the
    class of empirical regression this module's development already
    demonstrated a real risk of misremembering (see volcano_one.py's
    caught calibration errors). FLAGGED EXPLICITLY: verify against
    current literature (e.g. Froehlich 2016 update) before any real
    dam-safety use -- do not trust the specific numeric coefficients
    below without independent verification.
    """

    def __init__(self, dam: DamGeometry):
        self.dam = dam

    def breach_parameters(self, breach_volume_fraction: float = 0.5) -> dict:
        """
        FLAGGED regression estimates (illustrative form, not verified
        against a specific literature source in this session):
            final_breach_width ~ 3 * H_dam   (typical earthen dam breach
                width is commonly cited as a few times dam height; exact
                multiplier varies significantly with dam material/failure
                mode)
            breach_formation_time ~ 0.5-4 hours depending on dam type,
                here approximated as scaling with sqrt(H_dam) with a
                material-dependent coefficient (illustrative form).
        """
        H = self.dam.height_m
        width_multiplier = 3.0 if self.dam.dam_type == "earthen" else 1.5
        final_width = width_multiplier * H
        time_coeff = 0.3 if self.dam.dam_type == "earthen" else 0.1
        formation_time_hr = time_coeff * math.sqrt(H)
        return {"final_breach_width_m": final_width,
                "breach_formation_time_hr": formation_time_hr}

    def peak_breach_outflow(self, reservoir_head_m: float,
                             breach_params: Optional[dict] = None) -> float:
        """
        Peak breach discharge via the broad-crested weir equation at
        the final breach geometry (standard weir hydraulics -- the
        weir-flow FORM/exponents here are exact/definitional, only the
        breach geometry feeding into it carries the flagged uncertainty
        above):
            Q = Cd * (2/3) * sqrt(2g/3) * B * H^(3/2)   [broad-crested weir]
        Cd ~ 0.6 typical discharge coefficient for a broad-crested weir.
        """
        if breach_params is None:
            breach_params = self.breach_parameters()
        B = breach_params["final_breach_width_m"]
        Cd = 0.6
        H = reservoir_head_m
        return Cd * (2.0 / 3.0) * math.sqrt(2 * G_ACCEL / 3.0) * B * H ** 1.5

    def breach_outflow_hydrograph(self, reservoir_head_m: float,
                                   dt_min: float = 5.0) -> dict:
        """
        Simplified triangular-rise breach hydrograph: outflow grows
        linearly from 0 to peak over the breach formation time, matching
        the common simplified-hydrograph-shape assumption used in
        preliminary dam-breach screening (real breach hydrographs are
        asymmetric with a sharper rise and a recession tail governed by
        reservoir drawdown, not captured by this triangular
        simplification).
        """
        params = self.breach_parameters()
        Q_peak = self.peak_breach_outflow(reservoir_head_m, params)
        t_form_min = params["breach_formation_time_hr"] * 60.0
        n_steps = max(int(t_form_min / dt_min), 2)
        t = np.linspace(0, t_form_min, n_steps)
        Q = Q_peak * (t / t_form_min)
        return {"time_min": t, "outflow_m3_s": Q, "peak_outflow_m3_s": Q_peak,
                "breach_params": params}


# =====================================================================
# 6. FLOODPLAIN MODEL
# =====================================================================

class FloodplainModel:
    """
    Simplified floodplain extent (HAND-inspired: Height Above Nearest
    Drainage) and standard FEMA-style depth-damage assessment.
    """

    @staticmethod
    def inundation_extent_hand(elevation_grid_m: np.ndarray,
                                drainage_elevation_m: float,
                                water_surface_elevation_m: float) -> np.ndarray:
        """
        HAND-based (Height Above Nearest Drainage) binary inundation
        mask: a cell floods if its Height Above Nearest Drainage is less
        than the flood stage above the drainage reference elevation.
        This is a widely-used simplified floodplain-mapping approach
        (Nobre et al. 2011) -- much faster than a full 2D hydraulic
        model, at the cost of not resolving flow dynamics/timing, only
        static inundation extent from a given water-surface elevation.
        """
        hand = elevation_grid_m - drainage_elevation_m
        flood_stage = water_surface_elevation_m - drainage_elevation_m
        return hand <= flood_stage

    @staticmethod
    def inundation_depth(elevation_grid_m: np.ndarray,
                          water_surface_elevation_m: float) -> np.ndarray:
        depth = water_surface_elevation_m - elevation_grid_m
        return np.clip(depth, 0.0, None)

    # Simplified generic depth-damage curve (illustrative shape only --
    # real depth-damage functions are structure-type- and region-
    # specific, e.g. USACE/FEMA HAZUS curves; this is NOT a substitute
    # for those).
    @staticmethod
    def damage_fraction_generic(depth_m: np.ndarray, max_depth_for_total_loss_m: float = 3.0) -> np.ndarray:
        """
        Illustrative generic depth-damage fraction curve: damage rises
        steeply from 0 at zero depth to ~saturating near 1.0 (total loss)
        by max_depth_for_total_loss_m, using a smooth saturating form
        (1-exp) rather than a specific published structure-type curve.
        """
        d = np.clip(np.asarray(depth_m, dtype=float), 0, None)
        return 1.0 - np.exp(-2.0 * d / max_depth_for_total_loss_m)

    @classmethod
    def estimate_damage(cls, depth_m: np.ndarray, structure_value: np.ndarray,
                         max_depth_for_total_loss_m: float = 3.0) -> np.ndarray:
        frac = cls.damage_fraction_generic(depth_m, max_depth_for_total_loss_m)
        return frac * np.asarray(structure_value, dtype=float)


# =====================================================================
# 7. ORCHESTRATOR
# =====================================================================

class FloodOne:
    """
    Orchestrates the flood pipeline: design storm -> watershed runoff ->
    channel routing -> floodplain/damage assessment. Mirrors FireOne /
    SeismicOne / VolcanoOne's role in this ecosystem.
    """

    def __init__(self, watershed: WatershedHydrology, channel: ChannelHydraulics,
                 rainfall: Optional[RainfallModel] = None):
        self.watershed = watershed
        self.channel = channel
        self.rainfall = rainfall or RainfallModel()

    def run_design_storm(self, return_period_label: str, duration_min: float,
                          rainfall_depth_mm: float) -> dict:
        """
        Full pipeline for a single design storm: runoff depth -> unit
        hydrograph peak -> channel normal depth / flow regime at that
        discharge.
        """
        Q_runoff_mm = float(self.watershed.runoff_depth_mm(np.array([rainfall_depth_mm]))[0])
        uh = self.watershed.scs_triangular_unit_hydrograph(Q_runoff_mm)
        Q_peak = uh["peak_discharge_m3_s"]

        y_normal = self.channel.normal_depth(Q_peak)
        y_critical = self.channel.critical_depth(Q_peak)
        regime = self.channel.flow_regime(y_normal, Q_peak)

        return {
            "return_period": return_period_label,
            "rainfall_depth_mm": rainfall_depth_mm,
            "runoff_depth_mm": Q_runoff_mm,
            "time_of_concentration_min": self.watershed.time_of_concentration_min(),
            "peak_discharge_m3_s": Q_peak,
            "normal_depth_m": y_normal,
            "critical_depth_m": y_critical,
            "flow_regime": regime,
        }

    def summary(self, results: dict) -> str:
        lines = []
        lines.append(f"Design storm: {results['return_period']}, "
                      f"{results['rainfall_depth_mm']:.0f}mm rainfall")
        lines.append(f"Runoff depth: {results['runoff_depth_mm']:.1f} mm")
        lines.append(f"Time of concentration: {results['time_of_concentration_min']:.1f} min")
        lines.append(f"Peak discharge: {results['peak_discharge_m3_s']:.1f} m^3/s")
        lines.append(f"Normal depth: {results['normal_depth_m']:.2f} m "
                      f"(critical depth: {results['critical_depth_m']:.2f} m)")
        lines.append(f"Flow regime: {results['flow_regime']}")
        return "\n".join(lines)


if __name__ == "__main__":
    print(f"flood_one.py v{__version__} loaded OK")
