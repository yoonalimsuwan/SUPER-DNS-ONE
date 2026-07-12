"""
SEISMIC-DNS COUPLING ONE — Interface between SEISMIC ONE and SUPER DNS ONE
============================================================================

Couples earthquake engineering outputs (seismic_one.py) to fluid-dynamics
boundary/initial conditions consumable by a compressible CFD/DNS solver
such as SUPER DNS ONE.

IMPORTANT — INTEGRATION STATUS
--------------------------------
This module was written WITHOUT access to the actual SUPER DNS ONE source
file (it was not available in this session). It therefore exposes a
generic, solver-agnostic interface: plain numpy arrays, interpolators, and
callables with clearly documented physical meaning and units. Every class
has a `# HOOK:` comment marking the exact point where it should be wired
into SUPER DNS ONE's actual boundary-condition / source-term API (field
names, mesh indexing, timestepping call order, etc. will differ from
solver to solver). Treat this as a validated *physics* layer sitting one
adapter away from production use — the adapter itself needs your solver's
real BC hooks to be written against it.

Three coupling scenarios are implemented:

  1. TankSloshingCoupling
     Non-inertial-frame body force for fluid in a rigid or flexible tank
     rigidly attached to a shaking structure/foundation. Standard approach
     in earthquake tank-sloshing CFD: transform to the accelerating
     reference frame of the tank, which introduces a uniform pseudo body
     force  f_body(t) = -a_ground(t)  per unit mass, added to the momentum
     equation alongside gravity. Exact for rigid-tank problems; a good
     approximation for flexible tanks if a rigid dominant mode is assumed
     (for true fluid-structure interaction with tank wall flexibility, the
     wall motion itself needs to come from StructuralResponseLayer applied
     to the tank shell, which is a further extension, not implemented here).

  2. SeabedDisplacementCoupling
     Seabed vertical displacement field (from a fault-rupture or
     user-specified static displacement) mapped onto a free-surface
     initial condition for shallow-water / free-surface Navier-Stokes
     tsunami generation, via the standard Okada-type instantaneous
     seabed-to-surface transfer (long-wavelength limit: surface
     displacement = seabed displacement, appropriate when water depth <<
     horizontal rupture extent).

  3. LiquefiedSoilForcingCoupling
     Where LiquefactionAssessment finds FS < 1, provides an effective
     granular-fluid rheology (Bingham/Herschel-Bulkley parameters as a
     function of depth) so SUPER DNS ONE can be run on the liquefied zone
     as a non-Newtonian flow problem, with the SPT-based relative density
     controlling yield stress via a simple, documented empirical mapping.

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem authorship convention.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional, Callable

from seismic_one import (
    GroundMotion, SeismicOne, LiquefactionAssessment, G_ACCEL,
)

__version__ = "1.0.0"
__all__ = [
    "CFDTimeSeriesBC",
    "TankSloshingCoupling",
    "SeabedDisplacementCoupling",
    "LiquefiedSoilForcingCoupling",
]


# =====================================================================
# Shared: generic time-series boundary-condition container
# =====================================================================

@dataclass
class CFDTimeSeriesBC:
    """
    A generic, solver-agnostic time-dependent boundary/forcing condition.

    Wraps a seismic time history with an interpolator so a CFD solver can
    query the forcing at ITS OWN timestep (typically dt_CFD << dt_seismic
    for explicit compressible DNS, since acoustic/convective CFL is far
    stricter than the ~0.005-0.02s sampling typical of strong-motion
    records). Uses cubic interpolation by default; falls back to linear
    if fewer than 4 samples.
    """
    time: np.ndarray             # [s]
    values: np.ndarray           # shape (n_time,) or (n_time, n_components)
    name: str
    units: str

    def __post_init__(self):
        kind = "cubic" if len(self.time) >= 4 else "linear"
        self._interp = interp1d(
            self.time, self.values, axis=0, kind=kind,
            bounds_error=False, fill_value=(self.values[0], self.values[-1]),
        )

    def at(self, t: float | np.ndarray) -> np.ndarray:
        """Query the forcing/BC value at arbitrary CFD time(s) t [s]."""
        return self._interp(np.clip(t, self.time[0], self.time[-1]))

    def resample(self, dt_cfd: float) -> "CFDTimeSeriesBC":
        """Return a new CFDTimeSeriesBC pre-sampled at the CFD solver's dt."""
        t_new = np.arange(self.time[0], self.time[-1], dt_cfd)
        return CFDTimeSeriesBC(time=t_new, values=self.at(t_new), name=self.name, units=self.units)


# =====================================================================
# 1. TANK SLOSHING COUPLING
# =====================================================================

class TankSloshingCoupling:
    """
    Produces the non-inertial-frame body-force time series for fluid
    sloshing CFD, from a structural or free-field ground motion.

    Physics: in the reference frame of a rigidly-shaking tank, the fluid
    momentum equation gains a uniform pseudo body force equal in magnitude
    and opposite in sign to the tank's absolute acceleration:

        du/dt + (u.grad)u = -grad(p)/rho + nu*lap(u) + g_vec + f_seismic(t)
        f_seismic(t) = -a_ground(t)          [m/s^2, per unit mass]

    where a_ground(t) is the ABSOLUTE (total, not relative) acceleration
    of the tank base. If the tank sits on a building floor rather than
    directly on grade, use StructuralResponseLayer's
    `total_acceleration` at the relevant story (already absolute, i.e.
    includes ground motion + relative structural response) rather than
    the free-field surface motion.
    """

    def __init__(self, horizontal_motion: GroundMotion,
                 vertical_motion: Optional[GroundMotion] = None):
        """
        horizontal_motion: absolute horizontal acceleration time history
            of the tank base (e.g. SiteResponseLayer surface motion, or a
            StructuralResponseLayer story's total_acceleration).
        vertical_motion: optional absolute vertical acceleration. If not
            supplied, a common (conservative, code-based) approximation
            is used: vertical PGA ~ 2/3 * horizontal PGA, scaled to match
            the horizontal time history shape (NOT a substitute for an
            actual vertical component if one is available).
        """
        self.horizontal = horizontal_motion
        if vertical_motion is None:
            scale = (2.0 / 3.0)
            self.vertical = GroundMotion(
                time=horizontal_motion.time,
                accel=horizontal_motion.accel * scale,
                dt=horizontal_motion.dt,
                name=horizontal_motion.name + "_vertical_approx",
            )
            self._vertical_is_approx = True
        else:
            self.vertical = vertical_motion
            self._vertical_is_approx = False

    def body_force_series(self, direction: str = "x") -> CFDTimeSeriesBC:
        """
        Returns f_seismic(t) [m/s^2] as a CFDTimeSeriesBC, to be ADDED to
        gravity as a uniform body-force source term in the fluid momentum
        equation. direction: 'x' or 'y' (horizontal), or 'z' (vertical,
        added to/subtracted from gravitational acceleration).

        # HOOK: In SUPER DNS ONE, this should be added into whatever
        # function assembles the per-cell body-force / source term each
        # substep -- e.g. `momentum_source[:, :, :, dir_index] += bc.at(t)`
        # inside the RK substep loop, using the solver's own current time t.
        """
        if direction in ("x", "y"):
            f = -self.horizontal.accel
            src = self.horizontal
        elif direction == "z":
            f = -self.vertical.accel
            src = self.vertical
        else:
            raise ValueError("direction must be 'x', 'y', or 'z'")
        return CFDTimeSeriesBC(time=src.time, values=f, name=f"seismic_body_force_{direction}",
                                units="m/s^2 (added to momentum eq. as uniform source, "
                                      "vertical approximate from horizontal if not measured: "
                                      f"{self._vertical_is_approx if direction == 'z' else 'n/a'}")

    def full_vector_bc(self) -> CFDTimeSeriesBC:
        """3-component [ax, ay, az] body force, ay=0 (no lateral-2 motion supplied)."""
        t = self.horizontal.time
        ax = -self.horizontal.accel
        ay = np.zeros_like(ax)
        az = -self.vertical.accel
        values = np.stack([ax, ay, az], axis=-1)
        return CFDTimeSeriesBC(time=t, values=values, name="seismic_body_force_vector",
                                units="m/s^2, columns=[x,y,z]")


# =====================================================================
# 2. SEABED DISPLACEMENT -> TSUNAMI FREE-SURFACE IC
# =====================================================================

class SeabedDisplacementCoupling:
    """
    Maps a static seabed vertical displacement field (from an external
    fault-rupture model, or a user-specified analytic uplift patch) to a
    free-surface initial condition for tsunami-generation CFD.

    Uses the long-wavelength (shallow-water) approximation: for a seabed
    displacement field with horizontal length scale L much greater than
    water depth h, the free-surface displacement instantaneously mirrors
    the seabed displacement (Kajiura filtering, which smooths short
    wavelengths for finite water depth, is applied as an optional
    correction).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, seabed_uplift: np.ndarray,
                 water_depth: float):
        """
        x, y: 1D coordinate arrays [m] defining a regular grid
        seabed_uplift: 2D array (len(y), len(x)) of vertical seabed
            displacement [m], positive = upward
        water_depth: representative water depth [m] over the source region,
            used for the optional Kajiura smoothing filter
        """
        self.x = x
        self.y = y
        self.seabed_uplift = seabed_uplift
        self.water_depth = water_depth

    @staticmethod
    def okada_analytic_patch(x: np.ndarray, y: np.ndarray, x0: float, y0: float,
                              length: float, width: float, slip: float,
                              strike_deg: float, dip_deg: float, rake_deg: float,
                              depth: float) -> np.ndarray:
        """
        Stylized rectangular-dislocation seabed displacement, capturing the
        characteristic Okada (1985) near-field dip-slip pattern (uplift
        over the shallow/up-dip portion of the rupture, subsidence over
        the deep/down-dip portion, for a thrust mechanism) as a smoothed
        dipole: the difference of two Gaussian lobes centered at the
        up-dip and down-dip edges of the fault plane, with amplitude
        proportional to slip and sin(rake) (positive rake/thrust -> uplift
        on the shallow side, matching real subduction-zone tsunami sources;
        negative rake/normal faulting flips the pattern).

        This is NOT the exact elastic half-space Okada solution (no
        Poisson-ratio-dependent near-field singular terms, no exact
        along-strike taper) -- it is a smooth, sign-correct, non-singular
        approximation adequate for generating a physically-shaped test
        uplift/subsidence field for CFD tsunami-generation testing. For
        real hazard studies, replace with a validated Okada implementation
        or an externally supplied finite-fault slip model.
        """
        X, Y = np.meshgrid(x - x0, y - y0)
        strike = np.radians(strike_deg)
        dip = np.radians(dip_deg)
        rake = np.radians(rake_deg)

        # rotate into fault-strike-aligned coordinates (Xs along strike, Ys up-dip)
        Xs = X * np.cos(strike) + Y * np.sin(strike)
        Ys = -X * np.sin(strike) + Y * np.cos(strike)

        # horizontal (map-view) offset of the fault edges from the surface
        # trace, due to dip: shallow edge is up-dip (toward Ys<0 by
        # convention here), deep edge is down-dip.
        horiz_halfwidth = (width / 2.0) * np.cos(dip)
        updip_center = -horiz_halfwidth
        downdip_center = +horiz_halfwidth

        sigma_dip = max(horiz_halfwidth, 1e-3)
        sigma_strike = max(length / 2.0, 1e-3)

        def lobe(y_center):
            return np.exp(-0.5 * ((Ys - y_center) / sigma_dip) ** 2
                           - 0.5 * (Xs / sigma_strike) ** 2)

        # Depth attenuation: deeper rupture -> smaller, broader surface signal
        depth_atten = np.exp(-depth / max(width, 1.0))

        amplitude = slip * np.sin(rake) * depth_atten
        uplift = amplitude * (lobe(updip_center) - lobe(downdip_center))
        return uplift

    def kajiura_filter(self) -> np.ndarray:
        """
        Apply Kajiura (1963) filtering in the wavenumber domain: short
        seabed-displacement wavelengths (relative to water depth) are
        smoothed out before reaching the free surface, via the transfer
        function T(k) = 1/cosh(k*h).
        """
        ny, nx = self.seabed_uplift.shape
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        T = 1.0 / np.cosh(np.clip(K * self.water_depth, 0, 50))
        F = np.fft.fft2(self.seabed_uplift)
        surface = np.real(np.fft.ifft2(F * T))
        return surface

    def free_surface_ic(self, apply_kajiura: bool = True) -> dict:
        """
        Returns the free-surface initial condition to hand to SUPER DNS
        ONE's tsunami-generation setup: an eta(x,y) field [m] to add to
        the still-water level, and zero initial velocity (long-wave
        generation is essentially instantaneous relative to propagation
        timescales, so u=v=w=0 is the standard tsunami-source IC).

        # HOOK: In SUPER DNS ONE, feed `eta` into the free-surface / VOF
        # initialization routine as a perturbation on top of the
        # hydrostatic still-water surface, on the solver's own mesh (this
        # will require re-gridding `eta` from this module's x,y grid onto
        # the CFD mesh -- e.g. scipy.interpolate.RegularGridInterpolator).
        """
        eta = self.kajiura_filter() if apply_kajiura else self.seabed_uplift.copy()
        return {
            "x": self.x, "y": self.y, "eta": eta,
            "u0": np.zeros_like(eta), "v0": np.zeros_like(eta), "w0": np.zeros_like(eta),
            "max_uplift_m": float(np.max(eta)),
            "max_subsidence_m": float(np.min(eta)),
        }


# =====================================================================
# 3. LIQUEFIED SOIL AS NON-NEWTONIAN FLOW FORCING
# =====================================================================

class LiquefiedSoilForcingCoupling:
    """
    Converts a LiquefactionAssessment result (FS<1 zones) into rheological
    parameters for treating the liquefied soil column as a non-Newtonian
    fluid in SUPER DNS ONE, using a Herschel-Bulkley model:

        tau = tau_y + K * gamma_dot^n           (tau_y=0 reduces to power-law)

    Yield stress mapping (documented, simplified engineering estimate):
    residual undrained shear strength of liquefied sand is correlated to
    (N1)60cs via the Idriss & Boulanger (2008) / Seed & Harder-type
    residual strength charts. Here a smooth closed-form fit to that trend
    is used (Sr_ratio = Sr / sigma_v_eff'):

        Sr_ratio(N1_60cs) ~ 0.03 * exp(0.2 * N1_60cs),  clipped to [0, 0.35]

    This is a simplified, continuous approximation of the (scattered,
    empirical, chart-based) residual strength literature -- adequate for
    coupling to CFD as a first-pass estimate, NOT a substitute for
    site-specific residual strength assessment in a real hazard study.
    """

    def __init__(self, liquefaction_result: dict, viscosity_scale: float = 50.0,
                 flow_index_n: float = 0.5):
        """
        liquefaction_result: output dict from LiquefactionAssessment.assess_profile()
        viscosity_scale: K [Pa.s^n] consistency index for non-liquefied-adjacent
            transitional shear resistance (order-of-magnitude default for
            liquefied sand; calibrate against site data if available)
        flow_index_n: Herschel-Bulkley flow behavior index (n<1 = shear-thinning,
            typical for liquefied granular material)
        """
        self.result = liquefaction_result
        self.K = viscosity_scale
        self.n = flow_index_n

    @staticmethod
    def residual_strength_ratio(N160_cs: np.ndarray) -> np.ndarray:
        ratio = 0.03 * np.exp(0.2 * N160_cs)
        return np.clip(ratio, 0.0, 0.35)

    def herschel_bulkley_profile(self, N160_cs: np.ndarray) -> dict:
        """
        Returns depth-varying Herschel-Bulkley parameters, but ONLY where
        FS < 1 (liquefiable); non-liquefiable layers get tau_y -> inf
        proxy (returned as None / masked) since they should remain solid
        (structural, not fluid) in the coupled simulation.
        """
        FS = self.result["FS"]
        sigma_v_eff = self.result["sigma_v_eff"]
        liquefiable = self.result["liquefiable"]

        Sr_ratio = self.residual_strength_ratio(N160_cs)
        tau_y = Sr_ratio * sigma_v_eff  # Pa

        tau_y_masked = np.where(liquefiable, tau_y, np.nan)

        return {
            "depths": self.result["depths"],
            "liquefiable_mask": liquefiable,
            "tau_y_Pa": tau_y_masked,
            "K_consistency": np.where(liquefiable, self.K, np.nan),
            "flow_index_n": self.n,
            "note": "NaN entries = non-liquefiable depth, treat as rigid/solid "
                    "in the coupled model, not as fluid.",
        }

    def to_cfd_source_zones(self, herschel_bulkley_profile: dict) -> dict:
        """
        Packages liquefied-zone rheology into a depth-indexed lookup ready
        for a CFD solver to assign per-cell viscosity/yield-stress in the
        soil domain.

        # HOOK: In SUPER DNS ONE, this should populate whatever per-cell
        # material-property array the solver uses for variable-viscosity
        # (e.g. `mu_field[cell] = herschel_bulkley_apparent_viscosity(
        #     tau_y_Pa[depth_idx(cell)], K_consistency[depth_idx(cell)],
        #     flow_index_n, local_strain_rate)` computed each substep,
        # with non-liquefiable cells left at solid/near-infinite viscosity
        # or excluded from the fluid domain entirely).
        """
        depths = herschel_bulkley_profile["depths"]
        tau_y = herschel_bulkley_profile["tau_y_Pa"]
        K = herschel_bulkley_profile["K_consistency"]
        n = herschel_bulkley_profile["flow_index_n"]

        def apparent_viscosity(depth_query: float, strain_rate: float, eps: float = 1e-6) -> float:
            idx = np.argmin(np.abs(depths - depth_query))
            if np.isnan(tau_y[idx]):
                return np.inf  # solid, not liquefied
            gdot = max(abs(strain_rate), eps)
            return tau_y[idx] / gdot + K[idx] * gdot ** (n - 1)

        return {
            "depths": depths,
            "tau_y_Pa": tau_y,
            "K_consistency": K,
            "flow_index_n": n,
            "apparent_viscosity_fn": apparent_viscosity,
        }


if __name__ == "__main__":
    print(f"seismic_dns_coupling_one.py v{__version__} loaded OK")
