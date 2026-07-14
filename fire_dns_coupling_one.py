"""
FIRE-DNS COUPLING ONE — Interface between FIRE ONE and SUPER DNS ONE
========================================================================
# Author : PAI , Yoon A Limsuwan / MSPS NETWORK
# License: MIT

Couples fire engineering outputs (fire_one.py) to CFD source terms for
SUPER DNS ONE, via the canonical bridges in one_core_v3.py.

Two physical mechanisms, both real, verified hooks (not speculative):

  1. Buoyancy — fire plumes are driven by buoyant acceleration, and (as
     found while building seismic_dns_coupling_one.py) SUPER DNS ONE has
     NO gravity term anywhere in _compute_rhs. Buoyancy is therefore
     wired through the SAME SeismicDNSBridge used for seismic body
     forces: gravity is just a constant "ground acceleration" of
     magnitude G_ACCEL directed along whichever axis is "up" in your
     grid. No new bridge class needed -- this reuses
     one_core_v3.SeismicDNSBridge(dns, accel_z=G_ACCEL) directly.

  2. Heat release — combustion (and net radiative gain/loss) is a direct
     volumetric energy-equation source, which SeismicDNSBridge's
     mechanical-work-only buffers cannot represent. Wired through the new
     one_core_v3.HeatReleaseDNSBridge (v3.3.0+), which requires
     super_dns_one_v6_3.py v6.5+ (new _ext_q buffer).

FireSourceField shapes a DesignFireCurve's HRR(t) into a spatial
Gaussian-plume-like volumetric heat source at the fire's (x,y) location
and base height, consistent with the fire's own flame-height/plume-width
scale from PlumeCorrelations -- not a point source (which would be
singular in a finite-volume energy equation).

======================================================================
 LIFE-SAFETY NOTICE
======================================================================
See fire_one.py's module docstring. CFD-coupled fire simulation using
these modules is an engineering estimate, not a certified life-safety
tool. Real fire CFD work should be validated against NIST FDS and/or
experimental data before informing any actual evacuation or
firefighting decision.
======================================================================

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem attribution convention.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional

from fire_one import DesignFireCurve, PlumeCorrelations, G_ACCEL, T_AMBIENT

__version__ = "1.0.0"
__all__ = ["FireSourceField", "make_buoyancy_dns_bridge", "make_heat_release_dns_bridge"]


class FireSourceField:
    """
    Shapes a DesignFireCurve's HRR(t) [kW] into a 3D volumetric heat
    source field [W/m^3], for use with one_core_v3.HeatReleaseDNSBridge
    (via its duck-typed `.field(t) -> Tensor` protocol).

    The fire's HRR is distributed over a Gaussian volume centered at
    (x0, y0, z0) with horizontal std matching the fire diameter and
    vertical extent matching the Heskestad flame height at each instant
    (a physically motivated, non-singular volumetric proxy for
    "combustion happens within the flame envelope" -- NOT a resolved
    flame-sheet/mixture-fraction CFD combustion model; for that level of
    fidelity this source should feed a resolved species-transport
    solver, which SUPER DNS ONE does not currently have -- see the
    conversation's note on required physics for true fire CFD).
    """

    def __init__(
        self,
        fire_curve: DesignFireCurve,
        x0: float, y0: float, z0: float,
        diameter_m: float,
        grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
        chi_r: float = 0.35,
    ):
        """
        grid_x/y/z: 1D physical coordinate arrays for the DNS mesh axes
            (e.g. from CFDConfig), used to build the Gaussian source
            field on the solver's actual grid.
        """
        self.fire_curve = fire_curve
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.D = diameter_m
        self.chi_r = chi_r
        self.gx, self.gy, self.gz = grid_x, grid_y, grid_z

        X, Y, Z = np.meshgrid(grid_x - x0, grid_y - y0, grid_z - z0, indexing="ij")
        self._X2 = X**2
        self._Y2 = Y**2
        self._Z = Z   # keep raw Z for flame-height-dependent vertical shaping
        self._dV = (
            (grid_x[1] - grid_x[0]) * (grid_y[1] - grid_y[0]) * (grid_z[1] - grid_z[0])
        )

    def field(self, t: float) -> np.ndarray:
        """
        Returns the volumetric heat source [W/m^3] at time t, normalized
        so its integral over the grid equals the convective HRR at t
        (radiative fraction chi_r is assumed to leave the domain as a
        separate loss, not deposited locally -- see fire_one.RadiationModel
        for the optically-thin volumetric-loss term, which should be
        SUBTRACTED separately via the same _ext_q channel if you want net
        radiative exchange represented; this method alone only supplies
        the combustion source, matching common simplified fire-CFD
        practice of treating radiative loss as a fixed fraction removed
        at the source rather than resolving its transport).
        """
        q_kw = float(self.fire_curve.hrr_kw(np.array([t]))[0])
        if q_kw <= 1e-9:
            return np.zeros_like(self._X2)

        q_conv_w = PlumeCorrelations.convective_hrr(q_kw, self.chi_r) * 1000.0  # W
        Lf = max(PlumeCorrelations.flame_height(q_kw, self.D), 0.3 * self.D)

        sigma_xy = max(self.D / 2.355, 1e-3)   # D = FWHM-ish horizontal extent
        sigma_z  = max(Lf / 2.355, 1e-3)

        shape = np.exp(-0.5 * (self._X2 + self._Y2) / sigma_xy**2) * \
                np.exp(-0.5 * (self._Z - Lf / 2.0)**2 / sigma_z**2)
        # Only the region above the base (Z>=0) receives heat (flame sits
        # above the fuel surface, not below grade).
        shape = np.where(self._Z >= 0, shape, 0.0)

        integral = shape.sum() * self._dV
        if integral < 1e-12:
            return np.zeros_like(shape)
        return shape * (q_conv_w / integral)   # W/m^3, integrates to q_conv_w


def make_buoyancy_dns_bridge(dns_solver, vertical_axis: str = "z"):
    """
    Buoyancy via the canonical SeismicDNSBridge: gravity is just a
    constant downward "ground acceleration" of magnitude G_ACCEL. No new
    bridge class needed -- see this module's docstring for why.

    Args:
        vertical_axis: which DNS axis is "up" ('x', 'y', or 'z').

    Returns a one_core_v3.SeismicDNSBridge ready to sync()/step() in your
    own loop, exactly like the seismic tank-sloshing usage.
    """
    from one_core_v3 import SeismicDNSBridge
    kwargs = {"accel_x": None, "accel_y": None, "accel_z": None}
    kwargs[f"accel_{vertical_axis}"] = G_ACCEL
    return SeismicDNSBridge(dns_solver, **kwargs)


def make_heat_release_dns_bridge(dns_solver, fire_source: FireSourceField):
    """
    Wires a FireSourceField into the canonical HeatReleaseDNSBridge.
    Raises RuntimeError (via HeatReleaseDNSBridge's own constructor check)
    if dns_solver predates the v6.5 _ext_q fix.
    """
    from one_core_v3 import HeatReleaseDNSBridge
    return HeatReleaseDNSBridge(dns_solver, q_dot=fire_source)


if __name__ == "__main__":
    print(f"fire_dns_coupling_one.py v{__version__} loaded OK")
