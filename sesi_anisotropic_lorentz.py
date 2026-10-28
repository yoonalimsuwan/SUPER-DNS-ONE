"""
=============================================================================
SESI ANISOTROPIC LORENTZ-VIOLATION TEST
Direction-dependent GW propagation + sidereal sidebands + preferred-direction scan
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

PHYSICAL IDEA [C: SME gravity sector, Kostelecký & Mewes]
---------------------------------------------------------
If Lorentz symmetry is violated, the effective GW propagation speed can
depend on the DIRECTION of propagation. As the Earth rotates, a fixed
source's direction in the DETECTOR frame changes once per sidereal day.
Therefore an anisotropic correction imprints:

  (1) a direction-dependent dispersion phase, and
  (2) SIDEREAL SIDE_BANDS at f_carrier ± n·f_sidereal (n = 1, 2).

This module implements a PHENOMENOLOGICAL harmonic proxy:

    a(t) = 1 + Σ_{ℓ,m} ξ_{ℓm} Y_{ℓm}( n̂_detector(t) ),   ℓ ≤ 2

where n̂_detector(t) is obtained by rotating the source direction through
the sidereal angle into the local detector frame. The isotropic piece
(ℓ=0) produces NO sidereal modulation; ℓ=1,2 do.

PREFERRED DIRECTION IS A HYPOTHESIS, NOT AN INPUT ASSUMPTION
-----------------------------------------------------------
`preferred_direction_scan()` fits a candidate axis n_* on a sky grid and
ranks them by residual likelihood. It does NOT assume a direction, and it
does NOT claim a detection. The output is a profile + uncertainty handle.

LABELS: [C] SME/sidereal phenomenology  [D] implementation  [O] open

HONEST SCOPE
------------
• Real SME propagation involves spin-weighted harmonics, birefringence and
  frame-dependent coefficients. This is a real-harmonic (ℓ≤2) proxy.
• Sideband presence tests DIRECTION DEPENDENCE, not a specific theory.
• A best-fit sky axis from a noisy toy stream is NOT a measured preferred
  direction. Report the profile, not a point estimate alone.
• GR limit (all ξ=0) must reproduce an unmodulated signal EXACTLY.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.aniso_lorentz")

__all__ = [
    "AnisotropicConfig",
    "real_harmonics",
    "sidereal_angle",
    "celestial_to_local",
    "AnisotropicPropagationModel",
    "PreferredDirectionScan",
    "AnisotropicLorentzTest",
]

SIDEREAL_PERIOD_S = 86164.0905   # mean sidereal day


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnisotropicConfig:
    # Source on the sky (celestial)
    ra_deg: float = 120.0
    dec_deg: float = 35.0

    # Detector site
    latitude_deg: float = 46.45
    longitude_deg: float = -119.41
    gst0_rad: float = 0.0

    # Observation for the modulation trace
    n_hours: float = 12.0
    modulation_sample_interval_s: float = 120.0

    # Waveform
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    f_min: float = 20.0
    f_max: float = 400.0
    n_freq: int = 128

    # Coefficient prior half-width (dimensionless)
    xi_max: float = 0.5
    noise_std: float = 1e-3
    seed: int = 0

    MSUN_SEC: float = 4.925490947e-6
    MPC_M: float = 3.085677581e22


# ═══════════════════════════════════════════════════════════════════════
# [C] REAL SPHERICAL HARMONICS (ℓ ≤ 2)
# ═══════════════════════════════════════════════════════════════════════

def real_harmonics(theta: np.ndarray, phi: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Real orthonormal harmonics up to ℓ=2, evaluated at (theta, phi).
    theta = colatitude, phi = azimuth. Normalized on the sphere.
    """
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    inv = 1.0 / np.sqrt(4.0 * np.pi)
    return {
        "Y00": np.full_like(theta, inv, dtype=float),
        "Y10": np.sqrt(3.0) * ct * inv,
        "Y1c": np.sqrt(3.0) * st * cp * inv,
        "Y1s": np.sqrt(3.0) * st * sp * inv,
        "Y20": np.sqrt(5.0 / 16.0) * (3.0 * ct**2 - 1.0) * inv,
        "Y2c": np.sqrt(15.0) * st * ct * cp * inv,
        "Y2s": np.sqrt(15.0) * st * ct * sp * inv,
        "Y2cc": 0.5 * np.sqrt(15.0) * st**2 * np.cos(2 * phi) * inv,
        "Y2ss": 0.5 * np.sqrt(15.0) * st**2 * np.sin(2 * phi) * inv,
    }


def sidereal_angle(t_s: np.ndarray, gst0_rad: float = 0.0) -> np.ndarray:
    """Earth rotation angle (rad) at elapsed time t since observation start."""
    t = np.atleast_1d(t_s)
    return gst0_rad + 2.0 * np.pi * (t / SIDEREAL_PERIOD_S)


def celestial_to_local(u_celestial: np.ndarray,
                       latitude_deg: float,
                       longitude_deg: float,
                       t_s: np.ndarray,
                       gst0_rad: float = 0.0) -> np.ndarray:
    """
    Rotate a unit direction from celestial to the detector's local ENU frame.

    Steps:
      1. ECI → ECEF by the sidereal angle (minus site longitude)
      2. ECEF → local (East, North, Up) at the site latitude

    Returns array (..., 3) in (E, N, U).
    """
    u = np.atleast_2d(u_celestial)
    psi = sidereal_angle(t_s, gst0_rad) - np.deg2rad(longitude_deg)
    if psi.shape != (len(u),):
        psi = np.broadcast_to(psi, (len(u),))

    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    c, s = np.cos(psi), np.sin(psi)
    X = c * x + s * y
    Y = -s * x + c * y
    Z = np.broadcast_to(z, X.shape)

    lat = np.deg2rad(latitude_deg)
    cl, sl = np.cos(lat), np.sin(lat)
    up = cl * X + sl * Z
    north = -sl * X + cl * Z
    east = Y
    return np.stack([east, north, up], axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# [D] ANISOTROPIC PROPAGATION MODEL
# ═══════════════════════════════════════════════════════════════════════

class AnisotropicPropagationModel:
    """
    [D] Direction-dependent propagation factor and dispersion phase.

        a(t) = 1 + Σ ξ_lm Y_lm(n̂_local(t))
        δΨ(f,t) = (a(t) − 1) · β_D / f

    The factor a(t) modulates amplitude AND supplies the anisotropic
    dispersion; the isotropic piece ξ_00 gives a constant offset only.
    """

    def __init__(self, cfg: AnisotropicConfig):
        self.cfg = cfg
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)

    def source_direction(self) -> np.ndarray:
        ra = np.deg2rad(self.cfg.ra_deg)
        dec = np.deg2rad(self.cfg.dec_deg)
        return np.array([np.cos(dec) * np.cos(ra),
                         np.cos(dec) * np.sin(ra),
                         np.sin(dec)])

    def local_directions(self, t_s: np.ndarray) -> np.ndarray:
        u = self.source_direction()
        return celestial_to_local(
            np.tile(u, (len(np.atleast_1d(t_s)), 1)),
            self.cfg.latitude_deg, self.cfg.longitude_deg,
            np.atleast_1d(t_s), self.cfg.gst0_rad)

    def anisotropy_factor(self, local_dirs: np.ndarray,
                          coeffs: Dict[str, float]) -> np.ndarray:
        """a(t) for each local direction; clamped positive."""
        e, n, up = local_dirs[..., 0], local_dirs[..., 1], local_dirs[..., 2]
        theta = np.arccos(np.clip(up, -1.0, 1.0))
        phi = np.arctan2(n, e)
        Y = real_harmonics(theta, phi)
        a = np.ones_like(theta, dtype=float)
        for key, xi in coeffs.items():
            if key in Y and xi != 0.0:
                a = a + xi * Y[key]
        return np.maximum(a, 1e-3)

    def dispersion_phase(self, a_t: np.ndarray) -> np.ndarray:
        """
        Extra phase from the anisotropic propagation speed.
        Shape (..., n_freq) via broadcasting.
        """
        D_m = self.cfg.distance_mpc * self.cfg.MPC_M
        beta_D = math.pi * D_m / 1e26      # dimensionless proxy scale
        a_t = np.atleast_1d(a_t)
        return -(a_t[:, None] - 1.0) * beta_D / self.freqs[None, :]

    def modulation_trace(self, coeffs: Dict[str, float]
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Low-rate trace a(t) over the observing run (for sideband search)."""
        dt = self.cfg.modulation_sample_interval_s
        n = int(self.cfg.n_hours * 3600.0 / dt)
        t = np.arange(n) * dt
        dirs = self.local_directions(t)
        return t, self.anisotropy_factor(dirs, coeffs)

    def gr_limit_check(self) -> Dict[str, Any]:
        """All coefficients zero → a(t) ≡ 1 exactly."""
        t, a = self.modulation_trace({})
        return {
            "max_abs_deviation_from_1": float(np.max(np.abs(a - 1.0))),
            "passes": bool(np.allclose(a, 1.0, atol=1e-14)),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] SIDEREAL SIDEBAND DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════

class SiderealSidebandSearch:
    """
    [D] Search the modulation trace for power at f_sid and 2 f_sid.

    HONEST: this is a periodogram of the PROPAGATION MODULATION, not a
    full strain sideband search. It isolates the direction-dependence
    signature without needing the GW carrier in the same FFT.
    """

    def __init__(self, cfg: AnisotropicConfig):
        self.cfg = cfg

    def search(self, t_s: np.ndarray, trace: np.ndarray) -> Dict[str, Any]:
        fs = 1.0 / np.median(np.diff(t_s))
        x = trace - np.mean(trace)
        n = len(x)
        power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        f_sid = 1.0 / SIDEREAL_PERIOD_S
        median_power = float(np.median(power[1:])) if len(power) > 1 else 0.0
        floor = max(median_power, 1e-30)

        def peak_near(target: float) -> float:
            if target > freqs[-1]:
                return 0.0
            idx = int(np.argmin(np.abs(freqs - target)))
            lo, hi = max(0, idx - 2), min(len(power), idx + 3)
            return float(np.max(power[lo:hi]) / floor)

        return {
            "f_sid_hz": f_sid,
            "sidereal_power_ratio_1": peak_near(f_sid),
            "sidereal_power_ratio_2": peak_near(2 * f_sid),
            "anisotropy_detected": bool(peak_near(f_sid) > 20.0),
            "note": "periodogram of propagation modulation, not full strain",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] PREFERRED-DIRECTION SCAN (hypothesis, not assumption)
# ═══════════════════════════════════════════════════════════════════════

class PreferredDirectionScan:
    """
    [D] Rank candidate preferred axes n_* by how well the projection
        d(t) = n̂_local(t) · n_*
    explains the observed modulation m(t) = a(t) − 1.

    For each axis, fit a single amplitude ξ by least squares:
        ξ̂ = (d·m)/(d·d),   residual = m − ξ̂ d

    Returns the ranked profile. The axis is a FIT RESULT, never an input.
    """

    def __init__(self, cfg: AnisotropicConfig, n_axis: int = 72):
        self.cfg = cfg
        self.n_axis = n_axis

    def candidate_axes(self) -> np.ndarray:
        """Roughly uniform sphere via a Fibonacci lattice."""
        k = np.arange(self.n_axis) + 0.5
        z = 1.0 - 2.0 * k / self.n_axis
        r = np.sqrt(np.maximum(1.0 - z**2, 0.0))
        phi = k * (math.pi * (3.0 - math.sqrt(5.0)))
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)

    def scan(self, local_dirs: np.ndarray,
             modulation: np.ndarray) -> Dict[str, Any]:
        axes = self.candidate_axes()
        m = np.asarray(modulation) - 1.0
        rows = []
        for axis in axes:
            d = local_dirs @ axis
            dd = float(d @ d)
            xi_hat = float(d @ m) / dd if dd > 1e-12 else 0.0
            xi_hat = float(np.clip(xi_hat, -self.cfg.xi_max, self.cfg.xi_max))
            resid = m - xi_hat * d
            sse = float(resid @ resid)
            ra = math.degrees(math.atan2(axis[1], axis[0])) % 360.0
            dec = math.degrees(math.asin(float(np.clip(axis[2], -1, 1))))
            rows.append({"ra_deg": ra, "dec_deg": dec,
                         "xi_hat": xi_hat, "sse": sse})
        rows.sort(key=lambda r: r["sse"])

        # Profile uncertainty proxy: ΔSSE = 1 defines ~1σ in the toy model
        best = rows[0]
        sigma_sse = max(float(np.var(m)) * len(m) / max(self.n_axis, 1), 1e-12)
        within = [r for r in rows if r["sse"] - best["sse"] <= sigma_sse]

        return {
            "best_axis": best,
            "n_within_1sigma_proxy": len(within),
            "sky_coverage_of_1sigma": len(within) / len(rows),
            "profile_top10": rows[:10],
            "honest_note": (
                "A best-fit axis is NOT a measured preferred direction. "
                "Report the profile; with a short toy stream the 1σ region "
                "typically covers much of the sky."),
        }


# ═══════════════════════════════════════════════════════════════════════
# TEST DRIVER
# ═══════════════════════════════════════════════════════════════════════

class AnisotropicLorentzTest:
    """[D] GR-limit validation + isotropic/aniso sidebands + direction scan."""

    def __init__(self, cfg: Optional[AnisotropicConfig] = None):
        self.cfg = cfg or AnisotropicConfig()
        self.model = AnisotropicPropagationModel(self.cfg)
        self.sidebands = SiderealSidebandSearch(self.cfg)
        self.scanner = PreferredDirectionScan(self.cfg)

    def run(self) -> Dict[str, Any]:
        report = {"gr_limit": self.model.gr_limit_check()}

        # Isotropic modulation: ξ_00 only → constant a(t), no sidereal power
        t, a_iso = self.model.modulation_trace({"Y00": 0.2})
        report["isotropic_sidebands"] = self.sidebands.search(t, a_iso)

        # Anisotropic: dipole in the local Up direction → sidereal modulation
        t, a_aniso = self.model.modulation_trace({"Y10": 0.2})
        report["anisotropic_sidebands"] = self.sidebands.search(t, a_aniso)

        # Direction scan on a synthetic dipole along local Up at t=0
        dirs = self.model.local_directions(t)
        rng = np.random.default_rng(self.cfg.seed)
        m_obs = 1.0 + 0.15 * dirs[:, 2] + rng.normal(0, self.cfg.noise_std, len(t))
        report["preferred_direction"] = self.scanner.scan(dirs, m_obs)

        report["honest_note"] = (
            "Real-harmonic ℓ≤2 proxy, not full SME. Sidebands test direction "
            "dependence only. A best-fit axis is a hypothesis, not a discovery.")
        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Anisotropic Lorentz Test — Smoke Test")
    print("=" * 72)

    cfg = AnisotropicConfig(n_hours=12.0, modulation_sample_interval_s=120.0)
    test = AnisotropicLorentzTest(cfg)
    r = test.run()

    print("\n  ── [1] GR limit ──")
    g = r["gr_limit"]
    print(f"    max |a−1| = {g['max_abs_deviation_from_1']:.2e}")
    print(f"    [{'PASS' if g['passes'] else 'FAIL'}] ξ=0 ⇒ unmodulated")

    print("\n  ── [2] Sidereal sidebands ──")
    for key in ["isotropic_sidebands", "anisotropic_sidebands"]:
        s = r[key]
        print(f"    {key}: 1×f_sid power ratio = {s['sidereal_power_ratio_1']:.1f}, "
              f"anisotropy_detected = {s['anisotropy_detected']}")

    iso_ok = r["isotropic_sidebands"]["sidereal_power_ratio_1"] < 20.0
    aniso_ok = r["anisotropic_sidebands"]["sidereal_power_ratio_1"] > 20.0
    print(f"    [{'PASS' if iso_ok else 'FAIL'}] isotropic ⇒ no sidereal sideband")
    print(f"    [{'PASS' if aniso_ok else 'FAIL'}] anisotropic ⇒ sidereal sideband")

    print("\n  ── [3] Preferred-direction profile ──")
    p = r["preferred_direction"]
    b = p["best_axis"]
    print(f"    best axis: RA={b['ra_deg']:.1f}°, Dec={b['dec_deg']:.1f}°, "
          f"ξ̂={b['xi_hat']:.3f}")
    print(f"    1σ-proxy sky coverage: {p['sky_coverage_of_1sigma']:.2f}")
    print(f"    {p['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • ℓ≤2 real-harmonic proxy, not full SME spin-weighted propagation")
    print("  • Sidebands test DIRECTION DEPENDENCE, not a specific theory")
    print("  • Preferred axis is a fit result; report the profile, not a point")
    print("  • GR limit validated exactly before interpreting any signal")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
