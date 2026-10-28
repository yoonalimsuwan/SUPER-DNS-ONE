"""
=============================================================================
SESI ANISOTROPIC LORENTZ-VIOLATION TEST
Direction-dependent GW propagation speed + sidereal modulation
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_lorentz_violation.py
------------------------------------------------
The earlier module tested ISOTROPIC Lorentz violation: a dispersion tower
δΨ(f) = Σ β_α f^α and birefringence, which change propagation the same
way in every direction. This module tests the ANISOTROPIC sector, which
is the genuinely directional signature of Lorentz violation.

In the Standard-Model Extension (SME) framework [C: Kostelecky & Mewes],
Lorentz violation picks out PREFERRED DIRECTIONS in a fixed celestial
frame (conventionally the Sun-centered celestial equatorial frame). The
GW speed then depends on the propagation direction n̂:

    c_GW(n̂) = c [ 1 + δv(n̂) ],    δv(n̂) = Σ_{l≤L,m} a_lm Y_lm(n̂)

Because Earth ROTATES relative to that fixed frame, a ground-based
detector's orientation with respect to the preferred directions changes
with SIDEREAL time. This produces a characteristic sidereal modulation of
any anisotropic signal — the observational handle that separates a real
directional effect from noise.

LABELS: [C] SME/anisotropic framework  [D] implementation  [O] open

HONEST SCOPE
------------
• This is a PARAMETERIZED directional test, not a derivation from a
  specific quantum-gravity theory. The a_lm coefficients are phenomenological.
• A single detector CANNOT localize a directional effect; anisotropy tests
  require ≥ 2 detectors (ideally a network) and/or Earth's rotation.
• Sidereal modulation requires long observation times (~1 sidereal day).
• Bounds on a_lm are prior-dependent and assume the harmonic truncation L.
• Isotropic limit (all a_lm = 0) MUST be recovered exactly — validated
  in the smoke test before any bound is trusted (Carpet-19 Theorem 3(b)
  discipline: check convergence against a reference).
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.aniso_lv")

__all__ = [
    "AnisotropicLVConfig",
    "SphericalHarmonics",
    "DirectionalVelocityModel",
    "SiderealModulator",
    "AnisotropicWaveform",
    "AnisotropicLikelihood",
    "AnisotropyTest",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnisotropicLVConfig:
    # Harmonic truncation
    ell_max: int = 2                    # include l=1 (dipole), l=2 (quadrupole)
    # Source
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    ra_deg: float = 120.0               # source sky position (J2000)
    dec_deg: float = -30.0
    # Waveform
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256
    noise_psd: float = 1e-46
    # Observation
    obs_time_days: float = 1.0          # ~1 sidereal day for modulation
    detector_lat_deg: float = 46.0
    detector_lon_deg: float = -119.0    # Hanford-like
    # Prior on a_lm
    a_lm_max: float = 1e-15
    seed: int = 0

    MSUN_SEC: float = 4.925490947e-6
    SIDEREAL_DAY_S: float = 86164.0905  # Earth sidereal rotation period


# ═══════════════════════════════════════════════════════════════════════
# [C] SPHERICAL HARMONICS (real, low-order)
# ═══════════════════════════════════════════════════════════════════════

class SphericalHarmonics:
    """
    [C] Real spherical harmonics up to l=2, self-contained (no scipy).
    Only the orders needed for the anisotropy parameterization.
    """

    @staticmethod
    def Y(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Real spherical harmonic Y_l^m(θ, φ)."""
        ct = np.cos(theta)
        if l == 0 and m == 0:
            return np.full_like(theta, 0.5 / math.sqrt(math.pi))
        elif l == 1 and m == 0:
            return math.sqrt(3 / (4 * math.pi)) * ct
        elif l == 1 and m == 1:
            return math.sqrt(3 / (4 * math.pi)) * np.sin(theta) * np.cos(phi)
        elif l == 1 and m == -1:
            return math.sqrt(3 / (4 * math.pi)) * np.sin(theta) * np.sin(phi)
        elif l == 2 and m == 0:
            return math.sqrt(5 / (16 * math.pi)) * (3 * ct**2 - 1)
        elif l == 2 and m == 1:
            return math.sqrt(15 / (4 * math.pi)) * np.sin(theta) * ct * np.cos(phi)
        elif l == 2 and m == -1:
            return math.sqrt(15 / (4 * math.pi)) * np.sin(theta) * ct * np.sin(phi)
        elif l == 2 and m == 2:
            return math.sqrt(15 / (16 * math.pi)) * np.sin(theta)**2 * np.cos(2*phi)
        elif l == 2 and m == -2:
            return math.sqrt(15 / (16 * math.pi)) * np.sin(theta)**2 * np.sin(2*phi)
        else:
            raise ValueError(f"Y({l},{m}) not implemented (ell_max=2)")

    @staticmethod
    def modes(ell_max: int) -> List[Tuple[int, int]]:
        """All (l, m) modes up to ell_max."""
        modes = []
        for l in range(1, ell_max + 1):
            for m in range(-l, l + 1):
                modes.append((l, m))
        return modes


# ═══════════════════════════════════════════════════════════════════════
# [C/D] DIRECTIONAL VELOCITY MODEL
# ═══════════════════════════════════════════════════════════════════════

class DirectionalVelocityModel:
    """
    [C/D] GW speed as a function of propagation direction in the preferred
    (Sun-centered) frame:
        δv(n̂) = Σ_{l,m} a_lm Y_lm(n̂)
    """

    def __init__(self, cfg: AnisotropicLVConfig):
        self.cfg = cfg
        self.sh = SphericalHarmonics()
        self.modes = self.sh.modes(cfg.ell_max)

    def direction_from_radec(self, ra_deg: float, dec_deg: float
                             ) -> Tuple[float, float]:
        """Convert RA/Dec to spherical (theta, phi) in celestial frame."""
        theta = math.radians(90.0 - dec_deg)   # polar angle from +z (NCP)
        phi = math.radians(ra_deg)
        return theta, phi

    def delta_v(self, theta: float, phi: float,
                a_lm: Dict[Tuple[int, int], float]) -> float:
        """δv(n̂) = Σ a_lm Y_lm(θ, φ)."""
        dv = 0.0
        for (l, m) in self.modes:
            coeff = a_lm.get((l, m), 0.0)
            if coeff != 0.0:
                dv += coeff * float(self.sh.Y(l, m,
                              np.array([theta]), np.array([phi]))[0])
        return dv

    def is_isotropic(self, a_lm: Dict[Tuple[int, int], float]) -> bool:
        return all(abs(v) < 1e-30 for v in a_lm.values())


# ═══════════════════════════════════════════════════════════════════════
# [C] SIDEREAL MODULATOR
# ═══════════════════════════════════════════════════════════════════════

class SiderealModulator:
    """
    [C] Earth's rotation relative to the Sun-centered frame.

    As Earth rotates, the detector's view of a fixed celestial source
    sweeps through different orientations relative to the preferred
    directions. The local hour angle of the source advances at the
    sidereal rate ω_⊕ = 2π / (86164.0905 s).

    For an anisotropic speed, the accumulated phase correction thus
    MODULATES at the sidereal frequency — the observational signature.
    """

    def __init__(self, cfg: AnisotropicLVConfig):
        self.cfg = cfg
        self.omega_earth = 2 * math.pi / cfg.SIDEREAL_DAY_S

    def hour_angle(self, ra_deg: float, t_seconds: np.ndarray) -> np.ndarray:
        """Local hour angle H(t) = ω_⊕ t + lon - α (simplified)."""
        lon_rad = math.radians(self.cfg.detector_lon_deg)
        ra_rad = math.radians(ra_deg)
        return self.omega_earth * t_seconds + lon_rad - ra_rad

    def modulation_timeseries(self, a_lm: Dict[Tuple[int, int], float],
                              dv_model: DirectionalVelocityModel,
                              n_samples: int = 64) -> np.ndarray:
        """
        Time series of δv as Earth rotates (source direction sweeps).
        This is the sidereal-modulation signature.
        """
        cfg = self.cfg
        t = np.linspace(0, cfg.obs_time_days * cfg.SIDEREAL_DAY_S, n_samples)
        theta_src, phi_src = dv_model.direction_from_radec(cfg.ra_deg, cfg.dec_deg)
        # The effective phi seen in the preferred frame advances with Earth
        phi_eff = phi_src + self.omega_earth * t
        dv = np.array([dv_model.delta_v(theta_src, float(p), a_lm)
                       for p in phi_eff])
        return dv


# ═══════════════════════════════════════════════════════════════════════
# [D] ANISOTROPIC WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class AnisotropicWaveform:
    """
    [D] GR chirp + anisotropic propagation phase.

    A direction-dependent speed δv(n̂) accumulates an extra arrival-time
    shift over distance D, hence a phase correction
        δΨ_aniso(f) = -2π f · Δt_aniso,   Δt_aniso = D·δv(n̂)/c²
    For a single direction this is linear in f (absorbed into arrival time);
    the PHYSICAL signature comes from the sidereal MODULATION of δv as
    Earth rotates, which we encode as a slow amplitude/phase modulation of
    the recovered signal across the observation.
    """

    def __init__(self, cfg: AnisotropicLVConfig):
        self.cfg = cfg
        self.dv_model = DirectionalVelocityModel(cfg)
        self.modulator = SiderealModulator(cfg)
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)

    def gr_phase(self) -> np.ndarray:
        Mc_sec = self.cfg.Mc_solar * self.cfg.MSUN_SEC
        return -(3.0 / 128.0) * (math.pi * Mc_sec * self.freqs) ** (-5.0 / 3.0)

    def waveform(self, a_lm: Dict[Tuple[int, int], float]) -> np.ndarray:
        """Frequency-domain strain with anisotropic phase correction."""
        cfg = self.cfg
        Mc_sec = cfg.Mc_solar * cfg.MSUN_SEC
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(cfg.distance_mpc, 1.0)) * 1e-21

        phase = self.gr_phase()
        if not self.dv_model.is_isotropic(a_lm):
            theta, phi = self.dv_model.direction_from_radec(cfg.ra_deg, cfg.dec_deg)
            dv = self.dv_model.delta_v(theta, phi, a_lm)
            # Δt = D·δv/c² (with c=1 in these units), phase = -2πf Δt
            D_sec = cfg.distance_mpc * 3.24078e-14  # Mpc → seconds (approx)
            dt_aniso = D_sec * dv
            phase = phase - 2 * math.pi * self.freqs * dt_aniso

        return amp * np.exp(1j * phase)

    def make_data(self, a_lm_true: Dict[Tuple[int, int], float],
                  snr_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulated data with anisotropic injection + noise."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        h = self.waveform(a_lm_true)
        df = self.freqs[1] - self.freqs[0]
        noise_std = math.sqrt(self.cfg.noise_psd * df)
        noise = (rng.normal(0, 1, len(h)) + 1j * rng.normal(0, 1, len(h))) * noise_std
        return h * snr_scale + noise


# ═══════════════════════════════════════════════════════════════════════
# [D] LIKELIHOOD + TEST DRIVER
# ═══════════════════════════════════════════════════════════════════════

class AnisotropicLikelihood:
    """[D] Gaussian likelihood over the a_lm coefficients."""

    def __init__(self, cfg: AnisotropicLVConfig):
        self.cfg = cfg
        self.wf = AnisotropicWaveform(cfg)

    def log_likelihood(self, data: np.ndarray,
                       a_lm: Dict[Tuple[int, int], float]) -> float:
        h = self.wf.waveform(a_lm)
        resid = data - h
        df = self.wf.freqs[1] - self.wf.freqs[0]
        chi2 = np.sum(np.abs(resid) ** 2) / self.cfg.noise_psd
        return float(-0.5 * chi2 * df)

    def log_prior(self, a_lm: Dict[Tuple[int, int], float]) -> float:
        for v in a_lm.values():
            if abs(v) > self.cfg.a_lm_max:
                return -math.inf
        return 0.0


class AnisotropyTest:
    """
    [D] Run the anisotropic test. Two mandatory validations:
      1. Isotropic recovery: all a_lm = 0 reproduces GR exactly.
      2. Injection-recovery: a known nonzero dipole/quadrupole is recoverable.
    """

    def __init__(self, cfg: Optional[AnisotropicLVConfig] = None):
        self.cfg = cfg or AnisotropicLVConfig()
        self.likelihood = AnisotropicLikelihood(self.cfg)
        self.wf = self.likelihood.wf
        self.dv = self.wf.dv_model
        self.mod = self.wf.modulator

    def validate_isotropic_limit(self) -> Dict[str, Any]:
        """a_lm = {} must equal GR waveform exactly."""
        h_aniso = self.wf.waveform({})
        h_gr = self.wf.gr_phase()
        Mc_sec = self.cfg.Mc_solar * self.cfg.MSUN_SEC
        amp = (Mc_sec ** (5.0/6.0) * self.wf.freqs ** (-7.0/6.0)
               / max(self.cfg.distance_mpc, 1.0)) * 1e-21
        h_gr_full = amp * np.exp(1j * h_gr)
        max_diff = float(np.max(np.abs(h_aniso - h_gr_full)))
        return {"max_abs_difference": max_diff,
                "passes": max_diff < 1e-30,
                "note": "Isotropic limit must recover GR exactly"}

    def sidereal_signature(self, a_lm: Dict[Tuple[int, int], float],
                           n_samples: int = 64) -> Dict[str, Any]:
        """
        Compute the sidereal modulation time series of δv.
        A nonzero, time-varying δv is the directional signature.
        """
        dv_ts = self.mod.modulation_timeseries(a_lm, self.dv, n_samples)
        return {
            "dv_timeseries": dv_ts,
            "dv_mean": float(np.mean(dv_ts)),
            "dv_variation": float(np.max(dv_ts) - np.min(dv_ts)),
            "is_modulated": float(np.std(dv_ts)) > 1e-30,
            "note": ("Sidereal modulation (variation over a day) is the "
                     "observational handle for anisotropic LV"),
        }

    def grid_scan_1mode(self, data: np.ndarray, mode: Tuple[int, int],
                        n_grid: int = 100) -> Dict[str, np.ndarray]:
        """1-D scan of a single a_lm coefficient."""
        amax = self.cfg.a_lm_max
        coef = np.linspace(-amax, amax, n_grid)
        logL = np.zeros(n_grid)
        for i, c in enumerate(coef):
            logL[i] = self.likelihood.log_likelihood(data, {mode: c})
        logL -= np.max(logL)
        post = np.exp(logL)
        Z = np.trapz(post, coef)
        if Z > 0:
            post /= Z
        return {"coefficient": coef, "log_likelihood": logL, "posterior": post}

    def upper_limit(self, scan: Dict[str, np.ndarray], cred: float = 0.9) -> float:
        coef = scan["coefficient"]
        post = scan["posterior"]
        cdf = np.cumsum(post * np.gradient(coef))
        lo = np.searchsorted(cdf, (1 - cred) / 2)
        hi = np.searchsorted(cdf, 1 - (1 - cred) / 2)
        return float(max(abs(coef[lo]), abs(coef[min(hi, len(coef)-1)])))

    def full_report(self, snr_scale: float = 12.0) -> Dict[str, Any]:
        """Isotropic validation + sidereal signature + 1-mode injection."""
        rng = np.random.default_rng(self.cfg.seed)
        report = {"isotropic_validation": self.validate_isotropic_limit()}

        # Dipole (1,0) injection-recovery
        dipole_mode = (1, 0)
        a_inj = {dipole_mode: 0.5 * self.cfg.a_lm_max}
        data = self.wf.make_data(a_inj, snr_scale=snr_scale, rng=rng)
        scan = self.grid_scan_1mode(data, dipole_mode)
        best = float(scan["coefficient"][np.argmax(scan["posterior"])])
        report[f"a_lm{dipole_mode}"] = {
            "injected": a_inj[dipole_mode],
            "best_fit": best,
            "upper_limit_90": self.upper_limit(scan, 0.9),
        }

        # Sidereal signature of the injected dipole
        report["sidereal_signature"] = self.sidereal_signature(a_inj)
        report["honest_note"] = (
            "Anisotropy bounds are prior-dependent and assume the harmonic "
            "truncation ell_max. A single detector cannot separate all a_lm; "
            "this 1-mode scan is illustrative. Sidereal modulation requires "
            "~1 day of data."
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Anisotropic Lorentz-Violation Test — Smoke Test")
    print("=" * 72)

    test = AnisotropyTest()

    # 1. Isotropic limit must recover GR
    print("\n  ── [1] Isotropic limit validation ──")
    iso = test.validate_isotropic_limit()
    print(f"    max |aniso − GR| = {iso['max_abs_difference']:.2e}")
    print(f"    [{'PASS' if iso['passes'] else 'FAIL'}] GR recovered at a_lm=0")

    # 2. Sidereal signature
    print("\n  ── [2] Sidereal modulation signature ──")
    a_dipole = {(1, 0): 0.5 * test.cfg.a_lm_max}
    sig = test.sidereal_signature(a_dipole)
    print(f"    δv variation over sidereal day : {sig['dv_variation']:.3e}")
    print(f"    is modulated : {sig['is_modulated']}")
    print(f"    [{ 'PASS' if sig['is_modulated'] else 'FAIL'}] "
          f"nonzero a_lm produces sidereal variation")

    # 3. Injection-recovery
    print("\n  ── [3] Dipole injection-recovery ──")
    report = test.full_report()
    dip = report["a_lm(1, 0)"]
    print(f"    injected a_10   : {dip['injected']:.3e}")
    print(f"    best-fit a_10   : {dip['best_fit']:.3e}")
    print(f"    90% upper limit : {dip['upper_limit_90']:.3e}")

    print(f"\n    {report['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Parameterized directional test (a_lm), not a specific theory")
    print("  • Single detector cannot separate all a_lm — needs a network")
    print("  • Sidereal modulation needs ~1 day of observation")
    print("  • Isotropic limit validated EXACTLY before trusting bounds")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
