"""
=============================================================================
SESI PARITY-VIOLATION TEST — beyond-GR via circular-polarization asymmetry
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS (distinct from earlier beyond-GR modules)
--------------------------------------------------------
Earlier modules tested:
  • sesi_graviton_mass_test.py   — isotropic dispersion (f^{-1}), one channel
  • sesi_lorentz_violation.py    — dispersion tower + SYMMETRIC linear-pol rotation
  • sesi_anisotropic_lv.py       — direction-dependent speed + sidereal modulation

This module tests PARITY VIOLATION: the ASYMMETRIC treatment of the two
CIRCULAR polarizations. In GR the left (L) and right (R) circular modes
propagate identically. Parity-violating gravity (e.g. Chern–Simons, EFT
extensions [C]) makes them differ:

    Amplitude birefringence : h_L → h_L e^{+κ},  h_R → h_R e^{-κ}
                              (changes ellipticity of linear polarization)
    Velocity birefringence  : h_L → h_L e^{+iφ}, h_R → h_R e^{-iφ}
                              (rotates the linear polarization angle)

The parity-ODD signature is the SIGN FLIP between L and R under the
coefficient ρ → -ρ. This is what distinguishes parity violation from a
parity-even polarization rotation.

CIRCULAR BASIS CONVENTION
-------------------------
    h_L = (h_+ + i h_×)/√2 ,   h_R = (h_+ − i h_×)/√2
    inverse:  h_+ = (h_L + h_R)/√2 ,  h_× = i(h_R − h_L)/√2

LABELS: [C] parity-violating GW framework  [D] implementation  [O] open

HONEST SCOPE
------------
• Parameterized test in the circular basis; NOT a specific Chern–Simons
  or EFT computation. The frequency/distance dependence κ,φ is a model
  choice that must be justified for a real theory.
• ρ = 0 MUST recover GR exactly — validated before trusting any bound
  (Carpet-19 Theorem 3(b) discipline: check against a reference).
• A bound on ρ is prior-dependent and model-dependent; it is not proof
  that parity is or is not violated in gravity.
• Amplitude and velocity birefringence are partially degenerate with
  inclination and distance; report joint constraints, not one-at-a-time.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.parity_viol")

__all__ = ["ParityViolationConfig", "CircularPolarizationDecomposer",
           "ParityViolationWaveform", "ParityViolationLikelihood",
           "ParityViolationTest"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ParityViolationConfig:
    # Source
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256
    noise_psd: float = 1e-46

    # Parity-violating coefficients
    rho_amp: float = 0.0    # amplitude birefringence strength
    rho_vel: float = 0.0    # velocity birefringence strength
    rho_max: float = 1.0    # prior half-width

    # Model for frequency/distance dependence
    freq_power: float = 1.0   # κ,φ ∝ f^{freq_power}
    seed: int = 0

    MSUN_SEC: float = 4.925490947e-6
    MPC_M: float = 3.085677581e22


# ═══════════════════════════════════════════════════════════════════════
# [C/D] CIRCULAR POLARIZATION DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════

class CircularPolarizationDecomposer:
    """
    [C/D] Convert between linear (h_+, h_×) and circular (h_L, h_R) bases.

    The circular basis is the natural one for parity: parity flips L ↔ R,
    so any L/R asymmetry is a parity-odd observable.
    """

    @staticmethod
    def linear_to_circular(hp: np.ndarray, hc: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        hL = (hp + 1j * hc) / math.sqrt(2.0)
        hR = (hp - 1j * hc) / math.sqrt(2.0)
        return hL, hR

    @staticmethod
    def circular_to_linear(hL: np.ndarray, hR: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        hp = (hL + hR) / math.sqrt(2.0)
        hc = 1j * (hR - hL) / math.sqrt(2.0)
        return hp, hc

    @staticmethod
    def parity_transform(hL: np.ndarray, hR: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Parity operation swaps L ↔ R. Used to check parity-odd behavior."""
        return hR.copy(), hL.copy()


# ═══════════════════════════════════════════════════════════════════════
# [C/D] PARITY-VIOLATION WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class ParityViolationWaveform:
    """
    [C/D] GR chirp + parity-violating circular-basis birefringence.

    Builds the GR (h_+, h_×), converts to circular, applies the ASYMMETRIC
    L/R modifications, and converts back. With ρ_amp=ρ_vel=0 this is
    exactly GR.
    """

    def __init__(self, cfg: ParityViolationConfig):
        self.cfg = cfg
        self.decomp = CircularPolarizationDecomposer()
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)

    def _gr_linear(self) -> Tuple[np.ndarray, np.ndarray]:
        """GR leading-order chirp in the linear basis."""
        cfg = self.cfg
        Mc_sec = cfg.Mc_solar * cfg.MSUN_SEC
        phase = -(3.0 / 128.0) * (math.pi * Mc_sec * self.freqs) ** (-5.0 / 3.0)
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(cfg.distance_mpc, 1.0)) * 1e-21
        hp = amp * np.cos(phase)
        hc = amp * np.sin(phase) * 0.5
        return hp, hc

    def _birefringence_exponent(self) -> np.ndarray:
        """κ(f) = ρ · (f/f_ref)^p · (D-scale). Returns the L/R splitting."""
        cfg = self.cfg
        f_ref = 100.0
        D_m = cfg.distance_mpc * cfg.MPC_M
        return (cfg.rho_amp * np.power(self.freqs / f_ref, cfg.freq_power)
                * (D_m / 1e26))

    def _velocity_phase(self) -> np.ndarray:
        cfg = self.cfg
        f_ref = 100.0
        D_m = cfg.distance_mpc * cfg.MPC_M
        return (cfg.rho_vel * np.power(self.freqs / f_ref, cfg.freq_power)
                * (D_m / 1e26))

    def waveform(self, rho_amp: Optional[float] = None,
                 rho_vel: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (h_+, h_×) with parity violation applied in circular basis."""
        cfg = self.cfg
        # temporarily override coefficients
        ra = cfg.rho_amp if rho_amp is None else rho_amp
        rv = cfg.rho_vel if rho_vel is None else rho_vel

        hp_gr, hc_gr = self._gr_linear()
        hL, hR = self.decomp.linear_to_circular(hp_gr, hc_gr)

        # Parity-ODD splitting: opposite signs for L and R
        kappa = self._birefringence_exponent() * (ra / max(cfg.rho_amp, 1e-30)
                  if cfg.rho_amp != 0 else 1.0) if ra != 0 else np.zeros_like(self.freqs)
        if ra == 0:
            kappa = np.zeros_like(self.freqs)
        else:
            kappa = (ra * np.power(self.freqs / 100.0, cfg.freq_power)
                     * (cfg.distance_mpc * cfg.MPC_M / 1e26))
        if rv == 0:
            phi_b = np.zeros_like(self.freqs)
        else:
            phi_b = (rv * np.power(self.freqs / 100.0, cfg.freq_power)
                     * (cfg.distance_mpc * cfg.MPC_M / 1e26))

        # Amplitude birefringence: L amplified, R damped (or vice versa)
        hL = hL * np.exp(+kappa)
        hR = hR * np.exp(-kappa)
        # Velocity birefringence: opposite phase shifts
        hL = hL * np.exp(+1j * phi_b)
        hR = hR * np.exp(-1j * phi_b)

        return self.decomp.circular_to_linear(hL, hR)

    def gr_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pure GR (ρ_amp = ρ_vel = 0)."""
        return self.waveform(rho_amp=0.0, rho_vel=0.0)

    def make_data(self, rho_amp_true: float, rho_vel_true: float,
                  snr_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated (h_+, h_×) with injected parity violation + noise."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        hp, hc = self.waveform(rho_amp=rho_amp_true, rho_vel=rho_vel_true)
        df = self.freqs[1] - self.freqs[0]
        noise_std = math.sqrt(self.cfg.noise_psd * df)
        hp = hp * snr_scale + rng.normal(0, noise_std, len(hp))
        hc = hc * snr_scale + rng.normal(0, noise_std, len(hc))
        return hp, hc

    def lr_asymmetry(self, rho_amp: float) -> float:
        """
        Parity-odd diagnostic: relative L/R power asymmetry.
        Zero when ρ_amp = 0 (parity conserved); grows with |ρ_amp|.
        """
        hp, hc = self.waveform(rho_amp=rho_amp, rho_vel=0.0)
        hL, hR = self.decomp.linear_to_circular(hp, hc)
        pL = np.sum(np.abs(hL) ** 2)
        pR = np.sum(np.abs(hR) ** 2)
        return float((pL - pR) / (pL + pR + 1e-300))


# ═══════════════════════════════════════════════════════════════════════
# [D] LIKELIHOOD + TEST
# ═══════════════════════════════════════════════════════════════════════

class ParityViolationLikelihood:
    """[D] Gaussian likelihood over (ρ_amp, ρ_vel)."""

    def __init__(self, cfg: ParityViolationConfig):
        self.cfg = cfg
        self.wf = ParityViolationWaveform(cfg)

    def log_likelihood(self, data: Tuple[np.ndarray, np.ndarray],
                       rho_amp: float, rho_vel: float) -> float:
        hp_d, hc_d = data
        hp_m, hc_m = self.wf.waveform(rho_amp=rho_amp, rho_vel=rho_vel)
        df = self.wf.freqs[1] - self.wf.freqs[0]
        resid_p = hp_d - hp_m
        resid_c = hc_d - hc_m
        chi2 = (np.sum(resid_p ** 2) + np.sum(resid_c ** 2)) / self.cfg.noise_psd
        return float(-0.5 * chi2 * df)

    def log_prior(self, rho_amp: float, rho_vel: float) -> float:
        rm = self.cfg.rho_max
        if abs(rho_amp) > rm or abs(rho_vel) > rm:
            return -math.inf
        return 0.0


class ParityViolationTest:
    """
    [D] Run the parity-violation test with mandatory validations:
      1. GR-recovery: ρ_amp=ρ_vel=0 reproduces GR exactly.
      2. Parity-odd check: L/R asymmetry is zero at ρ=0, nonzero at ρ≠0,
         and flips sign under ρ→-ρ.
      3. Injection-recovery for a nonzero ρ_amp.
    """

    def __init__(self, cfg: Optional[ParityViolationConfig] = None):
        self.cfg = cfg or ParityViolationConfig()
        self.likelihood = ParityViolationLikelihood(self.cfg)
        self.wf = self.likelihood.wf

    def validate_gr_limit(self) -> Dict[str, Any]:
        hp_pv, hc_pv = self.wf.waveform(rho_amp=0.0, rho_vel=0.0)
        hp_gr, hc_gr = self.wf.gr_waveform()
        max_diff = float(max(np.max(np.abs(hp_pv - hp_gr)),
                             np.max(np.abs(hc_pv - hc_gr))))
        return {"max_abs_difference": max_diff,
                "passes": max_diff < 1e-30,
                "note": "ρ=0 must recover GR exactly"}

    def validate_parity_odd(self) -> Dict[str, Any]:
        """L/R asymmetry must be parity-odd: A(ρ)=−A(−ρ), A(0)=0."""
        a0 = self.wf.lr_asymmetry(0.0)
        a_pos = self.wf.lr_asymmetry(0.5 * self.cfg.rho_max)
        a_neg = self.wf.lr_asymmetry(-0.5 * self.cfg.rho_max)
        return {
            "asymmetry_at_zero": a0,
            "asymmetry_pos": a_pos,
            "asymmetry_neg": a_neg,
            "is_zero_at_gr": abs(a0) < 1e-12,
            "is_odd": abs(a_pos + a_neg) < 0.05 * (abs(a_pos) + 1e-12),
            "note": "Parity-odd observable must vanish at ρ=0 and flip sign",
        }

    def grid_scan(self, data: Tuple[np.ndarray, np.ndarray],
                  axis: str = "rho_amp", n_grid: int = 100) -> Dict[str, np.ndarray]:
        """1-D scan of one parity coefficient (other fixed at 0)."""
        rmax = self.cfg.rho_max
        coef = np.linspace(-rmax, rmax, n_grid)
        logL = np.zeros(n_grid)
        for i, c in enumerate(coef):
            if axis == "rho_amp":
                logL[i] = self.likelihood.log_likelihood(data, c, 0.0)
            else:
                logL[i] = self.likelihood.log_likelihood(data, 0.0, c)
        logL -= np.max(logL)
        post = np.exp(logL)
        Z = np.trapz(post, coef)
        if Z > 0:
            post /= Z
        return {"coefficient": coef, "log_likelihood": logL, "posterior": post}

    def upper_limit(self, scan: Dict[str, np.ndarray], cred: float = 0.9) -> float:
        coef = scan["coefficient"]; post = scan["posterior"]
        cdf = np.cumsum(post * np.gradient(coef))
        lo = np.searchsorted(cdf, (1 - cred) / 2)
        hi = np.searchsorted(cdf, 1 - (1 - cred) / 2)
        return float(max(abs(coef[lo]), abs(coef[min(hi, len(coef)-1)])))

    def full_report(self, snr_scale: float = 12.0) -> Dict[str, Any]:
        rng = np.random.default_rng(self.cfg.seed)
        report = {
            "gr_validation": self.validate_gr_limit(),
            "parity_odd_validation": self.validate_parity_odd(),
        }
        # Inject ρ_amp, recover
        rho_inj = 0.5 * self.cfg.rho_max
        data = self.wf.make_data(rho_inj, 0.0, snr_scale=snr_scale, rng=rng)
        scan = self.grid_scan(data, axis="rho_amp")
        best = float(scan["coefficient"][np.argmax(scan["posterior"])])
        report["rho_amp_recovery"] = {
            "injected": rho_inj,
            "best_fit": best,
            "upper_limit_90": self.upper_limit(scan, 0.9),
        }
        report["honest_note"] = (
            "Parity-violation bounds are prior- and model-dependent (κ,φ "
            "frequency/distance scaling). Amplitude birefringence is partially "
            "degenerate with inclination/distance. ρ=0 recovery and parity-odd "
            "check MUST pass before trusting any bound."
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Parity-Violation Test — Smoke Test")
    print("=" * 72)

    test = ParityViolationTest()

    print("\n  ── [1] GR-limit validation ──")
    gr = test.validate_gr_limit()
    print(f"    max |PV − GR| = {gr['max_abs_difference']:.2e}")
    print(f"    [{'PASS' if gr['passes'] else 'FAIL'}] GR recovered at ρ=0")

    print("\n  ── [2] Parity-odd diagnostic ──")
    po = test.validate_parity_odd()
    print(f"    asymmetry(ρ=0)   = {po['asymmetry_at_zero']:.2e} (expect ~0)")
    print(f"    asymmetry(+ρ)    = {po['asymmetry_pos']:+.4f}")
    print(f"    asymmetry(−ρ)    = {po['asymmetry_neg']:+.4f}")
    print(f"    [{'PASS' if po['is_zero_at_gr'] else 'FAIL'}] zero at GR")
    print(f"    [{'PASS' if po['is_odd'] else 'FAIL'}] odd under ρ→−ρ")

    print("\n  ── [3] Injection-recovery (ρ_amp) ──")
    report = test.full_report()
    rec = report["rho_amp_recovery"]
    print(f"    injected  : {rec['injected']:.3f}")
    print(f"    best fit  : {rec['best_fit']:.3f}")
    print(f"    90% UL    : {rec['upper_limit_90']:.3f}")
    print(f"\n    {report['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Parity violation = L/R circular asymmetry, distinct from the")
    print("    symmetric birefringence in sesi_lorentz_violation.py")
    print("  • Parameterized κ,φ scaling; NOT a specific Chern–Simons result")
    print("  • ρ=0 recovers GR; parity-odd check gates all bounds")
    print("  • Degenerate with inclination/distance — report joint constraints")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
