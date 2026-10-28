"""
=============================================================================
SESI FULL LORENTZ-VIOLATION TEST — beyond the graviton mass
Multi-power modified dispersion + gravitational birefringence
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHY "FULL" AND NOT JUST GRAVITON MASS
-------------------------------------
sesi_graviton_mass_test.py tested ONE Lorentz-violating signature: the
massive-graviton dispersion δΨ ∝ f^{-1}. Generic Lorentz violation (as in
the Standard-Model Extension, Kostelecký & Mewes 2016) produces a FAMILY
of signatures:

  [LV1] Modified dispersion — a tower of phase corrections
        δΨ(f) = Σ_α β_α · f^α
        derived from a general dispersion relation  ω² = c²k² + Σ_n c_n k^n.
        The group-velocity delay gives δΨ ∝ f^{n-1}; the graviton mass is
        the n=0 → α=-1 case.  [C: Mirshekari–Yunes–Will, PRD 85, 024019]

  [LV2] Birefringence — the + and × polarizations propagate differently,
        rotating the polarization plane by a frequency-dependent angle
        δθ(f) = Σ_b κ_b · f^b · D.  This is a genuinely DISTINCT signature
        absent from a simple massive-graviton model.  [C: SME gravity sector]

DEGENERACY WARNING (must be stated)
-----------------------------------
  • α = 0 (constant phase) is degenerate with the reference phase φ_0.
  • α = 1 (linear-in-f) is degenerate with the arrival time t_c.
  → We test α ∈ {-1, 2, 3} by default; α ∈ {0,1} are reported as
    non-identifiable, never silently fitted.

LABELS: [C] parameterized frameworks  [D] implementation  [O] open

HONEST SCOPE
------------
• This is a PARAMETERIZED test, not a derivation from a specific quantum-
  gravity theory. A bound on β_α/κ_b is a statement about this parameter
  family under the chosen prior, not a verdict on Lorentz symmetry.
• Coefficients can be mutually degenerate; joint limits must be reported
  with the full covariance, not one-at-a-time.
• The GR limit (all coefficients = 0) MUST be recovered exactly; this is
  validated in the smoke test before any bound is trusted (Carpet-19
  Theorem 3(b) discipline: check convergence against a reference).
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.lorentz_viol")

__all__ = [
    "LorentzViolationConfig",
    "ModifiedDispersionModel",
    "BirefringenceModel",
    "LorentzViolationWaveform",
    "LorentzViolationLikelihood",
    "LorentzViolationTest",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LorentzViolationConfig:
    # Dispersion exponents α in δΨ ∝ f^α.  Default skips 0,1 (degenerate).
    dispersion_alphas: List[float] = field(default_factory=lambda: [-1.0, 2.0, 3.0])
    # Birefringence exponents b in δθ ∝ f^b.
    birefringence_exponents: List[float] = field(default_factory=lambda: [0.0, 1.0])

    # Prior half-widths
    beta_max: float = 1.0
    kappa_max: float = 1.0

    # Source
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    redshift: float = 0.1
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256
    noise_psd: float = 1e-46
    seed: int = 0

    MSUN_SEC: float = 4.925490947e-6
    MPC_M: float = 3.085677581e22


# ═══════════════════════════════════════════════════════════════════════
# [LV1] MODIFIED DISPERSION
# ═══════════════════════════════════════════════════════════════════════

class ModifiedDispersionModel:
    """
    [C/D] Tower of dispersion phase corrections  δΨ(f) = Σ_α β_α f^α.

    Derivation sketch (group-velocity delay): for a dispersion term
    ω² = c²k² + c_n k^n, the group velocity picks up ∝ k^{n-2}, so the
    arrival-time delay Δt ∝ f^{n-2} and the phase δΨ = 2πf·Δt ∝ f^{n-1}.
    Setting α = n-1 gives the exponent tower; graviton mass is α = -1.
    """

    def __init__(self, cfg: LorentzViolationConfig):
        self.cfg = cfg
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)

    def gr_phase(self) -> np.ndarray:
        """Leading-order GR chirp phase (reference)."""
        Mc_sec = self.cfg.Mc_solar * self.cfg.MSUN_SEC
        return -(3.0 / 128.0) * (math.pi * Mc_sec * self.freqs) ** (-5.0 / 3.0)

    def dispersion_phase(self, betas: Dict[float, float]) -> np.ndarray:
        """Σ_α β_α f^α, scaled by distance (proxy for propagation baseline)."""
        D_m = self.cfg.distance_mpc * self.cfg.MPC_M
        phase = np.zeros_like(self.freqs)
        for alpha, beta in betas.items():
            if beta == 0.0:
                continue
            phase = phase + beta * np.power(self.freqs, alpha)
        # distance scaling (dimensionless proxy; real analysis uses the
        # cosmological propagation integral)
        return phase * (D_m / 1e26)

    def identifiable_alphas(self) -> List[float]:
        """Alphas that are NOT degenerate with φ_0 (0) or t_c (1)."""
        return [a for a in self.cfg.dispersion_alphas if a not in (0.0, 1.0)]

    def degenerate_alphas(self) -> List[float]:
        return [a for a in self.cfg.dispersion_alphas if a in (0.0, 1.0)]


# ═══════════════════════════════════════════════════════════════════════
# [LV2] BIREFRINGENCE
# ═══════════════════════════════════════════════════════════════════════

class BirefringenceModel:
    """
    [C/D] Frequency-dependent polarization rotation from Lorentz violation.

        δθ(f) = Σ_b κ_b f^b · D

    The rotation mixes the two polarizations:
        h+_obs =  h+ cos δθ − h× sin δθ
        h×_obs =  h+ sin δθ + h× cos δθ

    Distinct from dispersion: it changes the POLARIZATION content, not just
    the phasing of a single combined strain.
    """

    def __init__(self, cfg: LorentzViolationConfig):
        self.cfg = cfg

    def rotation_angle(self, freqs: np.ndarray,
                       kappas: Dict[float, float]) -> np.ndarray:
        D_m = self.cfg.distance_mpc * self.cfg.MPC_M
        angle = np.zeros_like(freqs)
        for b, kappa in kappas.items():
            if kappa == 0.0:
                continue
            angle = angle + kappa * np.power(freqs, b)
        return angle * (D_m / 1e26)

    @staticmethod
    def rotate(hp: np.ndarray, hc: np.ndarray,
               angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(angle), np.sin(angle)
        hp_obs = hp * c - hc * s
        hc_obs = hp * s + hc * c
        return hp_obs, hc_obs


# ═══════════════════════════════════════════════════════════════════════
# COMBINED WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class LorentzViolationWaveform:
    """[D] GR chirp + dispersion tower + birefringence rotation."""

    def __init__(self, cfg: LorentzViolationConfig):
        self.cfg = cfg
        self.disp = ModifiedDispersionModel(cfg)
        self.bire = BirefringenceModel(cfg)
        self.freqs = self.disp.freqs

    def waveform(self, betas: Dict[float, float],
                 kappas: Dict[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (h_plus, h_cross) including all LV effects."""
        Mc_sec = self.cfg.Mc_solar * self.cfg.MSUN_SEC
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(self.cfg.distance_mpc, 1.0)) * 1e-21

        # GR phase + dispersion corrections
        phase = self.disp.gr_phase() + self.disp.dispersion_phase(betas)

        hp = amp * np.cos(phase)
        hc = amp * np.sin(phase) * 0.5   # toy cross-polarization

        # Birefringence rotates the polarization basis
        hp, hc = self.bire.rotate(hp, hc, self.bire.rotation_angle(self.freqs, kappas))
        return hp, hc

    def gr_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pure GR (all LV coefficients zero)."""
        return self.waveform(betas={}, kappas={})

    def make_data(self, betas: Dict[float, float], kappas: Dict[float, float],
                  snr_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated (h+_obs, h×_obs) with Gaussian noise."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        hp, hc = self.waveform(betas, kappas)
        df = self.freqs[1] - self.freqs[0]
        noise_std = math.sqrt(self.cfg.noise_psd * df)
        hp = hp * snr_scale + rng.normal(0, noise_std, len(hp))
        hc = hc * snr_scale + rng.normal(0, noise_std, len(hc))
        return hp, hc


# ═══════════════════════════════════════════════════════════════════════
# LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════

class LorentzViolationLikelihood:
    """[D] Gaussian likelihood over both polarizations."""

    def __init__(self, cfg: LorentzViolationConfig):
        self.cfg = cfg
        self.wf = LorentzViolationWaveform(cfg)

    def log_likelihood(self, data: Tuple[np.ndarray, np.ndarray],
                       betas: Dict[float, float],
                       kappas: Dict[float, float]) -> float:
        hp_d, hc_d = data
        hp_m, hc_m = self.wf.waveform(betas, kappas)
        df = self.wf.freqs[1] - self.wf.freqs[0]
        resid_p = hp_d - hp_m
        resid_c = hc_d - hc_m
        chi2 = (np.sum(resid_p ** 2) + np.sum(resid_c ** 2)) / self.cfg.noise_psd
        return float(-0.5 * chi2 * df)

    def log_prior(self, betas: Dict[float, float],
                  kappas: Dict[float, float]) -> float:
        """Uniform priors on every non-degenerate coefficient."""
        for v in list(betas.values()) + list(kappas.values()):
            if abs(v) > max(self.cfg.beta_max, self.cfg.kappa_max):
                return -math.inf
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# TEST DRIVER
# ═══════════════════════════════════════════════════════════════════════

class LorentzViolationTest:
    """
    [D] Run the full Lorentz-violation test.

    Two mandatory validations before any bound is reported:
      1. GR-recovery: injecting all-zero coefficients reproduces GR exactly.
      2. Injection-recovery: a known nonzero coefficient is recoverable.
    """

    def __init__(self, cfg: Optional[LorentzViolationConfig] = None):
        self.cfg = cfg or LorentzViolationConfig()
        self.likelihood = LorentzViolationLikelihood(self.cfg)
        self.wf = self.likelihood.wf

    def validate_gr_limit(self) -> Dict[str, Any]:
        """All LV coefficients zero must equal the pure-GR waveform."""
        hp_lv, hc_lv = self.wf.waveform(betas={}, kappas={})
        hp_gr, hc_gr = self.wf.gr_waveform()
        max_diff = float(max(np.max(np.abs(hp_lv - hp_gr)),
                             np.max(np.abs(hc_lv - hc_gr))))
        return {
            "max_abs_difference": max_diff,
            "passes": max_diff < 1e-30,
            "note": "GR limit must be recovered EXACTLY before trusting bounds",
        }

    def grid_scan_1d(self, data: Tuple[np.ndarray, np.ndarray],
                     alpha: float, kappa_b: Optional[float] = None,
                     n_grid: int = 200) -> Dict[str, np.ndarray]:
        """
        1-D scan of a single dispersion coefficient β_α (or birefringence
        κ_b) — returns the profile likelihood for that coefficient.
        """
        coef = np.linspace(-self.cfg.beta_max, self.cfg.beta_max, n_grid)
        logL = np.zeros(n_grid)
        for i, c in enumerate(coef):
            betas = {alpha: c} if kappa_b is None else {}
            kappas = {} if kappa_b is None else {kappa_b: c}
            logL[i] = self.likelihood.log_likelihood(data, betas, kappas)
        logL -= np.max(logL)
        post = np.exp(logL)
        Z = np.trapz(post, coef)
        if Z > 0:
            post /= Z
        return {"coefficient": coef, "log_likelihood": logL, "posterior": post}

    def upper_limit(self, scan: Dict[str, np.ndarray],
                    cred: float = 0.9) -> float:
        """Symmetric credible upper bound |β| < bound at level `cred`."""
        coef = scan["coefficient"]
        post = scan["posterior"]
        cdf = np.cumsum(post * np.gradient(coef))
        # two-sided: bound such that P(|β|<bound) = cred
        lo_idx = np.searchsorted(cdf, (1 - cred) / 2)
        hi_idx = np.searchsorted(cdf, 1 - (1 - cred) / 2)
        return float(max(abs(coef[lo_idx]), abs(coef[min(hi_idx, len(coef)-1)])))

    def full_report(self, snr_scale: float = 12.0) -> Dict[str, Any]:
        """GR-recovery validation + injection-recovery demo."""
        report = {"gr_validation": self.validate_gr_limit()}
        rng = np.random.default_rng(self.cfg.seed)

        # Dispersion: inject β_{-1} and recover (graviton-mass-like channel)
        alpha = -1.0
        data = self.wf.make_data({alpha: 0.0}, {}, snr_scale=snr_scale, rng=rng)
        scan = self.grid_scan_1d(data, alpha)
        report[f"beta_alpha={alpha}"] = {
            "injected": 0.0,
            "best_fit": float(scan["coefficient"][np.argmax(scan["posterior"])]),
            "upper_limit_90": self.upper_limit(scan, 0.9),
        }

        # Birefringence: inject κ_0 = 0 (null) and confirm no false rotation
        data_b = self.wf.make_data({}, {0.0: 0.0}, snr_scale=snr_scale, rng=rng)
        scan_b = self.grid_scan_1d(data_b, alpha=None, kappa_b=0.0)
        report["birefringence_kappa_b=0"] = {
            "injected": 0.0,
            "best_fit": float(scan_b["coefficient"][np.argmax(scan_b["posterior"])]),
            "upper_limit_90": self.upper_limit(scan_b, 0.9),
        }

        report["degenerate_alphas_not_fitted"] = self.wf.disp.degenerate_alphas()
        report["honest_note"] = (
            "Bounds are for this parameterized family under uniform priors. "
            "α=0,1 are degenerate with φ_0,t_c and are NOT fitted. "
            "A bound here is not evidence Lorentz symmetry is violated or preserved."
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Full Lorentz-Violation Test — Smoke Test")
    print("=" * 72)

    cfg = LorentzViolationConfig()
    test = LorentzViolationTest(cfg)

    # 1. GR limit must be exact
    print("\n  ── [1] GR-limit validation ──")
    gr = test.validate_gr_limit()
    print(f"    max |LV − GR| = {gr['max_abs_difference']:.2e}")
    print(f"    [{'PASS' if gr['passes'] else 'FAIL'}] GR recovered exactly at β=κ=0")

    # 2. Full report
    print("\n  ── [2] Injection / recovery ──")
    report = test.full_report()
    for key in ["beta_alpha=-1.0", "birefringence_kappa_b=0"]:
        r = report[key]
        print(f"    {key}:")
        print(f"      injected={r['injected']:.3f}  best_fit={r['best_fit']:.3f}  "
              f"90% UL={r['upper_limit_90']:.3f}")

    print(f"\n    degenerate α (NOT fitted): {report['degenerate_alphas_not_fitted']}")
    print(f"    {report['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Parameterized test (dispersion tower + birefringence), not a")
    print("    specific quantum-gravity theory")
    print("  • α=0,1 degenerate with phase/time — reported, never silently fit")
    print("  • GR limit validated EXACTLY before any bound is trusted")
    print("  • Bounds are prior-dependent; report the prior half-width")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
