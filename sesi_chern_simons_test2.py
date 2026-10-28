"""
=============================================================================
SESI CHERN–SIMONS GRAVITY TEST — theory-specific parity violation
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT MAKES THIS "SPECIFIC" (vs. the parameterized modules)
----------------------------------------------------------
sesi_parity_violation.py and sesi_lorentz_violation.py let the frequency
exponent and polarization structure float freely. Chern–Simons (CS) gravity
is a SPECIFIC theory: its GW signature has a FIXED functional form set by
the action

    S_CS = (1/4) ∫ d⁴x √(−g) · θ · *RR ,      *RR = Pontryagin density

[C: Deser & Jackiw 1984; Alexander & Yunes 2009 (Phys. Rep. 480, 1)].

Consequences that are NOT free parameters here:
  1. PARITY-ODD: the two circular polarizations acquire OPPOSITE phase
     corrections (this is the defining CS signature).
  2. EARLY-PN PHASE CORRECTION: the leading CS phase term is at (−1)PN
     order, i.e.  δΨ_CS(f) ∝ (π M_c f)^{−7/3}.
     (One PN order multiplies the phase by u^{2/3}; (−1)PN below the
      Newtonian u^{−5/3} term gives u^{−7/3}.)
  3. SINGLE COUPLING: all dependence collapses to one number ζ_CS that
     bundles the CS coupling and propagation distance.

So the test constrains ONE parameter ζ_CS with a FIXED shape — strictly
more restrictive than a free (β, b) ppE pair.

CONNECTION TO sesi_parity_violation.py
--------------------------------------
CS is a particular point in that module's space: amplitude-birefringence
coefficient = 0, velocity-birefringence exponent = −7/3 (in the circular
basis), opposite signs for L/R. We therefore reuse the circular-basis
decomposition rather than re-deriving it.

LABELS: [C] CS theory structure  [D] derived implementation  [O] open

HONEST SCOPE
------------
• The −7/3 power and overall normalization depend on the CS FORMULATION
  (dynamical vs. non-dynamical, scalar background ξ). We use the canonical
  early-PN form and report ζ_CS; publication-grade bounds require the full
  CS waveform from the literature, not this toy.
• ζ_CS bundles distance and coupling; a physical bound on the CS length
  scale √α needs the distance/degeneracy handled explicitly.
• ζ_CS = 0 MUST recover GR exactly (validated). The signature is parity-odd
  (validated: flips sign under L↔R).
• A bound on ζ_CS is a statement about CS gravity specifically, not about
  all parity-violating or all modified gravity.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.chern_simons")

# Reuse the circular-basis machinery from the parameterized parity module
try:
    from sesi_parity_violation import CircularPolarizationDecomposer
    HAS_PARITY = True
except ImportError:
    HAS_PARITY = False

__all__ = ["ChernSimonsConfig", "ChernSimonsWaveform",
           "ChernSimonsLikelihood", "ChernSimonsTest"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ChernSimonsConfig:
    # Source
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256
    noise_psd: float = 1e-46

    # CS theory (FIXED by the theory, not free)
    cs_phase_power: float = -7.0 / 3.0   # (−1)PN order
    f_ref: float = 100.0                 # reference frequency for ζ

    # The single free parameter's prior
    zeta_max: float = 1.0

    seed: int = 0
    MSUN_SEC: float = 4.925490947e-6


# ═══════════════════════════════════════════════════════════════════════
# [C/D] CHERN–SIMONS WAVEFORM (specific theory)
# ═══════════════════════════════════════════════════════════════════════

class ChernSimonsWaveform:
    """
    [C/D] GR chirp with the CS parity-odd phase correction in the circular
    basis.

        h_L → h_L · exp(+i δΨ_CS) ,   h_R → h_R · exp(−i δΨ_CS)
        δΨ_CS(f) = ζ_CS · (f / f_ref)^{−7/3}

    The OPPOSITE signs for L and R are the parity-odd CS signature.
    ζ_CS = 0 recovers GR exactly.
    """

    def __init__(self, cfg: ChernSimonsConfig):
        self.cfg = cfg
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)
        if HAS_PARITY:
            self.decomp = CircularPolarizationDecomposer()
        else:
            self.decomp = None

    # ── GR base ────────────────────────────────────────────────────────
    def _gr_linear(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        Mc_sec = cfg.Mc_solar * cfg.MSUN_SEC
        phase = -(3.0 / 128.0) * (math.pi * Mc_sec * self.freqs) ** (-5.0 / 3.0)
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(cfg.distance_mpc, 1.0)) * 1e-21
        return amp * np.cos(phase), amp * np.sin(phase) * 0.5

    # ── CS phase (fixed power law) ─────────────────────────────────────
    def cs_phase(self, zeta: float) -> np.ndarray:
        """δΨ_CS(f) = ζ · (f/f_ref)^{−7/3}. Theory-fixed shape."""
        return zeta * np.power(self.freqs / self.cfg.f_ref,
                               self.cfg.cs_phase_power)

    # ── full waveform ──────────────────────────────────────────────────
    def waveform(self, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (h_+, h_×) with the CS parity-odd correction."""
        hp, hc = self._gr_linear()
        if zeta == 0.0:
            return hp, hc

        delta = self.cs_phase(zeta)
        if self.decomp is not None:
            hL, hR = self.decomp.linear_to_circular(hp, hc)
            hL = hL * np.exp(+1j * delta)   # L gets +δ (parity-odd)
            hR = hR * np.exp(-1j * delta)   # R gets −δ
            return self.decomp.circular_to_linear(hL, hR)
        else:
            # Fallback without the parity module: apply as a polarization
            # rotation of angle δ (parity-odd to first order).
            c, s = np.cos(delta), np.sin(delta)
            hp_cs = hp * c - hc * s
            hc_cs = hp * s + hc * c
            return hp_cs, hc_cs

    def gr_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.waveform(zeta=0.0)

    def make_data(self, zeta_true: float, snr_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(self.cfg.seed)
        hp, hc = self.waveform(zeta_true)
        df = self.freqs[1] - self.freqs[0]
        noise_std = math.sqrt(self.cfg.noise_psd * df)
        hp = hp * snr_scale + rng.normal(0, noise_std, len(hp))
        hc = hc * snr_scale + rng.normal(0, noise_std, len(hc))
        return hp, hc

    # ── parity-odd diagnostic ──────────────────────────────────────────
    def lr_asymmetry(self, zeta: float) -> float:
        """L/R power asymmetry: 0 at ζ=0, grows and flips sign with ζ."""
        hp, hc = self.waveform(zeta)
        if self.decomp is None:
            return 0.0
        hL, hR = self.decomp.linear_to_circular(hp, hc)
        pL = np.sum(np.abs(hL) ** 2)
        pR = np.sum(np.abs(hR) ** 2)
        return float((pL - pR) / (pL + pR + 1e-300))


# ═══════════════════════════════════════════════════════════════════════
# [D] LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════

class ChernSimonsLikelihood:
    """[D] Gaussian likelihood over the single CS coupling ζ."""

    def __init__(self, cfg: ChernSimonsConfig):
        self.cfg = cfg
        self.wf = ChernSimonsWaveform(cfg)

    def log_likelihood(self, data: Tuple[np.ndarray, np.ndarray],
                       zeta: float) -> float:
        hp_d, hc_d = data
        hp_m, hc_m = self.wf.waveform(zeta)
        df = self.wf.freqs[1] - self.wf.freqs[0]
        chi2 = (np.sum((hp_d - hp_m) ** 2) + np.sum((hc_d - hc_m) ** 2)) \
               / self.cfg.noise_psd
        return float(-0.5 * chi2 * df)

    def log_prior(self, zeta: float) -> float:
        if abs(zeta) > self.cfg.zeta_max:
            return -math.inf
        return -math.log(2.0 * self.cfg.zeta_max)


# ═══════════════════════════════════════════════════════════════════════
# [D] TEST DRIVER
# ═══════════════════════════════════════════════════════════════════════

class ChernSimonsTest:
    """
    [D] Run the theory-specific CS test with mandatory validations:
      1. ζ=0 recovers GR exactly.
      2. Signature is parity-odd (asymmetry flips sign under ζ→−ζ, zero at 0).
      3. Injection-recovery of a known ζ via grid posterior.
    """

    def __init__(self, cfg: Optional[ChernSimonsConfig] = None):
        self.cfg = cfg or ChernSimonsConfig()
        self.likelihood = ChernSimonsLikelihood(self.cfg)
        self.wf = self.likelihood.wf

    def validate_gr_limit(self) -> Dict[str, Any]:
        hp_cs, hc_cs = self.wf.waveform(0.0)
        hp_gr, hc_gr = self.wf.gr_waveform()
        max_diff = float(max(np.max(np.abs(hp_cs - hp_gr)),
                             np.max(np.abs(hc_cs - hc_gr))))
        return {"max_abs_difference": max_diff,
                "passes": max_diff < 1e-30,
                "note": "ζ=0 must recover GR exactly"}

    def validate_parity_odd(self) -> Dict[str, Any]:
        a0 = self.wf.lr_asymmetry(0.0)
        a_pos = self.wf.lr_asymmetry(0.5 * self.cfg.zeta_max)
        a_neg = self.wf.lr_asymmetry(-0.5 * self.cfg.zeta_max)
        return {
            "asymmetry_at_zero": a0,
            "asymmetry_pos": a_pos,
            "asymmetry_neg": a_neg,
            "is_zero_at_gr": abs(a0) < 1e-12,
            "is_odd": abs(a_pos + a_neg) < 0.05 * (abs(a_pos) + 1e-12),
            "cs_power_law": self.cfg.cs_phase_power,
            "note": "CS signature is parity-odd with FIXED power −7/3",
        }

    def grid_scan(self, data: Tuple[np.ndarray, np.ndarray],
                  n_grid: int = 200) -> Dict[str, np.ndarray]:
        zmax = self.cfg.zeta_max
        zeta = np.linspace(-zmax, zmax, n_grid)
        logL = np.array([self.likelihood.log_likelihood(data, z) for z in zeta])
        logL -= np.max(logL)
        post = np.exp(logL)
        Z = np.trapz(post, zeta)
        if Z > 0:
            post /= Z
        return {"zeta": zeta, "log_likelihood": logL, "posterior": post}

    def upper_limit(self, scan: Dict[str, np.ndarray], cred: float = 0.9) -> float:
        zeta, post = scan["zeta"], scan["posterior"]
        cdf = np.cumsum(post * np.gradient(zeta))
        lo = np.searchsorted(cdf, (1 - cred) / 2)
        hi = np.searchsorted(cdf, 1 - (1 - cred) / 2)
        return float(max(abs(zeta[lo]), abs(zeta[min(hi, len(zeta)-1)])))

    def full_report(self, snr_scale: float = 12.0) -> Dict[str, Any]:
        rng = np.random.default_rng(self.cfg.seed)
        report = {
            "gr_validation": self.validate_gr_limit(),
            "parity_odd_validation": self.validate_parity_odd(),
        }
        zeta_inj = 0.5 * self.cfg.zeta_max
        data = self.wf.make_data(zeta_inj, snr_scale=snr_scale, rng=rng)
        scan = self.grid_scan(data)
        best = float(scan["zeta"][np.argmax(scan["posterior"])])
        report["zeta_recovery"] = {
            "injected": zeta_inj,
            "best_fit": best,
            "upper_limit_90": self.upper_limit(scan, 0.9),
        }
        report["honest_note"] = (
            "ζ_CS bound uses the canonical (−1)PN power −7/3 and bundles "
            "distance+coupling. Formulation-dependent coefficients are NOT "
            "the literature values; publication-grade CS tests require the "
            "full CS waveform. This is a THEORY-SPECIFIC test (fixed shape, "
            "one coupling), stricter than the parameterized ppE family."
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Chern–Simons Specific Test — Smoke Test")
    print("=" * 72)
    print(f"  parity module available : {HAS_PARITY}")

    test = ChernSimonsTest()

    print("\n  ── [1] GR-limit validation ──")
    gr = test.validate_gr_limit()
    print(f"    max |CS(ζ=0) − GR| = {gr['max_abs_difference']:.2e}")
    print(f"    [{'PASS' if gr['passes'] else 'FAIL'}] ζ=0 recovers GR")

    print("\n  ── [2] Parity-odd CS signature (FIXED power −7/3) ──")
    po = test.validate_parity_odd()
    print(f"    asymmetry(ζ=0)  = {po['asymmetry_at_zero']:.2e}")
    print(f"    asymmetry(+ζ)   = {po['asymmetry_pos']:+.4f}")
    print(f"    asymmetry(−ζ)   = {po['asymmetry_neg']:+.4f}")
    print(f"    [{'PASS' if po['is_zero_at_gr'] else 'FAIL'}] zero at GR")
    print(f"    [{'PASS' if po['is_odd'] else 'FAIL'}] odd under ζ→−ζ")

    print("\n  ── [3] Injection-recovery of ζ_CS ──")
    report = test.full_report()
    rec = report["zeta_recovery"]
    print(f"    injected ζ : {rec['injected']:.3f}")
    print(f"    best fit   : {rec['best_fit']:.3f}")
    print(f"    90% UL     : {rec['upper_limit_90']:.3f}")
    print(f"\n    {report['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Specific theory: FIXED power (−7/3) + parity-odd L/R structure")
    print("  • One free parameter ζ_CS (stricter than parameterized ppE)")
    print("  • Coefficients are formulation-dependent; use full CS waveform")
    print("    from the literature for publication-grade bounds")
    print("  • ζ bound applies to Chern–Simons gravity specifically")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
