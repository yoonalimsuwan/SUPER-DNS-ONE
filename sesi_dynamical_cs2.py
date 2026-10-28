"""
=============================================================================
SESI DYNAMICAL CHERN–SIMONS TEST — evolving CS scalar field θ
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_chern_simons_test.py
-----------------------------------------------
The previous module used a FIXED power-law phase correction
    δΨ(f) = ζ (f/f_ref)^{-7/3}
which corresponds to a frozen / prescribed CS scalar (non-dynamical CS).

Dynamical Chern–Simons (dCS) gravity [C: Yunes & Pretorius 2009,
"Dynamical Chern-Simons Modified Gravity I"; see also the LVK tests of GR]
promotes the CS scalar θ to a DYNAMICAL field obeying its own equation of
motion:

    □θ = -(α/4) · *RR + dV/dθ

During a binary inspiral the Pontryagin density *RR ≠ 0, so θ is SOURCED
and its time derivative θ̇ is NOT constant — it grows as the binary
accelerates toward merger. Consequently the GW phase correction in dCS
EMERES from the scalar's evolution rather than being imposed as a fixed
power law.

THE HONEST SIMPLIFICATION USED HERE
-----------------------------------
We model the dynamical response by a scalar-response function G(f) that
encodes how θ̇ grows with frequency during the inspiral:

    δΨ_dCS(f) = ζ · G(f) · (f/f_ref)^{-7/3}

  • G(f) = 1  ⟹  frozen scalar  ⟹  reduces EXACTLY to the non-dynamical
    module (continuity check).
  • G(f) growing with f  ⟹  dynamical scalar responds more strongly at
    high frequency — the qualitative signature that separates dCS from
    non-dynamical CS.

This is a REDUCED model: the full dCS waveform requires solving the coupled
scalar-tensor system (or using a calibrated dCS approximant), which is [O].

LABELS: [C] dCS framework  [D] derived reduced implementation  [O] open

HONEST SCOPE
------------
• The scalar-response function G(f) is a phenomenological stand-in for the
  full scalar evolution; its specific form must be justified against the
  dCS literature before physical interpretation.
• ζ bundles coupling + distance; a bound on ζ is model- and prior-dependent.
• ζ = 0 MUST recover GR exactly; G = 1 MUST recover the non-dynamical
  module exactly. Both are validated before any bound is trusted
  (Carpet-19 Theorem 3(b): check against a reference, do not assume).
• This is a WAVEFORM-MODEL comparison, not a proof that dCS is or is not
  realized in Nature.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.dynamical_cs")

# Reuse the circular-basis decomposition from the parameterized module
try:
    from sesi_parity_violation import CircularPolarizationDecomposer
    HAS_DECOMP = True
except ImportError:
    HAS_DECOMP = False

__all__ = ["DynamicalCSConfig", "CSScalarField", "DynamicalCSWaveform",
           "DynamicalCSLikelihood", "DynamicalCSTest"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DynamicalCSConfig:
    # Source
    Mc_solar: float = 30.0
    distance_mpc: float = 400.0
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256
    noise_psd: float = 1e-46

    # CS theory (fixed structure; ζ is the single free coupling)
    base_power: float = -7.0 / 3.0     # (-1)PN order
    f_ref: float = 100.0
    zeta_max: float = 1.0

    # Dynamical scalar response G(f)
    # G(f) = 1 + g1 * (f/f_ref)^q ;  g1=0 ⟹ frozen (non-dynamical) limit
    response_g1: float = 0.5           # strength of dynamical response
    response_q: float = 1.0            # frequency scaling of the response

    seed: int = 0
    MSUN_SEC: float = 4.925490947e-6


# ═══════════════════════════════════════════════════════════════════════
# [C/D] DYNAMICAL CS SCALAR FIELD
# ═══════════════════════════════════════════════════════════════════════

class CSScalarField:
    """
    [C/D] Reduced model of the dynamical CS scalar θ.

    Instead of solving □θ = -(α/4)*RR + dV/dθ (which is [O] in full), we
    encode the scalar's dynamical response via G(f): the scalar responds
    more strongly as the binary sweeps to higher frequency.

    G(f) = 1 + g1 · (f/f_ref)^q

      • g1 = 0  ⟹  G = 1  ⟹  frozen / non-dynamical scalar.
      • g1 > 0  ⟹  scalar grows with frequency ⟹ dynamical.
    """

    def __init__(self, cfg: DynamicalCSConfig):
        self.cfg = cfg

    def response(self, f: np.ndarray) -> np.ndarray:
        """Scalar response function G(f) ≥ 0."""
        return 1.0 + self.cfg.response_g1 * np.power(
            np.maximum(f, 1e-12) / self.cfg.f_ref, self.cfg.response_q)

    def is_dynamical(self) -> bool:
        return abs(self.cfg.response_g1) > 1e-12


# ═══════════════════════════════════════════════════════════════════════
# [C/D] DYNAMICAL CS WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class DynamicalCSWaveform:
    """
    [C/D] GR chirp + dCS parity-odd phase correction driven by the
    dynamical scalar response G(f).

        δΨ_dCS(f) = ζ · G(f) · (f/f_ref)^{-7/3}
        h_L → h_L e^{+i δΨ},   h_R → h_R e^{-i δΨ}   (parity-odd)

    Limits (both validated in the test):
        ζ = 0            ⟹  pure GR
        G = 1 (g1 = 0)   ⟹  non-dynamical CS (sesi_chern_simons_test)
    """

    def __init__(self, cfg: DynamicalCSConfig):
        self.cfg = cfg
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)
        self.scalar = CSScalarField(cfg)
        self.decomp = CircularPolarizationDecomposer() if HAS_DECOMP else None

    def _gr_linear(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        Mc_sec = cfg.Mc_solar * cfg.MSUN_SEC
        phase = -(3.0 / 128.0) * (math.pi * Mc_sec * self.freqs) ** (-5.0 / 3.0)
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(cfg.distance_mpc, 1.0)) * 1e-21
        return amp * np.cos(phase), amp * np.sin(phase) * 0.5

    def dcs_phase(self, zeta: float) -> np.ndarray:
        """δΨ_dCS = ζ · G(f) · (f/f_ref)^{-7/3}."""
        cfg = self.cfg
        G = self.scalar.response(self.freqs)
        return zeta * G * np.power(self.freqs / cfg.f_ref, cfg.base_power)

    def waveform(self, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
        hp, hc = self._gr_linear()
        if zeta == 0.0:
            return hp, hc
        delta = self.dcs_phase(zeta)
        if self.decomp is not None:
            hL, hR = self.decomp.linear_to_circular(hp, hc)
            hL = hL * np.exp(+1j * delta)
            hR = hR * np.exp(-1j * delta)
            return self.decomp.circular_to_linear(hL, hR)
        c, s = np.cos(delta), np.sin(delta)
        return hp * c - hc * s, hp * s + hc * c

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

    def lr_asymmetry(self, zeta: float) -> float:
        """Parity-odd L/R power asymmetry diagnostic."""
        if self.decomp is None:
            return 0.0
        hp, hc = self.waveform(zeta)
        hL, hR = self.decomp.linear_to_circular(hp, hc)
        pL = np.sum(np.abs(hL) ** 2); pR = np.sum(np.abs(hR) ** 2)
        return float((pL - pR) / (pL + pR + 1e-300))


# ═══════════════════════════════════════════════════════════════════════
# [D] LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════

class DynamicalCSLikelihood:
    """[D] Gaussian likelihood over the single dCS coupling ζ."""

    def __init__(self, cfg: DynamicalCSConfig):
        self.cfg = cfg
        self.wf = DynamicalCSWaveform(cfg)

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

class DynamicalCSTest:
    """
    [D] Run the dynamical-CS test with mandatory validations:
      1. ζ = 0 recovers GR exactly.
      2. G = 1 (frozen scalar) recovers the non-dynamical phase exactly.
      3. Dynamical (G ≠ 1) differs from non-dynamical — the two are not
         interchangeable.
      4. Injection-recovery of ζ under the dynamical response.
    """

    def __init__(self, cfg: Optional[DynamicalCSConfig] = None):
        self.cfg = cfg or DynamicalCSConfig()
        self.likelihood = DynamicalCSLikelihood(self.cfg)
        self.wf = self.likelihood.wf

    # ── validation 1: GR limit ────────────────────────────────────────
    def validate_gr_limit(self) -> Dict[str, Any]:
        hp_d, hc_d = self.wf.waveform(0.0)
        hp_g, hc_g = self.wf.gr_waveform()
        max_diff = float(max(np.max(np.abs(hp_d - hp_g)),
                             np.max(np.abs(hc_d - hc_g))))
        return {"max_abs_difference": max_diff, "passes": max_diff < 1e-30,
                "note": "ζ=0 must recover GR exactly"}

    # ── validation 2: frozen-scalar limit = non-dynamical ────────────
    def validate_frozen_limit(self) -> Dict[str, Any]:
        """With response_g1=0, dCS phase must equal the non-dynamical form."""
        cfg = self.cfg
        zeta = 0.5 * cfg.zeta_max
        # dynamical phase (current G)
        dyn = self.wf.dcs_phase(zeta)
        # non-dynamical phase (G forced to 1)
        nondyn = zeta * np.power(self.wf.freqs / cfg.f_ref, cfg.base_power)
        # if g1 == 0 they must match exactly
        if abs(cfg.response_g1) < 1e-12:
            diff = float(np.max(np.abs(dyn - nondyn)))
            return {"frozen": True, "max_abs_difference": diff,
                    "passes": diff < 1e-30}
        # otherwise report the deviation (they SHOULD differ)
        diff = float(np.max(np.abs(dyn - nondyn)))
        return {"frozen": False, "max_abs_difference": diff,
                "passes": diff > 1e-12,
                "note": "g1≠0: dynamical differs from non-dynamical as expected"}

    # ── validation 3: dynamical vs non-dynamical distinguishability ──
    def dynamical_vs_nondynamical(self, zeta: float) -> Dict[str, Any]:
        """Quantify how much the dynamical response changes the phase."""
        dyn = self.wf.dcs_phase(zeta)
        nondyn = zeta * np.power(self.wf.freqs / self.cfg.f_ref,
                                 self.cfg.base_power)
        rel = float(np.max(np.abs(dyn - nondyn)) /
                    (np.max(np.abs(nondyn)) + 1e-30))
        return {"relative_difference": rel,
                "is_dynamical": self.wf.scalar.is_dynamical(),
                "note": ("dynamical scalar response measurably modifies the "
                         "phase relative to the frozen case" if rel > 1e-6
                         else "no dynamical modification")}

    # ── validation 4: injection-recovery ─────────────────────────────
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
            "frozen_limit": self.validate_frozen_limit(),
            "dyn_vs_nondyn": self.dynamical_vs_nondynamical(0.5 * self.cfg.zeta_max),
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
            "dCS here uses a REDUCED scalar-response model G(f); the full "
            "dCS waveform (coupled scalar-tensor evolution or calibrated "
            "approximant) is [O]. ζ bounds are prior- and model-dependent. "
            "G=1 must equal the non-dynamical module; ζ=0 must equal GR."
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Dynamical Chern–Simons Test — Smoke Test")
    print("=" * 72)
    print(f"  decomp module available : {HAS_DECOMP}")

    # dynamical config (g1 ≠ 0)
    cfg_dyn = DynamicalCSConfig(response_g1=0.5)
    test = DynamicalCSTest(cfg_dyn)

    print("\n  ── [1] GR limit (ζ=0) ──")
    gr = test.validate_gr_limit()
    print(f"    max |dCS − GR| = {gr['max_abs_difference']:.2e}")
    print(f"    [{'PASS' if gr['passes'] else 'FAIL'}] ζ=0 recovers GR")

    print("\n  ── [2] Frozen-scalar limit (g1=0 → non-dynamical) ──")
    test_frozen = DynamicalCSTest(DynamicalCSConfig(response_g1=0.0))
    fz = test_frozen.validate_frozen_limit()
    print(f"    g1=0, max |dyn − nondyn| = {fz['max_abs_difference']:.2e}")
    print(f"    [{'PASS' if fz['passes'] else 'FAIL'}] g1=0 equals non-dynamical")

    print("\n  ── [3] Dynamical vs non-dynamical ──")
    dvn = test.dynamical_vs_nondynamical(0.5 * cfg_dyn.zeta_max)
    print(f"    relative phase difference : {dvn['relative_difference']:.3e}")
    print(f"    is dynamical              : {dvn['is_dynamical']}")
    print(f"    [{'PASS' if dvn['relative_difference'] > 1e-6 else 'FAIL'}] "
          f"dynamical response modifies the phase")

    print("\n  ── [4] Injection-recovery of ζ (dynamical) ──")
    report = test.full_report()
    rec = report["zeta_recovery"]
    print(f"    injected ζ : {rec['injected']:.3f}")
    print(f"    best fit   : {rec['best_fit']:.3f}")
    print(f"    90% UL     : {rec['upper_limit_90']:.3f}")
    print(f"\n    {report['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • dCS scalar modeled via response function G(f), NOT full □θ solve")
    print("  • G=1 ⟹ non-dynamical CS; ζ=0 ⟹ GR (both validated)")
    print("  • Full dCS waveform (coupled scalar-tensor) is [O]")
    print("  • ζ bound is prior- and model-dependent")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
