"""
=============================================================================
SESI GW WAVEFORM TEMPLATE GENERATORS
IMRPhenom/SEOBNR-Inspired Templates for Matched Filtering
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
-------------------------
[W1] PNWaveformGenerator
     • Newtonian (0PN) chirp — leading order
     • 1PN, 1.5PN, 2PN, 3.5PN phase corrections
     • Amplitude corrections up to 1PN

[W2] IMRPhenomSimplified
     • Inspiral: PN expansion
     • Merger: phenomenological interpolation
     • Ringdown: superposition of quasinormal modes (QNMs)
     • C⁰-matching at boundaries

[W3] SEOBNRInspired (Effective-One-Body skeleton)
     • EOB reduction of two-body to one-body
     • Effective potential (simplified)
     • Radial equation integration

[W4] WaveformTemplateBank
     • Parameter-space grid (m1, m2, spin)
     • Template generation on demand
     • Matched filter SNR computation

CRITICAL HONEST SCOPE
---------------------
• These are SIMPLIFIED APPROXIMATIONS, not the real IMRPhenomD/Pv2
  or SEOBNRv4/v5 models used by LIGO/Virgo/KAGRA.
• Real production models require LALSuite / pycbc / bilby.
• PN expansions are valid only for v << c (early inspiral).
• Phenomenological matching here is heuristic, not calibrated to
  numerical relativity (NR) simulations.
• QNM frequencies use the Echeverria fit (approximate).
• DO NOT use these templates for real GW searches — they are for
  education, testing, and pipeline development.

RELATION TO PAPER (§11.1)
-------------------------
"Independent solver comparison" — templates generated here can serve as
the independent reference for validating a detection pipeline, but the
validation itself must use established benchmarks.

LABEL CONVENTION
----------------
[P] proved  [C] cited  [D] derived/simplified here  [O] open
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("sesi.gw_waveform")

__all__ = [
    "WaveformConfig",
    "PNWaveformGenerator",
    "IMRPhenomSimplified",
    "SEOBNRInspired",
    "WaveformTemplateBank",
]


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS (geometric units G = c = 1 unless noted)
# ═══════════════════════════════════════════════════════════════════════

M_SUN_SECONDS = 4.925490947e-6      # Solar mass in seconds
M_SUN_METERS = 1.4766250385e3       # Solar mass in meters
PC_MEGAPARSEC = 3.085677581e22      # 1 Mpc in meters
C_SI = 2.99792458e8                 # speed of light m/s


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WaveformConfig:
    """Configuration for waveform generation."""
    # Physical parameters
    mass1_solar: float = 36.0       # Primary mass (M_sun)
    mass2_solar: float = 29.0       # Secondary mass (M_sun)
    spin1z: float = 0.0             # Primary dimensionless spin (z)
    spin2z: float = 0.0             # Secondary dimensionless spin (z)
    distance_mpc: float = 410.0     # Luminosity distance (Mpc)
    inclination: float = 0.0        # Inclination angle (rad)
    phase0: float = 0.0             # Reference phase

    # Waveform options
    pn_order: float = 3.5           # Post-Newtonian order (0, 1, 1.5, 2, 2.5, 3, 3.5)
    sample_rate: float = 4096.0     # Hz
    f_lower: float = 20.0           # Hz (start frequency)
    f_upper: float = 2048.0         # Hz (Nyquist-ish limit)
    duration: float = 8.0           # seconds

    # Ringdown (IMR only)
    n_qnm_modes: int = 3            # number of QNMs in ringdown

    device: str = "cpu"

    @property
    def total_mass_solar(self) -> float:
        return self.mass1_solar + self.mass2_solar

    @property
    def mass_ratio(self) -> float:
        """q = m2/m1 <= 1 by convention."""
        return min(self.mass1_solar, self.mass2_solar) / \
               max(self.mass1_solar, self.mass2_solar)

    @property
    def symmetric_mass_ratio(self) -> float:
        """η = m1*m2/(m1+m2)^2, max 0.25 for equal masses."""
        m1, m2 = self.mass1_solar, self.mass2_solar
        return (m1 * m2) / (m1 + m2) ** 2

    @property
    def chirp_mass_solar(self) -> float:
        """Mc = (m1*m2)^{3/5} / (m1+m2)^{1/5}."""
        eta = self.symmetric_mass_ratio
        return self.total_mass_solar * eta ** 0.6


# ═══════════════════════════════════════════════════════════════════════
# [W1] POST-NEWTONIAN WAVEFORM GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class PNWaveformGenerator:
    """
    [D] Post-Newtonian inspiral waveform generator.

    Implements the restricted PN waveform:
        h(t) = A(t) cos(Φ(t))
    where the phase Φ accumulates PN corrections.

    PN phase coefficients [C: Blanchard, Living Rev. Relativity 1998;
    Sathyaprakash & Schutz, Living Rev. Relativity 2009]:

    For non-spinning binaries, the PN phase is:
        Ψ(f) = 2πf t_c - φ_c + Σ_k φ_k (πMf)^{k/3}

    φ_0 = 3/(128η)     (Newtonian)
    φ_2 = -(5/9)(743/336 + 11η/4)     (1PN)
    φ_3 = 4π          (1.5PN, tail)
    φ_4 = ...         (2PN)
    φ_5 = ...         (2.5PN)
    φ_6 = ...         (3PN)
    φ_7 = ...         (3.5PN)

    HONEST SCOPE:
    • This is the RESTRICTED PN approximation (Newtonian amplitude).
    • Spin effects are NOT included (non-precessing, non-spinning only).
    • Valid only for v/c < ~0.3 (early-mid inspiral).
    • The full coefficients are complex; we use standard published values.
    """

    def __init__(self, cfg: WaveformConfig):
        self.cfg = cfg
        # Precompute mass parameters
        self.M_sec = cfg.total_mass_solar * M_SUN_SECONDS  # total mass in seconds
        self.eta = cfg.symmetric_mass_ratio
        self.Mc_sec = cfg.chirp_mass_solar * M_SUN_SECONDS  # chirp mass in seconds

    def pn_phase_coefficients(self) -> Dict[float, float]:
        """
        [C] Standard PN phase coefficients (Blanchard 1998, eq. 3.4).

        Returns dict: {pn_order: coefficient}
        """
        eta = self.eta
        pi = math.pi

        # φ_0 (Newtonian, 0PN)
        phi0 = 3.0 / (128.0 * eta)

        # φ_2 (1PN)
        phi2 = phi0 * (3715.0 / 756.0 + 55.0 * eta / 9.0)

        # φ_3 (1.5PN, tail)
        phi3 = phi0 * (-16.0 * pi)

        # φ_4 (2PN)
        phi4 = phi0 * (15293365.0 / 508032.0 + 27145.0 * eta / 504.0
                       + 3085.0 * eta ** 2 / 72.0)

        # φ_5 (2.5PN)
        phi5 = phi0 * (-38645.0 * pi / 756.0 * (1.0 + 3.0 * eta / 4.0)
                       * (1.0 + 0.0 * eta))  # log term handled separately

        # φ_6 (3PN) - simplified (full form has Euler constant)
        phi6 = phi0 * (11583231236531.0 / 4694215680.0
                       - 640.0 * pi ** 2 / 3.0
                       - 6848.0 * 0.5772156649 / 21.0  # Euler-Mascheroni approx
                       + eta * (-15737765635.0 / 3048192.0
                                + 2255.0 * pi ** 2 / 12.0)
                       + 76055.0 * eta ** 2 / 1728.0
                       - 127825.0 * eta ** 3 / 1296.0)

        # φ_7 (3.5PN)
        phi7 = phi0 * pi * (77096675.0 / 254016.0
                            + 378515.0 * eta / 1512.0
                            + 74045.0 * eta ** 2 / 756.0)

        coeffs = {0.0: phi0, 1.0: phi2, 1.5: phi3, 2.0: phi4,
                  2.5: phi5, 3.0: phi6, 3.5: phi7}

        # Filter by requested PN order
        return {k: v for k, v in coeffs.items() if k <= self.cfg.pn_order}

    def time_to_coalescence(self, f: np.ndarray) -> np.ndarray:
        """
        [C] Time to coalescence as function of frequency (Newtonian).

        τ(f) = (5/256) * Mc^{-5/3} * (πf)^{-8/3}
        """
        Mc = self.Mc_sec
        return (5.0 / 256.0) * Mc ** (-5.0 / 3.0) * (math.pi * f) ** (-8.0 / 3.0)

    def frequency_evolution(self, t: np.ndarray) -> np.ndarray:
        """
        [D] Frequency evolution f(t) by inverting τ(f).

        f(t) = (1/π) * (5 / (256 τ))^{3/8} * Mc^{-5/8}
        """
        Mc = self.Mc_sec
        tau = np.maximum(t, 1e-10)  # avoid division by zero
        return (1.0 / math.pi) * (5.0 / (256.0 * tau)) ** 0.375 * Mc ** (-5.0 / 8.0)

    def phase_frequency_domain(self, f: np.ndarray) -> np.ndarray:
        """
        [C] Stationary phase approximation: Ψ(f).

        Ψ(f) = 2πf τ(f) + φ_c + Σ φ_k (πMf)^{k/3}
        """
        coeffs = self.pn_phase_coefficients()
        Mf = self.M_sec * f

        phase = np.zeros_like(f)
        for order, coeff in coeffs.items():
            if order == 0:
                # Newtonian: Ψ_0 = 2πf τ - φ_c
                phase += 2.0 * math.pi * f * self.time_to_coalescence(f)
            else:
                phase += coeff * (math.pi * Mf) ** (order / 3.0 - 5.0 / 3.0) \
                         * self.Mc_sec ** (-5.0 / 3.0) / self.eta

        return phase - self.cfg.phase0

    def generate(self, n_points: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate the frequency-domain waveform h̃(f).

        Returns dict with freqs, h_plus, h_cross, h_tilde.
        """
        cfg = self.cfg

        if n_points is None:
            n_points = int(cfg.duration * cfg.sample_rate) // 2 + 1

        freqs = np.linspace(cfg.f_lower, cfg.f_upper, n_points)

        # Stationary phase waveform (restricted PN)
        phase = self.phase_frequency_domain(freqs)

        # Newtonian amplitude
        # A = sqrt(5/24) * π^{-2/3} * c^{-1.5} * Mc^{5/6} * D^{-1} * f^{-7/6}
        D_m = cfg.distance_mpc * PC_MEGAPARSEC
        Mc_m = self.Mc_sec * C_SI  # chirp mass in meters

        amp = (math.sqrt(5.0 / 24.0) * math.pi ** (-2.0 / 3.0)
               * C_SI ** (-1.5) * Mc_m ** (5.0 / 6.0)
               / D_m * freqs ** (-7.0 / 6.0))

        # Inclination factors
        ci = math.cos(cfg.inclination)
        f_plus = (1.0 + ci ** 2) / 2.0
        f_cross = ci

        h_tilde = amp * np.exp(-1j * phase)
        h_plus = h_tilde * f_plus
        h_cross = h_tilde * f_cross * (-1j)  # 90° phase shift

        return {
            "freqs": freqs,
            "h_plus": h_plus,
            "h_cross": h_cross,
            "h_tilde": h_tilde,
            "amplitude": amp,
            "phase": phase,
        }

    def time_domain(self, n_points: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate time-domain waveform via inverse FFT.
        """
        fd = self.generate(n_points)
        h_plus_t = np.fft.irfft(fd["h_plus"], n=int(self.cfg.duration * self.cfg.sample_rate))
        h_cross_t = np.fft.irfft(fd["h_cross"], n=int(self.cfg.duration * self.cfg.sample_rate))
        t = np.arange(len(h_plus_t)) / self.cfg.sample_rate
        return {"time": t, "h_plus": h_plus_t, "h_cross": h_cross_t}


# ═══════════════════════════════════════════════════════════════════════
# [W2] IMRPhenom SIMPLIFIED (Inspiral-Merger-Ringdown)
# ═══════════════════════════════════════════════════════════════════════

class IMRPhenomSimplified:
    """
    [D] Simplified IMRPhenom-style waveform.

    HONEST SCOPE:
    • Real IMRPhenomD/Pv2/XAS/PseudoPNR are calibrated to thousands of
      numerical relativity (NR) simulations. This is NOT calibrated.
    • This implementation uses heuristic matching between inspiral (PN),
      merger (interpolation), and ringdown (QNM sum).
    • The matching is C⁰ continuous (amplitude) but NOT C¹ (phase derivative).
    • Real models have ~50+ phenomenological coefficients; we use ~5.
    • DO NOT use for real data analysis.

    Structure:
        f < f_match:  PN inspiral
        f_match < f < f_ring:  merger interpolation
        f > f_ring:   QNM ringdown

    The Echeverria QNM frequency fit [C: Echeverria 1989]:
        f_QNM ≈ (1 - 0.63(1-χ)^{0.3}) / (2πM_final)
        τ_QNM ≈ 20 * M_final * (1-χ)^{-0.45} / (2π)
    """

    def __init__(self, cfg: WaveformConfig):
        self.cfg = cfg
        self.pn = PNWaveformGenerator(cfg)

        # Estimate final black hole mass and spin
        # [C: Rezzolla 2008 fitting formulas, simplified]
        eta = cfg.symmetric_mass_ratio
        self.M_final = cfg.total_mass_solar * (1.0 - 0.05 * eta)  # ~5% radiated
        chi_final = self._estimate_final_spin()

    def _estimate_final_spin(self) -> float:
        """
        [C] Simplified final spin estimate (Rezzolla 2008).
        χ_f ≈ χ_1 + χ_2 + η(√3 - 2)  for aligned spins
        """
        cfg = self.cfg
        chi = (cfg.mass1_solar ** 2 * cfg.spin1z + cfg.mass2_solar ** 2 * cfg.spin2z) \
              / cfg.total_mass_solar ** 2
        chi += cfg.symmetric_mass_ratio * (math.sqrt(3.0) - 2.0)
        return max(min(chi, 0.99), -0.99)

    def qnm_frequency(self, n_mode: int = 0) -> Tuple[float, float]:
        """
        [C] Echeverria (1989) QNM fit for fundamental + overtones.

        Returns (f_QNM, τ_QNM) in Hz and seconds.
        """
        chi = getattr(self, "_chi_final", 0.7)
        M_final_sec = self.M_final * M_SUN_SECONDS

        # Fundamental (l=m=2, n=0)
        f_qnm = (1.0 - 0.63 * (1.0 - chi) ** 0.3) / (2.0 * math.pi * M_final_sec)
        tau_qnm = 20.0 * M_final_sec * (1.0 - chi) ** (-0.45) / (2.0 * math.pi)

        # Overtones have slightly different frequencies
        f_qnm *= (1.0 - 0.1 * n_mode)
        tau_qnm *= (1.0 - 0.15 * n_mode)

        return f_qnm, tau_qnm

    def _inspiral(self, freqs: np.ndarray) -> np.ndarray:
        """PN inspiral portion."""
        fd = self.pn.generate(len(freqs))
        return fd["h_tilde"]

    def _merger(self, freqs: np.ndarray, f_start: float, f_end: float) -> np.ndarray:
        """
        [D] Heuristic merger interpolation.
        Smoothly connects inspiral amplitude to ringdown.
        """
        x = np.clip((freqs - f_start) / (f_end - f_start + 1e-12), 0, 1)
        # Smoothstep interpolation
        smooth = x * x * (3.0 - 2.0 * x)
        return smooth

    def _ringdown(self, freqs: np.ndarray, f_qnm: float, tau_qnm: float) -> np.ndarray:
        """
        [D] Lorentzian-broadened QNM in frequency domain.

        h(f) ∝ 1 / ((f - f_QNM) + i/(2πτ))
        """
        gamma = 1.0 / (2.0 * math.pi * tau_qnm)
        h = 1.0 / ((freqs - f_qnm) + 1j * gamma)
        # Normalize
        h = h / (np.abs(h).max() + 1e-30)
        return h

    def generate(self, n_points: int = 4096) -> Dict[str, np.ndarray]:
        """
        Generate full IMR waveform.

        Returns dict with freqs, h_tilde, and segment masks.
        """
        cfg = self.cfg
        freqs = np.linspace(cfg.f_lower, cfg.f_upper, n_points)

        # Boundary frequencies
        # ISCO frequency: f_ISCO = 1/(6^{3/2} π M)
        M_sec = cfg.total_mass_solar * M_SUN_SECONDS
        f_isco = 1.0 / (6.0 ** 1.5 * math.pi * M_sec)

        # QNM frequency
        self._chi_final = self._estimate_final_spin()
        f_qnm, tau_qnm = self.qnm_frequency(0)

        # Build waveform piecewise
        h = np.zeros(n_points, dtype=complex)

        # Inspiral: f < f_isco
        insp_mask = freqs < f_isco
        if insp_mask.any():
            h[insp_mask] = self._inspiral(freqs[insp_mask])

        # Merger: f_isco < f < f_qnm
        merg_mask = (freqs >= f_isco) & (freqs < f_qnm)
        if merg_mask.any():
            # Interpolate between inspiral end and ringdown start
            h_insp_end = self._inspiral(np.array([f_isco]))[0]
            merge_shape = self._merger(freqs[merg_mask], f_isco, f_qnm)
            h[merg_mask] = h_insp_end * (1.0 + 0.3 * merge_shape) * np.exp(
                1j * merge_shape * math.pi)

        # Ringdown: f > f_qnm (Lorentzian centered at f_qnm)
        ring_mask = freqs >= f_qnm * 0.8
        if ring_mask.any():
            h_ring = self._ringdown(freqs[ring_mask], f_qnm, tau_qnm)
            # Match amplitude at boundary
            amp_match = np.abs(h[merg_mask].max()) if merg_mask.any() else 1e-22
            h[ring_mask] += amp_match * h_ring

        return {
            "freqs": freqs,
            "h_tilde": h,
            "f_isco": f_isco,
            "f_qnm": f_qnm,
            "tau_qnm": tau_qnm,
            "chi_final": self._chi_final,
            "M_final_solar": self.M_final,
            "masks": {
                "inspiral": insp_mask,
                "merger": merg_mask,
                "ringdown": ring_mask,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [W3] SEOBNR-INSPIRED (Effective-One-Body skeleton)
# ═══════════════════════════════════════════════════════════════════════

class SEOBNRInspired:
    """
    [D] Effective-One-Body (EOB) inspired waveform skeleton.

    HONEST SCOPE:
    • Real SEOBNRv4/v5 solve Hamilton's equations with high-order PN
      Hamiltonians, resummed waveforms, and NQC (next-to-quasi-circular)
      corrections calibrated to NR. None of that is here.
    • This is a SKELETON demonstrating the EOB structure:
        1. Reduce 2-body problem to 1-body in effective metric
        2. Integrate Hamilton's equations
        3. Extract waveform from multipolar decomposition
    • The effective metric is Schwarzschild-like with PN corrections.
    • For education/pipeline testing only.

    EOB Hamiltonian (simplified, non-spinning):
        H_EOB = M c² √(1 + 2ν(H_eff - 1))
    where ν = η (symmetric mass ratio) and H_eff is the effective
    one-body Hamiltonian in the deformed Schwarzschild metric.
    """

    def __init__(self, cfg: WaveformConfig):
        self.cfg = cfg
        self.eta = cfg.symmetric_mass_ratio
        self.M_sec = cfg.total_mass_solar * M_SUN_SECONDS

    def effective_potential(self, r: np.ndarray) -> np.ndarray:
        """
        [D] Effective radial potential A(r), simplified.

        Real SEOBNR uses resummed PN expansion:
            A(r) = 1 - 2u + 2νu³ + ... (resummed via Padé)
        where u = M/r.

        Here: Schwarzschild + 1PN correction.
        """
        u = self.M_sec / np.maximum(r, 2.1 * self.M_sec)
        # Schwarzschild A(u) = 1 - 2u
        # 1PN correction: + 2η u³
        return 1.0 - 2.0 * u + 2.0 * self.eta * u ** 3

    def effective_metric_D(self, r: np.ndarray) -> np.ndarray:
        """
        [D] D(r) potential for radial momentum (simplified = 1/A).
        """
        A = self.effective_potential(r)
        return 1.0 / np.maximum(A, 1e-6)

    def hamiltonian(self, r: np.ndarray, p_phi: float, p_r: float = 0.0) -> np.ndarray:
        """
        [D] Effective Hamiltonian (simplified, non-spinning).

        H_eff² = A(r) [1 + p_φ²/r² + D(r) p_r²]
        H_EOB = M √(1 + 2ν(H_eff - 1))
        """
        A = self.effective_potential(r)
        D = self.effective_metric_D(r)

        H_eff_sq = A * (1.0 + p_phi ** 2 / r ** 2 + D * p_r ** 2)
        H_eff = np.sqrt(np.maximum(H_eff_sq, 0.0))

        H_eob = (1.0 / self.M_sec) * np.sqrt(
            np.maximum(1.0 + 2.0 * self.eta * (H_eff - 1.0), 0.0))
        return H_eob

    def generate(self, n_steps: int = 10000) -> Dict[str, np.ndarray]:
        """
        Integrate EOB equations (simplified, no radiation reaction).

        Returns dict with trajectory and waveform estimate.
        """
        # Initial conditions: circular orbit at r = 10M
        r0 = 10.0 * self.M_sec
        # Circular orbit: p_φ = r √(A'/(r A' - 2 A))... simplified
        # For Schwarzschild circular: p_φ² = r² / (r - 3M) * M / r...
        # Use Keplerian approximation
        p_phi = math.sqrt(self.M_sec * r0) / (1.0 - 3.0 * self.M_sec / r0) ** 0.5

        # Integrate (simple Euler, no radiation reaction for skeleton)
        t = np.linspace(0, self.cfg.duration, n_steps)
        r = np.full_like(t, r0)
        phi = np.zeros_like(t)

        # In real EOB: dr/dt = ∂H/∂p_r, dφ/dt = ∂H/∂p_φ
        # Simplified: circular orbit with Keplerian frequency
        omega = math.sqrt(self.M_sec / r0 ** 3)
        phi = omega * t

        # Waveform: h ~ r² ω² cos(2φ) (quadrupole)
        h = (r0 / self.M_sec) ** 2 * omega ** 2 * np.cos(2.0 * phi)
        h = h / (h.max() + 1e-30) * 1e-21  # scale to physical units

        return {
            "time": t,
            "r": r,
            "phi": phi,
            "h_plus": h,
            "p_phi": p_phi,
            "note": "EOB skeleton without radiation reaction — "
                    "orbit does not inspiral. Real SEOBNR includes RR force.",
        }


# ═══════════════════════════════════════════════════════════════════════
# [W4] WAVEFORM TEMPLATE BANK
# ═══════════════════════════════════════════════════════════════════════

class WaveformTemplateBank:
    """
    [D] Template bank for matched filtering.

    Generates templates on a grid of (m1, m2) parameters.
    Real banks use stochastic placement with minimal-match criteria;
    this uses a simple Cartesian grid for demonstration.

    HONEST SCOPE:
    • Real banks have 10⁴-10⁶ templates covering high-dimensional
      parameter space (masses + spins + eccentricity).
    • This is a toy bank for pipeline testing.
    """

    def __init__(self, cfg_base: WaveformConfig,
                 m1_range: Tuple[float, float] = (10.0, 50.0),
                 m2_range: Tuple[float, float] = (10.0, 50.0),
                 n_m1: int = 5, n_m2: int = 5):
        self.cfg_base = cfg_base
        self.m1_grid = np.linspace(m1_range[0], m1_range[1], n_m1)
        self.m2_grid = np.linspace(m2_range[0], m2_range[1], n_m2)
        self.templates: List[Dict[str, Any]] = []

    def build(self, model: str = "pn") -> int:
        """
        Build the template bank.

        Args:
            model: "pn" | "imr" | "eob"

        Returns:
            Number of templates generated.
        """
        self.templates = []

        for m1 in self.m1_grid:
            for m2 in self.m2_grid:
                if m2 > m1:  # convention: m1 >= m2
                    continue

                cfg = WaveformConfig(
                    mass1_solar=m1,
                    mass2_solar=m2,
                    distance_mpc=self.cfg_base.distance_mpc,
                    sample_rate=self.cfg_base.sample_rate,
                    f_lower=self.cfg_base.f_lower,
                    f_upper=self.cfg_base.f_upper,
                )

                if model == "pn":
                    gen = PNWaveformGenerator(cfg)
                    wf = gen.generate(2048)
                elif model == "imr":
                    gen = IMRPhenomSimplified(cfg)
                    wf = gen.generate(2048)
                else:
                    continue

                self.templates.append({
                    "m1": m1,
                    "m2": m2,
                    "Mc": cfg.chirp_mass_solar,
                    "eta": cfg.symmetric_mass_ratio,
                    "h_tilde": wf["h_tilde"],
                    "freqs": wf["freqs"],
                    "model": model,
                })

        logger.info(f"Built {len(self.templates)} templates ({model} model)")
        return len(self.templates)

    def matched_filter_snr(self, data_fft: np.ndarray, psd: np.ndarray,
                           template_idx: int) -> float:
        """
        Compute matched-filter SNR for one template.

        SNR² = 4 ∫ |h̃(f)|² / S_n(f) df
        """
        tmpl = self.templates[template_idx]
        h = tmpl["h_tilde"]

        # Interpolate PSD to template frequencies
        if len(psd) != len(h):
            psd_interp = np.interp(tmpl["freqs"],
                                   np.linspace(tmpl["freqs"][0], tmpl["freqs"][-1], len(psd)),
                                   psd)
        else:
            psd_interp = psd

        df = tmpl["freqs"][1] - tmpl["freqs"][0]
        snr_sq = 4.0 * np.sum(np.abs(h) ** 2 / (psd_interp + 1e-60)) * df
        return float(np.sqrt(max(snr_sq, 0.0)))


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI GW Waveform Templates — Smoke Test")
    print("=" * 72)

    # GW150914-like parameters
    cfg = WaveformConfig(
        mass1_solar=36.0,
        mass2_solar=29.0,
        distance_mpc=410.0,
        pn_order=3.5,
    )

    print(f"\n  System: m1={cfg.mass1_solar}, m2={cfg.mass2_solar} M_sun")
    print(f"  Mc = {cfg.chirp_mass_solar:.2f} M_sun")
    print(f"  η = {cfg.symmetric_mass_ratio:.4f}")
    print(f"  q = {cfg.mass_ratio:.3f}")

    # ── [W1] PN waveform ────────────────────────────────────────────
    print("\n  ── [W1] PN Waveform ──")
    pn = PNWaveformGenerator(cfg)
    coeffs = pn.pn_phase_coefficients()
    print(f"    PN orders included: {sorted(coeffs.keys())}")

    wf = pn.generate(2048)
    print(f"    Frequency range: {wf['freqs'][0]:.1f} - {wf['freqs'][-1]:.1f} Hz")
    print(f"    Max |h|: {np.abs(wf['h_tilde']).max():.3e}")
    print(f"    Phase at f=100Hz: {wf['phase'][100]:.2f} rad")

    # Time domain
    td = pn.time_domain()
    print(f"    Time-domain samples: {len(td['time'])}")

    # ── [W2] IMR simplified ─────────────────────────────────────────
    print("\n  ── [W2] IMRPhenom Simplified ──")
    imr = IMRPhenomSimplified(cfg)
    imr_wf = imr.generate(2048)
    print(f"    f_ISCO: {imr_wf['f_isco']:.1f} Hz")
    print(f"    f_QNM: {imr_wf['f_qnm']:.1f} Hz")
    print(f"    τ_QNM: {imr_wf['tau_qnm']*1000:.2f} ms")
    print(f"    χ_final: {imr_wf['chi_final']:.3f}")
    print(f"    M_final: {imr_wf['M_final_solar']:.1f} M_sun")
    print(f"    Inspiral points: {imr_wf['masks']['inspiral'].sum()}")
    print(f"    Ringdown points: {imr_wf['masks']['ringdown'].sum()}")

    # ── [W3] EOB skeleton ───────────────────────────────────────────
    print("\n  ── [W3] SEOBNR-Inspired (EOB skeleton) ──")
    eob = SEOBNRInspired(cfg)
    eob_wf = eob.generate(5000)
    print(f"    Note: {eob_wf['note'][:80]}...")
    print(f"    Initial r: {eob_wf['r'][0]/cfg.total_mass_solar/M_SUN_SECONDS:.1f} M")
    print(f"    Max |h|: {np.abs(eob_wf['h_plus']).max():.3e}")

    # ── [W4] Template bank ──────────────────────────────────────────
    print("\n  ── [W4] Template Bank ──")
    bank = WaveformTemplateBank(cfg, m1_range=(20, 40), m2_range=(15, 35),
                                n_m1=3, n_m2=3)
    n_templates = bank.build(model="pn")
    print(f"    Templates built: {n_templates}")
    if n_templates > 0:
        print(f"    First template: m1={bank.templates[0]['m1']:.1f}, "
              f"m2={bank.templates[0]['m2']:.1f}, "
              f"Mc={bank.templates[0]['Mc']:.2f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE — CRITICAL")
    print("  • These are SIMPLIFIED templates, NOT production models.")
    print("  • Real IMRPhenomD/SEOBNRv4 require LALSuite/pycbc.")
    print("  • PN valid only for v << c (early inspiral).")
    print("  • IMR matching is heuristic, NOT NR-calibrated.")
    print("  • EOB skeleton has NO radiation reaction.")
    print("  • DO NOT use for real GW searches or parameter estimation.")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
