"""
=============================================================================
SESI WAVEFORM TEMPLATE BANK — IMRPhenom / SEOBNR INTERFACE
Real LALSimulation backend + honestly-labeled reduced surrogates
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

STATUS LABELS
-------------
[C] IMRPhenomD  : Purrer et al. (2016), phenomenological frequency-domain
[C] SEOBNRv4    : Bohé et al. (2017), EOB + NR-calibrated
[C] Ringdown    : Berti et al. (2006), Kerr quasi-normal modes
[D] Surrogates  : reduced structure ONLY — NOT calibrated, NOT for science

CRITICAL HONESTY
----------------
The surrogate below reproduces the *architecture* of an IMR-style waveform:
    inspiral phasing -> transition -> merger-ringdown
with leading/1PN phasing and a Lorentzian ringdown. It DOES NOT reproduce:
    * the calibrated IMRPhenomD amplitude pieces / phase coefficients
    * the EOB Hamiltonian, higher-order flux, or NR calibration of SEOBNRv4
    * the actual Berti ringdown fit coefficients
Ringdown here uses a documented TWO-ANCHOR interpolation, explicitly a toy.

For science: set backend="lalsimulation" and install lalsuite.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.waveforms")

try:
    import lalsimulation as lalsim
    import lal
    HAS_LAL = True
except ImportError:
    HAS_LAL = False

__all__ = [
    "WaveformConfig",
    "BinaryParameters",
    "KerrRingdown",
    "InspiralPhasing",
    "IMRPhenomSurrogate",
    "EOBInspiralSurrogate",
    "WaveformBank",
    "MatchedFilterMetrics",
]


# ═══════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BinaryParameters:
    """Binary source parameters (SI masses in solar masses)."""
    mass_1: float = 36.0
    mass_2: float = 29.0
    spin_1z: float = 0.0
    spin_2z: float = 0.0
    distance_mpc: float = 400.0
    inclination: float = 0.0
    phase_ref: float = 0.0
    ra: float = 0.0
    dec: float = 0.0

    @property
    def m_total(self) -> float:
        return self.mass_1 + self.mass_2

    @property
    def eta(self) -> float:
        """Symmetric mass ratio."""
        return (self.mass_1 * self.mass_2) / self.m_total**2

    @property
    def chirp_mass(self) -> float:
        """Mc = (m1 m2)^{3/5} / (m1+m2)^{1/5}."""
        return (self.mass_1 * self.mass_2)**0.6 / self.m_total**0.2

    @property
    def chi_s(self) -> float:
        return 0.5 * (self.spin_1z + self.spin_2z)

    @property
    def chi_a(self) -> float:
        return 0.5 * (self.spin_1z - self.spin_2z)


@dataclass
class WaveformConfig:
    sample_rate: float = 4096.0
    duration: float = 8.0
    f_min: float = 20.0
    f_max: float = 1024.0
    backend: str = "auto"   # auto | lalsimulation | surrogate
    approximant: str = "IMRPhenomD"
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [C/D] KERR RINGDOWN — honest two-anchor toy
# ═══════════════════════════════════════════════════════════════════════

class KerrRingdown:
    """
    [D-TOY] Two-anchor interpolation of the fundamental (l=m=2, n=0) mode.

    Anchors (well-known Schwarzschild limit + qualitative high-spin limit):
        a=0   : M*omega_R = 0.3737 , M*tau = 11.24
        a=0.9 : M*omega_R ~ 0.80   , M*tau ~  3.0

    !!! NOT the Berti et al. fit. NOT calibrated. !!!
    Provided only so the surrogate has a ringdown timescale to work with.
    For real values use lalsimulation or the published Berti coefficients.
    """

    A_LOW = (0.3737, 11.24)
    A_HIGH = (0.80, 3.0)
    SPIN_HIGH = 0.9

    def __call__(self, final_spin: float, total_mass_solar: float) -> Tuple[float, float]:
        """Return (f_RD [Hz], tau_RD [s])."""
        a = float(np.clip(abs(final_spin), 0.0, self.SPIN_HIGH))
        w = a / self.SPIN_HIGH
        # smooth weight
        w = w * w * (3.0 - 2.0 * w)
        M_omega = (1 - w) * self.A_LOW[0] + w * self.A_HIGH[0]
        M_tau = (1 - w) * self.A_LOW[1] + w * self.A_HIGH[1]

        # M_sun in seconds (G*M_sun/c^3)
        msun_sec = 4.925490947e-6
        M_sec = total_mass_solar * msun_sec

        f_rd = M_omega / (2.0 * np.pi * M_sec)
        tau_rd = M_tau * M_sec
        return float(f_rd), float(tau_rd)


# ═══════════════════════════════════════════════════════════════════════
# [C] INSPIRAL PHASING — TaylorF2 structure, 0PN + 1PN only
# ═══════════════════════════════════════════════════════════════════════

class InspiralPhasing:
    """
    [C-REDUCED] TaylorF2-style phasing, truncated at 1PN.

    Psi(f) = 2 pi f t_c - phi_c - pi/4
             + (3/(128 eta)) x^{-5/3} [1 + c1 x + c2 x^2 + ...]
    with x = (pi M f)^{2/3}, c1 = 3715/756 + 55 eta/9.

    Higher PN orders, spin-tidal terms and log terms are OMITTED.
    This is NOT the full IMRPhenomD phase.
    """

    def __init__(self, params: BinaryParameters):
        self.p = params
        self.msun_sec = 4.925490947e-6

    def __call__(self, f: np.ndarray, tc: float = 0.0) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        M_sec = self.p.m_total * self.msun_sec
        pifMf = np.pi * M_sec * np.maximum(f, 1e-6)
        x = pifMf ** (2.0 / 3.0)

        eta = max(self.p.eta, 1e-6)
        c1 = 3715.0 / 756.0 + 55.0 * eta / 9.0

        pref = 3.0 / (128.0 * eta)
        psi = pref * x ** (-5.0 / 3.0) * (1.0 + c1 * x)
        psi = psi - np.pi / 4.0 + 2.0 * np.pi * f * tc - self.p.phase_ref
        return psi


# ═══════════════════════════════════════════════════════════════════════
# [D] IMR-PHENOM-STYLE SURROGATE (three regions, NOT IMRPhenomD)
# ═══════════════════════════════════════════════════════════════════════

class IMRPhenomSurrogate:
    """
    [D-TOY] Reduced IMR-style frequency-domain waveform.

    Region structure mirrors IMRPhenom:
        inspiral  : A ~ f^{-7/6}, TaylorF2-style phase
        transition: smooth tanh blending
        ringdown  : Lorentzian at f_RD with damping tau_RD

    NOT a calibrated surrogate. Do not report scientific SNR with it.
    """

    def __init__(self, cfg: WaveformConfig, params: BinaryParameters):
        self.cfg, self.p = cfg, params
        self.phasing = InspiralPhasing(params)
        self.ringdown = KerrRingdown()

    def frequency_domain(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, h_tilde) for the reduced model."""
        df = self.cfg.sample_rate / int(self.cfg.sample_rate * self.cfg.duration)
        f = np.arange(self.cfg.f_min, self.cfg.f_max, df)
        if len(f) < 4:
            f = np.linspace(self.cfg.f_min, self.cfg.f_max, 512)

        # Final spin proxy (not the calibrated final-spin fit)
        chi_final = float(np.clip(
            self.p.chi_s + 0.5 * self.p.eta, -0.9, 0.9
        ))
        f_rd, tau_rd = self.ringdown(chi_final, self.p.m_total)

        # Leading-order inspiral amplitude, normalized at f_min
        amp_ins = f ** (-7.0 / 6.0)
        amp_ins = amp_ins / (amp_ins[0] + 1e-30)

        # Ringdown Lorentzian (unit peak)
        gamma_rd = 1.0 / (2.0 * max(tau_rd, 1e-6))
        lor = gamma_rd**2 / ((f - f_rd)**2 + gamma_rd**2 + 1e-30)

        # Smooth transition weight (tanh = C^infinity, per SESI convention)
        f_trans = 0.85 * f_rd
        width = max(0.08 * f_rd, df)
        w_rd = 0.5 * (1.0 + np.tanh((f - f_trans) / width))

        amp = (1.0 - w_rd) * amp_ins + w_rd * lor * 0.35

        # Distance scaling (schematic, not calibrated)
        amp = amp / max(self.p.distance_mpc, 1.0) * 100.0 * 1e-21

        psi = self.phasing(f)
        h_tilde = amp * np.exp(-1j * psi)

        # Zero out outside band
        return f, h_tilde

    def time_domain(self) -> Tuple[np.ndarray, np.ndarray]:
        """IFFT to time domain with proper normalization."""
        N = int(self.cfg.sample_rate * self.cfg.duration)
        f, ht = self.frequency_domain()

        ht_full = np.zeros(N // 2 + 1, dtype=complex)
        idx = np.searchsorted(np.fft.rfftfreq(N, 1.0 / self.cfg.sample_rate), f)
        valid = idx < len(ht_full)
        ht_full[idx[valid]] = ht[valid]

        # Normalize physical scaling (rfft convention)
        h = np.fft.irfft(ht_full * (N // 2 + 1), n=N)
        t = np.arange(N) / self.cfg.sample_rate
        return t, h


# ═══════════════════════════════════════════════════════════════════════
# [D] EOB-INSPIRED SURROGATE — PN energy balance, NOT SEOBNRv4
# ═══════════════════════════════════════════════════════════════════════

class EOBInspiralSurrogate:
    """
    [D-TOY] EOB-inspired structure via PN energy balance.

    Evolves x = (M Omega)^{2/3} using
        de/dx with e(x) = -x/2 (1 - 3x/4)
        flux  = (32/5) eta x^5
        dx/dt = -flux / (de/dx)

    This is a PN chirp, NOT the EOB Hamiltonian and NOT SEOBNRv4.
    No NR calibration, no higher-order flux, no spin-orbit coupling.
    """

    def __init__(self, cfg: WaveformConfig, params: BinaryParameters):
        self.cfg, self.p = cfg, params
        self.msun_sec = 4.925490947e-6

    def time_domain(self) -> Tuple[np.ndarray, np.ndarray]:
        fs = self.cfg.sample_rate
        N = int(fs * self.cfg.duration)
        t = np.arange(N) / fs

        eta = max(self.p.eta, 1e-4)
        M_sec = self.p.m_total * self.msun_sec

        # Start at f_min, integrate in geometric time
        x_start = (np.pi * M_sec * self.cfg.f_min) ** (2.0 / 3.0)
        x = max(float(x_start), 1e-4)

        # Geometric timestep for stability
        n_sub = 8
        dt_geo = (self.cfg.duration / M_sec) / (N * n_sub)

        phase = 0.0
        h = np.zeros(N)
        x_isco = 1.0 / 6.0  # toy ISCO marker

        for i in range(N):
            for _ in range(n_sub):
                de_dx = -0.5 * (1.0 - 1.5 * x)
                flux = (32.0 / 5.0) * eta * x**5
                dx_dt = -flux / (de_dx - 1e-12)
                x_new = x + dx_dt * dt_geo
                if x_new >= x_isco:
                    x_new = x_isco
                x = max(x_new, 1e-6)
                omega_geo = x**1.5     # Omega in units 1/M
                phase += omega_geo * dt_geo

            f_gw = omega_geo / (np.pi * M_sec) if M_sec > 0 else 0.0
            f_gw = float(np.clip(f_gw, self.cfg.f_min, fs * 0.45))

            # Leading-order amplitude
            amp = (self.p.chirp_mass ** (5.0 / 3.0) * f_gw ** (2.0 / 3.0)
                   / max(self.p.distance_mpc, 1.0) * 1e-20)
            taper = 1.0 if x < x_isco else 0.0
            h[i] = amp * np.cos(2.0 * phase + self.p.phase_ref) * taper

        return t, h


# ═══════════════════════════════════════════════════════════════════════
# REAL LAL BACKEND WRAPPER
# ═══════════════════════════════════════════════════════════════════════

def generate_lalsimulation(cfg: WaveformConfig, p: BinaryParameters) -> Optional[Dict]:
    """
    [C] Generate a REAL calibrated waveform via lalsimulation.

    Returns None if lalsuite is unavailable (caller must fall back and
    clearly mark the result as a surrogate).
    """
    if not HAS_LAL:
        return None

    try:
        dist_m = p.distance_mpc * 3.085677581e22
        m1_kg = p.mass_1 * lal.MSUN_SI
        m2_kg = p.mass_2 * lal.MSUN_SI

        approx = getattr(lalsim, f"IMRPhenomD")
        if cfg.approximant == "SEOBNRv4":
            approx = getattr(lalsim, "SEOBNRv4")

        hap = lalsim.SimInspiralChooseTDWaveform(
            m1_kg, m2_kg,
            0.0, 0.0, p.spin_1z,
            0.0, 0.0, p.spin_2z,
            dist_m, p.inclination, 0.0, 0.0, 0.0, p.phase_ref,
            -1.0, -1.0, -1.0,
            cfg.f_min, cfg.sample_rate, cfg.duration,
            approx, None,
        )

        t = np.arange(hap.data.length) / hap.deltaT + hap.epoch
        return {
            "t": t, "h_plus": np.array(hap.data.data),
            "approximant": cfg.approximant,
            "calibrated": True, "backend": "lalsimulation",
        }
    except Exception as e:
        logger.warning(f"LALSimulation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# MATCHED FILTER METRICS
# ═══════════════════════════════════════════════════════════════════════

class MatchedFilterMetrics:
    """Noise-weighted inner product, overlap, mismatch."""

    def __init__(self, cfg: WaveformConfig):
        self.cfg = cfg

    def inner_product(self, a: np.ndarray, b: np.ndarray,
                      psd: np.ndarray) -> float:
        N = len(a)
        fs = self.cfg.sample_rate
        freqs = np.fft.rfftfreq(N, 1.0 / fs)
        af, bf = np.fft.rfft(a), np.fft.rfft(b)
        psd_i = np.interp(freqs,
                          np.linspace(0, fs / 2, len(psd)),
                          np.maximum(psd, 1e-50))
        mask = (freqs >= self.cfg.f_min) & (freqs <= self.cfg.f_max)
        integrand = 4.0 * af * np.conj(bf) / (psd_i + 1e-60)
        integrand[~mask] = 0.0
        df = fs / N
        return float(4.0 * df * np.real(np.sum(integrand)))

    def overlap_max(self, a: np.ndarray, b: np.ndarray,
                    psd: np.ndarray) -> float:
        """Max over time shift and one global phase (approximate)."""
        na = math.sqrt(max(self.inner_product(a, a, psd), 1e-60))
        nb = math.sqrt(max(self.inner_product(b, b, psd), 1e-60))
        N = len(a)
        fs = self.cfg.sample_rate
        freqs = np.fft.rfftfreq(N, 1.0 / fs)
        psd_i = np.interp(freqs, np.linspace(0, fs/2, len(psd)),
                          np.maximum(psd, 1e-50))
        mask = (freqs >= self.cfg.f_min) & (freqs <= self.cfg.f_max)
        cross = 4.0 * np.fft.rfft(a) * np.conj(np.fft.rfft(b)) / (psd_i + 1e-60)
        cross[~mask] = 0.0
        corr = np.fft.irfft(cross, n=N) * fs / N
        return float(np.max(np.abs(corr)) / (na * nb))

    def mismatch(self, a: np.ndarray, b: np.ndarray,
                 psd: np.ndarray) -> float:
        return float(1.0 - np.clip(self.overlap_max(a, b, psd), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# TEMPLATE BANK with mismatch pruning
# ═══════════════════════════════════════════════════════════════════════

class WaveformBank:
    """Grid template bank with mismatch-based pruning."""

    def __init__(self, cfg: WaveformConfig, max_mismatch: float = 0.05):
        self.cfg, self.max_mismatch = cfg, max_mismatch
        self.templates: List[Dict] = []

    def build(self, n_mass: int = 4, n_spin: int = 2) -> List[Dict]:
        mf = MatchedFilterMetrics(self.cfg)
        N = int(self.cfg.sample_rate * self.cfg.duration)
        freqs = np.fft.rfftfreq(N, 1.0 / self.cfg.sample_rate)
        # Toy flat ASD-limited PSD for bank pruning ONLY
        psd = np.ones(len(freqs)) * 1e-46

        self.templates = []
        for i in range(n_mass):
            m2 = 5.0 + (25.0) * i / max(n_mass - 1, 1)
            for j in range(n_spin):
                spin = -0.5 + 1.0 * j / max(n_spin - 1, 1)
                p = BinaryParameters(mass_1=m2 * 1.8, mass_2=m2,
                                     spin_1z=spin, spin_2z=0.0)
                t, h = IMRPhenomSurrogate(self.cfg, p).time_domain()
                if len(h) != N:
                    h = np.resize(h, N)

                # Prune against kept templates
                redundant = False
                for kept in self.templates:
                    if mf.mismatch(h, kept["h"], psd) < self.max_mismatch:
                        redundant = True
                        break
                if not redundant:
                    self.templates.append({"params": p, "h": h, "t": t})

        logger.info(f"Bank: {n_mass*n_spin} candidates -> "
                    f"{len(self.templates)} after pruning")
        return self.templates


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Waveform Templates — Smoke Test")
    print("=" * 72)
    print(f"  lalsimulation available: {HAS_LAL}")

    cfg = WaveformConfig(sample_rate=2048, duration=4.0)
    p = BinaryParameters(mass_1=36., mass_2=29., spin_1z=0.3)

    print(f"\n  Binary: M={p.m_total}, eta={p.eta:.4f}, Mc={p.chirp_mass:.3f}")

    f_rd, tau_rd = KerrRingdown()(0.5, p.m_total)
    print(f"  [D-TOY] ringdown: f_RD={f_rd:.1f} Hz, tau={tau_rd:.4f} s")

    t1, h1 = IMRPhenomSurrogate(cfg, p).time_domain()
    t2, h2 = EOBInspiralSurrogate(cfg, p).time_domain()
    print(f"  IMR-style surrogate : {len(h1)} samples, peak={np.abs(h1).max():.2e}")
    print(f"  EOB-inspired chirp  : {len(h2)} samples, peak={np.abs(h2).max():.2e}")

    real = generate_lalsimulation(cfg, p)
    print(f"  Calibrated LAL waveform: {'YES' if real else 'NO (surrogate only)'}")

    mf = MatchedFilterMetrics(cfg)
    N = len(h1)
    psd = np.ones(N // 2 + 1) * 1e-46
    mm = mf.mismatch(h1, h2, psd)
    print(f"  Mismatch IMR-toy vs EOB-toy = {mm:.4f}  (diagnostic only)")

    bank = WaveformBank(cfg).build(n_mass=3, n_spin=2)
    print(f"  Pruned bank size: {len(bank)}")

    print("\n  [HONEST] Surrogates are NOT calibrated; no science SNR claims.")
    print("  [HONEST] Install lalsuite for IMRPhenomD / SEOBNRv4 production use.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
