"""
=============================================================================
SESI FULL VT WITH LALSIMULATION WAVEFORMS
Monte-Carlo sensitive volume-time over the full CBC parameter space
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT "FULL VT" MEANS (standard LVK methodology) [C]
---------------------------------------------------
VT = T_obs × ∫ dθ p(θ) p_det(θ) × V_ref

evaluated by Monte-Carlo over the compact-binary parameter space:
    θ = {m1, m2, spin1, spin2, distance, sky(ra,dec),
         inclination, polarization ψ, phase φ}

For each injection:
  1. Generate h+, h× with lalsimulation (IMRPhenom/SEOBNR) at a reference
     distance, then scale amplitude ∝ 1/d_L.
  2. Project onto each detector with antenna patterns F+, F×.
  3. Compute per-detector optimal SNR and combine into network SNR.
  4. "Found" if network SNR exceeds threshold (optionally per-detector too).

VT = V_ref × (N_found / N_total) × T_obs, with binomial uncertainty.

WAVEFORM BACKEND
----------------
• PRIMARY  : lalsimulation.SimInspiralChooseTDWaveform (if lalsuite present)
• FALLBACK : the labeled surrogate from sesi_waveform_templates — clearly
  flagged. VT on the surrogate is a METHOD DEMO, not a sensitivity claim.

LABELS: [C] LVK VT methodology  [D] derived implementation  [O] open

HONEST SCOPE
------------
• Antenna patterns use the standard JKS98 form in a wave frame; the full
  detector-frame transformation (GMST, site geometry) is simplified here.
  Production VT uses lal detector objects.
• Detection criterion is SNR-only by default; a real search adds χ²,
  coincidence, and ranking-statistic thresholds. This is flagged.
• VT is a property of PIPELINE + waveform + cosmology + threshold, not of
  Nature. The Monte-Carlo binomial uncertainty is always reported.
• Without lalsuite, results are synthetic and MUST NOT be cited as a
  physical sensitivity.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.vt_lal")

try:
    import lalsimulation as lalsim
    import lal
    HAS_LAL = True
except ImportError:
    HAS_LAL = False

try:
    from sesi_cosmology_vt import Cosmology
    HAS_COSMO = True
except ImportError:
    HAS_COSMO = False

__all__ = [
    "VTFullConfig",
    "LALWaveformGenerator",
    "AntennaPattern",
    "NetworkSNRCalculator",
    "CBCInjectionSampler",
    "FullVTCalculator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VTFullConfig:
    # Waveform
    approximant: str = "IMRPhenomD"
    f_min: float = 20.0
    sample_rate: float = 4096.0
    reference_distance_mpc: float = 100.0

    # Detectors
    detectors: List[str] = field(default_factory=lambda: ["H1", "L1", "V1"])
    psd_name: str = "aLIGOZeroDetHighPower"

    # Detection criterion
    network_snr_threshold: float = 12.0
    single_det_snr_threshold: float = 4.0
    require_n_detectors: int = 2

    # Monte-Carlo
    n_injections: int = 2000
    d_max_mpc: float = 1000.0      # injection volume horizon
    T_obs_yr: float = 1.0

    # Source distribution
    mass_min: float = 5.0
    mass_max: float = 60.0
    spin_max: float = 0.8

    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# [C/D] WAVEFORM GENERATOR (lalsimulation primary, surrogate fallback)
# ═══════════════════════════════════════════════════════════════════════

class LALWaveformGenerator:
    """
    [C/D] Generate h+, h× at a reference distance.

    Uses lalsimulation.SimInspiralChooseTDWaveform when available.
    Falls back to a clearly-labeled toy chirp otherwise.
    """

    MSUN_SI = 1.98892e30
    MPC_SI = 3.085677581e22

    def __init__(self, cfg: VTFullConfig):
        self.cfg = cfg
        self.backend = "lalsimulation" if HAS_LAL else "surrogate"
        if not HAS_LAL:
            logger.warning("lalsuite unavailable → VT will use a labeled "
                           "surrogate waveform (method demo only).")

    def generate(self, m1: float, m2: float, s1z: float, s2z: float,
                 distance_mpc: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Return (h_plus, h_cross, dt). Amplitude already at `distance_mpc`.
        """
        if HAS_LAL:
            return self._generate_lal(m1, m2, s1z, s2z, distance_mpc)
        return self._generate_surrogate(m1, m2, distance_mpc)

    def _generate_lal(self, m1, m2, s1z, s2z, distance_mpc):
        approx = getattr(lalsim, self.cfg.approximant)
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            m1 * self.MSUN_SI, m2 * self.MSUN_SI,
            0.0, 0.0, s1z, 0.0, 0.0, s2z,
            distance_mpc * self.MPC_SI,
            0.0, 0.0, 0.0, 0.0, 0.0,     # inclination, phi0, longAsc, ecc, anomaly
            -1.0, self.cfg.f_min, self.cfg.sample_rate, 0.0,
            approx, None)
        h_plus = np.array(hp.data.data)
        h_cross = np.array(hc.data.data)
        return h_plus, h_cross, hp.deltaT

    def _generate_surrogate(self, m1, m2, distance_mpc):
        """[D-TOY] leading-order chirp; clearly NOT calibrated."""
        fs = self.cfg.sample_rate
        n = int(fs * 4.0)
        t = np.arange(n) / fs
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        mc_sec = mc * 4.925490947e-6
        t_c = 4.0 * 0.7
        tau = np.maximum(t_c - t, 1e-4)
        f_gw = (1 / math.pi) * (5 / (256 * tau)) ** 0.375 * mc_sec ** -0.625
        f_gw = np.clip(f_gw, self.cfg.f_min, fs / 2 * 0.9)
        phase = 2 * math.pi * np.cumsum(f_gw) / fs
        amp = mc ** (5 / 6) * f_gw ** (2 / 3) / max(distance_mpc, 1.0)
        amp = amp / (amp.max() + 1e-30) * 1e-21
        h_plus = amp * np.cos(phase)
        h_cross = amp * np.sin(phase) * 0.5
        return h_plus, h_cross, 1.0 / fs


# ═══════════════════════════════════════════════════════════════════════
# [C] ANTENNA PATTERNS (JKS98 form)
# ═══════════════════════════════════════════════════════════════════════

class AntennaPattern:
    """
    [C] Standard interferometer antenna patterns F+, F× in the wave frame
    (Jaranowski-Krolak-Schutz 1998):
        F+ = ½(1+cos²θ) cos2φ cos2ψ − cosθ sin2φ sin2ψ
        F× = ½(1+cos²θ) cos2φ sin2ψ + cosθ sin2φ cos2ψ
    HONEST: full detector-frame transformation (site geometry + GMST) is
    simplified; production uses lal detector objects.
    """

    @staticmethod
    def response(theta: float, phi: float, psi: float) -> Tuple[float, float]:
        ct, st = math.cos(theta), math.sin(theta)
        c2p, s2p = math.cos(2 * phi), math.sin(2 * phi)
        c2psi, s2psi = math.cos(2 * psi), math.sin(2 * psi)
        F_plus = 0.5 * (1 + ct * ct) * c2p * c2psi - ct * s2p * s2psi
        F_cross = 0.5 * (1 + ct * ct) * c2p * s2psi + ct * s2p * c2psi
        return F_plus, F_cross


# ═══════════════════════════════════════════════════════════════════════
# [D] NETWORK SNR
# ═══════════════════════════════════════════════════════════════════════

class NetworkSNRCalculator:
    """
    [D] Optimal per-detector SNR and network combination.

    For a detector response h = F+ h+ + F× h×, the optimal SNR is
        ρ² = 4 ∫ |h̃(f)|² / Sn(f) df
    computed here in the time domain for white-noise simplicity, with the
    inclination/polarization factors folded into F+, F× and the waveform
    polarizations. Network SNR: ρ_net = sqrt(Σ ρ_i²).
    """

    def __init__(self, cfg: VTFullConfig, waveform_gen: LALWaveformGenerator):
        self.cfg = cfg
        self.wfg = waveform_gen
        self.antenna = AntennaPattern()

    def _optimal_snr(self, h: np.ndarray, dt: float, noise_amp: float) -> float:
        """Optimal SNR for a strain time series against white noise."""
        h_f = np.fft.rfft(h)
        df = 1.0 / (len(h) * dt)
        psd = noise_amp ** 2 / df  # white-noise PSD level
        psd = max(psd, 1e-60)
        integrand = np.abs(h_f) ** 2 / psd
        return float(math.sqrt(4.0 * np.sum(integrand) * df))

    def network_snr(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute network SNR for one injection.

        params: m1, m2, s1z, s2z, distance_mpc, ra, dec, iota, psi, phase
        """
        hp, hc, dt = self.wfg.generate(
            params["m1"], params["m2"], params["s1z"], params["s2z"],
            params["distance_mpc"])

        # inclination factors on the polarizations
        iota = params["iota"]
        incl_plus = 0.5 * (1.0 + math.cos(iota) ** 2)
        incl_cross = math.cos(iota)
        # phase rotation mixes + and × (simplified)
        phase = params["phase"]
        hp_rot = hp * math.cos(phase) - hc * math.sin(phase)
        hc_rot = hp * math.sin(phase) + hc * math.cos(phase)
        hp_eff = hp_rot * incl_plus
        hc_eff = hc_rot * incl_cross

        # wave-frame angles from sky position (simplified mapping)
        theta = math.acos(np.clip(math.sin(params["dec"]), -1, 1))
        phi = params["ra"]

        per_det_snr = {}
        # RMS antenna pattern ≈ 1/√5 per IFO for a single detector; we
        # evaluate an actual F+, F× at randomized sky for each detector.
        rng = np.random.default_rng(hash(tuple(sorted(params.items()))) % 2**31)
        for det in self.cfg.detectors:
            Fp, Fc = self.antenna.response(theta, phi + rng.uniform(0, 2 * math.pi),
                                           params["psi"])
            h_det = Fp * hp_eff + Fc * hc_eff
            noise_amp = 1e-23  # placeholder white-noise level
            per_det_snr[det] = self._optimal_snr(h_det, dt, noise_amp)

        snrs = np.array(list(per_det_snr.values()))
        # network SNR over detectors passing the single-detector threshold
        passing = snrs[snrs >= self.cfg.single_det_snr_threshold]
        if len(passing) >= self.cfg.require_n_detectors:
            net = float(np.sqrt(np.sum(passing ** 2)))
        else:
            net = 0.0

        return {
            "per_detector_snr": per_det_snr,
            "network_snr": net,
            "n_passing": int(len(passing)),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] CBC INJECTION SAMPLER
# ═══════════════════════════════════════════════════════════════════════

class CBCInjectionSampler:
    """
    [D] Draw CBC source parameters for the VT Monte-Carlo.

    Distance is drawn UNIFORM IN COMOVING VOLUME (d ∝ u^{1/3}); sky uniform
    on the sphere; inclination isotropic (cos ι uniform); spins/masses
    uniform within configured bounds.
    """

    def __init__(self, cfg: VTFullConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> List[Dict[str, float]]:
        inj = []
        for _ in range(n):
            u = self.rng.uniform()
            distance = self.cfg.d_max_mpc * u ** (1.0 / 3.0)
            m1 = self.rng.uniform(self.cfg.mass_min, self.cfg.mass_max)
            m2 = self.rng.uniform(self.cfg.mass_min, m1)
            inj.append({
                "m1": float(m1), "m2": float(m2),
                "s1z": float(self.rng.uniform(-self.cfg.spin_max, self.cfg.spin_max)),
                "s2z": float(self.rng.uniform(-self.cfg.spin_max, self.cfg.spin_max)),
                "distance_mpc": float(max(distance, 1.0)),
                "ra": float(self.rng.uniform(0, 2 * math.pi)),
                "dec": float(math.asin(self.rng.uniform(-1, 1))),
                "iota": float(math.acos(self.rng.uniform(-1, 1))),
                "psi": float(self.rng.uniform(0, math.pi)),
                "phase": float(self.rng.uniform(0, 2 * math.pi)),
            })
        return inj


# ═══════════════════════════════════════════════════════════════════════
# [C/D] FULL VT CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class FullVTCalculator:
    """
    [C/D] Monte-Carlo sensitive volume-time using the full CBC parameter
    space and lalsimulation waveforms.

        VT = V(d_max) × (N_found / N_total) × T_obs

    with binomial uncertainty on the found fraction. Optionally converts
    d_max to a redshift via the cosmology module for a cosmological VT.
    """

    def __init__(self, cfg: VTFullConfig):
        self.cfg = cfg
        self.wfg = LALWaveformGenerator(cfg)
        self.snr_calc = NetworkSNRCalculator(cfg, self.wfg)
        self.sampler = CBCInjectionSampler(cfg, seed=cfg.seed)

    def _volume_within(self, d_mpc: float) -> float:
        """Comoving volume (Mpc³), low-z Euclidean. Cosmology optional."""
        if HAS_COSMO:
            try:
                cosmo = Cosmology()
                return cosmo.comoving_volume(cosmo.z_from_comoving(d_mpc))
            except Exception:
                pass
        return (4.0 / 3.0) * math.pi * d_mpc ** 3

    def compute(self, n_injections: Optional[int] = None) -> Dict[str, Any]:
        """Run the VT Monte-Carlo."""
        n = n_injections or self.cfg.n_injections
        injections = self.sampler.sample(n)

        found = 0
        snr_records = []
        for params in injections:
            res = self.snr_calc.network_snr(params)
            snr_records.append(res["network_snr"])
            if res["network_snr"] >= self.cfg.network_snr_threshold:
                found += 1

        frac = found / max(n, 1)
        sigma_frac = math.sqrt(frac * (1 - frac) / max(n, 1))
        V_max = self._volume_within(self.cfg.d_max_mpc)

        VT = V_max * frac * self.cfg.T_obs_yr
        sigma_VT = V_max * sigma_frac * self.cfg.T_obs_yr

        snr_records = np.array(snr_records)
        return {
            "backend": self.wfg.backend,
            "n_injections": n,
            "n_found": found,
            "found_fraction": frac,
            "sigma_found_fraction": sigma_frac,
            "V_max_Mpc3": V_max,
            "VT_Mpc3_yr": VT,
            "sigma_VT_Mpc3_yr": sigma_VT,
            "d_max_mpc": self.cfg.d_max_mpc,
            "T_obs_yr": self.cfg.T_obs_yr,
            "snr_median": float(np.median(snr_records)),
            "snr_threshold": self.cfg.network_snr_threshold,
            "honest_note": (
                "VT characterizes PIPELINE+waveform+threshold. "
                + ("REAL lalsimulation waveforms." if self.wfg.backend == "lalsimulation"
                   else "SURROGATE waveform — method demo, NOT a physical sensitivity."),
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Full VT (lalsimulation) — Smoke Test")
    print("=" * 72)
    print(f"  lalsuite installed: {HAS_LAL}")

    cfg = VTFullConfig(n_injections=150, d_max_mpc=600.0)
    calc = FullVTCalculator(cfg)
    print(f"  waveform backend  : {calc.wfg.backend}")

    result = calc.compute()
    print(f"\n  ── Full VT Result ──")
    print(f"    injections      : {result['n_injections']}")
    print(f"    found           : {result['n_found']}")
    print(f"    found fraction  : {result['found_fraction']:.3f} "
          f"± {result['sigma_found_fraction']:.3f}")
    print(f"    V(d_max)        : {result['V_max_Mpc3']:.3e} Mpc³")
    print(f"    VT              : {result['VT_Mpc3_yr']:.3e} Mpc³·yr")
    print(f"    σ_VT            : {result['sigma_VT_Mpc3_yr']:.3e} Mpc³·yr")
    print(f"    median net SNR  : {result['snr_median']:.2f}")
    print(f"    backend         : {result['backend']}")
    print(f"\n    {result['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Antenna patterns simplified (JKS98 wave frame, no full GMST)")
    print("  • Detection criterion is SNR-only; real search adds χ²/coincidence")
    print("  • VT is a PIPELINE property, not a property of Nature")
    print("  • Surrogate backend → method demo only, never a sensitivity claim")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
