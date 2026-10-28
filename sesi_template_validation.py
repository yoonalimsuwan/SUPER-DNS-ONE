"""
=============================================================================
SESI TEMPLATE VALIDATION — Software Injections + Real-Data Checks
Verify that waveform templates recover known signals before trusting them
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHY THIS EXISTS
---------------
A template bank is only trustworthy if it RECOVERS signals that are known
to be present. The standard method (used by LIGO/Virgo pipelines) is:

  [V1] SOFTWARE INJECTION — add a known waveform h(θ_true) to noise,
       run the search, and check that θ_true is recovered with the
       expected SNR.
  [V2] SENSITIVITY CURVE — inject across a range of SNRs and measure
       detection efficiency (found fraction vs injected SNR).
  [V3] REAL-DATA CHECK — validate against GWOSC strain when available;
       otherwise clearly report synthetic-only status.

LABELS: [C] standard injection methodology (LIGO/Virgo practice)
        [D] derived implementation here   [O] open / needs real data

HONEST SCOPE
------------
• Synthetic noise (Gaussian, stationary) is NOT real detector noise.
  Results on synthetic noise are necessary-but-insufficient.
• Real-data validation requires the GWOSC connector + network access,
  and ideally a known event (e.g. GW150914) as ground truth.
• Mismatch and efficiency numbers here characterize the TEMPLATE + SEARCH
  implementation, not a claim about Nature.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.template_validation")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional connectors for real data
try:
    from sesi_ligo_virgo_connector import LIGOVirgoHub, LIGOConfig
    HAS_GWOSC = True
except ImportError:
    HAS_GWOSC = False

__all__ = [
    "TemplateValidationConfig",
    "WaveformGenerator",
    "SoftwareInjector",
    "TemplateValidator",
    "InjectionCampaign",
    "RealDataValidator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TemplateValidationConfig:
    sample_rate: float = 4096.0
    duration: float = 8.0
    f_min: float = 20.0
    noise_sigma: float = 1e-21          # synthetic Gaussian noise level
    snr_threshold: float = 5.5          # detection threshold
    # Injection campaign
    snr_grid: List[float] = field(default_factory=lambda:
                                  [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    n_trials_per_snr: int = 20
    # Mismatch
    mismatch_tolerance: float = 0.03    # typical search threshold
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# WAVEFORM GENERATOR (toy, clearly labeled)
# ═══════════════════════════════════════════════════════════════════════

class WaveformGenerator:
    """
    [D-TOY] Generate simple chirp waveforms for injection/recovery tests.

    This is a leading-order inspiral chirp — NOT a calibrated IMRPhenom/SEOBNR
    waveform. For real template-bank validation, swap in lalsimulation
    (see sesi_waveform_templates.generate_lalsimulation).
    """

    def __init__(self, cfg: TemplateValidationConfig):
        self.cfg = cfg
        self.msun_sec = 4.925490947e-6

    def chirp(self, m1: float, m2: float, distance_mpc: float = 400.0,
              phase0: float = 0.0) -> np.ndarray:
        """Return a time-domain chirp h(t) of length duration*sample_rate."""
        fs = self.cfg.sample_rate
        n = int(fs * self.cfg.duration)
        t = np.arange(n) / fs

        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        mc_sec = mc * self.msun_sec

        # coalescence placed near the end
        t_c = self.cfg.duration * 0.7
        tau = np.maximum(t_c - t, 1e-4)

        f_gw = (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** 0.375 * mc_sec ** (-0.625)
        f_gw = np.clip(f_gw, self.cfg.f_min, fs / 2 * 0.9)
        phase = 2.0 * np.pi * np.cumsum(f_gw) / fs + phase0

        amp = mc ** (5.0 / 6.0) * f_gw ** (2.0 / 3.0) / max(distance_mpc, 1.0)
        amp = amp / (amp.max() + 1e-30)

        return amp * np.cos(phase)

    def normalize_to_snr(self, h: np.ndarray, target_snr: float,
                         noise_sigma: float) -> np.ndarray:
        """Scale h so that its optimal matched-filter SNR equals target_snr."""
        # Optimal SNR for white noise ∝ ||h|| / sigma
        norm = np.sqrt(np.sum(h ** 2))
        if norm < 1e-30:
            return h
        # SNR = sqrt(sum(h^2)) / sigma  → scale to hit target
        scale = target_snr * noise_sigma / norm
        return h * scale


# ═══════════════════════════════════════════════════════════════════════
# SOFTWARE INJECTOR
# ═══════════════════════════════════════════════════════════════════════

class SoftwareInjector:
    """
    [C] Create noisy strain with a known signal injected.

    Returns both the noisy data and the injection parameters for later
    recovery comparison.
    """

    def __init__(self, cfg: TemplateValidationConfig):
        self.cfg = cfg
        self.wfg = WaveformGenerator(cfg)
        self.rng = np.random.default_rng(cfg.seed)

    def make_noise(self) -> np.ndarray:
        n = int(self.cfg.sample_rate * self.cfg.duration)
        return self.rng.normal(0, self.cfg.noise_sigma, n)

    def inject(self, m1: float, m2: float, target_snr: float) -> Dict[str, Any]:
        """Inject a chirp at a target SNR into fresh Gaussian noise."""
        h = self.wfg.chirp(m1, m2)
        h = self.wfg.normalize_to_snr(h, target_snr, self.cfg.noise_sigma)
        noise = self.make_noise()
        data = noise + h
        return {
            "data": data,
            "clean_signal": h,
            "noise": noise,
            "params": {"m1": m1, "m2": m2, "target_snr": target_snr},
        }


# ═══════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class TemplateValidator:
    """
    [D] Recover an injected signal with matched filtering and report
    whether the template bank + search recovers it.

    Metrics:
      • recovered_snr vs injected_snr
      • time-of-arrival offset
      • mismatch between injected and best-matching template
    """

    def __init__(self, cfg: TemplateValidationConfig):
        self.cfg = cfg
        self.wfg = WaveformGenerator(cfg)

    def matched_filter(self, data: np.ndarray, template: np.ndarray) -> Tuple[float, int]:
        """Return (optimal SNR, best time index) of template in data."""
        n = len(data)
        df = self.cfg.sample_rate / n
        d_f = np.fft.rfft(data)
        h_f = np.fft.rfft(template, n=n)

        # whiten by template norm (white-noise assumption)
        h_norm = np.sqrt(np.sum(template ** 2)) + 1e-30
        h_f = h_f / h_norm

        # matched-filter time series
        mf = np.fft.irfft(d_f * np.conj(h_f), n=n) * df * self.cfg.sample_rate
        snr_series = np.abs(mf) / (self.cfg.noise_sigma * self.cfg.sample_rate + 1e-30)

        best_idx = int(np.argmax(snr_series))
        return float(snr_series[best_idx]), best_idx

    def validate_single(self, injection: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one injection: recover it and compare."""
        data = injection["data"]
        clean = injection["clean_signal"]
        params = injection["params"]

        # Build a bank around the true parameters (small grid)
        m1_true, m2_true = params["m1"], params["m2"]
        candidates = []
        for dm1 in (-2.0, 0.0, 2.0):
            for dm2 in (-2.0, 0.0, 2.0):
                m1 = max(m1_true + dm1, 2.0)
                m2 = min(m1, max(m2_true + dm2, 2.0))
                tmpl = self.wfg.chirp(m1, m2)
                tmpl = self.wfg.normalize_to_snr(tmpl, 1.0, self.cfg.noise_sigma)
                candidates.append({"m1": m1, "m2": m2, "template": tmpl})

        best_snr = 0.0
        best = None
        for cand in candidates:
            snr, idx = self.matched_filter(data, cand["template"])
            cand["recovered_snr"] = snr
            cand["toa_idx"] = idx
            if snr > best_snr:
                best_snr = snr
                best = cand

        # Mismatch between injected signal and best template (both normalized)
        mismatch = self._mismatch(clean, best["template"])

        detected = best_snr >= self.cfg.snr_threshold

        return {
            "injected_snr": params["target_snr"],
            "recovered_snr": best_snr,
            "snr_ratio": best_snr / max(params["target_snr"], 1e-9),
            "mismatch": mismatch,
            "mismatch_within_tolerance": mismatch <= self.cfg.mismatch_tolerance,
            "detected": bool(detected),
            "recovered_m1": best["m1"],
            "recovered_m2": best["m2"],
            "true_m1": m1_true,
            "true_m2": m2_true,
        }

    @staticmethod
    def _mismatch(a: np.ndarray, b: np.ndarray) -> float:
        """Normalized mismatch = 1 - |<a,b>|/(||a|| ||b||)."""
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        num = np.abs(np.sum(a * b))
        den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-30
        return float(1.0 - num / den)


# ═══════════════════════════════════════════════════════════════════════
# INJECTION CAMPAIGN → SENSITIVITY CURVE
# ═══════════════════════════════════════════════════════════════════════

class InjectionCampaign:
    """
    [C] Run many injections across an SNR grid and build a sensitivity
    (detection-efficiency) curve: found fraction vs injected SNR.
    """

    def __init__(self, cfg: TemplateValidationConfig):
        self.cfg = cfg
        self.injector = SoftwareInjector(cfg)
        self.validator = TemplateValidator(cfg)

    def run(self, m1: float = 30.0, m2: float = 25.0) -> Dict[str, Any]:
        results = []
        for snr in self.cfg.snr_grid:
            n_found = 0
            mismatches = []
            for trial in range(self.cfg.n_trials_per_snr):
                inj = self.injector.inject(m1, m2, snr)
                val = self.validator.validate_single(inj)
                if val["detected"]:
                    n_found += 1
                mismatches.append(val["mismatch"])
            results.append({
                "injected_snr": snr,
                "n_trials": self.cfg.n_trials_per_snr,
                "n_found": n_found,
                "efficiency": n_found / self.cfg.n_trials_per_snr,
                "mean_mismatch": float(np.mean(mismatches)),
            })

        # 50% efficiency point (a common sensitivity summary)
        snr50 = self._snr_at_efficiency(results, 0.5)

        return {
            "curve": results,
            "snr50": snr50,
            "note": "Synthetic-Gaussian-noise sensitivity; NOT real detector sensitivity.",
        }

    @staticmethod
    def _snr_at_efficiency(results: List[Dict], target: float) -> Optional[float]:
        """Interpolate the injected SNR at a target detection efficiency."""
        snrs = [r["injected_snr"] for r in results]
        effs = [r["efficiency"] for r in results]
        for i in range(len(effs) - 1):
            if effs[i] <= target <= effs[i + 1]:
                # linear interpolation
                frac = (target - effs[i]) / max(effs[i + 1] - effs[i], 1e-12)
                return snrs[i] + frac * (snrs[i + 1] - snrs[i])
        return None

    @staticmethod
    def report(campaign_result: Dict[str, Any]) -> str:
        lines = ["=" * 60, "INJECTION CAMPAIGN — SENSITIVITY CURVE", "=" * 60]
        lines.append(f"{'inj_SNR':>8} {'found':>6} {'eff':>6} {'mm':>8}")
        for r in campaign_result["curve"]:
            lines.append(f"{r['injected_snr']:>8.1f} {r['n_found']:>6d} "
                         f"{r['efficiency']:>6.2f} {r['mean_mismatch']:>8.4f}")
        s50 = campaign_result["snr50"]
        lines.append("-" * 60)
        lines.append(f"SNR at 50% efficiency: {s50:.2f}" if s50 else
                     "SNR at 50% efficiency: not reached in grid")
        lines.append(campaign_result["note"])
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# REAL-DATA VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class RealDataValidator:
    """
    [O] Validate templates against REAL GWOSC strain, when available.

    Strategy:
      • Fetch strain for a known event (e.g. GW150914) via the connector.
      • Inject nothing (the real signal is already present) and check that
        the template bank produces a high-SNR match near the known time.
      • If the strain is synthetic (offline fallback), say so explicitly.
    """

    def __init__(self, cfg: TemplateValidationConfig):
        self.cfg = cfg
        self.hub = None
        if HAS_GWOSC:
            self.hub = LIGOVirgoHub(LIGOConfig())

    def validate_known_event(self, event_name: str = "GW150914",
                             detector: str = "H1") -> Dict[str, Any]:
        if self.hub is None:
            return {
                "status": "SKIPPED",
                "reason": "sesi_ligo_virgo_connector not available",
                "real_data": False,
            }

        strain_data = self.hub.get_strain(event_name, detector=detector)
        if strain_data is None:
            return {"status": "SKIPPED", "reason": "strain fetch returned None",
                    "real_data": False}

        is_synthetic = strain_data.get("synthetic", True)
        strain = np.array(strain_data["strain"])

        # Run matched filter with a BBH-like template
        wfg = WaveformGenerator(self.cfg)
        template = wfg.chirp(36.0, 29.0)  # GW150914-like masses
        template = wfg.normalize_to_snr(template, 1.0, strain.std() + 1e-30)

        validator = TemplateValidator(self.cfg)
        snr, toa = validator.matched_filter(strain, template)

        return {
            "status": "RUN",
            "event": event_name,
            "detector": detector,
            "real_data": not is_synthetic,
            "recovered_snr": float(snr),
            "toa_index": int(toa),
            "note": ("REAL GWOSC strain" if not is_synthetic else
                     "SYNTHETIC fallback — not a real-data validation"),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Template Validation — Smoke Test")
    print("=" * 72)

    cfg = TemplateValidationConfig(
        duration=2.0, sample_rate=2048.0,
        snr_grid=[4.0, 8.0, 12.0], n_trials_per_snr=5,
    )

    # ── Single injection + recovery ───────────────────────────────────
    print("\n  ── [V1] Single software injection ──")
    injector = SoftwareInjector(cfg)
    validator = TemplateValidator(cfg)

    inj = injector.inject(m1=30.0, m2=25.0, target_snr=12.0)
    val = validator.validate_single(inj)
    print(f"    injected SNR   : {val['injected_snr']:.1f}")
    print(f"    recovered SNR  : {val['recovered_snr']:.2f}")
    print(f"    SNR ratio      : {val['snr_ratio']:.2f}")
    print(f"    mismatch       : {val['mismatch']:.4f}")
    print(f"    detected       : {val['detected']}")

    # ── Injection campaign / sensitivity curve ───────────────────────
    print("\n  ── [V2] Injection campaign → sensitivity curve ──")
    campaign = InjectionCampaign(cfg)
    campaign_result = campaign.run(m1=30.0, m2=25.0)
    print(InjectionCampaign.report(campaign_result))

    # ── Real-data check ──────────────────────────────────────────────
    print("\n  ── [V3] Real-data validation (GWOSC) ──")
    real_validator = RealDataValidator(cfg)
    real_result = real_validator.validate_known_event("GW150914")
    if real_result["status"] == "SKIPPED":
        print(f"    SKIPPED: {real_result['reason']}")
    else:
        print(f"    event          : {real_result['event']}")
        print(f"    real_data      : {real_result['real_data']}")
        print(f"    recovered SNR  : {real_result['recovered_snr']:.2f}")
        print(f"    note           : {real_result['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Synthetic Gaussian noise ≠ real detector noise.")
    print("  • Toy chirp ≠ calibrated IMRPhenom/SEOBNR (swap in lalsimulation).")
    print("  • Real-data validation requires GWOSC access; otherwise clearly")
    print("    reported as synthetic fallback.")
    print("  • These numbers characterize the TEMPLATE + SEARCH, not Nature.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
