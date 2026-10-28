"""
=============================================================================
SESI GR vs BEYOND-GR — Bayesian model comparison via dynesty evidence
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS COMPARES
------------------
Two waveform hypotheses given the same detector data d:

  M_GR      : waveform phase = standard GR inspiral-merger phasing
  M_BEYOND  : waveform phase = GR + parametrized deviation

The deviation uses the parameterized post-Einsteinian (ppE) idea:
[C: Yunes & Pretorius 2009] add a correction term to the phase,
    Ψ_beyond(f) = Ψ_GR(f) + β_ppE · u^b
where u = (π M f)^{1/3} and b labels the PN order of the deviation.
Two canonical cases implemented:
    b = -1   → a -1PN-type correction (extra long-range force)
    b =  0   → a 0PN-type amplitude/phase shift

This is the SAME structure LIGO-Virgo use in their "testing GR" analyses
(bounded deviations, then check consistency with zero).

EVIDENCE & BAYES FACTOR
-----------------------
For each model, dynesty computes  Z = ∫ L(d|θ,M) π(θ|M) dθ.
The Bayes factor  B = Z_GR / Z_beyond  favors whichever model the data
support, WITH an automatic Occam penalty for the extra β_ppE dimension
in M_BEYOND.

LABELS: [C] ppE + dynesty  [D] implementation here  [O] open

HONEST SCOPE (critical)
-----------------------
• The Bayes factor is PRIOR-VOLUME sensitive. Widening the β_ppE prior
  penalizes M_BEYOND even if the data are indifferent. Always report the
  prior range used.
• This compares WAVEFORM MODELS, not fundamental theories. A non-zero
  β_ppE posterior is a consistency check, not a discovery of new physics.
• Without lalsuite, waveforms are analytic surrogates; results are a
  METHOD DEMO, not a publication-grade GR test.
• Detection vs parameter estimation: we assume the event is detected;
  the comparison is about its WAVEFORM, not about whether it exists.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.gr_beyond")

try:
    import dynesty
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

# Reuse the evidence machinery built earlier
try:
    from sesi_evidence_comparison import (
        EvidenceConfig, ModelEvidenceCalculator, BayesFactorComparator,
    )
    HAS_EVIDENCE = True
except ImportError:
    HAS_EVIDENCE = False

__all__ = ["BeyondGRConfig", "ParametrizedWaveform", "GRvsBeyondGRComparator"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BeyondGRConfig:
    # Source (fixed for the comparison; in production these are sampled too)
    Mc_solar: float = 30.0            # chirp mass
    distance_mpc: float = 400.0
    f_min: float = 20.0
    f_max: float = 512.0
    n_freq: int = 256

    # Noise
    noise_psd: float = 1e-46           # white-noise floor (toy)

    # beyond-GR deviation
    b_pn_order: float = -1.0           # -1PN deviation by default
    beta_prior_abs: float = 10.0       # β_ppE ∈ [-10, 10]; MUST be reported

    # dynesty
    nlive: int = 400
    dlogz: float = 0.05
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# [C/D] PARAMETRIZED WAVEFORM
# ═══════════════════════════════════════════════════════════════════════

class ParametrizedWaveform:
    """
    [D] Frequency-domain inspiral waveform with an optional ppE deviation.

    The GR part is the leading-order (Newtonian) chirp; the beyond-GR part
    adds β_ppE · u^b to the phase. Both are clearly-labeled analytic
    surrogates when lalsuite is unavailable.
    """

    MSUN_SEC = 4.925490947e-6

    def __init__(self, cfg: BeyondGRConfig):
        self.cfg = cfg
        self.freqs = np.linspace(cfg.f_min, cfg.f_max, cfg.n_freq)

    def _gr_phase(self, Mc_sec: float) -> np.ndarray:
        """Leading-order GR inspiral phase Ψ_GR(f) ∝ f^{-5/3}."""
        f = self.freqs
        # stationary-phase leading term
        return -(3.0 / 128.0) * (math.pi * Mc_sec * f) ** (-5.0 / 3.0)

    def waveform(self, beta_ppE: float = 0.0,
                 b: Optional[float] = None) -> np.ndarray:
        """
        Return the complex frequency-domain strain h̃(f).

        beta_ppE = 0 recovers pure GR.
        """
        cfg = self.cfg
        Mc_sec = cfg.Mc_solar * self.MSUN_SEC
        u = (math.pi * Mc_sec * self.freqs) ** (1.0 / 3.0)

        phase = self._gr_phase(Mc_sec)
        if beta_ppE != 0.0:
            b_val = cfg.b_pn_order if b is None else b
            phase = phase + beta_ppE * u ** b_val

        # Newtonian amplitude ∝ Mc^{5/6} f^{-7/6} / d
        amp = (Mc_sec ** (5.0 / 6.0) * self.freqs ** (-7.0 / 6.0)
               / max(cfg.distance_mpc, 1.0)) * 1e-21
        return amp * np.exp(1j * phase)

    def make_data(self, beta_true: float = 0.0, snr_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate (possibly injected) data d = h(β_true) + noise.

        beta_true = 0  → data consistent with pure GR.
        beta_true ≠ 0  → data contains a real deviation (recovery test).
        """
        rng = rng or np.random.default_rng(self.cfg.seed)
        h = self.waveform(beta_ppE=beta_true)
        noise_std = math.sqrt(self.cfg.noise_psd * (self.freqs[1] - self.freqs[0]))
        noise = (rng.normal(0, 1, len(h)) + 1j * rng.normal(0, 1, len(h))) * noise_std
        return h * snr_scale + noise


# ═══════════════════════════════════════════════════════════════════════
# [C/D] GR vs BEYOND-GR COMPARATOR
# ═══════════════════════════════════════════════════════════════════════

class GRvsBeyondGRComparator:
    """
    [C/D] Compute evidences for M_GR and M_BEYOND and their Bayes factor.

    Parameters sampled:
      M_GR     : none extra (Mc fixed) → trivial evidence integral
      M_BEYOND : β_ppE ∈ [-β_max, +β_max] (the deviation parameter)

    The likelihood is Gaussian in the frequency domain:
      log L = -½ Σ |d - h(β)|² / S_n
    """

    def __init__(self, cfg: BeyondGRConfig):
        self.cfg = cfg
        self.wf = ParametrizedWaveform(cfg)
        self.psd = np.full(cfg.n_freq, cfg.noise_psd)

    # ── Gaussian log-likelihood ────────────────────────────────────────
    def log_likelihood(self, d: np.ndarray, beta_ppE: float) -> float:
        h = self.wf.waveform(beta_ppE=beta_ppE)
        resid = d - h
        df = self.cfg.freqs[1] - self.cfg.freqs[0]
        # 4 Re∫ ... df, summed over positive frequencies
        return float(-2.0 * df * np.sum(np.abs(resid) ** 2 / self.psd))

    def build_models(self, d: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Return (model_GR, model_beyond) dicts for the BayesFactorComparator.

        M_GR is a point model (no free parameter) → evidence = L(d|GR) × 1.
        M_BEYOND integrates over β_ppE with a uniform prior.
        """
        bmax = self.cfg.beta_prior_abs

        # GR: delta model. We expose it as a 1-D "parameter" that is fixed
        # by the prior transform collapsing to β=0.
        def loglike_gr(theta):
            return self.log_likelihood(d, beta_ppE=0.0)
        def pt_gr(u):
            return np.array([0.0])  # collapsed to GR point

        # beyond-GR: uniform in [-bmax, +bmax]
        def loglike_beyond(theta):
            return self.log_likelihood(d, beta_ppE=float(theta[0]))
        def pt_beyond(u):
            return np.array([-bmax + u[0] * 2.0 * bmax])

        model_gr = {"log_likelihood": loglike_gr, "prior_transform": pt_gr,
                    "label": "GR"}
        model_beyond = {"log_likelihood": loglike_beyond,
                        "prior_transform": pt_beyond,
                        "label": f"beyond_GR(beta_prior=±{bmax})"}
        return model_gr, model_beyond

    def compare(self, beta_injected: float = 0.0,
                snr_scale: float = 1.0) -> Dict[str, Any]:
        """
        Full comparison. Injects data with beta_injected, then compares.

        Returns the Bayes factor + interpretation + prior-range reminder.
        """
        if not HAS_EVIDENCE:
            return {"error": "sesi_evidence_comparison not importable"}
        if not HAS_DYNESTY:
            return {"error": "dynesty not installed (pip install dynesty)"}

        rng = np.random.default_rng(self.cfg.seed)
        d = self.wf.make_data(beta_true=beta_injected,
                              snr_scale=snr_scale, rng=rng)

        model_gr, model_beyond = self.build_models(d)

        cfg_ev = EvidenceConfig(nlive=self.cfg.nlive, dlogz=self.cfg.dlogz,
                                ndim=1, seed=self.cfg.seed)
        comparator = BayesFactorComparator(cfg_ev)
        result = comparator.compare(model_gr, model_beyond)

        result["injected_beta"] = beta_injected
        result["beta_prior_range"] = [-self.cfg.beta_prior_abs,
                                      self.cfg.beta_prior_abs]
        result["pn_order"] = self.cfg.b_pn_order
        result["honest_note"] = (
            "Bayes factor is prior-volume sensitive: the beyond-GR model is "
            "penalized for its extra β dimension (Occam factor). A preference "
            "for GR here is a WAVEFORM-MODEL comparison, not a proof that "
            "gravity is GR or that no deviation exists."
        )
        return result


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI GR vs beyond-GR — Smoke Test")
    print("=" * 72)
    print(f"  dynesty : {HAS_DYNESTY}")
    print(f"  evidence module : {HAS_EVIDENCE}")

    if not (HAS_DYNESTY and HAS_EVIDENCE):
        print("  [SKIP] pip install dynesty")
        return

    cfg = BeyondGRConfig(nlive=250, beta_prior_abs=10.0)
    comp = GRvsBeyondGRComparator(cfg)

    # Case A: data generated by GR (β=0) → expect weak-to-moderate GR favor
    print("\n  ── Case A: injected β = 0 (pure GR data) ──")
    res_a = comp.compare(beta_injected=0.0, snr_scale=12.0)
    if "error" not in res_a:
        print(f"    ln B (GR/beyond) : {res_a['ln_bayes_factor']:.3f}")
        print(f"    interpretation   : {res_a['interpretation']}")
    else:
        print(f"    error: {res_a['error']}")

    # Case B: data with a real deviation (β=3) → beyond-GR should be favored
    print("\n  ── Case B: injected β = 3.0 (real deviation) ──")
    res_b = comp.compare(beta_injected=3.0, snr_scale=12.0)
    if "error" not in res_b:
        print(f"    ln B (GR/beyond) : {res_b['ln_bayes_factor']:.3f}")
        print(f"    interpretation   : {res_b['interpretation']}")
    else:
        print(f"    error: {res_b['error']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • β_ppE prior range MUST be reported (prior-volume / Occam effect)")
    print("  • Waveforms are analytic surrogates without lalsuite")
    print("  • Favors GR ≠ proves GR; disfavors GR ≠ discovers new physics")
    print("  • This is a waveform-model test, not a fundamental-theory verdict")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
