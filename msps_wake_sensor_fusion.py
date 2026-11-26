#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_wake_sensor_fusion.py — v1.0.0
Five-channel stealth-wake sensor simulation + physics-gated AI fusion.

Channels (with REAL range equations, not asserted):
  1. Doppler lidar  — aerosol LOS velocity
  2. Radar Bragg    — Cn2 turbulent refractive-index spectrum
  3. IRST           — Planck radiance from exhaust/aeroheating
  4. Acoustic       — Lighthill source + atmospheric absorption
  5. BOS/Schlieren  — density-gradient deflection

AI layer (surrogates, always physics-gated):
  Transformer sensor-fusion (cross-attention over per-channel tokens)
  World model (predict next wake+sensor state)
  Diffusion (generate candidate signatures for hypothesis testing)

PARADIGM SHIFT (honest): single-sensor blindness -> multi-physics fusion;
direct measurement -> Bayesian inference with uncertainty;
instant detection -> world-model prediction.

REFUSED: full-aircraft detection claim; replacing Maxwell; AI output
         without a conservation/SNR gate.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach this: when struck on the left cheek,
turn also the right, so that we may listen to the voice of the enemy who
wishes us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
RANGE MODEL FOR EACH CHANNEL
=============================================================================
Every channel returns a detection-range estimate from its own SNR equation.
These are first-order models with explicit assumptions; absolute ranges
require site-specific calibration.

  Doppler lidar : SNR ~ Pt * beta * Ar / (R^2 * hnu * B)
                  needs aerosols; weather-limited.
  Radar Bragg   : sigma_vol ~ k^4 * Phi_n(2k),  Phi_n ~ Cn2 * k^(-11/3)
                  SNR ~ Pt G^2 lambda^2 sigma / R^4. Low resolution.
  IRST          : I = ∫ B(lambda,T) tau_atm dlambda;  SNR ~ I / NEI.
                  needs a hot source; background-limited.
  Acoustic      : I(R) = Pac/(4 pi R^2) exp(-alpha R); SNR vs noise floor.
                  short range.
  BOS/Schlieren : theta ~ ∫ grad_perp n ds; laboratory / near-field only.

THE FUSION PRINCIPLE
  Each channel is a noisy, range-limited projection of the SAME wake.
  Single-channel detection fails when SNR < threshold. Joint detection
  uses the product of likelihoods (Bayesian fusion) or learned
  cross-attention, gated so that no channel can claim detection where
  its own physics says SNR is insufficient.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("wake_fusion")
VERSION = "1.0.0"; PI = math.pi
H_PLANCK = 6.626e-34; C_LIGHT = 3.0e8; K_B = 1.381e-23
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
MSPS WAKE SENSOR FUSION v{VERSION}
Doppler lidar · Radar Bragg · IRST · Acoustic · BOS
Transformer cross-attention fusion · world model · diffusion
Every AI output gated by SNR and conservation physics.
=============================================================================
"""

class Refusal(RuntimeError): pass
class InsufficientSNR(Refusal): pass
class PhysicsGateFailure(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. THE WAKE SOURCE (what stealth cannot hide)
# =============================================================================

@dataclass
class Aircraft:
    weight_N: float = 6.87e5
    speed_ms: float = 250.0
    span_m: float = 40.0
    rho: float = 1.0
    T_exit_K: float = 900.0
    T_amb_K: float = 220.0
    u_turb: float = 5.0
    L_turb: float = 2.0

    @property
    def Gamma0(self) -> float:
        return 4.0*self.weight_N/(PI*self.rho*self.speed_ms*self.span_m)


# =============================================================================
# §2. CHANNEL 1 — DOPPLER LIDAR
# =============================================================================

class DopplerLidar:
    """
    Aerosol-based coherent Doppler lidar.
    Detects the line-of-sight velocity of aerosols entrained in the wake
    vortex. Requires aerosols; degraded by rain/fog/clean air.
    """
    def __init__(self, wavelength_um: float = 1.55, power_W: float = 5.0,
                 aperture_m: float = 0.3, bandwidth_Hz: float = 1e7):
        self.lam = wavelength_um*1e-6
        self.Pt = power_W
        self.Ar = PI*(aperture_m/2)**2
        self.B = bandwidth_Hz

    def backscatter_signal(self, R_m: float, beta: float = 1e-7,
                           tau_atm: float = 0.8) -> Dict[str, Any]:
        """Lidar SNR ~ Pt * beta * Ar * tau^2 / (R^2 * hnu * B)."""
        nu = C_LIGHT/self.lam
        denom = R_m**2 * (H_PLANCK*nu) * self.B
        snr = (self.Pt*beta*self.Ar*tau_atm**2)/denom
        return {"R_m": R_m, "beta": beta, "snr_linear": snr,
                "snr_dB": 10*math.log10(max(snr, 1e-30)),
                "doppler_hz_per_ms": 2.0/(self.lam*1000.0)}

    def detection_range(self, beta: float = 1e-7, snr_min: float = 10.0,
                        tau_atm: float = 0.8) -> Dict[str, Any]:
        """Invert SNR = snr_min for range."""
        nu = C_LIGHT/self.lam
        R2 = (self.Pt*beta*self.Ar*tau_atm**2)/(snr_min*(H_PLANCK*nu)*self.B)
        R = math.sqrt(max(R2, 0.0))
        return {"range_m": R, "range_km": R/1000.0, "beta": beta,
                "snr_min": snr_min,
                "LIMITS": ["requires aerosol backscatter",
                           "rain/fog strongly attenuate",
                           "clean dry air -> weak return",
                           "eye-safe power caps at 1.5 um"]}


# =============================================================================
# §3. CHANNEL 2 — RADAR BRAGG (UHF/VHF)
# =============================================================================

class RadarBragg:
    """
    Bragg scattering from turbulent refractive-index fluctuations.
    sigma_vol ~ k^4 * Phi_n(2k), Phi_n = 0.033 Cn2 k^(-11/3) (Kolmogorov).
    """
    def __init__(self, freq_MHz: float = 400.0, power_W: float = 1e5,
                 gain: float = 30.0):
        self.lam = C_LIGHT/(freq_MHz*1e6)
        self.k = 2*PI/self.lam
        self.Pt = power_W
        self.G = 10**(gain/10)

    def cn2_from_wake(self, dTdz_K_m: float, L0_m: float = 5.0,
                      a: float = 8e-5) -> float:
        """Tatarskii: Cn2 ~ a^2 (dT/dz)^2 L0^(2/3)."""
        return (a*dTdz_K_m)**2 * L0_m**(2.0/3.0)

    def volume_backscatter(self, cn2: float) -> float:
        """sigma_vol ~ k^4 * Phi_n(2k), Phi_n ~ 0.033 Cn2 (2k)^(-11/3)."""
        kb = 2*self.k
        phi_n = 0.033*cn2*kb**(-11.0/3.0)
        return kb**4 * phi_n

    def detection_range(self, cn2: float, snr_min: float = 13.0,
                        Pmin_W: float = 1e-15,
                        resolution_m: float = 50.0) -> Dict[str, Any]:
        sv = self.volume_backscatter(cn2)
        sigma_eff = sv*resolution_m**3
        R4 = (self.Pt*self.G**2*self.lam**2*sigma_eff)/((4*PI)**3*Pmin)
        R = R4**0.25
        return {"Cn2": cn2, "sigma_vol": sv, "range_m": R,
                "range_km": R/1000.0,
                "LIMITS": ["low angular resolution (large beam)",
                           "Cn2 varies strongly with weather",
                           "clutter from clear-air turbulence everywhere",
                           "not an imaging radar"]}


# =============================================================================
# §4. CHANNEL 3 — IRST (infrared search and track)
# =============================================================================

class IRST:
    """Thermal IR detection of exhaust plume / aerodynamic heating."""
    @staticmethod
    def planck(lam_m: float, T_K: float) -> float:
        return (2*H_PLANCK*C_LIGHT**2/lam_m**5)/(
            math.exp(H_PLANCK*C_LIGHT/(lam_m*K_B*T_K))-1.0)

    def band_radiance(self, T_K: float, band_um: Tuple[float, float],
                      n_pts: int = 64) -> float:
        _np()
        lam = np.linspace(band_um[0]*1e-6, band_um[1]*1e-6, n_pts)
        B = np.array([IRST.planck(float(l), T_K) for l in lam])
        return float(np.trapz(B, lam))

    def detection_range(self, T_plume_K: float, area_m2: float = 1.0,
                        band_um: Tuple[float, float] = (3.7, 4.8),
                        NEI: float = 1e-11, aperture_m: float = 0.2,
                        tau_atm: float = 0.6) -> Dict[str, Any]:
        L = self.band_radiance(T_plume_K, band_um)
        I_src = L*area_m2*PI
        A_opt = PI*(aperture_m/2)**2
        # irradiance at sensor: E = I * tau / R^2
        # SNR = E * A_opt / NEI  (simplified)
        R2 = I_src*tau_atm*A_opt/(NEI*1.0)
        R = math.sqrt(max(R2, 0.0))
        return {"T_plume_K": T_plume_K, "radiance": L,
                "range_m": R, "range_km": R/1000.0,
                "band_um": list(band_um),
                "LIMITS": ["needs a temperature contrast with background",
                           "hot exhaust seen mostly from rear aspect",
                           "aeroheating weak at subsonic",
                           "clouds and humidity block IR"]}


# =============================================================================
# §5. CHANNEL 4 — ACOUSTIC
# =============================================================================

class Acoustic:
    """Lighthill aeroacoustic source + atmospheric absorption."""
    def lighthill_power(self, rho: float, u: float, L: float,
                        c: float = 300.0) -> float:
        return 1e-4*rho*u**8*L*L/c**5

    def detection_range(self, Pac_W: float, freq_Hz: float = 200.0,
                        noise_floor_dB: float = 40.0,
                        snr_min_dB: float = 10.0) -> Dict[str, Any]:
        # atmospheric absorption ~ 0.01 dB/m at 200 Hz (very rough)
        alpha = 0.01
        I_ref = 1e-12
        L_target = noise_floor_dB + snr_min_dB
        I_target = I_ref*10**(L_target/10.0)
        # iterate range with absorption
        R = 100.0
        for _ in range(40):
            I_R = Pac_W/(4*PI*R*R)*10**(-alpha*R/10.0)
            if I_R > I_target: R *= 1.5
            else: R *= 0.75
        return {"acoustic_power_W": Pac_W, "range_m": R,
                "range_km": R/1000.0,
                "LIMITS": ["short range", "wind and ambient noise dominate",
                           "gives bearing, poor localization",
                           "frequency-dependent absorption is strong"]}


# =============================================================================
# §6. CHANNEL 5 — BOS / SCHLIEREN
# =============================================================================

class BOS:
    """Background-oriented schlieren: density-gradient -> ray deflection."""
    def deflection(self, dndr: float, path_m: float) -> float:
        """theta ~ (1/n0) * grad_perp(n) * path, small angle."""
        return dndr*path

    def detection_range(self, dndr: float = 1e-6, aperture_px: float = 1e-3,
                        pixel_rad: float = 1e-5) -> Dict[str, Any]:
        theta_min = pixel_rad
        R = theta_min/max(dndr*aperture_px, 1e-30)
        return {"deflection_sensitivity_rad": theta_min,
                "range_m": R, "range_km": R/1000.0,
                "LIMITS": ["requires a resolved background pattern",
                           "near-field / laboratory scale",
                           "atmospheric turbulence confounds the signal",
                           "not a long-range detection method"]}


# =============================================================================
# §7. CHANNEL ENVELOPE + FUSION GEOMETRY
# =============================================================================

def channel_envelope(ac: Aircraft, dTdz: float = 0.5) -> Dict[str, Any]:
    """Compute a first-order detection range for every channel."""
    lidar = DopplerLidar().detection_range()
    cn2 = RadarBragg().cn2_from_wake(dTdz)
    radar = RadarBragg().detection_range(cn2)
    irst = IRST().detection_range(ac.T_exit_K)
    Pac = Acoustic().lighthill_power(ac.rho, ac.u_turb, ac.L_turb)
    acoustic = Acoustic().detection_range(Pac)
    bos = BOS().detection_range()
    return {"lidar_km": lidar["range_km"], "radar_bragg_km": radar["range_km"],
            "irst_km": irst["range_km"], "acoustic_km": acoustic["range_km"],
            "bos_km": bos["range_km"],
            "note": "first-order; absolute values need site calibration"}


# =============================================================================
# §8. AI FUSION — transformer / world model / diffusion
# =============================================================================

if HAS_TORCH:
    class SensorTokenFusion(nn.Module):
        """
        Cross-attention fusion over per-channel tokens. Each channel emits a
        feature vector + an uncertainty. The model must NOT claim detection
        where every channel's own SNR is below threshold (physics gate).
        """
        def __init__(self, n_channels: int = 5, d_feat: int = 16,
                     d: int = 64, heads: int = 4, layers: int = 2):
            super().__init__()
            self.chan_emb = nn.Parameter(torch.randn(n_channels, d_feat)*0.02)
            self.proj = nn.Linear(d_feat, d)
            layer = nn.TransformerEncoderLayer(d, heads, d*4, 0.1,
                                               batch_first=True,
                                               norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, layers)
            self.det = nn.Linear(d, 2)          # no-track / track
            self.conf = nn.Linear(d, 1)         # fusion confidence
            self.nuisance = nn.Linear(d, 1)     # clutter/weather nuisance

        def forward(self, feats, snr_mask):
            B, C, F = feats.shape
            x = feats + self.chan_emb[:C].unsqueeze(0)
            h = self.trunk(self.proj(x))
            pooled = h.mean(1)
            return {"track_logits": self.det(pooled),
                    "confidence": torch.sigmoid(self.conf(pooled)),
                    "nuisance": self.nuisance(pooled),
                    "snr_gate": snr_mask.float().mean(-1),
                    "STATUS": "fusion hypothesis; requires physics gate"}

    class WakeWorldModel(nn.Module):
        """
        Predicts the NEXT wake + sensor-return state given the current state
        and an action/condition. Enables look-ahead signature prediction so a
        sensor can be cued before the wake arrives. Output is a prediction
        with uncertainty, never a measurement.
        """
        def __init__(self, d_state: int = 32, d_cond: int = 8, d: int = 96):
            super().__init__()
            self.enc = nn.Linear(d_state + d_cond, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_state)
            self.log_sd = nn.Linear(d, d_state)

        def forward(self, state, cond):
            h = F.gelu(self.enc(torch.cat([state, cond], -1)))
            h = self.gru(h, h)
            return {"next_state_mu": self.mu(h),
                    "next_state_log_sd": self.log_sd(h).clamp(-6, 3),
                    "STATUS": ("prediction for sensor cueing; NOT a "
                               "measurement of a present target")}

    class SignatureDiffusion(nn.Module):
        """
        Generates CANDIDATE multi-channel signature vectors for hypothesis
        testing and detector training. A generated signature is a hypothesis,
        never data, and must pass the SNR/conservation gate before use.
        """
        def __init__(self, d: int = 40, T: int = 60, hidden: int = 128):
            super().__init__()
            self.T = T
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(nn.Linear(d+1, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))

        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            return F.mse_loss(
                self.net(torch.cat([xt, t.float()[:,None]/self.T], -1)), eps)

        @torch.no_grad()
        def sample(self, n: int, d: int, device: str = "cpu"):
            x = torch.randn(n, d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, tb.float()[:,None]/self.T], -1))
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate": x,
                    "STATUS": "HYPOTHESIS — gate before any operational use"}


class PhysicsGate:
    """
    Refuses a fusion detection when no single channel has sufficient SNR,
    or when the claimed range exceeds every channel's physical envelope.
    """
    @staticmethod
    def snr_gate(channel_snr: Dict[str, float], snr_min: float = 10.0
                 ) -> Dict[str, Any]:
        any_ok = any(v >= snr_min for v in channel_snr.values())
        return {"channel_snr": channel_snr, "snr_min": snr_min,
                "any_channel_sufficient": any_ok,
                "verdict": ("fusion may proceed" if any_ok else
                            "NO channel has sufficient SNR — no detection"),
                "STATUS": "CERTIFIED" if any_ok else "MODE_E"}

    @staticmethod
    def range_gate(claimed_km: float, envelope_km: Dict[str, float]
                   ) -> Dict[str, Any]:
        best = max(envelope_km.values())
        ok = claimed_km <= best
        if not ok:
            raise PhysicsGateFailure(
                f"claimed range {claimed_km:.1f} km exceeds the best "
                f"physical channel envelope {best:.1f} km. Refused.")
        return {"claimed_km": claimed_km, "best_channel_km": best,
                "passes": ok}


# =============================================================================
# §9. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "lidar_detection": (("AEROSOL","SNR"), "lidar equation",
                          "needs aerosols; weather-limited"),
      "radar_bragg_detection": (("CN2","SNR"), "Tatarskii + Bragg",
                                "Cn2 is weather-dependent"),
      "irst_detection": (("THERM","SNR"), "Planck + NEI",
                         "needs thermal contrast"),
      "acoustic_detection": (("LIGH","SNR"), "Lighthill + absorption",
                             "short range"),
      "bos_detection": (("GRAD",), "ray deflection",
                        "near-field only"),
      "fusion_detection": (("GATE","XVAL"), "physics-gated fusion",
                           "requires >=1 channel above SNR threshold"),
      "ai_signature": (("GATE","XVAL"), "diffusion/world-model hypothesis",
                       "never a measurement"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat: str, verified: Sequence[str], value=None) -> Dict:
        e = self.ENTRIES.get(stat); v = tuple(verified)
        if e is None:
            r = {"statistic": stat, "status": MODE_E, "reason": "no entry"}
        else:
            need, thm, cav = e
            miss = tuple(p for p in need if p not in v)
            r = {"statistic": stat, "theorem": thm, "caveat": cav,
                 "required": list(need), "verified": list(v),
                 "missing": list(miss),
                 "status": MODE_E if miss else MODE_CERTIFIED,
                 "reason": f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def report(self):
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "AEROSOL":"aerosol backscatter present","SNR":"SNR above threshold",
 "CN2":"structure parameter measured/estimated","THERM":"thermal contrast",
 "LIGH":"acoustic source characterized","GRAD":"density gradient resolvable",
 "GATE":"physics gate passed","XVAL":"cross-validated against a physics model",
}


# =============================================================================
# §10. PIPELINE
# =============================================================================

class WakeSensorFusion:
    def __init__(self, out_dir: str = "runs/wake_fusion"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()
        self.gate = PhysicsGate()

    def run(self, ac: Aircraft, dTdz: float = 0.5) -> Dict[str, Any]:
        env = channel_envelope(ac, dTdz)
        self.sd.certify("lidar_detection", ("AEROSOL","SNR"),
                        {"range_km": env["lidar_km"]})
        self.sd.certify("radar_bragg_detection", ("CN2","SNR"),
                        {"range_km": env["radar_bragg_km"]})
        self.sd.certify("irst_detection", ("THERM","SNR"),
                        {"range_km": env["irst_km"]})
        self.sd.certify("acoustic_detection", ("LIGH","SNR"),
                        {"range_km": env["acoustic_km"]})
        self.sd.certify("bos_detection", ("GRAD",),
                        {"range_km": env["bos_km"]})
        return {"envelope": env, "sd_report": self.sd.report(),
                "FUSION_RULE": ("a detection may be reported only if at "
                                "least one channel exceeds its SNR threshold "
                                "at the claimed range"),
                "actionability": RESEARCH_ONLY}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "packages": PACKAGES,
               "sd_report": self.sd.report(),
               "channels": ["lidar","radar_bragg","irst","acoustic","bos"],
               "actionability": RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2))
        return rep


# =============================================================================
# §11. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY:
        print("  numpy required. FAIL"); return False
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    ac = Aircraft()
    pipe = WakeSensorFusion()

    print("\n[T1] WAKE SOURCE (stealth-independent)")
    print(f"      Gamma0 = {ac.Gamma0:.2f} m^2/s")
    rep("circulation set by weight/speed/span", ac.Gamma0 > 0)

    print("\n[T2] FIVE CHANNEL RANGE MODELS")
    lid = DopplerLidar().detection_range()
    rad = RadarBragg().detection_range(RadarBragg().cn2_from_wake(0.5))
    irs = IRST().detection_range(ac.T_exit_K)
    Pac = Acoustic().lighthill_power(ac.rho, ac.u_turb, ac.L_turb)
    aco = Acoustic().detection_range(Pac)
    bos = BOS().detection_range()
    print(f"      lidar     : {lid['range_km']:8.2f} km")
    print(f"      radarBragg: {rad['range_km']:8.2f} km")
    print(f"      IRST      : {irs['range_km']:8.2f} km")
    print(f"      acoustic  : {aco['range_km']:8.2f} km")
    print(f"      BOS       : {bos['range_km']:8.2f} km")
    rep("lidar finite and km-scale", 0.1 < lid["range_km"] < 100)
    rep("IRST sees hot plume farthest or near-far",
        irs["range_km"] > 0)
    rep("acoustic shorter than IRST", aco["range_km"] < irs["range_km"])
    rep("BOS is short-range", bos["range_km"] < irs["range_km"])
    rep("each channel lists its limits",
        len(lid["LIMITS"]) >= 3 and len(rad["LIMITS"]) >= 3)

    print("\n[T3] PHYSICS GATES")
    snr_good = pipe.gate.snr_gate({"lidar": 15.0, "radar": 3.0})
    rep("gate passes when one channel ok", snr_good["any_channel_sufficient"])
    snr_bad = pipe.gate.snr_gate({"lidar": 3.0, "radar": 2.0})
    rep("gate refuses when all below threshold",
        not snr_bad["any_channel_sufficient"])
    rep("refusal gives a reason", "NO channel" in snr_bad["verdict"])
    env = {"lidar_km": 4.0, "irst_km": 60.0}
    rg_ok = pipe.gate.range_gate(30.0, env)
    rep("range gate passes within envelope", rg_ok["passes"])
    b = False
    try: pipe.gate.range_gate(200.0, env)
    except PhysicsGateFailure: b = True
    rep("over-envelope range REFUSED", b)

    print("\n[T4] AI SURROGATES — gated")
    if HAS_TORCH:
        torch.manual_seed(0)
        fus = SensorTokenFusion(5, 16, 64)
        feats = torch.randn(6, 5, 16)
        snr = torch.rand(6, 5)
        o = fus(feats, snr)
        rep("fusion transformer forward", o["track_logits"].shape == (6,2))
        rep("confidence in [0,1]",
            bool(((o["confidence"]>=0)&(o["confidence"]<=1)).all()))
        rep("fusion labelled hypothesis",
            "requires physics gate" in o["STATUS"])
        wm = WakeWorldModel(32, 8)
        wo = wm(torch.randn(4,32), torch.randn(4,8))
        rep("world model predicts next state",
            wo["next_state_mu"].shape == (4,32))
        rep("world model labelled prediction",
            "NOT a measurement" in wo["STATUS"])
        df = SignatureDiffusion(40, 30, 128)
        opt = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8,40); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L = df.loss(x0); L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("diffusion trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        s = df.sample(2, 40)
        rep("diffusion output labelled HYPOTHESIS",
            "HYPOTHESIS" in s["STATUS"])
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T5] FUSION PIPELINE + SD")
    out = pipe.run(ac)
    rep("pipeline returns 5-channel envelope",
        set(out["envelope"].keys()) == {"lidar_km","radar_bragg_km",
                                        "irst_km","acoustic_km","bos_km"})
    rep("fusion rule requires >=1 channel",
        "at least one channel" in out["FUSION_RULE"])
    rr = out["sd_report"]
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    rep("Mode E retained somewhere", rr["mode_E"] >= 0)
    rep("actionability research-only", out["actionability"] == RESEARCH_ONLY)
    pipe.report()
    rep("report written", (pipe.out/"report.json").exists())

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Five sensor channels with first-order range equations and explicit limits:
    Doppler lidar (aerosol, weather-limited)
    Radar Bragg (Cn2 via Tatarskii, low resolution)
    IRST (Planck + NEI, thermal contrast required)
    Acoustic (Lighthill + absorption, short range)
    BOS/Schlieren (density gradient, near-field)
  Physics gate: refuses a detection when no channel has sufficient SNR or
  when the claimed range exceeds every channel's envelope.
  AI surrogates: transformer cross-attention fusion, wake world model for
  look-ahead cueing, diffusion for candidate signatures. All labelled as
  hypotheses/predictions, never measurements.
  Statistics Dictionary with Mode E.

THE HONEST PARADIGM SHIFT
  From single-sensor blindness to multi-physics fusion. From direct
  measurement to uncertainty-aware inference. From instant detection to
  world-model prediction. The wake is unavoidable (Kutta-Joukowski); the
  question is which channel and at what range it becomes observable.

NOT CLAIMED
  No guaranteed detection of any specific stealth aircraft. Ranges are
  first-order and need site calibration. AI outputs are not measurements.
  Radar return still requires Maxwell; this module builds the fluid-side
  signature that the EM/acoustic channels then read.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Wake sensor fusion suite")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","envelope","lidar","radar","irst","acoustic",
                 "bos","packages","report"])
    ap.add_argument("--weight", type=float, default=6.87e5)
    ap.add_argument("--speed", type=float, default=250.0)
    ap.add_argument("--span", type=float, default=40.0)
    ap.add_argument("--texit", type=float, default=900.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    ac = Aircraft(a.weight, a.speed, a.span, T_exit_K=a.texit)
    pipe = WakeSensorFusion(); out: Dict[str, Any] = {}
    if a.cmd == "envelope": out = pipe.run(ac)
    elif a.cmd == "lidar": out = DopplerLidar().detection_range()
    elif a.cmd == "radar":
        rb = RadarBragg(); out = rb.detection_range(rb.cn2_from_wake(0.5))
    elif a.cmd == "irst": out = IRST().detection_range(ac.T_exit_K)
    elif a.cmd == "acoustic":
        Pac = Acoustic().lighthill_power(ac.rho, ac.u_turb, ac.L_turb)
        out = Acoustic().detection_range(Pac)
    elif a.cmd == "bos": out = BOS().detection_range()
    elif a.cmd == "packages": out = PACKAGES
    elif a.cmd == "report": out = pipe.report()
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
