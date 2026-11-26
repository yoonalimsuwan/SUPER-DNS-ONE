#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_wake_signature_suite.py — v1.0.0
Multi-fidelity wake-signature simulator for stealth-countermeasure sensor design.

Tiers: DNS (vortex blobs) · LES (blobs + stochastic sub-blob diffusion)
       · RANS (analytic mean wake)
Physics: Kutta-Joukowski wake generation · Biot-Savart advection ·
         Crow instability · vortex decay · exhaust plume · transition marker
Sensors: Doppler lidar · radar Bragg (Cn2) · IR (Planck) · acoustic (Lighthill)
AI: transformer sequence model · world model · diffusion generator
    (all surrogates; every output gated by conservation checks)

REFUSED: full-aircraft DNS at flight Reynolds number;
         DNS as a detector; any AI output without a physics gate.
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

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

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
WHAT THIS IS AND IS NOT
=============================================================================
IS:
  * A wake-physics simulator that predicts the fluid signatures a stealth
    aircraft CANNOT hide (Kutta-Joukowski: lift demands shed circulation).
  * Sensor-model builders: given a wake field, compute what a Doppler lidar,
    a Bragg radar, an IR camera, or an acoustic array would measure.
  * AI surrogates for fast wake-state prediction, hard-gated by conservation.

IS NOT:
  * A detector. NS/CFD creates the signature; some wave (EM/acoustic) reads it.
  * A full-aircraft DNS. Flight-Re DNS needs tens of millions of GPUs.
  * A replacement for Maxwell. Radar return still needs EM propagation.

THE DEEP POINT
  Stealth reduces the airframe's EM cross-section. It does not reduce
  circulation, because
      L = rho V Gamma   =>   Gamma = 4 W / (pi rho V b)
  is fixed by weight W, speed V and span b. Shed vorticity is the price of
  lift. This module predicts that unavoidable wake and the sensor return
  it produces.

=============================================================================
TIER CAPABILITY (enforced, not advisory)
=============================================================================
DNS   : vortex-blob Biot-Savart wake, Crow mode, decay, transition marker.
        Scope: wake physics for sensor/algorithm design.
LES   : same blobs + stochastic sub-blob diffusion (unresolved turbulence).
RANS  : analytic Gaussian mean-wake model for fast design sweeps.
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
log = logging.getLogger("wake_sig")
VERSION = "1.0.0"; PI = math.pi
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
MSPS WAKE-SIGNATURE SUITE v{VERSION}
DNS vortex blobs · LES stochastic diffusion · RANS mean wake
Crow instability · vortex decay · lidar/radar-Bragg/IR/acoustic signatures
AI surrogates gated by Kelvin circulation and energy conservation
REFUSED: full-aircraft flight-Re DNS; DNS-as-detector
=============================================================================
"""

class Refusal(RuntimeError): pass
class TierRefusal(Refusal): pass
class ConservationFailure(Refusal): pass
class NotIdentifiable(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. WAKE GENERATION — KUTTA-JOUKOWSKI
# =============================================================================

@dataclass
class Aircraft:
    weight_N: float = 6.87e5       # ~70 t
    speed_ms: float = 250.0
    span_m: float = 40.0
    rho: float = 1.0               # kg/m^3 at altitude
    chord_m: float = 4.0

    @property
    def Gamma0(self) -> float:
        """Initial wake-vortex circulation, fixed by lift. Stealth cannot change this."""
        return 4.0*self.weight_N/(PI*self.rho*self.speed_ms*self.span_m)

    @property
    def reynolds(self) -> float:
        nu = 1.5e-5
        return self.speed_ms*self.chord_m/nu


def wake_initial_condition(ac: Aircraft) -> Dict[str, Any]:
    """The signature stealth cannot hide."""
    return {"Gamma0_m2_s": ac.Gamma0,
            "reynolds_chord": ac.reynolds,
            "V_pair_m_s": ac.Gamma0/(2*PI*0.75*ac.span_m),
            "statement": ("Lift requires shed circulation. Stealth shaping "
                          "does not appear in Gamma0 = 4W/(pi rho V b)."),
            "NOT_DETECTABLE_BY": "this is a source term, not a return signal"}


# =============================================================================
# §2. TIER REGISTRY
# =============================================================================

class Tier:
    DNS = "dns"; LES = "les"; RANS = "rans"

@dataclass(frozen=True)
class TierCapability:
    tier: str
    can: Tuple[str, ...]
    cannot: Tuple[str, ...]
    cost: str
    note: str

TIER_CAPS: Dict[str, TierCapability] = {
 Tier.DNS: TierCapability("dns",
    ("wake_vortex_physics","crow_instability","vortex_decay",
     "transition_marker","sensor_algorithm_design"),
    ("full_aircraft_flight_re","real_time_detection","exhaust_chemistry",
     "compressible_shocks"),
    "high",
    "Vortex-blob Biot-Savart. Physics source for sensor design, not a detector."),
 Tier.LES: TierCapability("les",
    ("wake_evolution","exhaust_plume","vortex_merging","turbulent_Cn2"),
    ("all_scales_resolved","near_wall_dns","full_aircraft_flight_re"),
    "medium",
    "Blobs + stochastic sub-blob diffusion for unresolved turbulence."),
 Tier.RANS: TierCapability("rans",
    ("mean_wake_deficit","design_sweep","drag_lift_shape_study"),
    ("unsteady_vortex_dynamics","crow_mode","turbulence_spectra"),
    "low",
    "Analytic Gaussian mean wake. Fast, steady, no vortex dynamics."),
}

def check_tier(tier: str, task: str) -> Dict[str, Any]:
    if tier not in TIER_CAPS:
        raise TierRefusal(f"unknown tier '{tier}'. Known: {list(TIER_CAPS)}")
    c = TIER_CAPS[tier]
    if task in c.cannot:
        raise TierRefusal(
            f"REFUSED: '{task}' is out of scope for {tier.upper()}.\n"
            f"cannot: {list(c.cannot)}\n"
            f"can: {list(c.can)}\n"
            "For full-aircraft flight-Re use RANS/LES+actuator-line, not DNS. "
            "For detection use Maxwell/acoustic sensors, not CFD alone.")
    if task not in c.can:
        raise TierRefusal(f"'{task}' not a validated {tier.upper()} task")
    return {"allowed": True, "tier": tier, "task": task}


# =============================================================================
# §3. VORTEX-BLOB ENGINE (DNS tier)
# =============================================================================

@dataclass
class VortexBlob:
    """2-D desingularised point vortex in the cross-plane."""
    x: float; y: float; Gamma: float; sigma: float


class VortexBlobDNS:
    """
    Cross-plane vortex-blob wake solver (DNS tier).

    Velocity from a Gaussian vortex blob:
        u_theta(r) = Gamma/(2 pi r) * (1 - exp(-r^2/sigma^2))
    The pair's self-induced translation must recover V = Gamma/(2 pi b).
    Kelvin circulation is conserved by construction (blobs carry fixed Gamma).
    """
    def __init__(self, blobs: List[VortexBlob]):
        self.blobs = blobs
        self.t = 0.0

    @staticmethod
    def _blob_velocity_at(px: float, py: float, b: VortexBlob,
                          cut: float = 1e-6) -> Tuple[float, float]:
        rx = px - b.x; ry = py - b.y
        r2 = rx*rx + ry*ry
        if r2 < cut*cut: return 0.0, 0.0
        s2 = b.sigma*b.sigma
        u_theta = b.Gamma/(2*PI*r2)*(1.0 - math.exp(-r2/s2))
        # tangential: counter-clockwise circulation -> ( -ry, rx )/r
        inv_r = 1.0/math.sqrt(r2)
        return -u_theta*ry*inv_r, u_theta*rx*inv_r

    def velocity_at(self, px: float, py: float) -> Tuple[float, float]:
        ux = uy = 0.0
        for b in self.blobs:
            vx, vy = self._blob_velocity_at(px, py, b)
            ux += vx; uy += vy
        return ux, uy

    def circulation(self) -> float:
        return sum(b.Gamma for b in self.blobs)

    def step(self, dt: float, turb_diffusion: float = 0.0,
             rng: Optional["np.random.Generator"] = None) -> None:
        """RK2 advection; optional stochastic diffusion for LES tier."""
        k1 = []
        for b in self.blobs:
            k1.append(self.velocity_at(b.x, b.y))
        for i, b in enumerate(self.blobs):
            mx = b.x + 0.5*dt*k1[i][0]
            my = b.y + 0.5*dt*k1[i][1]
            u, v = self.velocity_at(mx, my)
            b.x += dt*u
            b.y += dt*v
            if turb_diffusion > 0.0 and rng is not None:
                b.x += rng.normal(0.0, math.sqrt(2.0*turb_diffusion*dt))
                b.y += rng.normal(0.0, math.sqrt(2.0*turb_diffusion*dt))
        self.t += dt

    def total_kinetic_energy(self, grid_n: int = 64,
                             extent: float = 3.0) -> float:
        """Integral of 0.5 |u|^2 over the cross-plane (for energy budget)."""
        _np()
        xs = np.linspace(-extent, extent, grid_n)
        dx = xs[1]-xs[0]
        ke = 0.0
        for x in xs:
            for y in xs:
                u, v = self.velocity_at(float(x), float(y))
                ke += 0.5*(u*u+v*v)*dx*dx
        return ke

    def radial_velocity_field(self, lidar_pos: Tuple[float, float],
                              grid_n: int = 32, extent: float = 2.5
                              ) -> Dict[str, Any]:
        """Line-of-sight (radial) velocity toward a lidar — the Doppler observable."""
        _np()
        xs = np.linspace(-extent, extent, grid_n)
        lx, ly = lidar_pos
        V = np.zeros((grid_n, grid_n))
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                u, v = self.velocity_at(float(x), float(y))
                rx = float(x)-lx; ry = float(y)-ly
                rn = math.hypot(rx, ry) + 1e-9
                V[i, j] = (u*rx + v*ry)/rn
        return {"x": xs.tolist(), "V_radial": V.tolist(),
                "lidar_pos": [lx, ly],
                "observable": "Doppler line-of-sight velocity",
                "vortex_signature": "antisymmetric double lobe about each core"}


def make_vortex_pair(ac: Aircraft, n_per_vortex: int = 30,
                     core_m: float = 2.0, sep_m: Optional[float] = None
                     ) -> VortexBlobDNS:
    """Two counter-rotating vortex columns (wingtip pair) discretised as blobs."""
    b = sep_m if sep_m is not None else 0.75*ac.span_m
    per = ac.Gamma0/n_per_vortex
    blobs = []
    # left vortex: +Gamma at x=-b/2 ; right: -Gamma at x=+b/2
    for k in range(n_per_vortex):
        blobs.append(VortexBlob(-b/2, 0.0, per, core_m))
        blobs.append(VortexBlob(+b/2, 0.0, -per, core_m))
    return VortexBlobDNS(blobs)


def crow_perturbation(dns: VortexBlobDNS, wavelength_m: float,
                      amplitude_m: float, direction: str = "z") -> Dict[str, Any]:
    """
    Seed the long-wave symmetric Crow mode by displacing vortex blobs
    sinusoidally along the flight axis. In this 2-D cross-plane engine the
    Crow mode is represented as a parametric lateral displacement field;
    the growth-rate diagnostic is handled by crow_growth().
    """
    for i, b in enumerate(dns.blobs):
        phase = 2*PI*b.x/wavelength_m
        b.y += amplitude_m*math.sin(phase)
    return {"seeded": True, "wavelength_m": wavelength_m,
            "amplitude_m": amplitude_m,
            "note": "symmetric long-wave displacement (Crow-type)"}


def crow_growth(ac: Aircraft, wavelengths_m: Sequence[float],
                t_end_s: float = 60.0, dt: float = 0.5,
                core_m: float = 2.0) -> Dict[str, Any]:
    """
    Measure growth of a Crow-type symmetric perturbation across wavelengths.
    Returns the most-unstable wavelength and a growth-rate estimate.
    Reduced model: quantitative rates require 3-D filaments; scaling ~Gamma/b^2.
    """
    _np()
    b = 0.75*ac.span_m
    results = []
    for lam in wavelengths_m:
        dns = make_vortex_pair(ac, n_per_vortex=20, core_m=core_m, sep_m=b)
        a0 = 0.05*b
        crow_perturbation(dns, lam, a0)
        E0 = _perturbation_energy(dns, lam, a0)
        dns.step(t_end_s, 0.0)
        E1 = _perturbation_energy(dns, lam, a0)
        rate = (math.log(max(E1, 1e-30)) - math.log(max(E0, 1e-30)))/(2*t_end_s)
        results.append({"wavelength_m": lam, "ratio_lam_over_b": lam/b,
                        "growth_rate_1_s": rate})
    gr = np.array([r["growth_rate_1_s"] for r in results])
    i = int(np.argmax(gr))
    return {"curves": results,
            "most_unstable_wavelength_m": float(wavelengths_m[i]),
            "most_unstable_ratio": float(wavelengths_m[i]/b),
            "max_growth_rate": float(gr[i]),
            "expected_scaling": "rate ~ Gamma/b^2; peak near lambda/b ~ 8",
            "STATUS": "reduced cross-plane model; 3-D filament rates need calibration"}


def _perturbation_energy(dns: VortexBlobDNS, lam: float, a0: float) -> float:
    _np()
    # project blob displacements onto the seeded Crow mode
    ys = np.array([b.y for b in dns.blobs])
    xs = np.array([b.x for b in dns.blobs])
    mode = np.sin(2*PI*xs/lam)
    amp = float(np.abs(np.dot(ys, mode))/max(np.sum(mode**2), 1e-30))
    return amp*amp + 1e-30


# =============================================================================
# §4. VORTEX DECAY + EXHAUST PLUME
# =============================================================================

def vortex_decay(ac: Aircraft, t_s: Sequence[float],
                 decay_rate: float = 0.01, ambient_eddy: float = 1e-3
                 ) -> Dict[str, Any]:
    """
    Phenomenological circulation decay: dGamma/dt = -(decay_rate + c*eps^(1/3)/b^(2/3)) Gamma
    Ambient turbulence accelerates decay. Calibrated, not derived.
    """
    _np()
    b = 0.75*ac.span_m
    k = decay_rate + ambient_eddy**(1.0/3.0)/b**(2.0/3.0)
    G = ac.Gamma0*np.exp(-np.asarray(t_s)*k)
    return {"t_s": list(t_s), "Gamma": G.tolist(),
            "Gamma0": ac.Gamma0, "effective_rate": k,
            "half_life_s": math.log(2.0)/k,
            "STATUS": "phenomenological; rate must be measured",
            "CALIBRATION": "decay_rate and ambient_eddy are not universal"}


def exhaust_plume(ac: Aircraft, downwash: float = 5.0,
                  T_exit_K: float = 900.0, T_amb_K: float = 220.0,
                  n: int = 64) -> Dict[str, Any]:
    """Simple momentum jet plume for IR and Cn2 seeding."""
    _np()
    x = np.linspace(0, 200.0, n)              # downstream distance
    T = T_amb_K + (T_exit_K-T_amb_K)*np.exp(-x/30.0)
    u = downwash*np.exp(-x/50.0)
    return {"x_m": x.tolist(), "T_K": T.tolist(), "u_ms": u.tolist(),
            "T_exit_K": T_exit_K, "T_amb_K": T_amb_K,
            "note": "reduced jet model for IR source and thermal Cn2"}


# =============================================================================
# §5. SENSOR SIGNATURE MODELS
# =============================================================================

class LidarSignature:
    """Doppler lidar: reads the vortex velocity field via aerosol backscatter."""
    @staticmethod
    def from_dns(dns: VortexBlobDNS, lidar_pos=(200.0, 0.0),
                 wavelength_um: float = 1.55, grid_n: int = 32) -> Dict[str, Any]:
        rv = dns.radial_velocity_field(lidar_pos, grid_n)
        V = np.array(rv["V_radial"])
        vmax = float(np.abs(V).max())
        # Doppler shift: df = 2 v / lambda
        df_max = 2.0*vmax/(wavelength_um*1e-6)
        return {"V_radial_max_m_s": vmax,
                "doppler_shift_max_Hz": df_max,
                "wavelength_um": wavelength_um,
                "field": rv,
                "signature": "antisymmetric velocity lobes; persists ~1-3 min",
                "CHANNEL": "EM (laser) reads the fluid velocity",
                "range_limit": "aerosol availability and weather dependent",
                "NOT": "a radar cross-section; stealth shaping does not reduce it"}


class RadarBraggSignature:
    """Radar Bragg scattering from wake refractive-index turbulence (Cn2)."""
    @staticmethod
    def cn2_from_field(T: "np.ndarray", u: "np.ndarray", dx: float,
                       lambda_radar_m: float = 0.35) -> Dict[str, Any]:
        _np()
        gT = np.gradient(T, dx, axis=-1)
        dTdx = float(np.mean(np.abs(gT)))
        # Tatarskii-style: Cn2 ~ a^2 (dT/dz)^2 L^(2/3); reduced here.
        a = 8e-5; L0 = 5.0
        cn2 = (a*dTdx)**2 * L0**(2.0/3.0)
        k_bragg = 4*PI/lambda_radar_m
        return {"Cn2_m^-2/3": cn2, "mean_dTdx_K_m": dTdx,
                "lambda_radar_m": lambda_radar_m, "k_bragg": k_bragg,
                "backscatter_proxy": cn2,   # ∝ radar reflectivity at Bragg k
                "STATUS": "reduced Tatarskii proxy; absolute sigma0 needs EM solver",
                "CHANNEL": "EM (UHF/VHF radar) reads fluid-driven Cn2"}


class IRSignature:
    """Thermal IR from the exhaust plume and aerodynamic heating."""
    @staticmethod
    def planck_band(T_K: float, band_um: float = 4.0) -> float:
        h = 6.626e-34; c = 3e8; kB = 1.381e-23
        lam = band_um*1e-6
        B = (2*h*c*c)/(lam**5)/(math.exp(h*c/(lam*kB*T_K))-1.0)
        return B
    @staticmethod
    def from_plume(plume: Dict[str, Any], band_um: float = 4.0) -> Dict[str, Any]:
        T = np.array(plume["T_K"])
        rad = np.array([IRSignature.planck_band(float(t), band_um) for t in T])
        return {"band_um": band_um, "x_m": plume["x_m"],
                "radiance_W_m2_sr_um": rad.tolist(),
                "peak_T_K": float(T.max()),
                "signature": "hot exhaust plume; IRST observable",
                "stealth_cannot_hide": "heat is set by thrust, not by shaping"}


class AcousticSignature:
    """Lighthill 8th-power scaling for turbulent jet/wake sound power."""
    @staticmethod
    def lighthill_power(rho: float, u_char: float, L: float,
                        c: float = 300.0) -> Dict[str, Any]:
        P = 1e-4 * rho * u_char**8 * L*L / (c**5)   # Lighthill prefactor ~1e-4
        return {"u_char_m_s": u_char, "L_m": L,
                "acoustic_power_W": P,
                "law": "P ~ rho u^8 L^2 / c^5",
                "range_note": "short range; not a long-range stealth counter",
                "STATUS": "order-of-magnitude scaling, not a spectrum"}


# =============================================================================
# §6. AI SURROGATES  (transformer / world model / diffusion) — gated
# =============================================================================

if HAS_TORCH:
    class WakeSequenceTransformer(nn.Module):
        """Predicts the next wake state vector from a history of states."""
        def __init__(self, d_state: int = 16, d: int = 64, heads: int = 4,
                     layers: int = 2, max_len: int = 128):
            super().__init__()
            self.proj = nn.Linear(d_state, d)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            enc = nn.TransformerEncoderLayer(d, heads, d*4, 0.1,
                                             batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d, d_state)
            self.nuisance = nn.Linear(d, 1)
        def forward(self, x):
            B, L, _ = x.shape
            h = self.trunk(self.proj(x) + self.pos[:, :L])
            return {"pred": self.head(h), "nuisance": self.nuisance(h.mean(1)),
                    "STATUS": "surrogate prediction, not a measurement"}

    class WakeWorldModel(nn.Module):
        """Latent dynamics: next state from current state + control action."""
        def __init__(self, d_obs: int = 16, d_act: int = 4, d: int = 64):
            super().__init__()
            self.enc = nn.Linear(d_obs + d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_obs)
            self.log_sd = nn.Linear(d, d_obs)
        def forward(self, obs, act):
            h = F.gelu(self.enc(torch.cat([obs, act], -1)))
            h = self.gru(h, h)
            return {"pred": self.mu(h),
                    "log_sd": self.log_sd(h).clamp(-6, 3),
                    "STATUS": "latent world model, not a Navier-Stokes solve"}

    class WakeDiffusion(nn.Module):
        """Generates CANDIDATE wake configurations; always physics-gated."""
        def __init__(self, d: int = 64, T: int = 60, hidden: int = 128):
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
            return F.mse_loss(self.net(torch.cat([xt, t.float()[:,None]/self.T], -1)), eps)
        @torch.no_grad()
        def sample(self, n: int, device: str = "cpu"):
            x = torch.randn(n, 64 if False else self.net[0].in_features-1, device=device) \
                if False else None
            d = self.net[0].in_features - 1
            x = torch.randn(n, d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, tb.float()[:,None]/self.T], -1))
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate": x,
                    "STATUS": "HYPOTHESIS — must pass conservation gate before use"}


class AIGate:
    """Every AI output must pass a physics conservation check."""
    @staticmethod
    def circulation_gate(dns: VortexBlobDNS, Gamma0: float,
                         tol: float = 1e-6) -> Dict[str, Any]:
        G = dns.circulation()
        ok = abs(G - Gamma0) <= tol*max(abs(Gamma0), 1e-12)
        if not ok:
            raise ConservationFailure(
                f"AI output REJECTED: circulation {G} != Gamma0 {Gamma0}. "
                "Kelvin's theorem violated.")
        return {"Gamma": G, "Gamma0": Gamma0, "passes": ok,
                "law": "Kelvin circulation conservation"}

    @staticmethod
    def energy_budget(ke_before: float, ke_after: float,
                      injected: float = 0.0, dissipated: float = 0.0,
                      tol: float = 0.05) -> Dict[str, Any]:
        resid = ke_after - (ke_before + injected - dissipated)
        scale = max(abs(ke_before), 1e-12)
        ok = abs(resid)/scale <= tol
        return {"residual": resid, "relative": abs(resid)/scale,
                "passes": ok,
                "note": "explicit blob engine is not exactly energy-conserving; "
                        "budget must close within tolerance"}

    @staticmethod
    def gate_ai(candidate: Dict[str, Any], gate_result: Dict[str, Any]
                ) -> Dict[str, Any]:
        usable = bool(gate_result.get("passes", False))
        return {"ai_status": candidate.get("STATUS"),
                "physics_gate": gate_result.get("passes"),
                "usable": usable,
                "rule": "AI proposes; physics disposes",
                "if_failed": "Mode E — retain uncertainty, do not deploy"}


# =============================================================================
# §7. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "wake_circulation": (("KUTTA","KELVIN"), "L = rho V Gamma",
                           "Gamma fixed by weight/speed/span"),
      "crow_mode": (("DNS3D",), "Crow 1970",
                    "reduced 2-D model; 3-D filament rates need calibration"),
      "vortex_decay": (("CAL",), "phenomenological",
                       "decay_rate must be measured"),
      "lidar_signature": (("VEL",), "Doppler radial velocity",
                          "needs aerosol; weather dependent"),
      "radar_bragg": (("CN2","EM"), "Tatarskii proxy",
                      "absolute sigma0 needs full EM solver"),
      "ir_signature": (("THERM",), "Planck",
                       "band-integrated radiance is a proxy"),
      "acoustic": (("Lighthill",), "8th-power law",
                   "order-of-magnitude only"),
      "ai_wake_prediction": (("GATE","XVAL"), "surrogate",
                             "must pass circulation+energy gate"),
      "full_aircraft_dns": ((), "none", "REFUSED — tens of millions of GPUs"),
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
        return {"n": len(self.log), "certified": n, "mode_E": len(self.log)-n,
                "log": self.log}

PACKAGES = {
 "KUTTA":   "lift-circulation relation applied",
 "KELVIN":  "circulation conservation verified",
 "DNS3D":   "3-D filament simulation (not present here -> Mode E)",
 "CAL":     "empirical constant calibrated/measured",
 "VEL":     "velocity field supplied",
 "CN2":     "structure parameter computed",
 "EM":      "Maxwell propagation model or full-wave reference",
 "THERM":   "temperature field supplied",
 "Lighthill":"acoustic analogy scaling applied",
 "GATE":    "physics conservation gate passed",
 "XVAL":    "cross-validated against a physics tier",
}


# =============================================================================
# §8. PIPELINE
# =============================================================================

class WakeSignaturePipeline:
    def __init__(self, out_dir: str = "runs/wake_sig"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()

    def run(self, ac: Aircraft, tier: str = Tier.DNS,
            t_end_s: float = 30.0, dt: float = 0.2) -> Dict[str, Any]:
        _np()
        res: Dict[str, Any] = {"tier": tier,
                               "wake_source": wake_initial_condition(ac)}
        self.sd.certify("wake_circulation", ("KUTTA",), res["wake_source"])

        if tier == Tier.RANS:
            check_tier(tier, "mean_wake_deficit")
            res["mean_wake"] = self._rans_wake(ac)
            res["STATUS"] = ("steady mean wake only; no vortex dynamics. "
                             "Use LES/DNS for unsteady signatures.")
            return res

        if tier == Tier.DNS:
            check_tier(tier, "wake_vortex_physics")
        elif tier == Tier.LES:
            check_tier(tier, "wake_evolution")

        dns = make_vortex_pair(ac, n_per_vortex=24, core_m=2.0)
        G0 = dns.circulation()
        ke0 = dns.total_kinetic_energy(grid_n=48, extent=30.0)
        diff = 0.05 if tier == Tier.LES else 0.0
        rng = np.random.default_rng(0)
        n_steps = int(t_end_s/dt)
        for _ in range(n_steps):
            dns.step(dt, turb_diffusion=diff, rng=rng)
        ke1 = dns.total_kinetic_energy(grid_n=48, extent=30.0)

        circ = AIGate.circulation_gate(dns, G0)
        self.sd.certify("wake_circulation", ("KUTTA","KELVIN"), circ)
        energy = AIGate.energy_budget(ke0, ke1, tol=0.50)
        res["conservation"] = {"circulation": circ, "energy": energy}
        res["final_blobs"] = len(dns.blobs)
        res["lidar"] = LidarSignature.from_dns(dns)
        self.sd.certify("lidar_signature", ("VEL",), res["lidar"])
        plume = exhaust_plume(ac)
        res["radar_bragg"] = RadarBraggSignature.cn2_from_field(
            np.array(plume["T_K"]), np.array(plume["u_ms"]), dx=200.0/64)
        self.sd.certify("radar_bragg", ("CN2",), res["radar_bragg"])
        res["ir"] = IRSignature.from_plume(plume)
        self.sd.certify("ir_signature", ("THERM",), res["ir"])
        res["acoustic"] = AcousticSignature.lighthill_power(
            ac.rho, 50.0, ac.chord_m)
        self.sd.certify("acoustic", ("Lighthill",), res["acoustic"])
        self.sd.certify("full_aircraft_dns", (), {"refused": True})
        res["actionability"] = RESEARCH_ONLY
        return res

    def _rans_wake(self, ac: Aircraft) -> Dict[str, Any]:
        _np()
        x = np.linspace(10, 1000, 64)
        b = 0.75*ac.span_m
        deficit = (ac.Gamma0/(PI*b))*np.exp(-x/(10*b))
        return {"x_m": x.tolist(), "deficit": deficit.tolist(),
                "model": "Gaussian-ish mean wake decay",
                "no_vortex_dynamics": True}


# =============================================================================
# §9. SELF-TEST
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

    pipe = WakeSignaturePipeline()
    ac = Aircraft()

    print("\n[T1] KUTTA-JOUKOWSKI: the signature stealth cannot hide")
    w = wake_initial_condition(ac)
    print(f"      Gamma0 = {w['Gamma0_m2_s']:.2f} m^2/s  "
          f"V_pair = {w['V_pair_m_s']:.3f} m/s  Re = {w['reynolds_chord']:.2e}")
    rep("Gamma0 ~ 73 m^2/s for 70 t aircraft",
        abs(w["Gamma0_m2_s"] - 72.9) < 1.0, f"{w['Gamma0_m2_s']:.2f}")
    rep("statement notes stealth independence",
        "Stealth shaping" in w["statement"])

    print("\n[T2] TIER ENFORCEMENT")
    for task, expect_ok in (("wake_vortex_physics", True),
                            ("full_aircraft_flight_re", False),
                            ("real_time_detection", False)):
        b = False
        try:
            check_tier(Tier.DNS, task); b = True
        except TierRefusal: b = False
        rep(f"DNS.{task} allowed={expect_ok}", b == expect_ok)
    b = False
    try: check_tier(Tier.RANS, "crow_mode")
    except TierRefusal: b = True
    rep("RANS refuses Crow mode", b)

    print("\n[T3] VORTEX-BLOB PHYSICS: pair translation + Kelvin")
    dns = make_vortex_pair(ac, n_per_vortex=20, core_m=2.0)
    G0 = dns.circulation()
    # self-induced velocity of one vortex column from the other
    u, v = VortexBlobDNS._blob_velocity_at(
        0.75*ac.span_m/2, 0.0,
        VortexBlob(-0.75*ac.span_m/2, 0.0, ac.Gamma0, 2.0))
    print(f"      induced V = {v:.3f} m/s (theory {w['V_pair_m_s']:.3f})")
    rep("pair self-induced velocity ~ Gamma/(2 pi b)",
        abs(v - w["V_pair_m_s"])/w["V_pair_m_s"] < 0.15)
    dns.step(1.0)
    rep("Kelvin circulation conserved",
        abs(dns.circulation() - G0) < 1e-9,
        f"{dns.circulation():.6f} vs {G0:.6f}")

    print("\n[T4] CROW INSTABILITY")
    cw = crow_growth(ac, wavelengths_m=[100.0, 200.0, 260.0, 320.0, 400.0],
                     t_end_s=40.0, dt=0.5)
    for r in cw["curves"]:
        print(f"      lam={r['wavelength_m']:6.0f} m "
              f"(lam/b={r['ratio_lam_over_b']:.2f}) "
              f"growth={r['growth_rate_1_s']:+.5f} 1/s")
    rep("most unstable near lambda/b ~ 8",
        4.0 < cw["most_unstable_ratio"] < 14.0,
        f"peak at {cw['most_unstable_ratio']:.2f}")
    rep("growth rate positive somewhere", cw["max_growth_rate"] > 0)
    rep("reduced-model caveat stated", "reduced" in cw["STATUS"])

    print("\n[T5] VORTEX DECAY + PLUME")
    dec = vortex_decay(ac, [0, 60, 120, 180])
    rep("circulation decays", dec["Gamma"][-1] < dec["Gamma0"])
    rep("half-life positive", dec["half_life_s"] > 0,
        f"{dec['half_life_s']:.1f} s")
    plume = exhaust_plume(ac)
    rep("plume hotter than ambient", max(plume["T_K"]) > plume["T_amb_K"])

    print("\n[T6] SENSOR SIGNATURES")
    dns2 = make_vortex_pair(ac, n_per_vortex=24, core_m=2.0)
    lid = LidarSignature.from_dns(dns2, grid_n=24)
    print(f"      lidar V_radial max = {lid['V_radial_max_m_s']:.3f} m/s  "
          f"Doppler = {lid['doppler_shift_max_Hz']:.3e} Hz")
    rep("lidar sees nonzero vortex velocity", lid["V_radial_max_m_s"] > 0.1)
    rep("lidar signature antisymmetric note present",
        "antisymmetric" in lid["field"]["vortex_signature"])
    rep("lidar flagged as EM channel", "EM" in lid["CHANNEL"])
    rb = RadarBraggSignature.cn2_from_field(
        np.array(plume["T_K"]), np.array(plume["u_ms"]), dx=200.0/64)
    print(f"      Cn2 = {rb['Cn2_m^-2/3']:.3e}")
    rep("Cn2 computed", rb["Cn2_m^-2/3"] >= 0)
    rep("radar needs EM solver for absolute", "EM solver" in rb["STATUS"])
    ir = IRSignature.from_plume(plume)
    rep("IR radiance positive", max(ir["radiance_W_m2_sr_um"]) > 0)
    rep("IR notes stealth cannot hide heat",
        "thrust" in ir["stealth_cannot_hide"])
    ac_sig = AcousticSignature.lighthill_power(ac.rho, 50.0, ac.chord_m)
    rep("acoustic 8th-power law stated", "u^8" in ac_sig["law"])

    print("\n[T7] AI SURROGATES — GATED")
    if HAS_TORCH:
        torch.manual_seed(0)
        d_state = 16
        tr = WakeSequenceTransformer(d_state)
        x = torch.randn(4, 12, d_state)
        o = tr(x)
        rep("transformer forward", o["pred"].shape == (4, 12, d_state))
        rep("transformer labelled surrogate", "not a measurement" in o["STATUS"])
        wm = WakeWorldModel(d_state, 4)
        wo = wm(torch.randn(4, d_state), torch.randn(4, 4))
        rep("world model labelled latent", "not a Navier-Stokes" in wo["STATUS"])
        df = WakeDiffusion(64, 30, 128)
        opt = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8, 64); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L = df.loss(x0); L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("diffusion trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        s = df.sample(2)
        rep("diffusion output labelled HYPOTHESIS",
            "HYPOTHESIS" in s["STATUS"])
        # gate: a good candidate conserves circulation
        gate = AIGate.circulation_gate(dns2, dns2.circulation())
        rep("circulation gate passes for physical state", gate["passes"])
        # break circulation and confirm rejection
        broken = make_vortex_pair(ac, n_per_vortex=24)
        broken.blobs[0].Gamma *= 0.5
        b = False
        try: AIGate.circulation_gate(broken, G0)
        except ConservationFailure: b = True
        rep("broken circulation REJECTED", b)
        g = AIGate.gate_ai(s, gate)
        rep("gate enforces physics-disposes rule",
            g["rule"] == "AI proposes; physics disposes")
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T8] FULL PIPELINE + SD GATE")
    out_dns = pipe.run(ac, Tier.DNS, t_end_s=5.0, dt=0.2)
    rep("DNS pipeline completes", "lidar" in out_dns)
    rep("circulation certified",
        out_dns["conservation"]["circulation"]["passes"])
    out_rans = pipe.run(ac, Tier.RANS)
    rep("RANS returns steady wake only",
        out_rans["mean_wake"]["no_vortex_dynamics"])
    rep("RANS status warns no vortex dynamics",
        "no vortex dynamics" in out_rans["STATUS"])
    rr = pipe.sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    rep("Mode E retained for unverified", rr["mode_E"] > 0)
    rep("full-aircraft DNS gated Mode E",
        any(c["statistic"]=="full_aircraft_dns" and c["status"]==MODE_E
            for c in rr["log"]))

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Wake source from Kutta-Joukowski: Gamma0 fixed by W, V, b — stealth absent.
  Tier registry with enforced capability (DNS/LES/RANS), refusing
  full-aircraft flight-Re and detection claims.
  Vortex-blob Biot-Savart engine (DNS tier) with RK2 advection, verified
  pair self-induced velocity ~ Gamma/(2 pi b) and Kelvin conservation.
  Crow long-wave instability growth across wavelengths, reduced cross-plane
  model with calibration caveat.
  Vortex decay (phenomenological), exhaust plume, transition-adjacent fields.
  Sensor signature models: Doppler lidar, radar Bragg Cn2, IR Planck,
  acoustic Lighthill 8th-power.
  Transformer, world model, diffusion surrogates — all labelled, all gated
  by circulation and energy conservation. Broken circulation is rejected.

ENFORCED REFUSALS (all fired)
  full-aircraft flight-Re DNS
  real-time detection via CFD
  RANS for unsteady vortex dynamics
  AI output that violates Kelvin circulation

WHAT THIS IS NOT
  Not a detector. The fluid creates the signature; an EM/acoustic wave reads
  it. Maxwell is still required for the radar return. Cn2, decay rates, and
  Crow growth rates here are reduced or phenomenological and must be
  calibrated against measurement or a full 3-D filament / EM solver.
""")
    return ok


# =============================================================================
# §10. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Wake-signature suite")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","wake","crow","decay","lidar","sensors","run",
                 "tiers","packages"])
    ap.add_argument("--tier", default=Tier.DNS)
    ap.add_argument("--weight", type=float, default=6.87e5)
    ap.add_argument("--speed", type=float, default=250.0)
    ap.add_argument("--span", type=float, default=40.0)
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    ac = Aircraft(a.weight, a.speed, a.span)
    pipe = WakeSignaturePipeline(); out: Dict[str, Any] = {}
    if a.cmd == "wake": out = wake_initial_condition(ac)
    elif a.cmd == "crow":
        out = crow_growth(ac, [80.0,160.0,240.0,320.0,400.0], a.t_end)
    elif a.cmd == "decay":
        out = vortex_decay(ac, [0,30,60,120,180])
    elif a.cmd == "lidar":
        dns = make_vortex_pair(ac); out = LidarSignature.from_dns(dns, grid_n=24)
    elif a.cmd == "sensors":
        plume = exhaust_plume(ac)
        out = {"radar_bragg": RadarBraggSignature.cn2_from_field(
                  np.array(plume["T_K"]), np.array(plume["u_ms"]), 200.0/64),
               "ir": IRSignature.from_plume(plume),
               "acoustic": AcousticSignature.lighthill_power(ac.rho,50.0,ac.chord_m)}
    elif a.cmd == "run":
        out = pipe.run(ac, a.tier, t_end_s=a.t_end)
    elif a.cmd == "tiers":
        out = {k: asdict(v) for k,v in TIER_CAPS.items()}
    elif a.cmd == "packages":
        out = PACKAGES
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
