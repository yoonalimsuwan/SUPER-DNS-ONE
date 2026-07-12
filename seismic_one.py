"""
SEISMIC ONE — Earthquake Engineering Module for the ONE Ecosystem
====================================================================

Production-grade seismic analysis module covering:
    1. GroundMotionEngine     — ground motion I/O, PGA/PGV/PGD, response spectra
    2. SiteResponseLayer      — 1D equivalent-linear site response (SHAKE-style)
    3. StructuralResponseLayer— MDOF shear-building, Newmark-beta + Bouc-Wen hysteresis
    4. LiquefactionAssessment — Seed-Idriss (1971/2001) simplified procedure
    5. SeismicOne             — orchestrator: bedrock motion -> site response
                                  -> structural response -> damage/liquefaction

Scope & honesty note
---------------------
This module is *structural/geotechnical* seismic engineering (elastodynamics,
site response, structural dynamics), NOT a seismic-wave-propagation solver for
the crust (that would require a 2D/3D elastodynamic FEM/spectral-element code,
e.g. coupling with a future `crustal_wave_one.py`). It is complementary to the
CFD modules (SUPER DNS ONE) for problems like tsunami generation or tank
sloshing, which remain fluid-dynamics problems fed by this module's outputs
(e.g. peak ground acceleration time histories).

Units: SI throughout (m, s, kg, N, Pa) unless noted otherwise.

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem authorship convention.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as spfft
from scipy.linalg import eigh
from dataclasses import dataclass, field
from typing import Optional, Callable
import warnings

__version__ = "1.0.0"
__all__ = [
    "GroundMotionEngine",
    "SiteResponseLayer",
    "StructuralResponseLayer",
    "BoucWenElement",
    "LiquefactionAssessment",
    "SeismicOne",
]

G_ACCEL = 9.80665  # m/s^2


def _safe_complex_exp(z: np.ndarray, max_real: float = 50.0) -> np.ndarray:
    """
    exp(z) for complex z, with the real part of the exponent clipped to
    prevent floating-point overflow. This occurs in layered transfer-matrix
    recursions (Thomson-Haskell) for thick/soft/high-damping layers at high
    frequency, where the up-going-wave amplitude ratio grows exponentially
    with depth. Clipping at exp(50) ~ 5e21 is a physically negligible
    truncation (such frequency components are effectively fully attenuated)
    and keeps the recursion finite without altering results in the
    frequency range that matters for engineering response spectra.
    """
    re = np.clip(z.real, -max_real, max_real)
    return np.exp(re + 1j * z.imag)


# =====================================================================
# 1. GROUND MOTION ENGINE
# =====================================================================

@dataclass
class GroundMotion:
    """A single ground-motion acceleration time history."""
    time: np.ndarray          # [s]
    accel: np.ndarray         # [m/s^2]
    dt: float                 # [s]
    name: str = "unnamed"

    @property
    def n(self) -> int:
        return len(self.accel)

    @property
    def pga(self) -> float:
        return float(np.max(np.abs(self.accel)))

    def velocity(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(
            0.5 * (self.accel[1:] + self.accel[:-1]) * self.dt)])

    def displacement(self) -> np.ndarray:
        v = self.velocity()
        return np.concatenate([[0.0], np.cumsum(
            0.5 * (v[1:] + v[:-1]) * self.dt)])

    @property
    def pgv(self) -> float:
        return float(np.max(np.abs(self.velocity())))

    @property
    def pgd(self) -> float:
        return float(np.max(np.abs(self.displacement())))


class GroundMotionEngine:
    """
    Ground motion generation, baseline correction, and response spectra.
    """

    @staticmethod
    def from_array(accel: np.ndarray, dt: float, name: str = "record") -> GroundMotion:
        t = np.arange(len(accel)) * dt
        return GroundMotion(time=t, accel=np.asarray(accel, dtype=float), dt=dt, name=name)

    @staticmethod
    def synthetic_motion(
        duration: float = 30.0,
        dt: float = 0.005,
        pga_target: float = 3.0,          # m/s^2
        dominant_freq: float = 2.0,        # Hz
        rng: Optional[np.random.Generator] = None,
        name: str = "synthetic",
    ) -> GroundMotion:
        """
        Generate a physically-reasonable synthetic ground motion using a
        Clough-Penzien-filtered white noise with a compactly-supported
        envelope (Jennings-type), then scaled to the target PGA.
        This is for testing/design-spectrum-matching workflows, not a
        substitute for recorded or simulated (physics-based) motions.
        """
        if rng is None:
            rng = np.random.default_rng()
        n = int(duration / dt)
        t = np.arange(n) * dt

        # Jennings envelope: rise - strong-motion plateau - decay
        t1, t2 = 0.15 * duration, 0.6 * duration
        env = np.ones(n)
        rise = t < t1
        decay = t > t2
        env[rise] = (t[rise] / t1) ** 2
        c = 0.3 / np.log(10)  # decay rate constant (empirical)
        env[decay] = np.exp(-c * (t[decay] - t2) / (duration - t2) * np.log(1000))

        white = rng.standard_normal(n)

        # Clough-Penzien filter: ground filter (bedrock) cascaded with a
        # secondary filter to suppress low-frequency drift energy.
        wg = 2 * np.pi * dominant_freq
        zg = 0.6
        wf = 2 * np.pi * dominant_freq * 0.1
        zf = 0.6

        freqs = spfft.rfftfreq(n, dt)
        w = 2 * np.pi * freqs
        # Avoid divide-by-zero at w=0
        w_safe = np.where(w == 0, 1e-8, w)

        H_g = (wg**2 + 2j * zg * wg * w_safe) / (wg**2 - w_safe**2 + 2j * zg * wg * w_safe)
        H_f = (w_safe**2) / (wf**2 - w_safe**2 + 2j * zf * wf * w_safe)
        H = H_g * H_f

        Wf = spfft.rfft(white)
        filtered = spfft.irfft(Wf * H, n=n)
        raw = filtered * env

        raw = raw - np.mean(raw)  # baseline shift removal
        raw = raw / (np.max(np.abs(raw)) + 1e-12) * pga_target

        return GroundMotion(time=t, accel=raw, dt=dt, name=name)

    @staticmethod
    def baseline_correct(motion: GroundMotion, poly_order: int = 2) -> GroundMotion:
        """Remove polynomial drift from displacement via least-squares fit on accel."""
        t = motion.time
        coeffs = np.polyfit(t, motion.accel, poly_order)
        trend = np.polyval(coeffs, t)
        corrected = motion.accel - trend
        return GroundMotion(time=t, accel=corrected, dt=motion.dt, name=motion.name + "_corrected")

    @staticmethod
    def response_spectrum(
        motion: GroundMotion,
        periods: np.ndarray,
        zeta: float = 0.05,
    ) -> dict:
        """
        Compute pseudo-spectral response (Sa, Sv, Sd) for a suite of SDOF
        oscillators via exact piecewise-linear (Nigam-Jennings) recursion.

        Returns dict with 'periods', 'Sa' (m/s^2), 'Sv' (m/s), 'Sd' (m).
        """
        dt = motion.dt
        p = motion.accel
        n = len(p)
        Sa = np.zeros_like(periods, dtype=float)
        Sv = np.zeros_like(periods, dtype=float)
        Sd = np.zeros_like(periods, dtype=float)

        for i, T in enumerate(periods):
            if T <= 1e-6:
                Sa[i] = motion.pga
                continue
            wn = 2 * np.pi / T
            wd = wn * np.sqrt(1 - zeta**2)
            u, v = 0.0, 0.0
            umax = 0.0
            # Nigam-Jennings exact recursive solution coefficients
            e = np.exp(-zeta * wn * dt)
            s = np.sin(wd * dt)
            c = np.cos(wd * dt)

            A11 = e * (zeta / np.sqrt(1 - zeta**2) * s + c)
            A12 = e * (s / wd)
            A21 = -wn * e * s / np.sqrt(1 - zeta**2)
            A22 = e * (c - zeta / np.sqrt(1 - zeta**2) * s)

            B11 = e * (((2 * zeta**2 - 1) / (wn**2 * dt) + zeta / wn) * s / wd
                       + (2 * zeta / (wn**3 * dt) + 1 / wn**2) * c) - 2 * zeta / (wn**3 * dt)
            B12 = -e * ((2 * zeta**2 - 1) / (wn**2 * dt) * s / wd
                        + 2 * zeta / (wn**3 * dt) * c) - 1 / wn**2 + 2 * zeta / (wn**3 * dt)
            B21 = e * (((2 * zeta**2 - 1) / (wn**2 * dt) + zeta / wn) * (c - zeta / np.sqrt(1 - zeta**2) * s)
                       - (2 * zeta / (wn**3 * dt) + 1 / wn**2) * (wd * s + zeta * wn * c)
                       ) + 1 / (wn**2 * dt)
            B22 = -e * ((2 * zeta**2 - 1) / (wn**2 * dt) * (c - zeta / np.sqrt(1 - zeta**2) * s)
                        - 2 * zeta / (wn**3 * dt) * (wd * s + zeta * wn * c)
                        ) - 1 / (wn**2 * dt)

            for k in range(n - 1):
                u_new = A11 * u + A12 * v + B11 * p[k] + B12 * p[k + 1]
                v_new = A21 * u + A22 * v + B21 * p[k] + B22 * p[k + 1]
                u, v = u_new, v_new
                if abs(u) > umax:
                    umax = abs(u)

            Sd[i] = umax
            Sv[i] = wn * umax
            Sa[i] = wn**2 * umax

        return {"periods": periods, "Sa": Sa, "Sv": Sv, "Sd": Sd}


# =====================================================================
# 2. SITE RESPONSE LAYER (1D equivalent-linear, SHAKE-style)
# =====================================================================

@dataclass
class SoilLayerProps:
    thickness: float       # m
    density: float         # kg/m^3
    Gmax: float             # Pa, small-strain shear modulus
    damping_min: float = 0.02
    # Modulus reduction / damping curves as functions of shear strain (fraction)
    G_over_Gmax_curve: Callable[[np.ndarray], np.ndarray] = None
    damping_curve: Callable[[np.ndarray], np.ndarray] = None

    def __post_init__(self):
        if self.G_over_Gmax_curve is None:
            self.G_over_Gmax_curve = default_darendeli_modulus_reduction
        if self.damping_curve is None:
            self.damping_curve = default_darendeli_damping


def default_darendeli_modulus_reduction(gamma: np.ndarray, gamma_r: float = 1.0e-4) -> np.ndarray:
    """
    Modified hyperbolic (Darendeli-type) modulus reduction:
    G/Gmax = 1/(1+(gamma/gamma_r)^a).

    KNOWN SIMPLIFICATION: gamma_r (reference strain) is held constant here.
    In reality gamma_r increases with effective confining stress and
    plasticity index (Darendeli 2001 gives gamma_r ~ (a1 + a2*PI*OCR^a3)*
    sigma0'^a4), so shallow/low-confinement soil is softer at a given
    strain than this constant-gamma_r model implies, and can converge to
    very low G/Gmax under strong shaking. For production site-specific
    work, override `G_over_Gmax_curve` / `damping_curve` per SoilLayerProps
    with laboratory-measured or depth/stress-adjusted curves rather than
    relying on this generic default.
    """
    a = 0.92
    return 1.0 / (1.0 + np.abs(gamma / gamma_r) ** a)


def default_darendeli_damping(gamma: np.ndarray, gamma_r: float = 1.0e-4, D_min: float = 0.02) -> np.ndarray:
    """Empirical damping growth curve consistent with modulus reduction (Masing-consistent trend)."""
    Gr = default_darendeli_modulus_reduction(gamma, gamma_r)
    D_mask = D_min + 0.25 * (1 - Gr) ** 1.1
    return np.clip(D_mask, D_min, 0.25)


class SiteResponseLayer:
    """
    1D equivalent-linear site response analysis in the frequency domain
    (SHAKE91 methodology): propagate bedrock (outcrop) motion through a
    horizontally-layered soil column via complex transfer functions,
    iterating G and damping to strain-compatible values.
    """

    def __init__(self, layers: list[SoilLayerProps], rock_density: float, rock_Vs: float):
        self.layers = layers
        self.rock_density = rock_density
        self.rock_Vs = rock_Vs
        self.n_layers = len(layers)

    def _layer_impedance(self, G: np.ndarray, D: np.ndarray, density: np.ndarray):
        """Complex shear modulus G* = G(1+2iD) and complex shear-wave velocity."""
        G_star = G * (1 + 2j * D)
        Vs_star = np.sqrt(G_star / density)
        return G_star, Vs_star

    def transfer_function(self, freqs: np.ndarray, G: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Compute the complex transfer function from bedrock outcrop to
        ground surface using the standard SHAKE (Schnabel et al. 1972;
        Kramer 1996, Sec. 5.4.2) recursive formulation for a layered
        viscoelastic medium over an elastic half-space.

        Recursion direction: TOP-DOWN. Layers are indexed 0 (surface) to
        n-1 (deepest soil layer); index n denotes the underlying rock
        half-space. The free-surface boundary condition (zero shear
        stress at z=0) fixes A_0 = B_0 = 1 (arbitrary normalization,
        cancels in the final ratio). Recursing downward through each
        layer's own thickness/wavenumber cannot overflow (it corresponds
        to the physically well-posed direction of wave decay), unlike a
        bottom-up recursion which is numerically unstable/ill-posed for
        damped media (and was found, in a bottom-up attempt, to make the
        impedance-ratio term cancel out of the surface amplitude entirely
        -- an algebraic tell that the recursion direction was wrong).

        H(w) = surface_motion / bedrock_outcrop_motion = 1 / A_n(w),
        where A_n is the up-going wave amplitude computed at the rock
        interface by propagating the surface condition down through all
        soil layers (bedrock outcrop motion = 2*A_n by the free-surface
        convention applied hypothetically at the rock elevation).
        """
        thicknesses = np.array([l.thickness for l in self.layers])
        densities = np.array([l.density for l in self.layers])
        G_star, Vs_star = self._layer_impedance(G, D, densities)

        n = self.n_layers
        w = 2 * np.pi * freqs
        w_safe = np.where(w == 0, 1e-10, w)

        k_star = w_safe[None, :] / Vs_star[:, None]  # (n_layers, n_freq)

        rock_G_star = self.rock_density * self.rock_Vs**2

        A = np.ones((n + 1, len(freqs)), dtype=complex)
        B = np.ones((n + 1, len(freqs)), dtype=complex)

        for i in range(n):
            Vs_i = Vs_star[i]
            rho_i = densities[i]
            if i == n - 1:
                Vs_ip1 = self.rock_Vs
                rho_ip1 = self.rock_density
            else:
                Vs_ip1 = Vs_star[i + 1]
                rho_ip1 = densities[i + 1]

            alpha_star = (rho_i * Vs_i) / (rho_ip1 * Vs_ip1)
            k_i = k_star[i]
            h_i = thicknesses[i]

            exp_p = _safe_complex_exp(1j * k_i * h_i)
            exp_m = _safe_complex_exp(-1j * k_i * h_i)

            A_i = A[i]
            B_i = B[i]

            A[i + 1] = 0.5 * A_i * (1 + alpha_star) * exp_p + 0.5 * B_i * (1 - alpha_star) * exp_m
            B[i + 1] = 0.5 * A_i * (1 - alpha_star) * exp_p + 0.5 * B_i * (1 + alpha_star) * exp_m

        A_bedrock = A[n]
        A_bedrock_safe = np.where(np.abs(A_bedrock) < 1e-300, 1e-300, A_bedrock)
        H = 1.0 / A_bedrock_safe
        return H

    def analyze(
        self,
        bedrock_motion: GroundMotion,
        n_iter: int = 8,
        strain_ratio: float = 0.65,
        tol: float = 0.02,
    ) -> dict:
        """
        Run the equivalent-linear iteration until G, D converge (or n_iter reached).
        Returns surface motion, converged G/Gmax and damping per layer, and
        peak shear strain per layer.
        """
        n = self.n_layers
        G = np.array([l.Gmax for l in self.layers], dtype=float)
        D = np.array([l.damping_min for l in self.layers], dtype=float)
        densities = np.array([l.density for l in self.layers])

        dt = bedrock_motion.dt
        acc = bedrock_motion.accel
        nfft = spfft.next_fast_len(len(acc))
        freqs = spfft.rfftfreq(nfft, dt)
        Acc_f = spfft.rfft(acc, n=nfft)

        history = []
        for it in range(n_iter):
            H = self.transfer_function(freqs, G, D)
            surf_acc_f = Acc_f * H
            surf_acc = spfft.irfft(surf_acc_f, n=nfft)[: len(acc)]

            # Estimate peak shear strain per layer using layer transfer
            # functions evaluated at layer mid-depth (simplified: use strain
            # from velocity gradient proportional to local particle velocity
            # amplitude / Vs, a standard SHAKE approximation).
            gamma_eff = np.zeros(n)
            for i in range(n):
                Vs_i = np.sqrt(G[i] / densities[i])
                # transfer function down to top of layer i (from rock)
                H_i = self._transfer_to_layer(freqs, G, D, i)
                iw = 2j * np.pi * freqs
                # Frequency-domain integration accel->velocity: divide by i*omega.
                # The DC (freq=0) term has no physical velocity content here
                # (it would represent unbounded drift) and must be zeroed
                # rather than divided, or a near-zero denominator creates a
                # spurious, unbounded low-frequency velocity spike.
                layer_vel_f = np.where(iw == 0, 0.0, Acc_f * H_i / np.where(iw == 0, 1.0, iw))
                layer_vel = spfft.irfft(layer_vel_f, n=nfft)[: len(acc)]
                peak_v = np.max(np.abs(layer_vel))
                gamma_max = peak_v / Vs_i
                gamma_eff[i] = strain_ratio * gamma_max

            G_new = np.array([
                self.layers[i].Gmax * self.layers[i].G_over_Gmax_curve(np.array([gamma_eff[i]]))[0]
                for i in range(n)
            ])
            D_new = np.array([
                self.layers[i].damping_curve(np.array([gamma_eff[i]]))[0]
                for i in range(n)
            ])

            rel_change = np.max(np.abs(G_new - G) / (G + 1e-12))
            history.append({"iter": it, "G": G_new.copy(), "D": D_new.copy(),
                             "gamma": gamma_eff.copy(), "rel_change": float(rel_change)})
            G, D = G_new, D_new
            if rel_change < tol:
                break

        H_final = self.transfer_function(freqs, G, D)
        surf_acc_f = Acc_f * H_final
        surf_acc = spfft.irfft(surf_acc_f, n=nfft)[: len(acc)]
        surface_motion = GroundMotion(time=bedrock_motion.time, accel=surf_acc,
                                       dt=dt, name=bedrock_motion.name + "_surface")

        return {
            "surface_motion": surface_motion,
            "converged_G": G,
            "converged_D": D,
            "peak_shear_strain": gamma_eff,
            "iterations": len(history),
            "history": history,
            "amplification_PGA": surface_motion.pga / max(bedrock_motion.pga, 1e-12),
        }

    def _transfer_to_layer(self, freqs, G, D, target_layer_idx):
        """
        Transfer function from bedrock outcrop motion to the TOP of a given
        soil layer index, using the same top-down recursion as
        transfer_function() (see docstring there for the direction
        rationale). Returns (A_target + B_target) / (2 * A_bedrock).
        """
        thicknesses = np.array([l.thickness for l in self.layers])
        densities = np.array([l.density for l in self.layers])
        G_star, Vs_star = self._layer_impedance(G, D, densities)
        n = self.n_layers
        w = 2 * np.pi * freqs
        w_safe = np.where(w == 0, 1e-10, w)
        k_star = w_safe[None, :] / Vs_star[:, None]

        A = np.ones((n + 1, len(freqs)), dtype=complex)
        B = np.ones((n + 1, len(freqs)), dtype=complex)
        for i in range(n):
            Vs_i = Vs_star[i]; rho_i = densities[i]
            if i == n - 1:
                Vs_ip1 = self.rock_Vs; rho_ip1 = self.rock_density
            else:
                Vs_ip1 = Vs_star[i + 1]; rho_ip1 = densities[i + 1]
            alpha_star = (rho_i * Vs_i) / (rho_ip1 * Vs_ip1)
            k_i = k_star[i]; h_i = thicknesses[i]
            exp_p = _safe_complex_exp(1j * k_i * h_i); exp_m = _safe_complex_exp(-1j * k_i * h_i)
            A_i = A[i]; B_i = B[i]
            A[i + 1] = 0.5 * A_i * (1 + alpha_star) * exp_p + 0.5 * B_i * (1 - alpha_star) * exp_m
            B[i + 1] = 0.5 * A_i * (1 - alpha_star) * exp_p + 0.5 * B_i * (1 + alpha_star) * exp_m

        A_bedrock_safe = np.where(np.abs(A[n]) < 1e-300, 1e-300, A[n])
        amp = A[target_layer_idx] + B[target_layer_idx]
        return amp / (2.0 * A_bedrock_safe)


# =====================================================================
# 3. STRUCTURAL RESPONSE LAYER (MDOF + Bouc-Wen hysteresis)
# =====================================================================

class BoucWenElement:
    """
    Bouc-Wen smooth hysteretic restoring-force model for one inter-story
    spring, used to capture nonlinear stiffness degradation / pinching
    under strong shaking (structural damage proxy).

        F = alpha*k*u + (1-alpha)*k*z
        dz/dt = A*du/dt - beta*|du/dt|*|z|^(n-1)*z - gamma*du/dt*|z|^n
    """

    def __init__(self, k: float, uy: float, alpha: float = 0.05,
                 beta: float = 0.5, gamma: float = 0.5, n: float = 1.0):
        self.k = k
        self.uy = uy
        self.alpha = alpha
        self.A = 1.0
        self.beta = beta / uy**n if n != 0 else beta
        self.gamma_bw = gamma / uy**n if n != 0 else gamma
        self.n = n

    def dz_dt(self, z: float, du_dt: float) -> float:
        return (self.A * du_dt
                - self.beta * abs(du_dt) * (abs(z) ** (self.n - 1)) * z
                - self.gamma_bw * du_dt * (abs(z) ** self.n))

    def force(self, u: float, z: float) -> float:
        return self.alpha * self.k * u + (1 - self.alpha) * self.k * self.uy * z


@dataclass
class Story:
    mass: float             # kg
    k_elastic: float        # N/m, initial inter-story stiffness
    uy: float               # m, yield inter-story drift
    alpha_ratio: float = 0.05   # post-yield stiffness ratio
    height: float = 3.0     # m, for drift-ratio reporting


class StructuralResponseLayer:
    """
    Lumped-mass MDOF shear-building model. Linear modal properties from
    eigen-analysis; nonlinear time-history response via Newmark-beta
    (average acceleration, unconditionally stable) with Bouc-Wen
    inter-story hysteresis and Rayleigh damping.
    """

    def __init__(self, stories: list[Story], zeta: float = 0.05):
        self.stories = stories
        self.n = len(stories)
        self.zeta = zeta
        self.M = np.diag([s.mass for s in stories])
        self.K_elastic = self._build_stiffness([s.k_elastic for s in stories])
        self.bw_elements = [
            BoucWenElement(k=s.k_elastic, uy=s.uy, alpha=s.alpha_ratio) for s in stories
        ]

    def _build_stiffness(self, k_list: list[float]) -> np.ndarray:
        n = self.n
        K = np.zeros((n, n))
        for i in range(n):
            k_i = k_list[i]
            k_ip1 = k_list[i + 1] if i + 1 < n else 0.0
            K[i, i] = k_i + k_ip1
            if i > 0:
                K[i, i - 1] = -k_i
                K[i - 1, i] = -k_i
        return K

    def modal_analysis(self) -> dict:
        eigvals, eigvecs = eigh(self.K_elastic, self.M)
        eigvals = np.clip(eigvals, 0, None)
        omegas = np.sqrt(eigvals)
        periods = np.where(omegas > 1e-9, 2 * np.pi / np.where(omegas > 1e-9, omegas, 1), np.inf)
        return {"omega": omegas, "periods": periods, "mode_shapes": eigvecs}

    def rayleigh_coeffs(self, omega1: float, omega2: float) -> tuple[float, float]:
        a0 = 2 * self.zeta * omega1 * omega2 / (omega1 + omega2)
        a1 = 2 * self.zeta / (omega1 + omega2)
        return a0, a1

    def time_history_analysis(
        self,
        ground_motion: GroundMotion,
        nonlinear: bool = True,
        beta_nm: float = 0.25,
        gamma_nm: float = 0.5,
    ) -> dict:
        """
        Newmark-beta average-acceleration integration of the equation of
        motion  M*u'' + C*u' + F_restoring(u,u') = -M*r*ag(t),  r = influence vector (ones).
        """
        modal = self.modal_analysis()
        omegas_sorted = np.sort(modal["omega"][modal["omega"] > 1e-9])
        if len(omegas_sorted) >= 2:
            w1, w2 = omegas_sorted[0], omegas_sorted[1]
        else:
            w1 = w2 = omegas_sorted[0] if len(omegas_sorted) else 1.0
        a0, a1 = self.rayleigh_coeffs(w1, w2)
        C = a0 * self.M + a1 * self.K_elastic

        n = self.n
        dt = ground_motion.dt
        ag = ground_motion.accel
        nt = len(ag)
        r = np.ones(n)

        u = np.zeros((nt, n))
        v = np.zeros((nt, n))
        a = np.zeros((nt, n))
        z = np.zeros((nt, n))  # Bouc-Wen hysteretic variables
        story_force = np.zeros((nt, n))

        Minv_diag = 1.0 / np.diag(self.M)

        def restoring_force(u_vec, z_vec):
            """Inter-story spring forces assembled into nodal force vector."""
            f_story = np.zeros(n)
            drifts = np.zeros(n)
            for i in range(n):
                drift = u_vec[i] - (u_vec[i - 1] if i > 0 else 0.0)
                drifts[i] = drift
                if nonlinear:
                    f_story[i] = self.bw_elements[i].force(drift, z_vec[i])
                else:
                    f_story[i] = self.stories[i].k_elastic * drift
            f_node = np.zeros(n)
            for i in range(n):
                f_node[i] += f_story[i]
                if i + 1 < n:
                    f_node[i] -= f_story[i + 1]
            return f_node, drifts, f_story

        # initial acceleration
        f_node0, drifts0, fstory0 = restoring_force(u[0], z[0])
        a[0] = Minv_diag * (-self.M @ r * ag[0] - C @ v[0] - f_node0)
        story_force[0] = fstory0

        for k in range(nt - 1):
            u_k, v_k, a_k, z_k = u[k], v[k], a[k], z[k]

            # Newmark predictors
            u_pred = u_k + dt * v_k + dt**2 * (0.5 - beta_nm) * a_k
            v_pred = v_k + dt * (1 - gamma_nm) * a_k

            # effective stiffness (tangent approximated via secant on BW state;
            # use fixed-point (Newton-like) iteration for the nonlinear step)
            u_new = u_pred.copy()
            z_new = z_k.copy()
            for _newton in range(20):
                drift_new = np.array([
                    u_new[i] - (u_new[i - 1] if i > 0 else 0.0) for i in range(n)
                ])
                drift_old = np.array([
                    u_k[i] - (u_k[i - 1] if i > 0 else 0.0) for i in range(n)
                ])
                v_trial = v_pred + gamma_nm * dt * a_k  # placeholder, refined below
                du = drift_new - drift_old
                if nonlinear:
                    for i in range(n):
                        dzdt = self.bw_elements[i].dz_dt(z_k[i], du[i] / dt if dt > 0 else 0.0)
                        z_new[i] = z_k[i] + dzdt * dt
                f_node, drifts, fstory = restoring_force(u_new, z_new)
                a_trial = Minv_diag * (-self.M @ r * ag[k + 1] - C @ (v_pred + gamma_nm * dt * a_k) - f_node)
                u_corr = u_pred + beta_nm * dt**2 * a_trial
                if np.max(np.abs(u_corr - u_new)) < 1e-10:
                    u_new = u_corr
                    break
                u_new = u_corr

            a_new = Minv_diag * (-self.M @ r * ag[k + 1] - C @ (v_pred + gamma_nm * dt * a[k]) - f_node)
            v_new = v_pred + gamma_nm * dt * a_new

            u[k + 1] = u_new
            v[k + 1] = v_new
            a[k + 1] = a_new
            z[k + 1] = z_new
            story_force[k + 1] = fstory

        drift_ratios = np.zeros((nt, n))
        for i in range(n):
            story_drift = u[:, i] - (u[:, i - 1] if i > 0 else 0.0)
            drift_ratios[:, i] = story_drift / self.stories[i].height

        max_drift_ratio = np.max(np.abs(drift_ratios), axis=0)
        max_abs_accel = np.max(np.abs(a + ag[:, None]), axis=0)

        # simple damage index (Park-Ang inspired, drift-based proxy):
        # 0 = elastic, >=1 = collapse-level drift exceedance
        damage_index = np.array([
            max_drift_ratio[i] / (self.stories[i].uy / self.stories[i].height) / 4.0
            for i in range(n)
        ])

        return {
            "time": ground_motion.time,
            "displacement": u,
            "velocity": v,
            "acceleration_relative": a,
            "total_acceleration": a + ag[:, None],
            "story_force": story_force,
            "drift_ratios": drift_ratios,
            "max_drift_ratio": max_drift_ratio,
            "max_abs_accel": max_abs_accel,
            "damage_index": damage_index,
            "modal": modal,
        }


# =====================================================================
# 4. LIQUEFACTION ASSESSMENT (Seed-Idriss simplified procedure)
# =====================================================================

class LiquefactionAssessment:
    """
    Simplified stress-based liquefaction triggering evaluation
    (Seed & Idriss 1971, updated per Idriss & Boulanger 2008/NCEER 1997
    conventions): compares Cyclic Stress Ratio (CSR) induced by shaking
    against Cyclic Resistance Ratio (CRR) from SPT blow counts.
    """

    @staticmethod
    def rd_depth_reduction(depth_m: np.ndarray) -> np.ndarray:
        """Idriss (1999) stress reduction coefficient rd(z)."""
        z = np.clip(depth_m, 0.1, 34.0)
        alpha = -1.012 - 1.126 * np.sin(z / 11.73 + 5.133)
        beta = 0.106 + 0.118 * np.sin(z / 11.28 + 5.142)
        rd = np.exp(alpha + beta * np.log(z))
        return np.clip(rd, 0.05, 1.0)

    @staticmethod
    def csr(pga_g: float, sigma_v: np.ndarray, sigma_v_eff: np.ndarray,
            depth_m: np.ndarray, Mw: float = 7.5) -> np.ndarray:
        """Cyclic Stress Ratio at each depth, MSF-corrected to Mw=7.5 equivalent."""
        rd = LiquefactionAssessment.rd_depth_reduction(depth_m)
        csr_raw = 0.65 * pga_g * (sigma_v / sigma_v_eff) * rd
        msf = (Mw / 7.5) ** (-2.56) if Mw > 0 else 1.0
        return csr_raw / msf

    @staticmethod
    def crr_from_spt(N160_cs: np.ndarray, sigma_v_eff: np.ndarray, Pa: float = 101.325e3) -> np.ndarray:
        """
        CRR7.5 from clean-sand-corrected SPT blow count (N1)60cs, using the
        Idriss & Boulanger (2008) SPT triggering correlation.
        """
        N = N160_cs
        crr = np.exp(N / 14.1 + (N / 126) ** 2 - (N / 23.6) ** 3 + (N / 25.4) ** 4 - 2.8)
        return crr

    @classmethod
    def assess_profile(
        cls,
        depths: np.ndarray,          # m
        N160_cs: np.ndarray,         # blow counts, clean-sand corrected
        unit_weight: np.ndarray,     # N/m^3, total unit weight per layer
        water_table_depth: float,    # m
        pga_g: float,                # peak ground accel in units of g
        Mw: float = 7.5,
    ) -> dict:
        """Returns factor of safety FS = CRR/CSR at each depth."""
        depths = np.asarray(depths, dtype=float)
        gamma_w = 9810.0  # N/m^3

        sigma_v = np.cumsum(unit_weight * np.gradient(depths, edge_order=1))
        u_pore = np.clip(depths - water_table_depth, 0, None) * gamma_w
        sigma_v_eff = np.clip(sigma_v - u_pore, 1e3, None)

        csr = cls.csr(pga_g, sigma_v, sigma_v_eff, depths, Mw)
        crr = cls.crr_from_spt(N160_cs, sigma_v_eff)
        FS = crr / np.clip(csr, 1e-6, None)

        return {
            "depths": depths,
            "CSR": csr,
            "CRR": crr,
            "FS": FS,
            "liquefiable": FS < 1.0,
            "sigma_v": sigma_v,
            "sigma_v_eff": sigma_v_eff,
        }


# =====================================================================
# 5. ORCHESTRATOR
# =====================================================================

class SeismicOne:
    """
    Orchestrates the full seismic analysis pipeline:
        bedrock ground motion
            -> SiteResponseLayer (soil amplification)
            -> StructuralResponseLayer (building response, damage)
            -> LiquefactionAssessment (foundation soil, optional)

    Mirrors the ONE Ecosystem convention of a top-level orchestrator class
    wrapping domain-specific layers (cf. cell_population_one.py pattern).
    """

    def __init__(
        self,
        site_layer: Optional[SiteResponseLayer] = None,
        structure_layer: Optional[StructuralResponseLayer] = None,
        liquefaction: Optional[LiquefactionAssessment] = None,
    ):
        self.site_layer = site_layer
        self.structure_layer = structure_layer
        self.liquefaction = liquefaction or LiquefactionAssessment()
        self.gme = GroundMotionEngine()

    def run(
        self,
        bedrock_motion: GroundMotion,
        do_site_response: bool = True,
        do_structure: bool = True,
        spectrum_periods: Optional[np.ndarray] = None,
    ) -> dict:
        results = {"bedrock_motion": bedrock_motion}

        if spectrum_periods is None:
            spectrum_periods = np.concatenate([[0.0], np.geomspace(0.02, 5.0, 60)])

        results["bedrock_spectrum"] = self.gme.response_spectrum(bedrock_motion, spectrum_periods)

        surface_motion = bedrock_motion
        if do_site_response and self.site_layer is not None:
            site_result = self.site_layer.analyze(bedrock_motion)
            results["site_response"] = site_result
            surface_motion = site_result["surface_motion"]
            results["surface_spectrum"] = self.gme.response_spectrum(surface_motion, spectrum_periods)

        results["surface_motion"] = surface_motion

        if do_structure and self.structure_layer is not None:
            struct_result = self.structure_layer.time_history_analysis(surface_motion)
            results["structural_response"] = struct_result

        return results

    def summary(self, results: dict) -> str:
        lines = []
        bm = results["bedrock_motion"]
        lines.append(f"Bedrock PGA: {bm.pga:.3f} m/s^2 ({bm.pga/G_ACCEL:.3f} g)")
        if "site_response" in results:
            sm = results["surface_motion"]
            amp = results["site_response"]["amplification_PGA"]
            lines.append(f"Surface PGA: {sm.pga:.3f} m/s^2 ({sm.pga/G_ACCEL:.3f} g), amplification x{amp:.2f}")
            lines.append(f"Site response iterations: {results['site_response']['iterations']}")
        if "structural_response" in results:
            sr = results["structural_response"]
            for i, (d, dmg) in enumerate(zip(sr["max_drift_ratio"], sr["damage_index"])):
                lines.append(f"  Story {i+1}: max drift ratio={d*100:.2f}%, damage index={dmg:.2f}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Minimal self-test / smoke test
    print(f"seismic_one.py v{__version__} loaded OK")
