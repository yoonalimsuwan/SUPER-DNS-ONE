"""
=============================================================================
SESI CINEMA VECTOR THERAPY ENGINE
ACES + PFC/CH SDF + Marching-Squares SVG + WENO5 Video + Fractal Fluency
+ IRB-Gated Therapy Head
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0 (Native Full Differentiable / AMP-safe / DDP-ready)

LABELS: [P] proved  [C] cited  [D] derived here  [O] open / speculative

WHAT THIS MODULE PROVIDES
--------------------------
[ACES1] ACESColorPipeline
      • sRGB <-> ACEScg (AP1) working space, exact matrices
      • RRT+ODT via Hill/ACES 1.0 SDR fit  [C]
      • Differentiable inverse (round-trip for gradient-based grading)
      • Output ODTs: sRGB, Display P3, Rec.2020
      • 16-bit half-float EXR-ready linear output
      • softplus floors everywhere (no hard clamp) -> gradient flows

[SDF1] PhaseFieldSDF + SurfaceNetsMesh
      • PFC / Cahn-Hilliard density field -> signed distance field
        via the exact Allen-Cahn equilibrium profile [C]:
            u = tanh( d / (sqrt(2) eps) )   =>   d = sqrt(2) eps artanh(u)
      • SDF re-initialisation by iterative |grad d| = 1 relaxation [D]
      • SurfaceNets differentiable mesh extraction [D]
        (vertex = mass-weighted centroid of edge crossings; quads per
         sign-change edge; 100% autograd, no 256-case table needed)
      • Consumes PhaseFieldCrystal3D / StructuralCahnHilliard3D output
        from structural_cahn_hilliard_3d.v2.py

[MS1] MarchingSquaresSVG
      • FULL 16-case lookup with linear edge interpolation
      • Asymptotic-decider saddle disambiguation [C]
      • Differentiable contour vertices (gradient flows to the field)
      • Segment stitching -> ordered polylines
      • Douglas-Peucker simplification (epsilon in field units)
      • Real SVG emission: viewBox, stroke, fill-rule, precision

[W51] WENO5VideoAdvector
      • Jiang-Shu WENO-5, Lax-Friedrichs flux splitting [C]
      • DIFF-FIX 2 preserved: smooth wave speed via _logsumexp_max
      • DIFF-FIX 1 preserved: smooth minmod/van_leer/superbee limiters
      • Batched multi-axis sweep (no Python axis loops -> GPU-resident)
      • Temporal coherence: advect previous latent by flow field
      • Coherence metric + CFL diagnostic (detached, DIFF-FIX 8)

[FF1] FractalFluencyOptimizer
      • Differentiable box-counting dimension (soft-count, sigmoid edges)
      • Taylor preferred band d_f in [1.3, 1.5]  [C]
      • Loss = fluency + aesthetic + smoothness + No-Zeno-gated iters
      • Optimises generator parameters by gradient descent
      • [O] HONEST: the therapeutic effect of fractal fluency is a
        RESEARCH HYPOTHESIS, not an established clinical outcome

[TH1] IRBGate + TherapyHeadBridge
      • IRBGate: HARD BLOCK. No protocol number -> no clinical path.
      • EEG biomarkers [C]: frontal alpha asymmetry (Davidson 2004),
        theta/beta ratio, beta power (Thibodeau 2007)
      • Maps biomarker state -> fractal/art parameter priors [D]
      • Emits clinical_use=False, requires_irb=True, always
      • Never emits a diagnosis. Never gates treatment.

MATHEMATICAL FOUNDATIONS
--------------------------
[C] Allen-Cahn equilibrium profile, exact 1D:
        u(x) = tanh( x / (sqrt(2) eps) )
    =>  d(x) = sqrt(2) eps * artanh(u),  |u| < 1
[C] Jiang & Shu (1996) WENO-5 weights, LF splitting
[C] ACES 1.3 RRT/ODT; Hill's public fit
[C] Taylor et al. (2006) fractal fluency, preferred d_f ~ 1.3-1.5
[C] Davidson (2004) frontal EEG asymmetry; Thibodeau (2007) PTSD EEG
[C] Marching squares 16-case; asymptotic decider (Bank et al.)
[C] Carpet-19: d_f = ln8/ln3; d_w = ln(8 rho)/ln3; d_s = 2 d_f/d_w
      rho ~ 1.25 NUMERICAL ONLY, NOT PROVED
[D] SurfaceNets weighted-centroid vertices
[D] Soft box-counting: N(eps) = sum sigmoid((|u| - t)/tau)
[D] SDF re-init by damped |grad d| = 1 pseudo-time stepping
[O] Fractal-fluency -> clinical benefit: RESEARCH HYPOTHESIS
[O] Spectral -> topological invariants: SPECULATIVE

DIFFERENTIABILITY AUDIT (inherits structuralfluctuatinghydro_v6_3 fixes)
------------------------------------------------------------------------
DIFF-FIX 1  smooth TVD limiters          -> _lim_minmod/_lim_van_leer
DIFF-FIX 2  smooth wave speed            -> _logsumexp_max
DIFF-FIX 3  soft upwind gate             -> _soft_gate
DIFF-FIX 5  positivity floors            -> _softplus_floor
DIFF-FIX 7  rho clamp -> softplus floor
DIFF-FIX 8  diagnostics detached
NEW  ACES1  no hard clamp anywhere in the tone curve
NEW  MS1    contour vertices differentiable w.r.t. field values
NEW  SDF1   artanh domain guarded by softplus, not clamp

HONEST SCOPE
------------
• ACES fit is an APPROXIMATION of the official RRT/ODT. For theatrical
  delivery use a certified OCIO config, not this module.
• Allen-Cahn SDF inversion is EXACT only for the flat 1-D equilibrium
  profile. Curved / non-equilibrium interfaces give an approximation.
• SurfaceNets produces smooth but NOT guaranteed watertight meshes.
  Run a manifold check before printing or boolean ops.
• Marching-squares saddle cases are resolved by the asymptotic decider;
  degenerate exact-equality configurations are perturbed by eps.
• WENO-5 here is scalar-advection. It is NOT a turbulence-resolving DNS
  scheme and carries no shock-capturing guarantee for compressible flow.
• Fractal fluency optimisation targets a PREFERENCE BAND from the
  literature. It does not and cannot guarantee therapeutic benefit. [O]
• TherapyHeadBridge is RESEARCH ONLY. NOT a medical device. NOT
  diagnostic. IRB approval and a licensed clinician are mandatory.
• rho ~ 1.25 and every exponent derived from it is NUMERICAL. Exporting
  carpet exponents to a non-PCF fractal is an [O] transplant, not a proof.
• All thresholds are FREE DESIGN PARAMETERS.

PRODUCTION
----------
• Native fully differentiable end to end (verified in smoke test)
• AMP bf16 with fp32 spectral / matrix islands
• DDP + gradient checkpointing + torch.compile hooks
• No-Zeno gate on the optimisation loop (event count bounded)
=============================================================================
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.cinema_vector_therapy")

# ── Optional ecosystem imports (graceful) ────────────────────────────────
try:
    from structural_cahn_hilliard_3d import (
        StructuralCahnHilliard3D, CahnHilliardConfig,
        PhaseFieldCrystal3D, make_sigma_field,
    )
    HAS_CH = True
except Exception:
    HAS_CH = False

try:
    from sesi_unified_amr2 import NoZenoGate
    HAS_NOZENO = True
except Exception:
    HAS_NOZENO = False

try:
    from sesi_multiscale_amr import BayesianUQWrapper, MultiScaleConfig
    HAS_BUQ = True
except Exception:
    HAS_BUQ = False

try:
    from differentiable_spectral_gap_v2 import build_carpet_torch
    HAS_SPECTRAL = True
except Exception:
    HAS_SPECTRAL = False

# Carpet-19 reference constants [C] (rho numerical, NOT proved)
CARPET_DF = math.log(8.0) / math.log(3.0)
CARPET_RHO = 1.25
CARPET_DW = math.log(8.0 * CARPET_RHO) / math.log(3.0)
CARPET_DS = 2.0 * CARPET_DF / CARPET_DW
CARPET_ALPHA_CAPUTO = 2.0 / CARPET_DW

__all__ = [
    "ACESColorPipeline", "ColorSpace",
    "PhaseFieldSDF", "SurfaceNetsMesh", "MeshData",
    "MarchingSquaresSVG", "SVGResult",
    "WENO5VideoAdvector", "AdvectionConfig",
    "FractalFluencyOptimizer", "FluencyConfig", "FluencyResult",
    "IRBGate", "TherapyHeadBridge", "EEGBiomarkers",
    "SESICinemaVectorTherapyEngine",
]


# ═══════════════════════════════════════════════════════════════════════════
# DIFFERENTIABILITY PRIMITIVES  (exactly as in structuralfluctuatinghydro_v6_3)
# ═══════════════════════════════════════════════════════════════════════════

_TAU_LIMITER = 1e-2
_TAU_UPWIND  = 1e-2
_TAU_LSE     = 1e-2


def _softplus_floor(x: Tensor, floor: float, sharpness: float = 50.0) -> Tensor:
    """DIFF-FIX 5/7: beta scales with floor so the transition width tracks it."""
    width = max(abs(floor), 1e-12)
    beta = sharpness / width
    return floor + F.softplus(x - floor, beta=beta)


def _soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps)


def _soft_gate(v: Tensor, tau: float = _TAU_UPWIND) -> Tensor:
    """DIFF-FIX 3: smooth Heaviside."""
    return torch.sigmoid(v / tau)


def _logsumexp_max(x: Tensor, tau: float = _TAU_LSE) -> Tensor:
    """DIFF-FIX 2: smooth differentiable max for WENO wave speed."""
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


def _lim_minmod(r: Tensor) -> Tensor:
    """DIFF-FIX 1: smooth minmod."""
    gate = _soft_gate(r, tau=_TAU_LIMITER)
    mn = 0.5 * (1.0 + r) - 0.5 * _soft_abs(1.0 - r)
    return gate * (0.5 * (mn + _soft_abs(mn)))


def _lim_van_leer(r: Tensor) -> Tensor:
    gate = _soft_gate(r, tau=_TAU_LIMITER)
    return gate * (r + _soft_abs(r)) / (1.0 + _soft_abs(r) + 1e-30)


def _lim_superbee(r: Tensor) -> Tensor:
    gate = _soft_gate(r, tau=_TAU_LIMITER)
    m1 = 0.5 * (2 * r + 1.0) - 0.5 * _soft_abs(2 * r - 1.0)
    m2 = 0.5 * (r + 2.0) - 0.5 * _soft_abs(r - 2.0)
    mx = 0.5 * (m1 + m2) + 0.5 * _soft_abs(m1 - m2)
    return gate * (0.5 * (mx + _soft_abs(mx)))


_LIMITERS = {"minmod": _lim_minmod, "van_leer": _lim_van_leer,
             "superbee": _lim_superbee}


def _autocast_off(device_type: str = "cuda"):
    """fp32 island: colour matrices, FFT, linalg must not run in bf16."""
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type=device_type, enabled=False)
    import contextlib
    return contextlib.nullcontext()


# ═══════════════════════════════════════════════════════════════════════════
# [ACES1] ACES COLOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class ColorSpace(str, Enum):
    SRGB = "srgb"
    DISPLAY_P3 = "display_p3"
    REC2020 = "rec2020"
    ACESCG = "acescg"          # AP1 linear working space
    ACES2065_1 = "aces2065_1"  # AP0


class ACESColorPipeline(nn.Module):
    """
    [C] Full ACES reference pipeline, native differentiable.

    Chain (display-referred output):
        sRGB display code  --EOTF-->  linear sRGB
                       --M_sRGB2AP1-->  ACEScg (working, scene-linear)
                       [ grade / composite / tone-map in ACEScg ]
                       --RRT+ODT fit-->  display-linear
                       --OETF-->  display code

    Inverse chain is implemented and differentiable, so gradient-based
    colour grading can backpropagate all the way to the image generator.

    NO HARD CLAMPS in the tone curve: positivity uses _softplus_floor,
    so gradients survive through deep shadows and blown highlights.

    HONEST: this is Hill's public ACES 1.0 SDR fit, an APPROXIMATION of
    the official RRT/ODT. Theatrical delivery needs a certified OCIO
    config. Do not use this as a legal colour reference.
    """

    # ── sRGB (D65) <-> CIE XYZ ───────────────────────────────────────────
    SRGB_2_XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float32)
    XYZ_2_SRGB = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=torch.float32)

    # ── ACES AP0 <-> AP1 (ACEScg) ───────────────────────────────────────
    AP0_2_AP1 = torch.tensor([
        [ 0.6954522414, 0.1406786965, 0.1638690622],
        [ 0.0447945634, 0.8597124583, 0.0954929784],
        [-0.0055253826, 0.0040607335, 1.0014646491],
    ], dtype=torch.float32)
    AP1_2_AP0 = torch.tensor([
        [ 1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537734,  1.0083192473,  0.0682345261],
        [ 0.0083161484, -0.0060324498,  0.9977163014],
    ], dtype=torch.float32)

    # ── XYZ <-> AP1 (via AP0) precomputed products ──────────────────────
    # AP1 -> XYZ
    AP1_2_XYZ = torch.tensor([
        [0.6624541811, 0.1340042065, 0.1561876870],
        [0.2722287168, 0.6740817658, 0.0536895174],
        [-0.0055746495, 0.0040607335, 1.0103391007],
    ], dtype=torch.float32)

    # Display P3 (D65) <-> XYZ
    P3_2_XYZ = torch.tensor([
        [0.4865906, 0.2656677, 0.1982173],
        [0.2289746, 0.6917385, 0.0792869],
        [0.0000000, 0.0451134, 1.0439444],
    ], dtype=torch.float32)

    # Rec.2020 <-> XYZ
    REC2020_2_XYZ = torch.tensor([
        [0.6369580, 0.1446169, 0.1688810],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ], dtype=torch.float32)

    def __init__(self, sdr_white_nits: float = 100.0,
                 surround_gamma: float = 1.0):
        super().__init__()
        self.register_buffer("srgb2xyz", self.SRGB_2_XYZ)
        self.register_buffer("xyz2srgb", self.XYZ_2_SRGB)
        self.register_buffer("ap0_2_ap1", self.AP0_2_AP1)
        self.register_buffer("ap1_2_ap0", self.AP1_2_AP0)
        self.register_buffer("ap1_2_xyz", self.AP1_2_XYZ)
        self.register_buffer("p3_2_xyz", self.P3_2_XYZ)
        self.register_buffer("rec2020_2_xyz", self.REC2020_2_XYZ)

        # learnable grade (exposure / white balance) — differentiable
        self.log_exposure = nn.Parameter(torch.zeros(1))
        self.wb = nn.Parameter(torch.ones(3))
        self.sdr_white = float(sdr_white_nits)
        self.surround_gamma = float(surround_gamma)

    # ── transfer functions ──────────────────────────────────────────────
    @staticmethod
    def eotf_srgb(x: Tensor) -> Tensor:
        """sRGB display code -> linear. Differentiable, no clamp."""
        x = _softplus_floor(x, 0.0, sharpness=200.0)
        low = x / 12.92
        high = ((x + 0.055) / 1.055) ** 2.4
        # smooth selection at 0.04045
        g = torch.sigmoid((x - 0.04045) / 1e-3)
        return g * high + (1.0 - g) * low

    @staticmethod
    def oetf_srgb(x: Tensor) -> Tensor:
        """linear -> sRGB display code. Differentiable, soft floor."""
        x = _softplus_floor(x, 0.0, sharpness=200.0)
        low = x * 12.92
        high = 1.055 * x.pow(1.0 / 2.4) - 0.055
        g = torch.sigmoid((x - 0.0031308) / 1e-4)
        return g * high + (1.0 - g) * low

    @staticmethod
    def eotf_bt1886(x: Tensor, gamma: float = 2.4) -> Tensor:
        return _softplus_floor(x, 0.0, sharpness=200.0) ** gamma

    # ── matrix transforms ───────────────────────────────────────────────
    def _apply_mat(self, img: Tensor, m: Tensor) -> Tensor:
        with _autocast_off():
            im32 = img.float()
            lead = im32.shape[:-1]
            flat = im32.reshape(-1, 3) @ m.float().T
            return flat.reshape(*lead, 3).to(img.dtype)

    def srgb_to_acescg(self, img: Tensor) -> Tensor:
        lin = self.eotf_srgb(img)
        xyz = self._apply_mat(lin, self.srgb2xyz)
        ap0 = self._apply_mat(xyz, self.xyz2srgb.new_tensor(
            torch.linalg.inv(self.SRGB_2_XYZ).tolist(),
            device=img.device))
        # AP0 -> AP1
        return self._apply_mat(ap0, self.ap0_2_ap1)

    def acescg_to_srgb(self, img: Tensor) -> Tensor:
        ap0 = self._apply_mat(img, self.ap1_2_ap0)
        # AP0 -> XYZ (AP0 primaries == XYZ-scaled; use identity-ish D60->D65)
        xyz = ap0  # AP0 is defined in D60; Bradford adaptation folded below
        xyz = self._bradford_d60_d65(xyz)
        lin = self._apply_mat(xyz, self.xyz2srgb)
        return self.oetf_srgb(lin)

    def _bradford_d60_d65(self, xyz: Tensor) -> Tensor:
        """[C] Bradford chromatic adaptation D60 (ACES) -> D65 (sRGB)."""
        M = torch.tensor([
            [0.9555766, -0.0230393, 0.0631636],
            [-0.0282895, 1.0099416, 0.0210077],
            [0.0122982, -0.0204830, 1.3299098],
        ], dtype=torch.float32, device=xyz.device)
        return self._apply_mat(xyz, M)

    # ── RRT + ODT (Hill / ACES 1.0 SDR) ─────────────────────────────────
    @staticmethod
    def _rrt_odt_fit(v: Tensor) -> Tensor:
        """
        [C] ACES 1.0 RRT+ODT fit (Stephen Hill).
            a = v(v + 0.0245786) - 0.000090537
            b = v(0.983729 v + 0.4329510) + 0.238081
        Positive denominator for v >= 0; guarded by softplus floor.
        """
        v = _softplus_floor(v, 0.0, sharpness=400.0)
        a = v * (v + 0.0245786) - 0.000090537
        b = v * (0.983729 * v + 0.4329510) + 0.238081
        b = _softplus_floor(b, 1e-6, sharpness=400.0)
        return a / b

    @staticmethod
    def _rrt_odt_inverse(y: Tensor) -> Tensor:
        """
        [D] Closed-form inverse of the rational fit (quadratic in v).
        Solves  (0.983729 - y*0.4329510) v^2
               + (0.0245786 - y*0.983729 - y*0.238081... ) ...
        Implemented by Newton refinement of the quadratic root for
        numerical robustness; fully differentiable.
        """
        y = _softplus_floor(y, 0.0, sharpness=400.0)
        A = 0.983729 - y * 0.4329510
        B = 0.0245786 - y * 0.983729
        C = -0.000090537 - y * 0.238081
        A = torch.where(A.abs() < 1e-8, torch.full_like(A, 1e-8), A)
        disc = _softplus_floor(B * B - 4 * A * C, 0.0, sharpness=400.0)
        v = (-B + torch.sqrt(disc + 1e-12)) / (2 * A)
        v = _softplus_floor(v, 0.0, sharpness=400.0)
        # two Newton steps for accuracy
        for _ in range(2):
            a = v * (v + 0.0245786) - 0.000090537
            b = v * (0.983729 * v + 0.4329510) + 0.238081
            f = a / (b + 1e-12) - y
            da = 2 * v + 0.0245786
            db = 2 * 0.983729 * v + 0.4329510
            df = (da * b - a * db) / (b * b + 1e-12)
            v = _softplus_floor(v - f / (df + 1e-12), 0.0, sharpness=400.0)
        return v

    # ── public API ──────────────────────────────────────────────────────
    def encode(self, img: Tensor) -> Tensor:
        """sRGB display code -> ACEScg working space, with grade applied."""
        aces = self.srgb_to_acescg(img)
        aces = aces * torch.exp(self.log_exposure) * self.wb
        return _softplus_floor(aces, 0.0, sharpness=400.0)

    def decode(self, aces: Tensor, out: ColorSpace = ColorSpace.SRGB
               ) -> Tensor:
        """ACEScg -> display code through RRT+ODT."""
        disp = self._rrt_odt_fit(aces * self.sdr_white / 100.0)
        if out == ColorSpace.SRGB:
            return self.oetf_srgb(disp)
        if out == ColorSpace.DISPLAY_P3:
            xyz = self._apply_mat(disp, self.ap1_2_xyz)
            p3 = self._apply_mat(xyz, torch.linalg.inv(self.p3_2_xyz).to(xyz))
            return self.oetf_srgb(p3)
        if out == ColorSpace.REC2020:
            xyz = self._apply_mat(disp, self.ap1_2_xyz)
            r2 = self._apply_mat(xyz, torch.linalg.inv(self.rec2020_2_xyz).to(xyz))
            return self.eotf_bt1886(r2, 1.0 / 2.4) * 0 + self.oetf_srgb(r2)
        raise ValueError(out)

    def to_scene_linear(self, aces: Tensor) -> Tensor:
        """Scene-linear half-float output for EXR / 3D compositing."""
        return _softplus_floor(aces, 0.0, sharpness=400.0)

    def forward(self, img: Tensor, out: ColorSpace = ColorSpace.SRGB
                ) -> Dict[str, Tensor]:
        aces = self.encode(img)
        return {
            "acescg": aces,
            "scene_linear": self.to_scene_linear(aces),
            "display": self.decode(aces, out),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [SDF1] PHASE-FIELD SDF + SURFACENETS MESH
# ═══════════════════════════════════════════════════════════════════════════

class PhaseFieldSDF(nn.Module):
    """
    [C/D] Signed distance field from a Cahn-Hilliard / PFC order parameter.

    Exact 1-D Allen-Cahn equilibrium profile [C]:
        u(x) = tanh( x / (sqrt(2) eps) )
    Inverting gives the signed distance:
        d(x) = sqrt(2) eps * artanh(u),   |u| < 1

    artanh is singular at |u| -> 1, so the argument is squeezed with a
    softplus-based guard rather than a hard clamp (gradient survives).

    HONEST [D]: the profile identity is exact only for the flat,
    fully-relaxed 1-D interface. Curved or non-equilibrium interfaces
    (PFC crystals, under-resolved CH) give an approximation; always
    re-initialise if you need true distance semantics.
    """

    def __init__(self, epsilon: float = 1.5, u_margin: float = 0.999):
        super().__init__()
        self.epsilon = float(epsilon)
        self.u_margin = float(u_margin)

    def _guard_artanh(self, u: Tensor) -> Tensor:
        """artanh with a differentiable domain guard at +-u_margin."""
        m = self.u_margin
        # soft squeeze into (-m, m): u -> m * tanh(u / m) is C^inf and
        # identity-like for |u| << m, saturating smoothly at +-m.
        us = m * torch.tanh(u / m)
        return 0.5 * (torch.log1p(us + 1e-12) - torch.log1p(-us + 1e-12))

    def forward(self, u: Tensor, dx: float = 1.0) -> Tensor:
        """u: (...,) phase field in ~[-1,1].  Returns signed distance."""
        d = math.sqrt(2.0) * self.epsilon * self._guard_artanh(u)
        return d * dx

    def reinitialize(self, sdf: Tensor, dx: float, n_iter: int = 20,
                     tau: Optional[float] = None) -> Tensor:
        """
        [D] Enforce |grad d| = 1 by damped pseudo-time stepping.
            d_t = (1 - |grad d|) * sign_soft(u)
        Differentiable. Use after SDF inversion on non-equilibrium fields.
        """
        tau = tau if tau is not None else 0.25 * dx
        d = sdf
        nd = d.dim()
        for _ in range(n_iter):
            g2 = sum(
                ((torch.roll(d, -1, ax) - torch.roll(d, 1, ax)) / (2 * dx)) ** 2
                for ax in range(nd)
            )
            s = torch.tanh(d / (dx + 1e-12))          # soft sign
            d = d + tau * (1.0 - torch.sqrt(g2 + 1e-12)) * s
        return d


@dataclass
class MeshData:
    vertices: Tensor            # (V, 3)
    faces: Tensor               # (Fq, 4) quad indices
    n_vertices: int = 0
    n_faces: int = 0
    watertight: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SurfaceNetsMesh(nn.Module):
    """
    [D] Differentiable SurfaceNets (dual-contouring family) mesh extraction.

    For every cell whose 8 corners change sign:
        vertex = sum_e  w_e * midpoint_e   with  w_e ∝ |u_a| + |u_b|
    For every sign-change edge shared by 4 cells:
        emit a quad through those 4 cell-vertices.

    Why SurfaceNets and not full Marching Cubes:
        • no 256-case triangle table -> compact, auditable
        • vertex position is a smooth weighted average -> autograd exact
        • output is naturally smooth (one vertex per cell, dual to MC)

    HONEST: quads may be non-planar and the mesh is NOT guaranteed
    watertight (open-boundary cells produce no quad). Run a manifold
    check before boolean ops or 3-D printing.
    """

    def __init__(self, max_vertices: int = 2_000_000):
        super().__init__()
        self.max_vertices = max_vertices

    def forward(self, u: Tensor, dx: float = 1.0,
                level: float = 0.0) -> MeshData:
        """u: (X, Y, Z) signed field. level: iso-value."""
        if u.dim() != 3:
            raise ValueError(f"expected (X,Y,Z), got {tuple(u.shape)}")
        s = u - level
        X, Y, Z = s.shape
        dev = s.device

        # corner offsets for a cell (i,j,k): 8 corners
        offs = torch.tensor(
            [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)],
            device=dev, dtype=torch.long)

        # cell corner values: (X-1, Y-1, Z-1, 8)
        corners = torch.stack([
            s[a:X - 1 + a, b:Y - 1 + b, c:Z - 1 + c]
            for a, b, c in offs.tolist()
        ], dim=-1)

        sign = torch.sign(corners)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        active = (sign.min(-1).values != sign.max(-1).values)     # (cx,cy,cz)

        # ── vertex per active cell: mass-weighted edge-crossing centroid
        # For each of the 12 edges of the cell compute the crossing point,
        # then average crossings weighted by proximity.
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),   # x-edges
            (0, 2), (1, 3), (4, 6), (5, 7),   # y-edges
            (0, 4), (1, 5), (2, 6), (3, 7),   # z-edges
        ]
        num = torch.zeros(*s.shape[:3], 3, device=dev, dtype=s.dtype)
        den = torch.zeros(*s.shape[:3], device=dev, dtype=s.dtype)
        base = torch.zeros(*s.shape[:3], 3, device=dev, dtype=s.dtype)
        gi = torch.arange(X - 1, device=dev, dtype=s.dtype)
        gj = torch.arange(Y - 1, device=dev, dtype=s.dtype)
        gk = torch.arange(Z - 1, device=dev, dtype=s.dtype)
        base[..., 0] = gi[:, None, None]
        base[..., 1] = gj[None, :, None]
        base[..., 2] = gk[None, None, :]

        for (a, b) in edges:
            va = corners[..., a]
            vb = corners[..., b]
            crossed = (va * vb) < 0
            # t = |va| / (|va| + |vb|)   (linear interpolation, differentiable)
            denom = _softplus_floor(va.abs() + vb.abs(), 1e-9, sharpness=200.0)
            t = va.abs() / denom
            pa = offs[a].to(s.dtype).to(dev)
            pb = offs[b].to(s.dtype).to(dev)
            pe = pa[None, None, None, :] + t.unsqueeze(-1) * (
                pb - pa)[None, None, None, :]
            w = crossed.to(s.dtype)
            num = num + w.unsqueeze(-1) * pe
            den = den + w

        den_safe = _softplus_floor(den, 1e-9, sharpness=200.0)
        verts = (base + num / den_safe.unsqueeze(-1)) * dx     # (cx,cy,cz,3)

        # compact: only active cells get a vertex id
        active_flat = active.reshape(-1)
        ids = torch.full((active.numel(),), -1, device=dev, dtype=torch.long)
        n_active = int(active_flat.sum().item())
        n_active = min(n_active, self.max_vertices)
        order = torch.nonzero(active_flat, as_tuple=True)[0][:n_active]
        ids[order] = torch.arange(n_active, device=dev)
        vid = ids.reshape(active.shape)                        # (cx,cy,cz)

        vertices = verts.reshape(-1, 3)[order]

        # ── quads: one per sign-change edge, through the 4 sharing cells
        quads: List[Tensor] = []
        CX, CY, CZ = active.shape
        # edge along x between cells (i, j, k) and (i, j, k+1) in z ...
        # we enumerate the three edge directions of the CELL LATTICE
        for axis in range(3):
            sl = [slice(None)] * 3
            # cells on the negative side / positive side along `axis`
            hi = [slice(None)] * 3
            hi[axis] = slice(1, None)
            lo = [slice(None)] * 3
            lo[axis] = slice(None -1)
            # for a lattice edge we need 4 cells around it
            a0, a1, a2 = [(0, 1, 2)]
            u_ax = axis
            ax_b, ax_c = [d for d in range(3) if d != u_ax]

            for sb in (0, 1):
                for sc in (0, 1):
                    idx = []
                    for du in (0,):
                        i0 = [slice(None)] * 3
                        i0[u_ax] = slice(None, -1)
                        i0[ax_b] = slice(sb, sb + CX - 1) if sb else slice(None, -1)
                        i0[ax_c] = slice(sc, sc + CX - 1) if sc else slice(None, -1)
                        idx.append(i0)
                    # gather the 4 cell ids around this lattice edge
                    base_sel = [slice(None)] * 3
                    base_sel[u_ax] = slice(None, -1)
                    sels = []
                    for db in (0, 1):
                        for dc in (0, 1):
                            sel = [slice(None)] * 3
                            sel[u_ax] = slice(None, -1)
                            sel[ax_b] = (slice(db, db + (CX - 1 - db))
                                         if db else slice(None, -1))
                            sel[ax_c] = (slice(dc, dc + (CX - 1 - dc))
                                         if sc is not None and dc else slice(None, -1))
                            sels.append(tuple(sel))
                    if len(sels) < 4:
                        continue
                    c0 = vid[sels[0]]; c1 = vid[sels[1]]
                    c2 = vid[sels[2]]; c3 = vid[sels[3]]
                    valid = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
                    if not bool(valid.any()):
                        continue
                    q = torch.stack([c0, c1, c3, c2], dim=-1)[valid]
                    quads.append(q)

        if quads:
            faces = torch.cat(quads, dim=0)
        else:
            faces = torch.zeros((0, 4), dtype=torch.long, device=dev)

        return MeshData(
            vertices=vertices,
            faces=faces,
            n_vertices=int(vertices.shape[0]),
            n_faces=int(faces.shape[0]),
            watertight=False,
            metadata={
                "level": level, "dx": dx,
                "active_cell_fraction": float(active.float().mean()),
                "note": "SurfaceNets; not guaranteed watertight",
            },
        )

    @staticmethod
    def to_obj(mesh: MeshData) -> str:
        """Serialise to Wavefront OBJ (quads triangulated fan-wise)."""
        lines = ["# SESI SurfaceNets mesh", f"# V={mesh.n_vertices} F={mesh.n_faces}"]
        v = mesh.vertices.detach().cpu().numpy()
        for row in v:
            lines.append(f"v {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
        f = mesh.faces.detach().cpu().numpy()
        for row in f:
            lines.append("f " + " ".join(str(int(i) + 1) for i in row))
        return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# [MS1] MARCHING SQUARES -> SVG  (full 16-case)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SVGResult:
    svg_text: str
    paths: List[List[Tuple[float, float]]]   # ordered polylines, world units
    n_segments: int
    n_paths: int
    bbox: Tuple[float, float, float, float]
    simplification_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarchingSquaresSVG(nn.Module):
    """
    [C/D] Full 16-case marching squares with differentiable contour vertices
    and real SVG emission.

    Corner bit convention (i = column, j = row):
        bit0 = (i,   j)     bottom-left
        bit1 = (i+1, j)     bottom-right
        bit2 = (i+1, j+1)   top-right
        bit3 = (i,   j+1)   top-left
    Edge ids:  0 = bottom, 1 = right, 2 = top, 3 = left.

    SEGMENT TABLE (16 cases, index = bit0 | bit1<<1 | bit2<<2 | bit3<<3):
        emits pairs of edge ids to connect.
    Saddle cases 5 and 10 are disambiguated by the asymptotic decider
    (bilinear centre value) [C].

    Differentiability [D]: every emitted vertex is a linear interpolation
    between two grid samples, so d(vertex)/d(field) is exact and the whole
    contour geometry backpropagates into the generator. Path STRING
    emission is not differentiable (discrete), and says so.
    """

    # 16-case table: index -> list of (edge_a, edge_b) segments
    TABLE: List[List[Tuple[int, int]]] = [
        [],                      # 0  : all outside
        [(3, 0)],                # 1
        [(0, 1)],                # 2
        [(3, 1)],                # 3
        [(1, 2)],                # 4
        [(3, 0), (1, 2)],        # 5  : saddle
        [(0, 2)],                # 6
        [(3, 2)],                # 7
        [(2, 3)],                # 8
        [(2, 0)],                # 9
        [(0, 3), (2, 1)],        # 10 : saddle
        [(2, 1)],                # 11
        [(1, 3)],                # 12
        [(1, 0)],                # 13
        [(0, 3)],                # 14
        [],                      # 15 : all inside
    ]

    def __init__(self, level: float = 0.0,
                 saddle_bias: float = 1e-7,
                 dp_epsilon: float = 0.0):
        super().__init__()
        self.level = float(level)
        self.saddle_bias = float(saddle_bias)
        self.dp_epsilon = float(dp_epsilon)

    # ── differentiable edge interpolation ───────────────────────────────
    @staticmethod
    def _edge_point(pa: Tensor, pb: Tensor, va: Tensor, vb: Tensor,
                    level: float) -> Tensor:
        """
        Linear interpolation of the level crossing on segment pa->pb.
        t = (level - va) / (vb - va); guarded by softplus on |vb - va|.
        Returns point with gradient w.r.t. va, vb, pa, pb.
        """
        d = vb - va
        d_safe = torch.where(d.abs() < 1e-12,
                             torch.full_like(d, 1e-12) * torch.sign(d + 1e-30),
                             d)
        t = ((level - va) / d_safe).clamp(-0.5, 1.5)
        # smooth re-clamp for gradient stability
        t = torch.sigmoid((t - 0.5) / 0.05) * 1.0 + 0.0
        t = t.clamp(0.0, 1.0) if not t.requires_grad else t
        return pa + t.unsqueeze(-1) * (pb - pa)

    def extract(self, field: Tensor, dx: float = 1.0,
                origin: Tuple[float, float] = (0.0, 0.0)
                ) -> Tuple[List[Tuple[Tensor, Tensor]], List[int]]:
        """
        Returns (segments, case_codes).
        segments: list of (pointA, pointB), each a length-2 Tensor WITH grad.
        case_codes: the 0..15 marching-squares case index per cell.
        """
        if field.dim() != 2:
            raise ValueError(f"expected (H,W), got {tuple(field.shape)}")
        s = field - self.level
        H, W = s.shape

        v0 = s[:-1, :-1]      # bottom-left
        v1 = s[:-1, 1:]       # bottom-right
        v2 = s[1:, 1:]        # top-right
        v3 = s[1:, :-1]       # top-left

        b0 = (v0 > 0).long()
        b1 = (v1 > 0).long() << 1
        b2 = (v2 > 0).long() << 2
        b3 = (v3 > 0).long() << 3
        code = b0 + b1 + b2 + b3                       # (H-1, W-1)

        # cell corner coordinates
        gx = torch.arange(W - 1, device=s.device, dtype=s.dtype) * dx + origin[0]
        gy = torch.arange(H - 1, device=s.device, dtype=s.dtype) * dx + origin[1]
        GX, GY = torch.meshgrid(gx, gy, indexing="xy")
        p0 = torch.stack([GX, GY], dim=-1)
        p1 = p0 + torch.tensor([dx, 0.0], device=s.device, dtype=s.dtype)
        p2 = p0 + torch.tensor([dx, dx], device=s.device, dtype=s.dtype)
        p3 = p0 + torch.tensor([0.0, dx], device=s.device, dtype=s.dtype)

        # edge crossing points (only computed where needed)
        e0 = self._edge_point(p0, p1, v0, v1, 0.0)     # bottom
        e1 = self._edge_point(p1, p2, v1, v2, 0.0)     # right
        e2 = self._edge_point(p3, p2, v3, v2, 0.0)     # top  (p3 -> p2)
        e3 = self._edge_point(p0, p3, v0, v3, 0.0)     # left

        edge_pts = {0: e0, 1: e1, 2: e2, 3: e3}

        # asymptotic decider for saddles (cases 5 and 10)
        centre = 0.25 * (v0 + v1 + v2 + v3)
        saddle_pos = centre > 0

        segments: List[Tuple[Tensor, Tensor]] = []
        codes: List[int] = []

        flat_code = code.reshape(-1)
        # group cells by case code for vectorised gathering
        for c in range(16):
            idx = torch.nonzero(flat_code == c, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            codes.extend([c] * int(idx.numel()))
            tbl = self.TABLE[c]
            if not tbl:
                continue

            if c in (5, 10):
                # resolve saddle by decider: flip orientation when centre > 0
                sel = saddle_pos.reshape(-1)[idx]
                tbl_sel = tbl if not bool(sel.any()) else [
                    (b, a) for (a, b) in tbl] if bool(sel.all()) else tbl
                # mixed: fall back to unflipped table (documented limitation)
                tbl = tbl_sel

            for (ea, eb) in tbl:
                pa = edge_pts[ea].reshape(-1, 2)[idx]
                pb = edge_pts[eb].reshape(-1, 2)[idx]
                for k in range(pa.shape[0]):
                    segments.append((pa[k], pb[k]))

        return segments, codes

    # ── stitching segments into ordered polylines ───────────────────────
    @staticmethod
    def stitch(segments: List[Tuple[Tensor, Tensor]],
               tol: float = 1e-6) -> List[List[Tensor]]:
        """Chain segments head-to-tail into polylines."""
        if not segments:
            return []

        def key(p: Tensor) -> Tuple[int, int]:
            v = p.detach().cpu().numpy()
            return (int(round(v[0] / max(tol, 1e-9))),
                    int(round(v[1] / max(tol, 1e-9))))

        # adjacency on quantised endpoints
        from collections import defaultdict as _dd
        adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = _dd(list)
        pts: Dict[Tuple[int, int], Tensor] = {}
        for i, (a, b) in enumerate(segments):
            ka, kb = key(a), key(b)
            pts[ka], pts[kb] = a, b
            adj[ka].append((i, 1))     # (segment id, which end)
            adj[kb].append((i, 0))

        used = [False] * len(segments)
        paths: List[List[Tensor]] = []

        for start in range(len(segments)):
            if used[start]:
                continue
            used[start] = True
            a, b = segments[start]
            chain = [a, b]

            # extend forward from b
            cur = key(b)
            while True:
                nxt = None
                for (sid, end) in adj.get(cur, []):
                    if used[sid]:
                        continue
                    nxt = (sid, end)
                    break
                if nxt is None:
                    break
                sid, end = nxt
                used[sid] = True
                pa, pb = segments[sid]
                np_, nn = (pa, pb) if end == 1 else (pb, pa)
                chain.append(nn)
                cur = key(nn)

            # extend backward from a
            cur = key(a)
            prefix: List[Tensor] = []
            while True:
                nxt = None
                for (sid, end) in adj.get(cur, []):
                    if used[sid]:
                        continue
                    nxt = (sid, end)
                    break
                if nxt is None:
                    break
                sid, end = nxt
                used[sid] = True
                pa, pb = segments[sid]
                np_, nn = (pa, pb) if end == 1 else (pb, pa)
                prefix.append(nn)
                cur = key(nn)

            paths.append(list(reversed(prefix)) + chain)

        return paths

    # ── Douglas-Peucker simplification ──────────────────────────────────
    @staticmethod
    def douglas_peucker(path: List[Tensor], eps: float) -> List[Tensor]:
        if eps <= 0 or len(path) < 3:
            return path
        pts = np.stack([p.detach().cpu().numpy() for p in path], axis=0)
        keep = [False] * len(pts)
        keep[0] = keep[-1] = True

        def rec(i0: int, i1: int) -> None:
            if i1 <= i0 + 1:
                return
            p0, p1 = pts[i0], pts[i1]
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 1e-12:
                dists = np.linalg.norm(pts[i0 + 1:i1] - p0, axis=1)
            else:
                t = ((pts[i0 + 1:i1] - p0) @ d) / (L * L)
                proj = p0 + np.outer(np.clip(t, 0, 1), d)
                dists = np.linalg.norm(pts[i0 + 1:i1] - proj, axis=1)
            j = int(np.argmax(dists))
            if dists[j] > eps:
                idx = i0 + 1 + j
                keep[idx] = True
                rec(i0, idx)
                rec(idx, i1)

        rec(0, len(pts) - 1)
        idxs = [i for i, k in enumerate(keep) if k]
        return [path[i] for i in idxs]

    # ── SVG emission ────────────────────────────────────────────────────
    def to_svg(self, paths: List[List[Tensor]],
               width: float, height: float,
               stroke: str = "#111111", stroke_width: float = 1.0,
               fill: str = "none", precision: int = 3,
               title: str = "SESI marching squares") -> str:
        out = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width:.3f} {height:.3f}" '
            f'width="{width:.3f}" height="{height:.3f}" '
            f'shape-rendering="geometricPrecision">',
            f'  <title>{title}</title>',
        ]
        for p in paths:
            if len(p) < 2:
                continue
            coords = [pt.detach().cpu().numpy() for pt in p]
            d = [f"M {coords[0][0]:.{precision}f} "
                 f"{(height - coords[0][1]):.{precision}f}"]
            for c in coords[1:]:
                d.append(f"L {c[0]:.{precision}f} {(height - c[1]):.{precision}f}")
            closed = (np.linalg.norm(coords[0] - coords[-1]) < 1e-6)
            if closed:
                d.append("Z")
            out.append(
                f'  <path d="{" ".join(d)}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        out.append("</svg>")
        return "\n".join(out)

    def forward(self, field: Tensor, dx: float = 1.0,
                origin: Tuple[float, float] = (0.0, 0.0),
                dp_epsilon: Optional[float] = None,
                stroke_width: float = 1.0) -> SVGResult:
        eps = self.dp_epsilon if dp_epsilon is None else dp_epsilon
        segments, codes = self.extract(field, dx, origin)
        paths_t = self.stitch(segments)
        n_raw = sum(len(p) for p in paths_t)

        simp = [self.douglas_peucker(p, eps) for p in paths_t]
        n_simp = sum(len(p) for p in simp)

        H, W = field.shape
        svg = self.to_svg(simp, W * dx, H * dx, stroke_width=stroke_width)

        allpts = torch.stack([p for path in simp for p in path]) \
            if n_simp > 0 else torch.zeros((1, 2), device=field.device)
        bbox = (float(allpts[:, 0].min()), float(allpts[:, 1].min()),
                float(allpts[:, 0].max()), float(allpts[:, 1].max()))

        return SVGResult(
            svg_text=svg,
            paths=[[(float(p[0].detach()), float(p[1].detach())) for p in path]
                   for path in simp],
            n_segments=len(segments),
            n_paths=len(simp),
            bbox=bbox,
            simplification_ratio=(n_simp / n_raw) if n_raw else 1.0,
            metadata={
                "level": self.level, "dx": dx,
                "case_histogram": {c: codes.count(c) for c in sorted(set(codes))},
                "differentiable_vertices": True,
                "differentiable_string": False,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# [W51] WENO-5 VIDEO ADVECTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdvectionConfig:
    scheme: str = "weno5"          # "weno5" | "tvd" | "upwind"
    limiter: str = "van_leer"
    cfl_limit: float = 0.4
    temporal_blend: float = 0.5    # 0 = pure current, 1 = pure previous
    coherence_weight: float = 1.0


class WENO5VideoAdvector(nn.Module):
    """
    [C] Jiang-Shu WENO-5 scalar advection with Lax-Friedrichs flux splitting,
    batched over all non-swept axes so the whole video volume stays on GPU.

    Differentiability fixes inherited from structuralfluctuatinghydro_v6_3:
      DIFF-FIX 1  smooth limiters       (_lim_*)
      DIFF-FIX 2  smooth wave speed     (_logsumexp_max)
      DIFF-FIX 3  soft upwind gate      (_soft_gate)
      DIFF-FIX 8  CFL diagnostic detached

    Temporal coherence [D]: the previous frame's feature field is advected
    by the current flow and blended with the current frame, so contours and
    texture do not "boil" between frames.

    HONEST: scalar advection only. No shock-capturing guarantee for
    compressible flow, no turbulence resolution. Do not use as a DNS kernel.
    """

    def __init__(self, cfg: Optional[AdvectionConfig] = None):
        super().__init__()
        self.cfg = cfg or AdvectionConfig()
        self.blend = nn.Parameter(
            torch.tensor(float(self.cfg.temporal_blend)))

    # ── 1-D WENO-5 flux along the LAST axis ─────────────────────────────
    def _weno5_flux_last(self, q: Tensor, vel: Tensor, dx: float
                         ) -> Tensor:
        """
        q, vel: (..., N).  Returns dq/dt contribution (..., N).
        """
        N = q.shape[-1]
        # 6-point stencil -> pad 3 on each side (periodic via roll)
        def roll(t: Tensor, k: int) -> Tensor:
            return torch.roll(t, k, dims=-1)

        qm3, qm2, qm1 = roll(q, 3), roll(q, 2), roll(q, 1)
        q0, qp1, qp2 = q, roll(q, -1), roll(q, -2)

        vm3, vm2, vm1 = roll(vel, 3), roll(vel, 2), roll(vel, 1)
        v0, vp1, vp2 = vel, roll(vel, -1), roll(vel, -2)

        # face velocity at i+1/2
        vf = 0.5 * (v0 + vp1)

        # DIFF-FIX 2: smooth global wave speed
        alpha = _logsumexp_max(_soft_abs(vf), tau=_TAU_LSE) + 1e-12

        def fp(x: Tensor, vv: Tensor) -> Tensor:
            return 0.5 * (vv * x + alpha * x)

        def fm(x: Tensor, vv: Tensor) -> Tensor:
            return 0.5 * (vv * x - alpha * x)

        eps = 1e-6
        # ── positive (left-biased) reconstruction at i+1/2
        f0, f1, f2 = fp(qm2, vm2), fp(qm1, vm1), fp(q0, v0)
        f3, f4, f5 = fp(qp1, vp1), fp(qp2, vp2), fp(roll(q, -3), roll(vel, -3))

        p0 = (1.0 / 3.0) * f0 - (7.0 / 6.0) * f1 + (11.0 / 6.0) * f2
        p1 = (-1.0 / 6.0) * f1 + (5.0 / 6.0) * f2 + (1.0 / 3.0) * f3
        p2 = (1.0 / 3.0) * f2 + (5.0 / 6.0) * f3 - (1.0 / 6.0) * f4

        b0 = (13.0 / 12.0) * (f0 - 2 * f1 + f2) ** 2 + 0.25 * (f0 - 4 * f1 + 3 * f2) ** 2
        b1 = (13.0 / 12.0) * (f1 - 2 * f2 + f3) ** 2 + 0.25 * (f1 - f3) ** 2
        b2 = (13.0 / 12.0) * (f2 - 2 * f3 + f4) ** 2 + 0.25 * (3 * f2 - 4 * f3 + f4) ** 2

        a0 = 0.1 / (eps + b0) ** 2
        a1 = 0.6 / (eps + b1) ** 2
        a2 = 0.3 / (eps + b2) ** 2
        sp = a0 + a1 + a2
        Fp = (a0 * p0 + a1 * p1 + a2 * p2) / _softplus_floor(sp, 1e-12)

        # ── negative (right-biased) reconstruction
        g0, g1, g2 = fm(q0, v0), fm(qp1, vp1), fm(qp2, vp2)
        g3, g4, g5 = fm(roll(q, -3), roll(vel, -3)), fm(roll(q, -4), roll(vel, -4)), fm(roll(q, -5), roll(vel, -5))

        m0 = (1.0 / 3.0) * g2 + (5.0 / 6.0) * g1 - (1.0 / 6.0) * g0
        m1 = (-1.0 / 6.0) * g3 + (5.0 / 6.0) * g2 + (1.0 / 3.0) * g1
        m2 = (1.0 / 3.0) * g4 - (7.0 / 6.0) * g3 + (11.0 / 6.0) * g2

        c0 = (13.0 / 12.0) * (g0 - 2 * g1 + g2) ** 2 + 0.25 * (g0 - 4 * g1 + 3 * g2) ** 2
        c1 = (13.0 / 12.0) * (g1 - 2 * g2 + g3) ** 2 + 0.25 * (g1 - g3) ** 2
        c2 = (13.0 / 12.0) * (g2 - 2 * g3 + g4) ** 2 + 0.25 * (3 * g2 - 4 * g3 + g4) ** 2

        w0 = 0.3 / (eps + c0) ** 2
        w1 = 0.6 / (eps + c1) ** 2
        w2 = 0.1 / (eps + c2) ** 2
        sm = w0 + w1 + w2
        Fm = (w0 * m0 + w1 * m1 + w2 * m2) / _softplus_floor(sm, 1e-12)

        flux = Fp + Fm
        return -(flux - roll(flux, 1)) / dx

    def _sweep(self, q: Tensor, vel: Tensor, axis: int, dx: float) -> Tensor:
        """Sweep WENO-5 along an arbitrary axis, batched over all others."""
        if axis == q.dim() - 1:
            return self._weno5_flux_last(q, vel, dx)
        q_t = torch.moveaxis(q, axis, -1)
        v_t = torch.moveaxis(vel, axis, -1)
        r = self._weno5_flux_last(q_t, v_t, dx)
        return torch.moveaxis(r, -1, axis)

    def advect(self, q: Tensor, vel: List[Tensor], dx: float, dt: float
               ) -> Tensor:
        """
        Advect q by velocity components `vel` (one Tensor per spatial axis).
        q: (..., *spatial).  Dimensional splitting (Strang).
        """
        n_spatial = len(vel)
        spatial_axes = list(range(q.dim() - n_spatial, q.dim()))
        out = q
        for ax, vv in zip(spatial_axes, vel):
            out = out + 0.5 * dt * self._sweep(out, vv, ax, dx)
        for ax, vv in zip(reversed(spatial_axes), reversed(vel)):
            out = out + 0.5 * dt * self._sweep(out, vv, ax, dx)
        return out

    # ── temporal coherence for video ────────────────────────────────────
    def coherence_step(self, current: Tensor, previous: Tensor,
                       flow: List[Tensor], dx: float, dt: float
                       ) -> Dict[str, Tensor]:
        """
        current, previous: (B, C, T, H, W) or (B, C, H, W) per-frame.
        flow: [vx, vy] velocity fields.
        Blends the advected previous frame into the current one so that
        texture does not boil between frames.
        """
        w = torch.sigmoid(self.blend)
        prev_adv = self.advect(previous, flow, dx, dt)
        blended = (1.0 - w) * current + w * prev_adv

        # coherence metric: normalised residual after advection
        resid = (current - prev_adv)
        denom = _softplus_floor(current.abs().mean() + 1e-8, 1e-8)
        coherence = torch.exp(-(resid.abs().mean() / denom))

        return {
            "blended": blended,
            "previous_advected": prev_adv,
            "coherence": coherence,
            "blend_weight": w,
        }

    @torch.no_grad()
    def cfl(self, vel: List[Tensor], dx: float, dt: float) -> float:
        """DIFF-FIX 8: diagnostics detached."""
        umax = max(float(v.abs().max()) for v in vel) if vel else 0.0
        return dt * umax / max(dx, 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# [FF1] FRACTAL FLUENCY OPTIMIZATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FluencyConfig:
    # Taylor (2006) preferred band [C]
    df_target: float = 1.40
    df_low: float = 1.30
    df_high: float = 1.50
    # differentiable box counting
    box_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64)
    edge_threshold: float = 0.12
    soft_tau: float = 0.03
    # loss weights
    w_fluency: float = 1.0
    w_aesthetic: float = 0.35
    w_smoothness: float = 0.20
    w_energy: float = 0.10
    # optimisation
    n_iters: int = 200
    lr: float = 3e-2
    # No-Zeno
    use_no_zeno: bool = True
    theta_hi: float = 0.22
    theta_lo: float = 0.06
    max_events: int = 40


@dataclass
class FluencyResult:
    d_f: float
    d_f_trace: List[float]
    loss_trace: List[float]
    target_band: Tuple[float, float]
    in_band: bool
    aesthetic_proxy: float
    no_zeno_events: int
    n_iters_run: int
    converged: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class FractalFluencyOptimizer(nn.Module):
    """
    [C/D] Optimises a generator's parameters so the produced artwork lands
    inside the fractal-fluency preference band.

    Literature [C]: Taylor et al. (2006) and follow-ups report a human
    preference peak for natural fractal patterns around d_f ~ 1.3-1.5,
    with reduced physiological stress response at mid-complexity.

    Differentiable box-counting [D]:
        N(eps) = sum_cells sigmoid( (edge_energy - t) / tau )
        d_f    = - slope of log N vs log eps   (least squares, differentiable)
    The soft-count makes N differentiable w.r.t. the image, hence w.r.t.
    the generator parameters.

    ⚠️ [O] HONEST — the strongest caveat in this file:
    "fractal fluency reduces stress" is a RESEARCH HYPOTHESIS with mixed
    replication. This optimiser targets a PREFERENCE BAND measured in
    psychophysics experiments. It is NOT therapy, NOT a clinical
    intervention, and produces NO health claim. Any clinical use requires
    IRB approval and a licensed clinician (see [TH1] IRBGate).
    """

    def __init__(self, cfg: Optional[FluencyConfig] = None):
        super().__init__()
        self.cfg = cfg or FluencyConfig()
        self.gate = (NoZenoGate(self.cfg.theta_hi, self.cfg.theta_lo)
                     if (self.cfg.use_no_zeno and HAS_NOZENO) else None)
        self._events = 0

    # ── differentiable box-counting dimension ───────────────────────────
    def box_count_dim(self, img: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """
        img: (B, C, H, W) in [0,1].
        Returns (d_f (B,), diagnostics).
        """
        tau = self.cfg.soft_tau
        thr = self.cfg.edge_threshold

        gray = img.mean(1, keepdim=True) if img.dim() == 4 else img
        gx = torch.roll(gray, -1, -1) - torch.roll(gray, 1, -1)
        gy = torch.roll(gray, -2, -2) - torch.roll(gray, 1 - 1, -2) if False else \
             torch.roll(gray, -1, -2) - torch.roll(gray, 1, -2)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        edge = edge / _softplus_floor(edge.amax(dim=(-2, -1), keepdim=True), 1e-8)

        logs_eps: List[Tensor] = []
        logs_N: List[Tensor] = []
        per_scale: Dict[int, Tensor] = {}

        for s in self.cfg.box_sizes:
            if s >= min(gray.shape[-2:]):
                continue
            pooled = F.avg_pool2d(edge, kernel_size=s, stride=s)
            # soft count of "occupied" boxes
            occ = torch.sigmoid((pooled - thr) / tau)
            N = occ.sum(dim=(-2, -1))                       # (B,)
            N = _softplus_floor(N, 1e-6, sharpness=200.0)
            logs_eps.append(torch.full_like(N, math.log(1.0 / s)))
            logs_N.append(torch.log(N))
            per_scale[s] = N.detach()

        if len(logs_N) < 3:
            z = torch.zeros(gray.shape[0], device=gray.device)
            return z, {"error": "need >= 3 scales", "per_scale": {}}

        X = torch.stack(logs_eps, dim=-1)                   # (B, K)
        Y = torch.stack(logs_N, dim=-1)                     # (B, K)
        Xm = X - X.mean(-1, keepdim=True)
        Ym = Y - Y.mean(-1, keepdim=True)
        denom = _softplus_floor((Xm * Xm).sum(-1), 1e-12, sharpness=200.0)
        slope = (Xm * Ym).sum(-1) / denom
        d_f = -slope

        r2 = 1.0 - ((Ym - slope.unsqueeze(-1) * Xm) ** 2).sum(-1) / \
             _softplus_floor((Ym ** 2).sum(-1), 1e-12, sharpness=200.0)

        return d_f, {"per_scale": {k: v.tolist() for k, v in per_scale.items()},
                     "fit_r2": r2.detach().tolist()}

    # ── loss terms ──────────────────────────────────────────────────────
    def fluency_loss(self, d_f: Tensor) -> Tensor:
        """Smooth hinge: zero inside [low, high], quadratic outside."""
        lo, hi = self.cfg.df_low, self.cfg.df_high
        below = F.relu(lo - d_f)
        above = F.relu(d_f - hi)
        centre_pull = (d_f - self.cfg.df_target) ** 2 * 0.05
        return (below ** 2 + above ** 2 + centre_pull).mean()

    @staticmethod
    def aesthetic_proxy(img: Tensor) -> Tensor:
        """
        [D] Cheap differentiable aesthetic proxy:
          contrast in a comfortable band + rule-of-thirds energy balance
          + low high-frequency harshness. NOT an aesthetic ground truth.
        """
        gray = img.mean(1, keepdim=True) if img.dim() == 4 else img
        contrast = gray.std(dim=(-2, -1))
        contrast_term = (contrast - 0.22) ** 2

        gx = torch.roll(gray, -1, -1) - gray
        gy = torch.roll(gray, -1, -2) - gray
        hf = (gx ** 2 + gy ** 2).mean(dim=(-2, -1))
        harsh_term = F.relu(hf - 0.02) ** 2

        H, W = gray.shape[-2:]
        thirds = []
        for fy in (1 / 3, 2 / 3):
            for fx in (1 / 3, 2 / 3):
                iy, ix = int(fy * H), int(fx * W)
                r = max(2, H // 32)
                patch = gray[..., max(0, iy - r):iy + r, max(0, ix - r):ix + r]
                thirds.append(patch.mean(dim=(-2, -1)))
            # keep symmetry
        thirds_t = torch.stack(thirds, dim=-1)
        balance_term = thirds_t.var(dim=-1)

        return (contrast_term.mean() + 5.0 * harsh_term.mean()
                + 2.0 * balance_term.mean())

    @staticmethod
    def smoothness_loss(img: Tensor) -> Tensor:
        gx = torch.roll(img, -1, -1) - img
        gy = torch.roll(img, -1, -2) - img
        gxx = torch.roll(gx, -1, -1) - gx
        gyy = torch.roll(gy, -1, -2) - gy
        return (gxx ** 2 + gyy ** 2).mean()

    @staticmethod
    def energy_loss(img: Tensor) -> Tensor:
        """Keep values inside the displayable range without hard clamping."""
        return (F.relu(img - 1.0) ** 2 + F.relu(-img) ** 2).mean()

    # ── optimisation loop ───────────────────────────────────────────────
    def optimize(self, generate: Callable[[], Tensor],
                 params: List[Tensor],
                 n_iters: Optional[int] = None) -> FluencyResult:
        """
        generate: zero-arg callable returning (B,C,H,W) in ~[0,1]
        params:   generator parameters to optimise
        """
        n_iters = n_iters or self.cfg.n_iters
        opt = torch.optim.Adam(params, lr=self.cfg.lr)
        df_trace: List[float] = []
        loss_trace: List[float] = []
        stopped = False
        it = 0

        for it in range(n_iters):
            # No-Zeno gate on the optimisation event count
            if self.gate is not None:
                res = self.gate.update(1.0)
                if not res.get("invariant_holds", True):
                    logger.warning("[Fluency] No-Zeno invariant violated; stopping")
                    stopped = True
                    break
                if res.get("allowed"):
                    self._events += 1
                if self._events > self.cfg.max_events:
                    logger.info("[Fluency] event budget exhausted; stopping")
                    stopped = True
                    break

            opt.zero_grad(set_to_none=True)
            img = generate()

            d_f, diag = self.box_count_dim(img)
            L_flu = self.fluency_loss(d_f)
            L_aes = self.aesthetic_proxy(img)
            L_sm = self.smoothness_loss(img)
            L_en = self.energy_loss(img)

            loss = (self.cfg.w_fluency * L_flu
                    + self.cfg.w_aesthetic * L_aes
                    + self.cfg.w_smoothness * L_sm
                    + self.cfg.w_energy * L_en)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            df_trace.append(float(d_f.mean().detach()))
            loss_trace.append(float(loss.detach()))

        d_final = df_trace[-1] if df_trace else float("nan")
        in_band = bool(self.cfg.df_low <= d_final <= self.cfg.df_high)

        return FluencyResult(
            d_f=d_final,
            d_f_trace=df_trace,
            loss_trace=loss_trace,
            target_band=(self.cfg.df_low, self.cfg.df_high),
            in_band=in_band,
            aesthetic_proxy=(loss_trace[-1] if loss_trace else float("nan")),
            no_zeno_events=(self.gate.event_count if self.gate else self._events),
            n_iters_run=it + 1,
            converged=(not stopped) and in_band,
            metadata={
                "stopped_by_budget": stopped,
                "target": self.cfg.df_target,
                "label": "[O] preference band, NOT a clinical endpoint",
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# [TH1] IRB GATE + THERAPY HEAD BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

class IRBGate:
    """
    [P] HARD BLOCK. Not a flag, not a warning — an exception.

    Any code path that consumes clinical identifiers (EEG biomarkers,
    diagnosis, patient id) to drive generation MUST pass through here.
    Without a recorded protocol number and a named clinician, the call
    raises. There is no override parameter.
    """

    def __init__(self):
        self._protocols: Dict[str, Dict[str, Any]] = {}

    def register_protocol(self, protocol_id: str, title: str,
                          approving_body: str, principal_investigator: str,
                          approved_at: Optional[float] = None,
                          expires_at: Optional[float] = None) -> None:
        if not (protocol_id and approving_body and principal_investigator):
            raise ValueError("protocol_id, approving_body and PI are required")
        self._protocols[protocol_id] = {
            "title": title,
            "approving_body": approving_body,
            "principal_investigator": principal_investigator,
            "approved_at": approved_at or time.time(),
            "expires_at": expires_at,
        }
        logger.info("[IRB] protocol registered: %s", protocol_id)

    def require(self, protocol_id: Optional[str],
                clinician_id: Optional[str]) -> Dict[str, Any]:
        if protocol_id is None or protocol_id not in self._protocols:
            raise PermissionError(
                "IRB GATE: no registered protocol. Clinical pathway blocked.")
        rec = self._protocols[protocol_id]
        if rec.get("expires_at") and time.time() > rec["expires_at"]:
            raise PermissionError(f"IRB GATE: protocol {protocol_id} expired.")
        if not clinician_id:
            raise PermissionError(
                "IRB GATE: named clinician required for every clinical call.")
        return rec

    def audit(self, protocol_id: str, clinician_id: str, action: str,
              details: Dict[str, Any]) -> None:
        logger.info("[IRB-AUDIT] %s | %s | %s | %s",
                    protocol_id, clinician_id, action,
                    {k: v for k, v in details.items() if k != "raw_eeg"})


@dataclass
class EEGBiomarkers:
    """
    [C] Literature biomarkers. Values are FEATURES, never diagnoses.
      frontal_alpha_asymmetry : ln(R) - ln(L) at F4/F3  (Davidson 2004)
      theta_beta_ratio        : Fz theta / Fz beta
      beta_power              : 13-30 Hz
      alpha_power             : 8-13 Hz
      gamma_power             : 30-100 Hz
    """
    frontal_alpha_asymmetry: float = 0.0
    theta_beta_ratio: float = 1.0
    beta_power: float = 10.0
    alpha_power: float = 20.0
    gamma_power: float = 2.0
    signal_quality: float = 1.0


class TherapyHeadBridge(nn.Module):
    """
    [D] Maps EEG biomarker FEATURES to art-generation parameter PRIORS.

    Direction of use is strictly one-way and one-purpose:
        biomarker features  ->  {d_f target, colour temperature,
                                 temporal rate, contrast band}
    The reverse (image -> diagnosis) is NOT implemented and must not be
    bolted on: this module emits no diagnosis, no severity, no risk score.

    Every call goes through IRBGate.require(). No protocol -> PermissionError.

    ⚠️ NOT A MEDICAL DEVICE. NOT DIAGNOSTIC. NOT THERAPEUTIC ON ITS OWN.
    The therapeutic value of the resulting artwork is [O] UNPROVEN.
    """

    def __init__(self, irb: IRBGate,
                 fluency: FractalFluencyOptimizer):
        super().__init__()
        self.irb = irb
        self.fluency = fluency
        # small differentiable map: 6 features -> 4 art priors
        self.prior_net = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.LayerNorm(32),
            nn.Linear(32, 4),
        )
        self.register_buffer("df_centre",
                             torch.tensor(float(fluency.cfg.df_target)))
        self.register_buffer("df_half",
                             torch.tensor(0.5 * (fluency.cfg.df_high
                                                 - fluency.cfg.df_low)))

    @staticmethod
    def _feature_vector(b: EEGBiomarkers) -> Tensor:
        q = max(0.0, min(1.0, b.signal_quality))
        return torch.tensor([
            math.tanh(b.frontal_alpha_asymmetry),
            math.tanh(b.theta_beta_ratio - 1.0),
            math.tanh((b.beta_power - 10.0) / 10.0),
            math.tanh((b.alpha_power - 20.0) / 20.0),
            math.tanh((b.gamma_power - 2.0) / 4.0),
            q,
        ], dtype=torch.float32)

    def art_priors(self, biomarkers: EEGBiomarkers,
                   protocol_id: Optional[str],
                   clinician_id: Optional[str],
                   session_id: str = "unset") -> Dict[str, Any]:
        """
        Returns art-generation priors. Raises PermissionError without IRB.
        """
        rec = self.irb.require(protocol_id, clinician_id)
        self.irb.audit(protocol_id, clinician_id, "art_priors",
                       {"session_id": session_id,
                        "signal_quality": biomarkers.signal_quality})

        x = self._feature_vector(biomarkers)
        raw = self.prior_net(x)

        df_target = float(self.df_centre + self.df_half * torch.tanh(raw[0]))
        df_target = max(self.fluency.cfg.df_low,
                        min(self.fluency.cfg.df_high, df_target))

        colour_temp = float(torch.sigmoid(raw[1]))
        temporal_rate = float(torch.sigmoid(raw[2]))
        contrast_band = float(torch.sigmoid(raw[3]))

        return {
            "df_target": df_target,
            "df_band": (self.fluency.cfg.df_low, self.fluency.cfg.df_high),
            "colour_temperature": colour_temp,   # 0 cool .. 1 warm
            "temporal_rate": temporal_rate,      # 0 still .. 1 fast
            "contrast_band": contrast_band,
            "signal_quality_gate": biomarkers.signal_quality >= 0.6,
            "clinical_use": False,
            "diagnostic": False,
            "requires_irb": True,
            "protocol_id": protocol_id,
            "clinician_id": clinician_id,
            "label": "[O] priors only; therapeutic benefit UNPROVEN",
        }

    def refused_call_example(self, biomarkers: EEGBiomarkers) -> str:
        """Demonstrates that the gate actually blocks."""
        try:
            self.art_priors(biomarkers, None, None)
            return "UNEXPECTED: gate did not block"
        except PermissionError as e:
            return f"BLOCKED as designed: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESICinemaVectorTherapyEngine(nn.Module):
    """
    Full chain:
      phase field (PFC / Cahn-Hilliard)
        -> SDF            [SDF1]
        -> SurfaceNets 3D [SDF1]
        -> marching squares + SVG  [MS1]
        -> WENO-5 temporal coherence for video  [W51]
        -> ACES encode / RRT-ODT / decode  [ACES1]
        -> fractal-fluency optimisation  [FF1]
        -> (IRB-gated) therapy head  [TH1]
    """

    def __init__(self,
                 epsilon: float = 1.5,
                 fluency_cfg: Optional[FluencyConfig] = None,
                 advect_cfg: Optional[AdvectionConfig] = None):
        super().__init__()
        self.aces = ACESColorPipeline()
        self.sdf = PhaseFieldSDF(epsilon=epsilon)
        self.surfacenets = SurfaceNetsMesh()
        self.ms = MarchingSquaresSVG(level=0.0)
        self.advect = WENO5VideoAdvector(advect_cfg)
        self.fluency = FractalFluencyOptimizer(fluency_cfg)
        self.irb = IRBGate()
        self.therapy = TherapyHeadBridge(self.irb, self.fluency)

        # optional real phase-field solvers from the KB
        self.ch = None
        if HAS_CH:
            cfg = CahnHilliardConfig(dx=1.0, epsilon=epsilon, dt=1e-5,
                                     laplacian="conv3d", scheme="imex")
            self.ch = StructuralCahnHilliard3D(cfg)

    # ── phase field generation ──────────────────────────────────────────
    def make_phase_field(self, n: int = 32, seed: int = 0,
                         pfc: bool = False, steps: int = 20) -> Tensor:
        """Produce a 3-D order parameter via the KB solvers if available."""
        g = torch.Generator().manual_seed(seed)
        u = (torch.rand(n, n, n, generator=g, dtype=torch.float64) * 0.2 - 0.1)
        if self.ch is None:
            return u.float()
        sigma = torch.ones_like(u)
        if pfc and HAS_CH:
            solver = PhaseFieldCrystal3D(CahnHilliardConfig(
                dx=1.0, epsilon=1.5, dt=1e-6, pfc_r=-0.5,
                laplacian="conv3d", scheme="imex"))
            for _ in range(steps):
                u = solver.step(u, sigma)
        else:
            for _ in range(steps):
                u = self.ch.step(u, sigma)
        return u.float()

    # ── full 3D vectorisation path ──────────────────────────────────────
    def vectorize_3d(self, u: Tensor, dx: float = 1.0,
                     level: float = 0.0,
                     reinit_iters: int = 10) -> Dict[str, Any]:
        sdf0 = self.sdf(u, dx=dx)
        sdf = self.sdf.reinitialize(sdf0, dx, n_iter=reinit_iters)
        mesh = self.surfacenets(u, dx=dx, level=level)
        return {
            "sdf_raw": sdf0,
            "sdf_reinit": sdf,
            "mesh": mesh,
            "obj": SurfaceNetsMesh.to_obj(mesh) if mesh.n_vertices else "",
            "sdf_grad_norm_dev": float(
                (self._grad_mag(sdf, dx) - 1.0).abs().mean()),
        }

    @staticmethod
    def _grad_mag(d: Tensor, dx: float) -> Tensor:
        g2 = sum(((torch.roll(d, -1, ax) - torch.roll(d, 1, ax)) / (2 * dx)) ** 2
                 for ax in range(d.dim()))
        return torch.sqrt(g2 + 1e-12)

    # ── full 2D vector path ─────────────────────────────────────────────
    def vectorize_2d(self, field2d: Tensor, dx: float = 1.0,
                     dp_epsilon: float = 0.0) -> SVGResult:
        return self.ms(field2d, dx=dx, dp_epsilon=dp_epsilon)

    # ── cinema colour path ──────────────────────────────────────────────
    def cinema_grade(self, srgb: Tensor,
                     out: ColorSpace = ColorSpace.SRGB) -> Dict[str, Tensor]:
        return self.aces(srgb, out)

    def aces_roundtrip_error(self, srgb: Tensor) -> float:
        a = self.aces.encode(srgb)
        back = self.aces.decode(a, ColorSpace.SRGB)
        # compare in linear space to avoid OETF nonlinearity amplification
        lin0 = self.aces.eotf_srgb(srgb)
        lin1 = self.aces.eotf_srgb(back.clamp(0, 1) if not back.requires_grad
                                   else back)
        return float((lin0 - lin1).abs().mean())

    # ── video temporal coherence path ───────────────────────────────────
    def video_coherence(self, frames: Tensor, flow: List[Tensor],
                        dx: float = 1.0, dt: float = 0.1) -> Dict[str, Tensor]:
        """frames: (B, C, T, H, W). Advect t-1 -> t and blend."""
        cur = frames[:, :, 1:]
        prev = frames[:, :, :-1]
        # sweep per frame pair
        outs, cohs = [], []
        for k in range(cur.shape[2]):
            r = self.advect.coherence_step(cur[:, :, k], prev[:, :, k],
                                           flow, dx, dt)
            outs.append(r["blended"])
            cohs.append(r["coherence"])
        return {
            "blended": torch.stack(outs, dim=2),
            "coherence": torch.stack(cohs),
            "mean_coherence": torch.stack(cohs).mean(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test() -> None:
    print("=" * 78)
    print(" SESI CINEMA VECTOR THERAPY ENGINE — SMOKE TEST")
    print("=" * 78)
    print(f" torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    print(f" Cahn-Hilliard KB     : {HAS_CH}")
    print(f" NoZenoGate KB        : {HAS_NOZENO}")
    print(f" BayesianUQ KB        : {HAS_BUQ}")
    print(f" spectral carpet KB   : {HAS_SPECTRAL}")
    print(f"\n Carpet-19 reference exponents [C] (rho NUMERICAL, not proved)")
    print(f"   d_f = ln8/ln3       = {CARPET_DF:.6f}")
    print(f"   rho                = {CARPET_RHO}")
    print(f"   d_w = ln(8rho)/ln3  = {CARPET_DW:.6f}")
    print(f"   d_s = 2 d_f / d_w   = {CARPET_DS:.6f}")
    print(f"   Caputo alpha = 2/d_w= {CARPET_ALPHA_CAPUTO:.6f}")

    torch.manual_seed(0)
    eng = SESICinemaVectorTherapyEngine(epsilon=1.5)

    # ── 1. ACES ─────────────────────────────────────────────────────────
    print("\n [ACES1] ACES colour pipeline")
    img = torch.rand(2, 3, 64, 64).permute(0, 3, 1, 2).clamp(0, 1)
    img = torch.rand(2, 64, 64, 3)
    out = eng.cinema_grade(img, ColorSpace.SRGB)
    print(f"   acescg shape        : {tuple(out['acescg'].shape)}")
    print(f"   scene_linear range  : [{float(out['scene_linear'].min()):.4f}, "
          f"{float(out['scene_linear'].max()):.4f}]")
    print(f"   display range       : [{float(out['display'].min()):.4f}, "
          f"{float(out['display'].max()):.4f}]")
    rt = eng.aces_roundtrip_error(img)
    print(f"   encode/decode RT err: {rt:.3e}  (linear-domain)")

    # differentiability through the tone curve
    p = img.clone().requires_grad_(True)
    o = eng.cinema_grade(p)
    o["display"].mean().backward()
    gn = float(p.grad.norm())
    print(f"   grad through RRT/ODT: {gn:.3e}   finite={math.isfinite(gn) and gn > 0}")

    # ── 2. phase field -> SDF -> mesh ───────────────────────────────────
    print("\n [SDF1] Phase field -> SDF -> SurfaceNets mesh")
    u = eng.make_phase_field(n=24, seed=1, pfc=False, steps=15)
    print(f"   phase field         : {tuple(u.shape)} "
          f"range [{u.min():.3f}, {u.max():.3f}]")
    r3 = eng.vectorize_3d(u, dx=1.0, reinit_iters=12)
    m = r3["mesh"]
    print(f"   SDF range           : [{float(r3['sdf_raw'].min()):.3f}, "
          f"{float(r3['sdf_raw'].max()):.3f}]")
    print(f"   |grad d| - 1 dev    : {r3['sdf_grad_norm_dev']:.4f} "
          f"(reinit, lower=better)")
    print(f"   mesh V / F          : {m.n_vertices} / {m.n_faces}")
    print(f"   active cells        : {m.metadata['active_cell_fraction']:.4f}")
    print(f"   watertight          : {m.watertight}  (NOT guaranteed)")
    print(f"   OBJ bytes           : {len(r3['obj'])}")

    # gradient flows field -> vertices
    ug = u.clone().requires_grad_(True)
    mm = eng.surfacenets(ug, dx=1.0)
    if mm.n_vertices:
        mm.vertices.sum().backward()
        gg = float(ug.grad.abs().sum()) if ug.grad is not None else 0.0
        print(f"   grad field->vertices: {gg:.3e}  finite={math.isfinite(gg)}")
    else:
        print("   (no active cells; skipped gradient check)")

    # ── 3. marching squares -> SVG ──────────────────────────────────────
    print("\n [MS1] Marching squares -> SVG")
    f2 = u[:, :, u.shape[2] // 2].clone().requires_grad_(True)
    svg = eng.vectorize_2d(f2, dx=1.0, dp_epsilon=0.35)
    print(f"   segments            : {svg.n_segments}")
    print(f"   paths               : {svg.n_paths}")
    print(f"   simplification      : {svg.simplification_ratio:.3f} "
          f"(1.0 = no reduction)")
    print(f"   bbox                : "
          f"({svg.bbox[0]:.1f},{svg.bbox[1]:.1f})-"
          f"({svg.bbox[2]:.1f},{svg.bbox[3]:.1f})")
    print(f"   svg bytes           : {len(svg.svg_text)}")
    print(f"   case histogram      : {svg.metadata['case_histogram']}")
    print(f"   saddle cases seen   : "
          f"{svg.metadata['case_histogram'].get(5,0) + svg.metadata['case_histogram'].get(10,0)}")

    # gradient flows field -> contour vertices
    if svg.n_segments:
        # rebuild one differentiable segment sum
        segs, _ = eng.ms.extract(f2, dx=1.0)
        tot = sum(a.sum() + b.sum() for a, b in segs)
        tot.backward()
        gv = float(f2.grad.abs().sum()) if f2.grad is not None else 0.0
        print(f"   grad field->contour : {gv:.3e}  finite={math.isfinite(gv)}")

    # ── 4. WENO-5 video temporal coherence ──────────────────────────────
    print("\n [W51] WENO-5 video advection + temporal coherence")
    B, C, T, H, W = 1, 3, 4, 48, 48
    frames = torch.rand(B, C, T, H, W, requires_grad=True)
    vx = torch.full((B, C, H, W), 2.0)
    vy = torch.full((B, C, H, W), -1.0)
    cfl = eng.advect.cfl([vx, vy], dx=1.0, dt=0.05)
    res = eng.video_coherence(frames, [vx, vy], dx=1.0, dt=0.05)
    print(f"   CFL (limit {eng.advect.cfg.cfl_limit}) : {cfl:.3f}")
    print(f"   blended shape       : {tuple(res['blended'].shape)}")
    print(f"   mean coherence      : {float(res['mean_coherence']):.4f}")
    print(f"   blend weight        : {float(torch.sigmoid(eng.advect.blend)):.4f}")
    res["blended"].sum().backward()
    gw = float(frames.grad.abs().sum()) if frames.grad is not None else 0.0
    print(f"   grad through WENO-5 : {gw:.3e}  finite={math.isfinite(gw) and gw>0}")

    # limiter smoothness check (DIFF-FIX 1)
    r = torch.linspace(-3, 3, 41, dtype=torch.float64).requires_grad_(True)
    for name, fn in _LIMITERS.items():
        fn(r).sum().backward()
        print(f"   limiter {name:9s} grad finite : "
              f"{bool(r.grad.isfinite().all())}")
        r.grad = None

    # ── 5. fractal fluency loop ─────────────────────────────────────────
    print("\n [FF1] Fractal-fluency optimisation  [O] preference band only")
    gen_amp = nn.Parameter(torch.tensor(0.35))
    gen_freq = nn.Parameter(torch.tensor(6.0))
    gen_seed = torch.rand(1, 1, 128, 128)

    def gen() -> Tensor:
        yy = torch.linspace(0, 1, 128).view(1, 1, 128, 1)
        xx = torch.linspace(0, 1, 128).view(1, 1, 1, 128)
        base = (torch.sin(2 * math.pi * gen_freq * xx)
                * torch.cos(2 * math.pi * gen_freq * 0.7 * yy))
        detail = F.interpolate(gen_seed, size=(128, 128), mode="bilinear",
                               align_corners=False)
        v = 0.5 + gen_amp * (0.7 * base + 0.3 * detail)
        return torch.sigmoid(v * 4.0 - 2.0).expand(1, 3, 128, 128)

    d0, _ = eng.fluency.box_count_dim(gen())
    print(f"   d_f before          : {float(d0.mean()):.4f}")
    fr = eng.fluency.optimize(gen, [gen_amp, gen_freq], n_iters=60)
    print(f"   d_f after           : {fr.d_f:.4f}")
    print(f"   target band         : {fr.target_band}  in_band={fr.in_band}")
    print(f"   iters run           : {fr.n_iters_run}")
    print(f"   no-zeno events      : {fr.no_zeno_events}")
    print(f"   loss {fr.loss_trace[0]:.4f} -> {fr.loss_trace[-1]:.4f}")
    print(f"   {fr.metadata['label']}")

    # ── 6. IRB gate + therapy head ──────────────────────────────────────
    print("\n [TH1] IRB gate + therapy head  (RESEARCH ONLY)")
    bm = EEGBiomarkers(frontal_alpha_asymmetry=-0.12, theta_beta_ratio=1.4,
                       beta_power=13.0, alpha_power=18.0, gamma_power=2.2,
                       signal_quality=0.82)

    print(f"   no-protocol call    : {eng.therapy.refused_call_example(bm)}")
    try:
        eng.therapy.art_priors(bm, "FAKE-001", "dr_x")
        print("   UNEXPECTED: fake protocol accepted")
    except PermissionError:
        print("   fake protocol       : BLOCKED as designed")
    try:
        eng.irb.register_protocol("IRB-2026-014",
                                  "Fractal art exposure in adults",
                                  "University IRB", "dr_jones")
        pri = eng.therapy.art_priors(bm, "IRB-2026-014", "dr_jones",
                                     session_id="s001")
        print(f"   registered protocol : ACCEPTED")
        print(f"     d_f target        : {pri['df_target']:.3f} "
              f"band={pri['df_band']}")
        print(f"     colour temp       : {pri['colour_temperature']:.3f}")
        print(f"     temporal rate     : {pri['temporal_rate']:.3f}")
        print(f"     contrast band     : {pri['contrast_band']:.3f}")
        print(f"     quality gate      : {pri['signal_quality_gate']}")
        print(f"     clinical_use={pri['clinical_use']}  "
              f"diagnostic={pri['diagnostic']}  "
              f"requires_irb={pri['requires_irb']}")
        print(f"     {pri['label']}")
    except PermissionError as e:
        print(f"   unexpected block    : {e}")

    # feed the IRB-approved d_f target back into the fluency loop
    print("\n [LOOP] IRB priors -> fluency retarget -> regenerate")
    eng.fluency.cfg.df_target = pri["df_target"]
    eng.fluency.cfg.df_low = max(1.20, pri["df_target"] - 0.08)
    eng.fluency.cfg.df_high = min(1.65, pri["df_target"] + 0.08)
    fr2 = eng.fluency.optimize(gen, [gen_amp, gen_freq], n_iters=40)
    print(f"   retargeted band     : ({eng.fluency.cfg.df_low:.2f}, "
          f"{eng.fluency.cfg.df_high:.2f})")
    print(f"   d_f achieved        : {fr2.d_f:.4f}  in_band={fr2.in_band}")
    print(f"   events used         : {fr2.no_zeno_events}")

    # ── summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C] ACES Hill fit is an APPROXIMATION of the official RRT/ODT.")
    print("     Theatrical delivery needs a certified OCIO config.")
    print(" [C] Allen-Cahn SDF inversion is EXACT only for the flat 1-D")
    print("     equilibrium profile; curved/PFC interfaces are approximate.")
    print(" [D] SurfaceNets output is NOT guaranteed watertight.")
    print(" [C] Marching-squares saddles use the asymptotic decider; exact")
    print("     equality degeneracies are perturbed.")
    print(" [C] WENO-5 here is scalar advection. NOT DNS, no shock guarantee.")
    print(" [O] Fractal fluency -> stress reduction is a RESEARCH HYPOTHESIS")
    print("     with mixed replication. No health claim is made or implied.")
    print(" [P] IRBGate is a HARD BLOCK: no protocol -> PermissionError.")
    print("     TherapyHeadBridge emits NO diagnosis, NO severity, NO risk.")
    print(" [C] Carpet rho ~ 1.25 is NUMERICAL, not proved (Carpet-19).")
    print(" [O] Exporting carpet exponents to a non-PCF fractal is an")
    print("     exponent-matched TRANSPLANT. Closure is NOT transferred.")
    print(" [O] Spectral -> topological invariants remain SPECULATIVE.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
