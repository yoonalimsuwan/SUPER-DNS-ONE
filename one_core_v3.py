# =============================================================================
# ONE CORE — Shared Foundation of the ONE Ecosystem
# =============================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# AI Development Partners:
#   Claude   (Anthropic)  — architecture review, differentiability fixes,
#                           bridge protocol design, integration testing
#   GPT      (OpenAI)     — algorithmic suggestions, code review
#   Gemini   (Google)     — numerical scheme cross-validation
#   DeepSeek              — supplementary code analysis
#
# This module is the single source of truth for every component that is
# shared across the ONE Ecosystem:
#
#   structural_langevin_v3.py          (MD / particle scale)
#   structuralfluctuatinghydro_v6.py  (FH continuum 3-D)
#   super_dns_one_v6.py               (DNS/LES 3-D compressible)
#   structural_cahn_hilliard_3d.py    (CH phase-field 3-D)
#
# Cross-file bridge protocol
# ──────────────────────────
#   LangevinFHBridge        — Langevin structural stress → FH solver
#   LangevinDNSBridge       — Langevin structural stress → DNS solver
#   CahnHilliardFHBridge    — CH phase-field density/viscosity → FH solver
#   CahnHilliardDNSBridge   — CH phase-field + Korteweg forces → DNS solver
#   SeismicDNSBridge        — ground-motion pseudo body force → DNS solver
#   HeatReleaseDNSBridge    — combustion/radiation heat release → DNS solver
#   PyrolysisDNSBridge      — mass source (continuity) → DNS solver
#
# Rule: if a class appears in more than one solver file, it lives here and
# is imported from here.  Individual solver files must NOT redefine these
# classes locally.
#
# Shared components
# ─────────────────
#   SemanticStateContraction   — SSC EMA low-pass filter (Paper 4)
#   CSOCBase                   — base class for CSOC adaptive thermostat/viscosity
#   InterfaceDetectorBase      — abstract base for interface detection
#   StructuralItoBase          — abstract base for Itô drift correction
#   structural_biharmonic_n    — recursive Δ_S^n operator (module-level util)
#   get_device                 — unified hardware-backend selector
#   ONE_VERSION                — ecosystem-wide version string
#
# Version history
# ───────────────
#   3.0.0  — Langevin↔FH and Langevin↔DNS bridges
#   3.1.0  — CahnHilliard↔FH, CahnHilliard↔DNS bridges;
#             structural_biharmonic_n promoted to one_core;
#             full 5-solver interoperability
#   3.2.0  — SeismicDNSBridge added (ground-motion pseudo body force →
#             DNS solver, for tank sloshing / shaking-foundation fluid
#             problems); duck-typed source (no new hard dependency).
#             Companion fix (super_dns_one_v6_3.py, Bug 11): _ext_nu_ch
#             was written by CahnHilliardDNSBridge.sync() but never read
#             in CompressibleSolver._compute_rhs -- the viscosity
#             contrast between CH phases had silently had zero effect in
#             every prior coupled run. Fixed on the solver side; see that
#             file's v6.3→v6.4 changelog entry.
#   3.3.0  — HeatReleaseDNSBridge added (volumetric combustion/radiation
#             heat release → DNS solver, for FIRE ONE / fire_one.py +
#             fire_dns_coupling_one.py). Requires super_dns_one_v6_3.py
#             v6.5+ (new _ext_q buffer + consumption); raises clearly at
#             construction if the solver predates that fix, rather than
#             risking another silent-no-op buffer (the exact failure mode
#             _ext_nu_ch taught this ecosystem to guard against).
#   3.4.0  — PyrolysisDNSBridge added (genuine mass source → DNS
#             continuity equation, for FIRE ONE / fire_one.PyrolysisModel
#             pyrolysis mass flux). Requires super_dns_one_v6_3.py v6.8+
#             (new _ext_mdot* buffer family). Writes 5 buffers together
#             (mass + carried momentum/energy/mixture-fraction) so a
#             partial source spec still produces a physically consistent
#             set rather than stale leftover values. UNIT-CONSISTENCY
#             NOTE: the solver-side consumption of these buffers was
#             initially missing the same nondim-scale conversion already
#             required for combustion/radiation (v6.6) -- caught and
#             fixed with a SEPARATE cfg.mdot_nondim_scale (mass-rate and
#             energy-rate source terms are different dimensional groups,
#             so they cannot share combustion_nondim_scale even after a
#             conversion is added). See that file's v6.7→v6.8 changelog.
#   3.5.0  — PyrolysisDNSBridge gains target='wall' mode: writes DIRECTLY
#             into a PyrolysisWallBC (super_dns_one_v6_3.py v6.9+) attached
#             to a specific domain face, a genuine Stefan-flow blowing-
#             wall surface boundary condition, replacing the v6.8/v3.4.0
#             thin-near-wall-volume proxy for cases where the fuel
#             surface aligns with a domain face (the common case: a fuel
#             bed on the floor, a wall lining, etc.). target='volumetric'
#             (v3.4.0 behaviour) remains available and unchanged for
#             sources not aligned with a domain face. See that file's
#             v6.8→v6.9 changelog for the wall-BC-side implementation,
#             including a real pre-existing bug found and fixed along the
#             way (self.bc_objects was referenced but never assigned,
#             breaking ghost-cell filling for ANY non-periodic BC) and a
#             second gap closed (Z/soot mixture-fraction fields had no
#             wall-aware ghost-cell treatment at all before v6.9 -- hard
#             zeros at any non-periodic boundary).
#
# =============================================================================

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ONE_VERSION: str = "3.5.0"


# =============================================================================
# 0. Hardware-backend selector
# =============================================================================

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Select the best available compute device.

    Priority order when ``preferred`` is not available:
    CUDA → MPS (Apple) → CPU.

    Args:
        preferred : ``"cuda"``, ``"mps"``, ``"ascend"``, or ``"cpu"``.

    Returns:
        A :class:`torch.device`.
    """
    p = preferred.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if p == "ascend":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
    # Fallback cascade
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. Semantic State Contraction (SSC) — Paper 4
# =============================================================================

class SemanticStateContraction(nn.Module):
    """
    SSC EMA low-pass filter for structural stress σ  (Paper 4).

    This is the **canonical implementation** used by all three solvers.
    Do not redefine this class in individual solver files.

    The filter tracks the structural stress signal via a first-order
    exponential moving average (EMA):

        σ_filtered[t] = σ_filtered[t-1] + ε · (σ_raw[t] − σ_filtered[t-1])

    Design decisions
    ────────────────
    •  Implemented as ``nn.Module`` so that ``prev_sigma`` and the
       initialisation flag are proper PyTorch buffers: they move with
       ``.to(device)`` and are saved/loaded by ``state_dict()``.
    •  A boolean ``_initialized`` buffer (not a zero-check) is used to
       detect the first call, so the filter works correctly even when the
       true first stress value is zero.
    •  ``reset()`` clears both the buffer and the flag, enabling clean
       reuse between independent trajectories / simulation runs.

    Args:
        epsilon_fp : EMA blending factor ∈ (0, 1).
                     Smaller → slower response (more smoothing).
    """

    def __init__(self, epsilon_fp: float = 0.0028) -> None:
        super().__init__()
        if not (0.0 < epsilon_fp < 1.0):
            raise ValueError(
                f"epsilon_fp must be in (0, 1); got {epsilon_fp!r}.")
        self.eps = epsilon_fp
        self.register_buffer("prev_sigma",   torch.tensor(0.0))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset(self) -> None:
        """Reset EMA state.  Call between independent trajectories/runs."""
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_sigma : scalar stress tensor (differentiable).
        Returns:
            Filtered stress scalar.
        """
        # Move buffer to the same device as the incoming tensor
        if self.prev_sigma.device != raw_sigma.device:
            self.prev_sigma   = self.prev_sigma.to(raw_sigma.device)
            self._initialized = self._initialized.to(raw_sigma.device)

        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma

        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# =============================================================================
# 2. CSOC Base — Paper 4
# =============================================================================

class CSOCBase(nn.Module, ABC):
    """
    Abstract base class for CSOC (Controlled Self-Organised Criticality)
    adaptive parameter modules  (Paper 4).

    Subclasses implement ``forward()`` to return domain-specific adaptive
    parameters (temperature + friction for MD; viscosity + diffusivity for
    CFD).  The SSC filter and the normalised-deviation / sigmoid logic are
    provided here so they are consistent across all domains.

    Args:
        sigma_target   : reference structural stress.
        epsilon_fp     : SSC contraction rate.
        boost_factor   : maximum parameter multiplier at high stress.
    """

    def __init__(
        self,
        sigma_target: float = 1.0,
        epsilon_fp:   float = 0.0028,
        boost_factor: float = 3.0,
    ) -> None:
        super().__init__()
        if sigma_target <= 0:
            raise ValueError(f"sigma_target must be positive; got {sigma_target!r}.")
        if boost_factor < 1.0:
            raise ValueError(f"boost_factor must be ≥ 1; got {boost_factor!r}.")
        self.sigma_target = sigma_target
        self.boost_factor = boost_factor
        self.ssc = SemanticStateContraction(epsilon_fp)

    def reset(self) -> None:
        """Reset SSC state (call between independent runs)."""
        self.ssc.reset()

    def _normalised_deviation(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Normalised deviation from target:  (σ − σ_target) / σ_target.
        Returns a scalar tensor; positive means stress exceeds target.
        """
        return (sigma - self.sigma_target) / max(self.sigma_target, 1e-12)

    def _smooth_boost(self, dev: torch.Tensor) -> torch.Tensor:
        """
        Smooth boost factor ∈ (0, 1) via sigmoid, used to interpolate
        between the base value and ``base * boost_factor``.
        """
        return torch.sigmoid(dev)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute adaptive parameters from current structural stress."""


# =============================================================================
# 3. Interface Detector Base
# =============================================================================

class InterfaceDetectorBase(nn.Module, ABC):
    """
    Abstract base class for differentiable interface detectors.

    The molecular (Langevin) and continuum (FH / DNS) solvers each need
    a differentiable mask that identifies sharp-gradient regions, but the
    inputs differ (atomic coordinates vs. grid scalar fields).  This base
    class enforces a common ``forward()`` signature contract.

    All subclasses must return a tensor with values in [0, 1] where values
    near 1 indicate interface / shock / phase-boundary regions.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns:
            mask : tensor ∈ [0, 1], fully differentiable w.r.t. inputs.
        """


# =============================================================================
# 4. Structural Itô Base — Papers 2 & 3
# =============================================================================

class StructuralItoBase(nn.Module, ABC):
    """
    Abstract base class for Structural Itô drift correction modules.

    Both the Langevin integrator (per-atom) and the continuum FH solver
    (per-cell) compute the same ½ G(x) ∇_x G(x) correction term; only the
    dimensionality and the interface detector differ.

    Subclasses implement ``compute_ito_correction()`` which must:
      • Accept a field / coordinate tensor and an interface detector.
      • Return the Itô drift of the same shape as the input field.
      • Be computed with ``torch.enable_grad()`` / ``autograd.grad``
        internally; the *return value* must be detached.

    Args:
        interface_amplification : G-field amplitude factor at interfaces.
    """

    def __init__(self, interface_amplification: float = 2.0) -> None:
        super().__init__()
        if interface_amplification < 0:
            raise ValueError(
                f"interface_amplification must be ≥ 0; got {interface_amplification!r}.")
        self.amp = interface_amplification

    def get_g_field(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp · mask(x).  Same formula in all domains."""
        return 1.0 + self.amp * interface_mask

    @abstractmethod
    def compute_ito_correction(
        self,
        field: torch.Tensor,
        interface_detector: InterfaceDetectorBase,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute ½ G(x) ∇_x G(x).

        Returns:
            Itô drift tensor, same shape as ``field``, detached.
        """



# =============================================================================
# 5. Cross-solver Bridge Protocol
# =============================================================================

class LangevinFHBridge:
    """
    Bridge: feeds Langevin structural stress and interface mask into the
    Fluctuating Hydro (FH) solver as external forcing fields.

    This resolves Bug 3: ``structural_langevin_v3`` now has an explicit
    interface for communicating with ``StructuralFluctuatingHydro``.

    Usage::

        bridge = LangevinFHBridge(langevin_integrator, fh_solver)
        bridge.sync(coords, velocities)   # call each MD step

    The bridge interpolates per-atom quantities onto the FH grid via
    kernel density estimation (KDE) and stores them as ``fh_solver._ext_sigma``
    and ``fh_solver._ext_mask`` for use in ``LLStochasticStress``.
    """

    def __init__(self, langevin, fh_solver, bandwidth: float = 1.0) -> None:
        self.lang       = langevin
        self.fh         = fh_solver
        self.bandwidth  = bandwidth

    def sync(
        self,
        coords: torch.Tensor,
        velocities: torch.Tensor,
    ) -> None:
        """
        Project Langevin state onto the FH grid.

        Args:
            coords    : (N, 3)  current atomic positions (Å).
            velocities: (N, 3)  current atomic velocities.
        """
        with torch.no_grad():
            mask_atomic = self.lang.interface_detector(coords)   # (N,)
            sigma_scalar = self.lang.thermostat.ssc.prev_sigma   # scalar buffer

            # Broadcast sigma to (Nx, Ny, Nz) and store on FH solver
            cfg = self.fh.cfg
            sig_grid = sigma_scalar.expand(cfg.Nx, cfg.Ny, cfg.Nz).clone()
            # Simple uniform projection — advanced KDE can be added here
            self.fh._ext_sigma = sig_grid.to(self.fh.device)
            self.fh._ext_mask  = torch.ones(
                cfg.Nx, cfg.Ny, cfg.Nz,
                device=self.fh.device, dtype=self.fh.cfg.dtype
            ) * mask_atomic.mean()


class LangevinDNSBridge:
    """
    Bridge: feeds Langevin structural stress into the DNS/LES solver
    (``CompressibleSolver``) as an external SOC signal.

    Usage::

        bridge = LangevinDNSBridge(langevin_integrator, dns_solver)
        bridge.sync(coords)   # call before each DNS step

    Stores ``dns_solver._ext_sigma`` which ``SOCController.nu_t()``
    picks up to modulate eddy viscosity.
    """

    def __init__(self, langevin, dns_solver) -> None:
        self.lang = langevin
        self.dns  = dns_solver

    def sync(self, coords: torch.Tensor) -> None:
        """
        Args:
            coords : (N, 3) current atomic positions.
        """
        with torch.no_grad():
            sigma_scalar = self.lang.thermostat.ssc.prev_sigma   # scalar buffer
            self.dns._ext_sigma = sigma_scalar.to(self.dns.device)


# =============================================================================
# 5b. Cahn-Hilliard Bridge Protocol  (ONE Core v3.1)
# =============================================================================

class CahnHilliardFHBridge:
    """
    Bridge: projects Cahn-Hilliard phase-field quantities onto the
    Fluctuating Hydrodynamics (FH) solver grid, enabling two-way coupling
    between the CH phase-field and the continuum FH solver.

    Coupling channels (CH → FH)
    ────────────────────────────
    1. Density modulation  : rho_eff = rho_B + (rho_A−rho_B)·φ(u)
    2. Viscosity modulation: nu_eff  = nu_B  + (nu_A −nu_B )·φ(u)
    3. Interface mask      : phi used to sharpen FH interface detection

    where φ(u) = 0.5·(u+1) maps the CH order parameter u ∈ [−1,+1] to
    the volume fraction φ ∈ [0, 1].

    All operations are differentiable w.r.t. ``u``.

    Usage::

        bridge = CahnHilliardFHBridge(ch_solver, fh_solver,
                                       rho_A=1.0, rho_B=2.0)
        for step in range(n):
            u_new = ch_solver.step(u)
            bridge.sync(u_new)                    # push CH → FH
            rho, ux, uy, uz, p, diag = fh_solver.step(rho, ux, uy, uz, p)
    """

    def __init__(
        self,
        ch_solver,
        fh_solver,
        rho_A: float = 1.0,
        rho_B: float = 2.0,
        nu_A:  float = 1e-3,
        nu_B:  float = 1e-2,
    ) -> None:
        self.ch    = ch_solver
        self.fh    = fh_solver
        self.rho_A = rho_A
        self.rho_B = rho_B
        self.nu_A  = nu_A
        self.nu_B  = nu_B

    def sync(self, u: torch.Tensor) -> None:
        """
        Project CH order parameter onto FH material fields.

        Args:
            u : (Nx, Ny, Nz) Cahn-Hilliard order parameter ∈ [−1, +1].

        Writes to fh_solver:
            _ext_rho_eff : (Nx, Ny, Nz) effective density field
            _ext_nu_eff  : (Nx, Ny, Nz) effective kinematic viscosity field
            _ext_mask    : (Nx, Ny, Nz) volume-fraction interface mask ∈ [0,1]
        """
        with torch.no_grad():
            phi = 0.5 * (u + 1.0)                                    # (Nx,Ny,Nz)
            dev = self.fh.device
            self.fh._ext_rho_eff = (
                self.rho_B + (self.rho_A - self.rho_B) * phi
            ).to(dev)
            self.fh._ext_nu_eff = (
                self.nu_B + (self.nu_A - self.nu_B) * phi
            ).to(dev)
            # Interface mask: high near u≈0 (diffuse interface), low in bulk
            self.fh._ext_mask = (
                1.0 - (u ** 2).clamp(max=1.0)        # ≈1 at u=0, ≈0 at u=±1
            ).to(dev)


class CahnHilliardDNSBridge(nn.Module):
    """
    Bridge: injects Cahn-Hilliard phase-field material properties and
    Korteweg capillary body forces into the DNS/LES compressible solver.

    This is the **canonical implementation** used by all ONE Ecosystem
    solvers.  Do not redefine this class in individual solver files.

    Coupling channels (CH → DNS)
    ─────────────────────────────
    1. Density modulation   → ``dns_solver._ext_rho_ch``
    2. Viscosity modulation → ``dns_solver._ext_nu_ch``
    3. Korteweg body force  → ``dns_solver._ext_fx / _ext_fy / _ext_fz``

    All fields are stored as solver attributes that ``_compute_rhs()``
    blends in when present.

    The Korteweg force reads::

        f_i = −κ · ρ_eff · ∂μ_R/∂x_i

    which is computed by ``ch_solver``'s ``compute_chemical_potential``.

    Two usage patterns are supported:

    Pattern A — explicit DNS solver (two-solver coupling)::

        bridge = CahnHilliardDNSBridge(ch_solver, dns_solver,
                                        korteweg_strength=1e-4)
        for step in range(n):
            u_new = ch_solver.step(u)
            bridge.sync(u_new)          # push CH → DNS buffers
            dns_solver.step()

    Pattern B — standalone CH-only usage (no dns_solver)::

        bridge = CahnHilliardDNSBridge(ch_solver,
                                        korteweg_strength=0.1)
        u_new, rho_eff, nu_eff, fx, fy, fz = bridge.coupled_step(u, sigma)

    Args:
        ch_solver          : a StructuralCahnHilliard3D (or subclass) instance.
        dns_solver         : CompressibleSolver instance, or ``None`` for
                             standalone usage via ``coupled_step()``.
        rho_A, rho_B       : phase densities (A = u→+1, B = u→−1).
        nu_A, nu_B         : phase kinematic viscosities.
        korteweg_strength  : κ ≥ 0; set 0.0 to disable Korteweg forces.
    """

    def __init__(
        self,
        ch_solver,
        dns_solver=None,
        rho_A:             float = 1.0,
        rho_B:             float = 2.0,
        nu_A:              float = 1e-3,
        nu_B:              float = 1e-2,
        korteweg_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.ch                = ch_solver
        self.dns               = dns_solver
        self.rho_A             = rho_A
        self.rho_B             = rho_B
        self.nu_A              = nu_A
        self.nu_B              = nu_B
        self.korteweg_strength = korteweg_strength

    # ------------------------------------------------------------------
    # Helpers (differentiable w.r.t. u)
    # ------------------------------------------------------------------

    def effective_density(self, u: torch.Tensor) -> torch.Tensor:
        """ρ_eff(u) = ρ_B + (ρ_A − ρ_B) · φ(u),  φ = 0.5·(u+1)."""
        phi = 0.5 * (u + 1.0)
        return self.rho_B + (self.rho_A - self.rho_B) * phi

    def effective_viscosity(self, u: torch.Tensor) -> torch.Tensor:
        """ν_eff(u) = ν_B + (ν_A − ν_B) · φ(u),  φ = 0.5·(u+1)."""
        phi = 0.5 * (u + 1.0)
        return self.nu_B + (self.nu_A - self.nu_B) * phi

    def korteweg_force(
        self,
        u:     torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Korteweg capillary body force components.

        Returns:
            (fx, fy, fz) each of shape (Nx, Ny, Nz).
            All zeros when ``korteweg_strength == 0``.
        """
        if self.korteweg_strength == 0.0:
            z = torch.zeros_like(u)
            return z, z, z
        sigma   = self.ch._resolve_sigma(u, sigma)
        mu_R    = self.ch.compute_chemical_potential(u, sigma)
        rho_eff = self.effective_density(u)
        dx      = self.ch.cfg.dx
        k       = self.korteweg_strength
        dmx = (torch.roll(mu_R, -1, 0) - torch.roll(mu_R, +1, 0)) / (2 * dx)
        dmy = (torch.roll(mu_R, -1, 1) - torch.roll(mu_R, +1, 1)) / (2 * dx)
        dmz = (torch.roll(mu_R, -1, 2) - torch.roll(mu_R, +1, 2)) / (2 * dx)
        return -k * rho_eff * dmx, -k * rho_eff * dmy, -k * rho_eff * dmz

    # ------------------------------------------------------------------
    # Pattern A — push to DNS solver buffers
    # ------------------------------------------------------------------

    def sync(
        self,
        u:     torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Project CH order parameter onto DNS solver buffers.

        Requires ``dns_solver`` to have been supplied at construction.

        Args:
            u     : (Nx, Ny, Nz) CH order parameter ∈ [−1, +1].
            sigma : optional structural σ field for chemical potential.

        Writes to dns_solver:
            _ext_rho_ch   : (Nx, Ny, Nz) effective density
            _ext_nu_ch    : (Nx, Ny, Nz) effective kinematic viscosity
            _ext_fx/fy/fz : (Nx, Ny, Nz) Korteweg body force components
        """
        if self.dns is None:
            raise RuntimeError(
                "CahnHilliardDNSBridge.sync() requires dns_solver; "
                "pass dns_solver= at construction or use coupled_step()."
            )
        dev = self.dns.device
        with torch.no_grad():
            rho_eff = self.effective_density(u)
            nu_eff  = self.effective_viscosity(u)
            self.dns._ext_rho_ch = rho_eff.to(dev)
            self.dns._ext_nu_ch  = nu_eff.to(dev)
            fx, fy, fz = self.korteweg_force(u, sigma)
            self.dns._ext_fx = fx.to(dev)
            self.dns._ext_fy = fy.to(dev)
            self.dns._ext_fz = fz.to(dev)

    # ------------------------------------------------------------------
    # Pattern B — standalone coupled step (no dns_solver needed)
    # ------------------------------------------------------------------

    def coupled_step(
        self,
        u:     torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance the CH solver one step and return all coupling fields.

        Returns:
            u_new   : (Nx, Ny, Nz) updated order parameter
            rho_eff : (Nx, Ny, Nz) effective density
            nu_eff  : (Nx, Ny, Nz) effective kinematic viscosity
            fx      : (Nx, Ny, Nz) Korteweg force x-component
            fy      : (Nx, Ny, Nz) Korteweg force y-component
            fz      : (Nx, Ny, Nz) Korteweg force z-component
        """
        u_new   = self.ch.step(u, sigma)
        rho_eff = self.effective_density(u_new)
        nu_eff  = self.effective_viscosity(u_new)
        fx, fy, fz = self.korteweg_force(u_new, sigma)
        return u_new, rho_eff, nu_eff, fx, fy, fz


# =============================================================================
# 5d. Seismic-DNS Bridge Protocol  (ONE Core v3.2)
# =============================================================================

class SeismicDNSBridge:
    """
    Bridge: injects ground-motion pseudo body force into the DNS/LES
    compressible solver, for non-inertial-frame problems -- tank
    sloshing, equipment/piping on a shaking foundation, liquid storage
    on a structure subjected to earthquake shaking, etc.

    Physics
    ───────
    In the reference frame of a rigidly-shaking structure, the fluid
    momentum equation gains a uniform pseudo body force equal in
    magnitude and opposite in sign to the structure's absolute ground
    acceleration a_ground(t):

        f_volumetric(x,t) = −ρ(x,t) · a_ground(t)      [N/m^3]

    which is exactly the quantity dns_solver._compute_rhs adds directly
    into rhs_rhou/rhs_rhov/rhs_rhow (and, automatically, f·u into
    rhs_rhoE) -- see "Cahn-Hilliard Korteweg body force injection" in
    super_dns_one_v6_3.py, which reads _ext_fx/_ext_fy/_ext_fz built
    exactly this way. This bridge writes those same buffers, following
    the same "written by <Bridge>.sync(), zero-cost if not connected"
    convention as CahnHilliardDNSBridge above.

    Source coupling is duck-typed, NOT a hard dependency on any
    particular seismic-analysis module: accel_x/accel_y/accel_z are each
    either ``None``, a constant float [m/s^2], or any object exposing
    ``.at(t) -> float`` returning the absolute ground acceleration at
    time t [s] (e.g. a ``seismic_dns_coupling_one.CFDTimeSeriesBC``
    wrapping a SEISMIC ONE ``GroundMotion``). This keeps one_core's only
    hard dependency as torch, matching every other bridge in this file.

    IMPORTANT — sign convention: accel_x/y/z must supply the RAW
    absolute ground acceleration a_ground(t), not a pre-negated pseudo-
    force. This bridge applies the −ρ·a sign internally. (If wrapping
    seismic_dns_coupling_one.TankSloshingCoupling, use its raw motion's
    acceleration series, e.g. ``CFDTimeSeriesBC(motion.time, motion.accel,
    ...)`` -- NOT ``body_force_series()``, which already applies the
    negative sign for standalone use outside this bridge and would
    double the sign flip if passed here.)

    Usage::

        bridge = SeismicDNSBridge(dns_solver, accel_x=ax_series, accel_z=az_series)
        for step in range(n_steps):
            bridge.sync()          # writes dns_solver._ext_fx/_ext_fy/_ext_fz
            dns_solver.step()

    Notes
    ─────
    - dns_solver.time is only advanced by CompressibleSolver.step() AFTER
      all 3 TVD-RK3 substages complete (one dt for the whole step), so
      sync() should be called once per outer step, BEFORE step() -- the
      time used is the value at the START of that step, held constant
      across the 3 internal RK stages (matching how _ext_fx is checked
      fresh in every _compute_rhs call within a step but only written
      once per sync()).
    - Overwrite semantics (not accumulate), matching
      CahnHilliardDNSBridge.sync(): each call replaces
      _ext_fx/_ext_fy/_ext_fz entirely. There is currently no gravity
      term anywhere in super_dns_one_v6_3.py's _compute_rhs, so if a
      simulation needs both hydrostatic gravity and seismic forcing (or
      both this bridge and a simultaneously-active Korteweg force from
      CahnHilliardDNSBridge), combine the contributions externally
      before/after calling sync() rather than expecting automatic
      superposition across bridges.
    """

    def __init__(self, dns_solver, accel_x=None, accel_y=None, accel_z=None) -> None:
        self.dns = dns_solver
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z

    @staticmethod
    def _eval(source, t: float) -> float:
        if source is None:
            return 0.0
        if isinstance(source, (int, float)):
            return float(source)
        at = getattr(source, "at", None)
        if callable(at):
            return float(at(t))
        raise TypeError(
            "accel_x/accel_y/accel_z must each be None, a number, or an "
            "object exposing .at(t) -> float (got "
            f"{type(source).__name__})."
        )

    def sync(self) -> None:
        """
        Call once per outer step, BEFORE dns_solver.step(). Reads
        dns_solver.time and dns_solver.rho; overwrites
        dns_solver._ext_fx/_ext_fy/_ext_fz in place with the correctly
        density-scaled volumetric seismic pseudo-force.
        """
        if self.dns.rho is None:
            raise RuntimeError(
                "dns_solver.rho is None -- call dns_solver.initialize(...) "
                "before the first SeismicDNSBridge.sync() call."
            )
        t   = float(self.dns.time)
        rho = self.dns.rho
        dev = self.dns.device

        with torch.no_grad():
            if self.accel_x is not None:
                ax = self._eval(self.accel_x, t)
                self.dns._ext_fx = (-rho * ax).to(dev)
            if self.accel_y is not None:
                ay = self._eval(self.accel_y, t)
                self.dns._ext_fy = (-rho * ay).to(dev)
            if self.accel_z is not None:
                az = self._eval(self.accel_z, t)
                self.dns._ext_fz = (-rho * az).to(dev)


# =============================================================================
# 5e. Heat-Release-Rate DNS Bridge Protocol  (ONE Core v3.3)
# =============================================================================

class HeatReleaseDNSBridge:
    """
    Bridge: injects a volumetric heat-release-rate (HRR) source field into
    the DNS/LES compressible solver's energy equation -- for FIRE ONE
    (fire_one.py) combustion/radiation coupling via
    fire_dns_coupling_one.py.

    Physics
    ───────
    Writes dns_solver._ext_q [W/m^3], added directly to rhs_rhoE
    (CompressibleSolver._compute_rhs, v6.5+). This is NOT the same
    mechanism as SeismicDNSBridge's _ext_fx/fy/fz: those enter the energy
    equation only indirectly via mechanical work (f.u) and cannot
    represent a direct heat addition. Combustion heat release (and net
    radiative gain/loss) is a direct volumetric energy source/sink, hence
    the dedicated buffer.

    Source is duck-typed, matching SeismicDNSBridge's convention:
    q_dot is either a constant float [W/m^3] (rare -- HRR is normally
    spatially localized), or any object exposing
    ``.field(t) -> Tensor[(nx,ny,nz)]`` returning the volumetric heat
    source at time t -- e.g. a fire_dns_coupling_one.FireSourceField
    that shapes a design fire's HRR(t) into a plume-like spatial Gaussian
    at the fire's location. This keeps one_core's only hard dependency as
    torch, matching every other bridge in this file.

    IMPORTANT: requires super_dns_one_v6_3.py v6.5+ (the _ext_q buffer
    and its consumption in _compute_rhs). On an older solver, _ext_q will
    simply not exist as an attribute; sync() checks for it explicitly and
    raises a clear error rather than silently creating an unused
    attribute that _compute_rhs never reads (the same failure mode Bug 11
    -- _ext_nu_ch -- taught this ecosystem to guard against).

    Usage::

        bridge = HeatReleaseDNSBridge(dns_solver, q_dot=fire_source_field)
        for step in range(n_steps):
            bridge.sync()          # writes dns_solver._ext_q
            dns_solver.step()
    """

    def __init__(self, dns_solver, q_dot=None) -> None:
        if not hasattr(dns_solver, "_ext_q"):
            raise RuntimeError(
                "dns_solver has no _ext_q buffer -- this bridge requires "
                "super_dns_one_v6_3.py v6.5+ (Bug-11-style fix: _ext_q "
                "declared in __init__ AND consumed in _compute_rhs). "
                "Writing to a nonexistent/unconsumed buffer would silently "
                "produce a no-op simulation."
            )
        self.dns = dns_solver
        self.q_dot = q_dot

    def sync(self) -> None:
        """
        Call once per outer step, BEFORE dns_solver.step(). Overwrites
        dns_solver._ext_q in place (not accumulate, matching every other
        bridge's convention here).
        """
        if self.q_dot is None:
            return
        t = float(self.dns.time)
        dev = self.dns.device
        with torch.no_grad():
            if isinstance(self.q_dot, (int, float)):
                self.dns._ext_q = torch.full_like(self.dns._ext_q, float(self.q_dot)).to(dev)
                return
            field_fn = getattr(self.q_dot, "field", None)
            if not callable(field_fn):
                raise TypeError(
                    "q_dot must be None, a number, or an object exposing "
                    f".field(t) -> Tensor (got {type(self.q_dot).__name__})."
                )
            self.dns._ext_q = field_fn(t).to(dev)


# =============================================================================
# 5f. Pyrolysis Mass-Source DNS Bridge Protocol  (ONE Core v3.4)
# =============================================================================

class PyrolysisDNSBridge:
    """
    Bridge: injects a genuine MASS source for solid-fuel pyrolysis
    (fire_one.PyrolysisModel) or any other real mass-injection process
    (fuel gasification, water-mist evaporation, etc.), in one of two
    modes:

    target='volumetric' (default, v6.8+): distributes mass through a
        thin near-wall VOLUME via the five _ext_mdot* buffers -- a proxy,
        not a true surface condition, but works with any solver grid/BC
        setup and doesn't require a domain face to be configured as a
        pyrolysis wall.

    target='wall' (v6.9+, RECOMMENDED for an actual solid fuel surface):
        writes DIRECTLY into a PyrolysisWallBC attached to a specific
        domain face (dns_solver.bc_objects[wall_face]) -- mass genuinely
        leaves the domain boundary via a Stefan-flow blowing-wall
        condition, the physically correct representation, rather than a
        distributed proxy. Requires that face to have been configured
        with cfg.bc_<face> = 'pyrolysis_wall' when the solver was built.

    Physics (volumetric mode)
    ──────────────────────────
    Writes FIVE buffers together (dns_solver v6.8+): _ext_mdot (mass),
    _ext_mdot_u/v/w (momentum carried by the injected mass at its
    injection velocity), _ext_mdot_e (energy carried by the injected
    mass), and _ext_mdot_Z (mixture-fraction flux -- pyrolyzate is pure
    fuel vapor, Z=1, so _ext_mdot_Z = _ext_mdot for a pyrolysis source).

    Physics (wall mode)
    ────────────────────
    Writes bc.mdot_field_nondim, bc.T_wall, bc.Z_inject on the target
    PyrolysisWallBC, converting from the source's real (dimensional)
    units via dns_solver.T_ref (temperature) and
    dns_solver.mdot_nondim_scale (mass flux) -- the SAME two conversion
    factors already established for the volumetric path and for
    combustion/radiation in v6.6, kept consistent rather than
    introducing a third, different scaling convention. Momentum in wall
    mode is NOT independently prescribed (u_inject/v_inject/w_inject
    keys are ignored in this mode) -- PyrolysisWallBC derives the
    blowing velocity from mass conservation (v_blow = mdot/rho_wall)
    rather than accepting a prescribed injection velocity, which is the
    physically correct treatment for a solid surface (only the
    NORMAL blowing velocity is meaningful; there is no independent
    tangential injection velocity for gas leaving a solid).

    Source is duck-typed: `pyrolysis_source` is an object exposing
    ``.field(t) -> dict``. Volumetric mode reads 'mdot' (kg/(m^3.s)
    Tensor) plus optional 'u_inject'/'v_inject'/'w_inject'/'T_inject_K'/
    'Z_inject'. Wall mode reads 'mdot' as (kg/(m^2.s), a 2D Tensor
    matching the target face's shape) plus optional 'T_inject_K'/
    'Z_inject' (velocity keys ignored, see above).

    IMPORTANT: requires super_dns_one_v6_3.py v6.8+ (volumetric mode) or
    v6.9+ (wall mode, needs PyrolysisWallBC + self.bc_objects). Raises
    clearly at construction otherwise, same pattern as
    HeatReleaseDNSBridge's _ext_q check.

    Usage (wall mode, recommended for an actual fuel surface)::

        # cfg.bc_z_min = 'pyrolysis_wall'  (set before constructing dns_solver)
        bridge = PyrolysisDNSBridge(dns_solver, pyrolysis_source=my_source,
                                     target='wall', wall_face='zmin')
        for step in range(n_steps):
            bridge.sync()
            dns_solver.step()
    """

    def __init__(self, dns_solver, pyrolysis_source=None,
                 target: str = "volumetric", wall_face: str = None) -> None:
        if target not in ("volumetric", "wall"):
            raise ValueError("target must be 'volumetric' or 'wall'")
        self.target = target
        self.dns = dns_solver
        self.source = pyrolysis_source
        self.cp_gas = float(getattr(dns_solver, "cp_gas", 1400.0))

        if target == "volumetric":
            required = ("_ext_mdot", "_ext_mdot_u", "_ext_mdot_v",
                        "_ext_mdot_w", "_ext_mdot_e", "_ext_mdot_Z")
            missing = [b for b in required if not hasattr(dns_solver, b)]
            if missing:
                raise RuntimeError(
                    f"dns_solver is missing buffer(s) {missing} -- this "
                    "bridge (target='volumetric') requires "
                    "super_dns_one_v6_3.py v6.8+ (mass-source buffer "
                    "family added to continuity). Writing to a "
                    "nonexistent/unconsumed buffer would silently produce "
                    "a no-op simulation (the exact failure mode "
                    "_ext_nu_ch taught this ecosystem to guard against)."
                )
        else:
            if wall_face is None:
                raise ValueError("wall_face is required when target='wall' "
                                  "(e.g. 'zmin', 'xmax', ...)")
            bc_objects = getattr(dns_solver, "bc_objects", None)
            if bc_objects is None or wall_face not in bc_objects:
                raise RuntimeError(
                    f"dns_solver.bc_objects['{wall_face}'] not found -- "
                    "this bridge (target='wall') requires "
                    "super_dns_one_v6_3.py v6.9+ and cfg.bc_<face> = "
                    "'pyrolysis_wall' to have been set for that face "
                    "before the solver was constructed."
                )
            bc = bc_objects[wall_face]
            if not hasattr(bc, "mdot_field_nondim"):
                raise RuntimeError(
                    f"dns_solver.bc_objects['{wall_face}'] is a "
                    f"{type(bc).__name__}, not a PyrolysisWallBC -- set "
                    f"cfg.bc_{wall_face.replace('min','_min').replace('max','_max')} "
                    "= 'pyrolysis_wall' when constructing the solver."
                )
            if not hasattr(dns_solver, "T_ref") or not hasattr(dns_solver, "mdot_nondim_scale"):
                raise RuntimeError(
                    "dns_solver is missing T_ref/mdot_nondim_scale -- "
                    "target='wall' requires super_dns_one_v6_3.py v6.9+."
                )
            self.wall_face = wall_face
            self.bc = bc

    def sync(self) -> None:
        """
        Call once per outer step, BEFORE dns_solver.step().

        Volumetric mode: overwrites all five _ext_mdot* buffers together
        (not accumulate), so a partial field dict still produces a fully
        self-consistent set of buffers each call -- missing keys default
        to physically well-defined values (rest injection, T=300K, pure
        fuel) rather than leaving stale values from a previous call.

        Wall mode: overwrites bc.mdot_field_nondim/T_wall/Z_inject on the
        target PyrolysisWallBC; source=None or a zero/absent 'mdot' sets
        bc.mdot_field_nondim back to None, which makes PyrolysisWallBC
        fall back to a plain NoSlipIsothermalWallBC for that step (no
        stale injection persists once the source stops).
        """
        if self.source is None:
            return
        field_fn = getattr(self.source, "field", None)
        if not callable(field_fn):
            raise TypeError(
                "pyrolysis_source must be None or an object exposing "
                f".field(t) -> dict (got {type(self.source).__name__})."
            )
        t = float(self.dns.time)
        dev = self.dns.device

        with torch.no_grad():
            f = field_fn(t)
            mdot = f["mdot"].to(dev)
            T_inj_K = f.get("T_inject_K", 300.0)
            Z_inj = f.get("Z_inject", 1.0)

            if self.target == "volumetric":
                u_inj = f.get("u_inject", torch.zeros_like(mdot))
                v_inj = f.get("v_inject", torch.zeros_like(mdot))
                w_inj = f.get("w_inject", torch.zeros_like(mdot))
                self.dns._ext_mdot   = mdot
                self.dns._ext_mdot_u = (mdot * u_inj).to(dev)
                self.dns._ext_mdot_v = (mdot * v_inj).to(dev)
                self.dns._ext_mdot_w = (mdot * w_inj).to(dev)
                self.dns._ext_mdot_e = (mdot * self.cp_gas * T_inj_K).to(dev)
                self.dns._ext_mdot_Z = (mdot * Z_inj).to(dev)
            else:
                if float(mdot.abs().max()) < 1e-30:
                    self.bc.mdot_field_nondim = None
                    return
                self.bc.mdot_field_nondim = (mdot * self.dns.mdot_nondim_scale).to(dev)
                self.bc.T_wall = T_inj_K / self.dns.T_ref
                self.bc.Z_inject = Z_inj


# =============================================================================
# 5c. Structural Biharmonic Operator (module-level, ONE Core v3.1)
# =============================================================================

def structural_biharmonic_n(
    field:        torch.Tensor,
    sigma:        torch.Tensor,
    n:            int,
    laplacian_fn,
) -> torch.Tensor:
    """
    Compute Δ_S^n u recursively (Section 3.1 of Limsuwan 2026).

    Promoted to ``one_core`` so that any solver in the ONE Ecosystem
    can call it without importing the full CH solver class.

    Parameters
    ----------
    field        : (Nx, Ny, Nz) input field
    sigma        : (Nx, Ny, Nz) structural regime field
    n            : operator order (n≥1)
    laplacian_fn : callable(field, sigma) → (Nx, Ny, Nz)

    Returns
    -------
    (Nx, Ny, Nz) = Δ_S^n field
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    result = laplacian_fn(field, sigma)
    for _ in range(n - 1):
        result = laplacian_fn(result, sigma)
    return result


# =============================================================================
# 6. Version banner (import-time)
# =============================================================================

logger.debug("ONE Core v%s loaded.", ONE_VERSION)
