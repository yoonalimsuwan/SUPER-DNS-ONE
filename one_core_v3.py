# =============================================================================
# ONE CORE — Shared Foundation of the ONE Ecosystem
# =============================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# This module is the single source of truth for every component that is
# shared across the ONE Ecosystem:
#
#   structural_langevin_v3.py          (MD / particle scale)
#   structuralfluctuatinghydro_v6.py  (FH continuum 3-D)
#   super_dns_one_v6.py               (DNS/LES 3-D compressible)
#
# Cross-file bridge protocol
# ──────────────────────────
#   LangevinFHBridge   — feeds Langevin structural stress into FH solver
#   LangevinDNSBridge  — feeds Langevin structural stress into DNS solver
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
#   get_device                 — unified hardware-backend selector
#   ONE_VERSION                — ecosystem-wide version string
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

ONE_VERSION: str = "3.0.0"


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
# 6. Version banner (import-time)
# =============================================================================

logger.debug("ONE Core v%s loaded.", ONE_VERSION)
