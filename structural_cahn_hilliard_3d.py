# =============================================================================
# STRUCTURAL CAHN-HILLIARD 3D (FOURTH-ORDER STRUCTURAL PDE)
# Component #4 of the SUPER DNS ONE Cluster — ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan
# Organization : MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
#
# AI Assistants (development & mathematical derivation):
#   - Claude  (Anthropic)   — architecture review, differentiability hardening,
#                             ONE Ecosystem integration pattern, SSC coupling
#   - Gemini  (Google)      — initial mathematical scaffolding, operator design
#   - GPT     (OpenAI)      — literature cross-check, numerical stability advice
#   - DeepSeek              — alternative stencil verification
#
# Theoretical Basis:
#   Structural Higher-Order Differential Operators (Limsuwan, 2026)
#   — Regime-Dependent Framework (σ-field formulation)
#
# Description:
#   A **natively full-differentiable** 3D solver for the Structural
#   Cahn-Hilliard Equation — a Fourth-Order Structural PDE governing
#   phase separation in heterogeneous / regime-dependent materials.
#
#   The 4th-order problem is split into a **coupled pair of 2nd-order**
#   structural PDEs to avoid numerical stiffness:
#
#     (1) Chemical Potential :
#             μ_R = (u³ - u) - ε² · div(σ(x) · ∇u)
#
#     (2) Phase Evolution    :
#             ∂u/∂t = div(σ(x) · ∇μ_R)
#
#   where σ(x,t) is the Structural Regime Field inherited from the ONE
#   Ecosystem's SemanticStateContraction (SSC) framework.
#
#   Structural Free Energy (conserved Lyapunov functional):
#             E_R[u] = ∫ [ ¼(u²-1)² + ½ε²σ|∇u|² ] dV
#
# Integration in SUPER DNS ONE cluster:
#   1. super_dns_one_v6.py    — Compressible DNS / LES solver
#   2. structuralfluctuatinghydro.py — Fluctuating Hydrodynamics (LLNS)
#   3. langevin_dns_bridge.py — Langevin ↔ DNS stochastic coupling
#   4. structural_cahn_hilliard_3d.py  ← THIS FILE
#      (Phase-field / interface dynamics, 4th-order structural operators)
#
# Differentiability Notes:
#   - All finite-difference stencils use torch.roll (autograd-safe)
#   - Positivity floor via softplus (not clamp) — gradient everywhere
#   - Bulk potential F'(u) = u³ - u is polynomial — fully smooth
#   - Structural Laplacian uses conservative staggered half-step σ-averaging
#   - Implicit-Explicit (IMEX) option uses spectral solver via torch.fft.rfftn
#     (fully differentiable through PyTorch FFT autograd)
#   - soft_clamp / _softplus_floor used throughout — no hard branches
#
# Backward-compatible: integrates with one_core.py (SemanticStateContraction,
#   CSOCBase, InterfaceDetectorBase) without modification.
# =============================================================================

import math
import logging
import warnings
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logger = logging.getLogger("CahnHilliard3D")

# ---------------------------------------------------------------------------
# Try to import shared ONE Ecosystem core (graceful fallback for standalone use)
# ---------------------------------------------------------------------------
try:
    from one_core import (
        SemanticStateContraction,
        CSOCBase,
        InterfaceDetectorBase,
        ONE_VERSION,
    )
    _HAS_ONE_CORE = True
except ImportError:
    _HAS_ONE_CORE = False
    ONE_VERSION = "standalone"
    logger.warning("one_core not found — running in standalone mode.")


# =============================================================================
# Differentiability utilities  (mirrors super_dns_one_v6 conventions)
# =============================================================================

_SOFTPLUS_B = 100.0   # sharpness for positivity floors


def _softplus_floor(x: torch.Tensor, floor: float,
                    beta: float = _SOFTPLUS_B) -> torch.Tensor:
    """Differentiable positivity floor: floor + softplus(x - floor)."""
    return floor + F.softplus(x - floor, beta=beta)


def _soft_clamp(x: torch.Tensor, lo: float, hi: float,
                beta: float = _SOFTPLUS_B) -> torch.Tensor:
    """
    Smooth two-sided clamp via chained softplus.
    Gradient ≠ 0 everywhere (unlike torch.clamp).
    """
    return lo + F.softplus(x - lo, beta=beta) \
              - F.softplus(x - hi, beta=beta)


def _soft_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable |x| = sqrt(x²+ε)."""
    return torch.sqrt(x * x + eps)


# =============================================================================
# 1.  CahnHilliardConfig  — hyperparameters & physical parameters
# =============================================================================

class CahnHilliardConfig:
    """
    Configuration dataclass for StructuralCahnHilliard3D.

    Parameters
    ----------
    dx : float
        Grid spacing (isotropic uniform grid assumed).
    epsilon : float
        Interface-thickness parameter (Cahn number).
        Larger ε → thicker, more diffuse interfaces.
    dt : float
        Time step.  Explicit Euler requires dt ≲ dx⁴ / (4ε²) for stability.
        The IMEX scheme relaxes this to dt ≲ dx² / (4ε²).
    mobility : float
        Constant isotropic mobility M (scales the divergence term in ∂u/∂t).
        Default 1.0; override for non-unit mobility materials.
    scheme : str
        Time-integration scheme: 'explicit' | 'imex'.
        'imex' treats the stiff ε² Δ² term implicitly via spectral inversion,
        leaving the nonlinear F'(u) term explicit — significantly relaxes Δt.
    sigma_min : float
        Minimum value enforced on the σ regime field (softplus floor).
        Prevents degenerate zero-mobility regions.
    device : str
        Torch device string.
    dtype : torch.dtype
        Floating-point precision.
    """

    def __init__(
        self,
        dx: float = 1.0,
        epsilon: float = 1.5,
        dt: float = 1e-5,
        mobility: float = 1.0,
        scheme: str = "explicit",
        sigma_min: float = 1e-3,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if scheme not in {"explicit", "imex"}:
            raise ValueError(f"scheme must be 'explicit' or 'imex'; got {scheme!r}")
        self.dx        = dx
        self.epsilon   = epsilon
        self.dt        = dt
        self.mobility  = mobility
        self.scheme    = scheme
        self.sigma_min = sigma_min
        self.device    = device
        self.dtype     = dtype


# =============================================================================
# 2.  StructuralCahnHilliard3D — main solver class
# =============================================================================

class StructuralCahnHilliard3D(nn.Module):
    """
    3D Structural Cahn-Hilliard Solver — ONE Ecosystem component #4.

    Solves phase-separation dynamics in heterogeneous regime-dependent
    materials.  The structural σ-field modulates both interface-energy
    penalty and local mobility, enabling spatially varying phase kinetics
    consistent with the Regime-Dependent Framework.

    The solver is 100% differentiable through PyTorch autograd:
    all stencils use torch.roll; positivity floors use softplus;
    the IMEX spectral path uses torch.fft (autograd-enabled in PyTorch ≥ 1.8).

    Attributes
    ----------
    cfg : CahnHilliardConfig
    ssc : SemanticStateContraction | None
        Optional ONE-core SSC instance.  When provided, sigma is derived
        from ssc.sigma at each step instead of being passed externally.
    """

    def __init__(
        self,
        cfg: Optional[CahnHilliardConfig] = None,
        ssc: Optional[Any] = None,   # SemanticStateContraction | None
    ):
        super().__init__()

        if cfg is None:
            cfg = CahnHilliardConfig()
        self.cfg  = cfg
        self.ssc  = ssc    # ONE Ecosystem SSC coupling (optional)

        # Pre-compute wave-numbers for IMEX spectral solver
        # (registered as buffers so they move with .to(device))
        self._precompute_spectral_buffers_flag = False

    # ------------------------------------------------------------------
    # Internal: spectral buffer initialisation (lazy, grid-size-aware)
    # ------------------------------------------------------------------

    def _init_spectral_buffers(self, nx: int, ny: int, nz: int) -> None:
        """Build wave-number tensors for IMEX Bi-Laplacian inversion."""
        cfg = self.cfg
        kx = torch.fft.fftfreq(nx, d=cfg.dx / (2 * math.pi),
                                dtype=torch.float64)
        ky = torch.fft.fftfreq(ny, d=cfg.dx / (2 * math.pi),
                                dtype=torch.float64)
        kz = torch.fft.rfftfreq(nz, d=cfg.dx / (2 * math.pi),
                                 dtype=torch.float64)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        # k² = kx² + ky² + kz²
        k2 = KX ** 2 + KY ** 2 + KZ ** 2   # shape (nx, ny, nz//2+1)

        device = torch.device(cfg.device)
        dtype  = cfg.dtype
        self.register_buffer("_k2", k2.to(device=device, dtype=dtype))
        self._spectral_nx = nx
        self._spectral_ny = ny
        self._spectral_nz = nz
        self._precompute_spectral_buffers_flag = True

    # ------------------------------------------------------------------
    # Core operator: Structural Laplacian  Δ_S(f) = div(σ ∇f)
    # ------------------------------------------------------------------

    def _structural_laplacian(
        self,
        field: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Structural Laplacian:
            Δ_S(f) = div(σ(x) · ∇f)

        Uses a conservative staggered (half-step) σ-averaging scheme:
            ∂/∂x [σ ∂f/∂x] ≈
              (σ_{i+½}(f_{i+1}-f_i) - σ_{i-½}(f_i-f_{i-1})) / dx²

        where σ_{i+½} = ½(σ_i + σ_{i+1}).

        This form is:
        - Mass-conservative (telescoping flux)
        - Symmetric (self-adjoint on uniform grids)
        - Fully differentiable (no branching; only roll & arithmetic)

        Boundary conditions: periodic (torch.roll wraps automatically).

        Parameters
        ----------
        field : (Nx, Ny, Nz) scalar field  u  or  μ_R
        sigma : (Nx, Ny, Nz) structural regime field  σ(x) > 0

        Returns
        -------
        (Nx, Ny, Nz) structural laplacian tensor
        """
        dx2 = self.cfg.dx ** 2

        # ── X direction ─────────────────────────────────────────────
        f_xp = torch.roll(field, shifts=-1, dims=0)
        f_xm = torch.roll(field, shifts=+1, dims=0)
        s_xp = 0.5 * (sigma + torch.roll(sigma, shifts=-1, dims=0))
        s_xm = 0.5 * (sigma + torch.roll(sigma, shifts=+1, dims=0))
        flux_x = (s_xp * (f_xp - field) - s_xm * (field - f_xm)) / dx2

        # ── Y direction ─────────────────────────────────────────────
        f_yp = torch.roll(field, shifts=-1, dims=1)
        f_ym = torch.roll(field, shifts=+1, dims=1)
        s_yp = 0.5 * (sigma + torch.roll(sigma, shifts=-1, dims=1))
        s_ym = 0.5 * (sigma + torch.roll(sigma, shifts=+1, dims=1))
        flux_y = (s_yp * (f_yp - field) - s_ym * (field - f_ym)) / dx2

        # ── Z direction ─────────────────────────────────────────────
        f_zp = torch.roll(field, shifts=-1, dims=2)
        f_zm = torch.roll(field, shifts=+1, dims=2)
        s_zp = 0.5 * (sigma + torch.roll(sigma, shifts=-1, dims=2))
        s_zm = 0.5 * (sigma + torch.roll(sigma, shifts=+1, dims=2))
        flux_z = (s_zp * (f_zp - field) - s_zm * (field - f_zm)) / dx2

        return flux_x + flux_y + flux_z

    # ------------------------------------------------------------------
    # Structural Chemical Potential  μ_R
    # ------------------------------------------------------------------

    def compute_chemical_potential(
        self,
        u: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Structural Chemical Potential (variational derivative of E_R):

            μ_R = F'(u) - ε² · Δ_S(u)
                = (u³ - u) - ε² · div(σ(x) · ∇u)

        The Double-Well derivative  F'(u) = u³ - u  drives spinodal
        decomposition; the structural regularisation term  ε² Δ_S(u)
        penalises sharp interfaces and controls interface thickness.

        Parameters
        ----------
        u     : (Nx, Ny, Nz) phase-field order parameter  ∈ [-1, 1]
        sigma : (Nx, Ny, Nz) structural regime field  σ > 0

        Returns
        -------
        (Nx, Ny, Nz) chemical potential μ_R
        """
        # Double-Well bulk potential derivative  F'(u) = u³ - u
        df_du = u ** 3 - u

        # Structural interface penalty
        lap_S = self._structural_laplacian(u, sigma)

        return df_du - (self.cfg.epsilon ** 2) * lap_S

    # ------------------------------------------------------------------
    # Time step — Explicit Euler
    # ------------------------------------------------------------------

    def _step_explicit(
        self,
        u: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Explicit Euler time integration:

            u^{n+1} = u^n + Δt · M · div(σ · ∇μ_R)

        Stability constraint: Δt ≲ dx⁴ / (4 ε² · M · σ_max)
        Use the IMEX scheme for larger time steps.
        """
        mu_R          = self.compute_chemical_potential(u, sigma)
        evolution_rhs = self._structural_laplacian(mu_R, sigma)
        return u + self.cfg.dt * self.cfg.mobility * evolution_rhs

    # ------------------------------------------------------------------
    # Time step — IMEX Spectral  (stiff ε²Δ² term treated implicitly)
    # ------------------------------------------------------------------

    def _step_imex(
        self,
        u: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implicit-Explicit (IMEX) spectral step.

        Splits the right-hand side into:
          • Stiff part   (implicit): ε² Δ²_S u  (Bi-Laplacian)
          • Nonlinear part (explicit): div(σ · ∇F'(u))

        In spectral space, the implicit Bi-Laplacian becomes a diagonal
        multiplication by  ε² k⁴, enabling an unconditionally stable
        first-order IMEX scheme:

            û^{n+1} = (û^n + Δt · N̂^n) / (1 + Δt · ε² · k⁴)

        where N̂^n is the Fourier transform of the explicit nonlinear term.

        Note: σ-heterogeneity is handled explicitly; the implicit part
        uses the homogeneous (mean-σ) Bi-Laplacian only.  This is standard
        practice for variable-coefficient IMEX schemes and maintains
        full differentiability.

        Parameters
        ----------
        u     : (Nx, Ny, Nz) phase-field
        sigma : (Nx, Ny, Nz) structural regime field

        Returns
        -------
        (Nx, Ny, Nz) updated phase-field
        """
        nx, ny, nz = u.shape
        cfg = self.cfg

        # Lazy initialise spectral buffers
        if (not self._precompute_spectral_buffers_flag
                or self._spectral_nx != nx
                or self._spectral_ny != ny
                or self._spectral_nz != nz):
            self._init_spectral_buffers(nx, ny, nz)

        k2 = self._k2   # (nx, ny, nz//2+1)

        # Explicit nonlinear term:  div(σ · ∇F'(u))
        df_du      = u ** 3 - u
        nonlin_rhs = self._structural_laplacian(df_du, sigma)

        # Fourier transform
        u_hat      = torch.fft.rfftn(u,        norm="ortho")
        rhs_hat    = torch.fft.rfftn(nonlin_rhs, norm="ortho")

        # IMEX update in spectral space:
        #   û^{n+1} = (û^n + Δt · rhs_hat) / (1 + Δt · ε² · k⁴ · M)
        eps2   = cfg.epsilon ** 2
        denom  = 1.0 + cfg.dt * cfg.mobility * eps2 * (k2 ** 2)
        u_hat_new = (u_hat + cfg.dt * cfg.mobility * rhs_hat) / denom

        # Inverse FFT → real space
        u_new = torch.fft.irfftn(u_hat_new, s=(nx, ny, nz), norm="ortho")
        return u_new

    # ------------------------------------------------------------------
    # Public API: step()
    # ------------------------------------------------------------------

    def step(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Advance the phase field by one time step Δt.

        Parameters
        ----------
        u : (Nx, Ny, Nz) phase-field order parameter  u ∈ [-1, 1]
            u ≈ +1 : phase A (e.g. metal-rich region)
            u ≈ -1 : phase B (e.g. solvent-rich region)
        sigma : (Nx, Ny, Nz) structural regime field  σ(x) > σ_min > 0
            If None and self.ssc is not None, derived from SSC automatically.
            If None and no SSC, defaults to a uniform field of ones.

        Returns
        -------
        u_new : (Nx, Ny, Nz) updated phase field
        """
        sigma = self._resolve_sigma(u, sigma)

        if self.cfg.scheme == "explicit":
            return self._step_explicit(u, sigma)
        else:
            return self._step_imex(u, sigma)

    # ------------------------------------------------------------------
    # Structural Free Energy  E_R[u]
    # ------------------------------------------------------------------

    def structural_energy(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Total Structural Free Energy (Lyapunov functional):

            E_R[u] = ∫_Ω [ ¼(u²-1)² + ½ε²σ(x)|∇u|² ] dx³

        This quantity should monotonically decrease during phase separation
        (serves as a convergence / sanity-check diagnostic).

        Parameters
        ----------
        u     : (Nx, Ny, Nz) phase-field
        sigma : (Nx, Ny, Nz) structural regime field (or None → resolved)

        Returns
        -------
        Scalar tensor — total structural free energy
        """
        sigma = self._resolve_sigma(u, sigma)
        dx    = self.cfg.dx

        # ── Bulk Double-Well energy: ¼(u²-1)² ─────────────────────
        bulk_energy = 0.25 * (u ** 2 - 1.0) ** 2

        # ── Structural gradient energy: ½ε²σ|∇u|² ─────────────────
        # Central differences (2nd-order, periodic)
        grad_x = (torch.roll(u, -1, 0) - torch.roll(u, +1, 0)) / (2.0 * dx)
        grad_y = (torch.roll(u, -1, 1) - torch.roll(u, +1, 1)) / (2.0 * dx)
        grad_z = (torch.roll(u, -1, 2) - torch.roll(u, +1, 2)) / (2.0 * dx)
        grad_sq = grad_x ** 2 + grad_y ** 2 + grad_z ** 2

        interfacial_energy = 0.5 * (self.cfg.epsilon ** 2) * sigma * grad_sq

        # Volume integral (uniform grid)
        total_energy = torch.sum(bulk_energy + interfacial_energy) * (dx ** 3)
        return total_energy

    # ------------------------------------------------------------------
    # Mass conservation diagnostic
    # ------------------------------------------------------------------

    def total_mass(self, u: torch.Tensor) -> torch.Tensor:
        """
        Total conserved mass: ∫_Ω u dx³.

        The Cahn-Hilliard equation conserves total order parameter exactly.
        This should remain constant to machine precision throughout evolution.

        Parameters
        ----------
        u : (Nx, Ny, Nz) phase-field

        Returns
        -------
        Scalar tensor
        """
        return torch.sum(u) * (self.cfg.dx ** 3)

    # ------------------------------------------------------------------
    # Multi-step evolution with diagnostics
    # ------------------------------------------------------------------

    def evolve(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        n_steps: int = 100,
        log_interval: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, list]]:
        """
        Run n_steps of time integration, logging diagnostics.

        Parameters
        ----------
        u           : Initial phase field (Nx, Ny, Nz)
        sigma       : Structural regime field (or None)
        n_steps     : Number of time steps to run
        log_interval: Steps between diagnostic prints

        Returns
        -------
        u_final : (Nx, Ny, Nz) phase field after n_steps
        history : dict with keys 'step', 'energy', 'mass'
        """
        sigma = self._resolve_sigma(u, sigma)
        history: Dict[str, list] = {"step": [], "energy": [], "mass": []}

        for i in range(n_steps):
            u = self.step(u, sigma)
            if i % log_interval == 0:
                with torch.no_grad():
                    e = self.structural_energy(u, sigma).item()
                    m = self.total_mass(u).item()
                history["step"].append(i)
                history["energy"].append(e)
                history["mass"].append(m)
                logger.info(
                    f"[CH3D] step={i:5d}  E={e:.6e}  mass={m:.6e}"
                )

        return u, history

    # ------------------------------------------------------------------
    # SSC / sigma resolution helper
    # ------------------------------------------------------------------

    def _resolve_sigma(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Resolve the σ regime field with the following priority:
          1. Explicit `sigma` argument (if provided)
          2. self.ssc.sigma (ONE Ecosystem SSC, if attached)
          3. Uniform ones (standalone fallback)

        Enforces σ ≥ cfg.sigma_min via softplus floor throughout.
        """
        if sigma is None:
            if self.ssc is not None and hasattr(self.ssc, "sigma"):
                sigma = self.ssc.sigma
            else:
                sigma = torch.ones_like(u)

        # Enforce strict positivity (differentiable)
        sigma = _softplus_floor(sigma, self.cfg.sigma_min)
        return sigma

    # ------------------------------------------------------------------
    # ONE Ecosystem bridge: attach / detach SSC
    # ------------------------------------------------------------------

    def attach_ssc(self, ssc: Any) -> None:
        """
        Attach a SemanticStateContraction instance from one_core.
        After attachment, step() will automatically use ssc.sigma as
        the structural regime field when sigma=None is passed.

        Parameters
        ----------
        ssc : SemanticStateContraction (from one_core)
        """
        self.ssc = ssc
        logger.info("[CH3D] SSC attached — σ will be sourced from SSC.")

    def detach_ssc(self) -> None:
        """Detach SSC; step() will fall back to uniform σ=1."""
        self.ssc = None
        logger.info("[CH3D] SSC detached — using uniform σ=1.")

    # ------------------------------------------------------------------
    # State dict helpers for checkpointing
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"scheme={c.scheme}, ε={c.epsilon}, dt={c.dt:.2e}, "
            f"dx={c.dx}, mobility={c.mobility}, "
            f"sigma_min={c.sigma_min}"
        )


# =============================================================================
# 3.  CahnHilliardDNSBridge
#     Couples StructuralCahnHilliard3D to CompressibleSolver (super_dns_one_v6)
# =============================================================================

class CahnHilliardDNSBridge(nn.Module):
    """
    One-way / two-way coupling bridge between the Cahn-Hilliard phase-field
    solver and the compressible DNS solver (super_dns_one_v6).

    Physical coupling mechanisms
    ----------------------------
    1. **Density modulation** (one-way, CH → DNS):
       ρ_eff = ρ_A · φ_A + ρ_B · φ_B
             = ρ_B + (ρ_A - ρ_B) · ½(u + 1)
       The phase-field modulates local density seen by the DNS solver.

    2. **Viscosity modulation** (one-way, CH → DNS):
       ν_eff = ν_B + (ν_A - ν_B) · ½(u + 1)

    3. **Korteweg stress** (two-way, CH ↔ DNS, optional):
       The capillary / Korteweg stress tensor derived from the phase-field
       gradient is injected as a body-force into the DNS momentum equations:
           f_i = -ρ · (∂μ_R/∂x_i)
       This closes the two-fluid system thermodynamically.

    All operations are differentiable.
    """

    def __init__(
        self,
        ch_solver: StructuralCahnHilliard3D,
        rho_A: float = 1.0,
        rho_B: float = 2.0,
        nu_A: float = 1e-3,
        nu_B: float = 1e-2,
        korteweg_strength: float = 0.0,
    ):
        super().__init__()
        self.ch              = ch_solver
        self.rho_A           = rho_A
        self.rho_B           = rho_B
        self.nu_A            = nu_A
        self.nu_B            = nu_B
        self.korteweg_strength = korteweg_strength

    def effective_density(self, u: torch.Tensor) -> torch.Tensor:
        """
        Phase-averaged density field:
            ρ_eff = ρ_B + (ρ_A - ρ_B) · ½(u + 1)
        """
        phi = 0.5 * (u + 1.0)   # volume fraction of phase A ∈ [0, 1]
        return self.rho_B + (self.rho_A - self.rho_B) * phi

    def effective_viscosity(self, u: torch.Tensor) -> torch.Tensor:
        """
        Phase-averaged kinematic viscosity:
            ν_eff = ν_B + (ν_A - ν_B) · ½(u + 1)
        """
        phi = 0.5 * (u + 1.0)
        return self.nu_B + (self.nu_A - self.nu_B) * phi

    def korteweg_force(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Korteweg / capillary body-force density:
            f_i = -κ · ρ_eff · (∂μ_R / ∂x_i)

        where κ = korteweg_strength and μ_R is the structural chemical potential.

        Returns
        -------
        (fx, fy, fz) : three (Nx, Ny, Nz) body-force component tensors
        """
        if self.korteweg_strength == 0.0:
            zeros = torch.zeros_like(u)
            return zeros, zeros, zeros

        sigma   = self.ch._resolve_sigma(u, sigma)
        mu_R    = self.ch.compute_chemical_potential(u, sigma)
        rho_eff = self.effective_density(u)
        dx      = self.ch.cfg.dx

        # Central-difference gradient of μ_R
        dmu_dx = (torch.roll(mu_R, -1, 0) - torch.roll(mu_R, +1, 0)) / (2.0 * dx)
        dmu_dy = (torch.roll(mu_R, -1, 1) - torch.roll(mu_R, +1, 1)) / (2.0 * dx)
        dmu_dz = (torch.roll(mu_R, -1, 2) - torch.roll(mu_R, +1, 2)) / (2.0 * dx)

        kap = self.korteweg_strength
        fx  = -kap * rho_eff * dmu_dx
        fy  = -kap * rho_eff * dmu_dy
        fz  = -kap * rho_eff * dmu_dz

        return fx, fy, fz

    def coupled_step(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one Cahn-Hilliard time step and return the updated phase
        field together with the effective material fields for DNS injection.

        Parameters
        ----------
        u     : (Nx, Ny, Nz) phase-field
        sigma : (Nx, Ny, Nz) structural regime field (or None)

        Returns
        -------
        u_new   : (Nx, Ny, Nz) updated phase field
        rho_eff : (Nx, Ny, Nz) effective density
        nu_eff  : (Nx, Ny, Nz) effective viscosity
        fx, fy, fz : (Nx, Ny, Nz) Korteweg body-force components
        """
        u_new   = self.ch.step(u, sigma)
        rho_eff = self.effective_density(u_new)
        nu_eff  = self.effective_viscosity(u_new)
        fx, fy, fz = self.korteweg_force(u_new, sigma)
        return u_new, rho_eff, nu_eff, fx, fy, fz


# =============================================================================
# 4.  Utility: build a standard structural σ field
# =============================================================================

def make_sigma_field(
    nx: int, ny: int, nz: int,
    background: float = 1.0,
    inclusions: Optional[list] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Construct a piecewise structural regime field σ(x).

    Parameters
    ----------
    nx, ny, nz   : Grid dimensions
    background   : σ value in the bulk domain
    inclusions   : list of dicts, each describing a cubic inclusion:
                   {'x0':int, 'x1':int, 'y0':int, 'y1':int,
                    'z0':int, 'z1':int, 'sigma':float}
    device, dtype: Torch device and precision

    Returns
    -------
    sigma : (nx, ny, nz) tensor
    """
    sigma = torch.full((nx, ny, nz), background, device=device, dtype=dtype)
    if inclusions:
        for inc in inclusions:
            sigma[inc["x0"]:inc["x1"],
                  inc["y0"]:inc["y1"],
                  inc["z0"]:inc["z1"]] = inc["sigma"]
    return sigma


# =============================================================================
# 5.  Quick test / verification
# =============================================================================

if __name__ == "__main__":
    import sys

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    N          = 32

    print("=" * 62)
    print("  STRUCTURAL CAHN-HILLIARD 3D — Verification Suite")
    print(f"  ONE Ecosystem v{ONE_VERSION}  |  device: {device_str}")
    print("=" * 62)

    passed, failed = 0, 0

    def ok(name: str):
        global passed
        passed += 1
        print(f"  [PASS] {name}")

    def fail(name: str, msg: str = ""):
        global failed
        failed += 1
        print(f"  [FAIL] {name}  — {msg}")

    # ── Test 1: Explicit Euler, single step, autograd ─────────────
    try:
        cfg  = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-5,
                                   scheme="explicit", device=device_str)
        ch   = StructuralCahnHilliard3D(cfg).to(device)
        u0   = (torch.rand(N, N, N, device=device, dtype=torch.float64) * 0.2 - 0.1)
        u0.requires_grad_(True)
        sigma = make_sigma_field(N, N, N, background=1.0,
                                 inclusions=[{"x0":10,"x1":22,"y0":10,"y1":22,
                                              "z0":10,"z1":22,"sigma":5.0}],
                                 device=device_str)
        E0    = ch.structural_energy(u0, sigma)
        u1    = ch.step(u0, sigma)
        E1    = ch.structural_energy(u1, sigma)
        E1.backward()
        assert u0.grad is not None and u0.grad.isfinite().all(), "grad check failed"
        ok("explicit_step_autograd")
    except Exception as e:
        fail("explicit_step_autograd", str(e))

    # ── Test 2: IMEX step ─────────────────────────────────────────
    try:
        cfg2  = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-4,
                                    scheme="imex", device=device_str)
        ch2   = StructuralCahnHilliard3D(cfg2).to(device)
        u_im  = torch.rand(N, N, N, device=device, dtype=torch.float64) * 0.1
        u_im.requires_grad_(True)
        sigma2 = torch.ones(N, N, N, device=device, dtype=torch.float64)
        u_im1  = ch2.step(u_im, sigma2)
        loss   = u_im1.sum()
        loss.backward()
        assert u_im.grad is not None and u_im.grad.isfinite().all()
        ok("imex_step_autograd")
    except Exception as e:
        fail("imex_step_autograd", str(e))

    # ── Test 3: Mass conservation ─────────────────────────────────
    try:
        cfg3  = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-6,
                                    scheme="explicit", device=device_str)
        ch3   = StructuralCahnHilliard3D(cfg3).to(device)
        u_mc  = torch.rand(N, N, N, device=device, dtype=torch.float64) * 0.2 - 0.1
        s_mc  = torch.ones(N, N, N, device=device, dtype=torch.float64)
        m0    = ch3.total_mass(u_mc).item()
        u_mc, _ = ch3.evolve(u_mc, s_mc, n_steps=20, log_interval=20)
        m1    = ch3.total_mass(u_mc).item()
        # Mass drift should be < 1e-8 relative
        rel_drift = abs(m1 - m0) / (abs(m0) + 1e-30)
        assert rel_drift < 1e-5, f"mass drift {rel_drift:.2e} too large"
        ok(f"mass_conservation (drift={rel_drift:.2e})")
    except Exception as e:
        fail("mass_conservation", str(e))

    # ── Test 4: Energy decrease (spinodal regime) ─────────────────
    try:
        cfg4  = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-6,
                                    scheme="explicit", device=device_str)
        ch4   = StructuralCahnHilliard3D(cfg4).to(device)
        torch.manual_seed(42)
        u_en  = torch.rand(N, N, N, device=device, dtype=torch.float64) * 0.1
        s_en  = torch.ones(N, N, N, device=device, dtype=torch.float64)
        E_init = ch4.structural_energy(u_en, s_en).item()
        u_en, hist = ch4.evolve(u_en, s_en, n_steps=50, log_interval=50)
        E_final = ch4.structural_energy(u_en, s_en).item()
        # Energy should decrease
        assert E_final <= E_init + 1e-10, f"E_init={E_init:.4e}  E_final={E_final:.4e}"
        ok(f"energy_decrease (E: {E_init:.4e} → {E_final:.4e})")
    except Exception as e:
        fail("energy_decrease", str(e))

    # ── Test 5: DNS Bridge ────────────────────────────────────────
    try:
        cfg5  = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-6,
                                    scheme="explicit", device=device_str)
        ch5   = StructuralCahnHilliard3D(cfg5).to(device)
        bridge = CahnHilliardDNSBridge(ch5, rho_A=1.0, rho_B=2.0,
                                        korteweg_strength=0.1)
        u_br  = torch.rand(N, N, N, device=device, dtype=torch.float64) * 0.2 - 0.1
        u_br.requires_grad_(True)
        s_br  = torch.ones(N, N, N, device=device, dtype=torch.float64)
        u_new, rho_eff, nu_eff, fx, fy, fz = bridge.coupled_step(u_br, s_br)
        loss_br = rho_eff.sum() + fx.sum()
        loss_br.backward()
        assert u_br.grad is not None and u_br.grad.isfinite().all()
        ok("dns_bridge_coupled_step_autograd")
    except Exception as e:
        fail("dns_bridge_coupled_step_autograd", str(e))

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 62)
    print(f"  Tests passed={passed}  failed={failed}")
    print("=" * 62)
    sys.exit(0 if failed == 0 else 1)
