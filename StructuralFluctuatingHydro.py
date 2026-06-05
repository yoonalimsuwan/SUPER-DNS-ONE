# =============================================================================
# STRUCTURAL LANGEVIN FOR FLUCTUATING HYDRODYNAMICS (CFD BRIDGE)
# =============================================================================
# Developer: Yoon A Limsuwan / MSPS NETWORK
# License: MIT
# Year: 2026
#
# A Fluctuating Hydrodynamics (FH) solver bridging the Structural Calculus
# Langevin framework to continuum CFD via the Landau–Lifshitz
# Navier–Stokes (LLNS) equations.
#
# Physical Model:
#   ∂ρ/∂t + ∇·(ρu) = 0                          (continuity)
#   ∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·(τ + S̃)    (momentum + stochastic stress)
#
# where S̃ is the Landau–Lifshitz stochastic stress tensor, here constructed
# via the Structural Itô / CSOC framework from Paper 3 & 4.
#
# Extensions from Structural Calculus:
#   1. BV Jump Measures at sharp interfaces (shocks, phase boundaries)
#   2. Multiplicative structural noise concentrated near interfaces
#   3. Itô drift correction for state-dependent noise amplitude
#   4. CSOC adaptive viscosity thermostat modulated by structural stress
#
# Grid convention:
#   Staggered 2-D Cartesian mesh (MAC / Marker-and-Cell layout).
#   Scalars (ρ, p) live at cell centres.
#   Velocities (u, v) live at face centres.
#
# Numerical scheme:
#   Time  : Fractional-step (projection) method, 1st-order explicit.
#   Space : 2nd-order central differences.
#   Noise : Consistent discrete Landau–Lifshitz stochastic stress tensor.
#
# Dependencies: torch >= 2.0
# =============================================================================

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility: finite difference stencils on 2D MAC grid
# ---------------------------------------------------------------------------

def div_u(ux: torch.Tensor, uy: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Divergence of velocity field on MAC grid.
    ux : (Nx+1, Ny)  x-face velocities
    uy : (Nx, Ny+1)  y-face velocities
    returns: (Nx, Ny) cell-centred divergence
    """
    dudx = (ux[1:, :] - ux[:-1, :]) / dx
    dvdy = (uy[:, 1:] - uy[:, :-1]) / dy
    return dudx + dvdy


def grad_p(p: torch.Tensor, dx: float, dy: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pressure gradient, returns (gx, gy) on MAC face centres.
    p  : (Nx, Ny)
    gx : (Nx+1, Ny)  — zero-padded at boundaries
    gy : (Nx, Ny+1)
    """
    gx = torch.zeros(p.shape[0] + 1, p.shape[1], device=p.device, dtype=p.dtype)
    gy = torch.zeros(p.shape[0], p.shape[1] + 1, device=p.device, dtype=p.dtype)
    gx[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx
    gy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy
    return gx, gy


def laplacian(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Cell-centred 5-point Laplacian with zero-Neumann boundary padding.
    f : (Nx, Ny)
    """
    fp = torch.nn.functional.pad(f.unsqueeze(0).unsqueeze(0),
                                  (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
    return (fp[2:, 1:-1] - 2 * f + fp[:-2, 1:-1]) / dx**2 + \
           (fp[1:-1, 2:] - 2 * f + fp[1:-1, :-2]) / dy**2


# ---------------------------------------------------------------------------
# Module 1 — Differentiable Interface Detector (CFD version)
# ---------------------------------------------------------------------------

class CFDInterfaceDetector(nn.Module):
    """
    Detects sharp-gradient regions in a scalar field (density, phase-indicator)
    on a 2D cell-centred grid.  Returns a soft mask ∈ [0, 1] that is
    differentiable w.r.t. the input field — required for correct Itô correction.

    Criterion: normalised gradient magnitude |∇φ| / (mean|∇φ| + ε).
    High values indicate shocks, phase boundaries, or flame fronts.
    """

    def __init__(self, sharpness: float = 4.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, phi: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """
        Args:
            phi : (Nx, Ny) scalar field (e.g. density ρ).
            dx, dy : grid spacings.
        Returns:
            mask : (Nx, Ny) interface score ∈ [0, 1], differentiable.
        """
        fp = torch.nn.functional.pad(phi.unsqueeze(0).unsqueeze(0),
                                      (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
        dphidx = (fp[2:, 1:-1] - fp[:-2, 1:-1]) / (2.0 * dx)
        dphidy = (fp[1:-1, 2:] - fp[1:-1, :-2]) / (2.0 * dy)
        grad_mag = torch.sqrt(dphidx**2 + dphidy**2 + 1e-12)

        # Normalise by mean magnitude so the threshold is field-adaptive
        norm_grad = grad_mag / (grad_mag.mean() + 1e-8)
        return torch.sigmoid(self.sharpness * (norm_grad - 1.0))


# ---------------------------------------------------------------------------
# Module 2 — SSC Filter (identical to MD version, operates on scalars)
# ---------------------------------------------------------------------------

class SemanticStateContraction(nn.Module):
    """
    EMA low-pass filter for structural stress σ.
    Identical to the MD version; operates on any scalar tensor.
    """

    def __init__(self, epsilon_fp: float = 0.0028):
        super().__init__()
        if not (0.0 < epsilon_fp < 1.0):
            raise ValueError(f"epsilon_fp must be in (0,1), got {epsilon_fp}")
        self.eps = epsilon_fp
        self.register_buffer('prev_sigma', torch.tensor(0.0))
        self.register_buffer('_initialized', torch.tensor(False))

    def reset(self) -> None:
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma
        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# ---------------------------------------------------------------------------
# Module 3 — CSOC Adaptive Viscosity (CFD thermostat analogue)
# ---------------------------------------------------------------------------

class CSOCAdaptiveViscosity(nn.Module):
    """
    CSOC-driven adaptive viscosity and thermal diffusivity for CFD.

    In Fluctuating Hydrodynamics, the noise amplitude scales with √(η k_B T).
    This module modulates the effective kinematic viscosity ν and thermal
    diffusivity α based on real-time structural stress (analogous to the
    temperature thermostat in the MD version).

    Physical interpretation:
        High structural stress  → near a shock / interface
        → increase ν (numerical stabilisation + physical SGS model)
        → increase noise amplitude (stronger fluctuations at interface)
    """

    def __init__(
        self,
        base_viscosity: float = 1e-3,       # m²/s (water ≈ 1e-6; adjust per fluid)
        base_diffusivity: float = 1e-5,
        sigma_target: float = 0.1,
        viscosity_boost: float = 5.0,
        epsilon_fp: float = 0.0028,
    ):
        super().__init__()
        self.base_nu = base_viscosity
        self.base_alpha = base_diffusivity
        self.sigma_target = sigma_target
        self.viscosity_boost = viscosity_boost
        self.ssc = SemanticStateContraction(epsilon_fp)

    def reset(self) -> None:
        self.ssc.reset()

    def forward(
        self,
        rho: torch.Tensor,
        rho_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rho      : (Nx, Ny) current density field.
            rho_prev : (Nx, Ny) density at previous time step.
        Returns:
            nu    : scalar adaptive kinematic viscosity.
            alpha : scalar adaptive thermal diffusivity.
            sigma : scalar SSC-filtered field stress.
        """
        raw_sigma = (rho - rho_prev).abs().mean()
        sigma = self.ssc(raw_sigma)

        dev = (sigma - self.sigma_target) / max(self.sigma_target, 1e-8)

        # Smooth boost via sigmoid — no hard clamp needed
        nu    = self.base_nu    * (1.0 + (self.viscosity_boost - 1.0) * torch.sigmoid(dev))
        alpha = self.base_alpha * (1.0 + (self.viscosity_boost - 1.0) * torch.sigmoid(dev))

        return nu, alpha, sigma


# ---------------------------------------------------------------------------
# Module 4 — Landau–Lifshitz Stochastic Stress Tensor
# ---------------------------------------------------------------------------

class LLStochasticStress(nn.Module):
    """
    Discrete Landau–Lifshitz stochastic stress tensor S̃ for 2D MAC grid.

    S̃_{ij} = √(2 η k_B T / (V Δt)) * (W_{ij} + W_{ji})

    where W_{ij} are i.i.d. standard normal random matrices, η is the
    dynamic viscosity, k_B T is thermal energy, V = dx*dy*dz is cell volume,
    and Δt is the time step.

    Structural extensions:
      • Multiplicative noise: amplitude amplified near interfaces via G(x).
      • Itô correction: ½ G ∇G applied as a deterministic drift correction.
    """

    def __init__(
        self,
        kb_T: float = 4.11e-21,     # k_B * T at 298 K in Joules
        dz: float = 1.0,            # out-of-plane depth for 2D (m)
        interface_amplification: float = 3.0,
    ):
        super().__init__()
        self.kb_T = kb_T
        self.dz = dz
        self.amp = interface_amplification

    def _noise_prefactor(
        self,
        nu: torch.Tensor,
        rho: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
    ) -> torch.Tensor:
        """
        σ_noise = √(2 ρ ν k_B T / (V Δt))
        V = dx * dy * dz
        Returns a (Nx, Ny) field of noise amplitudes.
        """
        V = dx * dy * self.dz
        eta = rho * nu                          # dynamic viscosity field (Nx, Ny)
        prefactor = torch.sqrt(2.0 * eta * self.kb_T / (V * dt) + 1e-30)
        return prefactor                        # (Nx, Ny)

    def _g_matrix(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp * mask, shape (Nx, Ny)."""
        return 1.0 + self.amp * interface_mask

    def _ito_correction(
        self,
        rho: torch.Tensor,
        interface_detector: CFDInterfaceDetector,
        dx: float,
        dy: float,
    ) -> torch.Tensor:
        """
        Itô drift correction: ½ G ∇G, cell-centred scalar field.
        Uses autograd through the differentiable interface_detector.
        """
        with torch.enable_grad():
            rho_g = rho.detach().requires_grad_(True)
            mask = interface_detector(rho_g, dx, dy)
            G = 1.0 + self.amp * mask
            G_sum = G.sum()
            grad_G = torch.autograd.grad(G_sum, rho_g,
                                          create_graph=False)[0]  # (Nx, Ny)
            if grad_G is None:
                return torch.zeros_like(rho)
            ito = 0.5 * G * grad_G          # (Nx, Ny) scalar correction
        return ito.detach()

    def forward(
        self,
        rho: torch.Tensor,
        nu: torch.Tensor,
        interface_mask: torch.Tensor,
        interface_detector: CFDInterfaceDetector,
        dx: float,
        dy: float,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute stochastic stress divergence terms for the x- and y-momentum
        equations, plus the Itô scalar correction.

        Returns:
            Sx   : (Nx, Ny) x-momentum stochastic forcing
            Sy   : (Nx, Ny) y-momentum stochastic forcing
            ito  : (Nx, Ny) Itô drift correction (density equation)
        """
        prefactor = self._noise_prefactor(nu, rho, dx, dy, dt)  # (Nx, Ny)
        G = self._g_matrix(interface_mask)                       # (Nx, Ny)
        amplitude = prefactor * G                                # (Nx, Ny)

        # Symmetric noise matrix W + W^T (2D: xx, xy, yy components)
        Wxx = torch.randn_like(rho)
        Wxy = torch.randn_like(rho)
        Wyy = torch.randn_like(rho)

        # Stress tensor components
        Sxx = amplitude * (Wxx + Wxx)          # 2 W_xx
        Sxy = amplitude * (Wxy + Wxy) * 0.5    # symmetric off-diagonal
        Syy = amplitude * (Wyy + Wyy)

        # Divergence of stochastic stress → momentum forcing (central diff)
        fp = torch.nn.functional.pad(
            torch.stack([Sxx, Sxy, Syy], dim=0).unsqueeze(0),
            (1, 1, 1, 1), mode='replicate'
        ).squeeze(0)

        dSxx_dx = (fp[0, 2:, 1:-1] - fp[0, :-2, 1:-1]) / (2.0 * dx)
        dSxy_dy = (fp[1, 1:-1, 2:] - fp[1, 1:-1, :-2]) / (2.0 * dy)
        dSxy_dx = (fp[1, 2:, 1:-1] - fp[1, :-2, 1:-1]) / (2.0 * dx)
        dSyy_dy = (fp[2, 1:-1, 2:] - fp[2, 1:-1, :-2]) / (2.0 * dy)

        Sx = dSxx_dx + dSxy_dy
        Sy = dSxy_dx + dSyy_dy

        ito = self._ito_correction(rho, interface_detector, dx, dy)

        return Sx, Sy, ito


# ---------------------------------------------------------------------------
# Core Solver — Fluctuating Hydrodynamics (2D MAC, projection method)
# ---------------------------------------------------------------------------

class StructuralFluctuatingHydro(nn.Module):
    """
    2D Fluctuating Hydrodynamics solver using the Structural Calculus framework.

    Solves the Landau–Lifshitz Navier–Stokes equations on a staggered MAC grid
    with CSOC-adaptive viscosity and Structural Itô stochastic stress.

    State variables (all cell-centred unless noted):
        rho  : (Nx, Ny)     mass density
        ux   : (Nx+1, Ny)   x-velocity (x-face)
        uy   : (Nx, Ny+1)   y-velocity (y-face)
        p    : (Nx, Ny)     pressure

    Usage::

        solver = StructuralFluctuatingHydro(Nx=64, Ny=64, Lx=1.0, Ly=1.0)
        rho, ux, uy, p = solver.initialize_taylor_green()

        for step in range(num_steps):
            rho, ux, uy, p, diagnostics = solver.step(rho, ux, uy, p)
            if step % 100 == 0:
                print(diagnostics)
    """

    def __init__(
        self,
        Nx: int = 64,
        Ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        dt: float = 1e-4,
        base_viscosity: float = 1e-3,
        base_diffusivity: float = 1e-5,
        kb_T: float = 4.11e-21,
        rho0: float = 1.0,
        enable_fluctuations: bool = True,
    ):
        """
        Args:
            Nx, Ny           : number of cells in x and y.
            Lx, Ly           : domain size (m).
            dt               : time step (s).
            base_viscosity   : kinematic viscosity ν₀ (m²/s).
            base_diffusivity : thermal diffusivity α₀ (m²/s).
            kb_T             : thermal energy k_B T (J).
            rho0             : reference density (kg/m³).
            enable_fluctuations: toggle stochastic stress on/off.
        """
        super().__init__()
        if Nx < 4 or Ny < 4:
            raise ValueError("Grid must be at least 4×4.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = Lx / Nx, Ly / Ny
        self.dt = dt
        self.rho0 = rho0
        self.enable_fluctuations = enable_fluctuations

        # Sub-modules
        self.interface_detector = CFDInterfaceDetector()
        self.csoc_viscosity     = CSOCAdaptiveViscosity(base_viscosity, base_diffusivity)
        self.ll_stress          = LLStochasticStress(kb_T)

        # State buffer for CSOC (previous density)
        self.register_buffer('_rho_prev', torch.ones(Nx, Ny) * rho0)
        self.register_buffer('_state_ready', torch.tensor(False))

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def initialize_uniform(
        self, device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quiescent uniform state."""
        rho = torch.ones(self.Nx, self.Ny,     device=device) * self.rho0
        ux  = torch.zeros(self.Nx + 1, self.Ny, device=device)
        uy  = torch.zeros(self.Nx, self.Ny + 1, device=device)
        p   = torch.zeros(self.Nx, self.Ny,     device=device)
        return rho, ux, uy, p

    def initialize_taylor_green(
        self, device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Taylor–Green vortex initial condition (incompressible limit)."""
        dx, dy = self.dx, self.dy
        Nx, Ny = self.Nx, self.Ny

        # x-face centres
        xf = (torch.arange(Nx + 1, device=device) * dx)
        yc = ((torch.arange(Ny,     device=device) + 0.5) * dy)
        ux = torch.sin(2 * math.pi * xf.unsqueeze(1)) * \
             torch.cos(2 * math.pi * yc.unsqueeze(0))

        # y-face centres
        xc = ((torch.arange(Nx,     device=device) + 0.5) * dx)
        yf = (torch.arange(Ny + 1, device=device) * dy)
        uy = -torch.cos(2 * math.pi * xc.unsqueeze(1)) * \
              torch.sin(2 * math.pi * yf.unsqueeze(0))

        rho = torch.ones(Nx, Ny, device=device) * self.rho0
        p   = torch.zeros(Nx, Ny, device=device)
        return rho, ux, uy, p

    # ------------------------------------------------------------------
    # Projection (pressure-velocity coupling)
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_pressure_poisson(
        div: torch.Tensor, dx: float, dy: float, dt: float, rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate pressure solve via Jacobi iteration.
        For production, replace with torch.linalg or a proper FFT-based solver.
        ∇²p = (ρ/Δt) ∇·u*
        """
        p = torch.zeros_like(div)
        rhs = (rho / dt) * div
        for _ in range(50):
            p_pad = torch.nn.functional.pad(
                p.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular'
            ).squeeze(0).squeeze(0)
            p = (
                (p_pad[2:, 1:-1] + p_pad[:-2, 1:-1]) / dx**2 +
                (p_pad[1:-1, 2:] + p_pad[1:-1, :-2]) / dy**2 -
                rhs
            ) / (2.0 / dx**2 + 2.0 / dy**2)
        return p

    # ------------------------------------------------------------------
    # Single time step
    # ------------------------------------------------------------------

    def step(
        self,
        rho: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        p:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Advance the FH state by one time step using:
          1. CSOC adaptive viscosity from structural stress.
          2. Viscous diffusion of velocity.
          3. Landau–Lifshitz stochastic stress (if enabled).
          4. Pressure projection for divergence-free velocity.
          5. Density advection.

        Args:
            rho : (Nx, Ny)     density
            ux  : (Nx+1, Ny)   x-face velocity
            uy  : (Nx, Ny+1)   y-face velocity
            p   : (Nx, Ny)     pressure
        Returns:
            rho_new, ux_new, uy_new, p_new : updated state
            diagnostics : dict with T_eff, sigma, nu, div_max
        """
        dx, dy, dt = self.dx, self.dy, self.dt
        device = rho.device

        # ── Initialise prev_rho ──────────────────────────────────────
        if not self._state_ready.item():
            self._rho_prev = rho.detach().clone()
            self._state_ready.fill_(True)

        # ── 1. CSOC adaptive viscosity ───────────────────────────────
        nu, alpha, sigma = self.csoc_viscosity(rho, self._rho_prev)
        self._rho_prev = rho.detach().clone()

        # ── 2. Interface detection (differentiable) ──────────────────
        interface_mask = self.interface_detector(rho, dx, dy)

        # ── 3. Cell-centred velocity (interpolated from faces) ───────
        uc_x = 0.5 * (ux[:-1, :] + ux[1:, :])   # (Nx, Ny)
        uc_y = 0.5 * (uy[:, :-1] + uy[:, 1:])   # (Nx, Ny)

        # ── 4. Viscous diffusion (explicit Laplacian) ────────────────
        lap_ux_c = laplacian(uc_x, dx, dy)
        lap_uy_c = laplacian(uc_y, dx, dy)

        visc_x = nu * lap_ux_c    # (Nx, Ny)
        visc_y = nu * lap_uy_c

        # ── 5. Landau–Lifshitz stochastic stress ─────────────────────
        if self.enable_fluctuations:
            Sx, Sy, ito_corr = self.ll_stress(
                rho, nu, interface_mask, self.interface_detector, dx, dy, dt
            )
            stoch_x = Sx / rho
            stoch_y = Sy / rho
        else:
            stoch_x = stoch_y = ito_corr = torch.zeros_like(rho)

        # ── 6. Intermediate velocity u* (no pressure) ────────────────
        # Pressure gradient (cell-centred, from previous p)
        gx_c = torch.zeros_like(uc_x)
        gy_c = torch.zeros_like(uc_y)
        gx_c[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx  # internal x
        gy_c[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy  # internal y

        ux_star_c = uc_x + dt * (visc_x - gx_c / rho + stoch_x)
        uy_star_c = uc_y + dt * (visc_y - gy_c / rho + stoch_y)

        # Reconstruct face velocities from cell-centred intermediates
        ux_star = torch.zeros(self.Nx + 1, self.Ny, device=device, dtype=rho.dtype)
        uy_star = torch.zeros(self.Nx, self.Ny + 1, device=device, dtype=rho.dtype)
        ux_star[1:-1, :] = 0.5 * (ux_star_c[:-1, :] + ux_star_c[1:, :])
        uy_star[:, 1:-1] = 0.5 * (uy_star_c[:, :-1] + uy_star_c[:, 1:])

        # ── 7. Pressure projection ───────────────────────────────────
        div_star = div_u(ux_star, uy_star, dx, dy)
        p_new = self._solve_pressure_poisson(div_star, dx, dy, dt, rho)

        gx_new, gy_new = grad_p(p_new, dx, dy)
        ux_new = ux_star - (dt / rho.mean()) * gx_new
        uy_new = uy_star - (dt / rho.mean()) * gy_new

        # ── 8. Density advection (upwind, 1st order) ─────────────────
        uc_x_adv = 0.5 * (ux_new[:-1, :] + ux_new[1:, :])
        uc_y_adv = 0.5 * (uy_new[:, :-1] + uy_new[:, 1:])

        rho_pad = torch.nn.functional.pad(
            rho.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate'
        ).squeeze(0).squeeze(0)

        drho_dx = torch.where(
            uc_x_adv >= 0,
            (rho - rho_pad[:-2, 1:-1]) / dx,
            (rho_pad[2:, 1:-1] - rho) / dx,
        )
        drho_dy = torch.where(
            uc_y_adv >= 0,
            (rho - rho_pad[1:-1, :-2]) / dy,
            (rho_pad[1:-1, 2:] - rho) / dy,
        )

        rho_new = rho - dt * (uc_x_adv * drho_dx + uc_y_adv * drho_dy)

        # Apply Itô correction to density (structural noise in continuity eq.)
        if self.enable_fluctuations:
            rho_new = rho_new + dt * ito_corr

        rho_new = rho_new.clamp(min=1e-6)   # positivity constraint

        # ── Diagnostics ──────────────────────────────────────────────
        diagnostics = {
            'sigma':    sigma.item(),
            'nu':       nu.item(),
            'alpha':    alpha.item(),
            'div_max':  div_u(ux_new, uy_new, dx, dy).abs().max().item(),
            'rho_mean': rho_new.mean().item(),
        }

        return rho_new, ux_new, uy_new, p_new, diagnostics

    def reset(self) -> None:
        """Reset solver state between independent runs."""
        self._rho_prev.fill_(self.rho0)
        self._state_ready.fill_(False)
        self.csoc_viscosity.reset()


# =============================================================================
# Quick self-test  (python structuralfluctuatinghydro.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Running StructuralFluctuatingHydro self-test ...")
    torch.manual_seed(0)

    solver = StructuralFluctuatingHydro(
        Nx=32, Ny=32, Lx=1.0, Ly=1.0, dt=1e-4,
        base_viscosity=1e-3, enable_fluctuations=True,
    )

    rho, ux, uy, p = solver.initialize_taylor_green()

    print(f"  {'Step':>5}  {'sigma':>8}  {'nu':>10}  {'div_max':>10}  {'rho_mean':>10}")
    for step in range(5):
        rho, ux, uy, p, diag = solver.step(rho, ux, uy, p)
        print(f"  {step:>5}  {diag['sigma']:>8.5f}  {diag['nu']:>10.2e}"
              f"  {diag['div_max']:>10.2e}  {diag['rho_mean']:>10.6f}")

    print("Self-test passed.")
    sys.exit(0)
