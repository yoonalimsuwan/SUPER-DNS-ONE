# =============================================================================
# Near-to-Far Field Transformation:Radar Cross Section (RCS)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

import math
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn


class PhysicalConstants:
    """SI Physical Constants for Electromagnetic Computations."""
    C0: float = 299792458.0              # Speed of light in vacuum (m/s)
    MU0: float = 4.0 * math.pi * 1e-7     # Vacuum permeability (H/m)
    EPS0: float = 1.0 / (C0**2 * MU0)    # Vacuum permittivity (F/m)
    ETA0: float = math.sqrt(MU0 / EPS0)  # Free-space wave impedance (~376.73 Ohm)


class DFTAccumulator3D(nn.Module):
    """
    Real-time Discrete Fourier Transform (DFT) Accumulator for Time-Domain Solvers (e.g., FDTD).
    Extracts complex phasors at target frequency on-the-fly without storing entire time history.
    """
    def __init__(self, target_freq_hz: float, dt: float, field_shape: Tuple[int, int, int], device: torch.device):
        super().__init__()
        self.target_freq = target_freq_hz
        self.dt = dt
        self.omega = 2.0 * math.pi * target_freq_hz
        self.device = device
        
        # Accumulators for E and H field components
        self.real_parts = {k: torch.zeros(field_shape, device=device) for k in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}
        self.imag_parts = {k: torch.zeros(field_shape, device=device) for k in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}
        self.step_count = 0

    def update(self, fields_t: Dict[str, torch.Tensor], current_time: float) -> None:
        """Accumulates continuous time-domain fields into running Fourier integral."""
        phase = self.omega * current_time
        cos_val = math.cos(phase) * self.dt
        sin_val = math.sin(phase) * self.dt

        for key in fields_t:
            if key in self.real_parts:
                self.real_parts[key] = self.real_parts[key] + fields_t[key] * cos_val
                self.imag_parts[key] = self.imag_parts[key] - fields_t[key] * sin_val  # e^(-j * omega * t)
        self.step_count += 1

    def get_phasors(self) -> Dict[str, torch.Tensor]:
        """Returns normalized complex phasors."""
        phasors = {}
        for key in self.real_parts:
            phasors[key] = torch.complex(self.real_parts[key], self.imag_parts[key])
        return phasors


class StructuralNTFTRCS3D(nn.Module):
    """
    Production-Grade 3D Near-to-Far-Field Transformation (NTFT) & RCS Calculator.
    
    Computes far-field radiation patterns (E_theta, E_phi) and Radar Cross Section (m^2, dBsm)
    from 3D near-field phasors on a 6-face Huygens closed bounding box.
    """
    def __init__(
        self,
        grid_spacing: Tuple[float, float, float],
        huygens_box_indices: Tuple[int, int, int, int, int, int],
        freq_hz: float,
        e_inc_amp: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64
    ):
        """
        Args:
            grid_spacing: Spatial grid steps (dx, dy, dz) in meters.
            huygens_box_indices: (i_min, i_max, j_min, j_max, k_min, k_max) defining bounding box.
            freq_hz: Frequency of interest in Hertz.
            e_inc_amp: Amplitude of incident electric field |E_inc| (V/m).
            device: Computing device (torch.device('cuda') or torch.device('cpu')).
            dtype: Complex precision type (torch.complex64 or torch.complex128).
        """
        super().__init__()
        self.dx, self.dy, self.dz = grid_spacing
        self.i_min, self.i_max, self.j_min, self.j_max, self.k_min, self.k_max = huygens_box_indices
        self.freq_hz = freq_hz
        self.e_inc_amp = e_inc_amp
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        # Wave propagation parameters
        self.omega = 2.0 * math.pi * self.freq_hz
        self.k0 = self.omega / PhysicalConstants.C0
        self.eta0 = PhysicalConstants.ETA0

        # Pre-build coordinate grids for Huygens surface faces
        self._build_huygens_surface_mesh()

    def _build_huygens_surface_mesh(self) -> None:
        """Constructs tensor coordinates for all 6 faces of the Huygens box."""
        x_grid = torch.arange(self.i_min, self.i_max + 1, device=self.device, dtype=self.real_dtype) * self.dx
        y_grid = torch.arange(self.j_min, self.j_max + 1, device=self.device, dtype=self.real_dtype) * self.dy
        z_grid = torch.arange(self.k_min, self.k_max + 1, device=self.device, dtype=self.real_dtype) * self.dz

        # Face +X / -X
        Y_x, Z_x = torch.meshgrid(y_grid, z_grid, indexing='ij')
        X_px = torch.full_like(Y_x, x_grid[-1])
        X_nx = torch.full_like(Y_x, x_grid[0])

        # Face +Y / -Y
        X_y, Z_y = torch.meshgrid(x_grid, z_grid, indexing='ij')
        Y_py = torch.full_like(X_y, y_grid[-1])
        Y_ny = torch.full_like(X_y, y_grid[0])

        # Face +Z / -Z
        X_z, Y_z = torch.meshgrid(x_grid, y_grid, indexing='ij')
        Z_pz = torch.full_like(X_z, z_grid[-1])
        Z_nz = torch.full_like(X_z, z_grid[0])

        # Register buffers for coordinate vectors
        self.register_buffer('x_px', X_px.reshape(-1)); self.register_buffer('y_px', Y_x.reshape(-1)); self.register_buffer('z_px', Z_x.reshape(-1))
        self.register_buffer('x_nx', X_nx.reshape(-1)); self.register_buffer('y_nx', Y_x.reshape(-1)); self.register_buffer('z_nx', Z_x.reshape(-1))
        self.register_buffer('x_py', X_y.reshape(-1)); self.register_buffer('y_py', Y_py.reshape(-1)); self.register_buffer('z_py', Z_y.reshape(-1))
        self.register_buffer('x_ny', X_y.reshape(-1)); self.register_buffer('y_ny', Y_ny.reshape(-1)); self.register_buffer('z_ny', Z_y.reshape(-1))
        self.register_buffer('x_pz', X_z.reshape(-1)); self.register_buffer('y_pz', Y_z.reshape(-1)); self.register_buffer('z_pz', Z_pz.reshape(-1))
        self.register_buffer('x_nz', X_z.reshape(-1)); self.register_buffer('y_nz', Y_z.reshape(-1)); self.register_buffer('z_nz', Z_nz.reshape(-1))

        # Surface differential areas
        self.ds_yz = self.dy * self.dz
        self.ds_xz = self.dx * self.dz
        self.ds_xy = self.dx * self.dy

    def _extract_equivalent_currents(
        self,
        E_phasors: Dict[str, torch.Tensor],
        H_phasors: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculates J_s = n x H and M_s = -n x E on each face."""
        i0, i1 = self.i_min, self.i_max
        j0, j1 = self.j_min, self.j_max
        k0, k1 = self.k_min, self.k_max

        currents = {}

        # 1. +X Face (n = +x) -> J_s = (0, -Hz, Hy), M_s = (0, Ez, -Ey)
        Ey, Ez = E_phasors['Ey'][i1, j0:j1+1, k0:k1+1].reshape(-1), E_phasors['Ez'][i1, j0:j1+1, k0:k1+1].reshape(-1)
        Hy, Hz = H_phasors['Hy'][i1, j0:j1+1, k0:k1+1].reshape(-1), H_phasors['Hz'][i1, j0:j1+1, k0:k1+1].reshape(-1)
        currents['+X'] = (
            torch.stack([torch.zeros_like(Hz), -Hz, Hy], dim=0),
            torch.stack([torch.zeros_like(Ez), Ez, -Ey], dim=0)
        )

        # 2. -X Face (n = -x) -> J_s = (0, Hz, -Hy), M_s = (0, -Ez, Ey)
        Ey, Ez = E_phasors['Ey'][i0, j0:j1+1, k0:k1+1].reshape(-1), E_phasors['Ez'][i0, j0:j1+1, k0:k1+1].reshape(-1)
        Hy, Hz = H_phasors['Hy'][i0, j0:j1+1, k0:k1+1].reshape(-1), H_phasors['Hz'][i0, j0:j1+1, k0:k1+1].reshape(-1)
        currents['-X'] = (
            torch.stack([torch.zeros_like(Hz), Hz, -Hy], dim=0),
            torch.stack([torch.zeros_like(Ez), -Ez, Ey], dim=0)
        )

        # 3. +Y Face (n = +y) -> J_s = (Hz, 0, -Hx), M_s = (-Ez, 0, Ex)
        Ex, Ez = E_phasors['Ex'][i0:i1+1, j1, k0:k1+1].reshape(-1), E_phasors['Ez'][i0:i1+1, j1, k0:k1+1].reshape(-1)
        Hx, Hz = H_phasors['Hx'][i0:i1+1, j1, k0:k1+1].reshape(-1), H_phasors['Hz'][i0:i1+1, j1, k0:k1+1].reshape(-1)
        currents['+Y'] = (
            torch.stack([Hz, torch.zeros_like(Hz), -Hx], dim=0),
            torch.stack([-Ez, torch.zeros_like(Ez), Ex], dim=0)
        )

        # 4. -Y Face (n = -y) -> J_s = (-Hz, 0, Hx), M_s = (Ez, 0, -Ex)
        Ex, Ez = E_phasors['Ex'][i0:i1+1, j0, k0:k1+1].reshape(-1), E_phasors['Ez'][i0:i1+1, j0, k0:k1+1].reshape(-1)
        Hx, Hz = H_phasors['Hx'][i0:i1+1, j0, k0:k1+1].reshape(-1), H_phasors['Hz'][i0:i1+1, j0, k0:k1+1].reshape(-1)
        currents['-Y'] = (
            torch.stack([-Hz, torch.zeros_like(Hz), Hx], dim=0),
            torch.stack([Ez, torch.zeros_like(Ez), -Ex], dim=0)
        )

        # 5. +Z Face (n = +z) -> J_s = (-Hy, Hx, 0), M_s = (Ey, -Ex, 0)
        Ex, Ey = E_phasors['Ex'][i0:i1+1, j0:j1+1, k1].reshape(-1), E_phasors['Ey'][i0:i1+1, j0:j1+1, k1].reshape(-1)
        Hx, Hy = H_phasors['Hx'][i0:i1+1, j0:j1+1, k1].reshape(-1), H_phasors['Hy'][i0:i1+1, j0:j1+1, k1].reshape(-1)
        currents['+Z'] = (
            torch.stack([-Hy, Hx, torch.zeros_like(Hx)], dim=0),
            torch.stack([Ey, -Ex, torch.zeros_like(Ex)], dim=0)
        )

        # 6. -Z Face (n = -z) -> J_s = (Hy, -Hx, 0), M_s = (-Ey, Ex, 0)
        Ex, Ey = E_phasors['Ex'][i0:i1+1, j0:j1+1, k0].reshape(-1), E_phasors['Ey'][i0:i1+1, j0:j1+1, k0].reshape(-1)
        Hx, Hy = H_phasors['Hx'][i0:i1+1, j0:j1+1, k0].reshape(-1), H_phasors['Hy'][i0:i1+1, j0:j1+1, k0].reshape(-1)
        currents['-Z'] = (
            torch.stack([Hy, -Hx, torch.zeros_like(Hx)], dim=0),
            torch.stack([-Ey, Ex, torch.zeros_like(Ex)], dim=0)
        )

        return currents

    def forward(
        self,
        E_phasors: Dict[str, torch.Tensor],
        H_phasors: Dict[str, torch.Tensor],
        theta: torch.Tensor,
        phi: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates Far-Field RCS over specified theta and phi observation grid.

        Args:
            E_phasors: Near-field complex E phasors {'Ex', 'Ey', 'Ez'} of shape (Nx, Ny, Nz)
            H_phasors: Near-field complex H phasors {'Hx', 'Hy', 'Hz'} of shape (Nx, Ny, Nz)
            theta: Polar angles in radians [0, pi] (Tensor of shape [N] or [N, M])
            phi: Azimuth angles in radians [0, 2*pi] (Tensor matching theta shape)

        Returns:
            Dict containing:
                - 'rcs_m2': Radar Cross Section in square meters (m^2)
                - 'rcs_dbsm': Radar Cross Section in decibel-square-meters (dBsm)
                - 'E_theta': Far-field E_theta complex potential
                - 'E_phi': Far-field E_phi complex potential
        """
        assert theta.shape == phi.shape, "theta and phi tensors must have identical shapes"
        out_shape = theta.shape
        th_vec = theta.reshape(-1)
        ph_vec = phi.reshape(-1)

        # Spherical unit vectors in Cartesian components
        sin_th, cos_th = torch.sin(th_vec), torch.cos(th_vec)
        sin_ph, cos_ph = torch.sin(ph_vec), torch.cos(ph_vec)

        r_hat_x = sin_th * cos_ph
        r_hat_y = sin_th * sin_ph
        r_hat_z = cos_th

        # Extract current sources
        currents = self._extract_equivalent_currents(E_phasors, H_phasors)

        N_x = torch.zeros_like(th_vec, dtype=self.dtype)
        N_y = torch.zeros_like(th_vec, dtype=self.dtype)
        N_z = torch.zeros_like(th_vec, dtype=self.dtype)
        L_x = torch.zeros_like(th_vec, dtype=self.dtype)
        L_y = torch.zeros_like(th_vec, dtype=self.dtype)
        L_z = torch.zeros_like(th_vec, dtype=self.dtype)

        faces_meta = [
            ('+X', self.x_px, self.y_px, self.z_px, self.ds_yz),
            ('-X', self.x_nx, self.y_nx, self.z_nx, self.ds_yz),
            ('+Y', self.x_py, self.y_py, self.z_py, self.ds_xz),
            ('-Y', self.x_ny, self.y_ny, self.z_ny, self.ds_xz),
            ('+Z', self.x_pz, self.y_pz, self.z_pz, self.ds_xy),
            ('-Z', self.x_nz, self.y_nz, self.z_nz, self.ds_xy),
        ]

        # Surface integration via vectorized matrix multiplication
        for name, xs, ys, zs, dS in faces_meta:
            J_s, M_s = currents[name]  # Shapes: (3, N_pts)

            # Phase factor matrix: (N_angles, N_pts)
            phase = self.k0 * (
                torch.outer(r_hat_x, xs) +
                torch.outer(r_hat_y, ys) +
                torch.outer(r_hat_z, zs)
            )
            exp_phase = torch.exp(1j * phase.to(self.dtype))

            # Numerical Surface Integration
            N_x = N_x + torch.matmul(exp_phase, J_s[0]) * dS
            N_y = N_y + torch.matmul(exp_phase, J_s[1]) * dS
            N_z = N_z + torch.matmul(exp_phase, J_s[2]) * dS

            L_x = L_x + torch.matmul(exp_phase, M_s[0]) * dS
            L_y = L_y + torch.matmul(exp_phase, M_s[1]) * dS
            L_z = L_z + torch.matmul(exp_phase, M_s[2]) * dS

        # Spherical Transformation
        N_theta = N_x * cos_th * cos_ph + N_y * cos_th * sin_ph - N_z * sin_th
        N_phi   = -N_x * sin_ph + N_y * cos_ph

        L_theta = L_x * cos_th * cos_ph + L_y * cos_th * sin_ph - L_z * sin_th
        L_phi   = -L_x * sin_ph + L_y * cos_ph

        # Far-field Electric Fields
        coeff = 1j * self.k0 / (4.0 * math.pi)
        E_theta = -coeff * (L_phi + self.eta0 * N_theta)
        E_phi   =  coeff * (L_theta - self.eta0 * N_phi)

        # Bistatic RCS Calculation: sigma = (k0^2 / 4pi |E_inc|^2) * (|L_phi + eta*N_th|^2 + |L_th - eta*N_ph|^2)
        rcs_m2 = (self.k0**2 / (4.0 * math.pi * (self.e_inc_amp**2))) * (
            torch.abs(L_phi + self.eta0 * N_theta)**2 +
            torch.abs(L_theta - self.eta0 * N_phi)**2
        )

        # Convert to dBsm with zero-guard
        rcs_dbsm = 10.0 * torch.log10(torch.clamp(rcs_m2, min=1e-18))

        return {
            'rcs_m2': rcs_m2.reshape(out_shape),
            'rcs_dbsm': rcs_dbsm.reshape(out_shape),
            'E_theta': E_theta.reshape(out_shape),
            'E_phi': E_phi.reshape(out_shape)
        }
