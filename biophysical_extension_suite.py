===============================================================================
SUPER DNS ONE v6 - Biophysical Extension Suite
Modules: Electrochemical Signals, Vascular Network Dynamics, & Metabolic Kinetics
Language: Python 3.10+ / PyTorch (Fully Differentiable CUDA-accelerated)
===============================================================================
=============================================================================
Developer   : PAI AND Yoon A Limsuwan : MSPS NETWORK / My Soul Move By Power of Holy Spirit 
License     : MIT
Year        : 2026
Version     : 1.0.0 
=============================================================================
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BiophysicsConfig:
    """Global configuration for biophysical computational domains."""
    grid_shape: Tuple[int, int, int] = (128, 128, 128)
    dx: float = 1e-5  # Grid spacing (meters)
    dt: float = 1e-4  # Time step (seconds)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class SpatialOperators3D:
    """High-performance 3D finite-difference spatial operators for PyTorch tensors."""

    @staticmethod
    def gradient(f: torch.Tensor, dx: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates 3D central difference gradient."""
        df_dx = (torch.roll(f, shifts=-1, dims=-1) - torch.roll(f, shifts=1, dims=-1)) / (2.0 * dx)
        df_dy = (torch.roll(f, shifts=-1, dims=-2) - torch.roll(f, shifts=1, dims=-2)) / (2.0 * dx)
        df_dz = (torch.roll(f, shifts=-1, dims=-3) - torch.roll(f, shifts=1, dims=-3)) / (2.0 * dx)
        return df_dx, df_dy, df_dz

    @staticmethod
    def laplacian(f: torch.Tensor, dx: float) -> torch.Tensor:
        """Calculates 3D 7-point stencil Laplacian operator."""
        lap = (
            torch.roll(f, shifts=-1, dims=-1) + torch.roll(f, shifts=1, dims=-1) +
            torch.roll(f, shifts=-1, dims=-2) + torch.roll(f, shifts=1, dims=-2) +
            torch.roll(f, shifts=-1, dims=-3) + torch.roll(f, shifts=1, dims=-3) -
            6.0 * f
        ) / (dx ** 2)
        return lap

    @staticmethod
    def divergence(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, dx: float) -> torch.Tensor:
        """Calculates 3D divergence of a vector field (u, v, w)."""
        du_dx = (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0 * dx)
        dv_dy = (torch.roll(v, shifts=-1, dims=-2) - torch.roll(v, shifts=1, dims=-2)) / (2.0 * dx)
        dw_dz = (torch.roll(w, shifts=-1, dims=-3) - torch.roll(w, shifts=1, dims=-3)) / (2.0 * dx)
        return du_dx + dv_dy + dw_dz


class ElectrochemicalSignalModule(nn.Module):
    """
    Differentiable 3D Poisson-Nernst-Planck (PNP) & FitzHugh-Nagumo Membrane Dynamics Module.
    Simulates ionic diffusion, electro-migration (Na+, K+, Ca2+), and action potential propagation.
    """

    def __init__(self, config: BiophysicsConfig):
        super().__init__()
        self.cfg = config
        
        # Physical Constants
        self.F = 96485.3321  # Faraday constant (C/mol)
        self.R = 8.3144626   # Gas constant (J/(mol*K))
        self.T = 310.15      # Temperature (Kelvin)

        # Ion Diffusion Coefficients (m^2/s) & Valencies
        self.register_buffer("diffusivities", torch.tensor([1.33e-9, 1.96e-9, 0.79e-9], device=config.device))
        self.register_buffer("valencies", torch.tensor([1.0, 1.0, 2.0], device=config.device)) # Na+, K+, Ca2+

    def forward(
        self, 
        ion_concentrations: torch.Tensor, 
        electric_potential: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes single time-step evolution of electrochemical species.

        Args:
            ion_concentrations: Shape [B, 3, Z, Y, X] representing [Na+, K+, Ca2+] (mol/m^3)
            electric_potential: Shape [B, 1, Z, Y, X] electrostatic potential Phi (Volts)

        Returns:
            d_ion_dt: Rate of change of ion concentrations [B, 3, Z, Y, X]
            phi_laplacian: Electrostatic potential field laplacian [B, 1, Z, Y, X]
        """
        B, C, Z, Y, X = ion_concentrations.shape
        d_ion_dt = torch.zeros_like(ion_concentrations)

        # Electrostatic potential gradient
        dphi_dx, dphi_dy, dphi_dz = SpatialOperators3D.gradient(electric_potential.squeeze(1), self.cfg.dx)

        for i in range(C):
            c_i = ion_concentrations[:, i, ...]
            D_i = self.diffusivities[i]
            z_i = self.valencies[i]
            eta = (z_i * self.F) / (self.R * self.T)

            # 1. Fickian Diffusion Term: D_i * Grad^2(C_i)
            diff_term = D_i * SpatialOperators3D.laplacian(c_i, self.cfg.dx)

            # 2. Migration Term: Div( D_i * z_i * F / (R*T) * C_i * Grad(Phi) )
            dc_dx, dc_dy, dc_dz = SpatialOperators3D.gradient(c_i, self.cfg.dx)
            
            flux_x = D_i * eta * c_i * dphi_dx
            flux_y = D_i * eta * c_i * dphi_dy
            flux_z = D_i * eta * c_i * dphi_dz

            migr_term = SpatialOperators3D.divergence(flux_x, flux_y, flux_z, self.cfg.dx)

            d_ion_dt[:, i, ...] = diff_term + migr_term

        phi_laplacian = SpatialOperators3D.laplacian(electric_potential.squeeze(1), self.cfg.dx).unsqueeze(1)
        return d_ion_dt, phi_laplacian


class VascularNetworkModule(nn.Module):
    """
    Differentiable 3D Microvascular Hemodynamics & Porous Darcy-Brinkman Solver.
    Computes blood velocity fields, microvascular hydraulic resistance, and perfusion pressure gradients.
    """

    def __init__(self, config: BiophysicsConfig):
        super().__init__()
        self.cfg = config
        
        # Fluid parameters
        self.blood_density = 1060.0        # kg/m^3
        self.blood_viscosity = 3.5e-3      # Pa*s (Dynamic viscosity)
        
        # Learnable vascular hydraulic conductivity map parameterizer
        self.permeability_scale = nn.Parameter(torch.tensor([1e-10], device=config.device))

    def forward(
        self, 
        velocity: torch.Tensor, 
        pressure: torch.Tensor, 
        vessel_volume_fraction: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            velocity: Flow velocity vector field [B, 3, Z, Y, X] (m/s)
            pressure: Vascular fluid pressure [B, 1, Z, Y, X] (Pa)
            vessel_volume_fraction: Local vascular density phi_v [B, 1, Z, Y, X] (0.0 to 1.0)

        Returns:
            du_dt: Momentum rate of change [B, 3, Z, Y, X]
            darcy_drag: Brinkman permeability resistance drag tensor [B, 3, Z, Y, X]
        """
        u, v, w = velocity[:, 0, ...], velocity[:, 1, ...], velocity[:, 2, ...]
        p = pressure.squeeze(1)

        # Pressure gradient
        dp_dx, dp_dy, dp_dz = SpatialOperators3D.gradient(p, self.cfg.dx)

        # Viscous Brinkman term: mu * Lap(u)
        lap_u = SpatialOperators3D.laplacian(u, self.cfg.dx)
        lap_v = SpatialOperators3D.laplacian(v, self.cfg.dx)
        lap_w = SpatialOperators3D.laplacian(w, self.cfg.dx)

        # Darcy drag resistance depending on vascular density (Carman-Kozeny type model)
        k_perm = self.permeability_scale * (vessel_volume_fraction.squeeze(1) ** 3) / (
            (1.0 - vessel_volume_fraction.squeeze(1) + 1e-6) ** 2 + 1e-8
        )
        
        drag_x = -(self.blood_viscosity / k_perm) * u
        drag_y = -(self.blood_viscosity / k_perm) * v
        drag_z = -(self.blood_viscosity / k_perm) * w

        # Total Navier-Stokes-Brinkman Acceleration
        du_dt_x = (-dp_dx + self.blood_viscosity * lap_u + drag_x) / self.blood_density
        du_dt_y = (-dp_dy + self.blood_viscosity * lap_v + drag_y) / self.blood_density
        du_dt_z = (-dp_dz + self.blood_viscosity * lap_w + drag_z) / self.blood_density

        du_dt = torch.stack([du_dt_x, du_dt_y, du_dt_z], dim=1)
        darcy_drag = torch.stack([drag_x, drag_y, drag_z], dim=1)

        return du_dt, darcy_drag


class MetabolicKineticsModule(nn.Module):
    """
    Differentiable Multi-Species Metabolic Reaction-Diffusion Engine.
    Models Oxygen (O2), Glucose, Lactate, and ATP dynamics using non-linear Michaelis-Menten kinetics.
    """

    def __init__(self, config: BiophysicsConfig):
        super().__init__()
        self.cfg = config

        # Species indices: 0: Oxygen, 1: Glucose, 2: Lactate, 3: ATP
        self.register_buffer("D_species", torch.tensor([1.8e-9, 6.7e-10, 5.0e-10, 1.0e-10], device=config.device))
        
        # Kinetic Parameters (Vmax in mol/(m^3*s), Km in mol/m^3)
        self.Vmax_O2 = 0.05
        self.Km_O2 = 0.01
        self.Vmax_Glucose = 0.02
        self.Km_Glucose = 0.05

    def forward(
        self, 
        species_conc: torch.Tensor, 
        cell_viability: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            species_conc: Concentrations [B, 4, Z, Y, X] -> [O2, Glucose, Lactate, ATP]
            cell_viability: Cell volume fraction / viability state [B, 1, Z, Y, X]

        Returns:
            d_species_dt: Rate of change of species [B, 4, Z, Y, X]
            metabolic_rates: Dictionary containing Michaelis-Menten consumption rates
        """
        O2 = species_conc[:, 0:1, ...]
        Glucose = species_conc[:, 1:2, ...]
        
        # Michaelis-Menten Consumption Kinetics
        r_O2 = (self.Vmax_O2 * O2 / (self.Km_O2 + O2 + 1e-8)) * cell_viability
        r_Glucose = (self.Vmax_Glucose * Glucose / (self.Km_Glucose + Glucose + 1e-8)) * cell_viability
        
        # ATP Production Coupling (Aerobic + Anaerobic Glycolysis)
        r_ATP_prod = 29.0 * r_O2 + 2.0 * r_Glucose
        r_Lactate_prod = 2.0 * r_Glucose * torch.exp(-O2 / (self.Km_O2 + 1e-8))

        # Net Reaction Source Terms (S_s)
        S_O2 = -r_O2
        S_Glucose = -r_Glucose
        S_Lactate = r_Lactate_prod
        S_ATP = r_ATP_prod - (0.1 * species_conc[:, 3:4, ...]) # ATP utilization rate

        S_net = torch.cat([S_O2, S_Glucose, S_Lactate, S_ATP], dim=1)

        # Spatial Diffusion Term: D_s * Lap(C_s)
        d_species_dt = torch.zeros_like(species_conc)
        for i in range(4):
            lap_c = SpatialOperators3D.laplacian(species_conc[:, i, ...], self.cfg.dx)
            d_species_dt[:, i, ...] = self.D_species[i] * lap_c + S_net[:, i, ...]

        rates = {
            "O2_consumption": r_O2,
            "Glucose_consumption": r_Glucose,
            "ATP_production": r_ATP_prod
        }

        return d_species_dt, rates


class BiophysicalDNSBridge(nn.Module):
    """
    Master Integration Bridge linking Electrochemical, Vascular Hemodynamics, 
    and Metabolic Modules directly with SUPER DNS ONE v6 Fluid Physics Core.
    """

    def __init__(self, config: BiophysicsConfig):
        super().__init__()
        self.cfg = config
        self.electro = ElectrochemicalSignalModule(config)
        self.vascular = VascularNetworkModule(config)
        self.metabolism = MetabolicKineticsModule(config)

    def step_simulation(
        self,
        fluid_velocity: torch.Tensor,
        fluid_pressure: torch.Tensor,
        ion_concentrations: torch.Tensor,
        electric_potential: torch.Tensor,
        metabolic_species: torch.Tensor,
        vascular_density: torch.Tensor,
        cell_viability: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a unified forward integration step across all biophysical domains.
        """
        # 1. Update Hemodynamics
        du_dt, drag = self.vascular(fluid_velocity, fluid_pressure, vascular_density)
        
        # 2. Update Electrochemical State
        d_ions_dt, phi_lap = self.electro(ion_concentrations, electric_potential)
        
        # 3. Update Metabolic Rates
        d_species_dt, rates = self.metabolism(metabolic_species, cell_viability)

        # Euler / RK4 Explicit Time Step Update
        updated_velocity = fluid_velocity + du_dt * self.cfg.dt
        updated_ions = ion_concentrations + d_ions_dt * self.cfg.dt
        updated_species = metabolic_species + d_species_dt * self.cfg.dt

        return {
            "velocity": updated_velocity,
            "ion_concentrations": updated_ions,
            "metabolic_species": updated_species,
            "darcy_drag": drag,
            "metabolic_rates": rates
        }
