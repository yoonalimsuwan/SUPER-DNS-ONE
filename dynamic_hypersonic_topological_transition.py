# =============================================================================
# HYPERSONIC PLASMA THERMODYNAMIC COUPLING MODULE (SESI)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Description: A fully differentiable, production-optimized module for coupling
# Activation Energy (\Delta E) and Fluctuation Variance (\sigma^2) to local 
# hypersonic shockwave thermodynamics. Utilizes a Straight-Through Estimator 
# (STE) to enable end-to-end backpropagation through discrete topological jumps.
# =============================================================================

import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["DifferentiableHypersonicTopologicalTransition"]

class DifferentiableHypersonicTopologicalTransition(nn.Module):
    """
    A fully differentiable neural module simulating spatially-varying topological 
    transitions (Nucleation, Merging, Branching) induced by extreme hypersonic 
    plasma sheath thermodynamics.
    """
    def __init__(
        self, 
        c1: float = 1.0, 
        k_b: float = 1.380649e-23,  # Boltzmann constant (J/K)
        ref_density: float = 1.225, # Standard sea-level air density (kg/m^3)
        ionization_energy_base: float = 1.0,
        ste_tau: float = 0.1        # Temperature parameter for Gumbel-Softmax/STE relaxation
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.k_b = k_b
        self.ref_density = ref_density
        self.ionization_base = ionization_energy_base
        self.ste_tau = ste_tau

    @torch.jit.export
    def compute_dynamic_parameters(
        self, 
        temperature: torch.Tensor, 
        density: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes 3D tensor fields for Activation Energy (\Delta E) and 
        Fluctuation Variance (\sigma^2) based on local thermodynamics.
        Highly optimized for vectorized tensor execution.
        """
        # 1. Fluctuation Variance (\sigma^2): Proportional to local thermal energy (k_B * T)
        # Clamped to prevent zero-division or zero-variance in free-stream regions.
        sigma_sq = torch.clamp(self.k_b * temperature, min=1e-12)

        # 2. Activation Energy (\Delta E): Logarithmic scaling with compression ratio.
        # Uses torch.log1p(x) for mathematically stable computation of log(1 + x)
        compression_ratio = density / self.ref_density
        delta_e = self.ionization_base * torch.log1p(compression_ratio)
        
        return delta_e, sigma_sq

    def forward(
        self, 
        order_parameter: torch.Tensor, 
        dt: float, 
        temperature: torch.Tensor, 
        density: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the Double-Exponential extreme-value bound element-wise and 
        injects structurally localized stochastic noise.
        
        Returns:
            u_next (torch.Tensor): The structurally updated order parameter.
            jump_flag (torch.Tensor): A continuous scalar [0, 1] representing 
                                      macroscopic jump intensity for downstream resets.
        """
        delta_e, sigma_sq = self.compute_dynamic_parameters(temperature, density)
        
        # Double-Exponential Extreme-Value Bound (Gumbel-type)
        # P(t) <= exp[-C1 * exp(\Delta E / (\sigma^2 * dt))]
        exponent = delta_e / (sigma_sq * dt)
        
        # Clamp exponent to prevent arithmetic overflow in exp() during extreme gradients
        exponent = torch.clamp(exponent, max=50.0)
        prob_bound = torch.exp(-self.c1 * torch.exp(exponent))
        
        # =====================================================================
        # STRAIGHT-THROUGH ESTIMATOR (STE) FOR STOCHASTIC JUMP MASK
        # =====================================================================
        # Uniform sampling for stochastic triggering
        rand_tensor = torch.rand_like(prob_bound)
        
        # Differentiable soft mask using continuous relaxation (Sigmoid approximation)
        soft_mask = torch.sigmoid((prob_bound - rand_tensor) / self.ste_tau)
        
        # Hard binary mask for operational application (non-differentiable intrinsically)
        hard_mask = (prob_bound > rand_tensor).float()
        
        # The STE trick: Forward pass uses hard_mask, backward pass uses soft_mask gradient.
        # This bridges the Zeno Trap mathematically while preserving the Autograd graph.
        jump_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        # =====================================================================
        # TOPOLOGICAL TRANSITION OPERATOR INJECTION
        # =====================================================================
        # Generate Gaussian noise scaled by the local variance \sigma
        noise = torch.randn_like(order_parameter) * torch.sqrt(sigma_sq)
        
        # Element-wise jump application (Differentiable)
        u_jumped = order_parameter + (jump_mask * noise)
        
        # Macroscopic jump indicator: Instead of a boolean, we return a differentiable 
        # continuous scalar indicating the maximum jump intensity in the domain.
        # Downstream accumulators can use this to apply differentiable dampening/resets.
        jump_intensity = jump_mask.max()

        return u_jumped, jump_intensity
