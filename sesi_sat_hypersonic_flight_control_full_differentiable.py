=============================================================================
SESI HYPERSONIC FLIGHT CONTROL MODULE (PRODUCTION RELEASE)
=============================================================================
Framework   : Self-Evolving Structural Interfaces (SESI)
Module      : Autopilot & Aero-Topological Interface Controller
Developer   : PAI AND Yoon A Limsuwan : MSPS NETWORK
License     : MIT
Year        : 2026
Version     : 2.0.0 (Native Differentiable / Log-Domain Optimization)
=============================================================================
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("SESIHypersonicFlight_Diff")

@dataclass
class HypersonicFlightConfig:
    """Configuration for SESI Hypersonic Flight Controller."""
    mach_number: float = 7.0
    c1_geom: float = 1.2
    dt_control_loop: float = 1e-4
    gumbel_safety_cutoff: float = 1e-9  
    max_control_deflection: float = 15.0 # Degrees
    gating_temperature: float = 10.0     # Controls the steepness of the differentiable gate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32   # Downgraded to fp32 for perf, Log-domain ensures stability


class AeroLogGumbelEngine(nn.Module):
    """
    Computes Log-Probability of double-exponential extreme-value bounds.
    Operating in the log domain prevents gradient vanishing/explosion 
    inherent to nested exponentials.
    """
    def __init__(self, config: HypersonicFlightConfig):
        super().__init__()
        self.config = config
        # Precompute the log of the safety cutoff for efficiency
        self.log_cutoff = math.log(self.config.gumbel_safety_cutoff)
        self.target_inner_exp = math.log(-self.log_cutoff / self.config.c1_geom)

    def compute_log_bound(self, delta_E: torch.Tensor, turbulence_sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Computes ln(P) = -C1 * exp(delta_E / (sigma^2 * dt))
        """
        denom = (turbulence_sigma_sq * self.config.dt_control_loop).clamp(min=1e-8)
        inner_exponent = (delta_E / denom).clamp(max=50.0) # Prevent inf in exp
        
        # ln(P)
        log_prob = -self.config.c1_geom * torch.exp(inner_exponent)
        return log_prob, inner_exponent


class SESIHypersonicFlightController(nn.Module):
    """
    Fully Differentiable SESI Flight Controller.
    Uses continuous quotient mapping to avoid branching, resolving the Zeno trap
    in structural interfaces without breaking the PyTorch computation graph.
    """

    def __init__(self, config: Optional[HypersonicFlightConfig] = None):
        super().__init__()
        self.config = config or HypersonicFlightConfig()
        self.ev_engine = AeroLogGumbelEngine(self.config)

    def calculate_aerodynamic_energy(self, shock_state: torch.Tensor, turbulence: torch.Tensor) -> torch.Tensor:
        """
        Estimates structural energy using differentiable tensor contractions.
        """
        # Batched / Vectorized bulk pressure and stress
        bulk_pressure = 0.5 * torch.sum(shock_state ** 2)
        
        # Differentiable gradient calculation (using central difference via convolution/pad or simplified diff)
        # Using simplified gradient magnitude proxy for speed
        grad_x = shock_state - torch.roll(shock_state, shifts=1, dims=0)
        interface_stress = torch.sum(turbulence * torch.abs(grad_x))
        
        return bulk_pressure + interface_stress

    def forward(
        self,
        current_shock_state: torch.Tensor,
        current_turbulence_field: torch.Tensor,
        proposed_control_deflection: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Main control loop execution (100% Branch-Free / Differentiable).
        """
        # 1. Structural Energy E(Γ(τ^-))
        e_current = self.calculate_aerodynamic_energy(current_shock_state, current_turbulence_field)
        
        # 2. Perturbation mimicking topological transition (nucleation/merging/branching)
        # We use reparameterization (fixed seed/noise map) if stochasticity gradients are needed, 
        # but standard randn is fine if we only differentiate wrt state.
        noise = torch.randn_like(current_shock_state) * 0.15
        distorted_shock = current_shock_state * (1.0 + noise)
        e_transition = self.calculate_aerodynamic_energy(distorted_shock, current_turbulence_field)
        
        # 3. Activation Energy (ΔE)
        delta_e = F.softplus(e_transition - e_current) + 1e-3 # Smooth clamping
        
        # 4. Gumbel Log-Probability
        sigma_sq = torch.mean(current_turbulence_field ** 2)
        log_failure_prob, inner_exp = self.ev_engine.compute_log_bound(delta_e, sigma_sq)
        
        # 5. Differentiable No-Zeno Intervention (Topological Gating)
        # Instead of if/else, we use a smooth sigmoid gate. 
        # gate approaches 1.0 when log_failure_prob > log_cutoff, and 0.0 otherwise.
        gate = torch.sigmoid((log_failure_prob - self.ev_engine.log_cutoff) * self.config.gating_temperature)
        
        # Required delta E to reach safe state
        required_delta_e = self.ev_engine.target_inner_exp * (sigma_sq * self.config.dt_control_loop)
        energy_deficit = F.relu(required_delta_e - delta_e) # Smooth minimum bound
        
        # Proportional adjustment bounded smoothly
        raw_adjustment = energy_deficit * 0.05
        deflection_adjustment = self.config.max_control_deflection * torch.tanh(raw_adjustment / self.config.max_control_deflection)
        
        # Optimized Control Vector (Convex combination / residual addition via gate)
        optimized_deflection = proposed_control_deflection + (gate * deflection_adjustment)
        
        # Final smooth clamp
        optimized_deflection = self.config.max_control_deflection * torch.tanh(optimized_deflection / self.config.max_control_deflection)

        return {
            "optimized_control_deflection": optimized_deflection,
            "aerodynamic_activation_energy": delta_e,
            "log_shock_transition_probability": log_failure_prob,
            "zeno_intervention_gate": gate # Replaces binary boolean with continuous metric
        }

# =============================================================================
# SELF-TEST & GRADIENT VALIDATION
# =============================================================================
if __name__ == "__main__":
    print("====================================================================")
    print(" Executing SESI Native Differentiable Flight Controller Test")
    print("====================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HypersonicFlightConfig(mach_number=8.0, device=str(device))
    autopilot = SESIHypersonicFlightController(config).to(device)
    
    # Enable Autograd for Inputs
    grid_res = 32
    shock_density = torch.ones(grid_res, grid_res, grid_res, device=device, dtype=config.dtype, requires_grad=True)
    turbulence = (torch.rand(grid_res, grid_res, grid_res, device=device, dtype=config.dtype) * 0.2).requires_grad_(True)
    proposed_pitch = torch.tensor([2.0], device=device, dtype=config.dtype, requires_grad=True)
    
    # 1. Forward Pass
    telemetry = autopilot(shock_density, turbulence, proposed_pitch)
    
    # 2. Backward Pass (Proof of native differentiability)
    loss = telemetry["optimized_control_deflection"].sum()
    loss.backward()
    
    print(f"  [TELEMETRY] Proposed Pitch       : 2.000 deg")
    print(f"  [TELEMETRY] Optimized Pitch      : {telemetry['optimized_control_deflection'].item():.3f} deg")
    print(f"  [TELEMETRY] Flow Act. Energy (ΔE): {telemetry['aerodynamic_activation_energy'].item():.3f}")
    print(f"  [TELEMETRY] Log Transition Prob  : {telemetry['log_shock_transition_probability'].item():.3f}")
    print(f"  [STATUS]    Zeno Active (Gate)   : {telemetry['zeno_intervention_gate'].item():.3f} (0=Off, 1=On)")
    print(f"  [GRADIENTS] Shock Density Grad   : {shock_density.grad.norm().item():.5f} (Success)")
    print(f"  [GRADIENTS] Proposed Pitch Grad  : {proposed_pitch.grad.item():.5f} (Success)")
    print("====================================================================")
