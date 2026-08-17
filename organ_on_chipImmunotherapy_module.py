# =============================================================================
# Organ On Chip Immunotherapy Module
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganOnChipImmunotherapyModule(nn.Module):
    """
    Production-grade, fully differentiable PyTorch module for real-time 
    Organ-on-a-Chip tissue sensor monitoring and immunotherapy toxicity prediction 
    (e.g., Cytokine Release Syndrome - CRS, ICANS).
    
    Incorporates double-exponential topological barrier dynamics and 
    structural tensor contraction for high-efficiency optimization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Structural Tensor Contraction Layers (Inspired by structural calculus Phi_U mapping)
        self.tensor_contractor = nn.Linear(input_dim, hidden_dim, bias=True)
        self.tissue_gate = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Parameters for Double-Exponential (Gumbel-type) Topological Transition Barrier
        # Controls the activation threshold for severe inflammatory phase changes (CRS / Toxicity)
        self.barrier_energy = nn.Parameter(torch.tensor(1.5))
        self.noise_variance = nn.Parameter(torch.tensor(0.2))
        
        # Output classification/regression head for toxicity risk
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights for production stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_double_exponential_barrier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the strict No-Zeno double-exponential transition probability 
        to model sharp physiological threshold crossings (e.g., sudden cytokine spikes).
        Fully differentiable with respect to barrier energy and fluctuation variance.
        """
        # Ensure numerical stability of variance
        sigma_sq = torch.clamp(self.noise_variance, min=1e-4)
        delta_e = torch.clamp(self.barrier_energy, min=1e-3)
        
        # Gumbel-type activation scaling factor
        # P(tau_{k+1} - tau_k < dt) <= exp(-C1 * exp(Delta E / (sigma^2 * dt)))
        # Here we map this into a dynamic gating weight tensor
        scaling_factor = torch.exp(-torch.exp(delta_e / (sigma_sq * (torch.abs(x) + 1e-5))))
        return scaling_factor

    def forward(self, sensor_stream: torch.Tensor, baseline_reference: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for real-time Organ-on-a-Chip telemetry.
        
        Args:
            sensor_stream (Tensor): Real-time tissue inflammation and micro-sensor data [Batch, Seq_Len, Input_Dim] or [Batch, Input_Dim].
            baseline_reference (Tensor, optional): Reference tissue geometry or baseline state for pullback alignment.
            
        Returns:
            logits (Tensor): Toxicity risk classification logits (e.g., Safe vs. High CRS Risk).
            transition_metrics (Tensor): Double-exponential topological transition indicators.
        ```
        """
        # Step 1: Structural Tensor Contraction (Mapping high-dimensional micro-sensor inputs)
        contracted_state = self.tensor_contractor(sensor_stream)
        
        if baseline_reference is not None and baseline_reference.shape == contracted_state.shape:
            contracted_state = contracted_state - baseline_reference
            
        # Step 2: Non-linear semantic state transformation with gating
        gate = torch.sigmoid(self.tissue_gate(contracted_state))
        modulated_features = contracted_state * gate
        
        # Step 3: Apply Double-Exponential Topological Transition Modulation
        transition_factor = self.compute_double_exponential_barrier(modulated_features)
        structural_manifold = modulated_features * (1.0 - transition_factor)
        
        # Step 4: Final prediction head evaluation
        logits = self.head(structural_manifold)
        
        return logits, transition_factor
