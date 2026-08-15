# =============================================================================
# MICRO-TARGET BIOLOGICAL DISCRIMINATOR MODULE (SESI FRAMEWORK)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Description: Ultra-lightweight, O(N) optimized native module for differentiating 
# Micro-Drones from Biological Entities (Birds) using pre-extracted fields.
# Focuses on Enstrophy, Thermal Gradients, and Dielectric Permittivity.
# =============================================================================
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
from typing import Dict

__all__ = ["MicroBiologicalDiscriminator"]

class MicroBiologicalDiscriminator(nn.Module):
    """
    Evaluates micro-signatures (Enstrophy, Thermal Gradients, Dielectric) to 
    classify small entities as Synthetic (Micro-Drone) or Biological (Bird).
    Fully differentiable for end-to-end production training.
    """
    def __init__(
        self, 
        c1: float = 1.0, 
        ste_tau: float = 0.05,
        water_dielectric_baseline: float = 80.0, # Baseline for biological water content
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.ste_tau = ste_tau
        self.water_baseline = water_dielectric_baseline
        
        # Learnable topological weights for micro-classification
        self.enstrophy_weight = nn.Parameter(torch.tensor(1.0))
        self.thermal_grad_weight = nn.Parameter(torch.tensor(1.5))
        self.dielectric_weight = nn.Parameter(torch.tensor(2.0))
        
        # Ultra-lightweight classification MLP for O(N) production speed[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span)
        self.classifier_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    @torch.jit.export
    def compute_biological_divergence_bound(
        self, 
        divergence_field: torch.Tensor, 
        variance_sq: torch.Tensor, 
        dt: float
    ) -> torch.Tensor:
        """
        Computes the probability bound of an entity deviating from natural 
        biological baselines using the SESI Gumbel-type extreme value statistics[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span).
        """
        # Clamp exponent to prevent NaN in extreme edge cases[span_6](start_span)[span_6](end_span)[span_7](start_span)[span_7](end_span)
        exponent = torch.clamp(divergence_field / (variance_sq * dt + 1e-12), max=50.0)
        prob_bound = torch.exp(-self.c1 * torch.exp(exponent))
        return prob_bound

    def forward(
        self, 
        vorticity_field: torch.Tensor,    # Curl of velocity field (Vortices)
        thermal_field: torch.Tensor,      # High-res thermal footprint
        dielectric_field: torch.Tensor,   # Material permittivity (Water vs Synthetic)
        propulsion_emf: torch.Tensor,     # Battery micro-leaks[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span)
        dt: float
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Kinematic Divergence (Vortical Enstrophy)
        # Birds create organized wake; Drones create chaotic shear enstrophy.
        # Enstrophy = 0.5 * |curl(V)|^2
        enstrophy = 0.5 * (vorticity_field ** 2).sum(dim=1, keepdim=True)
        
        # 2. Thermodynamic Divergence (Thermal Gradient Entropy)
        # We simulate spatial thermal variance. Drones have high variance (hot battery, cold chassis).
        # Approximated by the square of the thermal field relative to its mean.
        thermal_mean = thermal_field.mean(dim=1, keepdim=True)
        thermal_variance = (thermal_field - thermal_mean) ** 2
        
        # 3. Material Divergence (Dielectric Deficit)
        # Biological entities are mostly water (e_r ~ 80). Stealth composites are much lower.
        dielectric_deficit = torch.abs(self.water_baseline - dielectric_field)
        
        # Total synthetic variance signature
        sigma_sq = enstrophy + thermal_variance + (propulsion_emf ** 2)
        
        # 4. Construct Biological Anomaly Barrier[span_10](start_span)[span_10](end_span)[span_11](start_span)[span_11](end_span)
        # High deficit indicates synthetic material
        divergence_ratio = dielectric_deficit / torch.clamp(dielectric_field, min=1e-6)
        delta_bio_field = torch.log1p(divergence_ratio) + sigma_sq
        
        # 5. Evaluate Structural Anomaly Probability Bound[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span)
        prob_bound = self.compute_biological_divergence_bound(delta_bio_field, sigma_sq, dt)
        
        # 6. Spatial Pooling for Macro-Signatures[span_14](start_span)[span_14](end_span)[span_15](start_span)[span_15](end_span)
        pooled_enstrophy = torch.mean(enstrophy * prob_bound) * self.enstrophy_weight
        pooled_thermal_grad = torch.mean(thermal_variance * prob_bound) * self.thermal_grad_weight
        pooled_material = torch.mean(dielectric_deficit * prob_bound) * self.dielectric_weight
        
        features = torch.stack([pooled_enstrophy, pooled_thermal_grad, pooled_material], dim=-1)
        
        # 7. Differentiable Target Classification (1.0 = Micro-Drone, 0.0 = Bird)[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span)
        logit = self.classifier_head(features)
        
        # Continuous relaxation (Soft routing) via Straight-Through Estimator (STE)[span_18](start_span)[span_18](end_span)[span_19](start_span)[span_19](end_span)
        soft_classification = torch.sigmoid(logit / self.ste_tau)
        
        rand_tensor = torch.rand_like(soft_classification)
        hard_classification = (soft_classification > rand_tensor).float()
        
        final_prediction = hard_classification.detach() - soft_classification.detach() + soft_classification
        
        return {
            "is_micro_drone": final_prediction,
            "classification_confidence": soft_classification,
            "extracted_features": features
        }
