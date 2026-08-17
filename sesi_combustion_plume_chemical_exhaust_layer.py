# =============================================================================
# COMBUSTION PLUME & CHEMICAL EXHAUST LAYER (SESI FRAMEWORK)
# SUPER DNS ONE Cluster / ONE Ecosystem - Production Release
# =============================================================================
# Description: Ultra-optimized, native, fully differentiable O(N) layer.
# Evaluates extreme thermal gradients and chemical species turbulence 
# (Combustion Plumes) for traditional Hypersonic targets.
# Integrates seamlessly in parallel to the Battery/EMF Tensor Network.
# =============================================================================
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# =============================================================================

import torch
import torch.nn as nn
from typing import Dict

__all__ = ["CombustionPlumeTensorLayer"]

class CombustionPlumeTensorLayer(nn.Module):
    """
    Extracts structural signatures of chemical exhaust and combustion plumes 
    from hypersonic bodies. Uses gradient-free (O(N) pooled) proxies to evaluate
    turbulent shear and localized core heating without expensive 3D convolutions.
    """
    def __init__(
        self, 
        c2: float = 1.0, 
        ste_tau: float = 0.05,
        mach_threshold: float = 5.0
    ) -> None:
        super().__init__()
        self.c2 = c2
        self.ste_tau = ste_tau
        self.mach_threshold = mach_threshold
        
        # Learnable topological weights for Chemical & Thermal structural bounds
        self.plume_core_thermal_weight = nn.Parameter(torch.tensor(1.8))
        self.chemical_shear_weight = nn.Parameter(torch.tensor(1.4))
        self.exhaust_velocity_weight = nn.Parameter(torch.tensor(1.2))
        
        # Ultra-lightweight MLP for O(N) integration into the main Tensor Network
        self.plume_classifier_head = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1)
        )

    @torch.jit.export
    def compute_plume_anomaly_bound(
        self, 
        localized_heat_sq: torch.Tensor, 
        chemical_variance: torch.Tensor, 
        dt: float
    ) -> torch.Tensor:
        """
        Computes the Double-Exponential anomaly bound for the combustion plume.
        Uses squared thermal and chemical variances to avoid square-root overhead.
        """
        denominator = chemical_variance * dt + 1e-12
        exponent = torch.clamp(localized_heat_sq / denominator, max=50.0)
        
        # Gumbel-type bound: exp[-C2 * exp(Heat / (ChemVar * dt))]
        inner_exp = torch.exp(exponent)
        prob_bound = torch.exp(-self.c2 * inner_exp)
        
        return prob_bound

    def forward(
        self, 
        velocity_field: torch.Tensor,          # Ambient vs Plume velocity
        thermal_field: torch.Tensor,           # Localized extreme temperatures
        chemical_density: torch.Tensor,        # Unburned hydrocarbons / OH- radicals
        ambient_temperature: torch.Tensor,     # Baseline for gradient extraction
        dt: float
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Combustion Core Proxies (O(N) Operations)
        # Subtract ambient to isolate localized extreme exhaust heat without exact gradients
        delta_thermal = torch.relu(thermal_field - ambient_temperature)
        localized_heat_sq = delta_thermal ** 2
        
        # 2. Chemical Shear & Turbulence Proxy
        # Plumes create massive chemical density variances. 
        # Using simple squared density as a proxy for chemical momentum.
        chemical_variance = chemical_density ** 2
        
        # 3. Exhaust Velocity Proxy (Relative to main body velocity)
        # Approximated by the velocity field variance in the exhaust region
        v_sq = (velocity_field ** 2).sum(dim=1, keepdim=True)
        exhaust_kinetic_energy = 0.5 * chemical_density * v_sq
        
        # 4. Plume Anomaly Boundary (No-Zeno Bound Equivalent)
        prob_bound = self.compute_plume_anomaly_bound(localized_heat_sq, chemical_variance, dt)
        
        # 5. Specialized Feature Pooling for Combustion
        pooled_thermal_core = torch.mean(localized_heat_sq * prob_bound) * self.plume_core_thermal_weight
        pooled_chem_shear = torch.mean(chemical_variance * prob_bound) * self.chemical_shear_weight
        pooled_exhaust_ke = torch.mean(exhaust_kinetic_energy * prob_bound) * self.exhaust_velocity_weight
        
        # Stack features for the Tensor Network queue
        plume_features = torch.stack([pooled_thermal_core, pooled_chem_shear, pooled_exhaust_ke], dim=-1)
        
        # 6. Differentiable Target Classification (Combustion Confirmation)
        logit = self.plume_classifier_head(plume_features)
        
        # Continuous relaxation (Soft routing)
        soft_classification = torch.sigmoid(logit / self.ste_tau)
        
        # Hard binary mask (1.0 = Chemical Combustion Target, 0.0 = Background/Battery)
        rand_tensor = torch.rand_like(soft_classification)
        hard_classification = (soft_classification > rand_tensor).float()
        
        # Straight-Through Estimator (STE)
        final_prediction = hard_classification.detach() - soft_classification.detach() + soft_classification
        
        return {
            "is_combustion_target": final_prediction,
            "combustion_confidence": soft_classification,
            "plume_features_tensor": plume_features # Output this to queue into the main Omni-Detector
        }

