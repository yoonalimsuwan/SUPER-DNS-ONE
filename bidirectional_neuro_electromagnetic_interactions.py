# =============================================================================
# Neuro-Electromagnetic Bridge
# =============================================================================
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

import math

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# Import the 5 SUPER DNS ONE modules precisely as named
from sesi_ntft_rcs3d import PiecewiseDFTAccumulator3D
from sesi_covariant_4vector_potential_maxwell_structural_bridge import CovariantMaxwellStructuralBridge
from sesi_exact_analytical_maxwell_structural_bridge import ExactMaxwellStructuralSolver
from structural_cahn_hilliard_3d_v2 import StructuralCahnHilliard3D, CahnHilliardConfig
from structural_langevin_v3_2 import AdvancedStructuralLangevin

class RemoteNeuroMonitorBridge(nn.Module):
    """
    Fully Differentiable Neuro-Electromagnetic Structural Bridge.
    Models EM stimulation of neural tissue and remote scattered-field monitoring.
    """
    def __init__(
        self, 
        grid_shape: Tuple[int, int, int],
        dx: float = 1.0,
        dt: float = 0.001,
        target_freq_hz: float = 2.4e9,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.device = device
        self.dt = dt
        
        # 1. Continuous Neural Tissue Phase Solver (Highly optimized Conv3D Laplacian)
        ch_cfg = CahnHilliardConfig(dx=dx, dt=dt, laplacian="conv3d", scheme="explicit")
        self.tissue_solver = StructuralCahnHilliard3D(ch_cfg).to(device)
        
        # 2. Discrete Ion/Neurotransmitter Stochastic Solver
        self.ion_solver = AdvancedStructuralLangevin(dt=dt).to(device)
        
        # 3. Exact Maxwell Solver for Tissue EM Propagation
        self.maxwell_solver = ExactMaxwellStructuralSolver(dx=dx, dt=dt, device=device)
        
        # 4. NTFT Accumulator for Remote Observable Monitoring (Radar Cross Section)
        self.ntft_monitor = PiecewiseDFTAccumulator3D(
            target_freq_hz=target_freq_hz, dt=dt, field_shape=grid_shape, device=device
        )

    def forward(
        self,
        e_field: torch.Tensor,
        b_field: torch.Tensor,
        tissue_phase: torch.Tensor,
        ion_coords: torch.Tensor,
        ion_vel: torch.Tensor,
        current_time: float,
        ion_forces_fn: callable
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes one fully differentiable, multi-physics coupled step.
        """
        # A. Update discrete neural ion dynamics (Stochastic BAOAB step)
        ion_coords_next, ion_vel_next, _, _ = self.ion_solver.full_step(
            coords=ion_coords,
            velocities=ion_vel,
            force_fn=ion_forces_fn
        )
        
        # B. Update continuous neural tissue phase
        # Delta_S operator handles the structural gradient natively
        tissue_phase_next = self.tissue_solver.step(u=tissue_phase)
        
        # C. Propagate Electromagnetic Field through the structurally updated brain tissue
        # Evaluates exact Maxwell Stress Tensor and applies structural \Delta_S operator
        e_next, b_next, u_adapted, has_jumped = self.maxwell_solver.step(
            e_field=e_field, 
            b_field=b_field, 
            order_parameter=tissue_phase_next
        )
        
        # D. Accumulate remote monitoring data (Scatter Fields)
        # Inherently manages Zeno constraints via topological jump resets
        fields_t = {
            'Ex': e_next[0], 'Ey': e_next[1], 'Ez': e_next[2],
            'Hx': b_next[0], 'Hy': b_next[1], 'Hz': b_next[2]
        }
        self.ntft_monitor.update(fields_t, current_time, has_jumped=has_jumped)
        
        # Extract remote sensor phasors
        remote_phasors = self.ntft_monitor.get_phasors()
        
        return e_next, b_next, u_adapted, ion_coords_next, ion_vel_next, remote_phasors
