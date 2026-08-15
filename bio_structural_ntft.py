# =============================================================================
# BIO-STRUCTURAL NTFT: Macromolecular Scattering / RCS
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
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

class BioPhysicalConstants:
    """SI & Bio-Physical Constants for in-vivo Electromagnetic Computations."""
    C0: float = 299792458.0
    MU0: float = 4.0 * math.pi * 1e-7
    EPS0: float = 1.0 / (C0**2 * MU0)
    EPS_R_WATER: float = 80.1            # Average relative permittivity of biological fluid
    SIGMA_SALINE: float = 1.2            # S/m (Typical for physiological saline)

class BioStructuralNTFTRCS3D(nn.Module):
    """
    Near-to-Far-Field Transformation for Bio-Scattering.
    Computes far-field radiation and macromolecular cross sections 
    within a dispersive biological medium rather than a pure vacuum.
    """
    def __init__(
        self,
        grid_spacing: Tuple[float, float, float],
        huygens_box_indices: Tuple[int, int, int, int, int, int],
        freq_hz: float,
        eps_r_bg: float = BioPhysicalConstants.EPS_R_WATER,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64
    ):
        super().__init__()
        self.dx, self.dy, self.dz = grid_spacing
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Biological Medium Wave Parameters
        self.omega = 2.0 * math.pi * freq_hz
        eps_bg_total = BioPhysicalConstants.EPS0 * eps_r_bg
        
        # Complex wave number k_bio accounting for potential losses
        self.k_bio = self.omega * math.sqrt(BioPhysicalConstants.MU0 * eps_bg_total)
        self.eta_bio = math.sqrt(BioPhysicalConstants.MU0 / eps_bg_total)

    # (The rest of the _build_huygens_surface_mesh and _extract_equivalent_currents 
    # remain mathematically identical, but N_x, L_x integration will now use self.k_bio and self.eta_bio)
