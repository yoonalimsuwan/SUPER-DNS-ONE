# =============================================================================
# SESI Differentiable Hypersonic Battery Solver (Production Grade)
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
import torch.nn.functional as F

class SESIFullDifferentiableHypersonicBatterySolver(torch.nn.Module):
    """
    Native Full-Differentiable Advanced Battery Dynamics Solver for Hypersonic Applications.
    Enables end-to-end gradient descent optimization for microstructural design.
    """
    def __init__(self, grid_size: int = 128, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.grid_size = grid_size
        
        # Learnable or Fixed Physical Constants (Fully Differentiable Parameters)
        self.C1 = nn_param = torch.nn.Parameter(torch.tensor(1.05, device=self.device))
        self.sigma_sq = torch.nn.Parameter(torch.tensor(0.02, device=self.device))
        self.dt = 0.001
        
        # Pre-allocate Laplacian Kernel
        self.register_buffer('laplacian_kernel', self._build_laplacian_kernel())

    def _build_laplacian_kernel(self) -> torch.Tensor:
        kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                               [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                               [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], 
                              dtype=torch.float32, device=self.device)
        return kernel.view(1, 1, 3, 3, 3)

    def calculate_activation_energy(self, interface_state: torch.Tensor, perturbed_state: torch.Tensor) -> torch.Tensor:
        energy_current = torch.norm(interface_state, p=2, dim=(1, 2, 3))
        energy_new = torch.norm(perturbed_state, p=2, dim=(1, 2, 3))
        delta_E = torch.relu(energy_new - energy_current)
        return delta_E + 1e-6

    def differentiable_gumbel_relax_filter(self, delta_E: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Soft/Differentiable relaxation of the Zeno trigger mask using Gumbel-Sigmoid 
        to allow gradient flow through discrete topological decision points.
        """
        exponent = delta_E / (torch.clamp(self.sigma_sq, min=1e-4) * self.dt)
        probability_bound = torch.exp(-self.C1 * torch.exp(exponent))
        
        # Reparameterized Soft Thresholding for Differentiability
        if self.training:
            # Soft relaxation for gradient propagation
            soft_mask = torch.sigmoid((probability_bound - torch.rand_like(probability_bound)) / temperature)
            return soft_mask
        else:
            # Hard thresholding for inference
            return (torch.rand_like(probability_bound) < probability_bound).float()

    def forward(self, interface_state: torch.Tensor, steps: int = 100) -> torch.Tensor:
        current_interface = interface_state.to(self.device)
        
        for _ in range(steps):
            # 1. Continuous SDE Phase with Autograd Support
            stochastic_noise = torch.randn_like(current_interface) * 0.01
            drift = F.conv3d(current_interface.unsqueeze(1), self.laplacian_kernel, padding=1).squeeze(1) * self.dt
            perturbed_state = current_interface + drift + stochastic_noise
            
            # 2. Activation Energy
            delta_E = self.calculate_activation_energy(current_interface, perturbed_state)
            
            # 3. Differentiable Trigger Mask
            trigger_mask = self.differentiable_gumbel_relax_filter(delta_E)
            
            # 4. Differentiable Topological Operators (Branching & Merging Soft Interpolation)
            laplacian = F.conv3d(perturbed_state.unsqueeze(1), self.laplacian_kernel, padding=1).squeeze(1)
            structural_jump = torch.where(laplacian > 0.5, perturbed_state * 1.5, perturbed_state * 0.5)
            
            # Weighted blend instead of hard discrete branch to preserve gradients
            mask_expanded = trigger_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            current_interface = mask_expanded * structural_jump + (1.0 - mask_expanded) * perturbed_state
            
        return current_interface
