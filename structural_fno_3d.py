# =============================================================================
# STRUCTURAL FOURIER NEURAL OPERATOR (SFNO 3D)
# AI-Physics Surrogate Model for One-Shot Structural PDE Prediction
# =============================================================================
# Developer    : Yoon A Limsuwan
# Organization : MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Assisted by  : Gemini (AI)
# License      : MIT
# Year         : 2026
#
# Description:
#   A novel Fourier Neural Operator rooted in the Structural Calculus framework.
#   Instead of a standard FNO, this model explicitly incorporates the 
#   Structural Regime Field sigma(x) into its latent spatial convolutions.
#   
#   It learns the mapping operator: G : (u(x,0), sigma(x)) ↦ u(x,T)
#   enabling O(1) time-complexity predictions for Structural Cahn-Hilliard 
#   and Structural Navier-Stokes equations across extreme interfaces.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution Layer.
    Transforms spatial inputs to the Fourier domain, multiplies by 
    learnable complex weights (truncated at 'modes'), and transforms back.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable complex weights
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # Complex multiplication: (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        
        # Compute Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        
        # Initialize output tensor
        out_ft = torch.zeros(batchsize, self.out_channels, x.shape[-3], x.shape[-2], x_ft.shape[-1], 
                             dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant Fourier modes
        m = self.modes
        out_ft[:, :, :m, :m, :m] = self.compl_mul3d(x_ft[:, :, :m, :m, :m], self.weights1)
        out_ft[:, :, -m:, :m, :m] = self.compl_mul3d(x_ft[:, :, -m:, :m, :m], self.weights2)
        out_ft[:, :, :m, -m:, :m] = self.compl_mul3d(x_ft[:, :, :m, -m:, :m], self.weights3)
        out_ft[:, :, -m:, -m:, :m] = self.compl_mul3d(x_ft[:, :, -m:, -m:, :m], self.weights4)
        
        # Return to physical space
        x_out = torch.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]), norm="ortho")
        return x_out

class StructuralFNOLayer(nn.Module):
    """
    Combines Spectral Convolution with the Structural Regime Field (sigma).
    This enforces the regime boundaries in the latent neural representation.
    """
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes)
        self.w = nn.Conv3d(width, width, 1)
        self.sigma_proj = nn.Conv3d(1, width, 1) # Projects sigma to latent width

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # 1. Global Spectral operation
        x1 = self.spectral_conv(x)
        
        # 2. Local Linear operation
        x2 = self.w(x)
        
        # 3. Structural Regime Modulation (sigma acts as a physical gate/multiplier)
        # D_S(u) = sigma * grad(u) -> Latent analogue: sigma_features * (x1 + x2)
        s_feat = torch.sigmoid(self.sigma_proj(sigma)) 
        
        # Combine and apply activation
        return F.gelu(s_feat * (x1 + x2))

class StructuralFNO3D(nn.Module):
    """
    The full Structural Fourier Neural Operator.
    Maps: (u(x,0), sigma(x)) -> u(x,T)
    """
    def __init__(self, modes: int = 8, width: int = 32, num_layers: int = 4):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Input lifting: (u_initial, x, y, z grid coordinates) -> width
        # Assuming input has 4 channels: u_0, and 3 spatial grids
        self.p = nn.Conv3d(4, width, 1)
        
        # Structural FNO layers
        self.layers = nn.ModuleList([StructuralFNOLayer(width, modes) for _ in range(num_layers)])
        
        # Output projection: width -> 1 (u_final)
        self.q1 = nn.Conv3d(width, 128, 1)
        self.q2 = nn.Conv3d(128, 1, 1)

    def _get_grid3d(self, shape, device):
        # Create normalized coordinate grids [-1, 1]
        b, c, nx, ny, nz = shape
        x = torch.linspace(-1, 1, nx, device=device)
        y = torch.linspace(-1, 1, ny, device=device)
        z = torch.linspace(-1, 1, nz, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([X, Y, Z], dim=0).unsqueeze(0).repeat(b, 1, 1, 1, 1)
        return grid

    def forward(self, u_initial: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_initial : (Batch, 1, Nx, Ny, Nz) Initial state at t=0
            sigma     : (Batch, 1, Nx, Ny, Nz) Structural Regime Field
        Returns:
            u_final   : (Batch, 1, Nx, Ny, Nz) Predicted state at t=T (One-Shot)
        """
        grid = self._get_grid3d(u_initial.shape, u_initial.device)
        
        # Concat input with spatial coordinates
        x = torch.cat([u_initial, grid], dim=1)
        
        # Lift to latent space
        x = self.p(x)
        
        # Apply Structural FNO layers
        for layer in self.layers:
            x = layer(x, sigma)
            
        # Project back to physical state
        x = F.gelu(self.q1(x))
        u_final = self.q2(x)
        
        return u_final

# =============================================================================
# Verification Code
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Structural FNO 3D on {device}...")
    
    # 1 Batch, 1 Channel, 32x32x32 Grid
    B, C, N = 2, 1, 32 
    u_0 = torch.randn(B, C, N, N, N, device=device)
    sigma = torch.ones(B, C, N, N, N, device=device)
    
    model = StructuralFNO3D(modes=8, width=32, num_layers=4).to(device)
    
    # One-shot prediction
    u_T_pred = model(u_0, sigma)
    
    print(f"Input shape  : {u_0.shape}")
    print(f"Output shape : {u_T_pred.shape}")
    print("Forward pass successful. Model is ready for integration with AGI ONE.")
