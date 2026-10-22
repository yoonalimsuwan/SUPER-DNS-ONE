"""HCSN Turbulence Closure — learns the Reynolds-stress closure term
that 'collapses' unresolved turbulence into effective viscosity.
Domain meaning: decoherence channel of HCSN == energy cascade to
unresolved scales; we learn the decay rate as a function of flow state.
This is ML-for-RANS-closure, the honest framing (NOT a sequence LM).

Developers : PAI AND Yoon A Limsuwan 
Organization : MSPS NETWORK 
Licence : MIT
Year : 2026 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClosureNet(nn.Module):
    """Given coarse flow state (U, k, eps gradients) predict eddy
    viscosity nu_t — the HCSN 'decoherence rate' of the flow."""
    def __init__(self, d_in=8, hidden=64):
        super().__init__()
        # chi-like bounded output: nu_t in [0, nu_max], like sigmoid(log_chi)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Softplus())   # positive viscosity
        self.log_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, state):                        # (B,N,d_in)
        return self.net(state).squeeze(-1) * torch.exp(self.log_scale) * 10

class CoarseFlowSimulator(nn.Module):
    """1-D advection-diffusion with learned closure. Sanity check:
    with ground-truth analytic nu, does the net recover it?"""
    def __init__(self, n_grid=64):
        super().__init__()
        self.closure = ClosureNet()
        self.dx = 1.0 / n_grid

    def step(self, u, dt=0.01):
        # central differences for a simple diffusion step with learned nu_t
        lap = (torch.roll(u, -1, -1) - 2 * u + torch.roll(u, 1, -1)) / self.dx ** 2
        state = torch.stack([u, lap], -1)               # crude 'features'
        nu = self.closure(state).unsqueeze(-1)          # (B,N,1)
        return u + dt * nu * lap

# --- smoke test: train to match fixed nu=0.02 diffusion ---
if __name__ == "__main__":
    torch.manual_seed(0)
    sim = CoarseFlowSimulator()
    opt = torch.optim.AdamW(sim.parameters(), lr=1e-2)
    x = torch.linspace(0, 1, 64)
    for step in range(200):
        u0 = torch.sin(2 * 3.14159 * x).unsqueeze(0)
        # target: analytic diffusion with nu=0.02 after 10 steps
        u_t = u0.clone()
        for _ in range(10):
            lap = (torch.roll(u_t, -1, -1) - 2 * u_t + torch.roll(u_t, 1, -1)) / (1/64)**2
            u_t = u_t + 0.01 * 0.02 * lap
        u_p = u0.clone()
        for _ in range(10):
            u_p = sim.step(u_p)
        loss = F.mse_loss(u_p, u_t)
        opt.zero_grad(); loss.backward(); opt.step()
    print("cfd OK — closure MSE:", round(loss.item(), 8))
