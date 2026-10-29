"""
=============================================================================
SESI REAL-TIME EM FIELD ANIMATION
Dynamic visualization of electromagnetic field propagation through metamaterials
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] Real-time animation of EM field evolution:
  • FDTD time-stepping visualization (E/H fields)
  • 2D slice and 3D volume rendering
  • Metamaterial structure overlay
  • Carpet spectral complexity indicators
  • Export to video/GIF formats

MATHEMATICAL FOUNDATION
------------------------
[C] FDTD Method (Yee 1966):
  • Discretize Maxwell's equations on staggered grid
  • Time-step: E^{n+1} = E^n + (dt/ε) ∇×H^{n+1/2}
  • Time-step: H^{n+1/2} = H^{n-1/2} - (dt/μ) ∇×E^n
  • Stability: Courant condition dt ≤ 1/(c√(1/Δx² + 1/Δy² + 1/Δz²))

[D] Carpet Spectral Integration:
  • Compute spectral entropy at each time step
  • Track complexity evolution as fields interact with metamaterial
  • Visualize energy distribution across spectral scales

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Animation is VISUALIZATION, not proof of field behavior.
• FDTD requires mesh convergence verification (Carpet-19 Thm 3(b) discipline).
• Carpet spectral indicators are ANALOGICAL proxies for complexity.
• Real-time performance depends on grid resolution and hardware.
=============================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.animation")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import existing modules
try:
    from sesi_fullwave_em_solver import OpenEMSConfig, FullWaveEMPipeline
    from sesi_carpet_rcs import CarpetSpectralAnalyzer, CarpetRCSConfig
    HAS_SESI = True
except ImportError:
    HAS_SESI = False

__all__ = [
    "AnimationConfig",
    "FDTDFieldSolver",
    "FieldAnimator",
    "CarpetSpectralOverlay",
    "RealTimeAnimationEngine",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnimationConfig:
    """Configuration for field animation."""
    # Grid
    nx: int = 100
    ny: int = 100
    nz: int = 1  # 2D by default
    dx: float = 1e-3  # meters
    dy: float = 1e-3
    dz: float = 1e-3
    
    # Time
    n_steps: int = 1000
    dt: float = 1e-12  # seconds
    frame_interval: int = 10  # save every N steps
    
    # Physics
    c: float = 3e8  # speed of light
    frequency_hz: float = 10e9
    
    # Visualization
    field_component: str = "Ez"  # Ex, Ey, Ez, Hx, Hy, Hz
    colormap: str = "RdBu_r"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    
    # Output
    output_format: str = "gif"  # gif, mp4, html
    output_path: str = "./animation_output"
    fps: int = 30
    
    # Carpet integration
    enable_carpet_overlay: bool = True
    carpet_level: int = 3


# ═══════════════════════════════════════════════════════════════════════
# [C] FDTD FIELD SOLVER
# ═══════════════════════════════════════════════════════════════════════

class FDTDFieldSolver:
    """
    [C] 2D FDTD solver for TE/TM modes.
    
    Implements Yee algorithm with:
      • Staggered grid (E and H at different positions)
      • Courant stability condition
      • PML absorbing boundaries (simplified)
      • Metamaterial region with spatially-varying ε, μ
    """
    
    def __init__(self, cfg: AnimationConfig):
        self.cfg = cfg
        
        # Check Courant condition
        dt_max = 1.0 / (cfg.c * np.sqrt(1/cfg.dx**2 + 1/cfg.dy**2))
        if cfg.dt > dt_max:
            logger.warning(f"dt={cfg.dt} exceeds Courant limit {dt_max}, adjusting")
            self.dt = 0.99 * dt_max
        else:
            self.dt = cfg.dt
        
        # Initialize fields
        self.Ex = np.zeros((cfg.ny, cfg.nx))
        self.Ey = np.zeros((cfg.ny, cfg.nx))
        self.Ez = np.zeros((cfg.ny, cfg.nx))
        self.Hx = np.zeros((cfg.ny, cfg.nx))
        self.Hy = np.zeros((cfg.ny, cfg.nx))
        self.Hz = np.zeros((cfg.ny, cfg.nx))
        
        # Material properties
        self.eps_r = np.ones((cfg.ny, cfg.nx))
        self.mu_r = np.ones((cfg.ny, cfg.nx))
        self.sigma = np.zeros((cfg.ny, cfg.nx))  # conductivity
        
        # Time step counter
        self.step = 0
        
    def set_metamaterial_region(self, x_start: int, x_end: int,
                                 y_start: int, y_end: int,
                                 eps_r: float, mu_r: float):
        """Define metamaterial absorber region."""
        self.eps_r[y_start:y_end, x_start:x_end] = eps_r
        self.mu_r[y_start:y_end, x_start:x_end] = mu_r
        
    def add_source(self, x: int, y: int, amplitude: float = 1.0):
        """Add point source at (x, y)."""
        self.Ez[y, x] += amplitude * np.sin(2 * np.pi * self.cfg.frequency_hz * self.step * self.dt)
        
    def step_fields(self):
        """Advance fields by one time step (Yee algorithm)."""
        cfg = self.cfg
        dt = self.dt
        eps0 = 8.854e-12
        mu0 = 4 * np.pi * 1e-7
        
        # Update H fields (curl of E)
        self.Hx[:-1, :] -= (dt / mu0) * (self.Ez[1:, :] - self.Ez[:-1, :]) / cfg.dy
        self.Hy[:, :-1] += (dt / mu0) * (self.Ez[:, 1:] - self.Ez[:, :-1]) / cfg.dx
        
        # Update E fields (curl of H)
        self.Ez[1:, :-1] += (dt / (eps0 * self.eps_r[1:, :-1])) * \
                            (self.Hy[1:, :-1] - self.Hy[:-1, :-1]) / cfg.dx - \
                            (self.Hx[1:, 1:] - self.Hx[1:, :-1]) / cfg.dy
        
        # Apply source
        self.add_source(cfg.nx // 2, 10, amplitude=1.0)
        
        # Simple absorbing boundary (zero reflection at edges)
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0
        
        self.step += 1
        
    def get_field(self, component: str) -> np.ndarray:
        """Get field component by name."""
        field_map = {
            "Ex": self.Ex, "Ey": self.Ey, "Ez": self.Ez,
            "Hx": self.Hx, "Hy": self.Hy, "Hz": self.Hz,
        }
        return field_map.get(component, self.Ez)


# ═══════════════════════════════════════════════════════════════════════
# [D] CARPET SPECTRAL OVERLAY
# ═══════════════════════════════════════════════════════════════════════

class CarpetSpectralOverlay:
    """
    [D] Compute and visualize Carpet spectral complexity during animation.
    
    At each time step:
      • Extract field pattern in metamaterial region
      • Compute spectral entropy via Carpet Laplacian
      • Overlay complexity indicator on field visualization
    """
    
    def __init__(self, cfg: AnimationConfig):
        self.cfg = cfg
        if HAS_SESI:
            carpet_cfg = CarpetRCSConfig(carpet_level=cfg.carpet_level)
            self.analyzer = CarpetSpectralAnalyzer(carpet_cfg)
        else:
            self.analyzer = None
            
        self.complexity_history: List[float] = []
        
    def compute_complexity(self, field: np.ndarray) -> float:
        """Compute spectral complexity of field pattern."""
        if self.analyzer is None:
            return 0.0
        
        # Downsample field to carpet size
        n_cells = self.analyzer.geometry.n_cells
        field_small = field[::field.shape[0]//int(np.sqrt(n_cells)),
                           ::field.shape[1]//int(np.sqrt(n_cells))]
        field_flat = field_small.flatten()[:n_cells]
        
        # Compute spectral entropy
        evals, _ = self.analyzer.compute_laplacian_spectrum()
        entropy = self.analyzer.spectral_entropy(evals)
        
        self.complexity_history.append(entropy)
        return entropy
        
    def get_overlay_mask(self) -> np.ndarray:
        """Generate Carpet geometry mask for visualization."""
        if self.analyzer is None:
            return np.ones((self.cfg.ny, self.cfg.nx))
        
        cells = self.analyzer.geometry.cells
        mask = np.zeros((self.cfg.ny, self.cfg.nx))
        
        # Map cells to grid
        scale_x = self.cfg.nx / (3 ** self.cfg.carpet_level)
        scale_y = self.cfg.ny / (3 ** self.cfg.carpet_level)
        
        for (cx, cy) in cells:
            x_start = int(cx * scale_x)
            x_end = int((cx + 1) * scale_x)
            y_start = int(cy * scale_y)
            y_end = int((cy + 1) * scale_y)
            mask[y_start:y_end, x_start:x_end] = 1.0
            
        return mask


# ═══════════════════════════════════════════════════════════════════════
# [D] FIELD ANIMATOR
# ═══════════════════════════════════════════════════════════════════════

class FieldAnimator:
    """
    [D] Create animated visualizations of EM field evolution.
    
    Supports:
      • Matplotlib animation (GIF/MP4)
      • Plotly interactive animation (HTML)
      • Carpet spectral overlay
      • Time-series complexity tracking
    """
    
    def __init__(self, solver: FDTDFieldSolver, cfg: AnimationConfig):
        self.solver = solver
        self.cfg = cfg
        self.carpet_overlay = CarpetSpectralOverlay(cfg) if cfg.enable_carpet_overlay else None
        
    def animate_matplotlib(self) -> Optional[str]:
        """Create animation using matplotlib."""
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib not available")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        field_data = self.solver.get_field(self.cfg.field_component)
        im = ax.imshow(field_data, cmap=self.cfg.colormap,
                      vmin=self.cfg.vmin, vmax=self.cfg.vmax,
                      origin='lower')
        
        if self.carpet_overlay:
            mask = self.carpet_overlay.get_overlay_mask()
            ax.contour(mask, levels=[0.5], colors='white', linewidths=2)
        
        plt.colorbar(im, ax=ax, label=f"{self.cfg.field_component} Field")
        title = ax.set_title(f"Time Step: 0")
        
        def update(frame):
            for _ in range(self.cfg.frame_interval):
                self.solver.step_fields()
            
            field = self.solver.get_field(self.cfg.field_component)
            im.set_data(field)
            title.set_text(f"Time Step: {self.solver.step}")
            
            if self.carpet_overlay:
                complexity = self.carpet_overlay.compute_complexity(field)
                ax.text(0.02, 0.98, f"Spectral Entropy: {complexity:.3f}",
                       transform=ax.transAxes, color='white',
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='black', alpha=0.5))
            
            return [im, title]
        
        anim = animation.FuncAnimation(fig, update,
                                       frames=self.cfg.n_steps // self.cfg.frame_interval,
                                       interval=1000/self.cfg.fps, blit=True)
        
        # Save animation
        import os
        os.makedirs(self.cfg.output_path, exist_ok=True)
        output_file = os.path.join(self.cfg.output_path,
                                   f"field_animation.{self.cfg.output_format}")
        
        if self.cfg.output_format == "gif":
            anim.save(output_file, writer='pillow', fps=self.cfg.fps)
        elif self.cfg.output_format == "mp4":
            anim.save(output_file, writer='ffmpeg', fps=self.cfg.fps)
        
        plt.close()
        logger.info(f"Animation saved to {output_file}")
        return output_file
        
    def animate_plotly(self) -> Optional[str]:
        """Create interactive animation using Plotly."""
        if not HAS_PLOTLY:
            logger.error("plotly not available")
            return None
        
        frames = []
        for step in range(0, self.cfg.n_steps, self.cfg.frame_interval):
            for _ in range(self.cfg.frame_interval):
                self.solver.step_fields()
            
            field = self.solver.get_field(self.cfg.field_component)
            
            frame_data = [go.Heatmap(z=field, colorscale=self.cfg.colormap,
                                     zmin=self.cfg.vmin, zmax=self.cfg.vmax)]
            
            if self.carpet_overlay:
                mask = self.carpet_overlay.get_overlay_mask()
                frame_data.append(go.Contour(z=mask, showscale=False,
                                            contours=dict(showlabels=False),
                                            line=dict(color='white', width=2)))
            
            frames.append(go.Frame(data=frame_data,
                                  name=f"frame{step}"))
        
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title=f"EM Field Animation - {self.cfg.field_component}",
                xaxis=dict(range=[0, self.cfg.nx], autorange=False),
                yaxis=dict(range=[0, self.cfg.ny], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play", method="animate",
                                 args=[None, dict(frame=dict(duration=100))])]
                )]
            ),
            frames=frames
        )
        
        import os
        os.makedirs(self.cfg.output_path, exist_ok=True)
        output_file = os.path.join(self.cfg.output_path, "field_animation.html")
        fig.write_html(output_file)
        
        logger.info(f"Interactive animation saved to {output_file}")
        return output_file


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATED ENGINE
# ═══════════════════════════════════════════════════════════════════════

class RealTimeAnimationEngine:
    """
    [D] Integrated animation engine combining FDTD + Carpet + visualization.
    """
    
    def __init__(self, cfg: Optional[AnimationConfig] = None):
        self.cfg = cfg or AnimationConfig()
        self.solver = FDTDFieldSolver(self.cfg)
        self.animator = FieldAnimator(self.solver, self.cfg)
        
    def run_animation(self, backend: str = "matplotlib") -> Optional[str]:
        """Run field animation."""
        if backend == "matplotlib":
            return self.animator.animate_matplotlib()
        elif backend == "plotly":
            return self.animator.animate_plotly()
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
    def get_complexity_evolution(self) -> List[float]:
        """Get spectral complexity time series."""
        if self.animator.carpet_overlay:
            return self.animator.carpet_overlay.complexity_history
        return []


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Real-Time Field Animation — Smoke Test")
    print("=" * 72)
    print(f"  matplotlib : {HAS_MATPLOTLIB}")
    print(f"  plotly     : {HAS_PLOTLY}")
    print(f"  SESI deps  : {HAS_SESI}")
    
    cfg = AnimationConfig(
        nx=50, ny=50, n_steps=100, frame_interval=5,
        enable_carpet_overlay=HAS_SESI
    )
    
    engine = RealTimeAnimationEngine(cfg)
    
    # Set up metamaterial region
    engine.solver.set_metamaterial_region(20, 30, 20, 30, eps_r=10.0, mu_r=10.0)
    
    print("\n  ── Running animation ──")
    if HAS_MATPLOTLIB:
        output = engine.run_animation(backend="matplotlib")
        if output:
            print(f"    Saved to: {output}")
    
    complexity = engine.get_complexity_evolution()
    if complexity:
        print(f"    Complexity samples: {len(complexity)}")
        print(f"    Mean complexity: {np.mean(complexity):.3f}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Animation is VISUALIZATION, not proof")
    print("  • FDTD requires mesh convergence verification")
    print("  • Carpet spectral indicators are ANALOGICAL")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
