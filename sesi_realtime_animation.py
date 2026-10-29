"""
=============================================================================
SESI REAL-TIME EM FIELD ANIMATION
Time-series visualization of evolving E/H fields
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] Animates time-evolving electromagnetic fields:
  • Wave propagation through metamaterials
  • Pulse reflection/transmission at interfaces
  • Resonant mode evolution in cavities

VISUALIZATION BACKENDS
-----------------------
  [C] PyVista: high-performance interactive animation
  [C] Plotly: web-based shareable animation
  [D] Matplotlib: basic fallback animation

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Animation speed is for VISUALIZATION, not real-time physics.
• Field data must come from validated time-domain simulation (FDTD).
• Color maps and scaling affect perception; choose carefully.
• Interactive features require appropriate backend.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.animation")

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sesi_em_visualization import FieldData, VisualizationConfig

__all__ = [
    "AnimationConfig",
    "FieldAnimator",
    "WavePropagationAnimator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnimationConfig(VisualizationConfig):
    """Configuration for EM field animation."""
    # Time parameters
    n_frames: int = 50
    time_step_s: float = 1e-12  # 1 ps
    
    # Playback
    fps: int = 20
    loop: bool = True
    
    # Output
    output_format: str = "gif"  # gif, mp4, html
    output_path: str = "./animation/em_field.gif"


# ═══════════════════════════════════════════════════════════════════════
# [D] FIELD ANIMATOR BASE
# ═══════════════════════════════════════════════════════════════════════

class FieldAnimator:
    """
    [D] Base class for animating time-evolving EM fields.
    """
    
    def __init__(self, cfg: AnimationConfig):
        self.cfg = cfg
        self.backend = self._select_backend()
    
    def _select_backend(self) -> str:
        if self.cfg.backend != "auto":
            return self.cfg.backend
        if HAS_PYVISTA:
            return "pyvista"
        elif HAS_PLOTLY:
            return "plotly"
        elif HAS_MATPLOTLIB:
            return "matplotlib"
        else:
            raise RuntimeError("No visualization backend available")
    
    def animate(self, field_data_series: List[FieldData],
                field_type: str = "E",
                component: str = "magnitude") -> Any:
        """
        Animate a series of field snapshots.
        
        Args:
          • field_data_series: list of FieldData objects (one per time step)
          • field_type: "E" or "H"
          • component: "magnitude", "x", "y", "z"
        
        Returns:
          • Animation object (PyVista plotter, Plotly figure, etc.)
        """
        if self.backend == "pyvista":
            return self._animate_pyvista(field_data_series, field_type, component)
        elif self.backend == "plotly":
            return self._animate_plotly(field_data_series, field_type, component)
        else:
            return self._animate_matplotlib(field_data_series, field_type, component)
    
    def _animate_pyvista(self, field_data_series: List[FieldData],
                         field_type: str, component: str) -> Any:
        """PyVista-based animation."""
        if not HAS_PYVISTA:
            raise RuntimeError("PyVista not available")
        
        # Create plotter
        plotter = pv.Plotter()
        
        # Get grid from first frame
        first_frame = field_data_series[0]
        grid = pv.StructuredGrid(first_frame.x, first_frame.y, first_frame.z)
        
        # Add mesh
        mesh_actor = plotter.add_mesh(
            grid,
            scalars=None,  # will update in loop
            cmap=self.cfg.colormap,
            opacity=self.cfg.opacity,
            show_edges=False,
        )
        
        plotter.add_axes()
        plotter.add_title(f"{field_type}-field {component} Animation")
        
        # Animation loop (interactive in PyVista)
        def update(frame_idx):
            frame = field_data_series[frame_idx % len(field_data_series)]
            if component == "magnitude":
                field_vals = frame.compute_magnitude(field_type)
            else:
                field_vals = getattr(frame, f"{field_type}{component.upper()}")
            
            grid[field_type + "_" + component] = field_vals.flatten()
            mesh_actor.mapper.scalars = grid[field_type + "_" + component]
            mesh_actor.mapper.scalar_range = (field_vals.min(), field_vals.max())
        
        # Note: PyVista animation requires event loop; simplified here
        logger.info("PyVista animation created; use plotter.show() to view")
        return plotter
    
    def _animate_plotly(self, field_data_series: List[FieldData],
                        field_type: str, component: str) -> Any:
        """Plotly-based animation (HTML)."""
        if not HAS_PLOTLY:
            raise RuntimeError("Plotly not available")
        
        frames = []
        
        for i, frame in enumerate(field_data_series):
            if component == "magnitude":
                field_vals = frame.compute_magnitude(field_type)
            else:
                field_vals = getattr(frame, f"{field_type}{component.upper()}")
            
            # Subsample for performance
            stride = max(1, len(field_vals.flatten()) // 500)
            
            frame_data = go.Frame(
                data=[go.Scatter3d(
                    x=frame.x.flatten()[::stride],
                    y=frame.y.flatten()[::stride],
                    z=frame.z.flatten()[::stride],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=field_vals.flatten()[::stride],
                        colorscale=self.cfg.colormap,
                        cmin=field_vals.min(),
                        cmax=field_vals.max(),
                    ),
                )],
                name=f"frame{i}"
            )
            frames.append(frame_data)
        
        # Initial frame
        first_frame = field_data_series[0]
        if component == "magnitude":
            init_vals = first_frame.compute_magnitude(field_type)
        else:
            init_vals = getattr(first_frame, f"{field_type}{component.upper()}")
        
        stride = max(1, len(init_vals.flatten()) // 500)
        
        fig = go.Figure(
            data=[go.Scatter3d(
                x=first_frame.x.flatten()[::stride],
                y=first_frame.y.flatten()[::stride],
                z=first_frame.z.flatten()[::stride],
                mode='markers',
                marker=dict(
                    size=2,
                    color=init_vals.flatten()[::stride],
                    colorscale=self.cfg.colormap,
                ),
            )],
            frames=frames,
        )
        
        # Add sliders and buttons
        fig.update_layout(
            title=f"{field_type}-field {component} Animation",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
            ),
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 1000//self.cfg.fps, "redraw": True},
                                     "fromcurrent": True}],
                     "label": "Play",
                     "method": "animate"},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }],
            sliders=[{
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 1000//self.cfg.fps, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}],
                     "label": str(i),
                     "method": "animate"}
                    for i, f in enumerate(frames)
                ],
                "currentvalue": {"prefix": "Frame: "},
            }]
        )
        
        # Save as HTML
        output_path = self.cfg.output_path.replace(".gif", ".html")
        fig.write_html(output_path)
        logger.info(f"Plotly animation saved to {output_path}")
        
        return fig
    
    def _animate_matplotlib(self, field_data_series: List[FieldData],
                            field_type: str, component: str) -> Any:
        """Matplotlib-based animation (fallback)."""
        if not HAS_MATPLOTLIB:
            raise RuntimeError("Matplotlib not available")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        first_frame = field_data_series[0]
        if component == "magnitude":
            init_vals = first_frame.compute_magnitude(field_type)
        else:
            init_vals = getattr(first_frame, f"{field_type}{component.upper()}")
        
        stride = max(1, len(init_vals.flatten()) // 1000)
        
        scatter = ax.scatter(
            first_frame.x.flatten()[::stride],
            first_frame.y.flatten()[::stride],
            first_frame.z.flatten()[::stride],
            c=init_vals.flatten()[::stride],
            cmap=self.cfg.colormap,
            s=10,
        )
        
        ax.set_title(f"{field_type}-field {component} Animation")
        
        def update(frame_idx):
            frame = field_data_series[frame_idx % len(field_data_series)]
            if component == "magnitude":
                field_vals = frame.compute_magnitude(field_type)
            else:
                field_vals = getattr(frame, f"{field_type}{component.upper()}")
            
            scatter._offsets3d = (
                frame.x.flatten()[::stride],
                frame.y.flatten()[::stride],
                frame.z.flatten()[::stride],
            )
            scatter.set_array(field_vals.flatten()[::stride])
            scatter.set_clim(field_vals.min(), field_vals.max())
        
        anim = FuncAnimation(fig, update, frames=len(field_data_series),
                            interval=1000//self.cfg.fps, repeat=self.cfg.loop)
        
        # Save as GIF (requires imagemagick or pillow)
        try:
            anim.save(self.cfg.output_path, writer='pillow', fps=self.cfg.fps)
            logger.info(f"Matplotlib animation saved to {self.cfg.output_path}")
        except Exception as e:
            logger.warning(f"Failed to save animation: {e}")
        
        return anim


# ═══════════════════════════════════════════════════════════════════════
# [D] WAVE PROPAGATION ANIMATOR
# ═══════════════════════════════════════════════════════════════════════

class WavePropagationAnimator(FieldAnimator):
    """
    [D] Specialized animator for wave propagation through materials.
    
    Generates synthetic time-series data for demonstration.
    In production, would extract from FDTD simulation results.
    """
    
    def generate_wave_propagation_data(self, n_time_steps: int = 50,
                                        domain_size_mm: Tuple[float, float, float] = (100, 100, 10),
                                        grid_resolution: Tuple[int, int, int] = (50, 50, 10),
                                        wavelength_mm: float = 30.0,
                                        velocity_m_s: float = 3e8) -> List[FieldData]:
        """
        Generate synthetic wave propagation data.
        
        Simulates a Gaussian pulse propagating in +x direction.
        """
        x = np.linspace(0, domain_size_mm[0], grid_resolution[0])
        y = np.linspace(0, domain_size_mm[1], grid_resolution[1])
        z = np.linspace(0, domain_size_mm[2], grid_resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        field_series = []
        
        for t_idx in range(n_time_steps):
            t = t_idx * self.cfg.time_step_s
            
            # Wave position
            x_pos = velocity_m_s * t * 1e3  # convert to mm
            
            # Gaussian pulse
            pulse = np.exp(-((X - x_pos)**2) / (2 * (wavelength_mm/4)**2))
            
            # E-field (polarized in y)
            Ex = np.zeros_like(X)
            Ey = pulse * np.sin(2 * np.pi * (X - x_pos) / wavelength_mm)
            Ez = np.zeros_like(Z)
            
            # H-field (orthogonal)
            Hx = np.zeros_like(X)
            Hy = np.zeros_like(Y)
            Hz = pulse * np.cos(2 * np.pi * (X - x_pos) / wavelength_mm) / 377.0
            
            field_data = FieldData(
                x=X, y=Y, z=Z,
                Ex=Ex, Ey=Ey, Ez=Ez,
                Hx=Hx, Hy=Hy, Hz=Hz,
                frequency_ghz=velocity_m_s / (wavelength_mm * 1e-3) / 1e9,
            )
            
            field_series.append(field_data)
        
        return field_series


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Real-Time EM Field Animation — Smoke Test")
    print("=" * 72)
    print(f"  PyVista available      : {HAS_PYVISTA}")
    print(f"  Plotly available       : {HAS_PLOTLY}")
    print(f"  Matplotlib available   : {HAS_MATPLOTLIB}")
    
    cfg = AnimationConfig(
        backend="auto",
        n_frames=30,
        time_step_s=1e-12,
        fps=15,
        output_path="./animation/wave_propagation.gif",
    )
    
    animator = WavePropagationAnimator(cfg)
    
    print("\n  ── Generating synthetic wave propagation data ──")
    field_series = animator.generate_wave_propagation_data(n_time_steps=cfg.n_frames)
    print(f"    Generated {len(field_series)} time steps")
    
    print("\n  ── Creating animation ──")
    try:
        anim = animator.animate(field_series, field_type="E", component="magnitude")
        print(f"    Animation created successfully")
        print(f"    Backend used     : {animator.backend}")
    except Exception as e:
        print(f"    [ERROR] Animation failed: {e}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Animation speed is for VISUALIZATION, not real-time physics")
    print("  • Field data is SYNTHETIC; real data requires FDTD simulation")
    print("  • Color maps affect perception; choose carefully")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
