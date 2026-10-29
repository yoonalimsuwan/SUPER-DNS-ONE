"""
=============================================================================
SESI 3D ELECTROMAGNETIC FIELD VISUALIZATION
Interactive E/H field rendering using PyVista/Plotly
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] 3D visualization of electromagnetic field distributions:
  • E-field (electric field) magnitude and vector
  • H-field (magnetic field) magnitude and vector
  • S-parameter plots (frequency domain)
  • Far-field radiation patterns
  • Interactive rotation, slicing, animation

VISUALIZATION BACKENDS
-----------------------
  [C] PyVista (VTK-based): high-quality 3D rendering, interactive
  [C] Plotly 3D: web-based, shareable, less performance
  [D] Matplotlib 3D: fallback, basic 3D plots

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Visualization is INTERPRETIVE AID, not proof of field behavior.
• Field data must come from validated simulation (CST/OpenEMS).
• Color maps and scaling affect perception; choose carefully.
• Interactive features require appropriate backend (PyVista/Plotly).
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.em_viz")

# Try visualization backends
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
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__all__ = [
    "VisualizationConfig",
    "FieldData",
    "EMFieldVisualizer",
    "SParameterPlotter",
    "FarFieldPlotter",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VisualizationConfig:
    """Configuration for EM field visualization."""
    # Backend preference
    backend: str = "auto"  # auto, pyvista, plotly, matplotlib
    
    # Field visualization
    field_component: str = "magnitude"  # magnitude, x, y, z, vector
    colormap: str = "viridis"
    opacity: float = 0.8
    
    # Slicing
    enable_slicing: bool = True
    slice_normal: str = "z"  # x, y, z
    
    # Animation
    enable_animation: bool = False
    animation_fps: int = 30
    
    # Output
    output_dir: str = "./visualization"
    save_format: str = "png"  # png, jpg, svg, html
    
    def __post_init__(self):
        from pathlib import Path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# [D] FIELD DATA
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FieldData:
    """
    [D] Container for EM field data.
    
    Attributes:
      • x, y, z: coordinate arrays (1D or 3D grid)
      • Ex, Ey, Ez: electric field components (3D arrays)
      • Hx, Hy, Hz: magnetic field components (3D arrays)
      • frequency_ghz: operating frequency
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    Ex: Optional[np.ndarray] = None
    Ey: Optional[np.ndarray] = None
    Ez: Optional[np.ndarray] = None
    Hx: Optional[np.ndarray] = None
    Hy: Optional[np.ndarray] = None
    Hz: Optional[np.ndarray] = None
    frequency_ghz: float = 10.0
    
    def compute_magnitude(self, field_type: str = "E") -> np.ndarray:
        """Compute field magnitude."""
        if field_type == "E":
            if self.Ex is None or self.Ey is None or self.Ez is None:
                raise ValueError("E-field components not available")
            return np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2)
        else:
            if self.Hx is None or self.Hy is None or self.Hz is None:
                raise ValueError("H-field components not available")
            return np.sqrt(self.Hx**2 + self.Hy**2 + self.Hz**2)


# ═══════════════════════════════════════════════════════════════════════
# [D] EM FIELD VISUALIZER
# ═══════════════════════════════════════════════════════════════════════

class EMFieldVisualizer:
    """
    [D] 3D visualization of EM field distributions.
    
    Supports:
      • PyVista: high-quality interactive 3D
      • Plotly: web-based shareable plots
      • Matplotlib: basic 3D fallback
    """
    
    def __init__(self, cfg: VisualizationConfig):
        self.cfg = cfg
        self.backend = self._select_backend()
    
    def _select_backend(self) -> str:
        """Select visualization backend."""
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
    
    def visualize_field(self, field_data: FieldData,
                        field_type: str = "E",
                        component: str = "magnitude",
                        output_path: Optional[str] = None) -> Any:
        """
        Visualize EM field distribution.
        
        Args:
          • field_data: FieldData object
          • field_type: "E" or "H"
          • component: "magnitude", "x", "y", "z", or "vector"
          • output_path: save path (optional)
        
        Returns:
          • Visualization object (PyVista plotter, Plotly figure, etc.)
        """
        if self.backend == "pyvista":
            return self._visualize_pyvista(field_data, field_type, component, output_path)
        elif self.backend == "plotly":
            return self._visualize_plotly(field_data, field_type, component, output_path)
        else:
            return self._visualize_matplotlib(field_data, field_type, component, output_path)
    
    def _visualize_pyvista(self, field_data: FieldData, field_type: str,
                           component: str, output_path: Optional[str]) -> Any:
        """PyVista-based 3D visualization."""
        if not HAS_PYVISTA:
            raise RuntimeError("PyVista not available")
        
        # Create grid
        grid = pv.StructuredGrid(
            field_data.x, field_data.y, field_data.z
        )
        
        # Add field data
        if component == "magnitude":
            field_mag = field_data.compute_magnitude(field_type)
            grid[field_type + "_magnitude"] = field_mag.flatten()
            scalars = field_type + "_magnitude"
        else:
            # Component (x, y, z)
            field_comp = getattr(field_data, f"{field_type}{component.upper()}")
            grid[field_type + component] = field_comp.flatten()
            scalars = field_type + component
        
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            scalars=scalars,
            cmap=self.cfg.colormap,
            opacity=self.cfg.opacity,
            show_edges=False,
        )
        
        # Add slicing plane
        if self.cfg.enable_slicing:
            if self.cfg.slice_normal == "z":
                normal = (0, 0, 1)
                origin = (0, 0, 0)
            elif self.cfg.slice_normal == "y":
                normal = (0, 1, 0)
                origin = (0, 0, 0)
            else:
                normal = (1, 0, 0)
                origin = (0, 0, 0)
            
            plotter.add_mesh_slice(
                grid,
                normal=normal,
                origin=origin,
                cmap=self.cfg.colormap,
            )
        
        plotter.add_axes()
        plotter.add_title(f"{field_type}-field {component} at {field_data.frequency_ghz} GHz")
        
        if output_path:
            plotter.screenshot(output_path)
        
        return plotter
    
    def _visualize_plotly(self, field_data: FieldData, field_type: str,
                          component: str, output_path: Optional[str]) -> Any:
        """Plotly-based 3D visualization."""
        if not HAS_PLOTLY:
            raise RuntimeError("Plotly not available")
        
        # Compute field data
        if component == "magnitude":
            field_vals = field_data.compute_magnitude(field_type)
        else:
            field_vals = getattr(field_data, f"{field_type}{component.upper()}")
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=field_data.x.flatten(),
            y=field_data.y.flatten(),
            z=field_data.z.flatten(),
            mode='markers',
            marker=dict(
                size=2,
                color=field_vals.flatten(),
                colorscale=self.cfg.colormap,
                opacity=0.8,
                colorbar=dict(title=f"{field_type}-field {component}"),
            ),
        ))
        
        fig.update_layout(
            title=f"{field_type}-field {component} at {field_data.frequency_ghz} GHz",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
            ),
        )
        
        if output_path:
            if output_path.endswith(".html"):
                fig.write_html(output_path)
            else:
                fig.write_image(output_path)
        
        return fig
    
    def _visualize_matplotlib(self, field_data: FieldData, field_type: str,
                              component: str, output_path: Optional[str]) -> Any:
        """Matplotlib-based 3D visualization (fallback)."""
        if not HAS_MATPLOTLIB:
            raise RuntimeError("Matplotlib not available")
        
        # Compute field data
        if component == "magnitude":
            field_vals = field_data.compute_magnitude(field_type)
        else:
            field_vals = getattr(field_data, f"{field_type}{component.upper()}")
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for performance
        stride = max(1, len(field_vals.flatten()) // 1000)
        x_sub = field_data.x.flatten()[::stride]
        y_sub = field_data.y.flatten()[::stride]
        z_sub = field_data.z.flatten()[::stride]
        vals_sub = field_vals.flatten()[::stride]
        
        scatter = ax.scatter(
            x_sub, y_sub, z_sub,
            c=vals_sub,
            cmap=self.cfg.colormap,
            alpha=self.cfg.opacity,
            s=10,
        )
        
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{field_type}-field {component} at {field_data.frequency_ghz} GHz")
        
        plt.colorbar(scatter, label=f"{field_type}-field {component}")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig


# ═══════════════════════════════════════════════════════════════════════
# [D] S-PARAMETER PLOTTER
# ═══════════════════════════════════════════════════════════════════════

class SParameterPlotter:
    """
    [D] Plot S-parameters (frequency domain).
    """
    
    def __init__(self, cfg: VisualizationConfig):
        self.cfg = cfg
    
    def plot_s11(self, frequencies_ghz: np.ndarray, s11_db: np.ndarray,
                 output_path: Optional[str] = None) -> Any:
        """Plot S11 parameter (return loss)."""
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frequencies_ghz,
                y=s11_db,
                mode='lines',
                name='S11',
                line=dict(color='blue', width=2),
            ))
            fig.update_layout(
                title="S11 Parameter (Return Loss)",
                xaxis_title="Frequency (GHz)",
                yaxis_title="S11 (dB)",
                template="plotly_dark",
            )
            
            if output_path:
                if output_path.endswith(".html"):
                    fig.write_html(output_path)
                else:
                    fig.write_image(output_path)
            
            return fig
        
        elif HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(frequencies_ghz, s11_db, 'b-', linewidth=2, label='S11')
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("S11 (dB)")
            ax.set_title("S11 Parameter (Return Loss)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        else:
            raise RuntimeError("No plotting backend available")


# ═══════════════════════════════════════════════════════════════════════
# [D] FAR-FIELD PLOTTER
# ═══════════════════════════════════════════════════════════════════════

class FarFieldPlotter:
    """
    [D] Plot far-field radiation patterns.
    """
    
    def __init__(self, cfg: VisualizationConfig):
        self.cfg = cfg
    
    def plot_radiation_pattern(self, theta_deg: np.ndarray, phi_deg: np.ndarray,
                                gain_db: np.ndarray,
                                output_path: Optional[str] = None) -> Any:
        """Plot 3D radiation pattern."""
        if HAS_PLOTLY:
            # Convert to Cartesian for 3D plot
            theta_rad = np.radians(theta_deg)
            phi_rad = np.radians(phi_deg)
            
            r = 10 ** (gain_db / 20)  # linear scale
            
            x = r * np.sin(theta_rad) * np.cos(phi_rad)
            y = r * np.sin(theta_rad) * np.sin(phi_rad)
            z = r * np.cos(theta_rad)
            
            fig = go.Figure(data=go.Scatter3d(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                mode='markers',
                marker=dict(
                    size=2,
                    color=gain_db.flatten(),
                    colorscale='Viridis',
                    colorbar=dict(title="Gain (dB)"),
                ),
            ))
            
            fig.update_layout(
                title="Far-Field Radiation Pattern",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode='cube',
                ),
            )
            
            if output_path:
                if output_path.endswith(".html"):
                    fig.write_html(output_path)
                else:
                    fig.write_image(output_path)
            
            return fig
        
        else:
            raise RuntimeError("Plotly required for 3D radiation pattern")


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI 3D EM Field Visualization — Smoke Test")
    print("=" * 72)
    print(f"  PyVista available      : {HAS_PYVISTA}")
    print(f"  Plotly available       : {HAS_PLOTLY}")
    print(f"  Matplotlib available   : {HAS_MATPLOTLIB}")
    
    cfg = VisualizationConfig(backend="auto")
    visualizer = EMFieldVisualizer(cfg)
    print(f"  Selected backend       : {visualizer.backend}")
    
    # Create synthetic field data
    print("\n  ── Creating synthetic field data ──")
    x = np.linspace(-50, 50, 20)
    y = np.linspace(-50, 50, 20)
    z = np.linspace(-5, 5, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Synthetic E-field (dipole-like)
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6
    Ex = X / r**2
    Ey = Y / r**2
    Ez = Z / r**2
    
    field_data = FieldData(
        x=X, y=Y, z=Z,
        Ex=Ex, Ey=Ey, Ez=Ez,
        frequency_ghz=10.0,
    )
    
    print(f"    Grid size            : {X.shape}")
    print(f"    E-field magnitude    : [{field_data.compute_magnitude('E').min():.3e}, "
          f"{field_data.compute_magnitude('E').max():.3e}]")
    
    # Visualize
    print("\n  ── Visualizing E-field ──")
    try:
        output_path = f"{cfg.output_dir}/e_field_magnitude.png"
        viz = visualizer.visualize_field(
            field_data,
            field_type="E",
            component="magnitude",
            output_path=output_path,
        )
        print(f"    Visualization created: {output_path}")
    except Exception as e:
        print(f"    [ERROR] Visualization failed: {e}")
    
    # S-parameter plot
    print("\n  ── S-parameter plot ──")
    s_plotter = SParameterPlotter(cfg)
    freqs = np.linspace(8, 12, 100)
    s11 = -20 + 15 * np.exp(-((freqs - 10) / 0.5)**2)
    
    try:
        output_path = f"{cfg.output_dir}/s11_parameter.png"
        fig = s_plotter.plot_s11(freqs, s11, output_path)
        print(f"    S11 plot created     : {output_path}")
    except Exception as e:
        print(f"    [ERROR] S-parameter plot failed: {e}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Visualization is INTERPRETIVE AID, not proof")
    print("  • Field data must come from validated simulation")
    print("  • Color maps affect perception; choose carefully")
    print("  • Interactive features require appropriate backend")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
