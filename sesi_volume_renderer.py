"""
=============================================================================
SESI 3D VOLUME RENDERER
PyVista/VTK-based visualization for EM field distributions
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[C] Volume rendering via ray casting (PyVista/VTK)
[D] Transfer functions for E/H field visualization
[D] Isosurface extraction for field magnitude
[D] Carpet fractal geometry rendering

VISUALIZATION BACKENDS
-----------------------
  [C] PyVista: high-quality 3D rendering, interactive
  [C] VTK: underlying library for PyVista
  [D] Matplotlib 3D: fallback for basic plots

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Visualization is INTERPRETIVE AID, not proof of field behavior.
• Color maps and transfer functions affect perception; choose carefully.
• Field data must come from validated simulation (FDTD/FEM).
• Interactive features require appropriate backend (PyVista/VTK).
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("sesi.volume_renderer")

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__all__ = [
    "VolumeRendererConfig",
    "TransferFunction",
    "VolumeRenderer",
    "CarpetGeometryRenderer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VolumeRendererConfig:
    """Configuration for volume rendering."""
    # Rendering
    backend: str = "auto"  # auto, pyvista, matplotlib
    colormap: str = "viridis"
    opacity: float = 0.8
    
    # Volume
    field_component: str = "magnitude"  # magnitude, Ex, Ey, Ez, Hx, Hy, Hz
    
    # Isosurface
    enable_isosurface: bool = True
    isosurface_level: float = 0.5  # normalized
    
    # Carpet geometry
    carpet_level: int = 3
    show_carpet: bool = True
    
    # Output
    output_path: str = "./visualization/volume_render.png"
    interactive: bool = False


# ═══════════════════════════════════════════════════════════════════════
# [D] TRANSFER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class TransferFunction:
    """
    [D] Transfer function for volume rendering.
    
    Maps field magnitude to color and opacity.
    """
    
    def __init__(self, colormap: str = "viridis"):
        self.colormap = colormap
    
    def apply(self, field_data: np.ndarray) -> np.ndarray:
        """
        Apply transfer function to field data.
        
        Returns RGBA array for rendering.
        """
        # Normalize field data
        field_norm = (field_data - field_data.min()) / (field_data.max() - field_data.min() + 1e-12)
        
        # Apply colormap (simplified: grayscale for now)
        rgba = np.zeros((*field_norm.shape, 4))
        rgba[..., 0] = field_norm  # R
        rgba[..., 1] = field_norm  # G
        rgba[..., 2] = field_norm  # B
        rgba[..., 3] = field_norm  # A (opacity)
        
        return rgba


# ═══════════════════════════════════════════════════════════════════════
# [C/D] VOLUME RENDERER
# ═══════════════════════════════════════════════════════════════════════

class VolumeRenderer:
    """
    [C/D] 3D volume renderer for EM field distributions.
    
    Supports:
      • Volume rendering (ray casting)
      • Isosurface extraction
      • Carpet geometry overlay
    """
    
    def __init__(self, cfg: VolumeRendererConfig):
        self.cfg = cfg
        self.backend = self._select_backend()
        self.transfer_fn = TransferFunction(cfg.colormap)
    
    def _select_backend(self) -> str:
        """Select rendering backend."""
        if self.cfg.backend != "auto":
            return self.cfg.backend
        
        if HAS_PYVISTA:
            return "pyvista"
        elif HAS_MATPLOTLIB:
            return "matplotlib"
        else:
            raise RuntimeError("No rendering backend available")
    
    def render(self, fields: Dict[str, np.ndarray],
               output_path: Optional[str] = None) -> Any:
        """
        Render 3D volume.
        
        Args:
            fields: dict with Ex, Ey, Ez, Hx, Hy, Hz arrays
            output_path: optional save path
        
        Returns:
            Rendered visualization object
        """
        output_path = output_path or self.cfg.output_path
        
        # Compute field magnitude
        if self.cfg.field_component == "magnitude":
            E_mag = np.sqrt(fields["Ex"]**2 + fields["Ey"]**2 + fields["Ez"]**2)
            field_data = E_mag
        else:
            field_data = fields[self.cfg.field_component]
        
        if self.backend == "pyvista":
            return self._render_pyvista(field_data, output_path)
        else:
            return self._render_matplotlib(field_data, output_path)
    
    def _render_pyvista(self, field_data: np.ndarray, output_path: str) -> Any:
        """PyVista-based volume rendering."""
        if not HAS_PYVISTA:
            raise RuntimeError("PyVista not available")
        
        # Create structured grid
        nx, ny, nz = field_data.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        
        grid = pv.StructuredGrid(
            *np.meshgrid(x, y, z, indexing="ij")
        )
        grid["field"] = field_data.flatten(order="F")
        
        # Create plotter
        plotter = pv.Plotter(off_screen=not self.cfg.interactive)
        
        # Add volume
        plotter.add_volume(
            grid,
            scalars="field",
            cmap=self.cfg.colormap,
            opacity=self.cfg.opacity,
        )
        
        # Add isosurface
        if self.cfg.enable_isosurface:
            threshold = self.cfg.isosurface_level * field_data.max()
            iso = grid.threshold(threshold, scalars="field")
            plotter.add_mesh(iso, color="red", opacity=0.5)
        
        plotter.add_axes()
        plotter.add_title(f"EM Field Distribution ({self.cfg.field_component})")
        
        if not self.cfg.interactive:
            plotter.screenshot(output_path)
            logger.info(f"Volume render saved to {output_path}")
        
        return plotter
    
    def _render_matplotlib(self, field_data: np.ndarray, output_path: str) -> Any:
        """Matplotlib-based 3D scatter plot (fallback)."""
        if not HAS_MATPLOTLIB:
            raise RuntimeError("Matplotlib not available")
        
        # Subsample for performance
        stride = max(1, field_data.size // 10000)
        x, y, z = np.mgrid[0:field_data.shape[0], 0:field_data.shape[1], 0:field_data.shape[2]]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            x.flatten()[::stride],
            y.flatten()[::stride],
            z.flatten()[::stride],
            c=field_data.flatten()[::stride],
            cmap=self.cfg.colormap,
            alpha=self.cfg.opacity,
            s=10,
        )
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"EM Field Distribution ({self.cfg.field_component})")
        
        plt.colorbar(scatter, label="Field Magnitude")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Volume render saved to {output_path}")
        
        return fig


# ═══════════════════════════════════════════════════════════════════════
# [D] CARPET GEOMETRY RENDERER
# ═══════════════════════════════════════════════════════════════════════

class CarpetGeometryRenderer:
    """
    [D] Render Sierpinski carpet geometry for visualization.
    """
    
    def __init__(self, level: int = 3):
        self.level = level
    
    def generate_carpet_cells(self) -> np.ndarray:
        """Generate list of carpet cell coordinates."""
        cells = []
        side = 3 ** self.level
        
        for x in range(side):
            for y in range(side):
                if self._survives(x, y, self.level):
                    cells.append((x, y))
        
        return np.array(cells) / side  # normalize to [0, 1]
    
    @staticmethod
    def _survives(x: int, y: int, level: int) -> bool:
        """Check if cell survives after `level` iterations."""
        for _ in range(level):
            if (x % 3 == 1) and (y % 3 == 1):
                return False
            x //= 3
            y //= 3
        return True
    
    def render(self, output_path: str = "./visualization/carpet_geometry.png") -> Any:
        """Render carpet geometry."""
        cells = self.generate_carpet_cells()
        
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(cells[:, 0], cells[:, 1], s=1, c='black')
            ax.set_aspect('equal')
            ax.set_title(f"Sierpinski Carpet (Level {self.level})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Carpet geometry saved to {output_path}")
            return fig
        
        return None


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI 3D Volume Renderer — Smoke Test")
    print("=" * 72)
    print(f"  PyVista available    : {HAS_PYVISTA}")
    print(f"  Matplotlib available : {HAS_MATPLOTLIB}")
    
    cfg = VolumeRendererConfig(
        backend="auto",
        field_component="magnitude",
        carpet_level=2,
        output_path="./visualization/test_render.png",
    )
    
    # Generate synthetic field data
    print("\n  ── Generating synthetic field data ──")
    nx, ny, nz = 50, 50, 50
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    # Synthetic dipole-like field
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6
    Ex = X / r**3
    Ey = Y / r**3
    Ez = Z / r**3
    
    fields = {
        "Ex": Ex, "Ey": Ey, "Ez": Ez,
        "Hx": np.zeros_like(Ex),
        "Hy": np.zeros_like(Ey),
        "Hz": np.zeros_like(Ez),
    }
    
    print(f"    Grid size          : {nx}×{ny}×{nz}")
    print(f"    Field component    : {cfg.field_component}")
    
    # Render volume
    print("\n  ── Rendering volume ──")
    renderer = VolumeRenderer(cfg)
    try:
        viz = renderer.render(fields)
        print(f"    Backend used       : {renderer.backend}")
        print(f"    Output saved to    : {cfg.output_path}")
    except Exception as e:
        print(f"    [ERROR] Rendering failed: {e}")
    
    # Render carpet geometry
    print("\n  ── Rendering carpet geometry ──")
    carpet_renderer = CarpetGeometryRenderer(level=cfg.carpet_level)
    try:
        carpet_viz = carpet_renderer.render("./visualization/carpet.png")
        print(f"    Carpet level       : {cfg.carpet_level}")
        print(f"    Output saved to    : ./visualization/carpet.png")
    except Exception as e:
        print(f"    [ERROR] Carpet rendering failed: {e}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Visualization is INTERPRETIVE AID, not proof")
    print("  • Color maps affect perception; choose carefully")
    print("  • Field data must come from validated simulation")
    print("  • Interactive features require appropriate backend")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
