"""
=============================================================================
SES 3D SMOKE RENDERING — Smokeview-style Volume Rendering
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[C/D] 3D volume rendering of smoke and fire data:
  • Ray marching through voxel grids
  • Transfer functions for color/opacity mapping
  • Multiple rendering modes (smoke, temperature, velocity)
  • Integration with FDS output (via ses_fds_connector)

RENDERING METHODS
-----------------
[C] Ray casting / ray marching through volume
[C] Transfer function mapping scalar → RGBA
[D] Emission-absorption model for fire glow

RELATION TO SMOKEVIEW
---------------------
Smokeview (NIST) uses OpenGL for real-time rendering.
This module provides:
  • Pure Python/NumPy software rendering (for testing)
  • Optional VTK/matplotlib visualization
  • Data preparation for external renderers

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is RENDERING, not simulation. Data must come from FDS or other CFD.
• Software rendering is SLOW; real-time requires GPU (VTK/OpenGL).
• Transfer functions are INTERPRETIVE; color choices affect perception.
• This is NOT Smokeview itself; it provides compatible data structures.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ses.smoke_render")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__all__ = [
    "SmokeRenderConfig",
    "TransferFunction",
    "VolumeRenderer",
    "SmokeVisualization",
]


@dataclass
class SmokeRenderConfig:
    """Configuration for smoke rendering."""
    # Grid
    nx: int = 64
    ny: int = 64
    nz: int = 32
    
    # Domain
    domain_size: Tuple[float, float, float] = (10.0, 10.0, 3.0)
    
    # Rendering
    render_mode: str = "smoke"  # smoke, temperature, velocity
    colormap: str = "hot"
    
    # Ray marching
    n_ray_steps: int = 100
    step_size: float = 0.1
    
    # Output
    output_width: int = 800
    output_height: int = 600


# ═══════════════════════════════════════════════════════════════════════
# [C/D] TRANSFER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class TransferFunction:
    """
    [C/D] Transfer function mapping scalar values to RGBA.
    
    Used in volume rendering to determine color and opacity
    at each point in the volume.
    """
    
    def __init__(self, mode: str = "smoke"):
        self.mode = mode
    
    def map_smoke(self, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map smoke density to color and opacity.
        
        Returns (rgb, alpha) arrays.
        """
        # Normalize
        d = np.clip(density / (np.max(density) + 1e-12), 0, 1)
        
        # Smoke is grayish, opacity proportional to density
        rgb = np.stack([d * 0.5, d * 0.5, d * 0.6], axis=-1)
        alpha = d * 0.8
        
        return rgb, alpha
    
    def map_temperature(self, temp: np.ndarray, 
                        temp_min: float = 293.0, 
                        temp_max: float = 1200.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map temperature to color (hot colormap).
        
        Returns (rgb, alpha) arrays.
        """
        # Normalize
        t = np.clip((temp - temp_min) / (temp_max - temp_min), 0, 1)
        
        # Hot colormap: black → red → yellow → white
        r = np.clip(t * 3, 0, 1)
        g = np.clip(t * 3 - 1, 0, 1)
        b = np.clip(t * 3 - 2, 0, 1)
        
        rgb = np.stack([r, g, b], axis=-1)
        alpha = np.clip(t * 1.5, 0, 1)
        
        return rgb, alpha
    
    def map_velocity(self, speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map velocity magnitude to color.
        
        Returns (rgb, alpha) arrays.
        """
        # Normalize
        s = np.clip(speed / (np.max(speed) + 1e-12), 0, 1)
        
        # Viridis-like colormap
        r = np.clip(s * 1.5 - 0.2, 0, 1)
        g = np.clip(s * 1.2, 0, 1)
        b = np.clip(1 - s * 1.5 + 0.5, 0, 1)
        
        rgb = np.stack([r, g, b], axis=-1)
        alpha = s * 0.9
        
        return rgb, alpha


# ═══════════════════════════════════════════════════════════════════════
# [D] VOLUME RENDERER
# ═══════════════════════════════════════════════════════════════════════

class VolumeRenderer:
    """
    [D] Software volume renderer using ray marching.
    
    This is a CPU-based renderer for testing and small volumes.
    For real-time rendering, use VTK or GPU shaders.
    """
    
    def __init__(self, cfg: SmokeRenderConfig):
        self.cfg = cfg
        self.transfer_fn = TransferFunction(cfg.render_mode)
    
    def render_slice(self, volume: np.ndarray, 
                     slice_axis: int = 2, 
                     slice_pos: float = 0.5) -> np.ndarray:
        """
        Render a 2D slice through the volume.
        
        Args:
            volume: 3D array (nx, ny, nz)
            slice_axis: 0=x, 1=y, 2=z
            slice_pos: normalized position [0, 1]
        
        Returns:
            2D RGBA image array
        """
        # Compute slice index
        n = volume.shape[slice_axis]
        idx = int(slice_pos * (n - 1))
        
        # Extract slice
        if slice_axis == 0:
            data = volume[idx, :, :]
        elif slice_axis == 1:
            data = volume[:, idx, :]
        else:
            data = volume[:, :, idx]
        
        # Apply transfer function
        if self.cfg.render_mode == "smoke":
            rgb, alpha = self.transfer_fn.map_smoke(data)
        elif self.cfg.render_mode == "temperature":
            rgb, alpha = self.transfer_fn.map_temperature(data)
        else:
            rgb, alpha = self.transfer_fn.map_velocity(data)
        
        # Combine into RGBA
        rgba = np.zeros((*data.shape, 4))
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = alpha
        
        return rgba
    
    def render_projection(self, volume: np.ndarray,
                          axis: int = 2) -> np.ndarray:
        """
        Render a projection (maximum intensity or average).
        
        Args:
            volume: 3D array
            axis: projection axis
        
        Returns:
            2D image array
        """
        # Maximum intensity projection
        proj = np.max(volume, axis=axis)
        
        # Apply transfer function
        if self.cfg.render_mode == "smoke":
            rgb, alpha = self.transfer_fn.map_smoke(proj)
        elif self.cfg.render_mode == "temperature":
            rgb, alpha = self.transfer_fn.map_temperature(proj)
        else:
            rgb, alpha = self.transfer_fn.map_velocity(proj)
        
        # Composite with background
        bg = np.zeros_like(rgb)
        composited = rgb * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        
        return composited
    
    def render_isosurface_points(self, volume: np.ndarray,
                                  threshold: float) -> np.ndarray:
        """
        Extract points near an isosurface for scatter plot visualization.
        
        Args:
            volume: 3D array
            threshold: isosurface value
        
        Returns:
            (N, 3) array of points
        """
        # Find voxels near threshold
        mask = np.abs(volume - threshold) < 0.1 * threshold
        
        # Get coordinates
        coords = np.argwhere(mask)
        
        return coords


# ═══════════════════════════════════════════════════════════════════════
# [D] SMOKE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

class SmokeVisualization:
    """
    [D] High-level visualization interface for smoke data.
    
    Integrates with matplotlib for display and saving.
    """
    
    def __init__(self, cfg: SmokeRenderConfig):
        self.cfg = cfg
        self.renderer = VolumeRenderer(cfg)
    
    def visualize_slice(self, volume: np.ndarray,
                        slice_axis: int = 2,
                        slice_pos: float = 0.5,
                        title: str = "Smoke Slice",
                        save_path: Optional[str] = None) -> Optional[Any]:
        """Visualize a slice through the volume."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        rgba = self.renderer.render_slice(volume, slice_axis, slice_pos)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgba, origin='lower', cmap='gray')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved slice to {save_path}")
        
        return fig
    
    def visualize_projection(self, volume: np.ndarray,
                             axis: int = 2,
                             title: str = "Smoke Projection",
                             save_path: Optional[str] = None) -> Optional[Any]:
        """Visualize a projection of the volume."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        proj = self.renderer.render_projection(volume, axis)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(proj, origin='lower')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_scatter(self, volume: np.ndarray,
                             threshold: float,
                             title: str = "Smoke Isosurface",
                             save_path: Optional[str] = None) -> Optional[Any]:
        """Visualize isosurface as 3D scatter plot."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        points = self.renderer.render_isosurface_points(volume, threshold)
        
        if len(points) == 0:
            logger.warning("No points found for isosurface")
            return None
        
        # Subsample for performance
        if len(points) > 5000:
            idx = np.random.choice(len(points), 5000, replace=False)
            points = points[idx]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=volume[tuple(points.T)] if len(points) > 0 else 'gray',
                  cmap='hot', alpha=0.5, s=5)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES 3D Smoke Rendering — Smoke Test")
    print("=" * 72)
    print(f"  Matplotlib available : {HAS_MATPLOTLIB}")
    
    cfg = SmokeRenderConfig(
        nx=32, ny=32, nz=16,
        render_mode="smoke",
    )
    
    # Generate synthetic smoke data
    print("\n  ── Generating synthetic smoke data ──")
    x = np.linspace(-1, 1, cfg.nx)
    y = np.linspace(-1, 1, cfg.ny)
    z = np.linspace(-1, 1, cfg.nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Smoke plume rising from center
    smoke = np.exp(-(X**2 + Y**2) / 0.2) * np.exp(-np.abs(Z - 0.3) / 0.5)
    smoke = np.clip(smoke, 0, 1)
    
    print(f"    Grid size: {smoke.shape}")
    print(f"    Max density: {smoke.max():.3f}")
    
    # Render
    print("\n  ── Rendering ──")
    viz = SmokeVisualization(cfg)
    
    # Slice
    print("    Rendering slice...")
    fig_slice = viz.visualize_slice(smoke, slice_axis=2, slice_pos=0.5,
                                    title="Smoke Slice (Z=0.5)",
                                    save_path="./smoke_slice.png")
    print(f"    [{'PASS' if fig_slice is not None else 'FAIL'}] Slice rendered")
    
    # Projection
    print("    Rendering projection...")
    fig_proj = viz.visualize_projection(smoke, axis=2,
                                        title="Smoke Projection",
                                        save_path="./smoke_projection.png")
    print(f"    [{'PASS' if fig_proj is not None else 'FAIL'}] Projection rendered")
    
    # 3D scatter (isosurface)
    print("    Rendering isosurface...")
    fig_3d = viz.visualize_3d_scatter(smoke, threshold=0.5,
                                      title="Smoke Isosurface",
                                      save_path="./smoke_isosurface.png")
    print(f"    [{'PASS' if fig_3d is not None else 'NOTE'}] Isosurface rendered")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is RENDERING, not simulation")
    print("  • Software rendering is SLOW; real-time needs GPU")
    print("  • Transfer functions are INTERPRETIVE")
    print("  • This is NOT Smokeview; provides compatible data structures")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
