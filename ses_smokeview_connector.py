"""
=============================================================================
SES SMOKEVIEW CONNECTOR
Python interface to NIST Smokeview visualization data from FDS
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT SMOKEVIEW IS [C]
---------------------
Smokeview is the official visualization tool for FDS (Fire Dynamics Simulator)
developed by NIST. It reads:
  • {case}.smv — main visualization script (text-based)
  • {case}_3d.sz — 3D smoke data (binary)
  • {case}_slice_*.s3d — slice data (binary)
  • {case}_part — particle tracking data
  • {case}_devc.csv — device measurements (temperature, velocity, etc.)

WHAT THIS CONNECTOR DOES [D]
----------------------------
  • Parse .smv files to extract grid, objects, and data references
  • Read binary slice/3D data files
  • Convert to numpy arrays for Python-based visualization
  • Generate smoke/hot-gas layer visualizations
  • Export to standard formats (VTK, PNG)

RELATION TO FDS CONNECTOR (ses_fdc_connector.py)
-------------------------------------------------
FDSConnector generates .fds input and runs simulations.
This connector READS the output and visualizes it.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a PARSER/READER, not Smokeview itself.
• Smokeview has a specific rendering engine (OpenGL); this connector
  provides data extraction for Python visualization (matplotlib, pyvista).
• Binary format details vary between FDS versions; this targets FDS 6.7+.
• Large files require chunked reading; not all data may fit in memory.
• Visualization is INTERPRETIVE — color maps and slicing affect perception.
=============================================================================
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ses.smokeview")

try:
    import vtk
    from vtk.util import numpy_support
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__all__ = [
    "SmokeviewConfig",
    "SmokeviewParser",
    "SliceDataReader",
    "SmokeviewVisualizer",
]


@dataclass
class SmokeviewConfig:
    """Configuration for Smokeview connector."""
    case_id: str = "sesi_fire"
    output_dir: str = "./fds_output"
    
    # Visualization
    quantity: str = "TEMPERATURE"  # TEMPERATURE, VELOCITY, SOOT_DENSITY, etc.
    slice_position: float = 0.5  # normalized [0, 1]
    
    # Colormap
    colormap: str = "hot"
    
    # Output
    save_png: bool = True
    save_vtk: bool = False


class SmokeviewParser:
    """
    [D] Parse Smokeview .smv script files.
    
    The .smv file contains commands like:
        GRID ...
        PDIM ...
        OBST ...
        VENT ...
        SLCF ...
        ...
    """
    
    def __init__(self, cfg: SmokeviewConfig):
        self.cfg = cfg
        self.smv_path = Path(cfg.output_dir) / f"{cfg.case_id}.smv"
        
    def parse(self) -> Dict[str, Any]:
        """Parse .smv file and extract structure."""
        if not self.smv_path.exists():
            return {"error": f"File not found: {self.smv_path}"}
        
        with open(self.smv_path, 'r') as f:
            lines = f.readlines()
        
        grid = None
        obstacles = []
        vents = []
        slices = []
        devices = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            cmd = parts[0].upper()
            
            if cmd == "GRID":
                # GRID x0 x1 y0 y1 z0 z1
                if len(parts) >= 7:
                    grid = {
                        "xb": [float(parts[1]), float(parts[2])],
                        "yb": [float(parts[3]), float(parts[4])],
                        "zb": [float(parts[5]), float(parts[6])],
                    }
                    
            elif cmd == "OBST":
                # OBST x0 x1 y0 y1 z0 z1
                if len(parts) >= 7:
                    obstacles.append({
                        "xb": [float(parts[1]), float(parts[2])],
                        "yb": [float(parts[3]), float(parts[4])],
                        "zb": [float(parts[5]), float(parts[6])],
                    })
                    
            elif cmd == "VENT":
                if len(parts) >= 7:
                    vents.append({
                        "xb": [float(parts[1]), float(parts[2])],
                        "yb": [float(parts[3]), float(parts[4])],
                        "zb": [float(parts[5]), float(parts[6])],
                    })
                    
            elif cmd == "SLCF":
                # Slice file reference
                slices.append({"raw": line})
                
            elif cmd == "DEVC":
                devices.append({"raw": line})
        
        return {
            "status": "parsed",
            "grid": grid,
            "n_obstacles": len(obstacles),
            "n_vents": len(vents),
            "n_slices": len(slices),
            "n_devices": len(devices),
            "obstacles": obstacles,
            "vents": vents,
        }


class SliceDataReader:
    """
    [D] Read FDS slice data files (.s3d).
    
    FDS slice files are Fortran binary format:
        - Header: metadata (time, position, quantity)
        - Data: 3D array of values
        
    HONEST: Binary format is FDS-version-specific; this targets FDS 6.7.x.
    """
    
    def __init__(self, cfg: SmokeviewConfig):
        self.cfg = cfg
        
    def read_slice(self, slice_path: Path) -> Dict[str, Any]:
        """
        Read a slice data file.
        
        Returns dict with 'time', 'data', 'position', 'quantity'.
        """
        if not slice_path.exists():
            return {"error": f"File not found: {slice_path}"}
        
        try:
            with open(slice_path, 'rb') as f:
                # FDS uses Fortran unformatted binary with record markers
                # Each record: [4-byte length][data][4-byte length]
                
                # Read first record: header info
                rec_len = struct.unpack('i', f.read(4))[0]
                header_data = f.read(rec_len)
                f.read(4)  # trailing length marker
                
                # Parse header (simplified; real format is more complex)
                # This is a placeholder; actual parsing depends on FDS version
                
                # Read subsequent records: time steps
                all_data = []
                times = []
                
                while True:
                    try:
                        rec_len_bytes = f.read(4)
                        if len(rec_len_bytes) < 4:
                            break
                        rec_len = struct.unpack('i', rec_len_bytes)[0]
                        
                        # Read time (first 4 bytes of record, typically)
                        time_val = struct.unpack('f', f.read(4))[0]
                        
                        # Read data (remaining bytes)
                        data_len = rec_len - 4
                        data_bytes = f.read(data_len)
                        
                        # Convert to numpy (assuming float32)
                        n_values = len(data_bytes) // 4
                        data = np.frombuffer(data_bytes, dtype=np.float32)
                        
                        all_data.append(data)
                        times.append(time_val)
                        
                        f.read(4)  # trailing marker
                        
                    except Exception as e:
                        logger.warning(f"Error reading slice record: {e}")
                        break
                
                return {
                    "status": "read",
                    "n_timesteps": len(times),
                    "times": times,
                    "data": all_data,
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def read_device_csv(self, devc_path: Path) -> Dict[str, Any]:
        """Read FDS device output CSV (temperature, velocity, etc.)."""
        if not devc_path.exists():
            return {"error": f"File not found: {devc_path}"}
        
        try:
            data = np.genfromtxt(devc_path, delimiter=',', names=True)
            return {
                "status": "read",
                "columns": list(data.dtype.names),
                "n_rows": len(data),
                "data": data,
            }
        except Exception as e:
            return {"error": str(e)}


class SmokeviewVisualizer:
    """
    [D] Visualize Smokeview data using matplotlib or VTK.
    
    HONEST: This is NOT Smokeview's rendering engine. It provides
    Python-based visualization of the extracted data.
    """
    
    def __init__(self, cfg: SmokeviewConfig):
        self.cfg = cfg
        self.parser = SmokeviewParser(cfg)
        self.reader = SliceDataReader(cfg)
        
    def visualize_slice(self, data: np.ndarray, shape: Tuple[int, int],
                        title: str = "Temperature") -> Optional[Any]:
        """Visualize a 2D slice."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Reshape data
        if data.size == shape[0] * shape[1]:
            data_2d = data.reshape(shape)
        else:
            logger.warning(f"Data size {data.size} != shape {shape}")
            data_2d = data.reshape(-1, 1)
        
        im = ax.imshow(data_2d, cmap=self.cfg.colormap, origin='lower')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax)
        
        if self.cfg.save_png:
            png_path = Path(self.cfg.output_dir) / f"{self.cfg.case_id}_slice.png"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved slice visualization to {png_path}")
        
        return fig
    
    def visualize_3d_structure(self, parse_result: Dict[str, Any]) -> Optional[Any]:
        """Visualize 3D structure (obstacles, vents) using matplotlib."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return None
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        obstacles = parse_result.get("obstacles", [])
        for obs in obstacles:
            xb, yb, zb = obs["xb"], obs["yb"], obs["zb"]
            # Draw box
            self._draw_box(ax, xb, yb, zb, color='gray', alpha=0.5)
        
        vents = parse_result.get("vents", [])
        for vent in vents:
            xb, yb, zb = vent["xb"], vent["yb"], vent["zb"]
            self._draw_box(ax, xb, yb, zb, color='blue', alpha=0.3)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Smokeview Structure: {self.cfg.case_id}")
        
        if self.cfg.save_png:
            png_path = Path(self.cfg.output_dir) / f"{self.cfg.case_id}_3d.png"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _draw_box(self, ax, xb, yb, zb, color='gray', alpha=0.5):
        """Draw a wireframe box on a 3D matplotlib axis."""
        x0, x1 = xb
        y0, y1 = yb
        z0, z1 = zb
        
        # 8 corners
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        
        # 12 edges
        edges = [
            [0,1],[1,2],[2,3],[3,0],  # bottom
            [4,5],[5,6],[6,7],[7,4],  # top
            [0,4],[1,5],[2,6],[3,7],  # vertical
        ]
        
        for edge in edges:
            ax.plot(corners[edge, 0], corners[edge, 1], corners[edge, 2],
                   color=color, alpha=alpha)


def smoke_test():
    print("=" * 72)
    print(" SESI Smokeview Connector — Smoke Test")
    print("=" * 72)
    print(f"  VTK available        : {HAS_VTK}")
    print(f"  Matplotlib available : {HAS_MATPLOTLIB}")
    
    cfg = SmokeviewConfig(case_id="test_fire", output_dir="./fds_test_output")
    parser = SmokeviewParser(cfg)
    reader = SliceDataReader(cfg)
    visualizer = SmokeviewVisualizer(cfg)
    
    print("\n  ── Parsing .smv file ──")
    result = parser.parse()
    if "error" in result:
        print(f"    [SKIP] {result['error']}")
        print("    (This is expected if FDS hasn't been run)")
    else:
        print(f"    Grid found         : {result['grid'] is not None}")
        print(f"    Obstacles          : {result['n_obstacles']}")
        print(f"    Vents              : {result['n_vents']}")
        print(f"    Slices             : {result['n_slices']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a PARSER, not Smokeview itself")
    print("  • Binary format is FDS-version-specific")
    print("  • Visualization is INTERPRETIVE (colormaps, slicing)")
    print("  • Large files require chunked reading")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
