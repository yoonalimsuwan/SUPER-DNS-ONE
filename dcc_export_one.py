"""
DCC EXPORT ONE — Digital Content Creation / CAD / Game Engine Export Bridge
================================================================================

Exports ONE Ecosystem simulation results (seismic, fire, flood, storm,
volcano, and raw DNS field data) into standard interchange formats
consumable by external creative/engineering/game-development tools.

Organized by FORMAT, not by target application, since most listed tools
share the same native/importable formats:

    1. ColorFieldExporter    — PNG image sequences / heightmaps
                                (Adobe Photoshop, Corel PHOTO-PAINT, Corel Painter)
    2. VectorGraphicsExporter — SVG contours/polygons
                                (Adobe Illustrator, CorelDRAW)
    3. CADExporter            — DXF (hand-written, no ezdxf dependency)
                                (AutoCAD; also importable by Illustrator/CorelDRAW)
    4. MeshExporter           — OBJ isosurfaces (marching cubes) / point clouds
                                (Autodesk 3ds Max, Autodesk Maya, Unreal Engine, Unity)
    5. VolumetricExporter     — simple documented volumetric grid format
                                (NOT real OpenVDB -- see class docstring)
    6. GameEngineBridge       — JSON time-series export + live TCP streaming
                                (Unreal Engine, Unity)

======================================================================
 HONESTY NOTICE
======================================================================
This module was built WITHOUT network access to install the standard
libraries real production pipelines use for several of these formats
(ezdxf for DXF, openvdb for real OpenVDB volumes, trimesh/FBX SDK for
FBX). Where a library was unavailable, a dependency-free hand-written
writer is used instead (DXF, OBJ) covering the common entity/element
subset needed for the exports in this module -- NOT a full
implementation of the format specification. Where no reasonable
dependency-free approximation exists (true OpenVDB, true FBX), this
module exports a clearly-labeled proxy format and documents the
external conversion step needed, rather than claiming support it
doesn't have.
======================================================================

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem attribution convention.
"""

from __future__ import annotations

import numpy as np
import struct
import json
import socket
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Callable
from dataclasses import dataclass, field

__version__ = "1.0.0"
__all__ = [
    "ColorFieldExporter",
    "VectorGraphicsExporter",
    "CADExporter",
    "MeshExporter",
    "VolumetricExporter",
    "GameEngineBridge",
]


# =====================================================================
# 1. COLOR FIELD EXPORTER  (Photoshop / Corel PHOTO-PAINT / Corel Painter)
# =====================================================================

class ColorFieldExporter:
    """
    Exports 2D/3D scalar fields as PNG raster images -- the universal
    interchange format for Adobe Photoshop, Corel PHOTO-PAINT, and Corel
    Painter (all read PNG natively; no proprietary-format writer is
    needed or attempted here).
    """

    @staticmethod
    def _apply_colormap(field_2d: np.ndarray, cmap_name: str = "viridis",
                         vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        vmin = float(np.min(field_2d)) if vmin is None else vmin
        vmax = float(np.max(field_2d)) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        rgba = cmap(norm(field_2d))   # (H,W,4), float [0,1]
        return (rgba[..., :3] * 255).astype(np.uint8)

    @classmethod
    def export_png(cls, field_2d: np.ndarray, output_path: str,
                    cmap_name: str = "viridis", vmin: Optional[float] = None,
                    vmax: Optional[float] = None) -> str:
        """Exports a single 2D field as a colormapped PNG."""
        from PIL import Image
        rgb = cls._apply_colormap(field_2d, cmap_name, vmin, vmax)
        Image.fromarray(rgb, mode="RGB").save(output_path)
        return output_path

    @classmethod
    def export_png_sequence(cls, field_3d: np.ndarray, output_dir: str,
                             axis: int = 2, prefix: str = "frame",
                             cmap_name: str = "inferno") -> list:
        """
        Exports every slice along `axis` of a 3D field as a numbered PNG
        sequence (e.g. for a smoke/fire/ash density field, or a DNS
        field's z-slices) -- a standard image-sequence workflow any of
        these tools can open (Photoshop: File > Import > Image Sequence;
        Painter/PHOTO-PAINT: open individual frames or batch-process).
        Uses a SHARED vmin/vmax across the whole sequence (computed once
        from the full field) so brightness is consistent frame-to-frame.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        vmin, vmax = float(np.min(field_3d)), float(np.max(field_3d))
        n = field_3d.shape[axis]
        paths = []
        for i in range(n):
            sl = [slice(None)] * 3
            sl[axis] = i
            slice_2d = field_3d[tuple(sl)]
            out_path = str(Path(output_dir) / f"{prefix}_{i:04d}.png")
            cls.export_png(slice_2d, out_path, cmap_name, vmin, vmax)
            paths.append(out_path)
        return paths

    @staticmethod
    def export_heightmap_png16(elevation_2d: np.ndarray, output_path: str) -> str:
        """
        Exports a 2D elevation/depth field as a 16-bit grayscale PNG
        heightmap (standard terrain-heightmap convention used across
        DCC/game tools, including Photoshop's own 16-bit grayscale
        workflow for terrain authoring) -- e.g. for a flood_one.py DEM,
        volcano_one.py topography, or a DNS field's magnitude.
        """
        from PIL import Image
        z = np.asarray(elevation_2d, dtype=float)
        z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-12)
        z16 = (z_norm * 65535).astype(np.uint16)
        Image.fromarray(z16, mode="I;16").save(output_path)
        return output_path


# =====================================================================
# 2. VECTOR GRAPHICS EXPORTER  (Adobe Illustrator / CorelDRAW)
# =====================================================================

class VectorGraphicsExporter:
    """
    Exports 2D contours and polygons as SVG -- the standard vector
    interchange format both Adobe Illustrator and CorelDRAW import
    natively (File > Open/Place; no proprietary .ai or .cdr writer is
    attempted here, since both those formats are undocumented/
    proprietary binary formats this module cannot safely hand-write).
    """

    @staticmethod
    def _svg_header(width: float, height: float) -> str:
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')

    @classmethod
    def export_svg_contours(cls, field_2d: np.ndarray, output_path: str,
                             levels: Optional[Sequence[float]] = None,
                             scale: float = 1.0, stroke_color: str = "#0066cc",
                             stroke_width: float = 1.0) -> str:
        """
        Extracts contour lines from a 2D scalar field (via
        skimage.measure.find_contours, marching-squares) and writes them
        as SVG polylines -- e.g. flood-depth isolines, temperature
        contours, hazard-probability contours. `levels` defaults to 5
        evenly-spaced levels across the field's range if not given.
        """
        from skimage import measure
        field_2d = np.asarray(field_2d, dtype=float)
        H, W = field_2d.shape
        if levels is None:
            levels = np.linspace(field_2d.min(), field_2d.max(), 7)[1:-1]

        svg = [cls._svg_header(W * scale, H * scale)]
        for level in levels:
            contours = measure.find_contours(field_2d, level=level)
            for c in contours:
                pts = " ".join(f"{y*scale:.2f},{x*scale:.2f}" for x, y in c)
                svg.append(f'  <polyline points="{pts}" fill="none" '
                           f'stroke="{stroke_color}" stroke-width="{stroke_width}" '
                           f'data-level="{level:.4g}" />\n')
        svg.append("</svg>\n")
        Path(output_path).write_text("".join(svg))
        return output_path

    @classmethod
    def export_svg_polygons(cls, polygons: Sequence[np.ndarray], output_path: str,
                             canvas_size: tuple, labels: Optional[Sequence[str]] = None,
                             fill_colors: Optional[Sequence[str]] = None,
                             scale: float = 1.0) -> str:
        """
        Writes a list of 2D polygons (each an (N,2) array of x,y points)
        as filled SVG shapes with optional text labels -- e.g. hazard
        zones (flood extent, PDC runout circle rendered as a polygon,
        tornado damage swath, storm-surge inundation boundary).
        """
        W, H = canvas_size
        svg = [cls._svg_header(W * scale, H * scale)]
        default_colors = ["#ff000055", "#ff880055", "#ffff0055", "#00ff0055"]
        for i, poly in enumerate(polygons):
            poly = np.asarray(poly, dtype=float)
            pts = " ".join(f"{x*scale:.2f},{y*scale:.2f}" for x, y in poly)
            color = (fill_colors[i] if fill_colors and i < len(fill_colors)
                     else default_colors[i % len(default_colors)])
            svg.append(f'  <polygon points="{pts}" fill="{color}" stroke="#333" stroke-width="1" />\n')
            if labels and i < len(labels):
                cx, cy = poly.mean(axis=0) * scale
                svg.append(f'  <text x="{cx:.1f}" y="{cy:.1f}" font-size="14" '
                           f'text-anchor="middle">{labels[i]}</text>\n')
        svg.append("</svg>\n")
        Path(output_path).write_text("".join(svg))
        return output_path


# =====================================================================
# 3. CAD EXPORTER  (AutoCAD)  — hand-written minimal DXF, no ezdxf
# =====================================================================

class CADExporter:
    """
    Writes a minimal (but valid) ASCII DXF file -- the standard AutoCAD
    interchange format -- WITHOUT the `ezdxf` library (unavailable, no
    network access to install it in the environment that built this
    module). Implements only the entity subset needed for this module's
    exports: LINE, LWPOLYLINE, CIRCLE, TEXT, in a single "0" layer of a
    minimal but structurally valid DXF R12-format file (the most widely
    compatible DXF version for cross-application import, including
    AutoCAD, Illustrator, and CorelDRAW).

    NOT a general-purpose DXF writer -- e.g. no layers, no blocks, no
    3D solids, no color-by-entity beyond the DXF default palette index.
    For real engineering-drawing production, use `ezdxf` (a real,
    actively-maintained DXF library) if you have network access to
    install it; this hand-written writer exists specifically for this
    module's dependency-free operation.
    """

    def __init__(self):
        self._entities: list = []

    def add_line(self, x1, y1, x2, y2, z=0.0, color=7):
        self._entities.append(("LINE", (x1, y1, z, x2, y2, z), color))

    def add_polyline(self, points: np.ndarray, closed: bool = False, color: int = 7):
        self._entities.append(("LWPOLYLINE", (np.asarray(points, dtype=float), closed), color))

    def add_circle(self, cx, cy, radius, z=0.0, color=7):
        self._entities.append(("CIRCLE", (cx, cy, z, radius), color))

    def add_text(self, x, y, text: str, height: float = 2.5, z=0.0, color=7):
        self._entities.append(("TEXT", (x, y, z, height, text), color))

    def _entity_dxf(self, etype, data, color) -> str:
        lines = []
        if etype == "LINE":
            x1, y1, z1, x2, y2, z2 = data
            lines += ["0", "LINE", "8", "0", "62", str(color),
                      "10", f"{x1}", "20", f"{y1}", "30", f"{z1}",
                      "11", f"{x2}", "21", f"{y2}", "31", f"{z2}"]
        elif etype == "LWPOLYLINE":
            points, closed = data
            lines += ["0", "LWPOLYLINE", "8", "0", "62", str(color),
                      "90", str(len(points)), "70", "1" if closed else "0"]
            for x, y in points:
                lines += ["10", f"{x}", "20", f"{y}"]
        elif etype == "CIRCLE":
            cx, cy, z, r = data
            lines += ["0", "CIRCLE", "8", "0", "62", str(color),
                      "10", f"{cx}", "20", f"{cy}", "30", f"{z}", "40", f"{r}"]
        elif etype == "TEXT":
            x, y, z, h, text = data
            lines += ["0", "TEXT", "8", "0", "62", str(color),
                      "10", f"{x}", "20", f"{y}", "30", f"{z}",
                      "40", f"{h}", "1", text]
        return "\n".join(lines) + "\n"

    def write(self, output_path: str) -> str:
        """
        Writes the minimal valid DXF R12 structure: HEADER (minimal),
        TABLES (empty/default), ENTITIES, EOF -- the minimum a DXF
        reader (including AutoCAD) requires to parse the file
        successfully.
        """
        parts = [
            "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
            "0", "SECTION", "2", "TABLES", "0", "ENDSEC",
            "0", "SECTION", "2", "ENTITIES",
        ]
        body = "\n".join(parts) + "\n"
        for etype, data, color in self._entities:
            body += self._entity_dxf(etype, data, color)
        body += "0\nENDSEC\n0\nEOF\n"
        Path(output_path).write_text(body)
        return output_path

    @classmethod
    def export_contours_as_dxf(cls, field_2d: np.ndarray, output_path: str,
                                levels: Optional[Sequence[float]] = None) -> str:
        """Convenience: contour lines from a 2D field, straight to DXF polylines."""
        from skimage import measure
        field_2d = np.asarray(field_2d, dtype=float)
        if levels is None:
            levels = np.linspace(field_2d.min(), field_2d.max(), 7)[1:-1]
        cad = cls()
        for level in levels:
            for c in measure.find_contours(field_2d, level=level):
                pts = np.stack([c[:, 1], c[:, 0]], axis=-1)   # (x,y) from (row,col)
                cad.add_polyline(pts, closed=False)
        return cad.write(output_path)


# =====================================================================
# 4. MESH EXPORTER  (3ds Max, Maya, Unreal Engine, Unity)
# =====================================================================

class MeshExporter:
    """
    Writes Wavefront OBJ mesh files -- the simplest, most universally
    supported 3D mesh interchange format, importable by 3ds Max, Maya,
    Unreal Engine, and Unity without any plugin. Uses
    `skimage.measure.marching_cubes` (available in this environment) for
    isosurface extraction; no `trimesh` dependency (unavailable).

    For a real VFX/game production pipeline, FBX or Alembic (.abc) is
    often preferred (animation/rig support, industry-standard sim
    caching) -- both are complex binary formats this module does NOT
    attempt to hand-write; OBJ is the dependency-free common denominator
    that all four target applications import directly.
    """

    @staticmethod
    def export_obj_isosurface(field_3d: np.ndarray, iso_level: float,
                               output_path: str, voxel_size: float = 1.0,
                               origin: tuple = (0.0, 0.0, 0.0)) -> dict:
        """
        Extracts an isosurface (e.g. a flame envelope, flood boundary
        surface, or any resolved DNS field's threshold surface) via
        marching cubes and writes it as an OBJ mesh with vertex normals.
        """
        from skimage import measure
        field_3d = np.asarray(field_3d, dtype=float)
        verts, faces, normals, _ = measure.marching_cubes(field_3d, level=iso_level)
        verts = verts * voxel_size + np.array(origin)

        with open(output_path, "w") as f:
            f.write(f"# Exported by dcc_export_one.py v{__version__}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in faces:
                i0, i1, i2 = face + 1   # OBJ is 1-indexed
                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")

        return {"path": output_path, "n_vertices": len(verts), "n_faces": len(faces)}

    @staticmethod
    def export_obj_point_cloud(points: np.ndarray, output_path: str,
                                colors: Optional[np.ndarray] = None) -> str:
        """
        Writes a point cloud (e.g. tephra/soot/particle positions) as an
        OBJ file of vertex-only points (optionally with per-vertex color,
        a widely-supported OBJ extension). Point clouds have no faces --
        importing tools typically render them as points or convert them
        to a particle system (Unity/Unreal) on import.
        """
        points = np.asarray(points, dtype=float)
        with open(output_path, "w") as f:
            f.write(f"# Point cloud exported by dcc_export_one.py v{__version__}\n")
            for i, p in enumerate(points):
                if colors is not None:
                    c = colors[i]
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                else:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        return output_path


# =====================================================================
# 5. VOLUMETRIC EXPORTER  (smoke/fire/ash density fields)
# =====================================================================

class VolumetricExporter:
    """
    Exports a 3D scalar/vector field as a volumetric grid for smoke/
    fire/ash-cloud rendering.

    IMPORTANT: this does NOT write real OpenVDB. The `openvdb` Python
    library requires a compiled C++ binding this module's build
    environment had no network access to install, and OpenVDB's binary
    format is not reasonably hand-writable (it uses a hierarchical
    sparse tree structure, not a simple documented flat format). Instead
    this exports a simple, fully-documented flat format (a `.npz`
    archive: the raw dense grid array plus voxel size/origin metadata)
    that YOU (or a small conversion script run in an environment with
    the real `openvdb` or `pyopenvdb` package installed) can convert to
    real .vdb losslessly -- see `to_vdb_conversion_snippet()` below for
    exactly that script.

    3ds Max, Maya, Unreal, and Unity all consume real .vdb via plugins
    (e.g. Unreal's built-in Sparse Volume Texture / VDB import, Maya's
    Bifrost, 3ds Max's native VDB support) -- none of them read this
    module's .npz proxy format directly.
    """

    @staticmethod
    def export_grid_npz(field_3d: np.ndarray, output_path: str,
                         voxel_size: float = 1.0, origin: tuple = (0.0, 0.0, 0.0),
                         field_name: str = "density") -> str:
        np.savez_compressed(output_path, **{field_name: field_3d.astype(np.float32)},
                             voxel_size=voxel_size, origin=np.array(origin))
        return output_path

    @staticmethod
    def to_vdb_conversion_snippet(npz_path: str, vdb_path: str, field_name: str = "density") -> str:
        """
        Returns (as a STRING, not executed here -- this module has no
        openvdb available) a ready-to-run Python snippet that converts
        this module's .npz proxy grid to a real .vdb file, for use in an
        environment that has the real `openvdb`/`pyopenvdb` package
        installed (e.g. a Houdini Python environment, or `pip install
        openvdb` where network access allows it).
        """
        return f'''# Run this in an environment with `openvdb` installed (this module's own
# environment does not have it). Converts {npz_path} -> {vdb_path} losslessly.
import numpy as np
import openvdb

data = np.load("{npz_path}")
grid_array = data["{field_name}"]
voxel_size = float(data["voxel_size"])
origin = data["origin"]

grid = openvdb.FloatGrid()
grid.copyFromArray(grid_array.astype(np.float32))
grid.transform = openvdb.createLinearTransform(voxelSize=voxel_size)
grid.transform.postTranslate(origin.tolist())
grid.name = "{field_name}"
openvdb.write("{vdb_path}", grids=[grid])
'''


# =====================================================================
# 6. GAME ENGINE BRIDGE  (Unreal Engine, Unity)
# =====================================================================

@dataclass
class StreamFrame:
    time: float
    field_name: str
    shape: tuple
    data: np.ndarray


class GameEngineBridge:
    """
    Two integration paths for Unreal Engine / Unity, neither requiring
    any proprietary Epic/Unity file format:

      1. export_json_timeseries() -- static, authoring-time export: a
         JSON manifest + raw binary field files, for a C# (Unity) or
         C++/Blueprint (Unreal) importer script you write in-engine to
         read once and drive a VFX Graph / Niagara system, terrain
         heightmap, or mesh deformation.

      2. LiveDataStreamServer -- a live TCP socket server streaming
         simulation frames in real time (length-prefixed binary
         protocol, documented below) to a connected Unity/Unreal client
         DURING a running simulation, for real-time visualization (e.g.
         watching a flood or fire simulation evolve inside the game
         engine's viewport as it computes).

    Neither path requires Unity/Unreal-side proprietary format writers
    -- both engines have straightforward APIs for reading raw binary
    buffers or opening a TCP socket from a client script.
    """

    @staticmethod
    def export_json_timeseries(frames: Sequence[dict], output_dir: str,
                                manifest_name: str = "manifest.json") -> str:
        """
        frames: list of dicts like {"time": t, "field_name": "temperature",
                "shape": (nx,ny,nz), "data": np.ndarray}. Writes each
        frame's raw data as a separate .bin file (row-major float32,
        documented in the manifest) plus a JSON manifest describing the
        sequence -- straightforward to parse from C# (BinaryReader) or
        C++ (ifstream) without any external library.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        manifest = {"version": __version__, "frame_count": len(frames), "frames": []}
        for i, fr in enumerate(frames):
            bin_name = f"frame_{i:05d}_{fr['field_name']}.bin"
            data = np.asarray(fr["data"], dtype=np.float32)
            data.tofile(str(Path(output_dir) / bin_name))
            manifest["frames"].append({
                "index": i, "time": fr["time"], "field_name": fr["field_name"],
                "shape": list(data.shape), "dtype": "float32", "file": bin_name,
            })
        manifest_path = str(Path(output_dir) / manifest_name)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2))
        return manifest_path


class LiveDataStreamServer:
    """
    Minimal TCP server streaming simulation frames to a connected
    Unity/Unreal client in real time.

    WIRE PROTOCOL (documented so any client language can implement it,
    no proprietary serialization library needed):
        For each frame, in order:
            [4 bytes]  uint32 little-endian: length of JSON header, N
            [N bytes]  UTF-8 JSON header: {"time":float,"field_name":str,
                       "shape":[nx,ny,nz],"dtype":"float32"}
            [4 bytes]  uint32 little-endian: length of binary payload, M
            [M bytes]  raw float32 data, row-major, product(shape)*4 == M

    Usage::
        server = LiveDataStreamServer(port=9998)
        server.start()
        for t, field in simulation_loop():
            server.push_frame(t, "temperature", field)
        server.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9998):
        self.host, self.port = host, port
        self._server_socket: Optional[socket.socket] = None
        self._clients: list = []
        self._lock = threading.Lock()
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(0.5)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                with self._lock:
                    self._clients.append(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def push_frame(self, t: float, field_name: str, data: np.ndarray) -> int:
        """Broadcasts one frame to all connected clients. Returns count of successful sends."""
        data = np.asarray(data, dtype=np.float32)
        header = json.dumps({"time": t, "field_name": field_name,
                              "shape": list(data.shape), "dtype": "float32"}).encode("utf-8")
        payload = data.tobytes()
        packet = (struct.pack("<I", len(header)) + header +
                  struct.pack("<I", len(payload)) + payload)

        sent = 0
        with self._lock:
            dead = []
            for conn in self._clients:
                try:
                    conn.sendall(packet)
                    sent += 1
                except OSError:
                    dead.append(conn)
            for d in dead:
                self._clients.remove(d)
        return sent

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._server_socket:
            self._server_socket.close()
        if self._accept_thread:
            self._accept_thread.join(timeout=2.0)


if __name__ == "__main__":
    print(f"dcc_export_one.py v{__version__} loaded OK")
