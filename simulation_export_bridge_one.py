"""
SIMULATION EXPORT BRIDGE ONE — Wires every ONE Ecosystem module's outputs
into dcc_export_one.py's format writers.
================================================================================

Two sections:

  A. HAZARD MODULE ADAPTERS — seismic_one.py, fire_one.py, flood_one.py,
     storm_one.py, volcano_one.py: take each module's `.run()` result
     dict and produce ready-made deliverables (hazard-zone DXF for
     AutoCAD, depth/temperature PNG for Photoshop, wind-field SVG
     contours for Illustrator, etc.)

  B. DNS CLUSTER ADAPTERS — super_dns_one_v6_3.py (CompressibleSolver),
     structuralfluctuatinghydro_v6_3.py (FH), structural_langevin_v3.py,
     structural_cahn_hilliard_3d_v3.py (including its
     ThinFilmStructuralCahnHilliard3D and PhaseFieldCrystal3D
     subclasses), structural_fno_3d_v2_3.py (FNO surrogate predictions):
     take a solver instance or a raw field tensor and export its current
     state via dcc_export_one's PNG/OBJ/volumetric writers.

Every adapter converts torch.Tensor inputs to numpy via `.detach().cpu().
numpy()` if given a tensor, so this module works whether called from a
live GPU-resident simulation or from pre-saved numpy arrays. No adapter
here reimplements any export FORMAT logic -- all of that lives in
dcc_export_one.py; this file is purely "which field, which exporter, in
what deliverable shape."

Co-developers: Yoon (MSPS NETWORK) with Claude, GPT, Gemini, DeepSeek as AI
co-developers, consistent with the ONE Ecosystem attribution convention.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Any

from dcc_export_one import (
    ColorFieldExporter, VectorGraphicsExporter, CADExporter,
    MeshExporter, VolumetricExporter, GameEngineBridge,
)

__version__ = "1.0.0"
__all__ = [
    "SeismicExportAdapter", "FireExportAdapter", "FloodExportAdapter",
    "StormExportAdapter", "VolcanoExportAdapter",
    "SuperDNSExportAdapter", "CahnHilliardExportAdapter",
    "FluctuatingHydroExportAdapter", "LangevinExportAdapter",
    "FNOPredictionExportAdapter",
]


def _to_numpy(x: Any) -> np.ndarray:
    """Converts a torch.Tensor (if given one) to numpy; passes numpy arrays through."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# A. HAZARD MODULE ADAPTERS
# =============================================================================

class SeismicExportAdapter:
    """Wires seismic_one.py results into dcc_export_one exporters."""

    @staticmethod
    def export_liquefaction_profile(liquefaction_result: dict, output_dir: str,
                                     prefix: str = "liquefaction") -> dict:
        """
        LiquefactionAssessment.assess_profile() result -> SVG line plot
        (FS vs depth, with FS=1 threshold marked) and DXF cross-section
        (for import into an AutoCAD geotechnical drawing).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        depths = _to_numpy(liquefaction_result["depths"])
        FS = _to_numpy(liquefaction_result["FS"])

        # Build as a simple 2-column "field" so VectorGraphicsExporter's
        # generic contour machinery isn't needed -- write a minimal
        # dedicated line-chart SVG directly (FS vs depth is a profile,
        # not a 2D field with contours).
        W, H = 400, 600
        d_min, d_max = float(depths.min()), float(depths.max())
        fs_min, fs_max = 0.0, max(float(FS.max()) * 1.1, 2.0)

        def _sx(fs): return 40 + (fs - fs_min) / (fs_max - fs_min) * (W - 80)
        def _sy(d): return 40 + (d - d_min) / (d_max - d_min) * (H - 80)

        svg = [f'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
               f'width="{W}" height="{H}">\n']
        # FS=1 threshold line (liquefaction boundary)
        x1 = _sx(1.0)
        svg.append(f'  <line x1="{x1}" y1="40" x2="{x1}" y2="{H-40}" '
                   f'stroke="red" stroke-dasharray="4" stroke-width="1.5" />\n')
        pts = " ".join(f"{_sx(fs):.1f},{_sy(d):.1f}" for fs, d in zip(FS, depths))
        svg.append(f'  <polyline points="{pts}" fill="none" stroke="#0066cc" stroke-width="2" />\n')
        for fs, d, liq in zip(FS, depths, liquefaction_result["liquefiable"]):
            color = "red" if liq else "green"
            svg.append(f'  <circle cx="{_sx(fs):.1f}" cy="{_sy(d):.1f}" r="3" fill="{color}" />\n')
        svg.append("</svg>\n")
        svg_path = str(Path(output_dir) / f"{prefix}.svg")
        Path(svg_path).write_text("".join(svg))

        # DXF cross-section: same profile as a polyline (x=FS, y=-depth so
        # it reads top-down like a real boring log in AutoCAD)
        cad = CADExporter()
        dxf_pts = np.stack([FS, -depths], axis=-1)
        cad.add_polyline(dxf_pts, closed=False)
        cad.add_line(1.0, -d_min, 1.0, -d_max)   # FS=1 threshold
        cad.add_text(1.05, -d_min, "FS=1.0 (liquefaction threshold)", height=0.3)
        dxf_path = cad.write(str(Path(output_dir) / f"{prefix}.dxf"))

        return {"svg_path": svg_path, "dxf_path": dxf_path}

    @staticmethod
    def export_story_drift_diagram(structural_response_result: dict, output_dir: str,
                                    story_height_m: float = 3.5,
                                    prefix: str = "story_drift") -> str:
        """
        StructuralResponseLayer.time_history_analysis() result -> a DXF
        building-elevation diagram: one rectangle per story, colored by
        damage_index (green->yellow->red), annotated with max drift
        ratio -- a direct AutoCAD structural-engineering deliverable.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        damage = _to_numpy(structural_response_result["damage_index"])
        drift = _to_numpy(structural_response_result["max_drift_ratio"])
        n_stories = len(damage)

        cad = CADExporter()
        width = 8.0
        for i in range(n_stories):
            y0 = i * story_height_m
            y1 = y0 + story_height_m
            # color: DXF standard palette index approx green=3, yellow=2, red=1
            color = 1 if damage[i] > 0.7 else (2 if damage[i] > 0.3 else 3)
            cad.add_polyline(np.array([[0, y0], [width, y0], [width, y1], [0, y1]]),
                              closed=True, color=color)
            cad.add_text(width + 0.5, y0 + story_height_m / 2,
                         f"Story {i+1}: drift={drift[i]*100:.2f}%, damage={damage[i]:.2f}",
                         height=0.25, color=color)
        return cad.write(str(Path(output_dir) / f"{prefix}.dxf"))

    @staticmethod
    def export_response_spectrum_svg(spectrum: dict, output_path: str) -> str:
        """GroundMotionEngine.response_spectrum() result -> Sa vs Period SVG chart."""
        T = _to_numpy(spectrum["periods"])
        Sa = _to_numpy(spectrum["Sa"])
        W, H = 500, 350
        t_max, sa_max = float(T.max()), float(Sa.max()) * 1.1

        def _sx(t): return 50 + (t / t_max) * (W - 80)
        def _sy(sa): return H - 40 - (sa / sa_max) * (H - 80)

        pts = " ".join(f"{_sx(t):.1f},{_sy(sa):.1f}" for t, sa in zip(T, Sa))
        svg = (f'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
               f'width="{W}" height="{H}">\n'
               f'  <polyline points="{pts}" fill="none" stroke="#cc3300" stroke-width="2" />\n'
               f'  <text x="10" y="15" font-size="12">Sa (m/s^2) vs Period (s)</text>\n'
               f'</svg>\n')
        Path(output_path).write_text(svg)
        return output_path


class FireExportAdapter:
    """Wires fire_one.py / fire_dns_coupling_one.py results into dcc_export_one exporters."""

    @staticmethod
    def export_fire_source_volumetric(fire_source_field, t: float, output_path: str) -> str:
        """
        fire_dns_coupling_one.FireSourceField.field(t) -> volumetric NPZ
        (smoke/heat density proxy) for 3ds Max/Maya/Unreal/Unity via the
        VDB-conversion path documented in VolumetricExporter.
        """
        field = _to_numpy(fire_source_field.field(t))
        return VolumetricExporter.export_grid_npz(field, output_path, field_name="heat_release")

    @staticmethod
    def export_plume_profile_png(fire_run_result: dict, output_path: str) -> str:
        """
        FireOne.run() result -> a time-vs-height colormapped PNG of
        centerline plume temperature rise (centerline_dT_K), for
        Photoshop/PHOTO-PAINT/Painter.
        """
        dT = _to_numpy(fire_run_result["centerline_dT_K"]).T   # (heights, time)
        return ColorFieldExporter.export_png(dT, output_path, cmap_name="inferno")

    @staticmethod
    def export_hrr_curve_svg(fire_run_result: dict, output_path: str) -> str:
        t = _to_numpy(fire_run_result["time"])
        hrr = _to_numpy(fire_run_result["hrr_kw"])
        W, H = 500, 300
        t_max, hrr_max = float(t.max()), float(hrr.max()) * 1.1

        def _sx(ti): return 40 + (ti / t_max) * (W - 60)
        def _sy(h): return H - 30 - (h / hrr_max) * (H - 60)

        pts = " ".join(f"{_sx(ti):.1f},{_sy(h):.1f}" for ti, h in zip(t, hrr))
        svg = (f'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
               f'width="{W}" height="{H}">\n'
               f'  <polyline points="{pts}" fill="none" stroke="#ff6600" stroke-width="2" />\n'
               f'  <text x="10" y="15" font-size="12">HRR (kW) vs Time (s)</text>\n</svg>\n')
        Path(output_path).write_text(svg)
        return output_path


class FloodExportAdapter:
    """Wires flood_one.py results into dcc_export_one exporters."""

    @staticmethod
    def export_inundation_map_png(elevation_grid: np.ndarray, water_surface_elevation_m: float,
                                   output_path: str) -> str:
        """FloodplainModel.inundation_depth() -> colormapped PNG depth map."""
        from flood_one import FloodplainModel
        depth = FloodplainModel.inundation_depth(_to_numpy(elevation_grid), water_surface_elevation_m)
        return ColorFieldExporter.export_png(depth, output_path, cmap_name="Blues")

    @staticmethod
    def export_inundation_boundary(elevation_grid: np.ndarray, drainage_elevation_m: float,
                                    water_surface_elevation_m: float, output_dir: str,
                                    prefix: str = "flood_extent") -> dict:
        """
        FloodplainModel.inundation_extent_hand() boolean mask -> SVG + DXF
        boundary contour (the flood-extent line itself, level=0.5 on the
        float-cast mask) for site-plan overlay in Illustrator/CorelDRAW/
        AutoCAD.
        """
        from flood_one import FloodplainModel
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mask = FloodplainModel.inundation_extent_hand(
            _to_numpy(elevation_grid), drainage_elevation_m, water_surface_elevation_m)
        mask_f = mask.astype(float)
        svg_path = VectorGraphicsExporter.export_svg_contours(
            mask_f, str(Path(output_dir) / f"{prefix}.svg"), levels=[0.5],
            stroke_color="#0033cc", stroke_width=2.0)
        dxf_path = CADExporter.export_contours_as_dxf(
            mask_f, str(Path(output_dir) / f"{prefix}.dxf"), levels=[0.5])
        return {"svg_path": svg_path, "dxf_path": dxf_path}

    @staticmethod
    def export_damage_heatmap_png(depth_grid: np.ndarray, structure_values: np.ndarray,
                                   output_path: str, max_depth_for_total_loss_m: float = 3.0) -> str:
        from flood_one import FloodplainModel
        damage = FloodplainModel.estimate_damage(
            _to_numpy(depth_grid), _to_numpy(structure_values), max_depth_for_total_loss_m)
        return ColorFieldExporter.export_png(damage, output_path, cmap_name="Reds")


class StormExportAdapter:
    """Wires storm_one.py results into dcc_export_one exporters."""

    @staticmethod
    def export_tornado_wind_field(tornado_model, extent_m: float, resolution: int,
                                   output_dir: str, prefix: str = "tornado") -> dict:
        """
        TornadoVortexModel.wind_field() -> PNG speed heatmap + SVG
        isotach (constant-wind-speed) contours, e.g. for overlaying an
        EF-scale damage-radius diagram on a site plan.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        x = np.linspace(-extent_m, extent_m, resolution)
        y = np.linspace(-extent_m, extent_m, resolution)
        field = tornado_model.wind_field(x, y)
        speed = _to_numpy(field["speed"])

        png_path = ColorFieldExporter.export_png(speed, str(Path(output_dir) / f"{prefix}_speed.png"),
                                                   cmap_name="turbo")
        svg_path = VectorGraphicsExporter.export_svg_contours(
            speed, str(Path(output_dir) / f"{prefix}_isotachs.svg"))
        return {"png_path": png_path, "svg_path": svg_path}

    @staticmethod
    def export_hurricane_wind_field(tc_model, extent_m: float, resolution: int,
                                     output_dir: str, prefix: str = "hurricane") -> dict:
        """
        TropicalCycloneModel radial Holland profile, gridded onto a 2D
        plane (axisymmetric assumption) -> PNG wind-speed map + SVG
        isotach contours.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        x = np.linspace(-extent_m, extent_m, resolution)
        y = np.linspace(-extent_m, extent_m, resolution)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)
        speed = tc_model.wind_speed(r)

        png_path = ColorFieldExporter.export_png(speed, str(Path(output_dir) / f"{prefix}_speed.png"),
                                                   cmap_name="turbo")
        svg_path = VectorGraphicsExporter.export_svg_contours(
            speed, str(Path(output_dir) / f"{prefix}_isotachs.svg"))
        return {"png_path": png_path, "svg_path": svg_path}


class VolcanoExportAdapter:
    """Wires volcano_one.py results into dcc_export_one exporters."""

    @staticmethod
    def export_pdc_hazard_zone(pdc_result: dict, vent_xy: tuple, output_dir: str,
                                prefix: str = "pdc_hazard") -> dict:
        """
        PyroclasticDensityCurrentModel.hazard_radius_from_column_collapse()
        -> a DXF/SVG circle (radially-symmetric hazard boundary, matching
        the model's own documented limitation of not being topography-
        aware) centered at the vent, for GIS/CAD hazard-zone-map overlay.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        vx, vy = vent_xy
        R = pdc_result["runout_distance_m"]

        cad = CADExporter()
        cad.add_circle(vx, vy, R)
        cad.add_text(vx, vy + R + 50, f"PDC hazard zone, R={R/1000:.1f}km "
                     f"(radially symmetric, NOT topography-aware)", height=30)
        dxf_path = cad.write(str(Path(output_dir) / f"{prefix}.dxf"))

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_pts = np.stack([vx + R * np.cos(theta), vy + R * np.sin(theta)], axis=-1)
        svg_path = VectorGraphicsExporter.export_svg_polygons(
            [circle_pts], str(Path(output_dir) / f"{prefix}.svg"),
            canvas_size=(vx + R * 1.2, vy + R * 1.2),
            labels=[f"PDC R={R/1000:.1f}km"])
        return {"dxf_path": dxf_path, "svg_path": svg_path}


# =============================================================================
# B. DNS CLUSTER ADAPTERS
# =============================================================================

class SuperDNSExportAdapter:
    """
    Wires super_dns_one_v6_3.py's CompressibleSolver state into
    dcc_export_one exporters. Reads whichever fields are actually active
    on the solver instance (Z/Y_soot only if enable_combustion/
    enable_soot were set) rather than assuming all fields exist.
    """

    @staticmethod
    def export_state_snapshot(solver, output_dir: str, mid_slice_only: bool = True) -> dict:
        """
        Exports solver.rho (density) and, if active, solver.Z (mixture
        fraction) and solver.Y_soot as PNG mid-slices (Photoshop/PHOTO-
        PAINT/Painter) and NPZ volumetric grids (3ds Max/Maya/Unreal/
        Unity via the VDB-conversion path). mid_slice_only=False also
        exports a full PNG sequence per field (heavier).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        outputs = {}

        rho = _to_numpy(solver.rho)
        outputs["rho_volumetric"] = VolumetricExporter.export_grid_npz(
            rho, str(Path(output_dir) / "rho.npz"), field_name="density")
        mid_z = rho.shape[2] // 2
        outputs["rho_png"] = ColorFieldExporter.export_png(
            rho[:, :, mid_z], str(Path(output_dir) / "rho_midslice.png"), cmap_name="viridis")
        if not mid_slice_only:
            outputs["rho_png_sequence"] = ColorFieldExporter.export_png_sequence(
                rho, str(Path(output_dir) / "rho_seq"), axis=2)

        if getattr(solver, "enable_combustion", False):
            Z = _to_numpy(solver.Z)
            outputs["Z_volumetric"] = VolumetricExporter.export_grid_npz(
                Z, str(Path(output_dir) / "Z.npz"), field_name="mixture_fraction")
            outputs["Z_png"] = ColorFieldExporter.export_png(
                Z[:, :, mid_z], str(Path(output_dir) / "Z_midslice.png"), cmap_name="inferno")
            outputs["flame_isosurface_obj"] = MeshExporter.export_obj_isosurface(
                Z, iso_level=float(getattr(solver, "z_stoich", 0.055)),
                output_path=str(Path(output_dir) / "flame_surface.obj"))

        if getattr(solver, "enable_soot", False):
            Ysoot = _to_numpy(solver.Y_soot)
            outputs["soot_volumetric"] = VolumetricExporter.export_grid_npz(
                Ysoot, str(Path(output_dir) / "soot.npz"), field_name="soot")
            outputs["soot_png"] = ColorFieldExporter.export_png(
                Ysoot[:, :, mid_z], str(Path(output_dir) / "soot_midslice.png"), cmap_name="Greys")

        return outputs

    @staticmethod
    def stream_live_state(solver, stream_server, field: str = "rho") -> int:
        """
        Pushes the solver's current `field` (e.g. 'rho', 'Z', 'Y_soot')
        to a running dcc_export_one.LiveDataStreamServer -- call this
        once per N steps inside your own simulation loop for real-time
        Unity/Unreal visualization.
        """
        data = _to_numpy(getattr(solver, field))
        return stream_server.push_frame(float(solver.time), field, data)


class CahnHilliardExportAdapter:
    """
    Wires structural_cahn_hilliard_3d_v3.py's StructuralCahnHilliard3D
    (and its ThinFilmStructuralCahnHilliard3D / PhaseFieldCrystal3D
    subclasses, which share the same order-parameter field convention)
    into dcc_export_one exporters. These are functional (`.step(field,
    sigma) -> field`) modules, not stateful solver objects, so adapters
    here take the field TENSOR directly rather than a solver instance.
    """

    @staticmethod
    def export_phase_field_snapshot(u_field, output_dir: str, prefix: str = "ch3d",
                                     interface_level: float = 0.0) -> dict:
        """
        u_field: (Nx,Ny,Nz) order-parameter tensor/array (works
        identically for StructuralCahnHilliard3D, ThinFilmStructural
        CahnHilliard3D, or PhaseFieldCrystal3D -- all three expose the
        same field convention). Exports a mid-slice PNG and an OBJ
        interface isosurface (u=interface_level, the standard CH
        phase-boundary convention).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        u = _to_numpy(u_field)
        # strip leading batch/channel dims if present (e.g. (1,1,Nx,Ny,Nz))
        while u.ndim > 3:
            u = u[0]
        mid_z = u.shape[2] // 2

        png_path = ColorFieldExporter.export_png(
            u[:, :, mid_z], str(Path(output_dir) / f"{prefix}_midslice.png"), cmap_name="coolwarm")
        vol_path = VolumetricExporter.export_grid_npz(
            u, str(Path(output_dir) / f"{prefix}.npz"), field_name="order_parameter")

        result = {"png_path": png_path, "volumetric_path": vol_path}
        if u.min() < interface_level < u.max():
            result["interface_obj_path"] = MeshExporter.export_obj_isosurface(
                u, iso_level=interface_level, output_path=str(Path(output_dir) / f"{prefix}_interface.obj"))
        return result


class FluctuatingHydroExportAdapter:
    """
    Wires structuralfluctuatinghydro_v6_3.py's StructuralFluctuatingHydro
    output fields into dcc_export_one exporters. Functional module (same
    pattern as CahnHilliardExportAdapter) -- takes field tensors directly.
    """

    @staticmethod
    def export_velocity_field_snapshot(rho_field, velocity_field, output_dir: str,
                                        prefix: str = "fh") -> dict:
        """
        rho_field: (Nx,Ny,Nz); velocity_field: (3,Nx,Ny,Nz) or
        (Nx,Ny,Nz,3). Exports density as PNG/NPZ and velocity MAGNITUDE
        as a separate PNG/NPZ (the vector field itself isn't directly
        renderable as a single scalar image; magnitude is the standard
        visualization proxy -- for a real vector-field visualization in
        3ds Max/Maya/Unreal/Unity, export velocity components as three
        separate NPZ grids and reconstruct vectors on the DCC side).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rho = _to_numpy(rho_field)
        while rho.ndim > 3:
            rho = rho[0]
        v = _to_numpy(velocity_field)
        while v.ndim > 4:
            v = v[0]
        if v.shape[0] == 3:   # (3,Nx,Ny,Nz) -> magnitude over axis 0
            speed = np.sqrt((v**2).sum(axis=0))
        else:                  # (Nx,Ny,Nz,3) -> magnitude over last axis
            speed = np.sqrt((v**2).sum(axis=-1))

        mid_z = rho.shape[2] // 2
        outputs = {
            "rho_png": ColorFieldExporter.export_png(
                rho[:, :, mid_z], str(Path(output_dir) / f"{prefix}_rho.png"), cmap_name="viridis"),
            "rho_volumetric": VolumetricExporter.export_grid_npz(
                rho, str(Path(output_dir) / f"{prefix}_rho.npz"), field_name="density"),
            "speed_png": ColorFieldExporter.export_png(
                speed[:, :, mid_z], str(Path(output_dir) / f"{prefix}_speed.png"), cmap_name="plasma"),
            "speed_volumetric": VolumetricExporter.export_grid_npz(
                speed, str(Path(output_dir) / f"{prefix}_speed.npz"), field_name="speed"),
        }
        return outputs


class LangevinExportAdapter:
    """
    Wires structural_langevin_v3.py's AdvancedStructuralLangevin output
    (structural stress / sigma field) into dcc_export_one exporters.
    Functional module -- takes the field tensor directly.
    """

    @staticmethod
    def export_stress_field_snapshot(sigma_field, output_dir: str, prefix: str = "langevin") -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sigma = _to_numpy(sigma_field)
        while sigma.ndim > 3:
            sigma = sigma[0]
        mid_z = sigma.shape[2] // 2
        return {
            "png_path": ColorFieldExporter.export_png(
                sigma[:, :, mid_z], str(Path(output_dir) / f"{prefix}_midslice.png"), cmap_name="magma"),
            "volumetric_path": VolumetricExporter.export_grid_npz(
                sigma, str(Path(output_dir) / f"{prefix}.npz"), field_name="structural_stress"),
        }


class FNOPredictionExportAdapter:
    """
    Wires structural_fno_3d_v2_3.py's StructuralFNO3D predictions into
    dcc_export_one exporters -- both single-shot forward() output and
    forward_rollout() trajectories, and multi-channel (fire-configured,
    n_state_channels>1) predictions.
    """

    @staticmethod
    def export_prediction(mean, log_var, output_dir: str, prefix: str = "fno_pred",
                           channel_names: Optional[list] = None) -> dict:
        """
        mean, log_var: (B, C, Nx, Ny, Nz) from StructuralFNO3D.forward().
        Exports each channel's mean as PNG/NPZ and its predictive std
        (from log_var) as a separate uncertainty-heatmap PNG -- useful
        for visualizing WHERE the surrogate is confident vs. uncertain,
        e.g. as a texture overlay in a DCC tool.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mean_np = _to_numpy(mean)[0]      # drop batch dim -> (C,Nx,Ny,Nz)
        log_var_np = _to_numpy(log_var)[0]
        C = mean_np.shape[0]
        names = channel_names or [f"channel{i}" for i in range(C)]

        outputs = {}
        for c in range(C):
            field = mean_np[c]
            std = np.exp(0.5 * log_var_np[c])
            mid_z = field.shape[2] // 2
            outputs[names[c]] = {
                "mean_png": ColorFieldExporter.export_png(
                    field[:, :, mid_z], str(Path(output_dir) / f"{prefix}_{names[c]}_mean.png")),
                "mean_volumetric": VolumetricExporter.export_grid_npz(
                    field, str(Path(output_dir) / f"{prefix}_{names[c]}.npz"), field_name=names[c]),
                "uncertainty_png": ColorFieldExporter.export_png(
                    std[:, :, mid_z], str(Path(output_dir) / f"{prefix}_{names[c]}_std.png"),
                    cmap_name="Reds"),
            }
        return outputs

    @staticmethod
    def export_rollout_trajectory(rollout_result: dict, output_dir: str,
                                   prefix: str = "fno_rollout",
                                   channel_names: Optional[list] = None) -> str:
        """
        StructuralFNO3DEncoderAdapter.predict_seismic_scenario() /
        predict_fire_scenario() result (from AGI ONE) -> a
        GameEngineBridge JSON time-series export, ready for a Unity C#
        or Unreal C++/Blueprint importer to drive an animated
        visualization of the predicted scenario evolving over time.
        """
        traj = _to_numpy(rollout_result["trajectory"])   # (n_steps+1, B, C, Nx,Ny,Nz)
        n_steps, B = traj.shape[0], traj.shape[1]
        C = traj.shape[2]
        names = channel_names or [f"channel{i}" for i in range(C)]

        frames = []
        for step in range(n_steps):
            for c in range(C):
                frames.append({"time": float(step), "field_name": names[c],
                                "data": traj[step, 0, c]})
        return GameEngineBridge.export_json_timeseries(frames, output_dir, f"{prefix}_manifest.json")


if __name__ == "__main__":
    print(f"simulation_export_bridge_one.py v{__version__} loaded OK")
