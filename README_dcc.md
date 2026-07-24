# DCC EXPORT ONE — Digital Content Creation / CAD / Game Engine Export Bridge

Part of the **ONE Ecosystem**. Exports simulation and hazard-analysis
results into standard interchange formats consumable by external
creative, engineering, and game-development tools.

| File | Role |
|---|---|
| `dcc_export_one.py` | Format-level exporters (PNG, SVG, DXF, OBJ, volumetric grid, JSON/TCP streaming) — no hazard-module-specific knowledge |
| `simulation_export_bridge_one.py` | Wires each ONE Ecosystem module's actual outputs into the format exporters above |

---

## 1. Design principle: organize by format, not by target application

Ten target applications were requested (Adobe Photoshop, Adobe
Illustrator, Corel PHOTO-PAINT, Corel Painter, CorelDRAW, Autodesk 3ds
Max, Autodesk Maya, AutoCAD, Unreal Engine, Unity), but most of them
share the same native/importable interchange formats. Rather than ten
separate per-application writers, `dcc_export_one.py` implements six
format-level exporters:

| Class | Format | Target applications |
|---|---|---|
| `ColorFieldExporter` | PNG (8-bit colormap + 16-bit heightmap) | Adobe Photoshop, Corel PHOTO-PAINT, Corel Painter |
| `VectorGraphicsExporter` | SVG (contours, filled polygons with labels) | Adobe Illustrator, CorelDRAW |
| `CADExporter` | DXF (hand-written, no `ezdxf` dependency) | AutoCAD; also importable by Illustrator/CorelDRAW |
| `MeshExporter` | OBJ (marching-cubes isosurfaces, point clouds) | 3ds Max, Maya, Unreal Engine, Unity |
| `VolumetricExporter` | `.npz` proxy grid + a generated OpenVDB conversion script | 3ds Max, Maya, Unreal Engine, Unity (via VDB import) |
| `GameEngineBridge` | JSON manifest + raw binary frames, or live TCP streaming | Unreal Engine, Unity |

## 2. What's genuinely supported vs. what's a documented proxy

This module was built **without network access** to install the
libraries a real production pipeline would use for several formats
(`ezdxf` for DXF, `openvdb`/`pyopenvdb` for real OpenVDB volumes,
`trimesh` or the FBX SDK for FBX/Alembic). The honesty rule applied
throughout:

- **Genuinely implemented, tested, format-correct:** PNG (via Pillow,
  already available), SVG (plain text, no library needed), OBJ (via
  scikit-image's `marching_cubes`, already available), a hand-written
  minimal DXF writer (ASCII DXF is a documented, human-writable format;
  validated structurally — see §4), JSON manifests, and a documented TCP
  wire protocol (validated end-to-end with a real client/server test —
  see §4).
- **Documented proxy, not the real format:** `VolumetricExporter` does
  **not** write real OpenVDB — OpenVDB's binary format is a
  hierarchical sparse-tree structure that isn't reasonably hand-writable
  without the actual library. It writes a simple `.npz` grid instead,
  plus a ready-to-run Python conversion script
  (`to_vdb_conversion_snippet()`) for use in an environment that does
  have `openvdb` installed (e.g. a Houdini Python environment).
- **Not attempted at all:** native `.ai` (Illustrator) and `.cdr`
  (CorelDRAW) files are undocumented proprietary binary formats — SVG is
  used instead, which both applications import natively. FBX and
  Alembic (`.abc`) are complex binary formats also not attempted — OBJ
  is the dependency-free common denominator all four 3D
  applications/engines import directly without a plugin.

## 3. `simulation_export_bridge_one.py` — wiring each module in

### Section A: hazard modules

| Adapter | Source module | Produces |
|---|---|---|
| `SeismicExportAdapter` | `seismic_one.py` | Liquefaction FS-vs-depth profile (SVG boring-log style + DXF cross-section), story-drift building elevation (DXF, colored by damage index), response spectrum (SVG) |
| `FireExportAdapter` | `fire_one.py` / `fire_dns_coupling_one.py` | Fire-source volumetric grid, plume temperature time-height PNG, HRR(t) curve (SVG) |
| `FloodExportAdapter` | `flood_one.py` | Inundation depth map (PNG), flood-extent boundary (SVG + DXF), damage heatmap (PNG) |
| `StormExportAdapter` | `storm_one.py` | Tornado/hurricane wind-speed field (PNG heatmap + SVG isotach contours) |
| `VolcanoExportAdapter` | `volcano_one.py` | PDC hazard-zone circle (DXF + SVG, radially symmetric per the source model's own documented limitation) |

### Section B: DNS cluster modules

| Adapter | Source module | Notes |
|---|---|---|
| `SuperDNSExportAdapter` | `super_dns_one_v6_3.py` (`CompressibleSolver`) | Reads whichever fields are actually active (`rho` always; `Z`/`Y_soot` only if `enable_combustion`/`enable_soot` are set) rather than assuming every field exists. Includes a flame-surface OBJ isosurface at `Z = z_stoich`, and `stream_live_state()` for pushing frames to a running `LiveDataStreamServer` during an active simulation. |
| `CahnHilliardExportAdapter` | `structural_cahn_hilliard_3d_v3.py` | Works identically for `StructuralCahnHilliard3D`, `ThinFilmStructuralCahnHilliard3D`, and `PhaseFieldCrystal3D` — all three share the same order-parameter field convention, confirmed by inspecting the source rather than assumed. **Note:** "ThinFilm 3D" and "PFC 3D" are not separate files; they are subclasses within `structural_cahn_hilliard_3d_v3.py`. |
| `FluctuatingHydroExportAdapter` (FH) | `structuralfluctuatinghydro_v6_3.py` | Exports density directly and velocity as its scalar **magnitude** (a vector field isn't directly renderable as one image); for true vector-field visualization in a DCC tool, export the three velocity components as separate `.npz` grids and reconstruct on the DCC side. |
| `LangevinExportAdapter` | `structural_langevin_v3.py` | Structural stress field snapshot. |
| `FNOPredictionExportAdapter` | `structural_fno_3d_v2_3.py` | Exports both the predicted mean AND the predictive uncertainty (derived from `log_var`) as separate heatmaps per channel — useful for visualizing where the surrogate model is confident vs. uncertain. Also exports full `forward_rollout()` / AGI ONE's `predict_fire_scenario()` trajectories as a Unity/Unreal-ready JSON time series. |

**Architectural note:** `CompressibleSolver` is a stateful object (holds
`self.rho`, `self.Z`, etc. as live attributes), so
`SuperDNSExportAdapter` takes a solver *instance*. The Cahn-Hilliard/FH/
Langevin modules are functional (`.step(field, ...) -> field`), so their
adapters take the field *tensor* directly rather than a solver object —
this difference reflects each source module's own actual architecture,
confirmed by reading each file rather than assumed to be uniform.

## 4. Validation summary

All 12 `dcc_export_one.py` format tests and all hazard/DNS-cluster
adapter tests in `simulation_export_bridge_one.py` were run and passed
in this development session:

| Check | Result |
|---|---|
| PNG round-trips through Pillow with correct size | Verified |
| PNG sequence: all frames written | Verified |
| 16-bit heightmap: correct image mode | Verified |
| SVG contours/polygons: well-formed tags, correct element counts | Verified |
| DXF: group-code structure valid (every code line parses as an integer) | Verified across all 90 lines of a multi-entity test file |
| DXF: all four entity types (LINE, CIRCLE, LWPOLYLINE, TEXT) present and structurally correct | Verified |
| OBJ marching-cubes isosurface: sphere test gives mean vertex radius 9.996 (expected 10) | Verified |
| OBJ point cloud: correct point count | Verified |
| Volumetric `.npz`: round-trips exactly | Verified |
| VDB conversion snippet: generated correctly (not executed — no `openvdb` available) | Verified (generation only) |
| JSON timeseries + binary frame data: round-trips exactly | Verified |
| Live TCP streaming: real client/server test, wire protocol parsed correctly, data matches exactly | Verified end-to-end |
| `SeismicExportAdapter`: liquefaction profile, story-drift diagram | Verified against real `seismic_one.py` output |
| `FireExportAdapter`: HRR curve, plume profile | Verified against real `fire_one.py` output |
| `FloodExportAdapter`: inundation map, boundary, damage heatmap | Verified against real `flood_one.py` output |
| `StormExportAdapter`: tornado and hurricane wind fields | Verified against real `storm_one.py` output |
| `VolcanoExportAdapter`: PDC hazard zone | Verified against real `volcano_one.py` output |
| DNS cluster adapters (CH3D, FH, Langevin, FNO, SuperDNS) | Verified with numpy-array mocks standing in for torch tensors (no GPU/torch available in this session — see §5) |

## 5. Known gaps / before production use

- **No real OpenVDB support** — see §2. The `.npz` proxy + conversion
  snippet path has not been executed end-to-end (no `openvdb` package
  available to test the actual conversion in this session).
- **DNS cluster adapters tested with numpy mocks only** — this session
  had no GPU/torch available, so `CahnHilliardExportAdapter`,
  `FluctuatingHydroExportAdapter`, `LangevinExportAdapter`,
  `FNOPredictionExportAdapter`, and `SuperDNSExportAdapter` were
  validated with plain numpy arrays standing in for `torch.Tensor`
  inputs (the `_to_numpy()` conversion path itself is simple and
  low-risk, but the full pipeline has not been run against a live
  GPU-resident simulation). Run a real smoke test before production use.
- **`ezdxf`-free DXF writer is minimal** — only LINE, LWPOLYLINE,
  CIRCLE, and TEXT entities, a single default layer, no blocks, no true
  3D solids. Sufficient for this module's own hazard-zone/contour/
  building-elevation exports, not a general-purpose DXF library.
- **FH velocity export is magnitude-only** — see §3's architectural
  note; a true vector-field DCC visualization needs the three
  component grids exported separately and reconstructed on the
  receiving side.

---

*Co-developed with Claude (Anthropic), as an AI co-developer, consistent
with the ONE Ecosystem attribution convention.*
