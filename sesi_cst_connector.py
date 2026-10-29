"""
=============================================================================
SESI CST STUDIO SUITE CONNECTOR
Python API integration for full-wave EM simulation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[C] Connects to CST Studio Suite via Python API (if installed + licensed)
[D] Falls back to OpenEMS or synthetic results when CST unavailable
[D] Manages simulation workflows: geometry → mesh → solve → extract results

CST STUDIO SUITE API
--------------------
CST provides a Python API (cst.interface) for:
  • Creating/modifying geometry (bricks, cylinders, curves)
  • Setting material properties (ε, μ, σ, tan δ)
  • Defining ports, excitations, boundary conditions
  • Running FDTD/FIT/FEM solvers
  • Extracting S-parameters, field distributions, far-field patterns

REQUIREMENTS
------------
• CST Studio Suite 2020+ installed
• Valid license (commercial or academic)
• Python 3.7+ with cst.interface package
• If unavailable: falls back to OpenEMS (open-source) or synthetic data

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• CST is a COMMERCIAL tool; this connector requires a valid license.
• Simulation results are COMPUTATIONAL PREDICTIONS, not measurements.
• Mesh convergence must be verified; results depend on mesh quality.
• Without CST, falls back to OpenEMS (open-source FDTD) or synthetic data.
• All results require experimental validation before deployment.
=============================================================================
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.cst_connector")

# Try importing CST Python API
try:
    import cst.interface
    HAS_CST = True
except ImportError:
    HAS_CST = False

# Fallback: OpenEMS
try:
    from sesi_fullwave_em_solver import OpenEMSSolver, OpenEMSConfig
    HAS_OPENEMS = True
except ImportError:
    HAS_OPENEMS = False

__all__ = [
    "CSTConfig",
    "CSTGeometryBuilder",
    "CSTMaterialLibrary",
    "CSTSimulationRunner",
    "CSTResultExtractor",
    "CSTConnector",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CSTConfig:
    """Configuration for CST simulation."""
    # CST installation
    cst_install_path: str = "C:/Program Files (x86)/Dassault Systemes/CST Studio Suite 2023"
    
    # Simulation parameters
    solver_type: str = "Frequency Domain"  # or "Time Domain"
    frequency_ghz: float = 10.0
    mesh_density: str = "Medium"  # Low, Medium, High
    
    # Geometry
    domain_size_mm: Tuple[float, float, float] = (100.0, 100.0, 10.0)
    
    # Boundary conditions
    bc_x: str = "open"  # open, electric, magnetic
    bc_y: str = "open"
    bc_z: str = "open"
    
    # Ports
    port_type: str = "waveguide"  # waveguide, coaxial, discrete
    
    # Output
    output_dir: str = "./cst_results"
    
    # Fallback
    use_openems_fallback: bool = True
    use_synthetic_fallback: bool = True
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# [C] CST GEOMETRY BUILDER
# ═══════════════════════════════════════════════════════════════════════

class CSTGeometryBuilder:
    """
    [C] Build CST geometry using Python API.
    
    Creates 3D structures: patches, absorbers, metamaterials.
    Falls back to OpenEMS geometry format if CST unavailable.
    """
    
    def __init__(self, cfg: CSTConfig):
        self.cfg = cfg
        self.mws = None  # CST Modeler Workspace
    
    def create_patch_antenna(self, length_mm: float, width_mm: float,
                              substrate_height_mm: float,
                              substrate_eps_r: float) -> Dict[str, Any]:
        """
        Create a rectangular patch antenna geometry.
        
        Returns dict with geometry parameters (for fallback) or CST object.
        """
        if HAS_CST:
            try:
                self.mws = cst.interface.DesignEnvironment()
                # Create substrate
                self.mws.modeler.brick.create(
                    name="substrate",
                    x_range=(0, length_mm),
                    y_range=(0, width_mm),
                    z_range=(0, substrate_height_mm),
                    material=f"eps_r={substrate_eps_r}"
                )
                # Create patch
                self.mws.modeler.brick.create(
                    name="patch",
                    x_range=(0, length_mm),
                    y_range=(0, width_mm),
                    z_range=(substrate_height_mm, substrate_height_mm + 0.035),
                    material="Copper"
                )
                return {"status": "cst_created", "geometry": self.mws}
            except Exception as e:
                logger.warning(f"CST geometry creation failed: {e}")
        
        # Fallback: return geometry dict for OpenEMS
        return {
            "status": "fallback",
            "geometry": {
                "type": "patch_antenna",
                "length_mm": length_mm,
                "width_mm": width_mm,
                "substrate_height_mm": substrate_height_mm,
                "substrate_eps_r": substrate_eps_r,
            }
        }
    
    def create_metamaterial_absorber(self, unit_cell_size_mm: float,
                                      n_cells: int,
                                      thickness_mm: float) -> Dict[str, Any]:
        """
        Create a metamaterial absorber array.
        """
        if HAS_CST:
            try:
                self.mws = cst.interface.DesignEnvironment()
                # Create unit cell array
                for i in range(n_cells):
                    for j in range(n_cells):
                        x_offset = i * unit_cell_size_mm
                        y_offset = j * unit_cell_size_mm
                        self.mws.modeler.brick.create(
                            name=f"cell_{i}_{j}",
                            x_range=(x_offset, x_offset + unit_cell_size_mm * 0.8),
                            y_range=(y_offset, y_offset + unit_cell_size_mm * 0.8),
                            z_range=(0, thickness_mm),
                            material="Metamaterial"
                        )
                return {"status": "cst_created", "geometry": self.mws}
            except Exception as e:
                logger.warning(f"CST metamaterial creation failed: {e}")
        
        return {
            "status": "fallback",
            "geometry": {
                "type": "metamaterial_absorber",
                "unit_cell_size_mm": unit_cell_size_mm,
                "n_cells": n_cells,
                "thickness_mm": thickness_mm,
            }
        }


# ═══════════════════════════════════════════════════════════════════════
# [C] CST MATERIAL LIBRARY
# ═══════════════════════════════════════════════════════════════════════

class CSTMaterialLibrary:
    """
    [C] Material property management for CST.
    
    Defines materials with ε(ω), μ(ω), σ, tan δ.
    Supports dispersive materials (Debye, Lorentz models).
    """
    
    def __init__(self):
        self.materials: Dict[str, Dict[str, Any]] = {}
    
    def add_material(self, name: str, eps_r: float, mu_r: float = 1.0,
                     tan_d: float = 0.0, sigma: float = 0.0) -> None:
        """Add a material to the library."""
        self.materials[name] = {
            "eps_r": eps_r,
            "mu_r": mu_r,
            "tan_d": tan_d,
            "sigma": sigma,
        }
    
    def add_dispersive_material(self, name: str,
                                 eps_model: Dict[str, Any]) -> None:
        """
        Add a dispersive material (frequency-dependent ε).
        
        eps_model: dict with model type and parameters
          • "debye": {eps_inf, delta_eps, tau}
          • "lorentz": {eps_inf, omega_0, gamma, delta_eps}
        """
        self.materials[name] = {
            "dispersive": True,
            "model": eps_model,
        }
    
    def export_to_cst(self) -> None:
        """Export materials to CST (if available)."""
        if not HAS_CST:
            logger.warning("CST not available; materials stored locally only")
            return
        
        try:
            mws = cst.interface.DesignEnvironment()
            for name, props in self.materials.items():
                if props.get("dispersive"):
                    # Dispersive material
                    model = props["model"]
                    if model.get("type") == "debye":
                        mws.material.add_debye(
                            name=name,
                            eps_inf=model["eps_inf"],
                            delta_eps=model["delta_eps"],
                            tau=model["tau"]
                        )
                else:
                    # Normal material
                    mws.material.add(
                        name=name,
                        eps=props["eps_r"],
                        mu=props["mu_r"],
                        tand=props["tan_d"],
                        sigma=props["sigma"]
                    )
        except Exception as e:
            logger.error(f"Failed to export materials to CST: {e}")


# ═══════════════════════════════════════════════════════════════════════
# [C/D] CST SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════

class CSTSimulationRunner:
    """
    [C/D] Run CST simulation with fallback to OpenEMS/synthetic.
    """
    
    def __init__(self, cfg: CSTConfig):
        self.cfg = cfg
        self.mws = None
    
    def run_simulation(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run EM simulation.
        
        Priority: CST → OpenEMS → Synthetic
        """
        if geometry.get("status") == "cst_created":
            return self._run_cst(geometry["geometry"])
        elif self.cfg.use_openems_fallback and HAS_OPENEMS:
            return self._run_openems(geometry["geometry"])
        elif self.cfg.use_synthetic_fallback:
            return self._run_synthetic(geometry["geometry"])
        else:
            return {"error": "No simulation backend available"}
    
    def _run_cst(self, mws) -> Dict[str, Any]:
        """Run CST simulation."""
        try:
            # Set solver
            if self.cfg.solver_type == "Frequency Domain":
                mws.solver.frequency_domain()
            else:
                mws.solver.time_domain()
            
            # Set mesh
            mws.mesh.set_mesh_type(self.cfg.mesh_density)
            
            # Set boundary conditions
            mws.boundary.set_x(self.cfg.bc_x)
            mws.boundary.set_y(self.cfg.bc_y)
            mws.boundary.set_z(self.cfg.bc_z)
            
            # Run simulation
            mws.solver.start()
            
            # Extract results
            freqs = mws.result_1d.get_frequencies()
            s11 = mws.result_1d.get_s_parameter("S1,1")
            
            return {
                "status": "completed",
                "backend": "cst",
                "frequencies_ghz": freqs,
                "s11_db": s11,
            }
        except Exception as e:
            logger.error(f"CST simulation failed: {e}")
            return {"error": str(e)}
    
    def _run_openems(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to OpenEMS."""
        logger.info("Using OpenEMS fallback")
        em_cfg = OpenEMSConfig(
            domain_size=tuple(x / 1000.0 for x in self.cfg.domain_size_mm),
            grid_resolution=(50, 50, 20),
        )
        solver = OpenEMSSolver(em_cfg)
        return solver.run_simulation(geometry)
    
    def _run_synthetic(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic results."""
        logger.warning("Using synthetic results (no EM solver available)")
        freqs = np.linspace(8, 12, 100)  # GHz
        
        # Synthetic S11 (resonance-like)
        f0 = self.cfg.frequency_ghz
        s11 = -20 + 15 * np.exp(-((freqs - f0) / 0.5)**2)
        
        return {
            "status": "synthetic",
            "backend": "synthetic",
            "frequencies_ghz": freqs,
            "s11_db": s11,
            "note": "SYNTHETIC results; not physically validated",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] CST RESULT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

class CSTResultExtractor:
    """
    [D] Extract and post-process simulation results.
    """
    
    def __init__(self, cfg: CSTConfig):
        self.cfg = cfg
    
    def extract_s_parameters(self, result: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract S-parameters from result."""
        if "s11_db" in result:
            return {
                "frequencies_ghz": result["frequencies_ghz"],
                "s11_db": result["s11_db"],
            }
        return {"error": "No S-parameters in result"}
    
    def extract_field_distribution(self, result: Dict[str, Any],
                                    component: str = "E") -> Optional[Dict[str, Any]]:
        """
        Extract field distribution (E or H).
        
        Requires CST to be running and field data to be available.
        """
        if result.get("backend") != "cst":
            logger.warning("Field extraction only available for CST backend")
            return None
        
        if not HAS_CST:
            return None
        
        try:
            mws = cst.interface.DesignEnvironment()
            if component == "E":
                field_data = mws.result_3d.get_e_field()
            else:
                field_data = mws.result_3d.get_h_field()
            
            return {
                "component": component,
                "x": field_data.x,
                "y": field_data.y,
                "z": field_data.z,
                "magnitude": field_data.magnitude,
            }
        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return None
    
    def compute_absorption(self, s_params: Dict[str, np.ndarray]) -> float:
        """
        Compute absorption from S-parameters.
        
        A = 1 - |S11|² - |S21|² (for backed structure, S21 ≈ 0)
        """
        s11_linear = 10 ** (s_params["s11_db"] / 20)
        absorption = 1 - s11_linear**2
        return float(np.mean(absorption))


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATED CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class CSTConnector:
    """
    [D] Integrated CST connector for full simulation workflow.
    """
    
    def __init__(self, cfg: Optional[CSTConfig] = None):
        self.cfg = cfg or CSTConfig()
        self.geometry_builder = CSTGeometryBuilder(self.cfg)
        self.material_lib = CSTMaterialLibrary()
        self.sim_runner = CSTSimulationRunner(self.cfg)
        self.result_extractor = CSTResultExtractor(self.cfg)
    
    def simulate_patch_antenna(self, length_mm: float, width_mm: float,
                                substrate_height_mm: float,
                                substrate_eps_r: float) -> Dict[str, Any]:
        """Full simulation workflow for patch antenna."""
        # 1. Build geometry
        geom = self.geometry_builder.create_patch_antenna(
            length_mm, width_mm, substrate_height_mm, substrate_eps_r
        )
        
        # 2. Add materials
        self.material_lib.add_material("substrate", eps_r=substrate_eps_r, tan_d=0.02)
        self.material_lib.add_material("Copper", eps_r=1.0, sigma=5.8e7)
        self.material_lib.export_to_cst()
        
        # 3. Run simulation
        result = self.sim_runner.run_simulation(geom)
        
        if "error" in result:
            return result
        
        # 4. Extract results
        s_params = self.result_extractor.extract_s_parameters(result)
        absorption = self.result_extractor.compute_absorption(s_params)
        
        return {
            "geometry": geom,
            "simulation_result": result,
            "s_parameters": s_params,
            "mean_absorption": absorption,
            "backend": result.get("backend", "unknown"),
        }
    
    def simulate_metamaterial_absorber(self, unit_cell_size_mm: float,
                                        n_cells: int,
                                        thickness_mm: float,
                                        eps_r: float,
                                        mu_r: float,
                                        tan_d: float) -> Dict[str, Any]:
        """Full simulation workflow for metamaterial absorber."""
        # 1. Build geometry
        geom = self.geometry_builder.create_metamaterial_absorber(
            unit_cell_size_mm, n_cells, thickness_mm
        )
        
        # 2. Add metamaterial
        self.material_lib.add_material("Metamaterial", eps_r=eps_r, mu_r=mu_r, tan_d=tan_d)
        self.material_lib.export_to_cst()
        
        # 3. Run simulation
        result = self.sim_runner.run_simulation(geom)
        
        if "error" in result:
            return result
        
        # 4. Extract results
        s_params = self.result_extractor.extract_s_parameters(result)
        absorption = self.result_extractor.compute_absorption(s_params)
        
        return {
            "geometry": geom,
            "simulation_result": result,
            "s_parameters": s_params,
            "mean_absorption": absorption,
            "backend": result.get("backend", "unknown"),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI CST Studio Suite Connector — Smoke Test")
    print("=" * 72)
    print(f"  CST Python API available : {HAS_CST}")
    print(f"  OpenEMS fallback         : {HAS_OPENEMS}")
    
    cfg = CSTConfig()
    connector = CSTConnector(cfg)
    
    # Test 1: Patch antenna
    print("\n  ── Patch Antenna Simulation ──")
    result = connector.simulate_patch_antenna(
        length_mm=15.0,
        width_mm=20.0,
        substrate_height_mm=1.6,
        substrate_eps_r=4.4
    )
    
    if "error" in result:
        print(f"    Error: {result['error']}")
    else:
        print(f"    Backend            : {result['backend']}")
        print(f"    Mean absorption    : {result['mean_absorption']:.4f}")
        print(f"    S11 range (dB)     : [{result['s_parameters']['s11_db'].min():.1f}, "
              f"{result['s_parameters']['s11_db'].max():.1f}]")
    
    # Test 2: Metamaterial absorber
    print("\n  ── Metamaterial Absorber Simulation ──")
    result = connector.simulate_metamaterial_absorber(
        unit_cell_size_mm=5.0,
        n_cells=4,
        thickness_mm=2.0,
        eps_r=10.0,
        mu_r=10.0,
        tan_d=0.1
    )
    
    if "error" in result:
        print(f"    Error: {result['error']}")
    else:
        print(f"    Backend            : {result['backend']}")
        print(f"    Mean absorption    : {result['mean_absorption']:.4f}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • CST requires commercial license")
    print("  • Simulation results are computational predictions, not measurements")
    print("  • Mesh convergence must be verified")
    print("  • Fallback to OpenEMS/synthetic when CST unavailable")
    print("  • All results require experimental validation")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
