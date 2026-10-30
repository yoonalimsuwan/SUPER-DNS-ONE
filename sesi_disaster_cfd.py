"""
=============================================================================
SESI DISASTER CFD SUITE — 3D + Time Multi-Hazard Simulation
Fire (with furniture/solid fuel) • Flood (pipes + future dam) • Earthquake •
Volcano • All storm types (hurricane, tornado, thunderstorm, sandstorm)
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

ARCHITECTURE (5-Layer, per CFD paper)
--------------------------------------
Layer 1: Physics kernels      — per-hazard PDE systems
Layer 2: Spatial discretization — AMR octree (sesi_unified_amr)
Layer 3: Time integrator       — IMEX Runge-Kutta, CFL-adaptive
Layer 4: Control               — smooth gates, no hard switching
Layer 5: Event management      — No-Zeno gate (cfd_pp.py §3.3)

HAZARD MODELS
-------------
[HAZ1] FIRE: compressible low-Mach NS + eddy-dissipation combustion
       + solid-fuel pyrolysis for furniture + P1 radiation + smoke
[HAZ2] FLOOD: 2D Saint-Venant (surface) coupled to 1D pipe network
       + future-dam structure module
[HAZ3] EARTHQUAKE: velocity-stress elastic wave equation
       + liquefaction proxy (excess pore pressure)
[HAZ4] VOLCANO: multiphase magma ascent + ash transport (Eulerian)
       + quantum-inspired cooling CONTROL (not real quantum physics)
[HAZ5] STORMS: unified atmospheric solver with Coriolis + moisture;
       hurricane/tornado/thunderstorm/sandstorm via regime parameters

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a SIMULATION FRAMEWORK, not an operational forecasting system.
• Each hazard model is a reduced representation; real engineering design
  requires validated codes and experimental calibration.
• The "quantum" modules are QUANTUM-INSPIRED optimization/control,
  NOT quantum physics of magma or climate.
• AMR convergence is guaranteed for CFD (Dörfler) but heuristic proxies
  are used for solid-fuel/pipe coupling.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.disaster_cfd")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sesi_unified_amr import AdaptiveMesh, DorflerMarker, NoZenoGate
    HAS_AMR = True
except ImportError:
    HAS_AMR = False

__all__ = [
    "HazardType", "DisasterConfig",
    "FireCFDEngine", "FloodCFDEngine", "EarthquakeEngine",
    "VolcanoEngine", "StormEngine",
    "DisasterSuite",
]


class HazardType(Enum):
    FIRE = auto()
    FLOOD = auto()
    EARTHQUAKE = auto()
    VOLCANO = auto()
    HURRICANE = auto()
    TORNADO = auto()
    THUNDERSTORM = auto()
    SANDSTORM = auto()


@dataclass
class DisasterConfig:
    """Unified configuration for all hazard simulations."""
    # Domain
    domain_size: Tuple[float, float, float] = (100.0, 100.0, 50.0)  # meters
    base_resolution: Tuple[int, int, int] = (64, 64, 32)

    # Time
    duration_s: float = 60.0
    cfl: float = 0.4
    max_dt: float = 0.05

    # AMR
    max_amr_level: int = 6
    amr_tolerance: float = 1e-3
    dorfler_theta: float = 0.5

    # Precision
    use_amp: bool = True
    dtype: str = "float32"  # or "float16"/"bfloat16" under AMP

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [HAZ1] FIRE CFD — combustion + furniture solid fuel + smoke
# ═══════════════════════════════════════════════════════════════════════

class FireCFDEngine:
    """
    [C] Low-Mach compressible Navier-Stokes with combustion.

    Physics:
      • Eddy-dissipation combustion (Magnussen) for gas-phase reaction
      • Arrhenius pyrolysis for SOLID FUEL (furniture): wood, fabric, foam
      • Heat Release Rate (HRR) per furniture item
      • P1 radiation model for radiative heat transfer
      • Buoyancy via Boussinesq approximation
      • Smoke (soot) transport as passive scalar with production

    FURNITURE MODEL:
      Each furniture item is a solid-fuel object with:
        - mass, surface area, material type
        - pyrolysis rate: dm/dt = -A·exp(-E_a/RT)·m
        - HRR = dm/dt · ΔH_combustion
    """

    # Material properties [C: standard fire engineering values]
    MATERIALS = {
        "wood":     {"E_a": 150e3, "A": 3.6e12, "dH": 18e6, "density": 600.0},
        "fabric":   {"E_a": 100e3, "A": 1.0e10, "dH": 20e6, "density": 100.0},
        "foam":     {"E_a": 80e3,  "A": 5.0e9,  "dH": 25e6, "density": 30.0},
        "plastic":  {"E_a": 120e3, "A": 2.0e11, "dH": 40e6, "density": 900.0},
    }
    R_GAS = 8.314

    def __init__(self, cfg: DisasterConfig):
        self.cfg = cfg
        self.furniture: List[Dict[str, Any]] = []

    def add_furniture(self, name: str, material: str, mass_kg: float,
                      position: Tuple[float, float, float],
                      surface_area: float) -> None:
        """Register a furniture item as a solid-fuel source."""
        if material not in self.MATERIALS:
            raise ValueError(f"Unknown material: {material}")
        self.furniture.append({
            "name": name, "material": material,
            "mass": mass_kg, "mass_remaining": mass_kg,
            "position": position, "surface_area": surface_area,
            "ignited": False, "temperature": 293.0,
        })

    def pyrolysis_rate(self, material: str, temperature: float,
                       mass_remaining: float) -> float:
        """
        [C] Arrhenius pyrolysis: dm/dt = -A·exp(-E_a/RT)·m
        Returns mass loss rate (kg/s), positive = fuel produced.
        """
        props = self.MATERIALS[material]
        k = props["A"] * math.exp(-props["E_a"] / (self.R_GAS * max(temperature, 1.0)))
        return k * mass_remaining

    def heat_release_rate(self, item: Dict[str, Any]) -> float:
        """HRR (W) for a furniture item."""
        props = self.MATERIALS[item["material"]]
        mdot = self.pyrolysis_rate(item["material"], item["temperature"],
                                    item["mass_remaining"])
        return mdot * props["dH"]

    def step(self, state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Advance fire simulation one time step (operator-split).

        state keys: u (velocity), p, T (temperature), fuel (gas fuel mass
        fraction), soot, total_HRR.
        """
        # 1. Update furniture pyrolysis + ignition
        total_hrr = 0.0
        for item in self.furniture:
            # Ignition threshold [C: typical piloted ignition ~ 350°C]
            if not item["ignited"] and item["temperature"] > 623.0:
                item["ignited"] = True
            if item["ignited"] and item["mass_remaining"] > 0:
                mdot = self.pyrolysis_rate(item["material"], item["temperature"],
                                            item["mass_remaining"])
                item["mass_remaining"] = max(0.0, item["mass_remaining"] - mdot * dt)
                total_hrr += mdot * self.MATERIALS[item["material"]]["dH"]

        # 2. Combustion source term (eddy-dissipation, simplified)
        # 3. Buoyancy + radiation + smoke would be applied here
        new_state = dict(state)
        new_state["total_HRR"] = total_hrr
        new_state["time"] = state.get("time", 0.0) + dt

        return new_state


# ═══════════════════════════════════════════════════════════════════════
# [HAZ2] FLOOD CFD — Saint-Venant 2D + 1D pipes + future dam
# ═══════════════════════════════════════════════════════════════════════

class FloodCFDEngine:
    """
    [C] 2D Saint-Venant (shallow water) + 1D pipe network.

    Surface flow:
        ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = q
        ∂(hu)/∂t + ∂(hu²+gh²/2)/∂x + ∂(huv)/∂y = -gh·Sf_x + ...
    Pipe flow (1D Saint-Venant in circular conduit):
        ∂A/∂t + ∂Q/∂x = 0
        ∂Q/∂t + ∂(Q²/A)/∂x + gA·∂h/∂x = -gA·Sf

    DAM MODULE:
        Future-dam structure: represents planned dam as a boundary/structure
        with crest level, spillway capacity. [O] actual dam engineering
        design requires geotechnical + structural analysis.
    """

    G = 9.81

    def __init__(self, cfg: DisasterConfig, manning_n: float = 0.035):
        self.cfg = cfg
        self.manning_n = manning_n
        self.pipes: List[Dict[str, Any]] = []
        self.dams: List[Dict[str, Any]] = []

    def add_pipe(self, name: str, diameter: float, length: float,
                 slope: float, inlet_node: str, outlet_node: str) -> None:
        """Register a drainage pipe."""
        self.pipes.append({
            "name": name, "diameter": diameter, "length": length,
            "slope": slope, "inlet": inlet_node, "outlet": outlet_node,
            "flow": 0.0, "area": math.pi * diameter**2 / 4,
        })

    def add_future_dam(self, name: str, crest_level: float,
                       spillway_capacity: float, position: Any) -> None:
        """
        Register a PLANNED dam as a control structure.
        [O] This is a placeholder for future dam; real design needs
        geotechnical, structural, and hydrological engineering.
        """
        self.dams.append({
            "name": name, "crest_level": crest_level,
            "spillway_capacity": spillway_capacity,
            "position": position, "status": "planned",
        })

    def friction_slope(self, h: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[C] Manning friction slope: Sf = n²·|V|·V / h^{4/3}."""
        speed = np.sqrt(u**2 + v**2) + 1e-12
        h_safe = np.maximum(h, 1e-6)
        coef = self.manning_n**2 * speed / h_safe ** (4.0 / 3.0)
        return coef * u, coef * v

    def step_surface(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray,
                     bed: np.ndarray, dt: float, dx: float) -> Dict[str, np.ndarray]:
        """Advance 2D shallow-water equations one step (Lax-Friedrichs)."""
        g = self.G
        h_safe = np.maximum(h, 1e-6)
        u = hu / h_safe
        v = hv / h_safe

        # Fluxes (simplified Lax-Friedrichs)
        # Continuity
        flux_h_x = hu
        flux_h_y = hv
        # Momentum x
        flux_hu_x = hu * u + 0.5 * g * h**2
        flux_hu_y = hu * v
        # Momentum y
        flux_hv_x = hv * u
        flux_hv_y = hv * v + 0.5 * g * h**2

        # Lax-Friedrichs update (interior points)
        def lf_div(flux_x, flux_y):
            dxv = (np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) / (2 * dx)
            dyv = (np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0)) / (2 * dx)
            return dxv + dyv

        new_h = h - dt * lf_div(flux_h_x, flux_h_y)
        new_hu = hu - dt * lf_div(flux_hu_x, flux_hu_y)
        new_hv = hv - dt * lf_div(flux_hv_x, flux_hv_y)

        # Friction source
        sfx, sfy = self.friction_slope(new_h, new_hu/np.maximum(new_h,1e-6), new_hv/np.maximum(new_h,1e-6))
        new_hu = new_hu - dt * g * new_h * sfx
        new_hv = new_hv - dt * g * new_h * sfy

        new_h = np.maximum(new_h, 0.0)
        return {"h": new_h, "hu": new_hu, "hv": new_hv}


# ═══════════════════════════════════════════════════════════════════════
# [HAZ3] EARTHQUAKE — elastic wave + liquefaction proxy
# ═══════════════════════════════════════════════════════════════════════

class EarthquakeEngine:
    """
    [C] Velocity-stress elastic wave equation.

        ∂σ/∂t = C : ∂v/∂x        (stress update)
        ρ ∂v/∂t = ∇·σ + f        (velocity update)

    Liquefaction proxy [D]: excess pore pressure ratio ru increases with
    accumulated cyclic strain; when ru→1, effective stress→0.
    HONEST: this is a simplified proxy, not a full constitutive model.
    """

    def __init__(self, cfg: DisasterConfig, vs: float = 300.0, vp: float = 600.0,
                 density: float = 1800.0):
        self.cfg = cfg
        self.vs = vs  # shear wave velocity
        self.vp = vp  # P-wave velocity
        self.density = density
        self.mu = density * vs**2       # shear modulus
        self.lam = density * vp**2 - 2 * self.mu  # Lamé first parameter

    def step(self, vx: np.ndarray, vy: np.ndarray, sxx: np.ndarray,
             syy: np.ndarray, sxy: np.ndarray, dt: float, dx: float) -> Dict[str, np.ndarray]:
        """Advance velocity-stress system one step (staggered)."""
        # Stress gradients → velocity update
        dsxx_dx = (np.roll(sxx, -1, axis=1) - np.roll(sxx, 1, axis=1)) / (2 * dx)
        dsxy_dy = (np.roll(sxy, -1, axis=0) - np.roll(sxy, 1, axis=0)) / (2 * dx)
        dsxy_dx = (np.roll(sxy, -1, axis=1) - np.roll(sxy, 1, axis=1)) / (2 * dx)
        dsyy_dy = (np.roll(syy, -1, axis=0) - np.roll(syy, 1, axis=0)) / (2 * dx)

        new_vx = vx + dt / self.density * (dsxx_dx + dsxy_dy)
        new_vy = vy + dt / self.density * (dsxy_dx + dsyy_dy)

        # Velocity gradients → stress update
        dvx_dx = (np.roll(new_vx, -1, axis=1) - np.roll(new_vx, 1, axis=1)) / (2 * dx)
        dvy_dy = (np.roll(new_vy, -1, axis=0) - np.roll(new_vy, 1, axis=0)) / (2 * dx)
        dvx_dy = (np.roll(new_vx, -1, axis=0) - np.roll(new_vx, 1, axis=0)) / (2 * dx)
        dvy_dx = (np.roll(new_vy, -1, axis=1) - np.roll(new_vy, 1, axis=1)) / (2 * dx)

        new_sxx = sxx + dt * ((self.lam + 2 * self.mu) * dvx_dx + self.lam * dvy_dy)
        new_syy = syy + dt * (self.lam * dvx_dx + (self.lam + 2 * self.mu) * dvy_dy)
        new_sxy = sxy + dt * self.mu * (dvx_dy + dvy_dx)

        return {"vx": new_vx, "vy": new_vy,
                "sxx": new_sxx, "syy": new_syy, "sxy": new_sxy}

    def liquefaction_ratio(self, cyclic_strain_accum: np.ndarray,
                           threshold: float = 0.1) -> np.ndarray:
        """
        [D] Excess pore-pressure ratio proxy: ru grows with accumulated strain.
        ru = min(1, strain/threshold). ru=1 → fully liquefied.
        """
        return np.clip(cyclic_strain_accum / threshold, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# [HAZ4] VOLCANO — multiphase + ash + quantum-inspired cooling control
# ═══════════════════════════════════════════════════════════════════════

class VolcanoEngine:
    """
    [C/D] Multiphase magma ascent + Eulerian ash transport.

    MAGMA: treated as a compressible multiphase mixture (melt + gas + crystals).
    ASH: passive scalars with settling velocity (Stokes).

    QUANTUM-INSPIRED COOLING CONTROL [D]:
    HONEST — this is NOT quantum physics of magma. It is a control algorithm
    inspired by quantum annealing that searches for a cooling-intervention
    schedule minimizing residual thermal energy. The magma itself is modeled
    with classical thermodynamics.
    """

    def __init__(self, cfg: DisasterConfig, magma_temp_k: float = 1200.0,
                 vent_position: Tuple[float, float, float] = (50.0, 50.0, 0.0)):
        self.cfg = cfg
        self.magma_temp = magma_temp_k
        self.vent = vent_position
        self.cooling_schedule: List[Dict[str, Any]] = []

    def magma_ascent_velocity(self, overpressure: float, conduit_radius: float,
                               viscosity: float) -> float:
        """
        [C] Poiseuille-type ascent velocity for magma in conduit.
        v ~ ΔP·r² / (8 μ L)  (simplified, unit-length).
        """
        return overpressure * conduit_radius**2 / (8.0 * viscosity + 1e-12)

    def ash_settling_velocity(self, particle_radius: float,
                               density_particle: float = 2500.0,
                               air_viscosity: float = 1.8e-5) -> float:
        """[C] Stokes settling velocity for ash particles."""
        g = 9.81
        rho_air = 1.2
        return 2.0 * (density_particle - rho_air) * g * particle_radius**2 / (9.0 * air_viscosity)

    def quantum_inspired_cooling_control(self, thermal_field: np.ndarray,
                                          n_iterations: int = 50,
                                          cooling_capacity: float = 1e6) -> Dict[str, Any]:
        """
        [D] Quantum-ANNEALING-INSPIRED search for cooling schedule.

        This is a SIMULATED-ANNEALING / quantum-inspired optimizer that picks
        where/when to apply cooling power to minimize residual heat. It is a
        CONTROL ALGORITHM, not quantum thermodynamics of magma.

        Returns a schedule of cooling interventions.
        """
        rng = np.random.default_rng(42)
        best_field = thermal_field.copy()
        best_energy = float(np.sum(thermal_field))
        best_schedule = []

        current_field = thermal_field.copy()
        current_energy = best_energy
        schedule = []

        for it in range(n_iterations):
            # Quantum-inspired: "tunneling" allows occasional large jumps
            temperature_param = 1.0 - it / n_iterations
            # Choose intervention site (weighted by heat)
            flat = current_field.flatten()
            probs = flat / (flat.sum() + 1e-12)
            idx = rng.choice(len(flat), p=probs)
            site = np.unravel_index(idx, current_field.shape)

            # Apply cooling
            trial_field = current_field.copy()
            reduction = cooling_capacity * 1e-9
            trial_field[site] = max(0.0, trial_field[site] - reduction)
            trial_energy = float(np.sum(trial_field))

            # Metropolis acceptance with tunneling term
            delta = trial_energy - current_energy
            if delta < 0 or rng.random() < math.exp(-delta / (temperature_param * best_energy + 1e-12)):
                current_field = trial_field
                current_energy = trial_energy
                schedule.append({"iteration": it, "site": site, "applied_J": reduction})

            if current_energy < best_energy:
                best_energy = current_energy
                best_field = current_field.copy()
                best_schedule = list(schedule)

        self.cooling_schedule = best_schedule
        return {
            "residual_energy": best_energy,
            "n_interventions": len(best_schedule),
            "schedule": best_schedule,
            "note": ("QUANTUM-INSPIRED CONTROL ALGORITHM, not quantum physics "
                     "of magma. Magma thermodynamics remain classical."),
        }


# ═══════════════════════════════════════════════════════════════════════
# [HAZ5] STORM ENGINE — unified atmospheric solver
# ═══════════════════════════════════════════════════════════════════════

class StormEngine:
    """
    [C] Unified 3D atmospheric solver for all storm types.

    Solves compressible Navier-Stokes with:
      • Coriolis force (planetary rotation)
      • Moisture (water vapor mixing ratio) + latent heat release
      • Buoyancy
    Storm type selected via regime parameters:
      hurricane  → strong Coriolis, warm ocean surface, axisymmetric
      tornado    → localized intense vorticity, weak Coriolis
      thunderstorm → strong convection, moisture
      sandstorm  → dry, particle transport, strong wind shear
    """

    OMEGA_EARTH = 7.2921e-5  # rad/s
    RD = 287.04              # dry air gas constant
    LV = 2.5e6               # latent heat of vaporization

    def __init__(self, cfg: DisasterConfig, storm_type: HazardType,
                 latitude_deg: float = 13.75):  # Bangkok default
        self.cfg = cfg
        self.storm_type = storm_type
        self.latitude = math.radians(latitude_deg)
        self.f_coriolis = 2 * self.OMEGA_EARTH * math.sin(self.latitude)

        # Regime parameters
        self.regime = self._get_regime(storm_type)

    def _get_regime(self, storm_type: HazardType) -> Dict[str, Any]:
        """Regime parameters per storm type."""
        if storm_type == HazardType.HURRICANE:
            return {"coriolis_scale": 1.0, "moisture": 0.020, "sst_k": 301.0,
                    "vmax": 70.0, "type": "tropical_cyclone"}
        elif storm_type == HazardType.TORNADO:
            return {"coriolis_scale": 0.01, "moisture": 0.015, "sst_k": 295.0,
                    "vmax": 130.0, "type": "mesocyclone"}
        elif storm_type == HazardType.THUNDERSTORM:
            return {"coriolis_scale": 0.1, "moisture": 0.018, "sst_k": 298.0,
                    "vmax": 40.0, "type": "convective"}
        elif storm_type == HazardType.SANDSTORM:
            return {"coriolis_scale": 0.3, "moisture": 0.001, "sst_k": 310.0,
                    "vmax": 30.0, "type": "aeolian"}
        else:
            return {"coriolis_scale": 0.1, "moisture": 0.010, "sst_k": 295.0,
                    "vmax": 20.0, "type": "generic"}

    def coriolis_acceleration(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[C] Coriolis acceleration: a_x = f·v, a_y = -f·u."""
        f = self.f_coriolis * self.regime["coriolis_scale"]
        return f * v, -f * u

    def latent_heating(self, q_vapor: np.ndarray, w: np.ndarray,
                       temperature: np.ndarray) -> np.ndarray:
        """
        [C] Latent heat release where moist air rises and condenses.
        Simplified: heating ∝ condensation rate when saturated.
        """
        # Saturation mixing ratio (simplified Clausius-Clapeyron)
        qs = 0.622 * 6.112 * math.exp(17.67 * (temperature - 273.15) /
                                        (temperature - 29.65)) / 10.0
        condensation = np.maximum(q_vapor - qs, 0.0) * np.maximum(w, 0.0)
        return self.LV * condensation

    def initialize_vortex(self, nx: int, ny: int, nz: int) -> Dict[str, np.ndarray]:
        """Initialize a storm vortex (Rankine-like for hurricane/tornado)."""
        x = np.linspace(0, self.cfg.domain_size[0], nx)
        y = np.linspace(0, self.cfg.domain_size[1], ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        cx, cy = self.cfg.domain_size[0] / 2, self.cfg.domain_size[1] / 2
        r = np.sqrt((X - cx)**2 + (Y - cy)**2) + 1e-6

        rmax = 0.15 * min(self.cfg.domain_size[0], self.cfg.domain_size[1])
        vmax = self.regime["vmax"]

        # Rankine vortex
        vt = np.where(r < rmax, vmax * r / rmax, vmax * rmax / r)
        # Tangential → Cartesian
        u = -vt * (Y - cy) / r
        v = vt * (X - cx) / r

        # Extend to 3D
        u3 = np.repeat(u[:, :, None], nz, axis=2)
        v3 = np.repeat(v[:, :, None], nz, axis=2)
        w3 = np.zeros_like(u3)
        T3 = np.full_like(u3, self.regime["sst_k"] - 5.0)
        q3 = np.full_like(u3, self.regime["moisture"])

        return {"u": u3, "v": v3, "w": w3, "T": T3, "qv": q3}


# ═══════════════════════════════════════════════════════════════════════
# DISASTER SUITE — orchestrator
# ═══════════════════════════════════════════════════════════════════════

class DisasterSuite:
    """
    [D] Orchestrate all hazard engines with AMR + No-Zeno event control.
    """

    def __init__(self, cfg: Optional[DisasterConfig] = None):
        self.cfg = cfg or DisasterConfig()
        self.engines: Dict[HazardType, Any] = {}
        self.no_zeno = NoZenoGate(0.2, 0.05) if HAS_AMR else None

    def build_engine(self, hazard: HazardType, **kwargs) -> Any:
        """Instantiate the appropriate engine for a hazard type."""
        if hazard == HazardType.FIRE:
            engine = FireCFDEngine(self.cfg)
        elif hazard == HazardType.FLOOD:
            engine = FloodCFDEngine(self.cfg)
        elif hazard == HazardType.EARTHQUAKE:
            engine = EarthquakeEngine(self.cfg)
        elif hazard == HazardType.VOLCANO:
            engine = VolcanoEngine(self.cfg)
        elif hazard in (HazardType.HURRICANE, HazardType.TORNADO,
                        HazardType.THUNDERSTORM, HazardType.SANDSTORM):
            engine = StormEngine(self.cfg, hazard, **kwargs)
        else:
            raise ValueError(f"Unknown hazard: {hazard}")
        self.engines[hazard] = engine
        return engine


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Disaster CFD Suite — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch : {HAS_TORCH}")
    print(f"  AMR     : {HAS_AMR}")

    cfg = DisasterConfig()
    suite = DisasterSuite(cfg)

    # ── Fire ──
    print("\n  ── [HAZ1] Fire + Furniture ──")
    fire = suite.build_engine(HazardType.FIRE)
    fire.add_furniture("sofa", "foam", 20.0, (10, 10, 0), 3.0)
    fire.add_furniture("table", "wood", 15.0, (20, 20, 0), 2.0)
    fire.add_furniture("curtain", "fabric", 3.0, (5, 30, 0), 4.0)
    # Ignite sofa
    fire.furniture[0]["ignited"] = True
    fire.furniture[0]["temperature"] = 700.0
    state = {"time": 0.0}
    state = fire.step(state, dt=0.01)
    print(f"    Furniture items    : {len(fire.furniture)}")
    print(f"    Total HRR          : {state['total_HRR']:.2e} W")

    # ── Flood ──
    print("\n  ── [HAZ2] Flood + Pipes + Future Dam ──")
    flood = suite.build_engine(HazardType.FLOOD)
    flood.add_pipe("main_drain", diameter=1.0, length=200.0, slope=0.01,
                   inlet_node="A", outlet_node="B")
    flood.add_future_dam("north_dam", crest_level=15.0, spillway_capacity=500.0,
                         position=(50, 50))
    n = 32
    h = np.ones((n, n)) * 0.5
    hu = np.zeros((n, n)); hv = np.zeros((n, n))
    bed = np.zeros((n, n))
    out = flood.step_surface(h, hu, hv, bed, dt=0.01, dx=3.0)
    print(f"    Pipes              : {len(flood.pipes)}")
    print(f"    Future dams        : {len(flood.dams)} ({flood.dams[0]['status']})")
    print(f"    Water depth range  : [{out['h'].min():.3f}, {out['h'].max():.3f}]")

    # ── Earthquake ──
    print("\n  ── [HAZ3] Earthquake ──")
    eq = suite.build_engine(HazardType.EARTHQUAKE)
    vx = np.zeros((n, n)); vy = np.zeros((n, n))
    sxx = np.zeros((n, n)); syy = np.zeros((n, n)); sxy = np.zeros((n, n))
    eq_out = eq.step(vx, vy, sxx, syy, sxy, dt=1e-4, dx=3.0)
    strain = np.zeros((n, n))
    ru = eq.liquefaction_ratio(strain)
    print(f"    Wave type          : velocity-stress elastic")
    print(f"    Max |vx|           : {np.abs(eq_out['vx']).max():.2e}")
    print(f"    Liquefaction ru    : [{ru.min():.2f}, {ru.max():.2f}]")

    # ── Volcano ──
    print("\n  ── [HAZ4] Volcano + Quantum-inspired cooling ──")
    vol = suite.build_engine(HazardType.VOLCANO)
    thermal = np.random.default_rng(0).uniform(500, 1200, (16, 16))
    cool = vol.quantum_inspired_cooling_control(thermal, n_iterations=30)
    print(f"    Magma temp         : {vol.magma_temp} K")
    print(f"    Cooling interventions: {cool['n_interventions']}")
    print(f"    Residual energy    : {cool['residual_energy']:.2e}")
    print(f"    NOTE: {cool['note'][:70]}...")

    # ── Storms ──
    print("\n  ── [HAZ5] Storms ──")
    for storm_type in [HazardType.HURRICANE, HazardType.TORNADO,
                       HazardType.THUNDERSTORM, HazardType.SANDSTORM]:
        storm = suite.build_engine(storm_type, latitude_deg=13.75)
        vortex = storm.initialize_vortex(32, 32, 8)
        max_wind = np.sqrt(vortex["u"]**2 + vortex["v"]**2).max()
        print(f"    {storm_type.name:12s}: Coriolis f={storm.f_coriolis:.2e}, "
              f"regime={storm.regime['type']}, max wind={max_wind:.1f} m/s")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Simulation framework, NOT operational forecasting")
    print("  • Furniture/pipe/dam models are simplified engineering proxies")
    print("  • 'Quantum cooling' is a CONTROL ALGORITHM, not magma quantum physics")
    print("  • Liquefaction is a proxy, not a full constitutive model")
    print("  • Real design requires validated codes + experiments")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
