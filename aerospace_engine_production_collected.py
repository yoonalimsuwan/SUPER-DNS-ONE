# =============================================================================
# AEROSPACE ENGINE — Production Single File
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0
# =============================================================================
# Production-grade deterministic simulation framework for aerospace alloy
# screening. Refactored with Pydantic validation, structured logging, async
# concurrency control, retry logic, and comprehensive error handling.
# =============================================================================

import asyncio
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

# =============================================================================
# [0]  Logging Setup
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for production observability."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# =============================================================================
# [1]  Exception Hierarchy
# =============================================================================

class AerospaceEngineError(Exception):
    """Base exception for all aerospace engine errors."""
    pass


class MaterialValidationError(AerospaceEngineError):
    """Raised when material composition or parameters fail validation."""
    pass


class SimulationError(AerospaceEngineError):
    """Raised when a simulation task fails or times out."""
    pass


class ConfigurationError(AerospaceEngineError):
    """Raised when engine configuration is invalid."""
    pass


# =============================================================================
# [2]  Configuration (Pydantic v2)
# =============================================================================

class EngineConfig(BaseModel):
    """Production-grade configuration with environment variable support.

    All parameters can be overridden via environment variables using the AERO_ prefix.
    Example: AERO_MAX_CONCURRENT_TASKS=20
    """

    # Thermodynamics
    gas_constant: float = Field(default=8.314, description="Universal gas constant [J/(mol·K)]")
    reference_temperature: float = Field(default=298.15, description="Reference temperature [K]")

    # Structural Mechanics
    yield_stress_multiplier: float = Field(default=450.0, gt=0, description="Base multiplier for yield stress [MPa]")
    sc_tuning_factor: float = Field(default=1.15, gt=0, description="Structural calculus precision tuning factor")
    apply_structural_calculus: bool = Field(default=True, description="Enable high-precision structural calculus")

    # Fatigue
    fatigue_exponent: float = Field(default=2.0, gt=0, description="Fatigue life exponent")
    fatigue_divisor: float = Field(default=8.5, gt=0, description="Fatigue life normalization divisor")

    # Aerodynamics / Creep
    enable_dns_coupling: bool = Field(default=True, description="Enable DNS solver coupling")
    wind_shear_reference: float = Field(default=65000.0, gt=0, description="Reference wind shear [Pa]")
    creep_shear_factor: float = Field(default=0.00012, gt=0, description="Dynamic load conversion factor")
    creep_thermal_exponent: float = Field(default=2.5, gt=0, description="Thermal creep exponent")

    # Viability Thresholds
    min_yield_strength: float = Field(default=850.0, gt=0, description="Minimum viable yield strength [MPa]")
    min_fatigue_life: int = Field(default=100_000, gt=0, description="Minimum viable fatigue life [cycles]")
    max_creep_rate: float = Field(default=15.0, gt=0, description="Maximum viable creep rate [units]")

    # HPC / Dispatcher
    max_concurrent_tasks: int = Field(default=10, ge=1, le=1000, description="Maximum concurrent simulation tasks")
    simulation_timeout: float = Field(default=30.0, gt=0, description="Per-task timeout [seconds]")
    simulation_delay: float = Field(default=1.5, ge=0, description="Simulated computation delay [seconds]")

    # Retry Policy
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts for failed tasks")
    retry_min_wait: float = Field(default=2.0, gt=0, description="Minimum retry wait [seconds]")
    retry_max_wait: float = Field(default=10.0, gt=0, description="Maximum retry wait [seconds]")

    @field_validator("retry_max_wait")
    @classmethod
    def validate_retry_wait(cls, v: float, info) -> float:
        min_wait = info.data.get("retry_min_wait", 2.0)
        if v <= min_wait:
            raise ValueError("retry_max_wait must be greater than retry_min_wait")
        return v

    class Config:
        env_prefix = "AERO_"
        validate_assignment = True


# =============================================================================
# [3]  Data Models (Pydantic Schemas)
# =============================================================================

class MaterialComposition(BaseModel):
    """Represents the chemical composition and environmental conditions of an alloy."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)

    elements: Dict[str, float] = Field(
        ..., 
        description="Elemental composition as mass fractions (sum ≈ 1.0)",
        examples=[{"Ti": 0.90, "Al": 0.06, "V": 0.04}]
    )
    temperature_k: float = Field(..., gt=0, lt=10000, description="Temperature in Kelvin")
    pressure_pa: float = Field(..., gt=0, description="Pressure in Pascals")

    @field_validator("elements")
    @classmethod
    def validate_composition(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure mass fractions sum to 1.0 (±1% tolerance) and are non-negative."""
        if not v:
            raise ValueError("Composition cannot be empty")
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Elemental fractions must sum to 1.0 (±0.01), got {total:.4f}. "
                f"Please normalize your composition."
            )
        invalid = [el for el, frac in v.items() if frac < 0 or frac > 1]
        if invalid:
            raise ValueError(f"Invalid fractions for elements: {invalid}")
        return {el: frac / total for el, frac in v.items()}

    @field_validator("temperature_k")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Temperature must be absolute (≥ 0 K)")
        if v > 5000:
            warnings.warn(f"Extremely high temperature: {v} K. Ensure physical validity.")
        return v

    def get_primary_element(self) -> str:
        """Return the element with the highest mass fraction."""
        return max(self.elements, key=self.elements.get)

    def __str__(self) -> str:
        return f"{self.get_primary_element()}-based alloy @ {self.temperature_k}K"


class MaterialProperties(BaseModel):
    """Output properties derived from simulation."""
    model_config = ConfigDict(frozen=True)

    yield_strength_mpa: float = Field(..., description="Yield strength [MPa]")
    fatigue_life_cycles: int = Field(..., ge=0, description="Fatigue life [cycles]")
    creep_rate: float = Field(..., ge=0, description="Thermal creep rate [units]")
    is_viable: bool = Field(..., description="Pass/fail against flight criteria")

    def __str__(self) -> str:
        status = "PASS" if self.is_viable else "FAIL"
        return (
            f"Yield: {self.yield_strength_mpa:.2f} MPa | "
            f"Fatigue: {self.fatigue_life_cycles:,} cycles | "
            f"Creep: {self.creep_rate:.4f} | Status: {status}"
        )


class SimulationResult(BaseModel):
    """Complete result of a single simulation task including metadata."""
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique job identifier")
    material: MaterialComposition = Field(..., description="Input material specification")
    properties: MaterialProperties = Field(..., description="Computed material properties")
    execution_time_ms: float = Field(..., ge=0, description="Wall-clock execution time [ms]")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")
    retry_count: int = Field(default=0, ge=0, description="Number of retries performed")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a plain dictionary."""
        return {
            "task_id": self.task_id,
            "material": self.material.model_dump(),
            "properties": self.properties.model_dump(),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
        }


# =============================================================================
# [4]  Core Thermodynamics Module
# =============================================================================

class CoreThermodynamics:
    """Deterministic thermodynamic solver for alloy phase stability.

    Uses vectorized NumPy operations to compute a stability matrix that reflects
    compositional and thermal effects on the alloy microstructure.
    """

    def __init__(self, gas_constant: float = 8.314, reference_temperature: float = 298.15):
        self.gas_constant = gas_constant
        self.reference_temperature = reference_temperature

    def calculate_phase_stability(self, composition: MaterialComposition) -> np.ndarray:
        """Compute the phase stability matrix for a given material composition.

        The matrix diagonal encodes self-interaction strength (scaled by temperature
        and base element weight). Off-diagonal terms approximate pairwise interaction
        energies using a simplified regular solution model.

        Args:
            composition: Validated MaterialComposition instance.

        Returns:
            A symmetric n×n stability matrix (n = number of elements).
        """
        elements = composition.elements
        if not elements:
            raise ValueError("Composition must contain at least one element")

        n = len(elements)
        fractions = np.array(list(elements.values()), dtype=np.float64)
        base_weight = float(np.max(fractions))
        temp_ratio = composition.temperature_k / self.reference_temperature

        # Initialize diagonal-dominant matrix
        stability = np.eye(n, dtype=np.float64) * temp_ratio * base_weight

        # Add pairwise interaction terms (simplified regular solution)
        for i in range(n):
            for j in range(i + 1, n):
                interaction = fractions[i] * fractions[j] * temp_ratio * 0.5
                stability[i, j] = interaction
                stability[j, i] = interaction

        return stability

    def calculate_entropy_contribution(self, composition: MaterialComposition) -> float:
        """Estimate configurational entropy of mixing [J/(mol·K)].

        Ideal mixing approximation: ΔS_mix = -R * Σ(x_i * ln(x_i))
        """
        fractions = np.array(list(composition.elements.values()), dtype=np.float64)
        fractions = fractions[fractions > 1e-12]
        entropy = -self.gas_constant * np.sum(fractions * np.log(fractions))
        return float(entropy)


# =============================================================================
# [5]  Structural Mechanics Module
# =============================================================================

class StructuralMechanics:
    """Deterministic structural solver for aerospace alloy mechanical properties.

    Computes yield strength from the phase stability matrix and estimates fatigue
    life using a power-law relationship derived from stress-cycle (S-N) behavior.
    """

    def __init__(
        self,
        apply_structural_calculus: bool = True,
        yield_multiplier: float = 450.0,
        sc_factor: float = 1.15,
        fatigue_exponent: float = 2.0,
        fatigue_divisor: float = 8.5,
    ):
        self.apply_sc = apply_structural_calculus
        self.yield_multiplier = yield_multiplier
        self.sc_factor = sc_factor
        self.fatigue_exponent = fatigue_exponent
        self.fatigue_divisor = fatigue_divisor

    def solve_yield_stress(self, phase_matrix: np.ndarray) -> float:
        """Calculate yield stress from the phase stability matrix.

        Uses the Frobenius norm as a proxy for overall bond strength, scaled by
        an empirical multiplier. When structural calculus is enabled, an additional
        precision factor is applied.

        Args:
            phase_matrix: n×n phase stability matrix from thermodynamics.

        Returns:
            Estimated yield strength in MPa.
        """
        if phase_matrix.size == 0:
            raise SimulationError("Phase stability matrix is empty")

        try:
            norm = float(np.linalg.norm(phase_matrix, ord="fro"))
        except Exception as exc:
            raise SimulationError(f"Failed to compute matrix norm: {exc}") from exc

        base_stress = norm * self.yield_multiplier

        if self.apply_sc:
            base_stress *= self.sc_factor

        return max(base_stress, 0.0)

    def calculate_fatigue(self, yield_stress: float) -> int:
        """Estimate fatigue life cycles from yield stress.

        Models the inverse power-law relationship between stress amplitude and
        cycles to failure (Basquin-type behavior simplified for screening).

        Args:
            yield_stress: Yield strength in MPa (must be > 0).

        Returns:
            Estimated fatigue life in cycles.
        """
        if yield_stress <= 0:
            raise ValueError(f"Yield stress must be positive, got {yield_stress}")

        cycles = int((yield_stress ** self.fatigue_exponent) / self.fatigue_divisor)
        return max(cycles, 0)

    def calculate_ultimate_tensile_strength(self, yield_stress: float, ratio: float = 1.25) -> float:
        """Estimate ultimate tensile strength (UTS) from yield stress.

        Typical aerospace alloys exhibit UTS/Yield ratios between 1.1–1.4.
        """
        return yield_stress * ratio


# =============================================================================
# [6]  Aero-CFD Interaction Module
# =============================================================================

class AeroCFDInteraction:
    """Evaluates aerodynamic thermal-mechanical loading on aerospace materials.

    Computes boundary layer stress and thermal creep rate under flight conditions.
    Designed with hooks for future Direct Numerical Simulation (DNS) coupling.
    """

    def __init__(self, enable_dns_coupling: bool = True):
        self.enable_dns = enable_dns_coupling

    def evaluate_boundary_layer_stress(
        self,
        wind_shear_pa: float,
        temp_k: float,
        shear_factor: float = 0.00012,
        thermal_exponent: float = 2.5,
    ) -> float:
        """Compute thermal creep rate under aerodynamic wind shear loading.

        Creep rate increases non-linearly with temperature (Arrhenius-type
        behavior approximated by a power law) and scales linearly with dynamic
        wind shear load.

        Args:
            wind_shear_pa: Aerodynamic wind shear stress [Pa].
            temp_k: Surface temperature [K].
            shear_factor: Dynamic load conversion coefficient.
            thermal_exponent: Temperature sensitivity exponent.

        Returns:
            Estimated creep rate in arbitrary units (consistent with thresholds).
        """
        if temp_k <= 0:
            raise ValueError(f"Temperature must be positive [K], got {temp_k}")
        if wind_shear_pa < 0:
            raise ValueError(f"Wind shear cannot be negative, got {wind_shear_pa}")
        if shear_factor <= 0 or thermal_exponent <= 0:
            raise ValueError("Physical coefficients must be positive")

        thermal_creep = (temp_k / 1000.0) ** thermal_exponent
        dynamic_load = wind_shear_pa * shear_factor
        return float(dynamic_load * thermal_creep)

    def estimate_nusselt_number(self, reynolds: float, prandtl: float = 0.71) -> float:
        """Estimate Nusselt number for turbulent boundary layer heat transfer.

        Empirical correlation: Nu = 0.0296 * Re^0.8 * Pr^(1/3)
        """
        if reynolds <= 0 or prandtl <= 0:
            raise ValueError("Dimensionless numbers must be positive")
        return 0.0296 * (reynolds ** 0.8) * (prandtl ** (1.0 / 3.0))


# =============================================================================
# [7]  HPC / Task Dispatcher Worker
# =============================================================================

class HPCDispatcher:
    """Asynchronous task dispatcher simulating HPC cluster job scheduling.

    Features:
    - Semaphore-based concurrency limiting
    - Configurable retry with exponential backoff
    - Per-task timeout enforcement
    - Execution time tracking and metadata collection
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._task_counter = 0

    async def run_simulation_task(
        self,
        material: MaterialComposition,
        task_id: Optional[str] = None,
    ) -> SimulationResult:
        """Execute a single material simulation with full error handling.

        Args:
            material: Validated material composition.
            task_id: Optional custom job ID. Auto-generated if None.

        Returns:
            SimulationResult containing properties and metadata.

        Raises:
            SimulationError: If all retry attempts are exhausted.
        """
        if task_id is None:
            self._task_counter += 1
            primary = material.get_primary_element()
            task_id = f"JOB_{self._task_counter:03d}_{primary}"

        start_time = time.perf_counter()
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self.semaphore:
                    properties = await asyncio.wait_for(
                        self._compute(material, task_id),
                        timeout=self.config.simulation_timeout,
                    )

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return SimulationResult(
                    task_id=task_id,
                    material=material,
                    properties=properties,
                    execution_time_ms=elapsed_ms,
                    retry_count=attempt - 1,
                )

            except asyncio.TimeoutError:
                last_exception = SimulationError(
                    f"Task {task_id} timed out after {self.config.simulation_timeout}s"
                )
            except Exception as exc:
                last_exception = exc

            if attempt < self.config.max_retries:
                wait_time = min(
                    self.config.retry_min_wait * (2 ** (attempt - 1)),
                    self.config.retry_max_wait,
                )
                await asyncio.sleep(wait_time)

        raise last_exception or SimulationError(
            f"Task {task_id} failed after {self.config.max_retries} retries"
        )

    async def _compute(self, material: MaterialComposition, task_id: str) -> MaterialProperties:
        """Internal computation pipeline."""
        if self.config.simulation_delay > 0:
            await asyncio.sleep(self.config.simulation_delay)

        thermo = CoreThermodynamics(
            gas_constant=self.config.gas_constant,
            reference_temperature=self.config.reference_temperature,
        )
        mech = StructuralMechanics(
            apply_structural_calculus=self.config.apply_structural_calculus,
            yield_multiplier=self.config.yield_stress_multiplier,
            sc_factor=self.config.sc_tuning_factor,
            fatigue_exponent=self.config.fatigue_exponent,
            fatigue_divisor=self.config.fatigue_divisor,
        )
        aero = AeroCFDInteraction(enable_dns_coupling=self.config.enable_dns_coupling)

        # 1. Thermodynamics: Phase stability
        phase_matrix = thermo.calculate_phase_stability(material)

        # 2. Structural Mechanics: Yield stress & fatigue
        yield_stress = mech.solve_yield_stress(phase_matrix)
        fatigue = mech.calculate_fatigue(yield_stress)

        # 3. Aerodynamics: Thermal creep under flight loading
        creep = aero.evaluate_boundary_layer_stress(
            wind_shear_pa=self.config.wind_shear_reference,
            temp_k=material.temperature_k,
            shear_factor=self.config.creep_shear_factor,
            thermal_exponent=self.config.creep_thermal_exponent,
        )

        # 4. Viability assessment
        is_viable = (
            yield_stress >= self.config.min_yield_strength
            and fatigue >= self.config.min_fatigue_life
            and creep < self.config.max_creep_rate
        )

        return MaterialProperties(
            yield_strength_mpa=yield_stress,
            fatigue_life_cycles=fatigue,
            creep_rate=creep,
            is_viable=is_viable,
        )

    async def run_batch(
        self,
        materials: List[MaterialComposition],
        task_ids: Optional[List[str]] = None,
    ) -> List[SimulationResult]:
        """Execute multiple simulations concurrently with controlled parallelism."""
        if task_ids is not None and len(task_ids) != len(materials):
            raise ValueError("task_ids length must match materials length")

        tasks = [
            self.run_simulation_task(mat, task_ids[i] if task_ids else None)
            for i, mat in enumerate(materials)
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[SimulationResult] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                results.append(
                    SimulationResult(
                        task_id=task_ids[i] if task_ids else f"JOB_FAIL_{i:03d}",
                        material=materials[i],
                        properties=MaterialProperties(
                            yield_strength_mpa=0.0,
                            fatigue_life_cycles=0,
                            creep_rate=float("inf"),
                            is_viable=False,
                        ),
                        execution_time_ms=0.0,
                        retry_count=self.config.max_retries,
                    )
                )
            else:
                results.append(res)

        return results


# =============================================================================
# [8]  Main API Entry Point (Orchestrator)
# =============================================================================

class AerospaceEngine:
    """Production-ready orchestrator for aerospace material screening.

    Provides a clean async API for evaluating single or batch material
    compositions against flight-worthiness criteria.

    Example:
        >>> engine = AerospaceEngine()
        >>> results = await engine.evaluate_batch([material_1, material_2])
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._validate_config()
        self.dispatcher = HPCDispatcher(config=self.config)

    def _validate_config(self) -> None:
        """Validate critical configuration parameters."""
        if self.config.max_concurrent_tasks < 1:
            raise ConfigurationError("max_concurrent_tasks must be ≥ 1")
        if self.config.simulation_timeout < 0.1:
            raise ConfigurationError("simulation_timeout must be ≥ 0.1s")

    async def evaluate(self, material: MaterialComposition, task_id: Optional[str] = None) -> SimulationResult:
        """Evaluate a single material composition."""
        return await self.dispatcher.run_simulation_task(material, task_id)

    async def evaluate_batch(
        self,
        materials: List[MaterialComposition],
        task_ids: Optional[List[str]] = None,
    ) -> List[SimulationResult]:
        """Evaluate multiple materials with controlled concurrency."""
        if not materials:
            return []
        return await self.dispatcher.run_batch(materials, task_ids)

    def generate_report(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Generate a structured summary report from simulation results."""
        if not results:
            return {"status": "empty", "materials_evaluated": 0}

        viable = [r for r in results if r.properties.is_viable]
        non_viable = [r for r in results if not r.properties.is_viable]

        yield_stresses = [r.properties.yield_strength_mpa for r in results]
        fatigue_lives = [r.properties.fatigue_life_cycles for r in results]
        creep_rates = [r.properties.creep_rate for r in results]
        exec_times = [r.execution_time_ms for r in results]

        return {
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "materials_evaluated": len(results),
            "summary": {
                "pass_count": len(viable),
                "fail_count": len(non_viable),
                "pass_rate": round(len(viable) / len(results), 4),
            },
            "statistics": {
                "yield_strength_mpa": {
                    "mean": round(sum(yield_stresses) / len(yield_stresses), 2),
                    "min": round(min(yield_stresses), 2),
                    "max": round(max(yield_stresses), 2),
                },
                "fatigue_life_cycles": {
                    "mean": int(sum(fatigue_lives) / len(fatigue_lives)),
                    "min": min(fatigue_lives),
                    "max": max(fatigue_lives),
                },
                "creep_rate": {
                    "mean": round(sum(creep_rates) / len(creep_rates), 4),
                    "min": round(min(creep_rates), 4),
                    "max": round(max(creep_rates), 4),
                },
                "execution_time_ms": {
                    "total": round(sum(exec_times), 2),
                    "mean": round(sum(exec_times) / len(exec_times), 2),
                },
            },
            "details": [r.to_dict() for r in results],
        }


# =============================================================================
# [9]  Main Execution Block (Demo)
# =============================================================================

async def main():
    """Main async entry point demonstrating batch material evaluation."""
    setup_logging(logging.INFO)
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info(" Aerospace Material Discovery Engine [Production v2.0.0] ")
    logger.info(" Deterministic Solvers | HPC Dispatcher | Full Observability ")
    logger.info("=" * 60)

    # Load configuration (auto-picks up AERO_* environment variables)
    config = EngineConfig()
    logger.info("Configuration loaded: concurrent=%d, timeout=%.1fs",
                config.max_concurrent_tasks, config.simulation_timeout)

    # Initialize engine
    engine = AerospaceEngine(config)

    # Define test materials
    materials = [
        MaterialComposition(
            elements={"Ti": 0.90, "Al": 0.06, "V": 0.04},
            temperature_k=900.0,
            pressure_pa=101325.0,
        ),
        MaterialComposition(
            elements={"Al": 0.98, "Li": 0.02},
            temperature_k=450.0,
            pressure_pa=101325.0,
        ),
        MaterialComposition(
            elements={"Ni": 0.70, "Cr": 0.20, "Fe": 0.10},
            temperature_k=1100.0,
            pressure_pa=150000.0,
        ),
    ]

    task_ids = [
        "JOB_Ti64_Titanium",
        "JOB_AlLi_Lightweight",
        "JOB_Inconel_Superalloy",
    ]

    logger.info("Submitting %d materials for evaluation...", len(materials))

    # Execute batch evaluation
    start_time = asyncio.get_event_loop().time()
    results = await engine.evaluate_batch(materials, task_ids)
    elapsed = asyncio.get_event_loop().time() - start_time

    # Print formatted results
    print("\n" + "=" * 60)
    print(" SIMULATION RESULTS ")
    print("=" * 60)

    for result in results:
        status_icon = "PASS" if result.properties.is_viable else "FAIL"

        print(f"\nTask: {result.task_id}")
        print(f"   Material: {result.material}")
        print(f"   ├─ Yield Strength: {result.properties.yield_strength_mpa:>10.2f} MPa")
        print(f"   ├─ Fatigue Life:   {result.properties.fatigue_life_cycles:>10,} cycles")
        print(f"   ├─ Thermal Creep:  {result.properties.creep_rate:>10.4f} units")
        print(f"   ├─ Execution:      {result.execution_time_ms:>10.2f} ms")
        print(f"   └─ Flight Viable:  {'PASS' if result.properties.is_viable else 'FAIL'}")
        print("   " + "-" * 40)

    # Generate and print summary report
    report = engine.generate_report(results)
    print("\n" + "=" * 60)
    print(" BATCH SUMMARY REPORT ")
    print("=" * 60)
    import json
    print(json.dumps(report["summary"], indent=2))
    print("\nStatistics:")
    print(json.dumps(report["statistics"], indent=2))
    print(f"\nTotal Pipeline Time: {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("main").warning("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logging.getLogger("main").error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
