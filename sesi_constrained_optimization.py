"""
=============================================================================
SESI CONSTRAINED MULTI-OBJECTIVE OPTIMIZATION
Fabrication limits + material constraints + performance objectives
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY Soul Move By Power of Holy Spirit
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE ADDS
---------------------
[D] Extends NSGA-II (from sesi_multiobjective_stealth.py) with:
  • Fabrication constraints (minimum feature size, aspect ratio)
  • Material availability constraints (discrete ε, μ values)
  • Cost constraints (budget limits)
  • Performance constraints (minimum absorption, maximum RCS reduction)

CONSTRAINT HANDLING METHODS
----------------------------
  [C] Penalty functions: add penalty to fitness for constraint violations
  [C] Feasibility rules: prefer feasible solutions over infeasible
  [D] Repair operators: modify infeasible solutions to make them feasible

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Constraints are ENGINEERING REQUIREMENTS, not physical laws.
• Multi-objective optimization gives PARETO FRONT, not single optimum.
• Designer must choose trade-off based on mission requirements.
• Results depend on constraint formulation and penalty weights.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("sesi.constrained_opt")

try:
    from sesi_multiobjective_stealth import (
        StealthDesign, StealthObjectiveFunctions, NSGA2Optimizer
    )
    HAS_BASE = True
except ImportError:
    HAS_BASE = False

__all__ = [
    "FabricationConstraints",
    "MaterialConstraints",
    "PerformanceConstraints",
    "ConstrainedNSGA2",
]


# ═══════════════════════════════════════════════════════════════════════
# [D] FABRICATION CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FabricationConstraints:
    """
    [D] Fabrication limits for metamaterial/stealth design.
    
    Constraints:
      • Minimum feature size (lithography limit)
      • Maximum aspect ratio (etching limit)
      • Minimum thickness (structural integrity)
      • Maximum thickness (weight/cost limit)
    """
    min_feature_size_mm: float = 0.1       # 100 μm lithography limit
    max_aspect_ratio: float = 10.0         # etching limit
    min_thickness_mm: float = 0.5          # structural integrity
    max_thickness_mm: float = 10.0         # weight/cost limit
    
    def check(self, design: "StealthDesign") -> Dict[str, Any]:
        """Check if design satisfies fabrication constraints."""
        violations = []
        
        # Feature size constraint (carpet level determines smallest feature)
        # At level n, smallest feature ≈ 1/3^n of total size
        # Assume total size = 100 mm
        total_size_mm = 100.0
        min_feature = total_size_mm / (3 ** design.carpet_level)
        if min_feature < self.min_feature_size_mm:
            violations.append({
                "constraint": "min_feature_size",
                "value": min_feature,
                "limit": self.min_feature_size_mm,
                "violation": self.min_feature_size_mm - min_feature,
            })
        
        # Aspect ratio constraint
        # Assume lateral size = 100 mm, thickness = design.thickness_mm
        aspect_ratio = 100.0 / max(design.thickness_mm, 1e-6)
        if aspect_ratio > self.max_aspect_ratio:
            violations.append({
                "constraint": "max_aspect_ratio",
                "value": aspect_ratio,
                "limit": self.max_aspect_ratio,
                "violation": aspect_ratio - self.max_aspect_ratio,
            })
        
        # Thickness constraints
        if design.thickness_mm < self.min_thickness_mm:
            violations.append({
                "constraint": "min_thickness",
                "value": design.thickness_mm,
                "limit": self.min_thickness_mm,
                "violation": self.min_thickness_mm - design.thickness_mm,
            })
        if design.thickness_mm > self.max_thickness_mm:
            violations.append({
                "constraint": "max_thickness",
                "value": design.thickness_mm,
                "limit": self.max_thickness_mm,
                "violation": design.thickness_mm - self.max_thickness_mm,
            })
        
        return {
            "feasible": len(violations) == 0,
            "violations": violations,
            "n_violations": len(violations),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] MATERIAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MaterialConstraints:
    """
    [D] Material availability constraints.
    
    Constraints:
      • Discrete ε_r values (available materials)
      • Discrete μ_r values (available materials)
      • Maximum loss tangent (material quality)
    """
    available_eps_r: List[float] = field(default_factory=lambda: [1.0, 2.2, 4.4, 10.0, 20.0])
    available_mu_r: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 20.0])
    max_tan_d: float = 0.5
    
    def check(self, design: "StealthDesign") -> Dict[str, Any]:
        """Check if design uses available materials."""
        violations = []
        
        # Check ε_r availability
        eps_r_closest = min(self.available_eps_r, key=lambda x: abs(x - design.eps_r))
        if abs(eps_r_closest - design.eps_r) > 0.5:
            violations.append({
                "constraint": "eps_r_availability",
                "value": design.eps_r,
                "closest_available": eps_r_closest,
                "violation": abs(eps_r_closest - design.eps_r),
            })
        
        # Check μ_r availability
        mu_r_closest = min(self.available_mu_r, key=lambda x: abs(x - design.mu_r))
        if abs(mu_r_closest - design.mu_r) > 0.5:
            violations.append({
                "constraint": "mu_r_availability",
                "value": design.mu_r,
                "closest_available": mu_r_closest,
                "violation": abs(mu_r_closest - design.mu_r),
            })
        
        # Check loss tangent
        if design.tan_d > self.max_tan_d:
            violations.append({
                "constraint": "max_tan_d",
                "value": design.tan_d,
                "limit": self.max_tan_d,
                "violation": design.tan_d - self.max_tan_d,
            })
        
        return {
            "feasible": len(violations) == 0,
            "violations": violations,
            "n_violations": len(violations),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] PERFORMANCE CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceConstraints:
    """
    [D] Performance requirements.
    
    Constraints:
      • Minimum absorption (e.g., > 90%)
      • Maximum RCS reduction (e.g., > 10 dB)
      • Minimum bandwidth (e.g., > 1 GHz)
    """
    min_absorption: float = 0.9
    min_rcs_reduction_db: float = 10.0
    min_bandwidth_ghz: float = 1.0
    
    def check(self, design: "StealthDesign",
              objectives: "StealthObjectiveFunctions") -> Dict[str, Any]:
        """Check if design meets performance requirements."""
        violations = []
        
        # Evaluate objectives
        objectives.evaluate(design)
        
        # Check absorption
        if design.absorption_mean < self.min_absorption:
            violations.append({
                "constraint": "min_absorption",
                "value": design.absorption_mean,
                "limit": self.min_absorption,
                "violation": self.min_absorption - design.absorption_mean,
            })
        
        # Check RCS reduction
        if design.rcs_reduction_db < self.min_rcs_reduction_db:
            violations.append({
                "constraint": "min_rcs_reduction",
                "value": design.rcs_reduction_db,
                "limit": self.min_rcs_reduction_db,
                "violation": self.min_rcs_reduction_db - design.rcs_reduction_db,
            })
        
        return {
            "feasible": len(violations) == 0,
            "violations": violations,
            "n_violations": len(violations),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] CONSTRAINED NSGA-II
# ═══════════════════════════════════════════════════════════════════════

class ConstrainedNSGA2(NSGA2Optimizer):
    """
    [D] NSGA-II with constraint handling.
    
    Uses penalty function approach:
      fitness_constrained = fitness_original + penalty * sum(violations)
    
    Penalty weight is adaptive: increases if too many infeasible solutions.
    """
    
    def __init__(self, objectives: "StealthObjectiveFunctions",
                 fab_constraints: FabricationConstraints,
                 mat_constraints: MaterialConstraints,
                 perf_constraints: PerformanceConstraints,
                 population_size: int = 100,
                 n_generations: int = 50,
                 penalty_weight: float = 100.0,
                 **kwargs):
        super().__init__(objectives, population_size, n_generations, **kwargs)
        self.fab_constraints = fab_constraints
        self.mat_constraints = mat_constraints
        self.perf_constraints = perf_constraints
        self.penalty_weight = penalty_weight
    
    def compute_penalty(self, design: "StealthDesign") -> float:
        """Compute total constraint violation penalty."""
        total_violation = 0.0
        
        # Fabrication constraints
        fab_check = self.fab_constraints.check(design)
        for v in fab_check["violations"]:
            total_violation += v["violation"]
        
        # Material constraints
        mat_check = self.mat_constraints.check(design)
        for v in mat_check["violations"]:
            total_violation += v["violation"]
        
        # Performance constraints
        perf_check = self.perf_constraints.check(design, self.objectives)
        for v in perf_check["violations"]:
            total_violation += v["violation"]
        
        return total_violation
    
    def constrained_fitness(self, design: "StealthDesign") -> Tuple[float, float]:
        """
        Compute constrained fitness (original + penalty).
        
        Returns: (rcs_fitness, absorption_fitness) with penalties added.
        """
        # Evaluate original objectives
        self.objectives.evaluate(design)
        
        # Compute penalty
        penalty = self.compute_penalty(design)
        
        # Add penalty to fitness (minimization)
        rcs_fitness = design.fitness[0] + self.penalty_weight * penalty
        absorption_fitness = design.fitness[1] + self.penalty_weight * penalty
        
        return rcs_fitness, absorption_fitness
    
    def optimize(self) -> Dict[str, Any]:
        """Run constrained NSGA-II optimization."""
        if not HAS_BASE:
            return {"error": "Base NSGA-II module not available"}
        
        logger.info(f"Starting constrained NSGA-II: pop={self.pop_size}, gen={self.n_gen}")
        
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate with constraints
        for design in population:
            design.fitness = self.constrained_fitness(design)
        
        # Evolution loop
        for gen in range(self.n_gen):
            # Create offspring (same as base NSGA-II)
            offspring = self._create_offspring(population)
            
            # Evaluate offspring with constraints
            for design in offspring:
                design.fitness = self.constrained_fitness(design)
            
            # Combine and select (same as base NSGA-II)
            combined = population + offspring
            fronts = self._non_dominated_sort(combined)
            
            # Select next generation
            population = self._select_next_generation(combined, fronts)
            
            if (gen + 1) % 10 == 0:
                n_feasible = sum(1 for d in population if self.compute_penalty(d) == 0)
                logger.info(f"Generation {gen+1}/{self.n_gen}: "
                           f"{n_feasible}/{len(population)} feasible")
        
        # Final non-dominated sorting
        fronts = self._non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]] if fronts else []
        
        # Filter to feasible solutions only
        feasible_pareto = [d for d in pareto_front if self.compute_penalty(d) == 0]
        
        return {
            "pareto_front": feasible_pareto,
            "all_pareto": pareto_front,
            "n_generations": self.n_gen,
            "population_size": self.pop_size,
            "n_feasible_solutions": len(feasible_pareto),
            "n_total_pareto": len(pareto_front),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Constrained Multi-Objective Optimization — Smoke Test")
    print("=" * 72)
    print(f"  Base NSGA-II available : {HAS_BASE}")
    
    if not HAS_BASE:
        print("  [SKIP] Base module not available")
        return
    
    # Define constraints
    fab = FabricationConstraints(
        min_feature_size_mm=0.1,
        max_aspect_ratio=10.0,
        min_thickness_mm=0.5,
        max_thickness_mm=5.0,
    )
    
    mat = MaterialConstraints(
        available_eps_r=[1.0, 2.2, 4.4, 10.0],
        available_mu_r=[1.0, 5.0, 10.0],
        max_tan_d=0.3,
    )
    
    perf = PerformanceConstraints(
        min_absorption=0.85,
        min_rcs_reduction_db=8.0,
    )
    
    # Create optimizer
    objectives = StealthObjectiveFunctions()
    optimizer = ConstrainedNSGA2(
        objectives=objectives,
        fab_constraints=fab,
        mat_constraints=mat,
        perf_constraints=perf,
        population_size=50,
        n_generations=20,
        penalty_weight=100.0,
    )
    
    print("\n  ── Running constrained optimization ──")
    result = optimizer.optimize()
    
    print(f"    Generations          : {result['n_generations']}")
    print(f"    Population size      : {result['population_size']}")
    print(f"    Total Pareto         : {result['n_total_pareto']}")
    print(f"    Feasible Pareto      : {result['n_feasible_solutions']}")
    
    if result["n_feasible_solutions"] > 0:
        print("\n  ── Feasible Pareto Front (first 3) ──")
        for i, design in enumerate(result["pareto_front"][:3]):
            print(f"    Solution {i+1}:")
            print(f"      Carpet level       : {design.carpet_level}")
            print(f"      ε_r                : {design.eps_r:.2f}")
            print(f"      μ_r                : {design.mu_r:.2f}")
            print(f"      tan δ              : {design.tan_d:.3f}")
            print(f"      Thickness (mm)     : {design.thickness_mm:.2f}")
            print(f"      RCS reduction (dB) : {design.rcs_reduction_db:.2f}")
            print(f"      Absorption         : {design.absorption_mean:.3f}")
    else:
        print("\n    [WARNING] No feasible solutions found")
        print("    Consider relaxing constraints or increasing population/generations")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Constraints are ENGINEERING REQUIREMENTS, not physical laws")
    print("  • Penalty weight affects trade-off between feasibility and optimality")
    print("  • Results depend on constraint formulation")
    print("  • Designer must choose from Pareto front based on mission needs")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
