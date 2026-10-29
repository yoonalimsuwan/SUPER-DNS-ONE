"""
=============================================================================
SESI MULTI-OBJECTIVE OPTIMIZATION — Stealth + Absorption Pareto Front
NSGA-II for hypersonic stealth design
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS OPTIMIZES
--------------------
Two competing objectives:
  1. Minimize RCS (stealth) — Carpet spectral reduction
  2. Maximize absorption — Ordinal-tensor impedance matching

Design variables:
  • Carpet level (discrete: 2, 3, 4, 5)
  • Material permittivity ε_r ∈ [1, 20]
  • Material permeability μ_r ∈ [1, 20]
  • Loss tangent tan_δ ∈ [0.01, 0.5]
  • Absorber thickness d ∈ [0.5mm, 5mm]

ALGORITHM
---------
[C] NSGA-II (Deb et al. 2002): Non-dominated Sorting Genetic Algorithm
    • Population-based evolutionary optimization
    • Pareto ranking + crowding distance
    • Handles multiple objectives without scalarization

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• NSGA-II gives PARETO FRONT, not a single "optimal" solution.
• Designer must choose trade-off based on mission requirements.
• Optimization is HEURISTIC; no guarantee of global optimum.
• Results depend on objective function quality (Carpet RCS is analogical).
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("sesi.multiobjective")

__all__ = [
    "StealthDesign",
    "StealthObjectiveFunctions",
    "NSGA2Optimizer",
    "ParetoFrontAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# DESIGN VARIABLE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StealthDesign:
    """A single stealth design configuration."""
    carpet_level: int          # 2, 3, 4, 5
    eps_r: float               # relative permittivity [1, 20]
    mu_r: float                # relative permeability [1, 20]
    tan_d: float               # loss tangent [0.01, 0.5]
    thickness_mm: float        # absorber thickness [0.5, 5]
    
    # Objectives (computed after evaluation)
    rcs_reduction_db: float = 0.0
    absorption_mean: float = 0.0
    fitness: Tuple[float, float] = (0.0, 0.0)
    
    def to_vector(self) -> np.ndarray:
        """Convert to optimization vector."""
        return np.array([
            self.carpet_level,
            self.eps_r,
            self.mu_r,
            self.tan_d,
            self.thickness_mm,
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "StealthDesign":
        """Create from optimization vector."""
        return cls(
            carpet_level=int(np.clip(vec[0], 2, 5)),
            eps_r=float(np.clip(vec[1], 1, 20)),
            mu_r=float(np.clip(vec[2], 1, 20)),
            tan_d=float(np.clip(vec[3], 0.01, 0.5)),
            thickness_mm=float(np.clip(vec[4], 0.5, 5)),
        )


# ═══════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

class StealthObjectiveFunctions:
    """
    [D] Objective functions for stealth optimization.
    
    Objective 1: Minimize RCS (maximize reduction in dB)
    Objective 2: Maximize mean absorption
    """
    
    def __init__(self):
        pass
    
    def evaluate(self, design: StealthDesign) -> Tuple[float, float]:
        """
        Evaluate both objectives for a design.
        
        Returns: (rcs_reduction_db, absorption_mean)
        Note: RCS reduction is NEGATED for minimization.
        """
        # Objective 1: RCS reduction (Carpet spectral model)
        rcs_reduction = self._compute_rcs_reduction(design)
        
        # Objective 2: Absorption (ordinal-tensor impedance model)
        absorption = self._compute_absorption(design)
        
        # Store in design
        design.rcs_reduction_db = rcs_reduction
        design.absorption_mean = absorption
        design.fitness = (-rcs_reduction, -absorption)  # negate for minimization
        
        return design.fitness
    
    def _compute_rcs_reduction(self, design: StealthDesign) -> float:
        """Compute RCS reduction using Carpet spectral model."""
        # Simplified model: reduction depends on carpet level and material
        level_factor = (design.carpet_level - 1) / 4.0  # normalized [0, 1]
        material_factor = 1.0 / (1.0 + design.eps_r * design.mu_r)
        
        # Base reduction (heuristic)
        base_reduction = 10.0 * level_factor * material_factor
        
        # Add loss contribution
        loss_contribution = 5.0 * design.tan_d
        
        return base_reduction + loss_contribution
    
    def _compute_absorption(self, design: StealthDesign) -> float:
        """Compute absorption using ordinal-tensor impedance model."""
        # Impedance matching quality
        Z_material = 377.0 * np.sqrt(design.mu_r / (design.eps_r * (1 - 1j * design.tan_d)))
        Gamma = np.abs((Z_material - 377.0) / (Z_material + 377.0))
        
        # Base absorption from impedance matching
        base_absorption = 1.0 - Gamma**2
        
        # Thickness resonance effect (quarter-wavelength)
        # Simplified: assume optimal at specific thickness
        f0 = 10e9  # 10 GHz
        lambda_0 = 3e8 / f0
        optimal_thickness = lambda_0 / (4.0 * np.sqrt(design.eps_r)) / 1e-3  # mm
        
        thickness_factor = np.exp(-((design.thickness_mm - optimal_thickness) / 1.0)**2)
        
        # Loss contribution
        loss_factor = 1.0 - np.exp(-design.tan_d * 10.0)
        
        return float(base_absorption * thickness_factor * loss_factor)


# ═══════════════════════════════════════════════════════════════════════
# [C] NSGA-II OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

class NSGA2Optimizer:
    """
    [C] NSGA-II multi-objective optimizer.
    
    Reference: Deb et al. (2002), "A fast and elitist multiobjective
    genetic algorithm: NSGA-II"
    
    Features:
      • Non-dominated sorting
      • Crowding distance
      • Elitism
      • Tournament selection
    """
    
    def __init__(self, objectives: StealthObjectiveFunctions,
                 population_size: int = 100,
                 n_generations: int = 50,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1):
        self.objectives = objectives
        self.pop_size = population_size
        self.n_gen = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Design variable bounds
        self.bounds = {
            "carpet_level": (2, 5),
            "eps_r": (1.0, 20.0),
            "mu_r": (1.0, 20.0),
            "tan_d": (0.01, 0.5),
            "thickness_mm": (0.5, 5.0),
        }
    
    def _initialize_population(self) -> List[StealthDesign]:
        """Initialize random population."""
        rng = np.random.default_rng(42)
        population = []
        
        for _ in range(self.pop_size):
            design = StealthDesign(
                carpet_level=int(rng.integers(2, 6)),
                eps_r=float(rng.uniform(*self.bounds["eps_r"])),
                mu_r=float(rng.uniform(*self.bounds["mu_r"])),
                tan_d=float(rng.uniform(*self.bounds["tan_d"])),
                thickness_mm=float(rng.uniform(*self.bounds["thickness_mm"])),
            )
            population.append(design)
        
        return population
    
    def _non_dominated_sort(self, population: List[StealthDesign]
                             ) -> List[List[int]]:
        """Perform non-dominated sorting."""
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for idx in fronts[i]:
                for j in dominated_set[idx]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # remove empty last front
    
    def _dominates(self, p1: StealthDesign, p2: StealthDesign) -> bool:
        """Check if p1 dominates p2 (all objectives ≤, at least one <)."""
        f1, f2 = p1.fitness, p2.fitness
        return (all(f1[i] <= f2[i] for i in range(len(f1))) and
                any(f1[i] < f2[i] for i in range(len(f1))))
    
    def _crowding_distance(self, population: List[StealthDesign],
                            front: List[int]) -> np.ndarray:
        """Compute crowding distance for a front."""
        n = len(front)
        distances = np.zeros(n)
        
        if n <= 2:
            distances[:] = np.inf
            return distances
        
        # For each objective
        for m in range(2):
            # Sort by objective
            sorted_idx = np.argsort([population[i].fitness[m] for i in front])
            
            # Boundary points get infinite distance
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            # Interior points
            obj_range = (population[front[sorted_idx[-1]]].fitness[m] -
                        population[front[sorted_idx[0]]].fitness[m])
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        population[front[sorted_idx[i+1]]].fitness[m] -
                        population[front[sorted_idx[i-1]]].fitness[m]
                    ) / obj_range
        
        return distances
    
    def _crossover(self, parent1: StealthDesign, parent2: StealthDesign
                    ) -> Tuple[StealthDesign, StealthDesign]:
        """SBX crossover (simulated binary crossover)."""
        rng = np.random.default_rng()
        
        v1 = parent1.to_vector()
        v2 = parent2.to_vector()
        
        eta = 20.0  # distribution index
        
        child1 = v1.copy()
        child2 = v2.copy()
        
        for i in range(len(v1)):
            if rng.random() < 0.5:
                if abs(v1[i] - v2[i]) > 1e-14:
                    if v1[i] < v2[i]:
                        y1, y2 = v1[i], v2[i]
                    else:
                        y1, y2 = v2[i], v1[i]
                    
                    rand = rng.random()
                    beta = 1.0 + (2.0 * (y1 - (y1 if i == 0 else self.bounds[list(self.bounds.keys())[i-1]][0])) /
                                  (y2 - y1 + 1e-14))
                    alpha = 2.0 - beta**(-(eta + 1.0))
                    
                    if rand <= (1.0 / alpha):
                        betaq = (rand * alpha)**(1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1.0))
                    
                    child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
        
        return StealthDesign.from_vector(child1), StealthDesign.from_vector(child2)
    
    def _mutate(self, design: StealthDesign) -> StealthDesign:
        """Polynomial mutation."""
        rng = np.random.default_rng()
        
        v = design.to_vector().astype(float)
        bounds = np.array([
            [2, 5],
            [1, 20],
            [1, 20],
            [0.01, 0.5],
            [0.5, 5.0],
        ])
        
        eta = 20.0  # distribution index
        
        for i in range(len(v)):
            if rng.random() < self.mutation_prob:
                y = v[i]
                yl, yu = bounds[i]
                
                delta1 = (y - yl) / (yu - yl + 1e-14)
                delta2 = (yu - y) / (yu - yl + 1e-14)
                
                rand = rng.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(eta + 1.0))
                    deltaq = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(eta + 1.0))
                    deltaq = 1.0 - val**mut_pow
                
                v[i] = y + deltaq * (yu - yl)
                v[i] = np.clip(v[i], yl, yu)
        
        return StealthDesign.from_vector(v)
    
    def optimize(self) -> Dict[str, Any]:
        """Run NSGA-II optimization."""
        logger.info(f"Starting NSGA-II optimization: pop={self.pop_size}, gen={self.n_gen}")
        
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        for design in population:
            self.objectives.evaluate(design)
        
        # Evolution loop
        for gen in range(self.n_gen):
            # Create offspring
            offspring = []
            
            rng = np.random.default_rng(gen)
            
            for _ in range(self.pop_size // 2):
                # Tournament selection
                idx1, idx2 = rng.integers(0, self.pop_size, 2)
                parent1 = population[idx1] if population[idx1].fitness[0] < population[idx2].fitness[0] else population[idx2]
                
                idx3, idx4 = rng.integers(0, self.pop_size, 2)
                parent2 = population[idx3] if population[idx3].fitness[0] < population[idx4].fitness[0] else population[idx4]
                
                # Crossover
                if rng.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            for design in offspring:
                self.objectives.evaluate(design)
            
            # Combine parent + offspring
            combined = population + offspring
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined)
            
            # Select next generation
            population = []
            for front in fronts:
                if len(population) + len(front) <= self.pop_size:
                    population.extend([combined[i] for i in front])
                else:
                    # Need to select based on crowding distance
                    remaining = self.pop_size - len(population)
                    distances = self._crowding_distance(combined, front)
                    sorted_idx = np.argsort(-distances)[:remaining]
                    population.extend([combined[front[i]] for i in sorted_idx])
                    break
            
            if (gen + 1) % 10 == 0:
                logger.info(f"Generation {gen+1}/{self.n_gen}")
        
        # Final non-dominated sorting
        fronts = self._non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]] if fronts else []
        
        return {
            "pareto_front": pareto_front,
            "n_generations": self.n_gen,
            "population_size": self.pop_size,
            "n_pareto_solutions": len(pareto_front),
        }


# ═══════════════════════════════════════════════════════════════════════
# PARETO FRONT ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class ParetoFrontAnalyzer:
    """
    [D] Analyze and visualize Pareto front.
    """
    
    def __init__(self, pareto_front: List[StealthDesign]):
        self.pareto_front = pareto_front
    
    def get_trade_off_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trade-off curve (RCS vs absorption)."""
        rcs = np.array([d.rcs_reduction_db for d in self.pareto_front])
        abs_vals = np.array([d.absorption_mean for d in self.pareto_front])
        
        # Sort by RCS
        sort_idx = np.argsort(rcs)
        return rcs[sort_idx], abs_vals[sort_idx]
    
    def select_knee_point(self) -> StealthDesign:
        """
        Select knee point (maximum curvature) as balanced solution.
        """
        if not self.pareto_front:
            return None
        
        rcs, abs_vals = self.get_trade_off_curve()
        
        # Normalize
        rcs_norm = (rcs - rcs.min()) / (rcs.max() - rcs.min() + 1e-12)
        abs_norm = (abs_vals - abs_vals.min()) / (abs_vals.max() - abs_vals.min() + 1e-12)
        
        # Distance from ideal point (min RCS, max absorption)
        distances = np.sqrt(rcs_norm**2 + (1 - abs_norm)**2)
        
        # Knee point: maximum distance from line connecting endpoints
        p1 = np.array([rcs_norm[0], abs_norm[0]])
        p2 = np.array([rcs_norm[-1], abs_norm[-1]])
        
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-12:
            return self.pareto_front[0]
        
        line_unit = line_vec / line_len
        
        max_dist = 0
        knee_idx = 0
        
        for i in range(len(rcs_norm)):
            p = np.array([rcs_norm[i], abs_norm[i]])
            v = p - p1
            proj = np.dot(v, line_unit) * line_unit
            perp = v - proj
            dist = np.linalg.norm(perp)
            
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return self.pareto_front[knee_idx]
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary of Pareto front."""
        if not self.pareto_front:
            return {"error": "empty Pareto front"}
        
        rcs_vals = [d.rcs_reduction_db for d in self.pareto_front]
        abs_vals = [d.absorption_mean for d in self.pareto_front]
        
        knee = self.select_knee_point()
        
        return {
            "n_solutions": len(self.pareto_front),
            "rcs_range_db": (min(rcs_vals), max(rcs_vals)),
            "absorption_range": (min(abs_vals), max(abs_vals)),
            "knee_point": {
                "design": knee,
                "rcs_reduction_db": knee.rcs_reduction_db if knee else None,
                "absorption": knee.absorption_mean if knee else None,
            },
            "note": "Pareto front shows trade-off; knee point is balanced solution",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Multi-Objective Stealth Optimization — Smoke Test")
    print("=" * 72)
    
    objectives = StealthObjectiveFunctions()
    optimizer = NSGA2Optimizer(
        objectives=objectives,
        population_size=50,
        n_generations=20,
    )
    
    print("\n  ── Running NSGA-II optimization ──")
    result = optimizer.optimize()
    
    print(f"    Generations        : {result['n_generations']}")
    print(f"    Population size    : {result['population_size']}")
    print(f"    Pareto solutions   : {result['n_pareto_solutions']}")
    
    # Analyze Pareto front
    analyzer = ParetoFrontAnalyzer(result["pareto_front"])
    summary = analyzer.summary()
    
    print(f"\n  ── Pareto Front Summary ──")
    print(f"    RCS range (dB)     : {summary['rcs_range_db']}")
    print(f"    Absorption range   : {summary['absorption_range']}")
    
    knee = summary["knee_point"]["design"]
    if knee:
        print(f"\n  ── Knee Point (Balanced Solution) ──")
        print(f"    Carpet level       : {knee.carpet_level}")
        print(f"    ε_r                : {knee.eps_r:.2f}")
        print(f"    μ_r                : {knee.mu_r:.2f}")
        print(f"    tan δ              : {knee.tan_d:.3f}")
        print(f"    Thickness (mm)     : {knee.thickness_mm:.2f}")
        print(f"    RCS reduction (dB) : {knee.rcs_reduction_db:.2f}")
        print(f"    Mean absorption    : {knee.absorption_mean:.3f}")
    
    print(f"\n    {summary['note']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • NSGA-II gives PARETO FRONT, not single optimum")
    print("  • Designer must choose trade-off based on requirements")
    print("  • Optimization is HEURISTIC; no global optimum guarantee")
    print("  • Results depend on objective function quality")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
