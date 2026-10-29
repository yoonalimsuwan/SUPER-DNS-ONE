"""
=============================================================================
SESI PARAMETRIC SWEEP AUTOMATION
Batch simulation over parameter space for design optimization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] Automates batch simulations over parameter ranges:
  • Material properties (ε_r, μ_r, tan δ)
  • Geometric parameters (thickness, unit cell size)
  • Frequency sweeps
  • Multi-parameter combinations

FEATURES
--------
  • Parallel execution (multiprocessing)
  • Progress tracking
  • Result aggregation and analysis
  • Early stopping based on convergence criteria

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Parametric sweeps are COMPUTATIONAL EXPLORATIONS, not proofs.
• Results depend on simulation accuracy and mesh convergence.
• Large parameter spaces require significant computational resources.
• Parallel execution may have overhead; tune based on hardware.
=============================================================================
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.parametric_sweep")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import simulation backend
try:
    from sesi_cst_connector import CSTConnector, CSTConfig
    HAS_CST = True
except ImportError:
    HAS_CST = False

__all__ = [
    "ParameterRange",
    "SweepConfig",
    "ParametricSweeper",
    "SweepResultAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ParameterRange:
    """Definition of a parameter to sweep."""
    name: str
    values: List[float]
    unit: str = ""
    
    def __len__(self):
        return len(self.values)


@dataclass
class SweepConfig:
    """Configuration for parametric sweep."""
    # Parameters to sweep
    parameters: List[ParameterRange] = field(default_factory=list)
    
    # Simulation settings
    sim_config: Optional[Any] = None  # CSTConfig or similar
    
    # Execution
    n_workers: int = 4  # number of parallel workers
    timeout_per_sim_s: int = 3600  # 1 hour timeout
    
    # Output
    output_dir: str = "./sweep_results"
    save_intermediate: bool = True
    
    # Early stopping
    enable_early_stopping: bool = False
    convergence_threshold: float = 0.01  # stop if improvement < threshold


# ═══════════════════════════════════════════════════════════════════════
# [D] PARAMETRIC SWEEPER
# ═══════════════════════════════════════════════════════════════════════

class ParametricSweeper:
    """
    [D] Automates batch simulations over parameter space.
    """
    
    def __init__(self, cfg: SweepConfig,
                 simulation_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.simulation_fn = simulation_fn or self._default_simulation_fn
        self.results: List[Dict[str, Any]] = []
    
    def _default_simulation_fn(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Default simulation function (placeholder).
        
        In production, this would call CST/OpenEMS solver.
        """
        # Synthetic result
        eps_r = params.get("eps_r", 10.0)
        thickness = params.get("thickness_mm", 2.0)
        
        # Simple absorption model
        absorption = 0.9 * (1 - np.exp(-eps_r * thickness / 10.0))
        
        return {
            "params": params,
            "absorption": float(absorption),
            "rcs_reduction_db": float(10 * np.log10(1 + eps_r)),
            "status": "completed",
        }
    
    def _generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate all combinations of parameter values."""
        if not self.cfg.parameters:
            return [{}]
        
        # Start with first parameter
        combos = [{p.name: v} for v in p.values]
        
        # Iterate over remaining parameters
        for p in self.cfg.parameters[1:]:
            new_combos = []
            for combo in combos:
                for v in p.values:
                    new_combo = combo.copy()
                    new_combo[p.name] = v
                    new_combos.append(new_combo)
            combos = new_combos
        
        return combos
    
    def _run_single_simulation(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Run a single simulation with given parameters."""
        try:
            result = self.simulation_fn(params)
            result["params"] = params
            return result
        except Exception as e:
            logger.error(f"Simulation failed for {params}: {e}")
            return {
                "params": params,
                "status": "failed",
                "error": str(e),
            }
    
    def run_sweep(self) -> List[Dict[str, Any]]:
        """
        Run full parametric sweep.
        
        Returns list of results (one per parameter combination).
        """
        combos = self._generate_parameter_combinations()
        n_total = len(combos)
        
        logger.info(f"Starting parametric sweep: {n_total} simulations")
        
        if self.cfg.n_workers > 1:
            # Parallel execution
            with mp.Pool(processes=self.cfg.n_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._run_single_simulation, combos),
                    total=n_total,
                    desc="Sweep progress"
                )) if HAS_TQDM else pool.map(self._run_single_simulation, combos)
        else:
            # Sequential execution
            results = []
            for i, params in enumerate(combos):
                if HAS_TQDM:
                    logger.info(f"Progress: {i+1}/{n_total}")
                result = self._run_single_simulation(params)
                results.append(result)
                
                # Early stopping check
                if self.cfg.enable_early_stopping and len(results) > 10:
                    recent = results[-10:]
                    if self._check_convergence(recent):
                        logger.info("Early stopping: convergence reached")
                        break
        
        self.results = results
        logger.info(f"Sweep completed: {len(results)} results")
        
        return results
    
    def _check_convergence(self, recent_results: List[Dict[str, Any]]) -> bool:
        """Check if results have converged (for early stopping)."""
        if len(recent_results) < 2:
            return False
        
        # Check variance of key metric (e.g., absorption)
        absorptions = [r.get("absorption", 0) for r in recent_results if r.get("status") == "completed"]
        
        if len(absorptions) < 2:
            return False
        
        variance = np.var(absorptions)
        mean = np.mean(absorptions)
        
        # Coefficient of variation
        cv = variance / (mean + 1e-12)
        
        return cv < self.cfg.convergence_threshold
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save sweep results to file."""
        import json
        from pathlib import Path
        
        output_path = output_path or f"{self.cfg.output_dir}/sweep_results.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        serializable_results = [
            {k: convert(v) for k, v in r.items()}
            for r in self.results
        ]
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════
# [D] SWEEP RESULT ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class SweepResultAnalyzer:
    """
    [D] Analyze and visualize parametric sweep results.
    """
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
    
    def find_optimal(self, metric: str = "absorption",
                     maximize: bool = True) -> Dict[str, Any]:
        """Find parameter combination that optimizes given metric."""
        valid_results = [r for r in self.results if r.get("status") == "completed"]
        
        if not valid_results:
            return {"error": "no valid results"}
        
        if maximize:
            best = max(valid_results, key=lambda r: r.get(metric, 0))
        else:
            best = min(valid_results, key=lambda r: r.get(metric, 0))
        
        return {
            "optimal_params": best["params"],
            "optimal_value": best.get(metric, 0),
            "metric": metric,
        }
    
    def sensitivity_analysis(self, metric: str = "absorption") -> Dict[str, Any]:
        """
        Perform sensitivity analysis: how does metric vary with each parameter?
        """
        valid_results = [r for r in self.results if r.get("status") == "completed"]
        
        if not valid_results:
            return {"error": "no valid results"}
        
        # Extract unique parameter values and corresponding metrics
        param_names = list(valid_results[0]["params"].keys())
        sensitivity = {}
        
        for param_name in param_names:
            # Group by parameter value
            groups = {}
            for r in valid_results:
                val = r["params"][param_name]
                if val not in groups:
                    groups[val] = []
                groups[val].append(r.get(metric, 0))
            
            # Compute mean and std for each group
            param_values = sorted(groups.keys())
            means = [np.mean(groups[v]) for v in param_values]
            stds = [np.std(groups[v]) for v in param_values]
            
            sensitivity[param_name] = {
                "values": param_values,
                "means": means,
                "stds": stds,
            }
        
        return sensitivity
    
    def plot_sensitivity(self, metric: str = "absorption",
                         output_path: Optional[str] = None) -> Any:
        """Plot sensitivity analysis results."""
        import matplotlib.pyplot as plt
        
        sensitivity = self.sensitivity_analysis(metric)
        
        if "error" in sensitivity:
            return None
        
        n_params = len(sensitivity)
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for idx, (param_name, data) in enumerate(sensitivity.items()):
            ax = axes[idx]
            ax.errorbar(data["values"], data["means"], yerr=data["stds"],
                       marker='o', capsize=3)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.set_title(f"Sensitivity: {param_name}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Parametric Sweep Automation — Smoke Test")
    print("=" * 72)
    
    # Define parameter ranges
    eps_r_range = ParameterRange(name="eps_r", values=[2.2, 4.4, 10.0, 20.0], unit="")
    thickness_range = ParameterRange(name="thickness_mm", values=[1.0, 2.0, 3.0, 4.0], unit="mm")
    
    cfg = SweepConfig(
        parameters=[eps_r_range, thickness_range],
        n_workers=2,
        output_dir="./sweep_results",
    )
    
    sweeper = ParametricSweeper(cfg)
    
    print("\n  ── Running parametric sweep ──")
    results = sweeper.run_sweep()
    
    print(f"    Total simulations    : {len(results)}")
    print(f"    Successful           : {sum(1 for r in results if r['status'] == 'completed')}")
    
    # Analyze results
    analyzer = SweepResultAnalyzer(results)
    
    print("\n  ── Optimal design ──")
    optimal = analyzer.find_optimal(metric="absorption", maximize=True)
    print(f"    Optimal ε_r          : {optimal['optimal_params'].get('eps_r', 'N/A')}")
    print(f"    Optimal thickness    : {optimal['optimal_params'].get('thickness_mm', 'N/A')} mm")
    print(f"    Max absorption       : {optimal['optimal_value']:.4f}")
    
    print("\n  ── Sensitivity analysis ──")
    sensitivity = analyzer.sensitivity_analysis(metric="absorption")
    for param, data in sensitivity.items():
        print(f"    {param}:")
        for v, m, s in zip(data["values"], data["means"], data["stds"]):
            print(f"      {v}: {m:.4f} ± {s:.4f}")
    
    # Save results
    output_path = sweeper.save_results()
    print(f"\n    Results saved to     : {output_path}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Parametric sweeps are COMPUTATIONAL EXPLORATIONS")
    print("  • Results depend on simulation accuracy")
    print("  • Large parameter spaces require significant resources")
    print("  • Early stopping is heuristic; verify convergence")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
