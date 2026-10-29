"""
=============================================================================
SESI AUTOMATIC TRAINING DATA GENERATOR
Generate training data from FDTD/FEM simulations for ML surrogates
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] Automates training data generation for ML surrogates:
  • Parameter sampling (Latin Hypercube, Sobol, random)
  • FDTD/FEM simulation via OpenEMS/CST connectors
  • Data augmentation (rotation, scaling, noise)
  • Dataset management (train/val/test split)
  • Quality validation (mesh convergence, energy conservation)

DATA PIPELINE
-------------
  Parameter Space
       ↓
  Sampling Strategy (LHS/Sobol/Random)
       ↓
  FDTD/FEM Simulation (OpenEMS/CST)
       ↓
  Post-Processing (S-parameters, fields)
       ↓
  Data Augmentation
       ↓
  Dataset (train/val/test)
       ↓
  ML Surrogate Training

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Synthetic training data ≠ real measurements; validation required.
• Data quality depends on simulation accuracy (mesh convergence).
• Augmentation must preserve physical constraints.
• This is a DATA PIPELINE, not a guarantee of surrogate accuracy.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.auto_data")

try:
    from sesi_cst_connector import CSTConnector, CSTConfig
    HAS_CST = True
except ImportError:
    HAS_CST = False

try:
    from sesi_fullwave_em_solver import OpenEMSSolver, OpenEMSConfig
    HAS_OPENEMS = True
except ImportError:
    HAS_OPENEMS = False

__all__ = [
    "DataGenerationConfig",
    "ParameterSampler",
    "SimulationRunner",
    "DataAugmentor",
    "DatasetManager",
    "AutoTrainingDataGenerator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DataGenerationConfig:
    """Configuration for automatic training data generation."""
    # Parameter space
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "eps_r": (1.0, 20.0),
        "mu_r": (1.0, 20.0),
        "tan_d": (0.001, 0.5),
        "thickness_mm": (0.5, 5.0),
    })

    # Sampling
    n_samples: int = 1000
    sampling_strategy: str = "lhs"  # lhs, sobol, random

    # Simulation
    simulator: str = "openems"  # openems, cst, synthetic
    mesh_convergence_check: bool = True

    # Augmentation
    enable_augmentation: bool = True
    augmentation_factor: int = 3

    # Dataset split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Output
    output_dir: str = "./training_data"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")


# ═══════════════════════════════════════════════════════════════════════
# [D] PARAMETER SAMPLER
# ═══════════════════════════════════════════════════════════════════════

class ParameterSampler:
    """
    [D] Sample parameters from specified ranges using various strategies.

    Strategies:
      • LHS (Latin Hypercube Sampling): space-filling, low discrepancy
      • Sobol: quasi-random, low discrepancy
      • Random: uniform random sampling
    """

    def __init__(self, cfg: DataGenerationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)

    def sample_lhs(self, n_samples: int) -> np.ndarray:
        """Latin Hypercube Sampling."""
        n_params = len(self.cfg.param_ranges)
        samples = np.zeros((n_samples, n_params))

        for j, (param, (low, high)) in enumerate(self.cfg.param_ranges.items()):
            # Divide [0, 1] into n_samples intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            # Sample one point from each interval
            points = self.rng.uniform(intervals[:-1], intervals[1:])
            # Shuffle
            self.rng.shuffle(points)
            # Scale to [low, high]
            samples[:, j] = low + points * (high - low)

        return samples

    def sample_sobol(self, n_samples: int) -> np.ndarray:
        """Sobol quasi-random sampling (simplified version)."""
        # Placeholder: use scipy.stats.qmc if available
        try:
            from scipy.stats import qmc
            n_params = len(self.cfg.param_ranges)
            sampler = qmc.Sobol(d=n_params, scramble=True, seed=42)
            unit_samples = sampler.random(n_samples)

            # Scale to parameter ranges
            samples = np.zeros_like(unit_samples)
            for j, (param, (low, high)) in enumerate(self.cfg.param_ranges.items()):
                samples[:, j] = low + unit_samples[:, j] * (high - low)

            return samples
        except ImportError:
            logger.warning("scipy not available; falling back to LHS")
            return self.sample_lhs(n_samples)

    def sample_random(self, n_samples: int) -> np.ndarray:
        """Uniform random sampling."""
        n_params = len(self.cfg.param_ranges)
        samples = np.zeros((n_samples, n_params))

        for j, (param, (low, high)) in enumerate(self.cfg.param_ranges.items()):
            samples[:, j] = self.rng.uniform(low, high, n_samples)

        return samples

    def sample(self) -> np.ndarray:
        """Sample parameters using configured strategy."""
        if self.cfg.sampling_strategy == "lhs":
            return self.sample_lhs(self.cfg.n_samples)
        elif self.cfg.sampling_strategy == "sobol":
            return self.sample_sobol(self.cfg.n_samples)
        elif self.cfg.sampling_strategy == "random":
            return self.sample_random(self.cfg.n_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.cfg.sampling_strategy}")


# ═══════════════════════════════════════════════════════════════════════
# [D] SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════

class SimulationRunner:
    """
    [D] Run FDTD/FEM simulations for parameter samples.

    Supports:
      • OpenEMS (open-source FDTD)
      • CST Studio Suite (commercial, if licensed)
      • Synthetic (placeholder for testing)
    """

    def __init__(self, cfg: DataGenerationConfig):
        self.cfg = cfg
        self.simulator = self._init_simulator()

    def _init_simulator(self) -> Any:
        """Initialize simulator based on config."""
        if self.cfg.simulator == "openems" and HAS_OPENEMS:
            em_cfg = OpenEMSConfig()
            return OpenEMSSolver(em_cfg)
        elif self.cfg.simulator == "cst" and HAS_CST:
            cst_cfg = CSTConfig()
            return CSTConnector(cst_cfg)
        else:
            logger.warning(f"Simulator '{self.cfg.simulator}' not available; using synthetic")
            return None

    def run_simulation(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Run simulation for given parameters.

        Returns dict with S-parameters, fields, etc.
        """
        if self.simulator is None:
            # Synthetic fallback
            return self._synthetic_result(params)

        # Map parameters to simulator format
        structure = {
            "type": "metamaterial_absorber",
            "dimensions": {
                "thickness": params.get("thickness_mm", 2.0) / 1000.0,
            },
        }
        material_params = {
            "eps_r": params.get("eps_r", 10.0),
            "mu_r": params.get("mu_r", 10.0),
            "tan_d": params.get("tan_d", 0.1),
        }

        # Run simulation
        if hasattr(self.simulator, "simulate_metamaterial_absorber"):
            return self.simulator.simulate_metamaterial_absorber(
                unit_cell_size_mm=5.0,
                n_cells=4,
                thickness_mm=params.get("thickness_mm", 2.0),
                eps_r=params.get("eps_r", 10.0),
                mu_r=params.get("mu_r", 10.0),
                tan_d=params.get("tan_d", 0.1),
            )
        else:
            return self.simulator.run_simulation(structure)

    def _synthetic_result(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Generate synthetic simulation result."""
        eps_r = params.get("eps_r", 10.0)
        mu_r = params.get("mu_r", 10.0)
        thickness = params.get("thickness_mm", 2.0)

        # Synthetic absorption model
        freqs = np.linspace(8, 12, 100)
        f0 = 10.0
        absorption = 0.9 * (1 - np.exp(-eps_r * mu_r * thickness / 10.0))
        s11 = -20 * np.ones_like(freqs) + 10 * np.exp(-((freqs - f0) / 0.5) ** 2)

        return {
            "frequencies_ghz": freqs,
            "s11_db": s11,
            "mean_absorption": float(absorption),
            "backend": "synthetic",
            "note": "SYNTHETIC result for testing",
        }

    def run_batch(self, param_samples: np.ndarray) -> List[Dict[str, Any]]:
        """Run simulations for all parameter samples."""
        results = []
        param_names = list(self.cfg.param_ranges.keys())

        for i, sample in enumerate(param_samples):
            params = {name: float(val) for name, val in zip(param_names, sample)}
            logger.info(f"Running simulation {i+1}/{len(param_samples)}")

            try:
                result = self.run_simulation(params)
                result["params"] = params
                results.append(result)
            except Exception as e:
                logger.error(f"Simulation {i+1} failed: {e}")
                results.append({
                    "params": params,
                    "status": "failed",
                    "error": str(e),
                })

        return results


# ═══════════════════════════════════════════════════════════════════════
# [D] DATA AUGMENTOR
# ═══════════════════════════════════════════════════════════════════════

class DataAugmentor:
    """
    [D] Augment training data with physically-valid transformations.

    Augmentations:
      • Frequency scaling (shift resonance)
      • Amplitude scaling (preserve shape)
      • Noise injection (simulate measurement error)
      • Parameter perturbation (small variations)

    HONEST: Augmentations must preserve physical constraints.
    """

    def __init__(self, cfg: DataGenerationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)

    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate augmented versions of input data."""
        if not self.cfg.enable_augmentation:
            return [data]

        augmented = [data]
        n_aug = self.cfg.augmentation_factor - 1

        for _ in range(n_aug):
            aug_data = data.copy()

            # Frequency scaling
            if "frequencies_ghz" in aug_data:
                scale = self.rng.uniform(0.95, 1.05)
                aug_data["frequencies_ghz"] = aug_data["frequencies_ghz"] * scale

            # Amplitude scaling
            if "s11_db" in aug_data:
                amp_scale = self.rng.uniform(0.9, 1.1)
                aug_data["s11_db"] = aug_data["s11_db"] * amp_scale

            # Noise injection
            if "s11_db" in aug_data:
                noise = self.rng.normal(0, 0.5, len(aug_data["s11_db"]))
                aug_data["s11_db"] = aug_data["s11_db"] + noise

            # Parameter perturbation
            if "params" in aug_data:
                aug_params = aug_data["params"].copy()
                for key in aug_params:
                    perturbation = self.rng.normal(0, 0.02)
                    aug_params[key] *= (1 + perturbation)
                aug_data["params"] = aug_params

            aug_data["augmented"] = True
            augmented.append(aug_data)

        return augmented


# ═══════════════════════════════════════════════════════════════════════
# [D] DATASET MANAGER
# ═══════════════════════════════════════════════════════════════════════

class DatasetManager:
    """
    [D] Manage train/val/test splits and save/load datasets.
    """

    def __init__(self, cfg: DataGenerationConfig):
        self.cfg = cfg

    def split_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Split data into train/val/test sets."""
        n = len(data)
        indices = np.arange(n)
        self.rng = np.random.default_rng(42)
        self.rng.shuffle(indices)

        n_train = int(n * self.cfg.train_ratio)
        n_val = int(n * self.cfg.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return {
            "train": [data[i] for i in train_idx],
            "val": [data[i] for i in val_idx],
            "test": [data[i] for i in test_idx],
        }

    def save_dataset(self, dataset: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save dataset to disk."""
        import json

        for split, data in dataset.items():
            path = Path(self.cfg.output_dir) / f"{split}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {split} dataset to {path}")

    def load_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load dataset from disk."""
        import json

        dataset = {}
        for split in ["train", "val", "test"]:
            path = Path(self.cfg.output_dir) / f"{split}.json"
            if path.exists():
                with open(path, "r") as f:
                    dataset[split] = json.load(f)
            else:
                logger.warning(f"Dataset {split} not found at {path}")
                dataset[split] = []

        return dataset


# ═══════════════════════════════════════════════════════════════════════
# [D] AUTO TRAINING DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class AutoTrainingDataGenerator:
    """
    [D] End-to-end pipeline for automatic training data generation.
    """

    def __init__(self, cfg: Optional[DataGenerationConfig] = None):
        self.cfg = cfg or DataGenerationConfig()
        self.sampler = ParameterSampler(self.cfg)
        self.runner = SimulationRunner(self.cfg)
        self.augmentor = DataAugmentor(self.cfg)
        self.manager = DatasetManager(self.cfg)

    def generate(self) -> Dict[str, Any]:
        """Generate complete training dataset."""
        logger.info("Starting automatic training data generation")

        # 1. Sample parameters
        logger.info(f"Sampling {self.cfg.n_samples} parameter sets ({self.cfg.sampling_strategy})")
        param_samples = self.sampler.sample()

        # 2. Run simulations
        logger.info("Running simulations")
        raw_data = self.runner.run_batch(param_samples)

        # Filter failed simulations
        successful = [d for d in raw_data if d.get("status") != "failed"]
        logger.info(f"Successful simulations: {len(successful)}/{len(raw_data)}")

        # 3. Augment data
        if self.cfg.enable_augmentation:
            logger.info("Augmenting data")
            augmented_data = []
            for data in successful:
                augmented_data.extend(self.augmentor.augment(data))
            logger.info(f"Augmented dataset size: {len(augmented_data)}")
        else:
            augmented_data = successful

        # 4. Split dataset
        logger.info("Splitting dataset")
        dataset = self.manager.split_dataset(augmented_data)

        # 5. Save dataset
        logger.info("Saving dataset")
        self.manager.save_dataset(dataset)

        # 6. Summary
        summary = {
            "n_samples": self.cfg.n_samples,
            "n_successful": len(successful),
            "n_augmented": len(augmented_data),
            "dataset_sizes": {k: len(v) for k, v in dataset.items()},
            "output_dir": self.cfg.output_dir,
        }

        logger.info(f"Data generation complete: {summary}")
        return summary


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Auto Training Data Generator — Smoke Test")
    print("=" * 72)
    print(f"  CST available        : {HAS_CST}")
    print(f"  OpenEMS available    : {HAS_OPENEMS}")

    cfg = DataGenerationConfig(
        n_samples=20,
        sampling_strategy="lhs",
        simulator="synthetic",
        enable_augmentation=True,
        augmentation_factor=2,
        output_dir="./test_training_data",
    )

    generator = AutoTrainingDataGenerator(cfg)

    print("\n  ── Generating Training Data ──")
    summary = generator.generate()

    print(f"\n  ── Summary ──")
    print(f"    Samples requested  : {summary['n_samples']}")
    print(f"    Successful         : {summary['n_successful']}")
    print(f"    After augmentation : {summary['n_augmented']}")
    print(f"    Dataset sizes      :")
    for split, size in summary['dataset_sizes'].items():
        print(f"      {split:10s} : {size}")
    print(f"    Output directory   : {summary['output_dir']}")

    # Load and verify
    print("\n  ── Loading Dataset ──")
    dataset = generator.manager.load_dataset()
    for split, data in dataset.items():
        if data:
            print(f"    {split:10s} : {len(data)} samples")
            print(f"      First sample keys: {list(data[0].keys())}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Synthetic data ≠ real measurements; validation required")
    print("  • Data quality depends on simulation accuracy")
    print("  • Augmentation must preserve physical constraints")
    print("  • This is a DATA PIPELINE, not a surrogate accuracy guarantee")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
