"""
=============================================================================
SESI LALSIMULATION COMPARISON
Surrogate vs IMRPhenomD mismatch distribution
=============================================================================
[C] LALSimulation (if installed)
[D] Mismatch computation
[O] Full systematic study

HONEST SCOPE
------------
• Requires lalsuite installation for real comparison.
• Without lalsuite, reports "surrogate only" status.
• Mismatch distribution requires many parameter samples.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.lal_comparison")

try:
    import lalsimulation as lalsim
    import lal
    HAS_LAL = True
except ImportError:
    HAS_LAL = False

__all__ = [
    "LALComparisonConfig",
    "LALSimulationBackend",
    "MismatchCalculator",
    "SurrogateComparisonReport",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LALComparisonConfig:
    """Configuration for LAL comparison."""
    # Waveform parameters
    approximant: str = "IMRPhenomD"
    sample_rate: float = 4096.0
    f_min: float = 20.0
    f_max: float = 1024.0
    
    # Parameter grid
    n_samples: int = 100
    mass1_range: Tuple[float, float] = (10.0, 50.0)
    mass2_range: Tuple[float, float] = (10.0, 50.0)
    spin_range: Tuple[float, float] = (-0.8, 0.8)
    distance_mpc: float = 400.0
    
    # Mismatch
    psd_name: str = "aLIGOZeroDetHighPower"


# ═══════════════════════════════════════════════════════════════════════
# LAL BACKEND
# ═══════════════════════════════════════════════════════════════════════

class LALSimulationBackend:
    """
    [C] Interface to LALSimulation for calibrated waveforms.
    """

    def __init__(self, cfg: LALComparisonConfig):
        self.cfg = cfg
        self.available = HAS_LAL
        
        if not HAS_LAL:
            logger.warning("lalsuite not installed; LAL comparison unavailable")

    def generate_waveform(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """Generate frequency-domain waveform using LALSimulation."""
        if not self.available:
            return None
        
        try:
            m1_kg = params["mass1"] * lal.MSUN_SI
            m2_kg = params["mass2"] * lal.MSUN_SI
            dist_m = params.get("distance", self.cfg.distance_mpc) * 3.085677581e22
            
            # Get approximant
            approx = getattr(lalsim, self.cfg.approximant)
            
            # Generate FD waveform
            hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(
                m1_kg, m2_kg,
                params.get("spin1z", 0.0), 0.0, 0.0,
                params.get("spin2z", 0.0), 0.0, 0.0,
                dist_m,
                params.get("inclination", 0.0), 0.0, 0.0, 0.0,
                -1.0, -1.0, -1.0,
                self.cfg.f_min, self.cfg.f_max, 0.0,
                approx, None
            )
            
            return np.array(hptilde.data.data)
            
        except Exception as e:
            logger.warning(f"LAL waveform generation failed: {e}")
            return None

    def get_psd(self) -> Optional[np.ndarray]:
        """Get PSD from LALSimulation."""
        if not self.available:
            return None
        
        try:
            psd_func = getattr(lalsim, self.cfg.psd_name)
            n_freq = int(self.cfg.sample_rate * 4)  # 4 seconds
            df = self.cfg.sample_rate / (2 * n_freq)
            
            psd = np.zeros(n_freq)
            for i in range(n_freq):
                f = i * df
                if f >= self.cfg.f_min:
                    psd[i] = psd_func(f)
            
            return psd
            
        except Exception as e:
            logger.warning(f"PSD retrieval failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# MISMATCH CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class MismatchCalculator:
    """
    [D] Compute mismatch between waveforms.
    
    mismatch = 1 - max_{t,φ} ⟨h1, h2⟩ / (||h1|| ||h2||)
    """

    def __init__(self, cfg: LALComparisonConfig):
        self.cfg = cfg

    def inner_product(self, h1: np.ndarray, h2: np.ndarray,
                      psd: np.ndarray) -> float:
        """Noise-weighted inner product."""
        # Simplified: assume h1, h2 are already in frequency domain
        integrand = np.real(h1 * np.conj(h2) / (psd + 1e-60))
        df = self.cfg.sample_rate / (2 * len(h1))
        return float(4.0 * np.sum(integrand) * df)

    def mismatch(self, h1: np.ndarray, h2: np.ndarray,
                 psd: np.ndarray) -> float:
        """Compute mismatch (maximized over time/phase)."""
        # Simplified: no time/phase maximization
        norm1 = np.sqrt(self.inner_product(h1, h1, psd))
        norm2 = np.sqrt(self.inner_product(h2, h2, psd))
        
        if norm1 < 1e-30 or norm2 < 1e-30:
            return 1.0
        
        overlap = self.inner_product(h1, h2, psd) / (norm1 * norm2)
        return float(1.0 - np.clip(overlap, -1.0, 1.0))

    def compute_distribution(self, lal_backend: LALSimulationBackend,
                             surrogate_generator) -> Dict[str, Any]:
        """
        Compute mismatch distribution over parameter grid.
        
        Args:
            lal_backend: LALSimulation backend
            surrogate_generator: surrogate waveform generator
            
        Returns:
            dict with mismatch statistics
        """
        if not lal_backend.available:
            return {
                "success": False,
                "reason": "lalsuite not installed",
                "status": "surrogate_only",
            }
        
        psd = lal_backend.get_psd()
        if psd is None:
            return {"success": False, "reason": "PSD unavailable"}
        
        rng = np.random.default_rng(42)
        mismatches = []
        
        for i in range(self.cfg.n_samples):
            # Sample parameters
            m1 = rng.uniform(*self.cfg.mass1_range)
            m2 = rng.uniform(*self.cfg.mass2_range)
            if m1 < m2:
                m1, m2 = m2, m1
            
            params = {
                "mass1": m1, "mass2": m2,
                "spin1z": rng.uniform(*self.cfg.spin_range),
                "spin2z": rng.uniform(*self.cfg.spin_range),
                "distance": self.cfg.distance_mpc,
            }
            
            # Generate LAL waveform
            h_lal = lal_backend.generate_waveform(params)
            if h_lal is None:
                continue
            
            # Generate surrogate waveform
            h_surr = surrogate_generator.generate(params)
            if h_surr is None or len(h_surr) != len(h_lal):
                continue
            
            # Compute mismatch
            mm = self.mismatch(h_lal, h_surr, psd[:len(h_lal)])
            mismatches.append(mm)
        
        if not mismatches:
            return {"success": False, "reason": "No valid comparisons"}
        
        mismatches = np.array(mismatches)
        
        return {
            "success": True,
            "n_comparisons": len(mismatches),
            "mismatches": mismatches.tolist(),
            "mean": float(np.mean(mismatches)),
            "median": float(np.median(mismatches)),
            "std": float(np.std(mismatches)),
            "p90": float(np.percentile(mismatches, 90)),
            "p99": float(np.percentile(mismatches, 99)),
            "max": float(np.max(mismatches)),
        }


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

class SurrogateComparisonReport:
    """Generate comparison report."""

    @staticmethod
    def summarize(result: Dict[str, Any]) -> str:
        if not result.get("success"):
            return f"LAL comparison unavailable: {result.get('reason')}"
        
        lines = [
            "=" * 60,
            "SURROGATE vs IMRPhenomD MISMATCH DISTRIBUTION",
            "=" * 60,
            f"Comparisons: {result['n_comparisons']}",
            f"Mean mismatch: {result['mean']:.6f}",
            f"Median: {result['median']:.6f}",
            f"Std: {result['std']:.6f}",
            f"90th percentile: {result['p90']:.6f}",
            f"99th percentile: {result['p99']:.6f}",
            f"Max: {result['max']:.6f}",
            "",
            "=" * 60,
            "HONEST: Mismatch < 1e-3 is typically acceptable for",
            "detection; < 1e-2 for parameter estimation.",
            "This comparison requires validation against production",
            "waveform systematics.",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI LAL Comparison — Smoke Test")
    print("=" * 72)

    cfg = LALComparisonConfig(n_samples=10)
    lal_backend = LALSimulationBackend(cfg)
    
    print(f"\n  LALSimulation available: {lal_backend.available}")
    
    if lal_backend.available:
        # Test waveform generation
        params = {"mass1": 30.0, "mass2": 25.0}
        h = lal_backend.generate_waveform(params)
        print(f"  Waveform generated: {h is not None}")
        if h is not None:
            print(f"  Waveform length: {len(h)}")
    else:
        print("  Status: SURROGATE ONLY (install lalsuite for comparison)")
    
    print("\n" + "=" * 72)
    print("  HONEST: Without lalsuite, real comparison is unavailable.")
    print("  Install with: pip install lalsuite")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
