"""
=============================================================================
SESI MULTI-SENSOR FUSION ENGINE
Bayesian + Dempster-Shafer fusion of EM, GW, and other sensors
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[F1] Bayesian fusion of multiple sensor observations
[F2] Dempster-Shafer evidence combination
[F3] Temporal correlation across sensors
[F4] Anomaly classification with uncertainty quantification

HONEST SCOPE
------------
• Multi-sensor fusion is HEURISTIC; no claim of definitive detection.
• Bayesian priors are subjective; results depend on prior choice.
• Dempster-Shafer assumes sensor independence (may not hold).
• All results require independent verification.

LABEL CONVENTION
----------------
[P] proved  [C] cited  [D] derived  [O] open
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.multi_sensor")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "SensorType",
    "SensorObservation",
    "FusionConfig",
    "BayesianFusion",
    "DempsterShaferFusion",
    "MultiSensorFusionEngine",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

class SensorType(Enum):
    """Types of sensors in the fusion engine."""
    ELECTROMAGNETIC = auto()      # Optical, radio, X-ray, etc.
    GRAVITATIONAL_WAVE = auto()   # LIGO/Virgo/KAGRA
    NEUTRINO = auto()             # IceCube, Super-K
    COSMIC_RAY = auto()           # Pierre Auger
    UNKNOWN = auto()


@dataclass
class SensorObservation:
    """A single observation from a sensor."""
    sensor_type: SensorType
    timestamp: float              # MJD or GPS time
    position: Optional[Tuple[float, float, float]] = None  # (ra, dec, distance)
    signal_strength: float = 0.0  # SNR or flux
    uncertainty: float = 1.0      # 1-sigma uncertainty
    anomaly_score: float = 0.0    # from Carpet A^11 filter
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    """Configuration for multi-sensor fusion."""
    # Bayesian
    prior_anomaly: float = 1e-6   # prior probability of anomaly
    likelihood_ratio_signal: float = 10.0
    likelihood_ratio_noise: float = 0.1

    # Dempster-Shafer
    conflict_threshold: float = 0.9  # high conflict → distrust fusion

    # Temporal
    time_window: float = 1.0  # seconds for temporal correlation

    # Classification
    anomaly_threshold: float = 3.0  # sigma
    high_confidence_threshold: float = 0.9

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [F1] BAYESIAN FUSION
# ═══════════════════════════════════════════════════════════════════════

class BayesianFusion:
    """
    Bayesian fusion of multiple sensor observations.

    Uses Bayes' theorem to update posterior probability of anomaly
    given observations from multiple sensors.

    P(anomaly | data) ∝ P(data | anomaly) * P(anomaly)
    """

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg

    def update_posterior(
        self,
        prior: float,
        observations: List[SensorObservation],
    ) -> float:
        """
        Update posterior probability given observations.

        Args:
            prior: prior probability of anomaly
            observations: list of sensor observations

        Returns:
            posterior probability
        """
        posterior = prior
        for obs in observations:
            # Likelihood ratio
            if obs.anomaly_score > self.cfg.anomaly_threshold:
                likelihood_ratio = self.cfg.likelihood_ratio_signal
            else:
                likelihood_ratio = self.cfg.likelihood_ratio_noise

            # Bayes update
            posterior = (likelihood_ratio * posterior) / \
                       (likelihood_ratio * posterior + (1 - posterior))

        return posterior

    def fuse(self, observations: List[SensorObservation]) -> Dict[str, float]:
        """Fuse observations and return posterior statistics."""
        prior = self.cfg.prior_anomaly
        posterior = self.update_posterior(prior, observations)

        # Compute Bayes factor
        bf = (posterior / (1 - posterior)) / (prior / (1 - prior))

        return {
            "prior": prior,
            "posterior": posterior,
            "bayes_factor": bf,
            "log_bayes_factor": math.log(bf + 1e-30),
        }


# ═══════════════════════════════════════════════════════════════════════
# [F2] DEMPSTER-SHAFER FUSION
# ═══════════════════════════════════════════════════════════════════════

class DempsterShaferFusion:
    """
    Dempster-Shafer evidence combination.

    Combines belief functions from multiple sensors using Dempster's rule.
    Handles uncertainty and conflict explicitly.

    HONEST: Assumes sensor independence; may not hold in practice.
    """

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg

    def combine_mass_functions(
        self,
        mass_functions: List[Dict[str, float]],
    ) -> Tuple[Dict[str, float], float]:
        """
        Combine multiple mass functions using Dempster's rule.

        Args:
            mass_functions: list of dicts mapping hypotheses to mass

        Returns:
            (combined_mass, conflict)
        """
        if not mass_functions:
            return {}, 0.0

        combined = mass_functions[0].copy()
        total_conflict = 0.0

        for m2 in mass_functions[1:]:
            new_combined = {}
            conflict = 0.0

            for h1, m1 in combined.items():
                for h2, m2_val in m2.items():
                    # Intersection
                    if h1 == h2:
                        h_intersect = h1
                    elif h1 == "ignorance" or h2 == "ignorance":
                        h_intersect = h1 if h1 != "ignorance" else h2
                    else:
                        h_intersect = "empty"

                    mass_product = m1 * m2_val

                    if h_intersect == "empty":
                        conflict += mass_product
                    else:
                        new_combined[h_intersect] = new_combined.get(h_intersect, 0.0) + mass_product

            # Normalize
            total_conflict += conflict
            normalization = 1.0 - conflict
            if normalization > 1e-10:
                combined = {k: v / normalization for k, v in new_combined.items()}
            else:
                combined = {"ignorance": 1.0}

        return combined, total_conflict

    def observation_to_mass(self, obs: SensorObservation) -> Dict[str, float]:
        """Convert observation to mass function."""
        if obs.anomaly_score > self.cfg.anomaly_threshold:
            # Strong evidence for anomaly
            return {"anomaly": 0.8, "no_anomaly": 0.1, "ignorance": 0.1}
        elif obs.anomaly_score > self.cfg.anomaly_threshold * 0.5:
            # Weak evidence
            return {"anomaly": 0.3, "no_anomaly": 0.3, "ignorance": 0.4}
        else:
            # Evidence against anomaly
            return {"anomaly": 0.1, "no_anomaly": 0.8, "ignorance": 0.1}

    def fuse(self, observations: List[SensorObservation]) -> Dict[str, Any]:
        """Fuse observations using Dempster-Shafer."""
        mass_functions = [self.observation_to_mass(obs) for obs in observations]
        combined, conflict = self.combine_mass_functions(mass_functions)

        # Compute belief and plausibility
        belief_anomaly = combined.get("anomaly", 0.0)
        plausibility_anomaly = belief_anomaly + combined.get("ignorance", 0.0)

        return {
            "belief_anomaly": belief_anomaly,
            "plausibility_anomaly": plausibility_anomaly,
            "conflict": conflict,
            "high_conflict": conflict > self.cfg.conflict_threshold,
            "combined_mass": combined,
        }


# ═══════════════════════════════════════════════════════════════════════
# [F3-F4] MULTI-SENSOR FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class MultiSensorFusionEngine:
    """
    Unified multi-sensor fusion engine.

    Combines Bayesian and Dempster-Shafer approaches with temporal
    correlation and anomaly classification.
    """

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.bayesian = BayesianFusion(cfg)
        self.dempster_shafer = DempsterShaferFusion(cfg)
        self.observations: List[SensorObservation] = []

    def add_observation(self, obs: SensorObservation) -> None:
        """Add a sensor observation."""
        self.observations.append(obs)

    def temporal_correlation(self, window: Optional[float] = None) -> List[List[SensorObservation]]:
        """Group observations by temporal proximity."""
        if not self.observations:
            return []

        window = window or self.cfg.time_window
        sorted_obs = sorted(self.observations, key=lambda o: o.timestamp)

        groups = []
        current_group = [sorted_obs[0]]

        for obs in sorted_obs[1:]:
            if abs(obs.timestamp - current_group[-1].timestamp) <= window:
                current_group.append(obs)
            else:
                groups.append(current_group)
                current_group = [obs]

        if current_group:
            groups.append(current_group)

        return groups

    def spatial_correlation(self, angular_threshold_deg: float = 1.0) -> List[List[SensorObservation]]:
        """Group observations by spatial proximity."""
        if not self.observations:
            return []

        groups = []
        used = set()

        for i, obs1 in enumerate(self.observations):
            if i in used or obs1.position is None:
                continue

            group = [obs1]
            used.add(i)

            for j, obs2 in enumerate(self.observations):
                if j in used or obs2.position is None:
                    continue

                # Angular separation
                ra1, dec1, _ = obs1.position
                ra2, dec2, _ = obs2.position
                sep = math.acos(
                    math.sin(math.radians(dec1)) * math.sin(math.radians(dec2)) +
                    math.cos(math.radians(dec1)) * math.cos(math.radians(dec2)) *
                    math.cos(math.radians(ra1 - ra2))
                )
                sep_deg = math.degrees(sep)

                if sep_deg <= angular_threshold_deg:
                    group.append(obs2)
                    used.add(j)

            groups.append(group)

        return groups

    def fuse_all(self) -> Dict[str, Any]:
        """Fuse all observations and classify anomalies."""
        # Temporal grouping
        temporal_groups = self.temporal_correlation()

        results = []
        for group in temporal_groups:
            # Bayesian fusion
            bayes_result = self.bayesian.fuse(group)

            # Dempster-Shafer fusion
            ds_result = self.dempster_shafer.fuse(group)

            # Classification
            if bayes_result["posterior"] > self.cfg.high_confidence_threshold:
                classification = "HIGH_CONFIDENCE_ANOMALY"
            elif bayes_result["posterior"] > self.cfg.prior_anomaly * 10:
                classification = "MODERATE_ANOMALY"
            elif ds_result["belief_anomaly"] > 0.5:
                classification = "LOW_CONFIDENCE_ANOMALY"
            else:
                classification = "NO_ANOMALY"

            results.append({
                "n_observations": len(group),
                "sensor_types": list(set(obs.sensor_type.name for obs in group)),
                "bayesian": bayes_result,
                "dempster_shafer": ds_result,
                "classification": classification,
            })

        return {
            "total_observations": len(self.observations),
            "temporal_groups": len(temporal_groups),
            "results": results,
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Multi-Sensor Fusion Engine — Smoke Test")
    print("=" * 72)

    cfg = FusionConfig()
    engine = MultiSensorFusionEngine(cfg)

    # Add synthetic observations
    print("\n  ── Adding Synthetic Observations ──")
    observations = [
        SensorObservation(
            sensor_type=SensorType.ELECTROMAGNETIC,
            timestamp=59000.0,
            position=(180.0, 45.0, 100.0),
            anomaly_score=4.5,
        ),
        SensorObservation(
            sensor_type=SensorType.GRAVITATIONAL_WAVE,
            timestamp=59000.0 + 0.5/86400,  # 0.5 seconds later
            position=(180.1, 45.1, 100.0),
            anomaly_score=3.8,
        ),
        SensorObservation(
            sensor_type=SensorType.NEUTRINO,
            timestamp=59000.0 + 1.0/86400,  # 1 second later
            position=(180.0, 45.0, 100.0),
            anomaly_score=2.1,
        ),
    ]

    for obs in observations:
        engine.add_observation(obs)
        print(f"    Added {obs.sensor_type.name}: anomaly_score={obs.anomaly_score:.2f}")

    # Fuse
    print("\n  ── Fusion Results ──")
    results = engine.fuse_all()
    print(f"    Total observations: {results['total_observations']}")
    print(f"    Temporal groups: {results['temporal_groups']}")

    for i, result in enumerate(results["results"]):
        print(f"\n    Group {i+1}:")
        print(f"      Observations: {result['n_observations']}")
        print(f"      Sensor types: {result['sensor_types']}")
        print(f"      Bayesian posterior: {result['bayesian']['posterior']:.6f}")
        print(f"      Bayes factor: {result['bayesian']['bayes_factor']:.2f}")
        print(f"      DS belief: {result['dempster_shafer']['belief_anomaly']:.4f}")
        print(f"      DS conflict: {result['dempster_shafer']['conflict']:.4f}")
        print(f"      Classification: {result['classification']}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Multi-sensor fusion is HEURISTIC")
    print("  • Bayesian priors are subjective")
    print("  • Dempster-Shafer assumes sensor independence")
    print("  • No claim of definitive anomaly detection")
    print("  • All results require independent verification")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
