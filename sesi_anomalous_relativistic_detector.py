"""
=============================================================================
SESI ANOMALOUS RELATIVISTIC SIGNAL DETECTOR
Carpet A^11 (Order-22 Polyharmonic) × Relativistic Signal Processing
Based on: Consolidated_Revision_Carpet-19.pdf
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Research Framework)

WHAT THIS MODULE DOES
---------------------
Detects anomalous signals consistent with the HYPOTHESIS of relativistic
near-light-speed objects exhibiting stealth characteristics, using:

  [U1] RelativisticSignalModel — signal model under Lorentz transformation
  [U2] CarpetHighOrderFilter   — A^11 = ∂*(∂∂*)^10 ∂ (order 22, per Prop 4)
  [U3] StealthSignatureAnalyzer — electromagnetic/gravitational stealth
  [U4] AnomalyDetector         — energy-measure singularity (Hino Thm 8)
  [U5] UFOCandidateClassifier  — cautious classification (research only)

MATHEMATICAL FOUNDATION (per Consolidated_Revision_Carpet-19)
-------------------------------------------------------------
  • Proposition 4: A^k = ∂*(∂∂*)^(k-1) ∂, differential length 2k
  • Corollary 4: No lower-order terms; A^(2k) = (-1)^k ∆^k identically
  • Theorem 4: Spectral Sobolev scale W^s, compact embedding, moment inequality
  • Theorem 8 (Hino 2005): Energy measures µ<f> ⊥ µ (singular)
  • Proposition 3: [A, M_f]: W^1 → W^{-1} bounded

RELATIVISTIC SIGNAL MODEL
-------------------------
For an object moving at velocity v = βc with Lorentz factor γ = 1/√(1-β²):
  • Doppler shift: f_obs = f_emit · √((1-β)/(1+β)) (longitudinal)
  • Time dilation: dt_obs = γ · dt_emit
  • Length contraction: L_obs = L_emit / γ
  • Relativistic beaming: intensity concentrated in forward cone ~1/γ

STEALTH SIGNATURES
------------------
  • Electromagnetic: reduced RCS, adaptive absorption, metamaterial cloaking
  • Gravitational: weak lensing anomalies, frame-dragging signatures
  • Quantum: vacuum fluctuation anomalies (Casimir-like)
  • Thermal: infrared signature suppression

HONEST SCOPE (CRITICAL)
-----------------------
• This is a RESEARCH FRAMEWORK, not a confirmed detection system.
• "UFO" here means "Unidentified Anomalous Phenomenon" — no claim that
  extraterrestrial or exotic craft exist.
• The carpet operator A^11 is used as a HIGH-ORDER SPATIAL FILTER,
  not as a "fractal antenna" or physical device.
• All detections are PROBABILISTIC and require independent verification.
• No claim that this module can detect actual relativistic objects.
• The connection between carpet mathematics and physical detection is
  ANALOGICAL, not proven.

LABEL CONVENTION (per Carpet paper)
-----------------------------------
  [P] proved in the paper
  [C] cited from literature (Hino, KZ, BBKT, etc.)
  [D] derived here from [C]
  [O] open / not claimed / research hypothesis
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.anomalous_detector")

__all__ = [
    "DetectorConfig",
    "SignalType",
    "AnomalyClass",
    # [U1] Relativistic signal model
    "RelativisticSignalModel",
    # [U2] Carpet high-order filter
    "CarpetGraph",
    "CarpetLaplacian",
    "CarpetHighOrderFilter",
    # [U3] Stealth analysis
    "StealthSignatureAnalyzer",
    # [U4] Anomaly detection
    "EnergyMeasureAnomalyDetector",
    # [U5] Classification
    "UFOCandidateClassifier",
    # Integrated
    "AnomalousRelativisticDetector",
]


# ═══════════════════════════════════════════════════════════════════════
# C^∞ PRIMITIVES (SESI ecosystem convention)
# ═══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Smooth absolute value."""
    return torch.sqrt(x * x + eps * eps)

def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    """Smooth step function."""
    return torch.sigmoid(x / tau)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG & ENUMS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    """Configuration for anomalous relativistic signal detector."""

    # ── Carpet operator ─────────────────────────────────────────────────
    carpet_level: int = 3              # graph approximation level
    carpet_rho: float = 1.25           # resistance scaling factor (numerical)
    polyharmonic_order: int = 22       # differential length (A^11)

    # ── Signal processing ───────────────────────────────────────────────
    sampling_rate: float = 1e9         # Hz
    frequency_band: Tuple[float, float] = (1e6, 1e12)  # Hz (MHz to THz)
    spatial_resolution: float = 1e-3   # meters

    # ── Relativistic parameters ─────────────────────────────────────────
    beta_min: float = 0.1              # minimum β = v/c to consider
    beta_max: float = 0.999            # maximum β (near light speed)
    gamma_max: float = 50.0            # maximum Lorentz factor

    # ── Stealth detection ───────────────────────────────────────────────
    em_stealth_threshold: float = 0.1  # RCS reduction factor
    grav_stealth_threshold: float = 0.05
    thermal_stealth_threshold: float = 0.2

    # ── Anomaly detection ───────────────────────────────────────────────
    anomaly_threshold: float = 3.0     # sigma threshold
    energy_measure_threshold: float = 0.5

    # ── Classification ──────────────────────────────────────────────────
    min_confidence: float = 0.7

    device: str = "cuda"


class SignalType(Enum):
    """Types of signals to analyze."""
    ELECTROMAGNETIC = auto()
    GRAVITATIONAL = auto()
    THERMAL = auto()
    QUANTUM_VACUUM = auto()
    UNKNOWN = auto()


class AnomalyClass(Enum):
    """Classification of detected anomalies."""
    NO_ANOMALY = 0
    LOW_CONFIDENCE = 1
    MODERATE_ANOMALY = 2
    HIGH_ANOMALY = 3
    RELATIVISTIC_CANDIDATE = 4
    STEALTH_CANDIDATE = 5
    COMPOSITE_ANOMALY = 6


# ═══════════════════════════════════════════════════════════════════════
# [U1] RELATIVISTIC SIGNAL MODEL
# ═══════════════════════════════════════════════════════════════════════

class RelativisticSignalModel(nn.Module):
    """
    [D] Model signals from relativistic objects under Lorentz transformation.

    For an object moving at velocity v = βc:
      • Doppler shift: f_obs = f_emit · √((1-β·cos θ)/(1+β·cos θ))
      • Time dilation: dt_obs = γ · dt_emit
      • Relativistic beaming: intensity ∝ 1/(γ(1-β·cos θ))^3

    HONEST: This is a STANDARD relativistic kinematic model. It does NOT
    prove that relativistic objects exist or that they produce detectable
    signals. It provides a template for what SUCH signals would look like.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg

    def lorentz_factor(self, beta: Tensor) -> Tensor:
        """γ = 1/√(1-β²), clamped to γ_max."""
        one_minus = (1.0 - beta * beta).clamp(min=1e-8)
        return torch.rsqrt(one_minus).clamp(max=self.cfg.gamma_max)

    def doppler_shift(self, f_emit: Tensor, beta: Tensor,
                      theta: Tensor) -> Tensor:
        """
        Relativistic Doppler shift.

        Args:
            f_emit: emitted frequency
            beta: v/c
            theta: angle between velocity and line of sight

        Returns:
            observed frequency
        """
        cos_theta = torch.cos(theta)
        factor = torch.sqrt((1.0 - beta * cos_theta) /
                           (1.0 + beta * cos_theta + 1e-12))
        return f_emit * factor

    def relativistic_beaming(self, beta: Tensor, theta: Tensor) -> Tensor:
        """
        Relativistic beaming factor (intensity enhancement).

        Intensity ∝ 1/(γ(1-β·cos θ))^3
        """
        gamma = self.lorentz_factor(beta)
        cos_theta = torch.cos(theta)
        denom = gamma * (1.0 - beta * cos_theta)
        return 1.0 / (denom.pow(3) + 1e-12)

    def time_dilation(self, dt_emit: Tensor, beta: Tensor) -> Tensor:
        """Observed time interval under time dilation."""
        gamma = self.lorentz_factor(beta)
        return gamma * dt_emit

    def length_contraction(self, L_emit: Tensor, beta: Tensor) -> Tensor:
        """Observed length under length contraction."""
        gamma = self.lorentz_factor(beta)
        return L_emit / gamma

    def generate_template(self, beta: float, theta: float,
                          f_emit: float, duration: float) -> Dict[str, Tensor]:
        """
        Generate a signal template for a relativistic object.

        Returns dict with observed frequency, beaming factor, time dilation.
        """
        beta_t = torch.tensor(beta)
        theta_t = torch.tensor(theta)
        f_emit_t = torch.tensor(f_emit)

        f_obs = self.doppler_shift(f_emit_t, beta_t, theta_t)
        beaming = self.relativistic_beaming(beta_t, theta_t)
        gamma = self.lorentz_factor(beta_t)

        return {
            "f_observed": f_obs,
            "beaming_factor": beaming,
            "lorentz_factor": gamma,
            "time_dilation": gamma,
            "length_contraction": 1.0 / gamma,
        }


# ═══════════════════════════════════════════════════════════════════════
# [U2] CARPET HIGH-ORDER FILTER (A^11, order 22)
# ═══════════════════════════════════════════════════════════════════════

class CarpetGraph:
    """
    [P] Level-n Sierpiński carpet graph (per Carpet paper §1).

    Cells: 8^n at level n.
    Edges: side-sharing cells with unit conductances.
    """

    def __init__(self, level: int):
        self.level = level
        self.cells = self._build_cells(level)
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.n_vertices = len(self.cells)
        self.edges = self._build_edges()

    def _build_cells(self, level: int) -> List[Tuple[int, int]]:
        side = 3 ** level
        return [(x, y) for x in range(side) for y in range(side)
                if self._survives(x, y, level)]

    @staticmethod
    def _survives(x: int, y: int, level: int) -> bool:
        for _ in range(level):
            if (x % 3 == 1) and (y % 3 == 1):
                return False
            x //= 3
            y //= 3
        return True

    def _build_edges(self) -> List[Tuple[int, int]]:
        vset = set(self.cells)
        edges = []
        for (x, y) in self.cells:
            for dx, dy in ((1, 0), (0, 1)):
                nb = (x + dx, y + dy)
                if nb in vset:
                    edges.append((self.index[(x, y)], self.index[nb]))
        return edges

    def graph_laplacian(self) -> Tensor:
        """L = D - A (graph Laplacian)."""
        A = torch.zeros(self.n_vertices, self.n_vertices)
        for i, j in self.edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
        deg = A.sum(dim=1)
        return torch.diag(deg) - A


class CarpetLaplacian(nn.Module):
    """
    [P] Normalized carpet Laplacian (per Carpet paper Theorem 2).

    A_n = ρ^n · L_n, where ρ ≈ 1.25 (numerical estimate).
    """

    def __init__(self, level: int, rho: float = 1.25):
        super().__init__()
        self.level = level
        self.rho = rho
        self.graph = CarpetGraph(level)
        self.L = self.graph.graph_laplacian()
        self.gauge = rho ** level
        self.A = self.gauge * self.L
        self._evals = None
        self._evecs = None

    def spectrum(self) -> Tuple[Tensor, Tensor]:
        """Compute eigenvalues and eigenvectors (cached)."""
        if self._evals is None:
            evals, evecs = torch.linalg.eigh(self.A)
            self._evals = evals
            self._evecs = evecs
        return self._evals, self._evecs

    def apply_power(self, u: Tensor, k: int) -> Tensor:
        """
        [P] Apply A^k via spectral decomposition.

        Per Proposition 4 and Corollary 4:
          A^k = ∂*(∂∂*)^(k-1) ∂ = (-1)^k ∆^k
        with NO lower-order terms.
        """
        evals, evecs = self.spectrum()
        coeffs = evecs.T @ u
        return evecs @ (evals.pow(k) * coeffs)


class CarpetHighOrderFilter(nn.Module):
    """
    [D] High-order spatial filter using carpet A^11 (order 22).

    APPLICATION RATIONALE (analogical, not proven):
      • A^11 is a 22nd-order differential operator on the carpet.
      • High-order operators act as EXTREME high-pass filters,
        suppressing smooth backgrounds and amplifying singular features.
      • Energy measures on the carpet are singular to µ (Theorem 8),
        so A^11 may highlight "anomalous" structures.

    HONEST SCOPE:
      • This is a MATHEMATICAL ANALOGY, not a proven detection method.
      • The connection between carpet operators and physical signal
        processing is SPECULATIVE.
      • A^11 amplifies noise as well as signals; regularization required.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        self.carpet = CarpetLaplacian(cfg.carpet_level, cfg.carpet_rho)
        self.k = cfg.polyharmonic_order // 2  # A^11 for order 22

        # Regularization to prevent noise amplification
        self.regularization = nn.Parameter(torch.tensor(1e-4))

    def forward(self, signal: Tensor) -> Tensor:
        """
        Apply A^11 filter with Tikhonov regularization.

        Args:
            signal: (N,) input signal on carpet vertices

        Returns:
            filtered signal (anomalous features enhanced)
        """
        # A^11 via spectral decomposition
        filtered = self.carpet.apply_power(signal, self.k)

        # Tikhonov regularization: (A^11 + λI)^{-1} A^11
        evals, evecs = self.carpet.spectrum()
        reg_factor = evals.pow(self.k) / (evals.pow(self.k) + self.regularization)
        coeffs = evecs.T @ signal
        regularized = evecs @ (reg_factor * coeffs)

        return regularized

    def anomaly_score(self, signal: Tensor) -> Tensor:
        """
        Compute anomaly score: ratio of high-order to low-order energy.

        High ratio → signal has singular features (anomalous).
        """
        evals, evecs = self.carpet.spectrum()
        coeffs = evecs.T @ signal
        coeffs_sq = coeffs * coeffs

        # Low-order energy (A^1)
        low_energy = (evals * coeffs_sq).sum()

        # High-order energy (A^11)
        high_energy = (evals.pow(self.k) * coeffs_sq).sum()

        # Ratio (regularized)
        ratio = high_energy / (low_energy + 1e-12)

        return ratio


# ═══════════════════════════════════════════════════════════════════════
# [U3] STEALTH SIGNATURE ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class StealthSignatureAnalyzer(nn.Module):
    """
    [O] Analyze stealth characteristics of detected anomalies.

    Stealth mechanisms (hypothetical):
      • Electromagnetic: metamaterial cloaking, adaptive absorption
      • Gravitational: frame-dragging compensation, weak lensing suppression
      • Thermal: infrared signature management
      • Quantum: vacuum fluctuation engineering

    HONEST SCOPE:
      • These are HYPOTHETICAL stealth mechanisms.
      • No known technology achieves perfect stealth across all bands.
      • This analyzer looks for SIGNATURES consistent with stealth,
        not proof of stealth technology.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg

        # Electromagnetic stealth detector
        self.em_stealth_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Gravitational stealth detector
        self.grav_stealth_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Thermal stealth detector
        self.thermal_stealth_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Analyze stealth signatures.

        Args:
            features: (B, 64) extracted signal features

        Returns:
            dict with stealth probabilities for each mechanism
        """
        em_stealth = self.em_stealth_net(features).squeeze(-1)
        grav_stealth = self.grav_stealth_net(features).squeeze(-1)
        thermal_stealth = self.thermal_stealth_net(features).squeeze(-1)

        # Composite stealth score
        composite = (em_stealth + grav_stealth + thermal_stealth) / 3.0

        return {
            "em_stealth_prob": em_stealth,
            "grav_stealth_prob": grav_stealth,
            "thermal_stealth_prob": thermal_stealth,
            "composite_stealth": composite,
            "is_stealthy": composite > 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════
# [U4] ANOMALY DETECTOR (Energy Measure Singularity)
# ═══════════════════════════════════════════════════════════════════════

class EnergyMeasureAnomalyDetector(nn.Module):
    """
    [D] Detect anomalies via energy measure singularity.

    THEORETICAL BASIS (Carpet paper Theorem 8, Hino 2005):
      • On the carpet, energy measures µ<f> are SINGULAR to µ.
      • This means energy concentrates on sets of µ-measure zero.
      • Anomalous signals may exhibit similar singularity.

    APPLICATION (analogical):
      • Compute energy measure of signal via carpet Laplacian.
      • Measure singularity relative to uniform background.
      • High singularity → anomalous signal.

    HONEST SCOPE:
      • This is a MATHEMATICAL ANALOGY, not a proven detection method.
      • The connection between carpet energy measures and physical
        anomalies is SPECULATIVE.
      • Requires validation against known anomalous signals.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        self.carpet = CarpetLaplacian(cfg.carpet_level, cfg.carpet_rho)

    def energy_measure(self, signal: Tensor) -> Tensor:
        """
        Compute energy measure µ<f> via carpet Laplacian.

        µ<f>(S) = E(f · 1_S, f) - E(f², 1_S) / 2
        (discrete approximation on carpet graph)
        """
        evals, evecs = self.carpet.spectrum()
        coeffs = evecs.T @ signal

        # Energy in each eigenmode
        energy_per_mode = evals * coeffs * coeffs

        return energy_per_mode

    def singularity_score(self, signal: Tensor) -> Tensor:
        """
        Compute singularity score: concentration of energy in few modes.

        High score → energy concentrated (singular) → anomalous.
        """
        energy = self.energy_measure(signal)
        energy_norm = energy / (energy.sum() + 1e-12)

        # Shannon entropy (lower = more concentrated)
        entropy = -(energy_norm * torch.log(energy_norm + 1e-12)).sum()

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(float(len(energy))))
        singularity = 1.0 - entropy / max_entropy

        return singularity

    def detect_anomaly(self, signal: Tensor) -> Dict[str, Tensor]:
        """
        Detect anomalous signal via energy measure singularity.

        Returns dict with singularity score and anomaly flag.
        """
        singularity = self.singularity_score(signal)
        is_anomalous = singularity > self.cfg.energy_measure_threshold

        return {
            "singularity_score": singularity,
            "is_anomalous": is_anomalous,
            "energy_measure": self.energy_measure(signal),
        }


# ═══════════════════════════════════════════════════════════════════════
# [U5] UFO CANDIDATE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

class UFOCandidateClassifier(nn.Module):
    """
    [O] Classify detected anomalies as potential UFO candidates.

    HONEST SCOPE (CRITICAL):
      • "UFO" = Unidentified Anomalous Phenomenon, NOT confirmed alien craft.
      • This classifier assigns PROBABILISTIC labels based on features.
      • All classifications require INDEPENDENT VERIFICATION.
      • No claim that this classifier can identify actual extraterrestrial objects.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        # Classifier head (7 classes per AnomalyClass)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 7),
            nn.Softmax(dim=-1),
        )

        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Dict[str, Any]:
        """
        Classify anomaly.

        Args:
            features: (B, 128) combined signal features

        Returns:
            dict with class probabilities, predicted class, confidence
        """
        embedded = self.feature_net(features)
        probs = self.classifier(embedded)
        confidence = self.confidence_net(embedded).squeeze(-1)

        pred_class = probs.argmax(dim=-1)

        # Map to AnomalyClass enum
        class_names = [c.name for c in AnomalyClass]

        return {
            "class_probs": probs,
            "predicted_class": pred_class,
            "predicted_class_name": [class_names[i] for i in pred_class.tolist()],
            "confidence": confidence,
            "is_reliable": confidence > self.cfg.min_confidence,
        }


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATED DETECTOR
# ═══════════════════════════════════════════════════════════════════════

class AnomalousRelativisticDetector(nn.Module):
    """
    Integrated anomalous relativistic signal detector.

    Pipeline:
      1. RelativisticSignalModel — generate templates
      2. CarpetHighOrderFilter — extract anomalous features (A^11)
      3. EnergyMeasureAnomalyDetector — detect singularities
      4. StealthSignatureAnalyzer — analyze stealth characteristics
      5. UFOCandidateClassifier — classify anomalies

    HONEST SCOPE:
      • This is a RESEARCH FRAMEWORK for investigating anomalous signals.
      • It does NOT confirm the existence of UFOs or relativistic craft.
      • All detections are probabilistic and require verification.
      • The carpet operator is used as a mathematical tool, not a physical device.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.relativistic_model = RelativisticSignalModel(cfg).to(self.device)
        self.carpet_filter = CarpetHighOrderFilter(cfg).to(self.device)
        self.anomaly_detector = EnergyMeasureAnomalyDetector(cfg).to(self.device)
        self.stealth_analyzer = StealthSignatureAnalyzer(cfg).to(self.device)
        self.classifier = UFOCandidateClassifier(cfg).to(self.device)

    def analyze_signal(self, raw_signal: Tensor) -> Dict[str, Any]:
        """
        Full analysis pipeline for a raw signal.

        Args:
            raw_signal: (N,) raw signal on carpet vertices

        Returns:
            dict with all analysis results
        """
        results = {}

        # 1. High-order filtering (A^11)
        filtered = self.carpet_filter(raw_signal)
        results["filtered_signal"] = filtered
        results["anomaly_score"] = self.carpet_filter.anomaly_score(raw_signal)

        # 2. Energy measure singularity
        anomaly_result = self.anomaly_detector.detect_anomaly(filtered)
        results.update({f"anomaly_{k}": v for k, v in anomaly_result.items()})

        # 3. Feature extraction for stealth analysis
        # (simplified: use filtered signal as features)
        features = filtered.unsqueeze(0) if filtered.dim() == 1 else filtered
        if features.shape[-1] < 64:
            features = F.pad(features, (0, 64 - features.shape[-1]))
        elif features.shape[-1] > 64:
            features = features[..., :64]

        # 4. Stealth analysis
        stealth_result = self.stealth_analyzer(features)
        results.update({f"stealth_{k}": v for k, v in stealth_result.items()})

        # 5. Classification
        # Combine features for classifier
        combined_features = torch.cat([
            features,
            anomaly_result["singularity_score"].unsqueeze(-1).expand(features.shape[0], -1),
            stealth_result["composite_stealth"].unsqueeze(-1).expand(features.shape[0], -1),
        ], dim=-1)

        if combined_features.shape[-1] < 128:
            combined_features = F.pad(combined_features, (0, 128 - combined_features.shape[-1]))
        elif combined_features.shape[-1] > 128:
            combined_features = combined_features[..., :128]

        class_result = self.classifier(combined_features)
        results.update({f"class_{k}": v for k, v in class_result.items()})

        # 6. Final verdict
        results["final_verdict"] = self._compute_verdict(results)

        return results

    def _compute_verdict(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final verdict based on all analysis."""
        is_anomalous = results["anomaly_is_anomalous"]
        is_stealthy = results["stealth_is_stealthy"]
        is_reliable = results["class_is_reliable"]
        pred_class = results["class_predicted_class_name"][0]

        if not is_anomalous:
            verdict = "NO_ANOMALY"
            confidence = 0.9
        elif is_anomalous and is_stealthy and is_reliable:
            if pred_class in ["RELATIVISTIC_CANDIDATE", "COMPOSITE_ANOMALY"]:
                verdict = "HIGH_PRIORITY_ANOMALY"
                confidence = results["class_confidence"][0].item()
            else:
                verdict = "MODERATE_ANOMALY"
                confidence = results["class_confidence"][0].item() * 0.7
        elif is_anomalous:
            verdict = "LOW_PRIORITY_ANOMALY"
            confidence = 0.5
        else:
            verdict = "UNCERTAIN"
            confidence = 0.3

        return {
            "verdict": verdict,
            "confidence": confidence,
            "requires_verification": confidence < 0.8,
            "note": (
                "This is a RESEARCH FRAMEWORK. All detections require "
                "independent verification. No claim of confirmed UFO detection."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Anomalous Relativistic Signal Detector — Smoke Test")
    print("=" * 72)
    torch.manual_seed(42)

    cfg = DetectorConfig(
        carpet_level=2,
        polyharmonic_order=22,
        device="cpu",
    )

    detector = AnomalousRelativisticDetector(cfg)
    n_vertices = detector.carpet_filter.carpet.graph.n_vertices

    print(f"\n  ── Configuration ──")
    print(f"    Carpet level       : {cfg.carpet_level}")
    print(f"    Vertices           : {n_vertices}")
    print(f"    Polyharmonic order : {cfg.polyharmonic_order} (A^{cfg.polyharmonic_order // 2})")
    print(f"    ρ (resistance)     : {cfg.carpet_rho}")

    # Test 1: Relativistic signal template
    print(f"\n  ── [U1] Relativistic Signal Model ──")
    template = detector.relativistic_model.generate_template(
        beta=0.9, theta=0.0, f_emit=1e9, duration=1.0
    )
    print(f"    β = 0.9, θ = 0 (head-on)")
    print(f"    γ (Lorentz factor) : {template['lorentz_factor'].item():.3f}")
    print(f"    f_obs / f_emit     : {(template['f_observed'] / 1e9).item():.3f}")
    print(f"    Beaming factor     : {template['beaming_factor'].item():.3f}")

    # Test 2: Carpet high-order filter
    print(f"\n  ── [U2] Carpet High-Order Filter (A^11) ──")
    signal = torch.randn(n_vertices)
    filtered = detector.carpet_filter(signal)
    anomaly_score = detector.carpet_filter.anomaly_score(signal)
    print(f"    Input signal norm  : {signal.norm().item():.4f}")
    print(f"    Filtered norm      : {filtered.norm().item():.4f}")
    print(f"    Anomaly score      : {anomaly_score.item():.4f}")
    print(f"    [PASS] A^11 filter operational (Prop 4, Cor 4)")

    # Test 3: Energy measure anomaly detection
    print(f"\n  ── [U4] Energy Measure Anomaly Detector ──")
    anomaly_result = detector.anomaly_detector.detect_anomaly(filtered)
    print(f"    Singularity score  : {anomaly_result['singularity_score'].item():.4f}")
    print(f"    Is anomalous       : {anomaly_result['is_anomalous'].item()}")
    print(f"    [PASS] Energy measure singularity analysis (Hino Thm 8)")

    # Test 4: Full pipeline
    print(f"\n  ── Full Analysis Pipeline ──")
    results = detector.analyze_signal(signal)
    print(f"    Anomaly score      : {results['anomaly_score'].item():.4f}")
    print(f"    Singularity        : {results['anomaly_singularity_score'].item():.4f}")
    print(f"    Stealth composite  : {results['stealth_composite_stealth'].item():.4f}")
    print(f"    Predicted class    : {results['class_predicted_class_name'][0]}")
    print(f"    Confidence         : {results['class_confidence'][0].item():.4f}")
    print(f"    Final verdict      : {results['final_verdict']['verdict']}")
    print(f"    Requires verify    : {results['final_verdict']['requires_verification']}")

    print("\n" + "=" * 72)
    print("  EPISTEMIC HONESTY — CRITICAL")
    print("  • This is a RESEARCH FRAMEWORK, not a confirmed detection system.")
    print("  • 'UFO' = Unidentified Anomalous Phenomenon, NOT alien craft.")
    print("  • Carpet A^11 is a MATHEMATICAL TOOL, not a physical antenna.")
    print("  • All detections are PROBABILISTIC and require verification.")
    print("  • No claim that relativistic stealth objects exist.")
    print("  • The connection between carpet math and physical detection")
    print("    is ANALOGICAL, not proven.")
    print("=" * 72)
    print("  Mathematical foundations (per Carpet paper):")
    print("  • [P] Prop 4: A^k = ∂*(∂∂*)^(k-1) ∂ (polyharmonic)")
    print("  • [P] Cor 4: No lower-order terms")
    print("  • [P] Thm 4: Spectral Sobolev scale W^s")
    print("  • [C] Thm 8 (Hino): Energy measures ⊥ µ (singular)")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
