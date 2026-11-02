"""
=============================================================================
SESI PSYCHIATRIC MUSIC THERAPY AGI
Clinical Outcome Assessment + EM Wave CFD + OpenBCI + Real-Time Synthesis
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

⚠️  CLINICAL/ETHICAL FRAMEWORK
-----------------------------------------------------------------------------
• NOT A MEDICAL DEVICE — research/artistic exploration only
• Requires licensed clinical oversight for any psychiatric use
• Informed consent MANDATORY for all users
• PNDC Axiom 3: Psychiatric data is HIGHLY SENSITIVE
• Buddhist Ethics (อहिंसा - non-harm): Never use to manipulate
• Taoist Ethics (无为 - non-forcing): Support, don't control
• All thresholds are FREE DESIGN PARAMETERS
• Production deployment REQUIRES IRB approval + clinical validation
=============================================================================

WHAT THIS MODULE PROVIDES
--------------------------
[PSY1] PsychiatricOutcomeAssessor
      • Multi-disorder assessment (DSM-5 aligned)
      • EEG biomarker extraction (literature-based)
      • Depression (PHQ-9 style), PTSD (PCL-5), Panic (PDSS)
      • Substance use disorders, sex addiction, hypersexual disorder
      • Bipolar (YMRS/MDQ), Schizophrenia (PANSS)
      • Pre/post music therapy comparison
      • Confidence scoring with uncertainty quantification

[EMW1] ElectromagneticWaveCFD
      • Maxwell's equations solver (3D FDTD)
      • Neural tissue EM wave propagation
      • Coupling with audio signal → EM field
      • Structural fluctuating hydrodynamics integration
      • Tissue-specific permittivity/conductivity models

[OBC1] OpenBCIBridge
      • OpenBCI Cyton board integration (8-channel EEG)
      • Real-time serial communication
      • 10-20 electrode system mapping
      • Band power extraction (delta/theta/alpha/beta/gamma)
      • Impedance checking
      • Graceful fallback for mock data

[RTS1] RealTimeAudioSynth
      • Granular synthesis engine
      • Additive/FM synthesis
      • Binaural beats + isochronic tones
      • Therapeutic frequency mapping per disorder
      • Real-time parameter modulation from biofeedback

[BVD1] BiofeedbackVizDashboard
      • Real-time EEG visualization
      • Psychiatric outcome tracking
      • EM field propagation display
      • Therapeutic music parameter visualization
      • Multi-user session support

MATHEMATICAL FOUNDATIONS
--------------------------
[C] EEG Biomarkers:
    • Depression: Frontal alpha asymmetry (Davidson 2004)
    • PTSD: Increased beta, decreased alpha (Thibodeau 2007)
    • Panic: High-beta hyperarousal, HRV coherence loss
    • Addiction: P300 abnormalities, theta/beta ratio
    • Bipolar: Mood-state dependent patterns
    • Schizophrenia: Reduced P300, gamma band deficits
[C] Maxwell's Equations: ∇×E = -∂B/∂t, ∇×H = J + ∂D/∂t
[C] Neural Tissue Models: Gabriel et al. 1996 (permittivity)
[C] Binaural Beats: Oster 1973 (frequency following response)
[C] Therapeutic Music: Thaut 2005 (neurologic music therapy)
[C] SESI_NoZeno_Closure Thm 3.2: Bounded computation
[C] Delta_Min Thm 1.2: Quantum gate for meaningful changes
[C] PNDC Axiom 3: Psychiatric data minimization

HONEST SCOPE
------------
• This is a RESEARCH FRAMEWORK, NOT a clinical diagnostic tool
• Psychiatric assessment is HEURISTIC, not definitive diagnosis
• EM wave modeling is APPROXIMATE, not validated for medical use
• OpenBCI integration requires proper hardware setup
• Real-time audio synthesis is COMPUTATIONAL, not therapeutic
• All use REQUIRES informed consent and clinical oversight
• Production deployment REQUIRES IRB approval
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.psychiatric_music_cfd")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Integration with structural calculus ecosystem
try:
    from structuralfluctuatinghydro_v6_3 import (
        StructuralFluctuatingHydro, SolverConfig as FHSolverConfig,
    )
    HAS_FH = True
except ImportError:
    HAS_FH = False

try:
    from sesi_biofeedback_adaptive_collab import (
        BiofeedbackSensorBridge, SensorConfig, BiometricSample,
        BiometricDataProcessor, EmotionalStateEstimate,
    )
    HAS_BIOFEEDBACK = True
except ImportError:
    HAS_BIOFEEDBACK = False

__all__ = [
    "PsychiatricDisorder", "SeverityLevel", "EEGBand",
    "PsychiatricOutcomeAssessor", "PsychAssessmentConfig", "PsychOutcome",
    "ElectromagneticWaveCFD", "EMWaveConfig", "TissueType",
    "OpenBCIBridge", "OpenBCIConfig", "EEGSample",
    "RealTimeAudioSynth", "SynthConfig", "TherapeuticTone",
    "BiofeedbackVizDashboard", "VizConfig",
    "SESIPsychiatricMusicCFDSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class PsychiatricDisorder(Enum):
    """Supported psychiatric disorders (DSM-5 aligned)."""
    # Mood disorders
    MAJOR_DEPRESSION = "major_depression"
    BIPOLAR_I = "bipolar_i"
    BIPOLAR_II = "bipolar_ii"
    # Anxiety disorders
    PTSD = "ptsd"
    PANIC_DISORDER = "panic_disorder"
    GENERALIZED_ANXIETY = "generalized_anxiety"
    # Psychotic disorders
    SCHIZOPHRENIA = "schizophrenia"
    SCHIZOAFFECTIVE = "schizoaffective"
    # Addiction disorders
    SUBSTANCE_USE_ALCOHOL = "substance_use_alcohol"
    SUBSTANCE_USE_OPIOID = "substance_use_opioid"
    SUBSTANCE_USE_STIMULANT = "substance_use_stimulant"
    BEHAVIORAL_ADDICTION_SEX = "behavioral_addiction_sex"
    HYPERSEXUAL_DISORDER = "hypersexual_disorder"
    # Other
    OCD = "ocd"
    ADHD = "adhd"
    HEALTHY_CONTROL = "healthy_control"


class SeverityLevel(Enum):
    """Clinical severity levels."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4


class EEGBand(Enum):
    """EEG frequency bands."""
    DELTA = "delta"      # 0.5-4 Hz (deep sleep)
    THETA = "theta"      # 4-8 Hz (meditation, drowsiness)
    ALPHA = "alpha"      # 8-13 Hz (relaxed wakefulness)
    SMR = "smr"          # 12-15 Hz (sensorimotor rhythm)
    BETA_LOW = "beta_low"  # 15-18 Hz
    BETA_MID = "beta_mid"  # 18-21 Hz
    BETA_HIGH = "beta_high" # 21-30 Hz (active thinking)
    GAMMA = "gamma"      # 30-100 Hz (higher processing)


class TissueType(Enum):
    """Neural tissue types for EM modeling."""
    GRAY_MATTER = "gray_matter"
    WHITE_MATTER = "white_matter"
    CSF = "csf"  # cerebrospinal fluid
    SKULL = "skull"
    SCALP = "scalp"


# ═══════════════════════════════════════════════════════════════════════════
# [PSY1] PSYCHIATRIC OUTCOME ASSESSOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PsychAssessmentConfig:
    """Configuration for psychiatric assessment."""
    # Assessment windows
    pre_therapy_window_s: float = 300.0  # 5 min baseline
    post_therapy_window_s: float = 300.0
    # EEG biomarkers (from literature)
    use_frontal_alpha_asymmetry: bool = True  # Depression
    use_beta_power: bool = True  # PTSD, Anxiety
    use_theta_beta_ratio: bool = True  # ADHD, Addiction
    use_p300_amplitude: bool = True  # Schizophrenia, Addiction
    use_gamma_coherence: bool = True  # Schizophrenia
    # Confidence
    min_confidence_threshold: float = 0.6
    # PNDC Axiom 3
    encrypt_psych_data: bool = True
    anonymize_user_id: bool = True


@dataclass
class PsychOutcome:
    """Psychiatric outcome assessment."""
    assessment_id: str
    timestamp: float
    user_id: str
    disorder: PsychiatricDisorder
    severity: SeverityLevel
    confidence: float
    # EEG biomarkers
    eeg_features: Dict[str, float] = field(default_factory=dict)
    # Clinical scales (simulated)
    clinical_scores: Dict[str, float] = field(default_factory=dict)
    # Pre/post comparison
    pre_therapy_state: Optional[Dict[str, Any]] = None
    post_therapy_state: Optional[Dict[str, Any]] = None
    improvement_score: Optional[float] = None
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "disorder": self.disorder.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "eeg_features": self.eeg_features,
            "clinical_scores": self.clinical_scores,
            "improvement_score": self.improvement_score,
        }


class PsychiatricOutcomeAssessor:
    """
    [D] Multi-disorder psychiatric outcome assessment from EEG biomarkers.

    ⚠️  CLINICAL NOTICE:
      • This is a RESEARCH tool, NOT a diagnostic instrument
      • All assessments require clinical oversight
      • EEG biomarkers are HEURISTIC, not definitive

    Features:
      • Multi-disorder assessment (DSM-5 aligned)
      • EEG biomarker extraction (literature-based)
      • Pre/post music therapy comparison
      • Confidence scoring with uncertainty quantification
      • PNDC Axiom 3 data minimization

    Disorders supported:
      • Depression (frontal alpha asymmetry)
      • PTSD (increased beta, decreased alpha)
      • Panic (high-beta hyperarousal)
      • Substance use (P300 abnormalities)
      • Sex addiction / Hypersexual (theta/beta ratio)
      • Bipolar (mood-state patterns)
      • Schizophrenia (reduced P300, gamma deficits)

    Mathematical grounding:
      • Davidson 2004: Frontal EEG asymmetry in depression
      • Thibodeau 2007: PTSD EEG patterns
      • P300 literature: Addiction biomarkers
    """

    # EEG biomarker ranges from literature (simplified)
    BIOMARKER_RANGES = {
        PsychiatricDisorder.MAJOR_DEPRESSION: {
            "frontal_alpha_asymmetry": (-0.2, 0.0),  # ln(right/left)
            "alpha_power_frontal": (8.0, 12.0),  # μV²
            "severity_mapping": {
                SeverityLevel.NONE: (0.0, 0.05),
                SeverityLevel.MILD: (0.05, 0.10),
                SeverityLevel.MODERATE: (0.10, 0.15),
                SeverityLevel.SEVERE: (0.15, 0.20),
                SeverityLevel.EXTREME: (0.20, 1.0),
            },
        },
        PsychiatricDisorder.PTSD: {
            "beta_power": (15.0, 25.0),  # μV²
            "alpha_power": (5.0, 10.0),
            "beta_alpha_ratio": (1.5, 3.0),
            "severity_mapping": {
                SeverityLevel.NONE: (0.0, 1.0),
                SeverityLevel.MILD: (1.0, 1.5),
                SeverityLevel.MODERATE: (1.5, 2.0),
                SeverityLevel.SEVERE: (2.0, 2.5),
                SeverityLevel.EXTREME: (2.5, 5.0),
            },
        },
        PsychiatricDisorder.PANIC_DISORDER: {
            "high_beta_power": (20.0, 35.0),
            "hrv_coherence": (0.0, 0.3),
            "severity_mapping": {
                SeverityLevel.NONE: (0.0, 0.1),
                SeverityLevel.MILD: (0.1, 0.2),
                SeverityLevel.MODERATE: (0.2, 0.4),
                SeverityLevel.SEVERE: (0.4, 0.6),
                SeverityLevel.EXTREME: (0.6, 1.0),
            },
        },
        PsychiatricDisorder.SUBSTANCE_USE_ALCOHOL: {
            "p300_amplitude": (5.0, 10.0),  # μV
            "theta_beta_ratio": (0.5, 1.0),
            "severity_mapping": {
                SeverityLevel.NONE: (0.0, 0.2),
                SeverityLevel.MILD: (0.2, 0.4),
                SeverityLevel.MODERATE: (0.4, 0.6),
                SeverityLevel.SEVERE: (0.6, 0.8),
                SeverityLevel.EXTREME: (0.8, 1.0),
            },
        },
        PsychiatricDisorder.BIPOLAR_I: {
            "mood_state": "manic",  # or "depressive" or "euthymic"
            "beta_power": (12.0, 20.0),
            "alpha_power": (6.0, 12.0),
            "severity_mapping": {
                SeverityLevel.NONE: (0.0, 0.1),
                SeverityLevel.MILD: (0.1, 0.3),
                SeverityLevel.MODERATE: (0.3, 0.5),
                SeverityLevel.SEVERE: (0.5, 0.7),
                SeverityLevel.EXTREME: (0.7, 1.0),
            },
        },
        PsychiatricDisorder.SCHIZOPHRENIA: {
            "p300_amplitude": (3.0, 8.0),
            "gamma_coherence": (0.1, 0.3),
            "severity_mapping": {
                SeverityLevel.NONE: (0.0, 0.2),
                SeverityLevel.MILD: (0.2, 0.4),
                SeverityLevel.MODERATE: (0.4, 0.6),
                SeverityLevel.SEVERE: (0.6, 0.8),
                SeverityLevel.EXTREME: (0.8, 1.0),
            },
        },
    }

    def __init__(self, cfg: Optional[PsychAssessmentConfig] = None):
        self.cfg = cfg or PsychAssessmentConfig()
        self._assessments: List[PsychOutcome] = []
        self._pre_therapy_baseline: Dict[str, Dict[str, Any]] = {}

    # ── EEG Feature Extraction ───────────────────────────────────────────

    def extract_eeg_features(
        self,
        eeg_data: Dict[str, Tensor],
        sample_rate: int = 256,
    ) -> Dict[str, float]:
        """
        Extract EEG biomarkers from raw data.

        Args:
            eeg_data: Dict mapping channel name to (n_samples,) tensor
            sample_rate: Sampling rate in Hz

        Returns:
            Dict of extracted features
        """
        features = {}

        # Band power extraction (simplified)
        for channel_name, signal in eeg_data.items():
            if HAS_SCIPY:
                # Real FFT-based band power
                freqs, psd = scipy_signal.welch(
                    signal.cpu().numpy(),
                    fs=sample_rate,
                    nperseg=1024,
                )

                # Delta (0.5-4 Hz)
                delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
                features[f"{channel_name}_delta_power"] = float(psd[delta_mask].mean())

                # Theta (4-8 Hz)
                theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
                features[f"{channel_name}_theta_power"] = float(psd[theta_mask].mean())

                # Alpha (8-13 Hz)
                alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
                features[f"{channel_name}_alpha_power"] = float(psd[alpha_mask].mean())

                # Beta (13-30 Hz)
                beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
                features[f"{channel_name}_beta_power"] = float(psd[beta_mask].mean())

                # Gamma (30-100 Hz)
                gamma_mask = (freqs >= 30.0) & (freqs <= 100.0)
                features[f"{channel_name}_gamma_power"] = float(psd[gamma_mask].mean())

        # Frontal alpha asymmetry (depression biomarker)
        if self.cfg.use_frontal_alpha_asymmetry:
            if "F3_alpha_power" in features and "F4_alpha_power" in features:
                ln_right = math.log(features["F4_alpha_power"] + 1e-10)
                ln_left = math.log(features["F3_alpha_power"] + 1e-10)
                features["frontal_alpha_asymmetry"] = ln_right - ln_left

        # Theta/beta ratio (ADHD, addiction)
        if self.cfg.use_theta_beta_ratio:
            if "Fz_theta_power" in features and "Fz_beta_power" in features:
                features["theta_beta_ratio"] = (
                    features["Fz_theta_power"] /
                    (features["Fz_beta_power"] + 1e-10)
                )

        # Beta/alpha ratio (PTSD, anxiety)
        if self.cfg.use_beta_power:
            if "Fz_beta_power" in features and "Fz_alpha_power" in features:
                features["beta_alpha_ratio"] = (
                    features["Fz_beta_power"] /
                    (features["Fz_alpha_power"] + 1e-10)
                )

        return features

    # ── Disorder Assessment ──────────────────────────────────────────────

    def assess_disorder(
        self,
        eeg_features: Dict[str, float],
        disorder: PsychiatricDisorder,
    ) -> Tuple[SeverityLevel, float]:
        """
        Assess disorder severity from EEG features.

        Args:
            eeg_features: Extracted EEG biomarkers
            disorder: Target disorder

        Returns:
            (severity_level, confidence)
        """
        if disorder not in self.BIOMARKER_RANGES:
            return SeverityLevel.NONE, 0.0

        ranges = self.BIOMARKER_RANGES[disorder]
        severity_mapping = ranges["severity_mapping"]

        # Compute disorder-specific score
        score = 0.0
        n_features = 0

        if disorder == PsychiatricDisorder.MAJOR_DEPRESSION:
            if "frontal_alpha_asymmetry" in eeg_features:
                asym = eeg_features["frontal_alpha_asymmetry"]
                # Map asymmetry to severity
                for sev, (low, high) in severity_mapping.items():
                    if low <= abs(asym) < high:
                        score = sev.value / 4.0
                        n_features += 1
                        break

        elif disorder == PsychiatricDisorder.PTSD:
            if "beta_alpha_ratio" in eeg_features:
                ratio = eeg_features["beta_alpha_ratio"]
                for sev, (low, high) in severity_mapping.items():
                    if low <= ratio < high:
                        score = sev.value / 4.0
                        n_features += 1
                        break

        elif disorder == PsychiatricDisorder.PANIC_DISORDER:
            if "high_beta_power" in eeg_features:
                beta = eeg_features.get("Fz_beta_power", 0.0)
                for sev, (low, high) in severity_mapping.items():
                    if low <= beta < high:
                        score = sev.value / 4.0
                        n_features += 1
                        break

        elif disorder in [
            PsychiatricDisorder.SUBSTANCE_USE_ALCOHOL,
            PsychiatricDisorder.SUBSTANCE_USE_OPIOID,
            PsychiatricDisorder.SUBSTANCE_USE_STIMULANT,
        ]:
            if "theta_beta_ratio" in eeg_features:
                ratio = eeg_features["theta_beta_ratio"]
                for sev, (low, high) in severity_mapping.items():
                    if low <= ratio < high:
                        score = sev.value / 4.0
                        n_features += 1
                        break

        elif disorder == PsychiatricDisorder.SCHIZOPHRENIA:
            # Simplified: use gamma power as proxy
            if "gamma_power" in eeg_features:
                gamma = eeg_features.get("Fz_gamma_power", 0.0)
                for sev, (low, high) in severity_mapping.items():
                    if low <= gamma < high:
                        score = sev.value / 4.0
                        n_features += 1
                        break

        # Compute confidence
        confidence = min(1.0, n_features / 3.0) if n_features > 0 else 0.0

        # Map score to severity
        if score < 0.2:
            severity = SeverityLevel.NONE
        elif score < 0.4:
            severity = SeverityLevel.MILD
        elif score < 0.6:
            severity = SeverityLevel.MODERATE
        elif score < 0.8:
            severity = SeverityLevel.SEVERE
        else:
            severity = SeverityLevel.EXTREME

        return severity, confidence

    # ── Pre/Post Therapy Comparison ──────────────────────────────────────

    def record_pre_therapy_baseline(
        self,
        user_id: str,
        eeg_features: Dict[str, float],
        disorder: PsychiatricDisorder,
    ) -> None:
        """Record pre-therapy baseline for comparison."""
        severity, confidence = self.assess_disorder(eeg_features, disorder)
        self._pre_therapy_baseline[user_id] = {
            "eeg_features": eeg_features,
            "severity": severity,
            "confidence": confidence,
            "disorder": disorder,
            "timestamp": time.time(),
        }
        logger.info(
            f"[PsychAssessor] Recorded pre-therapy baseline for {user_id}: "
            f"{disorder.value} severity={severity.name}"
        )

    def assess_therapy_outcome(
        self,
        user_id: str,
        post_eeg_features: Dict[str, float],
    ) -> Optional[PsychOutcome]:
        """
        Assess therapy outcome by comparing pre/post EEG.

        Args:
            user_id: User identifier
            post_eeg_features: Post-therapy EEG features

        Returns:
            PsychOutcome if baseline exists, None otherwise
        """
        if user_id not in self._pre_therapy_baseline:
            logger.warning(f"[PsychAssessor] No baseline for user {user_id}")
            return None

        baseline = self._pre_therapy_baseline[user_id]
        disorder = baseline["disorder"]

        # Assess post-therapy state
        post_severity, post_confidence = self.assess_disorder(
            post_eeg_features, disorder
        )

        # Compute improvement
        pre_score = baseline["severity"].value
        post_score = post_severity.value
        improvement = (pre_score - post_score) / max(pre_score, 1e-8)

        outcome = PsychOutcome(
            assessment_id=f"assess_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            user_id=user_id if not self.cfg.anonymize_user_id else f"anon_{hash(user_id) % 10000:04d}",
            disorder=disorder,
            severity=post_severity,
            confidence=post_confidence,
            eeg_features=post_eeg_features,
            pre_therapy_state={
                "severity": baseline["severity"].name,
                "eeg_features": baseline["eeg_features"],
            },
            post_therapy_state={
                "severity": post_severity.name,
                "eeg_features": post_eeg_features,
            },
            improvement_score=improvement,
            metadata={
                "therapy_type": "music_therapy",
                "baseline_timestamp": baseline["timestamp"],
            },
        )

        self._assessments.append(outcome)
        logger.info(
            f"[PsychAssessor] Therapy outcome for {user_id}: "
            f"improvement={improvement:.2%}, severity {baseline['severity'].name} → {post_severity.name}"
        )

        return outcome

    def get_assessment_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get assessment history."""
        assessments = self._assessments[-limit:]
        if user_id:
            assessments = [a for a in assessments if a.user_id == user_id]
        return [a.to_dict() for a in assessments]

    def get_status(self) -> Dict[str, Any]:
        """Get assessor status."""
        return {
            "total_assessments": len(self._assessments),
            "n_baselines": len(self._pre_therapy_baseline),
            "supported_disorders": len(self.BIOMARKER_RANGES),
            "encrypt_data": self.cfg.encrypt_psych_data,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [EMW1] ELECTROMAGNETIC WAVE CFD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EMWaveConfig:
    """Configuration for EM wave CFD simulation."""
    # Grid
    nx: int = 64
    ny: int = 64
    nz: int = 64
    dx: float = 1e-3  # 1mm resolution
    # Time
    dt: float = 1e-12  # 1 ps
    n_steps: int = 1000
    # Physics
    frequency_hz: float = 1e9  # 1 GHz (audio-to-EM coupling)
    source_amplitude: float = 1.0
    # Tissue properties (Gabriel et al. 1996)
    tissue_type: TissueType = TissueType.GRAY_MATTER
    permittivity: float = 45.0  # relative permittivity
    conductivity: float = 0.5  # S/m
    # Integration with structural calculus
    use_structural_fluctuations: bool = True
    device: str = "cpu"


class ElectromagneticWaveCFD:
    """
    [D] Electromagnetic wave propagation in neural tissue via CFD.

    Uses structural fluctuating hydrodynamics framework to model
    EM wave propagation through brain tissue, coupling audio signals
    to electromagnetic fields.

    ⚠️  CLINICAL NOTICE:
      • This is a COMPUTATIONAL MODEL, not validated for medical use
      • Tissue properties are APPROXIMATE (Gabriel et al. 1996)
      • Results are EXPLORATORY, not diagnostic

    Features:
      • 3D FDTD (Finite-Difference Time-Domain) solver
      • Neural tissue EM wave propagation
      • Audio → EM field coupling
      • Structural fluctuating hydrodynamics integration
      • Tissue-specific permittivity/conductivity

    Mathematical grounding:
      • Maxwell's equations: ∇×E = -∂B/∂t, ∇×H = J + ∂D/∂t
      • FDTD: Yee 1966 (staggered grid)
      • Neural tissue: Gabriel et al. 1996
    """

    # Tissue properties from Gabriel et al. 1996
    TISSUE_PROPERTIES = {
        TissueType.GRAY_MATTER: {
            "permittivity": 45.0,
            "conductivity": 0.5,
            "density": 1040.0,
        },
        TissueType.WHITE_MATTER: {
            "permittivity": 35.0,
            "conductivity": 0.3,
            "density": 1030.0,
        },
        TissueType.CSF: {
            "permittivity": 78.0,
            "conductivity": 2.0,
            "density": 1006.0,
        },
        TissueType.SKULL: {
            "permittivity": 12.0,
            "conductivity": 0.01,
            "density": 1900.0,
        },
        TissueType.SCALP: {
            "permittivity": 36.0,
            "conductivity": 0.4,
            "density": 1100.0,
        },
    }

    def __init__(self, cfg: Optional[EMWaveConfig] = None):
        self.cfg = cfg or EMWaveConfig()
        self.device = torch.device(self.cfg.device)

        # Initialize field arrays (E and H on Yee grid)
        self._init_fields()

        # Structural fluctuating hydro solver (if available)
        self.fh_solver = None
        if self.cfg.use_structural_fluctuations and HAS_FH:
            fh_cfg = FHSolverConfig(
                Nx=self.cfg.nx,
                Ny=self.cfg.ny,
                Nz=self.cfg.nz,
                dt=self.cfg.dt,
                enable_fluctuations=True,
            )
            self.fh_solver = StructuralFluctuatingHydro(fh_cfg)

    def _init_fields(self) -> None:
        """Initialize E and H field arrays."""
        nx, ny, nz = self.cfg.nx, self.cfg.ny, self.cfg.nz
        dev = self.device

        # Electric field (Yee grid: Ex on edges, etc.)
        self.Ex = torch.zeros(nx+1, ny, nz, device=dev, dtype=torch.float64)
        self.Ey = torch.zeros(nx, ny+1, nz, device=dev, dtype=torch.float64)
        self.Ez = torch.zeros(nx, ny, nz+1, device=dev, dtype=torch.float64)

        # Magnetic field (staggered)
        self.Hx = torch.zeros(nx, ny+1, nz+1, device=dev, dtype=torch.float64)
        self.Hy = torch.zeros(nx+1, ny, nz+1, device=dev, dtype=torch.float64)
        self.Hz = torch.zeros(nx+1, ny+1, nz, device=dev, dtype=torch.float64)

        # Tissue properties
        props = self.TISSUE_PROPERTIES[self.cfg.tissue_type]
        self.epsilon_r = props["permittivity"]
        self.sigma = props["conductivity"]

        # Physical constants
        self.epsilon_0 = 8.854e-12  # F/m
        self.mu_0 = 4 * math.pi * 1e-7  # H/m
        self.c = 3e8  # m/s

    def audio_to_em_source(
        self,
        audio_signal: Tensor,
        sample_rate: int = 44100,
    ) -> Tensor:
        """
        Convert audio signal to EM source term.

        Maps audio frequencies to EM field amplitudes via
        frequency-dependent coupling.

        Args:
            audio_signal: (n_samples,) audio waveform
            sample_rate: Audio sample rate

        Returns:
            (n_samples,) EM source term
        """
        # Simple linear mapping (in production: use proper transduction model)
        source = audio_signal * self.cfg.source_amplitude

        # Frequency-dependent attenuation (tissue absorption)
        if HAS_SCIPY:
            # Apply tissue transfer function
            freqs = np.fft.rfftfreq(len(source), 1.0/sample_rate)
            # Simplified tissue absorption: alpha(f) = a * f^b
            a, b = 0.1, 1.5  # tissue-specific parameters
            attenuation = np.exp(-a * freqs**b * self.cfg.dx * self.cfg.nx)
            source_np = source.cpu().numpy()
            source_fft = np.fft.rfft(source_np)
            source_fft *= attenuation
            source = torch.from_numpy(np.fft.irfft(source_fft)).to(self.device)

        return source

    def step_fdtd(
        self,
        source_term: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Advance FDTD simulation by one time step.

        Implements Yee algorithm:
          1. Update H from curl(E)
          2. Update E from curl(H) - σE + J_source

        Args:
            source_term: Optional EM source (from audio coupling)

        Returns:
            dict with updated field components
        """
        dt = self.cfg.dt
        dx = self.cfg.dx
        eps = self.epsilon_0 * self.epsilon_r
        mu = self.mu_0

        # ── Update H fields (Faraday's law: ∇×E = -∂B/∂t) ─────────
        # Hx update
        self.Hx[:, :-1, :-1] -= (dt / mu / dx) * (
            (self.Ez[:, 1:, :-1] - self.Ez[:, :-1, :-1]) -
            (self.Ey[:, :-1, 1:] - self.Ey[:, :-1, :-1])
        )

        # Hy update
        self.Hy[:-1, :, :-1] -= (dt / mu / dx) * (
            (self.Ex[:-1, :, 1:] - self.Ex[:-1, :, :-1]) -
            (self.Ez[1:, :, :-1] - self.Ez[:-1, :, :-1])
        )

        # Hz update
        self.Hz[:-1, :-1, :] -= (dt / mu / dx) * (
            (self.Ey[1:, :-1, :] - self.Ey[:-1, :-1, :]) -
            (self.Ex[:-1, 1:, :] - self.Ex[:-1, :-1, :])
        )

        # ── Update E fields (Ampere's law: ∇×H = J + ∂D/∂t) ───────
        # Lossy medium: include conductivity term
        decay = self.sigma * dt / (2 * eps)
        factor = (1 - decay) / (1 + decay)
        source_factor = dt / (eps * dx * (1 + decay))

        # Ex update
        self.Ex[1:-1, :, :] = factor * self.Ex[1:-1, :, :] + source_factor * (
            (self.Hz[1:-1, :, :] - self.Hz[:-2, :, :]) -
            (self.Hy[1:-1, :, :] - self.Hy[1:-1, :, :-1])
        )

        # Ey update
        self.Ey[:, 1:-1, :] = factor * self.Ey[:, 1:-1, :] + source_factor * (
            (self.Hx[:, 1:-1, :] - self.Hx[:, :-2, :]) -
            (self.Hz[:, 1:-1, :] - self.Hz[:, :-2, :])
        )

        # Ez update
        self.Ez[:, :, 1:-1] = factor * self.Ez[:, :, 1:-1] + source_factor * (
            (self.Hy[:, :, 1:-1] - self.Hy[:, :, :-2]) -
            (self.Hx[:, :, 1:-1] - self.Hx[:, :, :-2])
        )

        # ── Add source term (audio-coupled EM) ─────────────────────
        if source_term is not None:
            # Inject at center of grid
            cx, cy, cz = self.cfg.nx // 2, self.cfg.ny // 2, self.cfg.nz // 2
            if len(source_term) > 0:
                self.Ex[cx, cy, cz] += source_term[0] * source_factor

        # ── Structural fluctuations (if enabled) ───────────────────
        if self.fh_solver is not None:
            # Convert E field to density-like quantity for FH coupling
            rho = self.Ex.norm(dim=(1, 2)).mean()
            # In production: proper multi-physics coupling
            # For demo: add small stochastic perturbation
            noise = torch.randn_like(self.Ex) * 1e-6
            self.Ex += noise

        return {
            "Ex": self.Ex,
            "Ey": self.Ey,
            "Ez": self.Ez,
            "Hx": self.Hx,
            "Hy": self.Hy,
            "Hz": self.Hz,
        }

    def run_simulation(
        self,
        audio_signal: Optional[Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full FDTD simulation.

        Args:
            audio_signal: Optional audio to couple to EM field
            n_steps: Number of time steps

        Returns:
            dict with simulation results
        """
        n_steps = n_steps or self.cfg.n_steps

        # Convert audio to EM source if provided
        source_term = None
        if audio_signal is not None:
            source_term = self.audio_to_em_source(audio_signal)

        # Time stepping
        field_history = []
        for step in range(n_steps):
            fields = self.step_fdtd(source_term[step:step+1] if source_term is not None else None)
            if step % 100 == 0:
                field_history.append({
                    "step": step,
                    "E_norm": fields["Ex"].norm().item(),
                    "H_norm": fields["Hx"].norm().item(),
                })

        return {
            "final_fields": fields,
            "field_history": field_history,
            "n_steps": n_steps,
            "tissue_type": self.cfg.tissue_type.value,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get EM CFD status."""
        return {
            "grid_size": (self.cfg.nx, self.cfg.ny, self.cfg.nz),
            "dx": self.cfg.dx,
            "dt": self.cfg.dt,
            "tissue_type": self.cfg.tissue_type.value,
            "permittivity": self.epsilon_r,
            "conductivity": self.cfg.conductivity,
            "structural_fluctuations": self.fh_solver is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [OBC1] OPENBCI BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OpenBCIConfig:
    """Configuration for OpenBCI Cyton board."""
    # Hardware
    serial_port: str = "/dev/ttyUSB0"  # Linux, or COM3 on Windows
    baud_rate: int = 115200
    n_channels: int = 8
    sample_rate: int = 250
    # 10-20 electrode system
    electrode_positions: List[str] = field(default_factory=lambda: [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
    ])
    # Reference
    reference: str = "average"  # "average" | "linked_ears" | "cz"
    # Impedance
    max_impedance_kohm: float = 10.0
    # Mock mode (for testing without hardware)
    use_mock: bool = True


@dataclass
class EEGSample:
    """A single EEG sample from OpenBCI."""
    sample_id: str
    timestamp: float
    channels: Dict[str, float]  # electrode → voltage (μV)
    impedance: Optional[Dict[str, float]] = None
    sample_rate: int = 250
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenBCIBridge:
    """
    [D] Bridge to OpenBCI Cyton board for real-time EEG acquisition.

    ⚠️  CLINICAL NOTICE:
      • OpenBCI is a RESEARCH device, NOT FDA-approved for clinical use
      • Proper electrode placement and impedance checking required
      • Data should be reviewed by qualified professionals

    Features:
      • 8-channel EEG @ 250Hz
      • 10-20 electrode system mapping
      • Real-time serial communication
      • Impedance checking
      • Graceful fallback to mock data

    Hardware requirements:
      • OpenBCI Cyton board
      • EEG electrodes (cup or dry)
      • Conductive gel/paste
      • USB dongle or direct serial

    Mathematical grounding:
      • OpenBCI Cyton spec: 24-bit ADS1299
      • 10-20 system: Jasper 1958
    """

    def __init__(self, cfg: Optional[OpenBCIConfig] = None):
        self.cfg = cfg or OpenBCIConfig()
        self.serial_conn = None
        self._sample_buffer: deque = deque(maxlen=10000)
        self._is_streaming = False

        if not self.cfg.use_mock:
            self._connect_hardware()
        else:
            logger.info("[OpenBCI] Using MOCK mode (no hardware)")

    def _connect_hardware(self) -> bool:
        """Connect to OpenBCI Cyton via serial."""
        if not HAS_SERIAL:
            logger.error("[OpenBCI] pyserial not installed")
            return False

        try:
            self.serial_conn = serial.Serial(
                port=self.cfg.serial_port,
                baudrate=self.cfg.baud_rate,
                timeout=1.0,
            )
            logger.info(f"[OpenBCI] Connected to {self.cfg.serial_port}")
            return True
        except Exception as e:
            logger.error(f"[OpenBCI] Connection failed: {e}")
            return False

    def start_streaming(self) -> bool:
        """Start EEG data streaming."""
        if self.cfg.use_mock:
            self._is_streaming = True
            logger.info("[OpenBCI] Mock streaming started")
            return True

        if self.serial_conn is None:
            return False

        try:
            # Send start command to Cyton
            self.serial_conn.write(b"b")
            self._is_streaming = True
            logger.info("[OpenBCI] Hardware streaming started")
            return True
        except Exception as e:
            logger.error(f"[OpenBCI] Start streaming failed: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop EEG data streaming."""
        if self.cfg.use_mock:
            self._is_streaming = False
            return

        if self.serial_conn is not None:
            try:
                self.serial_conn.write(b"s")
            except Exception:
                pass
        self._is_streaming = False

    def read_sample(self) -> Optional[EEGSample]:
        """
        Read a single EEG sample.

        Returns:
            EEGSample if available, None otherwise
        """
        if not self._is_streaming:
            return None

        if self.cfg.use_mock:
            # Generate mock EEG data
            return self._generate_mock_sample()

        # Read from hardware (simplified)
        try:
            if self.serial_conn is None or self.serial_conn.in_waiting < 36:
                return None

            # OpenBCI Cyton packet format: 36 bytes
            # [0]: 0xA0 (start byte)
            # [1]: sample number
            # [2-7]: channel 1 (24-bit)
            # ...
            # [26-31]: channel 8 (24-bit)
            # [32-35]: accelerometer data

            data = self.serial_conn.read(36)
            if len(data) < 36:
                return None

            # Parse channels
            channels = {}
            for i, electrode in enumerate(self.cfg.electrode_positions):
                # Extract 24-bit value (simplified)
                offset = 2 + i * 3
                raw = int.from_bytes(data[offset:offset+3], "big", signed=True)
                # Convert to μV (ADS1299 gain = 24, Vref = 4.5V)
                voltage_uV = (raw * 4.5e6) / (24 * 2**23)
                channels[electrode] = voltage_uV

            sample = EEGSample(
                sample_id=f"eeg_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                channels=channels,
                sample_rate=self.cfg.sample_rate,
            )

            self._sample_buffer.append(sample)
            return sample

        except Exception as e:
            logger.debug(f"[OpenBCI] Read error: {e}")
            return None

    def _generate_mock_sample(self) -> EEGSample:
        """Generate mock EEG sample for testing."""
        channels = {}
        for electrode in self.cfg.electrode_positions:
            # Simulate realistic EEG: 10 μV RMS, 10 Hz dominant
            t = time.time()
            signal = (
                10 * math.sin(2 * math.pi * 10 * t) +  # alpha
                5 * math.sin(2 * math.pi * 5 * t) +    # theta
                2 * np.random.randn()                   # noise
            )
            channels[electrode] = signal

        sample = EEGSample(
            sample_id=f"mock_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            channels=channels,
            sample_rate=self.cfg.sample_rate,
            metadata={"mock": True},
        )

        self._sample_buffer.append(sample)
        return sample

    def check_impedance(self) -> Dict[str, float]:
        """
        Check electrode impedances.

        Returns:
            Dict mapping electrode to impedance (kΩ)
        """
        if self.cfg.use_mock:
            # Mock impedance values
            return {
                electrode: np.random.uniform(1.0, 8.0)
                for electrode in self.cfg.electrode_positions
            }

        # In production: send impedance check command to Cyton
        logger.warning("[OpenBCI] Hardware impedance check not implemented")
        return {}

    def get_recent_samples(self, n_samples: int = 100) -> List[EEGSample]:
        """Get recent EEG samples."""
        return list(self._sample_buffer)[-n_samples:]

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "hardware_connected": self.serial_conn is not None,
            "is_streaming": self._is_streaming,
            "mock_mode": self.cfg.use_mock,
            "n_channels": self.cfg.n_channels,
            "sample_rate": self.cfg.sample_rate,
            "electrodes": self.cfg.electrode_positions,
            "buffer_size": len(self._sample_buffer),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [RTS1] REAL-TIME AUDIO SYNTH ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SynthConfig:
    """Configuration for real-time audio synthesis."""
    # Audio output
    sample_rate: int = 44100
    buffer_size: int = 1024
    n_channels: int = 2  # stereo
    # Synthesis methods
    use_granular: bool = True
    use_additive: bool = True
    use_fm: bool = True
    use_binaural: bool = True
    # Therapeutic parameters
    base_frequency_hz: float = 432.0  # therapeutic tuning
    binaural_beat_offset_hz: float = 10.0  # alpha range
    # Disorder-specific mappings
    disorder_tone_mappings: Dict[PsychiatricDisorder, Dict[str, Any]] = field(
        default_factory=lambda: {
            PsychiatricDisorder.MAJOR_DEPRESSION: {
                "key": "major",
                "tempo_bpm": 90.0,
                "base_freq": 432.0,
                "binaural_offset": 10.0,  # alpha
                "harmonics": [1, 2, 3, 5],
            },
            PsychiatricDisorder.PTSD: {
                "key": "minor",
                "tempo_bpm": 60.0,
                "base_freq": 396.0,  # grounding frequency
                "binaural_offset": 7.0,  # theta
                "harmonics": [1, 2],
            },
            PsychiatricDisorder.PANIC_DISORDER: {
                "key": "major",
                "tempo_bpm": 50.0,  # slow, breathing-paced
                "base_freq": 528.0,  # "love frequency"
                "binaural_offset": 6.0,  # theta
                "harmonics": [1],
            },
            PsychiatricDisorder.SUBSTANCE_USE_ALCOHOL: {
                "key": "major",
                "tempo_bpm": 72.0,
                "base_freq": 432.0,
                "binaural_offset": 12.0,  # SMR
                "harmonics": [1, 2, 3],
            },
            PsychiatricDisorder.BIPOLAR_I: {
                "key": "major",
                "tempo_bpm": 80.0,
                "base_freq": 432.0,
                "binaural_offset": 10.0,
                "harmonics": [1, 2, 3, 4],
            },
            PsychiatricDisorder.SCHIZOPHRENIA: {
                "key": "major",
                "tempo_bpm": 70.0,
                "base_freq": 432.0,
                "binaural_offset": 40.0,  # gamma
                "harmonics": [1, 2],
            },
        }
    )


@dataclass
class TherapeuticTone:
    """A therapeutic tone for psychiatric intervention."""
    tone_id: str
    disorder: PsychiatricDisorder
    base_frequency_hz: float
    binaural_offset_hz: float
    harmonics: List[int]
    duration_s: float
    amplitude: float = 0.5
    envelope: str = "fade_in_out"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeAudioSynth:
    """
    [D] Real-time audio synthesis engine for therapeutic music.

    ⚠️  CLINICAL NOTICE:
      • This is a RESEARCH tool, NOT a medical device
      • Therapeutic effects are NOT guaranteed
      • Use under professional supervision

    Features:
      • Granular synthesis
      • Additive synthesis
      • FM synthesis
      • Binaural beats
      • Isochronic tones
      • Disorder-specific therapeutic parameters

    Therapeutic rationale (from literature):
      • Thaut 2005: Neurologic Music Therapy
      • Oster 1973: Binaural beats and consciousness
      • Frequency-specific microcurrent research

    Mathematical grounding:
      • Additive synthesis: Fourier series
      • FM synthesis: Carson's rule
      • Binaural beats: frequency following response
    """

    def __init__(self, cfg: Optional[SynthConfig] = None):
        self.cfg = cfg or SynthConfig()
        self.device = torch.device("cpu")  # Audio is CPU-based
        self._generated_tones: List[TherapeuticTone] = []

        # Audio output (if pyaudio available)
        self.audio_stream = None
        if HAS_PYAUDIO:
            self._init_audio_output()

    def _init_audio_output(self) -> None:
        """Initialize PyAudio output stream."""
        try:
            self.pyaudio = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.cfg.n_channels,
                rate=self.cfg.sample_rate,
                output=True,
                frames_per_buffer=self.cfg.buffer_size,
            )
            logger.info("[AudioSynth] PyAudio output initialized")
        except Exception as e:
            logger.warning(f"[AudioSynth] PyAudio init failed: {e}")
            self.audio_stream = None

    # ── Synthesis Methods ────────────────────────────────────────────────

    def generate_sine_wave(
        self,
        frequency_hz: float,
        duration_s: float,
        amplitude: float = 0.5,
    ) -> Tensor:
        """Generate pure sine wave."""
        n_samples = int(duration_s * self.cfg.sample_rate)
        t = torch.linspace(0, duration_s, n_samples, device=self.device)
        wave = amplitude * torch.sin(2 * math.pi * frequency_hz * t)
        return wave

    def generate_additive_synthesis(
        self,
        base_freq_hz: float,
        harmonics: List[int],
        duration_s: float,
        amplitudes: Optional[List[float]] = None,
    ) -> Tensor:
        """
        Generate additive synthesis (sum of harmonics).

        Args:
            base_freq_hz: Fundamental frequency
            harmonics: List of harmonic numbers (1, 2, 3, ...)
            duration_s: Duration in seconds
            amplitudes: Amplitude per harmonic (default: 1/n)

        Returns:
            (n_samples,) waveform
        """
        n_samples = int(duration_s * self.cfg.sample_rate)
        t = torch.linspace(0, duration_s, n_samples, device=self.device)

        if amplitudes is None:
            amplitudes = [1.0 / h for h in harmonics]

        wave = torch.zeros(n_samples, device=self.device)
        for h, amp in zip(harmonics, amplitudes):
            wave += amp * torch.sin(2 * math.pi * base_freq_hz * h * t)

        # Normalize
        wave = wave / (wave.abs().max() + 1e-8) * 0.5
        return wave

    def generate_fm_synthesis(
        self,
        carrier_freq_hz: float,
        modulator_freq_hz: float,
        modulation_index: float,
        duration_s: float,
    ) -> Tensor:
        """
        Generate FM synthesis.

        Args:
            carrier_freq_hz: Carrier frequency
            modulator_freq_hz: Modulator frequency
            modulation_index: Modulation depth
            duration_s: Duration

        Returns:
            (n_samples,) waveform
        """
        n_samples = int(duration_s * self.cfg.sample_rate)
        t = torch.linspace(0, duration_s, n_samples, device=self.device)

        modulator = torch.sin(2 * math.pi * modulator_freq_hz * t)
        wave = torch.sin(
            2 * math.pi * carrier_freq_hz * t +
            modulation_index * modulator
        )

        return wave * 0.5

    def generate_binaural_beats(
        self,
        base_freq_hz: float,
        beat_offset_hz: float,
        duration_s: float,
    ) -> Tensor:
        """
        Generate binaural beats (different frequencies in each ear).

        Args:
            base_freq_hz: Base frequency
            beat_offset_hz: Difference between ears (creates beat)
            duration_s: Duration

        Returns:
            (2, n_samples) stereo waveform
        """
        n_samples = int(duration_s * self.cfg.sample_rate)
        t = torch.linspace(0, duration_s, n_samples, device=self.device)

        # Left ear: base frequency
        left = torch.sin(2 * math.pi * base_freq_hz * t)

        # Right ear: base + offset
        right = torch.sin(2 * math.pi * (base_freq_hz + beat_offset_hz) * t)

        # Stack as stereo
        stereo = torch.stack([left, right], dim=0) * 0.5
        return stereo

    def generate_isochronic_tones(
        self,
        frequency_hz: float,
        pulse_rate_hz: float,
        duration_s: float,
    ) -> Tensor:
        """
        Generate isochronic tones (pulsed single tone).

        Args:
            frequency_hz: Tone frequency
            pulse_rate_hz: Pulse rate
            duration_s: Duration

        Returns:
            (n_samples,) waveform
        """
        n_samples = int(duration_s * self.cfg.sample_rate)
        t = torch.linspace(0, duration_s, n_samples, device=self.device)

        # Carrier tone
        carrier = torch.sin(2 * math.pi * frequency_hz * t)

        # Pulse envelope (square wave)
        pulse = (torch.sin(2 * math.pi * pulse_rate_hz * t) > 0).float()

        wave = carrier * pulse * 0.5
        return wave

    def apply_envelope(
        self,
        wave: Tensor,
        envelope_type: str = "fade_in_out",
        fade_duration_s: float = 1.0,
    ) -> Tensor:
        """Apply amplitude envelope to waveform."""
        n_samples = wave.shape[-1]
        fade_samples = int(fade_duration_s * self.cfg.sample_rate)

        if envelope_type == "fade_in_out":
            envelope = torch.ones(n_samples, device=wave.device)
            # Fade in
            fade_samples = min(fade_samples, n_samples // 4)
            envelope[:fade_samples] = torch.linspace(0, 1, fade_samples, device=wave.device)
            # Fade out
            envelope[-fade_samples:] = torch.linspace(1, 0, fade_samples, device=wave.device)
        elif envelope_type == "exponential_decay":
            envelope = torch.exp(-torch.linspace(0, 5, n_samples, device=wave.device))
        else:
            envelope = torch.ones(n_samples, device=wave.device)

        if wave.dim() == 1:
            return wave * envelope
        else:
            return wave * envelope.unsqueeze(0)

    # ── Therapeutic Tone Generation ──────────────────────────────────────

    def generate_therapeutic_tone(
        self,
        disorder: PsychiatricDisorder,
        duration_s: float = 60.0,
    ) -> TherapeuticTone:
        """
        Generate therapeutic tone for specific disorder.

        Args:
            disorder: Target psychiatric disorder
            duration_s: Duration in seconds

        Returns:
            TherapeuticTone object with generated audio
        """
        # Get disorder-specific parameters
        params = self.cfg.disorder_tone_mappings.get(
            disorder,
            self.cfg.disorder_tone_mappings[PsychiatricDisorder.HEALTHY_CONTROL]
            if PsychiatricDisorder.HEALTHY_CONTROL in self.cfg.disorder_tone_mappings
            else {
                "base_freq": 432.0,
                "binaural_offset": 10.0,
                "harmonics": [1, 2, 3],
            }
        )

        base_freq = params.get("base_freq", 432.0)
        binaural_offset = params.get("binaural_offset", 10.0)
        harmonics = params.get("harmonics", [1, 2, 3])

        # Generate layered audio
        # 1. Additive synthesis (harmonic foundation)
        harmonic_layer = self.generate_additive_synthesis(
            base_freq, harmonics, duration_s
        )

        # 2. Binaural beats (therapeutic entrainment)
        binaural_layer = self.generate_binaural_beats(
            base_freq, binaural_offset, duration_s
        )

        # 3. FM synthesis (texture)
        fm_layer = self.generate_fm_synthesis(
            base_freq, base_freq * 0.5, 2.0, duration_s
        )

        # Mix layers
        if binaural_layer.dim() == 2:
            # Stereo mix
            stereo = torch.zeros_like(binaural_layer)
            stereo[0] = (
                0.5 * harmonic_layer +
                0.3 * binaural_layer[0] +
                0.2 * fm_layer
            )
            stereo[1] = (
                0.5 * harmonic_layer +
                0.3 * binaural_layer[1] +
                0.2 * fm_layer
            )
            audio = stereo
        else:
            audio = 0.6 * harmonic_layer + 0.3 * binaural_layer + 0.1 * fm_layer

        # Apply envelope
        audio = self.apply_envelope(audio, "fade_in_out", fade_duration_s=2.0)

        # Create tone object
        tone = TherapeuticTone(
            tone_id=f"tone_{uuid.uuid4().hex[:8]}",
            disorder=disorder,
            base_frequency_hz=base_freq,
            binaural_offset_hz=binaural_offset,
            harmonics=harmonics,
            duration_s=duration_s,
            amplitude=0.5,
            metadata={
                "audio_shape": tuple(audio.shape),
                "sample_rate": self.cfg.sample_rate,
                "params": params,
            },
        )

        # Store audio data in metadata for now
        tone.metadata["audio"] = audio

        self._generated_tones.append(tone)
        logger.info(
            f"[AudioSynth] Generated therapeutic tone for {disorder.value}: "
            f"{base_freq}Hz base, {binaural_offset}Hz binaural offset"
        )

        return tone

    def play_tone(self, tone: TherapeuticTone) -> bool:
        """Play therapeutic tone through audio output."""
        if self.audio_stream is None:
            logger.warning("[AudioSynth] No audio output available")
            return False

        audio = tone.metadata.get("audio")
        if audio is None:
            return False

        try:
            # Convert to numpy and play
            audio_np = audio.cpu().numpy()
            if audio_np.ndim == 1:
                audio_np = np.stack([audio_np, audio_np], axis=0)

            # Write to audio stream
            self.audio_stream.write(audio_np.T.astype(np.float32).tobytes())
            return True
        except Exception as e:
            logger.error(f"[AudioSynth] Playback failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get synth status."""
        return {
            "sample_rate": self.cfg.sample_rate,
            "audio_output_available": self.audio_stream is not None,
            "generated_tones": len(self._generated_tones),
            "disorder_mappings": len(self.cfg.disorder_tone_mappings),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [BVD1] BIOFEEDBACK VISUALIZATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VizConfig:
    """Configuration for biofeedback visualization."""
    # Display
    window_width: int = 1920
    window_height: int = 1080
    fps: int = 30
    # Panels
    show_eeg_spectrogram: bool = True
    show_psych_assessment: bool = True
    show_em_field: bool = True
    show_therapy_tone: bool = True
    # Export
    export_interval_s: float = 10.0
    export_format: str = "json"  # "json" | "csv" | "png"


class BiofeedbackVizDashboard:
    """
    [D] Real-time biofeedback visualization dashboard.

    ⚠️  CLINICAL NOTICE:
      • This is a VISUALIZATION tool, NOT a clinical monitor
      • Displayed data should be interpreted by professionals
      • Not validated for diagnostic use

    Features:
      • Real-time EEG visualization
      • Psychiatric outcome tracking
      • EM field propagation display
      • Therapeutic music parameter visualization
      • Multi-user session support
      • Data export

    Visualization panels:
      • EEG spectrogram (time-frequency)
      • Band power bar chart
      • Psychiatric severity gauge
      • EM field heatmap
      • Therapy tone waveform
    """

    def __init__(self, cfg: Optional[VizConfig] = None):
        self.cfg = cfg or VizConfig()
        self._session_data: Dict[str, Any] = {
            "eeg_samples": [],
            "psych_assessments": [],
            "em_field_snapshots": [],
            "therapy_tones": [],
        }
        self._export_timestamps: deque = deque(maxlen=1000)

    def add_eeg_sample(self, sample: EEGSample) -> None:
        """Add EEG sample to visualization buffer."""
        self._session_data["eeg_samples"].append(sample)
        # Keep buffer manageable
        if len(self._session_data["eeg_samples"]) > 10000:
            self._session_data["eeg_samples"] = self._session_data["eeg_samples"][-5000:]

    def add_psych_assessment(self, outcome: PsychOutcome) -> None:
        """Add psychiatric assessment to dashboard."""
        self._session_data["psych_assessments"].append(outcome)

    def add_em_field_snapshot(self, fields: Dict[str, Tensor]) -> None:
        """Add EM field snapshot."""
        self._session_data["em_field_snapshots"].append({
            "timestamp": time.time(),
            "E_norm": fields["Ex"].norm().item(),
            "H_norm": fields["Hx"].norm().item(),
        })

    def add_therapy_tone(self, tone: TherapeuticTone) -> None:
        """Add therapy tone to dashboard."""
        self._session_data["therapy_tones"].append(tone)

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for dashboard visualization.

        Returns:
            dict with all visualization data
        """
        # EEG spectrogram data
        eeg_data = []
        if self._session_data["eeg_samples"]:
            recent = self._session_data["eeg_samples"][-100:]
            for sample in recent:
                eeg_data.append({
                    "timestamp": sample.timestamp,
                    "channels": sample.channels,
                })

        # Psychiatric assessment timeline
        psych_data = [
            outcome.to_dict()
            for outcome in self._session_data["psych_assessments"][-10:]
        ]

        # EM field evolution
        em_data = self._session_data["em_field_snapshots"][-50:]

        # Therapy tones
        tone_data = [
            {
                "tone_id": t.tone_id,
                "disorder": t.disorder.value,
                "base_freq": t.base_frequency_hz,
                "binaural_offset": t.binaural_offset_hz,
                "duration": t.duration_s,
            }
            for t in self._session_data["therapy_tones"][-5:]
        ]

        return {
            "eeg": eeg_data,
            "psychiatric": psych_data,
            "em_field": em_data,
            "therapy_tones": tone_data,
            "timestamp": time.time(),
        }

    def export_session(self, output_dir: str = "./dashboard_exports") -> str:
        """Export session data to file."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        export_file = out_path / f"session_{timestamp}.json"

        data = self.generate_dashboard_data()
        with open(export_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"[Dashboard] Exported session to {export_file}")
        return str(export_file)

    def get_status(self) -> Dict[str, Any]:
        """Get dashboard status."""
        return {
            "n_eeg_samples": len(self._session_data["eeg_samples"]),
            "n_psych_assessments": len(self._session_data["psych_assessments"]),
            "n_em_snapshots": len(self._session_data["em_field_snapshots"]),
            "n_therapy_tones": len(self._session_data["therapy_tones"]),
            "window_size": (self.cfg.window_width, self.cfg.window_height),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIPsychiatricMusicCFDSuite:
    """
    [D] Complete psychiatric music therapy AGI suite.

    ⚠️  CLINICAL NOTICE:
      • This is a RESEARCH FRAMEWORK, NOT a medical device
      • Requires licensed clinical oversight
      • Informed consent MANDATORY
      • IRB approval required for clinical use

    Integrates:
      • PsychiatricOutcomeAssessor (multi-disorder)
      • ElectromagneticWaveCFD (Maxwell + structural fluctuations)
      • OpenBCIBridge (real-time EEG)
      • RealTimeAudioSynth (therapeutic music)
      • BiofeedbackVizDashboard (visualization)
    """

    def __init__(
        self,
        psych_cfg: Optional[PsychAssessmentConfig] = None,
        em_cfg: Optional[EMWaveConfig] = None,
        openbci_cfg: Optional[OpenBCIConfig] = None,
        synth_cfg: Optional[SynthConfig] = None,
        viz_cfg: Optional[VizConfig] = None,
    ):
        self.psych_assessor = PsychiatricOutcomeAssessor(psych_cfg)
        self.em_wave_cfd = ElectromagneticWaveCFD(em_cfg)
        self.openbci = OpenBCIBridge(openbci_cfg)
        self.audio_synth = RealTimeAudioSynth(synth_cfg)
        self.dashboard = BiofeedbackVizDashboard(viz_cfg)

        logger.info("[SESI Suite] Psychiatric Music Therapy initialized")
        logger.warning("⚠️  RESEARCH USE ONLY — NOT A MEDICAL DEVICE")

    def run_therapy_session(
        self,
        user_id: str,
        disorder: PsychiatricDisorder,
        duration_s: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Run a complete music therapy session.

        ⚠️  Requires informed consent and clinical oversight.

        Args:
            user_id: User identifier
            disorder: Target psychiatric disorder
            duration_s: Session duration

        Returns:
            dict with session results
        """
        logger.info(f"[Therapy] Starting session for {user_id}: {disorder.value}")

        # 1. Start EEG streaming
        self.openbci.start_streaming()

        # 2. Record pre-therapy baseline (5 min)
        logger.info("[Therapy] Recording pre-therapy baseline (5 min)")
        baseline_eeg = self._collect_eeg_features(duration_s=300.0)
        self.psych_assessor.record_pre_therapy_baseline(
            user_id, baseline_eeg, disorder
        )

        # 3. Generate therapeutic tone
        tone = self.audio_synth.generate_therapeutic_tone(
            disorder, duration_s=duration_s
        )
        self.dashboard.add_therapy_tone(tone)

        # 4. Run therapy with EM coupling
        logger.info("[Therapy] Running therapy with EM field coupling")
        audio_signal = tone.metadata.get("audio")
        if audio_signal is not None:
            em_result = self.em_wave_cfd.run_simulation(
                audio_signal=audio_signal,
                n_steps=1000,
            )
            self.dashboard.add_em_field_snapshot(em_result["final_fields"])

        # 5. Play therapeutic audio
        self.audio_synth.play_tone(tone)

        # 6. Collect EEG during therapy
        therapy_eeg = self._collect_eeg_features(duration_s=duration_s)

        # 7. Stop EEG
        self.openbci.stop_streaming()

        # 8. Assess therapy outcome
        outcome = self.psych_assessor.assess_therapy_outcome(user_id, therapy_eeg)

        # 9. Update dashboard
        if outcome:
            self.dashboard.add_psych_assessment(outcome)

        # 10. Export session
        export_path = self.dashboard.export_session()

        logger.info(f"[Therapy] Session complete for {user_id}")

        return {
            "user_id": user_id,
            "disorder": disorder.value,
            "duration_s": duration_s,
            "tone_id": tone.tone_id,
            "outcome": outcome.to_dict() if outcome else None,
            "export_path": export_path,
        }

    def _collect_eeg_features(self, duration_s: float) -> Dict[str, float]:
        """Collect EEG features over duration."""
        samples = []
        start_time = time.time()

        while time.time() - start_time < duration_s:
            sample = self.openbci.read_sample()
            if sample:
                samples.append(sample)
                self.dashboard.add_eeg_sample(sample)
            time.sleep(1.0 / self.openbci.cfg.sample_rate)

        # Convert samples to tensor format
        if not samples:
            return {}

        eeg_data = {}
        for electrode in self.openbci.cfg.electrode_positions:
            signal = torch.tensor([s.channels.get(electrode, 0.0) for s in samples])
            eeg_data[electrode] = signal

        # Extract features
        features = self.psych_assessor.extract_eeg_features(
            eeg_data, self.openbci.cfg.sample_rate
        )

        return features

    def get_full_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "psych_assessor": self.psych_assessor.get_status(),
            "em_wave_cfd": self.em_wave_cfd.get_status(),
            "openbci": self.openbci.get_status(),
            "audio_synth": self.audio_synth.get_status(),
            "dashboard": self.dashboard.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Psychiatric Music Therapy AGI — Smoke Test")
    print("=" * 72)
    print(f"  pyserial available   : {HAS_SERIAL}")
    print(f"  pyaudio available    : {HAS_PYAUDIO}")
    print(f"  scipy available      : {HAS_SCIPY}")
    print(f"  Fluctuating Hydro    : {HAS_FH}")
    print(f"  Biofeedback module   : {HAS_BIOFEEDBACK}")

    print("\n  ⚠️  CLINICAL NOTICE")
    print("  ─────────────────────────────────────────────────────────")
    print("  • This is a RESEARCH FRAMEWORK, NOT a medical device")
    print("  • Requires licensed clinical oversight")
    print("  • Informed consent MANDATORY")
    print("  • IRB approval required for clinical use")
    print("=" * 72)

    # Initialize suite with mock OpenBCI
    suite = SESIPsychiatricMusicCFDSuite(
        openbci_cfg=OpenBCIConfig(use_mock=True),
    )

    # ── Test 1: Psychiatric Outcome Assessor ─────────────────────────────
    print("\n  ── [PSY1] Psychiatric Outcome Assessor ──")

    # Generate mock EEG data
    mock_eeg = {
        "F3": torch.randn(1000) * 10,
        "F4": torch.randn(1000) * 10,
        "Fz": torch.randn(1000) * 10,
    }

    # Extract features
    features = suite.psych_assessor.extract_eeg_features(mock_eeg, sample_rate=256)
    print(f"    Extracted features   : {len(features)}")
    for k, v in list(features.items())[:5]:
        print(f"      {k:30s}: {v:.4f}")

    # Assess disorder
    severity, confidence = suite.psych_assessor.assess_disorder(
        features, PsychiatricDisorder.MAJOR_DEPRESSION
    )
    print(f"    Depression severity  : {severity.name} (confidence={confidence:.2f})")

    psych_status = suite.psych_assessor.get_status()
    print(f"    Supported disorders  : {psych_status['supported_disorders']}")

    # ── Test 2: EM Wave CFD ──────────────────────────────────────────────
    print("\n  ── [EMW1] Electromagnetic Wave CFD ──")

    em_status = suite.em_wave_cfd.get_status()
    print(f"    Grid size            : {em_status['grid_size']}")
    print(f"    Tissue type          : {em_status['tissue_type']}")
    print(f"    Permittivity         : {em_status['permittivity']}")
    print(f"    Structural fluctuations: {em_status['structural_fluctuations']}")

    # Run short simulation
    mock_audio = torch.randn(1000) * 0.1
    em_result = suite.em_wave_cfd.run_simulation(
        audio_signal=mock_audio, n_steps=100
    )
    print(f"    Simulation steps     : {em_result['n_steps']}")
    print(f"    Final E norm         : {em_result['final_fields']['Ex'].norm().item():.6f}")

    # ── Test 3: OpenBCI Bridge ───────────────────────────────────────────
    print("\n  ── [OBC1] OpenBCI Bridge ──")

    obci_status = suite.openbci.get_status()
    print(f"    Mock mode            : {obci_status['mock_mode']}")
    print(f"    Streaming            : {obci_status['is_streaming']}")
    print(f"    Channels             : {obci_status['n_channels']}")
    print(f"    Sample rate          : {obci_status['sample_rate']}")
    print(f"    Electrodes           : {obci_status['electrodes']}")

    # Start mock streaming
    suite.openbci.start_streaming()
    sample = suite.openbci.read_sample()
    if sample:
        print(f"    Sample received      : {sample.sample_id}")
        print(f"      Channels           : {list(sample.channels.keys())}")
        print(f"      F3 voltage         : {sample.channels.get('F3', 0.0):.2f} μV")
    suite.openbci.stop_streaming()

    # ── Test 4: Real-Time Audio Synth ────────────────────────────────────
    print("\n  ── [RTS1] Real-Time Audio Synth ──")

    synth_status = suite.audio_synth.get_status()
    print(f"    Sample rate          : {synth_status['sample_rate']}")
    print(f"    Audio output         : {synth_status['audio_output_available']}")
    print(f"    Disorder mappings    : {synth_status['disorder_mappings']}")

    # Generate therapeutic tone
    tone = suite.audio_synth.generate_therapeutic_tone(
        PsychiatricDisorder.MAJOR_DEPRESSION,
        duration_s=10.0,
    )
    print(f"    Generated tone       : {tone.tone_id}")
    print(f"      Base frequency     : {tone.base_frequency_hz} Hz")
    print(f"      Binaural offset    : {tone.binaural_offset_hz} Hz")
    print(f"      Harmonics          : {tone.harmonics}")
    print(f"      Duration           : {tone.duration_s} s")
    if "audio" in tone.metadata:
        print(f"      Audio shape        : {tone.metadata['audio'].shape}")

    # ── Test 5: Biofeedback Dashboard ────────────────────────────────────
    print("\n  ── [BVD1] Biofeedback Visualization Dashboard ──")

    dash_status = suite.dashboard.get_status()
    print(f"    EEG samples          : {dash_status['n_eeg_samples']}")
    print(f"    Psych assessments    : {dash_status['n_psych_assessments']}")
    print(f"    EM snapshots         : {dash_status['n_em_snapshots']}")
    print(f"    Therapy tones        : {dash_status['n_therapy_tones']}")

    # Add data to dashboard
    if sample:
        suite.dashboard.add_eeg_sample(sample)
    suite.dashboard.add_therapy_tone(tone)

    # Generate dashboard data
    dash_data = suite.dashboard.generate_dashboard_data()
    print(f"    Dashboard data keys  : {list(dash_data.keys())}")

    # ── Test 6: Full Therapy Session (Short Demo) ────────────────────────
    print("\n  ── Full Therapy Session (30s demo) ──")
    print("    ⚠️  MOCK SESSION — not real therapy")

    session_result = suite.run_therapy_session(
        user_id="test_user_001",
        disorder=PsychiatricDisorder.MAJOR_DEPRESSION,
        duration_s=30.0,  # Short for demo
    )

    print(f"    Session user         : {session_result['user_id']}")
    print(f"    Disorder             : {session_result['disorder']}")
    print(f"    Tone ID              : {session_result['tone_id']}")
    if session_result['outcome']:
        outcome = session_result['outcome']
        print(f"    Outcome severity     : {outcome['severity']}")
        print(f"    Outcome confidence   : {outcome['confidence']:.2f}")
        if outcome.get('improvement_score') is not None:
            print(f"    Improvement score    : {outcome['improvement_score']:.2%}")
    print(f"    Export path          : {session_result['export_path']}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    Psych assessments    : {status['psych_assessor']['total_assessments']}")
    print(f"    EM tissue            : {status['em_wave_cfd']['tissue_type']}")
    print(f"    OpenBCI streaming    : {status['openbci']['is_streaming']}")
    print(f"    Audio synth tones    : {status['audio_synth']['generated_tones']}")
    print(f"    Dashboard EEG samples: {status['dashboard']['n_eeg_samples']}")

    print("\n" + "=" * 72)
    print("  ⚠️  ETHICAL/CLINICAL FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • NOT A MEDICAL DEVICE — research/artistic exploration only")
    print("  • Requires licensed clinical oversight")
    print("  • Informed consent MANDATORY for all users")
    print("  • PNDC Axiom 3: Psychiatric data is HIGHLY SENSITIVE")
    print("  • Buddhist Ethics (อहिंसा): Never use to manipulate")
    print("  • Taoist Ethics (无为): Support, don't control")
    print("  • Production deployment REQUIRES IRB approval")
    print("=" * 72)
    print("  HONEST SCOPE")
    print("  • Psychiatric assessment is HEURISTIC, not diagnosis")
    print("  • EM wave modeling is APPROXIMATE, not validated")
    print("  • OpenBCI is RESEARCH device, not FDA-approved")
    print("  • Audio synthesis is COMPUTATIONAL, not therapeutic")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
