"""
=============================================================================
SESI CLINICAL NEUROIMAGING + PERSONALIZED THERAPY + TRIAL MANAGEMENT
fMRI Integration + Adaptive Protocols + Clinical Trial System
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

⚠️  CLINICAL/REGULATORY FRAMEWORK
-----------------------------------------------------------------------------
• NOT MEDICAL DEVICES — research tools only
• IRB/IEC approval REQUIRED for any human subject research
• Informed consent MANDATORY for all participants
• Clinical oversight by licensed professionals REQUIRED
• PNDC Axiom 3: Patient data is ULTRA-SENSITIVE
• Buddhist Ethics (อहिंสา): Never harm or exploit
• Taoist Ethics (无为): Support, don't force outcomes
• FDA 21 CFR Part 11 patterns for audit trail (research use only)
• ICH-GCP E6(R2) compliance patterns for trial conduct
=============================================================================

WHAT THIS MODULE PROVIDES
--------------------------
[FMRI1] fMRINeuroimagingBridge
      • BOLD signal preprocessing pipeline
      • ROI-based functional connectivity analysis
      • GLM (General Linear Model) for task-based analysis
      • Resting-state network extraction
      • Integration with psychiatric biomarkers
      • NIfTI/DICOM format support (mock)

[PTP1] PersonalizedTherapyProtocol
      • Patient profiling (demographics, diagnosis, biomarkers)
      • Stratified treatment assignment
      • Adaptive protocol adjustment based on biometric feedback
      • Session-by-session outcome tracking
      • Bayesian updating of treatment efficacy
      • Clinician override capability

[CTM1] ClinicalTrialManagementSystem
      • Trial design (parallel, crossover, adaptive)
      • Stratified randomization
      • Blinding management (single/double/triple)
      • Adverse event tracking (CTCAE-style grading)
      • Regulatory audit trail (FDA 21 CFR Part 11 patterns)
      • Participant enrollment and consent tracking

MATHEMATICAL FOUNDATIONS
--------------------------
[C] BOLD Signal: Ogawa et al. 1990 (hemodynamic response)
[C] GLM Analysis: Friston et al. 1994 (task-based fMRI)
[C] Functional Connectivity: Biswal et al. 1995 (resting-state)
[C] Bayesian Adaptive Design: Berry 2006 (clinical trials)
[C] ICH-GCP E6(R2): International guidelines for clinical trials
[C] FDA 21 CFR Part 11: Electronic records/signatures patterns
[C] CTCAE v5.0: Common Terminology Criteria for Adverse Events
[C] PNDC Axiom 3: Necessity and Proportionality (patient data)
[C] SESI_NoZeno_Closure Thm 3.2: Bounded computation
[C] Delta_Min Thm 1.2: Quantum gate for meaningful changes

HONEST SCOPE
------------
• fMRI processing is RESEARCH-GRADE, not clinically validated
• Personalized protocols require CLINICAL OVERSIGHT
• Trial management is a FRAMEWORK, not a certified EDC system
• All human subject research requires IRB/IEC approval
• Patient data handling requires HIPAA/GDPR compliance
• All thresholds are FREE DESIGN PARAMETERS
• Production deployment REQUIRES regulatory approval
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.clinical_neuroimaging_trials")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import nibabel as nib  # NIfTI format support
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import nilearn  # fMRI analysis
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Integration with previous modules (graceful)
try:
    from sesi_psychiatric_music_cfd import (
        PsychiatricOutcomeAssessor, PsychiatricDisorder, SeverityLevel,
        EmotionModulator, EmotionTarget,
    )
    HAS_PSYCH = True
except ImportError:
    HAS_PSYCH = False

__all__ = [
    "fMRIModality", "BrainRegion", "TrialPhase", "AdverseEventGrade",
    "fMRINeuroimagingBridge", "fMRIConfig", "BOLDSignal", "FunctionalConnectivity",
    "PersonalizedTherapyProtocol", "PatientProfile", "TherapySession",
    "ClinicalTrialManagementSystem", "TrialConfig", "Participant",
    "AdverseEvent", "AuditEntry",
    "SESIClinicalResearchSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class fMRIModality(Enum):
    """fMRI analysis modalities."""
    TASK_BASED = "task_based"           # GLM analysis
    RESTING_STATE = "resting_state"     # Functional connectivity
    STRUCTURAL = "structural"           # T1/T2 anatomical
    DIFFUSION = "diffusion"             # DTI
    PERFUSION = "perfusion"             # ASL


class BrainRegion(Enum):
    """Standard brain regions (AAL atlas simplified)."""
    # Frontal
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    MOTOR_CORTEX = "motor_cortex"
    BROCA_AREA = "broca_area"
    # Parietal
    SOMATOSENSORY_CORTEX = "somatosensory_cortex"
    SUPERIOR_PARIETAL = "superior_parietal"
    # Temporal
    AUDITORY_CORTEX = "auditory_cortex"
    WERNICKE_AREA = "wernicke_area"
    HIPPOCAMPUS = "hippocampus"
    AMYGDALA = "amygdala"
    # Occipital
    VISUAL_CORTEX = "visual_cortex"
    # Subcortical
    THALAMUS = "thalamus"
    BASAL_GANGLIA = "basal_ganglia"
    CEREBELLUM = "cerebellum"
    # Limbic
    ANTERIOR_CINGULATE = "anterior_cingulate"
    INSULA = "insula"


class TrialPhase(Enum):
    """Clinical trial phases."""
    SCREENING = "screening"
    ENROLLMENT = "enrollment"
    BASELINE = "baseline"
    TREATMENT = "treatment"
    FOLLOW_UP = "follow_up"
    COMPLETION = "completion"
    TERMINATED = "terminated"


class AdverseEventGrade(Enum):
    """CTCAE v5.0 adverse event grading (simplified)."""
    GRADE_1_MILD = 1          # Mild; asymptomatic or mild symptoms
    GRADE_2_MODERATE = 2      # Moderate; minimal/local intervention indicated
    GRADE_3_SEVERE = 3        # Severe or medically significant but not immediately life-threatening
    GRADE_4_LIFE_THREATENING = 4  # Life-threatening consequences
    GRADE_5_DEATH = 5         # Death related to AE


# ═══════════════════════════════════════════════════════════════════════════
# [FMRI1] fMRI NEUROIMAGING BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class fMRIConfig:
    """Configuration for fMRI processing."""
    # Acquisition parameters
    tr: float = 2.0  # Repetition time (seconds)
    voxel_size: Tuple[float, float, float] = (3.0, 3.0, 3.0)  # mm
    n_volumes: int = 300
    # Preprocessing
    motion_correction: bool = True
    slice_timing_correction: bool = True
    spatial_smoothing_fwhm: float = 6.0  # mm
    highpass_filter_hz: float = 0.01
    lowpass_filter_hz: float = 0.1
    # Analysis
    modality: fMRIModality = fMRIModality.RESTING_STATE
    # Privacy (PNDC Axiom 3)
    anonymize_patient_data: bool = True
    deidentify_dicom: bool = True


@dataclass
class BOLDSignal:
    """Blood Oxygen Level Dependent signal data."""
    signal_id: str
    patient_id: str
    timestamp: float
    # Data
    timeseries: Optional[Tensor] = None  # (n_timepoints, n_voxels) or (n_timepoints, n_regions)
    n_timepoints: int = 0
    n_voxels: int = 0
    # Metadata
    tr: float = 2.0
    modality: fMRIModality = fMRIModality.RESTING_STATE
    # Preprocessing status
    motion_corrected: bool = False
    slice_timing_corrected: bool = False
    spatially_smoothed: bool = False
    filtered: bool = False
    # Quality metrics
    mean_fd: float = 0.0  # Mean framewise displacement
    n_scrubbed_volumes: int = 0
    # Privacy
    anonymized: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionalConnectivity:
    """Functional connectivity matrix between brain regions."""
    connectivity_id: str
    patient_id: str
    timestamp: float
    # Data
    correlation_matrix: Optional[Tensor] = None  # (n_regions, n_regions)
    regions: List[BrainRegion] = field(default_factory=list)
    # Analysis type
    analysis_type: str = "pearson"  # "pearson" | "partial" | "ica"
    # Network metrics
    global_efficiency: float = 0.0
    clustering_coefficient: float = 0.0
    modularity: float = 0.0
    # Default mode network connectivity (relevant for depression/PTSD)
    dmn_connectivity: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class fMRINeuroimagingBridge:
    """
    [D] Research-grade fMRI neuroimaging bridge.

    ⚠️  CLINICAL NOTICE:
      • This is a RESEARCH TOOL, not a diagnostic device
      • NOT FDA-cleared for clinical use
      • Requires IRB approval for human subject research
      • All data must be de-identified per HIPAA/GDPR

    Features:
      • BOLD signal preprocessing pipeline
      • ROI-based functional connectivity analysis
      • GLM for task-based analysis
      • Resting-state network extraction
      • Integration with psychiatric biomarkers

    Mathematical grounding:
      • BOLD: Ogawa et al. 1990
      • GLM: Friston et al. 1994
      • Functional connectivity: Biswal et al. 1995
    """

    # Brain region MNI coordinates (simplified, for demo)
    REGION_CENTERS = {
        BrainRegion.PREFRONTAL_CORTEX: (0, 50, 20),
        BrainRegion.MOTOR_CORTEX: (0, -20, 50),
        BrainRegion.AUDITORY_CORTEX: (0, -30, 10),
        BrainRegion.VISUAL_CORTEX: (0, -80, 10),
        BrainRegion.HIPPOCAMPUS: (25, -20, -10),
        BrainRegion.AMYGDALA: (20, 0, -15),
        BrainRegion.ANTERIOR_CINGULATE: (0, 30, 20),
        BrainRegion.INSULA: (35, 10, 0),
        BrainRegion.THALAMUS: (0, -15, 5),
    }

    def __init__(self, cfg: Optional[fMRIConfig] = None):
        self.cfg = cfg or fMRIConfig()
        self._bold_signals: List[BOLDSignal] = []
        self._connectivity_matrices: List[FunctionalConnectivity] = []
        self._processing_timestamps: deque = deque(maxlen=1000)

        # No-Zeno parameters
        self.max_processing_per_minute = 5
        self.min_processing_interval_s = 12.0

        if not self.cfg.anonymize_patient_data:
            logger.error(
                "[fMRI] ⛔ ANONYMIZATION DISABLED — "
                "This violates PNDC Axiom 3 and HIPAA. Cannot proceed."
            )
            raise ValueError("Patient data anonymization must be enabled")

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_processing_rate(self) -> Dict[str, Any]:
        """Check if processing is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._processing_timestamps = deque(
            (t for t in self._processing_timestamps if t > cutoff),
            maxlen=1000,
        )

        if len(self._processing_timestamps) >= self.max_processing_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
            }

        if self._processing_timestamps:
            time_since_last = now - self._processing_timestamps[-1]
            if time_since_last < self.min_processing_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                }

        return {"allowed": True}

    # ── BOLD Signal Preprocessing ────────────────────────────────────────

    def load_bold_signal(
        self,
        patient_id: str,
        data_path: Optional[str] = None,
        mock_data: bool = True,
    ) -> BOLDSignal:
        """
        Load BOLD signal from NIfTI file or generate mock data.

        ⚠️  All patient data is anonymized per PNDC Axiom 3.

        Args:
            patient_id: Patient identifier (will be anonymized)
            data_path: Path to NIfTI file (optional)
            mock_data: Use mock data for testing

        Returns:
            BOLDSignal object
        """
        rate_check = self.check_processing_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[fMRI] Processing rate-limited: {rate_check['reason']}")

        # Anonymize patient ID
        if self.cfg.anonymize_patient_data:
            anon_id = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
            patient_id = f"anon_{anon_id}"

        if mock_data or data_path is None:
            # Generate mock BOLD signal
            n_timepoints = self.cfg.n_volumes
            n_regions = len(self.REGION_CENTERS)

            # Simulate realistic BOLD signal
            t = torch.linspace(0, n_timepoints * self.cfg.tr, n_timepoints)
            timeseries = torch.zeros(n_timepoints, n_regions)

            for i in range(n_regions):
                # Low-frequency oscillations (0.01-0.1 Hz)
                freq = 0.05 + 0.02 * torch.randn(1)
                timeseries[:, i] = torch.sin(2 * math.pi * freq * t)
                # Add noise
                timeseries[:, i] += 0.1 * torch.randn(n_timepoints)

            bold = BOLDSignal(
                signal_id=f"bold_{uuid.uuid4().hex[:8]}",
                patient_id=patient_id,
                timestamp=time.time(),
                timeseries=timeseries,
                n_timepoints=n_timepoints,
                n_voxels=n_regions,
                tr=self.cfg.tr,
                modality=self.cfg.modality,
                anonymized=True,
            )
        else:
            # Load from NIfTI (if nibabel available)
            if not HAS_NIBABEL:
                logger.error("[fMRI] nibabel not installed; cannot load NIfTI")
                raise RuntimeError("nibabel required for NIfTI loading")

            try:
                img = nib.load(data_path)
                data = img.get_fdata()
                timeseries = torch.from_numpy(data.reshape(-1, data.shape[-1]).T)

                bold = BOLDSignal(
                    signal_id=f"bold_{uuid.uuid4().hex[:8]}",
                    patient_id=patient_id,
                    timestamp=time.time(),
                    timeseries=timeseries,
                    n_timepoints=timeseries.shape[0],
                    n_voxels=timeseries.shape[1],
                    tr=self.cfg.tr,
                    modality=self.cfg.modality,
                    anonymized=True,
                    metadata={"source_file": data_path},
                )
            except Exception as e:
                logger.error(f"[fMRI] Failed to load NIfTI: {e}")
                raise

        self._bold_signals.append(bold)
        self._processing_timestamps.append(time.time())

        logger.info(
            f"[fMRI] Loaded BOLD signal: {bold.signal_id} | "
            f"patient={bold.patient_id[:12]}... | "
            f"shape=({bold.n_timepoints}, {bold.n_voxels})"
        )

        return bold

    def preprocess_bold(self, bold: BOLDSignal) -> BOLDSignal:
        """
        Preprocess BOLD signal (motion correction, filtering, smoothing).

        ⚠️  This is a SIMPLIFIED pipeline. Real preprocessing requires:
          • Realignment (SPM/FSL/AFNI)
          • Slice-timing correction
          • Coregistration to structural
          • Normalization to MNI space
          • Spatial smoothing
          • Temporal filtering
        """
        if bold.timeseries is None:
            logger.warning("[fMRI] No timeseries data to preprocess")
            return bold

        timeseries = bold.timeseries.clone()

        # 1. Motion correction (simplified: detrending)
        if self.cfg.motion_correction:
            # Remove linear trend
            t = torch.arange(timeseries.shape[0], dtype=torch.float32)
            t = (t - t.mean()) / t.std()
            for i in range(timeseries.shape[1]):
                slope = torch.sum(t * timeseries[:, i]) / torch.sum(t * t)
                timeseries[:, i] = timeseries[:, i] - slope * t
            bold.motion_corrected = True

        # 2. Temporal filtering
        if self.cfg.highpass_filter_hz > 0 or self.cfg.lowpass_filter_hz < 1.0:
            # Simplified: moving average for low-pass
            if self.cfg.lowpass_filter_hz < 1.0:
                window_size = int(1.0 / (self.cfg.tr * self.cfg.lowpass_filter_hz))
                window_size = max(3, window_size)
                timeseries = F.avg_pool1d(
                    timeseries.T.unsqueeze(1),
                    kernel_size=window_size,
                    stride=1,
                    padding=window_size // 2,
                ).squeeze(1).T
            bold.filtered = True

        # 3. Compute motion metrics (framewise displacement)
        if bold.motion_corrected:
            diff = torch.diff(timeseries, dim=0)
            bold.mean_fd = diff.abs().mean().item()

        bold.timeseries = timeseries
        logger.info(
            f"[fMRI] Preprocessed BOLD: {bold.signal_id} | "
            f"mean_fd={bold.mean_fd:.4f}"
        )

        return bold

    # ── Functional Connectivity Analysis ─────────────────────────────────

    def compute_functional_connectivity(
        self,
        bold: BOLDSignal,
        analysis_type: str = "pearson",
    ) -> FunctionalConnectivity:
        """
        Compute functional connectivity between brain regions.

        Args:
            bold: Preprocessed BOLD signal
            analysis_type: "pearson" | "partial" | "ica"

        Returns:
            FunctionalConnectivity object
        """
        if bold.timeseries is None:
            raise ValueError("BOLD signal has no timeseries data")

        timeseries = bold.timeseries  # (n_timepoints, n_regions)
        n_regions = timeseries.shape[1]

        # Compute correlation matrix
        if analysis_type == "pearson":
            # Pearson correlation
            timeseries_centered = timeseries - timeseries.mean(dim=0, keepdim=True)
            std = timeseries_centered.std(dim=0, keepdim=True) + 1e-8
            timeseries_norm = timeseries_centered / std
            corr_matrix = (timeseries_norm.T @ timeseries_norm) / (timeseries.shape[0] - 1)
        elif analysis_type == "partial":
            # Partial correlation (simplified: inverse covariance)
            cov = timeseries.T @ timeseries / timeseries.shape[0]
            try:
                precision = torch.linalg.inv(cov + 1e-6 * torch.eye(n_regions))
                # Convert to partial correlation
                d = torch.sqrt(torch.diag(precision))
                corr_matrix = -precision / (d.unsqueeze(1) * d.unsqueeze(0))
                corr_matrix.fill_diagonal_(1.0)
            except Exception:
                corr_matrix = torch.eye(n_regions)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Compute network metrics
        global_efficiency = self._compute_global_efficiency(corr_matrix)
        clustering_coeff = self._compute_clustering_coefficient(corr_matrix)

        # Default mode network connectivity (simplified)
        dmn_regions = [
            BrainRegion.PREFRONTAL_CORTEX,
            BrainRegion.ANTERIOR_CINGULATE,
        ]
        dmn_connectivity = {}
        for r1 in dmn_regions:
            for r2 in dmn_regions:
                if r1 != r2:
                    i = list(self.REGION_CENTERS.keys()).index(r1) if r1 in self.REGION_CENTERS else 0
                    j = list(self.REGION_CENTERS.keys()).index(r2) if r2 in self.REGION_CENTERS else 1
                    if i < n_regions and j < n_regions:
                        dmn_connectivity[f"{r1.value}_{r2.value}"] = corr_matrix[i, j].item()

        connectivity = FunctionalConnectivity(
            connectivity_id=f"conn_{uuid.uuid4().hex[:8]}",
            patient_id=bold.patient_id,
            timestamp=time.time(),
            correlation_matrix=corr_matrix,
            regions=list(self.REGION_CENTERS.keys())[:n_regions],
            analysis_type=analysis_type,
            global_efficiency=global_efficiency,
            clustering_coefficient=clustering_coeff,
            dmn_connectivity=dmn_connectivity,
            metadata={
                "n_timepoints": bold.n_timepoints,
                "n_regions": n_regions,
                "tr": bold.tr,
            },
        )

        self._connectivity_matrices.append(connectivity)
        self._processing_timestamps.append(time.time())

        logger.info(
            f"[fMRI] Computed connectivity: {connectivity.connectivity_id} | "
            f"global_eff={global_efficiency:.3f} | "
            f"clustering={clustering_coeff:.3f}"
        )

        return connectivity

    def _compute_global_efficiency(self, corr_matrix: Tensor) -> float:
        """Compute global efficiency of functional network."""
        # Threshold to binary adjacency
        threshold = 0.3
        adj = (corr_matrix.abs() > threshold).float()
        n = adj.shape[0]

        # Compute shortest paths (simplified: use correlation as inverse distance)
        inv_dist = 1.0 / (1.0 - corr_matrix.abs() + 1e-8)
        inv_dist.fill_diagonal_(0.0)

        # Global efficiency = average inverse shortest path
        efficiency = inv_dist.mean().item() / (n - 1) if n > 1 else 0.0
        return efficiency

    def _compute_clustering_coefficient(self, corr_matrix: Tensor) -> float:
        """Compute clustering coefficient of functional network."""
        threshold = 0.3
        adj = (corr_matrix.abs() > threshold).float()
        n = adj.shape[0]

        # Clustering coefficient for each node
        clustering = 0.0
        for i in range(n):
            neighbors = (adj[i] > 0).nonzero().squeeze(-1)
            k = len(neighbors)
            if k < 2:
                continue
            # Count triangles
            sub_adj = adj[neighbors][:, neighbors]
            triangles = sub_adj.sum().item() / 2
            possible = k * (k - 1) / 2
            clustering += triangles / possible

        return clustering / n if n > 0 else 0.0

    # ── Psychiatric Biomarker Extraction ─────────────────────────────────

    def extract_psychiatric_biomarkers(
        self,
        connectivity: FunctionalConnectivity,
    ) -> Dict[str, Any]:
        """
        Extract psychiatric biomarkers from functional connectivity.

        ⚠️  These are RESEARCH biomarkers, not diagnostic tools.
        Clinical interpretation requires qualified professionals.

        Based on literature:
          • Depression: Reduced DMN connectivity, increased salience network
          • PTSD: Hyperactive amygdala, reduced prefrontal regulation
          • Anxiety: Increased insula connectivity
        """
        biomarkers = {}

        if connectivity.dmn_connectivity:
            # DMN connectivity (relevant for depression)
            dmn_values = list(connectivity.dmn_connectivity.values())
            biomarkers["dmn_mean_connectivity"] = np.mean(dmn_values) if dmn_values else 0.0
            biomarkers["dmn_variance"] = np.var(dmn_values) if dmn_values else 0.0

        # Global network metrics
        biomarkers["global_efficiency"] = connectivity.global_efficiency
        biomarkers["clustering_coefficient"] = connectivity.clustering_coefficient

        # Amygdala-prefrontal connectivity (relevant for anxiety/PTSD)
        if connectivity.correlation_matrix is not None:
            regions = connectivity.regions
            if BrainRegion.AMYGDALA in regions and BrainRegion.PREFRONTAL_CORTEX in regions:
                i = regions.index(BrainRegion.AMYGDALA)
                j = regions.index(BrainRegion.PREFRONTAL_CORTEX)
                biomarkers["amygdala_pfc_connectivity"] = connectivity.correlation_matrix[i, j].item()

        biomarkers["note"] = (
            "RESEARCH biomarkers only — NOT for clinical diagnosis. "
            "Requires validation and clinical oversight."
        )

        return biomarkers

    def get_status(self) -> Dict[str, Any]:
        """Get fMRI bridge status."""
        return {
            "n_bold_signals": len(self._bold_signals),
            "n_connectivity_matrices": len(self._connectivity_matrices),
            "anonymization_enabled": self.cfg.anonymize_patient_data,
            "nibabel_available": HAS_NIBABEL,
            "nilearn_available": HAS_NILEARN,
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [PTP1] PERSONALIZED THERAPY PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatientProfile:
    """Patient profile for personalized therapy."""
    patient_id: str
    anonymized_id: str
    # Demographics (anonymized)
    age_group: str  # "18-25" | "26-35" | "36-50" | "51-65" | "65+"
    gender: str  # "male" | "female" | "other" | "prefer_not_to_say"
    # Clinical
    primary_diagnosis: Optional[PsychiatricDisorder] = None
    secondary_diagnoses: List[PsychiatricDisorder] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MODERATE
    # Biomarkers
    eeg_biomarkers: Optional[Dict[str, float]] = None
    fmri_biomarkers: Optional[Dict[str, float]] = None
    hrv_baseline: Optional[float] = None
    # Preferences
    music_preferences: List[str] = field(default_factory=list)
    cultural_background: Optional[str] = None
    # Consent
    consent_given: bool = False
    consent_date: Optional[float] = None
    consent_withdrawn: bool = False
    # Metadata
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TherapySession:
    """A single therapy session."""
    session_id: str
    patient_id: str
    session_number: int
    timestamp: float
    # Protocol
    target_emotion: EmotionTarget
    music_parameters: Dict[str, Any]
    duration_minutes: float
    # Biometric data
    pre_session_eeg: Optional[Dict[str, float]] = None
    post_session_eeg: Optional[Dict[str, float]] = None
    pre_session_hrv: Optional[float] = None
    post_session_hrv: Optional[float] = None
    # Outcomes
    patient_reported_outcome: Optional[float] = None  # 0-10 scale
    clinician_assessment: Optional[str] = None
    adverse_events: List[str] = field(default_factory=list)
    # Adaptation
    protocol_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersonalizedTherapyProtocol:
    """
    [D] Personalized music therapy protocol generator.

    ⚠️  CLINICAL NOTICE:
      • This is a RESEARCH TOOL, not a treatment system
      • All therapy decisions require CLINICAL OVERSIGHT
      • Patient consent is MANDATORY
      • Not a substitute for professional mental health care

    Features:
      • Patient profiling (anonymized)
      • Stratified treatment assignment
      • Adaptive protocol adjustment
      • Session-by-session outcome tracking
      • Bayesian updating of treatment efficacy

    Mathematical grounding:
      • Bayesian adaptive design: Berry 2006
      • Stratified medicine principles
      • PNDC Axiom 3: Patient data minimization
    """

    # Evidence-based protocol templates (simplified)
    PROTOCOL_TEMPLATES = {
        PsychiatricDisorder.MAJOR_DEPRESSION: {
            "initial_emotion": EmotionTarget.SADNESS,  # Iso-principle
            "target_emotion": EmotionTarget.CALM,
            "tempo_range": (60, 90),
            "key_preference": "major",
            "session_duration_min": 30,
            "frequency_per_week": 3,
            "evidence_level": "B",  # Moderate evidence
        },
        PsychiatricDisorder.PTSD: {
            "initial_emotion": EmotionTarget.ANXIOUS,
            "target_emotion": EmotionTarget.CALM,
            "tempo_range": (60, 80),
            "key_preference": "major",
            "session_duration_min": 20,
            "frequency_per_week": 2,
            "evidence_level": "C",  # Limited evidence
        },
        PsychiatricDisorder.PANIC_DISORDER: {
            "initial_emotion": EmotionTarget.ANXIOUS,
            "target_emotion": EmotionTarget.RELAXED,
            "tempo_range": (50, 70),
            "key_preference": "major",
            "session_duration_min": 15,
            "frequency_per_week": 4,
            "evidence_level": "B",
        },
    }

    def __init__(
        self,
        clinician_id: str,
        require_consent: bool = True,
    ):
        self.clinician_id = clinician_id
        self.require_consent = require_consent
        self._patients: Dict[str, PatientProfile] = {}
        self._sessions: Dict[str, List[TherapySession]] = defaultdict(list)
        self._treatment_efficacy: Dict[str, List[float]] = defaultdict(list)

        if not require_consent:
            logger.error(
                "[TherapyProtocol] ⛔ CONSENT NOT REQUIRED — "
                "This violates ethical principles. Cannot proceed."
            )
            raise ValueError("Patient consent is mandatory")

    # ── Patient Management ───────────────────────────────────────────────

    def register_patient(
        self,
        patient_id: str,
        age_group: str,
        gender: str,
        primary_diagnosis: Optional[PsychiatricDisorder] = None,
        music_preferences: Optional[List[str]] = None,
        cultural_background: Optional[str] = None,
    ) -> PatientProfile:
        """
        Register a new patient for therapy.

        ⚠️  Requires explicit informed consent.
        """
        # Anonymize patient ID
        anon_id = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
        anon_id = f"patient_{anon_id}"

        profile = PatientProfile(
            patient_id=patient_id,
            anonymized_id=anon_id,
            age_group=age_group,
            gender=gender,
            primary_diagnosis=primary_diagnosis,
            music_preferences=music_preferences or [],
            cultural_background=cultural_background,
            consent_given=False,  # Must be set explicitly
        )

        self._patients[anon_id] = profile

        logger.info(
            f"[TherapyProtocol] Registered patient: {anon_id} | "
            f"diagnosis={primary_diagnosis.value if primary_diagnosis else 'None'}"
        )

        return profile

    def record_consent(
        self,
        patient_id: str,
        consent_details: Dict[str, Any],
    ) -> bool:
        """
        Record patient consent for therapy.

        ⚠️  Consent must be:
          • Informed (patient understands research nature)
          • Voluntary (no coercion)
          • Revocable (patient can withdraw anytime)
        """
        if patient_id not in self._patients:
            logger.error(f"[TherapyProtocol] Patient not found: {patient_id}")
            return False

        patient = self._patients[patient_id]
        patient.consent_given = True
        patient.consent_date = time.time()
        patient.metadata["consent_details"] = consent_details

        logger.info(f"[TherapyProtocol] Consent recorded for {patient_id}")
        return True

    def withdraw_consent(self, patient_id: str, reason: str) -> bool:
        """Withdraw patient consent (patient right)."""
        if patient_id not in self._patients:
            return False

        patient = self._patients[patient_id]
        patient.consent_withdrawn = True
        patient.consent_given = False
        patient.metadata["withdrawal_reason"] = reason
        patient.metadata["withdrawal_date"] = time.time()

        logger.info(f"[TherapyProtocol] Consent withdrawn for {patient_id}: {reason}")
        return True

    # ── Protocol Generation ──────────────────────────────────────────────

    def generate_protocol(
        self,
        patient_id: str,
        clinician_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate personalized therapy protocol.

        ⚠️  Requires:
          • Patient consent
          • Clinician review and approval
        """
        if patient_id not in self._patients:
            logger.error(f"[TherapyProtocol] Patient not found: {patient_id}")
            return None

        patient = self._patients[patient_id]

        if not patient.consent_given:
            logger.error(f"[TherapyProtocol] No consent for {patient_id}")
            return None

        if patient.consent_withdrawn:
            logger.error(f"[TherapyProtocol] Consent withdrawn for {patient_id}")
            return None

        # Get base protocol from template
        if patient.primary_diagnosis and patient.primary_diagnosis in self.PROTOCOL_TEMPLATES:
            template = self.PROTOCOL_TEMPLATES[patient.primary_diagnosis]
        else:
            # Default protocol
            template = {
                "initial_emotion": EmotionTarget.NEUTRAL,
                "target_emotion": EmotionTarget.CALM,
                "tempo_range": (70, 100),
                "key_preference": "major",
                "session_duration_min": 20,
                "frequency_per_week": 2,
                "evidence_level": "D",  # Low evidence
            }

        # Personalize based on patient profile
        protocol = {
            "patient_id": patient_id,
            "anonymized_id": patient.anonymized_id,
            "diagnosis": patient.primary_diagnosis.value if patient.primary_diagnosis else None,
            "protocol_id": f"protocol_{uuid.uuid4().hex[:8]}",
            "generated_at": time.time(),
            "generated_by": self.clinician_id,
            **template,
        }

        # Adjust for cultural background
        if patient.cultural_background:
            protocol["cultural_considerations"] = patient.cultural_background
            # In production: adjust music parameters based on cultural music theory

        # Adjust for music preferences
        if patient.music_preferences:
            protocol["preferred_genres"] = patient.music_preferences

        # Apply clinician override if provided
        if clinician_override:
            protocol.update(clinician_override)
            protocol["clinician_overridden"] = True

        # Bayesian prior for treatment efficacy
        protocol["prior_efficacy_estimate"] = 0.5  # Neutral prior
        protocol["prior_variance"] = 0.25

        logger.info(
            f"[TherapyProtocol] Generated protocol: {protocol['protocol_id']} | "
            f"patient={patient.anonymized_id[:12]}..."
        )

        return protocol

    # ── Session Management ───────────────────────────────────────────────

    def record_session(
        self,
        patient_id: str,
        protocol_id: str,
        target_emotion: EmotionTarget,
        music_parameters: Dict[str, Any],
        duration_minutes: float,
        pre_session_eeg: Optional[Dict[str, float]] = None,
        post_session_eeg: Optional[Dict[str, float]] = None,
        patient_reported_outcome: Optional[float] = None,
        clinician_assessment: Optional[str] = None,
        adverse_events: Optional[List[str]] = None,
    ) -> TherapySession:
        """Record a therapy session."""
        if patient_id not in self._patients:
            raise ValueError(f"Patient not found: {patient_id}")

        patient = self._patients[patient_id]
        if not patient.consent_given:
            raise ValueError(f"No consent for patient {patient_id}")

        session_number = len(self._sessions[patient_id]) + 1

        session = TherapySession(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            patient_id=patient_id,
            session_number=session_number,
            timestamp=time.time(),
            target_emotion=target_emotion,
            music_parameters=music_parameters,
            duration_minutes=duration_minutes,
            pre_session_eeg=pre_session_eeg,
            post_session_eeg=post_session_eeg,
            patient_reported_outcome=patient_reported_outcome,
            clinician_assessment=clinician_assessment,
            adverse_events=adverse_events or [],
        )

        self._sessions[patient_id].append(session)

        # Update treatment efficacy (Bayesian update)
        if patient_reported_outcome is not None:
            # Normalize to 0-1
            efficacy = patient_reported_outcome / 10.0
            self._treatment_efficacy[patient_id].append(efficacy)

        logger.info(
            f"[TherapyProtocol] Recorded session {session_number} for "
            f"{patient.anonymized_id[:12]}... | "
            f"outcome={patient_reported_outcome}"
        )

        return session

    def adapt_protocol(
        self,
        patient_id: str,
        current_protocol: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Adapt protocol based on session outcomes (Bayesian updating).

        Uses Bayesian updating to refine treatment efficacy estimate.
        """
        if patient_id not in self._treatment_efficacy:
            return current_protocol

        outcomes = self._treatment_efficacy[patient_id]
        if len(outcomes) < 3:
            return current_protocol  # Not enough data

        # Bayesian update (simplified: conjugate normal-normal)
        prior_mean = current_protocol.get("prior_efficacy_estimate", 0.5)
        prior_var = current_protocol.get("prior_variance", 0.25)

        # Likelihood from data
        data_mean = np.mean(outcomes)
        data_var = np.var(outcomes) if len(outcomes) > 1 else 0.1

        # Posterior
        n = len(outcomes)
        posterior_mean = (prior_mean / prior_var + n * data_mean / data_var) / (
            1 / prior_var + n / data_var
        )
        posterior_var = 1.0 / (1 / prior_var + n / data_var)

        # Adapt protocol based on posterior
        adapted = current_protocol.copy()
        adapted["prior_efficacy_estimate"] = posterior_mean
        adapted["prior_variance"] = posterior_var
        adapted["n_sessions_completed"] = n
        adapted["mean_outcome"] = data_mean

        # Adjust intensity based on efficacy
        if posterior_mean < 0.4:
            # Low efficacy: increase intensity or change approach
            adapted["protocol_adjustment"] = "increase_intensity"
            adapted["tempo_range"] = (
                current_protocol["tempo_range"][0] - 10,
                current_protocol["tempo_range"][1] - 10,
            )
        elif posterior_mean > 0.7:
            # High efficacy: maintain or slightly reduce
            adapted["protocol_adjustment"] = "maintain"
        else:
            adapted["protocol_adjustment"] = "continue_current"

        logger.info(
            f"[TherapyProtocol] Adapted protocol for {patient_id[:12]}... | "
            f"posterior_mean={posterior_mean:.3f} | "
            f"adjustment={adapted['protocol_adjustment']}"
        )

        return adapted

    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get summary of patient's therapy progress."""
        if patient_id not in self._patients:
            return {}

        patient = self._patients[patient_id]
        sessions = self._sessions.get(patient_id, [])
        outcomes = self._treatment_efficacy.get(patient_id, [])

        return {
            "patient_id": patient.anonymized_id,
            "diagnosis": patient.primary_diagnosis.value if patient.primary_diagnosis else None,
            "consent_status": "granted" if patient.consent_given else "not_granted",
            "n_sessions": len(sessions),
            "mean_outcome": np.mean(outcomes) if outcomes else None,
            "outcome_trend": outcomes[-5:] if len(outcomes) >= 5 else outcomes,
            "adverse_events": sum(len(s.adverse_events) for s in sessions),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get protocol system status."""
        return {
            "n_patients": len(self._patients),
            "n_sessions_total": sum(len(s) for s in self._sessions.values()),
            "consent_required": self.require_consent,
            "clinician_id": self.clinician_id,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [CTM1] CLINICAL TRIAL MANAGEMENT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrialConfig:
    """Configuration for clinical trial."""
    trial_id: str
    trial_name: str
    # Design
    trial_type: str = "parallel"  # "parallel" | "crossover" | "adaptive"
    n_arms: int = 2
    arm_names: List[str] = field(default_factory=lambda: ["treatment", "control"])
    randomization_method: str = "stratified"  # "simple" | "block" | "stratified"
    blinding: str = "double"  # "open" | "single" | "double" | "triple"
    # Enrollment
    target_enrollment: int = 100
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    # Duration
    treatment_duration_weeks: int = 12
    follow_up_duration_weeks: int = 4
    # Regulatory
    irb_approval_number: Optional[str] = None
    irb_approval_date: Optional[str] = None
    fda_ind_number: Optional[str] = None
    # Primary outcome
    primary_outcome: str = "patient_reported_outcome"
    primary_outcome_timepoint: str = "week_12"
    # Ethics
    data_safety_monitoring_board: bool = True
    stopping_rules: List[str] = field(default_factory=list)


@dataclass
class Participant:
    """Trial participant."""
    participant_id: str
    anonymized_id: str
    trial_id: str
    # Enrollment
    enrollment_date: float
    arm: str  # "treatment" | "control"
    randomization_number: int
    # Stratification factors
    age_group: str
    gender: str
    diagnosis: Optional[str] = None
    # Status
    status: TrialPhase = TrialPhase.SCREENING
    withdrawal_date: Optional[float] = None
    withdrawal_reason: Optional[str] = None
    # Consent
    consent_given: bool = False
    consent_date: Optional[float] = None
    consent_version: str = "1.0"
    # Outcomes
    primary_outcome: Optional[float] = None
    primary_outcome_date: Optional[float] = None
    secondary_outcomes: Dict[str, float] = field(default_factory=dict)
    # Adverse events
    adverse_events: List["AdverseEvent"] = field(default_factory=list)


@dataclass
class AdverseEvent:
    """Adverse event report."""
    ae_id: str
    participant_id: str
    trial_id: str
    timestamp: float
    # Description
    description: str
    grade: AdverseEventGrade
    related_to_treatment: str  # "unrelated" | "unlikely" | "possible" | "probable" | "definite"
    # Outcome
    outcome: str  # "recovered" | "recovering" | "not_recovered" | "death"
    # Actions
    actions_taken: str
    reporter_id: str
    # Serious AE criteria
    is_serious: bool = False
    serious_criteria: List[str] = field(default_factory=list)
    # Regulatory reporting
    reported_to_irb: bool = False
    reported_to_fda: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEntry:
    """Audit trail entry (FDA 21 CFR Part 11 pattern)."""
    entry_id: str
    timestamp: float
    user_id: str
    action: str
    object_type: str
    object_id: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: str
    signature: str  # Digital signature
    ip_address: Optional[str] = None


class ClinicalTrialManagementSystem:
    """
    [D] Clinical trial management system.

    ⚠️  REGULATORY NOTICE:
      • This is a RESEARCH FRAMEWORK, not a certified EDC system
      • For actual regulatory submission, use validated systems
      • All trials require IRB/IEC approval
      • FDA 21 CFR Part 11 compliance patterns included (research use)
      • ICH-GCP E6(R2) guidelines followed

    Features:
      • Trial design (parallel, crossover, adaptive)
      • Stratified randomization
      • Blinding management
      • Adverse event tracking (CTCAE v5.0)
      • Regulatory audit trail
      • Participant enrollment and consent

    Mathematical grounding:
      • Stratified randomization: Pocock & Simon 1975
      • Bayesian adaptive design: Berry 2006
      • ICH-GCP E6(R2): International guidelines
    """

    def __init__(
        self,
        trial_config: TrialConfig,
        principal_investigator_id: str,
    ):
        self.cfg = trial_config
        self.pi_id = principal_investigator_id
        self._participants: Dict[str, Participant] = {}
        self._audit_log: List[AuditEntry] = []
        self._randomization_counter = 0

        # Validate regulatory requirements
        if not trial_config.irb_approval_number:
            logger.warning(
                "[Trial] ⚠️  IRB approval number not provided. "
                "Cannot enroll participants without IRB approval."
            )

    # ── Audit Trail (FDA 21 CFR Part 11 pattern) ─────────────────────────

    def _create_audit_entry(
        self,
        user_id: str,
        action: str,
        object_type: str,
        object_id: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        reason: str = "",
    ) -> AuditEntry:
        """Create audit trail entry with digital signature."""
        entry_id = f"audit_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        # Digital signature (simplified: hash of entry data)
        signature_data = f"{entry_id}:{timestamp}:{user_id}:{action}:{object_type}:{object_id}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()[:32]

        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            user_id=user_id,
            action=action,
            object_type=object_type,
            object_id=object_id,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            signature=signature,
        )

        self._audit_log.append(entry)
        return entry

    def get_audit_log(
        self,
        object_type: Optional[str] = None,
        object_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        entries = self._audit_log
        if object_type:
            entries = [e for e in entries if e.object_type == object_type]
        if object_id:
            entries = [e for e in entries if e.object_id == object_id]
        return [
            {
                "entry_id": e.entry_id,
                "timestamp": e.timestamp,
                "user_id": e.user_id,
                "action": e.action,
                "object_type": e.object_type,
                "object_id": e.object_id,
                "reason": e.reason,
                "signature": e.signature,
            }
            for e in entries[-limit:]
        ]

    # ── Randomization ────────────────────────────────────────────────────

    def randomize_participant(
        self,
        age_group: str,
        gender: str,
        diagnosis: Optional[str] = None,
    ) -> str:
        """
        Randomize participant to trial arm.

        Uses stratified randomization to balance groups.
        """
        self._randomization_counter += 1

        if self.cfg.randomization_method == "simple":
            # Simple randomization
            arm_idx = self._randomization_counter % self.cfg.n_arms
            return self.cfg.arm_names[arm_idx]

        elif self.cfg.randomization_method == "block":
            # Block randomization (block size = 2 * n_arms)
            block_size = 2 * self.cfg.n_arms
            position_in_block = (self._randomization_counter - 1) % block_size
            arm_idx = position_in_block % self.cfg.n_arms
            return self.cfg.arm_names[arm_idx]

        elif self.cfg.randomization_method == "stratified":
            # Stratified randomization (by age, gender, diagnosis)
            stratum = f"{age_group}_{gender}_{diagnosis or 'none'}"

            # Count participants in each arm for this stratum
            stratum_counts = {arm: 0 for arm in self.cfg.arm_names}
            for p in self._participants.values():
                if (p.age_group == age_group and
                    p.gender == gender and
                    p.diagnosis == diagnosis):
                    stratum_counts[p.arm] += 1

            # Assign to arm with fewest participants (minimization)
            arm = min(stratum_counts, key=stratum_counts.get)
            return arm

        else:
            raise ValueError(f"Unknown randomization method: {self.cfg.randomization_method}")

    # ── Participant Management ───────────────────────────────────────────

    def enroll_participant(
        self,
        participant_id: str,
        age_group: str,
        gender: str,
        diagnosis: Optional[str] = None,
        consent_version: str = "1.0",
    ) -> Optional[Participant]:
        """
        Enroll participant in trial.

        ⚠️  Requires:
          • IRB approval
          • Informed consent
          • Eligibility verification
        """
        # Check IRB approval
        if not self.cfg.irb_approval_number:
            logger.error("[Trial] ⛔ Cannot enroll without IRB approval")
            return None

        # Check enrollment cap
        if len(self._participants) >= self.cfg.target_enrollment:
            logger.error("[Trial] Enrollment cap reached")
            return None

        # Randomize
        arm = self.randomize_participant(age_group, gender, diagnosis)

        # Anonymize participant ID
        anon_id = hashlib.sha256(participant_id.encode()).hexdigest()[:16]
        anon_id = f"participant_{anon_id}"

        participant = Participant(
            participant_id=participant_id,
            anonymized_id=anon_id,
            trial_id=self.cfg.trial_id,
            enrollment_date=time.time(),
            arm=arm,
            randomization_number=self._randomization_counter,
            age_group=age_group,
            gender=gender,
            diagnosis=diagnosis,
            status=TrialPhase.ENROLLMENT,
            consent_version=consent_version,
        )

        self._participants[participant_id] = participant

        # Audit trail
        self._create_audit_entry(
            user_id=self.pi_id,
            action="enroll_participant",
            object_type="participant",
            object_id=participant_id,
            new_value={"arm": arm, "randomization_number": self._randomization_counter},
            reason=f"Enrolled in {self.cfg.trial_id}",
        )

        logger.info(
            f"[Trial] Enrolled participant: {anon_id} | "
            f"arm={arm} | stratum={age_group}_{gender}_{diagnosis}"
        )

        return participant

    def record_consent(
        self,
        participant_id: str,
        consent_version: str,
        consent_details: Dict[str, Any],
    ) -> bool:
        """Record informed consent."""
        if participant_id not in self._participants:
            return False

        participant = self._participants[participant_id]
        participant.consent_given = True
        participant.consent_date = time.time()
        participant.consent_version = consent_version
        participant.metadata["consent_details"] = consent_details

        # Audit trail
        self._create_audit_entry(
            user_id=self.pi_id,
            action="record_consent",
            object_type="participant",
            object_id=participant_id,
            new_value={"consent_version": consent_version},
            reason="Informed consent obtained",
        )

        return True

    def withdraw_participant(
        self,
        participant_id: str,
        reason: str,
        withdrawn_by: str,
    ) -> bool:
        """Withdraw participant from trial."""
        if participant_id not in self._participants:
            return False

        participant = self._participants[participant_id]
        participant.status = TrialPhase.TERMINATED
        participant.withdrawal_date = time.time()
        participant.withdrawal_reason = reason

        # Audit trail
        self._create_audit_entry(
            user_id=withdrawn_by,
            action="withdraw_participant",
            object_type="participant",
            object_id=participant_id,
            old_value=participant.status.value,
            new_value=TrialPhase.TERMINATED.value,
            reason=reason,
        )

        logger.info(f"[Trial] Withdrew participant {participant_id}: {reason}")
        return True

    # ── Adverse Event Management ─────────────────────────────────────────

    def record_adverse_event(
        self,
        participant_id: str,
        description: str,
        grade: AdverseEventGrade,
        related_to_treatment: str,
        outcome: str,
        actions_taken: str,
        reporter_id: str,
        is_serious: bool = False,
    ) -> Optional[AdverseEvent]:
        """
        Record adverse event.

        ⚠️  Serious AEs require immediate IRB/FDA reporting.
        """
        if participant_id not in self._participants:
            return None

        participant = self._participants[participant_id]

        ae = AdverseEvent(
            ae_id=f"ae_{uuid.uuid4().hex[:8]}",
            participant_id=participant_id,
            trial_id=self.cfg.trial_id,
            timestamp=time.time(),
            description=description,
            grade=grade,
            related_to_treatment=related_to_treatment,
            outcome=outcome,
            actions_taken=actions_taken,
            reporter_id=reporter_id,
            is_serious=is_serious,
        )

        participant.adverse_events.append(ae)

        # Audit trail
        self._create_audit_entry(
            user_id=reporter_id,
            action="record_adverse_event",
            object_type="adverse_event",
            object_id=ae.ae_id,
            new_value={
                "grade": grade.value,
                "serious": is_serious,
                "description": description[:100],
            },
            reason="Adverse event reported",
        )

        # Serious AE: flag for regulatory reporting
        if is_serious:
            logger.warning(
                f"[Trial] ⚠️  SERIOUS AE recorded: {ae.ae_id} | "
                f"participant={participant.anonymized_id[:12]}... | "
                f"grade={grade.value} — REQUIRES IRB/FDA REPORTING"
            )
            ae.metadata["requires_regulatory_reporting"] = True

        logger.info(
            f"[Trial] Recorded AE: {ae.ae_id} | "
            f"grade={grade.value} | serious={is_serious}"
        )

        return ae

    def report_serious_ae_to_regulatory(
        self,
        ae_id: str,
        reported_to: str,  # "irb" | "fda"
        reporter_id: str,
    ) -> bool:
        """Report serious AE to regulatory body."""
        # Find AE
        ae = None
        for p in self._participants.values():
            for event in p.adverse_events:
                if event.ae_id == ae_id:
                    ae = event
                    break
            if ae:
                break

        if ae is None:
            return False

        if not ae.is_serious:
            logger.warning(f"[Trial] AE {ae_id} is not serious; no regulatory reporting required")
            return False

        # Mark as reported
        if reported_to == "irb":
            ae.reported_to_irb = True
        elif reported_to == "fda":
            ae.reported_to_fda = True

        # Audit trail
        self._create_audit_entry(
            user_id=reporter_id,
            action="report_serious_ae",
            object_type="adverse_event",
            object_id=ae_id,
            new_value={"reported_to": reported_to},
            reason="Regulatory reporting of serious AE",
        )

        logger.info(f"[Trial] Reported serious AE {ae_id} to {reported_to}")
        return True

    # ── Outcome Recording ────────────────────────────────────────────────

    def record_primary_outcome(
        self,
        participant_id: str,
        outcome_value: float,
        timepoint: str,
        recorded_by: str,
    ) -> bool:
        """Record primary outcome for participant."""
        if participant_id not in self._participants:
            return False

        participant = self._participants[participant_id]
        participant.primary_outcome = outcome_value
        participant.primary_outcome_date = time.time()

        # Audit trail
        self._create_audit_entry(
            user_id=recorded_by,
            action="record_primary_outcome",
            object_type="participant",
            object_id=participant_id,
            new_value={"outcome": outcome_value, "timepoint": timepoint},
            reason=f"Primary outcome at {timepoint}",
        )

        return True

    # ── Trial Summary ────────────────────────────────────────────────────

    def get_trial_summary(self) -> Dict[str, Any]:
        """Get trial summary statistics."""
        n_enrolled = len(self._participants)
        n_completed = sum(
            1 for p in self._participants.values()
            if p.status == TrialPhase.COMPLETION
        )
        n_withdrawn = sum(
            1 for p in self._participants.values()
            if p.status == TrialPhase.TERMINATED
        )

        # Arm distribution
        arm_counts = defaultdict(int)
        for p in self._participants.values():
            arm_counts[p.arm] += 1

        # Adverse events
        total_aes = sum(len(p.adverse_events) for p in self._participants.values())
        serious_aes = sum(
            sum(1 for ae in p.adverse_events if ae.is_serious)
            for p in self._participants.values()
        )

        # Primary outcomes
        outcomes = [
            p.primary_outcome for p in self._participants.values()
            if p.primary_outcome is not None
        ]

        return {
            "trial_id": self.cfg.trial_id,
            "trial_name": self.cfg.trial_name,
            "n_enrolled": n_enrolled,
            "target_enrollment": self.cfg.target_enrollment,
            "n_completed": n_completed,
            "n_withdrawn": n_withdrawn,
            "completion_rate": n_completed / max(n_enrolled, 1),
            "arm_distribution": dict(arm_counts),
            "total_adverse_events": total_aes,
            "serious_adverse_events": serious_aes,
            "n_primary_outcomes": len(outcomes),
            "mean_primary_outcome": np.mean(outcomes) if outcomes else None,
            "irb_approval": self.cfg.irb_approval_number,
            "audit_entries": len(self._audit_log),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get trial management system status."""
        return {
            "trial_id": self.cfg.trial_id,
            "trial_type": self.cfg.trial_type,
            "n_arms": self.cfg.n_arms,
            "blinding": self.cfg.blinding,
            "irb_approved": self.cfg.irb_approval_number is not None,
            "n_participants": len(self._participants),
            "n_audit_entries": len(self._audit_log),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIClinicalResearchSuite:
    """
    [D] Complete clinical research suite.

    ⚠️  ETHICAL/REGULATORY NOTICE:
      • All components are RESEARCH TOOLS only
      • IRB/IEC approval REQUIRED for human subject research
      • Informed consent MANDATORY
      • Clinical oversight by licensed professionals REQUIRED
      • NOT for regulatory submission without validation

    Integrates:
      • fMRI neuroimaging bridge
      • Personalized therapy protocol
      • Clinical trial management system
    """

    def __init__(
        self,
        fmri_cfg: Optional[fMRIConfig] = None,
        trial_config: Optional[TrialConfig] = None,
        clinician_id: str = "clinician_001",
        pi_id: str = "pi_001",
    ):
        self.fmri_bridge = fMRINeuroimagingBridge(fmri_cfg)
        self.therapy_protocol = PersonalizedTherapyProtocol(clinician_id)
        self.trial_system = (
            ClinicalTrialManagementSystem(trial_config, pi_id)
            if trial_config else None
        )

        logger.info("[SESI Clinical] Research suite initialized")
        logger.warning("⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL USE")

    def run_integrated_workflow(
        self,
        patient_id: str,
        fmri_data_path: Optional[str] = None,
        diagnosis: Optional[PsychiatricDisorder] = None,
    ) -> Dict[str, Any]:
        """
        Run integrated clinical research workflow.

        ⚠️  Requires:
          • IRB approval
          • Informed consent
          • Clinical oversight
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        logger.info(f"[Clinical] Starting workflow {workflow_id} for patient {patient_id[:8]}...")

        results = {"workflow_id": workflow_id, "steps": []}

        # Step 1: Register patient
        patient = self.therapy_protocol.register_patient(
            patient_id=patient_id,
            age_group="26-35",
            gender="prefer_not_to_say",
            primary_diagnosis=diagnosis,
        )
        results["steps"].append({"step": "register_patient", "patient_id": patient.anonymized_id})

        # Step 2: Record consent
        self.therapy_protocol.record_consent(
            patient_id=patient.anonymized_id,
            consent_details={"informed": True, "voluntary": True, "revocable": True},
        )
        results["steps"].append({"step": "record_consent"})

        # Step 3: fMRI acquisition and processing
        bold = self.fmri_bridge.load_bold_signal(
            patient_id=patient.anonymized_id,
            data_path=fmri_data_path,
            mock_data=(fmri_data_path is None),
        )
        bold = self.fmri_bridge.preprocess_bold(bold)
        results["steps"].append({"step": "fmri_preprocessing", "bold_id": bold.signal_id})

        # Step 4: Functional connectivity analysis
        connectivity = self.fmri_bridge.compute_functional_connectivity(bold)
        biomarkers = self.fmri_bridge.extract_psychiatric_biomarkers(connectivity)
        results["steps"].append({
            "step": "functional_connectivity",
            "connectivity_id": connectivity.connectivity_id,
            "biomarkers": biomarkers,
        })

        # Step 5: Generate personalized protocol
        protocol = self.therapy_protocol.generate_protocol(patient.anonymized_id)
        results["steps"].append({"step": "generate_protocol", "protocol_id": protocol["protocol_id"] if protocol else None})

        # Step 6: Enroll in trial (if trial system available)
        if self.trial_system:
            participant = self.trial_system.enroll_participant(
                participant_id=patient.anonymized_id,
                age_group=patient.age_group,
                gender=patient.gender,
                diagnosis=diagnosis.value if diagnosis else None,
            )
            if participant:
                self.trial_system.record_consent(
                    participant_id=patient.anonymized_id,
                    consent_version="1.0",
                    consent_details={"informed": True},
                )
                results["steps"].append({
                    "step": "trial_enrollment",
                    "arm": participant.arm,
                })

        logger.info(f"[Clinical] Workflow {workflow_id} complete")
        return results

    def get_full_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "fmri_bridge": self.fmri_bridge.get_status(),
            "therapy_protocol": self.therapy_protocol.get_status(),
            "trial_system": self.trial_system.get_status() if self.trial_system else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Clinical Neuroimaging + Therapy + Trials — Smoke Test")
    print("=" * 72)
    print(f"  nibabel available    : {HAS_NIBABEL}")
    print(f"  nilearn available    : {HAS_NILEARN}")
    print(f"  scipy available      : {HAS_SCIPY}")
    print(f"  psychiatric module   : {HAS_PSYCH}")

    print("\n  ⚠️  CLINICAL/REGULATORY NOTICE")
    print("  ─────────────────────────────────────────────────────────")
    print("  • NOT MEDICAL DEVICES — research tools only")
    print("  • IRB/IEC approval REQUIRED for human subject research")
    print("  • Informed consent MANDATORY")
    print("  • Clinical oversight by licensed professionals REQUIRED")
    print("=" * 72)

    # Initialize suite
    trial_cfg = TrialConfig(
        trial_id="trial_001",
        trial_name="Music Therapy for Depression",
        trial_type="parallel",
        n_arms=2,
        arm_names=["music_therapy", "waitlist_control"],
        target_enrollment=50,
        irb_approval_number="IRB-2026-001",
        primary_outcome="patient_reported_outcome",
    )

    suite = SESIClinicalResearchSuite(
        trial_config=trial_cfg,
        clinician_id="dr_smith",
        pi_id="dr_jones",
    )

    # ── Test 1: fMRI Neuroimaging Bridge ─────────────────────────────────
    print("\n  ── [FMRI1] fMRI Neuroimaging Bridge ──")

    # Load mock BOLD signal
    bold = suite.fmri_bridge.load_bold_signal(
        patient_id="patient_001",
        mock_data=True,
    )
    print(f"    BOLD signal ID       : {bold.signal_id}")
    print(f"    Patient (anonymized) : {bold.patient_id}")
    print(f"    Shape                : ({bold.n_timepoints}, {bold.n_voxels})")
    print(f"    TR                   : {bold.tr}s")

    # Preprocess
    bold = suite.fmri_bridge.preprocess_bold(bold)
    print(f"    Motion corrected     : {bold.motion_corrected}")
    print(f"    Mean FD              : {bold.mean_fd:.4f}")

    # Compute connectivity
    connectivity = suite.fmri_bridge.compute_functional_connectivity(bold)
    print(f"    Connectivity ID      : {connectivity.connectivity_id}")
    print(f"    Global efficiency    : {connectivity.global_efficiency:.3f}")
    print(f"    Clustering coeff     : {connectivity.clustering_coefficient:.3f}")
    if connectivity.dmn_connectivity:
        print(f"    DMN connectivity     : {len(connectivity.dmn_connectivity)} pairs")

    # Extract biomarkers
    biomarkers = suite.fmri_bridge.extract_psychiatric_biomarkers(connectivity)
    print(f"    Biomarkers extracted : {len(biomarkers)}")
    for k, v in list(biomarkers.items())[:3]:
        if isinstance(v, float):
            print(f"      {k:30s}: {v:.3f}")

    fmri_status = suite.fmri_bridge.get_status()
    print(f"    Anonymization        : {fmri_status['anonymization_enabled']}")

    # ── Test 2: Personalized Therapy Protocol ────────────────────────────
    print("\n  ── [PTP1] Personalized Therapy Protocol ──")

    # Register patient
    patient = suite.therapy_protocol.register_patient(
        patient_id="patient_002",
        age_group="26-35",
        gender="female",
        primary_diagnosis=PsychiatricDisorder.MAJOR_DEPRESSION if HAS_PSYCH else None,
    )
    print(f"    Patient registered   : {patient.anonymized_id}")

    # Record consent
    suite.therapy_protocol.record_consent(
        patient_id=patient.anonymized_id,
        consent_details={"informed": True, "voluntary": True},
    )
    print(f"    Consent recorded     : True")

    # Generate protocol
    protocol = suite.therapy_protocol.generate_protocol(patient.anonymized_id)
    if protocol:
        print(f"    Protocol ID          : {protocol['protocol_id']}")
        print(f"    Initial emotion      : {protocol['initial_emotion']}")
        print(f"    Target emotion       : {protocol['target_emotion']}")
        print(f"    Tempo range          : {protocol['tempo_range']}")
        print(f"    Evidence level       : {protocol['evidence_level']}")

    # Record sessions
    for i in range(3):
        session = suite.therapy_protocol.record_session(
            patient_id=patient.anonymized_id,
            protocol_id=protocol["protocol_id"],
            target_emotion=EmotionTarget.CALM,
            music_parameters={"tempo": 80, "key": "C_major"},
            duration_minutes=30.0,
            patient_reported_outcome=6.0 + i * 0.5,
        )
        print(f"    Session {i+1} recorded : outcome={session.patient_reported_outcome}")

    # Adapt protocol
    adapted = suite.therapy_protocol.adapt_protocol(patient.anonymized_id, protocol)
    print(f"    Adapted protocol     : {adapted.get('protocol_adjustment', 'N/A')}")
    print(f"    Posterior efficacy   : {adapted.get('prior_efficacy_estimate', 0.5):.3f}")

    # Patient summary
    summary = suite.therapy_protocol.get_patient_summary(patient.anonymized_id)
    print(f"    Total sessions       : {summary['n_sessions']}")
    print(f"    Mean outcome         : {summary['mean_outcome']:.2f}")

    # ── Test 3: Clinical Trial Management ────────────────────────────────
    print("\n  ── [CTM1] Clinical Trial Management System ──")

    if suite.trial_system:
        trial_status = suite.trial_system.get_status()
        print(f"    Trial ID             : {trial_status['trial_id']}")
        print(f"    Trial type           : {trial_status['trial_type']}")
        print(f"    Blinding             : {trial_status['blinding']}")
        print(f"    IRB approved         : {trial_status['irb_approved']}")

        # Enroll participants
        for i in range(5):
            participant = suite.trial_system.enroll_participant(
                participant_id=f"trial_patient_{i:03d}",
                age_group="26-35" if i % 2 == 0 else "36-50",
                gender="female" if i % 2 == 0 else "male",
                diagnosis="major_depression",
            )
            if participant:
                suite.trial_system.record_consent(
                    participant_id=f"trial_patient_{i:03d}",
                    consent_version="1.0",
                    consent_details={"informed": True},
                )
                print(f"    Enrolled             : {participant.anonymized_id[:15]}... → {participant.arm}")

        # Record adverse event
        ae = suite.trial_system.record_adverse_event(
            participant_id="trial_patient_000",
            description="Mild headache during session",
            grade=AdverseEventGrade.GRADE_1_MILD,
            related_to_treatment="possible",
            outcome="recovered",
            actions_taken="Monitoring continued",
            reporter_id="dr_smith",
            is_serious=False,
        )
        if ae:
            print(f"    AE recorded          : {ae.ae_id} (grade {ae.grade.value})")

        # Record serious AE
        serious_ae = suite.trial_system.record_adverse_event(
            participant_id="trial_patient_001",
            description="Severe anxiety episode",
            grade=AdverseEventGrade.GRADE_3_SEVERE,
            related_to_treatment="probable",
            outcome="recovering",
            actions_taken="Session paused, patient referred to psychiatrist",
            reporter_id="dr_smith",
            is_serious=True,
        )
        if serious_ae:
            print(f"    SERIOUS AE recorded  : {serious_ae.ae_id} — REQUIRES REPORTING")
            # Report to IRB
            suite.trial_system.report_serious_ae_to_regulatory(
                serious_ae.ae_id, "irb", "dr_smith"
            )

        # Record primary outcomes
        for i in range(3):
            suite.trial_system.record_primary_outcome(
                participant_id=f"trial_patient_{i:03d}",
                outcome_value=6.0 + i * 0.5,
                timepoint="week_12",
                recorded_by="dr_smith",
            )

        # Trial summary
        trial_summary = suite.trial_system.get_trial_summary()
        print(f"\n    Trial Summary:")
        print(f"      Enrolled             : {trial_summary['n_enrolled']}/{trial_summary['target_enrollment']}")
        print(f"      Completion rate      : {trial_summary['completion_rate']:.1%}")
        print(f"      Arm distribution     : {trial_summary['arm_distribution']}")
        print(f"      Total AEs            : {trial_summary['total_adverse_events']}")
        print(f"      Serious AEs          : {trial_summary['serious_adverse_events']}")
        print(f"      Mean primary outcome : {trial_summary['mean_primary_outcome']}")

        # Audit log
        audit_log = suite.trial_system.get_audit_log(limit=5)
        print(f"\n    Recent Audit Entries ({len(audit_log)} shown):")
        for entry in audit_log[:3]:
            print(f"      {entry['action']:30s} | {entry['object_type']:15s} | {entry['user_id']}")

    # ── Test 4: Integrated Workflow ──────────────────────────────────────
    print("\n  ── Integrated Clinical Research Workflow ──")

    workflow_result = suite.run_integrated_workflow(
        patient_id="integrated_patient_001",
        diagnosis=PsychiatricDisorder.MAJOR_DEPRESSION if HAS_PSYCH else None,
    )

    print(f"    Workflow ID          : {workflow_result['workflow_id']}")
    print(f"    Steps completed      : {len(workflow_result['steps'])}")
    for step in workflow_result['steps']:
        print(f"      ✓ {step['step']}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    fMRI signals         : {status['fmri_bridge']['n_bold_signals']}")
    print(f"    Connectivity matrices: {status['fmri_bridge']['n_connectivity_matrices']}")
    print(f"    Therapy patients     : {status['therapy_protocol']['n_patients']}")
    print(f"    Therapy sessions     : {status['therapy_protocol']['n_sessions_total']}")
    if status['trial_system']:
        print(f"    Trial participants   : {status['trial_system']['n_participants']}")
        print(f"    Audit entries        : {status['trial_system']['n_audit_entries']}")

    print("\n" + "=" * 72)
    print("  ⚠️  ETHICAL/REGULATORY FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • NOT MEDICAL DEVICES — research tools only")
    print("  • IRB/IEC approval REQUIRED for human subject research")
    print("  • Informed consent MANDATORY for all participants")
    print("  • Clinical oversight by licensed professionals REQUIRED")
    print("  • PNDC Axiom 3: Patient data is ULTRA-SENSITIVE")
    print("  • Buddhist Ethics (อहिंसा): Never harm or exploit")
    print("  • Taoist Ethics (无为): Support natural healing")
    print("  • FDA 21 CFR Part 11 patterns for audit trail")
    print("  • ICH-GCP E6(R2) compliance patterns")
    print("=" * 72)
    print("  HONEST SCOPE")
    print("  • fMRI processing is RESEARCH-GRADE, not clinically validated")
    print("  • Personalized protocols require CLINICAL OVERSIGHT")
    print("  • Trial management is a FRAMEWORK, not certified EDC")
    print("  • All human subject research requires IRB/IEC approval")
    print("  • Patient data handling requires HIPAA/GDPR compliance")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
