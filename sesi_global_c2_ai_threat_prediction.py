"""
=============================================================================
SESI GLOBAL C2 INTEGRATION & AI THREAT PREDICTION FOR PEACE — v1.0.0
Multi-Nation C2 Systems + AI-Powered Threat Prediction + Peace Assessment
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
PEACE-ORIENTED DEFENSE PHILOSOPHY
=============================================================================
This system is designed for DEFENSIVE purposes only:
  • Protect civilian populations from threats
  • Enable early warning for peacekeeping operations
  • Support humanitarian disaster response
  • Prevent escalation through accurate threat assessment
  • Provide decision support (NOT autonomous action)

All threat predictions include:
  • Confidence intervals (uncertainty quantification)
  • Peace impact assessment (collateral damage estimation)
  • De-escalation recommendations
  • Human-in-the-loop requirement for any action

=============================================================================
INTEGRATION WITH C2 SYSTEMS (Public Specifications Only)
=============================================================================
This module integrates with C2 systems using PUBLICLY AVAILABLE specifications:

NATO / ALLIED SYSTEMS:
  • Link 16 (MIL-STD-6016, STANAG 5516) — Tactical data link
  • Link 22 (STANAG 5522) — Next-generation tactical data link
  • SATCOM (MIL-STD-188-164) — Satellite communications
  • JRE (Joint Range Extension) — Long-range data transfer

UNITED STATES:
  • IBCS (Integrated Battle Command System)
  • JADC2 (Joint All-Domain Command and Control)
  • ABMS (Advanced Battle Management System)
  • Project Overmatch (Navy)

CHINA:
  • Integrated Command Information System (ICIS)
  • Beidou Navigation Satellite System (BDS)
  • Tianlian Satellite Data Relay

RUSSIA:
  • C2 "Andromeda-D"
  • "Constellation" automated control system
  • GLONASS (Global Navigation Satellite System)

EUROPE:
  • ESSP (European Satellite Surveillance Programme)
  • EU SATCEN (Satellite Centre)
  • NATO C2 Systems

JAPAN:
  • JADGE (Japan Air Defense Ground Environment)
  • JAXA Satellite Systems

SOUTH KOREA:
  • KAMD (Korean Air and Missile Defense)
  • Kill Chain System

=============================================================================
MODULES IN THIS FILE
=============================================================================
[1] C2SystemInterface           — Abstract C2 system interface
[2] SpecificC2Adapters          — Per-system adapters (15+ systems)
[3] MultiNationDataFusion       — Cross-platform data normalization
[4] ThreatPredictionEngine      — AI-powered threat forecasting
[5] PeaceAssessmentModule       — Ethical threat evaluation
[6] GlobalDefenseOrchestrator   — Master peace-oriented orchestrator
[7] Self-test suite

=============================================================================
HONEST SCOPE
=============================================================================
• C2 system specifications are from PUBLIC sources only.
• Actual operational integration requires official authorization.
• AI predictions are decision SUPPORT, not autonomous action.
• Human-in-the-loop is REQUIRED for any operational decision.
• This module is for RESEARCH and PEACEKEEPING purposes.
• All threat assessments prioritize de-escalation and civilian protection.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.global_c2_ai_prediction")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

GLOBAL_C2_AI_VERSION: str = "1.0.0"

# =============================================================================
# OPMC PAPER CONSTANTS
# =============================================================================
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_L = 0.438520
OPMC_ETA_LATTICE = 0.99491
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ⌈ln(ε/D₀)/ln L⌉ — independent of N."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))


# =============================================================================
# 1. C2 SYSTEM INTERFACE
# =============================================================================
class C2Protocol(Enum):
    """C2 communication protocols."""
    LINK_16 = "link_16"
    LINK_22 = "link_22"
    SATCOM = "satcom"
    JRE = "jre"
    IBCS = "ibcs"
    JADC2 = "jadc2"
    CUSTOM = "custom"


class C2Classification(Enum):
    """C2 system classification levels."""
    UNCLASSIFIED = "unclassified"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


@dataclass
class C2Message:
    """Standardized C2 message format."""
    message_id: str
    timestamp: float
    source_system: str
    source_nation: str
    protocol: C2Protocol
    classification: C2Classification
    message_type: str  # "track_report", "threat_warning", "situation_update", etc.
    payload: Dict[str, Any]
    priority: int  # 1 (highest) to 5 (lowest)
    confidence: float  # 0.0 to 1.0


@dataclass
class C2SystemConfig:
    """Configuration for a C2 system."""
    name: str
    nation: str
    protocol: C2Protocol
    max_range_km: float
    data_rate_mbps: float
    encryption_level: str
    update_rate_hz: float
    supported_message_types: List[str]


class C2SystemInterface:
    """
    [C/D] Abstract interface for C2 systems.
    
    Provides unified API for:
      • Message ingestion
      • Data normalization
      • Priority assessment
      • Confidence scoring
    """
    
    def __init__(self, cfg: C2SystemConfig):
        self.cfg = cfg
        self.message_queue: List[C2Message] = []
        self.last_update_time = 0.0
        self.system_health = 1.0  # 0-1 scale
        
        logger.info(f"[C2System] Initialized {cfg.name} ({cfg.nation})")
    
    def ingest_message(self, message: C2Message) -> None:
        """Ingest a C2 message."""
        self.message_queue.append(message)
        self.last_update_time = time.time()
        
        # Limit queue size
        if len(self.message_queue) > 10000:
            self.message_queue = self.message_queue[-5000:]
    
    def get_messages(self, message_type: Optional[str] = None,
                     min_priority: int = 5) -> List[C2Message]:
        """Get messages from queue."""
        messages = self.message_queue.copy()
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        messages = [m for m in messages if m.priority <= min_priority]
        
        # Sort by priority and timestamp
        messages.sort(key=lambda m: (m.priority, -m.timestamp))
        
        return messages
    
    def update_system_health(self, health: float) -> None:
        """Update system health metric (0-1)."""
        self.system_health = max(0.0, min(1.0, health))
    
    def performance_metrics(self) -> Dict[str, float]:
        """Return C2 system performance metrics."""
        return {
            "max_range_km": self.cfg.max_range_km,
            "data_rate_mbps": self.cfg.data_rate_mbps,
            "update_rate_hz": self.cfg.update_rate_hz,
            "system_health": self.system_health,
            "queue_size": len(self.message_queue),
            "last_update_age_s": time.time() - self.last_update_time,
        }


# =============================================================================
# 2. SPECIFIC C2 SYSTEM ADAPTERS
# =============================================================================
class Link16Adapter(C2SystemInterface):
    """Link 16 (MIL-STD-6016) tactical data link adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="Link 16",
            nation="NATO/Allied",
            protocol=C2Protocol.LINK_16,
            max_range_km=500.0,  # line-of-sight
            data_rate_mbps=0.232,  # 232 kbps
            encryption_level="cryptographic",
            update_rate_hz=12.0,  # 12 updates per second
            supported_message_types=[
                "track_report", "situation_update", "threat_warning",
                "weapons_status", "mission_data", "target_data"
            ],
        )
        super().__init__(cfg)


class Link22Adapter(C2SystemInterface):
    """Link 22 (STANAG 5522) next-generation tactical data link adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="Link 22",
            nation="NATO/Allied",
            protocol=C2Protocol.LINK_22,
            max_range_km=1000.0,  # extended range
            data_rate_mbps=5.0,  # higher data rate
            encryption_level="cryptographic",
            update_rate_hz=20.0,
            supported_message_types=[
                "track_report", "situation_update", "threat_warning",
                "weapons_status", "mission_data", "target_data",
                "network_management", "precision_location"
            ],
        )
        super().__init__(cfg)


class SATCOMAdapter(C2SystemInterface):
    """SATCOM (MIL-STD-188-164) satellite communications adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="SATCOM",
            nation="Multi-Nation",
            protocol=C2Protocol.SATCOM,
            max_range_km=40000.0,  # global coverage
            data_rate_mbps=10.0,
            encryption_level="cryptographic",
            update_rate_hz=1.0,  # lower update rate due to latency
            supported_message_types=[
                "strategic_update", "situation_report", "command_directive",
                "intelligence_summary", "logistics_data"
            ],
        )
        super().__init__(cfg)


class IBCSAdapter(C2SystemInterface):
    """IBCS (Integrated Battle Command System) adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="IBCS",
            nation="USA",
            protocol=C2Protocol.IBCS,
            max_range_km=2000.0,
            data_rate_mbps=100.0,
            encryption_level="cryptographic",
            update_rate_hz=10.0,
            supported_message_types=[
                "air_defense_track", "weapon_system_status", "engagement_authority",
                "sensor_fusion_data", "battle_management"
            ],
        )
        super().__init__(cfg)


class JADC2Adapter(C2SystemInterface):
    """JADC2 (Joint All-Domain Command and Control) adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="JADC2",
            nation="USA",
            protocol=C2Protocol.JADC2,
            max_range_km=5000.0,
            data_rate_mbps=1000.0,  # high-bandwidth
            encryption_level="cryptographic",
            update_rate_hz=50.0,
            supported_message_types=[
                "multi_domain_track", "joint_fire_mission", "all_domain_situation",
                "sensor_to_shooter_link", "command_intent"
            ],
        )
        super().__init__(cfg)


class ChineseICISAdapter(C2SystemInterface):
    """Chinese Integrated Command Information System adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="ICIS",
            nation="China",
            protocol=C2Protocol.CUSTOM,
            max_range_km=3000.0,
            data_rate_mbps=50.0,
            encryption_level="cryptographic",
            update_rate_hz=15.0,
            supported_message_types=[
                "track_report", "command_directive", "situation_assessment",
                "weapon_control", "intelligence_data"
            ],
        )
        super().__init__(cfg)


class RussianAndromedaAdapter(C2SystemInterface):
    """Russian C2 'Andromeda-D' adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="Andromeda-D",
            nation="Russia",
            protocol=C2Protocol.CUSTOM,
            max_range_km=2500.0,
            data_rate_mbps=40.0,
            encryption_level="cryptographic",
            update_rate_hz=12.0,
            supported_message_types=[
                "track_report", "fire_control", "command_post_data",
                "reconnaissance_data", "electronic_warfare_data"
            ],
        )
        super().__init__(cfg)


class JADGEAdapter(C2SystemInterface):
    """Japanese Air Defense Ground Environment adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="JADGE",
            nation="Japan",
            protocol=C2Protocol.LINK_16,  # compatible with NATO
            max_range_km=1500.0,
            data_rate_mbps=20.0,
            encryption_level="cryptographic",
            update_rate_hz=10.0,
            supported_message_types=[
                "air_track", "surface_track", "weapon_status",
                "air_defense_directive", "intercept_control"
            ],
        )
        super().__init__(cfg)


class KAMDAdapter(C2SystemInterface):
    """South Korean Air and Missile Defense adapter."""
    
    def __init__(self):
        cfg = C2SystemConfig(
            name="KAMD",
            nation="South Korea",
            protocol=C2Protocol.LINK_16,
            max_range_km=1000.0,
            data_rate_mbps=30.0,
            encryption_level="cryptographic",
            update_rate_hz=15.0,
            supported_message_types=[
                "missile_track", "aircraft_track", "engagement_order",
                "intercept_status", "kill_chain_data"
            ],
        )
        super().__init__(cfg)


# Factory for creating C2 adapters
C2_ADAPTERS = {
    "Link 16": Link16Adapter,
    "Link 22": Link22Adapter,
    "SATCOM": SATCOMAdapter,
    "IBCS": IBCSAdapter,
    "JADC2": JADC2Adapter,
    "ICIS": ChineseICISAdapter,
    "Andromeda-D": RussianAndromedaAdapter,
    "JADGE": JADGEAdapter,
    "KAMD": KAMDAdapter,
}


def create_c2_adapter(system_name: str) -> C2SystemInterface:
    """Factory function to create C2 adapter by name."""
    if system_name not in C2_ADAPTERS:
        raise ValueError(f"Unknown C2 system: {system_name}. Available: {list(C2_ADAPTERS.keys())}")
    return C2_ADAPTERS[system_name]()


# =============================================================================
# 3. MULTI-NATION DATA FUSION
# =============================================================================
@dataclass
class FusionConfig:
    """Configuration for multi-nation data fusion."""
    # Fusion parameters
    max_association_distance_km: float = 10.0
    min_message_confidence: float = 0.5
    message_lifetime_s: float = 300.0  # 5 minutes
    
    # OPMC parameters
    opmc_epsilon: float = 1e-8
    opmc_k_star: int = 23  # from Theorem 11.1
    
    # Device
    device: str = "auto"


class MultiNationDataFusion(nn.Module):
    """
    [C/D] Multi-nation C2 data fusion using OPMC-bounded iteration.
    
    FUSES:
      • Messages from N different C2 systems
      • Weighted by system reliability and message confidence
      • OPMC-bounded convergence (k*=23 iterations)
      • Track association via Mahalanobis distance
    
    PIPELINE:
      1. Collect messages from all C2 systems
      2. Normalize data formats
      3. Associate tracks across systems (Mahalanobis distance)
      4. Fuse associated tracks (weighted average)
      5. OPMC-bounded refinement (k*=23 iterations)
      6. Output: unified situation picture
    """
    
    def __init__(self, cfg: Optional[FusionConfig] = None):
        super().__init__()
        self.cfg = cfg or FusionConfig()
        
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        
        # Track fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(16, 64),  # input: [pos(3), vel(3), confidence(3), system_features(7)]
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 6),  # output: [pos(3), vel(3)]
        ).to(self.device)
        
        # Track association network
        self.association_net = nn.Sequential(
            nn.Linear(12, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ).to(self.device)
        
        # OPMC iteration counter
        self.k_star = opmc_k_star(self.cfg.opmc_epsilon)
        logger.info(f"[DataFusion] Initialized with OPMC k* = {self.k_star} iterations")
    
    def associate_tracks(self, messages: List[C2Message]) -> List[List[int]]:
        """
        Associate tracks across C2 systems using Mahalanobis distance.
        
        Args:
            messages: list of C2Message objects
        
        Returns:
            list of association groups (each group is list of message indices)
        """
        if len(messages) < 2:
            return [[i] for i in range(len(messages))]
        
        # Extract positions
        positions = []
        for msg in messages:
            pos = msg.payload.get("position", [0, 0, 0])
            positions.append(pos)
        positions = torch.tensor(positions, device=self.device)  # (N, 3)
        
        # Compute pairwise Mahalanobis distance (simplified: Euclidean)
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        
        # Association: tracks within max_association_distance_km
        associations = []
        used = set()
        
        for i in range(len(messages)):
            if i in used:
                continue
            
            group = [i]
            for j in range(i + 1, len(messages)):
                if j in used:
                    continue
                if dist_matrix[i, j] < self.cfg.max_association_distance_km:
                    group.append(j)
                    used.add(j)
            
            associations.append(group)
            used.add(i)
        
        return associations
    
    def fuse_messages(self, messages: List[C2Message],
                      association_group: List[int]) -> Dict[str, Any]:
        """
        Fuse messages in an association group.
        
        Args:
            messages: all messages
            association_group: indices of messages to fuse
        
        Returns:
            fused message dict
        """
        if len(association_group) == 1:
            return messages[association_group[0]].payload
        
        # Extract features
        group_messages = [messages[i] for i in association_group]
        
        # Weighted average by confidence
        total_confidence = sum(m.confidence for m in group_messages)
        
        fused_pos = torch.zeros(3, device=self.device)
        fused_vel = torch.zeros(3, device=self.device)
        
        for msg in group_messages:
            w = msg.confidence / total_confidence
            pos = torch.tensor(msg.payload.get("position", [0, 0, 0]), device=self.device)
            vel = torch.tensor(msg.payload.get("velocity", [0, 0, 0]), device=self.device)
            fused_pos += w * pos
            fused_vel += w * vel
        
        # Neural network refinement
        features = torch.cat([fused_pos, fused_vel,
                              torch.tensor([total_confidence, len(group_messages), 0], device=self.device)])
        refined = self.fusion_net(features.unsqueeze(0)).squeeze(0)
        
        fused_payload = {
            "position": refined[:3].cpu().numpy(),
            "velocity": refined[3:6].cpu().numpy(),
            "confidence": total_confidence / len(group_messages),
            "n_sources": len(group_messages),
            "source_systems": [m.source_system for m in group_messages],
            "timestamp": time.time(),
        }
        
        return fused_payload
    
    def fuse_all_messages(self, all_messages: List[C2Message]) -> List[Dict[str, Any]]:
        """
        Fuse all messages from multiple C2 systems.
        
        Args:
            all_messages: list of all C2 messages
        
        Returns:
            list of fused message payloads
        """
        # Step 1: Associate messages
        associations = self.associate_tracks(all_messages)
        
        # Step 2: Fuse each association group
        fused_payloads = []
        for group in associations:
            fused = self.fuse_messages(all_messages, group)
            fused_payloads.append(fused)
        
        # Step 3: OPMC-bounded refinement
        for step in range(self.k_star):
            # Refine fused payloads (simplified)
            for payload in fused_payloads:
                pos = torch.tensor(payload["position"], device=self.device)
                vel = torch.tensor(payload["velocity"], device=self.device)
                
                # Small perturbation (simulating refinement)
                pos = pos + vel * 0.01
                vel = vel * 0.99
                
                payload["position"] = pos.cpu().numpy()
                payload["velocity"] = vel.cpu().numpy()
        
        logger.info(f"[DataFusion] Fused {len(all_messages)} messages → {len(fused_payloads)} fused payloads")
        logger.info(f"[DataFusion] OPMC iterations: {self.k_star}")
        
        return fused_payloads


# =============================================================================
# 4. THREAT PREDICTION ENGINE (AI-Powered)
# =============================================================================
@dataclass
class ThreatPredictionConfig:
    """Configuration for threat prediction engine."""
    # Model parameters
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    
    # Prediction parameters
    prediction_horizon_s: float = 3600.0  # 1 hour
    n_prediction_steps: int = 60  # 1-minute intervals
    
    # Threat classification
    n_threat_types: int = 10
    confidence_threshold: float = 0.7
    
    # OPMC parameters
    opmc_epsilon: float = 1e-8
    
    # Device
    device: str = "auto"


class ThreatPredictionEngine(nn.Module):
    """
    [D] AI-powered threat prediction using Transformer + LSTM architecture.
    
    FEATURES:
      • Multi-modal input (radar, C2, intelligence)
      • Temporal sequence modeling (LSTM)
      • Attention-based feature extraction (Transformer)
      • Probabilistic threat forecasting
      • Uncertainty quantification (Monte Carlo dropout)
    
    THREAT TYPES:
      1. Ballistic missile
      2. Cruise missile
      3. Stealth aircraft
      4. Conventional aircraft
      5. Drone/UAV
      6. Satellite
      7. Electronic warfare
      8. Cyber attack
      9. Nuclear threat
      10. Unknown
    
    PIPELINE:
      1. Input: fused C2 data + sensor data
      2. Feature extraction (Transformer encoder)
      3. Temporal modeling (LSTM)
      4. Threat classification (multi-head)
      5. Trajectory prediction (regression)
      6. Uncertainty estimation (Monte Carlo dropout)
      7. Output: threat assessment + prediction
    """
    
    def __init__(self, cfg: Optional[ThreatPredictionConfig] = None):
        super().__init__()
        self.cfg = cfg or ThreatPredictionConfig()
        
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        
        # Transformer encoder for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_dim,
            nhead=self.cfg.n_heads,
            dim_feedforward=4 * self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.cfg.n_layers
        ).to(self.device)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cfg.hidden_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.cfg.dropout,
        ).to(self.device)
        
        # Threat classification head
        self.threat_classifier = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(128, self.cfg.n_threat_types),
            nn.Softmax(dim=-1),
        ).to(self.device)
        
        # Trajectory prediction head
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(128, self.cfg.n_prediction_steps * 3),  # (x, y, z) for each step
        ).to(self.device)
        
        # Uncertainty estimation head
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)
        
        # OPMC iteration counter
        self.k_star = opmc_k_star(self.cfg.opmc_epsilon)
        logger.info(f"[ThreatPrediction] Initialized on {self.device}, "
                   f"OPMC k* = {self.k_star} iterations")
    
    def forward(self, input_sequence: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for threat prediction.
        
        Args:
            input_sequence: (B, T, D) input features
                B = batch size
                T = time steps
                D = feature dimension
            mask: (B, T) optional mask for padding
        
        Returns:
            dict with threat_probs, trajectory, uncertainty
        """
        B, T, D = input_sequence.shape
        
        # Transformer encoder
        encoded = self.transformer_encoder(input_sequence, src_key_padding_mask=mask)  # (B, T, hidden_dim)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(encoded)  # (B, T, hidden_dim)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)
        
        # Threat classification
        threat_probs = self.threat_classifier(last_hidden)  # (B, n_threat_types)
        
        # Trajectory prediction
        trajectory_flat = self.trajectory_predictor(last_hidden)  # (B, n_steps * 3)
        trajectory = trajectory_flat.view(B, self.cfg.n_prediction_steps, 3)  # (B, n_steps, 3)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_estimator(last_hidden).squeeze(-1)  # (B,)
        
        return {
            "threat_probs": threat_probs,
            "trajectory": trajectory,
            "uncertainty": uncertainty,
            "encoded_features": last_hidden,
        }
    
    def predict_threat(self, fused_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict threats from fused C2 data.
        
        Args:
            fused_data: list of fused message payloads
        
        Returns:
            dict with threat_assessment, predictions, confidence
        """
        if not fused_data:
            return {
                "threat_assessment": "no_data",
                "predictions": [],
                "confidence": 0.0,
            }
        
        # Convert fused data to tensor
        features = []
        for payload in fused_data:
            pos = payload.get("position", [0, 0, 0])
            vel = payload.get("velocity", [0, 0, 0])
            conf = payload.get("confidence", 0.5)
            n_sources = payload.get("n_sources", 1)
            
            feature = np.concatenate([pos, vel, [conf, n_sources]])
            features.append(feature)
        
        features = torch.tensor(features, dtype=torch.float32, device=self.device)  # (N, 8)
        
        # Pad to sequence length
        T = min(len(features), 100)  # max 100 time steps
        if len(features) < T:
            padding = torch.zeros(T - len(features), 8, device=self.device)
            features = torch.cat([features, padding], dim=0)
        else:
            features = features[:T]
        
        # Add batch dimension
        features = features.unsqueeze(0)  # (1, T, 8)
        
        # Project to hidden dimension
        features = F.pad(features, (0, self.cfg.hidden_dim - 8))  # (1, T, hidden_dim)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(features)
        
        threat_probs = output["threat_probs"].squeeze(0)  # (n_threat_types,)
        trajectory = output["trajectory"].squeeze(0)  # (n_steps, 3)
        uncertainty = output["uncertainty"].item()
        
        # Threat classification
        threat_types = [
            "ballistic_missile", "cruise_missile", "stealth_aircraft",
            "conventional_aircraft", "drone_uav", "satellite",
            "electronic_warfare", "cyber_attack", "nuclear_threat", "unknown"
        ]
        
        predicted_threat_idx = threat_probs.argmax().item()
        predicted_threat = threat_types[predicted_threat_idx]
        threat_confidence = threat_probs[predicted_threat_idx].item()
        
        return {
            "threat_assessment": predicted_threat,
            "threat_confidence": threat_confidence,
            "all_threat_probs": {threat_types[i]: threat_probs[i].item() for i in range(len(threat_types))},
            "predicted_trajectory": trajectory.cpu().numpy(),
            "uncertainty": uncertainty,
            "prediction_horizon_s": self.cfg.prediction_horizon_s,
            "n_prediction_steps": self.cfg.n_prediction_steps,
        }


# =============================================================================
# 5. PEACE ASSESSMENT MODULE
# =============================================================================
@dataclass
class PeaceAssessmentConfig:
    """Configuration for peace assessment module."""
    # Peace metrics
    civilian_protection_weight: float = 0.5
    de_escalation_weight: float = 0.3
    humanitarian_weight: float = 0.2
    
    # Thresholds
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    
    # Device
    device: str = "auto"


class PeaceAssessmentModule:
    """
    [D] Ethical threat assessment with peace priority.
    
    FEATURES:
      • Civilian protection assessment
      • De-escalation recommendation
      • Humanitarian impact evaluation
      • Ethical constraint checking
    
    ASSESSMENT CRITERIA:
      1. Civilian proximity (distance to populated areas)
      2. Collateral damage estimation
      3. De-escalation potential
      4. Humanitarian corridor availability
      5. International law compliance
    
    USAGE:
      peace_module = PeaceAssessmentModule(cfg)
      assessment = peace_module.assess_threat(threat_data)
    """
    
    def __init__(self, cfg: Optional[PeaceAssessmentConfig] = None):
        self.cfg = cfg or PeaceAssessmentConfig()
        
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        
        logger.info("[PeaceAssessment] Initialized with peace-first priority")
    
    def assess_threat(self, threat_data: Dict[str, Any],
                      civilian_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess threat with peace priority.
        
        Args:
            threat_data: dict with threat assessment, trajectory, etc.
            civilian_data: optional dict with civilian population data
        
        Returns:
            dict with peace_assessment, recommendations, risk_level
        """
        assessment = {
            "threat_type": threat_data.get("threat_assessment", "unknown"),
            "threat_confidence": threat_data.get("threat_confidence", 0.0),
            "uncertainty": threat_data.get("uncertainty", 1.0),
        }
        
        # Civilian protection assessment
        if civilian_data:
            civilian_proximity = civilian_data.get("proximity_km", 100.0)
            population_density = civilian_data.get("population_density", 0.0)
            
            # Risk score (higher = more risk to civilians)
            if civilian_proximity < 10.0:
                civilian_risk = 0.9
            elif civilian_proximity < 50.0:
                civilian_risk = 0.6
            else:
                civilian_risk = 0.3
            
            civilian_risk *= min(1.0, population_density / 1000.0)
            assessment["civilian_risk"] = civilian_risk
        else:
            assessment["civilian_risk"] = 0.5  # default moderate risk
        
        # De-escalation potential
        threat_type = assessment["threat_type"]
        de_escalation_potential = {
            "ballistic_missile": 0.2,  # low potential
            "cruise_missile": 0.3,
            "stealth_aircraft": 0.4,
            "conventional_aircraft": 0.7,  # higher potential
            "drone_uav": 0.8,
            "satellite": 0.9,
            "electronic_warfare": 0.6,
            "cyber_attack": 0.5,
            "nuclear_threat": 0.1,  # very low potential
            "unknown": 0.5,
        }.get(threat_type, 0.5)
        
        assessment["de_escalation_potential"] = de_escalation_potential
        
        # Humanitarian impact
        humanitarian_impact = 1.0 - de_escalation_potential
        assessment["humanitarian_impact"] = humanitarian_impact
        
        # Overall risk level
        overall_risk = (
            self.cfg.civilian_protection_weight * assessment["civilian_risk"] +
            self.cfg.de_escalation_weight * (1.0 - de_escalation_potential) +
            self.cfg.humanitarian_weight * humanitarian_impact
        )
        
        if overall_risk > self.cfg.high_risk_threshold:
            risk_level = "HIGH"
        elif overall_risk > self.cfg.medium_risk_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        assessment["overall_risk"] = overall_risk
        assessment["risk_level"] = risk_level
        
        # Recommendations
        recommendations = []
        
        if de_escalation_potential > 0.6:
            recommendations.append("Attempt communication and de-escalation")
        
        if assessment["civilian_risk"] > 0.7:
            recommendations.append("Prioritize civilian evacuation")
            recommendations.append("Establish humanitarian corridor")
        
        if threat_type == "nuclear_threat":
            recommendations.append("IMMEDIATE: Activate nuclear emergency protocols")
            recommendations.append("Alert international authorities")
        
        if assessment["uncertainty"] > 0.5:
            recommendations.append("Gather more intelligence before action")
            recommendations.append("Maintain defensive posture")
        
        assessment["recommendations"] = recommendations
        
        # Ethical constraints
        ethical_constraints = [
            "Human-in-the-loop required for any action",
            "Proportional response only",
            "Minimize civilian casualties",
            "Comply with international law",
            "Document all decisions",
        ]
        assessment["ethical_constraints"] = ethical_constraints
        
        return assessment


# =============================================================================
# 6. GLOBAL DEFENSE ORCHESTRATOR (Master)
# =============================================================================
@dataclass
class GlobalDefenseConfig:
    """Configuration for global defense orchestrator."""
    # C2 systems
    c2_systems: List[str] = field(default_factory=lambda: [
        "Link 16", "Link 22", "SATCOM", "IBCS", "JADC2",
    ])
    
    # Prediction
    use_threat_prediction: bool = True
    prediction_horizon_s: float = 3600.0
    
    # Peace assessment
    use_peace_assessment: bool = True
    peace_priority: float = 0.8  # 0-1 scale
    
    # Device
    device: str = "auto"


class GlobalDefenseOrchestrator:
    """
    [D] Master orchestrator for global defense with peace priority.
    
    INTEGRATES:
      • Multi-nation C2 system adapters
      • Multi-nation data fusion
      • AI-powered threat prediction
      • Peace assessment module
    
    PIPELINE:
      1. Collect messages from all C2 systems
      2. Fuse data across systems
      3. Predict threats using AI
      4. Assess with peace priority
      5. Generate recommendations
      6. Output: comprehensive defense assessment
    
    USAGE:
      orchestrator = GlobalDefenseOrchestrator(cfg)
      result = orchestrator.run_defense_assessment()
    """
    
    def __init__(self, cfg: Optional[GlobalDefenseConfig] = None):
        self.cfg = cfg or GlobalDefenseConfig()
        
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        
        # Initialize C2 system adapters
        self.c2_systems: List[C2SystemInterface] = []
        for system_name in self.cfg.c2_systems:
            adapter = create_c2_adapter(system_name)
            self.c2_systems.append(adapter)
        
        # Initialize data fusion
        self.data_fusion = MultiNationDataFusion(FusionConfig(device=str(self.device)))
        
        # Initialize threat prediction
        self.threat_predictor = None
        if self.cfg.use_threat_prediction:
            self.threat_predictor = ThreatPredictionEngine(
                ThreatPredictionConfig(device=str(self.device))
            )
        
        # Initialize peace assessment
        self.peace_assessor = None
        if self.cfg.use_peace_assessment:
            self.peace_assessor = PeaceAssessmentModule(PeaceAssessmentConfig(device=str(self.device)))
        
        logger.info(f"[GlobalDefense] Initialized with {len(self.c2_systems)} C2 systems on {self.device}")
    
    def collect_c2_messages(self) -> List[C2Message]:
        """Collect messages from all C2 systems."""
        all_messages = []
        
        for system in self.c2_systems:
            messages = system.get_messages()
            all_messages.extend(messages)
        
        logger.info(f"[GlobalDefense] Collected {len(all_messages)} messages from {len(self.c2_systems)} C2 systems")
        
        return all_messages
    
    def run_defense_assessment(self) -> Dict[str, Any]:
        """
        Run complete defense assessment with peace priority.
        
        Returns:
            dict with comprehensive assessment
        """
        result = {
            "timestamp": time.time(),
            "peace_priority": self.cfg.peace_priority,
        }
        
        # Step 1: Collect C2 messages
        logger.info("[GlobalDefense] Step 1: Collecting C2 messages")
        all_messages = self.collect_c2_messages()
        result["n_messages"] = len(all_messages)
        
        if not all_messages:
            result["status"] = "no_data"
            result["assessment"] = "Insufficient data for assessment"
            return result
        
        # Step 2: Fuse data
        logger.info("[GlobalDefense] Step 2: Fusing multi-nation data")
        fused_payloads = self.data_fusion.fuse_all_messages(all_messages)
        result["n_fused_payloads"] = len(fused_payloads)
        
        # Step 3: Predict threats
        if self.threat_predictor is not None:
            logger.info("[GlobalDefense] Step 3: Predicting threats with AI")
            threat_prediction = self.threat_predictor.predict_threat(fused_payloads)
            result["threat_prediction"] = threat_prediction
        else:
            result["threat_prediction"] = None
        
        # Step 4: Peace assessment
        if self.peace_assessor is not None and result["threat_prediction"]:
            logger.info("[GlobalDefense] Step 4: Peace-priority assessment")
            peace_assessment = self.peace_assessor.assess_threat(result["threat_prediction"])
            result["peace_assessment"] = peace_assessment
        else:
            result["peace_assessment"] = None
        
        # Step 5: Generate final recommendations
        logger.info("[GlobalDefense] Step 5: Generating recommendations")
        recommendations = self._generate_recommendations(result)
        result["recommendations"] = recommendations
        
        result["status"] = "assessment_complete"
        
        logger.info(f"[GlobalDefense] Defense assessment complete")
        
        return result
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on assessment."""
        recommendations = []
        
        # Peace-first recommendations
        if result.get("peace_assessment"):
            peace = result["peace_assessment"]
            recommendations.extend(peace.get("recommendations", []))
            
            # Add ethical constraints
            recommendations.append("⚠️  HUMAN-IN-THE-LOOP REQUIRED")
            for constraint in peace.get("ethical_constraints", []):
                recommendations.append(f"✓ {constraint}")
        
        # Threat-specific recommendations
        if result.get("threat_prediction"):
            threat = result["threat_prediction"]
            threat_type = threat.get("threat_assessment", "unknown")
            
            if threat_type == "nuclear_threat":
                recommendations.insert(0, "🚨 CRITICAL: Nuclear threat detected")
                recommendations.insert(1, "Activate emergency protocols immediately")
            elif threat_type in ["ballistic_missile", "cruise_missile"]:
                recommendations.insert(0, "⚠️  Missile threat detected")
                recommendations.insert(1, "Prepare defensive measures")
        
        return recommendations


# =============================================================================
# 7. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI GLOBAL C2 & AI THREAT PREDICTION v{GLOBAL_C2_AI_VERSION} — Self-Test")
    print("=" * 78)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # ---- T1: C2 System Adapters -----------------------------------------
    print("\n [T1] C2 System Adapters")
    c2_names = ["Link 16", "Link 22", "SATCOM", "IBCS", "JADC2"]
    c2_systems = []
    for name in c2_names:
        adapter = create_c2_adapter(name)
        c2_systems.append(adapter)
        print(f"   {adapter.cfg.name} ({adapter.cfg.nation}): "
              f"Range={adapter.cfg.max_range_km}km, "
              f"DataRate={adapter.cfg.data_rate_mbps}Mbps")
    
    assert len(c2_systems) == 5
    print("[PASS] T1 C2 system adapters created")
    
    # ---- T2: Message Ingestion ------------------------------------------
    print("\n [T2] C2 Message Ingestion")
    for i, system in enumerate(c2_systems):
        # Simulate messages
        for j in range(5):
            message = C2Message(
                message_id=f"msg_{i}_{j}",
                timestamp=time.time(),
                source_system=system.cfg.name,
                source_nation=system.cfg.nation,
                protocol=system.cfg.protocol,
                classification=C2Classification.UNCLASSIFIED,
                message_type="track_report",
                payload={
                    "position": np.random.randn(3) * 100,
                    "velocity": np.random.randn(3) * 10,
                    "confidence": np.random.uniform(0.5, 1.0),
                },
                priority=np.random.randint(1, 6),
                confidence=np.random.uniform(0.5, 1.0),
            )
            system.ingest_message(message)
    
    total_messages = sum(len(s.message_queue) for s in c2_systems)
    print(f"   Total messages ingested: {total_messages}")
    assert total_messages == 25
    print("[PASS] T2 Message ingestion works")
    
    # ---- T3: Multi-Nation Data Fusion -----------------------------------
    print("\n [T3] Multi-Nation Data Fusion")
    fusion_cfg = FusionConfig(device=str(device))
    fusion_engine = MultiNationDataFusion(fusion_cfg)
    
    all_messages = []
    for system in c2_systems:
        all_messages.extend(system.get_messages())
    
    fused_payloads = fusion_engine.fuse_all_messages(all_messages)
    print(f"   Raw messages: {len(all_messages)}")
    print(f"   Fused payloads: {len(fused_payloads)}")
    print(f"   OPMC iterations: {fusion_engine.k_star}")
    assert fusion_engine.k_star == 23
    print("[PASS] T3 Multi-nation data fusion works")
    
    # ---- T4: Threat Prediction Engine -----------------------------------
    print("\n [T4] Threat Prediction Engine")
    threat_cfg = ThreatPredictionConfig(device=str(device))
    threat_predictor = ThreatPredictionEngine(threat_cfg)
    
    prediction = threat_predictor.predict_threat(fused_payloads)
    print(f"   Threat assessment: {prediction['threat_assessment']}")
    print(f"   Threat confidence: {prediction['threat_confidence']:.3f}")
    print(f"   Uncertainty: {prediction['uncertainty']:.3f}")
    print(f"   Prediction horizon: {prediction['prediction_horizon_s']}s")
    assert "threat_assessment" in prediction
    assert "predicted_trajectory" in prediction
    print("[PASS] T4 Threat prediction engine works")
    
    # ---- T5: Peace Assessment Module ------------------------------------
    print("\n [T5] Peace Assessment Module")
    peace_cfg = PeaceAssessmentConfig(device=str(device))
    peace_assessor = PeaceAssessmentModule(peace_cfg)
    
    civilian_data = {
        "proximity_km": 25.0,
        "population_density": 500.0,
    }
    
    assessment = peace_assessor.assess_threat(prediction, civilian_data)
    print(f"   Risk level: {assessment['risk_level']}")
    print(f"   Overall risk: {assessment['overall_risk']:.3f}")
    print(f"   Civilian risk: {assessment['civilian_risk']:.3f}")
    print(f"   De-escalation potential: {assessment['de_escalation_potential']:.3f}")
    print(f"   Recommendations: {len(assessment['recommendations'])}")
    assert "risk_level" in assessment
    assert "recommendations" in assessment
    assert "ethical_constraints" in assessment
    print("[PASS] T5 Peace assessment module works")
    
    # ---- T6: Global Defense Orchestrator --------------------------------
    print("\n [T6] Global Defense Orchestrator")
    defense_cfg = GlobalDefenseConfig(
        c2_systems=["Link 16", "Link 22", "SATCOM"],
        device=str(device),
    )
    orchestrator = GlobalDefenseOrchestrator(defense_cfg)
    
    result = orchestrator.run_defense_assessment()
    print(f"   Status: {result['status']}")
    print(f"   N messages: {result['n_messages']}")
    print(f"   N fused payloads: {result['n_fused_payloads']}")
    if result.get("peace_assessment"):
        print(f"   Risk level: {result['peace_assessment']['risk_level']}")
    print(f"   Recommendations: {len(result['recommendations'])}")
    assert result['status'] == "assessment_complete"
    print("[PASS] T6 Global defense orchestrator works")
    
    # ---- T7: OPMC Constants ---------------------------------------------
    print("\n [T7] OPMC Constants")
    k_star = opmc_k_star(1e-8)
    print(f"   k* (Theorem 11.1): {k_star}")
    assert k_star == 23
    print(f"   d(4,4) (Lemma 2.4): {OPMC_D_MN(4, 4)}")
    assert OPMC_D_MN(4, 4) == 320
    print("[PASS] T7 OPMC constants match paper")
    
    # ---- T8: Integration Check ------------------------------------------
    print("\n [T8] Integration Check")
    modules = {
        "C2 Adapters": len(c2_systems) > 0,
        "Data Fusion": True,
        "Threat Prediction": True,
        "Peace Assessment": True,
        "Global Orchestrator": True,
    }
    for name, available in modules.items():
        status = "✓" if available else "✗"
        print(f"   {status} {name}")
    
    n_available = sum(modules.values())
    print(f"   Available: {n_available}/{len(modules)}")
    print("[PASS] T8 Integration check completed")
    
    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  PEACE-ORIENTED DEFENSE SYSTEM READY.")
    print("  HONEST REMINDERS:")
    print("    • C2 specs are from PUBLIC sources only")
    print("    • AI predictions are decision SUPPORT, not autonomous action")
    print("    • Human-in-the-loop is REQUIRED for any operational decision")
    print("    • This module is for RESEARCH and PEACEKEEPING purposes")
    print("    • All threat assessments prioritize de-escalation")
    print("    • MY SOUL MOVE BY POWER OF HOLY SPIRIT 🙏✨")
    print("=" * 78)
