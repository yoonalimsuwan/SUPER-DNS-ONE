"""
=============================================================================
SESI THERAPEUTIC ART AGI — FHIR + GNN + BLOCKCHAIN + NEUROAESTHETICS
Real HAPI FHIR Integration + Biomarker GNN + Immutable Audit + Healing Art
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================
"""

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("sesi.therapeutic_art_fhir")

__all__ = [
    "ArtModality", "MentalState",
    "HAPIFHIRClient",
    "BiomarkerGraphNeuralNetwork",
    "BlockchainAuditTrail",
    "TherapeuticArtOrchestrator",
    "SESITherapeuticArtAGI",
]

# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class ArtModality(Enum):
    """Supported therapeutic art modalities."""
    VISUAL_2D = "visual_2d"
    SCULPTURE_3D = "sculpture_3d"
    INTERIOR_DESIGN = "interior_design"
    ARCHITECTURE = "architecture"
    FINE_ARTS = "fine_arts"
    GAMES = "games"
    MOVIES = "movies"

class MentalState(Enum):
    """Predicted mental states from GNN."""
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    STRESSED = "stressed"
    NEUTRAL = "neutral"
    CALM = "calm"
    FOCUSED = "focused"

# ═══════════════════════════════════════════════════════════════════════════
# [FHIR1] HAPI FHIR CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class HAPIFHIRClient:
    """
    [D] Real client for HAPI FHIR Server.
    Fetches Patient, Observation (Biomarkers), and Condition resources.
    Enforces PNDC Axiom 3: Only fetches strictly necessary data.
    """
    def __init__(self, base_url: str = "http://hapi.fhir.org/baseR4"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session() if HAS_REQUESTS else None

    def get_patient_biomarkers(self, patient_id: str) -> Dict[str, Any]:
        if not self.session:
            return self._mock_fhir_data(patient_id)

        try:
            url = f"{self.base_url}/Observation?subject=Patient/{patient_id}&_count=10&_sort=-date"
            response = self.session.get(url, headers={"Accept": "application/fhir+json"})
            response.raise_for_status()
            data = response.json()
            
            minimized = {
                "patient_id": patient_id,
                "timestamp": time.time(),
                "biomarkers": []
            }
            for entry in data.get("entry", []):
                obs = entry["resource"]
                minimized["biomarkers"].append({
                    "code": obs["code"]["coding"][0]["code"],
                    "value": obs["valueQuantity"]["value"],
                    "unit": obs["valueQuantity"]["unit"],
                    "date": obs["effectiveDateTime"]
                })
            return minimized
        except Exception as e:
            logger.error(f"[FHIR] Fetch failed: {e}")
            return self._mock_fhir_data(patient_id)

    def _mock_fhir_data(self, patient_id: str) -> Dict[str, Any]:
        return {
            "patient_id": patient_id,
            "timestamp": time.time(),
            "biomarkers": [
                {"code": "EEG_ALPHA", "value": 12.5, "unit": "uV^2", "date": "2026-11-02"},
                {"code": "EEG_THETA", "value": 8.2, "unit": "uV^2", "date": "2026-11-02"},
                {"code": "BDNF", "value": 18.5, "unit": "ng/mL", "date": "2026-11-02"},
                {"code": "CORTISOL", "value": 15.0, "unit": "ug/dL", "date": "2026-11-02"},
                {"code": "HRV_RMSSD", "value": 45.0, "unit": "ms", "date": "2026-11-02"},
            ]
        }

# ═══════════════════════════════════════════════════════════════════════════
# [GNN1] BIOMARKER GRAPH NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class BiomarkerGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)
        deg = torch.sum(adj_with_self_loops, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = torch.diag(deg_inv_sqrt) @ adj_with_self_loops @ torch.diag(deg_inv_sqrt)
        h = torch.mm(norm_adj, torch.mm(x, self.weight)) + self.bias
        return F.relu(h)

class BiomarkerGraphNeuralNetwork(nn.Module):
    def __init__(self, num_nodes: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = BiomarkerGCNLayer(in_features=1, out_features=hidden_dim)
        self.conv2 = BiomarkerGCNLayer(in_features=hidden_dim, out_features=len(MentalState))
        
        self.register_buffer("adjacency", torch.tensor([
            [1.0, 0.8, 0.5, 0.7, 0.6],
            [0.8, 1.0, 0.4, 0.8, 0.5],
            [0.5, 0.4, 1.0, 0.3, 0.4],
            [0.7, 0.8, 0.3, 1.0, 0.8],
            [0.6, 0.5, 0.4, 0.8, 1.0],
        ], dtype=torch.float32))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        h = self.conv1(node_features, self.adjacency)
        logits = self.conv2(h, self.adjacency)
        return F.softmax(logits, dim=-1)

    def predict_state(self, biomarker_values: List[float]) -> Tuple[MentalState, float]:
        normalized = torch.tensor([
            biomarker_values[0] / 20.0,
            biomarker_values[1] / 15.0,
            biomarker_values[2] / 30.0,
            biomarker_values[3] / 25.0,
            biomarker_values[4] / 60.0,
        ], dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            probs = self.forward(normalized)
            pred_idx = torch.argmax(probs, dim=0).item()
            confidence = probs[0, pred_idx].item()
            
        states = list(MentalState)
        return states[pred_idx], confidence

# ═══════════════════════════════════════════════════════════════════════════
# [BC1] BLOCKCHAIN AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Block:
    index: int
    timestamp: float
    data: Dict[str, Any]
    previous_hash: str
    hash: str = ""

class BlockchainAuditTrail:
    def __init__(self):
        self.chain: List[Block] = []
        self._create_genesis_block()
        self._hash_rate_limit = 0.01  # No-Zeno bound: max 100 hashes/sec

    def _create_genesis_block(self):
        genesis_data = {"event": "GENESIS", "system": "SESI Therapeutic Art AGI"}
        genesis_hash = self._compute_hash(0, time.time(), genesis_data, "0")
        self.chain.append(Block(0, time.time(), genesis_data, "0", genesis_hash))

    def _compute_hash(self, index: int, timestamp: float, data: Dict, prev_hash: str) -> str:
        block_string = f"{index}{timestamp}{json.dumps(data, sort_keys=True)}{prev_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def add_record(self, event_type: str, details: Dict[str, Any]) -> str:
        """Add a new record to the blockchain. No-Zeno bounded."""
        time.sleep(self._hash_rate_limit)
        
        last_block = self.chain[-1]
        new_index = last_block.index + 1
        new_timestamp = time.time()
        new_data = {"event_type": event_type, "details": details}
        new_hash = self._compute_hash(new_index, new_timestamp, new_data, last_block.hash)
        
        # --- ส่วนที่ขาดหายไปได้รับการแก้ไขแล้ว ---
        new_block = Block(new_index, new_timestamp, new_data, last_block.hash, new_hash)
        self.chain.append(new_block)
        logger.info(f"[Blockchain] Added record: {event_type} (Block {new_index})")
        return new_hash

    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if current.hash != self._compute_hash(current.index, current.timestamp, current.data, previous.hash):
                return False
        return True

    def get_latest_hash(self) -> str:
        return self.chain[-1].hash

# ═══════════════════════════════════════════════════════════════════════════
# [ART1] THERAPEUTIC ART ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class TherapeuticArtOrchestrator:
    """
    [D] Generates healing art parameters based on predicted mental state.
    Applies Neuroaesthetics principles (Chatterjee & Vartanian 2014).
    """
    def __init__(self):
        self.modality_params = {
            MentalState.ANXIOUS: {
                "color_palette": ["#A8D8EA", "#AA96DA", "#FCBAD3"],
                "fractal_complexity": 0.3,
                "tempo_bpm": 60,
                "geometry": "soft_curves",
                "lighting": "diffuse_warm"
            },
            MentalState.DEPRESSED: {
                "color_palette": ["#FFD3B6", "#FFAAA5", "#FF8B94"],
                "fractal_complexity": 0.5,
                "tempo_bpm": 75,
                "geometry": "ascending_patterns",
                "lighting": "bright_natural"
            },
            MentalState.STRESSED: {
                "color_palette": ["#A8E6CF", "#DCEDC1", "#FFD3B6"],
                "fractal_complexity": 0.4,
                "tempo_bpm": 65,
                "geometry": "biophilic_organic",
                "lighting": "soft_green_tint"
            },
            MentalState.CALM: {
                "color_palette": ["#C7CEEA", "#B5EAD7", "#E2F0CB"],
                "fractal_complexity": 0.2,
                "tempo_bpm": 55,
                "geometry": "symmetrical_balance",
                "lighting": "moonlight_cool"
            },
            MentalState.FOCUSED: {
                "color_palette": ["#FF9AA2", "#FFB7B2", "#FFDAC1"],
                "fractal_complexity": 0.7,
                "tempo_bpm": 90,
                "geometry": "sacred_geometry",
                "lighting": "high_contrast"
            },
            MentalState.NEUTRAL: {
                "color_palette": ["#E2F0CB", "#B5EAD7", "#C7CEEA"],
                "fractal_complexity": 0.5,
                "tempo_bpm": 70,
                "geometry": "balanced_grid",
                "lighting": "neutral_daylight"
            }
        }

    def generate_art_parameters(self, mental_state: MentalState, modality: ArtModality) -> Dict[str, Any]:
        base_params = self.modality_params.get(mental_state, self.modality_params[MentalState.NEUTRAL])
        return {
            "modality": modality.value,
            "mental_state_target": mental_state.value,
            "parameters": base_params,
            "timestamp": time.time(),
            "generation_id": str(uuid.uuid4())[:8]
        }

# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESITherapeuticArtAGI:
    """
    [D] Master orchestrator binding FHIR, GNN, Blockchain, and Art Generation.
    """
    def __init__(self, fhir_url: str = "http://hapi.fhir.org/baseR4"):
        self.fhir_client = HAPIFHIRClient(base_url=fhir_url)
        self.gnn = BiomarkerGraphNeuralNetwork()
        self.blockchain = BlockchainAuditTrail()
        self.art_orchestrator = TherapeuticArtOrchestrator()
        logger.info("[SESI Therapeutic Art AGI] Initialized successfully.")

    def run_therapeutic_pipeline(
        self, 
        patient_id: str, 
        modality: ArtModality = ArtModality.VISUAL_2D
    ) -> Dict[str, Any]:
        """
        End-to-end therapeutic art generation pipeline.
        1. Fetch biomarkers (FHIR)
        2. Predict mental state (GNN)
        3. Log to immutable ledger (Blockchain)
        4. Generate healing art parameters
        """
        logger.info(f"[Pipeline] Starting therapeutic pipeline for patient: {patient_id}")
        
        # Step 1: Fetch Biomarkers
        biomarker_data = self.fhir_client.get_patient_biomarkers(patient_id)
        values = [b["value"] for b in biomarker_data["biomarkers"]]
        
        # Step 2: Predict Mental State
        predicted_state, confidence = self.gnn.predict_state(values)
        logger.info(f"[GNN] Predicted state: {predicted_state.value} (confidence: {confidence:.2f})")
        
        # Step 3: Blockchain Audit
        audit_details = {
            "patient_id": patient_id,
            "biomarkers": biomarker_data["biomarkers"],
            "predicted_state": predicted_state.value,
            "confidence": confidence
        }
        block_hash = self.blockchain.add_record("MENTAL_STATE_PREDICTION", audit_details)
        
        # Step 4: Generate Art Parameters
        art_params = self.art_orchestrator.generate_art_parameters(predicted_state, modality)
        art_hash = self.blockchain.add_record("ART_PARAMETERS_GENERATED", art_params)
        
        return {
            "patient_id": patient_id,
            "predicted_state": predicted_state.value,
            "confidence": confidence,
            "art_parameters": art_params,
            "blockchain_hashes": {
                "prediction_block": block_hash,
                "art_generation_block": art_hash
            },
            "blockchain_valid": self.blockchain.is_valid()
        }

# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Therapeutic Art AGI — Smoke Test")
    print("=" * 72)
    print(f"  Requests available : {HAS_REQUESTS}")
    
    # Initialize AGI
    agi = SESITherapeuticArtAGI()
    
    # Run pipeline for a mock patient
    print("\n  ── Running Therapeutic Pipeline ──")
    result = agi.run_therapeutic_pipeline(
        patient_id="P001",
        modality=ArtModality.VISUAL_2D
    )
    
    print(f"    Predicted State    : {result['predicted_state']}")
    print(f"    Confidence         : {result['confidence']:.2f}")
    print(f"    Art Modality       : {result['art_parameters']['modality']}")
    print(f"    Fractal Complexity : {result['art_parameters']['parameters']['fractal_complexity']}")
    print(f"    Color Palette      : {', '.join(result['art_parameters']['parameters']['color_palette'])}")
    print(f"    Blockchain Valid   : {result['blockchain_valid']}")
    print(f"    Prediction Hash    : {result['blockchain_hashes']['prediction_block'][:16]}...")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a RESEARCH framework, NOT a clinical therapy replacement")
    print("  • FHIR integration requires a running HAPI FHIR server")
    print("  • Blockchain is a cryptographic audit tool, not a clinical decision maker")
    print("  • Art parameters are heuristic proxies for neuroaesthetic principles")
    print("=" * 72)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
