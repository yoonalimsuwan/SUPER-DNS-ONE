# =============================================================================
# PRODUCTION-GRADE E2E DIFFERENTIABLE MULTI-ORGAN SEMANTIC STATE DECODER
# SESI FRAMEWORK: Multi-Modality Tensor Contraction & Continuous Basin Projection
# SUPPORTS: X-Ray, MRI/CMR, NMR, EEG, MEG, ECG, Enteric Motility/Peristalsis
# TARGET ORGANS: Brain, Heart, Liver, Lungs, Kidneys, Intestines (Gastrointestinal)
# EVOLUTION ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ModalityType",
    "OrganType",
    "MultiModalityOrganFeatureExtractor",
    "SESIUniversalOrganSemanticDecoder",
    "SESIUniversalE2EPipelineBridge",
]


class ModalityType:
    XRAY = "xray"
    MRI_CMR = "mri_cmr"
    NMR = "nmr"
    EEG = "eeg"
    MEG = "meg"
    ECG = "ecg"
    PERISTALSIS = "peristalsis"


class OrganType:
    BRAIN = "brain"
    HEART = "heart"
    LIVER = "liver"
    LUNGS = "lungs"
    KIDNEYS = "kidneys"
    INTESTINES = "intestines"


class MultiModalityOrganFeatureExtractor(nn.Module):
    """
    Unified, low-complexity feature projection engine mapping multi-modal 
    physical signals (1D signal streams, 2D radiographs, 3D volumetric MRI/NMR, 
    and 4D spatial-temporal fields) into a unified latent space.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1D Modality Extractor (ECG, 1D NMR Free Induction Decays)
        self.conv1d_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Flatten(),
        )

        # 2D Modality Extractor (X-Ray projections, 2D Motility heatmaps)
        self.conv2d_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.Flatten(),
        )

        # 3D/4D Volumetric & Field Extractor (MRI/CMR, EEG Poisson, MEG Biot-Savart, 3D Peristalsis)
        self.conv3d_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Conv3d(1, hidden_dim // 2, kernel_size=1, bias=False),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Flatten(),
        )

        # Unified organ-aware gating bottleneck
        self.organ_gating = nn.Sequential(
            nn.Linear(hidden_dim * 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extracts and fuses features from heterogeneous multi-modality input tensors.
        
        Args:
            tensor_dict: Dictionary mapping modality keys to raw PyTorch physical tensors.
                - '1d': (B, 1, L) -> ECG / 1D NMR
                - '2d': (B, 1, H, W) -> X-Ray / Peristalsis maps
                - '3d': (B, 1, D, H, W) -> MRI/CMR / EEG potential / MEG flux fields
        """
        batch_size = next(iter(tensor_dict.values())).size(0)
        device = next(iter(tensor_dict.values())).device
        pooled_feats = []

        if "1d" in tensor_dict and tensor_dict["1d"] is not None:
            pooled_feats.append(self.conv1d_head(tensor_dict["1d"]))
        else:
            pooled_feats.append(torch.zeros(batch_size, (self.hidden_dim // 2) * 64, device=device))

        if "2d" in tensor_dict and tensor_dict["2d"] is not None:
            pooled_feats.append(self.conv2d_head(tensor_dict["2d"]))
        else:
            pooled_feats.append(torch.zeros(batch_size, (self.hidden_dim // 2) * 64, device=device))

        if "3d" in tensor_dict and tensor_dict["3d"] is not None:
            pooled_feats.append(self.conv3d_head(tensor_dict["3d"]))
        else:
            pooled_feats.append(torch.zeros(batch_size, (self.hidden_dim // 2) * 64, device=device))

        # Concatenate multi-modal representations along feature dimension
        fused_raw = torch.cat(pooled_feats, dim=-1)
        return self.organ_gating(fused_raw)


class SESIUniversalOrganSemanticDecoder(nn.Module):
    """
    Production-grade multi-organ, multi-modality semantic state decoder. 
    Translates physical scalar and vector fields across Brain, Heart, Liver, Lungs, 
    Kidneys, and Intestines directly into continuous amino acid embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 21,
        hidden_dim: int = 64,
        max_seq_len: int = 512,
        temperature: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.dev = device or torch.device("cpu")

        # Multi-Modality Signal Extractor
        self.feature_extractor = MultiModalityOrganFeatureExtractor(hidden_dim=hidden_dim)

        # Target Organ Embedding Conditioning (6 target organs)
        self.organ_embeddings = nn.Embedding(num_embeddings=6, embedding_dim=hidden_dim)
        self.organ_map = {
            OrganType.BRAIN: 0,
            OrganType.HEART: 1,
            OrganType.LIVER: 2,
            OrganType.LUNGS: 3,
            OrganType.KIDNEYS: 4,
            OrganType.INTESTINES: 5,
        }

        # Sequence Manifold Projector
        self.sequence_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, max_seq_len * vocab_size),
        )

        # Codebook Embedding Matrix for continuous backpropagation
        self.aa_codebook = nn.Parameter(torch.randn(vocab_size, embed_dim) / math.sqrt(embed_dim))

        self.to(self.dev)

    def forward(
        self,
        modality_tensors: Dict[str, torch.Tensor],
        organ_name: str = OrganType.BRAIN,
        hard: bool = False,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes end-to-end continuous projection from physical fields to amino acid sequences.

        Args:
            modality_tensors : Dictionary containing physical tensors ('1d', '2d', '3d').
            organ_name       : Target organ string (e.g., 'brain', 'heart', 'intestines').
            hard             : Straight-Through discrete quantization switch.
            tau              : Dynamic temperature for Gumbel-Softmax relaxation.

        Returns:
            Tuple of:
              - continuous_embeddings : (B, L, embed_dim) Fully differentiable sequence tensor.
              - sequence_logits       : (B, L, vocab_size) Unnormalized class logits.
              - hard_token_indices    : (B, L) Discrete token indices for inspection/logging.
        """
        tau = tau if tau is not None else self.temperature
        batch_size = next(iter(modality_tensors.values())).size(0)

        # Step A: Multi-Modality Spatial-Temporal Contraction
        fused_features = self.feature_extractor(modality_tensors)

        # Step B: Organ-Specific Dynamic Conditioning
        organ_idx = torch.tensor([self.organ_map.get(organ_name, 0)], device=self.dev).repeat(batch_size)
        organ_emb = self.organ_embeddings(organ_idx)
        conditioned_latent = torch.cat([fused_features, organ_emb], dim=-1)

        # Step C: Sequence Manifold Decoding
        logits = self.sequence_projector(conditioned_latent)
        logits = logits.view(batch_size, self.max_seq_len, self.vocab_size)

        # Step D: Continuous Gumbel-Softmax Relaxation for Gradient Continuity
        if self.training:
            soft_one_hot = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            soft_one_hot = F.softmax(logits / tau, dim=-1)

        # Step E: Matrix Multiplication onto Amino Acid Codebook
        continuous_embeddings = torch.matmul(soft_one_hot, self.aa_codebook)
        hard_token_indices = torch.argmax(logits, dim=-1)

        return continuous_embeddings, logits, hard_token_indices


class SESIUniversalE2EPipelineBridge(nn.Module):
    """
    Production E2E Pipeline Bridge connecting physical engines across multiple organs 
    and modalities directly to the downstream 3D Fold Refinement Engine.
    """

    def __init__(
        self,
        sesi_engine: nn.Module,
        universal_decoder: SESIUniversalOrganSemanticDecoder,
    ) -> None:
        super().__init__()
        self.sesi_engine = sesi_engine
        self.decoder = universal_decoder

    def process_multi_organ_step(
        self,
        raw_signals: Dict[str, torch.Tensor],
        modality: str,
        organ: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Routes raw signals through physical solvers (X-Ray, MRI, EEG, MEG, ECG, Motility) 
        and extracts continuous sequence embeddings.
        """
        processed_tensors = {}

        if modality == ModalityType.EEG:
            # Execute Poisson solver step
            v_next = self.sesi_engine.step_realtime_eeg(
                raw_signals["conductivity"],
                raw_signals["current_source"],
                raw_signals["scalar_potential"],
            )
            processed_tensors["3d"] = v_next

        elif modality == ModalityType.MEG:
            # Execute Biot-Savart solver step
            b_field = self.sesi_engine.step_realtime_meg(
                raw_signals["current_dipoles"],
                raw_signals["sensor_positions"],
            )
            processed_tensors["3d"] = b_field

        elif modality == ModalityType.MRI_CMR:
            # Execute Bloch equations solver step
            m_next = self.sesi_engine.step_realtime_mri(
                raw_signals["magnetization"],
                raw_signals["b_effective"],
                raw_signals["t1_map"],
                raw_signals["t2_map"],
                raw_signals["m0_equilibrium"],
            )
            processed_tensors["3d"] = m_next

        elif modality in [ModalityType.ECG, ModalityType.NMR]:
            processed_tensors["1d"] = raw_signals["signal_1d"]

        elif modality in [ModalityType.XRAY, ModalityType.PERISTALSIS]:
            processed_tensors["2d"] = raw_signals["signal_2d"]

        # Decode continuous embeddings with full gradient flow
        embeddings, logits, _ = self.decoder(processed_tensors, organ_name=organ)
        return embeddings, logits
