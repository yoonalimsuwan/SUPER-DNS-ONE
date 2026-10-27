"""
=============================================================================
SESI REAL DATA TRAINING — TCGA / PDB / UniProt Integration
Connect real biomedical data to SESI training pipeline
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DATA SOURCES
────────────
[D1] TCGA via GDC API — Cancer genomics (expression, mutation, clinical)
[D2] PDB via RCSB — Protein structures (coordinates, sequences)
[D3] UniProt — Protein sequences and function
[D4] Nextstrain — Viral sequences (open alternative to GISAID)

PIPELINE
────────
Fetch → Cache → Parse → PyTorch Dataset → DataLoader → SESI Training

HONEST SCOPE
────────────
• TCGA open-access only (controlled-access requires dbGaP)
• Real data downloads require internet
• Synthetic fallback when offline
• Data licenses must be respected (TCGA/PDB use policies)
• Clinical use requires regulatory approval
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.real_data_train")

__all__ = [
    "RealDataConfig",
    "TCGADataPipeline",
    "PDBDataPipeline",
    "RealDataTrainingPipeline",
]

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class RealDataConfig:
    """Configuration for real data training."""
    
    # ── Cache ─────────────────────────────────────────────────────────
    cache_dir: str = "./data_cache"
    
    # ── TCGA ──────────────────────────────────────────────────────────
    tcga_projects: List[str] = None
    tcga_max_files: int = 100
    
    # ── PDB ───────────────────────────────────────────────────────────
    pdb_ids: List[str] = None
    pdb_max_structures: int = 100
    
    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 8
    max_seq_len: int = 512
    device: str = "cuda"
    
    def __post_init__(self):
        if self.tcga_projects is None:
            self.tcga_projects = [
                "TCGA-BRCA",  # Breast cancer
                "TCGA-LUAD",  # Lung adenocarcinoma
                "TCGA-COAD",  # Colon adenocarcinoma
            ]
        if self.pdb_ids is None:
            self.pdb_ids = ["1UBQ", "1CRN", "2LZM", "1TIM", "1AKE"]


# ═══════════════════════════════════════════════════════════════════════
# TCGA DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class TCGADataPipeline:
    """
    Fetch and process TCGA data via GDC API.
    
    GDC API endpoints:
      • /cases — search cases
      • /files — search files
      • /data/{uuid} — download file
    
    HONEST: Open-access data only. Controlled-access requires dbGaP.
    """
    
    GDC_API = "https://api.gdc.cancer.gov"
    
    def __init__(self, cfg: RealDataConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "tcga"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_REQUESTS:
            logger.warning("requests not installed — using synthetic data")
    
    def search_cases(self, project_id: str, max_cases: int = 50) -> List[Dict]:
        """Search TCGA cases for a project."""
        cache_file = self.cache_dir / f"cases_{project_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        
        if not HAS_REQUESTS:
            return []
        
        try:
            query = {
                "filters": {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": project_id,
                    },
                },
                "fields": "case_id,submitter_id,diagnoses.primary_diagnosis",
                "format": "json",
                "size": max_cases,
            }
            
            resp = requests.post(f"{self.GDC_API}/cases", json=query, timeout=60)
            resp.raise_for_status()
            
            hits = resp.json().get("data", {}).get("hits", [])
            cache_file.write_text(json.dumps(hits))
            logger.info(f"TCGA {project_id}: {len(hits)} cases")
            return hits
            
        except Exception as e:
            logger.warning(f"TCGA case search failed: {e}")
            return []
    
    def fetch_expression_matrix(self, project_id: str) -> Optional[np.ndarray]:
        """
        Fetch gene expression data for a project.
        
        Returns: (n_samples, n_genes) array or None
        """
        cache_file = self.cache_dir / f"expr_{project_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        
        if not HAS_REQUESTS:
            return None
        
        try:
            # Search for expression files
            query = {
                "filters": {
                    "op": "and",
                    "content": [
                        {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
                    ],
                },
                "fields": "file_id,file_name",
                "format": "json",
                "size": self.cfg.tcga_max_files,
            }
            
            resp = requests.post(f"{self.GDC_API}/files", json=query, timeout=60)
            resp.raise_for_status()
            
            files = resp.json().get("data", {}).get("hits", [])
            
            if not files:
                logger.warning(f"No expression files found for {project_id}")
                return None
            
            # Download and parse first file as example
            file_id = files[0]["file_id"]
            data_resp = requests.get(f"{self.GDC_API}/data/{file_id}", timeout=120)
            data_resp.raise_for_status()
            
            # Parse TSV
            lines = data_resp.text.strip().split("\n")[1:]  # Skip header
            expression = []
            for line in lines[:1000]:  # Limit genes for demo
                parts = line.split("\t")
                try:
                    expression.append(float(parts[1]))
                except (ValueError, IndexError):
                    continue
            
            if expression:
                arr = np.array(expression, dtype=np.float32)
                np.save(cache_file, arr)
                logger.info(f"TCGA expression: {arr.shape}")
                return arr
            
            return None
            
        except Exception as e:
            logger.warning(f"TCGA expression fetch failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# PDB DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class PDBDataPipeline:
    """
    Fetch and process PDB structures via RCSB API.
    
    RCSB endpoints:
      • https://files.rcsb.org/download/{id}.pdb — PDB file
      • https://data.rcsb.org/rest/v1/core/entry/{id} — metadata
    """
    
    AA_3TO1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    def __init__(self, cfg: RealDataConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "pdb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_structure(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single PDB structure."""
        cache_file = self.cache_dir / f"{pdb_id.lower()}.pt"
        if cache_file.exists():
            return torch.load(cache_file, weights_only=False)
        
        if not HAS_REQUESTS:
            return None
        
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            
            data = self._parse_pdb(resp.text, pdb_id)
            
            if data:
                torch.save(data, cache_file)
                logger.info(f"PDB {pdb_id}: {data['length']} residues")
            
            return data
            
        except Exception as e:
            logger.warning(f"PDB {pdb_id} fetch failed: {e}")
            return None
    
    def _parse_pdb(self, pdb_text: str, pdb_id: str) -> Optional[Dict]:
        """Parse PDB format to extract sequence and CA coordinates."""
        sequence = []
        ca_coords = []
        
        for line in pdb_text.split("\n"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_name = line[17:20].strip()
                if res_name in self.AA_3TO1:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        sequence.append(self.AA_3TO1[res_name])
                        ca_coords.append([x, y, z])
                    except ValueError:
                        continue
        
        if len(sequence) < 10:
            return None
        
        return {
            "pdb_id": pdb_id,
            "sequence": "".join(sequence),
            "ca_coords": torch.tensor(ca_coords, dtype=torch.float32),
            "length": len(sequence),
        }
    
    def fetch_batch(self, pdb_ids: List[str]) -> List[Dict]:
        """Fetch multiple PDB structures."""
        results = []
        for pdb_id in pdb_ids:
            data = self.fetch_structure(pdb_id)
            if data:
                results.append(data)
        return results


# ═══════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════

class TCGADataset(Dataset):
    """PyTorch dataset for TCGA expression data."""
    
    def __init__(self, expression: np.ndarray, labels: Optional[np.ndarray] = None):
        self.expression = torch.tensor(expression, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    
    def __len__(self):
        return self.expression.shape[0]
    
    def __getitem__(self, idx):
        item = {"expression": self.expression[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class PDBDataset(Dataset):
    """PyTorch dataset for PDB structures."""
    
    AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    
    def __init__(self, structures: List[Dict], max_len: int = 512):
        self.structures = structures
        self.max_len = max_len
    
    def __len__(self):
        return len(self.structures)
    
    def _encode_seq(self, seq: str) -> Tensor:
        indices = [self.AA_TO_IDX.get(aa, 20) for aa in seq]
        return torch.tensor(indices, dtype=torch.long)
    
    def __getitem__(self, idx):
        struct = self.structures[idx]
        
        seq = struct["sequence"][:self.max_len]
        coords = struct["ca_coords"][:self.max_len]
        
        seq_ids = self._encode_seq(seq)
        L = len(seq_ids)
        
        # Pad
        if L < self.max_len:
            seq_ids = F.pad(seq_ids, (0, self.max_len - L), value=20)
            coords = F.pad(coords, (0, 0, 0, self.max_len - L))
        
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:L] = True
        
        return {
            "seq_ids": seq_ids,
            "ca_coords": coords,
            "mask": mask,
            "length": L,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class RealDataTrainingPipeline:
    """
    Unified pipeline connecting real data to SESI training.
    
    Workflow:
      1. Fetch real data (TCGA/PDB)
      2. Create PyTorch datasets
      3. Connect to existing SESI models
      4. Train with CloudDistributedTrainer
    """
    
    def __init__(self, cfg: RealDataConfig):
        self.cfg = cfg
        self.tcga_pipeline = TCGADataPipeline(cfg)
        self.pdb_pipeline = PDBDataPipeline(cfg)
    
    def prepare_tcga_data(self) -> Optional[DataLoader]:
        """Prepare TCGA data loader."""
        all_expression = []
        
        for project in self.cfg.tcga_projects:
            logger.info(f"Fetching TCGA project: {project}")
            expr = self.tcga_pipeline.fetch_expression_matrix(project)
            if expr is not None:
                all_expression.append(expr)
        
        if not all_expression:
            logger.warning("No TCGA data fetched — using synthetic")
            return None
        
        # Combine
        combined = np.concatenate(all_expression, axis=0)
        dataset = TCGADataset(combined)
        
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
        )
    
    def prepare_pdb_data(self) -> Optional[DataLoader]:
        """Prepare PDB data loader."""
        structures = self.pdb_pipeline.fetch_batch(self.cfg.pdb_ids)
        
        if not structures:
            logger.warning("No PDB data fetched — using synthetic")
            return None
        
        dataset = PDBDataset(structures, self.cfg.max_seq_len)
        
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
        )
    
    def get_synthetic_fallback(self, data_type: str = "pdb") -> DataLoader:
        """Generate synthetic data for testing when real data unavailable."""
        if data_type == "pdb":
            structures = []
            for i in range(20):
                length = np.random.randint(50, 150)
                seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), length))
                coords = torch.randn(length, 3) * 10
                
                structures.append({
                    "pdb_id": f"SYN{i:03d}",
                    "sequence": seq,
                    "ca_coords": coords,
                    "length": length,
                })
            
            dataset = PDBDataset(structures, self.cfg.max_seq_len)
        else:
            expression = np.random.randn(100, 1000).astype(np.float32)
            dataset = TCGADataset(expression)
        
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
