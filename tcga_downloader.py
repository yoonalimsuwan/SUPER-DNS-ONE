"""
=============================================================================
SESI TCGA DATA DOWNLOADER
Download real cancer genomics data from the Genomic Data Commons (GDC)
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DATA SOURCES (GDC Open API)
───────────────────────────
GDC API base:  https://api.gdc.cancer.gov
  • /cases          — ค้นหา case (ผู้ป่วย) ตามโปรเจกต์
  • /files          — ค้นหาไฟล์ข้อมูล
  • /data/{uuid}    — ดาวน์โหลดไฟล์
  • /annotations    — clinical annotations

DATA TYPES AVAILABLE (open-access, no dbGaP needed)
───────────────────────────────────────────────────
  • Gene Expression Quantification (STAR-Counts) — RNA-seq
  • Copy Number Segment
  • Masked Somatic Mutation (MAF)
  • Methylation Beta Value
  • Clinical data (via /cases)

HONEST SCOPE
────────────
• This downloads OPEN-ACCESS data only. Controlled-access data
  (germline variants, raw FASTQ) requires dbGaP authorization.
• Real downloads require internet connection and respect GDC
  rate limits. Use the --max-files flag to limit data volume.
• Data is cached locally; subsequent runs skip already-downloaded files.
• Always respect the GDC Data Use Agreement:
  https://gdc.cancer.gov/access-data/data-use-agreements
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
import hashlib
import tarfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.tcga_downloader")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = ["TCGAConfig", "TCGADownloader"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TCGAConfig:
    """Configuration for TCGA data download."""

    # ── GDC API ─────────────────────────────────────────────────────────
    gdc_api: str = "https://api.gdc.cancer.gov"
    timeout: int = 60

    # ── Projects to download ────────────────────────────────────────────
    # Common TCGA projects (see https://gdc.cancer.gov/resources-tcga-users)
    projects: List[str] = field(default_factory=lambda: [
        "TCGA-BRCA",   # Breast invasive carcinoma
        "TCGA-LUAD",   # Lung adenocarcinoma
        "TCGA-GBM",    # Glioblastoma multiforme
        "TCGA-COAD",   # Colon adenocarcinoma
    ])

    # ── Data type ───────────────────────────────────────────────────────
    data_type: str = "Gene Expression Quantification"
    workflow: str = "STAR - Counts"
    access: str = "open"          # "open" or "controlled"

    # ── Limits ──────────────────────────────────────────────────────────
    max_files_per_project: int = 50   # limit download volume
    max_genes: int = 5000             # keep top N genes for memory

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "./data/tcga"
    format: str = "npz"              # "npz" | "pt" | "csv"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# TCGA DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════

class TCGADownloader:
    """
    Downloads and processes TCGA data from the GDC.

    Workflow:
      1. Query GDC /files for matching data files
      2. Download each file (with caching)
      3. Extract and parse (TCGA files are tar.gz with TSV inside)
      4. Assemble into a matrix (samples × genes)
      5. Save in the requested format
    """

    def __init__(self, cfg: TCGAConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.output_dir) / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = Path(cfg.output_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")

    # ── Step 1: Query files ────────────────────────────────────────────
    def query_files(self, project_id: str) -> List[Dict]:
        """
        Query GDC for data files matching the project and data type.

        Returns list of file metadata dicts.
        """
        url = f"{self.cfg.gdc_api}/files"

        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id]
                }},
                {"op": "in", "content": {
                    "field": "data_type",
                    "value": [self.cfg.data_type]
                }},
                {"op": "in", "content": {
                    "field": "access",
                    "value": [self.cfg.access]
                }},
            ]
        }

        if self.cfg.workflow:
            filters["content"].append({
                "op": "in", "content": {
                    "field": "analysis.workflow_type",
                    "value": [self.cfg.workflow]
                }
            })

        params = {
            "filters": json.dumps(filters),
            "fields": ",".join([
                "file_id", "file_name", "md5sum", "file_size",
                "cases.submitter_id", "cases.case_id"
            ]),
            "format": "json",
            "size": self.cfg.max_files_per_project,
            "from": 0
        }

        try:
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)
            resp.raise_for_status()
            hits = resp.json().get("data", {}).get("hits", [])
            logger.info(f"[{project_id}] Found {len(hits)} files")
            return hits
        except Exception as e:
            logger.error(f"[{project_id}] Query failed: {e}")
            return []

    # ── Step 2: Download a single file ─────────────────────────────────
    def download_file(self, file_meta: Dict) -> Optional[Path]:
        """
        Download a single file from GDC. Returns cached path if exists.
        """
        file_id = file_meta["file_id"]
        file_name = file_meta.get("file_name", f"{file_id}.tar.gz")
        md5 = file_meta.get("md5sum", "")

        # Use file_id as cache key (stable across runs)
        cache_path = self.cache_dir / f"{file_id}.tar.gz"

        if cache_path.exists():
            logger.debug(f"Cache hit: {file_id}")
            return cache_path

        url = f"{self.cfg.gdc_api}/data/{file_id}"
        try:
            resp = requests.get(url, timeout=self.cfg.timeout, stream=True)
            resp.raise_for_status()

            # Stream to temp file, verify md5
            tmp_path = cache_path.with_suffix(".tmp")
            md5_hash = hashlib.md5()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    md5_hash.update(chunk)

            # Verify checksum
            if md5 and md5_hash.hexdigest() != md5:
                logger.warning(f"MD5 mismatch for {file_id}, skipping")
                tmp_path.unlink(missing_ok=True)
                return None

            tmp_path.rename(cache_path)
            logger.debug(f"Downloaded: {file_id} ({cache_path.stat().st_size} bytes)")
            return cache_path

        except Exception as e:
            logger.error(f"Download failed for {file_id}: {e}")
            return None

    # ── Step 3: Extract and parse ──────────────────────────────────────
    def extract_expression(self, tar_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract gene expression from a TCGA tar.gz file.

        TCGA STAR-Counts files contain a .tsv with columns:
          gene_id, gene_name, unstranded, stranded_first, stranded_second,
          tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded

        Returns dict with gene names and expression values.
        """
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                # Find the .tsv file inside
                tsv_members = [m for m in tar.getmembers()
                               if m.name.endswith(".tsv")]
                if not tsv_members:
                    return None

                tsv_file = tar.extractfile(tsv_members[0])
                if tsv_file is None:
                    return None

                lines = tsv_file.read().decode("utf-8").strip().split("\n")
                header = lines[0].split("\t")

                # Find column indices
                try:
                    gene_idx = header.index("gene_name")
                    # Use "unstranded" raw counts or "tpm_unstranded"
                    expr_col = "tpm_unstranded" if "tpm_unstranded" in header else "unstranded"
                    expr_idx = header.index(expr_col)
                except ValueError:
                    return None

                gene_names = []
                expr_values = []
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) > max(gene_idx, expr_idx):
                        try:
                            gene_names.append(parts[gene_idx])
                            expr_values.append(float(parts[expr_idx]))
                        except (ValueError, IndexError):
                            continue

                return {
                    "gene_names": np.array(gene_names),
                    "expression": np.array(expr_values, dtype=np.float32),
                }

        except Exception as e:
            logger.error(f"Extraction failed for {tar_path}: {e}")
            return None

    # ── Step 4: Download all files for a project ───────────────────────
    def download_project(self, project_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Download all files for a project and assemble into a matrix.

        Returns dict with:
          - 'expression': (n_samples, n_genes) matrix
          - 'gene_names': (n_genes,) array
          - 'sample_ids': (n_samples,) array
        """
        files = self.query_files(project_id)
        if not files:
            logger.warning(f"[{project_id}] No files found")
            return None

        all_expressions = []
        all_gene_names = None
        sample_ids = []

        for i, fmeta in enumerate(files):
            if i % 10 == 0:
                logger.info(f"[{project_id}] Processing file {i+1}/{len(files)}")

            tar_path = self.download_file(fmeta)
            if tar_path is None:
                continue

            expr_data = self.extract_expression(tar_path)
            if expr_data is None:
                continue

            # On first file, record gene order
            if all_gene_names is None:
                all_gene_names = expr_data["gene_names"]

            # Align to first file's gene order
            if not np.array_equal(expr_data["gene_names"], all_gene_names):
                # Build mapping
                name_to_idx = {g: idx for idx, g in enumerate(all_gene_names)}
                aligned = np.zeros(len(all_gene_names), dtype=np.float32)
                for gname, gval in zip(expr_data["gene_names"], expr_data["expression"]):
                    if gname in name_to_idx:
                        aligned[name_to_idx[gname]] = gval
                expr_data["expression"] = aligned

            all_expressions.append(expr_data["expression"])
            # Use case submitter_id as sample identifier
            case_id = fmeta.get("cases", [{}])[0].get("submitter_id", fmeta["file_id"])
            sample_ids.append(case_id)

            # Rate limiting: be polite to GDC
            time.sleep(0.5)

        if not all_expressions:
            return None

        # Stack into matrix: (n_samples, n_genes)
        expression_matrix = np.stack(all_expressions, axis=0)

        # Keep top N genes by variance (for memory efficiency)
        if self.cfg.max_genes and expression_matrix.shape[1] > self.cfg.max_genes:
            variances = np.var(expression_matrix, axis=0)
            top_idx = np.argsort(variances)[-self.cfg.max_genes:]
            expression_matrix = expression_matrix[:, top_idx]
            all_gene_names = all_gene_names[top_idx]
            logger.info(f"[{project_id}] Reduced to top {self.cfg.max_genes} genes by variance")

        return {
            "expression": expression_matrix,
            "gene_names": all_gene_names,
            "sample_ids": np.array(sample_ids),
            "project_id": project_id,
        }

    # ── Step 5: Save processed data ────────────────────────────────────
    def save(self, data: Dict[str, np.ndarray]) -> Path:
        """Save processed data in the configured format."""
        project_id = data["project_id"]
        fmt = self.cfg.format

        if fmt == "npz":
            out_path = self.processed_dir / f"{project_id}.npz"
            np.savez_compressed(
                out_path,
                expression=data["expression"],
                gene_names=data["gene_names"],
                sample_ids=data["sample_ids"],
            )
        elif fmt == "pt" and HAS_TORCH:
            out_path = self.processed_dir / f"{project_id}.pt"
            torch.save({
                "expression": torch.from_numpy(data["expression"]),
                "gene_names": list(data["gene_names"]),
                "sample_ids": list(data["sample_ids"]),
            }, out_path)
        elif fmt == "csv":
            out_path = self.processed_dir / f"{project_id}.csv"
            import csv
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id"] + list(data["gene_names"]))
                for sid, row in zip(data["sample_ids"], data["expression"]):
                    writer.writerow([sid] + list(row))
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"Saved: {out_path} "
                    f"({data['expression'].shape[0]} samples × "
                    f"{data['expression'].shape[1]} genes)")
        return out_path

    # ── Main entry point ───────────────────────────────────────────────
    def run(self) -> Dict[str, Path]:
        """Download all configured projects. Returns {project_id: path}."""
        results = {}
        for project_id in self.cfg.projects:
            logger.info(f"{'='*60}")
            logger.info(f"Downloading project: {project_id}")
            logger.info(f"{'='*60}")

            data = self.download_project(project_id)
            if data is None:
                logger.warning(f"[{project_id}] No data obtained")
                continue

            path = self.save(data)
            results[project_id] = path

        logger.info(f"\nComplete. Downloaded {len(results)}/{len(self.cfg.projects)} projects.")
        return results


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SESI TCGA Data Downloader")
    parser.add_argument("--projects", nargs="+",
                        default=["TCGA-BRCA", "TCGA-LUAD"],
                        help="TCGA project IDs to download")
    parser.add_argument("--max-files", type=int, default=50,
                        help="Max files per project")
    parser.add_argument("--max-genes", type=int, default=5000,
                        help="Keep top N genes by variance")
    parser.add_argument("--output-dir", type=str, default="./data/tcga")
    parser.add_argument("--format", type=str, default="npz",
                        choices=["npz", "pt", "csv"])
    parser.add_argument("--data-type", type=str,
                        default="Gene Expression Quantification")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")

    cfg = TCGAConfig(
        projects=args.projects,
        max_files_per_project=args.max_files,
        max_genes=args.max_genes,
        output_dir=args.output_dir,
        format=args.format,
        data_type=args.data_type,
    )

    downloader = TCGADownloader(cfg)
    results = downloader.run()

    print("\n" + "=" * 60)
    print("DOWNLOADED FILES")
    print("=" * 60)
    for proj, path in results.items():
        print(f"  {proj}: {path}")


if __name__ == "__main__":
    main()
