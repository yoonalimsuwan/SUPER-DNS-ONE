"""
=============================================================================
SESI GOOGLE VERTEX AI DEPLOYMENT
Custom Training → Model Upload → Endpoint with Autoscaling
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
License      : MIT | Year: 2026

REQUIREMENTS
────────────
  pip install google-cloud-aiplatform google-cloud-storage
  gcloud auth application-default login
  GCP project with Vertex AI API enabled

PIPELINE
────────
  1. Build/push Docker to Artifact Registry
  2. Submit Custom Training Job (multi-GPU/TPU)
  3. Upload model to Vertex Model Registry
  4. Deploy to Endpoint with autoscaling
  5. Optional: TPU training support

HONEST SCOPE
────────────
• Requires valid GCP project + billing + permissions
• TPU training needs TPU-compatible code (JAX or TF)
• This module uses google-cloud-aiplatform SDK; no simulation
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("sesi.vertex_ai")

__all__ = [
    "VertexAIConfig",
    "VertexAIDeployer",
]

try:
    from google.cloud import aiplatform
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False


@dataclass
class VertexAIConfig:
    # GCP
    project_id: str = "your-gcp-project"
    location: str = "us-central1"
    staging_bucket: str = "gs://sesi-ml-staging"

    # Artifact Registry
    repo_name: str = "sesi-models"
    image_name: str = "sesi-training"
    image_tag: str = "latest"

    # Training
    # GPU options: nvidia-tesla-a100, nvidia-l4, nvidia-tesla-t4
    # TPU options: tpu-v3-8, tpu-v4-8
    accelerator_type: str = "NVIDIA_TESLA_A100"
    accelerator_count: int = 1
    machine_type: str = "n1-standard-8"

    # TPU alternative
    use_tpu: bool = False
    tpu_type: str = "tpu-v3-8"

    # Model
    model_name: str = "sesi-protein-fold"
    model_version: str = "v1"

    # Endpoint
    endpoint_machine_type: str = "n1-standard-4"
    endpoint_accelerator: str = "NVIDIA_TESLA_T4"
    endpoint_accelerator_count: int = 1
    min_replica_count: int = 1
    max_replica_count: int = 5


# ═══════════════════════════════════════════════════════════════════════
# VERTEX AI DEPLOYER
# ═══════════════════════════════════════════════════════════════════════

class VertexAIDeployer:
    """Orchestrates the full Vertex AI lifecycle."""

    def __init__(self, cfg: VertexAIConfig):
        self.cfg = cfg

        if HAS_VERTEX:
            aiplatform.init(
                project=cfg.project_id,
                location=cfg.location,
                staging_bucket=cfg.staging_bucket,
            )
            logger.info(f"Vertex AI initialized: {cfg.project_id} / {cfg.location}")
        else:
            logger.warning("google-cloud-aiplatform not installed — dry-run mode")

    # ── Docker to Artifact Registry ────────────────────────────────────
    def push_image(self, code_dir: str = ".") -> Optional[str]:
        """Build and push Docker image to GCP Artifact Registry."""
        registry_host = f"{self.cfg.location}-docker.pkg.dev"
        full_image = (
            f"{registry_host}/{self.cfg.project_id}/"
            f"{self.cfg.repo_name}/{self.cfg.image_name}:{self.cfg.image_tag}"
        )

        # Ensure repo exists
        try:
            subprocess.run([
                "gcloud", "artifacts", "repositories", "create", self.cfg.repo_name,
                "--repository-format=docker",
                f"--location={self.cfg.location}",
                f"--project={self.cfg.project_id}",
            ], check=False, capture_output=True)
        except FileNotFoundError:
            logger.warning("gcloud CLI not found")
            return full_image

        # Docker build + push
        try:
            subprocess.run(["docker", "build", "-t", full_image, code_dir], check=True)
            subprocess.run(["docker", "push", full_image], check=True)
            logger.info(f"Image pushed → {full_image}")
            return full_image
        except Exception as e:
            logger.error(f"Image push failed: {e}")
            return full_image

    # ── Custom Training Job ────────────────────────────────────────────
    def create_training_job(
        self,
        image_uri: str,
        script_path: str = "train.py",
        args: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Submit a Vertex AI Custom Training Job."""
        job_args = args or ["--epochs", "100", "--batch-size", "16"]

        if not HAS_VERTEX:
            logger.info("[DRY-RUN] Would create CustomJob with:")
            logger.info(f"  image: {image_uri}")
            logger.info(f"  accelerator: {self.cfg.accelerator_type} x {self.cfg.accelerator_count}")
            return {"dry_run": True}

        # Choose worker pool spec based on GPU vs TPU
        if self.cfg.use_tpu:
            worker_pool_spec = {
                "machine_spec": {
                    "machine_type": "cloud-tpu",
                    "accelerator_type": self.cfg.tpu_type.upper(),
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", script_path],
                    "args": job_args,
                },
            }
        else:
            worker_pool_spec = {
                "machine_spec": {
                    "machine_type": self.cfg.machine_type,
                    "accelerator_type": self.cfg.accelerator_type,
                    "accelerator_count": self.cfg.accelerator_count,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", script_path],
                    "args": job_args,
                },
            }

        job = aiplatform.CustomJob(
            display_name=f"{self.cfg.model_name}-training",
            worker_pool_specs=[worker_pool_spec],
            staging_bucket=self.cfg.staging_bucket,
        )

        job.run(sync=False)
        logger.info(f"Training job submitted: {job.resource_name}")
        return {"job_resource": job.resource_name, "status": "running"}

    # ── Model Upload ───────────────────────────────────────────────────
    def upload_model(
        self,
        image_uri: str,
        artifact_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload trained model to Vertex Model Registry."""
        artifact_uri = artifact_uri or f"{self.cfg.staging_bucket}/model"

        if not HAS_VERTEX:
            return {"dry_run": True, "model_name": self.cfg.model_name}

        model = aiplatform.Model.upload(
            display_name=f"{self.cfg.model_name}-{self.cfg.model_version}",
            artifact_uri=artifact_uri,
            serving_container_image_uri=image_uri,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
        )

        logger.info(f"Model uploaded: {model.resource_name}")
        return {"model_resource": model.resource_name}

    # ── Endpoint Deployment ────────────────────────────────────────────
    def deploy_endpoint(self, model_resource: str) -> Dict[str, Any]:
        """Deploy model to a Vertex AI Endpoint with autoscaling."""
        if not HAS_VERTEX:
            return {"dry_run": True, "endpoint": f"{self.cfg.model_name}-endpoint"}

        # Get model object
        model = aiplatform.Model(model_resource)

        # Create endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name=f"{self.cfg.model_name}-endpoint",
        )

        # Deploy with autoscaling
        model.deploy(
            endpoint=endpoint,
            machine_type=self.cfg.endpoint_machine_type,
            accelerator_type=self.cfg.endpoint_accelerator,
            accelerator_count=self.cfg.endpoint_accelerator_count,
            min_replica_count=self.cfg.min_replica_count,
            max_replica_count=self.cfg.max_replica_count,
            traffic_split={"0": 100},
        )

        logger.info(f"Endpoint deployed: {endpoint.resource_name}")
        return {"endpoint_resource": endpoint.resource_name}

    # ── Full pipeline ──────────────────────────────────────────────────
    def run_full_pipeline(self, code_dir: str = ".") -> Dict[str, Any]:
        """Execute push → train → upload → deploy."""
        results = {}

        logger.info("Step 1: Push image")
        image_uri = self.push_image(code_dir)
        results["image_uri"] = image_uri

        logger.info("Step 2: Training job")
        results["training"] = self.create_training_job(image_uri)

        logger.info("Step 3: Upload model")
        results["model"] = self.upload_model(image_uri)

        logger.info("Step 4: Deploy endpoint")
        model_res = results["model"].get("model_resource", "PLACEHOLDER")
        results["endpoint"] = self.deploy_endpoint(model_res)

        return results


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Vertex AI Deployment — Smoke Test (DRY-RUN)")
    print("=" * 72)
    print(f"  google-cloud-aiplatform available: {HAS_VERTEX}")

    cfg = VertexAIConfig()
    deployer = VertexAIDeployer(cfg)

    # Test training job config (dry-run)
    result = deployer.create_training_job(
        image_uri=f"us-central1-docker.pkg.dev/{cfg.project_id}/{cfg.repo_name}/{cfg.image_name}:latest",
        args=["--epochs", "50"],
    )
    print(f"\n  Training job: {'dry_run' if result.get('dry_run') else result}")
    print(f"  Accelerator: {cfg.accelerator_type} x {cfg.accelerator_count}")
    print(f"  TPU mode: {cfg.use_tpu}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Real deployment needs GCP project + billing enabled")
    print("  • TPU training requires JAX/TensorFlow (not PyTorch)")
    print("  • A100 GPUs have quota limits on new accounts")
    print("  • Test with n1-standard + T4 first to control costs")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
