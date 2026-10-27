"""
=============================================================================
SESI CLOUD DEPLOYMENT — AWS SageMaker + Google Vertex AI
Unified cloud training and inference deployment
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0

CLOUD PROVIDERS
───────────────
[AWS SageMaker]
  • Training jobs (multi-GPU, distributed)
  • Model registry
  • Real-time / batch / serverless endpoints
  • Requires: boto3, sagemaker SDK, AWS credentials

[Google Vertex AI]
  • Custom training (GPU/TPU)
  • Model upload + registry
  • Endpoints with autoscaling
  • Requires: google-cloud-aiplatform, GCP credentials

DOCKER IMAGE
────────────
Both providers use a Docker image containing:
  • PyTorch + model code
  • Training/inference entry points
  • SESI dependencies

HONEST SCOPE
────────────
• Cloud deployment requires valid credentials and billing
• GPU/TPU instances incur real costs — test with small instances first
• Docker image must be built and pushed to ECR/GAR before training
• This module generates configs and calls SDKs; it does not simulate cloud
• IAM/service account permissions must be configured separately
• Actual training/inference behavior depends on cloud provider APIs
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("sesi.cloud_deploy")

__all__ = [
    "CloudDeployConfig",
    "DockerImageBuilder",
    "SageMakerDeployer",
    "VertexAIDeployer",
    "CloudDeploymentOrchestrator",
]

# ── Optional cloud SDK imports ────────────────────────────────────────
try:
    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch
    HAS_SAGEMAKER = True
except ImportError:
    HAS_SAGEMAKER = False

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import CustomTrainingJob, Model, Endpoint
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False

try:
    import docker as docker_lib
    HAS_DOCKER_LIB = True
except ImportError:
    HAS_DOCKER_LIB = False


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CloudDeployConfig:
    # ── General ─────────────────────────────────────────────────────────
    model_name: str = "sesi-protein-fold"
    model_version: str = "1.0.0"
    code_dir: str = "."              # directory with training code
    entry_point: str = "train.py"    # training script
    requirements_file: str = "requirements.txt"

    # ── Docker ──────────────────────────────────────────────────────────
    base_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    image_name: str = "sesi-training"
    image_tag: str = "latest"

    # ── AWS SageMaker ───────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_role_arn: str = ""           # IAM role for SageMaker
    s3_bucket: str = ""              # S3 bucket for artifacts
    sm_instance_type: str = "ml.g5.xlarge"  # GPU instance
    sm_instance_count: int = 1
    sm_volume_gb: int = 100
    sm_max_runtime_hours: int = 24

    # ── Google Vertex AI ────────────────────────────────────────────────
    gcp_project: str = ""
    gcp_region: str = "us-central1"
    gcp_staging_bucket: str = ""     # gs:// bucket for staging
    vertex_machine_type: str = "n1-standard-8"
    vertex_accelerator_type: str = "NVIDIA_TESLA_T4"
    vertex_accelerator_count: int = 1
    vertex_replica_count: int = 1

    # ── Inference ───────────────────────────────────────────────────────
    inference_instance_type: str = "ml.t3.medium"  # for SageMaker endpoint
    endpoint_min_replicas: int = 1
    endpoint_max_replicas: int = 3

    # ── Hyperparameters ─────────────────────────────────────────────────
    hyperparameters: Dict[str, str] = field(default_factory=lambda: {
        "epochs": "10",
        "batch-size": "16",
        "lr": "3e-4",
    })

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "./cloud_deploy"


# ═══════════════════════════════════════════════════════════════════════
# DOCKER IMAGE BUILDER
# ═══════════════════════════════════════════════════════════════════════

class DockerImageBuilder:
    """
    Builds Docker image for cloud training/inference.

    Generates Dockerfile and optionally builds/pushes the image.
    """

    DOCKERFILE_TEMPLATE = '''FROM {base_image}

# Install dependencies
COPY {requirements_file} /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training code
COPY {code_dir}/ /opt/ml/code/
WORKDIR /opt/ml/code

# Environment
ENV PYTHONUNBUFFERED=TRUE
ENV SAGEMAKER_PROGRAM={entry_point}

# Default entrypoint
ENTRYPOINT ["python", "{entry_point}"]
'''

    def __init__(self, cfg: CloudDeployConfig):
        self.cfg = cfg

    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        return self.DOCKERFILE_TEMPLATE.format(
            base_image=self.cfg.base_image,
            requirements_file=self.cfg.requirements_file,
            code_dir=self.cfg.code_dir,
            entry_point=self.cfg.entry_point,
        )

    def write_dockerfile(self, output_dir: str = ".") -> str:
        """Write Dockerfile to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dockerfile_path = Path(output_dir) / "Dockerfile.cloud"
        dockerfile_path.write_text(self.generate_dockerfile())
        logger.info(f"Dockerfile written → {dockerfile_path}")
        return str(dockerfile_path)

    def build(self, dockerfile_path: str) -> bool:
        """Build Docker image locally."""
        full_tag = f"{self.cfg.image_name}:{self.cfg.image_tag}"
        cmd = ["docker", "build", "-f", dockerfile_path, "-t", full_tag, self.cfg.code_dir]
        logger.info(f"Building image: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Image built: {full_tag}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Docker build failed: {e}")
            return False

    def push_to_ecr(self, account_id: str) -> Optional[str]:
        """Push image to AWS ECR."""
        ecr_uri = f"{account_id}.dkr.ecr.{self.cfg.aws_region}.amazonaws.com/{self.cfg.image_name}:{self.cfg.image_tag}"
        local_tag = f"{self.cfg.image_name}:{self.cfg.image_tag}"
        try:
            # Login to ECR
            subprocess.run([
                "aws", "ecr", "get-login-password", "--region", self.cfg.aws_region,
            ], check=True, capture_output=True)
            # Tag and push
            subprocess.run(["docker", "tag", local_tag, ecr_uri], check=True)
            subprocess.run(["docker", "push", ecr_uri], check=True)
            logger.info(f"Image pushed to ECR: {ecr_uri}")
            return ecr_uri
        except Exception as e:
            logger.error(f"ECR push failed: {e}")
            return None

    def push_to_gar(self) -> Optional[str]:
        """Push image to Google Artifact Registry."""
        gar_uri = (
            f"{self.cfg.gcp_region}-docker.pkg.dev/"
            f"{self.cfg.gcp_project}/sesi-repo/"
            f"{self.cfg.image_name}:{self.cfg.image_tag}"
        )
        local_tag = f"{self.cfg.image_name}:{self.cfg.image_tag}"
        try:
            subprocess.run(["docker", "tag", local_tag, gar_uri], check=True)
            subprocess.run(["docker", "push", gar_uri], check=True)
            logger.info(f"Image pushed to GAR: {gar_uri}")
            return gar_uri
        except Exception as e:
            logger.error(f"GAR push failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# AWS SAGEMAKER DEPLOYER
# ═══════════════════════════════════════════════════════════════════════

class SageMakerDeployer:
    """
    AWS SageMaker deployment: training jobs, model registry, endpoints.

    Requires:
      • boto3 and sagemaker Python packages
      • AWS credentials configured (aws configure or env vars)
      • IAM role with SageMaker permissions
      • S3 bucket for artifacts
    """

    def __init__(self, cfg: CloudDeployConfig):
        self.cfg = cfg
        self.available = HAS_SAGEMAKER
        if not self.available:
            logger.warning(
                "SageMaker SDK not installed. Install with: "
                "pip install boto3 sagemaker"
            )

    def _check_prerequisites(self) -> List[str]:
        """Check that prerequisites are met."""
        issues = []
        if not self.available:
            issues.append("sagemaker SDK not installed")
        if not self.cfg.aws_role_arn:
            issues.append("aws_role_arn not configured")
        if not self.cfg.s3_bucket:
            issues.append("s3_bucket not configured")
        return issues

    def create_training_job(
        self,
        image_uri: str,
        job_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a SageMaker training job.

        Args:
            image_uri: Docker image URI (ECR)
            job_name: optional job name (auto-generated if None)

        Returns:
            dict with job status
        """
        result = {
            "success": False,
            "job_name": None,
            "status": None,
            "errors": [],
        }

        issues = self._check_prerequisites()
        if issues:
            result["errors"] = issues
            logger.error(f"SageMaker prerequisites not met: {issues}")
            return result

        if job_name is None:
            import time
            job_name = f"{self.cfg.model_name}-train-{int(time.time())}"

        try:
            estimator = PyTorch(
                image_uri=image_uri,
                role=self.cfg.aws_role_arn,
                instance_count=self.cfg.sm_instance_count,
                instance_type=self.cfg.sm_instance_type,
                volume_size=self.cfg.sm_volume_gb,
                max_run=self.cfg.sm_max_runtime_hours * 3600,
                hyperparameters=self.cfg.hyperparameters,
                output_path=f"s3://{self.cfg.s3_bucket}/sagemaker-output",
                base_job_name=job_name,
            )

            # Start training (async)
            estimator.fit(wait=False)

            result["success"] = True
            result["job_name"] = job_name
            result["status"] = "submitted"
            logger.info(f"SageMaker training job submitted: {job_name}")

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"SageMaker training job failed: {e}")

        return result

    def deploy_endpoint(
        self,
        model_data_s3: str,
        image_uri: str,
        endpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deploy a real-time inference endpoint.

        Args:
            model_data_s3: S3 path to model.tar.gz
            image_uri: Docker image URI for inference
            endpoint_name: optional endpoint name

        Returns:
            dict with endpoint status
        """
        result = {
            "success": False,
            "endpoint_name": None,
            "status": None,
            "errors": [],
        }

        issues = self._check_prerequisites()
        if issues:
            result["errors"] = issues
            return result

        if endpoint_name is None:
            endpoint_name = f"{self.cfg.model_name}-endpoint"

        try:
            from sagemaker.model import Model
            from sagemaker.predictor import Predictor

            model = Model(
                model_data=model_data_s3,
                image_uri=image_uri,
                role=self.cfg.aws_role_arn,
                predictor_cls=Predictor,
            )

            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=self.cfg.inference_instance_type,
                endpoint_name=endpoint_name,
            )

            result["success"] = True
            result["endpoint_name"] = endpoint_name
            result["status"] = "deploying"
            logger.info(f"SageMaker endpoint deploying: {endpoint_name}")

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"SageMaker endpoint deployment failed: {e}")

        return result

    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """Check training job status."""
        if not self.available:
            return {"status": "unknown", "error": "SDK not available"}

        try:
            sm_client = boto3.client("sagemaker", region_name=self.cfg.aws_region)
            response = sm_client.describe_training_job(TrainingJobName=job_name)
            return {
                "status": response["TrainingJobStatus"],
                "secondary_status": response.get("SecondaryStatus", ""),
                "failure_reason": response.get("FailureReason", ""),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# GOOGLE VERTEX AI DEPLOYER
# ═══════════════════════════════════════════════════════════════════════

class VertexAIDeployer:
    """
    Google Vertex AI deployment: custom training, model upload, endpoints.

    Requires:
      • google-cloud-aiplatform Python package
      • GCP project with Vertex AI API enabled
      • Application Default Credentials or service account
      • GCS bucket for staging
    """

    def __init__(self, cfg: CloudDeployConfig):
        self.cfg = cfg
        self.available = HAS_VERTEX
        self._initialized = False

        if not self.available:
            logger.warning(
                "Vertex AI SDK not installed. Install with: "
                "pip install google-cloud-aiplatform"
            )

    def _check_prerequisites(self) -> List[str]:
        """Check that prerequisites are met."""
        issues = []
        if not self.available:
            issues.append("google-cloud-aiplatform not installed")
        if not self.cfg.gcp_project:
            issues.append("gcp_project not configured")
        if not self.cfg.gcp_staging_bucket:
            issues.append("gcp_staging_bucket not configured")
        return issues

    def _init_vertex(self) -> bool:
        """Initialize Vertex AI SDK."""
        if self._initialized:
            return True

        issues = self._check_prerequisites()
        if issues:
            logger.error(f"Vertex AI prerequisites not met: {issues}")
            return False

        try:
            aiplatform.init(
                project=self.cfg.gcp_project,
                location=self.cfg.gcp_region,
                staging_bucket=self.cfg.gcp_staging_bucket,
            )
            self._initialized = True
            logger.info(f"Vertex AI initialized: {self.cfg.gcp_project}/{self.cfg.gcp_region}")
            return True
        except Exception as e:
            logger.error(f"Vertex AI initialization failed: {e}")
            return False

    def create_training_job(
        self,
        image_uri: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a Vertex AI custom training job.

        Args:
            image_uri: Docker image URI (Artifact Registry)
            display_name: optional job display name

        Returns:
            dict with job status
        """
        result = {
            "success": False,
            "job_name": None,
            "status": None,
            "errors": [],
        }

        if not self._init_vertex():
            result["errors"] = self._check_prerequisites()
            return result

        if display_name is None:
            import time
            display_name = f"{self.cfg.model_name}-train-{int(time.time())}"

        try:
            # Build worker pool spec
            worker_pool_spec = [{
                "machine_spec": {
                    "machine_type": self.cfg.vertex_machine_type,
                    "accelerator_type": self.cfg.vertex_accelerator_type,
                    "accelerator_count": self.cfg.vertex_accelerator_count,
                },
                "replica_count": self.cfg.vertex_replica_count,
                "container_spec": {
                    "image_uri": image_uri,
                    "args": [
                        f"--epochs={self.cfg.hyperparameters.get('epochs', '10')}",
                        f"--batch-size={self.cfg.hyperparameters.get('batch-size', '16')}",
                        f"--lr={self.cfg.hyperparameters.get('lr', '3e-4')}",
                    ],
                },
            }]

            job = aiplatform.CustomJob(
                display_name=display_name,
                worker_pool_specs=worker_pool_spec,
                staging_bucket=self.cfg.gcp_staging_bucket,
            )

            # Submit async
            job.run(sync=False)

            result["success"] = True
            result["job_name"] = display_name
            result["status"] = "submitted"
            logger.info(f"Vertex AI training job submitted: {display_name}")

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Vertex AI training job failed: {e}")

        return result

    def upload_model(
        self,
        artifact_uri: str,
        image_uri: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a trained model to Vertex AI Model Registry.

        Args:
            artifact_uri: GCS path to model artifacts (gs://...)
            image_uri: Docker image URI for serving
            display_name: optional model display name

        Returns:
            dict with model resource name
        """
        result = {
            "success": False,
            "model_resource": None,
            "errors": [],
        }

        if not self._init_vertex():
            result["errors"] = self._check_prerequisites()
            return result

        if display_name is None:
            display_name = f"{self.cfg.model_name}-{self.cfg.model_version}"

        try:
            model = aiplatform.Model.upload(
                display_name=display_name,
                artifact_uri=artifact_uri,
                serving_container_image_uri=image_uri,
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                serving_container_ports=[8080],
            )

            result["success"] = True
            result["model_resource"] = model.resource_name
            logger.info(f"Model uploaded to Vertex AI: {model.resource_name}")

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Vertex AI model upload failed: {e}")

        return result

    def deploy_endpoint(
        self,
        model_resource: str,
        endpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deploy model to a Vertex AI Endpoint.

        Args:
            model_resource: model resource name from upload
            endpoint_name: optional endpoint name

        Returns:
            dict with endpoint resource name
        """
        result = {
            "success": False,
            "endpoint_resource": None,
            "errors": [],
        }

        if not self._init_vertex():
            result["errors"] = self._check_prerequisites()
            return result

        if endpoint_name is None:
            endpoint_name = f"{self.cfg.model_name}-endpoint"

        try:
            model = aiplatform.Model(model_resource)

            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
            )

            model.deploy(
                endpoint=endpoint,
                machine_type=self.cfg.vertex_machine_type,
                accelerator_type=self.cfg.vertex_accelerator_type,
                accelerator_count=self.cfg.vertex_accelerator_count,
                min_replica_count=self.cfg.endpoint_min_replicas,
                max_replica_count=self.cfg.endpoint_max_replicas,
                traffic_split={"0": 100},
            )

            result["success"] = True
            result["endpoint_resource"] = endpoint.resource_name
            logger.info(f"Vertex AI endpoint deployed: {endpoint.resource_name}")

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Vertex AI endpoint deployment failed: {e}")

        return result

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Check training job status."""
        if not self._initialized:
            return {"status": "unknown", "error": "Not initialized"}

        try:
            jobs = aiplatform.CustomJob.list(filter=f'display_name="{job_name}"')
            if jobs:
                job = jobs[0]
                return {
                    "status": job.state.name,
                    "resource_name": job.resource_name,
                }
            return {"status": "not_found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# CLOUD DEPLOYMENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class CloudDeploymentOrchestrator:
    """
    Unified orchestrator for cloud deployment.

    Coordinates:
      • Docker image building
      • AWS SageMaker training + deployment
      • Google Vertex AI training + deployment
      • Status tracking across providers
    """

    def __init__(self, cfg: CloudDeployConfig):
        self.cfg = cfg
        self.docker_builder = DockerImageBuilder(cfg)
        self.sagemaker = SageMakerDeployer(cfg)
        self.vertex = VertexAIDeployer(cfg)

    def prepare_docker(self, build: bool = True) -> Dict[str, Any]:
        """Generate Dockerfile and optionally build image."""
        result = {"dockerfile": None, "image_built": False, "image_uri": None}

        # Write Dockerfile
        dockerfile_path = self.docker_builder.write_dockerfile(self.cfg.output_dir)
        result["dockerfile"] = dockerfile_path

        # Build if requested
        if build:
            built = self.docker_builder.build(dockerfile_path)
            result["image_built"] = built
            if built:
                result["image_uri"] = f"{self.cfg.image_name}:{self.cfg.image_tag}"

        return result

    def deploy_to_sagemaker(
        self,
        image_uri: str,
        deploy_endpoint: bool = False,
    ) -> Dict[str, Any]:
        """Deploy to AWS SageMaker."""
        result = {"training": None, "endpoint": None}

        # Training job
        result["training"] = self.sagemaker.create_training_job(image_uri)

        # Endpoint (optional, usually after training completes)
        if deploy_endpoint and result["training"]["success"]:
            model_data = f"s3://{self.cfg.s3_bucket}/sagemaker-output/model.tar.gz"
            result["endpoint"] = self.sagemaker.deploy_endpoint(model_data, image_uri)

        return result

    def deploy_to_vertex(
        self,
        image_uri: str,
        deploy_endpoint: bool = False,
    ) -> Dict[str, Any]:
        """Deploy to Google Vertex AI."""
        result = {"training": None, "model": None, "endpoint": None}

        # Training job
        result["training"] = self.vertex.create_training_job(image_uri)

        # Model upload + endpoint (usually after training completes)
        if deploy_endpoint and result["training"]["success"]:
            artifact_uri = f"{self.cfg.gcp_staging_bucket}/model"
            model_result = self.vertex.upload_model(artifact_uri, image_uri)
            result["model"] = model_result

            if model_result["success"]:
                endpoint_result = self.vertex.deploy_endpoint(model_result["model_resource"])
                result["endpoint"] = endpoint_result

        return result

    def deploy_all(
        self,
        providers: List[str] = ["sagemaker", "vertex"],
        build_docker: bool = True,
        deploy_endpoints: bool = False,
    ) -> Dict[str, Any]:
        """
        Deploy to all requested cloud providers.

        Args:
            providers: list of "sagemaker" and/or "vertex"
            build_docker: whether to build Docker image
            deploy_endpoints: whether to deploy inference endpoints

        Returns:
            dict with deployment results per provider
        """
        report = {
            "docker": None,
            "sagemaker": None,
            "vertex": None,
            "summary": {},
        }

        # Prepare Docker image
        if build_docker:
            report["docker"] = self.prepare_docker(build=True)

        # Get image URI (use local name if not pushed to registry)
        image_uri = self.cfg.image_name + ":" + self.cfg.image_tag

        # Deploy to requested providers
        if "sagemaker" in providers:
            report["sagemaker"] = self.deploy_to_sagemaker(
                image_uri, deploy_endpoint=deploy_endpoints
            )

        if "vertex" in providers:
            report["vertex"] = self.deploy_to_vertex(
                image_uri, deploy_endpoint=deploy_endpoints
            )

        # Summary
        for provider in ["sagemaker", "vertex"]:
            if report[provider]:
                training = report[provider].get("training", {})
                report["summary"][provider] = {
                    "training_submitted": training.get("success", False),
                    "job_name": training.get("job_name"),
                    "errors": training.get("errors", []),
                }

        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Smoke test for ONNX Android + Cloud deployment."""
    print("=" * 72)
    print(" SESI ONNX Android + Cloud Deployment — Smoke Test")
    print("=" * 72)

    device = torch.device("cpu")
    torch.manual_seed(42)

    # ── Test 1: ONNX Android conversion ──────────────────────────────
    print("\n  ── [1] ONNX Android Conversion ──")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 3)
            )
        def forward(self, x):
            return self.net(x)

    model = SimpleModel()
    example_input = torch.randn(1, 20)

    onnx_cfg = ONNXAndroidConfig(
        model_name="sesi_test_model",
        input_shape=[1, 20],
        quantize=False,  # skip quantization for smoke test
        validate_accuracy=False,
        output_dir="/tmp/sesi_onnx_test",
    )

    converter = ONNXAndroidConverter(onnx_cfg)

    if HAS_ONNX:
        report = converter.convert(model, example_input)
        print(f"    Export success: {report['success']}")
        if report['final_path']:
            print(f"    ONNX path: {report['final_path']}")
    else:
        print("    [SKIP] onnxruntime not installed")

    # ── Test 2: Android wrapper generation ───────────────────────────
    print("\n  ── [2] Android Wrapper Generation ──")
    wrapper_gen = AndroidWrapperGenerator(onnx_cfg)
    files = wrapper_gen.generate("sesi_test_model.onnx")
    print(f"    Generated {len(files)} files:")
    for fname, path in files.items():
        print(f"      • {fname}")

    # ── Test 3: Docker image builder ─────────────────────────────────
    print("\n  ── [3] Docker Image Builder ──")
    cloud_cfg = CloudDeployConfig(
        model_name="sesi-test",
        output_dir="/tmp/sesi_cloud_test",
    )
    docker_builder = DockerImageBuilder(cloud_cfg)
    dockerfile = docker_builder.generate_dockerfile()
    print(f"    Dockerfile generated ({len(dockerfile)} chars)")
    print(f"    Base image: {cloud_cfg.base_image}")

    # ── Test 4: Cloud deployer availability ──────────────────────────
    print("\n  ── [4] Cloud Deployer Availability ──")
    print(f"    SageMaker SDK available: {HAS_SAGEMAKER}")
    print(f"    Vertex AI SDK available: {HAS_VERTEX}")

    sm_deployer = SageMakerDeployer(cloud_cfg)
    vertex_deployer = VertexAIDeployer(cloud_cfg)

    sm_issues = sm_deployer._check_prerequisites()
    vertex_issues = vertex_deployer._check_prerequisites()

    print(f"    SageMaker prerequisites: {'OK' if not sm_issues else sm_issues}")
    print(f"    Vertex prerequisites: {'OK' if not vertex_issues else vertex_issues}")

    # ── Test 5: Orchestrator ─────────────────────────────────────────
    print("\n  ── [5] Cloud Deployment Orchestrator ──")
    orchestrator = CloudDeploymentOrchestrator(cloud_cfg)
    print(f"    Orchestrator created: {orchestrator is not None}")
    print(f"    Docker builder: {orchestrator.docker_builder is not None}")
    print(f"    SageMaker deployer: {orchestrator.sagemaker is not None}")
    print(f"    Vertex deployer: {orchestrator.vertex is not None}")

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • ONNX Runtime Android requires API 24+ (Android 7.0)")
    print("  • NNAPI support is device-dependent")
    print("  • Cloud deployment requires valid credentials + billing")
    print("  • GPU/TPU instances incur real costs")
    print("  • Always validate model accuracy after conversion")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
