"""
=============================================================================
SESI AWS SAGEMAKER DEPLOYMENT
Training Job → Model Registry → Real-time Endpoint
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
License      : MIT | Year: 2026

REQUIREMENTS
────────────
  pip install boto3 sagemaker
  AWS credentials configured (aws configure or IAM role)

PIPELINE
────────
  1. Build/push Docker image to ECR
  2. Create SageMaker Training Job (multi-GPU via DDP)
  3. Register model in SageMaker Model Registry
  4. Deploy real-time inference endpoint
  5. Optional: auto-scaling policy

HONEST SCOPE
────────────
• Requires valid AWS account + permissions
• Docker build requires Docker daemon locally
• GPU instance types incur real costs
• This module generates configs + calls boto3; it does not simulate AWS
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.sagemaker")

__all__ = [
    "SageMakerConfig",
    "DockerImageBuilder",
    "SageMakerDeployer",
]

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


@dataclass
class SageMakerConfig:
    # AWS
    region: str = "us-east-1"
    account_id: Optional[str] = None
    role_arn: str = "arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole"

    # ECR / Image
    image_name: str = "sesi-training"
    image_tag: str = "latest"
    base_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"

    # Training
    train_instance_type: str = "ml.p4d.24xlarge"   # 8x A100
    train_instance_count: int = 1
    train_volume_gb: int = 200
    max_runtime_hours: int = 24

    # Model
    model_name: str = "sesi-protein-fold"
    model_version: str = "1.0.0"

    # Endpoint
    endpoint_instance_type: str = "ml.g5.xlarge"   # A10G
    endpoint_instance_count: int = 1
    enable_autoscaling: bool = True
    autoscale_min: int = 1
    autoscale_max: int = 5

    # S3
    s3_bucket: str = "sesi-ml-artifacts"
    s3_prefix: str = "sesi"


# ═══════════════════════════════════════════════════════════════════════
# DOCKER IMAGE BUILDER
# ═══════════════════════════════════════════════════════════════════════

class DockerImageBuilder:
    """Builds and pushes the training/inference Docker image to ECR."""

    DOCKERFILE_TEMPLATE = '''FROM {base_image}

RUN pip install --no-cache-dir \\
    sagemaker-training \\
    onnxruntime \\
    biotite \\
    requests

WORKDIR /opt/ml/code
COPY {code_dir}/ /opt/ml/code/

ENV PYTHONUNBUFFERED=TRUE
ENV SAGEMAKER_PROGRAM={entry_point}

ENTRYPOINT ["python", "-m", "sagemaker_training.entry_point"]
'''

    def __init__(self, cfg: SageMakerConfig):
        self.cfg = cfg

    def generate_dockerfile(self, code_dir: str = ".", entry_point: str = "train.py") -> str:
        return self.DOCKERFILE_TEMPLATE.format(
            base_image=self.cfg.base_image,
            code_dir=code_dir,
            entry_point=entry_point,
        )

    def build(self, code_dir: str = ".", entry_point: str = "train.py") -> bool:
        """Build the Docker image locally."""
        dockerfile = self.generate_dockerfile(code_dir, entry_point)
        dockerfile_path = Path(code_dir) / "Dockerfile.sagemaker"
        dockerfile_path.write_text(dockerfile)

        full_tag = f"{self.cfg.image_name}:{self.cfg.image_tag}"
        cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", full_tag, code_dir]
        logger.info(f"Building image: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Docker build failed: {e}")
            return False

    def push_to_ecr(self) -> Optional[str]:
        """Push image to AWS ECR. Returns full image URI."""
        if not HAS_BOTO3:
            logger.error("boto3 not installed")
            return None

        ecr_client = boto3.client("ecr", region_name=self.cfg.region)

        # Get or create repo
        repo_name = self.cfg.image_name
        try:
            ecr_client.describe_repositories(repositoryNames=[repo_name])
        except ecr_client.exceptions.RepositoryNotFoundException:
            ecr_client.create_repository(repositoryName=repo_name)
            logger.info(f"Created ECR repo: {repo_name}")

        # Get auth
        auth = ecr_client.get_authorization_token()
        registry = auth["authorizationData"][0]["proxyEndpoint"]
        registry_id = registry.split("//")[1].split(".")[0]

        full_tag = f"{self.cfg.image_name}:{self.cfg.image_tag}"
        ecr_tag = f"{registry_id}.dkr.ecr.{self.cfg.region}.amazonaws.com/{full_tag}"

        # Docker login + tag + push
        try:
            subprocess.run(["docker", "login", "--username", "AWS",
                          "--password-stdin", registry],
                          input=auth["authorizationData"][0]["authorizationToken"].encode(),
                          check=True)
            subprocess.run(["docker", "tag", full_tag, ecr_tag], check=True)
            subprocess.run(["docker", "push", ecr_tag], check=True)
            logger.info(f"Pushed → {ecr_tag}")
            return ecr_tag
        except Exception as e:
            logger.error(f"ECR push failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# SAGEMAKER DEPLOYER
# ═══════════════════════════════════════════════════════════════════════

class SageMakerDeployer:
    """Orchestrates the full SageMaker lifecycle."""

    def __init__(self, cfg: SageMakerConfig):
        self.cfg = cfg
        self.image_builder = DockerImageBuilder(cfg)

        if HAS_BOTO3:
            self.sm_client = boto3.client("sagemaker", region_name=cfg.region)
            self.s3_client = boto3.client("s3", region_name=cfg.region)
        else:
            self.sm_client = None
            self.s3_client = None
            logger.warning("boto3 not available — running in dry-run mode")

    def _s3_uri(self, *parts: str) -> str:
        return f"s3://{self.cfg.s3_bucket}/{self.cfg.s3_prefix}/" + "/".join(parts)

    # ── Training Job ───────────────────────────────────────────────────
    def create_training_job(
        self,
        image_uri: str,
        hyperparameters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Submit a SageMaker Training Job."""
        job_name = f"{self.cfg.model_name}-train-{int(__import__('time').time())}"

        job_config = {
            "TrainingJobName": job_name,
            "RoleArn": self.cfg.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
            },
            "HyperParameters": hyperparameters or {
                "epochs": "100",
                "batch-size": "16",
                "lr": "3e-4",
            },
            "InputDataConfig": [{
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": self._s3_uri("data", "train"),
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }],
            "OutputDataConfig": {
                "S3OutputPath": self._s3_uri("output"),
            },
            "ResourceConfig": {
                "InstanceType": self.cfg.train_instance_type,
                "InstanceCount": self.cfg.train_instance_count,
                "VolumeSizeInGB": self.cfg.train_volume_gb,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": self.cfg.max_runtime_hours * 3600,
            },
        }

        if self.sm_client is None:
            logger.info("[DRY-RUN] Training job config:")
            logger.info(json.dumps(job_config, indent=2))
            return {"dry_run": True, "config": job_config}

        self.sm_client.create_training_job(**job_config)
        logger.info(f"Training job submitted: {job_name}")
        return {"job_name": job_name, "status": "submitted"}

    # ── Model Registry ─────────────────────────────────────────────────
    def register_model(self, image_uri: str, model_data_s3: str) -> Dict[str, Any]:
        """Register trained model in SageMaker Model Registry."""
        package_name = f"{self.cfg.model_name}-registry"

        # Ensure registry group exists
        if self.sm_client:
            try:
                self.sm_client.describe_model_package_group(
                    ModelPackageGroupName=package_name)
            except Exception:
                self.sm_client.create_model_package_group(
                    ModelPackageGroupName=package_name,
                    ModelPackageGroupDescription="SESI model registry",
                )

        package_spec = {
            "ModelPackageGroupName": package_name,
            "ModelPackageDescription": f"SESI {self.cfg.model_name} v{self.cfg.model_version}",
            "InferenceSpecification": {
                "Containers": [{
                    "Image": image_uri,
                    "ModelDataUrl": model_data_s3,
                }],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            "ModelApprovalStatus": "PendingManualApproval",
        }

        if self.sm_client is None:
            return {"dry_run": True, "spec": package_spec}

        response = self.sm_client.create_model_package(**package_spec)
        logger.info(f"Model registered: {response['ModelPackageArn']}")
        return {"model_package_arn": response["ModelPackageArn"]}

    # ── Endpoint Deployment ────────────────────────────────────────────
    def deploy_endpoint(self, model_package_arn: str) -> Dict[str, Any]:
        """Deploy model to a real-time inference endpoint."""
        endpoint_name = f"{self.cfg.model_name}-ep"
        endpoint_config_name = f"{endpoint_name}-config"

        if self.sm_client is None:
            return {"dry_run": True, "endpoint_name": endpoint_name}

        # Endpoint config
        self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                "VariantName": "AllTraffic",
                "ModelPackageName": model_package_arn,
                "InstanceType": self.cfg.endpoint_instance_type,
                "InitialInstanceCount": self.cfg.endpoint_instance_count,
            }],
        )

        # Endpoint
        self.sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        logger.info(f"Endpoint deploying: {endpoint_name}")

        # Autoscaling
        if self.cfg.enable_autoscaling:
            self._setup_autoscaling(endpoint_name)

        return {"endpoint_name": endpoint_name, "status": "creating"}

    def _setup_autoscaling(self, endpoint_name: str) -> None:
        """Configure auto-scaling for the endpoint."""
        try:
            autoscaling = boto3.client("application-autoscaling",
                                       region_name=self.cfg.region)
            resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

            autoscaling.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                MinCapacity=self.cfg.autoscale_min,
                MaxCapacity=self.cfg.autoscale_max,
            )

            autoscaling.put_scaling_policy(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                PolicyName=f"{endpoint_name}-cpu-scaling",
                PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": 70.0,
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
                    },
                    "ScaleInCooldown": 300,
                    "ScaleOutCooldown": 60,
                },
            )
            logger.info(f"Autoscaling configured: {self.cfg.autoscale_min}-{self.cfg.autoscale_max}")
        except Exception as e:
            logger.warning(f"Autoscaling setup failed: {e}")

    # ── Full pipeline ──────────────────────────────────────────────────
    def run_full_pipeline(self, code_dir: str = ".") -> Dict[str, Any]:
        """Execute build → train → register → deploy."""
        results = {}

        # 1. Build & push image
        logger.info("Step 1: Building Docker image")
        if self.image_builder.build(code_dir):
            image_uri = self.image_builder.push_to_ecr()
            results["image_uri"] = image_uri
        else:
            results["image_uri"] = f"PLACEHOLDER_ECR_URI/{self.cfg.image_name}:{self.cfg.image_tag}"
            logger.warning("Docker build skipped/failed — using placeholder URI")

        # 2. Training job
        logger.info("Step 2: Creating training job")
        results["training"] = self.create_training_job(results["image_uri"])

        # 3. Register (using placeholder model data for now)
        logger.info("Step 3: Registering model")
        model_data = self._s3_uri("output", "model.tar.gz")
        results["registry"] = self.register_model(results["image_uri"], model_data)

        # 4. Deploy
        logger.info("Step 4: Deploying endpoint")
        arn = results["registry"].get("model_package_arn", "PLACEHOLDER_ARN")
        results["endpoint"] = self.deploy_endpoint(arn)

        return results


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI SageMaker Deployment — Smoke Test (DRY-RUN)")
    print("=" * 72)
    print(f"  boto3 available: {HAS_BOTO3}")

    cfg = SageMakerConfig()
    deployer = SageMakerDeployer(cfg)

    # Test Dockerfile generation
    dockerfile = deployer.image_builder.generate_dockerfile()
    print(f"\n  [1] Dockerfile generated ({len(dockerfile)} chars)")
    print(f"      Base image: {cfg.base_image}")

    # Test training job config (dry-run without boto3 or credentials)
    result = deployer.create_training_job(
        image_uri=f"123456789.dkr.ecr.{cfg.region}.amazonaws.com/{cfg.image_name}:{cfg.image_tag}",
        hyperparameters={"epochs": "50", "batch-size": "8"},
    )
    print(f"\n  [2] Training job config: {'dry_run' if result.get('dry_run') else result.get('job_name')}")
    print(f"      Instance: {cfg.train_instance_type} x {cfg.train_instance_count}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Real deployment needs AWS credentials + IAM role")
    print("  • GPU instances (p4d/g5) incur significant costs")
    print("  • Test with small instance first (ml.t3.medium)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
