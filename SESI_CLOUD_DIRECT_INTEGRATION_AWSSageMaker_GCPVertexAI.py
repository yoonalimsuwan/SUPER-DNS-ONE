"""
=============================================================================
SESI CLOUD DIRECT INTEGRATION — AWS SageMaker × GCP Vertex AI
Direct cloud provider connection without intermediate layer
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Direct Cloud Provider Integration)

WHAT THIS MODULE PROVIDES
─────────────────────────
[AWS] SageMakerDirectIntegration
  • Training Job (PyTorchEstimator)
  • Model Registry (ModelPackage)
  • Real-time Endpoint + Autoscaling
  • Batch Transform
  • Processing Job

[GCP] VertexAIDirectIntegration
  • CustomTrainingJob (GPU/TPU)
  • Model Upload + Registry
  • Endpoint + Autoscaling
  • Batch Prediction
  • Pipelines (optional)

HONEST SCOPE
────────────
• Requires valid AWS/GCP credentials and billing enabled
• GPU instances incur real costs — test with small instances first
• Training jobs run asynchronously; status polling required
• Model artifacts stored in S3/GCS
• No simulation — this calls real cloud APIs
• IAM/Service Account permissions must be configured separately
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("sesi.cloud_direct")

__all__ = [
    "SageMakerDirectConfig",
    "SageMakerDirectIntegration",
    "VertexAIDirectConfig",
    "VertexAIDirectIntegration",
]

# ── Optional cloud SDK imports ────────────────────────────────────────
try:
    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch as SageMakerPyTorch
    from sagemaker.model import Model as SageMakerModel
    from sagemaker.transformer import Transformer
    HAS_SAGEMAKER = True
except ImportError:
    HAS_SAGEMAKER = False

try:
    from google.cloud import aiplatform as vertex_ai
    from google.cloud.aiplatform import CustomTrainingJob, Model, Endpoint
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False


# ═══════════════════════════════════════════════════════════════════════
# AWS SAGEMAKER DIRECT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SageMakerDirectConfig:
    """Configuration for AWS SageMaker direct integration."""
    
    # ── AWS account ───────────────────────────────────────────────────
    region: str = "ap-southeast-1"  # Singapore
    role_arn: str = ""  # arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
    s3_bucket: str = ""  # S3 bucket for artifacts
    profile: Optional[str] = None  # AWS CLI profile name
    
    # ── Training ──────────────────────────────────────────────────────
    instance_type: str = "ml.g5.xlarge"  # A10G GPU
    instance_count: int = 1
    volume_size_gb: int = 100
    max_run_seconds: int = 3600 * 8  # 8 hours
    
    # ── Image ─────────────────────────────────────────────────────────
    image_uri: Optional[str] = None  # Custom image; None = SageMaker PyTorch DLC
    framework_version: str = "2.1.0"
    py_version: str = "py310"
    
    # ── Model ─────────────────────────────────────────────────────────
    model_name: str = "sesi-protein-fold"
    model_version: str = "1.0.0"
    
    # ── Endpoint ──────────────────────────────────────────────────────
    endpoint_instance_type: str = "ml.g5.xlarge"
    endpoint_initial_count: int = 1
    enable_autoscaling: bool = True
    autoscale_min: int = 1
    autoscale_max: int = 5
    
    # ── Hyperparameters ───────────────────────────────────────────────
    hyperparameters: Dict[str, str] = field(default_factory=lambda: {
        "epochs": "50",
        "batch-size": "16",
        "lr": "3e-4",
    })


class SageMakerDirectIntegration:
    """
    Direct AWS SageMaker integration without intermediate wrappers.
    
    Uses sagemaker Python SDK for:
    • Training: PyTorchEstimator
    • Model: sagemaker.model.Model
    • Endpoint: Real-time inference with autoscaling
    • Batch: Batch Transform for offline inference
    """
    
    def __init__(self, cfg: SageMakerDirectConfig):
        self.cfg = cfg
        self._session = None
        self._sm_client = None
        
        if not HAS_SAGEMAKER:
            logger.warning(
                "sagemaker SDK not installed. Install with: "
                "pip install sagemaker boto3"
            )
    
    @property
    def session(self):
        """Lazy-init SageMaker session."""
        if self._session is None:
            if not HAS_SAGEMAKER:
                raise RuntimeError("sagemaker SDK not available")
            
            if self.cfg.profile:
                import boto3
                boto_session = boto3.Session(profile_name=self.cfg.profile,
                                              region_name=self.cfg.region)
            else:
                boto_session = boto3.Session(region_name=self.cfg.region)
            
            self._session = sagemaker.Session(boto_session=boto_session)
            self._sm_client = boto_session.client("sagemaker")
        
        return self._session
    
    # ── Training Job ──────────────────────────────────────────────────
    
    def create_training_job(
        self,
        entry_point: str,
        source_dir: str,
        requirements: Optional[List[str]] = None,
        use_spot: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a SageMaker training job.
        
        Args:
            entry_point: Training script filename (e.g., "train.py")
            source_dir: Directory containing training code
            requirements: Optional list of pip requirements
            use_spot: Use spot instances for cost savings
            
        Returns:
            dict with job name and status
        """
        if not HAS_SAGEMAKER:
            return {"error": "sagemaker SDK not installed", "dry_run": True}
        
        job_name = f"{self.cfg.model_name}-train-{int(time.time())}"
        
        # Use SageMaker PyTorch DLC if no custom image
        estimator_kwargs = {
            "entry_point": entry_point,
            "source_dir": source_dir,
            "role": self.cfg.role_arn,
            "instance_count": self.cfg.instance_count,
            "instance_type": self.cfg.instance_type,
            "framework_version": self.cfg.framework_version,
            "py_version": self.cfg.py_version,
            "hyperparameters": self.cfg.hyperparameters,
            "output_path": f"s3://{self.cfg.s3_bucket}/sagemaker-output",
            "code_location": f"s3://{self.cfg.s3_bucket}/sagemaker-code",
            "volume_size": self.cfg.volume_size_gb,
            "max_run": self.cfg.max_run_seconds,
            "sagemaker_session": self.session,
            "base_job_name": self.cfg.model_name,
        }
        
        if requirements:
            estimator_kwargs["requirements"] = requirements
        
        if self.cfg.image_uri:
            estimator_kwargs["image_uri"] = self.cfg.image_uri
            estimator = sagemaker.estimator.Estimator(**estimator_kwargs)
        else:
            estimator = SageMakerPyTorch(**estimator_kwargs)
        
        # Spot instances for cost savings
        if use_spot:
            estimator.max_wait = self.cfg.max_run_seconds * 2
            estimator.use_spot_instances = True
            estimator.checkpoint_s3_uri = f"s3://{self.cfg.s3_bucket}/checkpoints"
        
        # Submit training job
        logger.info(f"Submitting training job: {job_name}")
        estimator.fit(wait=False)
        
        return {
            "job_name": estimator.latest_training_job.name,
            "status": "submitted",
            "instance_type": self.cfg.instance_type,
            "instance_count": self.cfg.instance_count,
        }
    
    def get_training_status(self, job_name: str) -> Dict[str, Any]:
        """Get training job status."""
        if not self._sm_client:
            return {"error": "No session"}
        
        try:
            response = self._sm_client.describe_training_job(
                TrainingJobName=job_name
            )
            return {
                "status": response["TrainingJobStatus"],
                "secondary_status": response.get("SecondaryStatus", ""),
                "failure_reason": response.get("FailureReason", ""),
                "model_artifacts": response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ── Model Registration ────────────────────────────────────────────
    
    def register_model(
        self,
        model_artifact_s3: str,
        image_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register trained model in SageMaker Model Registry.
        
        Args:
            model_artifact_s3: S3 URI of model.tar.gz
            image_uri: Inference image URI (optional)
        """
        if not HAS_SAGEMAKER:
            return {"error": "sagemaker SDK not installed", "dry_run": True}
        
        package_group = f"{self.cfg.model_name}-registry"
        
        # Create model package group if not exists
        try:
            self._sm_client.describe_model_package_group(
                ModelPackageGroupName=package_group
            )
        except Exception:
            self._sm_client.create_model_package_group(
                ModelPackageGroupName=package_group,
                ModelPackageGroupDescription="SESI model registry",
            )
        
        # Determine image URI
        if image_uri is None:
            from sagemaker import image_uris
            image_uri = image_uris.retrieve(
                framework="pytorch",
                region=self.cfg.region,
                version=self.cfg.framework_version,
                image_scope="inference",
            )
        
        # Create model package
        response = self._sm_client.create_model_package(
            ModelPackageGroupName=package_group,
            ModelPackageDescription=f"SESI {self.cfg.model_name} v{self.cfg.model_version}",
            InferenceSpecification={
                "Containers": [{
                    "Image": image_uri,
                    "ModelDataUrl": model_artifact_s3,
                }],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            ModelApprovalStatus="PendingManualApproval",
        )
        
        return {
            "model_package_arn": response["ModelPackageArn"],
            "package_group": package_group,
        }
    
    # ── Endpoint Deployment ───────────────────────────────────────────
    
    def deploy_endpoint(
        self,
        model_package_arn: str,
    ) -> Dict[str, Any]:
        """
        Deploy model to real-time endpoint.
        
        Args:
            model_package_arn: ARN of registered model package
        """
        if not self._sm_client:
            return {"error": "No session"}
        
        endpoint_name = f"{self.cfg.model_name}-endpoint"
        endpoint_config_name = f"{endpoint_name}-config"
        
        try:
            # Create endpoint config
            self._sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    "VariantName": "AllTraffic",
                    "ModelPackageName": model_package_arn,
                    "InstanceType": self.cfg.endpoint_instance_type,
                    "InitialInstanceCount": self.cfg.endpoint_initial_count,
                }],
            )
            
            # Create endpoint
            self._sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
            
            # Setup autoscaling if enabled
            if self.cfg.enable_autoscaling:
                self._setup_autoscaling(endpoint_name)
            
            return {
                "endpoint_name": endpoint_name,
                "status": "creating",
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _setup_autoscaling(self, endpoint_name: str) -> None:
        """Configure target tracking autoscaling."""
        try:
            autoscaling = boto3.client(
                "application-autoscaling",
                region_name=self.cfg.region,
            )
            
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
                PolicyName=f"{endpoint_name}-scaling",
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
    
    # ── Invoke Endpoint ───────────────────────────────────────────────
    
    def invoke_endpoint(
        self,
        payload: Dict[str, Any],
        endpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke real-time endpoint for inference."""
        endpoint_name = endpoint_name or f"{self.cfg.model_name}-endpoint"
        
        try:
            runtime = boto3.client("sagemaker-runtime", region_name=self.cfg.region)
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            return json.loads(response["Body"].read().decode())
        except Exception as e:
            return {"error": str(e)}
    
    # ── Batch Transform ───────────────────────────────────────────────
    
    def batch_transform(
        self,
        model_name: str,
        input_s3: str,
        output_s3: str,
    ) -> Dict[str, Any]:
        """Run batch transform job for offline inference."""
        if not HAS_SAGEMAKER:
            return {"error": "sagemaker SDK not installed"}
        
        transformer = Transformer(
            model_name=model_name,
            instance_count=1,
            instance_type="ml.m5.large",
            output_path=output_s3,
            sagemaker_session=self.session,
        )
        
        transformer.transform(
            data=input_s3,
            content_type="application/json",
            split_type="Line",
            wait=False,
        )
        
        return {
            "job_name": transformer.latest_transform_job.name,
            "status": "submitted",
        }


# ═══════════════════════════════════════════════════════════════════════
# GCP VERTEX AI DIRECT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VertexAIDirectConfig:
    """Configuration for GCP Vertex AI direct integration."""
    
    # ── GCP project ───────────────────────────────────────────────────
    project_id: str = ""
    location: str = "asia-southeast1"  # Singapore
    staging_bucket: str = ""  # gs://bucket-name
    
    # ── Training ──────────────────────────────────────────────────────
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 1
    
    # ── TPU option ────────────────────────────────────────────────────
    use_tpu: bool = False
    tpu_type: str = "tpu-v3-8"
    
    # ── Image ─────────────────────────────────────────────────────────
    image_uri: Optional[str] = None  # Custom container; None = Vertex prebuilt
    training_script: str = "train.py"
    
    # ── Model ─────────────────────────────────────────────────────────
    model_name: str = "sesi-protein-fold"
    model_version: str = "1.0.0"
    
    # ── Endpoint ──────────────────────────────────────────────────────
    endpoint_machine_type: str = "n1-standard-4"
    endpoint_accelerator: str = "NVIDIA_TESLA_T4"
    endpoint_accelerator_count: int = 1
    min_replicas: int = 1
    max_replicas: int = 5
    
    # ── Hyperparameters ───────────────────────────────────────────────
    hyperparameters: Dict[str, str] = field(default_factory=lambda: {
        "epochs": "50",
        "batch-size": "16",
        "lr": "3e-4",
    })


class VertexAIDirectIntegration:
    """
    Direct GCP Vertex AI integration.
    
    Uses google-cloud-aiplatform SDK for:
    • Training: CustomTrainingJob (GPU/TPU)
    • Model: Model Upload + Registry
    • Endpoint: Real-time inference with autoscaling
    • Batch: Batch Prediction
    """
    
    def __init__(self, cfg: VertexAIDirectConfig):
        self.cfg = cfg
        self._initialized = False
        
        if not HAS_VERTEX:
            logger.warning(
                "google-cloud-aiplatform not installed. Install with: "
                "pip install google-cloud-aiplatform"
            )
    
    def _init_vertex(self) -> bool:
        """Initialize Vertex AI SDK."""
        if self._initialized:
            return True
        
        if not HAS_VERTEX:
            logger.error("google-cloud-aiplatform not installed")
            return False
        
        if not self.cfg.project_id:
            logger.error("project_id not configured")
            return False
        
        try:
            vertex_ai.init(
                project=self.cfg.project_id,
                location=self.cfg.location,
                staging_bucket=self.cfg.staging_bucket,
            )
            self._initialized = True
            logger.info(f"Vertex AI initialized: {self.cfg.project_id}")
            return True
        except Exception as e:
            logger.error(f"Vertex AI init failed: {e}")
            return False
    
    # ── Training Job ──────────────────────────────────────────────────
    
    def create_training_job(
        self,
        script_path: str,
        container_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create Vertex AI custom training job.
        
        Args:
            script_path: Path to training script
            container_uri: Custom container URI (optional)
        """
        if not self._init_vertex():
            return {"error": "Vertex AI not initialized", "dry_run": True}
        
        job_display_name = f"{self.cfg.model_name}-train-{int(time.time())}"
        
        # Build machine spec
        if self.cfg.use_tpu:
            machine_spec = {
                "machine_type": "cloud-tpu",
                "accelerator_type": self.cfg.tpu_type.upper(),
                "accelerator_count": 1,
            }
        else:
            machine_spec = {
                "machine_type": self.cfg.machine_type,
                "accelerator_type": self.cfg.accelerator_type,
                "accelerator_count": self.cfg.accelerator_count,
            }
        
        # Worker pool spec
        worker_pool_spec = {
            "machine_spec": machine_spec,
            "replica_count": self.cfg.replica_count,
            "container_spec": {
                "image_uri": container_uri or self._get_prebuilt_image(),
                "command": ["python", script_path],
                "args": [f"--{k}={v}" for k, v in self.cfg.hyperparameters.items()],
            },
        }
        
        try:
            job = vertex_ai.CustomJob(
                display_name=job_display_name,
                worker_pool_specs=[worker_pool_spec],
                staging_bucket=self.cfg.staging_bucket,
            )
            
            job.run(sync=False)
            
            return {
                "job_name": job.resource_name,
                "status": "submitted",
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_prebuilt_image(self) -> str:
        """Get prebuilt Vertex AI container URI."""
        # Default PyTorch image
        return f"{self.cfg.location}-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest"
    
    def get_training_status(self, job_resource: str) -> Dict[str, Any]:
        """Get training job status."""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        try:
            job = vertex_ai.CustomJob.get(job_resource)
            return {
                "status": job.state.name,
                "resource_name": job.resource_name,
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ── Model Upload ──────────────────────────────────────────────────
    
    def upload_model(
        self,
        artifact_uri: str,
        serving_image: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload trained model to Vertex AI Model Registry.
        
        Args:
            artifact_uri: GCS URI of model artifacts
            serving_image: Custom serving container (optional)
        """
        if not self._init_vertex():
            return {"error": "Vertex AI not initialized"}
        
        display_name = f"{self.cfg.model_name}-{self.cfg.model_version}"
        
        # Use TorchServe image if no custom
        if serving_image is None:
            serving_image = f"{self.cfg.location}-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest"
        
        try:
            model = vertex_ai.Model.upload(
                display_name=display_name,
                artifact_uri=artifact_uri,
                serving_container_image_uri=serving_image,
                serving_container_predict_route="/v1/models/model:predict",
                serving_container_health_route="/v1/models/model:health",
                serving_container_ports=[8080],
            )
            
            return {
                "model_resource": model.resource_name,
                "display_name": display_name,
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ── Endpoint Deployment ───────────────────────────────────────────
    
    def deploy_endpoint(
        self,
        model_resource: str,
    ) -> Dict[str, Any]:
        """
        Deploy model to Vertex AI Endpoint.
        
        Args:
            model_resource: Model resource name
        """
        if not self._initialized:
            return {"error": "Not initialized"}
        
        endpoint_name = f"{self.cfg.model_name}-endpoint"
        
        try:
            # Create endpoint
            endpoint = vertex_ai.Endpoint.create(
                display_name=endpoint_name,
            )
            
            # Get model
            model = vertex_ai.Model(model_resource)
            
            # Deploy
            model.deploy(
                endpoint=endpoint,
                machine_type=self.cfg.endpoint_machine_type,
                accelerator_type=self.cfg.endpoint_accelerator,
                accelerator_count=self.cfg.endpoint_accelerator_count,
                min_replica_count=self.cfg.min_replicas,
                max_replica_count=self.cfg.max_replicas,
                traffic_split={"0": 100},
            )
            
            return {
                "endpoint_resource": endpoint.resource_name,
                "status": "deployed",
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ── Predict ───────────────────────────────────────────────────────
    
    def predict(
        self,
        endpoint_resource: str,
        instances: List[Dict],
    ) -> Dict[str, Any]:
        """Invoke endpoint for prediction."""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        try:
            endpoint = vertex_ai.Endpoint(endpoint_resource)
            response = endpoint.predict(instances=instances)
            return {"predictions": response.predictions}
        except Exception as e:
            return {"error": str(e)}
    
    # ── Batch Prediction ──────────────────────────────────────────────
    
    def batch_predict(
        self,
        model_resource: str,
        gcs_source: str,
        gcs_destination: str,
    ) -> Dict[str, Any]:
        """Run batch prediction job."""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        try:
            model = vertex_ai.Model(model_resource)
            
            batch_job = model.batch_predict(
                job_display_name=f"{self.cfg.model_name}-batch-{int(time.time())}",
                gcs_source=[gcs_source],
                gcs_destination_prefix=gcs_destination,
                instances_format="jsonl",
                machine_type=self.cfg.machine_type,
                starting_replica_count=1,
                max_replica_count=3,
                sync=False,
            )
            
            return {
                "job_resource": batch_job.resource_name,
                "status": "submitted",
            }
        except Exception as e:
            return {"error": str(e)}
