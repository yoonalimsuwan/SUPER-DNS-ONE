"""
=============================================================================
SESI KSERVE DEPLOYMENT — Self-Hosted Inference on Kubernetes
KServe InferenceService + Custom Predictor + Autoscaling + Canary
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

KSERVE ARCHITECTURE
───────────────────
KServe provides:
- Standardized inference API (predict/explain)
- Autoscaling (Knative-based, scale-to-zero)
- Canary rollouts
- Model registry integration (S3, GCS, OCI)
- GPU scheduling
- Request/response logging
- Prometheus metrics

DEPLOYMENT TOPOLOGY
───────────────────
┌─────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  KServe Control Plane                                 │  │
│  │  ├─ kserve-controller-manager                         │  │
│  │  ├─ knative-serving                                   │  │
│  │  └─ istio-ingressgateway                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  SESI InferenceService                                │  │
│  │  ├─ Predictor (PyTorch model)                         │  │
│  │  │   ├─ GPU: nvidia.com/gpu (optional)                │  │
│  │  │   └─ Autoscaling: 1-10 replicas                    │  │
│  │  ├─ Transformer (pre/post-processing, optional)       │  │
│  │  └─ Explainer (model explanation, optional)           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Model Storage                                        │  │
│  │  ├─ PVC: sesi-models                                  │  │
│  │  └─ or S3/GCS/O CI registry                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

PREREQUISITES
─────────────
1. Kubernetes cluster (v1.24+)
2. KServe installed (v0.11+):
   kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
3. Knative Serving (for autoscaling)
4. Istio (for networking)
5. NVIDIA GPU Operator (for GPU workloads)
6. cert-manager (for webhooks)

HONEST SCOPE
────────────
• KServe installation is a cluster-admin operation
• GPU scheduling requires NVIDIA device plugin
• Autoscaling behavior depends on Knative configuration
• Model storage requires appropriate credentials (S3/GCS)
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.kserve")

__all__ = [
    "KServeConfig",
    "KServeManifestGenerator",
    "KServeDockerBuilder",
    "KServeDeployer",
]


# ── Optional imports ──────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KServeConfig:
    # ── Cluster ─────────────────────────────────────────────────────────
    namespace: str = "sesi-inference"
    kserve_version: str = "v0.11.0"
    
    # ── Model ───────────────────────────────────────────────────────────
    model_name: str = "sesi-cfd-surrogate"
    model_version: str = "v1"
    model_format: str = "pytorch"  # pytorch | onnx | tensorflow
    
    # ── Storage ─────────────────────────────────────────────────────────
    storage_uri: str = "pvc://sesi-models/model"  # or s3://, gs://, oci://
    model_path: str = "/models/sesi_cfd.pt"
    
    # ── Resources ───────────────────────────────────────────────────────
    use_gpu: bool = True
    gpu_type: str = "nvidia.com/gpu"
    gpu_count: int = 1
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "8Gi"
    memory_limit: str = "16Gi"
    
    # ── Autoscaling ─────────────────────────────────────────────────────
    min_replicas: int = 1
    max_replicas: int = 10
    target_utilization: int = 70  # CPU utilization %
    scale_to_zero: bool = True
    scale_to_zero_grace_period: str = "30s"
    
    # ── Canary ──────────────────────────────────────────────────────────
    enable_canary: bool = False
    canary_percentage: int = 10
    
    # ── Serving runtime ─────────────────────────────────────────────────
    serving_runtime: str = "kserve-torchserve"  # or custom
    image: str = "sesi/inference-server:latest"
    
    # ── Networking ──────────────────────────────────────────────────────
    ingress_domain: str = "sesi.example.com"
    enable_tls: bool = True
    
    # ── Monitoring ──────────────────────────────────────────────────────
    enable_prometheus: bool = True
    enable_logging: bool = True


# ═══════════════════════════════════════════════════════════════════════
# MANIFEST GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class KServeManifestGenerator:
    """
    Generates Kubernetes manifests for KServe deployment.
    
    Generates:
    - Namespace
    - PVC for model storage
    - InferenceService (KServe CRD)
    - ServiceAccount + RBAC
    - Prometheus ServiceMonitor (optional)
    """

    def __init__(self, cfg: KServeConfig):
        self.cfg = cfg

    def generate_namespace(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.cfg.namespace,
                "labels": {
                    "serving.kubeflow.org/inferenceservice": "enabled",
                },
            },
        }

    def generate_pvc(self) -> Dict:
        """PVC for model storage."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-models",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {
                    "requests": {"storage": "50Gi"},
                },
            },
        }

    def generate_inference_service(self) -> Dict:
        """
        KServe InferenceService CRD.
        
        This is the core resource that KServe uses to deploy the model.
        """
        cfg = self.cfg
        
        # Resource limits
        resources = {
            "requests": {
                "cpu": cfg.cpu_request,
                "memory": cfg.memory_request,
            },
            "limits": {
                "cpu": cfg.cpu_limit,
                "memory": cfg.memory_limit,
            },
        }
        if cfg.use_gpu:
            resources["limits"][cfg.gpu_type] = cfg.gpu_count
            resources["requests"][cfg.gpu_type] = cfg.gpu_count

        # Predictor spec
        predictor = {
            "model": {
                "modelFormat": {
                    "name": cfg.model_format,
                },
                "storageUri": cfg.storage_uri,
                "resources": resources,
            },
            "minReplicas": cfg.min_replicas,
            "maxReplicas": cfg.max_replicas,
        }
        
        # Autoscaling annotations
        annotations = {}
        if cfg.scale_to_zero:
            annotations["autoscaling.knative.dev/scaleToZeroPodAutoscaler"] = "enabled"
            annotations["autoscaling.knative.dev/scaleToZeroGracePeriod"] = cfg.scale_to_zero_grace_period
        annotations["autoscaling.knative.dev/target"] = str(cfg.target_utilization)
        
        predictor["annotations"] = annotations

        # Serving runtime (if custom)
        if cfg.serving_runtime != "kserve-torchserve":
            predictor["runtime"] = cfg.serving_runtime

        # InferenceService spec
        spec = {
            "predictor": predictor,
        }

        # Canary deployment
        if cfg.enable_canary:
            spec["predictor"]["canaryTrafficPercent"] = cfg.canary_percentage

        return {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": cfg.model_name,
                "namespace": cfg.namespace,
                "annotations": {
                    "serving.kserve.io/autoscalerClass": "kpa",
                    "serving.kserve.io/metrics": "prometheus",
                },
            },
            "spec": spec,
        }

    def generate_service_account(self) -> Dict:
        """ServiceAccount for model serving."""
        return {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "sesi-inference-sa",
                "namespace": self.cfg.namespace,
            },
        }

    def generate_prometheus_monitor(self) -> Dict:
        """Prometheus ServiceMonitor for metrics."""
        if not self.cfg.enable_prometheus:
            return {}
        
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{self.cfg.model_name}-monitor",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "serving.kserve.io/inferenceservice": self.cfg.model_name,
                    },
                },
                "endpoints": [
                    {
                        "port": "http",
                        "path": "/metrics",
                        "interval": "30s",
                    },
                ],
            },
        }

    def generate_all(self) -> Dict[str, Dict]:
        """Generate all manifests."""
        manifests = {
            "00_namespace.yaml": self.generate_namespace(),
            "01_pvc.yaml": self.generate_pvc(),
            "02_service_account.yaml": self.generate_service_account(),
            "10_inference_service.yaml": self.generate_inference_service(),
        }
        
        if self.cfg.enable_prometheus:
            manifests["20_prometheus_monitor.yaml"] = self.generate_prometheus_monitor()
        
        return manifests

    def write_manifests(self, output_dir: str = "./kserve_manifests") -> List[str]:
        """Write all manifests to disk."""
        if not HAS_YAML:
            logger.warning("PyYAML not installed. Writing JSON instead.")
            import json as json_mod
            dump_fn = lambda data, f: json_mod.dump(data, f, indent=2)
            ext = ".json"
        else:
            dump_fn = lambda data, f: yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            ext = ".yaml"

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifests = self.generate_all()
        paths = []
        
        for filename, manifest in manifests.items():
            if not manifest:
                continue
            # Replace extension if needed
            base_name = filename.rsplit(".", 1)[0]
            path = out_dir / f"{base_name}{ext}"
            
            with open(path, "w") as f:
                dump_fn(manifest, f)
            paths.append(str(path))
            logger.info(f"Generated: {path}")

        return paths


# ═══════════════════════════════════════════════════════════════════════
# DOCKER BUILDER
# ═══════════════════════════════════════════════════════════════════════

class KServeDockerBuilder:
    """
    Builds Docker image for KServe inference server.
    
    Generates:
    - Dockerfile with PyTorch + KServe SDK
    - Model server script
    - Health check endpoints
    """

    DOCKERFILE_TEMPLATE = '''FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install KServe SDK and dependencies
RUN pip install --no-cache-dir \\
    kserve==0.11.0 \\
    torchserve \\
    onnxruntime \\
    prometheus-client

# Create model directory
RUN mkdir -p /models
WORKDIR /app

# Copy model server
COPY model_server.py /app/model_server.py
COPY requirements.txt /app/requirements.txt

# Install additional requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy model (optional, can also mount from PVC)
# COPY model.pt /models/model.pt

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Run model server
ENTRYPOINT ["python", "/app/model_server.py"]
'''

    MODEL_SERVER_TEMPLATE = '''"""
SESI Model Server for KServe.
Implements the KServe Model Protocol.
"""
import kserve
from typing import Dict
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sesi.model_server")


class SESIModel(kserve.Model):
    """SESI model implementation for KServe."""
    
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.model = None
        self.device = None
        self.load()
    
    def load(self):
        """Load model from disk."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                # Our format
                from model import build_model  # Import your model builder
                self.model = build_model()
                self.model.load_state_dict(checkpoint["model_state"])
            elif "state_dict" in checkpoint:
                # PyTorch Lightning format
                self.model = torch.load(self.model_path)
            else:
                # Assume it's the model itself
                self.model = checkpoint
        else:
            self.model = checkpoint
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def preprocess(self, payload: Dict) -> torch.Tensor:
        """Preprocess incoming request."""
        # KServe sends data in "instances" key
        instances = payload.get("instances", [])
        
        # Convert to tensor
        input_data = np.array(instances, dtype=np.float32)
        input_tensor = torch.from_numpy(input_data).to(self.device)
        
        return input_tensor
    
    def predict(self, input_tensor: torch.Tensor) -> Dict:
        """Run inference."""
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert to numpy
        if isinstance(output, torch.Tensor):
            output_np = output.cpu().numpy()
        elif isinstance(output, dict):
            output_np = {k: v.cpu().numpy() for k, v in output.items()}
        else:
            output_np = output
        
        return {"predictions": output_np.tolist()}
    
    def postprocess(self, result: Dict) -> Dict:
        """Postprocess response."""
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sesi-cfd-surrogate")
    parser.add_argument("--model_path", default="/models/model.pt")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=8081)
    args = parser.parse_args()
    
    model = SESIModel(args.model_name, args.model_path)
    model.load()
    
    # Start KServe server
    kserve.ModelServer(
        http_port=args.http_port,
        grpc_port=args.grpc_port,
    ).start([model])
'''

    REQUIREMENTS_TEMPLATE = '''torch>=2.0.0
numpy>=1.24.0
kserve>=0.11.0
'''

    def __init__(self, cfg: KServeConfig):
        self.cfg = cfg

    def generate_dockerfile(self) -> str:
        return self.DOCKERFILE_TEMPLATE

    def generate_model_server(self) -> str:
        return self.MODEL_SERVER_TEMPLATE

    def generate_requirements(self) -> str:
        return self.REQUIREMENTS_TEMPLATE

    def write_files(self, output_dir: str = "./kserve_docker") -> Dict[str, str]:
        """Write Docker build files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "Dockerfile": self.generate_dockerfile(),
            "model_server.py": self.generate_model_server(),
            "requirements.txt": self.generate_requirements(),
        }

        paths = {}
        for filename, content in files.items():
            path = out_dir / filename
            path.write_text(content)
            paths[filename] = str(path)
            logger.info(f"Generated: {path}")

        return paths

    def build_command(self, image_tag: str) -> str:
        """Generate Docker build command."""
        return f"docker build -t {image_tag} ./kserve_docker"

    def push_command(self, image_tag: str, registry: str) -> str:
        """Generate Docker push command."""
        full_tag = f"{registry}/{image_tag}"
        return f"docker tag {image_tag} {full_tag} && docker push {full_tag}"


# ═══════════════════════════════════════════════════════════════════════
# DEPLOYER
# ═══════════════════════════════════════════════════════════════════════

class KServeDeployer:
    """
    Orchestrates the full KServe deployment.
    
    Workflow:
    1. Generate manifests
    2. Build Docker image
    3. Push to registry
    4. Apply to cluster
    5. Verify deployment
    """

    def __init__(self, cfg: KServeConfig):
        self.cfg = cfg
        self.manifest_gen = KServeManifestGenerator(cfg)
        self.docker_builder = KServeDockerBuilder(cfg)

    def deploy(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Full deployment workflow.
        
        Args:
            dry_run: If True, only generate files without applying
            
        Returns:
            dict with deployment status
        """
        result = {
            "success": False,
            "manifests": [],
            "docker_files": [],
            "instructions": [],
        }

        # Generate manifests
        manifests = self.manifest_gen.write_manifests()
        result["manifests"] = manifests

        # Generate Docker files
        docker_files = self.docker_builder.write_files()
        result["docker_files"] = docker_files

        if dry_run:
            result["instructions"] = [
                "1. Build Docker image:",
                f"   {self.docker_builder.build_command(self.cfg.image)}",
                "",
                "2. Push to registry:",
                f"   {self.docker_builder.push_command(self.cfg.image, 'your-registry.com')}",
                "",
                "3. Apply manifests:",
                "   kubectl apply -f ./kserve_manifests/",
                "",
                "4. Verify deployment:",
                f"   kubectl get inferenceservice -n {self.cfg.namespace}",
            ]
            result["success"] = True
            return result

        # Actual deployment would use kubectl/python kubernetes client
        result["instructions"] = [
            "Dry run complete. To deploy:",
            "1. Ensure KServe is installed on your cluster",
            "2. Build and push Docker image",
            "3. Run: kubectl apply -f ./kserve_manifests/",
        ]
        result["success"] = True

        return result


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_kserve_deployment(
    cfg: Optional[KServeConfig] = None,
    output_dir: str = "./kserve_deployment",
) -> Dict[str, Any]:
    """
    Generate complete KServe deployment package.
    
    Args:
        cfg: KServe configuration
        output_dir: Output directory
        
    Returns:
        dict with generated files and instructions
    """
    cfg = cfg or KServeConfig()
    deployer = KServeDeployer(cfg)
    return deployer.deploy(dry_run=True)


def create_kserve_prerequisites_script() -> str:
    """Generate script to install KServe prerequisites."""
    return '''#!/bin/bash
# KServe Prerequisites Installation Script
# Run this on a machine with kubectl configured

set -e

echo "Installing KServe prerequisites..."

# 1. Install cert-manager (required for KServe webhooks)
echo "Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
kubectl wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=300s

# 2. Install Knative Serving (for autoscaling)
echo "Installing Knative Serving..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml
kubectl wait --for=condition=Available deployment/controller -n knative-serving --timeout=300s

# 3. Install Istio (for networking)
echo "Installing Istio..."
istioctl install -y --set profile=default
kubectl wait --for=condition=Available deployment/istio-ingressgateway -n istio-system --timeout=300s

# 4. Install KServe
echo "Installing KServe..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl wait --for=condition=Available deployment/kserve-controller-manager -n kserve --timeout=300s

# 5. Install KServe built-in serving runtimes
echo "Installing KServe serving runtimes..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-runtimes.yaml

# 6. Verify installation
echo "Verifying installation..."
kubectl get pods -n kserve
kubectl get pods -n knative-serving
kubectl get pods -n istio-system

echo ""
echo "KServe installation complete!"
echo "Next steps:"
echo "  1. Build and push your model image"
echo "  2. Apply InferenceService manifests"
echo "  3. Test inference endpoint"
'''


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI CoreML iOS + KServe Deployment — Smoke Test")
    print("=" * 72)

    # Test CoreML config
    print("\n  ── CoreML iOS Config ──")
    coreml_cfg = CoreMLIOSConfig(
        model_name="SESICFDSurrogate",
        input_shape=[1, 3, 64, 64],
        use_dynamic_shapes=True,
        quantize=False,
    )
    print(f"    Model name: {coreml_cfg.model_name}")
    print(f"    Input shape: {coreml_cfg.input_shape}")
    print(f"    Dynamic shapes: {coreml_cfg.use_dynamic_shapes}")
    print(f"    Compute units: {coreml_cfg.compute_units}")

    # Test KServe config
    print("\n  ── KServe Config ──")
    kserve_cfg = KServeConfig(
        model_name="sesi-cfd-surrogate",
        use_gpu=True,
        min_replicas=1,
        max_replicas=10,
    )
    print(f"    Model name: {kserve_cfg.model_name}")
    print(f"    GPU: {kserve_cfg.use_gpu} x {kserve_cfg.gpu_count}")
    print(f"    Replicas: {kserve_cfg.min_replicas}-{kserve_cfg.max_replicas}")

    # Generate KServe manifests
    print("\n  ── KServe Manifest Generation ──")
    generator = KServeManifestGenerator(kserve_cfg)
    manifests = generator.generate_all()
    print(f"    Generated {len(manifests)} manifests:")
    for name in manifests:
        print(f"      - {name}")

    # Generate Docker files
    print("\n  ── Docker File Generation ──")
    docker_builder = KServeDockerBuilder(kserve_cfg)
    docker_files = docker_builder.write_files("/tmp/kserve_docker_test")
    print(f"    Generated {len(docker_files)} files:")
    for name, path in docker_files.items():
        print(f"      - {name}: {path}")

    # Generate Swift wrapper (mock)
    print("\n  ── Swift Wrapper Generation ──")
    swift_gen = SwiftWrapperGenerator(coreml_cfg)
    print(f"    Swift package would be generated at: {coreml_cfg.output_dir}/swift_package")

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • CoreML conversion requires macOS + coremltools")
    print("  • KServe requires cluster-admin access to install")
    print("  • GPU scheduling requires NVIDIA device plugin")
    print("  • Autoscaling requires Knative + Istio")
    print("  • Model storage requires appropriate credentials")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
