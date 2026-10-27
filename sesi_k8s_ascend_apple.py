"""
=============================================================================
SESI K8s MANIFESTS — Huawei Ascend NPU + Apple Silicon
=============================================================================
สร้าง Kubernetes manifests สำหรับ:
  [A] Huawei Ascend 910/310 NPU (ใช้ Volcano + ascend device plugin)
  [B] Apple Silicon (M1/M2/M3) — ใช้ k3s/minikube บน macOS + MLX

HONEST SCOPE:
  • Ascend NPU ต้องติดตั้ง:
    - Ascend Device Plugin (huawei/ascend-device-plugin)
    - Volcano scheduler (volcano-sh/volcano)
    - CANN toolkit ใน container
  • Apple Silicon:
    - Kubernetes บน macOS ยังจำกัด (k3s, minikube, Docker Desktop)
    - GPU acceleration ผ่าน Metal/MLX ยังไม่รองรับใน K8s โดยตรง
    - ใช้เป็น CPU-only node หรือใช้ภายนอก (เช่น ผ่าน Ray)
=============================================================================
"""

from __future__ import annotations

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("sesi.k8s_ascend_apple")

__all__ = [
    "K8sAscendAppleConfig",
    "AscendManifestGenerator",
    "AppleSiliconManifestGenerator",
]


@dataclass
class K8sAscendAppleConfig:
    namespace: str = "sesi"
    image: str = "sesi/training:latest"

    # Ascend
    ascend_npu_per_worker: int = 1
    ascend_workers: int = 4
    ascend_chip_type: str = "Ascend910"   # Ascend910 | Ascend310
    ascend_memory_gb: int = 32
    ascend_cpu_per_worker: int = 8

    # Apple Silicon
    apple_replicas: int = 2
    apple_cpu_per_pod: int = 8
    apple_memory_gb: int = 16

    # Shared
    data_pvc_size: str = "500Gi"
    model_pvc_size: str = "100Gi"
    storage_class: str = "standard"


class AscendManifestGenerator:
    """
    สร้าง K8s manifests สำหรับ Huawei Ascend NPU

    Requirements:
      • Volcano scheduler (volcano-sh)
      • Ascend Device Plugin (huawei.com/Ascend910)
      • CANN toolkit ใน container image
    """

    def __init__(self, cfg: K8sAscendAppleConfig):
        self.cfg = cfg
        self.output_dir = Path("./manifests/ascend")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_namespace(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": self.cfg.namespace},
        }

    def generate_data_pvc(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-data-pvc",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": self.cfg.storage_class,
                "resources": {"requests": {"storage": self.cfg.data_pvc_size}},
            },
        }

    def generate_model_pvc(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-model-pvc",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": self.cfg.storage_class,
                "resources": {"requests": {"storage": self.cfg.model_pvc_size}},
            },
        }

    def generate_volcano_job(self) -> Dict:
        """
        Volcano Job สำหรับ distributed training บน Ascend NPU

        ใช้:
          • huawei.com/Ascend910 resource
          • Volcano gang-scheduling
          • NCCL/HCCL สำหรับ communication
        """
        cfg = self.cfg
        npu_resource = f"huawei.com/{cfg.ascend_chip_type}"

        return {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": {
                "name": "sesi-ascend-train",
                "namespace": cfg.namespace,
            },
            "spec": {
                "minAvailable": cfg.ascend_workers,
                "schedulerName": "volcano",
                "plugins": {
                    "svc": ["--publish-not-ready-addresses"],
                    "pytorch": ["--master=master", "--worker=worker"],
                },
                "tasks": [
                    {
                        "name": "master",
                        "replicas": 1,
                        "template": {
                            "metadata": {"labels": {"role": "master"}},
                            "spec": {
                                "containers": [{
                                    "name": "trainer",
                                    "image": cfg.image,
                                    "command": [
                                        "python", "train.py",
                                        "--mode", "train",
                                        "--distributed",
                                        "--backend", "hccl",
                                    ],
                                    "resources": {
                                        "limits": {
                                            npu_resource: cfg.ascend_npu_per_worker,
                                            "cpu": cfg.ascend_cpu_per_worker,
                                            "memory": f"{cfg.ascend_memory_gb}Gi",
                                        },
                                        "requests": {
                                            npu_resource: cfg.ascend_npu_per_worker,
                                            "cpu": cfg.ascend_cpu_per_worker // 2,
                                            "memory": f"{cfg.ascend_memory_gb // 2}Gi",
                                        },
                                    },
                                    "env": [
                                        {"name": "ASCEND_VISIBLE_DEVICES",
                                         "value": ",".join(str(i) for i in range(cfg.ascend_npu_per_worker))},
                                        {"name": "HCCL_CONNECT_TIMEOUT", "value": "3600"},
                                        {"name": "HCCL_EXEC_TIMEOUT", "value": "7200"},
                                    ],
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {"name": "models", "mountPath": "/models"},
                                    ],
                                }],
                                "volumes": [
                                    {"name": "data",
                                     "persistentVolumeClaim": {"claimName": "sesi-data-pvc"}},
                                    {"name": "models",
                                     "persistentVolumeClaim": {"claimName": "sesi-model-pvc"}},
                                ],
                                "restartPolicy": "OnFailure",
                            },
                        },
                    },
                    {
                        "name": "worker",
                        "replicas": cfg.ascend_workers - 1,
                        "template": {
                            "metadata": {"labels": {"role": "worker"}},
                            "spec": {
                                "containers": [{
                                    "name": "trainer",
                                    "image": cfg.image,
                                    "command": [
                                        "python", "train.py",
                                        "--mode", "train",
                                        "--distributed",
                                        "--backend", "hccl",
                                    ],
                                    "resources": {
                                        "limits": {
                                            npu_resource: cfg.ascend_npu_per_worker,
                                            "cpu": cfg.ascend_cpu_per_worker,
                                            "memory": f"{cfg.ascend_memory_gb}Gi",
                                        },
                                        "requests": {
                                            npu_resource: cfg.ascend_npu_per_worker,
                                            "cpu": cfg.ascend_cpu_per_worker // 2,
                                            "memory": f"{cfg.ascend_memory_gb // 2}Gi",
                                        },
                                    },
                                    "env": [
                                        {"name": "HCCL_CONNECT_TIMEOUT", "value": "3600"},
                                        {"name": "HCCL_EXEC_TIMEOUT", "value": "7200"},
                                    ],
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {"name": "models", "mountPath": "/models"},
                                    ],
                                }],
                                "volumes": [
                                    {"name": "data",
                                     "persistentVolumeClaim": {"claimName": "sesi-data-pvc"}},
                                    {"name": "models",
                                     "persistentVolumeClaim": {"claimName": "sesi-model-pvc"}},
                                ],
                                "restartPolicy": "OnFailure",
                            },
                        },
                    },
                ],
            },
        }

    def generate_device_plugin_note(self) -> Dict:
        """ConfigMap เก็บคำแนะนำการติดตั้ง Ascend Device Plugin"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "ascend-setup-notes",
                "namespace": self.cfg.namespace,
            },
            "data": {
                "INSTALL.md": """\
# Huawei Ascend NPU Setup for Kubernetes

## 1. Install Ascend Device Plugin
```bash
helm repo add volcano https://volcano-sh.github.io/helm-charts
helm install volcano volcano/volcano -n volcano-system --create-namespace

kubectl apply -f https://raw.githubusercontent.com/Ascend/ascend-device-plugin/main/deploy/ascend-device-plugin.yaml
```

## 2. Verify NPU visibility
```bash
kubectl get nodes -o json | jq '.items[].status.capacity' | grep -i ascend
```

## 3. Install CANN Toolkit in container
```dockerfile
FROM ascendhub.huawei.com/public-ascendhub/mindspore:latest
RUN pip install torch torch_npu
```

## 4. Test with a simple pod
```bash
kubectl run npu-test --image=your-image --restart=Never \\
  --overrides='{"spec":{"containers":[{"name":"npu-test","image":"your-image",
  "resources":{"limits":{"huawei.com/Ascend910":"1"}}}]}}'
```
"""
            },
        }

    def write_all(self) -> List[Path]:
        manifests = {
            "00_namespace.yaml": self.generate_namespace(),
            "01_data_pvc.yaml": self.generate_data_pvc(),
            "02_model_pvc.yaml": self.generate_model_pvc(),
            "10_volcano_job.yaml": self.generate_volcano_job(),
            "90_setup_notes.yaml": self.generate_device_plugin_note(),
        }
        paths = []
        for name, manifest in manifests.items():
            path = self.output_dir / name
            with open(path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            paths.append(path)
            logger.info(f"Wrote {path}")
        return paths


class AppleSiliconManifestGenerator:
    """
    สร้าง K8s manifests สำหรับ Apple Silicon (M1/M2/M3)

    HONEST NOTE:
      • K8s บน macOS ยังมีข้อจำกัด (ไม่มี GPU passthrough)
      • ใช้เป็น CPU-only training nodes หรือ inference
      • สำหรับ MLX จริงๆ ควรใช้ภายนอก (เช่น ผ่าน Ray หรือตรงๆ)
      • เหมาะกับ: k3s, minikube, Docker Desktop + K8s
    """

    def __init__(self, cfg: K8sAscendAppleConfig):
        self.cfg = cfg
        self.output_dir = Path("./manifests/apple_silicon")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_deployment(self) -> Dict:
        """Deployment สำหรับ Apple Silicon nodes (CPU-only)"""
        cfg = self.cfg
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "sesi-apple-train",
                "namespace": cfg.namespace,
                "labels": {"app": "sesi-apple", "platform": "apple-silicon"},
            },
            "spec": {
                "replicas": cfg.apple_replicas,
                "selector": {"matchLabels": {"app": "sesi-apple"}},
                "template": {
                    "metadata": {
                        "labels": {"app": "sesi-apple", "platform": "apple-silicon"},
                    },
                    "spec": {
                        # เลือกเฉพาะ Apple Silicon nodes
                        "nodeSelector": {
                            "kubernetes.io/arch": "arm64",
                            "kubernetes.io/os": "linux",  # K8s nodes ต้องรัน Linux
                            "node.kubernetes.io/instance-type": "apple-silicon",
                        },
                        "tolerations": [
                            {
                                "key": "apple-silicon",
                                "operator": "Exists",
                                "effect": "NoSchedule",
                            }
                        ],
                        "containers": [{
                            "name": "trainer",
                            "image": cfg.image,
                            "command": [
                                "python", "train.py",
                                "--mode", "train",
                                "--device", "cpu",  # Apple Silicon ใน K8s ใช้ CPU
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": cfg.apple_cpu_per_pod // 2,
                                    "memory": f"{cfg.apple_memory_gb // 2}Gi",
                                },
                                "limits": {
                                    "cpu": cfg.apple_cpu_per_pod,
                                    "memory": f"{cfg.apple_memory_gb}Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/data"},
                                {"name": "models", "mountPath": "/models"},
                            ],
                        }],
                        "volumes": [
                            {"name": "data",
                             "persistentVolumeClaim": {"claimName": "sesi-data-pvc"}},
                            {"name": "models",
                             "persistentVolumeClaim": {"claimName": "sesi-model-pvc"}},
                        ],
                    },
                },
            },
        }

    def generate_mlx_note(self) -> Dict:
        """ConfigMap อธิบายวิธีใช้ MLX นอก K8s"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "apple-mlx-notes",
                "namespace": self.cfg.namespace,
            },
            "data": {
                "MLX_GUIDE.md": """\
# Using Apple MLX with SESI (outside Kubernetes)

Kubernetes on macOS does not support GPU/NPU passthrough.
For MLX acceleration, run training directly on the host:

## 1. Install MLX
```bash
pip install mlx mlx-nn
```

## 2. Convert PyTorch model to MLX
```python
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Load PyTorch weights
import torch
pt_state = torch.load("model.pt", map_location="cpu")

# Convert to MLX format
mlx_params = {}
for k, v in pt_state.items():
    mlx_params[k] = mx.array(v.numpy())
```

## 3. Run training with MLX
```python
import mlx.optimizers as optim

optimizer = optim.AdamW(learning_rate=3e-4)

for epoch in range(100):
    for batch in dataloader:
        loss, grads = mx.value_and_grad(model, *batch)
        optimizer.update(model.parameters(), grads)
```

## 4. For K8s integration, use Ray:
```bash
pip install ray
ray start --head --num-cpus=8
# Then submit jobs via Ray API
```

## Performance Notes (Apple M3 Max):
  • MLX on M3 Max: ~2x faster than PyTorch CPU
  • Unified memory allows large models without OOM
  • Metal backend for GPU acceleration (not via K8s)
"""
            },
        }

    def write_all(self) -> List[Path]:
        manifests = {
            "10_apple_deployment.yaml": self.generate_deployment(),
            "90_mlx_guide.yaml": self.generate_mlx_note(),
        }
        paths = []
        for name, manifest in manifests.items():
            path = self.output_dir / name
            with open(path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            paths.append(path)
            logger.info(f"Wrote {path}")
        return paths


def generate_all_k8s_manifests(cfg: Optional[K8sAscendAppleConfig] = None) -> Dict[str, List[Path]]:
    """สร้าง manifests ทั้งหมด"""
    cfg = cfg or K8sAscendAppleConfig()

    ascend_gen = AscendManifestGenerator(cfg)
    apple_gen = AppleSiliconManifestGenerator(cfg)

    return {
        "ascend": ascend_gen.write_all(),
        "apple_silicon": apple_gen.write_all(),
    }


# ───────────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────────

def smoke_test():
    print("=" * 72)
    print(" K8s Manifests — Ascend NPU + Apple Silicon")
    print("=" * 72)

    cfg = K8sAscendAppleConfig()
    results = generate_all_k8s_manifests(cfg)

    for platform, paths in results.items():
        print(f"\n  {platform}:")
        for p in paths:
            print(f"    • {p}")

    print("\n" + "=" * 72)
    print("  HONEST NOTES:")
    print("  • Ascend NPU ต้องการ: Volcano + Ascend Device Plugin + CANN")
    print("  • Apple Silicon ใน K8s ใช้ได้เฉพาะ CPU (ไม่มี GPU passthrough)")
    print("  • สำหรับ MLX จริงๆ ควรใช้ภายนอกหรือผ่าน Ray")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
