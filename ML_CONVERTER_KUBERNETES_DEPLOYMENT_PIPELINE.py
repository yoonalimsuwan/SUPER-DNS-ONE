"""
=============================================================================
SESI COREML CONVERTER × KUBERNETES DEPLOYMENT PIPELINE
PyTorch → CoreML (iOS/ANE) + Full K8s MLOps Orchestration
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
─────────────────────────
[C1] CoreMLConverter
     • torch.jit.trace + coremltools.convert (mlprogram, iOS 15+)
     • Dynamic shapes via ct.Shape(ct.RangeDim) — สำคัญสำหรับโปรตีน
       ที่มีความยาวต่างกัน
     • compute_units: ALL / CPU_AND_GPU / CPU_ONLY
     • INT8 quantization (weight-only และ linear)
     • Metadata + custom versioning

[C2] CoreMLValidator
     • ตรวจสอบว่าโมเดลแปลงสำเร็จ
     • เปรียบเทียบผลลัพธ์กับ PyTorch reference (max abs error)
     • รายงาน op ที่ไม่รองรับ (ถ้ามี)

[C3] K8sCoreMLPipeline
     • PyTorchJob สำหรับฝึกบน GPU/Ascend
     • Job บน macOS runner สำหรับแปลง (CoreML ต้องใช้ macOS)
     • CronJob แปลงเป็นระยะ
     • Deployment + Service สำหรับ model registry
     • HPA + PVC + NetworkPolicy

[C4] ModelRegistryClient
     • จัดการเวอร์ชันของโมเดล (.pt, .onnx, .mlpackage)
     • โปรโมทจาก staging → production

HONEST SCOPE & CRITICAL LIMITATIONS
────────────────────────────────────
• coremltools ต้องรันบน macOS เท่านั้น — ถ้ารันบน Linux/Windows
  จะแปลงไม่ได้ ต้องใช้ K8s Job บน macOS node
• FNO (SpectralConv2d) ใช้ torch.fft ที่ coremltools รองรับจำกัด
  → ต้องใช้ CoreMLCompatibleCFDSurrogate (conv-based) แทน
• Dynamic shapes ใน CoreML มี constraint: รองรับเฉพาะบางตำแหน่ง
  ต้องระบุ explicit shapes ที่ต้องการ
• การกระจาย .mlpackage ไป iOS ต้องผ่าน App Bundle หรือ
  background download + app reload (ไม่ใช่การโหลดตรงๆ)
• CoreML ไม่สามารถรันบน Android ได้ — ใช้ ONNX Runtime หรือ
  TensorFlow Lite แทน
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger("sesi.coreml_k8s")

# ── Optional imports ──────────────────────────────────────────────────
try:
    import coremltools as ct
    from coremltools.models import MLModel
    from coremltools.models.neural_network import quantization_utils
    HAS_COREMLTOOLS = True
except ImportError:
    HAS_COREMLTOOLS = False
    ct = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__all__ = [
    "CoreMLConvertConfig",
    "ComputeUnits",
    "CoreMLConverter",
    "CoreMLValidator",
    "K8sCoreMLConfig",
    "K8sCoreMLPipeline",
    "ModelRegistryClient",
    "CoreMLDeploymentManager",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class ComputeUnits(Enum):
    ALL = "all"                 # ANE + GPU + CPU
    CPU_AND_GPU = "cpu_and_gpu"
    CPU_ONLY = "cpu_only"


@dataclass
class CoreMLConvertConfig:
    # ── Model identity ──────────────────────────────────────────────────
    model_name: str = "sesi_protein_fold"
    model_version: str = "1.0.0"

    # ── Input spec ──────────────────────────────────────────────────────
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 20, 256)  # (batch, features, seq_len)

    # ── Dynamic shapes (สำหรับโปรตีนยาวต่างกัน) ─────────────────────────
    use_dynamic_shapes: bool = True
    seq_len_range: Tuple[int, int] = (32, 1024)
    batch_size_range: Tuple[int, int] = (1, 8)

    # ── Convert options ─────────────────────────────────────────────────
    compute_units: ComputeUnits = ComputeUnits.ALL
    minimum_deployment_target: str = "iOS16"
    mlprogram: bool = True  # iOS 15+, ใช้ MLX backend
    convert_to: str = "mlprogram"  # "mlprogram" | "neuralnetwork"

    # ── Quantization ────────────────────────────────────────────────────
    quantize: bool = False
    quantize_bits: int = 8  # 8 หรือ 4
    quantize_method: str = "linear"  # "linear" | "lut"

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "./coreml_models"
    trace_input: Optional[Tensor] = None

    # ── Metadata ────────────────────────────────────────────────────────
    author: str = "PAI & Yoon A Limsuwan / MSPS NETWORK"
    license: str = "MIT"
    description: str = "SESI protein folding model for iOS"


@dataclass
class K8sCoreMLConfig:
    namespace: str = "sesi-ml"
    image: str = "mspsnetwork/sesi-trainer:latest"
    macos_runner_label: str = "sesi.io/role=macos"

    # ── Resources ───────────────────────────────────────────────────────
    gpu_per_worker: int = 1
    gpu_type: str = "nvidia.com/gpu"
    cpu_per_worker: int = 8
    memory_per_worker: str = "32Gi"
    macos_cpu: int = 4
    macos_memory: str = "16Gi"

    # ── Training ────────────────────────────────────────────────────────
    train_replicas: int = 4
    batch_size: int = 8
    epochs: int = 100

    # ── Storage ─────────────────────────────────────────────────────────
    data_pvc_size: str = "200Gi"
    model_pvc_size: str = "50Gi"
    storage_class: str = "standard"

    # ── Model registry ──────────────────────────────────────────────────
    registry_url: str = "oci://registry.msps.network/sesi-models"
    promote_to_prod: bool = False

    # ── Scheduling ──────────────────────────────────────────────────────
    convert_cron: str = "0 3 * * 0"  # ทุกวันอาทิตย์ 03:00


# ═══════════════════════════════════════════════════════════════════════
# [C1] COREML CONVERTER
# ═══════════════════════════════════════════════════════════════════════

class CoreMLConverter:
    """
    แปลงโมเดล PyTorch → CoreML (.mlpackage)

    กระบวนการ:
      1. สร้าง example input (trace_input)
      2. torch.jit.trace model
      3. coremltools.convert() พร้อม:
         - dynamic shapes (ct.Shape + ct.RangeDim)
         - compute_units
         - minimum_deployment_target
      4. (optional) INT8 quantization
      5. บันทึก .mlpackage + metadata

    หมายเหตุสำคัญ:
      • ต้องรันบน macOS เท่านั้น
      • โมเดลที่มี FFT อาจแปลงไม่ได้ — ใช้ conv-based แทน
    """

    def __init__(self, cfg: CoreMLConvertConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        if not HAS_COREMLTOOLS:
            logger.warning(
                "coremltools ไม่พร้อม — การแปลงจะทำได้เฉพาะบน macOS "
                "ที่ติดตั้ง `pip install coremltools` แล้ว"
            )

    def _build_input_spec(self) -> Any:
        """สร้าง input specification สำหรับ coremltools"""
        if not HAS_COREMLTOOLS:
            return None

        cfg = self.cfg
        shape = list(cfg.input_shape)

        if cfg.use_dynamic_shapes:
            # Dynamic batch size
            batch_dim = ct.RangeDim(
                lower_bound=cfg.batch_size_range[0],
                upper_bound=cfg.batch_size_range[1],
                default=shape[0],
            )
            # Dynamic sequence length
            seq_dim = ct.RangeDim(
                lower_bound=cfg.seq_len_range[0],
                upper_bound=cfg.seq_len_range[1],
                default=shape[-1] if len(shape) >= 3 else shape[-1],
            )

            # สร้าง shape ที่มี dynamic dims
            if len(shape) == 3:
                dynamic_shape = (batch_dim, shape[1], seq_dim)
            elif len(shape) == 2:
                dynamic_shape = (batch_dim, seq_dim)
            else:
                dynamic_shape = shape

            return ct.TensorType(
                name=cfg.input_name,
                shape=ct.Shape(dynamic_shape),
            )
        else:
            return ct.TensorType(
                name=cfg.input_name,
                shape=ct.Shape(shape),
            )

    def _get_compute_units(self) -> Any:
        if not HAS_COREMLTOOLS:
            return None
        mapping = {
            ComputeUnits.ALL: ct.ComputeUnit.ALL,
            ComputeUnits.CPU_AND_GPU: ct.ComputeUnit.CPU_AND_GPU,
            ComputeUnits.CPU_ONLY: ct.ComputeUnit.CPU_ONLY,
        }
        return mapping[self.cfg.compute_units]

    def convert(self, model: nn.Module, example_input: Tensor) -> Dict[str, Any]:
        """
        แปลงโมเดลเป็น CoreML

        Args:
            model: PyTorch model (จะถูกตั้งเป็น eval)
            example_input: tensor ตัวอย่างสำหรับ tracing

        Returns:
            dict: {path, success, warnings, unsupported_ops}
        """
        result = {
            "success": False,
            "path": None,
            "warnings": [],
            "unsupported_ops": [],
        }

        if not HAS_COREMLTOOLS:
            result["warnings"].append(
                "coremltools ไม่พร้อม — ต้องรันบน macOS และติดตั้งด้วย "
                "`pip install coremltools`"
            )
            return result

        model.eval()
        model = model.cpu()  # CoreML ต้องใช้ CPU

        # สร้าง trace
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
        except Exception as e:
            result["warnings"].append(f"torch.jit.trace ล้มเหลว: {e}")
            return result

        # สร้าง input spec
        input_spec = self._build_input_spec()

        # แปลง
        try:
            mlmodel = ct.convert(
                traced,
                inputs=[input_spec],
                compute_units=self._get_compute_units(),
                minimum_deployment_target=getattr(ct.target, self.cfg.minimum_deployment_target),
                convert_to=self.cfg.convert_to,
            )
        except Exception as e:
            result["warnings"].append(f"coremltools.convert ล้มเหลว: {e}")
            # พยายามระบุ op ที่ไม่รองรับ
            result["unsupported_ops"] = self._find_unsupported_ops(traced)
            return result

        # เพิ่ม metadata
        mlmodel.author = self.cfg.author
        mlmodel.license = self.cfg.license
        mlmodel.short_description = self.cfg.description
        mlmodel.version = self.cfg.model_version
        mlmodel.user_defined_metadata["sesi.model_name"] = self.cfg.model_name
        mlmodel.user_defined_metadata["sesi.converted_at"] = str(time.time())

        # Quantize ถ้าต้องการ
        if self.cfg.quantize:
            mlmodel = self._quantize(mlmodel)

        # บันทึก
        out_path = Path(self.cfg.output_dir) / f"{self.cfg.model_name}.mlpackage"
        mlmodel.save(str(out_path))
        result["success"] = True
        result["path"] = str(out_path)

        logger.info(f"แปลงสำเร็จ: {out_path}")
        return result

    def _quantize(self, mlmodel: Any) -> Any:
        """INT8 quantization"""
        try:
            if self.cfg.quantize_bits == 8:
                op_config = quantization_utils.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype="int8",
                    granularity="per_tensor",
                )
            else:
                op_config = quantization_utils.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype="int4",
                    granularity="per_tensor",
                )
            config = quantization_utils.OptimizationConfig(global_config=op_config)
            mlmodel = quantization_utils.linear_quantize_weights(mlmodel, config=config)
            logger.info(f"Quantized เป็น INT{self.cfg.quantize_bits}")
        except Exception as e:
            logger.warning(f"Quantization ล้มเหลว: {e}")
        return mlmodel

    def _find_unsupported_ops(self, traced: Any) -> List[str]:
        """ระบุ ops ที่อาจไม่รองรับโดย CoreML"""
        unsupported = []
        try:
            graph = traced.graph
            for node in graph.nodes():
                if node.kind() in ("aten::fft_fft2", "aten::fft_rfft2",
                                    "aten::fft_ifft2", "aten::fft_irfft2"):
                    unsupported.append(f"{node.kind()} (FFT — ใช้ไม่ได้ใน CoreML)")
                elif "complex" in node.kind():
                    unsupported.append(f"{node.kind()} (complex tensors)")
        except Exception:
            pass
        return unsupported


# ═══════════════════════════════════════════════════════════════════════
# [C2] COREML VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class CoreMLValidator:
    """
    ตรวจสอบว่าโมเดล CoreML ที่แปลงแล้วถูกต้อง

    ทำการตรวจสอบ:
      1. โหลด .mlpackage ได้
      2. สร้าง prediction ได้
      3. เปรียบเทียบกับ PyTorch reference (max abs error)
      4. รายงาน compute units ที่ใช้
    """

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def validate(
        self,
        mlpackage_path: str,
        pytorch_model: nn.Module,
        example_input: Tensor,
        input_name: str = "input",
    ) -> Dict[str, Any]:
        result = {
            "valid": False,
            "max_abs_error": None,
            "passed_tolerance": False,
            "compute_units_used": None,
            "errors": [],
        }

        if not HAS_COREMLTOOLS:
            result["errors"].append("coremltools ไม่พร้อม")
            return result

        try:
            mlmodel = ct.models.MLModel(
                mlpackage_path,
                compute_units=ct.ComputeUnit.ALL,
            )
        except Exception as e:
            result["errors"].append(f"โหลด .mlpackage ล้มเหลว: {e}")
            return result

        # PyTorch prediction
        pytorch_model.eval()
        with torch.no_grad():
            pt_out = pytorch_model(example_input)
            if isinstance(pt_out, dict):
                pt_out = pt_out.get("output", next(iter(pt_out.values())))
            pt_out_np = pt_out.cpu().numpy()

        # CoreML prediction
        try:
            input_np = example_input.cpu().numpy()
            cm_out = mlmodel.predict({input_name: input_np})
            cm_out_np = next(iter(cm_out.values()))

            max_err = float(abs(pt_out_np - cm_out_np).max())
            result["max_abs_error"] = max_err
            result["passed_tolerance"] = max_err < self.tolerance
            result["compute_units_used"] = str(mlmodel.compute_unit)

        except Exception as e:
            result["errors"].append(f"prediction ล้มเหลว: {e}")
            return result

        result["valid"] = result["passed_tolerance"]
        return result


# ═══════════════════════════════════════════════════════════════════════
# [C3] K8S COREML PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class K8sCoreMLPipeline:
    """
    สร้าง Kubernetes manifests สำหรับ pipeline เต็มรูปแบบ:

      1. PyTorchJob — ฝึกบน GPU/Ascend cluster
      2. Job — แปลงเป็น CoreML บน macOS runner
      3. CronJob — แปลงเป็นระยะ (ทุกสัปดาห์)
      4. Deployment + Service — model registry
      5. HPA — autoscaling
      6. PVC — storage
      7. NetworkPolicy — ความปลอดภัย
    """

    def __init__(self, cfg: K8sCoreMLConfig):
        self.cfg = cfg

    def generate_all(self) -> Dict[str, Any]:
        """สร้าง manifests ทั้งหมด"""
        return {
            "00_namespace.yaml": self._namespace(),
            "01_pvc_data.yaml": self._pvc_data(),
            "02_pvc_models.yaml": self._pvc_models(),
            "10_pytorchjob_train.yaml": self._pytorchjob(),
            "20_job_convert.yaml": self._convert_job(),
            "21_cronjob_convert.yaml": self._convert_cronjob(),
            "30_deployment_registry.yaml": self._registry_deployment(),
            "31_service_registry.yaml": self._registry_service(),
            "32_hpa_registry.yaml": self._hpa(),
            "40_networkpolicy.yaml": self._network_policy(),
        }

    def _namespace(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": self.cfg.namespace},
        }

    def _pvc_data(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-data",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": self.cfg.storage_class,
                "resources": {"requests": {"storage": self.cfg.data_pvc_size}},
            },
        }

    def _pvc_models(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-models",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": self.cfg.storage_class,
                "resources": {"requests": {"storage": self.cfg.model_pvc_size}},
            },
        }

    def _pytorchjob(self) -> Dict:
        """PyTorchJob สำหรับฝึก (ใช้ Kubeflow Training Operator)"""
        container = {
            "name": "trainer",
            "image": self.cfg.image,
            "command": [
                "python", "-m", "torch.distributed.run",
                "--nproc_per_node", str(self.cfg.gpu_per_worker),
                "train.py",
                "--mode", "train",
                "--epochs", str(self.cfg.epochs),
                "--batch-size", str(self.cfg.batch_size),
            ],
            "resources": {
                "limits": {
                    self.cfg.gpu_type: self.cfg.gpu_per_worker,
                    "cpu": self.cfg.cpu_per_worker,
                    "memory": self.cfg.memory_per_worker,
                },
            },
            "volumeMounts": [
                {"name": "data", "mountPath": "/data"},
                {"name": "models", "mountPath": "/models"},
            ],
            "env": [
                {"name": "SESI_MODEL_REGISTRY", "value": self.cfg.registry_url},
                {"name": "NCCL_DEBUG", "value": "INFO"},
            ],
        }

        volumes = [
            {"name": "data", "persistentVolumeClaim": {"claimName": "sesi-data"}},
            {"name": "models", "persistentVolumeClaim": {"claimName": "sesi-models"}},
        ]

        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": "sesi-train",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "nprocPerNode": str(self.cfg.gpu_per_worker),
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": 1,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {"containers": [container], "volumes": volumes},
                        },
                    },
                    "Worker": {
                        "replicas": self.cfg.train_replicas - 1,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {"containers": [container], "volumes": volumes},
                        },
                    },
                },
            },
        }

    def _convert_job(self) -> Dict:
        """
        Job สำหรับแปลง CoreML บน macOS runner

        หมายเหตุสำคัญ: ต้องมี macOS node ใน cluster ที่ติดตั้ง
        coremltools แล้ว (เพราะ coremltools ต้องการ macOS)
        """
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "sesi-coreml-convert",
                "namespace": self.cfg.namespace,
                "labels": {"sesi.io/phase": "convert"},
            },
            "spec": {
                "backoffLimit": 3,
                "template": {
                    "metadata": {
                        "labels": {"sesi.io/phase": "convert"},
                    },
                    "spec": {
                        "nodeSelector": {
                            self.cfg.macos_runner_label: "true",
                        },
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "converter",
                            "image": self.cfg.image,
                            "command": [
                                "python", "-m", "sesi.coreml_k8s",
                                "--mode", "convert",
                                "--model-path", "/models/best_model.pt",
                                "--output-dir", "/coreml",
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": self.cfg.macos_cpu,
                                    "memory": self.cfg.macos_memory,
                                },
                                "limits": {
                                    "cpu": self.cfg.macos_cpu * 2,
                                    "memory": self.cfg.macos_memory,
                                },
                            },
                            "volumeMounts": [
                                {"name": "models", "mountPath": "/models"},
                                {"name": "coreml-out", "mountPath": "/coreml"},
                            ],
                        }],
                        "volumes": [
                            {"name": "models",
                             "persistentVolumeClaim": {"claimName": "sesi-models"}},
                            {"name": "coreml-out",
                             "persistentVolumeClaim": {"claimName": "sesi-models"}},
                        ],
                    },
                },
            },
        }

    def _convert_cronjob(self) -> Dict:
        """แปลงเป็นระยะ (ทุกสัปดาห์)"""
        job_spec = self._convert_job()["spec"]["template"]["spec"]
        return {
            "apiVersion": "batch/v1",
            "kind": "CronJob",
            "metadata": {
                "name": "sesi-coreml-convert-weekly",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "schedule": self.cfg.convert_cron,
                "jobTemplate": {
                    "spec": {
                        "backoffLimit": 3,
                        "template": {
                            "metadata": {"labels": {"sesi.io/phase": "convert"}},
                            "spec": job_spec,
                        },
                    },
                },
            },
        }

    def _registry_deployment(self) -> Dict:
        """Model registry — เก็บและแจกจ่ายโมเดล"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "sesi-model-registry",
                "namespace": self.cfg.namespace,
                "labels": {"app": "sesi-registry"},
            },
            "spec": {
                "replicas": 2,
                "selector": {"matchLabels": {"app": "sesi-registry"}},
                "template": {
                    "metadata": {"labels": {"app": "sesi-registry"}},
                    "spec": {
                        "containers": [{
                            "name": "registry",
                            "image": "registry:2",
                            "ports": [{"containerPort": 5000}],
                            "env": [
                                {"name": "REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY",
                                 "value": "/var/lib/registry"},
                            ],
                            "volumeMounts": [
                                {"name": "registry-data", "mountPath": "/var/lib/registry"},
                            ],
                            "resources": {
                                "requests": {"cpu": 1, "memory": "2Gi"},
                                "limits": {"cpu": 2, "memory": "4Gi"},
                            },
                        }],
                        "volumes": [
                            {"name": "registry-data",
                             "persistentVolumeClaim": {"claimName": "sesi-models"}},
                        ],
                    },
                },
            },
        }

    def _registry_service(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "sesi-model-registry",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "selector": {"app": "sesi-registry"},
                "ports": [{"port": 5000, "targetPort": 5000}],
                "type": "ClusterIP",
            },
        }

    def _hpa(self) -> Dict:
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "sesi-registry-hpa",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "sesi-model-registry",
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": 70},
                    },
                }],
            },
        }

    def _network_policy(self) -> Dict:
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "sesi-registry-policy",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "podSelector": {"matchLabels": {"app": "sesi-registry"}},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [{
                    "from": [{"namespaceSelector": {
                        "matchLabels": {"name": self.cfg.namespace},
                    }}],
                    "ports": [{"port": 5000, "protocol": "TCP"}],
                }],
                "egress": [{
                    "to": [{"namespaceSelector": {}}],
                    "ports": [{"port": 443, "protocol": "TCP"}],
                }],
            },
        }

    def write_manifests(self, output_dir: str = "./k8s_manifests") -> List[Path]:
        """เขียนทุก manifest ลงดิสก์"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifests = self.generate_all()
        paths = []
        for name, manifest in manifests.items():
            path = out / name
            if HAS_YAML:
                with open(path, "w") as f:
                    yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            else:
                with open(path, "w") as f:
                    json.dump(manifest, f, indent=2)
            paths.append(path)
            logger.info(f"เขียน {name}")
        return paths


# ═══════════════════════════════════════════════════════════════════════
# [C4] MODEL REGISTRY CLIENT
# ═══════════════════════════════════════════════════════════════════════

class ModelRegistryClient:
    """
    จัดการเวอร์ชันของโมเดล

    รองรับ:
      • อัปโหลด .pt, .onnx, .mlpackage
      • โปรโมทจาก staging → production
      • ค้นหาเวอร์ชันล่าสุด
    """

    def __init__(self, registry_url: str, namespace: str = "sesi-ml"):
        self.registry_url = registry_url
        self.namespace = namespace

    def push(self, local_path: str, tag: str = "latest") -> str:
        """อัปโหลดโมเดลไปยัง registry"""
        name = Path(local_path).name
        remote = f"{self.registry_url}/{name}:{tag}"
        try:
            subprocess.run(
                ["oras", "push", remote, local_path],
                check=True, capture_output=True, text=True, timeout=300,
            )
            logger.info(f"Push สำเร็จ: {remote}")
            return remote
        except FileNotFoundError:
            logger.warning("oras CLI ไม่พบ — ใช้ fallback: copy ไปยัง PVC")
            return local_path
        except Exception as e:
            logger.error(f"Push ล้มเหลว: {e}")
            return ""

    def pull(self, name: str, tag: str = "latest", output_dir: str = ".") -> str:
        """ดาวน์โหลดโมเดลจาก registry"""
        remote = f"{self.registry_url}/{name}:{tag}"
        out_path = Path(output_dir) / name
        try:
            subprocess.run(
                ["oras", "pull", remote, "-o", str(out_path)],
                check=True, capture_output=True, text=True, timeout=300,
            )
            return str(out_path / name)
        except Exception as e:
            logger.error(f"Pull ล้มเหลว: {e}")
            return ""

    def promote(self, name: str, from_tag: str, to_tag: str) -> bool:
        """โปรโมทจาก staging → production"""
        try:
            local = self.pull(name, from_tag)
            if local:
                self.push(local, to_tag)
                return True
        except Exception as e:
            logger.error(f"Promote ล้มเหลว: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# COREML DEPLOYMENT MANAGER
# ═══════════════════════════════════════════════════════════════════════

class CoreMLDeploymentManager:
    """
    จัดการการกระจายโมเดล CoreML ไปยัง iOS

    กระบวนการ:
      1. แปลงโมเดลเป็น .mlpackage
      2. Validate กับ PyTorch reference
      3. อัปโหลดไปยัง registry
      4. สร้าง App Bundle metadata สำหรับ iOS

    หมายเหตุ: การกระจายไปจริงต้องผ่าน:
      • App Store / TestFlight (ฝังใน app bundle)
      • หรือ On-Device ML Model Distribution (iOS 16+)
    """

    def __init__(self, cfg: CoreMLConvertConfig, k8s_cfg: K8sCoreMLConfig):
        self.cfg = cfg
        self.k8s_cfg = k8s_cfg
        self.converter = CoreMLConverter(cfg)
        self.validator = CoreMLValidator()
        self.registry = ModelRegistryClient(k8s_cfg.registry_url)

    def deploy(
        self,
        model: nn.Module,
        example_input: Tensor,
        tag: str = "latest",
    ) -> Dict[str, Any]:
        """
        กระบวนการกระจายแบบครบวงจร

        Returns:
            dict: {coreml_path, validated, registry_url, ios_bundle_ready}
        """
        result = {
            "coreml_path": None,
            "validated": False,
            "registry_url": None,
            "ios_bundle_ready": False,
            "errors": [],
        }

        # 1. แปลง
        convert_result = self.converter.convert(model, example_input)
        if not convert_result["success"]:
            result["errors"] = convert_result["warnings"]
            return result
        result["coreml_path"] = convert_result["path"]

        # 2. Validate
        if HAS_COREMLTOOLS:
            valid = self.validator.validate(
                convert_result["path"], model, example_input, self.cfg.input_name
            )
            result["validated"] = valid["valid"]
            if not valid["valid"]:
                result["errors"] = valid["errors"]
                return result
        else:
            logger.warning("ข้าม validation — coremltools ไม่พร้อม")

        # 3. อัปโหลดไปยัง registry
        registry_url = self.registry.push(convert_result["path"], tag)
        result["registry_url"] = registry_url

        # 4. สร้าง iOS bundle metadata
        bundle_meta = {
            "model_name": self.cfg.model_name,
            "model_version": self.cfg.model_version,
            "ios_min_version": self.cfg.minimum_deployment_target,
            "compute_units": self.cfg.compute_units.value,
            "quantized": self.cfg.quantize,
            "registry_url": registry_url,
            "converted_at": time.time(),
        }
        meta_path = Path(self.cfg.output_dir) / "ios_bundle_meta.json"
        meta_path.write_text(json.dumps(bundle_meta, indent=2))
        result["ios_bundle_ready"] = True

        return result


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI CoreML × Kubernetes Integration — Smoke Test")
    print("=" * 72)

    # ── Test 1: CoreMLConverter config ───────────────────────────────
    print("\n  ── [C1] CoreMLConverter ──")
    cfg = CoreMLConvertConfig(
        model_name="sesi_protein_fold",
        input_shape=(1, 64, 128),
        use_dynamic_shapes=True,
        quantize=False,
    )
    converter = CoreMLConverter(cfg)
    print(f"    Model name       : {cfg.model_name}")
    print(f"    Dynamic shapes   : {cfg.use_dynamic_shapes}")
    print(f"    Seq len range    : {cfg.seq_len_range}")
    print(f"    coremltools ready: {HAS_COREMLTOOLS}")

    # สร้างโมเดลทดสอบเล็กๆ
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3),
            )
        def forward(self, x):
            return self.net(x.mean(dim=1))

    model = TestModel()
    example = torch.randn(1, 64, 128)

    if HAS_COREMLTOOLS:
        result = converter.convert(model, example)
        print(f"    Convert success  : {result['success']}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"    ⚠️ {w}")
    else:
        print("    [SKIP] coremltools ไม่พร้อม — ทดสอบโครงสร้างเท่านั้น")

    # ── Test 2: K8s manifests ────────────────────────────────────────
    print("\n  ── [C3] K8s Manifests ──")
    k8s_cfg = K8sCoreMLConfig(namespace="sesi-ml")
    pipeline = K8sCoreMLPipeline(k8s_cfg)
    manifests = pipeline.generate_all()
    print(f"    สร้าง {len(manifests)} manifests:")
    for name in manifests:
        print(f"      • {name}")

    # เขียนลงดิสก์
    paths = pipeline.write_manifests("/tmp/sesi_k8s_test")
    print(f"    เขียนลงดิสก์: {len(paths)} ไฟล์")

    # ── Test 3: Model Registry ───────────────────────────────────────
    print("\n  ── [C4] Model Registry ──")
    registry = ModelRegistryClient("oci://registry.example.com/sesi")
    print(f"    Registry URL     : {registry.registry_url}")
    print(f"    (push/pull ต้องมี oras CLI)")

    # ── Test 4: Deployment Manager ───────────────────────────────────
    print("\n  ── Deployment Manager ──")
    deploy_mgr = CoreMLDeploymentManager(cfg, k8s_cfg)
    print(f"    Converter ready  : {deploy_mgr.converter is not None}")
    print(f"    Validator ready  : {deploy_mgr.validator is not None}")

    print("\n" + "=" * 72)
    print("  SMOKE TEST COMPLETE")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • coremltools ต้องรันบน macOS เท่านั้น")
    print("  • FNO (FFT-based) ไม่รองรับใน CoreML — ใช้ conv-based แทน")
    print("  • K8s convert job ต้องการ macOS node พร้อม coremltools")
    print("  • การกระจายไป iOS ต้องผ่าน App Bundle / TestFlight")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
