"""
=============================================================================
SESI CLOUD ORCHESTRATOR — Unified Deployment Pipeline
TFLite + SageMaker + Vertex AI in one interface
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn

from sesi_tflite_converter import TFLiteConverter, TFLiteConvertConfig
from sesi_sagemaker_deploy import SageMakerDeployer, SageMakerConfig
from sesi_vertex_ai_deploy import VertexAIDeployer, VertexAIConfig

logger = logging.getLogger("sesi.cloud_orchestrator")

__all__ = ["CloudDeploymentOrchestrator"]


@dataclass
class CloudDeployConfig:
    enable_tflite: bool = True
    enable_sagemaker: bool = False
    enable_vertex: bool = False

    tflite: TFLiteConvertConfig = None
    sagemaker: SageMakerConfig = None
    vertex: VertexAIConfig = None

    def __post_init__(self):
        if self.tflite is None:
            self.tflite = TFLiteConvertConfig()
        if self.sagemaker is None:
            self.sagemaker = SageMakerConfig()
        if self.vertex is None:
            self.vertex = VertexAIConfig()


class CloudDeploymentOrchestrator:
    """
    One-stop deployment orchestrator.

    Usage:
        orchestrator = CloudDeploymentOrchestrator(cfg)
        report = orchestrator.deploy_all(model)
    """

    def __init__(self, cfg: CloudDeployConfig):
        self.cfg = cfg
        self.results: Dict[str, Any] = {}

    def deploy_tflite(self, model: nn.Module) -> Dict[str, Any]:
        """Convert and export TFLite for mobile/edge."""
        logger.info("=== TFLite Deployment ===")
        converter = TFLiteConverter(self.cfg.tflite)
        result = converter.convert(model)

        # Validate if conversion succeeded
        if result["success"]:
            validation = converter.validate(model)
            result["validation"] = validation
            logger.info(f"Validation max_diff: {validation.get('max_abs_diff', 'N/A')}")

        self.results["tflite"] = result
        return result

    def deploy_sagemaker(self, code_dir: str = ".") -> Dict[str, Any]:
        """Deploy to AWS SageMaker."""
        logger.info("=== SageMaker Deployment ===")
        deployer = SageMakerDeployer(self.cfg.sagemaker)
        result = deployer.run_full_pipeline(code_dir)
        self.results["sagemaker"] = result
        return result

    def deploy_vertex(self, code_dir: str = ".") -> Dict[str, Any]:
        """Deploy to Google Vertex AI."""
        logger.info("=== Vertex AI Deployment ===")
        deployer = VertexAIDeployer(self.cfg.vertex)
        result = deployer.run_full_pipeline(code_dir)
        self.results["vertex"] = result
        return result

    def deploy_all(self, model: nn.Module, code_dir: str = ".") -> Dict[str, Any]:
        """Run all enabled deployment targets."""
        report = {"targets": {}}

        if self.cfg.enable_tflite:
            report["targets"]["tflite"] = self.deploy_tflite(model)

        if self.cfg.enable_sagemaker:
            report["targets"]["sagemaker"] = self.deploy_sagemaker(code_dir)

        if self.cfg.enable_vertex:
            report["targets"]["vertex"] = self.deploy_vertex(code_dir)

        # Summary
        report["summary"] = {
            target: {
                "success": r.get("success", r.get("dry_run", False)),
                "artifacts": [
                    v for k, v in r.items()
                    if k in ("output_path", "image_uri", "endpoint_name",
                            "endpoint_resource", "model_resource")
                ],
            }
            for target, r in report["targets"].items()
        }

        return report


def smoke_test():
    print("=" * 72)
    print(" SESI Cloud Orchestrator — Smoke Test")
    print("=" * 72)

    cfg = CloudDeployConfig(
        enable_tflite=True,
        enable_sagemaker=True,   # will dry-run without AWS creds
        enable_vertex=True,      # will dry-run without GCP creds
    )

    orchestrator = CloudDeploymentOrchestrator(cfg)

    # Simple test model
    import torch
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    model = TestModel()
    cfg.tflite.input_shape = [1, 3, 32, 32]
    cfg.tflite.output_path = "/tmp/sesi_cloud_test/model.tflite"

    report = orchestrator.deploy_all(model, code_dir=".")

    print(f"\n  Deployment summary:")
    for target, info in report["summary"].items():
        print(f"    {target}: success={info['success']}")

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
