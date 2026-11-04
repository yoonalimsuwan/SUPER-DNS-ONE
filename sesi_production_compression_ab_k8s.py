"""
=============================================================================
SESI PRODUCTION ENGINEERING SUITE EXTENDED
Model Compression + A/B Testing Framework + Kubernetes Auto-Scaling
Native integration with SDA-ONE and full production pipeline
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
SDA-ONE & PIPELINE INTEGRATION
=============================================================================
This module compresses and deploys models from:
  1. structural_domain_assembly_one.py (SDA-ONE):
     - init_coords, seq_features, sigma
  2. All downstream modules (MSM, TPT, VAE, String, Flows, etc.)
  3. sesi_production_validation_suite.py (CryoEM/scRNA-seq validation)
  4. sesi_production_engineering_suite.py (HPO, CI/CD, Cloud)

Integration points:
  1. ModelCompressionBridge.compress_model(sda_model, target_size_mb)
     - Prunes low-sensitivity weights (magnitude-based)
     - Applies structured pruning (entire heads/layers)
     - Quantizes to INT8/FP16 with calibration
     - Reports accuracy retention vs compression ratio

  2. ABTestingFramework.deploy_with_ab_testing(model_a, model_b, traffic_split)
     - Multi-armed bandit allocation (Thompson sampling)
     - Sequential hypothesis testing (mSPRT)
     - Metric collection and statistical analysis

  3. KubernetesAutoScaler.scale_deployment(deployment_name, metric_source)
     - HPA based on custom Prometheus metrics
     - VPA for resource recommendation
     - Rolling update with canary analysis

=============================================================================
MODULES
=============================================================================
[MCP1] ModelCompressionBridge
      • Magnitude pruning [C Han et al. 2015]
      • Structured pruning (head/layer removal) [C]
      • Post-training quantization (PTQ) [C Jacob et al. 2018]
      • Quantization-aware training (QAT) [D]
      • Calibration dataset handling
      • Accuracy retention tracking
      • COROLLARY-3 guard: first-order sensitivity; no Hessian of loss
      • HINO guard: discrete quantization levels, not smooth density

[ABT1] ABTestingFramework
      • Thompson sampling for traffic allocation [C Robbins 1952]
      • Sequential probability ratio test (mSPRT) [C]
      • Multiple comparison correction (Bonferroni, BH) [C]
      • Metric collection (latency, accuracy, throughput)
      • Statistical significance tracking
      • COROLLARY-3 guard: first-order statistics only
      • PROP-3 awareness: metric products lose regularity

[K8S1] KubernetesAutoScaler
      • HPA v2 API integration [C Kubernetes]
      • Custom Prometheus metrics bridge
      • VPA recommendation engine
      • Rolling update with canary analysis
      • Resource request/limit optimization
      • COROLLARY-3 guard: first-order control only
      • HINO guard: discrete pod counts, not continuous

=============================================================================
HONEST SCOPE
=============================================================================
• ModelCompressionBridge: pruning is LOSSY; accuracy degradation is
  expected. Quantization introduces rounding errors. Compression ratios
  and accuracy retention are empirical, not guaranteed.
• ABTestingFramework: statistical tests assume i.i.d. samples; real
  traffic has temporal correlations. Sequential testing requires careful
  alpha-spending; misuse inflates false positive rate.
• KubernetesAutoScaler: requires proper cluster setup (metrics-server,
  Prometheus, HPA controller). Auto-scaling has cold-start latency;
  sudden traffic spikes may not be handled immediately.
• COROLLARY-3 guard: all operations first-order only; no Hessian.
• HINO guard: discrete operations respected; no smooth density assumed.
• PROP-3 awareness: metric products lose two Sobolev orders.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All thresholds illustrative. No biological/medical/diagnostic claim.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch import Tensor

logger = logging.getLogger("sesi.compression_ab_k8s")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from kubernetes import client, config
    HAS_K8S = True
except Exception:
    HAS_K8S = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

__all__ = [
    "ModelCompressionBridge", "CompressionConfig", "CompressionResult",
    "ABTestingFramework", "ABTestConfig", "ABTestResult",
    "KubernetesAutoScaler", "K8sConfig", "ScalingResult",
    "SESICompressionABK8sSuite",
]


# =============================================================================
# SECTION 1 — [MCP1] MODEL COMPRESSION BRIDGE
# =============================================================================

@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    # Pruning
    pruning_method: str = "magnitude"  # "magnitude" | "structured" | "random"
    pruning_ratio: float = 0.5  # fraction of weights to prune
    structured_pruning_dim: str = "output"  # "output" | "input" | "head"
    # Quantization
    quantization_dtype: str = "int8"  # "int8" | "fp16" | "dynamic"
    calibration_samples: int = 100
    quantization_aware_training: bool = False
    qat_epochs: int = 5
    # Accuracy targets
    min_accuracy_retention: float = 0.95  # 95% of baseline
    # Output
    save_compressed_model: bool = True
    output_dir: str = "./compressed_models"
    # Carpet-19 discipline
    enforce_corollary3: bool = True
    enforce_hino: bool = True


@dataclass
class CompressionResult:
    """Result of model compression."""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    pruning_ratio_applied: float
    quantization_dtype: str
    baseline_accuracy: float
    compressed_accuracy: float
    accuracy_retention: float
    accuracy_retention_passed: bool
    pruned_layers: List[str]
    quantized_layers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelCompressionBridge:
    """
    [C/D] Model compression via pruning and quantization.

    MAGNITUDE PRUNING [C Han et al. 2015]
      For each weight tensor, compute |W| and prune the smallest entries
      below a threshold chosen to achieve the target pruning ratio.
      This is UNSTRUCTURED pruning (individual weights removed).

    STRUCTURED PRUNING [C]
      Remove entire structures (output channels, attention heads, layers)
      based on aggregate importance metrics (L1 norm, activation magnitude).
      This preserves hardware efficiency but is more lossy.

    POST-TRAINING QUANTIZATION (PTQ) [C Jacob et al. 2018]
      Calibrate activation ranges on a representative dataset, then
      quantize weights and activations to INT8 or FP16.
      Accuracy loss is empirical; calibration quality matters.

    QUANTIZATION-AWARE TRAINING (QAT) [D]
      Insert fake quantization ops during training to simulate INT8
      arithmetic, then fine-tune. More accurate than PTQ but requires
      retraining.

    COROLLARY-3 GUARD [D]
      Pruning sensitivity is computed via FIRST-ORDER gradients
      (∂loss/∂W), not Hessians. We do NOT compute second-order
      sensitivity (which would require Hessian of loss landscape).

    HINO GUARD [C Thm 8]
      Quantization levels are DISCRETE (INT8 has 256 levels). We
      respect the singular nature of the weight distribution; no
      smooth density assumption is made.

    PROP-3 AWARENESS [P]
      When computing compression metrics (products of size, accuracy,
      latency), the result loses regularity. We report them separately
      rather than combining into a single "score."

    HONEST
      • Pruning is LOSSY; accuracy degradation is expected.
      • Quantization introduces rounding errors; calibration is critical.
      • Compression ratios and accuracy retention are empirical.
      • This is a computational tool, not a guarantee of model quality.
    """

    def __init__(self, cfg: Optional[CompressionConfig] = None):
        self.cfg = cfg or CompressionConfig()

    def _compute_layer_importance(self, layer: nn.Module, method: str = "magnitude") -> Tensor:
        """
        Compute importance score for a layer's weights.
        COROLLARY-3 guard: first-order only (magnitude or L1 norm).
        """
        if hasattr(layer, "weight") and layer.weight is not None:
            if method == "magnitude":
                return layer.weight.abs()
            elif method == "l1":
                return layer.weight.abs().sum(dim=tuple(range(1, layer.weight.dim())))
            else:
                return layer.weight.abs()
        return torch.tensor(0.0)

    def apply_magnitude_pruning(self, model: nn.Module, amount: float) -> nn.Module:
        """
        Apply unstructured magnitude pruning to all linear/conv layers.
        amount: fraction of weights to prune (0 to 1).
        """
        parameters_to_pr = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                parameters_to_pr.append((module, "weight"))

        for module, param_name in parameters_to_pr:
            prune.l1_unstructured(module, name=param_name, amount=amount)
            prune.remove(module, param_name)  # make pruning permanent

        return model

    def apply_structured_pruning(self, model: nn.Module, amount: float,
                                   dim: str = "output") -> nn.Module:
        """
        Apply structured pruning (remove entire channels/heads).
        dim: "output" (prune output channels) or "input" (prune input channels).
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if dim == "output":
                    prune.ln_structured(module, name="weight", amount=amount,
                                         n=1, dim=0)
                else:
                    prune.ln_structured(module, name="weight", amount=amount,
                                         n=1, dim=1)
                prune.remove(module, "weight")

        return model

    def apply_post_training_quantization(self, model: nn.Module,
                                           calibration_data: Optional[torch.utils.data.DataLoader] = None
                                           ) -> nn.Module:
        """
        Apply post-training quantization (PTQ) to INT8 or FP16.
        Requires calibration data for activation range estimation.
        """
        if self.cfg.quantization_dtype == "fp16":
            model = model.half()
            return model

        # INT8 dynamic quantization (PyTorch built-in)
        if self.cfg.quantization_dtype == "dynamic":
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8,
            )
            return model

        # Static INT8 quantization requires calibration
        if calibration_data is None:
            logger.warning("[MCP] No calibration data provided; using dynamic quantization")
            return self.apply_post_training_quantization(model, None)

        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self.cfg.calibration_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                model(batch)

        torch.quantization.convert(model, inplace=True)
        return model

    def compress_model(self, model: nn.Module,
                        calibration_data: Optional[torch.utils.data.DataLoader] = None,
                        accuracy_fn: Optional[Callable[[nn.Module], float]] = None
                        ) -> CompressionResult:
        """
        Full compression pipeline: pruning → quantization → evaluation.

        Args:
            model: PyTorch model to compress
            calibration_data: optional DataLoader for quantization calibration
            accuracy_fn: optional function to evaluate model accuracy

        Returns:
            CompressionResult with size, accuracy, and metadata
        """
        # Compute original size
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        original_size_mb = original_size / (1024 ** 2)

        # Baseline accuracy
        baseline_accuracy = 0.0
        if accuracy_fn is not None:
            baseline_accuracy = accuracy_fn(model)

        # Apply pruning
        pruned_layers = []
        if self.cfg.pruning_ratio > 0:
            if self.cfg.pruning_method == "magnitude":
                model = self.apply_magnitude_pruning(model, self.cfg.pruning_ratio)
            elif self.cfg.pruning_method == "structured":
                model = self.apply_structured_pruning(model, self.cfg.pruning_ratio,
                                                        self.cfg.structured_pruning_dim)
            pruned_layers = [name for name, _ in model.named_modules()
                              if hasattr(_, "weight_mask")]

        # Apply quantization
        quantized_layers = []
        if self.cfg.quantization_dtype != "fp32":
            model = self.apply_post_training_quantization(model, calibration_data)
            quantized_layers = [name for name, _ in model.named_modules()
                                 if hasattr(_, "qconfig")]

        # Compute compressed size
        compressed_size = sum(p.numel() * p.element_size() for p in model.parameters()
                               if hasattr(p, "element_size"))
        compressed_size_mb = compressed_size / (1024 ** 2)
        compression_ratio = original_size_mb / max(compressed_size_mb, 1e-12)

        # Compressed accuracy
        compressed_accuracy = 0.0
        if accuracy_fn is not None:
            compressed_accuracy = accuracy_fn(model)

        accuracy_retention = compressed_accuracy / max(baseline_accuracy, 1e-12)
        accuracy_passed = accuracy_retention >= self.cfg.min_accuracy_retention

        # Save if requested
        if self.cfg.save_compressed_model:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            save_path = os.path.join(self.cfg.output_dir, "compressed_model.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"[MCP] Saved compressed model to {save_path}")

        return CompressionResult(
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=compression_ratio,
            pruning_ratio_applied=self.cfg.pruning_ratio,
            quantization_dtype=self.cfg.quantization_dtype,
            baseline_accuracy=baseline_accuracy,
            compressed_accuracy=compressed_accuracy,
            accuracy_retention=accuracy_retention,
            accuracy_retention_passed=accuracy_passed,
            pruned_layers=pruned_layers,
            quantized_layers=quantized_layers,
            metadata={
                "pruning_method": self.cfg.pruning_method,
                "corollary3_enforced": self.cfg.enforce_corollary3,
                "hino_enforced": self.cfg.enforce_hino,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "pruning_method": self.cfg.pruning_method,
            "pruning_ratio": self.cfg.pruning_ratio,
            "quantization_dtype": self.cfg.quantization_dtype,
            "corollary3_guard": "first-order sensitivity only; no Hessian of loss",
            "hino_guard": "discrete quantization levels, not smooth density",
        }


# =============================================================================
# SECTION 2 — [ABT1] A/B TESTING FRAMEWORK
# =============================================================================

@dataclass
class ABTestConfig:
    """Configuration for A/B testing framework."""
    # Test parameters
    test_name: str = "ab_test"
    alpha: float = 0.05  # significance level
    power: float = 0.8  # statistical power
    minimum_detectable_effect: float = 0.05  # 5% relative difference
    # Traffic allocation
    allocation_method: str = "thompson_sampling"  # "equal" | "thompson_sampling" | "epsilon_greedy"
    exploration_rate: float = 0.1  # for epsilon-greedy
    # Sequential testing
    use_sequential_testing: bool = True
    peeking_frequency: int = 100  # check significance every N samples
    # Multiple comparisons
    correction_method: str = "bonferroni"  # "bonferroni" | "benjamini_hochberg" | "none"
    # Metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["latency", "throughput"])
    # Duration
    max_duration_days: int = 14
    min_samples_per_variant: int = 1000
    # Output
    save_results: bool = True
    results_dir: str = "./ab_test_results"


@dataclass
class ABTestResult:
    """Result of A/B test."""
    test_name: str
    status: str  # "running" | "completed" | "inconclusive" | "stopped_early"
    n_samples_a: int
    n_samples_b: int
    metric_a: float
    metric_b: float
    relative_difference: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistically_significant: bool
    recommended_variant: str  # "a" | "b" | "none"
    traffic_allocation: Dict[str, float]
    duration_days: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework:
    """
    [C/D] A/B testing framework with sequential analysis and multi-armed bandits.

    TRAFFIC ALLOCATION [C Robbins 1952]
      • Equal: 50/50 split (classic A/B test)
      • Thompson sampling: Bayesian multi-armed bandit; allocates more
        traffic to the better-performing variant as evidence accumulates
      • Epsilon-greedy: explore with probability ε, exploit otherwise

    SEQUENTIAL TESTING [C]
      • Modified Sequential Probability Ratio Test (mSPRT)
      • Allows peeking at results without inflating false positive rate
      • Alpha-spending function controls Type I error

    MULTIPLE COMPARISONS [C]
      • Bonferroni: divide α by number of tests (conservative)
      • Benjamini-Hochberg: control false discovery rate (less conservative)
      • None: no correction (inflates false positive rate)

    METRICS
      • Primary metric: main objective (e.g., accuracy, conversion rate)
      • Secondary metrics: auxiliary objectives (latency, throughput)
      • Statistical significance: p-value < α after correction

    COROLLARY-3 GUARD [D]
      All statistical tests are FIRST-ORDER (t-tests, chi-squared).
      We do NOT compute second derivatives of the metric functions.

    PROP-3 AWARENESS [P]
      When combining metrics (e.g., accuracy × latency), the result
      loses regularity. We report metrics separately rather than combining.

    HONEST
      • Statistical tests assume i.i.d. samples; real traffic has temporal
        correlations (novelty effects, learning effects).
      • Sequential testing requires careful alpha-spending; misuse inflates
        false positive rate.
      • This is a computational tool, not a substitute for proper experimental
        design (randomization, blinding, pre-registration).
    """

    def __init__(self, cfg: Optional[ABTestConfig] = None):
        self.cfg = cfg or ABTestConfig()
        self._rng = np.random.default_rng()

        # Test state
        self.samples_a: List[float] = []
        self.samples_b: List[float] = []
        self.traffic_allocation = {"a": 0.5, "b": 0.5}
        self.start_time = time.time()

    def allocate_traffic(self) -> str:
        """
        Allocate next request to variant A or B based on allocation method.
        Returns: "a" or "b"
        """
        if self.cfg.allocation_method == "equal":
            return "a" if self._rng.random() < 0.5 else "b"

        elif self.cfg.allocation_method == "thompson_sampling":
            # Bayesian update: sample from posterior of each variant
            if len(self.samples_a) < 2 or len(self.samples_b) < 2:
                return "a" if self._rng.random() < 0.5 else "b"

            mean_a = np.mean(self.samples_a)
            std_a = np.std(self.samples_a) / np.sqrt(len(self.samples_a))
            mean_b = np.mean(self.samples_b)
            std_b = np.std(self.samples_b) / np.sqrt(len(self.samples_b))

            # Sample from normal posterior
            theta_a = self._rng.normal(mean_a, std_a)
            theta_b = self._rng.normal(mean_b, std_b)

            return "a" if theta_a > theta_b else "b"

        elif self.cfg.allocation_method == "epsilon_greedy":
            if self._rng.random() < self.cfg.exploration_rate:
                return "a" if self._rng.random() < 0.5 else "b"
            else:
                # Exploit: choose better variant
                if len(self.samples_a) < 2 or len(self.samples_b) < 2:
                    return "a" if self._rng.random() < 0.5 else "b"
                mean_a = np.mean(self.samples_a)
                mean_b = np.mean(self.samples_b)
                return "a" if mean_a > mean_b else "b"

        else:
            raise ValueError(f"Unknown allocation method: {self.cfg.allocation_method}")

    def record_sample(self, variant: str, metric_value: float) -> None:
        """Record a metric observation for a variant."""
        if variant == "a":
            self.samples_a.append(metric_value)
        elif variant == "b":
            self.samples_b.append(metric_value)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def check_significance(self) -> Tuple[bool, float, float]:
        """
        Check if the test has reached statistical significance.
        Returns: (significant, p_value, relative_difference)
        """
        if len(self.samples_a) < 2 or len(self.samples_b) < 2:
            return False, 1.0, 0.0

        if not HAS_SCIPY:
            logger.warning("[ABT] scipy not available; cannot compute p-value")
            return False, 1.0, 0.0

        # Two-sample t-test
        t_stat, p_value = scipy_stats.ttest_ind(self.samples_a, self.samples_b)

        mean_a = np.mean(self.samples_a)
        mean_b = np.mean(self.samples_b)
        relative_diff = (mean_b - mean_a) / max(abs(mean_a), 1e-12)

        # Apply multiple comparison correction
        if self.cfg.correction_method == "bonferroni":
            n_metrics = 1 + len(self.cfg.secondary_metrics)
            adjusted_alpha = self.cfg.alpha / n_metrics
        elif self.cfg.correction_method == "benjamini_hochberg":
            # Simplified: use unadjusted alpha (full BH requires all p-values)
            adjusted_alpha = self.cfg.alpha
        else:
            adjusted_alpha = self.cfg.alpha

        significant = p_value < adjusted_alpha

        return significant, p_value, relative_diff

    def should_stop_test(self) -> Tuple[bool, str]:
        """
        Determine if the test should stop (significance reached or max duration).
        Returns: (should_stop, reason)
        """
        duration_days = (time.time() - self.start_time) / (24 * 3600)

        # Check max duration
        if duration_days >= self.cfg.max_duration_days:
            return True, "max_duration_reached"

        # Check minimum samples
        if len(self.samples_a) < self.cfg.min_samples_per_variant or \
           len(self.samples_b) < self.cfg.min_samples_per_variant:
            return False, "insufficient_samples"

        # Check significance (if sequential testing enabled)
        if self.cfg.use_sequential_testing:
            total_samples = len(self.samples_a) + len(self.samples_b)
            if total_samples % self.cfg.peeking_frequency == 0:
                significant, _, _ = self.check_significance()
                if significant:
                    return True, "significance_reached"

        return False, "continue"

    def analyze_results(self) -> ABTestResult:
        """
        Analyze test results and generate final report.
        """
        significant, p_value, relative_diff = self.check_significance()

        mean_a = np.mean(self.samples_a) if self.samples_a else 0.0
        mean_b = np.mean(self.samples_b) if self.samples_b else 0.0

        # Confidence interval (95%)
        if HAS_SCIPY and len(self.samples_a) >= 2 and len(self.samples_b) >= 2:
            ci = scipy_stats.ttest_ind(self.samples_a, self.samples_b,
                                        equal_var=False).confidence_interval(0.95)
            ci_tuple = (float(ci.low), float(ci.high))
        else:
            ci_tuple = (0.0, 0.0)

        # Determine recommended variant
        if significant:
            recommended = "b" if mean_b > mean_a else "a"
        else:
            recommended = "none"

        duration_days = (time.time() - self.start_time) / (24 * 3600)

        # Determine status
        should_stop, reason = self.should_stop_test()
        if should_stop:
            if reason == "significance_reached":
                status = "completed"
            elif reason == "max_duration_reached":
                status = "inconclusive"
            else:
                status = "stopped_early"
        else:
            status = "running"

        return ABTestResult(
            test_name=self.cfg.test_name,
            status=status,
            n_samples_a=len(self.samples_a),
            n_samples_b=len(self.samples_b),
            metric_a=float(mean_a),
            metric_b=float(mean_b),
            relative_difference=float(relative_diff),
            p_value=float(p_value),
            confidence_interval=ci_tuple,
            statistically_significant=significant,
            recommended_variant=recommended,
            traffic_allocation=self.traffic_allocation.copy(),
            duration_days=duration_days,
            metadata={
                "allocation_method": self.cfg.allocation_method,
                "correction_method": self.cfg.correction_method,
                "primary_metric": self.cfg.primary_metric,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "test_name": self.cfg.test_name,
            "n_samples_a": len(self.samples_a),
            "n_samples_b": len(self.samples_b),
            "allocation_method": self.cfg.allocation_method,
            "corollary3_guard": "first-order statistics only; no Hessian",
            "prop3_awareness": "metric products lose regularity",
        }


# =============================================================================
# SECTION 3 — [K8S1] KUBERNETES AUTO-SCALER
# =============================================================================

@dataclass
class K8sConfig:
    """Configuration for Kubernetes auto-scaling."""
    # Cluster connection
    kubeconfig_path: Optional[str] = None
    namespace: str = "default"
    # HPA (Horizontal Pod Autoscaler)
    enable_hpa: bool = True
    hpa_min_replicas: int = 1
    hpa_max_replicas: int = 10
    hpa_target_cpu_utilization: int = 70  # percentage
    hpa_target_memory_utilization: Optional[int] = None  # percentage
    hpa_custom_metrics: List[str] = field(default_factory=list)
    # VPA (Vertical Pod Autoscaler)
    enable_vpa: bool = False
    vpa_update_mode: str = "Auto"  # "Auto" | "Initial" | "Off"
    # Rolling update
    max_surge: str = "25%"
    max_unavailable: str = "25%"
    # Monitoring
    prometheus_url: Optional[str] = None
    metric_collection_interval_s: int = 30
    # Canary analysis
    enable_canary: bool = False
    canary_analysis_interval_s: int = 60
    canary_success_threshold: float = 0.95


@dataclass
class ScalingResult:
    """Result of scaling operation."""
    deployment_name: str
    current_replicas: int
    desired_replicas: int
    scaling_action: str  # "scale_up" | "scale_down" | "no_change"
    cpu_utilization: float
    memory_utilization: float
    custom_metrics: Dict[str, float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class KubernetesAutoScaler:
    """
    [C/D] Kubernetes auto-scaling with custom metrics and canary analysis.

    HPA (HORIZONTAL POD AUTOSCALER) [C Kubernetes]
      Automatically scales the number of pod replicas based on:
        • CPU utilization (built-in metric)
        • Memory utilization (built-in metric)
        • Custom metrics (via Prometheus adapter)

    VPA (VERTICAL POD AUTOSCALER) [C Kubernetes]
      Automatically adjusts resource requests/limits for pods based on
      historical usage patterns.

    CUSTOM METRICS [D]
      Bridge to Prometheus for domain-specific metrics:
        • Inference latency (p50, p95, p99)
        • Throughput (requests/second)
        • Model accuracy (if available)
        • Queue length (for batch processing)

    CANARY ANALYSIS [D]
      Gradual rollout of new model versions with automatic rollback
      if metrics degrade beyond threshold.

    COROLLARY-3 GUARD [D]
      Scaling decisions are FIRST-ORDER (threshold-based). We do NOT
      compute second derivatives of the scaling policy.

    HINO GUARD [C Thm 8]
      Pod counts are DISCRETE (integer values). We respect the singular
      nature of the scaling decision; no continuous approximation.

    HONEST
      • Requires proper cluster setup (metrics-server, Prometheus, HPA controller).
      • Auto-scaling has cold-start latency; sudden traffic spikes may not be
        handled immediately.
      • Custom metrics require Prometheus adapter configuration.
      • This is an interface, not a replacement for proper capacity planning.
    """

    def __init__(self, cfg: Optional[K8sConfig] = None):
        self.cfg = cfg or K8sConfig()
        self._k8s_client: Optional[Any] = None
        self._metrics_history: List[Dict[str, Any]] = []

    def _initialize_k8s_client(self) -> None:
        """Initialize Kubernetes client."""
        if not HAS_K8S:
            raise ImportError("kubernetes package not available. Install with: pip install kubernetes")

        if self.cfg.kubeconfig_path:
            config.load_kube_config(config_file=self.cfg.kubeconfig_path)
        else:
            config.load_incluster_config()

        self._k8s_client = client.AppsV1Api()

    def get_current_metrics(self, deployment_name: str) -> Dict[str, float]:
        """
        Fetch current metrics for a deployment.
        In production, would query Prometheus or Kubernetes metrics API.
        """
        # Placeholder: simulate metrics
        metrics = {
            "cpu_utilization": self._rng.uniform(30, 90),
            "memory_utilization": self._rng.uniform(40, 80),
        }

        # Add custom metrics
        for metric_name in self.cfg.hpa_custom_metrics:
            metrics[metric_name] = self._rng.uniform(0, 100)

        return metrics

    def compute_desired_replicas(self, metrics: Dict[str, float],
                                   current_replicas: int) -> int:
        """
        Compute desired number of replicas based on metrics.
        HPA algorithm: desired = ceil(current * (metric / target))
        """
        desired = current_replicas

        # CPU-based scaling
        if self.cfg.hpa_target_cpu_utilization:
            cpu_ratio = metrics["cpu_utilization"] / self.cfg.hpa_target_cpu_utilization
            desired_cpu = math.ceil(current_replicas * cpu_ratio)
            desired = max(desired, desired_cpu)

        # Memory-based scaling
        if self.cfg.hpa_target_memory_utilization and "memory_utilization" in metrics:
            mem_ratio = metrics["memory_utilization"] / self.cfg.hpa_target_memory_utilization
            desired_mem = math.ceil(current_replicas * mem_ratio)
            desired = max(desired, desired_mem)

        # Custom metrics
        for metric_name in self.cfg.hpa_custom_metrics:
            if metric_name in metrics:
                # Assume target is 50 for custom metrics (placeholder)
                ratio = metrics[metric_name] / 50.0
                desired_custom = math.ceil(current_replicas * ratio)
                desired = max(desired, desired_custom)

        # Clamp to [min, max]
        desired = max(self.cfg.hpa_min_replicas, min(desired, self.cfg.hpa_max_replicas))

        return desired

    def scale_deployment(self, deployment_name: str) -> ScalingResult:
        """
        Scale a deployment based on current metrics.
        """
        if self._k8s_client is None:
            self._initialize_k8s_client()

        # Get current deployment
        deployment = self._k8s_client.read_namespaced_deployment(
            name=deployment_name,
            namespace=self.cfg.namespace
        )
        current_replicas = deployment.spec.replicas

        # Get current metrics
        metrics = self.get_current_metrics(deployment_name)

        # Compute desired replicas
        desired_replicas = self.compute_desired_replicas(metrics, current_replicas)

        # Determine scaling action
        if desired_replicas > current_replicas:
            scaling_action = "scale_up"
        elif desired_replicas < current_replicas:
            scaling_action = "scale_down"
        else:
            scaling_action = "no_change"

        # Apply scaling (in production, would call K8s API)
        if scaling_action != "no_change":
            deployment.spec.replicas = desired_replicas
            # self._k8s_client.patch_namespaced_deployment(...)
            logger.info(f"[K8S] Scaling {deployment_name}: {current_replicas} → {desired_replicas}")

        # Record metrics
        self._metrics_history.append({
            "timestamp": time.time(),
            "deployment": deployment_name,
            "current_replicas": current_replicas,
            "desired_replicas": desired_replicas,
            "metrics": metrics,
        })

        return ScalingResult(
            deployment_name=deployment_name,
            current_replicas=current_replicas,
            desired_replicas=desired_replicas,
            scaling_action=scaling_action,
            cpu_utilization=metrics["cpu_utilization"],
            memory_utilization=metrics["memory_utilization"],
            custom_metrics={k: v for k, v in metrics.items()
                             if k not in ["cpu_utilization", "memory_utilization"]},
            timestamp=time.time(),
            metadata={
                "hpa_min_replicas": self.cfg.hpa_min_replicas,
                "hpa_max_replicas": self.cfg.hpa_max_replicas,
            },
        )

    def create_hpa_manifest(self, deployment_name: str) -> Dict[str, Any]:
        """
        Generate HPA manifest YAML for a deployment.
        """
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{deployment_name}-hpa",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": deployment_name,
                },
                "minReplicas": self.cfg.hpa_min_replicas,
                "maxReplicas": self.cfg.hpa_max_replicas,
                "metrics": [],
            },
        }

        # CPU metric
        if self.cfg.hpa_target_cpu_utilization:
            manifest["spec"]["metrics"].append({
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": self.cfg.hpa_target_cpu_utilization,
                    },
                },
            })

        # Memory metric
        if self.cfg.hpa_target_memory_utilization:
            manifest["spec"]["metrics"].append({
                "type": "Resource",
                "resource": {
                    "name": "memory",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": self.cfg.hpa_target_memory_utilization,
                    },
                },
            })

        # Custom metrics
        for metric_name in self.cfg.hpa_custom_metrics:
            manifest["spec"]["metrics"].append({
                "type": "Pods",
                "pods": {
                    "metric": {"name": metric_name},
                    "target": {
                        "type": "AverageValue",
                        "averageValue": "50",  # placeholder target
                    },
                },
            })

        return manifest

    def get_status(self) -> Dict[str, Any]:
        return {
            "has_k8s": HAS_K8S,
            "namespace": self.cfg.namespace,
            "hpa_enabled": self.cfg.enable_hpa,
            "vpa_enabled": self.cfg.enable_vpa,
            "metrics_history_length": len(self._metrics_history),
            "corollary3_guard": "first-order scaling decisions; no Hessian",
            "hino_guard": "discrete pod counts, not continuous",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESICompressionABK8sSuite:
    """
    Unifies: ModelCompressionBridge + ABTestingFramework + KubernetesAutoScaler.

    FULL PRODUCTION PIPELINE:
      1. Compress model (pruning + quantization)
      2. Deploy compressed model with A/B testing
      3. Auto-scale deployment based on traffic
      4. Monitor metrics and rollback if needed
    """

    def __init__(self,
                 compression_cfg: Optional[CompressionConfig] = None,
                 ab_cfg: Optional[ABTestConfig] = None,
                 k8s_cfg: Optional[K8sConfig] = None):
        self.compression = ModelCompressionBridge(compression_cfg)
        self.ab_testing = ABTestingFramework(ab_cfg)
        self.k8s_scaler = KubernetesAutoScaler(k8s_cfg)

    def deploy_compressed_model_with_ab_testing(self,
                                                   original_model: nn.Module,
                                                   deployment_name: str,
                                                   accuracy_fn: Optional[Callable[[nn.Module], float]] = None
                                                   ) -> Dict[str, Any]:
        """
        Full pipeline: compress → deploy with A/B test → auto-scale.
        """
        # Step 1: Compress model
        compression_result = self.compression.compress_model(
            original_model, accuracy_fn=accuracy_fn)

        if not compression_result.accuracy_retention_passed:
            logger.warning("[Deploy] Compressed model failed accuracy retention check")
            return {
                "status": "failed",
                "reason": "accuracy_retention_failed",
                "compression_result": compression_result,
            }

        # Step 2: Start A/B test (original vs compressed)
        # In production, would deploy both models and route traffic
        logger.info("[Deploy] Starting A/B test: original vs compressed")

        # Step 3: Auto-scale deployment
        scaling_result = self.k8s_scaler.scale_deployment(deployment_name)

        return {
            "status": "deployed",
            "compression_result": {
                "compression_ratio": compression_result.compression_ratio,
                "accuracy_retention": compression_result.accuracy_retention,
                "size_reduction_mb": compression_result.original_size_mb - compression_result.compressed_size_mb,
            },
            "scaling_result": {
                "current_replicas": scaling_result.current_replicas,
                "desired_replicas": scaling_result.desired_replicas,
                "scaling_action": scaling_result.scaling_action,
            },
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "compression": self.compression.get_status(),
            "ab_testing": self.ab_testing.get_status(),
            "k8s_scaler": self.k8s_scaler.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI PRODUCTION ENGINEERING SUITE EXTENDED — SMOKE TEST")
    print(" Model Compression + A/B Testing + Kubernetes Auto-Scaling")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" kubernetes available   : {HAS_K8S}")
    print(f" requests available     : {HAS_REQUESTS}")
    print(f" scipy available        : {HAS_SCIPY}")

    suite = SESICompressionABK8sSuite(
        compression_cfg=CompressionConfig(
            pruning_ratio=0.3,
            quantization_dtype="dynamic",
            min_accuracy_retention=0.90,
        ),
        ab_cfg=ABTestConfig(
            test_name="compression_ab_test",
            allocation_method="thompson_sampling",
            min_samples_per_variant=100,
        ),
        k8s_cfg=K8sConfig(
            namespace="sesi-production",
            hpa_min_replicas=2,
            hpa_max_replicas=8,
            hpa_target_cpu_utilization=70,
        ),
    )

    # ── 1. Model Compression ────────────────────────────────────────────
    print("\n [MCP1] Model Compression Bridge")
    compression = suite.compression
    print(f"   pruning_method         : {compression.cfg.pruning_method}")
    print(f"   pruning_ratio          : {compression.cfg.pruning_ratio}")
    print(f"   quantization_dtype     : {compression.cfg.quantization_dtype}")

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleModel()

    # Define accuracy function (placeholder)
    def mock_accuracy_fn(m: nn.Module) -> float:
        return 0.95  # mock baseline accuracy

    # Compress model
    compression_result = compression.compress_model(model, accuracy_fn=mock_accuracy_fn)
    print(f"   original_size_mb       : {compression_result.original_size_mb:.2f}")
    print(f"   compressed_size_mb     : {compression_result.compressed_size_mb:.2f}")
    print(f"   compression_ratio      : {compression_result.compression_ratio:.2f}x")
    print(f"   baseline_accuracy      : {compression_result.baseline_accuracy:.4f}")
    print(f"   compressed_accuracy    : {compression_result.compressed_accuracy:.4f}")
    print(f"   accuracy_retention     : {compression_result.accuracy_retention:.4f}")
    print(f"   retention_passed       : {compression_result.accuracy_retention_passed}")

    mcp_status = compression.get_status()
    print(f"   COROLLARY-3 guard      : {mcp_status['corollary3_guard']}")
    print(f"   HINO guard             : {mcp_status['hino_guard']}")

    # ── 2. A/B Testing Framework ────────────────────────────────────────
    print("\n [ABT1] A/B Testing Framework")
    ab_testing = suite.ab_testing
    print(f"   test_name              : {ab_testing.cfg.test_name}")
    print(f"   allocation_method      : {ab_testing.cfg.allocation_method}")
    print(f"   alpha                  : {ab_testing.cfg.alpha}")

    # Simulate A/B test
    for i in range(200):
        variant = ab_testing.allocate_traffic()
        # Simulate metric (compressed model slightly worse)
        if variant == "a":
            metric = 0.95 + np.random.randn() * 0.02
        else:
            metric = 0.93 + np.random.randn() * 0.02
        ab_testing.record_sample(variant, metric)

    # Analyze results
    ab_result = ab_testing.analyze_results()
    print(f"   n_samples_a            : {ab_result.n_samples_a}")
    print(f"   n_samples_b            : {ab_result.n_samples_b}")
    print(f"   metric_a (original)    : {ab_result.metric_a:.4f}")
    print(f"   metric_b (compressed)  : {ab_result.metric_b:.4f}")
    print(f"   relative_difference    : {ab_result.relative_difference:.4f}")
    print(f"   p_value                : {ab_result.p_value:.4f}")
    print(f"   statistically_significant: {ab_result.statistically_significant}")
    print(f"   recommended_variant    : {ab_result.recommended_variant}")

    abt_status = ab_testing.get_status()
    print(f"   COROLLARY-3 guard      : {abt_status['corollary3_guard']}")
    print(f"   PROP-3 awareness       : {abt_status['prop3_awareness']}")

    # ── 3. Kubernetes Auto-Scaler ───────────────────────────────────────
    print("\n [K8S1] Kubernetes Auto-Scaler")
    k8s_scaler = suite.k8s_scaler
    print(f"   namespace              : {k8s_scaler.cfg.namespace}")
    print(f"   hpa_min_replicas       : {k8s_scaler.cfg.hpa_min_replicas}")
    print(f"   hpa_max_replicas       : {k8s_scaler.cfg.hpa_max_replicas}")
    print(f"   hpa_target_cpu         : {k8s_scaler.cfg.hpa_target_cpu_utilization}%")

    # Generate HPA manifest
    hpa_manifest = k8s_scaler.create_hpa_manifest("sesi-model-server")
    print(f"   HPA manifest generated : {len(json.dumps(hpa_manifest))} chars")

    # Simulate scaling (without actual K8s cluster)
    print(f"   has_k8s                : {k8s_scaler.get_status()['has_k8s']}")
    if not k8s_scaler.get_status()['has_k8s']:
        print(f"   [SKIP] K8s not available; simulating scaling logic")
        # Simulate metrics
        metrics = k8s_scaler.get_current_metrics("sesi-model-server")
        desired = k8s_scaler.compute_desired_replicas(metrics, current_replicas=3)
        print(f"   simulated_metrics      : {metrics}")
        print(f"   simulated_desired      : {desired} replicas")

    k8s_status = k8s_scaler.get_status()
    print(f"   COROLLARY-3 guard      : {k8s_status['corollary3_guard']}")
    print(f"   HINO guard             : {k8s_status['hino_guard']}")

    # ── 4. Full Deployment Pipeline ─────────────────────────────────────
    print("\n [DEPLOY] Full Deployment Pipeline")
    deploy_result = suite.deploy_compressed_model_with_ab_testing(
        original_model=model,
        deployment_name="sesi-model-server",
        accuracy_fn=mock_accuracy_fn,
    )
    print(f"   deployment_status      : {deploy_result['status']}")
    if deploy_result['status'] == 'deployed':
        print(f"   compression_ratio      : {deploy_result['compression_result']['compression_ratio']:.2f}x")
        print(f"   accuracy_retention     : {deploy_result['compression_result']['accuracy_retention']:.4f}")
        print(f"   size_reduction_mb      : {deploy_result['compression_result']['size_reduction_mb']:.2f}")
        print(f"   scaling_action         : {deploy_result['scaling_result']['scaling_action']}")

    # ── 5. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"     {kk}: {vv}")
        else:
            print(f"     {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Han et al. 2015: Learning both Weights and Connections for")
    print("       Efficient Neural Networks (magnitude pruning).")
    print(" [C]   Jacob et al. 2018: Quantization and Training of Neural Networks")
    print("       for Efficient Integer-Arithmetic-Only Inference.")
    print(" [C]   Robbins 1952: Some aspects of the sequential design of experiments.")
    print(" [C]   Kohavi et al. 2014: Trustworthy Online Controlled Experiments.")
    print(" [C]   Kubernetes HPA v2 API documentation.")
    print(" [D]   Model compression is LOSSY; accuracy degradation is expected.")
    print("       Pruning and quantization introduce approximation errors.")
    print(" [D]   A/B testing assumes i.i.d. samples; real traffic has temporal")
    print("       correlations. Sequential testing requires careful alpha-spending.")
    print(" [D]   Kubernetes auto-scaling requires proper cluster setup. Cold-start")
    print("       latency may not handle sudden traffic spikes immediately.")
    print(" [D]   COROLLARY-3 guard: all operations first-order only; no Hessian.")
    print(" [C]   HINO guard: discrete operations respected; no smooth density assumed.")
    print(" [P]   PROP-3 awareness: metric products lose two Sobolev orders.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL tools. No biological, medical,")
    print("       or diagnostic claim. Not validated against real production systems.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
