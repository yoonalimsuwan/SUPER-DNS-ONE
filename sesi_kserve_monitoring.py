"""
=============================================================================
SESI KSERVE MONITORING — Prometheus + Grafana Integration
Complete observability stack for KServe inference services
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

MONITORING STACK
────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Grafana (Dashboards)                                               │
│  └── Prometheus (Metrics)                                           │
│       └── KServe InferenceService                                   │
│            └── Model Server (PyTorch/TorchServe)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Metrics collected:                                                 │
│  • Request rate (QPS)                                               │
│  • Latency (p50, p95, p99)                                          │
│  • Error rate                                                       │
│  • Model-specific (predictions/sec)                                 │
│  • GPU/CPU utilization                                              │
│  • Autoscaler metrics                                               │
└─────────────────────────────────────────────────────────────────────┘

COMPONENTS DEPLOYED
───────────────────
1. Prometheus — metrics collection
2. Grafana — visualization dashboards
3. ServiceMonitor — KServe metrics scraping
4. PrometheusRule — alerting rules
5. Dashboard ConfigMap — pre-built SESI dashboards

HONEST SCOPE
────────────
• Requires K8s cluster with KServe installed
• Prometheus Operator recommended (kube-prometheus-stack)
• Grafana dashboards need provisioning or manual import
• Alertmanager configuration for alert routing
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("sesi.kserve_monitoring")

__all__ = [
    "KServeMonitoringConfig",
    "KServeMonitoringGenerator",
]

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class KServeMonitoringConfig:
    """Configuration for KServe monitoring stack."""
    
    def __init__(
        self,
        namespace: str = "sesi-serving",
        prometheus_namespace: str = "monitoring",
        grafana_namespace: str = "monitoring",
        model_name: str = "sesi-biomedical",
        scrape_interval: str = "30s",
        retention_days: int = 15,
    ):
        self.namespace = namespace
        self.prometheus_namespace = prometheus_namespace
        self.grafana_namespace = grafana_namespace
        self.model_name = model_name
        self.scrape_interval = scrape_interval
        self.retention_days = retention_days


class KServeMonitoringGenerator:
    """
    Generate complete monitoring stack manifests for KServe.
    
    Generates:
    1. ServiceMonitor for KServe metrics
    2. PrometheusRule for alerting
    3. Grafana Dashboard ConfigMap
    4. Deployment instructions
    """
    
    def __init__(self, cfg: KServeMonitoringConfig):
        self.cfg = cfg
    
    # ── ServiceMonitor ────────────────────────────────────────────────
    
    def generate_service_monitor(self) -> Dict:
        """
        ServiceMonitor to scrape KServe model metrics.
        
        Requires Prometheus Operator (kube-prometheus-stack).
        """
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{self.cfg.model_name}-monitor",
                "namespace": self.cfg.namespace,
                "labels": {
                    "release": "prometheus",  # Match kube-prometheus-stack label
                    "app": "sesi-kserve",
                },
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "serving.kserve.io/inferenceservice": self.cfg.model_name,
                    },
                },
                "namespaceSelector": {
                    "matchNames": [self.cfg.namespace],
                },
                "endpoints": [
                    {
                        "port": "http",
                        "path": "/metrics",
                        "interval": self.cfg.scrape_interval,
                    },
                ],
            },
        }
    
    # ── PrometheusRule ────────────────────────────────────────────────
    
    def generate_prometheus_rules(self) -> Dict:
        """
        PrometheusRule for SESI model alerting.
        
        Alerts:
        • High error rate (>5% for 5 min)
        • High latency (p99 > 1s for 5 min)
        • Model server down
        • High GPU utilization (>95% for 10 min)
        """
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PrometheusRule",
            "metadata": {
                "name": f"{self.cfg.model_name}-rules",
                "namespace": self.cfg.namespace,
                "labels": {
                    "release": "prometheus",
                    "app": "sesi-kserve",
                },
            },
            "spec": {
                "groups": [
                    {
                        "name": f"{self.cfg.model_name}.alerts",
                        "rules": [
                            {
                                "alert": "SESIModelHighErrorRate",
                                "expr": f"""
                                    sum(rate(inferenceservice_request_total{{
                                        inferenceservice="{self.cfg.model_name}",
                                        response_code=~"5.."
                                    }}[5m]))
                                    /
                                    sum(rate(inferenceservice_request_total{{
                                        inferenceservice="{self.cfg.model_name}"
                                    }}[5m])) > 0.05
                                """,
                                "for": "5m",
                                "labels": {"severity": "critical"},
                                "annotations": {
                                    "summary": f"High error rate for {self.cfg.model_name}",
                                    "description": "Error rate > 5% for 5 minutes",
                                },
                            },
                            {
                                "alert": "SESIModelHighLatency",
                                "expr": f"""
                                    histogram_quantile(0.99,
                                        sum(rate(inferenceservice_request_duration_seconds_bucket{{
                                            inferenceservice="{self.cfg.model_name}"
                                        }}[5m])) by (le)
                                    ) > 1.0
                                """,
                                "for": "5m",
                                "labels": {"severity": "warning"},
                                "annotations": {
                                    "summary": f"High latency for {self.cfg.model_name}",
                                    "description": "p99 latency > 1s for 5 minutes",
                                },
                            },
                            {
                                "alert": "SESIModelServerDown",
                                "expr": f"""
                                    absent(up{{
                                        job=~".*{self.cfg.model_name}.*"
                                    }} == 1)
                                """,
                                "for": "1m",
                                "labels": {"severity": "critical"},
                                "annotations": {
                                    "summary": f"{self.cfg.model_name} server down",
                                    "description": "No metrics received from model server",
                                },
                            },
                            {
                                "alert": "SESIModelHighGPUUsage",
                                "expr": f"""
                                    sum(DCGM_FI_DEV_GPU_UTIL{{
                                        pod=~".*{self.cfg.model_name}.*"
                                    }}) > 95
                                """,
                                "for": "10m",
                                "labels": {"severity": "warning"},
                                "annotations": {
                                    "summary": f"High GPU usage for {self.cfg.model_name}",
                                    "description": "GPU utilization > 95% for 10 minutes",
                                },
                            },
                        ],
                    },
                ],
            },
        }
    
    # ── Grafana Dashboard ─────────────────────────────────────────────
    
    def generate_grafana_dashboard(self) -> Dict:
        """
        Grafana dashboard for SESI model monitoring.
        
        Panels:
        1. Request Rate (QPS)
        2. Latency Percentiles (p50/p95/p99)
        3. Error Rate
        4. Active Replicas
        5. GPU Utilization
        6. Autoscaler Status
        """
        dashboard = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.cfg.model_name}-dashboard",
                "namespace": self.cfg.grafana_namespace,
                "labels": {
                    "grafana_dashboard": "1",  # Grafana sidecar label
                },
            },
            "data": {
                f"{self.cfg.model_name}.json": json.dumps({
                    "dashboard": {
                        "title": f"SESI {self.cfg.model_name} Monitoring",
                        "tags": ["sesi", "kserve", "ml"],
                        "timezone": "browser",
                        "refresh": "30s",
                        "panels": [
                            {
                                "id": 1,
                                "title": "Request Rate (QPS)",
                                "type": "timeseries",
                                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                                "targets": [{
                                    "expr": f'sum(rate(inferenceservice_request_total{{inferenceservice="{self.cfg.model_name}"}}[1m]))',
                                    "legendFormat": "QPS",
                                }],
                            },
                            {
                                "id": 2,
                                "title": "Latency Percentiles",
                                "type": "timeseries",
                                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                                "targets": [
                                    {
                                        "expr": f'histogram_quantile(0.50, sum(rate(inferenceservice_request_duration_seconds_bucket{{inferenceservice="{self.cfg.model_name}"}}[5m])) by (le))',
                                        "legendFormat": "p50",
                                    },
                                    {
                                        "expr": f'histogram_quantile(0.95, sum(rate(inferenceservice_request_duration_seconds_bucket{{inferenceservice="{self.cfg.model_name}"}}[5m])) by (le))',
                                        "legendFormat": "p95",
                                    },
                                    {
                                        "expr": f'histogram_quantile(0.99, sum(rate(inferenceservice_request_duration_seconds_bucket{{inferenceservice="{self.cfg.model_name}"}}[5m])) by (le))',
                                        "legendFormat": "p99",
                                    },
                                ],
                            },
                            {
                                "id": 3,
                                "title": "Error Rate",
                                "type": "timeseries",
                                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                                "targets": [{
                                    "expr": f"""
                                        sum(rate(inferenceservice_request_total{{
                                            inferenceservice="{self.cfg.model_name}",
                                            response_code=~"5.."
                                        }}[5m]))
                                        /
                                        sum(rate(inferenceservice_request_total{{
                                            inferenceservice="{self.cfg.model_name}"
                                        }}[5m]))
                                    """,
                                    "legendFormat": "Error Rate",
                                }],
                            },
                            {
                                "id": 4,
                                "title": "Active Replicas",
                                "type": "stat",
                                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
                                "targets": [{
                                    "expr": f'count(up{{job=~".*{self.cfg.model_name}.*"}} == 1)',
                                    "legendFormat": "Replicas",
                                }],
                            },
                            {
                                "id": 5,
                                "title": "GPU Utilization",
                                "type": "gauge",
                                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
                                "targets": [{
                                    "expr": f'avg(DCGM_FI_DEV_GPU_UTIL{{pod=~".*{self.cfg.model_name}.*"}})',
                                    "legendFormat": "GPU %",
                                }],
                            },
                        ],
                        "time": {"from": "now-1h", "to": "now"},
                    },
                    "overwrite": True,
                }),
            },
        }
        return dashboard
    
    # ── Install Script ────────────────────────────────────────────────
    
    def generate_install_script(self) -> str:
        """Generate installation script for monitoring stack."""
        return f'''#!/bin/bash
#=============================================================================
# SESI KServe Monitoring Installation
# Model: {self.cfg.model_name}
#=============================================================================

set -e

echo "=============================================="
echo " SESI KServe Monitoring Setup"
echo "=============================================="

# 1. Install Prometheus Operator (kube-prometheus-stack) if not present
echo "[1/4] Checking Prometheus Operator..."
if ! helm list -n {self.cfg.prometheus_namespace} | grep -q prometheus; then
    echo "Installing kube-prometheus-stack..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack \\
        --namespace {self.cfg.prometheus_namespace} \\
        --create-namespace \\
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \\
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \\
        --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \\
        --set grafana.sidecar.dashboards.enabled=true \\
        --set grafana.sidecar.dashboards.label=grafana_dashboard
else
    echo "Prometheus already installed."
fi

# 2. Apply ServiceMonitor
echo "[2/4] Applying ServiceMonitor..."
kubectl apply -f service_monitor.yaml

# 3. Apply PrometheusRule
echo "[3/4] Applying PrometheusRule..."
kubectl apply -f prometheus_rules.yaml

# 4. Apply Grafana Dashboard
echo "[4/4] Applying Grafana Dashboard..."
kubectl apply -f grafana_dashboard.yaml

echo ""
echo "=============================================="
echo " Monitoring setup complete!"
echo "=============================================="
echo ""
echo " Access Grafana:"
echo "   kubectl port-forward -n {self.cfg.grafana_namespace} svc/prometheus-grafana 3000:80"
echo "   Then open http://localhost:3000 (admin/prom-operator)"
echo ""
echo " Access Prometheus:"
echo "   kubectl port-forward -n {self.cfg.prometheus_namespace} svc/prometheus-kube-prometheus-prometheus 9090:9090"
echo "   Then open http://localhost:9090"
echo ""
echo " Verify metrics are being scraped:"
echo "   kubectl get servicemonitor -n {self.cfg.namespace}"
echo ""
'''
    
    # ── Write All ─────────────────────────────────────────────────────
    
    def write_all(self, output_dir: str = "./monitoring") -> Dict[str, str]:
        """Write all monitoring manifests to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        manifests = {
            "service_monitor.yaml": self.generate_service_monitor(),
            "prometheus_rules.yaml": self.generate_prometheus_rules(),
            "grafana_dashboard.yaml": self.generate_grafana_dashboard(),
        }
        
        paths = {}
        
        for filename, manifest in manifests.items():
            path = out_dir / filename
            if HAS_YAML:
                with open(path, "w") as f:
                    yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            else:
                with open(path, "w") as f:
                    json.dump(manifest, f, indent=2)
            paths[filename] = str(path)
            logger.info(f"Generated: {path}")
        
        # Write install script
        script_path = out_dir / "install_monitoring.sh"
        script_path.write_text(self.generate_install_script())
        script_path.chmod(0o755)
        paths["install_script"] = str(script_path)
        
        return paths


# ── JSON import for dashboard ─────────────────────────────────────────
import json


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Verify monitoring manifests can be generated."""
    print("=" * 72)
    print(" SESI KServe Monitoring — Smoke Test")
    print("=" * 72)
    
    cfg = KServeMonitoringConfig(
        model_name="sesi-biomedical",
        namespace="sesi-serving",
    )
    
    generator = KServeMonitoringGenerator(cfg)
    
    # Generate ServiceMonitor
    sm = generator.generate_service_monitor()
    print(f"\n  [1] ServiceMonitor: {sm['metadata']['name']}")
    print(f"      Namespace: {sm['metadata']['namespace']}")
    
    # Generate PrometheusRule
    rules = generator.generate_prometheus_rules()
    n_rules = len(rules["spec"]["groups"][0]["rules"])
    print(f"\n  [2] PrometheusRule: {rules['metadata']['name']}")
    print(f"      Alert rules: {n_rules}")
    
    # Generate Grafana Dashboard
    dashboard = generator.generate_grafana_dashboard()
    print(f"\n  [3] Grafana Dashboard: {dashboard['metadata']['name']}")
    
    # Write all
    paths = generator.write_all("/tmp/sesi_monitoring_test")
    print(f"\n  [4] Generated files:")
    for name, path in paths.items():
        print(f"      • {name}: {path}")
    
    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Requires Prometheus Operator (kube-prometheus-stack)")
    print("  • Grafana dashboards need sidecar or manual import")
    print("  • DCGM exporter needed for GPU metrics")
    print("  • Alertmanager config needed for alert routing")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
