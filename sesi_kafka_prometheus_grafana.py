"""
=============================================================================
SESI KAFKA → PROMETHEUS → GRAFANA REAL-TIME OBSERVABILITY STACK
Multi-messenger pipeline metrics + pre-built Grafana dashboards
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

ARCHITECTURE
------------
  GCN Kafka stream (sesi_gcn_kafka_dashboard)
        │
        ▼
  KafkaMetricsBridge  ──► PrometheusExporter  ──► /metrics (HTTP)
        │                                              │
        │                                              ▼
        └──► NoZenoAlertThrottler              Prometheus scraper
                                                         │
                                                         ▼
                                                  Grafana dashboards

METRICS EXPOSED (Prometheus naming convention)
----------------------------------------------
  sesi_gcn_alerts_total{messenger}        Counter   — parsed alerts per messenger
  sesi_gcn_parse_failures_total           Counter   — NEVER suppressed (§3.3)
  sesi_coincidences_total                 Counter   — GW×(ν/EM) coincidences
  sesi_candidates_total                   Counter   — candidates RECORDED
  sesi_alerts_emitted_total               Counter   — No-Zeno-EMITTED alerts
  sesi_alerts_deferred_total              Counter   — deferred (recorded, not dropped)
  sesi_throttle_invariant_holds           Gauge     — N(T)≤1+⌊T/τ_e⌋ status
  sesi_pipeline_latency_seconds           Histogram — end-to-end latency

LABELS: [C] Prometheus exposition format  [D] derived here  [O] live ops

NO-ZENO DISCIPLINE (cfd_pp.py §3.3)
-----------------------------------
• Metrics separate RECORDED candidates from EMITTED alerts — the throttle
  controls emission only.
• Parse/consumer FAILURES get their own counter and are always visible;
  they never disappear into a suppressed stream.
• The invariant gauge lets Grafana alert if the bound were ever violated.

HONEST SCOPE
------------
• Requires `prometheus_client` for the full exporter; falls back to a
  minimal hand-rolled exposition otherwise.
• The Grafana dashboard JSON is a REAL dashboard model; import it into
  Grafana. Live Kafka needs credentials ([O]).
=============================================================================
"""

from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.prom_grafana")

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server,
    )
    HAS_PROM = True
except ImportError:
    HAS_PROM = False

# Reuse the real GCN consumer + throttle from previous modules
try:
    from sesi_gcn_kafka_dashboard import GCNKafkaConfig, GCNKafkaConsumer, SyntheticAlertSource
    HAS_KAFKA_MOD = True
except ImportError:
    HAS_KAFKA_MOD = False

try:
    from sesi_bayestar_icecube_mm import NoZenoAlertThrottler
    HAS_THROTTLE = True
except ImportError:
    HAS_THROTTLE = False

__all__ = ["MetricsConfig", "PrometheusExporter", "KafkaMetricsBridge",
           "GrafanaDashboardBuilder", "write_monitoring_stack"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetricsConfig:
    exporter_port: int = 9100
    tau_e: float = 5.0                 # No-Zeno min alert spacing
    latency_buckets: tuple = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
    grafana_dir: str = "./monitoring/grafana"
    prometheus_dir: str = "./monitoring/prometheus"


# ═══════════════════════════════════════════════════════════════════════
# [C/D] PROMETHEUS EXPORTER
# ═══════════════════════════════════════════════════════════════════════

class PrometheusExporter:
    """
    [C/D] Defines all pipeline metrics. Uses prometheus_client when
    available, else a minimal hand-rolled fallback with the same names.
    """

    def __init__(self, cfg: MetricsConfig):
        self.cfg = cfg
        if HAS_PROM:
            self.gcn_alerts = Counter(
                "sesi_gcn_alerts_total", "Parsed GCN alerts", ["messenger"])
            self.parse_failures = Counter(
                "sesi_gcn_parse_failures_total", "Alert parse/consumer failures")
            self.coincidences = Counter(
                "sesi_coincidences_total", "GW x (nu/EM) coincidences")
            self.candidates = Counter(
                "sesi_candidates_total", "Candidates recorded (never dropped)")
            self.emitted = Counter(
                "sesi_alerts_emitted_total", "No-Zeno-emitted alerts")
            self.deferred = Counter(
                "sesi_alerts_deferred_total", "Deferred alerts (recorded)")
            self.invariant = Gauge(
                "sesi_throttle_invariant_holds", "1 if N(T)<=1+floor(T/tau_e)")
            self.latency = Histogram(
                "sesi_pipeline_latency_seconds", "End-to-end latency",
                buckets=cfg.latency_buckets)
        else:
            # fallback counters as plain dicts
            self._fallback = {
                "sesi_gcn_alerts_total": {},
                "sesi_gcn_parse_failures_total": 0,
                "sesi_coincidences_total": 0,
                "sesi_candidates_total": 0,
                "sesi_alerts_emitted_total": 0,
                "sesi_alerts_deferred_total": 0,
            }
            self._invariant = 1.0
            self._latencies: List[float] = []

    # ── increment helpers (work in both modes) ────────────────────────
    def inc_alert(self, messenger: str):
        if HAS_PROM:
            self.gcn_alerts.labels(messenger=messenger).inc()
        else:
            d = self._fallback["sesi_gcn_alerts_total"]
            d[messenger] = d.get(messenger, 0) + 1

    def inc_parse_failure(self):
        if HAS_PROM:
            self.parse_failures.inc()
        else:
            self._fallback["sesi_gcn_parse_failures_total"] += 1

    def inc_coincidence(self):
        if HAS_PROM:
            self.coincidences.inc()
        else:
            self._fallback["sesi_coincidences_total"] += 1

    def inc_candidate(self):
        if HAS_PROM:
            self.candidates.inc()
        else:
            self._fallback["sesi_candidates_total"] += 1

    def inc_emitted(self):
        if HAS_PROM:
            self.emitted.inc()
        else:
            self._fallback["sesi_alerts_emitted_total"] += 1

    def inc_deferred(self):
        if HAS_PROM:
            self.deferred.inc()
        else:
            self._fallback["sesi_alerts_deferred_total"] += 1

    def set_invariant(self, holds: bool):
        if HAS_PROM:
            self.invariant.set(1.0 if holds else 0.0)
        else:
            self._invariant = 1.0 if holds else 0.0

    def observe_latency(self, seconds: float):
        if HAS_PROM:
            self.latency.observe(seconds)
        else:
            self._latencies.append(seconds)

    # ── exposition ────────────────────────────────────────────────────
    def render(self) -> str:
        """Return Prometheus text exposition."""
        if HAS_PROM:
            return generate_latest().decode("utf-8")
        # hand-rolled fallback
        lines = []
        for mess, val in self._fallback["sesi_gcn_alerts_total"].items():
            lines.append(f'sesi_gcn_alerts_total{{messenger="{mess}"}} {val}')
        for k in ["sesi_gcn_parse_failures_total", "sesi_coincidences_total",
                  "sesi_candidates_total", "sesi_alerts_emitted_total",
                  "sesi_alerts_deferred_total"]:
            lines.append(f"{k} {self._fallback[k]}")
        lines.append(f"sesi_throttle_invariant_holds {self._invariant}")
        return "\n".join(lines) + "\n"

    def serve(self):
        """Start the metrics HTTP server (prometheus_client only)."""
        if HAS_PROM:
            start_http_server(self.cfg.exporter_port)
            logger.info(f"Prometheus exporter on :{self.cfg.exporter_port}/metrics")
        else:
            logger.warning("prometheus_client missing; use render() manually")


# ═══════════════════════════════════════════════════════════════════════
# [D] KAFKA → METRICS BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class KafkaMetricsBridge:
    """
    [D] Poll the GCN Kafka source and update Prometheus metrics.

    Applies the No-Zeno throttle to alert EMISSION and records every
    candidate. Failures are counted, never swallowed.
    """

    def __init__(self, cfg: MetricsConfig,
                 kafka_cfg: Optional[Any] = None):
        self.cfg = cfg
        self.exporter = PrometheusExporter(cfg)
        self.throttler = NoZenoAlertThrottler(tau_e=cfg.tau_e) if HAS_THROTTLE else None

        # data source: real Kafka if configured, else synthetic (flagged)
        self.synthetic = True
        self.source = None
        if HAS_KAFKA_MOD:
            kcfg = kafka_cfg or GCNKafkaConfig()
            self.consumer = GCNKafkaConsumer(kcfg)
            if self.consumer.connect():
                self.source = self.consumer
                self.synthetic = False
            else:
                logger.warning("Kafka unavailable → SYNTHETIC source (flagged)")
        if self.source is None:
            self.source = SyntheticAlertSource() if HAS_KAFKA_MOD else None

    def poll_once(self) -> Dict[str, int]:
        """One poll→metrics cycle. Returns counts for this cycle."""
        t0 = time.time()
        counts = {"alerts": 0, "candidates": 0, "emitted": 0, "deferred": 0}

        if self.source is None:
            self.exporter.inc_parse_failure()
            return counts

        try:
            alerts = self.source.poll()
        except Exception as e:
            # §3.3: record the failure, never suppress
            self.exporter.inc_parse_failure()
            if self.throttler is not None:
                self.throttler.record_warning(f"poll failed: {e}")
            logger.warning(f"poll failed: {e}")
            return counts

        for alert in alerts:
            mess = alert.get("messenger", "UNKNOWN")
            self.exporter.inc_alert(mess)
            counts["alerts"] += 1

            # every candidate is recorded
            self.exporter.inc_candidate()
            counts["candidates"] += 1

            # throttle EMISSION only
            if self.throttler is not None:
                emitted = self.throttler.try_emit(alert, t=time.time())
                if emitted:
                    self.exporter.inc_emitted()
                    counts["emitted"] += 1
                else:
                    self.exporter.inc_deferred()
                    counts["deferred"] += 1
            else:
                self.exporter.inc_emitted()
                counts["emitted"] += 1

        self.exporter.observe_latency(time.time() - t0)

        # update invariant gauge
        if self.throttler is not None:
            T = max(time.time() - getattr(self.throttler, "_start", time.time()), 1e-3)
            chk = self.throttler.bound_check(T)
            self.exporter.set_invariant(chk["invariant_holds"])

        return counts


# ═══════════════════════════════════════════════════════════════════════
# [D] GRAFANA DASHBOARD BUILDER
# ═══════════════════════════════════════════════════════════════════════

class GrafanaDashboardBuilder:
    """
    [D] Generate a real Grafana dashboard JSON with panels for the
    multi-messenger pipeline.
    """

    def __init__(self, cfg: MetricsConfig):
        self.cfg = cfg

    def _panel(self, pid: int, title: str, expr: str, ptype: str = "timeseries",
               gridPos: Optional[Dict] = None) -> Dict:
        return {
            "id": pid, "type": ptype, "title": title,
            "gridPos": gridPos or {"h": 8, "w": 12, "x": 0, "y": (pid - 1) * 8},
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "targets": [{"expr": expr, "refId": "A"}],
            "fieldConfig": {"defaults": {}, "overrides": []},
        }

    def build(self) -> Dict[str, Any]:
        panels = [
            self._panel(1, "GCN alerts rate (by messenger)",
                        'rate(sesi_gcn_alerts_total[1m])',
                        gridPos={"h": 8, "w": 12, "x": 0, "y": 0}),
            self._panel(2, "Candidates vs emitted vs deferred",
                        'rate(sesi_candidates_total[1m])',
                        gridPos={"h": 8, "w": 12, "x": 12, "y": 0}),
            self._panel(3, "Coincidences rate",
                        'rate(sesi_coincidences_total[1m])',
                        gridPos={"h": 8, "w": 12, "x": 0, "y": 8}),
            self._panel(4, "Parse/consumer failures (NEVER suppressed)",
                        'rate(sesi_gcn_parse_failures_total[1m])',
                        gridPos={"h": 8, "w": 12, "x": 12, "y": 8}),
            self._panel(5, "Pipeline latency p95",
                        'histogram_quantile(0.95, rate(sesi_pipeline_latency_seconds_bucket[1m]))',
                        gridPos={"h": 8, "w": 12, "x": 0, "y": 16}),
            self._panel(6, "No-Zeno throttle invariant (1=holds)",
                        'sesi_throttle_invariant_holds', ptype="stat",
                        gridPos={"h": 8, "w": 12, "x": 12, "y": 16}),
        ]
        return {
            "uid": "sesi-mm-pipeline",
            "title": "SESI Multi-Messenger Pipeline",
            "tags": ["sesi", "multimessenger", "gcn"],
            "timezone": "utc",
            "schemaVersion": 39,
            "refresh": "5s",
            "panels": panels,
            "time": {"from": "now-1h", "to": "now"},
        }

    def write(self, path: Optional[str] = None) -> str:
        import os
        path = path or os.path.join(self.cfg.grafana_dir, "dashboard.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.build(), f, indent=2)
        logger.info(f"Grafana dashboard written to {path}")
        return path


# ═══════════════════════════════════════════════════════════════════════
# [D] MONITORING STACK FILES (docker-compose, prometheus.yml, provisioning)
# ═══════════════════════════════════════════════════════════════════════

def write_monitoring_stack(cfg: MetricsConfig) -> Dict[str, str]:
    """Write prometheus.yml, Grafana provisioning, and docker-compose."""
    import os
    files = {}

    # prometheus.yml
    prom_dir = cfg.prometheus_dir
    os.makedirs(prom_dir, exist_ok=True)
    prom_yml = f"""global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'sesi-mm-pipeline'
    static_configs:
      - targets: ['exporter:{cfg.exporter_port}']
"""
    prom_path = os.path.join(prom_dir, "prometheus.yml")
    with open(prom_path, "w") as f:
        f.write(prom_yml)
    files["prometheus_yml"] = prom_path

    # Grafana datasource provisioning
    gdir = cfg.grafana_dir
    os.makedirs(os.path.join(gdir, "provisioning", "datasources"), exist_ok=True)
    os.makedirs(os.path.join(gdir, "provisioning", "dashboards"), exist_ok=True)
    ds_path = os.path.join(gdir, "provisioning", "datasources", "prometheus.yml")
    with open(ds_path, "w") as f:
        f.write("""apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
""")
    files["datasource"] = ds_path

    db_prov = os.path.join(gdir, "provisioning", "dashboards", "sesi.yml")
    with open(db_prov, "w") as f:
        f.write("""apiVersion: 1
providers:
  - name: SESI
    folder: SESI
    type: file
    options:
      path: /var/lib/grafana/dashboards
""")
    files["dashboard_provider"] = db_prov

    # dashboard json
    db_path = GrafanaDashboardBuilder(cfg).write(os.path.join(gdir, "dashboard.json"))
    files["dashboard"] = db_path

    # docker-compose
    compose_path = os.path.join(cfg.grafana_dir, "..", "docker-compose.monitoring.yml")
    compose_path = os.path.normpath(compose_path)
    with open(compose_path, "w") as f:
        f.write(f"""version: "3.8"
services:
  exporter:
    build: ..
    command: python -m sesi_kafka_prometheus_grafana --serve
    ports: ["{cfg.exporter_port}:{cfg.exporter_port}"]
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./{os.path.basename(cfg.prometheus_dir)}/prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./{os.path.basename(cfg.grafana_dir)}/provisioning:/etc/grafana/provisioning
      - ./{os.path.basename(cfg.grafana_dir)}/dashboard.json:/var/lib/grafana/dashboards/dashboard.json
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=sesi
""")
    files["docker_compose"] = compose_path

    return files


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Kafka→Prometheus→Grafana Stack — Smoke Test")
    print("=" * 72)
    print(f"  prometheus_client : {HAS_PROM}")
    print(f"  kafka module      : {HAS_KAFKA_MOD}")
    print(f"  throttle module   : {HAS_THROTTLE}")

    cfg = MetricsConfig(tau_e=0.01)
    bridge = KafkaMetricsBridge(cfg)
    print(f"\n  source synthetic  : {bridge.synthetic}")

    # Run a few poll cycles
    print("\n  ── Poll cycles ──")
    totals = {"alerts": 0, "candidates": 0, "emitted": 0, "deferred": 0}
    for i in range(4):
        counts = bridge.poll_once()
        for k in totals:
            totals[k] += counts[k]
        time.sleep(0.005)
    print(f"    totals: {totals}")

    # Render metrics
    print("\n  ── /metrics exposition (excerpt) ──")
    text = bridge.exporter.render()
    for line in text.splitlines()[:10]:
        print(f"    {line}")

    # Build monitoring stack files
    print("\n  ── Monitoring stack files ──")
    files = write_monitoring_stack(cfg)
    for name, path in files.items():
        print(f"    {name:20s} → {path}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Live Kafka needs credentials; synthetic source is flagged")
    print("  • Parse failures are counted, NEVER suppressed (cfd_pp.py §3.3)")
    print("  • Throttle controls EMISSION; candidates are always recorded")
    print("  • Launch: docker compose -f docker-compose.monitoring.yml up")
    print("    then open Grafana at http://localhost:3000 (admin/sesi)")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if "--serve" in sys.argv:
        cfg = MetricsConfig()
        bridge = KafkaMetricsBridge(cfg)
        bridge.exporter.serve()
        while True:
            bridge.poll_once()
            time.sleep(1.0)
    else:
        smoke_test()
