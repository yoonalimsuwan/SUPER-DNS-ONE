"""
=============================================================================
SESI GRAFANA ALERT RULES — Pipeline anomaly detection
Prometheus alert rules + Grafana unified alerting for the MM pipeline
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

PURPOSE
-------
sesi_kafka_prometheus_grafana.py exposes metrics. This module defines the
ALERT RULES that fire when the pipeline misbehaves, written both as
Prometheus rule YAML and Grafana unified-alerting JSON.

ALERT PHILOSOPHY (per CFD paper §3.3 & Six Problems §6)
--------------------------------------------------------
The rules encode the project's two non-negotiable monitoring principles:

  1. DEAD-MAN'S SWITCH — a silent pipeline is a FAILURE, not a quiet sky.
     "ConsumerDown" fires when no alerts arrive for too long. This must
     never be masked by a throttle.

  2. INVARIANT VIOLATION IS CRITICAL — if the No-Zeno bound
     N(T) ≤ 1 + ⌊T/τ_e⌋ is ever violated (sesi_throttle_invariant_holds==0),
     that is a logic bug, not a tuning issue → page immediately.

  3. FAILURES ARE NEVER SWALLOWED — parse/consumer failures get their own
     rate alert, so errors cannot be hidden by a dropping message rate.

SEVERITY MODEL
--------------
  critical : logic invariant broken, or pipeline down → page
  warning  : degraded but functioning → ticket / dashboard
  info     : notable but healthy → dashboard annotation

These rules monitor PIPELINE HEALTH, not astrophysics. A burst of
coincidences is interesting but NOT an alert condition here; alerting on
science candidates is a separate, human-in-the-loop process.

LABELS: [C] Prometheus alerting conventions  [D] derived here  [O] open
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("sesi.grafana_alerts")

__all__ = ["AlertSeverity", "PipelineAlertRule", "PipelineAlertLibrary",
           "GrafanaAlertWriter"]


class AlertSeverity:
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ═══════════════════════════════════════════════════════════════════════
# RULE DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineAlertRule:
    """A single Prometheus/Grafana alert rule."""
    name: str
    expr: str
    for_duration: str              # e.g. "5m"
    severity: str
    summary: str
    description: str
    runbook: str = ""              # link to remediation steps

    def to_prometheus_dict(self) -> Dict[str, Any]:
        return {
            "alert": self.name,
            "expr": self.expr,
            "for": self.for_duration,
            "labels": {"severity": self.severity},
            "annotations": {
                "summary": self.summary,
                "description": self.description,
                "runbook_url": self.runbook,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] RULE LIBRARY
# ═══════════════════════════════════════════════════════════════════════

class PipelineAlertLibrary:
    """
    [D] The canonical set of pipeline-health alert rules.

    Each rule targets a metric defined in sesi_kafka_prometheus_grafana.py.
    """

    @staticmethod
    def consumer_down() -> PipelineAlertRule:
        """[CRITICAL] Dead-man's switch: no alerts flowing = pipeline down."""
        return PipelineAlertRule(
            name="SESIPipelineConsumerDown",
            expr='absent(sesi_gcn_alerts_total) or '
                 'rate(sesi_gcn_alerts_total[5m]) == 0',
            for_duration="10m",
            severity=AlertSeverity.CRITICAL,
            summary="Multi-messenger pipeline is not ingesting alerts",
            description=(
                "No GCN alerts have been ingested for 10 minutes. A silent "
                "pipeline is a FAILURE, not a quiet sky. Check the Kafka "
                "consumer connection and credentials."),
            runbook="runbooks/consumer_down.md",
        )

    @staticmethod
    def parse_failures_high() -> PipelineAlertRule:
        """[WARNING] Parse/consumer failures are rising (never swallowed)."""
        return PipelineAlertRule(
            name="SESIPipelineParseFailuresHigh",
            expr='rate(sesi_gcn_parse_failures_total[5m]) > 0.1',
            for_duration="5m",
            severity=AlertSeverity.WARNING,
            summary="GCN message parse failures above threshold",
            description=(
                "More than 0.1 parse failures/sec. Malformed messages may "
                "indicate a schema change in GCN. Failures are recorded, "
                "never suppressed."),
            runbook="runbooks/parse_failures.md",
        )

    @staticmethod
    def nozeno_invariant_violated() -> PipelineAlertRule:
        """[CRITICAL] The No-Zeno event-count bound was broken → logic bug."""
        return PipelineAlertRule(
            name="SESINoZenoInvariantViolated",
            expr='sesi_throttle_invariant_holds == 0',
            for_duration="0s",
            severity=AlertSeverity.CRITICAL,
            summary="No-Zeno throttle invariant N(T)<=1+floor(T/tau_e) violated",
            description=(
                "The event-count bound from CFD paper Prop 1 was violated. "
                "This is a LOGIC BUG in the throttle, not a tuning issue. "
                "The scheduler emitted more events than the minimum-spacing "
                "rule permits. Page immediately and inspect the event log."),
            runbook="runbooks/nozeno_violation.md",
        )

    @staticmethod
    def latency_high() -> PipelineAlertRule:
        """[WARNING] End-to-end latency p95 above the low-latency budget."""
        return PipelineAlertRule(
            name="SESIPipelineLatencyHigh",
            expr=('histogram_quantile(0.95, '
                  'rate(sesi_pipeline_latency_seconds_bucket[5m])) > 1.0'),
            for_duration="5m",
            severity=AlertSeverity.WARNING,
            summary="Pipeline p95 latency above 1 second",
            description=(
                "The low-latency promise is being degraded. For rapid "
                "follow-up, multi-messenger alerts should be processed in "
                "sub-second time. Check the NPE ranking stage and Kafka "
                "consumer lag."),
            runbook="runbooks/latency.md",
        )

    @staticmethod
    def deferred_alerts_accumulating() -> PipelineAlertRule:
        """[WARNING] Too many deferred candidates — throttle may be too tight."""
        return PipelineAlertRule(
            name="SESIDeferredAlertsAccumulating",
            expr='rate(sesi_alerts_deferred_total[5m]) > '
                 'rate(sesi_alerts_emitted_total[5m])',
            for_duration="10m",
            severity=AlertSeverity.WARNING,
            summary="Deferred alerts outnumber emitted alerts",
            description=(
                "The No-Zeno throttle is deferring more candidates than it "
                "emits. Deferred candidates are recorded (never dropped), but "
                "a sustained deferral excess suggests tau_e is too aggressive "
                "or a real burst is occurring. Review before tuning."),
            runbook="runbooks/deferred_accumulation.md",
        )

    @staticmethod
    def coincidence_rate_spike() -> PipelineAlertRule:
        """[INFO] Coincidence rate spike — notable, but NOT an auto-claim."""
        return PipelineAlertRule(
            name="SESICoincidenceRateSpike",
            expr='rate(sesi_coincidences_total[5m]) > '
                 '10 * avg_over_time(rate(sesi_coincidences_total[5m])[1h:5m])',
            for_duration="2m",
            severity=AlertSeverity.INFO,
            summary="Unusually high multi-messenger coincidence rate",
            description=(
                "The coincidence rate is >10x its 1-hour average. This is an "
                "INFORMATIONAL alert: a spike is a candidate for review, NOT a "
                "confirmed astrophysical association. A human must examine the "
                "events before any scientific claim."),
            runbook="runbooks/coincidence_spike.md",
        )

    @classmethod
    def all_rules(cls) -> List[PipelineAlertRule]:
        return [
            cls.consumer_down(),
            cls.parse_failures_high(),
            cls.nozeno_invariant_violated(),
            cls.latency_high(),
            cls.deferred_alerts_accumulating(),
            cls.coincidence_rate_spike(),
        ]


# ═══════════════════════════════════════════════════════════════════════
# [D] WRITER — Prometheus YAML + Grafana unified alerting
# ═══════════════════════════════════════════════════════════════════════

class GrafanaAlertWriter:
    """
    [D] Emit the rule library as:
      • Prometheus alerting-rules YAML (for Prometheus to evaluate), and
      • Grafana unified-alerting rule JSON (for Grafana-managed alerts).
    """

    def __init__(self, rules: Optional[List[PipelineAlertRule]] = None):
        self.rules = rules or PipelineAlertLibrary.all_rules()

    # ── Prometheus YAML ───────────────────────────────────────────────
    def to_prometheus_yaml(self) -> str:
        """Render Prometheus alerting rules as YAML (no external deps)."""
        lines = ["groups:", "  - name: sesi-mm-pipeline", "    rules:"]
        for r in self.rules:
            d = r.to_prometheus_dict()
            lines.append(f"      - alert: {d['alert']}")
            lines.append(f"        expr: {d['expr']}")
            lines.append(f"        for: {d['for']}")
            lines.append("        labels:")
            lines.append(f"          severity: {d['labels']['severity']}")
            lines.append("        annotations:")
            for k, v in d["annotations"].items():
                # quote safely
                v_esc = str(v).replace('"', '\\"')
                lines.append(f'          {k}: "{v_esc}"')
        return "\n".join(lines) + "\n"

    # ── Grafana unified alerting ─────────────────────────────────────
    def to_grafana_json(self, folder: str = "SESI") -> List[Dict[str, Any]]:
        """Render Grafana unified-alerting rule objects."""
        grafana_rules = []
        for r in self.rules:
            grafana_rules.append({
                "title": r.name,
                "condition": "C",
                "data": [
                    {"refId": "A", "relativeTimeRange": {"from": 600, "to": 0},
                     "datasourceUid": "prometheus",
                     "model": {"expr": r.expr, "refId": "A"}},
                    {"refId": "C", "relativeTimeRange": {"from": 0, "to": 0},
                     "datasourceUid": "__expr__",
                     "model": {"type": "threshold", "refId": "C",
                               "conditions": [{"evaluator": {"type": "gt", "params": [0]}}]}},
                ],
                "noDataState": "Alerting",   # dead-man's switch on missing data
                "execErrState": "Alerting",  # query error is itself an alert
                "for": r.for_duration,
                "annotations": {
                    "summary": r.summary,
                    "description": r.description,
                },
                "labels": {"severity": r.severity, "folder": folder},
            })
        return grafana_rules

    # ── File output ──────────────────────────────────────────────────
    def write_files(self, outdir: str = "./monitoring/alerts") -> Dict[str, str]:
        """Write Prometheus rules + Grafana rules + provisioning."""
        os.makedirs(outdir, exist_ok=True)
        paths = {}

        prom_path = os.path.join(outdir, "sesi_pipeline_alerts.yml")
        with open(prom_path, "w") as f:
            f.write(self.to_prometheus_yaml())
        paths["prometheus_rules"] = prom_path

        grafana_path = os.path.join(outdir, "grafana_alert_rules.json")
        with open(grafana_path, "w") as f:
            json.dump(self.to_grafana_json(), f, indent=2)
        paths["grafana_rules"] = grafana_path

        # Prometheus config snippet that loads the rules
        prom_cfg_path = os.path.join(outdir, "prometheus_with_rules.yml")
        with open(prom_cfg_path, "w") as f:
            f.write("""global:
  scrape_interval: 5s
rule_files:
  - "sesi_pipeline_alerts.yml"
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]
scrape_configs:
  - job_name: "sesi-mm-pipeline"
    static_configs:
      - targets: ["exporter:9100"]
""")
        paths["prometheus_config"] = prom_cfg_path

        return paths


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Grafana Alert Rules — Smoke Test")
    print("=" * 72)

    rules = PipelineAlertLibrary.all_rules()
    print(f"\n  ── Rule library ({len(rules)} rules) ──")
    for r in rules:
        print(f"    [{r.severity.upper():8s}] {r.name:<35s} for={r.for_duration}")

    # Verify the two non-negotiable rules exist
    names = [r.name for r in rules]
    has_deadman = "SESIPipelineConsumerDown" in names
    has_invariant = "SESINoZenoInvariantViolated" in names
    print(f"\n    [{'PASS' if has_deadman else 'FAIL'}] dead-man's switch present")
    print(f"    [{'PASS' if has_invariant else 'FAIL'}] No-Zeno invariant rule present")

    # Invariant rule must be critical + immediate
    inv = next(r for r in rules if r.name == "SESINoZenoInvariantViolated")
    print(f"    [{'PASS' if inv.severity == AlertSeverity.CRITICAL else 'FAIL'}] "
          f"invariant rule is CRITICAL")
    print(f"    [{'PASS' if inv.for_duration == '0s' else 'FAIL'}] "
          f"invariant rule fires immediately (for=0s)")

    # Write files
    writer = GrafanaAlertWriter(rules)
    paths = writer.write_files(outdir="./monitoring/alerts_smoke")
    print(f"\n  ── Written files ──")
    for k, p in paths.items():
        exists = os.path.exists(p)
        print(f"    [{'PASS' if exists else 'FAIL'}] {k}: {p}")

    # Show the Prometheus YAML for the critical invariant rule
    print(f"\n  ── Prometheus YAML (excerpt) ──")
    yaml_text = writer.to_prometheus_yaml()
    for line in yaml_text.splitlines():
        if "NoZeno" in line or "invariant" in line.lower():
            print(f"    {line}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • These rules monitor PIPELINE HEALTH, not astrophysics")
    print("  • A coincidence spike is INFO, not an auto-detection")
    print("  • noDataState/execErrState = Alerting keeps the dead-man's")
    print("    switch alive (CFD §3.3: never suppress failure evidence)")
    print("  • Tuning thresholds requires real traffic; values here are")
    print("    sensible defaults, not validated operating points")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
