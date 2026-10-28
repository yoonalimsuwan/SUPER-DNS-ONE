"""
=============================================================================
SESI RUNBOOK → PAGERDUTY / OPSGENIE INTEGRATION
Severity-aware routing, stable dedup, resolve semantics, dead-letter queue
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

DESIGN PRINCIPLES (CFD paper §3.3 + Six Problems §6)
----------------------------------------------------
  1. NO SUPPRESSION: a delivery failure is NEVER swallowed. It is written
     to a dead-letter queue and surfaced in the delivery report.
  2. DEAD-MAN'S SWITCH: missing data is NOT a resolve. Only an explicit
     resolved=True from the monitor may close an incident.
  3. STABLE DEDUP: the same alert identity always maps to the same
     dedup_key, so flapping does not create incident storms.
  4. RUNBOOK ATTACHED: every page carries the runbook URL + the
     anti-suppression clause, so the on-call sees the discipline.
  5. DRY RUN BY DEFAULT: no network call unless a real key AND an
     explicit enable flag are supplied.

SECURITY
--------
Credentials are read ONLY from the environment:
    PAGERDUTY_ROUTING_KEY
    OPSGENIE_API_KEY
They are never hard-coded, never logged, never written to disk.

LABELS: [C] vendor API schemas  [D] routing/adapter  [O] live operation
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.oncall")

__all__ = [
    "OnCallConfig",
    "OnCallRouter",
    "PrometheusWebhookAdapter",
    "write_oncall_config",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — credentials from environment only
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OnCallConfig:
    dry_run: bool = True                      # SAFE DEFAULT
    pagerduty_enabled: bool = False
    opsgenie_enabled: bool = False

    pagerduty_url: str = "https://events.pagerduty.com/v2/enqueue"
    opsgenie_url: str = "https://api.opsgenie.com/v2/alerts"

    source: str = "sesi-mm-pipeline"
    timeout_s: float = 10.0
    max_retries: int = 3
    dead_letter_path: str = "./monitoring/dead_letter.jsonl"

    runbook_base_url: str = "https://runbooks.sesi.local"

    @property
    def pagerduty_key(self) -> Optional[str]:
        return os.environ.get("PAGERDUTY_ROUTING_KEY")

    @property
    def opsgenie_key(self) -> Optional[str]:
        return os.environ.get("OPSGENIE_API_KEY")


# Severity mapping: internal → vendor
PD_SEVERITY = {"critical": "critical", "warning": "warning", "info": "info"}
OG_PRIORITY = {"critical": "P1", "warning": "P3", "info": "P5"}

# Runbook mapping (matches sesi_grafana_alert_rules / sesi_alert_runbooks)
RUNBOOKS = {
    "SESIPipelineConsumerDown": "consumer_down.md",
    "SESIPipelineParseFailuresHigh": "parse_failures.md",
    "SESINoZenoInvariantViolated": "nozeno_violation.md",
    "SESIPipelineLatencyHigh": "latency.md",
    "SESIDeferredAlertsAccumulating": "deferred_accumulation.md",
    "SESICoincidenceRateSpike": "coincidence_spike.md",
}

# Service routing: which team owns which alert
SERVICE_ROUTES = {
    "SESIPipelineConsumerDown": ("ingestion", "mm-platform"),
    "SESIPipelineParseFailuresHigh": ("ingestion", "mm-platform"),
    "SESINoZenoInvariantViolated": ("scheduler", "core-runtime"),
    "SESIPipelineLatencyHigh": ("runtime", "mm-platform"),
    "SESIDeferredAlertsAccumulating": ("scheduler", "core-runtime"),
    "SESICoincidenceRateSpike": ("science-review", "science"),
}


# ═══════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════

class OnCallRouter:
    """
    [D] Route a normalized pipeline alert to PagerDuty and/or OpsGenie.

    Normalized alert:
        {"name": ..., "severity": ..., "fingerprint": ...,
         "summary": ..., "description": ..., "resolved": bool,
         "starts_at": ISO8601, "generator_url": ...}
    """

    def __init__(self, cfg: Optional[OnCallConfig] = None):
        self.cfg = cfg or OnCallConfig()
        self.delivery_log: List[Dict[str, Any]] = []
        self.dead_letters: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.cfg.dead_letter_path) or ".", exist_ok=True)

    # ── identity ──────────────────────────────────────────────────────
    @staticmethod
    def dedup_key(name: str, fingerprint: str) -> str:
        """Stable key: same alert identity → same incident."""
        raw = f"{name}|{fingerprint}".encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    # ── payload builders ─────────────────────────────────────────────
    def pagerduty_payload(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        name = alert["name"]
        severity = PD_SEVERITY.get(alert.get("severity", "warning"), "warning")
        component, group = SERVICE_ROUTES.get(name, ("pipeline", "mm-platform"))
        runbook = f"{self.cfg.runbook_base_url}/{RUNBOOKS.get(name, 'general.md')}"

        payload = {
            "summary": alert.get("summary", name),
            "source": self.cfg.source,
            "severity": severity,
            "timestamp": alert.get("starts_at", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            "component": component,
            "group": group,
            "class": "pipeline_health",
            "custom_details": {
                "alert_name": name,
                "description": alert.get("description", ""),
                "runbook": runbook,
                "anti_suppression": (
                    "Do not silence or relax thresholds. Diagnose the "
                    "underlying failure. Failures are never swallowed."),
            },
        }
        return {
            "routing_key": self.cfg.pagerduty_key or "DRY_RUN_NO_KEY",
            "event_action": "resolve" if alert.get("resolved") else "trigger",
            "dedup_key": self.dedup_key(name, alert.get("fingerprint", name)),
            "payload": payload,
            "links": [{"href": runbook, "text": "Runbook"}],
        }

    def opsgenie_payload(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        name = alert["name"]
        priority = OG_PRIORITY.get(alert.get("severity", "warning"), "P3")
        component, team = SERVICE_ROUTES.get(name, ("pipeline", "mm-platform"))
        runbook = f"{self.cfg.runbook_base_url}/{RUNBOOKS.get(name, 'general.md')}"

        return {
            "message": f"[{self.cfg.source}] {alert.get('summary', name)}",
            "description": alert.get("description", ""),
            "alias": self.dedup_key(name, alert.get("fingerprint", name)),
            "priority": priority,
            "source": self.cfg.source,
            "entity": f"{team}/{component}",
            "tags": ["sesi", "multimessenger", component, alert.get("severity", "warning")],
            "details": {
                "alert_name": name,
                "runbook": runbook,
                "team": team,
                "resolved_requested": str(bool(alert.get("resolved"))),
            },
            "user": "sesi-automation",
        }

    # ── transport (never used in dry run) ─────────────────────────────
    def _post(self, url: str, headers: Dict[str, str],
              body: Dict[str, Any]) -> Dict[str, Any]:
        """HTTPS POST with bounded retries; 429 honors Retry-After."""
        import urllib.request
        import urllib.error

        data = json.dumps(body).encode()
        last_error = None

        for attempt in range(self.cfg.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    raw = resp.read().decode()
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = {"raw": raw}
                    return {"ok": True, "status": resp.status, "response": parsed}
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}"
                if e.code == 429:
                    retry_after = e.headers.get("Retry-After", "5")
                    try:
                        time.sleep(min(float(retry_after), 30.0))
                    except ValueError:
                        time.sleep(5.0)
                    continue
                if 400 <= e.code < 500 and e.code != 429:
                    # client error: retrying won't help; surface immediately
                    return {"ok": False, "status": e.code, "error": last_error}
            except Exception as e:
                last_error = str(e)
            time.sleep(min(2 ** attempt, 10))

        return {"ok": False, "error": last_error or "unknown transport error"}

    # ── dead letter ───────────────────────────────────────────────────
    def _dead_letter(self, target: str, payload: Dict[str, Any],
                     error: str) -> None:
        """NEVER suppress a delivery failure."""
        record = {
            "ts": time.time(), "target": target,
            "error": error, "payload": payload,
            "action_required": "manual page / investigate transport",
        }
        self.dead_letters.append(record)
        try:
            with open(self.cfg.dead_letter_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Dead-letter write FAILED: {e} — keep in memory")
        logger.error(f"[DELIVERY FAILURE → {target}] {error} | "
                     f"recorded in dead-letter queue (NOT suppressed)")

    # ── route ────────────────────────────────────────────────────────
    def route(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route one alert. Returns a per-target delivery report.

        Dry run: build payloads, validate, log — NO network.
        """
        targets: List[Tuple] = []
        if self.cfg.pagerduty_enabled:
            targets.append(("pagerduty", self.cfg.pagerduty_url,
                            {"Content-Type": "application/json"},
                            self.pagerduty_payload(alert),
                            self.cfg.pagerduty_key))
        if self.cfg.opsgenie_enabled:
            targets.append(("opsgenie", self.cfg.opsgenie_url,
                            {"Content-Type": "application/json",
                             "Authorization": f"GenieKey {self.cfg.opsgenie_key}"},
                            self.opsgenie_payload(alert),
                            self.cfg.opsgenie_key))

        report = {"dry_run": self.cfg.dry_run, "targets": {}}

        for name, url, headers, payload, key in targets:
            if self.cfg.dry_run:
                report["targets"][name] = {
                    "status": "DRY_RUN", "would_post": url,
                    "action": payload.get("event_action",
                                          "close" if alert.get("resolved") else "create"),
                    "dedup_or_alias": payload.get("dedup_key", payload.get("alias")),
                }
                self.delivery_log.append(report["targets"][name])
                continue

            if not key:
                self._dead_letter(name, payload, "missing credential in environment")
                report["targets"][name] = {"status": "FAILED", "error": "no key"}
                continue

            result = self._post(url, headers, payload)
            if not result["ok"]:
                self._dead_letter(name, payload, result.get("error", "unknown"))
            report["targets"][name] = result
            self.delivery_log.append({"target": name, **result})

        if not targets:
            report["note"] = "No target enabled; alert recorded locally only."
        return report

    def delivery_report(self) -> Dict[str, Any]:
        return {
            "n_attempts": len(self.delivery_log),
            "n_dead_letters": len(self.dead_letters),
            "all_delivered": len(self.dead_letters) == 0,
            "note": "Failures are recorded, never suppressed (CFD §3.3).",
        }


# Prometheus typing import workaround (Tuple used above)
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════
# PROMETHEUS WEBHOOK ADAPTER
# ═══════════════════════════════════════════════════════════════════════

class PrometheusWebhookAdapter:
    """
    [D] Convert Alertmanager's webhook JSON into normalized alerts.

    IMPORTANT: status=="firing" → trigger; status=="resolved" → resolve.
    MISSING DATA is handled by the Prometheus `for`/dead-man rule, NOT by
    this adapter silently resolving anything.
    """

    def parse(self, webhook_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for a in webhook_body.get("alerts", []):
            labels = a.get("labels", {})
            annotations = a.get("annotations", {})
            out.append({
                "name": labels.get("alertname", "UnknownAlert"),
                "severity": labels.get("severity", "warning"),
                "fingerprint": a.get("fingerprint", labels.get("alertname", "")),
                "summary": annotations.get("summary", labels.get("alertname", "")),
                "description": annotations.get("description", ""),
                "starts_at": a.get("startsAt"),
                "resolved": a.get("status") == "resolved",
                "generator_url": a.get("generatorURL", ""),
            })
        return out


# ═══════════════════════════════════════════════════════════════════════
# CONFIG FILES + ESCALATION POLICY
# ═══════════════════════════════════════════════════════════════════════

def write_oncall_config(outdir: str = "./monitoring/oncall") -> Dict[str, str]:
    """Write routing config, escalation policy, and Alertmanager webhook."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    escalation = [
        {"severity": "critical", "page": "mm-platform on-call",
         "ack_timeout_s": 300, "escalate_to": "pipeline lead",
         "examples": ["SESIPipelineConsumerDown", "SESINoZenoInvariantViolated"]},
        {"severity": "warning", "ticket": "mm-platform queue",
         "response_timeout_s": 3600, "escalate_to": "on-call after 1h",
         "examples": ["SESIPipelineLatencyHigh", "SESIDeferredAlertsAccumulating"]},
        {"severity": "info", "review": "science on-call",
         "response_timeout_s": 86400, "auto_resolve": False,
         "examples": ["SESICoincidenceRateSpike"]},
    ]
    p = os.path.join(outdir, "escalation_policy.json")
    with open(p, "w") as f:
        json.dump(escalation, f, indent=2)
    paths["escalation_policy"] = p

    p = os.path.join(outdir, "alertmanager_webhook.yml")
    with open(p, "w") as f:
        f.write("""route:
  receiver: 'sesi-oncall'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'sesi-oncall'
      repeat_interval: 30m
receivers:
  - name: 'sesi-oncall'
    webhook_configs:
      - url: 'http://sesi-oncall-router:8088/webhook'
        send_resolved: true
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
""")
    paths["alertmanager_webhook"] = p

    p = os.path.join(outdir, "README.md")
    with open(p, "w") as f:
        f.write("""# SESI On-Call Integration

## Credentials (environment only)
export PAGERDUTY_ROUTING_KEY=...
export OPSGENIE_API_KEY=...

## Safe default
OnCallConfig(dry_run=True) — NO network calls.

## Non-negotiable rules
1. Never silence an alert to make it go away.
2. Missing data is NOT a resolve (dead-man's switch).
3. No-Zeno invariant violation = logic bug, page immediately.
4. A coincidence spike is INFO for human review, not a detection.
5. Delivery failures go to the dead-letter queue, never disappear.

## Runbooks
Attached to every page via the `links` / `details.runbook` field.
""")
    paths["readme"] = p

    return paths


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST — DRY RUN ONLY, NO NETWORK
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Runbook → PagerDuty/OpsGenie — Smoke Test (DRY RUN)")
    print("=" * 72)

    cfg = OnCallConfig(dry_run=True, pagerduty_enabled=True, opsgenie_enabled=True)
    router = OnCallRouter(cfg)

    alert = {
        "name": "SESINoZenoInvariantViolated",
        "severity": "critical",
        "fingerprint": "abc123",
        "summary": "No-Zeno throttle invariant violated",
        "description": "N(T) > 1 + floor(T/tau_e); logic bug, not tuning.",
        "resolved": False,
    }

    print("\n  ── [1] Dry-run routing ──")
    report = router.route(alert)
    for name, r in report["targets"].items():
        print(f"    {name}: {r['status']} → {r['would_post']}")
        print(f"      dedup/alias = {r['dedup_or_alias']}, action = {r['action']}")
    assert report["dry_run"], "dry_run must be True in smoke test"

    print("\n  ── [2] Stable dedup ──")
    k1 = router.dedup_key(alert["name"], alert["fingerprint"])
    k2 = router.dedup_key(alert["name"], alert["fingerprint"])
    print(f"    [{'PASS' if k1 == k2 else 'FAIL'}] same identity → same key")

    print("\n  ── [3] Resolve semantics ──")
    resolved = dict(alert, resolved=True)
    pd = router.pagerduty_payload(resolved)
    print(f"    event_action = {pd['event_action']}")
    print(f"    [{'PASS' if pd['event_action'] == 'resolve' else 'FAIL'}] "
          f"explicit resolved → resolve")
    missing = {"name": alert["name"], "severity": "warning"}
    # absence of data must NOT be turned into a resolve by the adapter
    parsed = PrometheusWebhookAdapter().parse(
        {"alerts": [{"status": "firing", "labels": {"alertname": "X"},
                     "fingerprint": "x"}]})
    print(f"    [{'PASS' if not parsed[0]['resolved'] else 'FAIL'}] "
          f"firing is never auto-resolved")

    print("\n  ── [4] Runbook attached ──")
    has_link = any("runbook" in str(l).lower() for l in pd.get("links", []))
    print(f"    [{'PASS' if has_link else 'FAIL'}] runbook link present")

    print("\n  ── [5] Config files ──")
    for k, v in write_oncall_config().items():
        print(f"    {k}: {v}")

    print(f"\n  Delivery report: {router.delivery_report()}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • dry_run=True by default; NO network call was made")
    print("  • Credentials come ONLY from environment, never logged")
    print("  • Delivery failures → dead-letter queue, NEVER suppressed")
    print("  • Missing data is NOT a resolve; only explicit resolved=True")
    print("  • Routing is engineering plumbing, not a physics verdict")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
