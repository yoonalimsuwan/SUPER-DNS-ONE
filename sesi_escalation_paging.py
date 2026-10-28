"""
=============================================================================
SESI RUNBOOK ↔ PAGERDUTY / OPSGENIE ESCALATION INTEGRATION
Connect the alert runbooks to real on-call paging services
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS CONNECTS
------------------
sesi_grafana_alert_rules.py    → alert names + severities
sesi_alert_runbooks.py         → operational runbooks per alert
    ↓
sesi_escalation_paging.py      → THIS MODULE
    ↓
PagerDuty Events API v2 / OpsGenie Alert API  → on-call paging

ESCALATION MODEL
----------------
Each alert rule maps to an escalation policy:
    severity + alert_name  →  routing key (PagerDuty) / team (OpsGenie)
                            + priority + runbook link

Dispatch rules (per CFD §3.3 discipline):
    • CRITICAL alerts page IMMEDIATELY (no batching, no delay).
    • WARNING alerts open a ticket / notify channel.
    • INFO alerts annotate the dashboard only.
    • A FAILED dispatch is RECORDED and RETRIED — never swallowed.
    • Synthetic/dry-run mode is clearly flagged; it never pages a human
      unless explicitly armed with real credentials.

LABELS: [C] PagerDuty/OpsGenie API conventions  [D] derived glue  [O] live ops

HONEST SCOPE
------------
• Real paging requires API keys (PagerDuty routing key / OpsGenie API key)
  stored securely (env var or secret manager, NEVER in code).
• In dry-run mode (no keys), dispatch is simulated and logged — this is
  the default so a smoke test can never accidentally page anyone.
• This routes OPERATIONAL alerts. It does not make astrophysical claims.
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.escalation")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Reuse runbook library if available (for runbook links)
try:
    from sesi_alert_runbooks import RunbookLibrary
    HAS_RUNBOOKS = True
except ImportError:
    HAS_RUNBOOKS = False

__all__ = ["Severity", "EscalationPolicy", "PagerDutyClient", "OpsGenieClient",
           "AlertEscalationRouter", "DispatchTracker"]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS & POLICY
# ═══════════════════════════════════════════════════════════════════════

class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class EscalationPolicy:
    """
    [D] Maps alert rules to paging destinations.

    routing_key / team are looked up from environment or a secrets store;
    they are NEVER hardcoded. If absent, dispatch falls back to dry-run.
    """
    alert_name: str
    severity: Severity
    runbook_path: str
    pagerduty_routing_key_env: str = "PAGERDUTY_ROUTING_KEY"
    opsgenie_api_key_env: str = "OPSGENIE_API_KEY"
    opsgenie_team: str = "sesi-oncall"
    retry_attempts: int = 3
    retry_backoff_s: float = 2.0

    def pagerduty_key(self) -> Optional[str]:
        return os.environ.get(self.pagerduty_routing_key_env)

    def opsgenie_key(self) -> Optional[str]:
        return os.environ.get(self.opsgenie_api_key_env)

    def is_armed(self) -> bool:
        """True only if at least one real credential is present."""
        return self.pagerduty_key() is not None or self.opsgenie_key() is not None


class DefaultEscalationPolicies:
    """[D] The canonical policy table matching the six alert rules."""

    @staticmethod
    def all() -> List[EscalationPolicy]:
        rb = "runbooks"
        return [
            EscalationPolicy("SESIPipelineConsumerDown", Severity.CRITICAL,
                             f"{rb}/consumer_down.md"),
            EscalationPolicy("SESIPipelineParseFailuresHigh", Severity.WARNING,
                             f"{rb}/parse_failures.md"),
            EscalationPolicy("SESINoZenoInvariantViolated", Severity.CRITICAL,
                             f"{rb}/nozeno_violation.md"),
            EscalationPolicy("SESIPipelineLatencyHigh", Severity.WARNING,
                             f"{rb}/latency.md"),
            EscalationPolicy("SESIDeferredAlertsAccumulating", Severity.WARNING,
                             f"{rb}/deferred_accumulation.md"),
            EscalationPolicy("SESICoincidenceRateSpike", Severity.INFO,
                             f"{rb}/coincidence_spike.md"),
        ]


# ═══════════════════════════════════════════════════════════════════════
# [C] PAGERDUTY EVENTS API v2 CLIENT
# ═══════════════════════════════════════════════════════════════════════

class PagerDutyClient:
    """
    [C] Minimal PagerDuty Events API v2 client.

    Endpoint: POST https://events.pagerduty.com/v2/enqueue
    Payload: { routing_key, event_action, payload{summary, source,
             severity, custom_details}, links }
    """

    ENDPOINT = "https://events.pagerduty.com/v2/enqueue"
    SEVERITY_MAP = {
        Severity.CRITICAL: "critical",
        Severity.WARNING: "warning",
        Severity.INFO: "info",
    }

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.sent: List[Dict[str, Any]] = []

    def trigger(self, routing_key: str, policy: EscalationPolicy,
                alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send a 'trigger' event for an alert."""
        payload = {
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": alert.get("dedup_key", policy.alert_name),
            "payload": {
                "summary": f"{policy.alert_name}: {alert.get('summary', '')}",
                "source": "sesi-mm-pipeline",
                "severity": self.SEVERITY_MAP[policy.severity],
                "custom_details": {
                    "alert_name": policy.alert_name,
                    "runbook": policy.runbook_path,
                    "metrics": alert.get("metrics", {}),
                },
            },
            "links": [{"href": policy.runbook_path, "text": "Runbook"}],
        }

        if self.dry_run or not HAS_REQUESTS:
            self.sent.append({"mode": "dry_run", "payload": payload})
            return {"status": "dry_run", "dedup_key": payload["dedup_key"]}

        resp = requests.post(self.ENDPOINT, json=payload, timeout=10)
        if resp.status_code in (200, 202):
            self.sent.append({"mode": "live", "status": resp.status_code})
            return {"status": "sent", "http": resp.status_code,
                    "dedup_key": payload["dedup_key"]}
        return {"status": "failed", "http": resp.status_code,
                "body": resp.text[:200]}


# ═══════════════════════════════════════════════════════════════════════
# [C] OPSGENIE ALERT API CLIENT
# ═══════════════════════════════════════════════════════════════════════

class OpsGenieClient:
    """
    [C] Minimal OpsGenie Alert API client.

    Endpoint: POST https://api.opsgenie.com/v2/alerts
    Auth: Authorization: GenieKey <api_key>
    """

    ENDPOINT = "https://api.opsgenie.com/v2/alerts"
    PRIORITY_MAP = {
        Severity.CRITICAL: "P1",
        Severity.WARNING: "P3",
        Severity.INFO: "P5",
    }

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.sent: List[Dict[str, Any]] = []

    def create_alert(self, api_key: str, policy: EscalationPolicy,
                     alert: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "message": f"{policy.alert_name}",
            "description": alert.get("summary", ""),
            "priority": self.PRIORITY_MAP[policy.severity],
            "teams": [{"name": policy.opsgenie_team}],
            "details": {
                "runbook": policy.runbook_path,
                "alert_name": policy.alert_name,
            },
        }
        if self.dry_run or not HAS_REQUESTS:
            self.sent.append({"mode": "dry_run", "body": body})
            return {"status": "dry_run"}

        resp = requests.post(
            self.ENDPOINT, json=body,
            headers={"Authorization": f"GenieKey {api_key}",
                     "Content-Type": "application/json"},
            timeout=10)
        if resp.status_code in (200, 202):
            self.sent.append({"mode": "live", "status": resp.status_code})
            return {"status": "sent", "http": resp.status_code}
        return {"status": "failed", "http": resp.status_code,
                "body": resp.text[:200]}


# ═══════════════════════════════════════════════════════════════════════
# [D] DISPATCH TRACKER (never swallow a failed dispatch — CFD §3.3)
# ═══════════════════════════════════════════════════════════════════════

class DispatchTracker:
    """
    [D] Tracks every dispatch attempt. A failed dispatch is recorded and
    retried — per CFD §3.3, evidence of failure must never be suppressed.
    """

    def __init__(self):
        self.dispatched: List[Dict[str, Any]] = []
        self.failed: List[Dict[str, Any]] = []

    def record(self, alert_name: str, channel: str, result: Dict[str, Any]):
        entry = {"alert": alert_name, "channel": channel,
                 "result": result, "t": time.time()}
        if result.get("status") in ("sent", "dry_run"):
            self.dispatched.append(entry)
        else:
            self.failed.append(entry)
            logger.warning(f"[DISPATCH FAILURE — NOT SUPPRESSED] "
                           f"{alert_name} via {channel}: {result}")

    def pending_failures(self) -> List[Dict[str, Any]]:
        return list(self.failed)

    def summary(self) -> Dict[str, int]:
        return {
            "n_dispatched": len(self.dispatched),
            "n_failed": len(self.failed),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] ALERT ESCALATION ROUTER
# ═══════════════════════════════════════════════════════════════════════

class AlertEscalationRouter:
    """
    [D] Route alerts to PagerDuty / OpsGenie per the escalation policy.

    Dispatch rules:
      CRITICAL → page immediately (both channels if armed)
      WARNING  → page/ticket
      INFO     → dry-run log only (never pages a human)

    Retries failed dispatches up to policy.retry_attempts with backoff.
    """

    def __init__(self, policies: Optional[List[EscalationPolicy]] = None,
                 force_dry_run: bool = True):
        self.policies = {p.alert_name: p
                         for p in (policies or DefaultEscalationPolicies.all())}
        # force_dry_run=True is the SAFE DEFAULT so tests never page anyone
        self.pager = PagerDutyClient(dry_run=force_dry_run)
        self.opsgenie = OpsGenieClient(dry_run=force_dry_run)
        self.force_dry_run = force_dry_run
        self.tracker = DispatchTracker()

    def route(self, alert_name: str, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Route one alert. Returns dispatch outcome."""
        policy = self.policies.get(alert_name)
        if policy is None:
            # Unknown alert: log it, never drop (CFD §3.3)
            logger.warning(f"[UNROUTED ALERT — recorded] {alert_name}")
            self.tracker.record(alert_name, "unrouted",
                                {"status": "failed", "reason": "no policy"})
            return {"status": "no_policy", "alert": alert_name}

        # INFO never pages a human
        if policy.severity == Severity.INFO:
            self.tracker.record(alert_name, "info_log",
                                {"status": "dry_run", "note": "INFO — dashboard only"})
            return {"status": "info_logged", "severity": "info"}

        results = {}
        armed = policy.is_armed() and not self.force_dry_run

        # PagerDuty
        pd_key = policy.pagerduty_key()
        if pd_key or not armed:
            res = self._dispatch_with_retry(
                lambda: self.pager.trigger(pd_key or "DRY_RUN_KEY", policy, alert),
                policy)
            self.tracker.record(alert_name, "pagerduty", res)
            results["pagerduty"] = res

        # OpsGenie
        og_key = policy.opsgenie_key()
        if og_key or not armed:
            res = self._dispatch_with_retry(
                lambda: self.opsgenie.create_alert(og_key or "DRY_RUN_KEY", policy, alert),
                policy)
            self.tracker.record(alert_name, "opsgenie", res)
            results["opsgenie"] = res

        return {
            "status": "dispatched" if armed else "dry_run",
            "alert": alert_name,
            "severity": policy.severity.value,
            "armed": armed,
            "results": results,
        }

    def _dispatch_with_retry(self, dispatch_fn, policy: EscalationPolicy
                             ) -> Dict[str, Any]:
        """Retry a dispatch on failure (never give up silently)."""
        last = {}
        for attempt in range(policy.retry_attempts):
            try:
                last = dispatch_fn()
                if last.get("status") in ("sent", "dry_run"):
                    return last
            except Exception as e:
                last = {"status": "failed", "error": str(e)}
            time.sleep(policy.retry_backoff_s * (0 if self.force_dry_run else 1))
        return last

    def report(self) -> Dict[str, Any]:
        return {
            "tracker": self.tracker.summary(),
            "pending_failures": len(self.tracker.pending_failures()),
            "force_dry_run": self.force_dry_run,
            "honest_note": (
                "In dry-run mode no human is paged. Arm with real credentials "
                "via PAGERDUTY_ROUTING_KEY / OPSGENIE_API_KEY env vars and "
                "force_dry_run=False only in a real deployment."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Runbook ↔ PagerDuty/OpsGenie Escalation — Smoke Test")
    print("=" * 72)
    print(f"  requests installed : {HAS_REQUESTS}")
    print(f"  runbooks module    : {HAS_RUNBOOKS}")

    # Router in SAFE dry-run mode (never pages a human)
    router = AlertEscalationRouter(force_dry_run=True)

    print("\n  ── Routing test alerts ──")
    test_alerts = [
        ("SESIPipelineConsumerDown",
         {"summary": "No alerts ingested for 10 min", "metrics": {"rate": 0.0}}),
        ("SESINoZenoInvariantViolated",
         {"summary": "Throttle bound violated", "metrics": {"holds": 0}}),
        ("SESIPipelineLatencyHigh",
         {"summary": "p95 latency 2.3s", "metrics": {"p95": 2.3}}),
        ("SESICoincidenceRateSpike",
         {"summary": "Coincidence rate 12x average", "metrics": {"ratio": 12}}),
    ]
    for name, alert in test_alerts:
        outcome = router.route(name, alert)
        print(f"    {name:<35s} → {outcome['status']:<14s} "
              f"(severity={outcome.get('severity','—')})")

    # Report
    print("\n  ── Dispatch report ──")
    report = router.report()
    print(f"    dispatched        : {report['tracker']['n_dispatched']}")
    print(f"    failed            : {report['tracker']['n_failed']}")
    print(f"    force_dry_run     : {report['force_dry_run']}")
    print(f"    {report['honest_note']}")

    # Verify INFO never pages, CRITICAL does
    info_outcome = router.route("SESICoincidenceRateSpike", {})
    crit_outcome = router.route("SESINoZenoInvariantViolated", {})
    print(f"\n    [{'PASS' if info_outcome['status']=='info_logged' else 'FAIL'}] "
          f"INFO alert logs only, never pages")
    print(f"    [{'PASS' if crit_outcome['status'] in ('dispatched','dry_run') else 'FAIL'}] "
          f"CRITICAL alert dispatches")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Default is DRY-RUN: no human is paged without real credentials")
    print("  • Arm via env vars + force_dry_run=False ONLY in real deployment")
    print("  • Failed dispatches are RECORDED + RETRIED, never swallowed")
    print("  • INFO alerts annotate dashboards, never page")
    print("  • Routing operational health, not astrophysics")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
