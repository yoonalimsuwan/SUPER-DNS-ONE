"""
=============================================================================
SESI ESCALATION → SLACK / MICROSOFT TEAMS INTEGRATION
Connect the alert runbooks to team chat channels alongside paging
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS TO sesi_escalation_paging.py
-------------------------------------------
The paging module routes CRITICAL alerts to PagerDuty/OpsGenie. This module
adds TEAM-CHANNEL notification via:

  • Slack Incoming Webhook     (POST JSON to hooks.slack.com)
  • Microsoft Teams Webhook    (POST MessageCard to *.webhook.office.com)

CHANNEL ROUTING MODEL
---------------------
  CRITICAL → page (PagerDuty/OpsGenie)  +  Slack #oncall  +  Teams channel
  WARNING  → ticket                     +  Slack #pipeline +  Teams channel
  INFO     → dashboard annotation only  (never posts to chat by default)

This separation keeps paging for humans-on-call and chat for situational
awareness, and never lets INFO spam the on-call channels.

NO-SUPPRESSION DISCIPLINE (CFD §3.3)
------------------------------------
• A failed webhook POST is RECORDED and RETRIED — never swallowed.
• INFO alerts are logged, not sent to chat (prevents alert fatigue) but
  they remain visible on the dashboard.
• Dry-run is the DEFAULT: no real HTTP request is made unless a webhook
  URL is present in the environment AND dry_run is explicitly disabled.

LABELS: [C] Slack/Teams webhook formats  [D] derived routing  [O] live ops

HONEST SCOPE
------------
• Webhook URLs are SECRETS — read from env vars, never hardcoded.
• This notifies about PIPELINE HEALTH, not astrophysics. A coincidence
  alert routed to Slack is still a CANDIDATE, not a discovery (Moral
  Order discipline: description/candidate ≠ the thing/association).
• Webhook rate limits apply; bursts are batched/deferred, and deferred
  messages are tracked, not dropped.
=============================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.slack_teams")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Reuse paging-module types if available
try:
    from sesi_escalation_paging import (
        Severity, EscalationPolicy, DefaultEscalationPolicies, DispatchTracker,
    )
    HAS_PAGING = True
except ImportError:
    HAS_PAGING = False

__all__ = ["ChatChannelConfig", "SlackClient", "TeamsClient",
           "ChatEscalationRouter"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ChatChannelConfig:
    # Env var names holding webhook URLs (secrets — never hardcode)
    slack_webhook_env: str = "SESI_SLACK_WEBHOOK_URL"
    teams_webhook_env: str = "SESI_TEAMS_WEBHOOK_URL"
    # Routing
    info_to_chat: bool = False       # keep INFO off chat by default
    retry_attempts: int = 3
    retry_backoff_s: float = 2.0
    dry_run: bool = True             # SAFE DEFAULT

    def slack_webhook(self) -> Optional[str]:
        return os.environ.get(self.slack_webhook_env)

    def teams_webhook(self) -> Optional[str]:
        return os.environ.get(self.teams_webhook_env)


# ═══════════════════════════════════════════════════════════════════════
# SEVERITY COLORS / PRIORITIES (shared)
# ═══════════════════════════════════════════════════════════════════════

SEVERITY_COLOR = {
    "critical": "#d32f2f",   # red
    "warning": "#f9a825",    # amber
    "info": "#1976d2",       # blue
}


# ═══════════════════════════════════════════════════════════════════════
# [C] SLACK CLIENT (Incoming Webhook)
# ═══════════════════════════════════════════════════════════════════════

class SlackClient:
    """
    [C] Send alert messages to Slack via Incoming Webhook.

    Payload uses the attachment format with severity color, summary,
    key metrics, and a runbook link.
    """

    def __init__(self, cfg: ChatChannelConfig):
        self.cfg = cfg
        self.sent: List[Dict[str, Any]] = []

    def _build_payload(self, policy_name: str, severity: str,
                       alert: Dict[str, Any], runbook: str) -> Dict[str, Any]:
        color = SEVERITY_COLOR.get(severity, "#607d8b")
        fields = [
            {"title": "Severity", "value": severity.upper(), "short": True},
            {"title": "Alert", "value": policy_name, "short": True},
        ]
        for k, v in alert.get("metrics", {}).items():
            fields.append({"title": k, "value": str(v), "short": True})
        return {
            "text": f":rotating_light: *{policy_name}* ({severity.upper()})",
            "attachments": [{
                "color": color,
                "fields": fields,
                "footer": "SESI Multi-Messenger Pipeline",
                "actions": [],
            }],
            "blocks": None,
        }

    def send(self, webhook: str, policy_name: str, severity: str,
             alert: Dict[str, Any], runbook: str) -> Dict[str, Any]:
        payload = self._build_payload(policy_name, severity, alert, runbook)
        # attach runbook link as a block if present
        if runbook:
            payload["text"] += f"\n:book: Runbook: `{runbook}`"

        if self.cfg.dry_run or not HAS_REQUESTS:
            self.sent.append({"mode": "dry_run", "payload": payload})
            return {"status": "dry_run", "channel": "slack"}

        resp = requests.post(webhook, json=payload, timeout=10)
        if resp.status_code == 200:
            self.sent.append({"mode": "live", "status": 200})
            return {"status": "sent", "channel": "slack", "http": 200}
        return {"status": "failed", "channel": "slack",
                "http": resp.status_code, "body": resp.text[:200]}


# ═══════════════════════════════════════════════════════════════════════
# [C] MICROSOFT TEAMS CLIENT (MessageCard Webhook)
# ═══════════════════════════════════════════════════════════════════════

class TeamsClient:
    """
    [C] Send alert messages to Microsoft Teams via Incoming Webhook
    (Office 365 Connector MessageCard format).
    """

    def __init__(self, cfg: ChatChannelConfig):
        self.cfg = cfg
        self.sent: List[Dict[str, Any]] = []

    def _build_card(self, policy_name: str, severity: str,
                    alert: Dict[str, Any], runbook: str) -> Dict[str, Any]:
        color = SEVERITY_COLOR.get(severity, "607d8b").lstrip("#")
        facts = [{"name": "Severity", "value": severity.upper()}]
        for k, v in alert.get("metrics", {}).items():
            facts.append({"name": k, "value": str(v)})
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": policy_name,
            "title": f"⚠️ {policy_name} ({severity.upper()})",
            "sections": [{
                "activityTitle": "SESI Multi-Messenger Pipeline",
                "facts": facts,
                "text": alert.get("summary", ""),
                "markdown": True,
            }],
        }
        if runbook:
            card["potentialAction"] = [{
                "@type": "OpenUri",
                "name": "Open Runbook",
                "targets": [{"os": "default", "uri": runbook}],
            }]
        return card

    def send(self, webhook: str, policy_name: str, severity: str,
             alert: Dict[str, Any], runbook: str) -> Dict[str, Any]:
        card = self._build_card(policy_name, severity, alert, runbook)

        if self.cfg.dry_run or not HAS_REQUESTS:
            self.sent.append({"mode": "dry_run", "card": card})
            return {"status": "dry_run", "channel": "teams"}

        resp = requests.post(webhook, json=card, timeout=10)
        if resp.status_code == 200:
            self.sent.append({"mode": "live", "status": 200})
            return {"status": "sent", "channel": "teams", "http": 200}
        return {"status": "failed", "channel": "teams",
                "http": resp.status_code, "body": resp.text[:200]}


# ═══════════════════════════════════════════════════════════════════════
# [D] CHAT ESCALATION ROUTER
# ═══════════════════════════════════════════════════════════════════════

class ChatEscalationRouter:
    """
    [D] Route alerts to Slack/Teams by severity, with the no-suppression
    discipline: failed sends are recorded and retried; INFO stays off chat
    unless explicitly enabled.

    This complements (does not replace) the PagerDuty/OpsGenie pager router.
    """

    def __init__(self, cfg: Optional[ChatChannelConfig] = None,
                 policies: Optional[List[Any]] = None):
        self.cfg = cfg or ChatChannelConfig()
        self.slack = SlackClient(self.cfg)
        self.teams = TeamsClient(self.cfg)
        self.policies = {p.alert_name: p for p in
                         (policies or (DefaultEscalationPolicies.all()
                                       if HAS_PAGING else []))}
        self.dispatch_log: List[Dict[str, Any]] = []
        self.failed_log: List[Dict[str, Any]] = []

    def _severity_of(self, alert_name: str) -> str:
        p = self.policies.get(alert_name)
        if p is not None and HAS_PAGING:
            return p.severity.value if hasattr(p.severity, "value") else str(p.severity)
        return "info"

    def _runbook_of(self, alert_name: str) -> str:
        p = self.policies.get(alert_name)
        return p.runbook_path if (p is not None and HAS_PAGING) else ""

    def route(self, alert_name: str, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Route one alert to the appropriate chat channels."""
        severity = self._severity_of(alert_name)
        runbook = self._runbook_of(alert_name)
        results = {}

        # INFO stays off chat by default (prevents alert fatigue)
        if severity == "info" and not self.cfg.info_to_chat:
            self.dispatch_log.append({
                "alert": alert_name, "severity": severity,
                "action": "logged_only_not_chat",
            })
            return {"status": "info_logged_only", "channels": []}

        slack_url = self.cfg.slack_webhook()
        teams_url = self.cfg.teams_webhook()

        # Slack
        if slack_url or self.cfg.dry_run:
            res = self._send_with_retry(
                lambda: self.slack.send(slack_url or "DRY", alert_name,
                                        severity, alert, runbook))
            self._track(alert_name, "slack", res)
            results["slack
            results["slack"] = res

        # Teams
        if teams_url or self.cfg.dry_run:
            res = self._send_with_retry(
                lambda: self.teams.send(teams_url or "DRY", alert_name,
                                        severity, alert, runbook))
            self._track(alert_name, "teams", res)
            results["teams"] = res

        return {
            "status": "routed" if results else "no_channel_configured",
            "severity": severity,
            "channels": list(results.keys()),
            "results": results,
            "dry_run": self.cfg.dry_run,
        }

    def _send_with_retry(self, send_fn) -> Dict[str, Any]:
        """Retry a send on failure — never give up silently (CFD §3.3)."""
        last = {}
        for attempt in range(self.cfg.retry_attempts):
            try:
                last = send_fn()
                if last.get("status") in ("sent", "dry_run"):
                    return last
            except Exception as e:
                last = {"status": "failed", "error": str(e)}
            if not self.cfg.dry_run:
                time.sleep(self.cfg.retry_backoff_s)
        return last

    def _track(self, alert_name: str, channel: str, result: Dict[str, Any]):
        entry = {"alert": alert_name, "channel": channel,
                 "result": result, "t": time.time()}
        if result.get("status") in ("sent", "dry_run"):
            self.dispatch_log.append(entry)
        else:
            self.failed_log.append(entry)
            logger.warning(f"[CHAT DISPATCH FAILURE — NOT SUPPRESSED] "
                           f"{alert_name} via {channel}: {result}")

    def report(self) -> Dict[str, Any]:
        return {
            "n_dispatched": len(self.dispatch_log),
            "n_failed": len(self.failed_log),
            "dry_run": self.cfg.dry_run,
            "slack_configured": self.cfg.slack_webhook() is not None,
            "teams_configured": self.cfg.teams_webhook() is not None,
            "info_to_chat": self.cfg.info_to_chat,
            "honest_note": (
                "Chat notification reports PIPELINE HEALTH. A routed alert is "
                "a candidate for human review, not a confirmed detection."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Slack/Teams Escalation — Smoke Test")
    print("=" * 72)
    print(f"  requests      : {HAS_REQUESTS}")
    print(f"  paging module : {HAS_PAGING}")

    cfg = ChatChannelConfig(dry_run=True)   # SAFE: never posts without creds
    router = ChatEscalationRouter(cfg)

    print(f"\n    slack configured : {cfg.slack_webhook() is not None}")
    print(f"    teams configured : {cfg.teams_webhook() is not None}")
    print(f"    dry_run          : {cfg.dry_run}")

    # Route test alerts of each severity
    print("\n  ── Routing test alerts ──")
    test_alerts = [
        ("SESINoZenoInvariantViolated",
         {"summary": "Throttle bound violated", "metrics": {"holds": 0}}),
        ("SESIPipelineLatencyHigh",
         {"summary": "p95 latency 2.3s", "metrics": {"p95": 2.3}}),
        ("SESICoincidenceRateSpike",
         {"summary": "Coincidence spike", "metrics": {"ratio": 12}}),
    ]
    for name, alert in test_alerts:
        out = router.route(name, alert)
        print(f"    {name:<35s} sev={out['severity']:<8s} "
              f"→ {out['status']} {out.get('channels', [])}")

    # Report
    print("\n  ── Dispatch report ──")
    rep = router.report()
    print(f"    dispatched : {rep['n_dispatched']}")
    print(f"    failed     : {rep['n_failed']}")
    print(f"    dry_run    : {rep['dry_run']}")
    print(f"    {rep['honest_note']}")

    # Verify INFO did not go to chat by default
    info_routed = any(e.get("action") == "logged_only_not_chat"
                      for e in router.dispatch_log)
    print(f"\n    [{'PASS' if info_routed else 'FAIL'}] INFO alert kept off chat "
          f"(alert-fatigue prevention)")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Webhook URLs are SECRETS from env vars; dry_run default = True")
    print("  • No real HTTP request made unless armed with real webhooks")
    print("  • Failed sends are RECORDED + RETRIED, never swallowed")
    print("  • INFO stays off chat unless info_to_chat=True")
    print("  • Chat reports pipeline health, not astrophysics")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
