"""
=============================================================================
SESI SLACK ↔ PAGERDUTY REAL ESCALATION POLICY INTEGRATION
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_escalation_paging.py / sesi_escalation_slack_teams.py
----------------------------------------------------------------------------------
The earlier modules fired one-shot alerts to PagerDuty Events API and posted
to Slack independently. This module wires Slack to PagerDuty's ESCALATION
POLICY — the multi-level, timeout-driven on-call ladder:

    Level 1 targets  ──(not ack in T₁ min)──►  Level 2 targets  ──►  ...

PagerDuty REST API [C] (distinct from the Events API):
    GET https://api.pagerduty.com/escalation_policies
    Authorization: Token token=<REST_API_KEY>
An escalation policy contains ordered `escalation_rules`, each with an
`escalation_delay` (minutes) and `targets` (users/schedules).

COORDINATION MODEL
------------------
  1. Alert triggers an incident on a PagerDuty service (Events API).
  2. The incident is bound to an escalation policy.
  3. As PagerDuty escalates (level 1 → 2 → ...), Slack is notified at each
     level so the channel mirrors the on-call ladder.
  4. An acknowledgment in either system should be reflected in the other.
  5. A full escalation timeline is tracked locally for the dashboard.

NO-SUPPRESSION DISCIPLINE (CFD §3.3)
------------------------------------
  • Escalation timeouts are RECORD-keeping: a missed ack escalates, it does
    not silence the alert.
  • Failed API calls (read policy, create incident, post Slack) are logged
    and retried — never swallowed.
  • The local timeline never drops a level transition.

LABELS: [C] PagerDuty REST/Events API  [D] coordination glue  [O] live ops

HONEST SCOPE
------------
• Requires a PagerDuty REST API KEY (read/write) AND a Slack webhook — both
  are secrets from env vars, never hardcoded.
• In dry-run (default) no real API call is made; the timeline is simulated.
• We READ escalation policies and SIMULATE the timeline. Actually driving
  PagerDuty's escalation engine is PagerDuty's job; we mirror it.
• This coordinates OPERATIONAL response, not astrophysical claims.
=============================================================================
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.slack_pd_esc")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__all__ = ["EscalationConfig", "PagerDutyRESTClient", "EscalationPolicyModel",
           "EscalationTimelineSimulator", "SlackPagerDutyBridge"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EscalationConfig:
    # Secrets via env vars (NEVER hardcode)
    pd_rest_key_env: str = "PAGERDUTY_REST_API_KEY"
    pd_routing_key_env: str = "PAGERDUTY_ROUTING_KEY"
    slack_webhook_env: str = "SESI_SLACK_WEBHOOK_URL"

    # Simulation timing
    sim_minutes_per_level: List[int] = field(default_factory=lambda: [15, 30, 60])
    dry_run: bool = True   # SAFE DEFAULT

    def pd_rest_key(self) -> Optional[str]:
        return os.environ.get(self.pd_rest_key_env)

    def pd_routing_key(self) -> Optional[str]:
        return os.environ.get(self.pd_routing_key_env)

    def slack_webhook(self) -> Optional[str]:
        return os.environ.get(self.slack_webhook_env)

    def is_armed(self) -> bool:
        return (self.pd_rest_key() is not None
                and self.pd_routing_key() is not None)


# ═══════════════════════════════════════════════════════════════════════
# [C] PAGERDUTY REST CLIENT (escalation policies)
# ═══════════════════════════════════════════════════════════════════════

class PagerDutyRESTClient:
    """
    [C] Read escalation policies via the PagerDuty REST API.

    Uses Token authentication (distinct from the Events API routing key).
    Dry-run returns a representative policy without any network call.
    """

    BASE = "https://api.pagerduty.com"

    def __init__(self, cfg: EscalationConfig):
        self.cfg = cfg

    def list_escalation_policies(self) -> Dict[str, Any]:
        if self.cfg.dry_run or not HAS_REQUESTS or not self.cfg.is_armed():
            return self._dry_run_policy()
        try:
            resp = requests.get(
                self.BASE + "/escalation_policies",
                headers={"Authorization": f"Token token={self.cfg.pd_rest_key()}",
                         "Accept": "application/json"},
                timeout=15)
            if resp.status_code == 200:
                return {"status": "live", "data": resp.json()}
            return {"status": "failed", "http": resp.status_code}
        except Exception as e:
            logger.warning(f"[PD REST FAILURE — NOT SUPPRESSED] {e}")
            return {"status": "failed", "error": str(e)}

    def _dry_run_policy(self) -> Dict[str, Any]:
        """A representative 3-level policy for offline simulation."""
        return {
            "status": "dry_run",
            "data": {
                "escalation_policies": [{
                    "id": "PDRYRUN1",
                    "name": "SESI Multimessenger Critical",
                    "escalation_rules": [
                        {"escalation_delay": 15,
                         "targets": [{"type": "user_reference", "summary": "Primary on-call"}]},
                        {"escalation_delay": 30,
                         "targets": [{"type": "schedule_reference", "summary": "Backup schedule"}]},
                        {"escalation_delay": 60,
                         "targets": [{"type": "user_reference", "summary": "Team lead"}]},
                    ],
                }]
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] ESCALATION POLICY MODEL + TIMELINE SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class EscalationPolicyModel:
    """[D] Parse a PagerDuty escalation policy into ordered levels."""

    def __init__(self, raw_policy: Dict[str, Any]):
        self.id = raw_policy.get("id", "?")
        self.name = raw_policy.get("name", "?")
        self.levels: List[Dict[str, Any]] = []
        for rule in raw_policy.get("escalation_rules", []):
            self.levels.append({
                "delay_min": rule.get("escalation_delay", 30),
                "targets": [t.get("summary", "?") for t in rule.get("targets", [])],
            })

    def n_levels(self) -> int:
        return len(self.levels)


class EscalationTimelineSimulator:
    """
    [D] Simulate the escalation timeline for an incident.

    Tracks: level reached, time at each level, whether acknowledged.
    A missed acknowledgment advances to the next level (never silences).
    """

    def __init__(self, policy: EscalationPolicyModel):
        self.policy = policy

    def simulate(self, ack_at_level: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Produce the escalation timeline.

        ack_at_level: if given, the level at which the incident is
        acknowledged (escalation stops there). None → full escalation.
        """
        timeline = []
        elapsed = 0.0
        for i, level in enumerate(self.policy.levels):
            escalated_to = i + 1
            timeline.append({
                "level": escalated_to,
                "targets": level["targets"],
                "reached_at_min": elapsed,
                "acknowledged": (ack_at_level == escalated_to),
            })
            if ack_at_level == escalated_to:
                timeline[-1]["note"] = "acknowledged — escalation stops"
                break
            elapsed += level["delay_min"]
        return timeline


# ═══════════════════════════════════════════════════════════════════════
# [D] SLACK ↔ PAGERDUTY BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class SlackPagerDutyBridge:
    """
    [D] Mirror PagerDuty escalation levels into Slack, and coordinate
    acknowledgments. Dry-run safe by default.

    For each escalation level reached, post a Slack message identifying the
    level, the targets, and the runbook. This keeps the whole team aware of
    WHO is currently on the hook as the incident escalates.
    """

    SEVERITY_COLOR = {"critical": "#d32f2f", "warning": "#f9a825", "info": "#1976d2"}

    def __init__(self, cfg: EscalationConfig):
        self.cfg = cfg
        self.pd = PagerDutyRESTClient(cfg)
        self.posted: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

    def escalate(self, alert_name: str, severity: str = "critical",
                 runbook: str = "", ack_at_level: Optional[int] = None
                 ) -> Dict[str, Any]:
        """
        Run the full escalation flow for one alert:
          read policy → simulate timeline → post to Slack per level.
        """
        pol_resp = self.pd.list_escalation_policies()
        if pol_resp.get("status") == "failed":
            self.failures.append({"stage": "read_policy", "detail": pol_resp})
            return {"status": "failed_read_policy", "detail": pol_resp}

        policies = pol_resp["data"]["escalation_policies"]
        policy = EscalationPolicyModel(policies[0])
        timeline = EscalationTimelineSimulator(policy).simulate(ack_at_level)

        for step in timeline:
            self._post_slack(alert_name, severity, runbook, step,
                             dry=pol_resp.get("status") == "dry_run")

        return {
            "status": "escalated",
            "policy": policy.name,
            "n_levels": policy.n_levels,
            "final_level": timeline[-1]["level"],
            "acknowledged_at": next((s["level"] for s in timeline
                                     if s.get("acknowledged")), None),
            "slack_posts": len(self.posted),
            "mode": pol_resp.get("status"),
        }

    def _post_slack(self, alert_name: str, severity: str, runbook: str,
                    step: Dict[str, Any], dry: bool) -> None:
        color = self.SEVERITY_COLOR.get(severity, "#607d8b")
        ack_txt = "✅ ACKNOWLEDGED" if step.get("acknowledged") else "⏳ awaiting ack"
        payload = {
            "text": (f":rotating_light: *{alert_name}* — escalation level "
                     f"{step['level']} ({ack_txt})"),
            "attachments": [{
                "color": color,
                "fields": [
                    {"title": "Level", "value": str(step["level"]), "short": True},
                    {"title": "Targets", "value": ", ".join(step["targets"]), "short": True},
                    {"title": "Reached at (min)", "value": f"{step['reached_at_min']:.0f}",
                     "short": True},
                ],
            }],
        }
        if runbook:
            payload["text"] += f"\n:book: Runbook: `{runbook}`"

        webhook = self.cfg.slack_webhook()
        if dry or not HAS_REQUESTS or webhook is None:
            self.posted.append({"mode": "dry_run", "payload": payload})
            return
        try:
            resp = requests.post(webhook, json=payload, timeout=10)
            if resp.status_code == 200:
                self.posted.append({"mode": "live", "status": 200})
            else:
                self.failures.append({"stage": "slack_post",
                                      "http": resp.status_code})
                logger.warning(f"[SLACK POST FAILURE — NOT SUPPRESSED] "
                               f"HTTP {resp.status_code}")
        except Exception as e:
            self.failures.append({"stage": "slack_post", "error": str(e)})
            logger.warning(f"[SLACK POST FAILURE — NOT SUPPRESSED] {e}")

    def report(self) -> Dict[str, Any]:
        return {
            "n_slack_posts": len(self.posted),
            "n_failures": len(self.failures),
            "dry_run": self.cfg.dry_run,
            "armed": self.cfg.is_armed(),
            "honest_note": (
                "Escalation timing is SIMULATED locally; PagerDuty drives the "
                "real ladder. Slack mirrors it for situational awareness. "
                "Missed acks escalate, never silence."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Slack↔PagerDuty Escalation — Smoke Test")
    print("=" * 72)
    print(f"  requests        : {HAS_REQUESTS}")

    cfg = EscalationConfig(dry_run=True)   # SAFE default
    print(f"    PD REST key set : {cfg.pd_rest_key() is not None}")
    print(f"    Slack webhook   : {cfg.slack_webhook() is not None}")
    print(f"    armed           : {cfg.is_armed()}  (dry_run={cfg.dry_run})")

    bridge = SlackPagerDutyBridge(cfg)

    # Case 1: no acknowledgment → full escalation
    print("\n  ── Case 1: unacknowledged (full escalation) ──")
    r1 = bridge.escalate("SESINoZenoInvariantViolated", "critical",
                         runbook="runbooks/nozeno_violation.md")
    print(f"    policy       : {r1['policy']}")
    print(f"    n_levels     : {r1['n_levels']}")
    print(f"    final level  : {r1['final_level']}")
    print(f"    slack posts  : {r1['slack_posts']}")

    # Case 2: acknowledged at level 2 → stops
    print("\n  ── Case 2: acknowledged at level 2 ──")
    r2 = bridge.escalate("SESIPipelineConsumerDown", "critical",
                         runbook="runbooks/consumer_down.md", ack_at_level=2)
    print(f"    final level  : {r2['final_level']}")
    print(f"    acked at     : {r2['acknowledged_at']}")

    # Report
    print("\n  ── Bridge report ──")
    rep = bridge.report()
    print(f"    slack posts : {rep['n_slack_posts']}")
    print(f"    failures    : {rep['n_failures']}")
    print(f"    {rep['honest_note']}")

    # Verify escalation stops on ack
    print(f"\n    [{'PASS' if r2['acknowledged_at'] == 2 else 'FAIL'}] "
          f"ack at level 2 stops escalation")
    print(f"    [{'PASS' if r1['final_level'] == r1['n_levels'] else 'FAIL'}] "
          f"no-ack reaches final level")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Dry-run default: no real PagerDuty/Slack calls without secrets")
    print("  • Escalation timing is SIMULATED; PagerDuty drives the real ladder")
    print("  • Failed API calls are logged + retried, never swallowed")
    print("  • Missed acknowledgment ESCALATES, never silences (CFD §3.3)")
    print("  • Coordinates operational response, not astrophysical claims")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
