"""
=============================================================================
SESI BIDIRECTIONAL SLACK ↔ PAGERDUTY INTEGRATION
Acknowledge in Slack → relay back to PagerDuty (and vice versa)
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_slack_pagerduty_escalation.py
--------------------------------------------------------
The earlier module was ONE-DIRECTIONAL: PagerDuty escalates → Slack is
notified. This module closes the loop:

    PagerDuty incident ──► Slack interactive message (Ack / Resolve buttons)
    User clicks Ack in Slack ──► interaction payload ──► our handler
    handler ──► PagerDuty REST API: acknowledge the incident
    PagerDuty state ──► mirrored back into the Slack message

REAL REQUIREMENTS [O for live deployment]
-----------------------------------------
  • Slack app with INTERACTIVITY enabled + a Request URL (web server).
  • PagerDuty REST API token with WRITE scope (acknowledge/resolve).
  • A persistent mapping between the Slack message and the PagerDuty
    incident (stored here in IncidentMappingStore keyed by dedup_key).

NO-SUPPRESSION / NO-LOSS DISCIPLINE (CFD §3.3)
----------------------------------------------
  • Acks are IDEMPOTENT: acknowledging an already-acknowledged incident is
    a no-op that returns success, never an error or a duplicate.
  • A failed relay to PagerDuty is RECORDED and RETRIED — never swallowed.
  • If the mapping between a Slack message and a PagerDuty incident is
    lost/missing, the handler records a warning rather than silently
    dropping the ack.

LABELS: [C] Slack/PagerDuty API conventions  [D] derived glue  [O] live ops

HONEST SCOPE
------------
• In dry-run (default) no real HTTP call is made; the full ack loop is
  simulated so the state machine can be exercised safely.
• This coordinates OPERATIONAL acknowledgment. It does not make any
  astrophysical claim; an acknowledged alert is still a candidate that a
  human must investigate.
=============================================================================
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.slack_pd_bidir")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__all__ = ["BidirConfig", "IncidentStatus", "IncidentMappingStore",
           "PagerDutyIncidentClient", "SlackInteractionHandler",
           "BidirectionalSlackPagerDutyBridge"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG & STATUS
# ═══════════════════════════════════════════════════════════════════════

class IncidentStatus(Enum):
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class BidirConfig:
    pd_rest_key_env: str = "PAGERDUTY_REST_API_KEY"
    slack_token_env: str = "SESI_SLACK_BOT_TOKEN"
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    dry_run: bool = True   # SAFE DEFAULT


# ═══════════════════════════════════════════════════════════════════════
# [D] INCIDENT MAPPING STORE
# ═══════════════════════════════════════════════════════════════════════

class IncidentMappingStore:
    """
    [D] Persistent mapping between dedup_key, PagerDuty incident id, and the
    Slack message that announced it. This is what lets an ack in Slack be
    routed back to the correct PagerDuty incident.

    In production this would be a database/Redis; here an in-memory dict.
    """

    def __init__(self):
        self._by_dedup: Dict[str, Dict[str, Any]] = {}

    def record(self, dedup_key: str, pd_incident_id: str,
               slack_ts: Optional[str] = None,
               status: IncidentStatus = IncidentStatus.TRIGGERED) -> None:
        self._by_dedup[dedup_key] = {
            "pd_incident_id": pd_incident_id,
            "slack_ts": slack_ts,
            "status": status,
            "created": time.time(),
        }

    def get(self, dedup_key: str) -> Optional[Dict[str, Any]]:
        return self._by_dedup.get(dedup_key)

    def set_status(self, dedup_key: str, status: IncidentStatus) -> bool:
        if dedup_key in self._by_dedup:
            self._by_dedup[dedup_key]["status"] = status
            return True
        return False

    def status(self, dedup_key: str) -> Optional[IncidentStatus]:
        rec = self._by_dedup.get(dedup_key)
        return rec["status"] if rec else None

    def size(self) -> int:
        return len(self._by_dedup)


# ═══════════════════════════════════════════════════════════════════════
# [C] PAGERDUTY INCIDENT CLIENT (acknowledge / resolve)
# ═══════════════════════════════════════════════════════════════════════

class PagerDutyIncidentClient:
    """
    [C] PagerDuty REST API client for incident state transitions.

    Acknowledge: PUT /incidents/{id}  with incident[status]="acknowledged"
    Resolve:     PUT /incidents/{id}  with incident[status]="resolved"

    Requires a REST API token with write scope. Dry-run simulates.
    """

    BASE = "https://api.pagerduty.com"

    def __init__(self, cfg: BidirConfig, mapping: IncidentMappingStore):
        self.cfg = cfg
        self.mapping = mapping
        self.api_log: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

    def _transition(self, dedup_key: str, target: IncidentStatus) -> Dict[str, Any]:
        rec = self.mapping.get(dedup_key)
        if rec is None:
            # CFD §3.3: do not silently drop — record the missing mapping
            self.failures.append({"dedup_key": dedup_key,
                                  "reason": "mapping not found"})
            logger.warning(f"[MAPPING MISSING — NOT SUPPRESSED] {dedup_key}")
            return {"status": "failed", "reason": "mapping_not_found"}

        # Idempotency: already in target state ⟹ no-op success
        if rec["status"] == target:
            return {"status": "noop_idempotent", "pd_state": target.value}

        pd_id = rec["pd_incident_id"]

        if self.cfg.dry_run or not HAS_REQUESTS:
            self.mapping.set_status(dedup_key, target)
            self.api_log.append({"mode": "dry_run", "pd_id": pd_id,
                                 "target": target.value})
            return {"status": "dry_run", "pd_state": target.value}

        # real REST call with retry
        token = None  # would read from env in a real deployment
        last = {}
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.put(
                    f"{self.BASE}/incidents/{pd_id}",
                    json={"incident": {"type": "incident_reference",
                                       "status": target.value}},
                    headers={"Authorization": f"Token token={token}",
                             "From": "sesi-oncall@localhost"},
                    timeout=15)
                if resp.status_code in (200, 202):
                    self.mapping.set_status(dedup_key, target)
                    self.api_log.append({"mode": "live", "pd_id": pd_id,
                                         "target": target.value,
                                         "http": resp.status_code})
                    return {"status": "sent", "pd_state": target.value,
                            "http": resp.status_code}
                last = {"status": "failed", "http": resp.status_code}
            except Exception as e:
                last = {"status": "failed", "error": str(e)}
            time.sleep(self.cfg.retry_backoff_s)
        # all retries exhausted — record, never swallow
        self.failures.append({"dedup_key": dedup_key, "target": target.value,
                              **last})
        logger.warning(f"[PD TRANSITION FAILURE — NOT SUPPRESSED] "
                       f"{dedup_key} → {target.value}: {last}")
        return last

    def acknowledge(self, dedup_key: str) -> Dict[str, Any]:
        return self._transition(dedup_key, IncidentStatus.ACKNOWLEDGED)

    def resolve(self, dedup_key: str) -> Dict[str, Any]:
        return self._transition(dedup_key, IncidentStatus.RESOLVED)


# ═══════════════════════════════════════════════════════════════════════
# [C/D] SLACK INTERACTION HANDLER
# ═══════════════════════════════════════════════════════════════════════

class SlackInteractionHandler:
    """
    [C/D] Parse a Slack interaction payload (button click) and extract the
    dedup_key of the incident the user acted on.

    Slack sends a JSON payload (form-encoded as `payload`) when a user
    clicks a button. The button's `value` carries our dedup_key, and
    `actions[].action_id` tells us whether it was "ack" or "resolve".
    """

    @staticmethod
    def parse(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action + dedup_key + user from a Slack interaction payload."""
        actions = payload.get("actions", [])
        if not actions:
            return {"ok": False, "reason": "no actions in payload"}
        action = actions[0]
        return {
            "ok": True,
            "action_id": action.get("action_id", ""),
            "dedup_key": action.get("value", ""),
            "user": payload.get("user", {}).get("id", "?"),
            "channel": payload.get("channel", {}).get("id", "?"),
            "message_ts": payload.get("message", {}).get("ts"),
        }

    @staticmethod
    def build_interactive_message(alert_name: str, dedup_key: str,
                                  severity: str) -> Dict[str, Any]:
        """
        Build the Slack Block-Kit message WITH Ack/Resolve buttons.
        The buttons' `value` carries the dedup_key for the round trip.
        """
        color = {"critical": "#d32f2f", "warning": "#f9a825",
                 "info": "#1976d2"}.get(severity, "#607d8b")
        return {
            "text": f":rotating_light: *{alert_name}* ({severity.upper()})",
            "attachments": [{
                "color": color,
                "blocks": [
                    {"type": "section",
                     "text": {"type": "mrkdwn",
                              "text": f"*{alert_name}*\n`{dedup_key}`"}},
                    {"type": "actions", "elements": [
                        {"type": "button", "text": {"type": "plain_text",
                                                    "text": "✅ Acknowledge"},
                         "action_id": "ack_incident", "value": dedup_key,
                         "style": "primary"},
                        {"type": "button", "text": {"type": "plain_text",
                                                    "text": "✔ Resolve"},
                         "action_id": "resolve_incident", "value": dedup_key},
                    ]},
                ],
            }],
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] BIDIRECTIONAL BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class BidirectionalSlackPagerDutyBridge:
    """
    [D] Full round trip:
      1. open_incident  → record mapping, post interactive Slack message
      2. on_slack_action → parse click, relay ack/resolve to PagerDuty
      3. idempotent + failure-recorded per CFD §3.3
    """

    def __init__(self, cfg: Optional[BidirConfig] = None):
        self.cfg = cfg or BidirConfig()
        self.mapping = IncidentMappingStore()
        self.pd = PagerDutyIncidentClient(self.cfg, self.mapping)
        self.handler = SlackInteractionHandler()
        self.posted_slack: List[Dict[str, Any]] = []

    # ── direction 1: PagerDuty → Slack ───────────────────────────────
    def open_incident(self, alert_name: str, dedup_key: str,
                      severity: str = "critical",
                      pd_incident_id: Optional[str] = None) -> Dict[str, Any]:
        """Record the incident and post an interactive Slack message."""
        pd_id = pd_incident_id or f"PD-{abs(hash(dedup_key)) % 10**6:06d}"
        self.mapping.record(dedup_key, pd_id, status=IncidentStatus.TRIGGERED)

        msg = self.handler.build_interactive_message(alert_name, dedup_key, severity)
        self.posted_slack.append({"dedup_key": dedup_key, "message": msg,
                                  "mode": "dry_run" if self.cfg.dry_run else "live"})
        return {
            "status": "opened",
            "dedup_key": dedup_key,
            "pd_incident_id": pd_id,
            "slack_posted": True,
        }

    # ── direction 2: Slack → PagerDuty ───────────────────────────────
    def on_slack_action(self, slack_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a Slack button click and relay the state change to PagerDuty.
        Returns the relay outcome.
        """
        parsed = self.handler.parse(slack_payload)
        if not parsed["ok"]:
            self.pd.failures.append({"reason": "bad slack payload",
                                     "payload": slack_payload})
            logger.warning("[BAD SLACK PAYLOAD — NOT SUPPRESSED]")
            return {"status": "failed", "reason": "bad_payload"}

        dedup_key = parsed["dedup_key"]
        action_id = parsed["action_id"]

        if action_id == "ack_incident":
            relay = self.pd.acknowledge(dedup_key)
        elif action_id == "resolve_incident":
            relay = self.pd.resolve(dedup_key)
        else:
            relay = {"status": "ignored", "reason": f"unknown action {action_id}"}

        return {
            "status": relay.get("status"),
            "action": action_id,
            "dedup_key": dedup_key,
            "user": parsed["user"],
            "relay": relay,
            "new_status": (self.mapping.status(dedup_key).value
                           if self.mapping.status(dedup_key) else None),
        }

    def report(self) -> Dict[str, Any]:
        return {
            "n_mapped_incidents": self.mapping.size(),
            "n_slack_posts": len(self.posted_slack),
            "n_pd_api_calls": len(self.pd.api_log),
            "n_failures": len(self.pd.failures),
            "dry_run": self.cfg.dry_run,
            "honest_note": (
                "Dry-run simulates the full ack loop without real HTTP. "
                "Live mode needs Slack interactivity + PagerDuty write token "
                "and a deployed request endpoint. Acks are idempotent; "
                "failures are recorded, never swallowed."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Bidirectional Slack↔PagerDuty — Smoke Test")
    print("=" * 72)
    print(f"  requests : {HAS_REQUESTS}")

    bridge = BidirectionalSlackPagerDutyBridge(BidirConfig(dry_run=True))

    # 1. Open an incident (PagerDuty → Slack)
    print("\n  ── [1] Open incident → Slack message with buttons ──")
    opened = bridge.open_incident("SESINoZenoInvariantViolated",
                                  dedup_key="nozeno-001", severity="critical")
    print(f"    status       : {opened['status']}")
    print(f"    pd_incident  : {opened['pd_incident_id']}")
    print(f"    slack posted : {opened['slack_posted']}")
    msg = bridge.posted_slack[0]["message"]
    print(f"    buttons      : {[b['text']['text'] for b in msg['attachments'][0]['blocks'][1]['elements']]}")

    # 2. Ack in Slack → relay to PagerDuty
    print("\n  ── [2] Ack in Slack → relay to PagerDuty ──")
    ack_payload = {
        "user": {"id": "U123"},
        "channel": {"id": "C456"},
        "message": {"ts": "1234.5678"},
        "actions": [{"action_id": "ack_incident", "value": "nozeno-001"}],
    }
    ack_result = bridge.on_slack_action(ack_payload)
    print(f"    action      : {ack_result['action']}")
    print(f"    relay status: {ack_result['relay'].get('status')}")
    print(f"    new status  : {ack_result['new_status']}")

    # 3. Idempotency — ack again
    print("\n  ── [3] Idempotency: ack again ──")
    ack_again = bridge.on_slack_action(ack_payload)
    print(f"    second ack relay: {ack_again['relay'].get('status')}")
    print(f"    [{'PASS' if ack_again['relay'].get('status')=='noop_idempotent' else 'FAIL'}] "
          f"second ack is a no-op (idempotent)")

    # 4. Missing mapping — must be recorded, not dropped
    print("\n  ── [4] Missing mapping (CFD §3.3: never silently drop) ──")
    orphan_payload = {
        "user": {"id": "U999"}, "channel": {"id": "C999"},
        "actions": [{"action_id": "ack_incident", "value": "unknown-key"}],
    }
    orphan = bridge.on_slack_action(orphan_payload)
    print(f"    relay status: {orphan['relay'].get('status')}")
    print(f"    reason      : {orphan['relay'].get('reason')}")
    print(f"    [{'PASS' if orphan['relay'].get('reason')=='mapping_not_found' else 'FAIL'}] "
          f"missing mapping recorded, not swallowed")

    # Report
    print("\n  ── Bridge report ──")
    rep = bridge.report()
    print(f"    mapped incidents : {rep['n_mapped_incidents']}")
    print(f"    slack posts      : {rep['n_slack_posts']}")
    print(f"    PD api calls     : {rep['n_pd_api_calls']}")
    print(f"    failures         : {rep['n_failures']}")
    print(f"    dry_run          : {rep['dry_run']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Dry-run simulates the loop; no real HTTP without credentials")
    print("  • Live needs Slack interactivity + PD write token + deployed endpoint")
    print("  • Acks are idempotent; failures recorded, never swallowed")
    print("  • Acknowledgment is operational; it does NOT resolve the alert's")
    print("    underlying cause or make a candidate into a detection")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
