"""
=============================================================================
SESI ALERT RUNBOOKS — operational response procedures for each pipeline alert
Companion to sesi_grafana_alert_rules.py
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

PURPOSE
-------
Each alert defined in sesi_grafana_alert_rules.py references a runbook.
This module generates those runbooks as Markdown, one per alert, encoding
the project's non-negotiable monitoring discipline:

  [R1] DEAD-MAN'S SWITCH  — a silent pipeline is a failure, not a quiet sky.
  [R2] FAILURES NEVER SUPPRESSED — diagnosis steps must surface errors,
       never mask them (CFD paper §3.3).
  [R3] THROTTLE ≠ IGNORE  — reducing alert noise must never mean ignoring
       the underlying condition.
  [R4] CANDIDATE ≠ DISCOVERY — coincidence spikes go to a human, not to a
       press release.

RUNBOOK STRUCTURE (per alert)
-----------------------------
  Summary / Impact / Severity / Diagnosis (concrete PromQL + checks) /
  Remediation / Escalation / Related / Anti-suppression clause.

LABELS: [D] derived operational content  [O] live-environment specifics

HONEST SCOPE
------------
• These are RESEARCH-PIPELINE runbooks. Commands reference the metrics
  from sesi_kafka_prometheus_grafana.py; adapt paths/credentials for a
  real deployment.
• Runbooks do not diagnose astrophysics; they keep the PIPELINE healthy so
  that science candidates are neither lost nor fabricated.
=============================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("sesi.runbooks")

__all__ = ["Runbook", "RunbookLibrary", "RunbookGenerator"]


# ═══════════════════════════════════════════════════════════════════════
# RUNBOOK STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Runbook:
    """A single operational runbook."""
    alert_name: str
    severity: str
    summary: str
    impact: str
    diagnosis: List[str] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)
    escalation: str = ""
    related_dashboards: List[str] = field(default_factory=list)
    anti_suppression_clause: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# Runbook: `{self.alert_name}`",
            "",
            f"**Severity:** `{self.severity}`",
            "",
            "## Summary",
            self.summary,
            "",
            "## Impact",
            self.impact,
            "",
            "## Diagnosis",
        ]
        lines += [f"{i+1}. {step}" for i, step in enumerate(self.diagnosis)]
        lines += ["", "## Remediation"]
        lines += [f"{i+1}. {step}" for i, step in enumerate(self.remediation)]
        lines += [
            "",
            "## Escalation",
            self.escalation,
            "",
            "## Related dashboards / metrics",
        ]
        lines += [f"- `{d}`" for d in self.related_dashboards]
        lines += [
            "",
            "## ⚠️ Anti-suppression clause (CFD §3.3 discipline)",
            self.anti_suppression_clause,
            "",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# [D] RUNBOOK LIBRARY — one per alert rule
# ═══════════════════════════════════════════════════════════════════════

class RunbookLibrary:
    """
    [D] Canonical runbooks matching the alert names in
    sesi_grafana_alert_rules.py (PipelineAlertLibrary).
    """

    @staticmethod
    def consumer_down() -> Runbook:
        return Runbook(
            alert_name="SESIPipelineConsumerDown",
            severity="critical",
            summary=("The multi-messenger pipeline has not ingested any GCN "
                     "alert for 10 minutes. A silent pipeline is a FAILURE, "
                     "not a quiet sky."),
            impact=("No gravitational-wave, neutrino, or EM alerts are being "
                    "received. Any real astrophysical event occurring now will "
                    "be missed entirely. Follow-up capability is zero."),
            diagnosis=[
                "Check the consumer is running: `rate(sesi_gcn_alerts_total[5m])` "
                "should be > 0 during active observing.",
                "Verify Kafka connectivity and GCN credentials: `kafka-consumer-groups "
                "--describe --group sesi-mm-dashboard`.",
                "Confirm the network path to `kafka.gcn.nasa.gov:9092` is open.",
                "Check consumer logs for SCRAM-SHA-512 authentication errors.",
                "Look at `sesi_gcn_parse_failures_total` — a sudden stop with "
                "rising failures suggests malformed messages, not silence.",
            ],
            remediation=[
                "Restart the consumer service: `systemctl restart sesi-gcn-consumer`.",
                "If credentials expired, renew at https://gcn.nasa.gov and reload.",
                "If Kafka is unreachable, fail over to the synthetic source ONLY "
                "for smoke-testing; never present synthetic alerts as live.",
                "After recovery, confirm alerts flow and the dead-man's switch clears.",
            ],
            escalation=("Page the on-call multi-messenger engineer immediately. "
                        "If down > 30 min, escalate to the pipeline lead. A dead "
                        "consumer during an observing run is a highest-priority outage."),
            related_dashboards=[
                "sesi_gcn_alerts_total",
                "sesi_gcn_parse_failures_total",
                "GCN status page (gcn.nasa.gov)",
            ],
            anti_suppression_clause=(
                "DO NOT silence this alert or extend the `for` window to make it "
                "go away. If the sky is genuinely quiet, the consumer should still "
                "be connected — verify connection state, then acknowledge only "
                "after confirming the pipeline is receiving a live stream."),
        )

    @staticmethod
    def parse_failures_high() -> Runbook:
        return Runbook(
            alert_name="SESIPipelineParseFailuresHigh",
            severity="warning",
            summary=("More than 0.1 GCN message parse failures per second. "
                     "Malformed or schema-changed messages are being dropped "
                     "from the candidate pool."),
            impact=("Affected alerts are not turned into candidates. If a real "
                    "event arrives in an unrecognized format, it will be missed. "
                    "A schema change upstream can silently blind the pipeline."),
            diagnosis=[
                "Inspect `rate(sesi_gcn_parse_failures_total[5m])` for the trend.",
                "Pull recent raw payloads and identify the failing schema "
                "(JSON vs VOEvent XML vs new field names).",
                "Check whether GCN announced a schema/topic change.",
                "Verify the `AlertMessageParser` handles the current format.",
            ],
            remediation=[
                "Log representative failing payloads for inspection.",
                "Update the parser to handle the new schema; add a regression test.",
                "If failures are from a single bad producer, coordinate upstream.",
                "Replay the dropped window if the messages are recoverable.",
            ],
            escalation=("Open a ticket for the parser owner. If failures exceed "
                        "50% of traffic for > 15 min, escalate to warning→critical "
                        "and treat like an outage."),
            related_dashboards=[
                "sesi_gcn_parse_failures_total",
                "sesi_gcn_alerts_total",
            ],
            anti_suppression_clause=(
                "Parse failures are RECORDED, never swallowed. Do not disable the "
                "failure counter to 'clean up' the dashboard. Every dropped message "
                "is a potential missed event and must be accounted for."),
        )

    @staticmethod
    def nozeno_invariant_violated() -> Runbook:
        return Runbook(
            alert_name="SESINoZenoInvariantViolated",
            severity="critical",
            summary=("The No-Zeno event-count bound N(T) ≤ 1 + ⌊T/τ_e⌋ was "
                     "violated. The throttle emitted more events than the "
                     "minimum-spacing rule permits."),
            impact=("This is a LOGIC BUG in the event scheduler, not a tuning "
                    "issue. The bound that guarantees finite controller activity "
                    "is broken, so downstream event accounting is unreliable."),
            diagnosis=[
                "Read `sesi_throttle_invariant_holds` (should be 1; it is 0).",
                "Dump the event log and the throttle state (last_alert_t, tau_e).",
                "Check for concurrent writers that bypass the throttle.",
                "Check for clock skew or negative time deltas corrupting spacing.",
                "Reproduce with the recorded event sequence in a unit test.",
            ],
            remediation=[
                "STOP trusting event counts from the affected window.",
                "Patch the scheduler so spacing is enforced atomically.",
                "Add a regression test that asserts the bound for the reproducer.",
                "Redeploy and confirm `sesi_throttle_invariant_holds == 1`.",
            ],
            escalation=("Page immediately. This invariant is a proved property "
                        "(CFD paper Prop 1 / Six Problems Prop 6.1); its violation "
                        "means the implementation contradicts the specification and "
                        "must be fixed before any science output is trusted."),
            related_dashboards=[
                "sesi_throttle_invariant_holds",
                "sesi_alerts_emitted_total",
                "sesi_alerts_deferred_total",
            ],
            anti_suppression_clause=(
                "NEVER work around this by raising tau_e, widening the bound, or "
                "suppressing the gauge. The fix is to make the code honor the "
                "proved invariant, not to relax the invariant to match the code."),
        )

    @staticmethod
    def latency_high() -> Runbook:
        return Runbook(
            alert_name="SESIPipelineLatencyHigh",
            severity="warning",
            summary=("End-to-end pipeline p95 latency exceeds 1 second. "
                     "Low-latency follow-up depends on sub-second processing."),
            impact=("Rapid multi-messenger follow-up (e.g. for a BNS merger) is "
                    "degraded. The later alerts reach telescopes, the more "
                    "faint/transient counterpart signal has faded."),
            diagnosis=[
                "Check `histogram_quantile(0.95, rate(sesi_pipeline_latency_seconds_bucket[1m]))`.",
                "Identify the slow stage: ingest, parse, NPE ranking, or throttle.",
                "Check CPU/GPU saturation and Kafka consumer lag.",
                "Look for a burst that temporarily overloaded the pipeline.",
            ],
            remediation=[
                "If burst-driven, allow it to drain and confirm latency recovers.",
                "If sustained, scale out consumers or increase batch efficiency.",
                "Profile the NPE ranking stage; move heavy inference off the hot path.",
                "Tune `n_posterior_samples` down only if science impact is assessed.",
            ],
            escalation=("Ticket the pipeline owner. If p95 > 5 s for > 10 min, "
                        "escalate — low-latency is a core requirement, not a nicety."),
            related_dashboards=[
                "sesi_pipeline_latency_seconds",
                "Kafka consumer lag",
            ],
            anti_suppression_clause=(
                "Do not hide latency by loosening the threshold or dropping the "
                "histogram. Latency is a real degradation of follow-up capability "
                "and must be resolved, not relabeled."),
        )

    @staticmethod
    def deferred_accumulation() -> Runbook:
        return Runbook(
            alert_name="SESIDeferredAlertsAccumulating",
            severity="warning",
            summary=("Deferred alerts outnumber emitted alerts. The No-Zeno "
                     "throttle is holding back more candidates than it releases."),
            impact=("Candidates are still RECORDED (never dropped), but real-time "
                    "dissemination is throttled. If this persists, time-critical "
                    "follow-up is delayed even though nothing is lost."),
            diagnosis=[
                "Compare `rate(sesi_alerts_deferred_total[5m])` vs "
                "`rate(sesi_alerts_emitted_total[5m])`.",
                "Check whether tau_e is set too aggressively for the current rate.",
                "Check whether a real burst (many coincidences) is occurring.",
                "Confirm deferred candidates are still being logged for review.",
            ],
            remediation=[
                "If a genuine burst: let it pass; review deferred candidates after.",
                "If tau_e is too tight: reconsider the spacing based on the science "
                "requirement (do this deliberately, not to silence the alert).",
                "Ensure a batch review of deferred candidates runs periodically.",
            ],
            escalation=("Ticket the pipeline owner. If deferred candidates include "
                        "high-ranking coincidences, escalate for immediate human review."),
            related_dashboards=[
                "sesi_alerts_deferred_total",
                "sesi_alerts_emitted_total",
                "sesi_candidates_total",
            ],
            anti_suppression_clause=(
                "Deferral is NOT deletion. Every deferred candidate remains in the "
                "record and must be reviewed. Do not clear the deferred log to make "
                "this alert stop; address why candidates are piling up."),
        )

    @staticmethod
    def coincidence_spike() -> Runbook:
        return Runbook(
            alert_name="SESICoincidenceRateSpike",
            severity="info",
            summary=("Multi-messenger coincidence rate is > 10× its 1-hour "
                     "average. Notable, but NOT an auto-detection."),
            impact=("Possibly interesting astrophysics, or possibly an instrumental/"
                    "environmental correlation. No scientific claim can be made "
                    "from the spike alone."),
            diagnosis=[
                "List the coincident candidates in the spike window.",
                "Check each messenger independently — is one stream misbehaving?",
                "Look for known instrumental/environmental correlations (e.g. a "
                "calibration injection, a terrestrial event).",
                "Verify the No-Zeno invariant and throttle are healthy.",
            ],
            remediation=[
                "Assemble the candidate list with SNR, sky position, and Δt.",
                "Hand to the science review team for human assessment.",
                "If a real candidate, initiate standard follow-up protocols.",
                "If instrumental, document the cause to refine future filtering.",
            ],
            escalation=("Notify the science on-call for review. This is an "
                        "INFORMATIONAL alert — escalation to a detection claim "
                        "requires independent human verification and must follow "
                        "the collaboration's review process."),
            related_dashboards=[
                "sesi_coincidences_total",
                "Sky map panel",
                "Coincidence table",
            ],
            anti_suppression_clause=(
                "A candidate is NOT an association. Do not announce, publish, or "
                "trigger autonomous follow-up solely on this spike. Equally, do not "
                "dismiss it — record it and route it to human review. The Moral "
                "Order discipline applies: a convincing signal is not yet a "
                "confirmed phenomenon."),
        )

    @classmethod
    def all_runbooks(cls) -> List[Runbook]:
        return [
            cls.consumer_down(),
            cls.parse_failures_high(),
            cls.nozeno_invariant_violated(),
            cls.latency_high(),
            cls.deferred_accumulation(),
            cls.coincidence_spike(),
        ]


# ═══════════════════════════════════════════════════════════════════════
# [D] GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class RunbookGenerator:
    """[D] Write all runbooks to disk and validate the discipline clauses."""

    # Alert name → filename mapping (matches PipelineAlertRule.runbook)
    FILENAMES = {
        "SESIPipelineConsumerDown": "consumer_down.md",
        "SESIPipelineParseFailuresHigh": "parse_failures.md",
        "SESINoZenoInvariantViolated": "nozeno_violation.md",
        "SESIPipelineLatencyHigh": "latency.md",
        "SESIDeferredAlertsAccumulating": "deferred_accumulation.md",
        "SESICoincidenceRateSpike": "coincidence_spike.md",
    }

    REQUIRED_SECTIONS = ["Summary", "Impact", "Diagnosis", "Remediation",
                         "Escalation", "Anti-suppression"]

    def write_all(self, outdir: str = "./runbooks") -> Dict[str, str]:
        os.makedirs(outdir, exist_ok=True)
        paths = {}
        for rb in RunbookLibrary.all_runbooks():
            fname = self.FILENAMES.get(rb.alert_name, f"{rb.alert_name}.md")
            path = os.path.join(outdir, fname)
            with open(path, "w") as f:
                f.write(rb.to_markdown())
            paths[rb.alert_name] = path
        logger.info(f"Wrote {len(paths)} runbooks to {outdir}")
        return paths

    def validate_discipline(self) -> Dict[str, Any]:
        """
        Verify every runbook carries the required sections and the
        anti-suppression clause — the CFD §3.3 discipline made explicit.
        """
        issues = []
        for rb in RunbookLibrary.all_runbooks():
            md = rb.to_markdown()
            for section in self.REQUIRED_SECTIONS:
                if section.lower() not in md.lower():
                    issues.append(f"{rb.alert_name}: missing '{section}'")
            if not rb.anti_suppression_clause.strip():
                issues.append(f"{rb.alert_name}: empty anti-suppression clause")
        return {
            "n_runbooks": len(RunbookLibrary.all_runbooks()),
            "issues": issues,
            "all_valid": len(issues) == 0,
            "note": ("Every runbook must contain an anti-suppression clause; "
                     "this is the operational expression of CFD §3.3."),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Alert Runbooks — Smoke Test")
    print("=" * 72)

    # Discipline validation
    gen = RunbookGenerator()
    check = gen.validate_discipline()
    print(f"\n  ── Discipline validation ──")
    print(f"    runbooks     : {check['n_runbooks']}")
    print(f"    all valid    : {check['all_valid']}")
    if check["issues"]:
        for issue in check["issues"]:
            print(f"    [FAIL] {issue}")
    else:
        print(f"    [PASS] all runbooks have required sections + anti-suppression")

    # Write runbooks
    paths = gen.write_all(outdir="./runbooks_smoke")
    print(f"\n  ── Written runbooks ──")
    for name, path in paths.items():
        exists = os.path.exists(path)
        print(f"    [{'PASS' if exists else 'FAIL'}] {name} → {path}")

    # Spot-check the No-Zeno runbook content
    nz = RunbookLibrary.nozeno_invariant_violated()
    print(f"\n  ── Spot-check: NoZenoInvariantViolated ──")
    print(f"    severity : {nz.severity}")
    print(f"    escalation mentions 'proved property': "
          f"{'proved' in nz.escalation.lower()}")
    print(f"    anti-suppression forbids relaxing bound: "
          f"{'never' in nz.anti_suppression_clause.lower()}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Runbooks are operational guides, not astrophysical diagnosis")
    print("  • Commands assume the metrics from sesi_kafka_prometheus_grafana")
    print("  • The anti-suppression clause enforces CFD §3.3: failures surface,")
    print("    throttle ≠ ignore, candidate ≠ discovery")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
