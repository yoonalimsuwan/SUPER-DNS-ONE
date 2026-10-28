"""
=============================================================================
SESI RESOLUTION NOTE TEMPLATES — recommended structures per alert type
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
My Soul Move By Power of Holy Spirit
Licence : MIT
Year : 2026

WHAT THIS ADDS BEYOND sesi_resolve_with_note.py
-----------------------------------------------
The resolve-with-note bridge requires a non-empty note but leaves the
engineer to write it from scratch. This module provides RECOMMENDED note
TEMPLATES tailored to each alert type from sesi_alert_runbooks.py, so the
note captures the right structure without forcing the engineer to invent
it under pressure.

TEMPLATE STRUCTURE (each template has these sections)
-----------------------------------------------------
  • Root cause        — what actually went wrong (must be filled, not boilerplate)
  • Action taken      — what was done in response
  • Verification      — how the fix was confirmed
  • Follow-up         — remaining work / tickets

DESIGN DISCIPLINE (Moral Order §11 + CFD §3.3)
----------------------------------------------
  • A template is a RECOMMENDATION, not a substitute for analysis. Each
    template carries an explicit "analysis confirmation" field that the
    engineer must affirm, so templates cannot be used to close an alert
    without genuine engagement (the failure mode resolve-with-note exists
    to prevent).
  • Templates never auto-fill root cause; they provide the SCAFFOLD and
    example phrasing only. The actual cause must be determined by a human.
  • This is operational guidance, not astrophysics.

LABELS: [D] derived operational templates  [O] live tuning per deployment
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.note_templates")

__all__ = ["NoteTemplate", "NoteTemplateLibrary", "NoteTemplateRecommender"]


# ═══════════════════════════════════════════════════════════════════════
# TEMPLATE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NoteTemplate:
    """A recommended resolution-note structure for one alert type."""
    alert_name: str
    title: str
    sections: Dict[str, str]          # section name → guiding prompt
    example_fragments: Dict[str, str] # section name → example phrasing
    analysis_confirmation: str        # must be affirmed by the engineer

    def render_scaffold(self) -> str:
        """Render an empty scaffold for the engineer to fill."""
        lines = [f"# Resolution note — {self.title}", ""]
        for section, prompt in self.sections.items():
            lines.append(f"## {section}")
            lines.append(f"[{prompt}]")
            example = self.example_fragments.get(section)
            if example:
                lines.append(f"  (example: {example})")
            lines.append("")
        lines.append("## Analysis confirmation")
        lines.append(f"[ ] {self.analysis_confirmation}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# [D] TEMPLATE LIBRARY (one per alert from sesi_alert_runbooks.py)
# ═══════════════════════════════════════════════════════════════════════

class NoteTemplateLibrary:
    """[D] Recommended note templates keyed by alert name."""

    @staticmethod
    def all_templates() -> Dict[str, NoteTemplate]:
        return {t.alert_name: t for t in [
            NoteTemplate(
                alert_name="SESIPipelineConsumerDown",
                title="Pipeline consumer down",
                sections={
                    "Root cause": "Why did the consumer stop ingesting?",
                    "Action taken": "What was done to restore ingestion?",
                    "Verification": "How was restored flow confirmed?",
                    "Follow-up": "Preventive steps / tickets.",
                },
                example_fragments={
                    "Root cause": "GCN credentials expired at <time>; consumer "
                                  "lost SCRAM auth.",
                    "Action taken": "Renewed credentials and restarted consumer.",
                    "Verification": "rate(sesi_gcn_alerts_total[5m]) > 0 and "
                                    "no parse failures.",
                    "Follow-up": "Add credential-expiry alerting (ticket #…)",
                },
                analysis_confirmation=(
                    "I verified the consumer is actually receiving live data, "
                    "not merely that the alert cleared."),
            ),
            NoteTemplate(
                alert_name="SESINoZenoInvariantViolated",
                title="No-Zeno throttle invariant violated",
                sections={
                    "Root cause": "What logic allowed more events than the bound?",
                    "Action taken": "How was the scheduler corrected?",
                    "Verification": "Reproduction test + invariant re-check.",
                    "Follow-up": "Regression test added.",
                },
                example_fragments={
                    "Root cause": "Concurrent writers bypassed the throttle lock.",
                    "Action taken": "Serialized event emission; added unit test.",
                    "Verification": "sesi_throttle_invariant_holds == 1 over a "
                                    "replay of the offending sequence.",
                    "Follow-up": "Added regression test asserting the bound.",
                },
                analysis_confirmation=(
                    "I confirmed the fix addresses the actual scheduling logic, "
                    "not just a relaxation of the bound."),
            ),
            NoteTemplate(
                alert_name="SESIPipelineLatencyHigh",
                title="Pipeline latency high",
                sections={
                    "Root cause": "Which stage is slow and why?",
                    "Action taken": "What was optimized / scaled?",
                    "Verification": "Latency percentiles after the change.",
                    "Follow-up": "Ongoing monitoring / capacity plan.",
                },
                example_fragments={
                    "Root cause": "NPE ranking stage saturated during a burst.",
                    "Action taken": "Reduced posterior samples; batched inference.",
                    "Verification": "p95(sesi_pipeline_latency_seconds) < 1s.",
                    "Follow-up": "Profile weekly; capacity review ticket #…",
                },
                analysis_confirmation=(
                    "I measured the slow stage directly rather than guessing, "
                    "and the fix does not trade away detection quality silently."),
            ),
            NoteTemplate(
                alert_name="SESIPipelineParseFailuresHigh",
                title="High parse failure rate",
                sections={
                    "Root cause": "What message shape is failing to parse?",
                    "Action taken": "Parser update / upstream coordination.",
                    "Verification": "Parse failure rate after the fix.",
                    "Follow-up": "Schema regression tests.",
                },
                example_fragments={
                    "Root cause": "GCN added a new field the parser rejects.",
                    "Action taken": "Extended parser; added regression case.",
                    "Verification": "rate(sesi_gcn_parse_failures_total) < 0.1/s.",
                    "Follow-up": "Subscribe to GCN schema-change notices.",
                },
                analysis_confirmation=(
                    "I inspected real failing payloads; dropped messages were "
                    "accounted for, not ignored."),
            ),
            NoteTemplate(
                alert_name="SESIDeferredAlertsAccumulating",
                title="Deferred alerts accumulating",
                sections={
                    "Root cause": "Is this a real burst or too-tight throttling?",
                    "Action taken": "Reviewed deferred candidates / tuned τ_e.",
                    "Verification": "Deferred backlog drained; candidates reviewed.",
                    "Follow-up": "Periodic deferred-candidate review process.",
                },
                example_fragments={
                    "Root cause": "Genuine coincidence burst exceeded τ_e spacing.",
                    "Action taken": "Let burst pass; batch-reviewed deferred items.",
                    "Verification": "All deferred candidates logged and triaged.",
                    "Follow-up": "Schedule periodic deferred-candidate review.",
                },
                analysis_confirmation=(
                    "Every deferred candidate was reviewed; none were discarded "
                    "to clear the alert."),
            ),
            NoteTemplate(
                alert_name="SESICoincidenceRateSpike",
                title="Coincidence rate spike",
                sections={
                    "Root cause": "Astrophysical, instrumental, or environmental?",
                    "Action taken": "Assembled candidate list / ruled out artifacts.",
                    "Verification": "Cross-checked messengers independently.",
                    "Follow-up": "Routed to human science review.",
                },
                example_fragments={
                    "Root cause": "Correlated terrestrial noise in two channels.",
                    "Action taken": "Flagged as instrumental; documented cause.",
                    "Verification": "Spike absent when either channel is vetoed.",
                    "Follow-up": "Added veto channel; science team notified.",
                },
                analysis_confirmation=(
                    "I did NOT treat this spike as a detection; a human review "
                    "was initiated before any scientific claim."),
            ),
        ]}


# ═══════════════════════════════════════════════════════════════════════
# [D] RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════

class NoteTemplateRecommender:
    """
    [D] Recommend the appropriate note template for an alert, and render its
    scaffold. Unknown alerts get a generic template (never nothing).
    """

    def __init__(self):
        self.library = NoteTemplateLibrary.all_templates()

    def recommend(self, alert_name: str) -> NoteTemplate:
        if alert_name in self.library:
            return self.library[alert_name]
        # generic fallback so every alert has SOME structure
        return NoteTemplate(
            alert_name=alert_name,
            title=f"{alert_name} (generic)",
            sections={
                "Root cause": "What happened?",
                "Action taken": "What was done?",
                "Verification": "How was it confirmed?",
                "Follow-up": "What remains?",
            },
            example_fragments={},
            analysis_confirmation=(
                "I investigated this alert directly; no specific template "
                "existed, so I applied the generic structure."),
        )

    def scaffold_for(self, alert_name: str) -> str:
        return self.recommend(alert_name).render_scaffold()

    def coverage_report(self) -> Dict[str, Any]:
        """Report which known alerts have dedicated templates."""
        known_alerts = [
            "SESIPipelineConsumerDown", "SESIPipelineParseFailuresHigh",
            "SESINoZenoInvariantViolated", "SESIPipelineLatencyHigh",
            "SESIDeferredAlertsAccumulating", "SESICoincidenceRateSpike",
        ]
        covered = [a for a in known_alerts if a in self.library]
        return {
            "n_known_alerts": len(known_alerts),
            "n_covered": len(covered),
            "covered": covered,
            "all_covered": len(covered) == len(known_alerts),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Note Templates — Smoke Test")
    print("=" * 72)

    rec = NoteTemplateRecommender()

    # Coverage of known alerts
    print("\n  ── Template coverage ──")
    cov = rec.coverage_report()
    print(f"    known alerts : {cov['n_known_alerts']}")
    print(f"    covered      : {cov['n_covered']}")
    print(f"    [{'PASS' if cov['all_covered'] else 'FAIL'}] all known alerts have templates")

    # Render one scaffold
    print("\n  ── Example scaffold (No-Zeno violation) ──")
    scaffold = rec.scaffold_for("SESINoZenoInvariantViolated")
    for line in scaffold.splitlines():
        print(f"    {line}")

    # Verify analysis-confirmation is present in every template
    print("\n  ── Analysis-confirmation discipline ──")
    all_have_confirmation = all(
        t.analysis_confirmation for t in rec.library.values())
    print(f"    [{'PASS' if all_have_confirmation else 'FAIL'}] "
          f"every template requires an analysis confirmation")

    # Unknown alert → generic template, never nothing
    generic = rec.recommend("SomeUnknownAlert")
    print(f"\n    [{'PASS' if generic is not None else 'FAIL'}] "
          f"unknown alert falls back to a generic template")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Templates are RECOMMENDATIONS, not substitutes for analysis")
    print("  • Each template requires an explicit analysis confirmation, so")
    print("    they cannot be used to close alerts without engagement")
    print("  • Root cause is NEVER auto-filled — scaffold + examples only")
    print("  • Operational guidance; says nothing about astrophysics")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
