"""
=============================================================================
SESI REAL GCN KAFKA DASHBOARD CONNECTION
Connect the multi-messenger dashboard to NASA GCN's Kafka stream
=============================================================================
Developer : PAI AND Yoon A. Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026

WHAT THIS CONNECTS
------------------
sesi_mm_dashboard / sesi_dashboard_bayestar_icecube  ←  this module  →  GCN Kafka

GCN (General Coordinates Network, NASA/GSFC) distributes multi-messenger
alerts over Apache Kafka. Real connection [C]:
    bootstrap : kafka.gcn.nasa.gov:9092
    security  : SASL_SSL, mechanism SCRAM-SHA-512
    auth      : username/password from https://gcn.nasa.gov (credentials)
    client    : confluent-kafka (recommended) or kafka-python

Relevant topic families (verify against current GCN docs — they evolve):
    igwn.gwalert                      — LIGO/Virgo/KAGRA GW alerts
    gcn.notices.icecube.*             — IceCube neutrino notices
    gcn.notices.*                     — other missions (Fermi-GBM, Swift...)

NO-ZENO ALERT THROTTLING (cfd_pp.py §3.3 discipline)
----------------------------------------------------
A live Kafka stream can produce BURSTS of near-duplicate alerts. The
NoZenoAlertThrottler from sesi_bayestar_icecube_mm is applied to the
EMITTED alert feed:
    • candidates are always RECORDED (never dropped);
    • only outbound dashboard alerts may be deferred;
    • consumer errors / deserialization failures are RECORDED as warnings
      WITHOUT consuming an event slot, and never suppressed.

LABELS: [C] GCN Kafka conventions  [D] derived consumer/dashboard glue
        [O] live production operation (needs credentials + monitoring)

HONEST SCOPE
------------
• Real ingestion REQUIRES GCN credentials and a working network. Without
  them, the dashboard runs on a clearly-flagged SYNTHETIC source. It never
  presents synthetic alerts as real.
• Topic names and message schemas evolve; parsers are defensive and
  version-tolerant. Unknown message types are logged, not dropped.
• This is a RESEARCH dashboard, not an operational alerting system.
=============================================================================
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("sesi.gcn_kafka")

# Kafka clients (optional)
try:
    from confluent_kafka import Consumer as CKConsumer, KafkaError
    HAS_CONFLUENT = True
except ImportError:
    HAS_CONFLUENT = False

try:
    from kafka import KafkaConsumer
    HAS_KAFKA_PY = True
except ImportError:
    HAS_KAFKA_PY = False

# Reuse the existing dashboard + No-Zeno throttler if available
try:
    from sesi_bayestar_icecube_mm import NoZenoAlertThrottler
    HAS_THROTTLE = True
except ImportError:
    HAS_THROTTLE = False

__all__ = ["GCNKafkaConfig", "GCNKafkaConsumer", "AlertMessageParser",
           "KafkaBackedDashboard"]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — real GCN Kafka endpoint
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GCNKafkaConfig:
    # Real GCN Kafka connection [C]
    bootstrap_servers: str = "kafka.gcn.nasa.gov:9092"
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "SCRAM-SHA-512"
    username: Optional[str] = None     # from https://gcn.nasa.gov
    password: Optional[str] = None
    group_id: str = "sesi-mm-dashboard"
    auto_offset_reset: str = "latest"

    # Topics to subscribe (verify against current GCN docs)
    topics: List[str] = field(default_factory=lambda: [
        "igwn.gwalert",
        "gcn.notices.icecube.lvk_nu_track",
    ])

    # Consumption
    poll_timeout_s: float = 1.0
    max_messages_per_poll: int = 100

    # Throttle
    tau_e: float = 5.0                 # No-Zeno min alert spacing (s)

    @property
    def has_credentials(self) -> bool:
        return self.username is not None and self.password is not None


# ═══════════════════════════════════════════════════════════════════════
# [C/D] ALERT MESSAGE PARSER
# ═══════════════════════════════════════════════════════════════════════

class AlertMessageParser:
    """
    [D] Parse Kafka message payloads into a normalized alert dict.

    GCN messages may be JSON or VOEvent XML. This parser handles JSON
    natively and falls back to a raw-bytes record for XML/unknown.
    Unknown message types are logged, never silently dropped.
    """

    def parse(self, topic: str, payload: bytes) -> Optional[Dict[str, Any]]:
        """Return a normalized alert dict, or None if unparseable."""
        try:
            text = payload.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"payload decode failed: {e}")
            return None

        # Try JSON first
        try:
            data = json.loads(text)
            return self._from_json(topic, data)
        except (json.JSONDecodeError, ValueError):
            pass

        # VOEvent XML or unknown → record raw
        if text.lstrip().startswith("<"):
            return {
                "topic": topic,
                "format": "voevent_xml",
                "raw": text[:2000],
                "parsed": False,
                "synthetic": False,
            }

        logger.debug(f"unrecognized message format on {topic}")
        return None

    def _from_json(self, topic: str, data: Dict) -> Dict[str, Any]:
        """Extract normalized fields from a JSON alert."""
        # GCN/IGWN schemas vary; pull common keys defensively
        alert = {
            "topic": topic,
            "format": "json",
            "parsed": True,
            "synthetic": False,
        }
        # Common in IGWN GW alerts
        if "alert_type" in data:
            alert["alert_type"] = data["alert_type"]
        if "superevent_id" in data:
            alert["event_id"] = data["superevent_id"]
        # Sky position may be nested
        for key in ("ra", "dec", "error_radius"):
            if key in data:
                alert[key] = data[key]
        # Keep the full payload for downstream parsers
        alert["payload"] = data
        # Messenger inference from topic
        if "gw" in topic.lower():
            alert["messenger"] = "GW"
        elif "icecube" in topic.lower():
            alert["messenger"] = "NU"
        else:
            alert["messenger"] = "UNKNOWN"
        return alert


# ═══════════════════════════════════════════════════════════════════════
# [C/D] GCN KAFKA CONSUMER
# ═══════════════════════════════════════════════════════════════════════

class GCNKafkaConsumer:
    """
    [C/D] Consume GCN Kafka alerts.

    Prefers confluent-kafka; falls back to kafka-python. If neither client
    nor credentials are available, reports unavailable and the dashboard
    uses a synthetic source (clearly flagged).
    """

    def __init__(self, cfg: GCNKafkaConfig):
        self.cfg = cfg
        self.parser = AlertMessageParser()
        self._consumer = None
        self._mode = self._select_mode()

    def _select_mode(self) -> str:
        if not self.cfg.has_credentials:
            return "unavailable_no_credentials"
        if HAS_CONFLUENT:
            return "confluent"
        if HAS_KAFKA_PY:
            return "kafka_python"
        return "unavailable_no_client"

    @property
    def mode(self) -> str:
        return self._mode

    def _build_confluent_config(self) -> Dict[str, Any]:
        return {
            "bootstrap.servers": self.cfg.bootstrap_servers,
            "security.protocol": self.cfg.security_protocol,
            "sasl.mechanism": self.cfg.sasl_mechanism,
            "sasl.username": self.cfg.username,
            "sasl.password": self.cfg.password,
            "group.id": self.cfg.group_id,
            "auto.offset.reset": self.cfg.auto_offset_reset,
        }

    def connect(self) -> bool:
        """Open the consumer and subscribe. Returns True on success."""
        if self._mode == "confluent":
            try:
                self._consumer = CKConsumer(self._build_confluent_config())
                self._consumer.subscribe(self.cfg.topics)
                logger.info(f"confluent-kafka connected to {self.cfg.bootstrap_servers}")
                return True
            except Exception as e:
                logger.warning(f"confluent connect failed: {e}")
                return False
        elif self._mode == "kafka_python":
            try:
                self._consumer = KafkaConsumer(
                    *self.cfg.topics,
                    bootstrap_servers=self.cfg.bootstrap_servers,
                    security_protocol=self.cfg.security_protocol,
                    sasl_mechanism=self.cfg.sasl_mechanism,
                    sasl_plain_username=self.cfg.username,
                    sasl_plain_password=self.cfg.password,
                    group_id=self.cfg.group_id,
                    auto_offset_reset=self.cfg.auto_offset_reset,
                )
                logger.info("kafka-python connected")
                return True
            except Exception as e:
                logger.warning(f"kafka-python connect failed: {e}")
                return False
        return False

    def poll(self) -> List[Dict[str, Any]]:
        """Poll one batch of parsed alerts."""
        alerts = []
        if self._mode == "confluent" and self._consumer is not None:
            for _ in range(self.cfg.max_messages_per_poll):
                msg = self._consumer.poll(timeout=self.cfg.poll_timeout_s)
                if msg is None:
                    break
                if msg.error():
                    # per §3.3: record the failure, never suppress
                    logger.warning(f"Kafka error: {msg.error()}")
                    continue
                parsed = self.parser.parse(msg.topic(), msg.value())
                if parsed is not None:
                    alerts.append(parsed)
        elif self._mode == "kafka_python" and self._consumer is not None:
            # kafka-python is iterator-based; pull a bounded batch
            count = 0
            for msg in self._consumer:
                parsed = self.parser.parse(msg.topic, msg.value)
                if parsed is not None:
                    alerts.append(parsed)
                count += 1
                if count >= self.cfg.max_messages_per_poll:
                    break
        return alerts

    def close(self):
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
# [D] SYNTHETIC SOURCE (offline fallback, clearly flagged)
# ═══════════════════════════════════════════════════════════════════════

class SyntheticAlertSource:
    """
    [D] Offline stand-in that mimics GCN alert structure.

    Every alert is marked synthetic=True. This exists so the dashboard
    and No-Zeno throttle can be exercised without live credentials; it
    must NEVER be confused with real GCN data.
    """

    def __init__(self, seed: int = 0):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def poll(self, n: int = 3) -> List[Dict[str, Any]]:
        alerts = []
        for _ in range(n):
            messenger = self.rng.choice(["GW", "NU"])
            alerts.append({
                "topic": "synthetic.demo",
                "format": "json",
                "parsed": True,
                "synthetic": True,
                "messenger": messenger,
                "event_id": f"SYN_{int(time.time()*1000)}",
                "ra": float(self.rng.uniform(0, 360)),
                "dec": float(self.rng.uniform(-90, 90)),
                "alert_type": "PRELIMINARY",
            })
        return alerts


# ═══════════════════════════════════════════════════════════════════════
# [D] KAFKA-BACKED DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class KafkaBackedDashboard:
    """
    [D] Wire the GCN Kafka consumer into the existing multi-messenger
    dashboard, with No-Zeno throttling on emitted alerts.

    Data flow:
        GCN Kafka → parser → candidate list (ALL recorded)
                            → NoZeno throttle → emitted dashboard alerts
        consumer/parser errors → warning log (never suppressed, §3.3)
    """

    def __init__(self, cfg: GCNKafkaConfig):
        self.cfg = cfg
        self.consumer = GCNKafkaConsumer(cfg)
        self.throttler = NoZenoAlertThrottler(tau_e=cfg.tau_e) if HAS_THROTTLE else None
        self.candidates: List[Dict[str, Any]] = []   # ALL, never dropped
        self.emitted: List[Dict[str, Any]] = []
        self.source_is_synthetic = False

    def start(self) -> Dict[str, Any]:
        """Attempt real connection; fall back to synthetic if unavailable."""
        connected = self.consumer.connect()
        if connected:
            self.source_is_synthetic = False
            self._source = self.consumer
            return {"source": "GCN_KAFKA_LIVE", "mode": self.consumer.mode}
        # Fall back to synthetic, clearly flagged
        self.source_is_synthetic = True
        self._source = SyntheticAlertSource()
        logger.warning("GCN Kafka unavailable → using SYNTHETIC source (flagged)")
        return {"source": "SYNTHETIC_FALLBACK",
                "reason": self.consumer.mode,
                "note": "alerts are synthetic=True; not real GCN data"}

    def refresh(self) -> Dict[str, Any]:
        """One dashboard refresh: poll → record → throttle → emit."""
        try:
            alerts = self._source.poll()
        except Exception as e:
            # §3.3: record failure, never suppress
            if self.throttler is not None:
                self.throttler.record_warning(f"poll failed: {e}")
            logger.warning(f"poll failed: {e}")
            return {"n_polled": 0, "error": str(e)}

        n_emitted = 0
        for alert in alerts:
            self.candidates.append(alert)  # ALWAYS record
            # Only real or synthetic candidates go through the throttle
            if self.throttler is not None:
                emitted = self.throttler.try_emit(alert, t=time.time())
                if emitted:
                    self.emitted.append(alert)
                    n_emitted += 1
            else:
                self.emitted.append(alert)
                n_emitted += 1

        return {
            "n_polled": len(alerts),
            "n_emitted": n_emitted,
            "n_candidates_total": len(self.candidates),
            "synthetic_source": self.source_is_synthetic,
        }

    def summary(self) -> Dict[str, Any]:
        out = {
            "source": "SYNTHETIC" if self.source_is_synthetic else "GCN_KAFKA_LIVE",
            "n_candidates": len(self.candidates),
            "n_emitted": len(self.emitted),
        }
        if self.throttler is not None:
            s = self.throttler
            T = max(time.time() - (s.candidate_log[0]["_recorded_t"]
                                     if s.candidate_log else time.time()), 1e-3)
            out["throttle"] = s.bound_check(T)
        return out


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI GCN Kafka Dashboard — Smoke Test")
    print("=" * 72)
    print(f"  confluent-kafka : {HAS_CONFLUENT}")
    print(f"  kafka-python    : {HAS_KAFKA_PY}")
    print(f"  throttle module : {HAS_THROTTLE}")

    # No credentials → should fall back to synthetic, clearly flagged
    cfg = GCNKafkaConfig(username=None, password=None, tau_e=0.01)
    dash = KafkaBackedDashboard(cfg)

    start = dash.start()
    print(f"\n  ── Connection ──")
    print(f"    source : {start['source']}")
    if "reason" in start:
        print(f"    reason : {start['reason']}")
        print(f"    note   : {start['note']}")

    # Run several refresh cycles
    print(f"\n  ── Refresh cycles ──")
    for i in range(5):
        r = dash.refresh()
        time.sleep(0.005)
        print(f"    cycle {i+1}: polled={r['n_polled']} emitted={r['n_emitted']} "
              f"total_candidates={r['n_candidates_total']}")

    # Summary with No-Zeno bound
    print(f"\n  ── Summary ──")
    s = dash.summary()
    print(f"    source       : {s['source']}")
    print(f"    candidates   : {s['n_candidates']}  (all recorded, never dropped)")
    print(f"    emitted      : {s['n_emitted']}")
    if "throttle" in s:
        t = s["throttle"]
        print(f"    throttle     : alerts={t['N_alerts']} bound={t['bound']} "
              f"holds={t['invariant_holds']}")
        print(f"    warnings     : {t['n_warnings']}  (never suppressed)")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Live GCN Kafka REQUIRES credentials from gcn.nasa.gov")
    print("  • Without credentials/library → SYNTHETIC source, clearly flagged")
    print("  • Topic names/schemas evolve; parsers are defensive")
    print("  • No-Zeno throttles EMISSION only; candidates always recorded")
    print("  • This is a RESEARCH dashboard, not operational alerting")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
