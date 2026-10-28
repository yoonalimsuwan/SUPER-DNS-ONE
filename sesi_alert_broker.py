"""
=============================================================================
SESI REAL-TIME MULTI-MESSENGER ALERT BROKER
GraceDB/GCN-style alert ingestion, filtering, cross-matching, dispatch
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT A REAL ALERT PIPELINE LOOKS LIKE [C]
-----------------------------------------
  detector strain
      -> low-latency search (GstLAL / PyCBC Live / MBTA / SPIIR)
      -> GraceDB (gracedb.ligo.org) — superevent + VOEvent
      -> GCN (gcn.nasa.gov) — Kafka + VOEvent fan-out
      -> observatory brokers (this module's layer)
      -> telescope scheduling / follow-up

This module implements the BROKER layer: ingestion, schema validation,
filtering, spatial+temporal cross-matching, and pub/sub dispatch.

REAL ACCESS NOTES
-----------------
• GraceDB requires LIGO/Virgo/KAGRA credentials (X.509 proxy or token).
• GCN Kafka now offers public topics for selected notices.
• Without credentials, this broker runs in LOCAL/SIMULATED mode with
  clearly-labeled synthetic alerts. It never pretends a synthetic alert
  is a real detection.

HONEST SCOPE
------------
• This is broker INFRASTRUCTURE, not a detection algorithm.
• Cross-match coincidence probabilities are heuristic until tied to a
  real false-alarm-rate budget.
• No claim that any alert implies an astrophysical (let alone artificial)
  source; alerts are CANDIDATES pending human + instrument review.
=============================================================================
"""

from __future__ import annotations

import json
import math
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("sesi.alert_broker")

__all__ = ["MessengerType", "AlertEvent", "VOEventParser", "AlertFilter",
           "CrossMatcher", "AlertBroker", "GraceDBClient", "GCNKafkaClient"]


class MessengerType(Enum):
    GRAVITATIONAL_WAVE = "GW"
    NEUTRINO = "NU"
    GAMMA_RAY = "GRB"
    OPTICAL_TRANSIENT = "OPT"
    UNKNOWN = "UNK"


@dataclass
class AlertEvent:
    """Normalized multi-messenger alert schema."""
    alert_id: str
    messenger: MessengerType
    trigger_time: float              # Unix/GPS seconds
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    error_radius_deg: Optional[float] = None
    energy: Optional[float] = None   # messenger-appropriate units
    snr: Optional[float] = None
    far: Optional[float] = None      # false alarm rate (Hz)
    payload: Dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False          # MUST be True for simulated alerts

    def has_position(self) -> bool:
        return self.ra_deg is not None and self.dec_deg is not None


class VOEventParser:
    """
    [D] Parse IVOA VOEvent XML into AlertEvent.

    VOEvent is the standard alert format used by GCN. Real packets carry
    a <WhereWhen> (space-time position) and <Param> list. We extract the
    essential fields; full VOEvent 2.0 parsing is delegated to a library
    in production (e.g. voevent-parse).
    """

    def parse(self, xml_text: str) -> Optional[AlertEvent]:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
        except Exception as e:
            logger.warning(f"VOEvent parse failed: {e}")
            return None

        def _find(tag):
            # namespace-agnostic search
            for el in root.iter():
                if el.tag.split('}')[-1] == tag:
                    return el
            return None

        ivorn_el = _find("Ivorn")
        alert_id = ivorn_el.text if ivorn_el is not None else "unknown"

        # Position from <Position2Unit> or RA/Dec params
        ra = dec = err = None
        for el in root.iter():
            t = el.tag.split('}')[-1]
            if t in ("RA", "ra") and el.text:
                ra = self._hms_to_deg(el.text)
            elif t in ("Dec", "dec") and el.text:
                dec = self._dms_to_deg(el.text)
            elif t == "Error2Radius" and el.text:
                try: err = float(el.text)
                except ValueError: pass

        return AlertEvent(
            alert_id=str(alert_id),
            messenger=MessengerType.GRAVITATIONAL_WAVE,  # refine by packet type
            trigger_time=time.time(),
            ra_deg=ra, dec_deg=dec, error_radius_deg=err,
            synthetic=False,
            payload={"raw_ivorn": str(alert_id)},
        )

    @staticmethod
    def _hms_to_deg(hms: str) -> Optional[float]:
        try:
            parts = [float(x) for x in hms.replace(":", " ").split()]
            h, m, s = (parts + [0.0] * 3)[:3]
            return (h + m / 60 + s / 3600) * 15.0
        except Exception:
            return None

    @staticmethod
    def _dms_to_deg(dms: str) -> Optional[float]:
        try:
            parts = [float(x) for x in dms.replace(":", " ").split()]
            d, m, s = (parts + [0.0] * 3)[:3]
            sign = -1 if d < 0 or dms.strip().startswith("-") else 1
            return sign * (abs(d) + m / 60 + s / 3600)
        except Exception:
            return None


class AlertFilter:
    """
    [D] Declarative alert filtering rules.

    A broker typically drops low-significance alerts early to protect
    downstream subscribers.
    """

    def __init__(self):
        self.rules: List[Callable[[AlertEvent], bool]] = []

    def add_rule(self, name: str, fn: Callable[[AlertEvent], bool]):
        self.rules.append(fn)
        logger.info(f"Filter rule added: {name}")

    def keep_above_far(self, max_far: float):
        """Keep alerts with FAR below a threshold (more significant)."""
        self.add_rule(
            f"far<{max_far}",
            lambda a: (a.far is None) or (a.far <= max_far),
        )

    def keep_with_position(self):
        self.add_rule("has_position", lambda a: a.has_position())

    def passes(self, alert: AlertEvent) -> bool:
        return all(rule(alert) for rule in self.rules)


class CrossMatcher:
    """
    [D] Spatial + temporal cross-matching of alerts.

    Two alerts are a coincidence candidate if:
      • |Δt| < time_window
      • angular separation < sum of error radii (or a fixed radius)
    """

    def __init__(self, time_window_s: float = 300.0,
                 fallback_radius_deg: float = 5.0):
        self.time_window_s = time_window_s
        self.fallback_radius_deg = fallback_radius_deg

    @staticmethod
    def angular_separation(ra1, dec1, ra2, dec2) -> float:
        r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
        cos_sep = (math.sin(d1) * math.sin(d2)
                   + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))

    def match(self, a: AlertEvent, b: AlertEvent) -> Optional[Dict[str, Any]]:
        if not (a.has_position() and b.has_position()):
            return None
        dt = abs(a.trigger_time - b.trigger_time)
        if dt > self.time_window_s:
            return None
        sep = self.angular_separation(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg)
        allowed = (a.error_radius_deg or self.fallback_radius_deg) + \
                  (b.error_radius_deg or self.fallback_radius_deg)
        if sep > allowed:
            return None
        return {
            "alert_a": a.alert_id, "alert_b": b.alert_id,
            "messenger_a": a.messenger.value, "messenger_b": b.messenger.value,
            "dt_s": dt, "separation_deg": sep, "allowed_deg": allowed,
            "synthetic_involved": a.synthetic or b.synthetic,
        }


class AlertBroker:
    """
    [D] In-memory pub/sub alert broker.

    Production deployments replace the queue with Kafka and the store
    with a real database; the routing/filter/match logic is the same.
    """

    def __init__(self, filter_rules: Optional[AlertFilter] = None,
                 matcher: Optional[CrossMatcher] = None):
        self.filter_rules = filter_rules or AlertFilter()
        self.matcher = matcher or CrossMatcher()
        self._alerts: List[AlertEvent] = []
        self._subscribers: Dict[str, Callable[[AlertEvent], None]] = {}
        self._coincidences: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def subscribe(self, name: str, callback: Callable[[AlertEvent], None]):
        self._subscribers[name] = callback

    def publish(self, alert: AlertEvent) -> bool:
        """Ingest one alert; returns True if it passed filters."""
        if not self.filter_rules.passes(alert):
            logger.debug(f"Alert {alert.alert_id} dropped by filter")
            return False
        with self._lock:
            # Cross-match against existing alerts
            for existing in self._alerts:
                coincidence = self.matcher.match(alert, existing)
                if coincidence:
                    self._coincidences.append(coincidence)
                    logger.info(f"Coincidence: {coincidence['alert_a']} "
                                f"x {coincidence['alert_b']}")
            self._alerts.append(alert)
        # Dispatch to subscribers
        for name, cb in self._subscribers.items():
            try:
                cb(alert)
            except Exception as e:
                logger.warning(f"Subscriber {name} failed: {e}")
        return True

    @property
    def alerts(self) -> List[AlertEvent]:
        return list(self._alerts)

    @property
    def coincidences(self) -> List[Dict[str, Any]]:
        return list(self._coincidences)


class GraceDBClient:
    """
    [O] GraceDB REST client — REQUIRES LVK CREDENTIALS.

    Without credentials this returns a clear 'unauthenticated' status
    rather than fabricating events.
    """

    BASE = "https://gracedb.ligo.org/api/"

    def __init__(self, cert: Optional[str] = None, token: Optional[str] = None):
        self.cert = cert
        self.token = token

    def authenticated(self) -> bool:
        return self.cert is not None or self.token is not None

    def list_superevents(self, n: int = 10) -> Dict[str, Any]:
        if not self.authenticated():
            return {"error": "GraceDB requires LVK credentials; not authenticated",
                    "authenticated": False}
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            r = requests.get(self.BASE + "superevents/", params={"N": n},
                             headers=headers, cert=self.cert, timeout=30)
            if r.status_code == 200:
                return {"authenticated": True, "data": r.json()}
            return {"authenticated": True, "status": r.status_code,
                    "note": "check credentials/scopes"}
        except Exception as e:
            return {"error": str(e)}


class GCNKafkaClient:
    """
    [O] GCN Kafka consumer — public topics available without full auth.

    Requires `confluent-kafka` or `kafka-python` for a real connection.
    """

    def __init__(self, bootstrap: str = "kafka.gcn.nasa.gov:9092"):
        self.bootstrap = bootstrap

    def available(self) -> bool:
        try:
            import kafka  # noqa
            return True
        except ImportError:
            return False

    def consume(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        if not self.available():
            return [{"error": "kafka-python/confluent-kafka not installed"}]
        # Real implementation would poll a KafkaConsumer here.
        return [{"note": "Kafka consumer scaffold; wire real poll loop"}]


def smoke_test():
    print("=" * 72)
    print(" SESI Real-Time Alert Broker — Smoke Test")
    print("=" * 72)

    broker = AlertBroker(matcher=CrossMatcher(time_window_s=120.0))
    broker.filter_rules.keep_above_far(max_far=1e-7)

    received = []
    broker.subscribe("logger", lambda a: received.append(a.alert_id))

    # Two synthetic GW alerts close in space+time (clearly marked synthetic)
    now = time.time()
    a1 = AlertEvent("GW_TEST_1", MessengerType.GRAVITATIONAL_WAVE, now,
                    ra_deg=120.0, dec_deg=30.0, error_radius_deg=2.0,
                    snr=12.0, far=1e-9, synthetic=True)
    a2 = AlertEvent("NU_TEST_1", MessengerType.NEUTRINO, now + 30.0,
                    ra_deg=121.0, dec_deg=30.5, error_radius_deg=1.5,
                    energy=2e5, synthetic=True)
    a3 = AlertEvent("GW_LOW_SIG", MessengerType.GRAVITATIONAL_WAVE, now,
                    ra_deg=200.0, dec_deg=-10.0, far=1e-3, synthetic=True)

    broker.publish(a1)
    broker.publish(a2)
    broker.publish(a3)   # should be filtered out by FAR rule

    print(f"  Alerts ingested   : {len(broker.alerts)}")
    print(f"  Coincidences      : {len(broker.coincidences)}")
    print(f"  Dispatched to sub : {received}")
    for c in broker.coincidences:
        print(f"    {c['alert_a']} x {c['alert_b']} "
              f"(dt={c['dt_s']:.0f}s, sep={c['separation_deg']:.2f} deg, "
              f"synthetic={c['synthetic_involved']})")

    # GraceDB / GCN auth status
    gdb = GraceDBClient()
    print(f"\n  GraceDB auth      : {gdb.authenticated()} "
          f"({'ready' if gdb.authenticated() else 'credentials required'})")
    gcn = GCNKafkaClient()
    print(f"  GCN Kafka client  : {'kafka lib present' if gcn.available() else 'kafka lib absent'}")

    print("\n  HONEST SCOPE")
    print("  • All test alerts are SYNTHETIC (synthetic=True)")
    print("  • GraceDB needs LVK credentials; GCN Kafka needs a client lib")
    print("  • A coincidence is a CANDIDATE, not a confirmed source")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
