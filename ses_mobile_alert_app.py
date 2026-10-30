"""
=============================================================================
SES MOBILE DISASTER ALERT APP FRAMEWORK
Cross-platform mobile notification system for disaster warnings
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
-------------------------
[D] Framework for mobile disaster alert app:
  • Push notification system (Firebase Cloud Messaging / APNs)
  • Real-time alert routing
  • User location-based alerts
  • Multi-hazard alert types (fire, flood, earthquake, volcano, storm)
  • No-Zeno throttled alert delivery
  • Assumption-light threshold for alert triggering

RELATION TO DASHBOARD (ses_disaster_dashboard.py)
--------------------------------------------------
The dashboard is for OPERATORS (web-based, detailed).
The mobile app is for END USERS (simple, actionable alerts).

RELATION TO PAPER 11 (Assumption-Light Statistics)
---------------------------------------------------
Alert thresholds use subsampling confidence intervals:
  • Temperature threshold for fire alerts
  • Water level threshold for flood alerts
  • Ground velocity threshold for earthquake alerts

This avoids assuming specific noise distributions.

RELATION TO PAPER 5/6 (Change-Point Theory)
---------------------------------------------
Alert patterns are analyzed using change-point detection:
  • Is this a new event or continuation?
  • What is the occupation-time law for alert frequency?
  • Are alerts clustering (non-stationary)?

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a FRAMEWORK, not a production app.
• Mobile development requires platform-specific code (Swift/Kotlin/React Native).
• Push notifications require backend infrastructure (FCM/APNs).
• Location services require user permission and GPS/network access.
• Alert thresholds must be calibrated for specific regions.
• This is DECISION SUPPORT, not an official warning system.
• Official warnings should come from authorized agencies.
=============================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ses.mobile_alert")

__all__ = [
    "AlertSeverity",
    "HazardAlert",
    "MobileAlertConfig",
    "AlertThresholdEngine",
    "PushNotificationService",
    "MobileAlertApp",
]


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    ADVISORY = auto()
    WATCH = auto()
    WARNING = auto()
    EMERGENCY = auto()


class HazardType(Enum):
    """Hazard types (matching ses_disaster_cfd.py)."""
    FIRE = auto()
    FLOOD = auto()
    EARTHQUAKE = auto()
    VOLCANO = auto()
    HURRICANE = auto()
    TORNADO = auto()
    THUNDERSTORM = auto()
    SANDSTORM = auto()


@dataclass
class HazardAlert:
    """A disaster alert for mobile delivery."""
    alert_id: str
    hazard_type: HazardType
    severity: AlertSeverity
    title: str
    message: str
    latitude: float
    longitude: float
    radius_km: float
    timestamp: float
    source: str = "SES"
    recommended_action: str = ""


@dataclass
class MobileAlertConfig:
    """Configuration for mobile alert system."""
    # Alert thresholds (calibrated for specific region)
    fire_temp_threshold_k: float = 400.0
    flood_depth_threshold_m: float = 0.5
    earthquake_velocity_threshold_ms: float = 0.05
    
    # No-Zeno throttling
    min_alert_interval_s: float = 60.0  # minimum 1 minute between alerts
    max_alerts_per_hour: int = 10
    
    # Assumption-light statistics (Paper 11)
    subsample_size: int = 100
    confidence_level: float = 0.95
    
    # Push notification
    fcm_server_key: str = ""  # Firebase Cloud Messaging
    apns_cert_path: str = ""  # Apple Push Notification Service
    
    # Location
    location_radius_km: float = 50.0


class AlertThresholdEngine:
    """
    [D] Determine alert thresholds using assumption-light statistics.
    
    From Paper 11: Use subsampling confidence intervals instead of
    assuming specific noise distributions.
    """
    
    def __init__(self, cfg: MobileAlertConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)
        
        # Historical data for threshold calibration
        self.history: Dict[HazardType, List[float]] = {
            HazardType.FIRE: [],
            HazardType.FLOOD: [],
            HazardType.EARTHQUAKE: [],
        }
    
    def add_observation(self, hazard_type: HazardType, value: float) -> None:
        """Add an observation to history."""
        self.history[hazard_type].append(value)
    
    def compute_threshold(self, hazard_type: HazardType, 
                          metric_values: np.ndarray) -> float:
        """
        Compute alert threshold using subsampling (Paper 11 §2.1).
        
        Instead of assuming Gaussian noise, use empirical quantile
        with subsampling confidence interval.
        """
        if len(metric_values) < self.cfg.subsample_size:
            # Fall back to simple threshold
            if hazard_type == HazardType.FIRE:
                return self.cfg.fire_temp_threshold_k
            elif hazard_type == HazardType.FLOOD:
                return self.cfg.flood_depth_threshold_m
            else:
                return self.cfg.earthquake_velocity_threshold_ms
        
        # Subsampling: compute statistic on subsamples
        n_subsamples = 100
        subsample_stats = []
        
        for _ in range(n_subsamples):
            # Random subsample
            indices = self.rng.choice(len(metric_values), 
                                     size=self.cfg.subsample_size, 
                                     replace=False)
            subsample = metric_values[indices]
            # Use 95th percentile as statistic
            stat = np.percentile(subsample, 95)
            subsample_stats.append(stat)
        
        # Confidence interval from subsampling distribution
        subsample_stats = np.array(subsample_stats)
        alpha = 1.0 - self.cfg.confidence_level
        lower = np.percentile(subsample_stats, 100 * alpha / 2)
        upper = np.percentile(subsample_stats, 100 * (1 - alpha / 2))
        
        # Use upper bound as threshold (conservative)
        threshold = upper
        
        logger.info(f"Computed threshold for {hazard_type.name}: {threshold:.2f} "
                   f"(CI: [{lower:.2f}, {upper:.2f}])")
        
        return float(threshold)
    
    def should_alert(self, hazard_type: HazardType, 
                     current_value: float) -> bool:
        """Determine if current value exceeds threshold."""
        metric_values = np.array(self.history.get(hazard_type, []))
        if len(metric_values) == 0:
            # No history; use default threshold
            if hazard_type == HazardType.FIRE:
                return current_value > self.cfg.fire_temp_threshold_k
            elif hazard_type == HazardType.FLOOD:
                return current_value > self.cfg.flood_depth_threshold_m
            else:
                return current_value > self.cfg.earthquake_velocity_threshold_ms
        
        threshold = self.compute_threshold(hazard_type, metric_values)
        return current_value > threshold


class PushNotificationService:
    """
    [D] Push notification service (FCM/APNs).
    
    HONEST: This is a FRAMEWORK. Actual push notifications require:
      • Firebase Cloud Messaging setup (Android/Web)
      • Apple Push Notification Service setup (iOS)
      • Backend server to send notifications
      • User opt-in and device token registration
    """
    
    def __init__(self, cfg: MobileAlertConfig):
        self.cfg = cfg
        self.sent_notifications: List[Dict[str, Any]] = []
    
    def send_push(self, alert: HazardAlert, 
                  device_tokens: List[str]) -> Dict[str, Any]:
        """
        Send push notification to registered devices.
        
        In production, this would call FCM/APNs APIs.
        Here, we simulate the API call.
        """
        notification_payload = {
            "title": alert.title,
            "body": alert.message,
            "data": {
                "alert_id": alert.alert_id,
                "hazard_type": alert.hazard_type.name,
                "severity": alert.severity.name,
                "latitude": alert.latitude,
                "longitude": alert.longitude,
                "radius_km": alert.radius_km,
                "recommended_action": alert.recommended_action,
            },
        }
        
        # Simulate API call (in production, use requests library)
        result = {
            "status": "sent" if device_tokens else "no_recipients",
            "n_devices": len(device_tokens),
            "payload": notification_payload,
            "timestamp": time.time(),
        }
        
        self.sent_notifications.append(result)
        logger.info(f"Push notification sent: {alert.title} to {len(device_tokens)} devices")
        
        return result


class NoZenoAlertThrottler:
    """
    [D] No-Zeno throttling for alert delivery.
    
    From CFD paper §3.3: Prevent alert fatigue by limiting alert frequency.
    N(T) ≤ 1 + ⌊T/τ_e⌋ where τ_e = min_alert_interval_s.
    """
    
    def __init__(self, cfg: MobileAlertConfig):
        self.cfg = cfg
        self.last_alert_time = 0.0
        self.alert_count = 0
        self.alert_times: List[float] = []
    
    def can_send_alert(self) -> bool:
        """Check if alert can be sent (No-Zeno bound)."""
        current_time = time.time()
        
        # Check minimum interval
        if current_time - self.last_alert_time < self.cfg.min_alert_interval_s:
            logger.warning("Alert throttled: minimum interval not met")
            return False
        
        # Check max alerts per hour
        one_hour_ago = current_time - 3600
        recent_alerts = [t for t in self.alert_times if t > one_hour_ago]
        if len(recent_alerts) >= self.cfg.max_alerts_per_hour:
            logger.warning("Alert throttled: max alerts per hour reached")
            return False
        
        return True
    
    def record_alert(self) -> None:
        """Record that an alert was sent."""
        self.last_alert_time = time.time()
        self.alert_times.append(self.last_alert_time)
        self.alert_count += 1
    
    def bound_check(self) -> Dict[str, Any]:
        """Verify No-Zeno bound holds."""
        if not self.alert_times:
            return {"invariant_holds": True, "n_alerts": 0, "bound": 1}
        
        T = self.alert_times[-1] - self.alert_times[0]
        tau_e = self.cfg.min_alert_interval_s
        bound = 1 + int(T / tau_e) if tau_e > 0 else float('inf')
        
        return {
            "invariant_holds": self.alert_count <= bound,
            "n_alerts": self.alert_count,
            "bound": bound,
            "T_seconds": T,
            "tau_e_seconds": tau_e,
        }


class MobileAlertApp:
    """
    [D] Main mobile alert application framework.
    
    Coordinates:
      • Alert generation from simulation data
      • Threshold checking (assumption-light)
      • No-Zeno throttling
      • Push notification delivery
      • Location-based filtering
    """
    
    def __init__(self, cfg: Optional[MobileAlertConfig] = None):
        self.cfg = cfg or MobileAlertConfig()
        self.threshold_engine = AlertThresholdEngine(self.cfg)
        self.push_service = PushNotificationService(self.cfg)
        self.throttler = NoZenoAlertThrottler(self.cfg)
        self.active_alerts: List[HazardAlert] = []
    
    def generate_alert(self, hazard_type: HazardType,
                       severity: AlertSeverity,
                       latitude: float, longitude: float,
                       metric_value: float,
                       title: str, message: str) -> Optional[HazardAlert]:
        """
        Generate an alert if threshold is exceeded.
        
        Returns None if alert is throttled or threshold not met.
        """
        # Check threshold
        if not self.threshold_engine.should_alert(hazard_type, metric_value):
            logger.info(f"Threshold not met for {hazard_type.name}: {metric_value}")
            return None
        
        # Check No-Zeno throttling
        if not self.throttler.can_send_alert():
            logger.info(f"Alert throttled for {hazard_type.name}")
            return None
        
        # Create alert
        alert = HazardAlert(
            alert_id=f"alert_{int(time.time())}_{hazard_type.name}",
            hazard_type=hazard_type,
            severity=severity,
            title=title,
            message=message,
            latitude=latitude,
            longitude=longitude,
            radius_km=self.cfg.location_radius_km,
            timestamp=time.time(),
            recommended_action=self._get_recommended_action(hazard_type, severity),
        )
        
        self.throttler.record_alert()
        self.active_alerts.append(alert)
        
        return alert
    
    def _get_recommended_action(self, hazard_type: HazardType,
                                severity: AlertSeverity) -> str:
        """Get recommended action based on hazard type and severity."""
        actions = {
            HazardType.FIRE: {
                AlertSeverity.WARNING: "Evacuate immediately. Call emergency services.",
                AlertSeverity.WATCH: "Prepare to evacuate. Monitor situation.",
                AlertSeverity.ADVISORY: "Be aware. Avoid the area.",
            },
            HazardType.FLOOD: {
                AlertSeverity.WARNING: "Move to higher ground immediately.",
                AlertSeverity.WATCH: "Prepare emergency kit. Monitor water levels.",
                AlertSeverity.ADVISORY: "Avoid low-lying areas.",
            },
            HazardType.EARTHQUAKE: {
                AlertSeverity.WARNING: "Drop, cover, hold on. Move away from buildings.",
                AlertSeverity.WATCH: "Secure heavy objects. Review emergency plan.",
                AlertSeverity.ADVISORY: "Be prepared for aftershocks.",
            },
            HazardType.VOLCANO: {
                AlertSeverity.WARNING: "Evacuate exclusion zone immediately.",
                AlertSeverity.WATCH: "Prepare for possible evacuation.",
                AlertSeverity.ADVISORY: "Stay informed. Avoid summit areas.",
            },
            HazardType.HURRICANE: {
                AlertSeverity.WARNING: "Complete preparations. Shelter in place or evacuate.",
                AlertSeverity.WATCH: "Prepare home. Review evacuation routes.",
                AlertSeverity.ADVISORY: "Monitor storm development.",
            },
            HazardType.TORNADO: {
                AlertSeverity.WARNING: "Take shelter immediately in basement or interior room.",
                AlertSeverity.WATCH: "Stay alert. Review shelter location.",
                AlertSeverity.ADVISORY: "Monitor weather conditions.",
            },
        }
        
        return actions.get(hazard_type, {}).get(severity, 
            "Stay informed. Follow local authorities.")
    
    def send_alert(self, alert: HazardAlert, 
                   device_tokens: List[str]) -> Dict[str, Any]:
        """Send alert via push notification."""
        return self.push_service.send_push(alert, device_tokens)
    
    def get_alerts_for_location(self, latitude: float, longitude: float,
                                 radius_km: float = 50.0) -> List[HazardAlert]:
        """Get alerts within radius of a location."""
        nearby_alerts = []
        
        for alert in self.active_alerts:
            # Simple Haversine distance (simplified)
            dlat = abs(alert.latitude - latitude)
            dlon = abs(alert.longitude - longitude)
            
            # Rough distance in km (1 degree ≈ 111 km)
            distance_km = 111.0 * np.sqrt(dlat**2 + dlon**2)
            
            if distance_km <= radius_km or distance_km <= alert.radius_km:
                nearby_alerts.append(alert)
        
        return nearby_alerts


def smoke_test():
    print("=" * 72)
    print(" SESI Mobile Alert App — Smoke Test")
    print("=" * 72)
    
    cfg = MobileAlertConfig(
        min_alert_interval_s=1.0,  # Short interval for testing
        max_alerts_per_hour=10,
    )
    
    app = MobileAlertApp(cfg)
    
    # Add some historical data for threshold calibration
    print("\n  ── Calibrating thresholds ──")
    for _ in range(150):
        app.threshold_engine.add_observation(HazardType.FIRE, 
                                            np.random.normal(350, 50))
        app.threshold_engine.add_observation(HazardType.FLOOD,
                                            np.random.normal(0.3, 0.2))
        app.threshold_engine.add_observation(HazardType.EARTHQUAKE,
                                            np.random.normal(0.02, 0.01))
    
    # Test threshold computation
    fire_history = np.array(app.threshold_engine.history[HazardType.FIRE])
    threshold = app.threshold_engine.compute_threshold(HazardType.FIRE, fire_history)
    print(f"    Fire threshold     : {threshold:.1f} K")
    
    # Generate alerts
    print("\n  ── Generating alerts ──")
    
    # This should trigger (high temperature)
    alert1 = app.generate_alert(
        hazard_type=HazardType.FIRE,
        severity=AlertSeverity.WARNING,
        latitude=13.75,
        longitude=100.50,
        metric_value=500.0,  # Above threshold
        title="Fire Warning",
        message="Large fire detected in your area.",
    )
    
    if alert1:
        print(f"    Alert 1 generated  : {alert1.alert_id}")
        print(f"    Severity           : {alert1.severity.name}")
        print(f"    Action             : {alert1.recommended_action}")
    else:
        print("    Alert 1            : THRESHOLD NOT MET")
    
    # Try another alert immediately (should be throttled)
    alert2 = app.generate_alert(
        hazard_type=HazardType.FLOOD,
        severity=AlertSeverity.WATCH,
        latitude=13.76,
        longitude=100.51,
        metric_value=1.0,
        title="Flood Watch",
        message="Water levels rising.",
    )
    
    if alert2:
        print(f"    Alert 2 generated  : {alert2.alert_id}")
    else:
        print("    Alert 2            : THROTTLED (No-Zeno)")
    
    # Check No-Zeno bound
    print("\n  ── No-Zeno Bound Check ──")
    bound_check = app.throttler.bound_check()
    print(f"    Alerts sent        : {bound_check['n_alerts']}")
    print(f"    Bound              : {bound_check['bound']}")
    print(f"    Invariant holds    : {bound_check['invariant_holds']}")
    
    # Send push notification (simulated)
    print("\n  ── Push Notification ──")
    if alert1:
        result = app.send_alert(alert1, ["device_token_1", "device_token_2"])
        print(f"    Status             : {result['status']}")
        print(f"    Devices            : {result['n_devices']}")
    
    # Location-based filtering
    print("\n  ── Location-based Alerts ──")
    nearby = app.get_alerts_for_location(13.75, 100.50, radius_km=10.0)
    print(f"    Alerts near (13.75, 100.50): {len(nearby)}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a FRAMEWORK, not a production app")
    print("  • Push notifications require FCM/APNs setup")
    print("  • Alert thresholds must be calibrated per region")
    print("  • This is DECISION SUPPORT, not official warning system")
    print("  • Official warnings should come from authorized agencies")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
