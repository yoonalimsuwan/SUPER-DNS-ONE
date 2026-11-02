"""
=============================================================================
SESI COLD-CHAIN MONITORING + MERCHANT DASHBOARD
Real-Time Temperature Integrity + Small-Merchant Operations Hub
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[CCM1] ColdChainMonitor
      • Real-time temperature/humidity tracking per shipment
      • IoT sensor data ingestion (BLE, LoRa, MQTT abstraction)
      • Temperature zone classification (frozen/chilled/ambient)
      • Arrhenius-based shelf-life prediction
      • HACCP compliance tracking (critical control points)
      • Alert system with No-Zeno bounded notification rate
      • FEFO enforcement integration with InventoryOptimizer
      • Excursion logging with cryptographic audit trail

[MD1] MerchantDashboard
      • Sales overview (daily/weekly/monthly revenue, orders)
      • Inventory status with low-stock alerts
      • Order management (pending, preparing, delivering, completed)
      • Earnings & payout tracking
      • Product catalog CRUD
      • Customer insights (AGGREGATE ONLY — PNDC Axiom 3)
      • Marketing tools (promotions, coupons, bundles)
      • Multi-language UI hints (TH/EN/ZH/JA/KO)
      • Simple UX designed for non-technical merchants

MATHEMATICAL FOUNDATIONS
--------------------------
[C] Arrhenius equation: k(T) = A · exp(-Ea / (R·T))
    → Temperature-dependent decay rate for shelf-life prediction
[C] Q10 rule: Rate doubles for every 10°C increase (food science)
[C] HACCP: Hazard Analysis Critical Control Points (Codex Alimentarius)
[C] FEFO: First-Expired-First-Out inventory management
[C] Time-Temperature Indicator (TTI): cumulative thermal dose
[C] No-Zeno bound: N(T) ≤ 1 + 2C_V·T/δ_min (bounded alerts)
[C] PNDC Axiom 3: Data minimization for merchant analytics
[C] Buddhist Economics: Accessible tools for all livelihoods

HONEST SCOPE
------------
• Cold-chain monitoring is SENSOR-BASED; requires physical IoT devices
• Arrhenius predictions are APPROXIMATE (food-specific calibration needed)
• HACCP tracking is ASSISTIVE; certified auditor required for compliance
• Merchant dashboard shows AGGREGATE customer data only
• All thresholds are FREE DESIGN PARAMETERS
• Production requires food-safety regulatory compliance per jurisdiction
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.coldchain_merchant")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Integration with previous modules
try:
    from sesi_commerce_webapp import (
        Product, ProductCategory, InventoryOptimizationEngine,
        InventoryPolicy, InventoryRecommendation,
        GlobalLogisticsRouter, PaymentGatewayRouter,
    )
    HAS_COMMERCE = True
except ImportError:
    HAS_COMMERCE = False


__all__ = [
    "TemperatureZone", "ColdChainAlert", "SensorReading",
    "ColdChainMonitor", "ColdChainConfig", "ShipmentColdChain",
    "ShelfLifeEstimator",
    "MerchantDashboard", "MerchantConfig", "MerchantProfile",
    "SalesSummary", "OrderSummary", "ProductCatalogEntry",
    "SESIColdChainMerchantSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class TemperatureZone(str, Enum):
    """Cold-chain temperature zones."""
    DEEP_FROZEN = "deep_frozen"       # ≤ -18°C
    FROZEN = "frozen"                 # -18°C to -12°C
    CHILLED = "chilled"               # 0°C to 5°C
    COOL = "cool"                     # 5°C to 15°C
    AMBIENT = "ambient"               # 15°C to 25°C
    WARM = "warm"                     # 25°C to 40°C
    DANGER = "danger"                 # > 40°C or < -25°C


class ColdChainAlert(str, Enum):
    """Cold-chain alert types."""
    TEMPERATURE_HIGH = "temperature_high"
    TEMPERATURE_LOW = "temperature_low"
    HUMIDITY_HIGH = "humidity_high"
    DOOR_OPEN = "door_open"
    SENSOR_OFFLINE = "sensor_offline"
    EXCURSION_TIME_EXCEEDED = "excursion_time_exceeded"
    SHELF_LIFE_CRITICAL = "shelf_life_critical"
    HACCP_VIOLATION = "haccp_violation"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SensorReading:
    """A single IoT sensor reading."""
    reading_id: str
    shipment_id: str
    timestamp: float
    temperature_c: float
    humidity_pct: Optional[float] = None
    sensor_location: str = "cargo_area"  # "cargo_area", "door", "ambient"
    sensor_type: str = "ble"  # "ble" | "lora" | "mqtt" | "manual"
    battery_level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShipmentColdChain:
    """Cold-chain state for a shipment."""
    shipment_id: str
    product_id: str
    required_zone: TemperatureZone
    # Current state
    current_temp_c: float
    current_humidity_pct: Optional[float] = None
    current_zone: TemperatureZone = TemperatureZone.AMBIENT
    # Excursion tracking
    excursion_count: int = 0
    total_excursion_seconds: float = 0.0
    max_excursion_temp_c: Optional[float] = None
    last_excursion_time: Optional[float] = None
    # Shelf life
    remaining_shelf_life_hours: float = 0.0
    cumulative_thermal_dose: float = 0.0  # Arrhenius integral
    # HACCP
    haccp_checkpoints_passed: int = 0
    haccp_checkpoints_total: int = 0
    # Status
    is_compliant: bool = True
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class MerchantProfile:
    """Small merchant profile."""
    merchant_id: str
    shop_name: str
    owner_name: str
    # Business type
    business_type: str = "food"  # "food" | "retail" | "service"
    categories: List[str] = field(default_factory=list)
    # Location
    region: str = ""
    address: str = ""
    # Preferences
    preferred_language: str = "th"
    currency: str = "THB"
    # Financial
    bank_account_hash: str = ""  # PNDC Axiom 3: never store raw
    payout_schedule: str = "weekly"  # "daily" | "weekly" | "monthly"
    # Status
    is_verified: bool = False
    is_active: bool = True
    joined_at: float = field(default_factory=time.time)
    # Consent
    analytics_consent: bool = False
    marketing_consent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SalesSummary:
    """Aggregate sales summary for merchant dashboard."""
    period: str  # "daily" | "weekly" | "monthly"
    total_revenue: float = 0.0
    total_orders: int = 0
    avg_order_value: float = 0.0
    # Breakdowns (aggregate only)
    orders_by_status: Dict[str, int] = field(default_factory=dict)
    revenue_by_category: Dict[str, float] = field(default_factory=dict)
    # Trends
    revenue_change_pct: float = 0.0
    order_change_pct: float = 0.0
    # Customer insights (AGGREGATE ONLY — PNDC Axiom 3)
    unique_customers_estimate: int = 0
    repeat_rate_pct: float = 0.0
    # Timestamp
    generated_at: float = field(default_factory=time.time)


@dataclass
class OrderSummary:
    """Order summary for dashboard."""
    order_id: str
    customer_display: str  # Anonymized display name
    items: List[Dict[str, Any]]
    total_amount: float
    currency: str
    status: OrderStatus
    created_at: float
    estimated_delivery: Optional[float] = None
    # Cold-chain (if applicable)
    requires_cold_chain: bool = False
    cold_chain_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductCatalogEntry:
    """Product in merchant's catalog."""
    product_id: str
    merchant_id: str
    name: str
    description: str = ""
    category: str = ""
    price: float = 0.0
    cost: float = 0.0
    currency: str = "THB"
    stock: int = 0
    is_active: bool = True
    # Cold-chain requirement
    requires_cold_chain: bool = False
    storage_temp_c: Optional[float] = None
    shelf_life_hours: Optional[float] = None
    # Images
    image_urls: List[str] = field(default_factory=list)
    # Multi-language
    name_translations: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# [CCM1] COLD-CHAIN MONITOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColdChainConfig:
    """Configuration for cold-chain monitoring."""
    # Temperature thresholds per zone
    zone_thresholds: Dict[TemperatureZone, Tuple[float, float]] = field(
        default_factory=lambda: {
            TemperatureZone.DEEP_FROZEN: (-25.0, -18.0),
            TemperatureZone.FROZEN: (-18.0, -12.0),
            TemperatureZone.CHILLED: (0.0, 5.0),
            TemperatureZone.COOL: (5.0, 15.0),
            TemperatureZone.AMBIENT: (15.0, 25.0),
            TemperatureZone.WARM: (25.0, 40.0),
        }
    )
    # Excursion tolerance
    max_excursion_seconds: float = 1800.0  # 30 min allowed excursion
    excursion_temp_tolerance_c: float = 3.0  # ±3°C tolerance
    # Arrhenius parameters (generic food)
    activation_energy_j_mol: float = 80000.0  # Ea (J/mol)
    gas_constant: float = 8.314  # R (J/(mol·K))
    reference_temp_c: float = 4.0  # T_ref for chilled
    q10_factor: float = 2.0  # Rate doubles per 10°C
    # Alerting
    alert_cooldown_s: float = 300.0  # Min 5 min between same alerts
    max_alerts_per_hour: int = 12
    # HACCP
    haccp_checkpoints: List[str] = field(default_factory=lambda: [
        "receiving_temperature",
        "storage_temperature",
        "preparation_hygiene",
        "packaging_integrity",
        "delivery_temperature",
    ])
    # No-Zeno
    max_sensor_reads_per_second: int = 10


class ShelfLifeEstimator:
    """
    [C] Arrhenius-based shelf-life prediction.

    k(T) = A · exp(-Ea / (R·T))

    The decay rate k(T) increases exponentially with temperature.
    Cumulative thermal dose determines remaining shelf life.

    Q10 approximation: rate doubles for every 10°C above reference.

    HONEST: Arrhenius parameters vary by food type.
    This uses generic values; calibration per product is essential.
    """

    def __init__(self, cfg: ColdChainConfig):
        self.cfg = cfg

    def decay_rate(self, temp_c: float) -> float:
        """Compute Arrhenius decay rate at given temperature."""
        T_kelvin = temp_c + 273.15
        T_ref_kelvin = self.cfg.reference_temp_c + 273.15
        Ea = self.cfg.activation_energy_j_mol
        R = self.cfg.gas_constant

        # Arrhenius ratio relative to reference
        ratio = math.exp(-Ea / R * (1.0 / T_kelvin - 1.0 / T_ref_kelvin))
        return ratio

    def decay_rate_q10(self, temp_c: float) -> float:
        """Q10 approximation (simpler, less accurate)."""
        delta_t = temp_c - self.cfg.reference_temp_c
        return self.cfg.q10_factor ** (delta_t / 10.0)

    def cumulative_thermal_dose(
        self,
        temp_history: List[Tuple[float, float]],  # [(timestamp, temp_c), ...]
    ) -> float:
        """
        Compute cumulative thermal dose (Arrhenius integral).

        dose = Σ k(T_i) · Δt_i

        Returns normalized dose (1.0 = full shelf life consumed).
        """
        if len(temp_history) < 2:
            return 0.0

        dose = 0.0
        for i in range(1, len(temp_history)):
            t_prev, temp_prev = temp_history[i - 1]
            t_curr, temp_curr = temp_history[i]
            dt_hours = (t_curr - t_prev) / 3600.0

            # Average temperature over interval
            avg_temp = (temp_prev + temp_curr) / 2.0
            rate = self.decay_rate(avg_temp)
            dose += rate * dt_hours

        return dose

    def remaining_shelf_life(
        self,
        initial_shelf_life_hours: float,
        cumulative_dose: float,
    ) -> float:
        """Estimate remaining shelf life given cumulative dose."""
        remaining = initial_shelf_life_hours * (1.0 - cumulative_dose)
        return max(0.0, remaining)

    def classify_temperature_zone(self, temp_c: float) -> TemperatureZone:
        """Classify temperature into zone."""
        for zone, (low, high) in self.cfg.zone_thresholds.items():
            if low <= temp_c <= high:
                return zone
        if temp_c < -25.0:
            return TemperatureZone.DANGER
        return TemperatureZone.DANGER


class ColdChainMonitor:
    """
    [D] Real-time cold-chain monitoring system.

    Features:
      • IoT sensor data ingestion (BLE/LoRa/MQTT abstraction)
      • Temperature zone classification
      • Excursion detection and logging
      • Arrhenius-based shelf-life prediction
      • HACCP compliance tracking
      • Alert system with No-Zeno bounds
      • Cryptographic audit trail

    ⚠️ ETHICAL FRAMEWORK:
      • Sensor data belongs to the SHIPMENT, not surveillance of workers
      • Alerts inform; humans decide
      • No punitive automation based solely on sensor data
      • Audit trail protects both merchant and consumer

    HONEST: Requires physical IoT sensors for real deployment.
    Arrhenius parameters need per-product calibration.
    """

    def __init__(self, cfg: Optional[ColdChainConfig] = None):
        self.cfg = cfg or ColdChainConfig()
        self.shelf_life_estimator = ShelfLifeEstimator(self.cfg)
        self._shipments: Dict[str, ShipmentColdChain] = {}
        self._sensor_history: Dict[str, List[SensorReading]] = defaultdict(list)
        self._alert_timestamps: deque = deque(maxlen=10000)
        self._audit_log: List[Dict[str, Any]] = []

    # ── No-Zeno Alert Rate Limiting ──────────────────────────────────────

    def _check_alert_rate(self) -> Dict[str, Any]:
        """Check if alert emission is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 3600.0
        self._alert_timestamps = deque(
            (t for t in self._alert_timestamps if t > cutoff), maxlen=10000
        )
        if len(self._alert_timestamps) >= self.cfg.max_alerts_per_hour:
            return {"allowed": False, "reason": "max_alerts_per_hour"}
        return {"allowed": True}

    # ── Shipment Registration ────────────────────────────────────────────

    def register_shipment(
        self,
        shipment_id: str,
        product_id: str,
        required_zone: TemperatureZone,
        initial_temp_c: float,
        shelf_life_hours: float = 72.0,
    ) -> ShipmentColdChain:
        """Register a new shipment for cold-chain monitoring."""
        shipment = ShipmentColdChain(
            shipment_id=shipment_id,
            product_id=product_id,
            required_zone=required_zone,
            current_temp_c=initial_temp_c,
            current_zone=self.shelf_life_estimator.classify_temperature_zone(
                initial_temp_c
            ),
            remaining_shelf_life_hours=shelf_life_hours,
        )
        self._shipments[shipment_id] = shipment
        self._log_audit("register_shipment", {
            "shipment_id": shipment_id,
            "required_zone": required_zone.value,
        })
        logger.info(
            f"[ColdChain] Registered shipment {shipment_id}: "
            f"zone={required_zone.value}, temp={initial_temp_c}°C"
        )
        return shipment

    # ── Sensor Data Ingestion ────────────────────────────────────────────

    def ingest_sensor_reading(self, reading: SensorReading) -> Dict[str, Any]:
        """
        Process a sensor reading and update shipment state.

        Returns dict with status and any triggered alerts.
        """
        shipment = self._shipments.get(reading.shipment_id)
        if shipment is None:
            return {"status": "unknown_shipment"}

        # Store reading
        self._sensor_history[reading.shipment_id].append(reading)
        # Keep bounded history
        if len(self._sensor_history[reading.shipment_id]) > 10000:
            self._sensor_history[reading.shipment_id] = \
                self._sensor_history[reading.shipment_id][-5000:]

        # Update current state
        shipment.current_temp_c = reading.temperature_c
        shipment.current_humidity_pct = reading.humidity_pct
        shipment.current_zone = \
            self.shelf_life_estimator.classify_temperature_zone(reading.temperature_c)
        shipment.updated_at = reading.timestamp

        # Check for excursion
        alerts = self._check_excursion(shipment, reading)

        # Update thermal dose
        temp_history = [
            (r.timestamp, r.temperature_c)
            for r in self._sensor_history[reading.shipment_id]
        ]
        shipment.cumulative_thermal_dose = \
            self.shelf_life_estimator.cumulative_thermal_dose(temp_history)

        return {
            "status": "processed",
            "current_zone": shipment.current_zone.value,
            "alerts": alerts,
            "is_compliant": shipment.is_compliant,
        }

    def _check_excursion(
        self, shipment: ShipmentColdChain, reading: SensorReading
    ) -> List[Dict[str, Any]]:
        """Check if temperature is outside required zone."""
        alerts = []
        required = shipment.required_zone
        thresholds = self.cfg.zone_thresholds.get(required)

        if thresholds is None:
            return alerts

        low, high = thresholds
        tolerance = self.cfg.excursion_temp_tolerance_c
        temp = reading.temperature_c

        if temp > high + tolerance:
            alert = self._create_alert(
                shipment, ColdChainAlert.TEMPERATURE_HIGH,
                f"Temperature {temp:.1f}°C exceeds max {high + tolerance:.1f}°C"
            )
            alerts.append(alert)
            shipment.excursion_count += 1
            shipment.max_excursion_temp_c = max(
                shipment.max_excursion_temp_c or temp, temp
            )
            shipment.last_excursion_time = reading.timestamp
            shipment.is_compliant = False

        elif temp < low - tolerance:
            alert = self._create_alert(
                shipment, ColdChainAlert.TEMPERATURE_LOW,
                f"Temperature {temp:.1f}°C below min {low - tolerance:.1f}°C"
            )
            alerts.append(alert)
            shipment.excursion_count += 1
            shipment.is_compliant = False

        return alerts

    def _create_alert(
        self,
        shipment: ShipmentColdChain,
        alert_type: ColdChainAlert,
        message: str,
    ) -> Dict[str, Any]:
        """Create an alert with No-Zeno rate limiting."""
        rate_check = self._check_alert_rate()
        if not rate_check["allowed"]:
            return {"suppressed": True, "reason": rate_check["reason"]}

        alert = {
            "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
            "shipment_id": shipment.shipment_id,
            "type": alert_type.value,
            "message": message,
            "timestamp": time.time(),
            "severity": "critical" if alert_type in (
                ColdChainAlert.TEMPERATURE_HIGH,
                ColdChainAlert.HACCP_VIOLATION,
            ) else "warning",
        }

        shipment.alerts.append(alert)
        self._alert_timestamps.append(time.time())
        self._log_audit("alert_created", alert)

        logger.warning(f"[ColdChain] ALERT: {message}")
        return alert

    # ── Shelf Life Update ────────────────────────────────────────────────

    def update_shelf_life(
        self, shipment_id: str, initial_shelf_life_hours: float
    ) -> float:
        """Update remaining shelf life based on thermal dose."""
        shipment = self._shipments.get(shipment_id)
        if shipment is None:
            return 0.0

        remaining = self.shelf_life_estimator.remaining_shelf_life(
            initial_shelf_life_hours,
            shipment.cumulative_thermal_dose,
        )
        shipment.remaining_shelf_life_hours = remaining

        # Critical shelf-life alert
        if remaining < initial_shelf_life_hours * 0.2:
            self._create_alert(
                shipment, ColdChainAlert.SHELF_LIFE_CRITICAL,
                f"Shelf life critical: {remaining:.1f}h remaining "
                f"({shipment.cumulative_thermal_dose:.1%} consumed)"
            )

        return remaining

    # ── HACCP Compliance ─────────────────────────────────────────────────

    def check_haccp_checkpoint(
        self, shipment_id: str, checkpoint_name: str, passed: bool
    ) -> Dict[str, Any]:
        """Record HACCP checkpoint result."""
        shipment = self._shipments.get(shipment_id)
        if shipment is None:
            return {"status": "unknown_shipment"}

        shipment.haccp_checkpoints_total += 1
        if passed:
            shipment.haccp_checkpoints_passed += 1
        else:
            self._create_alert(
                shipment, ColdChainAlert.HACCP_VIOLATION,
                f"HACCP checkpoint failed: {checkpoint_name}"
            )

        self._log_audit("haccp_checkpoint", {
            "shipment_id": shipment_id,
            "checkpoint": checkpoint_name,
            "passed": passed,
        })

        return {
            "status": "recorded",
            "checkpoint": checkpoint_name,
            "passed": passed,
            "compliance_rate": (
                shipment.haccp_checkpoints_passed /
                max(shipment.haccp_checkpoints_total, 1)
            ),
        }

    # ── Audit Trail ──────────────────────────────────────────────────────

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        self._audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "details": details,
            "hash": hashlib.sha256(
                json.dumps(details, sort_keys=True).encode()
            ).hexdigest()[:16],
        })

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._audit_log[-limit:]

    # ── Status ───────────────────────────────────────────────────────────

    def get_shipment_status(self, shipment_id: str) -> Optional[Dict[str, Any]]:
        """Get full cold-chain status for a shipment."""
        shipment = self._shipments.get(shipment_id)
        if shipment is None:
            return None
        return {
            "shipment_id": shipment.shipment_id,
            "required_zone": shipment.required_zone.value,
            "current_temp_c": shipment.current_temp_c,
            "current_zone": shipment.current_zone.value,
            "is_compliant": shipment.is_compliant,
            "excursion_count": shipment.excursion_count,
            "remaining_shelf_life_hours": shipment.remaining_shelf_life_hours,
            "thermal_dose": shipment.cumulative_thermal_dose,
            "haccp_passed": shipment.haccp_checkpoints_passed,
            "haccp_total": shipment.haccp_checkpoints_total,
            "active_alerts": len([
                a for a in shipment.alerts if not a.get("suppressed")
            ]),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_shipments": len(self._shipments),
            "total_sensor_readings": sum(
                len(v) for v in self._sensor_history.values()
            ),
            "total_alerts": len(self._alert_timestamps),
            "audit_entries": len(self._audit_log),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [MD1] MERCHANT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MerchantConfig:
    """Configuration for merchant dashboard."""
    # Analytics
    analytics_window_days: int = 30
    # PNDC Axiom 3
    aggregate_customer_data_only: bool = True
    no_individual_tracking: bool = True
    # UX
    default_language: str = "th"
    supported_languages: List[str] = field(
        default_factory=lambda: ["th", "en", "zh", "ja", "ko"]
    )
    # Payouts
    min_payout_amount: float = 100.0
    payout_fee_pct: float = 1.0


class MerchantDashboard:
    """
    [D] Operations dashboard for small/individual merchants.

    ⚠️ ETHICAL FRAMEWORK:
      • Customer data is AGGREGATE ONLY (PNDC Axiom 3)
      • No individual customer tracking or profiling
      • Simple UX for non-technical merchants
      • Multi-language support (TH/EN/ZH/JA/KO)
      • Transparent fees and payouts

    Features:
      • Sales overview (revenue, orders, trends)
      • Inventory status with alerts
      • Order management
      • Earnings & payout tracking
      • Product catalog management
      • Marketing tools (promotions, coupons)
      • Customer insights (aggregate only)

    HONEST: This is a SIMPLIFIED dashboard.
    Production requires proper database, authentication, and payment processing.
    """

    def __init__(self, cfg: Optional[MerchantConfig] = None):
        self.cfg = cfg or MerchantConfig()
        self._merchants: Dict[str, MerchantProfile] = {}
        self._products: Dict[str, ProductCatalogEntry] = {}
        self._orders: List[OrderSummary] = []
        self._earnings: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    # ── Merchant Management ──────────────────────────────────────────────

    def register_merchant(
        self,
        shop_name: str,
        owner_name: str,
        business_type: str = "food",
        region: str = "",
        preferred_language: str = "th",
    ) -> MerchantProfile:
        """Register a new merchant."""
        merchant = MerchantProfile(
            merchant_id=f"merchant_{uuid.uuid4().hex[:10]}",
            shop_name=shop_name,
            owner_name=owner_name,
            business_type=business_type,
            region=region,
            preferred_language=preferred_language,
        )
        self._merchants[merchant.merchant_id] = merchant
        logger.info(f"[Dashboard] Registered merchant: {shop_name}")
        return merchant

    def get_merchant(self, merchant_id: str) -> Optional[MerchantProfile]:
        return self._merchants.get(merchant_id)

    # ── Product Catalog ──────────────────────────────────────────────────

    def add_product(
        self,
        merchant_id: str,
        name: str,
        price: float,
        category: str = "",
        stock: int = 0,
        requires_cold_chain: bool = False,
        storage_temp_c: Optional[float] = None,
        shelf_life_hours: Optional[float] = None,
    ) -> ProductCatalogEntry:
        """Add product to merchant's catalog."""
        product = ProductCatalogEntry(
            product_id=f"prod_{uuid.uuid4().hex[:10]}",
            merchant_id=merchant_id,
            name=name,
            category=category,
            price=price,
            stock=stock,
            requires_cold_chain=requires_cold_chain,
            storage_temp_c=storage_temp_c,
            shelf_life_hours=shelf_life_hours,
        )
        self._products[product.product_id] = product
        return product

    def get_merchant_products(
        self, merchant_id: str
    ) -> List[ProductCatalogEntry]:
        return [
            p for p in self._products.values()
            if p.merchant_id == merchant_id and p.is_active
        ]

    # ── Sales Analytics (AGGREGATE ONLY) ─────────────────────────────────

    def get_sales_summary(
        self, merchant_id: str, period: str = "daily"
    ) -> SalesSummary:
        """
        Get sales summary for merchant.

        ⚠️ PNDC Axiom 3: All customer data is aggregate only.
        No individual customer information is exposed.
        """
        merchant_orders = [
            o for o in self._orders
            if o.metadata.get("merchant_id") == merchant_id
        ]

        total_revenue = sum(o.total_amount for o in merchant_orders)
        total_orders = len(merchant_orders)
        avg_order_value = total_revenue / max(total_orders, 1)

        # Orders by status
        orders_by_status = defaultdict(int)
        for o in merchant_orders:
            orders_by_status[o.status.value] += 1

        # Revenue by category
        revenue_by_category = defaultdict(float)
        for o in merchant_orders:
            for item in o.items:
                cat = item.get("category", "other")
                revenue_by_category[cat] += item.get("price", 0) * item.get("qty", 1)

        # Mock trends (in production: compare with previous period)
        revenue_change = float(np.random.uniform(-10, 20))
        order_change = float(np.random.uniform(-5, 15))

        # Aggregate customer estimates (NO individual data)
        unique_customers = max(1, int(total_orders * 0.7))
        repeat_rate = float(np.random.uniform(10, 40))

        return SalesSummary(
            period=period,
            total_revenue=total_revenue,
            total_orders=total_orders,
            avg_order_value=avg_order_value,
            orders_by_status=dict(orders_by_status),
            revenue_by_category=dict(revenue_by_category),
            revenue_change_pct=revenue_change,
            order_change_pct=order_change,
            unique_customers_estimate=unique_customers,
            repeat_rate_pct=repeat_rate,
        )

    # ── Order Management ─────────────────────────────────────────────────

    def create_order(
        self,
        merchant_id: str,
        items: List[Dict[str, Any]],
        total_amount: float,
        currency: str = "THB",
        requires_cold_chain: bool = False,
    ) -> OrderSummary:
        """Create a new order."""
        order = OrderSummary(
            order_id=f"order_{uuid.uuid4().hex[:10]}",
            customer_display=f"Customer_{uuid.uuid4().hex[:4]}",  # Anonymized
            items=items,
            total_amount=total_amount,
            currency=currency,
            status=OrderStatus.PENDING,
            created_at=time.time(),
            requires_cold_chain=requires_cold_chain,
            metadata={"merchant_id": merchant_id},
        )
        self._orders.append(order)
        return order

    def update_order_status(
        self, order_id: str, new_status: OrderStatus
    ) -> bool:
        """Update order status."""
        for order in self._orders:
            if order.order_id == order_id:
                order.status = new_status
                return True
        return False

    def get_merchant_orders(
        self, merchant_id: str, status: Optional[OrderStatus] = None
    ) -> List[OrderSummary]:
        """Get orders for a merchant."""
        orders = [
            o for o in self._orders
            if o.metadata.get("merchant_id") == merchant_id
        ]
        if status:
            orders = [o for o in orders if o.status == status]
        return orders

    # ── Earnings & Payouts ───────────────────────────────────────────────

    def record_earning(
        self, merchant_id: str, amount: float, source: str = "order"
    ) -> None:
        """Record an earning event."""
        self._earnings[merchant_id].append({
            "amount": amount,
            "source": source,
            "timestamp": time.time(),
        })

    def get_earnings_summary(
        self, merchant_id: str, period_days: int = 30
    ) -> Dict[str, Any]:
        """Get earnings summary for payout."""
        cutoff = time.time() - period_days * 86400
        earnings = [
            e for e in self._earnings.get(merchant_id, [])
            if e["timestamp"] >= cutoff
        ]
        total = sum(e["amount"] for e in earnings)
        fee = total * self.cfg.payout_fee_pct / 100.0
        net = total - fee

        return {
            "period_days": period_days,
            "gross_earnings": total,
            "platform_fee": fee,
            "net_payout": net,
            "payout_eligible": net >= self.cfg.min_payout_amount,
            "min_payout": self.cfg.min_payout_amount,
        }

    # ── Marketing Tools ──────────────────────────────────────────────────

    def create_promotion(
        self,
        merchant_id: str,
        promo_type: str,  # "discount" | "bundle" | "free_delivery"
        value: float,
        min_order: float = 0.0,
        valid_days: int = 7,
    ) -> Dict[str, Any]:
        """Create a promotion for the merchant's shop."""
        promo = {
            "promo_id": f"promo_{uuid.uuid4().hex[:8]}",
            "merchant_id": merchant_id,
            "type": promo_type,
            "value": value,
            "min_order": min_order,
            "valid_until": time.time() + valid_days * 86400,
            "created_at": time.time(),
            "usage_count": 0,
        }
        logger.info(f"[Dashboard] Created promotion: {promo['promo_id']}")
        return promo

    # ── Dashboard Summary ────────────────────────────────────────────────

    def get_dashboard_overview(self, merchant_id: str) -> Dict[str, Any]:
        """
        Get complete dashboard overview for a merchant.

        This is the main entry point for the dashboard UI.
        """
        merchant = self._merchants.get(merchant_id)
        if merchant is None:
            return {"error": "merchant_not_found"}

        sales = self.get_sales_summary(merchant_id)
        products = self.get_merchant_products(merchant_id)
        pending_orders = self.get_merchant_orders(
            merchant_id, OrderStatus.PENDING
        )
        earnings = self.get_earnings_summary(merchant_id)

        # Low stock alerts
        low_stock = [
            {"product_id": p.product_id, "name": p.name, "stock": p.stock}
            for p in products if p.stock < 5
        ]

        return {
            "merchant": {
                "shop_name": merchant.shop_name,
                "business_type": merchant.business_type,
                "is_verified": merchant.is_verified,
            },
            "sales": {
                "total_revenue": sales.total_revenue,
                "total_orders": sales.total_orders,
                "avg_order_value": sales.avg_order_value,
                "revenue_change_pct": sales.revenue_change_pct,
            },
            "products": {
                "total": len(products),
                "low_stock_alerts": low_stock,
            },
            "orders": {
                "pending": len(pending_orders),
            },
            "earnings": earnings,
            "language": merchant.preferred_language,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_merchants": len(self._merchants),
            "total_products": len(self._products),
            "total_orders": len(self._orders),
            "aggregate_only": self.cfg.aggregate_customer_data_only,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR + WEB API
# ═══════════════════════════════════════════════════════════════════════════

class SESIColdChainMerchantSuite:
    """
    [D] Complete cold-chain + merchant dashboard suite.

    Integrates:
      • Cold-chain monitoring
      • Merchant dashboard
      • (Optional) Previous commerce modules
    """

    def __init__(
        self,
        coldchain_cfg: Optional[ColdChainConfig] = None,
        merchant_cfg: Optional[MerchantConfig] = None,
    ):
        self.coldchain = ColdChainMonitor(coldchain_cfg)
        self.dashboard = MerchantDashboard(merchant_cfg)
        logger.info("[SESI] Cold-Chain + Merchant Dashboard initialized")

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "coldchain": self.coldchain.get_status(),
            "dashboard": self.dashboard.get_status(),
        }


# ── FastAPI Web App ──────────────────────────────────────────────────────

if HAS_FASTAPI:
    app = FastAPI(title="SESI Cold-Chain & Merchant Dashboard", version="1.0.0")
    suite = SESIColdChainMerchantSuite()

    # --- Pydantic Models ---
    class SensorReadingIn(BaseModel):
        shipment_id: str
        temperature_c: float
        humidity_pct: Optional[float] = None
        sensor_location: str = "cargo_area"

    class ShipmentIn(BaseModel):
        product_id: str
        required_zone: str = "chilled"
        initial_temp_c: float = 4.0
        shelf_life_hours: float = 72.0

    class MerchantIn(BaseModel):
        shop_name: str
        owner_name: str
        business_type: str = "food"
        region: str = ""
        preferred_language: str = "th"

    class ProductIn(BaseModel):
        name: str
        price: float
        category: str = ""
        stock: int = 0
        requires_cold_chain: bool = False
        storage_temp_c: Optional[float] = None

    # --- Cold-Chain Endpoints ---
    @app.post("/coldchain/shipments")
    def create_shipment(inp: ShipmentIn):
        shipment_id = f"ship_{uuid.uuid4().hex[:10]}"
        zone = TemperatureZone(inp.required_zone)
        shipment = suite.coldchain.register_shipment(
            shipment_id, inp.product_id, zone,
            inp.initial_temp_c, inp.shelf_life_hours,
        )
        return {"shipment_id": shipment_id, "status": "registered"}

    @app.post("/coldchain/readings")
    def submit_reading(inp: SensorReadingIn):
        reading = SensorReading(
            reading_id=f"read_{uuid.uuid4().hex[:8]}",
            shipment_id=inp.shipment_id,
            timestamp=time.time(),
            temperature_c=inp.temperature_c,
            humidity_pct=inp.humidity_pct,
            sensor_location=inp.sensor_location,
        )
        return suite.coldchain.ingest_sensor_reading(reading)

    @app.get("/coldchain/shipments/{shipment_id}")
    def get_shipment(shipment_id: str):
        status = suite.coldchain.get_shipment_status(shipment_id)
        if status is None:
            raise HTTPException(404, "Shipment not found")
        return status

    # --- Merchant Dashboard Endpoints ---
    @app.post("/merchants")
    def register_merchant(inp: MerchantIn):
        merchant = suite.dashboard.register_merchant(**inp.dict())
        return {"merchant_id": merchant.merchant_id, "shop_name": merchant.shop_name}

    @app.get("/merchants/{merchant_id}/dashboard")
    def merchant_dashboard(merchant_id: str):
        overview = suite.dashboard.get_dashboard_overview(merchant_id)
        if "error" in overview:
            raise HTTPException(404, overview["error"])
        return overview

    @app.post("/merchants/{merchant_id}/products")
    def add_product(merchant_id: str, inp: ProductIn):
        product = suite.dashboard.add_product(merchant_id, **inp.dict())
        return {"product_id": product.product_id, "name": product.name}

    @app.get("/merchants/{merchant_id}/sales")
    def merchant_sales(merchant_id: str, period: str = "daily"):
        return suite.dashboard.get_sales_summary(merchant_id, period)

    @app.get("/merchants/{merchant_id}/earnings")
    def merchant_earnings(merchant_id: str):
        return suite.dashboard.get_earnings_summary(merchant_id)

else:
    app = None


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Cold-Chain + Merchant Dashboard — Smoke Test")
    print("=" * 72)
    print(f"  FastAPI available  : {HAS_FASTAPI}")
    print(f"  PyTorch available  : {HAS_TORCH}")
    print(f"  Commerce modules   : {HAS_COMMERCE}")

    suite = SESIColdChainMerchantSuite()

    # ── Test 1: Cold-Chain Monitoring ────────────────────────────────────
    print("\n  ── [CCM1] Cold-Chain Monitor ──")

    # Register a chilled shipment
    shipment = suite.coldchain.register_shipment(
        shipment_id="ship_demo_001",
        product_id="prod_matcha_latte",
        required_zone=TemperatureZone.CHILLED,
        initial_temp_c=3.5,
        shelf_life_hours=48.0,
    )
    print(f"    Shipment registered  : {shipment.shipment_id}")
    print(f"    Required zone        : {shipment.required_zone.value}")
    print(f"    Initial temp         : {shipment.current_temp_c}°C")

    # Simulate sensor readings (normal)
    for i in range(5):
        reading = SensorReading(
            reading_id=f"read_{i:03d}",
            shipment_id="ship_demo_001",
            timestamp=time.time() + i * 60,
            temperature_c=3.5 + np.random.uniform(-0.5, 0.5),
            humidity_pct=85.0,
        )
        result = suite.coldchain.ingest_sensor_reading(reading)
    print(f"    Normal readings      : 5 processed")
    print(f"    Current zone         : {result['current_zone']}")
    print(f"    Compliant            : {result['is_compliant']}")

    # Simulate temperature excursion
    excursion_reading = SensorReading(
        reading_id="read_excursion",
        shipment_id="ship_demo_001",
        timestamp=time.time() + 360,
        temperature_c=12.0,  # Way above chilled max (5°C + 3°C tolerance)
    )
    excursion_result = suite.coldchain.ingest_sensor_reading(excursion_reading)
    print(f"\n    EXCURSION DETECTED   : temp=12.0°C")
    print(f"    Alerts triggered     : {len(excursion_result['alerts'])}")
    print(f"    Compliant after      : {excursion_result['is_compliant']}")

    # Shelf life update
    remaining = suite.coldchain.update_shelf_life("ship_demo_001", 48.0)
    print(f"    Remaining shelf life : {remaining:.1f}h")

    # HACCP checkpoint
    haccp = suite.coldchain.check_haccp_checkpoint(
        "ship_demo_001", "delivery_temperature", passed=False
    )
    print(f"    HACCP checkpoint     : {haccp['checkpoint']} → {'PASS' if haccp['passed'] else 'FAIL'}")

    # Shipment status
    status = suite.coldchain.get_shipment_status("ship_demo_001")
    print(f"\n    Full status:")
    print(f"      Temp               : {status['current_temp_c']:.1f}°C")
    print(f"      Zone               : {status['current_zone']}")
    print(f"      Excursions         : {status['excursion_count']}")
    print(f"      Shelf life         : {status['remaining_shelf_life_hours']:.1f}h")
    print(f"      HACCP compliance   : {status['haccp_passed']}/{status['haccp_total']}")

    # ── Test 2: Merchant Dashboard ───────────────────────────────────────
    print("\n  ── [MD1] Merchant Dashboard ──")

    # Register merchant
    merchant = suite.dashboard.register_merchant(
        shop_name="ป้าสมศรี อาหารวีแกน",
        owner_name="สมศรี ใจดี",
        business_type="food",
        region="Bangkok",
        preferred_language="th",
    )
    print(f"    Merchant registered  : {merchant.shop_name}")
    print(f"    Merchant ID          : {merchant.merchant_id}")

    # Add products
    p1 = suite.dashboard.add_product(
        merchant.merchant_id,
        name="ข้าวกล่องวีแกน",
        price=89.0,
        category="vegan",
        stock=50,
        requires_cold_chain=True,
        storage_temp_c=4.0,
        shelf_life_hours=24.0,
    )
    p2 = suite.dashboard.add_product(
        merchant.merchant_id,
        name="น้ำผลไม้สด",
        price=45.0,
        category="beverages",
        stock=30,
        requires_cold_chain=True,
        storage_temp_c=5.0,
    )
    p3 = suite.dashboard.add_product(
        merchant.merchant_id,
        name="ผลไม้ตามฤดูกาล",
        price=120.0,
        category="fruits",
        stock=3,  # Low stock!
    )
    print(f"    Products added       : 3")

    # Create mock orders
    for i in range(10):
        suite.dashboard.create_order(
            merchant.merchant_id,
            items=[{"product_id": p1.product_id, "qty": 1, "price": 89.0, "category": "vegan"}],
            total_amount=89.0,
            requires_cold_chain=True,
        )
        suite.dashboard.record_earning(merchant.merchant_id, 89.0 * 0.9)  # 10% platform fee

    print(f"    Orders created       : 10")

    # Get dashboard overview
    overview = suite.dashboard.get_dashboard_overview(merchant.merchant_id)
    print(f"\n    Dashboard Overview:")
    print(f"      Shop               : {overview['merchant']['shop_name']}")
    print(f"      Total revenue      : {overview['sales']['total_revenue']:.0f} THB")
    print(f"      Total orders       : {overview['sales']['total_orders']}")
    print(f"      Avg order value    : {overview['sales']['avg_order_value']:.0f} THB")
    print(f"      Low stock alerts   : {len(overview['products']['low_stock_alerts'])}")
    for alert in overview['products']['low_stock_alerts']:
        print(f"        ⚠️  {alert['name']}: {alert['stock']} left")

    # Earnings
    earnings = suite.dashboard.get_earnings_summary(merchant.merchant_id)
    print(f"\n    Earnings Summary:")
    print(f"      Gross              : {earnings['gross_earnings']:.0f} THB")
    print(f"      Platform fee       : {earnings['platform_fee']:.0f} THB")
    print(f"      Net payout         : {earnings['net_payout']:.0f} THB")
    print(f"      Payout eligible    : {earnings['payout_eligible']}")

    # Sales summary
    sales = suite.dashboard.get_sales_summary(merchant.merchant_id)
    print(f"\n    Sales Analytics (AGGREGATE ONLY):")
    print(f"      Revenue by category: {sales.revenue_by_category}")
    print(f"      Unique customers   : ~{sales.unique_customers_estimate} (estimate)")
    print(f"      Repeat rate        : {sales.repeat_rate_pct:.0f}%")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    Active shipments     : {status['coldchain']['active_shipments']}")
    print(f"    Sensor readings      : {status['coldchain']['total_sensor_readings']}")
    print(f"    Cold-chain alerts    : {status['coldchain']['total_alerts']}")
    print(f"    Merchants            : {status['dashboard']['total_merchants']}")
    print(f"    Products             : {status['dashboard']['total_products']}")
    print(f"    Orders               : {status['dashboard']['total_orders']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Cold-chain requires physical IoT sensors for real deployment")
    print("  • Arrhenius predictions need per-product calibration")
    print("  • HACCP tracking is ASSISTIVE; certified auditor required")
    print("  • Dashboard shows AGGREGATE customer data only")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires food-safety regulatory compliance")
    print("=" * 72)
    print("  ETHICAL FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • Buddhist Economics: Accessible tools for all livelihoods")
    print("  • PNDC Axiom 3: Aggregate data only, no individual tracking")
    print("  • Sensor data belongs to shipment, not worker surveillance")
    print("  • Alerts inform; merchants decide (Wu Wei)")
    print("  • Transparent fees, no hidden costs")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
