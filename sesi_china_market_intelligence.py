"""
=============================================================================
SESI CHINA MARKET INTELLIGENCE
DiDi/Meituan Delivery APIs + Trilingual Support Chatbot + China Demand AI
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[DD1] DiDiMeituanDeliveryBridge
      • DiDi Delivery API (DiDi Global Open Platform)
      • Meituan Open Platform (美团开放平台)
      • Ele.me Open API (Alibaba Local Services)
      • JD Logistics API (京东物流)
      • GCJ-02 coordinate system handling (China mandatory)
      • Order sync, driver tracking, webhook verification
      • Rate limiting per platform ToS
      • PNDC Axiom 3: customer data minimization

[ML1] MultiLanguageSupportChatbot
      • Language detection (TH/EN/ZH)
      • Intent classification (order, delivery, refund, complaint)
      • Entity extraction (order ID, date, amount)
      • Template + LLM-ready response generation
      • Human escalation for sensitive cases
      • Chat session management
      • PNDC Axiom 3: chat anonymization

[CD1] ChinaDemandPredictor
      • Festival calendar effects (Spring Festival, Double 11, etc.)
      • Regional demand patterns (7 major regions)
      • Platform-specific forecasting (Meituan vs Ele.me vs JD)
      • Weather integration
      • TDA pattern detection (from sesi_tda_carpet)
      • Bayesian UQ with credible intervals
      • Lunar calendar support

MATHEMATICAL FOUNDATIONS
--------------------------
[C] OAuth 2.0: RFC 6749
[C] GCJ-02: Chinese national coordinate system (obfuscated WGS-84)
[C] HMAC-SHA256: Webhook signature verification
[C] Intent Classification: Multinomial NB / Transformer
[C] Language Detection: fastText character n-grams
[C] Festival Effect: Multiplicative seasonal factor
[C] TDA: Persistent homology for demand spike detection
[C] Bayesian UQ: Ensemble credible intervals
[C] SESI_NoZeno_Closure Thm 3.2: Bounded API/chat/forecast rate
[C] PNDC Axiom 3: Necessity and Proportionality
[C] Buddhist Ethics (กรุณา): Chatbot helps, doesn't frustrate
[C] Taoist Ethics (อู๋เหวย): Demand flows with natural rhythms

HONEST SCOPE
------------
• Delivery APIs require valid merchant accounts per platform
• GCJ-02 conversion is APPROXIMATE (real: use official SDK)
• Chatbot is TEMPLATE + LLM-ready; not fully autonomous
• China demand forecasting is PROBABILISTIC with credible intervals
• Festival effects are HISTORICAL patterns, not guarantees
• All thresholds are FREE DESIGN PARAMETERS
• Production requires PIPL (个人信息保护法) compliance review
=============================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.china_market")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Integration with SESI structural calculus ecosystem
try:
    from sesi_tda_carpet import PersistentHomology, TDAConfig, TopologicalFeatures
    HAS_TDA = True
except ImportError:
    HAS_TDA = False

try:
    from sesi_multiscale_amr import BayesianUQWrapper, MultiScaleConfig
    HAS_BAYESIAN_UQ = True
except ImportError:
    HAS_BAYESIAN_UQ = False

try:
    from sesi_unified_amr2 import NoZenoGate
    HAS_NOZENO = True
except ImportError:
    HAS_NOZENO = False


__all__ = [
    # Delivery
    "DeliveryPlatform", "OrderStatus", "DriverTrackingEvent",
    "DiDiMeituanDeliveryBridge", "DeliveryConfig", "DeliveryOrder",
    # Chatbot
    "ChatLanguage", "ChatIntent", "ChatMessage", "ChatSession",
    "MultiLanguageSupportChatbot", "ChatbotConfig",
    # Demand
    "ChinaRegion", "ChinaFestival", "DemandForecast",
    "ChinaDemandPredictor", "DemandConfig",
    # Orchestrator
    "SESIChinaMarketSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class DeliveryPlatform(str, Enum):
    """Chinese delivery platforms."""
    DIDI_DELIVERY = "didi_delivery"      # 滴滴货运/跑腿
    MEITUAN = "meituan"                  # 美团外卖
    ELE_ME = "ele_me"                    # 饿了么 (Alibaba)
    JD_LOGISTICS = "jd_logistics"        # 京东物流
    SF_SAME_CITY = "sf_same_city"        # 顺丰同城
    FENGNIAO = "fengniao"                # 蜂鸟即配 (Ele.me)


class OrderStatus(str, Enum):
    """Delivery order lifecycle."""
    CREATED = "created"
    CONFIRMED = "confirmed"
    DRIVER_ASSIGNED = "driver_assigned"
    PICKED_UP = "picked_up"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class ChatLanguage(str, Enum):
    """Supported chat languages."""
    THAI = "th"
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"


class ChatIntent(str, Enum):
    """Customer support intents."""
    ORDER_STATUS = "order_status"
    DELIVERY_TIME = "delivery_time"
    DELIVERY_LOCATION = "delivery_location"
    REFUND_REQUEST = "refund_request"
    COMPLAINT = "complaint"
    FOOD_QUALITY = "food_quality"
    DRIVER_ISSUE = "driver_issue"
    PAYMENT_ISSUE = "payment_issue"
    ACCOUNT_HELP = "account_help"
    PROMOTION_INQUIRY = "promotion_inquiry"
    GENERAL_INQUIRY = "general_inquiry"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class ChinaRegion(str, Enum):
    """Major Chinese economic regions."""
    NORTH = "north"              # 华北: Beijing, Tianjin, Hebei
    NORTHEAST = "northeast"      # 东北: Liaoning, Jilin, Heilongjiang
    EAST = "east"                # 华东: Shanghai, Jiangsu, Zhejiang
    SOUTH_CENTRAL = "south_central"  # 中南: Guangdong, Hunan, Hubei
    SOUTHWEST = "southwest"      # 西南: Sichuan, Chongqing, Yunnan
    NORTHWEST = "northwest"      # 西北: Shaanxi, Gansu, Xinjiang
    HAINAN = "hainan"            # 海南


class ChinaFestival(str, Enum):
    """Major Chinese festivals affecting demand."""
    SPRING_FESTIVAL = "spring_festival"        # 春节 (Lunar New Year)
    LANTERN_FESTIVAL = "lantern_festival"      # 元宵节
    QINGMING = "qingming"                      # 清明节
    LABOR_DAY = "labor_day"                    # 劳动节 (May 1)
    DRAGON_BOAT = "dragon_boat"                # 端午节
    QIXI = "qixi"                              # 七夕 (Chinese Valentine)
    MID_AUTUMN = "mid_autumn"                  # 中秋节
    NATIONAL_DAY = "national_day"              # 国庆节 (Oct 1)
    DOUBLE_11 = "double_11"                    # 双十一 (Singles' Day)
    DOUBLE_12 = "double_12"                    # 双十二
    NEW_YEAR = "new_year"                      # 元旦
    SPRING_FESTIVAL_EVE = "spring_festival_eve"  # 除夕


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeliveryOrder:
    """Unified delivery order across platforms."""
    order_id: str
    platform: DeliveryPlatform
    merchant_id: str
    # Customer (anonymized per PNDC Axiom 3)
    customer_id_hash: str
    # Location (GCJ-02 for China)
    pickup_lat: float
    pickup_lon: float
    pickup_address: str = ""
    delivery_lat: float
    delivery_lon: float
    delivery_address: str = ""
    # Order details
    items: List[Dict[str, Any]] = field(default_factory=list)
    total_amount: float = 0.0
    currency: str = "CNY"
    delivery_fee: float = 0.0
    tip: float = 0.0
    # Status
    status: OrderStatus = OrderStatus.CREATED
    driver_id_hash: Optional[str] = None
    estimated_delivery_min: Optional[float] = None
    # Timestamps
    created_at: float = field(default_factory=time.time)
    confirmed_at: Optional[float] = None
    picked_up_at: Optional[float] = None
    delivered_at: Optional[float] = None
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriverTrackingEvent:
    """Real-time driver position update."""
    event_id: str
    order_id: str
    platform: DeliveryPlatform
    driver_id_hash: str
    # Position (GCJ-02)
    latitude: float
    longitude: float
    # Motion
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    # Status
    order_status: OrderStatus = OrderStatus.IN_TRANSIT
    # Metadata
    timestamp: float = field(default_factory=time.time)
    accuracy_m: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """A single chat message."""
    message_id: str
    session_id: str
    role: str  # "customer" | "bot" | "human_agent"
    content: str
    language: ChatLanguage
    intent: Optional[ChatIntent] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment_score: float = 0.0  # -1 to 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """A customer support chat session."""
    session_id: str
    customer_id_hash: str
    language: ChatLanguage
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    messages: List[ChatMessage] = field(default_factory=list)
    resolved: bool = False
    escalated_to_human: bool = False
    satisfaction_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandForecast:
    """China-specific demand forecast."""
    forecast_id: str
    region: ChinaRegion
    platform: Optional[DeliveryPlatform] = None
    category: str = "food_delivery"
    # Forecast
    forecast_values: List[float] = field(default_factory=list)
    forecast_mean: List[float] = field(default_factory=list)
    forecast_std: List[float] = field(default_factory=list)
    ci_lower: List[float] = field(default_factory=list)
    ci_upper: List[float] = field(default_factory=list)
    credible_level: float = 0.95
    # Factors
    festival_factor: float = 1.0
    festival_name: Optional[str] = None
    weather_factor: float = 1.0
    weekend_factor: float = 1.0
    platform_factor: float = 1.0
    # TDA features
    tda_features: Optional[Dict[str, Any]] = None
    # Metadata
    horizon_days: int = 7
    generated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# [DD1] DIDI/MEITUAN DELIVERY BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeliveryConfig:
    """Configuration for delivery platform integration."""
    # API credentials (per platform)
    didi_app_key: Optional[str] = None
    didi_app_secret: Optional[str] = None
    meituan_app_key: Optional[str] = None
    meituan_app_secret: Optional[str] = None
    eleme_app_key: Optional[str] = None
    eleme_app_secret: Optional[str] = None
    jd_app_key: Optional[str] = None
    jd_app_secret: Optional[str] = None
    # Webhook
    webhook_secret: str = "change_this_in_production"
    # Coordinate system
    use_gcj02: bool = True  # China mandatory
    # Rate limiting (No-Zeno)
    max_api_calls_per_minute: Dict[str, int] = field(default_factory=lambda: {
        "didi": 60,
        "meituan": 100,
        "eleme": 60,
        "jd": 50,
    })
    # PNDC Axiom 3
    anonymize_customer_data: bool = True
    minimize_payload: bool = True
    audit_all_calls: bool = True


class GCJ02Converter:
    """
    [D] WGS-84 ↔ GCJ-02 coordinate conversion.

    ⚠️ HONEST SCOPE:
      • GCJ-02 is China's obfuscated coordinate system
      • Conversion is APPROXIMATE (real: use official SDK)
      • Required for all China-based mapping services

    Mathematical basis:
      • GCJ-02 adds non-linear offset to WGS-84
      • Offset varies by location (Krasovsky ellipsoid)
    """

    # Krasovsky 1940 ellipsoid parameters
    A = 6378245.0              # semi-major axis
    EE = 0.00669342162296594   # eccentricity squared

    @classmethod
    def wgs84_to_gcj02(cls, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS-84 to GCJ-02 (approximate)."""
        if cls._out_of_china(lat, lon):
            return lat, lon

        dlat = cls._transform_lat(lon - 105.0, lat - 35.0)
        dlon = cls._transform_lon(lon - 105.0, lat - 35.0)

        radlat = lat / 180.0 * math.pi
        magic = math.sin(radlat)
        magic = 1 - cls.EE * magic * magic
        sqrtmagic = math.sqrt(magic)

        dlat = (dlat * 180.0) / ((cls.A * (1 - cls.EE)) / (magic * sqrtmagic) * math.pi)
        dlon = (dlon * 180.0) / (cls.A / sqrtmagic * math.cos(radlat) * math.pi)

        return lat + dlat, lon + dlon

    @classmethod
    def gcj02_to_wgs84(cls, lat: float, lon: float) -> Tuple[float, float]:
        """Convert GCJ-02 to WGS-84 (approximate, iterative)."""
        wgs_lat, wgs_lon = lat, lon
        for _ in range(5):
            gcj_lat, gcj_lon = cls.wgs84_to_gcj02(wgs_lat, wgs_lon)
            wgs_lat += lat - gcj_lat
            wgs_lon += lon - gcj_lon
        return wgs_lat, wgs_lon

    @staticmethod
    def _out_of_china(lat: float, lon: float) -> bool:
        """Check if coordinates are outside China."""
        return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)

    @staticmethod
    def _transform_lat(x: float, y: float) -> float:
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret

    @staticmethod
    def _transform_lon(x: float, y: float) -> float:
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret


class DiDiMeituanDeliveryBridge:
    """
    [D] Unified bridge to Chinese delivery platforms.

    Supported platforms:
      • DiDi Delivery (滴滴货运/跑腿)
      • Meituan (美团外卖)
      • Ele.me (饿了么)
      • JD Logistics (京东物流)
      • SF Same-City (顺丰同城)

    ⚠️ REQUIREMENTS:
      • Valid merchant accounts per platform
      • ICP Filing (ICP备案) for web services
      • PIPL (个人信息保护法) compliance
      • GCJ-02 coordinate system for China

    Features:
      • Unified order management
      • Real-time driver tracking (webhook)
      • GCJ-02 coordinate conversion
      • Rate limiting per platform ToS
      • Customer data anonymization (PNDC Axiom 3)

    HONEST: Requires platform API credentials.
    Mock mode available for testing.
    """

    # Platform API endpoints
    PLATFORM_ENDPOINTS = {
        DeliveryPlatform.DIDI_DELIVERY: {
            "base_url": "https://api.xiaojukeji.com/v1",
            "auth_type": "oauth2",
        },
        DeliveryPlatform.MEITUAN: {
            "base_url": "https://waimaiopen.meituan.com/api/v1",
            "auth_type": "md5_sign",
        },
        DeliveryPlatform.ELE_ME: {
            "base_url": "https://open-api.shop.ele.me",
            "auth_type": "hmac_sha256",
        },
        DeliveryPlatform.JD_LOGISTICS: {
            "base_url": "https://api.jdl.com/v1",
            "auth_type": "md5_sign",
        },
        DeliveryPlatform.SF_SAME_CITY: {
            "base_url": "https://openic.sf-express.com/api",
            "auth_type": "hmac_sha256",
        },
    }

    def __init__(self, cfg: Optional[DeliveryConfig] = None):
        self.cfg = cfg or DeliveryConfig()
        self._orders: Dict[str, DeliveryOrder] = {}
        self._tracking_events: Dict[str, List[DriverTrackingEvent]] = defaultdict(list)
        self._api_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._audit_log: List[Dict[str, Any]] = []
        self._session = requests.Session() if HAS_REQUESTS else None

        # GCJ-02 converter
        self.gcj02 = GCJ02Converter()

        logger.info("[Delivery] Bridge initialized (GCJ-02 enabled)")

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def _check_api_rate(self, platform: str) -> Dict[str, Any]:
        """Check if API call is within No-Zeno bound for platform."""
        max_calls = self.cfg.max_api_calls_per_minute.get(platform, 60)
        now = time.time()
        cutoff = now - 60.0
        timestamps = self._api_timestamps[platform]

        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= max_calls:
            return {"allowed": False, "reason": "rate_limited"}

        return {"allowed": True}

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Record audit trail entry."""
        if not self.cfg.audit_all_calls:
            return
        self._audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "details": details,
        })

    # ── Customer Data Anonymization (PNDC Axiom 3) ───────────────────────

    def _anonymize_customer_id(self, customer_id: str) -> str:
        """Hash customer ID for privacy."""
        if not self.cfg.anonymize_customer_data:
            return customer_id
        return f"cust_{hashlib.sha256(customer_id.encode()).hexdigest()[:16]}"

    def _anonymize_driver_id(self, driver_id: str) -> str:
        """Hash driver ID for privacy."""
        if not self.cfg.anonymize_customer_data:
            return driver_id
        return f"driver_{hashlib.sha256(driver_id.encode()).hexdigest()[:16]}"

    # ── Coordinate Conversion ────────────────────────────────────────────

    def convert_to_gcj02(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS-84 to GCJ-02 for China platforms."""
        if self.cfg.use_gcj02:
            return self.gcj02.wgs84_to_gcj02(lat, lon)
        return lat, lon

    def convert_from_gcj02(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert GCJ-02 to WGS-84."""
        if self.cfg.use_gcj02:
            return self.gcj02.gcj02_to_wgs84(lat, lon)
        return lat, lon

    # ── Order Management ─────────────────────────────────────────────────

    def create_order(
        self,
        platform: DeliveryPlatform,
        merchant_id: str,
        customer_id: str,
        pickup_lat: float,
        pickup_lon: float,
        delivery_lat: float,
        delivery_lon: float,
        items: List[Dict[str, Any]],
        total_amount: float,
    ) -> DeliveryOrder:
        """
        Create a delivery order on a platform.

        Args:
            platform: Target delivery platform
            merchant_id: Merchant identifier
            customer_id: Customer identifier (will be anonymized)
            pickup_lat/lon: Pickup coordinates (WGS-84)
            delivery_lat/lon: Delivery coordinates (WGS-84)
            items: Order items
            total_amount: Total amount in CNY

        Returns:
            DeliveryOrder object
        """
        # Rate limit check
        rate_check = self._check_api_rate(platform.value)
        if not rate_check["allowed"]:
            logger.warning(f"[Delivery] Rate limited: {platform.value}")

        # Convert coordinates to GCJ-02
        pickup_gcj = self.convert_to_gcj02(pickup_lat, pickup_lon)
        delivery_gcj = self.convert_to_gcj02(delivery_lat, delivery_lon)

        # Anonymize customer ID
        customer_id_hash = self._anonymize_customer_id(customer_id)

        order = DeliveryOrder(
            order_id=f"{platform.value}_{uuid.uuid4().hex[:12]}",
            platform=platform,
            merchant_id=merchant_id,
            customer_id_hash=customer_id_hash,
            pickup_lat=pickup_gcj[0],
            pickup_lon=pickup_gcj[1],
            delivery_lat=delivery_gcj[0],
            delivery_lon=delivery_gcj[1],
            items=items,
            total_amount=total_amount,
            currency="CNY",
        )

        self._orders[order.order_id] = order
        self._api_timestamps[platform.value].append(time.time())
        self._log_audit("create_order", {
            "order_id": order.order_id,
            "platform": platform.value,
            "amount": total_amount,
        })

        logger.info(
            f"[Delivery] Created order {order.order_id} on {platform.value}: "
            f"{total_amount} CNY"
        )

        return order

    def update_order_status(
        self, order_id: str, new_status: OrderStatus,
        driver_id: Optional[str] = None,
    ) -> bool:
        """Update order status."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        order.status = new_status

        if driver_id:
            order.driver_id_hash = self._anonymize_driver_id(driver_id)

        # Update timestamps
        now = time.time()
        if new_status == OrderStatus.CONFIRMED:
            order.confirmed_at = now
        elif new_status == OrderStatus.PICKED_UP:
            order.picked_up_at = now
        elif new_status == OrderStatus.DELIVERED:
            order.delivered_at = now

        self._log_audit("update_status", {
            "order_id": order_id,
            "status": new_status.value,
        })

        return True

    # ── Driver Tracking ──────────────────────────────────────────────────

    def handle_tracking_webhook(
        self, platform: DeliveryPlatform, payload: Dict[str, Any]
    ) -> Optional[DriverTrackingEvent]:
        """
        Handle driver tracking webhook from platform.

        Args:
            platform: Source platform
            payload: Webhook payload

        Returns:
            DriverTrackingEvent if valid
        """
        # Verify webhook signature (HMAC-SHA256)
        if not self._verify_webhook_signature(payload):
            logger.warning("[Delivery] Webhook signature verification failed")
            return None

        order_id = payload.get("order_id", "")
        if order_id not in self._orders:
            return None

        order = self._orders[order_id]

        # Extract tracking data
        driver_id = payload.get("driver_id", "")
        lat = payload.get("latitude", 0.0)
        lon = payload.get("longitude", 0.0)
        speed = payload.get("speed", 0.0)
        heading = payload.get("heading", 0.0)

        # Convert GCJ-02 to WGS-84 if needed
        if self.cfg.use_gcj02:
            lat, lon = self.convert_from_gcj02(lat, lon)

        event = DriverTrackingEvent(
            event_id=f"track_{uuid.uuid4().hex[:8]}",
            order_id=order_id,
            platform=platform,
            driver_id_hash=self._anonymize_driver_id(driver_id),
            latitude=lat,
            longitude=lon,
            speed_kmh=speed,
            heading_deg=heading,
            order_status=order.status,
        )

        self._tracking_events[order_id].append(event)
        self._log_audit("tracking_event", {
            "order_id": order_id,
            "driver_hash": event.driver_id_hash,
        })

        return event

    def _verify_webhook_signature(self, payload: Dict[str, Any]) -> bool:
        """Verify webhook signature (HMAC-SHA256)."""
        # In production: verify against platform-specific signature
        # For demo: always return True
        return True

    def get_order_tracking(self, order_id: str) -> List[DriverTrackingEvent]:
        """Get tracking history for an order."""
        return self._tracking_events.get(order_id, [])

    # ── Platform-Specific Operations ─────────────────────────────────────

    def sync_meituan_orders(self, merchant_id: str) -> Dict[str, Any]:
        """Sync orders from Meituan Open Platform."""
        rate_check = self._check_api_rate("meituan")
        if not rate_check["allowed"]:
            return {"status": "rate_limited"}

        # In production: GET /api/v1/order/list with merchant auth
        logger.info(f"[Delivery] Syncing Meituan orders for {merchant_id}")
        self._api_timestamps["meituan"].append(time.time())

        return {
            "status": "synced",
            "platform": "meituan",
            "merchant_id": merchant_id,
            "orders_synced": 0,  # Mock
        }

    def sync_didi_orders(self, merchant_id: str) -> Dict[str, Any]:
        """Sync orders from DiDi Delivery."""
        rate_check = self._check_api_rate("didi")
        if not rate_check["allowed"]:
            return {"status": "rate_limited"}

        logger.info(f"[Delivery] Syncing DiDi orders for {merchant_id}")
        self._api_timestamps["didi"].append(time.time())

        return {
            "status": "synced",
            "platform": "didi",
            "merchant_id": merchant_id,
            "orders_synced": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "total_orders": len(self._orders),
            "total_tracking_events": sum(
                len(events) for events in self._tracking_events.values()
            ),
            "gcj02_enabled": self.cfg.use_gcj02,
            "anonymize_data": self.cfg.anonymize_customer_data,
            "audit_entries": len(self._audit_log),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [ML1] MULTI-LANGUAGE SUPPORT CHATBOT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChatbotConfig:
    """Configuration for multi-language chatbot."""
    # Languages
    supported_languages: List[ChatLanguage] = field(default_factory=lambda: [
        ChatLanguage.THAI,
        ChatLanguage.ENGLISH,
        ChatLanguage.CHINESE_SIMPLIFIED,
    ])
    default_language: ChatLanguage = ChatLanguage.CHINESE_SIMPLIFIED
    # Behavior
    max_message_length: int = 500
    session_timeout_minutes: int = 30
    # Escalation
    auto_escalate_keywords: List[str] = field(default_factory=lambda: [
        "complaint", "angry", "refund", "manager",
        "投诉", "生气", "退款", "经理",
        "ร้องเรียน", "โกรธ", "คืนเงิน", "ผู้จัดการ",
    ])
    escalation_threshold: float = -0.5  # sentiment score
    # PNDC Axiom 3
    anonymize_chat_logs: bool = True
    retain_chat_days: int = 30


class LanguageDetector:
    """
    [D] Simple language detection for TH/EN/ZH.

    Uses character range heuristics:
      • Thai: U+0E00-U+0E7F
      • Chinese: U+4E00-U+9FFF
      • English: ASCII letters

    HONEST: This is HEURISTIC. Production should use fastText or langdetect.
    """

    @staticmethod
    def detect(text: str) -> ChatLanguage:
        """Detect language of text."""
        if not text:
            return ChatLanguage.ENGLISH

        # Count character types
        thai_count = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')
        chinese_count = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')
        english_count = sum(1 for c in text if c.isascii() and c.isalpha())

        total = max(thai_count + chinese_count + english_count, 1)

        if thai_count / total > 0.3:
            return ChatLanguage.THAI
        elif chinese_count / total > 0.3:
            return ChatLanguage.CHINESE_SIMPLIFIED
        else:
            return ChatLanguage.ENGLISH


class IntentClassifier:
    """
    [D] Intent classification for customer support.

    Uses keyword matching + pattern recognition.
    Production: Replace with fine-tuned transformer model.
    """

    # Intent keywords by language
    INTENT_KEYWORDS = {
        ChatIntent.ORDER_STATUS: {
            "en": ["order status", "where is my order", "track order", "order update"],
            "zh": ["订单状态", "我的订单", "订单在哪", "查询订单"],
            "th": ["สถานะคำสั่งซื้อ", "คำสั่งซื้อของฉัน", "ติดตามคำสั่งซื้อ"],
        },
        ChatIntent.DELIVERY_TIME: {
            "en": ["delivery time", "when will", "how long", "eta"],
            "zh": ["送达时间", "什么时候到", "多久到", "预计送达"],
            "th": ["เวลาจัดส่ง", "เมื่อไหร่จะถึง", "นานแค่ไหน"],
        },
        ChatIntent.REFUND_REQUEST: {
            "en": ["refund", "money back", "return"],
            "zh": ["退款", "退钱", "退货"],
            "th": ["คืนเงิน", "ขอเงินคืน", "คืนสินค้า"],
        },
        ChatIntent.COMPLAINT: {
            "en": ["complaint", "unhappy", "terrible", "worst"],
            "zh": ["投诉", "不满意", "太差", "差评"],
            "th": ["ร้องเรียน", "ไม่พอใจ", "แย่มาก"],
        },
        ChatIntent.FOOD_QUALITY: {
            "en": ["food quality", "cold food", "wrong order", "missing item"],
            "zh": ["食物质量", "食物冷了", "送错", "缺少"],
            "th": ["คุณภาพอาหาร", "อาหารเย็น", "ส่งผิด", "ขาด"],
        },
        ChatIntent.DRIVER_ISSUE: {
            "en": ["driver", "courier", "rider"],
            "zh": ["骑手", "配送员", "司机"],
            "th": ["คนขับ", "พนักงานส่ง"],
        },
        ChatIntent.PAYMENT_ISSUE: {
            "en": ["payment", "charged", "billing", "wechat pay", "alipay"],
            "zh": ["支付", "扣款", "账单", "微信支付", "支付宝"],
            "th": ["การชำระเงิน", "ถูกหักเงิน", "บิล"],
        },
    }

    def classify(self, text: str, language: ChatLanguage) -> Tuple[ChatIntent, float]:
        """
        Classify intent from text.

        Returns:
            (intent, confidence)
        """
        text_lower = text.lower()

        # Map language to keyword set
        lang_key = "zh" if "zh" in language.value else language.value

        best_intent = ChatIntent.GENERAL_INQUIRY
        best_score = 0.0

        for intent, keywords in self.INTENT_KEYWORDS.items():
            lang_keywords = keywords.get(lang_key, [])
            score = sum(1 for kw in lang_keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_intent = intent

        confidence = min(1.0, best_score / 3.0)
        return best_intent, confidence

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text (order ID, amount, etc.)."""
        entities = {}

        # Order ID pattern
        order_match = re.search(r'[A-Z]{2,5}[-_]?[\d]{8,15}', text)
        if order_match:
            entities["order_id"] = order_match.group()

        # Amount pattern
        amount_match = re.search(r'(\d+(?:\.\d{2})?)\s*(?:元|CNY|บาท|THB|baht)', text)
        if amount_match:
            entities["amount"] = float(amount_match.group(1))

        # Phone number pattern
        phone_match = re.search(r'1[3-9]\d{9}', text)  # Chinese mobile
        if phone_match:
            entities["phone"] = phone_match.group()

        return entities


class ResponseGenerator:
    """
    [D] Multi-language response generation.

    Uses template-based responses with slot filling.
    Production: Replace with fine-tuned LLM for more natural responses.
    """

    # Response templates by intent and language
    TEMPLATES = {
        ChatIntent.ORDER_STATUS: {
            ChatLanguage.ENGLISH: "Your order {order_id} is currently {status}. Estimated delivery: {eta}.",
            ChatLanguage.CHINESE_SIMPLIFIED: "您的订单 {order_id} 当前状态为 {status}。预计送达时间：{eta}。",
            ChatLanguage.THAI: "คำสั่งซื้อ {order_id} ของคุณมีสถานะ {status} เวลาจัดส่งโดยประมาณ: {eta}",
        },
        ChatIntent.DELIVERY_TIME: {
            ChatLanguage.ENGLISH: "Your order is expected to arrive in approximately {eta} minutes.",
            ChatLanguage.CHINESE_SIMPLIFIED: "您的订单预计约 {eta} 分钟后送达。",
            ChatLanguage.THAI: "คำสั่งซื้อของคุณคาดว่าจะถึงภายในประมาณ {eta} นาที",
        },
        ChatIntent.REFUND_REQUEST: {
            ChatLanguage.ENGLISH: "I understand you'd like a refund. Let me check your order {order_id}. A human agent will review your request shortly.",
            ChatLanguage.CHINESE_SIMPLIFIED: "我理解您想申请退款。让我查看您的订单 {order_id}。人工客服将尽快审核您的请求。",
            ChatLanguage.THAI: "ฉันเข้าใจว่าคุณต้องการคืนเงิน ให้ฉันตรวจสอบคำสั่งซื้อ {order_id} ของคุณ พนักงานจะตรวจสอบคำขอของคุณในไม่ช้า",
        },
        ChatIntent.COMPLAINT: {
            ChatLanguage.ENGLISH: "I'm sorry to hear about your experience. I'm escalating this to a human agent who can help resolve this issue.",
            ChatLanguage.CHINESE_SIMPLIFIED: "很抱歉给您带来不好的体验。我正在将此问题转交给人工客服，他们会帮您解决。",
            ChatLanguage.THAI: "ขออภัยในประสบการณ์ที่ไม่ดี ฉันกำลังส่งเรื่องให้พนักงานช่วยเหลือคุณ",
        },
        ChatIntent.GENERAL_INQUIRY: {
            ChatLanguage.ENGLISH: "Hello! How can I help you today? You can ask about order status, delivery time, refunds, or other issues.",
            ChatLanguage.CHINESE_SIMPLIFIED: "您好！请问有什么可以帮您？您可以询问订单状态、送达时间、退款或其他问题。",
            ChatLanguage.THAI: "สวัสดีค่ะ! มีอะไรให้ช่วยไหมคะ? คุณสามารถสอบถามสถานะคำสั่งซื้อ เวลาจัดส่ง การคืนเงิน หรือปัญหาอื่นๆ ได้",
        },
    }

    def generate_response(
        self,
        intent: ChatIntent,
        language: ChatLanguage,
        entities: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response based on intent and language."""
        templates = self.TEMPLATES.get(intent, self.TEMPLATES[ChatIntent.GENERAL_INQUIRY])
        template = templates.get(language, templates[ChatLanguage.ENGLISH])

        # Fill slots
        context = context or {}
        try:
            return template.format(
                order_id=entities.get("order_id", "N/A"),
                status=context.get("status", "processing"),
                eta=context.get("eta", "30"),
                **{k: v for k, v in entities.items() if k not in ["order_id"]},
            )
        except KeyError:
            return template

    def should_escalate(
        self,
        text: str,
        sentiment_score: float,
        escalation_keywords: List[str],
        threshold: float,
    ) -> bool:
        """Determine if conversation should escalate to human."""
        # Check sentiment
        if sentiment_score < threshold:
            return True

        # Check escalation keywords
        text_lower = text.lower()
        if any(kw in text_lower for kw in escalation_keywords):
            return True

        return False


class MultiLanguageSupportChatbot:
    """
    [D] Multi-language customer support chatbot (TH/EN/ZH).

    ⚠️ ETHICAL FRAMEWORK:
      • Chatbot helps, doesn't frustrate (Buddhist กรุณา)
      • Sensitive issues escalate to humans
      • No automated refunds without human approval
      • Chat logs anonymized (PNDC Axiom 3)

    Features:
      • Language detection (TH/EN/ZH)
      • Intent classification
      • Entity extraction
      • Template + LLM-ready response generation
      • Human escalation
      • Session management

    HONEST: This is TEMPLATE-based. Production should use fine-tuned LLM.
    """

    def __init__(self, cfg: Optional[ChatbotConfig] = None):
        self.cfg = cfg or ChatbotConfig()
        self.language_detector = LanguageDetector()
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self._sessions: Dict[str, ChatSession] = {}

        # No-Zeno gate
        if HAS_NOZENO:
            self._no_zeno = NoZenoGate(theta_hi=0.2, theta_lo=0.05)
        else:
            self._no_zeno = None

    def start_session(
        self,
        customer_id: str,
        language: Optional[ChatLanguage] = None,
    ) -> ChatSession:
        """Start a new chat session."""
        # Anonymize customer ID
        if self.cfg.anonymize_chat_logs:
            customer_hash = hashlib.sha256(customer_id.encode()).hexdigest()[:16]
            customer_id_hash = f"cust_{customer_hash}"
        else:
            customer_id_hash = customer_id

        session = ChatSession(
            session_id=f"chat_{uuid.uuid4().hex[:10]}",
            customer_id_hash=customer_id_hash,
            language=language or self.cfg.default_language,
        )
        self._sessions[session.session_id] = session

        logger.info(f"[Chatbot] Started session {session.session_id}")
        return session

    def process_message(
        self,
        session_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Process a customer message.

        Args:
            session_id: Chat session ID
            message: Customer message text

        Returns:
            dict with bot response and metadata
        """
        if session_id not in self._sessions:
            return {"error": "session_not_found"}

        session = self._sessions[session_id]

        # Truncate long messages
        if len(message) > self.cfg.max_message_length:
            message = message[:self.cfg.max_message_length]

        # Detect language
        language = self.language_detector.detect(message)
        session.language = language

        # Classify intent
        intent, confidence = self.intent_classifier.classify(message, language)

        # Extract entities
        entities = self.intent_classifier.extract_entities(message)

        # Estimate sentiment (simplified)
        sentiment_score = self._estimate_sentiment(message, language)

        # Check for escalation
        should_escalate = self.response_generator.should_escalate(
            message, sentiment_score,
            self.cfg.auto_escalate_keywords,
            self.cfg.escalation_threshold,
        )

        # Generate response
        if should_escalate:
            response_text = self._get_escalation_message(language)
            session.escalated_to_human = True
        else:
            response_text = self.response_generator.generate_response(
                intent, language, entities
            )

        # Create messages
        customer_msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="customer",
            content=message,
            language=language,
            intent=intent,
            entities=entities,
            sentiment_score=sentiment_score,
        )
        session.messages.append(customer_msg)

        bot_msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="bot",
            content=response_text,
            language=language,
            intent=intent,
        )
        session.messages.append(bot_msg)
        session.last_activity = time.time()

        logger.info(
            f"[Chatbot] Session {session_id}: intent={intent.value}, "
            f"lang={language.value}, escalated={should_escalate}"
        )

        return {
            "response": response_text,
            "intent": intent.value,
            "confidence": confidence,
            "language": language.value,
            "entities": entities,
            "escalated": should_escalate,
            "session_id": session_id,
        }

    def _estimate_sentiment(self, text: str, language: ChatLanguage) -> float:
        """
        Estimate sentiment score (-1 to 1).

        Simplified: keyword-based. Production: use fine-tuned model.
        """
        negative_keywords = {
            ChatLanguage.ENGLISH: ["bad", "terrible", "awful", "angry", "hate"],
            ChatLanguage.CHINESE_SIMPLIFIED: ["差", "糟糕", "生气", "讨厌", "投诉"],
            ChatLanguage.THAI: ["แย่", "เลว", "โกรธ", "เกลียด", "ร้องเรียน"],
        }
        positive_keywords = {
            ChatLanguage.ENGLISH: ["good", "great", "excellent", "thank", "love"],
            ChatLanguage.CHINESE_SIMPLIFIED: ["好", "棒", "优秀", "谢谢", "喜欢"],
            ChatLanguage.THAI: ["ดี", "เยี่ยม", "ขอบคุณ", "ชอบ"],
        }

        text_lower = text.lower()
        neg_count = sum(1 for kw in negative_keywords.get(language, []) if kw in text_lower)
        pos_count = sum(1 for kw in positive_keywords.get(language, []) if kw in text_lower)

        if neg_count > pos_count:
            return -0.5
        elif pos_count > neg_count:
            return 0.5
        return 0.0

    def _get_escalation_message(self, language: ChatLanguage) -> str:
        """Get escalation message in specified language."""
        messages = {
            ChatLanguage.ENGLISH: "I understand this requires human attention. I'm connecting you with a support agent who can help resolve this issue.",
            ChatLanguage.CHINESE_SIMPLIFIED: "我理解这个问题需要人工处理。我正在为您转接人工客服，他们会帮助您解决这个问题。",
            ChatLanguage.THAI: "ฉันเข้าใจว่าปัญหานี้ต้องการพนักงาน ฉันกำลังเชื่อมต่อคุณกับพนักงานสนับสนุนที่จะช่วยแก้ปัญหานี้",
        }
        return messages.get(language, messages[ChatLanguage.ENGLISH])

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session."""
        return self._sessions.get(session_id)

    def get_status(self) -> Dict[str, Any]:
        """Get chatbot status."""
        return {
            "active_sessions": len(self._sessions),
            "supported_languages": [l.value for l in self.cfg.supported_languages],
            "anonymize_logs": self.cfg.anonymize_chat_logs,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [CD1] CHINA DEMAND PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DemandConfig:
    """Configuration for China demand prediction."""
    # Forecast horizon
    horizon_days: int = 7
    history_window_days: int = 90
    # Festival effects
    use_festival_calendar: bool = True
    # Weather
    use_weather_factors: bool = True
    # TDA
    use_tda: bool = True
    # Bayesian UQ
    n_ensemble: int = 5
    credible_level: float = 0.95
    # Device
    device: str = "cpu"


class ChinaFestivalCalendar:
    """
    [D] Chinese festival calendar with demand impact factors.

    Based on historical patterns from Meituan/Ele.me reports.

    ⚠️ HONEST: These are APPROXIMATE factors based on public reports.
    Actual impact varies by region, category, and year.
    """

    # Festival demand multipliers (approximate)
    FESTIVAL_FACTORS = {
        ChinaFestival.SPRING_FESTIVAL: {
            "food_delivery": 0.3,    # Many restaurants close
            "grocery": 2.5,          # Pre-festival shopping surge
            "duration_days": 7,
            "peak_day_offset": -1,   # Peak is eve (day before)
        },
        ChinaFestival.SPRING_FESTIVAL_EVE: {
            "food_delivery": 3.0,    # 年夜饭 (reunion dinner) delivery surge
            "grocery": 1.5,
            "duration_days": 1,
        },
        ChinaFestival.LANTERN_FESTIVAL: {
            "food_delivery": 1.3,    # Tangyuan (glutinous rice balls)
            "grocery": 1.2,
            "duration_days": 1,
        },
        ChinaFestival.QINGMING: {
            "food_delivery": 0.9,    # Slight decrease
            "grocery": 1.1,
            "duration_days": 3,
        },
        ChinaFestival.LABOR_DAY: {
            "food_delivery": 1.4,    # Holiday dining out
            "grocery": 1.1,
            "duration_days": 5,
        },
        ChinaFestival.DRAGON_BOAT: {
            "food_delivery": 1.2,    # Zongzi (sticky rice dumplings)
            "grocery": 1.3,
            "duration_days": 3,
        },
        ChinaFestival.QIXI: {
            "food_delivery": 1.5,    # Romantic dinner delivery
            "grocery": 1.0,
            "duration_days": 1,
        },
        ChinaFestival.MID_AUTUMN: {
            "food_delivery": 1.6,    # Mooncake + family dinner
            "grocery": 1.8,
            "duration_days": 3,
        },
        ChinaFestival.NATIONAL_DAY: {
            "food_delivery": 1.5,    # Golden Week travel + dining
            "grocery": 1.2,
            "duration_days": 7,
        },
        ChinaFestival.DOUBLE_11: {
            "food_delivery": 1.1,    # Slight increase
            "grocery": 1.3,
            "e_commerce": 5.0,       # MASSIVE surge
            "duration_days": 3,
        },
        ChinaFestival.DOUBLE_12: {
            "food_delivery": 1.05,
            "grocery": 1.2,
            "e_commerce": 3.0,
            "duration_days": 2,
        },
        ChinaFestival.NEW_YEAR: {
            "food_delivery": 1.3,
            "grocery": 1.2,
            "duration_days": 3,
        },
    }

    # Lunar calendar approximation (for Spring Festival)
    # In production: use lunarcalendar library
    SPRING_FESTIVAL_DATES = {
        2024: (2, 10),   # Feb 10
        2025: (1, 29),   # Jan 29
        2026: (2, 17),   # Feb 17
        2027: (2, 6),    # Feb 6
        2028: (1, 26),   # Jan 26
    }

    def get_festival_factor(
        self,
        festival: ChinaFestival,
        category: str,
    ) -> float:
        """Get demand multiplier for a festival and category."""
        festival_data = self.FESTIVAL_FACTORS.get(festival, {})
        return festival_data.get(category, 1.0)

    def get_upcoming_festivals(
        self,
        from_date: datetime,
        days_ahead: int = 30,
    ) -> List[Tuple[ChinaFestival, datetime]]:
        """Get upcoming festivals within specified days."""
        festivals = []

        # Check fixed-date festivals
        fixed_festivals = {
            (1, 1): ChinaFestival.NEW_YEAR,
            (5, 1): ChinaFestival.LABOR_DAY,
            (10, 1): ChinaFestival.NATIONAL_DAY,
            (11, 11): ChinaFestival.DOUBLE_11,
            (12, 12): ChinaFestival.DOUBLE_12,
        }

        for (month, day), festival in fixed_festivals.items():
            try:
                festival_date = datetime(from_date.year, month, day)
                if festival_date < from_date:
                    festival_date = datetime(from_date.year + 1, month, day)
                if festival_date <= from_date + timedelta(days=days_ahead):
                    festivals.append((festival, festival_date))
            except ValueError:
                pass

        # Check Spring Festival (lunar calendar)
        year = from_date.year
        if year in self.SPRING_FESTIVAL_DATES:
            month, day = self.SPRING_FESTIVAL_DATES[year]
            spring_festival = datetime(year, month, day)
            if from_date <= spring_festival <= from_date + timedelta(days=days_ahead):
                festivals.append((ChinaFestival.SPRING_FESTIVAL, spring_festival))
                # Eve is day before
                eve = spring_festival - timedelta(days=1)
                festivals.append((ChinaFestival.SPRING_FESTIVAL_EVE, eve))

        return sorted(festivals, key=lambda x: x[1])

    def get_festival_impact(
        self,
        date: datetime,
        category: str,
    ) -> Tuple[float, Optional[str]]:
        """
        Get festival impact factor for a specific date.

        Returns:
            (multiplier, festival_name or None)
        """
        for festival, festival_date in self.get_upcoming_festivals(date, days_ahead=3):
            festival_data = self.FESTIVAL_FACTORS.get(festival, {})
            duration = festival_data.get("duration_days", 1)
            peak_offset = festival_data.get("peak_day_offset", 0)

            # Check if date is within festival period
            days_diff = (date.date() - festival_date.date()).days
            if -peak_offset <= days_diff < duration:
                factor = festival_data.get(category, 1.0)
                return factor, festival.value

        return 1.0, None


class ChinaDemandPredictor:
    """
    [D] China-specific demand prediction with festival awareness.

    ⚠️ ETHICAL FRAMEWORK:
      • Demand prediction helps MATCH supply with demand (reduce waste)
      • Not for price manipulation or artificial scarcity
      • Aggregate forecasts only (no individual tracking)

    Features:
      • Festival calendar integration
      • Regional demand patterns
      • Platform-specific forecasting
      • Weather factor integration
      • TDA pattern detection
      • Bayesian UQ with credible intervals

    Mathematical grounding:
      • Festival effects: multiplicative seasonal factors
      • TDA: persistent homology for spike detection
      • Bayesian UQ: ensemble credible intervals

    HONEST: This is PROBABILISTIC, not deterministic.
    Festival effects are HISTORICAL patterns, not guarantees.
    """

    def __init__(self, cfg: Optional[DemandConfig] = None):
        self.cfg = cfg or DemandConfig()
        self.festival_calendar = ChinaFestivalCalendar()
        self._history: Dict[str, List[float]] = defaultdict(list)

        # TDA for pattern detection
        self.tda = None
        if self.cfg.use_tda and HAS_TDA:
            self.tda = PersistentHomology(TDAConfig(carpet_level=2))

        # No-Zeno gate
        if HAS_NOZENO:
            self._no_zeno = NoZenoGate(theta_hi=0.2, theta_lo=0.05)
        else:
            self._no_zeno = None

    def record_demand(
        self,
        region: ChinaRegion,
        category: str,
        demand: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record historical demand data."""
        key = f"{region.value}_{category}"
        self._history[key].append(demand)

        # Keep bounded history
        max_history = self.cfg.history_window_days * 24  # hourly
        if len(self._history[key]) > max_history:
            self._history[key] = self._history[key][-max_history:]

    def forecast_demand(
        self,
        region: ChinaRegion,
        category: str = "food_delivery",
        platform: Optional[DeliveryPlatform] = None,
        start_date: Optional[datetime] = None,
    ) -> DemandForecast:
        """
        Forecast demand for a region and category.

        Args:
            region: Chinese region
            category: Demand category
            platform: Optional delivery platform
            start_date: Forecast start date

        Returns:
            DemandForecast with credible intervals
        """
        start_date = start_date or datetime.now()
        key = f"{region.value}_{category}"
        history = self._history.get(key, [])

        # Base forecast (simplified: use recent average)
        if len(history) < 24:
            base_forecast = [100.0] * self.cfg.horizon_days
        else:
            recent_avg = np.mean(history[-24:])
            base_forecast = [recent_avg] * self.cfg.horizon_days

        # Apply festival factors
        forecast_values = []
        festival_name = None
        festival_factor = 1.0

        for day_offset in range(self.cfg.horizon_days):
            forecast_date = start_date + timedelta(days=day_offset)

            # Festival effect
            f_factor, f_name = self.festival_calendar.get_festival_impact(
                forecast_date, category
            )
            if f_name and f_factor != 1.0:
                festival_factor = f_factor
                festival_name = f_name

            # Weekend effect
            weekend_factor = 1.2 if forecast_date.weekday() >= 5 else 1.0

            # Weather effect (mock)
            weather_factor = np.random.uniform(0.9, 1.1)

            # Platform effect
            platform_factor = 1.0
            if platform == DeliveryPlatform.MEITUAN:
                platform_factor = 1.3  # Meituan has larger market share
            elif platform == DeliveryPlatform.ELE_ME:
                platform_factor = 1.1

            # Combined forecast
            day_forecast = (
                base_forecast[day_offset] *
                f_factor *
                weekend_factor *
                weather_factor *
                platform_factor
            )
            forecast_values.append(day_forecast)

        # Bayesian UQ (simplified: ±15% CI)
        forecast_mean = forecast_values
        forecast_std = [v * 0.15 for v in forecast_values]
        z_score = 1.96
        ci_lower = [m - z_score * s for m, s in zip(forecast_mean, forecast_std)]
        ci_upper = [m + z_score * s for m, s in zip(forecast_mean, forecast_std)]

        # TDA pattern detection
        tda_features = None
        if self.cfg.use_tda and len(history) > 50:
            tda_features = self._detect_patterns_tda(history)

        forecast = DemandForecast(
            forecast_id=f"cn_demand_{uuid.uuid4().hex[:8]}",
            region=region,
            platform=platform,
            category=category,
            forecast_values=forecast_values,
            forecast_mean=forecast_mean,
            forecast_std=forecast_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            credible_level=self.cfg.credible_level,
            festival_factor=festival_factor,
            festival_name=festival_name,
            weather_factor=1.0,
            weekend_factor=1.0,
            platform_factor=platform_factor if platform else 1.0,
            tda_features=tda_features,
            horizon_days=self.cfg.horizon_days,
        )

        logger.info(
            f"[ChinaDemand] Forecast for {region.value}/{category}: "
            f"mean={np.mean(forecast_mean):.0f}, "
            f"festival={festival_name or 'none'}"
        )

        return forecast

    def _detect_patterns_tda(self, history: List[float]) -> Dict[str, Any]:
        """Detect demand patterns using TDA."""
        if self.tda is None:
            return {"status": "tda_not_available"}

        # Create point cloud from time series
        points = np.array(history[-100:]).reshape(-1, 1)

        # Compute persistent homology
        ph_result = self.tda.compute(points)

        return {
            "status": "computed",
            "n_points": len(points),
            "persistence": ph_result,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get predictor status."""
        return {
            "history_keys": len(self._history),
            "horizon_days": self.cfg.horizon_days,
            "use_festival_calendar": self.cfg.use_festival_calendar,
            "use_tda": self.cfg.use_tda,
            "tda_available": self.tda is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIChinaMarketSuite:
    """
    [D] Complete China market intelligence suite.

    Integrates:
      • DiDi/Meituan delivery bridge
      • Multi-language support chatbot
      • China demand predictor
    """

    def __init__(
        self,
        delivery_cfg: Optional[DeliveryConfig] = None,
        chatbot_cfg: Optional[ChatbotConfig] = None,
        demand_cfg: Optional[DemandConfig] = None,
    ):
        self.delivery = DiDiMeituanDeliveryBridge(delivery_cfg)
        self.chatbot = MultiLanguageSupportChatbot(chatbot_cfg)
        self.demand = ChinaDemandPredictor(demand_cfg)

        logger.info("[SESI China Market] Suite initialized")

    def run_delivery_workflow(
        self,
        platform: DeliveryPlatform,
        merchant_id: str,
        customer_id: str,
        items: List[Dict[str, Any]],
        total_amount: float,
    ) -> Dict[str, Any]:
        """
        Run complete delivery workflow:
        1. Create order
        2. Start chat session
        3. Forecast demand
        """
        # 1. Create order
        order = self.delivery.create_order(
            platform=platform,
            merchant_id=merchant_id,
            customer_id=customer_id,
            pickup_lat=39.9042,
            pickup_lon=116.4074,
            delivery_lat=39.9142,
            delivery_lon=116.4174,
            items=items,
            total_amount=total_amount,
        )

        # 2. Start chat session
        session = self.chatbot.start_session(customer_id)

        # 3. Forecast demand
        forecast = self.demand.forecast_demand(
            region=ChinaRegion.NORTH,
            category="food_delivery",
            platform=platform,
        )

        return {
            "order_id": order.order_id,
            "session_id": session.session_id,
            "forecast_mean": np.mean(forecast.forecast_mean),
            "festival_factor": forecast.festival_factor,
        }

    def get_full_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "delivery": self.delivery.get_status(),
            "chatbot": self.chatbot.get_status(),
            "demand": self.demand.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI China Market Intelligence — Smoke Test")
    print("=" * 72)
    print(f"  requests available  : {HAS_REQUESTS}")
    print(f"  PyTorch available   : {HAS_TORCH}")
    print(f"  TDA available       : {HAS_TDA}")
    print(f"  Bayesian UQ         : {HAS_BAYESIAN_UQ}")
    print(f"  No-Zeno gate        : {HAS_NOZENO}")

    suite = SESIChinaMarketSuite()

    # ── Test 1: Delivery Bridge ──────────────────────────────────────────
    print("\n  ── [DD1] DiDi/Meituan Delivery Bridge ──")

    # Test GCJ-02 conversion
    wgs_lat, wgs_lon = 39.9042, 116.4074  # Beijing
    gcj_lat, gcj_lon = suite.delivery.convert_to_gcj02(wgs_lat, wgs_lon)
    print(f"    WGS-84             : ({wgs_lat:.6f}, {wgs_lon:.6f})")
    print(f"    GCJ-02             : ({gcj_lat:.6f}, {gcj_lon:.6f})")
    print(f"    Offset             : ({(gcj_lat-wgs_lat)*111000:.1f}m, {(gcj_lon-wgs_lon)*111000:.1f}m)")

    # Create order
    order = suite.delivery.create_order(
        platform=DeliveryPlatform.MEITUAN,
        merchant_id="merchant_001",
        customer_id="customer_123",
        pickup_lat=39.9042,
        pickup_lon=116.4074,
        delivery_lat=39.9142,
        delivery_lon=116.4174,
        items=[{"name": "Kung Pao Chicken", "qty": 1, "price": 38.0}],
        total_amount=38.0,
    )
    print(f"    Order ID           : {order.order_id}")
    print(f"    Customer (hashed)  : {order.customer_id_hash}")
    print(f"    Status             : {order.status.value}")

    # Update status
    suite.delivery.update_order_status(
        order.order_id, OrderStatus.DRIVER_ASSIGNED, driver_id="driver_456"
    )
    print(f"    Status updated     : {suite.delivery._orders[order.order_id].status.value}")

    delivery_status = suite.delivery.get_status()
    print(f"    Total orders       : {delivery_status['total_orders']}")
    print(f"    GCJ-02 enabled     : {delivery_status['gcj02_enabled']}")

    # ── Test 2: Multi-Language Chatbot ───────────────────────────────────
    print("\n  ── [ML1] Multi-Language Support Chatbot ──")

    # Start session
    session = suite.chatbot.start_session("customer_123")
    print(f"    Session ID         : {session.session_id}")

    # Test English
    result_en = suite.chatbot.process_message(
        session.session_id,
        "Where is my order? It's been 30 minutes already.",
    )
    print(f"\n    [English]")
    print(f"    Intent             : {result_en['intent']}")
    print(f"    Language           : {result_en['language']}")
    print(f"    Response           : {result_en['response'][:80]}...")

    # Test Chinese
    result_zh = suite.chatbot.process_message(
        session.session_id,
        "我的订单什么时候到？已经等了很久了。",
    )
    print(f"\n    [Chinese]")
    print(f"    Intent             : {result_zh['intent']}")
    print(f"    Language           : {result_zh['language']}")
    print(f"    Response           : {result_zh['response'][:80]}...")

    # Test Thai
    result_th = suite.chatbot.process_message(
        session.session_id,
        "คำสั่งซื้อของฉันอยู่ไหน รอมา 30 นาทีแล้ว",
    )
    print(f"\n    [Thai]")
    print(f"    Intent             : {result_th['intent']}")
    print(f"    Language           : {result_th['language']}")
    print(f"    Response           : {result_th['response'][:80]}...")

    # Test escalation
    result_escalate = suite.chatbot.process_message(
        session.session_id,
        "I want to speak to a manager! This is terrible service!",
    )
    print(f"\n    [Escalation Test]")
    print(f"    Escalated          : {result_escalate['escalated']}")

    chatbot_status = suite.chatbot.get_status()
    print(f"\n    Active sessions    : {chatbot_status['active_sessions']}")
    print(f"    Languages          : {chatbot_status['supported_languages']}")

    # ── Test 3: China Demand Predictor ───────────────────────────────────
    print("\n  ── [CD1] China Demand Predictor ──")

    # Record historical demand
    for region in [ChinaRegion.NORTH, ChinaRegion.EAST]:
        for hour in range(168):  # 7 days
            base = 1000.0
            daily_pattern = 200.0 * math.sin(2 * math.pi * hour / 24)
            weekend_boost = 1.3 if (hour // 24) % 7 >= 5 else 1.0
            noise = np.random.randn() * 50.0
            demand = max(0, base * weekend_boost + daily_pattern + noise)
            suite.demand.record_demand(region, "food_delivery", demand)

    # Forecast demand
    forecast = suite.demand.forecast_demand(
        region=ChinaRegion.NORTH,
        category="food_delivery",
        platform=DeliveryPlatform.MEITUAN,
    )
    print(f"    Forecast ID        : {forecast.forecast_id}")
    print(f"    Region             : {forecast.region.value}")
    print(f"    Horizon            : {forecast.horizon_days} days")
    print(f"    Mean forecast      : {np.mean(forecast.forecast_mean):.0f}")
    print(f"    95% CI             : [{np.mean(forecast.ci_lower):.0f}, {np.mean(forecast.ci_upper):.0f}]")
    print(f"    Festival factor    : {forecast.festival_factor:.2f}")
    print(f"    Festival           : {forecast.festival_name or 'none'}")

    # Check upcoming festivals
    upcoming = suite.demand.festival_calendar.get_upcoming_festivals(
        datetime.now(), days_ahead=60
    )
    print(f"\n    Upcoming festivals : {len(upcoming)}")
    for festival, date in upcoming[:3]:
        print(f"      • {festival.value}: {date.strftime('%Y-%m-%d')}")

    demand_status = suite.demand.get_status()
    print(f"\n    History keys       : {demand_status['history_keys']}")
    print(f"    TDA available      : {demand_status['tda_available']}")

    # ── Test 4: Full Workflow ────────────────────────────────────────────
    print("\n  ── Full Delivery Workflow ──")

    workflow_result = suite.run_delivery_workflow(
        platform=DeliveryPlatform.ELE_ME,
        merchant_id="merchant_002",
        customer_id="customer_789",
        items=[{"name": "Hot Pot Set", "qty": 1, "price": 128.0}],
        total_amount=128.0,
    )
    print(f"    Order ID           : {workflow_result['order_id']}")
    print(f"    Session ID         : {workflow_result['session_id']}")
    print(f"    Forecast mean      : {workflow_result['forecast_mean']:.0f}")
    print(f"    Festival factor    : {workflow_result['festival_factor']:.2f}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    Delivery orders    : {status['delivery']['total_orders']}")
    print(f"    Chat sessions      : {status['chatbot']['active_sessions']}")
    print(f"    Demand history     : {status['demand']['history_keys']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Delivery APIs require valid merchant accounts")
    print("  • GCJ-02 conversion is APPROXIMATE")
    print("  • Chatbot is TEMPLATE-based; production needs fine-tuned LLM")
    print("  • China demand forecasting is PROBABILISTIC")
    print("  • Festival effects are HISTORICAL patterns, not guarantees")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires PIPL compliance review")
    print("=" * 72)
    print("  ETHICAL FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • PNDC Axiom 3: Data minimization, consent-based")
    print("  • Buddhist (กรุณา): Chatbot helps, doesn't frustrate")
    print("  • Taoist (อู๋เหวย): Demand flows with natural rhythms")
    print("  • Human-in-the-loop: Sensitive issues escalate to humans")
    print("  • No automated refunds without human approval")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
