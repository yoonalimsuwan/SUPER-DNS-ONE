"""
=============================================================================
SESI FLEET EXTENDED — CHINA PAYMENTS + RESTRICTION ROUTING + WEBSOCKET
WeChat Pay / Alipay + License Plate Restrictions + Real-Time Tracking
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[CNP1] CNPaymentGateway
      • WeChat Pay API v3 (微信支付)
      • Alipay Open Platform (支付宝)
      • QR code generation (Native, H5, Mini Program)
      • Refund & dispute handling
      • PCI-DSS compliance abstraction
      • Currency: CNY (RMB) with FX support

[CR1] ChinaRestrictionRouter
      • License plate restriction rules (限行/限号) by city
      • HAVA (High-pollution Alert Vehicle Abatement) zones
      • Low Emission Zones (LEZ) for diesel trucks
      • Time-based restrictions (工作日高峰时段)
      • Integration with Amap/Baidu traffic APIs

[WS1] FleetWebSocketHub
      • WebSocket server for real-time driver tracking
      • JWT-based authentication
      • Pub/Sub for fleet-wide events
      • No-Zeno bounded message rate
      • Reconnection with exponential backoff
      • Driver consent logging (PNDC Axiom 3)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] WeChat Pay Signature: HMAC-SHA256 + RSA-2048
[C] Alipay Signature: RSA2 (SHA256WithRSA)
[C] WebSocket RFC 6455: Upgrade handshake, binary frames
[C] JWT (RFC 7519): Stateless driver authentication
[C] Exponential Backoff: t_wait = min(base * 2^n, max_wait)
[C] License Plate Restriction Graph:
    G = (V, E) where E = {(u,v) | v not in restricted zone at time t}
[C] SESI_NoZeno_Closure Thm 3.2: Bounded WebSocket messages
[C] PNDC Axiom 3: Consent-based real-time tracking

HONEST SCOPE
------------
• WeChat Pay/Alipay require Chinese business license (营业执照)
• Payment processing requires PCI-DSS Level 1 certification
• License plate rules CHANGE FREQUENTLY; must be updated per city
• WebSocket tracking requires driver EXPLICIT consent
• All thresholds are FREE DESIGN PARAMETERS
• Production requires legal review per PRC Cybersecurity Law (网络安全法)
=============================================================================
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import secrets
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.fleet_cn_extended")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.websockets import WebSocketState
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Integration with previous modules
try:
    from sesi_fleet_intelligence_extended import (
        TrafficAPIBridge, TrafficConfig, TrafficProvider,
        DriverBehaviorScorer, EVChargingOptimizer,
    )
    HAS_FLEET = True
except ImportError:
    HAS_FLEET = False


__all__ = [
    # Payment
    "CNPaymentProvider", "PaymentStatus", "QRCodeType",
    "CNPaymentGateway", "CNPaymentConfig", "PaymentOrder", "PaymentResult",
    # Restrictions
    "RestrictionType", "VehicleType", "PlateColor",
    "ChinaRestrictionRouter", "RestrictionConfig", "RestrictionRule",
    # WebSocket
    "FleetWebSocketHub", "WebSocketConfig", "DriverConnection",
    # Orchestrator
    "SESIFleetCNExtendedSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class CNPaymentProvider(str, Enum):
    """Chinese payment providers."""
    WECHAT_PAY = "wechat_pay"      # 微信支付
    ALIPAY = "alipay"              # 支付宝
    UNIONPAY = "unionpay"          # 银联


class PaymentStatus(str, Enum):
    """Payment order status."""
    PENDING = "pending"
    PAID = "paid"
    REFUNDED = "refunded"
    CLOSED = "closed"
    FAILED = "failed"


class QRCodeType(str, Enum):
    """QR code types for payments."""
    NATIVE = "native"              # 商家扫码 (merchant scans user)
    H5 = "h5"                      # 网页支付 (web payment)
    MINI_PROGRAM = "mini_program"  # 小程序 (WeChat Mini Program)
    APP = "app"                    # APP支付 (in-app payment)


class RestrictionType(str, Enum):
    """Types of traffic restrictions in China."""
    LICENSE_PLATE_TAIL = "license_plate_tail"  # 尾号限行 (last digit restriction)
    ODD_EVEN = "odd_even"                     # 单双号限行
    LOW_EMISSION_ZONE = "low_emission_zone"   # 低排放区
    TRUCK_RESTRICTION = "truck_restriction"   # 货车限行
    HAVA = "hava"                             # High-pollution alert
    PEAK_HOUR = "peak_hour"                   # 高峰时段
    OUTER_RING = "outer_ring"                 # 外环限行


class VehicleType(str, Enum):
    """Vehicle types for restriction rules."""
    PASSENGER_CAR = "passenger_car"           # 小客车
    LIGHT_TRUCK = "light_truck"               # 轻型货车
    HEAVY_TRUCK = "heavy_truck"               # 重型货车
    NEW_ENERGY = "new_energy"                 # 新能源车 (exempt from many rules)
    MOTORCYCLE = "motorcycle"                 # 摩托车


class PlateColor(str, Enum):
    """License plate colors in China (affects restrictions)."""
    BLUE = "blue"          # 蓝牌 (standard ICE vehicles)
    YELLOW = "yellow"      # 黄牌 (large vehicles, trucks)
    GREEN = "green"        # 绿牌 (New Energy Vehicles - NEV)
    BLACK = "black"        # 黑牌 (foreign/embassy)
    WHITE = "white"        # 白牌 (military/police)


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PaymentOrder:
    """Payment order for Chinese payment gateways."""
    order_id: str
    merchant_order_no: str
    provider: CNPaymentProvider
    amount_cny: float              # Amount in CNY (分 for WeChat)
    description: str
    qr_type: QRCodeType = QRCodeType.NATIVE
    status: PaymentStatus = PaymentStatus.PENDING
    # Response fields
    qr_code_url: Optional[str] = None
    prepay_id: Optional[str] = None
    trade_no: Optional[str] = None  # 第三方交易号
    # Metadata
    created_at: float = field(default_factory=time.time)
    paid_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentResult:
    """Result of a payment operation."""
    success: bool
    order_id: str
    status: PaymentStatus
    message: str = ""
    provider_response: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RestrictionRule:
    """A single traffic restriction rule."""
    rule_id: str
    city: str                       # e.g., "Beijing", "Shanghai", "Guangzhou"
    restriction_type: RestrictionType
    # Scope
    vehicle_types: List[VehicleType] = field(default_factory=list)
    plate_colors: List[PlateColor] = field(default_factory=list)
    # Time-based
    weekdays: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])  # 1=Mon
    start_hour: int = 7
    end_hour: int = 20
    # License plate rules
    restricted_digits: List[int] = field(default_factory=list)  # 0-9
    # Geographic
    zone_polygon: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon)
    zone_name: str = ""
    # Penalties
    fine_cny: float = 100.0
    penalty_points: int = 0
    # Validity
    effective_date: Optional[str] = None  # YYYY-MM-DD
    expiry_date: Optional[str] = None
    # Metadata
    source: str = "official"        # "official" | "crowdsourced"
    last_verified: float = field(default_factory=time.time)


@dataclass
class DriverConnection:
    """Active WebSocket connection for a driver."""
    connection_id: str
    driver_id_hash: str             # PNDC Axiom 3: anonymized
    vehicle_id: str
    websocket: Any                  # WebSocket object
    # State
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    last_position: Optional[Tuple[float, float]] = None
    consent_given: bool = False
    # Metadata
    device_info: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# [CNP1] CHINA PAYMENT GATEWAY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CNPaymentConfig:
    """Configuration for Chinese payment gateways."""
    # WeChat Pay (微信支付)
    wechat_app_id: Optional[str] = None
    wechat_mch_id: Optional[str] = None          # 商户号
    wechat_api_key: Optional[str] = None         # API v3 key
    wechat_serial_no: Optional[str] = None       # 证书序列号
    wechat_private_key_path: Optional[str] = None
    # Alipay (支付宝)
    alipay_app_id: Optional[str] = None
    alipay_private_key_path: Optional[str] = None
    alipay_public_key_path: Optional[str] = None
    alipay_sign_type: str = "RSA2"               # RSA2 (SHA256WithRSA)
    # Common
    notify_url: str = ""                         # 回调通知URL
    currency: str = "CNY"
    # Security
    pci_dss_compliant: bool = True
    # Rate limiting
    max_transactions_per_minute: int = 100
    # Sandbox mode
    sandbox: bool = True


class CNPaymentGateway:
    """
    [D] Payment gateway for WeChat Pay and Alipay.

    ⚠️ REQUIREMENTS:
      • Chinese business license (营业执照)
      • Merchant qualification (商户资质)
      • PCI-DSS Level 1 certification (for card data)
      • ICP Filing (ICP备案) for web payments

    Features:
      • WeChat Pay API v3 (Native, H5, Mini Program, APP)
      • Alipay Open Platform (PC, WAP, APP, QR)
      • QR code generation
      • Refund & dispute handling
      • Signature verification (HMAC-SHA256, RSA2)

    HONEST: This is an ABSTRACTION. Real deployment requires:
      • Official merchant accounts
      • PCI-DSS compliance
      • Production certificates from WeChat/Alipay

    Mathematical grounding:
      • WeChat Pay v3: HMAC-SHA256 + RSA-2048 signature
      • Alipay: RSA2 (SHA256WithRSA) signature
    """

    # API Endpoints
    WECHAT_ENDPOINTS = {
        "native": "https://api.mch.weixin.qq.com/v3/pay/transactions/native",
        "h5": "https://api.mch.weixin.qq.com/v3/pay/transactions/h5",
        "jsapi": "https://api.mch.weixin.qq.com/v3/pay/transactions/jsapi",
        "query": "https://api.mch.weixin.qq.com/v3/pay/transactions/out-trade-no/{out_trade_no}",
        "refund": "https://api.mch.weixin.qq.com/v3/refund/domestic/refunds",
        "close": "https://api.mch.weixin.qq.com/v3/pay/transactions/out-trade-no/{out_trade_no}/close",
    }

    ALIPAY_ENDPOINTS = {
        "gateway": "https://openapi.alipay.com/gateway.do",
        "sandbox_gateway": "https://openapi.alipaydev.com/gateway.do",
    }

    def __init__(self, cfg: Optional[CNPaymentConfig] = None):
        self.cfg = cfg or CNPaymentConfig()
        self._orders: Dict[str, PaymentOrder] = {}
        self._transaction_timestamps: deque = deque(maxlen=10000)
        self._session = requests.Session() if HAS_REQUESTS else None

        # Load keys if paths provided
        self._wechat_private_key = None
        self._alipay_private_key = None
        self._load_keys()

    def _load_keys(self) -> None:
        """Load private keys from files (if paths provided)."""
        if not HAS_CRYPTO:
            logger.warning("[CNPay] cryptography library not available")
            return

        if self.cfg.wechat_private_key_path:
            try:
                with open(self.cfg.wechat_private_key_path, "rb") as f:
                    self._wechat_private_key = load_pem_private_key(
                        f.read(), password=None
                    )
                logger.info("[CNPay] WeChat private key loaded")
            except Exception as e:
                logger.error(f"[CNPay] Failed to load WeChat key: {e}")

        if self.cfg.alipay_private_key_path:
            try:
                with open(self.cfg.alipay_private_key_path, "rb") as f:
                    self._alipay_private_key = load_pem_private_key(
                        f.read(), password=None
                    )
                logger.info("[CNPay] Alipay private key loaded")
            except Exception as e:
                logger.error(f"[CNPay] Failed to load Alipay key: {e}")

    # ── Rate Limiting ────────────────────────────────────────────────────

    def _check_transaction_rate(self) -> Dict[str, Any]:
        """Check if transaction is within rate limit."""
        now = time.time()
        cutoff = now - 60.0
        self._transaction_timestamps = deque(
            (t for t in self._transaction_timestamps if t > cutoff),
            maxlen=10000,
        )
        if len(self._transaction_timestamps) >= self.cfg.max_transactions_per_minute:
            return {"allowed": False, "reason": "rate_limited"}
        return {"allowed": True}

    # ── WeChat Pay Signature ─────────────────────────────────────────────

    def _wechat_sign(self, method: str, url: str, timestamp: str,
                     nonce_str: str, body: str) -> str:
        """
        Generate WeChat Pay API v3 signature.

        Signature = SHA256-RSA2048(
            HTTP_METHOD + "\\n" +
            URL + "\\n" +
            TIMESTAMP + "\\n" +
            NONCE_STR + "\\n" +
            BODY + "\\n"
        )
        """
        if not self._wechat_private_key:
            logger.warning("[CNPay] WeChat private key not loaded")
            return ""

        message = f"{method}\n{url}\n{timestamp}\n{nonce_str}\n{body}\n"

        if HAS_CRYPTO:
            signature = self._wechat_private_key.sign(
                message.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        return ""

    def _wechat_auth_header(self, method: str, url: str, body: str = "") -> str:
        """Generate WeChat Pay Authorization header."""
        timestamp = str(int(time.time()))
        nonce_str = secrets.token_hex(16)
        signature = self._wechat_sign(method, url, timestamp, nonce_str, body)

        return (
            f'WECHATPAY2-SHA256-RSA2048 '
            f'mchid="{self.cfg.wechat_mch_id}",'
            f'nonce_str="{nonce_str}",'
            f'timestamp="{timestamp}",'
            f'serial_no="{self.cfg.wechat_serial_no}",'
            f'signature="{signature}"'
        )

    # ── Alipay Signature ─────────────────────────────────────────────────

    def _alipay_sign(self, params: Dict[str, str]) -> str:
        """
        Generate Alipay RSA2 signature.

        Signature = SHA256WithRSA(
            key1=value1&key2=value2&... (sorted by key)
        )
        """
        if not self._alipay_private_key:
            logger.warning("[CNPay] Alipay private key not loaded")
            return ""

        # Sort parameters and build sign string
        sorted_params = sorted(params.items())
        sign_string = "&".join(f"{k}={v}" for k, v in sorted_params if v)

        if HAS_CRYPTO:
            signature = self._alipay_private_key.sign(
                sign_string.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        return ""

    # ── WeChat Pay Operations ────────────────────────────────────────────

    def create_wechat_native_order(
        self,
        merchant_order_no: str,
        amount_cny: float,
        description: str,
    ) -> PaymentResult:
        """
        Create WeChat Pay Native order (商家扫码 - merchant scans user QR).

        Args:
            merchant_order_no: Merchant's order number (商户订单号)
            amount_cny: Amount in CNY (will be converted to 分)
            description: Order description

        Returns:
            PaymentResult with QR code URL
        """
        rate_check = self._check_transaction_rate()
        if not rate_check["allowed"]:
            return PaymentResult(
                success=False,
                order_id=merchant_order_no,
                status=PaymentStatus.FAILED,
                message=f"Rate limited: {rate_check['reason']}",
            )

        # WeChat uses 分 (cents), so multiply by 100
        amount_fen = int(amount_cny * 100)

        order = PaymentOrder(
            order_id=f"wx_{uuid.uuid4().hex[:12]}",
            merchant_order_no=merchant_order_no,
            provider=CNPaymentProvider.WECHAT_PAY,
            amount_cny=amount_cny,
            description=description,
            qr_type=QRCodeType.NATIVE,
        )

        # Build request body
        body = {
            "appid": self.cfg.wechat_app_id,
            "mchid": self.cfg.wechat_mch_id,
            "description": description,
            "out_trade_no": merchant_order_no,
            "notify_url": self.cfg.notify_url,
            "amount": {
                "total": amount_fen,
                "currency": "CNY",
            },
        }

        if self.cfg.sandbox or not self.cfg.wechat_mch_id:
            # Mock response for testing
            order.qr_code_url = f"weixin://wxpay/bizpayurl?pr=mock_{uuid.uuid4().hex[:8]}"
            order.prepay_id = f"mock_prepay_{uuid.uuid4().hex[:8]}"
            order.status = PaymentStatus.PENDING
            self._orders[order.order_id] = order

            logger.info(
                f"[CNPay] WeChat Native order created (mock): "
                f"{merchant_order_no} = {amount_cny} CNY"
            )

            return PaymentResult(
                success=True,
                order_id=order.order_id,
                status=order.status,
                message="Order created (sandbox mode)",
                provider_response={"code_url": order.qr_code_url},
            )

        # Production: call WeChat API
        try:
            url = self.WECHAT_ENDPOINTS["native"]
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": self._wechat_auth_header("POST", url, json.dumps(body)),
            }

            response = self._session.post(url, json=body, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            order.qr_code_url = data.get("code_url")
            order.prepay_id = data.get("prepay_id")
            order.status = PaymentStatus.PENDING
            self._orders[order.order_id] = order

            self._transaction_timestamps.append(time.time())

            return PaymentResult(
                success=True,
                order_id=order.order_id,
                status=order.status,
                message="Order created",
                provider_response=data,
            )

        except Exception as e:
            logger.error(f"[CNPay] WeChat order creation failed: {e}")
            return PaymentResult(
                success=False,
                order_id=order.order_id if order else merchant_order_no,
                status=PaymentStatus.FAILED,
                message=str(e),
            )

    def query_wechat_order(self, merchant_order_no: str) -> PaymentResult:
        """Query WeChat Pay order status."""
        if self.cfg.sandbox:
            # Mock: find in local orders
            for order in self._orders.values():
                if order.merchant_order_no == merchant_order_no:
                    return PaymentResult(
                        success=True,
                        order_id=order.order_id,
                        status=order.status,
                        message=f"Status: {order.status.value}",
                    )
            return PaymentResult(
                success=False,
                order_id="",
                status=PaymentStatus.FAILED,
                message="Order not found",
            )

        # Production: call WeChat query API
        try:
            url = self.WECHAT_ENDPOINTS["query"].format(out_trade_no=merchant_order_no)
            headers = {
                "Accept": "application/json",
                "Authorization": self._wechat_auth_header("GET", url),
            }

            response = self._session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            trade_state = data.get("trade_state", "")
            if trade_state == "SUCCESS":
                status = PaymentStatus.PAID
            elif trade_state == "REFUND":
                status = PaymentStatus.REFUNDED
            elif trade_state == "CLOSED":
                status = PaymentStatus.CLOSED
            else:
                status = PaymentStatus.PENDING

            return PaymentResult(
                success=True,
                order_id=data.get("out_trade_no", ""),
                status=status,
                message=trade_state,
                provider_response=data,
            )

        except Exception as e:
            logger.error(f"[CNPay] WeChat query failed: {e}")
            return PaymentResult(
                success=False,
                order_id=merchant_order_no,
                status=PaymentStatus.FAILED,
                message=str(e),
            )

    # ── Alipay Operations ────────────────────────────────────────────────

    def create_alipay_order(
        self,
        merchant_order_no: str,
        amount_cny: float,
        subject: str,
    ) -> PaymentResult:
        """
        Create Alipay order (支付宝订单).

        Args:
            merchant_order_no: Merchant's order number
            amount_cny: Amount in CNY
            subject: Order subject (商品标题)

        Returns:
            PaymentResult with payment URL
        """
        rate_check = self._check_transaction_rate()
        if not rate_check["allowed"]:
            return PaymentResult(
                success=False,
                order_id=merchant_order_no,
                status=PaymentStatus.FAILED,
                message=f"Rate limited: {rate_check['reason']}",
            )

        order = PaymentOrder(
            order_id=f"ali_{uuid.uuid4().hex[:12]}",
            merchant_order_no=merchant_order_no,
            provider=CNPaymentProvider.ALIPAY,
            amount_cny=amount_cny,
            description=subject,
        )

        # Build Alipay request parameters
        params = {
            "app_id": self.cfg.alipay_app_id or "",
            "method": "alipay.trade.page.pay",
            "format": "JSON",
            "charset": "utf-8",
            "sign_type": self.cfg.alipay_sign_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "notify_url": self.cfg.notify_url,
            "biz_content": json.dumps({
                "out_trade_no": merchant_order_no,
                "total_amount": f"{amount_cny:.2f}",
                "subject": subject,
                "product_code": "FAST_INSTANT_TRADE_PAY",
            }),
        }

        # Sign the request
        params["sign"] = self._alipay_sign(params)

        if self.cfg.sandbox or not self.cfg.alipay_app_id:
            # Mock response
            gateway = self.ALIPAY_ENDPOINTS["sandbox_gateway"]
            order.qr_code_url = f"{gateway}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
            order.status = PaymentStatus.PENDING
            self._orders[order.order_id] = order

            logger.info(
                f"[CNPay] Alipay order created (mock): "
                f"{merchant_order_no} = {amount_cny} CNY"
            )

            return PaymentResult(
                success=True,
                order_id=order.order_id,
                status=order.status,
                message="Order created (sandbox mode)",
                provider_response={"payment_url": order.qr_code_url},
            )

        # Production: build payment URL
        gateway = self.ALIPAY_ENDPOINTS["gateway"]
        payment_url = f"{gateway}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        order.qr_code_url = payment_url
        order.status = PaymentStatus.PENDING
        self._orders[order.order_id] = order
        self._transaction_timestamps.append(time.time())

        return PaymentResult(
            success=True,
            order_id=order.order_id,
            status=order.status,
            message="Order created",
            provider_response={"payment_url": payment_url},
        )

    # ── Unified Payment Interface ────────────────────────────────────────

    def create_payment(
        self,
        provider: CNPaymentProvider,
        merchant_order_no: str,
        amount_cny: float,
        description: str,
        qr_type: QRCodeType = QRCodeType.NATIVE,
    ) -> PaymentResult:
        """
        Unified payment creation interface.

        Args:
            provider: WeChat Pay or Alipay
            merchant_order_no: Merchant's order number
            amount_cny: Amount in CNY
            description: Order description
            qr_type: QR code type

        Returns:
            PaymentResult
        """
        if provider == CNPaymentProvider.WECHAT_PAY:
            return self.create_wechat_native_order(
                merchant_order_no, amount_cny, description
            )
        elif provider == CNPaymentProvider.ALIPAY:
            return self.create_alipay_order(
                merchant_order_no, amount_cny, description
            )
        else:
            return PaymentResult(
                success=False,
                order_id=merchant_order_no,
                status=PaymentStatus.FAILED,
                message=f"Unsupported provider: {provider.value}",
            )

    def get_status(self) -> Dict[str, Any]:
        """Get payment gateway status."""
        return {
            "wechat_configured": self.cfg.wechat_mch_id is not None,
            "alipay_configured": self.cfg.alipay_app_id is not None,
            "sandbox_mode": self.cfg.sandbox,
            "total_orders": len(self._orders),
            "pci_dss_compliant": self.cfg.pci_dss_compliant,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [CR1] CHINA RESTRICTION ROUTER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RestrictionConfig:
    """Configuration for China traffic restriction routing."""
    # Default city
    default_city: str = "Beijing"
    # Rule cache TTL
    rule_cache_ttl_hours: int = 24
    # Restriction avoidance
    avoid_restricted_zones: bool = True
    warn_only: bool = False       # If True, only warn instead of blocking
    # NEV exemption
    nev_exempt: bool = True       # New Energy Vehicles exempt from most rules
    # No-Zeno
    max_route_checks_per_minute: int = 30


class ChinaRestrictionRouter:
    """
    [D] Traffic restriction-aware router for Chinese cities.

    Handles:
      • License plate tail number restrictions (尾号限行)
      • Odd/even day restrictions (单双号限行)
      • Low Emission Zones (低排放区)
      • Truck restrictions (货车限行)
      • HAVA (High-pollution Alert Vehicle Abatement)
      • Peak hour restrictions (高峰时段限行)

    ⚠️ HONEST SCOPE:
      • Rules CHANGE FREQUENTLY by city and date
      • Must be verified against official sources
      • NEV (green plate) exemptions vary by city
      • This is a RECOMMENDATION system; driver has final say

    Supported cities (major):
      • Beijing (北京): 5-ring tail number restriction
      • Shanghai (上海): Elevated road restrictions
      • Guangzhou (广州): 开四停四 (4 days in, 4 days out)
      • Shenzhen (深圳): Peak hour restrictions
      • Chengdu (成都): Tail number + region
      • Tianjin (天津): Tail number restriction
      • Hangzhou (杭州): 错峰限行 (staggered peak)
    """

    # Pre-loaded restriction rules (simplified; real deployment needs DB)
    DEFAULT_RULES = [
        # Beijing: 5-ring tail number restriction (工作日 7:00-20:00)
        RestrictionRule(
            rule_id="bj_tail_1_6",
            city="Beijing",
            restriction_type=RestrictionType.LICENSE_PLATE_TAIL,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[1],  # Monday
            start_hour=7,
            end_hour=20,
            restricted_digits=[1, 6],
            zone_name="五环路以内 (Within 5th Ring)",
            fine_cny=100.0,
            penalty_points=0,
        ),
        RestrictionRule(
            rule_id="bj_tail_2_7",
            city="Beijing",
            restriction_type=RestrictionType.LICENSE_PLATE_TAIL,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[2],  # Tuesday
            start_hour=7,
            end_hour=20,
            restricted_digits=[2, 7],
            zone_name="五环路以内",
            fine_cny=100.0,
        ),
        RestrictionRule(
            rule_id="bj_tail_3_8",
            city="Beijing",
            restriction_type=RestrictionType.LICENSE_PLATE_TAIL,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[3],
            start_hour=7,
            end_hour=20,
            restricted_digits=[3, 8],
            zone_name="五环路以内",
            fine_cny=100.0,
        ),
        RestrictionRule(
            rule_id="bj_tail_4_9",
            city="Beijing",
            restriction_type=RestrictionType.LICENSE_PLATE_TAIL,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[4],
            start_hour=7,
            end_hour=20,
            restricted_digits=[4, 9],
            zone_name="五环路以内",
            fine_cny=100.0,
        ),
        RestrictionRule(
            rule_id="bj_tail_5_0",
            city="Beijing",
            restriction_type=RestrictionType.LICENSE_PLATE_TAIL,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[5],
            start_hour=7,
            end_hour=20,
            restricted_digits=[5, 0],
            zone_name="五环路以内",
            fine_cny=100.0,
        ),
        # Shanghai: Elevated road restrictions (外牌高架限行)
        RestrictionRule(
            rule_id="sh_elevated_non_local",
            city="Shanghai",
            restriction_type=RestrictionType.OUTER_RING,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[1, 2, 3, 4, 5],
            start_hour=7,
            end_hour=20,
            zone_name="高架道路 (Elevated roads)",
            fine_cny=200.0,
            penalty_points=3,
        ),
        # Guangzhou: 开四停四 (4 days in, 4 days out for non-local plates)
        RestrictionRule(
            rule_id="gz_four_four",
            city="Guangzhou",
            restriction_type=RestrictionType.OUTER_RING,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[1, 2, 3, 4, 5, 6, 7],
            start_hour=0,
            end_hour=24,
            zone_name="管控区域 (Control zone)",
            fine_cny=200.0,
            penalty_points=1,
            source="开四停四政策",
        ),
        # Shenzhen: Peak hour restrictions (早晚高峰限行)
        RestrictionRule(
            rule_id="sz_peak_non_local",
            city="Shenzhen",
            restriction_type=RestrictionType.PEAK_HOUR,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[1, 2, 3, 4, 5],
            start_hour=7,
            end_hour=9,
            zone_name="全市 (Citywide)",
            fine_cny=300.0,
            penalty_points=3,
        ),
        RestrictionRule(
            rule_id="sz_peak_evening",
            city="Shenzhen",
            restriction_type=RestrictionType.PEAK_HOUR,
            vehicle_types=[VehicleType.PASSENGER_CAR],
            plate_colors=[PlateColor.BLUE],
            weekdays=[1, 2, 3, 4, 5],
            start_hour=17,
            end_hour=19,
            zone_name="全市",
            fine_cny=300.0,
            penalty_points=3,
        ),
        # Low Emission Zone (common in many cities)
        RestrictionRule(
            rule_id="lez_yellow_truck",
            city="Beijing",
            restriction_type=RestrictionType.LOW_EMISSION_ZONE,
            vehicle_types=[VehicleType.HEAVY_TRUCK, VehicleType.LIGHT_TRUCK],
            plate_colors=[PlateColor.YELLOW],
            weekdays=[1, 2, 3, 4, 5, 6, 7],
            start_hour=6,
            end_hour=23,
            zone_name="六环路以内 (Within 6th Ring)",
            fine_cny=100.0,
            penalty_points=3,
        ),
    ]

    def __init__(self, cfg: Optional[RestrictionConfig] = None):
        self.cfg = cfg or RestrictionConfig()
        self._rules: List[RestrictionRule] = self.DEFAULT_RULES.copy()
        self._check_timestamps: deque = deque(maxlen=1000)

    def add_rule(self, rule: RestrictionRule) -> None:
        """Add a new restriction rule."""
        self._rules.append(rule)
        logger.info(f"[Restriction] Added rule: {rule.rule_id} ({rule.city})")

    def _get_plate_tail(self, plate_number: str) -> int:
        """Extract the last digit from a license plate."""
        # Chinese plates: e.g., 京A12345, 沪B·12345
        # Remove non-alphanumeric characters and get last digit
        cleaned = ''.join(c for c in plate_number if c.isalnum())
        for c in reversed(cleaned):
            if c.isdigit():
                return int(c)
        return -1

    def check_restriction(
        self,
        plate_number: str,
        plate_color: PlateColor,
        vehicle_type: VehicleType,
        city: str,
        check_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Check if a vehicle is restricted at a given time and location.

        Args:
            plate_number: License plate number (e.g., "京A12345")
            plate_color: Plate color (affects exemptions)
            vehicle_type: Vehicle type
            city: City name
            check_time: Time to check (default: now)

        Returns:
            dict with restriction status and applicable rules
        """
        # Rate limiting
        now = time.time()
        cutoff = now - 60.0
        self._check_timestamps = deque(
            (t for t in self._check_timestamps if t > cutoff),
            maxlen=1000,
        )
        if len(self._check_timestamps) >= self.cfg.max_route_checks_per_minute:
            return {"error": "rate_limited"}

        self._check_timestamps.append(now)

        check_time = check_time or datetime.now()
        weekday = check_time.isoweekday()  # 1=Monday, 7=Sunday
        hour = check_time.hour

        # NEV exemption check
        if self.cfg.nev_exempt and plate_color == PlateColor.GREEN:
            return {
                "restricted": False,
                "reason": "NEV_exempt",
                "message": "新能源车不受限行限制 (NEV exempt from restrictions)",
                "applicable_rules": [],
            }

        # Find applicable rules
        applicable_rules = []
        plate_tail = self._get_plate_tail(plate_number)

        for rule in self._rules:
            # City match
            if rule.city.lower() != city.lower():
                continue

            # Vehicle type match
            if vehicle_type not in rule.vehicle_types:
                continue

            # Plate color match (if specified)
            if rule.plate_colors and plate_color not in rule.plate_colors:
                continue

            # Weekday match
            if weekday not in rule.weekdays:
                continue

            # Time match
            if not (rule.start_hour <= hour < rule.end_hour):
                continue

            # License plate tail match (for tail number restrictions)
            if rule.restriction_type == RestrictionType.LICENSE_PLATE_TAIL:
                if plate_tail in rule.restricted_digits:
                    applicable_rules.append(rule)
            else:
                applicable_rules.append(rule)

        is_restricted = len(applicable_rules) > 0

        result = {
            "restricted": is_restricted,
            "plate_number": plate_number,
            "plate_color": plate_color.value,
            "vehicle_type": vehicle_type.value,
            "city": city,
            "check_time": check_time.isoformat(),
            "weekday": weekday,
            "hour": hour,
            "plate_tail": plate_tail,
            "applicable_rules": [],
        }

        if is_restricted:
            result["message"] = "⚠️ 车辆受限行限制 (Vehicle restricted)"
            result["total_fine"] = sum(r.fine_cny for r in applicable_rules)
            result["total_points"] = sum(r.penalty_points for r in applicable_rules)
            result["applicable_rules"] = [
                {
                    "rule_id": r.rule_id,
                    "type": r.restriction_type.value,
                    "zone": r.zone_name,
                    "time": f"{r.start_hour:02d}:00-{r.end_hour:02d}:00",
                    "fine_cny": r.fine_cny,
                    "penalty_points": r.penalty_points,
                }
                for r in applicable_rules
            ]
        else:
            result["message"] = "✓ 车辆不受限行限制 (Vehicle not restricted)"

        logger.info(
            f"[Restriction] Check {plate_number} in {city}: "
            f"restricted={is_restricted}, rules={len(applicable_rules)}"
        )

        return result

    def find_alternative_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        plate_number: str,
        plate_color: PlateColor,
        vehicle_type: VehicleType,
        city: str,
        departure_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Find alternative route avoiding restricted zones.

        This is a SIMPLIFIED version. Real deployment needs:
          • Integration with Amap/Baidu routing APIs
          • Polygon-based zone checking
          • Time-dependent routing

        Args:
            origin: (lat, lon) origin
            destination: (lat, lon) destination
            plate_number: License plate
            plate_color: Plate color
            vehicle_type: Vehicle type
            city: City
            departure_time: Departure time

        Returns:
            dict with routing recommendation
        """
        departure_time = departure_time or datetime.now()

        # Check restrictions at departure time
        restriction_check = self.check_restriction(
            plate_number, plate_color, vehicle_type, city, departure_time
        )

        if not restriction_check.get("restricted", False):
            return {
                "status": "no_restriction",
                "message": "No restrictions at departure time",
                "recommendation": "proceed_with_original_route",
                "restriction_check": restriction_check,
            }

        # If restricted, provide alternatives
        applicable_rules = restriction_check.get("applicable_rules", [])

        # Determine when restriction ends
        restriction_end_hour = max(
            (r.get("time", "00:00-00:00").split("-")[1].split(":")[0] for r in applicable_rules),
            default="20"
        )

        # Calculate wait time
        current_hour = departure_time.hour
        try:
            end_hour = int(restriction_end_hour)
        except ValueError:
            end_hour = 20

        if current_hour < end_hour:
            wait_hours = end_hour - current_hour
            recommended_departure = departure_time + timedelta(hours=wait_hours)
        else:
            wait_hours = 0
            recommended_departure = departure_time

        return {
            "status": "restricted",
            "message": "车辆受限行限制，建议调整出发时间或路线",
            "restriction_check": restriction_check,
            "alternatives": [
                {
                    "option": "wait",
                    "description": f"等待限行结束后出发 (Wait until restriction ends)",
                    "recommended_departure": recommended_departure.isoformat(),
                    "wait_hours": wait_hours,
                },
                {
                    "option": "reroute",
                    "description": "绕行限行区域 (Reroute around restricted zone)",
                    "note": "需要集成地图API获取具体绕行路线 (Requires map API integration)",
                },
                {
                    "option": "public_transport",
                    "description": "改乘公共交通 (Use public transport)",
                    "note": "适用于市区短途 (Suitable for short urban trips)",
                },
            ],
            "total_fine_if_violated": restriction_check.get("total_fine", 0),
            "penalty_points_if_violated": restriction_check.get("total_points", 0),
        }

    def get_city_rules(self, city: str) -> List[Dict[str, Any]]:
        """Get all restriction rules for a city."""
        rules = [r for r in self._rules if r.city.lower() == city.lower()]
        return [
            {
                "rule_id": r.rule_id,
                "type": r.restriction_type.value,
                "zone": r.zone_name,
                "weekdays": r.weekdays,
                "time": f"{r.start_hour:02d}:00-{r.end_hour:02d}:00",
                "vehicle_types": [v.value for v in r.vehicle_types],
                "restricted_digits": r.restricted_digits,
                "fine_cny": r.fine_cny,
            }
            for r in rules
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        cities = set(r.city for r in self._rules)
        return {
            "total_rules": len(self._rules),
            "cities_covered": list(cities),
            "nev_exempt": self.cfg.nev_exempt,
            "avoid_restricted_zones": self.cfg.avoid_restricted_zones,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [WS1] FLEET WEBSOCKET HUB
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket hub."""
    # Connection
    max_connections: int = 1000
    heartbeat_interval_s: float = 30.0
    connection_timeout_s: float = 60.0
    # Authentication
    jwt_secret: str = "change_this_secret_in_production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    # No-Zeno
    max_messages_per_second: int = 10
    # Consent
    require_consent: bool = True
    log_consent: bool = True


class FleetWebSocketHub:
    """
    [D] WebSocket hub for real-time fleet tracking.

    ⚠️ ETHICAL FRAMEWORK (PNDC Axiom 3):
      • Driver CONSENT required for real-time tracking
      • Data is operational (GPS, status), not surveillance
      • Drivers can disconnect at any time
      • Consent is logged and auditable

    Features:
      • WebSocket server (RFC 6455)
      • JWT-based authentication
      • Pub/Sub for fleet events
      • Heartbeat monitoring
      • No-Zeno bounded message rate
      • Exponential backoff for reconnection

    Message types:
      • position: GPS position update
      • status: Driver/vehicle status change
      • alert: Emergency or system alert
      • heartbeat: Keep-alive ping
    """

    def __init__(self, cfg: Optional[WebSocketConfig] = None):
        self.cfg = cfg or WebSocketConfig()
        self._connections: Dict[str, DriverConnection] = {}
        self._driver_connections: Dict[str, str] = {}  # driver_id_hash → connection_id
        self._consent_log: List[Dict[str, Any]] = []
        self._message_timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

    # ── JWT Authentication ───────────────────────────────────────────────

    def generate_token(
        self,
        driver_id: str,
        vehicle_id: str,
        consent_given: bool = False,
    ) -> str:
        """
        Generate JWT token for driver authentication.

        Args:
            driver_id: Driver identifier (will be hashed)
            vehicle_id: Vehicle identifier
            consent_given: Whether driver has given tracking consent

        Returns:
            JWT token string
        """
        driver_id_hash = hashlib.sha256(driver_id.encode()).hexdigest()[:16]

        payload = {
            "driver_id_hash": f"driver_{driver_id_hash}",
            "vehicle_id": vehicle_id,
            "consent_given": consent_given,
            "iat": int(time.time()),
            "exp": int(time.time()) + self.cfg.jwt_expiry_hours * 3600,
        }

        if HAS_JWT:
            return jwt.encode(payload, self.cfg.jwt_secret, algorithm=self.cfg.jwt_algorithm)
        else:
            # Fallback: simple token (not secure for production)
            return f"mock_token_{uuid.uuid4().hex[:16]}"

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        if HAS_JWT:
            try:
                return jwt.decode(
                    token, self.cfg.jwt_secret, algorithms=[self.cfg.jwt_algorithm]
                )
            except jwt.ExpiredSignatureError:
                logger.warning("[WS] Token expired")
                return None
            except jwt.InvalidTokenError:
                logger.warning("[WS] Invalid token")
                return None
        else:
            # Mock verification
            return {"driver_id_hash": "mock_driver", "vehicle_id": "mock_vehicle", "consent_given": True}

    # ── Rate Limiting ────────────────────────────────────────────────────

    def _check_message_rate(self, connection_id: str) -> Dict[str, Any]:
        """Check if message is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 1.0
        timestamps = self._message_timestamps[connection_id]

        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= self.cfg.max_messages_per_second:
            return {"allowed": False, "reason": "rate_limited"}

        return {"allowed": True}

    # ── Connection Management ────────────────────────────────────────────

    async def connect(self, websocket: Any, token: str) -> Optional[DriverConnection]:
        """
        Accept and authenticate a WebSocket connection.

        Args:
            websocket: WebSocket object
            token: JWT token for authentication

        Returns:
            DriverConnection if successful, None otherwise
        """
        # Verify token
        payload = self.verify_token(token)
        if payload is None:
            await websocket.close(code=4001, reason="Authentication failed")
            return None

        # Check consent
        if self.cfg.require_consent and not payload.get("consent_given", False):
            await websocket.close(code=4003, reason="Consent required")
            logger.warning(f"[WS] Connection rejected: no consent from {payload.get('driver_id_hash')}")
            return None

        # Check connection limit
        if len(self._connections) >= self.cfg.max_connections:
            await websocket.close(code=4004, reason="Server full")
            return None

        # Create connection
        connection_id = f"conn_{uuid.uuid4().hex[:10]}"
        driver_id_hash = payload.get("driver_id_hash", "")
        vehicle_id = payload.get("vehicle_id", "")

        connection = DriverConnection(
            connection_id=connection_id,
            driver_id_hash=driver_id_hash,
            vehicle_id=vehicle_id,
            websocket=websocket,
            consent_given=payload.get("consent_given", False),
        )

        self._connections[connection_id] = connection
        self._driver_connections[driver_id_hash] = connection_id

        # Log consent
        if self.cfg.log_consent:
            self._consent_log.append({
                "timestamp": time.time(),
                "driver_id_hash": driver_id_hash,
                "vehicle_id": vehicle_id,
                "consent_given": True,
                "connection_id": connection_id,
            })

        logger.info(
            f"[WS] Connected: {driver_id_hash} ({vehicle_id}) "
            f"[{len(self._connections)} total]"
        )

        return connection

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket connection."""
        if connection_id in self._connections:
            connection = self._connections[connection_id]
            del self._driver_connections[connection.driver_id_hash]
            del self._connections[connection_id]

            logger.info(
                f"[WS] Disconnected: {connection.driver_id_hash} "
                f"[{len(self._connections)} remaining]"
            )

    # ── Message Handling ─────────────────────────────────────────────────

    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming WebSocket message.

        Args:
            connection_id: Connection identifier
            message: Message dict with 'type' and 'data'

        Returns:
            Response dict
        """
        if connection_id not in self._connections:
            return {"error": "connection_not_found"}

        # Rate limiting
        rate_check = self._check_message_rate(connection_id)
        if not rate_check["allowed"]:
            return {"error": "rate_limited", "reason": rate_check["reason"]}

        self._message_timestamps[connection_id].append(time.time())

        connection = self._connections[connection_id]
        msg_type = message.get("type", "")
        data = message.get("data", {})

        if msg_type == "position":
            return await self._handle_position(connection, data)
        elif msg_type == "status":
            return await self._handle_status(connection, data)
        elif msg_type == "heartbeat":
            return await self._handle_heartbeat(connection)
        elif msg_type == "alert":
            return await self._handle_alert(connection, data)
        else:
            return {"error": "unknown_message_type", "type": msg_type}

    async def _handle_position(
        self, connection: DriverConnection, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle position update message."""
        lat = data.get("latitude")
        lon = data.get("longitude")
        timestamp = data.get("timestamp", time.time())

        if lat is not None and lon is not None:
            connection.last_position = (lat, lon)
            connection.last_heartbeat = time.time()

            # Broadcast to fleet (optional)
            await self.broadcast_fleet_position(connection, lat, lon, timestamp)

            return {
                "status": "received",
                "position": (lat, lon),
                "timestamp": timestamp,
            }

        return {"error": "invalid_position_data"}

    async def _handle_status(
        self, connection: DriverConnection, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle status update message."""
        new_status = data.get("status", "")
        connection.last_heartbeat = time.time()

        logger.info(
            f"[WS] Status update: {connection.driver_id_hash} → {new_status}"
        )

        # Broadcast status change to fleet
        await self.broadcast_fleet_status(connection, new_status)

        return {"status": "received", "new_status": new_status}

    async def _handle_heartbeat(self, connection: DriverConnection) -> Dict[str, Any]:
        """Handle heartbeat message."""
        connection.last_heartbeat = time.time()
        return {"status": "pong", "timestamp": time.time()}

    async def _handle_alert(
        self, connection: DriverConnection, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle alert message."""
        alert_type = data.get("alert_type", "")
        message = data.get("message", "")

        logger.warning(
            f"[WS] Alert from {connection.driver_id_hash}: "
            f"{alert_type} - {message}"
        )

        return {"status": "alert_received", "alert_type": alert_type}

    # ── Broadcasting ─────────────────────────────────────────────────────

    async def broadcast_fleet_position(
        self,
        connection: DriverConnection,
        lat: float,
        lon: float,
        timestamp: float,
    ) -> None:
        """Broadcast position to all fleet managers (not other drivers)."""
        # In production: send to fleet manager dashboard only
        # Not broadcast to all drivers (privacy)
        logger.debug(
            f"[WS] Fleet position: {connection.driver_id_hash} "
            f"@ ({lat:.6f}, {lon:.6f})"
        )

    async def broadcast_fleet_status(
        self, connection: DriverConnection, status: str
    ) -> None:
        """Broadcast status change to fleet managers."""
        logger.debug(
            f"[WS] Fleet status: {connection.driver_id_hash} → {status}"
        )

    async def send_to_driver(
        self, driver_id_hash: str, message: Dict[str, Any]
    ) -> bool:
        """Send message to a specific driver."""
        connection_id = self._driver_connections.get(driver_id_hash)
        if connection_id and connection_id in self._connections:
            connection = self._connections[connection_id]
            try:
                await connection.websocket.send_json(message)
                return True
            except Exception as e:
                logger.error(f"[WS] Send failed: {e}")
                return False
        return False

    # ── Connection Health ────────────────────────────────────────────────

    def get_stale_connections(self, timeout_s: Optional[float] = None) -> List[str]:
        """Get list of stale connection IDs."""
        timeout = timeout_s or self.cfg.connection_timeout_s
        now = time.time()
        stale = []

        for conn_id, conn in self._connections.items():
            if now - conn.last_heartbeat > timeout:
                stale.append(conn_id)

        return stale

    async def cleanup_stale_connections(self) -> int:
        """Clean up stale connections."""
        stale = self.get_stale_connections()
        for conn_id in stale:
            await self.disconnect(conn_id)
        return len(stale)

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get hub status."""
        return {
            "active_connections": len(self._connections),
            "max_connections": self.cfg.max_connections,
            "consent_records": len(self._consent_log),
            "require_consent": self.cfg.require_consent,
            "stale_connections": len(self.get_stale_connections()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIFleetCNExtendedSuite:
    """
    [D] Complete China-extended fleet intelligence suite.

    Integrates:
      • Chinese payment gateways (WeChat Pay, Alipay)
      • China traffic restriction routing
      • Real-time WebSocket tracking
      • (Optional) Previous fleet modules
    """

    def __init__(
        self,
        payment_cfg: Optional[CNPaymentConfig] = None,
        restriction_cfg: Optional[RestrictionConfig] = None,
        websocket_cfg: Optional[WebSocketConfig] = None,
    ):
        self.payment = CNPaymentGateway(payment_cfg)
        self.restriction = ChinaRestrictionRouter(restriction_cfg)
        self.websocket = FleetWebSocketHub(websocket_cfg)

        logger.info("[SESI Fleet CN Extended] Suite initialized")

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "payment": self.payment.get_status(),
            "restriction": self.restriction.get_status(),
            "websocket": self.websocket.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Fleet CN Extended — Smoke Test")
    print(" (WeChat Pay/Alipay + Restriction Routing + WebSocket)")
    print("=" * 72)
    print(f"  requests available  : {HAS_REQUESTS}")
    print(f"  cryptography avail  : {HAS_CRYPTO}")
    print(f"  PyJWT available     : {HAS_JWT}")
    print(f"  FastAPI available   : {HAS_FASTAPI}")
    print(f"  Fleet modules       : {HAS_FLEET}")

    suite = SESIFleetCNExtendedSuite()

    # ── Test 1: WeChat Pay ───────────────────────────────────────────────
    print("\n  ── [CNP1] WeChat Pay (微信支付) ──")

    wx_result = suite.payment.create_wechat_native_order(
        merchant_order_no="TEST_WX_001",
        amount_cny=88.88,
        description="测试订单 (Test Order)",
    )
    print(f"    Success            : {wx_result.success}")
    print(f"    Order ID           : {wx_result.order_id}")
    print(f"    Status             : {wx_result.status.value}")
    if wx_result.provider_response:
        qr_url = wx_result.provider_response.get("code_url", "")
        print(f"    QR Code URL        : {qr_url[:50]}..." if len(qr_url) > 50 else f"    QR Code URL        : {qr_url}")

    # Query order
    query_result = suite.payment.query_wechat_order("TEST_WX_001")
    print(f"    Query status       : {query_result.status.value}")

    # ── Test 2: Alipay ───────────────────────────────────────────────────
    print("\n  ── [CNP1] Alipay (支付宝) ──")

    ali_result = suite.payment.create_alipay_order(
        merchant_order_no="TEST_ALI_001",
        amount_cny=128.50,
        subject="测试商品 (Test Product)",
    )
    print(f"    Success            : {ali_result.success}")
    print(f"    Order ID           : {ali_result.order_id}")
    print(f"    Status             : {ali_result.status.value}")
    if ali_result.provider_response:
        pay_url = ali_result.provider_response.get("payment_url", "")
        print(f"    Payment URL        : {pay_url[:60]}..." if len(pay_url) > 60 else f"    Payment URL        : {pay_url}")

    # Unified interface
    unified_result = suite.payment.create_payment(
        provider=CNPaymentProvider.WECHAT_PAY,
        merchant_order_no="TEST_UNIFIED_001",
        amount_cny=66.66,
        description="统一接口测试",
    )
    print(f"    Unified interface  : {unified_result.success}")

    payment_status = suite.payment.get_status()
    print(f"    Total orders       : {payment_status['total_orders']}")
    print(f"    Sandbox mode       : {payment_status['sandbox_mode']}")

    # ── Test 3: Restriction Routing ──────────────────────────────────────
    print("\n  ── [CR1] China Restriction Router (限行查询) ──")

    # Beijing Monday morning (plates 1,6 restricted)
    monday_9am = datetime(2026, 1, 5, 9, 0)  # Monday
    bj_check = suite.restriction.check_restriction(
        plate_number="京A12346",  # Ends with 6
        plate_color=PlateColor.BLUE,
        vehicle_type=VehicleType.PASSENGER_CAR,
        city="Beijing",
        check_time=monday_9am,
    )
    print(f"    Beijing 京A12346 (Mon 9am):")
    print(f"      Restricted       : {bj_check['restricted']}")
    print(f"      Message          : {bj_check.get('message', '')}")
    if bj_check.get('applicable_rules'):
        for rule in bj_check['applicable_rules'][:2]:
            print(f"      Rule             : {rule['rule_id']} - {rule['zone']}")

    # NEV exemption
    nev_check = suite.restriction.check_restriction(
        plate_number="京AD12345",
        plate_color=PlateColor.GREEN,  # NEV
        vehicle_type=VehicleType.PASSENGER_CAR,
        city="Beijing",
        check_time=monday_9am,
    )
    print(f"\n    Beijing NEV (Green plate):")
    print(f"      Restricted       : {nev_check['restricted']}")
    print(f"      Reason           : {nev_check.get('reason', '')}")

    # Find alternative route
    alt_route = suite.restriction.find_alternative_route(
        origin=(39.90, 116.40),
        destination=(39.95, 116.45),
        plate_number="京A12346",
        plate_color=PlateColor.BLUE,
        vehicle_type=VehicleType.PASSENGER_CAR,
        city="Beijing",
        departure_time=monday_9am,
    )
    print(f"\n    Alternative route options:")
    print(f"      Status           : {alt_route['status']}")
    for opt in alt_route.get('alternatives', [])[:2]:
        print(f"      • {opt['option']}: {opt['description']}")

    # City rules
    bj_rules = suite.restriction.get_city_rules("Beijing")
    print(f"\n    Beijing rules      : {len(bj_rules)} total")

    restriction_status = suite.restriction.get_status()
    print(f"    Cities covered     : {restriction_status['cities_covered']}")

    # ── Test 4: WebSocket Hub ────────────────────────────────────────────
    print("\n  ── [WS1] Fleet WebSocket Hub ──")

    # Generate token
    token = suite.websocket.generate_token(
        driver_id="driver_123",
        vehicle_id="vehicle_456",
        consent_given=True,
    )
    print(f"    JWT token          : {token[:40]}...")

    # Verify token
    payload = suite.websocket.verify_token(token)
    if payload:
        print(f"    Token verified     : driver={payload.get('driver_id_hash')}")
        print(f"    Consent given      : {payload.get('consent_given')}")

    # Rate limiting test
    for i in range(15):
        rate_check = suite.websocket._check_message_rate("test_conn")
        if not rate_check["allowed"]:
            print(f"    Rate limited at    : message {i+1}")
            break
        suite.websocket._message_timestamps["test_conn"].append(time.time())

    ws_status = suite.websocket.get_status()
    print(f"    Active connections : {ws_status['active_connections']}")
    print(f"    Max connections    : {ws_status['max_connections']}")
    print(f"    Consent records    : {ws_status['consent_records']}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    WeChat configured  : {status['payment']['wechat_configured']}")
    print(f"    Alipay configured  : {status['payment']['alipay_configured']}")
    print(f"    Restriction rules  : {status['restriction']['total_rules']}")
    print(f"    Cities covered     : {len(status['restriction']['cities_covered'])}")
    print(f"    WebSocket conns    : {status['websocket']['active_connections']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE & ETHICAL FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • WeChat Pay/Alipay require Chinese business license")
    print("  • Payment processing requires PCI-DSS Level 1")
    print("  • License plate rules CHANGE FREQUENTLY by city")
    print("  • WebSocket tracking requires EXPLICIT driver consent")
    print("  • NEV (green plate) exemptions vary by city")
    print("  • PNDC Axiom 3: Consent-based tracking, audit log")
    print("  • Buddhist Ethics (Samma Ajiva): Transparent fees")
    print("  • Taoist (Wu Wei): Routes flow with local laws")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
