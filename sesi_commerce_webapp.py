"""
=============================================================================
SESI COMMERCE WEB APP — PAYMENTS + INVENTORY + MULTILINGUAL + LOGISTICS
Modern Trade Local + Long-distance Marketplace + Goods/Food Delivery Only
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026

IMPORTANT POLICY
----------------
• GOODS / FOOD DELIVERY ONLY
• NO passenger transport
• NO taxi / motorcycle taxi service
• Driver recruitment is opt-in only
• Background screening requires explicit consent
• No automated adverse employment decision
• Human review + appeal required
• No scraping private social network data
• Use official APIs and consent-based lead forms only

Run:
    pip install fastapi uvicorn requests
    uvicorn sesi_commerce_webapp:app --reload

Smoke test:
    python sesi_commerce_webapp.py
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sesi.commerce.webapp")
logging.basicConfig(level=logging.INFO)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# =============================================================================
# ENUMS
# =============================================================================

class BusinessMode(str, Enum):
    LOCAL_MODERN_TRADE = "local_modern_trade"       # บริการใกล้บ้าน
    LONG_DISTANCE_MARKETPLACE = "long_distance_marketplace"
    FOOD_DELIVERY_C2C = "food_delivery_c2c"         # รายย่อยขายอาหารเอง
    GOODS_DELIVERY_C2C = "goods_delivery_c2c"


class PaymentProvider(str, Enum):
    STRIPE = "stripe"
    PAYPAL = "paypal"
    PROMPTPAY = "promptpay"


class LogisticsProvider(str, Enum):
    DHL = "dhl"
    FEDEX = "fedex"
    UPS = "ups"
    USPS = "usps"
    THAILAND_POST = "thailand_post"
    KERRY = "kerry"
    FLASH = "flash"
    J_AND_T = "j_and_t"
    SF_EXPRESS = "sf_express"
    CAINIAO = "cainiao"
    JD_LOGISTICS = "jd_logistics"
    YAMATO = "yamato"
    SAGAWA = "sagawa"
    CJ_LOGISTICS = "cj_logistics"
    CDEK = "cdek"
    DPD = "dpd"
    GLS = "gls"
    LOCAL_COURIER = "local_courier"


class ScreeningStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    CONSENT_REQUIRED = "consent_required"
    PENDING_VENDOR = "pending_vendor"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    CLEAR_FOR_GOODS_DELIVERY = "clear_for_goods_delivery"
    NOT_CLEARED = "not_cleared"
    APPEAL_AVAILABLE = "appeal_available"


class ProductCategory(str, Enum):
    FOOD = "food"
    BEVERAGE = "beverage"
    FRUIT = "fruit"
    VEGAN = "vegan"
    ORGANIC = "organic"
    FRESH = "fresh"
    FROZEN = "frozen"
    SNACK = "snack"
    HOUSEHOLD = "household"
    PERSONAL_CARE = "personal_care"
    ELECTRONICS = "electronics"
    FASHION = "fashion"
    PET = "pet"
    HEALTH = "health"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Product:
    product_id: str
    name: str
    category: ProductCategory
    description: str
    price: float
    cost: float
    currency: str = "THB"
    stock: int = 0
    perishable: bool = False
    shelf_life_days: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentIntent:
    payment_id: str
    provider: PaymentProvider
    amount: float
    currency: str
    status: str
    checkout_url: Optional[str] = None
    qr_payload: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InventoryPolicy:
    lead_time_days: float = 3.0
    daily_demand_mean: float = 10.0
    daily_demand_std: float = 3.0
    ordering_cost: float = 100.0
    holding_cost_per_unit_year: float = 10.0
    service_z_score: float = 1.65  # ~95%


@dataclass
class InventoryRecommendation:
    product_id: str
    reorder_point: float
    safety_stock: float
    eoq: float
    recommended_order_qty: float
    fefo_warning: Optional[str] = None


@dataclass
class DriverApplicant:
    applicant_id: str
    display_name: str
    source_platform: str
    social_handle_hash: str
    consent_given: bool = False
    service_type: str = "goods_delivery_only"
    documents: Dict[str, Any] = field(default_factory=dict)
    screening_status: ScreeningStatus = ScreeningStatus.NOT_REQUESTED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogisticsQuote:
    provider: LogisticsProvider
    origin: str
    destination: str
    weight_kg: float
    distance_km: float
    price: float
    eta_days: float
    cold_chain: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PAYMENT GATEWAYS
# =============================================================================

class PaymentGatewayRouter:
    """
    Unified payment interface.
    Production:
      • Stripe: PaymentIntent API
      • PayPal: Orders API
      • PromptPay: through bank / Opn / 2C2P / Omise / KBank API
    """

    def __init__(
        self,
        stripe_key: Optional[str] = None,
        paypal_client_id: Optional[str] = None,
        paypal_secret: Optional[str] = None,
        promptpay_provider_key: Optional[str] = None,
    ):
        self.stripe_key = stripe_key
        self.paypal_client_id = paypal_client_id
        self.paypal_secret = paypal_secret
        self.promptpay_provider_key = promptpay_provider_key

    def create_payment(
        self,
        provider: PaymentProvider,
        amount: float,
        currency: str,
        order_id: str,
    ) -> PaymentIntent:
        if provider == PaymentProvider.STRIPE:
            return self._stripe_payment(amount, currency, order_id)
        if provider == PaymentProvider.PAYPAL:
            return self._paypal_payment(amount, currency, order_id)
        if provider == PaymentProvider.PROMPTPAY:
            return self._promptpay_payment(amount, currency, order_id)
        raise ValueError(f"Unsupported provider: {provider}")

    def _stripe_payment(self, amount: float, currency: str, order_id: str) -> PaymentIntent:
        # Production: use stripe.PaymentIntent.create(...)
        return PaymentIntent(
            payment_id=f"stripe_{uuid.uuid4().hex[:10]}",
            provider=PaymentProvider.STRIPE,
            amount=amount,
            currency=currency,
            status="requires_checkout",
            checkout_url=f"https://checkout.stripe.com/pay/mock_{order_id}",
            metadata={"order_id": order_id},
        )

    def _paypal_payment(self, amount: float, currency: str, order_id: str) -> PaymentIntent:
        # Production: PayPal Orders API
        return PaymentIntent(
            payment_id=f"paypal_{uuid.uuid4().hex[:10]}",
            provider=PaymentProvider.PAYPAL,
            amount=amount,
            currency=currency,
            status="requires_checkout",
            checkout_url=f"https://paypal.com/checkoutnow?token=mock_{order_id}",
            metadata={"order_id": order_id},
        )

    def _promptpay_payment(self, amount: float, currency: str, order_id: str) -> PaymentIntent:
        # Production: bank / Opn / 2C2P PromptPay QR API
        # Do not generate banking QR manually without certified provider.
        return PaymentIntent(
            payment_id=f"promptpay_{uuid.uuid4().hex[:10]}",
            provider=PaymentProvider.PROMPTPAY,
            amount=amount,
            currency="THB",
            status="requires_qr_scan",
            qr_payload=f"PROVIDER_QR_REQUIRED::{order_id}::{amount:.2f}",
            metadata={
                "order_id": order_id,
                "note": "Use certified Thai bank/payment provider for real PromptPay QR",
            },
        )


# =============================================================================
# INVENTORY OPTIMIZATION
# =============================================================================

class InventoryOptimizationEngine:
    """
    Inventory engine:
      • Safety Stock = z σ_daily sqrt(L)
      • Reorder Point = μ_daily L + Safety Stock
      • EOQ = sqrt(2DS/H)
      • FEFO for perishable goods
    """

    def optimize(
        self,
        product: Product,
        policy: InventoryPolicy,
        annual_demand: float = 3650.0,
    ) -> InventoryRecommendation:
        safety_stock = (
            policy.service_z_score
            * policy.daily_demand_std
            * math.sqrt(policy.lead_time_days)
        )
        reorder_point = policy.daily_demand_mean * policy.lead_time_days + safety_stock

        eoq = math.sqrt(
            2 * annual_demand * policy.ordering_cost
            / max(policy.holding_cost_per_unit_year, 1e-8)
        )

        recommended_qty = 0.0
        if product.stock <= reorder_point:
            recommended_qty = eoq

        fefo_warning = None
        if product.perishable:
            fefo_warning = (
                "Use FEFO: First-Expired-First-Out. "
                "Check expiry dates and cold-chain temperature."
            )

        return InventoryRecommendation(
            product_id=product.product_id,
            reorder_point=round(reorder_point, 2),
            safety_stock=round(safety_stock, 2),
            eoq=round(eoq, 2),
            recommended_order_qty=round(recommended_qty, 2),
            fefo_warning=fefo_warning,
        )


# =============================================================================
# MULTI-LANGUAGE PRODUCT DESCRIPTION GENERATOR
# =============================================================================

class MultiLanguageDescriptionGenerator:
    """
    Product description generator.
    Production can connect to LLM / Transformers.
    This template version is deterministic, safe, and cheap.
    """

    SUPPORTED_LANGS = ["th", "en", "zh", "ja", "ko", "ru", "de", "fr", "es", "ar"]

    def generate(
        self,
        product: Product,
        languages: List[str],
        tone: str = "trustworthy",
    ) -> Dict[str, Dict[str, str]]:
        results = {}
        for lang in languages:
            if lang not in self.SUPPORTED_LANGS:
                continue
            results[lang] = self._template(product, lang, tone)
        return results

    def _template(self, product: Product, lang: str, tone: str) -> Dict[str, str]:
        if lang == "th":
            return {
                "title": product.name,
                "short": f"{product.name} คุณภาพดี เหมาะสำหรับ kebutuhan اليومية",
                "long": (
                    f"{product.name} Kategorี {product.category.value} "
                    f"คัดสรรเพื่อคุณภาพ ความสดใหม่ และความปลอดภัย "
                    f"เหมาะสำหรับการสั่งซื้อออนไลน์และบริการใกล้บ้าน"
                ),
                "seo_keywords": f"{product.name}, {product.category.value}, ซื้อออนไลน์, ส่งใกล้บ้าน",
            }
        if lang == "zh":
            return {
                "title": product.name,
                "short": f"{product.name}，品质可靠，适合日常使用。",
                "long": f"{product.name} 属于 {product.category.value} 类别，适合线上购买和本地配送。",
                "seo_keywords": f"{product.name}, 网上购物, 本地配送, {product.category.value}",
            }
        if lang == "ja":
            return {
                "title": product.name,
                "short": f"{product.name} は毎日の利用に適した高品質商品です。",
                "long": f"{product.name} は {product.category.value} カテゴリの商品で、オンライン注文と地域配送に対応します。",
                "seo_keywords": f"{product.name}, オンライン购物, 地域配送",
            }
        if lang == "ko":
            return {
                "title": product.name,
                "short": f"{product.name} 은 일상 사용에 적합한 고품질 상품입니다.",
                "long": f"{product.name} 은 {product.category.value} 카테고리 상품으로 온라인 주문과 지역 배송에 적합합니다.",
                "seo_keywords": f"{product.name}, 온라인 쇼핑, 지역 배송",
            }

        # Default English
        return {
            "title": product.name,
            "short": f"{product.name} — quality-selected for everyday needs.",
            "long": (
                f"{product.name} in the {product.category.value} category, "
                f"selected for quality, convenience, and reliable delivery. "
                f"Suitable for local commerce and online marketplace orders."
            ),
            "seo_keywords": f"{product.name}, {product.category.value}, online shopping, local delivery",
        }


# =============================================================================
# LOGISTICS CONNECTOR
# =============================================================================

class GlobalLogisticsRouter:
    """
    Unified logistics quote interface.
    Production:
      • DHL/FedEx/UPS APIs
      • SF Express/Cainiao/JD China
      • Yamato/Sagawa Japan
      • CJ Korea
      • CDEK Russia
      • DPD/GLS Europe
      • Thailand Post/Kerry/Flash/J&T Thailand
    """

    PROVIDERS_BY_REGION = {
        "TH": [LogisticsProvider.THAILAND_POST, LogisticsProvider.KERRY, LogisticsProvider.FLASH, LogisticsProvider.J_AND_T],
        "CN": [LogisticsProvider.SF_EXPRESS, LogisticsProvider.CAINIAO, LogisticsProvider.JD_LOGISTICS],
        "US": [LogisticsProvider.UPS, LogisticsProvider.FEDEX, LogisticsProvider.USPS],
        "EU": [LogisticsProvider.DHL, LogisticsProvider.DPD, LogisticsProvider.GLS],
        "JP": [LogisticsProvider.YAMATO, LogisticsProvider.SAGAWA],
        "KR": [LogisticsProvider.CJ_LOGISTICS],
        "RU": [LogisticsProvider.CDEK],
        "GLOBAL": [LogisticsProvider.DHL, LogisticsProvider.FEDEX, LogisticsProvider.UPS],
    }

    def quote(
        self,
        origin_country: str,
        destination_country: str,
        weight_kg: float,
        distance_km: float,
        cold_chain: bool = False,
    ) -> List[LogisticsQuote]:
        providers = []
        if origin_country == destination_country:
            providers += self.PROVIDERS_BY_REGION.get(origin_country, [])
        providers += self.PROVIDERS_BY_REGION["GLOBAL"]

        quotes = []
        for provider in list(dict.fromkeys(providers)):
            base = 30 + weight_kg * 12 + distance_km * 0.8
            if cold_chain:
                base *= 1.5
            eta = max(1, distance_km / 600 + (2 if origin_country != destination_country else 0))
            quotes.append(LogisticsQuote(
                provider=provider,
                origin=origin_country,
                destination=destination_country,
                weight_kg=weight_kg,
                distance_km=distance_km,
                price=round(base, 2),
                eta_days=round(eta, 1),
                cold_chain=cold_chain,
            ))

        return sorted(quotes, key=lambda q: q.price)


# =============================================================================
# SOCIAL DRIVER RECRUITMENT + SAFETY SCREENING
# =============================================================================

class SocialDriverRecruiter:
    """
    Opt-in recruitment from social networks.
    NO scraping private profiles.
    Facebook API can be used for:
      • Page posts
      • Lead ads with consent
      • Messenger opt-in
    """

    def create_campaign_link(self, platform: str, campaign_name: str) -> str:
        encoded = campaign_name.replace(" ", "%20")
        return f"https://example.com/driver-apply?source={platform}&campaign={encoded}"

    def intake_applicant(
        self,
        display_name: str,
        source_platform: str,
        social_handle: str,
        consent_given: bool,
    ) -> DriverApplicant:
        social_hash = hashlib.sha256(social_handle.encode()).hexdigest()[:16]

        applicant = DriverApplicant(
            applicant_id=f"driver_{uuid.uuid4().hex[:10]}",
            display_name=display_name,
            source_platform=source_platform,
            social_handle_hash=social_hash,
            consent_given=consent_given,
            service_type="goods_delivery_only",
        )
        return applicant


class ComplianceScreeningOrchestrator:
    """
    International screening workflow.

    IMPORTANT:
      • Requires applicant consent
      • Uses authorized vendors only
      • No scraping criminal databases
      • No automatic rejection
      • Human review required
      • Appeal process required
    """

    def request_screening(
        self,
        applicant: DriverApplicant,
        countries: List[str],
        vendor: str = "authorized_background_provider",
    ) -> Dict[str, Any]:
        if not applicant.consent_given:
            applicant.screening_status = ScreeningStatus.CONSENT_REQUIRED
            return {
                "status": applicant.screening_status.value,
                "message": "Explicit consent required before screening",
            }

        if applicant.service_type != "goods_delivery_only":
            return {
                "status": "blocked",
                "message": "Passenger transport is not supported by this platform",
            }

        applicant.screening_status = ScreeningStatus.HUMAN_REVIEW_REQUIRED

        return {
            "status": applicant.screening_status.value,
            "applicant_id": applicant.applicant_id,
            "countries": countries,
            "vendor": vendor,
            "requires_human_review": True,
            "appeal_available": True,
            "notes": [
                "No automated adverse decision",
                "Jurisdiction-specific compliance required",
                "Only authorized background-check providers should be used",
            ],
        }


# =============================================================================
# WEB APP
# =============================================================================

if HAS_FASTAPI:
    app = FastAPI(title="SESI Commerce Web App", version="1.0.0")

    payment_router = PaymentGatewayRouter()
    inventory_engine = InventoryOptimizationEngine()
    description_generator = MultiLanguageDescriptionGenerator()
    logistics_router = GlobalLogisticsRouter()
    recruiter = SocialDriverRecruiter()
    screening = ComplianceScreeningOrchestrator()

    PRODUCTS: Dict[str, Product] = {}
    DRIVERS: Dict[str, DriverApplicant] = {}

    class ProductIn(BaseModel):
        name: str
        category: ProductCategory
        description: str = ""
        price: float
        cost: float
        currency: str = "THB"
        stock: int = 0
        perishable: bool = False
        shelf_life_days: Optional[int] = None

    class PaymentIn(BaseModel):
        provider: PaymentProvider
        amount: float
        currency: str = "THB"
        order_id: str

    class DescriptionIn(BaseModel):
        product_id: str
        languages: List[str] = ["th", "en", "zh", "ja", "ko"]

    class LogisticsIn(BaseModel):
        origin_country: str
        destination_country: str
        weight_kg: float
        distance_km: float
        cold_chain: bool = False

    class DriverIn(BaseModel):
        display_name: str
        source_platform: str
        social_handle: str
        consent_given: bool

    @app.get("/")
    def root():
        return {
            "name": "SESI Commerce Web App",
            "modes": [m.value for m in BusinessMode],
            "policy": "Goods/Food delivery only. No passenger transport.",
        }

    @app.post("/products")
    def create_product(inp: ProductIn):
        product = Product(
            product_id=f"prod_{uuid.uuid4().hex[:10]}",
            **inp.dict(),
        )
        PRODUCTS[product.product_id] = product
        return product.__dict__

    @app.post("/payments/create")
    def create_payment(inp: PaymentIn):
        return payment_router.create_payment(
            inp.provider, inp.amount, inp.currency, inp.order_id
        ).__dict__

    @app.post("/inventory/optimize/{product_id}")
    def optimize_inventory(product_id: str):
        if product_id not in PRODUCTS:
            raise HTTPException(404, "Product not found")
        product = PRODUCTS[product_id]
        policy = InventoryPolicy()
        return inventory_engine.optimize(product, policy).__dict__

    @app.post("/descriptions/generate")
    def generate_descriptions(inp: DescriptionIn):
        if inp.product_id not in PRODUCTS:
            raise HTTPException(404, "Product not found")
        return description_generator.generate(PRODUCTS[inp.product_id], inp.languages)

    @app.post("/logistics/quote")
    def logistics_quote(inp: LogisticsIn):
        quotes = logistics_router.quote(**inp.dict())
        return [q.__dict__ for q in quotes]

    @app.post("/drivers/apply")
    def driver_apply(inp: DriverIn):
        applicant = recruiter.intake_applicant(**inp.dict())
        DRIVERS[applicant.applicant_id] = applicant
        return applicant.__dict__

    @app.post("/drivers/screen/{applicant_id}")
    def driver_screen(applicant_id: str, countries: List[str]):
        if applicant_id not in DRIVERS:
            raise HTTPException(404, "Applicant not found")
        return screening.request_screening(DRIVERS[applicant_id], countries)

else:
    app = None


# =============================================================================
# SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 72)
    print("SESI Commerce Web App — Smoke Test")
    print("=" * 72)
    print(f"FastAPI available   : {HAS_FASTAPI}")
    print(f"requests available  : {HAS_REQUESTS}")

    product = Product(
        product_id="prod_demo",
        name="Organic Vegan Matcha Latte",
        category=ProductCategory.VEGAN,
        description="Plant-based matcha latte",
        price=89,
        cost=35,
        stock=8,
        perishable=True,
        shelf_life_days=5,
    )

    payment = PaymentGatewayRouter().create_payment(
        PaymentProvider.PROMPTPAY, 89, "THB", "order_demo"
    )
    print("\nPayment:", payment)

    inv = InventoryOptimizationEngine().optimize(product, InventoryPolicy())
    print("\nInventory:", inv)

    desc = MultiLanguageDescriptionGenerator().generate(product, ["th", "en", "zh", "ja", "ko"])
    print("\nDescriptions:", list(desc.keys()))

    quotes = GlobalLogisticsRouter().quote("TH", "TH", 1.2, 12, cold_chain=True)
    print("\nLogistics quotes:", [q.provider.value for q in quotes])

    applicant = SocialDriverRecruiter().intake_applicant(
        "Demo Driver", "facebook", "@demo_driver", consent_given=True
    )
    print("\nDriver applicant:", applicant)

    screen_result = ComplianceScreeningOrchestrator().request_screening(
        applicant, ["TH", "JP", "US"]
    )
    print("\nScreening:", screen_result)

    print("\nPolicy: Goods/food delivery only. No passenger transport.")
    print("=" * 72)


if __name__ == "__main__":
    smoke_test()
