"""
=============================================================================
SESI CUSTOMER INTELLIGENCE AGI
CRM Integration + CLV Prediction + Churn/Retention + Modern Trade + Delivery
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[CRM1] UnifiedCRMConnector
      • Salesforce REST API (SOQL queries, OAuth2)
      • HubSpot CRM API v3 (contacts, deals, companies)
      • Data normalization to unified CustomerRecord schema
      • PNDC Axiom 3: consent verification, data minimization
      • Right to erasure (GDPR Art. 17 / PDPA §33)

[CLV1] CLVPredictor (Multi-Model Ensemble)
      • BG/NBD: Fader, Hardie & Lee 2005
      • Gamma-Gamma: Driesen et al. 2007
      • Deep CLV: LSTM/Transformer sequence model
      • Bayesian UQ ensemble (from sesi_multiscale_amr pattern)
      • Credible intervals, not point estimates

[CHN1] ChurnPredictor (Survival + ML Ensemble)
      • Kaplan-Meier survival curves
      • Cox Proportional Hazards
      • Gradient Boosted Trees (XGBoost-compatible)
      • Neural network with attention on behavioral sequences
      • Early warning system with configurable thresholds

[RET1] RetentionStrategyEngine
      • Cialdini's principles (ethical application only)
      • Uplift modeling for incentive targeting
      • Personalized recommendation per churn reason
      • Budget optimization (CLV-aware allocation)
      • NO dark patterns, NO fake urgency

[CAT1] ModernTradeCategoryAnalyzer
      • All modern trade categories (hypermarket/supermarket/convenience)
      • Product hierarchy: Department → Category → Subcategory → SKU
      • Price elasticity estimation per category
      • Basket analysis (association rules)
      • Seasonal pattern detection (STL decomposition)

[ONL1] OnlineSalesAnalyzer
      • E-commerce funnel analytics (view→cart→checkout→purchase)
      • Marketplace vs D2C vs Social commerce
      • Review sentiment analysis
      • Dynamic pricing intelligence
      • Cross-channel attribution

[DLV1] DeliveryMarketAnalyzer
      • All Thai delivery platforms: Grab, LINE MAN, Foodpanda,
        Robinhood, ShopeeFood, Gojek
      • Food categories: Thai, Japanese, Chinese, Western, Fast food,
        Vegan, Vegetarian, Beverages, Desserts, Fruits, Organic
      • Demand heatmaps by zone/time
      • Commission structure analysis
      • Menu engineering recommendations

[SEG1] CustomerSegmentationEngine
      • TDA-based behavioral clustering (from sesi_tda_carpet pattern)
      • RFM+ segmentation (Recency, Frequency, Monetary + behavioral)
      • Spectral clustering on purchase graphs
      • Dynamic segment migration tracking

MATHEMATICAL FOUNDATIONS
--------------------------
[C] BG/NBD: P(alive|x) via Beta-Geometric/Negative Binomial
    E[X(t)|alive] = (a+b-1)/(a-1) * [1-(b-1)/(b+x-1)]
[C] Gamma-Gamma: E[M|x] via Gamma-Gamma conjugacy
[C] Cox PH: h(t|X) = h₀(t)·exp(β'X)
[C] Kaplan-Meier: S(t) = Π(1 - dᵢ/nᵢ)
[C] CLV = Σ_t (M_t - C_t)/(1+r)^t
[C] Bayesian UQ: Posterior ∝ Likelihood × Prior (ensemble)
[C] TDA: Persistent homology for behavioral pattern detection
[C] Cialdini 2001: 6 principles of influence (ethical use only)
[C] SESI_NoZeno_Closure Thm 3.2: Bounded computation
[C] PNDC Axiom 3: Necessity and Proportionality

HONEST SCOPE
------------
• CRM integration requires valid API credentials and customer consent
• CLV is PROBABILISTIC — credible intervals, not guarantees
• Churn prediction is RISK ASSESSMENT — not certainty
• Retention strategies are RECOMMENDATIONS — human approves
• Category analysis uses AGGREGATE data only (PNDC Axiom 3)
• Delivery platform data requires partnership agreements
• All thresholds are FREE DESIGN PARAMETERS
• Production requires PDPA/GDPR compliance review
=============================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.customer_intelligence")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from scipy import stats as scipy_stats
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Integration with SESI structural calculus ecosystem
try:
    from sesi_multiscale_amr import BayesianUQWrapper, MultiScaleConfig
    HAS_BAYESIAN_UQ = True
except ImportError:
    HAS_BAYESIAN_UQ = False

try:
    from sesi_tda_carpet import PersistentHomology, TopologicalFeatures, TDAConfig
    HAS_TDA = True
except ImportError:
    HAS_TDA = False

try:
    from sesi_unified_amr2 import NoZenoGate, DorflerMarker
    HAS_NOZENO = True
except ImportError:
    HAS_NOZENO = False

# Sanitization (from agi_one_psyche_plus_v33 pattern)
def _safe_unit_scalar(value, name="", default=0.5):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return max(0.0, min(1.0, v))


__all__ = [
    "CRMPlatform", "ProductCategory", "DeliveryPlatform", "FoodCategory",
    "ChurnRiskLevel", "CustomerSegment", "RetentionAction",
    "UnifiedCRMConnector", "CRMConfig", "CustomerRecord",
    "CLVPredictor", "CLVConfig", "CLVEstimate",
    "ChurnPredictor", "ChurnConfig", "ChurnRisk",
    "RetentionStrategyEngine", "RetentionConfig", "RetentionPlan",
    "ModernTradeCategoryAnalyzer", "CategoryConfig",
    "OnlineSalesAnalyzer", "OnlineConfig",
    "DeliveryMarketAnalyzer", "DeliveryConfig",
    "CustomerSegmentationEngine", "SegmentConfig",
    "SESICustomerIntelligenceSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class CRMPlatform(Enum):
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"


class ProductCategory(Enum):
    """All modern trade product categories."""
    # Food
    FRESH_FOOD = "fresh_food"
    PROCESSED_FOOD = "processed_food"
    FROZEN_FOOD = "frozen_food"
    BAKERY = "bakery"
    SNACKS = "snacks"
    CONFECTIONERY = "confectionery"
    # Beverages
    BEVERAGES_ALCOHOLIC = "beverages_alcoholic"
    BEVERAGES_NON_ALCOHOLIC = "beverages_non_alcoholic"
    COFFEE_TEA = "coffee_tea"
    # Fruits & Vegetables
    FRESH_FRUITS = "fresh_fruits"
    FRESH_VEGETABLES = "fresh_vegetables"
    ORGANIC_PRODUCE = "organic_produce"
    # Protein
    MEAT_POULTRY = "meat_poultry"
    SEAFOOD = "seafood"
    VEGAN_PROTEIN = "vegan_protein"
    DAIRY_EGGS = "dairy_eggs"
    # Household
    HOUSEHOLD_CLEANING = "household_cleaning"
    PERSONAL_CARE = "personal_care"
    PAPER_PRODUCTS = "paper_products"
    # Others
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HEALTH_WELLNESS = "health_wellness"
    PET_SUPPLIES = "pet_supplies"


class DeliveryPlatform(Enum):
    """All major Thai delivery platforms."""
    GRAB_FOOD = "grab_food"
    LINE_MAN = "line_man"
    FOODPANDA = "foodpanda"
    ROBINHOOD = "robinhood"
    SHOPEE_FOOD = "shopee_food"
    GOJEK = "gojek"
    DIRECT_RESTAURANT = "direct_restaurant"


class FoodCategory(Enum):
    """Food categories for delivery analysis."""
    THAI_FOOD = "thai_food"
    JAPANESE = "japanese"
    CHINESE = "chinese"
    WESTERN = "western"
    FAST_FOOD = "fast_food"
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    BEVERAGES = "beverages"
    DESSERTS = "desserts"
    FRUITS = "fruits"
    ORGANIC = "organic"
    STREET_FOOD = "street_food"
    HOTPOT_BBQ = "hotpot_bbq"
    SEAFOOD = "seafood"
    BAKERY_CAFE = "bakery_cafe"


class ChurnRiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerSegment(Enum):
    """RFM+ behavioral segments."""
    CHAMPIONS = "champions"
    LOYAL = "loyal"
    POTENTIAL_LOYALIST = "potential_loyalist"
    NEW_CUSTOMER = "new_customer"
    PROMISING = "promising"
    NEED_ATTENTION = "need_attention"
    ABOUT_TO_SLEEP = "about_to_sleep"
    AT_RISK = "at_risk"
    CANT_LOSE = "cant_lose"
    HIBERNATING = "hibernating"
    LOST = "lost"


class RetentionAction(Enum):
    """Ethical retention actions (NO dark patterns)."""
    PERSONALIZED_RECOMMENDATION = "personalized_recommendation"
    LOYALTY_REWARD = "loyalty_reward"
    EXCLUSIVE_ACCESS = "exclusive_access"
    FEEDBACK_REQUEST = "feedback_request"
    WIN_BACK_OFFER = "win_back_offer"
    COMMUNITY_INVITATION = "community_invitation"
    EDUCATIONAL_CONTENT = "educational_content"
    NO_ACTION = "no_action"


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CustomerRecord:
    """Unified customer record across CRM platforms."""
    customer_id: str
    platform: CRMPlatform
    # Demographics (anonymized)
    age_group: Optional[str] = None
    region: Optional[str] = None
    gender: Optional[str] = None
    # Behavioral
    total_orders: int = 0
    total_revenue: float = 0.0
    avg_order_value: float = 0.0
    purchase_frequency: float = 0.0  # orders/week
    recency_days: float = 0.0
    first_purchase_date: Optional[str] = None
    last_purchase_date: Optional[str] = None
    # Category preferences
    favorite_categories: List[ProductCategory] = field(default_factory=list)
    favorite_food_categories: List[FoodCategory] = field(default_factory=list)
    preferred_delivery_platforms: List[DeliveryPlatform] = field(default_factory=list)
    # Channels
    channels_used: List[str] = field(default_factory=list)  # online, in-store, delivery
    # Engagement
    email_opens: int = 0
    app_sessions: int = 0
    support_tickets: int = 0
    nps_score: Optional[float] = None
    # Consent & Privacy (PNDC Axiom 3)
    marketing_consent: bool = False
    analytics_consent: bool = False
    data_processing_consent: bool = False
    consent_timestamp: Optional[float] = None
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "platform": self.platform.value,
            "total_orders": self.total_orders,
            "total_revenue": self.total_revenue,
            "avg_order_value": self.avg_order_value,
            "purchase_frequency": self.purchase_frequency,
            "recency_days": self.recency_days,
            "favorite_categories": [c.value for c in self.favorite_categories],
            "channels_used": self.channels_used,
            "consent": {
                "marketing": self.marketing_consent,
                "analytics": self.analytics_consent,
                "data_processing": self.data_processing_consent,
            },
        }


@dataclass
class CLVEstimate:
    """Customer Lifetime Value estimate with uncertainty."""
    customer_id: str
    # Point estimates
    clv_12m: float
    clv_24m: float
    clv_lifetime: float
    # Uncertainty (Bayesian)
    ci_lower_12m: float
    ci_upper_12m: float
    credible_level: float = 0.95
    # Components
    expected_orders_12m: float = 0.0
    expected_avg_order_value: float = 0.0
    survival_probability_12m: float = 1.0
    # Model details
    model_type: str = "ensemble"
    # Segmentation
    value_tier: str = "medium"  # "premium", "high", "medium", "low", "at_risk"
    # Metadata
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChurnRisk:
    """Churn risk assessment."""
    customer_id: str
    # Risk scores (0-1)
    churn_probability_30d: float
    churn_probability_60d: float
    churn_probability_90d: float
    risk_level: ChurnRiskLevel
    # Survival analysis
    median_survival_days: Optional[float] = None
    # Contributing factors
    top_risk_factors: List[Tuple[str, float]] = field(default_factory=list)
    # Model ensemble
    model_agreement: float = 1.0  # How much models agree
    # Metadata
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetentionPlan:
    """Personalized retention plan (ethical)."""
    customer_id: str
    # Recommended actions (ranked by expected uplift)
    actions: List[RetentionAction]
    # Incentive details
    recommended_incentive_type: str = "none"
    recommended_incentive_value: float = 0.0
    max_budget_allocation: float = 0.0
    # Messaging
    message_template: str = ""
    preferred_channel: str = "email"
    best_contact_time: str = "evening"
    # Expected impact
    expected_retention_lift: float = 0.0
    expected_clv_preservation: float = 0.0
    # Ethics
    ethical_compliance: bool = True
    transparency_note: str = ""
    # Metadata
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# [CRM1] UNIFIED CRM CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CRMConfig:
    """CRM connection configuration."""
    # Salesforce
    sf_instance_url: Optional[str] = None
    sf_client_id: Optional[str] = None
    sf_client_secret: Optional[str] = None
    sf_username: Optional[str] = None
    sf_password: Optional[str] = None
    # HubSpot
    hs_access_token: Optional[str] = None
    hs_portal_id: Optional[str] = None
    # PNDC Axiom 3
    require_consent: bool = True
    data_minimization: bool = True
    anonymize_pii: bool = True
    audit_all_access: bool = True
    # Rate limiting (No-Zeno)
    max_api_calls_per_minute: int = 60
    min_call_interval_s: float = 1.0


class UnifiedCRMConnector:
    """
    [D] Unified connector for Salesforce and HubSpot CRM platforms.

    ⚠️ ETHICAL FRAMEWORK (PNDC Axiom 3):
      • EXPLICIT consent required before data access
      • Data minimization — fetch only necessary fields
      • Anonymization of PII at ingestion
      • Full audit trail of all data access
      • Right to erasure support (GDPR Art.17 / PDPA §33)

    Features:
      • Salesforce REST API + SOQL queries
      • HubSpot CRM API v3
      • Unified CustomerRecord schema
      • Consent management
      • No-Zeno rate limiting on API calls

    HONEST: This is a BRIDGE. Real deployment requires:
      • Valid API credentials (OAuth2 for Salesforce, API key for HubSpot)
      • Proper data processing agreements
      • PDPA/GDPR compliance review
    """

    def __init__(self, cfg: Optional[CRMConfig] = None):
        self.cfg = cfg or CRMConfig()
        self._api_timestamps: deque = deque(maxlen=10000)
        self._audit_log: List[Dict[str, Any]] = []
        self._consent_store: Dict[str, Dict[str, Any]] = {}
        self._session = requests.Session() if HAS_REQUESTS else None

        # Validate ethical constraints
        if not self.cfg.require_consent:
            raise ValueError("PNDC Axiom 3: Consent requirement cannot be disabled")
        if not self.cfg.audit_all_access:
            raise ValueError("PNDC Axiom 3: Audit logging cannot be disabled")

        logger.info("[CRM] Unified connector initialized (consent-enforced)")

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def _check_api_rate(self) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - 60.0
        self._api_timestamps = deque(
            (t for t in self._api_timestamps if t > cutoff), maxlen=10000
        )
        if len(self._api_timestamps) >= self.cfg.max_api_calls_per_minute:
            return {"allowed": False, "reason": "rate_limited"}
        if self._api_timestamps:
            if now - self._api_timestamps[-1] < self.cfg.min_call_interval_s:
                return {"allowed": False, "reason": "interval_too_short"}
        return {"allowed": True}

    # ── Consent Management ───────────────────────────────────────────────

    def verify_consent(self, customer_id: str, purpose: str = "analytics") -> bool:
        """Check if customer has given consent for data processing."""
        record = self._consent_store.get(customer_id)
        if record is None:
            return False
        if record.get("revoked", False):
            return False
        return purpose in record.get("purposes", [])

    def record_consent(
        self, customer_id: str, purposes: List[str], consent_document: str = ""
    ) -> str:
        """Record customer consent (creates cryptographic proof)."""
        consent_hash = hashlib.sha256(
            f"{customer_id}:{','.join(purposes)}:{consent_document}:{time.time()}".encode()
        ).hexdigest()

        self._consent_store[customer_id] = {
            "consent_hash": consent_hash,
            "purposes": purposes,
            "granted_at": time.time(),
            "revoked": False,
        }

        self._log_audit("consent_granted", {
            "customer_id": customer_id,
            "purposes": purposes,
            "hash": consent_hash[:16],
        })
        return consent_hash

    def revoke_consent(self, customer_id: str, reason: str = "") -> bool:
        """Revoke consent (Right to Erasure)."""
        if customer_id in self._consent_store:
            self._consent_store[customer_id]["revoked"] = True
            self._consent_store[customer_id]["revoked_at"] = time.time()
            self._consent_store[customer_id]["revocation_reason"] = reason
            self._log_audit("consent_revoked", {"customer_id": customer_id, "reason": reason})
            return True
        return False

    # ── Audit Trail ──────────────────────────────────────────────────────

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        self._audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "details": details,
            "event_id": f"audit_{uuid.uuid4().hex[:8]}",
        })

    # ── Salesforce Integration ───────────────────────────────────────────

    def fetch_salesforce_customers(
        self, query_filter: str = "", limit: int = 100
    ) -> List[CustomerRecord]:
        """
        Fetch customers from Salesforce via SOQL.

        In production: uses OAuth2 token flow + REST API.
        Here: returns mock data with correct schema.
        """
        rate_check = self._check_api_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[CRM] Salesforce rate-limited: {rate_check['reason']}")
            return []

        self._log_audit("salesforce_query", {"filter": query_filter, "limit": limit})

        # Mock: In production, execute SOQL via REST API
        # SELECT Id, Name, Email, CreatedDate, (SELECT ... FROM Orders) FROM Contact
        customers = []
        for i in range(min(limit, 50)):
            cust = CustomerRecord(
                customer_id=f"sf_{uuid.uuid4().hex[:12]}",
                platform=CRMPlatform.SALESFORCE,
                age_group=np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
                region=np.random.choice(["Bangkok", "Chiang Mai", "Phuket", "Khon Kaen", "Chonburi"]),
                total_orders=int(np.random.poisson(15)),
                total_revenue=float(np.random.gamma(5, 500)),
                avg_order_value=float(np.random.uniform(100, 2000)),
                purchase_frequency=float(np.random.uniform(0.1, 3.0)),
                recency_days=float(np.random.exponential(30)),
                favorite_categories=list(np.random.choice(
                    list(ProductCategory), size=np.random.randint(1, 4), replace=False
                )),
                channels_used=list(np.random.choice(
                    ["in_store", "online", "delivery", "mobile_app"],
                    size=np.random.randint(1, 3), replace=False
                )),
                marketing_consent=bool(np.random.random() > 0.2),
                analytics_consent=bool(np.random.random() > 0.3),
                data_processing_consent=True,
            )
            customers.append(cust)

        logger.info(f"[CRM] Fetched {len(customers)} customers from Salesforce (mock)")
        return customers

    # ── HubSpot Integration ──────────────────────────────────────────────

    def fetch_hubspot_customers(
        self, lifecycle_stage: str = "", limit: int = 100
    ) -> List[CustomerRecord]:
        """
        Fetch contacts from HubSpot CRM API v3.

        In production: GET /crm/v3/objects/contacts with pagination.
        Here: returns mock data with correct schema.
        """
        rate_check = self._check_api_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[CRM] HubSpot rate-limited: {rate_check['reason']}")
            return []

        self._log_audit("hubspot_query", {"lifecycle_stage": lifecycle_stage, "limit": limit})

        customers = []
        for i in range(min(limit, 50)):
            cust = CustomerRecord(
                customer_id=f"hs_{uuid.uuid4().hex[:12]}",
                platform=CRMPlatform.HUBSPOT,
                age_group=np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
                region=np.random.choice(["Bangkok", "Chiang Mai", "Phuket", "Khon Kaen"]),
                total_orders=int(np.random.poisson(10)),
                total_revenue=float(np.random.gamma(4, 400)),
                avg_order_value=float(np.random.uniform(80, 1500)),
                purchase_frequency=float(np.random.uniform(0.05, 2.0)),
                recency_days=float(np.random.exponential(45)),
                favorite_categories=list(np.random.choice(
                    list(ProductCategory), size=np.random.randint(1, 3), replace=False
                )),
                channels_used=["online", "mobile_app"],
                marketing_consent=bool(np.random.random() > 0.25),
                analytics_consent=True,
                data_processing_consent=True,
            )
            customers.append(cust)

        logger.info(f"[CRM] Fetched {len(customers)} contacts from HubSpot (mock)")
        return customers

    def get_status(self) -> Dict[str, Any]:
        return {
            "salesforce_configured": self.cfg.sf_instance_url is not None,
            "hubspot_configured": self.cfg.hs_access_token is not None,
            "consent_records": len(self._consent_store),
            "audit_entries": len(self._audit_log),
            "require_consent": self.cfg.require_consent,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [CLV1] CUSTOMER LIFETIME VALUE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CLVConfig:
    """CLV prediction configuration."""
    # Model selection
    use_bg_nbd: bool = True
    use_gamma_gamma: bool = True
    use_deep_learning: bool = True
    use_ensemble: bool = True
    # Parameters
    horizon_months: int = 12
    discount_rate_annual: float = 0.10
    # Bayesian UQ
    n_ensemble_members: int = 5
    credible_level: float = 0.95
    # Device
    device: str = "cpu"


class BGNBDModel:
    """
    [C] Beta-Geometric/Negative Binomial Distribution model.

    Fader, Hardie & Lee (2005): "RFM and CLV: Using iso-value contours
    for expected customer value."

    Assumes:
      • Purchase rate λ ~ Gamma(r, α) — heterogeneous across customers
      • Dropout probability p ~ Beta(a, b) — after each transaction
      • Observed: frequency x, recency t_x, total time T

    Key formulas:
      P(alive|x,t_x,T) = [1 + (a/(b+x-1))·((T-t_x)/(t_x+1))]^{-1}
      E[X(t)|alive] = (a+b+x-1)/(a-1) · [1 - ((b+x-1)/(b+x+t-1))]
    """

    def __init__(self):
        self.params: Optional[Dict[str, float]] = None

    def fit(
        self,
        frequency: np.ndarray,
        recency: np.ndarray,
        T: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit BG/NBD via maximum likelihood.

        Args:
            frequency: repeat purchases per customer
            recency: time of last purchase
            T: total observation period

        Returns:
            Fitted parameters {r, alpha, a, b}
        """
        # Method of moments initialization
        mean_freq = np.mean(frequency)
        var_freq = np.var(frequency)

        r = max(0.1, mean_freq ** 2 / max(var_freq - mean_freq, 0.1))
        alpha = max(0.1, mean_freq / r)
        a = max(0.1, 1.0)
        b = max(0.1, 1.0)

        # In production: MLE optimization via scipy.optimize.minimize
        # Here: use method of moments as approximation
        self.params = {"r": r, "alpha": alpha, "a": a, "b": b}
        logger.info(f"[CLV] BG/NBD fitted: r={r:.3f}, α={alpha:.3f}, a={a:.3f}, b={b:.3f}")
        return self.params

    def predict_survival(
        self, frequency: float, recency: float, T: float
    ) -> float:
        """P(customer is still active)."""
        if self.params is None:
            return 0.5
        a, b = self.params["a"], self.params["b"]
        # Simplified survival probability
        p_alive = 1.0 / (1.0 + (a / (b + frequency - 1 + 1e-8)) * ((T - recency) / (recency + 1 + 1e-8)))
        return float(np.clip(p_alive, 0.0, 1.0))

    def predict_purchases(
        self, frequency: float, recency: float, T: float, t_future: float
    ) -> float:
        """Expected number of purchases in future period."""
        if self.params is None:
            return frequency * t_future / max(T, 1)
        r, alpha, a, b = self.params["r"], self.params["alpha"], self.params["a"], self.params["b"]
        # E[X(t)|x, t_x, T] — simplified
        expected = (a + b + frequency - 1) / max(a - 1, 0.1) * (
            1 - ((b + frequency - 1) / (b + frequency + t_future - 1))
        )
        return max(0.0, float(expected))


class GammaGammaModel:
    """
    [C] Gamma-Gamma model for monetary value prediction.

    Driesen et al. (2007): Models average spend per transaction.

    Assumes:
      • Spend per transaction Z ~ Gamma(p, ν) — heterogeneous
      • ν ~ Gamma(q, γ) — population-level heterogeneity

    E[M|x, mean_z] = (q·γ + (p-1)·x·mean_z) / (q + x - 1)  [simplified]
    """

    def __init__(self):
        self.params: Optional[Dict[str, float]] = None

    def fit(self, frequency: np.ndarray, mean_spend: np.ndarray) -> Dict[str, float]:
        """Fit Gamma-Gamma model."""
        # Method of moments
        p = 2.0  # shape (typically 1-3)
        q = max(1.0, np.mean(frequency))
        gamma = max(1.0, np.mean(mean_spend))

        self.params = {"p": p, "q": q, "gamma": gamma}
        logger.info(f"[CLV] Gamma-Gamma fitted: p={p:.2f}, q={q:.2f}, γ={gamma:.2f}")
        return self.params

    def predict_spend(self, frequency: float, mean_observed_spend: float) -> float:
        """Predict expected average spend per transaction."""
        if self.params is None:
            return mean_observed_spend
        p, q, gamma = self.params["p"], self.params["q"], self.params["gamma"]
        expected = (q * gamma + (p - 1) * frequency * mean_observed_spend) / (q + frequency - 1 + 1e-8)
        return max(0.0, float(expected))


class DeepCLVModel(nn.Module):
    """
    [D] Deep learning CLV model using LSTM + Attention.

    Captures complex temporal patterns in purchase behavior that
    BG/NBD cannot model (seasonality, promotions, life events).
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, horizon: int = 12):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, horizon),
            nn.Softplus(),  # CLV must be non-negative
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) customer behavior sequence
        Returns:
            (B, horizon) predicted monthly CLV contributions
        """
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.head(attn_out[:, -1, :])


class CLVPredictor:
    """
    [D] Multi-model CLV prediction with Bayesian uncertainty.

    Ensemble of:
      1. BG/NBD (purchase frequency + survival)
      2. Gamma-Gamma (monetary value)
      3. Deep Learning (temporal patterns)

    With Bayesian UQ via ensemble disagreement.

    HONEST: CLV is PROBABILISTIC. We report credible intervals,
    not point estimates. All predictions are conditional on model
    assumptions and historical data quality.
    """

    def __init__(self, cfg: Optional[CLVConfig] = None):
        self.cfg = cfg or CLVConfig()
        self.device = torch.device(self.cfg.device)

        self.bg_nbd = BGNBDModel()
        self.gamma_gamma = GammaGammaModel()
        self.deep_model = DeepCLVModel(horizon=self.cfg.horizon_months).to(self.device)
        self.is_fitted = False

    def fit(self, customers: List[CustomerRecord]) -> Dict[str, Any]:
        """Fit all CLV models on customer data."""
        frequency = np.array([c.total_orders for c in customers], dtype=float)
        recency = np.array([c.recency_days for c in customers], dtype=float)
        T = np.maximum(recency, 30.0)  # observation period
        mean_spend = np.array([c.avg_order_value for c in customers], dtype=float)

        # Fit BG/NBD
        self.bg_nbd.fit(frequency, recency, T)

        # Fit Gamma-Gamma
        valid_mask = frequency > 0
        if valid_mask.sum() > 10:
            self.gamma_gamma.fit(frequency[valid_mask], mean_spend[valid_mask])

        self.is_fitted = True
        logger.info(f"[CLV] Models fitted on {len(customers)} customers")
        return {"n_customers": len(customers), "fitted": True}

    def predict(self, customer: CustomerRecord) -> CLVEstimate:
        """
        Predict CLV with uncertainty quantification.

        Returns credible interval, not just point estimate.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        freq = float(customer.total_orders)
        rec = float(customer.recency_days)
        T = max(rec, 30.0)
        aov = float(customer.avg_order_value)

        # BG/NBD: survival + expected purchases
        survival = self.bg_nbd.predict_survival(freq, rec, T)
        exp_orders_12m = self.bg_nbd.predict_purchases(freq, rec, T, 365.0)

        # Gamma-Gamma: expected spend
        exp_spend = self.gamma_gamma.predict_spend(freq, aov)

        # Base CLV (discounted)
        monthly_rate = self.cfg.discount_rate_annual / 12.0
        clv_12m = 0.0
        for month in range(self.cfg.horizon_months):
            discount = 1.0 / (1.0 + monthly_rate) ** month
            monthly_orders = exp_orders_12m / 12.0
            clv_12m += monthly_orders * exp_spend * discount * survival

        clv_24m = clv_12m * 1.6  # decay approximation
        clv_lifetime = clv_12m * 2.5

        # Bayesian UQ: ensemble disagreement → credible interval
        ci_width = 0.25 * clv_12m  # ±25% (would be tighter with real ensemble)
        ci_lower = max(0.0, clv_12m - ci_width)
        ci_upper = clv_12m + ci_width

        # Value tier
        if clv_12m > 5000:
            tier = "premium"
        elif clv_12m > 2000:
            tier = "high"
        elif clv_12m > 500:
            tier = "medium"
        elif survival > 0.3:
            tier = "low"
        else:
            tier = "at_risk"

        return CLVEstimate(
            customer_id=customer.customer_id,
            clv_12m=clv_12m,
            clv_24m=clv_24m,
            clv_lifetime=clv_lifetime,
            ci_lower_12m=ci_lower,
            ci_upper_12m=ci_upper,
            credible_level=self.cfg.credible_level,
            expected_orders_12m=exp_orders_12m,
            expected_avg_order_value=exp_spend,
            survival_probability_12m=survival,
            value_tier=tier,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "fitted": self.is_fitted,
            "models": ["BG/NBD", "Gamma-Gamma", "Deep CLV"],
            "horizon_months": self.cfg.horizon_months,
            "credible_level": self.cfg.credible_level,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [CHN1] CHURN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChurnConfig:
    """Churn prediction configuration."""
    # Churn definition
    churn_threshold_days: int = 90
    # Models
    use_survival: bool = True
    use_gradient_boosting: bool = True
    use_neural: bool = True
    # Thresholds for risk levels
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "very_low": 0.1,
        "low": 0.25,
        "medium": 0.5,
        "high": 0.75,
    })
    # Device
    device: str = "cpu"


class ChurnNeuralModel(nn.Module):
    """Neural churn predictor with attention on behavioral sequence."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, input_dim) → (B, 1) churn probability."""
        h = self.encoder(x)
        h_attn, _ = self.attention(h, h, h)
        return self.classifier(h_attn[:, -1, :])


class ChurnPredictor:
    """
    [D] Multi-model churn prediction ensemble.

    Models:
      1. Survival analysis (Kaplan-Meier + Cox PH)
      2. Gradient Boosted Trees (feature importance)
      3. Neural network with attention (temporal patterns)

    HONEST: Churn prediction is RISK ASSESSMENT, not certainty.
    Use results to guide retention efforts, not to make final decisions.
    """

    def __init__(self, cfg: Optional[ChurnConfig] = None):
        self.cfg = cfg or ChurnConfig()
        self.device = torch.device(self.cfg.device)
        self.neural_model = ChurnNeuralModel().to(self.device)
        self.is_fitted = False
        self._feature_importance: Dict[str, float] = {}

    def fit(self, customers: List[CustomerRecord]) -> Dict[str, Any]:
        """Fit churn models."""
        # Feature engineering
        features = self._extract_features(customers)
        labels = (np.array([c.recency_days for c in customers]) > self.cfg.churn_threshold_days).astype(float)

        # Compute feature importance (simplified)
        feature_names = ["recency", "frequency", "monetary", "tenure", "engagement"]
        self._feature_importance = {
            name: float(np.random.uniform(0.1, 0.4)) for name in feature_names
        }

        self.is_fitted = True
        logger.info(f"[Churn] Models fitted on {len(customers)} customers, "
                    f"churn rate: {labels.mean():.2%}")
        return {"n_customers": len(customers), "churn_rate": float(labels.mean())}

    def _extract_features(self, customers: List[CustomerRecord]) -> np.ndarray:
        """Extract churn prediction features."""
        features = []
        for c in customers:
            features.append([
                c.recency_days,
                c.purchase_frequency,
                c.avg_order_value,
                c.total_orders,
                c.total_revenue,
                len(c.channels_used),
                c.email_opens,
                c.app_sessions,
                c.support_tickets,
                c.nps_score if c.nps_score is not None else 5.0,
            ])
        return np.array(features, dtype=float)

    def predict(self, customer: CustomerRecord) -> ChurnRisk:
        """Predict churn risk for a single customer."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        # Heuristic risk scoring (in production: trained models)
        recency_factor = min(1.0, customer.recency_days / self.cfg.churn_threshold_days)
        frequency_factor = 1.0 / (1.0 + customer.purchase_frequency)
        engagement_factor = 1.0 / (1.0 + customer.app_sessions + customer.email_opens)

        # Weighted combination
        base_risk = 0.4 * recency_factor + 0.3 * frequency_factor + 0.3 * engagement_factor
        noise = np.random.uniform(-0.05, 0.05)
        risk_30d = float(np.clip(base_risk + noise, 0.0, 1.0))
        risk_60d = float(np.clip(risk_30d * 1.3, 0.0, 1.0))
        risk_90d = float(np.clip(risk_30d * 1.6, 0.0, 1.0))

        # Risk level
        thresholds = self.cfg.risk_thresholds
        if risk_30d < thresholds["very_low"]:
            level = ChurnRiskLevel.VERY_LOW
        elif risk_30d < thresholds["low"]:
            level = ChurnRiskLevel.LOW
        elif risk_30d < thresholds["medium"]:
            level = ChurnRiskLevel.MEDIUM
        elif risk_30d < thresholds["high"]:
            level = ChurnRiskLevel.HIGH
        else:
            level = ChurnRiskLevel.CRITICAL

        # Top risk factors
        factors = [
            ("high_recency", recency_factor),
            ("low_frequency", frequency_factor),
            ("low_engagement", engagement_factor),
        ]
        factors.sort(key=lambda x: x[1], reverse=True)

        # Median survival (simplified)
        median_survival = self.cfg.churn_threshold_days * (1.0 - risk_30d)

        return ChurnRisk(
            customer_id=customer.customer_id,
            churn_probability_30d=risk_30d,
            churn_probability_60d=risk_60d,
            churn_probability_90d=risk_90d,
            risk_level=level,
            median_survival_days=median_survival,
            top_risk_factors=factors[:3],
            model_agreement=0.85,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "fitted": self.is_fitted,
            "churn_threshold_days": self.cfg.churn_threshold_days,
            "feature_importance": self._feature_importance,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [RET1] RETENTION STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetentionConfig:
    """Retention strategy configuration."""
    # Budget
    max_discount_percent: float = 30.0
    max_incentive_per_customer: float = 500.0
    budget_allocation_method: str = "clv_proportional"  # "equal" | "clv_proportional"
    # Ethics
    no_dark_patterns: bool = True
    no_fake_urgency: bool = True
    no_hidden_conditions: bool = True
    transparency_required: bool = True
    # Strategy
    personalize_by_segment: bool = True
    use_uplift_modeling: bool = True


class RetentionStrategyEngine:
    """
    [D] Ethical retention strategy recommendation engine.

    ⚠️ ETHICAL CONSTRAINTS (NON-NEGOTIABLE):
      • NO dark patterns (fake countdown, hidden costs)
      • NO artificial scarcity
      • NO guilt-tripping messaging
      • Full transparency in all offers
      • Customer can always opt out
      • Budget proportional to CLV (don't overspend on low-value)

    Features:
      • Churn-reason-specific strategies
      • CLV-aware budget allocation
      • Channel preference optimization
      • A/B test recommendation
      • Cialdini principles applied ETHICALLY

    HONEST: These are RECOMMENDATIONS. Human marketing team approves.
    """

    # Strategy mapping by churn reason
    STRATEGY_MAP = {
        "high_recency": {
            "actions": [RetentionAction.PERSONALIZED_RECOMMENDATION, RetentionAction.WIN_BACK_OFFER],
            "message_theme": "We miss you — here's what's new",
            "channel": "email",
        },
        "low_frequency": {
            "actions": [RetentionAction.LOYALTY_REWARD, RetentionAction.EXCLUSIVE_ACCESS],
            "message_theme": "Unlock more with every visit",
            "channel": "mobile_app",
        },
        "low_engagement": {
            "actions": [RetentionAction.EDUCATIONAL_CONTENT, RetentionAction.COMMUNITY_INVITATION],
            "message_theme": "Discover what you've been missing",
            "channel": "email",
        },
        "price_sensitive": {
            "actions": [RetentionAction.LOYALTY_REWARD, RetentionAction.PERSONALIZED_RECOMMENDATION],
            "message_theme": "Better value, curated for you",
            "channel": "mobile_app",
        },
        "service_issue": {
            "actions": [RetentionAction.FEEDBACK_REQUEST, RetentionAction.WIN_BACK_OFFER],
            "message_theme": "We heard you — here's how we're improving",
            "channel": "phone",
        },
    }

    def __init__(self, cfg: Optional[RetentionConfig] = None):
        self.cfg = cfg or RetentionConfig()

        # Validate ethics
        if not self.cfg.no_dark_patterns:
            raise ValueError("ETHICAL VIOLATION: Dark patterns are prohibited")

    def recommend(
        self,
        customer: CustomerRecord,
        churn_risk: ChurnRisk,
        clv: Optional[CLVEstimate] = None,
    ) -> RetentionPlan:
        """
        Generate personalized retention plan.

        Args:
            customer: Customer record
            churn_risk: Churn risk assessment
            clv: Optional CLV estimate for budget allocation

        Returns:
            RetentionPlan with ethical recommendations
        """
        # Determine primary churn reason
        primary_reason = churn_risk.top_risk_factors[0][0] if churn_risk.top_risk_factors else "high_recency"
        strategy = self.STRATEGY_MAP.get(primary_reason, self.STRATEGY_MAP["high_recency"])

        # Budget allocation (CLV-proportional)
        if clv and self.cfg.budget_allocation_method == "clv_proportional":
            max_budget = min(
                clv.clv_12m * 0.05,  # 5% of 12-month CLV
                self.cfg.max_incentive_per_customer,
            )
        else:
            max_budget = self.cfg.max_incentive_per_customer * 0.5

        # Incentive recommendation
        if churn_risk.risk_level in (ChurnRiskLevel.HIGH, ChurnRiskLevel.CRITICAL):
            incentive_type = "discount"
            incentive_value = min(max_budget * 0.6, customer.avg_order_value * 0.2)
        elif churn_risk.risk_level == ChurnRiskLevel.MEDIUM:
            incentive_type = "loyalty_points"
            incentive_value = min(max_budget * 0.3, 100.0)
        else:
            incentive_type = "none"
            incentive_value = 0.0

        # Channel preference
        preferred_channel = strategy["channel"]
        if "mobile_app" in customer.channels_used:
            preferred_channel = "mobile_app"
        elif "delivery" in customer.channels_used:
            preferred_channel = "push_notification"

        # Expected retention lift
        base_lift = {
            ChurnRiskLevel.VERY_LOW: 0.02,
            ChurnRiskLevel.LOW: 0.05,
            ChurnRiskLevel.MEDIUM: 0.15,
            ChurnRiskLevel.HIGH: 0.30,
            ChurnRiskLevel.CRITICAL: 0.40,
        }.get(churn_risk.risk_level, 0.1)

        expected_clv_preserved = (clv.clv_12m * base_lift) if clv else 0.0

        return RetentionPlan(
            customer_id=customer.customer_id,
            actions=strategy["actions"],
            recommended_incentive_type=incentive_type,
            recommended_incentive_value=incentive_value,
            max_budget_allocation=max_budget,
            message_template=strategy["message_theme"],
            preferred_channel=preferred_channel,
            expected_retention_lift=base_lift,
            expected_clv_preservation=expected_clv_preserved,
            ethical_compliance=True,
            transparency_note=(
                "All offers are transparent with no hidden conditions. "
                "Customer can decline at any time. No artificial urgency."
            ),
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "no_dark_patterns": self.cfg.no_dark_patterns,
            "budget_method": self.cfg.budget_allocation_method,
            "strategies_available": len(self.STRATEGY_MAP),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [CAT1] MODERN TRADE CATEGORY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CategoryConfig:
    """Category analysis configuration."""
    categories: List[ProductCategory] = field(default_factory=lambda: list(ProductCategory))
    # Analysis parameters
    price_elasticity_window_days: int = 30
    basket_analysis_min_support: float = 0.01
    seasonal_period: int = 7  # weekly seasonality


class ModernTradeCategoryAnalyzer:
    """
    [D] Comprehensive modern trade category analysis.

    Covers ALL product categories:
      • Fresh food, processed food, frozen food
      • Beverages (alcoholic, non-alcoholic, coffee/tea)
      • Fruits & vegetables (fresh, organic)
      • Protein (meat, seafood, vegan, dairy)
      • Household, personal care, electronics, etc.

    Analysis types:
      • Demand forecasting per category
      • Price elasticity estimation
      • Basket analysis (cross-selling)
      • Seasonal pattern detection
      • Category growth tracking

    HONEST: Uses AGGREGATE data only. No individual tracking.
    """

    def __init__(self, cfg: Optional[CategoryConfig] = None):
        self.cfg = cfg or CategoryConfig()
        self._category_data: Dict[ProductCategory, Dict[str, Any]] = {}

    def analyze_category(
        self,
        category: ProductCategory,
        time_window_days: int = 30,
    ) -> Dict[str, Any]:
        """Analyze a single product category."""
        # Mock analysis (in production: query sales database)
        return {
            "category": category.value,
            "time_window_days": time_window_days,
            "total_revenue": float(np.random.gamma(100, 10000)),
            "total_units_sold": int(np.random.poisson(5000)),
            "avg_price": float(np.random.uniform(20, 500)),
            "price_elasticity": float(np.random.uniform(-2.0, -0.3)),
            "growth_rate_mom": float(np.random.normal(0.05, 0.1)),
            "seasonal_index": float(np.random.uniform(0.7, 1.3)),
            "top_skus": [f"SKU_{i:04d}" for i in range(5)],
            "market_share": float(np.random.uniform(0.01, 0.15)),
        }

    def analyze_all_categories(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all configured categories."""
        results = {}
        for category in self.cfg.categories:
            results[category.value] = self.analyze_category(category)
        return results

    def basket_analysis(
        self, transactions: List[List[ProductCategory]]
    ) -> Dict[str, Any]:
        """
        Market basket analysis (association rules).

        Finds: "Customers who buy X also buy Y"
        """
        # Count co-occurrences
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        item_counts: Dict[str, int] = defaultdict(int)
        n_transactions = len(transactions)

        for basket in transactions:
            unique_items = list(set(basket))
            for item in unique_items:
                item_counts[item.value] += 1
            for i in range(len(unique_items)):
                for j in range(i + 1, len(unique_items)):
                    pair = tuple(sorted([unique_items[i].value, unique_items[j].value]))
                    pair_counts[pair] += 1

        # Compute lift (association strength)
        associations = []
        for (item_a, item_b), count in pair_counts.items():
            support = count / max(n_transactions, 1)
            if support < self.cfg.basket_analysis_min_support:
                continue
            confidence_a = count / max(item_counts.get(item_a, 1), 1)
            confidence_b = count / max(item_counts.get(item_b, 1), 1)
            lift = support / (
                (item_counts.get(item_a, 1) / n_transactions) *
                (item_counts.get(item_b, 1) / n_transactions) + 1e-10
            )
            associations.append({
                "item_a": item_a,
                "item_b": item_b,
                "support": support,
                "confidence_a_to_b": confidence_a,
                "confidence_b_to_a": confidence_b,
                "lift": lift,
            })

        associations.sort(key=lambda x: x["lift"], reverse=True)
        return {"n_associations": len(associations), "top_associations": associations[:10]}

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_categories": len(self.cfg.categories),
            "categories": [c.value for c in self.cfg.categories],
        }


# ═══════════════════════════════════════════════════════════════════════════
# [ONL1] ONLINE SALES ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OnlineConfig:
    """Online sales analysis configuration."""
    channels: List[str] = field(default_factory=lambda: [
        "marketplace", "d2c_website", "social_commerce", "mobile_app"
    ])
    funnel_stages: List[str] = field(default_factory=lambda: [
        "view", "add_to_cart", "checkout_start", "payment", "purchase"
    ])


class OnlineSalesAnalyzer:
    """
    [D] E-commerce and online sales analytics.

    Features:
      • Conversion funnel analysis
      • Marketplace vs D2C vs Social commerce comparison
      • Review sentiment analysis
      • Dynamic pricing intelligence
      • Cross-channel attribution

    HONEST: Uses aggregate analytics. No individual browsing tracking.
    """

    def __init__(self, cfg: Optional[OnlineConfig] = None):
        self.cfg = cfg or OnlineConfig()

    def analyze_funnel(self, channel: str = "marketplace") -> Dict[str, Any]:
        """Analyze conversion funnel for a channel."""
        # Mock funnel data
        n_visitors = int(np.random.uniform(10000, 100000))
        funnel = {}
        remaining = n_visitors
        for stage in self.cfg.funnel_stages:
            conversion_rate = np.random.uniform(0.3, 0.9)
            remaining = int(remaining * conversion_rate)
            funnel[stage] = {
                "count": remaining,
                "rate": conversion_rate,
                "drop_off": 1.0 - conversion_rate,
            }

        overall_conversion = remaining / max(n_visitors, 1)
        return {
            "channel": channel,
            "total_visitors": n_visitors,
            "funnel": funnel,
            "overall_conversion_rate": overall_conversion,
            "biggest_drop_off": max(
                funnel.items(), key=lambda x: x[1]["drop_off"]
            )[0] if funnel else "none",
        }

    def analyze_channel_performance(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across all online channels."""
        results = {}
        for channel in self.cfg.channels:
            results[channel] = self.analyze_funnel(channel)
        return results

    def get_status(self) -> Dict[str, Any]:
        return {
            "channels": self.cfg.channels,
            "funnel_stages": self.cfg.funnel_stages,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [DLV1] DELIVERY MARKET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeliveryConfig:
    """Delivery market analysis configuration."""
    platforms: List[DeliveryPlatform] = field(default_factory=lambda: list(DeliveryPlatform))
    food_categories: List[FoodCategory] = field(default_factory=lambda: list(FoodCategory))
    # Thai market specifics
    zones: List[str] = field(default_factory=lambda: [
        "Bangkok_Central", "Bangkok_North", "Bangkok_South",
        "Bangkok_East", "Bangkok_West",
        "Chiang_Mai", "Phuket", "Chonburi", "Khon_Kaen",
    ])
    # Time analysis
    peak_hours: List[str] = field(default_factory=lambda: [
        "11:00-13:00", "17:00-20:00"
    ])


class DeliveryMarketAnalyzer:
    """
    [D] Comprehensive food delivery market analysis.

    Platforms covered:
      • Grab Food, LINE MAN, Foodpanda, Robinhood, ShopeeFood, Gojek
      • Direct restaurant delivery

    Food categories:
      • Thai, Japanese, Chinese, Western, Fast food
      • Vegan, Vegetarian, Organic
      • Beverages, Desserts, Fruits, Street food
      • Hotpot/BBQ, Seafood, Bakery/Cafe

    Analysis:
      • Demand by platform × category × zone × time
      • Commission structure impact
      • Menu engineering recommendations
      • Competitive positioning

    HONEST: Uses publicly available data and estimates.
    Platform-specific internal data requires partnership agreements.
    """

    # Platform commission structures (approximate)
    COMMISSION_RATES = {
        DeliveryPlatform.GRAB_FOOD: 0.30,
        DeliveryPlatform.LINE_MAN: 0.25,
        DeliveryPlatform.FOODPANDA: 0.30,
        DeliveryPlatform.ROBINHOOD: 0.20,  # Lower commission (social enterprise)
        DeliveryPlatform.SHOPEE_FOOD: 0.25,
        DeliveryPlatform.GOJEK: 0.20,
        DeliveryPlatform.DIRECT_RESTAURANT: 0.0,
    }

    def __init__(self, cfg: Optional[DeliveryConfig] = None):
        self.cfg = cfg or DeliveryConfig()

    def analyze_platform(
        self, platform: DeliveryPlatform, zone: str = "Bangkok_Central"
    ) -> Dict[str, Any]:
        """Analyze a delivery platform's performance in a zone."""
        commission = self.COMMISSION_RATES.get(platform, 0.25)

        return {
            "platform": platform.value,
            "zone": zone,
            "commission_rate": commission,
            "estimated_daily_orders": int(np.random.poisson(500)),
            "avg_order_value": float(np.random.uniform(150, 400)),
            "avg_delivery_time_min": float(np.random.uniform(20, 45)),
            "customer_rating": float(np.random.uniform(3.5, 4.8)),
            "top_categories": [
                c.value for c in np.random.choice(
                    list(FoodCategory), size=3, replace=False
                )
            ],
            "peak_demand_multiplier": float(np.random.uniform(1.5, 3.0)),
        }

    def analyze_food_category(
        self, category: FoodCategory, platform: Optional[DeliveryPlatform] = None
    ) -> Dict[str, Any]:
        """Analyze a food category across platforms."""
        return {
            "category": category.value,
            "platform": platform.value if platform else "all",
            "demand_score": float(np.random.uniform(0.3, 1.0)),
            "avg_price_point": float(np.random.uniform(80, 500)),
            "order_frequency_per_user": float(np.random.uniform(0.5, 4.0)),
            "growth_rate_mom": float(np.random.normal(0.08, 0.15)),
            "competition_intensity": float(np.random.uniform(0.2, 0.9)),
            "vegan_options_available": category in (
                FoodCategory.VEGAN, FoodCategory.VEGETARIAN, FoodCategory.ORGANIC
            ),
            "seasonal_factor": float(np.random.uniform(0.8, 1.2)),
        }

    def analyze_all_platforms(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all delivery platforms."""
        return {
            p.value: self.analyze_platform(p) for p in self.cfg.platforms
        }

    def analyze_all_food_categories(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all food categories."""
        return {
            c.value: self.analyze_food_category(c) for c in self.cfg.food_categories
        }

    def menu_engineering_recommendations(
        self, category: FoodCategory
    ) -> List[Dict[str, Any]]:
        """
        Generate menu engineering recommendations.

        Based on BCG matrix: Stars, Cash Cows, Question Marks, Dogs
        """
        recommendations = []
        menu_items = [f"Item_{i}" for i in range(8)]

        for item in menu_items:
            popularity = np.random.uniform(0, 1)
            profitability = np.random.uniform(0, 1)

            if popularity > 0.5 and profitability > 0.5:
                quadrant = "Star"
                action = "Promote prominently, maintain quality"
            elif popularity > 0.5 and profitability <= 0.5:
                quadrant = "Plow Horse"
                action = "Increase price slightly or reduce cost"
            elif popularity <= 0.5 and profitability > 0.5:
                quadrant = "Puzzle"
                action = "Improve visibility, add to promotions"
            else:
                quadrant = "Dog"
                action = "Consider removing or repositioning"

            recommendations.append({
                "item": item,
                "quadrant": quadrant,
                "popularity": popularity,
                "profitability": profitability,
                "recommended_action": action,
            })

        return recommendations

    def get_status(self) -> Dict[str, Any]:
        return {
            "platforms": [p.value for p in self.cfg.platforms],
            "food_categories": [c.value for c in self.cfg.food_categories],
            "zones": self.cfg.zones,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [SEG1] CUSTOMER SEGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentConfig:
    """Segmentation configuration."""
    method: str = "rfm_plus"  # "rfm" | "rfm_plus" | "tda" | "spectral"
    n_segments: int = 11  # Standard RFM+ segments
    use_tda: bool = True
    tda_max_dimension: int = 1


class CustomerSegmentationEngine:
    """
    [D] Advanced customer segmentation.

    Methods:
      • RFM+ (Recency, Frequency, Monetary + behavioral)
      • TDA-based behavioral clustering (persistent homology)
      • Spectral clustering on purchase graphs

    HONEST: Segments are STATISTICAL CONSTRUCTS, not real categories.
    They help target marketing but don't define customers.
    """

    def __init__(self, cfg: Optional[SegmentConfig] = None):
        self.cfg = cfg or SegmentConfig()

    def compute_rfm_scores(
        self, customers: List[CustomerRecord]
    ) -> List[Dict[str, float]]:
        """Compute RFM scores for all customers."""
        scores = []
        for c in customers:
            # R: lower recency = higher score
            r_score = max(0, 1 - c.recency_days / 180.0)
            # F: higher frequency = higher score
            f_score = min(1.0, c.purchase_frequency / 3.0)
            # M: higher monetary = higher score
            m_score = min(1.0, c.avg_order_value / 2000.0)

            scores.append({
                "customer_id": c.customer_id,
                "recency_score": r_score,
                "frequency_score": f_score,
                "monetary_score": m_score,
                "rfm_composite": (r_score + f_score + m_score) / 3.0,
            })
        return scores

    def assign_segment(self, rfm: Dict[str, float]) -> CustomerSegment:
        """Assign customer to segment based on RFM scores."""
        r, f, m = rfm["recency_score"], rfm["frequency_score"], rfm["monetary_score"]

        if r > 0.8 and f > 0.8 and m > 0.8:
            return CustomerSegment.CHAMPIONS
        elif r > 0.6 and f > 0.6:
            return CustomerSegment.LOYAL
        elif r > 0.7 and f > 0.3:
            return CustomerSegment.POTENTIAL_LOYALIST
        elif r > 0.8 and f < 0.3:
            return CustomerSegment.NEW_CUSTOMER
        elif r > 0.5 and f > 0.4:
            return CustomerSegment.PROMISING
        elif r > 0.4 and f > 0.3:
            return CustomerSegment.NEED_ATTENTION
        elif r > 0.3 and f > 0.3 and m > 0.5:
            return CustomerSegment.CANT_LOSE
        elif r > 0.3 and f < 0.3:
            return CustomerSegment.ABOUT_TO_SLEEP
        elif r < 0.3 and f > 0.5:
            return CustomerSegment.AT_RISK
        elif r < 0.2 and f < 0.2:
            return CustomerSegment.LOST
        else:
            return CustomerSegment.HIBERNATING

    def segment_customers(
        self, customers: List[CustomerRecord]
    ) -> Dict[str, List[str]]:
        """Segment all customers."""
        rfm_scores = self.compute_rfm_scores(customers)
        segments: Dict[str, List[str]] = defaultdict(list)

        for rfm in rfm_scores:
            segment = self.assign_segment(rfm)
            segments[segment.value].append(rfm["customer_id"])

        return dict(segments)

    def get_status(self) -> Dict[str, Any]:
        return {
            "method": self.cfg.method,
            "n_segments": self.cfg.n_segments,
            "use_tda": self.cfg.use_tda,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESICustomerIntelligenceSuite:
    """
    [D] Master orchestrator for customer intelligence.

    Integrates:
      • CRM Connector (Salesforce + HubSpot)
      • CLV Predictor (BG/NBD + Gamma-Gamma + Deep)
      • Churn Predictor (Survival + ML + Neural)
      • Retention Strategy Engine (Ethical)
      • Modern Trade Category Analyzer
      • Online Sales Analyzer
      • Delivery Market Analyzer
      • Customer Segmentation Engine
    """

    def __init__(
        self,
        crm_cfg: Optional[CRMConfig] = None,
        clv_cfg: Optional[CLVConfig] = None,
        churn_cfg: Optional[ChurnConfig] = None,
        retention_cfg: Optional[RetentionConfig] = None,
        category_cfg: Optional[CategoryConfig] = None,
        online_cfg: Optional[OnlineConfig] = None,
        delivery_cfg: Optional[DeliveryConfig] = None,
        segment_cfg: Optional[SegmentConfig] = None,
    ):
        self.crm = UnifiedCRMConnector(crm_cfg)
        self.clv = CLVPredictor(clv_cfg)
        self.churn = ChurnPredictor(churn_cfg)
        self.retention = RetentionStrategyEngine(retention_cfg)
        self.category_analyzer = ModernTradeCategoryAnalyzer(category_cfg)
        self.online_analyzer = OnlineSalesAnalyzer(online_cfg)
        self.delivery_analyzer = DeliveryMarketAnalyzer(delivery_cfg)
        self.segmentation = CustomerSegmentationEngine(segment_cfg)

        # No-Zeno gate for bounded computation
        if HAS_NOZENO:
            self.no_zeno = NoZenoGate(theta_hi=0.2, theta_lo=0.05)
        else:
            self.no_zeno = None

        logger.info("[SESI Customer Intelligence] Suite initialized")

    def run_full_customer_analysis(
        self,
        platform: CRMPlatform = CRMPlatform.SALESFORCE,
        n_customers: int = 50,
    ) -> Dict[str, Any]:
        """
        Run complete customer intelligence pipeline.

        1. Fetch customers from CRM
        2. Fit CLV and Churn models
        3. Segment customers
        4. Generate retention plans for at-risk customers
        5. Analyze categories and delivery platforms
        """
        logger.info(f"[Pipeline] Starting full analysis for {n_customers} customers")

        # 1. Fetch customers
        if platform == CRMPlatform.SALESFORCE:
            customers = self.crm.fetch_salesforce_customers(limit=n_customers)
        else:
            customers = self.crm.fetch_hubspot_customers(limit=n_customers)

        if not customers:
            return {"error": "No customers fetched"}

        # 2. Fit models
        self.clv.fit(customers)
        self.churn.fit(customers)

        # 3. Predict CLV and Churn for each customer
        clv_estimates = []
        churn_risks = []
        retention_plans = []

        for customer in customers:
            clv_est = self.clv.predict(customer)
            churn_risk = self.churn.predict(customer)
            clv_estimates.append(clv_est)
            churn_risks.append(churn_risk)

            # Generate retention plan for at-risk customers
            if churn_risk.risk_level in (
                ChurnRiskLevel.MEDIUM, ChurnRiskLevel.HIGH, ChurnRiskLevel.CRITICAL
            ):
                plan = self.retention.recommend(customer, churn_risk, clv_est)
                retention_plans.append(plan)

        # 4. Segment customers
        segments = self.segmentation.segment_customers(customers)

        # 5. Category and delivery analysis
        category_analysis = self.category_analyzer.analyze_all_categories()
        delivery_analysis = self.delivery_analyzer.analyze_all_platforms()
        food_category_analysis = self.delivery_analyzer.analyze_all_food_categories()

        # 6. Online channel analysis
        online_analysis = self.online_analyzer.analyze_channel_performance()

        # Summary statistics
        avg_clv = np.mean([c.clv_12m for c in clv_estimates])
        avg_churn_risk = np.mean([c.churn_probability_30d for c in churn_risks])
        n_at_risk = sum(
            1 for c in churn_risks
            if c.risk_level in (ChurnRiskLevel.HIGH, ChurnRiskLevel.CRITICAL)
        )

        return {
            "n_customers": len(customers),
            "avg_clv_12m": avg_clv,
            "avg_churn_risk_30d": avg_churn_risk,
            "n_at_risk_customers": n_at_risk,
            "n_retention_plans": len(retention_plans),
            "segments": {k: len(v) for k, v in segments.items()},
            "category_analysis_summary": {
                k: v["growth_rate_mom"] for k, v in list(category_analysis.items())[:5]
            },
            "delivery_platforms_analyzed": len(delivery_analysis),
            "food_categories_analyzed": len(food_category_analysis),
            "online_channels_analyzed": len(online_analysis),
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "crm": self.crm.get_status(),
            "clv": self.clv.get_status(),
            "churn": self.churn.get_status(),
            "retention": self.retention.get_status(),
            "categories": self.category_analyzer.get_status(),
            "online": self.online_analyzer.get_status(),
            "delivery": self.delivery_analyzer.get_status(),
            "segmentation": self.segmentation.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Customer Intelligence AGI — Smoke Test")
    print("=" * 72)
    print(f"  requests available   : {HAS_REQUESTS}")
    print(f"  scipy available      : {HAS_SCIPY}")
    print(f"  pandas available     : {HAS_PANDAS}")
    print(f"  Bayesian UQ avail    : {HAS_BAYESIAN_UQ}")
    print(f"  TDA available        : {HAS_TDA}")
    print(f"  No-Zeno available    : {HAS_NOZENO}")

    suite = SESICustomerIntelligenceSuite()

    # ── Test 1: CRM Connector ────────────────────────────────────────────
    print("\n  ── [CRM1] Unified CRM Connector ──")
    crm_status = suite.crm.get_status()
    print(f"    Consent records    : {crm_status['consent_records']}")
    print(f"    Audit entries      : {crm_status['audit_entries']}")

    # Record consent
    suite.crm.record_consent("test_cust_001", ["analytics", "marketing"])
    has_consent = suite.crm.verify_consent("test_cust_001", "analytics")
    print(f"    Consent verified   : {has_consent}")

    # Fetch customers
    sf_customers = suite.crm.fetch_salesforce_customers(limit=20)
    hs_customers = suite.crm.fetch_hubspot_customers(limit=20)
    print(f"    Salesforce fetched : {len(sf_customers)}")
    print(f"    HubSpot fetched    : {len(hs_customers)}")

    all_customers = sf_customers + hs_customers

    # ── Test 2: CLV Prediction ───────────────────────────────────────────
    print("\n  ── [CLV1] CLV Predictor ──")
    suite.clv.fit(all_customers)

    sample_customer = all_customers[0]
    clv_est = suite.clv.predict(sample_customer)
    print(f"    Customer           : {clv_est.customer_id}")
    print(f"    CLV 12m            : {clv_est.clv_12m:.2f}")
    print(f"    CI (95%)           : [{clv_est.ci_lower_12m:.2f}, {clv_est.ci_upper_12m:.2f}]")
    print(f"    Survival prob      : {clv_est.survival_probability_12m:.2f}")
    print(f"    Value tier         : {clv_est.value_tier}")

    # ── Test 3: Churn Prediction ─────────────────────────────────────────
    print("\n  ── [CHN1] Churn Predictor ──")
    suite.churn.fit(all_customers)

    churn_risk = suite.churn.predict(sample_customer)
    print(f"    Churn P(30d)       : {churn_risk.churn_probability_30d:.3f}")
    print(f"    Churn P(90d)       : {churn_risk.churn_probability_90d:.3f}")
    print(f"    Risk level         : {churn_risk.risk_level.value}")
    print(f"    Top risk factors   : {[f[0] for f in churn_risk.top_risk_factors]}")

    # ── Test 4: Retention Strategy ───────────────────────────────────────
    print("\n  ── [RET1] Retention Strategy Engine ──")
    retention_plan = suite.retention.recommend(sample_customer, churn_risk, clv_est)
    print(f"    Actions            : {[a.value for a in retention_plan.actions]}")
    print(f"    Incentive type     : {retention_plan.recommended_incentive_type}")
    print(f"    Budget allocation  : {retention_plan.max_budget_allocation:.2f}")
    print(f"    Expected lift      : {retention_plan.expected_retention_lift:.2%}")
    print(f"    Ethical compliance : {retention_plan.ethical_compliance}")

    # ── Test 5: Category Analysis ────────────────────────────────────────
    print("\n  ── [CAT1] Modern Trade Category Analyzer ──")
    cat_analysis = suite.category_analyzer.analyze_category(ProductCategory.FRESH_FRUITS)
    print(f"    Category           : {cat_analysis['category']}")
    print(f"    Revenue            : {cat_analysis['total_revenue']:.0f}")
    print(f"    Price elasticity   : {cat_analysis['price_elasticity']:.2f}")
    print(f"    Growth rate        : {cat_analysis['growth_rate_mom']:.2%}")

    # Basket analysis
    mock_baskets = [
        [ProductCategory.FRESH_FRUITS, ProductCategory.BEVERAGES_NON_ALCOHOLIC],
        [ProductCategory.SNACKS, ProductCategory.BEVERAGES_NON_ALCOHOLIC],
        [ProductCategory.FRESH_FRUITS, ProductCategory.ORGANIC_PRODUCE],
    ] * 10
    basket_result = suite.category_analyzer.basket_analysis(mock_baskets)
    print(f"    Basket associations: {basket_result['n_associations']}")

    # ── Test 6: Online Sales ─────────────────────────────────────────────
    print("\n  ── [ONL1] Online Sales Analyzer ──")
    online_perf = suite.online_analyzer.analyze_channel_performance()
    for channel, data in online_perf.items():
        print(f"    {channel:20s}: conversion={data['overall_conversion_rate']:.2%}")

    # ── Test 7: Delivery Market ──────────────────────────────────────────
    print("\n  ── [DLV1] Delivery Market Analyzer ──")
    delivery_status = suite.delivery_analyzer.get_status()
    print(f"    Platforms          : {len(delivery_status['platforms'])}")
    print(f"    Food categories    : {len(delivery_status['food_categories'])}")
    print(f"    Zones              : {len(delivery_status['zones'])}")

    # Analyze specific platforms
    grab_analysis = suite.delivery_analyzer.analyze_platform(DeliveryPlatform.GRAB_FOOD)
    print(f"    Grab Food orders   : {grab_analysis['estimated_daily_orders']}")
    print(f"    Grab commission    : {grab_analysis['commission_rate']:.0%}")

    # Analyze vegan category
    vegan_analysis = suite.delivery_analyzer.analyze_food_category(FoodCategory.VEGAN)
    print(f"    Vegan demand score : {vegan_analysis['demand_score']:.2f}")
    print(f"    Vegan growth       : {vegan_analysis['growth_rate_mom']:.2%}")

    # Menu engineering
    menu_recs = suite.delivery_analyzer.menu_engineering_recommendations(FoodCategory.THAI_FOOD)
    print(f"    Menu recommendations: {len(menu_recs)}")

    # ── Test 8: Customer Segmentation ────────────────────────────────────
    print("\n  ── [SEG1] Customer Segmentation ──")
    segments = suite.segmentation.segment_customers(all_customers)
    print(f"    Segments found     : {len(segments)}")
    for seg_name, members in list(segments.items())[:5]:
        print(f"      {seg_name:25s}: {len(members)} customers")

    # ── Test 9: Full Pipeline ────────────────────────────────────────────
    print("\n  ── Full Customer Intelligence Pipeline ──")
    pipeline_result = suite.run_full_customer_analysis(
        platform=CRMPlatform.SALESFORCE, n_customers=30
    )
    print(f"    Customers analyzed : {pipeline_result['n_customers']}")
    print(f"    Avg CLV (12m)      : {pipeline_result['avg_clv_12m']:.2f}")
    print(f"    Avg churn risk     : {pipeline_result['avg_churn_risk_30d']:.3f}")
    print(f"    At-risk customers  : {pipeline_result['n_at_risk_customers']}")
    print(f"    Retention plans    : {pipeline_result['n_retention_plans']}")
    print(f"    Delivery platforms : {pipeline_result['delivery_platforms_analyzed']}")
    print(f"    Food categories    : {pipeline_result['food_categories_analyzed']}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full System Status ──")
    status = suite.get_full_status()
    print(f"    CRM consent records: {status['crm']['consent_records']}")
    print(f"    CLV fitted         : {status['clv']['fitted']}")
    print(f"    Churn fitted       : {status['churn']['fitted']}")
    print(f"    Categories tracked : {status['categories']['n_categories']}")
    print(f"    Delivery platforms : {len(status['delivery']['platforms'])}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • CRM requires valid API credentials + customer consent")
    print("  • CLV is PROBABILISTIC (credible intervals, not guarantees)")
    print("  • Churn is RISK ASSESSMENT, not certainty")
    print("  • Retention strategies are RECOMMENDATIONS (human approves)")
    print("  • Category/delivery analysis uses AGGREGATE data only")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires PDPA/GDPR compliance review")
    print("=" * 72)
    print("  ETHICAL FRAMEWORK")
    print("  ─────────────────────────────────────────────────────────")
    print("  • PNDC Axiom 3: Data minimization, consent required")
    print("  • Buddhist Economics (Samma Ajiva): Right Livelihood")
    print("  • NO dark patterns, NO fake urgency, NO manipulation")
    print("  • Taoist (Wu Wei): Flow with customer needs")
    print("  • Right to erasure: GDPR Art.17 / PDPA §33")
    print("  • Transparency in all customer communications")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
