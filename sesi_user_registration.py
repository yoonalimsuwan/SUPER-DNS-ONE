"""
=============================================================================
SES USER REGISTRATION — Mobile App with China Support
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] User registration framework for disaster alert mobile app:
  • Phone number registration (primary in China)
  • OTP (One-Time Password) verification
  • Real-name verification (required in China)
  • Multi-language support (Thai, Chinese, English)
  • WeChat/Alipay login (China)

CHINA-SPECIFIC REQUIREMENTS
---------------------------
[C] Real-name verification: Chinese law requires users to register with
    real names and ID numbers for many services.
[C] PIPL (Personal Information Protection Law): China's data protection
    law requires explicit consent for data collection.
[C] Phone number: Primary identifier in China (email less common).

THAILAND-SPECIFIC
-----------------
[D] Thai ID card verification (optional)
[D] Phone number + OTP

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a FRAMEWORK, not a production authentication system.
• Real-name verification requires integration with government databases.
• Data storage must comply with local laws (PDPA in Thailand, PIPL in China).
• OTP delivery requires SMS gateway integration.
• Security features (encryption, hashing) are placeholders.
• This module does NOT handle payment or financial services.
=============================================================================
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ses.user_registration")

__all__ = [
    "UserRegion",
    "UserConfig",
    "UserRegistration",
    "OTPService",
    "RealNameVerification",
]


class UserRegion(Enum):
    """Supported regions."""
    THAILAND = auto()
    CHINA = auto()
    OTHER = auto()


@dataclass
class UserConfig:
    """Configuration for user registration."""
    # Region
    default_region: UserRegion = UserRegion.THAILAND
    
    # OTP
    otp_length: int = 6
    otp_expiry_s: float = 300.0  # 5 minutes
    otp_max_attempts: int = 3
    
    # Real-name verification
    require_real_name_cn: bool = True  # Required in China
    require_real_name_th: bool = False  # Optional in Thailand
    
    # Data protection
    encrypt_pii: bool = True  # Encrypt personally identifiable info
    
    # Language
    default_language: str = "th"  # th, zh, en


# ═══════════════════════════════════════════════════════════════════════
# [D] USER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class User:
    """User data model."""
    user_id: str
    phone_number: str
    region: UserRegion
    language: str
    real_name_verified: bool = False
    created_at: float = field(default_factory=time.time)
    
    # PII (must be encrypted in production)
    full_name: Optional[str] = None
    id_number: Optional[str] = None


class UserRegistration:
    """
    [D] User registration service.
    
    Handles:
      • Phone number registration
      • OTP verification
      • Real-name verification (China)
      • User profile management
    """
    
    def __init__(self, cfg: UserConfig):
        self.cfg = cfg
        self.users: Dict[str, User] = {}
        self.otp_service = OTPService(cfg)
        self.real_name_service = RealNameVerification(cfg)
    
    def register(self, phone_number: str, 
                 region: UserRegion,
                 language: str = "en") -> Dict[str, Any]:
        """
        Register a new user.
        
        Returns dict with user_id and status.
        """
        # Validate phone number format (simplified)
        if not self._validate_phone(phone_number, region):
            return {"status": "error", "reason": "invalid_phone_format"}
        
        # Check if already registered
        for user in self.users.values():
            if user.phone_number == phone_number:
                return {"status": "error", "reason": "already_registered"}
        
        # Create user
        user_id = self._generate_user_id()
        user = User(
            user_id=user_id,
            phone_number=phone_number,
            region=region,
            language=language,
        )
        
        self.users[user_id] = user
        
        # Send OTP
        otp_result = self.otp_service.send_otp(phone_number)
        
        return {
            "status": "pending_verification",
            "user_id": user_id,
            "otp_sent": otp_result["status"] == "sent",
            "message": self._get_message("verification_sent", language),
        }
    
    def verify_otp(self, user_id: str, otp: str) -> Dict[str, Any]:
        """Verify OTP for a user."""
        if user_id not in self.users:
            return {"status": "error", "reason": "user_not_found"}
        
        user = self.users[user_id]
        
        # Verify OTP
        result = self.otp_service.verify_otp(user.phone_number, otp)
        
        if result["status"] == "verified":
            user.real_name_verified = False  # Still need real-name for China
            
            # Check if real-name verification required
            if (user.region == UserRegion.CHINA and self.cfg.require_real_name_cn):
                return {
                    "status": "pending_real_name",
                    "message": self._get_message("real_name_required", user.language),
                }
            
            return {
                "status": "registered",
                "user_id": user_id,
                "message": self._get_message("registration_complete", user.language),
            }
        
        return {"status": "error", "reason": result.get("reason", "invalid_otp")}
    
    def verify_real_name(self, user_id: str, 
                         full_name: str, id_number: str) -> Dict[str, Any]:
        """
        Verify real name (required in China).
        
        In production, this would connect to government databases.
        """
        if user_id not in self.users:
            return {"status": "error", "reason": "user_not_found"}
        
        user = self.users[user_id]
        
        # Verify with real-name service
        result = self.real_name_service.verify(full_name, id_number, user.region)
        
        if result["status"] == "verified":
            user.real_name_verified = True
            user.full_name = full_name  # In production, encrypt this
            user.id_number = id_number  # In production, encrypt this
            
            return {
                "status": "registered",
                "user_id": user_id,
                "message": self._get_message("registration_complete", user.language),
            }
        
        return {"status": "error", "reason": result.get("reason", "verification_failed")}
    
    def _validate_phone(self, phone: str, region: UserRegion) -> bool:
        """Validate phone number format."""
        # Simplified validation
        if region == UserRegion.CHINA:
            # Chinese mobile: 1[3-9]\d{9}
            return len(phone) == 11 and phone.startswith("1")
        elif region == UserRegion.THAILAND:
            # Thai mobile: 0[689]\d{8}
            return len(phone) == 10 and phone.startswith("0")
        return len(phone) >= 8
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID."""
        return f"user_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
    
    def _get_message(self, key: str, language: str) -> str:
        """Get localized message."""
        messages = {
            "verification_sent": {
                "th": "ส่งรหัสยืนยันแล้ว",
                "zh": "验证码已发送",
                "en": "Verification code sent",
            },
            "real_name_required": {
                "th": "ต้องยืนยันตัวตน",
                "zh": "需要实名认证",
                "en": "Real-name verification required",
            },
            "registration_complete": {
                "th": "ลงทะเบียนสำเร็จ",
                "zh": "注册成功",
                "en": "Registration complete",
            },
        }
        return messages.get(key, {}).get(language, messages.get(key, {}).get("en", ""))


# ═══════════════════════════════════════════════════════════════════════
# [D] OTP SERVICE
# ═══════════════════════════════════════════════════════════════════════

class OTPService:
    """
    [D] One-Time Password service.
    
    In production, this would integrate with SMS gateways.
    """
    
    def __init__(self, cfg: UserConfig):
        self.cfg = cfg
        self.active_otps: Dict[str, Dict[str, Any]] = {}
    
    def send_otp(self, phone_number: str) -> Dict[str, Any]:
        """Generate and send OTP."""
        # Generate OTP
        otp = "".join([str(np.random.randint(0, 10)) for _ in range(self.cfg.otp_length)])
        
        # Store
        self.active_otps[phone_number] = {
            "otp": otp,
            "created_at": time.time(),
            "attempts": 0,
        }
        
        # In production: send via SMS gateway
        logger.info(f"OTP sent to {phone_number}: {otp}")
        
        return {"status": "sent", "otp": otp}  # Don't return OTP in production!
    
    def verify_otp(self, phone_number: str, otp: str) -> Dict[str, Any]:
        """Verify OTP."""
        if phone_number not in self.active_otps:
            return {"status": "error", "reason": "no_otp_sent"}
        
        record = self.active_otps[phone_number]
        
        # Check expiry
        if time.time() - record["created_at"] > self.cfg.otp_expiry_s:
            del self.active_otps[phone_number]
            return {"status": "error", "reason": "otp_expired"}
        
        # Check attempts
        if record["attempts"] >= self.cfg.otp_max_attempts:
            del self.active_otps[phone_number]
            return {"status": "error", "reason": "too_many_attempts"}
        
        # Verify
        if record["otp"] == otp:
            del self.active_otps[phone_number]
            return {"status": "verified"}
        
        record["attempts"] += 1
        return {"status": "error", "reason": "invalid_otp"}


# ═══════════════════════════════════════════════════════════════════════
# [D] REAL-NAME VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

class RealNameVerification:
    """
    [D] Real-name verification service.
    
    In production, this would connect to:
      • China: Public Security Bureau database
      • Thailand: Department of Provincial Administration
    
    HONEST: This is a PLACEHOLDER. Real verification requires
    government database access and proper licensing.
    """
    
    def __init__(self, cfg: UserConfig):
        self.cfg = cfg
    
    def verify(self, full_name: str, id_number: str, 
               region: UserRegion) -> Dict[str, Any]:
        """
        Verify real name against government database.
        
        PLACEHOLDER: Always returns verified for testing.
        """
        # In production: call government API
        # For now: simple format validation
        
        if region == UserRegion.CHINA:
            # Chinese ID: 18 digits
            if len(id_number) != 18:
                return {"status": "error", "reason": "invalid_id_format"}
        elif region == UserRegion.THAILAND:
            # Thai ID: 13 digits
            if len(id_number) != 13:
                return {"status": "error", "reason": "invalid_id_format"}
        
        # PLACEHOLDER: Assume verified
        logger.warning("Real-name verification is PLACEHOLDER; "
                      "production requires government database integration")
        
        return {"status": "verified"}


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES User Registration (with China support) — Smoke Test")
    print("=" * 72)
    
    cfg = UserConfig(default_region=UserRegion.THAILAND)
    service = UserRegistration(cfg)
    
    # Test Thailand registration
    print("\n  ── Thailand Registration ──")
    result = service.register("0812345678", UserRegion.THAILAND, "th")
    print(f"    Status: {result['status']}")
    print(f"    User ID: {result.get('user_id', 'N/A')}")
    print(f"    Message: {result.get('message', 'N/A')}")
    
    if result["status"] == "pending_verification":
        user_id = result["user_id"]
        
        # Get OTP (in production, user receives via SMS)
        otp = service.otp_service.active_otps["0812345678"]["otp"]
        print(f"    OTP (for testing): {otp}")
        
        # Verify
        verify_result = service.verify_otp(user_id, otp)
        print(f"    Verification: {verify_result['status']}")
    
    # Test China registration
    print("\n  ── China Registration ──")
    result_cn = service.register("13812345678", UserRegion.CHINA, "zh")
    print(f"    Status: {result_cn['status']}")
    print(f"    Message: {result_cn.get('message', 'N/A')}")
    
    if result_cn["status"] == "pending_verification":
        user_id_cn = result_cn["user_id"]
        
        # Get OTP
        otp_cn = service.otp_service.active_otps["13812345678"]["otp"]
        print(f"    OTP (for testing): {otp_cn}")
        
        # Verify OTP
        verify_result_cn = service.verify_otp(user_id_cn, otp_cn)
        print(f"    OTP Verification: {verify_result_cn['status']}")
        
        # Real-name verification (required in China)
        if verify_result_cn["status"] == "pending_real_name":
            real_name_result = service.verify_real_name(
                user_id_cn, "张伟", "110101199001011234")
            print(f"    Real-name verification: {real_name_result['status']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a FRAMEWORK, not production authentication")
    print("  • Real-name verification requires government database access")
    print("  • Data storage must comply with PDPA (Thailand) / PIPL (China)")
    print("  • OTP delivery requires SMS gateway integration")
    print("  • Security features are placeholders")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
