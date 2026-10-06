"""
nsem/hardware/interlock.py

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026

"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class InterlockDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONSTRAIN = "constrain"


@dataclass
class SafetyEnvelope:
    max_charge_uc_cm2: float
    max_current_ua: float
    max_pulse_width_us: float
    max_temperature_rise_c: float
    allowed_electrodes: frozenset
    therapy_mode: str


@dataclass
class ValidationReceipt:
    payload_digest: str
    envelope_token: str
    implant_id: str
    decision: InterlockDecision
    timestamp_ns: int
    nonce: bytes


class SynapticInterlock:
    """
    Software-side model of the hardware interlock.

    In silicon this is a secure microcontroller with an independent
    watchdog, a charge-balance monitor, and a temperature sensor.
    Here we model the policy layer so it can be unit-tested and
    differentiable-checkpointed during training.
    """

    def __init__(
        self,
        envelope: SafetyEnvelope,
        implant_id: str,
        secret: bytes,
    ):
        self.envelope = envelope
        self.implant_id = implant_id
        self._secret = secret
        self._nonce_counter = 0

    def validate(
        self,
        command: dict,               # {"charge_uc_cm2": ..., "current_ua": ..., ...}
        patient_state: dict,
        device_health: dict,
    ) -> tuple[InterlockDecision, ValidationReceipt]:
        reasons = []

        if command["charge_uc_cm2"] > self.envelope.max_charge_uc_cm2:
            reasons.append("charge_density_exceeded")
        if command["current_ua"] > self.envelope.max_current_ua:
            reasons.append("current_exceeded")
        if command["pulse_width_us"] > self.envelope.max_pulse_width_us:
            reasons.append("pulse_width_exceeded")
        if patient_state.get("temperature_rise_c", 0) > self.envelope.max_temperature_rise_c:
            reasons.append("temperature_limit")
        if not set(command.get("electrodes", [])).issubset(self.envelope.allowed_electrodes):
            reasons.append("electrode_not_authorized")
        if device_health.get("charge_balance_residual_uc", 0) > 0.05:
            reasons.append("charge_imbalance")
        if device_health.get("impedance_mohm", 0) > 10.0:
            reasons.append("impedance_out_of_range")

        decision = InterlockDecision.DENY if reasons else InterlockDecision.ALLOW
        self._nonce_counter += 1
        receipt = ValidationReceipt(
            payload_digest=hashlib.sha256(repr(command).encode()).hexdigest(),
            envelope_token=hashlib.sha256(
                repr(self.envelope).encode() + self._secret
            ).hexdigest(),
            implant_id=self.implant_id,
            decision=decision,
            timestamp_ns=time.time_ns(),
            nonce=self._nonce_counter.to_bytes(8, "little"),
        )
        return decision, receipt
