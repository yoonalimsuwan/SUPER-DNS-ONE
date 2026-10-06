import dataclasses
import math
import os
import unittest

from nsem.hardware.interlock import (
    InterlockDecision as D, ReceiptError, ReceiptVerifier,
    SafetyEnvelope, SynapticInterlock,
)

SECRET = os.urandom(32)
ENV = SafetyEnvelope(
    max_charge_uc_cm2=30.0, max_current_ua=100.0, max_pulse_width_us=200.0,
    max_temperature_rise_c=1.0, allowed_electrodes=frozenset(range(8)),
    therapy_mode="sensory_restoration", max_volume_nl=2.5,
)
GOOD_CMD = dict(charge_uc_cm2=20.0, current_ua=80.0, pulse_width_us=200.0,
                electrodes=[0, 1, 2])
PATIENT = dict(temperature_rise_c=0.2)
HEALTH = dict(charge_balance_residual_uc=0.01, impedance_mohm=2.0)


def mk():
    return SynapticInterlock(ENV, "IMPLANT-T", SECRET)


class TestInterlock(unittest.TestCase):
    def test_allow(self):
        d, r = mk().validate(GOOD_CMD, PATIENT, HEALTH)
        self.assertEqual(d, D.ALLOW)
        self.assertEqual(r.scale, 1.0)

    def test_nan_is_denied(self):
        # Original code let NaN pass every comparison.
        for key in ("charge_uc_cm2", "current_ua", "pulse_width_us"):
            cmd = {**GOOD_CMD, key: float("nan")}
            d, r = mk().validate(cmd, PATIENT, HEALTH)
            self.assertEqual(d, D.DENY, key)

    def test_missing_sensor_is_denied(self):
        d, _ = mk().validate(GOOD_CMD, {}, HEALTH)
        self.assertEqual(d, D.DENY)
        d, _ = mk().validate(GOOD_CMD, PATIENT, {})
        self.assertEqual(d, D.DENY)

    def test_negative_current_is_checked(self):
        cmd = {**GOOD_CMD, "current_ua": -500.0}
        d, r = mk().validate(cmd, PATIENT, HEALTH)
        self.assertEqual(d, D.CONSTRAIN)
        self.assertLess(r.scale, 1.0)
        self.assertLessEqual(500.0 * r.scale, ENV.max_current_ua)

    def test_constrain_scale_satisfies_all_limits(self):
        cmd = {**GOOD_CMD, "charge_uc_cm2": 90.0, "current_ua": 150.0}
        d, r = mk().validate(cmd, PATIENT, HEALTH)
        self.assertEqual(d, D.CONSTRAIN)
        self.assertLessEqual(90.0 * r.scale, ENV.max_charge_uc_cm2)
        self.assertLessEqual(150.0 * r.scale, ENV.max_current_ua)

    def test_hard_denies(self):
        cases = [
            ({**GOOD_CMD, "pulse_width_us": 400.0}, PATIENT, HEALTH),
            ({**GOOD_CMD, "electrodes": [99]}, PATIENT, HEALTH),
            (GOOD_CMD, dict(temperature_rise_c=2.0), HEALTH),
            (GOOD_CMD, PATIENT, {**HEALTH, "charge_balance_residual_uc": 0.5}),
            (GOOD_CMD, PATIENT, {**HEALTH, "impedance_mohm": 50.0}),
            ({**GOOD_CMD, "volume_nl": 10.0}, PATIENT, HEALTH),
        ]
        for c, p, h in cases:
            d, r = mk().validate(c, p, h)
            self.assertEqual(d, D.DENY, (c, p, h))
            self.assertEqual(r.scale, 0.0)

    def test_weak_secrets_rejected(self):
        for bad in (b"\x00" * 32, b"short", b"ab" * 16):
            with self.assertRaises((ValueError, TypeError)):
                SynapticInterlock(ENV, "x", bad)

    def test_receipt_verify_ok_then_replay(self):
        il, ver = mk(), ReceiptVerifier(SECRET, "IMPLANT-T")
        _, r = il.validate(GOOD_CMD, PATIENT, HEALTH)
        ver.verify(r)
        with self.assertRaises(ReceiptError) as cm:
            ver.verify(r)
        self.assertEqual(str(cm.exception), "replay")

    def test_receipt_tamper_detected(self):
        il, ver = mk(), ReceiptVerifier(SECRET, "IMPLANT-T")
        _, r = il.validate({**GOOD_CMD, "current_ua": 500.0}, PATIENT, HEALTH)
        forged = dataclasses.replace(r, decision=D.ALLOW, scale=1.0)
        with self.assertRaises(ReceiptError) as cm:
            ver.verify(forged)
        self.assertEqual(str(cm.exception), "bad_mac")

    def test_receipt_wrong_key_and_stale(self):
        il = mk()
        _, r = il.validate(GOOD_CMD, PATIENT, HEALTH)
        with self.assertRaises(ReceiptError):
            ReceiptVerifier(os.urandom(32), "IMPLANT-T").verify(r)
        with self.assertRaises(ReceiptError) as cm:
            ReceiptVerifier(SECRET, "IMPLANT-T", max_age_s=1.0).verify(
                r, now_ns=r.timestamp_ns + 10 * 10**9)
        self.assertEqual(str(cm.exception), "stale")

    def test_nonces_strictly_increase(self):
        il = mk()
        ns = [il.validate(GOOD_CMD, PATIENT, HEALTH)[1].nonce for _ in range(50)]
        self.assertEqual(ns, sorted(set(ns)))


if __name__ == "__main__":
    unittest.main()
