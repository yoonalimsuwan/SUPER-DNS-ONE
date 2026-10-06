import unittest
from tests import _torch  # noqa: F401
import math, os
import torch

from nsem.core import NSEMConfig, NeuroSensoryEnhancementModule
from nsem.hardware.dac import DACSpec
from nsem.hardware.interlock import InterlockDecision as D
from nsem.module_production import ProductionConfig, ProductionNSEM

SMALL = dict(biological_channels=32, hidden_dim=16, vision_latent_dim=8,
             audio_bands=4, chemical_receptors=6, motor_dof=8, dropout=0.0)
PATIENT = dict(temperature_rise_c=0.1)
HEALTH = dict(charge_balance_residual_uc=0.0, impedance_mohm=1.0)


def prod(**kw):
    cfg = ProductionConfig(nsem=NSEMConfig(**SMALL), dac=DACSpec(channels=8),
                           interlock_secret=os.urandom(32), **kw)
    return ProductionNSEM(cfg)


class TestCore(unittest.TestCase):
    def test_shapes_and_memory(self):
        m = NeuroSensoryEnhancementModule(NSEMConfig(**SMALL))
        out, mem = m(torch.randn(3, 32))
        self.assertEqual(out["vision_latent"].shape, (3, 8))
        self.assertEqual(out["motor_control"].shape, (3, 8))
        self.assertEqual(mem.shape, (3, 16))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))

    def test_every_parameter_receives_gradient(self):   # DDP static-graph claim
        m = NeuroSensoryEnhancementModule(NSEMConfig(**SMALL))
        out, _ = m(torch.randn(3, 32))
        sum(v.sum() for v in out.values()).backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])


class TestProduction(unittest.TestCase):
    def test_requires_state_by_default(self):
        with self.assertRaises(ValueError):
            prod()(torch.randn(2, 32))

    def test_allow(self):
        out, _ = prod()(torch.randn(2, 32), None, PATIENT, HEALTH)
        self.assertIs(out["interlock_decision"], D.ALLOW)

    def test_deny_zeroes_everything(self):
        out, _ = prod()(torch.randn(2, 32), None, dict(temperature_rise_c=9.0), HEALTH)
        self.assertIs(out["interlock_decision"], D.DENY)
        for k in ("dac_cathodic_ua", "dac_anodic_ua", "chem_flow_nl_s",
                  "chem_bolus_um", "chem_volume_nl", "chem_receptor_occupancy"):
            self.assertEqual(out[k].abs().max().item(), 0.0, k)

    def test_nan_sensor_is_denied(self):
        out, _ = prod()(torch.randn(2, 32), None,
                        dict(temperature_rise_c=float("nan")), HEALTH)
        self.assertIs(out["interlock_decision"], D.DENY)

    def test_constrain_scales_down(self):
        m = prod()
        with torch.no_grad():
            m.nsem.motor_head[0].bias.fill_(10.0)                  # tanh -> 1
            m.dac.anodic_gain_raw.fill_(math.log(math.expm1(1.4)))  # anodic 140 uA
        out, _ = m(torch.zeros(1, 32), None, PATIENT, HEALTH)
        self.assertIs(out["interlock_decision"], D.CONSTRAIN)
        self.assertLessEqual(out["dac_anodic_ua"].abs().max().item(), 100.0 + 1e-3)

    def test_weak_secret_rejected(self):
        with self.assertRaises(ValueError):
            ProductionNSEM(ProductionConfig(nsem=NSEMConfig(**SMALL),
                           dac=DACSpec(channels=8), interlock_secret=b"\x00" * 32))


if __name__ == "__main__":
    unittest.main()
