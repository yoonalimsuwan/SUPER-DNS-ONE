import unittest
from tests import _torch  # noqa: F401  (skips module if torch missing)
import torch

from nsem.hardware.dac import DACSpec, NeuralStimulatorFrontend, QuantizedDAC
from nsem.hardware.chemical import MicrofluidicPulseModel


class TestDAC(unittest.TestCase):
    def test_spec_units(self):
        s = DACSpec()
        self.assertAlmostEqual(s.max_safe_current_ua, 150.0, places=6)
        self.assertAlmostEqual(s.charge_density_uc_cm2(100.0), 20.0, places=6)

    def test_zero_in_zero_out_and_symmetric(self):
        s = DACSpec()
        x = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        i = QuantizedDAC.apply(x, s)
        self.assertEqual(i[0, 2].item(), 0.0)                 # was +50 uA before
        self.assertTrue(torch.allclose(i, -i.flip(-1)))
        self.assertAlmostEqual(i[0, -1].item(), 100.0, places=4)

    def test_quantisation_error_within_half_lsb(self):
        s = DACSpec()
        x = torch.linspace(-1, 1, 1001).unsqueeze(0)
        err = (QuantizedDAC.apply(x, s) - x * s.full_scale_current_ua).abs().max()
        self.assertLessEqual(err.item(), s.lsb_current_ua / 2 + 1e-5)

    def test_saturates_at_charge_limit(self):
        s = DACSpec(pulse_width_us=1000.0)                    # limit -> 30 uA
        i = QuantizedDAC.apply(torch.tensor([[1.0, -1.0]]), s)
        self.assertLessEqual(i.abs().max().item(), 30.0 + 1e-5)

    def test_ste_gradient(self):
        s = DACSpec()
        x = torch.tensor([0.3, 2.0], requires_grad=True)
        QuantizedDAC.apply(x, s).sum().backward()
        self.assertAlmostEqual(x.grad[0].item(), 100.0, places=4)
        self.assertEqual(x.grad[1].item(), 0.0)               # outside [-1,1]

    def test_frontend_balanced_at_init_and_trainable(self):
        spec = DACSpec(channels=8)
        fe = NeuralStimulatorFrontend(spec)
        out = fe(torch.rand(4, 8) * 2 - 1)
        self.assertLess(out["residual_charge_uc"].abs().max().item(), 1e-6)
        self.assertLessEqual(out["charge_density_uc_cm2"].max().item(), 30.0 + 1e-4)
        out["cathodic_ua"].sum().backward()
        self.assertIsNotNone(fe.gain_trim.grad)
        # residual is now a real, differentiable quantity
        fe.zero_grad()
        fe(torch.rand(4, 8))["residual_charge_uc"].abs().sum().backward()
        self.assertGreater(fe.anodic_gain_raw.grad.abs().sum().item(), 0.0)


class TestChemical(unittest.TestCase):
    def test_gradient_alive_at_extreme_params(self):
        m = MicrofluidicPulseModel(n_receptors=4)
        with torch.no_grad():
            m.log_tau.fill_(-20.0); m.log_kd.fill_(-20.0)     # old clamp -> zero grad
        out = m(torch.full((2, 4), 0.5))
        out["receptor_occupancy"].sum().backward()
        self.assertGreater(m.log_tau.grad.abs().sum().item(), 0.0)
        self.assertGreater(m.log_kd.grad.abs().sum().item(), 0.0)

    def test_volume_matches_flow_times_pulse(self):
        m = MicrofluidicPulseModel(n_receptors=3)
        o = m(torch.ones(1, 3))
        self.assertTrue(torch.allclose(o["volume_nl"], torch.full((1, 3), 2.5)))


if __name__ == "__main__":
    unittest.main()
