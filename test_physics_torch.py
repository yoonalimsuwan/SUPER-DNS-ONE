import unittest
from tests import _torch  # noqa: F401
import torch

from nsem.optogenetics.transport_fixes import DiffusionPhotonTransport, PennesBioheatSolverCG
from nsem.optogenetics.advanced_optogenetics import AdvancedOptogeneticEngine, KNOWN_OPSINS


class TestPennesCG(unittest.TestCase):
    def test_matches_dense_solve(self):
        n = 48
        s = PennesBioheatSolverCG(n, dz_mm=0.1, dt=1e-3, cg_iters=30).double()
        g = torch.Generator().manual_seed(0)
        Tp = 37.0 + 0.3 * torch.randn(2, n, generator=g, dtype=torch.float64)
        Q = 5e5 * torch.rand(2, n, generator=g, dtype=torch.float64)
        got = s(Tp, Q)
        L = torch.zeros(n, n, dtype=torch.float64)
        idx = torch.arange(n)
        L[idx, idx] = -2.0
        L[idx[1:], idx[:-1]] = 1.0
        L[idx[:-1], idx[1:]] = 1.0
        L = L / s.dz ** 2
        A = (1 / s.dt + s.omega_b) * torch.eye(n, dtype=torch.float64) - s.D * L
        rhs = Tp / s.dt + s.omega_b * s.T_a + Q / s.rho_c
        rhs[:, 0] += s.D * s.T_amb / s.dz ** 2
        rhs[:, -1] += s.D * s.T_amb / s.dz ** 2
        ref = torch.linalg.solve(A, rhs.T).T
        self.assertTrue(torch.allclose(got, ref, atol=1e-9))

    def test_no_source_stays_at_ambient(self):
        s = PennesBioheatSolverCG(16)
        T = torch.full((1, 16), 37.0)
        self.assertTrue(torch.allclose(s(T, torch.zeros(1, 16)), T, atol=1e-4))

    def test_heating_is_positive_and_differentiable(self):
        s = PennesBioheatSolverCG(16)
        T = torch.full((1, 16), 37.0)
        Q = torch.full((1, 16), 1e5, requires_grad=True)
        out = s(T, Q)
        self.assertGreater((out - 37.0).min().item(), 0.0)
        out.sum().backward()
        self.assertGreater(Q.grad.abs().sum().item(), 0.0)


class TestDiffusion(unittest.TestCase):
    def test_monotone_decay_and_gradient(self):
        m = DiffusionPhotonTransport(3, [0.5, 0.5, 0.5])
        I = [m(torch.tensor(1.0), layer_idx=i).item() for i in range(3)]
        self.assertTrue(0 < I[2] < I[1] < I[0] < 1)
        m(torch.tensor(1.0)).backward()
        self.assertLess(m.mu_a_raw.grad.sum().item(), 0.0)    # more absorption -> less light

    def test_known_value(self):
        # mu_a=0.1, mu_s=10, g=0.9 -> mu_eff = sqrt(3*0.1*1.1)
        m = DiffusionPhotonTransport(1, [1.0], 0.1, 10.0, 0.9)
        want = torch.exp(-torch.sqrt(torch.tensor(0.33)))
        self.assertAlmostEqual(m(torch.tensor(1.0)).item(), want.item(), places=4)


class TestKinetics(unittest.TestCase):
    def _engine(self, integ="expm"):
        lam = torch.linspace(400, 700, 31)
        specs = [KNOWN_OPSINS["ChR2"], KNOWN_OPSINS["Chrimson"], KNOWN_OPSINS["GtACR2"]]
        return AdvancedOptogeneticEngine(4, specs, lam, dt=1e-3, integrator=integ)

    def test_simplex_invariant_over_rollout(self):
        e = self._engine()
        x = e.initial_states(2)
        v = torch.full((2, 4), -65.0)
        irr = torch.zeros(2, 31); irr[:, 10] = 5.0
        for _ in range(300):
            _, x = e(irr, v, x)
            self.assertTrue(torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-4))
            self.assertGreaterEqual(x.min().item(), -1e-6)

    def test_dark_stays_dark_adapted(self):
        e = self._engine()
        x0 = e.initial_states(1)
        _, x1 = e(torch.zeros(1, 31), torch.full((1, 4), -65.0), x0)
        self.assertTrue(torch.allclose(x1[..., 0], torch.ones_like(x1[..., 0]), atol=5e-3))

    def test_gradients_reach_conductance(self):
        e = self._engine()
        i, _ = e(torch.full((1, 31), 3.0), torch.full((1, 4), -65.0), e.initial_states(1))
        i.sum().backward()
        self.assertGreater(e.g_raw.grad.abs().sum().item(), 0.0)

    def test_rk4_close_to_expm_small_dt(self):
        a, b = self._engine("expm"), self._engine("rk4")
        irr = torch.full((1, 31), 2.0); v = torch.full((1, 4), -65.0)
        xa = a.initial_states(1); xb = b.initial_states(1)
        for _ in range(50):
            _, xa = a(irr, v, xa); _, xb = b(irr, v, xb)
        self.assertTrue(torch.allclose(xa, xb, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
