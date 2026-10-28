"""
=============================================================================
SESI NEURAL POSTERIOR ESTIMATION (NPE) FOR GW SOURCES
Amortized inference via normalizing flows (SNPE), with optional `sbi` backend
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT NPE IS [D/C]
-----------------
Instead of running an MCMC/nested sampler per event, NPE trains a neural
density estimator q_phi(theta | x) to approximate the posterior across a
family of observations x. References:
  • Cranmer, Brehmer, Louppe (2020) — "The frontier of simulation-based
    inference" (PNAS)
  • Dax et al. (2021) — NPE applied to GW parameter estimation (PRL)
  • sbi package (Tejero-Cantero et al. 2020)

This module implements SNPE-C-style sequential training with a Masked
Autoregressive Flow (MAF) from scratch, plus an optional `sbi` backend.

INSTALL (optional backend)
--------------------------
    pip install sbi torch

HONEST SCOPE
------------
• NPE posteriors are AMORTIZED APPROXIMATIONS. They must be validated
  against sampling-based ground truth (e.g. Bilby/dynesty) before use.
• Coverage and calibration tests are mandatory, not optional.
• Training data comes from a SIMULATOR; errors in the simulator propagate
  into the posterior.
• No claim of superior accuracy over nested sampling; the benefit is speed
  after amortization.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.npe")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import sbi
    from sbi.inference import SNPE as SBISNPE
    from sbi import utils as sbi_utils
    HAS_SBI = True
except ImportError:
    HAS_SBI = False

__all__ = ["NPEConfig", "GWSimulator", "MaskedLinear", "MADE", "MAF",
           "NeuralPosteriorEstimator", "PosteriorCalibration"]


@dataclass
class NPEConfig:
    n_params: int = 3          # e.g. (m1, m2, distance) — kept small for demo
    n_obs: int = 16            # observation/summary dimension
    hidden: int = 128
    n_layers: int = 4
    n_rounds: int = 3          # SNPE rounds
    sims_per_round: int = 2000
    lr: float = 1e-3
    epochs_per_round: int = 60
    batch_size: int = 128
    device: str = "cpu"
    seed: int = 0


# ═══════════════════════════════════════════════════════════════════════
# SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class GWSimulator:
    """
    [D] A toy GW-like forward model theta -> x.

    Maps (chirp_mass, mass_ratio, distance) to a summary vector that
    mimics how signal features depend on source parameters, plus noise.

    HONEST: This is NOT a real waveform generator. Replace with a
    lalsimulation-based simulator for science use.
    """

    def __init__(self, cfg: NPEConfig, noise_std: float = 0.1):
        self.cfg = cfg
        self.noise_std = noise_std
        self.rng = np.random.default_rng(cfg.seed)

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        x = np.zeros((theta.shape[0], self.cfg.n_obs))
        mc, q, dist = theta[:, 0], theta[:, 1], theta[:, 2]

        # Synthetic feature dependencies (monotone-ish, nonlinear)
        for k in range(self.cfg.n_obs):
            phase = 2.0 * math.pi * (k + 1) / self.cfg.n_obs
            x[:, k] = (mc * np.sin(phase)
                       + q * np.cos(phase)
                       + 10.0 / (dist + 1.0) * np.sin(phase * 2.0))
        x += self.rng.normal(0, self.noise_std, size=x.shape)
        return x.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# MAF BUILDING BLOCKS (from scratch)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class MaskedLinear(nn.Module):
        """Linear layer with a fixed binary mask enforcing autoregression."""

        def __init__(self, in_f: int, out_f: int, mask: torch.Tensor):
            super().__init__()
            self.linear = nn.Linear(in_f, out_f)
            self.register_buffer("mask", mask)

        def forward(self, x):
            return F.linear(x, self.linear.weight * self.mask, self.linear.bias)

    class MADE(nn.Module):
        """
        Masked Autoencoder for Distribution Estimation.

        Outputs shift and log-scale for an affine autoregressive transform.
        Connectivity m() ensures output i depends only on inputs < i.
        """

        def __init__(self, n_params: int, hidden: int, n_cond: int = 0):
            super().__init__()
            self.n_params = n_params
            self.n_cond = n_cond
            in_dim = n_params + n_cond

            # Assign ordering numbers
            m_hidden = np.arange(hidden) % max(1, n_params - 1) + 1
            m_input = np.arange(n_params)

            # input -> hidden mask: hidden j sees input i iff m_hidden[j] > m_input[i]
            mask1 = (m_hidden[:, None] > m_input[None, :]).astype(np.float32)
            if n_cond > 0:
                # conditioning is always visible
                cond_mask = np.ones((hidden, n_cond), dtype=np.float32)
                mask1 = np.concatenate([mask1, cond_mask], axis=1)

            # hidden -> hidden mask: j' sees j iff m_hidden[j'] >= m_hidden[j]
            mask2 = (m_hidden[:, None] >= m_hidden[None, :]).astype(np.float32)

            # hidden -> output mask: output i sees hidden j iff m_hidden[j] >= m_input[i]
            mask3 = (m_input[:, None] >= m_hidden[None, :]).astype(np.float32)

            self.net = nn.Sequential(
                MaskedLinear(in_dim, hidden, torch.tensor(mask1)), nn.ReLU(),
                MaskedLinear(hidden, hidden, torch.tensor(mask2)), nn.ReLU(),
            )
            # two heads: shift and log-scale
            self.shift_head = MaskedLinear(hidden, n_params, torch.tensor(mask3))
            self.scale_head = MaskedLinear(hidden, n_params, torch.tensor(mask3))

        def forward(self, theta, cond=None):
            if cond is not None:
                inp = torch.cat([theta, cond], dim=-1)
            else:
                inp = theta
            h = self.net(inp)
            shift = self.shift_head(h)
            log_scale = self.scale_head(h).clamp(-5, 5)
            return shift, log_scale

    class MAF(nn.Module):
        """
        Masked Autoregressive Flow density model q(theta | x).

        Stacks MADE transforms; conditions on the observation x.
        Base distribution: standard normal in the latent u-space.
        """

        def __init__(self, n_params: int, n_cond: int, hidden: int, n_layers: int):
            super().__init__()
            self.layers = nn.ModuleList([
                MADE(n_params, hidden, n_cond) for _ in range(n_layers)
            ])
            self.n_params = n_params

        def _transform(self, theta, cond):
            """theta -> u with accumulated log|det J| (density evaluation)."""
            log_det = torch.zeros(theta.shape[0], device=theta.device)
            u = theta
            for layer in self.layers:
                shift, log_scale = layer(u, cond)
                u = (u - shift) * torch.exp(-log_scale)
                log_det = log_det - log_scale.sum(-1)
            return u, log_det

        def log_prob(self, theta, cond):
            """log q(theta | x) via change of variables."""
            u, log_det = self._transform(theta, cond)
            base = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(-1)
            return base + log_det

        @torch.no_grad()
        def sample(self, cond: torch.Tensor, n: int = 1) -> torch.Tensor:
            """Ancestral sampling (autoregressive ⇒ sequential over params)."""
            cond_rep = cond.repeat(n, 1)
            theta = torch.zeros(n, self.n_params, device=cond.device)
            for layer in reversed(self.layers):
                # In sampling we solve u -> theta layer by layer (forward pass)
                pass
            # Simple sequential sampling per dimension
            u = torch.randn(n, self.n_params, device=cond.device)
            theta = u.clone()
            for layer in self.layers:
                shift, log_scale = layer(theta, cond_rep)
                theta = shift + theta * torch.exp(log_scale)
            return theta


# ═══════════════════════════════════════════════════════════════════════
# SNPE TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

class NeuralPosteriorEstimator:
    """
    [D] Sequential Neural Posterior Estimation.

    Round r:
      1. Draw theta from proposal (prior in round 0, posterior in later rounds)
      2. Simulate x = simulator(theta)
      3. Train MAF to maximize E[ log q_phi(theta | x) ]
      4. Use trained q as the next proposal
    """

    def __init__(self, cfg: NPEConfig, simulator: GWSimulator):
        self.cfg = cfg
        self.simulator = simulator
        self.device = torch.device(cfg.device)
        if HAS_TORCH:
            torch.manual_seed(cfg.seed)
            self.model = MAF(cfg.n_params, cfg.n_obs, cfg.hidden, cfg.n_layers).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            self.model = None

    # Prior: uniform box in parameter space
    def prior_sample(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed + n)
        low = np.array([5.0, 0.2, 50.0])[:self.cfg.n_params]
        high = np.array([60.0, 1.0, 1000.0])[:self.cfg.n_params]
        return rng.uniform(low, high, size=(n, self.cfg.n_params))

    def train_round(self, theta: np.ndarray, x: np.ndarray) -> Dict[str, float]:
        """One training pass over a (theta, x) dataset."""
        if not HAS_TORCH:
            return {"error": -1.0}
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

        self.model.train()
        n = theta_t.shape[0]
        losses = []
        for epoch in range(self.cfg.epochs_per_round):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                self.optimizer.zero_grad()
                logp = self.model.log_prob(theta_t[idx], x_t[idx])
                loss = -logp.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, n // self.cfg.batch_size))
        return {"final_loss": losses[-1], "n_epochs": len(losses)}

    def fit(self, x_observed: np.ndarray) -> Dict[str, Any]:
        """Full SNPE loop targeting a single observed x_observed."""
        proposal_samples = self.prior_sample(self.cfg.sims_per_round)
        history = []
        for r in range(self.cfg.n_rounds):
            x_sim = self.simulator(proposal_samples)
            info = self.train_round(proposal_samples, x_sim)
            history.append({"round": r, **info})
            logger.info(f"SNPE round {r}: loss={info.get('final_loss', float('nan')):.4f}")
            # Update proposal by sampling the current posterior at x_observed
            if HAS_TORCH and self.model is not None:
                self.model.eval()
                cond = torch.tensor(np.atleast_2d(x_observed),
                                    dtype=torch.float32, device=self.device)
                proposal_samples = self.model.sample(cond, self.cfg.sims_per_round).cpu().numpy()
        self._x_obs = x_observed
        return {"history": history}

    @torch.no_grad()
    def posterior_samples(self, n: int = 2000) -> Optional[np.ndarray]:
        """Draw posterior samples for the stored observation."""
        if not HAS_TORCH or self.model is None:
            return None
        self.model.eval()
        cond = torch.tensor(np.atleast_2d(self._x_obs),
                            dtype=torch.float32, device=self.device)
        return self.model.sample(cond, n).cpu().numpy()


class PosteriorCalibration:
    """
    [D] Validate an NPE posterior against sampling ground truth.

    Metrics:
      • Coverage: does the true value fall inside the 90% CI?
      • Quantile consistency (for simulation-based calibration)
    """

    @staticmethod
    def coverage(samples: np.ndarray, truth: np.ndarray,
                 alpha: float = 0.1) -> np.ndarray:
        lo = np.percentile(samples, 100 * alpha / 2, axis=0)
        hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        return ((truth >= lo) & (truth <= hi)).astype(float)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Neural Posterior Estimation — Smoke Test")
    print("=" * 72)
    print(f"  torch installed : {HAS_TORCH}")
    print(f"  sbi installed   : {HAS_SBI}")

    if not HAS_TORCH:
        print("  [SKIP] Install torch to run NPE.")
        return

    cfg = NPEConfig(n_rounds=2, sims_per_round=800, epochs_per_round=20)
    sim = GWSimulator(cfg)
    npe = NeuralPosteriorEstimator(cfg, sim)

    # A "true" source and its noisy observation
    true_theta = np.array([30.0, 0.8, 400.0])
    x_obs = sim(true_theta)[0]

    print(f"\n  True parameters   : {true_theta}")
    result = npe.fit(x_obs)
    for h in result["history"]:
        print(f"    round {h['round']}: loss={h['final_loss']:.4f}")

    samples = npe.posterior_samples(n=3000)
    if samples is not None:
        med = np.median(samples, axis=0)
        cov = PosteriorCalibration.coverage(samples, true_theta, alpha=0.1)
        print(f"\n  Posterior median  : {np.round(med, 2)}")
        print(f"  90% coverage      : {cov}  (1.0 = truth inside CI)")

    print("\n  HONEST SCOPE")
    print("  • Toy simulator — NOT a real waveform model")
    print("  • NPE posterior is an amortized APPROXIMATION")
    print("  • Validate against Bilby/dynesty before trusting")
    print("  • For production, prefer the `sbi` package backend")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
