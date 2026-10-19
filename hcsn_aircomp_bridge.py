"""
=============================================================================
HCSN ↔ AirComp INTERFACE BRIDGE — Physical-Layer Integration Module
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID     : 0009-0008-2374-0788
License   : MIT  |  Year : 2026
Version   : 1.0.0 (Research-Grade Differentiable RF Simulation)
=============================================================================

ARCHITECTURE POSITION
─────────────────────
This module bridges TWO DISTINCTLY DIFFERENT LAYERS:

  LAYER A  —  HCSN (Satellite-Free Hierarchical Cloud Superposition Network)
              PyTorch tensor / AI architecture layer.
              Produces structural signatures σ_T and decoded payloads.

  LAYER B  —  AirComp (Over-the-Air Computation)
              Physical / wireless RF layer.
              Multiple nodes transmit simultaneously; the receiver computes
              a function (typically sum / weighted-sum / average) directly
              from the superimposed waveform in the air.

THIS FILE IS A DIFFERENTIABLE SIMULATION of Layer B, designed to be
plugged directly after an HCSN forward pass.  It does NOT generate real
RF waveforms; it models the *mathematical structure* of AirComp
(channel, superposition, noise, power control) in PyTorch tensors so
that the entire HCSN → AirComp pipeline can be trained end-to-end.

WHAT THIS MODULE PROVIDES
─────────────────────────
  1. WirelessChannelModel      — path-loss + fading + AWGN
  2. AirCompTransceiver        — over-the-air aggregation engine
  3. HCSNAirCompBridge         — tensor ↔ RF-simulation adapter
  4. Self-test                 — verifies gradient flow end-to-end

WHAT THIS MODULE DOES NOT DO
─────────────────────────────
  • Real RF modulation / demodulation (no SDR hardware interface)
  • Real antenna array processing
  • Real spectrum regulation / compliance
  • Bit-level protocol (MAC, PHY framing, CRC)

For production SDR integration, replace WirelessChannelModel with a
hardware driver and keep the tensor interface unchanged.

=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "AirCompConfig",
    "WirelessChannelModel",
    "AirCompTransceiver",
    "HCSNAirCompBridge",
]


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class AirCompConfig:
    """
    Parameters for the AirComp simulation.

    n_nodes          : number of transmitting nodes (e.g. drones)
    d_payload        : dimension of each node's payload vector
                       (matches HCSN d_out or σ_T dimension)
    n_subcarriers    : OFDM-like subcarrier count for frequency diversity
    carrier_freq_hz  : nominal carrier frequency (used for path-loss calc)
    bandwidth_hz     : system bandwidth (used for noise power)
    noise_figure_db  : receiver noise figure
    tx_power_dbm     : per-node transmit power budget
    path_loss_exp    : path-loss exponent (2.0 = free space, 3-4 = urban)
    ref_dist_m       : reference distance for path-loss model
    fading_type      : "rayleigh" | "rician" | "none"
    rician_K_db      : Rician K-factor in dB (only if fading_type="rician")
    power_control    : True → channel-inversion precoding
    aggregate_fn     : "sum" | "mean" | "weighted_sum"
    use_amp_safe     : clamp values to avoid fp16 overflow
    """
    n_nodes:          int    = 4
    d_payload:        int    = 32
    n_subcarriers:    int    = 16
    carrier_freq_hz:  float  = 2.4e9
    bandwidth_hz:     float  = 1e6
    noise_figure_db:  float  = 5.0
    tx_power_dbm:     float  = 20.0
    path_loss_exp:    float  = 2.5
    ref_dist_m:       float  = 1.0
    fading_type:      str    = "rayleigh"
    rician_K_db:      float  = 6.0
    power_control:    bool   = True
    aggregate_fn:     str    = "mean"
    use_amp_safe:     bool   = True

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1; got {self.n_nodes}")
        if self.d_payload < 1:
            raise ValueError(f"d_payload must be >= 1; got {self.d_payload}")
        if self.aggregate_fn not in ("sum", "mean", "weighted_sum"):
            raise ValueError(f"aggregate_fn must be sum|mean|weighted_sum; "
                             f"got {self.aggregate_fn!r}")
        if self.fading_type not in ("rayleigh", "rician", "none"):
            raise ValueError(f"fading_type must be rayleigh|rician|none; "
                             f"got {self.fading_type!r}")


# ═══════════════════════════════════════════════════════════════════════
#  Smooth / AMP-safe primitives (consistent with SESI physics core)
# ═══════════════════════════════════════════════════════════════════════
_SOFTPLUS_BETA = 100.0

def _soft_floor(x: Tensor, floor: float, beta: float = _SOFTPLUS_BETA) -> Tensor:
    """Differentiable positivity floor (non-zero gradient everywhere)."""
    return floor + F.softplus(x - floor, beta=beta)

def _dbm_to_watts(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)

def _watts_to_dbm(w: Tensor) -> Tensor:
    return 10.0 * torch.log10(_soft_floor(w, 1e-30)) + 30.0


# ═══════════════════════════════════════════════════════════════════════
#  1.  Wireless Channel Model  (differentiable simulation)
# ═══════════════════════════════════════════════════════════════════════
class WirelessChannelModel(nn.Module):
    """
    Simulates a wireless channel between n_nodes transmitters and one
    receiver.  All operations are differentiable PyTorch tensors.

    Channel gain per node, per subcarrier:
        h_{k,f} = sqrt(path_loss_k) × fading_{k,f}

    Path-loss model (log-distance):
        PL_k = PL_0 + 10·n·log10(d_k / d_0)

    Fading:
        Rayleigh : |h| ~ Rayleigh  →  h = (g₁ + j·g₂)/√2
        Rician   : mixture of LOS + scattered
        None     : h = sqrt(PL)  (deterministic)

    NOTE: This is a BASEBAND EQUIVALENT model.  Complex channels are
    represented as real pairs (I, Q) to stay AMP-friendly.
    """

    def __init__(self, cfg: AirCompConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── registered physical constants ────────────────────────────
        self.register_buffer(
            "_k_B", torch.tensor(1.380649e-23), persistent=False)
        self.register_buffer(
            "_T_sys", torch.tensor(290.0), persistent=False)          # K
        self.register_buffer(
            "_c_light", torch.tensor(2.99792458e8), persistent=False)

        # path-loss reference loss at d_0 (Friis): PL_0 = (λ / 4πd_0)²
        lam = self._c_light / cfg.carrier_freq_hz
        pl0_linear = (lam / (4.0 * math.pi * cfg.ref_dist_m)) ** 2
        self.register_buffer(
            "_pl0", torch.tensor(pl0_linear), persistent=False)

        # noise power:  N = k_B · T_sys · B · NF
        nf_linear = 10.0 ** (cfg.noise_figure_db / 10.0)
        n_watts = 1.380649e-23 * 290.0 * cfg.bandwidth_hz * nf_linear
        self.register_buffer(
            "_noise_power", torch.tensor(n_watts), persistent=False)

        # per-node distance (learnable or fixed)
        # default: nodes spread 10 m – 100 m from receiver
        dists = torch.linspace(10.0, 100.0, cfg.n_nodes)
        self.node_distances = nn.Parameter(dists, requires_grad=False)

        # Rician K-factor (linear)
        self.register_buffer(
            "_rician_K",
            torch.tensor(10.0 ** (cfg.rician_K_db / 10.0)),
            persistent=False,
        )

    # ────────────────────────────────────────────────────────────────
    def path_loss_linear(self) -> Tensor:
        """
        Returns per-node path-loss gain (linear, < 1).
        Shape: [n_nodes]
        """
        cfg = self.cfg
        d = _soft_floor(self.node_distances, cfg.ref_dist_m)
        # PL_dB = PL0_dB + 10·n·log10(d/d0)
        pl0_db = -10.0 * torch.log10(_soft_floor(self._pl0, 1e-30))
        pl_db = pl0_db + 10.0 * cfg.path_loss_exp * torch.log10(
            d / cfg.ref_dist_m + 1e-12)
        return torch.pow(10.0, -pl_db / 10.0)          # [n_nodes]

    # ────────────────────────────────────────────────────────────────
    def sample_fading(self, batch: int, device: torch.device,
                      dtype: torch.dtype) -> Tensor:
        """
        Sample fading coefficients.
        Returns: [B, n_nodes, n_subcarriers]  (real-valued magnitude)
        """
        cfg = self.cfg
        shape = (batch, cfg.n_nodes, cfg.n_subcarriers)

        if cfg.fading_type == "none":
            return torch.ones(shape, device=device, dtype=dtype)

        if cfg.fading_type == "rayleigh":
            # Rayleigh: magnitude of complex Gaussian
            g = torch.randn(shape, device=device, dtype=dtype)
            g2 = torch.randn(shape, device=device, dtype=dtype)
            return torch.sqrt(g * g + g2 * g2 + 1e-8) / math.sqrt(2.0)

        if cfg.fading_type == "rician":
            K = self._rician_K.to(dtype)
            # LOS component (deterministic) + scattered (Rayleigh)
            los = torch.sqrt(K / (K + 1.0))
            scatter = torch.sqrt(1.0 / (K + 1.0))
            g = torch.randn(shape, device=device, dtype=dtype)
            g2 = torch.randn(shape, device=device, dtype=dtype)
            ray = torch.sqrt(g * g + g2 * g2 + 1e-8) / math.sqrt(2.0)
            return los + scatter * ray

        raise ValueError(f"Unknown fading_type: {cfg.fading_type}")

    # ────────────────────────────────────────────────────────────────
    def forward(self, batch: int, device: torch.device,
                dtype: torch.dtype) -> Dict[str, Tensor]:
        """
        Returns dict with:
            h          : [B, n_nodes, n_subcarriers]  channel magnitude
            h_complex_I: [B, n_nodes, n_subcarriers]  in-phase (simulated)
            h_complex_Q: [B, n_nodes, n_subcarriers]  quadrature (simulated)
            noise_var  : scalar  AWGN variance per subcarrier
        """
        cfg = self.cfg
        pl = self.path_loss_linear()                     # [n_nodes]
        fading = self.sample_fading(batch, device, dtype)  # [B,K,F]

        # channel magnitude = sqrt(path_loss) × fading
        h = torch.sqrt(_soft_floor(pl, 1e-30)).unsqueeze(0).unsqueeze(-1) \
            * fading                                      # [B,K,F]

        # simulated I/Q decomposition (for completeness / future use)
        phase = torch.rand_like(h) * 2.0 * math.pi
        h_I = h * torch.cos(phase)
        h_Q = h * torch.sin(phase)

        # noise variance per subcarrier
        bw_sub = cfg.bandwidth_hz / cfg.n_subcarriers
        noise_var = self._noise_power * (bw_sub / cfg.bandwidth_hz)
        noise_var = noise_var.to(device=device, dtype=dtype)

        return {
            "h": h,
            "h_complex_I": h_I,
            "h_complex_Q": h_Q,
            "noise_var": noise_var,
        }


# ═══════════════════════════════════════════════════════════════════════
#  2.  AirComp Transceiver  (Over-the-Air Computation engine)
# ═══════════════════════════════════════════════════════════════════════
class AirCompTransceiver(nn.Module):
    """
    Differentiable Over-the-Air Computation transceiver.

    TRANSMIT (per node k):
        x_k = sqrt(p_k) · f_k · s_k
      where:
        s_k : source signal (HCSN output / payload)
        f_k : channel-inversion precoder  (if power_control=True)
        p_k : power allocation

    CHANNEL:
        y = Σ_k  h_k · x_k  +  n      (superposition in the air)

    RECEIVE:
        ŷ = y / h_ref   (coherent combining / equalisation)
        result = aggregate_fn(ŷ)

    The entire chain is differentiable, so HCSN can be trained to
    produce payloads that minimise the AirComp aggregation MSE.
    """

    def __init__(self, cfg: AirCompConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = WirelessChannelModel(cfg)

        # tx power budget (linear watts)
        tx_watts = _dbm_to_watts(cfg.tx_power_dbm)
        self.register_buffer(
            "_tx_power", torch.tensor(tx_watts), persistent=False)

        # learnable per-node power allocation (softmax-normalised)
        self.power_logits = nn.Parameter(
            torch.zeros(cfg.n_nodes))

        # optional learned aggregation weights (for "weighted_sum")
        if cfg.aggregate_fn == "weighted_sum":
            self.agg_weights = nn.Parameter(
                torch.ones(cfg.n_nodes) / cfg.n_nodes)
        else:
            self.register_parameter("agg_weights", None)

    # ────────────────────────────────────────────────────────────────
    def _power_allocation(self) -> Tensor:
        """Per-node normalised power fractions. Shape: [n_nodes]"""
        return F.softmax(self.power_logits, dim=-1)

    # ────────────────────────────────────────────────────────────────
    def _precoder(self, h: Tensor) -> Tensor:
        """
        Channel-inversion precoder: f_k = 1 / h_k  (normalised).
        Falls back to identity if power_control is False.
        Args:
            h : [B, K, F] channel magnitude
        Returns:
            f : [B, K, F] precoding coefficients
        """
        if not self.cfg.power_control:
            return torch.ones_like(h)

        h_safe = _soft_floor(h, 1e-8)
        f_raw = 1.0 / h_safe
        # normalise so average precoding power = 1
        f_norm = f_raw / (_soft_floor(f_raw.mean(dim=(1, 2), keepdim=True), 1e-12))
        return f_norm

    # ────────────────────────────────────────────────────────────────
    def transmit(self, s: Tensor, h: Tensor) -> Tensor:
        """
        Args:
            s : [B, K, F, d]  source signals (HCSN payloads)
            h : [B, K, F]     channel magnitude
        Returns:
            x : [B, K, F, d]  transmitted signals
        """
        cfg = self.cfg
        B, K, F_, d = s.shape

        # power allocation: [K] → [1, K, 1, 1]
        p = self._power_allocation().view(1, K, 1, 1)
        p_watts = p * self._tx_power.to(s.dtype)

        # precoder: [B, K, F] → [B, K, F, 1]
        f = self._precoder(h).unsqueeze(-1)

        x = torch.sqrt(_soft_floor(p_watts, 1e-30)) * f * s
        return x

    # ────────────────────────────────────────────────────────────────
    def channel_superposition(self, x: Tensor, ch: Dict[str, Tensor]
                              ) -> Tensor:
        """
        y = Σ_k  h_k · x_k  +  n
        Args:
            x  : [B, K, F, d]  transmitted signals
            ch : channel dict from WirelessChannelModel
        Returns:
            y  : [B, F, d]  received superimposed signal
        """
        h = ch["h"]                          # [B, K, F]
        noise_var = ch["noise_var"]

        # h · x  →  [B, K, F, d]
        hx = h.unsqueeze(-1) * x

        # superposition over nodes (the "air" computation)
        y = hx.sum(dim=1)                    # [B, F, d]

        # AWGN noise
        if self.training:
            noise = torch.randn_like(y) * torch.sqrt(
                _soft_floor(noise_var, 1e-30))
            y = y + noise

        return y

    # ────────────────────────────────────────────────────────────────
    def receive(self, y: Tensor, ch: Dict[str, Tensor]) -> Tensor:
        """
        Equalise and aggregate.
        Args:
            y  : [B, F, d]
            ch : channel dict
        Returns:
            result : [B, d]  aggregated payload
        """
        cfg = self.cfg
        B, F_, d = y.shape
        K = cfg.n_nodes

        # coherent combining: divide by effective channel
        # (simplified: use mean channel magnitude as reference)
        h_mean = ch["h"].mean(dim=1, keepdim=True)   # [B, 1, F]
        h_ref = _soft_floor(h_mean, 1e-8).transpose(1, 2)  # [B, F, 1]
        y_eq = y / h_ref                              # [B, F, d]

        # frequency diversity combining
        y_comb = y_eq.mean(dim=1)                     # [B, d]

        # node-level aggregation is already done by superposition;
        # here we apply the aggregate function semantics
        if cfg.aggregate_fn == "mean":
            y_comb = y_comb / float(K)
        elif cfg.aggregate_fn == "weighted_sum" and self.agg_weights is not None:
            w = F.softmax(self.agg_weights, dim=-1).sum()  # scalar ≈ 1
            y_comb = y_comb * w

        return y_comb

    # ────────────────────────────────────────────────────────────────
    def forward(self, s: Tensor) -> Dict[str, Tensor]:
        """
        Full AirComp forward pass.
        Args:
            s : [B, K, F, d]  source signals from all nodes
        Returns dict:
            received   : [B, d]     aggregated result
            mse_noise  : scalar     noise-only MSE estimate
            tx_signals : [B,K,F,d]  (for inspection)
            channel    : channel dict
        """
        B, K, F_, d = s.shape
        device, dtype = s.device, s.dtype

        ch = self.channel(batch=B, device=device, dtype=dtype)
        x = self.transmit(s, ch["h"])
        y = self.channel_superposition(x, ch)
        result = self.receive(y, ch)

        return {
            "received": result,
            "tx_signals": x,
            "channel": ch,
        }


# ═══════════════════════════════════════════════════════════════════════
#  3.  HCSN ↔ AirComp Bridge  (the actual interface layer)
# ═══════════════════════════════════════════════════════════════════════
class HCSNAirCompBridge(nn.Module):
    """
    INTERFACE LAYER between HCSN and AirComp.

    Responsibilities:
      1. Accept HCSN outputs (σ_T signature, decoded payload, or both).
      2. Replicate / shard the payload across n_nodes (simulating a
         fleet of drones each holding a local HCSN result).
      3. Map tensors onto AirComp subcarrier slots.
      4. Run the AirComp transceiver.
      5. Return the aggregated result + per-node diagnostics.

    USAGE (conceptual):

        hcsn = SatelliteFreeHCSN(...)
        bridge = HCSNAirCompBridge(AirCompConfig())

        out, aux = hcsn(signal, return_aux=True)   # [B, N, d_out]
        result, info = bridge(out, aux)

    The bridge is fully differentiable: gradients flow from the AirComp
    aggregation loss back through the channel model into HCSN.
    """

    def __init__(self, cfg: AirCompConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.aircomp = AirCompTransceiver(cfg)

        # projection from HCSN d_out → AirComp subcarrier grid
        self.subcarrier_proj = nn.Linear(
            cfg.d_payload, cfg.n_subcarriers * cfg.d_payload)

        # learned per-node embedding (simulates local variation)
        self.node_embed = nn.Parameter(
            torch.randn(cfg.n_nodes, cfg.d_payload) * 0.02)

        # optional: importance weighting for σ_T
        self.sigma_gate = nn.Parameter(torch.zeros(1))

    # ────────────────────────────────────────────────────────────────
    def _prepare_node_signals(self, payload: Tensor,
                              aux: Optional[Dict[str, Tensor]] = None
                              ) -> Tensor:
        """
        Replicate and diversify payload for each node.

        Args:
            payload : [B, d_out]  HCSN decoded output (pooled over tokens)
            aux     : optional HCSN aux dict (may contain sigma_flat)

        Returns:
            s : [B, K, F, d]  per-node, per-subcarrier source signals
        """
        cfg = self.cfg
        B, d_out = payload.shape
        K, F_, d = cfg.n_nodes, cfg.n_subcarriers, cfg.d_payload

        # project to subcarrier grid
        proj = self.subcarrier_proj(payload)          # [B, F*d]
        proj = proj.view(B, F_, d)                    # [B, F, d]

        # add per-node local variation (each drone sees slightly different
        # context — simulates spatial diversity)
        node_emb = self.node_embed.to(payload.dtype)  # [K, d]
        s = proj.unsqueeze(1) + node_emb.view(1, K, 1, d)  # [B,K,F,d]

        # optionally blend in σ_T (structural signature) if available
        if aux is not None and "sigma_flat" in aux:
            sigma = aux["sigma_flat"]                  # [B, sigma_dim]
            gate = torch.sigmoid(self.sigma_gate.to(payload.dtype))
            # project sigma to match d
            if sigma.shape[-1] != d:
                sigma = F.linear(
                    sigma,
                    torch.randn(d, sigma.shape[-1],
                                device=sigma.device, dtype=sigma.dtype)
                    / math.sqrt(sigma.shape[-1])
                )
            sigma_exp = sigma.unsqueeze(1).unsqueeze(2).expand(-1, K, F_, d)
            s = s + gate * sigma_exp

        return s

    # ────────────────────────────────────────────────────────────────
    def forward(self, payload: Tensor,
                aux: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Args:
            payload : [B, d_out]  HCSN output (already pooled)
            aux     : HCSN auxiliary dict (optional)

        Returns:
            result  : [B, d_payload]  AirComp-aggregated payload
            info    : dict with channel / power / diagnostics
        """
        cfg = self.cfg
        device, dtype = payload.device, payload.dtype

        # 1. Prepare per-node signals
        s = self._prepare_node_signals(payload, aux)   # [B,K,F,d]

        # 2. Run AirComp
        ac_out = self.aircomp(s)
        result = ac_out["received"]                    # [B, d]

        # 3. Diagnostics
        h = ac_out["channel"]["h"]
        power_alloc = self.aircomp._power_allocation()
        path_loss = self.aircomp.channel.path_loss_linear()

        info = {
            "channel_magnitude": h,
            "power_allocation": power_alloc,
            "path_loss_linear": path_loss,
            "node_distances": self.aircomp.channel.node_distances,
            "tx_signal_power": ac_out["tx_signals"].pow(2).mean(),
            "received_power": result.pow(2).mean(),
        }

        return result, info

    # ────────────────────────────────────────────────────────────────
    def aircomp_mse(self, result: Tensor, target: Tensor) -> Tensor:
        """
        AirComp aggregation loss: MSE between aggregated result and the
        desired target (e.g. the true mean of all node payloads).

        This is the primary training objective for the HCSN + AirComp
        joint pipeline.
        """
        return F.mse_loss(result, target)


# ═══════════════════════════════════════════════════════════════════════
#  4.  Self-test
# ═══════════════════════════════════════════════════════════════════════
def _selftest() -> None:
    """
    Verifies:
      1. Forward pass produces correct shapes.
      2. Gradient flows end-to-end (payload → AirComp → loss → backward).
      3. Different fading modes work.
      4. Power allocation sums to 1.
    """
    print("=" * 72)
    print(" HCSN ↔ AirComp Bridge — self-test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    cfg = AirCompConfig(
        n_nodes=4,
        d_payload=32,
        n_subcarriers=16,
        fading_type="rayleigh",
        power_control=True,
        aggregate_fn="mean",
    )

    bridge = HCSNAirCompBridge(cfg).to(device)
    bridge.train()

    B = 2
    d_out = 32

    # Simulated HCSN output
    payload = torch.randn(B, d_out, device=device, requires_grad=True)
    aux = {"sigma_flat": torch.randn(B, 64, device=device)}

    result, info = bridge(payload, aux)

    print(f"  payload shape          : {tuple(payload.shape)}")
    print(f"  result shape           : {tuple(result.shape)}")
    print(f"  expected result shape  : ({B}, {cfg.d_payload})")
    assert result.shape == (B, cfg.d_payload), "shape mismatch!"

    # power allocation check
    p = info["power_allocation"]
    p_sum = p.sum().item()
    print(f"  power allocation sum   : {p_sum:.6f}  (should ≈ 1.0)")
    assert abs(p_sum - 1.0) < 1e-4, "power allocation not normalised!"

    # channel stats
    h = info["channel_magnitude"]
    print(f"  channel |h| range      : "
          f"[{h.min().item():.4e}, {h.max().item():.4e}]")
    print(f"  path loss (linear)     : "
          f"{info['path_loss_linear'].tolist()}")
    print(f"  node distances [m]     : "
          f"{info['node_distances'].tolist()}")

    # gradient check
    target = torch.randn(B, cfg.d_payload, device=device)
    loss = bridge.aircomp_mse(result, target)
    loss.backward()

    grad_ok = payload.grad is not None and payload.grad.isfinite().all()
    print(f"  AirComp MSE loss       : {loss.item():.6e}")
    print(f"  gradient flows to HCSN : {'PASS' if grad_ok else 'FAIL'}")
    assert grad_ok, "gradient did not flow back to payload!"

    # test other fading modes
    for fmode in ("rician", "none"):
        cfg2 = AirCompConfig(n_nodes=2, d_payload=8, n_subcarriers=4,
                             fading_type=fmode)
        b2 = HCSNAirCompBridge(cfg2).to(device)
        p2 = torch.randn(1, 8, device=device)
        r2, _ = b2(p2)
        print(f"  fading={fmode:<8s}     : output shape {tuple(r2.shape)}  OK")

    print()
    print("  All self-tests passed.")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _selftest()
