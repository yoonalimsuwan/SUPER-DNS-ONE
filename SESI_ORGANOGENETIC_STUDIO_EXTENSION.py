"""
=============================================================================
SESI ORGANOGENETIC STUDIO EXTENSION
Multi-Regime Coarsening Tau + LZ4 Crate Tokens + Field Merkle Anti-Entropy
+ In-Silico Organoid Ontogenetic World Model
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
License   : MIT
Year      : 2026

SCOPE
-----
COMPUTATIONAL / IN-SILICO ONLY.
No wet-lab protocols.
No cell culture instructions.
No gene editing instructions.
No clinical treatment claims.
No consciousness / sentience claims.

Carpet-19 guards:
  • rho≈1.25 is numerical, not proved.
  • Hino: Gamma(f,.) singular wrt mu; no L2 energy density assumption.
  • D(A) not an algebra; no second-order structure Hessian.
  • [A,M_f] bounded W1->W-1, loses two orders.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import lz4.frame
    HAS_LZ4 = True
except Exception:
    HAS_LZ4 = False


# =============================================================================
# 0. Shared guards / utilities
# =============================================================================

CARPET_DF = math.log(8.0) / math.log(3.0)
CARPET_RHO_NUMERICAL = 1.25
CARPET_DW_NUM = math.log(8.0 * CARPET_RHO_NUMERICAL) / math.log(3.0)
CARPET_DS_NUM = 2.0 * CARPET_DF / CARPET_DW_NUM


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tensor_bytes(x: Tensor) -> bytes:
    arr = x.detach().cpu().contiguous().numpy()
    return arr.tobytes()


def _no_second_order_guard():
    raise RuntimeError(
        "Second-order structure Hessian blocked: Carpet-19 Corollary 3 "
        "states D(A) is not an algebra; f^2 need not belong to D(A)."
    )


# =============================================================================
# 1. Multi-regime coarsening τ learner
# =============================================================================

class CoarseningRegime(str, Enum):
    EARLY = "early"
    CROSSOVER = "crossover"
    LATE = "late"
    UNKNOWN = "unknown"


@dataclass
class CoarseningSegment:
    regime: CoarseningRegime
    i0: int
    i1: int
    exponent: float
    intercept: float
    r2: float
    tau_scale: float


@dataclass
class MultiRegimeTauReport:
    times: List[float]
    lengths: List[float]
    widths: List[float]
    segments: List[CoarseningSegment]
    tau_targets: List[float]
    regime_labels: List[str]
    accepted: bool
    notes: List[str]


@dataclass
class MultiRegimeTauConfig:
    tau_base: float = 0.05
    tau_min: float = 0.005
    tau_max: float = 0.5
    late_expected_alpha: float = 1.0 / 3.0
    late_tolerance: float = 0.20
    early_prior_alpha: float = 0.50
    crossover_prior_alpha: float = 0.33
    min_points_per_segment: int = 3
    width_factor: float = 1.0


class MultiRegimeCoarseningTauLearner(nn.Module):
    """
    Learn tau across early / crossover / late coarsening regimes.

    Input:
      t_i, L(t_i), W_eff(t_i)

    Fit:
      log L = alpha_r log t + b_r
      for 3 contiguous segments.

    Tau target:
      tau_i = clamp(width_factor * W_eff_i / grid_scale, tau_min, tau_max)

    Notes:
      • Late stage expects alpha≈1/3 for conserved CH coarsening.
      • Early/crossover exponents are empirical.
      • First-order adaptation only.
    """

    def __init__(self, cfg: Optional[MultiRegimeTauConfig] = None):
        super().__init__()
        self.cfg = cfg or MultiRegimeTauConfig()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def fit_power_law(self, t: np.ndarray, L: np.ndarray) -> Tuple[float, float, float]:
        ok = (t > 0) & (L > 0)
        if ok.sum() < 2:
            return float("nan"), float("nan"), 0.0
        x = np.log(t[ok])
        y = np.log(L[ok])
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return float(a), float(b), r2

    def segment_fit(self, times: List[float], lengths: List[float],
                    widths: List[float]) -> MultiRegimeTauReport:
        t = np.asarray(times, dtype=float)
        L = np.asarray(lengths, dtype=float)
        W = np.asarray(widths, dtype=float)
        n = len(t)

        notes = []
        if n < 3 * self.cfg.min_points_per_segment:
            notes.append("insufficient points for 3 robust segments; using heuristic split")

        # Search two breakpoints for contiguous 3-segment power-law fit
        best = None
        min_pts = self.cfg.min_points_per_segment
        if n >= 3 * min_pts:
            candidates = range(min_pts, n - min_pts)
            for i in candidates:
                for j in range(i + min_pts, n - min_pts + 1):
                    segs = [(0, i), (i, j), (j, n)]
                    score = 0.0
                    fits = []
                    for (a, b) in segs:
                        alpha, intercept, r2 = self.fit_power_law(t[a:b], L[a:b])
                        score += (1.0 - r2)
                        fits.append((alpha, intercept, r2))
                    if best is None or score < best[0]:
                        best = (score, segs, fits)
        else:
            i = max(1, n // 3)
            j = max(i + 1, 2 * n // 3)
            segs = [(0, i), (i, j), (j, n)]
            fits = [self.fit_power_law(t[a:b], L[a:b]) for a, b in segs]
            best = (0.0, segs, fits)

        _, seg_indices, fits = best
        regimes = [CoarseningRegime.EARLY, CoarseningRegime.CROSSOVER, CoarseningRegime.LATE]
        segments = []
        labels = ["unknown"] * n

        for reg, (a, b), (alpha, intercept, r2) in zip(regimes, seg_indices, fits):
            tau_scale = float(np.median(self._tau_from_width(W[a:b], scale=np.median(L[a:b]) + 1.0)))
            segments.append(CoarseningSegment(
                regime=reg, i0=a, i1=b, exponent=alpha,
                intercept=intercept, r2=r2, tau_scale=tau_scale
            ))
            for k in range(a, b):
                labels[k] = reg.value

        tau_targets = self._tau_from_width(W, scale=max(float(np.median(L)), 1.0)).tolist()

        late = segments[-1]
        late_ok = abs(late.exponent - self.cfg.late_expected_alpha) <= self.cfg.late_tolerance
        if not late_ok:
            notes.append(
                f"late exponent {late.exponent:.3f} differs from 1/3; late tau provisional"
            )

        accepted = late_ok and all(s.r2 > 0.70 for s in segments if math.isfinite(s.r2))

        return MultiRegimeTauReport(
            times=times,
            lengths=lengths,
            widths=widths,
            segments=segments,
            tau_targets=tau_targets,
            regime_labels=labels,
            accepted=accepted,
            notes=notes,
        )

    def _tau_from_width(self, W: np.ndarray, scale: float) -> np.ndarray:
        tau = self.cfg.width_factor * W / max(scale, 1e-6)
        return np.clip(tau, self.cfg.tau_min, self.cfg.tau_max)

    def features(self, t: float, L: float, W: float,
                 alpha: float, regime: CoarseningRegime) -> Tensor:
        regime_id = {
            CoarseningRegime.EARLY: 0.0,
            CoarseningRegime.CROSSOVER: 0.5,
            CoarseningRegime.LATE: 1.0,
            CoarseningRegime.UNKNOWN: -1.0,
        }[regime]
        return torch.tensor([
            math.log(max(t, 1e-9)),
            math.log(max(L, 1e-9)),
            math.log(max(W, 1e-9)),
            alpha,
            regime_id,
        ], dtype=torch.float32)

    def fit_predictor(self, report: MultiRegimeTauReport, epochs: int = 100):
        X, y = [], []
        for i, tau in enumerate(report.tau_targets):
            reg = CoarseningRegime(report.regime_labels[i])
            seg = next((s for s in report.segments if s.regime == reg), report.segments[-1])
            X.append(self.features(report.times[i], report.lengths[i],
                                   report.widths[i], seg.exponent, reg))
            y.append(math.log(max(tau, 1e-9) / self.cfg.tau_base))
        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        losses = []
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.net(X).squeeze(-1)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {"final_loss": losses[-1], "loss_history": losses}

    def predict_tau(self, t: float, L: float, W: float,
                    alpha: float, regime: CoarseningRegime) -> float:
        x = self.features(t, L, W, alpha, regime).unsqueeze(0)
        with torch.no_grad():
            log_tau = self.net(x).squeeze()
            tau = self.cfg.tau_base * torch.exp(log_tau)
            tau = torch.clamp(tau, self.cfg.tau_min, self.cfg.tau_max)
        return float(tau)

    def hessian(self, *args, **kwargs):
        _no_second_order_guard()


# =============================================================================
# 2. LZ4 token compression for crate payloads
# =============================================================================

@dataclass
class CrateTokenBlock:
    tokens: List[str]
    compressed: bool
    codec: str
    raw_bytes: int
    encoded_bytes: int
    payload: bytes


class LZ4CrateTokenCompressor:
    """
    Token compression layer for our minimal crate writer.

    If lz4 is installed:
      codec = lz4-frame
    else:
      codec = raw

    This is a token-section compressor, not a full USD crate compressor.
    Full .usdc token compression must be validated against pxr/usdcat.
    """

    MAGIC = b"SESILZ4T"

    def encode(self, tokens: List[str]) -> CrateTokenBlock:
        raw = json.dumps(tokens, ensure_ascii=False).encode("utf-8")
        if HAS_LZ4:
            comp = lz4.frame.compress(raw)
            payload = self.MAGIC + b"\x01" + comp
            codec = "lz4-frame"
            compressed = True
        else:
            payload = self.MAGIC + b"\x00" + raw
            codec = "raw"
            compressed = False

        return CrateTokenBlock(
            tokens=tokens,
            compressed=compressed,
            codec=codec,
            raw_bytes=len(raw),
            encoded_bytes=len(payload),
            payload=payload,
        )

    def decode(self, payload: bytes) -> List[str]:
        if not payload.startswith(self.MAGIC):
            raise ValueError("invalid SESI token block")
        flag = payload[len(self.MAGIC)]
        body = payload[len(self.MAGIC) + 1:]
        if flag == 1:
            if not HAS_LZ4:
                raise RuntimeError("payload is LZ4-compressed but lz4 is unavailable")
            raw = lz4.frame.decompress(body)
        else:
            raw = body
        return json.loads(raw.decode("utf-8"))

    def ratio(self, block: CrateTokenBlock) -> float:
        return block.raw_bytes / max(block.encoded_bytes, 1)


# =============================================================================
# 3. Field-level Merkle anti-entropy
# =============================================================================

@dataclass
class FieldTileHash:
    tile_id: str
    start: Tuple[int, ...]
    shape: Tuple[int, ...]
    hash: str


class FieldMerkleTree:
    """
    Merkle tree over FIELD TILES, not graph nodes.

    Useful for CH/PFC/organoid fields where a small spatial patch changes
    but the node graph may stay unchanged.
    """

    def __init__(self, tile_shape: Tuple[int, ...] = (8, 8, 8)):
        self.tile_shape = tile_shape

    def tiles(self, field: Tensor):
        shape = field.shape
        assert len(shape) == len(self.tile_shape)
        ranges = []
        for d, ts in zip(shape, self.tile_shape):
            ranges.append(range(0, d, ts))
        for starts in np.ndindex(*[len(r) for r in ranges]):
            start = tuple(ranges[i][s] for i, s in enumerate(starts))
            sl = tuple(slice(st, min(st + ts, shape[i]))
                       for i, (st, ts) in enumerate(zip(start, self.tile_shape)))
            tile = field[sl]
            yield start, tuple(tile.shape), tile

    def leaf_hashes(self, field: Tensor) -> List[FieldTileHash]:
        leaves = []
        for start, shape, tile in self.tiles(field):
            h = _hash_bytes(_tensor_bytes(tile))
            leaves.append(FieldTileHash(
                tile_id=f"tile_{start}",
                start=start,
                shape=shape,
                hash=h,
            ))
        return leaves

    def root(self, field: Tensor) -> str:
        leaves = self.leaf_hashes(field)
        return self.root_from_leaves(leaves)

    def root_from_leaves(self, leaves: List[FieldTileHash]) -> str:
        hs = [x.hash for x in leaves]
        if not hs:
            return _hash_bytes(b"empty")
        while len(hs) > 1:
            nxt = []
            for i in range(0, len(hs), 2):
                chunk = hs[i:i+2]
                nxt.append(_hash_bytes("".join(chunk).encode()))
            hs = nxt
        return hs[0]

    def diff_tiles(self, a: Tensor, b: Tensor) -> List[FieldTileHash]:
        ha = {x.tile_id: x for x in self.leaf_hashes(a)}
        hb = {x.tile_id: x for x in self.leaf_hashes(b)}
        ids = sorted(set(ha) | set(hb))
        diff = []
        for tid in ids:
            if tid not in ha or tid not in hb or ha[tid].hash != hb[tid].hash:
                diff.append(ha.get(tid) or hb[tid])
        return diff


class FieldAntiEntropySync:
    """
    Field-level anti-entropy.

    Exchange only changed spatial tiles when roots differ.
    This is independent of CRDT node-level sync.
    """

    def __init__(self, tile_shape=(8, 8, 8), trigger_interval_s: float = 10.0):
        self.tree = FieldMerkleTree(tile_shape)
        self.trigger_interval_s = trigger_interval_s
        self.last_sync = 0.0

    def should_sync(self) -> bool:
        return time.time() - self.last_sync >= self.trigger_interval_s

    def sync(self, source: Tensor, target: Tensor) -> Dict[str, Any]:
        """
        Copy changed tiles from source into target.
        Returns bandwidth diagnostics.
        """
        root_s = self.tree.root(source)
        root_t = self.tree.root(target)
        if root_s == root_t:
            self.last_sync = time.time()
            return {
                "in_sync": True,
                "tiles_changed": 0,
                "bytes_transferred": 0,
                "root": root_s,
            }

        diffs = self.tree.diff_tiles(source, target)
        bytes_transferred = 0
        out = target.clone()

        for leaf in diffs:
            sl = tuple(slice(st, st + sh)
                       for st, sh in zip(leaf.start, leaf.shape))
            tile = source[sl]
            out[sl] = tile
            bytes_transferred += tile.numel() * tile.element_size()

        self.last_sync = time.time()
        return {
            "in_sync": False,
            "tiles_changed": len(diffs),
            "bytes_transferred": bytes_transferred,
            "full_field_bytes": target.numel() * target.element_size(),
            "bandwidth_reduction_x": (target.numel() * target.element_size()) / max(bytes_transferred, 1),
            "new_root": self.tree.root(out),
            "target_updated": out,
        }


# =============================================================================
# 4. In-silico organoid ontogenetic world model
# =============================================================================

class OrganoidStage(str, Enum):
    AGGREGATION = "aggregation"
    POLARIZATION = "polarization"
    PATTERNING = "patterning"
    DIFFERENTIATION = "differentiation"
    MATURATION = "maturation"


@dataclass
class OrganoidOntogenyConfig:
    grid: int = 24
    dt: float = 0.02
    n_steps: int = 32
    morphogen_diffusion: float = 0.08
    morphogen_decay: float = 0.01
    proliferation_gain: float = 1.0
    differentiation_threshold: float = 0.55
    phase_epsilon: float = 1.2
    seed: int = 0
    device: str = "cpu"


@dataclass
class OrganoidState:
    """
    Abstract computational state only.

    Fields:
      M: morphogen-like scalar field
      P: proliferation propensity
      D: differentiation state
      U: phase-field / tissue boundary proxy

    No wet-lab meaning is assigned to parameters.
    """
    M: Tensor
    P: Tensor
    D: Tensor
    U: Tensor
    stage: OrganoidStage
    step: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrganoidOntogeneticWorldModel(nn.Module):
    """
    In-silico ontogenetic morphogenesis model.

    This is an abstract differentiable field model:
      M_t = Dm Lap M - decay M + source
      P   = sigmoid(aM - bD + c)
      D_t = D + eta * sigmoid(M - threshold)
      U_t = Allen-Cahn / CH-like relaxation proxy

    It is NOT a biological protocol.
    It does not prescribe culture conditions, media, growth factors,
    gene edits, timing, or experimental procedures.

    It can be trained on abstract or properly approved datasets, but
    this file provides only a safe computational scaffold.
    """

    def __init__(self, cfg: Optional[OrganoidOntogenyConfig] = None):
        super().__init__()
        self.cfg = cfg or OrganoidOntogenyConfig()

        self.stage_embed = nn.Embedding(5, 16)
        self.net = nn.Sequential(
            nn.Linear(4 + 16, 64), nn.GELU(),
            nn.Linear(64, 4),
        )

    def initial_state(self, batch: int = 1) -> OrganoidState:
        g = torch.Generator().manual_seed(self.cfg.seed)
        shape = (batch, 1, self.cfg.grid, self.cfg.grid, self.cfg.grid)

        # smooth random morphogen seed
        M = torch.rand(shape, generator=g)
        M = F.avg_pool3d(M, 3, stride=1, padding=1)

        P = torch.sigmoid(M - 0.5)
        D = torch.zeros_like(M)
        U = torch.tanh((M - 0.5) / 0.1)

        return OrganoidState(
            M=M, P=P, D=D, U=U,
            stage=OrganoidStage.AGGREGATION,
            step=0,
            metadata={"scope": "in-silico only"},
        )

    @staticmethod
    def lap3(x: Tensor) -> Tensor:
        return (
            torch.roll(x, 1, -3) + torch.roll(x, -1, -3)
            + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
            + torch.roll(x, 1, -1) + torch.roll(x, -1, -1)
            - 6.0 * x
        )

    def stage_for_step(self, step: int) -> OrganoidStage:
        frac = step / max(self.cfg.n_steps, 1)
        if frac < 0.20:
            return OrganoidStage.AGGREGATION
        if frac < 0.40:
            return OrganoidStage.POLARIZATION
        if frac < 0.65:
            return OrganoidStage.PATTERNING
        if frac < 0.85:
            return OrganoidStage.DIFFERENTIATION
        return OrganoidStage.MATURATION

    def step_state(self, s: OrganoidState) -> OrganoidState:
        dt = self.cfg.dt
        M, P, D, U = s.M, s.P, s.D, s.U

        # abstract morphogen dynamics
        source = 0.02 * torch.sin((s.step + 1) * 0.1 + M.mean())
        M_new = M + dt * (
            self.cfg.morphogen_diffusion * self.lap3(M)
            - self.cfg.morphogen_decay * M
            + source
        )
        M_new = torch.clamp(M_new, 0.0, 1.0)

        # abstract proliferation / differentiation
        P_new = torch.sigmoid(self.cfg.proliferation_gain * (M_new - D))
        D_new = torch.clamp(
            D + dt * 0.1 * torch.sigmoid(M_new - self.cfg.differentiation_threshold),
            0.0, 1.0,
        )

        # phase-field proxy, first-order only
        bulk = U - U ** 3
        U_new = U + dt * (self.cfg.phase_epsilon ** 2 * self.lap3(U) + bulk)
        U_new = torch.tanh(U_new)

        stage = self.stage_for_step(s.step + 1)

        return OrganoidState(
            M=M_new, P=P_new, D=D_new, U=U_new,
            stage=stage, step=s.step + 1,
            metadata={
                "scope": "in-silico only",
                "no_wetlab_protocol": True,
                "no_clinical_claim": True,
            },
        )

    def rollout(self, batch: int = 1) -> List[OrganoidState]:
        s = self.initial_state(batch)
        states = [s]
        for _ in range(self.cfg.n_steps):
            s = self.step_state(s)
            states.append(s)
        return states

    def summarize_state(self, s: OrganoidState) -> Dict[str, Any]:
        return {
            "step": s.step,
            "stage": s.stage.value,
            "M_mean": float(s.M.mean()),
            "P_mean": float(s.P.mean()),
            "D_mean": float(s.D.mean()),
            "U_mean": float(s.U.mean()),
            "scope": "in-silico only",
        }

    def hessian(self, *args, **kwargs):
        _no_second_order_guard()


# =============================================================================
# 5. Orchestrator
# =============================================================================

class SESIOntogeneticStudioExtension:
    """
    Unified extension:
      • multi-regime quench tau
      • LZ4 token compression
      • field-level anti-entropy
      • in-silico organoid ontogenetic world model
    """

    def __init__(self):
        self.coarsen = MultiRegimeCoarseningTauLearner()
        self.tokens = LZ4CrateTokenCompressor()
        self.field_ae = FieldAntiEntropySync(tile_shape=(8, 8, 8))
        self.organoid = OrganoidOntogeneticWorldModel(
            OrganoidOntogenyConfig(grid=16, n_steps=12)
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "lz4_available": HAS_LZ4,
            "scope": "in-silico organoid ontogenetic modeling only",
            "carpet_guards": [
                "rho numerical not proved",
                "Hino: no L2 energy density",
                "D(A) not algebra: no second-order structure Hessian",
                "commutator W1->W-1 loses two orders",
            ],
        }


# =============================================================================
# 6. Smoke test
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI ORGANOGENETIC STUDIO EXTENSION — SMOKE TEST")
    print("=" * 78)
    print(f"LZ4 available: {HAS_LZ4}")

    suite = SESIOntogeneticStudioExtension()

    # 1. multi-regime coarsening
    print("\n[1] Multi-regime coarsening tau")
    t = np.logspace(-2, 1.2, 18)
    # synthetic early/crossover/late lengths
    L = np.piecewise(
        t,
        [t < 0.25, (t >= 0.25) & (t < 1.5), t >= 1.5],
        [
            lambda x: 0.5 * x ** 0.55,
            lambda x: 0.7 * x ** 0.38,
            lambda x: 0.85 * x ** (1 / 3),
        ],
    )
    W = 3.0 * np.exp(-0.35 * t) + 1.2
    report = suite.coarsen.segment_fit(t.tolist(), L.tolist(), W.tolist())
    for seg in report.segments:
        print(f"  {seg.regime.value:10s}: alpha={seg.exponent:.3f}, R2={seg.r2:.3f}, tau_scale={seg.tau_scale:.4f}")
    print("  accepted:", report.accepted, "notes:", report.notes)

    fit = suite.coarsen.fit_predictor(report, epochs=30)
    print("  predictor loss:", fit["final_loss"])
    tau_pred = suite.coarsen.predict_tau(
        t[10], L[10], W[10],
        report.segments[1].exponent,
        CoarseningRegime.CROSSOVER,
    )
    print("  predicted tau crossover:", tau_pred)

    try:
        suite.coarsen.hessian()
    except RuntimeError:
        print("  Hessian guard: BLOCKED")

    # 2. LZ4 token compression
    print("\n[2] Crate token compression")
    tokens = [
        "SESIMesh", "points", "faceVertexCounts", "faceVertexIndices",
        "primvars:displayColor", "subdivisionScheme", "none", "constant",
    ] * 20
    block = suite.tokens.encode(tokens)
    decoded = suite.tokens.decode(block.payload)
    print("  codec:", block.codec)
    print("  raw bytes:", block.raw_bytes, "encoded:", block.encoded_bytes)
    print("  ratio:", suite.tokens.ratio(block))
    print("  roundtrip:", decoded == tokens)

    # 3. field-level anti-entropy
    print("\n[3] Field-level Merkle anti-entropy")
    a = torch.randn(16, 16, 16)
    b = a.clone()
    b[8:16, 8:16, 8:16] += torch.randn(8, 8, 8) * 0.1
    sync = suite.field_ae.sync(a, b)
    print("  changed tiles:", sync["tiles_changed"])
    print("  bytes transferred:", sync["bytes_transferred"])
    print("  reduction x:", sync.get("bandwidth_reduction_x"))
    print("  new root equals source:", sync["new_root"] == suite.field_ae.tree.root(a))

    # 4. organoid ontogenetic in-silico rollout
    print("\n[4] In-silico organoid ontogenetic rollout")
    states = suite.organoid.rollout(batch=1)
    for s in states[::4]:
        print(" ", suite.organoid.summarize_state(s))

    try:
        suite.organoid.hessian()
    except RuntimeError:
        print("  Organoid Hessian guard: BLOCKED")

    print("\n[STATUS]")
    print(json.dumps(suite.get_status(), indent=2))

    print("\n" + "=" * 78)
    print("HONEST SCOPE")
    print("- Organoid module is computational / in-silico only.")
    print("- No wet-lab protocol, no culture conditions, no gene editing.")
    print("- No clinical claim, no sentience/consciousness claim.")
    print("- tau coarsening targets are empirical across regimes.")
    print("- LZ4 block is a token-section layer, not full USD crate compression.")
    print("- Field anti-entropy repairs spatial field divergence, not CRDT semantics.")
    print("- Carpet-19 guards preserved: Hino singularity, no D(A) algebra,")
    print("  no second-order structure Hessian, commutator W1->W-1 only.")
    print("=" * 78)


if __name__ == "__main__":
    smoke_test()
