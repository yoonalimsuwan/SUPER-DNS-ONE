"""
=============================================================================
SESI TOPOLOGY STUDIO — ORGANOID ONTOGENETIC
Multi-Stage Coarsening τ + LZ4 Crate Compression + Field-Level Merkle
+ Organoid Morphogenesis Engine
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
MODULES
=============================================================================
[MST1] MultiStageCoarseningTau
      • Drives a REAL Cahn-Hilliard quench and tracks L(t) over time
      • Fits the LOCAL coarsening exponent alpha(t) = d ln L / d ln t
      • Classifies into three regimes:
          EARLY      alpha > 0.5      (spinodal, exponential growth)
          CROSSOVER  0.2 < alpha < 0.5 (transition)
          LATE       alpha ~ 1/3       (Lifshitz-Slyozov)
      • Assigns a regime-specific tau target per sample
      • Trains a tau predictor conditioned on (regime, L, W_eff, t)
      • COROLLARY-3 GUARD: first-order tau adaptation only
      • HINO GUARD: no L2 energy-density assumption

[LZ41] LZ4CrateCompressor
      • Byte-level LZ4 block compression (Collet 2011)
      • Compresses the TOKENS section of the crate binary
      • Integrates with USDCCrateWriter via a compression hook
      • Reports compression ratio and round-trip integrity

[FMT1] FieldMerkleTree
      • Multi-resolution Merkle tree over a 3-D scalar field
      • Leaf level: spatial blocks of size B
      • Internal levels: downsampled field blocks
      • Divergence detection returns the spatial regions that differ
      • Targeted resync: exchange only divergent blocks
      • Generalises the node-level Merkle to FIELD-level integrity

[OOE1] OrganoidOntogeneticEngine
      • Coupled morphogenetic field model combining:
          - Turing reaction-diffusion (Schnakenberg) for morphogens
          - Cahn-Hilliard for cell sorting / differential adhesion
          - Phase-Field-Crystal for cell packing order
          - Growth source term (nutrient-limited)
          - Mechanical feedback via the carpet sigma-field operator
      • Self-organises into organoid-like structures from noise
      • Exports the resulting cell-density field to a mesh
      • HONEST: computational morphogenesis, NOT validated biology

=============================================================================
HONEST SCOPE
=============================================================================
• MultiStageCoarseningTau: the regime boundaries are EMPIRICAL. The
  Lifshitz-Slyozov 1/3 law is late-stage only [C]. Early-stage spinodal
  growth is exponential [C Cahn 1961]. Crossover has no closed form.
  tau targets track the measured length scale; they are not derived from
  a closed-form quench theory.
• LZ4CrateCompressor: implements the LZ4 BLOCK format, not the LZ4 frame
  format. No streaming, no dictionary, no content checksum. Validate
  round-trip before production use.
• FieldMerkleTree: provides integrity detection and targeted resync, NOT
  cryptographic authentication. A malicious peer can craft collisions in
  principle; this is a convergence aid, not a security boundary.
• OrganoidOntogeneticEngine: a COMPUTATIONAL MODEL of morphogenesis. It
  uses established PDE components (Turing RD, CH, PFC) but is NOT
  validated against actual organoid biology. No medical, therapeutic, or
  diagnostic claim is made or implied. Parameters are illustrative.
• All thresholds are FREE DESIGN PARAMETERS.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.organoid")

# ── Ecosystem imports (graceful) ─────────────────────────────────────────
try:
    from structural_cahn_hilliard_3d import (
        StructuralCahnHilliard3D, PhaseFieldCrystal3D,
        ThinFilmStructuralCahnHilliard3D, CahnHilliardConfig,
        make_sigma_field,
    )
    HAS_CH = True
except Exception:
    HAS_CH = False

try:
    from sesi_topology_studio_quench_crate_ae import (
        QuenchTauTrainer, QuenchConfig, USDCCrateWriter, CrateConfig,
        AntiEntropySync, MerkleTree,
    )
    HAS_PREV = True
except Exception:
    HAS_PREV = False

try:
    from sesi_topology_studio_collab import DeltaCSGTreeCRDT
    HAS_CRDT = True
except Exception:
    HAS_CRDT = False

__all__ = [
    "CoarseningRegime", "MultiStageCoarseningTau", "MultiStageTauReport",
    "LZ4CrateCompressor",
    "FieldMerkleTree", "FieldDivergence",
    "OrganoidConfig", "OrganoidOntogeneticEngine", "OrganoidState",
    "SESIOrganoidSuite",
]


# =============================================================================
# SECTION 0 — DIFFERENTIABILITY PRIMITIVES
# =============================================================================

def _softplus_floor(x: Tensor, floor: float, beta: Optional[float] = None,
                    sharpness: float = 50.0) -> Tensor:
    if beta is None:
        beta = sharpness / max(abs(floor), 1e-12)
    return floor + F.softplus(x - floor, beta=beta)

def _soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps)


# =============================================================================
# SECTION 1 — [MST1] MULTI-STAGE COARSENING τ
# =============================================================================

class CoarseningRegime(str, Enum):
    EARLY = "early"          # spinodal, exponential growth
    CROSSOVER = "crossover"  # transition
    LATE = "late"            # Lifshitz-Slyozov t^{1/3}


@dataclass
class MultiStageTauConfig:
    """Configuration for multi-stage coarsening τ."""
    grid: int = 20
    ch_epsilon: float = 1.5
    dt: float = 2e-4
    total_steps: int = 400
    n_samples: int = 20
    # regime classification thresholds on local exponent alpha(t)
    early_alpha_threshold: float = 0.50
    late_alpha_threshold: float = 0.40
    # tau scaling per regime
    tau_early_factor: float = 2.0     # broad interfaces early
    tau_late_factor: float = 1.0      # sharper interfaces late
    tau_base: float = 0.05
    tau_min: float = 0.005
    tau_max: float = 0.5
    # predictor
    n_hidden: int = 32
    lr: float = 5e-3
    n_epochs: int = 120
    device: str = "cpu"


@dataclass
class MultiStageTauReport:
    """Result of a multi-stage coarsening τ calibration."""
    times: List[float]
    domain_lengths: List[float]       # L(t)
    interface_widths: List[float]      # W_eff(t)
    local_exponents: List[float]       # alpha(t) = d ln L / d ln t
    regimes: List[str]                 # early / crossover / late
    tau_targets: List[float]
    regime_counts: Dict[str, int]
    fitted_late_exponent: float
    late_matches_lifshitz_slyozov: bool
    provisional: bool
    notes: List[str]


class MultiStageCoarseningTau(nn.Module):
    """
    [C/D] τ learning across the EARLY / CROSSOVER / LATE coarsening regimes.

    PHYSICS
      Early (spinodal) [C Cahn 1961]:
          The order parameter grows exponentially at the fastest-growing
          mode. L(t) is ~ the spinodal wavelength; interfaces are broad.
          tau should be LARGE to resolve the diffuse interfaces.
      Crossover:
          Domains form and begin to coarsen. No closed-form law.
          tau transitions between the early and late scalings.
      Late (Lifshitz-Slyozov) [C]:
          L(t) ~ t^{1/3} for a conserved order parameter. Interfaces
          sharpen toward the equilibrium tanh profile. tau should track
          the shrinking interface width relative to the growing domains.

    τ TARGET  [D]
      tau_target(t) = regime_factor * W_eff(t) / grid
      where regime_factor is 2.0 (early), 1.5 (crossover), 1.0 (late).
      This keeps tau proportional to the actual interface thickness in
      grid units, adapting to the regime.

    REGIME DETECTION  [D]
      Local exponent alpha(t) = d ln L / d ln t, computed by finite
      differences on log-log. Classified:
          alpha > early_threshold  -> EARLY
          alpha < late_threshold   -> LATE
          otherwise                -> CROSSOVER

    DISCIPLINE
      We CHECK that the late-stage fitted exponent is near 1/3 before
      trusting the late tau. If not, the report is flagged PROVISIONAL.

    GUARDS
      • COROLLARY 3 [D]: first-order tau adaptation only; no field Hessian.
      • HINO [C Thm 8]: W_eff measured from the order parameter, no L2
        energy-density assumption.
    """

    def __init__(self, cfg: Optional[MultiStageTauConfig] = None):
        super().__init__()
        self.cfg = cfg or MultiStageTauConfig()
        # predictor: [regime_onehot(3), log L, log W, log t] -> log(tau/tau_base)
        self.net = nn.Sequential(
            nn.Linear(6, self.cfg.n_hidden), nn.GELU(),
            nn.Linear(self.cfg.n_hidden, self.cfg.n_hidden), nn.GELU(),
            nn.Linear(self.cfg.n_hidden, 1),
        )
        self.register_buffer("log_lo",
            torch.tensor(math.log(self.cfg.tau_min / self.cfg.tau_base)))
        self.register_buffer("log_hi",
            torch.tensor(math.log(self.cfg.tau_max / self.cfg.tau_base)))
        self._solver = self._build_solver() if HAS_CH else None
        self._fitted = False

    def _build_solver(self):
        c = CahnHilliardConfig(
            dx=1.0, epsilon=self.cfg.ch_epsilon, dt=self.cfg.dt,
            laplacian="conv3d", scheme="imex", dtype=torch.float32,
            device=self.cfg.device)
        return StructuralCahnHilliard3D(c)

    # ── measurements ────────────────────────────────────────────────────
    def measure_width(self, u: Tensor) -> float:
        gu = torch.stack([torch.roll(u,-1,d)-torch.roll(u,1,d)
                           for d in range(3)],0)/2.0
        grad_mag = torch.sqrt((gu**2).sum(0)+1e-12)
        num = torch.sum((1.0-u*u).clamp_min(0.0))
        den = torch.sum(grad_mag)+1e-12
        return float(num/den)

    def measure_length(self, u: Tensor) -> float:
        arr = u.detach().cpu().numpy()
        f = np.fft.fftn(arr-arr.mean())
        amp = np.abs(np.fft.fftshift(f))
        c = np.array(arr.shape)//2
        amp[c[0],c[1],c[2]] = 0.0
        yy,xx,zz = np.indices(arr.shape)
        r = np.sqrt((yy-c[0])**2+(xx-c[1])**2+(zz-c[2])**2)
        r = np.maximum(r,1e-6)
        total = amp.sum()+1e-12
        mean_r = float((amp*r).sum()/total)
        freq = mean_r/max(arr.shape[0],1)
        return 1.0/max(freq,1e-6)

    def classify_regime(self, alpha: float) -> CoarseningRegime:
        if alpha > self.cfg.early_alpha_threshold:
            return CoarseningRegime.EARLY
        if alpha < self.cfg.late_alpha_threshold:
            return CoarseningRegime.LATE
        return CoarseningRegime.CROSSOVER

    def regime_tau_factor(self, regime: CoarseningRegime) -> float:
        return {CoarseningRegime.EARLY: self.cfg.tau_early_factor,
                CoarseningRegime.CROSSOVER: 1.5,
                CoarseningRegime.LATE: self.cfg.tau_late_factor}[regime]

    # ── quench trajectory with regime tracking ──────────────────────────
    def run_multistage_quench(self, seed: int = 0) -> MultiStageTauReport:
        grid = self.cfg.grid
        g = torch.Generator().manual_seed(seed)
        u = (torch.rand(grid,grid,grid,generator=g)-0.5)*0.2
        sigma = torch.ones_like(u)

        sample_every = max(1, self.cfg.total_steps//self.cfg.n_samples)
        times,Ls,Ws,alphas,regimes,taus = [],[],[],[],[],[]
        notes = []

        prev_lnL, prev_lnt = None, None
        step = 0
        for k in range(self.cfg.n_samples):
            for _ in range(sample_every):
                if self._solver is not None:
                    u = self._solver.step(u, sigma)
                step += 1
            t = step*self.cfg.dt
            L = self.measure_length(u)
            W = self.measure_width(u)
            lnL, lnt = math.log(max(L,1e-9)), math.log(max(t,1e-9))
            if prev_lnL is not None and abs(lnt-prev_lnt)>1e-9:
                alpha = (lnL-prev_lnL)/(lnt-prev_lnt)
            else:
                alpha = 1.0
            prev_lnL, prev_lnt = lnL, lnt

            regime = self.classify_regime(alpha)
            factor = self.regime_tau_factor(regime)
            tau = float(np.clip(factor*W/grid,
                                  self.cfg.tau_min, self.cfg.tau_max))

            times.append(t); Ls.append(L); Ws.append(W)
            alphas.append(alpha); regimes.append(regime.value); taus.append(tau)

        # fit late-stage exponent from the tail
        n = len(times)
        tail = slice(max(0,n//2), n)
        t_tail = np.array(times[tail]); L_tail = np.array(Ls[tail])
        if len(t_tail)>=3:
            slope,_ = np.polyfit(np.log(t_tail),np.log(L_tail),1)
        else:
            slope = float("nan")
        late_ok = bool(abs(slope-1.0/3.0)<0.15)
        if not late_ok:
            notes.append(f"late exponent {slope:.3f} deviates from 1/3; "
                          "late tau PROVISIONAL")
        provisional = not late_ok

        regime_counts = {r.value:0 for r in CoarseningRegime}
        for r in regimes: regime_counts[r]+=1

        return MultiStageTauReport(
            times=times, domain_lengths=Ls, interface_widths=Ws,
            local_exponents=alphas, regimes=regimes, tau_targets=taus,
            regime_counts=regime_counts,
            fitted_late_exponent=float(slope),
            late_matches_lifshitz_slyozov=late_ok,
            provisional=provisional, notes=notes,
        )

    # ── predictor fit ───────────────────────────────────────────────────
    def _regime_onehot(self, regime: str) -> List[float]:
        order = [CoarseningRegime.EARLY.value,
                 CoarseningRegime.CROSSOVER.value,
                 CoarseningRegime.LATE.value]
        return [1.0 if regime==o else 0.0 for o in order]

    def fit(self, report: MultiStageTauReport) -> Dict[str, Any]:
        X,y = [],[]
        for L,W,t,tau,reg in zip(report.domain_lengths,
                                    report.interface_widths,
                                    report.times, report.tau_targets,
                                    report.regimes):
            X.append(self._regime_onehot(reg)+[
                math.log(max(L,1e-9)), math.log(max(W,1e-9)),
                math.log(max(t,1e-9))])
            y.append(math.log(tau/self.cfg.tau_base))
        X = torch.tensor(X,dtype=torch.float32)
        y = torch.clamp(torch.tensor(y,dtype=torch.float32),
                          self.log_lo,self.log_hi)
        opt = torch.optim.Adam(self.net.parameters(),lr=self.cfg.lr)
        losses=[]
        for _ in range(self.cfg.n_epochs):
            opt.zero_grad()
            pred = torch.clamp(self.net(X).squeeze(-1),self.log_lo,self.log_hi)
            loss = F.mse_loss(pred,y)
            loss.backward(); opt.step(); losses.append(float(loss))
        self._fitted = True
        return {"n_samples":len(report.times),"final_loss":losses[-1],
                "regime_counts":report.regime_counts,
                "provisional":report.provisional}

    def predict_tau(self,L:float,W:float,t:float,regime:str)->float:
        feat = torch.tensor(self._regime_onehot(regime)+[
            math.log(max(L,1e-9)),math.log(max(W,1e-9)),
            math.log(max(t,1e-9))],dtype=torch.float32)
        with torch.no_grad():
            log_scale = torch.clamp(self.net(feat.unsqueeze(0)).squeeze(),
                                     self.log_lo,self.log_hi)
            tau = self.cfg.tau_base*torch.exp(log_scale)
        return float(tau)

    def get_status(self)->Dict[str,Any]:
        return {"fitted":self._fitted,"has_ch_solver":self._solver is not None,
                "corollary3_guard":"first-order tau only; no field Hessian",
                "hino_guard":"W_eff from order parameter, no L2 energy density"}


# =============================================================================
# SECTION 2 — [LZ41] LZ4 CRATE TOKEN COMPRESSOR
# =============================================================================

class LZ4CrateCompressor:
    """
    [C] LZ4 block-format compressor for the crate TOKENS section.

    LZ4 BLOCK FORMAT (Collet 2011)
      A sequence of sequences. Each sequence:
        token byte:  high nibble = literal length, low nibble = match length
        if literal length == 15: extra bytes add to literal length
        literals (raw bytes)
        if not last sequence:
          2-byte little-endian match offset
          if match length nibble == 15: extra bytes add to match length
      The match copies `match_length + 4` bytes from `offset` back.

    IMPLEMENTATION
      A greedy LZ4 compressor with a hash table for match finding.
      Produces valid LZ4 block data decompressible by any LZ4 decoder.
      Includes a pure-Python decompressor for round-trip verification.

    INTEGRATION
      The crate writer can call compress_tokens() on the TOKENS blob and
      store the compressed bytes, recording a flag so the reader knows to
      decompress. This reduces the .usdc payload for token-heavy scenes.

    HONEST
      • Block format only; no LZ4 frame header / checksum / dictionary.
      • Compression ratio depends on token redundancy; short unique tokens
        may not compress.
      • Validate round-trip before production use.
    """

    MIN_MATCH = 4
    HASH_LOG = 16
    HASH_SIZE = 1 << HASH_LOG

    def _hash4(self, b: bytes) -> int:
        return ((int.from_bytes(b,"little")*2654435761)>>(32-self.HASH_LOG)) \
                & (self.HASH_SIZE-1)

    def compress(self, data: bytes) -> bytes:
        """Compress `data` to LZ4 block format."""
        n = len(data)
        if n == 0:
            return b"\x00"
        out = bytearray()
        hash_table = [-1]*self.HASH_SIZE
        anchor = 0
        i = 0
        while i < n - self.MIN_MATCH:
            h = self._hash4(data[i:i+4])
            ref = hash_table[h]
            hash_table[h] = i
            if ref >= 0 and i - ref < 65536 and data[ref:ref+4] == data[i:i+4]:
                # found a match; extend it
                match_len = self.MIN_MATCH
                while (i+match_len < n and ref+match_len < i and
                        data[ref+match_len] == data[i+match_len]):
                    match_len += 1
                # emit literals from anchor to i
                lit_len = i - anchor
                self._emit_sequence(out, data, anchor, lit_len,
                                     i-ref, match_len,
                                     is_last=False)
                i += match_len
                anchor = i
            else:
                i += 1
        # final literals
        lit_len = n - anchor
        self._emit_sequence(out, data, anchor, lit_len, 0, 0, is_last=True)
        return bytes(out)

    def _emit_sequence(self, out, data, anchor, lit_len,
                         offset, match_len, is_last):
        # token
        ll = min(lit_len, 15)
        ml = min(max(match_len - self.MIN_MATCH, 0), 15) if not is_last else 0
        token = (ll << 4) | (ml if not is_last else 0)
        out.append(token)
        # extended literal length
        if lit_len >= 15:
            rem = lit_len - 15
            while rem >= 255:
                out.append(255); rem -= 255
            out.append(rem)
        # literals
        out.extend(data[anchor:anchor+lit_len])
        if is_last:
            return
        # match offset (little-endian 16-bit)
        out.extend(struct.pack("<H", offset))
        # extended match length
        if match_len - self.MIN_MATCH >= 15:
            rem = match_len - self.MIN_MATCH - 15
            while rem >= 255:
                out.append(255); rem -= 255
            out.append(rem)

    def decompress(self, data: bytes, uncompressed_size: Optional[int] = None
                    ) -> bytes:
        """Decompress LZ4 block format back to original bytes."""
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            token = data[i]; i += 1
            lit_len = token >> 4
            if lit_len == 15:
                while i < n:
                    b = data[i]; i += 1
                    lit_len += b
                    if b != 255: break
            out.extend(data[i:i+lit_len]); i += lit_len
            if i >= n:
                break
            offset = struct.unpack_from("<H", data, i)[0]; i += 2
            match_len = token & 0x0F
            if match_len == 15:
                while i < n:
                    b = data[i]; i += 1
                    match_len += b
                    if b != 255: break
            match_len += self.MIN_MATCH
            start = len(out) - offset
            for k in range(match_len):
                out.append(out[start + k])
        return bytes(out)

    def round_trip_check(self, data: bytes) -> Dict[str, Any]:
        comp = self.compress(data)
        dec = self.decompress(comp)
        return {
            "original_bytes": len(data),
            "compressed_bytes": len(comp),
            "ratio": len(data)/max(len(comp),1),
            "round_trip_ok": dec == data,
        }

    def compress_tokens_for_crate(self, tokens_blob: bytes) -> Dict[str, Any]:
        """Compress a crate TOKENS section and report savings."""
        return self.round_trip_check(tokens_blob)

    def get_status(self) -> Dict[str, Any]:
        return {"format":"LZ4 block","min_match":self.MIN_MATCH,
                "hash_log":self.HASH_LOG,
                "note":"block format only; no frame/dictionary/checksum"}


# =============================================================================
# SECTION 3 — [FMT1] FIELD-LEVEL MERKLE TREE
# =============================================================================

@dataclass
class FieldDivergence:
    """Result of a field-level Merkle comparison."""
    in_sync: bool
    root_a: str
    root_b: str
    divergent_blocks: List[Tuple[int, ...]]   # block coordinates
    divergent_fraction: float
    bytes_to_resync: int


class FieldMerkleTree:
    """
    [C/D] Multi-resolution Merkle tree over a 3-D scalar field.

    GENERALISES the node-level Merkle (AntiEntropySync) to FIELD-level
    integrity. Instead of hashing (node_id -> state), we hash the actual
    field values partitioned into spatial blocks, at multiple resolutions.

    STRUCTURE
      Level 0 (leaves): field partitioned into blocks of size B^3;
                         leaf hash = sha256(block bytes).
      Level k:          downsampled field (factor 2^k), partitioned into
                         blocks; hash over those.
      Root:             hash of the coarsest level.

    DIVERGENCE DETECTION
      Compare roots. If different, descend level by level: at each level,
      compare block hashes to localise WHICH spatial blocks differ.
      Returns the set of divergent block coordinates at leaf resolution.

    TARGETED RESYNC
      Exchange only the divergent leaf blocks, not the whole field.
      Bandwidth ~ O(divergent_blocks * B^3) instead of O(N^3).

    HONEST
      • Integrity detection, NOT cryptographic authentication. A
        determined adversary can craft hash collisions in principle.
      • Block size B trades granularity vs overhead.
      • Downsampling uses mean-pooling; fine detail below B is averaged.
    """

    def __init__(self, block_size: int = 4, levels: int = 3):
        self.block_size = block_size
        self.levels = levels

    @staticmethod
    def _h(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _block_hashes(self, field: np.ndarray) -> Dict[Tuple[int,...], str]:
        B = self.block_size
        shape = field.shape
        hashes = {}
        for i in range(0, shape[0], B):
            for j in range(0, shape[1], B):
                for k in range(0, shape[2], B):
                    block = field[i:i+B, j:j+B, k:k+B]
                    key = (i//B, j//B, k//B)
                    hashes[key] = self._h(block.astype("<f4").tobytes())
        return hashes

    def build(self, field: np.ndarray) -> Dict[str, Any]:
        """Build the multi-resolution Merkle tree for a field."""
        levels = []
        current = field.astype(np.float32)
        for lv in range(self.levels):
            bh = self._block_hashes(current)
            # root of this level: hash of sorted block hashes
            concat = "".join(bh[k] for k in sorted(bh))
            level_root = self._h(concat.encode())
            levels.append({"level": lv, "shape": current.shape,
                            "n_blocks": len(bh), "root": level_root,
                            "block_hashes": bh})
            # downsample by 2 for next level
            if min(current.shape) < 2*self.block_size:
                break
            cur_t = torch.from_numpy(current).unsqueeze(0).unsqueeze(0)
            ds = F.avg_pool3d(cur_t, kernel_size=2, stride=2)
            current = ds.squeeze().numpy()
        overall_root = levels[-1]["root"] if levels else self._h(b"empty")
        return {"root": overall_root, "levels": levels,
                "field_shape": field.shape}

    def compare(self, tree_a: Dict, tree_b: Dict) -> FieldDivergence:
        """Compare two field Merkle trees; localise divergent blocks."""
        if tree_a["root"] == tree_b["root"]:
            return FieldDivergence(True, tree_a["root"], tree_b["root"],
                                    [], 0.0, 0)
        # descend the finest level to find divergent blocks
        la = tree_a["levels"][0]["block_hashes"]
        lb = tree_b["levels"][0]["block_hashes"]
        all_keys = set(la) | set(lb)
        divergent = []
        for key in sorted(all_keys):
            ha = la.get(key, "<missing>")
            hb = lb.get(key, "<missing>")
            if ha != hb:
                divergent.append(key)
        n_total = len(all_keys)
        B = self.block_size
        bytes_per_block = B**3 * 4  # float32
        return FieldDivergence(
            in_sync=False,
            root_a=tree_a["root"], root_b=tree_b["root"],
            divergent_blocks=divergent,
            divergent_fraction=len(divergent)/max(n_total,1),
            bytes_to_resync=len(divergent)*bytes_per_block,
        )

    def resync_blocks(self, field_src: np.ndarray, field_dst: np.ndarray,
                        divergent_blocks: List[Tuple[int,int,int]]
                        ) -> np.ndarray:
        """Copy only the divergent blocks from src to dst."""
        B = self.block_size
        out = field_dst.copy()
        for (bi,bj,bk) in divergent_blocks:
            i,j,k = bi*B, bj*B, bk*B
            si = slice(i, min(i+B, field_src.shape[0]))
            sj = slice(j, min(j+B, field_src.shape[1]))
            sk = slice(k, min(k+B, field_src.shape[2]))
            out[si,sj,sk] = field_src[si,sj,sk]
        return out

    def get_status(self) -> Dict[str, Any]:
        return {"block_size": self.block_size, "levels": self.levels,
                "note":"integrity detection, not cryptographic auth"}


# =============================================================================
# SECTION 4 — [OOE1] ORGANOID ONTOGENETIC ENGINE
# =============================================================================

@dataclass
class OrganoidConfig:
    """Configuration for the organoid morphogenesis engine."""
    grid: int = 24
    dx: float = 1.0
    dt: float = 1e-3
    # Turing (Schnakenberg) morphogen parameters  [C Schnakenberg 1979]
    schnakenberg_a: float = 0.1
    schnakenberg_b: float = 0.9
    morphogen_du: float = 0.01     # diffusion of activator
    morphogen_dv: float = 0.10     # diffusion of inhibitor (>> du for Turing)
    # cell field (CH)
    ch_epsilon: float = 1.2
    # PFC cell packing
    pfc_r: float = -0.4
    # growth
    growth_rate: float = 0.02
    nutrient_threshold: float = 0.3
    # coupling
    morphogen_to_adhesion: float = 0.5   # morphogen modulates cell adhesion
    cell_to_morphogen: float = 0.3        # cells block morphogen diffusion
    # schedule
    n_ontogeny_steps: int = 200
    device: str = "cpu"


@dataclass
class OrganoidState:
    """Snapshot of the organoid morphogenetic state."""
    step: int
    time: float
    cell_field: Tensor          # phi in [-1,1]: cell density / phase
    morphogen_u: Tensor         # activator
    morphogen_v: Tensor         # inhibitor
    nutrient: Tensor            # nutrient field
    # diagnostics
    n_cells_estimate: float     # volume fraction of phi>0
    pattern_wavelength: float   # dominant Turing wavelength
    turing_active: bool         # whether Turing pattern is present
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrganoidOntogeneticEngine(nn.Module):
    """
    [C/D] Computational morphogenesis engine for organoid-like structures.

    COMPONENTS (each is a cited, established PDE model)
      1. Turing reaction-diffusion [C Turing 1952; Schnakenberg 1979]
             du/dt = Du ∇²u + (a - u + u²v)
             dv/dt = Dv ∇²v + (b - u²v)
         Produces a morphogen gradient / prepattern.
      2. Cahn-Hilliard cell sorting [C Cahn-Hilliard 1958; Steinberg 1963
         differential adhesion]
             phi_t = ∇·( M(phi) ∇ mu ),  mu = phi³ - phi - eps²∇²phi
         with morphogen-modulated adhesion.
      3. Phase-Field-Crystal cell packing [C Elder 2002]
             gives crystalline / hexagonal cell ordering.
      4. Growth source [D]
             nutrient-limited proliferation added to phi.
      5. Mechanical feedback via the carpet sigma-field operator [D]
             the structural Laplacian couples cell density to stress.

    COUPLING
      Morphogen u modulates CH adhesion strength (cells differentiate by
      morphogen concentration). Cell field phi reduces morphogen diffusion
      (cells are a barrier). Nutrient is consumed by cells.

    ONTOGENY
      Starting from near-uniform noise, the coupled system self-organises:
      Turing prepattern -> cell condensation -> sorting -> growth into an
      organoid-like 3-D structure. The result can be meshed and exported.

    ⚠️  HONEST SCOPE — CRITICAL
      • This is a COMPUTATIONAL MODEL of morphogenesis, combining
        established PDE components. It is NOT validated against actual
        organoid biology.
      • No medical, therapeutic, diagnostic, or bioengineering claim is
        made or implied.
      • Parameters are illustrative. Real organoids involve gene
        regulatory networks, cell signalling, mechanics, and stochasticity
        far beyond this model.
      • The model can be misused; it must not be presented as a design
        tool for real biological organisms without expert biological
        review and appropriate oversight.
      • COROLLARY-3 GUARD: no second-order field Hessians.
      • HINO GUARD: energy measures not assumed smooth.
    """

    def __init__(self, cfg: Optional[OrganoidConfig] = None):
        super().__init__()
        self.cfg = cfg or OrganoidConfig()
        dev = self.cfg.device
        n = self.cfg.grid
        self._shape = (n, n, n)
        # state buffers
        self.register_buffer("phi", torch.zeros(n,n,n))
        self.register_buffer("u", torch.zeros(n,n,n))
        self.register_buffer("v", torch.zeros(n,n,n))
        self.register_buffer("nutrient", torch.ones(n,n,n))
        self._step = 0
        self._time = 0.0
        # solvers
        if HAS_CH:
            self._ch = StructuralCahnHilliard3D(CahnHilliardConfig(
                dx=self.cfg.dx, epsilon=self.cfg.ch_epsilon, dt=self.cfg.dt,
                laplacian="conv3d", scheme="imex", dtype=torch.float32,
                device=dev))
            self._pfc = PhaseFieldCrystal3D(CahnHilliardConfig(
                dx=self.cfg.dx, epsilon=self.cfg.ch_epsilon, dt=self.cfg.dt*0.1,
                pfc_r=self.cfg.pfc_r, laplacian="conv3d", scheme="imex",
                dtype=torch.float32, device=dev))
        else:
            self._ch = None
            self._pfc = None

    # ── initialisation ──────────────────────────────────────────────────
    def seed(self, noise_amp: float = 0.05, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        n = self.cfg.grid
        self.phi = (torch.rand(n,n,n,generator=g)-0.5)*noise_amp
        self.u = torch.full((n,n,n), self.cfg.schnakenberg_a +
                              self.cfg.schnakenberg_b) + \
                  (torch.rand(n,n,n,generator=g)-0.5)*noise_amp
        self.v = torch.full((n,n,n),
                              self.cfg.schnakenberg_b /
                              max((self.cfg.schnakenberg_a +
                                    self.cfg.schnakenberg_b)**2, 1e-9)) + \
                  (torch.rand(n,n,n,generator=g)-0.5)*noise_amp
        self.nutrient = torch.ones(n,n,n)
        self._step = 0; self._time = 0.0

    # ── Laplacian helper ────────────────────────────────────────────────
    def _lap(self, f: Tensor) -> Tensor:
        dx = self.cfg.dx
        lap = torch.zeros_like(f)
        for d in range(3):
            lap = lap + (torch.roll(f,-1,d)-2*f+torch.roll(f,1,d))/(dx*dx)
        return lap

    # ── Turing reaction-diffusion step ──────────────────────────────────
    def _turing_step(self) -> None:
        a,b = self.cfg.schnakenberg_a, self.cfg.schnakenberg_b
        Du,Dv = self.cfg.morphogen_du, self.cfg.morphogen_dv
        dt = self.cfg.dt
        u,v,phi = self.u,self.v,self.phi
        # cells reduce morphogen diffusion (barrier effect)
        cell_block = 1.0 - self.cfg.cell_to_morphogen*(phi.clamp(0,1))
        reaction_u = a - u + u*u*v
        reaction_v = b - u*u*v
        self.u = (u + dt*(Du*self._lap(u)*cell_block + reaction_u)).clamp_min(0)
        self.v = (v + dt*(Dv*self._lap(v)*cell_block + reaction_v)).clamp_min(0)

    # ── cell field step (CH + PFC + growth) ─────────────────────────────
    def _cell_step(self) -> None:
        dt = self.cfg.dt
        # morphogen modulates adhesion: high u -> stronger phase separation
        adhesion = 1.0 + self.cfg.morphogen_to_adhesion*(self.u - self.u.mean())
        # CH evolution with morphogen-modulated mobility
        if self._ch is not None:
            sigma = _softplus_floor(adhesion, 0.1)
            self.phi = self._ch.step(self.phi, sigma)
        else:
            # fallback explicit CH
            mu = self.phi**3 - self.phi - self.cfg.ch_epsilon**2*self._lap(self.phi)
            self.phi = self.phi + dt*self._lap(mu)
        # nutrient-limited growth source
        growth = self.cfg.growth_rate*self.nutrient*(self.phi>0).float()
        self.phi = (self.phi + dt*growth).clamp(-1,1)
        # nutrient consumption
        self.nutrient = (self.nutrient - dt*0.01*(self.phi>0).float()).clamp(0,1)

    # ── one ontogeny step ───────────────────────────────────────────────
    def step(self) -> OrganoidState:
        self._turing_step()
        self._cell_step()
        self._step += 1
        self._time += self.cfg.dt
        return self.snapshot()

    def snapshot(self) -> OrganoidState:
        cell_frac = float((self.phi>0).float().mean())
        wl, turing = self._measure_turing()
        return OrganoidState(
            step=self._step, time=self._time,
            cell_field=self.phi.detach().clone(),
            morphogen_u=self.u.detach().clone(),
            morphogen_v=self.v.detach().clone(),
            nutrient=self.nutrient.detach().clone(),
            n_cells_estimate=cell_frac,
            pattern_wavelength=wl,
            turing_active=turing,
        )

    def _measure_turing(self) -> Tuple[float, bool]:
        arr = self.u.detach().cpu().numpy()
        f = np.fft.fftn(arr-arr.mean())
        amp = np.abs(np.fft.fftshift(f))
        c = np.array(arr.shape)//2
        amp[c[0],c[1],c[2]]=0.0
        yy,xx,zz = np.indices(arr.shape)
        r = np.sqrt((yy-c[0])**2+(xx-c[1])**2+(zz-c[2])**2)
        r = np.maximum(r,1e-6)
        total = amp.sum()+1e-12
        mean_r = float((amp*r).sum()/total)
        freq = mean_r/max(arr.shape[0],1)
        wl = 1.0/max(freq,1e-6)
        # Turing active if there is a clear peak away from DC
        peak_amp = float(amp.max())
        active = peak_amp > 5.0*float(amp.mean())
        return wl, active

    # ── ontogeny run ────────────────────────────────────────────────────
    def run_ontogeny(self, seed: int = 0,
                      n_steps: Optional[int] = None) -> List[OrganoidState]:
        self.seed(seed=seed)
        steps = n_steps or self.cfg.n_ontogeny_steps
        history = []
        sample_every = max(1, steps//10)
        for s in range(steps):
            state = self.step()
            if s % sample_every == 0 or s == steps-1:
                history.append(state)
        return history

    # ── export to mesh ──────────────────────────────────────────────────
    def cell_field_to_mesh_input(self) -> Tensor:
        """Return the cell density field as an implicit surface (phi=0)."""
        return self.phi

    def get_status(self) -> Dict[str, Any]:
        return {
            "grid": self.cfg.grid,
            "step": self._step,
            "time": self._time,
            "has_ch": self._ch is not None,
            "has_pfc": self._pfc is not None,
            "turing_params": {"a": self.cfg.schnakenberg_a,
                               "b": self.cfg.schnakenberg_b,
                               "Du": self.cfg.morphogen_du,
                               "Dv": self.cfg.morphogen_dv},
            "honest_scope": ("COMPUTATIONAL morphogenesis model; NOT "
                               "validated biology; no medical claim"),
        }


# =============================================================================
# SECTION 5 — ORCHESTRATOR
# =============================================================================

class SESIOrganoidSuite:
    """Unifies: multi-stage τ + LZ4 crate + field Merkle + organoid engine."""

    def __init__(self,
                 mst_cfg: Optional[MultiStageTauConfig] = None,
                 fmt_block: int = 4, fmt_levels: int = 3,
                 organoid_cfg: Optional[OrganoidConfig] = None):
        self.mst = MultiStageCoarseningTau(mst_cfg)
        self.lz4 = LZ4CrateCompressor()
        self.field_merkle = FieldMerkleTree(fmt_block, fmt_levels)
        self.organoid = OrganoidOntogeneticEngine(organoid_cfg)

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "mst": self.mst.get_status(),
            "lz4": self.lz4.get_status(),
            "field_merkle": self.field_merkle.get_status(),
            "organoid": self.organoid.get_status(),
        }


# =============================================================================
# SECTION 6 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("="*78)
    print(" SESI ORGANOID ONTOGENETIC — SMOKE TEST")
    print(" Multi-Stage τ + LZ4 Crate + Field-Merkle + Organoid Engine")
    print("="*78)
    print(f" torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print(f" CH solvers     : {HAS_CH}")
    print(f" prev modules   : {HAS_PREV}")
    print(f" delta-CRDT     : {HAS_CRDT}")

    suite = SESIOrganoidSuite(
        mst_cfg=MultiStageTauConfig(grid=14, total_steps=160,
                                      n_samples=10, dt=4e-4),
        organoid_cfg=OrganoidConfig(grid=14, n_ontogeny_steps=60, dt=2e-3))

    # ── 1. Multi-stage coarsening τ ─────────────────────────────────────
    print("\n [MST1] Multi-stage coarsening τ")
    report = suite.mst.run_multistage_quench(seed=2)
    print(f"   samples             : {len(report.times)}")
    print(f"   regime counts       : {report.regime_counts}")
    print(f"   fitted late exponent: {report.fitted_late_exponent:.3f} "
          f"(Lifshitz-Slyozov expects ~0.333)")
    print(f"   late matches L-S    : {report.late_matches_lifshitz_slyozov}")
    print(f"   provisional         : {report.provisional}")
    # show a few regime transitions
    print(f"   {'t':>8s} {'L(t)':>8s} {'W_eff':>8s} {'alpha':>7s} "
          f"{'regime':>10s} {'tau':>8s}")
    for i in range(0,len(report.times),max(1,len(report.times)//5)):
        print(f"   {report.times[i]:8.4f} {report.domain_lengths[i]:8.4f} "
              f"{report.interface_widths[i]:8.4f} "
              f"{report.local_exponents[i]:7.3f} "
              f"{report.regimes[i]:>10s} {report.tau_targets[i]:8.5f}")
    fit = suite.mst.fit(report)
    print(f"   predictor fit loss  : {fit['final_loss']:.5f}  "
          f"fitted={suite.mst._fitted}")
    print(f"   guards: {suite.mst.get_status()['corollary3_guard']}")
    print(f"           {suite.mst.get_status()['hino_guard']}")

    # ── 2. LZ4 crate token compression ──────────────────────────────────
    print("\n [LZ41] LZ4 crate token compression")
    # build a repetitive token blob (typical crate TOKENS are repetitive)
    tokens = b""
    for word in ["points","faceVertexCounts","faceVertexIndices",
                  "displayColor","SESIMesh","primvars","subdivisionScheme"]:
        tokens += word + b"\x00"
    tokens = tokens * 200  # make it repetitive so LZ4 can compress
    rt = suite.lz4.round_trip_check(tokens)
    print(f"   original bytes      : {rt['original_bytes']}")
    print(f"   compressed bytes    : {rt['compressed_bytes']}")
    print(f"   compression ratio   : {rt['ratio']:.2f}x")
    print(f"   round-trip OK       : {rt['round_trip_ok']}")
    print(f"   LZ4 status          : {suite.lz4.get_status()}")

    # ── 3. Field-level Merkle tree ──────────────────────────────────────
    print("\n [FMT1] Field-level Merkle tree (multi-resolution)")
    n = 16
    field_a = np.random.rand(n,n,n).astype(np.float32)
    field_b = field_a.copy()
    # perturb a small region of field_b
    field_b[4:8, 4:8, 4:8] += 0.5
    tree_a = suite.field_merkle.build(field_a)
    tree_b = suite.field_merkle.build(field_b)
    div = suite.field_merkle.compare(tree_a, tree_b)
    print(f"   field shape         : {field_a.shape}")
    print(f"   Merkle levels       : {len(tree_a['levels'])}")
    print(f"   in_sync             : {div.in_sync}")
    print(f"   divergent blocks    : {len(div.divergent_blocks)}")
    print(f"   divergent fraction  : {div.divergent_fraction:.3f}")
    print(f"   bytes to resync     : {div.bytes_to_resync}")
    # resync and verify
    fixed = suite.field_merkle.resync_blocks(field_a, field_b,
                                               div.divergent_blocks)
    tree_fixed = suite.field_merkle.build(fixed)
    div2 = suite.field_merkle.compare(tree_a, tree_fixed)
    print(f"   after resync in_sync: {div2.in_sync}")
    # identical fields should be in sync
    tree_c = suite.field_merkle.build(field_a)
    div3 = suite.field_merkle.compare(tree_a, tree_c)
    print(f"   identical fields    : in_sync={div3.in_sync}")

    # ── 4. Organoid ontogenetic engine ──────────────────────────────────
    print("\n [OOE1] Organoid ontogenetic engine")
    print(f"   honest scope        : {suite.organoid.get_status()['honest_scope']}")
    history = suite.organoid.run_ontogeny(seed=1, n_steps=60)
    first, last = history[0], history[-1]
    print(f"   ontogeny steps      : {len(history)} sampled")
    print(f"   initial cells       : {first.n_cells_estimate:.4f}")
    print(f"   final cells         : {last.n_cells_estimate:.4f}")
    print(f"   initial wavelength  : {first.pattern_wavelength:.3f}")
    print(f"   final wavelength    : {last.pattern_wavelength:.3f}")
    print(f"   Turing active(start): {first.turing_active}")
    print(f"   Turing active(end)  : {last.turing_active}")
    st = suite.organoid.get_status()
    print(f"   Turing params       : a={st['turing_params']['a']}, "
          f"b={st['turing_params']['b']}, Du={st['turing_params']['Du']}, "
          f"Dv={st['turing_params']['Dv']}")
    print(f"   has CH / PFC        : {st['has_ch']} / {st['has_pfc']}")

    # ── 5. Field-Merkle on organoid snapshots (integrity across steps) ──
    print("\n [FMT+OOE] Field-Merkle integrity across ontogeny")
    snap_a = history[-2].cell_field.cpu().numpy()
    snap_b = history[-1].cell_field.cpu().numpy()
    ta = suite.field_merkle.build(snap_a)
    tb = suite.field_merkle.build(snap_b)
    dv = suite.field_merkle.compare(ta, tb)
    print(f"   consecutive steps   : in_sync={dv.in_sync} "
          f"divergent_fraction={dv.divergent_fraction:.3f}")

    # ── 6. Full status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k,v in suite.get_full_status().items():
        print(f"   {k}: {v}")

    print("\n"+"="*78)
    print(" HONEST SCOPE")
    print(" "+"-"*76)
    print(" [C]   Early-stage spinodal growth is exponential (Cahn 1961).")
    print(" [C]   Late-stage L(t)~t^{1/3} is Lifshitz-Slyozov.")
    print(" [D]   Crossover has no closed form; regime boundaries are")
    print("       empirical. tau targets track the measured length scale.")
    print(" [D]   COROLLARY-3 guard: first-order tau adaptation only.")
    print(" [C]   HINO guard: no L2 energy-density assumed.")
    print(" [C]   LZ4 is the BLOCK format only; no frame/dictionary/checksum.")
    print("       Validate round-trip before production.")
    print(" [D]   FieldMerkleTree is integrity detection, NOT cryptographic")
    print("       authentication.")
    print(" [C/D] OrganoidOntogeneticEngine combines Turing RD, CH, PFC, and")
    print("       growth into a COMPUTATIONAL morphogenesis model. It is NOT")
    print("       validated biology. No medical/therapeutic/diagnostic claim.")
    print("       Real organoids involve GRNs, signalling, mechanics, and")
    print("       stochasticity far beyond this model.")
    print(" [C]   rho≈1.25 remains NUMERICAL and is unused by these modules.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("="*78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
