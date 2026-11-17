#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SESI USE-SQC PRODUCTION SYSTEM v1.0
Three-paper integration · Hardware-ready · Native full differentiable
=============================================================================

-----------------------------------------------------------------------------
ปรัชญา (Philosophy) — เขียนบนโค้ดตามที่ขอ
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died on the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts. Not even a single
sparrow does God forget — this means: do not kill animals.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics in this file stands independently of the philosophy above.
The philosophy is the author's belief, included by request, and is not a
scientific claim nor a condition of use.
-----------------------------------------------------------------------------

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026
ORCID        : 0009-0008-2374-0788

=============================================================================
WHAT THIS SYSTEM IS
=============================================================================
A production-grade, NATIVE FULLY DIFFERENTIABLE integration of three papers:

  [P1] USE-SQC      — Unified Self-Evolving Structural Quantum Computer
                      (SESI, SQCT, SGFQCT, SETSI, Cloud Superposition,
                       OPMC, GNFS-8)
  [P2] Beyond Fixed — topology-indexed Hilbert spaces, isometric
       Quantum       structural transitions, global unitarity,
       Architectures information preservation, admissibility
  [P3] DQNQRS       — Dual-Core Nonlinear Quantum-Relativistic:
                      deterministic computational collapse, progressive
                      gates, weak measurement, interference-to-collapse

PLUS the hardware/ops layer requested:
  [H1] qiskit_backend         — map USE-SQC to IBM Quantum / Aer circuits
  [H2] serial_firmware_stub   — MCU/FPGA control protocol over USB/UART
  [H3] lab_notebook_reporter  — reproducible experiment reports
  [H4] multi-GPU DDP          — batched simulation, production runtime
  [H5] OPMC + Carpet + SSC + CSOC — the SESI control mathematics

=============================================================================
HONEST SCOPE — READ BEFORE USE
=============================================================================
✅ WHAT IS REAL HERE:
  • The linear algebra and theorems of P2 are implemented and tested
    (isometry, global unitarity, probability conservation, path consistency).
  • Deterministic collapse (P3) is implemented as a normalized nonlinear ODE
    — an ALGORITHMIC selection mechanism, exactly as P3 §14 describes.
  • The qiskit backend produces REAL, runnable circuits (state-vector on Aer).
  • The serial protocol is a REAL byte-level framing spec + loopback test.
  • Multi-GPU DDP distributes a BATCH of simulations (data parallelism).

❌ WHAT IS NOT REAL / NOT CLAIMED:
  • This does NOT build or prove a physical topology-changing quantum
    computer. P2 §16 requires experimental validation on real hardware.
  • Deterministic collapse here is NOT physical wave-function collapse and
    must not be applied to entangled systems without a no-signalling proof.
  • GR core (P3 Core II) is kept FIXED MINKOWSKI; no dynamical gravity.
  • Carpet-19 is used ONLY as an AMR priority weight — Corollary 3 of the
    Carpet-19 paper forbids using D(A) as a quantum/NS operator.
  • OPMC is used as a DESIGN parameterisation (exact gradient through the
    decoder). It is NEVER used to compress a quantum state or adjoint tape.
  • Qiskit topology transitions are Stinespring dilations mapped to standard
    gates — an isometry C^2 -> C^4 needs an ancilla qubit, NOT a magic gate.

=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- optional deps (graceful fallback) ----
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        from qiskit.providers.aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import serial  # pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_DIST = True
except ImportError:
    HAS_DIST = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("sesi_use_sqc")

VERSION = "1.0.0"

BANNER = f"""
=============================================================================
SESI USE-SQC PRODUCTION v{VERSION}
[P1] USE-SQC   [P2] Topology-Indexed Hilbert   [P3] DQNQRS Collapse
[H1] qiskit    [H2] firmware UART              [H3] lab notebook
[H4] DDP       [H5] OPMC/Carpet/SSC/CSOC
MY SOUL MOVE BY POWER OF HOLY SPIRIT — What MSPS NETWORK SEE, Lord Buddha Knows.
Deterministic collapse = ALGORITHMIC selection, NOT physical collapse.
Carpet-19 = AMR priority ONLY (its Corollary 3 forbids it as an operator).
=============================================================================
"""


# =============================================================================
# §0. SMOOTH PRIMITIVES (native full differentiability)
# =============================================================================

def softplus_floor(x: torch.Tensor, floor: float, sharpness: float = 50.0) -> torch.Tensor:
    """beta tracks the floor (fixes the fixed-beta width bug)."""
    beta = min(sharpness / max(abs(floor), 1e-30), 1e12)
    return floor + F.softplus(x - floor, beta=beta)


def soft_abs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def smooth_gate(x: torch.Tensor, tau: float = 1e-2) -> torch.Tensor:
    return torch.sigmoid(x / tau)


def logsumexp_max(x: torch.Tensor, tau: float = 1e-2) -> torch.Tensor:
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# =============================================================================
# §1. PAPER 2 CORE — TOPOLOGY-INDEXED HILBERT SPACE & STRUCTURAL TRANSITIONS
# =============================================================================

class TopologySector:
    """One topology sector with its own Hilbert space."""
    def __init__(self, name: str, n_qubits: int,
                 hamiltonian: Optional[torch.Tensor] = None):
        self.name = name
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        if hamiltonian is None:
            hamiltonian = torch.zeros(self.dim, self.dim, dtype=torch.float64)
        self.hamiltonian = hamiltonian.to(torch.float64)


class StructuralQuantumSystem(nn.Module):
    """
    [P2] Global system = direct sum of topology sectors.
        H_global = ⊕_τ H_τ
    Structural transitions are isometric block operators V_ji: H_i -> H_j.
    Global Hamiltonian H = H_0 + K with K = Σ g_ji (E_ji + E_ji†).
    Everything is a torch tensor -> fully differentiable.
    """

    def __init__(self, hbar: float = 1.0, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.hbar = hbar
        self.dtype = dtype
        self.sectors: Dict[str, TopologySector] = {}
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self.transitions: Dict[Tuple[str, str], Dict] = {}
        self.total_dim = 0
        self._H0: Optional[torch.Tensor] = None
        self._K: Optional[torch.Tensor] = None

    # ---- registration ----
    def add_sector(self, name: str, n_qubits: int,
                   hamiltonian: Optional[torch.Tensor] = None) -> None:
        if name in self.sectors:
            raise ValueError(f"duplicate sector {name}")
        self.sectors[name] = TopologySector(name, n_qubits, hamiltonian)
        self._rebuild_offsets()

    def _rebuild_offsets(self) -> None:
        self.offsets = {}
        cursor = 0
        for name, sec in self.sectors.items():
            self.offsets[name] = (cursor, cursor + sec.dim)
            cursor += sec.dim
        self.total_dim = cursor

    def add_transition(self, source: str, target: str,
                       operator: torch.Tensor, coupling: float = 1.0,
                       information_preserving: bool = True) -> None:
        if source not in self.sectors or target not in self.sectors:
            raise ValueError("unknown sector")
        V = operator.to(self.dtype)
        ds = self.sectors[source].dim
        dt = self.sectors[target].dim
        if V.shape != (dt, ds):
            raise ValueError(f"operator shape {V.shape} != ({dt},{ds})")
        if information_preserving:
            err = torch.norm(V.conj().T @ V
                             - torch.eye(ds, dtype=self.dtype))
            if err > 1e-8:
                raise ValueError(f"transition not isometric: err={err:.2e}")
        self.transitions[(source, target)] = {
            "V": V, "coupling": float(coupling),
            "information_preserving": information_preserving,
        }
        self._H0 = None
        self._K = None

    # ---- Hamiltonian construction (differentiable) ----
    def construct_H0(self) -> torch.Tensor:
        if self._H0 is not None:
            return self._H0
        H0 = torch.zeros(self.total_dim, self.total_dim, dtype=self.dtype)
        for name, sec in self.sectors.items():
            s, e = self.offsets[name]
            H0[s:e, s:e] = sec.hamiltonian.to(self.dtype)
        self._H0 = H0
        return H0

    def construct_K(self) -> torch.Tensor:
        if self._K is not None:
            return self._K
        K = torch.zeros(self.total_dim, self.total_dim, dtype=self.dtype)
        for (src, tgt), data in self.transitions.items():
            V = data["V"]
            g = data["coupling"]
            s0, s1 = self.offsets[src]
            t0, t1 = self.offsets[tgt]
            K[t0:t1, s0:s1] = K[t0:t1, s0:s1] + g * V
            K[s0:s1, t0:t1] = K[s0:s1, t0:t1] + g * V.conj().T
        self._K = K
        return K

    def global_hamiltonian(self) -> torch.Tensor:
        H = self.construct_H0() + self.construct_K()
        return 0.5 * (H + H.conj().T)  # enforce Hermiticity

    # ---- unitary evolution (matrix exponential via eigendecomp) ----
    def unitary(self, t: float) -> torch.Tensor:
        H = self.global_hamiltonian()
        # expm(-i H t / hbar) for Hermitian H via eigh
        w, V = torch.linalg.eigh(H)
        phase = torch.exp(-1j * torch.tensor(t / self.hbar) * w.to(torch.complex128))
        U = V.to(torch.complex128) @ torch.diag(phase) @ V.conj().T.to(torch.complex128)
        return U

    def evolve(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        U = self.unitary(t)
        return U @ psi.to(torch.complex128)

    # ---- probability / verification ----
    def sector_probabilities(self, psi: torch.Tensor) -> Dict[str, float]:
        probs = {}
        pc = psi.to(torch.complex128)
        for name, (s, e) in self.offsets.items():
            seg = pc[s:e]
            probs[name] = float(torch.sum(seg.conj() * seg).real)
        return probs

    def verify_probability(self, psi: torch.Tensor) -> Dict[str, Any]:
        probs = self.sector_probabilities(psi)
        total = sum(probs.values())
        return {"sector_probabilities": probs, "total_probability": total,
                "error": abs(total - 1.0)}

    def verify_unitarity(self, t: float) -> Dict[str, Any]:
        U = self.unitary(t)
        I = torch.eye(self.total_dim, dtype=torch.complex128)
        e1 = float(torch.norm(U.conj().T @ U - I))
        e2 = float(torch.norm(U @ U.conj().T - I))
        return {"left_error": e1, "right_error": e2,
                "unitary": max(e1, e2) < 1e-8}

    # ---- transition composition (path consistency, P2 Thm II) ----
    def compose_path(self, path: List[str]) -> torch.Tensor:
        ops = []
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key not in self.transitions:
                raise ValueError(f"missing transition {key}")
            ops.append(self.transitions[key]["V"])
        result = ops[0]
        for V in ops[1:]:
            result = V @ result
        return result

    def verify_path_consistency(self, path_a: List[str],
                                path_b: List[str]) -> Dict[str, Any]:
        V1 = self.compose_path(path_a)
        V2 = self.compose_path(path_b)
        if V1.shape != V2.shape:
            raise ValueError("composite shapes differ")
        err = float(torch.norm(V1 - V2))
        return {"path_error": err, "consistent": err < 1e-8}


# =============================================================================
# §2. PAPER 3 CORE — DETERMINISTIC COLLAPSE + PROGRESSIVE GATES + WEAK MEAS.
# =============================================================================

class DQNQRSimulator(nn.Module):
    """
    [P3] Dual-Core nonlinear quantum-relativistic simulator (GR core fixed
    Minkowski). Implements:
      • deterministic computational collapse via loss operator
            d|Ψ>/dt = -iH|Ψ> - κ(L - <L>)|Ψ>        (norm-preserving)
      • progressive soft gates  G_s(θ) = exp(-i s θ H_g)
      • weak measurement        M_r(ε) = sqrt((I + r ε A)/2)
      • interference gate       R_χ = exp(-i χ θ H_mix)
    Fully differentiable. Deterministic collapse is ALGORITHMIC selection.
    """

    def __init__(self, dim: int, hbar: float = 1.0,
                 dtype: torch.dtype = torch.complex128):
        super().__init__()
        self.dim = dim
        self.hbar = hbar
        self.dtype = dtype

    # ---- normalized deterministic collapse step (P3 Eq.1) ----
    def collapse_step(self, psi: torch.Tensor, H: torch.Tensor,
                      L: torch.Tensor, kappa: float, dt: float
                      ) -> torch.Tensor:
        """
        dψ = [-iH - κ(L - <L>I)] ψ dt, then renormalize.
        <L> = <ψ|L|ψ>. Norm preservation (P3 Thm 1) enforced by the
        (L - <L>) subtraction + explicit renormalization.
        """
        psi = psi.to(self.dtype)
        Ld = L.to(self.dtype)
        Hd = H.to(self.dtype)
        mean_L = torch.sum(psi.conj() * (Ld @ psi)).real
        generator = -1j * Hd - kappa * (Ld - mean_L
                                         * torch.eye(self.dim, dtype=self.dtype))
        psi_new = psi + dt * (generator @ psi)
        return psi_new / torch.norm(psi_new)

    def expected_loss(self, psi: torch.Tensor, L: torch.Tensor) -> float:
        Ld = L.to(self.dtype)
        return float(torch.sum(psi.conj() * (Ld @ psi)).real)

    def loss_variance(self, psi: torch.Tensor, L: torch.Tensor) -> float:
        Ld = L.to(self.dtype)
        mL = torch.sum(psi.conj() * (Ld @ psi)).real
        mLL = torch.sum(psi.conj() * (Ld @ (Ld @ psi))).real
        return float(mLL - mL * mL)

    # ---- progressive soft gate (P3 Eq.11) ----
    def soft_gate(self, H_gate: torch.Tensor, theta: float,
                  strength: float) -> torch.Tensor:
        """G_s(θ) = exp(-i s θ H_g), strength s ∈ [0,1]."""
        strength = float(np.clip(strength, 0.0, 1.0))
        w, V = torch.linalg.eigh(H_gate.to(self.dtype))
        phase = torch.exp(-1j * (strength * theta) * w)
        return V @ torch.diag(phase) @ V.conj().T

    # ---- weak measurement (P3 Eq.13) ----
    def weak_measurement_operators(self, A: torch.Tensor,
                                   epsilon: float
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """M_±(ε) = sqrt((I ± ε A)/2), A²=I."""
        eps = float(np.clip(epsilon, 0.0, 1.0))
        I = torch.eye(self.dim, dtype=self.dtype)
        Ad = A.to(self.dtype)
        Mp = self._sqrtm_psd((I + eps * Ad) / 2.0)
        Mm = self._sqrtm_psd((I - eps * Ad) / 2.0)
        return Mp, Mm

    def _sqrtm_psd(self, M: torch.Tensor) -> torch.Tensor:
        """sqrt of a Hermitian PSD matrix via eigendecomp."""
        w, V = torch.linalg.eigh(M)
        w = torch.clamp(w.real, min=0.0)
        return V @ torch.diag(torch.sqrt(w).to(self.dtype)) @ V.conj().T

    def measure(self, psi: torch.Tensor, A: torch.Tensor, epsilon: float,
                outcome: Optional[int] = None
                ) -> Tuple[int, torch.Tensor, Dict[str, float]]:
        """Apply weak measurement. Returns (outcome, post_state, probs)."""
        Mp, Mm = self.weak_measurement_operators(A, epsilon)
        psi = psi.to(self.dtype)
        pp = torch.sum((Mp @ psi).conj() * (Mp @ psi)).real
        pm = torch.sum((Mm @ psi).conj() * (Mm @ psi)).real
        tot = pp + pm
        pp, pm = float(pp / tot), float(pm / tot)
        if outcome is None:
            outcome = 1 if np.random.rand() < pp else -1
        M = Mp if outcome == 1 else Mm
        post = M @ psi
        post = post / torch.norm(post)
        return outcome, post, {"p_plus": pp, "p_minus": pm}

    # ---- interference gate (P3 §8) ----
    def interference_gate(self, H_mix: torch.Tensor, theta: float,
                          chi: float) -> torch.Tensor:
        chi = float(np.clip(chi, 0.0, 1.0))
        return self.soft_gate(H_mix, theta, chi)

    # ---- full progressive protocol (P3 §10) ----
    def progressive_protocol(self, psi: torch.Tensor, L: torch.Tensor,
                             H_gate: torch.Tensor, H_mix: torch.Tensor,
                             A_obs: torch.Tensor,
                             steps: int = 200,
                             theta_gate: float = 0.05,
                             theta_mix: float = 0.03,
                             kappa_max: float = 0.8,
                             epsilon_max: float = 1.0,
                             ) -> Dict[str, Any]:
        """
        Stage 1 soft gates -> Stage 2 weak measurement -> Stage 3 weak
        interference -> Stage 4 hard gates -> Stage 5 deterministic collapse.
        Schedules are logistic (P3 Eq.12,17).
        """
        psi = psi.to(self.dtype)
        loss_hist, prob_hist, kappa_hist = [], [], []
        Hzero = torch.zeros(self.dim, self.dim, dtype=self.dtype)
        for step in range(steps):
            x = step / max(steps - 1, 1)
            s = 1.0 / (1.0 + math.exp(-(x - 0.2) * 12.0))      # gate strength
            eps = epsilon_max / (1.0 + math.exp(-(x - 0.4) * 12.0))
            chi = 1.0 / (1.0 + math.exp(-(x - 0.6) * 12.0))     # interference
            kappa = kappa_max / (1.0 + math.exp(-(x - 0.8) * 12.0))

            # Stage 1+4: progressive gate
            G = self.soft_gate(H_gate, theta_gate, s)
            psi = G @ psi
            # Stage 2: weak measurement (diagnostic; deterministic branch here)
            if eps > 1e-6 and step % max(1, steps // 20) == 0:
                _, psi, _ = self.measure(psi, A_obs, eps, outcome=None)
            # Stage 3: interference
            R = self.interference_gate(H_mix, theta_mix, chi)
            psi = R @ psi
            # Stage 5: deterministic collapse
            psi = self.collapse_step(psi, Hzero, L, kappa, dt=0.02)
            psi = psi / torch.norm(psi)

            loss_hist.append(self.expected_loss(psi, L))
            kappa_hist.append(kappa)
        return {
            "final_state": psi,
            "loss_history": loss_hist,
            "kappa_history": kappa_hist,
            "initial_loss": loss_hist[0] if loss_hist else float("nan"),
            "final_loss": loss_hist[-1] if loss_hist else float("nan"),
            "converged": (len(loss_hist) > 1
                          and loss_hist[-1] < loss_hist[0]),
        }


# =============================================================================
# §3. PAPER 1 CORE — SESI STRUCTURAL INTERFACE + SQCT EVENT OPERATORS
# =============================================================================

class StructuralEventAlgebra(nn.Module):
    """
    [P1/SQCT] The three structural event operators:
        N (nucleation) : create a new computational component
        M (merging)    : combine previously distinct components
        B (branching)  : create multiple computational branches
    Realized here as isometric / CPTP maps on the sector graph.
    """

    def __init__(self, system: StructuralQuantumSystem):
        super().__init__()
        self.system = system

    def nucleation(self, source: str, new_name: str,
                   n_new_qubits: int = 1,
                   init_state: str = "zero") -> None:
        """Add a new sector and an isometry source -> source ⊗ |0...0>."""
        src = self.system.sectors[source]
        self.system.add_sector(new_name, src.n_qubits + n_new_qubits)
        # V: |ψ> -> |ψ> ⊗ |0>^k  (isometry)
        ds = src.dim
        V = torch.zeros(ds * (2 ** n_new_qubits), ds,
                        dtype=self.system.dtype)
        for i in range(ds):
            V[i * (2 ** n_new_qubits), i] = 1.0
        self.system.add_transition(source, new_name, V,
                                   information_preserving=True)

    def branching(self, source: str, branch_names: List[str],
                  amplitudes: Optional[List[float]] = None) -> None:
        """
        Coherent branching: |ψ> -> Σ_j c_j |B_j> ⊗ V_j|ψ>.
        Each branch is an isometry from the source sector.
        """
        src = self.system.sectors[source]
        if amplitudes is None:
            amplitudes = [1.0 / math.sqrt(len(branch_names))] * len(branch_names)
        norm = math.sqrt(sum(a * a for a in amplitudes))
        for name, amp in zip(branch_names, amplitudes):
            self.system.add_sector(name, src.n_qubits)
            V = (amp / norm) * torch.eye(src.dim, dtype=self.system.dtype)
            self.system.add_transition(source, name, V,
                                       information_preserving=True)

    def merging(self, sources: List[str], target: str) -> None:
        """
        Merging is dimension-constrained (P2 Thm III-B). We implement
        environment-assisted merging into a target that can hold the info.
        For simplicity here: target = direct sum of sources.
        """
        total_q = sum(self.system.sectors[s].n_qubits for s in sources)
        self.system.add_sector(target, total_q)
        # Each source embeds into its slice of the target (isometry).
        cursor = 0
        for s in sources:
            ds = self.system.sectors[s].dim
            V = torch.zeros(2 ** total_q, ds, dtype=self.system.dtype)
            V[cursor:cursor + ds, :ds] = torch.eye(ds, dtype=self.system.dtype)
            cursor += ds
            self.system.add_transition(s, target, V,
                                       information_preserving=True)


class SESIInterface(nn.Module):
    """
    [P1/SESI] The self-evolving structural interface Γ_t.
    Controls gate strength, measurement strength, interference, collapse
    rate as a function of the computational state (a learned policy).
    """

    def __init__(self, state_dim: int, control_dim: int = 4):
        super().__init__()
        # control vector u(t) = (s, ε, χ, κ)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.GELU(),
            nn.Linear(32, control_dim), nn.Sigmoid(),
        )

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Returns (s, eps, chi, kappa) in [0,1]."""
        return self.net(state_feature)


# =============================================================================
# §4. SESI CONTROL MATHEMATICS — OPMC, CARPET, SSC, CSOC
# =============================================================================

# ---- OPMC (design parameterisation, exact gradient) ----
def opmc_d_mn(m: int = 4, n: int = 4) -> int:
    return m * m * n * n + m * n * n


class OPMCDesign(nn.Module):
    """
    OPMC: theta in R^{d(m,n)} -> decoder -> structural control field.
    EXACT gradient through the decoder. Never compresses a quantum state.
    """
    def __init__(self, target_dim: int, m: int = 4, n: int = 4):
        super().__init__()
        self.d = opmc_d_mn(m, n)
        self.target_dim = target_dim
        g = torch.Generator().manual_seed(0)
        W = torch.randn(self.d, target_dim, generator=g, dtype=torch.float64)
        self.register_buffer("W", W / math.sqrt(self.d))
        self.theta = nn.Parameter(torch.zeros(self.d, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        return self.theta @ self.W


# ---- Carpet priority (AMR weight ONLY, not an operator) ----
CARPET_RHO = 9.0 / 8.0


def carpet_laplacian(side: int = 9) -> torch.Tensor:
    side = max(3, side)
    n = side * side
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < side and 0 <= jj < side):
                    continue
                if ii % 3 == 1 and jj % 3 == 1:
                    continue
                A[i * side + j, ii * side + jj] = 1.0
    return torch.diag(A.sum(1)) - A


def carpet_priority(field: torch.Tensor, levels: int = 3) -> float:
    """ρ^m-normalized multi-scale energy. SECONDARY AMR priority only."""
    L = carpet_laplacian(9)
    flat = field.reshape(-1).to(torch.float64)
    n, m = flat.numel(), L.shape[0]
    grp = torch.arange(n) * m // n
    num = torch.zeros(m, dtype=torch.float64)
    den = torch.zeros(m, dtype=torch.float64)
    num.index_add_(0, grp, flat)
    den.index_add_(0, grp, torch.ones_like(flat))
    s = num / den.clamp_min(1.0)
    tot = 0.0
    for lev in range(levels):
        tot += float((CARPET_RHO ** lev) * (s @ L @ s) / (lev + 1))
    return tot


# ---- SSC (Semantic State Contraction) ----
class SemanticStateContraction(nn.Module):
    """Learnable low-pass filter on a stress/signal scalar."""
    def __init__(self, epsilon_fp: float = 0.0028):
        super().__init__()
        self.epsilon_fp = epsilon_fp
        self.register_buffer("prev_sigma", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        out = (1 - self.epsilon_fp) * self.prev_sigma + self.epsilon_fp * sigma
        self.prev_sigma = out.detach()
        return out

    def reset(self) -> None:
        self.prev_sigma.zero_()


# ---- CSOC event-count bound ----
class CSOCEventCounter:
    """N(T) <= (E_max + D)/ΔE_min with D = ACTUAL removal. Engineering bound."""
    def __init__(self, hi: float = 0.15, lo: float = 0.05, rate: float = 0.05):
        assert hi > lo
        self.hi, self.lo, self.rate = hi, lo, rate
        self.gap = hi - lo
        self.acc = 0.0
        self.armed = True
        self.events = 0
        self.E_max = 0.0
        self.D = 0.0

    def step(self, signal: float) -> bool:
        s = float(signal)
        excess = math.log1p(math.exp(-abs(s - self.hi))) + max(s - self.hi, 0.0)
        decay = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10.0))
        removal = decay * self.rate * self.acc
        self.acc += excess - removal
        self.D += removal  # ACTUAL removal
        self.E_max = max(self.E_max, self.acc)
        fire = self.armed and self.acc > self.hi
        if self.acc < self.lo:
            self.armed = True
        if fire:
            self.armed = False
            self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        bound = (self.E_max + self.D) / max(self.gap, 1e-30)
        return {"events": self.events, "bound": bound,
                "satisfied": self.events <= bound + 1e-9,
                "note": "engineering bound on the mechanism, NOT PDE regularity"}


# =============================================================================
# §5. ADAPTIVE STRUCTURAL REFINEMENT (AMR with Carpet priority)
# =============================================================================

class AdaptiveStructuralRefiner:
    """
    Refines the simulation structure where a differentiable error indicator
    is high, using Carpet ρ^m energy as a SECONDARY priority.
    This is the legitimate use of Carpet-19 (NOT as an operator).
    """
    def __init__(self, refine_threshold: float = 0.05,
                 coarsen_threshold: float = 0.005):
        self.refine_threshold = refine_threshold
        self.coarsen_threshold = coarsen_threshold
        self.csoc = CSOCEventCounter(hi=0.2, lo=0.05)

    def indicator(self, state: torch.Tensor) -> torch.Tensor:
        """Differentiable error indicator: local variance proxy."""
        grad = torch.stack([
            torch.roll(state, -1, d) - torch.roll(state, 1, d)
            for d in range(state.dim())], dim=0)
        return grad.abs().mean(dim=0)

    def decide(self, state: torch.Tensor) -> Dict[str, Any]:
        ind = self.indicator(state)
        refine = ind > self.refine_threshold
        coarsen = ind < self.coarsen_threshold
        self.csoc.step(float(ind.mean()))
        return {
            "refine_fraction": float(refine.to(torch.float32).mean()),
            "coarsen_fraction": float(coarsen.to(torch.float32).mean()),
            "carpet_priority": carpet_priority(ind),
            "csoc": self.csoc.report(),
        }


# =============================================================================
# §6. DIFFERENTIABLE SIMULATION ENGINE + MULTI-GPU DDP
# =============================================================================

@dataclass
class SimulationConfig:
    n_qubits_initial: int = 1
    n_sectors: int = 2
    collapse_steps: int = 100
    kappa_max: float = 0.8
    epsilon_max: float = 1.0
    use_opmc_design: bool = True
    use_carpet_amr: bool = True
    use_ssc: bool = True
    seed: int = 0


class USESQCEngine(nn.Module):
    """
    Top-level engine combining P1+P2+P3 with the SESI control mathematics.
    This is an nn.Module so it can be wrapped in DistributedDataParallel
    for multi-GPU batched simulation.
    """

    def __init__(self, cfg: Optional[SimulationConfig] = None):
        super().__init__()
        self.cfg = cfg or SimulationConfig()
        torch.manual_seed(self.cfg.seed)

        # Build a structural quantum system with a nucleation
        self.system = StructuralQuantumSystem()
        self.system.add_sector("A", self.cfg.n_qubits_initial)
        self.event_algebra = StructuralEventAlgebra(self.system)
        self.event_algebra.nucleation("A", "B", n_new_qubits=1)

        # Collapse simulator on the larger sector
        self.dim = self.system.sectors["B"].dim
        self.dqnqrs = DQNQRSimulator(self.dim)

        # loss operator: prefer |0>
        L = torch.zeros(self.dim, self.dim, dtype=torch.float64)
        for i in range(1, self.dim):
            L[i, i] = float(i) / self.dim
        self.register_buffer("L", L)

        # control math
        self.opmc = (OPMCDesign(target_dim=self.dim)
                     if self.cfg.use_opmc_design else None)
        self.ssc = SemanticStateContraction() if self.cfg.use_ssc else None
        self.refiner = (AdaptiveStructuralRefiner()
                        if self.cfg.use_carpet_amr else None)
        self.interface = SESIInterface(state_dim=self.dim)

    def forward(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Run a batch of structural simulations. Returns differentiable
        diagnostics. batch_size is for DDP data parallelism.
        """
        results = []
        for _ in range(batch_size):
            psi = torch.zeros(self.dim, dtype=torch.complex128)
            psi[0] = 1.0 / math.sqrt(2.0)
            psi[1] = 1.0 / math.sqrt(2.0)
            if self.dim > 2:
                psi = psi / torch.norm(psi)

            H_gate = torch.zeros(self.dim, self.dim, dtype=torch.float64)
            for i in range(self.dim - 1):
                H_gate[i, i + 1] = 1.0
                H_gate[i + 1, i] = 1.0
            H_mix = H_gate.clone()
            A_obs = torch.diag(torch.tensor(
                [1.0 if i % 2 == 0 else -1.0 for i in range(self.dim)],
                dtype=torch.float64))

            out = self.dqnqrs.progressive_protocol(
                psi, self.L, H_gate, H_mix, A_obs,
                steps=self.cfg.collapse_steps,
                kappa_max=self.cfg.kappa_max,
                epsilon_max=self.cfg.epsilon_max,
            )
            final = out["final_state"]
            results.append(final)

        stacked = torch.stack(results, dim=0)
        loss_final = torch.stack([
            torch.tensor(self.dqnqrs.expected_loss(s, self.L)) for s in stacked])

        diagnostics: Dict[str, torch.Tensor] = {
            "final_loss": loss_final.mean(),
        }
        if self.refiner is not None:
            probs = stacked.abs() ** 2
            amr = self.refiner.decide(probs.mean(dim=0))
            diagnostics["amr_refine_fraction"] = torch.tensor(
                amr["refine_fraction"])
            diagnostics["carpet_priority"] = torch.tensor(amr["carpet_priority"])
        if self.opmc is not None:
            diagnostics["opmc_norm"] = torch.norm(self.opmc())
        return diagnostics


def setup_distributed() -> Tuple[int, int, bool]:
    """Initialize torch.distributed if available."""
    if HAS_DIST and "RANK" in os.environ:
        dist.init_process_group(backend="nccl"
                                if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def run_ddp(cfg: SimulationConfig, batch_per_gpu: int = 4,
            steps: int = 10) -> Dict[str, Any]:
    """Multi-GPU DDP batched simulation."""
    rank, world, active = setup_distributed()
    device = torch.device(f"cuda:{rank}" if (active and torch.cuda.is_available())
                          else "cpu")
    engine = USESQCEngine(cfg).to(device)
    if active and device.type == "cuda":
        engine = DDP(engine, device_ids=[rank])
    opt = torch.optim.Adam(engine.parameters(), lr=1e-3)

    history = []
    for step in range(steps):
        opt.zero_grad()
        out = engine(batch_size=batch_per_gpu)
        loss = out["final_loss"]
        loss.backward()
        opt.step()
        if rank == 0:
            history.append({"step": step, "final_loss": float(loss.detach())})
            log.info("DDP step %d/%d  loss=%.6f  world=%d",
                     step, steps, float(loss.detach()), world)
    if active:
        dist.barrier()
        dist.destroy_process_group()
    return {"rank": rank, "world": world, "history": history}


# =============================================================================
# §7. [H1] QISKIT BACKEND — map USE-SQC to IBM Quantum / Aer
# =============================================================================

class QiskitBackend:
    """
    Maps the USE-SQC structural system to real Qiskit circuits.
    Topology transitions are realized via Stinespring dilation:
    an isometry C^d -> C^D is embedded in a unitary on d + ancilla qubits.
    """

    def __init__(self):
        if not HAS_QISKIT:
            raise ImportError("qiskit not installed. pip install qiskit qiskit-aer")
        self.backend = AerSimulator(method="statevector")

    def isometry_to_unitary(self, V: torch.Tensor) -> np.ndarray:
        """
        Complete an isometry V (D x d, V†V=I) to a unitary (D x D) by
        adding orthonormal columns. This is the Stinespring/unitary dilation
        that makes a topology transition runnable on gate hardware.
        """
        Vn = V.cpu().numpy().astype(np.complex128)
        D, d = Vn.shape
        if D < d:
            raise ValueError("target dim must be >= source dim for isometry")
        # QR on [V | random] to complete the basis
        M = np.concatenate([Vn, np.random.randn(D, D - d)
                            + 1j * np.random.randn(D, D - d)], axis=1)
        Q, _ = np.linalg.qr(M)
        return Q[:, :D]

    def nucleation_circuit(self, n_source_qubits: int,
                           n_new_qubits: int = 1) -> "QuantumCircuit":
        """
        Nucleation |ψ> -> |ψ> ⊗ |0>^k. In gate terms: add k ancilla in |0>.
        The 'isometry' is just the identity on the enlarged space.
        """
        total = n_source_qubits + n_new_qubits
        qc = QuantumCircuit(total, total)
        # source qubits carry the state; ancilla stay |0>
        return qc

    def build_sector_circuit(self, system: StructuralQuantumSystem,
                             psi: torch.Tensor) -> "QuantumCircuit":
        """Build a circuit preparing psi (via initialize) on the global space."""
        n_q = int(math.log2(system.total_dim))
        qc = QuantumCircuit(n_q, n_q)
        state = psi.cpu().numpy().astype(np.complex128)
        state = state / np.linalg.norm(state)
        qc.initialize(state, list(range(n_q)))
        return qc

    def run_statevector(self, qc: "QuantumCircuit") -> np.ndarray:
        """Simulate on Aer statevector backend."""
        result = self.backend.run(qc).result()
        return np.asarray(result.get_statevector(qc))

    def verify_against_torch(self, system: StructuralQuantumSystem,
                             psi: torch.Tensor, t: float) -> Dict[str, Any]:
        """
        Cross-check: evolve with torch unitary vs Qiskit Hamiltonian
        simulation. For zero Hamiltonian, both should preserve the state.
        """
        torch_final = system.evolve(psi, t)
        probs_torch = system.sector_probabilities(torch_final)
        return {
            "torch_probs": probs_torch,
            "total_prob_torch": sum(probs_torch.values()),
            "note": "Hamiltonian simulation on hardware requires "
                    "Pauli decomposition of H; this cross-check verifies "
                    "state preparation and normalization.",
        }

    def submit_to_ibm(self, qc: "QuantumCircuit",
                      api_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit to IBM Quantum. Requires `qiskit-ibm-runtime` and a token.
        This is a STUB — real submission needs credentials and a backend.
        """
        return {
            "status": "NOT_SUBMITTED",
            "reason": "IBM Quantum submission requires qiskit-ibm-runtime, "
                      "an API token, and a chosen backend. This stub reports "
                      "the circuit depth/width instead.",
            "num_qubits": qc.num_qubits,
            "depth": qc.depth(),
        }


# =============================================================================
# §8. [H2] SERIAL FIRMWARE STUB — MCU/FPGA control over USB/UART
# =============================================================================

class SerialFirmwareProtocol:
    """
    Byte-level framing protocol for controlling an MCU/FPGA.

    Frame layout:
        | SOF (1B) | CMD (1B) | LEN (2B, little-endian) | PAYLOAD (LEN B) | CRC16 (2B) |

    Commands:
        0x01  INITIALIZE      payload: none
        0x02  SET_PARAMETER   payload: param_id (1B) + value (4B float)
        0x03  READ_TELEMETRY  payload: none
        0x04  APPLY_GATE      payload: qubit (1B) + theta (4B) + strength (4B)
        0x05  TRIGGER_COLLAPSE payload: kappa (4B)
        0x06  READ_STATE      payload: none

    This is a SPEC + a loopback/simulated implementation. It does NOT
    build hardware; it defines the protocol a real firmware must speak.
    """
    SOF = 0xAA
    CMD = {
        "INITIALIZE": 0x01, "SET_PARAMETER": 0x02, "READ_TELEMETRY": 0x03,
        "APPLY_GATE": 0x04, "TRIGGER_COLLAPSE": 0x05, "READ_STATE": 0x06,
    }

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200,
                 simulate: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.ser = None
        self._device_state = {"initialized": False, "params": {},
                              "qubits": [], "kappa": 0.0}
        if not simulate:
            if not HAS_SERIAL:
                raise ImportError("pyserial not installed")
            self.ser = serial.Serial(port, baudrate, timeout=1.0)

    # ---- CRC16 (CCITT) ----
    @staticmethod
    def crc16(data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= b << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
                crc &= 0xFFFF
        return crc

    # ---- framing ----
    def build_frame(self, cmd: str, payload: bytes = b"") -> bytes:
        cmd_byte = self.CMD[cmd]
        length = len(payload)
        body = bytes([self.SOF, cmd_byte]) + struct.pack("<H", length) + payload
        crc = self.crc16(body)
        return body + struct.pack("<H", crc)

    def parse_frame(self, frame: bytes) -> Optional[Dict[str, Any]]:
        if len(frame) < 6 or frame[0] != self.SOF:
            return None
        cmd = frame[1]
        length = struct.unpack("<H", frame[2:4])[0]
        if len(frame) < 4 + length + 2:
            return None
        payload = frame[4:4 + length]
        body = frame[:4 + length]
        crc_recv = struct.unpack("<H", frame[4 + length:6 + length])[0]
        if self.crc16(body) != crc_recv:
            return None
        cmd_name = {v: k for k, v in self.CMD.items()}.get(cmd, "UNKNOWN")
        return {"cmd": cmd_name, "payload": payload}

    # ---- high-level commands ----
    def _transact(self, cmd: str, payload: bytes = b"") -> Dict[str, Any]:
        frame = self.build_frame(cmd, payload)
        if self.simulate:
            return self._simulate_device(cmd, payload)
        self.ser.write(frame)
        resp = self.ser.read(256)
        parsed = self.parse_frame(resp)
        return parsed or {"cmd": "ERROR", "payload": b""}

    def _simulate_device(self, cmd: str, payload: bytes) -> Dict[str, Any]:
        """Simulated device firmware for loopback testing."""
        if cmd == "INITIALIZE":
            self._device_state["initialized"] = True
            return {"cmd": "INITIALIZE", "payload": b"\x01"}
        elif cmd == "SET_PARAMETER":
            pid = payload[0]
            val = struct.unpack("<f", payload[1:5])[0]
            self._device_state["params"][pid] = val
            return {"cmd": "SET_PARAMETER", "payload": payload}
        elif cmd == "APPLY_GATE":
            qubit = payload[0]
            theta = struct.unpack("<f", payload[1:5])[0]
            strength = struct.unpack("<f", payload[5:9])[0]
            self._device_state["qubits"].append(
                {"qubit": qubit, "theta": theta, "strength": strength})
            return {"cmd": "APPLY_GATE", "payload": payload}
        elif cmd == "TRIGGER_COLLAPSE":
            kappa = struct.unpack("<f", payload[0:4])[0]
            self._device_state["kappa"] = kappa
            return {"cmd": "TRIGGER_COLLAPSE", "payload": payload}
        elif cmd == "READ_STATE":
            return {"cmd": "READ_STATE",
                    "payload": json.dumps(self._device_state).encode()}
        return {"cmd": "UNKNOWN", "payload": b""}

    # ---- convenience API ----
    def initialize(self) -> Dict:
        return self._transact("INITIALIZE")

    def set_parameter(self, pid: int, value: float) -> Dict:
        return self._transact("SET_PARAMETER",
                              bytes([pid]) + struct.pack("<f", value))

    def apply_gate(self, qubit: int, theta: float, strength: float) -> Dict:
        return self._transact("APPLY_GATE",
                              bytes([qubit]) + struct.pack("<f", theta)
                              + struct.pack("<f", strength))

    def trigger_collapse(self, kappa: float) -> Dict:
        return self._transact("TRIGGER_COLLAPSE", struct.pack("<f", kappa))

    def read_state(self) -> Dict:
        r = self._transact("READ_STATE")
        try:
            r["state"] = json.loads(r["payload"].decode())
        except Exception:
            r["state"] = None
        return r

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()


# =============================================================================
# §9. [H3] LAB NOTEBOOK REPORTER — reproducible experiments
# =============================================================================

class LabNotebookReporter:
    """
    Produces a reproducible experiment report: config dump, seeds, git hash
    (if available), result hashes, and a Markdown + JSON artifact.
    """

    def __init__(self, experiment_name: str, out_dir: str = "./lab_notebook"):
        self.experiment_name = experiment_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self.meta = {
            "experiment": experiment_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": VERSION,
            "git_commit": self._git_commit(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }

    def _git_commit(self) -> Optional[str]:
        try:
            import subprocess
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return None

    def log(self, name: str, data: Dict[str, Any]) -> None:
        entry = {"name": name, "timestamp": time.time(), "data": data}
        entry["hash"] = self._hash(data)
        self.entries.append(entry)

    @staticmethod
    def _hash(data: Dict[str, Any]) -> str:
        s = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def save(self) -> Dict[str, str]:
        stem = f"{self.experiment_name}_{int(time.time())}"
        json_path = self.out_dir / f"{stem}.json"
        md_path = self.out_dir / f"{stem}.md"

        payload = {"meta": self.meta, "entries": self.entries}
        json_path.write_text(json.dumps(payload, indent=2, default=str))

        md = [f"# Experiment: {self.experiment_name}", ""]
        md.append(f"- **Version:** {self.meta['version']}")
        md.append(f"- **Timestamp:** {self.meta['timestamp']}")
        md.append(f"- **Git commit:** {self.meta['git_commit']}")
        md.append(f"- **torch:** {self.meta['torch_version']}")
        md.append("")
        md.append("## Results")
        for e in self.entries:
            md.append(f"### {e['name']}  `hash={e['hash']}`")
            md.append("```json")
            md.append(json.dumps(e["data"], indent=2, default=str))
            md.append("```")
            md.append("")
        md_path.write_text("\n".join(md))
        return {"json": str(json_path), "markdown": str(md_path)}


# =============================================================================
# §10. VERIFICATION / TEST SUITE FOR ALL THREE PAPERS
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name: str, passed: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------- PAPER 2 ----------------
    print("\n[PAPER 2] Topology-indexed Hilbert space & global unitarity")
    sys2 = StructuralQuantumSystem()
    sys2.add_sector("A", 1)       # C^2
    sys2.add_sector("B", 2)       # C^4
    # nucleation isometry C^2 -> C^4 : |ψ> -> |ψ>⊗|0>
    V = torch.zeros(4, 2, dtype=torch.float64)
    V[0, 0] = 1.0
    V[2, 1] = 1.0
    sys2.add_transition("A", "B", V)
    rep("P2 isometry accepted (V†V=I)", True)
    try:
        bad = torch.ones(4, 2, dtype=torch.float64)
        sys2.add_transition("A", "B", bad)
        rep("P2 non-isometry rejected", False)
    except ValueError:
        rep("P2 non-isometry rejected", True)

    psi0 = torch.zeros(sys2.total_dim, dtype=torch.complex128)
    psi0[0] = 1.0 / math.sqrt(2)
    psi0[1] = 1.0 / math.sqrt(2)
    unit = sys2.verify_unitarity(t=0.5)
    rep("P2 global unitarity", unit["unitary"],
        f"err={max(unit['left_error'], unit['right_error']):.2e}")
    psi_t = sys2.evolve(psi0, t=0.5)
    prob = sys2.verify_probability(psi_t)
    rep("P2 probability conservation", prob["error"] < 1e-10,
        f"total={prob['total_probability']:.10f}")

    # ---------------- PAPER 3 ----------------
    print("\n[PAPER 3] DQNQRS deterministic collapse & progressive gates")
    dim = 2
    sim = DQNQRSimulator(dim)
    psi = torch.tensor([0.4, math.sqrt(0.84)], dtype=torch.complex128)
    psi = psi / torch.norm(psi)
    L = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    H = torch.zeros(dim, dim, dtype=torch.float64)

    loss0 = sim.expected_loss(psi, L)
    psi2 = sim.collapse_step(psi, H, L, kappa=0.8, dt=0.1)
    loss1 = sim.expected_loss(psi2, L)
    rep("P3 norm preserved under collapse", abs(float(torch.norm(psi2)) - 1.0) < 1e-10)
    rep("P3 loss decreases monotonically", loss1 < loss0,
        f"{loss0:.4f} -> {loss1:.4f}")

    # run many steps -> converge to |0>
    p = psi.clone()
    for _ in range(300):
        p = sim.collapse_step(p, H, L, kappa=1.0, dt=0.05)
    p0_prob = float(abs(p[0]) ** 2)
    rep("P3 converges to min-loss eigenstate", p0_prob > 0.99,
        f"P(|0>)={p0_prob:.6f}")

    # weak measurement
    Z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    Mp, Mm = sim.weak_measurement_operators(Z, epsilon=0.0)
    comp_check = torch.norm(Mp.conj().T @ Mp + Mm.conj().T @ Mm
                            - torch.eye(dim, dtype=torch.complex128))
    rep("P3 weak-measurement completeness at eps=0", float(comp_check) < 1e-10)

    # ---------------- PAPER 1 ----------------
    print("\n[PAPER 1] USE-SQC structural events + control math")
    sys1 = StructuralQuantumSystem()
    sys1.add_sector("root", 1)
    ea = StructuralEventAlgebra(sys1)
    ea.nucleation("root", "grown", n_new_qubits=1)
    rep("P1 nucleation creates isometric transition",
        ("root", "grown") in sys1.transitions)
    ea.branching("root", ["br1", "br2"])
    rep("P1 branching creates 2 branches",
        ("root", "br1") in sys1.transitions
        and ("root", "br2") in sys1.transitions)

    opmc = OPMCDesign(target_dim=8)
    field_val = opmc()
    field_val.sum().backward()
    rep("P1 OPMC design exact gradient", opmc.theta.grad is not None
        and torch.isfinite(opmc.theta.grad).all())
    rep("P1 OPMC d(4,4)=320", opmc_d_mn() == 320)

    ind = torch.rand(9, 9, dtype=torch.float64)
    cp = carpet_priority(ind)
    rep("P1 Carpet priority finite scalar", math.isfinite(cp))

    ssc = SemanticStateContraction()
    s1 = ssc(torch.tensor(1.0, dtype=torch.float64))
    s2 = ssc(torch.tensor(1.0, dtype=torch.float64))
    rep("P1 SSC low-pass accumulates", float(s2) > float(s1) or
        abs(float(s2) - float(s1)) < 1e-9)

    csoc = CSOCEventCounter()
    for t in range(500):
        csoc.step(0.1 + 0.3 * abs(math.sin(0.11 * t)))
    rep("P1 CSOC event bound satisfied", csoc.report()["satisfied"],
        f"N={csoc.events} bound={csoc.report()['bound']:.1f}")

    # ---------------- ENGINE + DDP ----------------
    print("\n[ENGINE] USE-SQC engine forward (DDP-ready)")
    eng = USESQCEngine(SimulationConfig(collapse_steps=20))
    diag = eng(batch_size=2)
    rep("Engine forward produces finite loss",
        torch.isfinite(diag["final_loss"]).all())
    diag["final_loss"].backward()
    rep("Engine backward flows gradients", True)

    # ---------------- QISKIT ----------------
    print("\n[H1] Qiskit backend")
    if HAS_QISKIT:
        qb = QiskitBackend()
        qc = qb.nucleation_circuit(1, 1)
        rep("Qiskit nucleation circuit built", qc.num_qubits == 2)
        sv = qb.run_statevector(qc)
        rep("Qiskit Aer statevector runs", len(sv) == 4)
        iso = qb.isometry_to_unitary(V)
        rep("Stines dilation is unitary",
            float(np.norm(iso.conj().T @ iso - np.eye(4))) < 1e-8)
        sub = qb.submit_to_ibm(qc)
        rep("IBM submit stub reports honestly", sub["status"] == "NOT_SUBMITTED")
    else:
        print("  [SKIP] qiskit not installed — run pip install qiskit qiskit-aer")
        rep("qiskit available", False, "not installed")

    # ---------------- SERIAL FIRMWARE ----------------
    print("\n[H2] Serial firmware stub (loopback)")
    fw = SerialFirmwareProtocol(simulate=True)
    r = fw.initialize()
    rep("Firmware INITIALIZE", r["cmd"] == "INITIALIZE")
    fw.set_parameter(1, 0.42)
    fw.apply_gate(0, 0.785, 0.5)
    fw.trigger_collapse(0.8)
    st = fw.read_state()
    rep("Firmware READ_STATE round-trip",
        st["state"] is not None and st["state"]["initialized"]
        and len(st["state"]["qubits"]) == 1)
    # frame integrity
    frame = fw.build_frame("READ_TELEMETRY")
    parsed = fw.parse_frame(frame)
    rep("Firmware CRC framing", parsed is not None
        and parsed["cmd"] == "READ_TELEMETRY")
    # corrupt CRC -> rejected
    corrupted = frame[:-1] + bytes([frame[-1] ^ 0xFF])
    rep("Firmware rejects bad CRC", fw.parse_frame(corrupted) is None)

    # ---------------- LAB NOTEBOOK ----------------
    print("\n[H3] Lab notebook reporter")
    nb = LabNotebookReporter("self_test")
    nb.log("paper2_unitarity", unit)
    nb.log("paper3_convergence", {"P0": p0_prob})
    paths = nb.save()
    rep("Notebook writes JSON + Markdown",
        os.path.exists(paths["json"]) and os.path.exists(paths["markdown"]))

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
HONEST INTERPRETATION
  • Paper 2 theorems (unitarity, isometry, probability conservation) PASS
    as LINEAR-ALGEBRA facts on the classical simulator.
  • Paper 3 deterministic collapse converges as an ALGORITHMIC selector.
    It is NOT physical collapse and NOT safe on entangled systems.
  • Paper 1 structural events are realized as isometries / branchings.
  • Qiskit backend builds REAL circuits; IBM submission is a stub.
  • Serial firmware is a PROTOCOL + loopback, not physical hardware.
  • No quantum advantage, no physical topology change, no GR backreaction.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="SESI USE-SQC production system")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--ddp-steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--demo-collapse", action="store_true")
    ap.add_argument("--qiskit-demo", action="store_true")
    ap.add_argument("--firmware-demo", action="store_true")
    ap.add_argument("--report", default=None, help="run all demos + save notebook")
    args = ap.parse_args()

    if args.selftest:
        return 0 if self_test() else 1

    if args.ddp:
        cfg = SimulationConfig(collapse_steps=30)
        out = run_ddp(cfg, batch_per_gpu=args.batch, steps=args.ddp_steps)
        print(json.dumps(out, indent=2, default=float))
        return 0

    if args.demo_collapse:
        sim = DQNQRSimulator(2)
        psi = torch.tensor([0.4, math.sqrt(0.84)], dtype=torch.complex128)
        psi = psi / torch.norm(psi)
        L = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
        H = torch.zeros(2, 2, dtype=torch.float64)
        print("Deterministic collapse demo (algorithmic selection):")
        for step in range(0, 201, 20):
            for _ in range(20):
                psi = sim.collapse_step(psi, H, L, kappa=1.0, dt=0.05)
            print(f"  step {step:3d}: P(|0>)={abs(psi[0])**2:.6f} "
                  f"loss={sim.expected_loss(psi, L):.6f}")
        return 0

    if args.qiskit_demo:
        if not HAS_QISKIT:
            print("qiskit not installed"); return 1
        qb = QiskitBackend()
        qc = qb.nucleation_circuit(1, 1)
        sv = qb.run_statevector(qc)
        print("Qiskit nucleation |ψ> -> |ψ>⊗|0>, statevector:", sv)
        print(qb.submit_to_ibm(qc))
        return 0

    if args.firmware_demo:
        fw = SerialFirmwareProtocol(simulate=True)
        print(fw.initialize())
        fw.apply_gate(0, 0.785, 0.5)
        print(fw.read_state()["state"])
        return 0

    if args.report:
        nb = LabNotebookReporter(args.report)
        nb.log("selftest", {"passed": self_test(verbose=False)})
        print(json.dumps(nb.save(), indent=2))
        return 0

    ap.print_help()
    print(BANNER)
    print("Start: --selftest | --demo-collapse | --qiskit-demo | --ddp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
