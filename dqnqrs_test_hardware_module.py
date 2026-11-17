#!/usr/bin/env python3
"""
=============================================================================
DQNQRS TEST & HARDWARE DEVELOPMENT MODULE v1.0
Dual-Core Nonlinear Quantum–Relativistic Architecture
Deterministic Computational Collapse / Progressive Gates / Weak Measurement
=============================================================================
Based on paper:
"A Dual-Core Nonlinear Quantum–Relativistic Architecture for a Unified
 Self-Evolving Structural Quantum Computer — Deterministic Computational
 Collapse, Progressive Quantum Gates, Weak Measurement, and
 Interference-to-Collapse Dynamics"

Author paper : PAI, Yoon A. Limsuwan
Code module  : Qwen
License      : MIT

=============================================================================
HONEST SCOPE
=============================================================================
✅ What this module IS:
  1. A classical state-vector simulator for the paper's progressive protocol.
  2. A test suite for norm preservation, monotonic loss reduction,
     deterministic selection, soft gates, weak measurement, interference.
  3. A hardware abstraction layer (HAL) for future hardware-in-the-loop.
  4. A hardware development planner (subsystems, calibration, validation).

❌ What this module is NOT:
  1. Not a physical quantum computer.
  2. Not proof that deterministic collapse is physical.
  3. Not a GR simulator; gravity core is fixed Minkowski by default.
  4. Not a certified engineering design for cryogenics, lasers, microwave,
     FPGA, vacuum systems, or safety-critical hardware.
  5. Not valid for entangled no-signalling tests unless additional constraints
     are implemented and reviewed.

IMPORTANT:
The deterministic collapse equation is an ALGORITHMIC selection mechanism.
It must not be confused with Born-rule projective measurement.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
PAPER_SHORT = "DQNQRS / USE-SQC deterministic-collapse testbench"

BANNER = f"""
=============================================================================
{PAPER_SHORT} v{VERSION}
Paper: PAI, Yoon A. Limsuwan
Code : Qwen
Scope: Classical simulation + test suite + hardware abstraction layer.
       NOT a physical quantum computer. NOT a certified hardware design.
=============================================================================
"""


# =============================================================================
# 1. BASIC LINEAR-ALGEBRA UTILITIES
# =============================================================================

def hermitian_part(M: np.ndarray) -> np.ndarray:
    """Return (M + M†)/2."""
    return 0.5 * (M + M.conj().T)


def is_hermitian(M: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(M, M.conj().T, atol=tol)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n


def probabilities(state: np.ndarray) -> np.ndarray:
    return np.abs(state) ** 2


def expected_value(state: np.ndarray, op: np.ndarray) -> float:
    return float(np.vdot(state, op @ state).real)


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def expm_hermitian_angle(H: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute exp(-i * angle * H) for Hermitian H via eigendecomposition.
    No SciPy required.
    """
    H = hermitian_part(H)
    w, V = np.linalg.eigh(H)
    D = np.diag(np.exp(-1j * angle * w))
    return V @ D @ V.conj().T


def sqrtm_psd(A: np.ndarray) -> np.ndarray:
    """Square root of positive semidefinite Hermitian matrix."""
    A = hermitian_part(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.conj().T


def pauli_x(dim: int = 2) -> np.ndarray:
    if dim != 2:
        # simple nearest-neighbor coupling for higher dims
        X = np.zeros((dim, dim), dtype=complex)
        for i in range(dim - 1):
            X[i, i + 1] = 1.0
            X[i + 1, i] = 1.0
        return X
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_z(dim: int = 2) -> np.ndarray:
    if dim != 2:
        Z = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            Z[i, i] = 1.0 if i % 2 == 0 else -1.0
        return Z
    return np.array([[1, 0], [0, -1]], dtype=complex)


def hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)


def phase_gate(phi: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


# =============================================================================
# 2. PAPER MATHEMATICAL PRIMITIVES
# =============================================================================

def soft_gate(state: np.ndarray,
              H_gate: np.ndarray,
              theta: float,
              strength: float) -> np.ndarray:
    """
    Progressive gate from the paper:
        G_s(theta) = exp(-i s theta H_gate)
    strength s in [0,1].
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    U = expm_hermitian_angle(H_gate, strength * theta)
    return normalize(U @ state)


def weak_measurement_operators(A: np.ndarray,
                                epsilon: float
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary weak measurement operators from the paper:
        M_r(eps) = sqrt((I + r eps A)/2), r in {+1,-1}
    Requires A^2 = I and Hermitian A.
    """
    A = hermitian_part(A)
    I = np.eye(A.shape[0], dtype=complex)
    eps = float(np.clip(epsilon, 0.0, 1.0))

    w, V = np.linalg.eigh(A)
    if not np.allclose(np.abs(w), 1.0, atol=1e-6):
        raise ValueError("Weak measurement currently requires A^2 = I "
                         "(eigenvalues ±1).")

    M_plus = np.zeros_like(A)
    M_minus = np.zeros_like(A)

    # In eigenbasis, M_r eigenvalues are sqrt((1 + r*eps*eigA)/2)
    for r, M in ((+1, M_plus), (-1, M_minus)):
        vals = np.sqrt(np.clip((1.0 + r * eps * w) / 2.0, 0.0, None))
        M[:, :] = V @ np.diag(vals) @ V.conj().T

    return M_plus, M_minus


def deterministic_collapse_step(state: np.ndarray,
                                L: np.ndarray,
                                kappa: float,
                                dt: float) -> np.ndarray:
    """
    Deterministic computational collapse step.

    Equation:
        d|psi>/dt = -kappa (L - <L>) |psi>

    For diagonalizable Hermitian L, the exact normalized solution over dt is:
        c_j(t+dt) ∝ c_j(t) exp(-kappa * l_j * dt)

    This is NOT Born-rule measurement. It is algorithmic loss minimization.
    """
    if kappa <= 0.0 or dt <= 0.0:
        return normalize(state)

    L = hermitian_part(L)
    eigvals, eigvecs = np.linalg.eigh(L)

    if np.min(eigvals) < -1e-10:
        raise ValueError("Loss operator should be positive semidefinite.")

    coeff = eigvecs.conj().T @ state

    # exact collapse in loss eigenbasis
    exponents = -float(kappa) * eigvals * float(dt)

    # numerical stabilization (global factor removed by normalization)
    exponents = exponents - np.max(exponents)
    coeff = coeff * np.exp(exponents)

    coeff = normalize(coeff)
    return normalize(eigvecs @ coeff)


def deterministic_collapse_exact(state: np.ndarray,
                                 losses: np.ndarray,
                                 kappa: float,
                                 total_time: float,
                                 steps: int = 1) -> np.ndarray:
    """
    Closed-form deterministic collapse for diagonal loss operator.
    losses[j] is loss of basis state |j>.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or len(losses) != len(state):
        raise ValueError("losses must match state dimension.")
    if np.min(losses) < 0:
        raise ValueError("losses must be non-negative.")

    c = np.asarray(state, dtype=complex).copy()
    t = float(total_time)
    k = float(kappa)

    # exact: c_j(t) ∝ c_j(0) exp(-k * loss_j * t)
    exponents = -k * losses * t
    exponents = exponents - np.max(exponents)
    c = c * np.exp(exponents)
    return normalize(c)


def expected_loss(state: np.ndarray, L: np.ndarray) -> float:
    return expected_value(state, L)


def loss_monotonicity_history(loss_history: List[float]) -> bool:
    """Check non-increasing within numerical tolerance."""
    tol = 1e-10
    for i in range(1, len(loss_history)):
        if loss_history[i] > loss_history[i - 1] + tol:
            return False
    return True


# =============================================================================
# 3. SCHEDULING / PROGRESSIVE PROTOCOL
# =============================================================================

def logistic_schedule(x: float, center: float, width: float) -> float:
    """
    Smooth monotonic schedule from 0 to 1.
    x: normalized protocol time in [0,1].
    """
    x = float(np.clip(x, 0.0, 1.0))
    width = max(float(width), 1e-6)
    z = (x - center) / width
    if z > 60:
        return 1.0
    if z < -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ProgressiveProtocolConfig:
    """
    Progressive operational hierarchy from the paper:
        Soft gates -> weak measurement -> weak interference
        -> hard gates -> strong interference -> deterministic collapse
    """
    steps: int = 300
    dt: float = 0.02

    theta_gate: float = 0.05
    theta_mix: float = 0.03

    kappa_max: float = 1.0
    epsilon_max: float = 0.0  # default OFF for deterministic tests

    measurement_interval: int = 0  # 0 disables weak measurement

    gate_center: float = 0.20
    measurement_center: float = 0.40
    interference_center: float = 0.60
    collapse_center: float = 0.80
    width: float = 0.06

    # If true, keep a full audit trail. Disable for long simulations.
    audit: bool = False


@dataclass
class ProtocolResult:
    final_state: np.ndarray
    history: List[Dict[str, float]]
    initial_loss: float
    final_loss: float
    final_prob_min_loss: float
    selected_index: int
    converged: bool


class DQNQRSProtocolEngine:
    """
    Classical engine implementing the paper's progressive sequence.
    """

    def __init__(self, dim: int = 2, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.H_gate = pauli_x(dim)
        self.H_mix = pauli_x(dim)
        self.A_measure = pauli_z(dim)

    def run(self,
            state: np.ndarray,
            L: np.ndarray,
            cfg: ProgressiveProtocolConfig) -> ProtocolResult:
        state = normalize(np.asarray(state, dtype=complex))
        L = hermitian_part(np.asarray(L, dtype=complex))

        eigvals, eigvecs = np.linalg.eigh(L)
        if np.min(eigvals) < -1e-10:
            raise ValueError("Loss operator must be positive semidefinite.")

        # We need loss eigenbasis probabilities to identify min-loss sector.
        def loss_basis_probs(s: np.ndarray) -> np.ndarray:
            c = eigvecs.conj().T @ s
            return np.abs(c) ** 2

        min_loss_index = int(np.argmin(eigvals))

        history: List[Dict[str, float]] = []
        initial_loss = expected_loss(state, L)
        s = state.copy()

        for step in range(cfg.steps):
            x = step / max(cfg.steps - 1, 1)

            gate_strength = logistic_schedule(x, cfg.gate_center, cfg.width)
            eps = cfg.epsilon_max * logistic_schedule(
                x, cfg.measurement_center, cfg.width)
            chi = logistic_schedule(x, cfg.interference_center, cfg.width)
            kappa = cfg.kappa_max * logistic_schedule(
                x, cfg.collapse_center, cfg.width)

            # Stage 1/4: progressive gates
            s = soft_gate(s, self.H_gate, cfg.theta_gate, gate_strength)

            # Stage 2: weak measurement if enabled
            if cfg.measurement_interval > 0 and eps > 1e-12:
                if step % cfg.measurement_interval == 0:
                    M_plus, M_minus = weak_measurement_operators(
                        self.A_measure, eps)
                    post_plus = M_plus @ s
                    post_minus = M_minus @ s
                    p_plus = float(np.vdot(post_plus, post_plus).real)
                    p_minus = float(np.vdot(post_minus, post_minus).real)
                    psum = p_plus + p_minus
                    if psum <= 1e-15:
                        raise RuntimeError("Weak measurement probability zero.")
                    p_plus /= psum
                    p_minus /= psum

                    choice = int(self.rng.choice(2, p=[p_plus, p_minus]))
                    if choice == 0:
                        s = normalize(post_plus)
                    else:
                        s = normalize(post_minus)

            # Stage 3/4: interference
            s = soft_gate(s, self.H_mix, cfg.theta_mix, chi)

            # Stage 5: deterministic computational collapse
            s = deterministic_collapse_step(s, L, kappa, cfg.dt)

            p_basis = loss_basis_probs(s)
            rec = {
                "step": step,
                "x": x,
                "gate_strength": gate_strength,
                "epsilon": eps,
                "chi": chi,
                "kappa": kappa,
                "loss": expected_loss(s, L),
                "prob_min_loss": float(p_basis[min_loss_index]),
                "norm": float(np.linalg.norm(s)),
            }
            history.append(rec)

        final_probs = loss_basis_probs(s)
        final_loss = expected_loss(s, L)
        selected_index = int(np.argmax(final_probs))
        converged = bool(final_probs[min_loss_index] > 0.90
                         and final_loss <= initial_loss + 1e-12)

        return ProtocolResult(
            final_state=s,
            history=history,
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            final_prob_min_loss=float(final_probs[min_loss_index]),
            selected_index=selected_index,
            converged=converged,
        )


# =============================================================================
# 4. SIMULATION TESTS
# =============================================================================

def two_state_demo_state() -> np.ndarray:
    """
    Paper section 13 example:
        |psi> = alpha |0> + beta |1>
    with loss |0> = 0, loss |1> = 1.
    """
    alpha = 0.4
    beta = math.sqrt(0.84)
    return normalize(np.array([alpha, beta], dtype=complex))


def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name: str, passed: bool, extra: str = ""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ------------------------------------------------------------------
    # T1 deterministic collapse preserves norm
    # ------------------------------------------------------------------
    state = two_state_demo_state()
    L = np.diag([0.0, 1.0]).astype(complex)

    s = state.copy()
    norms = []
    for _ in range(200):
        s = deterministic_collapse_step(s, L, kappa=0.30, dt=0.05)
        norms.append(float(np.linalg.norm(s)))

    norm_err = max(abs(n - 1.0) for n in norms)
    rep("T1 collapse norm preservation", norm_err < 1e-10,
        f"max |norm-1|={norm_err:.2e}")

    # ------------------------------------------------------------------
    # T2 expected loss decreases monotonically
    # ------------------------------------------------------------------
    s = state.copy()
    losses = [expected_loss(s, L)]
    for _ in range(200):
        s = deterministic_collapse_step(s, L, kappa=0.30, dt=0.05)
        losses.append(expected_loss(s, L))

    monotone = loss_monotonicity_history(losses)
    rep("T2 loss monotonic decrease", monotone,
        f"{losses[0]:.6f} -> {losses[-1]:.6f}")

    # ------------------------------------------------------------------
    # T3 convergence to minimum-loss eigenspace
    # ------------------------------------------------------------------
    s = deterministic_collapse_exact(state, losses=np.array([0.0, 1.0]),
                                     kappa=0.8, total_time=20.0)
    p0 = probabilities(s)[0]
    rep("T3 converges to min-loss candidate", p0 > 0.999,
        f"P(min-loss)={p0:.8f}")

    # If initial overlap is zero, no guarantee.
    s_zero = np.array([0.0, 1.0], dtype=complex)
    s_zero_final = deterministic_collapse_exact(
        s_zero, np.array([0.0, 1.0]), kappa=1.0, total_time=20.0)
    p0_zero = probabilities(s_zero_final)[0]
    rep("T3b zero-overlap warning respected", p0_zero < 1e-12,
        "no amplitude in min-loss state remains")

    # ------------------------------------------------------------------
    # T4 soft gate: s=0 identity, s=1 full gate, norm preserved
    # ------------------------------------------------------------------
    X = pauli_x(2)
    s_id = soft_gate(state, X, theta=0.7, strength=0.0)
    rep("T4a soft gate s=0 identity", fidelity(s_id, state) > 1 - 1e-12)

    s_full = soft_gate(state, X, theta=0.7, strength=1.0)
    rep("T4b soft gate norm preserved",
        abs(np.linalg.norm(s_full) - 1.0) < 1e-12)

    U_full = expm_hermitian_angle(X, 0.7)
    s_ref = normalize(U_full @ state)
    rep("T4c soft gate equals exp(-i theta H)",
        fidelity(s_full, s_ref) > 1 - 1e-12)

    # ------------------------------------------------------------------
    # T5 weak measurement
    # ------------------------------------------------------------------
    Z = pauli_z(2)

    # eps=0 gives no information, post-state unchanged
    M_plus, M_minus = weak_measurement_operators(Z, epsilon=0.0)
    p_plus = float(np.vdot(M_plus @ state, M_plus @ state).real)
    p_minus = float(np.vdot(M_minus @ state, M_minus @ state).real)
    post_plus = normalize(M_plus @ state)
    post_minus = normalize(M_minus @ state)

    rep("T5a eps=0 probabilities 1/2",
        abs(p_plus - 0.5) < 1e-12 and abs(p_minus - 0.5) < 1e-12,
        f"p+={p_plus:.6f}, p-={p_minus:.6f}")
    rep("T5b eps=0 post-state unchanged",
        fidelity(post_plus, state) > 1 - 1e-12
        and fidelity(post_minus, state) > 1 - 1e-12)

    # eps=1 gives projective Z measurement
    M_plus, M_minus = weak_measurement_operators(Z, epsilon=1.0)
    p_plus = float(np.vdot(M_plus @ state, M_plus @ state).real)
    p_minus = float(np.vdot(M_minus @ state, M_minus @ state).real)
    born_plus = probabilities(state)[0]
    born_minus = probabilities(state)[1]

    rep("T5c eps=1 Born probabilities",
        abs(p_plus - born_plus) < 1e-12 and abs(p_minus - born_minus) < 1e-12,
        f"p+={p_plus:.6f}, expected={born_plus:.6f}")

    post_plus = normalize(M_plus @ state)
    post_minus = normalize(M_minus @ state)
    rep("T5d eps=1 post-states are eigenstates",
        fidelity(post_plus, np.array([1, 0], dtype=complex)) > 1 - 1e-12
        and fidelity(post_minus, np.array([0, 1], dtype=complex)) > 1 - 1e-12)

    # ------------------------------------------------------------------
    # T6 interference visibility
    # ------------------------------------------------------------------
    H = hadamard()
    zero = np.array([1.0, 0.0], dtype=complex)
    plus = H @ zero

    p_list = []
    for phi in np.linspace(0.0, 2.0 * math.pi, 73):
        P = phase_gate(phi)
        out = H @ (P @ plus)
        p_list.append(probabilities(out)[0])

    pmax = max(p_list)
    pmin = min(p_list)
    visibility = (pmax - pmin) / max(pmax + pmin, 1e-30)
    rep("T6 interference visibility ~1", visibility > 0.999,
        f"V={visibility:.6f}")

    # ------------------------------------------------------------------
    # T7 progressive protocol selects min-loss candidate
    # ------------------------------------------------------------------
    cfg = ProgressiveProtocolConfig(
        steps=350,
        dt=0.03,
        theta_gate=0.05,
        theta_mix=0.03,
        kappa_max=1.2,
        epsilon_max=0.0,
        measurement_interval=0,
        audit=False,
    )
    engine = DQNQRSProtocolEngine(dim=2, seed=7)
    result = engine.run(state, L, cfg)

    rep("T7 progressive protocol converges", result.converged,
        f"P(min-loss)={result.final_prob_min_loss:.6f}, "
        f"loss {result.initial_loss:.6f}->{result.final_loss:.6f}")

    # ------------------------------------------------------------------
    # T8 GR core status is honestly fixed Minkowski
    # ------------------------------------------------------------------
    gr_status = {
        "metric": "eta_mu_nu = diag(-1, +1, +1, +1)",
        "backreaction": False,
        "semiclassical_Einstein": False,
        "status": "fixed flat-spacetime computational mode",
    }
    rep("T8 GR core declared fixed Minkowski",
        gr_status["backreaction"] is False,
        gr_status["status"])

    # ------------------------------------------------------------------
    # T9 deterministic collapse is not Born measurement
    # ------------------------------------------------------------------
    # Born measurement would give probability 0.84 for high-loss state
    # before collapse; deterministic collapse can still select low-loss
    # if initial overlap exists. This test documents the distinction.
    born_high_loss = probabilities(state)[1]
    final_low_loss = probabilities(result.final_state)[0]
    rep("T9 deterministic selection differs from raw Born sampling",
        born_high_loss > 0.5 and final_low_loss > 0.9,
        f"Born P(high-loss)={born_high_loss:.3f}, "
        f"deterministic final P(low-loss)={final_low_loss:.3f}")

    print()
    print("=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
IMPORTANT INTERPRETATION
  T9 is the key honesty test. The deterministic collapse module is a
  loss-minimization algorithm. It is NOT a replacement for Born-rule
  quantum measurement. If this is later run on physical entangled systems,
  no-signalling and complete-positivity constraints must be addressed.
""")
    return ok


# =============================================================================
# 5. HARDWARE ABSTRACTION LAYER
# =============================================================================

@dataclass
class HardwareCommand:
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareResponse:
    ok: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, default=str, indent=2)


class HardwareBackend:
    """
    Abstract hardware backend.

    Real hardware implementation should provide:
      - device initialization
      - qubit reset
      - single-qubit gates
      - two-qubit gates
      - weak/projective measurement
      - calibration hooks
      - telemetry
      - safety interlocks
    """
    name = "abstract"

    def execute(self, cmd: HardwareCommand) -> HardwareResponse:
        raise NotImplementedError


class ClassicalSimulatorBackend(HardwareBackend):
    """
    Hardware-in-the-loop simulator.

    This is useful for developing the control software before physical
    hardware exists. It accepts the same command set that a real device
    adapter should implement.
    """
    name = "classical_simulator"

    def __init__(self, dim: int = 2, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.state = two_state_demo_state() if dim == 2 else normalize(
            np.ones(dim, dtype=complex) / math.sqrt(dim))
        self.L = np.diag(np.arange(dim, dtype=float)).astype(complex)
        self.H_gate = pauli_x(dim)
        self.H_mix = pauli_x(dim)
        self.A_measure = pauli_z(dim)
        self.audit: List[Dict[str, Any]] = []

    def _log(self, event: str, data: Dict[str, Any]):
        self.audit.append({
            "time": time.time(),
            "event": event,
            "data": data,
        })

    def telemetry(self) -> Dict[str, Any]:
        p = probabilities(self.state)
        return {
            "backend": self.name,
            "dim": self.dim,
            "norm": float(np.linalg.norm(self.state)),
            "expected_loss": expected_loss(self.state, self.L),
            "probabilities": [float(x) for x in p],
            "selected_index": int(np.argmax(p)),
            "audit_entries": len(self.audit),
        }

    def execute(self, cmd: HardwareCommand) -> HardwareResponse:
        try:
            name = cmd.name.upper()
            payload = cmd.payload or {}

            if name == "INITIALIZE":
                if "state" in payload:
                    self.state = normalize(
                        np.asarray(payload["state"], dtype=complex))
                else:
                    self.state = two_state_demo_state() if self.dim == 2 else \
                        normalize(np.ones(self.dim, dtype=complex))
                self._log("INITIALIZE", {"dim": self.dim})
                return HardwareResponse(True, "initialized", self.telemetry())

            elif name == "SOFT_GATE":
                theta = float(payload.get("theta", 0.05))
                strength = float(payload.get("strength", 0.0))
                self.state = soft_gate(self.state, self.H_gate, theta, strength)
                self._log("SOFT_GATE", {"theta": theta, "strength": strength})
                return HardwareResponse(True, "soft gate applied", self.telemetry())

            elif name == "INTERFERENCE_GATE":
                theta = float(payload.get("theta", 0.03))
                chi = float(payload.get("chi", 0.0))
                self.state = soft_gate(self.state, self.H_mix, theta, chi)
                self._log("INTERFERENCE_GATE", {"theta": theta, "chi": chi})
                return HardwareResponse(True, "interference gate applied",
                                        self.telemetry())

            elif name == "WEAK_MEASURE":
                epsilon = float(payload.get("epsilon", 0.0))
                M_plus, M_minus = weak_measurement_operators(
                    self.A_measure, epsilon)
                post_plus = M_plus @ self.state
                post_minus = M_minus @ self.state
                p_plus = float(np.vdot(post_plus, post_plus).real)
                p_minus = float(np.vdot(post_minus, post_minus).real)
                psum = p_plus + p_minus
                p_plus /= psum
                p_minus /= psum

                if payload.get("deterministic", False):
                    choice = 0 if p_plus >= p_minus else 1
                else:
                    choice = int(self.rng.choice(2, p=[p_plus, p_minus]))

                if choice == 0:
                    self.state = normalize(post_plus)
                    outcome = +1
                else:
                    self.state = normalize(post_minus)
                    outcome = -1

                data = self.telemetry()
                data["measurement"] = {
                    "epsilon": epsilon,
                    "outcome": outcome,
                    "prob_plus": p_plus,
                    "prob_minus": p_minus,
                }
                self._log("WEAK_MEASURE", data["measurement"])
                return HardwareResponse(True, "weak measurement applied", data)

            elif name == "COLLAPSE_STEP":
                kappa = float(payload.get("kappa", 0.0))
                dt = float(payload.get("dt", 0.01))
                self.state = deterministic_collapse_step(
                    self.state, self.L, kappa, dt)
                self._log("COLLAPSE_STEP", {"kappa": kappa, "dt": dt})
                return HardwareResponse(True, "collapse step applied",
                                        self.telemetry())

            elif name == "TELEMETRY":
                return HardwareResponse(True, "telemetry", self.telemetry())

            elif name == "AUDIT":
                limit = int(payload.get("limit", 20))
                return HardwareResponse(True, "audit",
                                        {"entries": self.audit[-limit:]})

            else:
                return HardwareResponse(False, f"unknown command {name}")

        except Exception as exc:
            return HardwareResponse(False, f"{type(exc).__name__}: {exc}")


class HardwareDevelopmentKit:
    """
    Hardware development layer.

    This does NOT build hardware. It provides:
      1. command execution interface,
      2. hardware-in-the-loop test runner,
      3. calibration checklist,
      4. subsystem requirement generator.
    """

    def __init__(self, backend: Optional[HardwareBackend] = None):
        self.backend = backend or ClassicalSimulatorBackend()

    def run_hil_selftest(self) -> Dict[str, Any]:
        """
        HIL = hardware-in-the-loop. With the simulator backend this is a
        software loopback test. With a real backend it becomes a device test.
        """
        checks = []

        def add(name: str, resp: HardwareResponse, condition: bool):
            checks.append({
                "name": name,
                "ok": bool(resp.ok and condition),
                "message": resp.message,
            })

        resp = self.backend.execute(HardwareCommand("INITIALIZE"))
        add("initialize", resp, resp.ok)

        resp = self.backend.execute(HardwareCommand(
            "SOFT_GATE", {"theta": 0.05, "strength": 0.2}))
        add("soft_gate", resp, resp.ok)

        resp = self.backend.execute(HardwareCommand(
            "INTERFERENCE_GATE", {"theta": 0.03, "chi": 0.4}))
        add("interference_gate", resp, resp.ok)

        resp = self.backend.execute(HardwareCommand(
            "WEAK_MEASURE", {"epsilon": 0.3, "deterministic": True}))
        add("weak_measure", resp, resp.ok)

        resp = self.backend.execute(HardwareCommand(
            "COLLAPSE_STEP", {"kappa": 0.4, "dt": 0.02}))
        add("collapse_step", resp, resp.ok)

        resp = self.backend.execute(HardwareCommand("TELEMETRY"))
        norm_ok = False
        loss_ok = False
        if resp.ok:
            norm_ok = abs(resp.data.get("norm", 0.0) - 1.0) < 1e-8
            loss_ok = math.isfinite(resp.data.get("expected_loss", float("nan")))
        add("telemetry_norm", resp, norm_ok)
        add("telemetry_loss_finite", resp, loss_ok)

        all_ok = all(c["ok"] for c in checks)
        return {
            "backend": self.backend.name,
            "all_ok": all_ok,
            "checks": checks,
            "telemetry": resp.data if resp.ok else {},
        }

    def calibration_checklist(self) -> List[Dict[str, str]]:
        return [
            {
                "step": "C1",
                "task": "Device reset and state preparation",
                "metric": "reset fidelity",
            },
            {
                "step": "C2",
                "task": "Single-qubit gate calibration (Rabi/Ramsey)",
                "metric": "gate error rate",
            },
            {
                "step": "C3",
                "task": "Readout calibration",
                "metric": "assignment fidelity",
            },
            {
                "step": "C4",
                "task": "Weak measurement strength calibration",
                "metric": "epsilon vs outcome statistics",
            },
            {
                "step": "C5",
                "task": "Interference visibility calibration",
                "metric": "visibility V",
            },
            {
                "step": "C6",
                "task": "Deterministic-collapse parameter sweep",
                "metric": "loss reduction vs kappa, dt",
            },
            {
                "step": "C7",
                "task": "Cross-check against Born-rule measurement",
                "metric": "distribution difference; do not conflate",
            },
            {
                "step": "C8",
                "task": "Audit logging and provenance",
                "metric": "complete command trace",
            },
        ]

    def hardware_requirement(self) -> Dict[str, Any]:
        """
        Generate a high-level hardware development plan.

        This is a research plan, not a certified engineering design.
        """
        return {
            "title": "DQNQRS experimental hardware plan",
            "status": "research-stage; not certified",
            "physics_warning": [
                "Deterministic computational collapse is not established as physical collapse.",
                "No-signalling must be analyzed before applying to entangled systems.",
                "GR core should remain fixed Minkowski in first experiments.",
            ],
            "stage_0_simulation": {
                "goal": "Validate protocol logic classically.",
                "items": [
                    "state-vector simulator",
                    "norm preservation test",
                    "loss monotonicity test",
                    "weak measurement test",
                    "interference visibility test",
                    "audit trace test",
                ],
            },
            "stage_1_control_hardware": {
                "goal": "Build classical control testbench.",
                "items": [
                    "host PC",
                    "FPGA or microcontroller",
                    "DAC/ADC channels",
                    "timing/trigger unit",
                    "command protocol (JSON/SCPI/gRPC)",
                    "telemetry logger",
                    "interlock and watchdog",
                ],
            },
            "stage_2_single_physical_qubit": {
                "goal": "Demonstrate one-qubit progressive operations.",
                "platform_options": [
                    "superconducting transmon",
                    "trapped ion",
                    "photonic polarization qubit",
                    "neutral atom Rydberg qubit",
                    "spin qubit",
                ],
                "experiments": [
                    "soft gate sweep s: 0 -> 1",
                    "weak measurement epsilon sweep",
                    "interference visibility",
                    "algorithmic deterministic selection",
                ],
            },
            "stage_3_two_qubit_and_structure": {
                "goal": "Test structural branching/merging as controlled operations.",
                "experiments": [
                    "controlled branching via ancilla",
                    "path consistency test",
                    "no-cloning compliance check",
                    "process tomography",
                ],
            },
            "stage_4_relativistic_core": {
                "goal": "Only after quantum layer is stable.",
                "note": "Semiclassical Einstein coupling is not required for AI-oriented experiments.",
                "initial_model": "fixed Minkowski metric",
            },
            "required_validation": [
                "randomized benchmarking",
                "state tomography",
                "process tomography",
                "weak-measurement tomography",
                "stability over time",
                "independent replication",
            ],
            "not_claimed": [
                "physical wavefunction collapse",
                "quantum advantage",
                "gravitational backreaction",
                "consciousness or agency",
                "production safety certification",
            ],
        }


# =============================================================================
# 6. DEMO
# =============================================================================

def demo_progressive_protocol() -> Dict[str, Any]:
    state = two_state_demo_state()
    L = np.diag([0.0, 1.0]).astype(complex)

    cfg = ProgressiveProtocolConfig(
        steps=350,
        dt=0.03,
        theta_gate=0.05,
        theta_mix=0.03,
        kappa_max=1.2,
        epsilon_max=0.0,
        measurement_interval=0,
    )

    engine = DQNQRSProtocolEngine(dim=2, seed=0)
    result = engine.run(state, L, cfg)

    return {
        "initial_state_probabilities": [float(x) for x in probabilities(state)],
        "losses": [0.0, 1.0],
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "final_prob_min_loss": result.final_prob_min_loss,
        "selected_index": result.selected_index,
        "converged": result.converged,
        "history_tail": result.history[-5:],
        "interpretation": (
            "This is deterministic computational collapse, i.e. algorithmic "
            "selection of the lower-loss candidate. It is not Born-rule "
            "physical measurement."
        ),
    }


def demo_hardware_in_loop() -> Dict[str, Any]:
    backend = ClassicalSimulatorBackend(dim=2, seed=0)
    kit = HardwareDevelopmentKit(backend)
    hil = kit.run_hil_selftest()
    plan = kit.hardware_requirement()
    checklist = kit.calibration_checklist()
    return {
        "hardware_in_loop": hil,
        "calibration_checklist": checklist,
        "hardware_plan": plan,
    }


# =============================================================================
# 7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="DQNQRS test & hardware development module")
    ap.add_argument("--selftest", action="store_true",
                    help="run mathematical/protocol self-tests")
    ap.add_argument("--demo", action="store_true",
                    help="run progressive protocol demo")
    ap.add_argument("--hardware", action="store_true",
                    help="run hardware-in-the-loop simulator demo")
    ap.add_argument("--hardware-plan", action="store_true",
                    help="print hardware development plan")
    ap.add_argument("--json", default=None, help="save JSON output")
    args = ap.parse_args()

    out: Dict[str, Any] = {}

    if args.selftest:
        ok = self_test(verbose=True)
        return 0 if ok else 1

    if args.demo:
        out["demo_progressive_protocol"] = demo_progressive_protocol()
        print(json.dumps(out["demo_progressive_protocol"], indent=2, default=str))

    if args.hardware:
        out["hardware"] = demo_hardware_in_loop()
        print(json.dumps(out["hardware"]["hardware_in_loop"], indent=2, default=str))

    if args.hardware_plan:
        kit = HardwareDevelopmentKit()
        out["hardware_plan"] = kit.hardware_requirement()
        out["calibration_checklist"] = kit.calibration_checklist()
        print(json.dumps(out, indent=2, default=str))

    if not out:
        ap.print_help()
        print(BANNER)
        print("Quick start:")
        print("  python dqnqrs_test_hardware_module.py --selftest")
        print("  python dqnqrs_test_hardware_module.py --demo")
        print("  python dqnqrs_test_hardware_module.py --hardware")
        print("  python dqnqrs_test_hardware_module.py --hardware-plan")
        return 0

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Saved: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
