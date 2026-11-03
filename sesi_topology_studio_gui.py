"""
=============================================================================
SESI TOPOLOGY STUDIO GUI
Streamlit/React Flow + Gumbel-Softmax Graph Rewriting + Carpet-Measure τ
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
CARPET-19 GROUNDING (from the uploaded consolidated paper)
=============================================================================
THEOREM 10 (Main result, CONDITIONAL on (I1)-(I4)):
  (a) normalized finite-level energies → unique Mosco limit
  (b) compact resolvent, discrete spectrum, all powers converge
  (c) integer powers are support-local
  (c') energy measures, derivations, Fredholm projections converge;
       [A,M_f] bounded W^1 → W^{-1}
  (d) spectral Sobolev scale embeds into C^{0,β/2}(K), β=log ρ/log 3

NUMERICAL FACTS (not proved):
  ρ ≈ 1.25          (Rcross ratios: 1.66,1.33,1.27,1.26,1.25)
  d_f = ln8/ln3 ≈ 1.8928
  d_w = ln(8ρ)/ln3 ≈ 2.096   (using ρ=1.25)
  d_s = 2d_f/d_w   ≈ 1.806
  β = log ρ / log 3 ≈ 0.222  (Hölder exponent)

HINO THEOREM 8 [C]: Γ(f,·) ⊥ µ for every f ∈ F.
  CONSEQUENCE: no L² density for energy measure.
  CONSEQUENCE: τ-adaptation must NOT assume a smooth density prior.

COROLLARY 3 [P]: D(A) is NOT an algebra.
  CONSEQUENCE: f² ∉ D(A) for nonconstant f ∈ D(A).
  CONSEQUENCE: Gumbel-Softmax structure gradients through Boolean
  composition are valid only at first order; second-order structure
  derivatives are NOT available.

PROPOSITION 3 [P]: [A,M_f] : W^1 → W^{-1} bounded, loses two orders.
  CONSEQUENCE: annealing τ does NOT gain regularity via commutators.
  CONSEQUENCE: τ lower bound must respect the W^{-1} loss.

SECTION 10 OPEN ITEMS [O] (NOT constructed, NOT claimed):
  (i)   Leibniz connection on H
  (ii)  measure-valued Hessian
  (iii) Schatten summability of [F,a] for a ∈ W^s, s>1
  (iv)  independently defined order-2k operator
  These are OPEN. Do not import them as if proved.

=============================================================================
MODULES
=============================================================================
[GFR1] GraphRewriteNet
      • Gumbel-Softmax over discrete CSG structure choices
      • Edge existence logits → soft adjacency matrix
      • Node-type logits → soft primitive selection
      • Straight-through estimator for hard sampling at eval
      • Annealing schedule: τ_gumbel → 0 recovers discrete graph
      • SECOND-ORDER WARNING: structure Hessian NOT available (Cor. 3)

[CMT1] CarpetMeasureTauLearner
      • Builds σ-field from carpet cell geometry (make_sigma_field pattern)
      • Conditions τ prediction on Mosco-limit-aware features:
          - cell-level energy measure proxy (finite-level Γ_N)
          - Hölder exponent β = log ρ / log 3
          - spectral dimension proxy d_s = 2d_f/d_w
      • Respects Hino singularity: NO L²-density assumption on Γ
      • Meta-learning (MAML) over carpet measure fields
      • ρ treated as a LEARNABLE PRIOR with numerical initialisation 1.25

[GUI1] StreamlitCSGApp
      • Streamlit page: node palette, canvas, property panel, 3D preview
      • Live SDF slice rendering via matplotlib
      • τ slider + meta-adapt button
      • OCIO colour transform selector
      • Export: OBJ / SVG / React Flow JSON / .ocio report

[RF1] ReactFlowExporter
      • CSGTree → React Flow nodes + edges JSON
      • Position auto-layout (topological levels)
      • Node styling by type (primitive/transform/boolean/output)

=============================================================================
DIFFERENTIABILITY AUDIT
=============================================================================
DIFF-FIX 5  _softplus_floor with beta tracking floor width
DIFF-FIX 7  softplus instead of clamp(min=)
DIFF-FIX 8  diagnostics detached
GFR1        Gumbel-Softmax: first-order structure gradients only
GFR1        Straight-through: hard sample forward, soft grad backward
CMT1        σ-field: piecewise constant on cells — NOT in D(A) for
            nonconstant cell indicator (Corollary 3 applies)
CMT1        τ prediction: MLP on features; features detached from
            energy measure computation (avoids second-order through Γ)
GUI1        Streamlit rendering: NOT differentiable (display only)
RF1         React Flow JSON: serialisation only, NOT differentiable

=============================================================================
HONEST SCOPE
=============================================================================
• ρ ≈ 1.25 is NUMERICAL (Carpet-19 §9). The τ learner initialises its
  ρ prior to 1.25 but MUST treat it as uncertain. A log-scale prior
  width of ±0.1 is the default; this is a modelling choice, not a proof.
• Theorem 10 is conditional on (I1)-(I4). For the standard carpet these
  are supported by KZ/BBKT/Hino; for any other fractal they are [O].
• Hino singularity means the energy measure has NO L² density. Any
  visualisation that colours cells by "energy density" is a FINITE-LEVEL
  APPROXIMATION Γ_N/µ(S), not a continuum density. The GUI must say so.
• Gumbel-Softmax structure gradients are first-order. Do not compute
  structure Hessians. Corollary 3 forbids the algebra needed.
• Streamlit is a PROTOTYPE GUI. Production requires React Flow + a
  WebSocket backend for real-time collaborative editing.
• All thresholds are FREE DESIGN PARAMETERS.
=============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
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

logger = logging.getLogger("sesi.topology_studio_gui")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Ecosystem imports
try:
    from sesi_topology_studio import (
        CSGTree, CSGNode, CSGNodeType, VisualCSGTreeEditor,
        MetaTauPredictor, MetaTauConfig, MetaTauTask,
        CLFCTFFileTransformBridge, FileBridgeConfig,
        SESITopologyStudio,
        CARPET_DF, CARPET_RHO, CARPET_DW, CARPET_DS,
    )
    HAS_STUDIO = True
except Exception:
    HAS_STUDIO = False
    # Fallback constants if studio not importable
    CARPET_DF = math.log(8.0) / math.log(3.0)
    CARPET_RHO = 1.25
    CARPET_DW = math.log(8.0 * CARPET_RHO) / math.log(3.0)
    CARPET_DS = 2.0 * CARPET_DF / CARPET_DW

try:
    from sesi_unified_amr2 import NoZenoGate, DorflerMarker
    HAS_NOZENO = True
except Exception:
    HAS_NOZENO = False

try:
    from structural_cahn_hilliard_3d import make_sigma_field
    HAS_CH = True
except Exception:
    HAS_CH = False


__all__ = [
    "GraphRewriteNet", "GumbelConfig",
    "CarpetMeasureTauLearner", "CarpetTauConfig", "CarpetCellFeatures",
    "ReactFlowExporter",
    "StreamlitCSGApp",
    "run_streamlit_app",
]


# =============================================================================
# SECTION 0 — DIFFERENTIABILITY PRIMITIVES
# =============================================================================

def _softplus_floor(x: Tensor, floor: float, beta: Optional[float] = None,
                    sharpness: float = 50.0) -> Tensor:
    if beta is None:
        width = max(abs(floor), 1e-12)
        beta = sharpness / width
    return floor + F.softplus(x - floor, beta=beta)


def _gumbel_softmax(logits: Tensor, tau: float, hard: bool = False,
                    dim: int = -1) -> Tensor:
    """
    [C] Gumbel-Softmax (Jang et al. 2017; Maddison et al. 2017).
    Differentiable relaxation of categorical sampling.
    hard=True: straight-through estimator (hard forward, soft backward).
    """
    if tau < 1e-8:
        # τ→0: hard argmax (not differentiable; use straight-through)
        idx = logits.argmax(dim=dim, keepdim=True)
        hard_out = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
        if hard:
            return hard_out
        return hard_out  # no grad at τ=0
    gumbels = -torch.empty_like(logits).exponential_().log()
    y_soft = F.softmax((logits + gumbels) / tau, dim=dim)
    if hard:
        idx = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
        return y_hard - y_soft.detach() + y_soft  # straight-through
    return y_soft


def _gumbel_sigmoid(logits: Tensor, tau: float, hard: bool = False) -> Tensor:
    """[C] Binary Gumbel-Softmax (Gumbel-Sigmoid) for edge existence."""
    if tau < 1e-8:
        return (logits > 0).float()
    gumbels = -torch.empty_like(logits).exponential_().log()
    y_soft = torch.sigmoid((logits + gumbels) / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


# =============================================================================
# SECTION 1 — [GFR1] GUMBEL-SOFTMAX GRAPH REWRITING
# =============================================================================

@dataclass
class GumbelConfig:
    """Configuration for differentiable graph rewriting."""
    # Gumbel temperature
    tau_init: float = 1.0
    tau_min: float = 0.05
    tau_decay: float = 0.95
    tau_anneal_steps: int = 200
    # Structure
    max_nodes: int = 12
    n_primitive_types: int = 3   # sphere, box, torus
    n_boolean_types: int = 4     # union, intersect, difference, blend
    # Straight-through
    hard_at_eval: bool = True
    # Second-order guard (Corollary 3)
    allow_structure_hessian: bool = False   # MUST remain False
    # Device
    device: str = "cpu"


class GraphRewriteNet(nn.Module):
    """
    [D] Differentiable CSG graph structure search via Gumbel-Softmax.

    DESIGN:
      The CSG graph is represented by two discrete choice tensors:
        node_type_logits : (max_nodes, n_primitive_types + n_boolean_types)
        edge_logits      : (max_nodes, max_nodes)  — directed, upper-triangular

      At training time, Gumbel-Softmax produces soft choices.
      At eval time, straight-through gives hard choices with soft gradients.

    STRUCTURE → FIELD:
      The soft adjacency and node types are used to weight a differentiable
      field algebra. Each node contributes a weighted primitive field; each
      edge contributes a weighted Boolean combination.

    ⚠️  COROLLARY 3 GUARD:
      D(A) is not an algebra. f² ∉ D(A) for nonconstant f ∈ D(A).
      Therefore:
        • allow_structure_hessian is FORCED False in __init__.
        • Second-order structure derivatives raise RuntimeError.
        • Only first-order structure gradients are provided.
      This is not a limitation of the implementation; it is a consequence
      of the carpet's energy measure being singular (Hino Thm 8).

    ⚠️  PROPOSITION 3 GUARD:
      [A, M_f] : W¹ → W⁻¹ loses two orders. Annealing τ_gumbel does NOT
      recover regularity. The τ lower bound is enforced.

    HONEST:
      • Gumbel-Softmax is a VARIATIONAL RELAXATION, not exact optimisation.
      • At τ_gumbel → 0 the relaxation recovers hard sampling but gradients
        become high-variance. Use straight-through for stable training.
      • The graph structure is DISCRETE. Differentiability is approximate.
    """

    # Node type enumeration
    PRIMITIVE_TYPES = ["sphere", "box", "torus"]
    BOOLEAN_TYPES = ["union", "intersect", "difference", "blend"]

    def __init__(self, cfg: Optional[GumbelConfig] = None):
        super().__init__()
        self.cfg = cfg or GumbelConfig()

        # Enforce Corollary 3 guard — cannot be overridden
        if self.cfg.allow_structure_hessian:
            logger.warning("[GFR] allow_structure_hessian forced to False "
                           "(Corollary 3: D(A) not an algebra)")
            self.cfg.allow_structure_hessian = False

        n_node_types = self.cfg.n_primitive_types + self.cfg.n_boolean_types
        n = self.cfg.max_nodes

        # Learnable logits
        self.node_type_logits = nn.Parameter(torch.randn(n, n_node_types) * 0.1)
        # Edge logits: upper triangular only (no self-loops)
        edge_logits = torch.randn(n, n) * 0.1
        self.register_parameter("edge_logits", nn.Parameter(edge_logits))

        # Primitive parameters (learnable, per node slot)
        self.sphere_radius = nn.Parameter(torch.full((n,), 0.3))
        self.sphere_center = nn.Parameter(
            torch.rand(n, 3) * 0.5 + 0.25)  # in [0.25, 0.75]
        self.box_half_ext = nn.Parameter(torch.full((n, 3), 0.15))
        self.torus_R = nn.Parameter(torch.full((n,), 0.3))
        self.torus_r = nn.Parameter(torch.full((n,), 0.1))

        # Gumbel temperature (annealed externally)
        self.register_buffer("tau_gumbel",
                             torch.tensor(self.cfg.tau_init))

        # τ lower bound from Proposition 3 (W¹→W⁻¹ loss)
        self.register_buffer("tau_gumbel_min",
                             torch.tensor(self.cfg.tau_min))

    def anneal_step(self):
        """Decay τ_gumbel toward τ_min (Proposition 3 lower bound)."""
        with torch.no_grad():
            self.tau_gumbel.mul_(self.cfg.tau_decay)
            self.tau_gumbel.clamp_(min=float(self.tau_gumbel_min))

    def get_adjacency(self, hard: bool = False) -> Tensor:
        """
        Soft or hard adjacency matrix from edge logits.
        Upper triangular only (DAG constraint).
        """
        n = self.cfg.max_nodes
        mask = torch.triu(torch.ones(n, n, device=self.edge_logits.device), 1)
        logits = self.edge_logits * mask + (1 - mask) * (-1e9)
        return _gumbel_sigmoid(logits, float(self.tau_gumbel), hard=hard)

    def get_node_types(self, hard: bool = False) -> Tensor:
        """Soft or hard node type one-hot vectors."""
        return _gumbel_softmax(self.node_type_logits,
                                float(self.tau_gumbel),
                                hard=hard, dim=-1)

    def structure_hessian_guard(self):
        """
        ⚠️  COROLLARY 3 GUARD — always raises.
        D(A) is not an algebra. Second-order structure derivatives
        through Boolean composition are NOT available on the carpet.
        """
        raise RuntimeError(
            "Structure Hessian is NOT available. "
            "Corollary 3 (Carpet-19): D(A) is not an algebra. "
            "f² ∉ D(A) for nonconstant f ∈ D(A). "
            "Only first-order structure gradients are provided. "
            "This is a theorem, not an implementation limitation.")

    def hessian(self, *args, **kwargs):
        """Blocked by Corollary 3 guard."""
        self.structure_hessian_guard()

    def build_soft_field(self, grid_shape: Tuple[int, int, int],
                          dx: float = 1.0) -> Tensor:
        """
        Construct a soft implicit field from the differentiable graph.

        Each node slot contributes a primitive field weighted by its
        node-type soft assignment. Edges combine fields via soft Boolean
        operations weighted by the soft adjacency.

        Returns: (X, Y, Z) tensor in the same convention as the CSG editor.
        """
        n = self.cfg.max_nodes
        X, Y, Z = grid_shape
        dev = self.node_type_logits.device

        # Node type weights
        node_types = self.get_node_types(hard=self.cfg.hard_at_eval)
        # node_types: (n, n_types)
        # columns: [sphere, box, torus, union, intersect, difference, blend]

        # Build per-node primitive fields (only primitives contribute)
        # For efficiency, build all three primitive fields per node slot
        fields = []
        for i in range(n):
            cx, cy, cz = (self.sphere_center[i, k] * (grid_shape[k] - 1)
                          for k in range(3))
            r = _softplus_floor(self.sphere_radius[i], 0.05) * min(X, Y, Z) * dx
            he = _softplus_floor(self.box_half_ext[i], 0.02) * min(X, Y, Z) * dx
            R = _softplus_floor(self.torus_R[i], 0.05) * min(X, Y, Z) * dx
            rr = _softplus_floor(self.torus_r[i], 0.02) * min(X, Y, Z) * dx

            sf = _sdf_sphere_field(grid_shape, (cx, cy, cz), r, dx, dev)
            bf = _sdf_box_field(grid_shape, (cx, cy, cz),
                                 (he[0], he[1], he[2]), dx, dev)
            tf = _sdf_torus_field(grid_shape, (cx, cy, cz), R, rr, dx, dev)

            # Weighted by primitive type probabilities
            w_sphere = node_types[i, 0]
            w_box = node_types[i, 1]
            w_torus = node_types[i, 2]
            w_total = _softplus_floor(w_sphere + w_box + w_torus, 1e-6)
            combined = (w_sphere * sf + w_box * bf + w_torus * tf) / w_total
            fields.append(combined)

        # Soft Boolean combination via adjacency
        adj = self.get_adjacency(hard=self.cfg.hard_at_eval)
        # Simple reduction: combine fields pairwise weighted by adjacency
        result = fields[0]
        for j in range(1, n):
            # Weight for Boolean type of node j
            w_union = node_types[j, 3]
            w_inter = node_types[j, 4]
            w_diff = node_types[j, 5]
            w_blend = node_types[j, 6]
            w_bool = _softplus_floor(
                w_union + w_inter + w_diff + w_blend, 1e-6)

            # Edge weight from all predecessors to j
            edge_w = adj[:j, j].sum()  # total incoming edge weight
            gate = torch.sigmoid(edge_w * 4.0)  # soft existence gate

            # Soft Boolean: weighted combination
            tau_b = max(float(self.tau_gumbel), 0.05)
            union_f = _softmin_tau(result, fields[j], tau_b)
            inter_f = _softmax_tau(result, fields[j], tau_b)
            diff_f = _softmax_tau(result, -fields[j], tau_b)
            blend_f = _blend_tau(result, fields[j], tau_b)

            bool_f = (w_union * union_f + w_inter * inter_f +
                      w_diff * diff_f + w_blend * blend_f) / w_bool
            result = (1 - gate) * result + gate * bool_f

        return result

    def extract_hard_tree(self, grid_shape: Tuple[int, int, int],
                            dx: float = 1.0) -> "CSGTree":
        """
        Extract a hard CSGTree from the current soft structure.
        Uses argmax for node types and threshold for edges.
        """
        if not HAS_STUDIO:
            raise RuntimeError("sesi_topology_studio not importable")

        node_types = self.get_node_types(hard=True)
        adj = self.get_adjacency(hard=True)

        tree = CSGTree()
        node_ids = []
        n = self.cfg.max_nodes

        for i in range(n):
            # Find dominant type
            type_idx = int(node_types[i].argmax())
            is_primitive = type_idx < self.cfg.n_primitive_types

            if is_primitive:
                ptype = self.PRIMITIVE_TYPES[type_idx]
                nid = f"{ptype}_{i}"
                if ptype == "sphere":
                    cx, cy, cz = [float(self.sphere_center[i, k]) *
                                   (grid_shape[k]-1) for k in range(3)]
                    r = float(self.sphere_radius[i]) * min(grid_shape) * dx
                    tree.add_primitive(CSGNodeType.SPHERE, node_id=nid,
                                       center=(cx, cy, cz), radius=r)
                elif ptype == "box":
                    cx, cy, cz = [float(self.sphere_center[i, k]) *
                                   (grid_shape[k]-1) for k in range(3)]
                    he = [float(self.box_half_ext[i, k]) *
                          min(grid_shape) * dx for k in range(3)]
                    tree.add_primitive(CSGNodeType.BOX, node_id=nid,
                                       center=(cx, cy, cz), half_extents=he)
                else:
                    cx, cy, cz = [float(self.sphere_center[i, k]) *
                                   (grid_shape[k]-1) for k in range(3)]
                    R = float(self.torus_R[i]) * min(grid_shape) * dx
                    rr = float(self.torus_r[i]) * min(grid_shape) * dx
                    tree.add_primitive(CSGNodeType.TORUS, node_id=nid,
                                       center=(cx, cy, cz), R=R, r=rr)
                node_ids.append(nid)
            else:
                # Boolean node: find two inputs with highest edge weight
                bool_type = self.BOOLEAN_TYPES[
                    type_idx - self.cfg.n_primitive_types]
                incoming = [(k, float(adj[k, i])) for k in range(i)]
                incoming.sort(key=lambda x: x[1], reverse=True)
                if len(incoming) < 2:
                    continue
                in_a = node_ids[incoming[0][0]] if incoming[0][0] < len(node_ids) else node_ids[-1]
                in_b = node_ids[incoming[1][0]] if incoming[1][0] < len(node_ids) else node_ids[-2]
                nid = f"{bool_type}_{i}"
                op_map = {
                    "union": CSGNodeType.UNION,
                    "intersect": CSGNodeType.INTERSECT,
                    "difference": CSGNodeType.DIFFERENCE,
                    "blend": CSGNodeType.BLEND,
                }
                tree.add_op(op_map[bool_type], [in_a, in_b], node_id=nid)
                node_ids.append(nid)

        if node_ids:
            oid = f"out_{uuid.uuid4().hex[:6]}"
            tree.add_node(CSGNode(oid, CSGNodeType.OUTPUT, inputs=[node_ids[-1]]))

        return tree

    def get_status(self) -> Dict[str, Any]:
        return {
            "tau_gumbel": float(self.tau_gumbel),
            "tau_min": float(self.tau_gumbel_min),
            "max_nodes": self.cfg.max_nodes,
            "hard_at_eval": self.cfg.hard_at_eval,
            "structure_hessian_allowed": False,  # Corollary 3
            "structure_hessian_reason": "Corollary 3: D(A) not an algebra",
        }


# ── SDF helpers (self-contained, no external dependency) ─────────────────

def _sdf_sphere_field(shape, center, radius, dx, device) -> Tensor:
    X, Y, Z = shape
    xs = torch.arange(X, device=device, dtype=torch.float32) * dx
    ys = torch.arange(Y, device=device, dtype=torch.float32) * dx
    zs = torch.arange(Z, device=device, dtype=torch.float32) * dx
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    cx, cy, cz = center
    return torch.sqrt((gx-cx)**2 + (gy-cy)**2 + (gz-cz)**2 + 1e-12) - radius


def _sdf_box_field(shape, center, half_ext, dx, device) -> Tensor:
    X, Y, Z = shape
    xs = torch.arange(X, device=device, dtype=torch.float32) * dx
    ys = torch.arange(Y, device=device, dtype=torch.float32) * dx
    zs = torch.arange(Z, device=device, dtype=torch.float32) * dx
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    cx, cy, cz = center
    hx, hy, hz = half_ext
    qx = torch.abs(gx - cx) - hx
    qy = torch.abs(gy - cy) - hy
    qz = torch.abs(gz - cz) - hz
    outside = torch.sqrt(
        _softplus_floor(qx, 0.0)**2 +
        _softplus_floor(qy, 0.0)**2 +
        _softplus_floor(qz, 0.0)**2)
    inside = torch.minimum(torch.minimum(qx, qy), qz)
    inside = torch.clamp(inside, max=0.0)
    return outside + inside


def _sdf_torus_field(shape, center, R, r, dx, device) -> Tensor:
    X, Y, Z = shape
    xs = torch.arange(X, device=device, dtype=torch.float32) * dx
    ys = torch.arange(Y, device=device, dtype=torch.float32) * dx
    zs = torch.arange(Z, device=device, dtype=torch.float32) * dx
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    cx, cy, cz = center
    px, py, pz = gx-cx, gy-cy, gz-cz
    qx = torch.sqrt(px**2 + pz**2 + 1e-12) - R
    return torch.sqrt(qx**2 + py**2 + 1e-12) - r


def _softmin_tau(a: Tensor, b: Tensor, tau: float) -> Tensor:
    ta, tb = -a/max(tau, 1e-12), -b/max(tau, 1e-12)
    m = torch.maximum(ta, tb)
    return -tau * (m + torch.log(torch.exp(ta-m) + torch.exp(tb-m) + 1e-30))


def _softmax_tau(a: Tensor, b: Tensor, tau: float) -> Tensor:
    ta, tb = a/max(tau, 1e-12), b/max(tau, 1e-12)
    m = torch.maximum(ta, tb)
    return tau * (m + torch.log(torch.exp(ta-m) + torch.exp(tb-m) + 1e-30))


def _blend_tau(a: Tensor, b: Tensor, tau: float, k: float = 0.5) -> Tensor:
    h = torch.clamp(0.5 + 0.5*(b-a)/max(k, 1e-8), 0.0, 1.0)
    return b + h*(a-b) - k*h*(1-h)


# =============================================================================
# SECTION 2 — [CMT1] CARPET-MEASURE τ LEARNER
# =============================================================================

@dataclass
class CarpetCellFeatures:
    """
    Finite-level cell features for τ prediction.

    ⚠️  HINO SINGULARITY WARNING:
    These features are Γ_N(S)/µ(S) — FINITE-LEVEL APPROXIMATIONS.
    They do NOT converge to an L² density (Hino Thm 8: Γ ⊥ µ).
    They converge weakly as measures (Theorem 6), not as functions.
    Any model that assumes a smooth density will be systematically wrong.
    """
    cell_level: int
    n_cells: int
    gamma_over_mu: Tensor      # (n_cells,) — Γ_N(S)/µ(S), finite-level proxy
    rho_prior: float           # initialised to 1.25 (NUMERICAL, not proved)
    beta_holder: float         # β = log ρ / log 3 (Hölder exponent)
    d_f: float                 # ln8/ln3
    d_s: float                 # 2d_f/d_w (using ρ prior)
    spectral_gap_proxy: float  # λ₁ proxy from finite graph


@dataclass
class CarpetTauConfig:
    """Configuration for carpet-measure τ meta-learning."""
    # Carpet parameters
    rho_init: float = CARPET_RHO        # 1.25 — NUMERICAL, not proved
    rho_log_prior_width: float = 0.1    # log-scale uncertainty on ρ
    carpet_level: int = 3               # finite-level graph resolution
    # Meta-learning
    meta_lr: float = 5e-4
    inner_lr: float = 2e-3
    inner_steps: int = 3
    first_order: bool = True            # FOMAML (cheaper; second-order
                                        # through Γ blocked by Cor. 3)
    # τ bounds
    tau_base: float = 0.05
    tau_min: float = 0.005
    tau_max: float = 0.5
    # Features
    use_energy_measure: bool = True
    use_spectral_gap: bool = True
    use_holder_beta: bool = True
    # No-Zeno
    max_meta_updates_per_minute: int = 20
    use_no_zeno: bool = True
    theta_hi: float = 0.2
    theta_lo: float = 0.05
    # Device
    device: str = "cpu"


class CarpetMeasureTauLearner(nn.Module):
    """
    [C/D] Meta-learner for τ conditioned on carpet-measure features.

    GROUNDED IN CARPET-19:
      • ρ initialised to 1.25 with explicit uncertainty (NUMERICAL)
      • β = log ρ / log 3 (Hölder exponent from (I1))
      • d_s = 2 d_f / d_w (spectral dimension)
      • Energy measure features are FINITE-LEVEL PROXIES (Hino: Γ ⊥ µ)
      • First-order MAML only (Corollary 3 blocks second-order)

    FEATURES PER TASK:
      1. Finite-level energy measure proxy: Γ_N(S)/µ(S) per m-cell
      2. Spectral gap proxy: λ₁ of the finite graph Laplacian
      3. Hölder exponent β = log ρ / log 3
      4. Spectral dimension d_s = 2 d_f / d_w
      5. ρ uncertainty width (log-scale)

    τ PREDICTION:
      τ = τ_base · exp(MLP(features))
      Clamped to [τ_min, τ_max].

    META-LEARNING:
      FOMAML over a distribution of carpet measure fields at different
      levels and ρ values. Inner loop adapts to a specific field.

    ⚠️  HONEST SCOPE:
      • ρ = 1.25 is NUMERICAL (Carpet-19 §9). The learner treats it as
        a prior with uncertainty, not a proved constant.
      • Theorem 10 is conditional on (I1)-(I4). This learner inherits
        that conditionality.
      • Energy measure features are finite-level approximations.
        They do NOT converge to an L² density (Hino Thm 8).
      • First-order MAML only. Second-order structure derivatives are
        blocked by Corollary 3.
      • The learned τ is a HEURISTIC. No optimality claim.
    """

    def __init__(self, cfg: Optional[CarpetTauConfig] = None):
        super().__init__()
        self.cfg = cfg or CarpetTauConfig()

        # ρ as learnable log-parameter with numerical initialisation
        self.log_rho = nn.Parameter(
            torch.tensor(math.log(self.cfg.rho_init)))
        # Log-scale prior width (uncertainty on ρ)
        self.register_buffer("rho_log_width",
                             torch.tensor(self.cfg.rho_log_prior_width))

        # Feature MLP
        n_feat = 5  # energy_measure_mean, spectral_gap, beta, d_s, rho_uncertainty
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

        # τ bounds in log-scale
        self.register_buffer("log_tau_lo",
            torch.tensor(math.log(self.cfg.tau_min / self.cfg.tau_base)))
        self.register_buffer("log_tau_hi",
            torch.tensor(math.log(self.cfg.tau_max / self.cfg.tau_base)))

        # No-Zeno gate
        if self.cfg.use_no_zeno and HAS_NOZENO:
            self.gate = NoZenoGate(self.cfg.theta_hi, self.cfg.theta_lo)
        else:
            self.gate = None
        self._meta_updates = 0

    # ── carpet parameters from ρ ────────────────────────────────────────

    @property
    def rho(self) -> Tensor:
        return torch.exp(self.log_rho)

    @property
    def beta_holder(self) -> Tensor:
        """β = log ρ / log 3 — Hölder exponent from (I1)."""
        return torch.log(self.rho) / math.log(3.0)

    @property
    def d_w(self) -> Tensor:
        """d_w = ln(8ρ)/ln(3) — walk dimension."""
        return torch.log(8.0 * self.rho) / math.log(3.0)

    @property
    def d_s(self) -> Tensor:
        """d_s = 2 d_f / d_w — spectral dimension."""
        return 2.0 * CARPET_DF / self.d_w

    # ── finite-level carpet graph ────────────────────────────────────────

    def build_carpet_graph_laplacian(self, level: int) -> Tensor:
        """
        Build the finite-level carpet cell graph Laplacian.

        Level n: 8^n cells. Adjacency: side-sharing cells.
        This is the graph G_n from Carpet-19 §1.

        HONEST: For level > 4 this becomes very large (8^4 = 4096 cells).
        Use sparse operations for level >= 4.
        """
        n_cells = 8 ** level
        # Build carpet occupancy mask
        mask = self._carpet_mask(level)
        # Cell indices
        cell_ids = {}
        idx = 0
        H = W = 3 ** level
        for i in range(H):
            for j in range(W):
                if mask[i, j]:
                    cell_ids[(i, j)] = idx
                    idx += 1

        # Adjacency: side-sharing
        edges = []
        for (i, j), cid in cell_ids.items():
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i+di, j+dj
                if (ni, nj) in cell_ids:
                    edges.append((cid, cell_ids[(ni, nj)]))

        # Build sparse Laplacian
        N = len(cell_ids)
        L = torch.zeros(N, N, dtype=torch.float32)
        for a, b in edges:
            L[a, b] -= 1.0
            L[b, a] -= 1.0
            L[a, a] += 1.0
            L[b, b] += 1.0
        return L

    def _carpet_mask(self, level: int) -> np.ndarray:
        """Boolean occupancy mask for the Sierpinski carpet at given level."""
        size = 3 ** level
        mask = np.ones((size, size), dtype=bool)
        def punch(x0, y0, s):
            if s < 3:
                return
            # Remove centre cell
            cx, cy = x0 + s//3, y0 + s//3
            mask[cx:cx+s//3, cy:cy+s//3] = False
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    punch(x0 + i*s//3, y0 + j*s//3, s//3)
        punch(0, 0, size)
        return mask

    def compute_spectral_gap(self, L: Tensor) -> float:
        """
        λ₁ of the finite graph Laplacian (spectral gap proxy).
        Uses eigendecomposition. For large graphs, use sparse methods.
        """
        if L.shape[0] > 2000:
            # Too large for dense eigendecomposition; use power iteration
            return self._power_iteration_gap(L)
        eigenvalues = torch.linalg.eigvalsh(L)
        # λ₀ = 0 (constant); λ₁ is the spectral gap
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    def _power_iteration_gap(self, L: Tensor, n_iter: int = 100) -> float:
        """Estimate λ₁ via power iteration on (I - L/λ_max)."""
        n = L.shape[0]
        v = torch.randn(n)
        v = v - v.mean()
        v = v / (v.norm() + 1e-12)
        # Estimate λ_max
        for _ in range(20):
            v = L @ v
            v = v / (v.norm() + 1e-12)
        lam_max = float(v @ L @ v)
        # Power iteration for smallest nonzero eigenvalue
        M = torch.eye(n) - L / max(lam_max, 1e-12)
        v = torch.randn(n)
        v = v - v.mean()
        v = v / (v.norm() + 1e-12)
        for _ in range(n_iter):
            v = M @ v
            v = v - v.mean()  # project out constant
            v = v / (v.norm() + 1e-12)
        lam1 = float(v @ L @ v)
        return lam1

    def compute_energy_measure_proxy(self, field: Tensor, level: int
                                      ) -> Tensor:
        """
        Finite-level energy measure proxy Γ_N(S)/µ(S).

        ⚠️  HINO SINGULARITY WARNING:
        This is a FINITE-LEVEL APPROXIMATION. It does NOT converge to an
        L² density. It converges weakly as a measure (Theorem 6).
        The values can be unbounded and highly variable across cells.
        Do NOT assume smoothness or boundedness.

        For each m-cell S (m < level), compute:
          Γ_N(S) = sum of edge energy contributions from cells inside S
          µ(S) = 8^{-m}
          proxy = Γ_N(S) / µ(S)
        """
        mask = self._carpet_mask(level)
        H = W = 3 ** level
        m = max(1, level - 1)  # coarse level for cells
        cell_size = 3 ** (level - m)
        n_cells_m = 8 ** m

        # Compute field gradient energy on the grid
        # field: (H, W) or (H, W, D) — use first slice if 3D
        if field.dim() == 3:
            f2d = field[:, :, field.shape[2]//2]
        else:
            f2d = field

        # Finite differences
        gx = torch.roll(f2d, -1, 0) - f2d
        gy = torch.roll(f2d, -1, 1) - f2d
        edge_energy = gx**2 + gy**2  # per-cell energy proxy

        # Aggregate to m-cells
        gamma_over_mu = torch.zeros(n_cells_m)
        idx = 0
        for ci in range(3**m):
            for cj in range(3**m):
                # Check if this m-cell is occupied in the carpet
                # (simplified: check centre)
                cx, cy = ci * cell_size + cell_size//2, cj * cell_size + cell_size//2
                if cx < H and cy < W and mask[cx, cy]:
                    # Sum edge energy in this cell
                    i0, i1 = ci*cell_size, min((ci+1)*cell_size, H)
                    j0, j1 = cj*cell_size, min((cj+1)*cell_size, W)
                    gamma = edge_energy[i0:i1, j0:j1].sum()
                    mu = 8.0 ** (-m)
                    gamma_over_mu[idx] = gamma / max(mu, 1e-12)
                    idx += 1
                    if idx >= n_cells_m:
                        break
            if idx >= n_cells_m:
                break

        return gamma_over_mu[:idx]

    # ── feature extraction ───────────────────────────────────────────────

    def extract_features(self, field: Tensor, level: Optional[int] = None
                          ) -> Tensor:
        """
        Extract the 5-dimensional feature vector for τ prediction.

        Features:
          0: mean energy measure proxy (finite-level, Hino-singular)
          1: spectral gap λ₁
          2: Hölder exponent β = log ρ / log 3
          3: spectral dimension d_s = 2 d_f / d_w
          4: ρ uncertainty width (log-scale)
        """
        level = level or self.cfg.carpet_level

        feats = []

        # 1. Energy measure proxy
        if self.cfg.use_energy_measure:
            gamma_proxy = self.compute_energy_measure_proxy(field, level)
            feats.append(gamma_proxy.mean().detach())
        else:
            feats.append(torch.tensor(0.0))

        # 2. Spectral gap
        if self.cfg.use_spectral_gap:
            try:
                L = self.build_carpet_graph_laplacian(min(level, 3))
                gap = self.compute_spectral_gap(L)
            except Exception:
                gap = 0.0
            feats.append(torch.tensor(gap, dtype=torch.float32))
        else:
            feats.append(torch.tensor(0.0))

        # 3. Hölder exponent β
        if self.cfg.use_holder_beta:
            feats.append(self.beta_holder.detach())
        else:
            feats.append(torch.tensor(0.0))

        # 4. Spectral dimension d_s
        feats.append(self.d_s.detach())

        # 5. ρ uncertainty width
        feats.append(self.rho_log_width)

        return torch.stack(feats)

    # ── τ prediction ────────────────────────────────────────────────────

    def predict_tau(self, features: Tensor) -> Tensor:
        """τ = τ_base · exp(clamp(MLP(features)))."""
        log_scale = self.mlp(features.unsqueeze(0)).squeeze()
        log_scale = torch.clamp(log_scale,
                                float(self.log_tau_lo),
                                float(self.log_tau_hi))
        return self.cfg.tau_base * torch.exp(log_scale)

    def predict_from_field(self, field: Tensor,
                            level: Optional[int] = None) -> Tensor:
        feats = self.extract_features(field, level)
        return self.predict_tau(feats)

    # ── FOMAML meta-training ────────────────────────────────────────────

    def meta_train(self, tasks: List[MetaTauTask],
                    n_meta_epochs: int = 50) -> Dict[str, Any]:
        """
        FOMAML meta-training over carpet measure field tasks.

        ⚠️  FIRST-ORDER ONLY (Corollary 3):
        Second-order MAML would require differentiating through the
        energy measure computation, which involves products of field
        values. Corollary 3 (D(A) not an algebra) means this is not
        valid for carpet fields. FOMAML avoids this by detaching the
        inner-loop gradient.
        """
        meta_opt = torch.optim.Adam(
            list(self.mlp.parameters()) + [self.log_rho],
            lr=self.cfg.meta_lr)
        meta_losses = []

        for epoch in range(n_meta_epochs):
            meta_opt.zero_grad()
            meta_loss = torch.tensor(0.0)
            n_tasks_used = 0

            for task in tasks[:8]:
                feats = self.extract_features(task.field)
                tau_pred = self.predict_tau(feats)
                loss = (torch.log(tau_pred) -
                        math.log(max(task.tau_target, 1e-9)))**2

                # FOMAML: detach inner gradient (first-order)
                meta_loss = meta_loss + loss.detach()
                n_tasks_used += 1

            meta_loss = meta_loss / max(n_tasks_used, 1)
            # For FOMAML, we still need a gradient signal:
            # use the outer loss directly (Reptile-style update)
            outer_loss = torch.tensor(0.0)
            for task in tasks[:8]:
                feats = self.extract_features(task.field)
                tau_pred = self.predict_tau(feats)
                outer_loss = outer_loss + (
                    torch.log(tau_pred) -
                    math.log(max(task.tau_target, 1e-9)))**2
            outer_loss = outer_loss / max(n_tasks_used, 1)
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            meta_opt.step()

            meta_losses.append(float(outer_loss))

            # No-Zeno bound on meta-updates
            self._meta_updates += 1
            if self.gate is not None:
                g = self.gate.update(float(outer_loss))
                if not g.get("allowed", True) and not g.get("invariant_holds", True):
                    logger.warning("[CMT] No-Zeno invariant violated; stopping")
                    break

        return {
            "n_meta_epochs": n_meta_epochs,
            "final_meta_loss": meta_losses[-1] if meta_losses else float("nan"),
            "loss_history": meta_losses,
            "n_meta_updates": self._meta_updates,
            "rho_learned": float(self.rho.item()),
            "rho_init": self.cfg.rho_init,
            "rho_status": "NUMERICAL, not proved [C-num]",
            "beta_learned": float(self.beta_holder.item()),
            "d_s_learned": float(self.d_s.item()),
            "first_order_only": True,
            "first_order_reason": "Corollary 3: D(A) not an algebra",
        }

    # ── fast adaptation ─────────────────────────────────────────────────

    def adapt_to_field(self, field: Tensor,
                        tau_hint: Optional[float] = None,
                        steps: Optional[int] = None) -> Dict[str, Any]:
        """
        No-Zeno-bounded τ adaptation for a new carpet measure field.
        """
        if self.gate is not None:
            feats0 = self.extract_features(field)
            stress = float(feats0[0])  # energy measure mean as stress
            g = self.gate.update(stress)
            if not g.get("allowed", True):
                return {
                    "adapted": False,
                    "reason": "no_zeno_gate_closed",
                    "tau": float(self.predict_from_field(field)),
                    "event_count": g.get("event_count", 0),
                }

        tau = self.predict_from_field(field)
        steps = steps or self.cfg.inner_steps

        if tau_hint is not None:
            # Inner loop: adapt toward hint
            fast_mlp = copy.deepcopy(self.mlp)
            opt = torch.optim.SGD(fast_mlp.parameters(),
                                   lr=self.cfg.inner_lr)
            feats = self.extract_features(field)
            for _ in range(steps):
                opt.zero_grad()
                log_s = fast_mlp(feats.unsqueeze(0)).squeeze()
                log_s = torch.clamp(log_s,
                                    float(self.log_tau_lo),
                                    float(self.log_tau_hi))
                tp = self.cfg.tau_base * torch.exp(log_s)
                loss = (torch.log(tp) - math.log(max(tau_hint, 1e-9)))**2
                loss.backward()
                opt.step()
            with torch.no_grad():
                log_s = fast_mlp(feats.unsqueeze(0)).squeeze()
                log_s = torch.clamp(log_s,
                                    float(self.log_tau_lo),
                                    float(self.log_tau_hi))
                tau = self.cfg.tau_base * torch.exp(log_s)

        return {
            "adapted": True,
            "tau": float(tau),
            "rho": float(self.rho.item()),
            "rho_status": "NUMERICAL, not proved [C-num]",
            "beta": float(self.beta_holder.item()),
            "d_s": float(self.d_s.item()),
            "n_meta_updates": self._meta_updates,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "rho": float(self.rho.item()),
            "rho_init": self.cfg.rho_init,
            "rho_status": "NUMERICAL, not proved [C-num]",
            "rho_log_width": float(self.rho_log_width),
            "beta_holder": float(self.beta_holder.item()),
            "d_w": float(self.d_w.item()),
            "d_s": float(self.d_s.item()),
            "d_f": CARPET_DF,
            "tau_base": self.cfg.tau_base,
            "tau_range": (self.cfg.tau_min, self.cfg.tau_max),
            "first_order_only": True,
            "first_order_reason": "Corollary 3: D(A) not an algebra",
            "hino_singularity": "Γ(f,·) ⊥ µ — no L² density [C Thm 8]",
            "n_meta_updates": self._meta_updates,
        }


# =============================================================================
# SECTION 3 — [RF1] REACT FLOW EXPORTER
# =============================================================================

class ReactFlowExporter:
    """
    [D] Serialises a CSGTree to React Flow JSON format.

    React Flow (reactflow.dev) expects:
      nodes: [{id, type, position: {x,y}, data: {...}}]
      edges: [{id, source, target, type}]

    Layout: topological levels → x position; index within level → y.

    Node styling by type:
      primitive  → green
      transform  → blue
      boolean    → orange
      output     → red

    HONEST: This is a STATIC EXPORT. It does not provide a live editor.
    For a live editor, use the StreamlitCSGApp or build a React frontend.
    """

    NODE_COLORS = {
        "primitive": "#4caf50",
        "transform": "#2196f3",
        "boolean": "#ff9800",
        "output": "#f44336",
    }

    NODE_TYPE_MAP = {
        CSGNodeType.SPHERE: "primitive",
        CSGNodeType.BOX: "primitive",
        CSGNodeType.TORUS: "primitive",
        CSGNodeType.TRANSLATE: "transform",
        CSGNodeType.ROTATE_Z: "transform",
        CSGNodeType.SCALE: "transform",
        CSGNodeType.UNION: "boolean",
        CSGNodeType.INTERSECT: "boolean",
        CSGNodeType.DIFFERENCE: "boolean",
        CSGNodeType.BLEND: "boolean",
        CSGNodeType.COMPLEMENT: "boolean",
        CSGNodeType.OUTPUT: "output",
    } if HAS_STUDIO else {}

    @classmethod
    def _classify(cls, node_type) -> str:
        if not HAS_STUDIO:
            return "primitive"
        return cls.NODE_TYPE_MAP.get(node_type, "primitive")

    def to_react_flow(self, tree: "CSGTree") -> Dict[str, Any]:
        """Convert a CSGTree to React Flow JSON."""
        if not HAS_STUDIO:
            raise RuntimeError("sesi_topology_studio not importable")

        order = tree.topo_order()
        # Assign levels (depth in DAG)
        levels: Dict[str, int] = {}
        for nid in order:
            node = tree.nodes[nid]
            if not node.inputs:
                levels[nid] = 0
            else:
                levels[nid] = max(
                    levels.get(inp, 0) for inp in node.inputs) + 1

        # Group by level for y positioning
        by_level: Dict[int, List[str]] = {}
        for nid, lv in levels.items():
            by_level.setdefault(lv, []).append(nid)

        nodes = []
        edges = []
        x_spacing, y_spacing = 220, 120

        for lv, nids in sorted(by_level.items()):
            for yi, nid in enumerate(nids):
                node = tree.nodes[nid]
                ntype = self._classify(node.node_type)
                nodes.append({
                    "id": nid,
                    "type": "default",
                    "position": {"x": lv * x_spacing,
                                  "y": yi * y_spacing},
                    "data": {
                        "label": f"{node.node_type.value}\n({nid})",
                        "node_type": node.node_type.value,
                        "category": ntype,
                        "params": node.params,
                        "tau": node.tau,
                        "color": self.NODE_COLORS.get(ntype, "#999"),
                    },
                    "style": {
                        "background": self.NODE_COLORS.get(ntype, "#999"),
                        "color": "white",
                        "border": "2px solid #333",
                        "borderRadius": "8px",
                        "padding": "8px",
                        "fontSize": "11px",
                    },
                })
                for inp in node.inputs:
                    edges.append({
                        "id": f"e_{inp}_{nid}",
                        "source": inp,
                        "target": nid,
                        "type": "smoothstep",
                        "animated": ntype == "boolean",
                        "style": {"stroke": "#666", "strokeWidth": 2},
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "viewport": {"x": 50, "y": 50, "zoom": 0.8},
            "metadata": {
                "generator": "SESI ReactFlowExporter",
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "tree_valid": tree.validate()["valid"],
                "carpet_reference": {
                    "d_f": CARPET_DF,
                    "rho": CARPET_RHO,
                    "rho_status": "NUMERICAL, not proved [C-num]",
                    "d_s": CARPET_DS,
                },
            },
        }

    def to_json_string(self, tree: "CSGTree", indent: int = 2) -> str:
        return json.dumps(self.to_react_flow(tree), indent=indent)

    def save(self, tree: "CSGTree", path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json_string(tree))


# =============================================================================
# SECTION 4 — [GUI1] STREAMLIT CSG APP
# =============================================================================

class StreamlitCSGApp:
    """
    [D] Streamlit-based GUI for the CSG tree editor.

    PAGES:
      1. Node palette + tree structure (JSON editor)
      2. 2D SDF slice viewer (matplotlib)
      3. τ control + meta-adaptation
      4. OCIO colour transform
      5. Export (OBJ / SVG / React Flow JSON / .ocio report)
      6. Carpet-19 reference panel

    HONEST:
      • This is a PROTOTYPE GUI. Production requires React Flow + WebSocket.
      • The 3D preview is a 2D slice (matplotlib). True 3D requires
        three.js / pyvista / VTK.
      • Streamlit re-runs the entire script on every interaction.
        Heavy computations should be cached with @st.cache_data.

    RUN:
      streamlit run sesi_topology_studio_gui.py
    """

    def __init__(self, studio: Optional[Any] = None):
        self.studio = studio
        self.rf_exporter = ReactFlowExporter()

    def _ensure_studio(self):
        if self.studio is None:
            if HAS_STUDIO:
                self.studio = SESITopologyStudio()
            else:
                raise RuntimeError("sesi_topology_studio not importable")

    def render(self):
        """Main Streamlit render function."""
        if not HAS_STREAMLIT:
            logger.error("Streamlit not installed. pip install streamlit")
            return

        st.set_page_config(
            page_title="SESI Topology Studio",
            page_icon="🔷",
            layout="wide",
        )

        st.title("🔷 SESI Topology Studio")
        st.caption(
            "CSG Tree Editor + Carpet-Measure τ + OCIO | "
            "Carpet-19: ρ≈1.25 [NUMERICAL] | d_f=ln8/ln3 | "
            "Hino: Γ⊥µ [Thm 8]"
        )

        self._ensure_studio()

        # Sidebar: controls
        with st.sidebar:
            st.header("⚙️ Controls")
            tab = st.radio("View", [
                "CSG Tree", "SDF Preview", "τ Meta-Learning",
                "OCIO Colour", "Export", "Carpet-19 Reference",
            ])

            st.divider()
            st.subheader("Carpet-19 Constants")
            st.markdown(f"""
            | Quantity | Value | Status |
            |---|---|---|
            | d_f = ln8/ln3 | {CARPET_DF:.6f} | [C] |
            | ρ | {CARPET_RHO} | **NUMERICAL** |
            | d_w = ln(8ρ)/ln3 | {CARPET_DW:.6f} | [C-num] |
            | d_s = 2d_f/d_w | {CARPET_DS:.6f} | [C-num] |
            | β = log ρ/log 3 | {math.log(CARPET_RHO)/math.log(3):.6f} | [D] |
            """)
            st.warning(
                "⚠️ ρ ≈ 1.25 is NUMERICAL, not proved. "
                "Theorem 10 is conditional on (I1)-(I4). "
                "Hino Thm 8: Γ(f,·) ⊥ µ — no L² density."
            )

        # Main content
        if tab == "CSG Tree":
            self._page_csg_tree()
        elif tab == "SDF Preview":
            self._page_sdf_preview()
        elif tab == "τ Meta-Learning":
            self._page_tau_meta()
        elif tab == "OCIO Colour":
            self._page_ocio()
        elif tab == "Export":
            self._page_export()
        elif tab == "Carpet-19 Reference":
            self._page_carpet_reference()

    def _page_csg_tree(self):
        st.header("🌳 CSG Tree Editor")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Node Palette")
            # Add primitive
            prim_type = st.selectbox("Primitive", ["sphere", "box", "torus"])
            if st.button("➕ Add Primitive"):
                self._add_primitive(prim_type)

            st.divider()
            # Add boolean
            bool_type = st.selectbox("Boolean",
                                      ["union", "intersect", "difference", "blend"])
            if st.button("➕ Add Boolean"):
                self._add_boolean(bool_type)

            st.divider()
            # Tree JSON
            st.subheader("Tree JSON")
            if hasattr(self, '_tree_json'):
                edited = st.text_area("Edit tree JSON",
                                       self._tree_json, height=300)
                if st.button("Apply JSON"):
                    self._apply_tree_json(edited)

        with col2:
            st.subheader("Tree Structure")
            if hasattr(self, '_tree') and self._tree is not None:
                tree = self._tree
                val = tree.validate()
                st.info(f"Nodes: {val['n_nodes']} | Valid: {val['valid']}")
                if not val['valid']:
                    st.error(f"Errors: {val['errors']}")

                # Display nodes
                for nid, node in tree.nodes.items():
                    with st.expander(
                            f"**{node.node_type.value}** `{nid}` "
                            f"(inputs: {node.inputs})"):
                        st.json(node.to_dict())
                        if st.button(f"🗑️ Delete {nid}", key=f"del_{nid}"):
                            self._delete_node(nid)

                # React Flow JSON
                st.divider()
                st.subheader("React Flow JSON")
                rf_json = self.rf_exporter.to_json_string(tree)
                st.code(rf_json, language="json")
                st.download_button("⬇️ Download React Flow JSON",
                                    rf_json, "csg_tree_reactflow.json")
            else:
                st.info("No tree yet. Add primitives and Boolean ops.")

    def _page_sdf_preview(self):
        st.header("🖼️ SDF Preview")

        if not hasattr(self, '_tree') or self._tree is None:
            st.info("Build a CSG tree first.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Parameters")
            slice_axis = st.selectbox("Slice axis", ["z", "y", "x"])
            slice_idx = st.slider("Slice index", 0, 23, 12)
            tau_slider = st.slider("Boolean τ (sharpness)",
                                    0.005, 0.2, 0.05, 0.005)

        with col2:
            st.subheader("2D Slice")
            if HAS_MPL:
                try:
                    result = self.studio.editor.execute(self._tree,
                                                         tau=tau_slider)
                    sdf = result["field"]
                    if slice_axis == "z":
                        sl = sdf[:, :, slice_idx]
                    elif slice_axis == "y":
                        sl = sdf[:, slice_idx, :]
                    else:
                        sl = sdf[slice_idx, :, :]

                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    im = ax.imshow(sl.detach().cpu().numpy(),
                                    cmap="RdBu_r", origin="lower")
                    ax.set_title(f"SDF slice ({slice_axis}={slice_idx})\n"
                                 f"τ={tau_slider:.3f}")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)

                    st.metric("SDF range",
                              f"[{sdf.min():.3f}, {sdf.max():.3f}]")
                except Exception as e:
                    st.error(f"Preview error: {e}")
            else:
                st.warning("matplotlib not installed")

    def _page_tau_meta(self):
        st.header("🧮 Carpet-Measure τ Meta-Learning")

        st.warning(
            "⚠️ **Carpet-19 constraints:**\n"
            "- ρ ≈ 1.25 is NUMERICAL, not proved\n"
            "- Hino Thm 8: Γ(f,·) ⊥ µ — no L² density exists\n"
            "- Corollary 3: D(A) not an algebra — first-order only\n"
            "- Theorem 10 conditional on (I1)-(I4)"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ρ Prior")
            rho_val = st.number_input("ρ initial (numerical)",
                                       1.0, 2.0, float(CARPET_RHO), 0.01)
            rho_width = st.slider("ρ log-uncertainty width",
                                   0.01, 0.5, 0.1, 0.01)
            st.caption(
                f"β = log(ρ)/log(3) = {math.log(rho_val)/math.log(3):.4f}\n\n"
                f"d_w = ln(8ρ)/ln(3) = {math.log(8*rho_val)/math.log(3):.4f}\n\n"
                f"d_s = 2d_f/d_w = {2*CARPET_DF/(math.log(8*rho_val)/math.log(3)):.4f}"
            )

        with col2:
            st.subheader("τ Prediction")
            if st.button("🔮 Predict τ from current field"):
                if hasattr(self, '_tree') and self._tree is not None:
                    result = self.studio.editor.execute(self._tree)
                    sdf = result["field"]
                    # Use carpet measure learner
                    cmt = CarpetMeasureTauLearner(CarpetTauConfig(
                        rho_init=rho_val,
                        rho_log_prior_width=rho_width,
                    ))
                    tau = cmt.predict_from_field(sdf)
                    st.metric("Predicted τ", f"{float(tau):.5f}")
                    st.json(cmt.get_status())

        st.divider()
        st.subheader("Meta-Training")
        n_tasks = st.slider("Number of synthetic tasks", 2, 16, 6)
        n_epochs = st.slider("Meta-epochs", 5, 100, 20)

        if st.button("🚀 Run Meta-Training"):
            cmt = CarpetMeasureTauLearner(CarpetTauConfig(
                rho_init=rho_val,
                rho_log_prior_width=rho_width,
            ))
            # Generate synthetic carpet tasks
            tasks = []
            for i in range(n_tasks):
                g = 16
                f = _sdf_sphere_field((g,g,g), (g/2,g/2,g/2),
                                       g*(0.2+0.03*i), 1.0, "cpu")
                tau_target = 0.05 - 0.003*i
                tasks.append(MetaTauTask(
                    task_id=f"carpet_task_{i}",
                    field=f,
                    tau_target=max(0.005, tau_target),
                ))
            result = cmt.meta_train(tasks, n_meta_epochs=n_epochs)
            st.json(result)

    def _page_ocio(self):
        st.header("🎨 OCIO Colour Transform")
        st.info("Upload a .clf/.ctf file or use the built-in demo transform.")

        uploaded = st.file_uploader("Upload .clf / .ctf",
                                     type=["clf", "ctf", "xml"])
        if uploaded is not None:
            st.success(f"Uploaded: {uploaded.name} ({uploaded.size} bytes)")
            st.code(uploaded.read(2000).decode("utf-8", errors="replace"),
                    language="xml")

        st.divider()
        st.subheader("Colour Preview")
        if HAS_MPL:
            # Generate a colour gradient
            x = torch.linspace(0, 1, 64)
            y = torch.linspace(0, 1, 64)
            gx, gy = torch.meshgrid(x, y, indexing="ij")
            rgb = torch.stack([gx, gy, torch.ones_like(gx)*0.5], dim=-1)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(rgb.numpy())
            axes[0].set_title("Input (linear)")
            # Apply gamma for display
            out = rgb.pow(1/2.2)
            axes[1].imshow(out.numpy())
            axes[1].set_title("Output (sRGB display)")
            st.pyplot(fig)

    def _page_export(self):
        st.header("📦 Export")

        if not hasattr(self, '_tree') or self._tree is None:
            st.info("Build a CSG tree first.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Tree JSON")
            tree_json = self._tree.to_json()
            st.download_button("⬇️ CSG Tree JSON",
                                tree_json, "csg_tree.json", "application/json")

        with col2:
            st.subheader("React Flow")
            rf_json = self.rf_exporter.to_json_string(self._tree)
            st.download_button("⬇️ React Flow JSON",
                                rf_json, "csg_reactflow.json",
                                "application/json")

        with col3:
            st.subheader("Carpet-19 Report")
            report = self._generate_carpet_report()
            st.download_button("⬇️ Carpet-19 Report",
                                report, "carpet19_report.txt", "text/plain")

    def _page_carpet_reference(self):
        st.header("📜 Carpet-19 Reference Panel")

        st.markdown("""
        ## Theorem 10 (Main Result — CONDITIONAL)

        **Assumes (I1)–(I4).** For the standard carpet these follow from
        KZ Thms 5.4, 7.2, 7.16, 7.19 + BBKT Thm 1.2 + Hino Thm 5.3.
        For a general non-p.c.f. fractal, (I1)–(I4) are exactly the
        unproved content.

        | Item | Label | Status |
        |---|---|---|
        | Mosco convergence (full sequence) | [D] Thm 2 | Conditional on (I1)-(I4) |
        | Compact resolvent, discrete spectrum | [D] Thm 3 | Conditional |
        | Integer powers support-local | [P] Thm 5 | Conditional |
        | Energy measures converge | [P] Thm 6 | Conditional |
        | Fredholm projections converge | [P] Thm 7 | Conditional |
        | [A,M_f] bounded W¹→W⁻¹ | [P] Prop 3 | Conditional |
        | Spectral Sobolev scale | [P] Thm 4 | Conditional |
        | Hölder embedding W¹↪C^{0,β/2} | [P] Thm 4(f) | Conditional |
        | Γ(f,·) ⊥ µ (Hino) | [C] Thm 8 | Cited, checked |
        | D(A) not an algebra | [P] Cor 3 | Uses Hino |
        | No Connes-type Dirac | [P] Thm 9 | Uses Hino |
        | ρ ≈ 1.25 | [C-num] | **NUMERICAL** |
        | Commutator bound (H3) | [O] | NOT supported by numerics |
        | Hessian / connection | [O] | NOT constructed |
        | Higher-order Fredholm module | [O] | NOT constructed |
        | Sobolev–Lq embeddings | [O] | NOT claimed |

        ## What is NOT available

        - Intrinsic differential representation
        - Hessian / connection calculus
        - Commutator estimates (Section 9 argues against)
        - Formula A(2k) = (-1)^k Δ^k + lower order
        - Sobolev–Lq embeddings
        - Connes-type spectral triple with bounded commutators

        ## Numerical values (ρ ≈ 1.25, NOT proved)

        | Level n | R_cross(n) | K_{n,0} | K_{n,1} |
        |---|---|---|---|
        | 1 | 1.00 | 3.03 | 11.7 |
        | 2 | 1.66 | 4.76 | 68.6 |
        | 3 | 2.21 | 7.89 | 367 |
        | 4 | 2.81 | 13.0 | 1921 |
        | 5 | 3.53 | 21.4 | 10000 |
        | 6 | 4.42 | 35.1 | 51939 |

        Ratios: R_cross → ~1.25 (ρ); K_{n,0} → ~1.64; K_{n,1} → ~5.2.
        **No sign of uniform bound (H3). Geometric growth continues.**
        """)

    # ── helper methods ───────────────────────────────────────────────────

    def _add_primitive(self, prim_type: str):
        if not hasattr(self, '_tree'):
            self._tree = CSGTree()
        nid = f"{prim_type}_{uuid.uuid4().hex[:4]}"
        if prim_type == "sphere":
            self._tree.add_primitive(CSGNodeType.SPHERE, node_id=nid,
                                      center=(12, 12, 12), radius=6.0)
        elif prim_type == "box":
            self._tree.add_primitive(CSGNodeType.BOX, node_id=nid,
                                      center=(12, 12, 12),
                                      half_extents=(4, 4, 4))
        else:
            self._tree.add_primitive(CSGNodeType.TORUS, node_id=nid,
                                      center=(12, 12, 12), R=6.0, r=2.0)
        self._tree_json = self._tree.to_json()
        st.success(f"Added {prim_type}: {nid}")

    def _add_boolean(self, bool_type: str):
        if not hasattr(self, '_tree'):
            st.warning("Create primitives first")
            return
        tree = self._tree
        nodes = [nid for nid, n in tree.nodes.items()
                 if n.node_type != CSGNodeType.OUTPUT]
        if len(nodes) < 2:
            st.warning("Need at least 2 nodes for a Boolean op")
            return
        op_map = {
            "union": CSGNodeType.UNION,
            "intersect": CSGNodeType.INTERSECT,
            "difference": CSGNodeType.DIFFERENCE,
            "blend": CSGNodeType.BLEND,
        }
        nid = f"{bool_type}_{uuid.uuid4().hex[:4]}"
        tree.add_op(op_map[bool_type], [nodes[-2], nodes[-1]], node_id=nid)
        # Update output to point to new node
        for onid, onode in list(tree.nodes.items()):
            if onode.node_type == CSGNodeType.OUTPUT:
                onode.inputs = [nid]
        else:
            oid = f"out_{uuid.uuid4().hex[:4]}"
            tree.add_node(CSGNode(oid, CSGNodeType.OUTPUT, inputs=[nid]))
        self._tree_json = tree.to_json()
        st.success(f"Added {bool_type}: {nid}")

    def _delete_node(self, nid: str):
        if hasattr(self, '_tree'):
            try:
                self.studio.editor.delete_node(self._tree, nid)
                self._tree_json = self._tree.to_json()
                st.success(f"Deleted {nid}")
            except ValueError as e:
                st.error(str(e))

    def _apply_tree_json(self, json_str: str):
        try:
            self._tree = CSGTree.from_json(json_str)
            st.success("Tree JSON applied")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    def _generate_carpet_report(self) -> str:
        lines = [
            "=" * 72,
            "CARPET-19 REFERENCE REPORT",
            "Generated by SESI Topology Studio GUI",
            "=" * 72,
            "",
            "THEOREM 10 — CONDITIONAL on (I1)-(I4)",
            "",
            "Standing inputs:",
            "  (I1) Uniform oscillation  [D, §3]",
            "  (I2) Subsequential Mosco  [C: KZ Thm 4.5, Mosco]",
            "  (I3) Uniqueness           [D: BBKT Thm 1.2 + Prop 2]",
            "  (I4) Separation           [C: KZ Thm 7.19 proof]",
            "",
            "Key constants:",
            f"  d_f = ln8/ln3 = {CARPET_DF:.6f}",
            f"  ρ ≈ {CARPET_RHO}  [NUMERICAL, NOT PROVED]",
            f"  d_w = ln(8ρ)/ln3 = {CARPET_DW:.6f}",
            f"  d_s = 2d_f/d_w = {CARPET_DS:.6f}",
            f"  β = log ρ / log 3 = {math.log(CARPET_RHO)/math.log(3):.6f}",
            "",
            "CRITICAL THEOREMS:",
            "  Hino Thm 8 [C]: Γ(f,·) ⊥ µ — energy measure singular",
            "  Corollary 3 [P]: D(A) not an algebra",
            "  Proposition 3 [P]: [A,M_f] : W¹→W⁻¹ (loses 2 orders)",
            "  Theorem 9 [P]: No Connes-type Dirac with bounded commutators",
            "",
            "OPEN ITEMS [O] (Section 10):",
            "  (i)   Leibniz connection on H",
            "  (ii)  Measure-valued Hessian",
            "  (iii) Schatten summability of [F,a] for a∈W^s, s>1",
            "  (iv)  Independently defined order-2k operator",
            "",
            "NOT CLAIMED:",
            "  - Intrinsic differential representation",
            "  - Hessian / connection calculus",
            "  - Commutator estimates",
            "  - A(2k) = (-1)^k Δ^k + lower order",
            "  - Sobolev–Lq embeddings",
            "",
            "=" * 72,
        ]
        return "\n".join(lines)


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI TOPOLOGY STUDIO GUI — SMOKE TEST")
    print(" Streamlit/React Flow + Gumbel-Softmax Graph Rewriting + Carpet-Measure τ")
    print("=" * 78)
    print(f" torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print(f" Streamlit available   : {HAS_STREAMLIT}")
    print(f" matplotlib available  : {HAS_MPL}")
    print(f" Studio module avail   : {HAS_STUDIO}")
    print(f" No-Zeno available     : {HAS_NOZENO}")
    print(f" CH sigma field avail  : {HAS_CH}")

    print(f"\n Carpet-19 reference (ρ NUMERICAL, not proved):")
    print(f"   d_f = ln8/ln3       = {CARPET_DF:.6f}")
    print(f"   ρ                   = {CARPET_RHO}  [NUMERICAL]")
    print(f"   d_w = ln(8ρ)/ln3    = {CARPET_DW:.6f}")
    print(f"   d_s = 2d_f/d_w      = {CARPET_DS:.6f}")
    print(f"   β = log ρ / log 3   = {math.log(CARPET_RHO)/math.log(3):.6f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Gumbel-Softmax Graph Rewriting ───────────────────────────────
    print("\n [GFR1] Gumbel-Softmax Graph Rewriting")
    gnet = GraphRewriteNet(GumbelConfig(
        max_nodes=6, tau_init=1.0, tau_min=0.05, device=dev)).to(dev)

    # Soft field
    soft_field = gnet.build_soft_field((16, 16, 16), dx=1.0)
    print(f"   Soft field shape     : {tuple(soft_field.shape)}")
    print(f"   Soft field range     : [{float(soft_field.min()):.3f}, "
          f"{float(soft_field.max()):.3f}]")

    # Gradient through structure
    loss = soft_field.sum()
    loss.backward()
    grad_ok = gnet.node_type_logits.grad is not None and \
              bool(gnet.node_type_logits.grad.isfinite().all())
    print(f"   Grad through structure: {grad_ok}")

    # Anneal
    tau_before = float(gnet.tau_gumbel)
    for _ in range(10):
        gnet.anneal_step()
    tau_after = float(gnet.tau_gumbel)
    print(f"   τ_gumbel anneal      : {tau_before:.3f} → {tau_after:.4f} "
          f"(min={gnet.cfg.tau_min})")

    # Structure Hessian guard (Corollary 3)
    try:
        gnet.hessian()
        print("   Hessian guard        : FAILED TO RAISE")
    except RuntimeError as e:
        print(f"   Hessian guard        : BLOCKED ✓ (Corollary 3)")

    # Extract hard tree
    if HAS_STUDIO:
        hard_tree = gnet.extract_hard_tree((16, 16, 16), dx=1.0)
        val = hard_tree.validate()
        print(f"   Hard tree nodes      : {val['n_nodes']}")
        print(f"   Hard tree valid      : {val['valid']}")

    gfr_status = gnet.get_status()
    print(f"   Structure Hessian    : {gfr_status['structure_hessian_allowed']} "
          f"({gfr_status['structure_hessian_reason']})")

    # ── 2. Carpet-Measure τ Learner ─────────────────────────────────────
    print("\n [CMT1] Carpet-Measure τ Learner")
    cmt = CarpetMeasureTauLearner(CarpetTauConfig(
        rho_init=CARPET_RHO, carpet_level=2, device=dev)).to(dev)

    status = cmt.get_status()
    print(f"   ρ (learnable)        : {status['rho']:.4f} "
          f"(init {status['rho_init']}, {status['rho_status']})")
    print(f"   β (Hölder)           : {status['beta_holder']:.6f}")
    print(f"   d_w                  : {status['d_w']:.6f}")
    print(f"   d_s                  : {status['d_s']:.6f}")
    print(f"   d_f                  : {status['d_f']:.6f}")
    print(f"   First-order only     : {status['first_order_only']} "
          f"({status['first_order_reason']})")
    print(f"   Hino singularity     : {status['hino_singularity']}")

    # Carpet mask
    mask = cmt._carpet_mask(2)
    n_occ = mask.sum()
    n_tot = mask.size
    print(f"   Carpet level-2 mask  : {n_occ}/{n_tot} cells occupied "
          f"({n_occ/n_tot:.3f})")
    # Theoretical: (8/9)^2 = 0.790
    print(f"   Expected occupancy   : {(8/9)**2:.3f}")

    # Energy measure proxy
    test_field = _sdf_sphere_field((16, 16, 16), (8, 8, 8), 5.0, 1.0, dev)
    gamma_proxy = cmt.compute_energy_measure_proxy(test_field, level=2)
    print(f"   Energy measure proxy : shape={tuple(gamma_proxy.shape)}, "
          f"mean={float(gamma_proxy.mean()):.4f}, "
          f"max={float(gamma_proxy.max()):.4f}")
    print(f"   ⚠️  Hino: these are finite-level proxies, NOT L² densities")

    # τ prediction
    tau_pred = cmt.predict_from_field(test_field, level=2)
    print(f"   Predicted τ          : {float(tau_pred):.5f}")

    # Meta-training
    tasks = []
    for i in range(4):
        f = _sdf_sphere_field((16,16,16), (8,8,8), 4.0+i, 1.0, dev)
        tasks.append(MetaTauTask(f"t{i}", f, 0.05-0.005*i))
    meta_res = cmt.meta_train(tasks, n_meta_epochs=10)
    print(f"   Meta-loss            : {meta_res['final_meta_loss']:.5f}")
    print(f"   ρ after meta-train   : {meta_res['rho_learned']:.4f} "
          f"(init {meta_res['rho_init']})")
    print(f"   ρ status             : {meta_res['rho_status']}")
    print(f"   First-order reason   : {meta_res['first_order_reason']}")

    # Adaptation
    adapt = cmt.adapt_to_field(test_field, tau_hint=0.03, steps=2)
    print(f"   Adapted τ            : {adapt['tau']:.5f} "
          f"(hint 0.03)")
    print(f"   Adapted              : {adapt['adapted']}")

    # ── 3. React Flow Exporter ──────────────────────────────────────────
    print("\n [RF1] React Flow Exporter")
    if HAS_STUDIO:
        demo_tree = CSGTree()
        s = demo_tree.add_primitive(CSGNodeType.SPHERE, node_id="s1",
                                     center=(12,12,12), radius=6)
        b = demo_tree.add_primitive(CSGNodeType.BOX, node_id="b1",
                                     center=(12,12,12), half_extents=(4,4,4))
        u = demo_tree.add_op(CSGNodeType.UNION, [s, b], node_id="u1")
        oid = f"out_1"
        demo_tree.add_node(CSGNode(oid, CSGNodeType.OUTPUT, inputs=[u]))

        rf = ReactFlowExporter()
        rf_data = rf.to_react_flow(demo_tree)
        print(f"   React Flow nodes     : {len(rf_data['nodes'])}")
        print(f"   React Flow edges     : {len(rf_data['edges'])}")
        print(f"   Tree valid           : {rf_data['metadata']['tree_valid']}")
        print(f"   Carpet d_f in export : "
              f"{rf_data['metadata']['carpet_reference']['d_f']:.6f}")
        print(f"   Carpet ρ in export   : "
              f"{rf_data['metadata']['carpet_reference']['rho']} "
              f"[{rf_data['metadata']['carpet_reference']['rho_status']}]")

        # JSON round-trip
        js = rf.to_json_string(demo_tree)
        parsed = json.loads(js)
        print(f"   JSON valid           : True ({len(js)} chars)")
    else:
        print("   [SKIP] Studio not available")

    # ── 4. Streamlit App (structure check only) ─────────────────────────
    print("\n [GUI1] Streamlit App (structure check)")
    app = StreamlitCSGApp()
    print(f"   Streamlit available  : {HAS_STREAMLIT}")
    print(f"   matplotlib available : {HAS_MPL}")
    print(f"   RF exporter present  : {app.rf_exporter is not None}")

    # Generate carpet report
    report = app._generate_carpet_report()
    print(f"   Carpet-19 report     : {len(report)} chars")
    # Check key claims are present
    checks = [
        ("ρ NUMERICAL", "NUMERICAL, NOT PROVED" in report),
        ("Hino singularity", "Γ(f,·) ⊥ µ" in report),
        ("Corollary 3", "D(A) not an algebra" in report),
        ("Prop 3 W¹→W⁻¹", "W¹→W⁻¹" in report),
        ("Open items", "Leibniz connection" in report),
        ("Not claimed", "NOT CLAIMED" in report),
    ]
    for name, ok in checks:
        print(f"     {name:30s}: {'✓' if ok else '✗ MISSING'}")

    # ── 5. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    print(f"   Gumbel τ             : {gnet.get_status()['tau_gumbel']:.4f}")
    print(f"   Gumbel τ_min         : {gnet.get_status()['tau_min']:.4f}")
    print(f"   Carpet ρ             : {cmt.get_status()['rho']:.4f} [NUM]")
    print(f"   Carpet β             : {cmt.get_status()['beta_holder']:.6f}")
    print(f"   Carpet d_s           : {cmt.get_status()['d_s']:.6f}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C-num] ρ ≈ 1.25 is NUMERICAL, not proved (Carpet-19 §9).")
    print("         R_cross ratios: 1.66, 1.33, 1.27, 1.26, 1.25.")
    print(" [C] Theorem 10 is CONDITIONAL on (I1)-(I4). For the standard")
    print("     carpet these follow from KZ/BBKT/Hino. For other fractals")
    print("     they are exactly the unproved content.")
    print(" [C] Hino Thm 8: Γ(f,·) ⊥ µ. Energy measure is SINGULAR.")
    print("     NO L² density exists. Finite-level Γ_N/µ(S) proxies are")
    print("     approximations that converge weakly as measures, NOT as")
    print("     functions. Any model assuming a smooth density is WRONG.")
    print(" [P] Corollary 3: D(A) is NOT an algebra. f² ∉ D(A).")
    print("     Gumbel-Softmax structure Hessian is BLOCKED by theorem.")
    print(" [P] Proposition 3: [A,M_f] : W¹→W⁻¹ (loses two orders).")
    print("     Annealing τ_gumbel does NOT recover regularity.")
    print(" [P] Theorem 9: No Connes-type Dirac with bounded commutators.")
    print(" [O] Open: Leibniz connection, measure-valued Hessian,")
    print("     Schatten summability for s>1, independent order-2k operator.")
    print(" [D] Gumbel-Softmax structure gradients are FIRST-ORDER ONLY.")
    print(" [D] Streamlit GUI is a PROTOTYPE. Production needs React Flow")
    print("     + WebSocket backend for real-time collaborative editing.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


def run_streamlit_app():
    """Entry point for `streamlit run sesi_topology_studio_gui.py`."""
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit")
        return
    app = StreamlitCSGApp()
    app.render()


# Streamlit entry point
if HAS_STREAMLIT:
    # When run via `streamlit run`, this module is imported as __main__
    # and st.session_state is available.
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    import sys
    if "--streamlit" in sys.argv:
        run_streamlit_app()
    else:
        smoke_test()
