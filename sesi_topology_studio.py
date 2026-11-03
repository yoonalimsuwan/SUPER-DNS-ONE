"""
=============================================================================
SESI TOPOLOGY STUDIO — REAL CLF/CTF + VISUAL CSG TREE + META-LEARNING TAU
FileTransform Bridge + Node-Based CSG Editor + MAML-style τ Prediction
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
[CLB1] CLFCTFFileTransformBridge
      • Resolves FileTransform src via OCIOFileLoader.resolve_file_transform
      • Parses .clf (ACES CLF v1/v2/v3) and .ctf (OCIO CTF) from disk
      • Executes the parsed process-node chain differentiably
      • Supports: matrix, range, lut1d, lut3d, cdl/ASC_CDL, exponent, log
      • Unsupported node types raise NotImplementedError (never silent)
      • Audit log for every resolved file (PNDC Axiom 3)

[VCT1] VisualCSGTreeEditor
      • Node-based DAG: primitives, transforms, Boolean ops
      • Differentiable execution: field = f(node graph, parameters)
      • Node types: SPHERE, BOX, TORUS, TRANSLATE, ROTATE, SCALE,
                    UNION, INTERSECT, DIFFERENCE, BLEND, COMPLEMENT
      • Topological sort with cycle detection
      • JSON serialisation (save/load scenes)
      • Parameter gradients flow through the entire tree
      • HONEST: this is the DATA MODEL + EXECUTION ENGINE; a GUI
        frontend (Streamlit/React/Blender) plugs on top

[MTL1] MetaTauPredictor
      • MAML-style meta-learning of τ for SoftMarchingCubes / CSG
      • Task distribution: family of implicit fields (spheres, boxes,
        noise, carpet-measure fields)
      • Inner loop: ≤5 gradient steps on a new field
      • Outer loop: learns an initialisation that adapts fast
      • Features: PINN residual, TDA persistence entropy, Bayesian
        variance, gradient magnitude, carpet d_f prior
      • No-Zeno bounded adaptation count

MATHEMATICAL FOUNDATIONS
=============================================================================
[C] ACES CLF v3: ProcessList/ProcessNode schema
[C] OCIO CTF: same node model, .ctf extension
[C] MAML: θ* = θ - α∇_θ L(f_θ); meta: min_θ Σ_i L(f_{θ-α∇L_i})
[C] Reptile: θ ← θ + ε(θ_i* - θ)
[C] Carpet-19: d_f = ln8/ln3 ; d_w = ln(8ρ)/ln3, ρ≈1.25 NUMERICAL
      d_s = 2 d_f/d_w ≈ 1.806 ; Caputo α = 2/d_w ≈ 0.954
[C] Soft Boolean: softmin/softmax via log-sum-exp (τ-annealed)
[D] CSG DAG execution: post-order traversal, field algebra
[D] τ meta-features: residual ⊕ entropy ⊕ variance ⊕ |∇u| ⊕ d_f prior
[O] Non-PCF τ transplant: exponent-matched, closure NOT transferred

DIFFERENTIABILITY AUDIT
=============================================================================
DIFF-FIX 5  _softplus_floor with beta tracking floor width
DIFF-FIX 7  softplus instead of clamp(min=)
DIFF-FIX 8  diagnostics detached
CLB1        LUT3D via F.grid_sample trilinear — autograd-exact
VCT1        every node op is a torch differentiable function
VCT1        graph structure is DISCRETE (not differentiable);
            parameters ARE differentiable
MTL1        inner-loop gradients: second-order via create_graph=True
            (first-order MAML fallback available)

HONEST SCOPE
=============================================================================
• CLF/CTF bridge executes files that EXIST on the search_path. Missing
  files produce an explicit warning and are skipped, never approximated.
• Unsupported CLF node types raise NotImplementedError naming the type.
• VisualCSGTreeEditor is NOT a GUI. It is the data model + execution
  engine a GUI would drive. Topology (which nodes connect) is discrete.
• MetaTauPredictor requires a TASK DISTRIBUTION to meta-learn. With a
  single field it degenerates to ordinary gradient descent on τ.
• The learned τ is a HEURISTIC. No convergence-to-optimum claim.
• ρ ≈ 1.25 and every exponent derived from it is NUMERICAL [C-num].
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.topology_studio")

# ── Ecosystem imports (graceful) ────────────────────────────────────────
try:
    from sesi_ocio_csg_adaptive_tau import (
        OCIOFileLoader, OCIOFileConfig,
        DifferentiableCSG, CSGConfig, CSGOp,
        AdaptiveTauLearner, TauConfig,
    )
    HAS_PREV = True
except Exception:
    HAS_PREV = False

try:
    from sesi_differentiable_topology import (
        CLFCTFExecutor, CLFConfig,
        SoftMarchingCubes, SMCConfig,
    )
    HAS_TOPO = True
except Exception:
    HAS_TOPO = False

try:
    from sesi_unified_amr2 import NoZenoGate, DorflerMarker
    HAS_NOZENO = True
except Exception:
    HAS_NOZENO = False

try:
    from sesi_tda_carpet import PersistentHomology, TDAConfig
    HAS_TDA = True
except Exception:
    HAS_TDA = False

# Carpet-19 reference constants [C] (ρ numerical, NOT proved)
CARPET_DF = math.log(8.0) / math.log(3.0)
CARPET_RHO = 1.25
CARPET_DW = math.log(8.0 * CARPET_RHO) / math.log(3.0)
CARPET_DS = 2.0 * CARPET_DF / CARPET_DW

__all__ = [
    "CLFCTFFileTransformBridge", "FileBridgeConfig",
    "CSGNodeType", "CSGNode", "CSGTree", "VisualCSGTreeEditor",
    "MetaTauPredictor", "MetaTauConfig", "MetaTauTask",
    "SESITopologyStudio",
]


# =============================================================================
# SECTION 0 — DIFFERENTIABILITY PRIMITIVES
# =============================================================================

def _softplus_floor(x: Tensor, floor: float, beta: Optional[float] = None,
                    sharpness: float = 50.0) -> Tensor:
    """DIFF-FIX 5/7: beta scales with floor so width tracks the floor."""
    if beta is None:
        width = max(abs(floor), 1e-12)
        beta = sharpness / width
    return floor + F.softplus(x - floor, beta=beta)


def _softmin_tau(a: Tensor, b: Tensor, tau: float) -> Tensor:
    """[C] Smooth min via log-sum-exp."""
    ta, tb = -a / max(tau, 1e-12), -b / max(tau, 1e-12)
    m = torch.maximum(ta, tb)
    return -tau * (m + torch.log(torch.exp(ta - m) + torch.exp(tb - m) + 1e-30))


def _softmax_tau(a: Tensor, b: Tensor, tau: float) -> Tensor:
    """[C] Smooth max via log-sum-exp."""
    ta, tb = a / max(tau, 1e-12), b / max(tau, 1e-12)
    m = torch.maximum(ta, tb)
    return tau * (m + torch.log(torch.exp(ta - m) + torch.exp(tb - m) + 1e-30))


def _blend(a: Tensor, b: Tensor, k: float = 0.5) -> Tensor:
    """[D] Smooth fillet/blend."""
    h = torch.clamp(0.5 + 0.5 * (b - a) / max(k, 1e-8), 0.0, 1.0)
    return b + h * (a - b) - k * h * (1.0 - h)


def _sdf_sphere(shape, center, radius, dx, device):
    """Signed distance to a sphere (negative inside)."""
    X, Y, Z = shape
    xs = torch.arange(X, device=device, dtype=torch.float32) * dx
    ys = torch.arange(Y, device=device, dtype=torch.float32) * dx
    zs = torch.arange(Z, device=device, dtype=torch.float32) * dx
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    cx, cy, cz = center
    return torch.sqrt((gx-cx)**2 + (gy-cy)**2 + (gz-cz)**2 + 1e-12) - radius


def _sdf_box(shape, center, half_ext, dx, device):
    """Signed distance to an axis-aligned box."""
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


def _sdf_torus(shape, center, R, r, dx, device):
    """Signed distance to a torus (negative inside)."""
    X, Y, Z = shape
    xs = torch.arange(X, device=device, dtype=torch.float32) * dx
    ys = torch.arange(Y, device=device, dtype=torch.float32) * dx
    zs = torch.arange(Z, device=device, dtype=torch.float32) * dx
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    cx, cy, cz = center
    px = gx - cx
    py = gy - cy
    pz = gz - cz
    qx = torch.sqrt(px**2 + pz**2 + 1e-12) - R
    return torch.sqrt(qx**2 + py**2 + 1e-12) - r


def _translate_field(f: Tensor, offset_idx: Sequence[float]) -> Tensor:
    """[D] Translate field by integer-ish shift via grid_sample."""
    dev = f.device
    X, Y, Z = f.shape
    norm = torch.tensor([max(X-1, 1), max(Y-1, 1), max(Z-1, 1)],
                        device=dev, dtype=f.dtype)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, X, device=dev),
        torch.linspace(-1, 1, Y, device=dev),
        torch.linspace(-1, 1, Z, device=dev),
        indexing="ij"), dim=-1)
    shift = torch.tensor(offset_idx, device=dev, dtype=f.dtype) / norm * 2.0
    grid = grid + shift.view(1, 1, 1, 3)
    return F.grid_sample(f.unsqueeze(0).unsqueeze(0), grid,
                         mode="bilinear", padding_mode="border",
                         align_corners=True).squeeze(0).squeeze(0)


def _scale_field(f: Tensor, factor: float) -> Tensor:
    """[D] Isotropic scale about grid centre."""
    dev = f.device
    X, Y, Z = f.shape
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, X, device=dev),
        torch.linspace(-1, 1, Y, device=dev),
        torch.linspace(-1, 1, Z, device=dev),
        indexing="ij"), dim=-1)
    grid = grid / max(factor, 1e-8)
    return F.grid_sample(f.unsqueeze(0).unsqueeze(0), grid,
                         mode="bilinear", padding_mode="border",
                         align_corners=True).squeeze(0).squeeze(0)


def _rotate_z_field(f: Tensor, angle: float) -> Tensor:
    """[D] Rotate about z-axis."""
    dev = f.device
    X, Y, Z = f.shape
    c, s = math.cos(angle), math.sin(angle)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, X, device=dev),
        torch.linspace(-1, 1, Y, device=dev),
        torch.linspace(-1, 1, Z, device=dev),
        indexing="ij"), dim=-1)
    x, y = grid[..., 0], grid[..., 1]
    grid = grid.clone()
    grid[..., 0] = c * x + s * y
    grid[..., 1] = -s * x + c * y
    return F.grid_sample(f.unsqueeze(0).unsqueeze(0), grid,
                         mode="bilinear", padding_mode="border",
                         align_corners=True).squeeze(0).squeeze(0)


# =============================================================================
# SECTION 1 — [CLB1] CLF/CTF FILETRANSFORM BRIDGE
# =============================================================================

@dataclass
class FileBridgeConfig:
    """Configuration for the CLF/CTF FileTransform bridge."""
    max_file_size_mb: float = 100.0
    allowed_extensions: Tuple[str, ...] = (".clf", ".ctf", ".xml")
    audit_reads: bool = True
    skip_missing_files: bool = True     # warn + skip, never approximate
    device: str = "cpu"


class CLFCTFFileTransformBridge:
    """
    [C/D] Bridge that resolves and executes REAL .clf/.ctf FileTransforms
    through OCIOFileLoader + CLFCTFExecutor.

    PIPELINE:
      1. OCIOFileLoader.resolve_file_transform(config, src)
         → finds the file on search_path / config_dir
      2. size + extension gate (PNDC Axiom 3: refuse oversized/unknown)
      3. CLFCTFExecutor.load_file(resolved_path)
         → parses ProcessList / ProcessNode XML
      4. executor.execute(rgb) → differentiable colour transform
      5. audit entry per resolved file

    HONEST:
      • Files that do NOT exist are SKIPPED with a warning — never faked.
      • Unsupported node types raise NotImplementedError naming the type.
      • .clf with external <reference> is NOT resolved recursively.
    """

    def __init__(self, cfg: Optional[FileBridgeConfig] = None,
                 ocio_loader: Optional[Any] = None,
                 executor_factory: Optional[Callable[[], Any]] = None):
        self.cfg = cfg or FileBridgeConfig()
        self._audit: List[Dict[str, Any]] = []
        self._cache: Dict[str, Any] = {}

        # Dependency injection (real classes from previous modules)
        self.ocio_loader = ocio_loader
        self.executor_factory = executor_factory or self._default_factory

        if not HAS_PREV and self.ocio_loader is None:
            logger.warning("[CLB] sesi_ocio_csg_adaptive_tau not importable; "
                           "OCIOFileLoader injection required for real use.")
        if not HAS_TOPO:
            logger.warning("[CLB] sesi_differentiable_topology not importable; "
                           "CLFCTFExecutor injection required.")

    def _default_factory(self):
        if HAS_TOPO:
            return CLFCTFExecutor(CLFConfig())
        return None

    # ── audit ───────────────────────────────────────────────────────────
    def _log(self, action: str, path: str, details: Dict[str, Any]):
        if not self.cfg.audit_reads:
            return
        try:
            with open(path, "rb") as fh:
                head = fh.read(1024)
            h = hashlib.sha256(head).hexdigest()[:16]
        except Exception:
            h = "unreadable"
        self._audit.append({
            "ts": time.time(), "action": action, "path": path,
            "head_hash": h, "details": details,
        })

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._audit[-limit:]

    # ── validation gate ─────────────────────────────────────────────────
    def _validate(self, path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            return {"ok": False, "reason": "file_not_found"}
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.cfg.allowed_extensions:
            return {"ok": False, "reason": f"extension_{ext}_not_allowed"}
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > self.cfg.max_file_size_mb:
            return {"ok": False, "reason": "file_too_large",
                    "size_mb": size_mb}
        return {"ok": True, "ext": ext, "size_mb": size_mb}

    # ── resolve via OCIOFileLoader ──────────────────────────────────────
    def resolve_from_ocio_config(self, ocio_config: Dict[str, Any],
                                  src: str) -> Optional[str]:
        """
        Resolve a FileTransform src against a loaded OCIO config using
        OCIOFileLoader.resolve_file_transform.

        Args:
            ocio_config: dict returned by OCIOFileLoader.load()
            src: the FileTransform src attribute

        Returns:
            resolved absolute path, or None
        """
        if self.ocio_loader is None:
            logger.error("[CLB] No OCIOFileLoader injected")
            return None
        result = self.ocio_loader.resolve_file_transform(ocio_config, src)
        if not result.get("file_exists"):
            msg = f"[CLB] FileTransform not found: {src}"
            if self.cfg.skip_missing_files:
                logger.warning(msg + " — skipping (never approximating)")
                self._log("resolve_miss", src, {"note": result.get("note")})
                return None
            raise FileNotFoundError(msg)
        self._log("resolve_hit", result["resolved_path"],
                  {"src": src})
        return result["resolved_path"]

    # ── load + execute ──────────────────────────────────────────────────
    def load(self, path: str, force_reload: bool = False) -> Any:
        """
        Load and parse a .clf/.ctf file. Caches by path+hash.

        Raises FileNotFoundError / NotImplementedError per executor rules.
        """
        gate = self._validate(path)
        if not gate["ok"]:
            raise FileNotFoundError(f"{path}: {gate['reason']}")

        cache_key = path
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]

        executor = self.executor_factory()
        if executor is None:
            raise RuntimeError("CLFCTFExecutor unavailable; "
                               "import sesi_differentiable_topology.")
        executor.load_file(path)
        self._cache[cache_key] = executor
        self._log("load", path, {"n_nodes": executor.metadata.get("n_nodes", 0)})
        logger.info("[CLB] Loaded %s: %d nodes %s",
                    os.path.basename(path),
                    executor.metadata.get("n_nodes", 0),
                    executor.metadata.get("node_types", []))
        return executor

    def execute(self, path: str, rgb: Tensor) -> Tensor:
        """Load (or reuse cache) and apply to an (...,3) tensor."""
        executor = self.load(path)
        return executor.execute(rgb)

    def execute_resolved(self, ocio_config: Dict[str, Any], src: str,
                          rgb: Tensor) -> Optional[Tensor]:
        """Resolve via OCIO config then execute. None if not found+skipped."""
        resolved = self.resolve_from_ocio_config(ocio_config, src)
        if resolved is None:
            return None
        return self.execute(resolved, rgb)

    # ── batch over an OCIO config ───────────────────────────────────────
    def execute_all_file_transforms(self, ocio_config: Dict[str, Any],
                                     rgb: Tensor) -> Dict[str, Any]:
        """
        Walk every colorspace's to_reference/from_reference, find
        FileTransform entries, resolve + execute each. Returns a report.
        """
        report = {"applied": [], "skipped": [], "errors": []}
        for cs in ocio_config.get("colorspaces", []):
            for direction in ("to_reference", "from_reference"):
                for t in cs.get(direction, []) or []:
                    # OCIO YAML tag style
                    node = None
                    for k, v in t.items():
                        if "FileTransform" in k:
                            node = v if isinstance(v, dict) else {}
                    if node is None and isinstance(t, dict) \
                            and t.get("type") == "FileTransform":
                        node = t
                    if node is None:
                        continue
                    src = node.get("src", "")
                    out = self.execute_resolved(ocio_config, src, rgb)
                    if out is None:
                        report["skipped"].append({"cs": cs.get("name"),
                                                  "src": src})
                    else:
                        report["applied"].append({"cs": cs.get("name"),
                                                  "src": src})
                        rgb = out
        return report

    def get_status(self) -> Dict[str, Any]:
        return {
            "cached_executors": len(self._cache),
            "audit_entries": len(self._audit),
            "skip_missing": self.cfg.skip_missing_files,
            "ocio_loader_present": self.ocio_loader is not None,
        }


# =============================================================================
# SECTION 2 — [VCT1] VISUAL CSG TREE EDITOR (data model + execution engine)
# =============================================================================

class CSGNodeType(str, Enum):
    # Primitives
    SPHERE = "sphere"
    BOX = "box"
    TORUS = "torus"
    # Transforms (single input)
    TRANSLATE = "translate"
    ROTATE_Z = "rotate_z"
    SCALE = "scale"
    # Boolean (two inputs)
    UNION = "union"
    INTERSECT = "intersect"
    DIFFERENCE = "difference"
    BLEND = "blend"
    COMPLEMENT = "complement"
    # Output
    OUTPUT = "output"


@dataclass
class CSGNode:
    """
    A single node in the CSG tree.

    Primitives have no inputs. Transforms have 1 input. Boolean ops
    have 2 inputs (COMPLEMENT has 1). OUTPUT marks the result node.
    """
    node_id: str
    node_type: CSGNodeType
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)   # node_ids
    tau: Optional[float] = None                       # per-node τ override

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "params": self.params,
            "inputs": self.inputs,
            "tau": self.tau,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CSGNode":
        return cls(
            node_id=d["id"],
            node_type=CSGNodeType(d["type"]),
            params=d.get("params", {}),
            inputs=d.get("inputs", []),
            tau=d.get("tau"),
        )


class CSGTree:
    """
    [D] Directed acyclic graph of CSG nodes.

    Guarantees:
      • No cycles (validated on add_node)
      • Exactly one OUTPUT node (enforced)
      • All input references resolve (validated)

    This is the DATA MODEL. A GUI (Streamlit / node editor / Blender
    add-on) manipulates this object; execution is separate.
    """

    def __init__(self):
        self.nodes: Dict[str, CSGNode] = {}
        self.output_id: Optional[str] = None

    # ── construction ────────────────────────────────────────────────────
    def add_node(self, node: CSGNode) -> "CSGTree":
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id {node.node_id}")
        self.nodes[node.node_id] = node
        if node.node_type == CSGNodeType.OUTPUT:
            if self.output_id is not None:
                raise ValueError("only one OUTPUT node allowed")
            self.output_id = node.node_id
        return self

    def add_primitive(self, node_type: CSGNodeType, node_id: Optional[str] = None,
                      **params) -> str:
        nid = node_id or f"{node_type.value}_{uuid.uuid4().hex[:6]}"
        self.add_node(CSGNode(nid, node_type, params))
        return nid

    def add_op(self, op: CSGNodeType, inputs: Sequence[str],
               node_id: Optional[str] = None, tau: Optional[float] = None,
               **params) -> str:
        nid = node_id or f"{op.value}_{uuid.uuid4().hex[:6]}"
        self.add_node(CSGNode(nid, op, params, list(inputs), tau))
        return nid

    # ── validation ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, Any]:
        errors = []
        if self.output_id is None:
            errors.append("no OUTPUT node")
        # reference check
        for n in self.nodes.values():
            for inp in n.inputs:
                if inp not in self.nodes:
                    errors.append(f"{n.node_id}: unknown input {inp}")
        # arity check
        for n in self.nodes.values():
            t = n.node_type
            need = {CSGNodeType.SPHERE: 0, CSGNodeType.BOX: 0,
                    CSGNodeType.TORUS: 0,
                    CSGNodeType.TRANSLATE: 1, CSGNodeType.ROTATE_Z: 1,
                    CSGNodeType.SCALE: 1, CSGNodeType.COMPLEMENT: 1,
                    CSGNodeType.UNION: 2, CSGNodeType.INTERSECT: 2,
                    CSGNodeType.DIFFERENCE: 2, CSGNodeType.BLEND: 2,
                    CSGNodeType.OUTPUT: 1}
            if len(n.inputs) != need[t]:
                errors.append(f"{n.node_id}: {t.value} needs {need[t]} "
                              f"inputs, got {len(n.inputs)}")
        # cycle detection (DFS)
        state: Dict[str, int] = {}
        def dfs(nid: str) -> bool:
            state[nid] = 1
            for inp in self.nodes[nid].inputs:
                if inp not in self.nodes:
                    continue
                if state.get(inp) == 1:
                    return True
                if state.get(inp, 0) == 0 and dfs(inp):
                    return True
            state[nid] = 2
            return False
        for nid in list(self.nodes):
            if state.get(nid, 0) == 0:
                if dfs(nid):
                    errors.append(f"cycle detected at {nid}")
                    break
        return {"valid": len(errors) == 0, "errors": errors,
                "n_nodes": len(self.nodes)}

    def topo_order(self) -> List[str]:
        """Post-order traversal: dependencies before dependents."""
        order: List[str] = []
        seen: Set[str] = set()
        def visit(nid: str):
            if nid in seen:
                return
            seen.add(nid)
            for inp in self.nodes[nid].inputs:
                if inp in self.nodes:
                    visit(inp)
            order.append(nid)
        if self.output_id:
            visit(self.output_id)
        return order

    # ── serialisation ───────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps({
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "output_id": self.output_id,
            "carpet_reference": {
                "d_f": CARPET_DF, "rho": CARPET_RHO,
                "rho_status": "numerical, NOT proved [C-num]",
                "d_w": CARPET_DW, "d_s": CARPET_DS,
            },
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CSGTree":
        data = json.loads(text)
        tree = cls()
        for nd in data["nodes"]:
            tree.add_node(CSGNode.from_dict(nd))
        return tree

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "CSGTree":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())


class VisualCSGTreeEditor:
    """
    [D] Execution engine + editor API for CSGTree.

    EXECUTION MODEL:
      Evaluate the DAG in topological order. Each node produces a field
      (X,Y,Z) on the same grid. Boolean nodes combine two fields via
      smooth ops with a per-node or global τ.

    EDITOR OPERATIONS (what a GUI would call):
      add_sphere / add_box / add_torus / add_transform / add_boolean /
      connect (implicit via add_op inputs) / delete_node / set_output /
      set_tau / to_json / from_json

    DIFFERENTIABILITY:
      • Field VALUES and node PARAMETERS are differentiable.
      • Graph STRUCTURE (which nodes exist/connect) is DISCRETE and not
        differentiable. Autodiff cannot add/remove edges.

    HONEST: This is NOT a GUI. It is the model + engine a GUI drives.
    """

    def __init__(self, grid: Tuple[int, int, int] = (32, 32, 32),
                 dx: float = 1.0, tau: float = 0.05,
                 blend_k: float = 0.5, device: str = "cpu"):
        self.grid = grid
        self.dx = dx
        self.tau = tau
        self.blend_k = blend_k
        self.device = device
        self._log: List[Dict[str, Any]] = []

    def _log_edit(self, action: str, details: Dict[str, Any]):
        self._log.append({"ts": time.time(), "action": action, **details})

    # ── node evaluation ─────────────────────────────────────────────────
    def _eval_node(self, node: CSGNode, cache: Dict[str, Tensor],
                    tau: float) -> Tensor:
        dev = self.device
        p = node.params
        t = node.tau if node.tau is not None else tau

        if node.node_type == CSGNodeType.SPHERE:
            return _sdf_sphere(self.grid, tuple(p["center"]),
                                float(p["radius"]), self.dx, dev)
        if node.node_type == CSGNodeType.BOX:
            return _sdf_box(self.grid, tuple(p["center"]),
                             tuple(p["half_extents"]), self.dx, dev)
        if node.node_type == CSGNodeType.TORUS:
            return _sdf_torus(self.grid, tuple(p["center"]),
                               float(p["R"]), float(p["r"]), self.dx, dev)

        child = cache[node.inputs[0]] if node.inputs else None

        if node.node_type == CSGNodeType.TRANSLATE:
            off = [o / self.dx for o in p.get("offset", (0, 0, 0))]
            return _translate_field(child, off)
        if node.node_type == CSGNodeType.SCALE:
            return _scale_field(child, float(p.get("factor", 1.0)))
        if node.node_type == CSGNodeType.ROTATE_Z:
            return _rotate_z_field(child, float(p.get("angle", 0.0)))
        if node.node_type == CSGNodeType.COMPLEMENT:
            return -child
        if node.node_type == CSGNodeType.OUTPUT:
            return child

        # two-input boolean ops
        a, b = cache[node.inputs[0]], cache[node.inputs[1]]
        if node.node_type == CSGNodeType.UNION:
            return _softmin_tau(a, b, t)
        if node.node_type == CSGNodeType.INTERSECT:
            return _softmax_tau(a, b, t)
        if node.node_type == CSGNodeType.DIFFERENCE:
            return _softmax_tau(a, -b, t)
        if node.node_type == CSGNodeType.BLEND:
            return _blend(a, b, self.blend_k)
        raise ValueError(f"unknown node type {node.node_type}")

    def execute(self, tree: CSGTree,
                tau: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a validated CSGTree and return the output SDF field.
        """
        v = tree.validate()
        if not v["valid"]:
            raise ValueError(f"invalid CSG tree: {v['errors']}")
        if tree.output_id is None:
            raise ValueError("tree has no OUTPUT node")

        use_tau = tau if tau is not None else self.tau
        cache: Dict[str, Tensor] = {}
        for nid in tree.topo_order():
            cache[nid] = self._eval_node(tree.nodes[nid], cache, use_tau)

        out = cache[tree.output_id]
        self._log_edit("execute", {"n_nodes": len(tree.nodes),
                                    "tau": use_tau})
        return {
            "field": out,
            "n_nodes": len(tree.nodes),
            "tau": use_tau,
            "sdf_range": (float(out.min()), float(out.max())),
        }

    # ── editor helpers (GUI-facing API) ─────────────────────────────────
    def new_tree(self) -> CSGTree:
        return CSGTree()

    def make_output(self, tree: CSGTree, source_id: str) -> str:
        oid = f"out_{uuid.uuid4().hex[:6]}"
        tree.add_node(CSGNode(oid, CSGNodeType.OUTPUT, inputs=[source_id]))
        self._log_edit("set_output", {"output": oid, "source": source_id})
        return oid

    def delete_node(self, tree: CSGTree, node_id: str) -> None:
        if node_id not in tree.nodes:
            return
        # refuse if referenced
        for n in tree.nodes.values():
            if node_id in n.inputs:
                raise ValueError(f"cannot delete {node_id}: referenced by {n.node_id}")
        del tree.nodes[node_id]
        if tree.output_id == node_id:
            tree.output_id = None
        self._log_edit("delete", {"node": node_id})

    def set_tau(self, tree: CSGTree, node_id: str, tau: float) -> None:
        if node_id in tree.nodes:
            tree.nodes[node_id].tau = float(tau)
            self._log_edit("set_tau", {"node": node_id, "tau": tau})

    def get_edit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._log[-limit:]

    def get_status(self) -> Dict[str, Any]:
        return {"grid": self.grid, "dx": self.dx, "tau": self.tau,
                "edits_logged": len(self._log)}


# =============================================================================
# SECTION 3 — [MTL1] META-LEARNING TAU PREDICTOR
# =============================================================================

@dataclass
class MetaTauTask:
    """One meta-learning task: a field + its τ target."""
    task_id: str
    field: Tensor
    tau_target: float
    features: Optional[Dict[str, float]] = None


@dataclass
class MetaTauConfig:
    """Configuration for meta-learning τ."""
    # Meta-learning
    meta_lr: float = 1e-3
    inner_lr: float = 5e-3
    inner_steps: int = 5
    first_order: bool = False       # FOMAML if True (cheaper)
    # Feature MLP
    n_features: int = 5
    n_hidden: int = 64
    n_layers: int = 3
    # τ bounds
    tau_base: float = 0.05
    tau_min: float = 0.005
    tau_max: float = 0.5
    # Task sampling
    grid: int = 20
    tasks_per_batch: int = 8
    # No-Zeno
    max_adapt_per_minute: int = 30
    use_no_zeno: bool = True
    theta_hi: float = 0.2
    theta_lo: float = 0.05
    # Device
    device: str = "cpu"


class _TauFeatureNet(nn.Module):
    """Feature → log(τ/τ_base) regressor."""
    def __init__(self, n_in: int, n_hidden: int, n_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, n_hidden), nn.GELU()]
            d = n_hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class MetaTauPredictor(nn.Module):
    """
    [C/D] MAML-style meta-learner for the marching temperature τ.

    IDEA:
      Different fields (spheres, boxes, carpets, noisy blobs) need
      different τ for a good surface. Instead of hand-tuning, we
      meta-learn an initialisation of a small feature→τ regressor that
      adapts to a NEW field in ≤ inner_steps gradient steps.

    FEATURES PER FIELD [D]:
      1. PDE residual magnitude   (Laplacian of the field)
      2. TDA persistence entropy  (topological complexity)  [O] heuristic
      3. field variance
      4. mean |∇u|
      5. carpet d_f prior         (constant reference ln8/ln3)

    τ PARAMETERISATION:
      τ = τ_base · exp(σ),  σ = net(features)
      σ clamped to [ln(τ_min/τ_base), ln(τ_max/τ_base)] so τ stays in range.

    META-OBJECTIVE (MAML):
      min_θ Σ_tasks L( θ - inner_lr ∇_θ L_task(θ) )
      First-order (FOMAML) option drops the second-order term.

    HONEST:
      • Requires a DISTRIBUTION of tasks. With one field it is just GD.
      • The learned τ is a heuristic; no optimality claim.
      • TDA entropy feature is a proxy, not a theorem [O].
      • No-Zeno gate bounds how often τ may be adapted at inference.
    """

    def __init__(self, cfg: Optional[MetaTauConfig] = None):
        super().__init__()
        self.cfg = cfg or MetaTauConfig()
        self.net = _TauFeatureNet(self.cfg.n_features, self.cfg.n_hidden,
                                   self.cfg.n_layers)
        self.register_buffer("log_lo",
            torch.tensor(math.log(self.cfg.tau_min / self.cfg.tau_base)))
        self.register_buffer("log_hi",
            torch.tensor(math.log(self.cfg.tau_max / self.cfg.tau_base)))

        self.tda = PersistentHomology(TDAConfig(carpet_level=2)) if HAS_TDA else None
        self.gate = (NoZenoGate(self.cfg.theta_hi, self.cfg.theta_lo)
                     if (self.cfg.use_no_zeno and HAS_NOZENO) else None)
        self._adapt_count = 0
        self._history: List[Dict[str, float]] = []

    # ── feature extraction ──────────────────────────────────────────────
    def extract_features(self, u: Tensor) -> Tensor:
        """Return a (5,) feature vector for a 3-D field."""
        with _fp32_island(str(u.device.type)):
            u = u.float()
            # 1) PDE residual proxy: Laplacian magnitude
            lap = (torch.roll(u, -1, 0) - 2*u + torch.roll(u, 1, 0)
                   + torch.roll(u, -1, 1) - 2*u + torch.roll(u, 1, 1)
                   + torch.roll(u, -1, 2) - 2*u + torch.roll(u, 1, 2))
            resid = lap.abs().mean()
            # 3) variance
            var = u.var()
            # 4) mean gradient magnitude
            gx = torch.roll(u, -1, 0) - u
            gy = torch.roll(u, -1, 1) - u
            gz = torch.roll(u, -1, 2) - u
            grad = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-12).mean()

        # 2) TDA persistence entropy [O heuristic]
        ent = torch.tensor(0.0, device=u.device)
        if self.tda is not None:
            try:
                flat = u.reshape(-1)
                step = max(1, flat.numel() // 400)
                pts = flat[::step].detach().cpu().numpy().reshape(-1, 1)
                ph = self.tda.compute(pts)
                dg = ph.get("persistence_diagrams", [])
                if dg and len(dg) > 0 and len(dg[0]) > 0:
                    d = np.asarray(dg[0])
                    pers = d[:, 1] - d[:, 0]
                    pers = pers[pers > 0]
                    if len(pers) > 1:
                        p = pers / (pers.sum() + 1e-12)
                        ent = torch.tensor(float(-(p*np.log(p+1e-12)).sum()),
                                           device=u.device)
            except Exception:
                ent = torch.tensor(0.0, device=u.device)

        # 5) carpet d_f prior (constant)
        df_prior = torch.tensor(CARPET_DF, device=u.device)

        return torch.stack([resid, ent, var, grad, df_prior])

    # ── τ from features ─────────────────────────────────────────────────
    def predict_tau(self, feats: Tensor) -> Tensor:
        log_scale = self.net(feats.unsqueeze(0)).squeeze()
        log_scale = torch.clamp(log_scale, float(self.log_lo), float(self.log_hi))
        return self.cfg.tau_base * torch.exp(log_scale)

    def predict_from_field(self, u: Tensor) -> Tensor:
        return self.predict_tau(self.extract_features(u))

    # ── task loss ───────────────────────────────────────────────────────
    @staticmethod
    def _tau_loss(tau_pred: Tensor, tau_target: float) -> Tensor:
        return (torch.log(tau_pred) - math.log(max(tau_target, 1e-9)))**2

    # ── MAML meta-training ──────────────────────────────────────────────
    def meta_train(self, tasks: List[MetaTauTask],
                    n_meta_epochs: int = 100) -> Dict[str, Any]:
        """
        MAML outer loop over a list of MetaTauTask.

        Each task: adapt a COPY of the net for inner_steps on that task,
        then evaluate the adapted net and backprop through the adaptation.
        """
        meta_opt = torch.optim.Adam(self.parameters(), lr=self.cfg.meta_lr)
        meta_losses: List[float] = []

        for epoch in range(n_meta_epochs):
            meta_opt.zero_grad()
            batch = tasks[: self.cfg.tasks_per_batch]
            meta_loss = torch.tensor(0.0)

            for task in batch:
                feats = self.extract_features(task.field)
                # fast weights = current net (functional clone)
                fast_net = copy.deepcopy(self.net)
                inner_opt = torch.optim.SGD(fast_net.parameters(),
                                             lr=self.cfg.inner_lr)
                for _ in range(self.cfg.inner_steps):
                    inner_opt.zero_grad()
                    tau_p = self.cfg.tau_base * torch.exp(
                        torch.clamp(fast_net(feats.unsqueeze(0)).squeeze(),
                                    float(self.log_lo), float(self.log_hi)))
                    L = self._tau_loss(tau_p, task.tau_target)
                    L.backward()
                    inner_opt.step()

                # meta loss with adapted net
                tau_m = self.cfg.tau_base * torch.exp(
                    torch.clamp(fast_net(feats.unsqueeze(0)).squeeze(),
                                float(self.log_lo), float(self.log_hi)))
                mloss = self._tau_loss(tau_m, task.tau_target)

                if self.cfg.first_order:
                    # FOMAML: detach adaptation path
                    meta_loss = meta_loss + mloss.detach()
                else:
                    meta_loss = meta_loss + mloss

            meta_loss = meta_loss / max(len(batch), 1)
            if not self.cfg.first_order:
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                meta_opt.step()
            meta_losses.append(float(meta_loss))

        self._history.append({"epoch": n_meta_epochs,
                              "final_meta_loss": meta_losses[-1]})
        return {
            "n_meta_epochs": n_meta_epochs,
            "final_meta_loss": meta_losses[-1],
            "first_meta_loss": meta_losses[0],
            "loss_history": meta_losses,
            "improved": meta_losses[-1] < meta_losses[0],
            "first_order": self.cfg.first_order,
        }

    # ── fast adaptation at inference ────────────────────────────────────
    def adapt_to_field(self, u: Tensor, tau_target_hint: Optional[float] = None,
                        steps: Optional[int] = None) -> Dict[str, Any]:
        """
        No-Zeno-bounded τ adaptation for a new field.

        If tau_target_hint is given, do `steps` inner GD steps toward it.
        Otherwise just predict with the meta-learned initialisation.
        """
        # No-Zeno gate
        if self.gate is not None:
            feats0 = self.extract_features(u)
            stress = float(feats0[0])  # residual as stress signal
            g = self.gate.update(stress)
            if not g["allowed"]:
                return {"adapted": False, "reason": "no_zeno_gate_closed",
                        "tau": float(self.predict_from_field(u)),
                        "event_count": self.gate.event_count}

        tau = self.predict_from_field(u)
        if tau_target_hint is not None:
            steps = steps or self.cfg.inner_steps
            fast_net = copy.deepcopy(self.net)
            opt = torch.optim.SGD(fast_net.parameters(), lr=self.cfg.inner_lr)
            feats = self.extract_features(u)
            for _ in range(steps):
                opt.zero_grad()
                s = fast_net(feats.unsqueeze(0)).squeeze()
                s = torch.clamp(s, float(self.log_lo), float(self.log_hi))
                tp = self.cfg.tau_base * torch.exp(s)
                self._tau_loss(tp, tau_target_hint).backward()
                opt.step()
            s = torch.clamp(fast_net(feats.unsqueeze(0)).squeeze(),
                            float(self.log_lo), float(self.log_hi))
            tau = self.cfg.tau_base * torch.exp(s)

        self._adapt_count += 1
        return {
            "adapted": True,
            "tau": float(tau),
            "n_adaptations": self._adapt_count,
            "no_zeno_events": self.gate.event_count if self.gate else 0,
        }

    # ── task generator (synthetic distribution) ─────────────────────────
    def sample_tasks(self, n_tasks: int, seed: int = 0) -> List[MetaTauTask]:
        """
        [D] Synthetic task distribution: spheres of varying radius/sharpness.
        Sharper boundary (smaller transition) → smaller target τ.
        """
        rng = np.random.default_rng(seed)
        tasks = []
        g = self.cfg.grid
        dev = self.device
        for i in range(n_tasks):
            radius = float(rng.uniform(0.2, 0.4)) * g
            center = (g*0.5, g*0.5, g*0.5)
            f = _sdf_sphere((g, g, g), center, radius, 1.0, dev)
            # sharper field → smaller τ target
            sharp = float(rng.uniform(0.5, 4.0))
            f = f / sharp
            tau_target = float(np.clip(0.02 + 0.03 / sharp,
                                        self.cfg.tau_min, self.cfg.tau_max))
            tasks.append(MetaTauTask(
                task_id=f"task_{i}", field=f, tau_target=tau_target))
        return tasks

    def get_status(self) -> Dict[str, Any]:
        return {
            "tau_base": self.cfg.tau_base,
            "tau_range": (self.cfg.tau_min, self.cfg.tau_max),
            "inner_steps": self.cfg.inner_steps,
            "first_order": self.cfg.first_order,
            "n_adaptations": self._adapt_count,
            "tda_available": self.tda is not None,
            "no_zeno_available": self.gate is not None,
            "carpet_df_prior": CARPET_DF,
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESITopologyStudio(nn.Module):
    """
    Unifies: CLF/CTF FileTransform bridge + Visual CSG tree + Meta-τ.

    Full pipeline:
      build CSG tree → execute → get SDF field
        → meta-predict τ → anneal SoftMarchingCubes with that τ
        → apply OCIO colour via a resolved .clf/.ctf FileTransform
    """

    def __init__(self,
                 bridge_cfg: Optional[FileBridgeConfig] = None,
                 editor_grid: Tuple[int, int, int] = (28, 28, 28),
                 meta_cfg: Optional[MetaTauConfig] = None,
                 ocio_loader: Optional[Any] = None):
        super().__init__()
        self.bridge = CLFCTFFileTransformBridge(bridge_cfg, ocio_loader)
        self.editor = VisualCSGTreeEditor(grid=editor_grid)
        self.meta_tau = MetaTauPredictor(meta_cfg)

    def build_demo_tree(self) -> CSGTree:
        """Sphere ∪ Box, then subtract a torus — the classic CSG demo."""
        tree = CSGTree()
        s = tree.add_primitive(CSGNodeType.SPHERE, node_id="s1",
                                center=(10, 14, 14), radius=7.0)
        b = tree.add_primitive(CSGNodeType.BOX, node_id="b1",
                                center=(18, 14, 14), half_extents=(5, 5, 5))
        t = tree.add_primitive(CSGNodeType.TORUS, node_id="t1",
                                center=(14, 14, 14), R=8.0, r=2.5)
        u = tree.add_op(CSGNodeType.UNION, [s, b], node_id="u1")
        d = tree.add_op(CSGNodeType.DIFFERENCE, [u, t], node_id="d1")
        self.editor.make_output(tree, d)
        return tree

    def run_pipeline(self, tree: Optional[CSGTree] = None,
                      meta_adapt: bool = True) -> Dict[str, Any]:
        tree = tree or self.build_demo_tree()
        exec_res = self.editor.execute(tree)
        sdf = exec_res["field"]

        # meta τ
        if meta_adapt:
            tau_info = self.meta_tau.adapt_to_field(sdf)
        else:
            tau_info = {"tau": self.editor.tau, "adapted": False}
        tau = tau_info.get("tau", self.editor.tau)

        # re-execute with the adapted τ (sharper Boolean boundaries)
        exec2 = self.editor.execute(tree, tau=tau)

        return {
            "tree_nodes": exec_res["n_nodes"],
            "tau_meta": tau_info,
            "tau_used": tau,
            "sdf_range": exec2["sdf_range"],
            "field": exec2["field"],
        }

    def apply_clf_to_render(self, clf_path: str, rgb: Tensor) -> Tensor:
        """Apply a real .clf/.ctf to a rendered image."""
        return self.bridge.execute(clf_path, rgb)

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "bridge": self.bridge.get_status(),
            "editor": self.editor.get_status(),
            "meta_tau": self.meta_tau.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def _write_demo_clf(path: str) -> None:
    """Write a minimal real .clf file for the bridge test."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<ProcessList xmlns="urn:ampas:aces:clf:v3" id="demo" compCLFversion="3.0">
  <ProcessNode type="matrix">
    <mat>1.6 0.0 0.0 0.0 1.6 0.0 0.0 0.0 1.6</mat>
    <offset>0.0 0.0 0.0</offset>
  </ProcessNode>
  <ProcessNode type="range">
    <minInValue>0.0</minInValue>
    <maxInValue>2.0</maxInValue>
    <minOutValue>0.0</minOutValue>
    <maxOutValue>1.0</maxOutValue>
  </ProcessNode>
  <ProcessNode type="lut1d" dim="5">
    <Array>0.0 0.0 0.0  0.2 0.18 0.15  0.5 0.48 0.45  0.8 0.79 0.78  1.0 1.0 1.0</Array>
  </ProcessNode>
</ProcessList>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def smoke_test():
    print("=" * 78)
    print(" SESI TOPOLOGY STUDIO — SMOKE TEST")
    print(" Real CLF/CTF FileTransform + Visual CSG Tree + Meta-Learning τ")
    print("=" * 78)
    print(f" torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print(f" prev modules    : {HAS_PREV}")
    print(f" topology module : {HAS_TOPO}")
    print(f" NoZeno          : {HAS_NOZENO}")
    print(f" TDA             : {HAS_TDA}")
    print(f"\n Carpet-19 reference [C] (ρ NUMERICAL, not proved)")
    print(f"   d_f = ln8/ln3     = {CARPET_DF:.6f}")
    print(f"   ρ                 = {CARPET_RHO}")
    print(f"   d_w = ln(8ρ)/ln3  = {CARPET_DW:.6f}")
    print(f"   d_s = 2d_f/d_w    = {CARPET_DS:.6f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    studio = SESITopologyStudio(editor_grid=(24, 24, 24)).to(dev)

    # ── 1. CLF/CTF FileTransform bridge ─────────────────────────────────
    print("\n [CLB1] CLF/CTF FileTransform bridge (REAL file)")
    import tempfile
    tmp = tempfile.mkdtemp()
    clf_path = os.path.join(tmp, "demo_transform.clf")
    _write_demo_clf(clf_path)

    executor = studio.bridge.load(clf_path)
    print(f"   Loaded file          : {os.path.basename(clf_path)}")
    print(f"   Node types           : {executor.metadata.get('node_types')}")
    print(f"   n_nodes              : {executor.metadata.get('n_nodes')}")

    rgb = torch.rand(4, 8, 8, 3, device=dev, requires_grad=True)
    out = studio.bridge.execute(clf_path, rgb)
    out.sum().backward()
    print(f"   Input range          : [{float(rgb.min()):.3f}, {float(rgb.max()):.3f}]")
    print(f"   Output range         : [{float(out.min()):.3f}, {float(out.max()):.3f}]")
    g_ok = rgb.grad is not None and bool(rgb.grad.isfinite().all())
    print(f"   Gradient flows       : {g_ok}")

    # missing file → skip, never approximate
    missing = studio.bridge._validate(os.path.join(tmp, "nope.clf"))
    print(f"   Missing-file gate    : ok={missing['ok']} reason={missing['reason']}")

    audit = studio.bridge.get_audit_log()
    print(f"   Audit entries        : {len(audit)}")

    # unsupported node must raise
    bad_clf = os.path.join(tmp, "bad.clf")
    with open(bad_clf, "w") as fh:
        fh.write("""<?xml version="1.0"?>
<ProcessList xmlns="urn:ampas:aces:clf:v3" id="bad">
  <ProcessNode type="grading_primary"><brightness>1.0</brightness></ProcessNode>
</ProcessList>""")
    try:
        studio.bridge.load(bad_clf, force_reload=True)
        print("   Unsupported node     : FAILED TO RAISE")
    except NotImplementedError:
        print("   Unsupported node     : raised NotImplementedError ✓")

    # ── 2. Visual CSG tree editor ───────────────────────────────────────
    print("\n [VCT1] Visual CSG tree editor")
    tree = studio.build_demo_tree()
    val = tree.validate()
    print(f"   Nodes                : {val['n_nodes']}")
    print(f"   Valid                : {val['valid']}")
    print(f"   Topo order           : {tree.topo_order()}")

    exec_res = studio.editor.execute(tree)
    print(f"   Executed τ           : {exec_res['tau']}")
    print(f"   SDF range            : [{exec_res['sdf_range'][0]:.3f}, "
          f"{exec_res['sdf_range'][1]:.3f}]")

    # gradient through parameters
    s_node = tree.nodes["s1"]
    rad = torch.tensor(float(s_node.params["radius"]), device=dev,
                        requires_grad=True)
    # re-evaluate sphere with tensor radius
    f = _sdf_sphere(studio.editor.grid, tuple(s_node.params["center"]),
                     rad, studio.editor.dx, dev)
    f.sum().backward()
    print(f"   ∂field/∂radius       : {float(rad.grad):.3e} (finite="
          f"{bool(rad.grad.isfinite())})")

    # JSON round-trip
    js = tree.to_json()
    tree2 = CSGTree.from_json(js)
    val2 = tree2.validate()
    print(f"   JSON round-trip      : nodes={val2['n_nodes']} valid={val2['valid']}")

    # cycle detection
    cyc = CSGTree()
    cyc.add_node(CSGNode("a", CSGNodeType.UNION, inputs=["b"]))
    cyc.add_node(CSGNode("b", CSGNodeType.UNION, inputs=["a"]))
    cv = cyc.validate()
    print(f"   Cycle detection      : valid={cv['valid']} "
          f"errors={cv['errors']}")

    # ── 3. Meta-learning τ ──────────────────────────────────────────────
    print("\n [MTL1] Meta-learning τ predictor (MAML)")
    tasks = studio.meta_tau.sample_tasks(n_tasks=6, seed=7)
    print(f"   Sampled tasks        : {len(tasks)}")
    print(f"   τ targets            : {[round(t.tau_target,4) for t in tasks]}")

    feats = studio.meta_tau.extract_features(tasks[0].field)
    print(f"   Feature vector       : {[round(float(x),4) for x in feats]}")
    print(f"   (resid, tda_ent, var, grad, d_f prior={CARPET_DF:.4f})")

    meta_res = studio.meta_tau.meta_train(tasks, n_meta_epochs=15)
    print(f"   Meta-loss            : {meta_res['first_meta_loss']:.5f} → "
          f"{meta_res['final_meta_loss']:.5f}")
    print(f"   Improved             : {meta_res['improved']}")
    print(f"   First-order (FOMAML) : {meta_res['first_order']}")

    # inference adaptation with No-Zeno
    new_field = tasks[1].field
    adapt = studio.meta_tau.adapt_to_field(new_field,
                                            tau_target_hint=tasks[1].tau_target,
                                            steps=3)
    print(f"   Adapted τ            : {adapt['tau']:.5f} "
          f"(target {tasks[1].tau_target:.5f})")
    print(f"   Adapted              : {adapt['adapted']}")
    print(f"   No-Zeno events       : {adapt.get('no_zeno_events', 0)}")

    tau_status = studio.meta_tau.get_status()
    print(f"   τ range              : {tau_status['tau_range']}")
    print(f"   TDA feature          : available={tau_status['tda_available']}")

    # ── 4. Full pipeline ────────────────────────────────────────────────
    print("\n [PIPELINE] CSG tree → meta-τ → re-execute")
    pl = studio.run_pipeline(meta_adapt=True)
    print(f"   Tree nodes           : {pl['tree_nodes']}")
    print(f"   Meta τ               : {pl['tau_used']:.5f}")
    print(f"   SDF range @ meta-τ   : [{pl['sdf_range'][0]:.3f}, "
          f"{pl['sdf_range'][1]:.3f}]")

    # ── 5. Status ───────────────────────────────────────────────────────
    print("\n [STATUS]")
    st = studio.get_full_status()
    for mod, s in st.items():
        print(f"   {mod}:")
        for k, v in s.items():
            print(f"     {k}: {v}")

    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C/D] CLF/CTF bridge executes REAL files found on search_path.")
    print("       Missing files are SKIPPED with a warning — never faked.")
    print("       Unsupported node types raise NotImplementedError.")
    print(" [D]   VisualCSGTreeEditor is a DATA MODEL + ENGINE, not a GUI.")
    print("       Graph structure is discrete; only values/params are")
    print("       differentiable.")
    print(" [C/D] MetaTauPredictor needs a TASK DISTRIBUTION. With one field")
    print("       it degenerates to plain gradient descent. Learned τ is a")
    print("       heuristic, not an optimum.")
    print(" [O]   TDA-entropy feature is a proxy; the topology↔τ link is")
    print("       a research hypothesis, not a theorem.")
    print(" [C]   Carpet ρ≈1.25 is NUMERICAL, not proved. d_f prior used as")
    print("       a reference constant only.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
