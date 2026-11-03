"""
=============================================================================
SESI TOPOLOGY STUDIO COLLAB
WebSocket Real-Time Editing + PyVista/Three.js 3D + Non-PCF τ Transplant
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
CARPET-19 FRACTAL STATUS TABLE (the single most important contract here)
=============================================================================
Fractal              p.c.f.?  Theorem 10 status        Hino Γ⊥µ        Label
-------------------- -------  -----------------------  --------------- -----
sierpinski_carpet    NO       conditional (I1)-(I4)    PROVED [C]      [C] anchor
                              supported by KZ/BBKT/    Hino 2005
                              Hino (Remark 6)          Thm 5.3
sierpinski_gasket    YES      KZ applies to p.c.f.     NOT covered [C] [C] p.c.f.
                              [C]                      needs separate
                                                      reference
vicsek               NO       (I1)-(I4) UNPROVED       UNKNOWN         [O] transplant
cantor_dust          NO       (I1)-(I4) UNPROVED       UNKNOWN         [O] transplant
                              + totally disconnected   (d_w undefined)

RULES ENFORCED BY CODE:
  • closure_transferred is HARD-CODED False for every non-anchor set.
  • Hino singularity is asserted ONLY where cited (standard carpet).
  • ρ_eff for non-anchor sets is an EXPONENT-MATCHED TRANSPLANT, never a
    proved resistance limit. ρ≈1.25 for the carpet is NUMERICAL, not proved.
  • τ predicted for an [O] set carries a warning that Theorem 10 does not
    apply and the result is heuristic.

=============================================================================
MODULES
=============================================================================
[WSC1] WebSocketCollabBackend
      • LamportClock [C] causal ordering
      • CollabMessageRouter: rooms, presence, op log, LWW conflict resolution
      • WebSocketCollabServer: async wrapper (requires `websockets`)
      • No-Zeno bounded per-client op rate
      • PNDC Axiom 3: op audit log, client display names are opaque ids

[VPV1] PyVistaThreeJSViewer
      • to_pyvista(): numpy mesh -> pyvista.PolyData (optional dep)
      • render_offscreen(): server-side screenshot (needs GPU/OSMesa)
      • to_threejs_dict() / to_threejs_html(): browser preview via CDN three.js
      • slice_matplotlib(): 2D slice fallback when pyvista absent
      • HONEST: rendering is display-only, NOT differentiable

[NPT1] NonPCFTauTransplantLearner
      • FractalRegistry: carpet / gasket / vicsek / cantor with exact masks
      • measure_exponents(): box-counting d_f + random-walk d_w
      • transplant identity: ρ_eff = base^{d_w}/n_kept  [O, NOT a theorem]
      • β = log ρ_eff / log base ;  d_s = 2 d_f / d_w ;  α_Caputo = 2/d_w
      • τ prediction MLP conditioned on (d_f, d_w, ρ_eff, β, d_s, |∇u|)
      • meta_fit() across fractal tasks
      • get_transplant_table(): the status table above, emitted with labels

=============================================================================
HONEST SCOPE
=============================================================================
• WebSocket backend is a PROTOCOL ROUTER + optional async server. Real
  deployment needs TLS, auth, and a production ASGI server.
• Conflict resolution is LAST-WRITER-WINS, not a full CRDT. Concurrent
  structural edits can still require manual merge. The op log is the
  source of truth; clients replay it.
• three.js HTML uses a CDN; offline use requires bundling three.js.
• PyVista offscreen rendering requires a working OpenGL/OSMesa context.
  Without it, the viewer returns None and the matplotlib slice is used.
• Non-PCF τ values are HEURISTICS on an unproved foundation. They must be
  validated against an analytic reference before any production use
  (Carpet-19 Theorem 3(b) discipline).
• All thresholds are FREE DESIGN PARAMETERS.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.topo_collab")

# ── Optional dependencies ───────────────────────────────────────────────
try:
    import websockets
    HAS_WEBSOCKETS = True
except Exception:
    HAS_WEBSOCKETS = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ── Ecosystem imports (graceful) ────────────────────────────────────────
try:
    from sesi_topology_studio_gui import (
        CarpetMeasureTauLearner, CarpetTauConfig,
        CARPET_DF, CARPET_RHO, CARPET_DW, CARPET_DS,
        _softplus_floor, _sdf_sphere_field,
    )
    HAS_GUI = True
except Exception:
    HAS_GUI = False
    CARPET_DF = math.log(8.0) / math.log(3.0)
    CARPET_RHO = 1.25
    CARPET_DW = math.log(8.0 * CARPET_RHO) / math.log(3.0)
    CARPET_DS = 2.0 * CARPET_DF / CARPET_DW
    CARPET_BETA = math.log(CARPET_RHO) / math.log(3.0)

    def _softplus_floor(x, floor, beta=None, sharpness=50.0):
        if beta is None:
            beta = sharpness / max(abs(floor), 1e-12)
        return floor + F.softplus(x - floor, beta=beta)

    def _sdf_sphere_field(shape, center, radius, dx, device):
        X, Y, Z = shape
        xs = torch.arange(X, device=device, dtype=torch.float32) * dx
        ys = torch.arange(Y, device=device, dtype=torch.float32) * dx
        zs = torch.arange(Z, device=device, dtype=torch.float32) * dx
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        cx, cy, cz = center
        return torch.sqrt((gx-cx)**2 + (gy-cy)**2 + (gz-cz)**2 + 1e-12) - radius

try:
    from sesi_differentiable_topology import MarchingCubesExtractor
    HAS_MC = True
except Exception:
    HAS_MC = False

try:
    from sesi_unified_amr2 import NoZenoGate
    HAS_NOZENO = True
except Exception:
    HAS_NOZENO = False

CARPET_BETA = math.log(CARPET_RHO) / math.log(3.0)

__all__ = [
    "CollabConfig", "CollabOp", "LamportClock", "CollabMessageRouter",
    "WebSocketCollabServer",
    "ViewerConfig", "PyVistaThreeJSViewer",
    "FractalSpec", "FractalRegistry", "TransplantDiagnostics",
    "NonPCFTauTransplantLearner",
    "SESICollabViewerSuite",
]


# =============================================================================
# SECTION 1 — [WSC1] WEBSOCKET COLLABORATIVE EDITING BACKEND
# =============================================================================

@dataclass
class CollabConfig:
    """Configuration for real-time collaboration."""
    room_capacity: int = 16
    max_ops_per_second_per_client: int = 20
    use_no_zeno: bool = True
    theta_hi: float = 0.2
    theta_lo: float = 0.05
    max_op_log: int = 10000


class LamportClock:
    """[C] Lamport 1978 logical clock for causal ordering."""
    def __init__(self):
        self.t = 0
    def tick(self, received: int = 0) -> int:
        self.t = max(self.t, received) + 1
        return self.t


@dataclass
class CollabOp:
    """A single collaborative edit operation on the CSG tree."""
    op_id: str
    client_id: str
    op_type: str            # add_node | update_params | delete_node | set_tau | set_output
    node_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    lamport: int = 0
    wall_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"op_id": self.op_id, "client_id": self.client_id,
                "op_type": self.op_type, "node_id": self.node_id,
                "payload": self.payload, "lamport": self.lamport,
                "wall_time": self.wall_time}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CollabOp":
        return cls(op_id=d.get("op_id", f"op_{uuid.uuid4().hex[:8]}"),
                   client_id=d["client_id"], op_type=d["op_type"],
                   node_id=d.get("node_id", ""),
                   payload=d.get("payload", {}),
                   lamport=d.get("lamport", 0),
                   wall_time=d.get("wall_time", time.time()))


@dataclass
class ClientPresence:
    client_id: str
    display_name: str
    joined_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    lamport: int = 0


@dataclass
class RoomState:
    room_id: str
    clients: Dict[str, ClientPresence] = field(default_factory=dict)
    op_log: List[CollabOp] = field(default_factory=list)
    clock: LamportClock = field(default_factory=LamportClock)
    tree_json: Optional[str] = None


class CollabMessageRouter:
    """
    [D] Pure-logic collaborative editing router (no network I/O).

    Guarantees:
      • Total order of operations via Lamport timestamps (ties broken by
        client_id for determinism).
      • Last-writer-wins on the same (node_id, field) — NOT a full CRDT.
      • No-Zeno bound on per-client op rate.
      • Full op audit log (PNDC Axiom 3).

    HONEST: concurrent structural edits (two clients adding Boolean ops
    with overlapping inputs) can still conflict; the router records both
    and flags the conflict for the clients to resolve. Replay of the op
    log is the source of truth.
    """

    def __init__(self, cfg: Optional[CollabConfig] = None):
        self.cfg = cfg or CollabConfig()
        self.rooms: Dict[str, RoomState] = {}
        self._client_ts: Dict[str, deque] = defaultdict(deque)
        self.gate = (NoZenoGate(self.cfg.theta_hi, self.cfg.theta_lo)
                     if (self.cfg.use_no_zeno and HAS_NOZENO) else None)

    # ── rooms & presence ────────────────────────────────────────────────
    def create_room(self, room_id: Optional[str] = None) -> RoomState:
        rid = room_id or f"room_{uuid.uuid4().hex[:8]}"
        room = RoomState(room_id=rid)
        self.rooms[rid] = room
        logger.info("[WSC] created room %s", rid)
        return room

    def join_room(self, room_id: str, client_id: str,
                  display_name: str = "") -> bool:
        room = self.rooms.get(room_id)
        if room is None or len(room.clients) >= self.cfg.room_capacity:
            return False
        room.clients[client_id] = ClientPresence(
            client_id=client_id, display_name=display_name or client_id)
        logger.info("[WSC] %s joined %s", client_id, room_id)
        return True

    def leave_room(self, room_id: str, client_id: str) -> None:
        room = self.rooms.get(room_id)
        if room and client_id in room.clients:
            del room.clients[client_id]

    # ── rate limiting ───────────────────────────────────────────────────
    def _check_rate(self, client_id: str) -> bool:
        now = time.time()
        dq = self._client_ts[client_id]
        while dq and dq[0] < now - 1.0:
            dq.popleft()
        if len(dq) >= self.cfg.max_ops_per_second_per_client:
            return False
        dq.append(now)
        return True

    # ── operation submission ────────────────────────────────────────────
    def submit_op(self, room_id: str, op: CollabOp) -> Dict[str, Any]:
        """
        Assign a Lamport timestamp, validate, append to the room log.
        Returns a receipt with conflict flags.
        """
        room = self.rooms.get(room_id)
        if room is None:
            return {"accepted": False, "reason": "room_not_found"}
        if op.client_id not in room.clients:
            return {"accepted": False, "reason": "not_a_member"}
        if not self._check_rate(op.client_id):
            return {"accepted": False, "reason": "rate_limited",
                    "hint": "No-Zeno per-client op bound"}

        op.lamport = room.clock.tick(op.lamport)
        op.wall_time = time.time()

        # Conflict detection: same node_id, same field, earlier lamport exists
        conflict_with: List[str] = []
        for prev in room.op_log[-64:]:
            if prev.node_id == op.node_id and prev.op_type == op.op_type \
                    and prev.client_id != op.client_id:
                conflict_with.append(prev.op_id)

        room.op_log.append(op)
        if len(room.op_log) > self.cfg.max_op_log:
            room.op_log = room.op_log[-self.cfg.max_op_log:]

        if self.gate is not None:
            self.gate.update(1.0)

        logger.info("[WSC] op %s (%s) node=%s lamport=%d conflicts=%d",
                    op.op_type, op.client_id, op.node_id, op.lamport,
                    len(conflict_with))

        return {"accepted": True, "lamport": op.lamport,
                "op_id": op.op_id, "conflicts": conflict_with}

    # ── deterministic ordering & replay ─────────────────────────────────
    def ordered_log(self, room_id: str) -> List[CollabOp]:
        room = self.rooms.get(room_id)
        if room is None:
            return []
        return sorted(room.op_log, key=lambda o: (o.lamport, o.client_id))

    def resolve_lww(self, room_id: str) -> Dict[str, CollabOp]:
        """
        Last-writer-wins reduction: for each node_id+op_type keep the op
        with the highest (lamport, client_id). Returns node_id -> winning op.
        """
        winners: Dict[Tuple[str, str], CollabOp] = {}
        for op in self.ordered_log(room_id):
            key = (op.node_id, op.op_type)
            prev = winners.get(key)
            if prev is None or (op.lamport, op.client_id) > \
                    (prev.lamport, prev.client_id):
                winners[key] = op
        return {op.node_id: op for op in winners.values()}

    def apply_ops_to_tree(self, room_id: str, tree: Any) -> Dict[str, Any]:
        """
        Replay the LWW-reduced op log onto a CSGTree (from
        sesi_topology_studio). Returns application stats.
        """
        from sesi_topology_studio import CSGNode, CSGNodeType
        winners = self.resolve_lww(room_id)
        applied, skipped = 0, 0
        for node_id, op in winners.items():
            try:
                if op.op_type == "add_node":
                    ntype = CSGNodeType(op.payload.get("node_type", "sphere"))
                    if node_id not in tree.nodes:
                        tree.add_node(CSGNode(node_id, ntype,
                                              op.payload.get("params", {})))
                    applied += 1
                elif op.op_type == "update_params":
                    if node_id in tree.nodes:
                        tree.nodes[node_id].params.update(op.payload)
                        applied += 1
                elif op.op_type == "set_tau":
                    if node_id in tree.nodes:
                        tree.nodes[node_id].tau = float(op.payload.get("tau", 0.05))
                        applied += 1
                elif op.op_type == "delete_node":
                    if node_id in tree.nodes:
                        del tree.nodes[node_id]
                        applied += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning("[WSC] apply_op failed for %s: %s", node_id, e)
                skipped += 1
        return {"applied": applied, "skipped": skipped,
                "total_ops": len(winners)}

    def get_room_status(self, room_id: str) -> Dict[str, Any]:
        room = self.rooms.get(room_id)
        if room is None:
            return {}
        return {
            "room_id": room_id,
            "n_clients": len(room.clients),
            "clients": list(room.clients.keys()),
            "n_ops": len(room.op_log),
            "lamport": room.clock.t,
            "no_zeno_events": self.gate.event_count if self.gate else 0,
        }


class WebSocketCollabServer:
    """
    [D] Async WebSocket server wrapping CollabMessageRouter.

    Requires the `websockets` package. Message protocol (JSON):
      {"type": "join",  "client_id": ..., "display_name": ...}
      {"type": "op",    "op": {...CollabOp fields...}}
      {"type": "state"}  -> returns full ordered op log
      {"type": "ping"}

    HONEST: no TLS / auth here. Production needs both, plus an ASGI server.
    """

    def __init__(self, router: CollabMessageRouter,
                 room_id: str, host: str = "localhost", port: int = 8765):
        self.router = router
        self.room_id = room_id
        self.host = host
        self.port = port
        self._clients: Dict[str, Any] = {}

    async def handler(self, ws: Any, path: str = "") -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "join":
                cid = msg.get("client_id", f"anon_{uuid.uuid4().hex[:6]}")
                self.router.join_room(self.room_id, cid,
                                      msg.get("display_name", ""))
                self._clients[cid] = ws
                await ws.send(json.dumps({"type": "joined", "client_id": cid}))
            elif mtype == "op":
                op = CollabOp.from_dict(msg.get("op", {}))
                receipt = self.router.submit_op(self.room_id, op)
                # broadcast receipt to all clients (fan-out)
                payload = json.dumps({"type": "op_receipt", **receipt,
                                       "op": op.to_dict()})
                for c in list(self._clients.values()):
                    try:
                        await c.send(payload)
                    except Exception:
                        pass
            elif mtype == "state":
                log = [o.to_dict() for o in self.router.ordered_log(self.room_id)]
                await ws.send(json.dumps({"type": "state", "ops": log}))
            elif mtype == "ping":
                await ws.send(json.dumps({"type": "pong"}))

    async def start(self) -> None:
        """Start serving. Blocks until cancelled."""
        if not HAS_WEBSOCKETS:
            raise RuntimeError("`websockets` package not installed")
        self.router.create_room(self.room_id)
        logger.info("[WSC] serving on ws://%s:%d room=%s",
                    self.host, self.port, self.room_id)
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

    def get_status(self) -> Dict[str, Any]:
        return {"websockets_available": HAS_WEBSOCKETS,
                "room_id": self.room_id,
                "host": self.host, "port": self.port,
                "connected_clients": len(self._clients)}


# =============================================================================
# SECTION 2 — [VPV1] PYVISTA / THREE.JS 3D VIEWER
# =============================================================================

@dataclass
class ViewerConfig:
    background: str = "#101018"
    mesh_color: str = "#4ecdc4"
    opacity: float = 1.0
    window_size: Tuple[int, int] = (720, 560)
    three_cdn: str = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"


class PyVistaThreeJSViewer:
    """
    [D] 3D preview: PyVista for server-side, three.js JSON/HTML for browser.

    DIFFERENTIABILITY: rendering is DISPLAY-ONLY and not differentiable.
    The mesh geometry passed in may carry gradients (from marching cubes),
    but the act of drawing detaches everything.

    HONEST:
      • pyvista is optional. Without it, render_offscreen returns None.
      • Offscreen rendering needs a working OpenGL / OSMesa context.
      • three.js is loaded from a CDN; bundle it for offline use.
    """

    def __init__(self, cfg: Optional[ViewerConfig] = None):
        self.cfg = cfg or ViewerConfig()

    @staticmethod
    def _hex_int(s: str) -> int:
        return int(s.lstrip("#"), 16)

    # ── pyvista ─────────────────────────────────────────────────────────
    def to_pyvista(self, vertices: np.ndarray,
                   faces: np.ndarray) -> Optional[Any]:
        if not HAS_PYVISTA:
            return None
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[0] == 0:
            return None
        cells = np.column_stack([
            np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
        mesh = pv.PolyData(vertices, cells)
        return mesh

    def render_offscreen(self, vertices: np.ndarray, faces: np.ndarray
                          ) -> Optional[np.ndarray]:
        """Return an RGBA screenshot array, or None if pyvista is unavailable."""
        mesh = self.to_pyvista(vertices, faces)
        if mesh is None:
            return None
        try:
            plotter = pv.Plotter(off_screen=True,
                                  window_size=self.cfg.window_size)
            plotter.add_mesh(mesh, color=self.cfg.mesh_color,
                             opacity=self.cfg.opacity)
            plotter.background_color = self.cfg.background
            plotter.show()
            return plotter.screenshot
        except Exception as e:
            logger.warning("[VPV] offscreen render failed: %s", e)
            return None

    # ── three.js ────────────────────────────────────────────────────────
    def to_threejs_dict(self, vertices: np.ndarray,
                        faces: np.ndarray) -> Dict[str, Any]:
        return {
            "metadata": {"formatVersion": 3, "generator": "SESI VPV",
                          "n_vertices": int(len(vertices)),
                          "n_faces": int(len(faces))},
            "data": {
                "vertices": np.asarray(vertices, dtype=np.float32).reshape(-1).tolist(),
                "indices": np.asarray(faces, dtype=np.int64).reshape(-1).tolist(),
            },
        }

    def to_threejs_html(self, vertices: np.ndarray, faces: np.ndarray,
                        title: str = "SESI Mesh") -> str:
        d = self.to_threejs_dict(vertices, faces)
        verts_json = json.dumps(d["data"]["vertices"])
        idx_json = json.dumps(d["data"]["indices"])
        color_hex = hex(self._hex_int(self.cfg.mesh_color))
        bg_hex = "#" + format(self._hex_int(self.cfg.background), "06x")
        opacity = float(self.cfg.opacity)
        return (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n"
            f"<title>{title}</title>\n"
            "<style>body{margin:0;overflow:hidden;background:" + bg_hex +
            ";}#info{position:absolute;top:8px;left:8px;color:#eee;"
            "font-family:monospace;font-size:12px;}</style>\n"
            f"<script src='{self.cfg.three_cdn}'></script>\n"
            "</head><body><div id='info'>" + title +
            f" | V={d['metadata']['n_vertices']} F={d['metadata']['n_faces']}"
            "</div><script>\n"
            "var VERTS=" + verts_json + ";\n"
            "var FACES=" + idx_json + ";\n"
            "var scene=new THREE.Scene();"
            "scene.background=new THREE.Color('" + bg_hex + "');\n"
            "var camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,2000);"
            "camera.position.set(2.2,1.8,2.2);camera.lookAt(0,0,0);\n"
            "var renderer=new THREE.WebGLRenderer({antialias:true});"
            "renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);"
            "document.body.appendChild(renderer.domElement);\n"
            "var geo=new THREE.BufferGeometry();"
            "geo.setAttribute('position',new THREE.Float32BufferAttribute(VERTS,3));"
            "if(FACES.length){geo.setIndex(FACES);}geo.computeVertexNormals();\n"
            "var mat=new THREE.MeshStandardMaterial({color:" + color_hex +
            ",metalness:0.15,roughness:0.65,side:THREE.DoubleSide,"
            f"transparent:{str(opacity < 1.0).lower()},opacity:{opacity}}});\n"
            "var mesh=new THREE.Mesh(geo,mat);scene.add(mesh);\n"
            "var key=new THREE.DirectionalLight(0xffffff,1.0);key.position.set(5,10,7);scene.add(key);"
            "scene.add(new THREE.AmbientLight(0xffffff,0.45));\n"
            "var grid=new THREE.GridHelper(4,8);grid.position.y=-1;scene.add(grid);\n"
            "function animate(){requestAnimationFrame(animate);"
            "mesh.rotation.y+=0.008;mesh.rotation.x+=0.003;"
            "renderer.render(scene,camera);}\n"
            "window.addEventListener('resize',function(){camera.aspect=innerWidth/innerHeight;"
            "camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});\n"
            "animate();\n</script></body></html>"
        )

    # ── matplotlib fallback ─────────────────────────────────────────────
    def slice_matplotlib(self, vol: np.ndarray, axis: int = 2,
                         index: Optional[int] = None) -> Optional[np.ndarray]:
        """2D slice preview when pyvista is unavailable. Display-only."""
        if not HAS_MPL:
            return None
        index = index if index is not None else vol.shape[axis] // 2
        sl = [slice(None)] * 3
        sl[axis] = index
        plane = vol[tuple(sl)]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(plane.T, origin="lower", cmap="RdBu_r")
        ax.set_title(f"slice {axis}={index}")
        fig.colorbar(im, ax=ax)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return img

    def get_status(self) -> Dict[str, Any]:
        return {"pyvista": HAS_PYVISTA, "matplotlib": HAS_MPL,
                "websockets": HAS_WEBSOCKETS,
                "marching_cubes": HAS_MC}


# =============================================================================
# SECTION 3 — [NPT1] NON-PCF τ TRANSPLANT LEARNER
# =============================================================================

@dataclass
class FractalSpec:
    """A fractal with its epistemic status under Carpet-19."""
    name: str
    base: int
    keep: List[List[int]]          # 0/1 keep pattern, base x base
    is_pcf: bool
    is_anchor: bool                # True ONLY for the standard carpet here
    theoretical_df: float
    theorem10_status: str
    hino_status: str
    transplant_label: str          # "[C] anchor" | "[C] p.c.f." | "[O] transplant"

    @property
    def n_kept(self) -> int:
        return int(sum(sum(row) for row in self.keep))


class FractalRegistry:
    """
    Exact recursive fractal masks + their Carpet-19 epistemic status.

    The status strings encode the paper's discipline:
      • Theorem 10 is CONDITIONAL on (I1)-(I4).
      • For the standard carpet, (I1)-(I4) are supported by KZ/BBKT/Hino.
      • For general non-p.c.f. fractals, (I1)-(I4) ARE the unproved content.
      • Hino's Γ⊥µ is a CITED theorem for carpets d>=2, not a general fact.
    """

    _SPECS: Dict[str, FractalSpec] = {}

    @classmethod
    def _init_specs(cls) -> None:
        if cls._SPECS:
            return
        cls._SPECS = {
            "sierpinski_carpet": FractalSpec(
                name="sierpinski_carpet", base=3,
                keep=[[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                is_pcf=False, is_anchor=True,
                theoretical_df=CARPET_DF,
                theorem10_status=(
                    "[C] conditional (I1)-(I4); for the standard carpet these "
                    "follow from KZ Thms 5.4/7.2/7.16/7.19 + BBKT Thm 1.2 + "
                    "Hino Thm 5.3 (Remark 6)"),
                hino_status="[C] PROVED: Γ(f,·)⊥µ for every f∈F (Hino 2005 Thm 5.3)",
                transplant_label="[C] anchor"),
            "sierpinski_gasket": FractalSpec(
                name="sierpinski_gasket", base=2,
                keep=[[1, 1], [1, 0]],
                is_pcf=True, is_anchor=False,
                theoretical_df=math.log(3) / math.log(2),
                theorem10_status="[C] p.c.f.: KZ theory applies directly",
                hino_status=(
                    "[C] NOT covered by Carpet-19 Hino result; Γ status here "
                    "requires a separate reference"),
                transplant_label="[C] p.c.f."),
            "vicsek": FractalSpec(
                name="vicsek", base=3,
                keep=[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                is_pcf=False, is_anchor=False,
                theoretical_df=math.log(5) / math.log(3),
                theorem10_status=(
                    "[O] (I1)-(I4) UNPROVED for this set; Theorem 10 does "
                    "NOT apply"),
                hino_status="[O] UNKNOWN; MUST NOT assume Γ⊥µ",
                transplant_label="[O] transplant"),
            "cantor_dust": FractalSpec(
                name="cantor_dust", base=3,
                keep=[[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                is_pcf=False, is_anchor=False,
                theoretical_df=math.log(4) / math.log(3),
                theorem10_status=(
                    "[O] (I1)-(I4) UNPROVED; set is totally disconnected "
                    "(d_w undefined for a connected walk)"),
                hino_status="[O] UNKNOWN; MUST NOT assume Γ⊥µ",
                transplant_label="[O] transplant"),
        }

    @classmethod
    def specs(cls) -> Dict[str, FractalSpec]:
        cls._init_specs()
        return cls._SPECS

    @classmethod
    def mask(cls, name: str, level: int) -> np.ndarray:
        """Exact recursive occupancy mask."""
        cls._init_specs()
        spec = cls._SPECS[name]
        pat = np.array(spec.keep, dtype=bool)
        m = np.ones((1, 1), dtype=bool)
        for _ in range(level):
            m = np.kron(m, pat).astype(bool)
        return m

    @classmethod
    def random_walk_dw(cls, mask: np.ndarray, n_steps: int = 200,
                        n_walks: int = 48, seed: int = 0
                        ) -> Tuple[float, float]:
        """
        [D] Estimate walk dimension d_w from mean-square displacement.
        <r^2(t)> ~ t^{2/d_w}  =>  d_w = 2 / slope(log MSD vs log t).
        Returns (d_w, r2). d_w = nan if the set is too disconnected.
        """
        occ = np.argwhere(mask)
        if len(occ) < 2:
            return float("nan"), 0.0
        occ_set = set(map(tuple, occ.tolist()))
        nd = mask.ndim
        rng = np.random.default_rng(seed)
        msd = np.zeros(n_steps + 1)
        counts = np.zeros(n_steps + 1)
        moved_walks = 0
        for _ in range(n_walks):
            pos = occ[rng.integers(len(occ))].astype(float).copy()
            origin = pos.copy()
            counts[0] += 1
            for t in range(1, n_steps + 1):
                axis = rng.integers(nd)
                step = rng.choice([-1, 1])
                cand = pos.copy()
                cand[axis] += step
                if tuple(cand.astype(int)) in occ_set:
                    pos = cand
                msd[t] += float(np.sum((pos - origin) ** 2))
                counts[t] += 1
            if np.sum((pos - origin) ** 2) > 0:
                moved_walks += 1
        if moved_walks < max(2, n_walks // 8):
            return float("nan"), 0.0  # effectively disconnected
        msd_avg = msd / np.maximum(counts, 1)
        lo, hi = 4, int(n_steps * 0.6)
        t = np.arange(lo, hi + 1)
        ok = msd_avg[t] > 0
        if ok.sum() < 5:
            return float("nan"), 0.0
        x = np.log(t[ok])
        y = np.log(msd_avg[t][ok])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        dw = 2.0 / max(slope, 1e-9)
        return float(dw), 1.0 - ss_res / ss_tot

    @classmethod
    def measure_df(cls, name: str, max_level: int = 4) -> Tuple[float, float]:
        """Box-counting d_f across levels; returns (d_f, r2)."""
        cls._init_specs()
        spec = cls._SPECS[name]
        xs, ys = [], []
        for l in range(1, max_level + 1):
            m = cls.mask(name, l)
            n_occ = int(m.sum())
            if n_occ <= 0:
                continue
            xs.append(l * math.log(spec.base))          # log(1/eps)
            ys.append(math.log(n_occ))                   # log N(eps)
        if len(xs) < 2:
            return spec.theoretical_df, 0.0
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * np.array(xs) + intercept
        ss_res = float(np.sum((np.array(ys) - pred) ** 2))
        ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2)) + 1e-12
        return float(slope), 1.0 - ss_res / ss_tot


@dataclass
class TransplantDiagnostics:
    """Full epistemic + numeric report for one fractal's τ transplant."""
    name: str
    transplant_label: str
    is_pcf: bool
    is_anchor: bool
    d_f_theoretical: float
    d_f_measured: float
    d_f_r2: float
    d_w_measured: float
    d_w_r2: float
    rho_eff: Optional[float]
    rho_status: str
    beta: Optional[float]
    d_s: Optional[float]
    alpha_caputo: Optional[float]
    tau_predicted: Optional[float]
    closure_transferred: bool
    theorem10_status: str
    hino_status: str
    caveats: List[str]

    def to_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.transplant_label,
            "pcf": self.is_pcf,
            "d_f": round(self.d_f_measured, 4),
            "d_w": round(self.d_w_measured, 4) if math.isfinite(self.d_w_measured) else None,
            "rho_eff": round(self.rho_eff, 4) if self.rho_eff else None,
            "beta": round(self.beta, 4) if self.beta else None,
            "d_s": round(self.d_s, 4) if self.d_s else None,
            "tau": round(self.tau_predicted, 5) if self.tau_predicted else None,
            "closure_transferred": self.closure_transferred,
            "theorem10": self.theorem10_status.split(";")[0],
            "hino": self.hino_status.split(";")[0],
        }


class NonPCFTauTransplantLearner(nn.Module):
    """
    [D/O] Meta-learner for τ on non-PCF fractals, with strict Carpet-19 labels.

    WHAT IT DOES:
      1. Builds the exact recursive mask for each registered fractal.
      2. Measures d_f (box-counting) and d_w (random-walk MSD).
      3. Forms the TRANSPLANT identities (NOT theorems):
             rho_eff = base^{d_w} / n_kept
             beta    = log(rho_eff) / log(base)
             d_s     = 2 d_f / d_w
             alpha   = 2 / d_w           (Caputo order)
      4. Predicts τ from an MLP on these features + field gradient magnitude.

    WHAT IT REFUSES:
      • closure_transferred is hard-coded False for every non-anchor set.
      • Hino singularity is reported only for the standard carpet.
      • For disconnected sets (Cantor dust) d_w is nan and τ falls back to
        tau_base with an explicit warning.

    HONEST:
      • τ targets are heuristic; without an analytic reference they are
        NOT validated. Carpet-19 Theorem 3(b) discipline: validate against
        an analytic solution before production use.
      • This is a transplant of the carpet's *form* of the exponents, not
        a proof that those exponents govern the target set.
    """

    def __init__(self, n_features: int = 6, n_hidden: int = 32,
                 tau_base: float = 0.05, tau_min: float = 0.005,
                 tau_max: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.tau_base = tau_base
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, 1),
        ).to(self.device)
        self.register_buffer(
            "log_lo", torch.tensor(math.log(tau_min / tau_base)))
        self.register_buffer(
            "log_hi", torch.tensor(math.log(tau_max / tau_base)))

    # ── feature extraction ──────────────────────────────────────────────
    def compute_features(self, name: str, level: int = 3,
                          grid: int = 24) -> Dict[str, Any]:
        spec = FractalRegistry.specs()[name]
        df, df_r2 = FractalRegistry.measure_df(name, max_level=4)
        mask = FractalRegistry.mask(name, level)
        dw, dw_r2 = FractalRegistry.random_walk_dw(mask, n_steps=180,
                                                     n_walks=40)
        # build a scalar field supported on the mask (crossing potential)
        m2 = mask.astype(np.float32)
        H, W = m2.shape
        xs = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
        field2d = m2 * xs  # x-coordinate restricted to the fractal
        gy = np.roll(field2d, -1, 1) - field2d
        gx = np.roll(field2d, -1, 0) - field2d
        grad_mag = float(np.sqrt(gx**2 + gy**2).mean())

        rho_eff = beta = d_s = alpha = None
        if math.isfinite(dw) and dw > 0:
            rho_eff = (spec.base ** dw) / spec.n_kept
            beta = math.log(rho_eff) / math.log(spec.base)
            d_s = 2.0 * df / dw
            alpha = 2.0 / dw

        return {"spec": spec, "d_f": df, "d_f_r2": df_r2,
                "d_w": dw, "d_w_r2": dw_r2, "rho_eff": rho_eff,
                "beta": beta, "d_s": d_s, "alpha_caputo": alpha,
                "grad_mag": grad_mag, "mask": mask, "field": field2d}

    def _feature_vector(self, feats: Dict[str, Any]) -> Tensor:
        dw = feats["d_w"] if math.isfinite(feats["d_w"]) else 2.0
        rho = feats["rho_eff"] if feats["rho_eff"] else CARPET_RHO
        beta = feats["beta"] if feats["beta"] else CARPET_BETA
        d_s = feats["d_s"] if feats["d_s"] else CARPET_DS
        return torch.tensor([
            feats["d_f"], dw, rho, beta, d_s, feats["grad_mag"],
        ], dtype=torch.float32, device=self.device)

    def predict_tau(self, feats: Dict[str, Any]) -> float:
        if feats["rho_eff"] is None:
            # disconnected / undefined walk dimension: fall back to base τ
            return self.tau_base
        v = self._feature_vector(feats)
        with torch.no_grad():
            log_scale = self.net(v.unsqueeze(0)).squeeze()
            log_scale = torch.clamp(log_scale, self.log_lo, self.log_hi)
            tau = self.tau_base * torch.exp(log_scale)
        return float(tau)

    # ── main entry point ────────────────────────────────────────────────
    def forward(self, name: str, level: int = 3) -> TransplantDiagnostics:
        feats = self.compute_features(name, level)
        spec: FractalSpec = feats["spec"]
        tau = self.predict_tau(feats)

        caveats: List[str] = []
        if spec.is_anchor:
            caveats.append("ρ≈1.25 is NUMERICAL, not proved (Carpet-19 §9)")
            caveats.append("Theorem 10 conditional on (I1)-(I4), supported "
                           "by KZ/BBKT/Hino for the standard carpet")
        elif spec.is_pcf:
            caveats.append("p.c.f. set: KZ applies; Hino carpet singularity "
                           "result does NOT cover this set")
        else:
            caveats.append("[O] (I1)-(I4) UNPROVED — Theorem 10 does NOT apply")
            caveats.append("[O] Hino Γ⊥µ MUST NOT be assumed on this set")
            caveats.append("[O] ρ_eff is an exponent-matched transplant, "
                           "not a resistance limit")
            caveats.append("closure NOT transferred — validate against an "
                           "analytic reference (Thm 3(b) discipline)")
        if feats["rho_eff"] is None:
            caveats.append("d_w undefined (disconnected set); τ fell back "
                           "to tau_base")

        return TransplantDiagnostics(
            name=name,
            transplant_label=spec.transplant_label,
            is_pcf=spec.is_pcf,
            is_anchor=spec.is_anchor,
            d_f_theoretical=spec.theoretical_df,
            d_f_measured=feats["d_f"],
            d_f_r2=feats["d_f_r2"],
            d_w_measured=feats["d_w"],
            d_w_r2=feats["d_w_r2"],
            rho_eff=feats["rho_eff"],
            rho_status=("NUMERICAL [C-num]" if spec.is_anchor
                        else ("[C] p.c.f." if spec.is_pcf
                              else "[O] transplant")),
            beta=feats["beta"],
            d_s=feats["d_s"],
            alpha_caputo=feats["alpha_caputo"],
            tau_predicted=tau,
            closure_transferred=False,      # HARD-CODED: never True here
            theorem10_status=spec.theorem10_status,
            hino_status=spec.hino_status,
            caveats=caveats,
        )

    # ── meta-fit across fractals ────────────────────────────────────────
    def meta_fit(self, names: Sequence[str], n_epochs: int = 30,
                  lr: float = 1e-3) -> Dict[str, Any]:
        """
        Train the τ MLP across fractal tasks. Targets are heuristic τ
        derived from the transplant exponents (sharper fractal -> smaller τ).
        HONEST: targets are not ground truth; this fits internal consistency.
        """
        feats_list = [self.compute_features(n) for n in names]
        X = torch.stack([self._feature_vector(f) for f in feats_list])
        with torch.no_grad():
            sharp = X[:, 5]  # grad magnitude as sharpness proxy
            target = self.tau_base * torch.exp(-0.5 * sharp)
            target = torch.clamp(target, self.tau_min, self.tau_max)
        y = torch.log(target / self.tau_base)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        losses = []
        for _ in range(n_epochs):
            opt.zero_grad()
            pred = self.net(X).squeeze()
            pred = torch.clamp(pred, self.log_lo, self.log_hi)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        return {"n_tasks": len(names), "n_epochs": n_epochs,
                "final_loss": losses[-1], "loss_history": losses}

    def get_transplant_table(self) -> List[Dict[str, Any]]:
        rows = []
        for name in FractalRegistry.specs():
            diag = self.forward(name)
            rows.append(diag.to_row())
        return rows

    def get_status(self) -> Dict[str, Any]:
        return {
            "fractals": list(FractalRegistry.specs().keys()),
            "tau_base": self.tau_base,
            "tau_range": (self.tau_min, self.tau_max),
            "closure_transferred_for_any": False,
            "carpet_rho_status": "NUMERICAL, not proved [C-num]",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESICollabViewerSuite:
    """
    Unifies: WebSocket collaboration + 3D preview + non-PCF τ transplant.
    """

    def __init__(self, collab_cfg: Optional[CollabConfig] = None,
                 viewer_cfg: Optional[ViewerConfig] = None,
                 device: str = "cpu"):
        self.router = CollabMessageRouter(collab_cfg)
        self.viewer = PyVistaThreeJSViewer(viewer_cfg)
        self.transplant = NonPCFTauTransplantLearner(device=device)
        self.mc = MarchingCubesExtractor() if HAS_MC else None

    def make_demo_field(self, grid: int = 20, device: str = "cpu") -> Tensor:
        """A sphere-minus-box field for demo meshing."""
        sph = _sdf_sphere_field((grid, grid, grid),
                                 (grid/2, grid/2, grid/2),
                                 grid * 0.35, 1.0, device)
        bx = _sdf_sphere_field((grid, grid, grid),
                                (grid/2, grid/2, grid/2),
                                grid * 0.15, 1.0, device)
        # simple smooth difference
        tau = 0.1
        ta, tb = sph / tau, -bx / tau
        m = torch.maximum(ta, tb)
        soft_diff = tau * (m + torch.log(torch.exp(ta - m) + torch.exp(tb - m)))
        return soft_diff

    def field_to_mesh(self, field: Tensor) -> Optional[Dict[str, Any]]:
        if self.mc is None:
            return None
        return self.mc.extract(field, dx=1.0)

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "viewer": self.viewer.get_status(),
            "transplant": self.transplant.get_status(),
            "rooms": list(self.router.rooms.keys()),
            "marching_cubes": HAS_MC,
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI TOPOLOGY STUDIO COLLAB — SMOKE TEST")
    print(" WebSocket + PyVista/Three.js + Non-PCF τ Transplant")
    print("=" * 78)
    print(f" torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print(f" websockets      : {HAS_WEBSOCKETS}")
    print(f" pyvista         : {HAS_PYVISTA}")
    print(f" matplotlib      : {HAS_MPL}")
    print(f" marching cubes  : {HAS_MC}")
    print(f" studio gui      : {HAS_GUI}")
    print(f" No-Zeno         : {HAS_NOZENO}")
    print(f"\n Carpet-19 reference (ρ NUMERICAL, not proved):")
    print(f"   d_f = ln8/ln3      = {CARPET_DF:.6f}")
    print(f"   ρ                  = {CARPET_RHO}  [NUMERICAL]")
    print(f"   d_w = ln(8ρ)/ln3   = {CARPET_DW:.6f}")
    print(f"   d_s = 2d_f/d_w     = {CARPET_DS:.6f}")
    print(f"   β = log ρ / log 3  = {CARPET_BETA:.6f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    suite = SESICollabViewerSuite(device=dev)

    # ── 1. WebSocket collaboration router ───────────────────────────────
    print("\n [WSC1] WebSocket collaboration router")
    room = suite.router.create_room("demo_room")
    for cid in ("alice", "bob", "carol"):
        suite.router.join_room(room.room_id, cid)
    print(f"   Room created        : {room.room_id}")
    print(f"   Clients joined      : {len(room.clients)}")

    # Concurrent edits on the same node -> conflict detection
    op1 = CollabOp(op_id="op_a", client_id="alice", op_type="update_params",
                   node_id="s1", payload={"radius": 6.0})
    op2 = CollabOp(op_id="op_b", client_id="bob", op_type="update_params",
                   node_id="s1", payload={"radius": 7.0})
    r1 = suite.router.submit_op(room.room_id, op1)
    r2 = suite.router.submit_op(room.room_id, op2)
    print(f"   Alice op lamport    : {r1['lamport']} accepted={r1['accepted']}")
    print(f"   Bob op lamport      : {r2['lamport']} accepted={r2['accepted']} "
          f"conflicts={len(r2['conflicts'])}")

    # Ordering is deterministic
    ordered = suite.router.ordered_log(room.room_id)
    lww = suite.router.resolve_lww(room.room_id)
    winner = lww.get("s1")
    print(f"   Ordered ops         : {[o.op_id for o in ordered]}")
    print(f"   LWW winner for s1   : {winner.op_id if winner else None} "
          f"(radius={winner.payload.get('radius') if winner else None})")

    status = suite.router.get_room_status(room.room_id)
    print(f"   Room lamport clock  : {status['lamport']}")
    print(f"   No-Zeno events      : {status['no_zeno_events']}")

    ws_server = WebSocketCollabServer(suite.router, room.room_id)
    ws_status = ws_server.get_status()
    print(f"   WS server available : {ws_status['websockets_available']} "
          f"(async wrapper ready={HAS_WEBSOCKETS})")

    # ── 2. 3D preview ───────────────────────────────────────────────────
    print("\n [VPV1] PyVista / three.js 3D viewer")
    field = suite.make_demo_field(grid=18, device=dev)
    mesh = suite.field_to_mesh(field)
    if mesh is not None and mesh.get("n_vertices", 0) > 0:
        verts = mesh["vertices"].detach().cpu().numpy()
        faces = mesh["faces"].detach().cpu().numpy()
        print(f"   Mesh V / F          : {len(verts)} / {len(faces)}")

        three_dict = suite.viewer.to_threejs_dict(verts, faces)
        print(f"   three.js vertices   : {three_dict['metadata']['n_vertices']}")
        print(f"   three.js faces      : {three_dict['metadata']['n_faces']}")

        html = suite.viewer.to_threejs_html(verts, faces, title="SESI demo")
        print(f"   three.js HTML bytes : {len(html)} "
              f"(CDN: {'cdn.jsdelivr' in html})")

        shot = suite.viewer.render_offscreen(verts, faces)
        print(f"   pyvista screenshot  : "
              f"{'available' if shot is not None else 'unavailable (no pyvista/GL)'}")
    else:
        print("   No mesh produced (marching cubes unavailable)")

    slice_img = suite.viewer.slice_matplotlib(
        field.detach().cpu().numpy(), axis=2)
    print(f"   matplotlib slice    : "
          f"{'available' if slice_img is not None else 'unavailable'}")
    print(f"   Viewer status       : {suite.viewer.get_status()}")

    # ── 3. Non-PCF τ transplant table ───────────────────────────────────
    print("\n [NPT1] Non-PCF τ transplant learner")

    # Meta-fit across all registered fractals
    names = list(FractalRegistry.specs().keys())
    fit = suite.transplant.meta_fit(names, n_epochs=15)
    print(f"   Meta-fit tasks      : {fit['n_tasks']} "
          f"final_loss={fit['final_loss']:.5f}")

    table = suite.transplant.get_transplant_table()
    print("\n   FRACTAL τ TRANSPLANT TABLE")
    print("   " + "-" * 100)
    hdr = (f"   {'name':20s} {'label':14s} {'pcf':4s} {'d_f':>7s} "
           f"{'d_w':>7s} {'ρ_eff':>7s} {'β':>6s} {'d_s':>6s} "
           f"{'τ':>8s} {'closure':>8s}")
    print(hdr)
    print("   " + "-" * 100)
    for row in table:
        dw = f"{row['d_w']:.4f}" if row['d_w'] is not None else "  nan "
        rho = f"{row['rho_eff']:.4f}" if row['rho_eff'] is not None else "  nan "
        beta = f"{row['beta']:.3f}" if row['beta'] is not None else " nan "
        d_s = f"{row['d_s']:.3f}" if row['d_s'] is not None else " nan "
        tau = f"{row['tau']:.5f}" if row['tau'] is not None else "  nan  "
        print(f"   {row['name']:20s} {row['label']:14s} "
              f"{str(row['pcf']):4s} {row['d_f']:7.4f} {dw:>7s} "
              f"{rho:>7s} {beta:>6s} {d_s:>6s} {tau:>8s} "
              f"{str(row['closure_transferred']):>8s}")
    print("   " + "-" * 100)

    # Detailed diagnostics for each fractal
    for name in names:
        diag = suite.transplant(name)
        print(f"\n   [{diag.name}]  label={diag.transplant_label}")
        print(f"     d_f theory={diag.d_f_theoretical:.4f} "
              f"measured={diag.d_f_measured:.4f} (R²={diag.d_f_r2:.3f})")
        dw_str = f"{diag.d_w_measured:.4f}" if math.isfinite(diag.d_w_measured) else "nan"
        print(f"     d_w={dw_str}  ρ_eff={diag.rho_eff}  [{diag.rho_status}]")
        print(f"     β={diag.beta}  d_s={diag.d_s}  α_Caputo={diag.alpha_caputo}")
        print(f"     τ={diag.tau_predicted}  closure_transferred={diag.closure_transferred}")
        print(f"     Theorem 10: {diag.theorem10_status}")
        print(f"     Hino      : {diag.hino_status}")
        for c in diag.caveats:
            print(f"     ⚠ {c}")

    # ── 4. Full status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    st = suite.get_full_status()
    for k, v in st.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Theorem 10 is CONDITIONAL on (I1)-(I4). For the standard")
    print("       carpet these follow from KZ Thms 5.4/7.2/7.16/7.19 +")
    print("       BBKT Thm 1.2 + Hino Thm 5.3 (Carpet-19 Remark 6).")
    print(" [C]   Hino 2005 Thm 5.3: Γ(f,·)⊥µ for EVERY f∈F on carpets d≥2.")
    print("       This is a CITED theorem for carpets, NOT a general fact.")
    print(" [C]   Gasket is p.c.f.: KZ applies. Hino carpet singularity does")
    print("       NOT cover it; Γ status needs a separate reference.")
    print(" [O]   Vicsek / Cantor dust: (I1)-(I4) are exactly the UNPROVED")
    print("       content. Theorem 10 does NOT apply. Γ⊥µ MUST NOT be")
    print("       assumed. ρ_eff is an exponent-matched transplant.")
    print(" [D]   closure_transferred is hard-coded False for all non-anchor")
    print("       sets. Validate against an analytic reference (Thm 3(b)")
    print("       discipline) before any production use.")
    print(" [D]   WebSocket router is LWW, not a full CRDT. No TLS/auth here.")
    print(" [D]   Rendering is display-only and NOT differentiable.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved (R_cross: 1.66,1.33,1.27,")
    print("       1.26,1.25). All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
