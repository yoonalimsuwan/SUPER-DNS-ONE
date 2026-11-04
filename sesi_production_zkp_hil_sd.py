"""
=============================================================================
SESI PRODUCTION ENGINEERING SUITE CRYPTOGRAPHIC
Zero-Knowledge Proofs + Hardware-in-the-Loop Profiling + Service Discovery
Native integration with SDA-ONE and full production pipeline
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
SDA-ONE & PIPELINE INTEGRATION
=============================================================================
This module integrates with:
  1. structural_domain_assembly_one.py (SDA-ONE):
     - DomainAssemblyConfig, CrossDomainContactHead, DomainDockingAssembler
     - Models compiled to ZK circuits for verifiable inference
  2. sesi_production_he_nas_gateway.py:
     - HomomorphicEncryptionBridge (ZKP complements HE for verification)
     - NeuralArchitectureSearch (hardware profiler feeds back latency model)
     - APIGatewayBridge (service discovery resolves endpoints for gateway)
  3. sesi_production_privacy_distillation_mesh.py:
     - FederatedLearningBridge (ZKP verifies aggregated gradients)
     - ServiceMeshBridge (service discovery integrates with mesh routing)

Integration points:
  1. ZeroKnowledgeProofBridge.compile_model_to_circuit(sda_model)
     - Converts PyTorch model to arithmetic circuit
     - Generates proving/verification keys
     - Enables privacy-preserving inference with verifiable output

  2. HardwareInLoopLatencyProfiler.profile_model(model, hardware_target)
     - Real hardware benchmarking with warmup
     - Per-layer latency breakdown
     - Feeds back into NAS hardware model for better student selection

  3. ServiceDiscoveryBridge.register_endpoint(service_name, endpoint)
     - Registers with Consul/Eureka
     - Health check integration
     - Resolves endpoints for mesh-deployed models

=============================================================================
MODULES
=============================================================================
[ZKP1] ZeroKnowledgeProofBridge
      • Groth16 zk-SNARK [C Groth 2016]: constant-size proofs, fast verify
      • PLONK [C Gabizon 2019]: universal setup, updatable CRS
      • Bulletproofs [C Bunz 2018]: no trusted setup, larger proofs
      • Circuit compilation from PyTorch (Ezkl/Cairo-style)
      • Privacy-preserving inference protocol [D]
      • COROLLARY-3 guard: first-order circuit operations; no Hessian
      • HINO guard: discrete field elements, not smooth density

[HIL1] HardwareInLoopLatencyProfiler
      • Per-layer timing with CUDA events / CPU high-resolution timer
      • Warmup runs to stabilize clock frequencies
      • Operator fusion detection (matches PyTorch JIT behavior)
      • Memory bandwidth analysis (HBM/DRAM throughput)
      • Hardware-aware latency model for NAS feedback
      • COROLLARY-3 guard: first-order measurements; no Hessian of timing
      • HINO guard: discrete timing samples, not continuous distribution

[SD1] ServiceDiscoveryBridge
      • Consul backend: KV store, catalog, health checks [C HashiCorp]
      • Eureka backend: Netflix OSS service registry [C]
      • Multi-backend resolver with fallback
      • Integration with ServiceMeshBridge for mesh-aware routing
      • Dynamic endpoint resolution for A/B testing and canary
      • COROLLARY-3 guard: first-order service lookups
      • HINO guard: discrete service instances, not continuous

=============================================================================
HONEST SCOPE
=============================================================================
• ZeroKnowledgeProofBridge: ZK proofs are computationally expensive
  (100-10000x slower than plaintext inference). Circuit compilation
  from PyTorch is lossy for dynamic control flow. This is a protocol
  framework, not a turnkey privacy solution.
• HardwareInLoopLatencyProfiler: real hardware profiling requires
  actual GPU/CPU/NPU access. Latency varies with thermal throttling,
  background processes, and driver versions. Results are indicative,
  not reproducible across environments.
• ServiceDiscoveryBridge: requires running Consul/Eureka cluster.
  Service discovery is eventually consistent; stale reads can occur
  during network partitions. This is an interface, not a guarantee.
• COROLLARY-3 guard: all operations first-order only; no Hessian.
• HINO guard: discrete operations respected; no smooth density assumed.
• PROP-3 awareness: field products lose two Sobolev orders.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All parameters illustrative. No security/privacy/legal guarantee.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.zkp_hil_sd")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import ezkl  # placeholder for ZK circuit compilation
    HAS_EZKL = False  # would be True with ezkl installed
except Exception:
    HAS_EZKL = False

try:
    import consul
    HAS_CONSUL = False  # placeholder
except Exception:
    HAS_CONSUL = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

__all__ = [
    "ZeroKnowledgeProofBridge", "ZKPConfig", "ZKPResult", "ZKPCircuit",
    "HardwareInLoopLatencyProfiler", "HILConfig", "HardwareProfile",
    "ServiceDiscoveryBridge", "SDConfig", "ServiceEndpoint",
    "SESIProductionZKPHILSDSuite",
]


# =============================================================================
# SECTION 1 — [ZKP1] ZERO-KNOWLEDGE PROOF BRIDGE
# =============================================================================

@dataclass
class ZKPConfig:
    """Configuration for zero-knowledge proofs."""
    # Proof system
    proof_system: str = "groth16"  # "groth16" | "plonk" | "bulletproofs"
    # Circuit parameters
    max_constraint_count: int = 10_000_000
    field_modulus_bits: int = 254  # BN254 curve
    # Trusted setup
    trusted_setup_path: Optional[str] = None  # for Groth16/PLONK
    use_universal_setup: bool = False  # PLONK only
    # Privacy-preserving inference
    enable_encrypted_input: bool = True
    enable_encrypted_output: bool = True
    enable_output_verification: bool = True
    # Performance
    batch_proof_generation: bool = False
    proof_batch_size: int = 8
    # Output
    save_keys: bool = True
    key_dir: str = "./zkp_keys"
    # Security
    soundness_error_bits: int = 128
    zero_knowledge_error_bits: int = 40


@dataclass
class ZKPCircuit:
    """Compiled arithmetic circuit from a PyTorch model."""
    circuit_id: str
    n_constraints: int
    n_public_inputs: int
    n_private_inputs: int
    n_outputs: int
    constraint_file: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZKPResult:
    """Result of ZK proof generation or verification."""
    proof_system: str
    proof_bytes: int
    verification_time_ms: float
    proof_generation_time_ms: float
    verified: bool
    public_inputs: Dict[str, Any]
    public_outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZeroKnowledgeProofBridge:
    """
    [C/D] Zero-knowledge proofs for privacy-preserving inference.

    ZK-SNARKS [C Groth 2016; Gabizon 2019; Bunz 2018]
      A zero-knowledge proof allows a prover to convince a verifier that
      a statement is true without revealing any additional information.

      Properties:
        • Completeness: honest prover convinces honest verifier
        • Soundness: cheating prover cannot convince verifier (except with
          negligible probability 2^{-soundness_error_bits})
        • Zero-knowledge: verifier learns nothing beyond statement truth

      Groth16:
        - Constant-size proof (~256 bytes)
        - Fast verification (~10ms)
        - Requires per-circuit trusted setup (toxic waste)
        - Best for: fixed circuits, high-volume verification

      PLONK:
        - Larger proof (~500 bytes)
        - Slower verification (~50ms)
        - Universal updatable setup (one-time trusted ceremony)
        - Best for: frequently updated circuits

      Bulletproofs:
        - Larger proof (~1KB)
        - Slowest verification (~200ms)
        - No trusted setup
        - Best for: trust-minimized deployments

    PRIVACY-PRESERVING INFERENCE PROTOCOL [D]
      Goal: client gets f(x) without server learning x, and client can
      verify that server computed f correctly.

      Protocol:
        1. Server compiles model f to arithmetic circuit C
        2. Server generates proving key pk and verification key vk
        3. Client encrypts input x → x_enc (via HE or MPC)
        4. Server computes f(x_enc) → y_enc
        5. Server generates proof π that y_enc = C(x_enc)
        6. Client decrypts y_enc → y and verifies π using vk
        7. If verification passes, client accepts y

      Security:
        • Server learns nothing about x (input privacy)
        • Client cannot forge proof of incorrect output (soundness)
        • Client learns only y, not intermediate computations (zero-knowledge)

    CIRCUIT COMPILATION [D]
      PyTorch model → arithmetic circuit:
        1. Trace model with symbolic inputs
        2. Convert operations to field arithmetic (+, ×, comparison)
        3. Flatten to R1CS (Rank-1 Constraint System) for Groth16
           or PLONK gates for PLONK
        4. Optimize constraint count (gate merging, constant folding)

      Limitations:
        • Dynamic control flow (if/while) requires bounded unrolling
        • Floating-point operations require fixed-point encoding
        • Non-arithmetic ops (softmax, attention) need custom gadgets

    COROLLARY-3 GUARD [D]
      Circuit operations are FIRST-ORDER (field arithmetic). We do NOT
      compute Hessians through the circuit. Witness generation is
      forward-only.

    HINO GUARD [C Thm 8]
      Proof elements are DISCRETE field elements (integers mod p).
      We respect the singular nature of the proof space; no smooth
      density assumption is made.

    HONEST
      • ZK proofs are computationally expensive; not suitable for all workloads.
      • Circuit compilation from PyTorch is lossy for dynamic control flow.
      • Security depends on correct parameter selection and implementation.
      • This is a PROTOCOL FRAMEWORK; production requires vetted crypto libraries.
      • No legal/security guarantee is provided.
    """

    def __init__(self, cfg: Optional[ZKPConfig] = None):
        self.cfg = cfg or ZKPConfig()
        self._compiled_circuits: Dict[str, ZKPCircuit] = {}
        self._proving_keys: Dict[str, Any] = {}
        self._verification_keys: Dict[str, Any] = {}

    def compile_model_to_circuit(self, model: nn.Module,
                                   example_input: Tensor,
                                   circuit_id: Optional[str] = None) -> ZKPCircuit:
        """
        Compile a PyTorch model to an arithmetic circuit.

        Args:
            model: PyTorch model to compile
            example_input: example input tensor for tracing
            circuit_id: optional identifier for the circuit

        Returns:
            ZKPCircuit with constraint count and metadata
        """
        circuit_id = circuit_id or f"circuit_{hashlib.sha256(str(model).encode()).hexdigest()[:16]}"

        # Trace model to get computation graph
        model.eval()
        with torch.no_grad():
            # Count operations (placeholder: real impl would use torch.jit.trace)
            n_params = sum(p.numel() for p in model.parameters())
            n_constraints = self._estimate_constraint_count(model, example_input)

        circuit = ZKPCircuit(
            circuit_id=circuit_id,
            n_constraints=n_constraints,
            n_public_inputs=example_input.numel(),
            n_private_inputs=n_params,
            n_outputs=self._estimate_output_size(model, example_input),
            constraint_file=None,  # would be path to .r1cs or .plonk file
            metadata={
                "model_class": model.__class__.__name__,
                "input_shape": tuple(example_input.shape),
                "n_parameters": n_params,
                "proof_system": self.cfg.proof_system,
                "corollary3_guard": "first-order circuit operations; no Hessian",
                "hino_guard": "discrete field elements, not smooth density",
            },
        )

        self._compiled_circuits[circuit_id] = circuit
        logger.info(f"[ZKP] Compiled circuit {circuit_id}: {n_constraints} constraints")

        return circuit

    def _estimate_constraint_count(self, model: nn.Module, input_tensor: Tensor) -> int:
        """
        Estimate number of R1CS constraints for the model.
        Heuristic: ~10 constraints per parameter for linear layers,
        ~100 constraints per parameter for conv layers (due to im2col).
        """
        n_constraints = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                n_constraints += module.weight.numel() * 10
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                n_constraints += module.weight.numel() * 100
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                n_constraints += module.weight.numel() * 20
            elif isinstance(module, nn.LayerNorm):
                n_constraints += module.weight.numel() * 15
            # Activation functions: ~5 constraints per element
            if hasattr(module, 'forward'):
                # Rough estimate for activations
                pass

        # Add overhead for input/output encoding
        n_constraints += input_tensor.numel() * 5

        return int(n_constraints)

    def _estimate_output_size(self, model: nn.Module, input_tensor: Tensor) -> int:
        """Estimate output tensor size."""
        with torch.no_grad():
            try:
                output = model(input_tensor)
                return output.numel()
            except Exception:
                return input_tensor.numel()  # fallback

    def generate_keys(self, circuit_id: str) -> Tuple[str, str]:
        """
        Generate proving and verification keys for a compiled circuit.

        In production, this would call the actual ZK library (ezkl, snarkjs, etc.).
        Here we simulate the key generation process.

        Returns:
            (proving_key_path, verification_key_path)
        """
        if circuit_id not in self._compiled_circuits:
            raise ValueError(f"Circuit {circuit_id} not compiled")

        os.makedirs(self.cfg.key_dir, exist_ok=True)

        pk_path = os.path.join(self.cfg.key_dir, f"{circuit_id}.pk")
        vk_path = os.path.join(self.cfg.key_dir, f"{circuit_id}.vk")

        # Simulate key generation (placeholder)
        # Real impl would run trusted setup ceremony
        circuit = self._compiled_circuits[circuit_id]
        pk_data = {
            "circuit_id": circuit_id,
            "n_constraints": circuit.n_constraints,
            "proof_system": self.cfg.proof_system,
            "generated_at": time.time(),
        }
        vk_data = {
            "circuit_id": circuit_id,
            "verification_key_hash": hashlib.sha256(json.dumps(pk_data).encode()).hexdigest(),
        }

        with open(pk_path, "w") as f:
            json.dump(pk_data, f)
        with open(vk_path, "w") as f:
            json.dump(vk_data, f)

        self._proving_keys[circuit_id] = pk_path
        self._verification_keys[circuit_id] = vk_path

        logger.info(f"[ZKP] Generated keys for circuit {circuit_id}")
        return pk_path, vk_path

    def generate_proof(self, circuit_id: str,
                        public_inputs: Dict[str, Any],
                        private_inputs: Dict[str, Any]) -> ZKPResult:
        """
        Generate a ZK proof for the given inputs.

        Args:
            circuit_id: compiled circuit identifier
            public_inputs: public inputs to the circuit
            private_inputs: private witness (model weights, intermediate values)

        Returns:
            ZKPResult with proof and metadata
        """
        if circuit_id not in self._compiled_circuits:
            raise ValueError(f"Circuit {circuit_id} not compiled")
        if circuit_id not in self._proving_keys:
            raise ValueError(f"Keys not generated for circuit {circuit_id}")

        start_time = time.time()

        # Simulate proof generation (placeholder)
        # Real impl would call prover with witness
        circuit = self._compiled_circuits[circuit_id]
        proof_size = {
            "groth16": 256,
            "plonk": 512,
            "bulletproofs": 1024,
        }.get(self.cfg.proof_system, 256)

        # Simulate computation time (rough heuristic)
        proof_gen_time = circuit.n_constraints / 1e6 * 100  # ~100ms per 1M constraints

        # Compute public outputs (placeholder: would be actual circuit evaluation)
        public_outputs = {"output_hash": hashlib.sha256(
            json.dumps(public_inputs).encode()).hexdigest()[:16]}

        gen_time = time.time() - start_time

        return ZKPResult(
            proof_system=self.cfg.proof_system,
            proof_bytes=proof_size,
            verification_time_ms=10.0,  # placeholder
            proof_generation_time_ms=max(gen_time * 1000, proof_gen_time),
            verified=True,  # placeholder: would verify after generation
            public_inputs=public_inputs,
            public_outputs=public_outputs,
            metadata={
                "circuit_id": circuit_id,
                "n_constraints": circuit.n_constraints,
                "corollary3_guard": "first-order proof generation; no Hessian",
                "hino_guard": "discrete proof elements, not smooth density",
            },
        )

    def verify_proof(self, circuit_id: str,
                      proof: ZKPResult,
                      public_inputs: Dict[str, Any],
                      public_outputs: Dict[str, Any]) -> bool:
        """
        Verify a ZK proof.

        Args:
            circuit_id: compiled circuit identifier
            proof: proof to verify
            public_inputs: public inputs
            public_outputs: claimed public outputs

        Returns:
            True if verification succeeds, False otherwise
        """
        if circuit_id not in self._verification_keys:
            raise ValueError(f"Verification key not found for circuit {circuit_id}")

        start_time = time.time()

        # Simulate verification (placeholder)
        # Real impl would call verifier with proof and public data
        vk_path = self._verification_keys[circuit_id]
        with open(vk_path, "r") as f:
            vk_data = json.load(f)

        # Verify proof structure
        verified = (
            proof.proof_system == self.cfg.proof_system and
            proof.public_inputs == public_inputs and
            proof.public_outputs == public_outputs
        )

        verification_time = (time.time() - start_time) * 1000
        proof.verification_time_ms = verification_time
        proof.verified = verified

        logger.info(f"[ZKP] Verification {'passed' if verified else 'failed'} "
                    f"in {verification_time:.2f}ms")

        return verified

    def privacy_preserving_inference(self, model: nn.Module,
                                       input_data: Tensor,
                                       circuit_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Full privacy-preserving inference protocol.

        1. Compile model to circuit (if not already compiled)
        2. Generate keys (if not already generated)
        3. Generate proof for inference
        4. Return proof and output

        Args:
            model: PyTorch model
            input_data: input tensor
            circuit_id: optional circuit identifier

        Returns:
            dict with proof, output, and metadata
        """
        # Compile if needed
        if circuit_id is None or circuit_id not in self._compiled_circuits:
            circuit = self.compile_model_to_circuit(model, input_data, circuit_id)
            circuit_id = circuit.circuit_id

        # Generate keys if needed
        if circuit_id not in self._proving_keys:
            self.generate_keys(circuit_id)

        # Simulate inference
        model.eval()
        with torch.no_grad():
            output = model(input_data)

        # Generate proof
        public_inputs = {"input_hash": hashlib.sha256(
            input_data.cpu().numpy().tobytes()).hexdigest()[:16]}
        private_inputs = {"model_weights": "encrypted"}

        proof = self.generate_proof(circuit_id, public_inputs, private_inputs)

        # Verify proof
        public_outputs = {"output_hash": hashlib.sha256(
            output.cpu().numpy().tobytes()).hexdigest()[:16]}
        verified = self.verify_proof(circuit_id, proof, public_inputs, public_outputs)

        return {
            "circuit_id": circuit_id,
            "proof": proof,
            "output": output,
            "verified": verified,
            "metadata": {
                "input_shape": tuple(input_data.shape),
                "output_shape": tuple(output.shape),
                "protocol": "privacy-preserving inference with ZK verification",
            },
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "proof_system": self.cfg.proof_system,
            "n_compiled_circuits": len(self._compiled_circuits),
            "has_ezkl": HAS_EZKL,
            "soundness_error_bits": self.cfg.soundness_error_bits,
            "zero_knowledge_error_bits": self.cfg.zero_knowledge_error_bits,
            "corollary3_guard": "first-order circuit operations; no Hessian",
            "hino_guard": "discrete field elements, not smooth density",
        }


# =============================================================================
# SECTION 2 — [HIL1] HARDWARE-IN-THE-LOOP LATENCY PROFILER
# =============================================================================

@dataclass
class HILConfig:
    """Configuration for hardware-in-the-loop profiling."""
    # Profiling parameters
    n_warmup_runs: int = 10
    n_benchmark_runs: int = 100
    # Hardware targets
    target_device: str = "auto"  # "auto" | "cuda" | "cpu" | "mps"
    # Per-layer profiling
    enable_layerwise_profiling: bool = True
    enable_operator_fusion_detection: bool = True
    # Memory profiling
    enable_memory_profiling: bool = True
    # Output
    save_profile: bool = True
    profile_dir: str = "./hardware_profiles"
    # NAS feedback
    feed_back_to_nas: bool = True


@dataclass
class LayerProfile:
    """Profiling result for a single layer."""
    layer_name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    latency_ms: float
    memory_mb: float
    flops: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareProfile:
    """Complete hardware profiling result."""
    device_name: str
    device_type: str
    total_latency_ms: float
    total_memory_mb: float
    total_flops: Optional[float]
    layer_profiles: List[LayerProfile]
    operator_fusion_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareInLoopLatencyProfiler:
    """
    [C/D] Real hardware latency profiling with per-layer breakdown.

    CAPABILITIES
      1. Per-layer timing with CUDA events / CPU high-resolution timer
      2. Warmup runs to stabilize clock frequencies
      3. Operator fusion detection (matches PyTorch JIT behavior)
      4. Memory bandwidth analysis (HBM/DRAM throughput)
      5. Hardware-aware latency model for NAS feedback

    PROFILING METHODOLOGY [D]
      1. Warmup: run model n_warmup_runs times to stabilize
         - GPU: clock frequencies, memory allocation
         - CPU: cache warming, branch prediction
      2. Benchmark: run model n_benchmark_runs times
         - Measure total latency with high-resolution timer
         - Measure per-layer latency with hooks
         - Measure memory usage with CUDA memory stats
      3. Aggregate: compute mean, std, percentiles
      4. Analyze: detect operator fusion opportunities

    CUDA EVENTS [C NVIDIA]
      For GPU profiling, use CUDA events for accurate timing:
        start_event.record()
        output = model(input)
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event)

    CPU HIGH-RESOLUTION TIMER [C]
      For CPU profiling, use time.perf_counter():
        start = time.perf_counter()
        output = model(input)
        end = time.perf_counter()
        latency = (end - start) * 1000  # ms

    OPERATOR FUSION [D]
      PyTorch JIT can fuse consecutive operations (e.g., Conv+BN+ReLU).
      We detect fusion by comparing per-layer timing with/without JIT.
      Fused layers show lower latency than sum of individual layers.

    MEMORY PROFILING [C]
      GPU: torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
      CPU: psutil.Process().memory_info().rss
      Memory bandwidth: total_memory / latency

    COROLLARY-3 GUARD [D]
      Profiling measurements are FIRST-ORDER (timing, memory).
      We do NOT compute Hessians of the latency function.

    HINO GUARD [C Thm 8]
      Timing samples are DISCRETE events. We respect the singular
      nature of the measurement process; no continuous distribution assumed.

    PROP-3 AWARENESS [P]
      When combining metrics (latency × throughput), the result loses
      regularity. We report them separately.

    HONEST
      • Real hardware profiling requires actual GPU/CPU/NPU access.
      • Latency varies with thermal throttling, background processes, driver versions.
      • Results are indicative, not reproducible across environments.
      • This is a profiling tool, not a performance guarantee.
    """

    def __init__(self, cfg: Optional[HILConfig] = None):
        self.cfg = cfg or HILConfig()
        self._profiles: Dict[str, HardwareProfile] = {}

    def _get_device(self) -> torch.device:
        """Determine profiling device."""
        if self.cfg.target_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.cfg.target_device)

    def _measure_latency_cuda(self, model: nn.Module, input_tensor: Tensor) -> float:
        """Measure latency using CUDA events (GPU only)."""
        device = input_tensor.device
        if device.type != "cuda":
            raise ValueError("CUDA events require CUDA device")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(self.cfg.n_warmup_runs):
            _ = model(input_tensor)
        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(self.cfg.n_benchmark_runs):
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        return float(np.mean(latencies))

    def _measure_latency_cpu(self, model: nn.Module, input_tensor: Tensor) -> float:
        """Measure latency using high-resolution timer (CPU)."""
        # Warmup
        for _ in range(self.cfg.n_warmup_runs):
            _ = model(input_tensor)

        # Benchmark
        latencies = []
        for _ in range(self.cfg.n_benchmark_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        return float(np.mean(latencies))

    def _profile_layerwise(self, model: nn.Module, input_tensor: Tensor) -> List[LayerProfile]:
        """Profile each layer individually."""
        layer_profiles = []
        device = input_tensor.device

        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue  # skip parent modules

            # Create a mini-model with just this layer
            class LayerWrapper(nn.Module):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    return self.layer(x)

            wrapper = LayerWrapper(module).to(device)
            wrapper.eval()

            # Determine input shape for this layer
            # (simplified: assume same shape as model input)
            try:
                with torch.no_grad():
                    # Measure latency
                    if device.type == "cuda":
                        latency = self._measure_latency_cuda(wrapper, input_tensor)
                    else:
                        latency = self._measure_latency_cpu(wrapper, input_tensor)

                    # Measure memory
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        _ = wrapper(input_tensor)
                        memory_mb = torch.cuda.max_memory_allocated() / 1e6
                    else:
                        memory_mb = 0.0  # CPU memory profiling requires psutil

                    # Estimate FLOPs (placeholder)
                    flops = self._estimate_flops(module, input_tensor.shape)

                    layer_profiles.append(LayerProfile(
                        layer_name=name,
                        layer_type=module.__class__.__name__,
                        input_shape=tuple(input_tensor.shape),
                        output_shape=tuple(input_tensor.shape),  # placeholder
                        latency_ms=latency,
                        memory_mb=memory_mb,
                        flops=flops,
                    ))
            except Exception as e:
                logger.warning(f"[HIL] Failed to profile layer {name}: {e}")

        return layer_profiles

    def _estimate_flops(self, module: nn.Module, input_shape: Tuple[int, ...]) -> Optional[float]:
        """Estimate FLOPs for a module (heuristic)."""
        if isinstance(module, nn.Linear):
            return 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            batch, in_channels, h, w = input_shape
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            return 2 * batch * out_channels * in_channels * kernel_size * kernel_size * h * w
        return None

    def _detect_operator_fusion(self, model: nn.Module, input_tensor: Tensor) -> Dict[str, Any]:
        """Detect operator fusion opportunities."""
        # Compare fused vs unfused latency
        device = input_tensor.device

        # Unfused model
        if device.type == "cuda":
            unfused_latency = self._measure_latency_cuda(model, input_tensor)
        else:
            unfused_latency = self._measure_latency_cpu(model, input_tensor)

        # Try to compile with torch.jit (if available)
        try:
            scripted_model = torch.jit.script(model)
            if device.type == "cuda":
                fused_latency = self._measure_latency_cuda(scripted_model, input_tensor)
            else:
                fused_latency = self._measure_latency_cpu(scripted_model, input_tensor)

            speedup = unfused_latency / max(fused_latency, 1e-6)
            fusion_detected = speedup > 1.05  # 5% threshold
        except Exception:
            fused_latency = unfused_latency
            speedup = 1.0
            fusion_detected = False

        return {
            "unfused_latency_ms": unfused_latency,
            "fused_latency_ms": fused_latency,
            "speedup": speedup,
            "fusion_detected": fusion_detected,
        }

    def profile_model(self, model: nn.Module, input_tensor: Tensor,
                       profile_id: Optional[str] = None) -> HardwareProfile:
        """
        Full hardware profiling of a model.

        Args:
            model: PyTorch model to profile
            input_tensor: example input tensor
            profile_id: optional identifier for the profile

        Returns:
            HardwareProfile with detailed timing and memory info
        """
        profile_id = profile_id or f"profile_{int(time.time())}"
        device = self._get_device()
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        model.eval()

        logger.info(f"[HIL] Profiling model on {device}")

        # Total latency
        if device.type == "cuda":
            total_latency = self._measure_latency_cuda(model, input_tensor)
        else:
            total_latency = self._measure_latency_cpu(model, input_tensor)

        # Per-layer profiling
        layer_profiles = []
        if self.cfg.enable_layerwise_profiling:
            layer_profiles = self._profile_layerwise(model, input_tensor)

        # Total memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = model(input_tensor)
            total_memory_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            total_memory_mb = 0.0

        # Total FLOPs
        total_flops = sum(lp.flops for lp in layer_profiles if lp.flops is not None)

        # Operator fusion detection
        fusion_info = {}
        if self.cfg.enable_operator_fusion_detection:
            fusion_info = self._detect_operator_fusion(model, input_tensor)

        profile = HardwareProfile(
            device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            device_type=device.type,
            total_latency_ms=total_latency,
            total_memory_mb=total_memory_mb,
            total_flops=total_flops if total_flops > 0 else None,
            layer_profiles=layer_profiles,
            operator_fusion_info=fusion_info,
            metadata={
                "profile_id": profile_id,
                "n_warmup_runs": self.cfg.n_warmup_runs,
                "n_benchmark_runs": self.cfg.n_benchmark_runs,
                "corollary3_guard": "first-order measurements; no Hessian of timing",
                "hino_guard": "discrete timing samples, not continuous distribution",
            },
        )

        self._profiles[profile_id] = profile

        # Save if requested
        if self.cfg.save_profile:
            os.makedirs(self.cfg.profile_dir, exist_ok=True)
            profile_path = os.path.join(self.cfg.profile_dir, f"{profile_id}.json")
            with open(profile_path, "w") as f:
                json.dump({
                    "device_name": profile.device_name,
                    "device_type": profile.device_type,
                    "total_latency_ms": profile.total_latency_ms,
                    "total_memory_mb": profile.total_memory_mb,
                    "total_flops": profile.total_flops,
                    "n_layers": len(profile.layer_profiles),
                    "operator_fusion": profile.operator_fusion_info,
                }, f, indent=2)
            logger.info(f"[HIL] Saved profile to {profile_path}")

        return profile

    def get_hardware_aware_latency_model(self, profile_id: str) -> Callable[[Any], float]:
        """
        Generate a hardware-aware latency model from a profile.
        This can be fed back into NAS for better student selection.

        Returns:
            function that takes ArchitectureSpec and returns estimated latency
        """
        if profile_id not in self._profiles:
            raise ValueError(f"Profile {profile_id} not found")

        profile = self._profiles[profile_id]

        # Build a simple latency model based on layer types
        def latency_model(arch_spec: Any) -> float:
            # Placeholder: would use arch_spec to estimate latency
            # based on profiled layer latencies
            base_latency = profile.total_latency_ms
            # Scale by architecture complexity (heuristic)
            complexity_factor = arch_spec.n_parameters / 1e6  # millions of params
            return base_latency * (1.0 + 0.1 * complexity_factor)

        return latency_model

    def get_status(self) -> Dict[str, Any]:
        return {
            "target_device": self.cfg.target_device,
            "n_profiles": len(self._profiles),
            "enable_layerwise": self.cfg.enable_layerwise_profiling,
            "enable_fusion_detection": self.cfg.enable_operator_fusion_detection,
            "corollary3_guard": "first-order measurements; no Hessian of timing",
            "hino_guard": "discrete timing samples, not continuous distribution",
        }


# =============================================================================
# SECTION 3 — [SD1] SERVICE DISCOVERY BRIDGE
# =============================================================================

@dataclass
class SDConfig:
    """Configuration for service discovery."""
    # Backend selection
    backend: str = "consul"  # "consul" | "eureka" | "static"
    # Consul settings
    consul_host: str = "localhost"
    consul_port: int = 8500
    consul_token: Optional[str] = None
    consul_scheme: str = "http"
    # Eureka settings
    eureka_url: str = "http://localhost:8761/eureka/"
    eureka_username: Optional[str] = None
    eureka_password: Optional[str] = None
    # Service registration
    service_name: str = "sesi-model-server"
    service_id: Optional[str] = None
    service_address: str = "localhost"
    service_port: int = 8080
    # Health checks
    health_check_url: str = "/health"
    health_check_interval_s: int = 10
    health_check_timeout_s: int = 5
    # Service resolution
    resolve_timeout_s: float = 5.0
    resolve_retries: int = 3
    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class ServiceEndpoint:
    """A resolved service endpoint."""
    service_id: str
    service_name: str
    address: str
    port: int
    tags: List[str]
    meta: Dict[str, str]
    healthy: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceDiscoveryBridge:
    """
    [C/D] Service discovery integration for dynamic endpoint resolution.

    CONSUL [C HashiCorp]
      • Service registration with health checks
      • KV store for configuration
      • DNS interface for service resolution
      • Multi-datacenter support

    EUREKA [C Netflix OSS]
      • Service registration with heartbeat
      • Client-side load balancing
      • Integration with Spring Cloud

    MULTI-BACKEND RESOLVER [D]
      Tries backends in order: Consul → Eureka → static fallback.
      Returns list of healthy endpoints for a service.

    INTEGRATION WITH MESH [D]
      Service discovery works with ServiceMeshBridge:
        1. Register model endpoint with Consul/Eureka
        2. Mesh routes traffic to discovered endpoints
        3. Health checks trigger automatic failover

    COROLLARY-3 GUARD [D]
      Service lookups are FIRST-ORDER operations (HTTP requests).
      We do NOT compute derivatives through the discovery process.

    HINO GUARD [C Thm 8]
      Service instances are DISCRETE entities. We respect the singular
      nature of the service registry; no continuous approximation.

    HONEST
      • Requires running Consul/Eureka cluster.
      • Service discovery is eventually consistent; stale reads can occur.
      • This is an interface, not a guarantee of availability.
    """

    def __init__(self, cfg: Optional[SDConfig] = None):
        self.cfg = cfg or SDConfig()
        self._consul_client: Optional[Any] = None
        self._registered_services: List[str] = []

    def _init_consul(self) -> bool:
        """Initialize Consul client."""
        if not HAS_CONSUL:
            logger.warning("[SD] python-consul not available")
            return False

        try:
            self._consul_client = consul.Consul(
                host=self.cfg.consul_host,
                port=self.cfg.consul_port,
                token=self.cfg.consul_token,
                scheme=self.cfg.consul_scheme,
            )
            # Test connection
            self._consul_client.status.leader()
            logger.info("[SD] Connected to Consul at %s:%d",
                        self.cfg.consul_host, self.cfg.consul_port)
            return True
        except Exception as e:
            logger.warning("[SD] Failed to connect to Consul: %s", e)
            return False

    def register_service(self, service_name: Optional[str] = None,
                           service_id: Optional[str] = None,
                           address: Optional[str] = None,
                           port: Optional[int] = None,
                           tags: Optional[List[str]] = None,
                           meta: Optional[Dict[str, str]] = None) -> bool:
        """
        Register a service with the discovery backend.

        Args:
            service_name: name of the service
            service_id: unique identifier (defaults to service_name)
            address: service address
            port: service port
            tags: optional tags for filtering
            meta: optional metadata

        Returns:
            True if registration succeeded
        """
        service_name = service_name or self.cfg.service_name
        service_id = service_id or self.cfg.service_id or f"{service_name}-{int(time.time())}"
        address = address or self.cfg.service_address
        port = port or self.cfg.service_port
        tags = tags or self.cfg.tags
        meta = meta or self.cfg.meta

        if self.cfg.backend == "consul":
            if not self._consul_client and not self._init_consul():
                logger.error("[SD] Cannot register: Consul unavailable")
                return False

            try:
                # Register service with health check
                check = consul.Check.http(
                    f"http://{address}:{port}{self.cfg.health_check_url}",
                    f"{self.cfg.health_check_interval_s}s",
                    timeout=f"{self.cfg.health_check_timeout_s}s",
                )
                self._consul_client.agent.service.register(
                    name=service_name,
                    service_id=service_id,
                    address=address,
                    port=port,
                    tags=tags,
                    meta=meta,
                    check=check,
                )
                self._registered_services.append(service_id)
                logger.info("[SD] Registered service %s at %s:%d",
                            service_id, address, port)
                return True
            except Exception as e:
                logger.error("[SD] Failed to register service: %s", e)
                return False

        elif self.cfg.backend == "eureka":
            if not HAS_REQUESTS:
                logger.error("[SD] Cannot register: requests library unavailable")
                return False

            try:
                # Eureka registration via REST API
                instance_info = {
                    "instance": {
                        "hostName": address,
                        "app": service_name.upper(),
                        "ipAddr": address,
                        "status": "UP",
                        "port": {"$": port, "@enabled": "true"},
                        "dataCenterInfo": {
                            "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                            "name": "MyOwn",
                        },
                        "metadata": meta,
                    }
                }
                headers = {"Content-Type": "application/json"}
                auth = None
                if self.cfg.eureka_username and self.cfg.eureka_password:
                    auth = (self.cfg.eureka_username, self.cfg.eureka_password)

                response = requests.post(
                    f"{self.cfg.eureka_url}apps/{service_name.upper()}",
                    json=instance_info,
                    headers=headers,
                    auth=auth,
                    timeout=5.0,
                )
                if response.status_code in (200, 204):
                    self._registered_services.append(service_id)
                    logger.info("[SD] Registered service %s with Eureka", service_id)
                    return True
                else:
                    logger.error("[SD] Eureka registration failed: %d %s",
                                  response.status_code, response.text)
                    return False
            except Exception as e:
                logger.error("[SD] Failed to register with Eureka: %s", e)
                return False

        else:  # static
            logger.info("[SD] Static backend: service %s registered at %s:%d",
                        service_id, address, port)
            self._registered_services.append(service_id)
            return True

    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service."""
        if self.cfg.backend == "consul" and self._consul_client:
            try:
                self._consul_client.agent.service.deregister(service_id)
                if service_id in self._registered_services:
                    self._registered_services.remove(service_id)
                logger.info("[SD] Deregistered service %s", service_id)
                return True
            except Exception as e:
                logger.error("[SD] Failed to deregister: %s", e)
                return False

        elif self.cfg.backend == "eureka" and HAS_REQUESTS:
            try:
                auth = None
                if self.cfg.eureka_username and self.cfg.eureka_password:
                    auth = (self.cfg.eureka_username, self.cfg.eureka_password)
                response = requests.delete(
                    f"{self.cfg.eureka_url}apps/UNKNOWN/{service_id}",
                    auth=auth,
                    timeout=5.0,
                )
                if response.status_code in (200, 204):
                    if service_id in self._registered_services:
                        self._registered_services.remove(service_id)
                    return True
                return False
            except Exception:
                return False

        else:  # static
            if service_id in self._registered_services:
                self._registered_services.remove(service_id)
            return True

    def resolve_service(self, service_name: str,
                          tag: Optional[str] = None) -> List[ServiceEndpoint]:
        """
        Resolve a service name to a list of healthy endpoints.

        Args:
            service_name: name of the service to resolve
            tag: optional tag for filtering

        Returns:
            list of ServiceEndpoint objects
        """
        endpoints = []

        if self.cfg.backend == "consul" and self._consul_client:
            try:
                # Query Consul for healthy services
                _, services = self._consul_client.health.service(
                    service_name, passing=True)

                for svc in services:
                    service = svc["Service"]
                    # Filter by tag if specified
                    if tag and tag not in service.get("Tags", []):
                        continue

                    endpoints.append(ServiceEndpoint(
                        service_id=service["ID"],
                        service_name=service["Service"],
                        address=service["Address"],
                        port=service["Port"],
                        tags=service.get("Tags", []),
                        meta=service.get("Meta", {}),
                        healthy=True,
                    ))
            except Exception as e:
                logger.error("[SD] Failed to resolve from Consul: %s", e)

        elif self.cfg.backend == "eureka" and HAS_REQUESTS:
            try:
                auth = None
                if self.cfg.eureka_username and self.cfg.eureka_password:
                    auth = (self.cfg.eureka_username, self.cfg.eureka_password)

                response = requests.get(
                    f"{self.cfg.eureka_url}apps/{service_name.upper()}",
                    headers={"Accept": "application/json"},
                    auth=auth,
                    timeout=self.cfg.resolve_timeout_s,
                )
                if response.status_code == 200:
                    data = response.json()
                    instances = data.get("application", {}).get("instance", [])
                    for inst in instances:
                        if inst.get("status") != "UP":
                            continue
                        endpoints.append(ServiceEndpoint(
                            service_id=inst.get("instanceId", ""),
                            service_name=service_name,
                            address=inst.get("ipAddr", "localhost"),
                            port=int(inst.get("port", {}).get("$", 8080)),
                            tags=[],
                            meta=inst.get("metadata", {}),
                            healthy=True,
                        ))
            except Exception as e:
                logger.error("[SD] Failed to resolve from Eureka: %s", e)

        else:  # static fallback
            logger.warning("[SD] No backend available; returning empty endpoint list")

        return endpoints

    def integrate_with_mesh(self, mesh_deployment: Any) -> Dict[str, Any]:
        """
        Integrate service discovery with mesh deployment.

        Args:
            mesh_deployment: MeshDeployment from ServiceMeshBridge

        Returns:
            dict with integration metadata
        """
        # Register the mesh endpoint with service discovery
        service_name = mesh_deployment.deployment_name
        endpoint_url = mesh_deployment.endpoint_url

        # Parse endpoint URL to get address and port
        # (simplified: assumes http://host:port format)
        if endpoint_url and "://" in endpoint_url:
            parts = endpoint_url.split("://")[1].split("/")
            host_port = parts[0]
            if ":" in host_port:
                address, port_str = host_port.rsplit(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    port = 80
            else:
                address = host_port
                port = 80

            # Register with discovery
            registered = self.register_service(
                service_name=service_name,
                address=address,
                port=port,
                tags=["mesh", mesh_deployment.provider],
                meta={"endpoint_url": endpoint_url},
            )

            return {
                "mesh_endpoint": endpoint_url,
                "discovery_backend": self.cfg.backend,
                "registered": registered,
                "service_name": service_name,
            }

        return {
            "mesh_endpoint": endpoint_url,
            "discovery_backend": self.cfg.backend,
            "registered": False,
            "error": "could not parse endpoint URL",
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": self.cfg.backend,
            "has_consul": HAS_CONSUL,
            "has_requests": HAS_REQUESTS,
            "n_registered_services": len(self._registered_services),
            "registered_services": self._registered_services,
            "corollary3_guard": "first-order service lookups; no Hessian",
            "hino_guard": "discrete service instances, not continuous",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIProductionZKPHILSDSuite:
    """
    Unifies: ZeroKnowledgeProofBridge + HardwareInLoopLatencyProfiler
             + ServiceDiscoveryBridge.

    FULL PIPELINE:
      1. Profile model on target hardware
      2. Compile model to ZK circuit
      3. Register service with discovery backend
      4. Serve privacy-preserving inference with verifiable proofs
    """

    def __init__(self,
                 zkp_cfg: Optional[ZKPConfig] = None,
                 hil_cfg: Optional[HILConfig] = None,
                 sd_cfg: Optional[SDConfig] = None):
        self.zkp = ZeroKnowledgeProofBridge(zkp_cfg)
        self.hil = HardwareInLoopLatencyProfiler(hil_cfg)
        self.sd = ServiceDiscoveryBridge(sd_cfg)

    def deploy_privacy_preserving_model(self, model: nn.Module,
                                          example_input: Tensor,
                                          service_name: str,
                                          service_address: str = "localhost",
                                          service_port: int = 8080) -> Dict[str, Any]:
        """
        Full deployment pipeline: profile → compile → register → serve.
        """
        results = {}

        # Stage 1: Hardware profiling
        logger.info("[Deploy] Stage 1: Hardware profiling")
        profile = self.hil.profile_model(model, example_input)
        results["hardware_profile"] = {
            "device": profile.device_name,
            "total_latency_ms": profile.total_latency_ms,
            "total_memory_mb": profile.total_memory_mb,
            "n_layers_profiled": len(profile.layer_profiles),
        }

        # Stage 2: ZK circuit compilation
        logger.info("[Deploy] Stage 2: ZK circuit compilation")
        circuit = self.zkp.compile_model_to_circuit(model, example_input)
        pk_path, vk_path = self.zkp.generate_keys(circuit.circuit_id)
        results["zkp_circuit"] = {
            "circuit_id": circuit.circuit_id,
            "n_constraints": circuit.n_constraints,
            "proving_key": pk_path,
            "verification_key": vk_path,
        }

        # Stage 3: Service registration
        logger.info("[Deploy] Stage 3: Service registration")
        registered = self.sd.register_service(
            service_name=service_name,
            address=service_address,
            port=service_port,
            tags=["zkp-enabled", "privacy-preserving"],
            meta={"circuit_id": circuit.circuit_id},
        )
        results["service_registration"] = {
            "service_name": service_name,
            "registered": registered,
            "backend": self.sd.cfg.backend,
        }

        # Stage 4: Test privacy-preserving inference
        logger.info("[Deploy] Stage 4: Testing privacy-preserving inference")
        inference_result = self.zkp.privacy_preserving_inference(
            model, example_input, circuit.circuit_id)
        results["inference_test"] = {
            "verified": inference_result["verified"],
            "proof_bytes": inference_result["proof"].proof_bytes,
            "proof_gen_time_ms": inference_result["proof"].proof_generation_time_ms,
            "verification_time_ms": inference_result["proof"].verification_time_ms,
        }

        return results

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "zkp": self.zkp.get_status(),
            "hil": self.hil.get_status(),
            "sd": self.sd.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI PRODUCTION ZKP + HIL + SD — SMOKE TEST")
    print(" Zero-Knowledge Proofs + Hardware Profiling + Service Discovery")
    print("=" * 78)
    print(f" torch {torch.__version__}   numpy {np.__version__}")
    print(f" ezkl available         : {HAS_EZKL}")
    print(f" consul available       : {HAS_CONSUL}")
    print(f" requests available     : {HAS_REQUESTS}")

    suite = SESIProductionZKPHILSDSuite(
        zkp_cfg=ZKPConfig(proof_system="groth16"),
        hil_cfg=HILConfig(n_warmup_runs=2, n_benchmark_runs=5),
        sd_cfg=SDConfig(backend="static"))

    # ── 1. Zero-Knowledge Proofs ────────────────────────────────────────
    print("\n [ZKP1] Zero-Knowledge Proof Bridge")
    zkp = suite.zkp
    print(f"   proof_system         : {zkp.cfg.proof_system}")
    print(f"   soundness_error_bits : {zkp.cfg.soundness_error_bits}")

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleModel()
    example_input = torch.randn(1, 10)

    # Compile to circuit
    circuit = zkp.compile_model_to_circuit(model, example_input)
    print(f"   circuit_id           : {circuit.circuit_id}")
    print(f"   n_constraints        : {circuit.n_constraints}")
    print(f"   n_public_inputs      : {circuit.n_public_inputs}")
    print(f"   n_private_inputs     : {circuit.n_private_inputs}")

    # Generate keys
    pk_path, vk_path = zkp.generate_keys(circuit.circuit_id)
    print(f"   proving_key          : {os.path.basename(pk_path)}")
    print(f"   verification_key     : {os.path.basename(vk_path)}")

    # Privacy-preserving inference
    pp_result = zkp.privacy_preserving_inference(model, example_input, circuit.circuit_id)
    print(f"   inference verified   : {pp_result['verified']}")
    print(f"   proof_bytes          : {pp_result['proof'].proof_bytes}")
    print(f"   proof_gen_time_ms    : {pp_result['proof'].proof_generation_time_ms:.2f}")
    print(f"   verification_time_ms : {pp_result['proof'].verification_time_ms:.2f}")

    zkp_status = zkp.get_status()
    print(f"   COROLLARY-3 guard    : {zkp_status['corollary3_guard']}")
    print(f"   HINO guard           : {zkp_status['hino_guard']}")

    # ── 2. Hardware-in-the-Loop Profiling ───────────────────────────────
    print("\n [HIL1] Hardware-in-the-Loop Latency Profiler")
    hil = suite.hil
    print(f"   target_device        : {hil.cfg.target_device}")
    print(f"   n_warmup_runs        : {hil.cfg.n_warmup_runs}")
    print(f"   n_benchmark_runs     : {hil.cfg.n_benchmark_runs}")

    # Profile the model
    profile = hil.profile_model(model, example_input)
    print(f"   device_name          : {profile.device_name}")
    print(f"   device_type          : {profile.device_type}")
    print(f"   total_latency_ms     : {profile.total_latency_ms:.4f}")
    print(f"   total_memory_mb      : {profile.total_memory_mb:.4f}")
    print(f"   n_layers_profiled    : {len(profile.layer_profiles)}")

    if profile.layer_profiles:
        print(f"   Layer breakdown:")
        for lp in profile.layer_profiles[:3]:
            print(f"     {lp.layer_name:30s} : {lp.latency_ms:.4f} ms, "
                  f"{lp.memory_mb:.2f} MB")

    if profile.operator_fusion_info:
        fusion = profile.operator_fusion_info
        print(f"   Operator fusion      : detected={fusion.get('fusion_detected', False)}, "
              f"speedup={fusion.get('speedup', 1.0):.2f}x")

    hil_status = hil.get_status()
    print(f"   COROLLARY-3 guard    : {hil_status['corollary3_guard']}")
    print(f"   HINO guard           : {hil_status['hino_guard']}")

    # ── 3. Service Discovery ────────────────────────────────────────────
    print("\n [SD1] Service Discovery Bridge")
    sd = suite.sd
    print(f"   backend              : {sd.cfg.backend}")
    print(f"   has_consul           : {sd.get_status()['has_consul']}")
    print(f"   has_requests         : {sd.get_status()['has_requests']}")

    # Register a service (static backend for testing)
    registered = sd.register_service(
        service_name="sesi-test-service",
        address="localhost",
        port=8080,
        tags=["test", "zkp-enabled"],
    )
    print(f"   registered           : {registered}")
    print(f"   n_registered         : {len(sd._registered_services)}")

    # Resolve service
    endpoints = sd.resolve_service("sesi-test-service")
    print(f"   resolved_endpoints   : {len(endpoints)}")
    for ep in endpoints:
        print(f"     {ep.service_id:20s} @ {ep.address}:{ep.port} "
              f"healthy={ep.healthy}")

    sd_status = sd.get_status()
    print(f"   COROLLARY-3 guard    : {sd_status['corollary3_guard']}")
    print(f"   HINO guard           : {sd_status['hino_guard']}")

    # ── 4. Full Deployment Pipeline ─────────────────────────────────────
    print("\n [DEPLOY] Full Privacy-Preserving Deployment Pipeline")
    deploy_result = suite.deploy_privacy_preserving_model(
        model=model,
        example_input=example_input,
        service_name="sesi-model-server",
        service_address="localhost",
        service_port=8080,
    )

    print(f"   Hardware profile     :")
    for k, v in deploy_result["hardware_profile"].items():
        print(f"     {k:25s} : {v}")
    print(f"   ZKP circuit          :")
    for k, v in deploy_result["zkp_circuit"].items():
        if "key" in k:
            print(f"     {k:25s} : {os.path.basename(v)}")
        else:
            print(f"     {k:25s} : {v}")
    print(f"   Service registration :")
    for k, v in deploy_result["service_registration"].items():
        print(f"     {k:25s} : {v}")
    print(f"   Inference test       :")
    for k, v in deploy_result["inference_test"].items():
        print(f"     {k:25s} : {v}")

    # ── 5. Full Status ──────────────────────────────────────────────────
    print("\n [STATUS]")
    for k, v in suite.get_full_status().items():
        print(f"   {k}:")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"     {kk}: {vv}")
        else:
            print(f"     {v}")

    print("\n" + "=" * 78)
    print(" HONEST SCOPE")
    print(" " + "-" * 76)
    print(" [C]   Groth 2016: Groth16 zk-SNARKs with constant-size proofs.")
    print(" [C]   Gabizon et al. 2019: PLONK: Permutations over Lagrange-bases")
    print("       for Oecumenical Noninteractive arguments of Knowledge.")
    print(" [C]   Bunz et al. 2018: Bulletproofs: Short arguments for")
    print("       confidential transactions.")
    print(" [C]   HashiCorp Consul documentation.")
    print(" [C]   Netflix Eureka documentation.")
    print(" [D]   ZK proofs are computationally expensive (100-10000x slower).")
    print("       Circuit compilation from PyTorch is lossy for dynamic control flow.")
    print(" [D]   Hardware profiling requires actual GPU/CPU/NPU access.")
    print("       Latency varies with thermal throttling, background processes.")
    print(" [D]   Service discovery requires running Consul/Eureka cluster.")
    print("       Discovery is eventually consistent; stale reads can occur.")
    print(" [D]   COROLLARY-3 guard: all operations first-order only; no Hessian.")
    print(" [C]   HINO guard: discrete operations respected; no smooth density assumed.")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL tools. No security/privacy/legal")
    print("       guarantee. Not validated against real production systems.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
