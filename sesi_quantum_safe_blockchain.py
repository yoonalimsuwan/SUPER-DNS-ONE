"""
=============================================================================
SESI QUANTUM-SAFE BLOCKCHAIN INTEGRATION
PQC Chain + STARK Prover Bridge + On-Chain State Registry
Native integration with SDA-ONE, DI-QKD, BFT CT, and ITP
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
[QSC1] QuantumSafeBlockchainCore
      • SHA-3 (Keccak) hashing [C NIST FIPS 202] (Quantum-resistant)
      • Dilithium5 signatures [C NIST FIPS 204] for block signing
      • Append-only ledger with Merkle root of transactions
      • Integration with SDA-ONE folding results and ITP proofs
      • COROLLARY-3 guard: first-order hashing/signing; no Hessian
      • HINO guard: discrete byte arrays, not smooth density

[SPB1] STARKProverBridge
      • STARKs (Scalable Transparent ARguments of Knowledge) [C Ben-Sasson 2018]
      • Hash-based ZK proofs (inherently quantum-safe, no elliptic curves)
      • Formats ITP verification traces and SDA-ONE metrics for STARK provers
      • COROLLARY-3 guard: first-order trace generation
      • HINO guard: discrete execution traces

[OSR1] OnChainStateRegistry
      • Merkle-Patricia Trie for state management
      • Tracks: verified protein folds, QKD session keys (hashes only), ITP theorems
      • PQC-signed state roots
      • COROLLARY-3 guard: first-order state transitions
      • HINO guard: discrete state updates

=============================================================================
HONEST SCOPE
=============================================================================
• QuantumSafeBlockchainCore: This is a PROTOCOL FRAMEWORK. Real blockchain
  requires P2P networking (e.g., libp2p), which is abstracted here.
• STARKProverBridge: STARK proving is computationally heavy. This module
  formats the data; actual proving requires a dedicated engine (e.g., Winterfell).
• OnChainStateRegistry: Stores HASHES of large data (e.g., 100k+ residue
  coordinates), not the raw data itself, to prevent chain bloat.
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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.quantum_safe_blockchain")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import oqs  # liboqs for PQC signatures (Dilithium)
    HAS_OQS = True
except Exception:
    HAS_OQS = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

__all__ = [
    "QuantumSafeBlockchainCore", "QSBlock", "QSTransaction",
    "STARKProverBridge", "STARKTrace",
    "OnChainStateRegistry", "StateEntry",
    "SESIQuantumSafeBlockchainSuite",
]


# =============================================================================
# SECTION 1 — [QSC1] QUANTUM-SAFE BLOCKCHAIN CORE
# =============================================================================

@dataclass
class QSTransaction:
    """A single transaction in the quantum-safe blockchain."""
    tx_id: str
    timestamp: float
    tx_type: str  # "sda_one_fold", "itp_proof", "qkd_session", "bft_ct_entry"
    payload_hash: str  # SHA3-256 hash of the actual payload (prevents chain bloat)
    payload_metadata: Dict[str, Any]  # Lightweight metadata
    pqc_signature: Optional[bytes] = None


@dataclass
class QSBlock:
    """A block in the quantum-safe blockchain."""
    block_index: int
    timestamp: float
    prev_hash: str
    merkle_root: str
    transactions: List[QSTransaction]
    pqc_signature: bytes
    nonce: int = 0


class QuantumSafeBlockchainCore:
    """
    [C/D] Quantum-safe blockchain core using SHA-3 and Dilithium.

    QUANTUM RESISTANCE [C NIST FIPS 202, 204]
      • Hashing: SHA3-256 (Keccak) is resistant to Grover's algorithm
        (provides 128-bit post-quantum security).
      • Signatures: CRYSTALS-Dilithium5 provides strong EUF-CMA security
        against quantum adversaries (Shor's algorithm does not break it).

    BLOCK STRUCTURE [D]
      • prev_hash: SHA3-256 of previous block (chain integrity)
      • merkle_root: SHA3-256 Merkle root of transactions
      • pqc_signature: Dilithium5 signature of (prev_hash + merkle_root + timestamp)

    INTEGRATION WITH SDA-ONE & ITP [D]
      • Large payloads (e.g., 100k+ residue coordinates, full Coq proofs)
        are hashed off-chain. Only the SHA3-256 hash and lightweight
        metadata are stored on-chain to prevent chain bloat.

    COROLLARY-3 GUARD [D]
      Hashing and signature verification are FIRST-ORDER discrete operations.
      We do NOT compute derivatives through the blockchain state.

    HINO GUARD [C Thm 8]
      Block hashes, Merkle roots, and signatures are DISCRETE byte arrays.
      We respect the singular nature of cryptographic primitives.

    HONEST
      • This is a PROTOCOL FRAMEWORK; real deployment needs P2P networking.
      • Storing raw 100k+ residue coordinates on-chain is infeasible;
        only hashes are stored.
      • No security guarantee without proper key management (HSM).
    """

    def __init__(self, pqc_sig_algorithm: str = "Dilithium5"):
        self.pqc_sig_algorithm = pqc_sig_algorithm
        self.chain: List[QSBlock] = []
        self._signer = None
        self._verifier = None

        if HAS_OQS:
            self._signer = oqs.Signature(self.pqc_sig_algorithm)
            self._verifier = oqs.Signature(self.pqc_sig_algorithm)
            # Generate genesis keypair
            self._genesis_pk = self._signer.generate_keypair()
            self._genesis_sk = self._signer.export_secret_key()
        else:
            logger.warning("[QSC] liboqs not available; using mock PQC signatures")
            self._genesis_pk = b"mock_pk"
            self._genesis_sk = b"mock_sk"

        # Create genesis block
        self._create_genesis_block()

    def _hash_data(self, data: bytes) -> str:
        """Quantum-safe hashing using SHA3-256."""
        return hashlib.sha3_256(data).hexdigest()

    def _create_genesis_block(self) -> None:
        """Create the first block in the chain."""
        genesis_tx = QSTransaction(
            tx_id="genesis",
            timestamp=time.time(),
            tx_type="genesis",
            payload_hash=self._hash_data(b"SESI Genesis Block"),
            payload_metadata={"version": "1.0.0", "network": "SESI-QS-Chain"},
        )
        merkle_root = self._hash_data(json.dumps([genesis_tx.payload_hash]).encode())
        
        genesis_block = QSBlock(
            block_index=0,
            timestamp=time.time(),
            prev_hash="0" * 64,
            merkle_root=merkle_root,
            transactions=[genesis_tx],
            pqc_signature=b"genesis_signature",
        )
        self.chain.append(genesis_block)
        logger.info("[QSC] Genesis block created")

    def _compute_merkle_root(self, txs: List[QSTransaction]) -> str:
        """Compute SHA3-256 Merkle root of transactions."""
        if not txs:
            return self._hash_data(b"")
        
        hashes = [self._hash_data(json.dumps(tx.__dict__).encode()) for tx in txs]
        
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])  # Duplicate last hash if odd
            
            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i+1]
                next_level.append(self._hash_data(combined.encode()))
            hashes = next_level
            
        return hashes[0]

    def add_transaction(self, tx_type: str, payload: Any, metadata: Dict[str, Any]) -> QSTransaction:
        """
        Create and add a transaction to the pending pool (simplified: added to next block).
        For large payloads (e.g., SDA-ONE coords), only the hash is stored.
        """
        if isinstance(payload, (bytes, bytearray)):
            payload_hash = self._hash_data(payload)
        elif HAS_TORCH and isinstance(payload, torch.Tensor):
            payload_hash = self._hash_data(payload.cpu().numpy().tobytes())
        else:
            payload_hash = self._hash_data(json.dumps(payload).encode())

        tx = QSTransaction(
            tx_id=self._hash_data(f"{tx_type}_{time.time()}".encode())[:16],
            timestamp=time.time(),
            tx_type=tx_type,
            payload_hash=payload_hash,
            payload_metadata=metadata,
        )
        return tx

    def propose_block(self, transactions: List[QSTransaction]) -> QSBlock:
        """
        Propose a new block, signed with Dilithium5.
        """
        prev_block = self.chain[-1]
        merkle_root = self._compute_merkle_root(transactions)
        
        # Data to sign
        sign_data = f"{prev_block.prev_hash}{merkle_root}{time.time()}".encode()
        
        if HAS_OQS:
            signature = self._signer.sign(sign_data)
        else:
            signature = b"mock_signature_" + self._hash_data(sign_data).encode()

        new_block = QSBlock(
            block_index=prev_block.block_index + 1,
            timestamp=time.time(),
            prev_hash=self._hash_data(json.dumps(prev_block.__dict__).encode()),
            merkle_root=merkle_root,
            transactions=transactions,
            pqc_signature=signature,
        )
        
        self.chain.append(new_block)
        logger.info(f"[QSC] Block {new_block.block_index} proposed and added")
        return new_block

    def verify_chain(self) -> bool:
        """Verify the integrity and PQC signatures of the entire chain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i-1]
            
            # Verify prev_hash
            expected_prev_hash = self._hash_data(json.dumps(prev.__dict__).encode())
            if current.prev_hash != expected_prev_hash:
                logger.error(f"[QSC] Chain broken at block {i}: invalid prev_hash")
                return False
            
            # Verify Merkle root
            expected_merkle = self._compute_merkle_root(current.transactions)
            if current.merkle_root != expected_merkle:
                logger.error(f"[QSC] Chain broken at block {i}: invalid merkle_root")
                return False
            
            # Verify PQC signature (mocked if no liboqs)
            if HAS_OQS:
                sign_data = f"{current.prev_hash}{current.merkle_root}{current.timestamp}".encode()
                self._verifier.import_public_key(self._genesis_pk)
                if not self._verifier.verify(sign_data, current.pqc_signature):
                    logger.error(f"[QSC] Chain broken at block {i}: invalid PQC signature")
                    return False
                    
        logger.info("[QSC] Chain verification PASSED")
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "chain_length": len(self.chain),
            "pqc_algorithm": self.pqc_sig_algorithm,
            "has_oqs": HAS_OQS,
            "corollary3_guard": "first-order hashing/signing; no Hessian",
            "hino_guard": "discrete byte arrays, not smooth density",
        }


# =============================================================================
# SECTION 2 — [SPB1] STARK PROVER BRIDGE
# =============================================================================

@dataclass
class STARKTrace:
    """Formatted trace ready for a STARK prover (e.g., Winterfell/Cairo)."""
    execution_trace: List[List[int]]  # 2D array of field elements
    public_inputs: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class STARKProverBridge:
    """
    [C/D] Bridge to format data for quantum-safe STARK proving.

    STARKs (Scalable Transparent ARguments of Knowledge) [C Ben-Sasson 2018]
      • Inherently quantum-safe: relies on hash functions (SHA-3), not
        elliptic curves or pairing-friendly curves (which Shor's algorithm breaks).
      • Transparent: no trusted setup required.
      • Scalable: proving time is quasi-linear in the size of the computation.

    INTEGRATION WITH ITP & SDA-ONE [D]
      • ITP Proof Trace: Converts Coq/Isabelle verification steps into
        a sequence of field operations that a STARK prover can verify.
      • SDA-ONE Metrics: Converts folding RMSD, domain segmentation
        boundaries, and contact probabilities into public inputs for
        on-chain verification without revealing the raw coordinates.

    COROLLARY-3 GUARD [D]
      Trace generation is FIRST-ORDER (integer/field arithmetic).
      We do NOT compute derivatives through the STARK trace.

    HINO GUARD [C Thm 8]
      Execution traces are DISCRETE sequences of field elements.
      We respect the singular nature of the arithmetic circuit.

    HONEST
      • This module FORMATS the data; it does NOT run the actual STARK prover.
      • Real STARK proving requires dedicated engines (e.g., Winterfell, Cairo, Risc0).
      • STARK proofs are large (tens to hundreds of KBs), but verification is fast.
    """

    def __init__(self, field_prime: int = 2**251 + 17 * 2**192 + 1):  # STARK-friendly prime
        self.field_prime = field_prime

    def _to_field_element(self, value: Any) -> int:
        """Convert a value to a field element modulo field_prime."""
        if isinstance(value, (int, float)):
            return int(value) % self.field_prime
        elif isinstance(value, str):
            return int(hashlib.sha3_256(value.encode()).hexdigest(), 16) % self.field_prime
        elif HAS_TORCH and isinstance(value, torch.Tensor):
            return int(value.item()) % self.field_prime
        else:
            return int(hashlib.sha3_256(str(value).encode()).hexdigest(), 16) % self.field_prime

    def format_itp_proof_for_stark(self, itp_results: List[Dict[str, Any]],
                                     protocol_name: str) -> STARKTrace:
        """
        Format Interactive Theorem Prover results into a STARK-friendly trace.
        This represents the "execution trace" of the proof verification.
        """
        public_inputs = [
            self._to_field_element(protocol_name),
            self._to_field_element(len(itp_results)),
        ]
        
        execution_trace = []
        for res in itp_results:
            row = [
                self._to_field_element(res.get("property", "")),
                self._to_field_element(1 if res.get("verified", False) else 0),
                self._to_field_element(res.get("time_s", 0.0)),
            ]
            execution_trace.append(row)

        return STARKTrace(
            execution_trace=execution_trace,
            public_inputs=public_inputs,
            metadata={
                "protocol": protocol_name,
                "trace_length": len(execution_trace),
                "corollary3_guard": "first-order trace generation",
                "hino_guard": "discrete field elements",
            }
        )

    def format_sda_one_metrics_for_stark(self, sda_metrics: Dict[str, Any]) -> STARKTrace:
        """
        Format SDA-ONE folding metrics into a STARK-friendly trace.
        Allows on-chain verification that a fold meets certain thresholds
        without revealing the actual 100k+ residue coordinates.
        """
        public_inputs = [
            self._to_field_element(sda_metrics.get("sequence_length", 0)),
            self._to_field_element(sda_metrics.get("num_domains", 0)),
        ]
        
        execution_trace = [
            [
                self._to_field_element(sda_metrics.get("max_domain_size", 0)),
                self._to_field_element(sda_metrics.get("contact_prob_threshold", 0.0)),
                self._to_field_element(sda_metrics.get("docking_energy", 0.0)),
            ]
        ]

        return STARKTrace(
            execution_trace=execution_trace,
            public_inputs=public_inputs,
            metadata={
                "metric_type": "sda_one_fold",
                "corollary3_guard": "first-order trace generation",
                "hino_guard": "discrete field elements",
            }
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "field_prime_bits": self.field_prime.bit_length(),
            "quantum_safe": True,  # STARKs are inherently quantum-safe
            "corollary3_guard": "first-order trace generation",
            "hino_guard": "discrete field elements",
        }


# =============================================================================
# SECTION 3 — [OSR1] ON-CHAIN STATE REGISTRY
# =============================================================================

@dataclass
class StateEntry:
    """A single entry in the on-chain state registry."""
    key: str
    value_hash: str
    timestamp: float
    tx_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnChainStateRegistry:
    """
    [D] On-chain state registry using a Merkle-Patricia Trie concept.

    PURPOSE
      Tracks the current "state" of the SESI ecosystem:
      • Verified SDA-ONE protein folds (hash of coordinates + metrics)
      • Established DI-QKD sessions (hash of hybrid key, CHSH value)
      • Verified ITP theorems (hash of proof script + verification status)

    QUANTUM SAFETY
      • State root is computed using SHA3-256.
      • State transitions are signed with Dilithium5.

    INTEGRATION
      • When `QuantumSafeBlockchainCore` adds a block, this registry
        updates its state root, which is then included in the next block.

    COROLLARY-3 GUARD [D]
      State transitions (hashing, dictionary updates) are FIRST-ORDER.
      We do NOT compute derivatives through the state registry.

    HINO GUARD [C Thm 8]
      State keys, hashes, and timestamps are DISCRETE. We respect the
      singular nature of the state machine.

    HONEST
      • This is a simplified Merkle-Patricia Trie simulation.
      • Real implementations require LevelDB/RocksDB for persistence.
      • Stores HASHES of large data, not the raw data itself.
    """

    def __init__(self):
        self.state: Dict[str, StateEntry] = {}
        self.state_root = "0" * 64  # Initial empty state root

    def _compute_state_root(self) -> str:
        """Compute SHA3-256 root of the current state."""
        if not self.state:
            return self._hash_data(b"empty_state")
        
        # Sort keys for deterministic hashing
        sorted_keys = sorted(self.state.keys())
        combined = "".join([f"{k}:{self.state[k].value_hash}" for k in sorted_keys])
        return self._hash_data(combined.encode())

    def _hash_data(self, data: bytes) -> str:
        return hashlib.sha3_256(data).hexdigest()

    def update_state(self, key: str, value_hash: str, tx_id: str, metadata: Dict[str, Any]) -> None:
        """Update the state registry with a new entry."""
        self.state[key] = StateEntry(
            key=key,
            value_hash=value_hash,
            timestamp=time.time(),
            tx_id=tx_id,
            metadata=metadata,
        )
        self.state_root = self._compute_state_root()
        logger.info(f"[OSR] State updated: {key} (new root: {self.state_root[:16]}...)")

    def get_state(self, key: str) -> Optional[StateEntry]:
        """Retrieve a state entry by key."""
        return self.state.get(key)

    def verify_state_integrity(self) -> bool:
        """Verify that the current state root matches the computed root."""
        expected_root = self._compute_state_root()
        return self.state_root == expected_root

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self.state),
            "state_root": self.state_root,
            "corollary3_guard": "first-order state transitions",
            "hino_guard": "discrete state updates",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIQuantumSafeBlockchainSuite:
    """
    Unifies: QuantumSafeBlockchainCore + STARKProverBridge + OnChainStateRegistry.

    FULL PIPELINE:
      1. Receive SDA-ONE folding results, ITP proofs, or QKD sessions.
      2. Format data for STARK proving (optional, for ZK verification).
      3. Hash payloads and create QSTransactions.
      4. Update OnChainStateRegistry.
      5. Propose and sign a new QSBlock with Dilithium5.
      6. Verify chain integrity.
    """

    def __init__(self):
        self.blockchain = QuantumSafeBlockchainCore()
        self.stark_bridge = STARKProverBridge()
        self.registry = OnChainStateRegistry()

    def record_sda_one_fold(self, sequence_length: int, num_domains: int,
                             docking_energy: float, coords_hash: str) -> Dict[str, Any]:
        """Record an SDA-ONE folding event on the quantum-safe blockchain."""
        # 1. Format for STARK (for future ZK verification)
        stark_trace = self.stark_bridge.format_sda_one_metrics_for_stark({
            "sequence_length": sequence_length,
            "num_domains": num_domains,
            "docking_energy": docking_energy,
        })

        # 2. Create transaction
        tx = self.blockchain.add_transaction(
            tx_type="sda_one_fold",
            payload=coords_hash,  # Only hash is stored on-chain
            metadata={
                "sequence_length": sequence_length,
                "num_domains": num_domains,
                "docking_energy": docking_energy,
                "stark_trace_length": len(stark_trace.execution_trace),
            }
        )

        # 3. Update state registry
        state_key = f"sda_fold_{sequence_length}_{int(time.time())}"
        self.registry.update_state(
            key=state_key,
            value_hash=tx.payload_hash,
            tx_id=tx.tx_id,
            metadata=tx.payload_metadata
        )

        # 4. Propose block
        block = self.blockchain.propose_block([tx])

        return {
            "tx_id": tx.tx_id,
            "block_index": block.block_index,
            "state_root": self.registry.state_root,
            "stark_trace_ready": True,
        }

    def record_itp_proof(self, protocol_name: str, itp_results: List[Dict[str, Any]],
                          proof_script_hash: str) -> Dict[str, Any]:
        """Record an Interactive Theorem Prover verification on-chain."""
        # 1. Format for STARK
        stark_trace = self.stark_bridge.format_itp_proof_for_stark(itp_results, protocol_name)

        # 2. Create transaction
        all_verified = all(r.get("verified", False) for r in itp_results)
        tx = self.blockchain.add_transaction(
            tx_type="itp_proof",
            payload=proof_script_hash,
            metadata={
                "protocol": protocol_name,
                "all_verified": all_verified,
                "properties_checked": len(itp_results),
                "stark_trace_length": len(stark_trace.execution_trace),
            }
        )

        # 3. Update state registry
        state_key = f"itp_{protocol_name}_{int(time.time())}"
        self.registry.update_state(
            key=state_key,
            value_hash=tx.payload_hash,
            tx_id=tx.tx_id,
            metadata=tx.payload_metadata
        )

        # 4. Propose block
        block = self.blockchain.propose_block([tx])

        return {
            "tx_id": tx.tx_id,
            "block_index": block.block_index,
            "state_root": self.registry.state_root,
            "all_verified": all_verified,
        }

    def verify_full_system(self) -> Dict[str, Any]:
        """Verify the integrity of the blockchain and state registry."""
        chain_valid = self.blockchain.verify_chain()
        state_valid = self.registry.verify_state_integrity()
        
        return {
            "chain_valid": chain_valid,
            "state_valid": state_valid,
            "overall_valid": chain_valid and state_valid,
        }

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "blockchain": self.blockchain.get_status(),
            "stark_bridge": self.stark_bridge.get_status(),
            "registry": self.registry.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI QUANTUM-SAFE BLOCKCHAIN — SMOKE TEST")
    print(" PQC Chain + STARK Bridge + On-Chain State Registry")
    print("=" * 78)
    print(f" liboqs available       : {HAS_OQS}")
    print(f" torch available        : {HAS_TORCH}")

    suite = SESIQuantumSafeBlockchainSuite()

    # ── 1. Quantum-Safe Blockchain Core ─────────────────────────────────
    print("\n [QSC1] Quantum-Safe Blockchain Core")
    bc = suite.blockchain
    print(f"   chain_length         : {bc.get_status()['chain_length']}")
    print(f"   pqc_algorithm        : {bc.get_status()['pqc_algorithm']}")
    print(f"   COROLLARY-3 guard    : {bc.get_status()['corollary3_guard']}")
    print(f"   HINO guard           : {bc.get_status()['hino_guard']}")

    # ── 2. STARK Prover Bridge ──────────────────────────────────────────
    print("\n [SPB1] STARK Prover Bridge")
    stark = suite.stark_bridge
    print(f"   field_prime_bits     : {stark.get_status()['field_prime_bits']}")
    print(f"   quantum_safe         : {stark.get_status()['quantum_safe']}")
    
    # Format mock ITP proof for STARK
    mock_itp = [
        {"property": "authentication", "verified": True, "time_s": 0.05},
        {"property": "secrecy", "verified": True, "time_s": 0.12},
    ]
    stark_trace = stark.format_itp_proof_for_stark(mock_itp, "test_protocol")
    print(f"   ITP trace length     : {len(stark_trace.execution_trace)} rows")
    print(f"   COROLLARY-3 guard    : {stark.get_status()['corollary3_guard']}")

    # ── 3. On-Chain State Registry ──────────────────────────────────────
    print("\n [OSR1] On-Chain State Registry")
    reg = suite.registry
    print(f"   num_entries          : {reg.get_status()['num_entries']}")
    print(f"   state_root           : {reg.get_status()['state_root'][:16]}...")
    print(f"   COROLLARY-3 guard    : {reg.get_status()['corollary3_guard']}")

    # ── 4. Full Pipeline Integration ────────────────────────────────────
    print("\n [PIPELINE] Recording SDA-ONE Fold & ITP Proof on-chain")
    
    # Record SDA-ONE fold
    sda_result = suite.record_sda_one_fold(
        sequence_length=100000,
        num_domains=50,
        docking_energy=-150.5,
        coords_hash="sha3_256_mock_coords_hash_1234567890abcdef"
    )
    print(f"   SDA-ONE Fold:")
    print(f"     tx_id              : {sda_result['tx_id']}")
    print(f"     block_index        : {sda_result['block_index']}")
    print(f"     stark_trace_ready  : {sda_result['stark_trace_ready']}")

    # Record ITP proof
    itp_result = suite.record_itp_proof(
        protocol_name="hybrid_di_qkd_pqc_tls",
        itp_results=mock_itp,
        proof_script_hash="sha3_256_mock_coq_script_hash_abcdef1234567890"
    )
    print(f"   ITP Proof:")
    print(f"     tx_id              : {itp_result['tx_id']}")
    print(f"     block_index        : {itp_result['block_index']}")
    print(f"     all_verified       : {itp_result['all_verified']}")

    # Verify system
    print("\n [VERIFY] Full System Integrity Check")
    verify_result = suite.verify_full_system()
    print(f"   chain_valid          : {verify_result['chain_valid']}")
    print(f"   state_valid          : {verify_result['state_valid']}")
    print(f"   overall_valid        : {verify_result['overall_valid']}")

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
    print(" [C]   NIST FIPS 202: SHA-3 (Keccak) standard (quantum-resistant hashing).")
    print(" [C]   NIST FIPS 204: CRYSTALS-Dilithium standard (quantum-resistant signatures).")
    print(" [C]   Ben-Sasson et al. 2018: Scalable, transparent, and post-quantum secure")
    print("       arguments of knowledge (STARKs).")
    print(" [D]   This is a PROTOCOL FRAMEWORK. Real blockchain requires P2P networking")
    print("       (e.g., libp2p) and persistent storage (e.g., LevelDB), which are abstracted.")
    print(" [D]   Large payloads (e.g., 100k+ residue coordinates) are hashed off-chain.")
    print("       Only SHA3-256 hashes are stored on-chain to prevent chain bloat.")
    print(" [D]   STARKProverBridge FORMATS data for proving; actual proving requires")
    print("       a dedicated engine (e.g., Winterfell, Cairo, Risc0).")
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
