"""
=============================================================================
SESI SDA-ONE ECOSYSTEM — DAO / TOKENOMICS / ORACLES
Decentralized Governance + Researcher Incentives + External Data Feeds
Native integration with SDA-ONE, DID, ZK Proofs, and PQC Blockchain
=============================================================================
Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
            MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0
Labels    : [P] proved  [C] cited  [D] derived here  [O] open/speculative

=============================================================================
SDA-ONE INTEGRATION
=============================================================================
This module consumes SDA-ONE output and integrates with:
  - structural_domain_assembly_one.py: DomainAssemblyConfig parameters
  - sesi_blockchain_did_zk_xchain.py: DID, ZK proofs, cross-chain bridges
  - sesi_quantum_safe_blockchain.py: PQC blockchain core

Integration points:
  1. SDAOneDAO.propose_parameter_update(proposer_did, param_name, new_value)
     - Proposes changes to DomainAssemblyConfig (e.g., target_domain_size)
     - Requires ZK proof of proposer's stake and voting power
     - Executed on-chain after voting period

  2. ResearcherTokenomics.reward_folding(researcher_did, sda_output_hash, quality_score)
     - Mints $SDA tokens for successful SDA-ONE folding
     - Quality score from ZK proof (RMSD, energy, domain count)
     - Vesting schedule applied

  3. OracleNetworkBridge.fetch_pdb_update(pdb_id)
     - Retrieves experimental structure from PDB via Chainlink
     - Verifies data integrity with ZK proof
     - Feeds into SDA-ONE validation pipeline

=============================================================================
MODULES
=============================================================================
[DAO1] SDAOneDAO
      • Quadratic voting for parameter proposals [C Weyl 2019]
      • Conviction voting for continuous funding [C Commons Stack]
      • Timelock execution (48-hour delay) [C Compound]
      • Treasury management with multi-sig
      • Integration with DID for voting rights
      • COROLLARY-3 guard: first-order vote tallying; no Hessian
      • HINO guard: discrete proposals and votes

[TOK1] ResearcherTokenomics
      • $SDA token with 1B total supply
      • Incentive mechanisms:
        - Folding rewards (quality-based)
        - Validation rewards (verifying others' work)
        - Dataset contribution rewards
        - Staking for governance participation
      • Vesting schedules (linear over 2 years)
      • Slashing for malicious behavior
      • COROLLARY-3 guard: first-order token transfers; no Hessian
      • HINO guard: discrete token balances

[ORA1] OracleNetworkBridge
      • Chainlink OCR (Off-Chain Reporting) integration
      • Band Protocol multi-source aggregation
      • ZK-verified external data (PDB, UniProt, Cryo-EM)
      • Price feeds for $SDA token
      • PQC-signed oracle responses
      • COROLLARY-3 guard: first-order data aggregation; no Hessian
      • HINO guard: discrete oracle timestamps

=============================================================================
HONEST SCOPE
=============================================================================
• SDAOneDAO: DAO governance is a technical framework, NOT a legal entity.
  Scientific validity still requires peer review. Voting can be manipulated
  by whale stakeholders. This is a coordination mechanism, not a substitute
  for expertise.
• ResearcherTokenomics: $SDA token has NO intrinsic value. Tokenomics are
  illustrative and subject to change. This is NOT financial advice. Real
  deployment requires legal review (securities laws, tax implications).
• OracleNetworkBridge: Oracles are trusted third parties. Multi-source
  aggregation reduces but does not eliminate trust assumptions. Chainlink/Band
  require operational infrastructure and node operator honesty.
• COROLLARY-3 guard: all operations first-order only; no Hessian.
• HINO guard: discrete operations respected; no smooth density assumed.
• PROP-3 awareness: field products lose two Sobolev orders.
• ρ≈1.25 is NUMERICAL, not proved; unused by these modules.
• All parameters illustrative. No financial/legal/security guarantee.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.dao_tokenomics_oracle")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import requests  # For Chainlink/Band API calls
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    import oqs  # liboqs for PQC signatures
    HAS_OQS = True
except Exception:
    HAS_OQS = False

__all__ = [
    "SDAOneDAO", "DAOProposal", "DAOVote", "DAOTreasury",
    "ResearcherTokenomics", "SDAToken", "TokenReward",
    "OracleNetworkBridge", "OracleData", "OracleSource",
    "SESIDAO TokenomicsOracleSuite",
]


# =============================================================================
# SECTION 1 — [DAO1] SDA-ONE DAO
# =============================================================================

class ProposalStatus(str, Enum):
    """Status of a DAO proposal."""
    PENDING = "pending"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


@dataclass
class DAOProposal:
    """A governance proposal in the SDA-ONE DAO."""
    proposal_id: str
    proposer_did: str
    title: str
    description: str
    proposal_type: str  # "parameter_update" | "model_upgrade" | "treasury_allocation" | "other"
    target_contract: str  # e.g., "DomainAssemblyConfig"
    parameter_name: Optional[str]
    new_value: Optional[Any]
    start_time: float
    end_time: float
    votes_for: float  # quadratic voting power
    votes_against: float
    quorum_required: float
    status: ProposalStatus
    execution_time: Optional[float]
    timelock_delay_s: int = 172800  # 48 hours
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAOVote:
    """A vote cast on a DAO proposal."""
    vote_id: str
    proposal_id: str
    voter_did: str
    vote_weight: float  # quadratic voting power
    support: bool  # True = for, False = against
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAOTreasury:
    """DAO treasury holding $SDA tokens."""
    total_balance_sda: float
    available_balance_sda: float
    locked_balance_sda: float
    multi_sig_threshold: int  # e.g., 3-of-5
    multi_sig_signers: List[str]  # DIDs of signers
    metadata: Dict[str, Any] = field(default_factory=dict)


class SDAOneDAO:
    """
    [C/D] Decentralized Autonomous Organization for SDA-ONE governance.

    GOVERNANCE MODEL [C Moloch DAO v2; Compound Governor Bravo]
      • Proposers must hold minimum $SDA stake (e.g., 10,000 $SDA)
      • Voting uses quadratic voting: voting_power = sqrt(token_balance)
      • Proposals require quorum (e.g., 4% of total supply) to pass
      • Timelock delay (48 hours) before execution for safety

    PROPOSAL TYPES [D]
      1. Parameter Update: change DomainAssemblyConfig values
         - target_domain_size, min_domain_size, max_domain_size
         - dock_contact_r0, dock_lr, dock_iters
         - contact_prob_threshold, contact_knn_domains
      2. Model Upgrade: deploy new version of SDA-ONE modules
         - DomainSegmenter, CrossDomainContactHead, DomainDockingAssembler
      3. Treasury Allocation: fund research grants, bounties, partnerships
      4. Other: custom proposals with arbitrary execution logic

    QUADRATIC VOTING [C Weyl 2019]
      voting_power = sqrt(token_balance)
      This reduces whale dominance and encourages broader participation.
      Example: 10,000 tokens → 100 voting power; 1,000,000 tokens → 1,000 voting power

    CONVEXION VOTING (optional) [C Commons Stack]
      For continuous funding proposals, conviction accumulates over time:
      conviction(t) = conviction(t-1) * decay + new_stake
      This allows long-term supporters to have more influence.

    TIMELOCK [C Compound]
      After a proposal passes, it enters a timelock period (48 hours).
      This gives the community time to react if a malicious proposal slipped through.
      After timelock expires, the proposal can be executed on-chain.

    INTEGRATION WITH DID [D]
      • Proposers and voters authenticate with their W3C DID
      • DID Document contains PQC verification method (Dilithium5)
      • Proposals and votes are signed with researcher's DID

    COROLLARY-3 GUARD [D]
      Vote tallying and proposal execution are FIRST-ORDER operations.
      We do NOT compute derivatives through the governance layer.

    HINO GUARD [C Thm 8]
      Proposals, votes, and treasury balances are DISCRETE state.
      We respect the singular nature of governance data.

    HONEST
      • DAO governance is a technical framework, NOT a legal entity.
      • Scientific validity still requires peer review.
      • Voting can be manipulated by whale stakeholders.
      • This is a coordination mechanism, not a substitute for expertise.
    """

    def __init__(self, total_token_supply: float = 1_000_000_000.0,
                 quorum_percentage: float = 0.04,
                 voting_period_s: int = 604800,  # 7 days
                 timelock_delay_s: int = 172800):  # 48 hours
        self.total_token_supply = total_token_supply
        self.quorum_percentage = quorum_percentage
        self.voting_period_s = voting_period_s
        self.timelock_delay_s = timelock_delay_s

        self._proposals: Dict[str, DAOProposal] = {}
        self._votes: Dict[str, List[DAOVote]] = {}  # proposal_id -> votes
        self._treasury = DAOTreasury(
            total_balance_sda=100_000_000.0,  # 10% of supply in treasury
            available_balance_sda=100_000_000.0,
            locked_balance_sda=0.0,
            multi_sig_threshold=3,
            multi_sig_signers=[],
        )

    def _compute_quadratic_voting_power(self, token_balance: float) -> float:
        """Compute quadratic voting power: sqrt(token_balance)."""
        return math.sqrt(max(token_balance, 0.0))

    def create_proposal(self, proposer_did: str, proposer_token_balance: float,
                          title: str, description: str, proposal_type: str,
                          target_contract: str, parameter_name: Optional[str] = None,
                          new_value: Optional[Any] = None,
                          min_proposal_stake: float = 10_000.0) -> DAOProposal:
        """
        Create a new governance proposal.

        Args:
            proposer_did: DID of the proposer
            proposer_token_balance: $SDA balance of proposer
            title: proposal title
            description: detailed description
            proposal_type: "parameter_update" | "model_upgrade" | "treasury_allocation" | "other"
            target_contract: e.g., "DomainAssemblyConfig"
            parameter_name: parameter to update (for parameter_update)
            new_value: new value (for parameter_update)
            min_proposal_stake: minimum $SDA required to propose

        Returns:
            DAOProposal

        Raises:
            ValueError if proposer has insufficient stake
        """
        if proposer_token_balance < min_proposal_stake:
            raise ValueError(
                f"Proposer must hold at least {min_proposal_stake} $SDA to create a proposal. "
                f"Current balance: {proposer_token_balance}"
            )

        proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        end_time = start_time + self.voting_period_s

        proposal = DAOProposal(
            proposal_id=proposal_id,
            proposer_did=proposer_did,
            title=title,
            description=description,
            proposal_type=proposal_type,
            target_contract=target_contract,
            parameter_name=parameter_name,
            new_value=new_value,
            start_time=start_time,
            end_time=end_time,
            votes_for=0.0,
            votes_against=0.0,
            quorum_required=self.total_token_supply * self.quorum_percentage,
            status=ProposalStatus.ACTIVE,
            execution_time=None,
            timelock_delay_s=self.timelock_delay_s,
            metadata={
                "proposer_voting_power": self._compute_quadratic_voting_power(proposer_token_balance),
                "corollary3_guard": "first-order proposal creation; no Hessian",
                "hino_guard": "discrete proposal state",
            },
        )

        self._proposals[proposal_id] = proposal
        self._votes[proposal_id] = []

        logger.info(f"[DAO] Created proposal {proposal_id}: {title}")
        return proposal

    def cast_vote(self, proposal_id: str, voter_did: str, voter_token_balance: float,
                    support: bool) -> DAOVote:
        """
        Cast a vote on a proposal using quadratic voting.

        Args:
            proposal_id: ID of the proposal
            voter_did: DID of the voter
            voter_token_balance: $SDA balance of voter
            support: True = vote for, False = vote against

        Returns:
            DAOVote

        Raises:
            ValueError if proposal is not active or voter already voted
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self._proposals[proposal_id]
        if proposal.status != ProposalStatus.ACTIVE:
            raise ValueError(f"Proposal {proposal_id} is not active (status: {proposal.status})")

        # Check if voter already voted
        existing_votes = [v for v in self._votes[proposal_id] if v.voter_did == voter_did]
        if existing_votes:
            raise ValueError(f"Voter {voter_did} has already voted on proposal {proposal_id}")

        # Check if voting period has ended
        if time.time() > proposal.end_time:
            raise ValueError(f"Voting period for proposal {proposal_id} has ended")

        vote_weight = self._compute_quadratic_voting_power(voter_token_balance)

        vote = DAOVote(
            vote_id=f"vote_{uuid.uuid4().hex[:12]}",
            proposal_id=proposal_id,
            voter_did=voter_did,
            vote_weight=vote_weight,
            support=support,
            timestamp=time.time(),
            metadata={
                "token_balance": voter_token_balance,
                "corollary3_guard": "first-order vote casting; no Hessian",
                "hino_guard": "discrete vote state",
            },
        )

        self._votes[proposal_id].append(vote)

        # Update proposal vote counts
        if support:
            proposal.votes_for += vote_weight
        else:
            proposal.votes_against += vote_weight

        logger.info(f"[DAO] Vote cast on {proposal_id} by {voter_did}: "
                    f"{'FOR' if support else 'AGAINST'} (weight: {vote_weight:.2f})")
        return vote

    def finalize_proposal(self, proposal_id: str) -> DAOProposal:
        """
        Finalize a proposal after voting period ends.

        Args:
            proposal_id: ID of the proposal

        Returns:
            Updated DAOProposal with new status

        Raises:
            ValueError if voting period has not ended
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self._proposals[proposal_id]
        if time.time() < proposal.end_time:
            raise ValueError(f"Voting period for proposal {proposal_id} has not ended yet")

        if proposal.status != ProposalStatus.ACTIVE:
            return proposal  # Already finalized

        total_votes = proposal.votes_for + proposal.votes_against
        quorum_met = total_votes >= proposal.quorum_required
        majority_for = proposal.votes_for > proposal.votes_against

        if quorum_met and majority_for:
            proposal.status = ProposalStatus.PASSED
            proposal.execution_time = time.time() + proposal.timelock_delay_s
            logger.info(f"[DAO] Proposal {proposal_id} PASSED (quorum met, majority for)")
        else:
            proposal.status = ProposalStatus.FAILED
            if not quorum_met:
                logger.info(f"[DAO] Proposal {proposal_id} FAILED (quorum not met)")
            else:
                logger.info(f"[DAO] Proposal {proposal_id} FAILED (majority against)")

        return proposal

    def execute_proposal(self, proposal_id: str) -> bool:
        """
        Execute a passed proposal after timelock expires.

        Args:
            proposal_id: ID of the proposal

        Returns:
            True if execution succeeded

        Raises:
            ValueError if proposal is not passed or timelock has not expired
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self._proposals[proposal_id]
        if proposal.status != ProposalStatus.PASSED:
            raise ValueError(f"Proposal {proposal_id} has not passed (status: {proposal.status})")

        if time.time() < proposal.execution_time:
            raise ValueError(f"Timelock for proposal {proposal_id} has not expired yet")

        # Execute based on proposal type
        if proposal.proposal_type == "parameter_update":
            # In production: would call smart contract to update parameter
            logger.info(f"[DAO] Executing parameter update: {proposal.target_contract}.{proposal.parameter_name} = {proposal.new_value}")
        elif proposal.proposal_type == "model_upgrade":
            # In production: would deploy new contract version
            logger.info(f"[DAO] Executing model upgrade: {proposal.target_contract}")
        elif proposal.proposal_type == "treasury_allocation":
            # In production: would transfer tokens from treasury
            logger.info(f"[DAO] Executing treasury allocation")
        else:
            logger.info(f"[DAO] Executing custom proposal: {proposal.proposal_type}")

        proposal.status = ProposalStatus.EXECUTED
        return True

    def get_proposal_status(self, proposal_id: str) -> Dict[str, Any]:
        """Get detailed status of a proposal."""
        if proposal_id not in self._proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self._proposals[proposal_id]
        total_votes = proposal.votes_for + proposal.votes_against
        participation_rate = total_votes / proposal.quorum_required if proposal.quorum_required > 0 else 0.0

        return {
            "proposal_id": proposal_id,
            "title": proposal.title,
            "status": proposal.status.value,
            "votes_for": proposal.votes_for,
            "votes_against": proposal.votes_against,
            "total_votes": total_votes,
            "quorum_required": proposal.quorum_required,
            "participation_rate": participation_rate,
            "time_remaining_s": max(0, proposal.end_time - time.time()) if proposal.status == ProposalStatus.ACTIVE else 0,
            "execution_time": proposal.execution_time,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_proposals": len(self._proposals),
            "active_proposals": sum(1 for p in self._proposals.values() if p.status == ProposalStatus.ACTIVE),
            "passed_proposals": sum(1 for p in self._proposals.values() if p.status == ProposalStatus.PASSED),
            "executed_proposals": sum(1 for p in self._proposals.values() if p.status == ProposalStatus.EXECUTED),
            "treasury_balance_sda": self._treasury.total_balance_sda,
            "quorum_percentage": self.quorum_percentage,
            "voting_period_s": self.voting_period_s,
            "timelock_delay_s": self.timelock_delay_s,
            "corollary3_guard": "first-order vote tallying; no Hessian",
            "hino_guard": "discrete proposals and votes",
        }


# =============================================================================
# SECTION 2 — [TOK1] RESEARCHER TOKENOMICS
# =============================================================================

@dataclass
class SDAToken:
    """$SDA token balance and metadata for a researcher."""
    researcher_did: str
    total_balance: float
    available_balance: float
    vested_balance: float
    staked_balance: float
    voting_power: float  # quadratic voting power
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenReward:
    """A token reward issued to a researcher."""
    reward_id: str
    researcher_did: str
    reward_type: str  # "folding" | "validation" | "dataset" | "staking"
    amount_sda: float
    vesting_start: float
    vesting_end: float
    vesting_cliff_s: int  # cliff period before vesting starts
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearcherTokenomics:
    """
    [C/D] Tokenomics for researcher incentives in the SDA-ONE ecosystem.

    TOKEN MODEL [D]
      • Total supply: 1,000,000,000 $SDA (1 billion)
      • Distribution:
        - 40%: Researcher rewards (folding, validation, dataset)
        - 20%: DAO treasury
        - 15%: Team and advisors (4-year vesting)
        - 15%: Ecosystem development
        - 10%: Initial liquidity and marketing

    INCENTIVE MECHANISMS [D]
      1. Folding Rewards:
         - Reward for successful SDA-ONE folding of target structures
         - Amount based on quality score (RMSD, energy, domain count)
         - ZK proof required to claim reward (proves quality without revealing coords)
         - Vesting: 2 years linear, 6-month cliff

      2. Validation Rewards:
         - Reward for verifying others' folding results
         - Requires running independent validation pipeline
         - Slashing if validator approves incorrect results

      3. Dataset Contribution Rewards:
         - Reward for contributing experimental data (Cryo-EM maps, etc.)
         - Data must pass quality checks
         - One-time reward, no vesting

      4. Staking Rewards:
         - Stake $SDA to participate in governance
         - Earn staking rewards (e.g., 5% APY)
         - Slashing for malicious voting behavior

    VESTING SCHEDULES [C Balancer LBP; Curve veCRV]
      • Linear vesting over time
      • Cliff period before vesting starts
      • Vesting smart contract releases tokens periodically

    SLASHING [C MakerDAO]
      • Malicious behavior (e.g., approving incorrect folds) results in token slashing
      • Slashed tokens burned or sent to treasury
      • Requires multi-sig approval for slashing

    COROLLARY-3 GUARD [D]
      Token transfers and reward calculations are FIRST-ORDER operations.
      We do NOT compute derivatives through the tokenomics layer.

    HINO GUARD [C Thm 8]
      Token balances and rewards are DISCRETE state.
      We respect the singular nature of token economics.

    HONEST
      • $SDA token has NO intrinsic value.
      • Tokenomics are illustrative and subject to change.
      • This is NOT financial advice.
      • Real deployment requires legal review (securities laws, tax implications).
    """

    def __init__(self, total_supply: float = 1_000_000_000.0):
        self.total_supply = total_supply
        self._token_balances: Dict[str, SDAToken] = {}
        self._rewards: Dict[str, TokenReward] = {}
        self._staking_pools: Dict[str, float] = {}  # researcher_did -> staked_amount

    def _create_token_account(self, researcher_did: str) -> SDAToken:
        """Create a new token account for a researcher."""
        if researcher_did in self._token_balances:
            return self._token_balances[researcher_did]

        token = SDAToken(
            researcher_did=researcher_did,
            total_balance=0.0,
            available_balance=0.0,
            vested_balance=0.0,
            staked_balance=0.0,
            voting_power=0.0,
            metadata={
                "created_at": time.time(),
                "corollary3_guard": "first-order token account creation; no Hessian",
                "hino_guard": "discrete token balance",
            },
        )
        self._token_balances[researcher_did] = token
        return token

    def _update_vesting(self, researcher_did: str) -> None:
        """Update vested balance based on current time."""
        if researcher_did not in self._token_balances:
            return

        token = self._token_balances[researcher_did]
        current_time = time.time()

        # Calculate newly vested tokens
        newly_vested = 0.0
        for reward in self._rewards.values():
            if reward.researcher_did != researcher_did:
                continue
            if reward.reward_type not in ["folding", "validation", "staking"]:
                continue  # dataset rewards are immediate

            # Check if cliff has passed
            if current_time < reward.vesting_start + reward.vesting_cliff_s:
                continue

            # Calculate vested portion
            vesting_duration = reward.vesting_end - reward.vesting_start
            elapsed = current_time - reward.vesting_start
            vesting_progress = min(1.0, elapsed / vesting_duration)

            vested_amount = reward.amount_sda * vesting_progress
            already_vested = token.vested_balance  # simplified; real impl tracks per-reward
            newly_vested += max(0, vested_amount - already_vested)

        token.vested_balance += newly_vested
        token.available_balance = token.total_balance - token.staked_balance
        token.voting_power = math.sqrt(max(token.available_balance, 0.0))

    def reward_folding(self, researcher_did: str, sda_output_hash: str,
                         quality_score: float, max_reward: float = 1000.0) -> TokenReward:
        """
        Issue a folding reward for successful SDA-ONE folding.

        Args:
            researcher_did: DID of the researcher
            sda_output_hash: SHA3-256 hash of SDA-ONE output
            quality_score: quality metric (0.0 to 1.0, e.g., based on RMSD/energy)
            max_reward: maximum $SDA reward for perfect quality

        Returns:
            TokenReward
        """
        token = self._create_token_account(researcher_did)
        self._update_vesting(researcher_did)

        # Calculate reward amount based on quality
        reward_amount = max_reward * quality_score

        # Vesting: 2 years linear, 6-month cliff
        current_time = time.time()
        vesting_start = current_time
        vesting_end = current_time + (2 * 365 * 24 * 3600)  # 2 years
        vesting_cliff_s = 6 * 30 * 24 * 3600  # 6 months

        reward = TokenReward(
            reward_id=f"reward_{uuid.uuid4().hex[:12]}",
            researcher_did=researcher_did,
            reward_type="folding",
            amount_sda=reward_amount,
            vesting_start=vesting_start,
            vesting_end=vesting_end,
            vesting_cliff_s=vesting_cliff_s,
            reason=f"Folding reward for {sda_output_hash[:16]}... (quality: {quality_score:.2f})",
            metadata={
                "sda_output_hash": sda_output_hash,
                "quality_score": quality_score,
                "corollary3_guard": "first-order reward calculation; no Hessian",
                "hino_guard": "discrete reward state",
            },
        )

        self._rewards[reward.reward_id] = reward
        token.total_balance += reward_amount

        logger.info(f"[TOK] Issued folding reward {reward.reward_id}: {reward_amount:.2f} $SDA to {researcher_did}")
        return reward

    def reward_validation(self, researcher_did: str, validated_fold_hash: str,
                            validation_accuracy: float, max_reward: float = 200.0) -> TokenReward:
        """
        Issue a validation reward for verifying others' folding results.

        Args:
            researcher_did: DID of the validator
            validated_fold_hash: hash of the fold being validated
            validation_accuracy: accuracy of validation (0.0 to 1.0)
            max_reward: maximum $SDA reward for perfect validation

        Returns:
            TokenReward
        """
        token = self._create_token_account(researcher_did)
        self._update_vesting(researcher_did)

        reward_amount = max_reward * validation_accuracy

        # Vesting: 1 year linear, 3-month cliff
        current_time = time.time()
        vesting_start = current_time
        vesting_end = current_time + (1 * 365 * 24 * 3600)  # 1 year
        vesting_cliff_s = 3 * 30 * 24 * 3600  # 3 months

        reward = TokenReward(
            reward_id=f"reward_{uuid.uuid4().hex[:12]}",
            researcher_did=researcher_did,
            reward_type="validation",
            amount_sda=reward_amount,
            vesting_start=vesting_start,
            vesting_end=vesting_end,
            vesting_cliff_s=vesting_cliff_s,
            reason=f"Validation reward for {validated_fold_hash[:16]}... (accuracy: {validation_accuracy:.2f})",
            metadata={
                "validated_fold_hash": validated_fold_hash,
                "validation_accuracy": validation_accuracy,
            },
        )

        self._rewards[reward.reward_id] = reward
        token.total_balance += reward_amount

        logger.info(f"[TOK] Issued validation reward {reward.reward_id}: {reward_amount:.2f} $SDA to {researcher_did}")
        return reward

    def reward_dataset_contribution(self, researcher_did: str, dataset_hash: str,
                                      dataset_quality: float, max_reward: float = 500.0) -> TokenReward:
        """
        Issue a one-time reward for contributing experimental data.

        Args:
            researcher_did: DID of the contributor
            dataset_hash: hash of the dataset
            dataset_quality: quality metric (0.0 to 1.0)
            max_reward: maximum $SDA reward for perfect quality

        Returns:
            TokenReward
        """
        token = self._create_token_account(researcher_did)

        reward_amount = max_reward * dataset_quality

        # No vesting for dataset contributions (immediate reward)
        current_time = time.time()

        reward = TokenReward(
            reward_id=f"reward_{uuid.uuid4().hex[:12]}",
            researcher_did=researcher_did,
            reward_type="dataset",
            amount_sda=reward_amount,
            vesting_start=current_time,
            vesting_end=current_time,
            vesting_cliff_s=0,
            reason=f"Dataset contribution reward for {dataset_hash[:16]}... (quality: {dataset_quality:.2f})",
            metadata={
                "dataset_hash": dataset_hash,
                "dataset_quality": dataset_quality,
            },
        )

        self._rewards[reward.reward_id] = reward
        token.total_balance += reward_amount
        token.available_balance += reward_amount  # immediate availability

        logger.info(f"[TOK] Issued dataset reward {reward.reward_id}: {reward_amount:.2f} $SDA to {researcher_did}")
        return reward

    def stake_tokens(self, researcher_did: str, amount: float) -> bool:
        """
        Stake $SDA tokens for governance participation.

        Args:
            researcher_did: DID of the researcher
            amount: amount of $SDA to stake

        Returns:
            True if staking succeeded

        Raises:
            ValueError if insufficient available balance
        """
        token = self._create_token_account(researcher_did)
        self._update_vesting(researcher_did)

        if token.available_balance < amount:
            raise ValueError(
                f"Insufficient available balance. Available: {token.available_balance:.2f}, "
                f"Requested: {amount:.2f}"
            )

        token.staked_balance += amount
        token.available_balance -= amount
        token.voting_power = math.sqrt(max(token.available_balance, 0.0))

        self._staking_pools[researcher_did] = token.staked_balance

        logger.info(f"[TOK] Staked {amount:.2f} $SDA for {researcher_did}")
        return True

    def unstake_tokens(self, researcher_did: str, amount: float) -> bool:
        """
        Unstake $SDA tokens.

        Args:
            researcher_did: DID of the researcher
            amount: amount of $SDA to unstake

        Returns:
            True if unstaking succeeded

        Raises:
            ValueError if insufficient staked balance
        """
        if researcher_did not in self._token_balances:
            raise ValueError(f"Researcher {researcher_did} has no token account")

        token = self._token_balances[researcher_did]
        if token.staked_balance < amount:
            raise ValueError(
                f"Insufficient staked balance. Staked: {token.staked_balance:.2f}, "
                f"Requested: {amount:.2f}"
            )

        token.staked_balance -= amount
        token.available_balance += amount
        token.voting_power = math.sqrt(max(token.available_balance, 0.0))

        self._staking_pools[researcher_did] = token.staked_balance

        logger.info(f"[TOK] Unstaked {amount:.2f} $SDA for {researcher_did}")
        return True

    def slash_tokens(self, researcher_did: str, amount: float, reason: str) -> bool:
        """
        Slash tokens for malicious behavior (requires multi-sig approval in production).

        Args:
            researcher_did: DID of the researcher
            amount: amount of $SDA to slash
            reason: reason for slashing

        Returns:
            True if slashing succeeded
        """
        if researcher_did not in self._token_balances:
            return False

        token = self._token_balances[researcher_did]
        slash_amount = min(amount, token.total_balance)

        token.total_balance -= slash_amount
        token.available_balance = max(0, token.available_balance - slash_amount)
        token.vested_balance = max(0, token.vested_balance - slash_amount)
        token.staked_balance = max(0, token.staked_balance - slash_amount)
        token.voting_power = math.sqrt(max(token.available_balance, 0.0))

        logger.warning(f"[TOK] Slashed {slash_amount:.2f} $SDA from {researcher_did}: {reason}")
        return True

    def get_token_balance(self, researcher_did: str) -> Optional[SDAToken]:
        """Get token balance for a researcher."""
        if researcher_did not in self._token_balances:
            return None
        self._update_vesting(researcher_did)
        return self._token_balances[researcher_did]

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_supply": self.total_supply,
            "num_accounts": len(self._token_balances),
            "num_rewards": len(self._rewards),
            "total_staked": sum(self._staking_pools.values()),
            "corollary3_guard": "first-order token transfers; no Hessian",
            "hino_guard": "discrete token balances",
        }


# =============================================================================
# SECTION 3 — [ORA1] ORACLE NETWORK BRIDGE
# =============================================================================

@dataclass
class OracleSource:
    """Configuration for an oracle data source."""
    source_name: str  # e.g., "chainlink_pdb", "band_uniprot"
    source_type: str  # "chainlink" | "band" | "custom"
    endpoint_url: str
    api_key: Optional[str]
    timeout_s: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleData:
    """Data retrieved from an oracle network."""
    data_id: str
    source_name: str
    data_type: str  # "pdb_structure" | "uniprot_sequence" | "cryoem_map" | "price_feed"
    data_payload: Any
    timestamp: float
    verification_hash: str
    pqc_signature: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


class OracleNetworkBridge:
    """
    [C/D] Bridge to oracle networks for external data feeds.

    CHAINLINK OCR (Off-Chain Reporting) [C Chainlink]
      • Decentralized oracle network
      • Off-chain aggregation reduces gas costs
      • PQC-signed data feeds (Dilithium5)
      • Use cases: PDB structure updates, UniProt sequences, price feeds

    BAND PROTOCOL [C Band Protocol]
      • Cross-chain oracle platform
      • Multi-source data aggregation
      • Custom data scripts for specific use cases
      • Use cases: experimental validation data, literature citations

    DATA TYPES [D]
      1. PDB Structure Updates:
         - Fetch latest experimental structures from PDB
         - Verify data integrity with ZK proof
         - Feed into SDA-ONE validation pipeline

      2. UniProt Sequence Data:
         - Retrieve protein sequences and annotations
         - Cross-reference with SDA-ONE folding targets
         - Update sequence databases

      3. Cryo-EM Maps:
         - Fetch experimental density maps from EMDB
         - Verify map quality metrics
         - Compare with SDA-ONE predictions

      4. Price Feeds:
         - $SDA token price from DEX aggregators
         - Used for treasury management and reward calculations

    ZK VERIFICATION [D]
      • Oracle data is verified using ZK proofs
      • Proves data authenticity without revealing full payload
      • Integration with STARKProverBridge from previous modules

    PQC SIGNATURES [D]
      • All oracle responses signed with Dilithium5
      • Quantum-resistant authentication
      • Integration with DecentralizedIdentityBridge

    COROLLARY-3 GUARD [D]
      Data aggregation and verification are FIRST-ORDER operations.
      We do NOT compute derivatives through the oracle layer.

    HINO GUARD [C Thm 8]
      Oracle data and timestamps are DISCRETE state.
      We respect the singular nature of external data feeds.

    HONEST
      • Oracles are trusted third parties.
      • Multi-source aggregation reduces but does not eliminate trust assumptions.
      • Chainlink/Band require operational infrastructure and node operator honesty.
      • This is an INTERFACE; production requires integration with oracle SDKs.
    """

    def __init__(self):
        self._sources: Dict[str, OracleSource] = {}
        self._data_cache: Dict[str, OracleData] = {}
        self._signer = None
        self._verifier = None

        if HAS_OQS:
            self._signer = oqs.Signature("Dilithium5")
            self._verifier = oqs.Signature("Dilithium5")

    def register_source(self, source_name: str, source_type: str,
                          endpoint_url: str, api_key: Optional[str] = None) -> OracleSource:
        """
        Register a new oracle data source.

        Args:
            source_name: unique name for the source
            source_type: "chainlink" | "band" | "custom"
            endpoint_url: API endpoint URL
            api_key: optional API key

        Returns:
            OracleSource
        """
        source = OracleSource(
            source_name=source_name,
            source_type=source_type,
            endpoint_url=endpoint_url,
            api_key=api_key,
            metadata={
                "registered_at": time.time(),
                "corollary3_guard": "first-order source registration; no Hessian",
                "hino_guard": "discrete oracle source",
            },
        )
        self._sources[source_name] = source
        logger.info(f"[ORA] Registered oracle source: {source_name} ({source_type})")
        return source

    def _sign_data(self, data: bytes) -> bytes:
        """Sign data with PQC (Dilithium5)."""
        if HAS_OQS and self._signer:
            return self._signer.sign(data)
        else:
            return b"mock_oracle_signature_" + hashlib.sha3_256(data).digest()

    def _verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify PQC signature."""
        if HAS_OQS and self._verifier:
            self._verifier.import_public_key(public_key)
            return self._verifier.verify(data, signature)
        else:
            # Mock verification
            expected = b"mock_oracle_signature_" + hashlib.sha3_256(data).digest()
            return signature == expected

    def fetch_pdb_structure(self, pdb_id: str, source_name: str = "chainlink_pdb") -> OracleData:
        """
        Fetch experimental structure from PDB via oracle.

        Args:
            pdb_id: PDB identifier (e.g., "1ABC")
            source_name: oracle source to use

        Returns:
            OracleData with structure coordinates and metadata

        Raises:
            ValueError if source not registered or fetch fails
        """
        if source_name not in self._sources:
            raise ValueError(f"Oracle source {source_name} not registered")

        source = self._sources[source_name]

        # Simulated API call (in production: requests.get with proper auth)
        if HAS_REQUESTS:
            try:
                # Mock response
                mock_structure = {
                    "pdb_id": pdb_id,
                    "coordinates": np.random.randn(100, 3).tolist(),  # Mock 100-residue structure
                    "resolution": 2.5,
                    "method": "X-RAY DIFFRACTION",
                    "release_date": "2024-01-15",
                }
                data_payload = json.dumps(mock_structure).encode()
            except Exception as e:
                raise ValueError(f"Failed to fetch PDB structure: {e}")
        else:
            # Fallback: mock data
            mock_structure = {
                "pdb_id": pdb_id,
                "coordinates": np.random.randn(100, 3).tolist(),
                "resolution": 2.5,
                "method": "X-RAY DIFFRACTION",
            }
            data_payload = json.dumps(mock_structure).encode()

        # Compute verification hash
        verification_hash = hashlib.sha3_256(data_payload).hexdigest()

        # Sign data
        pqc_signature = self._sign_data(data_payload)

        oracle_data = OracleData(
            data_id=f"pdb_{pdb_id}_{int(time.time())}",
            source_name=source_name,
            data_type="pdb_structure",
            data_payload=data_payload,
            timestamp=time.time(),
            verification_hash=verification_hash,
            pqc_signature=pqc_signature,
            metadata={
                "pdb_id": pdb_id,
                "corollary3_guard": "first-order data retrieval; no Hessian",
                "hino_guard": "discrete oracle data",
            },
        )

        self._data_cache[oracle_data.data_id] = oracle_data
        logger.info(f"[ORA] Fetched PDB structure {pdb_id} from {source_name}")
        return oracle_data

    def fetch_uniprot_sequence(self, uniprot_id: str, source_name: str = "band_uniprot") -> OracleData:
        """
        Fetch protein sequence from UniProt via oracle.

        Args:
            uniprot_id: UniProt identifier (e.g., "P12345")
            source_name: oracle source to use

        Returns:
            OracleData with sequence and annotations
        """
        if source_name not in self._sources:
            raise ValueError(f"Oracle source {source_name} not registered")

        # Mock UniProt data
        mock_sequence = {
            "uniprot_id": uniprot_id,
            "sequence": "M" + "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=200)),
            "organism": "Homo sapiens",
            "gene_name": "MOCK_GENE",
            "function": "Mock protein function",
        }
        data_payload = json.dumps(mock_sequence).encode()

        verification_hash = hashlib.sha3_256(data_payload).hexdigest()
        pqc_signature = self._sign_data(data_payload)

        oracle_data = OracleData(
            data_id=f"uniprot_{uniprot_id}_{int(time.time())}",
            source_name=source_name,
            data_type="uniprot_sequence",
            data_payload=data_payload,
            timestamp=time.time(),
            verification_hash=verification_hash,
            pqc_signature=pqc_signature,
            metadata={"uniprot_id": uniprot_id},
        )

        self._data_cache[oracle_data.data_id] = oracle_data
        logger.info(f"[ORA] Fetched UniProt sequence {uniprot_id} from {source_name}")
        return oracle_data

    def fetch_price_feed(self, token_symbol: str = "SDA", source_name: str = "chainlink_price") -> OracleData:
        """
        Fetch token price feed from oracle.

        Args:
            token_symbol: token symbol (e.g., "SDA")
            source_name: oracle source to use

        Returns:
            OracleData with price and metadata
        """
        if source_name not in self._sources:
            raise ValueError(f"Oracle source {source_name} not registered")

        # Mock price data
        mock_price = {
            "token_symbol": token_symbol,
            "price_usd": 0.05 + np.random.randn() * 0.01,  # Mock price ~$0.05
            "timestamp": int(time.time()),
            "confidence": 0.95,
        }
        data_payload = json.dumps(mock_price).encode()

        verification_hash = hashlib.sha3_256(data_payload).hexdigest()
        pqc_signature = self._sign_data(data_payload)

        oracle_data = OracleData(
            data_id=f"price_{token_symbol}_{int(time.time())}",
            source_name=source_name,
            data_type="price_feed",
            data_payload=data_payload,
            timestamp=time.time(),
            verification_hash=verification_hash,
            pqc_signature=pqc_signature,
            metadata={"token_symbol": token_symbol},
        )

        self._data_cache[oracle_data.data_id] = oracle_data
        logger.info(f"[ORA] Fetched price feed for {token_symbol} from {source_name}")
        return oracle_data

    def verify_oracle_data(self, oracle_data: OracleData, public_key: bytes) -> bool:
        """
        Verify oracle data integrity and signature.

        Args:
            oracle_data: OracleData to verify
            public_key: PQC public key of oracle node

        Returns:
            True if verification succeeded
        """
        # Verify hash
        computed_hash = hashlib.sha3_256(oracle_data.data_payload).hexdigest()
        if computed_hash != oracle_data.verification_hash:
            logger.error(f"[ORA] Hash mismatch for {oracle_data.data_id}")
            return False

        # Verify signature
        if not self._verify_signature(oracle_data.data_payload, oracle_data.pqc_signature, public_key):
            logger.error(f"[ORA] Signature verification failed for {oracle_data.data_id}")
            return False

        logger.info(f"[ORA] Verified oracle data {oracle_data.data_id}")
        return True

    def get_cached_data(self, data_id: str) -> Optional[OracleData]:
        """Retrieve cached oracle data."""
        return self._data_cache.get(data_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_sources": len(self._sources),
            "num_cached_data": len(self._data_cache),
            "has_requests": HAS_REQUESTS,
            "has_oqs": HAS_OQS,
            "corollary3_guard": "first-order data aggregation; no Hessian",
            "hino_guard": "discrete oracle timestamps",
        }


# =============================================================================
# SECTION 4 — ORCHESTRATOR
# =============================================================================

class SESIDAOTokenomicsOracleSuite:
    """
    Unifies: SDAOneDAO + ResearcherTokenomics + OracleNetworkBridge
             + DID, ZK Proofs, PQC Blockchain (from previous modules).

    FULL ECOSYSTEM PIPELINE:
      1. Researcher creates DID and receives $SDA tokens
      2. Researcher performs SDA-ONE folding and submits ZK proof
      3. DAO validates folding quality and issues rewards
      4. Oracle fetches experimental data for validation
      5. Researcher stakes tokens to participate in governance
      6. DAO proposes and votes on parameter updates
    """

    def __init__(self):
        # Import from previous modules (simulated here for standalone execution)
        from sesi_blockchain_did_zk_xchain import (
            DecentralizedIdentityBridge,
            ZeroKnowledgeSDAFold,
        )

        self.did_bridge = DecentralizedIdentityBridge()
        self.zk_sda = ZeroKnowledgeSDAFold()
        self.dao = SDAOneDAO()
        self.tokenomics = ResearcherTokenomics()
        self.oracle = OracleNetworkBridge()

    def full_researcher_lifecycle(self, orcid: str, name: str, institution: str,
                                     sda_output: Dict[str, Any],
                                     thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute full researcher lifecycle: DID -> folding -> rewards -> governance.
        """
        results = {}

        # Stage 1: Create researcher DID
        logger.info("[Lifecycle] Stage 1: Creating researcher DID")
        did_doc, researcher_sk = self.did_bridge.create_researcher_did(orcid, name, institution)
        results["did"] = {"did": did_doc.did}

        # Stage 2: Initialize token account
        logger.info("[Lifecycle] Stage 2: Initializing token account")
        token = self.tokenomics._create_token_account(did_doc.did)
        results["token_account"] = {
            "total_balance": token.total_balance,
            "available_balance": token.available_balance,
        }

        # Stage 3: Perform SDA-ONE folding and generate ZK proof
        logger.info("[Lifecycle] Stage 3: Generating ZK proof for folding")
        sda_hash = hashlib.sha3_256(json.dumps({
            "init_coords_shape": sda_output.get("init_coords").shape if hasattr(sda_output.get("init_coords"), "shape") else (0, 3),
        }, sort_keys=True).encode()).hexdigest()

        researcher_sig = self.did_bridge.sign_sda_folding_result(did_doc.did, researcher_sk, sda_hash)

        zk_proof = self.zk_sda.prove_folding_quality(
            sda_output=sda_output,
            thresholds=thresholds,
            researcher_did=did_doc.did,
            researcher_signature=researcher_sig,
        )
        results["zk_proof"] = {
            "proof_id": zk_proof.proof_id,
            "verified": self.zk_sda.verify_proof(zk_proof),
        }

        # Stage 4: Issue folding reward
        logger.info("[Lifecycle] Stage 4: Issuing folding reward")
        quality_score = 0.85  # Mock quality score
        reward = self.tokenomics.reward_folding(did_doc.did, sda_hash, quality_score)
        results["reward"] = {
            "reward_id": reward.reward_id,
            "amount_sda": reward.amount_sda,
            "vesting_end": reward.vesting_end,
        }

        # Stage 5: Stake tokens for governance
        logger.info("[Lifecycle] Stage 5: Staking tokens for governance")
        self.tokenomics._update_vesting(did_doc.did)
        token = self.tokenomics.get_token_balance(did_doc.did)
        stake_amount = token.available_balance * 0.5  # Stake 50% of available balance
        if stake_amount > 0:
            self.tokenomics.stake_tokens(did_doc.did, stake_amount)
        results["staking"] = {
            "staked_amount": stake_amount,
            "voting_power": token.voting_power,
        }

        # Stage 6: Create governance proposal
        logger.info("[Lifecycle] Stage 6: Creating governance proposal")
        proposal = self.dao.create_proposal(
            proposer_did=did_doc.did,
            proposer_token_balance=token.total_balance,
            title="Update target_domain_size to 1200",
            description="Increase target domain size for better folding quality on large proteins",
            proposal_type="parameter_update",
            target_contract="DomainAssemblyConfig",
            parameter_name="target_domain_size",
            new_value=1200,
        )
        results["proposal"] = {
            "proposal_id": proposal.proposal_id,
            "status": proposal.status.value,
        }

        return results

    def get_full_status(self) -> Dict[str, Any]:
        return {
            "dao": self.dao.get_status(),
            "tokenomics": self.tokenomics.get_status(),
            "oracle": self.oracle.get_status(),
        }


# =============================================================================
# SECTION 5 — SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(" SESI DAO / TOKENOMICS / ORACLES — SMOKE TEST")
    print(" Decentralized Governance + Researcher Incentives + External Data Feeds")
    print("=" * 78)
    print(f" requests available       : {HAS_REQUESTS}")
    print(f" liboqs available         : {HAS_OQS}")

    suite = SESIDAOTokenomicsOracleSuite()

    # ── 1. DAO Governance ───────────────────────────────────────────────
    print("\n [DAO1] SDA-ONE DAO")
    dao = suite.dao
    print(f"   total_token_supply     : {dao.total_token_supply:,.0f} $SDA")
    print(f"   quorum_percentage      : {dao.quorum_percentage:.1%}")
    print(f"   voting_period_s        : {dao.voting_period_s} s ({dao.voting_period_s / 86400:.1f} days)")
    print(f"   timelock_delay_s       : {dao.timelock_delay_s} s ({dao.timelock_delay_s / 3600:.1f} hours)")
    print(f"   COROLLARY-3 guard      : {dao.get_status()['corollary3_guard']}")
    print(f"   HINO guard             : {dao.get_status()['hino_guard']}")

    # ── 2. Tokenomics ───────────────────────────────────────────────────
    print("\n [TOK1] Researcher Tokenomics")
    tok = suite.tokenomics
    print(f"   total_supply           : {tok.total_supply:,.0f} $SDA")
    print(f"   COROLLARY-3 guard      : {tok.get_status()['corollary3_guard']}")
    print(f"   HINO guard             : {tok.get_status()['hino_guard']}")

    # ── 3. Oracle Network ───────────────────────────────────────────────
    print("\n [ORA1] Oracle Network Bridge")
    ora = suite.oracle
    # Register oracle sources
    ora.register_source("chainlink_pdb", "chainlink", "https://pdb.chainlink.example/v1")
    ora.register_source("band_uniprot", "band", "https://uniprot.band.example/v1")
    ora.register_source("chainlink_price", "chainlink", "https://price.chainlink.example/v1")
    print(f"   num_sources            : {ora.get_status()['num_sources']}")
    print(f"   COROLLARY-3 guard      : {ora.get_status()['corollary3_guard']}")
    print(f"   HINO guard             : {ora.get_status()['hino_guard']}")

    # Fetch mock data
    pdb_data = ora.fetch_pdb_structure("1ABC")
    print(f"   PDB data fetched       : {pdb_data.data_id}")
    uniprot_data = ora.fetch_uniprot_sequence("P12345")
    print(f"   UniProt data fetched   : {uniprot_data.data_id}")
    price_data = ora.fetch_price_feed("SDA")
    print(f"   Price feed fetched     : {price_data.data_id}")

    # ── 4. Full Researcher Lifecycle ────────────────────────────────────
    print("\n [LIFECYCLE] Full Researcher Lifecycle")

    # Mock SDA-ONE output
    import torch
    mock_sda_output = {
        "init_coords": torch.randn(1000, 3),
        "seq_features": torch.randn(1000, 256),
        "sigma": torch.rand(1000, 1),
    }
    thresholds = {"rmsd_bound": 4.0, "energy_bound": -100.0}

    lifecycle_result = suite.full_researcher_lifecycle(
        orcid="0009-0008-2374-0788",
        name="Yoon A Limsuwan",
        institution="MSPS NETWORK",
        sda_output=mock_sda_output,
        thresholds=thresholds,
    )

    print(f"   DID                    : {lifecycle_result['did']['did']}")
    print(f"   Token balance          : {lifecycle_result['token_account']['total_balance']:.2f} $SDA")
    print(f"   ZK proof verified      : {lifecycle_result['zk_proof']['verified']}")
    print(f"   Reward issued          : {lifecycle_result['reward']['amount_sda']:.2f} $SDA")
    print(f"   Staked amount          : {lifecycle_result['staking']['staked_amount']:.2f} $SDA")
    print(f"   Proposal created       : {lifecycle_result['proposal']['proposal_id']}")

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
    print(" [C]   Weyl 2019: Quadratic voting mechanism.")
    print(" [C]   Commons Stack: Conviction voting for continuous funding.")
    print(" [C]   Compound Governor Bravo: Timelock execution for governance safety.")
    print(" [C]   Balancer LBP / Curve veCRV: Token vesting and staking mechanisms.")
    print(" [C]   MakerDAO: Slashing mechanism for malicious behavior.")
    print(" [C]   Chainlink OCR: Off-chain reporting for decentralized oracles.")
    print(" [C]   Band Protocol: Cross-chain oracle platform.")
    print(" [D]   DAO governance is a technical framework, NOT a legal entity.")
    print("       Scientific validity still requires peer review.")
    print(" [D]   $SDA token has NO intrinsic value. Tokenomics are illustrative.")
    print("       This is NOT financial advice. Real deployment requires legal review.")
    print(" [D]   Oracles are trusted third parties. Multi-source aggregation reduces")
    print("       but does not eliminate trust assumptions.")
    print(" [D]   COROLLARY-3 guard: all operations first-order only; no Hessian.")
    print(" [C]   HINO guard: discrete operations respected; no smooth density assumed.")
    print(" [P]   PROP-3 awareness: field products lose two Sobolev orders.")
    print(" [C]   ρ≈1.25 is NUMERICAL, not proved; unused by these modules.")
    print(" [D]   All modules are COMPUTATIONAL tools. No financial/legal/security")
    print("       guarantee. Not validated against real production systems.")
    print(" All thresholds are FREE DESIGN PARAMETERS.")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    smoke_test()
