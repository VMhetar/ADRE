"""
ADRE Belief Management
Manages belief states, updates confidence, and tracks state history
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from adre_core_structures import BeliefState, Claim, ClaimType, REFRESH_INTERVALS
from adre_epistemic_engine import EpistemicEngine


class BeliefManager:
    """
    Manages the lifecycle of belief states.
    Handles confidence updates, status changes, and historical tracking.
    """

    def __init__(self, epistemic_engine: EpistemicEngine = None):
        """
        Initialize belief manager.
        
        Args:
            epistemic_engine: Engine for confidence computation (optional)
        """
        self.engine = epistemic_engine or EpistemicEngine()
        self.beliefs: Dict[str, BeliefState] = {}

    def create_belief(self, claim: Claim, base_confidence: float = 0.5) -> BeliefState:
        """
        Create new belief state from claim.
        
        Args:
            claim: Claim to create belief for
            base_confidence: Initial confidence level
            
        Returns:
            New BeliefState object
        """
        belief = BeliefState(
            claim=claim,
            confidence=base_confidence,
            uncertainty=1.0 - base_confidence
        )
        self.beliefs[claim.claim_id] = belief
        return belief

    def update_confidence(self, belief: BeliefState) -> float:
        """
        Recompute confidence based on evidence.
        Updates uncertainty as well.
        
        Args:
            belief: BeliefState to update
            
        Returns:
            New confidence value
        """
        belief.confidence = self.engine.compute_confidence(
            evidence_list=belief.evidence,
            last_verified=belief.last_verified,
            claim_type=belief.claim.claim_type
        )
        
        belief.uncertainty = self.engine.uncertainty_score(belief.confidence)
        
        return belief.confidence

    def update_status(self, belief: BeliefState) -> str:
        """
        Update belief status based on current confidence.
        Also logs change to history.
        
        Args:
            belief: BeliefState to update
            
        Returns:
            New status string
        """
        old_status = belief.status
        belief.status = self.engine.belief_status(belief.confidence)
        belief.last_updated = datetime.now(timezone.utc)
        
        # Log status change to history
        if belief.status != old_status:
            self._log_to_history(belief, f"status_change_{old_status}_to_{belief.status}")
        
        return belief.status

    def refresh_belief(self, belief: BeliefState) -> BeliefState:
        """
        Perform full refresh cycle:
        1. Recompute confidence
        2. Update status
        3. Update timestamps
        
        Args:
            belief: BeliefState to refresh
            
        Returns:
            Updated BeliefState
        """
        self.update_confidence(belief)
        self.update_status(belief)
        belief.last_verified = datetime.now(timezone.utc)
        
        self._log_to_history(belief, "belief_refreshed")
        
        return belief

    def needs_refresh(self, belief: BeliefState) -> bool:
        """
        Check if belief needs refreshing based on type and age.
        
        Args:
            belief: BeliefState to check
            
        Returns:
            True if refresh is needed
        """
        interval = REFRESH_INTERVALS[belief.claim.claim_type]
        time_since_update = datetime.now(timezone.utc) - belief.last_updated
        return time_since_update >= interval

    def update_evidence(
        self,
        belief: BeliefState,
        new_evidence
    ) -> BeliefState:
        """
        Add new evidence to belief and recompute.
        
        Args:
            belief: BeliefState to update
            new_evidence: Evidence object or list of Evidence
            
        Returns:
            Updated BeliefState
        """
        if isinstance(new_evidence, list):
            belief.evidence.extend(new_evidence)
        else:
            belief.evidence.append(new_evidence)
        
        # Update contradiction count
        belief.contradiction_count = sum(
            1 for e in belief.evidence
            if e.support_strength < 0
        )
        
        # Update source diversity
        unique_sources = set(e.source_id for e in belief.evidence)
        belief.source_diversity = len(unique_sources) / len(belief.evidence) if belief.evidence else 0
        
        # Recompute confidence and status
        self.update_confidence(belief)
        self.update_status(belief)
        
        self._log_to_history(belief, f"evidence_updated_{len(new_evidence) if isinstance(new_evidence, list) else 1}")
        
        return belief

    def get_belief(self, claim_id: str) -> Optional[BeliefState]:
        """
        Retrieve belief by claim ID.
        
        Args:
            claim_id: ID of claim to retrieve
            
        Returns:
            BeliefState or None if not found
        """
        return self.beliefs.get(claim_id)

    def get_all_beliefs(self) -> List[BeliefState]:
        """
        Get all managed beliefs.
        
        Returns:
            List of all BeliefState objects
        """
        return list(self.beliefs.values())

    def get_beliefs_by_status(self, status: str) -> List[BeliefState]:
        """
        Get all beliefs with specific status.
        
        Args:
            status: Status to filter by ('rejected', 'speculative', 'contextual', 'stable')
            
        Returns:
            List of BeliefState objects with matching status
        """
        return [b for b in self.beliefs.values() if b.status == status]

    def get_beliefs_needing_refresh(self) -> List[BeliefState]:
        """
        Get all beliefs that need refreshing.
        
        Returns:
            List of BeliefState objects needing refresh
        """
        return [b for b in self.beliefs.values() if self.needs_refresh(b)]

    def _log_to_history(self, belief: BeliefState, event: str) -> None:
        """
        Log event to belief history.
        
        Args:
            belief: BeliefState to log for
            event: Event description
        """
        belief.history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            'status': belief.status,
            'confidence': belief.confidence,
            'evidence_count': len(belief.evidence),
            'contradiction_count': belief.contradiction_count
        })

    def get_history(self, claim_id: str) -> List[Dict]:
        """
        Get complete history for a belief.
        
        Args:
            claim_id: ID of claim
            
        Returns:
            List of historical events
        """
        belief = self.get_belief(claim_id)
        return belief.history if belief else []

    def downgrade_belief(self, belief: BeliefState, reason: str = "") -> BeliefState:
        """
        Manually downgrade belief to lower status.
        Used when belief has too many contradictions or expires.
        
        Args:
            belief: BeliefState to downgrade
            reason: Reason for downgrade
            
        Returns:
            Downgraded BeliefState
        """
        old_status = belief.status
        belief.status = "speculative"
        belief.confidence = max(0.0, belief.confidence - 0.2)
        
        self._log_to_history(
            belief,
            f"downgraded_from_{old_status}_{reason}"
        )
        
        return belief

    def upgrade_belief(self, belief: BeliefState, reason: str = "") -> BeliefState:
        """
        Upgrade belief to higher status based on new evidence.
        Used when contradicted beliefs recover or gain strong support.
        
        Args:
            belief: BeliefState to upgrade
            reason: Reason for upgrade
            
        Returns:
            Upgraded BeliefState
        """
        old_status = belief.status
        belief.confidence = min(1.0, belief.confidence + 0.2)
        belief.status = self.engine.belief_status(belief.confidence)
        
        self._log_to_history(
            belief,
            f"upgraded_from_{old_status}_{reason}"
        )
        
        return belief