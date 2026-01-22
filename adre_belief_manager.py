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
    Manages belief lifecycle.
    Handles confidence updates, status changes, and minimal history tracking.
    """

    def __init__(self, epistemic_engine: EpistemicEngine = None):
        self.engine = epistemic_engine or EpistemicEngine()
        self.beliefs: Dict[str, BeliefState] = {}

    def create_belief(self, claim: Claim, base_confidence: float = 0.5) -> BeliefState:
        """Create new belief state from claim."""
        belief = BeliefState(
            claim=claim,
            confidence=base_confidence,
            uncertainty=1.0 - base_confidence
        )
        self.beliefs[claim.claim_id] = belief
        return belief

    def update_confidence(self, belief: BeliefState) -> float:
        """Recompute confidence based on evidence."""
        belief.confidence = self.engine.compute_confidence(
            evidence_list=belief.evidence,
            last_verified=belief.last_verified,
            claim_type=belief.claim.claim_type
        )
        belief.uncertainty = self.engine.uncertainty_score(belief.confidence)
        return belief.confidence

    def update_status(self, belief: BeliefState) -> str:
        """Update belief status based on confidence."""
        old_status = belief.status
        belief.status = self.engine.belief_status(belief.confidence)
        belief.last_updated = datetime.now(timezone.utc)
        
        if belief.status != old_status:
            self._log_to_history(belief, f"status:{old_status}->{belief.status}")
        
        return belief.status

    def refresh_belief(self, belief: BeliefState) -> BeliefState:
        """Full refresh: recompute confidence and update status."""
        self.update_confidence(belief)
        self.update_status(belief)
        belief.last_verified = datetime.now(timezone.utc)
        self._log_to_history(belief, "refresh")
        return belief

    def needs_refresh(self, belief: BeliefState) -> bool:
        """Check if belief needs refreshing based on type and age."""
        interval = REFRESH_INTERVALS[belief.claim.claim_type]
        time_since = datetime.now(timezone.utc) - belief.last_updated
        return time_since >= interval

    def update_evidence(self, belief: BeliefState, new_evidence) -> BeliefState:
        """Add new evidence and recompute confidence."""
        if isinstance(new_evidence, list):
            belief.evidence.extend(new_evidence)
        else:
            belief.evidence.append(new_evidence)
        
        # Update metadata
        belief.contradiction_count = sum(
            1 for e in belief.evidence if e.support_strength < 0
        )
        
        unique_sources = set(e.source_id for e in belief.evidence)
        belief.source_diversity = len(unique_sources) / len(belief.evidence) if belief.evidence else 0
        
        # Recompute
        self.update_confidence(belief)
        self.update_status(belief)
        
        num_new = len(new_evidence) if isinstance(new_evidence, list) else 1
        self._log_to_history(belief, f"evidence:+{num_new}")
        
        return belief

    def bulk_update(self, beliefs: List[BeliefState]) -> List[BeliefState]:
        """Update multiple beliefs efficiently."""
        for belief in beliefs:
            self.update_confidence(belief)
            self.update_status(belief)
        return beliefs

    def get_belief(self, claim_id: str) -> Optional[BeliefState]:
        """Retrieve belief by claim ID."""
        return self.beliefs.get(claim_id)

    def get_all_beliefs(self) -> List[BeliefState]:
        """Get all managed beliefs."""
        return list(self.beliefs.values())

    def get_beliefs_by_status(self, status: str) -> List[BeliefState]:
        """Get beliefs with specific status."""
        return [b for b in self.beliefs.values() if b.status == status]

    def get_beliefs_needing_refresh(self) -> List[BeliefState]:
        """Get beliefs that need refreshing."""
        return [b for b in self.beliefs.values() if self.needs_refresh(b)]

    def _log_to_history(self, belief: BeliefState, event: str) -> None:
        """Log event to belief history - minimal 3-field logging."""
        belief.history.append({
            'ts': datetime.now(timezone.utc).isoformat(),
            'event': event,
            'conf': round(belief.confidence, 3)
        })

    def get_history(self, claim_id: str) -> List[Dict]:
        """Get complete history for a belief."""
        belief = self.get_belief(claim_id)
        return belief.history if belief else []

    def downgrade_belief(self, belief: BeliefState, reason: str = "") -> BeliefState:
        """Manually downgrade belief to lower status."""
        belief.confidence = max(0.0, belief.confidence - 0.2)
        belief.status = self.engine.belief_status(belief.confidence)
        self._log_to_history(belief, f"downgrade:{reason}")
        return belief

    def upgrade_belief(self, belief: BeliefState, reason: str = "") -> BeliefState:
        """Upgrade belief to higher status."""
        belief.confidence = min(1.0, belief.confidence + 0.2)
        belief.status = self.engine.belief_status(belief.confidence)
        self._log_to_history(belief, f"upgrade:{reason}")
        return belief
