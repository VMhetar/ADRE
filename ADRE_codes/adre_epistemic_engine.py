"""
ADRE Epistemic Engine
Core mathematical functions for confidence computation
"""

from typing import List, Set
from datetime import datetime, timezone
import math

from adre_core_structures import Evidence, ClaimType, DECAY_RATES


class EpistemicEngine:
    """
    Computes epistemic confidence for beliefs.
    All confidence calculations flow through this engine.
    """

    @staticmethod
    def evidence_support(evidence_list: List[Evidence]) -> float:
        """
        Compute weighted average support strength from evidence
        Takes into account source reliability scores
        
        Args:
            evidence_list: List of evidence objects
            
        Returns:
            Weighted average support strength (-1.0 to 1.0)
        """
        if not evidence_list:
            return 0.0

        weighted_support = sum(
            e.support_strength * e.reliability_score 
            for e in evidence_list
        )
        return weighted_support / len(evidence_list)

    @staticmethod
    def contradiction_penalty(evidence_list: List[Evidence]) -> float:
        """
        Penalize contradicting evidence.
        Contradictions hurt more than confirmations help (asymmetric penalty).
        
        Args:
            evidence_list: List of evidence objects
            
        Returns:
            Penalty amount (0.0 to 0.5)
        """
        contradictions = sum(
            1 for e in evidence_list 
            if e.support_strength < 0
        )
        penalty = contradictions * 0.15  # 15% penalty per contradiction
        return min(0.5, penalty)

    @staticmethod
    def source_diversity_bonus(evidence_list: List[Evidence]) -> float:
        """
        Reward evidence from diverse independent sources.
        Prevents single-point-of-failure trust and echo chambers.
        
        Args:
            evidence_list: List of evidence objects
            
        Returns:
            Diversity bonus (0.0 to 0.3)
        """
        if not evidence_list:
            return 0.0

        unique_sources: Set[str] = {e.source_id for e in evidence_list}
        diversity_ratio = len(unique_sources) / len(evidence_list)
        
        return min(0.3, diversity_ratio * 0.3)

    @staticmethod
    def time_decay(
        last_verified: datetime,
        decay_rate: float = 0.01
    ) -> float:
        """
        Apply exponential decay to confidence.
        Truth expires unless refreshed.
        
        Args:
            last_verified: When evidence was last gathered
            decay_rate: Decay rate (claim-type dependent)
            
        Returns:
            Decay penalty (0.0 to 1.0)
        """
        days_passed = (datetime.now(timezone.utc) - last_verified).days
        
        if days_passed <= 0:
            return 0.0

        return 1 - math.exp(-decay_rate * days_passed)

    @staticmethod
    def compute_confidence(
        evidence_list: List[Evidence],
        last_verified: datetime,
        claim_type: ClaimType,
        base_confidence: float = 0.5
    ) -> float:
        """
        Compute epistemic confidence for a belief.
        
        Formula:
            confidence = base + support + diversity - contradiction - decay
        
        Args:
            evidence_list: Supporting/contradicting evidence
            last_verified: When evidence was last gathered
            claim_type: Type of claim (determines decay rate)
            base_confidence: Starting confidence (default 0.5)
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        decay_rate = DECAY_RATES[claim_type]

        support = EpistemicEngine.evidence_support(evidence_list)
        contradiction = EpistemicEngine.contradiction_penalty(evidence_list)
        diversity = EpistemicEngine.source_diversity_bonus(evidence_list)
        decay = EpistemicEngine.time_decay(last_verified, decay_rate)

        confidence = (
            base_confidence
            + support
            + diversity
            - contradiction
            - decay
        )
        
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def belief_status(confidence: float) -> str:
        """
        Map confidence score to belief status.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Status: 'rejected', 'speculative', 'contextual', or 'stable'
        """
        if confidence < 0.3:
            return "rejected"
        elif confidence < 0.5:
            return "speculative"
        elif confidence < 0.75:
            return "contextual"
        else:
            return "stable"

    @staticmethod
    def uncertainty_score(confidence: float) -> float:
        """
        Compute explicit uncertainty representation.
        Not just (1 - confidence), but bounded for safety.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Uncertainty score (0.0 to 1.0)
        """
        return max(0.0, min(1.0, 1.0 - confidence))