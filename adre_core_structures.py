"""
ADRE Core Data Structures
Defines all fundamental data types and enums
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from enum import Enum


class ClaimType(Enum):
    """Types of claims with different decay characteristics"""
    STRUCTURAL = 'structural'
    EMPIRICAL = 'empirical'
    DYNAMIC = 'dynamic'
    NORMATIVE = 'normative'


# Decay rates for each claim type
DECAY_RATES = {
    ClaimType.STRUCTURAL: 0.0001,  # Barely decays (structural facts)
    ClaimType.EMPIRICAL: 0.005,    # Slow decay (scientific findings)
    ClaimType.DYNAMIC: 0.05,       # Fast decay (current events)
    ClaimType.NORMATIVE: 0.02      # Medium decay (norms/policies)
}

# Refresh intervals for each claim type
REFRESH_INTERVALS = {
    ClaimType.STRUCTURAL: timedelta(days=365),
    ClaimType.EMPIRICAL: timedelta(days=90),
    ClaimType.DYNAMIC: timedelta(days=7),
    ClaimType.NORMATIVE: timedelta(days=30),
}


@dataclass
class Evidence:
    """
    Represents evidence supporting or contradicting a claim
    
    Attributes:
        source_id: Unique identifier for the source
        source_type: Type of source ('news', 'official', 'api', 'academic', 'social_media')
        support_strength: -1.0 (contradicts) to 1.0 (strongly supports)
        timestamp: When evidence was gathered
        content: The actual evidence content
        reliability_score: Source trustworthiness (0.0 to 1.0)
    """
    source_id: str
    source_type: str
    support_strength: float
    timestamp: datetime
    content: str = ""
    reliability_score: float = 0.5


@dataclass
class Claim:
    """
    Extracted claim from raw data requiring validation
    
    Attributes:
        claim_id: Unique identifier
        text: The claim statement
        claim_type: Type of claim (affects decay rate)
        source_text: Original text where claim was extracted from
        extraction_confidence: How certain was the extraction (0.0 to 1.0)
        timestamp: When claim was extracted
    """
    claim_id: str
    text: str
    claim_type: ClaimType
    source_text: str
    extraction_confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BeliefState:
    """
    Complete epistemic state of a claim
    
    Attributes:
        claim: The claim being evaluated
        confidence: Epistemic confidence (0.0 to 1.0)
        uncertainty: Explicit uncertainty representation
        evidence: List of supporting/contradicting evidence
        contradiction_count: Number of contradicting evidence pieces
        source_diversity: Ratio of unique sources
        verification_count: Number of times verified
        status: One of 'rejected', 'speculative', 'contextual', 'stable'
        last_verified: When evidence was last gathered
        last_updated: When belief state was last modified
        history: Log of state changes over time
    """
    claim: Claim
    confidence: float = 0.5
    uncertainty: float = 0.5
    evidence: List[Evidence] = field(default_factory=list)
    contradiction_count: int = 0
    source_diversity: float = 0.0
    verification_count: int = 0
    status: str = "unverified"
    last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert belief state to dictionary"""
        return {
            'claim': self.claim.text,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'status': self.status,
            'evidence_count': len(self.evidence),
            'contradiction_count': self.contradiction_count,
            'source_diversity': self.source_diversity,
            'last_verified': self.last_verified.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

    def to_training_format(self) -> Dict|None:
        """Convert belief to training data format"""
        if self.status not in ["stable", "contextual"]:
            return None
        
        return {
            'claim': self.claim.text,
            'confidence': self.confidence,
            'evidence_count': len(self.evidence),
            'sources': list(set(e.source_id for e in self.evidence)),
            'status': self.status,
            'source_types': list(set(e.source_type for e in self.evidence))
        }