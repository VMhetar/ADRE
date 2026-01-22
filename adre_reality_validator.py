"""
ADRE Reality Validation Layer
Validates claims against multiple external sources
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timezone
import asyncio

from adre_core_structures import Evidence, Claim, ClaimType


class RealityValidator(ABC):
    """Base class for reality validation strategies."""

    @abstractmethod
    async def validate(self, claim: Claim) -> List[Evidence]:
        """Validate claim against reality sources."""
        pass


class MockValidator(RealityValidator):
    """
    Deterministic mock validator for testing.
    Generates reproducible evidence based on claim characteristics.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock validator.
        
        Args:
            seed: Seed for deterministic behavior (not used for random, but for hashing)
        """
        self.seed = seed
        self.sources = [
            ('official_db', 'official', 0.95),
            ('academic_journal', 'academic', 0.90),
            ('news_outlet_1', 'news', 0.75),
            ('news_outlet_2', 'news', 0.75),
            ('government_api', 'official', 0.92),
        ]

    async def validate(self, claim: Claim) -> List[Evidence]:
        """Generate deterministic evidence based on claim hash."""
        evidence_list = []
        
        # Deterministic selection based on claim text hash
        text_hash = hash(claim.text) % 100
        num_sources = 2 + (text_hash % 4)  # 2-5 sources
        
        for i in range(min(num_sources, len(self.sources))):
            source_id, source_type, reliability = self.sources[i]
            
            # Deterministic support strength
            support = self._compute_support(claim, i, text_hash)
            
            evidence = Evidence(
                source_id=f"{source_id}_{i}",
                source_type=source_type,
                support_strength=support,
                reliability_score=reliability,
                timestamp=datetime.now(timezone.utc),
                content=f"Evidence from {source_id}: {claim.text[:50]}"
            )
            evidence_list.append(evidence)
        
        return evidence_list

    def _compute_support(self, claim: Claim, source_idx: int, text_hash: int) -> float:
        """Compute deterministic support strength based on claim type and hash."""
        # Base support depends on claim type
        type_base = {
            ClaimType.STRUCTURAL: 0.8,
            ClaimType.EMPIRICAL: 0.6,
            ClaimType.DYNAMIC: 0.4,
            ClaimType.NORMATIVE: 0.5
        }
        base = type_base.get(claim.claim_type, 0.5)
        
        # Vary by source (deterministic)
        variation = ((text_hash + source_idx * 17) % 40 - 20) / 100  # -0.2 to +0.2
        support = base + variation
        
        # 10% chance of contradiction (deterministic)
        if (text_hash + source_idx) % 10 == 0:
            support = -0.5 - (text_hash % 30) / 100
        
        return max(-1.0, min(1.0, support))


class WebSearchValidator(RealityValidator):
    """
    Web search validation (placeholder - requires implementation).
    """

    def __init__(self, search_api_key: str = None, llm_service=None, max_sources: int = 5):
        self.api_key = search_api_key
        self.max_sources = max_sources
        self.llm_service = llm_service
        self.enabled = False  # Not implemented yet

    async def validate(self, claim: Claim) -> List[Evidence]:
        """Web search validation - not implemented."""
        if not self.enabled:
            return []
        
        # TODO: Implement web search
        return []


class APIValidator(RealityValidator):
    """
    API-based validation (placeholder - requires implementation).
    """

    def __init__(self):
        self.enabled = False  # Not implemented yet

    async def validate(self, claim: Claim) -> List[Evidence]:
        """API validation - not implemented."""
        if not self.enabled:
            return []
        
        # TODO: Implement API queries
        return []


class MultiSourceValidator(RealityValidator):
    """
    Orchestrates validation across multiple validators.
    Only uses enabled validators.
    """

    def __init__(self, include_mock: bool = True):
        self.validators = []
        
        if include_mock:
            self.validators.append(MockValidator())
        
        # Add other validators when implemented
        # self.validators.append(WebSearchValidator())
        # self.validators.append(APIValidator())

    async def validate(self, claim: Claim) -> List[Evidence]:
        """Validate across all enabled validators."""
        if not self.validators:
            # Fallback to mock if no validators
            mock = MockValidator()
            return await mock.validate(claim)
        
        # Run all validators in parallel
        tasks = [validator.validate(claim) for validator in self.validators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results, filter errors
        all_evidence = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Validator failed: {result}")
                continue
            all_evidence.extend(result)
        
        # Deduplicate by source
        return self._deduplicate(all_evidence)

    @staticmethod
    def _deduplicate(evidence_list: List[Evidence]) -> List[Evidence]:
        """Remove duplicate evidence from same source."""
        seen = set()
        unique = []
        
        for evidence in evidence_list:
            key = (evidence.source_id, evidence.source_type)
            if key not in seen:
                seen.add(key)
                unique.append(evidence)
        
        return unique
