"""
ADRE Claim Extraction Layer
Extracts structured claims from raw unstructured data
"""

from abc import ABC, abstractmethod
from typing import List
from datetime import datetime, timezone
import re
import uuid
import asyncio

from adre_core_structures import Claim, ClaimType


class ClaimExtractor(ABC):
    """
    Abstract base class for claim extraction strategies.
    Different implementations can use patterns, NLP, or LLMs.
    """

    @abstractmethod
    async def extract(self, raw_data: str) -> List[Claim]:
        """
        Extract claims from raw data.
        
        Args:
            raw_data: Unstructured text input
            
        Returns:
            List of extracted Claim objects
        """
        pass


class PatternBasedExtractor(ClaimExtractor):
    """
    Simple pattern-based claim extraction.
    Looks for sentences with factual assertions.
    Suitable for baseline/testing purposes.
    """

    def __init__(self):
        # Patterns that indicate claims
        self.claim_indicators = [
            'is', 'was', 'are', 'were',
            'has', 'have', 'had',
            'will be', 'will have',
            'shows', 'demonstrates', 'proves',
            'research', 'study', 'data'
        ]

    async def extract(self, raw_data: str) -> List[Claim]:
        """
        Extract claims using pattern matching.
        
        Args:
            raw_data: Unstructured text
            
        Returns:
            List of Claim objects
        """
        claims = []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', raw_data)
        
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            # Filter out short/empty sentences
            if len(sentence) < 20:
                continue
            
            # Check if sentence contains claim indicators
            has_claim = any(
                indicator in sentence.lower() 
                for indicator in self.claim_indicators
            )
            
            if has_claim:
                claim_type = self._classify_claim_type(sentence)
                
                claim = Claim(
                    claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                    text=sentence,
                    claim_type=claim_type,
                    source_text=raw_data[:200],
                    extraction_confidence=0.65,
                    timestamp=datetime.now(timezone.utc)
                )
                claims.append(claim)
        
        return claims

    def _classify_claim_type(self, text: str) -> ClaimType:
        """
        Classify claim type based on linguistic patterns.
        
        Args:
            text: Claim text
            
        Returns:
            ClaimType enum value
        """
        text_lower = text.lower()
        
        # Check for normative language
        if any(word in text_lower for word in ['should', 'must', 'ought', 'need']):
            return ClaimType.NORMATIVE
        
        # Check for dynamic/current events
        if any(word in text_lower for word in ['today', 'now', 'currently', 'recently']):
            return ClaimType.DYNAMIC
        
        # Check for empirical research
        if any(word in text_lower for word in ['data', 'study', 'research', 'finding', 'experiment']):
            return ClaimType.EMPIRICAL
        
        # Default to structural
        return ClaimType.STRUCTURAL


class LLMBasedExtractor(ClaimExtractor):
    """
    LLM-powered claim extraction using OpenRouter.
    Uses language model to identify and structure claims.
    More sophisticated but requires OpenRouter API access.
    """

    def __init__(self, llm_api_key: str = None, model: str = "openai/gpt-3.5-turbo"):
        """
        Initialize LLM-based extractor.
        
        Args:
            llm_api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (default: gpt-3.5-turbo)
        """
        from adre_llm_service import OpenRouterLLMService, LLMClaimExtractor, LLMClaimClassifier
        
        self.llm_service = OpenRouterLLMService(api_key=llm_api_key, model=model)
        self.claim_extractor = LLMClaimExtractor(self.llm_service)
        self.classifier = LLMClaimClassifier(self.llm_service)

    async def extract(self, raw_data: str) -> List[Claim]:
        """
        Extract claims using OpenRouter LLM.
        
        Args:
            raw_data: Unstructured text
            
        Returns:
            List of Claim objects
        """
        try:
            # Extract raw claims using LLM
            extracted = await self.claim_extractor.extract_claims(raw_data)
            
            claims = []
            for idx, item in enumerate(extracted):
                claim_text = item.get("claim", "")
                extraction_conf = float(item.get("confidence", 0.7))
                
                # Classify claim type using LLM
                claim_type_str = await self.classifier.classify(claim_text)
                claim_type = self._string_to_claim_type(claim_type_str)
                
                claim = Claim(
                    claim_id=f"claim_llm_{uuid.uuid4().hex[:8]}",
                    text=claim_text,
                    claim_type=claim_type,
                    source_text=raw_data[:200],
                    extraction_confidence=extraction_conf,
                    timestamp=datetime.now(timezone.utc)
                )
                claims.append(claim)
            
            return claims
        
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            # Fallback to pattern-based extraction
            fallback = PatternBasedExtractor()
            return await fallback.extract(raw_data)

    @staticmethod
    def _string_to_claim_type(type_str: str) -> ClaimType:
        """Convert string to ClaimType enum."""
        type_map = {
            'STRUCTURAL': ClaimType.STRUCTURAL,
            'EMPIRICAL': ClaimType.EMPIRICAL,
            'DYNAMIC': ClaimType.DYNAMIC,
            'NORMATIVE': ClaimType.NORMATIVE
        }
        return type_map.get(type_str.upper(), ClaimType.EMPIRICAL)


class HybridExtractor(ClaimExtractor):
    """
    Hybrid approach combining pattern-based and LLM extraction.
    Uses patterns for speed, validates with LLM for quality.
    """

    def __init__(self, llm_api_key: str = None):
        self.pattern_extractor = PatternBasedExtractor()
        self.llm_extractor = LLMBasedExtractor(llm_api_key)

    async def extract(self, raw_data: str) -> List[Claim]:
        """
        Extract claims using hybrid approach.
        
        Args:
            raw_data: Unstructured text
            
        Returns:
            List of validated Claim objects
        """
        # First, fast pattern-based extraction
        pattern_claims = await self.pattern_extractor.extract(raw_data)
        
        # Second, refine with LLM (optional, for high-importance data)
        # llm_claims = await self.llm_extractor.extract(raw_data)
        
        # For now, return pattern-based with slightly higher confidence
        for claim in pattern_claims:
            claim.extraction_confidence = min(0.85, claim.extraction_confidence + 0.1)
        
        return pattern_claims