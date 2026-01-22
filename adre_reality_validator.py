"""
ADRE Claim Extraction Layer
Extracts structured claims from raw unstructured data
"""

from abc import ABC, abstractmethod
from typing import List
from datetime import datetime, timezone
import re
import uuid

from adre_core_structures import Claim, ClaimType


class ClaimExtractor(ABC):
    """Base class for claim extraction strategies."""

    @abstractmethod
    def extract(self, raw_data: str) -> List[Claim]:
        """Extract claims from raw data."""
        pass


class PatternBasedExtractor(ClaimExtractor):
    """
    Pattern-based claim extraction.
    Fast, no external dependencies, no I/O.
    """

    def __init__(self):
        # Strong claim indicators
        self.factual_patterns = [
            r'\b(?:is|are|was|were)\b.*\b(?:a|an|the)\b',
            r'\b(?:has|have|had)\b.*\b(?:shown|demonstrated|proven)\b',
            r'\b(?:research|studies|data|evidence)\b.*\b(?:shows?|indicates?|suggests?)\b',
        ]
        
        # Compiled patterns
        self.compiled = [re.compile(p, re.IGNORECASE) for p in self.factual_patterns]

    def extract(self, raw_data: str) -> List[Claim]:
        """Extract claims using pattern matching - synchronous."""
        claims = []
        sentences = re.split(r'[.!?]+', raw_data)
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if len(sentence) < 20:
                continue
            
            # Check if matches factual patterns
            if self._is_factual_claim(sentence):
                claim_type = self._classify_claim_type(sentence)
                confidence = self._estimate_confidence(sentence)
                
                claim = Claim(
                    claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                    text=sentence,
                    claim_type=claim_type,
                    source_text=raw_data[:200],
                    extraction_confidence=confidence,
                    timestamp=datetime.now(timezone.utc)
                )
                claims.append(claim)
        
        return claims

    def _is_factual_claim(self, text: str) -> bool:
        """Check if text contains factual claim."""
        return any(pattern.search(text) for pattern in self.compiled)

    def _classify_claim_type(self, text: str) -> ClaimType:
        """Classify claim type with stronger heuristics."""
        lower = text.lower()
        
        # Normative - prescriptive language
        normative_keywords = {'should', 'must', 'ought', 'need to', 'required', 'necessary'}
        if any(kw in lower for kw in normative_keywords):
            return ClaimType.NORMATIVE
        
        # Dynamic - time-sensitive indicators
        dynamic_keywords = {'today', 'yesterday', 'currently', 'now', 'latest', 'recent'}
        if any(kw in lower for kw in dynamic_keywords):
            return ClaimType.DYNAMIC
        
        # Empirical - research/data language
        empirical_keywords = {'study', 'research', 'data shows', 'experiment', 'findings', 'evidence'}
        if any(kw in lower for kw in empirical_keywords):
            return ClaimType.EMPIRICAL
        
        # Structural - definitional/mathematical
        structural_keywords = {'defined as', 'equals', 'always', 'never', 'formula', 'theorem'}
        if any(kw in lower for kw in structural_keywords):
            return ClaimType.STRUCTURAL
        
        # Default to empirical
        return ClaimType.EMPIRICAL

    def _estimate_confidence(self, text: str) -> float:
        """Estimate extraction confidence based on claim strength."""
        lower = text.lower()
        confidence = 0.5
        
        # Strong indicators boost confidence
        strong_words = {'proven', 'demonstrated', 'established', 'confirmed'}
        if any(w in lower for w in strong_words):
            confidence += 0.2
        
        # Hedging reduces confidence
        hedge_words = {'might', 'could', 'possibly', 'perhaps', 'maybe'}
        if any(w in lower for w in hedge_words):
            confidence -= 0.15
        
        # Length factor
        if len(text.split()) > 15:
            confidence += 0.1
        
        return max(0.3, min(0.95, confidence))


class LLMBasedExtractor(ClaimExtractor):
    """
    LLM-powered claim extraction using OpenRouter.
    More sophisticated but requires API access.
    """

    def __init__(self, llm_api_key: str = None, model: str = "openai/gpt-3.5-turbo"):
        from adre_llm_service import OpenRouterLLMService, LLMClaimExtractor, LLMClaimClassifier
        
        try:
            self.llm_service = OpenRouterLLMService(api_key=llm_api_key, model=model)
            self.claim_extractor = LLMClaimExtractor(self.llm_service)
            self.classifier = LLMClaimClassifier(self.llm_service)
            self.fallback = PatternBasedExtractor()
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM service: {e}")

    def extract(self, raw_data: str) -> List[Claim]:
        """Extract claims using OpenRouter LLM with fallback - synchronous wrapper."""
        import asyncio
        
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context - create task
                return asyncio.create_task(self._async_extract(raw_data))
            except RuntimeError:
                # No running loop - create new one
                return asyncio.run(self._async_extract(raw_data))
        except Exception as e:
            print(f"LLM extraction failed: {e}, using fallback")
            return self.fallback.extract(raw_data)

    async def _async_extract(self, raw_data: str) -> List[Claim]:
        """Async extraction logic."""
        extracted = await self.claim_extractor.extract_claims(raw_data)
        
        if not extracted:
            print("No claims extracted by LLM, using fallback")
            return self.fallback.extract(raw_data)
        
        claims = []
        for item in extracted:
            claim_text = item.get("claim", "")
            if not claim_text:
                continue
                
            extraction_conf = float(item.get("confidence", 0.7))
            
            # Classify claim type
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
    Hybrid: fast patterns + optional LLM refinement.
    Best of both worlds.
    """

    def __init__(self, llm_api_key: str = None):
        self.pattern_extractor = PatternBasedExtractor()
        try:
            self.llm_extractor = LLMBasedExtractor(llm_api_key)
            self.has_llm = True
        except:
            self.has_llm = False

    def extract(self, raw_data: str) -> List[Claim]:
        """Extract with pattern first, optionally refine with LLM."""
        pattern_claims = self.pattern_extractor.extract(raw_data)
        
        if not self.has_llm:
            return pattern_claims
        
        # Boost confidence for hybrid extraction
        for claim in pattern_claims:
            claim.extraction_confidence = min(0.9, claim.extraction_confidence + 0.1)
        
        return pattern_claims
