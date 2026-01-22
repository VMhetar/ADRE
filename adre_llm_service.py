"""
ADRE LLM Service
Integrates with OpenRouter API for LLM-powered features
"""

import os
import httpx
import logging
import json
import asyncio
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService(ABC):
    """Abstract base for LLM services"""
    
    @abstractmethod
    async def call(self, prompt: str) -> str:
        """Make LLM call with prompt."""
        pass


class OpenRouterLLMService(LLMService):
    """
    OpenRouter LLM Service with retry logic and robust error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-3.5-turbo",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
        
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def call(self, prompt: str, temperature: float = 0.7) -> str:
        """Make LLM call with retry logic and rate limit handling."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.base_url,
                        headers=self.headers,
                        json=data
                    )
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        logger.error(f"API error {response.status_code}: {response.text}")
                        raise Exception(f"API error: {response.status_code}")
                    
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception("Unexpected response format")
            
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(1)
            except Exception as e:
                last_error = e
                logger.warning(f"Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                await asyncio.sleep(1)
        
        raise Exception(f"Failed after {self.max_retries} retries: {last_error}")

    async def call_json(self, prompt: str) -> Dict:
        """Make LLM call expecting JSON with robust parsing."""
        response = await self.call(prompt, temperature=0.0)
        
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code blocks
        if "```json" in response:
            try:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            except:
                pass
        
        # Try extracting JSON from plain markdown blocks
        if "```" in response:
            try:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            except:
                pass
        
        # Try finding JSON object/array in response
        for char in ['{', '[']:
            try:
                start = response.find(char)
                end = response.rfind('}' if char == '{' else ']') + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
        
        raise ValueError(f"Could not parse JSON from response: {response[:200]}")


class MockLLMService(LLMService):
    """Mock LLM for testing - deterministic responses."""

    def __init__(self):
        self.call_count = 0

    async def call(self, prompt: str) -> str:
        """Return mock response based on prompt content."""
        self.call_count += 1
        
        if "extract" in prompt.lower():
            return '''[
                {"claim": "The Earth orbits the Sun", "type": "STRUCTURAL", "confidence": 0.95},
                {"claim": "Machine learning requires quality data", "type": "EMPIRICAL", "confidence": 0.85}
            ]'''
        
        if "classify" in prompt.lower():
            return '{"type": "EMPIRICAL", "reasoning": "Mock classification"}'
        
        if "evidence" in prompt.lower() or "support" in prompt.lower():
            return '{"support_strength": 0.7, "reasoning": "Mock evidence analysis"}'
        
        return "Mock LLM response"

    async def call_json(self, prompt: str) -> Dict:
        """Return mock JSON response."""
        response = await self.call(prompt)
        return json.loads(response)


# ============================================================================
# LLM-POWERED FUNCTIONS
# ============================================================================

class LLMClaimExtractor:
    """Uses LLM to extract and structure claims."""

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def extract_claims(self, text: str) -> List[Dict]:
        """Extract claims from text using LLM."""
        prompt = f"""Extract factual claims from this text.

For each claim provide:
- claim: exact claim text
- type: STRUCTURAL, EMPIRICAL, DYNAMIC, or NORMATIVE
- confidence: extraction confidence (0.0-1.0)

Return ONLY a JSON array, nothing else.

Text:
{text}

JSON:"""

        try:
            response = await self.llm.call_json(prompt)
            if isinstance(response, list):
                return response
            return [response] if response else []
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []


class LLMEvidenceAnalyzer:
    """Uses LLM to analyze and score evidence."""

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def analyze_evidence(self, claim: str, evidence_text: str) -> float:
        """Analyze how much evidence supports claim."""
        prompt = f"""Analyze this evidence for the claim.

Claim: {claim}
Evidence: {evidence_text}

Return ONLY JSON:
{{"support_strength": <-1.0 to 1.0>, "reasoning": "<brief>"}}

Where:
1.0 = strongly supports
0.0 = neutral
-1.0 = strongly contradicts"""

        try:
            response = await self.llm.call_json(prompt)
            return float(response.get("support_strength", 0.0))
        except Exception as e:
            logger.error(f"Evidence analysis failed: {e}")
            return 0.0


class LLMClaimClassifier:
    """Uses LLM to classify claim types."""

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def classify(self, claim: str) -> str:
        """Classify claim into type."""
        prompt = f"""Classify this claim:

STRUCTURAL: fundamental facts (Earth orbits Sun)
EMPIRICAL: scientific findings (study results)
DYNAMIC: current events (today's news)
NORMATIVE: rules/norms (ethical statements)

Claim: {claim}

Return ONLY JSON:
{{"type": "<STRUCTURAL|EMPIRICAL|DYNAMIC|NORMATIVE>", "reasoning": "<brief>"}}"""

        try:
            response = await self.llm.call_json(prompt)
            claim_type = response.get("type", "EMPIRICAL").upper()
            
            valid = ["STRUCTURAL", "EMPIRICAL", "DYNAMIC", "NORMATIVE"]
            return claim_type if claim_type in valid else "EMPIRICAL"
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "EMPIRICAL"
