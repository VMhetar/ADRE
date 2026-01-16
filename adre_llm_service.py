"""
ADRE LLM Service
Integrates with OpenRouter API for LLM-powered features
"""

import os
import httpx
import logging
import json
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService(ABC):
    """Abstract base for LLM services"""
    
    @abstractmethod
    async def call(self, prompt: str) -> str:
        """
        Make LLM call with prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response text
        """
        pass


class OpenRouterLLMService(LLMService):
    """
    OpenRouter LLM Service
    Integrates with OpenRouter API for LLM calls
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-3.5-turbo",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ):
        """
        Initialize OpenRouter LLM service.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (default: gpt-3.5-turbo)
            base_url: OpenRouter API endpoint
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def call(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Make LLM call via OpenRouter.
        
        Args:
            prompt: Input prompt
            temperature: Creativity level (0.0-1.0)
            
        Returns:
            LLM response text
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=data
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenRouter API error: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    raise Exception(f"API error: {response.status_code}")
                
                result = response.json()
                
                # Extract message content
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception("Unexpected response format from OpenRouter")
        
        except httpx.TimeoutException:
            logger.error("OpenRouter API call timed out")
            raise
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise

    async def call_json(self, prompt: str) -> Dict:
        """
        Make LLM call expecting JSON response.
        
        Args:
            prompt: Input prompt (should ask for JSON)
            
        Returns:
            Parsed JSON response
        """
        response = await self.call(prompt, temperature=0.0)
        
        try:
            # Try to parse JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If response contains JSON, extract it
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
            
            raise ValueError(f"Could not parse JSON from response: {response}")


class MockLLMService(LLMService):
    """
    Mock LLM Service for testing
    Returns predefined responses without API calls
    """

    def __init__(self):
        self.call_count = 0

    async def call(self, prompt: str) -> str:
        """
        Return mock response.
        
        Args:
            prompt: Input prompt (ignored)
            
        Returns:
            Mock response
        """
        self.call_count += 1
        
        if "extract" in prompt.lower():
            return '''[
                {"claim": "The Earth orbits the Sun", "type": "STRUCTURAL", "confidence": 0.95},
                {"claim": "Machine learning requires quality data", "type": "EMPIRICAL", "confidence": 0.85}
            ]'''
        
        return "Mock LLM response for testing purposes"

    async def call_json(self, prompt: str) -> Dict:
        """Return mock JSON response."""
        response = await self.call(prompt)
        return json.loads(response)


# ============================================================================
# LLM-POWERED FUNCTIONS FOR ADRE
# ============================================================================

class LLMClaimExtractor:
    """Uses LLM to extract and structure claims from text"""

    def __init__(self, llm_service: LLMService = None):
        """
        Initialize LLM claim extractor.
        
        Args:
            llm_service: LLMService instance (defaults to OpenRouter)
        """
        if llm_service is None:
            llm_service = OpenRouterLLMService()
        
        self.llm = llm_service

    async def extract_claims(self, text: str) -> List[Dict]:
        """
        Extract claims from text using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted claims
        """
        prompt = f"""Extract all factual claims from the following text.
For each claim, provide:
1. claim: The exact claim text
2. type: One of STRUCTURAL, EMPIRICAL, DYNAMIC, NORMATIVE
3. confidence: Your confidence in extraction (0.0-1.0)

Return ONLY valid JSON array, no other text.

Text:
{text}"""

        try:
            response = await self.llm.call_json(prompt)
            if isinstance(response, list):
                return response
            return [response]
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []


class LLMEvidenceAnalyzer:
    """Uses LLM to analyze and score evidence"""

    def __init__(self, llm_service: LLMService = None):
        """
        Initialize LLM evidence analyzer.
        
        Args:
            llm_service: LLMService instance (defaults to OpenRouter)
        """
        if llm_service is None:
            llm_service = OpenRouterLLMService()
        
        self.llm = llm_service

    async def analyze_evidence(self, claim: str, evidence_text: str) -> float:
        """
        Analyze how much evidence supports claim.
        
        Args:
            claim: The claim being evaluated
            evidence_text: Text containing evidence
            
        Returns:
            Support strength (-1.0 to 1.0)
        """
        prompt = f"""Analyze the following evidence for the given claim.
Determine if the evidence supports, contradicts, or is neutral about the claim.

Claim: {claim}

Evidence: {evidence_text}

Respond with ONLY a JSON object:
{{"support_strength": <number from -1.0 to 1.0>, "reasoning": "<brief explanation>"}}

Where:
- 1.0 = strongly supports
- 0.5 = moderately supports
- 0.0 = neutral
- -0.5 = moderately contradicts
- -1.0 = strongly contradicts"""

        try:
            response = await self.llm.call_json(prompt)
            return float(response.get("support_strength", 0.0))
        except Exception as e:
            logger.error(f"Evidence analysis failed: {e}")
            return 0.0


class LLMClaimClassifier:
    """Uses LLM to classify claim types"""

    def __init__(self, llm_service: LLMService = None):
        """
        Initialize LLM claim classifier.
        
        Args:
            llm_service: LLMService instance (defaults to OpenRouter)
        """
        if llm_service is None:
            llm_service = OpenRouterLLMService()
        
        self.llm = llm_service

    async def classify(self, claim: str) -> str:
        """
        Classify claim into type.
        
        Args:
            claim: Claim text to classify
            
        Returns:
            Claim type: STRUCTURAL, EMPIRICAL, DYNAMIC, or NORMATIVE
        """
        prompt = f"""Classify this claim into one of four types:

STRUCTURAL: Fundamental facts that barely change (e.g., Earth orbits Sun)
EMPIRICAL: Scientific findings from research (e.g., study results)
DYNAMIC: Current events/news that change frequently (e.g., today's weather)
NORMATIVE: Rules, norms, or policies (e.g., ethical statements)

Claim: {claim}

Respond with ONLY a JSON object:
{{"type": "<STRUCTURAL|EMPIRICAL|DYNAMIC|NORMATIVE>", "reasoning": "<brief>"}}"""

        try:
            response = await self.llm.call_json(prompt)
            claim_type = response.get("type", "EMPIRICAL").upper()
            
            # Validate
            valid_types = ["STRUCTURAL", "EMPIRICAL", "DYNAMIC", "NORMATIVE"]
            if claim_type not in valid_types:
                claim_type = "EMPIRICAL"
            
            return claim_type
        except Exception as e:
            logger.error(f"Claim classification failed: {e}")
            return "EMPIRICAL"  # Default


# ============================================================================
# INTEGRATION WITH ADRE
# ============================================================================

async def get_llm_service(
    use_openrouter: bool = True,
    api_key: Optional[str] = None,
    model: str = "openai/gpt-3.5-turbo"
) -> LLMService:
    """
    Get configured LLM service.
    
    Args:
        use_openrouter: Use OpenRouter (True) or Mock (False)
        api_key: OpenRouter API key
        model: Model to use
        
    Returns:
        Configured LLMService instance
    """
    if use_openrouter:
        return OpenRouterLLMService(api_key=api_key, model=model)
    else:
        return MockLLMService()