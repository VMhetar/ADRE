# ADRE - Autonomous Data Reality Engine

Two-layer epistemic validation system that transforms raw data into training-ready knowledge.

## What It Does

**Layer 0: Raw Reality Intake**
- Extracts claims from unstructured text
- Everything starts untrusted (low confidence)

**Layer 1: Reality Confrontation**
- Validates claims against multiple sources
- Computes epistemic confidence
- Accepts/rejects based on evidence threshold

## File Structure

```
adre/
├── adre_core_structures.py      # Data types and enums
├── adre_epistemic_engine.py     # Confidence computation
├── adre_claim_extractor.py      # Claim extraction
├── adre_reality_validator.py    # Evidence gathering
├── adre_belief_manager.py       # Belief lifecycle
├── adre_llm_service.py          # OpenRouter integration
├── adre_pipeline.py             # Main orchestration
└── example_usage.py             # Usage examples
```

## Quick Start

```python
import asyncio
from adre_pipeline import ADREPipeline

async def main():
    pipeline = ADREPipeline(min_confidence=0.5)
    
    raw_data = """
    The Earth orbits the Sun.
    Machine learning requires quality training data.
    """
    
    beliefs = await pipeline.process_raw_data(raw_data)
    training_data = pipeline.get_training_data()
    stats = pipeline.get_statistics()

asyncio.run(main())
```

## Installation

```bash
# Core (no dependencies)
python --version  # Requires 3.7+

# For LLM features (optional)
pip install httpx
export OPENROUTER_API_KEY='your-key'
```

## Claim Types & Decay

```python
ClaimType.STRUCTURAL   # Barely decays (365d refresh)
ClaimType.EMPIRICAL    # Slow decay (90d refresh)
ClaimType.DYNAMIC      # Fast decay (7d refresh)
ClaimType.NORMATIVE    # Medium decay (30d refresh)
```

## Belief Status

- **`stable`** (≥0.75): High confidence, train on it
- **`contextual`** (0.5-0.75): Medium confidence, use with context
- **`speculative`** (0.3-0.5): Low confidence, uncertain
- **`rejected`** (<0.3): Don't use

## API

```python
# Initialize
pipeline = ADREPipeline(
    extractor=None,           # ClaimExtractor (default: PatternBased)
    validator=None,           # RealityValidator (default: Mock)
    belief_manager=None,      # BeliefManager
    min_confidence=0.5        # Acceptance threshold
)

# Process
beliefs = await pipeline.process_raw_data(raw_data)

# Export
training_data = pipeline.get_training_data()
json_data = pipeline.get_training_data_json()
stats = pipeline.get_statistics()
results = pipeline.export_results()
```

## Configuration Examples

```python
# Pattern-based extraction (fast)
from adre_claim_extractor import PatternBasedExtractor
pipeline = ADREPipeline(extractor=PatternBasedExtractor())

# LLM-based extraction (requires OpenRouter)
from adre_claim_extractor import LLMBasedExtractor
pipeline = ADREPipeline(extractor=LLMBasedExtractor())

# Multi-source validation
from adre_reality_validator import MultiSourceValidator
pipeline = ADREPipeline(validator=MultiSourceValidator())

# Custom threshold
pipeline = ADREPipeline(min_confidence=0.7)  # Strict
```

## Testing

```bash
python example_usage.py
```

## Current State

**Working:**
- Pattern-based claim extraction
- Mock validation (for testing)
- Confidence computation
- Belief lifecycle management
- OpenRouter LLM integration

**Placeholders:**
- WebSearchValidator (needs implementation)
- APIValidator (needs implementation)

## License

Experimental research infrastructure.
