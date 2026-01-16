# ADRE - Autonomous Data Reality Engine

Complete implementation of ADRE as separate, modular Python files.

## File Structure

```
adre/
├── adre_core_structures.py      # Core data types and enums
├── adre_epistemic_engine.py     # Confidence computation
├── adre_claim_extractor.py      # Claim extraction from raw data
├── adre_reality_validator.py    # Reality validation and evidence gathering
├── adre_belief_manager.py       # Belief state management
├── adre_pipeline.py             # Main orchestration
├── example_usage.py             # Usage examples and tests
└── ADRE_SETUP.md               # This file
```

## Overview

ADRE operates through two epistemic layers:

### Layer 0: Raw Reality Intake
- Extracts claims from unstructured data
- All incoming data starts with low confidence
- No data is trusted by default

### Layer 1: Reality Confrontation
- Validates claims against multiple sources
- Gathers evidence (supporting or contradicting)
- Computes epistemic confidence
- Makes decisions based on confidence thresholds

## Core Components

### 1. `adre_core_structures.py`
Defines fundamental data types:
- `ClaimType`: STRUCTURAL, EMPIRICAL, DYNAMIC, NORMATIVE
- `Evidence`: Supporting or contradicting evidence from sources
- `Claim`: Extracted claim requiring validation
- `BeliefState`: Complete epistemic state of a claim

### 2. `adre_epistemic_engine.py`
Mathematical functions for confidence computation:
- `evidence_support()`: Weighted support from evidence
- `contradiction_penalty()`: Penalizes conflicting evidence
- `source_diversity_bonus()`: Rewards diverse sources
- `time_decay()`: Exponential decay of older claims
- `compute_confidence()`: Main confidence formula
- `belief_status()`: Maps confidence to status

### 3. `adre_claim_extractor.py`
Extracts claims from raw data:
- `PatternBasedExtractor`: Fast pattern matching
- `LLMBasedExtractor`: Uses language models
- `HybridExtractor`: Combines pattern + LLM

### 4. `adre_reality_validator.py`
Validates claims against reality:
- `MockValidator`: Simulation (for testing)
- `WebSearchValidator`: Web search integration
- `APIValidator`: Query data APIs
- `MultiSourceValidator`: Combines all validators

### 5. `adre_belief_manager.py`
Manages belief lifecycles:
- Create, update, refresh beliefs
- Track confidence and status changes
- Maintain history of updates
- Upgrade/downgrade based on evidence

### 6. `adre_pipeline.py`
Main orchestration:
- Coordinates all components
- Processes raw data end-to-end
- Exports training-ready data
- Generates statistics and reports

## Installation

```bash
# No external dependencies required
# Uses only Python standard library

python --version  # Requires Python 3.7+
```

## Quick Start

```python
import asyncio
from adre_pipeline import ADREPipeline

async def main():
    # Initialize pipeline
    pipeline = ADREPipeline(min_confidence=0.5)
    
    # Raw data
    raw_data = """
    The Earth orbits the Sun. This is well-established fact.
    Machine learning requires quality training data.
    """
    
    # Process
    beliefs = await pipeline.process_raw_data(raw_data)
    
    # Get training-ready data
    training_data = pipeline.get_training_data()
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")

asyncio.run(main())
```

## Usage Examples

### Example 1: Basic Processing
```python
pipeline = ADREPipeline()
beliefs = await pipeline.process_raw_data(raw_data)
```

### Example 2: Custom Configuration
```python
from adre_claim_extractor import LLMBasedExtractor
from adre_reality_validator import MultiSourceValidator

pipeline = ADREPipeline(
    extractor=LLMBasedExtractor(api_key="your-key"),
    validator=MultiSourceValidator(),
    min_confidence=0.6
)
```

### Example 3: Export Results
```python
# Get training data
training = pipeline.get_training_data()

# Get JSON
json_data = pipeline.get_training_data_json()

# Get complete export
results = pipeline.export_results()
```

### Example 4: Analyze Results
```python
# By status
stable_beliefs = pipeline.get_beliefs_by_status('stable')

# Statistics
stats = pipeline.get_statistics()
print(f"Average confidence: {stats['average_confidence']}")
```

## API Reference

### ADREPipeline

```python
pipeline = ADREPipeline(
    extractor=None,           # ClaimExtractor instance
    validator=None,           # RealityValidator instance
    belief_manager=None,      # BeliefManager instance
    min_confidence=0.5        # Minimum confidence threshold
)

# Main processing
await pipeline.process_raw_data(
    raw_data: str,
    min_confidence: Optional[float] = None,
    verbose: bool = True
) -> List[BeliefState]

# Get results
pipeline.get_training_data() -> List[Dict]
pipeline.get_training_data_json(pretty=True) -> str
pipeline.get_statistics() -> Dict
pipeline.get_beliefs_by_status(status: str) -> List[BeliefState]
pipeline.export_results() -> Dict
```

### Claim Types

```python
from adre_core_structures import ClaimType, DECAY_RATES, REFRESH_INTERVALS

ClaimType.STRUCTURAL   # Barely decays (365 day refresh)
ClaimType.EMPIRICAL    # Slow decay (90 day refresh)
ClaimType.DYNAMIC      # Fast decay (7 day refresh)
ClaimType.NORMATIVE    # Medium decay (30 day refresh)
```

### Belief Status

- **`stable`**: High confidence (≥0.75). Ready for training.
- **`contextual`**: Medium confidence (0.5-0.75). Use with context.
- **`speculative`**: Low confidence (0.3-0.5). Uncertain.
- **`rejected`**: Very low confidence (<0.3). Don't use.

## Configuration

### Confidence Threshold

```python
# Strict: Only accept very confident claims
pipeline = ADREPipeline(min_confidence=0.7)

# Permissive: Accept less certain claims
pipeline = ADREPipeline(min_confidence=0.3)
```

### Extractors

```python
from adre_claim_extractor import PatternBasedExtractor, LLMBasedExtractor

# Fast pattern-based
extractor = PatternBasedExtractor()

# Sophisticated LLM-based
extractor = LLMBasedExtractor(llm_api_key="key")

# Hybrid approach
extractor = HybridExtractor(llm_api_key="key")
```

### Validators

```python
from adre_reality_validator import MockValidator, WebSearchValidator, APIValidator

# Testing/simulation
validator = MockValidator()

# Real web search
validator = WebSearchValidator(search_api_key="key")

# Query data APIs
validator = APIValidator()

# Combine all
validator = MultiSourceValidator()
```

## Integration with Existing Code

### From belief_manager.py (original)
```python
# All functions now in BeliefManager class
manager = BeliefManager()
manager.create_belief(claim)
manager.update_confidence(belief)
manager.update_status(belief)
manager.refresh_belief(belief)
```

### From belief_math.py (original)
```python
# All functions now in EpistemicEngine class
engine = EpistemicEngine()
engine.compute_confidence(evidence_list, last_verified, claim_type)
engine.belief_status(confidence)
engine.evidence_support(evidence_list)
```

### From belief_reviewer.py (original)
```python
# Available through BeliefManager
beliefs_needing_refresh = manager.get_beliefs_needing_refresh()
```

## Testing

Run the examples:

```bash
python example_usage.py
```

Or run individual examples:

```python
from example_usage import example_basic
asyncio.run(example_basic())
```

## Performance Characteristics

- **Claim Extraction**: O(n) where n = document length
- **Evidence Gathering**: O(m) where m = number of sources
- **Confidence Computation**: O(e) where e = number of evidence items
- **Overall**: O(n + m*e) per document

## Future Enhancements

1. **Real LLM Integration**: Implement actual LLMBasedExtractor
2. **Web Search Integration**: Real WebSearchValidator
3. **API Integration**: Real APIValidator with multiple data sources
4. **Learning Feedback Loops**: Adjust decay rates based on model performance
5. **Distributed Processing**: Parallelize across multiple machines
6. **Caching**: Cache evidence and validation results
7. **Source Credibility Learning**: Track which sources are most reliable

## Known Limitations

- `MockValidator` uses simulated evidence (for testing)
- `WebSearchValidator` not implemented (placeholder)
- `APIValidator` not implemented (placeholder)
- `LLMBasedExtractor` not fully implemented

## Next Steps

To make ADRE production-ready:

1. Implement real `WebSearchValidator` with actual search APIs
2. Integrate `LLMBasedExtractor` with your LLM service
3. Add source credibility tracking and learning
4. Implement feedback loops for continuous improvement
5. Add distributed processing for large-scale data
6. Build monitoring and alerting for pipeline health

## License

This is experimental research infrastructure.

## Contact

For questions or improvements, refer to the original ADRE concept documentation.