"""
ADRE Example Usage and Testing (CORRECTED VERSION)
Demonstrates complete pipeline functionality with error handling
"""

import asyncio
import json
import os
from typing import List, Dict, Any

from adre_pipeline import ADREPipeline
from adre_claim_extractor import PatternBasedExtractor, LLMBasedExtractor
from adre_reality_validator import MockValidator, MultiSourceValidator
from adre_belief_manager import BeliefManager


# ============================================================================
# EXAMPLE 1: Simple Pipeline with Default Components
# ============================================================================

async def example_basic():
    """Basic example with default configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pipeline")
    print("="*70)
    
    pipeline = ADREPipeline(min_confidence=0.3)
    
    raw_data = """
    The Earth orbits around the Sun. This is a fundamental fact of astronomy.
    
    Today's stock market closed with a 2% gain. Several analysts predict continued growth.
    
    Machine learning models require high-quality training data. Research demonstrates this clearly.
    
    COVID-19 emerged in late 2019. The pandemic affected global society profoundly.
    """
    
    beliefs = await pipeline.process_raw_data(raw_data)
    results = pipeline.export_results()
    
    print("\nExported Results:")
    print(json.dumps({
        'total': results['total'],
        'by_status': results['by_status'],
        'stats': results['stats']
    }, indent=2))


# ============================================================================
# EXAMPLE 2: Custom Configuration
# ============================================================================

async def example_custom():
    """Example with custom extractors and validators."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)
    
    extractor = PatternBasedExtractor()
    validator = MultiSourceValidator()
    belief_manager = BeliefManager()
    
    pipeline = ADREPipeline(
        extractor=extractor,
        validator=validator,
        belief_manager=belief_manager,
        min_confidence=0.5
    )
    
    raw_data = """
    Climate change is affecting global weather patterns. Multiple studies confirm this trend.
    
    Artificial intelligence is advancing rapidly. Companies are investing heavily in AI research.
    
    The internet has transformed human communication. This change occurred over two decades.
    """
    
    beliefs = await pipeline.process_raw_data(raw_data, verbose=True)
    
    training_data = pipeline.get_training_data()
    print(f"\n✓ Generated {len(training_data)} training-ready items")


# ============================================================================
# EXAMPLE 3: Advanced Analysis
# ============================================================================

async def example_analysis():
    """Example with detailed analysis of results."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Advanced Analysis")
    print("="*70)
    
    pipeline = ADREPipeline(min_confidence=0.4)
    
    raw_data = """
    Python is a popular programming language used widely in data science.
    
    The moon orbits Earth every 27.3 days according to astronomical measurements.
    
    Social media platforms are changing how people communicate globally.
    
    Renewable energy sources are becoming more cost-effective than fossil fuels.
    
    Quantum computing remains largely experimental at this stage.
    """
    
    beliefs = await pipeline.process_raw_data(raw_data, verbose=False)
    
    print("\nAnalysis by Status:")
    for status in ['stable', 'contextual', 'speculative', 'rejected']:
        beliefs_with_status = pipeline.get_beliefs_by_status(status)
        if beliefs_with_status:
            print(f"\n{status.upper()} ({len(beliefs_with_status)}):")
            for belief in beliefs_with_status:
                print(f"  • Confidence: {belief.confidence:.3f}")
                print(f"    Claim: {belief.claim.text[:60]}...")
                print(f"    Evidence: {len(belief.evidence)} sources, "
                      f"{belief.contradiction_count} contradictions")
    
    stats = pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Average Confidence: {stats['avg_confidence']:.3f}")
    print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
    print(f"  Avg Evidence: {stats['avg_evidence']}")


# ============================================================================
# EXAMPLE 4: Batch Processing
# ============================================================================

async def example_batch():
    """Example processing multiple data sources."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    pipeline = ADREPipeline(min_confidence=0.4)
    
    documents = [
        "Python was created by Guido van Rossum in 1991. It has become very popular.",
        "The Great Wall of China is one of the most impressive structures ever built.",
        "Climate scientists predict temperatures will continue rising this century.",
    ]
    
    all_results = []
    
    for idx, doc in enumerate(documents, 1):
        print(f"\nProcessing Document {idx}/{len(documents)}...")
        beliefs = await pipeline.process_raw_data(doc, verbose=False)
        results = pipeline.export_results()
        all_results.append(results)
    
    total_claims = sum(r['total'] for r in all_results)
    total_training = sum(len(r['training_data']) for r in all_results)
    
    print(f"\n✓ Processed {len(documents)} documents")
    print(f"✓ Extracted {total_claims} claims")
    print(f"✓ Generated {total_training} training-ready items")


# ============================================================================
# EXAMPLE 5: JSON Export
# ============================================================================

async def example_export():
    """Example exporting results as JSON."""
    print("\n" + "="*70)
    print("EXAMPLE 5: JSON Export")
    print("="*70)
    
    pipeline = ADREPipeline(min_confidence=0.3)
    
    raw_data = """
    The internet revolutionized global communication and commerce.
    
    Machine learning is a subset of artificial intelligence.
    
    Climate change poses significant challenges to ecosystems worldwide.
    """
    
    await pipeline.process_raw_data(raw_data, verbose=False)
    
    training_json = pipeline.get_training_data_json(pretty=True)
    
    print("\nTraining Data (JSON):")
    print(training_json)
    
    results = pipeline.export_results()
    print(f"\nComplete Results Export:")
    print(f"  Timestamp: {results['timestamp']}")
    print(f"  Total Processed: {results['total']}")
    print(f"  Status Breakdown: {results['by_status']}")


# ============================================================================
# EXAMPLE 6: Belief History Tracking
# ============================================================================

async def example_history():
    """Example showing belief history tracking."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Belief History Tracking")
    print("="*70)
    
    belief_manager = BeliefManager()
    pipeline = ADREPipeline(belief_manager=belief_manager, min_confidence=0.2)
    
    raw_data = """
    Renewable energy adoption is accelerating globally.
    Electric vehicles are becoming more affordable and available.
    """
    
    beliefs = await pipeline.process_raw_data(raw_data, verbose=False)
    
    print("\nBelief History:")
    for belief in beliefs:
        print(f"\nClaim: {belief.claim.text[:50]}...")
        print(f"Final Status: {belief.status}")
        print(f"Final Confidence: {belief.confidence:.3f}")
        
        if belief.history:
            print(f"History ({len(belief.history)} events):")
            for event in belief.history[-3:]:
                print(f"  • {event['event']}: conf={event['conf']:.3f}")


# ============================================================================
# EXAMPLE 7: OpenRouter LLM Integration
# ============================================================================

async def example_openrouter():
    """Example using OpenRouter for claim extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 7: OpenRouter LLM Integration")
    print("="*70)
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️  OPENROUTER_API_KEY not set")
        print("   Set it to use real LLM extraction:")
        print("   export OPENROUTER_API_KEY='sk-...'")
        print("\n   Using Pattern extractor for demonstration instead...\n")
        extractor = PatternBasedExtractor()
    else:
        print("\n✓ Using OpenRouter API\n")
        extractor = LLMBasedExtractor(model="openai/gpt-3.5-turbo")
    
    pipeline = ADREPipeline(
        extractor=extractor,
        validator=MockValidator(),
        min_confidence=0.3
    )
    
    raw_data = """
    Artificial intelligence is rapidly advancing. Major breakthroughs occur regularly.
    
    The COVID-19 pandemic has changed how people work globally.
    
    Electric vehicles are becoming more affordable and widespread.
    """
    
    beliefs = await pipeline.process_raw_data(raw_data, verbose=True)
    
    print(f"\n✓ Processed with extraction")
    print(f"✓ Generated {len(pipeline.get_training_data())} training-ready items")


# ============================================================================
# EXAMPLE 8: Comparison - Pattern vs LLM Extraction
# ============================================================================

async def example_comparison():
    """Compare pattern-based vs LLM-based extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Pattern vs LLM Extraction")
    print("="*70)
    
    raw_data = """
    Machine learning models require high-quality training data to perform well.
    Recent studies show that data quality matters more than data quantity.
    The field of AI is evolving rapidly with new techniques emerging monthly.
    """
    
    print("\n[PATTERN-BASED]")
    pipeline1 = ADREPipeline(
        extractor=PatternBasedExtractor(),
        validator=MockValidator(),
        min_confidence=0.3
    )
    beliefs1 = await pipeline1.process_raw_data(raw_data, verbose=False)
    stats1 = pipeline1.get_statistics()
    
    print(f"Claims extracted: {stats1['total']}")
    print(f"Avg confidence: {stats1['avg_confidence']:.3f}")
    
    if os.getenv("OPENROUTER_API_KEY"):
        print("\n[LLM-BASED]")
        pipeline2 = ADREPipeline(
            extractor=LLMBasedExtractor(),
            validator=MockValidator(),
            min_confidence=0.3
        )
        beliefs2 = await pipeline2.process_raw_data(raw_data, verbose=False)
        stats2 = pipeline2.get_statistics()
        
        print(f"Claims extracted: {stats2['total']}")
        print(f"Avg confidence: {stats2['avg_confidence']:.3f}")
    else:
        print("\n⚠️  OpenRouter API key not set, skipping LLM comparison")


# ============================================================================
# EXAMPLE 9: AG News Dataset Integration (Better Choice)
# ============================================================================

async def example_agnews():
    """Example processing AG News dataset with ADRE.
    
    This example demonstrates processing larger datasets and computing
    data reduction metrics with proper error handling for edge cases.
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: AG News Dataset Integration")
    print("="*70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n⚠️  'datasets' library not installed")
        print("   Install it with: pip install datasets")
        print("   Skipping this example...\n")
        return

    pipeline = ADREPipeline(min_confidence=0.6)

    print("\nLoading AG News dataset...")
    
    # AG News: news articles with categories (World, Sports, Business, Sci/Tech)
    # Much better for factual claim extraction than toxic comments
    dataset = load_dataset("ag_news", split="train[:100]")
    
    # Extract text from articles
    texts = [item['text'] for item in dataset]
    raw_text = "\n\n".join(texts)

    print(f"✓ Loaded {len(texts)} news articles")
    print(f"✓ Raw text size: {len(raw_text):,} characters")
    print(f"✓ Raw token estimate: {len(raw_text.split()):,}")

    # Process with ADRE
    beliefs = await pipeline.process_raw_data(raw_text, verbose=False)

    # Statistics
    stats = pipeline.get_statistics()
    training_data = pipeline.get_training_data()

    # Reduction metrics with safety checks
    filtered_chars = sum(len(b.claim.text) for b in beliefs)
    filtered_tokens = sum(len(b.claim.text.split()) for b in beliefs)

    print("\nResults Summary:")
    print(f"  Total Claims Processed: {stats['total']}")
    print(f"  Stable: {stats['stable']}")
    print(f"  Contextual: {stats['contextual']}")
    print(f"  Speculative: {stats['speculative']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Average Confidence: {stats['avg_confidence']:.3f}")
    print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
    print(f"  Training-Ready Items: {len(training_data)}")

    print("\nData Reduction:")
    print(f"  Raw Size (chars): {len(raw_text):,}")
    print(f"  Filtered Size (chars): {filtered_chars:,}")
    
    # FIXED: Added safety check for division by zero
    if len(raw_text) > 0:
        char_ratio = filtered_chars / len(raw_text)
        print(f"  Char Reduction Ratio: {char_ratio:.3f}")
    else:
        print(f"  Char Reduction Ratio: N/A (no raw text)")

    raw_tokens = len(raw_text.split())
    print(f"  Raw Tokens (approx): {raw_tokens:,}")
    print(f"  Filtered Tokens: {filtered_tokens:,}")
    
    # FIXED: Added safety check for division by zero
    if raw_tokens > 0:
        token_ratio = filtered_tokens / raw_tokens
        print(f"  Token Reduction Ratio: {token_ratio:.3f}")
    else:
        print(f"  Token Reduction Ratio: N/A (no tokens)")

    print("\nSample Validated Claims:")
    for belief in beliefs[:5]:
        print(f"  • {belief.claim.text[:80]}...")
        print(f"    Confidence: {belief.confidence:.3f}, "
              f"Evidence: {len(belief.evidence)} sources, "
              f"Status: {belief.status}")

    print("\n✓ AG News example completed successfully")


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def main():
    """Run all examples.
    
    This function executes each example sequentially. Each example is
    independent and can be run individually if needed.
    """
    print("\n" + "="*80)
    print("ADRE PIPELINE - EXAMPLES AND DEMONSTRATIONS")
    print("="*80)
    
    try:
        await example_basic()
    except Exception as e:
        print(f"\n❌ example_basic failed: {e}")
    
    try:
        await example_custom()
    except Exception as e:
        print(f"\n❌ example_custom failed: {e}")
    
    try:
        await example_analysis()
    except Exception as e:
        print(f"\n❌ example_analysis failed: {e}")
    
    try:
        await example_batch()
    except Exception as e:
        print(f"\n❌ example_batch failed: {e}")
    
    try:
        await example_export()
    except Exception as e:
        print(f"\n❌ example_export failed: {e}")
    
    try:
        await example_history()
    except Exception as e:
        print(f"\n❌ example_history failed: {e}")
    
    try:
        await example_openrouter()
    except Exception as e:
        print(f"\n❌ example_openrouter failed: {e}")
    
    try:
        await example_comparison()
    except Exception as e:
        print(f"\n❌ example_comparison failed: {e}")
    
    try:
        await example_agnews()
    except Exception as e:
        print(f"\n❌ example_agnews failed: {e}")
    
    print("\n" + "="*80)
    print("✓ All examples completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())