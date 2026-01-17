"""
ADRE Example Usage and Testing
Demonstrates complete pipeline functionality
"""

import asyncio
import json
import os
from datetime import datetime

from adre_pipeline import ADREPipeline
from adre_claim_extractor import PatternBasedExtractor, HybridExtractor, LLMBasedExtractor
from adre_reality_validator import MockValidator, MultiSourceValidator
from adre_belief_manager import BeliefManager
from adre_llm_service import OpenRouterLLMService, MockLLMService


# ============================================================================
# EXAMPLE 1: Simple Pipeline with Default Components
# ============================================================================

async def example_basic():
    """Basic example with default configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pipeline")
    print("="*70)
    
    # Initialize with defaults
    pipeline = ADREPipeline(min_confidence=0.3)
    
    raw_data = """
    The Earth orbits around the Sun. This is a fundamental fact of astronomy.
    
    Today's stock market closed with a 2% gain. Several analysts predict continued growth.
    
    Machine learning models require high-quality training data. Research demonstrates this clearly.
    
    COVID-19 emerged in late 2019. The pandemic affected global society profoundly.
    """
    
    # Process data
    beliefs = await pipeline.process_raw_data(raw_data)
    
    # Export results
    results = pipeline.export_results()
    
    print("\nExported Results:")
    print(json.dumps({
        'total': results['total_processed'],
        'by_status': results['by_status'],
        'stats': results['statistics']
    }, indent=2))


# ============================================================================
# EXAMPLE 2: Custom Configuration
# ============================================================================

async def example_custom():
    """Example with custom extractors and validators."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)
    
    # Use custom components
    extractor = PatternBasedExtractor()
    validator = MultiSourceValidator()  # Combines multiple validators
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
    
    # Get specific data formats
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
    
    # Analyze by status
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
    
    # Statistics
    stats = pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Average Confidence: {stats['average_confidence']:.3f}")
    print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
    print(f"  Avg Evidence: {stats['average_evidence_count']}")


# ============================================================================
# EXAMPLE 4: Batch Processing
# ============================================================================

async def example_batch():
    """Example processing multiple data sources."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    pipeline = ADREPipeline(min_confidence=0.4)
    
    # Multiple documents
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
    
    # Summary
    total_claims = sum(r['total_processed'] for r in all_results)
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
    
    # Export as JSON
    training_json = pipeline.get_training_data_json(pretty=True)
    
    print("\nTraining Data (JSON):")
    print(training_json)
    
    # Export complete results
    results = pipeline.export_results()
    print(f"\nComplete Results Export:")
    print(f"  Timestamp: {results['timestamp']}")
    print(f"  Total Processed: {results['total_processed']}")
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
    
    # Show history for each belief
    print("\nBelief History:")
    for belief in beliefs:
        print(f"\nClaim: {belief.claim.text[:50]}...")
        print(f"Final Status: {belief.status}")
        print(f"Final Confidence: {belief.confidence:.3f}")
        
        if belief.history:
            print(f"History ({len(belief.history)} events):")
            for event in belief.history[-3:]:  # Show last 3 events
                print(f"  • {event['event']}: confidence={event['confidence']:.3f}")


# ============================================================================
# ============================================================================

async def example_openrouter():
    """Example using OpenRouter for claim extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 7: OpenRouter LLM Integration")
    print("="*70)
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️  OPENROUTER_API_KEY not set")
        print("   Set it to use real LLM extraction:")
        print("   export OPENROUTER_API_KEY='sk-...'")
        print("\n   Using Mock LLM for demonstration instead...\n")
        
        # Use mock for demo
        from adre_claim_extractor import PatternBasedExtractor
        extractor = PatternBasedExtractor()
    else:
        # Use real OpenRouter LLM
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
    
    print(f"\n✓ Processed with LLM-based extraction")
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
    
    # Pattern-based
    print("\n[PATTERN-BASED]")
    pipeline1 = ADREPipeline(
        extractor=PatternBasedExtractor(),
        validator=MockValidator(),
        min_confidence=0.3
    )
    beliefs1 = await pipeline1.process_raw_data(raw_data, verbose=False)
    stats1 = pipeline1.get_statistics()
    
    print(f"Claims extracted: {stats1['total_processed']}")
    print(f"Avg confidence: {stats1['average_confidence']:.3f}")
    
    # LLM-based (if available)
    if os.getenv("OPENROUTER_API_KEY"):
        print("\n[LLM-BASED]")
        pipeline2 = ADREPipeline(
            extractor=LLMBasedExtractor(),
            validator=MockValidator(),
            min_confidence=0.3
        )
        beliefs2 = await pipeline2.process_raw_data(raw_data, verbose=False)
        stats2 = pipeline2.get_statistics()
        
        print(f"Claims extracted: {stats2['total_processed']}")
        print(f"Avg confidence: {stats2['average_confidence']:.3f}")
    else:
        print("\n⚠️  OpenRouter API key not set, skipping LLM comparison")


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ADRE PIPELINE - EXAMPLES AND DEMONSTRATIONS")
    print("="*80)
    
    # Run examples
    await example_basic()
    await example_custom()
    await example_analysis()
    await example_batch()
    await example_export()
    await example_history()
    await example_openrouter()
    await example_comparison()
    
    print("\n" + "="*80)
    print("✓ All examples completed successfully")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())