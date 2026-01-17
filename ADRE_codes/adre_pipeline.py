"""
ADRE Main Pipeline
Orchestrates the complete data validation workflow
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
import json
import asyncio

from adre_core_structures import BeliefState, Claim
from adre_claim_extractor import ClaimExtractor, PatternBasedExtractor
from adre_reality_validator import RealityValidator, MockValidator
from adre_belief_manager import BeliefManager
from adre_epistemic_engine import EpistemicEngine


class ADREPipeline:
    """
    Main orchestration engine for ADRE.
    Coordinates claim extraction, reality validation, and belief management.
    """

    def __init__(
        self,
        extractor: ClaimExtractor = None,
        validator: RealityValidator = None,
        belief_manager: BeliefManager = None,
        min_confidence: float = 0.5
    ):
        """
        Initialize ADRE pipeline.
        
        Args:
            extractor: Claim extraction strategy (default: PatternBasedExtractor)
            validator: Reality validation strategy (default: MockValidator)
            belief_manager: Belief management engine (default: new BeliefManager)
            min_confidence: Minimum confidence threshold for acceptance
        """
        self.extractor = extractor or PatternBasedExtractor()
        self.validator = validator or MockValidator()
        self.belief_manager = belief_manager or BeliefManager()
        self.min_confidence = min_confidence
        
        # Processing results
        self.processed_beliefs: List[BeliefState] = []
        self.pipeline_stats = {}

    async def process_raw_data(
        self,
        raw_data: str,
        min_confidence: Optional[float] = None,
        verbose: bool = True
    ) -> List[BeliefState]:
        """
        Main pipeline: raw_data → validated knowledge
        
        Layer 0 (Raw Reality Intake):
            - Extract claims from raw data
            - All claims start with low confidence
        
        Layer 1 (Hypothesis Injection):
            - Validate claims against reality sources
            - Generate evidence
            - Compute confidence
        
        Decision Mechanism:
            - Accept (confidence >= threshold): Send to training
            - Reject (confidence < threshold): Discard
        
        Args:
            raw_data: Unstructured text input
            min_confidence: Override minimum confidence threshold
            verbose: Print processing details
            
        Returns:
            List of validated BeliefState objects
        """
        if min_confidence:
            self.min_confidence = min_confidence
        
        if verbose:
            self._print_header("ADRE PIPELINE: Processing Raw Data")
        
        # ====================================================================
        # LAYER 0: Claim Extraction (Raw Reality Intake)
        # ====================================================================
        if verbose:
            print("\n[LAYER 0] CLAIM EXTRACTION")
            print("-" * 70)
        
        claims = await self.extractor.extract(raw_data)
        
        if verbose:
            print(f"Extracted {len(claims)} claims from raw data\n")
        
        # ====================================================================
        # LAYER 1: Reality Validation & Belief Computation
        # ====================================================================
        validated_beliefs = []
        
        for idx, claim in enumerate(claims, 1):
            if verbose:
                print(f"[{idx}/{len(claims)}] Processing Claim")
                print(f"    Text: {claim.text[:65]}...")
                print(f"    Type: {claim.claim_type.value}")
            
            # Create belief state
            belief = self.belief_manager.create_belief(claim)
            
            # Validate claim against reality
            if verbose:
                print(f"    → Validating against reality sources...")
            
            evidence = await self.validator.validate(claim)
            belief = self.belief_manager.update_evidence(belief, evidence)
            
            if verbose:
                print(f"    → Evidence collected: {len(evidence)} sources")
                print(f"    → Contradictions: {belief.contradiction_count}")
                print(f"    → Source diversity: {belief.source_diversity:.2f}")
                print(f"    → Confidence: {belief.confidence:.3f}")
                print(f"    → Status: {belief.status}")
            
            # ================================================================
            # DECISION MECHANISM
            # ================================================================
            if belief.confidence >= self.min_confidence:
                if verbose:
                    print(f"    ✓ ACCEPTED - Added to validated knowledge")
                validated_beliefs.append(belief)
            else:
                if verbose:
                    print(f"    ✗ REJECTED - Below confidence threshold ({self.min_confidence})")
            
            if verbose:
                print()
        
        # Store results
        self.processed_beliefs = validated_beliefs
        self._compute_statistics()
        
        if verbose:
            self._print_results()
        
        return validated_beliefs

    def get_training_data(self) -> List[Dict]:
        """
        Export validated beliefs as training data.
        Only 'stable' and 'contextual' beliefs are included.
        
        Returns:
            List of training-ready data dictionaries
        """
        training_data = []
        
        for belief in self.processed_beliefs:
            if belief.status in ["stable", "contextual"]:
                training_item = belief.to_training_format()
                if training_item:
                    training_data.append(training_item)
        
        return training_data

    def get_training_data_json(self, pretty: bool = True) -> str:
        """
        Export training data as JSON.
        
        Args:
            pretty: Pretty-print JSON (default: True)
            
        Returns:
            JSON string of training data
        """
        data = self.get_training_data()
        indent = 2 if pretty else None
        return json.dumps(data, indent=indent)

    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.pipeline_stats

    def get_beliefs_by_status(self, status: str) -> List[BeliefState]:
        """
        Get all beliefs with specific status.
        
        Args:
            status: One of 'rejected', 'speculative', 'contextual', 'stable'
            
        Returns:
            List of BeliefState objects
        """
        return [b for b in self.processed_beliefs if b.status == status]

    def export_results(self) -> Dict:
        """
        Export complete results including beliefs and statistics.
        
        Returns:
            Dictionary containing all results
        """
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_processed': len(self.processed_beliefs),
            'statistics': self.get_statistics(),
            'training_data': self.get_training_data(),
            'belief_details': [b.to_dict() for b in self.processed_beliefs],
            'by_status': {
                'stable': len(self.get_beliefs_by_status('stable')),
                'contextual': len(self.get_beliefs_by_status('contextual')),
                'speculative': len(self.get_beliefs_by_status('speculative')),
                'rejected': len(self.get_beliefs_by_status('rejected')),
            }
        }

    def _compute_statistics(self) -> None:
        """Compute and store pipeline statistics."""
        total = len(self.processed_beliefs)
        
        if total == 0:
            self.pipeline_stats = {
                'total_processed': 0,
                'stable': 0,
                'contextual': 0,
                'speculative': 0,
                'rejected': 0,
                'average_confidence': 0.0,
                'acceptance_rate': 0.0,
                'training_ready': 0,
                'average_evidence_count': 0
            }
            return
        
        stable = len(self.get_beliefs_by_status('stable'))
        contextual = len(self.get_beliefs_by_status('contextual'))
        speculative = len(self.get_beliefs_by_status('speculative'))
        rejected = len(self.get_beliefs_by_status('rejected'))
        
        avg_confidence = sum(b.confidence for b in self.processed_beliefs) / total
        training_ready = stable + contextual
        avg_evidence = sum(len(b.evidence) for b in self.processed_beliefs) / total
        
        self.pipeline_stats = {
            'total_processed': total,
            'stable': stable,
            'contextual': contextual,
            'speculative': speculative,
            'rejected': rejected,
            'average_confidence': round(avg_confidence, 3),
            'acceptance_rate': round(training_ready / total, 3),
            'training_ready': training_ready,
            'average_evidence_count': round(avg_evidence, 1)
        }

    def _print_header(self, text: str) -> None:
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(text)
        print("=" * 70)

    def _print_results(self) -> None:
        """Print formatted results."""
        self._print_header("RESULTS")
        
        stats = self.get_statistics()
        
        print(f"\nProcessing Summary:")
        print(f"  Total Claims Processed: {stats['total_processed']}")
        print(f"  Stable:      {stats['stable']}")
        print(f"  Contextual:  {stats['contextual']}")
        print(f"  Speculative: {stats['speculative']}")
        print(f"  Rejected:    {stats['rejected']}")
        
        print(f"\nQuality Metrics:")
        print(f"  Average Confidence: {stats['average_confidence']:.3f}")
        print(f"  Acceptance Rate:    {stats['acceptance_rate']:.1%}")
        print(f"  Training-Ready:     {stats['training_ready']}")
        print(f"  Avg Evidence Count: {stats['average_evidence_count']}")
        
        print(f"\nTraining Data Ready:")
        training = self.get_training_data()
        print(f"  Items: {len(training)}")
        
        if training:
            print(f"\n  Sample items:")
            for item in training[:3]:
                print(f"    • {item['claim'][:55]}...")
                print(f"      Confidence: {item['confidence']:.2f}, Sources: {len(item['sources'])}")
        
        print("\n" + "=" * 70)