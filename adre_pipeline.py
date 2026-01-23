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


class PipelinePrinter:
    """Handles all verbose output separately from pipeline logic."""
    
    @staticmethod
    def print_header(text: str) -> None:
        """Print formatted header."""
        print(f"\n{'='*70}")
        print(text)
        print(f"{'='*70}\n")
    
    @staticmethod
    def print_claim_processing(idx: int, total: int, claim: Claim, belief: BeliefState) -> None:
        """Print claim processing details."""
        print(f"[{idx}/{total}] {claim.text[:50]}...")
        print(f"  Evidence: {len(belief.evidence)} | "
              f"Contradictions: {belief.contradiction_count} | "
              f"Confidence: {belief.confidence:.3f} | "
              f"Status: {belief.status}")
    
    @staticmethod
    def print_decision(accepted: bool, threshold: float) -> None:
        """Print accept/reject decision."""
        if accepted:
            print(f"  ✓ ACCEPTED\n")
        else:
            print(f"  ✗ REJECTED (below {threshold})\n")
    
    @staticmethod
    def print_summary(stats: Dict, training_data: List[Dict]) -> None:
        """Print processing summary."""
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        
        print(f"\nProcessed: {stats['total']}")
        print(f"  Stable:      {stats['stable']}")
        print(f"  Contextual:  {stats['contextual']}")
        print(f"  Speculative: {stats['speculative']}")
        print(f"  Rejected:    {stats['rejected']}")
        
        print(f"\nMetrics:")
        print(f"  Avg Confidence:  {stats['avg_confidence']:.3f}")
        print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
        print(f"  Training Ready:  {stats['training_ready']}")
        print(f"  Avg Evidence:    {stats['avg_evidence']}")
        
        if training_data:
            print(f"\nSample Training Data:")
            for item in training_data[:2]:
                print(f"  • {item['claim'][:50]}...")
                print(f"    Conf: {item['confidence']:.2f} | Sources: {len(item['sources'])}")
        
        print(f"\n{'='*70}\n")


class ADREPipeline:
    """
    Main orchestration engine for ADRE.
    Coordinates claim extraction, validation, and belief management.
    """

    def __init__(
        self,
        extractor: ClaimExtractor = None,
        validator: RealityValidator = None,
        belief_manager: BeliefManager = None,
        min_confidence: float = 0.5
    ):
        self.extractor = extractor or PatternBasedExtractor()
        self.validator = validator or MockValidator()
        self.belief_manager = belief_manager or BeliefManager()
        self.min_confidence = min_confidence
        
        self.processed_beliefs: List[BeliefState] = []
        self._stats = None  # Lazy computation
        self.printer = PipelinePrinter()

    async def process_raw_data(
        self,
        raw_data: str,
        min_confidence: Optional[float] = None,
        verbose: bool = True
    ) -> List[BeliefState]:
        """
        Main pipeline: raw_data → validated knowledge
        
        Layer 0: Extract claims (low confidence)
        Layer 1: Validate against reality, compute confidence
        Decision: Accept if confidence >= threshold
        """
        threshold = min_confidence or self.min_confidence
        
        if verbose:
            self.printer.print_header("ADRE PIPELINE")
            print("[LAYER 0] Extracting claims...")
        
        # Layer 0: Claim Extraction (synchronous now)
        claims = self.extractor.extract(raw_data)
        
        if verbose:
            print(f"  → Extracted {len(claims)} claims\n")
            print("[LAYER 1] Validating claims...\n")
        
        if not claims:
            if verbose:
                print("No claims extracted")
            return []
        
        # Layer 1: Validation & Belief Computation
        validated = await self._validate_claims(claims, threshold, verbose)
        
        self.processed_beliefs = validated
        self._stats = None  # Reset cached stats
        
        if verbose:
            stats = self.get_statistics()
            training = self.get_training_data()
            self.printer.print_summary(stats, training)
        
        return validated

    async def _validate_claims(
        self,
        claims: List[Claim],
        threshold: float,
        verbose: bool
    ) -> List[BeliefState]:
        """Validate claims and filter by confidence threshold."""
        validated = []
        
        for idx, claim in enumerate(claims, 1):
            # Create belief
            belief = self.belief_manager.create_belief(claim)
            
            # Gather evidence
            evidence = await self.validator.validate(claim)
            belief = self.belief_manager.update_evidence(belief, evidence)
            
            if verbose:
                self.printer.print_claim_processing(idx, len(claims), claim, belief)
            
            # Decision mechanism
            accepted = belief.confidence >= threshold
            if verbose:
                self.printer.print_decision(accepted, threshold)
            
            if accepted:
                validated.append(belief)
        
        return validated

    def get_training_data(self) -> List[Dict]:
        """Export validated beliefs as training data."""
        training = []
        for belief in self.processed_beliefs:
            if belief.status in ["stable", "contextual"]:
                item = belief.to_training_format()
                if item:
                    training.append(item)
        return training

    def get_training_data_json(self, pretty: bool = True) -> str:
        """Export training data as JSON."""
        data = self.get_training_data()
        indent = 2 if pretty else None
        return json.dumps(data, indent=indent)

    def get_statistics(self) -> Dict:
        """Get pipeline statistics (lazy computation, cached)."""
        if self._stats is None:
            self._stats = self._compute_stats()
        return self._stats

    def get_beliefs_by_status(self, status: str) -> List[BeliefState]:
        """Get beliefs with specific status."""
        return [b for b in self.processed_beliefs if b.status == status]

    def export_results(self) -> Dict:
        """Export complete results."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total': len(self.processed_beliefs),
            'stats': self.get_statistics(),
            'training_data': self.get_training_data(),
            'by_status': {
                'stable': len(self.get_beliefs_by_status('stable')),
                'contextual': len(self.get_beliefs_by_status('contextual')),
                'speculative': len(self.get_beliefs_by_status('speculative')),
                'rejected': len(self.get_beliefs_by_status('rejected')),
            }
        }

    def _compute_stats(self) -> Dict:
        """Compute pipeline statistics."""
        total = len(self.processed_beliefs)
        
        if total == 0:
            return {
                'total': 0,
                'stable': 0,
                'contextual': 0,
                'speculative': 0,
                'rejected': 0,
                'avg_confidence': 0.0,
                'acceptance_rate': 0.0,
                'training_ready': 0,
                'avg_evidence': 0
            }
        
        stable = len(self.get_beliefs_by_status('stable'))
        contextual = len(self.get_beliefs_by_status('contextual'))
        speculative = len(self.get_beliefs_by_status('speculative'))
        rejected = len(self.get_beliefs_by_status('rejected'))
        
        avg_conf = sum(b.confidence for b in self.processed_beliefs) / total
        training_ready = stable + contextual
        avg_evidence = sum(len(b.evidence) for b in self.processed_beliefs) / total
        
        return {
            'total': total,
            'stable': stable,
            'contextual': contextual,
            'speculative': speculative,
            'rejected': rejected,
            'avg_confidence': round(avg_conf, 3),
            'acceptance_rate': round(training_ready / total, 3),
            'training_ready': training_ready,
            'avg_evidence': round(avg_evidence, 1)
        }
