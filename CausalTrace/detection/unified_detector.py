"""
Unified Attack Detector - combines semantic and causal detection.

This module provides a unified pipeline that intelligently combines:
1. SemanticInjectionDetector (GPT-4o-mini) - for single-turn jailbreaks (B3-style)
2. CausalEnsembleDetector - for multi-step exfiltration attacks (WASP-style)

Different attack types require different detection approaches:
- B3/ASB attacks: Single-turn social engineering → semantic understanding
- WASP/Toucan attacks: Multi-step data exfiltration → causal graph analysis

Runs both in parallel and fuses results.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from causaltrace.models.trajectory import Trajectory
from causaltrace.graph import GraphBuilder, CausalGraph
from .detector import BaseDetector, DetectionResult
from .semantic_detector import SemanticInjectionDetector

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDetectionResult(DetectionResult):
    """Extended result with both semantic and causal analysis."""

    # Semantic detection results
    semantic_is_attack: bool = False
    semantic_confidence: float = 0.0
    semantic_details: Dict[str, Any] = field(default_factory=dict)

    # Causal detection results
    causal_is_attack: bool = False
    causal_confidence: float = 0.0
    causal_details: Dict[str, Any] = field(default_factory=dict)

    # Fusion info
    fusion_method: str = "weighted"
    attack_type: str = "unknown"  # "jailbreak", "exfiltration", "hybrid", "benign"


class UnifiedAttackDetector(BaseDetector):
    """
    Unified detector combining semantic and causal analysis.

    Detection Strategy:
    ------------------
    1. Run SemanticInjectionDetector on trajectory observations
    2. Build causal graph and run EnsembleDetector
    3. Fuse results using configurable strategy

    Fusion Logic:
    ------------
    - If semantic confidence > 0.9: Early exit as attack (jailbreak)
    - If causal confidence > 0.8: Attack (exfiltration)
    - If both agree: High confidence
    - If disagree: Use weighted combination

    Usage:
    ------
    ```python
    detector = UnifiedAttackDetector()
    result = detector.detect(trajectory)
    print(f"Attack: {result.is_attack} ({result.attack_type})")
    print(f"Semantic: {result.semantic_confidence:.0%}")
    print(f"Causal: {result.causal_confidence:.0%}")
    ```
    """

    def __init__(
        self,
        # Semantic detector config
        use_semantic: bool = True,
        semantic_model: str = "gpt-4o-mini",
        semantic_threshold: float = 0.7,
        semantic_early_exit_threshold: float = 0.9,

        # Causal detector config
        use_causal: bool = True,
        use_counterfactual: bool = True,
        use_mechanism: bool = True,
        use_probabilistic: bool = True,
        causal_threshold: float = 0.5,

        # Fusion config
        fusion_method: str = "weighted",  # "weighted", "max", "semantic_first", "causal_first"
        semantic_weight: float = 0.6,
        causal_weight: float = 0.4,

        # General config
        verbose: bool = False,
    ):
        """
        Initialize unified detector.

        Args:
            use_semantic: Enable semantic (LLM) detection
            semantic_model: OpenAI model for semantic detection
            semantic_threshold: Min confidence to flag semantic attack
            semantic_early_exit_threshold: Confidence for early exit (skip causal)

            use_causal: Enable causal graph detection
            use_counterfactual: Include counterfactual analysis
            use_mechanism: Include mechanism integrity
            use_probabilistic: Include probabilistic CI
            causal_threshold: Min confidence to flag causal attack

            fusion_method: How to combine results
            semantic_weight: Weight for semantic in fusion
            causal_weight: Weight for causal in fusion

            verbose: Print detailed analysis
        """
        self.use_semantic = use_semantic
        self.semantic_model = semantic_model
        self.semantic_threshold = semantic_threshold
        self.semantic_early_exit_threshold = semantic_early_exit_threshold

        self.use_causal = use_causal
        self.use_counterfactual = use_counterfactual
        self.use_mechanism = use_mechanism
        self.use_probabilistic = use_probabilistic
        self.causal_threshold = causal_threshold

        self.fusion_method = fusion_method
        self.semantic_weight = semantic_weight
        self.causal_weight = causal_weight

        self.verbose = verbose

        # Initialize components
        self.semantic_detector = None
        self.causal_detector = None
        self.graph_builder = GraphBuilder()

        # Statistics
        self._stats = {
            'total_detections': 0,
            'semantic_attacks': 0,
            'causal_attacks': 0,
            'both_attacks': 0,
            'disagreements': 0,
        }

    def _init_semantic_detector(self):
        """Lazy init semantic detector."""
        if self.semantic_detector is None and self.use_semantic:
            self.semantic_detector = SemanticInjectionDetector(
                model=self.semantic_model,
                num_few_shot=5,
                temperature=0.0,
            )

    def _init_causal_detector(self):
        """Lazy init causal detector."""
        if self.causal_detector is None and self.use_causal:
            from .mechanism_detector import EnsembleDetector
            self.causal_detector = EnsembleDetector(
                use_counterfactual=self.use_counterfactual,
                use_mechanism=self.use_mechanism,
                use_probabilistic=self.use_probabilistic,
                use_ml=False,
                voting="weighted",
                verbose=self.verbose,
            )

    def detect(self, trajectory: Trajectory) -> UnifiedDetectionResult:
        """
        Detect attack using unified semantic + causal analysis.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            UnifiedDetectionResult with comprehensive analysis
        """
        self._stats['total_detections'] += 1

        semantic_result = None
        causal_result = None

        # Step 1: Semantic detection (on trajectory)
        if self.use_semantic:
            self._init_semantic_detector()
            semantic_result = self.semantic_detector.detect(trajectory)

            if self.verbose:
                print(f"Semantic: is_attack={semantic_result.is_attack}, "
                      f"confidence={semantic_result.confidence:.2f}")

            # Early exit for high-confidence semantic detection
            if (semantic_result.is_attack and
                semantic_result.confidence >= self.semantic_early_exit_threshold):

                if self.verbose:
                    print(f"Early exit: semantic confidence {semantic_result.confidence:.0%} >= "
                          f"{self.semantic_early_exit_threshold:.0%}")

                self._stats['semantic_attacks'] += 1

                return UnifiedDetectionResult(
                    is_attack=True,
                    confidence=semantic_result.confidence,
                    explanation=f"Semantic jailbreak detected (early exit): {semantic_result.explanation}",
                    semantic_is_attack=True,
                    semantic_confidence=semantic_result.confidence,
                    semantic_details=semantic_result.triggered_features or {},
                    causal_is_attack=False,
                    causal_confidence=0.0,
                    causal_details={"skipped": "semantic early exit"},
                    fusion_method="semantic_early_exit",
                    attack_type="jailbreak",
                )

        # Step 2: Build causal graph
        graph = self.graph_builder.build(trajectory)

        # Step 3: Causal detection (on graph)
        if self.use_causal:
            self._init_causal_detector()
            causal_result = self.causal_detector.predict(graph)

            if self.verbose:
                print(f"Causal: is_attack={causal_result.is_attack}, "
                      f"confidence={causal_result.confidence:.2f}")

        # Step 4: Fuse results
        return self._fuse_results(semantic_result, causal_result, trajectory, graph)

    def _fuse_results(
        self,
        semantic_result: Optional[DetectionResult],
        causal_result: Optional[DetectionResult],
        trajectory: Trajectory,
        graph: CausalGraph,
    ) -> UnifiedDetectionResult:
        """
        Fuse semantic and causal detection results.

        Args:
            semantic_result: Result from semantic detector (or None)
            causal_result: Result from causal detector (or None)
            trajectory: Original trajectory
            graph: Causal graph

        Returns:
            UnifiedDetectionResult with fused decision
        """
        # Extract values (with defaults)
        sem_attack = semantic_result.is_attack if semantic_result else False
        sem_conf = semantic_result.confidence if semantic_result else 0.0

        caus_attack = causal_result.is_attack if causal_result else False
        caus_conf = causal_result.confidence if causal_result else 0.0

        # Track agreement
        if sem_attack and caus_attack:
            self._stats['both_attacks'] += 1
        elif sem_attack != caus_attack:
            self._stats['disagreements'] += 1
        if sem_attack:
            self._stats['semantic_attacks'] += 1
        if caus_attack:
            self._stats['causal_attacks'] += 1

        # Determine attack type
        if sem_attack and not caus_attack:
            attack_type = "jailbreak"
        elif caus_attack and not sem_attack:
            attack_type = "exfiltration"
        elif sem_attack and caus_attack:
            attack_type = "hybrid"
        else:
            attack_type = "benign"

        # Fuse based on method
        if self.fusion_method == "semantic_first":
            is_attack = sem_attack
            confidence = sem_conf
            explanation = f"Semantic-first: {semantic_result.explanation if semantic_result else 'N/A'}"

        elif self.fusion_method == "causal_first":
            is_attack = caus_attack
            confidence = caus_conf
            explanation = f"Causal-first: {causal_result.explanation if causal_result else 'N/A'}"

        elif self.fusion_method == "max":
            # Attack if either detects
            is_attack = sem_attack or caus_attack
            confidence = max(sem_conf, caus_conf)
            explanation = f"Max fusion: semantic={sem_conf:.2f}, causal={caus_conf:.2f}"

        else:  # weighted (default)
            # Weighted combination
            weighted_score = (
                self.semantic_weight * (1.0 if sem_attack else 0.0) +
                self.causal_weight * (1.0 if caus_attack else 0.0)
            )
            is_attack = weighted_score > 0.5

            # Confidence = weighted average of confidences
            if is_attack:
                confidence = (
                    self.semantic_weight * sem_conf +
                    self.causal_weight * caus_conf
                )
            else:
                confidence = 1.0 - weighted_score

            explanation = (
                f"Weighted fusion ({self.semantic_weight:.0%}/{self.causal_weight:.0%}): "
                f"semantic={sem_attack}({sem_conf:.2f}), causal={caus_attack}({caus_conf:.2f})"
            )

        # Apply thresholds
        if sem_attack and sem_conf >= self.semantic_threshold:
            is_attack = True
        if caus_attack and caus_conf >= self.causal_threshold:
            is_attack = True

        return UnifiedDetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            explanation=explanation,
            semantic_is_attack=sem_attack,
            semantic_confidence=sem_conf,
            semantic_details=semantic_result.triggered_features if semantic_result else {},
            causal_is_attack=caus_attack,
            causal_confidence=caus_conf,
            causal_details={"explanation": causal_result.explanation} if causal_result else {},
            fusion_method=self.fusion_method,
            attack_type=attack_type,
        )

    def predict(self, graph: CausalGraph) -> DetectionResult:
        """
        Predict from graph only (for compatibility with BaseDetector).

        Note: This only runs causal detection since we don't have the trajectory.
        For full unified detection, use detect() with a trajectory.

        Args:
            graph: CausalGraph to analyze

        Returns:
            DetectionResult
        """
        self._init_causal_detector()
        return self.causal_detector.predict(graph)

    def fit(self, trajectories: List[Trajectory], labels: List[bool]) -> 'UnifiedAttackDetector':
        """
        Fit the detector (optional - tunes thresholds).

        Args:
            trajectories: Training trajectories
            labels: True if attack, False if benign

        Returns:
            self
        """
        # Could tune thresholds based on validation data
        # For now, use default thresholds
        return self

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        stats = dict(self._stats)

        if self.semantic_detector:
            stats['semantic_api_cost'] = self.semantic_detector.get_cost_summary()

        return stats

    def reset_statistics(self):
        """Reset detection statistics."""
        self._stats = {
            'total_detections': 0,
            'semantic_attacks': 0,
            'causal_attacks': 0,
            'both_attacks': 0,
            'disagreements': 0,
        }
        if self.semantic_detector:
            self.semantic_detector.reset_statistics()

    def __repr__(self) -> str:
        return (
            f"UnifiedAttackDetector("
            f"semantic={self.use_semantic}, "
            f"causal={self.use_causal}, "
            f"fusion={self.fusion_method})"
        )


__all__ = ['UnifiedAttackDetector', 'UnifiedDetectionResult']
