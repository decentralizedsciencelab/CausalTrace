"""
Mechanism Integrity Detector.

Detects attacks by identifying when external content has hijacked the
causal mechanism that should only be driven by user intent.

This provides a theoretically grounded approach based on:
- Causal Mechanism Independence (Schölkopf et al.)
- Independent Causal Mechanisms principle
- Intervention calculus (Pearl)

Key Innovation:
--------------
Instead of asking "would harm occur without injection?" (counterfactual),
we ask "has external content hijacked the causal mechanism?" (mechanism invariance).

This is complementary to counterfactual detection and can catch attacks that
counterfactual analysis might miss (and vice versa).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .detector import BaseDetector, DetectionResult


@dataclass
class MechanismDetectionResult(DetectionResult):
    """Extended result with mechanism integrity details."""

    # Overall integrity score (1.0 = intact, 0.0 = hijacked)
    integrity_score: float = 1.0

    # Mechanism divergence from expected
    mechanism_divergence: float = 0.0

    # Actions that were hijacked
    hijacked_actions: List[int] = None

    # Sources that caused the hijacking
    hijacking_sources: List[int] = None

    # Method used
    method: str = "mechanism_integrity"

    def __post_init__(self):
        if self.hijacked_actions is None:
            self.hijacked_actions = []
        if self.hijacking_sources is None:
            self.hijacking_sources = []


class MechanismIntegrityDetector(BaseDetector):
    """
    Detects attacks via causal mechanism integrity analysis.

    Theory:
    ------
    An agent has an INTENDED causal mechanism:
        User Intent → Agent Reasoning → Actions → Outcomes

    An attack HIJACKS this mechanism:
        Injection → Agent Reasoning → Actions → Outcomes

    We detect hijacking by measuring mechanism divergence:
        KL(P(action | user_intent) || P(action | user_intent, external_content))

    If divergence > threshold, the mechanism has been corrupted.

    Usage:
    ------
    ```python
    detector = MechanismIntegrityDetector()

    # For a single graph
    result = detector.predict(graph)
    print(f"Attack: {result.is_attack}, Integrity: {result.integrity_score}")

    # Batch prediction
    results = detector.predict_batch(graphs)
    ```
    """

    def __init__(
        self,
        integrity_threshold: float = 0.7,
        divergence_threshold: float = 0.3,
        require_sensitive_hijack: bool = True,
        verbose: bool = False
    ):
        """
        Initialize detector.

        Args:
            integrity_threshold: Min integrity score to be considered benign (default 0.7)
            divergence_threshold: Max mechanism divergence for benign (default 0.3)
            require_sensitive_hijack: Only flag attack if sensitive action hijacked (default True)
            verbose: Print detailed analysis (default False)
        """
        self.integrity_threshold = integrity_threshold
        self.divergence_threshold = divergence_threshold
        self.require_sensitive_hijack = require_sensitive_hijack
        self.verbose = verbose

    def predict(self, graph: Dict[str, Any]) -> MechanismDetectionResult:
        """
        Predict if graph represents an attack using mechanism integrity.

        Args:
            graph: CausalTrace graph dict

        Returns:
            MechanismDetectionResult with detailed analysis
        """
        from causaltrace.graph.mechanism_integrity import (
            CausalMechanismAnalyzer,
            detect_mechanism_hijacking
        )

        # Run mechanism integrity analysis
        result = detect_mechanism_hijacking(graph)

        # Build detection result
        is_attack = result["is_attack"]
        confidence = result["confidence"]

        # Apply thresholds
        integrity_score = result["integrity_score"]
        mechanism_divergence = result["mechanism_divergence"]

        # Override with threshold-based decision
        if integrity_score < self.integrity_threshold:
            is_attack = True
        if mechanism_divergence > self.divergence_threshold:
            is_attack = True

        # Require sensitive action hijacking if configured
        if self.require_sensitive_hijack and is_attack:
            # Check if any hijacked action is sensitive
            analyzer = CausalMechanismAnalyzer(graph)
            hijacked_set = set(result["hijacked_actions"])
            sensitive_hijacked = hijacked_set & analyzer.sensitive_action_nodes
            if not sensitive_hijacked:
                is_attack = False
                confidence *= 0.5  # Lower confidence

        if self.verbose:
            print(f"Mechanism Analysis: integrity={integrity_score:.2f}, "
                  f"divergence={mechanism_divergence:.2f}, attack={is_attack}")

        return MechanismDetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            explanation=result["explanation"],
            integrity_score=integrity_score,
            mechanism_divergence=mechanism_divergence,
            hijacked_actions=result["hijacked_actions"],
            hijacking_sources=result["hijacking_sources"],
            method="mechanism_integrity"
        )

    def predict_batch(self, graphs: List[Dict[str, Any]]) -> List[MechanismDetectionResult]:
        """
        Predict for multiple graphs.

        Args:
            graphs: List of CausalTrace graph dicts

        Returns:
            List of MechanismDetectionResult
        """
        return [self.predict(g) for g in graphs]

    def fit(self, graphs: List[Dict[str, Any]], labels: List[bool]) -> 'MechanismIntegrityDetector':
        """
        Fit detector (optional - can tune thresholds on training data).

        For mechanism integrity, fitting is optional since it's based on
        structural principles rather than learned patterns.

        Args:
            graphs: Training graphs
            labels: True if attack, False if benign

        Returns:
            self
        """
        # Optional: tune thresholds based on training data
        # For now, use default thresholds
        return self


class EnsembleDetector(BaseDetector):
    """
    Ensemble detector combining multiple detection methods.

    Combines:
    1. Counterfactual detection (Pearl Level 3 - graph approximation)
    2. Mechanism integrity detection (V1)
    3. Probabilistic Causal Inference (true Pearl counterfactuals) - NEW
    4. Optional: ML-based detection

    Uses voting or weighted combination for final decision.
    """

    def __init__(
        self,
        use_counterfactual: bool = True,
        use_mechanism: bool = True,
        use_probabilistic: bool = True,
        use_ml: bool = False,
        ml_detector: Optional[BaseDetector] = None,
        voting: str = "majority",  # "majority", "any", "all", "weighted"
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ):
        """
        Initialize ensemble detector.

        Args:
            use_counterfactual: Include counterfactual detection (graph-based)
            use_mechanism: Include mechanism integrity detection
            use_probabilistic: Include probabilistic causal inference (true Pearl)
            use_ml: Include ML-based detection
            ml_detector: Pre-trained ML detector (required if use_ml=True)
            voting: How to combine predictions
            weights: Weights for weighted voting
            verbose: Print detailed analysis
        """
        self.use_counterfactual = use_counterfactual
        self.use_mechanism = use_mechanism
        self.use_probabilistic = use_probabilistic
        self.use_ml = use_ml
        self.ml_detector = ml_detector
        self.voting = voting
        self.weights = weights or {
            "counterfactual": 0.25,
            "mechanism": 0.25,
            "probabilistic": 0.35,
            "ml": 0.15
        }
        self.verbose = verbose

        self.mechanism_detector = MechanismIntegrityDetector(verbose=verbose)

        # Use V2 mechanism detector by default (better recall)
        self.use_v2_mechanism = True

    def predict(self, graph: Dict[str, Any]) -> DetectionResult:
        """
        Predict using ensemble of methods.

        Args:
            graph: CausalTrace graph dict

        Returns:
            DetectionResult with ensemble decision
        """
        votes = []
        confidences = []
        explanations = []

        # Counterfactual detection
        if self.use_counterfactual:
            from causaltrace.graph.causal_inference import detect_attack_causal
            cf_result = detect_attack_causal(graph)
            votes.append(("counterfactual", cf_result["is_attack"], cf_result["confidence"]))
            explanations.append(f"Counterfactual: {cf_result['explanation'][:100]}")

        # Mechanism integrity detection (use V2 for better recall)
        if self.use_mechanism:
            if self.use_v2_mechanism:
                from causaltrace.graph.mechanism_integrity_v2 import CausalMechanismIntegrity
                v2_analyzer = CausalMechanismIntegrity(graph)
                v2_result = v2_analyzer.analyze_mechanism_integrity()
                votes.append(("mechanism", v2_result.is_attack, v2_result.confidence))
                explanations.append(f"MechanismV2: dev={v2_result.deviation_score:.2f}")
            else:
                mech_result = self.mechanism_detector.predict(graph)
                votes.append(("mechanism", mech_result.is_attack, mech_result.confidence))
                explanations.append(f"Mechanism: integrity={mech_result.integrity_score:.2f}")

        # Probabilistic Causal Inference (true Pearl counterfactuals)
        if self.use_probabilistic:
            from causaltrace.graph.probabilistic_causal import detect_attack_probabilistic
            prob_result = detect_attack_probabilistic(graph)
            pn = prob_result.get("causal_estimates", {}).get("max_probability_of_necessity", 0)
            votes.append(("probabilistic", prob_result["is_attack"], prob_result["confidence"]))
            explanations.append(f"ProbCI: PN={pn:.2f}")

        # ML detection
        if self.use_ml and self.ml_detector:
            ml_result = self.ml_detector.predict(graph)
            votes.append(("ml", ml_result.is_attack, ml_result.confidence))
            explanations.append(f"ML: confidence={ml_result.confidence:.2f}")

        # Combine votes
        if self.voting == "majority":
            attack_votes = sum(1 for _, is_attack, _ in votes if is_attack)
            is_attack = attack_votes > len(votes) / 2
        elif self.voting == "any":
            is_attack = any(is_attack for _, is_attack, _ in votes)
        elif self.voting == "all":
            is_attack = all(is_attack for _, is_attack, _ in votes)
        elif self.voting == "weighted":
            weighted_sum = sum(
                self.weights.get(method, 0.33) * (1.0 if is_attack else 0.0)
                for method, is_attack, _ in votes
            )
            is_attack = weighted_sum > 0.5
        else:
            is_attack = any(is_attack for _, is_attack, _ in votes)

        # Compute ensemble confidence
        if is_attack:
            # Confidence = max confidence among attack votes
            confidence = max(
                (conf for method, attack, conf in votes if attack),
                default=0.5
            )
        else:
            # Confidence = max confidence among benign votes
            confidence = max(
                (conf for method, attack, conf in votes if not attack),
                default=0.5
            )

        explanation = f"Ensemble ({self.voting}): " + " | ".join(explanations)

        if self.verbose:
            print(f"Votes: {votes}")
            print(f"Final: is_attack={is_attack}, confidence={confidence:.2f}")

        return DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            explanation=explanation
        )

    def fit(self, graphs: List[Dict[str, Any]], labels: List[bool]) -> 'EnsembleDetector':
        """
        Fit ensemble (trains ML detector if used).

        Args:
            graphs: Training graphs
            labels: True if attack, False if benign

        Returns:
            self
        """
        if self.use_ml and self.ml_detector:
            self.ml_detector.fit(graphs, labels)
        return self
