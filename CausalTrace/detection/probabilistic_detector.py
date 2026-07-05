"""
Probabilistic Causal Inference Detector.

Wrapper for true Pearl-style causal inference that integrates
with the CausalTrace detection pipeline.

This detector computes:
- Probability of Necessity (PN): Was injection necessary for attack?
- Average Treatment Effect (ATE): Causal effect of injection on attack
- Proper counterfactuals via abduction-action-prediction
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .detector import BaseDetector, DetectionResult


@dataclass
class ProbabilisticDetectionResult(DetectionResult):
    """Extended result with probabilistic causal estimates."""

    # Probability of Necessity: P(attack wouldn't happen | no injection)
    probability_of_necessity: float = 0.0

    # Average Treatment Effect: E[attack | do(inj=1)] - E[attack | do(inj=0)]
    ate: float = 0.0

    # Probability of Sufficiency: P(attack would happen | injection)
    probability_of_sufficiency: float = 0.0

    # Injection nodes identified
    injection_nodes: List[int] = field(default_factory=list)

    # Malicious action nodes
    malicious_nodes: List[int] = field(default_factory=list)

    # Method identifier
    method: str = "probabilistic_causal_inference"

    # Full causal analysis details
    causal_details: Dict[str, Any] = field(default_factory=dict)


class ProbabilisticCausalDetector(BaseDetector):
    """
    Detector using true Pearl-style probabilistic causal inference.

    Unlike reachability-based methods, this computes actual counterfactuals:
    1. ABDUCTION: Infer noise terms from observed trajectory
    2. ACTION: Intervene do(injection=blocked)
    3. PREDICTION: Would attack still occur?

    Key metric: Probability of Necessity (PN)
    - PN > 0.5 means injection was NECESSARY for attack
    - High PN = strong causal evidence

    Usage:
        detector = ProbabilisticCausalDetector()
        result = detector.predict(graph)
        print(f"Attack: {result.is_attack}")
        print(f"PN: {result.probability_of_necessity:.2f}")
    """

    def __init__(
        self,
        pn_threshold: float = 0.5,
        ate_threshold: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize detector.

        Args:
            pn_threshold: Min Probability of Necessity to flag as attack
            ate_threshold: Min Average Treatment Effect to flag as attack
            verbose: Print detailed analysis
        """
        self.pn_threshold = pn_threshold
        self.ate_threshold = ate_threshold
        self.verbose = verbose

    def predict(self, graph: Dict[str, Any]) -> ProbabilisticDetectionResult:
        """
        Predict if graph represents an attack using probabilistic CI.

        Args:
            graph: CausalTrace graph dict

        Returns:
            ProbabilisticDetectionResult with causal estimates
        """
        from causaltrace.graph.probabilistic_causal import (
            ProbabilisticSCM,
            detect_attack_probabilistic
        )

        # Run probabilistic causal inference
        result = detect_attack_probabilistic(graph)

        # Extract causal estimates
        causal_est = result.get("causal_estimates", {})
        max_pn = causal_est.get("max_probability_of_necessity", 0.0)
        max_ate = causal_est.get("max_ate", 0.0)

        # Get node info
        scm = ProbabilisticSCM(graph)
        injection_nodes = list(scm.injection_nodes)
        malicious_nodes = list(scm.malicious_nodes)

        # Apply thresholds
        is_attack = result["is_attack"]
        if max_pn >= self.pn_threshold:
            is_attack = True
        if max_ate >= self.ate_threshold:
            is_attack = True

        # Compute confidence
        confidence = result["confidence"]

        if self.verbose:
            print(f"Probabilistic CI: PN={max_pn:.2f}, ATE={max_ate:.2f}, attack={is_attack}")

        return ProbabilisticDetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            explanation=result["explanation"],
            probability_of_necessity=max_pn,
            ate=max_ate,
            probability_of_sufficiency=0.0,  # Can compute if needed
            injection_nodes=injection_nodes,
            malicious_nodes=malicious_nodes,
            method="probabilistic_causal_inference",
            causal_details=causal_est
        )

    def predict_batch(self, graphs: List[Dict[str, Any]]) -> List[ProbabilisticDetectionResult]:
        """Predict for multiple graphs."""
        return [self.predict(g) for g in graphs]

    def fit(self, graphs: List[Dict[str, Any]], labels: List[bool]) -> 'ProbabilisticCausalDetector':
        """
        Fit detector (optional - can tune thresholds).

        Probabilistic CI is principled and doesn't require training,
        but thresholds can be tuned on validation data.

        Args:
            graphs: Training graphs
            labels: True if attack, False if benign

        Returns:
            self
        """
        # Optional: tune thresholds using training data
        # For now, use defaults
        return self

    def __repr__(self) -> str:
        return f"ProbabilisticCausalDetector(pn_threshold={self.pn_threshold}, ate_threshold={self.ate_threshold})"


__all__ = ['ProbabilisticCausalDetector', 'ProbabilisticDetectionResult']
