"""
Causal inference detector for attack detection.

Uses counterfactual queries (do-calculus) to determine if an attack is causally
dependent on an injection, rather than simple reachability analysis.

Reachability: "Does a path exist from injection to malicious action?"
Causal: "Would the malicious action NOT occur if we intervened to block the injection?"
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from causaltrace.detection.detector import BaseDetector, DetectionResult
from causaltrace.graph.causal_graph import CausalGraph
from causaltrace.graph.causal_inference import (
    StructuralCausalModel,
    detect_attack_causal,
    compare_detection_methods,
    CausalEffect,
)


@dataclass
class CausalDetectionResult(DetectionResult):
    """Extended detection result with causal analysis details."""
    causal_effects: List[Dict[str, Any]] = None
    counterfactual_analysis: Dict[str, Any] = None
    reachability_agrees: bool = True

    def __post_init__(self):
        self.causal_effects = self.causal_effects or []
        self.counterfactual_analysis = self.counterfactual_analysis or {}


class CausalDetector(BaseDetector):
    """
    Detector using counterfactual causal inference.

    Asks: "Would the attack have occurred if the injection were blocked?"
    If no, the injection causally enabled the attack.
    """

    def __init__(
        self,
        causal_effect_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        require_counterfactual_dependency: bool = True
    ):
        """
        Initialize causal detector.

        Args:
            causal_effect_threshold: Minimum causal effect to classify as attack (0-1)
            confidence_threshold: Minimum confidence for positive detection
            require_counterfactual_dependency: If True, require counterfactual analysis
                                                to confirm the injection is necessary
        """
        self.causal_effect_threshold = causal_effect_threshold
        self.confidence_threshold = confidence_threshold
        self.require_counterfactual_dependency = require_counterfactual_dependency
        self.is_fitted = True  # No training needed (uses formal causal reasoning)

    def fit(self, graphs: List[CausalGraph], labels: List[bool]) -> "CausalDetector":
        """
        Fit the detector (no-op for causal detector as it's rule-based).

        Args:
            graphs: List of causal graphs
            labels: List of attack labels

        Returns:
            self
        """
        # Causal detector doesn't need training - it uses formal reasoning
        self._is_fitted = True
        return self

    def predict(self, graph: CausalGraph) -> CausalDetectionResult:
        """
        Predict if graph represents an attack using causal inference.

        Args:
            graph: CausalGraph to analyze

        Returns:
            CausalDetectionResult with causal analysis
        """
        # Export graph to dict format for SCM
        graph_dict = graph.export_to_json()

        # Run causal inference detection
        causal_result = detect_attack_causal(graph_dict)

        # Compare with reachability for diagnostics
        comparison = compare_detection_methods(graph_dict)

        # Determine final prediction
        is_attack = causal_result["is_attack"]
        confidence = causal_result.get("confidence", 0.5)
        max_causal_effect = causal_result.get("max_causal_effect", 0.0)

        # Apply thresholds
        if max_causal_effect < self.causal_effect_threshold:
            is_attack = False

        if confidence < self.confidence_threshold:
            is_attack = False

        # Build explanation
        if is_attack:
            explanation = (
                f"Causal attack detected: Injection is counterfactually necessary "
                f"for the malicious action (causal effect: {max_causal_effect:.2f}). "
                f"{causal_result.get('explanation', '')}"
            )
        else:
            explanation = (
                f"No causal attack: {causal_result.get('explanation', 'Insufficient causal evidence')}. "
                f"Max causal effect: {max_causal_effect:.2f}"
            )

        # Check if reachability and causal agree
        reachability_agrees = comparison.get("agree", True)
        if not reachability_agrees:
            explanation += f" Note: Reachability analysis disagrees - {comparison.get('disagreement_reason', '')}"

        return CausalDetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            explanation=explanation,
            triggered_features={
                "max_causal_effect": max_causal_effect,
                "method": "causal_inference",
                "reachability_result": comparison.get("reachability", {}).get("is_attack", False),
                "causal_result": causal_result["is_attack"],
            },
            causal_effects=causal_result.get("causal_effects", []),
            counterfactual_analysis={
                "injection_nodes": list(StructuralCausalModel(graph_dict).injection_nodes),
                "malicious_nodes": list(StructuralCausalModel(graph_dict).malicious_nodes),
                "reachability_agrees": reachability_agrees,
            },
            reachability_agrees=reachability_agrees,
        )

    def predict_batch(self, graphs: List[CausalGraph]) -> List[CausalDetectionResult]:
        """
        Predict for multiple graphs.

        Args:
            graphs: List of CausalGraphs

        Returns:
            List of CausalDetectionResults
        """
        return [self.predict(g) for g in graphs]

    def get_causal_explanation(self, graph: CausalGraph) -> Dict[str, Any]:
        """
        Get detailed causal explanation for a graph.

        This provides a full breakdown of the causal analysis including:
        - All injection-to-malicious pairs analyzed
        - Counterfactual worlds for each pair
        - Comparison with simple reachability

        Args:
            graph: CausalGraph to explain

        Returns:
            Detailed explanation dictionary
        """
        graph_dict = graph.export_to_json()

        scm = StructuralCausalModel(graph_dict)
        comparison = compare_detection_methods(graph_dict)

        explanation = {
            "summary": {
                "reachability_says_attack": comparison["reachability"]["is_attack"],
                "causal_says_attack": comparison["causal_inference"]["is_attack"],
                "methods_agree": comparison["agree"],
            },
            "injection_nodes": list(scm.injection_nodes),
            "malicious_nodes": list(scm.malicious_nodes),
            "reachability_analysis": comparison["reachability"],
            "causal_inference_analysis": comparison["causal_inference"],
        }

        if not comparison["agree"]:
            explanation["disagreement"] = {
                "reason": comparison.get("disagreement_reason", "Unknown"),
                "interpretation": (
                    "Reachability found a path but causal analysis determined "
                    "the attack would still occur without the injection (not causally necessary), "
                    "OR there are alternative causal paths that don't involve the injection."
                )
            }

        return explanation

    def save(self, path: str) -> None:
        """Save detector configuration."""
        import json
        config = {
            "type": "CausalDetector",
            "causal_effect_threshold": self.causal_effect_threshold,
            "confidence_threshold": self.confidence_threshold,
            "require_counterfactual_dependency": self.require_counterfactual_dependency,
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CausalDetector":
        """Load detector from configuration."""
        import json
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(
            causal_effect_threshold=config.get("causal_effect_threshold", 0.5),
            confidence_threshold=config.get("confidence_threshold", 0.6),
            require_counterfactual_dependency=config.get("require_counterfactual_dependency", True),
        )
