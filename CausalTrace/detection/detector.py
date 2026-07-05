"""
Base detector interface and detection result dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path


@dataclass
class DetectionResult:
    """
    Result of attack detection on a causal graph.

    Attributes:
        is_attack: Whether the trajectory is classified as an attack
        confidence: Confidence score between 0.0 and 1.0
        triggered_features: Features that contributed to the detection decision
        explanation: Human-readable explanation of the decision
        raw_scores: Optional raw model scores/probabilities
    """
    is_attack: bool
    confidence: float  # 0.0 to 1.0
    triggered_features: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    raw_scores: Optional[Dict[str, float]] = None
    watermark_tampered: Optional[bool] = None
    watermark_sensitive_nodes: Optional[List[int]] = None
    watermark_tampered_nodes: Optional[List[int]] = None
    watermark_token: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable representation."""
        result = "ATTACK" if self.is_attack else "BENIGN"
        watermark_info = ""
        if self.watermark_tampered is not None:
            watermark_info = f"\nWatermark Tampered: {self.watermark_tampered}"
            if self.watermark_tampered_nodes:
                watermark_info += f" (nodes {self.watermark_tampered_nodes})"
        return f"""
Detection Result: {result}
Confidence: {self.confidence:.2%}
Explanation: {self.explanation}
Key Features: {', '.join(f'{k}={v}' for k, v in list(self.triggered_features.items())[:5])}
        {watermark_info}""".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_attack': self.is_attack,
            'confidence': self.confidence,
            'triggered_features': self.triggered_features,
            'explanation': self.explanation,
            'raw_scores': self.raw_scores,
            'watermark_tampered': self.watermark_tampered,
            'watermark_sensitive_nodes': self.watermark_sensitive_nodes,
            'watermark_tampered_nodes': self.watermark_tampered_nodes,
            'watermark_token': self.watermark_token,
        }


class BaseDetector(ABC):
    """
    Abstract base class for all attack detectors.

    All detectors must implement fit() and predict() methods.
    The detector should work with CausalGraph objects and return
    DetectionResult instances.
    """

    def __init__(self):
        """Initialize the detector."""
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, graphs: List[Any], labels: List[bool]) -> 'BaseDetector':
        """
        Train the detector on labeled causal graphs.

        Args:
            graphs: List of CausalGraph objects
            labels: List of boolean labels (True = attack, False = benign)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If graphs and labels have different lengths
        """
        if len(graphs) != len(labels):
            raise ValueError(f"Mismatch: {len(graphs)} graphs vs {len(labels)} labels")
        pass

    @abstractmethod
    def predict(self, graph: Any) -> DetectionResult:
        """
        Predict whether a causal graph represents an attack.

        Args:
            graph: A CausalGraph object

        Returns:
            DetectionResult with classification and explanation

        Raises:
            RuntimeError: If detector has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        pass

    def predict_batch(self, graphs: List[Any]) -> List[DetectionResult]:
        """
        Predict for multiple graphs.

        Default implementation calls predict() for each graph.
        Subclasses can override for batch optimization.

        Args:
            graphs: List of CausalGraph objects

        Returns:
            List of DetectionResult objects
        """
        return [self.predict(graph) for graph in graphs]

    def predict_proba(self, graph: Any) -> float:
        """
        Get probability that the graph represents an attack.

        Args:
            graph: A CausalGraph object

        Returns:
            Probability between 0.0 and 1.0
        """
        result = self.predict(graph)
        return result.confidence if result.is_attack else (1.0 - result.confidence)

    def _attach_watermark_metadata(self, graph: Any, result: DetectionResult) -> DetectionResult:
        """
        Attach watermark metadata from the causal graph to the detection result.
        """
        if not hasattr(graph, "get_metadata"):
            return result

        token = graph.get_metadata("watermark_token", None)
        tampered = graph.get_metadata("watermark_tampered", None)
        sensitive_nodes = graph.get_metadata("watermark_sensitive_nodes", [])
        tampered_nodes = graph.get_metadata("watermark_tampered_nodes", [])

        if token is None and tampered is None and not sensitive_nodes and not tampered_nodes:
            return result

        result.watermark_token = token
        if tampered is not None:
            result.watermark_tampered = bool(tampered)
        if sensitive_nodes:
            result.watermark_sensitive_nodes = sensitive_nodes
        if tampered_nodes:
            result.watermark_tampered_nodes = tampered_nodes

        if result.watermark_tampered:
            suffix = f" Watermark missing on sensitive nodes {tampered_nodes}."
            result.explanation = (result.explanation + suffix).strip()

        return result

    def save(self, path: str) -> None:
        """
        Save trained detector to disk.

        Args:
            path: Path to save file
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print(f"Detector saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseDetector':
        """
        Load trained detector from disk.

        Args:
            path: Path to saved detector file

        Returns:
            Loaded detector instance

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        with open(path, 'rb') as f:
            detector = pickle.load(f)

        print(f"Detector loaded from {path}")
        return detector

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get detector metadata (hyperparameters, training info, etc.).

        Returns:
            Dictionary of metadata
        """
        return {
            'is_fitted': self.is_fitted,
            'detector_type': self.__class__.__name__,
            **self.metadata
        }

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_status})"
