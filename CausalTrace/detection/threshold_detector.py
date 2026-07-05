"""
Threshold-based attack detector.

Simple detector that flags attacks based on threshold rules applied to
key causal graph features. Useful as a baseline and for interpretability.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from .detector import BaseDetector, DetectionResult
from ..features import FeatureExtractor, FeatureVector
from ..graph import CausalGraph


class ThresholdDetector(BaseDetector):
    """
    Simple detector using thresholds on key features.

    Detection Logic:
    - Flag as attack if ANY of these conditions hold:
      1. chain_depth > threshold_1
      2. cross_domain_ratio > threshold_2
      3. max_bottleneck_score > threshold_3
      4. state_accumulation_score > threshold_4

    The thresholds can be manually set or automatically tuned using training data.
    """

    def __init__(
        self,
        chain_depth_threshold: int = 10,
        cross_domain_ratio_threshold: float = 0.3,
        bottleneck_threshold: int = 8,
        state_accumulation_threshold: float = 0.5,
        auto_tune: bool = False
    ):
        """
        Initialize threshold detector.

        Args:
            chain_depth_threshold: Maximum allowed chain depth
            cross_domain_ratio_threshold: Maximum ratio of cross-domain edges
            bottleneck_threshold: Maximum bottleneck score
            state_accumulation_threshold: Maximum state accumulation score
            auto_tune: If True, automatically tune thresholds during fit()
        """
        super().__init__()
        self.thresholds = {
            'chain_depth': chain_depth_threshold,
            'cross_domain_ratio': cross_domain_ratio_threshold,
            'max_bottleneck_score': bottleneck_threshold,
            'state_accumulation_score': state_accumulation_threshold,
        }
        self.auto_tune = auto_tune
        self.feature_extractor = FeatureExtractor()

        # Store training statistics
        self.feature_statistics: Dict[str, Dict[str, float]] = {}

        # Mark as fitted since we have default thresholds (training only tunes them)
        self.is_fitted = True

    def fit(self, graphs: List[CausalGraph], labels: List[bool]) -> 'ThresholdDetector':
        """
        Train the threshold detector.

        If auto_tune=True, this will tune thresholds to maximize F1 score.
        Otherwise, it just computes feature statistics.

        Args:
            graphs: List of CausalGraph objects
            labels: List of boolean labels (True = attack, False = benign)

        Returns:
            self (for method chaining)
        """
        super().fit(graphs, labels)  # Validates input

        # Extract features from all graphs
        features = self.feature_extractor.extract_batch(graphs)

        # Compute statistics for each feature
        self._compute_feature_statistics(features, labels)

        if self.auto_tune:
            print("Auto-tuning thresholds...")
            self._tune_thresholds(features, labels)

        self.is_fitted = True
        self.metadata['num_training_samples'] = len(graphs)
        self.metadata['num_attacks'] = sum(labels)
        self.metadata['thresholds'] = self.thresholds.copy()

        return self

    def _compute_feature_statistics(self, features: List[FeatureVector], labels: List[bool]) -> None:
        """Compute mean/std/percentiles for each feature by class."""
        benign_features = [f for f, label in zip(features, labels) if not label]
        attack_features = [f for f, label in zip(features, labels) if label]

        for feature_name in self.thresholds.keys():
            benign_values = [getattr(f, feature_name) for f in benign_features]
            attack_values = [getattr(f, feature_name) for f in attack_features]

            self.feature_statistics[feature_name] = {
                'benign_mean': np.mean(benign_values) if benign_values else 0.0,
                'benign_std': np.std(benign_values) if benign_values else 0.0,
                'benign_p95': np.percentile(benign_values, 95) if benign_values else 0.0,
                'attack_mean': np.mean(attack_values) if attack_values else 0.0,
                'attack_std': np.std(attack_values) if attack_values else 0.0,
                'attack_p05': np.percentile(attack_values, 5) if attack_values else 0.0,
            }

    def _tune_thresholds(self, features: List[FeatureVector], labels: List[bool]) -> None:
        """
        Automatically tune thresholds to maximize F1 score.

        Strategy: For each feature, try different percentiles of the training
        distribution and select the threshold that maximizes F1 when used alone.
        """
        for feature_name in self.thresholds.keys():
            best_threshold = self.thresholds[feature_name]
            best_f1 = 0.0

            # Get all values for this feature
            values = [getattr(f, feature_name) for f in features]
            percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 99]

            for p in percentiles:
                threshold = np.percentile(values, p)

                # Test this threshold
                predictions = []
                for f in features:
                    feature_value = getattr(f, feature_name)
                    predictions.append(feature_value > threshold)

                # Compute F1
                tp = sum(pred and label for pred, label in zip(predictions, labels))
                fp = sum(pred and not label for pred, label in zip(predictions, labels))
                fn = sum(not pred and label for pred, label in zip(predictions, labels))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            self.thresholds[feature_name] = best_threshold
            print(f"  {feature_name}: {best_threshold:.3f} (F1={best_f1:.3f})")

    def predict(self, graph: CausalGraph) -> DetectionResult:
        """
        Predict whether a causal graph represents an attack.

        Args:
            graph: CausalGraph object

        Returns:
            DetectionResult with classification and explanation
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        # Extract features
        features = self.feature_extractor.extract(graph)

        # Check each threshold
        triggered = {}
        explanations = []

        if features.chain_depth > self.thresholds['chain_depth']:
            triggered['chain_depth'] = features.chain_depth
            explanations.append(
                f"Chain depth {features.chain_depth} exceeds threshold {self.thresholds['chain_depth']}"
            )

        if features.cross_domain_ratio > self.thresholds['cross_domain_ratio']:
            triggered['cross_domain_ratio'] = features.cross_domain_ratio
            explanations.append(
                f"Cross-domain ratio {features.cross_domain_ratio:.2%} exceeds threshold {self.thresholds['cross_domain_ratio']:.2%}"
            )

        if features.max_bottleneck_score > self.thresholds['max_bottleneck_score']:
            triggered['max_bottleneck_score'] = features.max_bottleneck_score
            explanations.append(
                f"Max bottleneck score {features.max_bottleneck_score} exceeds threshold {self.thresholds['max_bottleneck_score']}"
            )

        if features.state_accumulation_score > self.thresholds['state_accumulation_score']:
            triggered['state_accumulation_score'] = features.state_accumulation_score
            explanations.append(
                f"State accumulation {features.state_accumulation_score:.2f} exceeds threshold {self.thresholds['state_accumulation_score']:.2f}"
            )

        # Determine result
        is_attack = len(triggered) > 0

        if is_attack:
            confidence = min(1.0, len(triggered) / len(self.thresholds))  # More triggers = higher confidence
            explanation = "Attack detected: " + "; ".join(explanations)
        else:
            confidence = 0.8  # High confidence for benign classification
            explanation = "No threshold violations detected - classified as benign"

        result = DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            triggered_features=triggered,
            explanation=explanation,
            raw_scores={'num_violations': len(triggered)}
        )
        return self._attach_watermark_metadata(graph, result)

    def get_threshold_info(self) -> Dict[str, Any]:
        """
        Get information about thresholds and feature statistics.

        Returns:
            Dictionary with thresholds and statistics
        """
        return {
            'thresholds': self.thresholds.copy(),
            'feature_statistics': self.feature_statistics.copy(),
            'auto_tuned': self.auto_tune,
        }

    def set_threshold(self, feature_name: str, value: float) -> None:
        """
        Manually set a threshold value.

        Args:
            feature_name: Name of the feature
            value: New threshold value

        Raises:
            ValueError: If feature_name is not recognized
        """
        if feature_name not in self.thresholds:
            raise ValueError(f"Unknown feature: {feature_name}. Valid features: {list(self.thresholds.keys())}")

        self.thresholds[feature_name] = value

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"ThresholdDetector({fitted_status}, auto_tune={self.auto_tune})"
