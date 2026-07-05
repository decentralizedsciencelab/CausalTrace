"""
Watermark-based attack detector.

This detector combines structural analysis with watermark tamper detection.
It detects attacks by checking if watermark tokens have been removed or
tampered with on sensitive nodes in the causal graph.

Watermark detection is a defensive technique where legitimate content is
tagged with invisible watermarks. If an agent processes injected content
that lacks these watermarks but performs sensitive actions based on it,
this indicates a potential attack.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .detector import BaseDetector, DetectionResult
from ..features import FeatureExtractor, FeatureVector
from ..graph import CausalGraph


@dataclass
class WatermarkConfig:
    """
    Configuration for watermark-based detection.

    Attributes:
        watermark_token: The token used to mark legitimate content
        require_watermark_lineage: If True, sensitive actions must trace back
                                   to watermarked content
        sensitive_action_types: Action types considered sensitive (e.g., TOOL_CALL)
        sensitive_domains: Domains where actions are considered sensitive
        sensitive_keywords: Keywords in action targets that indicate sensitivity
    """
    watermark_token: str
    require_watermark_lineage: bool = True
    sensitive_action_types: List[str] = field(default_factory=lambda: [
        "TOOL_CALL",
        "SUBMIT",
        "POST",
        "DELETE",
        "EXECUTE",
        "SEND",
        "TRANSFER",
        "CREATE_TOKEN",
        "ADD_SSH_KEY",
    ])
    sensitive_domains: List[str] = field(default_factory=lambda: [
        "bank",
        "payment",
        "admin",
        "api",
        "auth",
        "credential",
    ])
    sensitive_keywords: List[str] = field(default_factory=lambda: [
        "password",
        "token",
        "key",
        "secret",
        "credential",
        "api_key",
        "ssh",
        "transfer",
        "delete",
        "admin",
        "root",
        "sudo",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "watermark_token": self.watermark_token,
            "require_watermark_lineage": self.require_watermark_lineage,
            "sensitive_action_types": self.sensitive_action_types,
            "sensitive_domains": self.sensitive_domains,
            "sensitive_keywords": self.sensitive_keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatermarkConfig":
        """Create config from dictionary."""
        return cls(
            watermark_token=data["watermark_token"],
            require_watermark_lineage=data.get("require_watermark_lineage", True),
            sensitive_action_types=data.get("sensitive_action_types", []),
            sensitive_domains=data.get("sensitive_domains", []),
            sensitive_keywords=data.get("sensitive_keywords", []),
        )


class WatermarkDetector(BaseDetector):
    """
    Detector that combines structural analysis with watermark tamper detection.

    Detection Logic:
    1. Check if watermark is tampered (using graph metadata from GraphBuilder)
    2. If tampered, check if tampered nodes led to sensitive actions
    3. Additionally check structural features (chain depth, cross-domain edges)

    An attack is flagged if:
    - Watermark is tampered AND tampered nodes caused sensitive actions, OR
    - Structural features exceed thresholds (as a fallback)

    Example:
        >>> config = WatermarkConfig(
        ...     watermark_token="[[LEGITIMATE_CONTENT]]",
        ...     sensitive_action_types=["TOOL_CALL", "SUBMIT"]
        ... )
        >>> detector = WatermarkDetector(config)
        >>> result = detector.predict(graph)
        >>> if result.is_attack and result.watermark_tampered:
        ...     print("Attack detected via watermark tampering!")
    """

    def __init__(
        self,
        config: Optional[WatermarkConfig] = None,
        chain_depth_threshold: int = 10,
        cross_domain_ratio_threshold: float = 0.3,
        enable_structural_fallback: bool = True,
        watermark_weight: float = 0.7,
        structural_weight: float = 0.3,
    ):
        """
        Initialize watermark detector.

        Args:
            config: Watermark configuration. If None, uses defaults.
            chain_depth_threshold: Threshold for structural chain depth detection
            cross_domain_ratio_threshold: Threshold for cross-domain edge ratio
            enable_structural_fallback: If True, also check structural features
            watermark_weight: Weight for watermark-based detection in confidence
            structural_weight: Weight for structural detection in confidence
        """
        super().__init__()

        self.config = config or WatermarkConfig(
            watermark_token="[[CAUSAL_TRACE_WATERMARK]]"
        )
        self.chain_depth_threshold = chain_depth_threshold
        self.cross_domain_ratio_threshold = cross_domain_ratio_threshold
        self.enable_structural_fallback = enable_structural_fallback
        self.watermark_weight = watermark_weight
        self.structural_weight = structural_weight

        self.feature_extractor = FeatureExtractor()

        # Training statistics
        self.training_stats: Dict[str, Any] = {}

        # Mark as fitted since we have default thresholds
        self.is_fitted = True

    def fit(
        self,
        graphs: List[CausalGraph],
        labels: List[bool],
    ) -> "WatermarkDetector":
        """
        Train the watermark detector.

        Learns optimal thresholds from training data and computes statistics
        about watermark tampering patterns in attack vs benign trajectories.

        Args:
            graphs: List of CausalGraph objects
            labels: List of boolean labels (True = attack, False = benign)

        Returns:
            self (for method chaining)
        """
        super().fit(graphs, labels)  # Validates input

        # Extract features
        features = self.feature_extractor.extract_batch(graphs)

        # Analyze watermark patterns
        watermark_stats = self._analyze_watermark_patterns(graphs, labels)
        self.training_stats["watermark_patterns"] = watermark_stats

        # Compute structural statistics
        structural_stats = self._compute_structural_stats(features, labels)
        self.training_stats["structural"] = structural_stats

        # Auto-tune thresholds if we have enough data
        if len(graphs) >= 20:
            self._tune_thresholds(features, labels)

        self.is_fitted = True
        self.metadata["num_training_samples"] = len(graphs)
        self.metadata["num_attacks"] = sum(labels)
        self.metadata["watermark_token"] = self.config.watermark_token

        return self

    def _analyze_watermark_patterns(
        self,
        graphs: List[CausalGraph],
        labels: List[bool],
    ) -> Dict[str, Any]:
        """Analyze watermark tampering patterns in training data."""
        attack_tampered = 0
        attack_not_tampered = 0
        benign_tampered = 0
        benign_not_tampered = 0

        for graph, is_attack in zip(graphs, labels):
            tampered = graph.get_metadata("watermark_tampered", False)

            if is_attack:
                if tampered:
                    attack_tampered += 1
                else:
                    attack_not_tampered += 1
            else:
                if tampered:
                    benign_tampered += 1
                else:
                    benign_not_tampered += 1

        total_attacks = sum(labels)
        total_benign = len(labels) - total_attacks

        return {
            "attack_tampered_rate": attack_tampered / total_attacks if total_attacks > 0 else 0.0,
            "benign_tampered_rate": benign_tampered / total_benign if total_benign > 0 else 0.0,
            "attack_tampered": attack_tampered,
            "attack_not_tampered": attack_not_tampered,
            "benign_tampered": benign_tampered,
            "benign_not_tampered": benign_not_tampered,
        }

    def _compute_structural_stats(
        self,
        features: List[FeatureVector],
        labels: List[bool],
    ) -> Dict[str, Any]:
        """Compute structural feature statistics."""
        attack_chain_depths = []
        attack_cross_domain_ratios = []
        benign_chain_depths = []
        benign_cross_domain_ratios = []

        for feat, is_attack in zip(features, labels):
            if is_attack:
                attack_chain_depths.append(feat.chain_depth)
                attack_cross_domain_ratios.append(feat.cross_domain_ratio)
            else:
                benign_chain_depths.append(feat.chain_depth)
                benign_cross_domain_ratios.append(feat.cross_domain_ratio)

        import numpy as np

        return {
            "attack_chain_depth_mean": float(np.mean(attack_chain_depths)) if attack_chain_depths else 0.0,
            "attack_chain_depth_std": float(np.std(attack_chain_depths)) if attack_chain_depths else 0.0,
            "benign_chain_depth_mean": float(np.mean(benign_chain_depths)) if benign_chain_depths else 0.0,
            "benign_chain_depth_std": float(np.std(benign_chain_depths)) if benign_chain_depths else 0.0,
            "attack_cross_domain_ratio_mean": float(np.mean(attack_cross_domain_ratios)) if attack_cross_domain_ratios else 0.0,
            "benign_cross_domain_ratio_mean": float(np.mean(benign_cross_domain_ratios)) if benign_cross_domain_ratios else 0.0,
        }

    def _tune_thresholds(
        self,
        features: List[FeatureVector],
        labels: List[bool],
    ) -> None:
        """Auto-tune thresholds based on training data."""
        import numpy as np

        # Tune chain depth threshold
        chain_depths = [f.chain_depth for f in features]
        attack_depths = [d for d, l in zip(chain_depths, labels) if l]
        benign_depths = [d for d, l in zip(chain_depths, labels) if not l]

        if attack_depths and benign_depths:
            # Set threshold between benign 95th percentile and attack 25th percentile
            benign_p95 = np.percentile(benign_depths, 95)
            attack_p25 = np.percentile(attack_depths, 25)
            self.chain_depth_threshold = int((benign_p95 + attack_p25) / 2)

        # Tune cross-domain ratio threshold
        cross_domain_ratios = [f.cross_domain_ratio for f in features]
        attack_ratios = [r for r, l in zip(cross_domain_ratios, labels) if l]
        benign_ratios = [r for r, l in zip(cross_domain_ratios, labels) if not l]

        if attack_ratios and benign_ratios:
            benign_p95 = np.percentile(benign_ratios, 95)
            attack_p25 = np.percentile(attack_ratios, 25)
            self.cross_domain_ratio_threshold = (benign_p95 + attack_p25) / 2

    def predict(self, graph: CausalGraph) -> DetectionResult:
        """
        Predict whether a causal graph represents an attack.

        Combines watermark tamper detection with structural analysis.

        Args:
            graph: CausalGraph object

        Returns:
            DetectionResult with classification and explanation
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        # Get watermark metadata from graph
        watermark_tampered = graph.get_metadata("watermark_tampered", False)
        watermark_tampered_nodes = graph.get_metadata("watermark_tampered_nodes", [])
        watermark_sensitive_nodes = graph.get_metadata("watermark_sensitive_nodes", [])
        watermark_tagged_nodes = graph.get_metadata("watermark_tagged_nodes", [])

        # Check for watermark-based attack
        watermark_attack = False
        watermark_explanation = ""
        watermark_confidence = 0.0

        if watermark_tampered:
            # Check if tampered nodes led to sensitive actions
            tampered_to_sensitive = self._check_tampered_to_sensitive(
                graph, watermark_tampered_nodes, watermark_sensitive_nodes
            )

            if tampered_to_sensitive:
                watermark_attack = True
                watermark_confidence = 0.9
                watermark_explanation = (
                    f"Watermark tampering detected: {len(watermark_tampered_nodes)} "
                    f"tampered nodes led to {len(watermark_sensitive_nodes)} sensitive actions. "
                    f"Tampered nodes: {watermark_tampered_nodes[:5]}{'...' if len(watermark_tampered_nodes) > 5 else ''}"
                )
            else:
                watermark_confidence = 0.5
                watermark_explanation = (
                    f"Watermark tampering detected on {len(watermark_tampered_nodes)} nodes, "
                    "but no direct causal link to sensitive actions found."
                )

        # Check structural features (fallback or additional evidence)
        structural_attack = False
        structural_explanation = ""
        structural_confidence = 0.0

        if self.enable_structural_fallback:
            features = self.feature_extractor.extract(graph)
            structural_violations = []

            if features.chain_depth > self.chain_depth_threshold:
                structural_violations.append(
                    f"chain_depth={features.chain_depth} > {self.chain_depth_threshold}"
                )

            if features.cross_domain_ratio > self.cross_domain_ratio_threshold:
                structural_violations.append(
                    f"cross_domain_ratio={features.cross_domain_ratio:.2%} > {self.cross_domain_ratio_threshold:.2%}"
                )

            if structural_violations:
                structural_attack = True
                structural_confidence = min(0.8, 0.4 * len(structural_violations))
                structural_explanation = f"Structural violations: {'; '.join(structural_violations)}"

        # Combine decisions
        is_attack = watermark_attack or structural_attack

        # Calculate combined confidence
        if watermark_attack and structural_attack:
            # Both methods agree - high confidence
            confidence = min(1.0, self.watermark_weight * watermark_confidence + self.structural_weight * structural_confidence + 0.1)
            explanation = f"Attack detected via watermark AND structural analysis. {watermark_explanation} {structural_explanation}"
        elif watermark_attack:
            confidence = watermark_confidence
            explanation = f"Attack detected via watermark tampering. {watermark_explanation}"
        elif structural_attack:
            confidence = structural_confidence
            explanation = f"Attack detected via structural analysis. {structural_explanation}"
        else:
            confidence = 0.8
            explanation = "No attack indicators detected. Watermark intact and structural features within normal range."

        # Build triggered features
        triggered_features: Dict[str, Any] = {}
        if watermark_attack:
            triggered_features["watermark_tampered"] = True
            triggered_features["tampered_nodes_count"] = len(watermark_tampered_nodes)
            triggered_features["sensitive_nodes_count"] = len(watermark_sensitive_nodes)
        if structural_attack:
            triggered_features["structural_violations"] = True
            if self.enable_structural_fallback:
                features = self.feature_extractor.extract(graph)
                triggered_features["chain_depth"] = features.chain_depth
                triggered_features["cross_domain_ratio"] = features.cross_domain_ratio

        result = DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            triggered_features=triggered_features,
            explanation=explanation,
            raw_scores={
                "watermark_confidence": watermark_confidence,
                "structural_confidence": structural_confidence,
            },
            watermark_tampered=watermark_tampered,
            watermark_tampered_nodes=watermark_tampered_nodes,
            watermark_sensitive_nodes=watermark_sensitive_nodes,
            watermark_token=self.config.watermark_token,
        )

        return result

    def _check_tampered_to_sensitive(
        self,
        graph: CausalGraph,
        tampered_nodes: List[int],
        sensitive_nodes: List[int],
    ) -> bool:
        """
        Check if any tampered node has a causal path to a sensitive node.

        Args:
            graph: The causal graph
            tampered_nodes: List of node IDs with tampered watermarks
            sensitive_nodes: List of node IDs with sensitive actions

        Returns:
            True if there's a causal path from any tampered node to any sensitive node
        """
        if not tampered_nodes or not sensitive_nodes:
            return False

        tampered_set = set(tampered_nodes)
        sensitive_set = set(sensitive_nodes)

        # Check if any tampered node can reach any sensitive node
        for tampered_node in tampered_set:
            descendants = graph.get_descendants(tampered_node)
            if descendants & sensitive_set:
                return True

        return False

    def _is_sensitive_action(self, graph: CausalGraph, node_id: int) -> bool:
        """
        Check if a node represents a sensitive action.

        Args:
            graph: The causal graph
            node_id: Node ID to check

        Returns:
            True if the action is considered sensitive
        """
        node_data = graph.get_node_data(node_id)
        if not node_data:
            return False

        # Check action type
        action_type = node_data.get("action_type", "")
        if hasattr(action_type, "value"):
            action_type = action_type.value
        action_type_str = str(action_type).upper()

        if action_type_str in [t.upper() for t in self.config.sensitive_action_types]:
            return True

        # Check domain
        domain = node_data.get("domain", "").lower()
        for sensitive_domain in self.config.sensitive_domains:
            if sensitive_domain.lower() in domain:
                return True

        # Check target for sensitive keywords
        target = str(node_data.get("target", "")).lower()
        for keyword in self.config.sensitive_keywords:
            if keyword.lower() in target:
                return True

        return False

    def get_watermark_info(self) -> Dict[str, Any]:
        """
        Get information about watermark configuration and statistics.

        Returns:
            Dictionary with watermark config and training statistics
        """
        return {
            "config": self.config.to_dict(),
            "chain_depth_threshold": self.chain_depth_threshold,
            "cross_domain_ratio_threshold": self.cross_domain_ratio_threshold,
            "enable_structural_fallback": self.enable_structural_fallback,
            "training_stats": self.training_stats,
        }

    def save(self, path: str) -> None:
        """
        Save detector configuration to disk.

        Args:
            path: Path to save file
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "type": "WatermarkDetector",
            "config": self.config.to_dict(),
            "chain_depth_threshold": self.chain_depth_threshold,
            "cross_domain_ratio_threshold": self.cross_domain_ratio_threshold,
            "enable_structural_fallback": self.enable_structural_fallback,
            "watermark_weight": self.watermark_weight,
            "structural_weight": self.structural_weight,
            "training_stats": self.training_stats,
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"WatermarkDetector saved to {path}")

    @classmethod
    def load(cls, path: str) -> "WatermarkDetector":
        """
        Load detector from configuration file.

        Args:
            path: Path to saved configuration

        Returns:
            Loaded WatermarkDetector instance
        """
        with open(path, "r") as f:
            config_data = json.load(f)

        watermark_config = WatermarkConfig.from_dict(config_data.get("config", {}))

        detector = cls(
            config=watermark_config,
            chain_depth_threshold=config_data.get("chain_depth_threshold", 10),
            cross_domain_ratio_threshold=config_data.get("cross_domain_ratio_threshold", 0.3),
            enable_structural_fallback=config_data.get("enable_structural_fallback", True),
            watermark_weight=config_data.get("watermark_weight", 0.7),
            structural_weight=config_data.get("structural_weight", 0.3),
        )

        detector.training_stats = config_data.get("training_stats", {})
        detector.metadata = config_data.get("metadata", {})
        detector.is_fitted = True

        print(f"WatermarkDetector loaded from {path}")
        return detector

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"WatermarkDetector({fitted_status}, "
            f"token='{self.config.watermark_token[:20]}...', "
            f"structural_fallback={self.enable_structural_fallback})"
        )
