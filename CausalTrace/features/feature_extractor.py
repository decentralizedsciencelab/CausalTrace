"""
Feature extraction from causal graphs.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field

from causaltrace.graph import CausalGraph, EdgeType
from causaltrace.models import Trajectory
from causaltrace.taint import TaintPropagator, TaintAnalysisResult
from causaltrace.features.chain_features import (
    compute_chain_depth,
    compute_avg_chain_length,
    count_chains,
)
from causaltrace.features.domain_features import (
    count_cross_domain_edges,
    compute_cross_domain_ratio,
    count_unique_domains,
    get_cross_domain_edge_types,
    compute_domain_entropy,
)
from causaltrace.features.state_features import (
    compute_state_accumulation_score,
    compute_unused_state_ratio,
    count_state_producing_actions,
)
from causaltrace.features.bottleneck_features import (
    compute_max_bottleneck_score,
    compute_avg_bottleneck_score,
    identify_critical_bottlenecks,
)
from causaltrace.features.watermark_features import (
    compute_watermark_coverage,
    compute_watermark_propagation_depth,
    compute_watermark_sensitive_coverage,
    compute_watermark_tampered,
    compute_watermark_gap_count,
    compute_watermark_first_gap_depth,
)


@dataclass
class FeatureVector:
    """
    Feature vector extracted from a causal graph.

    Contains all features used for attack detection, organized by category.
    """

    # === Chain Features ===
    chain_depth: int = 0
    avg_chain_length: float = 0.0
    num_chains: int = 0

    # === Domain Features ===
    num_cross_domain_edges: int = 0
    cross_domain_ratio: float = 0.0
    unique_domains: int = 0
    domain_entropy: float = 0.0

    # === State Features ===
    state_accumulation_score: float = 0.0
    unused_state_ratio: float = 0.0
    num_state_producing: int = 0

    # === Bottleneck Features ===
    max_bottleneck_score: int = 0
    avg_bottleneck_score: float = 0.0
    num_critical_bottlenecks: int = 0

    # === Edge Type Distribution ===
    data_dependency_ratio: float = 0.0
    trust_transfer_ratio: float = 0.0
    state_enablement_ratio: float = 0.0

    # === Cross-Domain Edge Types ===
    cross_domain_data_dependency: int = 0
    cross_domain_trust_transfer: int = 0
    cross_domain_state_enablement: int = 0

    # === Graph Topology ===
    num_nodes: int = 0
    num_edges: int = 0
    graph_density: float = 0.0

    # === Taint Analysis Features ===
    taint_reaches_execution: bool = False  # Does tainted data reach code execution?
    taint_violation_count: int = 0         # Number of taint violations
    max_taint_path_length: int = 0         # Longest taint path to sensitive sink
    tainted_node_count: int = 0            # Number of nodes with taint
    sensitive_sink_count: int = 0          # Number of sensitive sinks
    untrusted_source_count: int = 0        # Number of untrusted sources

    # === Watermark Features ===
    watermark_coverage: float = 0.0              # Ratio of watermark-tagged nodes to total
    watermark_propagation_depth: int = 0         # Max depth of watermark propagation
    watermark_sensitive_coverage: float = 0.0    # Ratio of sensitive nodes with watermark
    watermark_tampered: int = 0                  # 1 if any sensitive node lacks watermark
    watermark_gap_count: int = 0                 # Number of sensitive nodes missing watermark
    watermark_first_gap_depth: int = 0           # Depth of first node where watermark breaks

    # === Additional Metadata (not used in ML) ===
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            float(self.chain_depth),
            self.avg_chain_length,
            float(self.num_chains),
            float(self.num_cross_domain_edges),
            self.cross_domain_ratio,
            float(self.unique_domains),
            self.domain_entropy,
            self.state_accumulation_score,
            self.unused_state_ratio,
            float(self.num_state_producing),
            float(self.max_bottleneck_score),
            self.avg_bottleneck_score,
            float(self.num_critical_bottlenecks),
            self.data_dependency_ratio,
            self.trust_transfer_ratio,
            self.state_enablement_ratio,
            float(self.cross_domain_data_dependency),
            float(self.cross_domain_trust_transfer),
            float(self.cross_domain_state_enablement),
            float(self.num_nodes),
            float(self.num_edges),
            self.graph_density,
            # Taint features
            float(self.taint_reaches_execution),
            float(self.taint_violation_count),
            float(self.max_taint_path_length),
            float(self.tainted_node_count),
            float(self.sensitive_sink_count),
            float(self.untrusted_source_count),
            # Watermark features
            self.watermark_coverage,
            float(self.watermark_propagation_depth),
            self.watermark_sensitive_coverage,
            float(self.watermark_tampered),
            float(self.watermark_gap_count),
            float(self.watermark_first_gap_depth),
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding metadata)."""
        return {
            # Chain features
            "chain_depth": self.chain_depth,
            "avg_chain_length": self.avg_chain_length,
            "num_chains": self.num_chains,
            # Domain features
            "num_cross_domain_edges": self.num_cross_domain_edges,
            "cross_domain_ratio": self.cross_domain_ratio,
            "unique_domains": self.unique_domains,
            "domain_entropy": self.domain_entropy,
            # State features
            "state_accumulation_score": self.state_accumulation_score,
            "unused_state_ratio": self.unused_state_ratio,
            "num_state_producing": self.num_state_producing,
            # Bottleneck features
            "max_bottleneck_score": self.max_bottleneck_score,
            "avg_bottleneck_score": self.avg_bottleneck_score,
            "num_critical_bottlenecks": self.num_critical_bottlenecks,
            # Edge type features
            "data_dependency_ratio": self.data_dependency_ratio,
            "trust_transfer_ratio": self.trust_transfer_ratio,
            "state_enablement_ratio": self.state_enablement_ratio,
            # Cross-domain edge types
            "cross_domain_data_dependency": self.cross_domain_data_dependency,
            "cross_domain_trust_transfer": self.cross_domain_trust_transfer,
            "cross_domain_state_enablement": self.cross_domain_state_enablement,
            # Graph topology
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "graph_density": self.graph_density,
            # Taint features
            "taint_reaches_execution": self.taint_reaches_execution,
            "taint_violation_count": self.taint_violation_count,
            "max_taint_path_length": self.max_taint_path_length,
            "tainted_node_count": self.tainted_node_count,
            "sensitive_sink_count": self.sensitive_sink_count,
            "untrusted_source_count": self.untrusted_source_count,
            # Watermark features
            "watermark_coverage": self.watermark_coverage,
            "watermark_propagation_depth": self.watermark_propagation_depth,
            "watermark_sensitive_coverage": self.watermark_sensitive_coverage,
            "watermark_tampered": self.watermark_tampered,
            "watermark_gap_count": self.watermark_gap_count,
            "watermark_first_gap_depth": self.watermark_first_gap_depth,
        }

    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names (matches to_numpy() order)."""
        return [
            "chain_depth",
            "avg_chain_length",
            "num_chains",
            "num_cross_domain_edges",
            "cross_domain_ratio",
            "unique_domains",
            "domain_entropy",
            "state_accumulation_score",
            "unused_state_ratio",
            "num_state_producing",
            "max_bottleneck_score",
            "avg_bottleneck_score",
            "num_critical_bottlenecks",
            "data_dependency_ratio",
            "trust_transfer_ratio",
            "state_enablement_ratio",
            "cross_domain_data_dependency",
            "cross_domain_trust_transfer",
            "cross_domain_state_enablement",
            "num_nodes",
            "num_edges",
            "graph_density",
            # Taint features
            "taint_reaches_execution",
            "taint_violation_count",
            "max_taint_path_length",
            "tainted_node_count",
            "sensitive_sink_count",
            "untrusted_source_count",
            # Watermark features
            "watermark_coverage",
            "watermark_propagation_depth",
            "watermark_sensitive_coverage",
            "watermark_tampered",
            "watermark_gap_count",
            "watermark_first_gap_depth",
        ]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureVector(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"chain_depth={self.chain_depth}, cross_domain={self.num_cross_domain_edges}, "
            f"state_accum={self.state_accumulation_score:.2f}, "
            f"max_bottleneck={self.max_bottleneck_score})"
        )


class FeatureExtractor:
    """
    Extract features from causal graphs for attack detection.

    This class coordinates all feature extraction and provides a unified interface.
    """

    def __init__(self, include_trajectory: bool = True, include_taint: bool = True):
        """
        Initialize the feature extractor.

        Args:
            include_trajectory: If True, state features will use trajectory context
                              (requires passing trajectory to extract())
            include_taint: If True, taint analysis features will be computed
                          (requires passing trajectory to extract())
        """
        self.include_trajectory = include_trajectory
        self.include_taint = include_taint
        self._taint_propagator = TaintPropagator() if include_taint else None

    def extract(
        self,
        graph: CausalGraph,
        trajectory: Optional[Trajectory] = None
    ) -> FeatureVector:
        """
        Extract all features from a causal graph.

        Args:
            graph: CausalGraph instance
            trajectory: Optional Trajectory for additional context (used in state features)

        Returns:
            FeatureVector with all extracted features
        """
        # === Chain Features ===
        chain_depth = compute_chain_depth(graph)
        avg_chain_length = compute_avg_chain_length(graph)
        num_chains = count_chains(graph)

        # === Domain Features ===
        num_cross_domain_edges = count_cross_domain_edges(graph)
        cross_domain_ratio = compute_cross_domain_ratio(graph)
        unique_domains = count_unique_domains(graph)
        domain_entropy = compute_domain_entropy(graph)

        # Get cross-domain edge type breakdown
        cross_domain_types = get_cross_domain_edge_types(graph)

        # === State Features ===
        state_accumulation_score = compute_state_accumulation_score(graph, trajectory)
        unused_state_ratio = compute_unused_state_ratio(graph, trajectory)
        num_state_producing = count_state_producing_actions(graph)

        # === Bottleneck Features ===
        max_bottleneck_score = compute_max_bottleneck_score(graph)
        avg_bottleneck_score = compute_avg_bottleneck_score(graph)
        num_critical_bottlenecks = len(identify_critical_bottlenecks(graph, threshold=5))

        # === Edge Type Distribution ===
        edge_type_counts = self._count_edge_types(graph)
        total_edges = graph.num_edges()

        data_dependency_ratio = edge_type_counts["data_dependency"] / total_edges if total_edges > 0 else 0.0
        trust_transfer_ratio = edge_type_counts["trust_transfer"] / total_edges if total_edges > 0 else 0.0
        state_enablement_ratio = edge_type_counts["state_enablement"] / total_edges if total_edges > 0 else 0.0

        # === Graph Topology ===
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        graph_density = self._compute_graph_density(graph)

        # === Taint Analysis Features ===
        taint_result = self._compute_taint_features(graph, trajectory)

        # === Watermark Features ===
        watermark_coverage = compute_watermark_coverage(graph)
        watermark_propagation_depth = compute_watermark_propagation_depth(graph)
        watermark_sensitive_coverage = compute_watermark_sensitive_coverage(graph)
        watermark_tampered = compute_watermark_tampered(graph)
        watermark_gap_count = compute_watermark_gap_count(graph)
        watermark_first_gap_depth = compute_watermark_first_gap_depth(graph)

        # Build feature vector
        return FeatureVector(
            # Chain
            chain_depth=chain_depth,
            avg_chain_length=avg_chain_length,
            num_chains=num_chains,
            # Domain
            num_cross_domain_edges=num_cross_domain_edges,
            cross_domain_ratio=cross_domain_ratio,
            unique_domains=unique_domains,
            domain_entropy=domain_entropy,
            # State
            state_accumulation_score=state_accumulation_score,
            unused_state_ratio=unused_state_ratio,
            num_state_producing=num_state_producing,
            # Bottleneck
            max_bottleneck_score=max_bottleneck_score,
            avg_bottleneck_score=avg_bottleneck_score,
            num_critical_bottlenecks=num_critical_bottlenecks,
            # Edge types
            data_dependency_ratio=data_dependency_ratio,
            trust_transfer_ratio=trust_transfer_ratio,
            state_enablement_ratio=state_enablement_ratio,
            # Cross-domain edge types
            cross_domain_data_dependency=cross_domain_types["data_dependency"],
            cross_domain_trust_transfer=cross_domain_types["trust_transfer"],
            cross_domain_state_enablement=cross_domain_types["state_enablement"],
            # Topology
            num_nodes=num_nodes,
            num_edges=num_edges,
            graph_density=graph_density,
            # Taint features
            taint_reaches_execution=taint_result.taint_reaches_execution if taint_result else False,
            taint_violation_count=taint_result.taint_violation_count if taint_result else 0,
            max_taint_path_length=taint_result.max_taint_path_length if taint_result else 0,
            tainted_node_count=taint_result.tainted_node_count if taint_result else 0,
            sensitive_sink_count=taint_result.sensitive_sink_count if taint_result else 0,
            untrusted_source_count=taint_result.untrusted_source_count if taint_result else 0,
            # Watermark features
            watermark_coverage=watermark_coverage,
            watermark_propagation_depth=watermark_propagation_depth,
            watermark_sensitive_coverage=watermark_sensitive_coverage,
            watermark_tampered=watermark_tampered,
            watermark_gap_count=watermark_gap_count,
            watermark_first_gap_depth=watermark_first_gap_depth,
        )

    def extract_batch(
        self,
        graphs: List[CausalGraph],
        trajectories: Optional[List[Trajectory]] = None
    ) -> List[FeatureVector]:
        """
        Extract features from multiple graphs.

        Args:
            graphs: List of CausalGraph instances
            trajectories: Optional list of Trajectory instances (must match length of graphs)

        Returns:
            List of FeatureVectors
        """
        if trajectories is not None and len(graphs) != len(trajectories):
            raise ValueError("Number of graphs and trajectories must match")

        features = []
        for i, graph in enumerate(graphs):
            trajectory = trajectories[i] if trajectories else None
            features.append(self.extract(graph, trajectory))

        return features

    def _count_edge_types(self, graph: CausalGraph) -> Dict[str, int]:
        """Count edges by type."""
        counts = {
            "data_dependency": 0,
            "trust_transfer": 0,
            "state_enablement": 0,
        }

        for edge in graph.get_all_edges():
            if edge.edge_type == EdgeType.DATA_DEPENDENCY:
                counts["data_dependency"] += 1
            elif edge.edge_type == EdgeType.TRUST_TRANSFER:
                counts["trust_transfer"] += 1
            elif edge.edge_type == EdgeType.STATE_ENABLEMENT:
                counts["state_enablement"] += 1

        return counts

    def _compute_graph_density(self, graph: CausalGraph) -> float:
        """
        Compute graph density: edges / (nodes * (nodes - 1)).

        For directed graphs, maximum edges = n * (n - 1).
        """
        n = graph.num_nodes()
        if n <= 1:
            return 0.0

        max_edges = n * (n - 1)
        return graph.num_edges() / max_edges if max_edges > 0 else 0.0

    def _compute_taint_features(
        self,
        graph: CausalGraph,
        trajectory: Optional[Trajectory],
    ) -> Optional[TaintAnalysisResult]:
        """
        Compute taint analysis features.

        Args:
            graph: CausalGraph instance
            trajectory: Trajectory for action metadata (optional if graph has stored metadata)

        Returns:
            TaintAnalysisResult or None if taint analysis is disabled
        """
        if not self.include_taint or self._taint_propagator is None:
            return None

        node_metadata = None

        # First, try to use trajectory if provided
        if trajectory is not None:
            node_metadata = {}
            for i, action in enumerate(trajectory.actions):
                # Use action_id if available, otherwise use index
                node_id = getattr(action, 'action_id', i)

                node_metadata[node_id] = {
                    "action_type": action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type),
                    "agent_name": action.context.get("agent_name") if action.context else None,
                    "data_produced": action.data_produced,
                    "data_consumed": action.data_consumed,
                    "result": action.result,
                }

        # Fallback: Try to use stored metadata from graph
        if node_metadata is None:
            stored_metadata = graph.get_metadata("node_metadata_for_taint")
            if stored_metadata and isinstance(stored_metadata, dict) and len(stored_metadata) > 0:
                node_metadata = stored_metadata

        # Still no metadata? Try to reconstruct from node attributes
        if node_metadata is None:
            node_metadata = {}
            for node_id in graph.graph.nodes():
                node_data = graph.get_node_data(node_id) or {}
                # Include provenance info for taint analysis
                provenance = node_data.get("provenance", {})
                node_metadata[node_id] = {
                    "action_type": node_data.get("action_type", "unknown"),
                    "agent_name": node_data.get("agent_name"),
                    "data_produced": node_data.get("data_produced", []),
                    "data_consumed": node_data.get("data_consumed", []),
                    "result": node_data.get("result_preview", ""),
                    # Add provenance flags for trust determination
                    "is_untrusted": provenance.get("is_untrusted", False),
                    "injection_detected": provenance.get("injection_detected", False),
                }

        # If no nodes, cannot run taint analysis
        if not node_metadata:
            return None

        # Get the underlying NetworkX graph
        nx_graph = graph.graph

        # Run taint analysis
        return self._taint_propagator.analyze(nx_graph, node_metadata)
