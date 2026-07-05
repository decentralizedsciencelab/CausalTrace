"""
Causal graph construction module.

This module provides tools for constructing and analyzing causal graphs
from agent trajectories.
"""

from causaltrace.graph.causal_graph import CausalGraph, CausalEdge, EdgeType
from causaltrace.graph.graph_builder import GraphBuilder
from causaltrace.graph.edge_detector import (
    DataDependencyDetector,
    TrustTransferDetector,
    StateEnablementDetector,
    EdgeDetector,
    DetectorConfig,
    DEFAULT_CONFIG,
)
from causaltrace.graph.edge_inference import (
    EdgeInferenceEngine,
    MASEdgeInferenceEngine,
    RealAgentEdgeInferenceEngine,
    InferenceResult,
)
from causaltrace.graph.agentdojo_inference import (
    AgentDojoEdgeInferenceEngine,
)
from causaltrace.graph.validators import (
    validate_dag,
    validate_edge_consistency,
    validate_edge_types,
    validate_graph_metrics
)
from causaltrace.graph.causal_inference import (
    StructuralCausalModel,
    CausalQuery,
    CausalEffect,
    detect_attack_causal,
    compare_detection_methods,
)
from causaltrace.graph.mechanism_integrity import (
    CausalMechanismAnalyzer,
    MechanismProfile,
    IntegrityResult,
    detect_mechanism_hijacking,
    compare_all_detection_methods,
)
from causaltrace.graph.mechanism_integrity_v2 import (
    CausalMechanismIntegrity,
    MechanismDeviationResult,
    detect_mechanism_integrity_v2,
)
from causaltrace.graph.mechanism_integrity_v3 import (
    CausalMechanismIntegrityV3,
    detect_mechanism_integrity_v3,
)

__all__ = [
    "CausalGraph",
    "CausalEdge",
    "EdgeType",
    "GraphBuilder",
    # Edge detectors
    "DataDependencyDetector",
    "TrustTransferDetector",
    "StateEnablementDetector",
    "EdgeDetector",  # Unified detector
    "DetectorConfig",
    "DEFAULT_CONFIG",
    # Inference engines
    "EdgeInferenceEngine",
    "MASEdgeInferenceEngine",
    "RealAgentEdgeInferenceEngine",
    "AgentDojoEdgeInferenceEngine",
    "InferenceResult",
    # Validators
    "validate_dag",
    "validate_edge_consistency",
    "validate_edge_types",
    "validate_graph_metrics",
    # Causal inference
    "StructuralCausalModel",
    "CausalQuery",
    "CausalEffect",
    "detect_attack_causal",
    "compare_detection_methods",
    # Mechanism integrity
    "CausalMechanismAnalyzer",
    "MechanismProfile",
    "IntegrityResult",
    "detect_mechanism_hijacking",
    "compare_all_detection_methods",
    # Mechanism integrity v2/v3
    "CausalMechanismIntegrity",
    "MechanismDeviationResult",
    "detect_mechanism_integrity_v2",
    "CausalMechanismIntegrityV3",
    "detect_mechanism_integrity_v3",
]
