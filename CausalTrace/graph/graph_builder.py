"""
Graph builder for constructing causal graphs from trajectories.
"""

import json
import os
import fnmatch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from causaltrace.models.trajectory import Trajectory
from causaltrace.graph.causal_graph import CausalGraph, CausalEdge, EdgeType
from causaltrace.graph.edge_detector import (
    DataDependencyDetector,
    TrustTransferDetector,
    StateEnablementDetector
)
from causaltrace.graph.edge_inference import (
    EdgeInferenceEngine,
    MASEdgeInferenceEngine,
    RealAgentEdgeInferenceEngine
)
from causaltrace.graph.agentdojo_inference import AgentDojoEdgeInferenceEngine
from causaltrace.graph.validators import validate_dag, validate_edge_consistency
from causaltrace.inference import DataFlowInferencer


class GraphBuilder:
    """
    Build causal graphs from trajectories.

    Supports two modes:
    - Standard: rule-based edge detectors (default)
    - Inference: EdgeInferenceEngine for sparse annotations
    """

    def __init__(
        self,
        use_inference: bool = False,
        use_mas_inference: bool = False,
        use_real_agent_inference: bool = False,
        use_agentdojo_inference: bool = False,
        inference_engine: Optional[EdgeInferenceEngine] = None,
        watermark_token: Optional[str] = None,
        sensitive_config_path: Optional[str] = None,
        auto_infer_data_flow: bool = True,
        auto_detect_source: bool = True,
    ):
        """
        Initialize the graph builder with edge detectors.

        Args:
            use_inference: If True, use EdgeInferenceEngine for sparse data
            use_mas_inference: If True, use MASEdgeInferenceEngine (implies use_inference)
            use_real_agent_inference: If True, use RealAgentEdgeInferenceEngine
            use_agentdojo_inference: If True, use AgentDojoEdgeInferenceEngine
            inference_engine: Custom inference engine (if None, creates default)
            watermark_token: Optional watermark token for propagation tracking
            sensitive_config_path: Optional path to sensitive action config
            auto_infer_data_flow: If True, automatically infer data_produced/data_consumed
                when trajectory lacks these annotations (default: True)
            auto_detect_source: If True, automatically select inference engine based on
                trajectory source (e.g., "agentdojo" -> AgentDojoEdgeInferenceEngine)
        """
        self.auto_infer_data_flow = auto_infer_data_flow
        self.auto_detect_source = auto_detect_source
        self._data_flow_inferencer = DataFlowInferencer() if auto_infer_data_flow else None
        # Standard detectors
        self.data_detector = DataDependencyDetector()
        self.trust_detector = TrustTransferDetector()
        self.state_detector = StateEnablementDetector()

        # Inference engine for sparse data
        self.use_inference = use_inference or use_mas_inference or use_real_agent_inference or use_agentdojo_inference
        self.use_agentdojo_inference = use_agentdojo_inference
        if inference_engine:
            self.inference_engine = inference_engine
        elif use_agentdojo_inference:
            self.inference_engine = AgentDojoEdgeInferenceEngine()
        elif use_real_agent_inference:
            self.inference_engine = RealAgentEdgeInferenceEngine()
        elif use_mas_inference:
            self.inference_engine = MASEdgeInferenceEngine()
        elif use_inference:
            self.inference_engine = EdgeInferenceEngine()
        else:
            self.inference_engine = None

        # Watermark configuration
        self.watermark_token = watermark_token or os.getenv("CAUSALTRACE_WATERMARK_TOKEN")
        config_env = os.getenv("CAUSALTRACE_SENSITIVE_CONFIG")
        config_path = sensitive_config_path or config_env
        self.sensitive_actions_config = self._load_sensitive_actions_config(config_path)
        self._has_sensitive_rules = any(
            bool(self.sensitive_actions_config.get(key, []))
            for key in ("always_action_types", "conditional_action_types", "target_domains", "keywords")
        )

    def build(self, trajectory: Trajectory) -> CausalGraph:
        """
        Build complete causal graph from trajectory.

        Algorithm:
        1. Create CausalGraph with action nodes
        2. If using inference engine:
           - Use EdgeInferenceEngine to detect all edges
        3. Otherwise, for each pair of actions (i, j) where i < j:
           a. Check for data dependency i → j
           b. Check for trust transfer i → j
           c. Check for state enablement i → j
        4. Add detected edges to graph
        5. Validate DAG property (no cycles)
        6. Tag watermark and run taint-based provenance tracking
        7. Return graph

        Args:
            trajectory: The trajectory to build graph from

        Returns:
            CausalGraph instance with detected edges
        """
        # Step 0: Infer data flow if annotations are missing
        if self.auto_infer_data_flow and self._data_flow_inferencer:
            # Check if trajectory lacks data annotations
            has_annotations = any(
                a.data_produced or a.data_consumed
                for a in trajectory.actions
            )
            if not has_annotations:
                trajectory = self._data_flow_inferencer.infer_data_flow(trajectory)

        # Step 1: Create graph with nodes
        graph = CausalGraph(trajectory)

        # Step 2: Auto-detect trajectory source and select appropriate inference engine
        inference_engine = self.inference_engine
        use_inference = self.use_inference

        if self.auto_detect_source and not inference_engine:
            # Check trajectory source
            source = getattr(trajectory, 'source', None) or ''
            if source.lower() == 'agentdojo':
                inference_engine = AgentDojoEdgeInferenceEngine()
                use_inference = True

        # Step 3: Detect edges (using inference or standard detectors)
        edges = []

        if use_inference and inference_engine:
            # Use edge inference engine for sparse data
            inference_result = inference_engine.infer_edges(trajectory)
            edges = inference_result.edges
        else:
            # Standard edge detection with priority:
            # DATA_DEPENDENCY > TRUST_TRANSFER > STATE_ENABLEMENT
            # Only one edge per pair (since NetworkX DiGraph doesn't support multi-edges)
            seen_pairs = set()

            for i, source in enumerate(trajectory.actions):
                for j, target in enumerate(trajectory.actions[i + 1:], start=i + 1):
                    pair = (source.action_id, target.action_id)

                    # Check for data dependency (highest priority)
                    data_edge = self.data_detector.detect(source, target, trajectory)
                    if data_edge:
                        edges.append(data_edge)
                        seen_pairs.add(pair)
                        continue  # Skip lower-priority edges for this pair

                    # Check for trust transfer (medium priority)
                    trust_edge = self.trust_detector.detect(source, target, trajectory)
                    if trust_edge:
                        edges.append(trust_edge)
                        seen_pairs.add(pair)
                        continue  # Skip lower-priority edges for this pair

                    # Check for state enablement (lowest priority)
                    state_edge = self.state_detector.detect(source, target, trajectory)
                    if state_edge:
                        edges.append(state_edge)
                        seen_pairs.add(pair)

        # Step 3: Add edges to graph
        for edge in edges:
            graph.add_edge(edge)

        # Step 4: Validate graph
        if not validate_dag(graph):
            # Log warning but continue (in practice, our construction should not create cycles)
            print(f"Warning: Graph for trajectory {trajectory.trajectory_id} is not a DAG")

        # Validate edge consistency
        issues = validate_edge_consistency(graph)
        if issues:
            print(f"Warning: Edge consistency issues in trajectory {trajectory.trajectory_id}:")
            for issue in issues:
                print(f"  - {issue}")

        # Tag watermark propagation and run invariant checks if configured
        self._tag_watermark_nodes(graph, trajectory)
        self._check_watermark_invariant(graph, trajectory)

        # NEW: Provenance-based watermark tagging (works without explicit token)
        self._tag_provenance_watermark(graph, trajectory)

        # NEW: Store trajectory metadata in graph for later taint analysis
        self._store_node_metadata(graph, trajectory)

        return graph

    def build_batch(self, trajectories: List[Trajectory]) -> List[CausalGraph]:
        """
        Build graphs for multiple trajectories.

        Args:
            trajectories: List of trajectories to process

        Returns:
            List of causal graphs, one per trajectory
        """
        graphs = []
        for trajectory in trajectories:
            graph = self.build(trajectory)
            graphs.append(graph)
        return graphs

    def build_with_stats(self, trajectory: Trajectory) -> Tuple[CausalGraph, Dict]:
        """
        Build graph and return construction statistics.

        Args:
            trajectory: The trajectory to build graph from

        Returns:
            Tuple of (CausalGraph, statistics dict)
        """
        graph = self.build(trajectory)

        stats = {
            "num_actions": len(trajectory.actions),
            "num_nodes": graph.num_nodes(),
            "num_edges": graph.num_edges(),
            "num_data_dependencies": len(graph.get_edges_by_type(
                EdgeType.DATA_DEPENDENCY
            )),
            "num_trust_transfers": len(graph.get_edges_by_type(
                EdgeType.TRUST_TRANSFER
            )),
            "num_state_enablements": len(graph.get_edges_by_type(
                EdgeType.STATE_ENABLEMENT
            )),
            "longest_path": graph.longest_path_length(),
            "num_cross_domain_edges": len(graph.get_cross_domain_edges()),
            "is_dag": validate_dag(graph)
        }

        return graph, stats

    # -------------------------------------------------------------------------
    # Watermark helpers
    # -------------------------------------------------------------------------

    def _load_sensitive_actions_config(self, path: Optional[str]) -> Dict[str, List[str]]:
        defaults = {
            "always_action_types": [],
            "conditional_action_types": [],
            "target_domains": [],
            "keywords": [],
        }
        candidate: Optional[Path] = None
        if path:
            candidate = Path(path)
        else:
            default_path = Path("experiments/watermark/sensitive_actions.json")
            if default_path.exists():
                candidate = default_path
        if not candidate:
            return defaults
        try:
            with candidate.open("r") as f:
                data = json.load(f)

            def flatten(value) -> List[str]:
                if value is None:
                    return []
                if isinstance(value, list):
                    flattened: List[str] = []
                    for item in value:
                        flattened.extend(flatten(item))
                    return flattened
                if isinstance(value, dict):
                    flattened: List[str] = []
                    for item in value.values():
                        flattened.extend(flatten(item))
                    return flattened
                return [str(value)]

            always_actions: List[str] = []
            conditional_actions: List[str] = []

            raw_actions = data.get("action_types", [])
            if isinstance(raw_actions, dict):
                # Support {"always": [...], "conditional": [...], "critical": [...], ...}
                for key, val in raw_actions.items():
                    if key.lower() == "conditional":
                        conditional_actions.extend(flatten(val))
                    else:
                        always_actions.extend(flatten(val))
            else:
                always_actions.extend(flatten(raw_actions))

            # Explicit conditional list overrides/adds
            conditional_actions.extend(flatten(data.get("conditional_action_types", [])))

            target_domains = flatten(data.get("target_domains", []))
            target_domains.extend(flatten(data.get("target_domain_groups", [])))
            keywords = flatten(data.get("keywords", []))
            keywords.extend(flatten(data.get("keyword_groups", [])))

            return {
                "always_action_types": sorted({s.lower() for s in always_actions}),
                "conditional_action_types": sorted({s.lower() for s in conditional_actions}),
                "target_domains": sorted({s.lower() for s in target_domains}),
                "keywords": sorted({s.lower() for s in keywords}),
            }
        except Exception as exc:
            print(f"Warning: unable to load sensitive action config from {candidate}: {exc}")
            return defaults

    def _tag_watermark_nodes(self, graph: CausalGraph, trajectory: Trajectory) -> None:
        """Mark nodes that contain or inherit the watermark token."""
        if not self.watermark_token:
            return

        token = self.watermark_token.lower()
        tagged: Dict[int, bool] = {}

        for action in trajectory.actions:
            node_data = graph.get_node_data(action.action_id) or {}
            text_fields = [
                action.target or "",
                action.result or "",
            ]
            if action.context:
                try:
                    text_fields.append(json.dumps(action.context, sort_keys=True))
                except Exception:
                    text_fields.append(str(action.context))
            if action.raw_data:
                try:
                    text_fields.append(json.dumps(action.raw_data, sort_keys=True))
                except Exception:
                    text_fields.append(str(action.raw_data))
            for chunk in node_data.get("observation_chunks", []):
                text_fields.append(chunk.get("content", ""))

            has_token = any(token in (field or "").lower() for field in text_fields if field)
            graph.set_node_attribute(action.action_id, "watermark_tagged", has_token)
            tagged[action.action_id] = has_token

        # Propagate along data dependency + trust transfer edges
        changed = True
        while changed:
            changed = False
            for edge in graph.get_all_edges():
                edge_type = edge.edge_type
                if isinstance(edge_type, str):
                    try:
                        edge_type = EdgeType(edge_type)
                    except ValueError:
                        edge_type = None
                if edge_type not in (EdgeType.DATA_DEPENDENCY, EdgeType.TRUST_TRANSFER):
                    continue
                if tagged.get(edge.source_action_id) and not tagged.get(edge.target_action_id):
                    tagged[edge.target_action_id] = True
                    graph.set_node_attribute(edge.target_action_id, "watermark_tagged", True)
                    changed = True

        graph.set_metadata("watermark_token", self.watermark_token)
        graph.set_metadata("watermark_tagged_nodes", [n for n, flagged in tagged.items() if flagged])

    def _check_watermark_invariant(self, graph: CausalGraph, trajectory: Trajectory) -> None:
        """Ensure all sensitive actions have watermark lineage."""
        if not self.watermark_token or not self._has_sensitive_rules:
            return

        sensitive_nodes: List[int] = []
        tampered_nodes: List[int] = []

        for action in trajectory.actions:
            if self._is_sensitive_action(action):
                sensitive_nodes.append(action.action_id)
                node_data = graph.get_node_data(action.action_id) or {}
                if not node_data.get("watermark_tagged"):
                    tampered_nodes.append(action.action_id)

        graph.set_metadata("watermark_sensitive_nodes", sensitive_nodes)
        graph.set_metadata("watermark_tampered_nodes", tampered_nodes)
        graph.set_metadata("watermark_tampered", bool(tampered_nodes))

    def _is_sensitive_action(self, action) -> bool:
        cfg = self.sensitive_actions_config or {}
        always_action_types = set(cfg.get("always_action_types", []))
        conditional_action_types = set(cfg.get("conditional_action_types", []))
        target_domains = cfg.get("target_domains", [])
        keywords = cfg.get("keywords", [])

        action_type_value = None
        if action.action_type:
            if hasattr(action.action_type, "value"):
                action_type_value = str(action.action_type.value).lower()
            else:
                action_type_value = str(action.action_type).lower()

        if action_type_value and action_type_value in always_action_types:
            return True

        domain = (action.domain or "").lower()
        domain_match = False
        for pattern in target_domains:
            if fnmatch.fnmatch(domain, pattern):
                domain_match = True
                break

        text_blob_parts = [action.target or "", action.result or ""]
        if action.context:
            try:
                text_blob_parts.append(json.dumps(action.context, sort_keys=True))
            except Exception:
                text_blob_parts.append(str(action.context))
        text_blob = " ".join(text_blob_parts).lower()

        keyword_match = False
        for keyword in keywords:
            if keyword in text_blob:
                keyword_match = True
                break

        if action_type_value and action_type_value in conditional_action_types and (domain_match or keyword_match):
            return True

        if domain_match or keyword_match:
            return True

        return False

    # -------------------------------------------------------------------------
    # Provenance-based watermark tagging (works without explicit token)
    # -------------------------------------------------------------------------

    def _tag_provenance_watermark(self, graph: CausalGraph, trajectory: Trajectory) -> None:
        """
        Tag nodes based on action provenance (trust level).

        This enables watermark features to be computed without an explicit
        watermark token by inferring provenance from action metadata.

        Trust sources (tagged as watermarked):
        - User input actions (trust_level >= 0.8 or context indicates user)
        - System/orchestrator actions
        - Actions with trusted domain origins

        Untrusted sources (not tagged):
        - Web fetch from external/untrusted domains
        - Actions consuming data from untrusted sources
        - Forum posts, comments, external content
        """
        # Skip if explicit watermark tagging already done
        existing_tagged = graph.get_metadata("watermark_tagged_nodes")
        if existing_tagged and len(existing_tagged) > 0:
            return

        tagged: Dict[int, bool] = {}
        sensitive_nodes: List[int] = []

        # Untrusted domains and patterns
        untrusted_patterns = [
            "forum", "reddit", "social", "comment", "post",
            "attacker", "evil", "malicious", "external",
            "*.foo.com", "*.attacker.com",
        ]

        # Sensitive action types (sinks that need provenance verification)
        sensitive_action_types = {
            "code_execution", "execute_python", "execute_bash",
            "tool_call", "send_email", "submit_form", "post_comment",
            "api_call", "write_file", "database_write",
        }

        for action in trajectory.actions:
            node_id = action.action_id

            # Get action type
            action_type_value = ""
            if action.action_type:
                if hasattr(action.action_type, "value"):
                    action_type_value = str(action.action_type.value).lower()
                else:
                    action_type_value = str(action.action_type).lower()

            # Check if sensitive action
            if action_type_value in sensitive_action_types or self._is_sensitive_action(action):
                sensitive_nodes.append(node_id)
                graph.set_node_attribute(node_id, "is_sensitive_sink", True)

            # Determine trust level from context
            trust_level = 0.5  # Default: unknown
            if action.context:
                ctx_trust = action.context.get("trust_level")
                if ctx_trust is not None:
                    try:
                        trust_level = float(ctx_trust)
                    except (ValueError, TypeError):
                        pass

            # Check for untrusted source indicators
            is_untrusted = False
            domain = (action.domain or "").lower()
            result_text = (action.result or "").lower()

            # Check for explicit is_injection_source flag (from CausalBench)
            if action.context:
                if action.context.get("is_injection_source"):
                    is_untrusted = True
                    trust_level = min(trust_level, 0.1)

            # Check domain patterns
            for pattern in untrusted_patterns:
                if pattern.startswith("*"):
                    if domain.endswith(pattern[1:]):
                        is_untrusted = True
                        break
                elif pattern in domain:
                    is_untrusted = True
                    break

            # Check for injection patterns in result
            injection_indicators = [
                "attention", "new instruction", "ignore previous",
                "objective:", "important:", "urgent:",
            ]
            for indicator in injection_indicators:
                if indicator in result_text:
                    is_untrusted = True
                    trust_level = min(trust_level, 0.2)
                    break

            # Web fetch from external domains is untrusted
            if action_type_value in ("web_fetch", "navigate", "fetch"):
                if not any(trusted in domain for trusted in ["internal", "corp", "localhost"]):
                    is_untrusted = True
                    trust_level = min(trust_level, 0.3)

            # High trust actions are tagged
            is_tagged = trust_level >= 0.7 and not is_untrusted

            tagged[node_id] = is_tagged
            graph.set_node_attribute(node_id, "watermark_tagged", is_tagged)
            graph.set_node_attribute(node_id, "provenance_trust_level", trust_level)
            graph.set_node_attribute(node_id, "is_untrusted_source", is_untrusted)

        # Propagate watermark along trusted edges
        changed = True
        iterations = 0
        max_iterations = len(trajectory.actions) * 2
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for edge in graph.get_all_edges():
                edge_type = edge.edge_type
                if isinstance(edge_type, str):
                    try:
                        edge_type = EdgeType(edge_type)
                    except ValueError:
                        edge_type = None

                # Only propagate through data dependency edges from tagged nodes
                if edge_type == EdgeType.DATA_DEPENDENCY:
                    if tagged.get(edge.source_action_id) and not tagged.get(edge.target_action_id):
                        # Check target isn't explicitly untrusted
                        target_untrusted = graph.get_node_data(edge.target_action_id).get("is_untrusted_source", False)
                        if not target_untrusted:
                            tagged[edge.target_action_id] = True
                            graph.set_node_attribute(edge.target_action_id, "watermark_tagged", True)
                            changed = True

        # Identify tampered nodes (sensitive but not watermarked)
        tampered_nodes = [n for n in sensitive_nodes if not tagged.get(n, False)]

        # Update graph metadata
        watermark_tagged_list = [n for n, flagged in tagged.items() if flagged]

        # Only set if not already set by explicit token method
        if not existing_tagged:
            graph.set_metadata("watermark_tagged_nodes", watermark_tagged_list)
            graph.set_metadata("watermark_sensitive_nodes", sensitive_nodes)
            graph.set_metadata("watermark_tampered_nodes", tampered_nodes)
            graph.set_metadata("watermark_tampered", bool(tampered_nodes))
            graph.set_metadata("watermark_method", "provenance_inference")

    def _store_node_metadata(self, graph: CausalGraph, trajectory: Trajectory) -> None:
        """
        Store trajectory action metadata in graph for later taint analysis.

        This allows taint features to be computed when loading graphs from JSON
        without requiring the original trajectory object.
        """
        node_metadata = {}

        for action in trajectory.actions:
            node_id = action.action_id

            # Get action type value
            action_type_value = "unknown"
            if action.action_type:
                if hasattr(action.action_type, "value"):
                    action_type_value = str(action.action_type.value)
                else:
                    action_type_value = str(action.action_type)

            # Get agent name from context
            agent_name = None
            if action.context:
                agent_name = action.context.get("agent_name")

            # Store metadata
            node_metadata[node_id] = {
                "action_type": action_type_value,
                "agent_name": agent_name,
                "data_produced": action.data_produced or [],
                "data_consumed": action.data_consumed or [],
                "domain": action.domain,
                "result_preview": (action.result or "")[:500],  # Truncate for storage
            }

            # Also set as node attributes for easy access
            graph.set_node_attribute(node_id, "action_type", action_type_value)
            graph.set_node_attribute(node_id, "agent_name", agent_name)
            graph.set_node_attribute(node_id, "data_produced", action.data_produced or [])
            graph.set_node_attribute(node_id, "data_consumed", action.data_consumed or [])

        # Store in graph metadata for taint analysis
        graph.set_metadata("node_metadata_for_taint", node_metadata)
        graph.set_metadata("has_stored_metadata", True)
