"""
Enhanced backward slicing with temporal fallback for chain recovery.

Addresses the issue where graphs with poor edge detection (e.g., BrowserGym)
have zero causal edges, making standard slicing ineffective.

Key enhancement:
- If causal graph has no edges, fall back to temporal ordering
- If all nodes are marked as injection (degenerate), use heuristics
- Provides robust chain recovery even for heterogeneous benchmarks
"""

from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
import networkx as nx

from causaltrace.models.trajectory import Action, Trajectory, ActionType
from causaltrace.graph.causal_graph import CausalGraph, CausalEdge, EdgeType
from causaltrace.graph.slicer import BackwardSlicer, SliceResult


@dataclass
class EnhancedSliceResult(SliceResult):
    """Enhanced slice result with additional metadata."""
    fallback_used: str = "none"  # "none", "temporal", "heuristic"
    confidence: float = 1.0  # Lower for fallback methods
    causal_edges_found: int = 0
    temporal_edges_added: int = 0


class EnhancedBackwardSlicer(BackwardSlicer):
    """
    Enhanced backward slicer with temporal fallback.

    When causal edges are missing, falls back to temporal ordering
    to still enable chain recovery.
    """

    # Action types that typically indicate injection sources
    INJECTION_ACTION_TYPES = {
        ActionType.READ, ActionType.NAVIGATE, ActionType.WEB_FETCH,
        ActionType.EXTRACT
    }

    # Action types that typically indicate sinks (malicious actions)
    SINK_ACTION_TYPES = {
        ActionType.SUBMIT, ActionType.TOOL_CALL, ActionType.SEND_EMAIL,
        ActionType.CLICK, ActionType.TYPE, ActionType.WRITE,
        ActionType.CODE_EXECUTION
    }

    # Keywords indicating injection in content/target
    INJECTION_KEYWORDS = [
        'inject', 'payload', 'malicious', 'attacker', 'untrusted',
        'attention', 'objective', 'ignore previous', 'new instruction',
        'you must', 'urgent', 'important:', 'security update'
    ]

    # Keywords indicating sensitive/sink actions
    SINK_KEYWORDS = [
        'exfil', 'transfer', 'send', 'post', 'submit', 'delete',
        'token', 'credential', 'password', 'api_key', 'ssh',
        'attacker.com', 'evil.com', 'backup-service'
    ]

    def __init__(self, graph: CausalGraph, trajectory: Trajectory):
        super().__init__(graph, trajectory)
        self._has_causal_edges = len(list(graph.iter_edges())) > 0

    def _build_temporal_graph(self) -> nx.DiGraph:
        """Build a graph with temporal edges (sequential ordering)."""
        G = nx.DiGraph()

        # Add all nodes
        for action in self.trajectory.actions:
            G.add_node(action.action_id)

        # Add sequential edges (i -> i+1)
        sorted_actions = sorted(self.trajectory.actions, key=lambda a: a.action_id)
        for i in range(len(sorted_actions) - 1):
            G.add_edge(
                sorted_actions[i].action_id,
                sorted_actions[i + 1].action_id
            )

        return G

    def _identify_injection_heuristic(self) -> List[int]:
        """
        Identify injection sources using heuristics when provenance is missing.

        Strategy:
        1. Look for READ/NAVIGATE actions with injection keywords
        2. Look for actions from untrusted domains
        3. Look for early actions in the trajectory (temporal assumption)
        """
        injection_candidates = []

        for action in self.trajectory.actions:
            score = 0

            # Check action type
            if action.action_type in self.INJECTION_ACTION_TYPES:
                score += 1

            # Check target/content for keywords
            target_lower = str(action.target or "").lower()
            result_lower = str(getattr(action, 'result', '') or "").lower()

            for kw in self.INJECTION_KEYWORDS:
                if kw in target_lower or kw in result_lower:
                    score += 2
                    break

            # Check domain
            domain = action.domain or ""
            untrusted_domains = ['forum', 'cdn', 'social', 'external', 'unknown']
            if any(ud in domain.lower() for ud in untrusted_domains):
                score += 1

            # Prefer earlier actions (temporal assumption)
            if action.action_id < len(self.trajectory.actions) // 2:
                score += 0.5

            if score >= 2:
                injection_candidates.append((action.action_id, score))

        # Sort by score and return top candidates
        injection_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return at least one injection source (the first action)
        if not injection_candidates:
            return [min(a.action_id for a in self.trajectory.actions)]

        # Return top injection candidates (at most 3)
        return [aid for aid, _ in injection_candidates[:3]]

    def _identify_sink_heuristic(self) -> List[int]:
        """
        Identify sink nodes using heuristics when not explicitly marked.

        Strategy:
        1. Look for SUBMIT/EXECUTE/TOOL_CALL actions
        2. Look for actions with sink keywords
        3. Look for later actions in the trajectory
        """
        sink_candidates = []

        for action in self.trajectory.actions:
            score = 0

            # Check action type
            if action.action_type in self.SINK_ACTION_TYPES:
                score += 1

            # Check target for sink keywords
            target_lower = str(action.target or "").lower()

            for kw in self.SINK_KEYWORDS:
                if kw in target_lower:
                    score += 2
                    break

            # Prefer later actions
            if action.action_id > len(self.trajectory.actions) // 2:
                score += 0.5

            if score >= 1:
                sink_candidates.append((action.action_id, score))

        # Sort by score
        sink_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return at least the last action
        if not sink_candidates:
            return [max(a.action_id for a in self.trajectory.actions)]

        return [aid for aid, _ in sink_candidates[:3]]

    def _is_degenerate_ground_truth(self, injection_sources: List[int]) -> bool:
        """
        Check if ground truth is degenerate (all nodes are injection sources).
        """
        all_node_ids = set(a.action_id for a in self.trajectory.actions)
        return set(injection_sources) == all_node_ids or \
               len(injection_sources) > len(all_node_ids) * 0.8

    def enhanced_backward_slice(
        self,
        target_action_id: int,
        max_depth: int = 50,
        min_bottleneck_downstream: int = 2,
    ) -> EnhancedSliceResult:
        """
        Perform enhanced backward slicing with temporal fallback.

        Args:
            target_action_id: Action to slice from
            max_depth: Maximum backward traversal depth
            min_bottleneck_downstream: Minimum downstream for bottleneck

        Returns:
            EnhancedSliceResult with complete analysis
        """
        causal_edges_count = len(list(self.graph.iter_edges()))
        fallback_used = "none"
        confidence = 1.0
        temporal_edges_added = 0

        # Try standard causal slicing first
        if self._has_causal_edges:
            result = super().backward_slice(
                target_action_id, max_depth, min_bottleneck_downstream,
                use_bidirectional=True
            )

            # Check if result is meaningful
            if result.attack_chain and len(result.attack_chain) > 1:
                return EnhancedSliceResult(
                    target_action_id=result.target_action_id,
                    slice_actions=result.slice_actions,
                    attack_chain=result.attack_chain,
                    injection_sources=result.injection_sources,
                    bottlenecks=result.bottlenecks,
                    explanation=result.explanation,
                    fallback_used="none",
                    confidence=1.0,
                    causal_edges_found=causal_edges_count,
                    temporal_edges_added=0
                )

        # Fallback: Use temporal ordering
        fallback_used = "temporal"
        confidence = 0.7

        # Build temporal graph
        temporal_graph = self._build_temporal_graph()
        temporal_edges_added = temporal_graph.number_of_edges()

        # Find injection sources
        injection_sources = self.find_injection_sources_in_slice(
            set(a.action_id for a in self.trajectory.actions)
        )

        # If degenerate (all nodes are injection), use heuristics
        if self._is_degenerate_ground_truth(injection_sources):
            injection_sources = self._identify_injection_heuristic()
            fallback_used = "heuristic"
            confidence = 0.5

        # Find sinks if target is not informative
        if target_action_id not in [a.action_id for a in self.trajectory.actions]:
            sinks = self._identify_sink_heuristic()
            target_action_id = sinks[0] if sinks else max(
                a.action_id for a in self.trajectory.actions
            )

        # Compute slice on temporal graph
        slice_actions = set()
        for source in injection_sources:
            try:
                path = nx.shortest_path(temporal_graph, source, target_action_id)
                slice_actions.update(path)
            except nx.NetworkXNoPath:
                # If no path, include all actions between source and target
                for a in self.trajectory.actions:
                    if source <= a.action_id <= target_action_id:
                        slice_actions.add(a.action_id)

        # Extract attack chain (shortest path from any injection to target)
        attack_chain = []
        shortest_length = float('inf')
        for source in injection_sources:
            try:
                path = nx.shortest_path(temporal_graph, source, target_action_id)
                if len(path) < shortest_length:
                    attack_chain = path
                    shortest_length = len(path)
            except nx.NetworkXNoPath:
                continue

        # If no path found, use temporal ordering
        if not attack_chain:
            min_inj = min(injection_sources) if injection_sources else 0
            attack_chain = [
                a.action_id for a in sorted(self.trajectory.actions, key=lambda x: x.action_id)
                if min_inj <= a.action_id <= target_action_id
            ]

        # Identify bottlenecks (in temporal graph, middle nodes)
        bottlenecks = []
        if len(attack_chain) > 2:
            # Middle node(s) are bottlenecks in temporal chain
            mid = len(attack_chain) // 2
            bottlenecks = [attack_chain[mid]]

        # Generate explanation
        explanation = self._generate_enhanced_explanation(
            target_action_id, attack_chain, injection_sources,
            bottlenecks, fallback_used
        )

        return EnhancedSliceResult(
            target_action_id=target_action_id,
            slice_actions=sorted(list(slice_actions)),
            attack_chain=attack_chain,
            injection_sources=injection_sources,
            bottlenecks=bottlenecks,
            explanation=explanation,
            fallback_used=fallback_used,
            confidence=confidence,
            causal_edges_found=causal_edges_count,
            temporal_edges_added=temporal_edges_added
        )

    def _generate_enhanced_explanation(
        self,
        target_action_id: int,
        attack_chain: List[int],
        injection_sources: List[int],
        bottlenecks: List[int],
        fallback_used: str
    ) -> str:
        """Generate explanation with fallback notice."""
        lines = []

        if fallback_used != "none":
            lines.append(f"WARNING: FALLBACK USED: {fallback_used}")
            lines.append(f"    (Causal edges were missing; using temporal ordering)")
            lines.append("")

        lines.append(f"Attack Chain Analysis for Action {target_action_id}")
        lines.append("=" * 60)

        # Target action
        target_action = self.action_map.get(target_action_id)
        if target_action:
            lines.append(f"\nTarget (Sink) Action:")
            lines.append(f"  [{target_action_id}] {target_action.action_type.value}: {target_action.target}")

        # Injection sources
        if injection_sources:
            lines.append(f"\nInjection Sources ({len(injection_sources)}):")
            for source_id in injection_sources[:3]:
                source_action = self.action_map.get(source_id)
                if source_action:
                    lines.append(f"  [{source_id}] {source_action.action_type.value}: {source_action.target}")

        # Attack chain
        if attack_chain:
            lines.append(f"\nRecovered Chain ({len(attack_chain)} actions):")
            for i, action_id in enumerate(attack_chain):
                action = self.action_map.get(action_id)
                if action:
                    marker = "->" if i < len(attack_chain) - 1 else "X"
                    lines.append(f"  [{action_id}] {action.action_type.value} {marker}")

        return "\n".join(lines)


def enhanced_backward_slice(
    graph: CausalGraph,
    trajectory: Trajectory,
    target_action_id: int,
) -> EnhancedSliceResult:
    """
    Convenience function for enhanced backward slicing.

    Args:
        graph: Causal graph
        trajectory: Full trajectory
        target_action_id: Action to slice from

    Returns:
        EnhancedSliceResult
    """
    slicer = EnhancedBackwardSlicer(graph, trajectory)
    return slicer.enhanced_backward_slice(target_action_id)
