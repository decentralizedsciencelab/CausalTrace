"""
Backward slicing for attack chain recovery.

Given a suspicious action, this module traces backward through the causal graph
to recover the minimal attack chain from injection source to malicious action.

Key concepts:
- Backward slice: Set of actions that causally influence a target action
- Attack chain: Minimal path from injection to malicious action
- Bottleneck: Critical action that enables entire downstream attack
"""

from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
import networkx as nx

from causaltrace.models.trajectory import Action, Trajectory
from causaltrace.graph.causal_graph import CausalGraph, CausalEdge, EdgeType


@dataclass
class SliceResult:
    """
    Result of backward slicing.

    Attributes:
        target_action_id: The action we sliced from
        slice_actions: Action IDs in the backward slice
        attack_chain: Minimal path from injection to target
        injection_sources: Action IDs containing injections
        bottlenecks: Critical actions (high fan-out)
        explanation: Human-readable explanation of the attack chain
    """
    target_action_id: int
    slice_actions: List[int]
    attack_chain: List[int]
    injection_sources: List[int]
    bottlenecks: List[int]
    explanation: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "target_action_id": self.target_action_id,
            "slice_actions": self.slice_actions,
            "attack_chain": self.attack_chain,
            "injection_sources": self.injection_sources,
            "bottlenecks": self.bottlenecks,
            "explanation": self.explanation,
        }


class BackwardSlicer:
    """
    Implements backward slicing on causal graphs.

    Backward slicing computes the set of actions that causally influence
    a target action by following edges backward through the graph.
    """

    def __init__(self, graph: CausalGraph, trajectory: Trajectory):
        """
        Initialize backward slicer.

        Args:
            graph: The causal graph to slice
            trajectory: The full trajectory (for accessing action details)
        """
        self.graph = graph
        self.trajectory = trajectory
        self.action_map = {a.action_id: a for a in trajectory.actions}

    def is_injection_source(self, action_id: int) -> bool:
        """
        Check if an action is an injection source.

        An action is an injection source if:
        1. Its provenance indicates injection was detected
        2. It comes from an untrusted domain
        3. Its data_consumed contains injection keywords
        4. Its observation contains injection keywords

        Args:
            action_id: Action to check

        Returns:
            True if action is injection source
        """
        action = self.action_map.get(action_id)
        if not action:
            return False

        # Check provenance
        if action.provenance:
            if action.provenance.injection_detected:
                return True
            if action.provenance.is_untrusted:
                return True

        # Check data_consumed for injection keywords
        if action.data_consumed:
            injection_keywords = ['inject', 'malicious', 'attacker', 'untrusted', 'payload']
            for data_item in action.data_consumed:
                data_str = str(data_item).lower()
                if any(kw in data_str for kw in injection_keywords):
                    return True

        # Check domain
        if action.domain and any(untrusted in action.domain for untrusted in ["forum", "cdn", "social", "attacker"]):
            return True

        return False

    def compute_backward_slice(
        self,
        target_action_id: int,
        max_depth: int = 50,
    ) -> Set[int]:
        """
        Compute backward slice from target action.

        A backward slice is the set of all actions that causally influence
        the target action, following edges backward through the DAG.

        Args:
            target_action_id: Starting point for backward slice
            max_depth: Maximum depth to traverse

        Returns:
            Set of action IDs in the backward slice
        """
        slice_set = set()
        frontier = {target_action_id}
        visited = set()

        while frontier and len(visited) < max_depth:
            current = frontier.pop()
            if current in visited:
                continue

            visited.add(current)
            slice_set.add(current)

            # Get all predecessors (actions that have edges TO current)
            predecessors = self.graph.get_predecessors(current)
            for pred_id in predecessors:
                if pred_id not in visited:
                    frontier.add(pred_id)

        return slice_set

    def compute_forward_slice(
        self,
        source_action_id: int,
        max_depth: int = 50,
    ) -> Set[int]:
        """
        Compute forward slice from source action.

        A forward slice is the set of all actions that are causally influenced
        BY the source action, following edges forward through the DAG.

        Args:
            source_action_id: Starting point for forward slice
            max_depth: Maximum depth to traverse

        Returns:
            Set of action IDs in the forward slice
        """
        slice_set = set()
        frontier = {source_action_id}
        visited = set()

        while frontier and len(visited) < max_depth:
            current = frontier.pop()
            if current in visited:
                continue

            visited.add(current)
            slice_set.add(current)

            # Get all successors (actions that current has edges TO)
            successors = self.graph.get_successors(current)
            for succ_id in successors:
                if succ_id not in visited:
                    frontier.add(succ_id)

        return slice_set

    def compute_bidirectional_slice(
        self,
        target_action_id: int,
        injection_sources: List[int] = None,
        max_depth: int = 50,
    ) -> Set[int]:
        """
        Compute bidirectional slice combining forward and backward analysis.

        This addresses the limitation where backward-only slicing misses nodes
        that don't have a causal path to the sink but are part of the attack.

        Strategy:
        1. Backward slice from sink to find all influencing actions
        2. Forward slice from injection sources to find all affected actions
        3. Union the results

        Args:
            target_action_id: Sink action (end of attack chain)
            injection_sources: Known injection source action IDs (optional)
            max_depth: Maximum traversal depth

        Returns:
            Set of action IDs in the bidirectional slice
        """
        # Backward slice from target
        backward_slice = self.compute_backward_slice(target_action_id, max_depth)

        # If no injection sources provided, find them in backward slice
        if injection_sources is None:
            injection_sources = self.find_injection_sources_in_slice(backward_slice)

        # If still no sources, try to identify from graph structure
        if not injection_sources:
            # Find nodes with injected data consumption
            for action_id, action in self.action_map.items():
                if action.provenance and action.provenance.injection_detected:
                    injection_sources.append(action_id)
                # Also check for consumed injected data in node attributes
                if hasattr(action, 'data_consumed'):
                    consumed = action.data_consumed or []
                    if any('inject' in str(d).lower() for d in consumed):
                        injection_sources.append(action_id)

        # Forward slice from all injection sources
        forward_slice = set()
        for source_id in injection_sources:
            forward_slice |= self.compute_forward_slice(source_id, max_depth)

        # Union of both slices
        bidirectional_slice = backward_slice | forward_slice

        return bidirectional_slice

    def find_injection_sources_in_slice(
        self,
        slice_actions: Set[int],
    ) -> List[int]:
        """
        Find injection sources within a backward slice.

        Args:
            slice_actions: Action IDs in the slice

        Returns:
            List of action IDs that are injection sources
        """
        injection_sources = []

        for action_id in slice_actions:
            if self.is_injection_source(action_id):
                injection_sources.append(action_id)

        # Sort by action ID (chronological order)
        injection_sources.sort()
        return injection_sources

    def extract_attack_chain(
        self,
        target_action_id: int,
        injection_sources: List[int],
    ) -> List[int]:
        """
        Extract minimal attack chain from injection to target.

        Finds the shortest path from any injection source to the target action.

        Args:
            target_action_id: Target action
            injection_sources: List of injection source action IDs

        Returns:
            List of action IDs forming the attack chain (chronological order)
        """
        if not injection_sources:
            return []

        shortest_path = None
        shortest_length = float('inf')

        for source_id in injection_sources:
            try:
                # Find shortest path in the NetworkX graph
                path = nx.shortest_path(
                    self.graph.graph,
                    source=source_id,
                    target=target_action_id
                )

                if len(path) < shortest_length:
                    shortest_path = path
                    shortest_length = len(path)

            except nx.NetworkXNoPath:
                # No path from this source to target
                continue

        return shortest_path if shortest_path else []

    def identify_bottlenecks(
        self,
        slice_actions: Set[int],
        min_downstream: int = 2,
    ) -> List[int]:
        """
        Identify bottleneck actions in the slice.

        A bottleneck is an action with high fan-out: many downstream actions
        depend on it. Blocking a bottleneck would prevent many downstream actions.

        Args:
            slice_actions: Action IDs in the slice
            min_downstream: Minimum downstream dependencies to be a bottleneck

        Returns:
            List of bottleneck action IDs, sorted by downstream count (descending)
        """
        bottlenecks = []

        for action_id in slice_actions:
            # Count descendants in the graph
            try:
                descendants = nx.descendants(self.graph.graph, action_id)
                # Only count descendants that are in the slice
                descendants_in_slice = descendants & slice_actions

                if len(descendants_in_slice) >= min_downstream:
                    bottlenecks.append((action_id, len(descendants_in_slice)))

            except nx.NetworkXError:
                continue

        # Sort by downstream count (descending)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)

        return [action_id for action_id, _ in bottlenecks]

    def generate_explanation(
        self,
        target_action_id: int,
        attack_chain: List[int],
        injection_sources: List[int],
        bottlenecks: List[int],
    ) -> str:
        """
        Generate human-readable explanation of the attack chain.

        Args:
            target_action_id: Target action
            attack_chain: Attack chain action IDs
            injection_sources: Injection source action IDs
            bottlenecks: Bottleneck action IDs

        Returns:
            Explanation string
        """
        lines = []

        lines.append(f"Attack Chain Analysis for Action {target_action_id}")
        lines.append("=" * 60)

        # Target action
        target_action = self.action_map.get(target_action_id)
        if target_action:
            lines.append(f"\nTarget Action:")
            lines.append(f"  [{target_action_id}] {target_action.action_type.value}: {target_action.target}")
            if target_action.domain:
                lines.append(f"      Domain: {target_action.domain}")

        # Injection sources
        if injection_sources:
            lines.append(f"\nInjection Sources ({len(injection_sources)}):")
            for source_id in injection_sources:
                source_action = self.action_map.get(source_id)
                if source_action:
                    lines.append(f"  [{source_id}] {source_action.action_type.value}: {source_action.target}")
                    if source_action.domain:
                        lines.append(f"      Domain: {source_action.domain} (untrusted)")

        # Attack chain
        if attack_chain:
            lines.append(f"\nMinimal Attack Chain ({len(attack_chain)} actions):")
            for i, action_id in enumerate(attack_chain):
                action = self.action_map.get(action_id)
                if action:
                    marker = "->" if i < len(attack_chain) - 1 else "X"
                    lines.append(f"  [{action_id}] {action.action_type.value}: {action.target} {marker}")

        # Bottlenecks
        if bottlenecks:
            lines.append(f"\nCritical Bottlenecks ({len(bottlenecks)}):")
            for bottleneck_id in bottlenecks[:3]:  # Top 3
                action = self.action_map.get(bottleneck_id)
                if action:
                    lines.append(f"  [{bottleneck_id}] {action.action_type.value}: {action.target}")
                    lines.append(f"      Blocking this would prevent downstream attack actions")

        return "\n".join(lines)

    def backward_slice(
        self,
        target_action_id: int,
        max_depth: int = 50,
        min_bottleneck_downstream: int = 2,
        use_bidirectional: bool = True,
    ) -> SliceResult:
        """
        Perform complete backward slicing analysis.

        This is the main entry point for backward slicing.

        Args:
            target_action_id: Action to slice from
            max_depth: Maximum backward traversal depth
            min_bottleneck_downstream: Minimum downstream for bottleneck
            use_bidirectional: If True, use bidirectional slicing (recommended)

        Returns:
            SliceResult with complete analysis
        """
        # First compute backward slice to find injection sources
        initial_backward = self.compute_backward_slice(target_action_id, max_depth)
        injection_sources = self.find_injection_sources_in_slice(initial_backward)

        # Compute slice (bidirectional or backward-only)
        if use_bidirectional:
            slice_actions = self.compute_bidirectional_slice(
                target_action_id,
                injection_sources,
                max_depth
            )
        else:
            slice_actions = initial_backward

        # Re-find injection sources in full slice
        injection_sources = self.find_injection_sources_in_slice(slice_actions)

        # Extract attack chain
        attack_chain = self.extract_attack_chain(target_action_id, injection_sources)

        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks(slice_actions, min_bottleneck_downstream)

        # Generate explanation
        explanation = self.generate_explanation(
            target_action_id,
            attack_chain,
            injection_sources,
            bottlenecks,
        )

        return SliceResult(
            target_action_id=target_action_id,
            slice_actions=sorted(list(slice_actions)),
            attack_chain=attack_chain,
            injection_sources=injection_sources,
            bottlenecks=bottlenecks,
            explanation=explanation,
        )


def backward_slice(
    graph: CausalGraph,
    trajectory: Trajectory,
    target_action_id: int,
) -> SliceResult:
    """
    Convenience function for backward slicing.

    Args:
        graph: Causal graph
        trajectory: Full trajectory
        target_action_id: Action to slice from

    Returns:
        SliceResult
    """
    slicer = BackwardSlicer(graph, trajectory)
    return slicer.backward_slice(target_action_id)
