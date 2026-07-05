"""
State accumulation feature extraction from causal graphs.

Features:
- State accumulation score (unused credentials, data)
- Unused state detection
- State consumption patterns
"""

from typing import List, Dict, Set, Any
from collections import defaultdict
from causaltrace.graph import CausalGraph, EdgeType
from causaltrace.models import Trajectory, ActionType


def compute_state_accumulation_score(graph: CausalGraph, trajectory: Trajectory = None) -> float:
    """
    Measure how much state (credentials, data) is accumulated but not used.

    This is a key indicator of attacks:
    - Login to services that aren't needed for the task
    - Extract data that isn't used for the stated goal
    - Accumulate credentials without corresponding resource access

    Args:
        graph: CausalGraph instance
        trajectory: Optional Trajectory for additional context

    Returns:
        Score in [0, 1] where higher values indicate more unused accumulation
        0.0 if no state accumulation detected
    """
    if graph.num_nodes() == 0:
        return 0.0

    # Track state accumulation and consumption
    accumulated_state = set()
    consumed_state = set()

    # Identify state-producing actions
    for action in graph.trajectory.actions:
        # State production: logins, data extraction, credentials
        if _is_state_producing_action(action):
            # Add all state keys produced by this action
            state_keys = _get_all_state_keys(action)
            accumulated_state.update(state_keys)

        # State consumption: using data/credentials in other actions
        if action.data_consumed:
            consumed_state.update(action.data_consumed)

    # Also check for state enablement edges (state that was actually used)
    for edge in graph.get_all_edges():
        if edge.edge_type == EdgeType.STATE_ENABLEMENT:
            source_node = graph.get_node(edge.source_action_id)
            if source_node:
                # Mark all state keys from this source as consumed
                state_keys = _get_all_state_keys(source_node)
                consumed_state.update(state_keys)

    # Also check data dependency edges
    for edge in graph.get_all_edges():
        if edge.edge_type == EdgeType.DATA_DEPENDENCY:
            source_node = graph.get_node(edge.source_action_id)
            if source_node and source_node.data_produced:
                # Mark the produced data as consumed (it was used by target)
                consumed_state.update(source_node.data_produced)

    # Compute unused state
    unused_state = accumulated_state - consumed_state

    if len(accumulated_state) == 0:
        return 0.0

    return len(unused_state) / len(accumulated_state)


def identify_unused_credentials(graph: CausalGraph, trajectory: Trajectory = None) -> List[str]:
    """
    Find credentials that were obtained but never used.

    Example: Login to service A, but never access service A resources.
    This is a strong attack indicator.

    Args:
        graph: CausalGraph instance
        trajectory: Optional Trajectory for additional context

    Returns:
        List of service/domain names with unused credentials
    """
    unused = []

    # Track which services were authenticated to
    authenticated_services = set()
    for action in graph.trajectory.actions:
        if action.action_type == ActionType.LOGIN:
            domain = action.domain or _extract_domain_from_target(action.target)
            if domain:
                authenticated_services.add(domain)

    # Track which services had resources accessed (post-authentication)
    accessed_services = set()
    for edge in graph.get_all_edges():
        if edge.edge_type == EdgeType.STATE_ENABLEMENT:
            # This edge indicates auth from source was used for target
            source_node = graph.get_node(edge.source_action_id)
            target_node = graph.get_node(edge.target_action_id)

            if source_node and target_node:
                if source_node.action_type == ActionType.LOGIN:
                    # Check if target action uses the authenticated domain
                    accessed_services.add(source_node.domain or "unknown")

    # Services authenticated but never accessed
    unused = list(authenticated_services - accessed_services)

    return unused


def compute_unused_state_ratio(graph: CausalGraph, trajectory: Trajectory = None) -> float:
    """
    Ratio of unused state to total accumulated state.

    Wrapper around compute_state_accumulation_score for consistency.

    Args:
        graph: CausalGraph instance
        trajectory: Optional Trajectory for additional context

    Returns:
        Ratio in [0, 1]
    """
    return compute_state_accumulation_score(graph, trajectory)


def count_state_producing_actions(graph: CausalGraph) -> int:
    """
    Count actions that produce state (credentials, data extraction).

    Args:
        graph: CausalGraph instance

    Returns:
        Number of state-producing actions
    """
    count = 0
    for action in graph.trajectory.actions:
        if _is_state_producing_action(action):
            count += 1
    return count


def count_state_consuming_actions(graph: CausalGraph) -> int:
    """
    Count actions that consume state (use credentials, data).

    Args:
        graph: CausalGraph instance

    Returns:
        Number of state-consuming actions
    """
    count = 0
    for action in graph.trajectory.actions:
        if action.data_consumed or _is_state_consuming_action(action):
            count += 1
    return count


def get_state_flow_graph(graph: CausalGraph) -> Dict[str, List[str]]:
    """
    Build a state flow graph showing how state moves through actions.

    Returns:
        Dictionary mapping state_key -> list of action_ids that consume it
    """
    flow = defaultdict(list)

    for edge in graph.get_all_edges():
        if edge.edge_type in [EdgeType.DATA_DEPENDENCY, EdgeType.STATE_ENABLEMENT]:
            source_node = graph.get_node(edge.source_action_id)
            if source_node:
                state_key = _get_state_key(source_node)
                if state_key:
                    flow[state_key].append(edge.target_action_id)

    return dict(flow)


def identify_state_accumulation_hotspots(graph: CausalGraph) -> List[int]:
    """
    Identify action nodes that accumulate significant state.

    These are potential attack pivot points.

    Args:
        graph: CausalGraph instance

    Returns:
        List of action_ids that are state accumulation hotspots
    """
    hotspots = []

    # Iterate over trajectory actions (not graph.nodes which doesn't exist)
    for action in graph.trajectory.actions:
        action_id = action.action_id
        # Check if this action produces state
        if not _is_state_producing_action(action):
            continue

        # Check how many downstream actions depend on this state
        dependent_count = 0
        for edge in graph.get_all_edges():
            if edge.source_action_id == action_id:
                if edge.edge_type in [EdgeType.DATA_DEPENDENCY, EdgeType.STATE_ENABLEMENT]:
                    dependent_count += 1

        # If produces state but has few/no dependents, it's a hotspot
        if dependent_count <= 1:  # Including 0 and 1 dependent
            hotspots.append(action_id)

    return hotspots


def compute_state_lifetime(graph: CausalGraph) -> Dict[str, int]:
    """
    Compute how long state persists before being consumed.

    Measured in number of actions between production and consumption.

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary mapping state_key -> lifetime (in action steps)
    """
    import networkx as nx

    # Convert to NetworkX for path calculations
    G = nx.DiGraph()
    for action_id in graph.graph.nodes():
        G.add_node(action_id)
    for edge in graph.get_all_edges():
        G.add_edge(edge.source_action_id, edge.target_action_id)

    lifetimes = {}

    for edge in graph.get_all_edges():
        if edge.edge_type in [EdgeType.DATA_DEPENDENCY, EdgeType.STATE_ENABLEMENT]:
            source_node = graph.get_node(edge.source_action_id)
            if source_node:
                state_key = _get_state_key(source_node)
                if state_key:
                    # Calculate path length from source to target
                    try:
                        path_length = nx.shortest_path_length(G, edge.source_action_id, edge.target_action_id)
                        if state_key not in lifetimes or path_length > lifetimes[state_key]:
                            lifetimes[state_key] = path_length
                    except (nx.NetworkXError, nx.NetworkXNoPath):
                        continue

    return lifetimes


def _is_state_producing_action(action: Any) -> bool:
    """Check if an action produces state (credentials, data)."""
    # Login actions produce authentication state
    if action.action_type == ActionType.LOGIN:
        return True

    # Data extraction actions
    if action.action_type in [ActionType.READ, ActionType.EXTRACT, ActionType.DOWNLOAD]:
        return True

    # MAS-specific action types that produce state
    # DELEGATION: orchestrator delegates task, may produce state
    if action.action_type == ActionType.DELEGATION:
        return True

    # STATE_MUTATION: explicitly changes state (e.g., memory poisoning)
    if action.action_type == ActionType.STATE_MUTATION:
        return True

    # WEB_FETCH: retrieves data from URLs (produces data)
    if action.action_type == ActionType.WEB_FETCH:
        return True

    # CODE_EXECUTION: may produce output data
    if action.action_type == ActionType.CODE_EXECUTION:
        return True

    # AGENT_RESPONSE with data_produced: agent produces data
    if action.action_type == ActionType.AGENT_RESPONSE and action.data_produced:
        return True

    # Check if action explicitly produces data
    if action.data_produced:
        return True

    # Check context for authentication or data extraction
    if "password" in str(action.context).lower():
        return True
    if "credential" in str(action.context).lower():
        return True
    if "token" in str(action.context).lower():
        return True

    return False


def _is_state_consuming_action(action: Any) -> bool:
    """Check if an action consumes state."""
    # Actions that typically consume auth state
    if action.action_type in [ActionType.SEND_EMAIL, ActionType.UPLOAD, ActionType.SUBMIT]:
        return True

    # Check if action explicitly consumes data
    if action.data_consumed:
        return True

    return False


def _get_state_key(action: Any) -> str:
    """
    Generate a state key for an action.

    Returns keys in a format that matches data_consumed entries:
    - For data_produced: returns the first key without prefix (matches data_consumed format)
    - For login: returns auth:<domain>
    - Fallback: returns state:<action_id>
    """
    # If action has data_produced, return the raw keys (matches data_consumed format)
    if action.data_produced:
        # Return first produced key (without prefix to match data_consumed)
        return action.data_produced[0] if action.data_produced else None

    if action.action_type == ActionType.LOGIN:
        return f"auth:{action.domain or action.target}"

    return f"state:{action.action_id}"


def _get_all_state_keys(action: Any) -> List[str]:
    """
    Get all state keys produced by an action.

    This handles actions that produce multiple data items.
    """
    keys = []

    # Add all produced data keys (without prefix to match data_consumed format)
    if action.data_produced:
        keys.extend(action.data_produced)

    # Add auth key for login actions
    if action.action_type == ActionType.LOGIN:
        keys.append(f"auth:{action.domain or action.target}")

    # Add fallback key if no other keys
    if not keys:
        keys.append(f"state:{action.action_id}")

    return keys


def _extract_domain_from_target(target: str) -> str:
    """Extract domain from a target URL or string."""
    import re

    # Try to extract domain from URL
    match = re.search(r'https?://([^/]+)', target)
    if match:
        return match.group(1)

    # Try to extract domain-like strings
    match = re.search(r'([a-z0-9-]+\.[a-z]{2,})', target.lower())
    if match:
        return match.group(1)

    return ""


def get_state_statistics(graph: CausalGraph, trajectory: Trajectory = None) -> Dict[str, Any]:
    """
    Get comprehensive state-related statistics.

    Returns:
        Dictionary with state metrics
    """
    return {
        "state_accumulation_score": compute_state_accumulation_score(graph, trajectory),
        "unused_credentials": identify_unused_credentials(graph, trajectory),
        "num_state_producing": count_state_producing_actions(graph),
        "num_state_consuming": count_state_consuming_actions(graph),
        "state_accumulation_hotspots": identify_state_accumulation_hotspots(graph),
        "avg_state_lifetime": sum(compute_state_lifetime(graph).values()) / max(len(compute_state_lifetime(graph)), 1),
    }
