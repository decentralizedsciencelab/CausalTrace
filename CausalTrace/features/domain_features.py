"""
Domain feature extraction from causal graphs.

Features:
- Cross-domain edge counts
- Cross-domain ratios
- Domain transition patterns
"""

import json
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from causaltrace.graph import CausalGraph, EdgeType
from causaltrace.utils.io import extract_domain


def get_action_domain(action: Any) -> Optional[str]:
    """
    Extract domain from action, checking multiple sources.

    This function handles the pajaMAS trajectory format where the domain
    is not directly set but can be extracted from:
    1. action.domain (if already set)
    2. context.tool_input.kwargs.url (pajaMAS web_fetch format)
    3. context.tool_input.args[0] (alternative URL format)
    4. result containing source_url (web_surf output)
    5. target field if it looks like a URL

    Args:
        action: Action object (from trajectory.actions)

    Returns:
        Domain string (e.g., "localhost") or None
    """
    # 1. Check if domain already set
    domain = getattr(action, 'domain', None)
    if domain:
        return domain

    # 2. Check context.tool_input for URL (pajaMAS format)
    context = getattr(action, 'context', None) or {}
    if isinstance(context, dict):
        tool_input = context.get('tool_input', {})
        if isinstance(tool_input, dict):
            # Check kwargs.url first (pajaMAS web_fetch format)
            kwargs = tool_input.get('kwargs', {})
            if isinstance(kwargs, dict):
                url = kwargs.get('url', '')
                if url and isinstance(url, str) and url.startswith('http'):
                    return extract_domain(url)

            # Check args[0] as fallback
            args = tool_input.get('args', [])
            if args and isinstance(args, list) and len(args) > 0:
                first_arg = args[0]
                if isinstance(first_arg, str) and first_arg.startswith('http'):
                    return extract_domain(first_arg)

    # 3. Check result for source_url (pajaMAS web_surf output format)
    result = getattr(action, 'result', '')
    if result and isinstance(result, str) and 'source_url' in result:
        try:
            # Try to parse as JSON
            result_dict = json.loads(result)
            if isinstance(result_dict, dict) and 'source_url' in result_dict:
                return extract_domain(result_dict['source_url'])
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Check target if it looks like a URL
    target = getattr(action, 'target', '')
    if target and isinstance(target, str) and target.startswith('http'):
        return extract_domain(target)

    return None


def count_cross_domain_edges(graph: CausalGraph) -> int:
    """
    Count edges where source and target nodes have different domains.

    Cross-domain edges may indicate:
    - Data exfiltration (sending data to external domain)
    - Trust transfer across unrelated sites
    - Complex multi-site attacks

    Args:
        graph: CausalGraph instance

    Returns:
        Number of cross-domain edges
    """
    # Use the built-in method from CausalGraph
    return len(graph.get_cross_domain_edges())


def compute_cross_domain_ratio(graph: CausalGraph) -> float:
    """
    Compute ratio of cross-domain edges to total edges.

    A high ratio suggests the agent is frequently transferring
    data or trust across different domains.

    Args:
        graph: CausalGraph instance

    Returns:
        Ratio in [0, 1], or 0.0 if no edges
    """
    total_edges = graph.num_edges()
    if total_edges == 0:
        return 0.0

    cross_domain = count_cross_domain_edges(graph)
    return cross_domain / total_edges


def count_unique_domains(graph: CausalGraph) -> int:
    """
    Count the number of unique domains visited in the trajectory.

    A higher number may indicate reconnaissance or multi-site attacks.

    Args:
        graph: CausalGraph instance

    Returns:
        Number of unique domains
    """
    domains = set()

    for action in graph.trajectory.actions:
        domain = get_action_domain(action)
        if domain:
            domains.add(domain)

    return len(domains)


def identify_domain_transitions(graph: CausalGraph) -> List[Tuple[str, str]]:
    """
    List all domain transitions (source_domain -> target_domain).

    Useful for analyzing attack patterns and understanding which
    domains are connected through causal dependencies.

    Args:
        graph: CausalGraph instance

    Returns:
        List of (source_domain, target_domain) tuples
    """
    transitions = []

    for edge in graph.get_all_edges():
        source_node = graph.get_node(edge.source_action_id)
        target_node = graph.get_node(edge.target_action_id)

        if source_node and target_node:
            source_domain = get_action_domain(source_node) or "unknown"
            target_domain = get_action_domain(target_node) or "unknown"

            if source_domain != target_domain:
                transitions.append((source_domain, target_domain))

    return transitions


def get_domain_transition_matrix(graph: CausalGraph) -> Dict[Tuple[str, str], int]:
    """
    Get a matrix of domain transitions with counts.

    This shows how frequently data flows from one domain to another.

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary mapping (source_domain, target_domain) to count
    """
    transitions = identify_domain_transitions(graph)

    matrix = defaultdict(int)
    for source, target in transitions:
        matrix[(source, target)] += 1

    return dict(matrix)


def get_cross_domain_edge_types(graph: CausalGraph) -> Dict[str, int]:
    """
    Count cross-domain edges by edge type.

    This helps identify what kind of cross-domain interactions occur:
    - DATA_DEPENDENCY: Data extracted from one domain used in another
    - TRUST_TRANSFER: Code/patterns from one domain executed in another
    - STATE_ENABLEMENT: Authentication from one domain enables access to another

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary mapping edge type to count of cross-domain occurrences
    """
    counts = {
        "data_dependency": 0,
        "trust_transfer": 0,
        "state_enablement": 0,
    }

    for edge in graph.get_all_edges():
        source_node = graph.get_node(edge.source_action_id)
        target_node = graph.get_node(edge.target_action_id)

        if source_node and target_node:
            source_domain = get_action_domain(source_node) or ""
            target_domain = get_action_domain(target_node) or ""

            # Only count if cross-domain
            if source_domain and target_domain and source_domain != target_domain:
                if edge.edge_type == EdgeType.DATA_DEPENDENCY:
                    counts["data_dependency"] += 1
                elif edge.edge_type == EdgeType.TRUST_TRANSFER:
                    counts["trust_transfer"] += 1
                elif edge.edge_type == EdgeType.STATE_ENABLEMENT:
                    counts["state_enablement"] += 1

    return counts


def identify_external_domains(graph: CausalGraph, internal_domains: Set[str]) -> Set[str]:
    """
    Identify domains that are external (not in the internal set).

    Useful for detecting data exfiltration to attacker-controlled domains.

    Args:
        graph: CausalGraph instance
        internal_domains: Set of domains considered "internal" or legitimate

    Returns:
        Set of external domain names
    """
    all_domains = set()

    for action in graph.trajectory.actions:
        domain = get_action_domain(action)
        if domain:
            all_domains.add(domain)

    return all_domains - internal_domains


def compute_domain_entropy(graph: CausalGraph) -> float:
    """
    Compute entropy of domain distribution.

    Low entropy: Actions concentrated in few domains (focused behavior)
    High entropy: Actions spread across many domains (broad reconnaissance)

    Args:
        graph: CausalGraph instance

    Returns:
        Shannon entropy of domain distribution
    """
    import math

    if graph.num_nodes() == 0:
        return 0.0

    # Count actions per domain
    domain_counts = defaultdict(int)
    total_actions = 0

    for action in graph.trajectory.actions:
        domain = get_action_domain(action) or "unknown"
        domain_counts[domain] += 1
        total_actions += 1

    if total_actions == 0:
        return 0.0

    # Compute Shannon entropy: -sum(p * log2(p))
    entropy = 0.0
    for count in domain_counts.values():
        if count > 0:
            probability = count / total_actions
            entropy -= probability * math.log2(probability)

    return entropy


def identify_suspicious_domain_patterns(graph: CausalGraph) -> Dict[str, any]:
    """
    Identify potentially suspicious domain interaction patterns.

    Returns:
        Dictionary with various suspicious pattern indicators
    """
    transitions = identify_domain_transitions(graph)
    domains = count_unique_domains(graph)
    cross_domain_edges = count_cross_domain_edges(graph)
    cross_domain_types = get_cross_domain_edge_types(graph)

    return {
        "high_domain_diversity": domains > 10,  # Many different domains
        "many_cross_domain_edges": cross_domain_edges > 5,
        "high_cross_domain_ratio": compute_cross_domain_ratio(graph) > 0.3,
        "trust_transfer_across_domains": cross_domain_types["trust_transfer"] > 0,
        "data_exfiltration_risk": cross_domain_types["data_dependency"] > 2,
        "domain_count": domains,
        "cross_domain_count": cross_domain_edges,
        "unique_transitions": len(set(transitions)),
    }


def get_domain_statistics(graph: CausalGraph) -> Dict[str, any]:
    """
    Get comprehensive domain statistics for a graph.

    Returns:
        Dictionary with domain-related metrics
    """
    return {
        "unique_domains": count_unique_domains(graph),
        "cross_domain_edges": count_cross_domain_edges(graph),
        "cross_domain_ratio": compute_cross_domain_ratio(graph),
        "domain_transitions": len(identify_domain_transitions(graph)),
        "domain_entropy": compute_domain_entropy(graph),
        "cross_domain_by_type": get_cross_domain_edge_types(graph),
    }
