"""
Bottleneck feature extraction from causal graphs.

Features:
- Bottleneck scores (number of downstream dependents)
- Critical bottleneck identification
- Graph centrality measures
"""

from typing import List, Dict, Tuple, Set
import networkx as nx
from causaltrace.graph import CausalGraph


def compute_bottleneck_scores(graph: CausalGraph) -> Dict[int, int]:
    """
    For each node, compute bottleneck score = number of downstream dependent nodes.

    A high bottleneck score indicates that many subsequent actions depend on
    this action. Removing this action would break many causal chains.

    In attacks, bottlenecks often represent:
    - Initial compromise
    - Credential acquisition
    - Privilege escalation

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary mapping action_id -> bottleneck_score
    """
    if graph.num_nodes() == 0:
        return {}

    # Convert to NetworkX for efficient reachability queries
    G = _to_networkx(graph)

    scores = {}

    for node_id in G.nodes():
        # Find all nodes reachable from this node
        try:
            reachable = nx.descendants(G, node_id)
            scores[node_id] = len(reachable)
        except (nx.NetworkXError, nx.NetworkXException):
            scores[node_id] = 0

    return scores


def identify_critical_bottlenecks(graph: CausalGraph, threshold: int = 5) -> List[int]:
    """
    Identify actions whose removal would disconnect significant portions of graph.

    These are critical points in the attack chain. Blocking these actions
    would prevent the attack.

    Args:
        graph: CausalGraph instance
        threshold: Minimum bottleneck score to be considered critical

    Returns:
        List of action IDs with bottleneck_score >= threshold
    """
    scores = compute_bottleneck_scores(graph)
    return [action_id for action_id, score in scores.items() if score >= threshold]


def compute_max_bottleneck_score(graph: CausalGraph) -> int:
    """
    Get the maximum bottleneck score in the graph.

    This represents the most critical single action.

    Args:
        graph: CausalGraph instance

    Returns:
        Maximum bottleneck score, 0 if graph is empty
    """
    scores = compute_bottleneck_scores(graph)
    return max(scores.values()) if scores else 0


def compute_avg_bottleneck_score(graph: CausalGraph) -> float:
    """
    Get the average bottleneck score across all nodes.

    Higher average indicates more interdependent actions.

    Args:
        graph: CausalGraph instance

    Returns:
        Average bottleneck score, 0.0 if graph is empty
    """
    scores = compute_bottleneck_scores(graph)
    return sum(scores.values()) / len(scores) if scores else 0.0


def identify_articulation_points(graph: CausalGraph) -> List[int]:
    """
    Find articulation points (cut vertices) in the graph.

    These are nodes whose removal increases the number of connected components.
    In directed graphs, we use the undirected version for this analysis.

    Args:
        graph: CausalGraph instance

    Returns:
        List of action IDs that are articulation points
    """
    if graph.num_nodes() <= 1:
        return []

    G = _to_networkx(graph)

    # Convert to undirected for articulation point analysis
    G_undirected = G.to_undirected()

    try:
        articulation_points = list(nx.articulation_points(G_undirected))
        return articulation_points
    except (nx.NetworkXError, nx.NetworkXException):
        return []


def compute_betweenness_centrality(graph: CausalGraph) -> Dict[int, float]:
    """
    Compute betweenness centrality for each node.

    Betweenness centrality measures how often a node appears on shortest paths
    between other nodes. High betweenness indicates the node is a key connector.

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary mapping action_id -> betweenness_centrality (normalized to [0,1])
    """
    if graph.num_nodes() <= 1:
        return {node_id: 0.0 for node_id in graph.graph.nodes()}

    G = _to_networkx(graph)

    try:
        centrality = nx.betweenness_centrality(G, normalized=True)
        return centrality
    except (nx.NetworkXError, nx.NetworkXException):
        return {node_id: 0.0 for node_id in G.nodes()}


def compute_pagerank(graph: CausalGraph) -> Dict[int, float]:
    """
    Compute PageRank for each node.

    PageRank measures importance based on incoming edges and the importance
    of source nodes. High PageRank indicates the node is important in the
    causal structure.

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary mapping action_id -> pagerank_score
    """
    if graph.num_nodes() == 0:
        return {}

    G = _to_networkx(graph)

    try:
        pagerank = nx.pagerank(G, max_iter=100)
        return pagerank
    except (nx.NetworkXError, nx.NetworkXException):
        return {node_id: 1.0 / graph.num_nodes() for node_id in graph.graph.nodes()}


def identify_bridge_edges(graph: CausalGraph) -> List[Tuple[int, int]]:
    """
    Find bridge edges whose removal would disconnect the graph.

    These edges represent critical dependencies.

    Args:
        graph: CausalGraph instance

    Returns:
        List of (source_id, target_id) tuples that are bridges
    """
    if graph.num_edges() == 0:
        return []

    G = _to_networkx(graph)

    # Convert to undirected for bridge analysis
    G_undirected = G.to_undirected()

    try:
        bridges = list(nx.bridges(G_undirected))
        return bridges
    except (nx.NetworkXError, nx.NetworkXException):
        return []


def compute_node_removal_impact(graph: CausalGraph, action_id: int) -> int:
    """
    Compute the impact of removing a specific node.

    Impact is measured as the number of nodes that would become unreachable
    from entry nodes if this node is removed.

    Args:
        graph: CausalGraph instance
        action_id: Node to analyze

    Returns:
        Number of nodes that would become disconnected
    """
    if action_id not in graph.graph.nodes():
        return 0

    G = _to_networkx(graph)

    # Find entry nodes
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    if not entry_nodes:
        # No clear entry nodes, use all nodes as potential sources
        entry_nodes = list(G.nodes())

    # Count reachable nodes before removal
    reachable_before = set()
    for entry in entry_nodes:
        try:
            reachable = nx.descendants(G, entry)
            reachable_before.update(reachable)
            reachable_before.add(entry)
        except (nx.NetworkXError, nx.NetworkXException):
            continue

    # Remove the node and count reachable nodes after
    G_after = G.copy()
    G_after.remove_node(action_id)

    reachable_after = set()
    for entry in entry_nodes:
        if entry == action_id:
            continue
        if entry not in G_after:
            continue
        try:
            reachable = nx.descendants(G_after, entry)
            reachable_after.update(reachable)
            reachable_after.add(entry)
        except (nx.NetworkXError, nx.NetworkXException):
            continue

    # Impact is the difference
    return len(reachable_before) - len(reachable_after)


def rank_nodes_by_criticality(graph: CausalGraph, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Rank nodes by overall criticality using multiple metrics.

    Combines bottleneck score, betweenness centrality, and PageRank.

    Args:
        graph: CausalGraph instance
        top_k: Number of top critical nodes to return

    Returns:
        List of (action_id, criticality_score) tuples, sorted by score (descending)
    """
    if graph.num_nodes() == 0:
        return []

    # Get metrics
    bottleneck_scores = compute_bottleneck_scores(graph)
    betweenness = compute_betweenness_centrality(graph)
    pagerank = compute_pagerank(graph)

    # Normalize bottleneck scores to [0, 1]
    max_bottleneck = max(bottleneck_scores.values()) if bottleneck_scores else 1
    normalized_bottleneck = {
        k: v / max_bottleneck for k, v in bottleneck_scores.items()
    }

    # Compute combined criticality score (weighted average)
    criticality = {}
    for node_id in graph.graph.nodes():
        score = (
            0.5 * normalized_bottleneck.get(node_id, 0) +
            0.3 * betweenness.get(node_id, 0) +
            0.2 * pagerank.get(node_id, 0)
        )
        criticality[node_id] = score

    # Sort by score and return top-k
    ranked = sorted(criticality.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def identify_attack_initiation_nodes(graph: CausalGraph) -> List[int]:
    """
    Identify nodes that likely represent attack initiation.

    These are typically:
    - Entry nodes (no predecessors)
    - High outgoing degree
    - High bottleneck score

    Args:
        graph: CausalGraph instance

    Returns:
        List of action IDs likely to be attack initiation points
    """
    if graph.num_nodes() == 0:
        return []

    G = _to_networkx(graph)
    bottleneck_scores = compute_bottleneck_scores(graph)

    candidates = []

    for node_id in G.nodes():
        # Check if it's an entry node or has few predecessors
        in_degree = G.in_degree(node_id)

        # Check outgoing degree
        out_degree = G.out_degree(node_id)

        # Check bottleneck score
        bottleneck = bottleneck_scores.get(node_id, 0)

        # Heuristic: entry node OR (low in-degree AND high out-degree AND high bottleneck)
        if in_degree == 0 or (in_degree <= 2 and out_degree >= 2 and bottleneck >= 5):
            candidates.append(node_id)

    return candidates


def _to_networkx(graph: CausalGraph) -> nx.DiGraph:
    """
    Convert CausalGraph to NetworkX DiGraph.

    Args:
        graph: CausalGraph instance

    Returns:
        NetworkX directed graph
    """
    # CausalGraph already contains a NetworkX graph
    # Just return a copy to avoid modifications
    return graph.graph.copy()


def get_bottleneck_statistics(graph: CausalGraph) -> Dict[str, any]:
    """
    Get comprehensive bottleneck statistics.

    Returns:
        Dictionary with bottleneck-related metrics
    """
    scores = compute_bottleneck_scores(graph)
    betweenness = compute_betweenness_centrality(graph)

    return {
        "max_bottleneck_score": compute_max_bottleneck_score(graph),
        "avg_bottleneck_score": compute_avg_bottleneck_score(graph),
        "num_critical_bottlenecks": len(identify_critical_bottlenecks(graph, threshold=5)),
        "num_articulation_points": len(identify_articulation_points(graph)),
        "num_bridge_edges": len(identify_bridge_edges(graph)),
        "top_5_critical_nodes": rank_nodes_by_criticality(graph, top_k=5),
        "avg_betweenness": sum(betweenness.values()) / len(betweenness) if betweenness else 0.0,
        "attack_initiation_candidates": identify_attack_initiation_nodes(graph),
    }
