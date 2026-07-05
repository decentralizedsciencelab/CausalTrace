"""
Chain feature extraction from causal graphs.

Features:
- Chain depth (longest path)
- Average chain length
- Number of distinct chains
"""

from typing import List, Set, Dict, Tuple
import networkx as nx
from causaltrace.graph import CausalGraph


def compute_chain_depth(graph: CausalGraph) -> int:
    """
    Compute the longest path in the causal graph (chain depth).

    This represents the maximum number of causal dependencies in a single chain.
    Attacks typically have longer chains as they require multiple steps.

    Args:
        graph: CausalGraph instance

    Returns:
        Length of the longest path (0 if graph is empty or has no paths)
    """
    if graph.num_nodes() == 0:
        return 0

    # Convert to NetworkX directed graph for efficient algorithms
    G = _to_networkx(graph)

    if len(G.nodes()) == 0:
        return 0

    # Check if graph is a DAG (should be, but validate)
    if not nx.is_directed_acyclic_graph(G):
        # If cycles exist, use longest simple path instead
        # This is computationally expensive but handles edge cases
        max_length = 0
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target))
                        if paths:
                            max_length = max(max_length, max(len(p) - 1 for p in paths))
                    except nx.NetworkXNoPath:
                        continue
        return max_length

    # For DAG, use efficient longest path algorithm
    try:
        return nx.dag_longest_path_length(G)
    except (nx.NetworkXError, nx.NetworkXException):
        return 0


def compute_avg_chain_length(graph: CausalGraph) -> float:
    """
    Compute average length of all paths from entry nodes to exit nodes.

    Entry node: no incoming edges (starting points)
    Exit node: no outgoing edges (ending points)

    Args:
        graph: CausalGraph instance

    Returns:
        Average path length, 0.0 if no paths exist
    """
    if graph.num_nodes() == 0:
        return 0.0

    G = _to_networkx(graph)

    if len(G.nodes()) == 0:
        return 0.0

    # Find entry nodes (no predecessors)
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    # Find exit nodes (no successors)
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    if not entry_nodes or not exit_nodes:
        # If no clear entry/exit, compute average shortest path
        try:
            return nx.average_shortest_path_length(G)
        except (nx.NetworkXError, nx.NetworkXException):
            return 0.0

    # Compute all paths from entry to exit nodes
    total_length = 0
    path_count = 0

    for entry in entry_nodes:
        for exit_node in exit_nodes:
            if entry != exit_node:
                try:
                    # For DAG, use shortest path
                    if nx.has_path(G, entry, exit_node):
                        path_length = nx.shortest_path_length(G, entry, exit_node)
                        total_length += path_length
                        path_count += 1
                except (nx.NetworkXError, nx.NetworkXNoPath):
                    continue

    return total_length / path_count if path_count > 0 else 0.0


def count_chains(graph: CausalGraph) -> int:
    """
    Count the number of distinct causal chains (paths from entry to exit nodes).

    This counts all simple paths, which can be computationally expensive for large graphs.
    For efficiency, we limit the count to a maximum threshold.

    Args:
        graph: CausalGraph instance

    Returns:
        Number of distinct chains (capped at 1000 for performance)
    """
    MAX_CHAINS = 1000  # Performance limit

    if graph.num_nodes() == 0:
        return 0

    G = _to_networkx(graph)

    if len(G.nodes()) == 0:
        return 0

    # Find entry and exit nodes
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    if not entry_nodes or not exit_nodes:
        # No clear chains - return number of weakly connected components
        return nx.number_weakly_connected_components(G)

    # Count all simple paths from entry to exit
    chain_count = 0

    for entry in entry_nodes:
        for exit_node in exit_nodes:
            if entry != exit_node:
                try:
                    # Count paths (with limit for performance)
                    paths = nx.all_simple_paths(G, entry, exit_node)
                    for _ in paths:
                        chain_count += 1
                        if chain_count >= MAX_CHAINS:
                            return MAX_CHAINS
                except (nx.NetworkXError, nx.NetworkXNoPath):
                    continue

    return chain_count


def identify_critical_paths(graph: CausalGraph, top_k: int = 5) -> List[List[int]]:
    """
    Identify the top-k longest/most critical paths in the graph.

    Useful for understanding the main attack chains or workflows.

    Args:
        graph: CausalGraph instance
        top_k: Number of paths to return

    Returns:
        List of paths (each path is a list of action_ids)
    """
    if graph.num_nodes() == 0:
        return []

    G = _to_networkx(graph)

    if len(G.nodes()) == 0:
        return []

    # Find entry and exit nodes
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    if not entry_nodes or not exit_nodes:
        # Return longest path in graph
        if nx.is_directed_acyclic_graph(G):
            try:
                longest = nx.dag_longest_path(G)
                return [longest] if longest else []
            except (nx.NetworkXError, nx.NetworkXException):
                return []
        return []

    # Collect all paths with their lengths
    paths_with_lengths: List[Tuple[int, List[int]]] = []

    for entry in entry_nodes:
        for exit_node in exit_nodes:
            if entry != exit_node:
                try:
                    all_paths = nx.all_simple_paths(G, entry, exit_node)
                    for path in all_paths:
                        paths_with_lengths.append((len(path), path))
                        # Limit total paths checked for performance
                        if len(paths_with_lengths) > 1000:
                            break
                except (nx.NetworkXError, nx.NetworkXNoPath):
                    continue
            if len(paths_with_lengths) > 1000:
                break
        if len(paths_with_lengths) > 1000:
            break

    # Sort by length (descending) and return top-k
    paths_with_lengths.sort(reverse=True, key=lambda x: x[0])
    return [path for _, path in paths_with_lengths[:top_k]]


def _to_networkx(graph: CausalGraph) -> nx.DiGraph:
    """
    Convert CausalGraph to NetworkX DiGraph for efficient algorithms.

    Args:
        graph: CausalGraph instance

    Returns:
        NetworkX directed graph
    """
    # CausalGraph already contains a NetworkX graph
    # Just return a copy to avoid modifications
    return graph.graph.copy()


def get_chain_statistics(graph: CausalGraph) -> Dict[str, float]:
    """
    Get comprehensive chain statistics for a graph.

    Returns:
        Dictionary with chain-related metrics
    """
    return {
        "chain_depth": compute_chain_depth(graph),
        "avg_chain_length": compute_avg_chain_length(graph),
        "num_chains": count_chains(graph),
        "num_critical_paths": len(identify_critical_paths(graph, top_k=10)),
    }
