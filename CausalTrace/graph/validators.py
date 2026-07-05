"""
Validators for causal graphs.

This module provides validation functions to ensure graph correctness,
including DAG property and edge consistency checks.
"""

import networkx as nx
from typing import List
from causaltrace.graph.causal_graph import CausalGraph


def validate_dag(graph: CausalGraph) -> bool:
    """
    Validate that graph is a directed acyclic graph.

    Args:
        graph: The causal graph to validate

    Returns:
        True if the graph is a valid DAG, False otherwise
    """
    try:
        return nx.is_directed_acyclic_graph(graph.graph)
    except Exception:
        return False


def validate_edge_consistency(graph: CausalGraph) -> List[str]:
    """
    Check for inconsistent edges in the graph.

    Checks for:
    1. Edges to earlier actions (violating temporal order)
    2. Self-loops
    3. Duplicate edges
    4. Edges referencing non-existent nodes

    Args:
        graph: The causal graph to validate

    Returns:
        List of issue descriptions (empty if no issues)
    """
    issues = []

    # Get all node IDs
    node_ids = set(graph.graph.nodes())

    # Check each edge
    seen_edges = set()
    for source, target, data in graph.graph.edges(data=True):
        # Check for non-existent nodes
        if source not in node_ids:
            issues.append(f"Edge source {source} not in graph nodes")
        if target not in node_ids:
            issues.append(f"Edge target {target} not in graph nodes")

        # Check for self-loops
        if source == target:
            issues.append(f"Self-loop detected at node {source}")

        # Check for temporal consistency (source should come before target)
        if source > target:
            issues.append(f"Edge {source} -> {target} violates temporal order (source > target)")

        # Check for duplicate edges (same source and target)
        edge_key = (source, target)
        if edge_key in seen_edges:
            issues.append(f"Duplicate edge detected: {source} -> {target}")
        seen_edges.add(edge_key)

    return issues


def validate_edge_types(graph: CausalGraph) -> List[str]:
    """
    Validate that all edges have valid edge types.

    Args:
        graph: The causal graph to validate

    Returns:
        List of issue descriptions (empty if no issues)
    """
    issues = []
    from causaltrace.graph.causal_graph import EdgeType

    valid_types = set(EdgeType)

    for source, target, data in graph.graph.edges(data=True):
        edge_type = data.get("edge_type")
        if edge_type is None:
            issues.append(f"Edge {source} -> {target} has no edge type")
        elif edge_type not in valid_types:
            issues.append(f"Edge {source} -> {target} has invalid edge type: {edge_type}")

    return issues


def validate_graph_metrics(graph: CausalGraph) -> dict:
    """
    Compute validation metrics for the graph.

    Returns:
        Dictionary of validation metrics
    """
    return {
        "is_dag": validate_dag(graph),
        "num_consistency_issues": len(validate_edge_consistency(graph)),
        "num_edge_type_issues": len(validate_edge_types(graph)),
        "num_isolated_nodes": len(list(nx.isolates(graph.graph))),
        "num_connected_components": nx.number_weakly_connected_components(graph.graph),
        "has_cycles": not validate_dag(graph),
    }
