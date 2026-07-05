"""
Edge enhancement for sparse causal graphs.

This module addresses the issue where synthetic graphs have sparse edge connectivity,
causing backward slicing to miss nodes that should be part of the attack chain.

Strategies:
1. Sequential edges: Add edges between consecutive actions
2. Data flow edges: Add edges where data flows between actions
3. Domain-based edges: Connect actions on the same domain
4. Injection propagation: Connect injection source to all downstream actions
"""

from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict


def enhance_graph_edges(graph: Dict[str, Any], strategy: str = "all") -> Dict[str, Any]:
    """
    Enhance graph with additional causal edges.

    Args:
        graph: Graph dictionary with nodes and edges
        strategy: Enhancement strategy ("sequential", "dataflow", "domain", "injection", "all")

    Returns:
        Enhanced graph with additional edges
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        return graph

    # Build existing edge set for deduplication
    existing_edges = set()
    for edge in edges:
        src = edge.get("source", 0)
        tgt = edge.get("target", 0)
        if isinstance(src, str):
            src = int(src.split("_")[-1]) if "_" in src else int(src.replace("node_", ""))
        if isinstance(tgt, str):
            tgt = int(tgt.split("_")[-1]) if "_" in tgt else int(tgt.replace("node_", ""))
        existing_edges.add((src, tgt))

    new_edges = []

    if strategy in ("sequential", "all"):
        new_edges.extend(_add_sequential_edges(nodes, existing_edges))

    if strategy in ("dataflow", "all"):
        new_edges.extend(_add_dataflow_edges(nodes, existing_edges))

    if strategy in ("domain", "all"):
        new_edges.extend(_add_domain_edges(nodes, existing_edges))

    if strategy in ("injection", "all"):
        new_edges.extend(_add_injection_propagation_edges(nodes, existing_edges))

    # Add new edges to graph
    enhanced_graph = graph.copy()
    enhanced_graph["edges"] = edges + new_edges
    enhanced_graph["num_edges"] = len(enhanced_graph["edges"])

    return enhanced_graph


def _add_sequential_edges(nodes: List[Dict], existing: Set[Tuple[int, int]]) -> List[Dict]:
    """Add edges between consecutive actions (temporal ordering)."""
    new_edges = []

    # Sort nodes by timestamp or action_id
    sorted_nodes = sorted(nodes, key=lambda n: (n.get("timestamp", 0), n.get("action_id", 0)))

    for i in range(len(sorted_nodes) - 1):
        src_id = sorted_nodes[i].get("action_id", i)
        tgt_id = sorted_nodes[i + 1].get("action_id", i + 1)

        if (src_id, tgt_id) not in existing:
            new_edges.append({
                "source": src_id,
                "target": tgt_id,
                "edge_type": "temporal_sequence",
                "metadata": {"evidence": "Sequential temporal ordering", "added_by": "edge_enhancer"}
            })
            existing.add((src_id, tgt_id))

    return new_edges


def _add_dataflow_edges(nodes: List[Dict], existing: Set[Tuple[int, int]]) -> List[Dict]:
    """Add edges where data flows between actions."""
    new_edges = []

    # Track data production and consumption
    data_producers = defaultdict(list)  # data_key -> [(action_id, timestamp)]
    data_consumers = []  # [(action_id, data_keys_consumed)]

    for node in nodes:
        action_id = node.get("action_id", 0)
        timestamp = node.get("timestamp", 0)

        # Track produced data
        produced = node.get("data_produced", [])
        for data_key in produced:
            data_producers[data_key].append((action_id, timestamp))

        # Track consumed data
        consumed = node.get("data_consumed", [])
        if consumed:
            data_consumers.append((action_id, timestamp, consumed))

    # Connect producers to consumers
    for consumer_id, consumer_ts, consumed_keys in data_consumers:
        for data_key in consumed_keys:
            # Find the most recent producer before this consumer
            producers = data_producers.get(data_key, [])
            for producer_id, producer_ts in producers:
                if producer_ts < consumer_ts and (producer_id, consumer_id) not in existing:
                    new_edges.append({
                        "source": producer_id,
                        "target": consumer_id,
                        "edge_type": "data_dependency",
                        "metadata": {
                            "data_key": data_key,
                            "evidence": f"Data '{data_key}' flows from action {producer_id} to {consumer_id}",
                            "added_by": "edge_enhancer"
                        }
                    })
                    existing.add((producer_id, consumer_id))

    return new_edges


def _add_domain_edges(nodes: List[Dict], existing: Set[Tuple[int, int]]) -> List[Dict]:
    """Add edges connecting actions on the same domain (state enablement)."""
    new_edges = []

    # Group nodes by domain
    domain_nodes = defaultdict(list)
    for node in nodes:
        domain = node.get("domain", "")
        if domain:
            domain_nodes[domain].append(node)

    # For each domain, connect first action to all subsequent actions
    for domain, domain_node_list in domain_nodes.items():
        if len(domain_node_list) < 2:
            continue

        # Sort by timestamp
        sorted_domain = sorted(domain_node_list, key=lambda n: n.get("timestamp", 0))
        first_node = sorted_domain[0]
        first_id = first_node.get("action_id", 0)

        # Connect first to all others (state enablement)
        for node in sorted_domain[1:]:
            tgt_id = node.get("action_id", 0)
            if (first_id, tgt_id) not in existing:
                new_edges.append({
                    "source": first_id,
                    "target": tgt_id,
                    "edge_type": "state_enablement",
                    "metadata": {
                        "evidence": f"First action on {domain} enables subsequent actions",
                        "state": "domain_session",
                        "added_by": "edge_enhancer"
                    }
                })
                existing.add((first_id, tgt_id))

    return new_edges


def _add_injection_propagation_edges(nodes: List[Dict], existing: Set[Tuple[int, int]]) -> List[Dict]:
    """Add edges from injection source to all downstream actions."""
    new_edges = []

    # Find injection sources
    injection_sources = []
    for node in nodes:
        consumed = node.get("data_consumed", [])
        if any("inject" in str(d).lower() for d in consumed):
            injection_sources.append(node)

    if not injection_sources:
        return new_edges

    # Sort all nodes by timestamp
    sorted_nodes = sorted(nodes, key=lambda n: n.get("timestamp", 0))

    for source_node in injection_sources:
        source_id = source_node.get("action_id", 0)
        source_ts = source_node.get("timestamp", 0)

        # Connect to all subsequent nodes
        for node in sorted_nodes:
            tgt_id = node.get("action_id", 0)
            tgt_ts = node.get("timestamp", 0)

            if tgt_ts > source_ts and (source_id, tgt_id) not in existing:
                new_edges.append({
                    "source": source_id,
                    "target": tgt_id,
                    "edge_type": "injection_propagation",
                    "metadata": {
                        "evidence": f"Injection at action {source_id} may influence action {tgt_id}",
                        "added_by": "edge_enhancer"
                    }
                })
                existing.add((source_id, tgt_id))

    return new_edges


def analyze_graph_connectivity(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze graph connectivity to identify potential issues.

    Returns statistics about edge coverage and disconnected nodes.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    num_nodes = len(nodes)
    num_edges = len(edges)

    # Build adjacency lists
    has_incoming = set()
    has_outgoing = set()

    for edge in edges:
        src = edge.get("source", 0)
        tgt = edge.get("target", 0)
        if isinstance(src, str):
            src = int(src.split("_")[-1]) if "_" in src else int(src.replace("node_", ""))
        if isinstance(tgt, str):
            tgt = int(tgt.split("_")[-1]) if "_" in tgt else int(tgt.replace("node_", ""))
        has_outgoing.add(src)
        has_incoming.add(tgt)

    all_node_ids = set(n.get("action_id", i) for i, n in enumerate(nodes))

    # Find disconnected nodes
    no_incoming = all_node_ids - has_incoming
    no_outgoing = all_node_ids - has_outgoing
    isolated = no_incoming & no_outgoing

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edge_density": num_edges / max(num_nodes - 1, 1) if num_nodes > 1 else 0,
        "nodes_without_incoming": sorted(no_incoming),
        "nodes_without_outgoing": sorted(no_outgoing),
        "isolated_nodes": sorted(isolated),
        "connectivity_ratio": 1 - len(isolated) / max(num_nodes, 1),
    }
