"""
Watermark-based feature extraction from causal graphs.

Features:
- Watermark coverage (ratio of tagged nodes)
- Watermark propagation depth (max depth of watermark lineage)
- Watermark sensitive coverage (ratio of sensitive nodes with watermark)
- Watermark tampered detection (sensitive nodes without watermark lineage)
- Watermark gap analysis (count and depth of lineage breaks)

Watermark features are used to detect attacks where data lineage is broken
or tampered with, indicating potential injection or manipulation.

Graph metadata keys used:
- watermark_tagged_nodes: Set of node IDs that have watermark tags
- watermark_sensitive_nodes: Set of node IDs that are sensitive (e.g., sinks)
- watermark_tampered_nodes: Set of node IDs where watermark was expected but missing
"""

from typing import Set, Dict, Any, Optional
from causaltrace.graph import CausalGraph


def compute_watermark_coverage(graph: CausalGraph) -> float:
    """
    Compute the ratio of watermark-tagged nodes to total nodes.

    A higher coverage indicates better tracking of data provenance.
    Lower coverage may indicate injection points or untracked data flows.

    Args:
        graph: CausalGraph instance

    Returns:
        Ratio of watermark-tagged nodes (0.0 to 1.0), 0.0 if no metadata
    """
    total_nodes = graph.num_nodes()
    if total_nodes == 0:
        return 0.0

    watermark_tagged = _get_watermark_tagged_nodes(graph)
    if watermark_tagged is None:
        return 0.0

    return len(watermark_tagged) / total_nodes


def compute_watermark_propagation_depth(graph: CausalGraph) -> int:
    """
    Compute the maximum depth of watermark propagation in the graph.

    This measures how far watermark tags propagate through causal chains.
    A higher depth indicates better tracking through complex workflows.

    Args:
        graph: CausalGraph instance

    Returns:
        Maximum depth of watermark propagation, 0 if no watermark metadata
    """
    watermark_tagged = _get_watermark_tagged_nodes(graph)
    if watermark_tagged is None or len(watermark_tagged) == 0:
        return 0

    # Find the maximum depth among all watermarked nodes
    max_depth = 0
    for node_id in watermark_tagged:
        depth = graph.get_node_depth(node_id)
        max_depth = max(max_depth, depth)

    return max_depth


def compute_watermark_sensitive_coverage(graph: CausalGraph) -> float:
    """
    Compute the ratio of sensitive nodes that have watermark lineage.

    Sensitive nodes (e.g., code execution, data exfiltration points) should
    have complete watermark lineage. Missing watermarks on sensitive nodes
    may indicate attack vectors.

    Args:
        graph: CausalGraph instance

    Returns:
        Ratio of sensitive nodes with watermark (0.0 to 1.0), 0.0 if no metadata
    """
    sensitive_nodes = _get_watermark_sensitive_nodes(graph)
    if sensitive_nodes is None or len(sensitive_nodes) == 0:
        return 0.0

    watermark_tagged = _get_watermark_tagged_nodes(graph)
    if watermark_tagged is None:
        return 0.0

    # Count sensitive nodes that have watermark tags
    covered_sensitive = len(sensitive_nodes & watermark_tagged)

    return covered_sensitive / len(sensitive_nodes)


def compute_watermark_tampered(graph: CausalGraph) -> int:
    """
    Detect if any sensitive node lacks expected watermark lineage.

    Returns 1 if tampering detected (any sensitive node without watermark),
    0 if all sensitive nodes have proper watermark coverage or no metadata.

    Args:
        graph: CausalGraph instance

    Returns:
        1 if tampering detected, 0 otherwise
    """
    sensitive_nodes = _get_watermark_sensitive_nodes(graph)
    if sensitive_nodes is None or len(sensitive_nodes) == 0:
        return 0

    tampered_nodes = _get_watermark_tampered_nodes(graph)
    if tampered_nodes is not None and len(tampered_nodes) > 0:
        return 1

    # Alternatively, check if any sensitive node lacks watermark
    watermark_tagged = _get_watermark_tagged_nodes(graph)
    if watermark_tagged is None:
        return 0

    # If any sensitive node is not watermarked, consider it tampered
    unwatermarked_sensitive = sensitive_nodes - watermark_tagged
    return 1 if len(unwatermarked_sensitive) > 0 else 0


def compute_watermark_gap_count(graph: CausalGraph) -> int:
    """
    Count the number of sensitive nodes missing watermark lineage.

    This provides more granularity than the binary tampered flag,
    indicating the extent of potential manipulation.

    Args:
        graph: CausalGraph instance

    Returns:
        Number of sensitive nodes without watermark, 0 if no metadata
    """
    sensitive_nodes = _get_watermark_sensitive_nodes(graph)
    if sensitive_nodes is None or len(sensitive_nodes) == 0:
        return 0

    # First check explicit tampered nodes metadata
    tampered_nodes = _get_watermark_tampered_nodes(graph)
    if tampered_nodes is not None:
        # Return count of tampered nodes that are sensitive
        return len(tampered_nodes & sensitive_nodes)

    # Fallback: compute from watermark coverage
    watermark_tagged = _get_watermark_tagged_nodes(graph)
    if watermark_tagged is None:
        return 0

    unwatermarked_sensitive = sensitive_nodes - watermark_tagged
    return len(unwatermarked_sensitive)


def compute_watermark_first_gap_depth(graph: CausalGraph) -> int:
    """
    Find the depth of the first node where watermark lineage breaks.

    This indicates how early in the causal chain the integrity was compromised.
    Earlier breaks (lower depth) may indicate more fundamental injection points.

    Args:
        graph: CausalGraph instance

    Returns:
        Depth of first gap, -1 if no gaps found, 0 if no metadata
    """
    sensitive_nodes = _get_watermark_sensitive_nodes(graph)
    if sensitive_nodes is None or len(sensitive_nodes) == 0:
        return 0

    # Get nodes with gaps (tampered or unwatermarked sensitive nodes)
    tampered_nodes = _get_watermark_tampered_nodes(graph)
    if tampered_nodes is not None and len(tampered_nodes) > 0:
        gap_nodes = tampered_nodes & sensitive_nodes
    else:
        watermark_tagged = _get_watermark_tagged_nodes(graph)
        if watermark_tagged is None:
            return 0
        gap_nodes = sensitive_nodes - watermark_tagged

    if len(gap_nodes) == 0:
        return -1  # No gaps found

    # Find the minimum depth among gap nodes
    min_depth = float('inf')
    for node_id in gap_nodes:
        depth = graph.get_node_depth(node_id)
        min_depth = min(min_depth, depth)

    return int(min_depth) if min_depth != float('inf') else 0


def get_watermark_statistics(graph: CausalGraph) -> Dict[str, Any]:
    """
    Get comprehensive watermark statistics for a graph.

    Args:
        graph: CausalGraph instance

    Returns:
        Dictionary with all watermark-related metrics
    """
    return {
        "watermark_coverage": compute_watermark_coverage(graph),
        "watermark_propagation_depth": compute_watermark_propagation_depth(graph),
        "watermark_sensitive_coverage": compute_watermark_sensitive_coverage(graph),
        "watermark_tampered": compute_watermark_tampered(graph),
        "watermark_gap_count": compute_watermark_gap_count(graph),
        "watermark_first_gap_depth": compute_watermark_first_gap_depth(graph),
    }


# =========================================================================
# Helper functions for accessing watermark metadata
# =========================================================================


def _infer_provenance_from_nodes(graph: CausalGraph) -> tuple:
    """
    Infer watermark provenance from node attributes when metadata is missing.

    This allows watermark features to work on pre-built graphs that lack
    explicit provenance metadata. Uses existing provenance.is_untrusted and
    provenance.injection_detected fields if present.

    Returns:
        Tuple of (tagged_nodes, sensitive_nodes, tampered_nodes)
    """
    tagged_nodes = set()
    sensitive_nodes = set()
    tampered_nodes = set()

    # Untrusted domain patterns
    untrusted_patterns = [
        "forum", "reddit", "social", "comment", "post",
        "attacker", "evil", "malicious", "external",
    ]

    # Sensitive action types
    sensitive_types = {
        "tool_call", "code_execution", "execute_python", "execute_bash",
        "send_email", "submit_form", "post_comment", "api_call",
        "write_file", "database_write", "fill_form", "submit",
    }

    for node_id in graph.graph.nodes():
        node_data = graph.get_node_data(node_id) or {}

        # Get action type
        action_type = str(node_data.get("action_type", "")).lower()

        # Get domain
        domain = str(node_data.get("domain", "")).lower()

        # Check if sensitive
        if action_type in sensitive_types:
            sensitive_nodes.add(node_id)

        # Check existing provenance data (from graph construction)
        provenance = node_data.get("provenance", {})
        is_untrusted = False

        if provenance:
            # Use existing provenance flags if available
            is_untrusted = provenance.get("is_untrusted", False)
            injection_detected = provenance.get("injection_detected", False)

            if injection_detected:
                is_untrusted = True

            # Check untrusted_domains list
            untrusted_domains = provenance.get("untrusted_domains", [])
            if untrusted_domains:
                is_untrusted = True

        # Additional domain checks
        if not is_untrusted:
            for pattern in untrusted_patterns:
                if pattern in domain:
                    is_untrusted = True
                    break

        # Check observation chunks for injection patterns
        if not is_untrusted:
            observation_chunks = node_data.get("observation_chunks", [])
            for chunk in observation_chunks:
                content = str(chunk.get("content", "")).lower()
                if any(ind in content for ind in ["attention", "ignore previous", "new instruction", "objective:"]):
                    is_untrusted = True
                    break

        # Tag as watermarked if NOT untrusted
        if not is_untrusted:
            tagged_nodes.add(node_id)

    # Propagate watermark along data dependency edges
    # (untrusted nodes block propagation)
    changed = True
    iterations = 0
    max_iterations = len(graph.graph.nodes()) * 2
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for edge in graph.get_all_edges():
            src = edge.source_action_id
            tgt = edge.target_action_id
            if src in tagged_nodes and tgt not in tagged_nodes:
                # Check if target is untrusted
                target_data = graph.get_node_data(tgt) or {}
                target_prov = target_data.get("provenance", {})
                target_untrusted = target_prov.get("is_untrusted", False) or target_prov.get("injection_detected", False)
                if not target_untrusted:
                    tagged_nodes.add(tgt)
                    changed = True

    # Find tampered nodes (sensitive but not tagged)
    tampered_nodes = sensitive_nodes - tagged_nodes

    return tagged_nodes, sensitive_nodes, tampered_nodes


def _get_watermark_tagged_nodes(graph: CausalGraph) -> Optional[Set[int]]:
    """
    Get set of watermark-tagged node IDs from graph metadata.

    If metadata is not present, attempts to infer provenance from node attributes.

    Args:
        graph: CausalGraph instance

    Returns:
        Set of node IDs with watermark tags, or None if cannot determine
    """
    tagged = graph.get_metadata("watermark_tagged_nodes")

    # If no metadata, try to infer from node attributes
    if tagged is None:
        inferred = _infer_provenance_from_nodes(graph)
        tagged = inferred[0]  # tagged_nodes
        if tagged:
            return tagged
        return None

    # Handle both set and list inputs
    if isinstance(tagged, set):
        return tagged
    elif isinstance(tagged, (list, tuple)):
        return set(tagged)
    else:
        return None


def _get_watermark_sensitive_nodes(graph: CausalGraph) -> Optional[Set[int]]:
    """
    Get set of sensitive node IDs from graph metadata.

    Sensitive nodes are typically sinks like code execution, data exfiltration,
    or privilege escalation points that require watermark verification.

    If metadata is not present, attempts to infer from node attributes.

    Args:
        graph: CausalGraph instance

    Returns:
        Set of sensitive node IDs, or None if cannot determine
    """
    sensitive = graph.get_metadata("watermark_sensitive_nodes")

    # If no metadata, try to infer from node attributes
    if sensitive is None:
        inferred = _infer_provenance_from_nodes(graph)
        sensitive = inferred[1]  # sensitive_nodes
        if sensitive:
            return sensitive
        return None

    # Handle both set and list inputs
    if isinstance(sensitive, set):
        return sensitive
    elif isinstance(sensitive, (list, tuple)):
        return set(sensitive)
    else:
        return None


def _get_watermark_tampered_nodes(graph: CausalGraph) -> Optional[Set[int]]:
    """
    Get set of tampered node IDs from graph metadata.

    Tampered nodes are those where watermark lineage was expected but
    not found, indicating potential injection or manipulation.

    If metadata is not present, attempts to infer from node attributes.

    Args:
        graph: CausalGraph instance

    Returns:
        Set of tampered node IDs, or None if cannot determine
    """
    tampered = graph.get_metadata("watermark_tampered_nodes")

    # If no metadata, try to infer from node attributes
    if tampered is None:
        inferred = _infer_provenance_from_nodes(graph)
        tampered = inferred[2]  # tampered_nodes
        if tampered:
            return tampered
        return None

    # Handle both set and list inputs
    if isinstance(tampered, set):
        return tampered
    elif isinstance(tampered, (list, tuple)):
        return set(tampered)
    else:
        return None
