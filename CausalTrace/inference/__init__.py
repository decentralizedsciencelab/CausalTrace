"""
Data flow inference module for CausalTrace.

This module provides tools for inferring data_produced and data_consumed
annotations on trajectory actions, enabling accurate causal edge detection.

Key Classes:
    DataFlowInferencer: Main class for annotating trajectories
    DataItem: Represents a piece of data flowing through the trajectory
    DataFlow: Represents a data flow from source to sink action

Key Functions:
    infer_data_flow: Convenience function for trajectory annotation

Usage:
    from causaltrace.inference import DataFlowInferencer, infer_data_flow

    # Option 1: Use convenience function
    annotated_trajectory = infer_data_flow(trajectory)

    # Option 2: Use class for more control
    inferencer = DataFlowInferencer(
        enable_credential_detection=True,
        enable_url_tracking=True,
        min_match_length=8,
    )
    annotated = inferencer.infer_data_flow(trajectory)
    flows = inferencer.get_detected_flows()
    trust_transfers = inferencer.get_trust_transfers()
"""

from causaltrace.inference.data_flow import (
    DataFlowInferencer,
    DataItem,
    DataFlow,
    infer_data_flow,
)

from causaltrace.inference.patterns import (
    DataItemType,
    PatternMatch,
    extract_credentials,
    extract_urls,
    extract_dom_elements,
    detect_injection_content,
    extract_identifiers,
    is_attacker_url,
)

__all__ = [
    # Main classes
    "DataFlowInferencer",
    "DataItem",
    "DataFlow",
    # Types
    "DataItemType",
    "PatternMatch",
    # Convenience functions
    "infer_data_flow",
    # Pattern extraction
    "extract_credentials",
    "extract_urls",
    "extract_dom_elements",
    "detect_injection_content",
    "extract_identifiers",
    "is_attacker_url",
]
