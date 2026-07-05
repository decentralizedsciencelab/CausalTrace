"""
Edge detection for causal graphs.

Detects three types of causal edges:
- Data Dependency: Action j uses data produced by action i
- Trust Transfer: Action j executes pattern/code from context of action i
- State Enablement: Action i creates state required by action j

Formal Specifications:
    See docs/formal_algorithms.md for mathematical definitions:
    - Algorithm 1: Data Dependency Detection (4 formal criteria)
    - Algorithm 2: Trust Transfer Detection (5 formal criteria)
    - Algorithm 3: State Enablement Detection (5 formal criteria)

    Each detector implements the corresponding algorithm using heuristic
    rules that match the formal criteria.
"""

from dataclasses import dataclass
from typing import Optional, List
from causaltrace.models.trajectory import Action, ActionType, Trajectory
from causaltrace.graph.causal_graph import CausalEdge, EdgeType
from causaltrace.utils.url import extract_domain


@dataclass
class DetectorConfig:
    """Configuration for edge detectors with tunable thresholds."""

    # Data dependency thresholds
    min_string_match_length: int = 10  # Minimum chars for generic data flow match

    # Trust transfer thresholds
    min_tool_name_length: int = 3  # Minimum tool name length to match

    # State enablement thresholds
    navigation_enablement_window: float = 60.0  # Seconds for navigation->action
    temporal_state_window: float = 30.0  # Seconds for sequential same-domain actions

    # Instruction keywords
    type_instruction_keywords: tuple = ("type", "enter", "input", "fill", "write")
    click_instruction_keywords: tuple = ("click", "press", "submit", "select")


# Default configuration
DEFAULT_CONFIG = DetectorConfig()


class DataDependencyDetector:
    """
    Detect data dependencies between actions.

    A data dependency exists when action j uses data that was produced by action i.

    Implements Algorithm 1 (Data Dependency Detection) from docs/formal_algorithms.md.

    Formal Criteria:
        1. Direct data flow: D^prod_i ∩ D^cons_j ≠ ∅
        2. String containment: normalize(d_p) ⊆ normalize(d_c)
        3. Result-to-input flow: For (READ->->TYPE), (EXTRACT->->SEND_EMAIL), (READ->->TOOL_CALL)
        4. Generic result propagation: result_i appears in target_j (|result| > threshold)

    All criteria require temporal constraint: timestamp_i < timestamp_j
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize detector with configuration.

        Args:
            config: Detection thresholds (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG

    def detect(
        self,
        source: Action,
        target: Action,
        trajectory: Trajectory
    ) -> Optional[CausalEdge]:
        """
        Detect if target action uses data produced by source action.

        Logic:
        1. Check if any data_produced by source is in data_consumed by target
        2. For READ -> TYPE: check if extracted text matches typed text
        3. For EXTRACT -> SEND_EMAIL: check if extracted data is in email body
        4. For READ -> TOOL_CALL: check if read content is in tool arguments

        Args:
            source: Source action
            target: Target action
            trajectory: Full trajectory for context

        Returns:
            CausalEdge if dependency detected, None otherwise
        """
        # Rule 1: Direct data flow (data_produced -> data_consumed)
        if source.data_produced and target.data_consumed:
            for produced in source.data_produced:
                for consumed in target.data_consumed:
                    # Check for exact match or substring match
                    if produced == consumed or produced in consumed or consumed in produced:
                        return CausalEdge(
                            source_action_id=source.action_id,
                            target_action_id=target.action_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            metadata={
                                "evidence": f"Data '{produced}' produced by source and consumed by target",
                                "match_type": "direct"
                            }
                        )

        # Rule 2: READ -> TYPE pattern
        if source.action_type == ActionType.READ and target.action_type == ActionType.TYPE:
            # Check if result of READ appears in target's target (typed text)
            if source.result and target.target:
                source_text = str(source.result).lower().strip()
                target_text = str(target.target).lower().strip()

                if source_text and target_text and source_text in target_text:
                    return CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={
                            "evidence": "Text from READ action appears in TYPE action",
                            "match_type": "read_to_type"
                        }
                    )

        # Rule 3: EXTRACT -> SEND_EMAIL pattern
        if source.action_type == ActionType.EXTRACT and target.action_type == ActionType.SEND_EMAIL:
            if source.result and target.context:
                source_text = str(source.result).lower()
                # Check if extracted data appears in email body
                email_body = str(target.context.get("body", "")).lower()

                if source_text and email_body and source_text in email_body:
                    return CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={
                            "evidence": "Extracted data appears in email body",
                            "match_type": "extract_to_email"
                        }
                    )

        # Rule 4: READ -> TOOL_CALL pattern
        if source.action_type == ActionType.READ and target.action_type == ActionType.TOOL_CALL:
            if source.result and target.target:
                source_text = str(source.result).lower()
                tool_args = str(target.target).lower()

                if source_text and tool_args and source_text in tool_args:
                    return CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={
                            "evidence": "Read content appears in tool arguments",
                            "match_type": "read_to_tool"
                        }
                    )

        # Rule 5: Any action producing data -> Any action consuming similar data
        if source.result and (target.target or target.context):
            source_result = str(source.result).lower()
            target_combined = str(target.target).lower() + " " + str(target.context).lower()

            # Only trigger if there's a substantial match
            if len(source_result) > self.config.min_string_match_length and source_result in target_combined:
                return CausalEdge(
                    source_action_id=source.action_id,
                    target_action_id=target.action_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={
                        "evidence": "Source result appears in target action",
                        "match_type": "generic_data_flow"
                    }
                )

        return None


class TrustTransferDetector:
    """
    Detect trust transfer dependencies.

    A trust transfer exists when action j executes code/pattern from context of action i.

    Implements Algorithm 2 (Trust Transfer Detection) from docs/formal_algorithms.md.

    Formal Criteria:
        1. Code execution: tool_name(target_j) ⊆ result_i (READ->->TOOL_CALL pattern)
        2. Link following: domain(target_j) ⊆ context_i (NAVIGATE->->NAVIGATE)
        3. Instruction following: keyword ∈ K_type ⊆ result_i (READ->->TYPE/CLICK)
        4. Cross-domain trust: domain_i ≠ domain_j (NAVIGATE->->SEND_EMAIL/TOOL_CALL)
        5. Similarity-based: sim(result_i, target_j) > τ (optional, Jaccard similarity)

    All criteria require temporal constraint: timestamp_i < timestamp_j
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize detector with configuration.

        Args:
            config: Detection thresholds (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG

    def detect(
        self,
        source: Action,
        target: Action,
        trajectory: Trajectory
    ) -> Optional[CausalEdge]:
        """
        Detect if target executes code/pattern from source context.

        Logic:
        1. If source is READ and target is TOOL_CALL: check if tool name/args came from read content
        2. If source is NAVIGATE and target is NAVIGATE: check if URL came from previous page
        3. Look for instruction patterns in source.context that match target behavior
        4. Check for code execution patterns (source contains code, target executes similar actions)

        Args:
            source: Source action
            target: Target action
            trajectory: Full trajectory for context

        Returns:
            CausalEdge if trust transfer detected, None otherwise
        """
        # Rule 1: READ -> TOOL_CALL (untrusted content suggests tool usage)
        if source.action_type == ActionType.READ and target.action_type == ActionType.TOOL_CALL:
            # Check if tool name appears in read content
            if source.result and target.target:
                source_content = str(source.result).lower()
                tool_name = str(target.target).split("(")[0].strip().lower()

                # Look for tool name or similar patterns in source
                if tool_name and len(tool_name) > self.config.min_tool_name_length and tool_name in source_content:
                    return CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.TRUST_TRANSFER,
                        metadata={
                            "evidence": f"Tool name '{tool_name}' appears in read content",
                            "match_type": "read_to_tool_execution"
                        }
                    )

        # Rule 2: NAVIGATE -> NAVIGATE (URL from previous page)
        if source.action_type == ActionType.NAVIGATE and target.action_type == ActionType.NAVIGATE:
            # Check if target URL appeared in source page context
            if source.context and target.target:
                source_page = str(source.context).lower()

                # Extract domain from target URL
                target_domain = extract_domain(target.target)

                if target_domain and target_domain in source_page:
                    return CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.TRUST_TRANSFER,
                        metadata={
                            "evidence": f"Navigation target '{target_domain}' found in source page",
                            "match_type": "navigate_to_linked_url"
                        }
                    )

        # Rule 3: READ -> TYPE (instructions to type specific text)
        if source.action_type == ActionType.READ and target.action_type == ActionType.TYPE:
            # Look for instruction keywords in source content
            if source.result and target.target:
                source_content = str(source.result).lower()

                for keyword in self.config.type_instruction_keywords:
                    if keyword in source_content:
                        return CausalEdge(
                            source_action_id=source.action_id,
                            target_action_id=target.action_id,
                            edge_type=EdgeType.TRUST_TRANSFER,
                            metadata={
                                "evidence": f"Instruction keyword '{keyword}' found in source",
                                "match_type": "instruction_following"
                            }
                        )

        # Rule 4: READ -> CLICK (instructions to click)
        if source.action_type == ActionType.READ and target.action_type == ActionType.CLICK:
            if source.result:
                source_content = str(source.result).lower()

                for keyword in self.config.click_instruction_keywords:
                    if keyword in source_content:
                        return CausalEdge(
                            source_action_id=source.action_id,
                            target_action_id=target.action_id,
                            edge_type=EdgeType.TRUST_TRANSFER,
                            metadata={
                                "evidence": f"Click instruction '{keyword}' found in source",
                                "match_type": "instruction_following"
                            }
                        )

        # Rule 5: Cross-domain navigation (sign of trust transfer)
        if source.action_type == ActionType.NAVIGATE and target.action_type in [ActionType.SEND_EMAIL, ActionType.TOOL_CALL]:
            source_domain = source.domain or extract_domain(source.target)
            target_domain = target.domain

            # If domains differ and target is a sensitive action, it's suspicious
            if source_domain and target_domain and source_domain != target_domain:
                return CausalEdge(
                    source_action_id=source.action_id,
                    target_action_id=target.action_id,
                    edge_type=EdgeType.TRUST_TRANSFER,
                    metadata={
                        "evidence": f"Cross-domain action: {source_domain} -> {target_domain}",
                        "match_type": "cross_domain_trust"
                    }
                )

        return None


class StateEnablementDetector:
    """
    Detect state enablement dependencies.

    A state enablement exists when action i creates state required by action j.

    Implements Algorithm 3 (State Enablement Detection) from docs/formal_algorithms.md.

    Formal Criteria:
        1. Authentication: LOGIN to domain_i enables actions on domain_i
        2. Navigation: NAVIGATE to domain_i enables interactions (within time window)
        3. File state: DOWNLOAD enables UPLOAD
        4. Data accumulation: EXTRACT enables SEND_EMAIL
        5. Temporal state: Sequential actions on same domain (within time window, different types)

    All criteria require temporal constraint: timestamp_i < timestamp_j

    State Persistence Model:
        - S_auth(t): Set of authenticated domains at time t
        - S_files(t): Set of available local files at time t
        - S_data(t): Accumulated data at time t
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize detector with configuration.

        Args:
            config: Detection thresholds (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG

    def detect(
        self,
        source: Action,
        target: Action,
        trajectory: Trajectory
    ) -> Optional[CausalEdge]:
        """
        Detect if source creates state required by target.

        Logic:
        1. Authentication: source adds to authenticated_services, target requires auth
        2. Permissions: source grants permission, target uses it
        3. Session state: source creates session, target uses session
        4. Navigation state: source navigates, target acts on that page
        5. Data accumulation: source adds data, target uses accumulated data

        Args:
            source: Source action
            target: Target action
            trajectory: Full trajectory for context

        Returns:
            CausalEdge if state enablement detected, None otherwise
        """
        # Rule 1: LOGIN -> Actions on authenticated service
        if source.action_type == ActionType.LOGIN:
            source_domain = source.domain or extract_domain(source.target)
            target_domain = target.domain

            # If target action is on same domain as login, there's state enablement
            if source_domain and target_domain and source_domain == target_domain:
                return CausalEdge(
                    source_action_id=source.action_id,
                    target_action_id=target.action_id,
                    edge_type=EdgeType.STATE_ENABLEMENT,
                    metadata={
                        "evidence": f"Login to {source_domain} enables action on same domain",
                        "match_type": "authentication"
                    }
                )

        # Rule 2: NAVIGATE -> Actions on same page
        if source.action_type == ActionType.NAVIGATE:
            source_url = source.target
            # Check if target action is on the same page
            if target.action_type in [ActionType.CLICK, ActionType.TYPE, ActionType.READ, ActionType.EXTRACT]:
                # If target happens shortly after source and on same domain
                source_domain = extract_domain(source_url)
                target_domain = target.domain

                if source_domain and target_domain and source_domain == target_domain:
                    # Check if target is within reasonable time after source
                    if source.timestamp and target.timestamp:
                        time_diff = target.timestamp - source.timestamp
                        if 0 < time_diff < self.config.navigation_enablement_window:
                            return CausalEdge(
                                source_action_id=source.action_id,
                                target_action_id=target.action_id,
                                edge_type=EdgeType.STATE_ENABLEMENT,
                                metadata={
                                    "evidence": f"Navigation to {source_domain} enables page interaction",
                                    "match_type": "navigation_enablement",
                                    "time_diff": time_diff
                                }
                            )

        # Rule 3: DOWNLOAD -> Actions using downloaded file
        if source.action_type == ActionType.DOWNLOAD and target.action_type == ActionType.UPLOAD:
            # If a file is downloaded then uploaded, download enabled the upload
            return CausalEdge(
                source_action_id=source.action_id,
                target_action_id=target.action_id,
                edge_type=EdgeType.STATE_ENABLEMENT,
                metadata={
                    "evidence": "Download enables subsequent upload",
                    "match_type": "file_state"
                }
            )

        # Rule 4: EXTRACT -> SEND_EMAIL (accumulated data enables email)
        if source.action_type == ActionType.EXTRACT and target.action_type == ActionType.SEND_EMAIL:
            # Extract accumulates data, email uses accumulated data
            return CausalEdge(
                source_action_id=source.action_id,
                target_action_id=target.action_id,
                edge_type=EdgeType.STATE_ENABLEMENT,
                metadata={
                    "evidence": "Data extraction enables email composition",
                    "match_type": "data_accumulation"
                }
            )

        # Rule 5: Sequential actions on same domain (temporal state)
        if source.domain and target.domain and source.domain == target.domain:
            # Actions on same domain in sequence likely have state dependency
            if source.timestamp and target.timestamp:
                time_diff = target.timestamp - source.timestamp
                # Within temporal window and different action types
                if 0 < time_diff < self.config.temporal_state_window and source.action_type != target.action_type:
                    return CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={
                            "evidence": f"Sequential actions on {source.domain}",
                            "match_type": "temporal_state",
                            "time_diff": time_diff
                        }
                    )

        return None


class EdgeDetector:
    """
    Unified edge detector combining all three edge types.

    This class consolidates DataDependencyDetector, TrustTransferDetector, and
    StateEnablementDetector into a single interface for simpler usage.

    Example:
        detector = EdgeDetector()
        edges = detector.detect_all(source_action, target_action, trajectory)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize unified detector with configuration.

        Args:
            config: Detection thresholds (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        self.data_detector = DataDependencyDetector(self.config)
        self.trust_detector = TrustTransferDetector(self.config)
        self.state_detector = StateEnablementDetector(self.config)

    def detect_all(
        self,
        source: Action,
        target: Action,
        trajectory: Trajectory
    ) -> List[CausalEdge]:
        """
        Detect all types of causal edges between two actions.

        Args:
            source: Source action
            target: Target action
            trajectory: Full trajectory for context

        Returns:
            List of detected CausalEdges (may be empty, or contain multiple)
        """
        edges = []

        # Check each edge type
        data_edge = self.data_detector.detect(source, target, trajectory)
        if data_edge:
            edges.append(data_edge)

        trust_edge = self.trust_detector.detect(source, target, trajectory)
        if trust_edge:
            edges.append(trust_edge)

        state_edge = self.state_detector.detect(source, target, trajectory)
        if state_edge:
            edges.append(state_edge)

        return edges

    def detect_first(
        self,
        source: Action,
        target: Action,
        trajectory: Trajectory
    ) -> Optional[CausalEdge]:
        """
        Detect the first (highest priority) edge between two actions.

        Priority order: DATA_DEPENDENCY > TRUST_TRANSFER > STATE_ENABLEMENT

        Args:
            source: Source action
            target: Target action
            trajectory: Full trajectory for context

        Returns:
            First detected CausalEdge, or None
        """
        # Check in priority order
        edge = self.data_detector.detect(source, target, trajectory)
        if edge:
            return edge

        edge = self.trust_detector.detect(source, target, trajectory)
        if edge:
            return edge

        edge = self.state_detector.detect(source, target, trajectory)
        if edge:
            return edge

        return None
