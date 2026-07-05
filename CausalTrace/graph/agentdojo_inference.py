"""
AgentDojo-specific edge inference for causal graphs.

AgentDojo trajectories have specific patterns that require specialized
edge detection:

1. Tool chains: read_tool -> action_tool (e.g., list_files -> send_email)
2. Sequential tool calls with implicit data flow
3. YAML-structured tool outputs containing data used by subsequent calls
4. Cross-domain data flow (drive -> email, calendar -> email, etc.)

Key differences from browser agent trajectories:
- No timestamps (timestamp=None)
- Rich tool result strings (YAML format with file content, emails, etc.)
- Semantic data annotations (file_list, email_content) that don't directly match
- Tool call arguments often contain data extracted from prior tool results
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass

from causaltrace.models.trajectory import Action, ActionType, Trajectory
from causaltrace.graph.causal_graph import CausalEdge, EdgeType
from causaltrace.graph.edge_inference import EdgeInferenceEngine, InferenceResult


# AgentDojo tool categories
READ_TOOLS = {
    "list_files", "search_files", "search_files_by_filename", "get_file_by_id",
    "get_received_emails", "get_sent_emails", "search_emails", "get_email_by_id",
    "get_unread_emails", "search_contacts", "search_contacts_by_name", "get_contacts",
    "search_calendar_events", "get_day_calendar_events", "get_current_day",
    "get_balance", "get_iban", "get_scheduled_transactions", "get_most_recent_transactions",
    "get_user_info", "get_user_name", "get_user_email",
    "read_channel_messages", "read_inbox", "get_channels", "get_users_in_channel",
    "get_flight_information", "get_all_restaurants_in_city", "get_rating_reviews_for_restaurants",
    "get_all_hotels_in_city", "get_rating_reviews_for_hotels", "get_hotels_prices",
    "get_all_car_rental_companies_in_city", "get_rating_reviews_for_car_rental",
    "get_car_types_available", "get_car_price_per_day", "get_car_fuel_options",
    "verify_reservation",
}

WRITE_TOOLS = {
    "send_email", "delete_email",
    "create_file", "delete_file", "append_to_file", "share_file",
    "create_calendar_event", "delete_calendar_event", "update_calendar_event",
    "cancel_calendar_event", "invite_to_event",
    "schedule_transaction", "send_money", "update_scheduled_transaction",
    "reserve_restaurant", "reserve_hotel", "reserve_car", "cancel_reservation",
    "send_channel_message", "send_direct_message",
    "add_user_to_channel", "remove_user_from_channel",
    "get_webpage", "post_webpage",
}

# Domain mappings for cross-domain flow detection
TOOL_DOMAINS = {
    # Drive
    "list_files": "drive", "search_files": "drive", "search_files_by_filename": "drive",
    "get_file_by_id": "drive", "create_file": "drive", "delete_file": "drive",
    "append_to_file": "drive", "share_file": "drive",
    # Email
    "get_received_emails": "email", "get_sent_emails": "email", "search_emails": "email",
    "get_email_by_id": "email", "send_email": "email", "delete_email": "email",
    "get_unread_emails": "email",
    # Contacts
    "search_contacts": "contacts", "search_contacts_by_name": "contacts",
    "get_contacts": "contacts",
    # Calendar
    "search_calendar_events": "calendar", "get_day_calendar_events": "calendar",
    "get_current_day": "calendar", "create_calendar_event": "calendar",
    "delete_calendar_event": "calendar", "update_calendar_event": "calendar",
    "cancel_calendar_event": "calendar", "invite_to_event": "calendar",
    # Banking
    "get_balance": "banking", "get_iban": "banking",
    "get_scheduled_transactions": "banking", "get_most_recent_transactions": "banking",
    "schedule_transaction": "banking", "send_money": "banking",
    "update_scheduled_transaction": "banking",
    # User
    "get_user_info": "user", "get_user_name": "user", "get_user_email": "user",
    # Slack
    "read_channel_messages": "slack", "read_inbox": "slack",
    "send_channel_message": "slack", "send_direct_message": "slack",
    "get_channels": "slack", "add_user_to_channel": "slack",
    "remove_user_from_channel": "slack", "get_users_in_channel": "slack",
    # Travel
    "get_flight_information": "travel", "get_all_restaurants_in_city": "travel",
    "get_rating_reviews_for_restaurants": "travel", "reserve_restaurant": "travel",
    "get_all_hotels_in_city": "travel", "get_rating_reviews_for_hotels": "travel",
    "reserve_hotel": "travel", "get_hotels_prices": "travel",
    "get_all_car_rental_companies_in_city": "travel", "get_rating_reviews_for_car_rental": "travel",
    "reserve_car": "travel", "get_car_types_available": "travel",
    "get_car_price_per_day": "travel", "get_car_fuel_options": "travel",
    "verify_reservation": "travel", "cancel_reservation": "travel",
    # Web
    "get_webpage": "web", "post_webpage": "web",
}

# Patterns for extracting data from tool results
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
IBAN_PATTERN = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{4,30}")
FILE_ID_PATTERN = re.compile(r"id_:\s*['\"]?(\d+)['\"]?")
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
RECIPIENT_PATTERN = re.compile(r"recipients?:\s*\[?([^\]]+)\]?", re.IGNORECASE)


@dataclass
class ExtractedData:
    """Data extracted from tool result for matching."""
    emails: Set[str]
    ibans: Set[str]
    file_ids: Set[str]
    dates: Set[str]
    text_snippets: Set[str]
    has_injection: bool


class AgentDojoEdgeInferenceEngine(EdgeInferenceEngine):
    """
    Specialized edge inference for AgentDojo trajectories.

    AgentDojo has specific patterns:
    1. READ tool -> WRITE tool chains (data extracted then used)
    2. File content contains data that appears in subsequent tool args
    3. Cross-domain flows (drive -> email, contacts -> email, etc.)
    4. YAML-structured results with nested content

    This engine augments the base inference with AgentDojo-specific strategies:
    - Tool chain detection (read -> write on same data)
    - Content extraction and matching from YAML results
    - Cross-domain flow detection
    - Sequential tool call linking
    - State enablement (get_current_day -> calendar)
    """

    # Injection indicators in AgentDojo
    INJECTION_INDICATORS = [
        "IMPORTANT", "IGNORE", "TODO:", "<INFORMATION>",
        "instruction", "Ignore all previous", "new task",
        "ATTENTION", "must", "urgently", "immediately",
    ]

    # State-providing tools that enable other tools
    STATE_ENABLERS = {
        "get_current_day": {"create_calendar_event", "get_day_calendar_events", "search_calendar_events"},
        "get_user_info": {"send_email", "create_file", "schedule_transaction"},
        "get_iban": {"schedule_transaction", "send_money"},
        "get_balance": {"schedule_transaction", "send_money"},
        "search_emails": {"send_email", "delete_email"},
        "search_files_by_filename": {"append_to_file", "delete_file", "share_file", "get_file_by_id"},
        "list_files": {"append_to_file", "delete_file", "share_file", "get_file_by_id"},
    }

    def __init__(
        self,
        use_tool_chain: bool = True,
        use_content_extraction: bool = True,
        use_cross_domain: bool = True,
        use_sequential: bool = True,
        use_state_enablement: bool = True,
        use_all_sequential: bool = True,  # NEW: link all consecutive tool calls
        min_content_length: int = 4,
        **kwargs
    ):
        """
        Initialize AgentDojo edge inference engine.

        Args:
            use_tool_chain: Detect read->write tool chains
            use_content_extraction: Extract and match content from results
            use_cross_domain: Detect cross-domain data flows
            use_sequential: Link sequential same-domain actions
            use_state_enablement: Detect state enablement patterns
            use_all_sequential: Link all consecutive tool calls
            min_content_length: Minimum length for content matching
        """
        # Disable irrelevant base strategies for AgentDojo
        kwargs.setdefault("use_explicit", True)  # Use semantic IDs
        kwargs.setdefault("use_delegation", False)  # No MAS delegation
        kwargs.setdefault("use_sequential", False)  # We handle this differently
        kwargs.setdefault("use_content", False)  # We have custom content matching
        super().__init__(**kwargs)

        self.use_tool_chain = use_tool_chain
        self.use_content_extraction = use_content_extraction
        self.use_cross_domain = use_cross_domain
        self.use_sequential_domain = use_sequential
        self.use_state_enablement = use_state_enablement
        self.use_all_sequential = use_all_sequential
        self.min_content_length = min_content_length

    def infer_edges(self, trajectory: Trajectory) -> InferenceResult:
        """Infer edges with AgentDojo-specific strategies."""
        all_edges: List[CausalEdge] = []
        stats = {
            "explicit": 0,
            "tool_chain": 0,
            "content_match": 0,
            "cross_domain": 0,
            "sequential": 0,
            "state_enablement": 0,
            "all_sequential": 0,
            "total": 0,
        }

        seen_edges: Set[Tuple[int, int, str]] = set()

        def add_edge(edge: CausalEdge, strategy: str) -> bool:
            """Add edge if not duplicate."""
            # Validate temporal order
            if edge.source_action_id >= edge.target_action_id:
                return False

            key = (edge.source_action_id, edge.target_action_id, edge.edge_type.value)
            if key not in seen_edges:
                seen_edges.add(key)
                all_edges.append(edge)
                stats[strategy] += 1
                stats["total"] += 1
                return True
            return False

        # Strategy 1: All sequential tool calls (most aggressive, catches most edges)
        if self.use_all_sequential:
            for edge in self._infer_all_sequential(trajectory):
                add_edge(edge, "all_sequential")

        # Strategy 2: Explicit data annotations
        if self.use_explicit:
            for edge in self._infer_explicit(trajectory):
                add_edge(edge, "explicit")

        # Strategy 3: Tool chain detection (READ -> WRITE patterns)
        if self.use_tool_chain:
            for edge in self._infer_tool_chains(trajectory):
                add_edge(edge, "tool_chain")

        # Strategy 4: Content extraction and matching
        if self.use_content_extraction:
            for edge in self._infer_content_matches(trajectory):
                add_edge(edge, "content_match")

        # Strategy 5: Cross-domain flows
        if self.use_cross_domain:
            for edge in self._infer_cross_domain(trajectory):
                add_edge(edge, "cross_domain")

        # Strategy 6: Sequential same-domain linking
        if self.use_sequential_domain:
            for edge in self._infer_sequential_domain(trajectory):
                add_edge(edge, "sequential")

        # Strategy 7: State enablement patterns
        if self.use_state_enablement:
            for edge in self._infer_state_enablement(trajectory):
                add_edge(edge, "state_enablement")

        return InferenceResult(edges=all_edges, stats=stats)

    def _infer_tool_chains(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Detect READ tool -> WRITE tool chains.

        Pattern: A read tool fetches data, a write tool uses that data.
        E.g., list_files -> send_email, get_contacts -> send_email

        \1 read actions to ALL subsequent write actions
        without requiring exact content match (common in AgentDojo).
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        # Index read actions by domain and content
        read_actions: Dict[str, List[Action]] = {}  # domain -> actions

        for action in actions:
            tool_name = self._get_tool_name(action)
            if tool_name in READ_TOOLS:
                domain = TOOL_DOMAINS.get(tool_name, "unknown")
                if domain not in read_actions:
                    read_actions[domain] = []
                read_actions[domain].append(action)

        # For each write action, find potential source read actions
        for action in actions:
            tool_name = self._get_tool_name(action)
            if tool_name not in WRITE_TOOLS:
                continue

            action_domain = TOOL_DOMAINS.get(tool_name, "unknown")

            # Check all prior read actions
            for source_domain, source_actions in read_actions.items():
                for source in source_actions:
                    if source.action_id >= action.action_id:
                        continue

                    # Skip if source has no result (failed/empty)
                    if not source.result or str(source.result) in ("[]", "None", ""):
                        continue

                    # Determine edge type based on domain relationship
                    has_injection = self._has_injection_content(source)
                    if source_domain == action_domain:
                        edge_type = EdgeType.DATA_DEPENDENCY
                        evidence = f"Same-domain tool chain: {self._get_tool_name(source)} -> {tool_name}"
                    else:
                        # Cross-domain = trust transfer if has injection
                        edge_type = EdgeType.TRUST_TRANSFER if has_injection else EdgeType.DATA_DEPENDENCY
                        evidence = f"Cross-domain tool chain: {source_domain} -> {action_domain}"

                    # \1 even without exact content match
                    # In AgentDojo, read actions almost always inform subsequent writes
                    edge = CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=action.action_id,
                        edge_type=edge_type,
                        metadata={
                            "evidence": evidence,
                            "match_type": "tool_chain",
                            "source_tool": self._get_tool_name(source),
                            "target_tool": tool_name,
                            "source_domain": source_domain,
                            "target_domain": action_domain,
                            "has_injection": has_injection,
                        }
                    )
                    edges.append(edge)

        return edges

    def _infer_content_matches(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Match content from tool results to subsequent tool arguments.

        Extracts emails, IBANs, file IDs, dates from results and checks
        if they appear in later action arguments.
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        # Extract data from each action's result
        extracted_data: Dict[int, ExtractedData] = {}
        for action in actions:
            if action.result:
                extracted_data[action.action_id] = self._extract_data(str(action.result))

        # Match against subsequent action inputs
        for i, target in enumerate(actions):
            target_input = self._get_action_input_text(target)
            if not target_input or len(target_input) < self.min_content_length:
                continue

            target_input_lower = target_input.lower()

            for source in actions[:i]:
                if source.action_id not in extracted_data:
                    continue

                data = extracted_data[source.action_id]
                match_type = None
                matched_value = None

                # Check email matches
                for email in data.emails:
                    if email.lower() in target_input_lower:
                        match_type = "email"
                        matched_value = email
                        break

                # Check IBAN matches
                if not match_type:
                    for iban in data.ibans:
                        if iban.lower() in target_input_lower:
                            match_type = "iban"
                            matched_value = iban
                            break

                # Check file ID matches
                if not match_type:
                    for file_id in data.file_ids:
                        if file_id in target_input:
                            match_type = "file_id"
                            matched_value = file_id
                            break

                # Check date matches
                if not match_type:
                    for date in data.dates:
                        if date in target_input:
                            match_type = "date"
                            matched_value = date
                            break

                # Check text snippet matches
                if not match_type:
                    for snippet in data.text_snippets:
                        if len(snippet) >= 10 and snippet.lower() in target_input_lower:
                            match_type = "text_snippet"
                            matched_value = snippet[:50]
                            break

                if match_type:
                    # Determine edge type - trust transfer if source has injection
                    edge_type = EdgeType.TRUST_TRANSFER if data.has_injection else EdgeType.DATA_DEPENDENCY

                    edge = CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=edge_type,
                        metadata={
                            "evidence": f"Content match ({match_type}): '{matched_value}'",
                            "match_type": f"content_{match_type}",
                            "matched_value": matched_value,
                            "has_injection": data.has_injection,
                        }
                    )
                    edges.append(edge)

        return edges

    def _infer_cross_domain(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Detect cross-domain data flows.

        Pattern: Data from one domain (e.g., drive) used in another (e.g., email).
        These are important for attack detection as they often indicate
        exfiltration or injection flows.
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        for i, target in enumerate(actions):
            target_tool = self._get_tool_name(target)
            target_domain = TOOL_DOMAINS.get(target_tool, "unknown")

            for source in actions[:i]:
                source_tool = self._get_tool_name(source)
                source_domain = TOOL_DOMAINS.get(source_tool, "unknown")

                # Skip same-domain (handled by tool_chains)
                if source_domain == target_domain:
                    continue

                # Skip if both unknown
                if source_domain == "unknown" or target_domain == "unknown":
                    continue

                # Check for actual content flow
                if not self._has_data_flow(source, target):
                    continue

                # Cross-domain flow detected
                has_injection = self._has_injection_content(source)
                edge_type = EdgeType.TRUST_TRANSFER if has_injection else EdgeType.DATA_DEPENDENCY

                edge = CausalEdge(
                    source_action_id=source.action_id,
                    target_action_id=target.action_id,
                    edge_type=edge_type,
                    metadata={
                        "evidence": f"Cross-domain flow: {source_domain} -> {target_domain}",
                        "match_type": "cross_domain",
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "has_injection": has_injection,
                    }
                )
                edges.append(edge)

        return edges

    def _infer_sequential_domain(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Link sequential actions in the same domain.

        Pattern: Consecutive tool calls in the same domain (e.g., list_files -> get_file_by_id)
        often have implicit state/data dependencies.
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        for i in range(1, len(actions)):
            prev = actions[i - 1]
            curr = actions[i]

            prev_tool = self._get_tool_name(prev)
            curr_tool = self._get_tool_name(curr)

            prev_domain = TOOL_DOMAINS.get(prev_tool, "unknown")
            curr_domain = TOOL_DOMAINS.get(curr_tool, "unknown")

            # Same domain and not both unknown
            if prev_domain == curr_domain and prev_domain != "unknown":
                # Skip agent_response actions
                if curr.action_type == ActionType.AGENT_RESPONSE:
                    continue
                if prev.action_type == ActionType.AGENT_RESPONSE:
                    continue

                edge = CausalEdge(
                    source_action_id=prev.action_id,
                    target_action_id=curr.action_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={
                        "evidence": f"Sequential {prev_domain} actions",
                        "match_type": "sequential_domain",
                        "domain": prev_domain,
                    }
                )
                edges.append(edge)

        return edges

    def _infer_all_sequential(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Link all consecutive tool calls (not just same-domain).

        Ensures we capture
        the causal chain even when specific content matching fails.
        In AgentDojo, consecutive tool calls almost always have data dependencies.
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        for i in range(1, len(actions)):
            prev = actions[i - 1]
            curr = actions[i]

            # Skip agent_response actions
            if curr.action_type == ActionType.AGENT_RESPONSE:
                continue
            if prev.action_type == ActionType.AGENT_RESPONSE:
                continue

            prev_tool = self._get_tool_name(prev)
            curr_tool = self._get_tool_name(curr)

            # Skip if neither has a tool name
            if not prev_tool and not curr_tool:
                continue

            # Determine edge type
            # If source has injection content, it's trust transfer
            has_injection = self._has_injection_content(prev)
            edge_type = EdgeType.TRUST_TRANSFER if has_injection else EdgeType.DATA_DEPENDENCY

            edge = CausalEdge(
                source_action_id=prev.action_id,
                target_action_id=curr.action_id,
                edge_type=edge_type,
                metadata={
                    "evidence": f"Sequential tool calls: {prev_tool} -> {curr_tool}",
                    "match_type": "all_sequential",
                    "has_injection": has_injection,
                }
            )
            edges.append(edge)

        return edges

    def _infer_state_enablement(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Detect state enablement patterns.

        Some tools provide state that enables other tools to function:
        - get_current_day -> create_calendar_event (need date)
        - get_iban -> send_money (need account)
        - search_files_by_filename -> delete_file (need file ID)
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        for i, source in enumerate(actions):
            source_tool = self._get_tool_name(source)
            enabled_tools = self.STATE_ENABLERS.get(source_tool, set())

            if not enabled_tools:
                continue

            # Look for subsequent actions that are enabled by this tool
            for target in actions[i + 1:]:
                target_tool = self._get_tool_name(target)

                if target_tool in enabled_tools:
                    edge = CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={
                            "evidence": f"State enablement: {source_tool} enables {target_tool}",
                            "match_type": "state_enablement",
                            "enabler_tool": source_tool,
                            "enabled_tool": target_tool,
                        }
                    )
                    edges.append(edge)

        return edges

    def _get_tool_name(self, action: Action) -> str:
        """Extract tool/function name from action."""
        if action.context:
            fn_name = action.context.get("function_name")
            if fn_name:
                return fn_name

            # Check raw_data
            if action.raw_data:
                fn_name = action.raw_data.get("function")
                if fn_name:
                    return fn_name

        # Fallback to target
        if action.target:
            # Clean up target like "file:team meeting minutes"
            target = str(action.target)
            if ":" in target:
                return target.split(":")[0]
            return target

        return ""

    def _extract_data(self, text: str) -> ExtractedData:
        """Extract structured data from tool result text."""
        emails = set(EMAIL_PATTERN.findall(text))
        ibans = set(IBAN_PATTERN.findall(text))
        file_ids = set(FILE_ID_PATTERN.findall(text))
        dates = set(DATE_PATTERN.findall(text))

        # Extract meaningful text snippets (names, subjects, etc.)
        text_snippets = set()

        # Extract names after "- David:", "- Linda:", etc.
        name_matches = re.findall(r"[-*]\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*:", text)
        text_snippets.update(name_matches)

        # Extract subjects
        subject_matches = re.findall(r"subject:\s*['\"]?([^'\"\n]+)['\"]?", text, re.IGNORECASE)
        text_snippets.update(subject_matches)

        # Extract filenames
        filename_matches = re.findall(r"filename:\s*['\"]?([^'\"\n]+)['\"]?", text, re.IGNORECASE)
        text_snippets.update(filename_matches)

        # Check for injection content
        has_injection = any(ind.lower() in text.lower() for ind in self.INJECTION_INDICATORS)

        return ExtractedData(
            emails=emails,
            ibans=ibans,
            file_ids=file_ids,
            dates=dates,
            text_snippets=text_snippets,
            has_injection=has_injection,
        )

    def _get_action_input_text(self, action: Action) -> str:
        """Get combined input text from action."""
        parts = []

        if action.target:
            parts.append(str(action.target))

        if action.context:
            args = action.context.get("args", {})
            if isinstance(args, dict):
                for key, value in args.items():
                    if value is not None:
                        parts.append(str(value))

        return " ".join(parts)

    def _has_data_flow(self, source: Action, target: Action) -> bool:
        """Check if there's actual data flow from source to target."""
        if not source.result:
            return False

        source_text = str(source.result).lower()
        target_text = self._get_action_input_text(target).lower()

        if not target_text or len(target_text) < self.min_content_length:
            return False

        # Extract key data items from source
        source_data = self._extract_data(source_text)

        # Check if any extracted data appears in target
        for email in source_data.emails:
            if email.lower() in target_text:
                return True

        for iban in source_data.ibans:
            if iban.lower() in target_text:
                return True

        for file_id in source_data.file_ids:
            if file_id in target_text:
                return True

        for snippet in source_data.text_snippets:
            if len(snippet) >= 4 and snippet.lower() in target_text:
                return True

        # Check for substantial substring overlap
        if len(target_text) >= 10:
            # Look for any 10+ char substring from source in target
            for i in range(0, len(source_text) - 10):
                chunk = source_text[i:i+10]
                if chunk.isalnum() and chunk in target_text:
                    return True

        return False

    def _has_injection_content(self, action: Action) -> bool:
        """Check if action result contains injection indicators."""
        if not action.result:
            return False

        result_text = str(action.result).lower()
        return any(ind.lower() in result_text for ind in self.INJECTION_INDICATORS)
