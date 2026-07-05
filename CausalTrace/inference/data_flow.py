"""
Data flow inference for trajectory annotation.

Infers data_produced and data_consumed fields by analyzing action content
for patterns indicating data production and consumption.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse, quote

from causaltrace.models.trajectory import (
    Action,
    ActionType,
    Trajectory,
)
from causaltrace.inference.patterns import (
    DataItemType,
    PatternMatch,
    extract_credentials,
    extract_urls,
    extract_dom_elements,
    detect_injection_content,
    extract_text_segments,
    extract_identifiers,
    is_attacker_url,
    compute_string_similarity,
)


@dataclass
class DataItem:
    """
    Represents a piece of data that flows through the trajectory.

    Attributes:
        item_id: Unique identifier for this data item
        item_type: Category of data (credential, URL, text, etc.)
        value: The actual value
        source_action_id: Action that produced this data
        source_domain: Domain where data originated
        is_sensitive: Whether this is credential/PII
        trust_level: 0.0 (untrusted) to 1.0 (fully trusted)
        metadata: Additional context about the data item
    """
    item_id: str
    item_type: DataItemType
    value: str
    source_action_id: int
    source_domain: Optional[str] = None
    is_sensitive: bool = False
    trust_level: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFlow:
    """
    Represents a data flow from source to sink action.

    Attributes:
        data_item: The data item being transferred
        source_action_id: Producer action
        sink_action_id: Consumer action
        flow_type: Type of causal relationship
        confidence: Detection confidence (0.0 to 1.0)
        evidence: Explanation of how flow was detected
    """
    data_item: DataItem
    source_action_id: int
    sink_action_id: int
    flow_type: str  # "data_dependency", "trust_transfer", "state_enablement"
    confidence: float
    evidence: str


class DataFlowInferencer:
    """
    Infers data_produced and data_consumed for actions in a trajectory.

    This class analyzes raw trajectory data to identify:
    1. What data each action produces (credentials, URLs, text, etc.)
    2. What data each action consumes from prior actions
    3. Trust transfer when untrusted content influences actions

    Works with existing extractor outputs - does not require re-running benchmarks.
    """

    # Trust levels for different domain types
    UNTRUSTED_DOMAIN_PATTERNS = ["forum", "social", "public", "user", "cdn"]
    TRUSTED_DOMAIN_PATTERNS = ["internal", "corp", "admin", "bank", "gitlab"]

    # AgentDojo-style key-value pairs we care about (id_, file references, IBANs, etc.)
    KEY_VALUE_FIELDS = (
        "id_", "file_id", "event_id", "iban", "recipient_iban",
        "message_id", "calendar_id", "reservation_id"
    )
    KEY_VALUE_PATTERN = re.compile(
        r"(?P<key>" + "|".join(KEY_VALUE_FIELDS) + r")\s*[:=]\s*['\"]?(?P<value>[A-Za-z0-9._-]+)['\"]?",
        re.IGNORECASE
    )

    def __init__(
        self,
        enable_credential_detection: bool = True,
        enable_url_tracking: bool = True,
        enable_dom_tracking: bool = True,
        enable_text_tracking: bool = True,
        enable_injection_detection: bool = True,
        trust_threshold: float = 0.5,
        min_match_length: int = 6,
    ):
        """
        Initialize the data flow inferencer.

        Args:
            enable_credential_detection: Detect API keys, tokens, passwords
            enable_url_tracking: Track URLs and domains across actions
            enable_dom_tracking: Track DOM element references
            enable_text_tracking: Track extracted text segments
            enable_injection_detection: Detect prompt injection content
            trust_threshold: Below this, mark data as untrusted
            min_match_length: Minimum string length for matching
        """
        self.enable_credential_detection = enable_credential_detection
        self.enable_url_tracking = enable_url_tracking
        self.enable_dom_tracking = enable_dom_tracking
        self.enable_text_tracking = enable_text_tracking
        self.enable_injection_detection = enable_injection_detection
        self.trust_threshold = trust_threshold
        self.min_match_length = min_match_length

        # Data store: item_id -> DataItem
        self._data_items: Dict[str, DataItem] = {}

        # Producer index: value_hash -> list of (action_id, item_id)
        self._producer_index: Dict[str, List[Tuple[int, str]]] = {}

        # Detected flows
        self._flows: List[DataFlow] = []

    def infer_data_flow(self, trajectory: Trajectory) -> Trajectory:
        """
        Annotate trajectory actions with data_produced and data_consumed.

        This is the main entry point. It:
        1. Indexes any pre-existing annotations (semantic IDs from benchmarks)
        2. Scans each action to extract data it produces
        3. Matches consumed data against previously produced data
        4. Updates action.data_produced and action.data_consumed lists
        5. Detects flows from both pre-existing and newly inferred annotations

        Note: Augments existing annotations rather than replacing them.

        Args:
            trajectory: Input trajectory (will be modified in-place)

        Returns:
            Same trajectory with data annotations filled in
        """
        # Reset state for new trajectory
        self._data_items.clear()
        self._producer_index.clear()
        self._flows.clear()

        # Phase 0: Index pre-existing annotations (from benchmarks like pilot data)
        # This creates DataItem placeholders for semantic IDs so we can track flows
        self._index_preexisting_annotations(trajectory)

        # Phase 1: Extract data produced by each action
        for action in trajectory.actions:
            produced_items = self._extract_data_produced(action, trajectory)
            new_ids = [item.item_id for item in produced_items]

            # AUGMENT existing annotations instead of replacing
            existing_produced = action.data_produced or []
            action.data_produced = list(set(existing_produced + new_ids))

            # Index produced items for consumption lookup
            for item in produced_items:
                self._data_items[item.item_id] = item
                self._index_data_item(item)

        # Phase 2: Detect data consumed by each action (for newly inferred items)
        for action in trajectory.actions:
            consumed_ids = self._detect_data_consumed(action, trajectory)

            # AUGMENT existing annotations instead of replacing
            existing_consumed = action.data_consumed or []
            action.data_consumed = list(set(existing_consumed + consumed_ids))

            # Record flows for newly detected consumption
            for item_id in consumed_ids:
                item = self._data_items.get(item_id)
                if item:
                    flow_type = self._determine_flow_type(item, action)
                    self._flows.append(DataFlow(
                        data_item=item,
                        source_action_id=item.source_action_id,
                        sink_action_id=action.action_id,
                        flow_type=flow_type,
                        confidence=0.8,
                        evidence=f"{item.item_type.value} from action {item.source_action_id}",
                    ))

        # Phase 3: Detect flows from pre-existing annotations
        # This matches consumed IDs to produced IDs using exact semantic matching
        self._detect_preexisting_flows(trajectory)

        return trajectory

    def get_detected_flows(self) -> List[DataFlow]:
        """Return all detected data flows after inference."""
        return self._flows.copy()

    def get_trust_transfers(self) -> List[DataFlow]:
        """Return only trust transfer flows (untrusted -> sensitive)."""
        return [f for f in self._flows if f.flow_type == "trust_transfer"]

    # =========================================================================
    # Data Production Extraction
    # =========================================================================

    def _extract_data_produced(
        self,
        action: Action,
        trajectory: Trajectory
    ) -> List[DataItem]:
        """
        Extract all data items produced by an action.

        Scans action.result, action.context, and action.raw_data
        for patterns indicating data production.
        """
        produced = []

        # Gather all text content from the action
        text_sources = self._get_action_text_content(action)

        # Detect injection in any text source
        has_injection = False
        if self.enable_injection_detection:
            for text, _ in text_sources:
                is_injection, _ = detect_injection_content(text)
                if is_injection:
                    has_injection = True
                    break

        # Base trust level for this action
        base_trust = self._compute_trust_level(action)
        if has_injection:
            base_trust = min(base_trust, 0.1)  # Injection content is untrusted

        # Credential detection
        if self.enable_credential_detection:
            for text, source_field in text_sources:
                for match in extract_credentials(text):
                    item = DataItem(
                        item_id=f"cred_{action.action_id}_{match.pattern_name}_{len(produced)}",
                        item_type=DataItemType.CREDENTIAL,
                        value=match.value,
                        source_action_id=action.action_id,
                        source_domain=action.domain,
                        is_sensitive=True,
                        trust_level=base_trust,
                        metadata={
                            "credential_type": match.pattern_name,
                            "source_field": source_field,
                            "has_injection": has_injection,
                        },
                    )
                    produced.append(item)

        # URL extraction
        if self.enable_url_tracking:
            for text, source_field in text_sources:
                for match in extract_urls(text):
                    is_attacker = match.metadata.get("is_attacker", False)
                    item = DataItem(
                        item_id=f"url_{action.action_id}_{len(produced)}",
                        item_type=DataItemType.URL,
                        value=match.value,
                        source_action_id=action.action_id,
                        source_domain=action.domain,
                        is_sensitive=False,
                        trust_level=0.0 if is_attacker else base_trust,
                        metadata={
                            "is_attacker_url": is_attacker,
                            "source_field": source_field,
                        },
                    )
                    produced.append(item)

        # DOM element tracking
        if self.enable_dom_tracking:
            for text, source_field in text_sources:
                for match in extract_dom_elements(text):
                    item = DataItem(
                        item_id=f"dom_{action.action_id}_{match.value}",
                        item_type=DataItemType.DOM_ELEMENT,
                        value=match.value,
                        source_action_id=action.action_id,
                        source_domain=action.domain,
                        trust_level=base_trust,
                        metadata={"element_type": match.pattern_name},
                    )
                    produced.append(item)

        # Form field values (TYPE actions)
        # TYPE actions primarily consume data from earlier actions.
        # We track the typed value for matching consumption.
        if action.action_type == ActionType.TYPE:
            value = self._extract_type_value(action)
            if value and len(value) >= self.min_match_length:
                # Store as user input - this helps track what was typed
                # but data consumption detection will match against earlier producers
                item = DataItem(
                    item_id=f"input_{action.action_id}",
                    item_type=DataItemType.USER_INPUT,
                    value=value,
                    source_action_id=action.action_id,
                    source_domain=action.domain,
                    trust_level=base_trust,
                    metadata={"is_form_input": True},
                )
                produced.append(item)

        # Text segments from READ actions
        if self.enable_text_tracking and action.action_type in {ActionType.READ, ActionType.NAVIGATE}:
            if action.result:
                result_str = str(action.result)
                segments = extract_text_segments(result_str)
                source_field = "result"
                for i, segment in enumerate(segments[:5]):  # Limit segments
                    if len(segment) >= self.min_match_length:
                        item = DataItem(
                            item_id=f"text_{action.action_id}_{i}",
                            item_type=DataItemType.EXTRACTED_TEXT,
                            value=segment,
                            source_action_id=action.action_id,
                            source_domain=action.domain,
                            trust_level=base_trust,
                            metadata={"has_injection": has_injection},
                        )
                        produced.append(item)

                # Also extract identifiers (account numbers, IDs, etc.)
                identifiers = extract_identifiers(result_str)
                for i, identifier in enumerate(identifiers[:5]):
                    if len(identifier) >= self.min_match_length:
                        item = DataItem(
                            item_id=f"id_{action.action_id}_{i}",
                            item_type=DataItemType.EXTRACTED_TEXT,
                            value=identifier,
                            source_action_id=action.action_id,
                            source_domain=action.domain,
                            trust_level=base_trust,
                            metadata={"is_identifier": True},
                        )
                        produced.append(item)

                # AgentDojo key-value identifiers (IDs, IBANs, reservation codes)
                kv_index = 0
                for match in self.KEY_VALUE_PATTERN.finditer(result_str):
                    key = match.group("key").lower()
                    value = match.group("value").strip()
                    if not value:
                        continue
                    item = DataItem(
                        item_id=f"kv_{action.action_id}_{key}_{kv_index}",
                        item_type=DataItemType.EXTRACTED_TEXT,
                        value=f"{key}:{value}",
                        source_action_id=action.action_id,
                        source_domain=action.domain,
                        trust_level=base_trust,
                        metadata={
                            "kv_key": key,
                            "kv_value": value.lower(),
                            "allow_short_match": True,
                            "source_field": source_field,
                        },
                    )
                    produced.append(item)
                    kv_index += 1

        return produced

    # =========================================================================
    # Data Consumption Detection
    # =========================================================================

    def _detect_data_consumed(
        self,
        action: Action,
        trajectory: Trajectory
    ) -> List[str]:
        """
        Detect which previously-produced data items this action consumes.

        Uses multiple matching strategies:
        1. Exact value match
        2. Substring containment
        3. URL component matching
        4. Credential matching with encoding handling
        5. TYPE action value matching
        """
        consumed_ids: Set[str] = set()

        # Get action's input content
        input_texts = self._get_action_input_content(action)

        # For TYPE actions, also include the typed value for matching
        if action.action_type == ActionType.TYPE:
            typed_value = self._extract_type_value(action)
            if typed_value:
                input_texts.append(typed_value)

        if not input_texts:
            return list(consumed_ids)

        combined_input = " ".join(input_texts).lower()

        # Check each previously produced data item
        for item_id, item in self._data_items.items():
            # Only consider items from earlier actions
            if item.source_action_id >= action.action_id:
                continue

            allow_short = bool(item.metadata.get("allow_short_match"))
            if len(item.value) < self.min_match_length and not allow_short:
                continue

            value_lower = item.value.lower()

            # Special handling for AgentDojo-style identifiers (short numeric tokens)
            if allow_short:
                kv_value = item.metadata.get("kv_value")
                if kv_value and self._match_short_identifier(str(kv_value), combined_input):
                    consumed_ids.add(item_id)
                    continue

            # Strategy 1: Exact substring match
            if value_lower in combined_input:
                consumed_ids.add(item_id)
                continue

            # Strategy 1b: Reverse containment (input is subset of produced)
            # Useful for TYPE actions where user types partial value
            if len(combined_input) >= self.min_match_length and combined_input in value_lower:
                consumed_ids.add(item_id)
                continue

            # Strategy 2: URL component matching
            if item.item_type == DataItemType.URL:
                if self._url_component_match(item.value, combined_input):
                    consumed_ids.add(item_id)
                    continue

            # Strategy 3: Credential matching (handle encoding)
            if item.item_type == DataItemType.CREDENTIAL:
                if self._credential_match(item.value, combined_input):
                    consumed_ids.add(item_id)
                    continue

            # Strategy 4: Identifier matching for TYPE actions
            if action.action_type == ActionType.TYPE and item.metadata.get("is_identifier"):
                typed_value = self._extract_type_value(action)
                if typed_value and typed_value.upper() == item.value.upper():
                    consumed_ids.add(item_id)
                    continue

            # Strategy 5: Fuzzy matching for longer text
            if item.item_type == DataItemType.EXTRACTED_TEXT and len(item.value) > 20:
                similarity = compute_string_similarity(item.value, combined_input)
                if similarity > 0.7:
                    consumed_ids.add(item_id)
                    continue

        return list(consumed_ids)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_action_text_content(self, action: Action) -> List[Tuple[str, str]]:
        """
        Extract all text content from an action.

        Returns list of (text, source_field) tuples.
        """
        content = []

        if action.result:
            content.append((str(action.result), "result"))

        if action.target:
            content.append((str(action.target), "target"))

        if action.context:
            for key, value in action.context.items():
                if isinstance(value, str) and len(value) > 0:
                    content.append((value, f"context.{key}"))

        if hasattr(action, "raw_data") and action.raw_data:
            for key in ["observation", "accessibility_tree", "dom_content", "page_html"]:
                if key in action.raw_data:
                    value = action.raw_data[key]
                    if isinstance(value, str):
                        content.append((value, f"raw_data.{key}"))

        return content

    def _get_action_input_content(self, action: Action) -> List[str]:
        """
        Extract input/consumed content from an action.

        Focuses on fields that represent action inputs.
        """
        parts: List[str] = []

        if action.target:
            parts.append(str(action.target))

        if action.context:
            input_keys = ["value", "input", "text", "query", "data", "body", "content"]
            for key in input_keys:
                if key in action.context and action.context[key] is not None:
                    parts.extend(self._flatten_input_value(action.context[key]))

            # Agent/tool-call arguments often live under context["args"].
            args = action.context.get("args")
            if isinstance(args, dict):
                for arg_key, arg_value in args.items():
                    flattened = self._flatten_input_value(arg_value, prefix=f"{arg_key}:")
                    parts.extend(flattened)

        return [p for p in parts if isinstance(p, str) and p.strip()]

    def _flatten_input_value(self, value: Any, prefix: str = "") -> List[str]:
        """
        Normalize nested context/argument values into comparable strings.
        """
        results: List[str] = []

        if value is None:
            return results

        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                results.append(f"{prefix}{text}" if prefix else text)
            return results

        if isinstance(value, dict):
            for key, inner in value.items():
                new_prefix = f"{prefix}{key}:" if prefix else f"{key}:"
                results.extend(self._flatten_input_value(inner, prefix=new_prefix))
            return results

        if isinstance(value, list):
            for item in value:
                results.extend(self._flatten_input_value(item, prefix=prefix))
            return results

        # Fallback to generic string representation
        text = str(value).strip()
        if text:
            results.append(f"{prefix}{text}" if prefix else text)
        return results

    def _match_short_identifier(self, token: str, haystack: str) -> bool:
        """
        Match short numeric identifiers safely using word boundaries.
        """
        if not token:
            return False
        pattern = rf"(?<![A-Za-z0-9]){re.escape(token.lower())}(?![A-Za-z0-9])"
        return re.search(pattern, haystack) is not None

    def _extract_type_value(self, action: Action) -> Optional[str]:
        """Extract the typed value from a TYPE action."""
        if action.context:
            for key in ["value", "text", "input"]:
                if key in action.context:
                    return str(action.context[key])
        return None

    def _compute_trust_level(self, action: Action) -> float:
        """
        Compute trust level for data from this action.

        Lower trust for:
        - Forum/social media domains
        - User-generated content
        - External/unknown sources
        """
        domain = (action.domain or "").lower()

        # Check untrusted patterns
        for pattern in self.UNTRUSTED_DOMAIN_PATTERNS:
            if pattern in domain:
                return 0.2

        # Check trusted patterns
        for pattern in self.TRUSTED_DOMAIN_PATTERNS:
            if pattern in domain:
                return 0.9

        return 0.5  # Unknown = moderate trust

    def _determine_flow_type(self, item: DataItem, sink_action: Action) -> str:
        """
        Determine the type of data flow.

        Returns:
            "trust_transfer" if untrusted data flows to sensitive action
            "data_dependency" otherwise
        """
        # Trust transfer: untrusted data -> sensitive action
        if item.trust_level < self.trust_threshold:
            sensitive_types = {
                ActionType.TOOL_CALL,
                ActionType.SEND_EMAIL,
                ActionType.SUBMIT,
            }
            if sink_action.action_type in sensitive_types:
                return "trust_transfer"

            # Cross-domain navigation with untrusted URL
            if sink_action.action_type == ActionType.NAVIGATE and item.item_type == DataItemType.URL:
                return "trust_transfer"

        return "data_dependency"

    def _url_component_match(self, url: str, input_text: str) -> bool:
        """Check if URL or its components appear in input."""
        input_lower = input_text.lower()

        # Direct match
        if url.lower() in input_lower:
            return True

        # Extract and check components
        try:
            parsed = urlparse(url)
            if parsed.netloc and parsed.netloc.lower() in input_lower:
                return True
            if parsed.path and len(parsed.path) > 5 and parsed.path.lower() in input_lower:
                return True
            if parsed.query and len(parsed.query) > 5 and parsed.query.lower() in input_lower:
                return True
        except Exception:
            pass

        return False

    def _credential_match(self, cred_value: str, input_text: str) -> bool:
        """Check if credential value appears in input (handles URL encoding)."""
        input_lower = input_text.lower()
        cred_lower = cred_value.lower()

        # Direct match
        if cred_lower in input_lower:
            return True

        # URL-encoded match
        try:
            encoded = quote(cred_value, safe='').lower()
            if encoded in input_lower:
                return True
        except Exception:
            pass

        return False

    def _index_data_item(self, item: DataItem) -> None:
        """Index a data item for efficient lookup."""
        # Hash the value for indexing
        value_hash = hashlib.md5(item.value.lower().encode()).hexdigest()[:16]
        if value_hash not in self._producer_index:
            self._producer_index[value_hash] = []
        self._producer_index[value_hash].append((item.source_action_id, item.item_id))

    def _index_preexisting_annotations(self, trajectory: Trajectory) -> None:
        """
        Index pre-existing data_produced annotations from benchmark data.

        This handles semantic IDs from pilot data, WASP, SafeArena etc.
        which have IDs like 'page_content', 'malware_url', 'msg_session'.

        Creates placeholder DataItems so flows can be tracked.
        """
        # Build producer map: semantic_id -> (action_id, action)
        producer_map: Dict[str, Tuple[int, Action]] = {}

        for action in trajectory.actions:
            if action.data_produced:
                for item_id in action.data_produced:
                    # Skip items we've already created (inferencer-generated IDs)
                    if item_id in self._data_items:
                        continue

                    # Create placeholder DataItem for semantic IDs
                    item = DataItem(
                        item_id=item_id,
                        item_type=self._infer_item_type_from_id(item_id),
                        value=item_id,  # Use ID as value for matching
                        source_action_id=action.action_id,
                        source_domain=action.domain,
                        is_sensitive=self._is_sensitive_id(item_id),
                        trust_level=self._compute_trust_level(action),
                        metadata={"is_preexisting": True},
                    )
                    self._data_items[item_id] = item
                    producer_map[item_id] = (action.action_id, action)

    def _detect_preexisting_flows(self, trajectory: Trajectory) -> None:
        """
        Detect flows from pre-existing data_consumed annotations.

        Matches consumed IDs against produced IDs using exact semantic matching.
        This handles benchmark data like pilot where IDs are semantic (e.g., 'page_content').
        """
        # Build producer map for quick lookup
        producer_map: Dict[str, int] = {}
        for action in trajectory.actions:
            if action.data_produced:
                for item_id in action.data_produced:
                    if item_id not in producer_map:
                        producer_map[item_id] = action.action_id

        # Find flows from pre-existing consumed annotations
        for action in trajectory.actions:
            if action.data_consumed:
                for consumed_id in action.data_consumed:
                    # Check if this is a pre-existing semantic ID match
                    if consumed_id in producer_map:
                        source_action_id = producer_map[consumed_id]

                        # Avoid self-loops and backward edges
                        if source_action_id >= action.action_id:
                            continue

                        # Check if we already recorded this flow
                        already_recorded = any(
                            f.source_action_id == source_action_id
                            and f.sink_action_id == action.action_id
                            and (f.data_item.item_id == consumed_id or consumed_id in f.evidence)
                            for f in self._flows
                        )
                        if already_recorded:
                            continue

                        # Get or create the DataItem
                        item = self._data_items.get(consumed_id)
                        if not item:
                            # Create placeholder for this semantic ID
                            item = DataItem(
                                item_id=consumed_id,
                                item_type=self._infer_item_type_from_id(consumed_id),
                                value=consumed_id,
                                source_action_id=source_action_id,
                                source_domain=None,
                                is_sensitive=self._is_sensitive_id(consumed_id),
                                trust_level=0.5,
                                metadata={"is_preexisting": True},
                            )
                            self._data_items[consumed_id] = item

                        # Record the flow
                        flow_type = self._determine_flow_type(item, action)
                        self._flows.append(DataFlow(
                            data_item=item,
                            source_action_id=source_action_id,
                            sink_action_id=action.action_id,
                            flow_type=flow_type,
                            confidence=0.9,  # Higher confidence for explicit annotations
                            evidence=f"Pre-existing annotation: '{consumed_id}' from action {source_action_id}",
                        ))

    def _infer_item_type_from_id(self, item_id: str) -> DataItemType:
        """Infer DataItemType from semantic ID."""
        id_lower = item_id.lower()

        if any(kw in id_lower for kw in ["url", "link", "href"]):
            return DataItemType.URL
        elif any(kw in id_lower for kw in ["token", "key", "secret", "password", "credential", "auth"]):
            return DataItemType.CREDENTIAL
        elif any(kw in id_lower for kw in ["input", "form", "value", "text", "message"]):
            return DataItemType.USER_INPUT
        elif any(kw in id_lower for kw in ["element", "button", "field", "selector"]):
            return DataItemType.DOM_ELEMENT
        else:
            return DataItemType.EXTRACTED_TEXT

    def _is_sensitive_id(self, item_id: str) -> bool:
        """Check if semantic ID indicates sensitive data."""
        sensitive_keywords = [
            "token", "key", "secret", "password", "credential", "auth",
            "cookie", "session", "private", "api_key", "access_token",
        ]
        id_lower = item_id.lower()
        return any(kw in id_lower for kw in sensitive_keywords)


# =============================================================================
# Convenience Functions
# =============================================================================

def infer_data_flow(trajectory: Trajectory) -> Trajectory:
    """
    Convenience function to infer data flow for a trajectory.

    Args:
        trajectory: Input trajectory

    Returns:
        Trajectory with data_produced/data_consumed populated
    """
    inferencer = DataFlowInferencer()
    return inferencer.infer_data_flow(trajectory)
