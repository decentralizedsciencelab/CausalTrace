"""
Enhanced edge inference for sparse trajectory data.

This module provides advanced edge detection for trajectories that lack
explicit data_produced/data_consumed annotations, such as Mind2Web traces.

Key inference rules:
1. Read → Type: Content from read appears in subsequent type
2. Navigate → Actions: Navigation enables subsequent same-domain actions
3. Form chains: Type fields → Submit
4. Auth chains: Login → Authenticated actions
5. Cross-domain flows: Content extraction → External send (attack indicator)
"""

import re
from typing import List, Optional, Set, Tuple
from difflib import SequenceMatcher

from causaltrace.models.trajectory import Trajectory, Action, ActionType
from causaltrace.graph.causal_graph import CausalEdge, EdgeType


class EnhancedEdgeInference:
    """
    Enhanced edge inference engine for sparse trajectory data.

    Uses heuristics and content analysis to infer causal edges
    when explicit data flow annotations are missing.
    """

    # Action types that produce data
    DATA_PRODUCERS = {
        ActionType.READ, ActionType.NAVIGATE, ActionType.CLICK,
        ActionType.SCROLL, ActionType.SELECT
    }

    # Action types that consume data
    DATA_CONSUMERS = {
        ActionType.TYPE, ActionType.SUBMIT, ActionType.CLICK,
        ActionType.TOOL_CALL, ActionType.SEND_EMAIL
    }

    # Auth-related action types
    AUTH_ACTIONS = {
        ActionType.LOGIN, ActionType.CLICK  # Click can be login button
    }

    # Actions requiring auth state
    AUTH_REQUIRED = {
        ActionType.SUBMIT, ActionType.TYPE, ActionType.SEND_EMAIL,
        ActionType.TOOL_CALL, ActionType.NAVIGATE
    }

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        max_lookback: int = 10,
        enable_state_enablement: bool = True,
        enable_data_dependency: bool = True,
        enable_trust_transfer: bool = True,
    ):
        """
        Initialize enhanced edge inference.

        Args:
            similarity_threshold: Min similarity for content matching (0-1)
            max_lookback: Max actions to look back for dependencies
            enable_state_enablement: Infer state enablement edges
            enable_data_dependency: Infer data dependency edges
            enable_trust_transfer: Infer trust transfer edges
        """
        self.similarity_threshold = similarity_threshold
        self.max_lookback = max_lookback
        self.enable_state_enablement = enable_state_enablement
        self.enable_data_dependency = enable_data_dependency
        self.enable_trust_transfer = enable_trust_transfer

    def infer_edges(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer all causal edges for a trajectory.

        Args:
            trajectory: The trajectory to analyze

        Returns:
            List of inferred CausalEdge objects
        """
        edges = []
        actions = trajectory.actions
        task_desc = trajectory.task_description or ""

        for j, target_action in enumerate(actions):
            # Look back at previous actions
            start_i = max(0, j - self.max_lookback)

            for i in range(start_i, j):
                source_action = actions[i]

                # Try each edge type
                if self.enable_data_dependency:
                    edge = self._check_data_dependency(
                        source_action, target_action, i, j, task_desc, actions
                    )
                    if edge:
                        edges.append(edge)
                        continue  # Only one edge per pair

                if self.enable_trust_transfer:
                    edge = self._check_trust_transfer(source_action, target_action, i, j)
                    if edge:
                        edges.append(edge)
                        continue

                if self.enable_state_enablement:
                    edge = self._check_state_enablement(source_action, target_action, i, j)
                    if edge:
                        edges.append(edge)

        return edges

    def _check_data_dependency(
        self, source: Action, target: Action, src_id: int, tgt_id: int,
        task_desc: str = "", all_actions: List[Action] = None
    ) -> Optional[CausalEdge]:
        """Check if target uses data from source."""

        source_value = self._get_action_value(source)
        target_value = self._get_action_value(target)

        # Rule 1: Task-derived data flow
        # If source TYPE/SELECT has value that appears in task, and target is subsequent
        # action that "uses" that data (click confirmation, another form field, etc.)
        if source.action_type in {ActionType.TYPE, ActionType.CLICK, ActionType.SELECT}:
            if source_value and task_desc:
                # Check if source value is derived from task description
                if self._value_in_task(source_value, task_desc):
                    # This action consumed task-derived data
                    # If target is immediately following click (confirmation) - data dependency
                    if tgt_id == src_id + 1 and target.action_type == ActionType.CLICK:
                        return CausalEdge(
                            source_action_id=src_id,
                            target_action_id=tgt_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            metadata={"confidence": 0.75, "evidence": "form_field_to_confirm"}
                        )

        # Rule 2: TYPE/SELECT → subsequent CLICK (form submission pattern)
        if source.action_type in {ActionType.TYPE, ActionType.SELECT}:
            if target.action_type == ActionType.CLICK:
                if self._same_domain(source, target):
                    # Close proximity suggests form submission
                    if tgt_id - src_id <= 3:  # Within 3 actions
                        return CausalEdge(
                            source_action_id=src_id,
                            target_action_id=tgt_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            metadata={"confidence": 0.65, "evidence": "form_input_to_click"}
                        )

        # Rule 3: CLICK (with value/selection) → TYPE (using selected context)
        if source.action_type == ActionType.CLICK and source_value:
            if target.action_type == ActionType.TYPE:
                if self._same_domain(source, target):
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={"confidence": 0.7, "evidence": "click_selection_to_type"}
                    )

        # Rule 4: Read → Type (content appears in type value)
        if source.action_type in {ActionType.READ, ActionType.NAVIGATE}:
            if target.action_type == ActionType.TYPE:
                source_content = self._get_action_content(source)
                target_input = self._get_action_input(target)

                if self._content_similarity(source_content, target_input) > self.similarity_threshold:
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={"confidence": 0.8, "evidence": "content_match"}
                    )

        # Rule 5: Explicit data_produced → data_consumed
        if source.data_produced and target.data_consumed:
            overlap = set(source.data_produced) & set(target.data_consumed)
            if overlap:
                return CausalEdge(
                    source_action_id=src_id,
                    target_action_id=tgt_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={"confidence": 1.0, "evidence": "explicit_data"}
                )

        # Rule 6: Search → Select (search results → click result)
        if source.action_type == ActionType.TYPE:
            if target.action_type == ActionType.CLICK:
                source_target = (source.target or "").lower()
                if "search" in source_target or source_value:
                    # Type followed by click on same domain = search then select
                    if self._same_domain(source, target) and tgt_id - src_id <= 2:
                        return CausalEdge(
                            source_action_id=src_id,
                            target_action_id=tgt_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            metadata={"confidence": 0.6, "evidence": "search_select_chain"}
                        )

        # Rule 7: Sequential TYPE actions (form field chain)
        if source.action_type == ActionType.TYPE and target.action_type == ActionType.TYPE:
            if self._same_domain(source, target) and tgt_id == src_id + 1:
                return CausalEdge(
                    source_action_id=src_id,
                    target_action_id=tgt_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={"confidence": 0.55, "evidence": "form_field_chain"}
                )

        return None

    def _get_action_value(self, action: Action) -> str:
        """Extract the primary value from an action (what was typed/selected)."""
        if action.context and isinstance(action.context, dict):
            value = action.context.get("value", "")
            if value:
                return str(value).strip()
        return ""

    def _value_in_task(self, value: str, task_desc: str) -> bool:
        """Check if action value appears to be derived from task description."""
        if not value or not task_desc:
            return False

        value_lower = value.lower().strip()
        task_lower = task_desc.lower()

        # Direct substring match
        if value_lower in task_lower:
            return True

        # Check individual words (for multi-word values)
        value_words = set(value_lower.split())
        task_words = set(task_lower.split())

        # If value is a single meaningful word found in task
        if len(value_words) == 1 and len(value_lower) >= 3:
            if value_lower in task_words:
                return True

        # If most value words appear in task
        if len(value_words) > 1:
            overlap = value_words & task_words
            if len(overlap) >= len(value_words) * 0.5:
                return True

        return False

    def _check_trust_transfer(
        self, source: Action, target: Action, src_id: int, tgt_id: int
    ) -> Optional[CausalEdge]:
        """Check if target executes code/instructions from source."""

        # Rule 1: Read external content → Execute/Tool call
        if source.action_type in {ActionType.READ, ActionType.NAVIGATE}:
            if target.action_type in {ActionType.TOOL_CALL, ActionType.CLICK}:
                source_content = self._get_action_content(source)
                target_input = self._get_action_input(target)

                # Check for instruction-like patterns
                if self._contains_instructions(source_content):
                    if self._content_similarity(source_content, target_input) > 0.2:
                        return CausalEdge(
                            source_action_id=src_id,
                            target_action_id=tgt_id,
                            edge_type=EdgeType.TRUST_TRANSFER,
                            metadata={"confidence": 0.7, "evidence": "instruction_execution"}
                        )

        # Rule 2: Cross-domain content → External action (attack indicator)
        if not self._same_domain(source, target):
            source_content = self._get_action_content(source)
            target_input = self._get_action_input(target)

            if source_content and target_input:
                if self._content_similarity(source_content, target_input) > self.similarity_threshold:
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.TRUST_TRANSFER,
                        metadata={"confidence": 0.9, "evidence": "cross_domain_data_transfer"}
                    )

        return None

    def _check_state_enablement(
        self, source: Action, target: Action, src_id: int, tgt_id: int
    ) -> Optional[CausalEdge]:
        """Check if source enables state required by target.

        Note: Only returns state_enablement for true state-changing patterns,
        not for sequential form actions (which should be data_dependency).
        """

        # Rule 1: Navigate → Same-domain actions (page load enables interaction)
        if source.action_type == ActionType.NAVIGATE:
            if target.action_type in self.AUTH_REQUIRED:
                if self._same_domain(source, target):
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={"confidence": 0.6, "evidence": "navigation_enables_action"}
                    )

        # Rule 2: Login → Authenticated actions
        if source.action_type == ActionType.LOGIN:
            if target.action_type in self.AUTH_REQUIRED:
                if self._same_domain(source, target):
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={"confidence": 0.9, "evidence": "auth_enables_action"}
                    )

        # Rule 3: Click (login/auth button pattern) → Subsequent actions
        if source.action_type == ActionType.CLICK:
            source_value = self._get_action_value(source)
            source_target_text = (source.target or "").lower() + " " + source_value.lower()
            auth_keywords = ["login", "sign in", "log in", "signin", "authenticate", "account"]
            if any(kw in source_target_text for kw in auth_keywords):
                if self._same_domain(source, target):
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={"confidence": 0.7, "evidence": "login_click_enables"}
                    )

        # Rule 4: CLICK that navigates/opens a section → subsequent actions in that section
        # Only for non-adjacent actions (adjacent should be data_dependency)
        if source.action_type == ActionType.CLICK and tgt_id > src_id + 1:
            if self._same_domain(source, target):
                # Only if this looks like section navigation, not form interaction
                source_value = self._get_action_value(source)
                nav_keywords = ["menu", "tab", "section", "page", "view", "show", "open", "expand"]
                if any(kw in source_value.lower() for kw in nav_keywords):
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={"confidence": 0.5, "evidence": "navigation_click_enables"}
                    )

        # Rule 5: Sequential CLICK actions (UI navigation chain)
        # Only when both are clicks with no data values (pure navigation)
        if source.action_type == ActionType.CLICK and target.action_type == ActionType.CLICK:
            source_value = self._get_action_value(source)
            target_value = self._get_action_value(target)

            # If neither has data values, this is pure UI navigation
            if not source_value and not target_value:
                if self._same_domain(source, target) and tgt_id == src_id + 1:
                    return CausalEdge(
                        source_action_id=src_id,
                        target_action_id=tgt_id,
                        edge_type=EdgeType.STATE_ENABLEMENT,
                        metadata={"confidence": 0.5, "evidence": "ui_navigation_chain"}
                    )

        # Rule 6: CLICK with selection value → next action (UI state change)
        if source.action_type == ActionType.CLICK:
            source_value = self._get_action_value(source)
            if source_value and self._same_domain(source, target) and tgt_id == src_id + 1:
                return CausalEdge(
                    source_action_id=src_id,
                    target_action_id=tgt_id,
                    edge_type=EdgeType.STATE_ENABLEMENT,
                    metadata={"confidence": 0.55, "evidence": "selection_enables_next"}
                )

        return None

    def _get_action_content(self, action: Action) -> str:
        """Extract content/output from an action."""
        parts = []

        if action.result:
            parts.append(str(action.result))

        if action.context:
            if isinstance(action.context, dict):
                for key in ["content", "text", "value", "result", "body", "observation"]:
                    if key in action.context:
                        parts.append(str(action.context[key]))
            else:
                parts.append(str(action.context))

        if action.raw_data:
            if isinstance(action.raw_data, dict):
                for key in ["content", "text", "value", "observation"]:
                    if key in action.raw_data:
                        parts.append(str(action.raw_data[key]))

        return " ".join(parts)

    def _get_action_input(self, action: Action) -> str:
        """Extract input/consumed data from an action."""
        parts = []

        if action.target:
            parts.append(str(action.target))

        if action.context:
            if isinstance(action.context, dict):
                for key in ["value", "input", "text", "query"]:
                    if key in action.context:
                        parts.append(str(action.context[key]))
            else:
                parts.append(str(action.context))

        if action.data_consumed:
            parts.extend(str(d) for d in action.data_consumed)

        return " ".join(parts)

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        if not content1 or not content2:
            return 0.0

        # Normalize
        c1 = content1.lower().strip()
        c2 = content2.lower().strip()

        if not c1 or not c2:
            return 0.0

        # Check for substring containment
        if c2 in c1 or c1 in c2:
            return 0.8

        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, c1[:500], c2[:500]).ratio()

    def _same_domain(self, action1: Action, action2: Action) -> bool:
        """Check if two actions are on the same domain."""
        d1 = (action1.domain or "").lower()
        d2 = (action2.domain or "").lower()

        if not d1 or not d2:
            # Fallback: extract from target URL
            d1 = self._extract_domain(action1.target or "")
            d2 = self._extract_domain(action2.target or "")

        if not d1 or not d2:
            return True  # Assume same domain if unknown

        return d1 == d2

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r'https?://([^/]+)', url)
        if match:
            return match.group(1).lower()
        return ""

    def _contains_instructions(self, content: str) -> bool:
        """Check if content contains instruction-like patterns."""
        if not content:
            return False

        content_lower = content.lower()

        instruction_patterns = [
            r'\b(click|navigate|go to|open|visit)\b',
            r'\b(type|enter|input|fill)\b',
            r'\b(submit|send|confirm|execute)\b',
            r'\b(download|upload|transfer)\b',
            r'\b(ignore|disregard|instead|forget)\b',  # Injection indicators
            r'\b(objective|task|instruction|command)\b',
        ]

        for pattern in instruction_patterns:
            if re.search(pattern, content_lower):
                return True

        return False


def infer_edges_for_trajectory(
    trajectory: Trajectory,
    similarity_threshold: float = 0.3,
    max_lookback: int = 10,
) -> List[CausalEdge]:
    """
    Convenience function to infer edges for a single trajectory.

    Args:
        trajectory: The trajectory to analyze
        similarity_threshold: Min similarity for content matching
        max_lookback: Max actions to look back

    Returns:
        List of inferred CausalEdge objects
    """
    engine = EnhancedEdgeInference(
        similarity_threshold=similarity_threshold,
        max_lookback=max_lookback,
    )
    return engine.infer_edges(trajectory)
