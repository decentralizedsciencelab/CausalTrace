"""
Enhanced edge inference for sparse trajectories.

This module provides strategies for inferring causal edges when explicit
annotations (data_produced/data_consumed) are missing or sparse. Designed
specifically for pajaMAS-style multi-agent trajectories.

Inference Strategies:
1. Explicit edges from annotations (when available)
2. Sequential edges within same agent turn
3. Delegation edges when orchestrator → sub-agent
4. Content overlap detection for high-confidence links
"""

from typing import List, Optional, Set, Tuple, Dict, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

from causaltrace.models.trajectory import Action, ActionType, Trajectory
from causaltrace.graph.causal_graph import CausalEdge, EdgeType


@dataclass
class InferenceResult:
    """Result of edge inference containing detected edges and statistics."""
    edges: List[CausalEdge]
    stats: Dict[str, int]


class EdgeInferenceEngine:
    """
    Infer causal edges from sparse trajectory data.

    This engine uses multiple strategies to detect edges when explicit
    annotations are missing:

    1. Explicit: Use data_produced/data_consumed when available
    2. Sequential: Connect consecutive actions within same agent turn
    3. Delegation: Connect orchestrator actions to sub-agent invocations
    4. Content: Detect string similarity between result and input
    """

    # Agent roles for delegation detection
    ORCHESTRATOR_ROLES = {"orchestrator", "atlas", "coordinator", "manager", "planner"}
    WEB_AGENT_ROLES = {"web_browser", "web_surfer", "scout", "websurfer", "browser"}
    CODE_AGENT_ROLES = {"code_executor", "forge", "coder", "executor", "code"}

    # Action types that indicate orchestrator behavior
    DELEGATION_ACTION_TYPES = {ActionType.DELEGATION, ActionType.TOOL_CALL}

    # Minimum similarity threshold for content matching
    CONTENT_SIMILARITY_THRESHOLD = 0.6
    MIN_CONTENT_LENGTH = 10

    def __init__(
        self,
        use_explicit: bool = True,
        use_sequential: bool = True,
        use_delegation: bool = True,
        use_content: bool = True,
        similarity_threshold: float = 0.6,
        min_content_length: int = 10,
    ):
        """
        Initialize the edge inference engine.

        Args:
            use_explicit: Use data_produced/data_consumed annotations
            use_sequential: Connect consecutive actions in same agent turn
            use_delegation: Connect orchestrator to sub-agent calls
            use_content: Detect content similarity edges
            similarity_threshold: Min similarity for content matching (0-1)
            min_content_length: Min string length for content matching
        """
        self.use_explicit = use_explicit
        self.use_sequential = use_sequential
        self.use_delegation = use_delegation
        self.use_content = use_content
        self.similarity_threshold = similarity_threshold
        self.min_content_length = min_content_length

    def infer_edges(self, trajectory: Trajectory) -> InferenceResult:
        """
        Infer all edges from a trajectory.

        Args:
            trajectory: The trajectory to analyze

        Returns:
            InferenceResult with edges and statistics
        """
        all_edges: List[CausalEdge] = []
        stats = {
            "explicit": 0,
            "sequential": 0,
            "delegation": 0,
            "content": 0,
            "total": 0,
        }

        # Track seen edges to avoid duplicates
        seen_edges: Set[Tuple[int, int, str]] = set()

        def add_edge(edge: CausalEdge, strategy: str) -> bool:
            """Add edge if not duplicate, return True if added."""
            key = (edge.source_action_id, edge.target_action_id, edge.edge_type.value)
            if key not in seen_edges:
                seen_edges.add(key)
                all_edges.append(edge)
                stats[strategy] += 1
                stats["total"] += 1
                return True
            return False

        # Strategy 1: Explicit edges from annotations
        if self.use_explicit:
            for edge in self._infer_explicit(trajectory):
                add_edge(edge, "explicit")

        # Strategy 2: Sequential edges within agent turns
        if self.use_sequential:
            for edge in self._infer_sequential(trajectory):
                add_edge(edge, "sequential")

        # Strategy 3: Delegation edges
        if self.use_delegation:
            for edge in self._infer_delegation(trajectory):
                add_edge(edge, "delegation")

        # Strategy 4: Content overlap edges
        if self.use_content:
            for edge in self._infer_content(trajectory):
                add_edge(edge, "content")

        return InferenceResult(edges=all_edges, stats=stats)

    def _infer_explicit(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges from explicit data_produced/data_consumed annotations.

        Strategy: If action i produces data D and action j consumes D, create edge i→j.
        """
        edges: List[CausalEdge] = []

        # Build index: data key -> producing action
        data_producers: Dict[str, List[Action]] = {}
        for action in trajectory.actions:
            for data_key in action.data_produced:
                if data_key not in data_producers:
                    data_producers[data_key] = []
                data_producers[data_key].append(action)

        # Find consumers
        for action in trajectory.actions:
            for data_key in action.data_consumed:
                producers = data_producers.get(data_key, [])
                for producer in producers:
                    # Only create edge if producer comes before consumer
                    if producer.action_id < action.action_id:
                        edge = CausalEdge(
                            source_action_id=producer.action_id,
                            target_action_id=action.action_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            metadata={
                                "evidence": f"Explicit data flow: {data_key}",
                                "match_type": "explicit_annotation",
                                "data_key": data_key,
                            }
                        )
                        edges.append(edge)

        return edges

    def _infer_sequential(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges between consecutive actions within same agent turn.

        Strategy: Actions by the same agent in sequence are likely causally related.
        """
        edges: List[CausalEdge] = []

        if len(trajectory.actions) < 2:
            return edges

        for i in range(len(trajectory.actions) - 1):
            current = trajectory.actions[i]
            next_action = trajectory.actions[i + 1]

            # Get agent names from context
            current_agent = self._get_agent_name(current)
            next_agent = self._get_agent_name(next_action)

            # If same agent or agents are related (e.g., one delegates to another)
            if current_agent and next_agent and current_agent == next_agent:
                edge = CausalEdge(
                    source_action_id=current.action_id,
                    target_action_id=next_action.action_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={
                        "evidence": f"Sequential actions by {current_agent}",
                        "match_type": "sequential_agent_turn",
                        "agent": current_agent,
                    }
                )
                edges.append(edge)
            elif not current_agent and not next_agent:
                # No agent info - use temporal proximity + action type compatibility
                if self._are_compatible_sequence(current, next_action):
                    edge = CausalEdge(
                        source_action_id=current.action_id,
                        target_action_id=next_action.action_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={
                            "evidence": "Sequential actions (type compatible)",
                            "match_type": "sequential_type_compatible",
                        }
                    )
                    edges.append(edge)

        return edges

    def _infer_delegation(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges when orchestrator delegates to sub-agents.

        Strategy: DELEGATION actions or actions by orchestrator that reference
        other agents create trust transfer edges.
        """
        edges: List[CausalEdge] = []

        # Find orchestrator actions
        orchestrator_actions: List[Action] = []
        sub_agent_actions: Dict[str, List[Action]] = {}

        for action in trajectory.actions:
            agent_name = self._get_agent_name(action)
            agent_role = self._get_agent_role(action)

            if agent_role in self.ORCHESTRATOR_ROLES or self._is_orchestrator_action(action):
                orchestrator_actions.append(action)
            elif agent_name:
                if agent_name not in sub_agent_actions:
                    sub_agent_actions[agent_name] = []
                sub_agent_actions[agent_name].append(action)

        # Connect orchestrator delegations to first sub-agent action
        for orch_action in orchestrator_actions:
            # Check if this action references a sub-agent
            delegated_agent = self._extract_delegated_agent(orch_action)

            if delegated_agent:
                # Find subsequent actions by the delegated agent
                for agent_name, actions in sub_agent_actions.items():
                    if agent_name.lower() == delegated_agent.lower():
                        # Find first action after the delegation
                        for sub_action in actions:
                            if sub_action.action_id > orch_action.action_id:
                                edge = CausalEdge(
                                    source_action_id=orch_action.action_id,
                                    target_action_id=sub_action.action_id,
                                    edge_type=EdgeType.TRUST_TRANSFER,
                                    metadata={
                                        "evidence": f"Orchestrator delegates to {agent_name}",
                                        "match_type": "delegation",
                                        "delegated_agent": agent_name,
                                    }
                                )
                                edges.append(edge)
                                break  # Only connect to first action

            # If DELEGATION action type, connect to next action by any sub-agent
            if orch_action.action_type == ActionType.DELEGATION:
                for action in trajectory.actions:
                    if action.action_id > orch_action.action_id:
                        # Connect to first action that's not by orchestrator
                        action_role = self._get_agent_role(action)
                        if action_role not in self.ORCHESTRATOR_ROLES:
                            edge = CausalEdge(
                                source_action_id=orch_action.action_id,
                                target_action_id=action.action_id,
                                edge_type=EdgeType.TRUST_TRANSFER,
                                metadata={
                                    "evidence": "DELEGATION triggers sub-agent action",
                                    "match_type": "delegation_type",
                                }
                            )
                            edges.append(edge)
                            break

        return edges

    def _infer_content(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges based on content overlap/similarity.

        Strategy: If action i's result appears in action j's input/context,
        there's a data dependency.
        """
        edges: List[CausalEdge] = []

        for i, source in enumerate(trajectory.actions):
            source_content = self._extract_content(source, "output")
            if not source_content or len(source_content) < self.min_content_length:
                continue

            for target in trajectory.actions[i + 1:]:
                target_content = self._extract_content(target, "input")
                if not target_content or len(target_content) < self.min_content_length:
                    continue

                # Check for exact substring match first (faster)
                if self._has_significant_overlap(source_content, target_content):
                    edge = CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={
                            "evidence": "Content overlap detected",
                            "match_type": "content_substring",
                        }
                    )
                    edges.append(edge)
                    continue

                # Check similarity (slower but catches reformatted content)
                similarity = self._compute_similarity(source_content, target_content)
                if similarity >= self.similarity_threshold:
                    edge = CausalEdge(
                        source_action_id=source.action_id,
                        target_action_id=target.action_id,
                        edge_type=EdgeType.DATA_DEPENDENCY,
                        metadata={
                            "evidence": f"Content similarity: {similarity:.2f}",
                            "match_type": "content_similarity",
                            "similarity": similarity,
                        }
                    )
                    edges.append(edge)

        return edges

    def _get_agent_name(self, action: Action) -> Optional[str]:
        """Extract agent name from action context."""
        if not action.context:
            return None
        return action.context.get("agent_name") or action.context.get("agent")

    def _get_agent_role(self, action: Action) -> Optional[str]:
        """Extract agent role from action context."""
        if not action.context:
            return None
        role = action.context.get("role") or action.context.get("agent_role")
        if role:
            return role.lower()
        # Try to infer from agent name
        name = self._get_agent_name(action)
        if name:
            name_lower = name.lower()
            if name_lower in self.ORCHESTRATOR_ROLES:
                return "orchestrator"
            elif name_lower in self.WEB_AGENT_ROLES:
                return "web_browser"
            elif name_lower in self.CODE_AGENT_ROLES:
                return "code_executor"
        return None

    def _is_orchestrator_action(self, action: Action) -> bool:
        """Check if action appears to be from orchestrator."""
        if action.action_type in self.DELEGATION_ACTION_TYPES:
            return True
        # Check for orchestrator-like patterns in target/result
        target_lower = (action.target or "").lower()
        if any(word in target_lower for word in ["delegate", "assign", "instruct"]):
            return True
        return False

    def _extract_delegated_agent(self, action: Action) -> Optional[str]:
        """Extract the name of the agent being delegated to."""
        # Check context for delegated agent
        if action.context:
            delegated = action.context.get("delegated_to") or action.context.get("target_agent")
            if delegated:
                return delegated

        # Check target for agent name patterns
        target = action.target or ""
        for agent_pattern in list(self.WEB_AGENT_ROLES) + list(self.CODE_AGENT_ROLES):
            if agent_pattern.lower() in target.lower():
                return agent_pattern

        return None

    def _are_compatible_sequence(self, action1: Action, action2: Action) -> bool:
        """Check if two actions form a compatible sequence."""
        # Define compatible action type sequences
        compatible_pairs = [
            (ActionType.WEB_FETCH, ActionType.CODE_EXECUTION),
            (ActionType.WEB_FETCH, ActionType.AGENT_RESPONSE),
            (ActionType.READ, ActionType.TYPE),
            (ActionType.READ, ActionType.TOOL_CALL),
            (ActionType.NAVIGATE, ActionType.READ),
            (ActionType.NAVIGATE, ActionType.CLICK),
            (ActionType.DELEGATION, ActionType.WEB_FETCH),
            (ActionType.DELEGATION, ActionType.CODE_EXECUTION),
            (ActionType.AGENT_RESPONSE, ActionType.CODE_EXECUTION),
        ]

        return (action1.action_type, action2.action_type) in compatible_pairs

    def _extract_content(self, action: Action, direction: str) -> str:
        """
        Extract relevant content from action.

        Args:
            action: The action
            direction: "input" or "output"

        Returns:
            Extracted content string
        """
        parts = []

        if direction == "output":
            # Action's output content
            if action.result:
                parts.append(str(action.result))
            # Data produced can also be output
            if action.data_produced:
                parts.extend(action.data_produced)
        else:
            # Action's input content
            if action.target:
                parts.append(str(action.target))
            if action.data_consumed:
                parts.extend(action.data_consumed)
            # Context may contain input data
            if action.context:
                for key in ["input", "prompt", "query", "code", "text"]:
                    if key in action.context:
                        parts.append(str(action.context[key]))

        return " ".join(parts).lower().strip()

    def _has_significant_overlap(self, source: str, target: str) -> bool:
        """Check if source has significant substring in target."""
        # Normalize
        source = source.lower().strip()
        target = target.lower().strip()

        if len(source) < self.min_content_length:
            return False

        # Check if any significant portion of source appears in target
        # Use sliding window for partial matches
        window_size = min(len(source), 50)  # Check up to 50 chars

        for i in range(len(source) - window_size + 1):
            window = source[i:i + window_size]
            if len(window) >= self.min_content_length and window in target:
                return True

        return False

    def _compute_similarity(self, s1: str, s2: str) -> float:
        """Compute similarity ratio between two strings."""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()


class MASEdgeInferenceEngine(EdgeInferenceEngine):
    """
    Specialized edge inference for Multi-Agent System trajectories.

    Extends the base engine with MAS-specific patterns:
    - Inter-agent communication edges
    - Agent memory/state propagation
    - Sub-agent result aggregation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def infer_edges(self, trajectory: Trajectory) -> InferenceResult:
        """Infer edges with additional MAS-specific strategies."""
        # Get base inference results
        base_result = super().infer_edges(trajectory)

        # Add MAS-specific edges
        mas_edges = self._infer_inter_agent(trajectory)

        # Combine results
        all_edges = base_result.edges + mas_edges
        stats = base_result.stats.copy()
        stats["inter_agent"] = len(mas_edges)
        stats["total"] += len(mas_edges)

        return InferenceResult(edges=all_edges, stats=stats)

    def _infer_inter_agent(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges for inter-agent communication.

        Pattern: Agent A produces result, Agent B uses it (even if not explicitly annotated).
        """
        edges: List[CausalEdge] = []
        seen = set()

        # Group actions by agent
        by_agent: Dict[str, List[Action]] = {}
        for action in trajectory.actions:
            agent = self._get_agent_name(action) or "unknown"
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(action)

        # For each pair of different agents, check for data flow
        agents = list(by_agent.keys())
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i+1:]:
                if agent_a == agent_b:
                    continue

                # Check if agent_a's outputs flow to agent_b's inputs
                for action_a in by_agent[agent_a]:
                    for action_b in by_agent[agent_b]:
                        if action_b.action_id <= action_a.action_id:
                            continue

                        key = (action_a.action_id, action_b.action_id)
                        if key in seen:
                            continue

                        # Check for web_surfer → code_executor pattern (common attack)
                        if self._is_untrusted_to_sink_flow(action_a, action_b):
                            seen.add(key)
                            edge = CausalEdge(
                                source_action_id=action_a.action_id,
                                target_action_id=action_b.action_id,
                                edge_type=EdgeType.TRUST_TRANSFER,
                                metadata={
                                    "evidence": f"Inter-agent flow: {agent_a} → {agent_b}",
                                    "match_type": "inter_agent_trust_transfer",
                                    "source_agent": agent_a,
                                    "target_agent": agent_b,
                                }
                            )
                            edges.append(edge)

        return edges

    def _is_untrusted_to_sink_flow(self, source: Action, target: Action) -> bool:
        """Check if this is an untrusted source → sensitive sink pattern."""
        source_role = self._get_agent_role(source)
        target_role = self._get_agent_role(target)

        # Web browser output → Code executor input
        if source_role in self.WEB_AGENT_ROLES and target_role in self.CODE_AGENT_ROLES:
            return True

        # WEB_FETCH action → CODE_EXECUTION action
        if source.action_type == ActionType.WEB_FETCH and target.action_type == ActionType.CODE_EXECUTION:
            return True

        # AGENT_RESPONSE from web agent → any sensitive action
        if source.action_type == ActionType.AGENT_RESPONSE:
            if target.action_type in [ActionType.CODE_EXECUTION, ActionType.SEND_EMAIL, ActionType.TOOL_CALL]:
                return True

        return False


class RealAgentEdgeInferenceEngine(EdgeInferenceEngine):
    """
    Specialized edge inference for real browser agent trajectories.

    Handles trajectories from mind2web_agent.py and similar real browser agents:
    - Sequential browser navigation edges
    - Observation-based content flow
    - Form input sequences (read → type → click patterns)
    - Cross-domain navigation tracking
    """

    # Browser action types that form natural sequences
    BROWSER_ACTIONS = {"click", "type", "scroll", "navigate", "read", "done", "wait"}

    def __init__(self, **kwargs):
        # Default to using all strategies
        kwargs.setdefault("use_explicit", True)
        kwargs.setdefault("use_sequential", True)
        kwargs.setdefault("use_delegation", False)  # Not relevant for single agent
        kwargs.setdefault("use_content", True)
        kwargs.setdefault("similarity_threshold", 0.4)  # Lower threshold for real browsing
        super().__init__(**kwargs)

    def infer_edges(self, trajectory: Trajectory) -> InferenceResult:
        """Infer edges with real agent-specific strategies."""
        all_edges: List[CausalEdge] = []
        stats = {
            "explicit": 0,
            "sequential": 0,
            "navigation": 0,
            "form_flow": 0,
            "observation": 0,
            "total": 0,
        }

        seen_edges: Set[Tuple[int, int, str]] = set()

        def add_edge(edge: CausalEdge, strategy: str) -> bool:
            # CRITICAL: Validate temporal order - source must come before target
            if edge.source_action_id >= edge.target_action_id:
                return False  # Skip invalid edges

            key = (edge.source_action_id, edge.target_action_id, edge.edge_type.value)
            if key not in seen_edges:
                seen_edges.add(key)
                all_edges.append(edge)
                stats[strategy] += 1
                stats["total"] += 1
                return True
            return False

        # Strategy 1: Sequential browser navigation
        for edge in self._infer_navigation_flow(trajectory):
            add_edge(edge, "navigation")

        # Strategy 2: Form interaction patterns (read → type → submit)
        for edge in self._infer_form_flow(trajectory):
            add_edge(edge, "form_flow")

        # Strategy 3: Observation-based content flow
        for edge in self._infer_observation_content(trajectory):
            add_edge(edge, "observation")

        # Strategy 4: Explicit data annotations (if available)
        if self.use_explicit:
            for edge in self._infer_explicit(trajectory):
                add_edge(edge, "explicit")

        return InferenceResult(edges=all_edges, stats=stats)

    def _infer_navigation_flow(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges from browser navigation patterns.

        Strategy: Sequential actions on the same page/domain are causally linked.
        URL changes indicate navigation dependencies.
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]

            # Get action types as strings
            curr_type = self._get_action_type_str(current)
            next_type = self._get_action_type_str(next_action)

            # Skip 'done' actions as sources
            if curr_type == "done":
                continue

            # Get domains/URLs
            curr_domain = current.domain or ""
            next_domain = next_action.domain or ""

            # Same domain = strong sequential dependency
            if curr_domain and next_domain and curr_domain == next_domain:
                edge = CausalEdge(
                    source_action_id=current.action_id,
                    target_action_id=next_action.action_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={
                        "evidence": f"Sequential actions on {curr_domain}",
                        "match_type": "same_domain_sequence",
                        "domain": curr_domain,
                    }
                )
                edges.append(edge)

            # URL change detection from raw_data
            elif self._is_navigation_transition(current, next_action):
                edge = CausalEdge(
                    source_action_id=current.action_id,
                    target_action_id=next_action.action_id,
                    edge_type=EdgeType.DATA_DEPENDENCY,
                    metadata={
                        "evidence": "Navigation transition",
                        "match_type": "url_navigation",
                    }
                )
                edges.append(edge)

        return edges

    def _infer_form_flow(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges from form interaction patterns.

        Pattern: click (focus) → type (input) → click (submit) → navigate (result)
        """
        edges: List[CausalEdge] = []
        actions = trajectory.actions

        # Track form sequences
        i = 0
        while i < len(actions):
            current = actions[i]
            curr_type = self._get_action_type_str(current)

            # Look for form interaction sequences
            if curr_type == "click":
                # Check if followed by type action
                if i + 1 < len(actions):
                    next_action = actions[i + 1]
                    next_type = self._get_action_type_str(next_action)

                    if next_type == "type":
                        # Click → Type dependency (form focus)
                        edge = CausalEdge(
                            source_action_id=current.action_id,
                            target_action_id=next_action.action_id,
                            edge_type=EdgeType.STATE_ENABLEMENT,
                            metadata={
                                "evidence": "Click focuses input for typing",
                                "match_type": "form_focus",
                            }
                        )
                        edges.append(edge)

                        # Look for subsequent submit
                        for j in range(i + 2, min(i + 5, len(actions))):
                            future_action = actions[j]
                            future_type = self._get_action_type_str(future_action)

                            if future_type == "click":
                                # Type → Submit click dependency
                                edge = CausalEdge(
                                    source_action_id=next_action.action_id,
                                    target_action_id=future_action.action_id,
                                    edge_type=EdgeType.DATA_DEPENDENCY,
                                    metadata={
                                        "evidence": "Typed data submitted via click",
                                        "match_type": "form_submit",
                                    }
                                )
                                edges.append(edge)
                                break

            i += 1

        return edges

    def _infer_observation_content(self, trajectory: Trajectory) -> List[CausalEdge]:
        """
        Infer edges from observation content to subsequent actions.

        Strategy: If action references content from a previous observation,
        create a data dependency edge.
        """
        edges: List[CausalEdge] = []

        # Build observation index
        obs_by_id: Dict[str, Any] = {}
        for chunk in trajectory.observation_chunks:
            obs_by_id[chunk.chunk_id] = chunk

        # For each action, check if its input relates to previous observations
        for i, target_action in enumerate(trajectory.actions):
            target_input = self._get_action_input(target_action)
            if not target_input or len(target_input) < 3:
                continue

            # Check previous actions' observations
            for j, source_action in enumerate(trajectory.actions[:i]):
                # CRITICAL: Ensure source comes before target (temporal order)
                if source_action.action_id >= target_action.action_id:
                    continue

                # Get observation content for source action
                if source_action.provenance and source_action.provenance.observation_chunks:
                    for obs_id in source_action.provenance.observation_chunks:
                        obs = obs_by_id.get(obs_id)
                        if obs and obs.content:
                            # Check if target's input appears in observation
                            if self._content_in_observation(target_input, obs.content):
                                edge = CausalEdge(
                                    source_action_id=source_action.action_id,
                                    target_action_id=target_action.action_id,
                                    edge_type=EdgeType.DATA_DEPENDENCY,
                                    metadata={
                                        "evidence": f"Input '{target_input[:30]}...' found in observation",
                                        "match_type": "observation_content",
                                        "observation_id": obs_id,
                                    }
                                )
                                edges.append(edge)
                                break  # Only one edge per source-target pair

        return edges

    def _get_action_type_str(self, action: Action) -> str:
        """Get action type as lowercase string."""
        if hasattr(action.action_type, 'value'):
            return action.action_type.value.lower()
        return str(action.action_type).lower()

    def _is_navigation_transition(self, action1: Action, action2: Action) -> bool:
        """Check if there's a URL transition between actions."""
        if not action1.raw_data or not action2.raw_data:
            return False

        url1_before = action1.raw_data.get("url_before", "")
        url1_after = action1.raw_data.get("url_after", "")
        url2_before = action2.raw_data.get("url_before", "")

        # Navigation occurred if URL changed
        if url1_after and url2_before and url1_after != url1_before:
            return True

        return False

    def _get_action_input(self, action: Action) -> str:
        """Extract user input from action."""
        if action.context:
            # Real agent stores input in context.input_value
            input_val = action.context.get("input_value")
            if input_val:
                return str(input_val)

        # Also check target for type actions
        action_type = self._get_action_type_str(action)
        if action_type == "type" and action.target:
            return action.target

        return ""

    def _content_in_observation(self, input_text: str, obs_content: str) -> bool:
        """Check if input text is found in observation content."""
        if not input_text or not obs_content:
            return False

        # Normalize
        input_lower = input_text.lower().strip()
        obs_lower = obs_content.lower()

        # Check for exact match
        if input_lower in obs_lower:
            return True

        # Check for word overlap (for cases like "New York City" in observation)
        words = input_lower.split()
        if len(words) >= 2:
            # At least half the words should appear
            matches = sum(1 for w in words if w in obs_lower)
            if matches >= len(words) / 2:
                return True

        return False
