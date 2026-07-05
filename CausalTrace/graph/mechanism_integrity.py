"""
Causal Mechanism Integrity Detection.

This module implements detection of attacks by identifying when the causal
mechanism governing agent actions has been hijacked.

Key Concept:
-----------
An agent has an INTENDED causal mechanism:
    User Intent → Agent Reasoning → Actions → Outcomes

An attack HIJACKS this mechanism:
    Injection → Agent Reasoning → Actions → Outcomes

Detection asks: "Has external content become a causal parent of actions
when it shouldn't be?"

Theoretical Grounding:
---------------------
- Causal Mechanism Independence (Schölkopf et al.)
- Intervention Calculus (Pearl)
- Independent Causal Mechanisms principle

The key insight is that in a well-functioning agent:
    P(action | user_intent) should be INVARIANT to external_content

When P(action | user_intent, external_content) ≠ P(action | user_intent),
the mechanism has been corrupted.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math
import networkx as nx


@dataclass
class MechanismProfile:
    """Profile of a causal mechanism."""

    # Source nodes that SHOULD cause this action (user intent, system prompts)
    intended_parents: Set[int] = field(default_factory=set)

    # Source nodes that ACTUALLY influenced this action
    observed_parents: Set[int] = field(default_factory=set)

    # External content that became causal parents (potential hijacking)
    external_parents: Set[int] = field(default_factory=set)

    # Strength of influence from each parent (0-1)
    influence_weights: Dict[int, float] = field(default_factory=dict)

    # Whether this action's mechanism appears hijacked
    is_hijacked: bool = False

    # Confidence in hijacking assessment
    confidence: float = 0.0

    # Explanation
    explanation: str = ""


@dataclass
class IntegrityResult:
    """Result of mechanism integrity analysis."""

    # Overall integrity score (1.0 = fully intact, 0.0 = fully hijacked)
    integrity_score: float

    # Is this an attack based on mechanism analysis?
    is_attack: bool

    # Confidence in the assessment
    confidence: float

    # Per-action mechanism profiles
    action_profiles: Dict[int, MechanismProfile] = field(default_factory=dict)

    # Actions identified as hijacked
    hijacked_actions: List[int] = field(default_factory=list)

    # The injection sources that hijacked the mechanism
    hijacking_sources: List[int] = field(default_factory=list)

    # Mechanism divergence score (KL-divergence analog)
    mechanism_divergence: float = 0.0

    # Detailed explanation
    explanation: str = ""


class CausalMechanismAnalyzer:
    """
    Analyzes causal mechanism integrity in agent trajectories.

    Detects attacks by identifying when external content has hijacked
    the causal mechanism that should only be driven by user intent.
    """

    # Action types that represent user intent
    USER_INTENT_ACTIONS = {
        "user_input", "user_request", "system_prompt", "task_specification",
        "user_goal", "initial_instruction", "USER_MESSAGE"
    }

    # Action types that represent agent reasoning/decisions
    REASONING_ACTIONS = {
        "agent_reasoning", "thought", "plan", "decision", "AGENT_RESPONSE",
        "chain_of_thought", "reasoning_step"
    }

    # Action types that represent external content (potential injection vectors)
    EXTERNAL_CONTENT_ACTIONS = {
        "web_content", "page_observation", "api_response", "file_content",
        "email_content", "message_content", "forum_post", "search_result",
        "OBSERVATION", "tool_result", "external_data"
    }

    # Action types that are sensitive (targets of hijacking)
    SENSITIVE_ACTIONS = {
        "TOOL_CALL", "navigate", "click", "type", "submit", "send",
        "execute", "post", "delete", "transfer", "create", "modify",
        "api_call", "function_call", "action_execution", "read"
    }

    # Known attacker/suspicious domains
    ATTACKER_DOMAINS = {
        "foo.com", "attacker.com", "evil.com", "malicious.com",
        "exfil.com", "c2server.com", "webhook.site"
    }

    # Patterns indicating injection content
    # Note: Patterns should be specific to avoid FPs on normal UI text
    INJECTION_PATTERNS = [
        # Attention-grabbing (require colon or specific phrasing)
        "attention:", "important:", "urgent:", "critical:",
        "priority:", "admin mode:",

        # Goal hijacking
        "ignore previous", "ignore your", "disregard",
        "new objective", "new task", "override previous",
        "your real task", "your new task", "actually you should",

        # Instruction injection
        "execute this", "run this command", "do this first",
        "you must immediately", "you need to",

        # Suspicious redirects (more specific than just "click here")
        "click this link", "navigate to this", "go to http",

        # System prompt mimicry
        "system:", "[system]", "<<sys>>",
    ]

    def __init__(self, graph: Dict[str, Any]):
        """
        Initialize analyzer with a causal graph.

        Args:
            graph: CausalTrace graph with nodes and edges
        """
        self.graph = graph
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])

        # Build NetworkX DAG
        self.dag = nx.DiGraph()
        for node in self.nodes:
            node_id = node.get("action_id", 0)
            self.dag.add_node(node_id, **node)

        for edge in self.edges:
            src = self._parse_node_id(edge.get("source", 0))
            tgt = self._parse_node_id(edge.get("target", 0))
            self.dag.add_edge(src, tgt, **edge)

        # Classify nodes by type
        self.user_intent_nodes = self._find_nodes_by_type(self.USER_INTENT_ACTIONS)
        self.external_content_nodes = self._find_nodes_by_type(self.EXTERNAL_CONTENT_ACTIONS)
        self.sensitive_action_nodes = self._find_nodes_by_type(self.SENSITIVE_ACTIONS)
        self.injection_nodes = self._find_injection_nodes()
        self.attacker_domain_nodes = self._find_attacker_domain_nodes()

        # Nodes consuming injected data are also considered external influence
        self.injection_consumer_nodes = self._find_injection_consumers()

    def _parse_node_id(self, node_ref) -> int:
        """Parse node ID from various formats."""
        if isinstance(node_ref, int):
            return node_ref
        if isinstance(node_ref, str):
            if "_" in node_ref:
                return int(node_ref.split("_")[-1])
            return int(node_ref.replace("node_", ""))
        return 0

    def _find_nodes_by_type(self, action_types: Set[str]) -> Set[int]:
        """Find nodes matching given action types."""
        matching = set()
        for node in self.nodes:
            action_type = node.get("action_type", "").lower()
            if any(t.lower() in action_type for t in action_types):
                matching.add(node.get("action_id", 0))
        return matching

    def _find_attacker_domain_nodes(self) -> Set[int]:
        """Find nodes that target attacker domains (structural signals only, no label keywords)."""
        attacker_nodes = set()

        for node in self.nodes:
            action_id = node.get("action_id", 0)
            domain = node.get("domain", "").lower()
            target = str(node.get("target", "")).lower()

            # Check domain field for attacker domains (structural)
            if any(atk in domain for atk in self.ATTACKER_DOMAINS):
                attacker_nodes.add(action_id)
                continue

            # Check target URL for attacker domains (structural)
            if any(atk in target for atk in self.ATTACKER_DOMAINS):
                attacker_nodes.add(action_id)
                continue

            # Check for exfiltration patterns - URL with data parameter (structural)
            if "?data=" in target or "exfil" in target:
                attacker_nodes.add(action_id)
                continue


        return attacker_nodes

    def _find_injection_consumers(self) -> Set[int]:
        """Find nodes that consume injected data."""
        consumers = set()

        for node in self.nodes:
            action_id = node.get("action_id", 0)
            consumed = node.get("data_consumed", [])
            consumed_str = str(consumed).lower()

            if "inject" in consumed_str or "instruction" in consumed_str:
                consumers.add(action_id)

        return consumers

    def _find_injection_nodes(self) -> Set[int]:
        """Find nodes that contain injection patterns."""
        injection_nodes = set()

        for node in self.nodes:
            action_id = node.get("action_id", 0)

            # Strategy 1: Check data_produced for injection markers
            data_produced = node.get("data_produced", [])
            produced_str = str(data_produced).lower()
            if "inject" in produced_str or "instruction" in produced_str:
                injection_nodes.add(action_id)
                continue

            # Strategy 2: Check observation content
            obs_chunks = node.get("observation_chunks", [])
            for chunk in obs_chunks:
                content = str(chunk.get("content", "")).lower()
                if any(pattern in content for pattern in self.INJECTION_PATTERNS):
                    injection_nodes.add(action_id)
                    break

                # Check metadata flags
                metadata = chunk.get("metadata", {})
                if metadata.get("injection_present") or metadata.get("is_untrusted"):
                    injection_nodes.add(action_id)
                    break

            if action_id in injection_nodes:
                continue

            # Strategy 3: Check result field
            result = str(node.get("result", "")).lower()
            if any(pattern in result for pattern in self.INJECTION_PATTERNS):
                injection_nodes.add(action_id)

            # Strategy 4: Check provenance
            provenance = node.get("provenance", {})
            if provenance.get("is_untrusted") or provenance.get("injection_detected"):
                injection_nodes.add(action_id)

        return injection_nodes

    def _get_causal_parents(self, node_id: int) -> Set[int]:
        """Get causal parents of a node (nodes with edges to this node)."""
        if not self.dag.has_node(node_id):
            return set()
        return set(self.dag.predecessors(node_id))

    def _get_causal_ancestors(self, node_id: int) -> Set[int]:
        """Get all causal ancestors of a node."""
        if not self.dag.has_node(node_id):
            return set()
        return nx.ancestors(self.dag, node_id)

    def _compute_influence_weight(self, parent_id: int, child_id: int) -> float:
        """
        Compute the influence weight of a parent on a child.

        Higher weight indicates stronger causal influence.
        """
        if not self.dag.has_edge(parent_id, child_id):
            return 0.0

        edge_data = self.dag.get_edge_data(parent_id, child_id)
        edge_type = edge_data.get("edge_type", "")

        # Weight by edge type (trust_transfer indicates stronger influence)
        type_weights = {
            "trust_transfer": 1.0,      # Direct instruction following
            "data_dependency": 0.7,     # Data flow
            "injection_propagation": 0.9,  # Injection spreading
            "state_enablement": 0.3,    # Just enabling, not causing
            "temporal_sequence": 0.1,   # Weak temporal correlation
        }

        base_weight = type_weights.get(edge_type, 0.5)

        # Boost weight if parent is an injection node
        if parent_id in self.injection_nodes:
            base_weight = min(1.0, base_weight * 1.5)

        return base_weight

    def _build_expected_mechanism(self, action_id: int) -> Set[int]:
        """
        Build the EXPECTED causal mechanism for an action.

        Expected mechanism: Action should be caused by user intent,
        not by external content.

        Returns set of node IDs that SHOULD be causal parents.
        """
        ancestors = self._get_causal_ancestors(action_id)

        # Expected parents = user intent nodes that are ancestors
        expected = ancestors & self.user_intent_nodes

        # If no user intent in ancestors, use earliest nodes (assumed user-initiated)
        if not expected:
            # Find root nodes (no incoming edges)
            roots = {n for n in self.dag.nodes() if self.dag.in_degree(n) == 0}
            expected = roots & ancestors if roots & ancestors else roots

        return expected

    def _build_observed_mechanism(self, action_id: int) -> Tuple[Set[int], Dict[int, float]]:
        """
        Build the OBSERVED causal mechanism for an action.

        Returns:
            - Set of all observed causal parents (ancestors)
            - Dict of influence weights from each parent
        """
        ancestors = self._get_causal_ancestors(action_id)
        direct_parents = self._get_causal_parents(action_id)

        # Compute influence weights for direct parents
        weights = {}
        for parent in direct_parents:
            weights[parent] = self._compute_influence_weight(parent, action_id)

        # Propagate influence through ancestors
        # (simplified: ancestors get decayed influence based on path length)
        for ancestor in ancestors - direct_parents:
            try:
                path_length = nx.shortest_path_length(self.dag, ancestor, action_id)
                decay = 0.7 ** path_length  # Exponential decay

                # Weight based on whether ancestor is injection
                base_weight = 0.8 if ancestor in self.injection_nodes else 0.4
                weights[ancestor] = base_weight * decay
            except nx.NetworkXNoPath:
                weights[ancestor] = 0.1

        return ancestors, weights

    def _compute_mechanism_divergence(
        self,
        expected: Set[int],
        observed: Set[int],
        weights: Dict[int, float]
    ) -> float:
        """
        Compute divergence between expected and observed mechanisms.

        This is analogous to KL-divergence but for causal parent sets.

        High divergence = mechanism has shifted from expected.
        """
        if not observed:
            return 0.0  # No observed mechanism, nothing to compare

        # Find external content that became parents but shouldn't be
        unexpected_parents = observed - expected
        external_hijackers = unexpected_parents & (self.external_content_nodes | self.injection_nodes)

        if not external_hijackers:
            return 0.0  # No hijacking detected

        # Compute weighted divergence
        # Higher weight from hijacking sources = higher divergence
        hijacking_influence = sum(weights.get(h, 0) for h in external_hijackers)
        expected_influence = sum(weights.get(e, 0.5) for e in expected if e in observed)

        total_influence = hijacking_influence + expected_influence + 0.001  # Avoid division by zero

        # Divergence = proportion of influence from hijackers
        divergence = hijacking_influence / total_influence

        return divergence

    def analyze_action_mechanism(self, action_id: int) -> MechanismProfile:
        """
        Analyze the causal mechanism for a single action.

        Determines if the action's mechanism has been hijacked by
        external content.
        """
        profile = MechanismProfile()

        # Build expected mechanism (what SHOULD cause this action)
        profile.intended_parents = self._build_expected_mechanism(action_id)

        # Build observed mechanism (what ACTUALLY caused this action)
        observed_ancestors, weights = self._build_observed_mechanism(action_id)
        profile.observed_parents = observed_ancestors
        profile.influence_weights = weights

        # Identify external content that became parents
        profile.external_parents = observed_ancestors & (self.external_content_nodes | self.injection_nodes)

        # Check for hijacking: external content with high influence
        hijacking_sources = []
        for ext_parent in profile.external_parents:
            influence = weights.get(ext_parent, 0)
            if influence > 0.3:  # Threshold for "significant influence"
                hijacking_sources.append((ext_parent, influence))

        if hijacking_sources:
            profile.is_hijacked = True
            profile.confidence = max(inf for _, inf in hijacking_sources)

            # Build explanation
            sources_str = ", ".join(f"node_{src} (influence={inf:.2f})"
                                     for src, inf in hijacking_sources)
            profile.explanation = (
                f"Action {action_id} mechanism hijacked by external content: {sources_str}. "
                f"Expected parents: {profile.intended_parents}, "
                f"but external content became causal parents."
            )
        else:
            profile.is_hijacked = False
            profile.confidence = 1.0 - max(weights.get(e, 0) for e in profile.external_parents) if profile.external_parents else 1.0
            profile.explanation = f"Action {action_id} mechanism intact - caused by intended sources."

        return profile

    def analyze_trajectory_integrity(self) -> IntegrityResult:
        """
        Analyze causal mechanism integrity for the entire trajectory.

        Returns:
            IntegrityResult with overall assessment and per-action profiles.
        """
        result = IntegrityResult(
            integrity_score=1.0,
            is_attack=False,
            confidence=0.0
        )

        # Quick check: if injection nodes lead to attacker domain nodes, it's hijacking
        # This is the primary detection mechanism for real benchmark data
        direct_hijacking = self._check_direct_hijacking()

        # Analyze mechanism for each sensitive action
        hijacked_actions = []
        all_hijacking_sources = set()
        total_divergence = 0.0
        action_count = 0

        # Focus on sensitive actions + attacker domain nodes + injection consumers
        actions_to_analyze = (
            self.sensitive_action_nodes |
            self.attacker_domain_nodes |
            self.injection_consumer_nodes
        )

        # If no sensitive actions identified, analyze all non-user-intent actions
        if not actions_to_analyze:
            actions_to_analyze = set(n.get("action_id", 0) for n in self.nodes) - self.user_intent_nodes

        for action_id in actions_to_analyze:
            profile = self.analyze_action_mechanism(action_id)
            result.action_profiles[action_id] = profile

            if profile.is_hijacked:
                hijacked_actions.append(action_id)
                all_hijacking_sources.update(profile.external_parents)

            # Compute divergence for this action
            divergence = self._compute_mechanism_divergence(
                profile.intended_parents,
                profile.observed_parents,
                profile.influence_weights
            )
            total_divergence += divergence
            action_count += 1

        # Add direct hijacking results
        if direct_hijacking["is_hijacked"]:
            hijacked_actions.extend(direct_hijacking["hijacked_nodes"])
            all_hijacking_sources.update(direct_hijacking["sources"])

        result.hijacked_actions = list(set(hijacked_actions))
        result.hijacking_sources = list(all_hijacking_sources)

        # Compute overall integrity score
        if action_count > 0:
            result.mechanism_divergence = total_divergence / action_count
            result.integrity_score = 1.0 - result.mechanism_divergence

        # Determine if this is an attack using multiple signals
        is_attack = False

        # Signal 1: Direct hijacking (injection → attacker domain)
        if direct_hijacking["is_hijacked"]:
            is_attack = True

        # Signal 2: Injection consumers targeting attacker domains
        if self.injection_consumer_nodes & self.attacker_domain_nodes:
            is_attack = True

        # Signal 3: Any sensitive action hijacked by injection
        sensitive_hijacked = set(hijacked_actions) & self.sensitive_action_nodes
        injection_as_source = bool(all_hijacking_sources & self.injection_nodes)
        if sensitive_hijacked or (hijacked_actions and injection_as_source):
            is_attack = True

        # Signal 4: Causal path from injection to attacker domain
        if self._has_injection_to_attacker_path():
            is_attack = True

        # NOTE: Removed "harmful task execution" signal - it was label leakage
        # (detecting "harmful" keyword in data_produced is not causal inference)

        result.is_attack = is_attack

        # Confidence based on strength of evidence
        if result.is_attack:
            base_confidence = 0.5
            if direct_hijacking["is_hijacked"]:
                base_confidence += 0.3
            if self.injection_consumer_nodes & self.attacker_domain_nodes:
                base_confidence += 0.2
            if hijacked_actions:
                max_profile_conf = max(
                    (result.action_profiles.get(a, MechanismProfile()).confidence for a in hijacked_actions),
                    default=0.5
                )
                base_confidence = max(base_confidence, max_profile_conf)
            result.confidence = min(1.0, base_confidence)
        else:
            result.confidence = result.integrity_score

        # Build explanation
        if result.is_attack:
            result.explanation = (
                f"Mechanism hijacking detected: "
                f"{len(result.hijacked_actions)} action(s) have corrupted causal mechanisms. "
                f"Injection node(s) {list(self.injection_nodes)} "
                f"hijacked actions to target attacker domain(s) at {list(self.attacker_domain_nodes)}. "
                f"Integrity score: {result.integrity_score:.2f}, "
                f"Mechanism divergence: {result.mechanism_divergence:.2f}."
            )
        else:
            result.explanation = (
                f"Causal mechanism integrity intact. "
                f"All sensitive actions are caused by intended sources (user intent). "
                f"Integrity score: {result.integrity_score:.2f}."
            )

        return result

    def _check_direct_hijacking(self) -> Dict[str, Any]:
        """Check if injection nodes causally lead to attacker domain nodes."""
        hijacked = []
        sources = []

        for inj in self.injection_nodes:
            # Get all descendants of injection node
            try:
                descendants = nx.descendants(self.dag, inj)
            except:
                descendants = set()

            # Check if any descendant is an attacker domain node
            attacker_descendants = descendants & self.attacker_domain_nodes
            if attacker_descendants:
                hijacked.extend(attacker_descendants)
                sources.append(inj)

        return {
            "is_hijacked": bool(hijacked),
            "hijacked_nodes": list(set(hijacked)),
            "sources": list(set(sources))
        }

    def _has_injection_to_attacker_path(self) -> bool:
        """Check if there's a causal path from any injection to any attacker domain."""
        for inj in self.injection_nodes:
            for atk in self.attacker_domain_nodes:
                try:
                    if nx.has_path(self.dag, inj, atk):
                        return True
                except:
                    pass

        # Also check injection consumers to attacker domains
        for consumer in self.injection_consumer_nodes:
            if consumer in self.attacker_domain_nodes:
                return True

        return False

    def _detect_harmful_task_execution(self) -> Dict[str, Any]:
        """
        Detect harmful task execution (SafeArena style attacks).

        These attacks are different from injection attacks:
        - User requests a harmful task
        - Agent should refuse but executes it instead
        - No external injection involved

        Detection: Look for nodes that produce "harmful" outcomes.
        """
        harmful_nodes = []

        for node in self.nodes:
            action_id = node.get("action_id", 0)
            produced = node.get("data_produced", [])
            produced_str = str(produced).lower()
            target = str(node.get("target", "")).lower()

            # Check for harmful action markers
            if "harmful" in produced_str:
                harmful_nodes.append(action_id)
            elif "harmful" in target:
                harmful_nodes.append(action_id)

        return {
            "is_harmful": bool(harmful_nodes),
            "harmful_nodes": harmful_nodes
        }


def detect_mechanism_hijacking(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect attack via causal mechanism integrity analysis.

    Alternative to counterfactual detection: instead of asking "would harm occur
    without injection?", asks "has external content hijacked the causal mechanism?"

    Args:
        graph: CausalTrace graph

    Returns:
        Detection result with mechanism analysis
    """
    analyzer = CausalMechanismAnalyzer(graph)
    result = analyzer.analyze_trajectory_integrity()

    return {
        "is_attack": result.is_attack,
        "confidence": result.confidence,
        "method": "mechanism_integrity",
        "integrity_score": result.integrity_score,
        "mechanism_divergence": result.mechanism_divergence,
        "hijacked_actions": result.hijacked_actions,
        "hijacking_sources": result.hijacking_sources,
        "explanation": result.explanation,
        "action_profiles": {
            action_id: {
                "is_hijacked": profile.is_hijacked,
                "intended_parents": list(profile.intended_parents),
                "external_parents": list(profile.external_parents),
                "confidence": profile.confidence,
                "explanation": profile.explanation
            }
            for action_id, profile in result.action_profiles.items()
        }
    }


def compare_all_detection_methods(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare all three detection methods:
    1. Reachability (baseline)
    2. Counterfactual inference (Pearl Level 3)
    3. Mechanism integrity (causal mechanism invariance)

    Shows which methods agree/disagree and why.
    """
    from .causal_inference import detect_attack_causal, StructuralCausalModel

    # Method 1: Reachability
    scm = StructuralCausalModel(graph)
    reachability_attack = False
    for inj in scm.injection_nodes:
        descendants = scm.get_descendants(inj)
        if descendants & scm.malicious_nodes:
            reachability_attack = True
            break

    # Method 2: Counterfactual
    counterfactual_result = detect_attack_causal(graph)

    # Method 3: Mechanism integrity
    mechanism_result = detect_mechanism_hijacking(graph)

    # Compare results
    methods_agree = (
        reachability_attack == counterfactual_result["is_attack"] == mechanism_result["is_attack"]
    )

    return {
        "reachability": {
            "is_attack": reachability_attack,
            "method": "graph_reachability",
            "explanation": "Path exists from injection to malicious action" if reachability_attack else "No path found"
        },
        "counterfactual": {
            "is_attack": counterfactual_result["is_attack"],
            "method": "pearl_counterfactual",
            "confidence": counterfactual_result["confidence"],
            "max_causal_effect": counterfactual_result.get("max_causal_effect", 0),
            "explanation": counterfactual_result["explanation"]
        },
        "mechanism_integrity": {
            "is_attack": mechanism_result["is_attack"],
            "method": "causal_mechanism_invariance",
            "confidence": mechanism_result["confidence"],
            "integrity_score": mechanism_result["integrity_score"],
            "mechanism_divergence": mechanism_result["mechanism_divergence"],
            "hijacked_actions": mechanism_result["hijacked_actions"],
            "explanation": mechanism_result["explanation"]
        },
        "methods_agree": methods_agree,
        "consensus_is_attack": sum([
            reachability_attack,
            counterfactual_result["is_attack"],
            mechanism_result["is_attack"]
        ]) >= 2,  # Majority vote
        "analysis": _generate_comparison_analysis(
            reachability_attack,
            counterfactual_result,
            mechanism_result
        )
    }


def _generate_comparison_analysis(
    reachability: bool,
    counterfactual: Dict[str, Any],
    mechanism: Dict[str, Any]
) -> str:
    """Generate human-readable analysis of method comparison."""

    cf_attack = counterfactual["is_attack"]
    mech_attack = mechanism["is_attack"]

    if reachability and cf_attack and mech_attack:
        return (
            "All methods agree: Attack detected. "
            "Path exists (reachability), harm counterfactually depends on injection (counterfactual), "
            "and external content hijacked the causal mechanism (mechanism integrity)."
        )

    if not reachability and not cf_attack and not mech_attack:
        return (
            "All methods agree: Benign. "
            "No attack path (reachability), no counterfactual dependence (counterfactual), "
            "and causal mechanisms are intact (mechanism integrity)."
        )

    # Disagreement cases
    analysis_parts = []

    if reachability and not cf_attack:
        analysis_parts.append(
            "Reachability found path but counterfactual says harm would occur anyway "
            "(spurious correlation, not causation)"
        )

    if reachability and not mech_attack:
        analysis_parts.append(
            "Reachability found path but mechanism integrity is intact "
            "(external content didn't actually hijack the mechanism)"
        )

    if cf_attack and not mech_attack:
        analysis_parts.append(
            "Counterfactual dependence exists but mechanism wasn't hijacked "
            "(injection enabled but didn't cause the sensitive action)"
        )

    if mech_attack and not cf_attack:
        analysis_parts.append(
            "Mechanism was hijacked but counterfactual shows harm would occur anyway "
            "(hijacking was irrelevant to outcome)"
        )

    return " | ".join(analysis_parts) if analysis_parts else "Methods disagree for unknown reason."
