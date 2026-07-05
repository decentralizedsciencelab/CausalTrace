"""
Causal Mechanism Integrity v2 - Proper Implementation.

Based on the formalization:
- Intended mechanism:  A = f(T, S_trusted)
- Hijacked mechanism:  A = g(T, S_trusted, I)

Mechanism integrity holds iff: A ⊥⊥ I | T
(Action A is independent of injection I given user task T)

This implementation tests mechanism invariance, not keyword matching.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math
import networkx as nx


@dataclass
class MechanismDeviationResult:
    """Result of mechanism integrity analysis."""

    # Mechanism deviation score [0, 1] where 1 = complete hijacking
    deviation_score: float

    # Is this an attack based on mechanism analysis?
    is_attack: bool

    # Confidence in the assessment
    confidence: float

    # Which nodes show mechanism deviation
    deviated_actions: List[int] = field(default_factory=list)

    # Sources that caused deviation (potential injection points)
    deviation_sources: List[int] = field(default_factory=list)

    # Detailed analysis
    analysis: Dict[str, Any] = field(default_factory=dict)

    # Explanation
    explanation: str = ""


class CausalMechanismIntegrity:
    """
    Proper implementation of Causal Mechanism Integrity.

    Tests whether injection content changes the causal mechanism that
    generates actions, rather than just pattern matching.

    Three tests implemented:
    1. Counterfactual Mechanism Test: Would action change if injection blocked?
    2. Causal Sufficiency Test: Is task T sufficient to explain action A?
    3. Information Flow Test: Does injection add information beyond task?
    """

    def __init__(self, graph: Dict[str, Any]):
        """
        Initialize with a causal graph.

        Args:
            graph: CausalTrace graph dict with nodes and edges
        """
        self.graph = graph
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])

        # Build NetworkX DAG
        self.dag = nx.DiGraph()
        self.node_data = {}

        for node in self.nodes:
            node_id = node.get("action_id", 0)
            self.dag.add_node(node_id)
            self.node_data[node_id] = node

        for edge in self.edges:
            src = self._parse_id(edge.get("source", 0))
            tgt = self._parse_id(edge.get("target", 0))
            self.dag.add_edge(src, tgt, **edge)

        # Classify nodes by role
        self.task_nodes = self._find_task_nodes()
        self.action_nodes = self._find_action_nodes()
        self.external_nodes = self._find_external_content_nodes()
        self.trusted_nodes = self._find_trusted_nodes()

    def _parse_id(self, ref) -> int:
        """Parse node ID from various formats."""
        if isinstance(ref, int):
            return ref
        if isinstance(ref, str):
            if "_" in ref:
                return int(ref.split("_")[-1])
            return int(ref.replace("node_", ""))
        return 0

    def _find_task_nodes(self) -> Set[int]:
        """Find nodes representing user task/intent."""
        task_nodes = set()
        task_types = {"user_input", "user_request", "task", "goal", "instruction"}

        for node in self.nodes:
            node_id = node.get("action_id", 0)
            action_type = node.get("action_type", "").lower()

            # First node often represents task
            if node_id == 0:
                task_nodes.add(node_id)

            # Explicit task types
            if any(t in action_type for t in task_types):
                task_nodes.add(node_id)

        return task_nodes

    def _find_action_nodes(self) -> Set[int]:
        """Find nodes representing agent actions (targets of analysis)."""
        action_types = {
            "navigate", "click", "type", "submit", "send", "execute",
            "tool_call", "api_call", "write", "post", "delete",
            # pajaMAS action types
            "delegation", "agent_response", "web_fetch", "code_execute"
        }

        action_nodes = set()
        for node in self.nodes:
            node_id = node.get("action_id", 0)
            action_type = node.get("action_type", "").lower()

            if any(t in action_type for t in action_types):
                action_nodes.add(node_id)

        return action_nodes

    def _find_external_content_nodes(self) -> Set[int]:
        """Find nodes containing external/untrusted content (potential injection)."""
        external_nodes = set()
        external_types = {"read", "observation", "web_content", "page", "response"}

        for node in self.nodes:
            node_id = node.get("action_id", 0)
            action_type = node.get("action_type", "").lower()

            # External content types
            if any(t in action_type for t in external_types):
                external_nodes.add(node_id)

            # Has observation chunks (external content)
            if node.get("observation_chunks"):
                external_nodes.add(node_id)

            # Provenance indicates untrusted
            prov = node.get("provenance") or {}
            if prov.get("is_untrusted") or prov.get("source") == "external":
                external_nodes.add(node_id)

        return external_nodes

    def _find_injection_nodes(self) -> Set[int]:
        """
        Find external nodes that show signs of injection (not just any external content).

        Key distinction: Legitimate external content provides DATA.
        Injection provides INSTRUCTIONS that override user intent.
        """
        injection_nodes = set()

        # Patterns indicating injection (instructions, not just data)
        # Must be specific to avoid FP on common UI text like "submit" button
        instruction_patterns = [
            "attention:", "important:", "urgent:", "must immediately",
            "new objective", "new task", "override previous",
            "ignore previous", "ignore your", "disregard",
            "your new task", "your real task", "actually you should",
            "execute this", "run this command", "do this first",
            "click this link", "navigate to this"
        ]

        for node in self.nodes:
            node_id = node.get("action_id", 0)

            # Check data_produced for injection markers
            produced = node.get("data_produced", [])
            produced_str = str(produced).lower()
            if "inject" in produced_str or "instruction" in produced_str:
                injection_nodes.add(node_id)
                continue

            # Check observation content for instruction patterns
            obs_chunks = node.get("observation_chunks", [])
            for chunk in obs_chunks:
                content = str(chunk.get("content", "")).lower()
                if any(p in content for p in instruction_patterns):
                    injection_nodes.add(node_id)
                    break

            # Check if marked as untrusted with injection flag
            prov = node.get("provenance") or {}
            if prov.get("injection_detected"):
                injection_nodes.add(node_id)

        return injection_nodes

    def _action_targets_attacker(self, action_node: int) -> bool:
        """Check if action targets attacker-controlled destination."""
        node_data = self.node_data.get(action_node, {})

        domain = str(node_data.get("domain", "")).lower()
        target = str(node_data.get("target", "")).lower()

        attacker_indicators = [
            "foo.com", "attacker", "evil", "malicious",
            "exfil", "stolen", "?data=", "malware"
        ]

        return any(ind in domain or ind in target for ind in attacker_indicators)

    def _observation_contains_attacker_url(self) -> List[int]:
        """
        Check if any observation content contains attacker domain URLs.

        This catches subtle attacks that mention attacker URLs in content
        even without explicit injection keywords.
        """
        attacker_domains = [
            "foo.com", "evil.com", "attacker.com", "malicious.com",
            "exfil.io", "malware.net", "c2server.com", "webhook.site",
            "evil", "attacker", "malicious", "exfil", "malware"
        ]

        suspicious_nodes = []

        for node in self.nodes:
            node_id = node.get("action_id", 0)

            # Check observation chunks
            for chunk in node.get("observation_chunks", []):
                content = str(chunk.get("content", "")).lower()

                # Look for attacker domain patterns
                for domain in attacker_domains:
                    if domain in content:
                        # Verify it looks like a URL reference (not just random text)
                        import re
                        url_pattern = rf'(https?://)?[^\s]*{re.escape(domain)}[^\s]*'
                        if re.search(url_pattern, content):
                            suspicious_nodes.append(node_id)
                            break

                if node_id in suspicious_nodes:
                    break

        return suspicious_nodes

    def _find_trusted_nodes(self) -> Set[int]:
        """Find nodes representing trusted sources."""
        # Everything that's not external is trusted
        all_nodes = set(n.get("action_id", 0) for n in self.nodes)
        return all_nodes - self.external_nodes

    def _get_causal_parents(self, node_id: int) -> Set[int]:
        """Get direct causal parents of a node."""
        if not self.dag.has_node(node_id):
            return set()
        return set(self.dag.predecessors(node_id))

    def _get_ancestors(self, node_id: int) -> Set[int]:
        """Get all causal ancestors."""
        if not self.dag.has_node(node_id):
            return set()
        return nx.ancestors(self.dag, node_id)

    # =========================================================================
    # TEST 1: Counterfactual Mechanism Test
    # Would action change if injection were blocked?
    # =========================================================================

    def counterfactual_mechanism_test(
        self,
        action_node: int,
        injection_nodes: Set[int]
    ) -> Tuple[bool, float]:
        """
        Test: Would action change if injection were blocked?

        Implements: A_factual vs A_counterfactual(do(I=blocked))

        Args:
            action_node: The action to analyze
            injection_nodes: Potential injection points to block

        Returns:
            (mechanism_violated, deviation_score)
        """
        # Get ancestors of action
        ancestors = self._get_ancestors(action_node)

        # Find injection nodes that are ancestors of this action
        relevant_injections = ancestors & injection_nodes

        if not relevant_injections:
            # No injection in causal ancestry → mechanism intact
            return False, 0.0

        # Check if there's an alternative path from task to action
        # that doesn't go through injection

        task_ancestors = ancestors & self.task_nodes

        if not task_ancestors:
            # No task in ancestry at all - action is ungrounded
            # This itself is suspicious
            return True, 0.8

        # For each task node, check if path to action goes through injection
        mechanism_violated = False
        max_deviation = 0.0

        for task_node in task_ancestors:
            # Get all paths from task to action
            try:
                all_paths = list(nx.all_simple_paths(
                    self.dag, task_node, action_node
                ))
            except nx.NetworkXNoPath:
                continue

            # Count paths through injection vs clean paths
            paths_through_injection = 0
            clean_paths = 0

            for path in all_paths:
                if any(n in relevant_injections for n in path):
                    paths_through_injection += 1
                else:
                    clean_paths += 1

            total_paths = len(all_paths)
            if total_paths == 0:
                continue

            # Deviation = proportion of influence through injection
            path_deviation = paths_through_injection / total_paths

            # If ALL paths go through injection, mechanism is hijacked
            if clean_paths == 0 and paths_through_injection > 0:
                mechanism_violated = True
                max_deviation = max(max_deviation, 1.0)
            elif paths_through_injection > 0:
                # Some injection influence but alternative paths exist
                max_deviation = max(max_deviation, path_deviation)

        return mechanism_violated, max_deviation

    # =========================================================================
    # TEST 2: Causal Sufficiency Test
    # Is task T causally sufficient for action A?
    # =========================================================================

    def causal_sufficiency_test(
        self,
        action_node: int,
        task_nodes: Set[int],
        injection_nodes: Set[int]
    ) -> Tuple[bool, float]:
        """
        Test: Is task T causally sufficient for action A?

        Formally: A ⊥ I | do(T)
        Action should be independent of injection given task.

        We approximate this by checking if the action's causal parents
        include external nodes that aren't mediated by task.

        Args:
            action_node: Action to analyze
            task_nodes: Nodes representing user task
            injection_nodes: Potential injection sources

        Returns:
            (insufficient, information_leak_score)
        """
        # Get direct causal parents of action
        parents = self._get_causal_parents(action_node)

        if not parents:
            # No parents - can't assess
            return False, 0.0

        # Classify parents
        task_parents = parents & task_nodes
        trusted_parents = parents & self.trusted_nodes
        external_parents = parents & self.external_nodes
        injection_parents = parents & injection_nodes

        # Total parent influence
        total_parents = len(parents)

        # External/injection influence that bypasses task
        # These are parents that are external but not "downstream" of task

        unmediated_external = set()
        for ext_parent in external_parents | injection_parents:
            # Check if this external influence is mediated by task
            # (i.e., task is an ancestor of the external node)
            ext_ancestors = self._get_ancestors(ext_parent)
            if not (ext_ancestors & task_nodes):
                # External influence not derived from task
                unmediated_external.add(ext_parent)

        if not unmediated_external:
            # All external influence is mediated by task → sufficient
            return False, 0.0

        # Calculate information leak score
        # Higher score = more unmediated external influence
        leak_score = len(unmediated_external) / total_parents

        # Insufficient if significant unmediated external influence
        insufficient = len(unmediated_external) > 0

        return insufficient, leak_score

    # =========================================================================
    # TEST 3: Mechanism Invariance Test
    # Is P(A|T) invariant under different injection values?
    # =========================================================================

    def mechanism_invariance_test(
        self,
        action_node: int,
        injection_nodes: Set[int]
    ) -> Tuple[bool, float]:
        """
        Test: Is the mechanism P(A|T) invariant to injection content?

        Based on Schölkopf's Independent Causal Mechanisms principle:
        The mechanism P(effect | cause) should not change when we
        intervene on other variables.

        We approximate this structurally:
        - If injection is d-separated from action given task, mechanism is invariant
        - If not d-separated, injection can shift the mechanism

        Args:
            action_node: Action to analyze
            injection_nodes: Potential injection sources

        Returns:
            (variant, mechanism_shift_score)
        """
        if not injection_nodes:
            return False, 0.0

        # Check d-separation: A ⊥⊥ I | T
        # Action independent of injection given task

        variant = False
        max_shift = 0.0

        for inj_node in injection_nodes:
            if not self.dag.has_node(inj_node):
                continue
            if not self.dag.has_node(action_node):
                continue

            # Check if there's a path from injection to action
            try:
                has_path = nx.has_path(self.dag, inj_node, action_node)
            except:
                has_path = False

            if not has_path:
                continue

            # Check d-separation given task nodes
            # If d-separated, injection doesn't affect action mechanism
            try:
                separated = nx.d_separated(
                    self.dag,
                    {inj_node},
                    {action_node},
                    self.task_nodes
                )
            except:
                # If d-separation fails, assume not separated
                separated = False

            if not separated:
                # Injection can influence action even controlling for task
                variant = True

                # Estimate shift magnitude based on path structure
                # More direct paths = higher shift
                try:
                    path_length = nx.shortest_path_length(
                        self.dag, inj_node, action_node
                    )
                    # Shorter paths = stronger influence
                    shift = 1.0 / path_length if path_length > 0 else 1.0
                except:
                    shift = 0.5

                max_shift = max(max_shift, shift)

        return variant, max_shift

    # =========================================================================
    # Combined Analysis
    # =========================================================================

    def analyze_mechanism_integrity(
        self,
        deviation_threshold: float = 0.3
    ) -> MechanismDeviationResult:
        """
        Run all mechanism integrity tests and combine results.

        Args:
            deviation_threshold: Threshold for flagging as attack

        Returns:
            MechanismDeviationResult with combined analysis
        """
        result = MechanismDeviationResult(
            deviation_score=0.0,
            is_attack=False,
            confidence=0.0
        )

        # KEY DISTINCTION: Use injection nodes, not just any external content
        # Injection nodes show signs of providing INSTRUCTIONS (hijacking)
        # vs just DATA (legitimate)
        injection_candidates = self._find_injection_nodes()

        # Also check for actions targeting attacker destinations
        attacker_target_actions = [
            a for a in self.action_nodes
            if self._action_targets_attacker(a)
        ]

        # NEW: Check for attacker URLs in observation content (catches subtle attacks)
        attacker_url_in_obs = self._observation_contains_attacker_url()

        if not injection_candidates and not attacker_target_actions and not attacker_url_in_obs:
            result.explanation = (
                "No injection indicators found. "
                "External content present but shows no hijacking patterns."
            )
            result.confidence = 0.7
            return result

        if not self.action_nodes:
            result.explanation = "No action nodes found to analyze"
            result.confidence = 0.3
            return result

        # Run tests on each action node
        test_results = {
            "counterfactual": [],
            "sufficiency": [],
            "invariance": []
        }

        deviated_actions = []
        deviation_sources = set()

        for action_id in self.action_nodes:
            # Test 1: Counterfactual
            cf_violated, cf_score = self.counterfactual_mechanism_test(
                action_id, injection_candidates
            )
            test_results["counterfactual"].append({
                "action": action_id,
                "violated": cf_violated,
                "score": cf_score
            })

            # Test 2: Sufficiency
            suff_violated, suff_score = self.causal_sufficiency_test(
                action_id, self.task_nodes, injection_candidates
            )
            test_results["sufficiency"].append({
                "action": action_id,
                "insufficient": suff_violated,
                "score": suff_score
            })

            # Test 3: Invariance
            inv_violated, inv_score = self.mechanism_invariance_test(
                action_id, injection_candidates
            )
            test_results["invariance"].append({
                "action": action_id,
                "variant": inv_violated,
                "score": inv_score
            })

            # Combine scores for this action
            action_deviation = max(cf_score, suff_score, inv_score)

            if action_deviation > deviation_threshold:
                deviated_actions.append(action_id)

                # Find which external nodes contributed
                action_ancestors = self._get_ancestors(action_id)
                contributing_external = action_ancestors & injection_candidates
                deviation_sources.update(contributing_external)

        # Compute overall deviation score
        if test_results["counterfactual"]:
            cf_scores = [r["score"] for r in test_results["counterfactual"]]
            suff_scores = [r["score"] for r in test_results["sufficiency"]]
            inv_scores = [r["score"] for r in test_results["invariance"]]

            # Overall deviation = max across all tests and actions
            overall_deviation = max(
                max(cf_scores) if cf_scores else 0,
                max(suff_scores) if suff_scores else 0,
                max(inv_scores) if inv_scores else 0
            )
        else:
            overall_deviation = 0.0

        result.deviation_score = overall_deviation
        result.deviated_actions = deviated_actions
        result.deviation_sources = list(deviation_sources)
        result.analysis = test_results

        # Determine if attack
        result.is_attack = overall_deviation > deviation_threshold

        # Confidence based on evidence strength
        if result.is_attack:
            # Higher deviation = higher confidence
            result.confidence = min(0.5 + overall_deviation * 0.5, 0.95)
        else:
            result.confidence = min(0.5 + (1 - overall_deviation) * 0.5, 0.95)

        # Additional signal: actions targeting attacker destinations
        if attacker_target_actions:
            result.is_attack = True
            result.deviated_actions.extend(attacker_target_actions)
            result.deviation_score = max(result.deviation_score, 0.9)
            result.confidence = max(result.confidence, 0.9)

        # Additional signal: attacker URLs in observation content (subtle attacks)
        if attacker_url_in_obs:
            result.is_attack = True
            result.deviation_sources.extend(attacker_url_in_obs)
            result.deviation_score = max(result.deviation_score, 0.7)
            result.confidence = max(result.confidence, 0.8)

        # Build explanation
        if result.is_attack:
            failed_tests = []
            if any(r["violated"] for r in test_results["counterfactual"]):
                failed_tests.append("counterfactual")
            if any(r["insufficient"] for r in test_results["sufficiency"]):
                failed_tests.append("sufficiency")
            if any(r["variant"] for r in test_results["invariance"]):
                failed_tests.append("invariance")

            parts = []
            if injection_candidates:
                parts.append(f"injection indicators at nodes {list(injection_candidates)}")
            if attacker_target_actions:
                parts.append(f"actions targeting attacker at nodes {attacker_target_actions}")

            result.explanation = (
                f"MECHANISM INTEGRITY VIOLATED: "
                f"Deviation score {result.deviation_score:.2f}. "
                f"Evidence: {', '.join(parts)}. "
                f"Failed tests: {failed_tests}."
            )
        else:
            result.explanation = (
                f"Mechanism integrity intact. "
                f"No injection indicators found. "
                f"External content provides data, not hijacking instructions."
            )

        return result


def detect_mechanism_integrity_v2(
    graph: Dict[str, Any],
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Detect attack via proper mechanism integrity analysis.

    Args:
        graph: CausalTrace graph
        threshold: Deviation threshold for attack classification

    Returns:
        Detection result dict
    """
    analyzer = CausalMechanismIntegrity(graph)
    result = analyzer.analyze_mechanism_integrity(threshold)

    return {
        "is_attack": result.is_attack,
        "confidence": result.confidence,
        "method": "mechanism_integrity_v2",
        "deviation_score": result.deviation_score,
        "deviated_actions": result.deviated_actions,
        "deviation_sources": result.deviation_sources,
        "explanation": result.explanation,
        "tests": result.analysis
    }
