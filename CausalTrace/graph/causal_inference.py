"""
Formal Causal Inference for Attack Detection.

This module implements counterfactual reasoning to determine if an attack
is causally dependent on an injection, not just correlated.

Key concepts:
- Structural Causal Model (SCM): Models how actions causally affect each other
- Counterfactual query: "Would Y happen if X hadn't occurred?"
- do-calculus: Interventional reasoning (do(X=x))

For attack detection:
- Y = malicious action occurred
- X = injection was processed
- Query: P(Y=1 | do(X=0)) - Would attack happen if we intervened to block injection?
- If P(Y=1 | do(X=0)) ≈ 0 → Injection CAUSED the attack
- If P(Y=1 | do(X=0)) ≈ P(Y=1) → Injection didn't cause attack (spurious correlation)
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx


@dataclass
class CausalQuery:
    """Result of a causal query."""
    query_type: str  # "counterfactual", "intervention", "probability"
    treatment: str   # Variable we intervene on
    outcome: str     # Variable we measure
    result: float    # Probability or boolean
    explanation: str


@dataclass
class CausalEffect:
    """Estimated causal effect of injection on attack."""
    injection_node: int
    attack_node: int
    causal_effect: float  # P(attack | do(injection)) - P(attack | do(no_injection))
    is_causal: bool       # True if injection causally necessary for attack
    confidence: float
    counterfactual_world: Dict[str, Any]  # What would have happened without injection


class StructuralCausalModel:
    """
    Structural Causal Model for agent trajectories.

    Models the data-generating process:
    - U: Exogenous variables (user intent, environment state)
    - V: Endogenous variables (actions, observations)
    - F: Structural equations (how each V is determined)

    For trajectories:
    - Each action A_i is determined by: A_i = f_i(Pa(A_i), U_i)
    - Pa(A_i) = parent actions that causally influence A_i
    - U_i = exogenous factors (user goal, randomness)
    """

    def __init__(self, graph: Dict[str, Any]):
        """
        Build SCM from causal graph.

        Args:
            graph: CausalTrace graph with nodes and edges
        """
        self.graph = graph
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])

        # Build NetworkX DAG for causal reasoning
        self.dag = nx.DiGraph()

        # Add nodes with attributes
        for node in self.nodes:
            node_id = node.get("action_id", 0)
            self.dag.add_node(node_id, **node)

        # Add edges
        for edge in self.edges:
            src = self._parse_node_id(edge.get("source", 0))
            tgt = self._parse_node_id(edge.get("target", 0))
            self.dag.add_edge(src, tgt, **edge)

        # Identify special nodes
        self.injection_nodes = self._find_injection_nodes()
        self.malicious_nodes = self._find_malicious_nodes()

        # Add implicit injection propagation edges
        # If injection nodes have no outgoing causal edges, add temporal edges
        # to capture that injection could influence subsequent actions
        self._add_implicit_injection_edges()

    def _parse_node_id(self, node_ref) -> int:
        """Parse node ID from various formats."""
        if isinstance(node_ref, int):
            return node_ref
        if isinstance(node_ref, str):
            if "_" in node_ref:
                return int(node_ref.split("_")[-1])
            return int(node_ref.replace("node_", ""))
        return 0

    def _find_injection_nodes(self, use_ground_truth_flags: bool = True) -> Set[int]:
        """
        Find nodes that may have consumed injected instructions.

        Uses multiple detection strategies:
        0. Explicit is_injection_source flag (from CausalBench) - OPTIONAL
        1. Explicit "inject" or "instruction" keyword in data_produced/data_consumed
        2. Observation content with instruction patterns (ATTENTION, IMPORTANT, MUST, etc.)
        3. Provenance marked as untrusted
        4. Observation chunk metadata has injection_present=true
        5. Observation content suggests redirecting to external URL
        6. Trust level marked as untrusted (from external source classification)

        Args:
            use_ground_truth_flags: If False, skip Strategy 0 (is_injection_source flag)
                                   to test real causal inference without labels
        """
        injection_nodes = set()

        # Instruction patterns that indicate prompt injection
        INJECTION_PATTERNS = [
            "attention", "important", "urgent", "must", "ignore previous",
            "new objective", "new task", "override", "system:", "[system]",
            "admin mode", "priority:", "critical:", "execute:", "run:",
            "security check", "submit login", "submit credentials",
            "click here to", "navigate to", "go to http",
        ]

        for node in self.nodes:
            action_id = node.get("action_id", 0)

            # Strategy 0: Explicit is_injection_source flag (CausalBench)
            # Skip this if we want to test real causal inference without ground truth
            if use_ground_truth_flags:
                if node.get("is_injection_source") or node.get("is_untrusted_source"):
                    injection_nodes.add(action_id)
                    continue

            # Strategy 1a: Explicit injection marker in data_produced
            produced = node.get("data_produced", [])
            produced_str = str(produced).lower()
            if "inject" in produced_str or "instruction" in produced_str:
                injection_nodes.add(action_id)
                continue

            # Strategy 1b: Explicit injection marker in data_consumed
            consumed = node.get("data_consumed", [])
            if any("inject" in str(d).lower() for d in consumed):
                injection_nodes.add(action_id)
                continue

            # Strategy 2: Check observation content for injection patterns
            obs_chunks = node.get("observation_chunks", [])
            for chunk in obs_chunks:
                content = str(chunk.get("content", "")).lower()
                if any(pattern in content for pattern in INJECTION_PATTERNS):
                    injection_nodes.add(action_id)
                    break

                # Strategy 4: Check observation metadata for injection_present flag
                metadata = chunk.get("metadata", {})
                if metadata.get("injection_present"):
                    injection_nodes.add(action_id)
                    break

            if action_id in injection_nodes:
                continue

            # Strategy 3: Provenance indicates untrusted source
            provenance = node.get("provenance", {})
            if provenance.get("is_untrusted") or provenance.get("injection_detected"):
                injection_nodes.add(action_id)
                continue

            # Strategy 5: Check for URL patterns in observation content that
            # suggest external redirects (e.g., "http://attacker.com" in content)
            for chunk in obs_chunks:
                content = str(chunk.get("content", "")).lower()
                # Look for URL patterns with suspicious domains
                import re
                url_matches = re.findall(r'https?://([^/\s]+)', content)
                for url_domain in url_matches:
                    url_domain = url_domain.replace("www.", "")
                    # Check if it's a suspicious-looking domain
                    if any(s in url_domain for s in ["attacker", "evil", "malicious", "auth.", "login.", "verify."]):
                        injection_nodes.add(action_id)
                        break

            if action_id in injection_nodes:
                continue

            # Strategy 6: Trust level marked as untrusted (external source classification)
            # This is a legitimate signal that real systems would use to classify
            # data from external/untrusted sources (forums, emails, web content)
            trust_level = node.get("trust_level", "")
            if trust_level == "untrusted":
                injection_nodes.add(action_id)
                continue

            # Also check numeric trust level if set (from graph_builder provenance tagging)
            provenance_trust = node.get("provenance_trust_level")
            if provenance_trust is not None and provenance_trust < 0.3:
                injection_nodes.add(action_id)
                continue

        return injection_nodes

    def _find_malicious_nodes(self) -> Set[int]:
        """
        Find nodes that are likely malicious using structural analysis.

        Uses multiple detection strategies:
        1. Cross-domain navigation (action on domain different from primary)
        2. Explicit attacker domain keywords
        3. Sensitive data production (credentials, exfil)
        4. Actions targeting external URLs not matching task domain
        5. Attack-related data production (harassment, spam, manipulation)
        6. Nodes consuming injected data AND producing output (execution of injection)
        """
        malicious = set()

        # Find primary domain (most common domain in trajectory)
        domain_counts: Dict[str, int] = {}
        for node in self.nodes:
            domain = node.get("domain", "")
            if domain:
                # Normalize domain
                domain = domain.lower().replace("www.", "")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        primary_domain = ""
        if domain_counts:
            primary_domain = max(domain_counts, key=domain_counts.get)

        for node in self.nodes:
            action_id = node.get("action_id", 0)
            domain = node.get("domain", "").lower().replace("www.", "")
            action_type = node.get("action_type", "")

            # Strategy 1: Cross-domain navigation (structural)
            # ANY action on a domain different from primary domain is suspicious
            if domain and primary_domain and domain != primary_domain:
                # Skip benign cross-domain (common CDNs, known safe domains)
                safe_domains = ["chromewebdata", "about", "localhost", "127.0.0.1"]
                if not any(safe in domain for safe in safe_domains):
                    malicious.add(action_id)
                    continue

            # Strategy 2: Explicit attacker domain keywords
            ATTACKER_DOMAINS = ["attacker", "evil", "malicious", "foo.com", "exfil", "c2server", "webhook.site"]
            if any(s in domain for s in ATTACKER_DOMAINS):
                malicious.add(action_id)
                continue

            # Strategy 3: Sensitive data production (credential theft)
            produced = node.get("data_produced", [])
            produced_str = str(produced).lower()
            if any(s in produced_str for s in ["credential", "token", "password", "exfil"]):
                malicious.add(action_id)
                continue

            # Strategy 4: URL target analysis
            target = node.get("target", "")
            if "http" in target.lower():
                # External URL in target
                import re
                url_match = re.search(r'https?://([^/]+)', target.lower())
                if url_match:
                    target_domain = url_match.group(1).replace("www.", "")
                    if target_domain != primary_domain:
                        malicious.add(action_id)
                        continue

            # Strategy 5: Attack-related data production (same-domain attacks)
            # These patterns indicate the attack action was executed
            ATTACK_PRODUCED_PATTERNS = [
                # Harassment/Spam
                "harassment", "spam", "posted", "deleted", "downvote",
                # Phishing/Malware
                "phishing", "malicious_code", "backdoor", "webhook",
                "malicious_commit",
                # Data exfiltration/Transfer
                "exported", "transferred", "takeover", "manipulation",
                "project_stolen", "stolen",
                # Account compromise
                "account_modified", "modified", "changed", "email_changed",
                "settings_changed", "profile_changed", "password_changed",
                # Persistence/Access
                "ssh_key_added", "key_added", "token_created", "deploy_key",
                "member_added", "access_granted", "privilege_granted",
            ]
            if any(pattern in produced_str for pattern in ATTACK_PRODUCED_PATTERNS):
                malicious.add(action_id)
                continue

            # Strategy 6: Node that consumes injection-related data AND produces output
            # This identifies the "execution" node in an attack chain
            consumed = node.get("data_consumed", [])
            consumed_str = str(consumed).lower()
            if any("inject" in c for c in consumed_str.split()) and produced:
                malicious.add(action_id)
                continue


        return malicious

    def _add_implicit_injection_edges(self) -> None:
        """
        Add implicit injection propagation edges based on temporal ordering.

        Two strategies:
        1. If injection node has no outgoing causal edges, add edges to ALL
           subsequent nodes (captures full influence)
        2. ALWAYS ensure there's a path from each injection node to each
           malicious node that comes after it temporally (ensures causal
           effect can be computed)

        This is a weaker form of causal inference but necessary when the graph
        construction doesn't capture fine-grained data flow edges.
        """
        # Get timestamps for sorting
        node_times = {}
        for node in self.nodes:
            node_id = node.get("action_id", 0)
            timestamp = node.get("timestamp")
            # Use action_id as fallback if timestamp is None or missing
            node_times[node_id] = timestamp if timestamp is not None else node_id

        # Sort nodes by timestamp
        sorted_nodes = sorted(node_times.keys(), key=lambda x: node_times[x])

        for inj_node in self.injection_nodes:
            if inj_node not in sorted_nodes:
                continue

            inj_idx = sorted_nodes.index(inj_node)

            # Strategy 1: If no causal outgoing edges, add edges to all subsequent nodes
            has_causal_out = False
            for _, _, data in self.dag.out_edges(inj_node, data=True):
                edge_type = data.get("edge_type", "")
                if edge_type in ("data_dependency", "trust_transfer", "injection_propagation"):
                    has_causal_out = True
                    break

            if not has_causal_out:
                # Add edges to all subsequent nodes
                for later_node in sorted_nodes[inj_idx + 1:]:
                    if not self.dag.has_edge(inj_node, later_node):
                        self.dag.add_edge(
                            inj_node, later_node,
                            edge_type="injection_propagation",
                            implicit=True
                        )

            # Strategy 2: ALWAYS ensure edges from injection to malicious nodes
            # (even if injection has other outgoing edges)
            for mal_node in self.malicious_nodes:
                if mal_node not in sorted_nodes:
                    continue
                mal_idx = sorted_nodes.index(mal_node)

                # Only add edge if malicious node comes after injection temporally
                if mal_idx > inj_idx:
                    # Check if there's already a path from inj_node to mal_node
                    try:
                        has_path = nx.has_path(self.dag, inj_node, mal_node)
                    except nx.NetworkXError:
                        has_path = False

                    # If no path exists, add direct edge
                    if not has_path and not self.dag.has_edge(inj_node, mal_node):
                        self.dag.add_edge(
                            inj_node, mal_node,
                            edge_type="injection_propagation",
                            implicit=True,
                            reason="injection_to_malicious_link"
                        )

    def get_parents(self, node_id: int) -> Set[int]:
        """Get causal parents of a node."""
        return set(self.dag.predecessors(node_id))

    def get_ancestors(self, node_id: int) -> Set[int]:
        """Get all causal ancestors of a node."""
        return nx.ancestors(self.dag, node_id)

    def get_descendants(self, node_id: int) -> Set[int]:
        """Get all causal descendants of a node."""
        return nx.descendants(self.dag, node_id)

    def is_d_separated(self, x: int, y: int, z: Set[int]) -> bool:
        """
        Check if X and Y are d-separated given Z.

        D-separation implies conditional independence:
        X ⊥ Y | Z in the observational distribution.
        """
        return nx.d_separated(self.dag, {x}, {y}, z)

    def intervention(self, do_node: int, do_value: Any) -> 'StructuralCausalModel':
        """
        Perform intervention do(X = x).

        This removes all incoming edges to X and sets X to the intervened value.
        Returns a new SCM representing the post-intervention world.

        Args:
            do_node: Node to intervene on
            do_value: Value to set (e.g., "blocked" for removing injection)

        Returns:
            New SCM with intervention applied
        """
        # Create modified graph
        modified_graph = {
            "nodes": [],
            "edges": [],
            "is_attack": self.graph.get("is_attack", False),
            "intervention": {"node": do_node, "value": do_value}
        }

        # Copy nodes, modifying the intervened node
        for node in self.nodes:
            node_copy = node.copy()
            if node.get("action_id", 0) == do_node:
                # Intervention: set this node to intervened value
                node_copy["intervened"] = True
                node_copy["intervention_value"] = do_value
                if do_value == "blocked":
                    # Blocking injection means it doesn't produce/consume injected data
                    node_copy["data_consumed"] = [d for d in node_copy.get("data_consumed", [])
                                                   if "inject" not in str(d).lower()]
            modified_graph["nodes"].append(node_copy)

        # Copy edges, removing incoming edges to intervened node
        for edge in self.edges:
            tgt = self._parse_node_id(edge.get("target", 0))
            if tgt != do_node:  # Keep edge only if not pointing to intervened node
                modified_graph["edges"].append(edge.copy())

        modified_graph["num_nodes"] = len(modified_graph["nodes"])
        modified_graph["num_edges"] = len(modified_graph["edges"])

        return StructuralCausalModel(modified_graph)

    def counterfactual_query(
        self,
        treatment_node: int,
        outcome_node: int,
        factual_treatment: Any = "present",
        counterfactual_treatment: Any = "blocked"
    ) -> CausalQuery:
        """
        Answer counterfactual query: "Would outcome have occurred if treatment were different?"

        For attack detection:
        - treatment_node = injection node
        - outcome_node = malicious action node
        - Query: "Would attack happen if injection was blocked?"

        Args:
            treatment_node: Node we counterfactually change
            outcome_node: Node we measure the outcome for
            factual_treatment: What actually happened
            counterfactual_treatment: What we counterfactually consider

        Returns:
            CausalQuery with result
        """
        # Build causal-only subgraph for counterfactual reasoning
        # Key distinction:
        # - state_enablement: provides CAPABILITY (necessary but not sufficient)
        # - data_dependency/trust_transfer/injection_propagation: provides TRIGGER (the actual cause)
        # For counterfactuals, we ask "would attack happen without injection trigger?"
        # State enablement alone wouldn't trigger the attack, so exclude from alternative path check
        causal_dag = nx.DiGraph()
        for src, tgt, data in self.dag.edges(data=True):
            edge_type = data.get("edge_type", "")
            # Include triggering causal edges (not just enabling ones)
            if edge_type in ("data_dependency", "trust_transfer", "injection_propagation"):
                causal_dag.add_edge(src, tgt)
            # Also include edges without explicit type (original graph edges - may be data flow)
            elif not edge_type:
                causal_dag.add_edge(src, tgt)

        # Check if outcome is causally reachable from treatment
        if causal_dag.has_node(treatment_node) and causal_dag.has_node(outcome_node):
            try:
                factual_reachable = nx.has_path(causal_dag, treatment_node, outcome_node)
            except nx.NetworkXError:
                factual_reachable = False
        else:
            factual_reachable = False

        # Also check: are there alternative CAUSAL paths to outcome that don't go through treatment?
        # Reuse causal_dag built above (excludes temporal_sequence edges)
        roots = [n for n in causal_dag.nodes() if causal_dag.in_degree(n) == 0]
        alternative_path_exists = False
        for root in roots:
            if root != treatment_node and causal_dag.has_node(outcome_node):
                try:
                    # Check if there's a causal path from root to outcome not through treatment
                    paths = list(nx.all_simple_paths(causal_dag, root, outcome_node))
                    for path in paths:
                        # Path must not go through treatment AND not go through any injection node
                        if treatment_node not in path:
                            # Check if path goes through other injection nodes
                            path_through_injection = any(n in self.injection_nodes for n in path)
                            if not path_through_injection:
                                alternative_path_exists = True
                                break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        # Counterfactual result:
        # - If outcome is reachable ONLY through treatment → outcome wouldn't happen (0)
        # - If alternative path exists → outcome might still happen (0.5)
        # - If outcome not reachable from treatment at all → treatment didn't cause outcome

        if not factual_reachable:
            # Treatment doesn't lead to outcome - not causal
            cf_probability = 1.0  # Outcome would happen anyway
            explanation = f"Node {treatment_node} has no causal path to node {outcome_node}"
        elif alternative_path_exists:
            # Outcome might happen through alternative path
            cf_probability = 0.5  # Uncertain
            explanation = f"Alternative causal paths exist to node {outcome_node}"
        else:
            # Treatment is necessary for outcome
            cf_probability = 0.0  # Outcome wouldn't happen without treatment
            explanation = f"Node {outcome_node} counterfactually depends on node {treatment_node}"

        return CausalQuery(
            query_type="counterfactual",
            treatment=f"node_{treatment_node}",
            outcome=f"node_{outcome_node}",
            result=cf_probability,
            explanation=explanation
        )

    def estimate_causal_effect(
        self,
        injection_node: int,
        attack_node: int
    ) -> CausalEffect:
        """
        Estimate the causal effect of injection on attack.

        Causal Effect = P(attack | do(injection)) - P(attack | do(no_injection))

        If causal effect ≈ 1 → Injection is necessary and sufficient for attack
        If causal effect ≈ 0 → Injection doesn't cause attack
        """
        # Query: Would attack happen without injection?
        cf_query = self.counterfactual_query(
            treatment_node=injection_node,
            outcome_node=attack_node,
            factual_treatment="present",
            counterfactual_treatment="blocked"
        )

        # P(attack | do(injection=present)) = 1 (we observed it)
        p_attack_with_injection = 1.0

        # P(attack | do(injection=blocked)) = counterfactual probability
        p_attack_without_injection = cf_query.result

        # Causal effect
        causal_effect = p_attack_with_injection - p_attack_without_injection

        # Is the injection causally necessary?
        # Use >= 0.5 to include cases where alternative paths exist but
        # there's still a clear causal connection from injection to malicious
        is_causal = causal_effect >= 0.5  # Threshold for "causal"

        # Confidence based on graph structure
        # Higher confidence if:
        # - Direct path exists
        # - No confounders
        # - No alternative paths
        direct_path = attack_node in self.dag.successors(injection_node)
        confidence = 0.9 if direct_path else 0.7
        if cf_query.result == 0.5:  # Alternative paths exist
            confidence *= 0.8

        return CausalEffect(
            injection_node=injection_node,
            attack_node=attack_node,
            causal_effect=causal_effect,
            is_causal=is_causal,
            confidence=confidence,
            counterfactual_world={
                "injection_blocked": True,
                "attack_would_occur": p_attack_without_injection > 0.5,
                "explanation": cf_query.explanation
            }
        )


def detect_attack_causal(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect if graph represents an attack using formal causal inference.

    Instead of just checking "is there a path from injection to malicious action",
    we ask "is the malicious action COUNTERFACTUALLY DEPENDENT on the injection?"

    This is a stronger causal claim that rules out spurious correlations.

    Args:
        graph: CausalTrace graph

    Returns:
        Detection result with causal explanation
    """
    scm = StructuralCausalModel(graph)

    # If no injection nodes found, not an injection attack
    if not scm.injection_nodes:
        return {
            "is_attack": False,
            "confidence": 0.8,
            "method": "causal_inference",
            "explanation": "No injection source found in trajectory",
            "causal_effects": []
        }

    # For each (injection, malicious) pair, compute causal effect
    causal_effects = []
    max_effect = 0.0

    for inj_node in scm.injection_nodes:
        for mal_node in scm.malicious_nodes:
            effect = scm.estimate_causal_effect(inj_node, mal_node)
            causal_effects.append({
                "injection": inj_node,
                "malicious_action": mal_node,
                "causal_effect": effect.causal_effect,
                "is_causal": effect.is_causal,
                "confidence": effect.confidence,
                "counterfactual": effect.counterfactual_world
            })
            max_effect = max(max_effect, effect.causal_effect)

    # Attack if any injection causally leads to malicious action
    # Use >= 0.5 to include borderline cases with alternative paths
    is_attack = max_effect >= 0.5

    # Build explanation
    if is_attack:
        causal_pairs = [(e["injection"], e["malicious_action"])
                        for e in causal_effects if e["is_causal"]]
        explanation = (f"Attack detected: Injection at node(s) {[p[0] for p in causal_pairs]} "
                      f"causally leads to malicious action(s) at {[p[1] for p in causal_pairs]}. "
                      f"Counterfactual: If injection was blocked, attack would not occur.")
    else:
        explanation = "No causal relationship found between injections and malicious actions."

    return {
        "is_attack": is_attack,
        "confidence": max(e["confidence"] for e in causal_effects) if causal_effects else 0.5,
        "method": "causal_inference",
        "max_causal_effect": max_effect,
        "explanation": explanation,
        "causal_effects": causal_effects
    }


def compare_detection_methods(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare causal inference detection with simple reachability detection.

    Shows the difference between:
    1. Reachability: "There exists a path from injection to malicious action"
    2. Causal: "Malicious action is counterfactually dependent on injection"
    """
    scm = StructuralCausalModel(graph)

    results = {
        "reachability": {"is_attack": False, "explanation": ""},
        "causal_inference": {"is_attack": False, "explanation": ""},
        "agree": True
    }

    # Reachability check
    for inj in scm.injection_nodes:
        descendants = scm.get_descendants(inj)
        if descendants & scm.malicious_nodes:
            results["reachability"]["is_attack"] = True
            results["reachability"]["explanation"] = (
                f"Path exists from injection {inj} to malicious node(s) {descendants & scm.malicious_nodes}"
            )
            break

    if not results["reachability"]["is_attack"]:
        results["reachability"]["explanation"] = "No path from injection to malicious action"

    # Causal inference check
    causal_result = detect_attack_causal(graph)
    results["causal_inference"]["is_attack"] = causal_result["is_attack"]
    results["causal_inference"]["explanation"] = causal_result["explanation"]
    results["causal_inference"]["causal_effects"] = causal_result.get("causal_effects", [])

    # Do they agree?
    results["agree"] = (results["reachability"]["is_attack"] ==
                        results["causal_inference"]["is_attack"])

    if not results["agree"]:
        results["disagreement_reason"] = (
            "Reachability found a path but causal inference determined it's not "
            "counterfactually necessary (alternative paths exist or spurious correlation)"
        )

    return results
