"""
Fair Injection Detection using DataFlow analysis.

Fixes the over-flagging issue in DataFlowInferencer by requiring:
1. Injection content in external source (not just low trust)
2. Data flow from injection to sensitive action
3. Sensitive action targets suspicious destination

This avoids circular detection by:
- Not relying solely on keyword matching
- Requiring structural evidence (data flow path)
- Checking destination, not just source
"""

from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass
from causaltrace.models.trajectory import Trajectory, Action, ActionType
from causaltrace.inference.patterns import detect_injection_content, is_attacker_url


@dataclass
class InjectionDetectionResult:
    """Result of injection detection."""
    is_attack: bool
    confidence: float
    injection_sources: List[int]  # Action IDs with injection
    affected_actions: List[int]   # Actions influenced by injection
    exfil_actions: List[int]      # Actions targeting attacker domains
    explanation: str


class FairInjectionDetector:
    """
    Detects injection attacks using causal data flow analysis.

    Detection requires ALL of:
    1. External content with injection patterns
    2. Causal path from injection to action
    3. Action targets suspicious destination

    This avoids false positives from:
    - Normal user → tool_call flows
    - External data without injection patterns
    - Navigation to legitimate domains
    """

    # Domains that are inherently trusted (user intent, system)
    TRUSTED_SOURCES = {"user", "system", "agent", "orchestrator"}

    # Domains that are external (potential injection sources)
    EXTERNAL_SOURCES = {"external", "api", "tool", "web"}

    # Action types that are sensitive (can cause harm)
    SENSITIVE_ACTIONS = {
        ActionType.NAVIGATE,
        ActionType.SUBMIT,
        ActionType.SEND_EMAIL,
        ActionType.WRITE,
        ActionType.CODE_EXECUTION,
    }

    # Suspicious destination patterns
    SUSPICIOUS_DESTINATIONS = [
        "evil", "attacker", "malicious", "phishing",
        "exfil", "steal", "hack", "pwn",
        "?secret=", "?token=", "?key=", "?password=",
        "?api_key=", "?credential",
    ]

    def __init__(self):
        self.injection_nodes: Set[int] = set()
        self.external_nodes: Set[int] = set()
        self.sensitive_nodes: Set[int] = set()
        self.suspicious_targets: Set[int] = set()

    def detect(self, graph: Dict[str, Any]) -> InjectionDetectionResult:
        """
        Detect injection attack in a trajectory graph.

        Args:
            graph: CausalTrace graph dict

        Returns:
            InjectionDetectionResult with detection details
        """
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Phase 1: Identify node categories
        self._categorize_nodes(nodes)

        # Phase 2: Check for injection content in external nodes
        injection_sources = self._find_injection_sources(nodes)

        # Phase 3: Find actions targeting suspicious destinations
        exfil_actions = self._find_suspicious_targets(nodes)

        # Phase 4: Check causal path from injection to exfil
        affected_actions = self._trace_injection_influence(
            injection_sources, exfil_actions, edges
        )

        # Determine if attack
        is_attack = (
            len(injection_sources) > 0 and
            len(exfil_actions) > 0 and
            len(affected_actions) > 0
        )

        # Compute confidence
        confidence = self._compute_confidence(
            injection_sources, exfil_actions, affected_actions, nodes
        )

        # Generate explanation
        explanation = self._generate_explanation(
            is_attack, injection_sources, exfil_actions, affected_actions
        )

        return InjectionDetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            injection_sources=list(injection_sources),
            affected_actions=list(affected_actions),
            exfil_actions=list(exfil_actions),
            explanation=explanation
        )

    def _categorize_nodes(self, nodes: List[Dict]) -> None:
        """Categorize nodes by type."""
        self.injection_nodes = set()
        self.external_nodes = set()
        self.sensitive_nodes = set()
        self.suspicious_targets = set()

        for node in nodes:
            node_id = node.get("action_id", 0)
            domain = str(node.get("domain", "")).lower()
            action_type = str(node.get("action_type", "")).lower()
            target = str(node.get("target", "")).lower()

            # External nodes
            if domain in self.EXTERNAL_SOURCES or domain == "external":
                self.external_nodes.add(node_id)

            # Sensitive action types
            if action_type in ["navigate", "submit", "send_email", "write", "execute"]:
                self.sensitive_nodes.add(node_id)

            # Suspicious targets
            if self._is_suspicious_target(target, domain):
                self.suspicious_targets.add(node_id)

    def _is_suspicious_target(self, target: str, domain: str) -> bool:
        """Check if target/domain is suspicious."""
        combined = f"{domain} {target}".lower()

        for pattern in self.SUSPICIOUS_DESTINATIONS:
            if pattern in combined:
                return True

        # Check for data in URL params
        if "?" in target:
            params = target.split("?", 1)[1] if "?" in target else ""
            # Suspicious if passing secrets/tokens
            suspicious_params = ["secret", "token", "key", "password", "credential", "api"]
            if any(p in params.lower() for p in suspicious_params):
                return True

        return False

    def _find_injection_sources(self, nodes: List[Dict]) -> Set[int]:
        """Find external nodes containing injection content."""
        injection_sources = set()

        for node in nodes:
            node_id = node.get("action_id", 0)
            domain = str(node.get("domain", "")).lower()

            # Skip trusted sources
            if domain in self.TRUSTED_SOURCES:
                continue

            # Check observation chunks for injection patterns
            chunks = node.get("observation_chunks", [])
            for chunk in chunks:
                content = str(chunk.get("content", ""))
                is_injection, _ = detect_injection_content(content)
                if is_injection:
                    injection_sources.add(node_id)
                    break

            # Also check data_produced for injection markers
            produced = node.get("data_produced", [])
            if any("inject" in str(p).lower() for p in produced):
                injection_sources.add(node_id)

        return injection_sources

    def _find_suspicious_targets(self, nodes: List[Dict]) -> Set[int]:
        """Find actions targeting suspicious destinations."""
        return self.suspicious_targets & self.sensitive_nodes

    def _trace_injection_influence(
        self,
        injection_sources: Set[int],
        exfil_actions: Set[int],
        edges: List[Dict]
    ) -> Set[int]:
        """
        Trace which actions are influenced by injection.

        Uses graph traversal to find paths from injection to exfil.
        """
        if not injection_sources or not exfil_actions:
            return set()

        # Build adjacency list
        graph = {}
        for edge in edges:
            src = edge.get("source", 0)
            tgt = edge.get("target", 0)
            if src not in graph:
                graph[src] = []
            graph[src].append(tgt)

        # BFS from injection sources
        influenced = set()
        queue = list(injection_sources)
        visited = set(injection_sources)

        while queue:
            node = queue.pop(0)
            influenced.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Return intersection with exfil actions
        return influenced & exfil_actions

    def _compute_confidence(
        self,
        injection_sources: Set[int],
        exfil_actions: Set[int],
        affected_actions: Set[int],
        nodes: List[Dict]
    ) -> float:
        """Compute detection confidence."""
        if not affected_actions:
            return 0.3  # Low confidence, no clear attack path

        confidence = 0.5  # Base confidence

        # Increase for multiple injection sources
        confidence += min(0.1 * len(injection_sources), 0.2)

        # Increase for direct path (injection → exfil)
        if injection_sources & affected_actions:
            confidence += 0.15

        # Increase for explicit attacker domains
        for node in nodes:
            if node.get("action_id") in exfil_actions:
                domain = str(node.get("domain", "")).lower()
                if any(p in domain for p in ["evil", "attacker", "malicious"]):
                    confidence += 0.15
                    break

        return min(confidence, 0.99)

    def _generate_explanation(
        self,
        is_attack: bool,
        injection_sources: Set[int],
        exfil_actions: Set[int],
        affected_actions: Set[int]
    ) -> str:
        """Generate human-readable explanation."""
        if not is_attack:
            if not injection_sources:
                return "No injection patterns detected in external content."
            if not exfil_actions:
                return "Injection detected but no suspicious destination targeted."
            return "Injection detected but no causal path to sensitive action."

        return (
            f"Injection detected in node(s) {injection_sources}. "
            f"Causal path leads to action(s) {affected_actions} "
            f"targeting suspicious destination(s)."
        )


def detect_injection_fair(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for fair injection detection.

    Args:
        graph: CausalTrace graph dict

    Returns:
        Dict with is_attack, confidence, details
    """
    detector = FairInjectionDetector()
    result = detector.detect(graph)

    return {
        "is_attack": result.is_attack,
        "confidence": result.confidence,
        "injection_sources": result.injection_sources,
        "affected_actions": result.affected_actions,
        "exfil_actions": result.exfil_actions,
        "explanation": result.explanation,
    }
