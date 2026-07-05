"""
Trajectory Validator for CausalBench

Validates generated trajectories for correctness, completeness, and attack labeling.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    level: ValidationLevel
    code: str
    message: str
    location: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of trajectory validation."""
    valid: bool
    trajectory_id: str
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, level: ValidationLevel, code: str, message: str, **kwargs):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(level=level, code=code, message=message, **kwargs))
        if level == ValidationLevel.ERROR:
            self.valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "trajectory_id": self.trajectory_id,
            "issues": [
                {
                    "level": issue.level.value,
                    "code": issue.code,
                    "message": issue.message,
                    "location": issue.location,
                    "details": issue.details
                }
                for issue in self.issues
            ],
            "metrics": self.metrics
        }


class TrajectoryValidator:
    """
    Validates CausalBench trajectories.

    Checks:
    1. Schema compliance (required fields)
    2. Causal graph validity (DAG properties)
    3. Attack labeling correctness
    4. Data flow consistency
    5. Trust level assignments
    """

    # Required fields at different levels.
    # `agent_id` is no longer required at trajectory level — it lives inside
    # agents[] for MAS scenarios and defaults to DEFAULT_AGENT_ID for single-agent.
    # The event collection may live under any of EVENTS_KEY_ALIASES.
    # The trajectory id may live under any of ID_KEY_ALIASES (trajectory_id is canonical).
    REQUIRED_TRAJECTORY_FIELDS: List[str] = []  # validated via aliases below
    ID_KEY_ALIASES = ("trajectory_id", "session_id")
    EVENTS_KEY_ALIASES = ("events", "trajectory", "nodes")
    DEFAULT_AGENT_ID = "agent_0"
    REQUIRED_EVENT_FIELDS = [
        "event_id", "timestamp", "event_type", "service", "request", "response"
    ]
    REQUIRED_GRAPH_FIELDS = ["nodes", "edges"]
    # When 'agents' is present the trajectory is multi-agent; each entry must
    # carry agent_id, and inter_agent_flows endpoints must reference valid ids.
    MAS_AGENT_REQUIRED_FIELDS = ["agent_id"]

    VALID_EVENT_TYPES = ["task", "observation", "llm_call", "action", "result", "error"]
    VALID_TRUST_LEVELS = ["trusted", "untrusted", "sensitive"]
    VALID_EDGE_TYPES = ["data_dependency", "trust_transfer", "state_enablement"]

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode

    def validate_trajectory(self, trajectory: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single trajectory.

        Args:
            trajectory: Trajectory dictionary

        Returns:
            ValidationResult with issues and metrics
        """
        trajectory_id = self._get_id_field(trajectory) or "unknown"
        result = ValidationResult(valid=True, trajectory_id=trajectory_id)

        # Schema validation
        self._validate_schema(trajectory, result)

        # If schema is valid, do deeper validation
        if result.valid:
            self._validate_events(self._get_events_field(trajectory) or [], result)
            self._validate_causal_graph(self._get_graph_field(trajectory), result)
            self._validate_attack_labels(trajectory, result)
            self._validate_data_flow(trajectory, result)
            self._validate_trust_levels(trajectory, result)

        # Compute metrics
        result.metrics = self._compute_metrics(trajectory)

        return result

    def validate_file(self, filepath: str) -> ValidationResult:
        """Validate a trajectory from a file."""
        try:
            with open(filepath, 'r') as f:
                trajectory = json.load(f)
            return self.validate_trajectory(trajectory)
        except json.JSONDecodeError as e:
            result = ValidationResult(valid=False, trajectory_id=filepath)
            result.add_issue(ValidationLevel.ERROR, "INVALID_JSON", f"Invalid JSON: {e}")
            return result
        except FileNotFoundError:
            result = ValidationResult(valid=False, trajectory_id=filepath)
            result.add_issue(ValidationLevel.ERROR, "FILE_NOT_FOUND", f"File not found: {filepath}")
            return result

    def validate_directory(self, dirpath: str) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """
        Validate all trajectories in a directory.

        Returns:
            Tuple of (list of results, aggregate statistics)
        """
        results = []
        path = Path(dirpath)

        for filepath in path.glob("*.json"):
            result = self.validate_file(str(filepath))
            results.append(result)

        # Aggregate statistics
        stats = {
            "total": len(results),
            "valid": sum(1 for r in results if r.valid),
            "invalid": sum(1 for r in results if not r.valid),
            "issues_by_code": {},
            "issues_by_level": {level.value: 0 for level in ValidationLevel}
        }

        for result in results:
            for issue in result.issues:
                stats["issues_by_level"][issue.level.value] += 1
                if issue.code not in stats["issues_by_code"]:
                    stats["issues_by_code"][issue.code] = 0
                stats["issues_by_code"][issue.code] += 1

        return results, stats

    def _get_id_field(self, trajectory: Dict[str, Any]) -> Optional[str]:
        """Return the trajectory identifier under whichever alias the writer used.

        Canonical key is `trajectory_id`; `session_id` is accepted for back-compat.
        """
        for key in self.ID_KEY_ALIASES:
            v = trajectory.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    def _get_events_field(self, trajectory: Dict[str, Any]) -> Optional[List[Dict]]:
        """Return the event collection under whichever alias the writer used.

        Writers may emit the event list as 'events', 'trajectory', or 'nodes'.
        Returns None if no list-shaped value is present under any alias.
        """
        for key in self.EVENTS_KEY_ALIASES:
            v = trajectory.get(key)
            if isinstance(v, list):
                return v
        return None

    def _get_graph_field(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Return a {'nodes', 'edges'} dict regardless of whether the writer
        nested it under 'causal_graph' or emitted top-level 'nodes'/'edges'.
        """
        graph = trajectory.get("causal_graph")
        if isinstance(graph, dict):
            return graph
        if "nodes" in trajectory and "edges" in trajectory:
            return {"nodes": trajectory.get("nodes", []),
                    "edges": trajectory.get("edges", [])}
        return {}

    def _validate_schema(self, trajectory: Dict[str, Any], result: ValidationResult):
        """Validate trajectory schema."""
        # 1. Trajectory must carry an id under one of the accepted aliases.
        if self._get_id_field(trajectory) is None:
            result.add_issue(
                ValidationLevel.ERROR,
                "MISSING_FIELD",
                f"Missing trajectory id (expected one of: "
                f"{', '.join(self.ID_KEY_ALIASES)})",
                location="trajectory"
            )
        # Any additional hard-required fields go here.
        for field in self.REQUIRED_TRAJECTORY_FIELDS:
            if field not in trajectory:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "MISSING_FIELD",
                    f"Missing required field: {field}",
                    location="trajectory"
                )

        # 2. Event collection must exist under one of the accepted aliases.
        if self._get_events_field(trajectory) is None:
            result.add_issue(
                ValidationLevel.ERROR,
                "MISSING_FIELD",
                f"Missing event collection (expected one of: "
                f"{', '.join(self.EVENTS_KEY_ALIASES)})",
                location="trajectory"
            )

        # 3. Causal graph must exist either nested under 'causal_graph'
        #    OR as top-level 'nodes' + 'edges'.
        has_top_graph = isinstance(trajectory.get("causal_graph"), dict)
        has_inline_graph = "nodes" in trajectory and "edges" in trajectory
        if not has_top_graph and not has_inline_graph:
            result.add_issue(
                ValidationLevel.ERROR,
                "MISSING_FIELD",
                "Missing causal graph (expected 'causal_graph' object "
                "OR top-level 'nodes' + 'edges')",
                location="trajectory"
            )

        # 4. Type checks on whichever alias is in use.
        for key in self.EVENTS_KEY_ALIASES:
            if key in trajectory and not isinstance(trajectory[key], list):
                result.add_issue(
                    ValidationLevel.ERROR,
                    "INVALID_TYPE",
                    f"Field '{key}' must be a list",
                    location=f"trajectory.{key}"
                )

        if "causal_graph" in trajectory and not isinstance(trajectory["causal_graph"], dict):
            result.add_issue(
                ValidationLevel.ERROR,
                "INVALID_TYPE",
                "Field 'causal_graph' must be a dictionary",
                location="trajectory.causal_graph"
            )

        # 5. Multi-agent branch: when 'agents' is present, validate MAS structure.
        if "agents" in trajectory:
            self._validate_mas(trajectory, result)

    def _validate_mas(self, trajectory: Dict[str, Any], result: ValidationResult):
        """Validate multi-agent fields when the trajectory has an 'agents' list.

        Requires each agent entry to carry an agent_id, and warns when
        inter_agent_flows reference agent_ids not present in agents[].
        """
        agents = trajectory.get("agents")
        if not isinstance(agents, list):
            result.add_issue(
                ValidationLevel.ERROR, "INVALID_TYPE",
                "Field 'agents' must be a list",
                location="trajectory.agents"
            )
            return
        agent_ids = set()
        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                result.add_issue(
                    ValidationLevel.ERROR, "INVALID_TYPE",
                    "Each agent entry must be a dictionary",
                    location=f"agents[{i}]"
                )
                continue
            for field in self.MAS_AGENT_REQUIRED_FIELDS:
                if not agent.get(field):
                    result.add_issue(
                        ValidationLevel.ERROR, "MISSING_FIELD",
                        f"Missing required field: {field}",
                        location=f"agents[{i}]"
                    )
            aid = agent.get("agent_id")
            if aid:
                agent_ids.add(aid)
        flows = trajectory.get("inter_agent_flows", []) or []
        if not isinstance(flows, list):
            result.add_issue(
                ValidationLevel.ERROR, "INVALID_TYPE",
                "Field 'inter_agent_flows' must be a list",
                location="trajectory.inter_agent_flows"
            )
            return
        for j, flow in enumerate(flows):
            if not isinstance(flow, dict):
                continue
            for end in ("source_agent", "target_agent"):
                ref = flow.get(end)
                if ref and ref not in agent_ids:
                    result.add_issue(
                        ValidationLevel.WARNING, "UNKNOWN_AGENT_ID",
                        f"{end}={ref!r} not declared in agents[]",
                        location=f"inter_agent_flows[{j}]"
                    )

    def _validate_events(self, events: List[Dict], result: ValidationResult):
        """Validate trajectory events."""
        seen_ids = set()
        prev_timestamp = None

        for i, event in enumerate(events):
            location = f"events[{i}]"

            # Check required fields
            for field in self.REQUIRED_EVENT_FIELDS:
                if field not in event:
                    result.add_issue(
                        ValidationLevel.ERROR,
                        "MISSING_EVENT_FIELD",
                        f"Missing required field: {field}",
                        location=location
                    )

            # Check for duplicate IDs
            event_id = event.get("event_id")
            if event_id:
                if event_id in seen_ids:
                    result.add_issue(
                        ValidationLevel.ERROR,
                        "DUPLICATE_EVENT_ID",
                        f"Duplicate event ID: {event_id}",
                        location=location
                    )
                seen_ids.add(event_id)

            # Check event type (case-insensitive — writer emits UPPERCASE,
            # validator's canonical list is lowercase; accept both, plus the
            # extractor-side TOOL_CALL synonym).
            event_type = event.get("event_type")
            normalized = (event_type or "").lower()
            accepted = set(self.VALID_EVENT_TYPES) | {"tool_call"}
            if event_type and normalized not in accepted:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "INVALID_EVENT_TYPE",
                    f"Unknown event type: {event_type}",
                    location=location
                )

            # Check trust level
            trust_level = event.get("trust_level")
            if trust_level and trust_level not in self.VALID_TRUST_LEVELS:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "INVALID_TRUST_LEVEL",
                    f"Unknown trust level: {trust_level}",
                    location=location
                )

            # Check timestamp ordering
            timestamp = event.get("timestamp")
            if timestamp and prev_timestamp:
                if timestamp < prev_timestamp:
                    result.add_issue(
                        ValidationLevel.WARNING,
                        "TIMESTAMP_ORDER",
                        "Events not in chronological order",
                        location=location
                    )
            prev_timestamp = timestamp

            # Check parent references
            parent_ids = event.get("parent_ids", [])
            for parent_id in parent_ids:
                if parent_id not in seen_ids:
                    result.add_issue(
                        ValidationLevel.ERROR,
                        "INVALID_PARENT_REF",
                        f"Parent event not found: {parent_id}",
                        location=location
                    )

    def _validate_causal_graph(self, graph: Dict[str, Any], result: ValidationResult):
        """Validate causal graph structure."""
        if not graph:
            result.add_issue(
                ValidationLevel.WARNING,
                "EMPTY_GRAPH",
                "Causal graph is empty"
            )
            return

        # Check required fields
        for field in self.REQUIRED_GRAPH_FIELDS:
            if field not in graph:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "MISSING_GRAPH_FIELD",
                    f"Missing required graph field: {field}",
                    location="causal_graph"
                )

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Check nodes
        node_ids = set()
        for i, node in enumerate(nodes):
            node_id = node.get("id") if isinstance(node, dict) else node
            if node_id:
                node_ids.add(node_id)

        # Check edges reference valid nodes
        for i, edge in enumerate(edges):
            source = edge.get("source")
            target = edge.get("target")

            if source and source not in node_ids:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "INVALID_EDGE_SOURCE",
                    f"Edge source not in nodes: {source}",
                    location=f"causal_graph.edges[{i}]"
                )

            if target and target not in node_ids:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "INVALID_EDGE_TARGET",
                    f"Edge target not in nodes: {target}",
                    location=f"causal_graph.edges[{i}]"
                )

            # Check edge type
            edge_type = edge.get("edge_type")
            if edge_type and edge_type not in self.VALID_EDGE_TYPES:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "INVALID_EDGE_TYPE",
                    f"Unknown edge type: {edge_type}",
                    location=f"causal_graph.edges[{i}]"
                )

        # Check for cycles (DAG property)
        if edges and nodes:
            has_cycle = self._check_cycle(node_ids, edges)
            if has_cycle:
                result.add_issue(
                    ValidationLevel.ERROR,
                    "GRAPH_CYCLE",
                    "Causal graph contains a cycle (must be DAG)"
                )

    def _check_cycle(self, nodes: set, edges: List[Dict]) -> bool:
        """Check if graph has a cycle using DFS."""
        # Build adjacency list
        adj = {node: [] for node in nodes}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source in adj and target:
                adj[source].append(target)

        # DFS with coloring
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in nodes}

        def dfs(node):
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True  # Back edge = cycle
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def _validate_attack_labels(self, trajectory: Dict[str, Any], result: ValidationResult):
        """Validate attack labeling consistency."""
        is_attack = trajectory.get("is_attack", trajectory.get("attack_executed", False))
        attack_type = trajectory.get("attack_type")
        events = self._get_events_field(trajectory) or []

        # If labeled as attack, should have injection source
        if is_attack:
            has_injection_source = any(
                event.get("is_injection_source", False)
                for event in events
            )
            if not has_injection_source:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "MISSING_INJECTION_SOURCE",
                    "Attack trajectory has no marked injection source"
                )

            # Should have attack type
            if not attack_type:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "MISSING_ATTACK_TYPE",
                    "Attack trajectory has no attack_type specified"
                )

        # If has injection source, should be labeled as attack
        has_injection = any(
            event.get("is_injection_source", False)
            for event in events
        )
        if has_injection and not is_attack:
            result.add_issue(
                ValidationLevel.WARNING,
                "UNLABELED_ATTACK",
                "Trajectory has injection source but not labeled as attack"
            )

    def _validate_data_flow(self, trajectory: Dict[str, Any], result: ValidationResult):
        """Validate data flow consistency."""
        events = self._get_events_field(trajectory) or []
        produced_data = set()
        consumed_data = set()

        for event in events:
            # Track produced data
            for data in event.get("data_produced", []):
                produced_data.add(data)

            # Track consumed data
            for data in event.get("data_consumed", []):
                consumed_data.add(data)
                # Check if data was produced before consumption
                if data not in produced_data:
                    result.add_issue(
                        ValidationLevel.WARNING,
                        "UNDEFINED_DATA_CONSUMPTION",
                        f"Data consumed before production: {data}",
                        location=f"event {event.get('event_id')}"
                    )

    def _validate_trust_levels(self, trajectory: Dict[str, Any], result: ValidationResult):
        """Validate trust level transitions."""
        events = self._get_events_field(trajectory) or []
        graph = trajectory.get("causal_graph", {})
        edges = graph.get("edges", [])

        # Build event trust level map
        trust_map = {}
        for event in events:
            event_id = event.get("event_id")
            trust_level = event.get("trust_level", "trusted")
            if event_id:
                trust_map[event_id] = trust_level

        # Check for suspicious trust transitions in edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("edge_type")

            source_trust = trust_map.get(source, "trusted")
            target_trust = trust_map.get(target, "trusted")

            # Untrusted -> Sensitive is suspicious
            if source_trust == "untrusted" and target_trust == "sensitive":
                if edge_type == "data_dependency":
                    result.add_issue(
                        ValidationLevel.INFO,
                        "SUSPICIOUS_TRUST_FLOW",
                        f"Data flows from untrusted to sensitive: {source} -> {target}",
                        details={"edge_type": edge_type}
                    )

    def _compute_metrics(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Compute trajectory metrics."""
        events = self._get_events_field(trajectory) or []
        graph = trajectory.get("causal_graph", {})

        return {
            "num_events": len(events),
            "num_nodes": len(graph.get("nodes", [])),
            "num_edges": len(graph.get("edges", [])),
            "services": list(set(e.get("service") for e in events if e.get("service"))),
            "event_types": list(set(e.get("event_type") for e in events if e.get("event_type"))),
            "has_injection": any(e.get("is_injection_source") for e in events),
            "has_sensitive_sink": any(e.get("is_sensitive_sink") for e in events)
        }


def validate_dataset(
    input_dir: str,
    output_file: Optional[str] = None,
    strict: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate an entire dataset.

    Args:
        input_dir: Directory containing trajectory JSON files
        output_file: Optional path to write validation report
        strict: Use strict validation mode

    Returns:
        Tuple of (all_valid, statistics)
    """
    validator = TrajectoryValidator(strict_mode=strict)
    results, stats = validator.validate_directory(input_dir)

    all_valid = stats["invalid"] == 0

    # Write report if requested
    if output_file:
        report = {
            "summary": stats,
            "all_valid": all_valid,
            "results": [r.to_dict() for r in results]
        }
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

    logger.info(f"Validation complete: {stats['valid']}/{stats['total']} valid")
    if stats["issues_by_code"]:
        logger.info(f"Issues by code: {stats['issues_by_code']}")

    return all_valid, stats
