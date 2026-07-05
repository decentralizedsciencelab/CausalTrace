"""
Trajectory Logger for CausalBench

Captures agent actions with causal metadata for graph construction.
"""

import uuid
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import hashlib


class EventType(str, Enum):
    TASK = "TASK"
    OBSERVATION = "OBSERVATION"
    LLM_CALL = "LLM_CALL"
    ACTION = "ACTION"
    RESULT = "RESULT"
    ERROR = "ERROR"


class TrustLevel(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    SENSITIVE = "sensitive"


class EdgeType(str, Enum):
    DATA_DEPENDENCY = "data_dependency"
    TRUST_TRANSFER = "trust_transfer"
    STATE_ENABLEMENT = "state_enablement"


@dataclass
class TrajectoryEvent:
    """Single event in an agent trajectory."""
    event_id: str
    session_id: str
    timestamp: str
    event_type: EventType
    service: str
    endpoint: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    parent_ids: List[str] = field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.TRUSTED
    is_injection_source: bool = False
    is_sensitive_sink: bool = False
    data_produced: List[str] = field(default_factory=list)
    data_consumed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['event_type'] = self.event_type.value
        d['trust_level'] = self.trust_level.value
        return d


@dataclass
class CausalEdge:
    """Edge in the causal graph."""
    source: str  # event_id
    target: str  # event_id
    edge_type: EdgeType
    data_key: Optional[str] = None  # What data flows through this edge
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "data_key": self.data_key,
            "confidence": self.confidence
        }


@dataclass
class CausalGraph:
    """Causal graph of a trajectory."""
    nodes: List[str]  # event_ids
    edges: List[CausalEdge]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges]
        }

    def has_path(self, source: str, target: str) -> bool:
        """Check if there's a path from source to target."""
        # Build adjacency list
        adj = {}
        for edge in self.edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)

        # BFS
        visited = set()
        queue = [source]
        while queue:
            node = queue.pop(0)
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj.get(node, []))
        return False

    def validate(self) -> List[str]:
        """Validate the graph (check for cycles, orphans, etc.)."""
        errors = []

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            adj = {}
            for edge in self.edges:
                if edge.source not in adj:
                    adj[edge.source] = []
                adj[edge.source].append(edge.target)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        visited = set()
        rec_stack = set()
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    errors.append("Graph contains a cycle")
                    break

        # Check for orphan edges
        node_set = set(self.nodes)
        for edge in self.edges:
            if edge.source not in node_set:
                errors.append(f"Edge source {edge.source} not in nodes")
            if edge.target not in node_set:
                errors.append(f"Edge target {edge.target} not in nodes")

        return errors


class TrajectoryLogger:
    """
    Logs agent trajectory events and builds causal graphs.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.events: List[TrajectoryEvent] = []
        self.data_store: Dict[str, str] = {}  # data_key -> producing event_id
        self._event_counter = 0

    def _generate_event_id(self) -> str:
        self._event_counter += 1
        return f"e_{self._event_counter:04d}"

    def _generate_data_key(self, service: str, endpoint: str, data: Any) -> str:
        """Generate a unique key for a piece of data."""
        content = f"{service}:{endpoint}:{json.dumps(data, sort_keys=True, default=str)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def log_event(
        self,
        event_type: EventType,
        service: str,
        endpoint: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        trust_level: TrustLevel = TrustLevel.TRUSTED,
        is_injection_source: bool = False,
        is_sensitive_sink: bool = False,
        data_consumed: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrajectoryEvent:
        """Log a single event."""
        event_id = self._generate_event_id()

        # Determine parent events based on data consumed
        parent_ids = []
        consumed = data_consumed or []
        for data_key in consumed:
            if data_key in self.data_store:
                parent_ids.append(self.data_store[data_key])

        # Generate data produced key
        data_produced = []
        if response and not isinstance(response, str):
            data_key = self._generate_data_key(service, endpoint, response)
            data_produced.append(data_key)
            self.data_store[data_key] = event_id

        event = TrajectoryEvent(
            event_id=event_id,
            session_id=self.session_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            service=service,
            endpoint=endpoint,
            request=request,
            response=response,
            parent_ids=parent_ids,
            trust_level=trust_level,
            is_injection_source=is_injection_source,
            is_sensitive_sink=is_sensitive_sink,
            data_produced=data_produced,
            data_consumed=consumed,
            metadata=metadata or {}
        )

        self.events.append(event)
        return event

    def log_task(self, task_description: str) -> TrajectoryEvent:
        """Log the initial task."""
        return self.log_event(
            event_type=EventType.TASK,
            service="system",
            endpoint="task",
            request={"task": task_description},
            response={},
            trust_level=TrustLevel.TRUSTED
        )

    def log_observation(
        self,
        service: str,
        endpoint: str,
        request: Dict,
        response: Dict,
        trust_level: TrustLevel = TrustLevel.TRUSTED,
        is_injection_source: bool = False,
        data_consumed: Optional[List[str]] = None
    ) -> TrajectoryEvent:
        """Log an observation (reading data)."""
        return self.log_event(
            event_type=EventType.OBSERVATION,
            service=service,
            endpoint=endpoint,
            request=request,
            response=response,
            trust_level=trust_level,
            is_injection_source=is_injection_source,
            data_consumed=data_consumed
        )

    def log_action(
        self,
        service: str,
        endpoint: str,
        request: Dict,
        response: Dict,
        trust_level: TrustLevel = TrustLevel.TRUSTED,
        is_sensitive_sink: bool = False,
        data_consumed: Optional[List[str]] = None
    ) -> TrajectoryEvent:
        """Log an action (writing data)."""
        return self.log_event(
            event_type=EventType.ACTION,
            service=service,
            endpoint=endpoint,
            request=request,
            response=response,
            trust_level=trust_level,
            is_sensitive_sink=is_sensitive_sink,
            data_consumed=data_consumed
        )

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str = "gpt-4",
        data_consumed: Optional[List[str]] = None
    ) -> TrajectoryEvent:
        """Log an LLM call."""
        return self.log_event(
            event_type=EventType.LLM_CALL,
            service="llm",
            endpoint=model,
            request={"prompt": prompt},
            response={"response": response},
            trust_level=TrustLevel.TRUSTED,
            data_consumed=data_consumed
        )

    def build_causal_graph(self) -> CausalGraph:
        """Build causal graph from logged events."""
        nodes = [e.event_id for e in self.events]
        edges = []

        # Build data dependency edges
        data_producers: Dict[str, str] = {}
        for event in self.events:
            # Track what this event produces
            for data_key in event.data_produced:
                data_producers[data_key] = event.event_id

        for event in self.events:
            # Create edges for data consumed
            for data_key in event.data_consumed:
                if data_key in data_producers:
                    source_id = data_producers[data_key]
                    if source_id != event.event_id:
                        edges.append(CausalEdge(
                            source=source_id,
                            target=event.event_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            data_key=data_key
                        ))

        # Build trust transfer edges (injection source -> action)
        injection_events = [e for e in self.events if e.is_injection_source]
        action_events = [e for e in self.events if e.event_type == EventType.ACTION]

        for inj in injection_events:
            for action in action_events:
                # If action consumed data that came from injection
                for data_key in action.data_consumed:
                    if data_key in inj.data_produced:
                        edges.append(CausalEdge(
                            source=inj.event_id,
                            target=action.event_id,
                            edge_type=EdgeType.TRUST_TRANSFER,
                            data_key=data_key
                        ))

        # Build state enablement edges (sequential dependencies)
        for i, event in enumerate(self.events[1:], 1):
            prev_event = self.events[i - 1]
            # If events are on same service and this one requires auth/state
            if (event.service == prev_event.service and
                event.event_type == EventType.ACTION and
                prev_event.event_type in [EventType.ACTION, EventType.OBSERVATION]):
                edges.append(CausalEdge(
                    source=prev_event.event_id,
                    target=event.event_id,
                    edge_type=EdgeType.STATE_ENABLEMENT
                ))

        return CausalGraph(nodes=nodes, edges=edges)

    def to_trajectory_dict(
        self,
        is_attack: bool = False,
        attack_type: Optional[str] = None,
        vertical: Optional[str] = None,
        task_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export trajectory to dictionary format."""
        graph = self.build_causal_graph()

        return {
            "session_id": self.session_id,
            "is_attack": is_attack,
            "attack_type": attack_type,
            "vertical": vertical,
            "task_description": task_description,
            "num_events": len(self.events),
            "trajectory": [e.to_dict() for e in self.events],
            "causal_graph": graph.to_dict(),
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "injection_sources": [e.event_id for e in self.events if e.is_injection_source],
                "sensitive_sinks": [e.event_id for e in self.events if e.is_sensitive_sink],
                "services_used": list(set(e.service for e in self.events)),
            }
        }

    def validate(self) -> List[str]:
        """Validate the trajectory."""
        errors = []

        if not self.events:
            errors.append("No events logged")
            return errors

        # Check first event is TASK
        if self.events[0].event_type != EventType.TASK:
            errors.append("First event should be TASK type")

        # Validate causal graph
        graph = self.build_causal_graph()
        errors.extend(graph.validate())

        # Check attack trajectories have path from injection to sink
        injection_events = [e for e in self.events if e.is_injection_source]
        sink_events = [e for e in self.events if e.is_sensitive_sink]

        if injection_events and sink_events:
            has_attack_path = False
            for inj in injection_events:
                for sink in sink_events:
                    if graph.has_path(inj.event_id, sink.event_id):
                        has_attack_path = True
                        break
            if not has_attack_path:
                errors.append("Attack trajectory has no path from injection to sink")

        return errors
