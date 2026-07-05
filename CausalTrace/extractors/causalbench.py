"""
CausalBench Extractor for CausalTrace.

Extracts trajectories from CausalBench dataset generator output format.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import glob

from causaltrace.extractors.base import BaseExtractor
from causaltrace.models import Trajectory, Action, ActionType, State


# Mapping from CausalBench event types to CausalTrace action types
EVENT_TYPE_MAPPING = {
    "TASK": ActionType.UNKNOWN,
    "OBSERVATION": ActionType.READ,
    "LLM_CALL": ActionType.UNKNOWN,  # Internal agent processing
    "ACTION": ActionType.TOOL_CALL,
    "TOOL_CALL": ActionType.TOOL_CALL,
}

# Mapping from service names to domains
SERVICE_DOMAINS = {
    "github": "github.com",
    "slack": "slack.com",
    "stripe": "stripe.com",
    "gmail": "mail.google.com",
    "dropbox": "dropbox.com",
    "notion": "notion.so",
    "trello": "trello.com",
    "http": "external",
    "llm": "internal",
    "system": "internal",
}

# Keywords that indicate sensitive operations
SENSITIVE_KEYWORDS = {
    "password", "secret", "token", "api_key", "credential",
    "payment", "credit_card", "ssn", "private_key"
}

EXFIL_KEYWORDS = {
    "collect", "exfil", "send", "post", "upload", "transfer"
}


class CausalBenchExtractor(BaseExtractor):
    """
    Extract trajectories from CausalBench generator output.

    CausalBench trajectories contain:
    - session_id: Unique identifier
    - is_attack: Boolean attack label
    - attack_type: Type of attack (if attack)
    - trajectory: List of events with:
        - event_id, event_type, service, endpoint
        - trust_level, is_injection_source, is_sensitive_sink
        - data_produced, data_consumed
    - causal_graph: Pre-computed graph with nodes and edges
    """

    def __init__(self, verbose: bool = False, include_llm_calls: bool = True):
        """
        Initialize extractor.

        Args:
            verbose: Print extraction progress
            include_llm_calls: Whether to include LLM_CALL events as actions (default: True)
                             LLM calls are essential for data flow tracking.
        """
        super().__init__(verbose=verbose)
        self.include_llm_calls = include_llm_calls

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Extract a single trajectory from a CausalBench JSON file.

        Args:
            log_path: Path to the trajectory JSON file

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"Path does not exist: {log_path}")
            return None

        try:
            with open(log_path, 'r') as f:
                data = json.load(f)

            return self._parse_trajectory(data, log_path)

        except json.JSONDecodeError as e:
            self._log(f"Invalid JSON in {log_path}: {e}")
            return None
        except Exception as e:
            self._log(f"Error parsing {log_path}: {e}")
            return None

    def extract_from_directory(self, dir_path: str, pattern: str = "trajectory_*.json") -> List[Trajectory]:
        """
        Extract all trajectories from a directory.

        Args:
            dir_path: Directory containing trajectory files
            pattern: Glob pattern for matching files

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory does not exist: {dir_path}")
            return []

        trajectories = []
        search_pattern = str(Path(dir_path) / pattern)
        files = glob.glob(search_pattern)

        self._log(f"Found {len(files)} files matching {pattern}")

        for file_path in sorted(files):
            trajectory = self.extract_from_log(file_path)
            if trajectory:
                trajectories.append(trajectory)

        self._log(f"Successfully extracted {len(trajectories)} trajectories")
        return trajectories

    def _parse_trajectory(self, data: Dict[str, Any], source_path: str) -> Optional[Trajectory]:
        """
        Parse a CausalBench trajectory dict into a Trajectory object.

        Args:
            data: Dictionary from JSON file
            source_path: Path to source file (for error messages)

        Returns:
            Trajectory object or None
        """
        # Extract basic info
        session_id = data.get('session_id', '')
        is_attack = data.get('is_attack', False)
        attack_type = data.get('attack_type', None)
        task_description = data.get('task_description', '') or ''

        # Parse events into actions
        events = data.get('trajectory', [])
        actions = []

        for i, event in enumerate(events):
            action = self._parse_event(event, i)
            if action:
                # Skip LLM calls if not included
                if event.get('event_type') == 'LLM_CALL' and not self.include_llm_calls:
                    continue
                actions.append(action)

        if not actions:
            self._log(f"No valid actions in {source_path}")
            return None

        # Build initial state
        initial_state = State(
            authenticated_services=list(self._get_services(events)),
            accumulated_data={},
            permissions=[],
        )

        # Build metadata
        metadata = {
            'attack_type': attack_type,
            'sophistication': data.get('sophistication', None),
            'scenario_id': data.get('scenario_id', None),
            'num_events': data.get('num_events', len(events)),
            'causal_graph': data.get('causal_graph', {}),
            'injection_sources': data.get('metadata', {}).get('injection_sources', []),
            'sensitive_sinks': data.get('metadata', {}).get('sensitive_sinks', []),
            'exfil_captured': data.get('metadata', {}).get('exfil_captured', []),
            'source_path': source_path,
        }

        return Trajectory(
            trajectory_id=session_id,
            source='causalbench',
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            initial_state=initial_state,
            metadata=metadata,
        )

    def _parse_event(self, event: Dict[str, Any], index: int) -> Optional[Action]:
        """
        Parse a CausalBench event into an Action.

        Args:
            event: Event dictionary
            index: Index in trajectory (used as action_id)

        Returns:
            Action object or None
        """
        event_type = event.get('event_type', 'UNKNOWN')
        service = event.get('service', 'unknown')
        endpoint = event.get('endpoint', '')

        # Determine action type
        action_type = EVENT_TYPE_MAPPING.get(event_type, ActionType.UNKNOWN)

        # For ACTION events, try to infer more specific type
        if event_type == 'ACTION':
            action_type = self._infer_action_type(service, endpoint)

        # Get domain from service
        domain = SERVICE_DOMAINS.get(service.lower(), service)

        # Build context with trust/injection info
        context = {
            'event_type': event_type,
            'service': service,
            'trust_level': event.get('trust_level', 'trusted'),
            'is_injection_source': event.get('is_injection_source', False),
            'is_sensitive_sink': event.get('is_sensitive_sink', False),
        }

        # Add request/response data if present
        if 'request' in event:
            context['request'] = event['request']
        if 'response' in event:
            context['response'] = event['response']

        # Get result
        result = None
        if 'response' in event:
            result = json.dumps(event['response']) if isinstance(event['response'], dict) else str(event['response'])

        return Action(
            action_id=index,
            action_type=action_type,
            target=endpoint,
            context=context,
            result=result,
            domain=domain,
            data_produced=event.get('data_produced', []),
            data_consumed=event.get('data_consumed', []),
        )

    def _infer_action_type(self, service: str, endpoint: str) -> ActionType:
        """
        Infer action type from service and endpoint.

        Args:
            service: Service name
            endpoint: Endpoint/action name

        Returns:
            Most appropriate ActionType
        """
        endpoint_lower = endpoint.lower()

        # Check for common patterns
        if 'send' in endpoint_lower or 'email' in endpoint_lower:
            return ActionType.SEND_EMAIL

        if 'upload' in endpoint_lower or 'create' in endpoint_lower:
            return ActionType.UPLOAD

        if 'download' in endpoint_lower or 'get' in endpoint_lower or 'read' in endpoint_lower:
            return ActionType.READ

        if 'http' in service.lower() or 'collect' in endpoint_lower:
            return ActionType.WEB_FETCH  # Exfiltration endpoint

        if 'navigate' in endpoint_lower or 'browse' in endpoint_lower:
            return ActionType.NAVIGATE

        # Default to tool call for API operations
        return ActionType.TOOL_CALL

    def _get_services(self, events: List[Dict[str, Any]]) -> set:
        """
        Get set of unique services from events.

        Args:
            events: List of event dictionaries

        Returns:
            Set of service names
        """
        services = set()
        for event in events:
            service = event.get('service', '')
            if service and service not in ('system', 'llm'):
                services.add(service)
        return services

    def get_attack_statistics(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """
        Compute statistics about attack trajectories.

        Args:
            trajectories: List of trajectories to analyze

        Returns:
            Dictionary with attack statistics
        """
        attacks = [t for t in trajectories if t.is_attack]
        benign = [t for t in trajectories if not t.is_attack]

        # Count attack types
        attack_types = {}
        for t in attacks:
            atype = t.metadata.get('attack_type', 'unknown')
            attack_types[atype] = attack_types.get(atype, 0) + 1

        # Compute average actions per trajectory
        avg_attack_actions = sum(len(t.actions) for t in attacks) / len(attacks) if attacks else 0
        avg_benign_actions = sum(len(t.actions) for t in benign) / len(benign) if benign else 0

        # Count injection sources and sensitive sinks
        total_injection_sources = 0
        total_sensitive_sinks = 0
        for t in attacks:
            total_injection_sources += len(t.metadata.get('injection_sources', []))
            total_sensitive_sinks += len(t.metadata.get('sensitive_sinks', []))

        return {
            'total_trajectories': len(trajectories),
            'num_attacks': len(attacks),
            'num_benign': len(benign),
            'attack_rate': len(attacks) / len(trajectories) if trajectories else 0,
            'attack_types': attack_types,
            'avg_attack_actions': avg_attack_actions,
            'avg_benign_actions': avg_benign_actions,
            'total_injection_sources': total_injection_sources,
            'total_sensitive_sinks': total_sensitive_sinks,
        }


__all__ = ['CausalBenchExtractor']
