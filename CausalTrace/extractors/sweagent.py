"""
SWE-agent trajectory extractor.

This module extracts trajectories from SWE-agent (Software Engineering Agent)
benchmark logs. SWE-agent generates trajectories with thought-action-observation
turns representing software engineering task solving attempts.

SWE-agent Format Notes:
- Trajectories stored in .traj files (JSON format)
- Each step contains: response, thought, action, observation, state, query
- Actions are typically shell commands executed in a repository context
- Observations contain terminal output, file contents, git diffs, etc.

Dataset source: https://huggingface.co/datasets/nebius/SWE-agent-trajectories
"""

import json
import re
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk
from causaltrace.extractors.base import BaseExtractor


@dataclass
class SWEAgentStep:
    """Represents a single step in a SWE-agent trajectory."""
    response: str
    thought: str
    action: str
    observation: str
    state: Dict[str, Any]
    query: List[Dict[str, str]]


class SWEAgentExtractor(BaseExtractor):
    """
    Extract trajectories from SWE-agent benchmark logs.

    SWE-agent logs contain information about:
    - GitHub issue descriptions (task)
    - Step-by-step thought-action-observation sequences
    - Shell commands and their outputs
    - File contents and modifications
    - Git operations
    """

    # Action type mappings for common shell commands
    COMMAND_ACTION_TYPES = {
        'ls': ActionType.READ,
        'cat': ActionType.READ,
        'head': ActionType.READ,
        'tail': ActionType.READ,
        'find': ActionType.READ,
        'grep': ActionType.READ,
        'less': ActionType.READ,
        'more': ActionType.READ,
        'file': ActionType.READ,
        'wc': ActionType.READ,
        'diff': ActionType.READ,
        'git diff': ActionType.READ,
        'git log': ActionType.READ,
        'git status': ActionType.READ,
        'git show': ActionType.READ,
        'git branch': ActionType.READ,
        'cd': ActionType.NAVIGATE,
        'pushd': ActionType.NAVIGATE,
        'popd': ActionType.NAVIGATE,
        'edit': ActionType.WRITE,
        'vim': ActionType.WRITE,
        'nano': ActionType.WRITE,
        'sed': ActionType.WRITE,
        'echo': ActionType.WRITE,
        'touch': ActionType.WRITE,
        'mkdir': ActionType.WRITE,
        'cp': ActionType.WRITE,
        'mv': ActionType.WRITE,
        'rm': ActionType.WRITE,
        'git add': ActionType.WRITE,
        'git commit': ActionType.WRITE,
        'git checkout': ActionType.WRITE,
        'git reset': ActionType.WRITE,
        'git stash': ActionType.WRITE,
        'python': ActionType.CODE_EXECUTION,
        'python3': ActionType.CODE_EXECUTION,
        'pytest': ActionType.CODE_EXECUTION,
        'pip': ActionType.CODE_EXECUTION,
        'pip3': ActionType.CODE_EXECUTION,
        'make': ActionType.CODE_EXECUTION,
        'npm': ActionType.CODE_EXECUTION,
        'node': ActionType.CODE_EXECUTION,
        'bash': ActionType.CODE_EXECUTION,
        'sh': ActionType.CODE_EXECUTION,
        'exit': ActionType.DONE,
        'submit': ActionType.DONE,
    }

    def __init__(self, verbose: bool = False):
        """
        Initialize SWE-agent extractor.

        Args:
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single SWE-agent log file into a Trajectory.

        Args:
            log_path: Path to the log file (typically a .traj or .json file)

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"Log file not found: {log_path}")
            return None

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            return self._parse_log_data(log_data, log_path)
        except json.JSONDecodeError as e:
            self._log(f"JSON decode error in {log_path}: {e}")
            return None
        except Exception as e:
            self._log(f"Error parsing log {log_path}: {e}")
            return None

    def extract_from_directory(self, dir_path: str, pattern: str = "*.traj") -> List[Trajectory]:
        """
        Parse all SWE-agent log files in a directory.

        Args:
            dir_path: Path to directory containing SWE-agent logs
            pattern: Glob pattern for matching log files (default: "*.traj")

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        dir_path = Path(dir_path)
        log_files = list(dir_path.rglob(pattern))

        # Also check for .json files that might be trajectories
        if pattern == "*.traj":
            log_files.extend(list(dir_path.rglob("*.json")))

        self._log(f"Found {len(log_files)} log files in {dir_path}")

        trajectories = []
        for log_file in log_files:
            trajectory = self.extract_from_log(str(log_file))
            if trajectory:
                trajectories.append(trajectory)

        self._log(f"Successfully extracted {len(trajectories)} trajectories")
        return trajectories

    def extract_from_hf_record(self, record: Dict[str, Any]) -> Optional[Trajectory]:
        """
        Extract trajectory from a HuggingFace dataset record.

        The nebius/SWE-agent-trajectories dataset has records with fields like:
        - instance_id: unique identifier
        - trajectory: list of steps (or raw trajectory data)
        - model_name: name of the model used
        - etc.

        Args:
            record: A single record from the HuggingFace dataset

        Returns:
            Trajectory object or None if parsing fails
        """
        try:
            return self._parse_hf_record(record)
        except Exception as e:
            self._log(f"Error parsing HuggingFace record: {e}")
            return None

    def _parse_log_data(self, log_data: Dict[str, Any], log_path: str) -> Optional[Trajectory]:
        """
        Parse log data into a Trajectory.

        Args:
            log_data: Dictionary containing log data
            log_path: Path to the log file (for ID generation)

        Returns:
            Trajectory object or None if parsing fails
        """
        # Generate trajectory ID from log path
        trajectory_id = Path(log_path).stem

        # Extract trajectory steps
        steps = self._extract_steps(log_data)
        if not steps:
            self._log(f"No steps found in {log_path}")
            return None

        # Extract task description
        task_description = self._extract_task_description(log_data, steps)

        # Parse actions and observation chunks
        actions, observation_chunks = self._parse_steps(steps)

        # Create initial state
        initial_state = self._extract_initial_state(log_data, steps)

        # Extract metadata
        metadata = self._extract_metadata(log_data)

        # Determine success
        success = self._extract_success(log_data)

        return Trajectory(
            trajectory_id=trajectory_id,
            source="sweagent",
            task_description=task_description,
            is_attack=False,  # SWE-agent trajectories are benign software engineering tasks
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=None,
            metadata=metadata,
            success=success,
        )

    def _parse_hf_record(self, record: Dict[str, Any]) -> Optional[Trajectory]:
        """
        Parse a HuggingFace dataset record into a Trajectory.

        Args:
            record: HuggingFace dataset record

        Returns:
            Trajectory object or None
        """
        # Get instance ID
        instance_id = record.get('instance_id', '')
        if not instance_id:
            instance_id = f"sweagent_{hashlib.md5(str(record).encode()).hexdigest()[:8]}"

        # The trajectory field may contain the actual trajectory data
        traj_data = record.get('trajectory', record)

        # Extract steps from various possible formats
        steps = self._extract_steps(traj_data if isinstance(traj_data, dict) else record)

        if not steps:
            # Try to extract from raw JSON string if present
            if isinstance(traj_data, str):
                try:
                    parsed = json.loads(traj_data)
                    steps = self._extract_steps(parsed)
                except:
                    pass

        if not steps:
            return None

        # Extract task description
        task_description = self._extract_task_description(record, steps)

        # Parse actions and observation chunks
        actions, observation_chunks = self._parse_steps(steps)

        # Create initial state
        initial_state = self._extract_initial_state(record, steps)

        # Extract metadata
        metadata = self._extract_metadata(record)
        metadata['instance_id'] = instance_id
        if 'model_name' in record:
            metadata['model_name'] = record['model_name']

        # Determine success
        success = self._extract_success(record)

        return Trajectory(
            trajectory_id=instance_id,
            source="sweagent",
            task_description=task_description,
            is_attack=False,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=None,
            metadata=metadata,
            success=success,
        )

    def _extract_steps(self, log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract trajectory steps from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            List of step dictionaries
        """
        # Try various possible locations for trajectory data
        if 'trajectory' in log_data and isinstance(log_data['trajectory'], list):
            return log_data['trajectory']

        if 'history' in log_data and isinstance(log_data['history'], list):
            return log_data['history']

        if 'steps' in log_data and isinstance(log_data['steps'], list):
            return log_data['steps']

        # Check if the log_data itself is a list of steps
        if isinstance(log_data, list):
            return log_data

        return []

    def _extract_task_description(self, log_data: Dict[str, Any], steps: List[Dict[str, Any]]) -> str:
        """
        Extract task description from log data.

        Args:
            log_data: Log data dictionary
            steps: List of trajectory steps

        Returns:
            Task description string
        """
        # Try various keys where task description might be stored
        for key in ['problem_statement', 'issue_text', 'task', 'goal', 'instruction', 'issue']:
            if key in log_data and log_data[key]:
                return str(log_data[key])[:2000]  # Truncate long descriptions

        # Try to extract from the first step's query (system message)
        if steps and 'query' in steps[0]:
            query = steps[0]['query']
            if isinstance(query, list):
                for msg in query:
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if 'ISSUE' in content or 'problem' in content.lower():
                            # Extract the issue content
                            return self._extract_issue_from_message(content)

        return "Software engineering task"

    def _extract_issue_from_message(self, content: str) -> str:
        """Extract issue text from a message containing the task."""
        # Look for common patterns in SWE-agent prompts
        patterns = [
            r'ISSUE:\s*\n(.*?)(?:\n\n|\Z)',
            r'Problem Statement:\s*\n(.*?)(?:\n\n|\Z)',
            r'Issue:\s*\n(.*?)(?:\n\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:2000]

        # Return truncated content as fallback
        return content[:500] + "..." if len(content) > 500 else content

    def _parse_steps(self, steps: List[Dict[str, Any]]) -> Tuple[List[Action], List[ObservationChunk]]:
        """
        Parse steps into actions and observation chunks.

        Args:
            steps: List of step dictionaries

        Returns:
            Tuple of (actions list, observation_chunks list)
        """
        actions = []
        observation_chunks = []

        for i, step in enumerate(steps):
            # Extract action
            action = self._parse_step_action(step, i)
            if action:
                actions.append(action)

            # Extract observation as chunk
            obs_chunk = self._parse_step_observation(step, i)
            if obs_chunk:
                observation_chunks.append(obs_chunk)

                # Link action to observation chunk
                if action:
                    action.provenance = self._create_provenance([obs_chunk.chunk_id])

        return actions, observation_chunks

    def _parse_step_action(self, step: Dict[str, Any], action_id: int) -> Optional[Action]:
        """
        Parse a single step into an Action.

        Args:
            step: Step dictionary
            action_id: ID for this action

        Returns:
            Action object or None
        """
        # Extract action command
        action_str = step.get('action', '')
        if not action_str:
            return None

        # Determine action type from command
        action_type = self._classify_action(action_str)

        # Extract state info
        state = step.get('state', {})
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except:
                state = {}

        working_dir = state.get('working_dir', '')
        open_file = state.get('open_file', '')

        # Build context
        context = {
            'thought': step.get('thought', ''),
            'working_dir': working_dir,
            'open_file': open_file,
        }

        # Get observation as result
        observation = step.get('observation', '')
        result = observation[:10000] if observation else None  # Truncate very long observations

        # Extract domain from working directory (repository path)
        domain = self._extract_domain_from_path(working_dir)

        return Action(
            action_id=action_id,
            action_type=action_type,
            target=action_str.strip()[:500],  # Truncate long commands
            context=context,
            result=result,
            timestamp=None,  # SWE-agent doesn't typically have timestamps
            domain=domain,
            data_produced=[],
            data_consumed=[],
            raw_data={'step_index': action_id},
        )

    def _parse_step_observation(self, step: Dict[str, Any], step_id: int) -> Optional[ObservationChunk]:
        """
        Parse step observation into an ObservationChunk.

        Args:
            step: Step dictionary
            step_id: Step index

        Returns:
            ObservationChunk or None
        """
        observation = step.get('observation', '')
        if not observation:
            return None

        # Truncate very long observations but preserve meaningful content
        content = observation[:20000] if len(observation) > 20000 else observation

        # Determine source type from observation content
        source = self._classify_observation_source(observation, step.get('action', ''))

        # Extract state for domain info
        state = step.get('state', {})
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except:
                state = {}

        domain = self._extract_domain_from_path(state.get('working_dir', ''))

        return ObservationChunk(
            chunk_id=f"obs_{step_id}",
            content=content,
            source=source,
            domain=domain,
            metadata={
                'action': step.get('action', '')[:200],
                'step_id': step_id,
            }
        )

    def _classify_action(self, action_str: str) -> ActionType:
        """
        Classify an action string into an ActionType.

        Args:
            action_str: The action command string

        Returns:
            ActionType enum value
        """
        action_lower = action_str.lower().strip()

        # Check for exact command matches first
        first_word = action_lower.split()[0] if action_lower.split() else ''

        if first_word in self.COMMAND_ACTION_TYPES:
            return self.COMMAND_ACTION_TYPES[first_word]

        # Check for compound commands
        for cmd, action_type in self.COMMAND_ACTION_TYPES.items():
            if action_lower.startswith(cmd):
                return action_type

        # Check for file editing patterns
        if 'edit' in action_lower or 'write' in action_lower:
            return ActionType.WRITE

        # Check for reading patterns
        if 'show' in action_lower or 'view' in action_lower or 'print' in action_lower:
            return ActionType.READ

        # Check for navigation
        if action_lower.startswith('goto') or 'scroll' in action_lower:
            return ActionType.NAVIGATE

        # Check for submission/completion
        if 'submit' in action_lower or 'exit' in action_lower or 'done' in action_lower:
            return ActionType.DONE

        return ActionType.TOOL_CALL

    def _classify_observation_source(self, observation: str, action: str) -> str:
        """
        Classify the source type of an observation.

        Args:
            observation: The observation content
            action: The action that produced the observation

        Returns:
            Source type string
        """
        action_lower = action.lower() if action else ''
        obs_lower = observation.lower()[:500]  # Check first 500 chars

        # Check for terminal/shell output indicators
        if any(x in obs_lower for x in ['error:', 'warning:', 'exception', 'traceback']):
            return 'terminal_error'

        if 'git' in action_lower:
            return 'git_output'

        if any(x in action_lower for x in ['python', 'pytest', 'pip']):
            return 'python_output'

        if any(x in action_lower for x in ['cat', 'head', 'tail', 'less', 'more']):
            return 'file_content'

        if any(x in action_lower for x in ['ls', 'find', 'tree']):
            return 'directory_listing'

        if 'diff' in action_lower:
            return 'diff_output'

        return 'terminal_output'

    def _extract_domain_from_path(self, path: str) -> str:
        """
        Extract a domain identifier from a file path.

        For SWE-agent, this is typically the repository name.

        Args:
            path: File path string

        Returns:
            Domain string
        """
        if not path:
            return 'sweagent.unknown'

        # Try to extract repository name from path
        # Common patterns: /repo/name, /home/user/repo, etc.
        parts = path.strip('/').split('/')

        # Look for common repo directory names
        for i, part in enumerate(parts):
            if part in ['repos', 'repositories', 'workspace', 'testbed']:
                if i + 1 < len(parts):
                    return f"sweagent.{parts[i+1]}"

        # Use the last meaningful directory component
        for part in reversed(parts):
            if part and part not in ['home', 'user', 'root', 'tmp', 'var']:
                return f"sweagent.{part}"

        return 'sweagent.repo'

    def _create_provenance(self, chunk_ids: List[str]) -> 'ProvenanceData':
        """Create provenance data linking to observation chunks."""
        from causaltrace.models.trajectory import ProvenanceData
        return ProvenanceData(
            observation_chunks=chunk_ids,
            confidence_scores={cid: 1.0 for cid in chunk_ids},
            attribution_method='temporal_sequence',
            is_untrusted=False,
        )

    def _extract_initial_state(self, log_data: Dict[str, Any], steps: List[Dict[str, Any]]) -> State:
        """
        Extract initial state from log data.

        Args:
            log_data: Log data dictionary
            steps: List of trajectory steps

        Returns:
            State object
        """
        state = State()

        # Try to get initial working directory from first step
        if steps and 'state' in steps[0]:
            step_state = steps[0]['state']
            if isinstance(step_state, str):
                try:
                    step_state = json.loads(step_state)
                except:
                    step_state = {}

            if 'working_dir' in step_state:
                state.current_url = step_state['working_dir']

        # Set authenticated to the SWE-agent environment
        state.authenticated_services.append('sweagent')

        # Store environment info
        if 'environment' in log_data:
            state.accumulated_data['environment'] = log_data['environment']

        if 'instance_id' in log_data:
            state.accumulated_data['instance_id'] = log_data['instance_id']

        return state

    def _extract_metadata(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Common metadata fields
        metadata_keys = [
            'environment', 'instance_id', 'model_name', 'model',
            'repo', 'base_commit', 'version', 'config',
            'exit_status', 'submission', 'test_result'
        ]

        for key in metadata_keys:
            if key in log_data:
                metadata[key] = log_data[key]

        return metadata

    def _extract_success(self, log_data: Dict[str, Any]) -> Optional[bool]:
        """
        Determine if the task was successful.

        Args:
            log_data: Log data dictionary

        Returns:
            True if successful, False if failed, None if unknown
        """
        # Check for explicit success indicators
        if 'exit_status' in log_data:
            return log_data['exit_status'] == 'submitted'

        if 'resolved' in log_data:
            return bool(log_data['resolved'])

        if 'test_result' in log_data:
            result = log_data['test_result']
            if isinstance(result, dict):
                return result.get('passed', False)
            return bool(result)

        if 'success' in log_data:
            return bool(log_data['success'])

        return None
