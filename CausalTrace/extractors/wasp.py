"""
WASP trajectory extractor.

This module extracts trajectories from WASP (Web Agent Security against Prompt Injection)
benchmark logs. WASP uses VisualWebArena's scaffolding and stores trajectories in
a specific format.

WASP Format Notes:
- Configuration stored in experiment_config.raw.json
- Results stored in /tmp/run_*.json files
- Trajectories include step-by-step actions with browser states
- Attack scenarios involve prompt injection via web content
"""

import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from causaltrace.models import Trajectory, Action, ActionType, State
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files, extract_domain


class WASPExtractor(BaseExtractor):
    """
    Extract trajectories from WASP benchmark logs.

    WASP logs contain information about:
    - User goals (benign tasks)
    - Prompt injections (attack instructions)
    - Step-by-step agent actions
    - Attack success metrics (ASR, attacker utility, user utility)
    """

    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize WASP extractor.

        Args:
            config_path: Path to experiment_config.raw.json (optional)
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.config = None
        if config_path and Path(config_path).exists():
            self.config = read_json(config_path)
            self._log(f"Loaded config from {config_path}")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single WASP log file into a Trajectory.

        Args:
            log_path: Path to the log file (typically a JSON file)

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"Log file not found: {log_path}")
            return None

        try:
            log_data = read_json(log_path)
            return self._parse_log_data(log_data, log_path)
        except Exception as e:
            self._log(f"Error parsing log {log_path}: {e}")
            return None

    def extract_from_directory(self, dir_path: str, pattern: str = "*.json") -> List[Trajectory]:
        """
        Parse all WASP log files in a directory.

        Args:
            dir_path: Path to directory containing WASP logs
            pattern: Glob pattern for matching log files (default: "*.json")

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        log_files = find_files(dir_path, pattern, recursive=True)
        self._log(f"Found {len(log_files)} log files in {dir_path}")

        trajectories = []
        for log_file in log_files:
            trajectory = self.extract_from_log(log_file)
            if trajectory:
                trajectories.append(trajectory)

        self._log(f"Successfully extracted {len(trajectories)} trajectories")
        return trajectories

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

        # Determine if this is an attack trajectory
        # In WASP, attacks are indicated by prompt injections
        is_attack = self._is_attack_trajectory(log_data)

        # Extract task description
        task_description = self._extract_task_description(log_data)

        # Parse actions
        actions = self._parse_actions(log_data)

        # Create initial state
        initial_state = self._extract_initial_state(log_data)

        # Create final state if available
        final_state = self._extract_final_state(log_data)

        # Extract metadata
        metadata = self._extract_metadata(log_data)

        # Determine success
        success = self._extract_success(log_data)

        return Trajectory(
            trajectory_id=trajectory_id,
            source="wasp",
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=success,
        )

    def _is_attack_trajectory(self, log_data: Dict[str, Any]) -> bool:
        """
        Determine if trajectory contains an attack.

        Args:
            log_data: Log data dictionary

        Returns:
            True if this is an attack trajectory
        """
        # Check for attack indicators in the log
        if "compromised" in log_data:
            return bool(log_data["compromised"])

        if "exfil" in log_data:
            return bool(log_data["exfil"])

        # Check for injection-related keys
        if "injection_format" in log_data or "prompt_injection" in log_data:
            return True

        return False

    def _extract_task_description(self, log_data: Dict[str, Any]) -> str:
        """
        Extract task description from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            Task description string
        """
        # Try various keys where task description might be stored
        for key in ["instruction", "free_form_name", "task", "goal", "user_goal"]:
            if key in log_data and log_data[key]:
                return str(log_data[key])

        return "Unknown task"

    def _parse_actions(self, log_data: Dict[str, Any]) -> List[Action]:
        """
        Parse actions from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            List of Action objects
        """
        actions = []

        # WASP logs may contain actions in various formats
        # We'll try to extract them from common locations
        if "actions" in log_data and isinstance(log_data["actions"], list):
            for i, action_data in enumerate(log_data["actions"]):
                action = self._parse_single_action(action_data, i)
                if action:
                    actions.append(action)

        # If no actions found, try to infer from other fields
        if not actions:
            actions = self._infer_actions_from_log(log_data)

        return actions

    def _parse_single_action(self, action_data: Dict[str, Any], action_id: int) -> Optional[Action]:
        """
        Parse a single action from action data.

        Args:
            action_data: Dictionary containing action information
            action_id: ID for this action

        Returns:
            Action object or None
        """
        # Extract action type
        action_type_str = action_data.get("action_type", action_data.get("type", "unknown"))
        action_type = ActionType.from_string(action_type_str)

        # Extract target
        target = action_data.get("target", action_data.get("url", ""))

        # Extract context
        context = {
            k: v for k, v in action_data.items()
            if k not in ["action_type", "type", "target", "url", "result", "timestamp"]
        }

        # Extract result
        result = action_data.get("result", action_data.get("observation"))

        # Extract timestamp
        timestamp = action_data.get("timestamp")

        # Extract domain
        domain = None
        if target:
            domain = extract_domain(target)

        return Action(
            action_id=action_id,
            action_type=action_type,
            target=str(target),
            context=context,
            result=str(result) if result else None,
            timestamp=timestamp,
            domain=domain,
            raw_data=action_data,
        )

    def _infer_actions_from_log(self, log_data: Dict[str, Any]) -> List[Action]:
        """
        Infer actions from log data when explicit action list is not available.

        Args:
            log_data: Log data dictionary

        Returns:
            List of Action objects
        """
        actions = []

        # Check for action_url (indicates navigation action)
        if "action_url" in log_data:
            action = Action(
                action_id=0,
                action_type=ActionType.NAVIGATE,
                target=log_data["action_url"],
                context={"inferred": True},
                domain=extract_domain(log_data["action_url"]),
            )
            actions.append(action)

        # Check for instruction (indicates task execution)
        if "instruction" in log_data:
            action = Action(
                action_id=len(actions),
                action_type=ActionType.TOOL_CALL,
                target="execute_instruction",
                context={"instruction": log_data["instruction"], "inferred": True},
            )
            actions.append(action)

        return actions

    def _extract_initial_state(self, log_data: Dict[str, Any]) -> State:
        """
        Extract initial state from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            State object
        """
        state = State()

        # Extract authenticated services from parameters
        if "parameters" in log_data:
            params = log_data["parameters"]
            if "user_username" in params and "user_password" in params:
                # User is authenticated to the service
                if "environment" in log_data:
                    state.authenticated_services.append(log_data["environment"])

        # Extract initial URL
        if "action_url" in log_data:
            state.current_url = log_data["action_url"]

        return state

    def _extract_final_state(self, log_data: Dict[str, Any]) -> Optional[State]:
        """
        Extract final state from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            State object or None
        """
        # WASP logs may not always contain final state
        # We'll try to infer it from evaluation results
        state = State()

        # Check if evaluation passed
        if "eval" in log_data:
            eval_data = log_data["eval"]
            if "program_html" in eval_data:
                # Agent completed some HTML-based evaluation
                for eval_item in eval_data["program_html"]:
                    if "url" in eval_item:
                        state.current_url = eval_item["url"]

        return state if state.current_url else None

    def _extract_metadata(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Store environment
        if "environment" in log_data:
            metadata["environment"] = log_data["environment"]

        # Store setup and cleanup functions
        if "setup_fn" in log_data:
            metadata["setup_fn"] = log_data["setup_fn"]
        if "cleanup_fn" in log_data:
            metadata["cleanup_fn"] = log_data["cleanup_fn"]

        # Store evaluation info
        if "eval" in log_data:
            metadata["eval"] = log_data["eval"]

        # Store parameters
        if "parameters" in log_data:
            metadata["parameters"] = log_data["parameters"]

        # Store exfiltration flag
        if "exfil" in log_data:
            metadata["exfil"] = log_data["exfil"]

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
        if "success" in log_data:
            return bool(log_data["success"])

        # For attack trajectories, check if attack succeeded
        if self._is_attack_trajectory(log_data):
            if "compromised" in log_data:
                return bool(log_data["compromised"])

        return None
