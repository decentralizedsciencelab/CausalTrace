"""
SafeArena trajectory extractor.

This module extracts trajectories from SafeArena benchmark logs.
SafeArena uses the AgentLab framework built on browsergym and evaluates
agents on both safe and harmful tasks.

SafeArena Format Notes:
- Based on AgentLab/BrowserGym framework
- Tasks are either "safe" or "harm" (250 of each)
- 5 harm categories: Misinformation, Illegal Activity, Harassment, Cybercrime, Social Bias
- Results typically stored in ~/agentlab_results/
- Uses Study and ExpResult objects for logging
"""

import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from causaltrace.models import Trajectory, Action, ActionType, State
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files, extract_domain


class SafeArenaExtractor(BaseExtractor):
    """
    Extract trajectories from SafeArena benchmark logs.

    SafeArena logs contain:
    - Task descriptions (safe or harmful)
    - Agent actions and observations
    - Success/failure metrics
    - Jailbreak attempt information
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize SafeArena extractor.

        Args:
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single SafeArena log file into a Trajectory.

        Args:
            log_path: Path to the log file

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
        Parse all SafeArena log files in a directory.

        Args:
            dir_path: Path to directory containing SafeArena logs
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

        # Determine if this is an attack/harmful trajectory
        is_attack = self._is_harmful_trajectory(log_data)

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
            source="safearena",
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=success,
        )

    def _is_harmful_trajectory(self, log_data: Dict[str, Any]) -> bool:
        """
        Determine if trajectory is harmful.

        Args:
            log_data: Log data dictionary

        Returns:
            True if this is a harmful trajectory
        """
        # Check for harm category
        if "harm_category" in log_data:
            return True

        # Check for task type
        if "task_type" in log_data:
            return log_data["task_type"] == "harm"

        # Check for jailbreak indicator
        if "jailbreak" in log_data:
            return bool(log_data["jailbreak"])

        # Check for harmful keywords in metadata
        if "metadata" in log_data:
            metadata = log_data["metadata"]
            harmful_keywords = ["misinformation", "illegal", "harassment", "cybercrime", "bias"]
            for keyword in harmful_keywords:
                if any(keyword in str(v).lower() for v in metadata.values()):
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
        for key in ["task", "goal", "instruction", "description", "task_description"]:
            if key in log_data and log_data[key]:
                return str(log_data[key])

        # Try nested metadata
        if "metadata" in log_data:
            metadata = log_data["metadata"]
            for key in ["task", "goal", "instruction"]:
                if key in metadata:
                    return str(metadata[key])

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

        # AgentLab/BrowserGym format typically has a steps or actions list
        steps_keys = ["steps", "actions", "trajectory", "history"]

        for key in steps_keys:
            if key in log_data and isinstance(log_data[key], list):
                for i, step_data in enumerate(log_data[key]):
                    action = self._parse_single_action(step_data, i)
                    if action:
                        actions.append(action)
                break

        # If no actions found, try to infer from log
        if not actions:
            actions = self._infer_actions_from_log(log_data)

        return actions

    def _parse_single_action(self, step_data: Dict[str, Any], action_id: int) -> Optional[Action]:
        """
        Parse a single action from step data.

        Args:
            step_data: Dictionary containing step/action information
            action_id: ID for this action

        Returns:
            Action object or None
        """
        # BrowserGym actions typically have an 'action' field
        action_data = step_data.get("action", step_data)

        # Extract action type from various possible fields
        action_type_str = "unknown"
        if isinstance(action_data, dict):
            action_type_str = action_data.get("intent", action_data.get("type", "unknown"))
        elif isinstance(action_data, str):
            action_type_str = action_data

        action_type = ActionType.from_string(str(action_type_str))

        # Extract target
        target = ""
        if isinstance(action_data, dict):
            target = action_data.get("url", action_data.get("element", action_data.get("target", "")))

        # Extract observation/result
        result = step_data.get("observation", step_data.get("result"))

        # Extract context from step data
        context = {
            k: v for k, v in step_data.items()
            if k not in ["action", "observation", "result", "timestamp"]
        }

        # Add action-specific context
        if isinstance(action_data, dict):
            context["action_details"] = {
                k: v for k, v in action_data.items()
                if k not in ["intent", "type", "url", "element", "target"]
            }

        # Extract timestamp
        timestamp = step_data.get("timestamp", step_data.get("time"))

        # Extract domain
        domain = None
        if target:
            domain = extract_domain(str(target))

        return Action(
            action_id=action_id,
            action_type=action_type,
            target=str(target),
            context=context,
            result=str(result) if result else None,
            timestamp=timestamp,
            domain=domain,
            raw_data=step_data,
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

        # Check for task execution
        if "task" in log_data or "goal" in log_data:
            task = log_data.get("task", log_data.get("goal"))
            action = Action(
                action_id=0,
                action_type=ActionType.TOOL_CALL,
                target="execute_task",
                context={"task": task, "inferred": True},
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

        # Extract initial environment info
        if "env_info" in log_data:
            env_info = log_data["env_info"]
            if "authenticated_services" in env_info:
                state.authenticated_services = env_info["authenticated_services"]

        # Extract starting URL
        if "start_url" in log_data:
            state.current_url = log_data["start_url"]

        return state

    def _extract_final_state(self, log_data: Dict[str, Any]) -> Optional[State]:
        """
        Extract final state from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            State object or None
        """
        # Try to extract final state from steps
        steps_keys = ["steps", "actions", "trajectory"]

        for key in steps_keys:
            if key in log_data and isinstance(log_data[key], list) and log_data[key]:
                last_step = log_data[key][-1]

                state = State()

                # Extract final URL from last observation
                if "observation" in last_step:
                    obs = last_step["observation"]
                    if isinstance(obs, dict) and "url" in obs:
                        state.current_url = obs["url"]

                return state if state.current_url else None

        return None

    def _extract_metadata(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Store harm category if present
        if "harm_category" in log_data:
            metadata["harm_category"] = log_data["harm_category"]

        # Store task type
        if "task_type" in log_data:
            metadata["task_type"] = log_data["task_type"]

        # Store jailbreak info
        if "jailbreak" in log_data:
            metadata["jailbreak"] = log_data["jailbreak"]

        # Store agent info
        if "agent" in log_data:
            metadata["agent"] = log_data["agent"]

        # Store model info
        if "model" in log_data:
            metadata["model"] = log_data["model"]

        # Store any additional metadata field
        if "metadata" in log_data:
            metadata.update(log_data["metadata"])

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
        for key in ["success", "completed", "task_success"]:
            if key in log_data:
                return bool(log_data[key])

        # Check for reward (positive reward = success)
        if "reward" in log_data:
            return log_data["reward"] > 0

        # Check for completion status
        if "status" in log_data:
            status = str(log_data["status"]).lower()
            if "success" in status or "complete" in status:
                return True
            if "fail" in status or "error" in status:
                return False

        return None
