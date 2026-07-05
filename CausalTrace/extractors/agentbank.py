"""
AgentBank extractor for parsing agent trajectories from the AgentBank dataset.

AgentBank is a large-scale trajectory dataset from Tsinghua University (THUNLP)
containing 50k+ high-quality interaction trajectories across 16 tasks.

Dataset: https://huggingface.co/datasets/Solaris99/AgentBank
Paper: https://arxiv.org/abs/2410.07706 (EMNLP 2024)

AgentBank uses a chatbot-style format with alternating human/agent turns:
- Human turns: Task description + Observation + Action prompt
- Agent turns: Thought (reasoning) + Action (to execute)
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from causaltrace.extractors.base import BaseExtractor
from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData


class AgentBankExtractor(BaseExtractor):
    """
    Extractor for AgentBank trajectories.

    AgentBank format:
    {
      "id": "alfred_1",
      "conversations": [
        {"from": "human", "value": "Task: ...\nObs: ...\nWhat is your next step?"},
        {"from": "gpt", "value": "Thought: ...\nAction: move_ahead"},
        ...
      ],
      "meta_information": {...}  // Optional
    }
    """

    # Mapping of AgentBank action patterns to CausalTrace ActionTypes
    ACTION_TYPE_MAPPING = {
        # Navigation
        "move_ahead": ActionType.NAVIGATE,
        "turn_left": ActionType.NAVIGATE,
        "turn_right": ActionType.NAVIGATE,
        "go_to": ActionType.NAVIGATE,
        "goto": ActionType.NAVIGATE,
        "navigate": ActionType.NAVIGATE,
        "click": ActionType.CLICK,

        # Manipulation
        "pick_up": ActionType.EXTRACT,
        "put": ActionType.WRITE,
        "open": ActionType.CLICK,
        "close": ActionType.CLICK,
        "toggle": ActionType.CLICK,
        "slice": ActionType.TOOL_CALL,

        # Information
        "look": ActionType.READ,
        "examine": ActionType.READ,
        "inventory": ActionType.READ,
        "read": ActionType.READ,

        # Web actions (from webshop, webarena, mind2web)
        "search": ActionType.TYPE,
        "type": ActionType.TYPE,
        "select": ActionType.CLICK,
        "submit": ActionType.SUBMIT,
        "scroll_up": ActionType.SCROLL,
        "scroll_down": ActionType.SCROLL,

        # Code/Tool actions (from apps, humaneval, mbpp, intercode)
        "execute": ActionType.TOOL_CALL,
        "run": ActionType.TOOL_CALL,
        "bash": ActionType.TOOL_CALL,
        "sql": ActionType.TOOL_CALL,
        "python": ActionType.TOOL_CALL,

        # Default
        "finish": ActionType.SUBMIT,
        "stop": ActionType.SUBMIT,
    }

    def __init__(self, verbose: bool = False, include_thoughts: bool = True):
        """
        Initialize the AgentBank extractor.

        Args:
            verbose: Whether to print verbose output
            include_thoughts: Whether to include agent reasoning in action context
        """
        super().__init__(verbose)
        self.include_thoughts = include_thoughts

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single AgentBank JSON file into a Trajectory.

        Args:
            log_path: Path to the AgentBank JSON file

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"File not found: {log_path}")
            return None

        try:
            with open(log_path, 'r') as f:
                data = json.load(f)

            return self._parse_agentbank_data(data, Path(log_path).stem)

        except Exception as e:
            self._log(f"Error parsing {log_path}: {e}")
            return None

    def extract_from_directory(self, dir_path: str, pattern: str = "*.json") -> List[Trajectory]:
        """
        Parse all AgentBank JSON files in a directory.

        Args:
            dir_path: Path to directory containing AgentBank JSON files
            pattern: Glob pattern for matching files (default: "*.json")

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        trajectories = []
        dir_path_obj = Path(dir_path)

        for json_file in dir_path_obj.glob(pattern):
            self._log(f"Processing {json_file}")
            trajectory = self.extract_from_log(str(json_file))
            if trajectory:
                trajectories.append(trajectory)

        self._log(f"Extracted {len(trajectories)} trajectories from {dir_path}")
        return trajectories

    def _parse_agentbank_data(self, data: Dict[str, Any], filename: str) -> Optional[Trajectory]:
        """
        Parse raw AgentBank data into a Trajectory.

        Args:
            data: Raw AgentBank data dictionary
            filename: Source filename for trajectory ID

        Returns:
            Trajectory object or None if parsing fails
        """
        trajectory_id = data.get("id", filename)
        conversations = data.get("conversations", [])
        meta_info = data.get("meta_information", {})

        if not conversations:
            self._log(f"No conversations found in trajectory {trajectory_id}")
            return None

        # Extract task description from first human turn
        task_description = self._extract_task_description(conversations)

        # Parse actions from conversation turns
        actions = []
        observation_chunks = []
        action_idx = 0
        chunk_idx = 0

        for i in range(len(conversations)):
            turn = conversations[i]

            if turn["from"] == "human":
                # Human turn contains observation
                obs_text = turn["value"]

                # Extract observation (skip task description in first turn)
                if i > 0 or not task_description:
                    chunk = self._create_observation_chunk(
                        chunk_idx=chunk_idx,
                        content=obs_text,
                        trajectory_id=trajectory_id
                    )
                    observation_chunks.append(chunk)
                    chunk_idx += 1

            elif turn["from"] == "gpt":
                # Agent turn contains thought + action
                agent_response = turn["value"]

                # Look ahead to get the next observation (result of this action)
                next_obs = None
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "human":
                    next_obs = conversations[i + 1]["value"]

                # Parse action from agent response
                action = self._parse_agent_action(
                    action_idx=action_idx,
                    agent_response=agent_response,
                    next_observation=next_obs,
                    trajectory_id=trajectory_id,
                    previous_chunk_id=f"{trajectory_id}_chunk_{chunk_idx - 1}" if chunk_idx > 0 else None
                )

                if action:
                    actions.append(action)
                    action_idx += 1

        # Determine if this is an attack trajectory
        # AgentBank is primarily benign training data, so default to False
        # Could be enhanced with keyword detection for adversarial examples
        is_attack = self._detect_attack_indicators(task_description, actions)

        # Create initial and final states
        initial_state = State()
        final_state = State()

        # Extract dataset information from metadata
        dataset = meta_info.get("dataset", "unknown")
        environment = meta_info.get("environment", "unknown")

        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            source="agentbank",
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            initial_state=initial_state,
            observation_chunks=observation_chunks,
            final_state=final_state,
            metadata={
                "dataset": dataset,
                "environment": environment,
                "meta_information": meta_info,
                "num_turns": len(conversations),
                "filename": filename
            },
            success=None  # AgentBank doesn't explicitly mark success
        )

        return trajectory

    def _extract_task_description(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Extract task description from first human turn.

        Args:
            conversations: List of conversation turns

        Returns:
            Task description string
        """
        if not conversations or conversations[0]["from"] != "human":
            return "Unknown task"

        first_turn = conversations[0]["value"]

        # Look for "Task: " prefix
        task_match = re.search(r'Task:\s*(.+?)(?:\n|$)', first_turn, re.IGNORECASE)
        if task_match:
            return task_match.group(1).strip()

        # Otherwise return first line
        first_line = first_turn.split('\n')[0]
        return first_line.strip()

    def _create_observation_chunk(self, chunk_idx: int, content: str, trajectory_id: str) -> ObservationChunk:
        """
        Create an observation chunk from text.

        Args:
            chunk_idx: Index of this chunk
            content: Observation content
            trajectory_id: Parent trajectory ID

        Returns:
            ObservationChunk object
        """
        # Try to extract domain from observation
        domain = self._extract_domain_from_text(content)

        return ObservationChunk(
            chunk_id=f"{trajectory_id}_chunk_{chunk_idx}",
            content=content,
            source="environment",
            domain=domain,
            metadata={"type": "observation"}
        )

    def _parse_agent_action(
        self,
        action_idx: int,
        agent_response: str,
        next_observation: Optional[str],
        trajectory_id: str,
        previous_chunk_id: Optional[str]
    ) -> Optional[Action]:
        """
        Parse action from agent response.

        Agent responses typically follow format:
        "Thought: <reasoning>\nAction: <action_name>[<target>]"

        Args:
            action_idx: Index of this action
            agent_response: Full agent response text
            next_observation: The observation received after executing this action
            trajectory_id: Parent trajectory ID
            previous_chunk_id: ID of observation chunk that influenced this action

        Returns:
            Action object or None if parsing fails
        """
        # Extract thought and action
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', agent_response, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r'Action:\s*(.+?)$', agent_response, re.DOTALL | re.IGNORECASE)

        if not action_match:
            self._log(f"No action found in response: {agent_response[:100]}")
            return None

        action_str = action_match.group(1).strip()
        thought = thought_match.group(1).strip() if thought_match else ""

        # Parse action name and target
        # Format: action_name[target] or just action_name
        action_parts = re.match(r'(\w+)(?:\[([^\]]+)\])?', action_str)
        if not action_parts:
            self._log(f"Could not parse action: {action_str}")
            return None

        action_name = action_parts.group(1).lower()
        target = action_parts.group(2) if action_parts.group(2) else action_str

        # Map to ActionType
        action_type = self._map_action_type(action_name)

        # Extract domain from target if it's a URL
        domain = self._extract_domain(target)

        # Create provenance data
        provenance = None
        if previous_chunk_id:
            provenance = ProvenanceData(
                observation_chunks=[previous_chunk_id],
                confidence_scores={previous_chunk_id: 1.0},
                attribution_method="sequential_heuristic"
            )

        # Build context
        context = {"raw_response": agent_response}
        if self.include_thoughts and thought:
            context["thought"] = thought

        action = Action(
            action_id=action_idx,
            action_type=action_type,
            target=target,
            context=context,
            result=next_observation,
            timestamp=float(action_idx),  # Use action index as timestamp
            domain=domain,
            provenance=provenance,
            raw_data={
                "action_str": action_str,
                "action_name": action_name,
                "trajectory_id": trajectory_id
            }
        )

        return action

    def _map_action_type(self, action_name: str) -> ActionType:
        """
        Map AgentBank action name to CausalTrace ActionType.

        Args:
            action_name: Action name from AgentBank (e.g., "move_ahead", "pick_up")

        Returns:
            Corresponding ActionType
        """
        action_name_lower = action_name.lower()

        # Try exact match
        if action_name_lower in self.ACTION_TYPE_MAPPING:
            return self.ACTION_TYPE_MAPPING[action_name_lower]

        # Try substring match
        for key, action_type in self.ACTION_TYPE_MAPPING.items():
            if key in action_name_lower or action_name_lower in key:
                return action_type

        # Default to TOOL_CALL for unknown actions
        return ActionType.TOOL_CALL

    def _extract_domain(self, target: str) -> Optional[str]:
        """
        Extract domain from target if it's a URL.

        Args:
            target: Action target (may be URL, selector, object ID, etc.)

        Returns:
            Domain string or None
        """
        if not target:
            return None

        # Check if target looks like a URL
        if target.startswith("http://") or target.startswith("https://"):
            try:
                parsed = urlparse(target)
                return parsed.netloc
            except:
                return None

        return None

    def _extract_domain_from_text(self, text: str) -> Optional[str]:
        """
        Extract domain from text containing URLs.

        Args:
            text: Text that may contain URLs

        Returns:
            Domain string or None
        """
        # Look for URLs in text
        url_pattern = r'https?://([^\s/]+)'
        match = re.search(url_pattern, text)
        if match:
            return match.group(1)
        return None

    def _detect_attack_indicators(self, task_description: str, actions: List[Action]) -> bool:
        """
        Detect if trajectory contains attack indicators.

        AgentBank is primarily benign, but may contain adversarial examples
        in certain subsets (e.g., math problems with misleading info).

        Args:
            task_description: Task description text
            actions: List of actions in trajectory

        Returns:
            True if attack indicators detected, False otherwise
        """
        # Keywords that might indicate adversarial content
        attack_keywords = [
            "ignore", "disregard", "instead", "actually", "new instruction",
            "override", "bypass", "exploit", "inject", "malicious"
        ]

        # Check task description
        task_lower = task_description.lower()
        for keyword in attack_keywords:
            if keyword in task_lower:
                return True

        # Check for suspicious action patterns
        # (e.g., many failed actions, cross-domain jumps)
        if len(actions) > 50:  # Very long trajectories might be confused
            return True

        # Check for multiple domain visits (could be data exfiltration)
        domains = set()
        for action in actions:
            if action.domain:
                domains.add(action.domain)

        if len(domains) > 3:  # More than 3 domains is suspicious
            return True

        return False
