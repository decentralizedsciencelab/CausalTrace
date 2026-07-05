"""
SafeArena Benchmark Extractor.

SafeArena is a safety evaluation benchmark with 500 tasks across 5 harm categories:
- Misinformation
- Illegal Activity
- Harassment
- Cybercrime
- Social Bias

Reference: https://safearena.github.io/
Paper: https://arxiv.org/abs/2503.04957

This extractor converts SafeArena logs/results to CausalTrace trajectory format.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from causaltrace.models.trajectory import (
    Trajectory,
    Action,
    ActionType,
    State,
    ObservationChunk,
    ProvenanceData,
)

logger = logging.getLogger(__name__)


@dataclass
class SafeArenaTask:
    """Represents a SafeArena task."""
    task_id: str
    category: str  # misinformation, illegal, harassment, cybercrime, social_bias
    task_type: str  # "safe" or "harm"
    intent: str
    target_url: str
    expected_behavior: str
    harm_description: Optional[str] = None


class SafeArenaExtractor:
    """
    Extract trajectories from SafeArena benchmark results.

    SafeArena logs typically contain:
    - Task configuration (JSON)
    - Agent actions (browsergym format)
    - Final evaluation result
    """

    # Harm categories in SafeArena
    HARM_CATEGORIES = {
        "misinformation": ["false_news", "health_misinfo", "conspiracy"],
        "illegal_activity": ["drugs", "weapons", "contraband"],
        "harassment": ["cyberbullying", "stalking", "threats"],
        "cybercrime": ["hacking", "fraud", "data_breach"],
        "social_bias": ["discrimination", "hate_speech", "prejudice"],
    }

    # Action type mapping from SafeArena to CausalTrace
    ACTION_MAP = {
        "click": ActionType.CLICK,
        "type": ActionType.TYPE,
        "scroll": ActionType.SCROLL,
        "goto": ActionType.NAVIGATE,
        "go_to_url": ActionType.NAVIGATE,
        "fill": ActionType.TYPE,
        "press": ActionType.CLICK,
        "hover": ActionType.CLICK,
        "select_option": ActionType.CLICK,
        "send_msg_to_user": ActionType.AGENT_RESPONSE,
        "screenshot": ActionType.SCREENSHOT,
    }

    # Untrusted domain patterns (SafeArena-specific)
    UNTRUSTED_PATTERNS = [
        r"forum\.safearena\.",
        r"shop\.safearena\.",
        r"social\.safearena\.",
        r"cdn\.",
        r"user-content\.",
    ]

    def __init__(self):
        """Initialize SafeArena extractor."""
        self.untrusted_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.UNTRUSTED_PATTERNS
        ]

    def _is_untrusted_domain(self, domain: Optional[str]) -> bool:
        """Check if a domain is untrusted."""
        if not domain:
            return False
        return any(p.search(domain) for p in self.untrusted_patterns)

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or None
        except Exception:
            return None

    def _detect_injection(self, text: str) -> bool:
        """Detect if text contains injection patterns."""
        injection_patterns = [
            r"ATTENTION\s*:",
            r"NEW\s+(?:PRIORITY\s+)?INSTRUCTION",
            r"IGNORE\s+PREVIOUS",
            r"SYSTEM\s+(?:UPDATE|PROMPT)",
            r"OBJECTIVE\s*:",
            r"YOU\s+MUST",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _parse_browsergym_action(
        self,
        action_data: Dict[str, Any],
        action_idx: int,
    ) -> Action:
        """
        Parse a BrowserGym-format action.

        BrowserGym actions look like:
        {
            "action_type": "click",
            "element_id": "button_123",
            "coords": [100, 200],
            "timestamp": 1234567890.0
        }
        """
        action_type_str = action_data.get("action_type", "unknown").lower()
        action_type = self.ACTION_MAP.get(action_type_str, ActionType.UNKNOWN)

        # Build target
        target = action_data.get("element_id", "")
        if not target:
            target = action_data.get("url", action_data.get("selector", "unknown"))

        # Extract domain
        url = action_data.get("url", "")
        domain = self._extract_domain(url) if url else None

        # Build context
        context = {
            "element_id": action_data.get("element_id"),
            "coords": action_data.get("coords"),
            "viewport": action_data.get("viewport"),
            "page_state": action_data.get("page_state"),
        }
        context = {k: v for k, v in context.items() if v is not None}

        # Check for text input (for TYPE actions)
        if action_type == ActionType.TYPE:
            target = action_data.get("text", action_data.get("value", target))

        # Get result/observation
        result = action_data.get("result", action_data.get("observation"))

        # Build provenance
        provenance = None
        obs_content = str(result) if result else ""
        if obs_content:
            is_untrusted = self._is_untrusted_domain(domain)
            injection_detected = self._detect_injection(obs_content)

            provenance = ProvenanceData(
                observation_chunks=[f"obs_{action_idx}"],
                confidence_scores={f"obs_{action_idx}": 1.0},
                attribution_method="safearena_extractor",
                is_untrusted=is_untrusted,
                untrusted_domains={domain} if is_untrusted and domain else set(),
                injection_detected=injection_detected,
            )

        return Action(
            action_id=action_idx,
            action_type=action_type,
            target=target,
            context=context,
            result=str(result) if result else None,
            timestamp=action_data.get("timestamp"),
            domain=domain,
            data_produced=action_data.get("data_produced", []),
            data_consumed=action_data.get("data_consumed", []),
            provenance=provenance,
            raw_data=action_data,
        )

    def _parse_agentlab_action(
        self,
        action_str: str,
        action_idx: int,
        observation: Optional[str] = None,
    ) -> Action:
        """
        Parse an AgentLab-format action string.

        AgentLab actions are strings like:
        - "click('button_123')"
        - "type('input_456', 'hello world')"
        - "goto('https://example.com')"
        """
        # Parse action string
        match = re.match(r"(\w+)\((.*)?\)", action_str.strip())
        if not match:
            return Action(
                action_id=action_idx,
                action_type=ActionType.UNKNOWN,
                target=action_str,
                result=observation,
            )

        action_name = match.group(1).lower()
        args_str = match.group(2) or ""

        action_type = self.ACTION_MAP.get(action_name, ActionType.UNKNOWN)

        # Parse arguments
        args = []
        if args_str:
            # Simple arg parsing (handles quoted strings)
            for arg in re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,]+)", args_str):
                arg_value = arg[0] or arg[1] or arg[2].strip()
                if arg_value:
                    args.append(arg_value)

        target = args[0] if args else ""

        # Extract domain for navigation
        domain = None
        if action_type == ActionType.NAVIGATE and target.startswith("http"):
            domain = self._extract_domain(target)

        # Build provenance
        provenance = None
        if observation:
            is_untrusted = self._is_untrusted_domain(domain)
            injection_detected = self._detect_injection(observation)

            provenance = ProvenanceData(
                observation_chunks=[f"obs_{action_idx}"],
                confidence_scores={f"obs_{action_idx}": 1.0},
                attribution_method="safearena_extractor",
                is_untrusted=is_untrusted,
                untrusted_domains={domain} if is_untrusted and domain else set(),
                injection_detected=injection_detected,
            )

        return Action(
            action_id=action_idx,
            action_type=action_type,
            target=target,
            context={"args": args[1:]} if len(args) > 1 else {},
            result=observation,
            domain=domain,
            provenance=provenance,
        )

    def extract_from_log(self, log_path: str) -> Trajectory:
        """
        Extract a trajectory from a SafeArena log file.

        Args:
            log_path: Path to the log file (JSON format)

        Returns:
            Extracted Trajectory
        """
        with open(log_path, 'r') as f:
            data = json.load(f)

        return self.extract_from_dict(data, log_path)

    def extract_from_dict(
        self,
        data: Dict[str, Any],
        source_id: str = "unknown",
    ) -> Trajectory:
        """
        Extract a trajectory from a SafeArena result dictionary.

        Args:
            data: Result dictionary
            source_id: Source identifier

        Returns:
            Extracted Trajectory
        """
        # Extract task info
        task_id = data.get("task_id", data.get("id", source_id))
        task_type = data.get("task_type", "unknown")
        category = data.get("category", data.get("harm_category", "unknown"))
        intent = data.get("intent", data.get("task", data.get("goal", "")))

        # Determine if attack based on task type and result
        is_attack = task_type == "harm" or data.get("is_harmful", False)

        # Extract actions
        actions = []
        action_data = data.get("actions", data.get("trajectory", []))
        observations = data.get("observations", [])

        for idx, action in enumerate(action_data):
            obs = observations[idx] if idx < len(observations) else None

            if isinstance(action, dict):
                parsed = self._parse_browsergym_action(action, idx)
            elif isinstance(action, str):
                parsed = self._parse_agentlab_action(action, idx, obs)
            else:
                parsed = Action(
                    action_id=idx,
                    action_type=ActionType.UNKNOWN,
                    target=str(action),
                )
            actions.append(parsed)

        # Build observation chunks
        observation_chunks = []
        for idx, obs in enumerate(observations):
            if obs:
                chunk = ObservationChunk(
                    chunk_id=f"obs_{idx}",
                    content=str(obs)[:1000],  # Truncate large observations
                    source="safearena",
                    domain=actions[idx].domain if idx < len(actions) else None,
                )
                observation_chunks.append(chunk)

        # Build initial state
        initial_state = State(
            current_url=data.get("start_url"),
            authenticated_services=data.get("authenticated_services", []),
        )

        # Build metadata
        metadata = {
            "category": category,
            "task_type": task_type,
            "harm_description": data.get("harm_description"),
            "expected_behavior": data.get("expected_behavior"),
            "agent": data.get("agent", data.get("model")),
            "success": data.get("success", data.get("completed")),
            "safety_violation": data.get("safety_violation", is_attack),
        }

        return Trajectory(
            trajectory_id=str(task_id),
            source="safearena",
            task_description=intent,
            is_attack=is_attack,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            metadata=metadata,
            success=data.get("success"),
        )

    def extract_from_directory(
        self,
        directory: str,
        pattern: str = "*.json",
    ) -> List[Trajectory]:
        """
        Extract trajectories from all log files in a directory.

        Args:
            directory: Path to directory containing log files
            pattern: Glob pattern for log files

        Returns:
            List of extracted Trajectories
        """
        trajectories = []
        dir_path = Path(directory)

        for log_file in dir_path.glob(pattern):
            try:
                traj = self.extract_from_log(str(log_file))
                trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Failed to extract from {log_file}: {e}")

        return trajectories

    def convert_safearena_results(
        self,
        results_dir: str,
        output_dir: str,
    ) -> Tuple[int, int]:
        """
        Convert SafeArena results to CausalTrace format.

        Args:
            results_dir: Path to SafeArena results directory
            output_dir: Path to output directory

        Returns:
            Tuple of (success_count, failure_count)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trajectories = self.extract_from_directory(results_dir)

        success_count = 0
        failure_count = 0

        for traj in trajectories:
            try:
                output_file = output_path / f"{traj.trajectory_id}.json"
                traj.to_json(str(output_file))
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to save trajectory {traj.trajectory_id}: {e}")
                failure_count += 1

        return success_count, failure_count


# Convenience function
def extract_safearena_trajectory(log_path: str) -> Trajectory:
    """
    Extract a single SafeArena trajectory from a log file.

    Args:
        log_path: Path to log file

    Returns:
        Extracted Trajectory
    """
    extractor = SafeArenaExtractor()
    return extractor.extract_from_log(log_path)
