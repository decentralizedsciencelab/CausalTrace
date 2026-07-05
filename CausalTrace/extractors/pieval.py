"""
LLM-PIEval trajectory extractor.

This module extracts trajectories from Amazon's LLM-PIEval benchmark.
LLM-PIEval evaluates prompt injection vulnerabilities in LLM-based API agents
through multi-turn conversations with injected responses.

PIEval Format Notes:
- API specifications in combined_apis_final.jsonl
- Attack prompts in blackbox_prompts/ directory
- 150 distinct APIs with risk profiles
- Multi-turn injection at specific conversation turns
- White-box and black-box attack modes

Attack Verticals:
- unauthorized_action: Invoke APIs user didn't intend
- distraction: Distract agent from legitimate tasks
- sensitive_info: Extract sensitive information

Reference: https://github.com/amazon-science/llm-pieval
Paper: NeurIPS 2024
"""

import json
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files


class RiskLevel(Enum):
    """Risk levels for APIs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackMode(Enum):
    """Attack modes in PIEval."""
    WHITE_BOX = "white_box"  # Attacker knows API specs
    BLACK_BOX = "black_box"  # Attacker doesn't know API specs


@dataclass
class APISpecification:
    """API specification with risk profile."""
    name: str
    description: str
    category: str
    risk_level: RiskLevel
    parameters: Dict[str, Any]
    requires_confirmation: bool = False


# API name to ActionType mapping
API_ACTION_MAPPING = {
    # Financial APIs
    "TransferFunds": ActionType.WRITE,
    "CheckAccountBalance": ActionType.READ,
    "GetTransactionHistory": ActionType.READ,
    "SendPayment": ActionType.WRITE,

    # Communication APIs
    "SendEmail": ActionType.SEND_EMAIL,
    "SendSMS": ActionType.SEND_EMAIL,
    "MakePhoneCall": ActionType.TOOL_CALL,
    "GetMessages": ActionType.READ,

    # File/Storage APIs
    "DeleteUserFiles": ActionType.WRITE,
    "UploadFile": ActionType.UPLOAD,
    "DownloadFile": ActionType.DOWNLOAD,
    "ShareFile": ActionType.WRITE,

    # System APIs
    "ExecuteCode": ActionType.CODE_EXECUTION,
    "RunCommand": ActionType.CODE_EXECUTION,
    "ModifySettings": ActionType.WRITE,

    # Smart Home APIs
    "LockDoors": ActionType.WRITE,
    "UnlockDoors": ActionType.WRITE,
    "AdjustTemperature": ActionType.WRITE,
    "ControlLights": ActionType.WRITE,

    # User/Auth APIs
    "ChangePassword": ActionType.WRITE,
    "UpdateEmail": ActionType.WRITE,
    "GetUserProfile": ActionType.READ,
}

# Injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+previous",
    r"new\s+task",
    r"instead\s+of",
    r"actually",
    r"urgent",
    r"important",
    r"override",
    r"disregard",
]


class LLMPIEvalExtractor(BaseExtractor):
    """
    Extract trajectories from LLM-PIEval benchmark.

    PIEval evaluates prompt injection in multi-turn API agent conversations.
    Attacks inject malicious instructions through API response content.

    Attributes:
        attack_modes: Filter by attack modes (white_box, black_box)
        min_risk_level: Minimum API risk level to include
        verbose: Whether to print verbose output
    """

    def __init__(
        self,
        attack_modes: Optional[List[str]] = None,
        min_risk_level: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the PIEval extractor.

        Args:
            attack_modes: Filter by attack modes (e.g., ["white_box"])
            min_risk_level: Minimum risk level (low, medium, high, critical)
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.attack_modes = attack_modes
        self.min_risk_level = RiskLevel(min_risk_level) if min_risk_level else None
        self._api_specs: Dict[str, APISpecification] = {}
        self._log("Initialized LLMPIEvalExtractor")

    def load_api_specifications(self, spec_path: str) -> None:
        """
        Load API specifications from file.

        Args:
            spec_path: Path to API specifications (JSONL or JSON)
        """
        if not self._validate_path(spec_path):
            self._log(f"Spec file not found: {spec_path}")
            return

        path = Path(spec_path)

        if path.suffix == ".jsonl":
            with open(spec_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            api_data = json.loads(line)
                            self._parse_api_spec(api_data)
                        except json.JSONDecodeError:
                            pass
        else:
            data = read_json(spec_path)
            if isinstance(data, list):
                for api_data in data:
                    self._parse_api_spec(api_data)
            elif isinstance(data, dict):
                self._parse_api_spec(data)

        self._log(f"Loaded {len(self._api_specs)} API specifications")

    def _parse_api_spec(self, api_data: Dict[str, Any]) -> None:
        """Parse an API specification entry."""
        name = api_data.get("name", api_data.get("api_name", ""))
        if not name:
            return

        # Determine risk level
        risk_str = api_data.get("risk_level", api_data.get("risk", "medium")).lower()
        risk_map = {"low": RiskLevel.LOW, "medium": RiskLevel.MEDIUM, "high": RiskLevel.HIGH, "critical": RiskLevel.CRITICAL}
        risk_level = risk_map.get(risk_str, RiskLevel.MEDIUM)

        spec = APISpecification(
            name=name,
            description=api_data.get("description", ""),
            category=api_data.get("category", "general"),
            risk_level=risk_level,
            parameters=api_data.get("parameters", {}),
            requires_confirmation=api_data.get("requires_confirmation", False),
        )
        self._api_specs[name] = spec

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single PIEval log into a Trajectory.

        Args:
            log_path: Path to the log file

        Returns:
            Trajectory object or None
        """
        if not self._validate_path(log_path):
            self._log(f"File not found: {log_path}")
            return None

        try:
            data = read_json(log_path)
            return self._parse_log_data(data, log_path)
        except Exception as e:
            self._log(f"Error parsing {log_path}: {e}")
            return None

    def extract_from_directory(
        self,
        dir_path: str,
        pattern: str = "*.json",
    ) -> List[Trajectory]:
        """
        Extract trajectories from PIEval directory.

        Expected structure:
        - combined_apis_final.jsonl (API specs)
        - blackbox_prompts/ (attack prompts)
        - results/ (evaluation logs)

        Args:
            dir_path: Path to PIEval directory
            pattern: Glob pattern for log files

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        trajectories = []

        # Load API specifications if available
        spec_files = find_files(dir_path, "*apis*.jsonl", recursive=True)
        for spec_file in spec_files:
            self.load_api_specifications(spec_file)

        # Extract from log files
        log_files = find_files(dir_path, pattern, recursive=True)
        # Also check for JSONL files
        jsonl_files = find_files(dir_path, "*.jsonl", recursive=True)

        # Exclude API spec files from processing
        all_files = set(log_files + jsonl_files) - set(spec_files)

        self._log(f"Found {len(all_files)} log files to process")

        for log_file in all_files:
            try:
                if log_file.endswith('.jsonl'):
                    file_trajs = self._extract_from_jsonl(log_file)
                    trajectories.extend(file_trajs)
                else:
                    traj = self.extract_from_log(log_file)
                    if traj and self._should_include(traj):
                        trajectories.append(traj)
            except Exception as e:
                self._log(f"Error processing {log_file}: {e}")

        self._log(f"Extracted {len(trajectories)} trajectories from PIEval")
        return trajectories

    def _extract_from_jsonl(self, jsonl_path: str) -> List[Trajectory]:
        """Extract trajectories from JSONL file."""
        trajectories = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    traj = self._parse_log_data(data, jsonl_path, idx)
                    if traj and self._should_include(traj):
                        trajectories.append(traj)
                except json.JSONDecodeError:
                    pass

        return trajectories

    def _should_include(self, trajectory: Trajectory) -> bool:
        """Check if trajectory passes filters."""
        # Filter by attack mode
        if self.attack_modes:
            mode = trajectory.metadata.get("attack_mode")
            if mode not in self.attack_modes:
                return False

        # Filter by risk level
        if self.min_risk_level:
            risk = trajectory.metadata.get("target_api_risk_level", "low")
            risk_order = ["low", "medium", "high", "critical"]
            if risk_order.index(risk) < risk_order.index(self.min_risk_level.value):
                return False

        return True

    def _parse_log_data(
        self,
        log_data: Dict[str, Any],
        log_path: str,
        idx: int = 0,
    ) -> Optional[Trajectory]:
        """
        Parse PIEval log data into a Trajectory.

        Args:
            log_data: Dictionary containing log data
            log_path: Path to source file
            idx: Index in source file

        Returns:
            Trajectory object or None
        """
        # Extract core fields
        test_id = log_data.get("test_id", log_data.get("id", f"pieval_{idx}"))
        attack_mode = log_data.get("attack_mode", "unknown")
        target_api = log_data.get("target_api", log_data.get("target_function", "unknown"))
        injection_turn = log_data.get("injection_turn", log_data.get("injection_index", 0))
        injection_text = log_data.get("injection_text", log_data.get("injected_prompt", ""))

        # Get messages/turns
        messages = log_data.get("messages", log_data.get("conversation", []))
        if not messages and not injection_text:
            return None

        # Generate trajectory ID
        trajectory_id = f"pieval_{test_id}_{target_api}_{idx}"

        # Create observation chunks and actions
        actions, observation_chunks = self._parse_conversation(
            messages=messages,
            injection_turn=injection_turn,
            injection_text=injection_text,
            target_api=target_api,
        )

        # Determine if attack succeeded
        attack_success = log_data.get("attack_success", log_data.get("target_api_invoked", False))

        # Create states
        initial_state = State()
        if target_api in self._api_specs:
            initial_state.authenticated_services.append(f"pieval.{self._api_specs[target_api].category}")
        else:
            initial_state.authenticated_services.append("pieval")

        final_state = State()
        final_state.accumulated_data["attack_success"] = attack_success
        final_state.accumulated_data["target_api"] = target_api

        # Get risk level
        risk_level = "medium"
        if target_api in self._api_specs:
            risk_level = self._api_specs[target_api].risk_level.value

        # Build metadata
        metadata = {
            "test_id": test_id,
            "attack_mode": attack_mode,
            "target_api": target_api,
            "target_api_risk_level": risk_level,
            "injection_turn": injection_turn,
            "attack_success": attack_success,
            "source_path": log_path,
            "benchmark": "pieval",
        }

        # Task description
        task_description = f"[PIEval/{attack_mode}] Target API: {target_api}"

        return Trajectory(
            trajectory_id=trajectory_id,
            source="pieval",
            task_description=task_description,
            is_attack=attack_success,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=attack_success,
        )

    def _parse_conversation(
        self,
        messages: List[Dict[str, Any]],
        injection_turn: int,
        injection_text: str,
        target_api: str,
    ) -> Tuple[List[Action], List[ObservationChunk]]:
        """
        Parse conversation messages into actions and observation chunks.

        Args:
            messages: List of conversation messages
            injection_turn: Turn index where injection occurs
            injection_text: The injected text
            target_api: Target API name

        Returns:
            Tuple of (actions, observation_chunks)
        """
        actions = []
        observation_chunks = []
        action_id = 0
        chunk_id = 0

        for turn_idx, msg in enumerate(messages):
            role = msg.get("role", msg.get("type", ""))
            content = self._extract_content(msg)

            is_injection_turn = (turn_idx == injection_turn)
            contains_injection = is_injection_turn or (injection_text and injection_text in content)

            if role in ["user", "system"]:
                # Create observation chunk
                chunk = ObservationChunk(
                    chunk_id=f"{role}_{chunk_id}",
                    content=content,
                    source=f"{role}_prompt",
                    domain="pieval",
                    metadata={
                        "turn_index": turn_idx,
                        "contains_injection": contains_injection,
                    },
                )
                observation_chunks.append(chunk)
                chunk_id += 1

            elif role == "assistant":
                # Check for function/tool calls
                tool_calls = msg.get("tool_calls", msg.get("function_calls", []))

                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        func_name = func.get("name", tc.get("name", "unknown"))
                        func_args = func.get("arguments", tc.get("args", {}))

                        if isinstance(func_args, str):
                            try:
                                func_args = json.loads(func_args)
                            except json.JSONDecodeError:
                                func_args = {"raw": func_args}

                        action_type = API_ACTION_MAPPING.get(func_name, ActionType.TOOL_CALL)
                        is_target = func_name == target_api

                        provenance = None
                        if is_target and contains_injection:
                            # Find injection chunk
                            inj_chunks = [c.chunk_id for c in observation_chunks if c.metadata.get("contains_injection")]
                            if inj_chunks:
                                provenance = ProvenanceData(
                                    observation_chunks=inj_chunks,
                                    confidence_scores={c: 0.9 for c in inj_chunks},
                                    attribution_method="injection_trace",
                                    is_untrusted=True,
                                    injection_detected=True,
                                )

                        action = Action(
                            action_id=action_id,
                            action_type=action_type,
                            target=func_name,
                            context={
                                "function_name": func_name,
                                "arguments": func_args,
                                "turn_index": turn_idx,
                                "is_target_api": is_target,
                            },
                            domain="pieval",
                            provenance=provenance,
                            raw_data=tc,
                        )
                        actions.append(action)
                        action_id += 1
                else:
                    # Plain response
                    action = Action(
                        action_id=action_id,
                        action_type=ActionType.AGENT_RESPONSE,
                        target="response",
                        context={"turn_index": turn_idx},
                        result=content[:500] if content else None,
                        domain="pieval",
                        raw_data=msg,
                    )
                    actions.append(action)
                    action_id += 1

            elif role in ["function", "tool"]:
                # Function result - create observation chunk
                func_name = msg.get("name", msg.get("function", "unknown"))
                chunk = ObservationChunk(
                    chunk_id=f"tool_{chunk_id}",
                    content=content,
                    source="function_output",
                    domain="pieval",
                    metadata={
                        "turn_index": turn_idx,
                        "function_name": func_name,
                        "contains_injection": contains_injection,
                    },
                )
                observation_chunks.append(chunk)
                chunk_id += 1

        # If no messages but we have injection text, create minimal trajectory
        if not actions and injection_text:
            chunk = ObservationChunk(
                chunk_id="injection_0",
                content=injection_text,
                source="injection",
                domain="pieval",
                metadata={"contains_injection": True},
            )
            observation_chunks.append(chunk)

            action = Action(
                action_id=0,
                action_type=ActionType.TOOL_CALL,
                target=target_api,
                context={"is_target_api": True},
                domain="pieval",
                provenance=ProvenanceData(
                    observation_chunks=["injection_0"],
                    confidence_scores={"injection_0": 1.0},
                    attribution_method="injection_only",
                    is_untrusted=True,
                    injection_detected=True,
                ),
                raw_data={"injection_text": injection_text},
            )
            actions.append(action)

        return actions, observation_chunks

    def _extract_content(self, msg: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        content = msg.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get("text", block.get("content", "")))
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)

        return str(content) if content else ""

    def get_risk_levels(self) -> List[str]:
        """Get list of risk levels."""
        return [r.value for r in RiskLevel]

    def get_attack_modes(self) -> List[str]:
        """Get list of attack modes."""
        return [m.value for m in AttackMode]

    def get_statistics(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Get statistics about extracted trajectories."""
        stats = {
            "total": len(trajectories),
            "attacks": sum(1 for t in trajectories if t.is_attack),
            "benign": sum(1 for t in trajectories if not t.is_attack),
            "by_attack_mode": {},
            "by_risk_level": {},
            "by_target_api": {},
            "avg_injection_turn": 0,
        }

        injection_turns = []

        for t in trajectories:
            mode = t.metadata.get("attack_mode", "unknown")
            risk = t.metadata.get("target_api_risk_level", "unknown")
            api = t.metadata.get("target_api", "unknown")
            turn = t.metadata.get("injection_turn")

            stats["by_attack_mode"][mode] = stats["by_attack_mode"].get(mode, 0) + 1
            stats["by_risk_level"][risk] = stats["by_risk_level"].get(risk, 0) + 1
            stats["by_target_api"][api] = stats["by_target_api"].get(api, 0) + 1

            if turn is not None:
                injection_turns.append(turn)

        if injection_turns:
            stats["avg_injection_turn"] = sum(injection_turns) / len(injection_turns)

        return stats
