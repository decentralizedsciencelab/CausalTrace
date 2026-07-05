"""
AgentDojo trajectory extractor.

This module extracts trajectories from AgentDojo (ETH Zurich, NeurIPS 2024)
benchmark logs. AgentDojo is a dynamic benchmark for evaluating LLM agent
security against prompt injection attacks.

AgentDojo Format Notes:
- Results stored in runs/{pipeline_name}/{suite_name}/{user_task_id}/{attack_type}/{injection_task_id}.json
- 4 suites: workspace (email/calendar/drive), banking, travel, slack
- 97 user tasks + 629 security test cases
- Messages follow OpenAI chat format with tool calls
- Tool calls include function name, args, and results
- Attack types: direct, ignore_previous, important_instructions, tool_knowledge, injecagent, dos, etc.

Key fields in AgentDojo logs:
- utility: bool - whether user task was completed successfully
- security: bool - whether attack was blocked (False = attack succeeded)
- attack_type: string - type of attack or "none" for benign
- injections: dict - mapping of injection locations to injection text
- messages: list - conversation history including tool calls

Reference: https://github.com/ethz-spylab/agentdojo
Paper: https://arxiv.org/abs/2401.13138 (NeurIPS 2024)
"""

import json
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files, extract_domain


# Tool name to ActionType mapping
TOOL_ACTION_MAPPING = {
    # Calendar tools
    "get_current_day": ActionType.READ,
    "search_calendar_events": ActionType.READ,
    "get_day_calendar_events": ActionType.READ,
    "create_calendar_event": ActionType.WRITE,
    "delete_calendar_event": ActionType.WRITE,
    "update_calendar_event": ActionType.WRITE,
    "cancel_calendar_event": ActionType.WRITE,
    "invite_to_event": ActionType.SEND_EMAIL,

    # Email tools
    "get_received_emails": ActionType.READ,
    "get_sent_emails": ActionType.READ,
    "search_emails": ActionType.READ,
    "get_email_by_id": ActionType.READ,
    "send_email": ActionType.SEND_EMAIL,
    "delete_email": ActionType.WRITE,
    "get_unread_emails": ActionType.READ,
    "search_contacts": ActionType.READ,
    "search_contacts_by_name": ActionType.READ,
    "get_contacts": ActionType.READ,

    # Cloud drive tools
    "list_files": ActionType.READ,
    "search_files": ActionType.READ,
    "create_file": ActionType.WRITE,
    "delete_file": ActionType.WRITE,
    "search_files_by_filename": ActionType.READ,
    "get_file_by_id": ActionType.READ,
    "append_to_file": ActionType.WRITE,
    "share_file": ActionType.WRITE,

    # Banking tools
    "get_balance": ActionType.READ,
    "get_scheduled_transactions": ActionType.READ,
    "get_most_recent_transactions": ActionType.READ,
    "schedule_transaction": ActionType.WRITE,
    "send_money": ActionType.WRITE,
    "get_iban": ActionType.READ,
    "get_user_info": ActionType.READ,
    "update_scheduled_transaction": ActionType.WRITE,

    # Travel tools
    "get_flight_information": ActionType.READ,
    "get_all_restaurants_in_city": ActionType.READ,
    "get_rating_reviews_for_restaurants": ActionType.READ,
    "reserve_restaurant": ActionType.WRITE,
    "get_all_hotels_in_city": ActionType.READ,
    "get_rating_reviews_for_hotels": ActionType.READ,
    "reserve_hotel": ActionType.WRITE,
    "get_hotels_prices": ActionType.READ,
    "get_all_car_rental_companies_in_city": ActionType.READ,
    "get_rating_reviews_for_car_rental": ActionType.READ,
    "reserve_car": ActionType.WRITE,
    "get_car_types_available": ActionType.READ,
    "get_car_price_per_day": ActionType.READ,
    "get_car_fuel_options": ActionType.READ,
    "verify_reservation": ActionType.READ,
    "cancel_reservation": ActionType.WRITE,

    # Slack tools
    "get_channels": ActionType.READ,
    "add_user_to_channel": ActionType.WRITE,
    "remove_user_from_channel": ActionType.WRITE,
    "read_channel_messages": ActionType.READ,
    "read_inbox": ActionType.READ,
    "send_channel_message": ActionType.SEND_EMAIL,  # Treated as messaging
    "send_direct_message": ActionType.SEND_EMAIL,
    "get_users_in_channel": ActionType.READ,
    "get_webpage": ActionType.NAVIGATE,
    "post_webpage": ActionType.WRITE,

    # User account tools
    "get_user_name": ActionType.READ,
    "get_user_email": ActionType.READ,

    # Web tools
    "get_webpage": ActionType.NAVIGATE,
    "post_webpage": ActionType.SUBMIT,
}

# Attack types in AgentDojo
ATTACK_TYPES = {
    "none": "No attack (benign trajectory)",
    "direct": "Direct instruction injection",
    "ignore_previous": "Ignore previous instructions attack",
    "important_instructions": "Fake important instructions attack",
    "important_instructions_no_model_name": "Important instructions without model name",
    "important_instructions_no_names": "Important instructions without any names",
    "important_instructions_no_user_name": "Important instructions without user name",
    "important_instructions_wrong_model_name": "Important instructions with wrong model name",
    "important_instructions_wrong_user_name": "Important instructions with wrong user name",
    "tool_knowledge": "Attack leveraging tool knowledge",
    "injecagent": "InjecAgent-style attack",
    "dos": "Denial of service attack",
    "captcha_dos": "CAPTCHA-based DoS",
    "felony_dos": "Felony-based DoS (attempts to make agent commit illegal act)",
    "offensive_email_dos": "Offensive email DoS",
    "swearwords_dos": "Swearwords DoS",
}

# Suites in AgentDojo
SUITES = ["workspace", "banking", "travel", "slack"]


class AgentDojoExtractor(BaseExtractor):
    """
    Extract trajectories from AgentDojo benchmark logs.

    AgentDojo is a dynamic benchmark for evaluating LLM agents against
    prompt injection attacks. It includes:
    - 97 realistic user tasks across 4 domains
    - 629 security test cases
    - Multiple attack strategies

    The benchmark uses a unique approach where injections are placed
    dynamically in tool outputs (e.g., in calendar events, emails, etc.),
    making it more realistic than static prompt injection benchmarks.

    Attributes:
        runs_dir: Path to the runs directory containing benchmark results
        verbose: Whether to print verbose output
        include_benign: Whether to include non-attack trajectories
        attack_types: List of attack types to include (None = all)
        suites: List of suites to include (None = all)
    """

    def __init__(
        self,
        runs_dir: Optional[str] = None,
        verbose: bool = False,
        include_benign: bool = True,
        attack_types: Optional[List[str]] = None,
        suites: Optional[List[str]] = None,
    ):
        """
        Initialize AgentDojo extractor.

        Args:
            runs_dir: Path to AgentDojo runs directory (optional)
            verbose: Whether to print verbose output
            include_benign: Whether to include non-attack (benign) trajectories
            attack_types: List of attack types to include (None = all)
            suites: List of suites to include (None = all: workspace, banking, travel, slack)
        """
        super().__init__(verbose)
        self.runs_dir = runs_dir
        self.include_benign = include_benign
        self.attack_types = attack_types
        self.suites = suites or SUITES

        # Track injection detection for provenance
        self._injection_patterns: List[str] = []

        self._log(f"Initialized AgentDojoExtractor")
        if runs_dir:
            self._log(f"Runs directory: {runs_dir}")
        self._log(f"Include benign: {include_benign}")
        self._log(f"Suites: {self.suites}")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single AgentDojo log file into a Trajectory.

        Args:
            log_path: Path to the log file (JSON)

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"Log file not found: {log_path}")
            return None

        try:
            log_data = read_json(log_path)
            return self._parse_log_data(log_data, log_path)
        except json.JSONDecodeError as e:
            self._log(f"JSON decode error in {log_path}: {e}")
            return None
        except Exception as e:
            self._log(f"Error parsing log {log_path}: {e}")
            return None

    def extract_from_directory(
        self,
        dir_path: str,
        pattern: str = "**/*.json",
    ) -> List[Trajectory]:
        """
        Parse all AgentDojo log files in a directory.

        AgentDojo organizes results in:
        runs/{pipeline_name}/{suite_name}/{user_task_id}/{attack_type}/{injection_task_id}.json

        Args:
            dir_path: Path to directory containing AgentDojo logs
            pattern: Glob pattern for matching log files (default: "**/*.json")

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        log_files = find_files(dir_path, pattern, recursive=True)
        self._log(f"Found {len(log_files)} log files in {dir_path}")

        trajectories = []
        skipped = {"benign": 0, "attack_type": 0, "suite": 0, "error": 0}

        for log_file in log_files:
            try:
                # Quick filter based on path structure before parsing
                if not self._should_process_file(log_file):
                    continue

                trajectory = self.extract_from_log(log_file)

                if trajectory:
                    # Apply filters
                    if not self.include_benign and not trajectory.is_attack:
                        skipped["benign"] += 1
                        continue

                    if self.attack_types and trajectory.metadata.get("attack_type"):
                        if trajectory.metadata["attack_type"] not in self.attack_types:
                            skipped["attack_type"] += 1
                            continue

                    if trajectory.metadata.get("suite_name"):
                        if trajectory.metadata["suite_name"] not in self.suites:
                            skipped["suite"] += 1
                            continue

                    trajectories.append(trajectory)
                else:
                    skipped["error"] += 1

            except Exception as e:
                self._log(f"Error processing {log_file}: {e}")
                skipped["error"] += 1

        self._log(f"Successfully extracted {len(trajectories)} trajectories")
        self._log(f"Skipped: {skipped}")
        return trajectories

    def extract_from_model_run(
        self,
        model_name: str,
        runs_base_dir: Optional[str] = None,
    ) -> List[Trajectory]:
        """
        Extract all trajectories for a specific model.

        Args:
            model_name: Model name (e.g., "gpt-4o-2024-05-13")
            runs_base_dir: Base directory containing runs (default: self.runs_dir)

        Returns:
            List of Trajectory objects for the model
        """
        base_dir = runs_base_dir or self.runs_dir
        if not base_dir:
            raise ValueError("No runs directory specified")

        model_dir = Path(base_dir) / model_name
        if not model_dir.exists():
            self._log(f"Model directory not found: {model_dir}")
            return []

        return self.extract_from_directory(str(model_dir))

    def _should_process_file(self, file_path: str) -> bool:
        """
        Quick filter to determine if file should be processed based on path.

        This is a preliminary filter based on file path structure.
        Full filtering (attack type, suite, benign) is done after parsing.

        Args:
            file_path: Path to the log file

        Returns:
            True if file should be processed
        """
        path = Path(file_path)
        path_parts = path.parts

        # Convert path to string for substring matching (more lenient)
        path_str = str(path)

        # Check if any suite name is in the path
        suite_in_path = any(suite in path_parts or suite in path_str for suite in self.suites)
        if not suite_in_path:
            return False

        # Path-based pre-filtering with attack_types

        # For path-based pre-filtering with attack_types, check if attack type
        # is in any path component or the filename
        if self.attack_types:
            attack_in_path = any(
                any(attack_type in part for part in path_parts)
                for attack_type in self.attack_types
            )
            # Also check if "none" attack type allows this path
            none_in_path = "none" in path_parts or path.stem == "none"

            if not attack_in_path and not none_in_path:
                # Path doesn't match any attack type filter
                # But we should still process it if include_benign is True
                # (benign logs may not have attack type in path)
                if not self.include_benign:
                    return False

        return True

    def _parse_log_data(
        self,
        log_data: Dict[str, Any],
        log_path: str,
    ) -> Optional[Trajectory]:
        """
        Parse log data into a Trajectory.

        Args:
            log_data: Dictionary containing log data
            log_path: Path to the log file (for ID generation)

        Returns:
            Trajectory object or None if parsing fails
        """
        # Extract core identifiers
        suite_name = log_data.get("suite_name", "unknown")
        pipeline_name = log_data.get("pipeline_name", "unknown")
        user_task_id = log_data.get("user_task_id", "unknown")
        injection_task_id = log_data.get("injection_task_id")
        attack_type = log_data.get("attack_type")

        # Generate trajectory ID
        trajectory_id = self._generate_trajectory_id(
            suite_name, pipeline_name, user_task_id, injection_task_id, attack_type
        )

        # Store injection patterns for provenance detection
        self._injection_patterns = []
        injections = log_data.get("injections", {})
        for key, injection_text in injections.items():
            if injection_text:
                self._injection_patterns.append(injection_text)

        # Determine if this is an attack trajectory
        # In AgentDojo: attack exists if attack_type is not None/none
        # Attack succeeded if security=False (agent was compromised)
        is_attack = self._is_attack_trajectory(log_data)

        # Extract task description from messages
        task_description = self._extract_task_description(log_data)

        # Parse messages into actions
        messages = log_data.get("messages", [])
        actions, observation_chunks = self._parse_messages(messages)

        # Create initial state
        initial_state = self._extract_initial_state(log_data)

        # Create final state
        final_state = self._extract_final_state(log_data, actions)

        # Extract metadata
        metadata = self._extract_metadata(log_data, log_path)

        # Determine success
        success = self._extract_success(log_data)

        # Extract error message
        error_message = log_data.get("error")

        return Trajectory(
            trajectory_id=trajectory_id,
            source="agentdojo",
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=success,
            error_message=error_message,
        )

    def _generate_trajectory_id(
        self,
        suite_name: str,
        pipeline_name: str,
        user_task_id: str,
        injection_task_id: Optional[str],
        attack_type: Optional[str],
    ) -> str:
        """
        Generate a unique trajectory ID.

        Args:
            suite_name: Suite name
            pipeline_name: Model/pipeline name
            user_task_id: User task ID
            injection_task_id: Injection task ID (if any)
            attack_type: Attack type (if any)

        Returns:
            Unique trajectory ID string
        """
        parts = [
            "agentdojo",
            pipeline_name.replace("/", "_"),
            suite_name,
            user_task_id,
        ]

        if attack_type and attack_type != "none":
            parts.append(attack_type)
            if injection_task_id:
                parts.append(injection_task_id)
        else:
            parts.append("benign")

        return "_".join(parts)

    def _is_attack_trajectory(self, log_data: Dict[str, Any]) -> bool:
        """
        Determine if trajectory contains an attack.

        In AgentDojo:
        - attack_type != None/none indicates attack scenario
        - security=True means attack was blocked (agent resisted)
        - security=False means attack succeeded (agent was compromised)

        We mark trajectories as "is_attack" if:
        1. An attack was present (attack_type is set), AND
        2. The attack succeeded (security=False)

        Args:
            log_data: Log data dictionary

        Returns:
            True if this trajectory contains a successful attack
        """
        attack_type = log_data.get("attack_type")

        # No attack if attack_type is None or "none"
        if not attack_type or attack_type == "none" or attack_type == "null":
            return False

        # Attack present - check if it succeeded
        # security=False means the attack succeeded (agent was compromised)
        security = log_data.get("security", True)

        return not security  # Attack succeeded if security=False

    def _extract_task_description(self, log_data: Dict[str, Any]) -> str:
        """
        Extract task description from log data.

        The user task is typically in the first user message.

        Args:
            log_data: Log data dictionary

        Returns:
            Task description string
        """
        messages = log_data.get("messages", [])

        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    # Handle content blocks format
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("content", ""))
                        elif isinstance(block, str):
                            texts.append(block)
                    return " ".join(texts)
                elif isinstance(content, str):
                    return content

        # Fallback to user_task_id
        user_task_id = log_data.get("user_task_id", "Unknown task")
        return f"Task: {user_task_id}"

    def _parse_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Action], List[ObservationChunk]]:
        """
        Parse chat messages into Actions and ObservationChunks.

        AgentDojo messages follow OpenAI chat format:
        - system: System prompt
        - user: User request
        - assistant: Agent response (may include tool_calls)
        - tool: Tool result (includes tool_call info)

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (list of Actions, list of ObservationChunks)
        """
        actions = []
        observation_chunks = []
        action_id = 0
        chunk_id = 0

        # Track tool calls to match with results
        pending_tool_calls: Dict[str, Dict[str, Any]] = {}

        for msg_idx, message in enumerate(messages):
            role = message.get("role", "")

            if role == "system":
                # System message - create observation chunk
                content = self._extract_message_content(message)
                if content:
                    chunk = ObservationChunk(
                        chunk_id=f"system_{chunk_id}",
                        content=content,
                        source="system_prompt",
                        domain="agentdojo",
                        metadata={"message_index": msg_idx, "role": role},
                    )
                    observation_chunks.append(chunk)
                    chunk_id += 1

            elif role == "user":
                # User message - create observation chunk
                content = self._extract_message_content(message)
                if content:
                    # Check if this contains injected content
                    is_injection = self._contains_injection(content)

                    chunk = ObservationChunk(
                        chunk_id=f"user_{chunk_id}",
                        content=content,
                        source="user_prompt",
                        domain="agentdojo",
                        metadata={
                            "message_index": msg_idx,
                            "role": role,
                            "contains_injection": is_injection,
                        },
                    )
                    observation_chunks.append(chunk)
                    chunk_id += 1

            elif role == "assistant":
                # Assistant message - may have tool calls
                tool_calls = message.get("tool_calls") or []
                content = self._extract_message_content(message)

                if tool_calls:
                    # Process tool calls
                    for tool_call in tool_calls:
                        function_name = tool_call.get("function", "")
                        args = tool_call.get("args", {})
                        call_id = tool_call.get("id", f"call_{action_id}")

                        # Store pending tool call
                        pending_tool_calls[call_id] = {
                            "function": function_name,
                            "args": args,
                            "action_id": action_id,
                        }

                        # Create action for the tool call
                        action = self._create_tool_call_action(
                            action_id=action_id,
                            function_name=function_name,
                            args=args,
                            call_id=call_id,
                            message_index=msg_idx,
                        )
                        actions.append(action)
                        action_id += 1

                elif content:
                    # Assistant response without tool calls
                    action = Action(
                        action_id=action_id,
                        action_type=ActionType.AGENT_RESPONSE,
                        target="response",
                        context={
                            "message_index": msg_idx,
                            "role": role,
                        },
                        result=content[:1000] if len(content) > 1000 else content,
                        raw_data=message,
                    )
                    actions.append(action)
                    action_id += 1

            elif role == "tool":
                # Tool result - match with pending tool call
                tool_call_id = message.get("tool_call_id", "")
                tool_call = message.get("tool_call", {})
                content = self._extract_message_content(message)
                error = message.get("error")

                # Create observation chunk for tool output
                is_injection = self._contains_injection(content) if content else False

                chunk = ObservationChunk(
                    chunk_id=f"tool_{chunk_id}",
                    content=content if content else (error or ""),
                    source="tool_output",
                    domain=self._infer_domain_from_tool(tool_call.get("function", "")),
                    metadata={
                        "message_index": msg_idx,
                        "role": role,
                        "tool_call_id": tool_call_id,
                        "function": tool_call.get("function", ""),
                        "contains_injection": is_injection,
                        "is_error": bool(error),
                    },
                )
                observation_chunks.append(chunk)
                chunk_id += 1

                # Update corresponding action with result
                if tool_call_id in pending_tool_calls:
                    call_info = pending_tool_calls[tool_call_id]
                    call_action_id = call_info["action_id"]

                    # Find and update the action
                    for action in actions:
                        if action.action_id == call_action_id:
                            action.result = content if content else error
                            action.context["has_result"] = True
                            action.context["is_error"] = bool(error)

                            # Add provenance if injection detected
                            if is_injection:
                                action.provenance = ProvenanceData(
                                    observation_chunks=[chunk.chunk_id],
                                    confidence_scores={chunk.chunk_id: 0.9},
                                    attribution_method="injection_pattern_match",
                                    is_untrusted=True,
                                    injection_detected=True,
                                )
                            break

                    del pending_tool_calls[tool_call_id]

        return actions, observation_chunks

    def _extract_message_content(self, message: Dict[str, Any]) -> str:
        """
        Extract text content from a message.

        Handles both string content and list of content blocks.

        Args:
            message: Message dictionary

        Returns:
            Extracted text content
        """
        content = message.get("content")

        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("content", ""))
                    elif block.get("type") == "thinking":
                        # Include thinking content
                        texts.append(f"[thinking] {block.get('content', '')}")
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)

        return str(content)

    def _contains_injection(self, content: str) -> bool:
        """
        Check if content contains injection patterns.

        Args:
            content: Content to check

        Returns:
            True if injection pattern detected
        """
        if not content or not self._injection_patterns:
            return False

        content_lower = content.lower()

        for pattern in self._injection_patterns:
            if pattern and pattern.lower() in content_lower:
                return True

        # Check for common injection indicators
        injection_indicators = [
            "TODO:",
            "IMPORTANT:",
            "ignore previous",
            "new instructions",
            "your new task",
            "forget everything",
        ]

        for indicator in injection_indicators:
            if indicator.lower() in content_lower:
                return True

        return False

    def _create_tool_call_action(
        self,
        action_id: int,
        function_name: str,
        args: Dict[str, Any],
        call_id: str,
        message_index: int,
    ) -> Action:
        """
        Create an Action from a tool call.

        Args:
            action_id: Action ID
            function_name: Name of the function being called
            args: Function arguments
            call_id: Tool call ID
            message_index: Index in the message list

        Returns:
            Action object
        """
        # Determine action type from function name
        action_type = TOOL_ACTION_MAPPING.get(function_name, ActionType.TOOL_CALL)

        # Determine target (varies by tool type)
        target = self._extract_target_from_tool_call(function_name, args)

        # Infer domain
        domain = self._infer_domain_from_tool(function_name)

        # Extract data keys
        data_produced, data_consumed = self._extract_data_keys(function_name, args)

        return Action(
            action_id=action_id,
            action_type=action_type,
            target=target,
            context={
                "function_name": function_name,
                "args": args,
                "call_id": call_id,
                "message_index": message_index,
                "has_result": False,
            },
            domain=domain,
            data_produced=data_produced,
            data_consumed=data_consumed,
            raw_data={"function": function_name, "args": args, "id": call_id},
        )

    def _extract_target_from_tool_call(
        self,
        function_name: str,
        args: Dict[str, Any],
    ) -> str:
        """
        Extract target from tool call arguments.

        Args:
            function_name: Function name
            args: Function arguments

        Returns:
            Target string
        """
        # Email targets
        if "recipients" in args:
            return str(args["recipients"])
        if "to" in args:
            return str(args["to"])
        if "recipient" in args:
            return str(args["recipient"])

        # Search targets
        if "query" in args:
            return f"search:{args['query']}"

        # File targets
        if "file_id" in args:
            return f"file:{args['file_id']}"
        if "filename" in args:
            return f"file:{args['filename']}"

        # Calendar targets
        if "event_id" in args:
            return f"event:{args['event_id']}"
        if "date" in args:
            return f"date:{args['date']}"

        # Web targets
        if "url" in args:
            return args["url"]

        # Banking targets
        if "iban" in args:
            return f"iban:{args['iban']}"
        if "recipient_iban" in args:
            return f"iban:{args['recipient_iban']}"

        # Slack targets
        if "channel" in args:
            return f"channel:{args['channel']}"
        if "user" in args:
            return f"user:{args['user']}"

        # Default to function name
        return function_name

    def _infer_domain_from_tool(self, function_name: str) -> str:
        """
        Infer domain from tool function name.

        Args:
            function_name: Function name

        Returns:
            Domain string
        """
        # Calendar tools
        if "calendar" in function_name or "event" in function_name:
            return "calendar.agentdojo"

        # Email tools
        if "email" in function_name or function_name in ["send_email", "get_received_emails", "get_sent_emails"]:
            return "email.agentdojo"

        # File/drive tools
        if "file" in function_name:
            return "drive.agentdojo"

        # Banking tools
        if function_name in ["get_balance", "send_money", "get_iban", "schedule_transaction",
                            "get_scheduled_transactions", "get_most_recent_transactions",
                            "update_scheduled_transaction"]:
            return "banking.agentdojo"

        # Travel tools
        if any(t in function_name for t in ["flight", "hotel", "restaurant", "car", "reservation"]):
            return "travel.agentdojo"

        # Slack tools
        if any(t in function_name for t in ["channel", "slack", "direct_message"]):
            return "slack.agentdojo"

        # Contacts
        if "contact" in function_name:
            return "contacts.agentdojo"

        # Web tools
        if "webpage" in function_name:
            return "web.agentdojo"

        # User tools
        if function_name in ["get_current_day", "get_user_name", "get_user_email", "get_user_info"]:
            return "user.agentdojo"

        return "agentdojo"

    def _extract_data_keys(
        self,
        function_name: str,
        args: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """
        Extract data keys produced and consumed by a tool call.

        Args:
            function_name: Function name
            args: Function arguments

        Returns:
            Tuple of (data_produced, data_consumed)
        """
        produced = []
        consumed = []

        # Data produced by read operations
        if function_name in ["get_received_emails", "get_sent_emails", "search_emails"]:
            produced.append("email_list")
        elif function_name == "get_email_by_id":
            produced.append("email_content")
        elif function_name in ["search_calendar_events", "get_day_calendar_events"]:
            produced.append("calendar_events")
        elif function_name in ["list_files", "search_files", "search_files_by_filename"]:
            produced.append("file_list")
        elif function_name == "get_file_by_id":
            produced.append("file_content")
        elif function_name == "get_balance":
            produced.append("account_balance")
        elif function_name == "get_iban":
            produced.append("iban")
        elif function_name in ["get_scheduled_transactions", "get_most_recent_transactions"]:
            produced.append("transactions")
        elif function_name == "get_current_day":
            produced.append("current_date")
        elif function_name in ["search_contacts", "search_contacts_by_name", "get_contacts"]:
            produced.append("contacts")
        elif function_name in ["read_channel_messages", "read_inbox"]:
            produced.append("messages")

        # Data consumed by write operations
        if function_name == "send_email":
            if "body" in args:
                consumed.append("email_content")
        elif function_name == "send_money":
            consumed.append("iban")
            consumed.append("account_balance")
        elif function_name in ["create_file", "append_to_file"]:
            consumed.append("file_content")
        elif function_name in ["send_channel_message", "send_direct_message"]:
            consumed.append("message_content")

        return produced, consumed

    def _extract_initial_state(self, log_data: Dict[str, Any]) -> State:
        """
        Extract initial state from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            State object
        """
        state = State()

        # User is always authenticated to the suite services
        suite_name = log_data.get("suite_name", "")

        if suite_name == "workspace":
            state.authenticated_services.extend([
                "email.agentdojo",
                "calendar.agentdojo",
                "drive.agentdojo",
            ])
        elif suite_name == "banking":
            state.authenticated_services.append("banking.agentdojo")
        elif suite_name == "travel":
            state.authenticated_services.append("travel.agentdojo")
        elif suite_name == "slack":
            state.authenticated_services.append("slack.agentdojo")

        # Store injections in accumulated data
        injections = log_data.get("injections", {})
        if injections:
            state.accumulated_data["injections"] = injections

        return state

    def _extract_final_state(
        self,
        log_data: Dict[str, Any],
        actions: List[Action],
    ) -> State:
        """
        Extract final state from log data and actions.

        Args:
            log_data: Log data dictionary
            actions: List of actions

        Returns:
            State object
        """
        state = State()

        # Copy authentication from initial state
        suite_name = log_data.get("suite_name", "")
        if suite_name == "workspace":
            state.authenticated_services.extend([
                "email.agentdojo",
                "calendar.agentdojo",
                "drive.agentdojo",
            ])
        elif suite_name == "banking":
            state.authenticated_services.append("banking.agentdojo")
        elif suite_name == "travel":
            state.authenticated_services.append("travel.agentdojo")
        elif suite_name == "slack":
            state.authenticated_services.append("slack.agentdojo")

        # Accumulate data from actions
        for action in actions:
            for data_key in action.data_produced:
                if action.result:
                    state.accumulated_data[data_key] = True

        return state

    def _extract_metadata(
        self,
        log_data: Dict[str, Any],
        log_path: str,
    ) -> Dict[str, Any]:
        """
        Extract metadata from log data.

        Args:
            log_data: Log data dictionary
            log_path: Path to the log file

        Returns:
            Metadata dictionary
        """
        metadata = {
            "suite_name": log_data.get("suite_name"),
            "pipeline_name": log_data.get("pipeline_name"),
            "user_task_id": log_data.get("user_task_id"),
            "injection_task_id": log_data.get("injection_task_id"),
            "attack_type": log_data.get("attack_type"),
            "injections": log_data.get("injections", {}),
            "utility": log_data.get("utility"),
            "security": log_data.get("security"),
            "duration": log_data.get("duration"),
            "log_path": log_path,
        }

        # Add evaluation timestamp if available
        if "evaluation_timestamp" in log_data:
            metadata["evaluation_timestamp"] = log_data["evaluation_timestamp"]

        # Add AgentDojo version if available
        if "agentdojo_package_version" in log_data:
            metadata["agentdojo_version"] = log_data["agentdojo_package_version"]

        return {k: v for k, v in metadata.items() if v is not None}

    def _extract_success(self, log_data: Dict[str, Any]) -> Optional[bool]:
        """
        Determine if the task was successful.

        In AgentDojo:
        - utility=True means user task was completed successfully
        - For attack trajectories, we consider it "successful" (from attacker POV) if security=False

        Args:
            log_data: Log data dictionary

        Returns:
            True if successful, False if failed, None if unknown
        """
        attack_type = log_data.get("attack_type")

        if attack_type and attack_type not in ["none", "null", None]:
            # Attack scenario - success means attack succeeded (security=False)
            security = log_data.get("security")
            if security is not None:
                return not security
        else:
            # Benign scenario - success means utility=True
            utility = log_data.get("utility")
            if utility is not None:
                return bool(utility)

        return None

    def get_attack_types(self) -> Dict[str, str]:
        """
        Get dictionary of attack types and descriptions.

        Returns:
            Dictionary mapping attack type to description
        """
        return ATTACK_TYPES.copy()

    def get_suites(self) -> List[str]:
        """
        Get list of available suites.

        Returns:
            List of suite names
        """
        return SUITES.copy()

    def get_statistics(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, Any]:
        """
        Get statistics about extracted trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total": len(trajectories),
            "attacks": sum(1 for t in trajectories if t.is_attack),
            "benign": sum(1 for t in trajectories if not t.is_attack),
            "by_suite": {},
            "by_attack_type": {},
            "by_model": {},
            "avg_actions": 0,
            "total_actions": 0,
        }

        for t in trajectories:
            suite = t.metadata.get("suite_name", "unknown")
            attack_type = t.metadata.get("attack_type", "none") or "none"
            model = t.metadata.get("pipeline_name", "unknown")

            stats["by_suite"][suite] = stats["by_suite"].get(suite, 0) + 1
            stats["by_attack_type"][attack_type] = stats["by_attack_type"].get(attack_type, 0) + 1
            stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
            stats["total_actions"] += len(t.actions)

        if trajectories:
            stats["avg_actions"] = stats["total_actions"] / len(trajectories)

        return stats
