"""
AgentBench trajectory extractor.

This module extracts trajectories from AgentBench (THUDM) benchmark logs.
AgentBench is a comprehensive benchmark evaluating LLMs as agents across
8 distinct environments.

AgentBench Format Notes:
- 8 environments: OS, Database, Knowledge Graph, Digital Card Game, Lateral Thinking,
  House-Holding (ALFWorld), Web Shopping (WebShop), Web Browsing (Mind2Web)
- Uses function-calling style prompts
- Results stored in outputs/ directory with YAML/JSON configs
- Tracks chat history with user/agent roles
- Multi-turn interaction (4k-13k LLM calls per benchmark)

Key structures in AgentBench logs:
- agent: Agent configuration (module, parameters)
- task: Task configuration (module, parameters)
- assignment: Agent-task pairing
- history: ChatHistoryItem[] with role/content
- status: Execution status
- result: Task outcome
- output: TaskOutput with index, status, result, history

Reference: https://github.com/THUDM/AgentBench
Paper: https://arxiv.org/abs/2308.03688 (ICLR 2024)
"""

import json
import re
import yaml
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import uuid

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files, extract_domain


# 8 environments in AgentBench
AGENTBENCH_ENVIRONMENTS = {
    "os_interaction": {
        "name": "Operating System",
        "short": "OS",
        "type": "newly_created",
        "description": "Interactive OS task completion via bash commands",
        "memory_mb": 500,
    },
    "dbbench": {
        "name": "Database",
        "short": "DB",
        "type": "newly_created",
        "description": "SQL database querying and manipulation",
        "memory_mb": 500,
    },
    "knowledgegraph": {
        "name": "Knowledge Graph",
        "short": "KG",
        "type": "newly_created",
        "description": "Knowledge graph querying and reasoning",
        "memory_mb": 500,
    },
    "dcg": {
        "name": "Digital Card Game",
        "short": "DCG",
        "type": "newly_created",
        "description": "Strategic digital card game play",
        "memory_mb": 500,
    },
    "ltp": {
        "name": "Lateral Thinking Puzzles",
        "short": "LTP",
        "type": "newly_created",
        "description": "Solving lateral thinking puzzles",
        "memory_mb": 500,
    },
    "alfworld": {
        "name": "House-Holding",
        "short": "HH",
        "type": "compiled",
        "source": "ALFWorld",
        "description": "Household task completion in simulated environment",
        "memory_mb": 500,
    },
    "webshop": {
        "name": "Web Shopping",
        "short": "WS",
        "type": "compiled",
        "source": "WebShop",
        "description": "Online shopping navigation and purchase",
        "memory_mb": 15000,
    },
    "mind2web": {
        "name": "Web Browsing",
        "short": "WB",
        "type": "compiled",
        "source": "Mind2Web",
        "description": "General web browsing and task completion",
        "memory_mb": 500,
    },
}

# Action type mapping for AgentBench
AGENTBENCH_ACTION_MAPPING = {
    # OS commands
    "bash": ActionType.CODE_EXECUTION,
    "execute": ActionType.CODE_EXECUTION,
    "run": ActionType.CODE_EXECUTION,
    "cd": ActionType.NAVIGATE,
    "ls": ActionType.READ,
    "cat": ActionType.READ,
    "echo": ActionType.WRITE,
    "mkdir": ActionType.WRITE,
    "rm": ActionType.WRITE,
    "cp": ActionType.WRITE,
    "mv": ActionType.WRITE,
    "grep": ActionType.READ,
    "find": ActionType.READ,
    "vim": ActionType.WRITE,
    "nano": ActionType.WRITE,

    # Database operations
    "select": ActionType.READ,
    "insert": ActionType.WRITE,
    "update": ActionType.WRITE,
    "delete": ActionType.WRITE,
    "create": ActionType.WRITE,
    "drop": ActionType.WRITE,
    "query": ActionType.READ,

    # Knowledge graph
    "sparql": ActionType.READ,
    "lookup": ActionType.READ,
    "traverse": ActionType.READ,

    # Web operations
    "click": ActionType.CLICK,
    "type": ActionType.TYPE,
    "scroll": ActionType.SCROLL,
    "search": ActionType.READ,
    "navigate": ActionType.NAVIGATE,
    "goto": ActionType.NAVIGATE,
    "back": ActionType.NAVIGATE,
    "select": ActionType.SELECT,
    "submit": ActionType.SUBMIT,

    # ALFWorld actions
    "go": ActionType.NAVIGATE,
    "take": ActionType.TOOL_CALL,
    "put": ActionType.TOOL_CALL,
    "open": ActionType.TOOL_CALL,
    "close": ActionType.TOOL_CALL,
    "use": ActionType.TOOL_CALL,
    "examine": ActionType.READ,
    "look": ActionType.READ,
    "inventory": ActionType.READ,
    "heat": ActionType.TOOL_CALL,
    "cool": ActionType.TOOL_CALL,
    "clean": ActionType.TOOL_CALL,
    "slice": ActionType.TOOL_CALL,

    # Card game actions
    "play": ActionType.SUBMIT,
    "attack": ActionType.SUBMIT,
    "defend": ActionType.SUBMIT,
    "draw": ActionType.READ,
    "discard": ActionType.WRITE,
    "end_turn": ActionType.SUBMIT,

    # General actions
    "think": ActionType.AGENT_RESPONSE,
    "answer": ActionType.AGENT_RESPONSE,
    "finish": ActionType.DONE,
    "done": ActionType.DONE,
}

# Status codes
class AgentBenchStatus:
    """Status codes from AgentBench."""
    COMPLETED = "completed"
    RUNNING = "running"
    PENDING = "pending"
    ERROR = "error"
    TIMEOUT = "timeout"


class AgentBenchExtractor(BaseExtractor):
    """
    Extract trajectories from AgentBench benchmark logs.

    AgentBench is a comprehensive benchmark evaluating LLMs as agents
    across 8 distinct environments. It includes:
    - 5 newly created environments (OS, DB, KG, DCG, LTP)
    - 3 environments compiled from published datasets (HH, WS, WB)

    This extractor parses AgentBench output logs and chat histories
    into CausalTrace Trajectory objects.

    Attributes:
        outputs_dir: Path to AgentBench outputs directory
        verbose: Whether to print verbose output
        environments: List of environments to include (None = all)
        include_errors: Whether to include error trajectories
    """

    def __init__(
        self,
        outputs_dir: Optional[str] = None,
        verbose: bool = False,
        environments: Optional[List[str]] = None,
        include_errors: bool = False,
    ):
        """
        Initialize AgentBench extractor.

        Args:
            outputs_dir: Path to AgentBench outputs directory (optional)
            verbose: Whether to print verbose output
            environments: List of environments to include (None = all)
            include_errors: Whether to include error/timeout trajectories
        """
        super().__init__(verbose)
        self.outputs_dir = outputs_dir
        self.environments = environments or list(AGENTBENCH_ENVIRONMENTS.keys())
        self.include_errors = include_errors

        self._log(f"Initialized AgentBenchExtractor")
        self._log(f"Environments: {self.environments}")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single AgentBench log file into a Trajectory.

        Args:
            log_path: Path to the log file (JSON or YAML)

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"Log file not found: {log_path}")
            return None

        try:
            # Support both JSON and YAML
            if log_path.endswith(".yaml") or log_path.endswith(".yml"):
                with open(log_path, 'r') as f:
                    log_data = yaml.safe_load(f)
            else:
                log_data = read_json(log_path)

            return self._parse_log_data(log_data, log_path)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self._log(f"Parse error in {log_path}: {e}")
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
        Parse all AgentBench log files in a directory.

        Args:
            dir_path: Path to directory containing AgentBench logs
            pattern: Glob pattern for matching log files

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        # Find both JSON and YAML files
        json_files = find_files(dir_path, "**/*.json", recursive=True)
        yaml_files = find_files(dir_path, "**/*.yaml", recursive=True)
        yml_files = find_files(dir_path, "**/*.yml", recursive=True)

        log_files = json_files + yaml_files + yml_files
        self._log(f"Found {len(log_files)} potential log files in {dir_path}")

        trajectories = []
        skipped = {"env": 0, "error": 0, "not_agentbench": 0, "parse_error": 0}

        for log_file in log_files:
            try:
                trajectory = self.extract_from_log(log_file)

                if trajectory is None:
                    skipped["not_agentbench"] += 1
                    continue

                # Filter by environment
                env = trajectory.metadata.get("environment")
                if env and env not in self.environments:
                    skipped["env"] += 1
                    continue

                # Filter errors if requested
                status = trajectory.metadata.get("status")
                if not self.include_errors and status in [AgentBenchStatus.ERROR, AgentBenchStatus.TIMEOUT]:
                    skipped["error"] += 1
                    continue

                trajectories.append(trajectory)

            except Exception as e:
                self._log(f"Error processing {log_file}: {e}")
                skipped["parse_error"] += 1

        self._log(f"Successfully extracted {len(trajectories)} trajectories")
        self._log(f"Skipped: {skipped}")
        return trajectories

    def extract_from_task_output(
        self,
        task_output: Dict[str, Any],
        task_name: str,
        agent_name: str,
    ) -> Optional[Trajectory]:
        """
        Extract trajectory from a TaskOutput structure.

        This is useful when processing AgentBench results programmatically.

        Args:
            task_output: TaskOutput dictionary
            task_name: Name of the task/environment
            agent_name: Name of the agent/model

        Returns:
            Trajectory object or None
        """
        log_data = {
            "task_output": task_output,
            "task_name": task_name,
            "agent_name": agent_name,
        }
        return self._parse_log_data(log_data, f"task_output_{task_name}_{agent_name}")

    def _parse_log_data(
        self,
        log_data: Dict[str, Any],
        log_path: str,
    ) -> Optional[Trajectory]:
        """
        Parse log data into a Trajectory.

        Args:
            log_data: Dictionary containing log data
            log_path: Path to the log file

        Returns:
            Trajectory or None if not a valid AgentBench log
        """
        # Validate this is an AgentBench log
        if not self._is_agentbench_log(log_data):
            self._log(f"Not an AgentBench log: {log_path}")
            return None

        # Extract environment/task info
        environment = self._detect_environment(log_data, log_path)
        task_index = log_data.get("index", log_data.get("task_index", 0))
        agent_name = self._extract_agent_name(log_data)

        # Generate trajectory ID
        trajectory_id = self._generate_trajectory_id(environment, task_index, agent_name)

        # Extract task description
        task_description = self._extract_task_description(log_data)

        # Parse chat history into actions
        history = self._extract_history(log_data)
        actions, observation_chunks = self._parse_history(history, environment)

        # Create initial state
        initial_state = self._extract_initial_state(log_data, environment)

        # Create final state
        final_state = self._extract_final_state(log_data, actions)

        # Determine attack status (AgentBench is benign)
        is_attack = self._detect_attack(log_data)

        # Extract status and success
        status = self._extract_status(log_data)
        success = self._extract_success(log_data)

        # Build metadata
        metadata = self._extract_metadata(log_data, log_path, environment)

        return Trajectory(
            trajectory_id=trajectory_id,
            source="agentbench",
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=success,
            error_message=log_data.get("error"),
        )

    def _is_agentbench_log(self, log_data: Dict[str, Any]) -> bool:
        """
        Check if log data appears to be from AgentBench.

        Args:
            log_data: Log data dictionary

        Returns:
            True if this appears to be an AgentBench log
        """
        # Check for AgentBench-specific fields
        indicators = [
            # TaskOutput structure
            "history" in log_data or "chat_history" in log_data,
            "status" in log_data or "result" in log_data,
            # Assignment structure
            "agent" in log_data or "task" in log_data,
            # Output structure
            "output" in log_data and isinstance(log_data.get("output"), dict),
            # Task output structure
            "task_output" in log_data,
            # Index/sample tracking
            "index" in log_data or "task_index" in log_data,
        ]

        # Environment detection in path or data
        path_str = str(log_data.get("_log_path", ""))
        env_in_data = any(
            env in str(log_data).lower()
            for env in AGENTBENCH_ENVIRONMENTS.keys()
        )

        return sum(indicators) >= 2 or env_in_data

    def _detect_environment(
        self,
        log_data: Dict[str, Any],
        log_path: str,
    ) -> str:
        """Detect which AgentBench environment this log is from."""
        # Check explicit field
        if "environment" in log_data:
            return log_data["environment"]
        if "task" in log_data and isinstance(log_data["task"], dict):
            env = log_data["task"].get("module", log_data["task"].get("name", ""))
            for key in AGENTBENCH_ENVIRONMENTS:
                if key in env.lower():
                    return key

        # Check path
        path_lower = log_path.lower()
        for env_key in AGENTBENCH_ENVIRONMENTS:
            if env_key in path_lower:
                return env_key

        # Check content for environment hints
        content_str = json.dumps(log_data).lower()
        env_scores = {env: content_str.count(env) for env in AGENTBENCH_ENVIRONMENTS}
        if env_scores:
            best_match = max(env_scores, key=env_scores.get)
            if env_scores[best_match] > 0:
                return best_match

        return "unknown"

    def _extract_agent_name(self, log_data: Dict[str, Any]) -> str:
        """Extract agent/model name from log data."""
        # Check various locations
        if "agent" in log_data:
            agent = log_data["agent"]
            if isinstance(agent, dict):
                return agent.get("name", agent.get("module", "unknown"))
            return str(agent)
        if "model" in log_data:
            return log_data["model"]
        if "agent_name" in log_data:
            return log_data["agent_name"]

        return "unknown"

    def _generate_trajectory_id(
        self,
        environment: str,
        task_index: Any,
        agent_name: str,
    ) -> str:
        """Generate unique trajectory ID."""
        unique_suffix = uuid.uuid4().hex[:8]
        agent_clean = re.sub(r'[^a-zA-Z0-9]', '_', agent_name)[:20]
        return f"agentbench_{environment}_{task_index}_{agent_clean}_{unique_suffix}"

    def _extract_task_description(self, log_data: Dict[str, Any]) -> str:
        """Extract task description from log data."""
        # Check task field
        if "task" in log_data and isinstance(log_data["task"], dict):
            if "description" in log_data["task"]:
                return log_data["task"]["description"]
            if "goal" in log_data["task"]:
                return log_data["task"]["goal"]

        # Check first user message
        history = self._extract_history(log_data)
        for item in history:
            if item.get("role") == "user":
                content = item.get("content", "")
                if content:
                    return content[:500]  # Truncate long prompts

        # Check explicit fields
        if "task_description" in log_data:
            return log_data["task_description"]
        if "prompt" in log_data:
            return log_data["prompt"][:500]

        return "AgentBench task"

    def _extract_history(self, log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract chat history from log data."""
        # Direct history field
        if "history" in log_data:
            return log_data["history"]
        if "chat_history" in log_data:
            return log_data["chat_history"]

        # Nested in output
        if "output" in log_data and isinstance(log_data["output"], dict):
            output = log_data["output"]
            if "history" in output:
                return output["history"]

        # Nested in task_output
        if "task_output" in log_data and isinstance(log_data["task_output"], dict):
            task_output = log_data["task_output"]
            if "history" in task_output:
                return task_output["history"]

        # Messages field (alternative format)
        if "messages" in log_data:
            return log_data["messages"]

        return []

    def _parse_history(
        self,
        history: List[Dict[str, Any]],
        environment: str,
    ) -> Tuple[List[Action], List[ObservationChunk]]:
        """
        Parse chat history into Actions and ObservationChunks.

        Args:
            history: List of ChatHistoryItem dicts with role/content
            environment: Environment name

        Returns:
            Tuple of (actions, observation_chunks)
        """
        actions = []
        observation_chunks = []
        action_id = 0
        chunk_id = 0

        for msg_idx, item in enumerate(history):
            role = item.get("role", "")
            content = item.get("content", "")

            if not content:
                continue

            if role == "user":
                # User messages become observation chunks
                chunk = ObservationChunk(
                    chunk_id=f"user_{chunk_id}",
                    content=content[:2000],
                    source="environment",
                    domain=f"agentbench.{environment}",
                    metadata={
                        "message_index": msg_idx,
                        "role": role,
                    },
                )
                observation_chunks.append(chunk)
                chunk_id += 1

            elif role == "agent" or role == "assistant":
                # Agent messages become actions
                action = self._parse_agent_response(
                    content=content,
                    action_id=action_id,
                    environment=environment,
                    message_index=msg_idx,
                )
                actions.append(action)
                action_id += 1

                # If the response contains a command/action, parse it
                parsed_actions = self._parse_commands_from_content(
                    content=content,
                    start_action_id=action_id,
                    environment=environment,
                    message_index=msg_idx,
                )
                for pa in parsed_actions:
                    actions.append(pa)
                    action_id += 1

            elif role == "system":
                # System messages are observation chunks
                chunk = ObservationChunk(
                    chunk_id=f"system_{chunk_id}",
                    content=content[:2000],
                    source="system",
                    domain=f"agentbench.{environment}",
                    metadata={
                        "message_index": msg_idx,
                        "role": role,
                    },
                )
                observation_chunks.append(chunk)
                chunk_id += 1

        return actions, observation_chunks

    def _parse_agent_response(
        self,
        content: str,
        action_id: int,
        environment: str,
        message_index: int,
    ) -> Action:
        """Parse an agent response into an Action."""
        # Detect action type from content
        action_type, target = self._detect_action_type(content, environment)

        return Action(
            action_id=action_id,
            action_type=action_type,
            target=target,
            context={
                "environment": environment,
                "message_index": message_index,
                "raw_content": content[:500],
            },
            result=None,  # Result comes from next user message
            domain=f"agentbench.{environment}",
            raw_data={"content": content, "role": "agent"},
        )

    def _detect_action_type(
        self,
        content: str,
        environment: str,
    ) -> Tuple[ActionType, str]:
        """
        Detect action type and target from content.

        Args:
            content: Agent response content
            environment: Environment name

        Returns:
            Tuple of (ActionType, target_string)
        """
        content_lower = content.lower()

        # Environment-specific patterns
        if environment == "os_interaction":
            # Look for bash commands
            bash_match = re.search(r'```(?:bash|shell)?\s*([\s\S]+?)```', content)
            if bash_match:
                cmd = bash_match.group(1).strip().split('\n')[0]
                cmd_name = cmd.split()[0] if cmd else "bash"
                action_type = AGENTBENCH_ACTION_MAPPING.get(cmd_name, ActionType.CODE_EXECUTION)
                return action_type, cmd[:100]

        elif environment == "dbbench":
            # Look for SQL queries
            sql_match = re.search(r'```(?:sql)?\s*([\s\S]+?)```', content)
            if sql_match:
                sql = sql_match.group(1).strip()
                sql_cmd = sql.split()[0].lower() if sql else "query"
                action_type = AGENTBENCH_ACTION_MAPPING.get(sql_cmd, ActionType.READ)
                return action_type, sql[:100]

        elif environment == "alfworld":
            # Look for ALFWorld actions
            for action_word in ["go to", "take", "put", "open", "close", "use", "examine", "look"]:
                if action_word in content_lower:
                    action_type = AGENTBENCH_ACTION_MAPPING.get(
                        action_word.split()[0],
                        ActionType.TOOL_CALL
                    )
                    return action_type, action_word

        elif environment in ["webshop", "mind2web"]:
            # Look for web actions
            for action_word in ["click", "type", "scroll", "search", "select"]:
                if action_word in content_lower:
                    action_type = AGENTBENCH_ACTION_MAPPING.get(action_word, ActionType.CLICK)
                    return action_type, action_word

        # Check for common patterns
        for pattern, action_type in AGENTBENCH_ACTION_MAPPING.items():
            if pattern in content_lower:
                return action_type, pattern

        # Check for finish/done
        if any(word in content_lower for word in ["finish", "done", "complete", "answer:"]):
            return ActionType.DONE, "finish"

        # Default to agent response
        return ActionType.AGENT_RESPONSE, "response"

    def _parse_commands_from_content(
        self,
        content: str,
        start_action_id: int,
        environment: str,
        message_index: int,
    ) -> List[Action]:
        """
        Extract explicit commands/actions from content.

        Some environments have specific action formats embedded in responses.
        """
        actions = []
        action_id = start_action_id

        # Look for code blocks
        code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]+?)```', content)
        for code in code_blocks:
            code = code.strip()
            if code:
                # Determine action type based on environment
                if environment == "os_interaction":
                    action_type = ActionType.CODE_EXECUTION
                elif environment == "dbbench":
                    action_type = ActionType.READ if code.lower().startswith("select") else ActionType.WRITE
                else:
                    action_type = ActionType.TOOL_CALL

                action = Action(
                    action_id=action_id,
                    action_type=action_type,
                    target=code.split('\n')[0][:100],
                    context={
                        "environment": environment,
                        "message_index": message_index,
                        "code_block": code[:1000],
                    },
                    domain=f"agentbench.{environment}",
                    raw_data={"code": code},
                )
                actions.append(action)
                action_id += 1

        # Look for specific action patterns
        # ALFWorld: "Action: go to cabinet 1"
        alfworld_actions = re.findall(r'Action:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        for action_text in alfworld_actions:
            action_text = action_text.strip()
            action_type = self._detect_action_type(action_text, "alfworld")[0]
            action = Action(
                action_id=action_id,
                action_type=action_type,
                target=action_text[:100],
                context={
                    "environment": environment,
                    "message_index": message_index,
                },
                domain=f"agentbench.{environment}",
            )
            actions.append(action)
            action_id += 1

        # WebShop: "search[query]" or "click[button]"
        webshop_actions = re.findall(r'(search|click|buy)\[([^\]]+)\]', content, re.IGNORECASE)
        for action_name, target in webshop_actions:
            action_type = AGENTBENCH_ACTION_MAPPING.get(action_name.lower(), ActionType.CLICK)
            action = Action(
                action_id=action_id,
                action_type=action_type,
                target=target[:100],
                context={
                    "environment": environment,
                    "message_index": message_index,
                    "action_name": action_name,
                },
                domain=f"agentbench.{environment}",
            )
            actions.append(action)
            action_id += 1

        return actions

    def _extract_initial_state(
        self,
        log_data: Dict[str, Any],
        environment: str,
    ) -> State:
        """Extract initial state from log data."""
        state = State()

        # Environment-specific initial state
        env_info = AGENTBENCH_ENVIRONMENTS.get(environment, {})
        state.accumulated_data["environment"] = environment
        state.accumulated_data["environment_type"] = env_info.get("type", "unknown")

        # Set domain for authenticated services
        state.authenticated_services.append(f"agentbench.{environment}")

        # Extract any initial context
        if "initial_state" in log_data:
            state.accumulated_data["initial_context"] = log_data["initial_state"]
        if "task" in log_data and isinstance(log_data["task"], dict):
            state.accumulated_data["task_params"] = log_data["task"].get("parameters", {})

        return state

    def _extract_final_state(
        self,
        log_data: Dict[str, Any],
        actions: List[Action],
    ) -> State:
        """Extract final state from log data and actions."""
        state = State()

        # Copy authentication
        state.authenticated_services.append(f"agentbench.{log_data.get('environment', 'unknown')}")

        # Track accumulated data from actions
        state.accumulated_data["num_actions"] = len(actions)
        state.accumulated_data["action_types"] = list(set(a.action_type.value for a in actions))

        # Extract result
        result = log_data.get("result", log_data.get("output", {}).get("result"))
        if result:
            state.accumulated_data["result"] = result

        return state

    def _detect_attack(self, log_data: Dict[str, Any]) -> bool:
        """
        Detect if trajectory contains an attack.

        AgentBench is a benign benchmark focused on capability evaluation,
        not security. However, we check for any explicit attack markers.
        """
        # Check explicit attack field
        if log_data.get("is_attack", False):
            return True
        if log_data.get("attack_type"):
            return True

        # AgentBench trajectories are benign by default
        return False

    def _extract_status(self, log_data: Dict[str, Any]) -> str:
        """Extract execution status from log data."""
        # Direct status field
        if "status" in log_data:
            return str(log_data["status"]).lower()

        # Nested in output
        if "output" in log_data and isinstance(log_data["output"], dict):
            if "status" in log_data["output"]:
                return str(log_data["output"]["status"]).lower()

        # Nested in task_output
        if "task_output" in log_data and isinstance(log_data["task_output"], dict):
            if "status" in log_data["task_output"]:
                return str(log_data["task_output"]["status"]).lower()

        return AgentBenchStatus.COMPLETED

    def _extract_success(self, log_data: Dict[str, Any]) -> Optional[bool]:
        """Determine if task was successful."""
        # Check explicit success field
        if "success" in log_data:
            return bool(log_data["success"])

        # Check result
        result = log_data.get("result", log_data.get("output", {}).get("result"))
        if result is not None:
            # Numeric score
            if isinstance(result, (int, float)):
                return result > 0
            # Boolean
            if isinstance(result, bool):
                return result
            # String success indicator
            if isinstance(result, str):
                return result.lower() in ["success", "true", "1", "pass", "correct"]

        # Check status
        status = self._extract_status(log_data)
        if status == AgentBenchStatus.COMPLETED:
            return True
        if status in [AgentBenchStatus.ERROR, AgentBenchStatus.TIMEOUT]:
            return False

        return None

    def _extract_metadata(
        self,
        log_data: Dict[str, Any],
        log_path: str,
        environment: str,
    ) -> Dict[str, Any]:
        """Extract metadata from log data."""
        metadata = {
            "environment": environment,
            "environment_name": AGENTBENCH_ENVIRONMENTS.get(environment, {}).get("name"),
            "environment_type": AGENTBENCH_ENVIRONMENTS.get(environment, {}).get("type"),
            "task_index": log_data.get("index", log_data.get("task_index")),
            "agent_name": self._extract_agent_name(log_data),
            "status": self._extract_status(log_data),
            "num_turns": len(self._extract_history(log_data)),
            "log_path": log_path,
        }

        # Add score/result if available
        result = log_data.get("result", log_data.get("output", {}).get("result"))
        if result is not None:
            metadata["result"] = result

        # Add duration if available
        if "duration" in log_data:
            metadata["duration"] = log_data["duration"]

        return {k: v for k, v in metadata.items() if v is not None}

    def get_environments(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of environments and their info."""
        return AGENTBENCH_ENVIRONMENTS.copy()

    def get_statistics(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, Any]:
        """
        Get statistics about extracted trajectories.

        Args:
            trajectories: List of Trajectory objects

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total": len(trajectories),
            "successful": sum(1 for t in trajectories if t.success),
            "failed": sum(1 for t in trajectories if t.success is False),
            "by_environment": {},
            "by_agent": {},
            "by_status": {},
            "avg_actions": 0,
            "total_actions": 0,
        }

        for t in trajectories:
            env = t.metadata.get("environment", "unknown")
            agent = t.metadata.get("agent_name", "unknown")
            status = t.metadata.get("status", "unknown")

            stats["by_environment"][env] = stats["by_environment"].get(env, 0) + 1
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["total_actions"] += len(t.actions)

        if trajectories:
            stats["avg_actions"] = stats["total_actions"] / len(trajectories)

        return stats
