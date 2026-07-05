"""
MARBLE (MultiAgentBench) trajectory extractor.

This module extracts trajectories from MARBLE (Multi-Agent cooRdination Backbone with
LLM Engine) benchmark logs. MARBLE evaluates multi-agent collaboration and competition
capabilities of LLM agents.

MARBLE Format Notes:
- 6 task scenarios: research, minecraft, database, coding, werewolf, bargaining
- Collaborative tasks: research, minecraft, database, coding
- Competitive tasks: werewolf (social deduction), bargaining
- Coordination topologies: star (centralized), tree (hierarchical), graph (mesh), chain (sequential)
- Each scenario has ~100 test cases with varying difficulty levels
- Logs contain: agent actions, messages, shared memory updates, milestone tracking

Key structures in MARBLE logs:
- agents: List of agent configurations (id, role, model, tools)
- messages: Inter-agent communication history
- actions: Individual agent actions with timestamps
- shared_memory: Global knowledge and collective decisions
- milestones: Task progress tracking
- coordination_protocol: topology type (star/tree/graph/chain)

Reference: https://github.com/ulab-uiuc/MARBLE
Paper: https://arxiv.org/abs/2503.01935 (ACL 2025)
"""

import json
import re
from typing import List, Optional, Dict, Any, Tuple, Set
from pathlib import Path
from datetime import datetime
import uuid

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.models.multi_agent import (
    MultiAgentTrajectory,
    AgentNode,
    AgentRole,
    InterAgentFlow,
    InterAgentFlowType,
    create_delegation_flow,
    create_injection_flow,
)
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files, extract_domain


# Task scenarios in MARBLE
MARBLE_SCENARIOS = {
    "research": {
        "type": "collaborative",
        "description": "Research proposal generation with multiple agents",
        "typical_roles": ["planner", "researcher", "writer", "reviewer"],
    },
    "minecraft": {
        "type": "collaborative",
        "description": "Minecraft building tasks with coordination",
        "typical_roles": ["architect", "builder", "resource_gatherer"],
    },
    "database": {
        "type": "collaborative",
        "description": "Database error analysis and correction",
        "typical_roles": ["analyst", "debugger", "validator"],
    },
    "coding": {
        "type": "collaborative",
        "description": "Programming challenges with pair/team coding",
        "typical_roles": ["planner", "coder", "tester", "reviewer"],
    },
    "werewolf": {
        "type": "competitive",
        "description": "Social deduction game (Mafia variant)",
        "typical_roles": ["werewolf", "villager", "seer", "doctor"],
    },
    "bargaining": {
        "type": "competitive",
        "description": "Resource negotiation between agents",
        "typical_roles": ["buyer", "seller", "mediator"],
    },
}

# Coordination protocols/topologies
COORDINATION_PROTOCOLS = {
    "star": "Centralized - single planner coordinates all actors",
    "tree": "Hierarchical - delegation through tree structure",
    "graph": "Mesh - interconnected agents with direct communication",
    "chain": "Sequential - handoff between agents in sequence",
}

# Action type mapping for MARBLE
MARBLE_ACTION_MAPPING = {
    # Planning actions
    "plan": ActionType.TOOL_CALL,
    "delegate": ActionType.DELEGATION,
    "assign": ActionType.DELEGATION,

    # Communication actions
    "send_message": ActionType.AGENT_RESPONSE,
    "broadcast": ActionType.AGENT_RESPONSE,
    "query": ActionType.READ,
    "respond": ActionType.AGENT_RESPONSE,

    # Tool actions
    "execute_tool": ActionType.TOOL_CALL,
    "search": ActionType.READ,
    "write_document": ActionType.WRITE,
    "edit_document": ActionType.WRITE,
    "read_document": ActionType.READ,
    "run_code": ActionType.CODE_EXECUTION,
    "execute_code": ActionType.CODE_EXECUTION,
    "compile": ActionType.CODE_EXECUTION,
    "test": ActionType.CODE_EXECUTION,

    # Database actions
    "query_db": ActionType.READ,
    "update_db": ActionType.WRITE,
    "analyze_data": ActionType.READ,

    # Minecraft actions
    "build": ActionType.WRITE,
    "mine": ActionType.READ,
    "craft": ActionType.TOOL_CALL,
    "move": ActionType.NAVIGATE,
    "place_block": ActionType.WRITE,

    # Game actions (werewolf/bargaining)
    "vote": ActionType.SUBMIT,
    "accuse": ActionType.AGENT_RESPONSE,
    "defend": ActionType.AGENT_RESPONSE,
    "negotiate": ActionType.AGENT_RESPONSE,
    "offer": ActionType.SUBMIT,
    "accept": ActionType.SUBMIT,
    "reject": ActionType.AGENT_RESPONSE,

    # Memory actions
    "update_memory": ActionType.STATE_MUTATION,
    "read_memory": ActionType.READ,
    "share_knowledge": ActionType.AGENT_RESPONSE,
}


class MARBLEExtractor(BaseExtractor):
    """
    Extract trajectories from MARBLE (MultiAgentBench) benchmark logs.

    MARBLE is a multi-agent benchmark evaluating collaboration and competition
    capabilities of LLM agents. It includes:
    - 6 task scenarios (4 collaborative, 2 competitive)
    - 4 coordination topologies (star, tree, graph, chain)
    - ~100 test cases per scenario with varying difficulty

    This extractor produces MultiAgentTrajectory objects that capture:
    - Individual agent trajectories
    - Inter-agent communication flows
    - Shared memory updates
    - Coordination protocol metadata

    Attributes:
        marble_path: Path to MARBLE repository
        verbose: Whether to print verbose output
        scenarios: List of scenarios to include (None = all)
        protocols: List of coordination protocols to include (None = all)
    """

    def __init__(
        self,
        marble_path: Optional[str] = None,
        verbose: bool = False,
        scenarios: Optional[List[str]] = None,
        protocols: Optional[List[str]] = None,
    ):
        """
        Initialize MARBLE extractor.

        Args:
            marble_path: Path to MARBLE repository (optional)
            verbose: Whether to print verbose output
            scenarios: List of scenarios to include (None = all)
            protocols: List of coordination protocols to include (None = all)
        """
        super().__init__(verbose)
        self.marble_path = marble_path
        self.scenarios = scenarios or list(MARBLE_SCENARIOS.keys())
        self.protocols = protocols or list(COORDINATION_PROTOCOLS.keys())

        self._log(f"Initialized MARBLEExtractor")
        self._log(f"Scenarios: {self.scenarios}")
        self._log(f"Protocols: {self.protocols}")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single MARBLE log file into a Trajectory.

        For multi-agent trajectories, this returns a flattened single-agent
        view. Use extract_multi_agent_from_log for the full multi-agent
        representation.

        Args:
            log_path: Path to the log file (JSON)

        Returns:
            Trajectory object or None if parsing fails
        """
        ma_trajectory = self.extract_multi_agent_from_log(log_path)
        if ma_trajectory is None:
            return None

        return self._flatten_to_single_trajectory(ma_trajectory)

    def extract_multi_agent_from_log(self, log_path: str) -> Optional[MultiAgentTrajectory]:
        """
        Parse a single MARBLE log file into a MultiAgentTrajectory.

        Args:
            log_path: Path to the log file (JSON)

        Returns:
            MultiAgentTrajectory object or None if parsing fails
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
        Parse all MARBLE log files in a directory.

        Args:
            dir_path: Path to directory containing MARBLE logs
            pattern: Glob pattern for matching log files

        Returns:
            List of Trajectory objects
        """
        ma_trajectories = self.extract_multi_agent_from_directory(dir_path, pattern)
        return [self._flatten_to_single_trajectory(ma) for ma in ma_trajectories]

    def extract_multi_agent_from_directory(
        self,
        dir_path: str,
        pattern: str = "**/*.json",
    ) -> List[MultiAgentTrajectory]:
        """
        Parse all MARBLE log files in a directory into MultiAgentTrajectories.

        Args:
            dir_path: Path to directory containing MARBLE logs
            pattern: Glob pattern for matching log files

        Returns:
            List of MultiAgentTrajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        log_files = find_files(dir_path, pattern, recursive=True)
        self._log(f"Found {len(log_files)} potential log files in {dir_path}")

        trajectories = []
        skipped = {"scenario": 0, "protocol": 0, "error": 0, "not_marble": 0}

        for log_file in log_files:
            try:
                # Try to parse as MARBLE log
                trajectory = self.extract_multi_agent_from_log(log_file)

                if trajectory is None:
                    skipped["not_marble"] += 1
                    continue

                # Apply filters
                scenario = trajectory.metadata.get("scenario")
                if scenario and scenario not in self.scenarios:
                    skipped["scenario"] += 1
                    continue

                protocol = trajectory.metadata.get("coordination_protocol")
                if protocol and protocol not in self.protocols:
                    skipped["protocol"] += 1
                    continue

                trajectories.append(trajectory)

            except Exception as e:
                self._log(f"Error processing {log_file}: {e}")
                skipped["error"] += 1

        self._log(f"Successfully extracted {len(trajectories)} multi-agent trajectories")
        self._log(f"Skipped: {skipped}")
        return trajectories

    def _parse_log_data(
        self,
        log_data: Dict[str, Any],
        log_path: str,
    ) -> Optional[MultiAgentTrajectory]:
        """
        Parse MARBLE log data into a MultiAgentTrajectory.

        Args:
            log_data: Dictionary containing log data
            log_path: Path to the log file

        Returns:
            MultiAgentTrajectory or None if not a valid MARBLE log
        """
        # Validate this is a MARBLE log
        if not self._is_marble_log(log_data):
            self._log(f"Not a MARBLE log: {log_path}")
            return None

        # Extract scenario and task info
        scenario = log_data.get("scenario", log_data.get("task_type", "unknown"))
        task_id = log_data.get("task_id", log_data.get("test_case_id", "unknown"))
        protocol = log_data.get("coordination_protocol", log_data.get("topology", "unknown"))

        # Generate trajectory ID
        trajectory_id = self._generate_trajectory_id(scenario, task_id, protocol)

        # Extract task description
        task_description = self._extract_task_description(log_data)

        # Parse agents
        agents = self._parse_agents(log_data)

        # Parse individual agent trajectories
        agent_trajectories = self._parse_agent_trajectories(log_data, agents)

        # Parse inter-agent flows
        inter_agent_flows = self._parse_inter_agent_flows(log_data, agents)

        # Determine attack status (MARBLE is primarily benign, but may have attack scenarios)
        is_attack, attack_type, compromised_agents = self._detect_attack(log_data)

        # Extract timestamps
        start_time, end_time = self._extract_timestamps(log_data)

        # Determine success
        success = self._extract_success(log_data)

        # Build metadata
        metadata = self._extract_metadata(log_data, log_path)

        return MultiAgentTrajectory(
            trajectory_id=trajectory_id,
            source="marble",
            task_description=task_description,
            agents=agents,
            trajectories=agent_trajectories,
            inter_agent_flows=inter_agent_flows,
            is_attack=is_attack,
            attack_type=attack_type,
            compromised_agents=compromised_agents,
            attack_success=None,  # Not applicable for MARBLE
            start_time=start_time,
            end_time=end_time,
            success=success,
            error_message=log_data.get("error"),
            metadata=metadata,
        )

    def _is_marble_log(self, log_data: Dict[str, Any]) -> bool:
        """
        Check if log data appears to be from MARBLE benchmark.

        Args:
            log_data: Log data dictionary

        Returns:
            True if this appears to be a MARBLE log
        """
        # Check for MARBLE-specific fields
        marble_indicators = [
            "agents" in log_data and isinstance(log_data.get("agents"), list),
            "scenario" in log_data or "task_type" in log_data,
            "coordination_protocol" in log_data or "topology" in log_data,
            "shared_memory" in log_data or "global_memory" in log_data,
            "milestones" in log_data or "checkpoints" in log_data,
            "messages" in log_data and isinstance(log_data.get("messages"), list),
        ]

        # Require at least 3 indicators
        return sum(marble_indicators) >= 3

    def _generate_trajectory_id(
        self,
        scenario: str,
        task_id: str,
        protocol: str,
    ) -> str:
        """Generate unique trajectory ID."""
        unique_suffix = uuid.uuid4().hex[:8]
        return f"marble_{scenario}_{task_id}_{protocol}_{unique_suffix}"

    def _extract_task_description(self, log_data: Dict[str, Any]) -> str:
        """Extract task description from log data."""
        # Try various fields
        if "task_description" in log_data:
            return log_data["task_description"]
        if "task" in log_data and isinstance(log_data["task"], dict):
            return log_data["task"].get("description", "Unknown task")
        if "prompt" in log_data:
            return log_data["prompt"]
        if "goal" in log_data:
            return log_data["goal"]

        scenario = log_data.get("scenario", "unknown")
        return f"MARBLE {scenario} task"

    def _parse_agents(self, log_data: Dict[str, Any]) -> List[AgentNode]:
        """
        Parse agent configurations from log data.

        Args:
            log_data: Log data dictionary

        Returns:
            List of AgentNode objects
        """
        agents = []
        agents_data = log_data.get("agents", [])

        for idx, agent_data in enumerate(agents_data):
            if isinstance(agent_data, dict):
                agent_id = agent_data.get("id", agent_data.get("agent_id", f"agent_{idx}"))
                role_str = agent_data.get("role", agent_data.get("agent_role", "worker"))

                # Map role string to AgentRole
                role = self._map_role(role_str)

                agent = AgentNode(
                    agent_id=str(agent_id),
                    agent_role=role,
                    agent_name=agent_data.get("name", agent_data.get("agent_name")),
                    model=agent_data.get("model", agent_data.get("llm_model")),
                    tools=agent_data.get("tools", []),
                    permissions=set(agent_data.get("permissions", [])),
                    trust_level=agent_data.get("trust_level", 1.0),
                    metadata={
                        "profile": agent_data.get("profile", ""),
                        "system_message": agent_data.get("system_message", ""),
                    },
                )
                agents.append(agent)
            elif isinstance(agent_data, str):
                # Simple agent ID string
                agents.append(AgentNode(
                    agent_id=agent_data,
                    agent_role=AgentRole.WORKER,
                ))

        return agents

    def _map_role(self, role_str: str) -> AgentRole:
        """Map role string to AgentRole enum."""
        role_lower = role_str.lower()

        # Orchestrator/planner roles
        if any(r in role_lower for r in ["orchestrator", "planner", "coordinator", "manager"]):
            return AgentRole.ORCHESTRATOR

        # Specialist roles
        if any(r in role_lower for r in ["specialist", "expert", "analyst"]):
            return AgentRole.SPECIALIST

        # Validator roles
        if any(r in role_lower for r in ["validator", "reviewer", "checker", "tester"]):
            return AgentRole.VALIDATOR

        # Messenger roles (in games like werewolf)
        if any(r in role_lower for r in ["messenger", "seer"]):
            return AgentRole.MESSENGER

        # Default to worker
        return AgentRole.WORKER

    def _parse_agent_trajectories(
        self,
        log_data: Dict[str, Any],
        agents: List[AgentNode],
    ) -> Dict[str, Trajectory]:
        """
        Parse individual agent trajectories from log data.

        Args:
            log_data: Log data dictionary
            agents: List of agent nodes

        Returns:
            Dictionary mapping agent_id to Trajectory
        """
        trajectories = {}

        # Get actions grouped by agent
        actions_data = log_data.get("actions", [])
        messages_data = log_data.get("messages", [])

        # Group actions by agent
        agent_actions: Dict[str, List[Dict]] = {a.agent_id: [] for a in agents}

        for action in actions_data:
            if isinstance(action, dict):
                agent_id = action.get("agent_id", action.get("agent", "unknown"))
                if agent_id in agent_actions:
                    agent_actions[agent_id].append(action)

        # Also extract actions from messages
        for message in messages_data:
            if isinstance(message, dict):
                sender = message.get("sender", message.get("from", message.get("agent_id")))
                if sender and sender in agent_actions:
                    # Convert message to action format
                    action_entry = {
                        "action_type": "send_message",
                        "agent_id": sender,
                        "content": message.get("content", ""),
                        "recipient": message.get("recipient", message.get("to")),
                        "timestamp": message.get("timestamp"),
                    }
                    agent_actions[sender].append(action_entry)

        # Create trajectory for each agent
        for agent in agents:
            agent_id = agent.agent_id
            actions = self._parse_actions(agent_actions.get(agent_id, []), agent_id)

            trajectory = Trajectory(
                trajectory_id=f"{agent_id}_trajectory",
                source="marble",
                task_description=f"Agent {agent_id} actions",
                is_attack=False,
                actions=actions,
                initial_state=State(),
                final_state=None,
                metadata={"agent_id": agent_id, "agent_role": agent.agent_role.value},
            )
            trajectories[agent_id] = trajectory

        return trajectories

    def _parse_actions(
        self,
        actions_data: List[Dict[str, Any]],
        agent_id: str,
    ) -> List[Action]:
        """
        Parse action entries into Action objects.

        Args:
            actions_data: List of action dictionaries
            agent_id: Agent ID

        Returns:
            List of Action objects
        """
        actions = []

        for idx, action_data in enumerate(actions_data):
            action_type_str = action_data.get("action_type", action_data.get("type", "unknown"))
            action_type = MARBLE_ACTION_MAPPING.get(
                action_type_str.lower(),
                ActionType.TOOL_CALL
            )

            # Extract target
            target = action_data.get("target", action_data.get("tool", action_type_str))

            # Extract timestamp
            timestamp = action_data.get("timestamp")
            if timestamp and isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp).timestamp()
                except ValueError:
                    timestamp = None

            # Extract result
            result = action_data.get("result", action_data.get("output", action_data.get("content")))
            if result and not isinstance(result, str):
                result = json.dumps(result)

            action = Action(
                action_id=idx,
                action_type=action_type,
                target=str(target) if target else action_type_str,
                context={
                    "agent_id": agent_id,
                    "original_type": action_type_str,
                    "args": action_data.get("args", action_data.get("parameters", {})),
                },
                result=result[:2000] if result else None,
                timestamp=timestamp,
                domain=f"marble.{agent_id}",
                data_produced=action_data.get("data_produced", []),
                data_consumed=action_data.get("data_consumed", []),
                raw_data=action_data,
            )
            actions.append(action)

        return actions

    def _parse_inter_agent_flows(
        self,
        log_data: Dict[str, Any],
        agents: List[AgentNode],
    ) -> List[InterAgentFlow]:
        """
        Parse inter-agent communication flows.

        Args:
            log_data: Log data dictionary
            agents: List of agent nodes

        Returns:
            List of InterAgentFlow objects
        """
        flows = []
        agent_ids = {a.agent_id for a in agents}

        # Parse messages for flows
        messages = log_data.get("messages", [])
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue

            sender = message.get("sender", message.get("from", message.get("agent_id")))
            recipient = message.get("recipient", message.get("to"))
            content = message.get("content", "")

            # Skip if we can't identify sender
            if not sender:
                continue

            # Determine flow type
            msg_type = message.get("type", message.get("message_type", ""))
            flow_type = self._determine_flow_type(msg_type, content)

            # Handle broadcast (no specific recipient)
            if not recipient or recipient == "all":
                # Create flow to each other agent
                for agent_id in agent_ids:
                    if agent_id != sender:
                        flow = InterAgentFlow(
                            source_agent=str(sender),
                            target_agent=agent_id,
                            data_item=content[:200] if content else f"message_{msg_idx}",
                            flow_type=InterAgentFlowType.BROADCAST,
                            metadata={
                                "message_index": msg_idx,
                                "original_type": msg_type,
                            },
                        )
                        flows.append(flow)
            else:
                # Single recipient flow
                flow = InterAgentFlow(
                    source_agent=str(sender),
                    target_agent=str(recipient),
                    data_item=content[:200] if content else f"message_{msg_idx}",
                    flow_type=flow_type,
                    metadata={
                        "message_index": msg_idx,
                        "original_type": msg_type,
                    },
                )
                flows.append(flow)

        # Parse delegations
        delegations = log_data.get("delegations", log_data.get("task_assignments", []))
        for deleg in delegations:
            if isinstance(deleg, dict):
                from_agent = deleg.get("from", deleg.get("delegator"))
                to_agent = deleg.get("to", deleg.get("delegatee"))
                task = deleg.get("task", deleg.get("description", ""))

                if from_agent and to_agent:
                    flow = create_delegation_flow(
                        orchestrator_id=str(from_agent),
                        worker_id=str(to_agent),
                        task=task[:200] if task else "delegation",
                    )
                    flows.append(flow)

        # Parse shared memory updates as synchronization flows
        memory_updates = log_data.get("shared_memory_updates", log_data.get("memory_updates", []))
        for update in memory_updates:
            if isinstance(update, dict):
                agent_id = update.get("agent_id", update.get("source"))
                key = update.get("key", "")

                if agent_id:
                    # Sync flow to all other agents
                    for other_id in agent_ids:
                        if other_id != agent_id:
                            flow = InterAgentFlow(
                                source_agent=str(agent_id),
                                target_agent=other_id,
                                data_item=f"memory:{key}",
                                flow_type=InterAgentFlowType.SYNCHRONIZATION,
                                metadata={"memory_key": key},
                            )
                            flows.append(flow)

        return flows

    def _determine_flow_type(self, msg_type: str, content: str) -> InterAgentFlowType:
        """Determine flow type from message type and content."""
        msg_type_lower = msg_type.lower() if msg_type else ""
        content_lower = content.lower() if content else ""

        if "delegation" in msg_type_lower or "assign" in msg_type_lower:
            return InterAgentFlowType.DELEGATION
        if "response" in msg_type_lower or "result" in msg_type_lower:
            return InterAgentFlowType.RESPONSE
        if "query" in msg_type_lower or "question" in msg_type_lower:
            return InterAgentFlowType.QUERY
        if "sync" in msg_type_lower or "update" in msg_type_lower:
            return InterAgentFlowType.SYNCHRONIZATION

        # Check content for delegation patterns
        if any(p in content_lower for p in ["please do", "your task is", "i need you to"]):
            return InterAgentFlowType.DELEGATION
        if any(p in content_lower for p in ["here is the result", "i completed", "done"]):
            return InterAgentFlowType.RESPONSE

        return InterAgentFlowType.DELEGATION  # Default

    def _detect_attack(
        self,
        log_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Detect if trajectory contains an attack.

        MARBLE is primarily a benign benchmark, but may include attack scenarios
        or compromised agent simulations.

        Args:
            log_data: Log data dictionary

        Returns:
            Tuple of (is_attack, attack_type, compromised_agent_ids)
        """
        # Check explicit attack markers
        if log_data.get("is_attack", False):
            return (
                True,
                log_data.get("attack_type", "unknown"),
                log_data.get("compromised_agents", []),
            )

        # Check for adversarial scenario (werewolf has adversarial roles)
        scenario = log_data.get("scenario", "")
        if scenario.lower() == "werewolf":
            # In werewolf, werewolves are technically "adversarial"
            agents = log_data.get("agents", [])
            werewolves = [
                a.get("id", a.get("agent_id"))
                for a in agents
                if isinstance(a, dict) and "werewolf" in a.get("role", "").lower()
            ]
            if werewolves:
                return (True, "adversarial_game", werewolves)

        return (False, None, [])

    def _extract_timestamps(
        self,
        log_data: Dict[str, Any],
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract start and end timestamps."""
        start_time = None
        end_time = None

        # Try explicit fields
        if "start_time" in log_data:
            start_time = self._parse_timestamp(log_data["start_time"])
        if "end_time" in log_data:
            end_time = self._parse_timestamp(log_data["end_time"])

        # Infer from actions/messages
        if not start_time or not end_time:
            timestamps = []
            for action in log_data.get("actions", []):
                if isinstance(action, dict) and action.get("timestamp"):
                    ts = self._parse_timestamp(action["timestamp"])
                    if ts:
                        timestamps.append(ts)
            for message in log_data.get("messages", []):
                if isinstance(message, dict) and message.get("timestamp"):
                    ts = self._parse_timestamp(message["timestamp"])
                    if ts:
                        timestamps.append(ts)

            if timestamps:
                if not start_time:
                    start_time = min(timestamps)
                if not end_time:
                    end_time = max(timestamps)

        return start_time, end_time

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        return None

    def _extract_success(self, log_data: Dict[str, Any]) -> Optional[bool]:
        """Determine if task was successful."""
        # Check explicit success field
        if "success" in log_data:
            return bool(log_data["success"])
        if "task_success" in log_data:
            return bool(log_data["task_success"])
        if "completed" in log_data:
            return bool(log_data["completed"])

        # Check milestone completion
        milestones = log_data.get("milestones", log_data.get("checkpoints", []))
        if milestones:
            completed = sum(
                1 for m in milestones
                if isinstance(m, dict) and m.get("completed", m.get("achieved"))
            )
            total = len(milestones)
            if total > 0:
                return completed >= total * 0.8  # 80% threshold

        return None

    def _extract_metadata(
        self,
        log_data: Dict[str, Any],
        log_path: str,
    ) -> Dict[str, Any]:
        """Extract metadata from log data."""
        metadata = {
            "scenario": log_data.get("scenario", log_data.get("task_type")),
            "task_id": log_data.get("task_id", log_data.get("test_case_id")),
            "coordination_protocol": log_data.get("coordination_protocol", log_data.get("topology")),
            "difficulty": log_data.get("difficulty"),
            "num_agents": len(log_data.get("agents", [])),
            "num_messages": len(log_data.get("messages", [])),
            "num_actions": len(log_data.get("actions", [])),
            "milestones_completed": self._count_completed_milestones(log_data),
            "milestones_total": len(log_data.get("milestones", log_data.get("checkpoints", []))),
            "coordination_score": log_data.get("coordination_score"),
            "communication_score": log_data.get("communication_score"),
            "planning_score": log_data.get("planning_score"),
            "kpi": log_data.get("kpi", log_data.get("task_score")),
            "log_path": log_path,
        }

        return {k: v for k, v in metadata.items() if v is not None}

    def _count_completed_milestones(self, log_data: Dict[str, Any]) -> int:
        """Count completed milestones."""
        milestones = log_data.get("milestones", log_data.get("checkpoints", []))
        return sum(
            1 for m in milestones
            if isinstance(m, dict) and m.get("completed", m.get("achieved"))
        )

    def _flatten_to_single_trajectory(
        self,
        ma_trajectory: MultiAgentTrajectory,
    ) -> Trajectory:
        """
        Flatten a MultiAgentTrajectory to a single Trajectory.

        This creates a unified view by interleaving agent actions
        chronologically.

        Args:
            ma_trajectory: MultiAgentTrajectory object

        Returns:
            Flattened Trajectory object
        """
        # Collect all actions with agent metadata
        all_actions = []
        for agent_id, agent_traj in ma_trajectory.trajectories.items():
            for action in agent_traj.actions:
                # Add agent context
                action.context["multi_agent_source"] = agent_id
                all_actions.append(action)

        # Sort by timestamp if available, otherwise by action_id
        all_actions.sort(
            key=lambda a: (a.timestamp or 0, a.action_id)
        )

        # Renumber action IDs
        for idx, action in enumerate(all_actions):
            action.action_id = idx

        # Create observation chunks from inter-agent flows
        observation_chunks = []
        for idx, flow in enumerate(ma_trajectory.inter_agent_flows):
            chunk = ObservationChunk(
                chunk_id=f"flow_{idx}",
                content=flow.data_item,
                source=f"agent:{flow.source_agent}",
                domain=f"marble.{ma_trajectory.metadata.get('scenario', 'unknown')}",
                metadata={
                    "flow_type": flow.flow_type.value,
                    "target_agent": flow.target_agent,
                    "is_malicious": flow.is_malicious,
                },
            )
            observation_chunks.append(chunk)

        return Trajectory(
            trajectory_id=ma_trajectory.trajectory_id,
            source="marble",
            task_description=ma_trajectory.task_description,
            is_attack=ma_trajectory.is_attack,
            actions=all_actions,
            observation_chunks=observation_chunks,
            initial_state=State(
                accumulated_data={
                    "num_agents": len(ma_trajectory.agents),
                    "coordination_protocol": ma_trajectory.metadata.get("coordination_protocol"),
                }
            ),
            final_state=State(
                accumulated_data={
                    "inter_agent_flows": len(ma_trajectory.inter_agent_flows),
                    "total_actions": ma_trajectory.total_actions(),
                }
            ),
            metadata={
                **ma_trajectory.metadata,
                "multi_agent": True,
                "agent_ids": [a.agent_id for a in ma_trajectory.agents],
            },
            success=ma_trajectory.success,
            error_message=ma_trajectory.error_message,
        )

    def get_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of scenarios and their info."""
        return MARBLE_SCENARIOS.copy()

    def get_protocols(self) -> Dict[str, str]:
        """Get dictionary of coordination protocols."""
        return COORDINATION_PROTOCOLS.copy()

    def get_statistics(
        self,
        trajectories: List[MultiAgentTrajectory],
    ) -> Dict[str, Any]:
        """
        Get statistics about extracted trajectories.

        Args:
            trajectories: List of MultiAgentTrajectory objects

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total": len(trajectories),
            "attacks": sum(1 for t in trajectories if t.is_attack),
            "benign": sum(1 for t in trajectories if not t.is_attack),
            "by_scenario": {},
            "by_protocol": {},
            "avg_agents": 0,
            "avg_actions": 0,
            "avg_flows": 0,
            "total_agents": 0,
            "total_actions": 0,
            "total_flows": 0,
        }

        for t in trajectories:
            scenario = t.metadata.get("scenario", "unknown")
            protocol = t.metadata.get("coordination_protocol", "unknown")

            stats["by_scenario"][scenario] = stats["by_scenario"].get(scenario, 0) + 1
            stats["by_protocol"][protocol] = stats["by_protocol"].get(protocol, 0) + 1
            stats["total_agents"] += len(t.agents)
            stats["total_actions"] += t.total_actions()
            stats["total_flows"] += len(t.inter_agent_flows)

        if trajectories:
            stats["avg_agents"] = stats["total_agents"] / len(trajectories)
            stats["avg_actions"] = stats["total_actions"] / len(trajectories)
            stats["avg_flows"] = stats["total_flows"] / len(trajectories)

        return stats
