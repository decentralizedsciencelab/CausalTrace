"""
ASB (Agent Security Bench) trajectory extractor.

This module extracts trajectories from the Agent Security Bench benchmark.
ASB evaluates LLM agents against 4 attack types across 10 agent scenarios.

ASB Format Notes:
- Attack tools in data/all_attack_tools.jsonl
- Normal tools in data/all_normal_tools.jsonl
- 4 attack types: DPI, OPI/IPI, Memory Poisoning, PoT Backdoor
- 10 agent scenarios (finance, healthcare, legal, education, etc.)
- Each tool has: Attacker Tool, Attacker Instruction, Description, Attack goal, Attack Type

Attack Types:
- DPI (Direct Prompt Injection): Tampering with user prompts
- OPI/IPI (Observation Prompt Injection): Altering observation data
- Memory Poisoning: Injecting malicious plans into agent memory
- PoT Backdoor (Plan-of-Thought): Concealed actions via triggers

Reference: https://github.com/agiresearch/ASB
Paper: Agent Security Bench (ICLR 2025)
"""

import json
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files


class ASBAttackType(Enum):
    """Attack types in ASB benchmark."""
    NONE = "none"
    DPI = "dpi"  # Direct Prompt Injection
    OPI = "opi"  # Observation Prompt Injection
    IPI = "ipi"  # Indirect Prompt Injection (alias for OPI)
    MEMORY_POISONING = "memory_poisoning"
    POT_BACKDOOR = "pot_backdoor"  # Plan-of-Thought Backdoor
    STEALTHY = "stealthy"
    AGGRESSIVE = "aggressive"

    @classmethod
    def from_string(cls, s: str) -> "ASBAttackType":
        """Parse attack type from string."""
        s_lower = s.lower().strip()
        mapping = {
            "dpi": cls.DPI,
            "direct": cls.DPI,
            "direct prompt injection": cls.DPI,
            "opi": cls.OPI,
            "ipi": cls.IPI,
            "observation": cls.OPI,
            "indirect": cls.OPI,
            "memory": cls.MEMORY_POISONING,
            "memory_poisoning": cls.MEMORY_POISONING,
            "memory poisoning": cls.MEMORY_POISONING,
            "pot": cls.POT_BACKDOOR,
            "pot_backdoor": cls.POT_BACKDOOR,
            "plan-of-thought": cls.POT_BACKDOOR,
            "backdoor": cls.POT_BACKDOOR,
            "stealthy": cls.STEALTHY,
            "stealthy attack": cls.STEALTHY,
            "aggressive": cls.AGGRESSIVE,
            "none": cls.NONE,
        }
        return mapping.get(s_lower, cls.NONE)


# Agent scenarios in ASB
ASB_SCENARIOS = {
    "system_admin_agent": {
        "domain": "it_security",
        "description": "IT system administration and security",
    },
    "financial_analyst_agent": {
        "domain": "finance",
        "description": "Financial analysis and advisory",
    },
    "medical_advisor_agent": {
        "domain": "healthcare",
        "description": "Medical advice and consultation",
    },
    "legal_consultant_agent": {
        "domain": "legal",
        "description": "Legal consultation and advice",
    },
    "education_consultant_agent": {
        "domain": "education",
        "description": "Educational guidance and tutoring",
    },
    "psychological_counselor_agent": {
        "domain": "mental_health",
        "description": "Psychological counseling and support",
    },
    "ecommerce_manager_agent": {
        "domain": "ecommerce",
        "description": "E-commerce management and sales",
    },
    "aerospace_engineer_agent": {
        "domain": "aerospace",
        "description": "Aerospace engineering and design",
    },
    "academic_search_agent": {
        "domain": "research",
        "description": "Academic research and literature search",
    },
    "autonomous_driving_agent": {
        "domain": "automotive",
        "description": "Autonomous vehicle operation",
    },
}

# Attack tool categories and their action types
ATTACK_TOOL_PATTERNS = {
    "exfiltrate": ActionType.READ,
    "steal": ActionType.READ,
    "harvest": ActionType.READ,
    "credential": ActionType.READ,
    "escalat": ActionType.WRITE,
    "hijack": ActionType.WRITE,
    "inject": ActionType.WRITE,
    "backdoor": ActionType.CODE_EXECUTION,
    "execute": ActionType.CODE_EXECUTION,
    "intercept": ActionType.READ,
    "spoof": ActionType.WRITE,
    "bypass": ActionType.TOOL_CALL,
    "malicious": ActionType.TOOL_CALL,
}


class ASBExtractor(BaseExtractor):
    """
    Extract trajectories from Agent Security Bench (ASB) benchmark.

    ASB provides 400+ attack tools across 10 agent scenarios with 4 attack types.
    This extractor converts attack tool specifications into CausalTrace trajectories.

    Attributes:
        attack_types: Filter by specific attack types
        scenarios: Filter by agent scenarios
        include_aggressive: Include aggressive attack tools
        verbose: Whether to print verbose output
    """

    def __init__(
        self,
        attack_types: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        include_aggressive: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the ASB extractor.

        Args:
            attack_types: Filter by attack types (e.g., ["dpi", "memory_poisoning"])
            scenarios: Filter by scenarios (e.g., ["finance", "healthcare"])
            include_aggressive: Include aggressive attack tools
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.attack_types = [ASBAttackType.from_string(at) for at in attack_types] if attack_types else None
        self.scenarios = scenarios
        self.include_aggressive = include_aggressive
        self._normal_tools: Dict[str, Dict] = {}
        self._log(f"Initialized ASBExtractor")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single ASB log/tool file into a Trajectory.

        Args:
            log_path: Path to the log file (JSON or JSONL)

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"File not found: {log_path}")
            return None

        try:
            # Handle JSONL (single line) or JSON
            path = Path(log_path)
            if path.suffix == ".jsonl":
                with open(log_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        data = json.loads(line)
                        return self._parse_attack_tool(data, 0, log_path)
            else:
                data = read_json(log_path)
                if isinstance(data, list) and len(data) > 0:
                    return self._parse_attack_tool(data[0], 0, log_path)
                elif isinstance(data, dict):
                    return self._parse_attack_tool(data, 0, log_path)

        except Exception as e:
            self._log(f"Error parsing {log_path}: {e}")

        return None

    def extract_from_directory(
        self,
        dir_path: str,
        pattern: str = "*.jsonl",
    ) -> List[Trajectory]:
        """
        Parse all ASB attack tools from a directory.

        Expected structure:
        - data/all_attack_tools.jsonl
        - data/all_normal_tools.jsonl

        Args:
            dir_path: Path to ASB data directory
            pattern: Glob pattern for matching files

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        trajectories = []

        # Find attack tools file
        attack_files = find_files(dir_path, "*attack_tools*.jsonl", recursive=True)

        # Optionally load normal tools for reference
        normal_files = find_files(dir_path, "*normal_tools*.jsonl", recursive=True)
        for nf in normal_files:
            self._load_normal_tools(nf)

        self._log(f"Found {len(attack_files)} attack tool files")

        for attack_file in attack_files:
            file_trajectories = self._extract_from_jsonl(attack_file)
            trajectories.extend(file_trajectories)

        self._log(f"Extracted {len(trajectories)} trajectories from ASB")
        return trajectories

    def _load_normal_tools(self, path: str) -> None:
        """Load normal (benign) tools for reference."""
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tool = json.loads(line)
                        name = tool.get("Tool", tool.get("name", ""))
                        if name:
                            self._normal_tools[name] = tool
            self._log(f"Loaded {len(self._normal_tools)} normal tools")
        except Exception as e:
            self._log(f"Error loading normal tools: {e}")

    def _extract_from_jsonl(self, jsonl_path: str) -> List[Trajectory]:
        """Extract trajectories from a JSONL file."""
        trajectories = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    tool_data = json.loads(line)
                    trajectory = self._parse_attack_tool(tool_data, idx, jsonl_path)
                    if trajectory and self._should_include(trajectory, tool_data):
                        trajectories.append(trajectory)
                except json.JSONDecodeError as e:
                    self._log(f"Error parsing line {idx}: {e}")

        return trajectories

    def _should_include(self, trajectory: Trajectory, tool_data: Dict) -> bool:
        """Check if trajectory passes filters."""
        # Filter by attack type
        if self.attack_types:
            attack_type = ASBAttackType.from_string(
                tool_data.get("Attack Type", "none")
            )
            if attack_type not in self.attack_types:
                return False

        # Filter by scenario
        if self.scenarios:
            agent = tool_data.get("Corresponding Agent", "")
            scenario_match = any(s.lower() in agent.lower() for s in self.scenarios)
            if not scenario_match:
                return False

        # Filter aggressive tools
        if not self.include_aggressive:
            if tool_data.get("Aggressive", "").lower() == "true":
                return False

        return True

    def _parse_attack_tool(
        self,
        tool_data: Dict[str, Any],
        idx: int,
        source_path: str,
    ) -> Optional[Trajectory]:
        """
        Parse an ASB attack tool into a Trajectory.

        Args:
            tool_data: Dictionary containing attack tool data
            idx: Index in source file
            source_path: Path to source file

        Returns:
            Trajectory object or None
        """
        # Extract core fields
        tool_name = tool_data.get("Attacker Tool", tool_data.get("Tool", "unknown"))
        instruction = tool_data.get("Attacker Instruction", "")
        description = tool_data.get("Description", "")
        attack_goal = tool_data.get("Attack goal", "")
        attack_type_str = tool_data.get("Attack Type", "none")
        agent = tool_data.get("Corresponding Agent", "unknown")
        is_aggressive = tool_data.get("Aggressive", "False").lower() == "true"

        if not instruction:
            return None

        # Parse attack type
        attack_type = ASBAttackType.from_string(attack_type_str)

        # Get scenario info
        scenario_info = self._get_scenario_info(agent)

        # Generate trajectory ID
        trajectory_id = f"asb_{tool_name}_{attack_type.value}_{idx}"

        # Create observation chunks
        observation_chunks = []

        # Chunk: Attack instruction
        instruction_chunk = ObservationChunk(
            chunk_id="instruction_0",
            content=instruction,
            source="attacker_instruction",
            domain=f"asb.{scenario_info['domain']}",
            metadata={
                "tool_name": tool_name,
                "attack_type": attack_type.value,
                "contains_injection": True,
            },
        )
        observation_chunks.append(instruction_chunk)

        # Chunk: Tool description
        if description:
            desc_chunk = ObservationChunk(
                chunk_id="description_1",
                content=description,
                source="tool_description",
                domain=f"asb.{scenario_info['domain']}",
                metadata={
                    "tool_name": tool_name,
                    "is_attack_tool": True,
                },
            )
            observation_chunks.append(desc_chunk)

        # Create actions
        actions = []

        # Action 1: Attacker provides instruction
        action_1 = Action(
            action_id=0,
            action_type=ActionType.TYPE,
            target="attacker_input",
            context={
                "role": "attacker",
                "tool_name": tool_name,
                "attack_type": attack_type.value,
            },
            result=None,
            domain=f"asb.{scenario_info['domain']}",
            data_produced=["attack_instruction"],
            provenance=ProvenanceData(
                observation_chunks=["instruction_0"],
                confidence_scores={"instruction_0": 1.0},
                attribution_method="ground_truth",
                is_untrusted=True,
                injection_detected=True,
            ),
            raw_data={"instruction": instruction},
        )
        actions.append(action_1)

        # Action 2: Agent executes malicious tool
        action_type_mapped = self._get_action_type(tool_name)
        action_2 = Action(
            action_id=1,
            action_type=action_type_mapped,
            target=tool_name,
            context={
                "tool_name": tool_name,
                "attack_goal": attack_goal,
                "is_attack_tool": True,
                "is_aggressive": is_aggressive,
            },
            result=f"Executing attack: {attack_goal}",
            domain=f"asb.{scenario_info['domain']}",
            data_consumed=["attack_instruction"],
            data_produced=["attack_result"],
            provenance=ProvenanceData(
                observation_chunks=["instruction_0"],
                confidence_scores={"instruction_0": 0.95},
                attribution_method="tool_execution",
                is_untrusted=True,
                injection_detected=True,
            ),
            raw_data=tool_data,
        )
        actions.append(action_2)

        # Create states
        initial_state = State()
        initial_state.authenticated_services.append(f"asb.{scenario_info['domain']}")
        initial_state.accumulated_data["agent_type"] = agent

        final_state = State()
        final_state.accumulated_data["attack_executed"] = True
        final_state.accumulated_data["attack_type"] = attack_type.value
        final_state.accumulated_data["attack_goal"] = attack_goal

        # Build metadata
        metadata = {
            "tool_name": tool_name,
            "attack_type": attack_type.value,
            "attack_goal": attack_goal,
            "corresponding_agent": agent,
            "domain": scenario_info["domain"],
            "is_aggressive": is_aggressive,
            "source_path": source_path,
            "benchmark": "asb",
        }

        # Task description
        task_description = f"[ASB/{attack_type.value}] {tool_name}: {attack_goal[:100]}"

        return Trajectory(
            trajectory_id=trajectory_id,
            source="asb",
            task_description=task_description,
            is_attack=True,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=True,  # ASB tools are designed to succeed
        )

    def _get_scenario_info(self, agent: str) -> Dict[str, str]:
        """Get scenario info from agent name."""
        agent_lower = agent.lower()

        for scenario, info in ASB_SCENARIOS.items():
            if scenario.lower() in agent_lower:
                return info

        # Infer from keywords
        if "finance" in agent_lower or "financial" in agent_lower:
            return {"domain": "finance", "description": "Financial operations"}
        if "medical" in agent_lower or "health" in agent_lower:
            return {"domain": "healthcare", "description": "Healthcare operations"}
        if "legal" in agent_lower:
            return {"domain": "legal", "description": "Legal operations"}

        return {"domain": "general", "description": "General agent operations"}

    def _get_action_type(self, tool_name: str) -> ActionType:
        """Infer ActionType from tool name."""
        tool_lower = tool_name.lower()

        for pattern, action_type in ATTACK_TOOL_PATTERNS.items():
            if pattern in tool_lower:
                return action_type

        return ActionType.TOOL_CALL

    def get_attack_types(self) -> Dict[str, str]:
        """Get dictionary of attack types and descriptions."""
        return {
            "dpi": "Direct Prompt Injection - tampering with user prompts",
            "opi": "Observation Prompt Injection - altering observation data",
            "memory_poisoning": "Memory Poisoning - injecting malicious plans into memory",
            "pot_backdoor": "Plan-of-Thought Backdoor - concealed actions via triggers",
        }

    def get_scenarios(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of agent scenarios."""
        return ASB_SCENARIOS.copy()

    def get_statistics(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Get statistics about extracted trajectories."""
        stats = {
            "total": len(trajectories),
            "by_attack_type": {},
            "by_domain": {},
            "by_agent": {},
            "aggressive_count": 0,
        }

        for t in trajectories:
            attack_type = t.metadata.get("attack_type", "unknown")
            domain = t.metadata.get("domain", "unknown")
            agent = t.metadata.get("corresponding_agent", "unknown")
            is_aggressive = t.metadata.get("is_aggressive", False)

            stats["by_attack_type"][attack_type] = stats["by_attack_type"].get(attack_type, 0) + 1
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1
            if is_aggressive:
                stats["aggressive_count"] += 1

        return stats
