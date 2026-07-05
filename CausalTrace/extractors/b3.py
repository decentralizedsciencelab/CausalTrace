"""
B3 (Backbone Breaker Benchmark) trajectory extractor.

This module extracts trajectories from Lakera's B3 benchmark dataset.
B3 contains successful prompt injection attacks from 194K+ human red-team attempts
against LLM-powered agents with varying defense levels.

B3 Format Notes:
- JSON array of attack samples
- Each sample has: app_name, app_slug, level (0/1/2), attack, attack_index, original_llm
- Security levels: 0=weak, 1=medium, 2=strong
- 10 agentic threat scenarios covering different attack types
- All samples are successful attacks (passed defense)

Key fields:
- app_slug: Identifies the threat scenario (e.g., cco_coach, trip_planner_phishing)
- level: Defense level (0=weak, 1=medium, 2=strong)
- attack: The successful attack prompt text
- original_llm: The LLM model that was attacked

Reference: https://huggingface.co/datasets/Lakera/b3-agent-security-benchmark-weak
Paper: https://www.lakera.ai/blog/the-backbone-breaker-benchmark
"""

import json
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files


# Security level mapping
SECURITY_LEVELS = {
    0: "weak",
    1: "medium",
    2: "strong",
    "weak": "weak",
    "medium": "medium",
    "strong": "strong",
}

# Threat scenario mapping: app_slug -> (threat_category, target_action_type)
THREAT_SCENARIOS = {
    "cco_coach": ("prompt_exfil", ActionType.AGENT_RESPONSE),
    "trip_planner_phishing": ("phishing", ActionType.NAVIGATE),
    "code_review_malicious": ("code_injection", ActionType.CODE_EXECUTION),
    "email_tool_invocation": ("tool_misuse", ActionType.TOOL_CALL),
    "legal_ti_exfiltration": ("data_exfil", ActionType.READ),
    "solace_profane_chat": ("content_injection", ActionType.AGENT_RESPONSE),
    "chatapp_refuse": ("policy_bypass", ActionType.AGENT_RESPONSE),
    "analytics_misleading": ("content_injection", ActionType.WRITE),
}

# Injection patterns for detection
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"new\s+instructions?",
    r"system\s*:?\s*prompt",
    r"disregard\s+(all\s+)?above",
    r"override\s+(the\s+)?instructions?",
    r"you\s+are\s+now",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"<\|system\|>",
    r"```\s*system",
    r"DEBUG\s*=\s*TRUE",
    r"sudo\s+su",
    r"root\s*\$",
]


class B3Extractor(BaseExtractor):
    """
    Extract trajectories from B3 (Backbone Breaker Benchmark) dataset.

    B3 contains successful attacks from 194K+ human red-team attempts against
    LLM agents with varying defense levels. All samples in the dataset are
    attacks that successfully bypassed the target model's defenses.

    Attributes:
        security_level: Filter by security level (None = all levels)
        threat_types: Filter by specific threat types
        verbose: Whether to print verbose output
    """

    def __init__(
        self,
        security_level: Optional[str] = None,
        threat_types: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the B3 extractor.

        Args:
            security_level: Filter by security level ("weak", "medium", "strong")
            threat_types: Filter by threat types (e.g., ["prompt_exfil", "phishing"])
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.security_level = security_level
        self.threat_types = threat_types
        self._log(f"Initialized B3Extractor (level={security_level}, threats={threat_types})")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single B3 log file into a Trajectory.

        For B3, a "log" is typically a JSON file containing an array of attack samples.
        This method extracts the first sample from the file.

        Args:
            log_path: Path to the JSON file

        Returns:
            Trajectory object or None if parsing fails
        """
        if not self._validate_path(log_path):
            self._log(f"File not found: {log_path}")
            return None

        try:
            data = read_json(log_path)

            # Handle both single sample and array of samples
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
            elif isinstance(data, dict):
                sample = data
            else:
                self._log(f"Unexpected data format in {log_path}")
                return None

            return self._parse_sample(sample, 0, log_path)

        except Exception as e:
            self._log(f"Error parsing {log_path}: {e}")
            return None

    def extract_from_directory(
        self,
        dir_path: str,
        pattern: str = "*.json",
    ) -> List[Trajectory]:
        """
        Parse all B3 samples from JSON files in a directory.

        B3 data is typically stored as a JSON array, so this method
        reads each JSON file and extracts all samples from it.

        Args:
            dir_path: Path to directory containing B3 JSON files
            pattern: Glob pattern for matching files

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        trajectories = []
        json_files = find_files(dir_path, pattern, recursive=True)

        self._log(f"Found {len(json_files)} JSON files in {dir_path}")

        for json_file in json_files:
            try:
                data = read_json(json_file)

                if isinstance(data, list):
                    # Array of samples
                    for idx, sample in enumerate(data):
                        trajectory = self._parse_sample(sample, idx, json_file)
                        if trajectory:
                            if self._should_include(trajectory):
                                trajectories.append(trajectory)
                elif isinstance(data, dict):
                    # Single sample
                    trajectory = self._parse_sample(data, 0, json_file)
                    if trajectory and self._should_include(trajectory):
                        trajectories.append(trajectory)

            except Exception as e:
                self._log(f"Error processing {json_file}: {e}")

        self._log(f"Extracted {len(trajectories)} trajectories from B3 data")
        return trajectories

    def extract_from_jsonl(self, jsonl_path: str) -> List[Trajectory]:
        """
        Extract trajectories from a JSONL file (one sample per line).

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(jsonl_path):
            self._log(f"File not found: {jsonl_path}")
            return []

        trajectories = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    trajectory = self._parse_sample(sample, idx, jsonl_path)
                    if trajectory and self._should_include(trajectory):
                        trajectories.append(trajectory)
                except json.JSONDecodeError as e:
                    self._log(f"Error parsing line {idx}: {e}")

        self._log(f"Extracted {len(trajectories)} trajectories from {jsonl_path}")
        return trajectories

    def _should_include(self, trajectory: Trajectory) -> bool:
        """Check if trajectory passes filters."""
        # Filter by security level
        if self.security_level:
            traj_level = trajectory.metadata.get("security_level")
            if traj_level != self.security_level:
                return False

        # Filter by threat type
        if self.threat_types:
            threat_type = trajectory.metadata.get("threat_category")
            if threat_type not in self.threat_types:
                return False

        return True

    def _parse_sample(
        self,
        sample: Dict[str, Any],
        idx: int,
        source_path: str,
    ) -> Optional[Trajectory]:
        """
        Parse a single B3 sample into a Trajectory.

        Args:
            sample: Dictionary containing B3 sample data
            idx: Index of sample in source file
            source_path: Path to source file

        Returns:
            Trajectory object or None
        """
        # Extract core fields
        attack_prompt = sample.get("attack", "")
        if not attack_prompt:
            return None

        app_slug = sample.get("app_slug", "unknown")
        app_name = sample.get("app_name", app_slug)
        level = sample.get("level", 0)
        attack_index = sample.get("attack_index", idx)
        original_llm = sample.get("original_llm", "unknown")

        # Normalize security level
        security_level = SECURITY_LEVELS.get(level, "unknown")

        # Get threat info
        threat_category, target_action_type = self._get_threat_info(app_slug)

        # Generate trajectory ID
        prompt_hash = hashlib.md5(attack_prompt.encode()).hexdigest()[:8]
        trajectory_id = f"b3_{app_slug}_{security_level}_{attack_index}_{prompt_hash}"

        # Create observation chunks
        observation_chunks = []

        # Chunk: Attack prompt
        attack_chunk = ObservationChunk(
            chunk_id="attack_0",
            content=attack_prompt,
            source="user_input",
            domain=f"b3.{app_slug}",
            metadata={
                "contains_injection": True,
                "threat_category": threat_category,
                "app_name": app_name,
            },
        )
        observation_chunks.append(attack_chunk)

        # Create actions
        actions = []

        # Action 1: User input (attack prompt)
        action_1 = Action(
            action_id=0,
            action_type=ActionType.TYPE,
            target="user_input",
            context={
                "role": "user",
                "app_slug": app_slug,
                "is_injection": True,
            },
            result=None,
            domain=f"b3.{app_slug}",
            data_produced=["attack_prompt"],
            provenance=ProvenanceData(
                observation_chunks=["attack_0"],
                confidence_scores={"attack_0": 1.0},
                attribution_method="ground_truth",
                is_untrusted=True,
                injection_detected=True,
            ),
            raw_data={"attack": attack_prompt},
        )
        actions.append(action_1)

        # Action 2: Agent compromised action
        action_2 = Action(
            action_id=1,
            action_type=target_action_type,
            target=self._get_attack_target(threat_category),
            context={
                "triggered_by": "attack_prompt",
                "threat_category": threat_category,
                "is_compromised": True,
            },
            result=f"Attack succeeded ({threat_category})",
            domain=f"b3.{app_slug}",
            data_consumed=["attack_prompt"],
            data_produced=[f"{threat_category}_result"],
            provenance=ProvenanceData(
                observation_chunks=["attack_0"],
                confidence_scores={"attack_0": 0.95},
                attribution_method="causal_inference",
                is_untrusted=True,
                injection_detected=True,
            ),
            raw_data={},
        )
        actions.append(action_2)

        # Create initial state
        initial_state = State()
        initial_state.authenticated_services.append(f"b3.{app_slug}")
        initial_state.accumulated_data["security_level"] = security_level

        # Create final state
        final_state = State()
        final_state.accumulated_data["attack_succeeded"] = True
        final_state.accumulated_data["threat_category"] = threat_category

        # Build metadata
        metadata = {
            "app_slug": app_slug,
            "app_name": app_name,
            "security_level": security_level,
            "threat_category": threat_category,
            "attack_index": attack_index,
            "original_llm": original_llm,
            "source_path": source_path,
            "benchmark": "b3",
        }

        # Create task description
        task_description = f"[B3/{app_slug}] Attack attempt at {security_level} defense level"

        return Trajectory(
            trajectory_id=trajectory_id,
            source="b3",
            task_description=task_description,
            is_attack=True,  # All B3 samples are successful attacks
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=True,  # B3 only contains successful attacks
        )

    def _get_threat_info(self, app_slug: str) -> Tuple[str, ActionType]:
        """Get threat category and action type for an app_slug."""
        app_slug_lower = app_slug.lower().replace("-", "_")

        if app_slug_lower in THREAT_SCENARIOS:
            return THREAT_SCENARIOS[app_slug_lower]

        # Infer from keywords
        if any(kw in app_slug_lower for kw in ["phish", "link", "url"]):
            return ("phishing", ActionType.NAVIGATE)
        if any(kw in app_slug_lower for kw in ["code", "exec", "malicious"]):
            return ("code_injection", ActionType.CODE_EXECUTION)
        if any(kw in app_slug_lower for kw in ["email", "tool"]):
            return ("tool_misuse", ActionType.TOOL_CALL)
        if any(kw in app_slug_lower for kw in ["exfil", "leak", "steal"]):
            return ("data_exfil", ActionType.READ)

        return ("prompt_injection", ActionType.AGENT_RESPONSE)

    def _get_attack_target(self, threat_category: str) -> str:
        """Get default target for a threat category."""
        targets = {
            "prompt_exfil": "system_prompt",
            "data_exfil": "sensitive_data",
            "phishing": "malicious_url",
            "code_injection": "code_execution",
            "content_injection": "response_content",
            "tool_misuse": "unauthorized_tool",
            "policy_bypass": "safety_guardrails",
        }
        return targets.get(threat_category, "agent_response")

    def get_statistics(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """
        Get statistics about extracted trajectories.

        Args:
            trajectories: List of extracted trajectories

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total": len(trajectories),
            "by_security_level": {},
            "by_threat_category": {},
            "by_app_slug": {},
            "by_llm": {},
        }

        for t in trajectories:
            level = t.metadata.get("security_level", "unknown")
            category = t.metadata.get("threat_category", "unknown")
            app = t.metadata.get("app_slug", "unknown")
            llm = t.metadata.get("original_llm", "unknown")

            stats["by_security_level"][level] = stats["by_security_level"].get(level, 0) + 1
            stats["by_threat_category"][category] = stats["by_threat_category"].get(category, 0) + 1
            stats["by_app_slug"][app] = stats["by_app_slug"].get(app, 0) + 1
            stats["by_llm"][llm] = stats["by_llm"].get(llm, 0) + 1

        return stats
