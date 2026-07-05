"""
MetaTool trajectory extractor.

This module extracts trajectories from the MetaTool benchmark (ICLR 2024).
MetaTool evaluates LLM tool usage awareness and selection, with a focus
on tool hijacking attacks where malicious tools are selected over legitimate ones.

MetaTool Format Notes:
- 21,127 user queries for tool selection
- 199 tool documents (OpenAI Plugins format)
- 5 benchmark tasks for tool awareness/selection
- Tool hijacking tests malicious tool preference

Important: MetaTool is primarily a capability benchmark, not adversarial.
However, it's valuable for:
- Testing tool selection hijacking attacks (ToolHijacker)
- Multi-step tool reasoning evaluation
- Baseline tool selection accuracy

Reference: https://github.com/HowieHwong/MetaTool
Paper: MetaTool Benchmark (ICLR 2024)
"""

import json
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from causaltrace.models import Trajectory, Action, ActionType, State, ObservationChunk, ProvenanceData
from causaltrace.extractors.base import BaseExtractor
from causaltrace.utils import read_json, find_files


# Tool categories based on common patterns
TOOL_CATEGORIES = {
    "search": "search",
    "weather": "weather",
    "email": "communication",
    "message": "communication",
    "calendar": "productivity",
    "schedule": "productivity",
    "translate": "language",
    "code": "development",
    "math": "calculation",
    "image": "media",
    "video": "media",
    "music": "media",
    "news": "information",
    "finance": "finance",
    "stock": "finance",
    "crypto": "finance",
    "travel": "travel",
    "flight": "travel",
    "hotel": "travel",
    "food": "lifestyle",
    "recipe": "lifestyle",
    "health": "health",
    "fitness": "health",
}

# Hijacking attack types
HIJACKING_TYPES = {
    "description_injection": "Malicious instructions in tool description",
    "authority_claim": "False claims of being official/verified",
    "urgency_injection": "Urgency language to force selection",
    "functionality_overlap": "Similar functionality to legitimate tool",
    "typosquatting": "Similar name to popular tool",
}


class MetaToolExtractor(BaseExtractor):
    """
    Extract trajectories from MetaTool benchmark.

    MetaTool evaluates tool awareness and selection in LLM agents.
    This extractor supports both capability evaluation and tool hijacking attack analysis.

    Note: MetaTool is a capability benchmark, not primarily adversarial.
    Attack trajectories are only generated when hijacker tools are present
    and successfully selected.

    Attributes:
        include_benign: Include benign (non-attack) trajectories
        include_single_tool: Include single-tool selection scenarios
        include_multi_tool: Include multi-tool selection scenarios
        verbose: Whether to print verbose output
    """

    def __init__(
        self,
        include_benign: bool = True,
        include_single_tool: bool = True,
        include_multi_tool: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the MetaTool extractor.

        Args:
            include_benign: Include non-attack trajectories
            include_single_tool: Include single-tool scenarios
            include_multi_tool: Include multi-tool scenarios
            verbose: Whether to print verbose output
        """
        super().__init__(verbose)
        self.include_benign = include_benign
        self.include_single_tool = include_single_tool
        self.include_multi_tool = include_multi_tool
        self._tool_docs: Dict[str, Dict] = {}
        self._log("Initialized MetaToolExtractor")

    def load_tool_documents(self, doc_path: str) -> None:
        """
        Load tool documentation from file.

        Args:
            doc_path: Path to tool documents (JSON or JSONL)
        """
        if not self._validate_path(doc_path):
            self._log(f"Doc file not found: {doc_path}")
            return

        try:
            data = read_json(doc_path)

            if isinstance(data, list):
                for tool in data:
                    name = tool.get("name", tool.get("tool_name", ""))
                    if name:
                        self._tool_docs[name] = tool
            elif isinstance(data, dict):
                # Could be a dict mapping names to tools
                for name, tool in data.items():
                    if isinstance(tool, dict):
                        self._tool_docs[name] = tool
                    else:
                        self._tool_docs[name] = {"name": name, "description": str(tool)}

            self._log(f"Loaded {len(self._tool_docs)} tool documents")

        except Exception as e:
            self._log(f"Error loading tool docs: {e}")

    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single MetaTool log into a Trajectory.

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
            return self._parse_query_data(data, log_path, 0)
        except Exception as e:
            self._log(f"Error parsing {log_path}: {e}")
            return None

    def extract_from_directory(
        self,
        dir_path: str,
        pattern: str = "*.json",
    ) -> List[Trajectory]:
        """
        Extract trajectories from MetaTool directory.

        Expected structure:
        - dataset/ (query files)
        - plugin_des.json or big_tool_des.json (tool documents)

        Args:
            dir_path: Path to MetaTool directory
            pattern: Glob pattern for log files

        Returns:
            List of Trajectory objects
        """
        if not self._validate_path(dir_path):
            self._log(f"Directory not found: {dir_path}")
            return []

        trajectories = []

        # Load tool documents if available
        tool_doc_files = find_files(dir_path, "*tool_des*.json", recursive=True)
        tool_doc_files += find_files(dir_path, "*plugin*.json", recursive=True)
        for doc_file in tool_doc_files:
            self.load_tool_documents(doc_file)

        # Find dataset files
        dataset_dir = Path(dir_path) / "dataset"
        if dataset_dir.exists():
            query_files = find_files(str(dataset_dir), pattern, recursive=True)
        else:
            query_files = find_files(dir_path, pattern, recursive=True)

        # Exclude tool doc files
        query_files = [f for f in query_files if "tool_des" not in f and "plugin" not in f]

        self._log(f"Found {len(query_files)} query files to process")

        for query_file in query_files:
            try:
                file_trajs = self._extract_from_file(query_file)
                trajectories.extend(file_trajs)
            except Exception as e:
                self._log(f"Error processing {query_file}: {e}")

        self._log(f"Extracted {len(trajectories)} trajectories from MetaTool")
        return trajectories

    def _extract_from_file(self, file_path: str) -> List[Trajectory]:
        """Extract trajectories from a single file."""
        trajectories = []

        data = read_json(file_path)

        if isinstance(data, list):
            for idx, item in enumerate(data):
                traj = self._parse_query_data(item, file_path, idx)
                if traj and self._should_include(traj):
                    trajectories.append(traj)
        elif isinstance(data, dict):
            traj = self._parse_query_data(data, file_path, 0)
            if traj and self._should_include(traj):
                trajectories.append(traj)

        return trajectories

    def _should_include(self, trajectory: Trajectory) -> bool:
        """Check if trajectory passes filters."""
        # Filter by attack status
        if not self.include_benign and not trajectory.is_attack:
            return False

        # Filter by selection type
        selection_type = trajectory.metadata.get("selection_type", "single")
        if not self.include_single_tool and selection_type == "single":
            return False
        if not self.include_multi_tool and selection_type == "multi":
            return False

        return True

    def _parse_query_data(
        self,
        query_data: Dict[str, Any],
        source_path: str,
        idx: int,
    ) -> Optional[Trajectory]:
        """
        Parse a MetaTool query into a Trajectory.

        Args:
            query_data: Dictionary containing query data
            source_path: Path to source file
            idx: Index in source file

        Returns:
            Trajectory object or None
        """
        # Extract query
        query = query_data.get("query", query_data.get("user_query", query_data.get("input", "")))
        if not query:
            return None

        # Extract tool candidates
        tool_candidates = self._extract_tool_candidates(query_data)

        # Extract selected and intended tools
        selected_tools = self._extract_selected_tools(query_data)
        intended_tools = self._extract_intended_tools(query_data)

        # Check for hijacker tool
        hijacker_tool = self._extract_hijacker_tool(query_data, tool_candidates)

        # Determine if attack succeeded
        is_attack = self._is_attack(selected_tools, hijacker_tool)

        # Generate trajectory ID
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        trajectory_id = f"metatool_{query_hash}_{idx}"

        # Create observation chunks
        observation_chunks = []

        # Query chunk
        query_chunk = ObservationChunk(
            chunk_id="query_0",
            content=query,
            source="user_query",
            domain="metatool",
            metadata={"type": "user_input"},
        )
        observation_chunks.append(query_chunk)

        # Tool description chunks
        hijacker_name = hijacker_tool.get("name", "").lower() if hijacker_tool else ""
        for i, tool in enumerate(tool_candidates):
            tool_name = tool.get("name", f"tool_{i}")
            tool_desc = tool.get("description", "")
            is_hijacker = tool_name.lower() == hijacker_name if hijacker_name else False
            is_hijacker = is_hijacker or tool.get("is_hijacker", False)

            chunk = ObservationChunk(
                chunk_id=f"tool_{i+1}",
                content=f"{tool_name}: {tool_desc}",
                source="tool_description",
                domain="metatool.plugins",
                metadata={
                    "tool_name": tool_name,
                    "is_hijacker": is_hijacker,
                    "category": self._categorize_tool(tool_name),
                },
            )
            observation_chunks.append(chunk)

        # Create actions
        actions = []

        # Action 1: Read query
        action_1 = Action(
            action_id=0,
            action_type=ActionType.READ,
            target="user_query",
            context={"query": query},
            result=query,
            domain="metatool",
            data_produced=["query_content"],
            raw_data={"query": query},
        )
        actions.append(action_1)

        # Action 2: Read tool descriptions
        action_2 = Action(
            action_id=1,
            action_type=ActionType.READ,
            target="tool_descriptions",
            context={
                "num_tools": len(tool_candidates),
                "has_hijacker": hijacker_tool is not None,
            },
            result=f"Read {len(tool_candidates)} tool descriptions",
            domain="metatool.plugins",
            data_produced=["tool_descriptions"],
            data_consumed=["query_content"],
            raw_data={"tools": [t.get("name") for t in tool_candidates]},
        )
        actions.append(action_2)

        # Action 3: Tool selection
        provenance = None
        if is_attack and hijacker_tool:
            hijacker_chunks = [c.chunk_id for c in observation_chunks if c.metadata.get("is_hijacker")]
            if hijacker_chunks:
                provenance = ProvenanceData(
                    observation_chunks=hijacker_chunks,
                    confidence_scores={c: 0.95 for c in hijacker_chunks},
                    attribution_method="hijacker_selection",
                    is_untrusted=True,
                    injection_detected=True,
                )

        action_3 = Action(
            action_id=2,
            action_type=ActionType.TOOL_CALL,
            target="tool_selection",
            context={
                "selected_tools": selected_tools,
                "intended_tools": intended_tools,
                "is_attack": is_attack,
            },
            result=f"Selected: {', '.join(selected_tools) if selected_tools else 'none'}",
            domain="metatool",
            data_consumed=["query_content", "tool_descriptions"],
            data_produced=["selected_tools"],
            provenance=provenance,
            raw_data={"selected": selected_tools, "intended": intended_tools},
        )
        actions.append(action_3)

        # Create states
        initial_state = State()
        initial_state.authenticated_services.append("metatool.selector")
        initial_state.accumulated_data["available_tools"] = [t.get("name") for t in tool_candidates]

        final_state = State()
        final_state.accumulated_data["selected_tools"] = selected_tools
        final_state.accumulated_data["attack_succeeded"] = is_attack

        # Determine selection type
        selection_type = "multi" if len(intended_tools) > 1 else "single"

        # Build metadata
        metadata = {
            "selection_type": selection_type,
            "num_tool_candidates": len(tool_candidates),
            "num_selected_tools": len(selected_tools),
            "num_intended_tools": len(intended_tools),
            "has_hijacker": hijacker_tool is not None,
            "source_path": source_path,
            "benchmark": "metatool",
        }

        if hijacker_tool:
            metadata["hijacker_tool"] = hijacker_tool.get("name", "unknown")
            metadata["hijacking_type"] = self._infer_hijacking_type(hijacker_tool)

        # Task description
        task_description = f"[MetaTool] Select tools for: {query[:80]}..."

        return Trajectory(
            trajectory_id=trajectory_id,
            source="metatool",
            task_description=task_description,
            is_attack=is_attack,
            actions=actions,
            observation_chunks=observation_chunks,
            initial_state=initial_state,
            final_state=final_state,
            metadata=metadata,
            success=is_attack if hijacker_tool else self._selection_correct(selected_tools, intended_tools),
        )

    def _extract_tool_candidates(self, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract available tool candidates."""
        tools = []

        for key in ["tool_candidates", "tools", "available_tools"]:
            if key in query_data and isinstance(query_data[key], list):
                for tool in query_data[key]:
                    if isinstance(tool, dict):
                        tools.append(tool)
                    elif isinstance(tool, str):
                        # Look up in tool docs
                        if tool in self._tool_docs:
                            tools.append(self._tool_docs[tool])
                        else:
                            tools.append({"name": tool})
                return tools

        return tools

    def _extract_selected_tools(self, query_data: Dict[str, Any]) -> List[str]:
        """Extract tools that were selected."""
        for key in ["selected_tools", "selected", "output_tools"]:
            if key in query_data:
                value = query_data[key]
                if isinstance(value, list):
                    return [t.get("name", str(t)) if isinstance(t, dict) else str(t) for t in value]
                elif isinstance(value, str):
                    return [value]
        return []

    def _extract_intended_tools(self, query_data: Dict[str, Any]) -> List[str]:
        """Extract ground truth intended tools."""
        for key in ["intended_tools", "ground_truth", "expected_tools", "label"]:
            if key in query_data:
                value = query_data[key]
                if isinstance(value, list):
                    return [t.get("name", str(t)) if isinstance(t, dict) else str(t) for t in value]
                elif isinstance(value, str):
                    return [value]
        return []

    def _extract_hijacker_tool(
        self,
        query_data: Dict[str, Any],
        tool_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract hijacker tool if present."""
        for key in ["hijacker_tool", "malicious_tool", "adversarial_tool"]:
            if key in query_data and query_data[key]:
                value = query_data[key]
                if isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    return {"name": value}

        # Check tool candidates for hijacker flag
        for tool in tool_candidates:
            if tool.get("is_hijacker", False) or tool.get("is_malicious", False):
                return tool

        return None

    def _is_attack(
        self,
        selected_tools: List[str],
        hijacker_tool: Optional[Dict[str, Any]],
    ) -> bool:
        """Determine if attack succeeded (hijacker was selected)."""
        if not hijacker_tool:
            return False

        hijacker_name = hijacker_tool.get("name", "")
        if not hijacker_name:
            return False

        selected_lower = [s.lower() for s in selected_tools]
        return hijacker_name.lower() in selected_lower

    def _selection_correct(
        self,
        selected_tools: List[str],
        intended_tools: List[str],
    ) -> bool:
        """Check if selection matches intended tools."""
        selected_set = set(s.lower() for s in selected_tools)
        intended_set = set(i.lower() for i in intended_tools)
        return selected_set == intended_set

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool based on name."""
        name_lower = tool_name.lower()

        for keyword, category in TOOL_CATEGORIES.items():
            if keyword in name_lower:
                return category

        return "other"

    def _infer_hijacking_type(self, hijacker_tool: Dict[str, Any]) -> str:
        """Infer hijacking attack type."""
        desc = hijacker_tool.get("description", "").lower()
        name = hijacker_tool.get("name", "").lower()

        if "ignore" in desc or "forget" in desc:
            return "description_injection"
        if "urgent" in desc or "important" in desc:
            return "urgency_injection"
        if "official" in desc or "verified" in desc:
            return "authority_claim"

        return "functionality_overlap"

    def get_hijacking_types(self) -> Dict[str, str]:
        """Get dictionary of hijacking types."""
        return HIJACKING_TYPES.copy()

    def get_tool_categories(self) -> Dict[str, str]:
        """Get dictionary of tool categories."""
        return TOOL_CATEGORIES.copy()

    def get_statistics(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Get statistics about extracted trajectories."""
        stats = {
            "total": len(trajectories),
            "attacks": sum(1 for t in trajectories if t.is_attack),
            "benign": sum(1 for t in trajectories if not t.is_attack),
            "single_tool": 0,
            "multi_tool": 0,
            "by_hijacking_type": {},
            "avg_tools_available": 0,
            "correct_selections": 0,
        }

        total_tools = 0

        for t in trajectories:
            selection_type = t.metadata.get("selection_type", "single")
            if selection_type == "multi":
                stats["multi_tool"] += 1
            else:
                stats["single_tool"] += 1

            if t.is_attack:
                hijacking_type = t.metadata.get("hijacking_type", "unknown")
                stats["by_hijacking_type"][hijacking_type] = stats["by_hijacking_type"].get(hijacking_type, 0) + 1

            total_tools += t.metadata.get("num_tool_candidates", 0)

            if not t.is_attack and t.success:
                stats["correct_selections"] += 1

        if trajectories:
            stats["avg_tools_available"] = total_tools / len(trajectories)

        return stats
