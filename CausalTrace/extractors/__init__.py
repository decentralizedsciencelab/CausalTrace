"""Extractors package for CausalTrace.

Main Extractors:
    - WASPExtractor: Basic WASP trajectory extractor
    - SafeArenaExtractor: SafeArena benchmark extractor
    - PajamasExtractor: pajaMAS multi-agent system extractor
    - AgentBankExtractor: AgentBank dataset extractor (50k+ trajectories)
    - AgentDojoExtractor: AgentDojo benchmark extractor (ETH Zurich, NeurIPS 2024)
    - ToolEmuExtractor: ToolEmu safety benchmark extractor (ICLR 2024 Spotlight)
    - MARBLEExtractor: MARBLE/MultiAgentBench extractor (multi-agent coordination, ACL 2025)
    - AgentBenchExtractor: AgentBench extractor (8 environments, ICLR 2024)
    - SWEAgentExtractor: SWE-agent trajectory extractor (80K software engineering trajectories)

New Benchmarks (January 2026):
    - B3Extractor: Lakera b3 benchmark (194K+ human attacks, 3 security levels)
    - ASBExtractor: Agent Security Bench (ICLR 2025, 400+ attack tools, 4 attack types)
    - LLMPIEvalExtractor: Amazon LLM-PIEval (NeurIPS 2024, 150 APIs, multi-turn injection)
    - MetaToolExtractor: MetaTool benchmark (ICLR 2024, 21K queries, tool hijacking)

Additional Extractors (import directly to avoid circular imports):
    - WASPBenchmarkExtractor: Comprehensive WASP benchmark log extractor
      Usage: from causaltrace.extractors.wasp_benchmark_extractor import WASPBenchmarkExtractor
"""

from causaltrace.extractors.base import BaseExtractor
from causaltrace.extractors.wasp import WASPExtractor
from causaltrace.extractors.safearena import SafeArenaExtractor
from causaltrace.extractors.pajamas import PajamasExtractor, TrajectoryLogger
from causaltrace.extractors.agentbank import AgentBankExtractor
from causaltrace.extractors.agentdojo import AgentDojoExtractor
from causaltrace.extractors.toolemu import ToolEmuExtractor
from causaltrace.extractors.marble import MARBLEExtractor
from causaltrace.extractors.agentbench import AgentBenchExtractor
from causaltrace.extractors.sweagent import SWEAgentExtractor
from causaltrace.extractors.causalbench import CausalBenchExtractor

# New benchmarks (January 2026)
from causaltrace.extractors.b3 import B3Extractor
from causaltrace.extractors.asb import ASBExtractor
from causaltrace.extractors.pieval import LLMPIEvalExtractor
from causaltrace.extractors.metatool import MetaToolExtractor

# Note: WASPBenchmarkExtractor must be imported directly from its module
# to avoid circular import issues:
#   from causaltrace.extractors.wasp_benchmark_extractor import WASPBenchmarkExtractor

__all__ = [
    "BaseExtractor",
    "WASPExtractor",
    "SafeArenaExtractor",
    "PajamasExtractor",
    "AgentBankExtractor",
    "AgentDojoExtractor",
    "ToolEmuExtractor",
    "MARBLEExtractor",
    "AgentBenchExtractor",
    "SWEAgentExtractor",
    "CausalBenchExtractor",
    "TrajectoryLogger",
    # New benchmarks (January 2026)
    "B3Extractor",
    "ASBExtractor",
    "LLMPIEvalExtractor",
    "MetaToolExtractor",
    # WASPBenchmarkExtractor available via direct import
]
