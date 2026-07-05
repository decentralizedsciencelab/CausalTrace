"""
Core modules for CausalBench Generator.

All API operations execute REAL calls - nothing is simulated.
Test resources are created, used, and cleaned up automatically.
"""

from .trajectory_logger import (
    TrajectoryLogger,
    TrajectoryEvent,
    CausalEdge,
    CausalGraph,
    EventType,
    TrustLevel
)
from .injection_engine import (
    InjectionEngine,
    InjectionPayload,
    AttackType,
    Sophistication
)
from .scenario_runner import (
    ScenarioRunner,
    ScenarioConfig,
    ExecutionResult
)
from .validator import (
    TrajectoryValidator,
    ValidationResult,
    ValidationIssue,
    ValidationLevel,
    validate_dataset
)
from .test_sandbox import (
    TestSandbox,
    TestResource,
    GitHubSandbox,
    SlackSandbox,
    DropboxSandbox,
    StripeSandbox,
    create_sandbox
)
from .exfil_collector import (
    ExfilCollector,
    ExfilEvent,
    get_collector,
    start_collector_server
)

__all__ = [
    # Trajectory
    "TrajectoryLogger",
    "TrajectoryEvent",
    "CausalEdge",
    "CausalGraph",
    "EventType",
    "TrustLevel",
    # Injection
    "InjectionEngine",
    "InjectionPayload",
    "AttackType",
    "Sophistication",
    # Runner
    "ScenarioRunner",
    "ScenarioConfig",
    "ExecutionResult",
    # Validator
    "TrajectoryValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "validate_dataset",
    # Test Sandbox
    "TestSandbox",
    "TestResource",
    "GitHubSandbox",
    "SlackSandbox",
    "DropboxSandbox",
    "StripeSandbox",
    "create_sandbox",
    # Exfiltration Collector
    "ExfilCollector",
    "ExfilEvent",
    "get_collector",
    "start_collector_server",
]
