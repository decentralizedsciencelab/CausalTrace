"""
CausalTrace Adversarial Evaluation Framework.

This module provides tools for generating adversarial attack trajectories
and evaluating CausalTrace's robustness against evasion techniques.
"""

from causaltrace.evaluation.adversarial_generator import (
    AdversarialGenerator,
    EvasionTechnique,
    AdversarialPattern,
    generate_adversarial_dataset,
)

from causaltrace.evaluation.red_team_scenarios import (
    RedTeamScenario,
    CHeaTBasedAttack,
    CTFExploitChain,
    generate_red_team_scenarios,
)

__all__ = [
    "AdversarialGenerator",
    "EvasionTechnique",
    "AdversarialPattern",
    "generate_adversarial_dataset",
    "RedTeamScenario",
    "CHeaTBasedAttack",
    "CTFExploitChain",
    "generate_red_team_scenarios",
]
