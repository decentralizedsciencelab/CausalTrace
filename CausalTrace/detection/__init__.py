"""
CausalTrace Detection Module

This module provides attack detection capabilities using causal graph features.
It includes threshold-based and ML-based detectors, evaluation metrics, and
benchmark evaluation tools for WASP and SafeArena.

SOTA Baselines (Tier 2):
- PromptGuard: Meta's transformer-based detector
- NeMoGuardrails: NVIDIA's production guardrails
- LLMGuard: Protect AI's multi-layer defense
- Rebuff: Protect AI's prompt injection firewall

GNN-based detectors (optional, requires torch_geometric):
- GNNDetector: GCN, GAT, GraphSAGE, GIN architectures

Domain Adaptation:
- Feature normalization (Z-score, MinMax, Robust)
- Distribution alignment (CORAL, MMD)
- Data augmentation (Edge dropout, Node masking, Mixup)
- Self-training with pseudo-labels
"""

from .detector import BaseDetector, DetectionResult
from .threshold_detector import ThresholdDetector
from .ml_detector import MLDetector
from .llm_detector import LLMJudgeDetector
from .causal_detector import CausalDetector, CausalDetectionResult
from .mechanism_detector import MechanismIntegrityDetector, MechanismDetectionResult, EnsembleDetector
from .probabilistic_detector import ProbabilisticCausalDetector, ProbabilisticDetectionResult
from .watermark_detector import WatermarkDetector, WatermarkConfig
from .enhanced_watermark_detector import EnhancedWatermarkDetector, EnhancedWatermarkConfig
from .semantic_detector import SemanticInjectionDetector
from .unified_detector import UnifiedAttackDetector, UnifiedDetectionResult
from .adaptive_detector import AdaptiveUnifiedDetector, AdaptiveDetectionResult, AttackType
from .transfer_learning import (
    TransferLearningPipeline,
    TransferLearningConfig,
    TransferredDetector,
    TrainingExample,
)
from .metrics import EvaluationMetrics, compute_metrics
from .evaluator import BenchmarkEvaluator
from .pipeline import AttackDetectionPipeline

# GNN components (optional)
try:
    from .gnn_detector import GNNDetector, GNNConfig, evaluate_gnn_models
    from .gnn_data import CausalGraphDataset, create_data_loaders
    from .gnn_trainer import GNNTrainer, TrainingResult
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    GNNDetector = None
    GNNConfig = None
    GNNTrainer = None

# SOTA baselines
from .sota_baselines import (
    SOTABaseDetector,
    SOTADetectionResult,
    PromptGuardDetector,
    NeMoGuardrailsDetector,
    LLMGuardDetector,
    RebuffDetector,
    get_sota_detector,
    compare_sota_methods,
)

# Custom baselines
from .baselines import (
    KeywordFilterDetector,
    PromptHardeningDetector,
    ActionWhitelistDetector,
    StepAnomalyDetector,
    CausalTraceDetector as BaselineCausalTraceDetector,
    get_detector as get_baseline_detector,
)

# Domain adaptation
from .domain_adaptation import (
    # Normalizers
    ZScoreNormalizer,
    MinMaxNormalizer,
    RobustNormalizer,
    DomainNormalizer,
    get_normalizer,
    # Distribution alignment
    CORALAdapter,
    compute_mmd,
    evaluate_domain_gap,
    # Augmentation
    GraphAugmenter,
    AugmentationConfig,
    # Self-training
    SelfTrainer,
    # Pipeline
    DomainAdaptationPipeline,
    DomainAdaptationConfig,
)

# Optional: MMD Aligner (requires PyTorch)
try:
    from .domain_adaptation import MMDAligner
    MMD_ALIGNER_AVAILABLE = True
except ImportError:
    MMD_ALIGNER_AVAILABLE = False
    MMDAligner = None

__all__ = [
    # Core detectors
    'BaseDetector',
    'DetectionResult',
    'ThresholdDetector',
    'MLDetector',
    'LLMJudgeDetector',
    'CausalDetector',
    'CausalDetectionResult',
    'MechanismIntegrityDetector',
    'MechanismDetectionResult',
    'EnsembleDetector',
    'ProbabilisticCausalDetector',
    'ProbabilisticDetectionResult',
    'WatermarkDetector',
    'WatermarkConfig',
    'EnhancedWatermarkDetector',
    'EnhancedWatermarkConfig',
    # Evaluation
    'EvaluationMetrics',
    'compute_metrics',
    'BenchmarkEvaluator',
    'AttackDetectionPipeline',
    # SOTA baselines
    'SOTABaseDetector',
    'SOTADetectionResult',
    'PromptGuardDetector',
    'NeMoGuardrailsDetector',
    'LLMGuardDetector',
    'RebuffDetector',
    'get_sota_detector',
    'compare_sota_methods',
    # Custom baselines
    'KeywordFilterDetector',
    'PromptHardeningDetector',
    'ActionWhitelistDetector',
    'StepAnomalyDetector',
    'BaselineCausalTraceDetector',
    'get_baseline_detector',
    # GNN detectors (optional)
    'GNN_AVAILABLE',
    'GNNDetector',
    'GNNConfig',
    'GNNTrainer',
    'TrainingResult',
    'CausalGraphDataset',
    'create_data_loaders',
    'evaluate_gnn_models',
    # Domain adaptation
    'ZScoreNormalizer',
    'MinMaxNormalizer',
    'RobustNormalizer',
    'DomainNormalizer',
    'get_normalizer',
    'CORALAdapter',
    'MMDAligner',
    'compute_mmd',
    'evaluate_domain_gap',
    'GraphAugmenter',
    'AugmentationConfig',
    'SelfTrainer',
    'DomainAdaptationPipeline',
    'DomainAdaptationConfig',
    'MMD_ALIGNER_AVAILABLE',
    # Adaptive detector (LLM-first)
    'AdaptiveUnifiedDetector',
    'AdaptiveDetectionResult',
    'AttackType',
    # Transfer learning
    'TransferLearningPipeline',
    'TransferLearningConfig',
    'TransferredDetector',
    'TrainingExample',
]
