"""
Feature extraction module for causal graphs.

Provides comprehensive feature extraction from causal graphs for attack detection:
- Chain features (depth, length, count)
- Domain features (cross-domain edges, transitions)
- State features (accumulation, unused state)
- Bottleneck features (critical nodes, centrality)
- Watermark features (coverage, propagation, tampering detection)
- Analysis tools (statistical comparison, visualization)
"""

# Import main classes from submodules
from causaltrace.features.feature_extractor import FeatureVector, FeatureExtractor
from causaltrace.features.normalizer import FeatureNormalizer, MinMaxNormalizer
from causaltrace.features.analysis import (
    compare_attack_vs_benign,
    plot_feature_distributions,
    get_top_discriminative_features,
    compute_feature_correlations,
    identify_redundant_features,
    generate_feature_importance_report,
    export_features_to_csv,
)

# Import individual feature modules for advanced usage
from causaltrace.features import chain_features
from causaltrace.features import domain_features
from causaltrace.features import state_features
from causaltrace.features import bottleneck_features
from causaltrace.features import watermark_features

__all__ = [
    # Main classes
    'FeatureVector',
    'FeatureExtractor',
    'FeatureNormalizer',
    'MinMaxNormalizer',

    # Analysis functions
    'compare_attack_vs_benign',
    'plot_feature_distributions',
    'get_top_discriminative_features',
    'compute_feature_correlations',
    'identify_redundant_features',
    'generate_feature_importance_report',
    'export_features_to_csv',

    # Feature modules
    'chain_features',
    'domain_features',
    'state_features',
    'bottleneck_features',
    'watermark_features',
]
