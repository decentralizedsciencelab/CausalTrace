"""
Feature analysis and comparison tools.

Provides utilities for comparing feature distributions between
attack and benign trajectories.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from causaltrace.features.feature_extractor import FeatureVector


def compare_attack_vs_benign(
    attack_features: List[FeatureVector],
    benign_features: List[FeatureVector]
) -> pd.DataFrame:
    """
    Compare feature distributions between attack and benign trajectories.

    Uses t-test to determine statistical significance of differences.

    Args:
        attack_features: List of FeatureVectors from attack trajectories
        benign_features: List of FeatureVectors from benign trajectories

    Returns:
        DataFrame with comparison statistics for each feature
    """
    from scipy import stats

    if not attack_features or not benign_features:
        raise ValueError("Both attack and benign features must be non-empty")

    # Convert to numpy arrays
    attack_array = np.array([f.to_numpy() for f in attack_features])
    benign_array = np.array([f.to_numpy() for f in benign_features])

    feature_names = FeatureVector.feature_names()

    results = []

    for i, feature_name in enumerate(feature_names):
        attack_vals = attack_array[:, i]
        benign_vals = benign_array[:, i]

        # Compute statistics
        attack_mean = np.mean(attack_vals)
        attack_std = np.std(attack_vals)
        benign_mean = np.mean(benign_vals)
        benign_std = np.std(benign_vals)

        # T-test for statistical significance
        t_stat, p_value = stats.ttest_ind(attack_vals, benign_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((attack_std**2 + benign_std**2) / 2)
        cohens_d = (attack_mean - benign_mean) / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': feature_name,
            'attack_mean': attack_mean,
            'attack_std': attack_std,
            'benign_mean': benign_mean,
            'benign_std': benign_std,
            'difference': attack_mean - benign_mean,
            'relative_difference': ((attack_mean - benign_mean) / benign_mean * 100) if benign_mean != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': _interpret_cohens_d(cohens_d),
        })

    df = pd.DataFrame(results)

    # Sort by absolute effect size
    df = df.sort_values('cohens_d', key=abs, ascending=False)

    return df


def plot_feature_distributions(
    attack_features: List[FeatureVector],
    benign_features: List[FeatureVector],
    output_dir: str,
    features_to_plot: List[str] = None
) -> None:
    """
    Generate comparison plots for all features.

    Creates histograms showing attack vs benign distributions.

    Args:
        attack_features: List of FeatureVectors from attack trajectories
        benign_features: List of FeatureVectors from benign trajectories
        output_dir: Directory to save plots
        features_to_plot: Optional list of specific features to plot (default: all)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn required for plotting. Install with: pip install matplotlib seaborn")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays
    attack_array = np.array([f.to_numpy() for f in attack_features])
    benign_array = np.array([f.to_numpy() for f in benign_features])

    feature_names = FeatureVector.feature_names()

    if features_to_plot:
        # Filter to requested features
        indices = [i for i, name in enumerate(feature_names) if name in features_to_plot]
        feature_names = [feature_names[i] for i in indices]
        attack_array = attack_array[:, indices]
        benign_array = benign_array[:, indices]

    # Set style
    sns.set_style("whitegrid")

    # Create subplot grid
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature_name in enumerate(feature_names):
        ax = axes[i]

        attack_vals = attack_array[:, i]
        benign_vals = benign_array[:, i]

        # Plot histograms
        ax.hist(benign_vals, bins=30, alpha=0.5, label='Benign', color='blue', density=True)
        ax.hist(attack_vals, bins=30, alpha=0.5, label='Attack', color='red', density=True)

        ax.set_xlabel(feature_name.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_name}')
        ax.legend()

    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved feature distribution plots to {output_path / 'feature_distributions.png'}")


def get_top_discriminative_features(
    attack_features: List[FeatureVector],
    benign_features: List[FeatureVector],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Get top-k features that best discriminate between attacks and benign.

    Uses absolute Cohen's d (effect size) as the ranking metric.

    Args:
        attack_features: List of FeatureVectors from attack trajectories
        benign_features: List of FeatureVectors from benign trajectories
        top_k: Number of top features to return

    Returns:
        List of (feature_name, cohens_d) tuples, sorted by absolute effect size
    """
    comparison = compare_attack_vs_benign(attack_features, benign_features)

    top_features = comparison.nlargest(top_k, 'cohens_d', keep='all')[['feature', 'cohens_d']]

    return list(zip(top_features['feature'], top_features['cohens_d']))


def compute_feature_correlations(features: List[FeatureVector]) -> pd.DataFrame:
    """
    Compute correlation matrix between features.

    Useful for identifying redundant features.

    Args:
        features: List of FeatureVectors

    Returns:
        DataFrame with correlation matrix
    """
    if not features:
        raise ValueError("Features list cannot be empty")

    # Convert to numpy array
    X = np.array([f.to_numpy() for f in features])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)

    # Create DataFrame
    feature_names = FeatureVector.feature_names()
    df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)

    return df


def identify_redundant_features(
    features: List[FeatureVector],
    threshold: float = 0.9
) -> List[Tuple[str, str, float]]:
    """
    Identify pairs of features with high correlation (potentially redundant).

    Args:
        features: List of FeatureVectors
        threshold: Correlation threshold (default: 0.9)

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    corr_matrix = compute_feature_correlations(features)

    redundant = []

    feature_names = FeatureVector.feature_names()

    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i < j:  # Only consider upper triangle
                corr = corr_matrix.loc[feat1, feat2]
                if abs(corr) >= threshold:
                    redundant.append((feat1, feat2, corr))

    # Sort by absolute correlation
    redundant.sort(key=lambda x: abs(x[2]), reverse=True)

    return redundant


def generate_feature_importance_report(
    attack_features: List[FeatureVector],
    benign_features: List[FeatureVector],
    output_file: str = None
) -> str:
    """
    Generate a comprehensive feature importance report.

    Args:
        attack_features: List of FeatureVectors from attack trajectories
        benign_features: List of FeatureVectors from benign trajectories
        output_file: Optional file path to save the report

    Returns:
        Report as a string
    """
    comparison = compare_attack_vs_benign(attack_features, benign_features)

    report_lines = [
        "=" * 80,
        "FEATURE IMPORTANCE REPORT",
        "=" * 80,
        "",
        f"Dataset Summary:",
        f"  Attack trajectories: {len(attack_features)}",
        f"  Benign trajectories: {len(benign_features)}",
        f"  Total features: {len(FeatureVector.feature_names())}",
        "",
        "=" * 80,
        "TOP 10 DISCRIMINATIVE FEATURES",
        "=" * 80,
        "",
    ]

    # Top 10 features by effect size
    top_10 = comparison.head(10)

    for _, row in top_10.iterrows():
        report_lines.extend([
            f"{row['feature']}:",
            f"  Attack:  {row['attack_mean']:.3f} ± {row['attack_std']:.3f}",
            f"  Benign:  {row['benign_mean']:.3f} ± {row['benign_std']:.3f}",
            f"  Difference: {row['difference']:.3f} ({row['relative_difference']:.1f}%)",
            f"  Effect size: {row['cohens_d']:.3f} ({row['effect_size']})",
            f"  P-value: {row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''}",
            "",
        ])

    report_lines.extend([
        "=" * 80,
        "STATISTICAL SUMMARY",
        "=" * 80,
        "",
        f"Features with significant difference (p < 0.05): {sum(comparison['significant'])} / {len(comparison)}",
        f"Features with large effect size (|d| > 0.8): {sum(abs(comparison['cohens_d']) > 0.8)}",
        f"Features with medium effect size (0.5 < |d| < 0.8): {sum((abs(comparison['cohens_d']) > 0.5) & (abs(comparison['cohens_d']) <= 0.8))}",
        f"Features with small effect size (0.2 < |d| < 0.5): {sum((abs(comparison['cohens_d']) > 0.2) & (abs(comparison['cohens_d']) <= 0.5))}",
        "",
    ])

    report = "\n".join(report_lines)

    if output_file:
        Path(output_file).write_text(report)
        print(f"Report saved to {output_file}")

    return report


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def export_features_to_csv(
    features: List[FeatureVector],
    labels: List[bool],
    output_file: str
) -> None:
    """
    Export features to CSV format for external analysis.

    Args:
        features: List of FeatureVectors
        labels: List of boolean labels (True = attack, False = benign)
        output_file: Path to output CSV file
    """
    if len(features) != len(labels):
        raise ValueError("Number of features and labels must match")

    # Convert to DataFrame
    data = [f.to_dict() for f in features]
    df = pd.DataFrame(data)

    # Add label column
    df['is_attack'] = labels

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Exported {len(features)} feature vectors to {output_file}")
