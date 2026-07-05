"""
Evaluation metrics for attack detection.

Provides comprehensive metrics including TPR, FPR, precision, recall, F1, ROC-AUC,
and confusion matrix analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for attack detection.

    Attributes:
        true_positive_rate: Recall, detection rate (TP / (TP + FN))
        false_positive_rate: FP / (FP + TN)
        precision: TP / (TP + FP)
        f1_score: Harmonic mean of precision and recall
        accuracy: (TP + TN) / (TP + TN + FP + FN)
        true_positives: Number of correctly detected attacks
        false_positives: Number of benign samples incorrectly flagged
        true_negatives: Number of correctly classified benign samples
        false_negatives: Number of missed attacks
        roc_auc: Area under ROC curve (if probabilities available)
    """

    # Core metrics
    true_positive_rate: float  # Recall, detection rate
    false_positive_rate: float
    precision: float
    f1_score: float
    accuracy: float

    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Additional
    roc_auc: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def total_samples(self) -> int:
        """Total number of samples."""
        return self.true_positives + self.false_positives + self.true_negatives + self.false_negatives

    @property
    def total_attacks(self) -> int:
        """Total number of actual attacks."""
        return self.true_positives + self.false_negatives

    @property
    def total_benign(self) -> int:
        """Total number of benign samples."""
        return self.false_positives + self.true_negatives

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            "=" * 60,
            "Detection Performance Metrics",
            "=" * 60,
            "",
            "Classification Results:",
            f"  True Positives:  {self.true_positives:4d}  (attacks correctly detected)",
            f"  False Positives: {self.false_positives:4d}  (benign incorrectly flagged)",
            f"  True Negatives:  {self.true_negatives:4d}  (benign correctly classified)",
            f"  False Negatives: {self.false_negatives:4d}  (attacks missed)",
            "",
            "Performance Metrics:",
            f"  True Positive Rate (Detection Rate): {self.true_positive_rate:6.2%}  ({self.true_positives} / {self.total_attacks})",
            f"  False Positive Rate:                 {self.false_positive_rate:6.2%}  ({self.false_positives} / {self.total_benign})",
            f"  Precision:                           {self.precision:6.2%}",
            f"  F1 Score:                            {self.f1_score:6.2%}",
            f"  Accuracy:                            {self.accuracy:6.2%}",
        ]

        if self.roc_auc is not None:
            lines.append(f"  ROC AUC:                             {self.roc_auc:6.4f}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'true_positive_rate': self.true_positive_rate,
            'false_positive_rate': self.false_positive_rate,
            'precision': self.precision,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'confusion_matrix': {
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives,
            },
        }

        if self.roc_auc is not None:
            result['roc_auc'] = self.roc_auc

        if self.metadata:
            result['metadata'] = self.metadata

        return result

    def summary(self) -> str:
        """One-line summary of key metrics."""
        return (
            f"TPR={self.true_positive_rate:.2%}, "
            f"FPR={self.false_positive_rate:.2%}, "
            f"F1={self.f1_score:.2%}"
        )


def compute_metrics(
    y_true: List[bool],
    y_pred: List[bool],
    y_proba: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (True = attack, False = benign)
        y_pred: Predicted labels
        y_proba: Optional predicted probabilities for attack class
        metadata: Optional metadata to attach to metrics

    Returns:
        EvaluationMetrics instance

    Raises:
        ValueError: If inputs have mismatched lengths
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions")

    if y_proba is not None and len(y_proba) != len(y_true):
        raise ValueError(f"Length mismatch: {len(y_true)} labels vs {len(y_proba)} probabilities")

    # Convert to numpy arrays
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()

    # Compute rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Compute precision, recall, F1
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0.0)
    recall = tpr  # Same as TPR
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0.0)
    accuracy = accuracy_score(y_true_arr, y_pred_arr)

    # Compute ROC AUC if probabilities provided
    roc_auc = None
    if y_proba is not None:
        y_proba_arr = np.array(y_proba)
        try:
            roc_auc = roc_auc_score(y_true_arr, y_proba_arr)
        except ValueError:
            # ROC AUC undefined if only one class present
            roc_auc = None

    return EvaluationMetrics(
        true_positive_rate=float(tpr),
        false_positive_rate=float(fpr),
        precision=float(precision),
        f1_score=float(f1),
        accuracy=float(accuracy),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        metadata=metadata or {}
    )


def compute_roc_curve(
    y_true: List[bool],
    y_proba: List[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for attack class

    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_proba_arr = np.array(y_proba)

    return roc_curve(y_true_arr, y_proba_arr)


def find_optimal_threshold(
    y_true: List[bool],
    y_proba: List[float],
    metric: str = "f1"
) -> tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ("f1", "balanced_accuracy", or "youden")

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_proba_arr = np.array(y_proba)

    # Try different thresholds
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (y_proba_arr >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true_arr, y_pred, zero_division=0.0)
        elif metric == "balanced_accuracy":
            score = (recall_score(y_true_arr, y_pred, zero_division=0.0) +
                    precision_score(y_true_arr, y_pred, zero_division=0.0)) / 2
        elif metric == "youden":
            # Youden's J statistic = TPR - FPR
            tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)


def compare_metrics(metrics_list: List[EvaluationMetrics], names: List[str]) -> str:
    """
    Compare multiple evaluation metrics side-by-side.

    Args:
        metrics_list: List of EvaluationMetrics instances
        names: Names for each metric set

    Returns:
        Formatted comparison table
    """
    if len(metrics_list) != len(names):
        raise ValueError("Number of metrics must match number of names")

    lines = [
        "=" * 80,
        "Metrics Comparison",
        "=" * 80,
        ""
    ]

    # Header
    header = f"{'Metric':<30}"
    for name in names:
        header += f"{name:>15}"
    lines.append(header)
    lines.append("-" * 80)

    # Rows
    metrics_to_show = [
        ('True Positive Rate', lambda m: f"{m.true_positive_rate:.2%}"),
        ('False Positive Rate', lambda m: f"{m.false_positive_rate:.2%}"),
        ('Precision', lambda m: f"{m.precision:.2%}"),
        ('F1 Score', lambda m: f"{m.f1_score:.2%}"),
        ('Accuracy', lambda m: f"{m.accuracy:.2%}"),
        ('ROC AUC', lambda m: f"{m.roc_auc:.4f}" if m.roc_auc else "N/A"),
    ]

    for metric_name, formatter in metrics_to_show:
        row = f"{metric_name:<30}"
        for metric in metrics_list:
            row += f"{formatter(metric):>15}"
        lines.append(row)

    lines.append("=" * 80)

    return "\n".join(lines)


__all__ = [
    'EvaluationMetrics',
    'compute_metrics',
    'compute_roc_curve',
    'find_optimal_threshold',
    'compare_metrics',
]
