"""
Benchmark evaluation for WASP and SafeArena datasets.

Provides utilities to evaluate detectors on standardized benchmarks
and perform cross-validation.
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json
from dataclasses import dataclass
from sklearn.model_selection import KFold
import numpy as np

from .detector import BaseDetector, DetectionResult
from .metrics import EvaluationMetrics, compute_metrics
from ..graph import CausalGraph, GraphBuilder


@dataclass
class BenchmarkResult:
    """Results from benchmark evaluation."""
    benchmark_name: str
    metrics: EvaluationMetrics
    num_samples: int
    num_attacks: int
    num_benign: int
    detector_name: str
    predictions: List[DetectionResult]
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"""
Benchmark: {self.benchmark_name}
Detector: {self.detector_name}
Samples: {self.num_samples} ({self.num_attacks} attacks, {self.num_benign} benign)

{self.metrics}
        """.strip()


class BenchmarkEvaluator:
    """
    Evaluate detector on WASP and SafeArena benchmarks.

    This class handles:
    1. Loading trajectories from benchmark result directories
    2. Building causal graphs
    3. Running detection
    4. Computing evaluation metrics
    """

    def __init__(self, detector: BaseDetector, graph_builder: Optional[GraphBuilder] = None):
        """
        Initialize evaluator.

        Args:
            detector: Trained detector to evaluate
            graph_builder: Optional graph builder (creates default if not provided)
        """
        self.detector = detector
        self.graph_builder = graph_builder or GraphBuilder()

    def evaluate_wasp(
        self,
        wasp_results_dir: str,
        max_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Evaluate on WASP benchmark.

        WASP Structure:
        - Attack scenarios in experiment_config.raw.json
        - Trajectory logs in results directory
        - Each trajectory is labeled as attack or benign

        Args:
            wasp_results_dir: Path to WASP results directory
            max_samples: Optional limit on number of samples to evaluate

        Returns:
            BenchmarkResult with evaluation metrics
        """
        print(f"Evaluating on WASP benchmark: {wasp_results_dir}")

        # Load WASP trajectories
        trajectories, labels = self._load_wasp_trajectories(wasp_results_dir, max_samples)

        # Evaluate
        result = self._evaluate_trajectories(
            trajectories=trajectories,
            labels=labels,
            benchmark_name="WASP",
            metadata={'wasp_results_dir': wasp_results_dir}
        )

        print(f"WASP Evaluation Complete: F1={result.metrics.f1_score:.2%}")
        return result

    def evaluate_safearena(
        self,
        safearena_results_dir: str,
        task_type: str = "both",
        max_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Evaluate on SafeArena benchmark.

        SafeArena Structure:
        - Safe tasks (benign)
        - Harmful tasks (attacks)
        - Trajectory logs in results directory

        Args:
            safearena_results_dir: Path to SafeArena results directory
            task_type: "safe", "harm", or "both"
            max_samples: Optional limit on number of samples to evaluate

        Returns:
            BenchmarkResult with evaluation metrics
        """
        print(f"Evaluating on SafeArena benchmark ({task_type}): {safearena_results_dir}")

        # Load SafeArena trajectories
        trajectories, labels = self._load_safearena_trajectories(
            safearena_results_dir,
            task_type,
            max_samples
        )

        # Evaluate
        result = self._evaluate_trajectories(
            trajectories=trajectories,
            labels=labels,
            benchmark_name=f"SafeArena ({task_type})",
            metadata={
                'safearena_results_dir': safearena_results_dir,
                'task_type': task_type
            }
        )

        print(f"SafeArena Evaluation Complete: F1={result.metrics.f1_score:.2%}")
        return result

    def evaluate_combined(
        self,
        wasp_dir: str,
        safearena_dir: str,
        max_samples_per_benchmark: Optional[int] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Evaluate on both WASP and SafeArena benchmarks.

        Args:
            wasp_dir: Path to WASP results
            safearena_dir: Path to SafeArena results
            max_samples_per_benchmark: Optional limit per benchmark

        Returns:
            Dictionary with results for each benchmark
        """
        print("Evaluating on combined benchmarks...")

        results = {}

        # Evaluate WASP
        try:
            results['wasp'] = self.evaluate_wasp(wasp_dir, max_samples_per_benchmark)
        except Exception as e:
            print(f"Warning: WASP evaluation failed: {e}")
            results['wasp'] = None

        # Evaluate SafeArena
        try:
            results['safearena'] = self.evaluate_safearena(
                safearena_dir,
                task_type="both",
                max_samples=max_samples_per_benchmark
            )
        except Exception as e:
            print(f"Warning: SafeArena evaluation failed: {e}")
            results['safearena'] = None

        # Combine results if both succeeded
        if results['wasp'] and results['safearena']:
            combined_predictions = results['wasp'].predictions + results['safearena'].predictions
            combined_labels = (
                [True] * results['wasp'].num_attacks + [False] * results['wasp'].num_benign +
                [True] * results['safearena'].num_attacks + [False] * results['safearena'].num_benign
            )

            y_pred = [p.is_attack for p in combined_predictions]
            y_proba = [p.confidence if p.is_attack else (1.0 - p.confidence)
                      for p in combined_predictions]

            combined_metrics = compute_metrics(combined_labels, y_pred, y_proba)

            results['combined'] = BenchmarkResult(
                benchmark_name="Combined (WASP + SafeArena)",
                metrics=combined_metrics,
                num_samples=len(combined_predictions),
                num_attacks=combined_metrics.total_attacks,
                num_benign=combined_metrics.total_benign,
                detector_name=self.detector.__class__.__name__,
                predictions=combined_predictions,
                metadata={
                    'wasp_dir': wasp_dir,
                    'safearena_dir': safearena_dir
                }
            )

        return results

    def cross_validate(
        self,
        graphs: List[CausalGraph],
        labels: List[bool],
        k: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ) -> List[EvaluationMetrics]:
        """
        Perform k-fold cross-validation.

        Args:
            graphs: List of CausalGraph objects
            labels: List of labels
            k: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed

        Returns:
            List of EvaluationMetrics for each fold
        """
        print(f"Performing {k}-fold cross-validation on {len(graphs)} samples...")

        kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(graphs)):
            print(f"  Fold {fold_idx + 1}/{k}...")

            # Split data
            train_graphs = [graphs[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_graphs = [graphs[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]

            # Train detector
            # Note: Create a fresh detector instance for each fold
            detector_class = type(self.detector)
            fold_detector = detector_class()
            fold_detector.fit(train_graphs, train_labels)

            # Predict
            predictions = fold_detector.predict_batch(test_graphs)
            y_pred = [p.is_attack for p in predictions]
            y_proba = [p.confidence if p.is_attack else (1.0 - p.confidence)
                      for p in predictions]

            # Compute metrics
            metrics = compute_metrics(
                test_labels,
                y_pred,
                y_proba,
                metadata={'fold': fold_idx + 1}
            )
            fold_metrics.append(metrics)

            print(f"    Fold {fold_idx + 1} F1: {metrics.f1_score:.2%}")

        # Print summary
        avg_f1 = np.mean([m.f1_score for m in fold_metrics])
        std_f1 = np.std([m.f1_score for m in fold_metrics])
        print(f"Cross-validation complete: F1 = {avg_f1:.2%} ± {std_f1:.2%}")

        return fold_metrics

    def _evaluate_trajectories(
        self,
        trajectories: List[Any],
        labels: List[bool],
        benchmark_name: str,
        metadata: Dict[str, Any]
    ) -> BenchmarkResult:
        """
        Internal method to evaluate on a set of trajectories.

        Args:
            trajectories: List of trajectory objects
            labels: List of labels (True = attack, False = benign)
            benchmark_name: Name of the benchmark
            metadata: Additional metadata

        Returns:
            BenchmarkResult
        """
        print(f"  Building causal graphs for {len(trajectories)} trajectories...")
        graphs = [self.graph_builder.build(t) for t in trajectories]

        print(f"  Running detection...")
        predictions = self.detector.predict_batch(graphs)

        print(f"  Computing metrics...")
        y_pred = [p.is_attack for p in predictions]
        y_proba = [p.confidence if p.is_attack else (1.0 - p.confidence)
                  for p in predictions]

        metrics = compute_metrics(labels, y_pred, y_proba, metadata={'benchmark': benchmark_name})

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            metrics=metrics,
            num_samples=len(trajectories),
            num_attacks=sum(labels),
            num_benign=len(labels) - sum(labels),
            detector_name=self.detector.__class__.__name__,
            predictions=predictions,
            metadata=metadata
        )

    def _load_wasp_trajectories(
        self,
        wasp_dir: str,
        max_samples: Optional[int] = None
    ) -> tuple[List[Any], List[bool]]:
        """
        Load WASP trajectories.

        the full trajectory loading logic.

        Args:
            wasp_dir: WASP results directory
            max_samples: Maximum samples to load

        Returns:
            Tuple of (trajectories, labels)
        """
        trajectories = []
        labels = []

        return trajectories, labels

    def _load_safearena_trajectories(
        self,
        safearena_dir: str,
        task_type: str,
        max_samples: Optional[int] = None
    ) -> tuple[List[Any], List[bool]]:
        """
        Load SafeArena trajectories.

        the full trajectory loading logic.

        Args:
            safearena_dir: SafeArena results directory
            task_type: "safe", "harm", or "both"
            max_samples: Maximum samples to load

        Returns:
            Tuple of (trajectories, labels)
        """
        trajectories = []
        labels = []

        return trajectories, labels


def evaluate_multiple_detectors(
    detectors: Dict[str, BaseDetector],
    graphs: List[CausalGraph],
    labels: List[bool]
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate multiple detectors on the same dataset.

    Args:
        detectors: Dictionary mapping detector names to detector instances
        graphs: List of CausalGraph objects
        labels: List of labels

    Returns:
        Dictionary mapping detector names to EvaluationMetrics
    """
    results = {}

    for name, detector in detectors.items():
        print(f"Evaluating {name}...")

        predictions = detector.predict_batch(graphs)
        y_pred = [p.is_attack for p in predictions]
        y_proba = [p.confidence if p.is_attack else (1.0 - p.confidence)
                  for p in predictions]

        metrics = compute_metrics(labels, y_pred, y_proba, metadata={'detector': name})
        results[name] = metrics

        print(f"  {name} F1: {metrics.f1_score:.2%}")

    return results


__all__ = ['BenchmarkEvaluator', 'BenchmarkResult', 'evaluate_multiple_detectors']
