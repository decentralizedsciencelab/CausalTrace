"""
GNN Training Pipeline for CausalTrace.

Provides a comprehensive training framework with:
- Train/val/test split handling
- Early stopping
- Learning rate scheduling
- Detailed metrics logging
- Model checkpointing
- Cross-validation
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
import time
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available")

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from ..graph import CausalGraph
from .gnn_detector import (
    GNNDetector,
    GNNConfig,
    causal_graph_to_pyg,
    get_node_feature_dim,
)
from .gnn_data import CausalGraphDataset


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    epoch: int
    train_loss: float
    train_accuracy: float
    train_f1: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_precision: float
    val_recall: float
    val_roc_auc: Optional[float] = None
    learning_rate: float = 0.0
    time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingResult:
    """Results from a complete training run."""

    model_type: str
    best_epoch: int
    best_val_loss: float
    best_val_f1: float
    test_metrics: Dict[str, float]
    training_history: List[TrainingMetrics]
    total_time_seconds: float
    config: Dict[str, Any]
    early_stopped: bool = False
    converged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "test_metrics": self.test_metrics,
            "training_history": [m.to_dict() for m in self.training_history],
            "total_time_seconds": self.total_time_seconds,
            "config": self.config,
            "early_stopped": self.early_stopped,
            "converged": self.converged,
        }

    def save(self, path: str) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class GNNTrainer:
    """
    Comprehensive GNN training pipeline.

    Handles:
    - Data splitting and loading
    - Training loop with early stopping
    - Learning rate scheduling
    - Metrics logging
    - Model checkpointing
    - Cross-validation
    """

    def __init__(
        self,
        config: Optional[GNNConfig] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            config: GNN configuration
            device: Device to train on ('cuda', 'cpu', or None for auto)
            checkpoint_dir: Directory for saving checkpoints
            verbose: Print training progress
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNNTrainer")

        self.config = config or GNNConfig()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.verbose = verbose

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.detector: Optional[GNNDetector] = None
        self.training_history: List[TrainingMetrics] = []
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

    def train(
        self,
        train_graphs: List[Union[CausalGraph, Dict[str, Any]]],
        train_labels: List[bool],
        val_graphs: Optional[List[Union[CausalGraph, Dict[str, Any]]]] = None,
        val_labels: Optional[List[bool]] = None,
        val_split: float = 0.15,
    ) -> TrainingResult:
        """
        Train a GNN detector.

        Args:
            train_graphs: Training graphs
            train_labels: Training labels
            val_graphs: Optional validation graphs (if None, splits from train)
            val_labels: Optional validation labels
            val_split: Validation split ratio if val_graphs not provided

        Returns:
            TrainingResult with metrics and history
        """
        start_time = time.time()

        # Create datasets
        if val_graphs is None:
            train_ds = CausalGraphDataset(train_graphs, train_labels)
            train_ds, val_ds, _ = train_ds.split(
                train_ratio=1 - val_split,
                val_ratio=val_split,
                test_ratio=0.0,
            )
        else:
            train_ds = CausalGraphDataset(train_graphs, train_labels)
            val_ds = CausalGraphDataset(val_graphs, val_labels)

        # Create loaders
        train_loader = train_ds.create_loader(
            batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = val_ds.create_loader(
            batch_size=self.config.batch_size, shuffle=False
        )

        # Initialize detector and model
        self.detector = GNNDetector(self.config)
        pyg_data = train_ds.get_pyg_list()
        in_channels = pyg_data[0].x.shape[1] if pyg_data else get_node_feature_dim()

        self.detector.model = self.detector._create_model(in_channels).to(self.device)
        self.detector._in_channels = in_channels

        # Class weights for imbalanced data
        num_attacks = sum(train_labels)
        num_benign = len(train_labels) - num_attacks
        if self.config.use_class_weights and num_attacks > 0 and num_benign > 0:
            weight = torch.tensor(
                [num_attacks / len(train_labels), num_benign / len(train_labels)]
            ).to(self.device)
        else:
            weight = None

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.detector.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = self._create_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss(weight=weight)

        # Training loop
        self.training_history = []
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        patience_counter = 0
        early_stopped = False

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.config.model_type.upper()} on {self.device}")
            print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
            print(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
            print(f"Class distribution: {num_attacks} attacks, {num_benign} benign")
            print(f"{'='*60}")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train epoch
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion, epoch
            )

            # Validate
            val_metrics = self._validate_epoch(val_loader, criterion)

            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if self.config.lr_scheduler == "plateau":
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                train_f1=train_metrics["f1"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_f1=val_metrics["f1"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_roc_auc=val_metrics.get("roc_auc"),
                learning_rate=current_lr,
                time_seconds=time.time() - epoch_start,
            )
            self.training_history.append(metrics)

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch + 1
                patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.detector.model.state_dict().items()
                }

                # Save checkpoint
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch + 1, val_metrics)
            else:
                patience_counter += 1

            # Logging
            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_f1={val_metrics['f1']:.3f}, "
                    f"lr={current_lr:.6f}"
                )

            # Early stopping
            if patience_counter >= self.config.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                early_stopped = True
                break

        # Load best model
        if self.best_model_state is not None:
            self.detector.model.load_state_dict(self.best_model_state)
            self.detector.model = self.detector.model.to(self.device)

        self.detector.is_fitted = True
        total_time = time.time() - start_time

        # Final validation metrics
        final_val_metrics = self._validate_epoch(val_loader, criterion)

        result = TrainingResult(
            model_type=self.config.model_type,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            best_val_f1=final_val_metrics["f1"],
            test_metrics=final_val_metrics,  # Will be updated if test set provided
            training_history=self.training_history,
            total_time_seconds=total_time,
            config=self.config.to_dict(),
            early_stopped=early_stopped,
            converged=not early_stopped,
        )

        if self.verbose:
            print(f"\nTraining complete in {total_time:.1f}s")
            print(f"Best epoch: {self.best_epoch}")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print(f"Best val F1: {final_val_metrics['f1']:.3f}")

        return result

    def evaluate(
        self,
        test_graphs: List[Union[CausalGraph, Dict[str, Any]]],
        test_labels: List[bool],
    ) -> Dict[str, float]:
        """
        Evaluate detector on test set.

        Args:
            test_graphs: Test graphs
            test_labels: Test labels

        Returns:
            Dictionary of test metrics
        """
        if self.detector is None or not self.detector.is_fitted:
            raise RuntimeError("Detector must be trained before evaluation")

        test_ds = CausalGraphDataset(test_graphs, test_labels)
        test_loader = test_ds.create_loader(
            batch_size=self.config.batch_size, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        metrics = self._validate_epoch(test_loader, criterion)

        if self.verbose:
            print(f"\nTest Results:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1']:.3f}")
            if metrics.get("roc_auc") is not None:
                print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")

        return metrics

    def cross_validate(
        self,
        graphs: List[Union[CausalGraph, Dict[str, Any]]],
        labels: List[bool],
        n_folds: int = 5,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            graphs: All graphs
            labels: All labels
            n_folds: Number of folds
            random_state: Random seed

        Returns:
            Dictionary with fold metrics and summary statistics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Cross-validation: {n_folds} folds")
            print(f"{'='*60}")

        kfold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

        fold_results = []
        all_labels = np.array(labels)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(graphs, labels)):
            if self.verbose:
                print(f"\nFold {fold_idx + 1}/{n_folds}")

            # Split data
            train_graphs = [graphs[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_graphs = [graphs[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]

            # Train
            self.train(train_graphs, train_labels)

            # Evaluate
            test_metrics = self.evaluate(test_graphs, test_labels)
            fold_results.append(
                {"fold": fold_idx + 1, "test_metrics": test_metrics}
            )

        # Compute summary statistics
        metric_names = ["accuracy", "precision", "recall", "f1"]
        summary = {}

        for metric in metric_names:
            values = [r["test_metrics"][metric] for r in fold_results]
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)

        if self.verbose:
            print(f"\n{'='*60}")
            print("Cross-validation Summary:")
            for metric in metric_names:
                mean_val = summary[f"{metric}_mean"]
                std_val = summary[f"{metric}_std"]
                print(f"  {metric.capitalize()}: {mean_val:.3f} +/- {std_val:.3f}")
            print(f"{'='*60}")

        return {
            "n_folds": n_folds,
            "fold_results": fold_results,
            "summary": summary,
        }

    def _train_epoch(
        self,
        loader: "DataLoader",
        optimizer: "torch.optim.Optimizer",
        criterion: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.detector.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            out = self.detector.model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

        num_samples = len(all_labels)
        return {
            "loss": total_loss / num_samples,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

    def _validate_epoch(
        self, loader: "DataLoader", criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.detector.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.detector.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)

                total_loss += loss.item() * batch.num_graphs
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        num_samples = len(all_labels)
        metrics = {
            "loss": total_loss / num_samples,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

        # ROC AUC if both classes present
        if len(set(all_labels)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
            except Exception:
                metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def _create_scheduler(
        self, optimizer: "torch.optim.Optimizer"
    ) -> Optional["torch.optim.lr_scheduler._LRScheduler"]:
        """Create learning rate scheduler based on config."""
        if self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
        elif self.config.lr_scheduler == "warmup":
            # Linear warmup then decay
            def warmup_lr(epoch):
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    return 0.5 ** ((epoch - warmup_epochs) // 30)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
        else:
            return None

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint_path = (
            self.checkpoint_dir
            / f"{self.config.model_type}_epoch{epoch}_f1{metrics['f1']:.3f}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.detector.model.state_dict(),
                "config": self.config.to_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )

    def get_detector(self) -> GNNDetector:
        """Get the trained detector."""
        if self.detector is None or not self.detector.is_fitted:
            raise RuntimeError("No trained detector available")
        return self.detector


# =============================================================================
# Experiment Runner
# =============================================================================


def run_architecture_comparison(
    graphs: List[Union[CausalGraph, Dict[str, Any]]],
    labels: List[bool],
    model_types: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    test_split: float = 0.2,
    val_split: float = 0.15,
    verbose: bool = True,
) -> Dict[str, TrainingResult]:
    """
    Compare multiple GNN architectures on the same data.

    Args:
        graphs: All graphs
        labels: All labels
        model_types: List of model types to compare
        config_overrides: Override default config parameters
        test_split: Test split ratio
        val_split: Validation split ratio
        verbose: Print progress

    Returns:
        Dictionary mapping model type to TrainingResult
    """
    from sklearn.model_selection import train_test_split as sk_split

    if model_types is None:
        model_types = ["gcn", "gat", "sage", "gin"]

    # Split data
    train_graphs, test_graphs, train_labels, test_labels = sk_split(
        graphs, labels, test_size=test_split, random_state=42, stratify=labels
    )

    results = {}

    for model_type in model_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()}")
            print(f"{'='*60}")

        config_params = {
            "model_type": model_type,
            "hidden_channels": 64,
            "num_layers": 3,
            "epochs": 100,
            "patience": 15,
        }
        if config_overrides:
            config_params.update(config_overrides)

        config = GNNConfig(**config_params)
        trainer = GNNTrainer(config=config, verbose=verbose)

        try:
            result = trainer.train(
                train_graphs, train_labels, val_split=val_split
            )

            # Evaluate on test set
            test_metrics = trainer.evaluate(test_graphs, test_labels)
            result.test_metrics = test_metrics

            results[model_type] = result

        except Exception as e:
            if verbose:
                print(f"Error training {model_type}: {e}")
            results[model_type] = None

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Architecture Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Model':<10} {'F1':>10} {'Precision':>12} {'Recall':>10} {'Accuracy':>10}")
        print("-" * 60)

        for model_type, result in results.items():
            if result is not None:
                m = result.test_metrics
                print(
                    f"{model_type.upper():<10} "
                    f"{m.get('f1', 0):.3f}        "
                    f"{m.get('precision', 0):.3f}         "
                    f"{m.get('recall', 0):.3f}       "
                    f"{m.get('accuracy', 0):.3f}"
                )
            else:
                print(f"{model_type.upper():<10} {'FAILED':>10}")

    return results


def run_hyperparameter_search(
    graphs: List[Union[CausalGraph, Dict[str, Any]]],
    labels: List[bool],
    model_type: str = "gin",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run hyperparameter search with cross-validation.

    Args:
        graphs: All graphs
        labels: All labels
        model_type: Base model type
        param_grid: Dictionary of parameters to search
        n_folds: Number of CV folds
        verbose: Print progress

    Returns:
        Dictionary with best parameters and results
    """
    if param_grid is None:
        param_grid = {
            "hidden_channels": [32, 64, 128],
            "num_layers": [2, 3, 4],
            "dropout": [0.3, 0.5, 0.7],
        }

    # Generate all parameter combinations
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    if verbose:
        print(f"\n{'='*60}")
        print(f"Hyperparameter Search: {len(combinations)} combinations")
        print(f"{'='*60}")

    best_f1 = 0
    best_params = None
    all_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params["model_type"] = model_type

        if verbose:
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

        config = GNNConfig(**params)
        trainer = GNNTrainer(config=config, verbose=False)

        try:
            cv_result = trainer.cross_validate(graphs, labels, n_folds=n_folds)
            mean_f1 = cv_result["summary"]["f1_mean"]

            all_results.append(
                {"params": params, "f1_mean": mean_f1, "f1_std": cv_result["summary"]["f1_std"]}
            )

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = params

            if verbose:
                print(f"  F1: {mean_f1:.3f} +/- {cv_result['summary']['f1_std']:.3f}")

        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
            all_results.append({"params": params, "error": str(e)})

    if verbose:
        print(f"\n{'='*60}")
        print(f"Best parameters: {best_params}")
        print(f"Best F1: {best_f1:.3f}")
        print(f"{'='*60}")

    return {
        "best_params": best_params,
        "best_f1": best_f1,
        "all_results": all_results,
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "GNNTrainer",
    "TrainingMetrics",
    "TrainingResult",
    "run_architecture_comparison",
    "run_hyperparameter_search",
]
