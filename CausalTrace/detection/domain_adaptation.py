"""
Domain Adaptation for CausalTrace Attack Detection.

This module implements techniques to reduce the synthetic-to-real gap in attack
detection models. It provides multiple adaptation methods that can be used
without requiring labeled real data.

Techniques implemented:
1. Feature Normalization (Z-score, MinMax, Robust)
2. Distribution Alignment (MMD, CORAL)
3. Data Augmentation (Edge dropout, Node masking, Feature perturbation, Mixup)
4. Self-Training (Pseudo-labeling)

Reference problem:
- Synthetic -> Real F1: 0.632 (poor generalization)
- Mixed -> Real F1: 1.000 (good with real data)
Goal: Improve synthetic->real performance WITHOUT requiring real labeled data.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import copy

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from causaltrace.graph import CausalGraph
from causaltrace.features import FeatureExtractor, FeatureVector


# =============================================================================
# Feature Normalization Strategies
# =============================================================================


class DomainNormalizer(ABC, BaseEstimator, TransformerMixin):
    """Base class for domain normalization strategies."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'DomainNormalizer':
        """Fit normalizer on source domain data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        pass

    @abstractmethod
    def adapt(self, X_target: np.ndarray) -> 'DomainNormalizer':
        """Adapt normalizer to target domain statistics (unlabeled)."""
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class ZScoreNormalizer(DomainNormalizer):
    """
    Z-score normalization with optional target domain adaptation.

    Standard z-score: (x - mean) / std
    Adapted z-score: Uses weighted combination of source and target statistics.
    """

    def __init__(self, epsilon: float = 1e-8, adaptation_weight: float = 0.5):
        """
        Args:
            epsilon: Small constant for numerical stability
            adaptation_weight: Weight for target domain stats (0 = source only, 1 = target only)
        """
        self.epsilon = epsilon
        self.adaptation_weight = adaptation_weight
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.source_mean_: Optional[np.ndarray] = None
        self.source_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'ZScoreNormalizer':
        """Fit on source domain data."""
        self.source_mean_ = np.mean(X, axis=0)
        self.source_std_ = np.std(X, axis=0)
        self.source_std_ = np.where(self.source_std_ < self.epsilon, 1.0, self.source_std_)

        # Initially use source stats
        self.mean_ = self.source_mean_.copy()
        self.std_ = self.source_std_.copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using current normalization parameters."""
        if self.mean_ is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (X - self.mean_) / self.std_

    def adapt(self, X_target: np.ndarray) -> 'ZScoreNormalizer':
        """Adapt to target domain using weighted combination."""
        if self.source_mean_ is None:
            raise RuntimeError("Normalizer must be fitted before adaptation")

        target_mean = np.mean(X_target, axis=0)
        target_std = np.std(X_target, axis=0)
        target_std = np.where(target_std < self.epsilon, 1.0, target_std)

        # Weighted combination
        w = self.adaptation_weight
        self.mean_ = (1 - w) * self.source_mean_ + w * target_mean
        self.std_ = (1 - w) * self.source_std_ + w * target_std
        return self


class MinMaxNormalizer(DomainNormalizer):
    """
    Min-Max normalization to [0, 1] range with domain adaptation.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1),
                 adaptation_weight: float = 0.5, epsilon: float = 1e-8):
        self.feature_range = feature_range
        self.adaptation_weight = adaptation_weight
        self.epsilon = epsilon
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.source_min_: Optional[np.ndarray] = None
        self.source_max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'MinMaxNormalizer':
        """Fit on source domain."""
        self.source_min_ = np.min(X, axis=0)
        self.source_max_ = np.max(X, axis=0)
        self.min_ = self.source_min_.copy()
        self.max_ = self.source_max_.copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to feature_range."""
        if self.min_ is None:
            raise RuntimeError("Normalizer must be fitted before transform")

        range_vals = self.max_ - self.min_
        range_vals = np.where(range_vals < self.epsilon, 1.0, range_vals)

        # Scale to [0, 1]
        X_01 = (X - self.min_) / range_vals

        # Scale to target range
        min_range, max_range = self.feature_range
        return X_01 * (max_range - min_range) + min_range

    def adapt(self, X_target: np.ndarray) -> 'MinMaxNormalizer':
        """Adapt to target domain."""
        if self.source_min_ is None:
            raise RuntimeError("Normalizer must be fitted before adaptation")

        target_min = np.min(X_target, axis=0)
        target_max = np.max(X_target, axis=0)

        w = self.adaptation_weight
        self.min_ = (1 - w) * self.source_min_ + w * target_min
        self.max_ = (1 - w) * self.source_max_ + w * target_max
        return self


class RobustNormalizer(DomainNormalizer):
    """
    Robust normalization using median and IQR, resistant to outliers.

    robust_score = (x - median) / IQR
    """

    def __init__(self, quantile_range: Tuple[float, float] = (25, 75),
                 adaptation_weight: float = 0.5, epsilon: float = 1e-8):
        self.quantile_range = quantile_range
        self.adaptation_weight = adaptation_weight
        self.epsilon = epsilon
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self.source_median_: Optional[np.ndarray] = None
        self.source_iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'RobustNormalizer':
        """Fit on source domain."""
        self.source_median_ = np.median(X, axis=0)
        q_low, q_high = self.quantile_range
        self.source_iqr_ = np.percentile(X, q_high, axis=0) - np.percentile(X, q_low, axis=0)
        self.source_iqr_ = np.where(self.source_iqr_ < self.epsilon, 1.0, self.source_iqr_)

        self.median_ = self.source_median_.copy()
        self.iqr_ = self.source_iqr_.copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using robust scaling."""
        if self.median_ is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (X - self.median_) / self.iqr_

    def adapt(self, X_target: np.ndarray) -> 'RobustNormalizer':
        """Adapt to target domain."""
        if self.source_median_ is None:
            raise RuntimeError("Normalizer must be fitted before adaptation")

        target_median = np.median(X_target, axis=0)
        q_low, q_high = self.quantile_range
        target_iqr = np.percentile(X_target, q_high, axis=0) - np.percentile(X_target, q_low, axis=0)
        target_iqr = np.where(target_iqr < self.epsilon, 1.0, target_iqr)

        w = self.adaptation_weight
        self.median_ = (1 - w) * self.source_median_ + w * target_median
        self.iqr_ = (1 - w) * self.source_iqr_ + w * target_iqr
        return self


# =============================================================================
# Distribution Alignment Methods
# =============================================================================


def compute_mmd(X_source: np.ndarray, X_target: np.ndarray,
                kernel: str = 'rbf', gamma: Optional[float] = None) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    MMD measures the distance between the mean embeddings of two distributions
    in a reproducing kernel Hilbert space (RKHS).

    Args:
        X_source: Source domain samples (n_source, n_features)
        X_target: Target domain samples (n_target, n_features)
        kernel: Kernel type ('rbf', 'linear', 'poly')
        gamma: RBF kernel bandwidth (auto-computed if None)

    Returns:
        MMD value (lower is better, 0 means identical distributions)
    """
    n_source = len(X_source)
    n_target = len(X_target)

    if gamma is None:
        # Median heuristic for bandwidth
        all_data = np.vstack([X_source, X_target])
        dists = np.sqrt(np.sum((all_data[:, None] - all_data[None, :]) ** 2, axis=-1))
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2)

    def rbf_kernel(X, Y, gamma):
        dists = np.sum(X ** 2, axis=1, keepdims=True) + \
                np.sum(Y ** 2, axis=1) - 2 * X @ Y.T
        return np.exp(-gamma * dists)

    def linear_kernel(X, Y):
        return X @ Y.T

    def poly_kernel(X, Y, degree=2, coef0=1):
        return (X @ Y.T + coef0) ** degree

    if kernel == 'rbf':
        K_ss = rbf_kernel(X_source, X_source, gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma)
        K_st = rbf_kernel(X_source, X_target, gamma)
    elif kernel == 'linear':
        K_ss = linear_kernel(X_source, X_source)
        K_tt = linear_kernel(X_target, X_target)
        K_st = linear_kernel(X_source, X_target)
    elif kernel == 'poly':
        K_ss = poly_kernel(X_source, X_source)
        K_tt = poly_kernel(X_target, X_target)
        K_st = poly_kernel(X_source, X_target)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2*E[k(x_s, x_t)]
    mmd_squared = (np.sum(K_ss) / (n_source * n_source) +
                   np.sum(K_tt) / (n_target * n_target) -
                   2 * np.sum(K_st) / (n_source * n_target))

    return max(0, mmd_squared) ** 0.5  # Return MMD (not squared)


class CORALAdapter:
    """
    CORrelation ALignment (CORAL) for domain adaptation.

    CORAL aligns the second-order statistics (covariances) of source and target
    distributions without requiring labels.

    Reference: Sun, Feng, & Saenko. "Return of Frustratingly Easy Domain Adaptation." AAAI 2016.
    """

    def __init__(self, regularization: float = 1e-6):
        """
        Args:
            regularization: Regularization for covariance matrix inversion
        """
        self.regularization = regularization
        self.A_: Optional[np.ndarray] = None
        self.source_mean_: Optional[np.ndarray] = None
        self.target_mean_: Optional[np.ndarray] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'CORALAdapter':
        """
        Fit CORAL transformation from source to target domain.

        Args:
            X_source: Source domain features
            X_target: Target domain features (unlabeled)
        """
        # Center data
        self.source_mean_ = np.mean(X_source, axis=0)
        self.target_mean_ = np.mean(X_target, axis=0)

        X_s_centered = X_source - self.source_mean_
        X_t_centered = X_target - self.target_mean_

        # Compute covariances
        n_s = X_source.shape[0]
        n_t = X_target.shape[0]

        C_s = (X_s_centered.T @ X_s_centered) / (n_s - 1) + self.regularization * np.eye(X_source.shape[1])
        C_t = (X_t_centered.T @ X_t_centered) / (n_t - 1) + self.regularization * np.eye(X_target.shape[1])

        # Compute transformation: A = C_s^{-1/2} @ C_t^{1/2}
        # Using eigendecomposition for numerical stability
        try:
            # Whitening transform for source
            eigvals_s, eigvecs_s = np.linalg.eigh(C_s)
            eigvals_s = np.maximum(eigvals_s, self.regularization)
            C_s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T

            # Coloring transform for target
            eigvals_t, eigvecs_t = np.linalg.eigh(C_t)
            eigvals_t = np.maximum(eigvals_t, self.regularization)
            C_t_sqrt = eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T

            self.A_ = C_s_inv_sqrt @ C_t_sqrt
        except np.linalg.LinAlgError:
            # Fall back to identity if decomposition fails
            warnings.warn("CORAL eigendecomposition failed, using identity transform")
            self.A_ = np.eye(X_source.shape[1])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform source domain data to align with target domain.

        Args:
            X: Source domain features

        Returns:
            Transformed features aligned with target distribution
        """
        if self.A_ is None:
            raise RuntimeError("CORALAdapter must be fitted before transform")

        # Center, transform, and re-center to target mean
        X_centered = X - self.source_mean_
        X_transformed = X_centered @ self.A_
        return X_transformed + self.target_mean_


class MMDAligner:
    """
    MMD-based feature alignment using gradient descent.

    Learns a linear transformation to minimize MMD between domains.
    Requires PyTorch for gradient-based optimization.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 num_iterations: int = 100,
                 kernel: str = 'rbf',
                 regularization: float = 0.01):
        """
        Args:
            learning_rate: Optimization learning rate
            num_iterations: Number of gradient descent iterations
            kernel: Kernel for MMD ('rbf', 'linear')
            regularization: L2 regularization on transformation
        """
        if not TORCH_AVAILABLE:
            raise ImportError("MMDAligner requires PyTorch. Install with: pip install torch")

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.kernel = kernel
        self.regularization = regularization
        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'MMDAligner':
        """
        Learn transformation to minimize MMD.

        Args:
            X_source: Source domain features
            X_target: Target domain features
        """
        n_features = X_source.shape[1]

        # Convert to tensors
        X_s = torch.FloatTensor(X_source)
        X_t = torch.FloatTensor(X_target)

        # Initialize transformation parameters
        W = torch.eye(n_features, requires_grad=True)
        b = torch.zeros(n_features, requires_grad=True)

        optimizer = torch.optim.Adam([W, b], lr=self.learning_rate)

        for i in range(self.num_iterations):
            optimizer.zero_grad()

            # Transform source
            X_s_transformed = X_s @ W + b

            # Compute MMD loss
            mmd_loss = self._compute_mmd_torch(X_s_transformed, X_t)

            # Add regularization to keep transformation close to identity
            reg_loss = self.regularization * (
                torch.sum((W - torch.eye(n_features)) ** 2) +
                torch.sum(b ** 2)
            )

            loss = mmd_loss + reg_loss
            loss.backward()
            optimizer.step()

        self.W_ = W.detach().numpy()
        self.b_ = b.detach().numpy()
        return self

    def _compute_mmd_torch(self, X_source: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
        """Compute MMD loss using PyTorch for gradient computation."""
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]

        # RBF kernel with median heuristic
        all_data = torch.cat([X_source, X_target], dim=0)
        dists = torch.cdist(all_data, all_data)
        gamma = 1.0 / (2 * torch.median(dists[dists > 0]) ** 2 + 1e-8)

        if self.kernel == 'rbf':
            K_ss = torch.exp(-gamma * torch.cdist(X_source, X_source) ** 2)
            K_tt = torch.exp(-gamma * torch.cdist(X_target, X_target) ** 2)
            K_st = torch.exp(-gamma * torch.cdist(X_source, X_target) ** 2)
        else:  # linear
            K_ss = X_source @ X_source.T
            K_tt = X_target @ X_target.T
            K_st = X_source @ X_target.T

        mmd = (torch.sum(K_ss) / (n_source * n_source) +
               torch.sum(K_tt) / (n_target * n_target) -
               2 * torch.sum(K_st) / (n_source * n_target))

        return mmd

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform source data using learned alignment."""
        if self.W_ is None:
            raise RuntimeError("MMDAligner must be fitted before transform")
        return X @ self.W_ + self.b_


# =============================================================================
# Data Augmentation for Graphs
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for graph data augmentation."""

    # Edge dropout
    edge_dropout_prob: float = 0.1

    # Node masking
    node_mask_prob: float = 0.1

    # Feature perturbation
    feature_noise_std: float = 0.05

    # Mixup
    mixup_alpha: float = 0.2

    # Number of augmented samples per original
    num_augmented: int = 2


class GraphAugmenter:
    """
    Data augmentation for causal graphs.

    Implements several augmentation strategies:
    1. Edge dropout: Randomly remove edges
    2. Node masking: Mask node features with zeros
    3. Feature perturbation: Add Gaussian noise to features
    4. Mixup: Interpolate between graph features
    """

    def __init__(self, config: Optional[AugmentationConfig] = None, seed: int = 42):
        """
        Args:
            config: Augmentation configuration
            seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        self.rng = np.random.RandomState(seed)

    def augment_graph(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply random augmentations to a graph dictionary.

        Args:
            graph_dict: Graph dictionary with 'nodes' and 'edges'

        Returns:
            Augmented graph dictionary
        """
        augmented = copy.deepcopy(graph_dict)

        # Edge dropout
        if self.config.edge_dropout_prob > 0:
            augmented = self._edge_dropout(augmented)

        # Node feature masking
        if self.config.node_mask_prob > 0:
            augmented = self._node_masking(augmented)

        return augmented

    def _edge_dropout(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly drop edges."""
        edges = graph_dict.get('edges', [])
        if not edges:
            return graph_dict

        # Keep edges with probability 1 - dropout_prob
        mask = self.rng.random(len(edges)) > self.config.edge_dropout_prob
        graph_dict['edges'] = [e for e, keep in zip(edges, mask) if keep]
        return graph_dict

    def _node_masking(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mask node features (set to default values)."""
        nodes = graph_dict.get('nodes', [])
        if not nodes:
            return graph_dict

        for node in nodes:
            if self.rng.random() < self.config.node_mask_prob:
                # Mask by setting to neutral values
                node['data_produced'] = []
                node['data_consumed'] = []
                if 'provenance' in node:
                    node['provenance']['is_untrusted'] = False
                    node['provenance']['injection_detected'] = False

        return graph_dict

    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to feature vectors.

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            Augmented features
        """
        augmented = features.copy()

        # Add Gaussian noise
        if self.config.feature_noise_std > 0:
            noise = self.rng.normal(0, self.config.feature_noise_std, features.shape)
            augmented = augmented + noise

        return augmented

    def mixup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation to features.

        Mixup creates virtual training examples by interpolating between
        pairs of examples and their labels.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Tuple of (mixed_X, mixed_y)
        """
        n_samples = len(X)

        # Sample mixing coefficients from Beta distribution
        lam = self.rng.beta(self.config.mixup_alpha, self.config.mixup_alpha, n_samples)
        lam = np.maximum(lam, 1 - lam)  # Ensure lambda >= 0.5

        # Sample random pairs
        indices = self.rng.permutation(n_samples)

        # Mix features
        lam_expanded = lam[:, np.newaxis]
        mixed_X = lam_expanded * X + (1 - lam_expanded) * X[indices]

        # Mix labels (for soft labels)
        y_float = y.astype(float)
        mixed_y = lam * y_float + (1 - lam) * y_float[indices]

        return mixed_X, mixed_y

    def generate_augmented_batch(
        self,
        graphs: List[Dict[str, Any]],
        labels: List[bool]
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        """
        Generate augmented versions of the input graphs.

        Args:
            graphs: List of graph dictionaries
            labels: List of labels

        Returns:
            Tuple of (augmented_graphs, augmented_labels)
        """
        augmented_graphs = []
        augmented_labels = []

        for graph, label in zip(graphs, labels):
            # Keep original
            augmented_graphs.append(graph)
            augmented_labels.append(label)

            # Generate augmented versions
            for _ in range(self.config.num_augmented):
                aug_graph = self.augment_graph(graph)
                augmented_graphs.append(aug_graph)
                augmented_labels.append(label)

        return augmented_graphs, augmented_labels


# =============================================================================
# Self-Training with Pseudo-Labels
# =============================================================================


class SelfTrainer:
    """
    Self-training for domain adaptation.

    1. Train on labeled source data
    2. Predict on unlabeled target data
    3. Select high-confidence predictions as pseudo-labels
    4. Retrain with source + pseudo-labeled target data
    5. Repeat until convergence
    """

    def __init__(
        self,
        base_detector: Any,
        confidence_threshold: float = 0.9,
        max_iterations: int = 10,
        min_pseudo_ratio: float = 0.1,
        max_pseudo_ratio: float = 0.5,
    ):
        """
        Args:
            base_detector: Any detector with fit/predict interface
            confidence_threshold: Min confidence for pseudo-labeling
            max_iterations: Maximum self-training iterations
            min_pseudo_ratio: Min ratio of target samples to pseudo-label
            max_pseudo_ratio: Max ratio to prevent overfitting to pseudo-labels
        """
        self.base_detector = base_detector
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_pseudo_ratio = min_pseudo_ratio
        self.max_pseudo_ratio = max_pseudo_ratio
        self.iteration_history: List[Dict[str, Any]] = []

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        graphs_source: Optional[List[Any]] = None,
        graphs_target: Optional[List[Any]] = None,
    ) -> 'SelfTrainer':
        """
        Perform self-training.

        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (unlabeled)
            graphs_source: Optional source graphs (for graph-based detectors)
            graphs_target: Optional target graphs
        """
        # Initial training on source
        self._train_detector(X_source, y_source, graphs_source)

        # Track which target samples have pseudo-labels
        pseudo_labels = np.full(len(X_target), -1, dtype=int)

        for iteration in range(self.max_iterations):
            # Predict on target
            confidences, predictions = self._predict_with_confidence(
                X_target, graphs_target
            )

            # Select high-confidence predictions
            high_conf_mask = confidences >= self.confidence_threshold

            # Update pseudo-labels
            new_pseudo = 0
            for i in range(len(X_target)):
                if high_conf_mask[i] and pseudo_labels[i] == -1:
                    pseudo_labels[i] = predictions[i]
                    new_pseudo += 1

            # Check stopping criteria
            pseudo_ratio = np.sum(pseudo_labels >= 0) / len(X_target)

            self.iteration_history.append({
                'iteration': iteration,
                'new_pseudo_labels': new_pseudo,
                'total_pseudo_labels': np.sum(pseudo_labels >= 0),
                'pseudo_ratio': pseudo_ratio,
                'mean_confidence': np.mean(confidences),
            })

            if new_pseudo == 0:
                print(f"Self-training converged at iteration {iteration}")
                break

            if pseudo_ratio >= self.max_pseudo_ratio:
                print(f"Reached max pseudo-label ratio at iteration {iteration}")
                break

            # Combine source and pseudo-labeled target
            pseudo_mask = pseudo_labels >= 0
            X_combined = np.vstack([X_source, X_target[pseudo_mask]])
            y_combined = np.concatenate([y_source, pseudo_labels[pseudo_mask]])

            if graphs_source is not None and graphs_target is not None:
                graphs_combined = graphs_source + [
                    graphs_target[i] for i in range(len(graphs_target)) if pseudo_mask[i]
                ]
            else:
                graphs_combined = None

            # Retrain
            self._train_detector(X_combined, y_combined, graphs_combined)

        return self

    def _train_detector(
        self,
        X: np.ndarray,
        y: np.ndarray,
        graphs: Optional[List[Any]] = None
    ) -> None:
        """Train the base detector."""
        if graphs is not None and hasattr(self.base_detector, 'fit'):
            # For graph-based detectors
            self.base_detector.fit(graphs, y.astype(bool).tolist())
        elif hasattr(self.base_detector, 'fit'):
            self.base_detector.fit(X, y)

    def _predict_with_confidence(
        self,
        X: np.ndarray,
        graphs: Optional[List[Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and confidence scores."""
        if graphs is not None and hasattr(self.base_detector, 'predict_batch'):
            results = self.base_detector.predict_batch(graphs)
            confidences = np.array([r.confidence for r in results])
            predictions = np.array([int(r.is_attack) for r in results])
        elif hasattr(self.base_detector, 'predict_proba'):
            proba = self.base_detector.predict_proba(X)
            predictions = np.argmax(proba, axis=1)
            confidences = np.max(proba, axis=1)
        else:
            raise ValueError("Detector must have predict_batch or predict_proba method")

        return confidences, predictions

    def predict(self, X: np.ndarray, graphs: Optional[List[Any]] = None) -> np.ndarray:
        """Predict using the trained detector."""
        _, predictions = self._predict_with_confidence(X, graphs)
        return predictions


# =============================================================================
# Unified Domain Adaptation Pipeline
# =============================================================================


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation pipeline."""

    # Normalization
    normalization: str = "zscore"  # zscore, minmax, robust, none
    adaptation_weight: float = 0.5

    # Distribution alignment
    alignment: str = "coral"  # coral, mmd, none
    mmd_iterations: int = 100

    # Augmentation
    augmentation: bool = True
    aug_edge_dropout: float = 0.1
    aug_node_mask: float = 0.1
    aug_feature_noise: float = 0.05
    aug_mixup_alpha: float = 0.2
    aug_num_samples: int = 2

    # Self-training
    self_training: bool = False
    st_confidence_threshold: float = 0.9
    st_max_iterations: int = 10


class DomainAdaptationPipeline:
    """
    Unified pipeline for domain adaptation in CausalTrace.

    Combines multiple techniques:
    1. Feature normalization with target adaptation
    2. Distribution alignment (CORAL or MMD)
    3. Data augmentation
    4. Optional self-training
    """

    def __init__(self, config: Optional[DomainAdaptationConfig] = None):
        """
        Args:
            config: Pipeline configuration
        """
        self.config = config or DomainAdaptationConfig()
        self.normalizer: Optional[DomainNormalizer] = None
        self.aligner: Optional[Union[CORALAdapter, MMDAligner]] = None
        self.augmenter: Optional[GraphAugmenter] = None
        self.self_trainer: Optional[SelfTrainer] = None

    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        graphs_source: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit on source and target, transform source to match target.

        Args:
            X_source: Source domain features
            X_target: Target domain features (unlabeled)
            graphs_source: Optional source graphs for augmentation

        Returns:
            Tuple of (transformed_source, normalized_target)
        """
        # 1. Initialize normalizer
        if self.config.normalization == "zscore":
            self.normalizer = ZScoreNormalizer(adaptation_weight=self.config.adaptation_weight)
        elif self.config.normalization == "minmax":
            self.normalizer = MinMaxNormalizer(adaptation_weight=self.config.adaptation_weight)
        elif self.config.normalization == "robust":
            self.normalizer = RobustNormalizer(adaptation_weight=self.config.adaptation_weight)
        else:
            self.normalizer = None

        # Fit normalizer on source, adapt to target
        if self.normalizer is not None:
            self.normalizer.fit(X_source)
            self.normalizer.adapt(X_target)
            X_source_norm = self.normalizer.transform(X_source)
            X_target_norm = self.normalizer.transform(X_target)
        else:
            X_source_norm = X_source.copy()
            X_target_norm = X_target.copy()

        # 2. Distribution alignment
        if self.config.alignment == "coral":
            self.aligner = CORALAdapter()
            self.aligner.fit(X_source_norm, X_target_norm)
            X_source_aligned = self.aligner.transform(X_source_norm)
        elif self.config.alignment == "mmd":
            if TORCH_AVAILABLE:
                self.aligner = MMDAligner(num_iterations=self.config.mmd_iterations)
                self.aligner.fit(X_source_norm, X_target_norm)
                X_source_aligned = self.aligner.transform(X_source_norm)
            else:
                warnings.warn("MMD alignment requires PyTorch, skipping")
                X_source_aligned = X_source_norm
        else:
            X_source_aligned = X_source_norm

        return X_source_aligned, X_target_norm

    def transform(self, X: np.ndarray, source_domain: bool = True) -> np.ndarray:
        """
        Transform features using fitted pipeline.

        Args:
            X: Features to transform
            source_domain: If True, apply full pipeline (norm + align).
                          If False, only normalize (for target domain).
        """
        if self.normalizer is not None:
            X_norm = self.normalizer.transform(X)
        else:
            X_norm = X

        if source_domain and self.aligner is not None:
            return self.aligner.transform(X_norm)
        return X_norm

    def setup_augmentation(self) -> GraphAugmenter:
        """Initialize graph augmenter."""
        aug_config = AugmentationConfig(
            edge_dropout_prob=self.config.aug_edge_dropout,
            node_mask_prob=self.config.aug_node_mask,
            feature_noise_std=self.config.aug_feature_noise,
            mixup_alpha=self.config.aug_mixup_alpha,
            num_augmented=self.config.aug_num_samples,
        )
        self.augmenter = GraphAugmenter(config=aug_config)
        return self.augmenter

    def setup_self_training(self, base_detector: Any) -> SelfTrainer:
        """Initialize self-trainer with a base detector."""
        self.self_trainer = SelfTrainer(
            base_detector=base_detector,
            confidence_threshold=self.config.st_confidence_threshold,
            max_iterations=self.config.st_max_iterations,
        )
        return self.self_trainer

    def get_mmd_score(self, X_source: np.ndarray, X_target: np.ndarray) -> float:
        """Compute MMD between transformed distributions."""
        X_s_transformed = self.transform(X_source, source_domain=True)
        X_t_transformed = self.transform(X_target, source_domain=False)
        return compute_mmd(X_s_transformed, X_t_transformed)


# =============================================================================
# Utility Functions
# =============================================================================


def get_normalizer(name: str, **kwargs) -> DomainNormalizer:
    """
    Factory function to get normalizer by name.

    Args:
        name: Normalizer type ('zscore', 'minmax', 'robust')
        **kwargs: Additional arguments for the normalizer

    Returns:
        DomainNormalizer instance
    """
    normalizers = {
        'zscore': ZScoreNormalizer,
        'minmax': MinMaxNormalizer,
        'robust': RobustNormalizer,
    }

    if name.lower() not in normalizers:
        raise ValueError(f"Unknown normalizer: {name}. Available: {list(normalizers.keys())}")

    return normalizers[name.lower()](**kwargs)


def evaluate_domain_gap(
    X_source: np.ndarray,
    X_target: np.ndarray,
    methods: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate domain gap using multiple metrics.

    Args:
        X_source: Source domain features
        X_target: Target domain features
        methods: List of metrics ('mmd_rbf', 'mmd_linear', 'feature_diff')

    Returns:
        Dictionary of metric names to values
    """
    if methods is None:
        methods = ['mmd_rbf', 'mmd_linear', 'feature_diff']

    results = {}

    if 'mmd_rbf' in methods:
        results['mmd_rbf'] = compute_mmd(X_source, X_target, kernel='rbf')

    if 'mmd_linear' in methods:
        results['mmd_linear'] = compute_mmd(X_source, X_target, kernel='linear')

    if 'feature_diff' in methods:
        # Mean absolute difference in feature means
        mean_diff = np.mean(np.abs(np.mean(X_source, axis=0) - np.mean(X_target, axis=0)))
        results['feature_diff'] = mean_diff

    if 'std_diff' in methods:
        # Difference in feature standard deviations
        std_diff = np.mean(np.abs(np.std(X_source, axis=0) - np.std(X_target, axis=0)))
        results['std_diff'] = std_diff

    return results
