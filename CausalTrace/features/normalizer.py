"""
Feature normalization for ML models.

Provides standardization (z-score normalization) for feature vectors.
"""

from typing import List, Optional
import numpy as np
from causaltrace.features.feature_extractor import FeatureVector


class FeatureNormalizer:
    """
    Normalize features for ML models using z-score normalization.

    Z-score normalization: (x - mean) / std

    This ensures all features have mean=0 and std=1, which improves
    performance for many ML algorithms.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the normalizer.

        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, features: List[FeatureVector]) -> "FeatureNormalizer":
        """
        Compute normalization parameters from training data.

        Args:
            features: List of FeatureVector instances

        Returns:
            Self (for method chaining)
        """
        if not features:
            raise ValueError("Cannot fit normalizer with empty feature list")

        # Convert to numpy array
        X = np.array([f.to_numpy() for f in features])

        # Compute mean and std
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)

        self.is_fitted_ = True

        return self

    def transform(self, features: FeatureVector) -> FeatureVector:
        """
        Normalize a single feature vector.

        Args:
            features: FeatureVector instance

        Returns:
            Normalized FeatureVector
        """
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before transform")

        # Convert to numpy
        x = features.to_numpy()

        # Normalize
        x_normalized = (x - self.mean_) / self.std_

        # Create new FeatureVector with normalized values
        feature_names = FeatureVector.feature_names()
        normalized_dict = dict(zip(feature_names, x_normalized))

        return FeatureVector(**normalized_dict, metadata=features.metadata)

    def transform_batch(self, features: List[FeatureVector]) -> List[FeatureVector]:
        """
        Normalize multiple feature vectors.

        Args:
            features: List of FeatureVector instances

        Returns:
            List of normalized FeatureVectors
        """
        return [self.transform(f) for f in features]

    def fit_transform(self, features: List[FeatureVector]) -> List[FeatureVector]:
        """
        Fit normalizer and transform in one step.

        Args:
            features: List of FeatureVector instances

        Returns:
            List of normalized FeatureVectors
        """
        self.fit(features)
        return self.transform_batch(features)

    def inverse_transform(self, features: FeatureVector) -> FeatureVector:
        """
        Reverse normalization to get original scale.

        Args:
            features: Normalized FeatureVector

        Returns:
            FeatureVector in original scale
        """
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        # Convert to numpy
        x_normalized = features.to_numpy()

        # Reverse normalization
        x = (x_normalized * self.std_) + self.mean_

        # Create new FeatureVector
        feature_names = FeatureVector.feature_names()
        original_dict = dict(zip(feature_names, x))

        return FeatureVector(**original_dict, metadata=features.metadata)

    def save(self, filepath: str) -> None:
        """Save normalizer parameters to disk."""
        import pickle

        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted normalizer")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean_,
                'std': self.std_,
                'epsilon': self.epsilon,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "FeatureNormalizer":
        """Load normalizer parameters from disk."""
        import pickle

        with open(filepath, 'rb') as f:
            params = pickle.load(f)

        normalizer = cls(epsilon=params['epsilon'])
        normalizer.mean_ = params['mean']
        normalizer.std_ = params['std']
        normalizer.is_fitted_ = True

        return normalizer

    def get_params(self) -> dict:
        """Get normalizer parameters."""
        if not self.is_fitted_:
            return {}

        feature_names = FeatureVector.feature_names()
        return {
            'mean': dict(zip(feature_names, self.mean_)),
            'std': dict(zip(feature_names, self.std_)),
            'epsilon': self.epsilon,
        }


class MinMaxNormalizer:
    """
    Normalize features using min-max scaling to [0, 1] range.

    Min-max normalization: (x - min) / (max - min)

    This is an alternative to z-score normalization, useful when you want
    features bounded to a specific range.
    """

    def __init__(self, feature_range: tuple = (0, 1), epsilon: float = 1e-8):
        """
        Initialize the normalizer.

        Args:
            feature_range: Desired range of transformed data (default: (0, 1))
            epsilon: Small constant to avoid division by zero
        """
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, features: List[FeatureVector]) -> "MinMaxNormalizer":
        """Compute min and max from training data."""
        if not features:
            raise ValueError("Cannot fit normalizer with empty feature list")

        X = np.array([f.to_numpy() for f in features])

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        # Avoid division by zero
        range_vals = self.max_ - self.min_
        range_vals = np.where(range_vals < self.epsilon, 1.0, range_vals)
        self.range_ = range_vals

        self.is_fitted_ = True

        return self

    def transform(self, features: FeatureVector) -> FeatureVector:
        """Normalize a single feature vector to [0, 1] range."""
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before transform")

        x = features.to_numpy()

        # Scale to [0, 1]
        x_01 = (x - self.min_) / self.range_

        # Scale to desired range
        min_range, max_range = self.feature_range
        x_normalized = x_01 * (max_range - min_range) + min_range

        # Create new FeatureVector
        feature_names = FeatureVector.feature_names()
        normalized_dict = dict(zip(feature_names, x_normalized))

        return FeatureVector(**normalized_dict, metadata=features.metadata)

    def transform_batch(self, features: List[FeatureVector]) -> List[FeatureVector]:
        """Normalize multiple feature vectors."""
        return [self.transform(f) for f in features]

    def fit_transform(self, features: List[FeatureVector]) -> List[FeatureVector]:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform_batch(features)
