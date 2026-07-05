"""
Machine learning-based attack detector.

Uses sklearn models (Random Forest, Gradient Boosting, Logistic Regression)
to classify causal graphs based on extracted features.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from .detector import BaseDetector, DetectionResult
from ..features import FeatureExtractor, FeatureVector
from ..graph import CausalGraph


class MLDetector(BaseDetector):
    """
    Machine learning-based detector using causal graph features.

    Supports multiple sklearn models:
    - Random Forest (good for feature importance and non-linear patterns)
    - Gradient Boosting (often best performance)
    - Logistic Regression (interpretable linear baseline)
    """

    SUPPORTED_MODELS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'logistic_regression': LogisticRegression,
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        normalize_features: bool = True
    ):
        """
        Initialize ML detector.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            model_params: Optional hyperparameters for the model
            normalize_features: Whether to standardize features before training

        Raises:
            ValueError: If model_type is not supported
        """
        super().__init__()

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_params = model_params or {}
        self.normalize_features = normalize_features

        self.model = self._create_model(model_type, self.model_params)
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler() if normalize_features else None

        self.feature_names = FeatureVector.feature_names()
        self.feature_importance_: Optional[Dict[str, float]] = None

    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """
        Create sklearn model with given parameters.

        Args:
            model_type: Model type
            params: Hyperparameters

        Returns:
            Sklearn model instance
        """
        default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            },
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': 42,
            },
        }

        # Merge default params with user-provided params
        final_params = {**default_params[model_type], **params}

        model_class = self.SUPPORTED_MODELS[model_type]
        return model_class(**final_params)

    def fit(self, graphs: List[CausalGraph], labels: List[bool]) -> 'MLDetector':
        """
        Train ML model on labeled causal graphs.

        Args:
            graphs: List of CausalGraph objects
            labels: List of boolean labels (True = attack, False = benign)

        Returns:
            self (for method chaining)
        """
        super().fit(graphs, labels)  # Validates input

        print(f"Training {self.model_type} on {len(graphs)} graphs...")

        # Extract features from all graphs
        features = self.feature_extractor.extract_batch(graphs)
        X = np.array([f.to_numpy() for f in features])
        y = np.array(labels, dtype=int)

        # Normalize if requested
        if self.normalize_features:
            X = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X, y)

        # Compute feature importance
        self._compute_feature_importance()

        # Cross-validation score (with dynamic cv based on dataset size)
        # Guard against small datasets and degenerate class distributions
        n_samples = len(graphs)
        n_positive = sum(labels)
        n_negative = n_samples - n_positive
        min_class_size = min(n_positive, n_negative)

        if n_samples >= 5 and min_class_size >= 2:
            # Use at most 5 folds, but no more than min_class_size
            n_folds = min(5, n_samples, min_class_size)
            try:
                cv_scores = cross_val_score(self.model, X, y, cv=n_folds, scoring='f1')
            except ValueError as e:
                # Fallback if CV still fails
                print(f"Warning: Cross-validation failed ({e}), skipping CV scoring")
                cv_scores = np.array([])
        else:
            # Dataset too small for CV
            print(f"Warning: Dataset too small for cross-validation ({n_samples} samples, {min_class_size} min class)")
            cv_scores = np.array([])

        self.is_fitted = True
        self.metadata['num_training_samples'] = len(graphs)
        self.metadata['num_attacks'] = sum(labels)
        self.metadata['model_type'] = self.model_type

        # Handle empty CV scores
        if len(cv_scores) > 0:
            self.metadata['cv_f1_mean'] = float(np.mean(cv_scores))
            self.metadata['cv_f1_std'] = float(np.std(cv_scores))
            print(f"Training complete. CV F1: {self.metadata['cv_f1_mean']:.3f} ± {self.metadata['cv_f1_std']:.3f}")
        else:
            self.metadata['cv_f1_mean'] = None
            self.metadata['cv_f1_std'] = None
            print("Training complete. (CV scores unavailable due to small dataset)")

        return self

    def _compute_feature_importance(self) -> None:
        """Compute and store feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names))

        self.feature_importance_ = {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }

    def predict(self, graph: CausalGraph) -> DetectionResult:
        """
        Predict whether a causal graph represents an attack.

        Args:
            graph: CausalGraph object

        Returns:
            DetectionResult with classification and explanation
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        # Extract features
        features = self.feature_extractor.extract(graph)
        X = np.array([features.to_numpy()])

        # Normalize if necessary
        if self.normalize_features:
            X = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        is_attack = bool(prediction == 1)
        confidence = float(probabilities[1] if is_attack else probabilities[0])

        # Generate explanation based on top contributing features
        explanation = self._generate_explanation(features, is_attack, confidence)

        # Get top triggered features
        triggered_features = self._get_top_features(features, n=5)

        result = DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            triggered_features=triggered_features,
            explanation=explanation,
            raw_scores={
                'attack_probability': float(probabilities[1]),
                'benign_probability': float(probabilities[0]),
            }
        )
        return self._attach_watermark_metadata(graph, result)

    def _generate_explanation(self, features: FeatureVector, is_attack: bool, confidence: float) -> str:
        """Generate human-readable explanation."""
        result = "attack" if is_attack else "benign"

        # Get top 3 most important features for this prediction
        top_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: -x[1]
        )[:3]

        feature_desc = ", ".join([
            f"{name}={getattr(features, name):.2f}" if isinstance(getattr(features, name), float)
            else f"{name}={getattr(features, name)}"
            for name, _ in top_features
        ])

        return f"Classified as {result} with {confidence:.1%} confidence. Key features: {feature_desc}"

    def _get_top_features(self, features: FeatureVector, n: int = 5) -> Dict[str, Any]:
        """Get top N most important features for this prediction."""
        feature_dict = features.to_dict()

        # Sort by importance
        sorted_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: -x[1]
        )[:n]

        return {
            name: feature_dict[name]
            for name, _ in sorted_features
        }

    def predict_batch(self, graphs: List[CausalGraph]) -> List[DetectionResult]:
        """
        Predict for multiple graphs (optimized batch version).

        Args:
            graphs: List of CausalGraph objects

        Returns:
            List of DetectionResult objects
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        # Extract features
        features_list = self.feature_extractor.extract_batch(graphs)
        X = np.array([f.to_numpy() for f in features_list])

        # Normalize if necessary
        if self.normalize_features:
            X = self.scaler.transform(X)

        # Batch predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Create results
        results = []
        for graph, features, pred, proba in zip(graphs, features_list, predictions, probabilities):
            is_attack = bool(pred == 1)
            confidence = float(proba[1] if is_attack else proba[0])

            explanation = self._generate_explanation(features, is_attack, confidence)
            triggered_features = self._get_top_features(features, n=5)

            result = DetectionResult(
                is_attack=is_attack,
                confidence=confidence,
                triggered_features=triggered_features,
                explanation=explanation,
                raw_scores={
                    'attack_probability': float(proba[1]),
                    'benign_probability': float(proba[0]),
                }
            )
            results.append(self._attach_watermark_metadata(graph, result))

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If detector has not been fitted
        """
        if not self.is_fitted or self.feature_importance_ is None:
            raise RuntimeError("Detector must be fitted before accessing feature importance")

        return self.feature_importance_.copy()

    def save(self, path: str) -> None:
        """
        Save trained detector to disk.

        Saves both the model and the scaler separately using joblib.

        Args:
            path: Path to save directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path_obj / "model.joblib"
        joblib.dump(self.model, model_path)

        # Save scaler
        if self.scaler is not None:
            scaler_path = path_obj / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata_path = path_obj / "metadata.joblib"
        joblib.dump({
            'model_type': self.model_type,
            'normalize_features': self.normalize_features,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata,
        }, metadata_path)

        print(f"Detector saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'MLDetector':
        """
        Load trained detector from disk.

        Args:
            path: Path to saved detector directory

        Returns:
            Loaded MLDetector instance
        """
        path_obj = Path(path)

        # Load metadata
        metadata_path = path_obj / "metadata.joblib"
        metadata = joblib.load(metadata_path)

        # Create detector instance
        detector = cls(
            model_type=metadata['model_type'],
            normalize_features=metadata['normalize_features']
        )

        # Load model
        model_path = path_obj / "model.joblib"
        detector.model = joblib.load(model_path)

        # Load scaler
        if metadata['normalize_features']:
            scaler_path = path_obj / "scaler.joblib"
            detector.scaler = joblib.load(scaler_path)

        # Restore state
        detector.feature_names = metadata['feature_names']
        detector.feature_importance_ = metadata['feature_importance']
        detector.is_fitted = metadata['is_fitted']
        detector.metadata = metadata['metadata']

        print(f"Detector loaded from {path}")
        return detector

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"MLDetector(model={self.model_type}, {fitted_status})"
