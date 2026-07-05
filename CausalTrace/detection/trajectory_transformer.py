"""
Trajectory Transformer Baseline for comparison with CausalTrace.

This module provides sequence-based baselines that operate on the SAME raw
trajectory data as CausalTrace, but without causal graph structure. This
enables fair comparison to show the value of graph-based representation.

Two baselines provided:
1. TFIDFSequenceDetector - TF-IDF + Logistic Regression (always available)
2. TrajectoryTransformerBaseline - Fine-tuned DistilBERT (requires transformers)

Usage:
    from causaltrace.detection.trajectory_transformer import get_sequence_baseline

    # Get appropriate baseline (falls back to TF-IDF if transformers unavailable)
    detector = get_sequence_baseline(use_transformer=False)
    detector.fit(train_trajectories, train_labels)
    result = detector.predict(test_trajectory)
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from causaltrace.models.trajectory import Trajectory, Action
from causaltrace.detection.detector import BaseDetector, DetectionResult

# Check if transformers available
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SequenceEncodingConfig:
    """Configuration for trajectory sequence encoding."""
    include_action_type: bool = True
    include_target: bool = True
    include_result: bool = True
    include_domain: bool = True
    max_result_length: int = 200
    action_separator: str = " [SEP] "
    truncate_actions: int = 50


class TrajectorySequenceEncoder:
    """Encode trajectories as text sequences for transformer/TF-IDF input."""

    def __init__(self, config: Optional[SequenceEncodingConfig] = None):
        self.config = config or SequenceEncodingConfig()

    def encode_action(self, action: Action) -> str:
        """Encode a single action as text."""
        parts = []

        if self.config.include_action_type:
            parts.append(f"[{action.action_type.value.upper()}]")

        if self.config.include_domain and action.domain:
            parts.append(f"domain:{action.domain}")

        if self.config.include_target and action.target:
            target = str(action.target)[:100]
            parts.append(f"target:{target}")

        if self.config.include_result and action.result:
            result = str(action.result)[:self.config.max_result_length]
            result = result.replace("\n", " ").strip()
            if result:
                parts.append(f"result:{result}")

        return " ".join(parts)

    def encode(self, trajectory: Trajectory) -> str:
        """Encode a full trajectory as text sequence."""
        actions = trajectory.actions[:self.config.truncate_actions]
        action_texts = [self.encode_action(a) for a in actions]
        return self.config.action_separator.join(action_texts)

    def encode_batch(self, trajectories: List[Trajectory]) -> List[str]:
        """Encode multiple trajectories."""
        return [self.encode(t) for t in trajectories]


class TFIDFSequenceDetector(BaseDetector):
    """
    TF-IDF + Logistic Regression baseline on trajectory sequences.

    This baseline:
    - Uses the SAME raw trajectory data as CausalTrace
    - Encodes trajectories as text sequences
    - Uses TF-IDF vectorization + Logistic Regression
    - Always available (no heavy dependencies)
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        encoding_config: Optional[SequenceEncodingConfig] = None,
    ):
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.encoder = TrajectorySequenceEncoder(encoding_config)
        self.pipeline = None
        self.is_fitted = False
        self.metadata = {"model_type": "tfidf_logreg"}

    def fit(
        self,
        trajectories: List[Trajectory],
        labels: List[bool],
        verbose: bool = True
    ) -> 'TFIDFSequenceDetector':
        """Train TF-IDF + Logistic Regression model."""
        if verbose:
            print(f"Training TF-IDF Sequence Detector on {len(trajectories)} trajectories...")

        start = time.time()
        sequences = self.encoder.encode_batch(trajectories)

        if verbose:
            avg_len = np.mean([len(s.split()) for s in sequences])
            print(f"  Encoded in {time.time() - start:.2f}s, avg length: {avg_len:.1f} tokens")

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=True
            )),
            ('classifier', LogisticRegression(
                C=self.C,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ))
        ])

        y = np.array(labels, dtype=int)
        self.pipeline.fit(sequences, y)

        if verbose:
            vocab_size = len(self.pipeline.named_steps['tfidf'].vocabulary_)
            print(f"  Training completed. Vocabulary size: {vocab_size}")

        self.is_fitted = True
        self.metadata['train_samples'] = len(trajectories)
        return self

    def predict(self, trajectory: Trajectory) -> DetectionResult:
        """Predict whether trajectory is an attack."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        sequence = self.encoder.encode(trajectory)
        pred = self.pipeline.predict([sequence])[0]
        proba = self.pipeline.predict_proba([sequence])[0]

        is_attack = bool(pred)
        confidence = float(proba[1]) if is_attack else float(proba[0])

        return DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            explanation=f"TF-IDF: {'ATTACK' if is_attack else 'BENIGN'} (p={proba[1]:.2%})",
        )

    def predict_batch(self, trajectories: List[Trajectory]) -> List[DetectionResult]:
        """Predict for multiple trajectories."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        sequences = self.encoder.encode_batch(trajectories)
        preds = self.pipeline.predict(sequences)
        probas = self.pipeline.predict_proba(sequences)

        results = []
        for pred, proba in zip(preds, probas):
            is_attack = bool(pred)
            confidence = float(proba[1]) if is_attack else float(proba[0])
            results.append(DetectionResult(
                is_attack=is_attack,
                confidence=confidence,
                explanation=f"TF-IDF: {'ATTACK' if is_attack else 'BENIGN'} (p={proba[1]:.2%})",
            ))
        return results

    def get_top_features(self, top_k: int = 20) -> Dict[str, List[tuple]]:
        """Get top features for attack and benign classes."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted")

        tfidf = self.pipeline.named_steps['tfidf']
        clf = self.pipeline.named_steps['classifier']

        feature_names = tfidf.get_feature_names_out()
        coefficients = clf.coef_[0]

        sorted_indices = np.argsort(coefficients)

        return {
            'attack': [(feature_names[i], coefficients[i]) for i in sorted_indices[-top_k:][::-1]],
            'benign': [(feature_names[i], coefficients[i]) for i in sorted_indices[:top_k]],
        }

    def save(self, path: str) -> None:
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'encoder_config': self.encoder.config,
                'metadata': self.metadata,
                'is_fitted': self.is_fitted,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'TFIDFSequenceDetector':
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        detector = cls(encoding_config=data['encoder_config'])
        detector.pipeline = data['pipeline']
        detector.metadata = data['metadata']
        detector.is_fitted = data['is_fitted']
        return detector


class TrajectoryTransformerBaseline(BaseDetector):
    """
    Transformer baseline on raw trajectory sequences.

    Requires: pip install transformers torch

    This baseline:
    - Uses the SAME raw trajectory data as CausalTrace
    - Fine-tunes a DistilBERT model on trajectory classification
    - Provides stronger sequence modeling than TF-IDF
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        encoding_config: Optional[SequenceEncodingConfig] = None,
        device: Optional[str] = None,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "TrajectoryTransformerBaseline requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.encoder = TrajectorySequenceEncoder(encoding_config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.is_fitted = False
        self.metadata = {
            "model_type": "transformer",
            "model_name": model_name,
            "device": self.device,
        }

    def fit(
        self,
        trajectories: List[Trajectory],
        labels: List[bool],
        verbose: bool = True
    ) -> 'TrajectoryTransformerBaseline':
        """Fine-tune transformer on trajectory classification."""
        if verbose:
            print(f"Training Transformer ({self.model_name}) on {len(trajectories)} trajectories...")
            print(f"  Device: {self.device}")

        sequences = self.encoder.encode_batch(trajectories)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

        # Create dataset
        class TrajectoryDataset(Dataset):
            def __init__(ds_self, texts, labels, tokenizer, max_length):
                ds_self.texts = texts
                ds_self.labels = labels
                ds_self.tokenizer = tokenizer
                ds_self.max_length = max_length

            def __len__(ds_self):
                return len(ds_self.texts)

            def __getitem__(ds_self, idx):
                enc = ds_self.tokenizer(
                    ds_self.texts[idx],
                    truncation=True,
                    max_length=ds_self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                return {
                    'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'labels': torch.tensor(int(ds_self.labels[idx]))
                }

        train_dataset = TrajectoryDataset(sequences, labels, self.tokenizer, self.max_length)

        training_args = TrainingArguments(
            output_dir='./transformer_baseline_ckpts',
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy='no',
            report_to='none',
            disable_tqdm=not verbose,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

        self.is_fitted = True
        self.metadata['train_samples'] = len(trajectories)

        if verbose:
            print("  Training completed.")

        return self

    def predict(self, trajectory: Trajectory) -> DetectionResult:
        """Predict whether trajectory is an attack."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        sequence = self.encoder.encode(trajectory)
        inputs = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        is_attack = bool(torch.argmax(outputs.logits, dim=1).item())
        attack_prob = probs[1].item()

        return DetectionResult(
            is_attack=is_attack,
            confidence=attack_prob if is_attack else probs[0].item(),
            explanation=f"Transformer: {'ATTACK' if is_attack else 'BENIGN'} (p={attack_prob:.2%})",
        )

    def predict_batch(self, trajectories: List[Trajectory]) -> List[DetectionResult]:
        """Predict for multiple trajectories."""
        return [self.predict(t) for t in trajectories]

    def save(self, path: str) -> None:
        """Save trained model."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        model_dir = path_obj.parent / f"{path_obj.stem}_model"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        with open(path, 'wb') as f:
            pickle.dump({
                'model_dir': str(model_dir),
                'encoder_config': self.encoder.config,
                'metadata': self.metadata,
                'is_fitted': self.is_fitted,
                'model_name': self.model_name,
                'max_length': self.max_length,
                'device': self.device,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'TrajectoryTransformerBaseline':
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        detector = cls(
            model_name=data['model_name'],
            max_length=data['max_length'],
            encoding_config=data['encoder_config'],
            device=data['device'],
        )
        detector.tokenizer = AutoTokenizer.from_pretrained(data['model_dir'])
        detector.model = AutoModelForSequenceClassification.from_pretrained(
            data['model_dir']
        ).to(detector.device)
        detector.metadata = data['metadata']
        detector.is_fitted = data['is_fitted']
        return detector


def get_sequence_baseline(use_transformer: bool = False, **kwargs) -> BaseDetector:
    """
    Get appropriate sequence baseline detector.

    Args:
        use_transformer: If True and transformers available, use transformer
        **kwargs: Passed to detector constructor

    Returns:
        TFIDFSequenceDetector or TrajectoryTransformerBaseline
    """
    if use_transformer and TRANSFORMERS_AVAILABLE:
        return TrajectoryTransformerBaseline(**kwargs)

    if use_transformer and not TRANSFORMERS_AVAILABLE:
        print("Warning: transformers not available, falling back to TF-IDF baseline")

    return TFIDFSequenceDetector(**kwargs)


__all__ = [
    'TrajectorySequenceEncoder',
    'SequenceEncodingConfig',
    'TFIDFSequenceDetector',
    'TrajectoryTransformerBaseline',
    'get_sequence_baseline',
    'TRANSFORMERS_AVAILABLE',
]
