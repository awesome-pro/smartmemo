"""Classifier training and inference utilities."""

from smartmemo.classifier.data import EmbeddedPairDataset, PairRecord, load_pair_records
from smartmemo.classifier.evaluate import EvaluationMetrics, compute_binary_metrics, evaluate_model
from smartmemo.classifier.model import PairClassifier, build_pair_features
from smartmemo.classifier.service import ClassifierService
from smartmemo.classifier.train import TrainingConfig, TrainingResult, train_classifier

__all__ = [
    "ClassifierService",
    "EmbeddedPairDataset",
    "EvaluationMetrics",
    "PairClassifier",
    "PairRecord",
    "TrainingConfig",
    "TrainingResult",
    "build_pair_features",
    "compute_binary_metrics",
    "evaluate_model",
    "load_pair_records",
    "train_classifier",
]
