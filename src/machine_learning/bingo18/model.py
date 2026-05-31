"""ML model for Bingo18 digit prediction with multiple algorithm support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from machine_learning.bingo18.features import Bingo18FeatureEngineer

ALGORITHMS = {
    "gradient_boosting": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "logistic_regression": LogisticRegression,
}

# Default params per algorithm
ALGORITHM_DEFAULTS = {
    "gradient_boosting": {"n_estimators": 100, "max_depth": 3, "random_state": 42},
    "random_forest": {"n_estimators": 100, "max_depth": 3, "random_state": 42},
    "extra_trees": {"n_estimators": 100, "max_depth": 3, "random_state": 42},
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
}


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    train_size: int = 0
    test_size: int = 0
    window: int = 0
    algorithm: str = ""
    per_digit: dict[int, dict[str, float]] = field(default_factory=dict)
    avg_log_loss: float = 0.0
    avg_brier: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "window": self.window,
            "algorithm": self.algorithm,
            "avg_log_loss": round(self.avg_log_loss, 4),
            "avg_brier": round(self.avg_brier, 4),
            "per_digit": {str(k): v for k, v in self.per_digit.items()},
        }

    def summary(self) -> str:
        lines = [
            f"Algorithm: {self.algorithm}",
            f"Training samples: {self.train_size}",
            f"Test samples: {self.test_size}",
            f"Window size: {self.window}",
            f"Average log loss: {self.avg_log_loss:.4f}",
            f"Average Brier score: {self.avg_brier:.4f}",
        ]
        return "\n".join(lines)


class Bingo18Model:
    """Train and predict Bingo18 digits using binary classifiers.

    Supports multiple algorithms: gradient_boosting, random_forest,
    extra_trees, logistic_regression.
    """

    def __init__(
        self,
        window: int = 30,
        algorithm: str = "gradient_boosting",
        min_digit: int = 1,
        max_digit: int = 6,
        **algo_params,
    ):
        self.window = window
        self.algorithm = algorithm
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.digits = list(range(min_digit, max_digit + 1))
        self.feature_engineer = Bingo18FeatureEngineer(window=window, min_digit=min_digit, max_digit=max_digit)
        self.algo_params = algo_params
        self.classifiers: dict[int, Any] = {}
        self._trained = False

    def _create_classifier(self) -> Any:
        """Create a classifier instance based on algorithm."""
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Choose from: {list(ALGORITHMS.keys())}")

        cls = ALGORITHMS[self.algorithm]
        params = {**ALGORITHM_DEFAULTS.get(self.algorithm, {}), **self.algo_params}
        return cls(**params)

    def train(self, df: pd.DataFrame, test_ratio: float = 0.2) -> TrainingMetrics:
        """Train all digit classifiers on historical data."""
        logger.info(f"Building features with window={self.window}...")
        X, y, _feature_names = self.feature_engineer.build_features(df)

        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Training {self.algorithm} on {len(X_train)} samples, testing on {len(X_test)}...")

        metrics = TrainingMetrics(
            train_size=len(X_train),
            test_size=len(X_test),
            window=self.window,
            algorithm=self.algorithm,
        )

        for j, d in enumerate(self.digits):
            clf = self._create_classifier()
            clf.fit(X_train, y_train[:, j])
            self.classifiers[d] = clf

            y_pred = clf.predict_proba(X_test)[:, 1]
            ll = log_loss(y_test[:, j], y_pred)
            bs = brier_score_loss(y_test[:, j], y_pred)

            metrics.per_digit[d] = {
                "log_loss": round(ll, 4),
                "brier": round(bs, 4),
            }

        metrics.avg_log_loss = np.mean([v["log_loss"] for v in metrics.per_digit.values()])
        metrics.avg_brier = np.mean([v["brier"] for v in metrics.per_digit.values()])
        self._trained = True

        logger.info(f"Training complete. Avg log loss: {metrics.avg_log_loss:.4f}, Avg Brier: {metrics.avg_brier:.4f}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> dict[int, float]:
        """Predict probability of each digit appearing in the next draw."""
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        probs = {}
        for d in self.digits:
            probs[d] = self.classifiers[d].predict_proba(X)[0, 1]
        return probs

    def predict_top_n(self, X: np.ndarray, n: int = 3) -> list[int]:
        """Predict the N most probable digits."""
        probs = self.predict_proba(X)
        sorted_digits = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted([d for d, _ in sorted_digits[:n]])

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifiers": self.classifiers,
                "window": self.window,
                "algorithm": self.algorithm,
                "min_digit": self.min_digit,
                "max_digit": self.max_digit,
                "algo_params": self.algo_params,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        self.classifiers = data["classifiers"]
        self.window = data["window"]
        self.algorithm = data.get("algorithm", "gradient_boosting")
        self.min_digit = data.get("min_digit", 1)
        self.max_digit = data.get("max_digit", 6)
        self.algo_params = data.get("algo_params", {})
        self.digits = list(range(self.min_digit, self.max_digit + 1))
        self.feature_engineer = Bingo18FeatureEngineer(
            window=self.window, min_digit=self.min_digit, max_digit=self.max_digit
        )
        self._trained = True
        logger.info(f"Model loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self._trained
