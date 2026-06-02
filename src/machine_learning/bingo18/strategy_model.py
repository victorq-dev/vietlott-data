"""Strategy learning model for Bingo18 betting optimization.

Learns WHEN to use WHICH bet type in different situations,
optimizing for net profit/ROI via reward-weighted regression.

Not predicting outcomes — learning optimal betting policy.
"""

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.neural_network import MLPClassifier

from machine_learning.bingo18.simulator import BetType

ALL_BET_TYPES = list(BetType)
N_BET_TYPES = len(ALL_BET_TYPES)
BET_TYPE_TO_IDX = {bt: i for i, bt in enumerate(ALL_BET_TYPES)}
SKIP_ACTION = N_BET_TYPES  # Index for "don't bet" action
N_ACTIONS = N_BET_TYPES + 1  # +1 for skip


class ContextBuilder:
    """Builds context vector from draw features + agent state.

    Context vector (~45 dims):
    - 31 features from Bingo18FeatureEngineer
    - 6 digit probabilities from Stage 1 model
    - Budget ratio, win streak, loss streak
    - Per-bet-type recent ROI (10 values)
    - Model confidence (max_prob - min_prob)
    """

    def __init__(self, n_features: int = 31, n_digits: int = 6, n_bet_types: int = N_BET_TYPES):
        self.n_features = n_features
        self.n_digits = n_digits
        self.n_bet_types = n_bet_types
        self.context_dim = n_features + n_digits + 3 + n_bet_types + 1

    def build(
        self,
        draw_features: np.ndarray,
        digit_probs: dict[int, float],
        budget_ratio: float,
        win_streak: int,
        loss_streak: int,
        bet_type_rois: dict[str, float],
    ) -> np.ndarray:
        """Build context vector from all available information."""
        ctx = np.zeros(self.context_dim, dtype=np.float32)

        # 31 draw features
        ctx[: self.n_features] = draw_features[: self.n_features]

        # 6 digit probabilities
        offset = self.n_features
        for d in range(1, self.n_digits + 1):
            ctx[offset + d - 1] = digit_probs.get(d, 1.0 / self.n_digits)

        # Budget ratio, win streak, loss streak
        offset += self.n_digits
        ctx[offset] = min(budget_ratio, 10.0)  # cap at 10x
        ctx[offset + 1] = min(win_streak, 20) / 20.0
        ctx[offset + 2] = min(loss_streak, 20) / 20.0

        # Per-bet-type recent ROI (normalized)
        offset += 3
        for i, bt in enumerate(ALL_BET_TYPES):
            roi = bet_type_rois.get(bt.value, 0.0)
            ctx[offset + i] = np.clip(roi / 100.0, -1.0, 1.0)

        # Model confidence
        offset += self.n_bet_types
        probs = list(digit_probs.values())
        ctx[offset] = max(probs) - min(probs) if probs else 0.0

        return ctx


class StrategyModel:
    """MLP-based policy model that learns bet type selection.

    Maps context vector → probability distribution over 10 bet types.
    Trained via reward-weighted regression on simulation outcomes.
    """

    def __init__(
        self,
        context_dim: int = 51,
        hidden_sizes: tuple[int, ...] = (64, 32),
        learning_rate: float = 0.001,
        random_state: int = 42,
        skip_threshold: float = 0.15,
    ):
        self.context_dim = context_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.skip_threshold = skip_threshold
        self._context_builder = ContextBuilder()

        self._mlp = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            learning_rate="constant",
            learning_rate_init=learning_rate,
            max_iter=200,
            random_state=random_state,
            warm_start=True,
            alpha=1e-4,
        )
        self._fitted = False

    @property
    def context_builder(self) -> ContextBuilder:
        return self._context_builder

    def predict_probs(self, context: np.ndarray) -> np.ndarray:
        """Predict probability distribution over bet types (including skip).

        Parameters
        ----------
        context : np.ndarray of shape (context_dim,) or (1, context_dim)

        Returns
        -------
        np.ndarray of shape (N_ACTIONS,) — probability for each action
        (first N_BET_TYPES for bet types, last one for skip)
        """
        if not self._fitted:
            # Uniform distribution if not trained
            return np.ones(N_ACTIONS) / N_ACTIONS

        if context.ndim == 1:
            context = context.reshape(1, -1)

        try:
            proba = self._mlp.predict_proba(context)[0]
            # Map classes to full action array
            result = np.zeros(N_ACTIONS)
            for i, cls in enumerate(self._mlp.classes_):
                result[int(cls)] = proba[i]
            # Normalize
            total = result.sum()
            if total > 0:
                result /= total
            else:
                result = np.ones(N_ACTIONS) / N_ACTIONS
            return result
        except Exception:
            return np.ones(N_ACTIONS) / N_ACTIONS

    def should_skip(self, context: np.ndarray) -> bool:
        """Check if the model recommends skipping this draw.

        Returns True if the skip probability is higher than the threshold
        or higher than any individual bet type probability.
        """
        probs = self.predict_probs(context)
        skip_prob = probs[SKIP_ACTION]
        max_bet_prob = probs[:N_BET_TYPES].max()
        return skip_prob > self.skip_threshold or skip_prob > max_bet_prob

    def select_bet_type(self, context: np.ndarray, rng: np.random.Generator | None = None) -> BetType | None:
        """Select a bet type by sampling from predicted distribution.

        Parameters
        ----------
        context : np.ndarray — context vector
        rng : numpy random generator (optional)

        Returns
        -------
        BetType or None — selected bet type, None if should skip
        """
        if self.should_skip(context):
            return None

        if rng is None:
            rng = np.random.default_rng()

        probs = self.predict_probs(context)[:N_BET_TYPES]  # exclude skip
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

        idx = rng.choice(N_BET_TYPES, p=probs)
        return ALL_BET_TYPES[idx]

    def select_top_n(self, context: np.ndarray, n: int = 3) -> list[BetType]:
        """Select top-N bet types by predicted probability.

        Returns empty list if model recommends skipping.

        Parameters
        ----------
        context : np.ndarray — context vector
        n : int — number of bet types to select

        Returns
        -------
        list[BetType] — top-N bet types sorted by probability (descending),
                        empty list if should skip
        """
        if self.should_skip(context):
            return []

        probs = self.predict_probs(context)[:N_BET_TYPES]  # exclude skip
        top_indices = np.argsort(probs)[::-1][:n]
        return [ALL_BET_TYPES[i] for i in top_indices]

    def train(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Train model using reward-weighted regression.

        For each context, we have the bet type chosen (action) and the reward.
        We train a classifier to predict the best action given context,
        weighting samples by reward (higher reward = higher weight).

        Parameters
        ----------
        contexts : np.ndarray of shape (n_samples, context_dim)
        actions : np.ndarray of shape (n_samples,) — bet type indices
        rewards : np.ndarray of shape (n_samples,) — net profit per bet
        n_epochs : int — number of training epochs
        batch_size : int — mini-batch size

        Returns
        -------
        dict with training metrics
        """
        if len(contexts) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "n_samples": 0}

        # Normalize rewards to [0, 1] for sample weights
        min_r, max_r = rewards.min(), rewards.max()
        if max_r > min_r:
            weights = (rewards - min_r) / (max_r - min_r)
        else:
            weights = np.ones_like(rewards) / len(rewards)

        # Boost positive rewards
        weights = np.where(rewards > 0, weights * 2.0 + 0.1, weights + 0.01)
        weights /= weights.sum()

        # Subsample if too many samples (for speed), preserving reward weighting
        max_samples = 50000
        if len(contexts) > max_samples:
            indices = np.random.choice(len(contexts), max_samples, replace=False, p=weights)
            contexts = contexts[indices]
            actions = actions[indices]
            weights = weights[indices]
            weights /= weights.sum()

        # Train MLP with sample weights
        self._mlp.max_iter = n_epochs * 20  # increase iterations per epoch
        self._mlp.fit(contexts, actions, sample_weight=weights)
        self._fitted = True

        # Compute metrics
        predictions = self._mlp.predict(contexts)
        accuracy = np.mean(predictions == actions)

        return {
            "loss": float(self._mlp.loss_) if hasattr(self._mlp, "loss_") else 0.0,
            "accuracy": float(accuracy),
            "n_samples": len(contexts),
        }

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "mlp": self._mlp,
                "fitted": self._fitted,
                "context_dim": self.context_dim,
                "hidden_sizes": self.hidden_sizes,
                "learning_rate": self.learning_rate,
            },
            path,
        )
        logger.info(f"Strategy model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        self._mlp = data["mlp"]
        self._fitted = data["fitted"]
        self.context_dim = data["context_dim"]
        self.hidden_sizes = data["hidden_sizes"]
        self.learning_rate = data["learning_rate"]
        logger.info(f"Strategy model loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self._fitted
