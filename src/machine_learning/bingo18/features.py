"""Feature engineering for Bingo18 lottery prediction."""

import numpy as np
import pandas as pd
from loguru import logger


class Bingo18FeatureEngineer:
    """Extract features from Bingo18 draw history for ML prediction.

    For each draw at index i, features are computed from draws [i-window, i).
    Target is a binary vector indicating which digits appeared.
    """

    def __init__(self, window: int = 30, min_digit: int = 1, max_digit: int = 6):
        self.window = window
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.digits = list(range(min_digit, max_digit + 1))
        self.n_digits = len(self.digits)

    def build_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build feature matrix and target from Bingo18 DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: 'result' (list of 3 ints), 'total', 'large_small'.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples, n_digits) — binary matrix of digit presence
        feature_names : list[str]
        """
        results = df["result"].tolist()
        totals = df["total"].values
        large_small = df["large_small"].values
        n = len(results)

        if n <= self.window:
            raise ValueError(f"Need at least {self.window + 1} draws, got {n}")

        feature_names = self._feature_names()
        n_features = len(feature_names)
        n_samples = n - self.window

        X = np.zeros((n_samples, n_features), dtype=np.float32)
        y = np.zeros((n_samples, self.n_digits), dtype=np.float32)

        for i in range(n_samples):
            idx = i + self.window
            window_results = results[i:idx]
            window_totals = totals[i:idx]
            window_ls = large_small[i:idx]

            X[i] = self._extract_row(window_results, window_totals, window_ls)

            target_digits = set(results[idx])
            for j, d in enumerate(self.digits):
                if d in target_digits:
                    y[i, j] = 1.0

        logger.info(f"Built features: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names

    def build_features_for_predict(
        self, recent_draws: list[list[int]], recent_totals: list[int], recent_ls: list[str]
    ) -> np.ndarray:
        """Build features for a single prediction from recent draw history."""
        if len(recent_draws) < self.window:
            raise ValueError(f"Need at least {self.window} recent draws, got {len(recent_draws)}")

        window_results = recent_draws[-self.window :]
        window_totals = np.array(recent_totals[-self.window :])
        window_ls = recent_ls[-self.window :]

        row = self._extract_row(window_results, window_totals, window_ls)
        return row.reshape(1, -1)

    def _extract_row(self, window_results: list, window_totals: np.ndarray, window_ls: list) -> np.ndarray:
        """Extract feature vector from a window of draws."""
        features = []
        w = len(window_results)

        # 1. Frequency of each digit (normalized)
        freq = np.zeros(self.n_digits)
        for draw in window_results:
            for d in draw:
                if self.min_digit <= d <= self.max_digit:
                    freq[d - self.min_digit] += 1
        freq /= w * 3
        features.extend(freq)

        # 2. Gap: draws since last appearance of each digit
        gaps = np.full(self.n_digits, w, dtype=np.float32)
        for i in range(w - 1, -1, -1):
            for d in window_results[i]:
                if self.min_digit <= d <= self.max_digit:
                    idx = d - self.min_digit
                    if gaps[idx] == w:
                        gaps[idx] = w - 1 - i
        features.extend(gaps)

        # 3. Sum statistics
        features.append(np.mean(window_totals))
        features.append(np.std(window_totals) if len(window_totals) > 1 else 0.0)

        # 4. Last 3 draws flattened (pad with -1 if fewer)
        last_draws = []
        for i in range(max(0, w - 3), w):
            last_draws.extend(window_results[i])
        while len(last_draws) < 9:
            last_draws.insert(0, -1)
        features.extend(last_draws)

        # 5. Odd/even ratio in last draw
        last = window_results[-1]
        odd_count = sum(1 for d in last if d % 2 == 1)
        features.append(odd_count / 3.0)
        features.append((3 - odd_count) / 3.0)

        # 6. Big/Small ratio
        big_count = sum(1 for ls in window_ls if "Lớn" in str(ls))
        features.append(big_count / w)

        # 7. Current streak
        streak_big = 0
        streak_small = 0
        for i in range(w - 1, -1, -1):
            if "Lớn" in str(window_ls[i]):
                if streak_small == 0:
                    streak_big += 1
                else:
                    break
            elif "Nhỏ" in str(window_ls[i]):
                if streak_big == 0:
                    streak_small += 1
                else:
                    break
            else:
                break
        features.append(streak_big)
        features.append(streak_small)

        # 8. Pair/Triple statistics in window
        pair_count = 0
        triple_count = 0
        for draw in window_results:
            counts = {}
            for d in draw:
                counts[d] = counts.get(d, 0) + 1
            max_same = max(counts.values()) if counts else 0
            if max_same >= 2:
                pair_count += 1
            if max_same >= 3:
                triple_count += 1
        features.append(pair_count / w)
        features.append(triple_count / w)

        # 9. Max same digits in last draw
        last = window_results[-1]
        last_counts = {}
        for d in last:
            last_counts[d] = last_counts.get(d, 0) + 1
        features.append(max(last_counts.values()) if last_counts else 0)

        return np.array(features, dtype=np.float32)

    def _feature_names(self) -> list[str]:
        """Return ordered feature names."""
        names = []
        names.extend([f"freq_{d}" for d in self.digits])
        names.extend([f"gap_{d}" for d in self.digits])
        names.extend(["sum_mean", "sum_std"])
        names.extend([f"last_draw_{i}" for i in range(9)])
        names.extend(["odd_ratio", "even_ratio"])
        names.append("big_ratio")
        names.extend(["streak_big", "streak_small"])
        names.extend(["pair_ratio", "triple_ratio", "max_same_last"])
        return names
