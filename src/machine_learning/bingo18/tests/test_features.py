"""Tests for Bingo18 feature engineering."""

import numpy as np
import pandas as pd
import pytest

from machine_learning.bingo18.features import Bingo18FeatureEngineer


@pytest.fixture
def sample_df():
    """Generate synthetic Bingo18 data with digits 1-6."""
    np.random.seed(42)
    n = 200
    results = [sorted(np.random.choice(range(1, 7), 3, replace=True).tolist()) for _ in range(n)]
    totals = [sum(r) for r in results]
    large_small = ["Lớn" if t >= 10 else "Nhỏ" for t in totals]
    return pd.DataFrame(
        {
            "date": [f"2025-01-{i:02d}" for i in range(1, n + 1)],
            "result": results,
            "total": totals,
            "large_small": large_small,
        }
    )


def test_build_features_shape(sample_df):
    window = 30
    engineer = Bingo18FeatureEngineer(window=window)
    X, y, names = engineer.build_features(sample_df)

    assert X.shape[0] == len(sample_df) - window
    assert X.shape[1] == len(names)
    assert y.shape == (X.shape[0], 6)  # digits 1-6


def test_build_features_no_nan(sample_df):
    engineer = Bingo18FeatureEngineer(window=30)
    X, y, _ = engineer.build_features(sample_df)

    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))


def test_feature_names_count(sample_df):
    engineer = Bingo18FeatureEngineer(window=30)
    _, _, names = engineer.build_features(sample_df)

    # freq_1..6 + gap_1..6 + sum_mean + sum_std + last_draw_0..8 + odd_ratio + even_ratio + big_ratio + streak_big + streak_small
    expected = 6 + 6 + 2 + 9 + 2 + 1 + 2
    assert len(names) == expected


def test_target_is_binary(sample_df):
    engineer = Bingo18FeatureEngineer(window=30)
    _, y, _ = engineer.build_features(sample_df)

    unique_values = np.unique(y)
    assert set(unique_values).issubset({0.0, 1.0})


def test_target_digits_match_actual(sample_df):
    """Verify that target correctly reflects actual draw digits."""
    window = 30
    engineer = Bingo18FeatureEngineer(window=window)
    _, y, _ = engineer.build_features(sample_df)

    results = sample_df["result"].tolist()
    for i in range(len(y)):
        actual_digits = set(results[i + window])
        for j, d in enumerate(engineer.digits):
            assert y[i, j] == (1.0 if d in actual_digits else 0.0)


def test_frequency_features_in_range(sample_df):
    engineer = Bingo18FeatureEngineer(window=30)
    X, _, names = engineer.build_features(sample_df)

    freq_indices = [i for i, n in enumerate(names) if n.startswith("freq_")]
    for idx in freq_indices:
        col = X[:, idx]
        assert np.all(col >= 0)
        assert np.all(col <= 1)


def test_gap_features_non_negative(sample_df):
    engineer = Bingo18FeatureEngineer(window=30)
    X, _, names = engineer.build_features(sample_df)

    gap_indices = [i for i, n in enumerate(names) if n.startswith("gap_")]
    for idx in gap_indices:
        assert np.all(X[:, idx] >= 0)


def test_build_features_for_predict_shape(sample_df):
    engineer = Bingo18FeatureEngineer(window=30)
    recent_draws = sample_df["result"].tolist()[:30]
    recent_totals = sample_df["total"].tolist()[:30]
    recent_ls = sample_df["large_small"].tolist()[:30]

    X = engineer.build_features_for_predict(recent_draws, recent_totals, recent_ls)
    assert X.shape == (1, len(engineer._feature_names()))


def test_too_few_draws_raises():
    engineer = Bingo18FeatureEngineer(window=30)
    df = pd.DataFrame(
        {
            "result": [[1, 2, 3]] * 10,
            "total": [6] * 10,
            "large_small": ["Nhỏ"] * 10,
        }
    )
    with pytest.raises(ValueError, match="Need at least"):
        engineer.build_features(df)
