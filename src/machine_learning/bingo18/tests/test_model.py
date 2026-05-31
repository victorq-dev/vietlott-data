"""Tests for Bingo18 ML model."""

import numpy as np
import pandas as pd
import pytest

from machine_learning.bingo18.model import ALGORITHMS, Bingo18Model


@pytest.fixture
def sample_df():
    """Generate synthetic Bingo18 data large enough for training."""
    np.random.seed(42)
    n = 500
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


def test_train_returns_metrics(sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    metrics = model.train(sample_df, test_ratio=0.2)

    assert metrics.train_size > 0
    assert metrics.test_size > 0
    assert metrics.window == 30
    assert len(metrics.per_digit) == 6
    assert metrics.avg_log_loss > 0
    assert metrics.avg_brier > 0


def test_predict_proba_returns_6_digits(sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)

    recent = sample_df["result"].tolist()[-30:]
    recent_totals = sample_df["total"].tolist()[-30:]
    recent_ls = sample_df["large_small"].tolist()[-30:]

    X = model.feature_engineer.build_features_for_predict(recent, recent_totals, recent_ls)
    probs = model.predict_proba(X)

    assert len(probs) == 6
    for d in range(1, 7):
        assert d in probs
        assert 0 <= probs[d] <= 1


def test_predict_top_n_returns_sorted(sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)

    recent = sample_df["result"].tolist()[-30:]
    recent_totals = sample_df["total"].tolist()[-30:]
    recent_ls = sample_df["large_small"].tolist()[-30:]

    X = model.feature_engineer.build_features_for_predict(recent, recent_totals, recent_ls)
    top = model.predict_top_n(X, n=3)

    assert len(top) == 3
    assert top == sorted(top)
    for d in top:
        assert 1 <= d <= 6


def test_untrained_model_raises():
    model = Bingo18Model(window=30)
    X = np.zeros((1, 28))
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict_proba(X)


def test_save_load(tmp_path, sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)

    path = tmp_path / "model.joblib"
    model.save(path)
    assert path.exists()

    model2 = Bingo18Model(window=30)
    model2.load(path)
    assert model2.is_trained

    recent = sample_df["result"].tolist()[-30:]
    recent_totals = sample_df["total"].tolist()[-30:]
    recent_ls = sample_df["large_small"].tolist()[-30:]

    X = model.feature_engineer.build_features_for_predict(recent, recent_totals, recent_ls)
    probs = model.predict_proba(X)
    assert len(probs) == 6


@pytest.mark.parametrize("algorithm", list(ALGORITHMS.keys()))
def test_all_algorithms_train(sample_df, algorithm):
    """Verify all supported algorithms can train successfully."""
    kwargs = {"window": 30, "algorithm": algorithm}
    if algorithm != "logistic_regression":
        kwargs["n_estimators"] = 10
        kwargs["max_depth"] = 2

    model = Bingo18Model(**kwargs)
    metrics = model.train(sample_df, test_ratio=0.2)

    assert metrics.algorithm == algorithm
    assert metrics.avg_log_loss > 0
    assert model.is_trained


def test_unknown_algorithm_raises():
    model = Bingo18Model(window=30, algorithm="unknown_algo")
    with pytest.raises(ValueError, match="Unknown algorithm"):
        model.train(pd.DataFrame({"result": [[1, 2, 3]] * 100, "total": [6] * 100, "large_small": ["Nhỏ"] * 100}))


def test_predict_total_proba(sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)

    recent = sample_df["result"].tolist()[-30:]
    recent_totals = sample_df["total"].tolist()[-30:]
    recent_ls = sample_df["large_small"].tolist()[-30:]
    X = model.feature_engineer.build_features_for_predict(recent, recent_totals, recent_ls)

    total_probs = model.predict_total_proba(X)
    assert len(total_probs) > 0
    assert all(3 <= t <= 18 for t in total_probs.keys())
    assert abs(sum(total_probs.values()) - 1.0) < 0.01


def test_predict_pair_proba(sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)

    recent = sample_df["result"].tolist()[-30:]
    recent_totals = sample_df["total"].tolist()[-30:]
    recent_ls = sample_df["large_small"].tolist()[-30:]
    X = model.feature_engineer.build_features_for_predict(recent, recent_totals, recent_ls)

    p_pair = model.predict_pair_proba(X)
    assert 0 <= p_pair <= 1


def test_predict_triple_proba(sample_df):
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)

    recent = sample_df["result"].tolist()[-30:]
    recent_totals = sample_df["total"].tolist()[-30:]
    recent_ls = sample_df["large_small"].tolist()[-30:]
    X = model.feature_engineer.build_features_for_predict(recent, recent_totals, recent_ls)

    p_triple = model.predict_triple_proba(X)
    assert 0 <= p_triple <= 1
