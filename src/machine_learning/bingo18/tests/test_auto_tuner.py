"""Tests for Bingo18 auto-tuner."""

import numpy as np
import pandas as pd
import pytest

from machine_learning.bingo18.auto_tuner import Bingo18AutoTuner, TunerResult, TunerSummary, render_tuner_results


@pytest.fixture
def sample_df():
    """Generate synthetic Bingo18 data with digits 1-6."""
    np.random.seed(42)
    n = 300
    results = [sorted(np.random.choice(range(1, 7), 3, replace=True).tolist()) for _ in range(n)]
    totals = [sum(r) for r in results]
    large_small = ["Lớn" if t >= 12 else ("Hòa" if t >= 10 else "Nhỏ") for t in totals]
    return pd.DataFrame(
        {
            "date": [f"2025-01-{i:02d}" for i in range(1, n + 1)],
            "id": [f"{i:07d}" for i in range(1, n + 1)],
            "result": results,
            "total": totals,
            "large_small": large_small,
        }
    )


def test_search_space_generation():
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting", "random_forest"],
        windows=[10, 30],
        n_estimators=[50],
        max_depths=[3],
        bet_types=["mot_so", "cong_tong"],
        strategies=["top_n", "threshold"],
        thresholds=[0.12],
    )
    space = tuner._get_search_space()

    # 2 algos × 2 windows × 1 est × 1 depth × 2 bet_types × (1 top_n + 1 threshold) = 16
    assert len(space) == 16

    # Each combo should have required keys
    for combo in space:
        assert "model_params" in combo
        assert "bet_type" in combo
        assert "strategy" in combo
        assert "threshold" in combo


def test_logistic_regression_no_tree_params():
    tuner = Bingo18AutoTuner(
        algorithms=["logistic_regression"],
        windows=[10],
        n_estimators=[100],
        max_depths=[5],
        bet_types=["mot_so"],
        strategies=["top_n"],
    )
    space = tuner._get_search_space()

    assert len(space) == 1
    params = space[0]["model_params"]
    assert params["algorithm"] == "logistic_regression"
    assert "n_estimators" not in params
    assert "max_depth" not in params


def test_auto_tuner_runs(sample_df):
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting"],
        windows=[30],
        n_estimators=[10],
        max_depths=[3],
        bet_types=["mot_so"],
        strategies=["top_n"],
        budget_levels=[500_000],
    )
    summary = tuner.run(sample_df, top_k=3)

    assert isinstance(summary, TunerSummary)
    assert summary.total_combinations == 1
    assert summary.total_simulations == 1
    assert len(summary.best_by_budget) == 1
    assert 500_000 in summary.best_by_budget
    assert len(summary.top_results) <= 3


def test_auto_tuner_multiple_budgets(sample_df):
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting"],
        windows=[30],
        n_estimators=[10],
        max_depths=[3],
        bet_types=["mot_so"],
        strategies=["top_n"],
        budget_levels=[100_000, 500_000],
    )
    summary = tuner.run(sample_df, top_k=5)

    assert len(summary.budget_levels) == 2
    assert len(summary.best_by_budget) == 2


def test_auto_tuner_multiple_algorithms(sample_df):
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting", "random_forest"],
        windows=[30],
        n_estimators=[10],
        max_depths=[3],
        bet_types=["mot_so"],
        strategies=["top_n"],
        budget_levels=[500_000],
    )
    summary = tuner.run(sample_df, top_k=5)

    assert summary.total_combinations == 2
    algos_seen = {r.algorithm for r in summary.top_results}
    assert len(algos_seen) >= 1


def test_tuner_result_to_dict():
    result = TunerResult(
        algorithm="gradient_boosting",
        window=30,
        n_estimators=50,
        max_depth=3,
        bet_type="mot_so",
        strategy="top_n",
        threshold=0.12,
        budget=1_000_000,
        final_budget=500_000,
        roi=-50.0,
        bets_survived=100,
        win_rate=0.4,
        max_drawdown=500_000,
        total_bets=100,
    )
    d = result.to_dict()
    assert d["algorithm"] == "gradient_boosting"
    assert d["final_budget"] == 500_000


def test_tuner_summary_to_dict(sample_df):
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting"],
        windows=[30],
        n_estimators=[10],
        max_depths=[3],
        bet_types=["mot_so"],
        strategies=["top_n"],
        budget_levels=[500_000],
    )
    summary = tuner.run(sample_df, top_k=3)
    d = summary.to_dict()

    assert "total_combinations" in d
    assert "best_by_budget" in d
    assert "top_results" in d


def test_render_tuner_results(sample_df):
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting"],
        windows=[30],
        n_estimators=[10],
        max_depths=[3],
        bet_types=["mot_so"],
        strategies=["top_n"],
        budget_levels=[500_000],
    )
    summary = tuner.run(sample_df, top_k=3)
    report = render_tuner_results(summary)

    assert "Auto-Tuner Results" in report
    assert "Best Strategy per Budget" in report
    assert "gradient_boosting" in report


def test_auto_tuner_save(sample_df, tmp_path):
    tuner = Bingo18AutoTuner(
        algorithms=["gradient_boosting"],
        windows=[30],
        n_estimators=[10],
        max_depths=[3],
        bet_types=["mot_so"],
        strategies=["top_n"],
        budget_levels=[500_000],
    )
    summary = tuner.run_and_save(sample_df, save_dir=tmp_path, top_k=2)

    # Check files were created
    assert (tmp_path / "best.json").exists()
    model_files = list(tmp_path.glob("model_*.joblib"))
    assert len(model_files) > 0

    # Check model_path is set in results
    for r in summary.top_results:
        if r.model_path:
            assert "model_" in r.model_path
