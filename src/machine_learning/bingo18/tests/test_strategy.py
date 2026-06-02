"""Tests for strategy model and trainer."""

import numpy as np
import pytest

from machine_learning.bingo18.simulator import BetType
from machine_learning.bingo18.strategy_model import (
    ALL_BET_TYPES,
    BET_TYPE_TO_IDX,
    N_ACTIONS,
    N_BET_TYPES,
    SKIP_ACTION,
    ContextBuilder,
    StrategyModel,
)


class TestContextBuilder:
    def test_context_dim(self):
        builder = ContextBuilder(n_features=31, n_digits=6, n_bet_types=10)
        assert builder.context_dim == 31 + 6 + 3 + 10 + 1

    def test_build_returns_correct_shape(self):
        builder = ContextBuilder(n_features=31, n_digits=6, n_bet_types=10)
        draw_features = np.zeros(31)
        digit_probs = {1: 0.1, 2: 0.2, 3: 0.15, 4: 0.25, 5: 0.15, 6: 0.15}
        ctx = builder.build(
            draw_features=draw_features,
            digit_probs=digit_probs,
            budget_ratio=1.0,
            win_streak=0,
            loss_streak=0,
            bet_type_rois={bt.value: 0.0 for bt in ALL_BET_TYPES},
        )
        assert ctx.shape == (builder.context_dim,)
        assert ctx.dtype == np.float32

    def test_build_clips_budget_ratio(self):
        builder = ContextBuilder(n_features=31, n_digits=6, n_bet_types=10)
        ctx = builder.build(
            draw_features=np.zeros(31),
            digit_probs={d: 1 / 6 for d in range(1, 7)},
            budget_ratio=100.0,
            win_streak=0,
            loss_streak=0,
            bet_type_rois={bt.value: 0.0 for bt in ALL_BET_TYPES},
        )
        offset = 31 + 6
        assert ctx[offset] == 10.0  # capped at 10


class TestStrategyModel:
    def test_predict_probs_untrained(self):
        model = StrategyModel(context_dim=45)
        ctx = np.zeros(45)
        probs = model.predict_probs(ctx)
        assert probs.shape == (N_ACTIONS,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.allclose(probs, 1.0 / N_ACTIONS)

    def test_select_bet_type_returns_bet_type_or_none(self):
        model = StrategyModel(context_dim=45)
        ctx = np.zeros(45)
        bt = model.select_bet_type(ctx)
        # Untrained model may skip or return a bet type
        assert bt is None or isinstance(bt, BetType)

    def test_should_skip_untrained(self):
        model = StrategyModel(context_dim=45, skip_threshold=0.05)
        ctx = np.zeros(45)
        # With low threshold, uniform skip prob (1/11) should trigger
        assert model.should_skip(ctx)

    def test_select_top_n_returns_correct_count_or_empty(self):
        model = StrategyModel(context_dim=45)
        ctx = np.zeros(45)
        types = model.select_top_n(ctx, n=3)
        # Untrained model may skip (return empty) or return bet types
        assert len(types) <= 3
        for bt in types:
            assert isinstance(bt, BetType)

    def test_train_and_predict(self):
        model = StrategyModel(context_dim=45)
        # Create dummy training data with skip actions included
        n_samples = 200
        contexts = np.random.randn(n_samples, 45).astype(np.float32)
        actions = np.random.randint(0, N_ACTIONS, n_samples)  # includes skip
        rewards = np.random.randn(n_samples)

        metrics = model.train(contexts, actions, rewards, n_epochs=5)
        assert metrics["n_samples"] == n_samples
        assert model.is_trained

        # Predict should return valid distribution
        probs = model.predict_probs(contexts[0])
        assert probs.shape == (N_ACTIONS,)
        assert np.isclose(probs.sum(), 1.0, atol=0.01)

    def test_save_and_load(self, tmp_path):
        model = StrategyModel(context_dim=45)
        # Train on dummy data
        contexts = np.random.randn(100, 45).astype(np.float32)
        actions = np.random.randint(0, N_BET_TYPES, 100)
        rewards = np.random.randn(100)
        model.train(contexts, actions, rewards, n_epochs=3)

        # Save
        path = tmp_path / "strategy.pkl"
        model.save(path)
        assert path.exists()

        # Load into new model
        model2 = StrategyModel(context_dim=45)
        model2.load(path)
        assert model2.is_trained

        # Predictions should match
        ctx = np.random.randn(45).astype(np.float32)
        p1 = model.predict_probs(ctx)
        p2 = model2.predict_probs(ctx)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_bet_type_mapping(self):
        assert len(BET_TYPE_TO_IDX) == N_BET_TYPES
        for bt in BetType:
            assert bt in BET_TYPE_TO_IDX
            assert 0 <= BET_TYPE_TO_IDX[bt] < N_BET_TYPES
