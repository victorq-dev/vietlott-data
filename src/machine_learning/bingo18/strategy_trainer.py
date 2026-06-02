"""Offline trainer for the Bingo18 strategy model.

Runs simulation on historical data, collects (context, action, reward) tuples,
and trains the StrategyModel using reward-weighted regression.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from machine_learning.bingo18.agent import (
    AdaptiveAgent,
    AgentGenome,
    BetTypeStats,
    _calculate_bet_amount,
)
from machine_learning.bingo18.features import Bingo18FeatureEngineer
from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.simulator import BetType, calculate_payout
from machine_learning.bingo18.strategy_model import BET_TYPE_TO_IDX, SKIP_ACTION, ContextBuilder, StrategyModel

ALL_BET_TYPES = list(BetType)


class StrategyTrainer:
    """Trains a StrategyModel via simulation on historical data.

    For each draw:
    1. Build context from features + agent state
    2. Let the current strategy model select a bet type
    3. Calculate payout (reward)
    4. Store (context, action, reward)
    5. Periodically retrain the model on collected experiences
    """

    def __init__(
        self,
        model: Bingo18Model,
        strategy_model: StrategyModel | None = None,
        budget: int = 500_000,
        bet_size: int = 10_000,
        n_epochs_per_update: int = 5,
        update_interval: int = 1000,
        exploration_rate: float = 0.3,
        random_state: int = 42,
    ):
        self.model = model
        self.strategy_model = strategy_model or StrategyModel(
            context_dim=model.feature_engineer.n_features + 6 + 3 + len(ALL_BET_TYPES) + 1,
        )
        self.budget = budget
        self.bet_size = bet_size
        self.n_epochs_per_update = n_epochs_per_update
        self.update_interval = update_interval
        self.exploration_rate = exploration_rate
        self.rng = np.random.default_rng(random_state)
        self._feature_engineer = model.feature_engineer
        self._context_builder = self.strategy_model.context_builder

    def train(
        self,
        df: pd.DataFrame,
        n_epochs: int = 10,
        test_ratio: float = 0.2,
    ) -> dict[str, Any]:
        """Train strategy model on historical data with walk-forward validation.

        Parameters
        ----------
        df : pd.DataFrame — historical draws with 'result', 'total', 'large_small'
        n_epochs : int — number of full passes through the data
        test_ratio : float — fraction of data for testing

        Returns
        -------
        dict with training metrics and test performance
        """
        results = df["result"].tolist()
        totals = df["total"].tolist()
        large_smalls = df["large_small"].tolist()
        n_draws = len(results)
        window = self.model.window

        # Split train/test
        split_idx = int(n_draws * (1 - test_ratio))
        logger.info(
            f"Strategy training: {n_draws} draws, "
            f"train={split_idx - window}, test={n_draws - split_idx}, "
            f"window={window}"
        )

        # Collectors for training data
        all_contexts: list[np.ndarray] = []
        all_actions: list[int] = []
        all_rewards: list[float] = []

        # Training metrics history
        history: list[dict[str, float]] = []

        for epoch in range(n_epochs):
            epoch_contexts: list[np.ndarray] = []
            epoch_actions: list[int] = []
            epoch_rewards: list[float] = []

            # Simulate walk-forward on training set
            budget = self.budget
            agent_state = {
                "budget": budget,
                "starting_budget": budget,
                "win_streak": 0,
                "loss_streak": 0,
                "bet_type_stats": {bt.value: BetTypeStats(bt.value) for bt in ALL_BET_TYPES},
            }

            for i in range(window, split_idx):
                # Build features
                recent_draws = results[i - window : i]
                recent_totals = totals[i - window : i]
                recent_ls = large_smalls[i - window : i]

                try:
                    X = self._feature_engineer.build_features_for_predict(
                        recent_draws, recent_totals, recent_ls
                    )
                except (ValueError, IndexError):
                    continue

                # Get digit probabilities from Stage 1 model
                digit_probs = self.model.predict_proba(X)
                draw_features = X.flatten()

                # Build context
                budget_ratio = budget / agent_state["starting_budget"]
                bet_type_rois = {
                    bt_name: stats.roi
                    for bt_name, stats in agent_state["bet_type_stats"].items()
                }
                context = self._context_builder.build(
                    draw_features=draw_features,
                    digit_probs=digit_probs,
                    budget_ratio=budget_ratio,
                    win_streak=agent_state["win_streak"],
                    loss_streak=agent_state["loss_streak"],
                    bet_type_rois=bet_type_rois,
                )

                # Select bet type (or skip)
                if self.rng.random() < self.exploration_rate:
                    # Explore: random bet type or skip
                    action = self.rng.integers(len(ALL_BET_TYPES) + 1)  # +1 for skip
                else:
                    # Exploit: use strategy model
                    bt = self.strategy_model.select_bet_type(context, rng=self.rng)
                    action = BET_TYPE_TO_IDX[bt] if bt is not None else SKIP_ACTION

                # Handle skip action
                if action == SKIP_ACTION:
                    epoch_contexts.append(context)
                    epoch_actions.append(SKIP_ACTION)
                    epoch_rewards.append(0.0)  # neutral reward for skipping
                    continue

                bt = ALL_BET_TYPES[action]

                # Select bet value (reuse existing logic)
                bet_value = self._select_bet_value(bt, digit_probs)

                # Calculate bet amount
                bet_amount = max(self.bet_size, int(budget * 0.02))
                bet_amount = min(bet_amount, budget)

                if budget < self.bet_size:
                    break  # bankrupt

                # Calculate reward
                actual_digits = results[i]
                actual_total = totals[i]
                matches, payout = calculate_payout(bt, bet_value, actual_digits, bet_amount)
                reward = payout - bet_amount
                budget += reward

                # Update agent state
                won = payout > 0
                agent_state["budget"] = budget
                stats = agent_state["bet_type_stats"][bt.value]
                stats.record(won, bet_amount, payout)

                if won:
                    agent_state["win_streak"] += 1
                    agent_state["loss_streak"] = 0
                else:
                    agent_state["loss_streak"] += 1
                    agent_state["win_streak"] = 0

                # Store experience
                action_idx = BET_TYPE_TO_IDX[bt]
                epoch_contexts.append(context)
                epoch_actions.append(action_idx)
                epoch_rewards.append(reward)

            # Periodically update strategy model
            if len(epoch_contexts) > 0:
                all_contexts.extend(epoch_contexts)
                all_actions.extend(epoch_actions)
                all_rewards.extend(epoch_rewards)

                if (epoch + 1) % 1 == 0 or epoch == n_epochs - 1:
                    contexts_arr = np.array(all_contexts)
                    actions_arr = np.array(all_actions)
                    rewards_arr = np.array(all_rewards)

                    metrics = self.strategy_model.train(
                        contexts=contexts_arr,
                        actions=actions_arr,
                        rewards=rewards_arr,
                        n_epochs=self.n_epochs_per_update,
                    )

                    # Epoch metrics
                    epoch_win_rate = sum(1 for r in epoch_rewards if r > 0) / len(epoch_rewards) if epoch_rewards else 0
                    epoch_roi = (sum(epoch_rewards) / (len(epoch_rewards) * self.bet_size) * 100) if epoch_rewards else 0

                    epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_win_rate": epoch_win_rate,
                        "train_roi": epoch_roi,
                        "train_budget": budget,
                        "model_accuracy": metrics["accuracy"],
                        "n_experiences": len(all_contexts),
                    }
                    history.append(epoch_metrics)
                    logger.info(
                        f"Epoch {epoch + 1}/{n_epochs}: "
                        f"win_rate={epoch_win_rate:.1%}, "
                        f"ROI={epoch_roi:.1f}%, "
                        f"budget={budget:,}, "
                        f"accuracy={metrics['accuracy']:.1%}"
                    )

        # Evaluate on test set
        test_metrics = self._evaluate(df, split_idx, n_draws)

        return {
            "history": history,
            "test": test_metrics,
            "n_train_experiences": len(all_contexts),
        }

    def _evaluate(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> dict[str, float]:
        """Evaluate strategy model on test set."""
        results = df["result"].tolist()
        totals = df["total"].tolist()
        large_smalls = df["large_small"].tolist()
        window = self.model.window

        budget = self.budget
        total_bets = 0
        total_wins = 0
        total_wagered = 0
        total_payout = 0

        for i in range(max(start_idx, window), end_idx):
            recent_draws = results[i - window : i]
            recent_totals = totals[i - window : i]
            recent_ls = large_smalls[i - window : i]

            try:
                X = self._feature_engineer.build_features_for_predict(
                    recent_draws, recent_totals, recent_ls
                )
            except (ValueError, IndexError):
                continue

            digit_probs = self.model.predict_proba(X)
            draw_features = X.flatten()

            context = self._context_builder.build(
                draw_features=draw_features,
                digit_probs=digit_probs,
                budget_ratio=budget / self.budget,
                win_streak=0,
                loss_streak=0,
                bet_type_rois={bt.value: 0.0 for bt in ALL_BET_TYPES},
            )

            bt = self.strategy_model.select_bet_type(context)
            if bt is None:
                continue  # skip this draw

            bet_value = self._select_bet_value(bt, digit_probs)
            bet_amount = max(self.bet_size, int(budget * 0.02))
            bet_amount = min(bet_amount, budget)

            if budget < self.bet_size:
                break

            actual_digits = results[i]
            matches, payout = calculate_payout(bt, bet_value, actual_digits, bet_amount)
            budget += payout - bet_amount

            total_bets += 1
            total_wagered += bet_amount
            total_payout += payout
            if payout > 0:
                total_wins += 1

        win_rate = total_wins / total_bets if total_bets > 0 else 0
        roi = ((total_payout - total_wagered) / total_wagered * 100) if total_wagered > 0 else 0

        return {
            "win_rate": win_rate,
            "roi": roi,
            "final_budget": budget,
            "total_bets": total_bets,
            "total_wagered": total_wagered,
            "total_payout": total_payout,
        }

    @staticmethod
    def _select_bet_value(bt: BetType, digit_probs: dict[int, float]) -> Any:
        """Select bet value for a bet type using digit probabilities."""
        digit_types = (
            BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG,
            BetType.TRUNG_3SO, BetType.TRUNG_2SO, BetType.TRUNG_3SO_ANY,
        )
        if bt in digit_types:
            return max(digit_probs, key=digit_probs.get)

        if bt in (BetType.CONG_TONG, BetType.CONG_TONG_MULT):
            # Estimate most likely total
            mean_digit = sum(d * p for d, p in digit_probs.items())
            return int(round(mean_digit * 3))

        if bt in (BetType.LON_HOA_NHO, BetType.LON_HOA_NHO_V2):
            low_prob = np.mean([digit_probs.get(d, 0) for d in [1, 2, 3]])
            high_prob = np.mean([digit_probs.get(d, 0) for d in [4, 5, 6]])
            if low_prob > high_prob:
                return "Nhỏ"
            elif high_prob > low_prob:
                return "Lớn"
            return "Hòa"

        return None
