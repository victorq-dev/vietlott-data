"""AdaptiveAgent system for Bingo18 ML betting simulator.

Provides an adaptive betting agent that:
- Selects bet types based on weighted probabilities
- Adjusts strategy based on performance (ROI, win rate, streaks)
- Manages budget with risk-aware bet sizing
- Supports diverse population of agents for ensemble simulation
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.simulator import (
    CONG_TONG_MULTIPLIER,
    CONG_TONG_PRIZE,
    HAI_SO_TRUNG_PRIZE,
    LON_HOA_NHO_PRIZE,
    LON_HOA_NHO_V2_MULTIPLIER,
    MOT_SO_PRIZE,
    TRUNG_2SO_MULTIPLIER,
    TRUNG_3SO_ANY_MULTIPLIER,
    TRUNG_3SO_MULTIPLIER,
    BetRecord,
    BetType,
    SimulationResult,
    calculate_payout,
)

# All available bet types for random exploration
ALL_BET_TYPES = list(BetType)

# Default weights for each bet type
DEFAULT_BET_TYPE_WEIGHTS: dict[str, float] = {
    "mot_so": 1.0,
    "hai_so_trung": 1.0,
    "ba_so_trung": 1.0,
    "cong_tong": 1.0,
    "lon_hoa_nho": 1.0,
    "cong_tong_mult": 1.0,
    "lon_hoa_nho_v2": 1.0,
    "trung_2so": 1.0,
    "trung_3so": 1.0,
    "trung_3so_any": 1.0,
}

# Weight clamp bounds
WEIGHT_MIN = 0.1
WEIGHT_MAX = 5.0

# Bet fraction clamp bounds
BET_FRACTION_MIN = 0.005
BET_FRACTION_MAX = 0.10


@dataclass(frozen=True)
class AgentGenome:
    """Immutable configuration for an AdaptiveAgent.

    Defines the agent's model parameters, risk profile, and adaptation behavior.
    """

    algorithm: str = "gradient_boosting"
    window: int = 30
    n_estimators: int = 100
    max_depth: int = 3
    bet_type_weights: tuple[tuple[str, float], ...] = ()
    risk_profile: str = "moderate"  # conservative|moderate|aggressive
    base_bet_fraction: float = 0.02
    adaptation_rate: float = 0.3
    exploration_rate: float = 0.10
    streak_sensitivity: float = 0.5
    min_bets_before_abandon: int = 10
    primary_strategy: str = "top_n"
    threshold: float = 0.12
    top_n: int = 1
    max_bets_per_draw: int = 3
    multi_bet_budget_share: float = 0.06


class BetTypeStats:
    """Mutable statistics tracker for a single bet type.

    Tracks wins, losses, ROI, and recent performance.
    """

    __slots__ = (
        "bet_type",
        "total_bets",
        "wins",
        "total_wagered",
        "total_payout",
        "recent_wins",
    )

    def __init__(self, bet_type: str) -> None:
        self.bet_type = bet_type
        self.total_bets: int = 0
        self.wins: int = 0
        self.total_wagered: int = 0
        self.total_payout: int = 0
        self.recent_wins: list[bool] = []

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.wins / self.total_bets

    @property
    def roi(self) -> float:
        if self.total_wagered == 0:
            return 0.0
        return ((self.total_payout - self.total_wagered) / self.total_wagered) * 100

    @property
    def recent_win_rate(self) -> float:
        if not self.recent_wins:
            return 0.0
        return sum(1 for w in self.recent_wins if w) / len(self.recent_wins)

    @property
    def expected_value(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return (self.total_payout - self.total_wagered) / self.total_bets

    def record(self, won: bool, wagered: int, payout: int) -> None:
        """Record a bet outcome."""
        self.total_bets += 1
        if won:
            self.wins += 1
        self.total_wagered += wagered
        self.total_payout += payout
        self.recent_wins.append(won)
        # Cap at 20 recent results
        if len(self.recent_wins) > 20:
            self.recent_wins = self.recent_wins[-20:]


class AdaptiveAgent:
    """Adaptive betting agent for Bingo18.

    Manages budget, selects bets using weighted random selection,
    adapts strategy based on performance, and tracks full history.
    """

    def __init__(
        self,
        agent_id: str,
        genome: AgentGenome,
        model: Bingo18Model,
        budget: int,
        bet_size: int = 10_000,
        adaptation_interval: int = 50,
        strategy_model: Any = None,
    ) -> None:
        self.agent_id = agent_id
        self.genome = genome
        self.model = model
        self.budget = budget
        self.bet_size = bet_size
        self.adaptation_interval = adaptation_interval
        self._strategy_model = strategy_model

        # Initialize mutable state
        self._starting_budget: int = budget
        self._bet_type_weights: dict[str, float] = dict(DEFAULT_BET_TYPE_WEIGHTS)
        # Override with genome weights if provided
        for bt_name, weight in genome.bet_type_weights:
            self._bet_type_weights[bt_name] = weight

        self._bet_type_stats: dict[str, BetTypeStats] = {bt.value: BetTypeStats(bt.value) for bt in BetType}
        self._current_streak: int = 0
        self._total_bets: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._bet_history: list[BetRecord] = []
        self._profit_curve: list[int] = [budget]
        self._draws_since_adaptation: int = 0
        self._generation: int = 0
        self._adaptation_log: list[dict[str, Any]] = []
        self._max_budget: int = budget
        self._min_budget: int = budget
        self._max_drawdown: int = 0
        self._base_bet_fraction: float = genome.base_bet_fraction
        self._rng = np.random.default_rng()

    @property
    def is_alive(self) -> bool:
        """Agent is alive if budget covers at least one bet."""
        return self.budget >= self.bet_size

    @property
    def roi(self) -> float:
        """Return on investment as percentage."""
        if self._starting_budget == 0:
            return 0.0
        return ((self.budget - self._starting_budget) / self._starting_budget) * 100

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        if self._total_bets == 0:
            return 0.0
        return self._wins / self._total_bets

    @property
    def max_drawdown(self) -> int:
        """Maximum drawdown from peak budget."""
        return self._max_drawdown

    def _calibrate_predictions(self, predictions: dict[int, float]) -> dict[int, float]:
        """Calibrate model predictions toward uniform to avoid false positive EV.

        Blends model predictions with uniform distribution (1/6 each).
        This prevents the model from creating overconfident predictions that
        appear to have positive EV when they don't.

        Parameters
        ----------
        predictions : dict[int, float]
            Raw model predictions

        Returns
        -------
        dict[int, float] : calibrated predictions
        """
        alpha = 1.0  # Use raw model predictions (no calibration)
        uniform = 1.0 / 6.0
        calibrated = {}
        for d in range(1, 7):
            raw = predictions.get(d, uniform)
            calibrated[d] = alpha * raw + (1 - alpha) * uniform
        return calibrated

    def _has_acceptable_ev_bet(self, predictions: dict[int, float], threshold: float = -0.30) -> bool:
        """Check if any bet type has EV above threshold.

        For fair 3d6, all bets have ~43-50% house edge (-0.43 to -0.50 EV).
        This gate skips draws where all bets are very negative EV, only
        allowing bets when model predictions suggest lower house edge.

        Parameters
        ----------
        predictions : dict[int, float]
            Digit probabilities
        threshold : float
            Minimum acceptable EV per unit bet (default -0.30 = 30% house edge)

        Returns
        -------
        bool : True if at least one bet type has EV > threshold
        """
        from machine_learning.bingo18.dice_probs import (
            compute_mot_so_ev, compute_cong_tong_ev, compute_lon_hoa_nho_ev,
        )

        # Check MOT_SO (digit bets) - per 10k bet, normalize to per-unit
        for d in range(1, 7):
            ev = compute_mot_so_ev(predictions, d)
            if ev / 10_000 > threshold:
                return True

        # Check CONG_TONG (sum bets)
        for t in range(3, 19):
            ev = compute_cong_tong_ev(predictions, t)
            if ev / 10_000 > threshold:
                return True

        # Check LON_HOA_NHO (category bets)
        for cat in ['Nhỏ', 'Hòa', 'Lớn']:
            ev = compute_lon_hoa_nho_ev(predictions, cat)
            if ev / 10_000 > threshold:
                return True

        return False

    def _best_ev_score(self, predictions: dict[int, float]) -> float:
        """Return the best (least negative) EV among all bet types.

        Used to decide skip probability: bet more when EV is less negative.
        Returns EV per unit bet (e.g., -0.43 for MOT_SO with fair dice).
        """
        from machine_learning.bingo18.dice_probs import (
            compute_mot_so_ev, compute_cong_tong_ev, compute_lon_hoa_nho_ev,
        )

        best_ev = -1.0

        for d in range(1, 7):
            ev = compute_mot_so_ev(predictions, d) / 10_000
            best_ev = max(best_ev, ev)

        for t in range(3, 19):
            ev = compute_cong_tong_ev(predictions, t) / 10_000
            best_ev = max(best_ev, ev)

        for cat in ['Nhỏ', 'Hòa', 'Lớn']:
            ev = compute_lon_hoa_nho_ev(predictions, cat) / 10_000
            best_ev = max(best_ev, ev)

        return best_ev

    def decide_bets(self, X: np.ndarray, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Decide which bets to place for the current draw.

        Uses EV-based skip probability: skip more when all bets have
        very negative EV, skip less when model predictions suggest
        lower house edge.
        """
        if not self.is_alive:
            return []

        if not predictions:
            return []

        # Store features for strategy model context
        self._last_features = X

        # Strategy model: learned policy for bet type selection
        if self._strategy_model is not None and self._strategy_model.is_trained:
            return self._strategy_bet(X, predictions)

        # Calculate bet amount based on budget, streak, and health
        bet_amount = _calculate_bet_amount(self)
        if bet_amount < self.bet_size or bet_amount > self.budget:
            return []

        # Exploration: random bet with exploration_rate probability (bypasses EV skip)
        if self._rng.random() < self.genome.exploration_rate:
            return self._random_bet(bet_amount, predictions)

        # EV-based skip: skip more when best EV is worse
        best_ev = self._best_ev_score(predictions)

        if best_ev > 0:
            # Positive EV: never skip
            skip_rate = 0.0
        else:
            # Map EV range [-0.50, 0] to skip rate [0.95, 0.40]
            clamped_ev = max(-0.50, min(0.0, best_ev))
            skip_rate = 0.95 + (clamped_ev + 0.50) * (0.40 - 0.95) / (0.0 + 0.50)
            skip_rate = max(0.40, min(0.95, skip_rate))

            # Additional skip when budget is low
            budget_ratio = self.budget / self._starting_budget if self._starting_budget > 0 else 1.0
            if budget_ratio < 0.25:
                skip_rate = max(skip_rate, 0.97)
            elif budget_ratio < 0.5:
                skip_rate = max(skip_rate, 0.85)

        if self._rng.random() < skip_rate:
            return []

        # Multi-bet mode: EV-driven selection of multiple bet types
        if self.genome.max_bets_per_draw > 1:
            return self._multi_bet(predictions)

        # Single-bet mode: weighted random selection
        return self._weighted_bet(bet_amount, predictions)

    def _weighted_bet(self, bet_amount: int, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Select bet using weighted random selection based on bet_type_weights.

        In a negative EV game, picks the best available bet type using weights.
        """
        # Filter to bet types that have weight > 0
        active_types = [
            (bt, self._bet_type_weights.get(bt.value, 1.0))
            for bt in BetType
            if self._bet_type_weights.get(bt.value, 1.0) > 0
        ]

        if not active_types:
            return []

        # Weighted random selection
        types, weights = zip(*active_types)
        weights_array = np.array(weights, dtype=np.float64)
        weights_array /= weights_array.sum()

        chosen_idx = self._rng.choice(len(types), p=weights_array)
        chosen_type = types[chosen_idx]

        # Select bet value based on predictions
        bet_value = self._select_bet_value(chosen_type, predictions)

        return [(chosen_type, bet_value, bet_amount)]

    def _random_bet(self, bet_amount: int, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Place a random exploratory bet. Skips when budget is low."""
        # Skip exploration when budget is critical
        if self._starting_budget > 0 and self.budget / self._starting_budget < 0.25:
            return []

        chosen_type = ALL_BET_TYPES[self._rng.integers(len(ALL_BET_TYPES))]
        bet_value = self._select_bet_value(chosen_type, predictions)
        return [(chosen_type, bet_value, bet_amount)]

    def _score_bets_for_agent(self, predictions: dict[int, float]) -> list[tuple[float, BetType, Any]]:
        """Score all bet type + value combinations by expected value.

        Adapted from Bingo18Simulator._score_bets() for agent use.
        Returns list of (ev, BetType, bet_value) sorted descending.
        """
        scored: list[tuple[float, BetType, Any]] = []
        cat_probs_cache: dict[str, float] | None = None

        for bt in BetType:
            if bt in (BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG):
                for d, p in predictions.items():
                    ev = self._ev_digit(bt, d, p)
                    scored.append((ev, bt, d))

            elif bt == BetType.CONG_TONG:
                for t in range(3, 19):
                    p = self._estimate_total_prob(predictions, t)
                    ev = p * CONG_TONG_PRIZE.get(t, 0) / 10_000
                    scored.append((ev, bt, t))

            elif bt == BetType.CONG_TONG_MULT:
                for t in range(3, 19):
                    p = self._estimate_total_prob(predictions, t)
                    ev = p * CONG_TONG_MULTIPLIER.get(t, 0)
                    scored.append((ev, bt, t))

            elif bt == BetType.LON_HOA_NHO:
                if cat_probs_cache is None:
                    cat_probs_cache = self._estimate_category_probs(predictions)
                for cat, p in cat_probs_cache.items():
                    ev = p * LON_HOA_NHO_PRIZE.get(cat, 0) / 10_000
                    scored.append((ev, bt, cat))

            elif bt == BetType.LON_HOA_NHO_V2:
                if cat_probs_cache is None:
                    cat_probs_cache = self._estimate_category_probs(predictions)
                for cat, p in cat_probs_cache.items():
                    ev = p * LON_HOA_NHO_V2_MULTIPLIER.get(cat, 0)
                    scored.append((ev, bt, cat))

            elif bt == BetType.TRUNG_2SO:
                for d in range(1, 7):
                    p = predictions.get(d, 0)
                    p_pair = 3 * p * p * (1 - p) + p * p * p
                    ev = p_pair * TRUNG_2SO_MULTIPLIER
                    scored.append((ev, bt, d))

            elif bt == BetType.TRUNG_3SO:
                for d in range(1, 7):
                    p = predictions.get(d, 0) ** 3
                    ev = p * TRUNG_3SO_MULTIPLIER
                    scored.append((ev, bt, d))

            elif bt == BetType.TRUNG_3SO_ANY:
                for d in range(1, 7):
                    p = predictions.get(d, 0) ** 3
                    ev = p * TRUNG_3SO_ANY_MULTIPLIER
                    scored.append((ev, bt, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    @staticmethod
    def _ev_digit(bt: BetType, _digit: int, p: float) -> float:
        """Estimate expected value for digit-based bets (per 10k bet)."""
        if bt == BetType.MOT_SO:
            # Exact EV: P(1)*12000 + P(2)*20000 + P(3)*30000 - 10000
            q = 1 - p
            p1 = 3 * p * q ** 2
            p2 = 3 * p ** 2 * q
            p3 = p ** 3
            return (p1 * 12_000 + p2 * 20_000 + p3 * 30_000) / 10_000 - 1.0
        elif bt == BetType.HAI_SO_TRUNG:
            p_at_least_2 = 3 * p * p * (1 - p) + p * p * p
            return p_at_least_2 * HAI_SO_TRUNG_PRIZE / 10_000
        elif bt == BetType.BA_SO_TRUNG:
            return p**3 * 1_200_000 / 10_000
        return 0.0

    @staticmethod
    def _estimate_total_prob(probs: dict[int, float], total: int) -> float:
        """Exact probability of total sum from digit probabilities via 3-fold convolution."""
        from machine_learning.bingo18.dice_probs import compute_total_probs

        total_probs = compute_total_probs(probs)
        return total_probs.get(total, 0.0)

    @staticmethod
    def _estimate_category_probs(probs: dict[int, float]) -> dict[str, float]:
        """Exact Nho/Hoa/Lon probabilities from digit probabilities via 3-fold convolution."""
        from machine_learning.bingo18.dice_probs import compute_category_probs

        return compute_category_probs(probs)

    def _multi_bet(self, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Select multiple bets using EV scoring with weight adjustment.

        Scores all bet types by EV, applies weight multiplier from adaptation,
        selects top-N positive-EV bets, and allocates budget proportionally.
        """
        scored = self._score_bets_for_agent(predictions)
        if not scored:
            return []

        # Apply weight multiplier from adaptation system
        weighted_scored = []
        for ev, bt, bv in scored:
            weight = self._bet_type_weights.get(bt.value, 1.0)
            weighted_ev = ev * weight
            weighted_scored.append((weighted_ev, bt, bv))

        # Select top-N positive-EV bets
        max_bets = self.genome.max_bets_per_draw
        selected = [(ev, bt, bv) for ev, bt, bv in weighted_scored if ev > 0][:max_bets]

        if not selected:
            return []

        # Calculate per-bet amount
        total_budget_share = min(
            int(self.budget * self.genome.multi_bet_budget_share),
            self.budget,
        )
        per_bet = max(total_budget_share // len(selected), self.bet_size)
        per_bet = min(per_bet, self.budget)

        if per_bet < self.bet_size:
            return []

        # Ensure total doesn't exceed budget
        total_needed = per_bet * len(selected)
        if total_needed > self.budget:
            per_bet = self.budget // len(selected)
            if per_bet < self.bet_size:
                return []

        return [(bt, bv, per_bet) for _, bt, bv in selected]

    def _strategy_bet(self, X: np.ndarray, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Select bets using the learned strategy model with progressive risk management.

        Builds context from current state, gets bet type distribution from
        strategy model, selects top-N types, and allocates budget with
        Kelly-inspired sizing that scales with budget health.
        """
        from machine_learning.bingo18.strategy_model import BET_TYPE_TO_IDX

        # Build context
        ctx_builder = self._strategy_model.context_builder
        budget_ratio = self.budget / self._starting_budget if self._starting_budget > 0 else 1.0
        bet_type_rois = {
            bt_name: stats.roi for bt_name, stats in self._bet_type_stats.items()
        }

        # Use actual draw features from X
        draw_features = X.flatten() if X.ndim > 1 else X
        if len(draw_features) < ctx_builder.n_features:
            draw_features = np.pad(draw_features, (0, ctx_builder.n_features - len(draw_features)))
        draw_features = draw_features[: ctx_builder.n_features]

        context = ctx_builder.build(
            draw_features=draw_features,
            digit_probs=predictions,
            budget_ratio=budget_ratio,
            win_streak=max(0, self._current_streak),
            loss_streak=max(0, -self._current_streak),
            bet_type_rois=bet_type_rois,
        )

        # Select top-N bet types (model handles skip internally)
        max_bets = max(1, self.genome.max_bets_per_draw)
        selected_types = self._strategy_model.select_top_n(context, n=max_bets)

        if not selected_types:
            return []

        # Progressive risk management: scale bet size with budget health
        # Start reducing at 50% budget, minimum at 10%
        if budget_ratio < 0.1:
            risk_scale = 0.1  # minimum: bet smallest possible
        elif budget_ratio < 0.25:
            risk_scale = 0.25
        elif budget_ratio < 0.5:
            risk_scale = 0.5
        elif budget_ratio < 0.75:
            risk_scale = 0.75
        else:
            risk_scale = 1.0

        # Base allocation with risk scaling
        base_share = self.genome.multi_bet_budget_share * risk_scale
        total_budget_share = min(
            int(self.budget * base_share),
            self.budget,
        )
        per_bet = max(total_budget_share // len(selected_types), self.bet_size)
        per_bet = min(per_bet, self.budget)

        if per_bet < self.bet_size:
            return []

        total_needed = per_bet * len(selected_types)
        if total_needed > self.budget:
            per_bet = self.budget // len(selected_types)
            if per_bet < self.bet_size:
                return []

        # Select bet values
        bets = []
        for bt in selected_types:
            bv = self._select_bet_value(bt, predictions)
            bets.append((bt, bv, per_bet))

        return bets

    def _select_bet_value(self, bet_type: BetType, predictions: dict[int, float]) -> Any:
        """Select the best bet value for a given bet type using predictions."""
        digit_types = (
            BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG,
            BetType.TRUNG_3SO, BetType.TRUNG_2SO, BetType.TRUNG_3SO_ANY,
        )
        if bet_type in digit_types:
            # Pick the digit with highest probability
            if predictions:
                return max(predictions, key=lambda d: predictions[d])
            return 3  # Default middle digit

        elif bet_type in (BetType.CONG_TONG, BetType.CONG_TONG_MULT):
            # Estimate best total from digit probabilities
            if predictions:
                sorted_digits = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                top_digits = [d for d, _ in sorted_digits[:3]]
                avg_digit = sum(top_digits) / len(top_digits)
                estimated_total = int(round(avg_digit * 3))
                return max(3, min(18, estimated_total))
            return 10  # Default middle total

        elif bet_type in (BetType.LON_HOA_NHO, BetType.LON_HOA_NHO_V2):
            if predictions:
                low_prob = np.mean([predictions.get(d, 0) for d in [1, 2, 3]])
                high_prob = np.mean([predictions.get(d, 0) for d in [4, 5, 6]])
                if low_prob > high_prob * 1.2:
                    return "Nhỏ"
                elif high_prob > low_prob * 1.2:
                    return "Lớn"
                return "Hòa"
            return "Lớn"  # Default

        return None

    def record_result(
        self,
        bet_type: BetType,
        bet_value: Any,
        bet_amount: int,
        actual_digits: list[int],
        actual_total: int,
        date: str,
        draw_id: str,
    ) -> None:
        """Record the outcome of a bet.

        Updates budget, streak, statistics, and history.
        """
        matches, payout = calculate_payout(bet_type, bet_value, actual_digits, bet_amount)
        won = payout > 0

        # Update budget
        self.budget += payout - bet_amount

        # Update streak
        if won:
            if self._current_streak > 0:
                self._current_streak += 1
            else:
                self._current_streak = 1
            self._wins += 1
        else:
            if self._current_streak < 0:
                self._current_streak -= 1
            else:
                self._current_streak = -1
            self._losses += 1

        self._total_bets += 1

        # Update bet type stats
        self._bet_type_stats[bet_type.value].record(won=won, wagered=bet_amount, payout=payout)

        # Track budget extremes
        if self.budget > self._max_budget:
            self._max_budget = self.budget
        if self.budget < self._min_budget:
            self._min_budget = self.budget
        drawdown = self._max_budget - self.budget
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        # Append to profit curve
        self._profit_curve.append(self.budget)

        # Create bet record
        record = BetRecord(
            date=date,
            draw_id=draw_id,
            bet_type=bet_type.value,
            bet_value=bet_value,
            actual_digits=actual_digits,
            actual_total=actual_total,
            matches=matches,
            bet_amount=bet_amount,
            payout=payout,
            budget_after=self.budget,
        )
        self._bet_history.append(record)

        # Increment adaptation counter
        self._draws_since_adaptation += 1

    def maybe_adapt(self) -> bool:
        """Check if adaptation should run and perform it if needed.

        Three-phase adaptation:
        Phase 1: Adjust bet type weights based on ROI and recent win rate.
        Phase 2: Adjust budget parameters (base_bet_fraction).
        Phase 3: Inject exploration if stuck.

        Returns
        -------
        bool — True if adaptation was performed.
        """
        if self._draws_since_adaptation < self.adaptation_interval:
            return False

        self._draws_since_adaptation = 0
        self._generation += 1
        adaptation_rate = self.genome.adaptation_rate

        log_entry: dict[str, Any] = {"generation": self._generation, "changes": []}

        # Phase 1: Adjust bet type weights
        for bt_name, stats in self._bet_type_stats.items():
            if stats.total_bets < 5:
                continue  # Not enough data

            current_weight = self._bet_type_weights.get(bt_name, 1.0)
            new_weight = current_weight

            if stats.roi > 0 and stats.recent_win_rate > 0.3:
                # Good performance: increase weight
                new_weight = current_weight * (1 + adaptation_rate)
                log_entry["changes"].append(f"{bt_name}: weight {current_weight:.2f} -> {new_weight:.2f} (good ROI)")
            elif stats.roi < -20 and stats.recent_win_rate < 0.2:
                # Poor performance: decrease weight
                new_weight = current_weight * (1 - adaptation_rate)
                log_entry["changes"].append(f"{bt_name}: weight {current_weight:.2f} -> {new_weight:.2f} (bad ROI)")

            # Clamp weight
            new_weight = max(WEIGHT_MIN, min(WEIGHT_MAX, new_weight))
            self._bet_type_weights[bt_name] = new_weight

        # Phase 2: Adjust budget parameters
        budget_ratio = self.budget / self._starting_budget if self._starting_budget > 0 else 1.0
        old_fraction = self._base_bet_fraction

        if budget_ratio > 2.0:
            # Winning big: increase bet fraction by 20%
            self._base_bet_fraction = min(BET_FRACTION_MAX, self._base_bet_fraction * 1.2)
            log_entry["changes"].append(
                f"bet_fraction: {old_fraction:.4f} -> {self._base_bet_fraction:.4f} (high budget)"
            )
        elif budget_ratio < 0.3:
            # Losing badly: decrease bet fraction by 20%
            self._base_bet_fraction = max(BET_FRACTION_MIN, self._base_bet_fraction * 0.8)
            log_entry["changes"].append(
                f"bet_fraction: {old_fraction:.4f} -> {self._base_bet_fraction:.4f} (low budget)"
            )

        # Phase 3: Inject exploration if stuck (low win rate over many bets)
        if self._total_bets >= self.genome.min_bets_before_abandon:
            overall_wr = self.win_rate
            if overall_wr < 0.15 and self.genome.exploration_rate < 0.3:
                # Can't mutate frozen genome, but we can increase exploration
                # by adjusting weights to be more uniform
                for bt_name in self._bet_type_weights:
                    current = self._bet_type_weights[bt_name]
                    # Move weight toward 1.0 (uniform)
                    self._bet_type_weights[bt_name] = current + (1.0 - current) * 0.3
                log_entry["changes"].append("exploration_injection: uniformizing weights (stuck)")

        self._adaptation_log.append(log_entry)
        if log_entry["changes"]:
            changes_summary = "; ".join(log_entry["changes"][:5])
            extra = f" +{len(log_entry['changes']) - 5} more" if len(log_entry["changes"]) > 5 else ""
            logger.debug(
                f"[{self.agent_id}] Adapted (gen {self._generation}): {changes_summary}{extra} | "
                f"fraction={self._base_bet_fraction:.3f} budget={self.budget:,}"
            )

        return True

    def to_simulation_result(self) -> SimulationResult:
        """Convert agent state to SimulationResult for compatibility.

        Returns
        -------
        SimulationResult with full history.
        """
        return SimulationResult(
            starting_budget=self._starting_budget,
            final_budget=self.budget,
            bet_size=self.bet_size,
            bet_type="adaptive",
            total_bets=self._total_bets,
            wins=self._wins,
            losses=self._losses,
            max_budget=self._max_budget,
            min_budget=self._min_budget,
            max_drawdown=self._max_drawdown,
            bet_history=list(self._bet_history),
            profit_curve=list(self._profit_curve),
        )


def _calculate_bet_amount(agent: AdaptiveAgent) -> int:
    """Calculate bet amount with Kelly-inspired progressive sizing.

    Scales bet size based on budget health:
    - >150% budget: slight increase (120%)
    - 100-150%: normal (100%)
    - 50-100%: reduced (70%)
    - 25-50%: heavily reduced (40%)
    - <25%: minimum bets only (20%)
    """
    budget = min(agent.budget, 10**12)
    base = (budget * round(agent._base_bet_fraction * 10000)) // 10000

    # Streak multiplier
    streak_num = 100
    if agent._current_streak > 3:
        streak_num = 100 + int(min(agent._current_streak - 3, 10) * 5 * agent.genome.streak_sensitivity)
    elif agent._current_streak < -3:
        streak_num = 100 - int(min(abs(agent._current_streak) - 3, 10) * 5 * agent.genome.streak_sensitivity)

    # Progressive health multiplier based on budget vs starting
    health_num = 100
    if agent._starting_budget > 0:
        if budget > agent._starting_budget * 3 // 2:
            health_num = 120
        elif budget < agent._starting_budget // 4:
            health_num = 20  # critical: minimum bets
        elif budget < agent._starting_budget // 2:
            health_num = 40  # low: heavily reduced
        elif budget < agent._starting_budget * 3 // 4:
            health_num = 70  # moderate: reduced

    amount = (base * streak_num * health_num) // 10000
    amount = min(max(amount, agent.bet_size), agent.budget)
    return amount


def create_diverse_agents(
    model: Bingo18Model,
    budget: int,
    bet_size: int = 10_000,
    n_agents: int = 12,
    adaptation_interval: int = 50,
) -> list[AdaptiveAgent]:
    """Create a diverse population of AdaptiveAgents.

    Generates agents with varied combinations of algorithm, risk profile,
    strategy, exploration rate, and adaptation rate.

    Parameters
    ----------
    model : Bingo18Model
        Trained model to share across agents.
    budget : int
        Starting budget for each agent.
    bet_size : int
        Base bet size.
    n_agents : int
        Number of agents to create.
    adaptation_interval : int
        Draws between adaptation cycles.

    Returns
    -------
    list of AdaptiveAgent with diverse genomes.
    """
    algorithms = ["gradient_boosting", "random_forest", "extra_trees"]
    risk_profiles = ["conservative", "moderate", "aggressive"]
    strategies = ["top_n", "threshold"]
    exploration_rates = [0.05, 0.10, 0.15, 0.20]
    adaptation_rates = [0.2, 0.3, 0.4]
    base_bet_fractions = [0.01, 0.02, 0.03, 0.05]
    max_bets_options = [1, 2, 3, 5]
    budget_shares = [0.03, 0.06, 0.10]

    agents: list[AdaptiveAgent] = []
    rng = np.random.default_rng(42)

    for i in range(n_agents):
        algo = algorithms[i % len(algorithms)]
        risk = risk_profiles[i % len(risk_profiles)]
        strategy = strategies[i % len(strategies)]
        exploration = float(rng.choice(exploration_rates))
        adaptation = float(rng.choice(adaptation_rates))
        bet_fraction = float(rng.choice(base_bet_fractions))
        max_bets = int(rng.choice(max_bets_options))
        budget_share = float(rng.choice(budget_shares))

        genome = AgentGenome(
            algorithm=algo,
            window=model.window,
            n_estimators=100,
            max_depth=3,
            risk_profile=risk,
            base_bet_fraction=bet_fraction,
            adaptation_rate=adaptation,
            exploration_rate=exploration,
            primary_strategy=strategy,
            max_bets_per_draw=max_bets,
            multi_bet_budget_share=budget_share,
        )

        agent = AdaptiveAgent(
            agent_id=f"agent_{i:03d}",
            genome=genome,
            model=model,
            budget=budget,
            bet_size=bet_size,
            adaptation_interval=adaptation_interval,
        )
        agents.append(agent)

    return agents
