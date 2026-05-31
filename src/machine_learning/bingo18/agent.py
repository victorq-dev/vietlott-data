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
    ) -> None:
        self.agent_id = agent_id
        self.genome = genome
        self.model = model
        self.budget = budget
        self.bet_size = bet_size
        self.adaptation_interval = adaptation_interval

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

    def decide_bets(self, X: np.ndarray, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Decide which bets to place for the current draw.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction.
        predictions : dict[int, float]
            Digit probability predictions from the model.

        Returns
        -------
        list of (BetType, bet_value, bet_amount) tuples.
        """
        if not self.is_alive:
            return []

        if not predictions:
            return []

        # Calculate bet amount based on budget, streak, and health
        bet_amount = _calculate_bet_amount(self)
        if bet_amount < self.bet_size or bet_amount > self.budget:
            return []

        # Exploration: random bet with exploration_rate probability
        if self._rng.random() < self.genome.exploration_rate:
            return self._random_bet(bet_amount, predictions)

        # Weighted selection of bet type
        return self._weighted_bet(bet_amount, predictions)

    def _weighted_bet(self, bet_amount: int, predictions: dict[int, float]) -> list[tuple[BetType, Any, int]]:
        """Select bet using weighted random selection based on bet_type_weights."""
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
        """Place a random exploratory bet."""
        chosen_type = ALL_BET_TYPES[self._rng.integers(len(ALL_BET_TYPES))]
        bet_value = self._select_bet_value(chosen_type, predictions)
        return [(chosen_type, bet_value, bet_amount)]

    def _select_bet_value(self, bet_type: BetType, predictions: dict[int, float]) -> Any:
        """Select the best bet value for a given bet type using predictions."""
        if bet_type in (BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG, BetType.TRUNG_3SO):
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

        elif bet_type == BetType.TRUNG_2SO:
            return None  # No selection needed

        elif bet_type == BetType.TRUNG_3SO_ANY:
            return None  # No selection needed

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
            logger.debug(f"Agent {self.agent_id} adapted (gen {self._generation}): {len(log_entry['changes'])} changes")

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
    """Calculate bet amount considering streak and budget health.

    Parameters
    ----------
    agent : AdaptiveAgent

    Returns
    -------
    int — bet amount in VND, clamped to [bet_size, budget]
    """
    # Base amount from budget fraction (integer arithmetic to avoid overflow)
    budget = min(agent.budget, 10**12)  # Cap at 1 trillion VND
    base = (budget * round(agent._base_bet_fraction * 10000)) // 10000

    # Streak multiplier (integer: numerator/denominator)
    streak_num = 100  # numerator
    if agent._current_streak > 3:
        streak_num = 100 + int(min(agent._current_streak - 3, 10) * 5 * agent.genome.streak_sensitivity)
    elif agent._current_streak < -3:
        streak_num = 100 - int(min(abs(agent._current_streak) - 3, 10) * 5 * agent.genome.streak_sensitivity)

    # Health multiplier based on budget vs starting
    health_num = 100  # numerator
    if agent._starting_budget > 0:
        # Use integer comparison to avoid float division of huge ints
        if budget > agent._starting_budget * 3 // 2:
            health_num = 120
        elif budget < agent._starting_budget // 2:
            health_num = 70

    amount = (base * streak_num * health_num) // 10000

    # Clamp: at least bet_size, but never exceed budget
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

    agents: list[AdaptiveAgent] = []
    rng = np.random.default_rng(42)

    for i in range(n_agents):
        algo = algorithms[i % len(algorithms)]
        risk = risk_profiles[i % len(risk_profiles)]
        strategy = strategies[i % len(strategies)]
        exploration = float(rng.choice(exploration_rates))
        adaptation = float(rng.choice(adaptation_rates))
        bet_fraction = float(rng.choice(base_bet_fractions))

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
