"""Tests for AdaptiveAgent system for Bingo18 ML betting simulator."""

import numpy as np
import pytest

from machine_learning.bingo18.agent import (
    AdaptiveAgent,
    AgentGenome,
    BetTypeStats,
    _calculate_bet_amount,
    create_diverse_agents,
)
from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.simulator import BetType, SimulationResult


# --- Fixtures ---


@pytest.fixture
def default_genome():
    """Default AgentGenome for testing."""
    return AgentGenome()


@pytest.fixture
def mock_model():
    """A mock-like model that is not trained (used for agent creation tests)."""
    return Bingo18Model(window=30, n_estimators=10, max_depth=2)


@pytest.fixture
def sample_predictions():
    """Sample prediction dict for 6 digits."""
    return {1: 0.15, 2: 0.18, 3: 0.20, 4: 0.17, 5: 0.16, 6: 0.14}


@pytest.fixture
def sample_X():
    """Small synthetic feature matrix."""
    rng = np.random.default_rng(42)
    return rng.random((1, 25)).astype(np.float32)


# --- Tests for AgentGenome ---


class TestAgentGenome:
    def test_default_values(self):
        genome = AgentGenome()
        assert genome.algorithm == "gradient_boosting"
        assert genome.window == 30
        assert genome.n_estimators == 100
        assert genome.max_depth == 3
        assert genome.risk_profile == "moderate"
        assert genome.base_bet_fraction == 0.02
        assert genome.adaptation_rate == 0.3
        assert genome.exploration_rate == 0.10
        assert genome.streak_sensitivity == 0.5
        assert genome.min_bets_before_abandon == 10
        assert genome.primary_strategy == "top_n"
        assert genome.threshold == 0.12
        assert genome.top_n == 1

    def test_immutability(self):
        genome = AgentGenome()
        with pytest.raises(AttributeError):
            genome.window = 50  # type: ignore[misc]

    def test_custom_values(self):
        genome = AgentGenome(
            algorithm="random_forest",
            window=50,
            risk_profile="aggressive",
            base_bet_fraction=0.05,
        )
        assert genome.algorithm == "random_forest"
        assert genome.window == 50
        assert genome.risk_profile == "aggressive"
        assert genome.base_bet_fraction == 0.05

    def test_bet_type_weights_tuple(self):
        genome = AgentGenome(bet_type_weights=(("mot_so", 1.5), ("cong_tong", 0.8)))
        assert genome.bet_type_weights == (("mot_so", 1.5), ("cong_tong", 0.8))
        # Ensure it's truly immutable
        with pytest.raises(AttributeError):
            genome.bet_type_weights = (("lon_hoa_nho", 1.0),)  # type: ignore[misc]


# --- Tests for BetTypeStats ---


class TestBetTypeStats:
    def test_initial_state(self):
        stats = BetTypeStats(bet_type="mot_so")
        assert stats.total_bets == 0
        assert stats.wins == 0
        assert stats.total_wagered == 0
        assert stats.total_payout == 0
        assert stats.win_rate == 0.0
        assert stats.roi == 0.0

    def test_record_win(self):
        stats = BetTypeStats(bet_type="mot_so")
        stats.record(won=True, wagered=10_000, payout=12_000)
        assert stats.total_bets == 1
        assert stats.wins == 1
        assert stats.total_wagered == 10_000
        assert stats.total_payout == 12_000
        assert stats.win_rate == 1.0
        assert stats.recent_wins == [True]

    def test_record_loss(self):
        stats = BetTypeStats(bet_type="mot_so")
        stats.record(won=False, wagered=10_000, payout=0)
        assert stats.total_bets == 1
        assert stats.wins == 0
        assert stats.total_wagered == 10_000
        assert stats.total_payout == 0
        assert stats.win_rate == 0.0
        assert stats.recent_wins == [False]

    def test_roi_calculation(self):
        stats = BetTypeStats(bet_type="mot_so")
        stats.record(won=True, wagered=10_000, payout=15_000)
        stats.record(won=False, wagered=10_000, payout=0)
        # Total wagered: 20k, total payout: 15k, loss: 5k
        # ROI = (15000 - 20000) / 20000 * 100 = -25%
        assert stats.roi == pytest.approx(-25.0)

    def test_recent_win_rate_capped_at_20(self):
        stats = BetTypeStats(bet_type="mot_so")
        # Add 25 results: first 20 losses, then 5 wins
        for _ in range(20):
            stats.record(won=False, wagered=10_000, payout=0)
        for _ in range(5):
            stats.record(won=True, wagered=10_000, payout=12_000)
        # recent_wins should only contain last 20
        assert len(stats.recent_wins) == 20
        # Last 20 out of 25: 15 losses + 5 wins
        assert stats.recent_win_rate == pytest.approx(5 / 20)

    def test_recent_win_rate_empty(self):
        stats = BetTypeStats(bet_type="mot_so")
        assert stats.recent_win_rate == 0.0

    def test_expected_value(self):
        stats = BetTypeStats(bet_type="mot_so")
        stats.record(won=True, wagered=10_000, payout=12_000)
        stats.record(won=False, wagered=10_000, payout=0)
        # Total wagered: 20k, total payout: 12k
        # EV = (12000 - 20000) / 2 = -4000 per bet
        assert stats.expected_value == pytest.approx(-4000.0)

    def test_expected_value_no_bets(self):
        stats = BetTypeStats(bet_type="mot_so")
        assert stats.expected_value == 0.0


# --- Tests for AdaptiveAgent creation ---


class TestAdaptiveAgentCreation:
    def test_agent_creation(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_001",
            genome=genome,
            model=mock_model,
            budget=1_000_000,
        )
        assert agent.agent_id == "test_001"
        assert agent.budget == 1_000_000
        assert agent.bet_size == 10_000
        assert agent.is_alive is True

    def test_agent_initial_state(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_002", genome=genome, model=mock_model, budget=500_000)
        assert agent.roi == 0.0
        assert agent.win_rate == 0.0
        assert agent._total_bets == 0
        assert agent._wins == 0
        assert agent._losses == 0
        assert agent._current_streak == 0
        assert agent._generation == 0

    def test_agent_is_alive_with_budget(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_003", genome=genome, model=mock_model, budget=50_000, bet_size=10_000)
        assert agent.is_alive is True

    def test_agent_not_alive_budget_below_bet_size(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_004", genome=genome, model=mock_model, budget=5_000, bet_size=10_000)
        assert agent.is_alive is False

    def test_agent_initial_weights_from_genome(self, mock_model):
        genome = AgentGenome(bet_type_weights=(("mot_so", 2.0), ("cong_tong", 0.5)))
        agent = AdaptiveAgent(agent_id="test_005", genome=genome, model=mock_model, budget=1_000_000)
        # Internal weights should be initialized from genome
        assert agent._bet_type_weights["mot_so"] == 2.0
        assert agent._bet_type_weights["cong_tong"] == 0.5


# --- Tests for decide_bets ---


class TestDecideBets:
    def test_decide_bets_returns_list(self, mock_model, sample_X, sample_predictions):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_d1", genome=genome, model=mock_model, budget=1_000_000)
        bets = agent.decide_bets(sample_X, sample_predictions)
        assert isinstance(bets, list)
        assert len(bets) > 0

    def test_decide_bets_returns_valid_tuples(self, mock_model, sample_X, sample_predictions):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_d2", genome=genome, model=mock_model, budget=1_000_000)
        bets = agent.decide_bets(sample_X, sample_predictions)
        for bet in bets:
            assert len(bet) == 3
            bet_type, bet_value, bet_amount = bet
            assert isinstance(bet_type, BetType)
            assert bet_amount > 0

    def test_decide_bets_respects_budget(self, mock_model, sample_X, sample_predictions):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_d3", genome=genome, model=mock_model, budget=10_000, bet_size=10_000)
        bets = agent.decide_bets(sample_X, sample_predictions)
        total_wagered = sum(b[2] for b in bets)
        assert total_wagered <= agent.budget

    def test_decide_bets_no_bets_when_dead(self, mock_model, sample_X, sample_predictions):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_d4", genome=genome, model=mock_model, budget=0, bet_size=10_000)
        bets = agent.decide_bets(sample_X, sample_predictions)
        assert len(bets) == 0

    def test_decide_bets_with_zero_exploration(self, mock_model, sample_X, sample_predictions):
        genome = AgentGenome(exploration_rate=0.0)
        agent = AdaptiveAgent(agent_id="test_d5", genome=genome, model=mock_model, budget=1_000_000)
        bets = agent.decide_bets(sample_X, sample_predictions)
        assert len(bets) > 0


# --- Tests for record_result ---


class TestRecordResult:
    def test_record_win_updates_budget(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r1", genome=genome, model=mock_model, budget=1_000_000)
        initial_budget = agent.budget
        agent.record_result(
            bet_type=BetType.MOT_SO,
            bet_value=3,
            bet_amount=10_000,
            actual_digits=[3, 4, 5],
            actual_total=12,
            date="2025-01-01",
            draw_id="0000001",
        )
        assert agent._total_bets == 1
        assert agent._wins == 1
        assert agent._losses == 0
        assert agent.budget > initial_budget - 10_000  # Budget = initial - 10k + payout

    def test_record_loss_updates_budget(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r2", genome=genome, model=mock_model, budget=1_000_000)
        initial_budget = agent.budget
        agent.record_result(
            bet_type=BetType.MOT_SO,
            bet_value=3,
            bet_amount=10_000,
            actual_digits=[1, 2, 4],
            actual_total=7,
            date="2025-01-01",
            draw_id="0000001",
        )
        assert agent._total_bets == 1
        assert agent._wins == 0
        assert agent._losses == 1
        assert agent.budget == initial_budget - 10_000

    def test_record_result_appends_to_history(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r3", genome=genome, model=mock_model, budget=1_000_000)
        agent.record_result(
            bet_type=BetType.MOT_SO,
            bet_value=3,
            bet_amount=10_000,
            actual_digits=[3, 4, 5],
            actual_total=12,
            date="2025-01-01",
            draw_id="0000001",
        )
        assert len(agent._bet_history) == 1
        assert agent._bet_history[0].bet_type == "mot_so"

    def test_record_result_updates_streak_win(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r4", genome=genome, model=mock_model, budget=1_000_000)
        agent.record_result(
            bet_type=BetType.MOT_SO, bet_value=3, bet_amount=10_000,
            actual_digits=[3, 4, 5], actual_total=12, date="2025-01-01", draw_id="001",
        )
        assert agent._current_streak == 1

    def test_record_result_updates_streak_loss(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r5", genome=genome, model=mock_model, budget=1_000_000)
        agent.record_result(
            bet_type=BetType.MOT_SO, bet_value=3, bet_amount=10_000,
            actual_digits=[1, 2, 4], actual_total=7, date="2025-01-01", draw_id="001",
        )
        assert agent._current_streak == -1

    def test_record_result_resets_streak_on_switch(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r6", genome=genome, model=mock_model, budget=1_000_000)
        # Win
        agent.record_result(
            bet_type=BetType.MOT_SO, bet_value=3, bet_amount=10_000,
            actual_digits=[3, 4, 5], actual_total=12, date="2025-01-01", draw_id="001",
        )
        assert agent._current_streak == 1
        # Loss
        agent.record_result(
            bet_type=BetType.MOT_SO, bet_value=3, bet_amount=10_000,
            actual_digits=[1, 2, 4], actual_total=7, date="2025-01-02", draw_id="002",
        )
        assert agent._current_streak == -1

    def test_record_result_updates_bet_type_stats(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r7", genome=genome, model=mock_model, budget=1_000_000)
        agent.record_result(
            bet_type=BetType.MOT_SO, bet_value=3, bet_amount=10_000,
            actual_digits=[3, 4, 5], actual_total=12, date="2025-01-01", draw_id="001",
        )
        stats = agent._bet_type_stats["mot_so"]
        assert stats.total_bets == 1
        assert stats.wins == 1

    def test_record_result_updates_profit_curve(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_r8", genome=genome, model=mock_model, budget=1_000_000)
        agent.record_result(
            bet_type=BetType.MOT_SO, bet_value=3, bet_amount=10_000,
            actual_digits=[3, 4, 5], actual_total=12, date="2025-01-01", draw_id="001",
        )
        # Initial profit_curve = [budget], then one append after record_result
        assert len(agent._profit_curve) == 2
        assert agent._profit_curve[-1] == agent.budget


# --- Tests for maybe_adapt ---


class TestMaybeAdapt:
    def test_adapt_does_not_trigger_before_interval(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_a1", genome=genome, model=mock_model,
            budget=1_000_000, adaptation_interval=50,
        )
        agent._draws_since_adaptation = 10
        assert agent.maybe_adapt() is False

    def test_adapt_triggers_at_interval(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_a2", genome=genome, model=mock_model,
            budget=1_000_000, adaptation_interval=50,
        )
        agent._draws_since_adaptation = 50
        # Need some bet history for adaptation to be meaningful
        stats = BetTypeStats(bet_type="mot_so")
        for _ in range(15):
            stats.record(won=True, wagered=10_000, payout=12_000)
        agent._bet_type_stats["mot_so"] = stats
        assert agent.maybe_adapt() is True

    def test_adapt_resets_counter(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_a3", genome=genome, model=mock_model,
            budget=1_000_000, adaptation_interval=50,
        )
        agent._draws_since_adaptation = 50
        stats = BetTypeStats(bet_type="mot_so")
        for _ in range(15):
            stats.record(won=True, wagered=10_000, payout=12_000)
        agent._bet_type_stats["mot_so"] = stats
        agent.maybe_adapt()
        assert agent._draws_since_adaptation == 0

    def test_adapt_increases_weight_on_good_roi(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_a4", genome=genome, model=mock_model,
            budget=1_000_000, adaptation_interval=10,
        )
        agent._draws_since_adaptation = 10
        # Good ROI and recent win rate
        stats = BetTypeStats(bet_type="mot_so")
        for _ in range(20):
            stats.record(won=True, wagered=10_000, payout=15_000)
        agent._bet_type_stats["mot_so"] = stats
        old_weight = agent._bet_type_weights.get("mot_so", 1.0)
        agent.maybe_adapt()
        assert agent._bet_type_weights["mot_so"] > old_weight

    def test_adapt_decreases_weight_on_bad_roi(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_a5", genome=genome, model=mock_model,
            budget=1_000_000, adaptation_interval=10,
        )
        agent._draws_since_adaptation = 10
        # Bad ROI and low recent win rate
        stats = BetTypeStats(bet_type="mot_so")
        for _ in range(20):
            stats.record(won=False, wagered=10_000, payout=0)
        agent._bet_type_stats["mot_so"] = stats
        old_weight = agent._bet_type_weights.get("mot_so", 1.0)
        agent.maybe_adapt()
        assert agent._bet_type_weights["mot_so"] < old_weight

    def test_adapt_increases_bet_fraction_on_high_budget(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02)
        agent = AdaptiveAgent(
            agent_id="test_a6", genome=genome, model=mock_model,
            budget=3_000_000, bet_size=10_000, adaptation_interval=10,
        )
        # Set starting budget artificially
        agent._starting_budget = 1_000_000
        agent._draws_since_adaptation = 10
        old_fraction = agent._base_bet_fraction
        agent.maybe_adapt()
        assert agent._base_bet_fraction > old_fraction

    def test_adapt_decreases_bet_fraction_on_low_budget(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02)
        agent = AdaptiveAgent(
            agent_id="test_a7", genome=genome, model=mock_model,
            budget=200_000, bet_size=10_000, adaptation_interval=10,
        )
        agent._starting_budget = 1_000_000
        agent._draws_since_adaptation = 10
        old_fraction = agent._base_bet_fraction
        agent.maybe_adapt()
        assert agent._base_bet_fraction < old_fraction

    def test_adaptation_weight_clamped(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(
            agent_id="test_a8", genome=genome, model=mock_model,
            budget=1_000_000, adaptation_interval=10,
        )
        agent._draws_since_adaptation = 10
        # Set weight very high, then trigger bad ROI to test clamping
        agent._bet_type_weights["mot_so"] = 6.0
        stats = BetTypeStats(bet_type="mot_so")
        for _ in range(20):
            stats.record(won=False, wagered=10_000, payout=0)
        agent._bet_type_stats["mot_so"] = stats
        agent.maybe_adapt()
        assert agent._bet_type_weights["mot_so"] >= 0.1
        assert agent._bet_type_weights["mot_so"] <= 5.0


# --- Tests for to_simulation_result ---


class TestToSimulationResult:
    def test_to_simulation_result_conversion(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_s1", genome=genome, model=mock_model, budget=1_000_000)
        # Simulate some bets
        agent._starting_budget = 1_000_000
        agent._total_bets = 10
        agent._wins = 4
        agent._losses = 6
        agent._max_budget = 1_050_000
        agent._min_budget = 950_000
        agent._max_drawdown = 50_000
        agent._profit_curve = [1_000_000, 1_010_000, 990_000, 1_050_000, 950_000]

        result = agent.to_simulation_result()
        assert isinstance(result, SimulationResult)
        assert result.starting_budget == 1_000_000
        assert result.total_bets == 10
        assert result.wins == 4
        assert result.losses == 6
        assert result.max_drawdown == 50_000

    def test_to_simulation_result_empty(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_s2", genome=genome, model=mock_model, budget=1_000_000)
        agent._starting_budget = 1_000_000
        result = agent.to_simulation_result()
        assert result.total_bets == 0
        assert result.wins == 0
        assert result.roi == 0.0


# --- Tests for properties ---


class TestAgentProperties:
    def test_roi_property(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_p1", genome=genome, model=mock_model, budget=1_000_000)
        agent._starting_budget = 1_000_000
        agent.budget = 1_200_000
        assert agent.roi == pytest.approx(20.0)

    def test_roi_property_zero_starting(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_p2", genome=genome, model=mock_model, budget=0)
        agent._starting_budget = 0
        assert agent.roi == 0.0

    def test_win_rate_property(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_p3", genome=genome, model=mock_model, budget=1_000_000)
        agent._total_bets = 10
        agent._wins = 3
        assert agent.win_rate == pytest.approx(0.3)

    def test_win_rate_no_bets(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_p4", genome=genome, model=mock_model, budget=1_000_000)
        assert agent.win_rate == 0.0

    def test_max_drawdown_property(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="test_p5", genome=genome, model=mock_model, budget=1_000_000)
        agent._max_budget = 1_200_000
        agent._min_budget = 900_000
        assert agent.max_drawdown == 300_000


# --- Tests for _calculate_bet_amount ---


class TestCalculateBetAmount:
    def test_base_amount(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02)
        agent = AdaptiveAgent(agent_id="test_b1", genome=genome, model=mock_model, budget=1_000_000)
        amount = _calculate_bet_amount(agent)
        # base = 1_000_000 * 0.02 = 20_000
        assert amount >= agent.bet_size

    def test_amount_clamped_to_budget(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02)
        agent = AdaptiveAgent(agent_id="test_b2", genome=genome, model=mock_model, budget=5_000, bet_size=10_000)
        amount = _calculate_bet_amount(agent)
        assert amount <= agent.budget

    def test_hot_streak_increases_amount(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02, streak_sensitivity=0.5)
        agent = AdaptiveAgent(agent_id="test_b3", genome=genome, model=mock_model, budget=1_000_000)
        agent._current_streak = 5  # Hot streak
        amount_hot = _calculate_bet_amount(agent)
        # Reset streak
        agent._current_streak = 0
        amount_normal = _calculate_bet_amount(agent)
        assert amount_hot >= amount_normal

    def test_cold_streak_decreases_amount(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02, streak_sensitivity=0.5)
        agent = AdaptiveAgent(agent_id="test_b4", genome=genome, model=mock_model, budget=1_000_000)
        agent._current_streak = -5  # Cold streak
        amount_cold = _calculate_bet_amount(agent)
        agent._current_streak = 0
        amount_normal = _calculate_bet_amount(agent)
        assert amount_cold <= amount_normal

    def test_high_budget_health_increases(self, mock_model):
        genome = AgentGenome(base_bet_fraction=0.02)
        agent = AdaptiveAgent(agent_id="test_b5", genome=genome, model=mock_model, budget=2_000_000)
        agent._starting_budget = 1_000_000
        amount_high = _calculate_bet_amount(agent)
        # Reset to normal budget
        agent.budget = 1_000_000
        amount_normal = _calculate_bet_amount(agent)
        # With budget_ratio > 1.5, health_mult = 1.2, so amount_high > amount_normal
        assert amount_high > amount_normal


# --- Tests for create_diverse_agents ---


class TestCreateDiverseAgents:
    def test_returns_correct_count(self, mock_model):
        agents = create_diverse_agents(model=mock_model, budget=1_000_000, bet_size=10_000, n_agents=12)
        assert len(agents) == 12

    def test_returns_adaptive_agents(self, mock_model):
        agents = create_diverse_agents(model=mock_model, budget=1_000_000, bet_size=10_000, n_agents=5)
        for agent in agents:
            assert isinstance(agent, AdaptiveAgent)

    def test_agents_have_unique_ids(self, mock_model):
        agents = create_diverse_agents(model=mock_model, budget=1_000_000, bet_size=10_000, n_agents=12)
        ids = [a.agent_id for a in agents]
        assert len(ids) == len(set(ids))

    def test_agents_have_diverse_genomes(self, mock_model):
        agents = create_diverse_agents(model=mock_model, budget=1_000_000, bet_size=10_000, n_agents=12)
        algorithms = {a.genome.algorithm for a in agents}
        risk_profiles = {a.genome.risk_profile for a in agents}
        # Should have at least 2 different algorithms and risk profiles
        assert len(algorithms) >= 2
        assert len(risk_profiles) >= 2

    def test_custom_n_agents(self, mock_model):
        agents = create_diverse_agents(model=mock_model, budget=500_000, bet_size=10_000, n_agents=3)
        assert len(agents) == 3


# --- Edge case tests ---


class TestEdgeCases:
    def test_agent_with_zero_budget(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="edge_1", genome=genome, model=mock_model, budget=0)
        assert agent.is_alive is False
        bets = agent.decide_bets(np.zeros((1, 25), dtype=np.float32), {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.1, 6: 0.1})
        assert len(bets) == 0

    def test_all_bets_lose(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="edge_2", genome=genome, model=mock_model, budget=100_000)
        for i in range(10):
            agent.record_result(
                bet_type=BetType.MOT_SO,
                bet_value=3,
                bet_amount=10_000,
                actual_digits=[1, 2, 4],
                actual_total=7,
                date=f"2025-01-{i+1:02d}",
                draw_id=f"{i+1:07d}",
            )
        assert agent._wins == 0
        assert agent._losses == 10
        assert agent.budget == 0

    def test_agent_with_empty_predictions(self, mock_model):
        genome = AgentGenome()
        agent = AdaptiveAgent(agent_id="edge_3", genome=genome, model=mock_model, budget=1_000_000)
        bets = agent.decide_bets(np.zeros((1, 25), dtype=np.float32), {})
        # Should handle gracefully (return empty or default bets)
        assert isinstance(bets, list)
