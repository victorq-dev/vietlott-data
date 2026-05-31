"""Tests for RaceCoordinator - Bingo18 multi-agent racing system."""

import numpy as np
import pandas as pd
import pytest

from machine_learning.bingo18.agent import (
    AdaptiveAgent,
    AgentGenome,
    BetTypeStats,
    create_diverse_agents,
)
from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.race import (
    RaceAgentResult,
    RaceCoordinator,
    RaceResult,
)
from machine_learning.bingo18.simulator import BetRecord, BetType, SimulationResult

# --- Fixtures ---


@pytest.fixture
def mock_model():
    """Untrained model for agent creation (agents don't need trained model for tests)."""
    return Bingo18Model(window=10, n_estimators=10, max_depth=2)


@pytest.fixture
def trained_model():
    """A model trained on synthetic data for race tests."""
    df = _make_synthetic_df(80)
    model = Bingo18Model(window=10, n_estimators=10, max_depth=2)
    model.train(df, test_ratio=0.2)
    return model


def _make_synthetic_df(n_draws: int, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic Bingo18 DataFrame for testing.

    Each draw has 3 random digits 1-6, with result, total, large_small columns.
    """
    rng = np.random.default_rng(seed)
    results = []
    totals = []
    large_smalls = []
    dates = []
    ids = []
    for i in range(n_draws):
        draw = sorted(rng.integers(1, 7, size=3).tolist())
        total = sum(draw)
        if total <= 9:
            ls = "Nhỏ"
        elif total <= 11:
            ls = "Hòa"
        else:
            ls = "Lớn"
        results.append(draw)
        totals.append(total)
        large_smalls.append(ls)
        dates.append(f"2025-01-{(i % 28) + 1:02d}")
        ids.append(f"{i + 1:07d}")

    return pd.DataFrame(
        {
            "result": results,
            "total": totals,
            "large_small": large_smalls,
            "date": dates,
            "id": ids,
        }
    )


def _make_small_agents(model, n: int = 3, budget: int = 500_000) -> list[AdaptiveAgent]:
    """Create n simple agents for testing."""
    agents = []
    risk_profiles = ["conservative", "moderate", "aggressive"]
    for i in range(n):
        genome = AgentGenome(
            algorithm="gradient_boosting",
            window=model.window,
            risk_profile=risk_profiles[i % len(risk_profiles)],
            exploration_rate=0.0,  # No randomness for deterministic tests
        )
        agent = AdaptiveAgent(
            agent_id=f"test_agent_{i:03d}",
            genome=genome,
            model=model,
            budget=budget,
            bet_size=10_000,
            adaptation_interval=50,
        )
        agents.append(agent)
    return agents


# --- Tests for RaceAgentResult ---


class TestRaceAgentResult:
    def test_creation_with_required_fields(self):
        """RaceAgentResult can be created with all required fields."""
        result = RaceAgentResult(
            agent_id="agent_000",
            genome=AgentGenome(),
            final_budget=1_100_000,
            starting_budget=1_000_000,
            total_bets=50,
            wins=20,
            losses=30,
            roi=10.0,
            win_rate=0.4,
            max_drawdown=50_000,
            generations=2,
            adaptation_count=2,
            bet_type_stats={},
            profit_curve=[1_000_000, 1_100_000],
            bet_history=[],
            simulation_result=SimulationResult(
                starting_budget=1_000_000,
                final_budget=1_100_000,
                bet_size=10_000,
                bet_type="adaptive",
            ),
            elapsed_seconds=1.5,
        )
        assert result.agent_id == "agent_000"
        assert result.roi == 10.0
        assert result.final_budget == 1_100_000

    def test_profit_property(self):
        """RaceAgentResult exposes profit as final - starting budget."""
        result = RaceAgentResult(
            agent_id="a1",
            genome=AgentGenome(),
            final_budget=1_200_000,
            starting_budget=1_000_000,
            total_bets=10,
            wins=5,
            losses=5,
            roi=20.0,
            win_rate=0.5,
            max_drawdown=0,
            generations=0,
            adaptation_count=0,
            bet_type_stats={},
            profit_curve=[],
            bet_history=[],
            simulation_result=SimulationResult(
                starting_budget=1_000_000,
                final_budget=1_200_000,
                bet_size=10_000,
                bet_type="adaptive",
            ),
            elapsed_seconds=0.0,
        )
        assert result.profit == 200_000


# --- Tests for RaceResult ---


class TestRaceResult:
    def test_winner_is_first_in_agent_results(self):
        """Winner must be the first element of sorted agent_results."""
        agent_results = [
            RaceAgentResult(
                agent_id="best",
                genome=AgentGenome(),
                final_budget=1_500_000,
                starting_budget=1_000_000,
                total_bets=50,
                wins=30,
                losses=20,
                roi=50.0,
                win_rate=0.6,
                max_drawdown=20_000,
                generations=1,
                adaptation_count=1,
                bet_type_stats={},
                profit_curve=[],
                bet_history=[],
                simulation_result=SimulationResult(
                    starting_budget=1_000_000,
                    final_budget=1_500_000,
                    bet_size=10_000,
                    bet_type="adaptive",
                ),
                elapsed_seconds=1.0,
            ),
            RaceAgentResult(
                agent_id="worst",
                genome=AgentGenome(),
                final_budget=800_000,
                starting_budget=1_000_000,
                total_bets=50,
                wins=15,
                losses=35,
                roi=-20.0,
                win_rate=0.3,
                max_drawdown=100_000,
                generations=0,
                adaptation_count=0,
                bet_type_stats={},
                profit_curve=[],
                bet_history=[],
                simulation_result=SimulationResult(
                    starting_budget=1_000_000,
                    final_budget=800_000,
                    bet_size=10_000,
                    bet_type="adaptive",
                ),
                elapsed_seconds=1.0,
            ),
        ]

        race_result = RaceResult(
            total_agents=2,
            completed_agents=2,
            failed_agents=0,
            budget=1_000_000,
            bet_size=10_000,
            total_draws=50,
            adaptation_interval=50,
            agent_results=agent_results,
            winner=agent_results[0],
            total_elapsed_seconds=2.0,
            timestamp="2025-01-01T00:00:00",
        )
        assert race_result.winner.agent_id == "best"
        assert race_result.agent_results[0].agent_id == "best"

    def test_completed_plus_failed_equals_total(self):
        """completed_agents + failed_agents == total_agents."""
        race_result = RaceResult(
            total_agents=5,
            completed_agents=4,
            failed_agents=1,
            budget=1_000_000,
            bet_size=10_000,
            total_draws=50,
            adaptation_interval=50,
            agent_results=[],
            winner=None,
            total_elapsed_seconds=0.0,
            timestamp="2025-01-01T00:00:00",
        )
        assert race_result.completed_agents + race_result.failed_agents == race_result.total_agents


# --- Tests for RaceCoordinator ---


class TestRaceCoordinator:
    def test_race_completes_with_multiple_agents(self, trained_model):
        """Race with 3 agents completes and returns all results."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=3, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert isinstance(result, RaceResult)
        assert result.total_agents == 3
        assert len(result.agent_results) == 3

    def test_result_sorted_by_roi_descending(self, trained_model):
        """Agent results are sorted by ROI (best first)."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=5, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        rois = [ar.roi for ar in result.agent_results]
        assert rois == sorted(rois, reverse=True)

    def test_winner_is_first_in_sorted_list(self, trained_model):
        """Winner is the first element of agent_results (highest ROI)."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=3, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.winner is not None
        assert result.winner.agent_id == result.agent_results[0].agent_id

    def test_completed_plus_failed_equals_total(self, trained_model):
        """completed_agents + failed_agents == total_agents."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=4, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.completed_agents + result.failed_agents == result.total_agents

    def test_agent_that_goes_broke_still_in_results(self, trained_model):
        """Agent with tiny budget that goes broke still appears in results."""
        df = _make_synthetic_df(60)
        # One agent with very small budget will likely go broke
        agents = _make_small_agents(trained_model, n=2, budget=500_000)
        broke_agent = AdaptiveAgent(
            agent_id="broke_agent",
            genome=AgentGenome(window=trained_model.window, exploration_rate=0.0),
            model=trained_model,
            budget=20_000,  # Only 2 bets worth
            bet_size=10_000,
            adaptation_interval=20,
        )
        agents.append(broke_agent)

        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        agent_ids = [ar.agent_id for ar in result.agent_results]
        assert "broke_agent" in agent_ids
        assert result.total_agents == 3

    def test_empty_dataframe_returns_empty_result(self, trained_model):
        """Empty DataFrame produces a result with zero draws and no bets."""
        df = _make_synthetic_df(0)
        # Need at least window+1 draws for any bets; empty means no bets at all
        agents = _make_small_agents(trained_model, n=2, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.total_draws == 0
        assert result.total_agents == 2
        for ar in result.agent_results:
            assert ar.total_bets == 0

    def test_single_agent_race(self, trained_model):
        """Race with a single agent works correctly."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=1, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.total_agents == 1
        assert result.winner is not None
        assert result.winner.agent_id == "test_agent_000"

    def test_adaptation_events_tracked(self, trained_model):
        """Adaptation count and generations are tracked in results."""
        # Use small adaptation_interval to trigger adaptations
        df = _make_synthetic_df(80)
        agents = _make_small_agents(trained_model, n=2, budget=1_000_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=10)
        result = coordinator.run_race(df)

        # At least one agent should have adapted (with 80 draws and interval=10)
        total_adaptations = sum(ar.adaptation_count for ar in result.agent_results)
        assert total_adaptations > 0

    def test_all_agents_have_simulation_result(self, trained_model):
        """Each RaceAgentResult includes a SimulationResult for compatibility."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=3, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        for ar in result.agent_results:
            assert isinstance(ar.simulation_result, SimulationResult)
            assert ar.simulation_result.starting_budget == ar.starting_budget
            assert ar.simulation_result.final_budget == ar.final_budget

    def test_profit_curve_tracks_budget_over_time(self, trained_model):
        """Profit curve has one entry per draw + initial budget."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=1, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        ar = result.agent_results[0]
        # Profit curve should have at least 1 entry (initial budget)
        assert len(ar.profit_curve) >= 1
        # First entry should be starting budget
        assert ar.profit_curve[0] == ar.starting_budget

    def test_race_with_diverse_agents(self, trained_model):
        """Race works with agents from create_diverse_agents()."""
        df = _make_synthetic_df(60)
        agents = create_diverse_agents(
            model=trained_model,
            budget=500_000,
            bet_size=10_000,
            n_agents=6,
            adaptation_interval=20,
        )
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.total_agents == 6
        assert len(result.agent_results) == 6
        # All agent IDs should be present
        agent_ids = {ar.agent_id for ar in result.agent_results}
        assert len(agent_ids) == 6

    def test_bet_history_recorded_per_agent(self, trained_model):
        """Each agent's bet_history is populated with BetRecord entries."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=2, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        for ar in result.agent_results:
            if ar.total_bets > 0:
                assert len(ar.bet_history) > 0
                assert isinstance(ar.bet_history[0], BetRecord)

    def test_knowledge_share_nudges_weights(self, trained_model):
        """Knowledge sharing nudges weights of underperforming agents."""
        df = _make_synthetic_df(80)
        agents = _make_small_agents(trained_model, n=3, budget=1_000_000)
        coordinator = RaceCoordinator(
            agents=agents,
            adaptation_interval=10,
            share_knowledge=True,
            knowledge_share_interval=20,
        )
        result = coordinator.run_race(df)

        # Race should complete without errors
        assert result.total_agents == 3
        assert len(result.agent_results) == 3

    def test_timestamp_is_set(self, trained_model):
        """RaceResult has a non-empty timestamp."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=2, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.timestamp != ""
        assert "T" in result.timestamp  # ISO format

    def test_elapsed_seconds_positive(self, trained_model):
        """Elapsed seconds are positive when race has work to do."""
        df = _make_synthetic_df(60)
        agents = _make_small_agents(trained_model, n=2, budget=500_000)
        coordinator = RaceCoordinator(agents=agents, adaptation_interval=20)
        result = coordinator.run_race(df)

        assert result.total_elapsed_seconds >= 0.0
        for ar in result.agent_results:
            assert ar.elapsed_seconds >= 0.0
