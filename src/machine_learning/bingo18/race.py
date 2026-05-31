"""Multi-agent racing system for Bingo18.

Coordinates multiple AdaptiveAgents running walk-forward through historical data.
Each agent independently decides bets, adapts strategy, and manages budget.
Results are aggregated and ranked to find the best strategy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from machine_learning.bingo18.agent import AdaptiveAgent, BetTypeStats
from machine_learning.bingo18.features import Bingo18FeatureEngineer
from machine_learning.bingo18.simulator import BetRecord, SimulationResult


@dataclass
class RaceAgentResult:
    """Result of a single agent in the race."""

    agent_id: str
    genome: Any  # AgentGenome
    final_budget: int
    starting_budget: int
    total_bets: int
    wins: int
    losses: int
    roi: float
    win_rate: float
    max_drawdown: int
    generations: int
    adaptation_count: int
    bet_type_stats: dict[str, BetTypeStats]
    profit_curve: list[int]
    bet_history: list[BetRecord]
    simulation_result: SimulationResult
    elapsed_seconds: float

    @property
    def profit(self) -> int:
        """Profit as final - starting budget."""
        return self.final_budget - self.starting_budget

    @property
    def score(self) -> float:
        """Composite score: ROI * survival_factor * consistency_factor."""
        survival = 1.0 if self.final_budget > 0 else 0.5
        consistency = 1.0 - min(self.max_drawdown / max(self.starting_budget, 1), 0.5)
        return self.roi * survival * consistency


@dataclass
class RaceResult:
    """Complete result of a race."""

    total_agents: int
    completed_agents: int
    failed_agents: int
    budget: int
    bet_size: int
    total_draws: int
    adaptation_interval: int
    agent_results: list[RaceAgentResult]
    winner: RaceAgentResult | None
    total_elapsed_seconds: float
    timestamp: str


class RaceCoordinator:
    """Manages the adaptive multi-agent race.

    Each agent runs draw-by-draw through the same historical data.
    Agents adapt independently every N draws.
    Optional: agents can share knowledge (top bet types broadcast).

    Parameters
    ----------
    agents : list[AdaptiveAgent]
        Agents participating in the race.
    adaptation_interval : int
        How often agents adapt (in draws).
    share_knowledge : bool
        Whether agents share top bet types periodically.
    knowledge_share_interval : int
        How often to share knowledge (in draws).
    """

    def __init__(
        self,
        agents: list[AdaptiveAgent],
        adaptation_interval: int = 50,
        share_knowledge: bool = False,
        knowledge_share_interval: int = 100,
    ) -> None:
        self.agents = agents
        self.adaptation_interval = adaptation_interval
        self.share_knowledge = share_knowledge
        self.knowledge_share_interval = knowledge_share_interval

    def run_race(self, df: pd.DataFrame) -> RaceResult:
        """Run the race across all draws in df.

        Walk-forward simulation: each draw is processed one at a time.
        Agents do NOT see future results.
        """
        start_time = time.perf_counter()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        if df.empty:
            return RaceResult(
                total_agents=len(self.agents),
                completed_agents=len(self.agents),
                failed_agents=0,
                budget=0,
                bet_size=0,
                total_draws=0,
                adaptation_interval=self.adaptation_interval,
                agent_results=[],
                winner=None,
                total_elapsed_seconds=0.0,
                timestamp=timestamp,
            )

        results = df["result"].tolist()
        totals = df["total"].tolist()
        large_smalls = df["large_small"].tolist()
        dates = df["date"].tolist() if "date" in df.columns else [f"draw_{i}" for i in range(len(df))]
        ids = df["id"].tolist() if "id" in df.columns else [f"{i:07d}" for i in range(len(df))]

        # Determine window from first agent
        window = self.agents[0].genome.window if self.agents else 30
        feature_engineer = Bingo18FeatureEngineer(window=window)

        n_draws = len(results)
        start_idx = window  # Need at least `window` draws for features

        logger.info(f"Starting race: {len(self.agents)} agents, {n_draws} draws, window={window}")

        # Walk-forward: process each draw one at a time
        for i in range(start_idx, n_draws):
            # Build features from previous window draws
            recent_draws = results[i - window : i]
            recent_totals = totals[i - window : i]
            recent_ls = large_smalls[i - window : i]

            try:
                X = feature_engineer.build_features_for_predict(recent_draws, recent_totals, recent_ls)
            except (ValueError, IndexError):
                continue

            actual_digits = results[i]
            actual_total = totals[i]
            date = dates[i]
            draw_id = ids[i]

            # Each agent decides and records
            for agent in self.agents:
                if not agent.is_alive:
                    continue

                # Get model predictions
                predictions = _get_predictions(agent, X)
                if not predictions:
                    continue

                # Agent decides bets
                bets = agent.decide_bets(X, predictions)

                # Process each bet
                for bet_type, bet_value, bet_amount in bets:
                    agent.record_result(
                        bet_type=bet_type,
                        bet_value=bet_value,
                        bet_amount=bet_amount,
                        actual_digits=actual_digits,
                        actual_total=actual_total,
                        date=date,
                        draw_id=draw_id,
                    )

                # Check adaptation
                agent.maybe_adapt()

            # Optional: knowledge sharing
            if self.share_knowledge and (i - start_idx) % self.knowledge_share_interval == 0 and i > start_idx:
                self._maybe_share_knowledge()

        # Create results for each agent
        agent_results = []
        elapsed = time.perf_counter() - start_time
        for agent in self.agents:
            sim_result = agent.to_simulation_result()
            race_result = RaceAgentResult(
                agent_id=agent.agent_id,
                genome=agent.genome,
                final_budget=agent.budget,
                starting_budget=agent._starting_budget,
                total_bets=agent._total_bets,
                wins=agent._wins,
                losses=agent._losses,
                roi=agent.roi,
                win_rate=agent.win_rate,
                max_drawdown=agent.max_drawdown,
                generations=agent._generation,
                adaptation_count=agent._generation,
                bet_type_stats=dict(agent._bet_type_stats),
                profit_curve=list(agent._profit_curve),
                bet_history=list(agent._bet_history),
                simulation_result=sim_result,
                elapsed_seconds=elapsed,
            )
            agent_results.append(race_result)

        # Sort by ROI (best first)
        agent_results.sort(key=lambda r: r.roi, reverse=True)

        total_elapsed = time.perf_counter() - start_time
        winner = agent_results[0] if agent_results else None

        if winner:
            logger.info(
                f"Race complete: {len(agent_results)} agents, "
                f"winner={winner.agent_id} (ROI={winner.roi:.1f}%)"
            )
        else:
            logger.info(f"Race complete: {len(agent_results)} agents, no winner")

        return RaceResult(
            total_agents=len(self.agents),
            completed_agents=len(agent_results),
            failed_agents=0,
            budget=self.agents[0]._starting_budget if self.agents else 0,
            bet_size=self.agents[0].bet_size if self.agents else 0,
            total_draws=n_draws,
            adaptation_interval=self.adaptation_interval,
            agent_results=agent_results,
            winner=winner,
            total_elapsed_seconds=total_elapsed,
            timestamp=timestamp,
        )

    def _maybe_share_knowledge(self) -> None:
        """Share top-performing bet types from leading agents to others.

        Strategy: broadcast the top 3 bet types from the top 3 agents.
        Other agents slightly increase their weights for those types.
        """
        # Rank agents by ROI
        alive_agents = [a for a in self.agents if a.is_alive]
        if len(alive_agents) < 2:
            return

        sorted_agents = sorted(alive_agents, key=lambda a: a.roi, reverse=True)
        top_agents = sorted_agents[:3]

        # Collect top bet types from top agents
        top_bet_types: dict[str, float] = {}
        for agent in top_agents:
            for bt_name, stats in agent._bet_type_stats.items():
                if stats.total_bets > 0 and stats.roi > 0:
                    top_bet_types[bt_name] = top_bet_types.get(bt_name, 0) + stats.roi

        if not top_bet_types:
            return

        # Nudge weights of other agents
        nudge_amount = 0.15
        for agent in sorted_agents[3:]:
            for bt_name in top_bet_types:
                current = agent._bet_type_weights.get(bt_name, 1.0)
                agent._bet_type_weights[bt_name] = min(current + nudge_amount, 5.0)

        logger.debug(f"Knowledge shared: top bet types {[bt for bt, _ in sorted(top_bet_types.items(), key=lambda x: x[1], reverse=True)[:3]]}")


def _get_predictions(agent: AdaptiveAgent, X: np.ndarray) -> dict[int, float] | None:
    """Get digit predictions from agent's model.

    Returns dict mapping digit (1-6) to probability.
    """
    try:
        predictions = agent.model.predict_proba(X)
        if not predictions:
            return None
        return predictions
    except Exception:
        return None
