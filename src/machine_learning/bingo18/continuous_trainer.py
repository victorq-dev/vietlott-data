"""Continuous parallel training for Bingo18 agents.

Runs multiple agents in parallel, each walking forward through historical data.
Agents self-heal on bankruptcy (reset budget, keep learned state) and persist
their state to disk for incremental training across sessions.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from machine_learning.bingo18.agent import AdaptiveAgent, AgentGenome, create_diverse_agents
from machine_learning.bingo18.features import Bingo18FeatureEngineer
from machine_learning.bingo18.model import Bingo18Model


@dataclass
class AgentTrainingResult:
    """Result of a single agent's training round."""

    agent_id: str
    roi: float
    final_budget: int
    starting_budget: int
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    bankruptcies: int
    generations: int
    survived: bool
    composite_score: float = 0.0


@dataclass
class ContinuousTrainingResult:
    """Result of a continuous training session."""

    total_rounds: int
    n_agents: int
    agent_results: list[AgentTrainingResult]
    total_elapsed_seconds: float
    timestamp: str


def _get_predictions(agent: AdaptiveAgent, X) -> dict[int, float] | None:
    """Get digit predictions from agent's model."""
    try:
        predictions = agent.model.predict_proba(X)
        if not predictions:
            return None
        return predictions
    except Exception:
        return None


def _share_knowledge(agents: list[AdaptiveAgent]) -> None:
    """Nudge bottom agents' weights toward top agents' weights."""
    alive = [a for a in agents if a.is_alive]
    if len(alive) < 4:
        return
    sorted_agents = sorted(alive, key=lambda a: a.roi, reverse=True)
    top_agents = sorted_agents[: max(1, len(sorted_agents) // 3)]

    # Average weights of top agents
    avg_weights: dict[str, float] = {}
    for bt_name in sorted_agents[0]._bet_type_weights:
        avg_weights[bt_name] = sum(a._bet_type_weights.get(bt_name, 1.0) for a in top_agents) / len(top_agents)

    # Nudge bottom 2/3 toward average
    for agent in sorted_agents[len(top_agents) :]:
        for bt_name, target in avg_weights.items():
            current = agent._bet_type_weights.get(bt_name, 1.0)
            agent._bet_type_weights[bt_name] = current + 0.15 * (target - current)

    top_ids = [a.agent_id for a in top_agents]
    logger.info(f"Knowledge shared from top agents: {top_ids}")


def _evolve_population(agents: list[AdaptiveAgent], rng: np.random.Generator) -> None:
    """Replace bottom 30% agents' strategies with mutations of top agents."""
    if len(agents) < 4:
        return
    sorted_agents = sorted(agents, key=lambda a: a.roi, reverse=True)
    n_elite = max(1, len(sorted_agents) // 3)
    n_replace = max(1, len(sorted_agents) * 3 // 10)
    elite = sorted_agents[:n_elite]
    to_replace = sorted_agents[-n_replace:]

    for agent in to_replace:
        # Pick a random elite agent as template
        template = elite[rng.integers(len(elite))]
        # Copy weights with small noise
        for bt_name, weight in template._bet_type_weights.items():
            noise = float(rng.uniform(0.85, 1.15))
            agent._bet_type_weights[bt_name] = float(np.clip(weight * noise, 0.1, 5.0))
        logger.info(f"[{agent.agent_id}] Evolved from {template.agent_id}")


def train_single_agent(
    agent: AdaptiveAgent,
    df: pd.DataFrame,
    n_rounds: int,
    agent_state_dir: Path,
    log_prefix: str = "",
) -> AgentTrainingResult:
    """Train a single agent through multiple rounds of data.

    Each round walks forward through all draws. If agent goes bankrupt,
    budget is reset but learned state is kept.

    Parameters
    ----------
    agent : AdaptiveAgent
        Agent to train.
    df : pd.DataFrame
        Historical draw data.
    n_rounds : int
        Number of full passes through the data.
    agent_state_dir : Path
        Directory to save agent state.
    log_prefix : str
        Prefix for log messages.

    Returns
    -------
    AgentTrainingResult
    """
    results = df["result"].tolist()
    totals = df["total"].tolist()
    large_smalls = df["large_small"].tolist()
    dates = df["date"].tolist() if "date" in df.columns else [f"draw_{i}" for i in range(len(df))]
    ids = df["id"].tolist() if "id" in df.columns else [f"{i:07d}" for i in range(len(df))]

    window = agent.genome.window
    feature_engineer = Bingo18FeatureEngineer(window=window)
    n_draws = len(results)
    start_idx = window

    bankruptcies = 0
    prefix = f"{log_prefix}[{agent.agent_id}]"

    for round_num in range(n_rounds):
        round_start = time.perf_counter()
        bets_this_round = 0

        for i in range(start_idx, n_draws):
            # Build features
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

            # Get predictions
            predictions = _get_predictions(agent, X)
            if not predictions:
                continue

            # Agent decides bets
            balance_before = agent.budget
            bets = agent.decide_bets(X, predictions)

            # Process each bet
            bet_details = []
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
                bet_details.append(f"{bet_type.value}({bet_value})={bet_amount:,}")
                bets_this_round += 1

            balance_after = agent.budget
            net = balance_after - balance_before

            # Log per-draw decisions
            if bet_details:
                logger.info(
                    f"{prefix} R{round_num + 1} D{i}: "
                    f"bets=[{', '.join(bet_details)}] "
                    f"result={actual_digits} total={actual_total} "
                    f"balance {balance_before:,}->{balance_after:,} ({net:+,})"
                )

            # Handle bankruptcy - self-heal
            if not agent.is_alive:
                bankruptcies += 1
                logger.warning(f"{prefix} BANKRUPT at draw {i}! Resetting budget (keep learned state)")
                agent.reset_budget()
                agent.save(agent_state_dir)

            # Increment draw counter once per draw, then adapt
            agent.increment_draw_counter()
            if agent.maybe_adapt():
                logger.info(f"{prefix} Adapted gen={agent._generation}")

        # End of round summary
        round_elapsed = time.perf_counter() - round_start
        logger.info(
            f"{prefix} Round {round_num + 1}/{n_rounds} done: "
            f"bets={bets_this_round}, budget={agent.budget:,}, "
            f"ROI={agent.roi:+.1f}%, win_rate={agent.win_rate:.1%}, "
            f"gen={agent._generation}, time={round_elapsed:.1f}s"
        )

        # Save after each round
        agent.save(agent_state_dir)

    bankrupt_rate = bankruptcies / max(1, n_rounds)
    composite_score = agent.roi * (1.0 - min(bankrupt_rate, 1.0))

    return AgentTrainingResult(
        agent_id=agent.agent_id,
        roi=agent.roi,
        final_budget=agent.budget,
        starting_budget=agent._starting_budget,
        total_bets=agent._total_bets,
        wins=agent._wins,
        losses=agent._losses,
        win_rate=agent.win_rate,
        bankruptcies=bankruptcies,
        generations=agent._generation,
        survived=agent.is_alive,
        composite_score=composite_score,
    )


def run_continuous_training(
    model: Bingo18Model,
    df: pd.DataFrame,
    n_agents: int = 6,
    n_rounds: int = 3,
    budget: int = 500_000,
    bet_size: int = 10_000,
    adaptation_interval: int = 50,
    agent_state_dir: str | Path = "models/bingo18/agents",
    load_existing: bool = True,
) -> ContinuousTrainingResult:
    """Run continuous training with multiple agents in parallel.

    Agents train concurrently through the data. Each agent self-heals
    on bankruptcy and persists state to disk.

    Parameters
    ----------
    model : Bingo18Model
        Trained Stage 1 model for digit predictions.
    df : pd.DataFrame
        Historical draw data.
    n_agents : int
        Number of agents to train in parallel.
    n_rounds : int
        Number of full passes through data per agent.
    budget : int
        Starting budget per agent.
    bet_size : int
        Base bet size.
    adaptation_interval : int
        Draws between adaptation cycles.
    agent_state_dir : str | Path
        Directory to save/load agent state.
    load_existing : bool
        Whether to load existing agent state from disk.

    Returns
    -------
    ContinuousTrainingResult
    """
    start_time = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    state_dir = Path(agent_state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Create or load agents
    agents: list[AdaptiveAgent] = []

    if load_existing:
        # Try to load existing agents
        existing_files = sorted(state_dir.glob("*.json"))
        for filepath in existing_files[:n_agents]:
            try:
                agent = AdaptiveAgent.load(filepath, model)
                agents.append(agent)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

    # Create new agents if needed
    n_new = n_agents - len(agents)
    if n_new > 0:
        new_agents = create_diverse_agents(
            model=model,
            budget=budget,
            bet_size=bet_size,
            n_agents=n_new,
            adaptation_interval=adaptation_interval,
        )
        # Offset agent IDs to avoid conflicts
        existing_ids = {a.agent_id for a in agents}
        for agent in new_agents:
            while agent.agent_id in existing_ids:
                parts = agent.agent_id.split("_")
                idx = int(parts[-1]) + 1 if parts[-1].isdigit() else 0
                agent = AdaptiveAgent(
                    agent_id=f"agent_{idx:03d}",
                    genome=agent.genome,
                    model=agent.model,
                    budget=agent.budget,
                    bet_size=agent.bet_size,
                    adaptation_interval=agent.adaptation_interval,
                    strategy_model=agent._strategy_model,
                )
            existing_ids.add(agent.agent_id)
            agents.append(agent)

    logger.info(
        f"Starting continuous training: {len(agents)} agents, "
        f"{n_rounds} rounds, budget={budget:,}, "
        f"loaded_existing={len(agents) - n_new}, new={n_new}"
    )

    # Run agents in parallel
    results: list[AgentTrainingResult] = []
    futures: dict = {}

    try:
        with ThreadPoolExecutor(max_workers=n_agents) as executor:
            futures = {
                executor.submit(
                    train_single_agent,
                    agent=agent,
                    df=df,
                    n_rounds=n_rounds,
                    agent_state_dir=state_dir,
                    log_prefix="",
                ): agent
                for agent in agents
            }

            for future in as_completed(futures):
                agent_obj = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"[{agent_obj.agent_id}] Training complete: "
                        f"ROI={result.roi:+.1f}%, budget={result.final_budget:,}, "
                        f"bets={result.total_bets}, bankruptcies={result.bankruptcies}"
                    )
                except Exception as e:
                    logger.error(f"[{agent_obj.agent_id}] Training failed: {e}")

    except KeyboardInterrupt:
        import os
        import signal
        # Block further Ctrl+C so the save loop can't be interrupted again
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.warning("Interrupted! Saving all agent states before exit...")
        for future in list(futures):
            future.cancel()
        for agent in agents:
            try:
                agent.save(state_dir)
                logger.info(f"[{agent.agent_id}] State saved.")
            except Exception as e:
                logger.error(f"[{agent.agent_id}] Failed to save: {e}")
        logger.info("All states saved. Exiting.")
        os._exit(0)  # bypass thread cleanup to avoid second-Ctrl+C race

    # Share knowledge and evolve population after all agents complete
    _share_knowledge(agents)
    _rng = np.random.default_rng()
    _evolve_population(agents, _rng)

    # Sort by composite_score
    results.sort(key=lambda r: r.composite_score, reverse=True)

    total_elapsed = time.perf_counter() - start_time

    # Print leaderboard
    logger.info("=" * 80)
    logger.info("CONTINUOUS TRAINING LEADERBOARD")
    logger.info("=" * 80)
    for i, r in enumerate(results, 1):
        status = "ALIVE" if r.survived else "DEAD"
        logger.info(
            f"  #{i} {r.agent_id}: ROI={r.roi:+.1f}%, composite={r.composite_score:+.2f}, "
            f"budget={r.final_budget:,}, win_rate={r.win_rate:.1%}, bets={r.total_bets}, "
            f"bankruptcies={r.bankruptcies}, gen={r.generations}, {status}"
        )
    logger.info("=" * 80)

    return ContinuousTrainingResult(
        total_rounds=n_rounds,
        n_agents=len(agents),
        agent_results=results,
        total_elapsed_seconds=total_elapsed,
        timestamp=timestamp,
    )
