"""Parallel strategy trainer for Bingo18.

Runs multiple training agents in parallel with different hyperparameters,
monitors their performance, and selects the best model.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.strategy_model import StrategyModel
from machine_learning.bingo18.strategy_trainer import StrategyTrainer


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a single training agent."""

    name: str
    learning_rate: float = 0.001
    hidden_sizes: tuple[int, ...] = (64, 32)
    exploration_rate: float = 0.3
    skip_threshold: float = 0.15
    n_epochs: int = 10
    budget: int = 500_000
    bet_size: int = 10_000
    random_state: int = 42


@dataclass
class TrainingResult:
    """Result from a single training agent."""

    config: TrainingConfig
    test_win_rate: float
    test_roi: float
    test_final_budget: int
    test_total_bets: int
    model_path: str | None
    history: list[dict[str, float]]
    elapsed_seconds: float


@dataclass
class ParallelTrainingResult:
    """Result from parallel training."""

    best: TrainingResult
    all_results: list[TrainingResult]
    total_elapsed: float


def _train_single(config: TrainingConfig, df: pd.DataFrame, model: Bingo18Model, output_dir: Path) -> TrainingResult:
    """Train a single strategy model with given config.

    Parameters
    ----------
    config : TrainingConfig
    df : pd.DataFrame
    model : Bingo18Model
    output_dir : Path

    Returns
    -------
    TrainingResult
    """
    start = time.time()

    n_features = len(model.feature_engineer._feature_names())
    strategy_model = StrategyModel(
        context_dim=n_features + 6 + 3 + 10 + 1,
        hidden_sizes=config.hidden_sizes,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        skip_threshold=config.skip_threshold,
    )

    trainer = StrategyTrainer(
        model=model,
        strategy_model=strategy_model,
        budget=config.budget,
        bet_size=config.bet_size,
        exploration_rate=config.exploration_rate,
        random_state=config.random_state,
    )

    result = trainer.train(df, n_epochs=config.n_epochs)

    # Save model
    model_path = output_dir / f"strategy_{config.name}.pkl"
    strategy_model.save(model_path)

    elapsed = time.time() - start

    test = result["test"]
    return TrainingResult(
        config=config,
        test_win_rate=test["win_rate"],
        test_roi=test["roi"],
        test_final_budget=test["final_budget"],
        test_total_bets=test["total_bets"],
        model_path=str(model_path),
        history=result["history"],
        elapsed_seconds=elapsed,
    )


def run_parallel_training(
    df: pd.DataFrame,
    model: Bingo18Model,
    configs: list[TrainingConfig],
    output_dir: str | Path = "/tmp/bingo18_parallel",
    max_workers: int | None = None,
) -> ParallelTrainingResult:
    """Run multiple training agents in parallel.

    Parameters
    ----------
    df : pd.DataFrame — historical draws
    model : Bingo18Model — trained Stage 1 model
    configs : list[TrainingConfig] — configs for each training agent
    output_dir : str | Path — directory to save models
    max_workers : int | None — max parallel workers (None = CPU count)

    Returns
    -------
    ParallelTrainingResult with best model and all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    results: list[TrainingResult] = []

    logger.info(f"Starting parallel training with {len(configs)} agents")

    # Run sequentially for now (ProcessPoolExecutor has issues with large data)
    for i, config in enumerate(configs):
        logger.info(f"[{i+1}/{len(configs)}] Training agent '{config.name}'...")
        try:
            result = _train_single(config, df, model, output_dir)
            results.append(result)
            logger.info(
                f"  [{config.name}] win_rate={result.test_win_rate:.1%}, "
                f"ROI={result.test_roi:.1f}%, budget={result.test_final_budget:,}, "
                f"bets={result.test_total_bets}, time={result.elapsed_seconds:.1f}s"
            )
        except Exception as e:
            logger.error(f"  [{config.name}] FAILED: {e}")

    elapsed = time.time() - start

    # Find best by ROI (least negative)
    valid_results = [r for r in results if r.test_total_bets > 0]
    if not valid_results:
        raise RuntimeError("All training agents failed")

    best = max(valid_results, key=lambda r: r.test_roi)

    logger.info(f"\n{'='*60}")
    logger.info(f"PARALLEL TRAINING COMPLETE ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")
    logger.info(f"Best agent: {best.config.name}")
    logger.info(f"  ROI: {best.test_roi:.1f}%")
    logger.info(f"  Win rate: {best.test_win_rate:.1%}")
    logger.info(f"  Final budget: {best.test_final_budget:,}")
    logger.info(f"  Model saved: {best.model_path}")
    logger.info(f"{'='*60}")

    return ParallelTrainingResult(
        best=best,
        all_results=results,
        total_elapsed=elapsed,
    )


def create_default_configs(seed: int = 42) -> list[TrainingConfig]:
    """Create a diverse set of training configs for parallel training.

    Returns configs with varied:
    - Learning rates
    - Hidden layer sizes
    - Exploration rates
    - Skip thresholds
    """
    rng = np.random.default_rng(seed)

    configs = []
    names_used = set()

    # Systematic grid
    for lr in [0.0005, 0.001, 0.002]:
        for hidden in [(64, 32), (128, 64), (64, 64, 32)]:
            for exploration in [0.2, 0.3, 0.4]:
                for skip in [0.10, 0.15, 0.20]:
                    name = f"lr{lr}_h{'x'.join(map(str,hidden))}_e{exploration}_s{skip}"
                    if name not in names_used:
                        configs.append(TrainingConfig(
                            name=name,
                            learning_rate=lr,
                            hidden_sizes=hidden,
                            exploration_rate=exploration,
                            skip_threshold=skip,
                            random_state=int(rng.integers(10000)),
                        ))
                        names_used.add(name)

    # Random configs
    for i in range(10):
        lr = float(rng.choice([0.0003, 0.0005, 0.001, 0.002, 0.003]))
        hidden = tuple(int(rng.choice([32, 64, 128])) for _ in range(int(rng.choice([2, 3]))))
        exploration = float(rng.choice([0.15, 0.2, 0.25, 0.3, 0.35, 0.4]))
        skip = float(rng.choice([0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]))

        name = f"random_{i:02d}"
        configs.append(TrainingConfig(
            name=name,
            learning_rate=lr,
            hidden_sizes=hidden,
            exploration_rate=exploration,
            skip_threshold=skip,
            random_state=int(rng.integers(10000)),
        ))

    return configs
