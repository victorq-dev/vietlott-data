"""CLI for Bingo18 ML training and simulation."""

from pathlib import Path

import click
import pandas as pd
import polars as pl
from loguru import logger

from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.report import render_report, save_report
from machine_learning.bingo18.simulator import Bingo18Simulator, BetType

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "bingo18.jsonl"
DEFAULT_MODEL_PATH = Path("bingo18_model.joblib")
DEFAULT_SAVE_DIR = Path("models/bingo18")

BET_TYPES = [bt.value for bt in BetType]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load Bingo18 data and convert to pandas."""
    logger.info(f"Loading data from {data_path}...")
    df = pl.read_ndjson(data_path)
    df_pd = df.to_pandas()
    logger.info(f"Loaded {len(df_pd)} draws")
    return df_pd


@click.command()
@click.option("--budget", type=int, required=True, help="Starting budget in VND")
@click.option("--bet-size", type=int, default=10_000, help="Bet per ticket in VND (default: 10000)")
@click.option(
    "--bet-type",
    type=click.Choice(BET_TYPES),
    default="mot_so",
    help="Bet type (default: mot_so)",
)
@click.option(
    "--strategy",
    type=click.Choice(["top_n", "threshold", "kelly"]),
    default="top_n",
    help="Betting strategy (default: top_n)",
)
@click.option("--top-n", type=int, default=1, help="Number of digits to consider (default: 1)")
@click.option("--threshold", type=float, default=0.12, help="Probability threshold (default: 0.12)")
@click.option("--target-total", type=int, default=None, help="Fixed total to bet on (for cong_tong)")
@click.option(
    "--target-category", type=click.Choice(["Nhỏ", "Hòa", "Lớn"]), default=None, help="Fixed category (for lon_hoa_nho)"
)
@click.option("--model-path", type=click.Path(), default=None, help="Path to saved model")
@click.option("--train-only", is_flag=True, help="Only train and save model, don't simulate")
@click.option("--data-path", type=click.Path(exists=True), default=str(DEFAULT_DATA_PATH), help="Path to bingo18.jsonl")
@click.option("--output", type=click.Path(), default=None, help="Output path for report")
@click.option("--window", type=int, default=30, help="Feature window size (default: 30)")
@click.option("--n-estimators", type=int, default=100, help="Number of estimators (default: 100)")
@click.option("--max-depth", type=int, default=3, help="Max depth (default: 3)")
@click.option(
    "--algorithm",
    type=click.Choice(["gradient_boosting", "random_forest", "extra_trees", "logistic_regression"]),
    default="gradient_boosting",
    help="ML algorithm (default: gradient_boosting)",
)
@click.option("--auto-tune", is_flag=True, help="Run auto-tuner to find best algorithm + strategy")
@click.option("--save-best", is_flag=True, help="Save best models from auto-tune (requires --auto-tune)")
@click.option(
    "--save-dir",
    type=click.Path(),
    default=str(DEFAULT_SAVE_DIR),
    help="Directory to save models (default: models/bingo18)",
)
@click.option("--top-k", type=int, default=10, help="Number of top results to show/save (default: 10)")
def bingo18_play(
    budget: int,
    bet_size: int,
    bet_type: str,
    strategy: str,
    top_n: int,
    threshold: float,
    target_total: int | None,
    target_category: str | None,
    model_path: str | None,
    train_only: bool,
    data_path: str,
    output: str | None,
    window: int,
    n_estimators: int,
    max_depth: int,
    algorithm: str,
    auto_tune: bool,
    save_best: bool,
    save_dir: str,
    top_k: int,
):
    """Train ML model and simulate auto-play on Bingo18.

    Examples:

        # Single simulation
        vietlott-bingo18-play --budget 10000000 --bet-type mot_so

        # Auto-tune to find best strategy
        vietlott-bingo18-play --budget 10000000 --auto-tune

        # Auto-tune and save best models
        vietlott-bingo18-play --budget 10000000 --auto-tune --save-best

        # Use saved model
        vietlott-bingo18-play --budget 10000000 --model-path models/bingo18/best.joblib
    """
    data_path = Path(data_path)
    df = load_data(data_path)

    if auto_tune:
        _run_auto_tune(df, bet_size, save_best, save_dir, top_k, output)
        return

    model = _load_or_train_model(model_path, algorithm, window, n_estimators, max_depth, df)

    if train_only:
        click.echo("Model trained and saved. Use without --train-only to run simulation.")
        return

    _run_simulation(
        model,
        budget,
        bet_size,
        bet_type,
        strategy,
        top_n,
        threshold,
        target_total,
        target_category,
        output,
        df=df,
    )


def _load_or_train_model(model_path, algorithm, window, n_estimators, max_depth, df):
    """Load existing model or train new one."""
    model = Bingo18Model(window=window, algorithm=algorithm, n_estimators=n_estimators, max_depth=max_depth)

    if model_path and Path(model_path).exists():
        logger.info(f"Loading existing model from {model_path}...")
        model.load(Path(model_path))
    else:
        logger.info(f"Training new model with {algorithm}...")
        metrics = model.train(df)
        logger.info(f"\n{metrics.summary()}")
        if model_path:
            model.save(Path(model_path))

    return model


def _run_simulation(
    model, budget, bet_size, bet_type, strategy, top_n, threshold, target_total, target_category, output, df=None
):
    """Run single simulation."""
    logger.info(
        f"Starting simulation: budget={budget:,}, bet_type={bet_type}, strategy={strategy}, bet_size={bet_size:,}"
    )

    simulator = Bingo18Simulator(
        model=model,
        budget=budget,
        bet_size=bet_size,
        bet_type=bet_type,
        strategy=strategy,
        top_n=top_n,
        threshold=threshold,
        target_total=target_total,
        target_category=target_category,
    )
    result = simulator.run(df)

    report = render_report(result)

    if output:
        save_report(result, Path(output))
        click.echo(f"Report saved to {output}")
    else:
        click.echo(report)


def _run_auto_tune(df, bet_size, save_best, save_dir, top_k, output):
    """Run auto-tuner."""
    from machine_learning.bingo18.auto_tuner import Bingo18AutoTuner, render_tuner_results

    logger.info("Starting auto-tuner...")
    tuner = Bingo18AutoTuner(bet_size=bet_size)

    if save_best:
        summary = tuner.run_and_save(df, save_dir=Path(save_dir), top_k=top_k)
    else:
        summary = tuner.run(df, top_k=top_k)

    report = render_tuner_results(summary)

    if output:
        Path(output).write_text(report)
        click.echo(f"Report saved to {output}")
    else:
        click.echo(report)
