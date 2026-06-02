"""CLI for Bingo18 ML training and simulation."""

import sys
from pathlib import Path

import click
import pandas as pd
import polars as pl
from loguru import logger

from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.report import render_report, save_report
from machine_learning.bingo18.simulator import BetType, Bingo18Simulator

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


@click.group()
def cli():
    """Bingo18 ML training, simulation, and racing tools."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="INFO")


@cli.command()
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
@click.option(
    "--mode",
    type=click.Choice(["single", "combine", "all_in", "skip"]),
    default="single",
    help="Betting mode: single (1 bet/draw), combine (multiple), all_in (best only), skip (low confidence)",
)
@click.option(
    "--bet-types",
    type=str,
    default=None,
    help="Comma-separated bet types for combined mode (e.g. 'mot_so,cong_tong_mult,trung_2so')",
)
@click.option("--confidence", type=float, default=0.0, help="Min confidence for skip mode (default: 0.0)")
def play(
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
    mode: str,
    bet_types: str | None,
    confidence: float,
):
    """Train ML model and simulate auto-play on Bingo18.

    Examples:

        # Single simulation
        vietlott-bingo18 play --budget 10000000 --bet-type mot_so

        # Combined mode: multiple bet types per draw
        vietlott-bingo18 play --budget 10000000 --mode combine --bet-types mot_so,cong_tong_mult,trung_2so

        # All-in mode: best bet only
        vietlott-bingo18 play --budget 10000000 --mode all_in --bet-types cong_tong_mult,lon_hoa_nho_v2

        # Skip mode: only bet when confident
        vietlott-bingo18 play --budget 10000000 --mode skip --bet-types mot_so --confidence 0.15

        # Auto-tune to find best strategy
        vietlott-bingo18 play --budget 10000000 --auto-tune

        # Use saved model
        vietlott-bingo18 play --budget 10000000 --model-path models/bingo18/best.joblib
    """
    data_path = Path(data_path)
    df = load_data(data_path)

    if auto_tune:
        combined_configs = None
        if bet_types and mode != "single":
            combined_configs = [{"bet_types": bet_types.split(","), "mode": mode, "confidence": confidence}]
        _run_auto_tune(df, bet_size, save_best, save_dir, top_k, output, combined_configs=combined_configs)
        return

    model = _load_or_train_model(model_path, algorithm, window, n_estimators, max_depth, df)

    if train_only:
        click.echo("Model trained and saved. Use without --train-only to run simulation.")
        return

    if mode != "single" and bet_types:
        _run_combined_simulation(
            model,
            budget,
            bet_size,
            bet_types.split(","),
            mode,
            confidence,
            output,
            df=df,
        )
    else:
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


def _run_combined_simulation(model, budget, bet_size, bet_types, mode, confidence, output, df=None):
    """Run combined simulation with multiple bet types."""
    logger.info(
        f"Starting combined simulation: budget={budget:,}, mode={mode}, bet_types={bet_types}, bet_size={bet_size:,}"
    )

    simulator = Bingo18Simulator(
        model=model,
        budget=budget,
        bet_size=bet_size,
    )
    result = simulator.run_combined(
        df,
        bet_types=bet_types,
        mode=mode,
        confidence_threshold=confidence,
    )

    report = render_report(result)

    if output:
        save_report(result, Path(output))
        click.echo(f"Report saved to {output}")
    else:
        click.echo(report)


def _run_auto_tune(df, bet_size, save_best, save_dir, top_k, output, combined_configs=None):
    """Run auto-tuner."""
    from machine_learning.bingo18.auto_tuner import Bingo18AutoTuner, render_tuner_results

    logger.info("Starting auto-tuner...")
    tuner = Bingo18AutoTuner(bet_size=bet_size)

    if combined_configs:
        logger.info(f"Combined auto-tune mode: {combined_configs}")
        summary = tuner.run_combined(df, combined_configs=combined_configs, top_k=top_k)
    elif save_best:
        summary = tuner.run_and_save(df, save_dir=Path(save_dir), top_k=top_k)
    else:
        summary = tuner.run(df, top_k=top_k)

    report = render_tuner_results(summary)

    if output:
        Path(output).write_text(report)
        click.echo(f"Report saved to {output}")
    else:
        click.echo(report)


@cli.command()
@click.option("--budget", type=int, default=500_000, help="Training budget in VND")
@click.option("--bet-size", type=int, default=10_000, help="Bet per ticket in VND")
@click.option("--epochs", type=int, default=10, help="Number of training epochs")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
@click.option("--output", type=click.Path(), default="models/strategy_model.pkl", help="Output model path")
def train_strategy(budget: int, bet_size: int, epochs: int, data_path: str | None, output: str):
    """Train a strategy model on historical data."""
    from machine_learning.bingo18.strategy_model import StrategyModel
    from machine_learning.bingo18.strategy_trainer import StrategyTrainer

    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    logger.info("Training Stage 1 model for feature extraction...")
    model = Bingo18Model()
    metrics = model.train(df)
    logger.info(f"Stage 1 model trained: {metrics.summary()}")

    logger.info(f"Training strategy model for {epochs} epochs...")
    strategy_model = StrategyModel(
        context_dim=model.feature_engineer.n_features + 6 + 3 + 10 + 1,
    )
    trainer = StrategyTrainer(
        model=model,
        strategy_model=strategy_model,
        budget=budget,
        bet_size=bet_size,
    )
    results = trainer.train(df, n_epochs=epochs)

    # Print results
    test = results["test"]
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  STRATEGY MODEL TRAINING RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Train experiences: {results['n_train_experiences']:,}")
    click.echo(f"  Test win rate:     {test['win_rate']:.1%}")
    click.echo(f"  Test ROI:          {test['roi']:.1f}%")
    click.echo(f"  Test final budget: {test['final_budget']:,.0f} VND")
    click.echo(f"  Test total bets:   {test['total_bets']:,}")
    click.echo(f"{'=' * 60}")

    # Save model
    output_path = Path(output)
    strategy_model.save(output_path)
    click.echo(f"\nModel saved to {output_path}")


@cli.command()
@click.option("--budget", type=int, default=10_000_000, help="Starting budget per agent in VND")
@click.option("--bet-size", type=int, default=10_000, help="Bet per ticket in VND")
@click.option("--n-agents", type=int, default=12, help="Number of agents in the race")
@click.option("--adaptation-interval", type=int, default=50, help="Draws between adaptations")
@click.option("--share-knowledge", is_flag=True, help="Enable knowledge sharing between agents")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
@click.option("--output", type=click.Path(), default=None, help="Output dir (with --visualize) or report file path")
@click.option("--strategy-model", type=click.Path(exists=True), default=None, help="Path to trained strategy model")
@click.option("--top-k", type=int, default=10, help="Number of top agents to show")
@click.option("--visualize", is_flag=True, help="Generate visualization charts")
@click.option("--verbose", "-v", is_flag=True, help="Show per-agent bet details (DEBUG logs)")
def race(
    budget: int,
    bet_size: int,
    n_agents: int,
    adaptation_interval: int,
    share_knowledge: bool,
    data_path: str | None,
    output: str | None,
    strategy_model: str | None,
    top_k: int,
    visualize: bool,
    verbose: bool,
):
    """Run adaptive multi-agent race on Bingo18 data.

    Creates diverse agents with different strategies and runs them
    through historical data to find the best performing approach.

    Examples:

        # Default race with 12 agents
        vietlott-bingo18 race

        # Custom budget and agent count
        vietlott-bingo18 race --budget 20000000 --n-agents 20

        # Enable knowledge sharing and visualization
        vietlott-bingo18 race --share-knowledge --visualize --output results/

        # Both --output and --visualize: charts in results/charts/, report in results/report.txt
        vietlott-bingo18 race --visualize --output results/
    """
    from machine_learning.bingo18.agent import create_diverse_agents
    from machine_learning.bingo18.race import RaceCoordinator

    # Enable DEBUG logging if verbose
    if verbose:
        logger.remove()
        logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="DEBUG")

    # Load data
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    # Train model
    logger.info("Training model for agent population...")
    model = Bingo18Model()
    metrics = model.train(df)
    logger.info(f"Model trained: {metrics.summary()}")

    # Load strategy model if provided
    loaded_strategy_model = None
    if strategy_model:
        from machine_learning.bingo18.strategy_model import StrategyModel

        loaded_strategy_model = StrategyModel(
            context_dim=len(model.feature_engineer._feature_names()) + 6 + 3 + 10 + 1,
        )
        loaded_strategy_model.load(Path(strategy_model))
        logger.info(f"Strategy model loaded from {strategy_model}")

    # Create diverse agents
    logger.info(f"Creating {n_agents} diverse agents (budget={budget:,}, bet_size={bet_size:,})...")
    agents = create_diverse_agents(
        model=model,
        budget=budget,
        bet_size=bet_size,
        n_agents=n_agents,
        adaptation_interval=adaptation_interval,
    )

    # Inject strategy model into first half of agents (if loaded)
    if loaded_strategy_model is not None:
        for i, agent in enumerate(agents):
            if i < len(agents) // 2:
                agent._strategy_model = loaded_strategy_model
                logger.info(f"Agent {agent.agent_id} using strategy model")

    # Run race
    logger.info(
        f"Starting race: {n_agents} agents, adaptation_interval={adaptation_interval}, "
        f"share_knowledge={share_knowledge}"
    )
    coordinator = RaceCoordinator(
        agents=agents,
        adaptation_interval=adaptation_interval,
        share_knowledge=share_knowledge,
    )
    result = coordinator.run_race(df)

    # Print leaderboard
    _print_leaderboard(result, top_k)

    # Print winner
    if result.winner:
        w = result.winner
        click.echo(
            f"\n  WINNER: {w.agent_id}\n"
            f"    Risk Profile : {w.genome.risk_profile}\n"
            f"    Strategy     : {w.genome.primary_strategy}\n"
            f"    ROI          : {w.roi:+.2f}%\n"
            f"    Final Budget : {w.final_budget:,} VND\n"
            f"    Win Rate     : {w.win_rate:.1%}\n"
            f"    Adaptations  : {w.adaptation_count}"
        )

    # Print race summary
    click.echo(
        f"\n  Race completed in {result.total_elapsed_seconds:.1f}s | "
        f"{result.completed_agents}/{result.total_agents} agents finished | "
        f"{result.total_draws} draws processed"
    )

    # Generate visualizations
    if visualize:
        if output:
            viz_dir = Path(output) / "charts"
        else:
            viz_dir = Path("race_output")
        _generate_visualizations(result, viz_dir)

    # Save report to file
    if output:
        if visualize:
            output_path = Path(output) / "report.txt"
        else:
            output_path = Path(output)
            if not output_path.suffix:
                output_path = output_path / "report.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_text = _format_report_text(result, top_k)
        output_path.write_text(report_text, encoding="utf-8")
        click.echo(f"\nReport saved to {output_path}")


def _print_leaderboard(result, top_k: int) -> None:
    """Print formatted leaderboard table."""
    agents = result.agent_results[:top_k]
    if not agents:
        click.echo("No agent results to display.")
        return

    click.echo(f"\n{'=' * 110}")
    click.echo(f"  BINGO18 AGENT RACE LEADERBOARD (Top {min(top_k, len(agents))} of {len(result.agent_results)})")
    click.echo(f"{'=' * 110}")
    click.echo(
        f"  {'Rank':<5} {'Agent ID':<12} {'Risk':<13} {'Strategy':<10} "
        f"{'Final Budget':>15} {'ROI':>10} {'Win Rate':>10} {'Drawdown':>12} {'Adapt':>6}"
    )
    click.echo(f"  {'-' * 105}")

    for i, agent in enumerate(agents, 1):
        roi_str = f"{agent.roi:+.2f}%"
        wr_str = f"{agent.win_rate:.1%}"
        budget_str = f"{agent.final_budget:,}"
        dd_str = f"{agent.max_drawdown:,}"
        click.echo(
            f"  {i:<5} {agent.agent_id:<12} {agent.genome.risk_profile:<13} "
            f"{agent.genome.primary_strategy:<10} {budget_str:>15} "
            f"{roi_str:>10} {wr_str:>10} {dd_str:>12} {agent.adaptation_count:>6}"
        )

    click.echo(f"{'=' * 110}")


def _generate_visualizations(result, output_dir: Path) -> None:
    """Generate and save visualization charts."""
    from machine_learning.bingo18.visualize import generate_race_report

    logger.info(f"Generating visualizations to {output_dir}...")

    # Prepare data for visualization
    agent_dicts = []
    for ar in result.agent_results:
        agent_dicts.append(
            {
                "agent_id": ar.agent_id,
                "roi": ar.roi,
                "final_budget": ar.final_budget,
                "win_rate": ar.win_rate,
                "max_drawdown": ar.max_drawdown,
                "total_bets": ar.total_bets,
            }
        )

    profit_curves = {ar.agent_id: ar.profit_curve for ar in result.agent_results}

    # Combine bet history from all agents
    all_bet_history = []
    for ar in result.agent_results:
        all_bet_history.extend(ar.bet_history)

    generate_race_report(
        agent_results=agent_dicts,
        output_dir=output_dir,
        starting_budget=result.budget,
        profit_curves=profit_curves,
        bet_history=all_bet_history,
    )
    click.echo(f"\nVisualizations saved to {output_dir}/")


def _format_report_text(result, top_k: int) -> str:
    """Format race result as text report."""
    lines = [
        "Bingo18 Multi-Agent Race Report",
        "=" * 50,
        f"Budget: {result.budget:,} VND per agent",
        f"Bet Size: {result.bet_size:,} VND",
        f"Total Draws: {result.total_draws}",
        f"Adaptation Interval: {result.adaptation_interval}",
        f"Agents: {result.completed_agents}/{result.total_agents}",
        f"Elapsed: {result.total_elapsed_seconds:.1f}s",
        "",
        "Leaderboard:",
        "-" * 100,
        f"{'Rank':<5} {'Agent':<12} {'Risk':<13} {'Strategy':<10} "
        f"{'Budget':>15} {'ROI':>10} {'WinRate':>10} {'Drawdown':>12} {'Adapt':>6}",
        "-" * 100,
    ]

    for i, agent in enumerate(result.agent_results[:top_k], 1):
        lines.append(
            f"{i:<5} {agent.agent_id:<12} {agent.genome.risk_profile:<13} "
            f"{agent.genome.primary_strategy:<10} {agent.final_budget:>15,} "
            f"{agent.roi:>+10.2f}% {agent.win_rate:>10.1%} "
            f"{agent.max_drawdown:>12,} {agent.adaptation_count:>6}"
        )

    if result.winner:
        w = result.winner
        lines.extend(
            [
                "",
                "Winner:",
                f"  Agent: {w.agent_id}",
                f"  Risk Profile: {w.genome.risk_profile}",
                f"  Strategy: {w.genome.primary_strategy}",
                f"  ROI: {w.roi:+.2f}%",
                f"  Final Budget: {w.final_budget:,} VND",
                f"  Win Rate: {w.win_rate:.1%}",
                f"  Adaptations: {w.adaptation_count}",
            ]
        )

    return "\n".join(lines)


@cli.command()
@click.option("--epochs", type=int, default=10, help="Number of training epochs")
@click.option("--budget", type=int, default=500_000, help="Starting budget per agent in VND")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
@click.option("--output", type=click.Path(), default=None, help="Path to save trained strategy model")
def train_strategy(epochs: int, budget: int, data_path: str | None, output: str | None):
    """Train a strategy model via simulation on historical data.

    The strategy model learns WHEN to use WHICH bet type in different
    situations, optimizing for net profit/ROI.

    Examples:

        # Train with default settings
        vietlott-bingo18 train-strategy

        # Train for 20 epochs with custom budget
        vietlott-bingo18 train-strategy --epochs 20 --budget 1000000

        # Save trained model
        vietlott-bingo18 train-strategy --output models/strategy_v1.joblib
    """
    from machine_learning.bingo18.strategy_model import N_BET_TYPES, StrategyModel
    from machine_learning.bingo18.strategy_trainer import StrategyTrainer

    # Load data
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    # Train Stage 1 model
    logger.info("Training Stage 1 model (digit prediction)...")
    model = Bingo18Model()
    metrics = model.train(df)
    logger.info(f"Stage 1 model trained: {metrics.summary()}")

    # Create strategy model
    context_dim = len(model.feature_engineer._feature_names()) + 6 + 3 + N_BET_TYPES + 1
    strategy_model = StrategyModel(context_dim=context_dim)

    # Train strategy model
    logger.info(f"Training strategy model for {epochs} epochs...")
    trainer = StrategyTrainer(
        model=model,
        strategy_model=strategy_model,
        budget=budget,
    )
    results = trainer.train(df, n_epochs=epochs)

    # Print results
    test = results["test"]
    click.echo(f"\n{'=' * 60}")
    click.echo("  STRATEGY MODEL TRAINING RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Train experiences: {results['n_train_experiences']:,}")
    click.echo(f"  Test win rate:     {test['win_rate']:.1%}")
    click.echo(f"  Test ROI:          {test['roi']:+.2f}%")
    click.echo(f"  Test final budget: {test['final_budget']:,.0f} VND")
    click.echo(f"  Test total bets:   {test['total_bets']:,}")
    click.echo(f"{'=' * 60}")

    # Save model
    if output:
        output_path = Path(output)
    else:
        output_path = Path("models/strategy_model.joblib")
    strategy_model.save(output_path)
    click.echo(f"\nStrategy model saved to {output_path}")


@cli.command()
@click.option("--budget", type=int, default=10_000_000, help="Starting budget per agent in VND")
@click.option("--bet-size", type=int, default=10_000, help="Bet per ticket in VND")
@click.option("--n-agents", type=int, default=6, help="Number of agents in the race")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
@click.option("--strategy-model", type=click.Path(exists=True), default=None, help="Path to trained strategy model")
def race_with_strategy(budget: int, bet_size: int, n_agents: int, data_path: str | None, strategy_model: str | None):
    """Race strategy agents against heuristic agents.

    Creates a mix of strategy-learning agents and heuristic agents,
    then runs them through historical data to compare performance.

    Examples:

        # Race with default strategy model
        vietlott-bingo18 race-with-strategy

        # Race with custom strategy model
        vietlott-bingo18 race-with-strategy --strategy-model models/strategy_v1.joblib
    """
    from machine_learning.bingo18.agent import create_diverse_agents
    from machine_learning.bingo18.race import RaceCoordinator
    from machine_learning.bingo18.strategy_model import N_BET_TYPES, StrategyModel

    # Load data
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    # Train Stage 1 model
    logger.info("Training Stage 1 model...")
    model = Bingo18Model()
    model.train(df)

    # Load or create strategy model
    if strategy_model:
        sm = StrategyModel()
        sm.load(Path(strategy_model))
        logger.info(f"Loaded strategy model from {strategy_model}")
    else:
        # Create untrained strategy model (will use uniform distribution)
        context_dim = len(model.feature_engineer._feature_names()) + 6 + 3 + N_BET_TYPES + 1
        sm = StrategyModel(context_dim=context_dim)
        logger.info("No strategy model provided, using untrained model")

    # Create agents: half with strategy model, half heuristic
    n_strategy = n_agents // 2
    n_heuristic = n_agents - n_strategy

    from machine_learning.bingo18.agent import AdaptiveAgent, AgentGenome

    agents: list[AdaptiveAgent] = []
    for i in range(n_strategy):
        genome = AgentGenome(max_bets_per_draw=3, multi_bet_budget_share=0.06)
        agent = AdaptiveAgent(
            agent_id=f"strategy_{i:03d}",
            genome=genome,
            model=model,
            budget=budget,
            bet_size=bet_size,
            strategy_model=sm,
        )
        agents.append(agent)

    heuristic_agents = create_diverse_agents(
        model=model,
        budget=budget,
        bet_size=bet_size,
        n_agents=n_heuristic,
    )
    agents.extend(heuristic_agents)

    # Run race
    logger.info(f"Starting race: {n_strategy} strategy + {n_heuristic} heuristic agents")
    coordinator = RaceCoordinator(
        agents=agents,
        adaptation_interval=50,
    )
    result = coordinator.run_race(df)

    # Print results
    click.echo(f"\n{'=' * 80}")
    click.echo("  RACE RESULTS: Strategy vs Heuristic")
    click.echo(f"{'=' * 80}")

    all_agents = sorted(result.agents, key=lambda a: a.roi, reverse=True)
    for rank, agent in enumerate(all_agents, 1):
        prefix = "[S]" if agent.agent_id.startswith("strategy") else "[H]"
        roi_str = f"{agent.roi:+.2f}%"
        wr_str = f"{agent.win_rate:.1%}"
        click.echo(
            f"  {rank:>3}. {prefix} {agent.agent_id:<15} "
            f"ROI={roi_str:>12} WinRate={wr_str:>8} "
            f"Budget={agent.final_budget:>15,}"
        )

    click.echo(f"{'=' * 80}")


@cli.command()
@click.option("--n-agents", type=int, default=5, help="Number of parallel training agents")
@click.option("--epochs", type=int, default=10, help="Training epochs per agent")
@click.option("--budget", type=int, default=500_000, help="Starting budget per agent")
@click.option("--output", type=click.Path(), default="models/bingo18/parallel", help="Output directory for parallel training models")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
def train_parallel(n_agents: int, epochs: int, budget: int, output: str, data_path: str | None):
    """Train multiple strategy models in parallel and select the best.

    Runs N training agents with different hyperparameters, monitors their
    performance, and saves the best model.

    Examples:

        # Train 5 agents in parallel
        vietlott-bingo18 train-parallel

        # Train 10 agents for 20 epochs each
        vietlott-bingo18 train-parallel --n-agents 10 --epochs 20
    """
    from machine_learning.bingo18.parallel_trainer import create_default_configs, run_parallel_training

    # Load data
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    # Train Stage 1 model
    logger.info("Training Stage 1 model (digit prediction)...")
    model = Bingo18Model()
    metrics = model.train(df)
    logger.info(f"Stage 1 model trained: {metrics.summary()}")

    # Create configs
    configs = create_default_configs()[:n_agents]
    for config in configs:
        object.__setattr__(config, 'n_epochs', epochs)
        object.__setattr__(config, 'budget', budget)

    # Run parallel training
    result = run_parallel_training(
        df=df,
        model=model,
        configs=configs,
        output_dir=output,
    )

    # Print results table
    click.echo(f"\n{'=' * 80}")
    click.echo("  PARALLEL TRAINING RESULTS")
    click.echo(f"{'=' * 80}")
    click.echo(f"  {'Agent':<30} {'ROI':>10} {'WinRate':>10} {'Budget':>15} {'Bets':>8} {'Time':>8}")
    click.echo(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*15} {'-'*8} {'-'*8}")

    for r in sorted(result.all_results, key=lambda x: x.test_roi, reverse=True):
        click.echo(
            f"  {r.config.name:<30} {r.test_roi:>9.1f}% {r.test_win_rate:>9.1%} "
            f"{r.test_final_budget:>15,} {r.test_total_bets:>8} {r.elapsed_seconds:>7.1f}s"
        )

    click.echo(f"\n  BEST: {result.best.config.name}")
    click.echo(f"  Model saved: {result.best.model_path}")
    click.echo(f"  Total time: {result.total_elapsed:.1f}s")
    click.echo(f"{'=' * 80}")


if __name__ == "__main__":
    cli()
